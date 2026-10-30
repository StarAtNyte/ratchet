# smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128 (matrix-atom assembly for 16-bit exchange faces: SW128 face → transposed LDSM read into a fragment isomorphic to `Ld16x256b`; warp-level MMA C fragment → STSM → UMMA B operand face)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry covers the **other two stations** of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md):
> that entry teaches the "TMEM accumulator → `stmatrix` → staging face → TMA store" station (output side). The same family of
> 8×8×16b matrix atoms has two more stations that on sm100 often degrade into per-element scalar `LDS.U16` / `STS.U16`:
>
> - **Station A**: a K-major SW128 face landed by TMA (R rows × C columns, C contiguous) has to **meet a TMEM accumulator
>   element by element** (subtract, multiply by a coefficient, serve as the A operand of the next MMA), while the accumulator's
>   lane is C and its column is R -- the face must be read **transposed** into a register tensor **isomorphic** to the
>   `tcgen05.ld.16x256b` fragment shape.
> - **Station B**: the C fragment of a warp-level `mma.sync` (4 warps tile 64×64), narrowed to 16-bit, is written into a
>   K-major SW128 face that **UMMA reads as the B operand through a descriptor**.
>
> Both stations share a single criterion: **the fragment shape decides the atom**. The `Ld32x32b` fragment shape (thread = lane,
> 32 contiguous 32-bit columns per thread) has no matrix-atom partner -- each (thread, column) holds only one 16-bit element,
> so a transposed write to the face has only the scalar `STS.U16` path left; the `Ld16x256b` fragment shape is exactly the
> `mma.m16n8` C fragment shape (each thread holds two rows 8 apart, two adjacent columns, one group per 8 columns), and it is
> bit-for-bit isomorphic to the fragment shape that `ldmatrix.x4.trans` / `stmatrix.x4` scatter. The public helper hard-codes
> this criterion: `get_smem_store_op` returns `StMatrix8x8x16bOp` only when the t2r is 16-datapath / 256-bit and the
> destination is 16-bit; 32-datapath always falls back to `CopyUniversalOp` (plain `st.shared`). So **any accumulator that
> has to meet an smem face through a matrix atom must be read back with `Ld16x256b`**; keep `Ld32x32b` for purely per-lane
> paths (state scaling, packing back into TMEM).

## Purpose

| Station | Hardware fact | Corresponding form in the DSL |
|---|---|---|
| A: SW128 face → fragment isomorphic to `Ld16x256b` (transposed) | each addressing lane of `ldmatrix` delivers 8 contiguous 16-bit values (16 B); the `.trans` member hands matrix **columns** to the fragment's row lane groups. When the contiguous dimension of the face is m (= TMEM lane), use the transposed member; the resulting thread-value map = the `tcgen05.ld.16x256b` map | the (C, R) view of the face comes from `cute.select(face.layout, [1, 0])` (a pure relabel; the swizzle stays on the pointer); the tiled copy is **derived from the TMEM copy**: `cute.make_tiled_copy_D(make_copy_atom(LdMatrix8x8x16bOp(True, 4), bf16), tiled_copy_t2r)`; the two sides are `thr.partition_S(face_view)` / `tiled.retile(rV)` (`retile` lives on the TiledCopy), and `rV` has the same shape as the t2r register tensor (`partition_D(tmem_tile).shape`), so `rV[i]` and `rAcc[i]` refer to the same (m, n) |
| B: warp-level MMA C fragment → SW128 B face | `stmatrix` is the inverse of `ldmatrix`; the non-transposed member writes register rows as face rows, the transposed member writes them as face columns; the UMMA descriptor reads "the face's own layout + the swizzle on the pointer", the same one the STSM write uses | the face comes from `make_smem_layout_b(mma, tiler, bf16, 1)` and is split at allocation into `layout=L.outer, swizzle=L.inner`; `tcC = cute.make_tiled_copy_C(make_copy_atom(StMatrix8x8x16bOp(transpose, n), bf16), hmma)`; `cute.copy(tcC, tcC.retile(c16), thr.partition_D(flat_divide(face_view, (16, tile_n))[(None, None, w, j)]))`; face view = `make_tensor(sB.iterator, make_layout((N, K), stride=(K, 1)))` (the face the helper produces has four modes ((N, K_mma), 1, MMA_K, STAGE) and cannot be sliced as (N, K, STAGE)); after writing, `fence_proxy("async.shared", space="cta")` + barrier, then `make_fragment_B(face)` feeds `cute.gemm` |
| C: accumulator → staging face → TMA store | see [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md): the M-major (`LayoutEnum.COL_MAJOR`) row = `Ld16x256b` + `StMatrix8x8x16b(transpose)` + `S<3,4,3>` face | not repeated; this entry only adds stations A and B |

The cost of not using matrix atoms (numerically all correct, invisible to the correctness gate): station A spends one
`LDS.U16` plus address arithmetic per element, station B one `STS.U16` per element; sub-word accesses where several lanes in
one wavefront land on different half-words of the same bank are **one** source of bank conflicts; the fragment's live range is
stretched by the gather loop → register cliff. **But `bankconf_st` / `bankconf_ld` are not owned exclusively by these three
stations**: after the first landing replaced all three stations with matrix atoms, `bankconf_st` dropped only 16% and
`bankconf_ld` actually rose 27% (readings and sources under "Certification boundary"). What this entry can guarantee is that
each station's instruction family was replaced (`STS.U16` × element count → `STSM` × element count/8), not that the NCU
bank-conflict axis returns to normal as a result.

## When needed (trigger)

- The dossier / ledger says `LDSM.16.MT88.4` (transposed read), `STSM.16.M88.4` / `STSM.16.MT88.4` (write), but the
  implementation uses per-element indexing like `face[(t, tidx)] = …` / `x = face[(t, tidx)]` -- even if a comment says
  "one wavefront, conflict-free";
- NCU: `bankconf_st` or `bankconf_ld` is **2–12×** the reference's magnitude while `smem_wavefronts` is only 1.0–1.3× higher;
  the loop body's `STS.U16` / `LDS.U16` count in SASS ≈ elements per thread;
- you chose `Ld32x32b` (thread = lane) for every TMEM readback and then found that "transposed write / transposed read can only
  be done per element" -- that is not a problem with the smem face, the fragment shape is wrong (gotcha 1);
- at build time: `make_tiled_copy_D(LdMatrix…, t2r_32x32b)` is rejected / `retile` reports a shape mismatch -- same gotcha.

## Certification boundary [UNVERIFIED-BOUNDARY]

This entry certifies the **bijection and the instruction family** (the contracts of the three probes under "Sources": the
fragment read by the transposed LDSM has the same (m, n) at the same index as the TMEM fragment; the face written by STSM, read
by the UMMA descriptor, yields `A · Rᵀ` / `A · R` bit-for-bit; `get_smem_store_op` returns the `StMatrix` family for a
`Ld16x256b` fragment). **"Bank conflicts return to the matrix-path magnitude" is not certified, and the first-landing readings
contradict it.** The same production tree (12 warp-specialized, state resident in TMEM, grid 148, `warp_active_pct` ≈ 3.6)
first landed station B (two 64×64 B operand faces: 32 `STS.U16` per thread → 4 `STSM.16.M88.4`), then stations A + C (V face
64 `LDS.U16` → 8 `LDSM.16.MT88.4`, O staging 64 `STS.U16` → `STSM.16.MT88.4`, TMEM readback `Ld32x32b:x32` → `Ld16x256b:x8`,
A face packing `St32x32b` × 2 → `St16x128b:x8`); both KEEPs passed numerically, same-job ratio 0.307 → 0.375 → 0.389 (ms was
bundled with other accompanying changes; it is not the price of the three stations alone):

| Reading | Before landing | Station B landed | Stations A + C landed |
|---|---|---|---|
| `bankconf_st` | 1,468,716 | 1,379,211 (−6%) | 1,229,880 (−16%) |
| `bankconf_ld` | 862,197 | 1,068,068 (**+24%**) | 1,096,613 (**+27%**) |
| `merge.scalar_sts_in_loop` | 238 | 174 | 110 |
| `tensor_pct` | 1.62 | 2.00 | 2.08 |
| `stall_mio` | 0.34 | 0.40 | 0.47 |
| `regs_per_thread` | 159 | 165 | 165 |

That is: after the three stations were switched to matrix atoms, the scalar STS in the loop body fell by more than half and
the tensor share went up, but `bankconf_st` is still 84% of its pre-landing value and `bankconf_ld` rose -- the dominant bank
conflict term of this tree is **not at these three stations** (the index reads that gather the C fragment from the padded work
buffer ahead of station B, and the per-field scalar reads of the gating array, are all in the same NCU count). Before adopting,
check the instruction family with gotcha 10's **per-station** machine-readable criterion; attribute `bankconf_*` separately and
do not treat it as this entry's acceptance bar. This library has no census reading yet for `STSM` / `LDSM` counts in SASS (the
landing record only has trace-time prints of the two sides' layouts); they are marked as expected in the "Sources" table.

## Minimal runnable example

Both snippets below were actually run by GPU-certified probes (verdicts under "Sources"). The face holds asymmetric small
integers (`((r + 3c) % 7) − 3`), so "transposed wrong" cannot accidentally equal "right"; each station is **bit-for-bit
equal** to the numpy reference.

### Station A: K-major SW128 face → fragment isomorphic to `Ld16x256b` (transposed LDSM), per element `acc[i] − face[i]`

The TMEM tile is (128 lanes × 64 columns) f32; the smem face is (64 rows × 128 columns) bf16, columns contiguous (the box shape
TMA lands: two SW128 boxes of 64 columns × 128 B stacked). The consumer wants `out[m, n] = acc[m, n] − face[n, m]`.

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05, warp
BF16, F32 = cutlass.BFloat16, cutlass.Float32
LANES, COLS = 128, 64                      # TMEM tile (m, n)
R_FACE, C_FACE = COLS, LANES               # smem face (n, m), m contiguous

@cute.jit
def host(gA, gV, gOut):
    op = tcgen05.MmaF16BF16Op(BF16, F32, (LANES, COLS, 16), tcgen05.CtaGroup.ONE,
                              tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K,
                              tcgen05.OperandMajorMode.K)
    mma = cute.make_tiled_mma(op)                                   # only borrow its accumulator layout
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    v_layout = cute.coalesce(cute.tile_to_shape(atom, (R_FACE, C_FACE, 1), order=(0, 1, 2)),
                             target_profile=(1, 1, 1))              # same discipline as a TMA face
    kernel(gA, gV, gOut, mma, v_layout).launch(grid=[1, 1, 1], block=[128, 1, 1])

@cute.kernel
def kernel(gA, gV, gOut, mma, v_layout: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator(); storage = smem.allocate(SharedStorage)   # tmem_holding_buf: Int32
    sV = smem.allocate_tensor(element_type=BF16, layout=v_layout.outer,
                              byte_alignment=1024, swizzle=v_layout.inner)
    sV0 = sV[(None, None, 0)]                                       # (R, C), C contiguous; filled by TMA or indexed writes
    …                                                               # fill the face; fence_view_async_shared + sync
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                               barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=128))
    tmem.allocate(512); tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(F32); tmem.relinquish_alloc_permit()
    acc = cute.make_tensor(tmem_ptr, mma.make_fragment_C(mma.partition_shape_C((LANES, COLS))).layout)
    plane = acc[((None, None), 0, 0)]                               # (LANES, COLS) TMEM tile
    idC = cute.make_identity_tensor((LANES, COLS))

    # ① TMEM → registers: 16x256b x8 (one per 16 lanes × 64 columns; 4 warps with 32 lanes each = two M iterations)
    ld = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), F32), plane)
    thr_ld = ld.get_slice(tidx)
    rAcc = cute.make_rmem_tensor(thr_ld.partition_D(plane).shape, F32)      # register side shaped by the D-side partition
    cAcc = thr_ld.partition_D(idC)                                          # absolute (m, n) of each element
    cute.copy(ld, thr_ld.partition_S(plane), rAcc)
    cute.arch.fence_view_async_tmem_load()

    # ② face → registers, same bijection: mode 0 (= m = lane) of the (C, R) view is contiguous → transposed member;
    #    the tiled copy is derived from the register side of the TMEM copy; retile on the register side, partition_S on the face side
    sVt = cute.make_tensor(sV0.iterator, cute.select(sV0.layout, [1, 0]))   # pure relabel, swizzle on the pointer
    rV = cute.make_rmem_tensor(rAcc.shape, BF16)
    ldsm = cute.make_tiled_copy_D(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BF16), ld)
    thr_l = ldsm.get_slice(tidx)
    cute.copy(ldsm, thr_l.partition_S(sVt), ldsm.retile(rV))               # 8 LDSM.16.MT88.4 per thread; retile is on the TiledCopy

    # ③ element-wise: the same index is the same (m, n)
    for i in cutlass.range_constexpr(cute.size(rAcc)):
        m, n = cAcc[i][0], cAcc[i][1]
        gOut[(m, n)] = rAcc[i] - F32(rV[i])
```

To use the result as the A operand of the next UMMA (resident in TMEM), continue with
[`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md)
+ [`tmem_r2t_recast_fragment`](tmem_r2t_recast_fragment.md): the `St16x128b` / `St16x256b` fragment shapes are in the same
family as `Ld16x256b`, and the packed 32-bit slot order is the A face column order. To write the result transposed into a
staging face for a TMA store, continue with station C.

### Station B: warp-level MMA C fragment → bf16 → `stmatrix` → UMMA B operand face (K-major SW128) → descriptor read

4 warps each hold a (16 × 64) m16n8 C fragment, together a 64×64 register tile `R`; UMMA 128×64×16, K = 64 (4 k-blocks),
A face `make_smem_layout_a`, B face `make_smem_layout_b`. **The member is decided by "is a row of the register tile the face's
N or its K"**:

| r in register tile `R[r, c]` is | View passed to partition_D | Member | Lands on the face as | UMMA computes |
|---|---|---|---|---|
| B's **N** (row = output column) | the `(N, K)` face itself (K contiguous) | `StMatrix8x8x16bOp(False, n)` → `STSM.16.M88.n` | `face[n, k] = R[n, k]` | `D = A · Rᵀ` |
| B's **K** (row = contraction dim) | the `(K, N)` transposed view (`select(face.layout, [1, 0])`, K contiguous) | `StMatrix8x8x16bOp(True, n)` → `STSM.16.MT88.n` | `face[n, k] = R[k, n]` | `D = A · R` |

```python
@cute.jit
def host(gA, gR, gC, trans: cutlass.Constexpr[bool]):
    op = tcgen05.MmaF16BF16Op(BF16, F32, (128, 64, 16), tcgen05.CtaGroup.ONE,
                              tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K,
                              tcgen05.OperandMajorMode.K)
    mma = cute.make_tiled_mma(op)
    a_layout = sm100_utils.make_smem_layout_a(mma, (128, 64, 64), BF16, 1)
    b_layout = sm100_utils.make_smem_layout_b(mma, (128, 64, 64), BF16, 1)   # (N, K, STAGE), K-major SW128
    hmma = cute.make_tiled_mma(warp.MmaF16BF16Op(BF16, F32, (16, 8, 16)), (1, 1, 1), (16, 16, 16))
    tile_n = hmma.get_tile_size(1)                                          # 16 → x4; without permutation 8 → x2
    tcC = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(trans, (16 * tile_n) // 64), BF16), hmma)
    kernel(gA, gR, gC, mma, hmma, tcC, a_layout, b_layout, tile_n, trans).launch(grid=[1, 1, 1], block=[128, 1, 1])

@cute.kernel
def kernel(gA, gR, gC, mma, hmma, tcC, a_layout, b_layout, tile_n: cutlass.Constexpr[int], trans: cutlass.Constexpr[bool]):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()); lane = cute.arch.lane_idx()
    sA = smem.allocate_tensor(element_type=BF16, layout=a_layout.outer, byte_alignment=1024, swizzle=a_layout.inner)
    sB = smem.allocate_tensor(element_type=BF16, layout=b_layout.outer, byte_alignment=1024, swizzle=b_layout.inner)
    …                                                          # mbarrier(1) init, TMEM allocation; the A face is filled through the same flat (M, K) view
    # the helper's face is ((N, K_mma), 1, MMA_K, STAGE): rank 4 with a nested first mode, so `sB[(None, None, 0)]` is rejected
    # (coord and shape are not weakly congruent). When a row is exactly 128 B (one SW128 atom wide), its affine part is
    # row-major (N, K):(K, 1) with the swizzle on the pointer, so the flat (N, K) view is a relabel:
    sB0 = cute.make_tensor(sB.iterator, cute.make_layout((64, 64), stride=(64, 1)))   # (N, K)
    if cutlass.const_expr(trans):
        face = cute.make_tensor(sB0.iterator, cute.select(sB0.layout, [1, 0]))   # (K, N) view
    else:
        face = sB0
    f2 = cute.flat_divide(face, (16, tile_n))                  # (16, tile_n, rows/16, cols/tile_n)
    thr_h = hmma.get_slice(lane)                               # warp-level MMA sliced by lane
    idC = thr_h.partition_C(cute.make_identity_tensor((16, tile_n)))
    c16 = cute.make_rmem_tensor(idC.shape, BF16)               # 16-bit version of the C fragment shape (partition_shape_C is a TiledMma method, not on ThrMma)
    thr_c = tcC.get_slice(lane)
    for j in cutlass.range_constexpr(64 // tile_n):
        for i in cutlass.range_constexpr(cute.size(c16)):
            r, c = idC[i][0], idC[i][1]
            c16[i] = BF16(gR[(warp_idx * 16 + r, j * tile_n + c)])    # in real use this is the mma.sync acc
        cute.copy(tcC, tcC.retile(c16), thr_c.partition_D(f2[(None, None, warp_idx, j)]))    # STSM; retile is on the TiledCopy
    cute.arch.fence_proxy("async.shared", space="cta")          # generic write → async proxy (descriptor) read
    cute.arch.sync_threads()
    frA = mma.make_fragment_A(sA); frB = mma.make_fragment_B(sB)          # B is defined by the face's own layout + pointer swizzle
    if warp_idx == 0:
        mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(cute.size(frA, mode=[2])):
            cute.gemm(mma, acc, frA[(None, None, kb, 0)], frB[(None, None, kb, 0)], acc)
            mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(mbar)
    cute.arch.mbarrier_wait(mbar, 0)
    …                                                          # readback: Ld32x32b x64 (the readback section of tcgen05_umma_issue_cta_pair)
```

### Station B″: `Ld16x256b` readback fragment of a TMEM accumulator → bf16 → `stmatrix` (non-transposed) → UMMA B operand face (K-major SW128)

Station B's register fragment comes from a warp-level MMA. When the fragment comes from a **`tcgen05.ld` `Ld16x256b`
readback** (the same fragment shape, see "the fragment shape decides the atom" at the top) and the destination is an (N, K),
K-contiguous UMMA B operand face, the assembly = station A's derivation + station B's destination face:

- the r2s tiled copy is **derived from the t2r**: `make_tiled_copy_D(atom, t2r)` -- D side = smem, inheriting the thread-value
  map of the t2r register side;
- the atom is chosen by the public helper from the t2r fragment shape + the destination face's leading dim:
  `get_smem_store_op(LayoutEnum.ROW_MAJOR, bf16, f32, t2r)` returns `StMatrix8x8x16bOp(False, 4)` ((N, K) face with K
  contiguous → leading dim 1 → `ROW_MAJOR` → `is_m_major=False` → non-transposed member, consistent with gotcha 2's criterion);
- the register side only does `retile`, the face side does `thr.partition_D(flat (N, K) view)`; after writing,
  `fence_proxy("async.shared", space="cta")` + barrier, then `make_fragment_B(face)` hands it to the descriptor read.

What is certified is the **bijection**: whether the (i, j) that the t2r `partition_D(identity)` assigns to each register
element is the (i, j) at which `retile` + `partition_D` land it on the face. This is an assertion about the consistency of two
partition call sites; it is not reasoned out but measured: the same fragment, the same `BF16(...)` conversion, a per-element
scalar `face[i, j] = v` path as the control, and two patterns that are injective at different granularities (how to write them:
[`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 11), each bit-for-bit equal at `0.0`. GPU-certified readings (128 threads,
64 × 64 tile):

| Item | Reading |
|---|---|
| t2r | `Ld16x256bOp(Repetition.x8)`, `make_tmem_copy` built on `flat_divide(plane, (64, 64))[(None, None, 0, 0)]` |
| threads defined by the copy / values per thread | 128 / 32 (`cute.size(t2r.layout_dst_tv_tiled, mode=[0])` = 128) |
| coordinate tensor layout (`partition_D(identity)`, after slicing off the rest modes) | `(((2,2,8),1),1,1):(((1@1,8@0,8@1),0),0,0)` |
| r2s atom | `StMatrix8x8x16bOp`: TV Src `(32,(2,4)):(2,(1,64))`, TV Dst `(32,8):(8,1)` |
| `r2s.retile(rF)` | `((8,4),1,1):((1,8),0,0)` |
| `thr_r2s.partition_D(face)` | `((8,4),1,1):((1,16),0,0)` |
| expected SASS (as stated in the landing record; this library's census axes have no `STSM` / `STS.U16` count, and the landing only left trace-time prints of the two sides' layouts, see "Certification boundary") | per thread per face 32 `STS.U16` → `STSM.16.M88.4` × (elements per thread / 8) = 4 |

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as bh
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils.layout import LayoutEnum
BF16, F32 = cutlass.BFloat16, cutlass.Float32
M, N, THREADS = 64, 64, 128          # accumulator tile = the face's (N_mma, K); 4 warps issue stmatrix as a group

@cute.struct
class Storage:
    tmem_holding_buf: cutlass.Int32

def _keep3(t):                        # plain Python helper: slice off the rest modes left by tiling (all 1), keep only (V, M_it, N_it)
    r = int(cute.rank(t))
    return t if r <= 3 else t[tuple([None] * 3 + [0] * (r - 3))]

@cute.jit
def host(gPat, gRef, gStm):
    mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        BF16, F32, (M, N, 16), tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))          # only borrow its accumulator layout
    atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.K_SW128, BF16)
    face_l = cute.coalesce(cute.tile_to_shape(atom, (M, N), order=(0, 1)), target_profile=(1, 1))
    kernel(gPat, gRef, gStm, mma, face_l).launch(grid=[1, 1, 1], block=[THREADS, 1, 1])

@cute.kernel
def kernel(gPat, gRef, gStm, mma: cute.TiledMma, face_l: cute.ComposedLayout):
    tidx, _, _ = cute.arch.thread_idx()
    smem = utils.SmemAllocator(); storage = smem.allocate(Storage)
    # allocate the face with operand-face discipline: layout=outer, swizzle=inner, 1024 B aligned. The second argument is a layout;
    # passing it a shape tuple is gotcha 13 (trace-time `'tuple' object is not callable`, without naming the line).
    sRef = smem.allocate_tensor(element_type=BF16, layout=face_l.outer, byte_alignment=1024, swizzle=face_l.inner)
    sStm = smem.allocate_tensor(element_type=BF16, layout=face_l.outer, byte_alignment=1024, swizzle=face_l.inner)
    flat = cute.make_layout((M, N), stride=(N, 1))         # row = 128 B = one SW128 atom → the flat view is a relabel (gotcha 3)
    wRef = cute.make_tensor(sRef.iterator, flat)
    wStm = cute.make_tensor(sStm.iterator, flat)
    for e in cutlass.range_constexpr(M * N // THREADS):      # sentinel prefill: a value the pattern cannot produce, exact in bf16 (power of 2)
        idx = tidx + e * THREADS
        wRef[idx // N, idx % N] = BF16(F32(-1024.0))
        wStm[idx // N, idx % N] = BF16(F32(-1024.0))
    cute.arch.sync_threads()

    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                               barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS))
    tmem.allocate(512); tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(F32); tmem.relinquish_alloc_permit()
    acc = cute.make_tensor(tmem_ptr, mma.make_fragment_C(mma.partition_shape_C((M, N))).layout)
    plane = acc[((None, None), 0, 0)]                         # (M, N) plane; this example only borrows the layout, it does not read TMEM

    # ① t2r: Ld16x256b x8, built on one tile of the plane after a whole-tile flat_divide
    t2r = tcgen05.make_tmem_copy(
        cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), F32),
        cute.flat_divide(plane, (M, N))[(None, None, 0, 0)])
    thr_t2r = t2r.get_slice(tidx)
    # ② absolute (i, j) of each register element: the identity goes first through the MMA's C partition (the accumulator's own frame), then the t2r D-side partition
    idC = mma.get_slice(0).partition_C(cute.make_identity_tensor((M, N)))[((None, None), 0, 0)]
    tcrd = _keep3(thr_t2r.partition_D(cute.flat_divide(idC, (M, N))))
    rF = cute.make_rmem_tensor(tcrd.shape, BF16)              # fragment shaped by the coordinate tensor (= D-side partition), not by partition_S

    # ③ r2s derived from t2r; the atom is chosen by the helper -- one of the contracts is that it must not fall back to CopyUniversalOp (gotcha 15)
    atom_r2s = bh.get_smem_store_op(LayoutEnum.ROW_MAJOR, BF16, F32, t2r)   # → StMatrix8x8x16bOp(False, 4)
    r2s = cute.make_tiled_copy_D(atom_r2s, t2r)
    thr_r2s = r2s.get_slice(tidx)

    for e in cutlass.range_constexpr(cute.size(rF)):
        i, j = tcrd[e][0], tcrd[e][1]
        v = BF16(gPat[i, j])                                  # same conversion on both paths, so any difference can only come from placement
        rF[e] = v
        wRef[i, j] = v                                        # control path: per-element scalar STS.U16
    src = r2s.retile(rF)
    dst = thr_r2s.partition_D(wStm)                           # print both layouts before the copy (gotcha 14)
    cute.copy(r2s, src, dst)                                  # STSM.16.M88.4 × 4 per thread
    cute.arch.fence_proxy("async.shared", space="cta")        # then hand to the UMMA descriptor read: mma.make_fragment_B(sStm)
    cute.arch.sync_threads()
    for e in cutlass.range_constexpr(M * N // THREADS):      # readback: gRef and gStm bit-for-bit equal, with no sentinel left
        idx = tidx + e * THREADS
        gRef[idx // N, idx % N] = wRef[idx // N, idx % N]
        gStm[idx // N, idx % N] = wStm[idx // N, idx % N]
    cute.arch.sync_threads()
    tmem.free(tmem_ptr)
```

### Station C write-side acceptance: read the SASS census, not `merge.scalar_sts_in_loop`

**Status: ✅ Verified** (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=0.0`, certification round 2).
For station C's assembly see [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md); this only adds "how to tell that the staging-face write side really uses only matrix atoms".

> **Measured readings**: pair form, C's contiguous dimension is MMA-M (`LayoutEnum.COL_MAJOR`), two slicings -- helper sub-blocks (several sub-blocks, several stages)
> and one whole block (column count not a power of 2, 1 stage; in the 64-lanes-per-CTA frame the helper chose x1 + 2 matrices for it, which differs from the choice in
> gotcha 19 of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md) in the 128-lane frame, so check the helper per frame) -- both go
> `get_tmem_load_op → get_smem_store_op → make_tiled_copy_D(atom, t2r)`, the helper picks
> `Ld16x256b` + `StMatrix8x8x16bOp` (transposed), SASS has `STSM.16.MT88.*` > 0 and `STS.U16` = 0, numerics bit-exact. **Both kernels have only one epilogue block
> and no user loop, yet the census's `merge.scalar_sts_in_loop` is still 4**, and none of the 4 are on the staging face: three 32-bit `STS [UR+0x38]` are writes of the
> TMEM base-address publish slot (one of them right next to the internal retry call of the tcgen05 allocation), one `STS.U8` is right next to
> `UVIRTCOUNT.DEALLOC.SMPOOL` (smem pool bookkeeping at teardown); the region partition puts them in `loop` because the allocation's retry back edge also counts
> as a back edge. **Conclusion: `scalar_sts_in_loop` ≈ 4 is the baseline of `TmemAllocator` allocation / publish + teardown, not a staging-face degradation.**

How to run the census (the parent process turns on cubin dumping for the compile subprocess, then disassembles; the tools are the census modules under this repo's `skills/sass/`):

```python
env.update({"CUTE_DSL_KEEP_CUBIN": "1", "CUTE_DSL_DUMP_DIR": dump_dir})
subprocess.run([sys.executable, probe, variant], env=env, cwd=dump_dir)
txt = subprocess.run(["nvdisasm", "-c", cubin], capture_output=True, text=True).stdout   # fallback: cuobjdump -sass
txt = orch_census_run.resolve_labels(txt)          # replace `(.L_x_N) with addresses, otherwise back edges are not recognized and every "loop" is empty
instrs = sass_tools.parse_sass(txt).instructions
regions = orch_tools.find_regions(instrs)          # innermost back-edge region = "loop"
for i, ins in enumerate(instrs):
    if orch_tools.fam_of(ins.opcode) == "STS" and "STSM" not in ins.opcode:
        r = orch_tools.region_of(regions, i)       # r.kind: loop / straight; r.label: spin / copy-producer / …
        print(ins.addr, ins.opcode, ins.operands, r.kind, r.label, instrs[i-3:i+4])   # context decides ownership
# contract: STSM.* count > 0 and STS.U16 == 0; list every other STS with its region and context as a fact row
```

Ownership and handling (`merge.scalar_sts_in_loop` is only a count; it does not distinguish what is being written):

- **`STS.U16` / with 16-bit data**: per-element degradation of the staging face; some epilogue instance did not get `StMatrix` -- check that instance's t2r atom (gotcha 1, gotcha 15).
- **32/64-bit `STS` with a constant or uniform address and phase / count / address data**: bookkeeping writes (TMEM base-address publish slot, state words). When they fall in a `loop` region, read the context first:
  most are one-time writes swept in by a wait / allocation retry back edge. It is only worth investigating when the count **exceeds** the baseline above, or an `STS` with 16-bit data appears.
- **Kernels compiled into several copies by shape**: census each cubin separately; censusing only the largest one misses degradation in the small instances.

## Gotchas

1. **The `Ld32x32b` fragment shape has no matrix-atom partner.** Thread = lane, values = 32 contiguous 32-bit columns: each
   (thread, column) holds only one 16-bit element, so the "a pair of adjacent 16-bit values per thread, 8 rows per group" that
   `stmatrix` / `ldmatrix` require does not exist, and a transposed write to the face has only `STS.U16` × element count left.
   Only switching the TMEM readback to `Ld16x256b` (`Repetition.x8` covers 64 columns) gives stations A / C their matrix path;
   keep `Ld32x32b` for per-lane paths (state-scaling write-back, pairwise packing into the A face). The public helper's
   condition table is exactly this criterion (`get_smem_store_op`: only `num_dp == 16 && num_bits == 256 && 16-bit destination`
   returns `StMatrix8x8x16bOp`).
2. **Choosing the wrong member is silent.** The verifier only checks 16 B row alignment and contiguity, not whether the
   `transpose` boolean is right; with the orientation reversed it still compiles and moves transposed data. The criterion is
   **which mode of the face is contiguous**: mode 0 (the register tile's row) contiguous → transposed member; mode 1 contiguous →
   non-transposed member. Station A's (C, R) view has mode 0 = m contiguous → `True`; station B's (N, K) face has mode 1 = K
   contiguous → `False`, and the (K, N) view has mode 0 = K contiguous → `True`. Run once with an asymmetric integer pattern
   (this entry's example `((r + 3c) % 7) − 3`).
3. **A transposed view is a relabel, not a second layout.** `cute.make_tensor(t.iterator, cute.select(t.layout, [1, 0]))`:
   the swizzle stays on the pointer and the mode order is swapped. A flat view is legal under one condition only: **a face row
   is exactly one SW128 atom wide (128 B, i.e. 64 columns of bf16)** -- then the face's affine part is row-major, and both
   `make_tensor(face.iterator, make_layout((rows, 64), stride=(64, 1)))` and its transpose `((64, rows), stride=(1, 64))` are
   relabels (station B's 64×64 B faces and the K = 64 faces produced by `make_smem_layout_a/b` all belong here; the helper's face
   has four modes ((MN, K_mma), 1, MMA_K, STAGE), slicing it as three modes (MN, K, STAGE) is rejected, so if you want a view you
   must go through the flat relabel). For a face whose columns exceed 128 B (two stacked SW128 boxes: (R, (64, 2)):(64, (1, R·64)))
   you **must not** hand-write `(C, R):(1, C)` -- column block 1 lands at `+R·64`, not `+64`, and the flat view silently reads
   half the face wrong; use `select`, or write it per box as `((64, 2), R):((1, R·64), 64)`.
4. **Start both sides from the same tiled copy.** Station A's LDSM copy is derived from the TMEM copy by
   `make_tiled_copy_D(atom, tiled_copy_t2r)` (D side = the thread-value layout of the t2r register side), with
   `partition_S(face_view)` on the face side and `retile(rV)` on the register side; `rV` is built from
   `partition_D(tmem_tile).shape` and has the same shape as `rAcc`. Hand-writing a TV layout, or building the register tensor
   from the `partition_S` shape, makes `cute.copy` report unequal rank / modes (same as gotchas 1–2 of
   [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md)).
5. **Move the M iteration mode too.** 4 warps × `Ld16x256b`: warp w covers lanes 32w..32w+31 in two 16-lane iterations, so
   mode 1 of `partition_S(plane)` / `partition_D(plane)` = 2; `cute.copy` moves the whole (V, 2, 1) at once, and the LDSM side's
   `partition_S(sVt)` is likewise (V, 2, 1) -- slicing off the iteration mode and moving only half is a silent error.
6. **Proxy discipline**: TMA write → LDSM read: waiting on the full mbarrier is enough (both are smem visible on the generic
   side); STSM write → UMMA descriptor read: every writer does `fence_proxy("async.shared", space="cta")` + barrier, then the
   issuing warp calls `cute.gemm`; `tcgen05.ld` is asynchronous: `fence_view_async_tmem_load()` before using the registers;
   `tcgen05.st` followed by an `ld` of the same columns: `fence_view_async_tmem_store()` + sync.
7. **Slice indices**: the warp-level MMA and its STSM copy are sliced by `lane_idx()`; the TMEM copy and the LDSM copy derived
   from it are sliced by `tidx` (0..127, 4 warps as a group); each pair of `get_slice` calls is consistent within itself and
   must not be mixed. Trace-time error text when mixed:
   `AttributeError: 'ThrCopy' object has no attribute 'partition_C'` (using a copy's thread slice for the MMA's C partition);
   the error when the two sides do not start from the same tiled copy is gotcha 4's
   `Expected source and destination tensors to have the same size in mode-1, but got 2 and 1`.
   Both appear at trace time and both carry line numbers; fix the assembly per gotchas 4 / 7 -- they are not mechanism problems.
8. **STSM sub-block coordinates may be dynamic**: `flat_divide(face_view, (16, tile_n))[(None, None, warp_idx, j)]`, with
   `warp_idx` a runtime value -- the `stmatrix` address is computed per lane at run time.
9. **`num_matrices` = tile element count ÷ 64.** A single-warp `m16n8k16` C tile is (16, 8) → x2; giving `make_tiled_mma` a
   `(16, 16, 16)` permutation widens the N tile to 16 → x4 (4 `STSM.x4` per warp instead of 8 `x2`).
10. **Acceptance cannot rely on `max_diff` alone, nor use `bankconf_*` as the acceptance bar.** The scalar path is numerically
    correct too. The machine-readable criterion is **per station**: the count of `ldmatrix … .trans` / `stmatrix` in PTX and the
    distribution of `.trans`, SASS `LDSM.16.MT88.4` / `STSM.16.M88.4` / `STSM.16.MT88.4` count = elements per thread / 8, the
    station's loop-body `STS.U16` / `LDS.U16` go to zero, and `merge.scalar_sts_in_loop` drops by that station's count.
    NCU `bankconf_st` / `bankconf_ld` are the whole kernel's combined account, and after this entry's first landing they did not
    return to normal (see "Certification boundary") -- their change must be attributed separately to other scalar stations and
    must not be read as this entry having or not having taken effect.
11. **Inside a `@cute.jit` / `@cute.kernel` body, do not write boolean operations on Python objects such as
    `getattr(a, …) or getattr(b, …)`.** The preprocessor rewrites `and` / `or` into DSL operations, and when the operands are
    functions / modules it reports `<class 'function'> is not supported` (the whole jit fails and it looks like "the kernel
    does not compile"). Resolve the helper / atom kind in a plain Python function and call it from the jit body.
12. **Station C's two certified disciplines still apply**: the staging face must carry a swizzle (otherwise the `stmatrix`
    write side has an 8-way conflict by construction), and a swizzled face has no second view (the TMA store descriptor is
    built from the face itself) -- see gotchas 11–12 of
    [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md).
13. **The second positional argument of `SmemAllocator.allocate_tensor` is a layout, not a shape.** Writing
    `smem.allocate_tensor(BF16, (16, 66), byte_alignment=128)` (meaning a 16 × 66 work buffer) dies at **trace time** with
    `TypeError: 'tuple' object is not callable`, and the traceback **does not name that line** -- it looks as if the kernel
    fails to compile somewhere else. The fix is `smem.allocate_tensor(BF16, cute.make_layout((16, 66)), byte_alignment=128)`
    (or the keyword `layout=`). This gotcha cost six GPU rounds before station B″ was certified: the changes in all six probe
    versions went into making `tile_n` a Constexpr, building the tiled copy in the host vs. the kernel, and adding or removing
    `retile` -- each one a construct that "looked suspicious" -- while the `allocate_tensor(dtype, shape_tuple)` call was never
    touched in those six versions; only after a seventh rewrite did the docstring record the cause of death. **When the
    trace-time error text does not contain your line number, bisect the kernel body first (comment out the second half and see
    whether it still dies); do not edit the most suspicious line** ([`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 12).
14. **Print both sides' layouts before `cute.copy`.** Print `src = r2s.retile(rF)` and `dst = thr.partition_D(face)` each with
    `print(str(x.layout))` (a trace-time print that lands in the job log), then call `cute.copy(r2s, src, dst)`: if the copy
    rejects the pair, the rank / mode mismatch error sits on record next to the two layouts that produced it, instead of a lone
    error message. Station B″'s legal readings are `((8,4),1,1):((1,8),0,0)` vs. `((8,4),1,1):((1,16),0,0)` (same shape, only
    the strides differ).
15. **`get_smem_store_op` silently falls back to `CopyUniversalOp`.** It returns `StMatrix8x8x16bOp` only when the t2r is
    16-datapath / 256-bit and the destination is 16-bit; in every other case it returns a plain `st.shared` atom, **which still
    compiles and is still numerically correct** -- the change just becomes "a scalar write with an extra detour". So code that
    uses it to pick the atom must write "the returned op is in the StMatrix family" as a **non-numerical contract** (log
    `type(atom.op).__name__` at trace time and judge on the host side); otherwise every later numerical contract passes while
    SASS still has `STS.U16`. Likewise write "threads defined by the copy == threads issuing it" as a contract
    (`cute.size(t2r.layout_dst_tv_tiled, mode=[0])`): `stmatrix` is a warp-collective instruction, and a predicated
    `cute.copy` is a different contract.
16. **Whether a `@cute.struct` scalar field needs `.ptr` changes with the DSL version.** This entry's examples (and several
    earlier certified probes) write `TmemAllocator(storage.tmem_holding_buf.ptr, …)`; on another image the same code throws
    `AttributeError: '_Pointer' object has no attribute 'ptr'` at trace time -- there the `cutlass.Int32` / `cutlass.Int64`
    field is itself a pointer, so pass `storage.tmem_holding_buf` directly (likewise `two_cta_tmem_dealloc_mbar_ptr=`). When you
    see this error, drop `.ptr`; `MemRange` fields still use `.data_ptr()`.

## Sources

- Certification probe -- **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, certification round 2; round 1 was a trace-time spelling error in the probe itself, not a mechanism error). All four contracts passed in the same round (`X0-lowers` n/a, `X1-ldsm-mt-iso` 0.0, `X2-stsm-m88-umma-b` 0.0, `X3-stsm-mt88-umma-b` 0.0). Readings: X2/X3 A face `S<3,4,3> o 0 o ((128,16),1,4,1):((64,1),0,16,0)`, B face `S<3,4,3> o 0 o ((64,16),1,4,1):((64,1),0,16,0)`, STSM built with `tile_n=16, num_matrices=4, trans=True`; X1 face `S<3,4,3> o 0 o (64,(64,2),1):(64,(1,4096),0)`, transposed view via `select` gives `((64,2),64):((1,4096),64)`, `make_tiled_copy_D` derives src/dst both `((8,8),1,1)`, tiler `((4,32):(32,1), 64:1)`; fact row `X1n`: the same derivation applied to `Ld32x32b x64` is rejected by the compiler (internal error, IR verification failed) -- the measured reading for gotcha 1. The four contracts:
  `X0-lowers` (all three variants compile to launchable kernels); `X1-ldsm-mt-iso` -- TMEM (128 × 64) f32 filled through `St32x32b`,
  read back with `Ld16x256b x8`; a K_SW128 (64 × 128) bf16 face read through `LdMatrix8x8x16bOp(True, 4)` (tiled copy derived from the t2r,
  transposed view) into a register tensor of the same shape; `out[m, n] = acc[i] − face[i]` written out at the t2r identity coordinates and judged against `A[m, n] − V[n, m]`
  with `max_diff == 0`; `X2-stsm-m88-umma-b` -- a 64×64 m16n8 register tile written through `StMatrix8x8x16bOp(False, n)` into
  the (N, K) face of `make_smem_layout_b`, read by UMMA 128×64×16 × 4 k-blocks through the descriptor, read back with `Ld32x32b x64`, judged against `A · Rᵀ`
  at 0; `X3-stsm-mt88-umma-b` -- the same tile written through `StMatrix8x8x16bOp(True, n)` into the (K, N) transposed view, judged against `A · R` at 0.
  Fact row `X1n`: the same derivation applied to a t2r of `Ld32x32b x64`; whether it is rejected / can lower is only recorded as a fact (gotcha 1's reading).
  Each variant runs in its own subprocess (`sys.executable`) with a 300 s budget; trace-time spelling candidates (the transposed view via `select` /
  `coalesce+select`, `make_tiled_copy_D` / `_S`, STSM x4 / x2) are swept inside the trace and recorded as fact rows.
- Certification probe for station B″ and gotchas 13–15 -- **Status: ✅ Verified**
  (`PROBE PASS max_diff=0.0`, passed after 7 rounds). All four contracts passed in the same round: `X0-atom` n/a --
  `get_smem_store_op` returns `StMatrix8x8x16bOp` for the shipped `Ld16x256b x8` t2r (reading: atom TV Src
  `(32,(2,4)):(2,(1,64))`, TV Dst `(32,8):(8,1)`); `X1-threads` n/a -- the copy is defined on 128 threads; `X2-fine` 0.0 --
  with a pattern injective within a 16 × 16 tile, `((i % 16) * 16 + (j % 16)) − 128`, the matrix path and the per-element scalar path are bit-for-bit equal; `X3-coarse` 0.0 --
  with a pattern injective across 16 tiles of 16 × 16 and constant within a tile, `((i // 16) * 4 + (j // 16)) * 17 − 128`, same verdict. Coordinate tensor
  `(((2,2,8),1),1,1):(((1@1,8@0,8@1),0),0,0)`, `retile` side `((8,4),1,1):((1,8),0,0)`, face side `((8,4),1,1):((1,16),0,0)`.
  The first six rounds produced no verdict: all six versions were a hand-built warp-level MMA toy geometry and died on the gotcha 13 call (recorded in the seventh version's docstring).
  The same construction was judged 0.0 again with an asymmetric small-integer pattern in contract `X5-stsm-m88-bface` of another probe (5 PASS in the same round)
  (that probe's docstring also records the trace-time errors of two failed direct landings, i.e. the two error texts added in gotcha 7).
- Certification probe for station C write-side acceptance + SASS census and gotcha 16 (its geometry is taken from a specific kernel instance) -- **Status: ✅ Verified**,
  `PROBE PASS max_diff=0.0` (certification round 2; in round 1 all five contracts FAILed, the cause being gotcha 16's `AttributeError`). All five contracts passed in the same round:
  `C0-lowers` n/a, `C1-exact-*` 0.0 (helper sub-blocks), `C2-exact-*` 0.0 (single block, column count not a power of 2), `C3-atom` n/a (t2r is 16dp/256b, r2s is
  `StMatrix8x8x16bOp`), `C4-sass` n/a (`STSM` > 0 and `STS.U16` == 0; every other scalar `STS` listed as a fact row with its region / context; both variants have
  `scalar_sts_in_loop` = 4, all of them bookkeeping writes).
- First-landing readings (**production price [UNVERIFIED-BOUNDARY]**, see "Certification boundary"): two KEEPs on the same production tree
  (station B: both B operand faces switched to `STSM.16.M88.4`, same-job ratio 0.307 → 0.375; stations A + C + `Ld16x256b`
  readback + `St16x128b` packing, 0.375 → 0.389), both numerically passing; NCU `bankconf_st` 1,468,716 → 1,379,211 → 1,229,880,
  `bankconf_ld` 862,197 → 1,068,068 → 1,096,613, `merge.scalar_sts_in_loop` 238 → 174 → 110, `tensor_pct`
  1.62 → 2.00 → 2.08. The two earlier direct landings (station B) FAILed correctness at trace time, with the two error texts of gotcha 7. The ms was not priced separately
  from the other changes in the same bundle, and the bank-conflict axis did not return to normal -- neither is part of this entry's certified surface.
- Public interfaces: `cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp` / `StMatrix8x8x16bOp` / `MmaF16BF16Op`;
  `cutlass.cute.nvgpu.tcgen05.Ld16x256bOp` / `Ld32x32bOp` / `St32x32bOp` / `Repetition` / `make_tmem_copy` /
  `make_smem_layout_atom` / `SmemLayoutAtomKind.K_SW128` / `MmaF16BF16Op` / `Field.ACCUMULATE` / `commit`;
  `cutlass.utils.blackwell_helpers.make_smem_layout_a` / `make_smem_layout_b` / `get_smem_store_op` / `get_tmem_load_op`;
  `cute.make_tiled_copy_D` / `make_tiled_copy_C` / `make_copy_atom` / `make_tiled_mma` (`permutation_mnk`) /
  `TiledCopy.retile` / `ThrCopy.partition_S` / `partition_D` / `TiledMma.partition_shape_C` / `make_fragment_A` /
  `make_fragment_B` / `make_fragment_C` / `cute.select` / `cute.coalesce` / `cute.flat_divide` / `cute.zipped_divide` /
  `cute.make_identity_tensor` / `cute.make_rmem_tensor` / `cute.tile_to_shape`;
  `cute.arch.fence_proxy` / `fence_view_async_shared` / `fence_view_async_tmem_load` / `fence_view_async_tmem_store` /
  `lane_idx` / `warp_idx` / `make_warp_uniform` / `elect_one` / `mbarrier_init` / `mbarrier_init_fence` / `mbarrier_wait`;
  `cutlass.utils.TmemAllocator` / `SmemAllocator`; `cutlass.pipeline.NamedBarrier` / `sync`.
- Official docs: PTX ISA `ldmatrix`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>,
  `stmatrix` <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-store-instruction-stmatrix>,
  the per-lane distribution table of the `mma.m16n8k16` C fragment
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>,
  the thread-data distribution diagrams of each `tcgen05.ld` shape (`.16x256b` and `.32x32b`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld>;
  CUTLASS CuTe DSL docs <https://docs.nvidia.com/cutlass/latest/> (the epilogue section of the Blackwell GEMM example;
  the stmatrix selection condition of `get_smem_store_op` is in the public helper source).
- Related entries: station C and "staging-face swizzle / no second view" in [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md);
  packing the A operand back into TMEM after station A in [`tmem_r2t_recast_fragment`](tmem_r2t_recast_fragment.md) and
  [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md); smem traffic triage in
  [`smem_traffic_triage`](smem_traffic_triage.md).
