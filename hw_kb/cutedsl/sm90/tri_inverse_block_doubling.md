# tri_inverse_block_doubling (**in-place** inversion of a 64×64 unit lower-triangular block on a 16-bit smem face: diagonal 8×8 forward substitution → `mma.sync` block doubling 8→16→32→64 → the same bytes read in and out as both fp16 and bf16)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Division of labour: [`tri_forward_subst_shfl`](tri_forward_subst_shfl.md) handles the serial forward substitution of
> the small diagonal blocks; §4 of [`warp_mma_f16`](warp_mma_f16.md) handles the acc→A register hookup;
> [`mma_sync_subtile_broadcast_splitk`](mma_sync_subtile_broadcast_splitk.md) handles the stride-0 broadcast view at
> h=8 and the warp-pair split-K reduction at h=32;
> [`wgmma_cfrag_publish_stsm`](wgmma_cfrag_publish_stsm.md) handles the C fragment → `stmatrix` ×4 publish path.
> This entry **assembles them into one complete, GPU-certified path**, and adds three things you only run into when
> assembling: (a) the A / B / C / broadcast operands of every inversion level are all **different read views of the
> same `MemRange`** (`get_tensor(layout, swizzle, dtype)`), with no pointer surgery; (b) the face is computed in
> **fp16** internally, while **the same bytes** are read and written in and out in the 16-bit type downstream wants
> (e.g. bf16); (c) the ordering of rendezvous and in-place overwrites between the seven segments.

## Purpose

A 128-thread warpgroup holds a 64×64 strictly lower-triangular matrix A (an f32 C fragment, from WGMMA or any
elementwise segment) and needs `(I + A)^{-1}` (optionally multiplied by a per-column / per-row scale) to hand to
downstream matrix instructions / descriptors.
Writing it as "one column per thread, a 64-step f32 scalar forward substitution" is numerically fine, but:

| Spelling | Per chunk per thread | Where it lands |
|---|---|---|
| per-column f32 scalar forward substitution | 2016 `LDS` (broadcast reads of A[r][c]) + a **truly serial chain** of 2016 `FFMA` + 64 resident f32 column values | `smem_wavefronts_ld` amplified several times, the warpgroup's register tier pushed up by the 64-deep column, tensor pipe idle |
| this entry: diagonal forward substitution + block doubling | diagonal 8×8: 28 `FFMA` + 16 `SHFL`; three doubling levels: 2 `m16n8k8` + 12 `m16n8k16` (shared over 4 warps); `ldmatrix`/`stmatrix` ≈ 30 each; 7 rendezvous | chain depth from 64 steps down to 8 steps + 3 levels; the working set is warp-level fragments (< 40 registers per level), no tier raise needed |

The formula is the public blocked triangular inverse: `[[T11, 0], [A21, T22]]^{-1} = [[T11^{-1}, 0], [−T22^{-1}·A21·T11^{-1}, T22^{-1}]]`.
The block edge h doubles every level, and each level only computes the corner block `Z = −(T22 · A21) · T11` (two
GEMM steps; T11 / T22 are already inverses at that point), landing it back at the position of A21.

## When needed (trigger)

- The contract / ledger says in-place inversion with `HMMA.1688 / 16816` + `LDSM/STSM`, while the implementation is
  `for r: for c<r: s += Float32(face[(r,c)]) * x[c]`;
- the role's segment timing / PM timeline: the inversion segment takes the bulk of the whole period, while the
  downstream peer waits for its full signal;
- NCU: `smem_wavefronts_ld` 5–10× that of an implementation of the same shape, `tensor_pct` collapsed to single digits,
  `spill ≠ 0` with the spill appearing after raising some role's register tier
  (the 64-deep f32 column + descriptor addresses + the live set of the elementwise segment broke through the tier);
- SASS: in that warpgroup's ledger `LDS` (non-`.128`) + `FFMA` take the top two places, and `HMMA` is absent.

## Mechanism: seven segments, four warps, nine read views of one MemRange

Face: 64×64 16-bit, native `K_SW128` atom (`warpgroup.make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, Float16)` +
`tile_to_shape(atom, (64, 64), order=(0, 1))`), element coordinates `(r, c) ↦ r·64 + c`, swizzle on the pointer; the
`MemRange` is aligned to **1024 B** (the swizzle period). Every read view is
`st.f.get_tensor(<layout>, swizzle=t_layout.inner, dtype=...)`:

| Read view | layout (in elements) | Used by | Contiguous mode → member |
|---|---|---|---|
| `(r, c)` fp16 | `t_layout.outer` | entry STSM, the diagonal forward substitution's `flat_divide(·,(8,8))`, exit LDSM | c contiguous |
| `(r, c)` bf16 | same, `dtype` omitted (the MemRange's declared type) | exit STSM (downstream reads bf16) | the same bytes |
| `((m, 2), k, rb, cb)` | `((64, 0), 1, 512, 8)` | A and D at h=8: an 8-row block replicated to 16 rows via stride 0 | k contiguous → `LdMatrix(False, 1)` / `StMatrix(False, 1)` |
| `(n, k, cb, rb)` | `(1, 64, 8, 512)` | B at h=8 (`(n, k)` = the transposed read of an 8×8 block) | n contiguous → `LdMatrix(True, 1)` |
| `(m, k, rb, cb)` | `(64, 1, 1024, 16)` | A / D at h=16 | `LdMatrix(False, 4)` / `StMatrix(False, 2)` |
| `(n, k, cb, rb)` | `(1, 64, 16, 1024)` | B at h=16 | `LdMatrix(True, 2)` |
| `(m, k, rb16, cb32)` | `(64, 1, 1024, 32)` | A at h=32 (16 rows × K=32) and the output strip | `LdMatrix(False, 4)` / `StMatrix(False, 2)` |
| `(n, k, cb16, rb32)` | `(1, 64, 16, 2048)` | B of GEMM1 at h=32 (N=16, K=32) | `LdMatrix(True, 2)` |
| `(n, k, cb32, rb16)` | `(1, 64, 32, 1024)` | B of GEMM2 at h=32 (N=32, K=16) | `LdMatrix(True, 2)` |

The first two modes are the operand tile, the last two are block coordinates; the kernel takes a block with
`view[(None, None, rb, cb)]` (`rb/cb` may be runtime values derived from `warp_idx()`).
The `flat_divide` + `cute.select` + `iterator.align(16)` combination is **not needed**: `.align()` on a swizzled pointer
drops the swizzle attribute (see gotcha 1), while the second read view via `get_tensor` keeps the swizzle on the
pointer as is.

| Segment | Who | Does what | Instructions (static count) |
|---|---|---|---|
| 1 entry | 128 threads | f32 C fragment → fp16 → face (`make_tiled_copy_S(get_smem_store_op(ROW_MAJOR, f16, f32), make_tiled_copy_C_atom(StMatrix(False,4), mma))`) | `STSM.16.M88.4` ×4 |
| 2 diagonal | warps 0/1 (4 blocks per warp, 8 lanes per block) | transposed-system back substitution of the 8 unit lower-triangular 8×8 blocks, resident rows `LDS.128`, `SHFL.IDX` off the chain, f32 FFMA, `STS.128` in place | `LDS.128` ×1, `SHFL` ×16, `FFMA` ×28, `STS.128` ×1 |
| 3 h=8 | 4 warps, one pair each | `Z = −(T22·A21)·T11`, A/D through the stride-0 broadcast view (the whole partition copied as usual; the second store is an alias) | `LDSM.x1` ×2 + `LDSM.T.x1` ×2, `HMMA.1688` ×2, `STSM.x1` ×2 |
| 4 h=16 | warps 0/1, one pair each | same as above, `m16n8k16` | `LDSM.x4` ×1 + `LDSM.T.x2` ×4, `HMMA.16816` ×4, `STSM.x2` ×2 |
| 5–6 h=32 | 2×2 warp grid | the N split of GEMM1 = the K split of GEMM2; two-barrier smem reduction (x=0 writes first, x=1 reads, adds and writes) | `LDSM.x4` ×2 + `LDSM.T.x2` ×8, `HMMA.16816` ×8, `STSM.x2` ×4 (+ reduction `LDSM.x2` ×4) |
| 7 exit (removable, gotcha 14) | 128 threads | read back in C-fragment order with `make_tiled_copy_C(LdMatrix(False,4), mma)` → f32 elementwise (e.g. per-column scale) → target 16-bit type → `stmatrix` ×4 back into **the same bytes** | `LDSM.x4` ×4, `STSM.16.M88.4` ×4 |

7 named 128-thread barriers between segments (`cute.arch.barrier(barrier_id=…, number_of_threads=128)`); between
GEMM1→GEMM2 of each segment there is **no** smem round trip (acc→A hookup), and the h=8 / h=16 levels have no
inter-warp reduction. The A operand T22 and the B operands A21^T / T11^T of levels 3–6 are all **bytes the previous
level finished writing**, so the rendezvous at the start of each level cannot be dropped; the output strip at h=32
overwrites A21, which is also the B of this level's GEMM1 — so first "all warps have finished loading", then write
(gotcha 4).

## Minimal runnable example

Every segment below was actually run by the GPU-certified probe (the probe is not included in this release;
verdicts in "Sources").

```python
import cutlass, cutlass.cute as cute, cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.utils import LayoutEnum
F16, BF16, F32, I32 = cutlass.Float16, cutlass.BFloat16, cutlass.Float32, cutlass.Int32
FACE, NT, BAR = 64, 128, 1

@cute.struct
class SS:                                   # one face; declared as the type downstream wants, re-read as fp16 internally
    f: cute.struct.Align[cute.struct.MemRange[BF16, FACE * FACE], 1024]

@cute.jit
def host(gA, gD, gO):
    atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, F16)
    t_layout = cute.tile_to_shape(atom, (FACE, FACE), order=(0, 1))
    mma_wg = sm90_utils.make_trivial_tiled_mma(F16, F16, cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K, F32, atom_layout_mnk=(1, 1, 1), tiler_mn=(FACE, FACE))
    E = LayoutEnum.ROW_MAJOR                # c contiguous -> non-transposing member, four sites from one source
    r2s16 = cute.make_tiled_copy_S(sm90_utils.get_smem_store_op(E, elem_ty_d=F16, elem_ty_acc=F32),
        cute.make_tiled_copy_C_atom(cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), F16), mma_wg))
    r2sbf = cute.make_tiled_copy_S(sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
        cute.make_tiled_copy_C_atom(cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), BF16), mma_wg))
    tcX = cute.make_tiled_copy_C(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), F16), mma_wg)
    mma8 = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 8)), (1, 1, 1))
    tcA8 = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 1), F16), mma8)
    tcB8 = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 1), F16), mma8)
    tcC8 = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 1), F16), mma8)
    mma16 = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 16)), (1, 1, 1))
    tcA16 = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), F16), mma16)
    tcB16 = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), F16), mma16)
    tcCst = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), F16), mma16)
    tcCld = cute.make_tiled_copy_C(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), F16), mma16)
    # operand-role read views (in elements; face coordinate (r, c) -> r*64 + c)
    L_b8    = cute.make_layout(((8, 2), 8, 8, 8), stride=((64, 0), 1, 512, 8))   # ((m,rep),k,rb,cb)
    L_t8    = cute.make_layout((8, 8, 8, 8),      stride=(1, 64, 8, 512))        # (n,k,cb,rb)
    L_16    = cute.make_layout((16, 16, 4, 4),    stride=(64, 1, 1024, 16))      # (m,k,rb,cb)
    L_t16   = cute.make_layout((16, 16, 4, 4),    stride=(1, 64, 16, 1024))      # (n,k,cb,rb)
    L_16x32 = cute.make_layout((16, 32, 4, 2),    stride=(64, 1, 1024, 32))      # (m,k,rb16,cb32)
    L_t16x32 = cute.make_layout((16, 32, 4, 2),   stride=(1, 64, 16, 2048))      # (n,k,cb16,rb32)
    L_t32x16 = cute.make_layout((32, 16, 2, 4),   stride=(1, 64, 32, 1024))      # (n,k,cb32,rb16)
    kern(gA, gD, gO, mma_wg, r2s16, r2sbf, tcX, mma8, tcA8, tcB8, tcC8, mma16, tcA16, tcB16,
         tcCst, tcCld, t_layout, L_b8, L_t8, L_16, L_t16, L_16x32, L_t16x32, L_t32x16
         ).launch(grid=[1, 1, 1], block=[NT, 1, 1])
```

Kernel body (`chain(thr, acc1, M, N)` is the acc→A hookup of `warp_mma_f16` §4, unchanged; for
`_f16_lo/_f16_hi/_sel_eq` see gotcha 6):

```python
@cute.kernel
def kern(gA, gD, gO, mma_wg, r2s16, r2sbf, tcX, mma8, tcA8, tcB8, tcC8, mma16, tcA16, tcB16,
         tcCst, tcCld, t_layout, L_b8, L_t8, L_16, L_t16, L_16x32, L_t16x32, L_t32x16):
    tidx, _, _ = cute.arch.thread_idx()
    lane, wid = cute.arch.lane_idx(), cute.arch.warp_idx()
    st = cutlass.utils.SmemAllocator().allocate(SS)
    sw = t_layout.inner
    sF    = st.f.get_tensor(t_layout.outer, swizzle=sw, dtype=F16)     # internal read view
    sFbf  = st.f.get_tensor(t_layout.outer, swizzle=sw)                # exit read view (bf16)
    sFb8  = st.f.get_tensor(L_b8,  swizzle=sw, dtype=F16)
    sFt8  = st.f.get_tensor(L_t8,  swizzle=sw, dtype=F16)
    sF16  = st.f.get_tensor(L_16,  swizzle=sw, dtype=F16)
    sFt16 = st.f.get_tensor(L_t16, swizzle=sw, dtype=F16)
    sF16x32  = st.f.get_tensor(L_16x32,  swizzle=sw, dtype=F16)
    sFt16x32 = st.f.get_tensor(L_t16x32, swizzle=sw, dtype=F16)
    sFt32x16 = st.f.get_tensor(L_t32x16, swizzle=sw, dtype=F16)

    # ---- seg 1: C fragment -> fp16 -> STSM x4 (the three-layer r2s construction of wgmma_cfrag_publish_stsm)
    thr_wg = mma_wg.get_slice(tidx)
    cS = thr_wg.partition_C(cute.make_identity_tensor((FACE, FACE)))
    acc = cute.make_rmem_tensor(cS.shape, F32)          # in a real kernel = the WGMMA accumulator
    for i in cutlass.range_constexpr(cute.size(cS)):
        m, n = cS[i]; acc[i] = F32(gA[(m, n)])
    thr_r = r2s16.get_slice(tidx)
    stg = cute.make_layout(cute.shape(thr_r.partition_S(sF))[:3])
    s32 = cute.make_rmem_tensor(stg.shape, F32); s16 = cute.make_rmem_tensor(stg.shape, F16)
    tacc = r2s16.retile(acc)
    for v in cutlass.range_constexpr(cute.size(s32)):
        s32[v] = tacc[v]
    s16.store(s32.load().to(F16))
    cute.copy(r2s16, s16, thr_r.partition_D(sF))
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 1

    # ---- seg 2: 8 diagonal 8x8 blocks, warps 0/1, transposed-system back substitution (tri_forward_subst_shfl, NB=8)
    f8 = cute.flat_divide(sF, (8, 8))                                    # (8, 8, 8, 8)
    if wid < 2:
        c = lane % 8; blk = wid * 4 + lane // 8; r0 = blk * 8; base = lane - c; rr = r0 + c
        rRow = cute.make_rmem_tensor((8, 1), F16)
        cute.autovec_copy(f8[(rr % 8, None, rr // 8, r0 // 8)], rRow[(None, 0)])   # LDS.128
        rW = cute.recast_tensor(rRow, I32)                              # word q = (A[rr][2q], A[rr][2q+1])
        y = [F32(0.0)] * 8
        for qq in cutlass.range_constexpr(4):
            q = 3 - qq; s_odd = F32(0.0); s_even = F32(0.0)
            for t in cutlass.range_constexpr(6 - 2 * q):
                j = 7 - t
                w = cute.arch.shuffle_sync(rW[(q, 0)], base + j)         # depends only on loads: off the chain
                s_odd = s_odd + _f16_hi(w) * y[j]; s_even = s_even + _f16_lo(w) * y[j]
            y[2 * q + 1] = _sel_eq(2 * q + 1, c, F32(1.0), -s_odd)
            w = cute.arch.shuffle_sync(rW[(q, 0)], base + 2 * q + 1)
            s_even = s_even + _f16_lo(w) * y[2 * q + 1]
            y[2 * q] = _sel_eq(2 * q, c, F32(1.0), -s_even)
        rOut = cute.make_rmem_tensor((8, 1), F16)
        for i in cutlass.range_constexpr(8):
            rOut[(i, 0)] = F16(y[i])
        cute.autovec_copy(rOut[(None, 0)], f8[(rr % 8, None, rr // 8, r0 // 8)])   # STS.128 in place
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 2

    # ---- seg 3: h=8, one block pair b per warp: T11=(2b,2b) T22=(2b+1,2b+1) A21=(2b+1,2b)
    b = wid
    A8 = sFb8[(None, None, 2 * b + 1, 2 * b + 1)]     # ((8,2),8): T22, rows 8..15 alias rows 0..7
    O8 = sFb8[(None, None, 2 * b + 1, 2 * b)]         # Z lands back on A21
    B1 = sFt8[(None, None, 2 * b, 2 * b + 1)]         # (n,k) = A21^T
    B2 = sFt8[(None, None, 2 * b, 2 * b)]             # (n,k) = T11^T
    thr = mma8.get_slice(lane)
    thrA, thrB, thrC = tcA8.get_slice(lane), tcB8.get_slice(lane), tcC8.get_slice(lane)
    frA = thr.make_fragment_A(thr.partition_A(A8));   cute.copy(tcA8, thrA.partition_S(A8), thrA.retile(frA))
    frB1 = thr.make_fragment_B(thr.partition_B(B1));  cute.copy(tcB8, thrB.partition_S(B1), thrB.retile(frB1))
    frB2 = thr.make_fragment_B(thr.partition_B(B2));  cute.copy(tcB8, thrB.partition_S(B2), thrB.retile(frB2))
    acc1 = cute.make_rmem_tensor(thr.partition_shape_C((16, 8)), F32); acc1.fill(0.0)
    cute.gemm(mma8, acc1, frA, frB1, acc1)                               # P = T22·A21
    for i in cutlass.range_constexpr(cute.size(acc1)):
        acc1[i] = -acc1[i]
    opA = chain(thr, acc1, 16, 8)                                        # C -> A, no smem
    acc2 = cute.make_rmem_tensor(thr.partition_shape_C((16, 8)), F32); acc2.fill(0.0)
    cute.gemm(mma8, acc2, opA, frB2, acc2)                               # Z = P·T11
    c16 = cute.make_fragment_like(acc2, F16)
    for i in cutlass.range_constexpr(cute.size(acc2)):
        c16[i] = F16(acc2[i])
    cute.copy(tcC8, thrC.retile(c16), thrC.partition_D(O8))              # 2 x STSM.x1 (alias)
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 3

    # ---- seg 4: h=16, warps 0/1 one pair each; same shape as seg 3, with mma16 / tcA16 / tcB16 / tcCst,
    #      block read views sF16 / sFt16, acc shape (16,16), chain(thr, acc1, 16, 16). Omitted.
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 4

    # ---- seg 5-6: h=32, 2x2 warp grid + split-K reduction (mma_sync_subtile_broadcast_splitk §2)
    x, y = wid // 2, wid % 2                          # x: N half of GEMM1 == K half of GEMM2; y: M half of the output
    A1 = sF16x32[(None, None, 2 + y, 1)]              # (16, K=32)  rows 16y.. of T22
    Oy = sF16x32[(None, None, 2 + y, 0)]              # (16, 32)    output strip, overwrites A21
    B1 = sFt16x32[(None, None, x, 1)]                 # (N=16, K=32) columns 16x.. of A21^T
    B2 = sFt32x16[(None, None, 0, x)]                 # (N=32, K=16) K half x of T11^T
    # ...loads, GEMM1 (16,16,32), negate, chain(thr, acc1, 16, 16), GEMM2 (16,32,16), narrow into c16 as above
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 5: all warps have finished reading A21 / T11
    thrS = tcCst.get_slice(lane); dst = thrS.partition_D(Oy); src = thrS.retile(c16)
    if x == 0:
        cute.copy(tcCst, src, dst)
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 6: x=0's partial sum visible
    if x == 1:
        thrL = tcCld.get_slice(lane); red = cute.make_fragment_like(c16)
        cute.copy(tcCld, thrL.partition_S(Oy), thrL.retile(red))
        for i in cutlass.range_constexpr(cute.size(c16)):
            c16[i] = c16[i] + red[i]
        cute.copy(tcCst, src, dst)
    cute.arch.barrier(barrier_id=BAR, number_of_threads=NT)              # 7

    # ---- seg 7: exit. LDSM read back in C-fragment order -> f32 elementwise -> bf16 -> STSM back into the same bytes
    thrX = tcX.get_slice(tidx)
    rT = cute.make_rmem_tensor(thr_wg.partition_shape_C((FACE, FACE)), F16)
    cute.copy(tcX, thrX.partition_S(sF), thrX.retile(rT))
    for i in cutlass.range_constexpr(cute.size(cS)):
        acc[i] = F32(rT[i]) * gD[(0, cS[i][1])]                          # e.g.: per-column scale
    thr_b = r2sbf.get_slice(tidx); tb = r2sbf.retile(acc)
    b32 = cute.make_rmem_tensor(stg.shape, F32); b16 = cute.make_rmem_tensor(stg.shape, BF16)
    for v in cutlass.range_constexpr(cute.size(b32)):
        b32[v] = tb[v]
    b16.store(b32.load().to(BF16))
    cute.copy(r2sbf, b16, thr_b.partition_D(sFbf))
    cute.arch.fence_view_async_shared()               # then hand to a descriptor / another warpgroup: fence + rendezvous
```

## Gotchas

1. **Do not build sub-blocks of a swizzled face from `flat_divide` + `cute.select` + `iterator.align(16)`.**
   `Pointer.align()` is "address to integer → align → `make_ptr`", and the new pointer has **no** swizzle attribute; a
   B operand view cut this way on a `K_SW128` face reads unswizzled addresses, a silent numerical error.
   This entry's approach is one **flat layout** per operand role + `MemRange.get_tensor(layout, swizzle=t_layout.inner, dtype=...)`:
   the swizzle stays on the pointer, the block coordinates become trailing modes of the layout, and a block is taken
   at run time with `view[(None, None, rb, cb)]`. The stride-0 broadcast (h=8) is also written directly into the
   layout's `((8, 2), ...)` mode, and `partition_A` / `ldmatrix` / `partition_D` / `stmatrix` **copy the whole
   partition as usual with `cute.copy`** on it
   (gotcha 2 of [`mma_sync_subtile_broadcast_splitk`](mma_sync_subtile_broadcast_splitk.md)).
2. **The `Align` of the `MemRange` must be ≥ the swizzle period (1024 B for `K_SW128`).** The swizzle acts on address
   bits; if the allocation start is not aligned, two read views of the same layout each see a different bit pattern.
3. **fp16 inside, bf16 in and out are two `get_tensor(dtype=...)` calls, not two buffers.** A `dtype` of the same width
   (16-bit) only changes the element interpretation; bytes, swizzle and coordinates are unchanged, so the exit
   `stmatrix` can land back on the entry's bytes, and the downstream wgmma descriptor / `ldmatrix` gets bf16.
   But this **only holds at the same width**: an f32 read view halves the coordinate map, so open a separate layout.
4. **None of the three in-place-overwrite rendezvous can be dropped.** (i) At the start of each level: the corner
   blocks written by the previous level's `stmatrix` are this level's T22 / A21^T; (ii) before writing the output
   strip at h=32: the A21 the strip overwrites is exactly the B of every warp's GEMM1 at this level — all 4 warps must
   finish loading before anyone writes; (iii) between x=0 finishing its write and x=1 reading, adding and writing.
   At the h=8 / h=16 levels the blocks of the different warps do not intersect, so no rendezvous is needed within a
   level. 7 in total; merging further (keeping the merge in registers) is the next lever for this path, outside the
   scope of this entry's certification.
5. **`x == 0` / `x == 1` are warp-uniform runtime branches, not `const_expr`.** Written as `const_expr(x == 0)`,
   trace time reports a dynamic condition; a plain `if` is compiled by ptxas into a uniform branch. Same for
   `if wid < 2:`.
6. **fp16 unpacking cannot copy bf16's `shl/and`.** bf16 is the high half-word of f32, so two integer instructions
   suffice; fp16 is not — use `mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, lo;`
   (three lines of PTX, `dsl_user_op` + `llvm.inline_asm`, form 3 of [`ptx_escape`](ptx_escape.md)). The resident
   row words of the diagonal segment are fp16 pairs, the broadcast sends the original words, and the `cvt` is off the
   chain.
7. **Mixing `m16n8k8` and `m16n8k16` changes the fragment grouping.** The h=8 level uses `(16, 8, 8)`: A has K=8, and
   the acc→A hookup has `ratio = 1`; h=16 / h=32 use `(16, 8, 16)`: `ratio = 2`. `chain()` computes it as
   `cute.size(opA, mode=[0]) // cute.size(acc1, mode=[0])`, the same code for both levels; a hand-written `ratio` goes
   wrong when switching levels.
8. **The acc→A hookup only holds when "GEMM2's M = GEMM1's M and GEMM2's K = GEMM1's N".** Block doubling satisfies
   it at every level (`Z = P · T11`, P is h×h and T11 is h×h); the split-K at h=32 splits GEMM1's N into two halves,
   each half 16 = GEMM2's K half, so it still holds.
9. **Numerical acceptance must use an f64 inverse as the reference with a tolerance set by one fp16 rounding; it
   cannot require `max_diff == 0`.** This path has three fp16 roundings inside (writing back the diagonal blocks,
   narrowing P and Z at each level, adding the two halves in fp16 at h=32). On a 64×64 with |A| ≤ 0.2, the per-element
   |emulated − exact| is 2–4×10⁻⁴ (≈ 0.2–0.4 ulp16 @ |T|≈1); any wrong orientation / missing level / missing
   reduction / wrongly transposed diagonal block is ≥ 0.1, two orders of magnitude apart. The probe uses 2.5×10⁻³.
10. **Only 2 warps work in the diagonal segment, the other 2 wait at the barrier** — this is right: 8 blocks of 8×8 =
    64 lanes. Using all 4 warps (2 blocks per warp) does not shorten the per-lane chain depth; it only doubles the SHFL
    issues.
11. **After the exit segment, before handing to an async proxy (wgmma descriptor / TMA), `fence_view_async_shared()` +
    a rendezvous are needed**; handing to `ldmatrix` in another warpgroup of the same CTA needs only the rendezvous.
    `stmatrix` goes through the generic proxy. The rendezvous is needed only **when there is an exit segment**
    (read-modify-write); without the exit segment, a per-thread fence + mbarrier full arrive (count = all producing
    threads) already carries the same ordering, see gotcha 14.
12. **The census must hard-code lower bounds by static count and an upper bound of 0 on the narrow forms of
    `st.shared`.** Machine-readable evidence that the path was built: both `ldmatrix` members present, `stmatrix` ≥ 12
    (static 16), `mma.sync.aligned.m16n8k8` ≥ 2, `m16n8k16` ≥ 10 (static 12), named `bar.sync` ≥ 7,
    `shfl.sync.idx` ≥ 16, non-vector `st.shared` = 0, `.local` = 0
    (runtime indices into scalar tables open a local frame, see the gotcha in
    [`tri_forward_subst_shfl`](tri_forward_subst_shfl.md)).

13. **When landing on a multi-stage ring slot, all nine read views must carry the same trailing stage mode, and
    validation must be "solve in one stage, the other stage bit-for-bit unchanged".**
    The slot is `tile_to_shape(atom, (64, 64, STAGE), order=(0, 1, 2))` rather than a bare face: element coordinates
    `(r, c, s) ↦ r·64 + c + s·4096`; each role layout gets one more trailing mode of stride 4096 after its two operand
    modes and two block-coordinate modes, and the stage index is passed in as a runtime trailing coordinate.
    Two silent failure points (wrong only gives wrong numbers, no crash): the stride assumption of the trailing mode,
    and the C-fragment mapping of the entry / exit tiled copies — take it from the kernel's real (bf16/bf16) solver mma
    rather than from an f16/f16 trivial mma (the C layout does not change with the A/B dtype, but measure it).
    Validation form (landing probe, `PROBE PASS`, all five in
    the same round): run seg 1–6 on stage 1 and read back as fp16, with stage 0's sentinel bit-for-bit unchanged;
    then run all 7 segments on stage 0 and read back as bf16, with stage 1's sentinel bit-for-bit unchanged; the census
    as in gotcha 12, where narrow `st.shared` = 0 and `.local` = 0 are the two items really worth watching here — the
    replaced scalar solve was exactly the source of the spill and the smem wavefronts. **Set the tolerance relative to
    |T|max, not as an absolute value**: under a dense |A| ≤ 0.5 envelope the inverse reaches |T| ≈ 5, the h = 32 level
    builds the lower half-rows from the whole block, and the magnitude of fp16 rounding follows |T|max rather than the
    landing element (measured R1 max_diff 0.0125921 within 2.5×10⁻³ × |T|max, while any orientation / linkage / stride
    error is ≥ 0.1). The probe's prologue must also write smem 128-bit wide, otherwise the census's narrow `st.shared`
    criterion measures the prologue rather than the path. Landing reading: replacing the first-green form "whole-block
    64×64 f32 scalar forward substitution, each thread holding a 64-deep column" was worth −92.90%, and the 16
    registers the solver role borrowed from the consumer went back to the table tier; the flip of the dossier's `rings`
    axis is codegen moving along (gotcha 11 of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md)). Load-bearing
    condition: this segment is serial per chunk and sits on the critical path of a middle role in front of two consumer
    warpgroups.
14. **The exit segment (seg 7) is not a necessary part of the path: when downstream can consume the fp16 face directly
    and fold the per-column scale into its own operand formation, delete the read-back-scale-narrow-write-back and the
    rendezvous before it, and the last level's `stmatrix` is the delivery.** Delivery spelling: each thread does
    `cute.arch.fence_view_async_shared()` then `mbarrier_arrive(full)`, arrive count = all 128 producing threads — once
    the consumer passes the full wait it sees every warp's stores (every arrive comes after its own fenced stores),
    stronger than the deleted rendezvous, so no replacement is needed; one fewer inter-segment rendezvous per chunk.
    Downstream read: a second tiled mma on the same warpgroup, differing from the bf16 one only in operand dtype (the
    fp16 form of `make_trivial_tiled_mma`, SASS `HGMMA.64x64x16.F32.F16`), with the B descriptor built on the slot's
    fp16 `get_tensor` read view, **for both ring stages** (the index `i % STAGE` alternates between the two stages; a
    probe that only validates stage 0 is blind to odd chunks); on the A side f32 → fp16 pair packing goes through
    inline `cvt.rn.f16x2.f32 $0, $2, $1` (the DSL only exports `cvt_f32x2_bf16x2`, see the gotcha in
    [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md)). Correctness of folding the scale comes down to two
    things: the scale is multiplied in exactly once (moved from one side of the face to the other side of the operand),
    and the fp16 value range is the only new risk — check with an f64 reference + a finiteness check (operand-path
    probe: Y1 packing + fp16 RS wgmma on the shipped slot
    `max_diff 0.0`, Y2 the scale-folding algebra against the f64 reference max_diff 0.046 at |ref|max ≈ 10 and all
    finite, Y3 the same issue on ring stage 1). Reading: worth −4.07%, with the two knobs (deleting the exit segment,
    switching the downstream chain to fp16 + folding the scale) in one inseparable package; the related empty count went
    128 → 384 as "the second consumer warpgroup also started waiting on this table ring"; the flip of the `rings` axis
    is codegen moving along.

## Sources

- PTX ISA, warp-level `mma` (the per-lane distribution of the A/B/C fragments of `m16n8k8` / `m16n8k16`):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
- PTX ISA, `ldmatrix` / `stmatrix` (addressing of one 16 B row per lane; why the stride-0 read view does not go out of
  bounds):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-store-instruction-stmatrix>
- PTX ISA, `shfl.sync`, the vector pack / unpack forms of `mov`, `cvt`, named barriers with `bar.sync`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-mov-2>
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
- CuTe DSL public API symbols (cutlass-dsl 4.6.1): `cute.struct.MemRange.get_tensor(layout, swizzle=, dtype=)`,
  `cute.struct.Align`, `warpgroup.make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, ·)`, `cute.tile_to_shape`,
  `cutlass.utils.hopper_helpers.make_trivial_tiled_mma` / `get_smem_store_op`, `cute.make_tiled_copy_S` / `make_tiled_copy_C_atom` /
  `make_tiled_copy_A/B/C`, `cutlass.cute.nvgpu.warp.MmaF16BF16Op` / `LdMatrix8x8x16bOp` / `StMatrix8x8x16bOp`,
  `cute.make_tiled_mma` / `make_copy_atom` / `make_layout` / `make_rmem_tensor` / `make_fragment_like` / `recast_tensor` /
  `flat_divide` / `autovec_copy` / `gemm` / `retile` / `partition_A/B/C/S/D` / `partition_shape_A/C`,
  `cute.arch.lane_idx` / `warp_idx` / `shuffle_sync` / `barrier(barrier_id, number_of_threads)` / `fence_view_async_shared`,
  `cutlass.cutlass_dsl.dsl_user_op` + `llvm.inline_asm`.
- This repo's GPU-certified probe — R1 (read back as fp16 after seg 1–6,
  per element against f64 `inv(I + A)`, diagonal exactly 1, upper triangle exactly 0), R2 (read back as bf16 after
  seg 1–7, against `inv(I + A)·diag(d)` within half a bf16 ulp + path slack),
  C1 (PTX census: both `ldmatrix` members, `stmatrix` ≥ 12, `m16n8k8` ≥ 2, `m16n8k16` ≥ 10, named `bar.sync` ≥ 7,
  `shfl.sync.idx` ≥ 16, narrow `st.shared` = 0, `.local` = 0). Each rung in its own subprocess with its own
  `cute.compile`.
  **Status: ✅ Verified** — `PROBE PASS max_diff=0.00126922` (certification round 1, all three contracts R1/R2/C1 passed).
- The two contract-axis KEEPs of landing in the main loop (gotchas 13 / 14; eval job numbers not entered in the status
  record): landing probe
  `PROBE PASS` (R1 max_diff=0.0125921 / R1b, R2b bit for bit / R2 bf16 /
  C1 census) and operand-path probe `PROBE PASS` (Y1 0.0 / Y2 0.046 / Y3).
- Related entries: [`tri_forward_subst_shfl`](tri_forward_subst_shfl.md), [`warp_mma_f16`](warp_mma_f16.md),
  [`mma_sync_subtile_broadcast_splitk`](mma_sync_subtile_broadcast_splitk.md), [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md),
  [`wgmma_cfrag_publish_stsm`](wgmma_cfrag_publish_stsm.md), [`stsm_transpose_exchange`](stsm_transpose_exchange.md),
  [`ptx_escape`](ptx_escape.md).
