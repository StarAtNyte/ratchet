# ldmatrix_stmatrix (LDSM read side: move an smem face into registers in a tiled MMA's fragment order with one instruction; STSM writes back the C fragment of a warp-level MMA)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Division of labour: [`warp_mma_f16`](warp_mma_f16.md) §3 already teaches `ldmatrix` feeding the A/B of a
> **warp-level** MMA (both faces K-contiguous, both non-transposed members);
> [`stsm_transpose_exchange`](stsm_transpose_exchange.md) already teaches the `stmatrix` member selection for
> **WGMMA C fragment → smem**. This entry adds the remaining three pieces: **(a)** the LDSM read side smem face →
> **WGMMA C-fragment order** (both members); **(b)** the **transposed LDSM member** when the B operand face of a
> warp-level MMA is **N-contiguous**, plus structural masks (triangular etc.) done **in registers** by the
> coordinate map rather than as a predicated scalar gather; **(c)** the warp-level MMA's fp32 C fragment → f16 →
> `stmatrix` write-back. Together, a 16-bit smem face is handled from write to read with matrix instructions only,
> without a single scalar `LDS`/`STS`.

## Purpose

`ldmatrix` / `stmatrix` are warp-collective instructions: one instruction moves 1/2/4 8×8 16-bit matrices for
the whole warp, each addressing lane supplies the address of one row (8 **contiguous** elements = 16 B), and the
hardware scatters these 8×8 elements into the registers of the 32 lanes **according to the MMA fragment's lane
distribution** (or gathers them back). It is therefore the only path between "an smem face ↔ some tiled MMA's
fragment order" that needs no per-element gather:

- **Read side feeding C-fragment order**: when a face must **align element by element** with some MMA's
  accumulator (subtraction, multiplying by coefficients, serving as the next step's A operand), use
  `make_tiled_copy_C(LdMatrix8x8x16bOp(·, 4), tiled_mma)` to read the face into a register tensor of shape
  `partition_shape_C(...)`; index `i` refers to the same `(m, n)` as the accumulator's `i`. This applies both to
  WGMMA's C fragment (64×N atom, 128 or 256 threads) and to the C fragment of warp-level `mma.sync`.
- **Read side feeding A/B fragments**: `make_tiled_copy_A/B(LdMatrix8x8x16bOp(·, n), tiled_mma)`; the face's
  contiguous mode decides which member to use (see the table below) — it is **not** limited to K-contiguous.
- **Write side collecting the C fragment**: `make_tiled_copy_C(StMatrix8x8x16bOp(·, n), tiled_mma)` writes the C
  fragment of a warp-level MMA (narrowed to 16-bit first) back to the face with one instruction.

The cost of not using it is concrete: if the same face is fetched per element as `face[(m, k)]` by the
coordinates of `partition_X(identity)`, each element takes 4–6 integer address instructions + one 16-bit `LDS`
(a sub-word access; several lanes in the same wavefront land on different half-words of the same bank →
**bank conflicts several times those of the matrix instructions**), the fragment's live range is stretched by the
whole gather loop → **register cliff → spill**; the instruction count is about 3× that of the matrix path. The
values are still all correct, so the correctness gate cannot see it.

## When needed (trigger)

- The exchange-surface ledger says `LDSM`/`LDSM.T`/`STSM` but the implementation reads/writes per element as
  `sX[(m, n)]` (even with an `if m >= n` structural mask);
- NCU: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}` **5–12×** the reference, while
  `smem_wavefronts` is only 1.0–1.3× higher (small volume, concentrated conflicts = the fingerprint of sub-word
  scalar accesses); at the same time `local_op_ld/st` is non-zero with **ld ≫ st** (in one observed run a set of
  loop invariants was stored once and reloaded every block = the live-range cliff punched through by the gather
  loop; what the trigger looks at is the order-of-magnitude relation `ld/st` far above 1, with the ratio ≈ the
  loop trip count, not some fixed multiplier — the exact value varies with the trip count);
- SASS: in the ledger of that warp group, `LDS`/`STS` (non-`.128`) and `IMAD`/`LEA`/`ISETP` are at the top, while
  `LDSM`/`STSM` are absent or far fewer than the ledger's count;
- Build time: `make_tiled_copy_C_atom(LdMatrix…)` builds, `retile` reports a shape mismatch, and so you "fall
  back to per element" — that is the wrong constructor (gotcha 1).

## Mechanism: the member is decided only by "which mode of the face is contiguous"

Each addressing lane of `ldmatrix`/`stmatrix` hands over **one row of 8 contiguous 16-bit elements**. The
non-transposed member assigns matrix **rows** to the fragment's "row" lane groups, the transposed member assigns
matrix **columns** to them. Against the fragment table of `mma.m16n8k16` (each lane takes 2 adjacent elements
along **K**; in the C fragment each lane takes 2 adjacent elements along **N**), this gives the unique
correspondence table:

| fragment | logical shape | **contiguous mode** on the face | member | PTX | SASS |
|---|---|---|---|---|---|
| A | (M, K) | K | `LdMatrix8x8x16bOp(False, n)` | `ldmatrix…x{n}` | `LDSM.16.M88.{n}` |
| A | (M, K) | M | `LdMatrix8x8x16bOp(True, n)` | `ldmatrix…x{n}.trans` | `LDSM.16.MT88.{n}` |
| B | (N, K) | K | `LdMatrix8x8x16bOp(False, n)` | no `.trans` | `LDSM.16.M88.{n}` |
| B | (N, K) | **N** | `LdMatrix8x8x16bOp(True, n)` | `.trans` | `LDSM.16.MT88.{n}` |
| C (read) | (M, N) | N | `LdMatrix8x8x16bOp(False, 4)` | no `.trans` | `LDSM.16.M88.4` |
| C (read) | (M, N) | **M** | `LdMatrix8x8x16bOp(True, 4)` | `.trans` | `LDSM.16.MT88.4` |
| C (write) | (M, N) | N | `StMatrix8x8x16bOp(False, n)` | no `.trans` | `STSM.16.M88.{n}` |
| C (write) | (M, N) | M | `StMatrix8x8x16bOp(True, n)` | `.trans` | `STSM.16.MT88.{n}` |

Three invariants:

1. **The member does not change logical values** — the same `(m, n)` reads back the same thing under both
   members; what changes is only the direction from which the hardware fetches the 16 B row. The wrong member =
   **transposed** data, compiles fine, silently wrong (gotcha 2).
2. **The "contiguous mode" is a physical property of the face**; the mode relabelling done by `cute.select` /
   `cute.composition` only renames coordinates and does not change the contiguous mode. So look up the table by
   the layout **at allocation time**, not by the mode order of the view you happen to hold.
3. **`num_matrices` is computed from the tiler's element count**: the tiler of `make_tiled_copy_{A,B,C}` is
   given by the tiled MMA (A: `(tile_M, tile_K)`, B: `(tile_N, tile_K)`, C: `(tile_M, tile_N)`), element count ÷
   64 = 1/2/4. WGMMA's C tile is 64×N (N ≥ 8) → always 4; for `m16n8k16`, A `(16,16)`→4, B `(8,16)`→2, C
   `(16,8)`→2.

## Minimal runnable example

Every snippet below has actually been run by the GPU-certification probe (the probe is not included in this
release; verdicts in "Sources"). All values on the faces are asymmetric small integers (`((r + 3c) % 7) − 3`), so
"transposed wrong" cannot happen to equal "right".

### 1. WGMMA C-fragment order ← face (both members)

The C fragment's M mode is contiguous on the face → transposed member. Typical shape: the face is written by TMA
as `(t, d)` with `d` contiguous and `K_SW128`, and the consumer wants `(d, t)` = the C fragment's `(m, n)`. **A
second `get_tensor` on the same `MemRange`** gives the read view, with the swizzle carried by the pointer:

```python
import cutlass, cutlass.cute as cute, cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warp, warpgroup
from typing import Callable                                # for the SmemT / Callable definitions
BF16, F16, F32 = cutlass.BFloat16, cutlass.Float16, cutlass.Float32
MJ_K = cute.nvgpu.OperandMajorMode.K
BLK, D = 64, 128

@cute.struct
class SmemT:                                   # smem struct used by the kernel: one (BLK, D) 16-bit face
    f: cute.struct.Align[cute.struct.MemRange[BF16, BLK * D], 1024]

@cute.jit
def launch(gV, gO):
    # two warpgroups stacked along M: C tile = (128, 64), 256 threads. Only its C-fragment map is borrowed; no MMA is issued.
    mma = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, MJ_K, MJ_K, F32, atom_layout_mnk=(2, 1, 1), tiler_mn=(64, 64))
    atom = warpgroup.make_smem_layout_atom(warpgroup.SmemLayoutAtomKind.K_SW128, BF16)
    v_layout  = cute.tile_to_shape(atom, (BLK, D), order=(0, 1))     # write view (t, d), d contiguous
    vt_layout = cute.select(cute.coalesce(v_layout, target_profile=(1, 1)), [1, 0])
    #                                                                 # read view (d, t): pure relabel
    # C fragment (m=d, n=t); what is contiguous on the face is d = M -> transposed member, always 4 matrices
    tc = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BF16), mma)
    kernel(gV, gO, mma, tc, v_layout, vt_layout, SmemT).launch(grid=[1,1,1], block=[256,1,1])

@cute.kernel
def kernel(gV, gO, mma, tc, v_layout, vt_layout, SS: cutlass.Constexpr[Callable]):
    tidx, _, _ = cute.arch.thread_idx()
    st  = cutlass.utils.SmemAllocator().allocate(SS)
    sV  = st.f.get_tensor(v_layout.outer,  swizzle=v_layout.inner)    # for the write side
    sVt = st.f.get_tensor(vt_layout.outer, swizzle=vt_layout.inner)   # for the read side, the same bytes
    ...                                    # fill sV; fence_view_async_shared + sync_threads
    thr = mma.get_slice(tidx)              # WGMMA: warpgroup thread index, not lane
    rV  = cute.make_rmem_tensor(thr.partition_shape_C((D, BLK)), BF16)
    thr_c = tc.get_slice(tidx)
    cute.copy(tc, thr_c.partition_S(sVt), thr_c.retile(rV))          # 4 LDSM.16.MT88.4
    # rV[i] refers to the same (m, n) as acc[i] of any C accumulator built with the same mma:
    cC = thr.partition_C(cute.make_identity_tensor((D, BLK)))
    for i in cutlass.range_constexpr(cute.size(rV)):
        m, n = cC[i]; gO[(m, n)] = F32(rV[i])
```

The C fragment's N mode is contiguous on the face → non-transposed member. Typical shape: a 16-bit face **tiled
from 8×8 tiles, no swizzle** (`tile_to_shape(make_layout((8, 8), stride=(8, 1)), (64, 64), order=(0, 1))`),
previously written by `STSM.16.M88.4` and, after some in-place transformation, to be read back in the same
C-fragment order:

```python
    mma = sm90_utils.make_trivial_tiled_mma(
        F16, F16, MJ_K, MJ_K, F32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, 64))   # 128 threads
    f_layout = cute.tile_to_shape(cute.make_layout((8, 8), stride=(8, 1)), (64, 64), order=(0, 1))
    tc = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), F16), mma)
    # inside the kernel:
    sF  = st.f.get_tensor(f_layout)                                    # no swizzle, plain get_tensor
    rF  = cute.make_rmem_tensor(thr.partition_shape_C((64, 64)), F16)
    cute.copy(tc, tc.get_slice(tidx).partition_S(sF), tc.get_slice(tidx).retile(rF))  # 4 LDSM.16.M88.4
```

After reading back, multiplying by coefficients element by element, subtracting another C fragment, or applying
a structural mask by the coordinate map `cC[i]` are all done on `rF[i]`; writing back again plugs into the r2s
three-layer construction of [`stsm_transpose_exchange`](stsm_transpose_exchange.md). The linear order of `i` is
**the same** across the read-back tensor, the accumulator and `partition_C(identity)`, which is the precondition
of every elementwise operation in this entry (gotcha 3).

### 2. Warp-level MMA: LDSM (both members) → structural mask in registers → `mma.sync` → STSM write-back

Each of 4 warps takes its own 16×16 sub-block of a 64×64 face, computes `tril(A) @ tril(G)`, and writes back to
another face. The A sub-block `(M, K)` is K-contiguous → non-transposed; the G sub-block is `(K, N)` with N
contiguous, **relabelled as `(N, K)`** to serve as B — the contiguous mode is still N → **transposed member**.
The mask zeroes elements by the coordinate map after they are loaded into the fragment, not by predicated reads on
the smem side:

```python
@cute.jit
def launch(gF, gO):
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 16)), (1, 1, 1))
    f_layout = cute.tile_to_shape(cute.make_layout((8, 8), stride=(8, 1)), (64, 64), order=(0, 1))
    def n_mat(rows, cols):                      # one 8×8×16b matrix = 64 elements
        n = (rows * cols) // 64; assert n in (1, 2, 4); return n
    tM, tN, tK = mma.get_tile_size(0), mma.get_tile_size(1), mma.get_tile_size(2)   # 16, 8, 16
    tcA = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, n_mat(tM, tK)), F16), mma)
    tcB = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True,  n_mat(tN, tK)), F16), mma)
    tcC = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, n_mat(tM, tN)), F16), mma)
    ...

@cute.kernel
def kernel(gF, gO, mma, tcA, tcB, tcC, f_layout, SS):
    ...                                          # sF / sO = two faces with the same layout; fill sF, fence + sync
    lane = cute.arch.lane_idx()                  # warp-level MMA: lane, not tidx (warp_mma_f16 gotcha 4)
    wid  = cute.arch.warp_idx()
    f4 = cute.flat_divide(sF, (16, 16))          # (16, 16, 4, 4): sub-block coordinates may be dynamic
    blkA = f4[(None, None, wid, wid)]            # A = (M, K), K contiguous
    blkG = f4[(None, None, wid, (wid + 1) % 4)]  # G = (K, N), N contiguous
    # B = G^T written as (N, K): a pure mode swap; the contiguous mode is still N => transposed member. align(16) gives
    # back to the verifier the 16 B alignment fact the sub-block slice dropped (each ldmatrix row address must be 16 B-aligned).
    Bt = cute.make_tensor(blkG.iterator.align(16), cute.select(blkG.layout, [1, 0]))
    blkO = cute.flat_divide(sO, (16, 16))[(None, None, wid, wid)]

    thr = mma.get_slice(lane)
    frA = thr.make_fragment_A(thr.partition_A(blkA))        # destination tensors built with the MMA's own profile
    frB = thr.make_fragment_B(thr.partition_B(Bt))
    cute.copy(tcA, tcA.get_slice(lane).partition_S(blkA), tcA.get_slice(lane).retile(frA))  # LDSM   ×1 (x4)
    cute.copy(tcB, tcB.get_slice(lane).partition_S(Bt),   tcB.get_slice(lane).retile(frB))  # LDSM.T ×2 (x2)

    # Structural mask: the identity partition has the same linear order as the fragment, frX[i] is exactly the cell cX[i]
    cA = thr.partition_A(cute.make_identity_tensor((16, 16)))
    cB = thr.partition_B(cute.make_identity_tensor((16, 16)))
    for i in cutlass.range_constexpr(cute.size(cA)):
        m, k = cA[i]
        if m < k: frA[i] = F16(0.0)              # A takes the lower triangle
    for i in cutlass.range_constexpr(cute.size(cB)):
        n, k = cB[i]
        if n > k: frB[i] = F16(0.0)              # G = B^T takes the lower triangle

    acc = cute.make_rmem_tensor(thr.partition_shape_C((16, 16)), F32)
    acc.fill(0.0)
    cute.gemm(mma, acc, frA, frB, acc)

    c16 = cute.make_fragment_like(acc, F16)      # same profile, different dtype
    for i in cutlass.range_constexpr(cute.size(acc)):
        c16[i] = F16(acc[i])
    cute.copy(tcC, tcC.get_slice(lane).retile(c16), tcC.get_slice(lane).partition_D(blkO))   # STSM ×2
```

The dynamic sub-block coordinates of `f4[(None, None, wid, wid)]` and the offset block selection
`(wid + 1) % 4` go straight into `partition_S` — LDSM addresses are computed per lane at runtime, so sub-block
positions need not be compile-time constants.

### 3. Machine-read acceptance: PTX census (values are blind to the member; this is mandatory)

```python
compiled = cute.compile(jit, *args, options="--dump-dir=%s --keep-ptx --keep-cubin" % d)
ptx = "".join(open(f).read() for f in glob.glob(os.path.join(d, "**", "*.ptx"), recursive=True))
ld  = re.findall(r"ldmatrix[^;\n]*", ptx);  ld_t = [x for x in ld if ".trans" in x]
st  = re.findall(r"stmatrix[^;\n]*", ptx);  st_t = [x for x in st if ".trans" in x]
# transposed member: len(ld_t) == len(ld) > 0; non-transposed: len(ld) > 0 and not ld_t;
# likewise for the write side. SASS-side names: LDSM.16.M88.{n} / LDSM.16.MT88.{n} / STSM.16.M88.{n} / STSM.16.MT88.{n}
```

### 4. Read-side idioms for K_SW128 face → C fragment: one atom for all read points, the row map, derived faces (`LDSM → FMUL → F2FP → STSM`)

§1 gives the member selection; this section gives three companion facts for laying "face → C-fragment order" over
**all non-accumulator operands of an elementwise segment**, all judged by a second GPU-certification probe
on the layout set below (four numeric contracts R1–R4 `PROBE PASS max_diff=0.0`, C1
PTX census with `.trans` 0 for both `ldmatrix`/`stmatrix`). The face is a `K_SW128` 64×64 bf16 tile landed by TMA
(one block per slot, N contiguous), read into the 64×64 WGMMA C fragment of 128 threads:

```python
E = LayoutEnum.ROW_MAJOR
atom = warpgroup.make_smem_layout_atom(
    sm90_utils.get_smem_layout_atom(E, BF16, 64), BF16)                       # K_SW128
lyt_face = cute.tile_to_shape(atom, (64, 64, NSLOT), order=(0, 1, 2))         # one 64×64 block per slot
mma = sm90_utils.make_trivial_tiled_mma(BF16, BF16, MM.K, MM.MN, F32,
                                        atom_layout_mnk=(1, 1, 1), tiler_mn=(64, 64))   # 128 threads
# read side: N contiguous -> non-transposed member; whole C tile -> make_tiled_copy_C (gotcha 1); ONE atom for all read points, all roles
l2r = cute.make_tiled_copy_C(
    cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma)
# write side: the same r2s as used for publishing (three-layer construction in stsm_transpose_exchange)
r2s = cute.make_tiled_copy_S(
    sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
    cute.make_tiled_copy_C_atom(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), BF16), mma))
```

**(a) One helper serves all read points** — slots landed by TMA (R1) and slots written by STSM (R4) are read with
**the same** member; the bytes from both sources have the same contiguous mode on the face:

```python
tw    = tidx % 128                            # warpgroup thread index (gotcha 5); the same index for all three get_slice
thr_c = mma.get_slice(tw)
cS    = thr_c.partition_C(cute.make_identity_tensor((64, 64)))
thr_l = l2r.get_slice(tw)
cshp  = thr_c.partition_shape_C((64, 64))

def ldsm_c(sT, slot):
    r = cute.make_rmem_tensor(cshp, BF16)     # bf16 fragment; converted to f32 per element when used, no f32 copy materialised
    cute.copy(l2r, thr_l.partition_S(sT[(None, None, slot)]), thr_l.retile(r))
    return r                                   # r[i] and acc[i] are both the (m, n) of cS[i]
```

One `LDSM.16.M88.4` per warp per 16×16 sub-block (four for a 64×64 tile), replacing one sub-word `LDS` per element
plus a string of swizzle address arithmetic — i.e. the source of the NCU fingerprints in "When needed".

**(b) The row map of the C fragment: per-row scalar factors collapse to 2 values per thread** (R2; the host
checks thread by thread against the coordinate map the kernel reports back): in the 64×64 WGMMA C fragment of 128
threads, a thread's 32 elements fall on exactly **2 rows** and **16 distinct columns**, with the row selected by
bit 1 of the fragment index — elements with `(i // 2) % 2 == 0` are on row `cS[0][0]`, the rest on row `cS[2][0]`.
So any **per-row** scalar factor (scale, exponent, gate …) needs only 2 reads and 2 evaluations per thread per
tile, not 32 + 32:

```python
f0 = f(sRow[cS[0][0]]); f1 = f(sRow[cS[2][0]])          # 2 LDS + 2 evaluations
rX = ldsm_c(sT, slot)
for i in cutlass.range_constexpr(cute.size(cS)):
    if cutlass.const_expr((i // 2) % 2 == 0):
        acc[i] = g(acc[i], F32(rX[i]), f0)
    else:
        acc[i] = g(acc[i], F32(rX[i]), f1)
```

The `const_expr` branch is folded away at trace time; SASS has neither a select nor a second copy of the factor.

**(c) Derived faces: `LDSM → FMUL → F2FP → STSM`, a 16-bit face handled from read to write with matrix
instructions only** (R3 end to end): for derived operands such as "row factor × face", `dst[m, n] = f_m ·
src[m, n]`, don't do it per element as `sD[(m, n)] = f * sS[(m, n)]`:

```python
rx = ldsm_c(sS, src)
ry = cute.make_rmem_tensor(cshp, BF16)
for i in cutlass.range_constexpr(cute.size(rx)):
    if cutlass.const_expr((i // 2) % 2 == 0):
        ry[i] = BF16(F32(rx[i]) * f0)
    else:
        ry[i] = BF16(F32(rx[i]) * f1)
stg  = cute.make_layout(cute.shape(thr_r.partition_S(sD[(None, None, 0)]))[:3])   # the same one used for publishing
s16  = cute.make_rmem_tensor(stg.shape, BF16)
tacc = r2s.retile(ry)
for v in cutlass.range_constexpr(cute.size(s16)):
    s16[v] = tacc[v]
cute.copy(r2s, s16, thr_r.partition_D(sD[(None, None, dst)]))
cute.arch.fence_view_async_shared()            # when this face feeds a WGMMA descriptor (gotcha 9), then barrier
```

Per warp a fixed 4 `LDSM` + 32 `FMUL` + conversions + 4 `STSM`, replacing 64 scalar `LDS` + 64 scalar `STS` + 64
row-factor evaluations per thread. R3's criterion is bit-equality with `bf16(f_r · X[r, c])` (factors are powers of
2, X asymmetric small integers, so products are exact in bf16 and a transposition error cannot pass by chance); R4
then reads the face written by this STSM back via (a), closing the `STSM → LDSM` round trip.


## Gotchas

1. **To read a whole C fragment use `make_tiled_copy_C(atom, tiled_mma)`, not `make_tiled_copy_C_atom` +
   `make_tiled_copy_D`.** `make_tiled_copy_C_atom` is one half of the r2s **subtile** skeleton (it truncates the C
   TV by the atom's value count, used together with `make_tiled_copy_S(store_op, ·)`, see
   [`stsm_transpose_exchange`](stsm_transpose_exchange.md) gotcha 2); the tiler it gives is an atom-sized
   sub-block, not the whole C tile. Using it for `partition_S(whole face)` / `retile(whole rmem)` yields rest
   modes that don't match the MMA's C profile and reports a shape mismatch — **the error is raised at `retile`,
   the real cause is the constructor**. `make_tiled_copy_C` uses `tv_layout_C_tiled` + `(tile_M, tile_N)`
   directly and is in the same family as `make_tiled_copy_A/B`. (Certified: both snippets of §1 and the STSM of
   §2 go through `make_tiled_copy_C`.)
2. **Choosing the wrong member is silent.** The verifier only checks alignment and 16 B row contiguity; it **does
   not check whether the `transpose` boolean is right**: orientation right, boolean wrong, it compiles and moves
   transposed data. Run once with an asymmetric integer pattern (§1's `((r+3c)%7)−3`), or look up the table above
   by the contiguous mode **at allocation time** — the mode order of a view after `cute.select` / `composition`
   **does not count**.
3. **Masks / elementwise ops act on the loaded fragment, not on smem.** `partition_X(identity)[i]` has the same
   linear order as `frX[i]` and `acc[i]`, so a "triangular mask" = the single line `if m < k: frA[i] = 0`, costing
   one predicated `SEL` per element; writing `x = 0; if m >= k: x = sF[(m, k)]; frA[i] = x` degenerates the whole
   read into predicated sub-word `LDS` + address arithmetic + a stretched live range — exactly the NCU
   fingerprints of "When needed".
4. **Destination rmem tensors must be built with the MMA's own profile**: `make_rmem_tensor(partition_shape_C(...))`,
   `make_fragment_A(partition_A(...))`, `make_fragment_like(acc, F16)`; then convert to the copy's profile via
   `thr_copy.retile(·)`. The other way round (building rmem from the shape of `partition_S` and feeding it to the
   MMA) the MMA does not accept.
5. **The slice index of a WGMMA C fragment is the warpgroup thread index (`tidx` or `tidx − 128·wg`); for a
   warp-level MMA it is `lane_idx()`.** The two `get_slice` calls — the tiled MMA's and the tiled copy's — must be
   passed **the same** index.
6. **When relabelling after slicing a sub-block, add `iterator.align(16)`.** After `flat_divide` + slicing with
   dynamic coordinates, the pointer's static alignment information may be reduced to the element width; calling
   `ptr.align(16)` before `make_tensor(ptr, select(layout, [1, 0]))` hands the fact "each row is 16 B-aligned" to
   the verifier (zero cost when the address is aligned anyway).
7. **Before using `cute.select` to swap modes on a `ComposedLayout`, `coalesce(·, target_profile=(1, 1))` first.**
   The swizzle face from `tile_to_shape` has hierarchical modes, and a direct `select` may be rejected; after
   coalescing it is flat. The fallback is the flat colex transposer
   `composition(layout, make_layout((c, r), stride=(r, 1)))`
   ([`wgmma_mn_major_operand`](wgmma_mn_major_operand.md) §3). Both are only relabels (invariant 2).
8. **An unswizzled 8×8 tile face is conflict-free for both members**: one 8×8 tile = 128 B contiguous = each of
   32 banks once, and the 8 row addresses of one `ldmatrix` are exactly the 8 rows of one tile. So such faces **do
   not need**, and should not get, an additional swizzle; LDSM.T works on a `K_SW128` face because the swizzle
   only permutes 16 B chunks and does not scatter the 8 elements inside a chunk.
9. **The proxy discipline is unchanged**: `ldmatrix`/`stmatrix` are both generic proxy. TMA write → LDSM read: an
   mbarrier wait suffices; STSM write → WGMMA descriptor read: `fence_view_async_shared()` + barrier
   ([`stsm_transpose_exchange`](stsm_transpose_exchange.md) gotcha 7); STSM write → LDSM read **in the same
   warp**: warp-collective instructions carry their own ordering, no extra barrier needed; across warps a
   NamedBarrier is needed.
10. **Acceptance cannot look only at `max_diff`** (same as [`stsm_transpose_exchange`](stsm_transpose_exchange.md)
    gotcha 9). The scalar path's values are correct. Two machine-read criteria: §3's PTX census (`ldmatrix`/
    `stmatrix` counts and the `.trans` distribution), plus NCU's `bank_conflicts_…_shared_op_{ld,st}` and
    `local_op_{ld,st}` — when the matrix path is built, the former returns to the reference's order of magnitude
    and the latter drops to zero.
11. **A KEEP of this path has no footprint of its own on the dossier's contract axes — "three axes zeroing at
    once" is not a mechanism reading.** One KEEP that changed all non-accumulator operands of an elementwise
    segment (5 read points + 2 derived faces + row-factor hoisting, one commit) from scalar `LDS`/`STS` to the §4
    path read on the dossier (verbatim): `supply` `["addr-source loads hoisted; prefetched"]` → `[]`; `commit`'s
    seven `mma-consumer` rings (`{LE 0x0 x14}`, `{LE 0x0 x18}`, `{LE 0x0 x2}`×4, `{LE 0x0 x8}`) → `[]`;
    `cache_ops` `["UTMALDG.2D", "UTMASTG.2D"]` → `["UTMALDG.2D"]`; count axes such as `loops` all moved in the
    same round. Gate window 2.2002 → 1.2093 ms (−45.04%). Reading: the whole rows of `commit` / `supply`
    disappearing and flipping in step with `loops` is the region blindness of
    [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 11 (in a later round that changed something
    else, the seven rings came back with their original readings); the entry missing from `cache_ops` is a
    **store** signature, while this change did not touch a single store — so not one axis in this dossier is a
    footprint of the LDSM path. The machine-read footprint is still the two of gotcha 10: `LDSM`/`STSM` counts
    and the `.trans` distribution in PTX/SASS (probe C1), plus NCU's `bank_conflicts_…_shared_op_{ld,st}` and
    `local_op_{ld,st}`.
    **Load-bearing conditions**: −45% is the reading of five read points + two derived faces + row-factor hoisting
    landed **as one bundle**, on a tree where "every operand of the elementwise segment originally went through
    sub-word scalar `LDS` + swizzle address arithmetic, and each of the two derived faces through a 64-element
    scalar `LDS`+`STS` loop"; the three cannot be separated; it is the price of the NCU fingerprints of "When
    needed" (several-fold bank conflicts, `local_op_ld ≫ st`), not the price of one `LDSM`.
12. **Build per-tile coefficient fragments (structural mask × row/column factors) once in accumulator order and
    reuse them across several elementwise segments; when the excluded region may be non-finite, use a select,
    not a multiply.** When three segments each need the same `[j ≤ i] · h(row_i, col_j)`, compute it once by
    §4(b)'s row map as an f32 fragment of shape `partition_shape_C` (2 row factors + 16 column factors + 32
    evaluations per thread), leaving each of the three segments a single `FMUL`; a segment needing the strict half
    (`j < i`) takes it from the same fragment with one select. When `h` may overflow to inf in the excluded region,
    the mask must be a **select** (`mk[i] = h if cond else 0`), not `h * [cond]` — `inf * 0 = NaN`. The cost is
    32 resident f32 for this fragment between its formation point and the last segment using it, which must enter
    the segment-wise live-set accounting ([`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)).


13. **smem→RMEM A-fragment load: `CopyUniversalOp` ↔ `LdMatrix8x8x16bOp(False, 4)` only swaps the atom; the
    footprint is stable, the sign of the price flips with the tree.** The knob: the atom in
    `make_tiled_copy_A(<atom>, tiled_mma)` changes from `cute.nvgpu.CopyUniversalOp()` to
    `warp.LdMatrix8x8x16bOp(False, 4)`; tiling, source partition, the 32-register RMEM destination and the A
    fragment retile are all untouched, and the output is bit-identical before and after.
    - **Footprint (consistent across two KEEP readings in opposite directions)**: SASS 32 scalar `LDS` per thread
      per tile ↔ 8 `LDSM.x4`; the scalar form carries a static local round trip of 5 `LDL` + 5 `STL` per thread
      per tile (in the tile prologue, not in the steady-state body), NCU `spill_ld / spill_st` 163840 / 21120
      sectors ↔ 0 / 0; the `x2` member (16 `LDSM`) reads in between and was not adopted. The dossier's `rings`
      axis gains / loses one `stride=0x2` along with it — a follow-on, not a footprint
      ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 16).
    - **Load-bearing conditions**: on a tree whose steady-state body still had a visible gap every trip
      (`tensor_pct` ≈ 78), switching to `LDSM` gave paired timing −1.39%, gate 0.2183 → 0.21298 ms; on a later
      tree of the same kernel (`tensor_pct` ≈ 84, already compacted outside the tile prologue) the same atom pair's
      paired timing **flipped sign** — the scalar form −1.17% (faster in 14 of 15 rounds), and after switching
      back the gate went 0.20421 → 0.20477 ms (within the noise floor), with spill back from 0 to 163840 / 21120
      while the time did not move.
    - **Reading**: in a prologue load done once per tile, "removing 5 + 5 local round trips" is not the objective
      function; `LDSM.x4` is fewer, wider shared reads, scalar `LDS` is more, narrower ones — which side is cheaper
      depends on what the smem read side is competing with at that moment (not isolated `[INFERRED]`). Use this
      atom as **a dial re-priced on every tree**: same-job interleaved pairing + a byte-identical copy as the
      noise floor; don't carry the previous tree's sign over.

## Out of scope for this entry

- `LdMatrix8x16x8bOp` (the 8-bit unpacking member), hand-made `make_tiled_copy_tv` when `num_matrices` does not
  match the tiler.
- The **algorithms** such as triangular inversion themselves — this entry only gives the mechanical action
  "zero the fragment by the coordinate map".

## Sources

- PTX ISA, `ldmatrix`: <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>
- PTX ISA, `stmatrix`: <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-store-instruction-stmatrix>
- PTX ISA, per-lane A/B/C fragment distribution tables of `mma.m16n8k16` (the member table above is derived from
  them): <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
- CuTe DSL public API symbols (cutlass-dsl 4.6.1): `cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp` /
  `StMatrix8x8x16bOp` / `MmaF16BF16Op`, `cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind` /
  `make_smem_layout_atom`, `cutlass.utils.hopper_helpers.make_trivial_tiled_mma`,
  `cute.make_tiled_mma` / `make_copy_atom` / `make_tiled_copy_A` / `make_tiled_copy_B` / `make_tiled_copy_C` /
  `TiledMma.get_tile_size` / `get_slice` / `partition_A` / `partition_B` / `partition_C` / `partition_shape_C` /
  `make_fragment_A` / `make_fragment_B` / `make_rmem_tensor` / `make_fragment_like` / `make_identity_tensor` /
  `ThrCopy.partition_S` / `partition_D` / `retile` / `cute.copy` / `cute.gemm` / `cute.flat_divide` /
  `cute.select` / `cute.coalesce` / `cute.tile_to_shape` / `Pointer.align` / `cute.arch.lane_idx` /
  `cute.arch.warp_idx`.
- This repo's GPU-certification probe: six contracts — three numeric
  contracts judged by `max_diff == 0`: R1 (WGMMA C fragment ← K_SW128 face, transposed member, 256 threads), R2
  (WGMMA C fragment ← 8×8 tile face, non-transposed member), R3 (warp-level MMA: non-transposed LDSM feeding A,
  transposed LDSM feeding the N-contiguous B, triangular mask in registers, `mma.sync`, STSM write-back); and
  three PTX census contracts C1–C3 judging the `.trans` distribution (`max_diff=n/a`). **Status: ✅
  GPU-certified — R1–R3 `PROBE PASS max_diff=0.0`, C1–C3 `PROBE PASS max_diff=n/a`, all six passed.**
- **§4 + gotchas 11 / 12**: the second GPU-certification probe (a single warpgroup of 128 threads, `K_SW128` 64×64 bf16 face
  × 4 slots: R1 a TMA-landed slot read into the C fragment with the non-transposed member, R2 the row map
  `(i // 2) % 2` and the 16-column set, R3 the derived face `LDSM → 2-row-factor FMUL → F2FP → STSM`, R4 reading
  back the slot written by STSM, four `PROBE PASS max_diff=0.0`; C1 PTX census with no `.trans` on either
  `ldmatrix`/`stmatrix`, `PROBE PASS max_diff=n/a`). KEEP certification on the production cubin's contract axes:
  KEEP in an eval job (job number not recorded), axis `supply` `["addr-source loads hoisted; prefetched"]` →
  `[]`, `commit` seven rings → `[]`, `cache_ops` `["UTMALDG.2D", "UTMASTG.2D"]` → `["UTMALDG.2D"]` (none of them a
  footprint of this path, see gotcha 11), gate 2.2002 → 1.2093 ms.
- Gotcha 13 (footprint and sign flip of the A-fragment load atom): the dossier census and NCU of two KEEP
  evaluations (eval job numbers not recorded), axis `rings` gaining / losing `stride=0x2` (follow-on), gate
  0.2183 → 0.21298 ms and 0.20421 → 0.20477 ms; paired pricing probes
  (one with 9 interleaved rounds × 15 launches, each variant bit-identical to the baseline; the other with
  15 interleaved rounds × 15 launches, byte-identical-copy noise
  floor |med| 0.03%). Status: ✅ KEEP-certified.
