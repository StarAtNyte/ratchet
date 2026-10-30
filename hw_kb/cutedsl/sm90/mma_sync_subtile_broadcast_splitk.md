# mma_sync_subtile_broadcast_splitk (warp-level `mma.sync`: feed an operand with fewer rows than the M atom through a stride-0 broadcast view (the whole partition is copied as usual; the repeated slice is an alias); reduce a warp pair's split-K partial sums through smem with two barriers)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Division of labour: [`warp_mma_f16`](warp_mma_f16.md) already teaches the construction of warp-level
> MMA, the `ldmatrix` trio, and §4's **direct acc→A chaining**; [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md)
> already teaches member selection and reading/writing in C-fragment order. This entry adds two
> **operand-shaping moves** and combines them with direct acc→A chaining into a complete two-step GEMM
> path: **(a)** when the operand has only 8 M rows (fewer than the M=16 atom of `m16n8k*`), how to feed
> `partition_A` / `ldmatrix` / `partition_D` / `stmatrix` without going out of bounds and without building
> identity padding; **(b)** a warp pair splits K in half, each computing half of the partial sums, and
> reduces them into one output through smem with two named barriers, "write first / read-add-write".

## Purpose

The A and C of `mma.sync.m16n8k8 / m16n8k16` are **16-row** atoms. When your A operand has only h = 8
rows (or the output only needs 8 rows), the hardware has no "half atom" — but you also **do not need** to
build data for the extra 8 rows:

- **stride-0 broadcast view**: wrap the (8, K) smem block with `make_tensor(ptr, make_layout(((8, 2), K),
  stride=((s0, 0), s1)))` into a ((8, 2), K) = 16-row view in which rows 8..15 **alias** rows 0..7.
  The shapes that `partition_A` / `make_tiled_copy_A(...).partition_S` cut from it are exactly the same as
  for a real 16 rows, and `ldmatrix` addressing always stays inside the block (which is exactly why it can
  replace the dilemma of "a 16-row tile based at row 16p+8 running off the face"). The view moves no
  bytes; zero cost.
- **The repeated M slice is an alias; copy the whole partition as usual**: the copy atom takes **one** 8×8
  matrix (`num_matrices=1`), and on the 16-row tiler the tiled copy gets an extra atom repetition of
  extent = 2 (in the second sub-mode of the value mode: `partition_S` looks like `((8, 2), 1, 1)`,
  `retile` like `((2, 2), 1, 1)`). Do `cute.copy` over the **whole** partition: the second `LDSM.x1`
  re-reads rows 0..7 through the stride-0 view, so the upper and lower halves of the A fragment are
  identical ⇒ rows 8..15 of both accumulators are **bit-exact copies** of rows 0..7; write-back likewise
  does a whole-partition `cute.copy` through the broadcast view, and the second `STSM.x1` lands the same
  values on the same 8-row block again — order-independent, race-free, no mask needed. The cost is one
  extra `LDSM.x1` + one `STSM.x1` per level (128 B of aliased traffic each), negligible compared with the
  doubled K + smem round trip + extra rendezvous that identity padding brings. **Do not** try to integer-
  index away the repeated sub-mode to move only "slice 0" — see gotcha 2.
- **Warp-pair split-K + smem reduction**: when the K of one output strip (16 rows × N columns) is too long
  and two warps share it, each warp computes half of K to get a partial sum, and the reduction goes through
  smem: barrier → the x=0 warp writes its partial sum with `stmatrix` → barrier → the x=1 warp reads it back
  with **`ldmatrix` in C-fragment order** (`make_tiled_copy_C(LdMatrix…)`), adds elementwise, and writes
  back with `stmatrix`. Three matrix instructions + two `bar.sync`, no scalar LDS/STS.

Put together, the three pieces are a two-step GEMM path **Z = −(X·Y)·W** (X: (h, K), Y: (K, N), W: (N,
N2)): GEMM1's fp32 accumulator is narrowed in place into GEMM2's f16 A fragment (`warp_mma_f16` §4, no
smem, no barrier); h = 8 uses (a), h = 32 uses (b), h = 16 needs neither (one warp is exactly one 16-row
atom).

The cost of not using it is concrete: generating identity rows in registers to pad an 8-row operand to 16
rows forces the GEMM's K chain from h to 2h (half the beats only "carry the identity along"), a detour
through smem STSM→rendezvous→LDSM between GEMM1→GEMM2 so the identity rows can read the previous step's
result, and several extra all-party rendezvous per level — for the same work, about 3× the `HMMA` count,
2–4 more rendezvous per level, several times the per-level latency. The values are still all correct; the
correctness gate cannot see it.

## When needed (trigger)

- The dataflow is a "small-block two-step GEMM chain" — typically one level of a blocked triangular
  inverse: given the inverses A⁻¹, D⁻¹ of the diagonal blocks and the corner block C, compute −D⁻¹·C·A⁻¹
  (Schur-style corner update); the block side h doubles per level (8 → 16 → 32), the A operand of the
  smallest level has only 8 rows, and one output strip of the largest level is shared by two warps.
- Symptoms: the implementation contains code that **builds data to fill the atom**, such as "generate
  identity / unit-matrix rows in registers to fill M=16" or `if ml < h: x = 1 if ml == kx
  else 0`; or there is an STSM → barrier → LDSM round trip between GEMM1 and GEMM2 although both steps
  have the same M and the second step's K equals the first step's N (they could be chained directly).
- Ledger: the inverse segment's `bar.sync` count exceeds "2–3 per level"; the SASS `HMMA` count is about
  2–3× the sum of `M_rest × N_rest × K_rest`; the main `long_scoreboard` stall site sits on the downstream
  `wait` for this segment's product.

## Minimal runnable example

Every snippet below is GPU-certified: each has actually been run by the certification probe (the
probe is not included in this release; verdicts in "Sources"). The face is an f16 face
tiled from 8×8 tiles (second mode contiguous,
`tile_to_shape(make_layout((8, 8), stride=(8, 1)), (64, 64), order=(0, 1))`), the values are small
integers, and products are exact in f16/fp32, so the criterion is a strict `max_diff == 0`.

### 0. Common construction

```python
import cutlass, cutlass.cute as cute, cutlass.utils
from cutlass.cute.nvgpu import warp
F16, F32 = cutlass.Float16, cutlass.Float32

def bcast_m(t, reps):
    """(h, K) block -> stride-0 M-broadcast view ((h, reps), K); bytes untouched."""
    return cute.make_tensor(
        t.iterator.align(16),
        cute.make_layout(((cute.size(t, mode=[0]), reps), cute.size(t, mode=[1])),
                         stride=((t.layout.stride[0], 0), t.layout.stride[1])))

def swap(t):
    """(K, N) block -> (N, K) view: pure relabel; the contiguous mode is still the allocated one (N) -> take the transposed LDSM member."""
    return cute.make_tensor(t.iterator.align(16), cute.select(t.layout, [1, 0]))

def chain_acc_to_A(thr, acc1, M, N):
    """warp_mma_f16 §4: GEMM1's fp32 accumulator (M, N) -> GEMM2's f16 A fragment (M, K=N)."""
    opA = thr.make_fragment_A(thr.partition_shape_A((M, N)))
    cv, av = cute.size(acc1, mode=[0]), cute.size(opA, mode=[0])
    ratio = av // cv
    m_rest, k_rest = cute.size(acc1, mode=[1]), cute.size(acc1, mode=[2]) // ratio
    view = cute.make_tensor(opA.iterator,
                            cute.make_layout((cv, m_rest, ratio, k_rest),
                                             stride=(1, av, cv, av * m_rest)))
    for i in cutlass.range_constexpr(cute.size(acc1)):
        view[i] = F16(acc1[i])
    return opA
```

### 1. h = 8: broadcast view + whole-partition load + direct chaining + whole-partition write-back (one block per warp, no barrier)

```python
@cute.jit
def build_h8():
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 8)), (1, 1, 1))
    # the copy atom takes just 1 8x8 matrix: an extent=2 atom repetition appears on the 16-row tiler; through the broadcast view it is an alias
    tcA = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 1), F16), mma)
    tcB = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 1), F16), mma)
    tcC = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 1), F16), mma)
    return mma, tcA, tcB, tcC

# Inside the kernel body, one 16x16 block per warp (blk), four 8x8 quadrants:
q  = cute.flat_divide(blk, (8, 8))          # (8, 8, 2, 2)
sX = q[(None, None, 1, 1)]                  # A of GEMM1: (h=8, K=8), K contiguous
sY = q[(None, None, 1, 0)]                  # B of GEMM1 as (K, N)
sW = q[(None, None, 0, 0)]                  # B of GEMM2 as (N, N2)
sO = q[(None, None, 1, 0)]                  # output (8, 8), lands on Y

sXb, sOb = bcast_m(sX, 2), bcast_m(sO, 2)   # 16-row views, rows 8..15 alias rows 0..7
lane = cute.arch.lane_idx()                 # warp-level MMA uses lane, not tidx (warp_mma_f16 gotcha 4)
thr, thrA, thrB, thrC = (mma.get_slice(lane), tcA.get_slice(lane),
                         tcB.get_slice(lane), tcC.get_slice(lane))

frX = thr.make_fragment_A(thr.partition_A(sXb))            # ((2,2),1,1): 16-row A fragment
cute.copy(tcA, thrA.partition_S(sXb), thrA.retile(frX))     # whole partition: two LDSM.x1,
                                                            # the second re-reads rows 0..7 via stride-0
frY = thr.make_fragment_B(thr.partition_B(swap(sY)))
cute.copy(tcB, thrB.partition_S(swap(sY)), thrB.retile(frY))
frW = thr.make_fragment_B(thr.partition_B(swap(sW)))
cute.copy(tcB, thrB.partition_S(swap(sW)), thrB.retile(frW))

acc1 = cute.make_rmem_tensor(thr.partition_shape_C((16, 8)), F32); acc1.fill(0.0)
cute.gemm(mma, acc1, frX, frY, acc1)        # rows 8..15 == rows 0..7 (both halves of A identical)
for i in cutlass.range_constexpr(cute.size(acc1)):
    acc1[i] = -acc1[i]
opA = chain_acc_to_A(thr, acc1, 16, 8)      # no smem, no barrier
acc2 = cute.make_rmem_tensor(thr.partition_shape_C((16, 8)), F32); acc2.fill(0.0)
cute.gemm(mma, acc2, opA, frW, acc2)

c16 = cute.make_fragment_like(acc2, F16)
for i in cutlass.range_constexpr(cute.size(acc2)):
    c16[i] = F16(acc2[i])
cute.copy(tcC, thrC.retile(c16), thrC.partition_D(sOb))    # whole partition: two STSM.x1,
                                                            # the second lands the same values on the same 8 rows
```

The shape of `partition_S(sXb)` / `partition_D(sOb)` is `((8, 2), 1, 1)` (8 elements of one row per lane
× 2 atom repetitions), and `retile(frX)` / `retile(c16)` is `((2, 2), 1, 1)` (one 32-bit register per
lane × 2) — the second sub-mode of the value mode is the atom repetition along M. `cute.copy`'s verifier
only requires the **repetition counts** on both sides to match; the 8 vs 2 inside the atom is interpreted
by the atom itself; so copying the whole partition directly passes, while integer-indexing that sub-mode
away makes it compare 8 vs 2 and error out (gotcha 2).

### 2. h = 32: 2×2 warp grid, GEMM1's N split = GEMM2's K split, smem reduction with two barriers

```python
mma = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 16)), (1, 1, 1))
tcA   = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), F16), mma)
tcB   = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), F16), mma)
tcCst = cute.make_tiled_copy_C(cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), F16), mma)
tcCld = cute.make_tiled_copy_C(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), F16), mma)

# Inside the kernel body: one 64x64 face, X=(1,1) Y=(1,0) W=(0,0) each 32x32; 4 warps arranged 2x2
lane = cute.arch.lane_idx()                 # warp-level MMA uses lane, not tidx (warp_mma_f16 gotcha 4)
BAR_ID = 1                                  # named barrier id; 0 is sync_threads' CTA barrier (gotcha 7)
wid = cute.arch.warp_idx()
x, y = wid // 2, wid % 2                    # x: GEMM1's N half == GEMM2's K half; y: the output's M half
A1 = cute.flat_divide(sX, (16, 32))[(None, None, y, 0)]           # (16, K=32)
B1 = swap(cute.flat_divide(sY, (32, 16))[(None, None, 0, x)])     # (N=16, K=32)
B2 = swap(cute.flat_divide(sW, (16, 32))[(None, None, x, 0)])     # (N=32, K=16)
Oy = cute.flat_divide(sO, (16, 32))[(None, None, y, 0)]           # output strip (16, 32)

# GEMM1: P[y,x] = -(X[y,:] @ Y[:,x])  (16,16), full K=32; chained into GEMM2's A (16, K=16)
# GEMM2 partial sum: Z[y,:] += P[y,x] @ W[x,:]  (16,32) -- each x holds one half
...(load, gemm, negate, chain_acc_to_A(thr, acc1, 16, 16), gemm, narrow into c16, as in §1)

# Reduction: barrier -> x=0 writes first -> barrier -> x=1 read-add-write
cute.arch.barrier(barrier_id=BAR_ID, number_of_threads=128)   # everyone has finished reading Y/W (Z overwrites Y)
thrS = tcCst.get_slice(lane)
dst, src = thrS.partition_D(Oy), thrS.retile(c16)
if x == 0:
    cute.copy(tcCst, src, dst)                                # STSM: x=0's partial sum
cute.arch.barrier(barrier_id=BAR_ID, number_of_threads=128)   # partial sum visible to the partner
if x == 1:
    thrL = tcCld.get_slice(lane)
    red = cute.make_fragment_like(c16)
    cute.copy(tcCld, thrL.partition_S(Oy), thrL.retile(red))  # LDSM reads back in C-fragment order
    for i in cutlass.range_constexpr(cute.size(c16)):
        c16[i] = c16[i] + red[i]
    cute.copy(tcCst, src, dst)                                # STSM: the reduced strip
```

`x == 0` / `x == 1` are warp-uniform runtime branches (derived from `warp_idx()`) and need no
`const_expr`. Barriers per level: zero for h=8 (the block is warp-private), one for h=16 (separating
levels), two for the reduction + one separator for h=32.

## Gotchas

1. **Do not build identity rows to fill M=16.** It stretches GEMM1's K from h to 2h and turns GEMM2's A
   into something that must be re-read from smem (the identity rows must read the previous step's result),
   so direct chaining no longer holds and every level gets two extra rendezvous. The broadcast view lets
   `partition_A` see 16 rows while the bytes are only 8 rows; the extra accumulator rows are just copies of
   the first 8 and **need no** mask whatsoever.
2. **Do not integer-index away the atom-repetition sub-mode to "move only slice 0".**
   `partition_S(...)[((None, 0), None, None)]` turns the value mode from `(8, 2)` into `8`, and
   `retile(...)[((None, 0), None, None)]` into `2`; `cute.copy`'s IR verifier then reports
   `'cute.copy' op expects same size in rank 0 but got srcShape: (8,(1,1)) and dstShape: (2,(1,1))`
   — it treats the 8 smem elements and 2 register elements inside the atom as a rest that must align.
   Copying the whole partition as usual is the certified spelling; through the stride-0 view the repeated
   slice is only an alias, costing one more `LDSM.x1` / `STSM.x1`.
   If you really want to save that one instruction, you have to keep the rank/profile and change the
   extent of the repeated sub-mode to 1 before copying
   (`make_tensor(t.iterator, make_layout(((8, 1), 1, 1), stride=t.layout.stride))` on both sides)
   — that spelling is outside this entry's certified scope; don't bet on it in a kernel before the gate.
3. **The relation between `num_matrices` and the tiler.** An A `(16, 8)` / C `(16, 8)` tiler = 128
   elements = 2 8×8 matrices. Taking 1 matrix ⇒ the tiled copy carries an extent=2 atom repetition (this
   entry's certified form); taking 2 matrices (`LDSM.x2` / `STSM.x2`) ⇒ one instruction covers 16 rows with
   no repetition mode — that spelling is **outside this entry's certified scope** (probe R1 only ran
   `num_matrices=1`; there is no rung for 2); don't bet on it in a kernel before the gate; for the other
   levels of the same run, using `num_matrices=1` uniformly is fine — one more instruction, but each one
   narrower, and already verified byte by byte by the same probe.
4. **The `iterator.align(16)` of the broadcast view cannot be omitted.** The pointer of a sub-block cut
   dynamically by `flat_divide` keeps only the element width as its static alignment information, while
   the `ldmatrix` verifier wants 16 B — same root as [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md) gotcha 6.
   The stride-0 view does not change the alignment facts, so declare it directly.
5. **The read side of the split-K reduction must use `ldmatrix` in C-fragment order
   (`make_tiled_copy_C(LdMatrix…)`)**, not `make_tiled_copy_A`: the registers read back must align
   index-for-index with `c16[i]` for `c16[i] += red[i]` to work. Pick the member by the face's contiguous
   mode: C fragment (M, N) on a face with N contiguous → the non-transposed member.
6. **Each of the two barriers has its own reason; neither can be dropped.** First: the output strip
   overwrites the B-operand area of some GEMM (in-place update), so it must wait until every warp has
   finished reading; second: x=0's STSM must be visible to x=1. Dropping the first = x=0 overwrites B before
   x=1 has read it (silently wrong, schedule-dependent, intermittent).
7. **Don't use id 0 for `cute.arch.barrier(barrier_id=…, number_of_threads=128)`.** 0 is the CTA barrier
   used by `sync_threads`; a reduction barrier inside one warpgroup takes a non-zero id that does not
   collide with pipeline-class objects.
8. **Do not write the x=0 / x=1 branch as `const_expr`.** `x` comes from `warp_idx()`, a runtime value;
   `if cutlass.const_expr(x == 0)` fails at trace time with a dynamic condition. It is an ordinary
   warp-uniform `if`, which ptxas compiles to a uniform branch with no divergence.
9. **The h = 16 level needs neither broadcast nor split-K.** One warp is exactly one 16-row atom, and K =
   16 is one beat; wrapping it in a broadcast (reps=1) or a reduction is pure overhead. When the three
   levels share one template, the two compile-time constants `reps = 16 // h` (8 → 2, 16 → 1, 32 → 1) and
   `split = h // 32` (0/0/1) are enough to dispatch.

## Acceptance lines (machine-checkable)

- **PTX census**: the h=8 level shows `ldmatrix` (both non-`.trans` and `.trans`), `stmatrix`,
  `mma.sync.aligned.m16n8k8` ≥ 2; the h=32 level `mma.sync.aligned.m16n8k16` ≥ 6, `bar.sync` ≥ 3.
  Spelling and pitfalls in [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md).
- **SASS**: the inverse segment's `HMMA` count ≈ the sum over levels of `M_rest × N_rest × K_rest` (h=8: 2
  `1688`; h=16: 2 + 2 `16816`; h=32: 2 + 4 `16816` per warp), no longer 2–3×; the segment's `BAR.SYNC` ≤ 3
  per level.

## Sources

- PTX ISA, fragment distribution of warp-level `mma` (row ownership of the M=16 atom; the `ratio` grouping
  of C→A follows from it):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
- PTX ISA, `ldmatrix` / `stmatrix` (the addressing rule of one 16 B row per lane; why the stride-0 view does
  not go out of bounds):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-store-instruction-stmatrix>
- PTX ISA, `bar.sync` named barriers (`barrier_id` + thread count):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>
- CuTe DSL public API symbols: `cutlass.cute.nvgpu.warp.MmaF16BF16Op` / `LdMatrix8x8x16bOp` /
  `StMatrix8x8x16bOp`, `cute.make_tiled_mma` / `make_copy_atom` / `make_tiled_copy_A/B/C` /
  `partition_A/B/C/S/D` / `partition_shape_A/C` / `make_fragment_A/B` / `make_fragment_like` /
  `make_rmem_tensor` / `retile` / `gemm` / `make_tensor` / `make_layout` / `flat_divide` /
  `cute.select` / `Pointer.align` / `cute.arch.lane_idx` / `cute.arch.warp_idx` /
  `cute.arch.barrier(barrier_id, number_of_threads)` (cutlass-dsl 4.6.1).
- This repo's GPU-certification probe: two numeric
  contracts R1 (h=8 broadcast view + whole-partition load/write-back + direct chaining) and R2 (h=32 2×2
  warp grid split-K + two-barrier smem reduction), and two PTX census contracts C1/C2; each rung in its own
  subprocess. The verdict line is as transcribed in this entry's row of `index.md`.
