# Warp-level fp16/bf16 MMA (`mma.sync` m16n8k8 / m16n8k16)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Companion entries: [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) (the minimal path of warpgroup-level WGMMA),
> [`stsm_transpose_exchange`](stsm_transpose_exchange.md) (member choice of `stmatrix` on the write side).

## Purpose

Matrix multiply on the tensor cores within **a single warp (32 threads)**, instruction granularity `M=16`. It is the
second tensor-core path on SM90 besides WGMMA: WGMMA's M-mode atom is fixed at 64, must be issued collectively by a
warpgroup of 4 consecutive warps, takes its operands through smem descriptors and is asynchronous; warp-level
`mma.sync` has an M-mode atom of 16, is issued independently by one warp, **has both operands in registers**, and is
a synchronous instruction (the result can be read by the very next instruction).

So it covers two kinds of shape that WGMMA cannot:

- the tile's M dimension is smaller than 64, or you want the 4 warps of one warpgroup to **each process an unrelated
  small matrix** (WGMMA cannot — it is a warpgroup-collective instruction);
- a **chained MMA**: the previous step's fp32 accumulator immediately becomes the next step's A operand. Because A is
  in registers anyway, this step needs no smem round trip, only an in-register **regrouping + narrowing store** (see §4).

Not using it has a concrete cost: once such small-matrix segments degrade into a scalar SIMT loop + fp16 smem round
trips, the top of that kernel's per-opcode ledger becomes `HADD2` / `FFMA` / `FMUL` / `LDS` / `STS` / `BSSY`, without
a single `HMMA`, and `tensor%` collapses to single digits — it still computes correctly, just slowly, and **if the
architecture-conformance line only checks kernel count / block size, this deviation is completely invisible at the
gate**.

## When needed (trigger)

- You need the tensor cores, but the tile's M < 64, or you need independent division of work at warp granularity;
- the dataflow is a chain "MMA → short elementwise segment → MMA", with the same M in both steps and the later step's
  K equal to the earlier step's N (then the register hookup of §4 holds and no transit smem buffer is needed);
- profiler symptom: some kernel's `tensor%` is clearly below a same-size reference while DRAM bytes / grid / block /
  register count are all the same — i.e. **not traffic, not occupancy, but instruction cost**; together with the
  per-opcode ledger, `HMMA` absent from the top eight families and `HADD2 + LDS + STS + BSSY` together over half is
  the form this entry fixes.

## Minimal runnable example

Every segment below was actually run by the GPU-certified probe (the probe is not included in this release; verdicts in §Sources).

### 1. Build the atom and the tiled MMA

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp          # <- warp level lives in this namespace

F16, F32 = cutlass.Float16, cutlass.Float32

# only two legal instruction shapes: (16,8,8) and (16,8,16).
op16 = warp.MmaF16BF16Op(F16, F32, (16, 8, 16))
op8  = warp.MmaF16BF16Op(F16, F32, (16, 8, 8))

# atom_layout_mnk=(1,1,1) -> one atom = one warp = 32 threads.
# No permutation_mnk: the tile is the atom's own MNK,
# and every rest mode comes from "partitioning a larger tensor" (see gotcha 5).
mma = cute.make_tiled_mma(op16, (1, 1, 1))
assert mma.size == 32                        # thread count; the same-named WGMMA construction gives 128
```

For `bf16` just use `cutlass.BFloat16` (acc must then be `Float32`).

### 2. Slice, partition, accumulator

```python
lane = cute.arch.lane_idx()                  # not thread_idx! see gotcha 4
thr  = mma.get_slice(lane)

# sA: (M, K) smem face; sB: (N, K) smem face. M/N/K may exceed the atom's 16/8/K;
# the excess becomes the rest modes of the partition result.
tA = thr.partition_A(sA)                     # -> (V, M_rest, K_rest)
tB = thr.partition_B(sB)                     # -> (V, N_rest, K_rest)

frA = thr.make_fragment_A(tA)                # rmem of the same shape
frB = thr.make_fragment_B(tB)
cute.autovec_copy(tA, frA)                   # layout-agnostic fallback s2r path
cute.autovec_copy(tB, frB)

acc = cute.make_rmem_tensor(thr.partition_shape_C((16, 16)), F32)
acc.fill(0.0)                                # see gotcha 9
cute.gemm(mma, acc, frA, frB, acc)           # D <- A@B^T_k + C, synchronous, no fence/commit
```

`cute.gemm` itself iterates over `M_rest / N_rest / K_rest`, so one call covers the whole tile.
In the example `(M, N, K) = (16, 16, 16)` while the atom is `(16, 8, 16)`, so `N_rest = 2`:
one warp issues two beats along N.

Read back using an identity tensor for the coordinates (the same `partition_C` map is both the write map and the read
map):

```python
cC = thr.partition_C(cute.make_identity_tensor((16, 16)))
for i in cutlass.range_constexpr(cute.size(acc)):
    c = cC[i]                                # c[0] = row, c[1] = column
    gOut[(c[0], c[1])] = acc[i]
```

### 3. Feeding operands with `ldmatrix` (the fast path)

`autovec_copy` is layout-agnostic but fetches fragment elements one at a time / in short vectors; `ldmatrix` moves a
set of 8×8 matrices for the whole warp in one instruction, and is where this path really saves smem transactions. The
price is that it **has hard requirements on the orientation of the smem face**: the non-transposing member takes its
source as "one run of 8 **contiguous** 16-bit elements per addressing lane", while the A and B fragments of
`m16n8k*` are both "each lane takes 2 adjacent elements along K" — so **the K mode of both operand faces must be the
contiguous mode**, and both sides take `transpose=False`. The copy atom also **must be derived from the tiled MMA**,
otherwise the thread-value mappings do not line up (see gotchas 6 and 7):

```python
sA = ...   # (M, K), K contiguous: cute.make_layout((M, K), stride=(K, 1))
sB = ...   # (N, K), K contiguous: cute.make_layout((N, K), stride=(K, 1))

def ldsm_atom(rows, cols):
    """The non-transposing member needed to cover a whole (rows, cols) operand tile.
    One 8x8 16-bit matrix = 64 elements; the member can only take 1 / 2 / 4 of them."""
    n = (rows * cols) // 64
    assert n in (1, 2, 4)
    return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, n), F16)

# the tiler is given by the tiled MMA itself: A side (tile_M, tile_K), B side (tile_N, tile_K).
# the positional arguments are (transpose, num_matrices).
tcA = cute.make_tiled_copy_A(
    ldsm_atom(mma.get_tile_size(0), mma.get_tile_size(2)), mma)
tcB = cute.make_tiled_copy_B(
    ldsm_atom(mma.get_tile_size(1), mma.get_tile_size(2)), mma)

thrA, thrB = tcA.get_slice(lane), tcB.get_slice(lane)
frA = thr.make_fragment_A(thr.partition_A(sA))
frB = thr.make_fragment_B(thr.partition_B(sB))
cute.copy(tcA, thrA.partition_S(sA), thrA.retile(frA))   # retile: see gotcha 6
cute.copy(tcB, thrB.partition_S(sB), thrB.retile(frB))
cute.gemm(mma, acc, frA, frB, acc)
```

`num_matrices` **is computed from the tiler's element count, not from K**: the `(16,8,16)` atom gives an A tiler
`(16, 16)` = 256 elements = 4 matrices, and a B tiler `(8, 16)` = 128 elements = 2 matrices. When the face is larger
than the tiler (e.g. `sB` is `(16, 16)` while the B tiler is `(8, 16)`), the excess goes through rest modes as usual,
and `cute.copy` issues a few more instructions along rest to fill it in — the profiles on both sides agree, no extra
handling needed.

**Assert the mapping once right after construction** (gotcha 6) — the cheapest insurance in this entry:

```python
# Neither helper is **decorated**, and both use a plain range -- see gotcha 11: this 32-lane scan
# must stay in an undecorated function; written inside a @cute.jit body it is rewritten by the preprocessor into a dynamic scf loop,
# so l becomes a runtime value and blows up the first time it is used for a Python-level index / formatting.
def coords(t):
    """The static (mode0, mode1) coordinate sequence of an identity-tensor partition, in the partition's own linear order."""
    return tuple((int(t[i][0]), int(t[i][1])) for i in range(cute.size(t)))

def assert_ldsm_map(mma, tcA, idA):
    for l in range(32):
        want = coords(mma.get_slice(l).partition_A(idA))
        got  = coords(tcA.get_slice(l).partition_D(idA))
        if got != want:
            raise RuntimeError(f"ldmatrix A landing map does not match the MMA A map @lane {l}: {got} vs {want}")

# call it like this from the @cute.jit body (partition_* needs a live IR context, so the scan itself runs at trace time):
assert_ldsm_map(mma, tcA, cute.make_identity_tensor((M, K)))
```

### 4. Chained MMA: the previous step's fp32 accumulator → the next step's fp16 A operand

This is the most valuable segment of this entry, and the easiest one to get wrong. **Hardware fact**: the C mapping
and the A mapping of the same `m16n8k*` atom hand the same `(row, column)` to the same lane — so "the accumulator as
the next step's A" **needs no cross-lane movement at all**, and hence no smem transit. The only difference is in the
**grouping**:

- one A k-block of `m16n8k16` has 8 values per thread, while one n8 accumulator tile has 4 values per thread, so
  **one A k-block == `ratio = 2` accumulator tiles adjacent along N, joined end to end**;
- the A k-block of `m16n8k8` has 4 values per thread, `ratio = 1`, and the two profiles line up directly.

The key is to **build the destination fragment with `make_fragment_A(partition_shape_A(...))`** (so `cute.gemm`
receives exactly the profile it recognises), then open a **strided view** over its registers and write into it in the
accumulator's linear order. The code below is the same path for `ratio == 1` and `ratio == 2`, with no `const_expr`
fork needed:

```python
# A of the second step = C of the first step: M the same as the first step, K equal to the first step's N.
opA = thr.make_fragment_A(thr.partition_shape_A((M, N)))

cv     = cute.size(acc1, mode=[0])       # values per lane per C atom tile
av     = cute.size(opA,  mode=[0])       # values per lane per A atom tile
ratio  = av // cv                        # k16 -> 2, k8 -> 1
m_rest = cute.size(acc1, mode=[1])
k_rest = cute.size(acc1, mode=[2]) // ratio

# the same registers, written through the accumulator's profile:
#   compact A index = v + av*m + cv*r + av*m_rest*k
#   acc linear order = (v, m, n), where r = n % ratio, k = n // ratio
view = cute.make_tensor(
    opA.iterator,
    cute.make_layout((cv, m_rest, ratio, k_rest),
                     stride=(1, av, cv, av * m_rest)))
for i in cutlass.range_constexpr(cute.size(acc1)):
    view[i] = F16(acc1[i])

acc2 = cute.make_rmem_tensor(thr.partition_shape_C((M, N2)), F32)
acc2.fill(0.0)
cute.gemm(mma, acc2, opA, frB2, acc2)    # no smem round trip, no barrier
```

The elementwise segment in between can be done in place on `acc1` (`acc1[i] = -acc1[i]` and the like), without
affecting the hookup above.

**Shape conditions of the hookup** (if they do not hold, only a real smem transpose remains, see gotcha 8): the
second step's `M` equals the first step's, the second step's `K` equals the first step's `N`, and `n_rest` is
divisible by `ratio`.

## Gotchas

1. **Same name, different thing: `warp.MmaF16BF16Op` ≠ `warpgroup.MmaF16BF16Op`.**
   Both namespaces have a class called `MmaF16BF16Op`, but they are two different hardware instructions:
   the one in `cute.nvgpu.warp` takes three arguments `(ab_dtype, acc_dtype, shape_mnk)`, M=16, 32 threads;
   the one in `cute.nvgpu.warpgroup` is WGMMA and takes six arguments (also `OperandSource` and the two A/B major
   modes), M=64, 128 threads. **Calling the warpgroup class with the three-argument warp-level call fails outright**;
   conversely, with the import written as `from cutlass.cute.nvgpu import warpgroup`, the warp-level op is nowhere to
   be found, and it is easy to conclude wrongly that "this DSL has no warp-level MMA".
   (Certified: R0.)

2. **`hopper_helpers.make_trivial_tiled_mma` will never give you a warp-level MMA.**
   It is the constructor of the warpgroup collective; no matter how small `tiler_mn` is, you cannot get a 32-thread
   tiled MMA from it — measured: for `tiler_mn=(16, 16)` it raises `expects the M-mode to be 64, but
   got 16` straight away. For warp level you must `make_tiled_mma(warp.MmaXXXOp(...), (1,1,1))` yourself.
   Likewise, the `warpgroup` namespace only has the fp16/bf16 and fp8 op families; asking it with operand types
   outside these two families gives `unsupported a_dtype and b_dtype` — that does not mean the hardware lacks the
   instruction, only that **WGMMA** does; the warp-level namespace has a separate corresponding op. (Certified: R0.)

3. **The instruction shape has only two legal values**: `(16, 8, 8)` and `(16, 8, 16)`.
   Writing `(16, 8, 32)`, `(16, 16, 16)`, or putting the tile size into M/N, all raise `OpError` when the op is
   constructed. **The tile size is not written into the op** (see gotcha 5).
   The operand types are also only `Float16` / `BFloat16` (with `BFloat16` acc must be `Float32`);
   fp32 operands are another op (the tf32 family), not covered by this entry. (Certified: R0.)

4. **`get_slice` takes `lane_idx()`, not `thread_idx()`.**
   The thread-value mapping of a warp MMA is only defined within a warp. Pass the CTA's `tidx` and warp 0 happens to
   be right while every other warp is wrong (and **silently** so: `partition_*` does not complain, nor does
   `cute.gemm`). Conversely, precisely because the slice is lane-level, the N warps of one CTA can each run completely
   unrelated tiles — as long as their smem faces and output positions are separated by `tidx // 32`. (Certified: R1,
   R2, R4, R5; the probe has 4 warps each compute once and each write its own output block, then compares per warp
   against the reference.)

5. **Where the tile size comes from: prefer letting `partition_*` take a larger tensor over turning `permutation_mnk`.**
   The tile of `make_tiled_mma(op, (1,1,1))` is the atom's own MNK (e.g. `(16, 8, 16)`).
   For one warp to cover `(16, 16, 16)`, hand a `(16,16)` sB directly to `partition_B`, get `N_rest = 2`, and
   `cute.gemm` issues two beats by itself — this entire path is certified by the probe.
   `permutation_mnk` is a different matter (it changes the tile size, the number of rest modes, and the tiler size
   derived by `make_tiled_copy_A/B/C` all at once); this entry **does not rely** on it; unless you know you need the
   permute semantics, leave it out — one fewer degree of freedom that can twist the copy side's profile.

6. **The three parts of `ldmatrix` must match, and must be asserted after construction.**
   `make_tiled_copy_{A,B,C}(atom, tiled_mma)` derives the tiling,
   `thr_copy.partition_S(smem_face)` takes the source, `thr_copy.retile(fragment)` takes the destination.
   Without `retile` you are matching the MMA's fragment profile against the copy's profile, and trace time reports a
   shape mismatch. Note that `ldmatrix` is a **cross-lane movement** instruction: a wrong construction is not "this
   thread's element order is scrambled" (which could still be rescued by re-indexing) but "this lane got someone
   else's share", which downstream cannot repair. So the assertion at the end of §3 is worth keeping permanently:
   `tc.get_slice(l).partition_D(identity)` must be **element for element identical** to
   `mma.get_slice(l).partition_A/B(identity)`, checked for all 32 lanes.
   It is a pure trace-time coordinate comparison, touching no device and costing no time, yet it turns "silently moved
   the wrong data" into an error that names the culprit.
   (Certified: R3 — 32 lanes each on the A side and the B side, all 64 maps identical.)

7. **Orientation (which mode is contiguous) is a property of the smem face, not something a view can change.**
   The non-transposing member takes its source as "one run of 8 contiguous 16-bit elements per addressing lane", so
   **both the A face `(M, K)` and the B face `(N, K)` need K contiguous** (`stride=(K, 1)`).
   If a face has the wrong orientation, what must change is the face's own stride (or the side that writes smem);
   swapping views with `cute.composition` cannot change which mode is contiguous. When you cannot get a K-contiguous
   face, fall back to `autovec_copy` from §2: it has no orientation requirement, it just saves no transactions — the
   probe's R1 deliberately writes the A face M-contiguous (`stride=(1, M)`) and passes, proving this fallback path
   really is orientation-agnostic.
   Also, the verifier checks contiguity/alignment, **not whether the `transpose` boolean is right**: with the right
   orientation but the boolean inverted, it compiles fine and moves transposed data — a silent error. So `transpose`
   cannot be confirmed by "it compiled"; use the coordinate assertion of gotcha 6, or compare numbers against the
   `autovec_copy` path.
   (Certified: R1 with an M-contiguous face through `autovec_copy`, R2 with K-contiguous faces through two
   non-transposing members; both give `max_diff=0` against the same reference.)

8. **The nastiest trap of the chained hookup: do not re-view the new fragment with the accumulator's own layout.**
   The seemingly easiest spelling is `cute.make_tensor(new_frag.iterator, acc.layout)` — writing that batch of new
   registers through the accumulator's layout. **This only happens to work when `M_rest == 1` (i.e. the tile's M is
   exactly the atom's 16)**: the linear index of a compact A fragment is `v + cv·r + cv·ratio·m + cv·ratio·M_rest·k`,
   while `acc.layout` gives `v + cv·m + cv·M_rest·(r + ratio·k)`, and the two are equal only when `M_rest == 1`.
   As soon as M is 32 or 64 (`M_rest = 2, 4`), it quietly writes garbage — compiles, no error, wrong numbers.
   The `view` of §4 explicitly splits `(r, k)` into two independent modes and holds for every `M_rest`.
   (Certified: R4 runs the chain with `M = 32` (`M_rest = 2`), exactly the tier where this shortcut breaks;
   R5 runs the same chain with `m16n8k8`, covering the other branch `ratio == 1`.)

9. **No fence / commit / wait_group, but no `ACCUMULATE` knob either.**
   `mma.sync.aligned` is a warp-synchronous instruction: D can be read by the very next instruction; **do not** add
   `warpgroup.fence()` / `commit_group()` / `wait_group()` (that is the WGMMA protocol; here it is pure overhead). The
   price is that WGMMA's trick of "first beat `ACCUMULATE=False` replaces zeroing" does not exist here:
   `cute.gemm(mma, acc, A, B, acc)` is always `D = A·B + C`, and before the first call you **must
   `acc.fill(0.0)` yourself**. Accumulating means continuing to pass the same `acc` as C.

10. **Operands are always in registers, so there is no major mode and no smem descriptor.**
    `partition_A/B` gathers from an smem face of any layout into fragment order, and the face needs no reorientation
    for it; there is also no WGMMA-style constraint "with A in smem both A and B can be transposed, with A in
    registers only B can". The price is a real s2r (`autovec_copy` or `ldmatrix`) for every k-block beat, and register
    pressure falls on the A/B fragments rather than on descriptors — on a large tile, reading the whole A/B fragment at
    once pushes up the peak; fetching per k-block just in time is cheaper.

11. **`range_constexpr` / `const_expr` can only be written in a decorated function body.**
    The DSL preprocessor only rewrites functions decorated with `@cute.jit` / `@cute.kernel`; pulling a loop with
    `cutlass.range_constexpr(...)` into an **undecorated** ordinary helper raises at trace time
    `[Internal Error] ... range_constexpr should be preprocessed by preprocessor.`
    — this error claims "it is a DSL bug, not a problem with your kernel", which is very misleading; it is really just
    a missing decorator. Two ways out: write such loops directly in the kernel body, or add `@cute.jit` to the helper.
    Conversely, **a trace-time helper that only uses plain `range` needs no decorator** — the two helpers of §3 iterate
    over static coordinates with plain `range` in undecorated functions and are called from the `@cute.jit` body,
    entirely normally (certified: all 64 maps of R3 were obtained exactly this way). The criterion is whether the loop
    body has constexpr constructs, not which level it runs at.

    **And for a scan like the one in §3, "undecorated" is required, not just optional.** The preprocessor rewrites
    **every** `for` in a decorated function body into a staged `scf` loop — including `for l in range(32)` whose range
    is entirely compile-time constant. After the rewrite `l` is a runtime value: fed to `get_slice()` it still looks
    fine, but the first time it is used as a Python-level value (`f"@lane {l}"`, `"%2d" % l`, indexing a Python list
    with it) it raises
    `[PHASE_DYNAMIC_INDEX]: Cannot use a Runtime value (Staged value) value as a
    list index or for loop range in plain Python code`. The error lands on that formatting statement, several frames
    away from the real cause (where the loop sits), and is easily read as "the coordinates were fetched wrong".
    Two ways out: when the loop body only uses plain `range`, move it entirely into an **undecorated** helper (the
    preprocessor does not enter undecorated callees, so `l` stays a real `int`); or keep it in the decorated body and
    rewrite it as `cutlass.range_constexpr(32)`, explicitly requesting compile-time unrolling.
    (Measured: the first round of R3 wrote `for lane in range(32)` directly in the `@cute.jit` body and was hit by
    `PHASE_DYNAMIC_INDEX` on `"A lane %2d" % lane`; it passed after moving into an undecorated helper.)
    See [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md).

12. **Putting several mechanisms into one module means one mistake knocks the whole batch down to "never reached the
    device".** `cute.compile` compiles a whole module: any trace / verification failure anywhere in the module denies
    a verdict to every mechanism covered by the same `cute.compile`. When writing a probe, **compile each mechanism
    separately** (more robust still, give each its own subprocess, which also isolates CUDA context faults), so the
    diagnosis lands on the item that actually failed.

## Acceptance bar (machine-checkable; recommended to write into the decision's acceptance clause)

Correctness alone cannot tell whether this path was built — the scalar SIMT version's **numbers are right**. To make
the deviation visible at the gate, hang at least one static or NCU-side line:

- **Static SASS census**: the kernel's `HMMA` count ≥ the number of chain beats (the sum of
  `M_rest × N_rest × K_rest` over every `cute.gemm` after unrolling). For the spelling and traps see
  [`mma_operand_source_verify`](mma_operand_source_verify.md) (threshold = number of chain beats, spill pairing,
  kernel-name check).
- **NCU side**: anchor a lower bound on the kernel's `tensor_pct`. It is sensitive to "not built at all"; it is
  insensitive to "built but not overlapped", which is another line's business.

## Sources

- PTX ISA, warp-level matrix instructions (the per-lane register distribution tables of the A/B/C fragments of
  `mma.m16n8k8` / `m16n8k16`, from which the `ratio` grouping of §4 is derived):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>
- PTX ISA, `ldmatrix`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>
- CuTe DSL public API symbols (the interface surface the student has to call anyway):
  `cutlass.cute.nvgpu.warp.MmaF16BF16Op`, `cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp`,
  `cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op`, `cutlass.utils.hopper_helpers.make_trivial_tiled_mma`,
  `cute.make_tiled_mma` / `make_mma_atom` / `make_copy_atom` /
  `make_tiled_copy_A` / `make_tiled_copy_B` / `make_tiled_copy_C` /
  `TiledMma.get_tile_size` / `TiledMma.size` /
  `partition_A` / `partition_B` / `partition_C` / `partition_D` / `partition_S` /
  `partition_shape_A` / `partition_shape_C` /
  `make_fragment_A` / `make_fragment_B` / `make_rmem_tensor` / `make_identity_tensor` /
  `autovec_copy` / `retile` / `gemm` / `cute.size(..., mode=[i])` / `cute.arch.lane_idx`
  (cutlass-dsl 4.6.1; in that version `make_fragment` has been renamed `make_rmem_tensor`).
- This repo's GPU-certified probe: six contracts R0–R5,
  each in its own subprocess, compiled separately, each producing its own verdict line (the verdict lines are
  transcribed in this entry's row of index.md).
