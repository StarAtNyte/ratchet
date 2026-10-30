# tmem_r2t_recast_fragment (writing a narrow-type operand from registers back into TMEM: the three constraints of a recast fragment)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell) only** (TMEM is new in this generation; there is no sm90 entry to port from).
>
> This entry answers exactly one question: **how does a sub-32-bit operand get written from registers into TMEM, and where is the cost booked.**
> The answer is **three mutually independent constraints**, and each one fails in a different way when violated:
>
> | | Constraint | What violating it looks like |
> |---|---|---|
> | **① Element type** | The copy atom must be built on a **32-bit** type; the narrow-type face is a `cute.recast_tensor` view of the same registers | With a 32-bit destination view it **dies at compile time**; mark the destination **as narrow too** and it **compiles** — the guard compares element widths only, and getting through does not mean the write is right |
> | **② Repetition depth** | Exactly one depth makes the recast fragment **congruent** with the t2r fragment it packs | **`cute.copy` matches by mode**: if the shapes differ you get a mode-size mismatch error, or it degrades into an element-wise path |
> | **③ Live range** | The register peak of this segment is set by **the last consumer of the source fragment**, **not by the granularity of the store** | **No error, no wrong numbers**: splitting the destination buys nothing, and under register-quota pressure the spill stays exactly the same |
>
> ① and ② error out on the spot, so they only cost time. **③ does not error** — it compiles, it runs, it computes correctly, and the cost is booked on
> `stack frame`, a number that is in no default NCU panel. That is the point of this entry,
> and that point is itself **negative**: do not start by changing how the store is written.
>
> When writing the **A face of a CTA pair (`cta_group::2`) MMA**, there is one more lane-layout rule: each CTA's 64 rows must be placed once in
> each of the two lane halves — see the "cta_group::2" section.

## Purpose

TMEM is addressed in **32-bit columns**. tcgen05 UMMA allows the A operand to live in TMEM, and that operand is a **narrow type**
(bf16 / fp16 / fp8). Put the two together and you get this path: write a narrow-type register fragment back into
TMEM so that the next MMA reads it directly from TMEM.

The path itself takes a detour: `tcgen05.st` moves one 32-bit column at a time (holding 2 16-bit values, or 4 8-bit values),
so **the copy atom's element type is 32-bit**, and the narrow-type face is a recast view of **the same registers**.
Write through the narrow-type view, and hand `cute.copy` the 32-bit tensor — two names, one storage.

## When needed (trigger)

**Structural trigger**: the code has both (a) tcgen05 and (b) a narrow-type register fragment that has to land in TMEM.
The typical shape is "read an f32 tile from TMEM → element-wise work in registers → write back to TMEM as the next MMA's operand".

**Symptom trigger** (you already hit it and come back here):

| Symptom | Which constraint |
|---|---|
| `expects atom and input memref to have the same element size but got 'bf16' for the atom while tmem memref type is 'f32'` | ① — the destination tensor is a 32-bit view, the atom is marked narrow, the constructor rejects it on the spot |
| `LLVM ERROR: unknow intrinsic signature for tcgen05.st` | ① dying the other way — the narrow-type atom passed the element-width check and died in lowering (**an LLVM abort, not a Python exception; `try/except` cannot catch it**) |
| **It compiled and ran to completion**, but the next MMA does not read what was just written | ① — the destination tensor was marked narrow, so the element widths match and the guard lets it through; "it compiles" is not evidence on this path |
| `Expected source and destination tensors to have the same size in mode-N, but got A and B` | ② |
| Under register-quota pressure this segment's `bytes stack frame` is non-zero, **and making the store finer or coarser does not move it** | ③ |
| CTA pair MMA with A read from TMEM: whole rows wrong, or **all errors in the second half of the N columns** | The "cta_group::2" section — both lane halves must hold this CTA's rows |
| Taking per-slot identity coordinates raises `` expects `coord` and shape of view are weakly congruent ``; or your own trace-time coverage check reports something like "r2t faces … do not tile 128 x 256 columns" while the copy itself raises nothing | Gotcha "`partition_D` is not one register per slot" |

**Anti-trigger**: when the destination is **smem** rather than TMEM, this entry does not apply — that is a different family of copy atoms, whose width is set by
the smem-side atom (likewise **not by the swizzle**, but the selection rule is not here). This entry covers only the TMEM side.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import SmemAllocator

F32, BF16 = cutlass.Float32, cutlass.BFloat16

# TMEM tiled-copy constructor: signature `(copy_atom, tmem_tensor) -> TiledCopy`; it tiles
# a copy atom over the given TMEM tensor. It is the only exported symbol in the `tcgen05`
# module that takes this pair of arguments (`dir(tcgen05)` finds it); the exact symbol name and
# two real call sites are the `tiled_r2t = ` / `tiled_t2r = ` lines of the certification probe
# listed in this entry's "Sources". Bind it and this example runs as is.
TMEM_COPY = ...

M      = 128          # TMEM rows: one 128-lane warpgroup, one row per lane
NARROW = 128          # number of narrow-type columns of the operand tile
NW     = NARROW // 2  # number of 32-bit columns it occupies (one bf16 pair per column); fp8 would be // 4
LANE   = 65536        # TMEM lane (row) stride, in units of 32-bit columns
P_COL  = 128          # start column of the operand region, disjoint from the f32 source region (columns 0..127)

@cute.kernel
def k(gDst: cute.Tensor):
    smem    = SmemAllocator()
    holding = smem.allocate_array(cutlass.Int32, 1, byte_alignment=16)
    warp    = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # The tcgen05 alloc instruction is warp-wide: the whole warp must arrive.
    # Adding elect_one() here strands 31 lanes and the CTA deadlocks at the next barrier.
    if warp == 0:
        cute.arch.alloc_tmem(512, holding, is_two_cta=False)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        F32, alignment=16, ptr_to_buffer_holding_addr=holding)

    # Pointer arithmetic is in units of 32-bit columns, not narrow-type elements.
    tS = cute.make_tensor(tmem_ptr,
                          cute.make_layout((M, NARROW), stride=(LANE, 1)))
    tP = cute.make_tensor(tmem_ptr + P_COL,
                          cute.make_layout((M, NW), stride=(LANE, 1)))

    # ---- ① atom built on F32; the narrow-type face is a recast view ----------
    tiled_r2t = TMEM_COPY(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x32, tcgen05.Unpack.NONE),
            F32),                      # <- BF16 here with tP a 32-bit view = compile-time dead end;
        tP)                            #    changing tP to a narrow view as well compiles, see gotcha ①
    thr_r2t = tiled_r2t.get_slice(tidx)
    rP32 = cute.make_rmem_tensor(thr_r2t.partition_S(tP).shape, F32)
    rP   = cute.recast_tensor(rP32, BF16)   # narrow-type face of the same registers

    # ---- source-side t2r: same op family, depth chosen by the equation below ----
    tiled_t2r = TMEM_COPY(
        cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), F32),
        tS)
    thr_t2r = tiled_t2r.get_slice(tidx)
    rS = cute.make_rmem_tensor(thr_t2r.partition_D(tS).shape, F32)
    cute.copy(tiled_t2r, thr_t2r.partition_S(tS), rS)
    cute.arch.fence_view_async_tmem_load()

    # ---- pack: when congruent, one whole-tensor store --------------------------
    # Any element-wise chain (scale / exp2 / mask ...) is done in place on rS;
    # this last step only does "convert to narrow type + land in the recast view".
    rP.store(rS.load().to(BF16))
    # When the handoff must be split into pieces (e.g. notify the consumer of half of P early), split mode-2 on both sides together:
    #   for j in cutlass.range_constexpr(cute.size(rP, mode=[2])):
    #       rP[None, None, j].store(rS[None, None, j].load().to(BF16))
    #       cute.copy(tiled_r2t, rP32[None, None, j],
    #                 thr_r2t.partition_D(tP)[None, None, j])
    # What splitting buys and what it does not: see gotcha ③ — usually it does not buy the thing you want.

    cute.copy(tiled_r2t, rP32, thr_r2t.partition_D(tP))   # hand over the 32-bit tensor
    cute.arch.fence_view_async_tmem_store()               # prerequisite for consumer visibility
```

### ② How to choose the depth

Both sides are **the same op family**, so a single equation suffices. The constructor tiles the atom over the given TMEM tensor:

- source side `Ld32x32bOp(Repetition.xE)` tiles `C` 32-bit columns → register face `((E,1),1,C/E)`;
- destination side `St32x32bOp(Repetition.xW)` tiles `N` 32-bit columns → register face `((W,1),1,N/W)`;
  recast to a `T`-bit type, the inner mode is multiplied by `32/T` → `((W*32/T,1),1,N/W)`.

**Condition for the two to be congruent**:

```
W * 32/T == E        and        N/W == C/E
```

The second equation holds automatically (the narrow operand occupies `N = C*T/32` 32-bit columns; substitute and you get the first),
so the only thing to solve is **`W = E*T/32`**: `E/2` for bf16, `E/4` for fp8.

An instance with `T=16`, `C=128`, `E=64`, `N=64`:

| Depth pair | r2t face (f32) | recast to bf16 | t2r face | Congruent |
|---|---|---|---|---|
| **ld x64 / st x32** | `((32,1),1,2)` | `((64,1),1,2)` | `((64,1),1,2)` | **yes** |
| ld x64 / st x16 | `((16,1),1,4)` | `((32,1),1,4)` | `((64,1),1,2)` | no |
| **ld x32 / st x16** | `((16,1),1,4)` | `((32,1),1,4)` | `((32,1),1,4)` | **yes** |
| ld x32 / st x32 | `((32,1),1,2)` | `((64,1),1,2)` | `((32,1),1,4)` | no |

**Congruence lies on the diagonal, not at some absolute depth**: `W` follows `E`. So the question "which Repetition should I use"
is the wrong question — fix the source side's `E` first (it is set by the epilogue granularity of the upstream MMA), and `W` has no freedom left.

Congruence is **the precondition for a whole-tensor `.store()`**: without it there is no single-shot write, only element-wise.
So **the depth is pinned by the source side's epilogue tile**; to change one, change both and redo the equation.

### cta_group::2: in TMEM the A face must hold this CTA's rows once in each of the two lane halves

In a CTA pair UMMA (`tcgen05.MmaF16BF16Op(bf16, f32, (128, N, 16), CtaGroup.TWO, OperandSource.TMEM, K, K)`),
M = 128 is the row count of **the whole pair**: `partition_shape_A((128, K))` reports **64** rows per CTA (CTA v owns rows `64v .. 64v+63`),
and N splits B in half between the two CTAs (each CTA's smem holds N/2 columns). But the TMEM A face given by `make_fragment_A(...)` is
**128 lanes** (`((128,16),1,K/16,1)`). What goes into those 128 lanes was measured (`N = 96`, 48 columns of B per CTA):

| Row placed in lane ℓ | Result | What is wrong |
|---|---|---|
| **`64v + ℓ % 64`**: this CTA's 64 rows, one copy in lanes 0..63 and **another copy** in lanes 64..127 | **byte-exact** | — |
| `ℓ`: by the pair's global row index (CTA 0's upper half holds the peer's rows, CTA 1's lower half holds the peer's rows) | wrong, `max_diff` 461 | all 128 rows wrong, errors in both halves of N |
| lanes 0..63 hold this CTA's rows, lanes 64..127 hold a constant | wrong, `max_diff` 7137 | all 128 rows wrong, **every error in the second half of N** (6143 / 6144), not one error in the first half of N |

The third row shows two things: the MMA **does read** lanes 64..127, and it reads them for the second half of the N columns — lane half h feeds the h-th half of N
(i.e. CTA h's half of B); the error counts of the second row are equal across row half × column half (3053 / 3061 vs 3061 / 3053), consistent with that reading.
So both lane halves must hold all 64 of this CTA's rows, and **the A face cannot be halved by "putting the second half of K into lanes 64..127"**: a bf16 A face with K = 512
is 256 32-bit columns, and the TMEM column budget (A face + resident accumulator ≤ 512 columns) must count it in full.

The write reuses the r2t recipe above; the only difference is that each slot's **absolute** (lane, column) must be asked from the copy itself and then converted to a row index (below is how the certification probe
writes it, K = 512, 128 threads):

```python
# ptr32: retrieve_ptr(Float32) after allocating with TmemAllocator(..., is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=...)
NCOL = K // 2                                         # 32-bit columns occupied by the bf16 A face (K = 512 -> 256)
tP = cute.make_tensor(ptr32 + COL_A, cute.make_layout((128, NCOL), stride=(65536, 1)))
r2t = tcgen05.make_tmem_copy(
    cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                        cutlass.Float32), tP)
thr = r2t.get_slice(tidx)
tPd = thr.partition_D(tP)
rP32 = cute.make_rmem_tensor(thr.partition_S(tP).shape, cutlass.Float32)    # ((8,1),1,32)
rP = cute.recast_tensor(rP32, cutlass.BFloat16)                            # ((16,1),1,32), same registers
tPc = thr.partition_S(cute.make_identity_tensor((128, NCOL)))             # congruent with rP32 (gotcha: do not use partition_D)
for j in cutlass.range_constexpr(NCOL // 8):
    for e in cutlass.range_constexpr(8):
        crd = tPc[((e, 0), 0, j)]
        ell, col = crd[0], crd[1]                     # absolute lane and 32-bit column of this slot
        row = v * cutlass.Int32(64) + ell % cutlass.Int32(64)
        rP[((2 * e, 0), 0, j)] = X[row, 2 * col]      # one column holds the bf16 pair (k = 2*col, 2*col + 1)
        rP[((2 * e + 1, 0), 0, j)] = X[row, 2 * col + 1]
    cute.copy(r2t, rP32[(None, 0, j)], tPd[(None, 0, j)])
cute.arch.fence_view_async_tmem_store()
cute.arch.barrier()
cute.arch.cluster_arrive_relaxed()                    # the MMA issued by the leader also reads the peer's A face: issue only after both sides are written
cute.arch.cluster_wait()

# MMA side: the library's own A layout, over a bf16 reinterpretation of the same columns (one TMEM block, two type views)
aL = tiled_mma.make_fragment_A(cute.append(tiled_mma.partition_shape_A((128, K)), 1)).layout
tA = cute.make_tensor(cute.recast_ptr(ptr32 + COL_A, None, dtype=cutlass.BFloat16), aL)
# utils.compute_tmem_cols_from_layout(aL, cutlass.BFloat16) == NCOL; check this before laying out the column budget
# one warp of the leader issues: tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
#   for kb in range_constexpr(K // 16):
#       cute.gemm(tiled_mma, acc, tA[(None, None, kb, 0)], tCrB[(None, None, kb, 0)], acc)
#       tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

Element-wise transforms can ride along on the write path: the same probe folds interleaved-pair RoPE into the last 64 dims of the A face (each slot takes cos / sin by the
(row, k) given by the identity partition), also byte-exact. The library has a ready-made spelling of the same layout: `cutlass.utils.mixed_input_helpers`, when the A source is
TMEM with a 128-lane A face but the smem input has only 64 rows, uses `cute.logical_product(layout, ((cute.make_layout(2, stride=0), None),
None, None, None))` to replicate 64 rows into 128 — exactly the first row of the table above (that source was read in 4.7.1; the certification job ran on a 4.6.2 image).

## Gotchas

- **A narrow-type atom with a 32-bit destination view dies on the spot; "marking the destination narrow as well" compiles — the latter is the trap.**
  The constructor's guard compares **element widths**, and only that:
  with a 32-bit destination view and a narrow atom, the widths disagree and it is **rejected at MLIR verification**
  (*expects atom and input memref to have the same element size but got 'bf16' for the
  atom while tmem memref type is 'f32'*), a catchable Python exception that says clearly what is wrong;
  mark the destination tensor **as narrow too** and the widths agree, **the guard lets it through and it compiles**
  \[✅ Verified: the two contracts `narrow_atom_rejected` / `narrow_view_builds` measure exactly these two outcomes].
  So **"it compiled" is no evidence at all on this path** — it only says the two sides have the same element width, not that the words
  land where the 32-bit-column consumer (the next MMA) actually reads. This entry certifies exactly one spelling:
  **the atom built on F32, the narrow face a register-side `cute.recast_tensor` view**; if you change the destination tensor's type,
  measure a readback yourself first.
  Separately, a narrow-type atom on this path has another way to die:
  `LLVM ERROR: unknow intrinsic signature for tcgen05.st`, **which is an LLVM abort;
  `try/except` cannot catch it and it takes the whole process down**. The two failure modes together have one operational consequence:
  when sweeping / comparing spellings on this path, **each variant must run in its own subprocess**, or one bad spelling eats every
  verdict of the whole sweep — the certification probe of this entry is built that way.

- **Store granularity is neutral; do not look for the spill here.** Whole-tensor `.store()` / per-sub-tile slices /
  element-wise writes to the narrow face: on a path **with register headroom** these three compile to **exactly the same** cost:
  `bytes stack frame` all 0, register counts identical. That is, the claim "writing the recast face element-wise is a sub-word write,
  defeats register promotion and sinks the whole thing into local memory" — which sounds plausible — **cannot be measured without
  quota pressure** (with all-constexpr indices the compiler promotes all three granularities).
  The practical corollary: **when you see this segment spill under quota pressure, the first move is not to change the store granularity**;
  the expected gain of that rung of the ladder is zero. Ask the next gotcha's question first.

- **What splitting buys and what it does not (live-set arithmetic).** The register peak of this segment is

  ```
  peak  ≥  |source fragment| (until its last consumer)  +  |destination slice currently in hand|
  ```

  Splitting the destination (mode-2 per-sub-tile pack + per-sub-tile `cute.copy`) shrinks the **second term**, from
  `|dst|` down to `|dst| / number of sub-tiles`. **It cannot touch the first term at all** — the source fragment's live range is set by
  its **last consumer**, independent of how the store is split. So: if the source fragment still has
  consumers after the handoff (row sums, a second reduction, anything that reads the whole row again), **the floor of splitting the destination is `|source|`**;
  expecting it to push the segment peak below `|source|` is arithmetically impossible, not a tuning miss.
  This is measurable: move one consumer of the source to **after** the handoff and make it depend on the readback value (so the compiler cannot
  hoist it), and the register count rises, or a stack appears where there was none — same code, only that one consumer's position differs.
  To move the first term, **the only way is to move that consumer into the pack loop** so the source dies piece by piece; whether that is worth it must be **measured**,
  not reasoned — it also changes the pressure of the surrounding code, and pressure is the independent variable of this segment's cost.

- **Numbers you only see under pressure are not general conclusions.** Run the same PTX through `ptxas` twice: without `-maxrregcount`
  you get one set of numbers, with a quota another. The difference under quota **varies with the surrounding code**; use it only to compare two spellings **within the same round**,
  never carry it across kernels. Diagnosis: `CUTE_DSL_KEEP=ptx,sass CUTE_DSL_DUMP_DIR=<dir>
  CUTE_DSL_LINEINFO=1` to keep the PTX, then `ptxas -arch=sm_100a -v --warn-on-spills
  [-maxrregcount=N]`, read `N bytes stack frame, M bytes spill stores`,
  and use lineinfo to attribute the spill to source lines.

- **The fence is a precondition for visibility, not etiquette.** Before `cute.arch.fence_view_async_tmem_store()`,
  other warps (or the next MMA) cannot see these words; symmetrically, after t2r you need `fence_view_async_tmem_load()`
  before you may assume the whole row is in registers. When the narrow operand is written into **the upper part of the source region itself** (a common way to save TMEM),
  this fence is the only place where the ordering "the whole row must be in registers before it may be overwritten" is enforced.

- **Two call sites, one bijection, so this segment is insensitive to "who owns which row".** t2r and r2t are the same op
  family: the permutation one applies, the other cancels exactly. The upside is that the pack segment needs no shuffle; **the trap** is that even when this segment is
  all green it **does not prove** what the thread/register → row/column mapping is. The first consumer that depends on an **absolute** row or column index
  (masks, boundaries, indices) must ask the copy itself via `partition_S/partition_D(cute.make_identity_tensor(...))`;
  it cannot be inferred from this segment's correctness.

- **Pointer arithmetic is in units of 32-bit columns.** The `P_COL` in `tmem_ptr + P_COL` counts 32-bit columns, not narrow-type
  elements: `NARROW` 16-bit columns occupy `NARROW/2` 32-bit columns. Offsetting by element count silently steps into the neighbouring region,
  and TMEM reports no out-of-bounds access.

- **On the TMEM side, `partition_D` is not "one register per slot": do the coverage accounting and per-slot coordinates with `partition_S`.**
  The destination of r2t is a TMEM tensor, and mode 0 of `partition_D` carries the instruction's own 32 lanes — certified reading
  `tPd = (((8,32),1),1,32):(((1,65536),0),0,8)`; that `32` has stride 65536, i.e. 32 TMEM lanes, not 32 values.
  So `size(partition_D) × r2t.size` overcounts the face 32×, and a trace-time "is it fully written" check based on it inevitably reports
  false errors like `r2t faces ((8, 1), 1, 32) / (((8, 32), 1), 1, 32) / ((16, 1), 1, 32) do not tile 128 x 256 columns`.
  On the source side, one `partition_S` slot is one register: `size(rP32) × r2t.size = 256 × 128 = 32768 =
  128 lanes × 256 columns`; do the accounting on this side.
  The same holds for taking per-slot absolute (lane, column): `partition_S(cute.make_identity_tensor((128, NCOL)))` is **congruent** with the register fragment
  and can be indexed with the same coordinate; building coordinates from `partition_D`'s shape and then indexing `partition_S`'s coordinate tensor raises, at trace time,
  `` expects `coord` and shape of view are weakly congruent ``. An observed failure mode: four consecutive probe rounds were lost on this one point without a single MMA issued:
  two died on this congruence error, one on a self-check that "the source partition and `partition_D` are not congruent", one on the coverage check above;
  the fifth round moved shape, coverage and coordinates all to the `partition_S` side and passed everything in one round.

## Sources

- PTX ISA, chapter "TensorCore 5th Generation Instructions", `tcgen05.ld` / `tcgen05.st`
  (TMEM's 32-bit-column addressing model, the `.x*` repetition family)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols: `cutlass.cute.nvgpu.tcgen05.St32x32bOp`,
  `tcgen05.Ld32x32bOp`, `tcgen05.Repetition.x{8,16,32,64,128}`, `tcgen05.Unpack.NONE`,
  the `tcgen05` TMEM tiled-copy constructor (`(copy_atom, tmem_tensor) -> TiledCopy`,
  written `TMEM_COPY` in the example; exact symbol name at the call sites of the certification probe below),
  `cute.make_copy_atom(op, dtype)`,
  `cute.recast_tensor(t, dtype)`, `cute.make_rmem_tensor(shape, dtype)`,
  `cute.make_identity_tensor(shape)`, `ThrCopy.partition_S/partition_D`,
  `cute.arch.alloc_tmem / relinquish_tmem_alloc_permit / retrieve_tmem_ptr`,
  `cute.arch.fence_view_async_tmem_load / fence_view_async_tmem_store`,
  `cutlass.utils.SmemAllocator`.
- Compile-time observation channel: `CUTE_DSL_KEEP=ptx,sass` + `CUTE_DSL_DUMP_DIR` + `CUTE_DSL_LINEINFO=1`
  to keep PTX/SASS, then `ptxas -v --warn-on-spills [-maxrregcount=N]`.
- GPU certification probe (not included in this release)
  — **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, fourth round of this certification;
  the probe's own judge is "PASS only if all seven pass", so PASS means all seven passed in the same round). The seven contracts:
  `roundtrip_col` / `roundtrip_row` (fill → t2r → pack → r2t → readback, per-column / per-row patterns reproduced exactly,
  both sides addressed through their own copy's identity partition, so a pass = the t2r face and the r2t recast face index the same set of columns);
  `narrow_atom_rejected` (a BF16 atom with a **32-bit** destination view must fail to compile, and the diagnostic must name the element width);
  `narrow_view_builds` (with the destination view marked **narrow as well**, the same atom must **compile** — that the guard compares only element
  width is itself an object of certification; the first half of gotcha ① rests on it);
  `shape_pairing_formula` (ld x{64,32} × st x{32,16}, four cells; congruence falls exactly on the diagonal predicted by `W = E*T/32`);
  `pack_granularity_neutral` (the three spellings must have identical ptxas `stack frame` and register counts,
  i.e. store granularity is not a lever);
  `source_liveness_floor` (moving one consumer of the source to after the handoff, dependent on the readback value, must raise the cost,
  i.e. the live range is the lever).
  At the end the probe has two more **report-only, unjudged** diagnostic lines (`NARROWVIEW off={1,2}xP_COL`): whether words written through a narrow-type destination
  view read back correctly through the 32-bit view, and whether the offset unit of a narrow-type TMEM pointer is 32-bit columns
  or narrow-type elements. They run **after** all verdict lines — a variant that did not run must not block contracts that already passed;
  in the certification round their values only go to the log and support no conclusion in this entry.
- GPU certification probe (the "cta_group::2" section and the `partition_S` gotcha)
  — **Status: ✅ Verified** (`PROBE PASS max_diff=0`, two contracts passed in the same
  round). `S15-tmemA`: A = the input as is, written into the TMEM A face through this entry's 32-bit atom + recast fragment, read by 32
  `(128,96,16)` CTA pair MMAs (B is a K-major library face of 48 columns per CTA); result byte-exact. `S15-tmemA-rope`: the same face with
  interleaved-pair RoPE folded into the last 64 dims, byte-exact. The three lane-layout cells are fact lines of the same job: `dup` exact, `full` `max_diff` 461
  (all 128 rows wrong, error counts by row half / column half 3053 / 3061 and 3061 / 3053), `own` `max_diff` 7137 (all 128 rows wrong,
  column-half counts 0 / 6143); the B face read back by its own coordinates is exact in all three cells, so the error can only be booked on the A face. The data are small integers, so every product and 512-term sum
  is an exact f32 integer and the judge is `max_diff = 0`, not a tolerance. Geometry readings from the same job: `partition_shape_A((128,512)) =
  ((64,16),1,32)`, `make_fragment_A` layout `((128,16),1,32,1):((131072,1),0,16,0)`, A face 256 columns.
