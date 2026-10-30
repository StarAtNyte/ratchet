# elect_one (single-thread election: the CuTe DSL spelling and scope of PTX `elect.sync`)

> **Certification source: the sm90 tree; the do-not-wrap rule and the boundary for the tcgen05 family were re-checked on sm100
> by a GPU probe (the probe is not included in this release); the rest was not re-checked.**
> Rules and mechanism are the same across the two generations; **the coverage is not**: the tcgen05 `Cp*` /
> `alloc` / `dealloc` in the "must not wrap" column of the table below and `tcgen05.commit` in the "must wrap" column
> are all new on Blackwell; the criterion is still that one warning sentence in the docstring.
>
> One entry per hardware interface. Operator-agnostic; mechanism only.
> This entry covers only **how to pick one thread in a warp to issue a given instruction**. Who should be picked to do
> what belongs to each interface's own entry: for mbarrier see `sm90/wgmma_tma_mbarrier`;
> for the role split under a register budget see `sm90/warp_specialization`.
> Names of the form `sm90/xxx` are entries of the Hopper tree and **are not in the read set of an sm100 run**: the loop
> hands a run only the one tree `hw_kb/cutedsl/<kb_arch>/` (`loop.toml [loop] kb_arch`). The names are written out so that you
> know they exist; every reading in them was taken on Hopper and **is not a conclusion for this architecture**; if you
> need one, re-earn it here.

## Purpose

`elect.sync` is the election primitive the hardware provides: from the active thread set of a warp it picks **exactly
one** thread (which one is not architecturally guaranteed), and the election itself carries `.sync` semantics. CuTe DSL
wraps it as a **context manager**.

### Read this first: wrapping it wrong is a silent failure in both directions

**"Whether to wrap in `elect_one`" depends on whether you go through a DSL copy/mma atom or raw PTX, not on the
instruction name.** The cost is asymmetric in the two directions, but neither raises an error:

| | Symptom | Examples |
|---|---|---|
| **Should wrap but did not** | The whole warp issues it 32 times, the semantics are multiplied by 32 (counts / reductions multiplied), or the mbarrier waits forever | `mbarrier.init`, `mbarrier.arrive.expect_tx`, `tcgen05.commit`, the raw-PTX `cp.async.bulk` family |
| **Should not wrap but did** | **Deadlock**. The atom already elects internally (membermask = the whole warp); the outer single-thread block keeps the other 31 lanes out of the atom's own `elect.sync` | DSL TMA copy atoms, tcgen05 `Cp*` (s2t / t2r / r2s) and `alloc` / `dealloc` |

The criterion is not the instruction name but **whether the official docstring carries that do-not-wrap warning** -- an
atom whose docstring has the warning necessarily does its own internal election; the two always appear together. The
item-by-item boundary and its sources are in Gotchas 6 below; the tcgen05 family in the table above was measured on sm100,
not inferred from Hopper.

## When needed (trigger)

- You need to issue an instruction that "may only be issued once per CTA / per warp", and on the current code path the
  whole warp (or the whole warpgroup) reaches it.
- **The decisive symptom on the DSL side: `AttributeError: module 'cutlass.cute.arch' has no attribute
  'elect_one_sync'`** (or `... has no attribute 'elect_sync'`). `elect_one_sync()`
  is the **CUTLASS C++** name (`cutlass::arch::elect_one_sync()`, returning a bool predicate);
  **that name does not exist in the Python DSL**, nor does any election function that returns a predicate. The DSL
  exports only `elect_one`, and it is **not a predicate but a `with` block**. Writing
  `if cute.arch.elect_one_sync():` in the shape of the C++ code necessarily raises AttributeError.
- The opposite symptom: some count / reduction result is **exactly 32 times the correct value** (or an mbarrier waits
  forever), which means an instruction that should have been issued by one thread was issued by the whole warp.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute

# ---- 1) Basic form: pick one thread in the warp ------------------------
#      elect_one() returns an IfOpRegion context manager; it can only be used with `with`.
#      It is WARP-scoped: a 128-thread warpgroup reaching here lets 4 threads into the block.
with cute.arch.elect_one():
    ...            # exactly one thread per warp executes this

# ---- 2) Only one thread in the whole CTA: warp predicate [outside] + elect_one [inside] ----
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:                      # the outer predicate must be warp-uniform
    with cute.arch.elect_one():
        ...        # exactly one thread of the whole CTA executes this

# ---- 3) Typical mandatory use: mbarrier initialization --------------------------------
smem = cutlass.utils.SmemAllocator()
mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)

if warp_idx == 0:
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(mbar, NTHREADS)   # single-thread instruction
cute.arch.mbarrier_init_fence()   # called by everyone: makes the init visible to this CTA
cute.arch.sync_threads()          # a CTA sync must separate initialization from first use

# Normal use afterwards: arrive is release, wait is acquire
cute.arch.mbarrier_arrive(mbar)
cute.arch.mbarrier_wait(mbar, 0)  # phase 0
```

Form comparison table (misspelling any of them gives an AttributeError or a type error):

| What you want | CUTLASS C++ | CuTe DSL |
|---|---|---|
| Pick one thread within a warp | `if (elect_one_sync()) {...}` | `with cute.arch.elect_one(): ...` |
| Tell the compiler a value is uniform within the warp | `__shfl_sync(mask, v, 0)` etc. | `cute.arch.make_warp_uniform(v)` |
| The fence for mbarrier init | `cutlass::arch::fence_barrier_init()` | `cute.arch.mbarrier_init_fence()` |

The **complete** set of election-related exports in `cute.arch` is two names: `elect_one`, `make_warp_uniform`
(the `# elect.py` section of `__all__` in `cutlass/cute/arch/__init__.py`). Not finding another name is not a version
problem; the name simply does not exist.

## Gotchas

1. **`elect_one` is not a predicate; it is a `with` block.** It returns `IfOpRegion` (`cute/arch/elect.py`
   L46-75), whose `__enter__` moves the MLIR insertion point into the then-block of an `scf.IfOp`. Writing
   `if cute.arch.elect_one():` does not elect -- the `IfOpRegion` object is always truthy, the whole warp enters,
   and the IR inside the block is built at the wrong insertion point. There is **no** variant that returns a `Boolean`.
2. **It is warp-scoped, not CTA-scoped.** In a 128-thread warpgroup,
   `with elect_one():` lets **4** threads in (one per warp), not 1. This is the number-one cause of "why is my
   mbarrier count 4x". To get a single thread per CTA you must wrap it in an outer warp predicate as in example 2 --
   that is how CUTLASS itself writes it (`cutlass/pipeline/helpers.py`
   `mbarrier_init`: `if_generate(warp_idx == 0, then_body)`, and only inside `then_body` is
   `with cute.arch.elect_one():`).
3. **Values produced inside the region cannot leave the region.** `IfOpRegion.__exit__` only emits
   `scf.yield_([])` -- this `scf.IfOp` **has no result**. So
   ```python
   with cute.arch.elect_one():
       v = tidx + 7        # v is defined inside the region
   mOut[tidx] = v          # <<< rejected at compile time (dominance does not hold)
   ```
   Python's `with` does not open a new scope, so the name is visible, but **in the IR it does not dominate**. To give
   the whole warp something the elected thread computed there are only two ways: move the consumer into the region too;
   or broadcast explicitly through smem / `shuffle_sync` (plus a sync). Do not count on "assign inside the block, use
   outside the block".
4. **The outer predicate must be warp-uniform.** The DSL's `elect_one()` calls the parameterless
   `_nvvm.elect_sync()` (CuTeDSL `elect.py` L147); **there is no membermask parameter to pass**, i.e. the election
   targets the whole warp. If the threads of the warp are divergent at `with elect_one():` (the predicate depends on
   `tidx % 32` / on data), you fall outside PTX `elect.sync`'s requirement that the membermask agree with the active
   thread set. Use only quantities of warp granularity or coarser in the outer predicate: `make_warp_uniform(warp_idx())`,
   `tidx // 32`, `tidx < 64` and the like.
5. **Do not put the `setmaxnreg` dec/inc inside `elect_one`.** That instruction is `.sync.aligned`:
   all 128 threads of the warpgroup must execute it; stuffing it into a single-thread block is a deadlock / undefined
   behaviour, not an optimization. See `sm90/warp_specialization` gotcha 3.
6. **Do not wrap a TMA copy in `elect_one` -- it deadlocks.** The DSL TMA copy atom already does a warp-level elect
   internally (membermask = the whole warp); wrapping it in another single-thread block means part of the warp cannot
   get in while the other part waits: the official docstring says outright "Wrapping in `elect_one()` can cause
   GPU deadlock" (`cute/nvgpu/cpasync/copy.py` L296-333, L466-489, L820-842). The correct form on the TMA path is
   `if warp_idx == K:` (the whole warp enters), see
   `sm90/tma_store_epilogue` gotcha 6.
   **Conversely**, the **raw-PTX** `cp.async.bulk` / `cp.reduce.async.bulk` of the same family are per-thread
   instructions without an internal elect, and you must pick one thread yourself (`if tidx == 0:` or this entry) -- see
   [`s2g_bulk_reduce`](../sm90/s2g_bulk_reduce.md) gotcha 3. **"Whether to elect" depends on whether you go through a
   DSL copy atom or raw PTX, not on the instruction name.**
7. **Which thread is picked is not guaranteed.** `elect.sync` only promises "exactly one active thread", not lane 0.
   Any code of the form "the picked thread is always lane 0, so lane 0's private value can be used directly" is wrong;
   if you need a specific lane write `if lane_idx() == 0:` (an ordinary predicate, not an election); use `elect_one`
   only when "any one" will do. For the same reason this entry's certification probe checks only **the number of
   picked threads** and the picked thread's own data, and never assumes which lane it is.
8. **`mbarrier_init` and its first use must be separated by `mbarrier_init_fence()` + a CTA sync.** The election only
   guarantees "one thread did the init", not that the other threads can see it. Order:
   `[elect] mbarrier_init → everyone mbarrier_init_fence() → everyone sync_threads() → use`.
   Missing the sync raises no error; it turns into an intermittent hang.

## Sources

- **DSL spelling and semantics (source, line by line)**: cutlass-dsl 4.6.1
  `cutlass/cute/arch/elect.py` -- `make_warp_uniform` L23-43; `IfOpRegion` L46-75
  (`__exit__` emits `scf.yield_([])`, i.e. the region has no result → gotcha 3); `elect_one` L78-149
  (`@dsl_user_op`, `check_arch(>= sm_90)`, `_nvvm.elect_sync()` without a membermask parameter
  → gotcha 4, returns `IfOpRegion` → gotcha 1). Export list
  `cutlass/cute/arch/__init__.py` L24-29 (the `# elect.py` section of `__all__` has only
  `make_warp_uniform` / `elect_one`).
- **The official list of "must be single-thread"**: the docstring of the same version's `cute/arch/elect.py` L91-127
  (required for `mbarrier_init` / `mbarrier_expect_tx` /
  `mbarrier_arrive_and_expect_tx` / tcgen05 commit; **NOT** for TMA copy),
  and `cute/arch/mbar.py` L34-64 (the "**Single-Thread Execution
  Required**" section of `mbarrier_init`), L106-132, L173-194.
- **The official wording for gotcha 6**: same version `cute/nvgpu/cpasync/copy.py` L296-333 / L466-489 /
  L820-842, `cute/nvgpu/tcgen05/copy.py` L833-1000 -- "TMA ... implicitly adding
  `elect_one()` around the copy operation", "WRONG: Do NOT wrap in elect_one
  (can cause deadlock)".
- **The official form for gotcha 2**: same version `cutlass/pipeline/helpers.py` L328-352
  (`mbarrier_init`: `warp_idx = make_warp_uniform(warp_idx())` →
  `if_generate(warp_idx == 0, then_body)` → inside `then_body`, `with cute.arch.elect_one():`),
  and `cutlass/pipeline/helpers.py` L340-347 / L440-467 (`arrive_tcgen05mma` / `arrive_and_expect_tx` both use
  `with cute.arch.elect_one():`).
- **PTX ISA**: *Parallel Synchronization and Communication Instructions:
  `elect.sync`*; for the mbarrier arrive=release / try_wait=acquire semantics see the mbarrier section of the same chapter.
- **GPU-certified**: the probe -- a single CTA of 128 threads (4 warps),
  six rungs, each in its own subprocess:
  R0 no-election control group (per-warp count == 32);
  R1 the per-warp picked count of `with elect_one():` == **[1,1,1,1]** (if it were CTA-scoped it would read
  [1,0,0,0]; the criterion separates the two hypotheses → gotcha 2);
  R2 the per-warp count of `if warp_idx == 0:` + `elect_one` == [1,0,0,0];
  R3 `elect_one → mbarrier_init → mbarrier_init_fence → sync_threads → single-thread smem write →
  everyone arrive/wait → readback`, all 128 threads read the published value (checks both no deadlock and visibility → gotcha 8);
  R4 produced inside the region, consumed inside the region (the legal form; it checks only "exactly one nonzero per
  warp, and that item equals its own tid+7", without assuming the lane → gotcha 7);
  R5 the same code with the use moved outside the region **must be rejected at compile time** (gotcha 3), with R4 as
  the control group; a non-numeric contract counted as `max_diff=n/a`.
  **Conclusion: all six rungs `PROBE PASS` on the GPU (`max_diff=0.0`) → this entry is certified ✅
  **
- This entry contains no operator algorithm logic: all of its content is the spelling, scope and synchronization order
  of the thread-election primitive.
