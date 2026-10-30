# elect_one (single-thread election: the CuTe DSL spelling and scope of PTX `elect.sync`)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry covers only **how to pick one thread within a warp to issue some instruction**. Who should
> be picked to do what belongs to the respective interface: mbarrier in
> [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md), the role split of the register budget in
> [`warp_specialization`](warp_specialization.md).

## Purpose

SM90 has a set of instructions that are correct only **when issued by a single thread**
(`mbarrier.init`, `mbarrier.expect_tx`, `mbarrier.arrive.expect_tx`, the raw-PTX `cp.async.bulk`
family …): issued by the whole warp they are issued 32 times, and the semantics are multiplied by 32
(doubled counts / doubled reductions) — **no error, silently wrong**.

`elect.sync` is the hardware's election primitive: it picks **exactly one** thread from the warp's
active thread set (which one is picked is not architecturally guaranteed), and the election itself has
`.sync` semantics. CuTe DSL wraps it as a **context manager**.

## When needed (trigger)

- You need to issue an instruction that "may only be issued once per CTA / per warp", but the whole warp
  (or the whole warpgroup) reaches the current code path.
- **The decisive symptom on the DSL side: `AttributeError: module 'cutlass.cute.arch' has no attribute
  'elect_one_sync'`** (or `... has no attribute 'elect_sync'`). `elect_one_sync()` is the **CUTLASS
  C++** name (`cutlass::arch::elect_one_sync()`, returning a bool predicate); **this name does not exist
  in the Python DSL**, nor does any election function that returns a predicate. The DSL only exports
  `elect_one`, and it **is not a predicate, it is a `with` block**. Writing
  `if cute.arch.elect_one_sync():` in the C++ shape inevitably raises AttributeError.
- The opposite symptom: some count/reduction result is **exactly 32 times the correct value** (or an
  mbarrier never completes), meaning an instruction that should be issued by a single thread was issued
  by the whole warp.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute

# ---- 1) Basic form: pick one thread within the warp -------------------
#      elect_one() returns an IfOpRegion context manager and can only be used with `with`.
#      It is WARP-scoped: a 128-thread warpgroup reaching here puts 4 threads into the block.
with cute.arch.elect_one():
    ...            # exactly one thread per warp executes

# ---- 2) One thread for the whole CTA: warp predicate [outside] + elect_one [nested inside] ----
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:                      # the outer predicate must be warp-uniform
    with cute.arch.elect_one():
        ...        # exactly one thread in the whole CTA executes

# ---- 3) Typical mandatory use: mbarrier initialisation ----------------
smem = cutlass.utils.SmemAllocator()
mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)

if warp_idx == 0:
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(mbar, NTHREADS)   # single-thread instruction
cute.arch.mbarrier_init_fence()   # called by everyone: makes the init visible to this CTA
cute.arch.sync_threads()          # a CTA sync is required between init and first use

# Normal use afterwards: arrive is release, wait is acquire
cute.arch.mbarrier_arrive(mbar)
cute.arch.mbarrier_wait(mbar, 0)  # phase 0
```

Form comparison table (misspell any of them and you get AttributeError or a type error):

| What you want | CUTLASS C++ | CuTe DSL |
|---|---|---|
| pick one thread within the warp | `if (elect_one_sync()) {...}` | `with cute.arch.elect_one(): ...` |
| tell the compiler a value is warp-uniform | `__shfl_sync(mask, v, 0)` etc. | `cute.arch.make_warp_uniform(v)` |
| the fence for mbarrier init | `cutlass::arch::fence_barrier_init()` | `cute.arch.mbarrier_init_fence()` |

The **complete** set of election-related exports in `cute.arch` is just two: `elect_one`,
`make_warp_uniform` (the `# elect.py` section of `__all__` in `cutlass/cute/arch/__init__.py`). Not
finding any other name is not a version problem; the name simply does not exist.

## Gotchas

1. **`elect_one` is not a predicate, it is a `with` block.** It returns `IfOpRegion` (`cute/arch/elect.py`
   L46-75), whose `__enter__` pushes the MLIR insertion point into the then-block of an `scf.IfOp`.
   Writing `if cute.arch.elect_one():` does not elect — the `IfOpRegion` object is always truthy, the
   whole warp goes in, and the IR inside the block is built at the wrong insertion point. There is **no**
   variant that returns a `Boolean`.
2. **It is warp-scoped, not CTA-scoped.** In a 128-thread warpgroup, `with elect_one():` lets **4**
   threads in (one per warp), not 1. This is the number-one reason for "why is my mbarrier count 4×".
   For a single thread in the whole CTA you must wrap a warp predicate around it as in example 2 — which
   is how CUTLASS itself writes it (`cutlass/pipeline/helpers.py`
   `mbarrier_init`: `if_generate(warp_idx == 0, then_body)`, and only inside `then_body` is
   `with cute.arch.elect_one():`).
3. **Values produced inside the region cannot leave it.** `IfOpRegion.__exit__` only emits
   `scf.yield_([])` — this `scf.IfOp` **has no result**. So
   ```python
   with cute.arch.elect_one():
       v = tidx + 7        # v is defined inside the region
   mOut[tidx] = v          # <<< rejected at compile time (dominance does not hold)
   ```
   Python's `with` does not open a new scope, so the name is visible, but **it does not dominate in the
   IR**. To give what the elected thread computed to the whole warp there are only two ways: move the
   consumer into the region as well; or broadcast explicitly via smem/`shuffle_sync` (paired with a
   sync). Don't count on "assign inside the block, use outside".
4. **The outer predicate must be warp-uniform.** The DSL's `elect_one()` calls the argument-less
   `_nvvm.elect_sync()` (CuTeDSL `elect.py` L147); **there is no membermask argument to pass**, i.e. the
   election targets the whole warp. If threads within the warp have diverged at `with elect_one():`
   (the predicate depends on `tidx % 32` / on data), you fall outside PTX `elect.sync`'s requirement
   that membermask and the active thread set agree. Use only quantities at warp granularity or coarser
   for the outer predicate: `make_warp_uniform(warp_idx())`, `tidx // 32`, `tidx < 64` and the like.
5. **Don't put `setmaxnreg` dec/inc inside `elect_one`.** That instruction is `.sync.aligned` and
   requires all 128 threads of the warpgroup to execute it; stuffing it into a single-thread block is a
   deadlock / undefined, not an optimisation. See [`warp_specialization`](warp_specialization.md)
   gotcha 3.
6. **Don't wrap a TMA copy in `elect_one` — it deadlocks.** The DSL's TMA copy atom already does a
   warp-level elect internally (membermask = whole warp); wrapping a single-thread block around it
   leaves part of the warp unable to enter while the other part waits: the official docstring says it
   outright, "Wrapping in `elect_one()` can cause GPU deadlock" (`cute/nvgpu/cpasync/copy.py` L296-333,
   L466-489, L820-842 of cutlass). The correct form for the TMA path is `if warp_idx == K:` (the whole
   warp goes in), see [`tma_store_epilogue`](tma_store_epilogue.md) gotcha 6.
   **Conversely**, the **raw PTX** `cp.async.bulk` / `cp.reduce.async.bulk` of the same family are
   per-thread instructions with no internal elect, and you must pick one thread yourself
   (`if tidx == 0:` or this entry) — see [`s2g_bulk_reduce`](s2g_bulk_reduce.md) gotcha 3. **"Whether to
   elect" depends on whether you go through the DSL copy atom or raw PTX, not on the instruction name.**
7. **Which thread gets picked is not guaranteed.** `elect.sync` only promises "exactly one active
   thread", not lane 0. Any form of "the picked one is always lane 0, so lane 0's private value can be
   used directly" is wrong; if you need a specific lane write `if lane_idx() == 0:` (that is an ordinary
   predicate, not an election), and use `elect_one` only when you need "any one". This entry's
   certification probe therefore only checks **the number of picked threads** and that thread's own
   data, never assuming which lane it is.
8. **Between `mbarrier_init` and first use there must be `mbarrier_init_fence()` + a CTA sync.** The
   election only guarantees "one thread did the init", not that the other threads can see it. Order:
   `[elect] mbarrier_init → everyone mbarrier_init_fence() → everyone sync_threads() → use`.
   Leaving out the sync raises no error; it turns into occasional hangs.

## Sources

- **DSL spelling and semantics (source, line by line)**: cutlass-dsl 4.6.1
  `cutlass/cute/arch/elect.py` — `make_warp_uniform` L23-43; `IfOpRegion` L46-75
  (`__exit__` emits `scf.yield_([])`, i.e. the region has no result → gotcha 3); `elect_one` L78-149
  (`@dsl_user_op`, `check_arch(>= sm_90)`, `_nvvm.elect_sync()` with no membermask argument
  → gotcha 4, returns `IfOpRegion` → gotcha 1). Export list
  `cutlass/cute/arch/__init__.py` L24-29 (the `# elect.py` section of `__all__` has only
  `make_warp_uniform` / `elect_one`).
- **The official "must be single-thread" list**: the docstring of `cute/arch/elect.py` L91-127 in the
  same version (required for `mbarrier_init` / `mbarrier_expect_tx` /
  `mbarrier_arrive_and_expect_tx` / tcgen05 commit; **NOT** for TMA copy),
  and `cute/arch/mbar.py` L34-64 (the "**Single-Thread Execution Required**" section of
  `mbarrier_init`), L106-132, L173-194.
- **Official wording for gotcha 6**: same version, `cute/nvgpu/cpasync/copy.py` L296-333 / L466-489 /
  L820-842, `cute/nvgpu/tcgen05/copy.py` L833-1000 — "TMA ... implicitly adding
  `elect_one()` around the copy operation", "WRONG: Do NOT wrap in elect_one
  (can cause deadlock)".
- **Official spelling for gotcha 2**: same version, `cutlass/pipeline/helpers.py` L328-352
  (`mbarrier_init`: `warp_idx = make_warp_uniform(warp_idx())` →
  `if_generate(warp_idx == 0, then_body)` → inside `then_body`, `with cute.arch.elect_one():`),
  and L340-347 / L440-467 (`arrive_tcgen05mma` / `arrive_and_expect_tx` both use
  `with cute.arch.elect_one():`).
- **PTX ISA**: *Parallel Synchronization and Communication Instructions:
  `elect.sync`*; for mbarrier's arrive=release / try_wait=acquire semantics see the mbarrier section of
  the same chapter.
- **GPU-certified** (by a GPU probe, not included in this release) — single CTA, 128 threads (4 warps), six
  rungs each in its own subprocess:
  R0 no-election control group (per-warp count == 32);
  R1 per-warp picked count of `with elect_one():` == **[1,1,1,1]** (if it were CTA-scoped it would read
  [1,0,0,0]; the two hypotheses have separated criteria → gotcha 2);
  R2 per-warp count of `if warp_idx == 0:` + `elect_one` == [1,0,0,0];
  R3 `elect_one → mbarrier_init → mbarrier_init_fence → sync_threads → single-thread smem write →
  everyone arrive/wait → read back`, all 128 threads read the published value (checks both no deadlock
  and visibility → gotcha 8);
  R4 produced inside the region, consumed inside the region (the legal form; it only checks "exactly one
  non-zero per warp, equal to its own tid+7", without assuming the lane → gotcha 7);
  R5 the same code with the use moved outside the region **must be rejected at compile time** (gotcha
  3), with R4 as its control group; a non-numeric contract counts as `max_diff=n/a`.
  **Conclusion: all six rungs `PROBE PASS` on the GPU (`max_diff=0.0`) → this entry is certified ✅
  (`sm90/index.md` is marked ✅ Verified accordingly).**
- This entry contains no operator algorithm logic: all of it is the spelling, scope and synchronisation
  order of the thread-election primitive.
