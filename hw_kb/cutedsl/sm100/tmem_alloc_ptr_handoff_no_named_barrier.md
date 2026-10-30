# tmem_alloc_ptr_handoff_no_named_barrier (publish the TMEM base address without a cross-role named barrier: a CTA pair rides on a second cluster barrier; `cta_group::1` can hand off via a count-1 mbarrier instead; plus where to call allocate / free)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell) only**.
>
> **Form A (CTA pair, published by a second cluster barrier) status: ✅ Verified** — certified by a GPU probe (the probe is not included in this release):
> `PROBE PASS max_diff=0.0` (all three contracts addr_publish / cross_role_tmem / nbar_footprint PASS;
> the cluster form's PTX has only barrier 0 and 2 release-form cluster arrives; in the control, the named-barrier form's PTX shows `bar.sync 1, 160`).
> The two errors in gotcha 8 (where to call allocate / free) were read from failed rounds of certification probes (round 2 of this probe, round 2 of another certification probe).
>
> **Form B (`cta_group::1`, smem base-address word + count-1 mbarrier handoff, a single relaxed cluster barrier in the prologue) status: ✅ Verified** —
> `PROBE PASS max_diff=0.0` (all three contracts addr_publish / cross_role_tmem / sync_footprint PASS).
>
> Form B's certification coverage extends only to `cta_group::1` (`TmemAllocator(..., is_two_cta=False)`), allocating once per launch; CTA pairs use form A.

## Purpose

`tcgen05.alloc` is issued by one warp, and the TMEM base address it obtains **is written into a holding word in smem**. Other warps
(the MMA-issue warp, epilogue warps, …) must first "see" this word before they can `retrieve_tmem_ptr`. So after allocation
there must be one synchronisation that publishes the word.

The default usage of `utils.TmemAllocator` is `wait_for_alloc()`, backed by a **named barrier**
whose width is "allocating warp + all consuming warps" (e.g. 32 + 128 = 160 threads). This entry gives two ways to publish without that named barrier:

| Form | prologue cluster rendezvous count | how the base address is published | who waits for the allocation | certification coverage |
|---|---|---|---|---|
| **A** | 2 (the first may be relaxed, the second is release) | allocate between the two rendezvous; the second, release rendezvous publishes the holding word to the whole cluster | everyone (including the TMA producer) | CTA pair (`is_two_cta=True`) |
| **B** | 1 (relaxed, only to make mbarrier init visible) | the allocating warp does `tcgen05.fence::before_thread_sync` → arrive on a **count-1** mbarrier; warps that need the base address do `mbarrier_wait(·, 0)` → `tcgen05.fence::after_thread_sync` → `retrieve_ptr` | only the warps that use TMEM | `cta_group::1` |

**Form A: do a second cluster barrier in the prologue** (`barrier.cluster.arrive` with default release semantics + `barrier.cluster.wait`
with default acquire semantics). This barrier already covers every thread of both CTAs in the cluster; placing it after the holding word is written
lets every warp read the word. After this, the kernel has **no named barrier between the allocating warp and the consuming warps**;
named barriers can be reserved for the epilogue's own internal rendezvous (e.g. 128 threads).

**Form B: smem base-address word + count-1 mbarrier.** Allocation under `cta_group::1` is CTA-local, so publication only needs CTA-scope
release / acquire: mbarrier arrive defaults to `.release.cta` and try_wait defaults to `.acquire.cta`; the write of the base-address word by `tcgen05.alloc` is a
tcgen05 operation, and under the PTX ISA tcgen05 memory consistency model it needs the two fences `before_thread_sync` / `after_thread_sync`
around the thread synchronisation. The prologue keeps only the one cluster rendezvous that makes mbarrier init visible; warps that do not need the base address (such as the TMA producer) never wait for the allocation.


## When needed (trigger)

- The synchronisation contract says "named barriers are only for rendezvous inside one role; every cross-role handoff goes through an mbarrier or a cluster barrier",
  but the kernel's named-barrier width list has an extra "allocating warp + consuming warps" width (e.g. `[[128],[160]]`, while the contract allows only `[[128]]`).
- NCU `stall_barrier` is higher than expected and the stalls concentrate in the prologue segment: the MMA warp and the epilogue warps wait on each other at the
  named barrier of `wait_for_alloc()`.
- A warp that **did not take part** in the named barrier of `wait_for_alloc()` also needs to read the TMEM base address (see gotcha 9).
- The kernel is already in cluster form (CTA pair, `cta_group::2`), and the prologue already has to do one cluster barrier
  to make mbarrier init visible to the whole cluster.
- `free()` raises at trace time `ValueError: num_columns must be between 32 and 512, and must be pow of 2, but got 0` (see gotcha 8).
- (Form B) A `cta_group::1` kernel prologue has **two** full-cluster rendezvous, the second only to publish the TMEM base address; barrier-class stalls
  concentrate on the cluster barrier wait (`UCGABAR_WAIT` in SASS), with almost no CTA-level barrier stalls. PTX criterion: two
  `barrier.cluster.arrive` in the prologue.

**Whole-kernel readings (form A → form B / CTA-local handoff, one `cta_group::1` tree)**:

- On a tree whose whole kernel takes about 9 µs, replacing the second cluster rendezvous with a CTA-local handoff, priced as a same-job lever: adding this change was 4.45% faster (the pricing used
  a named-barrier handoff, not form B); it then landed as form B, gate 0.0094 → 0.0090 ms (two re-measurements 0.0090 / 0.0089 ms).

This is a gate reading on a single tree (not a same-job A/B), not a rule of gain; on other trees one fewer rendezvous in the prologue is not necessarily faster; persistent kernels and CTA pairs have no readings.
Do your own same-job A/B before landing it.

**Anti-trigger**: the kernel is not a cluster launch (no cluster barrier to ride on), or the allocation happens on demand in the middle of the main loop
(not in the prologue). Form A is not usable in either case; form B's mbarrier handoff does not itself depend on clusters, but neither a non-cluster launch nor
allocation outside the prologue is within the certification coverage (gotcha 8 still applies). Do not apply form B to a CTA pair (`is_two_cta=True`): the peer CTA's MMA also uses this
TMEM and dealloc must do a remote arrive on the peer's mbarrier — that is form A's scenario (gotcha 15).

## Minimal runnable example

### Form A: a CTA pair rides on a second cluster barrier

The snippet below shows only the prologue: the allocation sits between two cluster barriers; after the barrier every warp can retrieve the pointer,
and role dispatch comes after the barrier. The spelling matches the cluster form of the certification probe.

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

ALLOC_WARP = 6          # allocating warp (usually the MMA-issue warp itself)
TMEM_COLS  = 512

@cute.struct
class Shared:
    # ... the pipeline's mbarrier arrays ...
    tmem_dealloc_mbar: cutlass.Int64
    tmem_hold: cutlass.Int32

@cute.kernel
def kern(...):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = utils.SmemAllocator()
    st = smem.allocate(Shared)

    # (a) init of other mbarriers / pipeline objects also goes before cluster barrier 1 (the probe has no pipeline objects; this line is uncertified)
    # ...
    # (b) TmemAllocator still requires a NamedBarrier object at construction (it is a mandatory argument).
    #     As long as wait_for_alloc() is never called, this object generates no barrier instruction (the probe's PTX footprint proves it).
    tmem = utils.TmemAllocator(
        st.tmem_hold,
        barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=32 + 128),
        allocator_warp_id=ALLOC_WARP,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc_mbar)   # the constructor inits the pair's dealloc mbarrier
    cute.arch.mbarrier_init_fence()

    # ---- cluster barrier 1: make mbarrier init (including the dealloc mbarrier) visible to the whole cluster ----
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()

    # ---- allocate: call at top level; allocate/relinquish already guard on allocator_warp_id internally ----
    tmem.allocate(TMEM_COLS)            # in pair form, the allocating warps of both CTAs must execute it
    tmem.relinquish_alloc_permit()

    # ---- cluster barrier 2: release/acquire, publishes the holding word to every thread in the cluster ----
    cute.arch.cluster_arrive()          # note: not _relaxed
    cute.arch.cluster_wait()

    tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)   # any warp may retrieve it; no further sync needed

    # ---- role dispatch starts here; no role calls tmem.wait_for_alloc() any more ----
    if warp == ALLOC_WARP:
        ...                                # MMA issue
    if warp < 4:
        ...                                # epilogue; named barriers only inside these 128 threads
    # ---- teardown: first make sure the warps reading TMEM are done (the probe uses one CTA-level barrier),
    #      call free at top level (internally only the allocating warp executes it), then one more cluster barrier.
    #      This is this entry's probe teardown; for the form with a narrowed rendezvous and no teardown cluster barrier, and its preconditions, see
    #      teardown_subset_rendezvous_early_exit example C and gotcha 9 ----
    cute.arch.barrier()
    tmem.free(tmem_ptr)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
```

**Cross-role use after publication**: the allocating warp does `tcgen05.st` with the pointer it retrieved, and the epilogue warp of the same TMEM sub-partition
(same warp index mod 4) does `tcgen05.ld` with the pointer it retrieved; both sides read/write the same set of (lane, column). This was verified bit-exact,
see "Sources".

### Form B: smem base-address word + count-1 mbarrier (`cta_group::1`)

The spelling matches the mbarrier form of the certification probe. Role numbers are symbolic; the two `tcgen05.fence` have no public wrapper in the DSL build used for certification,
so they use inline PTX (spelling in [`ptx_escape`](ptx_escape.md)).

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def tc_fence_before(*, loc=None, ip=None):      # tcgen05.fence::before_thread_sync
    llvm.inline_asm(None, [], "tcgen05.fence::before_thread_sync;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def tc_fence_after(*, loc=None, ip=None):       # tcgen05.fence::after_thread_sync
    llvm.inline_asm(None, [], "tcgen05.fence::after_thread_sync;", "", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

ALLOC_WARP = ...        # TmemAllocator's allocator_warp_id
TMEM_COLS  = ...        # power of 2, 32..512
RETRIEVE_BAR_ID = ...   # generates no instruction when wait_for_alloc() is not called; any unused id

@cute.struct
class Shared:
    # ... the pipeline's mbarriers ...
    tmem_ready: cutlass.Int64          # one-shot: only phase 0 completes per launch
    tmem_hold: cutlass.Int32           # the base-address word written by tcgen05.alloc

@cute.kernel
def kern(...):
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = cute.arch.lane_idx()
    st = utils.SmemAllocator().allocate(Shared)
    if tidx == 0:
        cute.arch.mbarrier_init(st.tmem_ready, 1)       # count 1: one lane of the allocating warp arrives
        # ... other mbarrier init ...
    tmem = utils.TmemAllocator(
        st.tmem_hold,
        barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=RETRIEVE_BAR_ID, num_threads=...),  # mandatory; generates no instruction unless wait_for_alloc() is called
        allocator_warp_id=ALLOC_WARP)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()                  # the only cluster rendezvous in the prologue: just for init visibility
    cute.arch.cluster_wait()

    tmem.allocate(TMEM_COLS)                            # top-level call (internally guarded on allocator_warp_id)
    tmem.relinquish_alloc_permit()
    if warp == ALLOC_WARP:
        tc_fence_before()
        if lane == 0:
            cute.arch.mbarrier_arrive(st.tmem_ready)    # publish the base-address word

    # In every role branch that uses TMEM, before the first use of the pointer (the allocating warp also does this):
    #     cute.arch.mbarrier_wait(st.tmem_ready, 0)
    #     tc_fence_after()
    #     tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
    #     acc = cute.make_tensor(tmem_ptr, ...)          # build TMEM tensors in each branch
    # The TMA producer does not use the base address: it does not wait on tmem_ready and issues its first load right after the rendezvous.
```

**Read → dealloc also goes through an mbarrier** (every hop of the probe's `cross_role_tmem` is this pair): the reading warp does
`fence_view_async_tmem_load()` (`tcgen05.wait::ld`) → `tc_fence_before()` → arrive on an mbarrier (init count = number of threads that actually arrive);
the allocating warp does `mbarrier_wait` → `tc_fence_after()` → `tmem.free(ptr)`. The tmem-empty phase the accumulator pipeline already has can serve as this signal.

## Gotchas

1. **A cluster barrier waits for every non-exited thread of both CTAs in the cluster.** `barrier.cluster.arrive/wait` counts threads
   (DSL default `.aligned`, the whole warp executes together). Putting it at top level, before role dispatch, is safest. It may go inside branches, but the branches together must cover every non-exited thread,
   with warp-uniform branch conditions: two complementary branches each arriving once ([`prologue_mbarrier_critical_path_init`](prologue_mbarrier_critical_path_init.md) gotcha 1)
   and warps that have already exited not being counted ([`teardown_subset_rendezvous_early_exit`](teardown_subset_rendezvous_early_exit.md) gotcha 11) are both GPU-certified.
   If only `if warp == ...` arrives and the other warps neither arrive nor exit, this barrier completes only once they exit (the reading in teardown gotcha 11);
   if they in turn wait on results from the threads in the branch (which is exactly the situation in the prologue), the kernel hangs (the hang was not run as a negative probe).
2. **The second barrier must use `cluster_arrive()`, not `cluster_arrive_relaxed()`.** Per the PTX ISA, `.relaxed` drops the
   release semantics of arrive and does not guarantee that the write of the holding word is visible to reads after the wait. The first barrier (only for mbarrier init visibility) may be relaxed
   because `mbarrier_init_fence()` precedes it. The second barrier publishes ordinary smem data and relies on release/acquire for visibility.
   The probe certified only the release form and did not run a "second one relaxed too" control — relaxed happening to read correctly would not show it is legal.
3. **Do not "keep both paths".** After switching to cluster-barrier publication, if `tmem.wait_for_alloc()` remains in the MMA warp or epilogue branch,
   that 160-thread named barrier is still generated and the width list does not change. The criterion is that the PTX no longer contains
   a non-zero-id `barrier.sync` / `bar.sync` (this is exactly what the probe's `nbar_footprint` checks).
4. **The dealloc mbarrier init must be published by a fenced cluster barrier before the peer's first remote arrive.** In pair form, `free()` does a remote arrive on the peer CTA's dealloc mbarrier,
   so this mbarrier's init must complete before the first cluster barrier; `TmemAllocator` does the init at construction
   (public source: in the constructor the allocating warp does `mbarrier_init` followed by `mbarrier_init_fence`), so **construction must come before barrier 1**.
   When the prologue is split into two groups, construction may also come after barrier 1 and before barrier 2's fence + release arrive (the peer's first remote arrive is in the teardown
   `free()`); that position is separately certified, see [`prologue_mbarrier_critical_path_init`](prologue_mbarrier_critical_path_init.md) gotcha 7.
   This entry's probe also does one cluster barrier at teardown (`cluster_arrive()` + `cluster_wait()`) to avoid one CTA exiting first while the peer is still
   arriving on its mbarrier. It is not unconditionally required: in a pair kernel where "every cross-CTA operation is waited on by some warp of this CTA",
   removing it is certified (real pipeline in flight, the same compiled entry launched 3 times in a row, bit-exact); the spelling and the preconditions to check one by one are in
   [`teardown_subset_rendezvous_early_exit`](teardown_subset_rendezvous_early_exit.md) example C and gotcha 9; keep it when the preconditions do not hold or cannot be checked.
5. **In pair form both CTAs must execute the allocation.** With `is_two_cta=True` the allocating warps of both CTAs must execute `allocate`;
   `retrieve_ptr` in each CTA reads that CTA's own holding word.
6. **(Form A) Cost: the TMA producer's first load is delayed until after the allocation.** The second barrier is a full-cluster rendezvous, so the producer warp
   also has to wait for the allocation to finish before issuing its first TMA. This happens only once, in the prologue, in exchange for removing one cross-role named barrier.
   In-probe readings (`cta_group::1`, `clock64` median counted from leaving the first cluster rendezvous): a warp that does not need the base address can go issue loads after 45 cycles under form B,
   709 cycles under the "second cluster rendezvous" control; a warp that obtains the base address 382 vs 845 cycles. Whole-kernel readings are under "When needed".
   For a CTA pair, the split-phase spelling that keeps the producer from waiting for the allocation (the producer arrives at the second barrier first, and waits only after issuing the first tile's TMA) and its in-probe
   cycle readings are in [`prologue_mbarrier_critical_path_init`](prologue_mbarrier_critical_path_init.md); that entry's whole-kernel same-job A/B is flat.
   If prologue latency itself is the bottleneck, measure first, then decide.
7. **The warps doing `tcgen05.st` and `tcgen05.ld` must map to the right sub-partition.** A warp can only access lanes `32*(warp%4) .. +31`.
   In the example, allocating warp 6 and epilogue warp 2 access the same lanes 64..95. The tiled copy takes its slice with `get_slice(tidx % 128)`,
   and the two warps get the same bijection.
8. **Call `tmem.allocate()` / `relinquish_alloc_permit()` / `free()` at top level; do not wrap them in a runtime `if warp == ...`.**
   These methods already guard on `allocator_warp_id` internally. The number of allocated columns is kept in a Python-side field of the allocator object
   (used by `free()` when no column count is passed); when `allocate()` is written inside a branch of a runtime `if` and `free()` is outside that branch (at top level or in another branch),
   that count does not leave the branch,
   `free()` deallocs with 0 columns and raises `ValueError: num_columns must be between 32 and 512, and must be pow of 2, but got 0`.
   This error was read in round 2 of the certification probe (the control form wrote `allocate` inside `if warp == ALLOC_WARP:`).
   Fix: move `allocate` + `relinquish_alloc_permit` back to top level (round 3 of the certification probe, uses exactly this spelling).
   **Passing the column count explicitly with `free(ptr, num_columns)` does not get around it**: in the public source `free` first asserts `num_columns <= the recorded column count`,
   so it still fails when the record is 0; the error read on the GPU is `num_columns must be less than or equal to num_allocated_columns`
   (round 2 of another probe's certification, see [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md) gotcha 9).
9. **`wait_for_alloc()` publishes the base address only to the threads taking part in that named barrier.** In the control group of the certification job the named barrier covered only the allocating warp
   and the 4 epilogue warps (160 threads); 3 other warps did not take part and read the holding word directly: across 3 launches they **read the pre-allocation stale value every time**
   (sentinel hits 42624 = 96 threads × 148 CTAs × 3 launches), while every participating thread read correctly. So if a warp outside the named barrier, such as the producer,
   also needs the base address, either add it to the named barrier (its width changes accordingly) or publish with this entry's cluster barrier / mbarrier.
10. **Scalar fields in a struct may themselves be pointers.** `tmem_hold` / `tmem_dealloc_mbar`, declared as `cutlass.Int32` / `cutlass.Int64` in `Shared`,
    are already pointers in the DSL environment used for certification: writing `st.tmem_hold.ptr` raises at trace time
    `AttributeError: '_Pointer' object has no attribute 'ptr'` (certification round 1). Pass the field itself as in the example;
    if in your DSL version the field is a wrapper object and you cannot get the pointer without `.ptr`, follow that version's spelling.
11. **(Form B) Write the two `tcgen05.fence` as a pair.** The allocating side does `tcgen05.fence::before_thread_sync` before the arrive;
    the waiting side does `tcgen05.fence::after_thread_sync` after the wait and before `retrieve_ptr` / the first tcgen05 instruction.
    The basis is the PTX ISA tcgen05 memory consistency model; the certification probe carries both fences and
    **did not test a "fences removed" control**.
12. **(Form B) Count, number of arrives, and phase.** Init count = number of threads that actually arrive: if only one lane arrives (`elect_one()` or
    `lane == 0`), init to 1; if all 32 lanes of a warp arrive but the init is 1, phase 0 flips early and the extra arrivals land in the next phase.
    This mbarrier completes only once per launch, so waiters always use phase 0. If a persistent kernel reallocates per tile, the phase must flip with the round —
    the certification covers only "allocate once per launch".
13. **(Form B) Every warp that reads the base address must wait itself first.** Do not `retrieve_ptr` at top level (before the wait) and then pass the pointer into the branches — that is the same
    "reading the base-address word without synchronisation" as gotcha 9 (in gotcha 9's control, threads that read without taking part in the sync read the pre-allocation stale value on every launch; sentinel hits 42624).
    Waiting again on the already completed phase 0 passes immediately and costs almost nothing;
    and do not count on the transitive chain "the MMA warp already waited, and the acc-full the epilogue waits on was committed by it, so the epilogue sees it too" — it has not been verified.
14. **(Form B) mbarrier init must still precede `mbarrier_init_fence()` and that cluster rendezvous.** If init comes after the rendezvous, other warps may
    wait on an uninitialised object.
15. **Form B is certified only for `cta_group::1`.** In a pair allocation with `is_two_cta=True`, the peer CTA's MMA also uses this TMEM, dealloc does a remote arrive on
    the peer's dealloc mbarrier, and the dealloc mbarrier init at construction must also precede the first cluster rendezvous (gotcha 4) —
    none of these is in form B's probe. Pairs use form A.

## Sources

- PTX ISA: `barrier.cluster` (arrive defaults to `.release`, wait defaults to `.acquire`, `.relaxed` drops the release semantics, `.aligned`)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster
- PTX ISA: `tcgen05.alloc` / `tcgen05.dealloc` / `tcgen05.relinquish_alloc_permit` (address written into smem, `.cta_group::2`)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit
- Public infra API symbols: `cutlass.utils.TmemAllocator` (`allocate` / `relinquish_alloc_permit` / `retrieve_ptr` / `free` /
  `wait_for_alloc`), `cutlass.pipeline.NamedBarrier`, `cute.arch.cluster_arrive` / `cluster_arrive_relaxed` / `cluster_wait`,
  `cute.arch.fence_view_async_tmem_load`,
  `tcgen05.St32x32bOp` / `Ld32x32bOp` / `make_tmem_copy`.
- GPU certification probe (form A): verdict `PROBE PASS max_diff=0.0` (certification round 3), three contracts:
  - `addr_publish`: after the second barrier, 256 threads read the same holding word, and none ever reads the sentinel value;
  - `cross_role_tmem`: the allocating warp writes, the epilogue warp reads, bit-exact; configuration 148 CTAs, cluster (2,1,1), 3 consecutive launches;
  - `nbar_footprint`: the cluster form's PTX has no non-zero-id named barrier, and at least 2 release-form cluster arrives
    (readings: `bar_ids=['0']`, 2 release arrives, 1 relaxed); the named-barrier form as the control reads `bar.sync 1, 160`.
- PTX ISA: `tcgen05.fence::before_thread_sync` / `after_thread_sync` (tcgen05 memory consistency model), `mbarrier.arrive` (default `.release`) /
  `mbarrier.try_wait` (default `.acquire`), `.cta` scope
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-memory-consistency-model
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier
- Public infra API symbols (added for form B): `cute.arch.mbarrier_init` / `mbarrier_init_fence` / `mbarrier_arrive` / `mbarrier_wait`,
  `cute.arch.elect_one`, `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`.
- GPU certification probe (form B): verdict `PROBE PASS max_diff=0.0` (certification round 1); `cta_group::1`, 1 CTA per SM,
  the same compiled entry launched 3 times in a row. Two variants: `mbar` (form B) and the control `cluster2` (a second release cluster rendezvous). Three contracts:
  - `addr_publish`: after the count-1 mbarrier wait, every thread of each CTA reads the same base-address word and never reads the sentinel;
  - `cross_role_tmem`: decoy write (reading warp) → `tcgen05.st` (allocating warp) → `tcgen05.ld` (reading warp) → dealloc (allocating warp), each hop one
    mbarrier + one pair of `tcgen05.fence`, bit-exact;
  - `sync_footprint`: form B's PTX has no `bar.sync` / `barrier.sync` at all, 0 release cluster arrives, 1 relaxed, 1 cluster wait;
    the control has 1 release, 1 relaxed, 2 waits.
  - Prologue `clock64` readings in gotcha 6.
- Error readings for gotcha 8: round 2 of this probe; the `free(ptr, n)` error read from round 2 of the certification probe of `tmem_wide_readback_stsm_mn_box`.
