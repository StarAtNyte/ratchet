# teardown_subset_rendezvous_early_exit (the teardown rendezvous before the TMEM release pulls in only the warps that read TMEM: `bar.arrive` + `bar.sync` on a fixed-width named barrier; the other role warps exit as soon as their own `tail()` is done)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**, kernels with warp specialization + a TMEM accumulator + a CTA pair (`cta_group::2`).
>
> **Status: ✅ Verified (correctness, PTX footprint and exit timing only)** — GPU probes (not included in this release):
> - Examples A / B (the subset rendezvous itself): `PROBE PASS max_diff=0.0` (passed on the first round of this certification; all four contracts passed in the same round).
> - Example C (warp-specialized pipeline: TMA / MMA warps exit right after `tail()`, only the epilogue warps rendezvous, no cluster barrier at teardown):
>   `PROBE PASS max_diff=0.0`
>   (round 2 of this certification; all five contracts passed in the same round).
> - Example E (idle / one-sided roles exit directly; the remote arrive into this CTA is awaited by a one-shot `exit_ok` mbarrier; only the surviving warps do the teardown cluster barrier):
>   `PROBE PASS max_diff=0.0` (round 4 of this certification; all four contracts passed in the same round).
> - Gotcha 10 (the same warp doing `bar.arrive` then `bar.sync` on the same id is counted twice): the `double_count` contract, `PROBE PASS` (round 2).
> - The probe for example D is listed under "Sources".
>
> **Certified with no time gain at all**; all three whole-kernel landing readings were flat (see "When needed").

## Purpose

Before `tcgen05.dealloc` you must be sure that **every warp that read TMEM has finished reading**. The laziest form is one CTA-level
`cute.arch.barrier()` (`bar.sync 0`, all threads rendezvous) followed by `tmem.free()`. It does two extra things:

1. it also pulls the **warps that never touched TMEM** into that rendezvous;
2. it makes every TMEM-reading warp **block** at the rendezvous point, while the only one that really needs to wait is the allocating warp that executes the dealloc.

This entry's form: use a **fixed-width named barrier** (width = number of participating threads), and split the rendezvous into two kinds of action:

| Role | Action | Afterwards |
|---|---|---|
| warp that read TMEM but is not responsible for the release | `tcgen05.wait::ld` → `bar.arrive ID, N` (**non-blocking**) | falls through to the end of the kernel and exits |
| allocating warp (responsible for the release) | `tcgen05.wait::ld` → `bar.sync ID, N` (blocks until N threads have arrived) | `tmem.free()` (in the pair form this includes the handshake with the peer CTA) |
| warp that never touched TMEM | does not take part in this barrier | exits after fulfilling its own teardown duties (gotcha 5), example C |
| warp with no role / no live work on this CTA | does not take part in this barrier | has no code at all after role dispatch, falls through to the end and exits, example E |

Per the PTX ISA, `bar.arrive` and `bar.sync` count into the same counter: once N threads have arrived (with whichever instruction), the `bar.sync` side is released.

## When needed (trigger)

- The kernel ends with a CTA-level `cute.arch.barrier()` immediately followed by `tmem.free()`, and in the SASS stall attribution the
  `BAR.SYNC` before `EXIT` is dominated by `barrier`.
- A role does **nothing** after its last issue and only keeps everyone company at the rendezvous.
- In a warp-specialized kernel the TMA warp, the MMA warp and the role-less warps finish their work early but sit on the teardown
  `bar.sync 0` until the epilogue ends; if the teardown is also followed by an all-thread `cluster_arrive()` / `cluster_wait()`, that
  likewise parks these warps until the peer CTA's epilogue is done too (gotcha 8). Example C removes both.
- `warp_active` is high: the role table has warps with no work on this CTA (idle warps, leader-only roles on the follower), yet they
  live from launch until the CTA ends, parked on the teardown rendezvous. Example E.

**Estimate the ceiling before you start.** This entry only removes the all-hands teardown rendezvous and the extra blocking; the saving
cannot exceed the "last accumulator commit → CTA exit" segment. **All three whole-kernel readings were flat**:

- On one pair-GEMM tree, landed per examples A / B (first the all-hands form, then narrowing the participants step by step): device
  medians of four evaluations 0.0157 / 0.0156 / 0.0157 / 0.0155 ms vs. 0.0157 ms before, all within noise;
- On another pair-GEMM tree, landed per example C (the same change also moved the TMEM allocate / release onto an epilogue warp):
  gate 0.0196 → 0.0200 ms; an evaluation of the same tree without kernel changes read 0.0181 / 0.0194 / 0.0177 ms, so the difference is within noise;
- On a third pair-GEMM tree, landed per examples C / E (no id-0 barrier after role dispatch, warps with no work exit directly, teardown
  cluster barrier removed, the remote arrive into this CTA covered by an existing tail wait):
  gate 0.1539 → 0.1538 ms; in the same evaluation NCU `stall_barrier` 7.83 → 2.25, `warp_active_pct` 11.65 → 9.41. Before landing, the
  same-job pricing of the "idle warps exit directly" lever on the same tree was add-one −0.37% / remove-one +0.39%, judged null. What
  disappears are the samples of warps waiting along, not the critical path.

So this entry is "the correct form of the teardown", not a known-effective performance lever. `stall_barrier` / `warp_active` will drop; don't expect ms to move.

**Anti-trigger**:

- Every warp still reads TMEM at teardown — then all you can do is swap `bar.sync` for `bar.arrive` (example B); no participant is removed.
- After the teardown there is smem that may only be reused once "all warps have passed some point" — that is a data dependency, not a teardown rendezvous.
- The kernel already has an all-hands `bar.sync 0` **after every warp's last TMEM read** (e.g. an smem data exchange): release on that one directly,
  no need for a dedicated id for the release, see example D.

## Minimal runnable example

Teardown segment only; for the prologue (allocate + publish the base address) see [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md).
Instance run by the certification probe for examples A / B: cluster (2,1,1), 8 warps, pair `TmemAllocator`; in the subset form 6 warps (including the allocating warp) read TMEM
and 2 never touch TMEM; in the all-hands form all 8 read.

```python
import cutlass
import cutlass.cute as cute

OWNER = ...          # the TmemAllocator's allocator_warp_id
NB_ID = ...          # named barrier id dedicated to teardown: one of 1..15 not used elsewhere (0 = CTA-level barrier())
N_TMEM_WARPS = ...   # number of warps that read TMEM, including OWNER
NPART = 32 * N_TMEM_WARPS

warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
reads_tmem = ...     # warp-uniform boolean computed from the warp index: did this warp read TMEM

if reads_tmem:
    ...                                         # tcgen05.ld reads TMEM, convert, write back
    cute.arch.fence_view_async_tmem_load()      # tcgen05.wait::ld: this warp's TMEM reads really completed

# ---- Example A: only the warps that read TMEM take part; the allocating warp blocks, the others arrive and leave ----
if warp == OWNER:
    cute.arch.barrier(barrier_id=NB_ID, number_of_threads=NPART)              # bar.sync NB_ID, NPART
else:
    if reads_tmem:
        cute.arch.barrier_arrive(barrier_id=NB_ID, number_of_threads=NPART)   # bar.arrive NB_ID, NPART
tmem.free(tmem_ptr)     # top-level call; internally only the allocating warp executes it (the pair form includes the peer handshake)

# ---- Example B: every warp reads TMEM ----
# NPART = 32 * total warps; everyone except OWNER does barrier_arrive, OWNER does barrier. No participant removed; only one blocking point left.
```

### Example C: the full teardown with warp specialization + CTA pair + a real pipeline in flight

Roles: one TMA warp (`PipelineTmaUmma` producer), one MMA warp (issues on the leader, `PipelineUmmaAsync` producer),
some role-less warps, a group of epilogue warps (the only warps that read TMEM; one of them doubles as the `TmemAllocator` allocating warp).
Prologue per [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md); **the teardown contains no CTA-level
`cute.arch.barrier()` and no teardown cluster barrier**. The form matches the certification probe's `subset` form.

```python
TMA_WARP, MMA_WARP = ..., ...
EPI_WARP0 = ...          # epilogue warps are consecutive, starting at a multiple of 4 (warp % 4 = the TMEM lane segment it can access)
N_EPI_WARPS = ...        # number of epilogue warps needed to cover all accumulator lanes
ALLOC_WARP = ...         # one of the epilogue warps; i.e. TmemAllocator(allocator_warp_id=ALLOC_WARP)
NB_ID = ...              # id dedicated to teardown; must not clash with the epilogue's internal group barriers or the allocator's retrieve barrier
NB_THREADS = 32 * N_EPI_WARPS   # counts only the epilogue warps

if warp == TMA_WARP:
    for kt in cutlass.range(num_k_tiles):
        h = ab_producer.acquire_and_advance()
        ...                                       # cute.copy(..., tma_bar_ptr=h.barrier)
    ab_producer.tail()                            # wait until every slot it issued ahead has been released by the MMA's commit
    # from here it falls through to the end of the kernel and exits — enters no barrier

if warp == MMA_WARP:
    if is_leader_cta:
        acc_h = acc_producer.acquire_and_advance()
        for kt in cutlass.range(num_k_tiles):
            f = ab_consumer.wait_and_advance()
            ...                                   # cute.gemm(...)
            f.release()
        acc_h.commit()
        acc_producer.tail()                       # wait until both CTAs' epilogues have released the accumulator
    # exit (on the follower this warp has no work and exits directly)

if warp >= EPI_WARP0:
    acc_full = acc_consumer.wait_and_advance()
    ...                                           # tcgen05.ld readback, convert, write out
    cute.arch.fence_view_async_tmem_load()        # tcgen05.wait::ld
    acc_full.release()
    if warp == ALLOC_WARP:
        cute.arch.barrier(barrier_id=NB_ID, number_of_threads=NB_THREADS)          # bar.sync NB_ID, NB_THREADS
    else:
        cute.arch.barrier_arrive(barrier_id=NB_ID, number_of_threads=NB_THREADS)   # bar.arrive NB_ID, NB_THREADS

tmem.free(tmem_ptr)       # top level; only ALLOC_WARP executes it (includes the dealloc mbarrier handshake with the peer's allocating warp)
# no cute.arch.barrier(), and no cluster_arrive()/cluster_wait()
```

Why no teardown cluster barrier is needed here (reasoning; the probe only verified the result for this one kernel): every cross-CTA
in-flight operation in the pair already has a point where "the receiver is waiting for it" — the peer's TMA accounts onto the leader's
full barrier (the leader's MMA warp waits on it); the leader's commit multicasts to the peer's empty / acc full (the peer's
`ab_producer.tail()` / epilogue wait on them); the peer epilogue's `release()` remotely arrives on the leader's acc empty (the leader's
`acc_producer.tail()` waits on it); the leader's MMA reads the peer's smem (the peer epilogue only proceeds after acc full, and acc full
comes after the last MMA completes); in the dealloc handshake the two allocating warps wait for each other. So no CTA exits while the peer
still has to touch it. When other remote writes are added (e.g. a DSMEM exchange, a hand-written remote arrive), recheck each one, see gotcha 9.

### Example D: release on an all-hands `bar.sync 0` the kernel already has

There is only one precondition: the kernel has a `cute.arch.barrier()` in which **all threads take part**, and in program order it comes
**after every warp's last TMEM read + `tcgen05.wait::ld`**. Then there is no need for a dedicated named barrier id for the release (one fewer `bar.arrive` / `bar.sync` pair):

```python
cute.arch.fence_view_async_tmem_load()               # every thread that read TMEM: tcgen05.wait::ld
...                                                  # the smem data this rendezvous publishes anyway
tcgen05_fence_before_thread_sync()                   # inline PTX: tcgen05.fence::before_thread_sync
cute.arch.barrier()                                  # the existing bar.sync 0 (all threads)
if warp == OWNER:
    tcgen05_fence_after_thread_sync()                # inline PTX: tcgen05.fence::after_thread_sync
tmem.free(tmem_ptr)                                  # top-level call
```

The inline-PTX spelling of the two `tcgen05.fence` calls is in [`ptx_escape`](ptx_escape.md) (`llvm.inline_asm` + `dsl_user_op`, `has_side_effects=True`).
In the certification probe this form still kept one teardown cluster barrier after the free (arrive after the free); removing it was not tested.
Cost: in the pair form the two allocating warps wait for each other inside `free()`, so the dealloc is pushed after that all-hands rendezvous; when
the two CTAs reach the rendezvous far apart in time, the allocating warp on the early side idles inside free (derived from the handshake order in the public sources, not measured).

### Example E: how the surviving warps tear down — warps with no work exit directly, and someone must wait for remote arrives into this CTA

Example C removed the teardown cluster barrier because "every cross-CTA operation has a receiver waiting for it". When the kernel has an
extra remote operation that **nobody waits for** (gotcha 9), there are two fixes (the certification scope of each is explained after the example):
E1 adds a one-shot `exit_ok` mbarrier and has the allocating warp wait on it before free; E2 keeps the teardown cluster barrier but has only
the still-alive warps do it. Roles in the certification probe: pair (cluster (2,1,1), `TmemAllocator(is_two_cta=True)`), one producer warp feeding a
group of TMEM-reading warps through this CTA's mbarrier ring, one warp that only has work on the leader, one idle warp, one allocating warp
that does not read TMEM; after its tail the producer spins for a while before exiting (simulating "the latest warp").

```python
OWNER = ...              # the TmemAllocator's allocator_warp_id; in this example it does not read TMEM (gotcha 10)
PROD = ...               # producer warp
LEADER_ONLY = ...        # warp that only has work on rank 0
NB_ID = ...              # id dedicated to teardown
NB_THREADS = 32 * (N_READ_WARPS + 1)   # TMEM-reading warps (arrive) + OWNER (sync)
# warps not listed in the role table = idle warps: no code at all after role dispatch

warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
exit_ok = smem.allocate_array(cutlass.Int64, 1)
if warp == OWNER:
    with cute.arch.elect_one():
        ...                                            # init the ring's full / empty and the other mbarriers
        cute.arch.mbarrier_init(exit_ok, 1)            # init together, before the prologue cluster barrier
# ... prologue: mbarrier_init_fence → cluster_arrive_relaxed / cluster_wait → allocate + relinquish_alloc_permit
#           → cluster_arrive / cluster_wait → retrieve_ptr (tmem_alloc_ptr_handoff_no_named_barrier form A)

if warp == PROD:
    ...                                                # mainloop
    ...                                                # tail: wait for each slot's last release
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(exit_ok, 1 - rank)   # E1: remote arrive on the peer CTA's exit_ok
    # falls through to the end and exits

if warp == LEADER_ONLY:
    if rank == 0:
        ...                                            # on the follower this warp exits directly

if reads_tmem:                                         # warp-uniform boolean; excludes OWNER
    ...                                                # consume the ring + tcgen05.ld
    cute.arch.fence_view_async_tmem_load()
    cute.arch.barrier_arrive(barrier_id=NB_ID, number_of_threads=NB_THREADS)
    # falls through to the end and exits

if warp == OWNER:
    cute.arch.mbarrier_wait(exit_ok, 0)                # E1: the remote arrive the peer sent into this CTA has landed
    cute.arch.barrier(barrier_id=NB_ID, number_of_threads=NB_THREADS)

tmem.free(tmem_ptr)                                    # top level; only OWNER executes it (includes the pair dealloc handshake)

# E2: only the still-alive warps do the teardown cluster barrier, inside a warp-uniform branch; warps that already exited do not take part
if warp == OWNER or reads_tmem:
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
```

Probe readings (median residency of each warp from entry to exit, `%globaltimer`, ns; the producer spins 100 µs after its tail):

| Form | OWNER | producer | leader-only | idle | TMEM-reading warps (range across warps) |
|---|---|---|---|---|---|
| E1 (no teardown cluster barrier) | 5888 | 105408 | 1184 | 1184 | 5760–5824 |
| E1 + E2 (surviving warps do the teardown cluster barrier) | 106048 | 105760 | 1216 | 1184 | 106048 (same for every warp) |
| Control (CTA-level `bar.sync 0` + all-hands cluster barrier) | 106016 | 106016 | 106016 | 106048 | 106016–106048 |

- E1: the surviving warps all finish before the producer exits (0% of CTAs later than the producer, median 99296 ns earlier).
- E2: the idle warp still exits at 1.2 µs without executing `barrier.cluster.arrive`, the surviving warps' cluster barrier completes as usual and all 3 launches return;
  but the surviving warps are dragged until after the producer exits (100% of CTAs; the latest surviving warp exits 320 ns after the producer, median). Interpretation of the reading
  (not separately certified): this cluster barrier waits for every thread in the cluster that has neither exited nor arrived to exit. So E2's cost is "the surviving warps keep the latest warp company".
- PTX footprint: in E1 / E2 the id-0 `bar.sync` count after role dispatch is 0, only the pair `bar.arrive NB_ID, NB_THREADS` + `bar.sync NB_ID, NB_THREADS`
  remains; `barrier.cluster.arrive` is down to the two in the prologue for E1, 3 for E2 and the control; all three forms have 1 dealloc.

Coverage of E1 (gotcha 9): in the probe the only remote operation into this CTA is the `exit_ok` arrive itself; using it to vouch for other remote operations
issued **earlier** by the same thread (remote release, DSMEM writes) relies on that arrive's memory-ordering semantics and scope, and is **uncertified**. The probe's E2 form also kept E1's wait,
so "covering remote operations with E2 alone" is reasoning too.

Key points:

- `number_of_threads` must **equal** the number of participating threads (arriving + syncing), a multiple of 32.
- Both actions sit in **warp-uniform** branches; `bar.*` has `.aligned` semantics — within one warp some lanes cannot arrive while
  others do not.
- `tmem.free()` is still called at the top level (it has an internal allocator-warp guard; the column count comes from the trace-time record of the top-level `allocate()`, see
  [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 8).

## Gotchas

1. **The width can only equal the number of participating threads.** Rule: N = 32 × (warps executing `barrier_arrive` + warps executing `barrier`).
   Per the PTX ISA, an N too large means `bar.sync` never fills up (hang); an N too small may release the allocating warp before the slowest reading warp arrives,
   so the dealloc races `tcgen05.ld`. **Neither failure was tested on the GPU**; the probe only ran the correct-width form. Moreover the probe runs one CTA
   per SM, and the released columns are not re-allocated by anyone, so even a race would not show in the values — the width can only be checked by inspection, not by numeric tests.
2. **`fence_view_async_tmem_load()` must come before the arrive.** `tcgen05.ld` is asynchronous; `bar.arrive` only says this thread got here.
   Every reading warp in the probe does this step before arriving; don't drop it when rewriting from the CTA-level `bar.sync 0` version.
3. **Non-participating warps must not participate at all.** If a warp executes `barrier_arrive` but the width does not count it, the count fills up early (equivalent to
   gotcha 1's N too small). Either do not participate at all, or also count it in the width.
4. **Do not reuse the barrier id elsewhere.** If the teardown id clashes with another named barrier in the kernel (e.g. a group rendezvous inside the epilogue),
   the counts of the two rendezvous interfere. Give each purpose its own id (PTX has 16; 0 is taken by the CTA-level `barrier()`).
5. **"Not taking part in the teardown barrier" does not mean "may exit any time".** Every warp that does not enter the barrier first fulfils its own teardown duties and then exits:
   - a producer using `PipelineTmaUmma` / `PipelineUmmaAsync` must `tail()` — without it the second launch of the same compiled entry hangs,
     see [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) gotcha 8. Example C's TMA warp / MMA warp exit right after `tail()`;
     this form is certified (3 back-to-back launches of the same compiled entry);
   - a warp that issued a TMA store (`cp.async.bulk` S2G) must still `cp_async_bulk_wait_group(0, read=True)` before exiting. Example C's probe writes out with
     plain gmem stores, no bulk store; if that warp is also a participant in the teardown rendezvous, whether the drain goes before or after the arrive is **uncertified**;
     putting it before the arrive is the conservative form (it only makes the allocating warp wait for this drain as well).
   Once these are done the warp may leave without waiting for the TMEM-reading warps — that is what this entry saves.
6. **The pair-form release handshake is still there.** With `is_two_cta=True`, inside `tmem.free()` the allocating warp and the peer CTA's allocating warp wait for each other through
   the dealloc mbarrier before deallocating. Don't remove it just because the teardown has been narrowed; its mbarrier init must still come before the first cluster
   barrier (see [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 4).
7. **Judge by the PTX, not by the values.** Once correct, the PTX teardown segment is `bar.arrive NB_ID, N` + `bar.sync NB_ID, N`, and the CTA-level id-0
   `bar.sync` count is **one fewer** than before (example A probe readings: subset form `bar.arrive 3, 192` + `bar.sync 3, 192` + 1 `bar.sync 0`,
   control 2 `bar.sync 0`; example C probe readings: `bar.arrive NB_ID, NB_THREADS` + `bar.sync NB_ID, NB_THREADS` (in the probe NB_THREADS = 4 epilogue warps), id-0 `bar.sync` 0 times, control 1).
   If the id-0 count did not change, a `cute.arch.barrier()` is still on the teardown path.
   When counting PTX note: the id and thread count of DSL-generated `bar.*` are usually register operands (like `bar.sync %r269, %r270;`); follow the `mov`s back
   to the immediates before comparing; a `bar.sync` with a single operand and no thread count is a whole-CTA rendezvous and counts in the id-0 class.
8. **The teardown cluster barrier is the second parking spot.** Replacing only `cute.arch.barrier()` and keeping the teardown `cluster_arrive()` /
   `cluster_wait()` still parks the TMA / MMA / role-less warps until all threads of this CTA (and the peer CTA) have arrived. The probe's `%globaltimer`
   readings: in the `subset_cl` form (example C + teardown cluster barrier) and the CTA-level-barrier control, the TMA warp of **every** CTA exits only after
   the epilogue's last TMEM read (0/148), and the median exit time of all roles is 1024 / 1056 ns later than the epilogue's last read;
   in example C (`subset`) the TMA warp exits first in 148/148 CTAs, with median lead TMA 2624 ns / MMA 6464 ns / role-less warps 6752 ns.
   These are single-tile timing readings inside the probe, proving "no longer waiting along at the rendezvous"; they are **not** whole-kernel time readings. PTX criterion: in example C
   `barrier.cluster.wait` is down to the two in the prologue (control 3).
9. **The precondition for removing the teardown cluster barrier must be checked item by item.** A CTA's smem exists as long as one of its threads is alive; once all warps have exited,
   no remote operation into that smem (remote arrive, DSMEM write) is guaranteed. Example C can drop it because every cross-CTA operation has some warp of this CTA
   waiting for it (see the reasoning after example C). As soon as the kernel has one more remote operation nobody waits for (e.g. a last release into a peer's slot, a DSMEM exchange),
   either give it a waiter or keep the teardown cluster barrier. Numeric tests do not reveal this kind of hole; the certification covers only the example C kernel.
   The forms and coverage of the two fixes are in example E (E1 `exit_ok`, E2 only the surviving warps do the teardown cluster barrier). The pair handshake in `tmem.free()` covers only
   the one arrive between the two allocating warps; it does **not** cover remote operations issued by other warps.
10. **The same warp must not `bar.arrive` and then `bar.sync` on the same id.** A named barrier counts arriving threads without knowing who they are: a warp
    that arrives and then syncs is counted twice, and the sync returns before the real partners have arrived, with no error. The easiest way to get this wrong at teardown is when the allocating warp itself also reads
    TMEM and falls into the "every warp that read TMEM does `barrier_arrive`" branch: it arrives once and syncs once, which is like counting one warp short in the width
    (gotcha 1's N too small), and the dealloc may precede the slowest reading warp. The allocating warp only takes the sync path; exclude it from the TMEM-reading branch condition.
    GPU measurement (the `double_count` contract, 296 CTAs × 3 launches): one warp arrives then syncs on an id of width 64,
    another warp spins, then writes an smem flag, then arrives; in 100% of CTAs the flag read after the sync returns is still the old value
    (`early_release_frac=1.000`, `flag_seen_frac=0.000`).
11. **Warps that already exited do not hang the teardown cluster barrier, but warps still running drag it out.** Example E2 readings: the idle warp exits early without executing
    `barrier.cluster.arrive`, and the surviving warps' cluster barrier completes as usual; meanwhile, if a non-participating warp is still running, the surviving warps wait until it exits.
    So E2 is only cheap when "the latest warp" is not much later than the surviving warps anyway.

## Sources

- PTX ISA: `bar.sync` / `bar.arrive` / `barrier.sync` (named barriers 0–15, the thread-count operand, arrive does not block, `.aligned`)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier
- PTX ISA: `barrier.cluster.arrive` / `barrier.cluster.wait`
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster
- PTX ISA: `tcgen05.wait::ld` / `tcgen05.fence::before_thread_sync` / `after_thread_sync` / `tcgen05.dealloc` and the thread-synchronization ordering of `tcgen05`
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions
- Public infra API symbols: `cute.arch.barrier(barrier_id=, number_of_threads=)`, `cute.arch.barrier_arrive(barrier_id=, number_of_threads=)`,
  `cute.arch.fence_view_async_tmem_load`, `cute.arch.make_warp_uniform`, `cute.arch.cluster_arrive` / `cluster_wait`,
  `PipelineTmaUmma` / `PipelineUmmaAsync` (`acquire_and_advance` / `wait_and_advance` / `release` / `commit` / `tail`),
  `%globaltimer` via `cutlass._mlir.dialects.llvm.inline_asm` (`mov.u64 $0, %globaltimer;`; the build used for certification has no `cute.arch` wrapper),
  `cutlass.utils.TmemAllocator` (`allocate` / `relinquish_alloc_permit` / `retrieve_ptr` / `free`, `is_two_cta`),
  `tcgen05.St32x32bOp` / `Ld32x32bOp` / `make_tmem_copy`; example E additionally uses `cute.arch.mbarrier_init` / `mbarrier_arrive(ptr, peer_cta_rank)` /
  `mbarrier_wait`, `cute.arch.elect_one`, `cute.arch.block_idx_in_cluster`, `cute.arch.cluster_arrive_relaxed`.
- GPU certification probe (examples A / B): ✅ `PROBE PASS max_diff=0.0`, round 1. Three forms × 148 CTAs × cluster (2,1,1) × 3 consecutive launches of the same compiled entry;
  each reading warp writes a known pattern with `tcgen05.st` and reads it back with `tcgen05.ld`:
  - `subset6_roundtrip` 0.0: 2 warps never touch TMEM and take part in no teardown barrier; the other 5 reading warps `bar.arrive(3,192)`, the allocating warp `bar.sync(3,192)` then pair free;
  - `all8_roundtrip` 0.0: 7 warps `bar.arrive(3,256)`, the allocating warp `bar.sync(3,256)`;
  - `control_roundtrip` 0.0: CTA-level `bar.sync 0` teardown (control);
  - `footprint`: both the subset and all-hands forms show arrive/sync on id 3 with width equal to the participating thread count, and one fewer id-0 `bar.sync` than the control; 1 dealloc in each of the three forms.
  - The per-launch times printed by the probe (including host overhead), 1002 / 992 / 992 µs, are dominated by host overhead and are **not** a time reading of this mechanism.
- GPU certification probe (example C): ✅ `PROBE PASS max_diff=0.0` (round 2).
  cluster (2,1,1), `CtaGroup.TWO`, 16 k-tiles through a 4-stage operand ring (the producer's `tail()` really has to wait for unreleased slots), 148 CTAs, 8 warps (TMA / MMA / 2 role-less /
  4 epilogue, the allocating warp among the epilogue warps); three teardown forms, each launched 3 times back to back from the same compiled entry: `subset` (example C), `subset_cl` (example C + teardown cluster barrier),
  `control` (`bar.sync 0` → free → cluster barrier). Readings:
  - `subset_exact` / `subset_cl_exact` / `control_exact` all 0.0 (C bit-equal to the fp32 reference);
  - `footprint`: (id-0 `bar.sync`, `barrier.cluster.wait`, dealloc) = `subset` (0, 2, 1), `subset_cl` (0, 3, 1), `control` (1, 3, 1); the first two have `bar.arrive 2,128` + `bar.sync 2,128`, the control does not;
  - `early_exit`: see the timestamp readings in gotcha 8 (148/148 CTAs have a timestamp row).
- GPU certification probe (example D): ✅ `PROBE PASS max_diff=0.0` (round 1).
  cluster (2,1,1), 2-CTA UMMA, every warp reads TMEM, 3 back-to-back launches of the same compiled entry; `T1-piggy-exact` 0.0 (example D: C bit-exact, smem exchange element-exact),
  `T2-named-exact` 0.0 (control: release on a dedicated `bar.arrive` / `bar.sync`), `T3-footprint`: SASS `BAR.SYNC` 4 vs 5, `BAR.ARV` 0 vs 1; example D's PTX uses one fewer named barrier id than the control.
  Verifies only correctness and instruction footprint; no timing readings.
- GPU certification probe (example E): ✅ `PROBE PASS max_diff=0.0` (round 4).
  pair `TmemAllocator(is_two_cta=True)`, 296 CTAs, role layout as in example E,
  the producer feeds the reading warps through this CTA's 2-slot mbarrier ring, each reading warp writes a known pattern with `tcgen05.st` then reads it back with `tcgen05.ld`; three forms, each launched 3 times back to back from the same compiled entry:
  - `exit_roundtrip` 0.0 (E1), `exit_cl_roundtrip` 0.0 (E1 + E2), `control_roundtrip` 0.0 (CTA-level `bar.sync 0` + all-hands cluster barrier); every warp reached its own exit;
  - `footprint`: E1 / E2 have 0 id-0 `bar.sync` and exactly one fixed-width arrive / sync pair, the control 1 id-0 `bar.sync`; `barrier.cluster.arrive` 2 / 3 / 3 times; 1 dealloc each;
  - residency readings: see the table in example E.
- GPU certification probe (gotcha 10): the `double_count` contract (round 2), `PROBE PASS`: `early_release_frac=1.000`, `flag_seen_frac=0.000` (296 CTAs × 3 launches).
