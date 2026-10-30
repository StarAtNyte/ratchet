# prologue_mbarrier_critical_path_init (split a CTA pair's prologue into a "first-issue critical group" and a "deferred group": the producer arrives early and waits late, overlapping the first TMA batch with TMEM allocation / the remaining mbarrier inits)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**, warp-specialized kernels with cluster (2,1,1) + `cta_group::2` (TMA producer / MMA / epilogue).
>
> **Status: ✅ Verified (correctness and in-prologue ordering; no whole-kernel gain is certified)** — certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=0.0`
> (certification round 1, all six contracts pass: `split_data` / `ctrl_data` / `split_counts` / `split_hold` / `ctrl_hold` all 0, `order` n/a).
> **Whole-kernel reading: flat in a same-job A/B** (see "When needed").

## Purpose

The common pair prologue is one serial chain ([`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) form A):

```
all mbarrier inits (ring + accumulator + scheduling + dealloc …) → fence → cluster barrier 1
→ tcgen05.alloc → cluster barrier 2 (publish the TMEM base) → role dispatch → producer issues the first TMA
```

The first TMA has to wait for the **whole chain**, yet it really depends on only one thing: the full / empty mbarriers of the operand ring are initialised and visible to the whole cluster
(the peer CTA's TMA records tx on the leader's full barrier). The TMEM base, the accumulator pipeline, the scheduling mbarriers and the dealloc mbarrier are
all untouched by the producer in the first tile.

This entry splits the prologue into two groups, relying on the **split phases** of `barrier.cluster`: `arrive` and `wait` are two instructions, so a thread can arrive first,
do something else, and wait later.

| Group | Content | How it becomes visible |
|---|---|---|
| **Critical group** (before barrier 1) | the operand ring's mbarriers (`PipelineTmaUmma.create(..., defer_sync=True)`); the producer's descriptor prefetch | `mbarrier_init_fence()` + `cluster_arrive_relaxed()` + `cluster_wait()` |
| **Deferred group** (between barrier 1 and barrier 2) | the accumulator pipeline, scheduling / other mbarriers, `TmemAllocator` construction (which inits the dealloc mbarrier), `allocate` + `relinquish_alloc_permit` | `mbarrier_init_fence()` + `cluster_arrive()` (release) + `cluster_wait()` (acquire) |

The producer warp `cluster_arrive()`s **immediately** after barrier 1 (it has nothing to publish for barrier 2), then issues the first tile's TMA;
its `cluster_wait()` for barrier 2 is pushed to **just before it first touches a deferred-group object** (e.g. the first read of a scheduling response), at the latest before the next cluster arrive.
The other warps do the deferred group at top level, then arrive + wait.

## When needed (trigger)

- The timeline / head breakdown shows the "entry → first TMA" stretch is long while the steady-state main loop is already at its bound; non-main-loop time clearly exceeds budget.
- In the prologue, ahead of the first TMA, there are: TMEM allocation, a second cluster barrier, a string of mbarrier inits the producer does not use.
- The kernel's producer uses neither TMEM nor a scheduling response in the first tile (the first tile's coordinates come straight from `block_idx`).

**Estimate the ceiling first.** The most this entry can save is the stretch "barrier 1 ends → barrier 2 ends" (the deferred group's inits + `tcgen05.alloc` + one cluster rendezvous),
and only **once per CTA, at its head**; it shortens neither the steady state nor the tail. Also be clear about the cost: in the probe, **the moment the other warps leave barrier 2 actually got later by
about 980 cycles** (entry→barrier2Exit 2028 → 3010; the deferred group was scheduled after barrier 1), so the MMA gets the TMEM base later; the first MMA waits
for both the data and barrier 2, and the net gain depends on which comes later.

**Whole-kernel reading (one pair persistent GEMM tree, three workloads): flat.** Two trees in the same job differing only in this prologue split: geometric mean −0.01%
(the three workloads −0.06% / +0.04% / +0.00%). Four single evaluations in which this entry was landed on the same tree showed same-job ratio changes against the previous evaluation between −5% and +5%,
in both directions, inside that tree's evaluation noise band. So this entry is "one correct way to order the prologue", not a known-effective performance lever; do your own same-job A/B before landing it.

**Anti-triggers**:

- Not a cluster launch: there is no `barrier.cluster` to split. A single CTA using `bar.sync` has no non-blocking CTA-level "all arrive" as a substitute.
- The producer's first tile already needs to read a deferred-group object (e.g. the first coordinates come from a scheduling response) — that object must then go into the critical group.

## Minimal runnable example

Only the skeleton of the prologue and the producer (the spelling matches the certification probe's `split` form; the full ring / MMA / teardown are in
[`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md), [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md)).

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

TMA_WARP, MMA_WARP = ..., ...      # warp-uniform role ids; MMA_WARP doubles as the TmemAllocator's allocating warp
RETRIEVE_BAR_ID = ...              # NamedBarrier required by the constructor; emits no instruction unless wait_for_alloc() is called, any unused id
S, ACC_STAGES = ..., ...           # operand ring stages, accumulator stages

# ... smem = utils.SmemAllocator(); st = smem.allocate(Storage); sA / sB ...
warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

# ================= critical group: only what the first TMA / the first full wait needs =================
if warp == TMA_WARP:
    cpasync.prefetch_descriptor(tma_a)          # depends on no smem object, can go first
    cpasync.prefetch_descriptor(tma_b)
ab_pipe = pipeline.PipelineTmaUmma.create(       # internally warp 0 inits 2*S mbarriers
    num_stages=S, producer_group=..., consumer_group=..., tx_count=tx_bytes,
    barrier_storage=st.ab_mbar.data_ptr(), cta_layout_vmnk=vmnk,
    defer_sync=True)                             # must defer: do not let it do its own whole-cluster sync
cute.arch.mbarrier_init_fence()
cute.arch.cluster_arrive_relaxed()               # cluster barrier 1: ring mbarriers visible to the cluster
cute.arch.cluster_wait()

# ================= deferred group: the producer arrives at barrier 2 first, the others arrive when done =================
if warp == TMA_WARP:
    cute.arch.cluster_arrive()                   # barrier 2's arrive; the wait is deferred
acc_pipe = pipeline.PipelineUmmaAsync.create(    # top-level call; internally warp 0 inits
    num_stages=ACC_STAGES, producer_group=..., consumer_group=...,
    barrier_storage=st.acc_mbar.data_ptr(), cta_layout_vmnk=vmnk, defer_sync=True)
# ... the scheduling mbarrier inits also go here (the first tile does not use them) ...
tmem = utils.TmemAllocator(st.tmem_hold,
                           barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=RETRIEVE_BAR_ID, num_threads=...),
                           allocator_warp_id=MMA_WARP, is_two_cta=True,
                           two_cta_tmem_dealloc_mbar_ptr=st.tmem_dealloc_mbar)   # the constructor inits the dealloc mbarrier
tmem.allocate(TMEM_COLS)                         # top-level call (internally guarded by allocator_warp_id)
tmem.relinquish_alloc_permit()
cute.arch.mbarrier_init_fence()                  # the deferred group's inits need a fence too
if warp != TMA_WARP:
    cute.arch.cluster_arrive()                   # release: publish the deferred group's inits + the TMEM base
    cute.arch.cluster_wait()

# ================= roles =================
if warp == TMA_WARP:
    ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
    t = cutlass.Int32(0)
    while valid:
        for kb in cutlass.range(K_TILES, unroll=1):
            ab_pipe.producer_acquire(ps)
            bar = ab_pipe.producer_get_barrier(ps)
            cute.copy(tma_a, tAgA[(None, tm, kb)], tAsA[(None, ps.index)], tma_bar_ptr=bar)
            cute.copy(tma_b, tBgB[(None, tn, kb)], tBsB[(None, ps.index)], tma_bar_ptr=bar)
            ps.advance()
        if t == 0:
            cute.arch.cluster_wait()             # the deferred barrier-2 wait: only after this may deferred-group objects be touched
        # ... read the scheduling response to get the next tile (tm, tn, valid) ...
        t = t + 1
    # if this CTA may get no tile at all, a wait must be added here (gotcha 2)
    ab_pipe.producer_tail(ps)
# The MMA / epilogue warps already waited for barrier 2 at top level; retrieve_ptr and use acc_pipe as usual.

# ================= teardown: every thread (including the producer) has waited for barrier 2 =================
cute.arch.barrier()                              # or narrowed per teardown_subset_rendezvous_early_exit
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tmem.free(tmem_ptr)
cute.arch.cluster_arrive()
cute.arch.cluster_wait()
```

## Gotchas

1. **Barrier 2 still needs all (non-exited) threads in the cluster to arrive.** Only "who arrives first" is split: the producer arrives in its own branch, the other warps arrive in
   `if warp != TMA_WARP`, and the two groups together are exactly all threads. Miss either group (e.g. putting the producer's arrive in a branch only the leader takes),
   and per the PTX ISA barrier 2 never completes and the kernel hangs. Both branch conditions must be **warp-uniform** (the warp id after `make_warp_uniform`).
2. **After an arrive, a wait is required before the next arrive.** Per the PTX ISA, between two `barrier.cluster.arrive`s of the same thread there must be a
   `barrier.cluster.wait`. If the producer's deferred wait sits on a path that may execute **zero times** (e.g. inside the `while valid` loop body, while this CTA
   gets no tile at all), the teardown `cluster_arrive()` becomes a second consecutive arrive. Fix: put the deferred wait where it is guaranteed to execute,
   or add a "wait if not waited yet" branch after the loop. Every CTA in the probe had 3 tiles, so **the zero-tile case is not covered**, and the add-a-wait spelling is not certified.
3. **Before the deferred wait, the producer must not touch anything of the deferred group.** That includes: reading the TMEM base, waiting on / arriving at accumulator or scheduling mbarriers, reading scheduling responses.
   Barrier 1 only guarantees visibility of the critical group's inits. The probe's producer reads the TMEM holding word only **after** the deferred wait, and across 148 CTAs × 3 launches it always read
   the published value (`split_hold`).
4. **The critical group must contain both full and empty of the whole operand ring, not just init `full[0]`.** The peer CTA's TMA records on the leader's full barrier, and the leader's
   MMA commit multicasts to both CTAs' empty barriers; on its second lap the producer already waits on its own empty. `PipelineTmaUmma.create` inits
   all 2·S at once anyway; there is no need to split them.
5. **Both `create`s need `defer_sync=True`.** Per the public source, without defer `create` itself does an `mbarrier_init_fence` + whole-cluster
   sync, so the one in the deferred group would become an extra cluster rendezvous; by then the producer has already arrived at barrier 2 and is issuing TMA, and this extra sync would get tangled with barrier 2's
   arrivals, with unpredictable results (the probe only ran the form with both deferred, not this negative case).
6. **The deferred group's inits also need `mbarrier_init_fence()`, followed by a release-form `cluster_arrive()`.** The dealloc mbarrier receives the peer's
   remote arrive and the accumulator empty receives the peer epilogue's remote release, so these inits must be visible cluster-wide; release + acquire also
   publish the TMEM holding word (see [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 2).
7. **`TmemAllocator` construction may go after barrier 1, as long as it is before barrier 2's fence + release arrive.** The dealloc mbarrier it inits
   is only arrived at by the peer in the teardown `free()`, by which time both sides are long past barrier 2. That is exactly the position in the probe's `split` form (bit-exact, 3 launches).
   "Put the construction before barrier 1" in [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 4 is
   the spelling for a single barrier 1 + serial prologue; the common requirement of both is "the init is visible before **some cluster barrier with a fence**, before the peer's first remote arrive".
8. **At teardown, put `retrieve_ptr` at top level, after all roles.** The producer only waits for barrier 2 inside its role; if the pointer is taken at top level before role dispatch, the producer's
   read happens before its wait (the value is unused, but do not leave such a read). The probe takes it after the teardown barrier.
9. **Criteria.** Once it is right: the **static** count of `barrier.cluster.arrive` (non-relaxed) in PTX is 1 higher than the serial spelling (barrier 2 has one site in each of the two branches),
   and `barrier.cluster.wait` is also 1 higher (the producer's deferred wait); the number of rendezvous at run time is unchanged. The timing criterion uses `%clock64`:
   "first TMA − barrier 2 exit" becomes negative after the split (the first TMA happens before barrier 2 completes).

## Sources

- PTX ISA: `barrier.cluster` (split arrive / wait phases, after an arrive each thread must wait before arriving again, arrive defaults to `.release`, `.relaxed`,
  wait defaults to `.acquire`, all non-exited threads in the cluster must participate)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster
- PTX ISA: `fence.mbarrier_init` (`.release.cluster`, makes mbarrier inits visible to other CTAs in the cluster)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
- PTX ISA: `tcgen05.alloc` / `relinquish_alloc_permit` (`.cta_group::2`)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit
- Public infra API symbols: `cutlass.pipeline.PipelineTmaUmma.create` / `PipelineUmmaAsync.create` (`defer_sync`),
  `cute.arch.mbarrier_init_fence`, `cute.arch.cluster_arrive` / `cluster_arrive_relaxed` / `cluster_wait`,
  `cpasync.prefetch_descriptor`, `cutlass.utils.TmemAllocator` (`allocate` / `relinquish_alloc_permit` / `retrieve_ptr` / `free`).
- GPU certification probe: `PROBE PASS max_diff=0.0` (round 1). The probe is the TMA / accumulator pipeline skeleton of a pair GEMM (the MMA warp
  consumes the operand ring but issues no UMMA; the accumulator pipeline is driven by `tcgen05.commit`), 148 CTAs × 3 launches, SM clock locked at 1500 MHz.
  Contracts: the in-ring data checksums of both forms, the number of k-blocks issued by the producer / accumulator consumptions, and the TMEM holding word read by every thread are all exact; `order` records the ordering.
  Cycle readings (median over CTA × launch):

  | Form | entry→firstTMA | entry→allocDone | entry→barrier2Exit | firstTMA − barrier2Exit | firstTMA − allocDone (min / median) |
  |---|---|---|---|---|---|
  | serial (ctrl) | 2339 | 976 | 2028 | +330 (0% of CTAs before barrier 2) | 963 / 1335 |
  | split | 1505 | 1136 | 3010 | −1515 (100% of CTAs before barrier 2) | −107 / 280 |

  Entry → first TMA is about 834 cycles earlier (≈ 0.56 µs @ 1.5 GHz), once per CTA at its head; the barrier 2 exit (warp 0 reading) is about 980 cycles later.
  Static PTX counts: in the split form `barrier.cluster.arrive` 2 → 3, `barrier.cluster.wait` 3 → 4 (consistent with the criterion in gotcha 9); `barrier.cluster.arrive.relaxed` 1,
  `mbarrier.init` 13, `tcgen05.alloc` 1 are the same in both forms.
- Whole-kernel A/B: the same job, two trees differing only in the prologue split, geometric mean −0.01% (−0.06% / +0.04% / +0.00%).
