# cluster_launch_control_work_queue (cluster launch control work queue: the grid is declared per work item, and resident clusters claim the next item with `try_cancel`)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; CuTe DSL's `cute.arch.issue_clc_query` / `cute.arch.clc_response`.
>
> What this entry records is **one knob + its SASS / profiler footprint + the conditions under which it bears load**, not a percentage:
>
> | | |
> |---|---|
> | **Knob** | the launch grid is declared as "one cluster per work item" (far more than the number of clusters that can be resident); every cluster that actually launches **first does the item of its own `blockIdx`**, and after that claims each item by using `clusterlaunchcontrol.try_cancel` to cancel a cluster that **has not launched yet** and taking over its work item; when the queue answers "invalid" it finishes. The response is 16 B, distributed within the cluster through a small smem ring (4 deep in this entry's certification) + two sets of mbarriers (full: count 1 + a 16 B transaction; empty: count = all threads that will read this slot) |
> | **Footprint** | profiler `grid_size` 148 → 8192 (it reads the **declared** grid, not the resident count); `smem_dynamic` 199208 → 199344 B (a ring of 4 × 16 B + 8 barriers + alignment padding); dossier `mbar_objs` 33 → 42, `mbar_tiers` `{CONST: 32, RING: 1}` → `{CONST: 37, RING: 4, THREAD: 1}`, `loops` `{copy-producer: 3, scalar: 16}` → `{copy-producer: 4, scalar: 25}`; contract axis `rings` `["stride=0x1,wrap=sel"]` → `["stride=0x1,slots=25", "stride=0x1,slots=9", "stride=0x30a6f,wrap=mask", "stride=0x400100"]` (none of the four reads as `slots=4`, see gotcha 9); `stall_barrier` 1.44 → 0.17 |
> | **Load-bearing conditions** | under the structure "1 CTA/SM (dynamic smem ≈ 199 KiB), 74 resident clusters, every work item the same amount of work (the same number of chunks), item counts per cluster differing by only 1 under static striding", the gate window went **0.9888 → 0.997 ms (+0.83%, worthless)**. Static striding had no imbalance for a dynamic queue to smooth out; all the queue bought was the declared shape of `grid_size`. The footprint appearing ≠ it being worth something |
>
> **What is certified**: the correctness of the queue + ring + two sets of barriers is certified by a GPU probe (every declared cluster processed exactly once,
> both CTAs of a pair reading the same sequence of work items, a real shape launched whole-machine to completion with no holes, 8 repeated launches on the same input bit-identical);
> the footprint and the ms readings are certified by the dossier census + gate window of one delivered KEEP.

## Purpose

A persistent kernel's "work item → CTA" dispatch has two spellings. **Static striding**: grid = the number of resident CTAs,
cluster c does `c, c + N, c + 2N, …`, and the trip count is a compile-time / launch-time constant. **The cluster launch control
(CLC) queue**: the grid is declared as the total number of work items, the hardware launches only the batch that fits, and each launched cluster, after finishing an item,
sends a `try_cancel` to the scheduler -- cancelling a cluster that has not launched yet and taking its `blockIdx` as its own next item.
The trip count becomes a run-time quantity, and who does which item is decided by the hardware's launch order.

## When needed (trigger)

- The work per work item is **uneven**: under static striding some CTAs finish early and some trail (this entry has **no** readings on such a tree;
  this is the scope the mechanism gives, not a priced gain).
- An architecture conformance check requires `grid_size` to equal the total number of work items (the declared shape itself is part of the contract).
- You are already writing the queue and are stuck on: where the first item comes from / what the counts of the two sets of barriers are / how to assemble the response-slot pointer /
  why the isolated probe passed but the whole machine hangs.

**Anti-trigger**: work items of equal work, item counts per CTA differing by only 1 under static striding -- this entry's KEEP reading is exactly such a tree, +0.83%.

## Minimal runnable example

Skeleton extracted from the certification probe's micro-kernel: cluster `(2,1,1)`, 128 threads per CTA, warp 0 doubling as the scheduler,
a large array padded into smem to push occupancy down to 1 CTA/SM (otherwise every cluster launches and the cancel path is never exercised -- the probe
prints this fact). Each CTA records the work items it did in `mLog`, and the host checks "each item exactly once, pairs in sync".

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

CLC_STAGES = 4                      # power of 2: slot = n & 3, phase = (n >> 2) & 1
CLC_MASK, CLC_SHIFT = CLC_STAGES - 1, 2
CLC_WORDS, CLC_BYTES = 4, 16        # the response of one try_cancel = 16 B
BLOCK, NCL = 128, 222               # threads per CTA; number of declared clusters
BAR_FULL, BAR_EMPTY = 0, CLC_STAGES # two sets of barriers, CLC_STAGES each


def clc_slot(clc, s):
    # ordinary Python helper: after the dynamic offset, **re-declare** the 16 B alignment (gotcha 3)
    return (clc + CLC_WORDS * s).align(16)


@cute.jit
def clc_issue(mbar, clc, j):
    # one thread in the leader's scheduler warp: arm a 16 B transaction on the full barrier of **each** of the two CTAs,
    # then issue a multicast query against the local barrier and slot -- the response and completion land at the same smem offset in every CTA of the cluster
    s = j & CLC_MASK
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(mbar + BAR_FULL + s, CLC_BYTES, 0)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar + BAR_FULL + s, CLC_BYTES, 1)
        cute.arch.issue_clc_query(mbar + BAR_FULL + s, clc_slot(clc, s), multicast=True)


@cute.jit
def clc_take(mbar, clc, n):
    # consumer side: **every thread** of both CTAs goes through every response once
    s = n & CLC_MASK
    cute.arch.mbarrier_wait(mbar + BAR_FULL + s, (n >> CLC_SHIFT) & 1)
    m_idx, _, _, valid = cute.arch.clc_response(clc_slot(clc, s))
    cute.arch.fence_proxy("async.shared", space="cta")     # finish reading before releasing the slot (gotcha 5)
    cute.arch.mbarrier_arrive(mbar + BAR_EMPTY + s, 0)      # remote arrive on the leader's copy
    return m_idx // 2, valid                                # the response gives the first CTA id of the cancelled cluster


@cute.jit
def clc_drain(mbar, n):
    # when the queue answers invalid, CLC_STAGES - 1 queries are still in flight: they still write this CTA's smem; wait for them to land before exiting
    for j in cutlass.range_constexpr(CLC_STAGES - 1):
        cute.arch.mbarrier_wait(mbar + BAR_FULL + ((n + j) & CLC_MASK),
                                ((n + j) >> CLC_SHIFT) & 1)


@cute.kernel
def k_clc(mLog: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    smem = utils.SmemAllocator()
    ball = smem.allocate_array(cutlass.Int32, 49500, byte_alignment=1024)   # 198 KB: 1 CTA/SM
    mbar = smem.allocate_array(cutlass.Int64, 2 * CLC_STAGES, byte_alignment=8)
    clc = smem.allocate_array(cutlass.Int32, CLC_WORDS * CLC_STAGES, byte_alignment=16)
    ball[tidx] = bidx
    if warp == 0:
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(CLC_STAGES):
                cute.arch.mbarrier_init(mbar + BAR_FULL + i, 1)
                cute.arch.mbarrier_init(mbar + BAR_EMPTY + i, 2 * BLOCK)   # all threads of the pair
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()

    if tidx == 0:
        mLog[(bidx, 0)] = (bidx // 2).to(cutlass.Int32)     # first item = its own blockIdx (gotcha 1)
    if warp == 0:
        if rank == 0:
            for j in cutlass.range_constexpr(CLC_STAGES):   # fill the ring first
                clc_issue(mbar, clc, j)
    n = cutlass.Int32(0)
    cnt = cutlass.Int32(1)
    live = cutlass.Int32(1)
    while live != 0:
        item, v = clc_take(mbar, clc, n)
        if warp == 0:
            if rank == 0:
                if v != 0:                                   # issue the next query only after everyone has released the slot
                    cute.arch.mbarrier_wait(mbar + BAR_EMPTY + (n & CLC_MASK),
                                            (n >> CLC_SHIFT) & 1)
                    clc_issue(mbar, clc, n + CLC_STAGES)
        if v != 0:
            if tidx == 0:
                mLog[(bidx, cnt)] = item.to(cutlass.Int32)   # in a real kernel this is a whole trip of work
            cnt = cnt + 1
        n = n + 1
        live = v
    clc_drain(mbar, n)


@cute.jit
def host(mLog: cute.Tensor, stream):
    k_clc(mLog).launch(grid=[2 * NCL, 1, 1], block=[BLOCK, 1, 1], cluster=[2, 1, 1],
                       min_blocks_per_mp=1, stream=stream)
```

Host-side acceptance (the probe's criteria): `mLog` prefilled with −1; the records of the two CTAs of each launched cluster are identical item by item; all records
concatenated = a permutation of `range(NCL)` (no duplicates, no missing items, no out-of-range); launched count < `NCL` (otherwise the cancel path was not exercised).

When wiring into a real kernel: each role's original counting loop `for k in range(n)` becomes a `while` carrying `(item, number of responses taken, continue flag)`,
and each role calls `clc_take` **exactly** once per trip; the other counts (ring stage indices, the phases of the barriers) are still computed from
the trip number, and their counts and phase expressions do not change at all. Make "use the queue or not" a module constant; when off, the same loop shape
falls back to static striding -- the certification probe kept this control variant and it passed numerically.

## Gotchas

1. **The first item of a launched cluster is its own `blockIdx`; the queue only provides the subsequent items.** `try_cancel` cancels clusters that **have not
   launched yet**, so a launched cluster's own item never comes out of the queue. The certified form is written this way.
   The control form (taking the first item from the queue too) was not run here; by the mechanism it loses one item per launched cluster -- the results are right everywhere,
   only as many rows as there are resident clusters are missing, so the coverage contract must count holes over **all** work items, not spot-check a few.
2. **The count of the empty barrier is "all threads that will read this slot"; warps without a role must tag along too.** The count is a compile-time
   constant; the certified form takes all threads of the pair (2 × threads per CTA), so **every** warp -- including warps with no work at all,
   and the warps on the follower that are asymmetric to the leader -- must have a loop that only takes responses. Miss one, and the leader's scheduler
   warp never sees that slot filled, and the whole grid stops: **a hang, not wrong numbers** (inferred from the semantics of mbarrier arrival counts;
   the control form missing one warp was not run here).
3. **The response-slot pointer must carry 16 B alignment.** `cute.arch.clc_response` reinterprets the slot as one 128-bit quantity read as a whole,
   and checks **the alignment attribute carried by the pointer**; the ring's base address is allocated at 16 B, but after "base + dynamic slot offset" the attribute is lost.
   Readings: in the probe round where the slot pointer was written as a bare offset, every stage that built the queue hit `DSLOperationBuildError`; written as
   `(base + 4 * s).align(16)`, the next round was 9/9 PASS. (The bare-offset control stage of the same round printed only fact lines, no verdict;
   this entry did not obtain its reading `[to be checked]`.)
4. **The full barrier has one copy on each of the two CTAs, and each must be armed.** The multicast query writes the response into the smem of every CTA in the cluster,
   and the transaction completion is recorded on **each CTA's own** barrier; so before the query the issuing thread does one
   `mbarrier_arrive_and_expect_tx(bar, 16, rank)` for rank 0 and one for rank 1. The precondition is that both CTAs have the same smem layout (the same
   allocation code) -- the response slot and the barrier are at the same offset on both sides.
5. **A proxy fence is needed after reading the response and before releasing the slot.** The slot is read by this thread in the generic proxy and rewritten by the next query in the async
   proxy; put `fence_proxy("async.shared", space="cta")` between `clc_response` and the empty arrive.
6. **Queries are still in flight at finish time.** At the moment the queue answers invalid, the ring still has "depth − 1" queries that have not landed, and they still write this CTA's
   smem; both CTAs must wait out these full barriers before exiting (`clc_drain` in the example).
7. **Passing the isolated probe does not mean the whole machine runs: the probe must run the kernel's own queue code.** This entry's first-round probe was an
   independent replica of the queue (the same thread count and smem envelope, the loop body replaced by stamping), with the queue logic inlined by the probe itself in its `@cute.kernel`
   body, not importing the kernel -- 4/4 PASS. The whole-machine eval right after it died **at trace time**: the kernel had written the queue
   issue side as an **undecorated** ordinary Python helper, in which an `if lane < CLUSTER_SHAPE[0]:` on the run-time lane number
   reported `PHASE_DYNAMIC_TO_STATIC_BOOL` ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 1); the kernel
   never reached the GPU -- **not a hang**, and unrelated to the barrier count, the ring depth or how the first item is taken. In the probe the same `if` sat in the kernel body
   staged by the preprocessor, so the replica could not catch it. In the certified form all queue helpers carry `@cute.jit`, and the probe
   imports the kernel directly to call these helpers, and runs the real kernel's small shape and whole-machine shape. Practice: the probe must link
   the kernel's **own** queue helpers and include a whole-machine smoke at one real shape ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)
   gotcha 3, [`gpu_probe_ladder`](gpu_probe_ladder.md)); when the whole machine gives "no verdict", read the job log first to tell a trace-time error
   from a hang; do not infer "hang" directly from "no verdict".
8. **The ring depth is set by "how many trips ahead the role that needs a response earliest is"; when the scheduler warp takes on other work, the refill point must move with it.** The certified form
   is 4 deep: the j-th response is issued after everyone has taken the (j − 4)-th. Later the scheduler warp also became a supply issuer, and the refill
   moved to the end of the next trip (waiting on empty means waiting for all threads to finish that trip; putting it at the end of this trip would make the first supply batch of the next trip wait behind the previous trip's
   wind-down); the set of queries is unchanged, the queue goes from 4 ahead to 3 ahead, and the number in flight at finish time is unchanged. The reading of that change came in the same package as moving the
   supply issue point (gate 0.9993 → 0.9844 ms) and cannot be separated.
9. **The `rings` axis does not give the ring depth.** A 4-deep response ring with slot number `n & 3` made it into the cubin, and the four the dossier reads are
   `stride=0x1,slots=25` / `stride=0x1,slots=9` / `stride=0x30a6f,wrap=mask` / `stride=0x400100`,
   with no `slots=4`; the "strides" of the last two are constants on the address chain. A later change completely unrelated to barriers (only adding cache-policy bits) rewrote
   `stride=0x400100` as `stride=0x400100,wrap=sel`, and a change that only altered another barrier's count rewrote `stride=0x30a6f`
   as `stride=0x3126f`.
   Whether the queue got built is shown by `grid_size`, `smem_dynamic` and `mbar_objs`; for the interpretation rules see
   [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11.
10. **`grid_size` reads the declared value.** The profiler's `grid_size` changes from 148 to 8192, while the resident count is still decided by smem occupancy
    (1 CTA/SM). Every host-side resource allocated per CTA (one descriptor slot per CTA and the like) must be allocated by the **declared**
    grid; every criterion that infers the resident count back from `grid_size` no longer holds once the queue is on.

## Sources

- PTX ISA: `clusterlaunchcontrol.try_cancel`, `mbarrier.arrive.expect_tx`,
  `fence.proxy.async`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL public API symbols (actually called by the certified code): `cutlass.cute.arch.issue_clc_query(mbar, slot, multicast=True)`,
  `cutlass.cute.arch.clc_response(slot)` (returns the three components of the first CTA id and a valid bit),
  `cutlass.cute.arch.mbarrier_arrive_and_expect_tx(bar, bytes, cta_rank)`, `mbarrier_arrive(bar, cta_rank)`,
  `mbarrier_wait`, `mbarrier_init` / `mbarrier_init_fence`, `fence_proxy`, `cluster_arrive_relaxed` /
  `cluster_wait`, `block_idx_in_cluster`, `elect_one`; `cutlass.utils.SmemAllocator.allocate_array`;
  the pointer's `.align(n)`. The DSL's built-in dynamic persistent scheduler and the CLC pipeline classes in `cutlass.pipeline` are wrappers of the same mechanism;
  this entry did not certify that path.
- Certification probe (a GPU probe, not included in this release) -- three rounds in sequence: round 1 (an independent replica of the queue, stamp conservation over 2048 items on a single CTA + 4096 items on a pair)
  `PROBE PASS (max_diff=0.0) — 4/4 PASS`; round 2 (switched to driving the kernel's own helpers, slot pointer as a bare offset)
  `PROBE FAIL — 1/8 PASS`, all failing stages `DSLOperationBuildError`; round 3 (`.align(16)` + the micro-kernel
  first) `PROBE PASS (max_diff=0.000364878) — 9/9 PASS`, contracts: in the micro-kernel each of the 222 declared clusters processed once
  and pairs in sync, all small-shape outputs without holes, numerics against an fp64 control, 8 repeated launches bit-identical, the real shape (8192 declared
  CTAs) launched whole-machine to completion with no unwritten bytes, the control variant with the queue off passing numerically. The job numbers of the three rounds were not recorded.
- KEEP evaluation (NCU dossier census + gate window; the eval job number was not recorded in the status record): axis `rings`
  `["stride=0x1,wrap=sel"]` → `["stride=0x1,slots=25", "stride=0x1,slots=9", "stride=0x30a6f,wrap=mask", "stride=0x400100"]`,
  gate 0.9888 → 0.997 ms; `grid_size` 148 → 8192, `smem_dynamic` 199208 → 199344, `mbar_objs` 33 → 42,
  `stall_barrier` 1.44 → 0.17, `stall_long_sb` 3.9 → 5.2, `tensor_pct` 62.76 → 63.63, spill 0 / 0 unchanged.
  An earlier whole-machine eval of the same decision failed to compile at trace time and never reached the GPU (gotcha 7).
