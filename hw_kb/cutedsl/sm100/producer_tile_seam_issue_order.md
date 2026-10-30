# producer_tile_seam_issue_order (issue order of a TMA producer at a tile seam: split taking the response into "read / release", leaving only read → acquire → first issue on the seam)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Status: ✅ **GPU-certified** — certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=0.0` (round 1; all four variants end / defer / hoist4 / hoist4_defer pass).
> Mechanism readings (probe synthetic load, median producer cycles per seam, total / `empty` wait / non-wait segment): end 318 / 96 / 222; defer 319 / 95 / 224; hoist4 203 / 95 / 107; hoist4_defer 169 / 97 / 72. **How to read them**: with H = 0, deferring "release" alone saves no cycles (222 → 224); the bulk of the saving comes from moving "read" into the tail (222 → 107), and on top of that, deferring "release" saves another ~35 cycles (107 → 72).
> **Whole-kernel readings are flat** (see the end of "When needed"); the most economical combination in the probe showed no gain on the whole kernel.
> Architecture scope: **sm100 (Blackwell)**, warp-specialized persistent kernels with hand-written mbarriers + a cluster launch control (CLC) queue;
> CuTe DSL `cute.arch.clc_response` / `mbarrier_wait` / `mbarrier_arrive` / `fence_proxy`.
>
> | | |
> |---|---|
> | **Knob** | The **program order** of the producer (the warp that issues TMA per k-block) at a tile seam. Split one "take the response" into two halves: **read** (`mbarrier_wait(full[s])` + `clc_response(slot s)` → next item's coordinates and valid bit) and **release** (`fence_proxy("async.shared", space="cta")` + `mbarrier_arrive(empty[s])`). Between "this item's last issue → next item's first issue" on the seam, keep only: read (or nothing: read already moved into this item's tail, see [`tile_seam_clc_take_hoist`](tile_seam_clc_take_hoist.md)) → coordinate decode → ring acquire → this beat's TMA. **Release** and all per-item bookkeeping (tile counter, previous item's coordinates, information published to other roles) move to **after** the next item's first issue; the last (invalid) response is released after the loop |
> | **Effect (mechanism level)** | The non-wait instruction segment on the seam shrinks from "read + fence + remote arrive + bookkeeping" to "read + decode"; the fence and arrive land after the next item's first issue and are absorbed by the next beat's `empty` wait |
> | **Load-bearing conditions** | **This entry is not priced; whole-kernel readings are flat.** The certified surface is correctness (with read and release separated, the queue still delivers each item exactly once and does not deadlock) + one mechanism reading (the producer's non-wait seam cycles, see gotcha 6). If the producer is already held back by `empty` backpressure at the seam (ring full), the saved cycles are swallowed by the wait |

## Purpose

Every time a persistent kernel's producer crosses a tile seam it has to do a few things unrelated to the data: confirm the next item (CLC response), return the response slot
to the queue, update its own counters and records. Only "the next item's coordinates" is something the first TMA really depends on; releasing the slot takes one
`fence.proxy.async` and one (possibly cross-CTA) `mbarrier.arrive`, and the bookkeeping is a handful of register / store instructions. Strung as-is
between two items, they form a supply gap that appears at every seam. This entry moves both of these to after the next item's first issue.

Relation to [`tile_seam_clc_take_hoist`](tile_seam_clc_take_hoist.md): that entry moves the **whole** take into the tail of this item's k loop,
so that the time spent "waiting for the response to land" is covered by the tail's issues; this entry handles the **non-wait** instruction segment left on the seam, and applies equally to H = 0
(take after the k loop). The two stack: read moved into the tail, release moved to after the next item's first issue.

## When needed (trigger)

- On the timeline, the producer's seam segment (`empty` wait ends → next item's first TMA) contains a visible instruction segment besides the `empty` / response-full waits;
  or the MMA issuer's `full` wait at the top of a tile is clearly longer than its `full` wait mid-tile.
- Code shape: `for k in range(K): …; item, valid = take(...) (containing wait / read / fence / arrive); tile_cnt += 1; last_x = …`,
  with the first `producer_acquire` only at the top of the next trip.

**Anti-trigger**: the producer's seam segment is almost entirely `empty` wait (ring full, MMA slow), and the non-wait segment is only a few dozen cycles. Then reordering
saves nothing; the bottleneck is the MMA chain or the ring depth ([`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md)).

**Whole-kernel readings**: on one GEMM tree with pair UMMA + a CLC queue, multi-workload geometric mean, interleaved timing in the same job,
relative to "whole take after the k loop, release not deferred" (ratio = after / before, < 1 is faster): release deferred only (H = 0) 0.9998; read moved into the tail, release not deferred
(H = 4) 0.9998; read moved into the tail + release deferred (H = 2 / H = 4) 1.0021 / 1.0011. All inside the noise; the combination that saved the most cycles in the probe showed
no gain on the whole kernel.

## Minimal runnable example

The producer segment of the certification probe (`issue(item, k)` stands for one k-block's acquire → TMA → advance; the probe uses
an smem write + `mbarrier_arrive(full)` in place of the TMA, at the same program point):

```python
def clc_read(clc_mbar, clc_resp, n):            # first half of taking the response: wait + read
    s = n % CLC_STAGES
    cute.arch.mbarrier_wait(clc_mbar + s, (n // CLC_STAGES) % 2)
    cx, cy, _, valid = cute.arch.clc_response((clc_resp + 4 * s).align(16))
    return cx, cy, valid

def clc_release(clc_mbar, n):                    # second half: fence -> release the slot
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.mbarrier_arrive(clc_mbar + CLC_STAGES + (n % CLC_STAGES), 0)   # 0 = the CTA that owns the slot

item = first_item                                # own blockIdx (cluster_launch_control_work_queue gotcha 1)
live = cutlass.Int32(1)
n_resp = cutlass.Int32(0)
t = cutlass.Int32(0)
while live != 0:
    issue(item, 0)                               # <- first thing after the seam: acquire + first TMA batch
    if t > 0:
        clc_release(clc_mbar, n_resp - 1)        # <- "release" of the previous read, moved after the first issue
    bookkeeping(t, item)                         # <- per-item bookkeeping also after the first issue
    for k in cutlass.range(1, K - H):
        issue(item, k)
    nxt, nv = clc_read(clc_mbar, clc_resp, n_resp)   # with H > 0 the read moves into the tail; with H = 0 it is after the k loop
    n_resp += 1
    if cutlass.const_expr(H > 0):
        for k in cutlass.range(K - H, K):
            issue(item, k)                       # the tail is still the current item
    item = nxt
    live = nv
    t += 1
clc_release(clc_mbar, n_resp - 1)                # the last (invalid) response: released after the loop
```

In a real kernel, `issue(item, 0)` is:

```python
pipe.producer_acquire(state)
bar = pipe.producer_get_barrier(state)
cute.copy(tma_a, tAgA[(None, item_a, 0)], tAsA[(None, state.index)], tma_bar_ptr=bar)
cute.copy(tma_b, tBgB[(None, item_b, 0)], tBsB[(None, state.index)], tma_bar_ptr=bar)
state.advance()
```

k = 0 is peeled out of `cutlass.range` as straight-line code so that "release + bookkeeping" can be inserted after it. The k loop then starts at 1.

## Gotchas

1. **Before releasing, the response must already be read into registers, and the fence must come before the arrive.** Once the slot is released, the scheduler may issue the next
   query on the same slot, and the async proxy rewrites those 16 B. The result of `clc_response` is a register value, so releasing arbitrarily late afterwards is safe; conversely, arriving first
   and reading afterwards may read the response of the next query: no hang, silently taking the wrong item. `fence_proxy("async.shared", space="cta")`
   sits between the generic-proxy read and the release, as in the unsplit version, just moved later as a whole.
2. **The last response must be released after the loop.** Each "release" in the loop releases the slot of the **previous** read; after an invalid response is read the loop exits, and that one
   is released by nobody. Before finishing, the scheduler waits until the last responses have been released by all readers (the teardown drain of [`cluster_launch_control_work_queue`](cluster_launch_control_work_queue.md));
   a missing release is a hang at teardown. In the probe, both deferred-release variants (`defer` / `hoist4_defer`) take this post-loop release path.
3. **Each response is released exactly once; the empty count does not change.** Splitting only changes the program point, not the count; the `t > 0` guard ensures the first item does not release
   "response −1". Do not forget to delete the fence + arrive inside the original `take` after splitting — that would release twice: empty flips early, and the scheduler queries
   a slot that has not finished being read.
4. **Deferring "release" creates no circular dependency, provided the query-issue rule does not run ahead.** The scheduler issues query n + 1 only after response n has been released by all readers.
   The moment the producer releases response n depends on the `empty` of the next item's first slot, i.e. on the consumer having released an earlier slot.
   That chain points back to no query that has not yet been issued, so it closes. If the query-issue rule is changed to run several items ahead and the response ring has no more slots than the lead,
   deferring the release makes the scheduler wait one more item, which shows up on the timeline as a longer slot wait for the scheduler. No hang, but read it.
5. **If the bookkeeping includes data published to other roles (e.g. finalisation information), moving the publish point must still honour its consumer contract.** This entry only moves
   the producer's own bookkeeping; anything others read through a barrier must be written before that barrier's arrive. Re-check after reordering.
6. **How to take the reading: seam cycles = time of the next item's first issue − time of this item's last issue, minus the `empty` wait inside it;
   what remains is the non-wait segment this entry can move.** The probe stamps `%clock64` after every issue; differences within the same warp are valid.
   Stamp the same way in a real kernel, but only for A/B (the stamping itself costs a few instructions). When the ring is full, the `empty` wait swallows the saved cycles,
   so report both; the total seam alone does not show this entry's effect.
7. **Once `issue(item, 0)` is peeled, the k-loop start and `ACCUMULATE`-style k == 0 tests are not on the producer side.**
   The producer only issues loads, so peeling k = 0 changes only the producer; whether the MMA issuer should peel its head likewise is a separate matter, not certified by this entry.

## Sources

- PTX ISA: `clusterlaunchcontrol.try_cancel` / `clusterlaunchcontrol.query_cancel`, `mbarrier.try_wait`,
  `mbarrier.arrive` (the remote `shared::cluster` form), `fence.proxy.async`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL public API symbols: `cutlass.cute.arch.clc_response`, `cutlass.cute.arch.issue_clc_query`,
  `cutlass.cute.arch.mbarrier_wait` / `mbarrier_arrive`, `cutlass.cute.arch.fence_proxy`,
  `cutlass.pipeline.PipelineTmaUmma.producer_acquire` / `producer_get_barrier`, `cutlass.range`, `cutlass.const_expr`.
- Certification probe: cluster (2,1,1), 1 CTA/SM (smem padding block),
  600 items declared, K = 16, a hand-written 4-slot ring. Four variants: `end` (H = 0, whole take + bookkeeping after the k loop, control) / `defer`
  (H = 0, read after the k loop, release + bookkeeping after the next item's first issue) / `hoist4` (H = 4, whole take) / `hoist4_defer` (H = 4 read,
  release deferred), 4 launches each. Contracts: each item exactly once, the two rows of a pair identical, the producer's self-recorded item sequence == the finaliser role's item sequence,
  per-item sums exactly equal to numpy, the 4 launches bit-identical. FACT: producer seam cycles (total / `empty` wait / non-wait segment).
- Related entries: [`tile_seam_clc_take_hoist`](tile_seam_clc_take_hoist.md) (read moved into the tail),
  [`cluster_launch_control_work_queue`](cluster_launch_control_work_queue.md) (the queue itself),
  [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) (encoding of the `empty` wait).
