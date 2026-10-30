# tile_seam_clc_take_hoist (tile seam: move "take the next work item" from after the k loop into the tail of the k loop)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Status: ✅ Verified (correctness and mechanism readings only, **no whole-kernel gain**) — two GPU probes (not included in this release):
> the general form on a hand-written ring (`PROBE PASS max_diff=0.0`, round 3, all three H settings passed);
> the H = 1 form peeling the last beat on a `PipelineTmaUmma` supply ring (`PROBE PASS max_diff=0.0`, round 1, including a negative control).
> The same-job whole-kernel A/Bs on one warp-specialized GEMM tree were all **flat** (see the end of "When needed").
> Architecture scope: **sm100 (Blackwell)**, warp-specialized persistent kernels with hand-written mbarrier or `cutlass.pipeline` rings + a cluster launch control (CLC) queue;
> CuTe DSL's `cute.arch.clc_response` / `mbarrier_wait` / `cutlass.range`.
>
> | | |
> |---|---|
> | **Knob** | Roles that pipeline per k-block (TMA supplier, MMA issuer) used to run, per work item, "`for k in range(K)` → take the CLC response → next item". Instead, **split** the k loop in two: `range(0, K − H)` → take the response (the next item's coordinates and valid bit go into `nxt`) → `range(K − H, K)` (**still on the current item**) → `cur = nxt`. H is a compile-time constant (`1 ≤ H ≤ K − 1`). The CLC ring, the query-issue rule, and the counts and phase expressions of both barrier groups are **all unchanged**; only the program point of the "take" moves |
> | **Effect (mechanism level)** | The `mbarrier_wait` + 16 B response read + proxy fence + remote arrive at the seam no longer sit in series between "this item's last k-block" and "the next item's first k-block"; they fall into the issue period of this item's last H k-blocks |
> | **Load-bearing condition** | **This entry is unpriced; the whole-kernel readings are flat.** How long the take waits is decided by "when the query was issued"; moving it too deep moves the wait back in (gotcha 3). Whether it is worth anything can only be read on your own tree with a same-job A/B |

## Purpose

In a persistent kernel, every streaming role does two data-independent things at the tile seam: ask the queue "who is next, and is there anything left",
and recompute addresses for the new coordinates. If this step comes after the k loop, it lands on the critical path: the supplier must wait for it before issuing
the next item's first batch of TMA, and the MMA issuer must wait for it before entering the next item's full wait. The CLC response has usually landed long ago (the query
was issued near the start of the previous item), so the "take" costs mostly an instruction segment + a remote arrive; moving it into the tail of this item's k loop overlaps that time with
the tail's issue.

## When needed (trigger)

- On the timeline, the tile seam segment of a streaming role (issue of the last k-block → the next item's first k-block full / empty wait)
  contains a "take the response" wait or instruction segment, and the seam period is a visible fraction of the tile period.
- The supplier's first TMA for the next item is issued after "take the response", not right after the previous item's last TMA.
- Code shape: `while valid: for k in range(K): ...; item, valid = clc_take(...)`.

**Anti-trigger**: the real wait in the seam segment is empty / full (data back-pressure), not the response take — that is a matter of pipeline depth or MMA throughput,
and this step cannot move it.

**Whole-kernel readings (look at these before deciding whether to change anything)**: on one GEMM tree with pair UMMA + a CLC queue, geometric mean over multiple workloads,
same-job interleaved timing (ratio = after / before, < 1 is faster):

| What was changed | H | Same-job ratio |
|---|---|---|
| supplier + MMA issuer moved together | 2 / 4 / 8 | 0.9990 / 0.9996 / 0.9999 |
| supplier only | 4 | 0.9998 |
| MMA issuer only (the issuer needs only the valid bit) | 2 / 4 | 0.9990 / 1.0014 |
| supplier's `PipelineTmaUmma` ring peels the last beat (the "H = 1 form" below) | 1 | no same-job A/B; a single cross-job reading is flat with the upper edge of a cross-job re-measure of the same base tree (about 0.06% higher) |

All within noise. What the probe shortens is a few hundred cycles at the seam; if the supplier is already held at the seam by `empty` back-pressure, the saved cycles
are absorbed by the next beat's wait — before changing anything, confirm with your own seam timestamps that the "take the response" segment really is on the critical path.

## Minimal runnable example

Skeleton (`step(item, k)` is your acquire → issue → advance for one k-block; not this entry's subject):

```python
H = TAKE_H                  # compile-time constant: take the response before the H-th-from-last k-block
KSPLIT = K - H

# ---- supplier / MMA issuer (each role that reads the response writes its own copy, same shape) ----
item = first_item           # first item = own blockIdx (see cluster_launch_control_work_queue gotcha 1)
live = cutlass.Int32(1)
n_resp = cutlass.Int32(0)
while live != 0:
    for k in cutlass.range(0, KSPLIT):
        step(item, k)
    nxt, nv = clc_take(clc_mbar, clc_resp, n_resp)   # take the next item: wait full → clc_response → fence → arrive empty
    n_resp += 1
    if cutlass.const_expr(KSPLIT < K):
        for k in cutlass.range(KSPLIT, K):
            step(item, k)                            # the tail is still the current item (gotcha 1)
    item = nxt
    live = nv
```

`clc_take` is exactly the segment in [`cluster_launch_control_work_queue`](cluster_launch_control_work_queue.md)
(`mbarrier_wait(full + s, (n // STAGES) % 2)` → `cute.arch.clc_response((resp + 4 * s).align(16))`
→ `cute.arch.fence_proxy("async.shared", space="cta")` → `cute.arch.mbarrier_arrive(empty + s, 0)`),
unchanged; this entry only changes where it is called.

The MMA issuer needs no coordinates, only the valid bit: put `_, nv = clc_take(...)` at the same split point; the `acc` commit still comes after both
k-loop segments, and the next item's `acc` acquire is still at the top of the next pass.

With the pipeline classes (`PipelineTmaUmma`'s `producer_acquire` / `consumer_wait` etc.) both loop segments share the same
`PipelineState`, which keeps `advance()`-ing across segments and items as usual — splitting the loop does not touch stage indices or phases.

### H = 1 form: peel the last beat off a `PipelineTmaUmma` supply ring

The shallowest setting. Write the last k-block as straight-line code and put the response take **before** its `producer_acquire`, so the "take" overlaps the last beat's
`empty` wait (the supplier would be held there by the consumer anyway):

```python
while valid != 0:
    for k in cutlass.range(0, K - 1):
        pipe.producer_acquire(ps)
        bar = pipe.producer_get_barrier(ps)
        cute.copy(tma_a, tAgA[(None, item, k)], tAsA[(None, ps.index)], tma_bar_ptr=bar)
        ps.advance()
    nxt, nv = clc_take(clc_mbar, clc_resp, n)         # ← before the last beat's empty wait
    n += 1
    pipe.producer_acquire(ps)                         # last beat: still the current item (gotcha 1)
    bar = pipe.producer_get_barrier(ps)
    cute.copy(tma_a, tAgA[(None, item, K - 1)], tAsA[(None, ps.index)], tma_bar_ptr=bar)
    ps.advance()
    item = nxt
    valid = nv
```

With two TMA streams per beat, the straight-line segment likewise gets two `cute.copy` calls. Per-item bookkeeping (tile count, previous item's coordinates) is recorded with the old `item`
**before** `item = nxt`. `PipelineState` keeps `advance()`-ing continuously across "loop segment → straight-line segment → next pass's loop"; the total beat count is still K,
and `producer_tail(ps)` still goes after the while.

## Gotchas

1. **The tail must use the current item's coordinates, not the next item just taken.** Put the taken coordinates into a new variable `nxt`, and do `item = nxt` only after
   the tail (or the peeled last beat) has been issued. If you write `item, valid = clc_take(...)` directly and then run the tail, the last H k-blocks read the **next item's**
   data: no hang, no error, each item's tail silently computes the wrong thing. The H = 1 probe has a dedicated negative-control setting with exactly this spelling; the per-item exact-sum contract
   caught it (`data_maxdiff = 6`), the other four settings read 0.
2. **Every role that reads the response takes exactly once per item, in unchanged order.** The empty barrier's count is the thread count of all readers;
   moving the position does not change "once per item", so counts and phases stay. Don't move one take and also leave the original one at the end of the loop.
3. **H's upper bound is set by the query-issue rule; going too deep moves the wait back in.** When query n is issued depends on "whom the warp issuing queries
   waits for to release response n − 1". If the slowest reader (e.g. a finishing role that only takes after accumulator full) has to wait for the previous item's MMA chain
   to finish before releasing the response, then a supplier that takes near the start of this item is waiting on the previous item's entire chain — the seam bubble moved, it did not
   disappear. The probe (K = 16 beats per item) prints one row of the supplier's "response-take cycles per item" for each of three H settings (same kernel, only H varies;
   74 clusters launched, 8–9 items per cluster): median / p90 are H = 0 → 162 / 162, H = 4 → 148 / 149,
   H = 15 (take at k = 1) → **742 / 887**. A shallow move lowers the wait slightly; moving it to the start of the k loop grows the wait 4.6×. Sweep up from small H; don't go all the way in one step.
4. **This does not deadlock, provided the response take's dependency chain does not point back to work this role has not issued yet.** Taking response n mid-item depends on
   "all readers released response n − 1", which at the latest depends on the previous item's completion; this role issued all of the previous item's k-blocks long ago, so the chain
   is closed. The H = 1 probe has one class of readers take only after the whole k loop (the latest reader), and it is still closed. If your query-issue rule
   changes to lead by more items, or some reader must wait for **this** item's completion before releasing the response, walk the chain again.
5. **Both loop segments use `cutlass.range` (runtime loops); the tail is wrapped entirely in `const_expr`.** With `H = 0` the whole segment is folded away at compile time
   and the shape reverts to the original, which makes A/B on the same code easy ([`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)).
6. **`nxt` / `nv` are first defined inside the while body and used up within the same pass.** The DSL accepts this shape; but values carried across passes
   (`item`, `live`, `n_resp`, counters) must be initialised with the same type before the `while` (`cutlass.Int32(...)`),
   otherwise trace time reports a type mismatch for loop-carried variables ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)).
7. **Move only the "take", not the other seam waits.** The next item's `acc` acquire and the k = 0 empty acquire each have their own dependencies;
   moving them forward too is a different change, uncertified here. Splitting the "take" into read / release halves and deferring the release until after the next item's first issue is in
   [`producer_tile_seam_issue_order`](producer_tile_seam_issue_order.md).
8. **The conditional "take early only if it's ready" form is not shorter.** Before the peeled last beat, check CLC full with `mbarrier.test_wait`, take early only if ready,
   otherwise issue the last beat and then take: its correctness was certified in the same round (both paths were exercised), but in the same probe the seam median / p90 was
   329 / 392 cycles, **no shorter than the original form (300 / 451)**, whereas the unconditional H = 1 form was 151 / 328 (the response take itself per seam
   151 → 138 cycles). No gain reading supports it; don't use it by default.
9. **The response buffer must be 16 B aligned.** `SmemAllocator.allocate_array` does not accept `byte_alignment`
   (the same TypeError is in [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md)); get the pointer with
   `smem.allocate(bytes, 16)` + `cute.recast_ptr(..., dtype=cutlass.Int32)`. A bare `cute.Pointer` cannot be
   indexed for reads / writes (trace time reports `ArithValue object cannot be interpreted as an integer`); for a buffer written element by element, use
   `smem.allocate_tensor(dtype, cute.make_layout(n), byte_alignment=...)` to get a tensor and index that.

## Sources

- PTX ISA: `clusterlaunchcontrol.try_cancel` / `clusterlaunchcontrol.query_cancel`,
  `mbarrier.try_wait` / `mbarrier.test_wait`, `mbarrier.arrive`, `fence.proxy.async`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL public API symbols: `cutlass.cute.arch.issue_clc_query`, `cutlass.cute.arch.clc_response`,
  `cutlass.cute.arch.mbarrier_wait` / `mbarrier_arrive` / `mbarrier_arrive_and_expect_tx`,
  `cutlass.cute.arch.fence_proxy`, `cutlass.pipeline.PipelineTmaUmma.producer_acquire` / `producer_get_barrier` /
  `producer_tail`, `cutlass.pipeline.make_pipeline_state`, `cutlass.range`, `cutlass.const_expr`.
- Certification probe 1 — cluster (2,1,1), 1 CTA/SM (smem padding),
  600 items declared, hand-written smem ring; both streaming roles (supply and consume) split their loops by H, the finishing role takes the response after accumulator full;
  three H settings ∈ {0, 4, 15} (K = 16), 4 launches each. Contracts: each item exactly once, the pair's two rows identical, launch count < declared count,
  per-item sums exactly equal to numpy, the 4 launches bit-identical. Verdict: ✅ `PROBE PASS max_diff=0.0` (round 3).
- Certification probe 2 —
  cluster (2,1,1), 1 CTA/SM, 600 items declared, K = 16, one `PipelineTmaUmma` operand ring per CTA, one small CLC response ring;
  variants: original form / H = 1 form / conditional form /
  conditional form + each query delayed by 30 µs (forcing the fallback path) / negative control (last beat uses the next item's coordinates), 3 launches each. Verdict:
  ✅ `PROBE PASS max_diff=0.0` (round 1) — the four positive settings are bit-exact, the negative control is caught.
- Related entries: [`cluster_launch_control_work_queue`](cluster_launch_control_work_queue.md) (the queue itself),
  [`producer_tile_seam_issue_order`](producer_tile_seam_issue_order.md) (splitting read / release),
  [`tile_boundary_wait_placement_prep_in_flight`](tile_boundary_wait_placement_prep_in_flight.md) (the same class of
  seam change: "move only program points, leave the barrier table alone").
