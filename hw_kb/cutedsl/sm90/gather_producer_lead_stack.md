# gather_producer_lead_stack (stacking two levers of a gather producer: lane-held indices × per-stage acquire, measured super-additive)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry puts [`gather_index_lane_owned`](gather_index_lane_owned.md) (how the indices are held)
> and "ring acquire granularity" (the producer waits only for the empty of the stage it is filling,
> instead of waiting for all of them at the top of the trip) on the same tree and measures them in the
> same job: add one, remove one, all on. Conclusion: **both are independent levers, and stacking them
> is super-additive** (all on +16.2% > sum of the single adds +14.7%, coupling +1.5 pp); together they
> push the certification tree from 0.604x to 0.702x, spin exposure 1.13 → 0.80 ms.

## Purpose

When a gather-type producer (sparse attention top-k KV, MoE routing) is slower than the consumer, the
consumer's spin exposure has two sources that can be cut on the producer side:

1. **The index round trip** stands in front of every copy (`gather_index_lane_owned`: one per lane,
   `shfl.sync.idx` broadcast).
2. **Ring acquire granularity**: if a one-trip-deep ring waits at the top of the trip with
   `for st: wait(empty[st])` until all stages are empty before filling, then between "the slowest
   consumer finishes releasing" and "the first stage of the next trip is complete" there is a whole
   gather round trip with nothing overlapped. Changing it to wait for the segment being filled lets the
   earliest-released stage (the one the consumer releases at `wait_group(1)`) start refilling while
   the consumer is still finishing PV. When smem is at its cap the ring cannot be deepened, and this is
   a zero-cost timing change.

The two move the same quantity (spin exposure) yet do not eat into each other: the index lever shortens
the wait **before each copy is issued**, the acquire-granularity lever shortens the **gap between
trips**; when one gets shorter, the other's overlap window actually gets fuller.

## When needed (trigger)

1. In the role exposure ledger the `spin` share is far above the reference and `long_scoreboard`
   dominates (the producer is the slow side).
2. The producer's empty-waits appear as a group at the top of the trip (SASS: N consecutive
   `SYNCS.PHASECHK` before the first `LDGSTS`), and the consumer's empty-arrives sit at two different
   depths (once after `wait_group(1)` and once after `wait_group(0)`) — meaning some stage has long been
   empty but nobody fills it.
3. smem is already at its limit; the ring cannot be deepened.

## Minimal runnable example (CuTe DSL)

```python
# producer trip loop. Baseline form: wait for all stages to be empty at the top of the trip
#   if t > 0:
#       for st in range_constexpr(NSTAGE): mbarrier_wait(mbar + EMPTY + st, (t-1) & 1)
#   for si in range_constexpr(NSTAGE): st = FILL_ORDER[si]; ...fill st...
# Changed to: wait for the segment being filled
for si in cutlass.range_constexpr(NSTAGE):
    st = FILL_ORDER[si]
    if t > 0:
        cute.arch.mbarrier_wait(mbar + (MBAR_EMPTY + st), (t - 1) & 1)   # only THIS stage
    ...fill stage st (indices per gather_index_lane_owned)...
    cute.arch.cp_async_mbarrier_arrive_noinc(mbar + (MBAR_FULL + st))
```

Changing FILL_ORDER to the consumer's release order (fill the earliest-released first) gave **no extra
gain** in this entry's measurement (`per_stage_early` 3.148 ms vs `per_stage` 3.135 ms); leave it alone.

## Machine-read footprint (acceptance criteria)

1. **Pricing table** (`lever_pricing_run.py`, three rounds in the same job, device ms jitter < 0.03%):

   | Variant | device ms | vs_ref | spin exposed ns | producer exposed ns | tensor |
   |---|---|---|---|---|---|
   | base | 3.4056 | 0.604 | 1,126,670 | 355,222 | 37.8% |
   | +lane_index | 3.2279 | 0.638 | 990,781 | 302,127 | 40.2% |
   | +ring_per_stage | 3.1314 | 0.660 | 879,610 | 331,541 | 41.3% |
   | **all** | **2.9866** | **0.702** | **800,845** | 296,861 | **43.6%** |
   | all − lane_index | 3.1323 | 0.660 | 876,382 | 331,726 | 41.2% |
   | all − ring_per_stage | 3.2280 | 0.638 | 991,464 | 304,166 | 40.1% |

   `LEVER lane_index add=+5.48% remove=-6.11% class=independent`;
   `LEVER ring_per_stage add=+9.23% remove=-9.22% class=independent`;
   `STACK all=+16.16% sum_singles=+14.71% coupling=+1.46%`.
2. **Remove-one ≈ −add-one** is the criterion for "independent": both are symmetric to within 0.6 pp.
   A complementary pair (add-one ≈ 0, remove-one < 0) did not occur in this set.
3. HGMMA 44 and LDGSTS 68 identical across all six versions (producer-only change); LDL/STL: base
   20/20, +ring 25/25 (four phases kept alive separately cost 5 more), +lane 12/12.

## Ceiling

With everything on, 0.80 ms of spin remains (share about 40%), tensor 43.6% against the reference's
63%. That part is in neither of these two levers, nor in the consumer's wait placement
([`midchain_stage_wait`](midchain_stage_wait.md) is already on the tree): it is the `long_scoreboard`
of the gather **data** itself — 64 LDGSTS make one DRAM round trip per trip, and the ring is only one
trip deep. Going further means either changing the geometry (smaller stages for more depth, R1
territory) or letting the producer lead by more than one trip (needs smem). This entry stops here; the
rest is a structural decision.

## Gotchas

1. **Waiting for all empties at the top of the trip looks "clean"**: one wait, short code, but it
   leaves the earliest-released stage idle for a whole gather round trip. Reading `SSSS…LDGSTS` in the
   dossier is this shape.
2. **Concluding from just one lever**: the single adds of +5.5% and +9.2% each fall short of 0.7x;
   only stacking gets past it. The point of a pricing round is to measure "all on" and "remove one"
   together; single adds cannot tell whether stacking is super-additive or mutually cannibalising.
3. **Using the jitter of the same-job ratio as the noise floor**: the reference side jitters too, and
   tripled it can reach 3.8%, judging a real 1–2% lever as null. The floor must be computed from the
   repeat spread of the student's device ms (`lever_pricing_run.py` already does this).
4. Changing FILL_ORDER to chase the release order: no gain on this tree; don't spend an extra round
   on it.

## Sources

- ✅ Verified: certified by a GPU lever-pricing probe (the probe is not included in this release;
  based on a gather-attention tree that had passed first-green)
  (H100, cutlass-dsl 4.6.1 image): six variants × three rounds, `PRICE-OK levers=2
  rounds=3`; all six versions correctness PASS, all with role ledger `ROLE-OK … tensor=`. Single-lever
  sources: separate pricing runs for lane_index and for ring_per_stage / per_stage_early.
- Price tags: `[PRICED: all +16.16% (lane_index ⊗ ring_per_stage)]`;
  `[PRICED: ring_per_stage +9.23% / lane_index +5.48% single add]`;
  `[PRICED: coupling +1.46 pp super-additive]`.
- Public interfaces: `cutlass.cute.arch.mbarrier_wait`, `cp_async_mbarrier_arrive_noinc`;
  the `shfl.sync.idx` inline PTX of `gather_index_lane_owned`.
