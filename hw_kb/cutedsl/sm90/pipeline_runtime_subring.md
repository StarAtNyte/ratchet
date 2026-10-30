# pipeline_runtime_subring (an independent pipeline stream per warpgroup: runtime-indexed barrier sub-ranges)

> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose

Split a TMA pipeline in which "all consumers hang on the same arrive count" into N sub-rings,
**each owned by one warpgroup** (its own full/empty barrier stream, its own phase), where the
index the consumer side uses to pick its ring (`wg_idx`) is a **runtime value**.

## When needed (trigger)

- Profiler / structural symptom: several math WGs idle in the same phase — every WG waits on the
  same barrier pair of the same stage, so fast WGs wait for slow ones; you want WG i to read stage
  `(k + phase_i) % STAGE` in round k, staggering the phases.
- Construction symptom: `wg_idx` is a runtime value, so **the N pipeline objects in a Python list
  cannot be indexed by it** (list indexing happens at trace time). The way out is not to pick an
  object, but to have each WG **build** its own pipeline from a runtime-sliced barrier sub-range.

## Minimal runnable example

```python
# N sub-rings share one barrier MemRange, viewed as (2*STAGE, N_RINGS): ring i owns column i.
# Key API fact: the runtime slice `bar_t[(None, ring)].iterator` can be passed directly as
# PipelineTmaAsync.create's barrier_storage (it is a cute.Pointer).
#
# This function is called inside staged if/for, so bar_t must be passed explicitly,
# not captured by a closure (SCOPE_CLOSURE_CAPTURE: a function called inside staged
# control flow must not close over runtime values).
def _make_pipe(bar_t, ring_idx, tx_count):
    return pipeline.PipelineTmaAsync.create(
        barrier_storage=bar_t[(None, ring_idx)].iterator,
        num_stages=STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        # arrive count = the real number of consumer warps of this sub-ring (here 1 WG = 4 warps),
        # not the number of all consumer warps
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
        tx_count=tx_count,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

# Inside the kernel body:
bar_t = storage.bars.get_tensor(              # MemRange[Int64, STAGE*2*N_RINGS]
    cute.make_layout((STAGE * 2, N_RINGS), stride=(1, STAGE * 2)))

if wg_idx == N_RINGS:                          # producer warpgroup
    if warp_idx == PRODUCER_WARP:
        # On the producer side the ring loop is constexpr: the N pipeline objects come from
        # the same traced expression and are fed ring by ring
        for ring in cutlass.range_constexpr(N_RINGS):
            pipe = _make_pipe(bar_t, ring, tx_bytes)
            phase = (ring * DELTA) % STAGE
            st = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, STAGE)
            for ti in cutlass.range(n_tile):
                slot = (st.index + phase) % STAGE      # phase relabelling
                sub = ring * STAGE + slot              # slot in the merged staged layout
                pipe.producer_acquire(st)
                bar = pipe.producer_get_barrier(st)
                cute.copy(tma_atom, gsrc[(None, ti * N_RINGS + ring)],
                          ssub[(None, sub)], tma_bar_ptr=bar)
                pipe.producer_commit(st)
                st.advance()
else:                                          # every math WG runs the same traced body
    pipe = _make_pipe(bar_t, wg_idx, tx_bytes)  # wg_idx is a runtime value -- this is the recipe
    phase = (wg_idx * DELTA) % STAGE
    st = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGE)
    for ti in cutlass.range(n_tile):
        pipe.consumer_wait(st)
        sub = wg_idx * STAGE + (st.index + phase) % STAGE   # same relabelling as the producer
        ...  # consume stage `sub`
        pipe.consumer_release(st)
        st.advance()
```

## Gotchas

1. **The arrive count must equal the real number of consumer warps of this sub-ring** (4 in the
   example). Keeping the pre-split total (e.g. 16) is not a numerical error but a **deadlock** —
   the empty barrier never fills up.
2. **Producer and consumer must independently compute the same phase relabelling**
   `(state.index + phase_i) % STAGE`. If the two sides disagree, the result is again a deadlock or
   stale data, not a numerical error — mismatches of this kind **all show up as silent-or-hang**;
   CPU reasoning cannot decide them, a GPU probe is required.
3. **tx_count is all the TMA bytes of one slot of that sub-ring** (including every tensor moved in
   the same slot, e.g. a data block plus its scale strip); leave one out → producer_commit never
   completes.
4. **SCOPE_CLOSURE_CAPTURE**: a helper called from a staged `if`/`for` must not capture runtime
   values in a closure (the barrier tensor and the indices must all be passed explicitly).
5. **The certification geometry must wrap around the cycle more than once**: `n_tile` must be
   strictly larger than STAGE and not equal to exactly one lap (`n_tile == STAGE` is exactly one lap,
   the **weakest** test of the phase relabelling — a wrong relabelling goes undetected).
6. Allocate the barrier MemRange once as `STAGE * 2 * N_RINGS` Int64s and slice rings
   column-major (`stride=(1, STAGE*2)`), so each ring's full/empty pairs are contiguous and do not
   overlap.

## Sources

GPU-certified probe (the probe is not included in this release; from an fp8 scoring-type operator):
all three contracts (runtime-sliced barrier accepted and four streams advancing independently /
4-sub-ring staging round trip + phase relabelling consistent on both sides / single-atom C-fragment
bijection) `PROBE PASS`; the same construction, in the finished kernel, passed the whole-kernel M0
gate (correctness + perf both passed).
