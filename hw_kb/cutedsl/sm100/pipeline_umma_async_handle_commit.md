# pipeline_umma_async_handle_commit (a pipeline's commit / release must be issued on the handle returned by acquire / wait)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry answers one question only: **after `acquire_and_advance()` / `wait_and_advance()`, which stage should the commit / release
> of a CuTe DSL pipeline participant (`PipelineProducer` / `PipelineConsumer`) target,
> and what does it look like when it targets the wrong one.**
>
> The answer in one sentence: the handle returned by `acquire_and_advance()` records **the stage just obtained**, while the participant's
> own state has **already advanced to the next stage** within the same call. So
>
> | Spelling | arrive lands on | Conclusion |
> |---|---|---|
> | `producer.commit(h)` (or `h.commit()`) | `full[h.index]` = the stage just acquired | **correct** |
> | `producer.commit()` (no handle) | `full[state.index]` = the **next** stage | off by one, `(t+1) % stages` |
> | `consumer.release(h)` (or `h.release()`) | `empty[h.index]` = the stage just waited on | **correct** |
> | `consumer.release()` (no handle) | `empty[state.index]` = the **next** stage | off by one |
>
> What this class of errors has in common: **numerics all correct, no deadlock (when the tile count is even), just slow** — double buffering is
> knocked back to single buffering, and every other tile the consumer is released a whole tile late. It raises no alarm by itself; it can only be recognised by its timeline
> fingerprint (see "When needed").

## Purpose

Every pipeline in `cutlass.pipeline` (`PipelineUmmaAsync`, `PipelineTmaUmma`, `PipelineAsync` …)
hands out a pair of participant objects through `make_participants()`. Each participant holds its own `PipelineState`
(`index` + `phase` + `count`), and **every** `acquire_and_advance` / `wait_and_advance` call does two things:

1. wait on the barrier the **current** state points to (the producer waits on `empty[index]`, the consumer on `full[index]`),
   then **copy** the current state into an immutable handle and return it to you;
2. `advance()` the participant's own state by one (`index += 1`, `phase ^= 1` on wrap-around).

So from the moment this call returns, **the participant's state already points to the next stage**; only the handle still remembers
which one you just obtained. Both `commit` and `release` have two spellings: the one with a handle uses the state in the handle,
the one without uses the participant's **current** state. Mixing these two APIs is the off-by-one.

This entry's target is Blackwell's **`PipelineUmmaAsync`** (the producer-side arrive is `tcgen05.commit`,
issued to the mbarrier by the tensor core async proxy after **all earlier** tcgen05 operations retire; the consumer side is a plain
thread arrive) — the most common handshake object for accumulator TMEM double buffering: the MMA issuer is the producer,
the epilogue is the consumer. But the API shape is the same for every `PipelineProducer` / `PipelineConsumer`.

## When needed (trigger)

**Structural trigger** (deterministic; check this first): the code contains both

1. `xxx.acquire_and_advance(...)` (or `wait_and_advance`), with or without its return value captured;
2. right after it (at the end of the same tile) an `xxx.commit()` / `xxx.release()` **without arguments**.

If 2 holds, it is off by one; no run is needed to prove it. The correct form is: `h = xxx.acquire_and_advance(tok)`
… `xxx.commit(h)` (or `h.commit()`).

**Symptom trigger** (to look back at once it has already run) — taking a 2-stage accumulator double buffer as the example, the producer is the MMA
issuer and the consumer is the epilogue:

| Reading | Correct form | Off by one |
|---|---|---|
| correctness | PASS | **PASS** (values not wrong, just late) |
| consumer's full wait | each tile waits only for its own commit; wait duration unimodal | **bimodal, alternating tile by tile**: even tiles wait until the **next** tile's commit (p75 tens to hundreds of times p25), odd tiles barely wait |
| producer's acquire wait (on `empty`) | waits for the epilogue of tile t−2, usually already done | likewise **bimodal, alternating**: the epilogue of even tiles is delayed by a whole tile, so the `empty` it releases is also a whole tile late |
| tensor occupancy | full | a chunk lower: every other tile, the epilogue does not overlap the MMA at all |
| odd tile count | normal | **the consumer of the last tile never gets released** (the commit it waits for never happens) → hang |
| stage count S > 2 | normal | one tile in every S has to wait for the commit of t+S−1 (same arithmetic; this entry is certified only at S=2) |

Criterion to tell "off by one" from "the barrier was built wrong": off by one **gives no wrong numbers and does not hang with an even tile count** — because
`tcgen05.commit` tracks **all** earlier issued tcgen05 operations, the late commit also covers
this tile's MMA, so the data the consumer gets is complete.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

STAGES = 2                      # accumulator double buffering
CONSUMER_THREADS = 128          # every consumer-side thread releases: the arrive count of empty

@cute.kernel
def kern(...):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    smem = cutlass.utils.SmemAllocator()
    # full[0..S-1] immediately followed by empty[0..S-1]; create() initialises them with this layout itself
    mbar = smem.allocate_array(cutlass.Int64, 2 * STAGES, byte_alignment=16)

    # every thread must reach this point: create() includes mbarrier init + fence + a whole-CTA sync
    prod, cons = pipeline.PipelineUmmaAsync.create(
        num_stages=STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, CONSUMER_THREADS),
        barrier_storage=mbar,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),   # single CTA; for a pair pass the real vmnk
    ).make_participants()

    if warp_idx == 0:                       # ---- producer (MMA issue warp) ----
        for t in cutlass.range(num_tiles):
            tok = prod.try_acquire()        # one non-blocking probe (optional)
            h = prod.acquire_and_advance(tok)   # wait on empty[h.index]; state already advanced
            buf = h.index                   # this tile's accumulator buffer index == the stage just acquired
            ... issue all of this tile's MMAs on TMEM buffer buf ...
            prod.commit(h)                  # tcgen05.commit -> full[h.index]  ✅
            # prod.commit()                 # ❌ lands on full[(t+1) % STAGES]: off by one
        # before exiting, wait back the last used empty, so no dangling arrive remains after the kernel exits
        prod.tail()

    if warp_idx >= 4:                       # ---- consumer (epilogue warps) ----
        for t in cutlass.range(num_tiles):
            tok = cons.try_wait()
            hc = cons.wait_and_advance(tok) # wait on full[hc.index]; state already advanced
            buf = hc.index                  # the buffer index read is the same sequence as the producer's h.index
            ... read back from TMEM buffer buf, write out ...
            cons.release(hc)                # arrive empty[hc.index]  ✅
            # cons.release()                # ❌ lands on empty[(t+1) % STAGES]: off by one
```

Three invariants, none optional:

1. **commit / release go only by the handle**: `prod.commit(h)`, `cons.release(hc)` (or `h.commit()` /
   `hc.release()`). The two argument-less spellings **always** point to the wrong stage after `xxx_and_advance`.
2. **The buffer index comes from the handle**: `h.index` / `hc.index`. Both sides get the same sequence
   `0,1,0,1,…` from their own handles — that is what "double buffering" means; taking it from the participant state is off by one in the same way.
3. **The producer's arrive is `tcgen05.commit`**: it is issued by the async proxy after all earlier tcgen05 operations retire;
   that is why the off-by-one still "gives no wrong numbers" — and also why it stays hidden.

## Gotchas

- **The argument-less `commit()` / `release()` are not shorthands**; they are meant for the split spelling "`acquire()` first, then a manual
  `advance()`", where the state has not yet advanced at commit time. Once you use
  `acquire_and_advance` / `wait_and_advance`, only the spelling with a handle is valid.
- **It compiles, runs and passes correctness.** The only run-time alarm is the hang with an odd tile count; with an even tile count
  it is just "about 5% slower, tensor occupancy a few points lower, wait durations bimodal". When the epilogue's full wait
  and the issuer's acquire wait **both** show tile-by-tile alternating bimodality, check this entry first.
- **Both call sites must be the same bijection**: the buffer index the producer writes and the one the consumer reads must both come from
  their own handles. Changing only one side (e.g. the producer uses `h.index`, the consumer its own computed `t & 1`) happens to agree with
  2 stages and goes out of alignment at 3 stages.
- **`tail()` is still required**: before the producer exits, `prod.tail()` (wait on the last used empty), otherwise after the kernel
  exits an async arrive may still land on freed smem. `tail()` uses the participant's current state and needs no
  handle — that is how it differs from commit.
- **How to verify**: after switching to the spelling with a handle, use a timeline probe to look at the producer's acquire wait and the
  consumer's full wait: **the bimodality should disappear and p75 should collapse to the same order as p25** (each tile leads by only one).
  A numerically equivalent variant that "only switches commit to the handle form, changing nothing else" is the cheapest way to price it; on a
  2-buffer accumulator / epilogue handshake its magnitude is a few percent of device time.
- **If generic-proxy smem writes before the commit must be visible to the consumer, use
  `cute.arch.fence_view_async_shared()`** (= `fence.proxy.async.shared::cta`). The producer's
  arrive goes through the async proxy and does not order earlier plain st.shared for you. Mind the spelling:
  `cute.arch.fence_proxy("async", space=...)` fails IR verification on this image
  (`only async_shared fence can have space attribute`), an ICE at compile time.
- **Bounded spins must exit early with `while`, not `for i in cutlass.range(N)`**: `cutlass.range` is a fixed-count
  `scf.for`; after the token turns true the remaining iterations still run, and the tail can reach tens of µs. Used as a timeline /
  progress probe it reads "released immediately" as "one beat late"; used as a real wait it burns a stretch of issue time for nothing. Spelling:
  `go = Int32(1); while go == 1: if tok == 0: tok = x.try_wait(); i += 1 else: go = Int32(0)`,
  plus an exit on `i >= CAP`.
- Combination with a pair (`cta_group=2`): `PipelineUmmaAsync.create(cta_layout_vmnk=...)` automatically turns the
  commit into a multicast to both halves of the pair and sends the release to the leader; this entry's handle rule is unchanged;
  for who is entitled to wait see [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md).

## Sources

- Official API: NVIDIA CUTLASS CuTe DSL `cutlass.pipeline` module — `PipelineProducer.acquire_and_advance`
  / `PipelineProducer.commit(handle)` / `PipelineProducer.ImmutableResourceHandle.commit`,
  `PipelineConsumer.wait_and_advance` / `PipelineConsumer.release(handle)`,
  `PipelineUmmaAsync.create(num_stages, producer_group, consumer_group, barrier_storage,
  cta_layout_vmnk)`, `PipelineState.advance`. DSL docs:
  https://docs.nvidia.com/cutlass/media/docs/pythonDSL/
- PTX: `tcgen05.commit` (tracks all earlier async tcgen05 operations and arrives on the mbarrier when they complete)
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit ;
  `mbarrier.try_wait.parity`
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
- GPU certification probe (the probe is not included in this release) — six contracts,
  one subprocess each, every wait a bounded `try_wait` spin: `stage_prod_handle` (one
  `acquire_and_advance` + `commit(h)` → `full[0]` flips, `full[1]` does not), `stage_prod_bare`
  (same + `commit()` → `full[1]` flips, `full[0]` does not), `stage_cons_handle` / `stage_cons_bare`
  (consumer side symmetric, looking at `empty[0]` / `empty[1]`), `loop_handle` (8 tiles, the producer commits once every 100 µs,
  both sides take `%globaltimer` timestamps, and the host counts how many commits had been issued when each consumer wait returned:
  exactly `t+1`), `loop_bare` (same but with `commit()`: the reading is `2,2,4,4,6,6,8,8` — even tiles a whole tile
  late, odd tiles released immediately, slot values still exact one by one).
  **Status: ✅ GPU-certified**, final verdict `PROBE PASS max_diff=0.0` (certification round 3; all six contracts passed in the same round, `loop_bare` reads exactly `2,2,4,4,6,6,8,8`, `loop_handle` exactly `t+1`). What failed in the first two rounds were the two loop cases that measure "how late", not the mechanism: in round 1, `loop_handle` and `loop_bare` both failed and neither emitted a reading; in round 2, only `loop_bare` failed, reading `2,3,4,5,6,7,8,8` (`max_diff=1`, the tail of a fixed-count spin). The four stage contracts passed in every round.
- **KEEP landing** (NCU dossier census + gate window; watermark confirmation run reads 0.3474 ms):
  the single change `acc_producer.commit()` → `acc_producer.commit(h)`, shipped in the **same package** as operand ring 4 → 6 — gate 0.400 → 0.3481 ms
  (−12.97%), `tensor_pct` 81.39 → 89.2, `stall_long_sb` 18.8 → 14.04, same-job reference ratio 0.901 → 1.0079; axis `rings` 2 → 4 entries
  follows the operand ring depth ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11); this entry's own change has **no footprint** on the contract axes
  (the commit's target slot is run-time state; the SASS is unchanged). Same-package readings are not split; this entry's single-variable price comes from a same-job NCU-clock
  ablation on the same tree before landing (the numerically correct handle form +7.24% clock). The depth share: see
  [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md) gotcha 7.
