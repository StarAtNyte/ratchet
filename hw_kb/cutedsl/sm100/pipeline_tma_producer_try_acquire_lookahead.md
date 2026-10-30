# pipeline_tma_producer_try_acquire_lookahead (one-beat lookahead on `empty` on the TMA supply side: `producer_try_acquire` → `producer_acquire(state, tok)`; participant form `prod.try_acquire()` → `prod.acquire_and_advance(tok)`)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry answers one question only: **on a `PipelineTmaUmma` ring, how does the TMA producer split "wait for the next slot's `empty`"
> into "probe once, non-blocking, right after issuing this beat's loads + at the top of the next beat decide from the probed token whether to really wait", and what this
> token can and cannot save.** It is the producer-side dual of the MMA-side one-beat lookahead (token handed to `consumer_wait(state, tok)`; placement and instruction in
> [`umma_consumer_full_peek_before_commit`](umma_consumer_full_peek_before_commit.md)). Handing the token back itself (`producer_acquire(state, tok)` hides an
> expect_tx arrive, so the acquire cannot be dropped) already has sm90 readings in example (c) of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md);
> what this entry adds is **the placement of the lookahead** and its certification on sm100 `PipelineTmaUmma`.
>
> When the ring is driven through the `PipelineProducer` obtained from `make_participants()`, both the spelling and the **advance timing** differ (you **must not**
> `advance()` before probing); see the "participant form" section at the end. Each form has its own certification probe.

## Purpose

`producer_acquire(state)` does two things by default: (1) **block** on `empty[state.index]` with `state.phase`
(a `mbarrier.try_wait.parity` spin), (2) the elected thread does
`mbarrier.arrive.expect_tx(tx_count)` on `full[state.index]`. Inlining (1) at the top of the k loop means that every beat takes
a full barrier query round trip before issuing its TMA.

`PipelineTmaUmma` (inheriting from `PipelineAsync`) provides the two halves separately:

- `tok = pipe.producer_try_acquire(state)` — one `try_wait.parity` on `empty[state.index]`,
  returning a `Boolean`: true if the slot has been released (this phase completed);
- `pipe.producer_acquire(state, tok)` — when `tok` is true it **skips only (1)** and still does (2); when `tok` is false or
  `None` it falls back to the blocking wait.

So right **after issuing the loads of beat k and advancing the state** you can probe beat k+1's slot, overlapping the probe round trip with the TMA
issue; at the top of the next beat, if the token is already true, go straight to expect_tx + TMA issue.

## When needed (trigger)

- In the supply role's loop body, `producer_acquire(state)` is called **without a token**, one blocking wait at the top of every beat; while the MMA
  side is already in the one-beat lookahead form (token handed to `consumer_wait(state, tok)`) — the two sides are asymmetric.
- Timeline / role-exposure readings: most of the supply role's cycles are spent in the `empty` wait segment, stalls are mostly long-scoreboard
  and fall in the supply loop body; the MMA issuer waits little on `full`.
- The supply loop uses `prod.acquire_and_advance()` from `make_participants()` (without a token) — follow the "participant form" section;
  do not copy the raw-API example below (the advance order is the opposite).

> This completes the **form**; it promises no gain: when `empty` is released is decided by the MMA side, and the lookahead does not make it
> release earlier; it only moves the query round trip off the critical path. The probe certifies semantics only (token truth, the arrive of the acquire),
> no timing; after the change, price it yourself with the same timeline / ablation readings. The participant form has one same-job whole-kernel A/B on a warp-specialized GEMM
> tree: flat (see gotcha P5 of that section).

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync

# Build the ring where all threads reach it (create() includes barrier init + fence + CTA/cluster sync)
pipe = pipeline.PipelineTmaUmma.create(
    num_stages=STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, N_EMPTY_ARRIVES),
    tx_count=STAGE_BYTES,
    barrier_storage=storage.ab_mbar.data_ptr(),
    # cta_layout_vmnk=...   # pass the cluster form unchanged; this entry does not touch it
)

# ---------------- supply warp (whole warp executes; arrive / TMA issue are already elected inside the DSL) ----------------
if warp_idx == WARP_TMA:
    ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
    tok = pipe.producer_try_acquire(ps)                 # before the loop: probe once for k = 0
    for kb in cutlass.range(K_BLOCKS, unroll=1):
        pipe.producer_acquire(ps, tok)                  # tok true: skip the empty wait; arm full either way
        bar = pipe.producer_get_barrier(ps)
        cute.copy(tma_x, tXgX[(None, kb)], tXsX[(None, ps.index)], tma_bar_ptr=bar)
        # ... the other TMAs of the same beat (same bar) ...
        ps.advance()                                    # advance first
        tok = pipe.producer_try_acquire(ps)             # then probe: the token belongs to the (index, phase) of k + 1
    pipe.producer_tail(ps)                              # tail unchanged

# ---------------- MMA issue warp (the certification probe's consumer side is a plain consumer_wait; MMA-side lookahead is outside this entry's certified surface) ----------------
if warp_idx == WARP_MMA:
    cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
    for kb in cutlass.range(K_BLOCKS, unroll=1):
        pipe.consumer_wait(cs)
        ... cute.gemm(...) ...
        pipe.consumer_release(cs)                       # tcgen05.commit -> empty[cs.index]
        cs.advance()
```

For where within the beat the MMA side places its one-beat lookahead on the next beat's `full`, and with which instruction, see [`umma_consumer_full_peek_before_commit`](umma_consumer_full_peek_before_commit.md).
Do not write `consumer_try_wait(nxt)` before this beat's `cute.gemm`: when the next beat has not landed, it waits until the phase completes or the time limit expires, and this beat's UMMA is delayed with it;
when any commit (including the accumulator commit at the end of a tile) is still queued behind the lookahead, use `test_wait` (pointer and token hand-back: see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (g));
do not probe after the last beat, see [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md) condition ② / gotcha 11.

Three invariants:

1. **The token and the next acquire use the same `(index, phase)`**: `advance()` first, then `producer_try_acquire`;
   or probe a `clone()`d copy that was advanced, but the next beat's acquire must then use exactly that state.
2. **`producer_acquire(state, tok)` must always be called**: the token only replaces the `empty` wait; the
   `arrive.expect_tx` on `full` is still issued by it.
3. **Still correct when the token is false**: it falls back to the blocking wait, data bit-identical; the lookahead affects only timing, not semantics.

## Gotchas

1. **Probe before advancing = probe the wrong slot.** `producer_try_acquire(ps)` before `ps.advance()` probes the slot **just acquired**,
   which is of course free (the token is always true), so the next beat skips its wait carrying a true token that does not belong to it —
   the TMA may overwrite a slot the MMA is still reading. This is a **silent** error: when the MMA reads fast the values happen to be right; they go wrong only when the MMA
   falls behind the supply. Structural criterion: there must be exactly one `advance()` between the probe and the acquire. (This one is derived from
   the parity semantics; the certification probe has no dedicated wrong-order case.)
2. **Dropping `producer_acquire` once you have a true token = hang.** What is dropped is not only the wait but also the
   arrive.expect_tx on `full`: the TMA's complete_tx still arrives, but `full` is missing that one arrive, so the phase can never
   complete, and the MMA side waits on `consumer_wait` forever (probe case `stage_skip_acquire`: `full[0]` does not flip within bounded
   polling).
3. **A fresh producer state gets true for the first `STAGES` probes and false for probe `STAGES+1`.**
   `make_pipeline_state(Producer, S)` starts with `phase = 1`, while a freshly initialised `empty` has not completed any phase;
   by parity semantics, a wait on "the preceding phase" is satisfied immediately. So the first lap gets true tokens without any release;
   after wrapping around the first lap (`phase` becomes 0) a real release is needed. Do not initialise a producer state to phase 0 yourself.
4. **`try_wait` is not a pure query.** PTX `mbarrier.try_wait` allows the thread to suspend and wait for an implementation-defined time limit
   (the pure query is `mbarrier.test_wait`). So "probing once" can itself take some time; the certification probe recorded the duration of every
   `producer_try_acquire` as a fact row (`%globaltimer` readings 32–64 ns, i.e. the order of the timer resolution),
   to judge whether the lookahead really is cheaper than the inline wait — do not go by intuition. Note: if the end timestamp does not depend on the probe's predicate, such a bracket
   cannot read a wait even when the phase has not completed ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 14); whether these 32–64 ns
   depend on the predicate was not checked, so read them only as "the issue overhead of the probe itself".
5. **The extra probe after the last beat is harmless — on the producer side only.** The probe for beat `K_BLOCKS` at the end of the loop only reads the barrier; it does not
   arrive and does not change the phase; `producer_tail(ps)` still finishes with its own blocking acquire, and the token is not passed to it
   (all three loop cases of the probe ran to completion with this extra probe, data bit-exact). It is harmless because the `empty`
   release it probes (the `consumer_release` of beat `K_BLOCKS + 1 − STAGES`) is guaranteed to happen. **The consumer side is not analogous**:
   a `consumer_try_wait` after the last beat asks about a fill that does not exist and sits out the full `try_wait` time limit, see
   [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md) condition ② and gotcha 11 (applies equally to single CTA, `CtaGroup.ONE`).
6. **Cluster / multicast forms are not in the certified surface.** The probe is single-CTA; by API semantics the token probes this CTA's
   `empty[index]`; the `empty` arrive count is set by `create()`'s `consumer_group`, and `is_leader_cta` decides who does
   expect_tx, all independent of the lookahead — but the lookahead in pair / multicast forms **has no GPU reading**; add a case yourself before using it.
7. **The token is a loop-carried variable**: give it its initial value with one real `producer_try_acquire` before the loop (its type is DSL
   `Boolean`) and reassign it in the loop body — that is how the certification probe writes it. Using a Python `True` as the initial value and then turning it into a DSL value inside a dynamic
   loop was not tested; do not do it.
8. **It does not make `empty` release earlier.** The release time is set by the MMA side's `consumer_release` (`tcgen05.commit`,
   which arrives only after all earlier tcgen05 operations have retired). If the producer is waiting on the MMA's real consumption progress,
   the lookahead only removes the query round trip; to free slots earlier, look at the ring depth ([`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md))
   or the placement of the MMA side's release.

## participant form: `prod.try_acquire()` → `h = prod.acquire_and_advance(tok)`

> **Status: ✅ Verified** — certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (certification round 1, all five cases PASS). Same certified surface as above: single CTA,
> no cluster / multicast, consumer side is a plain `cons.wait_and_advance()`; semantics only, no timing.

When the ring is driven by the `PipelineProducer` obtained from `pipe.make_participants()` (`prod` below), the participant **holds the state itself**, and
`acquire_and_advance` advances it within the same call (the handle records the slot just obtained, see
[`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md)). So the placement of the lookahead is exactly the opposite of the raw API:

| | raw API (above) | participant |
|---|---|---|
| Probe | `pipe.producer_try_acquire(ps)`: probes the slot `ps` points to | `prod.try_acquire()`: probes the slot the participant's **current** state points to, without advancing |
| Block + expect_tx | `pipe.producer_acquire(ps, tok)`, does not advance | `h = prod.acquire_and_advance(tok)`: tok true skips only the `empty` wait, expect_tx is still done, **then advances** |
| The next beat's probe goes | **after** issuing this beat's TMA and `ps.advance()` | **directly** after issuing this beat's TMA, with no `advance()` in between |
| This beat's slot / barrier | `ps.index` / `producer_get_barrier(ps)` | `h.index` / `h.barrier` |

```python
pipe = pipeline.PipelineTmaUmma.create(...)          # same as the example above
prod, cons = pipe.make_participants()

if warp_idx == WARP_TMA:
    tok = prod.try_acquire()                         # before the loop: probe once for beat 0 (a DSL Boolean as the initial value)
    for kb in cutlass.range(K_BLOCKS, unroll=1):
        h = prod.acquire_and_advance(tok)            # tok true: skip the empty wait; arm full and advance either way
        cute.copy(tma_x, tXgX[(None, kb)], tXsX[(None, h.index)], tma_bar_ptr=h.barrier)
        # ... the other TMAs of the same beat (same h.barrier) ...
        tok = prod.try_acquire()                     # state already advanced -> this probes the next beat's slot; no advance() here
    prod.tail()
```

Gotchas (participant form):

- **P1. There must be no `advance()` between `try_acquire()` and the next `acquire_and_advance(tok)`.** Copying the raw API's
  "`advance()` first, then probe" probes the slot two beats ahead and advances the state one extra step, so every later beat's `h.index` / phase is misaligned.
  This one is derived from the public source (`try_acquire` uses the participant's current state, `acquire_and_advance` = `acquire` + `advance`);
  the probe has no wrong-order case. What the probe proves is the correct order: in the `loop_prep_paced_consumer` case (the consumer spins 100 µs before each wait) the tokens are
  `[1]*4 + [0]*12` and the data is bit-exact — if it were probing the slot just acquired, the token would always be true.
- **P2. A true token must also go through `acquire_and_advance(tok)`.** Probe case `stage_advance_only`: after getting a true token, only
  `prod.advance()` and then the TMA issue — `full[0]` does not flip within bounded polling (the expect_tx arrive is missing); same mechanism as gotcha 2 above.
- **P3. A fresh participant gets true for the first `STAGES` probes and false for probe `STAGES+1`** (`stage_fresh_tokens`: `[1,1,1,1,0]`),
  same as gotcha 3 above: true tokens on the first lap say nothing about any progress on the consumer side.
- **P4. This beat's slot index is known before blocking.** With exactly one `acquire_and_advance` per beat, `h.index` is determined only by "which acquire this is":
  a loop-carried counter starting at 0 and wrapping at `STAGES` equalled `h.index` on every beat in both loop cases (difference always 0, for both cadences: all tokens true, and
  true for the first 4 beats then all false). `prod.current_handle().index`, read before blocking, should give the same value according to the 4.6.x public source (it builds the handle from the participant's current state; not tested on GPU), but this method does not exist
  in every version: in the cutlass-dsl 4.4.2 of the certification job, `PipelineProducer` does not have it (the probe prints `current_handle=False`),
  while the 4.6.0 / 4.6.1 public source does. The counter spelling works on both.
- **P5. Writing the slicing / slot views before the blocking point does not necessarily change the SASS; whole-kernel readings are flat.** On a warp-specialized GEMM tree, landing this section's
  lookahead together with "write each beat's gmem slices and smem slot views (using P4's counter) before `acquire_and_advance(tok)`",
  against the original spelling (`acquire_and_advance()` without a token, slicing afterwards), in a same-job whole-kernel A/B: geometric mean over four workloads
  `OFF/ON` = 1.0001 (flat). In the disassembly of both spellings, there are 16 instructions between the blocking `SYNCS.PHASECHK.TRANS64.TRYWAIT` and the first `UTMALDG`
  — the address arithmetic moved earlier in the source did not shorten the "wait returns → issue TMA" chain; what the lookahead form adds is another
  `TRYWAIT` (the probe) and an `@P BRA` that skips the blocking wait when the token is true. Read the SASS before changing the layout; do not infer it from source order. Only pure address / index computation
  can go before the blocking point; writing smem slots, issuing TMA and arriving must all stay after `acquire_and_advance`
  (a TMA issued early may overwrite a slot the consumer side is still reading; not tested).
  Two separate probes measured this chain (single CTA, two fp32 boxes recorded on the same `full`, consumer is a generic load): in the `PipelineTmaUmma`
  form, writing both gmem / smem slices and `producer_get_barrier(state)` before `producer_acquire(state)` took acquire return → second
  TMA issued from 27 → 20 cycles, constant per beat, the same whether the consumer spins every beat and the `empty` wait really blocks or not; the length of the acquire's own wait was the same for both spellings;
  in a hand-written `full[]` / `empty[]` ring (ring depth not a power of 2), switching stage / parity to loop-carried ([`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) (B))
  and writing the slices and both barrier addresses before the wait took wait return → last issue from 44 → 43 cycles. The hand-written ring form had a
  same-job whole-kernel A/B on a warp-specialized GEMM tree: geometric mean over 4 workloads 13.176 → 13.178 µs (flat); the `PipelineTmaUmma` form has no same-job whole-kernel A/B. I.e.: the saving is single-digit to a few cycles, far smaller than one `empty` wait.
  `arrive.expect_tx(full[s])` cannot be moved earlier: only after `empty[s]` lets it through is it guaranteed to land on the correct phase of `full[s]`;
  moved earlier, it may be counted into the previous phase (derived from PTX phase semantics, not tested on GPU).
- Gotcha 4 (`try_wait` is not a pure query), gotcha 6 (cluster / multicast not in the certified surface), gotcha 7 (initialise the token with a real probe) and
  gotcha 8 (it does not make `empty` release earlier) hold equally for the participant form. Probe readings: on beats with a true token,
  `acquire_and_advance(tok)` reads 0–64 ns (the order of the `%globaltimer` resolution); with a false token the blocking fallback takes about 95 µs per beat,
  i.e. the cadence of the slowed-down consumer side.

## Sources

- Public infra API: CUTLASS CuTe DSL `cutlass.pipeline` — `PipelineTmaUmma.create`,
  `PipelineAsync.producer_try_acquire(state)`, `PipelineTmaUmma.producer_acquire(state, try_acquire_token)`,
  `PipelineAsync.producer_get_barrier`, `PipelineAsync.producer_tail`, `PipelineAsync.consumer_try_wait`,
  `PipelineAsync.consumer_wait(state, try_wait_token)`, `PipelineTmaUmma.consumer_release`,
  `make_pipeline_state(PipelineUserType.Producer | Consumer, stages)`, `PipelineState.advance / clone`;
  `cute.arch.mbarrier_try_wait`. DSL docs: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/
- PTX: `mbarrier.try_wait.parity` (may suspend up to a time limit) and `mbarrier.test_wait` (pure query),
  `mbarrier.arrive.expect_tx`, `cp.async.bulk.tensor … mbarrier::complete_tx::bytes`
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
- GPU certification probe (lookahead form) — single CTA, 4 stages, float32 (8, 32) slots, 16 beats;
  six cases, one subprocess each: `stage_fresh_tokens` (first S true, probe S+1 false),
  `stage_token_arms` (true token + acquire → `full[0]` flips and the slot data is bit-exact), `stage_skip_acquire`
  (true token but acquire dropped → `full[0]` does not flip), `loop_paced_producer` (tokens all true, data bit-exact),
  `loop_paced_consumer` (the consumer spins 100 µs before each wait → tokens = `[1]*S + [0]*(KB−S)`, false tokens take the blocking fallback, data bit-exact),
  `loop_free` (data bit-exact).
  **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, all six cases PASS in the same job: `stage_fresh_tokens` tokens=`[1,1,1,1,0]`; `stage_token_arms` `full[0]` flips, slot maxdiff 0; `stage_skip_acquire` `full[0]` does not flip; `loop_paced_producer` tokens all 1; `loop_paced_consumer` tokens=`[1]*4+[0]*12` and data bit-exact; `loop_free` tokens all 1. A single `producer_try_acquire` measured 32–64 ns (the order of the `%globaltimer` resolution)).
  **Certified surface**: single CTA, no cluster / multicast, consumer side is a plain `consumer_wait` (not the MMA-side lookahead form); the SASS shape was not censused.
- Public infra API (participant form): `PipelineTmaUmma.make_participants`, `PipelineProducer.try_acquire` /
  `acquire_and_advance(try_acquire_token)` / `advance` / `tail` / `current_handle` (present in the 4.6.x source, absent in 4.4.2),
  `PipelineProducer.ImmutableResourceHandle.index / barrier`, `PipelineConsumer.wait_and_advance`,
  `PipelineConsumer.ImmutableResourceHandle.release`.
- GPU certification probe (participant form) — same geometry as the previous one (single CTA, 64 threads, 4 stages, float32 (8, 32) slots, 16 beats), five cases, one subprocess each:
  `stage_fresh_tokens` (`(try_acquire, advance) × (S+1)` → `[1,1,1,1,0]`), `stage_token_arms` (true token →
  `acquire_and_advance(tok)` → TMA: `full[0]` flips on the 0th poll, `h.index = 0`, slot data diff 0), `stage_advance_only`
  (true token, only `advance()`: `full[0]` does not flip), `loop_prep_free` (16 beats all true tokens, predicted slot index minus `h.index` always 0, data bit-exact),
  `loop_prep_paced_consumer` (tokens `[1]*4+[0]*12`, slot-index difference always 0, data bit-exact).
  **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, round 1, all five cases PASS; cutlass-dsl 4.4.2 inside the job).
  P5's whole-kernel readings and disassembly instruction counts come from a same-job A/B on another tree, not from this probe's contracts.
- GPU certification probe (issue-chain hoist) — `PipelineTmaUmma`, single CTA, two fp32 (8, 32) boxes,
  4 stages, 16 beats; four cases: slicing after / before the acquire × consumer spinning or not, both paths bit-exact. Verdict: ✅ `PROBE PASS max_diff=0.0` (round 1).
- GPU certification probe (prep before the `empty` wait) — hand-written `full[]` / `empty[]` ring (ring depth not a power of 2), single CTA,
  two fp32 (8, 32) boxes; four cases: `t % S` form vs carried `(s, ph)` + slicing hoisted × consumer spinning or not; data bit-equal, per-beat `(s, ph)` consistent with
  `(t % S, ((t // S) & 1) ^ 1)` (including at the wrap). Verdict: ✅ `PROBE PASS max_diff=0.0` (round 2).
