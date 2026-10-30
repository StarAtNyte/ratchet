# cta_pair_consumer_peek (two guards on the consumer-side peek: who is entitled to ask × is there a next wait)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry answers one question only: **under what conditions should a non-blocking peek (`consumer_try_wait`) be issued at all.**
> The answer is **two mutually independent conditions holding at the same time**; missing either one asks the same kind of thing -- a **phase that never
> flips**:
>
> | | Condition | Shape when it is missed |
> |---|---|---|
> | **(1) Entitlement** | on a `cta_group=2` pair the full barrier is **armed only on the leader**, so only the leader is entitled to ask | a bare peek on the follower, **paid once per k-tile** |
> | **(2) Existence** | a peek prefetches the token for **the next `consumer_wait`**; if that wait will not happen, there is nothing to ask | a last-round peek, **paid once per kernel call** |
>
> And PTX's `mbarrier.try_wait` is a **potentially blocking** instruction: when the phase is not complete it suspends the thread up to the system-default
> suspend time limit before reporting false. So "asking about a phase that never flips" is not a wasted instruction; it is **billed by the time limit**.
>
> "How to encode the wait" (suspending ↔ non-blocking + explicit spin) see
> [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md); "whether to use an mbarrier
> rather than an all-hands rendezvous" see [`handoff_sync_choice`](../sm90/handoff_sync_choice.md). These are all orthogonal:
> with the right primitive, the right encoding and **entitlement** guarded, **existence** can still be the guard that is wrong -- and the common
> feature of this whole class is **no deadlock, no wrong numbers, only slowness**, so it does not raise an alarm by itself.

## Purpose

`tcgen05` MMA has a `cta_group=2` variant: a CTA pair completes one MMA together, and **only the leader
CTA issues it**. So the AB pipeline within a pair is **asymmetric** at the barrier level, leaning in opposite directions on the two sides:

| Barrier | Who arms it | Who is entitled to wait on it | Who must deliver into it |
|---|---|---|---|
| **full** ("the data has all arrived") | **only the leader**, armed with the `tx_count` of the whole **pair** | **only the leader** | both CTAs (both must send their TMA completion to **the leader's** barrier) |
| **empty** ("the buffer has been freed") | the consumer side arrives on **both** CTAs of the pair when releasing | **both CTAs** | the leader (releases once it has consumed) |

`consumer_*` faces the full barrier. So **every** consumer-side call -- including the seemingly
harmless **non-blocking peek** -- must sit inside the leader guard; the producer side is exactly the opposite:
`producer_acquire` / `producer_try_acquire` must **run on both CTAs**, and adding a leader guard to them means the
follower can never issue its TMA, a direct deadlock. **The guards on the two sides point in opposite directions; this is the point of this entry most easily remembered backwards.**

Condition (2) has nothing to do with the pair; it is a general pipeline invariant, but here it collides with (1) on the same line of code, so it is written up
in this entry too: **the only purpose of a peek is to save the next wait one suspension**. Without a next wait, this peek has no
corresponding arrive -- it lands on exactly the same hardware path as a bare peek on the follower.

## When needed (trigger)

**Structural trigger** (look at this first; it is deterministic): the code contains all of

1. `tcgen05.CtaGroup.TWO` (or a pair shape of the kind where `mma_tiler[0]` within the cluster is twice the single-CTA row count);
2. a supply shape in which **the pair shares one full barrier** -- a ready-made pipeline (of the `pipeline.PipelineTmaUmma`
   kind; **note that this class is not present in every DSL build, so `dir()` it first**) or a hand-written equivalent;
   for the delivery side see [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md);
3. any `consumer_try_wait`.

**Precondition**: condition (1)'s "only the leader is entitled to wait" holds **only if** the pair really shares one barrier,
and that has to be built by `.cta_group::2` on the load side. If the supply has the shape "each CTA arms and waits on its own barrier",
this entry's (1) does not apply ((2) still does).

Then **go through every call site with two checkboxes**, one column per checkbox; do not merge them into "have I guarded it or not":

| Call site | (1) inside `if is_leader`? | (2) is there really another `consumer_wait` after it? |
|---|---|---|
| end of the prologue | required | holds when `k_tile_cnt >= 1` |
| end of the steady-state loop | required | always holds when the prefetch depth ≥ 1 |
| **end of the drain loop** | required | **does not hold in the last round** -- this cell is most often left empty |
| **end of the warp-specialized MMA loop** (a single loop of only wait / MMA / release) | required | **does not hold on the last beat** -- the loop body is symmetric, so it is most easily copied into the last beat |

**Symptom triggers** (only looked at once it has already run). The fingerprints of the two kinds of omission **differ**; classify by this first:

| | (1) entitlement missed | (2) existence missed |
|---|---|---|
| cost scale | **proportional to the number of k-tiles** -- the pipeline was never really built | **a fixed amount per kernel call**, independent of the number of k-tiles |
| tensor duty cycle | collapses to single digits | **normal**, short only by the last few percent |
| stall location | spread over the whole main loop | concentrated in the stretch **after the main loop, before the epilogue** |
| dominant stall | `long_scoreboard` | `long_scoreboard` |

The criteria common to both that separate this class from "the barrier is built wrong":

- the results are **numerically correct** and there is **no deadlock** -- this is the most important one;
- `dram` is also low (not a memory bottleneck), and `warp_active` is as low as the tensor duty cycle;
- the spin region has **very few instructions** (a handful), with `NANOSLEEP` in SASS -- a few instructions taking most of the samples means it is
  burning time in place, not computing;
- in pricing it is an **independent** lever: adding the guard alone cuts away that stretch of exposed spin.

**The fastest way to tell (2)**: double the number of k-tiles and measure again. The overhead of an entitlement miss doubles with it; that of an existence miss does not move
-- it is a fixed suspension **once per call**.

**Anti-trigger**: on a `cta_group=1` pipeline where both CTAs arm their own barriers, condition (1) does not apply (condition (2)
still applies; it has nothing to do with the pair); likewise for a single CTA. **Condition (2) has its own
GPU certification and whole-kernel readings on single-CTA / `CtaGroup.ONE` / cluster-multicast-supply pipelines, see gotcha 11** -- do not skip this entry for such kernels because the title says pair: count the calls of `consumer_try_wait`
in the consumer-side loop; if there are more than `consumer_wait` (with the loop running `NUM_KB` rounds, the correct answer is 1 before the loop + `NUM_KB − 1` inside it), condition (2) was missed.

## Mechanism: why both kinds of peek cost money

Three layers; missing any one leaves the cost unexplained.

**Layer 1: the follower's phase really never flips.** In the shape where the pair shares one barrier,
the full barrier's `arrive_and_expect_tx` runs once under the leader predicate, arming the bytes of the whole pair;
the follower's local full barrier is never arrived on even once. So asking about it on the follower is asking
a stagnant pond.

**Building this shape is itself another mechanism, not one line of pointer arithmetic here**: delivering the completion signal to the peer's
barrier requires the load op to be built on `CtaGroup.TWO` -- under the default `.cta_group::1`, redirecting to the peer is UB per the
PTX spec (no signal, no error, the barrier never fills). `map_dsmem_ptr(bar, leader_rank)`
(`leader_rank = cta_rank - coord_vmnk[0]`) is **one link** of that mechanism, not all of it;
miss any of the four requirements and it hangs, see [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md).
**This entry picks up from where that shape has already been built.**

**Layer 2: the last round's phase never flips either -- on the leader as well.** `consumer_try_wait`
does not advance state; what it asks about is **the next filling of the slot the current consumer state points to**. The whole
kernel has only `k_tile_cnt` fillings, so after the `k_tile_cnt`-th tile has been consumed,
the consumer state points to **the `(k_tile_cnt + 1)`-th filling** -- which does not exist, and no TMA will ever
arrive on it. At this point it no longer matters whether the peek is on the leader: **the entitlement is right, but there is nothing left to ask.**

In the split shape (prologue / steady state / drain) this cell is **the last round of the drain loop**, because
each drain round consumes exactly one tile, and the last round consumes the last tile. It is especially easy to miss because,
when writing it, it looks like just "the fixed tail of the loop body", while the loop's **end point** is exactly where the invariant breaks.

**Layer 3: `try_wait` is not a free query.** PTX's `mbarrier.try_wait` is a **potentially blocking** instruction:
when the phase is not complete it may suspend the executing thread until `suspendTimeHint`, or a **system-default limit**,
expires before reporting false. The real "ask and leave" is `mbarrier.test_wait`. So peeking on a phase that is never armed
is billed at **that time limit** on every call, rather than returning false immediately.

**Where the cost lands differs between the two**:

- **Entitlement miss**: this peek usually sits in **the same warp** as the TMA issue (warp 0 both issues loads and drives the pipeline).
  So the follower's supply issue is strung behind a wait doomed to time out, while what the leader's full barrier waits for
  is exactly the follower's half of the bytes. **The follower spins, the leader starves** -- in the stall ledger "who is spinning"
  and "who is slowed down" are not the same role, and attributing by role leads you toward the consumer depth.
- **Existence miss**: it sits between the main loop and `producer_tail` / the epilogue, and **there is no not-yet-issued
  async work after it that could mask it** -- the pipeline has drained, so this suspension is added purely serially on the critical path, once per
  wave.

## Minimal runnable example (CuTe DSL)

Main-loop skeleton. The placement of the guards and predicates is the whole content; the rest copies the usual pipeline shape:

```python
# Roles within the pair. With the cluster laid out along x, the leader is the one with v==0.
cta_rank = cute.arch.block_idx_in_cluster()
coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
is_leader = coord_vmnk[0] == 0
leader_rank = cta_rank - coord_vmnk[0]      # the rank of the pair's leader in the cluster

# --- prologue: both CTAs issue TMA, both deliver to the LEADER's barrier ---
for _ in cutlass.range_constexpr(prologue):
    ab_pipeline.producer_acquire(ab_prod)                 # no guard: both CTAs must issue
    bar = ab_pipeline.producer_get_barrier(ab_prod)       # returns the local one
    bar = cute.arch.map_dsmem_ptr(bar, leader_rank)       # redirected to the leader's. Valid only when the atom
                                                          # is built on CtaGroup.TWO
    cute.copy(tma_atom_a, tAgA[...], tAsA[(None, ab_prod.index)], tma_bar_ptr=bar)
    cute.copy(tma_atom_b, tBgB[...], tBsB[(None, ab_prod.index)], tma_bar_ptr=bar)
    ab_prod.advance()

peek_empty = ab_pipeline.producer_try_acquire(ab_prod)    # no guard: empty is arrived on both CTAs

# peek_full must be pre-bound outside the region: names newly created inside a branch of a run-time if do not escape
# (see cute_dsl_const_expr_if.md).
peek_full = cutlass.Boolean(False)
if is_leader:                                 # (1) entitlement; (2) existence: the first wait comes right after
    peek_full = ab_pipeline.consumer_try_wait(ab_cons)

# --- steady state: issue tile k+P first, then wait for tile k ---
for _ in cutlass.range(k_tile_cnt - prologue, unroll=1):
    ab_pipeline.producer_acquire(ab_prod, peek_empty)
    bar = cute.arch.map_dsmem_ptr(ab_pipeline.producer_get_barrier(ab_prod), leader_rank)
    cute.copy(tma_atom_a, tAgA[...], tAsA[(None, ab_prod.index)], tma_bar_ptr=bar)
    cute.copy(tma_atom_b, tBgB[...], tBsB[(None, ab_prod.index)], tma_bar_ptr=bar)
    ab_prod.advance()

    if is_leader:                                         # the consumer side is leader-only as a whole
        ab_pipeline.consumer_wait(ab_cons, peek_full)
        # <<< consume slot ab_cons.index here (issue MMA / read the stage)
        ab_pipeline.consumer_release(ab_cons)             # arrive on the empty of both CTAs
        ab_cons.advance()

    peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
    if is_leader:                                 # (1) entitlement; (2) existence: the drain still has P waits
        peek_full = ab_pipeline.consumer_try_wait(ab_cons)

# --- drain: consume only. After the last round there is no next wait, so the last round does not peek ---
# prologue is a host constant, so this loop unrolls at trace time and the predicate folds away directly --
# the last round never issues that try_wait at all.
for d in cutlass.range_constexpr(prologue):
    if is_leader:
        ab_pipeline.consumer_wait(ab_cons, peek_full)
        # <<< consume
        ab_pipeline.consumer_release(ab_cons)
        ab_cons.advance()
    if cutlass.const_expr(d + 1 < prologue):   # (2) existence: no next wait in the last round <<<<<<
        if is_leader:                          # (1) entitlement
            peek_full = ab_pipeline.consumer_try_wait(ab_cons)

ab_pipeline.producer_tail(ab_prod)                        # no guard: both CTAs wind down the producer side
```

**The equivalent spelling when fused into a single loop.** Some pipelines are not split into three stretches; one loop runs the full `k_tile_cnt`, with predicates
controlling issue and consumption. Then `k_tile_cnt` is a run-time value, and condition (2) is written as an explicit predicate alongside condition (1):

```python
for k_tile_idx in cutlass.range(k_tile_cnt):
    if k_tile_idx < k_tile_cnt - prologue:      # there are still tiles to issue
        ...                                     # producer_acquire + two TMAs
    if is_leader:
        ab_pipeline.consumer_wait(ab_cons, peek_full)
        ...                                     # consume + consumer_release + advance
    if k_tile_idx + 1 < k_tile_cnt - prologue:  # (2) existence (producer side)
        peek_empty = ab_pipeline.producer_try_acquire(ab_prod)
    if k_tile_idx + 1 < k_tile_cnt and is_leader:   # (2) existence ∧ (1) entitlement
        peek_full = ab_pipeline.consumer_try_wait(ab_cons)
```

**The warp-specialized shape: MMA issue occupies a warp of its own.** This is the most common spelling of a `tcgen05` pipeline, and also the shape where the existence guard is **most easily missed and most expensive when missed**:
producer and consumer are not in the same loop, the consumer loop has no predicate of the "there are still tiles to issue" kind to borrow as a reminder, and the loop body is
a symmetric "wait → MMA → release → advance → peek" that looks the same on every beat, so the last beat peeks too. **The last beat's peek
lands exactly between the last MMA and the accumulator commit** -- the epilogue must wait for that commit before starting, so this suspension is strung whole onto the critical path.

```python
if warp_idx == MMA_WARP:
    if is_leader:                                         # (1) entitlement: the whole consumer section is leader-only
        acc_pipeline.producer_acquire(acc_state)
        cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
        peek_full = ab_pipeline.consumer_try_wait(cons)   # prologue: when k_tile_cnt >= 1 a wait follows
        for kb in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(cons, peek_full)
            # <<< issue MMA on slot cons.index
            ab_pipeline.consumer_release(cons)
            cons.advance()
            if kb + 1 < k_tile_cnt:                       # (2) existence: no next wait on the last beat <<<<<<
                peek_full = ab_pipeline.consumer_try_wait(cons)
        acc_pipeline.producer_commit(acc_state)           # the epilogue starts from here
```

- `kb` is a run-time loop variable, so (2) is a run-time `if`; `peek_full` is already bound outside the loop, and reassignment inside the branch is carried out normally (a loop-carried variable);
  **do not** rewrite it as `const_expr` for this.
- Another equivalent fix is to peel the last beat out of the loop (`cutlass.range(k_tile_cnt - 1)` + a tail beat without the peek); same effect, at the cost of writing the MMA issue code
  twice.
- **Do not** move the peek to the loop head right before the wait to "get around" the last beat -- then there is no work between the peek and the wait, and the point of prefetching is gone.
- The tail of the **producer warp** of the same shape (the `producer_try_acquire` after `advance()`) asks about the empty of the slot of the `(k_tile_cnt + 1)`-th acquire.
  This slot **does** flip once the consumer releases it; it is not "never flips". **Do not also add `kb + 1 < k_tile_cnt` to it along the way**: in the only whole-kernel A/B,
  adding it was slower (gotcha 10).

The invariant of both spellings is the same: **peek only when "that wait / acquire really will still happen".**
The producer side needs only condition (2) (`try_acquire` must run on both CTAs; there is no entitlement question); the consumer side
needs both. Note that the producer side's (2) subtracts `prologue`: the last **issue** ends `prologue` rounds earlier than the last
**consumption**. The producer side's (2) is derived only from PTX semantics and has no probe certification; on the warp-specialized shape its whole-kernel reading is negative,
see gotcha 10.

The complete runnable file is the certification probe (path in the Sources section): four variants of the same kernel,
differing only in these guards/predicates, with identical geometry, bytes and all other instructions.

## Gotchas

1. **Guarding only the blocking side and missing the non-blocking side (missing condition (1)).** `consumer_wait` written inside `if is_leader:`,
   `consumer_try_wait` bare outside -- it looks like "the consumer is guarded", and it **does not deadlock and the numerics are all right**,
   court is green all the way. It speaks only through the tensor duty cycle. Count call sites; do not count "have I guarded it or not".
2. **Guarding "who asks" and forgetting "is there anything left to ask" (missing condition (2)).** The two guards are **orthogonal**: the leader guard
   answers "which CTA is entitled", the last-round predicate answers "does this ask still have a corresponding arrive".
   `if is_leader: peek()` written at the end of the drain loop body looks like both conditions are in place, but only one is --
   the loop's last round walks right through the invariant. **This guard is especially insidious**: it is worth only one fixed suspension per call, the pipeline looks
   built, the tensor duty cycle is normal, so it is easily filed under "short by the last few percent, add some depth", while
   depth-type levers price at zero on it.
3. **All three call sites must pass both checkboxes.** One at the end of the prologue, one at the end of the steady-state loop, one in the drain loop.
   Missing the entitlement checkbox **pays full price once per k-tile**; missing the existence checkbox **pays once per call**.
4. **Adding the guard to the producer side.** Getting the direction backwards gives a **deadlock**, not slowness: the follower no longer issues
   TMA, and the leader's full barrier never reaches the pair's `tx_count`. The table above is there to prevent this.
   Note the distinction: the producer side must not have an **entitlement** guard; an existence predicate can be added per PTX semantics, but it is uncertified, and on the warp-specialized shape the whole-kernel reading is slower with it
   (gotcha 10; [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) gotcha 5: the extra probe on the last beat
   is harmless to the data).
5. **Treating `try_wait` as a free query.** In PTX semantics it can suspend up to the system-default limit. The truly pure
   query is `mbarrier.test_wait` -- but **switching to test_wait is not this entry's fix**; that only trades sitting out the time limit each time for spinning idle each time;
   the fix is **not to issue this peek at all**.
6. **Feeding the result of a peek that should not exist to `consumer_wait`.** The "only slow, not dead" premise of this entry is that the always-false
   token is only produced, never consumed. If it is passed to `consumer_wait(state, token)`, it escalates from a throughput problem
   to a **deadlock**. The last-round predicate satisfies this naturally (there is no wait after the last round), but if the peek is moved to the loop head
   "for structural neatness", recheck this.
7. **Attributing by "who is spinning".** See mechanism layer 1: the follower spins, the leader starves.
   Adding stage depth / prefetch following the spinning role is the most common misdiagnosis of this symptom --
   depth-type levers price at about zero on it, because the bottleneck is not supply depth at all.
8. **Forgetting to pre-bind `peek_full` outside the region.** A name first bound inside a branch of a run-time `if` cannot leave the
   region, and the `NameError` is reported in the stack frame of `if_executor`. First
   `peek_full = cutlass.Boolean(False)`, then enter the guard, see
   [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md). With the last-round predicate added,
   `peek_full` is no longer assigned in the last round -- that is fine (it is no longer read either), but the pre-binding **must not**
   be deleted because of it.
9. **In the warp-specialized MMA loop, thinking "guarding the leader is enough".** The whole MMA warp sits inside `if is_leader`, so (1) is naturally satisfied, and it is easy to forget
   that (2) is still a separate checkbox. The last beat's peek sticks between the last MMA and the accumulator commit, delaying the whole epilogue by one suspension limit, while numerics, deadlock and
   tensor workload are all normal. Probe readings: the same pair GEMM differing only in this one guard line regressed 4.13 / 4.10 µs at two k-tile counts (4x apart) -- independent of the k-tile
   count, one suspension per call. On kernels with few k-tiles and a dozen-odd microseconds per call, this item is a large fraction of the total time.
10. **Do not guard the producer side's last-beat peek along with it.** On the same warp-specialized pair GEMM, a same-job A/B of the four combinations (medians after an L2 flush, all four combinations
    numerically correct):

    | producer last-beat `try_acquire` | MMA last-beat `try_wait` | median |
    |---|---|---|
    | bare | guarded | **21.504 µs** |
    | bare | bare | 25.856 µs |
    | guarded | guarded | 31.744 µs |
    | guarded | bare | 35.680 µs |

    The MMA-side guard saves 4.35 µs (the same order as the probe's single suspension); adding the guard on the producer side instead **adds about 10 µs**, cause undetermined (the empty phase of that peek
    does flip; without it the first acquire of `producer_tail` no longer carries a token and may have to sit out a wait by itself -- this is a guess, not isolated). The whole-kernel gate readings point the same way:
    both sides guarded 0.0254 ms, only the MMA side guarded 0.0145 ms (re-measured 0.0144 / 0.0145), before landing (both sides bare) 0.0190 ms.
    Only this one tree and one shape; the cost of the producer side's (2) on the split / fused shapes was not measured.
11. **Condition (2) has nothing to do with the pair: on a single CTA with `CtaGroup.ONE`, each call likewise sits out one full time limit.** Certification probe (single CTA, `CtaGroup.ONE`,
    a `PipelineTmaUmma` ring, one supply warp with a `producer_try_acquire` lookahead, one consumer warp idling at a set pace before each peek so that steady-state peeks
    always find the filling already arrived) readings: without the guard the last-beat peek takes 5184 ns (consistent over two repetitions), steady-state peeks median 32 ns, max 64 ns;
    the two fixes, the `kb + 1 < NUM_KB` guard / peeling the last beat, cut the loop time by 4992 / 5152 ns respectively (≈ one last-beat suspension), all five variants' data bit-exact;
    with the number of beats doubled the extra time goes 4992 → 4800 ns (ratio 0.96), a fixed amount per call. Whole kernel: on a `CtaGroup.ONE`, cluster-multicast-supplied
    warp-specialized GEMM tree, a same-job ablation removing the MMA warp's last-beat peek took two workloads 11.84 → 7.50 µs and 11.25 → 6.83 µs (saving about
    4.3–4.4 µs per call, the same order as the probe's single suspension, numerics still passing the judge); then landing only this one guard line on that `consumer_try_wait` gave gate 0.0138 → 0.00885 ms
    (geometric mean over four workloads; the confirmation job reads 0.0088).
    The producer side's tail probe is **asymmetric** to this: it probes the release needed for the `(NUM_KB + 1)`-th filling (the `consumer_release` of beat
    `NUM_KB + 1 − STAGES`), and that release will certainly happen, at most waiting for the consumer to catch up; in the probe above every supply-side probe (including the last beat) read 0–64 ns.
    "One extra probe on the producer side is harmless" does not imply "one extra probe on the consumer side is harmless too" (gotcha 10,
    [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) gotcha 5).
12. **When measuring this dead tail, the suspension is not charged to the `consumer_try_wait` line.** The predicate of `try_wait` has a variable latency, and timer reads unrelated to it issue first:
    in gotcha 11's probe, the end timestamp of the same bracket reads 5184 ns when it depends on the predicate and 0 ns when replaced by an independent timer read. On a profile this dead tail lands on the first instruction after the loop
    exit that reads this predicate, not on `SYNCS.PHASECHK` itself. Details see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 14.

## Sources

- Certification probe (warp-specialized MMA loop shape; the probes are not included in this release) -- **Status: ✅ GPU-certified**,
  `PROBE PASS` (passed at the first try in round 1 of this certification, all eight PROBE lines PASS; the `max_diff=0.0078` of the last line is the reading of the contract "relative difference of the
  regression between the two k-tile counts", not a numeric error). Two variants of the same `cta_group=2` pair GEMM (bare last-beat peek / `kb + 1 < k_tile_cnt`
  guard), differing only in that one guard line; contracts: both variants numerically equal to numpy (max_diff 0.0), the last-beat peek returns False on every leader,
  the bare variant slower by > 1 µs per call, the regression the same at both k-tile counts (relative difference < 0.5). Readings: few k-tiles bare 15.39 / guarded 11.26 µs,
  many k-tiles (4x) bare 19.46 / guarded 15.36 µs, regression 4.13 / 4.10 µs.
- Certification probe (pair shape) -- **Status: ✅ GPU-certified**, final verdict `PROBE PASS max_diff=0.0` (passed at the first try in round 1 of this certification), all seven contracts pass in the same round.
  Four variants of the same kernel (bare peek / entitlement guard only / entitlement + existence / existence calibration), seven contracts:
  three variants each numerically correct (showing this is a throughput bug, not a correctness bug), the last-round phase really never flips,
  the extra peeks have a measurable suspension cost, and the regression of that last-round one equals exactly **one** suspension.
- Certification probe (single CTA, `CtaGroup.ONE` shape, gotchas 11 / 12)
  -- single CTA, a `PipelineTmaUmma` ring, float32 slots, warp 0 supplies (with a `producer_try_acquire` lookahead), warp 1 consumes (`consumer_wait(cs, peek)`
  → read slot → `consumer_release`); five variants: unguarded / `kb + 1 < NUM_KB` guard / last beat peeled (same number of beats), unguarded / guarded (number of beats doubled).
  Contracts: every variant's data bit-exact, peek counts correct; in the unguarded variant the last-beat token is false and bounded re-probes of the same phase never flip; the last-beat peek's suspension ≥ 1 µs
  and far larger than steady-state peeks; with the guard the loop time drops by ≥ 0.5 × the last-beat suspension; with the number of beats doubled the extra time does not grow with the number of beats.
  **Status: ✅ GPU-certified**, `PROBE PASS max_diff=0.0` (round 2 of this certification), all eight contracts pass in the same round; readings in gotcha 11.
- Public interfaces (**the class names of the `pipeline` submodule have changed across DSL builds; `dir()` to confirm they exist before use**;
  if absent, build the hand-written equivalent; the mechanism is the same): `cutlass.pipeline.PipelineTmaUmma`
  (`producer_acquire` / `producer_get_barrier` / `producer_try_acquire` /
  `consumer_wait` / `consumer_try_wait` / `consumer_release` / `producer_tail`),
  `cutlass.pipeline.make_pipeline_state`, `cutlass.pipeline.pipeline_init_arrive` /
  `pipeline_init_wait`, `cutlass.cute.arch.map_dsmem_ptr`,
  `cutlass.cute.arch.block_idx_in_cluster`, `cutlass.cute.nvgpu.tcgen05.CtaGroup`,
  `cutlass.cute.nvgpu.cpasync.tma_partition`.
- PTX ISA, the blocking difference between `mbarrier.try_wait` / `mbarrier.test_wait` and
  the default-limit clause of `suspendTimeHint`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
