# umma_consumer_full_peek_before_commit (the MMA issuer's one-beat look-ahead at the next beat's `full` on a hand-written mbarrier ring: where in the beat to place the query and which instruction to use; the token is handed to the blocking wait at the top of the next beat)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> Prerequisite / orthogonal entries:
> - [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md): **how a wait is encoded**; its section (g) covers the difference between `try_wait` and `test_wait`
>   in a one-shot query, and the inline-PTX entry for `test_wait`. This entry covers only **where in the beat the MMA issuer places the query**;
> - [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md): the dual of the same trick on the **supply-side `empty`**
>   (the `cutlass.pipeline` wrapper surface; only the supply side is certified there, and the MMA side is sketched as a plain `consumer_wait`);
> - [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md): the two guards of a peek (on a pair only the leader may ask; do not ask on the last beat);
> - [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md): hoisting stage bookkeeping / descriptor addresses above the wait.
>
> **Status: ✅ Verified** — each of the three placements has one GPU certification (the probes are not included in this release): before the commit (`PROBE PASS max_diff=0.0`, round 3, all six rows PASS),
> mid-beat (`PROBE PASS max_diff=0.0`, round 1, all three contracts PASS), after the commit (`PROBE PASS max_diff=0.0`,
> round 2, all ten rows PASS); the phase aliasing of gotcha 9 additionally has `PROBE PASS max_diff=0.0` (round 2). What is certified is values, token semantics and SASS placement,
> **not gain**: the `test_wait` form landed on two UMMA trees with whole-kernel readings inside the noise; same-job readings of the `try_wait` form were all slower (see "When needed").

## Purpose

On a hand-written TMA → UMMA operand ring (two sets of bare mbarriers `full[s]` / `empty[s]`, runtime loop `s = t % S`, parity `(t // S) & 1`),
the MMA issuer's standard order per beat is:

```
mbarrier_wait(full[s], ph)  →  issue this beat's MPB UMMAs  →  tcgen05.commit(empty[s])
```

A one-beat look-ahead asks "has the next beat's data arrived" early and stores the answer as a loop-carried token; at the top of the next beat, a true token skips the blocking wait,
and only a false token falls back to the blocking wait. There are three places to ask:

| Placement | Query instruction | What this thread still issues after the query, before the top of the next beat | Certification |
|---|---|---|---|
| **before the commit** (after this beat's UMMAs, before `tcgen05.commit`) | `test_wait` | commit, advance, back edge | GPU probe |
| **mid-beat** (between two UMMAs of this beat) | `test_wait` | the rest of this beat's UMMAs, commit, advance, back edge | GPU probe |
| **after the commit** (after advancing stage / phase, before the back edge) | `try_wait` (certified form); for whole kernels the default is `test_wait`, see below | advance arithmetic, back edge | GPU probe |

The before-commit and mid-beat placements still have this beat's commit / UMMAs behind the query, so **only `test_wait` may be used**: when the phase has not completed, `try_wait` waits up to
its time limit, and the wait lands on the first instruction that uses the predicate ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (g) and gotcha 14); the commit behind it is delayed accordingly,
`empty` is released late, and the supply side follows late (readings in gotcha 2). In the after-commit position only the advance and the back edge remain, and an in-beat `try_wait` wait amounts to doing the next beat's top
blocking wait early; but if a tile's last beat looks ahead at the next tile's first beat, the accumulator commit is still behind the query, and `try_wait` delays it.
In same-job whole-kernel readings the `try_wait` form was never faster than `test_wait` (see "When needed"); **use `test_wait` by default**.

## When needed (trigger)

- Timeline / segment ledger: the MMA role's per-beat "wait full" segment stays at the same order of magnitude as the steady-state wait even when the data is actually already in the ring (e.g. in the tail, when all supply has been issued) —
  the wait segment is mostly the round trip of the wait primitive itself, not waiting for supply.
- Structure: the top of the MMA loop is a `cute.arch.mbarrier_wait(full + s, ph)` **without a token**; in SASS, `SYNCS.PHASECHK…TRYWAIT`
  appears only once, at the loop top after the back edge.

**Anti-trigger**: when the wait segment really is waiting for supply (supply side exposed, `full` arriving late), a look-ahead can only save the query round trip; it does not make the data arrive earlier — look at the supply side / ring depth first.

**Whole-kernel readings (no gain promised)**:
- Before the commit + `test_wait`: on a pair-UMMA tree, the same-job baseline ratio of the best code before landing (no look-ahead) was 1.0107 (confirmation medians 1.0065 / 1.0049),
  0.9976 after landing, and the next few changes built on top of it read 1.0033–1.0137 (one more at 0.9406); the whole series was eventually rolled back for lack of gain. The only same-job control was the variant "issue the predicate before the commit,
  move the token materialization after the commit": −0.32%, the same order as a same-job re-measure at −0.25%.
- After the commit + `try_wait`: same-job A/B on another tree (against the original spelling without look-ahead, geometric mean over five workloads, > 1 is faster)
  0.9998 (together with descriptor-advance reordering) / 0.9982 (look-ahead only, slightly slower, inside the noise).
- Other same-job readings of same-family `try_wait` look-aheads (time ratio, > 1 is slower) were all slower:
  - On a UMMA tree, `try_wait` on the next beat after each beat's release + advance, together with a `try_wait` on the next accumulator slot before the accumulator commit:
    all 8 shapes slower, geometric mean 1.0072; peeking only at the tile seam 1.0132, the same position with `test_wait` 0.9982 / 0.9984.
  - On another tree, peeking after the commit with no guard on the tile's last beat and the query placed before the accumulator commit: `try` on both MMA side and supply side was 8.5% / 8.7% slower,
    `test` on both sides was flat (−0.1% / +0.0%), and `try` on the supply side only was also flat (−0.1% / +0.0%).
  - Both groups placed the `try_wait` query before the accumulator commit or left the last beat unguarded, so they are not the pure in-beat (c); the pure in-beat (c) `try_wait` has only the 0.9982 reading above.
- Mid-beat + `test_wait`: same-job baseline ratio on the same tree before and after landing 1.0162 (median of two confirmations) → 1.0159 (cross-job).
- On the certification toy kernel (single CTA, balanced pacing) the per-beat cycles of the three placements are the same order as the no-peek baseline: baseline 580.9, mid-beat `test_wait` 589.5, mid-beat `try_wait` 595.2,
  after-commit `try_wait` 588.5 cycles / beat (single INFO).

After the change you must price it yourself with a same-job A/B.

## Minimal runnable example

`peek_test(bar, ph)` is the `test_wait` entry of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (g) (returns `Int32` 1/0; goes through inline PTX when the build does not export
`cute.arch.mbarrier_test_wait`). `full0` / `empty0`: base addresses of the two sets of bare mbarriers (`S` each, init count 1);
the supply side does the usual `mbarrier_wait(empty[s], ph ^ 1)` → `mbarrier_arrive_and_expect_tx(full[s], bytes)` → TMA.

### (a) Peek before the commit (single-CTA form; pair form see gotcha 6)

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05

Int32 = cutlass.Int32

tok = peek_test(full0, 0)                         # before the loop: peek once for t = 0 (a real call provides the initial value)
for t in cutlass.range(T, unroll=1):
    s = t % S
    ph = (t // S) & 1
    # (quantities unrelated to the wait, such as addresses / descriptors, go here — before the wait, see umma_issue_chain_precomputed_desc)
    if tok == 0:
        cute.arch.mbarrier_wait(full0 + s, ph)    # token false: fall back to the blocking wait
    # ---- issue all UMMAs of this beat ----
    ... cute.gemm(tiled_mma, acc, tCrA[(None, None, kb, s)], tCrB[(None, None, kb, s)], acc) ...
    # ---- look ahead at the next beat: before the commit, test_wait only ----
    tn = t + 1
    nt = Int32(0)
    if tn < T:                                    # no peek on the last beat (there is no next wait)
        nt = peek_test(full0 + tn % S, (tn // S) & 1)
    tok = nt
    with cute.arch.elect_one():
        tcgen05.commit(empty0 + s)                # release this beat (pair form carries mask / CtaGroup.TWO)
```

### (b) Mid-beat peek (tuck the query into the gaps between this beat's UMMA issues)

```python
PEEK_AT = MPB // 2 - 1                            # ask after UMMA number PEEK_AT (MPB = UMMAs per beat)

tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
tok = Int32(0)
for k in cutlass.range(KB, unroll=1):
    s = k % S
    ph = (k // S) & 1
    if tok == 0:                                  # block only if the mid-beat query missed
        cute.arch.mbarrier_wait(full0 + s, ph)
    for kb in cutlass.range_constexpr(MPB):
        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, s)], tCrB[(None, None, kb, s)], tCtAcc)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(kb == PEEK_AT):
            tok = Int32(0)
            if k + 1 < KB:                        # existence guard
                tok = peek_test(full0 + (k + 1) % S, ((k + 1) // S) & 1)
    with cute.arch.elect_one():
        tcgen05.commit(empty0 + s)
```

### (c) Peek after the commit (the certified form is `try_wait`; for whole kernels the default replaces both queries with `peek_test`)

```python
tok = Int32(0)
if cute.arch.mbarrier_try_wait(full0, 0):         # query for beat 0
    tok = Int32(1)
stage = Int32(0)
phase = Int32(0)
for k in cutlass.range(KB, unroll=1):
    if tok == 0:
        cute.arch.mbarrier_wait(full0 + stage, phase)
    for kb in cutlass.range_constexpr(MPB):
        cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, stage)], tCrB[(None, None, kb, stage)], tCtAcc)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
    with cute.arch.elect_one():
        tcgen05.commit(empty0 + stage)            # right after the last UMMA
    stage = stage + 1                             # advance first
    if stage == S:
        stage = Int32(0)
        phase = phase ^ 1
    tok = Int32(0)
    if k + 1 < KB:                                # then peek: existence guard
        if cute.arch.mbarrier_try_wait(full0 + stage, phase):
            tok = Int32(1)
```

Replacing both `if cute.arch.mbarrier_try_wait(...): tok = Int32(1)` with `tok = peek_test(...)` gives the `test_wait` version (SASS acceptance: see gotcha 8).
When a tile's last beat looks ahead at the next tile's first beat and an accumulator commit still follows, only `test_wait` may be used (same reason as the before-commit placement).

Invariants (the same for all three placements):

1. **What is peeked is the next beat's `(slot, parity)`**: `slot = (t+1) % S`, `parity = ((t+1) // S) & 1`; wrap-around follows automatically from these two formulas.
2. **A false token always falls back to the blocking wait**: the look-ahead affects only timing, not semantics; the data are bit-identical.
3. **The token is a loop-carried `Int32`**: initialized before the loop by one real call (or `Int32(0)`), reassigned every beat in the loop body.
4. **The look-ahead only reads the barrier**: no arrive, no phase change; one commit per beat, at the same position.

## Gotchas

1. **Peeking the wrong slot = silently reading data that has not landed, or the whole CTA hangs.** Peek with this beat's `(s, ph)` (just waited on, so of course ready) and the token is true at least every other beat
   (the certification probe read alternating 0/1); on the beats where it is true the wait is skipped and UMMAs are issued directly, reading a stage that has not landed yet. The consumer is no longer throttled correctly by `full`, `commit(empty[s])` runs ahead of supply, the same `empty[s]`
   is advanced several phases in a row, and the phase the producer waits on by parity no longer matches the actual phase → **hang**. GPU reading (the round before the certifying round of the before-commit probe): a ring that only peeks the wrong slot,
   everything else unchanged, does not return within 60 s; the negative-control cell (one correct blocking wait added after the read and before the commit, only to keep the ring from hanging) read data that had not landed, `data_maxdiff` 10177.5.
2. **Use `test_wait`, not `try_wait`, for the before-commit / mid-beat query.** Before-commit probe: with supply throttled so the next beat never arrived in time, the before-commit
   `try_wait` query cost 7819–7823 cycles per beat and `test_wait` 101 cycles (SM locked at 1500 MHz); this wait delayed this beat's commit. Mid-beat toy-kernel readings
   (balanced pacing): mid-beat `try_wait` 595.2 cycles per beat with the token true 2220/2220 (it waited in place), mid-beat `test_wait` 589.5,
   true 243/2220.
3. **Do not peek on the last beat.** Fill number `T` does not exist; a `test_wait` peek is just one wasted read, while a `try_wait` peek is billed up to the time limit with no following async work to hide it
   ([`cta_pair_consumer_peek`](cta_pair_consumer_peek.md) condition ②). If the main loop and the tail are split into two loops, **the "next beat" of the main loop's last beat is the tail's
   first beat**; that beat exists, peek as usual and carry the token into the tail loop; do not peek on the tail's last beat.
4. **A true token does not let you skip the commit.** The token only replaces the `full` wait; releasing `empty` is still done by this beat's `tcgen05.commit`.
5. **Do not spin on `test_wait` in place when the token is false.** Fall back to `cute.arch.mbarrier_wait`; the wait encoding is a different axis ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md));
   do not mix it with placement in one change.
6. **CTA pair (`cta_group=2`) form**: `full` is armed only on the leader, so both look-ahead and wait may only be done in the leader's MMA warp
   ([`cta_pair_consumer_peek`](cta_pair_consumer_peek.md) condition ①); `commit` carries the multicast mask and `CtaGroup.TWO` as before. All three certification probes are single-CTA.
7. **The token hit rate is set by the placement, and a high hit rate is not the same as fast.** The earlier you ask, the more likely you miss: mid-beat hits 2197/2220 when consumption is slow,
   0/2220 when production is slow, 243/2220 when balanced; after the commit all hits when consumption is slow, all misses when production is slow. Results are bit-correct in both cases.
   Before the commit: all hits except the first beat when consumption is slow / free-running, all misses when production is slow.
8. **Whether ptxas moves the query must be checked in SASS.** Acceptance reading: `test_wait` lowers to `SYNCS.PHASECHK.TRANS64` (without `.TRYWAIT`), `try_wait` to
   `SYNCS.PHASECHK.TRANS64.TRYWAIT`; the mid-beat form requires the `SYNCS.PHASECHK` in the MMA loop to have `UTC*MMA` on both sides, with no
   `UTCBAR` between it and the nearest MMA on either side (the placement contract of the mid-beat probe); the after-commit form requires one `SYNCS.PHASECHK` in the loop body after `UTCBAR` and before the back edge
   (census of the after-commit probe: 1–2 instructions after `UTCBAR` when issued via `cute.gemm`, 5–9 apart with hand-issued MMAs). Descriptor addition is pure arithmetic, and ptxas may
   schedule it ahead of the query.
9. **When peeking several beats at once, beats at distance ≥ S are phase-aliased into "ready".** A parity query only sees odd/even: block `t0 + j` (`j ≥ S`) and block `t0 + j − S`
   share a slot with opposite parity; while the latter has not landed the barrier sits one phase lower, and `test_wait(target parity)` returns true. Negative control of the gotcha 9 probe:
   with nothing landed, one snapshot of 2S beats read 0 in the low S bits and 1 in all high S bits (mask 0xF0). On a freshly initialized barrier `test_wait(parity = 1)` is true and
   `parity = 0` is false (the `peek_cost` cell of the before-commit probe, same answer from `try_wait`) for the same reason.
   The multi-beat snapshot spelling itself has no whole-kernel gain: in the steady-state ring of one tree, replacing the one-beat look-ahead with a W-beat snapshot (re-taken when the next beat is not ready) was 7.03% slower in the same job
   (geometric mean over four workloads).
10. **Write the token as `Int32`.** `tok = Int32(0); if try_wait(...): tok = Int32(1)` / `peek_test(...)`, and at the loop top `if tok == 0:`; writing `if not tok:` on a DSL `Boolean`
    evaluates `bool` of a dynamic value at trace time.

## Sources

- PTX ISA: `mbarrier.test_wait` (pure query) and `mbarrier.try_wait` (may suspend up to an implementation-defined time limit), the phase semantics of `.parity`, `tcgen05.mma`,
  `tcgen05.commit`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
  https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit
- Public infra API: `cute.arch.mbarrier_try_wait`, `cute.arch.mbarrier_wait`, `cute.arch.mbarrier_test_wait` (absent in some builds),
  `cute.arch.mbarrier_init`, `cute.arch.mbarrier_arrive_and_expect_tx`, `cute.arch.elect_one`, `cutlass.cute.nvgpu.tcgen05.commit`,
  `cute.gemm`, `tiled_mma.set(tcgen05.Field.ACCUMULATE, …)`, `cutlass.range(…, unroll=1)`, `cutlass.range_constexpr`, `cutlass.const_expr`,
  `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`. DSL documentation: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/
- GPU certification probe (before the commit) — single CTA, bare mbarrier ring, float32 slots; the consumer reads the slot with a plain load instead of a UMMA,
  release via `tcgen05.commit`; six cells: `peek_cost`, `test_paced_producer`, `test_paced_consumer`, `test_free`, `try_paced_producer`,
  `wrong_slot_peek` (negative control). **✅ `PROBE PASS max_diff=0.0` (round 3)**. Readings in gotchas 1 / 2 / 7.
- GPU certification probe (mid-beat) — single CTA, `cta_group::1`
  UMMA, four variants (baseline / mid-beat `test_wait` / mid-beat `try_wait` / after-commit `try_wait`) × three pacings; contracts: the results of six subprocesses bit-exact against a float64
  reference, SASS placement of the mid-beat form, and mid-beat `test_wait` exercising both the hit and the miss path under slow consumption / slow production.
  **✅ `PROBE PASS max_diff=0.0` (round 1)**; per-beat cycles and hit counts are same-job INFO.
- GPU certification probe (after the commit) — single CTA, `cta_group::1`
  UMMA, two geometries × baseline / look-ahead / look-ahead + hand-issued MMA, nine variants bit-equal to a numpy reference, plus a SASS census and token readings in the slow-consumption / slow-production cells.
  **✅ `PROBE PASS max_diff=0.0` (round 2)**.
- GPU certification probe (the aliasing negative control of gotcha 9) — **✅ `PROBE PASS max_diff=0.0` (round 2)**; this entry cites only its
  `window_alias_neg` cell (mask 0xF0).
