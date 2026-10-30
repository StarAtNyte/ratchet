# cta_pair_consumer_peek (two guards on a non-blocking peek: who is entitled to ask × whether one more wait follows)

> **Certification source: CUDA-mode GPU probes (a peek probe and a TMA-pair probe; not included in this release),
> `PROBE PASS` (`mbar-trywait-blocks`, `mbar-trailing-peek-fixed-cost` max_diff=0, `tma-pair2d` max_diff=0.0).**
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the delivery side (how the bytes of the pair's two halves are booked on the leader's barrier) see
> [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md).

## Purpose

A common pipeline spelling: after each wait, first peek at the next slot (`mbarrier.try_wait.parity` gets a token); if the token is true, the next wait
is skipped. This peek should be issued only when **both conditions hold**:

| | Condition | What it asks when missed |
|---|---|---|
| ① Entitlement | When a `cta_group::2` pair shares the leader's full barrier, only the leader's is armed; the follower's own full barrier never completes | the peek on the follower asks a never-completing phase every k-tile |
| ② Existence | the peek prefetches a token for the **next** wait; after the last wait no next slot is filled | the last-round peek asks a never-completing phase once per kernel call |

Asking a never-completing phase is not a free instruction: `mbarrier.try_wait` is **potentially blocking** — when the phase is incomplete the thread is suspended until a
system time limit before it returns false; what really returns immediately is `mbarrier.test_wait`. This entry's probe measured that cost.

## When needed (trigger)

- The code has a `try_wait`-style peek (whether hand-written or inside a wrapper). Go through each call site with two checks: is it inside `if (is_leader)`
  (when the pair shares a barrier)? Is there really one more wait after it?
- Symptom: numerics all right, no deadlock, a fixed idle wait after the main loop and before the epilogue (once per call, independent of the number of k-tiles); or
  the follower's issue warp idles everywhere in the main loop (proportional to the number of k-tiles). **Fastest way to tell them apart: double the number of k-tiles and measure again** —
  the cost of a missed existence check does not move, that of a missed entitlement check doubles.

## Minimal runnable example

```cpp
// warp-specialized MMA issue warp (for a pair the whole consumer section is inside if (is_leader) — condition ①)
uint32_t tok = mbar_try_wait(full[0], 0);                 // prologue peek: a wait follows
for (int k = 0; k < k_tiles; ++k) {
  const int s = k % STAGES; const uint32_t ph = (k / STAGES) & 1;
  if (!tok) while (!mbar_try_wait(full[s], ph)) {}
  /* ... issue MMA on stage s, release empty[s] ... */
  if (k + 1 < k_tiles)                                    // condition ②: no peek on the last beat, which has no next wait
    tok = mbar_try_wait(full[(k + 1) % STAGES], ((k + 1) / STAGES) & 1);
}
/* commit accumulator -> epilogue */

__device__ __forceinline__ uint32_t mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
```

## Gotchas

1. **Treating `try_wait` as a free query.** Probe readings (single thread, phase never arrived): `try_wait` on a never-arrived phase takes about **7800 SM cycles (about 5.2 µs, globaltimer)** to return false; `test_wait` and "`try_wait` on an already completed phase" are both **43 cycles**. `try_wait` with `suspendTimeHint` (two settings, 1000 ns and 100000 ns) is likewise about 5.2 µs — the hint did not shorten an empty poll. But **switching to `test_wait` is not the fix**
   — the fix is not to issue this peek at all.
2. **Guarding "who asks" but forgetting "is there anything left to ask".** When the loop body is symmetric (wait → consume → peek), the last beat copies the peek once more, and it lands exactly between the last
   MMA and the accumulator commit. Readings from the probe's 2-barrier ring (producer one phase ahead): the difference between a bare last-beat peek and the guarded version is **7782 cycles (about 5.2 µs)** both at K = 4 and K = 16, relative difference between the two 0.000 — exactly one suspension per call, independent of the trip count; that last-beat peek returns false.
3. **The follower's local full barrier never completes** (when the pair shares the leader's barrier). In the probe's `pair2d` cell the follower does one `try_wait` on its own
   full barrier, which returns false (the phase never completes), while the barrier the leader waits on flips normally — peeking at or waiting on it on the follower is asking stagnant water.
4. **No entitlement guard on the producer side**: the empty barrier is arrived on in both CTAs and `try_acquire` must run in both; give it a leader
   guard and the follower can never issue its half of the loads again (deadlock).
5. **An always-false token may only be produced, never used for real by the next wait**: the last-round predicate satisfies this naturally; recheck it when moving the peek to
   the loop head "to make the structure tidy".

## Sources

- PTX ISA, `mbarrier.try_wait` (potentially blocking, `suspendTimeHint`, system-dependent default time limit) and `mbarrier.test_wait` (non-blocking),
  the definition of a CTA pair: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe — `PROBE PASS max_diff=n/a` (`mbar-trywait-blocks`) and `PROBE PASS max_diff=0` (`mbar-trailing-peek-fixed-cost`; max_diff is the relative difference of the regression at the two K); the `pair2d` cell of the TMA-pair probe (the reading of `try_wait` on the follower's local full barrier), `PROBE PASS max_diff=0.0`.
