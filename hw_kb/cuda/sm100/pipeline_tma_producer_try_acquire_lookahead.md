# pipeline_tma_producer_try_acquire_lookahead (the TMA producer probes the next stage's empty one beat ahead with test_wait)

> **Certification source: B200 GPU-certified in CUDA mode (all 8 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: token semantics (fresh ring, arm / no arm, throttling on either side, wrong order), and the return time of a one-shot test_wait and a one-shot try_wait when the
> phase is incomplete. **Uncertified**: the whole-kernel gain (the lookahead only moves the "query" off the critical path; it does not make empty release earlier),
> cluster / multicast versions.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the ring itself see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md),
> for the wait encoding see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md).

## Purpose

After the TMA producer has issued beat k and advanced (stage, phase), it immediately does one non-blocking query on **the empty that the next beat will wait on**,
keeping the predicate (token); at the start of beat k+1, if the token is true it skips the blocking wait, and only if false does it take the blocking wait. What a true token saves is one query on the critical path (one test_wait measured at about 77 cycles). Whatever the token,
`mbarrier.arrive.expect_tx` + the copy are issued as usual.

**A one-shot query must use `mbarrier.test_wait`**: measured with the phase incomplete, a one-shot test_wait returns in a median **76 cycles**,
a one-shot hint-free try_wait in a median **7814 cycles** (it suspends until the phase completes or an implementation-defined time limit); a one-shot try_wait with a suspendTimeHint
returns after about 7924 cycles whether the hint says 100 ns or 100 µs — **the hint cannot shorten this suspension**.

## When needed (trigger)

- The producer pays a "query round trip" on the blocking wait every beat, while empty has usually long been released (on the timeline the producer's empty wait is short but
  present every beat).
- **Anti-trigger**: when the consumer is slow and empty really is released late, the lookahead gives a false token and blocks anyway; no gain.
- **The MMA issue side must not probe the next beat's full before this beat's gemm**: a one-shot try_wait suspends about 7.8k cycles when the phase is incomplete (T0),
  delaying this beat's UMMA issue by as much; if you probe, do it after this beat's commit, and with test_wait.

## Minimal runnable example

```cpp
// Taken from the certification probe
__device__ __forceinline__ uint32_t mbar_test_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok;
}

// producer (single thread)
int stage = 0; uint32_t phase = 0;
uint32_t token = mbar_test_wait(&empty[0], 1u);       // the first token comes from a real query
for (int k = 0; k < K; ++k) {
  if (!token) mbar_wait(&empty[stage], phase ^ 1u);   // block only when the token is false
  mbar_arrive_expect_tx(&full[stage], BYTES);         // always issued: the token cannot replace the arrive
  bulk_g2s(&slot[stage][0], src + k * ELEMS, BYTES, &full[stage]);
  if (++stage == S) { stage = 0; phase ^= 1u; }       // advance first
  token = mbar_test_wait(&empty[stage], phase ^ 1u);  // then probe: the same (stage, phase) as the next beat's blocking wait
}
```

## Gotchas

1. **Advance first, then probe.** What is probed must be the (stage, phase) the next beat waits on; probing before advancing hits the stage just acquired — it is of course already
   empty, the token is always true, and the next beat skips the wait and overwrites a slot the consumer has not read yet. Measured (consumer throttled): the first 5 tokens true,
   data overwritten, full arrived on once too often, after which the bounded waits on both sides time out (T7).
2. **Even when the token is true, `arrive.expect_tx` is required.** With only `mbarrier.expect_tx` (no arrive) + the copy, full never flips
   (T3): the pending arrival stays at 1, and landing bytes cannot make up for it.
3. **A fresh ring's token sequence is [1,1,1,1,0]** (S=4): the producer starts at phase 0 waiting on parity 1, so the first S are naturally true.
4. **Use test_wait for a one-shot query, not try_wait** (T0: 76 vs 7814 cycles). Using try_wait as the token leaves the numerics and the token
   sequence unchanged (with the throttled consumer the token is likewise `[1]*4 + [0]*12` and the data exact), only every false token wastes about 7.8k cycles suspended —
   a numeric probe cannot see it; only cycle measurement can.
5. **One extra probe after the last beat is harmless** (read-only).
6. **The lookahead does not make empty release earlier**: the release time is decided by the consumer; the lookahead only saves one query on the producer's critical path.
   With a throttled consumer the token sequence is `[1]*4 + [0]*12` (T4), with a throttled producer all 1 (T5).

## Sources

- PTX ISA: *mbarrier.test_wait / mbarrier.try_wait* (try_wait may suspend, test_wait never suspends),
  *mbarrier.arrive.expect_tx / mbarrier.expect_tx*, *cp.async.bulk*, https://docs.nvidia.com/cuda/parallel-thread-execution/
- GPU-certified by the probe:
  `PROBE PASS max_diff=n/a pipeline_tma_producer_try_acquire_lookahead/T0-oneshot-return-latency-incomplete-phase test_wait_cyc=76 try_wait_cyc=7814`,
  `PROBE PASS max_diff=0.0 pipeline_tma_producer_try_acquire_lookahead/T1-fresh-tokens tokens=[1, 1, 1, 1, 0]`,
  `PROBE PASS max_diff=0.0 pipeline_tma_producer_try_acquire_lookahead/T2-arm-flips-full`,
  `PROBE PASS max_diff=n/a pipeline_tma_producer_try_acquire_lookahead/T3-token-without-arrive-never-flips token=1 polls=-1`,
  `PROBE PASS max_diff=0.0 pipeline_tma_producer_try_acquire_lookahead/T4-paced-consumer`,
  `PROBE PASS max_diff=0.0 pipeline_tma_producer_try_acquire_lookahead/T5-paced-producer`,
  `PROBE PASS max_diff=0.0 pipeline_tma_producer_try_acquire_lookahead/T6-free-run-exact`,
  `PROBE PASS max_diff=n/a pipeline_tma_producer_try_acquire_lookahead/T7-wrong-order-breaks-protocol data_mismatch=True timeouts=[1, 1]`.
