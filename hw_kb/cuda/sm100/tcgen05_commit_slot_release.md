# tcgen05_commit_slot_release (when the slot release `tcgen05.commit` → empty[s] actually lands; whether moving instructions around the commit is worth it)

> **Certification source: B200 GPU-certified in CUDA mode (SM clock locked at 1500 MHz, all 6 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: single CTA, `cta_group::1`, `kind::f16` M128 N128 K16 MMAs; how the commit arrival time relates to the MMA count, a delay inserted before the commit, and an appended commit;
> numerics and producer-side timestamps of a TMA → UMMA ring. **Uncertified**: `cta_group::2` / multicast commits; whole-kernel gain (reordering around the waits in a real kernel was whole-kernel flat, see gotcha 2).
>
> One file per hardware interface. Operator-agnostic: mechanism only. For a commit on the wrong stage see [`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md);
> for producer lookahead see [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md);
> for ring depth see [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md); for the ring's phase rules see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md).

## Purpose

In a TMA → smem ring → `tcgen05.mma` pipeline, the MMA issue thread sends
`tcgen05.commit.cta_group::1.mbarrier::arrive::one` to empty[s] after the last MMA of each beat, returning the slot to the producer. This entry answers: **when** does this arrival land on the barrier;
and can moving instructions around the commit (moving the stage / parity advance and the next beat's address computation after the commit, having the producer compute coordinates before its wait) make the slot release earlier.

## When needed (trigger)

- The stall ledger shows the producer mainly stalled on the **empty wait**, and you want to know whether to change the MMA side or the producer side.
- You plan to "move the commit earlier", "move bookkeeping after the commit", or "issue an extra commit so the slot is released a bit earlier".
- On the timeline the MMA warp finishes issuing a beat very early, yet the producer gets empty very late.

## Mechanism (measured by probe, B200 @ 1500 MHz, M128 N128 K16 f16)

1. **The commit's arrival lands when the MMAs have finished executing, not when the commit is issued.** It arrives once only after all of the committing thread's earlier tcgen05 async operations complete.
   The flip time (counted from the first MMA issue) grows linearly with the number of MMAs before the commit, and the slope is the execution time of one MMA, about 64 cycles each:

   | MMAs before the commit, n | 0 | 1 | 2 | 4 | 8 | 16 | 32 |
   |---|---|---|---|---|---|---|---|
   | flip time (cycles) | 210 | 234 | 306 | 414 | 695 | 1206 | 2226 |
   | commit issue time (cycles) | 168 | 207 | 245 | 321 | 483 | 991 | 2015 |
   | issue → flip (cycles) | 39 | 27 | 61 | 93 | 212 | 213 | 211 |

2. **MMA issue has backpressure.** Once enough MMAs are in flight (n ≥ 8 in the table above), the issuing thread itself stalls on issue, the commit's issue time moves back by about 64 cycles per MMA,
   and "issue → flip" saturates at about 210 cycles. Corollary: in a deep pipeline the MMA warp's bookkeeping after the commit is already hidden in this backpressure.
3. **flip ≈ max(all MMAs executed, commit issue + c0), c0 ≈ 39 cycles.** Inserting a busy wait before the commit in a beat with n = 4:

   | Inserted (cycles) | 0 | 64 | 256 | 1024 | 4096 | 16384 |
   |---|---|---|---|---|---|---|
   | measured flip | 414 | 462.5 | 670 | 1437.5 | 4484 | 16778 |
   | formula prediction | 414 | 474 | 669 | 1449 | 4491 | 16778 |

   Relative prediction error ≤ 2.43%. Work before the commit is free only while "commit issue + c0" is still earlier than the end of MMA execution; at n = 4 this slack is only about 414 − 321 − 39 ≈ 54 cycles,
   and inserting 64 cycles already delays the flip by about 48 cycles.
4. **Release follows issue order.** Commit to A first, then issue 1 MMA writing another TMEM block and commit to B: B lands 56–82 cycles after A (about one MMA's execution), even though the two have no data dependency.
   Issuing nothing after commit A and immediately committing to B: B lands 14–23 cycles after A, never earlier.
5. Corollary: **the producer's empty wait ≈ the execution queued in the MMA pipeline ahead of the slot it wants to reuse.** A full ring with the producer stuck on empty means MMA execution (not issue) is gating;
   moving instructions around the commit saves at most the issue cycles of a few instructions; what is needed is shorter MMA execution per beat, or keeping the producer from adding anything to the MMA critical path.

Ring measurement (the probe's TMA → UMMA ring, NU MMAs + one commit per beat, numerics bitwise right): producer empty wait returns → all of this beat's copies issued is a constant 160 cycles;
commit issued → that slot reused by the producer 355–786 cycles (355–431 for the first few beats, 608–786 afterwards).

## Minimal runnable example (instrumentation for measuring the commit arrival time)

```cpp
// Single CTA, 128 threads. One elected thread of warp 1 issues n MMAs (optionally busy-waits D cycles) then commits to barA;
// one thread of warp 0 spins on barA with test_wait and stamps %clock64 the moment it sees the phase flip (same-SM counters, directly subtractable).
__device__ __forceinline__ uint64_t clk() { uint64_t c; asm volatile("mov.u64 %0, %%clock64;" : "=l"(c) :: "memory"); return c; }
__device__ __forceinline__ uint32_t mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok;
}
__device__ __forceinline__ void tc_commit(uint32_t bar) {   // the same PTX as CUTLASS cutlass::arch::umma_arrive
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
}

if (warp == 0 && lane == 0) {                       // observer: issues no tcgen05 instruction
  for (int r = 0; r < reps; ++r) {
    while (!mbar_test(barA, r & 1)) { }
    flip_at[r] = clk();
  }
} else if (warp == 1) {
  if (elect_one()) {                                // issuer: MMA and commit must be the same thread ((c) spelling; thread selection in umma_issue_form)
    for (int r = 0; r < reps; ++r) {
      t0[r] = clk();
      for (int i = 0; i < n; ++i) umma(tmem, adesc, bdesc, i != 0);
      if (D > 0) { const uint64_t s0 = clk(); while (clk() - s0 < D) { } }
      tc_commit(barA);
      t1[r] = clk();                                // commit issue time
      while (!mbar_test(barA, r & 1)) { }           // wait for this round to land before starting the next
    }
  }
  __syncwarp();
}
// flip - t0 = arrival time; flip - t1 = issue -> arrival
```

## Gotchas

1. **"Commit issued" ≠ "slot released".** Using the commit instruction's issue timestamp to account for the producer's wait misreads MMA execution time as synchronization overhead; in a deep pipeline the issue timestamp itself is already pushed back by backpressure,
   and "issue → flip" reflects only the last few MMAs. To measure the release, spin on `mbarrier.test_wait` in another warp and stamp `%clock64`.
2. **Moving instructions around the commit usually does not change the release time** (mechanisms 2, 3); accept by SASS and timeline readings first; do not presume a gain. Whole kernel in a real kernel: computing the producer's coordinates / slot addresses and the MMA side's
   descriptors / barrier addresses before their respective waits (reordering only, no semantic change) gave gate 0.0322 → 0.0322 ms, flat.
3. **A commit cannot release one particular slot early.** It tracks **all** of the issuing thread's earlier tcgen05 async operations, and release follows issue order (mechanism 4); to release slot s earlier, the only way is fewer or faster MMAs queued ahead of it.
4. **The commit must be issued by the same thread as the MMAs.** A commit issued by another thread does not track these MMAs and arrives too early, and the producer overwrites a slot the MMA is still reading (per the PTX ISA; not tested by the probe). For how to choose the thread see
   [`umma_issue_form`](umma_issue_form.md).
5. **No `fence.proxy.async` is needed before the commit** (MMA reads of smem are in the async proxy anyway); it is needed only for faces that threads wrote with generic instructions and hand to the MMA (per the PTX ISA; not tested by the probe).
6. For draining before exit see [`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md) gotcha 7.

## Sources

- PTX ISA: `tcgen05.commit` (`.mbarrier::arrive::one`, tracks all tcgen05 async operations previously issued by the executing thread), `tcgen05.mma`, `mbarrier.test_wait`, `%clock64`,
  `cp.async.bulk.tensor` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Public CUTLASS headers: `cutlass::arch::umma_arrive` (`include/cutlass/arch/barrier.h`).
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 tcgen05_commit_slot_release/R1-ring-exact err=0`, `PROBE PASS max_diff=n/a .../L1-flip-tracks-mma cyc_per_mma=63.8`,
  `PROBE PASS max_diff=n/a .../L1b-issue-backpressure`, `PROBE PASS max_diff=0.0243 .../L2-delay-hidden c0=39.0`,
  `PROBE PASS max_diff=n/a .../I1-in-order-release`, `PROBE PASS max_diff=n/a .../I2-extra-commit`.
