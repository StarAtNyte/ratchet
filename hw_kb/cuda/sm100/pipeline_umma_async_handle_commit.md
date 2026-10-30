# pipeline_umma_async_handle_commit (`tcgen05.commit` as a ring's producer release: the commit must hit the stage just filled)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: `tcgen05.commit.cta_group::1.mbarrier::arrive::one` as the full-side arrival of a 2-stage ring;
> release timing and numerics of the two spellings, commit on the right stage / on the next stage, and the hang with an odd tile count. The probe has **no real MMA / TMEM**
> (there is no tcgen05 async operation before the commit, so it arrives immediately); the `cta_group::2` multicast version is **uncertified**.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the wait encoding see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md).

## Purpose

After the MMA issue thread has issued one stage's `tcgen05.mma`, `tcgen05.commit ... mbarrier::arrive::one` makes the hardware arrive once on an mbarrier
**when all of that thread's earlier tcgen05 async operations have completed** — this is the release on the UMMA producer side (accumulator written) or the
UMMA consumer side (operands used up, can be recycled). This entry covers one of the most common silent errors: the commit hits the **next stage's**
barrier (the stage is taken after the counter has advanced).

## When needed (trigger)

- Writing the handoff of TMEM accumulator double buffering (the MMA thread commits to acc_full[stage], the epilogue waits on it) or of an operand ring
  (the MMA thread commits to empty[stage]).
- **Timeline fingerprint**: numerics all right, but the waits on both sides alternate bimodally by tile — even tiles wait long, odd tiles return immediately
  (or the other way round); problem sizes with an odd tile count hang on the last tile.
- Structural spelling: `commit(stage)` after `advance()`, or using a member stage after a call to a function that "takes the current stage and advances".

Measured on B200 (2-stage ring, producer spacing 100 µs per tile, lead[t] = number of commits already issued when the consumer's wait for tile t returns,
`%globaltimer` stamps):

| Spelling | T | lead | Values read back |
|---|---|---|---|
| commit on the stage just filled (`stage` taken before advancing) | 8 | `[1,2,3,4,5,6,7,8]` | all right |
| commit on the next stage (`(t+1) % 2`) | 8 | `[2,2,4,4,6,6,8,8]` | **still all right** |
| right stage | 7 | `[1..7]` | all right |
| next stage | 7 | `[2,2,4,4,6,6, —]`, the last wait never completes | first 6 right |

With the wrong stage every even tile waits an extra whole tile's production period, yet the numbers do not show it.

## Minimal runnable example

```cpp
// Taken from the certification probe
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// The same PTX as CUTLASS cutlass::arch::umma_arrive (cta_group::1)
__device__ __forceinline__ void umma_commit_arrive(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(smem_u32(bar)) : "memory");
}

// producer (single thread): stage / phase advance only after use
int stage = 0; uint32_t phase = 0;
for (int t = 0; t < T; ++t) {
  mbar_wait(&empty[stage], phase ^ 1u);
  // ... issue this stage's tcgen05.mma (or write this stage's data; after generic st.shared you need
  //     asm volatile("fence.proxy.async.shared::cta;" ::: "memory");) ...
  umma_commit_arrive(&full[stage]);              // right: hits the stage just filled
  if (++stage == 2) { stage = 0; phase ^= 1u; }  // advance after use
  // wrong: advance first, then umma_commit_arrive(&full[stage]) — hits the next stage
}
// consumer: mbar_wait(&full[stage], phase); read; mbar_arrive(&empty[stage]); advance
```

## Gotchas

1. **The stage used by the commit must be "the stage just filled"**: compute `(stage, phase)` first, issue the operations, commit, and only then advance
   the counter. Wrapping "take stage + advance" into a function that returns the new state and then taking stage from that state is an off-by-one.
2. **With the wrong stage the numerics are still right**: the commit tracks **all** earlier tcgen05 async operations of the issuing thread, so a commit one beat late still
   covers this stage's MMA; the symptoms show only in timing (bimodal lead) and in the hang with an odd tile count. The criterion must use timestamps / bounded waits,
   not max_diff.
3. **An even tile count masks the hang**: with S=2 and even T the two offsets cancel out and it runs to completion; with odd T the commit the last consumer
   waits on never exists. A probe must test both odd and even T.
4. **Both sides' buffer indices come from the same (stage, phase)**: one side computing `t & 1` by hand and the other using the advanced counter may happen to agree at S=2,
   and break at S=3.
5. **When smem written generically is read by the async proxy (tcgen05 / TMA) or published via commit, add
   `fence.proxy.async.shared::cta` before the commit.**
6. **The commit is issued by a single thread** (elect one lane); if the whole warp issues it, it arrives 32 times.
7. **The producer must drain before exiting**: the commits of the last stages arrive asynchronously; if the producer exits without waiting on empty and the CTA finishes before the arrivals,
   the arrivals land on freed smem.
8. **Debug hangs with bounded waits**: replace the blocking wait with a capped `mbarrier.test_wait` poll that sets a flag, and a hang becomes a readable
   timeout flag (this probe's H3 was read out exactly that way).

## Sources

- PTX ISA: *tcgen05.commit* (`.cta_group::1`, `.mbarrier::arrive::one`, tracks all earlier tcgen05
  async operations of the issuing thread), *mbarrier*, *fence.proxy.async*, https://docs.nvidia.com/cuda/parallel-thread-execution/
- Public API symbols: CUTLASS `cutlass::arch::umma_arrive` (`include/cutlass/arch/barrier.h`).
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 pipeline_umma_async_handle_commit/H1-commit-on-filled-stage lead=[1, 2, 3, 4, 5, 6, 7, 8]`,
  `PROBE PASS max_diff=0.0 pipeline_umma_async_handle_commit/H2-commit-on-next-stage-late-release lead=[2, 2, 4, 4, 6, 6, 8, 8]`,
  `PROBE PASS max_diff=n/a pipeline_umma_async_handle_commit/H3-odd-tile-count-last-wait-starves`.
