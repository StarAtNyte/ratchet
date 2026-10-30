# mbarrier_wait_encoding (encoding of an mbarrier wait: try_wait with a suspend hint ↔ try_wait without a hint ↔ test_wait spinning)

> **Certification source: B200 GPU-certified in CUDA mode (all 4 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: numerical correctness and determinism of the four spellings, the PTX spelling → SASS instruction family mapping, the parity convention on a fresh barrier.
> The timing readings come from a ping-pong microbenchmark in the same job (every wait really waits), **not a whole-kernel gain**; the gain of changing the wait encoding in a whole kernel is **uncertified**.
> Cluster-scoped waits (`.shared::cluster` / remote barriers) are not covered.
>
> One file per hardware interface. Operator-agnostic: mechanism only. This entry covers only **how a wait is encoded**; for how to build the ring / phase see
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md), for where ptxas places a wait's retry block see
> [`mbarrier_try_wait_fastpath_layout`](mbarrier_try_wait_fastpath_layout.md).

## Purpose

For the same mbarrier and the same phase, PTX offers three ways to wait with the same synchronization semantics (all acquire, all honoring the same parity);
the only difference is "what this warp does while the phase is not complete":

| Spelling | PTX | SASS (B200, measured with nvcc 13.1) |
|---|---|---|
| try_wait loop with a suspend hint | `mbarrier.try_wait.parity.shared::cta.b64 p, [bar], par, hint_ns;` | `SYNCS.PHASECHK.TRANS64.TRYWAIT` → `@!P0 NANOSLEEP.SYNCS 0x989680` → `@!P0 SYNCS.PHASECHK.TRANS64` (retries use plain PHASECHK) |
| try_wait loop without a hint | `mbarrier.try_wait.parity.shared::cta.b64 p, [bar], par;` | only `SYNCS.PHASECHK.TRANS64.TRYWAIT`, no NANOSLEEP |
| test_wait spinning | `mbarrier.test_wait.parity.shared::cta.b64 p, [bar], par;` | only plain `SYNCS.PHASECHK.TRANS64` (no `.TRYWAIT`), no NANOSLEEP |

`NANOSLEEP.SYNCS` appears only in the spelling with a hint: **counting NANOSLEEP in SASS tells you which encoding a wait point uses**.

## When needed (trigger)

- The sleep family has a significant share in the stall ledger, and the rows hit by `NANOSLEEP.SYNCS` trace back exactly to mbarrier wait points
  (producer waiting on empty / consumer waiting on full).
- Waits **really do happen often** (every beat waits on the other side): a 2-slot ping-pong on B200 with 148 CTAs × 2 warps (each handshake waits once on each side),
  time per handshake (same job, median of 20 launches):

  | Spelling | ns / handshake |
  |---|---|
  | label loop with hint 0x989680 | **185.1** |
  | label loop without a hint | **94.4** |
  | C++ `while` around a one-shot hint-free try_wait | **91.7** |
  | C++ `while` around a one-shot test_wait | **177.6** |

  That is, in this ping-pong shape the hinted loop is about 2× slower; **test_wait spinning is no faster than the hinted loop** — for a loop that "waits until the phase completes",
  choose the **hint-free try_wait loop**, not test_wait spinning. **This applies only to loops that wait until done**: a one-shot
  probe that "checks once and goes off to do something else if not ready" must use test_wait (a one-shot try_wait suspends for about 7.8k cycles when the phase is incomplete, see gotcha 5).
- **Anti-trigger**: when waits rarely really happen (the producer is far enough ahead, the stages deep enough), the fast path of all three spellings is one `TRYWAIT`
  / `PHASECHK` plus one predicated branch, with no measurable difference (fast-path measurements in `mbarrier_try_wait_fastpath_layout`).
  This is a lever that needs a same-job A/B, not a default.

## Minimal runnable example

```cpp
// Taken from the certification probe (4 variants x 12 launches bitwise consistent on B200)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// (a) label loop with a suspend hint: SASS will show NANOSLEEP.SYNCS 0x989680
__device__ __forceinline__ void mbar_wait_hint(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity), "r"(0x989680u) : "memory");
}

// (b) label loop without a hint: only TRYWAIT, no NANOSLEEP (the fastest setting in the ping-pong microbenchmark)
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra LAB_WAIT;\n"
      "}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}

// (c) one-shot query: returns the predicate, the caller writes the loop (while (!mbar_try_wait(bar, ph)) {})
__device__ __forceinline__ uint32_t mbar_try_wait(uint64_t* bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n"
      "}\n"
      : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  return ok;
}
// The test_wait version replaces try_wait with mbarrier.test_wait.parity (pure query, never suspends).
```

## Gotchas

1. **The parity convention is the same for both encodings** (measured, fresh barrier, count=1): before the arrival, try_wait / test_wait on parity 1
   is immediately true and on parity 0 false (a one-shot try_wait on parity 0 first suspends for a while before returning false; this suspension does not change with the hint —
   [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) measured it returning after about 7.9k cycles for hints from 100 ns to 100 µs);
   after one arrival completes phase 0, waiting on parity 0 is true and on parity 1 false. So the producer waits on empty starting from parity 1 and
   the consumer waits on full starting from parity 0; the producer's first S waits pass naturally, no prologue needed.
2. **The asm must be `volatile` with a `"memory"` clobber**: that is what keeps the smem reads after the wait and the smem writes before it from being
   moved across the wait point by the compiler. The wait's own acquire semantics govern only hardware order, not compiler order.
3. **Wrap labels inside the asm in `{ }`**: when the same wait function is inlined several times (producer, consumer, loops), PTX labels only avoid clashing
   within a `{}` scope.
4. **Use a register operand for parity** (`"r"`), with the caller maintaining `phase ^= 1`; do not compile one constant asm per phase.
5. **test_wait is not "the faster non-blocking"**: in the ping-pong it is as slow as the hinted loop (177.6 vs 185.1 ns/handshake);
   the hint-free try_wait is the fast setting (91.7–94.4 ns). try_wait has a built-in hardware suspension that wakes
   when the phase completes; each retry of a test_wait loop is one issue + one barrier read. **Conversely, "check once and go do something else if not ready"
   must use test_wait**: a one-shot try_wait likewise suspends when the phase is incomplete (until completion or an implementation-defined time limit),
   and used as a probe it stalls the caller — see [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md).
6. **A C++ `while` loop around a one-shot try_wait costs the same as an asm label loop** (91.7 vs 94.4 ns/handshake, SASS has only
   `TRYWAIT` in both): choose the spelling for readability, not performance.
7. **CUTLASS barrier wrappers may carry a hint**: before using a wrapper such as `cutlass::arch::ClusterBarrier::wait`,
   check in the local CUTLASS headers whether its try_wait carries a suspend-time operand (grep `0x989680`), then decide whether to
   write a hint-free version yourself.
8. **A numeric probe cannot see the encoding difference**: the four spellings give bitwise identical results (deterministic over 12 launches); the only criteria are the
   NANOSLEEP / TRYWAIT / PHASECHK counts in SASS plus same-job timing.

## Sources

- PTX ISA: *Parallel Synchronization and Communication Instructions: mbarrier* —
  `mbarrier.try_wait` / `mbarrier.test_wait` (phaseParity semantics, the suspendTimeHint operand, try_wait may suspend for an
  implementation-defined time, test_wait never suspends), https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide: *Inline PTX Assembly* (`asm volatile`, the `"memory"` clobber),
  https://docs.nvidia.com/cuda/inline-ptx-assembly/
- Public API symbols: CUTLASS `cutlass::arch::ClusterBarrier` (`include/cutlass/arch/barrier.h`).
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 mbarrier_wait_encoding/C1-exact-4arms-x12`,
  `PROBE PASS max_diff=n/a mbarrier_wait_encoding/C2-nanosleep-only-in-hint-arm nanosleep=2/0/0/0`,
  `PROBE PASS max_diff=n/a mbarrier_wait_encoding/C3-ptx-family-to-sass trywait=2/4/4/0 plain=2/0/0/2`,
  `PROBE PASS max_diff=0.0 mbarrier_wait_encoding/C4-parity-convention-fresh-barrier`.
