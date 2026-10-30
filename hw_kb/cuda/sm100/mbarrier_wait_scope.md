# mbarrier_wait_scope (choosing the `.sem.scope` of mbarrier waits / arrives: SASS footprint and cost — `CCTL.IVALL` for a cluster acquire, `MEMBAR.ALL.GPU` / `ERRBAR` for a cluster release)

> **Certification source: B200 GPU-certified in CUDA mode (all 6 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: with cluster = 2, the numerics of the three `try_wait.parity` spellings `.acquire.cta` / default / `.acquire.cluster` on a bulk-copy ring and a cross-CTA exchange,
> their SASS footprints, and two timings (per-stage cycles of the ring, ping-pong round-trip cycles with / without generic global loads after the wait); the SASS footprints of
> `mbarrier.arrive` `.release.cta` / `.release.cluster` (local and `mapa` remote).
> **Uncertified**: whether cluster scope is **required** for generic cross-CTA publication — in the probe, downgrading it to `.acquire.cta` was also numerically right (see gotcha 1); rows 4 and 5 of the classification table rest on
> the PTX memory model; the two completion modes `tcgen05.commit` multicast and 2-SM TMA accounting on the leader barrier were not run in the probe; downgraded spellings of the remote arrive (see gotcha 3).
> **Whole kernel in a real kernel**: downgrading the main loop and all waits from `.acquire.cluster` to `.acquire.cta` priced flat as a lever (+0.2% / +0.4%, within noise) — the cost of `CCTL.IVALL` in a hot ring
> could not be measured either in this entry's probe or on the whole kernel.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the wait loop encoding (try_wait / test_wait / hint) see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md);
> for cluster barrier footprints see [`cluster_xchg_transport_ladder`](cluster_xchg_transport_ladder.md).

## Purpose

`mbarrier.try_wait` carries an **acquire** when the phase completes: data the completer wrote before its arrive (release) becomes visible to the waiting thread from then on. `.scope` decides whom this
acquire pairs with:

- `.cta`: pairs only with releases by threads of this CTA. A `try_wait` without `.sem` / `.scope` is `.acquire.cta` (the PTX default).
- `.cluster`: also pairs with releases by **other CTAs** of the cluster.

The arrive side is analogous: `mbarrier.arrive` defaults to `.release.cta`; arriving on another CTA's barrier (a `shared::cluster` address from `mapa`) needs `.release.cluster`.
Scope changes no numbers, only SASS and cost, so choose it barrier by barrier according to "who wrote what at completion, and what is read or written after the wait"; do not share one wait wrapper hard-coded to
`.acquire.cluster` across the whole kernel.

## When needed (trigger)

- In SASS, `SYNCS.PHASECHK.TRANS64.TRYWAIT` is immediately followed by `CCTL.IVALL` (one per wait point); the code after the wait still has generic global loads.
- In SASS there is `MEMBAR.ALL.GPU` / `ERRBAR` / `CGAERRBAR` next to an arrive within this CTA: that arrive was written as `.release.cluster`.
- The source has only one `mbar_wait` wrapper, hard-coded to `.acquire.cluster`, shared by the FULL / EMPTY / accumulator / cross-CTA barriers.

## Measured footprint and cost (probe, B200)

SASS of single primitives (one per kernel):

| Spelling | SASS |
|---|---|
| `mbarrier.try_wait.parity.acquire.cta.shared::cta.b64` | `SYNCS.PHASECHK.TRANS64.TRYWAIT` ×2 (fast path + retry block), no `CCTL` |
| `mbarrier.try_wait.parity.shared::cta.b64` (default) | as above, identical to `.acquire.cta` |
| `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64` | `TRYWAIT` ×2 + `CCTL.IVALL` ×2 |
| `mbarrier.arrive.release.cta.shared::cta.b64` | `SYNCS.ARRIVE.TRANS64.A1T0` |
| `mbarrier.arrive.release.cluster.shared::cta.b64` (local barrier) | `MEMBAR.ALL.CTA` + `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` + `SYNCS.ARRIVE.TRANS64.A1T0` |
| `mbarrier.arrive.release.cluster.shared::cluster.b64` (`mapa` remote) | the same four + `SYNCS.ARRIVE.TRANS64.RED.A1T0` |
| `fence.mbarrier_init.release.cluster` | no synchronization-class instruction |

The same 4-slot bulk copy ring (producer waits on empty, consumer waits on full): the `.acquire.cta` and default versions have 0 `CCTL.IVALL`, the `.acquire.cluster` version 4, one per
`TRYWAIT`, the first right after a `TRYWAIT`.

Timing (`%clock64` cycles, median):

| Scenario | `.acquire.cta` | default | `.acquire.cluster` |
|---|---|---|---|
| ring: 256 beats, consumer per stage (wait full → copy the 4 KiB slot to global → arrive empty) | 424.5 | 424.5 | 426.6 |
| ping-pong: two threads round-tripping on two mbarriers of this CTA, the waiter does no memory access | 106.1 | 106.1 | 107.1 |
| ping-pong: the waiter reads 8 L1-resident global words after each wait | 147.1 | 147.2 | **407.2** |

How to read it: a cluster acquire costs almost nothing in the wait itself (ring +0.5%, ping-pong +1 cycle); the cost lands on **generic global loads after the wait** — `CCTL.IVALL`
invalidates this SM's L1, so later L1 hits become refetches, and the round trip grows from 147 to 407 cycles (cta / cluster = 0.361). In rings that after the wait only read smem / only issue TMA and MMA,
changing the scope makes no measurable difference.

## Minimal runnable example

```cpp
// Default to .acquire.cta: the completer is the async proxy (complete_tx / tcgen05.commit) or a barrier of this CTA's threads
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    asm volatile("{\n\t.reg .pred P1;\n"
                 "WAIT_%=:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
                 "@!P1 bra WAIT_%=;\n\t}"
                 :: "r"(bar), "r"(parity) : "memory");
}
// Only for the kind of wait where "another CTA wrote data with generic stores, then arrive.release.cluster'd onto this CTA's barrier"
__device__ __forceinline__ void mbar_wait_acq_cluster(uint32_t bar, uint32_t parity) {
    asm volatile("{\n\t.reg .pred P1;\n"
                 "WAIT_%=:\n\t"
                 "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n\t"
                 "@!P1 bra WAIT_%=;\n\t}"
                 :: "r"(bar), "r"(parity) : "memory");
}
// Arrive: .release.cta within this CTA; the remote arrive is the one spelling tested in the probe (.release.cluster + shared::cluster address), downgraded spellings untested (see gotcha 3)
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cluster_bar) {   // cluster_bar = mapa(local address, target rank)
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(cluster_bar) : "memory");
}
```

Classify barrier by barrier (the question is "who wrote what at completion, and what do I read or write after the wait"):

| Who completes the barrier | What the waiter does after the wait | Wait with | Basis |
|---|---|---|---|
| `complete_tx` of TMA / `cp.async.bulk` (including smem → this CTA's smem bulk copies initiated by another CTA) | reads the smem written by these async writes, or hands it to MMA | `.acquire.cta` | PTX complete_tx semantics; probe: same-CTA ring and cross-CTA bulk exchange numerically right under `.acquire.cta` |
| `tcgen05.commit ... mbarrier::arrive::one` | producer overwrites the smem slot; epilogue reads TMEM (`tcgen05.fence::after_thread_sync` before reading) | `.acquire.cta` | PTX; not covered by the probe |
| `mbarrier.arrive` (`.release.cta`) by this CTA's threads | reads/writes this CTA's smem | `.acquire.cta` | PTX; the probe ring's empty barrier |
| `arrive.release.cluster` by **another CTA**, publishing its **generic** writes (`st.shared::cluster`) | reads those writes | `.acquire.cluster` | PTX memory model (the probe cannot tell them apart, see gotcha 1) |
| remote arrive by **another CTA** meaning "I am done reading, you may overwrite" (the other side reads with generic loads) | overwrites memory the other side read | `.acquire.cluster` | PTX memory model (not covered by the probe) |

## Gotchas

1. **Scope does not affect numerics; a probe cannot detect "used cta where cluster was required".** Measured: downgrading the wait on the generic-publication path of a cross-CTA exchange to `.acquire.cta` was still numerically right in all 3 repetitions;
   that only shows it did not go wrong this time on this machine, not that it is legal. Rows 4 and 5 of the classification table keep cluster scope per the PTX memory model; such handshakes are usually one-shot, so keeping it costs only once.
2. **The cost depends on whether generic global loads follow the wait.** In rings that after the wait only read smem and only issue TMA / MMA, cta and cluster differ by only about 0.5%; on a path where an L1-resident global load
   directly follows the wait, the cluster version's round trip is 2.8× slower (147 → 407 cycles). First look at what follows `CCTL.IVALL` in SASS, then decide whether changing the scope there is worth it.
3. **`.sem.scope` of a remote arrive: uncertified / untested.** The probe's remote arrive has only one spelling, `.release.cluster` + a `.shared::cluster` address (numerics and footprint above).
   Per the PTX memory model, when a remote arrive publishes generic writes on behalf of the other side, the release must cover the waiter's CTA; whether a remote arrive that only means "done reading", or whose data is booked by complete_tx / commit,
   can use a lighter spelling, and the SASS and cost of each, this entry has no readings for; verify with your own probe.
4. **Do not write `.release.cluster` on an arrive within this CTA.** It lowers to `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` + `MEMBAR.ALL.CTA` (see the footprint table), with no benefit for this CTA's consumers.
   The timing of this footprint was not measured in the probe.
5. **A barrier completed by `complete_tx` needs no cluster acquire to "see" the data the peer wrote in.** The data is written by the async proxy, and visibility is given by complete_tx at phase completion,
   independent of the wait's `.scope`; the probe's cross-CTA bulk exchange is numerically right with `.acquire.cta` on the bulk path.
6. **One-shot setup / teardown synchronization** (`fence.mbarrier_init.release.cluster`, `barrier.cluster.arrive.release` / `wait.acquire`, `fence.acq_rel.cluster`)
   also carries `CCTL.IVALL` / `MEMBAR.ALL.GPU` / `ERRBAR` (except `fence.mbarrier_init`: no synchronization-class instruction). **Do not ignore them by default in short kernels**: the probe did not time them;
   on a whole kernel of about 20 µs, switching the arrive of the prologue / epilogue cluster barriers from `.release` to `.relaxed` priced as a lever 3.3% faster on the whole kernel (the correctness of that spelling depends on
   whether the other publication paths have their own release; not probe-certified, verify it yourself).
7. **Accept in SASS after the change**: there should be no more `CCTL.IVALL` near the hot ring's `TRYWAIT`; having it near one-shot waits that keep cluster scope is expected.

## Sources

- PTX ISA: `mbarrier.try_wait` / `test_wait` (`.sem` = `.acquire`, `.scope` = `.cta | .cluster`, default `.acquire.cta`), `mbarrier.arrive` (`.release.cta | .release.cluster`,
  `.shared::cluster` remote arrive), `expect_tx` / `complete_tx`, `mapa`, `cp.async.bulk`, `fence.mbarrier_init`, `barrier.cluster`; the memory consistency model (scope, release / acquire pairing) —
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Public CUTLASS headers: `cutlass::arch::ClusterBarrier` (`wait`, `arrive(cta_id, pred)` remote arrive), `cutlass::arch::ClusterTransactionBarrier` (`include/cutlass/arch/barrier.h`).
- GPU-certified by the probe (cluster = 2):
  `PROBE PASS max_diff=n/a` (B0-build), `PROBE PASS max_diff=0` (R1-ring-data), `PROBE PASS max_diff=0` (X1-xchg-data), `PROBE PASS max_diff=n/a` (T1-cta-not-slower),
  `PROBE PASS max_diff=n/a` (S1-sass-cta-clean), `PROBE PASS max_diff=n/a` (S2-sass-cluster); footprints and timings are same-job fact lines.
