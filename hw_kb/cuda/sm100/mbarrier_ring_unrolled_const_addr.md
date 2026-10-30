# mbarrier_ring_unrolled_const_addr (hand-unrolling the producer of an S-slot mbarrier ring: outer loop rolled by round, inner loop unrolled by S + guarded tail; stage becomes a compile-time constant, barrier / slot addresses fold into "base + immediate"; issue-segment readings for the 1D bulk and pair tensor-TMA paths)

> **Certification source: B200 GPU probes in CUDA mode (not included in this release), three probes, all `PROBE PASS max_diff=0.0`:**
> - 1D `cp.async.bulk` ring: U1-U3 probe (U1 numerics, U2 copy-point count, U3 issue segment, 3/3 PASS);
> - three loop shapes of a pair (`cta_group::2`) tensor-TMA producer: round-issue probe (3/3 PASS);
> - the wrap segment of a pair tensor-TMA producer (rolled / round shape / round shape + empty asm pinning registers): issue-segment probe (3/3 PASS).
>
> What is certified is numerics, the SASS footprint and issue-segment cycles. **The whole-kernel gain is not certified**: kernel times in the probes are flat or mixed across the three shapes; three whole-kernel landings were all within noise (see "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the rolled spelling itself (phase initial value, flip rule, `#pragma unroll 1`, the compiler fully unrolling constant trips) see
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md); for the same trade-off on the MMA warp side see
> [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md); for the wait encoding see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md);
> for pair load accounting see [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md).

## Purpose

The producer of a rolled ring computes `stage` and `phase` every beat: barrier address = base + 8·stage, slot address = base + stage·slot size,
and the wrap `if (++s == S) { s = 0; ph ^= 1; }` lowers to a compare + select. Hand-unrolling makes stage a compile-time constant: each stage has its own copy of the
wait, arm and copy code, addresses are a base register + immediate, and phase flips once per round (S beats).

Two unrolled shapes:

- **Unrolled by S (round shape)**: the outer rolled loop walks rounds, the inner `#pragma unroll` loop walks the S stages, phase flips at the end of each round;
  the remainder `K % S` is covered by a tail that is unrolled the same way with an `if (s < rem)` guard (stage starts from 0, phase carries on). The copy-point count is independent of the trip count.
- **Fully unrolled**: when both trip and S are compile-time constants, `#pragma unroll` the whole loop, and `stage = k % S`, `phase = (k / S) & 1` all constant-fold.
  This is what nvcc often does by itself for "constant trip, no `#pragma unroll 1`", but whether it unrolls depends on the loop body (see gotcha 5 of the rolled entry); if you want it unrolled, write `#pragma unroll` explicitly. Copy points = trip.

## When needed (trigger)

- The producer loop is written as `#pragma unroll 1` + runtime `s / ph`, the address operand of `SYNCS.PHASECHK` in the loop body has no immediate
  (`[R..+URZ]`), and there is an `ISETP` + `SEL` doing the stage wrap before the back edge;
- on the timeline the producer's per-beat wrap segment ("wait returns → next wait starts") is longer than the consumer's per-beat period, and the consumer's full wait lands in the producer's issue segment.
- **Anti-trigger / gain ceiling**:
  - The producer really spends most of its time waiting on `empty` (slow consumer): changing the loop shape only shortens the issue segment, and the cycles saved turn into waiting. All three probes of this entry are like that —
    the issue segment gets shorter, the wait correspondingly longer, kernel time flat or mixed (readings below).
  - **Whole-kernel readings (the same class of kernel, pair tensor-TMA producer + MMA ring): three landings, all within noise**: producer / drain / MMA ring fully unrolled,
    gate 0.0149 → 0.0150 ms; switched to the round shape, 0.0143 → 0.0142 ms; on the round shape, additionally removing the empty asm register pins and halving the loads per beat, 0.0136 → 0.0136 ms.
    Before changing it, first confirm the wrap segment is really on the critical path; the ring reading going empty in a census (after unrolling, the runtime stage index can no longer be recognized) is only the footprint of unrolling, not a gain.

## Minimal runnable example

```cpp
// Each beat: wait empty[s] -> arm full[s] -> this beat's copies land in slot s with complete_tx to full[s].
// Encodings of mbar_wait / mbar_init / mbar_arrive: see mbarrier_wait_encoding (try_wait loop without a hint).
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// s is a compile-time constant at the unrolled call site -> all three addresses below are base + immediate.
// a0 / b0 / full0 / empty0 are each computed once outside the loop with __cvta_generic_to_shared (32-bit shared-window addresses).
__device__ __forceinline__ void prod_step(uint32_t a0, uint32_t b0, uint32_t full0, uint32_t empty0,
                                          const int* ga, const int* gb, int k, int s, uint32_t ph) {
  mbar_wait(empty0 + 8 * s, ph ^ 1u);                 // new ring: the first round passes at once
  const uint32_t fb = full0 + 8 * s;
  mbar_arrive_expect_tx(fb, A_BYTES + B_BYTES);       // arm first (bytes = sum of this beat's copies), then issue the copies
  bulk_g2s(a0 + s * A_BYTES, ga + (size_t)k * (A_BYTES / 4), A_BYTES, fb);
  bulk_g2s(b0 + s * B_BYTES, gb + (size_t)k * (B_BYTES / 4), B_BYTES, fb);
}

// (a) round shape: runtime K, phase flips once per round, the remainder goes through a guarded unrolled tail
const int R = K / S, rem = K - R * S;
uint32_t ph = 0;
#pragma unroll 1
for (int r = 0; r < R; ++r) {
#pragma unroll
  for (int s = 0; s < S; ++s) prod_step(a0, b0, full0, empty0, ga, gb, r * S + s, s, ph);
  ph ^= 1u;                                           // fixed flip point
}
#pragma unroll
for (int s = 0; s < S - 1; ++s)
  if (s < rem) prod_step(a0, b0, full0, empty0, ga, gb, R * S + s, s, ph);

// (b) fully unrolled: KT and S both compile-time constants
#pragma unroll
for (int k = 0; k < KT; ++k) prod_step(a0, b0, full0, empty0, ga, gb, k, k % S, (k / S) & 1);

// drain before exit (needed only by spellings where the producer exits first): wait one more round of empty with the same (k -> stage, phase) bijection
```

The pair tensor-TMA path uses the same round shape: in `prod_step`, wait on this CTA's `empty[s]`, the leader arms its own `full[s]` with the whole pair's bytes,
and both CTAs' `cp.async.bulk.tensor.*.cta_group::2` hit the leader's `full[s]` after `mapa` (accounting in `cta_pair_bulk_load_cta_group2`);
the k coordinate is written as "round base + s × BK", with the round base incremented once on the outer back edge:

```cpp
int kc = k_begin;                                     // round base
#pragma unroll 1
for (int r = 0; r < R; ++r) {
#pragma unroll
  for (int s = 0; s < S; ++s) step_pair(p, s, ph, kc + s * BK);   // s constant -> barrier / slot address / coordinate offset are all immediates
  ph ^= 1u;
  kc += S * BK;
}
// tail as in (a): #pragma unroll + if (s < rem), coordinate kc + s * BK
```

## Readings

**1D `cp.async.bulk` ring** (1 CTA × 64 threads, two copies per beat sharing one full barrier, two S (S₁ < S₂), two trips (a / b, including instances not divisible by S);
"issue segment" = wait returns → this beat's last copy issued, excluding the inter-beat stage-advance code, unit cycles / beat):

| Shape | Issue segment S₁ (trip a / b) | Issue segment S₂ (trip a / b) | kernel µs S₁ (trip a / b) | kernel µs S₂ (trip a / b) |
|---|---|---|---|---|
| rolled (`#pragma unroll 1`) | 160.0 / 160.0 | 160.0 / 160.0 | 11.68 / 12.54 | 10.88 / 13.25 |
| round shape + tail | 120.5 / 123.7 | 128.6 / 126.5 | 11.42 / 13.18 | 11.36 / 12.35 |
| fully unrolled | 120.8 / 119.2 | 120.5 / 119.0 | 11.10 / 13.22 | 11.55 / 12.86 |

The issue segment is about 20–25% shorter, the producer's wait on empty correspondingly longer (S₁ trip a: 90.7 → 143.6 / 152.5 cycles / beat), kernel times mixed.
SASS: copy points = copies per beat × compile-time stage count (rolled 2; round shape 2 × (2S − 1); fully unrolled 2 × K), want == got for all instances;
the `PHASECHK` address is `[R3+URZ]` / `[R7+URZ]` when rolled, and `[R20+URZ+0x8]`, `[R20+URZ+0x10]` … in the unrolled shapes (the same base register, an immediate increasing by stage).

**Pair tensor-TMA producer** (cluster (2,1,1), NC `UTMALDG.*.2CTA` per CTA per beat):

Single lane = issue in an `if (lane == 0)` branch (in SASS each `UTMALDG` is wrapped in an `ELECT`; the readings include that cost); NC / NC/2 = the same bytes with the per-beat load count halved (rank-3 box):

| Shape | Issue segment (wait returns → last load) NC / NC/2 | Wrap segment (wait returns → next wait starts) NC / NC/2 |
|---|---|---|
| rolled, single lane | 217.0 / 129.0 | 247.0 / 157.0 |
| round shape, single lane | 200.6 / 114.6 | 210.9 / 122.4 |
| round shape, single lane, empty asm pinning addresses and coordinates before the wait | — | 210.7 / 121.9 |
| fully unrolled, single lane | 204.2 / 109.1 | — |

- The issue-segment and wrap-segment columns come from two probes with different conventions (the former excludes the inter-beat advance); do not subtract across columns. With 74 clusters running at once the readings barely change
  (wrap: rolled 247.0, round shape 209.4, pinned 209.3, NC).
- The round shape's wrap segment is 36 / 35 cycles shorter than rolled (NC / NC/2); fully unrolled vs round shape in the issue segment is +3.6 / −5.5 cycles, no consistent gain.
- **The bulk of the wrap segment is the `UTMALDG` count itself**: halving the per-beat load count for the same bytes (merging each operand's multiple SW128 atoms into one rank-3, see
  [`tma_3d_operand_box_2cta`](tma_3d_operand_box_2cta.md)) makes the round shape's wrap segment about 42% shorter (NC → NC/2: 210.9 → 122.4), far more than the differences between loop shapes; whole kernel flat (see "When needed").
- Probe kernel time (1 cluster) 18.59–19.68 µs across the three shapes, flat; the producer's wait on empty grows as the issue segment shrinks.
- SASS: copy points = NC × compile-time stage count (rolled NC, round shape NC × (2S − 1), fully unrolled NC × trip), all `UTMALDG.*.2CTA`;
  the round shape's `PHASECHK` address is a uniform base + an immediate increasing by 8 per stage (`[UR+imm]`), rolled is `[R+URZ]` without an immediate.

## Gotchas

1. **Addresses must be written as "base + constant × stage"** for the compiler to fold them into immediates. Written as `&bar[s]` with `bar` a pointer fetched inside the loop, or with stage
   going through a runtime array / runtime modulo, it is still a register address even when unrolled. Accept by SASS: the barrier operands of `SYNCS.PHASECHK` and `BLKCP` / `UTMALDG` carry immediates,
   different for each stage; whether the base lands in `R` or `UR` is up to ptxas (the two probes each read one kind); the criterion is the immediate, not the register class.
2. **phase flips only at the end of a round**. The round shape's inner loop must not contain another `if (++s == S)`; the tail's stage starts from 0 and uses the phase as flipped at the end of the last round.
   When fully unrolled, phase is `(k / S) & 1`. Flipped wrong, it either hangs or reads the previous lap's slot (silently wrong).
3. **Both sides (producer / consumer) use the same (k → stage, phase) bijection**. Unrolling only one side is fine (the 1D probe's consumer stays rolled),
   but for the same k both sides must compute the same stage / phase; check the tail especially when `K % S != 0` (the probe covers this with trips that are not multiples of S).
4. **Write the tail guard as `if (s < rem)` inside a constant-trip `#pragma unroll` loop**, so that s stays a constant. A tail written as a rolled small loop with a runtime trip
   puts stage back into a register.
5. **Code size**: the round shape's copy-point count is independent of trip (copies per beat × (2S − 1)); full unrolling grows linearly with trip. Use the round shape when trip is a runtime value or large.
6. **`#pragma unroll 1` turns unrolling off**; for a constant trip without it, nvcc may fully unroll or may not (depending on the loop body; two readings in gotcha 5 of the rolled entry); accept both directions by the SASS copy points.
7. **Pinning addresses / coordinates into registers with an empty `asm volatile("" : "+r"(x) ...)` before the wait saves no cycles**: on the round shape the wrap segment pinned vs unpinned is
   210.7 / 210.9 (NC), 121.9 / 122.4 (NC/2); the total SASS instruction count is the same, `PHASECHK` addresses are still uniform base + immediate, only the integer adds
   moved between `IADD` / `UIADD`. Addresses of a compile-time stage fold into immediates anyway; no need to manually "move them out of the wait → issue chain".
   In a rolled loop with a runtime stage, only hoisting + an empty asm pin makes it stick (address arithmetic 13 → 4, but `R2UR` 2 → 10, whole kernel flat), see
   [`tma_refill_post_wait_hoist`](tma_refill_post_wait_hoist.md).
8. **Write the k coordinate as "round base + s × BK"**, incrementing the round base once on the back edge. Mixing in a runtime index (a runtime-modulo wrap and the like) adds another integer chain before every load,
   and it sits right on the wait → issue dependency path.
9. **Change only the loop shape, not the protocol**: the barrier waited on, the parity, the `expect_tx` bytes, who arms, who commits all stay the same. After the change, first run the numeric comparison,
   then look at the SASS copy-point count (one per compile-time stage).
10. **Bracket the right interval when measuring the issue segment**: "wait returns → this beat's last copy issued" excludes the inter-beat stage advance / back edge; to compare loop shapes, measure
    "wait returns → next wait starts" (the wrap segment). The two pair probes' readings differ by 28–30 cycles on rolled and 8–10 cycles on the round shape
    (the two probes' code is not identical; use only as an order of magnitude).

## Sources

- PTX ISA: *mbarrier* (init / arrive / arrive.expect_tx / try_wait.parity, phase and tx-count semantics),
  *cp.async.bulk* (`.shared::cluster.global.mbarrier::complete_tx::bytes`), *cp.async.bulk.tensor* (`.cta_group::2`), *mapa*,
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide: `#pragma unroll`; CUDA Driver API: `cuTensorMapEncodeTiled`.
- GPU-certification probes (verdict lines verbatim):
  - U1-U3 probe:
    `PROBE PASS max_diff=0.0 mbarrier_ring_unrolled_const_addr/U1-exact-all-instances`,
    `PROBE PASS max_diff=n/a .../U2-copy-sites-per-compile-time-stage`, `PROBE PASS max_diff=n/a .../U3-unrolled-issue-segment-shorter cyc_per_k=S4:rolled160_vs_byS120,S6:rolled160_vs_byS129`.
  - round-issue probe: `R1-exact-all-forms` (max_diff=0.0), `R2-load-sites-per-compile-time-stage`,
    `R3-roundsS-issue-shorter-than-rolled cyc_per_k=rolled217.0_roundsS200.6`.
  - issue-segment probe: `R1-exact-all-variants` (max_diff=0.0), `R2-load-sites`,
    `R3-nc2-wrap-shorter-than-nc4 cyc_per_k=nc4_210.9_nc2_122.4`.
