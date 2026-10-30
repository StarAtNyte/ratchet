# mbarrier_ring_rolled_loop (S-slot mbarrier ring: rolled loop with runtime (stage, phase) ↔ compiler unrolling)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: an S-slot ring with a 1D `cp.async.bulk` + `mbarrier.arrive.expect_tx` producer and a generic-load consumer,
> S ∈ {3,4,6} × K ∈ {16,32} × {`#pragma unroll 1`, default}, consumer free-running / throttled, numerics all right;
> how the SASS counts change with S and K. The addendum to gotcha 5 (it may stay rolled even without the pragma) comes from another CUDA probe.
> Gotcha 10 (running counter vs computing from k every beat) comes from a third CUDA probe (`PROBE PASS max_diff=0.0`), certifying only numerics and the SASS footprint;
> the only whole-kernel reading is the one in gotcha 10. **Not covered**: probe certification of tensor-map TMA, a tcgen05.mma consumer side, 2-CTA pair / cluster versions; whole-kernel performance other than gotcha 10.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the wait encoding see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md),
> for choosing the ring depth see [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md);
> for the opposite direction (the producer hand-unrolled by S, addresses folded into base + immediate) see [`mbarrier_ring_unrolled_const_addr`](mbarrier_ring_unrolled_const_addr.md).

## Purpose

S slots of 1 KB + two 8-byte mbarrier tables `full[S]` / `empty[S]`. Each beat `stage = k mod S`,
`phase` flips when the stage wraps; the producer waits on `empty[stage]` with parity `phase ^ 1`, the consumer waits on `full[stage]`
with parity `phase`. This entry covers whether this loop is written under CUDA as **rolled** (one wait point / one copy point, stage and phase
in runtime registers) or left to the compiler to unroll, and the SASS footprint of each.

Measured on B200 / nvcc 13.1 (single CTA, one kernel per instance):

| Spelling | `BLKCP` (bulk copy) count | `PHASECHK` count | SASS instruction count |
|---|---|---|---|
| `#pragma unroll 1`, S=3/4/6, K=16/32 | **1** (same for all instances) | **4** (all the same) | 160–168 |
| no pragma, K a compile-time constant, K=16 | 16 | 40 | 520–568 |
| no pragma, K a compile-time constant, K=32 | 32 | 72 | 840–888 |

That is: **nvcc fully unrolls a ring loop with a compile-time constant trip count** (copy points = K, growing linearly with K, independent of S);
`#pragma unroll 1` makes the footprint independent of both S and K.

## When needed (trigger)

- Writing the producer / consumer loops of a TMA / bulk copy → mbarrier ring where the trip count is a runtime value or large (one unrolled copy per
  k-block makes the code grow linearly with K).
- In SASS the counts of `BLKCP` / `UTMALDG` and `SYNCS.PHASECHK` are integer multiples of one beat and change with the problem size (K):
  that is the compiler having unrolled a constant-trip loop.
- In a rolled loop (stage, phase) is computed from k every beat (`s = k % S; ph = (k / S) & 1`), S is not a power of 2, and SASS has `IMAD.HI`: see gotcha 10.
- **Anti-trigger**: rolled is not automatically faster. In this probe's single-CTA toy the unrolled version was actually faster (same job, median of 10, launch overhead included):

  | S / K | rolled | default unrolled |
  |---|---|---|
  | 3 / 16 | 10.85 µs | 9.25 µs |
  | 4 / 32 | 13.04 µs | 11.50 µs |
  | 6 / 32 | 13.06 µs | 10.77 µs |

  What rolled buys is code size independent of S and K and support for runtime trip counts; whether to roll is a matter for a same-job A/B.

## Minimal runnable example

```cpp
// Taken from the certification probe (S-slot 1 KB ring, numerics bitwise right for all instances on B200)
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
// mbar_init / mbar_arrive / mbar_wait: see mbarrier_wait_encoding

template <int S>
__global__ void ring(const int* __restrict__ src, long long* out, int K) {
  __shared__ __align__(128) int slot[S][256];        // 1 KB / slot
  __shared__ __align__(8) uint64_t full[S], empty[S]; // table length = 2*S
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < S; ++i) { mbar_init(&full[i], 1); mbar_init(&empty[i], 32); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  int stage = 0; uint32_t phase = 0;
  if (warp == 0) {
    if (lane == 0) {
#pragma unroll 1
      for (int k = 0; k < K; ++k) {
        mbar_wait(&empty[stage], phase ^ 1u);          // new ring: the first S waits pass at once, no prologue
        mbar_arrive_expect_tx(&full[stage], 256 * 4);  // arm first, then issue the copy
        bulk_g2s(&slot[stage][0], src + (size_t)k * 256, 256 * 4, &full[stage]);
        if (++stage == S) { stage = 0; phase ^= 1u; }
      }
    }
  } else {
    long long acc = 0;
#pragma unroll 1
    for (int k = 0; k < K; ++k) {
      mbar_wait(&full[stage], phase);
      const int4* p = reinterpret_cast<const int4*>(&slot[stage][0]);
      int4 a = p[lane * 2], b = p[lane * 2 + 1];
      acc += (long long)a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
      mbar_arrive(&empty[stage]);                      // count = 32: once per lane
      if (++stage == S) { stage = 0; phase ^= 1u; }
    }
    out[lane] = acc;
  }
}
```

## Gotchas

1. **Phase initial value: the producer waits on `phase ^ 1`, the consumer on `phase`, both starting from `phase = 0`.** The producer's first S
   waits therefore pass immediately on fresh barriers, no prologue needed; the trip count need not be a multiple of S, and no tail loop is needed.
2. **phase flips only when the stage wraps** (`if (++stage == S) { stage = 0; phase ^= 1; }`). Flipping every beat or forgetting to flip:
   the wait either never completes (hang) or reads the previous lap's stale data (silently wrong).
3. **One (stage, phase) bijection serves three places**: the copy's gmem source uses k, the smem slot address and both barriers use stage.
   Computing each of the three separately is very easy to get out of step.
4. **`arrive.expect_tx` first, then issue `cp.async.bulk`**; the expected byte count must equal the total bytes of all copies landing this beat,
   otherwise full never flips (expect too many) or flips early (expect too few, the consumer reads a half-filled slot).
5. **A constant-K loop without `#pragma unroll 1` is fully unrolled** (measured copy points = K). With a large trip count the instruction count grows linearly
   with K; if you want it rolled, write the pragma explicitly, **on both loops**.
   **But whether it unrolls depends on the loop body and cannot be judged from the source.** In another probe, a ring with K = 16 as a compile-time constant, no pragma, and
   slot index and parity computed with `k % S` / `k / S`, whose loop body had extra inner loops with runtime trip counts (a `%clock64` throttling spin, a per-thread strided copy loop),
   had only **1** copy point and 208 SASS instructions (the `#pragma unroll 1` + counter spelling in the same probe: 184), numerics right. So "no pragma" does
   not necessarily unroll, while "pragma written" never unrolls: if you want rolled, write the pragma, and always accept by counting the copy points in SASS.
6. **Do not count `PHASECHK` as "one per wait"**: one wait may lower to a TRYWAIT plus several in a retry block
   (the rolled version's two wait points read 4). Judge rolled by "copy points = 1 and the counts do not change with S, K".
7. **The `mbarrier.init` count growing with S is not unrolling**: the init loop itself being unrolled is normal.
8. **A `.shared::cluster` destination address is legal for a single CTA**: a non-cluster launch is a 1-CTA cluster, and this CTA's
   `__cvta_generic_to_shared` address lies in the shared::cluster window.
9. **Wait for the last lap to land before exiting**: if the producer exits right after issuing and the CTA finishes before the in-flight copies, the copies write into freed smem.
   In this probe the consumers exit only after waiting on every full; a spelling where the producer exits first must drain itself.
10. **Advancing (stage, phase) with a running counter (`if (++s == S) { s = 0; ph ^= 1u; }`, the spelling in the example above) vs computing it from k every beat**
    (`s = k % S; ph = (k / S) & 1`): the former has no `IMAD.HI`; in one whole-kernel reading the former was faster (see below), re-measure on another kernel. When S is not a power of 2, `k % S` and `k / S` compile to multiply-by-inverse division and SASS shows `IMAD.HI`.
    Measured on B200 / nvcc 13.1 (single CTA, 1D bulk copy producer + one consumer warp, `#pragma unroll 1` on both sides, runtime trip count,
    one copy each with a static `__shared__` array and with an `extern __shared__` block):
    - computed spelling: 0 `IMAD.HI` per kernel at S = 4, 2 each for the two instances where S is not a power of 2 (including S = 6);
    - counter spelling, and the spelling "the barrier / slot addresses themselves as loop-carried registers incremented every beat and reset on wrap": 0 for every S (the latter has the longest per-beat cycles of the three, see below);
    - all three spellings numerically bitwise right. So this is a **shape / performance question; it never computes wrong**.
    - Self-check: `IMAD.HI` in a rolled loop with S not a power of 2 means the computed spelling; after switching to a counter, `IMAD.HI` disappears and the back edge gains an `ISETP` + `SEL`.
    - **The probe's per-beat cycles do not support "the counter is faster"**: `%clock64` non-wait segments of the same job (per k, producer / consumer). With static smem the counter is
      109 / 135 cycles for every S, computed 113–115 / 130–133: the producer a few cycles shorter, the consumer a few longer; with `extern __shared__` the counter is 117 / 132–133,
      computed 111–117 / 119–130: neither side shorter. The address-increment spelling likewise has no `IMAD.HI` but the longest cycles: with static smem 142 / 167 for every S
      (both sides about 30 cycles longer than the counter), with `extern __shared__` 109 / 146–149 (shortest producer, longest consumer); the probe does not support choosing it to get rid of `IMAD.HI`. The previous version of the probe read the counter longer on both sides for the two non-power-of-2 S (129 / 142 vs
      115 / 137–138), shorter only at S = 4, so the "counter shorter on both sides" contract did not pass. The cycle differences in the toy go in mixed directions, cannot serve as a criterion, and do not explain the whole-kernel reading below.
    - **Whole-kernel reading (only one)**: on a `cta_group::2` tcgen05 GEMM (S not a power of 2, two rolled loops for producer and MMA), a single change —
      only replacing the computation in both loops with counters, k still used only for the TMA coordinates and the first MMA's accumulate predicate — made the gate window 10.9% faster; an earlier single-lever pricing on the same kernel pointed the same way
      (adding it 12.94% faster, removing it 11.13% slower). The source of the gain was not isolated: the probe's cycle differences are only a few cycles in mixed directions and cannot explain this magnitude. On another kernel go by a same-job A/B;
      when S is a power of 2 the computed spelling has no `IMAD.HI`; that case was not measured on a whole kernel.
    - **The "ring level" given by a SASS census cannot serve as a spelling criterion**: in the probe none of the three spellings reads as a ring under the census tool (`rings=[]`);
      nvcc lowered the address-increment spelling into a rotation between two registers, and the static-smem base chain also goes through `SR_CgaCtaId`. Judge the spelling by `IMAD.HI` and the `ISETP` / `SEL` on the back edge.

## Sources

- PTX ISA: *mbarrier* (init / arrive / arrive.expect_tx / try_wait.parity, phase and tx-count semantics),
  *cp.async.bulk* (`.shared::cluster.global.mbarrier::complete_tx::bytes`, 16-byte alignment and size requirements),
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide: `#pragma unroll`, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 mbarrier_ring_rolled_loop/R1-exact-all-instances-and-pacings`,
  `PROBE PASS max_diff=n/a mbarrier_ring_rolled_loop/R2-rolled-counts-invariant-in-S-K counts=[(1, 4)]`,
  `PROBE PASS max_diff=n/a mbarrier_ring_rolled_loop/R3-rolled-one-bulk-copy-site`.
- GPU-certified by a second probe (gotcha 5's "not necessarily unrolled"):
  `PROBE PASS max_diff=n/a` (B0-build), `PROBE PASS max_diff=0` (R1-ring-data), `PROBE PASS max_diff=n/a` (R2-ring-sass);
  copy points and instruction counts are same-job fact lines `bulk-copy issue sites: runtime ring=1  kb%STAGES ring=1`, `n_ins=184` / `n_ins=208`.
- GPU-certified by a third probe (gotcha 10, counter vs computed):
  `PROBE PASS max_diff=0.0 mbarrier_ring_counter_vs_modulo/C1-exact-all-instances`,
  `PROBE PASS max_diff=n/a mbarrier_ring_counter_vs_modulo/C2-modulo-nonpow2-has-IMAD.HI-counters-do-not`;
  the `IMAD.HI` counts, per-beat cycles and `rings` are same-job fact lines.
