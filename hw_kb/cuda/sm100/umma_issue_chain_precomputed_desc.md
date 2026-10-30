# umma_issue_chain_precomputed_desc (the MMA warp's K loop: stage number kept at runtime vs unrolled by the stage count into a compile-time constant; SASS chain length and timing readings on the CUDA spelling)

> **Certification source: re-certified on a GPU probe in CUDA mode** (variants `v64` / `v64u4` / `v64u8` / `v64s8`; the probe is not included in this release;
> verdicts, SASS census and timing under Sources). The mechanism is the same as the same-named entry in the CuTe DSL tree; **the readings were re-taken on the CUDA spelling and differ from the DSL entry's**:
> the CUDA spelling's runtime-stage loop itself has almost no bookkeeping between MMAs, and what unrolling buys in the probe is per-k-block cycles, not a shorter chain between MMAs.
> The probe above has no corresponding whole-kernel reading.
>
> The section "Ring depth not dividing the loads per round: unroll by lcm" has a separate CUDA probe: `PROBE PASS max_diff=0.0` (round 2),
> certifying the numerics (three spellings, two ring depths, two N, with / without tail, bitwise exact) and the SASS footprint (MMA site count, `IMAD.HI` count);
> that section's cycles / times are single fact lines of the same job; one whole-kernel landing, not priced (see that section).
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the readings at three residency levels and their breakdown see [`umma_mainloop_wrap_segment_issue_floor`](umma_mainloop_wrap_segment_issue_floor.md).

## Purpose

In the K loop of a TMA → UMMA operand ring, the MMA warp does each beat: wait on `full[s]`, issue NU `tcgen05.mma`, commit to `empty[s]`, then advance stage / phase.
Two spellings:

- **Runtime stage**: `for (k) { s = k % S; ... }` — the stage base, barrier address and phase are all computed from k;
- **Unrolled by S**: `for (kr) { #pragma unroll for (s = 0; s < S; ++s) kblock(kr*S + s, s); }` — s is a compile-time constant, descriptor offsets and barrier addresses fold into immediates,
  and phase flips every S beats (requires `KT % S == 0`; the remainder needs a separate tail).

Probe readings on the CUDA spelling (M = 128, N = 64, 4 MMAs per beat; SASS census on a kernel without timing instrumentation; timing kernels compiled separately):

| Spelling | SASS: non-MMA instructions between a group of 4 `UTCHMMA` | SASS: instruction distance from `try_wait` to the group's first MMA | Streaming wrap per k-block (cycles) | Resident per k-block (cycles) |
|---|---|---|---|---|
| runtime stage, S = 4 | [4, 0, 0] / [1, 0, 0] | 37–40 | 324 | 295 |
| runtime stage, S = 8 | as above | 37–40 | 324 | 297 |
| unrolled by S = 4 | [0, 0, 0] … [1, 0, 0] | 37–47 | 280 | 229 |
| unrolled by S = 8 | [0, 0, 0] … [7, 2, 3] | 28–41 | 272 | 214 |

How to read it:
- **In the CUDA runtime-stage loop there is almost no bookkeeping between MMAs**: ptxas computes the stage base once into a uniform register, and the descriptor low words of the 4 MMAs are
  `UIADD3 …, 0x2 / 0x4 / 0x6` (matching the descriptor's +2 K step), the high word `UMOV 0x40004040`, idesc one `UMOV`.
- In the probe, unrolling saves 44–83 cycles per k-block (streaming wrap / resident), but the chain between MMAs is not shorter (unrolled S = 8 even has 7 / 2 / 3 instructions in the later groups);
  where the saved cycles come from was not isolated (inference: loop back edge, `k % S`, computing the phase). After unrolling there is still one warp-level `elect.sync` / reconvergence per k-block,
  and the resident reading has dropped to 229 / 214, so the ≈ 280–295 of the rolled spelling is not the floor of the elect / reconvergence itself.
- The larger the MMA count per lap (S × NU), the more descriptors stay resident in uniform registers after unrolling; this probe's S = 8 × 4 = 32 read no spill (`LDL` / `STL` = 0),
  but move instructions already appear between MMAs. **Before switching, count the SASS on your own tree and do a same-job A/B.**

## When needed (trigger)

- In SASS, inside the K loop body, after `try_wait` and before the first `UTC*MMA`, and between MMAs, there is bookkeeping such as the stage base (`IMAD` / `LEA`), the wraparound (`ISETP` / `SEL`),
  the phase (`LOP3`);
- the timeline shows the MMA warp's per-beat part "outside the wait" clearly larger than the wait itself, and an ablation of "issue a few fewer UMMAs per beat" gives a speed-up beyond noise.

**Not the right remedy**: the MMA itself is the bottleneck (tensor pipe saturated); the full wait takes most of each beat (that is a feed problem).

## Minimal runnable example

Same spelling as the certification probe (inside `kblock`: the whole warp waits on `full[s]` → `tcgen05.fence::after_thread_sync` → NU MMAs +
commit `empty[s]` inside `elect.sync` → `__syncwarp()`; see [`umma_raw_desc_issue`](umma_raw_desc_issue.md)):

```cpp
auto kblock = [&](int k, int s) {                     // s is a compile-time constant in the unrolled spelling
  mbar_wait(full0 + 8 * s, (k / S) & 1);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (elect_one()) {
    const uint32_t a_s = a_lo0 + s * (A_BYTES >> 4), b_s = b_lo0 + s * (B_BYTES >> 4);
#pragma unroll
    for (int kk = 0; kk < NU; ++kk)
      umma_f16(tbase, a_s + 2 * kk, HI, b_s + 2 * kk, HI, IDESC, (k | kk) != 0);
    tc_commit(empty0 + 8 * s);
  }
  __syncwarp();
};
// runtime stage
#pragma unroll 1
for (int k = 0; k < KT; ++k) kblock(k, k % S);
// unrolled by S (KT % S == 0)
for (int kr = 0; kr < KT / S; ++kr) {
#pragma unroll
  for (int s = 0; s < S; ++s) kblock(kr * S + s, s);
}
```

## Gotchas

- **`KT % S != 0`**: the outer loop covers only whole groups, and the remaining beats need a tail after the loop (stage starting from 0, phase carrying on); this section's probe ran only the divisible case;
  the spelling with a tail is numerically certified by the probe of the section "Ring depth not dividing the loads per round" below.
- **NU as a runtime variable** leaves a branch before every MMA (an early version of the probe did exactly that: extra compares and jumps between MMAs in SASS); make NU a compile-time constant.
- **Always accept by SASS**: count the instructions from `try_wait` to the first `UTC*MMA`, the instructions between MMAs, `R2UR` and `LDL` / `STL`; moving source positions may be reordered back by ptxas.
- This entry changes only the MMA warp's loop shape; producer, barrier counts, TMEM layout and readback all stay the same.

## Ring depth not dividing the loads per round: unroll by lcm

In the spelling above one beat consumes one slot. Another common shape: each round of the outer loop consumes **P loads** (each load takes one slot of the ring, a pair of full / empty), and load j lands in slot `j % S`, phase `(j / S) & 1`:

- **S divides P**: after unrolling the inner loop by P, `j % S` depends only on the inner index and is a compile-time constant; the phase is one bit of the outer count;
- **S does not divide P** (S not a power of 2) and the slot number is computed from j: `j % S`, `j / S` depend on the outer count, so **every load needs one division by a constant**
  (SASS: `IMAD.HI.U32` + shift + multiply back), and the slot base needs another multiplication by the slot byte size; this integer string sits on the "full wait returns → MMA → commit" issue chain.

Having the MMA thread advance (s, ph) with a running counter (see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) gotcha 10) likewise has no `IMAD.HI` and does not grow the code; this section's probe did not compare against that spelling; the table below compares only "computed from j" with unrolling by lcm.

Spelling: take **LC = lcm(S, P) loads** (G = LC / P outer rounds) as one unrolled period. Within a period, the slot number `i % S` and phase offset `(i / S) & 1` of the i-th load
are compile-time constants; at the end of each period flip the phase base `(LC / S) & 1` times; compute each slot's descriptor low word outside the loop into a table accessed only with constant indices;
the period count R = KT / G and remainder rem = KT % G are computed on the host and passed in; the remaining rem·P loads are covered by an unrolled tail guarded per load.
Worked examples (not the probe geometry): S = 3, P = 2 → LC = 6, G = 3, each slot used LC / S = 2 times per period, phase base not flipped;
S = 2, P = 3 → LC = 6, LC / S = 3, phase base flipped once per period.

```cpp
// Two rings (A, B), both of depth S; each load = one SW128 K atom -> NU K16 MMAs; each load commits once to both empties.
// mbar_wait is a hint-free try_wait loop; for the bit fields of umma(d, alo, blo, acc) see umma_raw_desc_issue.
constexpr int gcd_c(int a, int b) { return b ? gcd_c(b, a % b) : a; }
constexpr int lcm_c(int a, int b) { return a / gcd_c(a, b) * b; }

template <int S, int P, int NU>
__device__ void mma_thread(uint32_t tmem, uint32_t a_lo0, uint32_t b_lo0,      // descriptor low words of slot 0
                           uint32_t a_slot_bytes, uint32_t b_slot_bytes,       // slot byte sizes (may be runtime values)
                           uint32_t aF, uint32_t aE, uint32_t bF, uint32_t bE, // full / empty bases (8 B per slot)
                           int R, int rem, uint32_t acc_bar) {                  // R = KT / G, rem = KT % G: computed on the host
  constexpr int LC = lcm_c(S, P), G = LC / P;
  constexpr uint32_t FLIP = (LC / S) & 1;
  uint32_t alo[S], blo[S];                                // accessed only with compile-time indices -> stays in registers
#pragma unroll
  for (int s = 0; s < S; ++s) { alo[s] = a_lo0 + ((s * a_slot_bytes) >> 4); blo[s] = b_lo0 + ((s * b_slot_bytes) >> 4); }

  auto step = [&](int s, uint32_t ph, bool first) {      // after unrolling, s and the offset of ph are both constants
    mbar_wait(aF + 8 * s, ph);
    mbar_wait(bF + 8 * s, ph);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#pragma unroll
    for (int kk = 0; kk < NU; ++kk)                       // K16 step: descriptor low word +2
      umma(tmem, alo[s] + 2 * kk, blo[s] + 2 * kk, (first && kk == 0) ? 0u : 1u);
    umma_commit(aE + 8 * s);                              // right after this load's last MMA
    umma_commit(bE + 8 * s);
  };

  uint32_t phb = 0;
#pragma unroll 1
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int i = 0; i < LC; ++i) step(i % S, phb ^ ((i / S) & 1), r == 0 && i == 0);
    phb ^= FLIP;
  }
#pragma unroll
  for (int i = 0; i < (G - 1) * P; ++i)                    // tail: rem rounds = rem * P loads, guarded per load
    if (i < rem * P) step(i % S, phb ^ ((i / S) & 1), R == 0 && i == 0);
  umma_commit(acc_bar);
}
// The spelling it replaces: for (k) { #pragma unroll for (p < P) { j = P*k + p; s = j % S; ph = (j / S) & 1; ... a_lo0 + ((s * a_slot_bytes) >> 4) ... } }
```

**Probe readings (single CTA, two rings of the same depth, NU MMAs per load; one ring depth with S ∤ P and one with S | P, two N each;
"non-wait cycles" = the MMA thread's whole-loop cycles minus full-wait cycles, averaged per load; median of 7 launches; single run in one job, no noise floor)**:

| Spelling | `IMAD.HI` | MMA sites | Non-wait cycles per load (two N) | Wait cycles per load | launch µs (two N) |
|---|---|---|---|---|---|
| S ∤ P, runtime `% S` (control) | 4 | P·NU | 353.1 / 357.3 | 104.5 / 104.1 | 40.38 / 29.38 |
| S ∤ P, unrolled by lcm | 0 | (LC + (G − 1)·P)·NU | **325.1 / 325.5** | 94.5 / 92.0 | 38.69 / 27.55 |
| S ∤ P, unrolled + next load's full wait moved before this load's commit | 0 | (LC + (G − 1)·P)·NU | 332.3 / 332.5 | 97.4 / 93.4 | 38.94 / 28.19 |
| S \| P, runtime `% S` | 0 | P·NU | 331.8 / 331.2 | 98.9 / 97.7 | 39.04 / 28.19 |
| S \| P, unrolled by period (LC = P) | 0 | P·NU | 326.3 / 326.5 | 94.8 / 91.6 | 38.24 / 27.52 |
| S \| P, unrolled + wait moved before the commit | 0 | P·NU | 337.2 / 337.2 | 97.6 / 94.7 | 39.20 / 28.35 |

How to read it:

- The S ∤ P runtime spelling has 21–26 more non-wait cycles per load than the S | P runtime spelling (+6.4% / +7.9%); unrolled by lcm, `IMAD.HI` 4 → 0,
  and the non-wait cycles return to the level of the S | P unrolled spelling (325 vs 326).
- With S | P, unrolling by period also saves about 5 cycles (−1.4% / −1.7%); the source was not isolated (inference: the descriptor table outside the loop replaces one slot-base multiplication per load).
- **Moving the next load's full wait before this load's commit**: all four instances got slower (7–11 more non-wait cycles than the unrolled spelling at the same ring depth, launch µs 0.6%–3.0% slower;
  with S | P even slower than the non-unrolled runtime spelling). MMA is asynchronous, so the wait already overlaps with in-flight MMAs; moving it only delays the commit's issue until after the wait returns
  (for when a commit lands see [`tcgen05_commit_slot_release`](tcgen05_commit_slot_release.md)). **The commit goes right after this load's last MMA.**
- launch µs wraps the whole call (including tensor map encoding and launch) in host-side events; it is only for side-by-side comparison, not kernel time.

**Code size**: MMA site count = (LC + (G − 1)·P)·NU; in the probe's S ∤ P instance the SASS instruction count and the MMA thread's `R2UR` both grow in proportion to the site count
(`LDL` / `STL` were not counted). When S and P are coprime, LC = S·P and the unrolled volume can be large; compute the site count before deciding.

**Whole-kernel reading (one tree, one landing, no single-variable re-measure)**: after landing, the device geometric mean was 0.0456 ms, within the range of the pre-landing tree's own three readings
(0.0442 / 0.0459 / 0.0465 ms, cross-job); the same-job ratio against a fixed baseline kernel was 1.036, versus 0.9549–1.0007 for the three pre-landing readings.
**Whole kernel not priced, no gain claimed**; that tree was later replaced wholesale for other reasons. Do not use it as a speed-up recipe; before switching, count `IMAD.HI` / `UTC*MMA` sites on your own tree and do a same-job A/B.

Key points and gotchas:

- **The phase base flips by `(LC / S) & 1`, not once per period**: when LC / S is even the phase base does not change at the end of a period; written as "flip once per round", it waits on the wrong phase from the second period on
  (waiting on an already completed phase returns at once and reads data that has not landed, or waiting on a phase that never comes hangs). Within a period the phase offset is `(i / S) & 1`.
- **Do not compute R, rem on the device as `KT / G`**: when G is not a power of 2 that brings back a division (only once outside the loop, but a SASS census will see `IMAD.HI`; untested, inference). Compute on the host and pass them in, or make KT a compile-time constant.
- **The descriptor table may only be accessed with compile-time indices**: any read of `alo[s]` with a runtime index puts the whole table into the stack frame (`LDL`); see [`rmem_array_runtime_index_local_frame`](rmem_array_runtime_index_local_frame.md).
  The address field of the descriptor low word is 14 bits (addr >> 4); table values plus the K step +2 must not carry beyond bit 14.
- **Guard the tail per load** (after unrolling, each site is one compare against the constant rem·P); do not guard per round and then compute `i % S` inside. The probe's instance with a tail (KT % G ≠ 0) is numerically bitwise right.
- A producer advancing stage / phase with an incrementing counter (`if (++s == S) { s = 0; ph ^= 1; }`, see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md)) has no division anyway and need not be unrolled along with it; in the probe the producer stays rolled, mixed with the unrolled consumer, numerically bitwise right. This section's probe uses two rings of the same depth.

## Sources

- PTX ISA: `tcgen05.mma`, `tcgen05.commit`, `mbarrier.try_wait` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
- GPU-certification probe (runtime stage S = 4 / 8, unrolled by S = 4 / 8, each bitwise exact over three launches;
  SASS from the `cuobjdump -sass` census of `run_all.py`, timing as `time` / `res` mode fact lines), verdict
  `PROBE PASS max_diff=0.0`: `A1-exact-v64`, `A1-exact-v64s8`, `A1-exact-v64u4`, `A1-exact-v64u8`; SASS and timing are fact lines of the same job.
- Second GPU-certification probe (section "Ring depth not dividing the loads per round"; single CTA, `cta_group::1`, two TMA rings of the same depth, producer rolled, three MMA-thread spellings:
  runtime `% S`, unrolled by lcm, unrolled + wait moved before the commit): ✅ `PROBE PASS max_diff=0.0`, round 2
  (round 1 FAIL: the probe's own extension module failed to load and never reached the contracts). All three contracts passed in the same round: `L1` (three spellings × two ring depths × two N ×
  with / without tail, integer data bitwise exact), `L2` (MMA site count per spelling), `L3` (no `IMAD.HI` in the unrolled spelling when S ∤ P).
  Cycles, launch µs and SASS counts are fact lines of the same job.
