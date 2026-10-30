# umma_issue_form (how a single-thread issue role picks its one thread: `lane == 0` branch vs `elect.sync` — ELECT / R2UR back jumps in SASS and the issue cadence)

> **Certification source: B200 GPU-certified in CUDA mode, two probes (the probes are not included in this release).**
> probe A (UMMA issue form): `PROBE PASS max_diff=0.0`
> (contracts B0-build / D1-form-data / T1-converged-cad / S1-sass all PASS);
> probe B (elect one): `PROBE PASS max_diff=0.0`
> (contracts elect / gemm_m0 / gemm_m1 / gemm_m2 all PASS).
> Certification covers: single-CTA `cta_group::1`, `kind::f16`, UMMA issue loops of one K16 per instruction and 1D bulk copy producers; numerics, SASS footprint and timing of several "pick one thread" spellings.
> **Uncertified**: issue spellings for the `cta_group::2` shape (used only in whole kernels, see "Measured readings"); a same-probe comparison with the "one `elect_one()` region per k-block" spelling.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the semantics of `elect.sync` itself (warp scope, outer predicate, which instructions may / may not be wrapped in an election) see [`elect_one`](elect_one.md);
> for the issue cost of one election region per k-block, and readings on a pair for "per-beat elect region / single-thread loop / single thread + unrolled by S", see [`umma_mainloop_wrap_segment_issue_floor`](umma_mainloop_wrap_segment_issue_floor.md).

## Purpose

`tcgen05.mma`, `tcgen05.commit` and `cp.async.bulk(.tensor)` are per-thread instructions in PTX: one executing thread issues one. So "let only one thread issue"
can be written several ways that compute identical bytes but differ in SASS and issue cadence. In SASS they are uniform-datapath instructions (`UTCHMMA` / `UBLKCP` / `UTMALDG`,
operands in `UR*`); measured: when the issue is wrapped in a lane-level branch such as `lane == 0`, `ELECT` + `R2UR` and **a `BRA` jumping backward** appear around the issue instructions;
switching to `elect.sync` once and then running the loop ((c)) takes the back jumps within the MMA range from 3 to 0 (probe B). The SASS of (d) is actually longer (see measured readings); judge the spellings by timing.
This entry's two probes cover `UTCHMMA` and `UBLKCP`, not `UTMALDG` (whole-kernel readings for switching a tensor-TMA producer's spelling are in gotcha 6).

Spellings compared in this entry:

| Label | Spelling |
|---|---|
| (a) | `if (warp == W && lane == 0) { the whole issue loop }` |
| (b) | `if (warp == W) { if (lane == 0) { the whole issue loop } }` (outer warp-uniform, inner still a lane branch) |
| (c) | `if (warp == W) { if (elect_one()) { the whole issue loop } }` (elect once; the selected thread runs the whole loop) |
| (d) | `if (warp == W) { the whole warp runs the issue loop }`, with `elect.sync` written inside the asm of each `tcgen05.mma` / `tcgen05.commit` as a guard (`@e`) |
| (e) | `if (warp == W) { the whole warp runs the issue loop }`, with a C++-level `if (elect_one()) mma(...)` before each MMA |

## When needed (trigger)

- A single-thread role (MMA issuer, TMA / bulk copy producer) is written as a `lane == 0` branch (including `lane == 0` nested inside a warp-uniform branch).
- SASS: `UTCHMMA` / `UTCQMMA` / `UBLKCP` / `UTMALDG` closely flanked by `ELECT`, `R2UR`, `PLOP3`, with a backward `BRA` between the first and last MMA;
  far more instructions in the MMA range than MMAs (the (d) spelling has these footprints too, so SASS alone cannot decide).
- On the timeline, the MMA issue role's per-beat issue segment (wait returns → next wait starts) is far longer than the tensor execution time of that beat's UMMAs.

**Anti-trigger: if it is already an `elect.sync` spelling, do not swap between elect spellings.** All of this entry's positive whole-kernel readings start from the lane branch (a);
on another `cta_group::2` tcgen05 GEMM kernel, switching from one elect spelling to another made the whole kernel slower in the same job:
- (c) (both producer and MMA issuer as `if (elect_one()) { the whole loop }`) → elect once outside the loop, the whole warp enters the loop, issue written as an asm `@p` predicate (≈ (d), with the election done only once):
  geometric mean over 4 M 9.480 → 9.583 µs (1.09% slower, `cta_group::2` kernel, same job); its standalone probe (1D bulk copy producer + async writeback issuer)
  likewise per launch: divergent single lane 119.05 µs, whole warp + asm `@p` 135.38 µs, whole warp + C++ `if` 146.69 µs
  (the producer's per-beat cycles in the two whole-warp shapes are actually 9–11% shorter at B = 256 and 5–7% longer at B = 64; go by wall-clock of the whole launch).

## Minimal runnable example

```cpp
// Spelling (c): split by warp on the outside (warp-uniform), elect.sync once on the inside; the selected thread runs the whole loop
__device__ __forceinline__ bool elect_one() {
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "elect.sync _|p, 0xffffffff;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t}" : "=r"(pred));
    return pred != 0;
}

if (warp == MMA_WARP) {                         // the condition is uniform across the whole warp
    if (elect_one()) {
        for (int k = 0; k < KB; ++k) {
            /* mbar_wait(full[s]) -> tcgen05.fence::after_thread_sync -> NU tcgen05.mma -> tcgen05.commit(empty[s]) */
        }
    }
    __syncwarp();                               // the other 31 lanes wait here for the selected lane to come back
}

// Counter-examples (a) / (b): numerics right, but every MMA / copy is surrounded by ELECT + R2UR + back jump
//   if (warp == MMA_WARP && lane == 0) { for (...) { ... } }
//   if (warp == MMA_WARP) { if (lane == 0) { for (...) { ... } } }
```

## Measured readings

**probe B** (single CTA, M = 128, N = 64, a 4-stage 1D bulk copy ring; the two single-thread roles, producer and MMA issuer, dispatched together as (a) / (b) / (c); K = 16384, grid = 148):

| Spelling | Device time | Relative to (a) | `ELECT` | Instructions in the MMA range | Backward `BRA` in the MMA range |
|---|---|---|---|---|---|
| (a) | 101.54 µs | 1.000 | 9 | 46 | 3 |
| (b) | 101.02 µs | 0.995 | 9 | 46 | 3 |
| (c) | 75.89 µs | **0.747** | 6 | 6 | 0 |

(a) and (b) have exactly the same SASS counts: moving `lane == 0` inside a warp-uniform branch does nothing; it has to become `elect.sync`.

**probe A** (single CTA, M = 128, a 4-slot self-throttled ring, 4 K16 MMAs + 1 commit per slot, the issuing thread timing cycles per UMMA with `%clock64`):

probe A's lane-branch shape is (b) in the source (probe B showed (a) / (b) have identical SASS), recorded in the table as (a)/(b):

| Spelling | N = 64: cycles / UMMA | Relative to (a)/(b) | N = 256: cycles / UMMA | Relative to (a)/(b) |
|---|---|---|---|---|
| (a)/(b) | 124.8 | 1.000 | 127.9 | 1.000 |
| (d) | 69.6 | **0.558** | 127.9 | 1.000 |
| (e) | 145.1 | 1.162 | 145.1 | 1.135 |

SASS at N = 64: (b) 4 `UTCHMMA`, issue span 49 instructions, `R2UR` 3, 3 backward jumps in the span; (d) 8 `UTCHMMA` (the predicated version duplicated), span 256, `R2UR` 24,
10 `BRA` in the span; (e) 4, span 64, with `BSSY` / `BSYNC`.

How to read it:
- (c) and (d) are both faster than the lane branch; (e) (one C++-level election branch per MMA) is even slower than the lane branch. (c) and (d) were not compared in this entry's probes;
  on another kernel, switching from (c) to "elect once + whole-warp `@p` issue" made the whole kernel 1.09% slower (see "Anti-trigger").
- At N = 256 the tensor execution time of each UMMA covers the issue overhead, and (a) and (d) are flat: switching spellings pays only when the issue segment is the bottleneck.
- (d) has more SASS instructions, `R2UR` and `BRA` than (a)/(b), yet is faster: **you cannot judge which spelling is better by SASS counts**; go by timing.
- Whole kernel in a real kernel (single-variable change, gate device window): on a `cta_group::2` TMA → UMMA kernel, switching the MMA issue from (a) to (d) (with the elect version of the `.multicast::cluster` commit),
  applied to two kernels in turn, gate 0.0196 → 0.0193 ms (−1.5%) and 0.0193 → 0.0189 ms (−2.1%); on another `cta_group::2` kernel, switching the MMA issue from (a) to (c),
  0.0111 → 0.0098 ms (−11.7%, still 0.0098 on one re-measure).

## Gotchas

1. **Every lane in the `elect.sync` membermask must execute it.** Do not execute `elect.sync` (`elect_one()` or an elect inside asm) inside `if (lane == 0)`; when switching spellings, **delete the
   lane branch entirely**, do not just swap the instruction for its elect version.
2. **In (d) the MMA and the commit must land on the same lane.** `tcgen05.commit` tracks only the tcgen05 async operations previously issued by the thread executing it. In (d) the MMA and the commit each do their own
   `elect.sync`, relying on "within the same warp and the same full mask, the same lane is selected every time" — measured on this machine the selected lane is always the lowest in the mask (see [`elect_one`](elect_one.md) gotcha 3),
   and probe A's (d) is numerically bitwise right; but the PTX ISA promises only "exactly one". If you do not want to rely on that, use (c): one election, so MMA and commit are naturally the same thread.
3. **Missing `@e` in (d) = every lane issues its own MMA.** probe A's negative control (MMA without the predicate, commit still with it): result / expectation median 30.989, minimum −247, maximum 342 —
   the extra MMAs are not tracked by the commit, and the readout timing is scrambled too. A commit without the predicate makes all 32 lanes arrive on a count-1 mbarrier, advancing the phase several steps at once.
4. **Numerics cannot verify whether the spelling is right.** The results of (a)–(e) are bitwise identical (probe A's D1, probe B's gemm_m0/m1/m2, max_diff = 0); look at SASS (`ELECT` / `R2UR` / back jumps in the MMA range)
   and timing.
5. **After (c), `__syncwarp()`** to bring the other lanes back before executing any `.sync.aligned` instruction (`tcgen05.ld`, `tcgen05.alloc` / `dealloc`, `barrier.cluster.*.aligned`).
6. **Measure the producer side separately.** probe B's three spellings apply to both the producer and the MMA issuer at once; this entry's positive whole-kernel readings cover only changes to the MMA issuer. Switching a producer's (TMA / bulk copy) spelling needs a same-job A/B on your own kernel:
   on a single-CTA tcgen05 GEMM, switching the tensor TMA producer from `lane == 0` to the elect spelling twice removed the `ELECT` between `UTMALDG`s in the whole-kernel SASS,
   yet the gate ratio was slightly slower both times (−1.3% / −0.7%, cross-job, at the edge of noise). This entry's probes do not cover `UTMALDG`; do not use this entry as the basis for changing a producer.

## Sources

- PTX ISA: `elect.sync` (membermask, predicate destination operand), `tcgen05.mma` (single-thread issue, `cta_group`, accumulate predicate), `tcgen05.commit` (tracks the executing thread's earlier
  tcgen05 async operations, `mbarrier::arrive::one`, `.multicast::cluster`), `cp.async.bulk` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Public CUTLASS headers: `cute::elect_one_sync()` (`include/cute/arch/cluster_sm90.hpp`, the C++-level spelling returning bool).
- GPU-certified in this repo:
  - probe A: `PROBE PASS max_diff=n/a` (B0-build), `PROBE PASS max_diff=0` (D1-form-data),
    `PROBE PASS max_diff=n/a` (T1-converged-cad), `PROBE PASS max_diff=n/a` (S1-sass); cycles and SASS counts are same-job fact lines.
  - probe B: `PROBE PASS max_diff=n/a` (elect), `PROBE PASS max_diff=0` (gemm_m0 / gemm_m1 / gemm_m2);
    device times and SASS counts are same-job fact lines.
- Whole-kernel readings in "Anti-trigger": fact lines of a same-job A/B on another kernel and of a standalone probe; the producer whole-kernel readings in gotcha 6 are cross-job gate ratios
  (the KB candidates behind these readings were not taken in).
