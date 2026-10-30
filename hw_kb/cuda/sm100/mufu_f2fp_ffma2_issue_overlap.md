# mufu_f2fp_ffma2_issue_overlap (issue cycles, dependency latency and pairwise mixed-issue overlap of the three instruction classes `MUFU.EX2`, `F2FP` packing and `FFMA2` at 1 / 2 warps per SMSP)

> **Certification source: B200 GPU-certified in CUDA mode (certified by a GPU probe (the probe is not included in this release), PROBE PASS max_diff=2.12e-05 (the contract max_diff comes from a class this entry does not include; this entry's three classes: MUFU 4.36e-08, F2FP / FFMA2 0); contracts B0 build / N1 numeric chains / N2 bitwise packing / T1 timing collection / S1 SASS all PASS).**
> What is certified is the numerics, the loop-body SASS and every cycle reading below (1 CTA, median of 7 launches per setting, same job).
> **Uncertified**: pipe attribution ("which two classes share a pipe" is an interpretation of the overlap; this entry did not certify it separately with NCU per-pipe counters); any whole-kernel gain (whole kernel not measured, see the last item of "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. For how `cvt.rn.bf16x2.f32` feeds `tcgen05.st` see
> [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md); for packed f32x2 spellings see
> [`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md); for telling issue-bound from latency-bound see
> [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md).

## Purpose

An element-wise exp2 → bf16x2 packing segment has three instruction classes: `ex2.approx.ftz.f32` (SASS `MUFU.EX2`), `cvt.rn.bf16x2.f32` (SASS `F2FP.BF16.F32.PACK_AB`),
and packed scaling `fma.rn.ftz.f32x2` (SASS `FFMA2`). "Segment floor = MUFU count × cycles each" holds only if the other two classes neither compete with MUFU for the same pipe nor saturate the issue slots.
This entry measures three things with a standalone microbenchmark:

1. each class's **issue cycles per instruction** (16 mutually independent chains) and **dependency-chain latency** (1 chain of length 16) at 1 and 2 warps per SMSP;
2. whether mixing two classes gives "the sum" (overlap 0: same pipe or issue-bound) or "the max" (overlap 1: no interference);
3. how far a segment mixed as "2 `MUFU.EX2` : 1 `F2FP` : 1 `FFMA2` per element pair" measures from "MUFU only".

## When needed (trigger)

- An element-wise exp2 segment is busy far longer than the paper floor "`MUFU.EX2` count × cycles each", the main stall is `wait` (fixed-latency dependency), and no pipe reports saturation —
  you want to know whether the gap is "MUFU queueing", "packing / scaling competing with MUFU for a pipe", or "latency not covered". First multiply the segment's SASS histogram class by class with this entry's per-instruction cycles.
- You want to move the `F2FP` packing or `FFMA2` scaling away / change their spelling to "free up the pipe for MUFU": look at the overlap first — on this probe `F2FP` and `MUFU.EX2` mixed have overlap 0.98;
  packing does not occupy MUFU's pipe.
- **This entry gives only the ledger, not a lever.** None of its readings were measured on a whole kernel; changes made according to this ledger in a real kernel were all not faster on the whole kernel in the same job:
  replacing the `F2FP` packing with integer-ALU spellings (four rounding / coverage variants) was 2.46%–21.68% slower, eight variants that changed the issue order within the segment etc. +0.48%–+2.24% (same-job repeated controls +0.29% / +0.47%).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// The three instruction classes under test (PTX spelling)
__device__ __forceinline__ float ex2a(float x) {                  // MUFU.EX2
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ uint32_t cvt_pk(float hi, float lo) {  // F2FP.BF16.F32.PACK_AB; the first source goes to the high half
  uint32_t d; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo)); return d;
}
__device__ __forceinline__ uint64_t ffma2(uint64_t a, uint64_t b, uint64_t c) {  // FFMA2
  uint64_t d; asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d;
}
__device__ __forceinline__ uint64_t f2pk(float a, float b) {
  uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r;
}

// Method: 1 CTA; blockDim = 128 -> 1 warp per SMSP (warp w lands on SMSP w % 4), blockDim = 256 -> 2 warps per SMSP.
// NCH mutually independent chains per iteration (measures issue / throughput), or pin the index to 0 to make 1 dependency chain of length NCH (measures latency);
// enabling two classes at once = mixed issue. Each chain's result feeds back into itself (a true dependency, cannot be hoisted out of the loop).
template <bool kMu, bool kCv, bool kF2, bool kLat>
__global__ void __launch_bounds__(256, 1) pipe_k(const float* init, float* out, int64_t* cyc, int iters) {
  constexpr int NCH = 16;
  float x[NCH], p[NCH]; uint64_t v[NCH];
  /* read initial values from init into x / p / v (omitted) */
  const uint64_t k2 = f2pk(0.5f, 0.5f), c2 = f2pk(0.25f, 0.25f);   // v <- 0.5*v + 0.25, converges to 0.5
  __syncthreads();
  const long long t0 = clock64();                       // SASS: CS2R Rx, SR_CLOCKLO
#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < NCH; ++i) {
      const int j = kLat ? 0 : i;
      if (kMu) x[j] = ex2a(-x[j]);                      // 2^-x converges to a fixed point, so numerics can be compared (the cost of negating: gotcha 1)
      if (kCv) p[j] = __uint_as_float(cvt_pk(p[j], 1.0f));
      if (kF2) v[j] = ffma2(v[j], k2, c2);
    }
  }
  const long long t1 = clock64();
  if ((threadIdx.x & 31) == 0) cyc[threadIdx.x / 32] = t1 - t0;   // cycles per warp / iters = cycles per iteration
  /* write x / p / v back to out (omitted) */
}
// overlap = (T_a + T_b - T_ab) / (T_a + T_b - max(T_a, T_b)): 0 = additive, 1 = fully parallel.
// Accept by SASS: in the loop body between the two SR_CLOCK reads, the target instruction count == NCH x instructions per iteration, no STL / LDL.
```

## Gotchas

1. **The probe's MUFU reading is an "`FADD.FTZ` + `MUFU.EX2`" pair.** This spelling negates the input (`"f"(-x)`), so every `MUFU.EX2` in the loop body is preceded by an `FADD.FTZ`
   (MUFU-mode loop body 39 instructions = 16 MUFU + 16 FADD + 7 loop overhead). In another certification probe the input of `ex2` comes directly from `FFMA2` without negation,
   and the segment has one `MUFU.EX2` per element and 0 scalar `FADD` — this `FADD` is brought in by the negation (inference, not isolated separately). Count by your own SASS; do not assume every MUFU comes with an FADD.
2. **At a single warp per SMSP every instruction takes at least one issue slot.** Loop overhead, `MOV` and `IMAD.MOV` all count. Compute the paper floor class by class from the SASS histogram; do not count only MUFU.
3. **When `FFMA2` is mixed with another class, the compiler inserts 64-bit register-pair moves.** This probe: FFMA2 issued alone has `MOV` ×19 + `IMAD.MOV` ×5 in the loop body (16 FFMA2),
   mixed with MUFU `MOV` ×24, mixed with `F2FP` `MOV` ×12 + `IMAD.MOV` ×13; only the segment mixed 2 : 1 : 1 is left with just `MOV` ×4.
   So FFMA2's "issue cycles per instruction" in the table below include these moves; it is not FFMA2's own throughput.
4. **`clock64` measures the cycles this warp itself sees**: only 1 CTA with 128 threads guarantees one warp per SMSP;
   multiple CTAs on the same SM put other warps on the same SMSP, and the reading becomes shared throughput.
5. **Two fact lines on `cvt.rn.bf16x2.f32`** (N2 section of the same job): subnormal inputs are not flushed to zero, and after RN rounding are bitwise identical to torch's bf16 conversion
   (e.g. input bit pattern `0x00018000` → bf16 `0x0002`); any NaN input outputs the canonical NaN `0x7FFF`. For the half order (the first source goes to the high half) see `cvt_f32x2_bf16x2_pack_into_r2t_fragment`.

## Readings (B200, SM clock locked at 1500 MHz; cycles / iteration, 16 per class per iteration, median of 7 launches)

**Single class** ("each" = cycles / iteration ÷ 16; at 2 warps / SMSP it is the value each warp sees; the two warps share one SMSP, so per SMSP per instruction ≈ half of it):

| Class | 1 warp / SMSP issue | 2 warps / SMSP issue (per warp) | Dependency-chain latency (1 / 2 warps) | Loop-body SASS (issue mode) |
|---|---|---|---|---|
| `MUFU.EX2` (+ negating `FADD`, gotcha 1) | 154.4 (9.65 / instr) | 276.0 (17.25 / instr) | 23.03 / 23.03 per hop | 39: MUFU ×16, FADD.FTZ ×16 |
| `F2FP.BF16.F32.PACK_AB` | 42.5 (2.65 / instr) | 64.2 (4.01 / instr) | 4.47 / 4.52 per hop | 23: F2FP ×16 |
| `FFMA2` (moves included, gotcha 3) | 61.6 (3.85 / instr) | 102.6 (6.41 / instr) | 4.47 / 4.53 per hop | 48: FFMA2 ×16, MOV ×19, IMAD.MOV ×5 |

**Pairwise mixed issue** (overlap formula on the last line of the example):

| Mix | 1 warp / SMSP: measured / sum of the two / larger → overlap | 2 warps / SMSP: measured / sum / larger → overlap | Loop-body instructions (1 warp) |
|---|---|---|---|
| MUFU + F2FP | 155.4 / 196.9 / 154.4 → **0.98** | 277.0 / 340.2 / 276.0 → **0.98** | 55 |
| MUFU + FFMA2 | 164.6 / 216.0 / 154.4 → 0.84 | 270.7 / 378.6 / 276.0 → 1.05 | 80 (MOV ×24) |
| F2FP + FFMA2 | 87.8 / 104.0 / 61.6 → **0.38** | 126.2 / 166.8 / 102.6 → 0.63 | 64 (MOV ×12, IMAD.MOV ×13) |

**Mixed segment** (per iteration 16 MUFU + 8 F2FP + 8 FFMA2, i.e. 2 : 1 : 1 per element pair; loop body 60 instructions: MUFU ×16, FADD.FTZ ×16, FFMA2 ×8, F2FP ×8, MOV ×4):

| | 1 warp / SMSP | 2 warps / SMSP |
|---|---|---|
| measured | 156.4 | 278.0 |
| MUFU only | 154.4 | 276.0 |
| sum of the classes alone (MUFU + F2FP/2 + FFMA2/2) | 206.5 | 359.4 |

What the readings show (interpretation; pipe attribution not certified with NCU):

- `F2FP` packing mixed with `MUFU.EX2` overlaps almost completely (0.98, at both residencies): packing does not compete with MUFU for a pipe.
- The 2 : 1 : 1 mixed segment is 2 cycles / iteration slower than MUFU only (1.3% / 0.7%), i.e. when mixed 2 : 1 : 1, F2FP and FFMA2 both hide under MUFU;
  in this microbenchmark the segment floor is MUFU's issue cycles; if a real segment is clearly slower than that, this entry cannot tell why; measure separately.
- The overlap between `F2FP` and `FFMA2` is only 0.38 / 0.63: one of them is queueing behind the other (or the 64-instruction loop body is already near the single-warp issue limit); mixed together they cannot be treated as free.
- Dependency-chain latency is independent of the number of resident warps (23.03 / 4.47 identical in both settings); at 2 warps / SMSP the single-class cycles each warp sees become 1.5–1.8× (MUFU 1.79×, FFMA2 1.67×, F2FP 1.51×) — MUFU / FFMA2 are already near issue- or pipe-bound at 1 warp, F2FP still has headroom.

## Sources

- PTX ISA: *Floating Point Instructions*: `ex2` (`.approx.ftz.f32`), `fma` (`.f32x2`); *Conversion*: `cvt` (`.rn.bf16x2.f32`);
  *Special Registers*: `%clock64` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide: the *Arithmetic Instructions* throughput table (results per clock per SM, by compute capability) —
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>
- Nsight Compute *Kernel Profiling Guide*: pipelines and warp stall reasons (`wait`, `math_pipe_throttle`) —
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- GPU-certification probe:
  `PROBE PASS max_diff=n/a` (B0-build), `PROBE PASS max_diff=2.120e-05` (N1-chains; that value comes from a class this entry does not include, this entry's three classes MUFU 4.361e-08, F2FP / FFMA2 0), `PROBE PASS max_diff=0.000e+00` (N2-pack),
  `PROBE PASS max_diff=n/a` (T1-timing), `PROBE PASS max_diff=n/a` (S1-sass). All cycles in the tables above are taken from the probe's `timing` / `overlap` / `seg` fact lines.
