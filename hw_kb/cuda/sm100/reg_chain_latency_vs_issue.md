# reg_chain_latency_vs_issue (register reduction chains: splitting accumulators vs cutting instructions; packed f32x2 is never generated automatically and never contracted; `fma.rn.f32.bf16` saves the widening)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> What is certified is the **numerics** (8 variants against an fp64 reference) and the **SASS counts** (packed instruction count = the count written in PTX, 0 `LDL/STL`, fewer instructions in the mixed-precision variant).
> The cycle table below is fact readings from the same job (median of 5 launches per setting), **with no significance test**; use it only for direction.
>
> One file per hardware interface. Operator-agnostic: mechanism only.
> This entry answers only: for a register chain "bf16x2 widening → square → in-thread reduction to one scalar", how many cycles does splitting into k independent accumulators / a balanced tree buy back;
> and what nvcc / ptxas do and do not do to fp32 chains.

## Purpose

The cycles of an in-thread reduction chain are set by the **larger** of two things:

- **dependency depth × single dependency latency** (latency-bound): a serial `A = fma(x, x, A)` has depth N; split into k accumulators it is `N/k + log2(k)`;
- **total instructions of all warps on this SMSP in this segment ÷ issue rate** (issue-bound): each SMSP issues at most one warp instruction per cycle,
  and widening (`LOP3` / `SHF`), packed FP, the tail tree and feedback all take issue slots.

Splitting accumulators only lowers the first term, and **adds** instructions (each extra accumulator adds one `FADD2` to the tail tree). In the probe shape below (16 bf16x2 words per thread,
all re-widened every round), even with only 1 warp resident per SMSP the serial chain is already no slower than the split ones — **the lever that actually works is cutting instructions**:
mixed-precision `fma.rn.f32.bf16` consumes bf16 directly, saving two widening instructions per word.

## When needed (trigger)

- You want to shorten a register chain such as "land → square → in-thread reduction → `SHFL`" with multiple accumulators / a balanced tree. **Estimate before building**:
  compare "depth × dependency latency" with "resident warps on this SMSP × SASS instructions in this segment" (count with `cuobjdump -sass`, including widening and `MOV`). If the latter is larger, it is issue-bound.
- The chain has more widening instructions (one `shl` + one `and` per bf16x2 word) than FP instructions: first see whether `fma.rn.f32.bf16` can be used.
- You assume nvcc will automatically pack two independent fp32 chains into `FFMA2`: **it will not** (gotcha 1).
- You wrote a balanced tree of `.rn` packed `mul` + `add` and expect ptxas to contract it into `FFMA2`: **it will not** (gotcha 2).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// (a) mixed-precision FMA: a, b are bf16 (.b16 registers), c, d are f32. SASS = FHFMA (no integer widening).
__device__ __forceinline__ float sumsq_bf16x2_mixed(uint32_t w, float acc) {
  asm("{\n"
      ".reg .b16 h0, h1;\n"
      "mov.b32 {h0, h1}, %1;\n"                  // low half word = the first element
      "fma.rn.f32.bf16 %0, h0, h0, %0;\n"
      "fma.rn.f32.bf16 %0, h1, h1, %0;\n"
      "}\n" : "+f"(acc) : "r"(w));
  return acc;
}

// (b) packed chain: widen into an f32 pair first, then accumulate with fma.rn.f32x2 (.b64 register = a pair of f32).
__device__ __forceinline__ uint64_t sumsq_bf16x2_packed(uint32_t w, uint64_t acc) {
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .b64 p;\n"
      "shl.b32 lo, %1, 16;\n"                    // low half-word bf16 -> f32
      "and.b32 hi, %1, 0xffff0000;\n"            // high half-word bf16 -> f32
      "mov.b64 p, {lo, hi};\n"
      "fma.rn.f32x2 %0, p, p, %0;\n"
      "}\n" : "+l"(acc) : "r"(w));
  return acc;
}

// (c) timing-loop skeleton: the whole loop in [one] asm block, with .pragma "nounroll" and inter-round feedback (z is a runtime 0),
//     otherwise NVVM / ptxas unroll, hoist or overlap rounds, and you measure throughput instead of this chain.
//   asm volatile("{\n .reg .b32 sb, tt, cnt;\n .reg .pred q;\n"
//                "mov.u64 %t0, %%clock64;\n mov.u32 cnt, 0;\n"
//                "RC_LOOP:\n .pragma \"nounroll\";\n"
//                "  ...chain body, result lands in sres...\n"
//                "  mov.b32 sb, sres;\n and.b32 tt, sb, %z;\n xor.b32 %w0, %w0, tt; ...(each input word)\n"
//                "  add.u32 cnt, cnt, 1;\n setp.lt.u32 q, cnt, %reps;\n @q bra RC_LOOP;\n"
//                "mov.u64 %t1, %%clock64;\n}\n" : ...);
```

Probe readings (B200, 1 CTA, 1/2/4/8 warps per SMSP, 16 bf16x2 words = 32 elements per round, cycles/round):

| Variant | SASS packed count (= PTX) | 1 warp | 2 | 4 | 8 | Registers |
|---|---|---|---|---|---|---|
| packed serial (1 `FMUL2` + 15 `FFMA2`) | 16 | 137 | 202 | 370 | 713 | 28 |
| packed 2 accumulators | 17 | 158 | 202 | 338 | 647 | 28 |
| packed 4 accumulators | 19 | 156 | 221 | 403 | 689 | 32 |
| packed 8 accumulators | 23 | 146 | 217 | 384 | 678 | 40 |
| packed full tree (16 `FMUL2` + 15 `FADD2`) | 31 | 159 | 236 | 411 | 741 | 40 |
| scalar serial (31 `FFMA`) | 0 | 179 | 209 | 358 | 644 | 26 |
| scalar 8 accumulators | 0 | 162 | 223 | 395 | 683 | 30 |
| **mixed precision 8 accumulators (32 `FHFMA`)** | 0 | **97** | **149** | **285** | **510** | 30 |

The mixed-precision variant's whole kernel has 112 SASS instructions (17 `LOP3`), packed serial 120 (33 `LOP3`) — what is saved is exactly the widening.

## Gotchas

1. **nvcc does not generate packed f32x2 automatically.** Pure C++ written as `fmaf` on each of two `float2` fields, or as 8 scalar accumulators,
   gives 32 `FFMA` and 0 `FFMA2` in SASS under `-O3 --use_fast_math -arch=sm_100a`. To get `FFMA2` / `FMUL2` / `FADD2` you must write
   inline PTX (`fma.rn.f32x2` etc.) or float2 intrinsics (`__ffma2_rn` etc., see [`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md)).
2. **ptxas does not reassociate, nor contract `.rn` packed mul + add.** The packed instruction counts in SASS equal the PTX one for one (16 / 17 / 19 / 23 / 31):
   the full-tree variant writes 16 `mul.rn.f32x2` + 15 `add.rn.f32x2` in PTX, and SASS is exactly 16 `FMUL2` + 15 `FADD2`, not a single `FFMA2`.
   If you want fused multiply-add, **write `fma.rn.f32x2` directly**. (This agrees with the PTX ISA rule for an explicit `.rn`: mul / add with a rounding suffix do not take part in contraction.)
3. **Do the accounting before splitting accumulators.** In this shape, at 1 warp/SMSP serial is 137 cycles, 8 accumulators 146, the full tree 159 — dropping depth from 16 to ~5 is actually slower,
   because issuing about a hundred instructions per round is the bottleneck; at 4 / 8 warps 2 accumulators are slightly faster (338 vs 370, 647 vs 713), and more ways get slower again.
   The direction changes with the shape; **measure in the same job; do not split by default**.
4. **Cutting instructions beats reducing depth.** `fma.rn.f32.bf16` turns "2 widening + FMA" into "FMA": 97 vs 137 at 1 warp (about −29%), 510 vs 713 at 8 warps.
   The operands are `.b16` registers: split the word with `mov.b32 {h0, h1}, w`, the low half word being the first element; the accumulator is f32; it needs sm_100 and a PTX ISA version that supports the instruction.
5. **The timing loop must be one asm block.** The loop, the `%clock64` reads, `.pragma "nounroll"` and the inter-round feedback (`bits(s) & z`, z a 0 passed in at runtime, unknown at compile time)
   all sit in the same block. With two `clock64()` calls wrapped around a short chain at the C level, ptxas reorders the clock reads with unrelated ALU instructions; without feedback,
   ptxas can hoist or overlap the independent parts of different rounds.
6. **Reading a special register in inline asm needs `%%clock64`** (`%` must be escaped); put labels inside the block in a `{ }` scope so they do not clash when the same function is inlined several times.

## Sources

- PTX ISA: *Floating Point Instructions: `fma` / `mul` / `add`* (`.f32x2` forms, from sm_100; mixed-precision `fma.rn.f32.bf16`),
  *Special Registers: `%clock64`*, *Directives: `.pragma "nounroll"`* — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA C++ Programming Guide: inline PTX (`asm volatile`, constraint letters `r` / `l` / `f`) —
  https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
- GPU-certified by the probe (main module + mixed-precision bf16 module):
  `PROBE PASS max_diff=1.171e-07 reg_chain/numeric` (8 variants, relative error against fp64 ≤ 1.2e-7);
  `PROBE PASS max_diff=n/a reg_chain/sass_eq_ptx sass_packed={serial 16, acc2 17, acc4 19, acc8 23, tree 31} ldl_stl=0`;
  `PROBE PASS max_diff=9.007e-08 reg_chain/mixed_fewer instrs_mixed=112 instrs_packed_serial=120 cyc1_mixed=97.0 cyc1_serial=137.2`.
  Fact lines: `cpp_autopack` both pure C++ kernels `FFMA2=0 FFMA=32`; cycle table above.
