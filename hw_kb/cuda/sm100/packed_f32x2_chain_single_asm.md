# packed_f32x2_chain_single_asm (packed f32x2 arithmetic on sm_100a: how to write `fma/mul/add/sub .f32x2` as a chain; three spellings: one asm block, one asm block per instruction, float2 intrinsics)

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: the three spellings are bitwise equal to a scalar `__f*_rn` chain, the packed instruction counts in SASS, the `.rm` rounding suffix, the absence of `max.f32x2`.
> Gotcha 8 (splitting into half words for integer arithmetic) is certified separately by another CUDA probe (all 5 contracts PROBE PASS; this entry uses only two of them, bitwise agreement and SASS counts).
> **Uncertified**: any performance gain (the probe does no timing; instruction counts below; gotcha 8 not measured separately on a whole kernel).
>
> One file per hardware interface. Operator-agnostic: mechanism only. For whether splitting a chain into multiple accumulators pays see [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md).

## Purpose

sm_100a adds `add / sub / mul / fma {.rnd}{.ftz} .f32x2`, which operate on a pair of fp32 (one 64-bit register pair) at once;
in SASS they are `FADD2` / `FMUL2` / `FFMA2`: one issue completes the work of two lanes, halving the FP instruction count. nvcc does not automatically pack scalar fp32 code
into these instructions (see [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md) gotcha 1); to use them you have to write them yourself.

## When needed (trigger)

- Paired fp32 operations in an element-wise / reduction chain take most of the issue slots (lots of `FFMA` / `FMUL` / `FADD` in SASS), and both lanes do the same operation.
- Hesitating while writing a packed chain: one large asm block, one small asm block per instruction, or float2 intrinsics? On this probe all three give bitwise identical results and nearly the same SASS (see gotchas 2, 3).
- Needing a packed `max` / `min`: **there is none** (gotcha 4); a clamp must be split back into scalars.
- After the packed chain's result (a `uint64_t` via `"l"`) comes out, each half word gets integer arithmetic, and SASS shows strings of `SHF.L.U64.HI` / `IADD3.X` / `LOP3.LUT` around that section: gotcha 8.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// 6-step chain (both lanes compute the same): v=fma(v,K1,K2); v=v*K3; v=v+K4; v=fma(v,K3,K1); v=v-K2; v=fma(v,K4,v)
// (a) one asm block: pack once, build the constants inside the block, unpack once
__device__ __forceinline__ void chain_single(float& x, float& y) {
  asm volatile(
      "{\n"
      ".reg .b64 v, k1, k2, k3, k4;\n"
      ".reg .f32 c1, c2, c3, c4;\n"        // one scratch per constant, to avoid false dependencies
      "mov.f32 c1, 0F3FC00000;\n"          // 1.5
      "mov.f32 c2, 0FBE800000;\n"          // -0.25
      "mov.f32 c3, 0F3F400000;\n"          // 0.75
      "mov.f32 c4, 0F40000000;\n"          // 2.0
      "mov.b64 k1, {c1, c1};\n"
      "mov.b64 k2, {c2, c2};\n"
      "mov.b64 k3, {c3, c3};\n"
      "mov.b64 k4, {c4, c4};\n"
      "mov.b64 v, {%0, %1};\n"
      "fma.rn.f32x2 v, v, k1, k2;\n"
      "mul.rn.f32x2 v, v, k3;\n"
      "add.rn.f32x2 v, v, k4;\n"
      "fma.rn.f32x2 v, v, k3, k1;\n"
      "sub.rn.f32x2 v, v, k2;\n"
      "fma.rn.f32x2 v, v, k4, v;\n"
      "mov.b64 {%0, %1}, v;\n"
      "}\n"
      : "+f"(x), "+f"(y));
}

// (b) one asm block per instruction, 64-bit "l" operands crossing the boundaries (SASS identical to (a))
__device__ __forceinline__ uint64_t pk(float a, float b) {
  uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t d; asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c)); return d;
}

// (c) float2 intrinsics (CUDA 12.8+, sm_100): no asm
//   v = __ffma2_rn(v, k1, k2); v = __fmul2_rn(v, k3); v = __fadd2_rn(v, k4); ...
//   rounding variants: __fadd2_rd / __fadd2_ru / __fadd2_rz etc.

// Rounding suffixes: the packed forms also accept .rn / .rm / .rz / .rp (and .ftz)
//   asm("add.rm.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));   // each lane equals __fadd_rd
```

## Gotchas

1. **Operands are 64-bit register pairs.** In inline asm the constraint is `"l"` (`uint64_t`); pack / unpack with `mov.b64 d, {lo, hi}` /
   `mov.b64 {lo, hi}, d` (`lo` is the first lane). A broadcast constant must also first be `mov.f32`'d into an f32 register, then `mov.b64 k, {c, c}`.
2. **Splitting into one asm block per instruction costs nothing extra under CUDA.** In the probe the SASS opcode sequences of (a) and (b) are identical: 32 instructions, 3 `FFMA2` + 1 `FMUL2` + 2 `FADD2`,
   0 `MOV`. ptxas sees through asm block boundaries, and the `mov.b64` packing / unpacking between blocks is eliminated. Choose the spelling for readability.
3. **float2 intrinsics give bitwise identical results, but on this probe 3 more `MOV`s** (40 vs 32 instructions, the same packed instruction count).
   For the tightest SASS use inline PTX; for portability / readability use the intrinsics, then take a look at the SASS.
4. **There is no `max.f32x2` / `min.f32x2`.** ptxas reports `Unexpected instruction types specified for 'max'`.
   A clamp in the middle of the chain must unpack into two scalars, do `fmaxf` / `fminf`, and repack.
5. **`sub.rn.f32x2` exists** and lowers in SASS to `FADD2` (with negation).
6. **Rounding suffixes take effect as usual.** Each lane of `add.rm.f32x2` equals `__fadd_rd` (in the probe it differs from the `.rn` result on 50% of lanes);
   `__fadd2_rd` is also bitwise equal to scalar `__fadd_rd`. When comparing, the `.ftz` must agree on both sides, otherwise lanes with subnormals will not match
   (`--use_fast_math` adds ftz to C++ scalar arithmetic; inline PTX is exactly what you wrote).
7. **Use explicit-rounding intrinsics for the comparison reference** (`__fmaf_rn` / `__fmul_rn` / `__fadd_rn` / `__fsub_rn`): they are never contracted into FMA,
   so bitwise comparison works; a plain `a * b + c` is contracted under `--use_fast_math` / `-fmad=true` and cannot serve as a bitwise reference.
8. **When packed results must be split into two half words for integer arithmetic, split with PTX `mov.b64 {lo, hi}`; do not truncate / shift / or a `uint64_t` in C++.**
   `"l"` operands crossing asm block boundaries are free (gotcha 2), but C++ `(uint32_t)v`, `(uint32_t)(v >> 32)`, `((uint64_t)hi << 32) | lo` are real 64-bit
   integer operations, and ptxas does not guarantee to recognize "this is just renaming registers":

   ```cpp
   __device__ __forceinline__ void u2upk(uint64_t v, uint32_t& lo, uint32_t& hi) {   // the register pair itself, emits no instruction
     asm("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(v));
   }
   // Recommended: uint32_t a0, a1; u2upk(v, a0, a1); do 32-bit arithmetic on each half and pass the results downstream as two 32-bit values, never re-joining into a uint64_t
   // Counter-example: const uint32_t a0 = (uint32_t)v, a1 = (uint32_t)(v >> 32); ... return ((uint64_t)r1 << 32) | r0;
   ```

   Probe: the result of the same packed chain split into two half words, each getting a few 32-bit integer operations before output; the C++ `uint64_t` split / join spelling has
   **4 more SASS instructions per pair** than the PTX `mov.b64 {lo, hi}` split spelling (one more each of `SHF.L.U64.HI`, `IADD3`, `LOP3.LUT`, `IADD3.X`, one fewer `LEA`,
   plus `NOP` / `IMAD` / `LDC` differences); criterion `u64=8 asm1=0 mov32=0` (count of 64-bit glue). The three spellings (C++ `uint64_t`, per-instruction `"l"` blocks + PTX half-word split,
   the whole section in one asm block) give bitwise identical output (`mismatched=0`); the latter two have the same SASS count, so choose by readability.
   The extra instructions all land on integer-ALU issue slots; whether that matters in a whole kernel must be measured in the same job against your own segment ledger (not measured separately on a whole kernel for this entry).

## Sources

- PTX ISA: *Floating Point Instructions: `add` / `sub` / `mul` / `fma`* (`.f32x2`, from sm_100a, `.rnd` / `.ftz` modifiers),
  *Data Movement: `mov`* (vector packing form) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Math API: single precision intrinsics `__ffma2_rn` / `__fmul2_rn` / `__fadd2_rn` / `__fadd2_rd`, `__fmaf_rn` / `__fadd_rd` —
  https://docs.nvidia.com/cuda/cuda-math-api/index.html
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.000e+00 packed_f32x2/single_bitexact mismatched=0`;
  `PROBE PASS max_diff=0.000e+00 packed_f32x2/multi_bitexact mismatched=0`;
  `PROBE PASS max_diff=n/a packed_f32x2/packed_in_sass` (both spellings FFMA2 3 / FMUL2 1 / FADD2 2 / MOV 0 / 32 instructions);
  `PROBE PASS max_diff=0.0 packed_f32x2/rm_suffix rm_vs_rd_mismatch=0 frac_rm_ne_rn=0.500`;
  `PROBE PASS max_diff=0.000e+00 packed_f32x2/intrin_bitexact mismatched=0` (intrinsic version 40 instructions, MOV 3).
  Fact lines: `max_f32x2 assembles=no` (ptxas: Unexpected instruction types specified for 'max'); `split_cost identical_opcodes=True`.
- GPU certification of gotcha 8 (the second CUDA probe):
  bitwise comparison of the three spellings `PROBE PASS max_diff=0.000e+00 ... mismatched=0`; 64-bit glue criterion `PROBE PASS max_diff=n/a ... u64=8 asm1=0 mov32=0`;
  SASS fact line: the C++ `uint64_t` spelling has 4 more per pair than the other two, which have the same count.
