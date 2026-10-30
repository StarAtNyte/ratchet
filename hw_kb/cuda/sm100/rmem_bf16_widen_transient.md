# rmem_bf16_widen_transient (bf16 → f32 widening should be "widen at each use": the compiler merging / hoisting widenings creates resident f32 copies; which spellings stop it)

> **Certification source: B200 GPU-certified in CUDA mode (all 6 contracts PROBE PASS, after an earlier first round; the probe is not included in this release).**
> Certification covers: the causal breakdown of register readings (`cuobjdump -res-usage`) and SASS widening-instruction counts, bitwise identical numerics, and that adding a register cap spills rather than recomputes.
> **Uncertified**: the time gain from fewer registers (kernel-time medians of the variants in the probe 22.6–28.9 µs, no significance test; the late-load variant is actually slightly slower).
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Widening bf16 → f32 costs only one integer instruction per element (low half word `shl.b32 x, w, 16`, high half word `and.b32 x, w, 0xffff0000`),
so "widen fresh at every use" is nearly free, whereas "widen once and keep the f32 copy" takes twice the registers (N f32 vs N/2 packed words).
By default the compiler does the latter:

- **Merging across use sites**: the same packed data widened once at each of two use sites (with a warp reduction in between) is merged into one widening by NVVM's CSE (C++ spelling) or by ptxas
  (asm with identical PTX text), and the f32 copy stays resident across the whole reduction;
- **Hoisting loop invariants**: for an invariant used every round of a loop, the C++-spelled widening is hoisted out of the loop by NVVM, and the f32 copy stays resident for the whole loop.

## When needed (trigger)

- REG in `cuobjdump -res-usage` is clearly above the "packed data + accumulators" estimate, while the widening instructions in SASS (`SHF` / `LOP3` / `IMAD.U32`) number
  only "once per element" — meaning they were merged / hoisted.
- After adding a `__launch_bounds__` cap, `STACK > 0` and lots of `LDL` / `STL` appear (probe: STACK 160, 81 of them): ptxas chooses to spill the f32 copies rather than switch to recomputing the widening (gotcha 4).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// Widening in two PTX shapes: same content, but ptxas will not merge prmt with shl/and
__device__ __forceinline__ float lo_a(uint32_t u) {          // shl/and shape
  uint32_t r; asm volatile("shl.b32 %0, %1, 16;" : "=r"(r) : "r"(u)); return __uint_as_float(r);
}
__device__ __forceinline__ float hi_a(uint32_t u) {
  uint32_t r; asm volatile("and.b32 %0, %1, 0xffff0000;" : "=r"(r) : "r"(u)); return __uint_as_float(r);
}
__device__ __forceinline__ float lo_p(uint32_t u) {          // prmt shape: selector 0x1044 = low half word << 16
  uint32_t r; asm volatile("prmt.b32 %0, %1, %2, 0x1044;" : "=r"(r) : "r"(u), "r"(0u)); return __uint_as_float(r);
}
__device__ __forceinline__ float hi_p(uint32_t u) {          // selector 0x3244 = high half word (= u & 0xffff0000)
  uint32_t r; asm volatile("prmt.b32 %0, %1, %2, 0x3244;" : "=r"(r) : "r"(u), "r"(0u)); return __uint_as_float(r);
}

// row loop (one warp per row; per lane 16 bf16x2 words of x, and the same 16 words of w for every row)
for (int row = warp; row < rows; row += nwarps) {
  // use site 1: sum of squares -> warp butterfly reduction. Widen x in the prmt shape
  float s = 0.f;
  for (int i = 0; i < 16; ++i) { float a = lo_p(xr[i]), b = hi_p(xr[i]); s = fmaf(a, a, s); s = fmaf(b, b, s); }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const float rstd = rsqrtf(s * (1.0f / 1024.0f) + 1e-6f);
  // use site 2: re-widen x in the shl/and shape; the invariant w is also widened here with asm volatile (not hoisted out of the loop)
  for (int i = 0; i < 16; ++i) {
    o[2 * i]     = (lo_a(xr[i]) * lo_a(wr[i])) * rstd;
    o[2 * i + 1] = (hi_a(xr[i]) * hi_a(wr[i])) * rstd;
  }
}
```

Probe readings (148×128 threads, one warp per row, 16 x words + 16 w words per lane; "widening instructions" = whole-kernel `SHF`+`LOP3`+`PRMT`+`IMAD.U32/SHL`):

| Variant | Two use sites of x | Invariant w | REG | Widening instructions |
|---|---|---|---|---|
| naive | C++ at both | C++ widening, loaded before the loop | 80 | 88 |
| xasm_wcpp | identical asm text at both | C++ widening, loaded before the loop | 80 | 88 |
| xcpp_wasm | C++ at both | asm volatile widening at the use site, loaded before the loop | 64 | 88 |
| opaque | identical asm text at both | asm volatile widening at the use site, loaded before the loop | 64 | 88 |
| late | identical asm text at both | asm volatile, loaded late at the use site per row | 64 | 90 |
| split_wpre | site 1 `prmt`, site 2 `shl`/`and` | asm volatile, loaded before the loop | 62 | 120 |
| **split** | site 1 `prmt`, site 2 `shl`/`and` | asm volatile, loaded late per row | **48** | 120 |
| xorz | split + site 2 widens `x ^ z` | as split | 55 | 138 |
| naive + `__launch_bounds__(128, 12)` | — | — | 40 (STACK 160, 81 `LDL`/`STL`) | 89 |

## Gotchas

1. **`asm volatile` cannot stop ptxas from merging identical PTX.** With the same `shl.b32` / `and.b32` text at both use sites, NVVM does not merge (volatile),
   but ptxas sees the same instruction on the same input register and merges anyway: opaque has 88 widening instructions, exactly the same as the C++ spelling (NVVM CSE);
   x written as C++ at both sites (xcpp_wasm) and as identical asm (opaque) both give REG 64. The f32 copy stays resident across the warp reduction regardless.
2. **Only giving the two use sites different PTX shapes splits them.** Use site 1 with `prmt`, use site 2 with `shl` / `and`: widening instructions 88 → 120 (32 extra `PRMT`),
   REG 64 → 48 — only the packed words of x stay resident during the reduction.
3. **The C++ widening of a loop invariant gets hoisted out of the loop.** When w is widened in the C++ spelling, NVVM hoists 32 f32 copies out of the loop as residents:
   with x's spelling unchanged and only w switched to C++ widening (xasm_wcpp), REG 64 → 80, the same as naive. Widen at the use site with `asm volatile`, and ptxas does not hoist it out of the loop.
   **Loading w late per row (`asm volatile ld.global`) helps only together with gotcha 2**: while x is still merged, the late load saves nothing (opaque 64 → late 64);
   once x is split, the late load saves another 14 (split_wpre 62 → split 48). Which cut works depends on who holds the peak; check `-res-usage` cut by cut.
4. **Adding a register cap does not buy recomputation.** The naive shape with `__launch_bounds__(128, 12)`: REG 40, STACK 160, 81 `LDL` / `STL`.
5. **xor with a runtime zero also splits the merge, but costs more**: xorz has 18 more widening / logic instructions than split, REG 55 vs 48.
6. **Fewer registers does not mean faster.** Kernel-time medians in this probe: naive 23.7, opaque 23.5, split 25.6, late 26.0, capped version 28.9 µs —
   the late load pays an extra load per row. It is worth doing only when REG limits occupancy or causes spills, and then it must be timed in the same job.

## Sources

- PTX ISA: *Integer Arithmetic: `shl` / `and`*, *Data Movement: `prmt`* (selector semantics), *`ld`* —
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA C++ Programming Guide: `asm volatile` in inline PTX, `__launch_bounds__` — https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
- GPU-certified by the probe:
  `PROBE PASS max_diff=2.021e-07 bf16_widen/numeric` (9 variants bitwise identical to naive);
  `PROBE PASS max_diff=n/a bf16_widen/no_stack` (uncapped variants STACK 0);
  `PROBE PASS max_diff=n/a bf16_widen/ptxas_merges_same_asm widen_ops_naive=88 widen_ops_opaque=88 REG_opaque=64 REG_xcpp_wasm=64`;
  `PROBE PASS max_diff=n/a bf16_widen/split_breaks_merge PRMT=32 widen_ops_split=120 REG_late=64 REG_split=48`;
  `PROBE PASS max_diff=n/a bf16_widen/hoist_costs_regs REG_naive=80 REG_xasm_wcpp=80 REG_opaque=64`;
  `PROBE PASS max_diff=n/a bf16_widen/cap_spills REG=40 STACK=160 LDL_STL=81`.
  Fact line: `late_load REG_opaque=64 REG_late=64 REG_split_wpre=62 REG_split=48 REG_xorz=55`.
