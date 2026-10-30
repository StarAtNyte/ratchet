# branchfree_masked_elementwise_pass (mbarrier-gated element-wise mask pass over a register fragment: reads outside the condition, select the product, write the scalar-only factor chain before the wait)

> **Certification source: B200 GPU-certified in CUDA mode (direct run on a B200 GPU node, all 7 contracts PROBE PASS; the probe is not included in this release; an earlier round of SASS / read-coalescing / timing readings is consistent).**
> The fragment is in registers (the probe does not go through TMEM); a fragment read out by `tcgen05.ld` is likewise a register fragment, but the TMEM part is not certified by this entry.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Compute `y_e = mask(r, e) ? phi(r, e) * x_e : 0` over a 32-element register fragment, where `phi` depends only on row / column scalars in smem
(e.g. `ex2(rowA[r] + colA[e]) * colB[e]`), and the data the fragment needs is gated by an mbarrier. Three levers:

1. **Move the smem scalar reads outside the condition**: with the read inside `if (mask)`, each read is one scalar `LDS`; with an unconditional read + a ternary on the product,
   nvcc merges reads of adjacent columns into `LDS.128` (an ordinary index works when alignment is provable; an explicit `float4` works too).
2. **Select the product**, not the exponent: a masked element is exactly 0 even when `x_e` is inf / NaN.
3. **Write the scalar-only factor chain (LDS → FADD → `ex2` → FMUL) in the source before the `mbarrier.try_wait` wait loop**, so that
   part of it lands in the wait's shadow. Note this is a **request**, not a guarantee: ptxas moves part of it back after the wait loop (gotcha 3).

## When needed (trigger)

- In an element-wise mask segment the same column scalar is read once per element, SASS shows dozens of scalar `LDS`, and the segment's main stall is
  `short_scoreboard` / `mio_throttle`.
- The segment directly follows an mbarrier wait, the producer beat sets the period (the wait dominates), and the segment contains a string of
  `MUFU.EX2` that does not depend on the awaited data.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// Taken from this entry's probe (one beat of the consumer warpgroup).
__device__ __forceinline__ float ex2_approx(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
__device__ __forceinline__ void mbar_wait_try(uint64_t* bar, uint32_t parity) {
  asm volatile("{\n\t.reg .pred p;\nWAIT_%=:\n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
               "@!p bra WAIT_%=;\n\t}"
               :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity) : "memory");
}

__shared__ __align__(16) float colA[32], colB[32];
float ph[32];
const float ra = rowA[r];
#pragma unroll
for (int q = 0; q < 8; ++q) {                                    // before the wait: depends only on scalars
  const float4 a = reinterpret_cast<const float4*>(colA)[q];     // LDS.128
  const float4 b = reinterpret_cast<const float4*>(colB)[q];
  ph[4*q+0] = ex2_approx(ra + a.x) * b.x;  ph[4*q+1] = ex2_approx(ra + a.y) * b.y;
  ph[4*q+2] = ex2_approx(ra + a.z) * b.z;  ph[4*q+3] = ex2_approx(ra + a.w) * b.w;
}
mbar_wait_try(&full_bar, it & 1);
const float g = gslot;                                            // the awaited data
#pragma unroll
for (int e = 0; e < 32; ++e)
  acc[e] += (((e + it) & 31) <= (r & 31)) ? ph[e] * x[e] * g : 0.f;   // select the product -> FSEL
```

## Gotchas

1. **A per-element `if` does not produce a branch under CUDA; what makes it expensive is that the reads cannot be merged.** In the probe the per-element `if` form and the three select forms have
   identical `BRA` / `BSSY` counts (11 / 1, all outside the element loop): nvcc predicated the whole condition body (273 predicated
   instructions in the if form). The difference is in the reads: the if form has `LDS=97` (all scalar), the select forms `LDS=18` (16 of them `LDS.128`); with 3 consumer
   warpgroups and no beat control (issue-bound) t(if)/t(select) = 1.114. An ordinary index and an explicit `float4` compile to the same SASS.
2. **Select the product, not the exponent**: `ex2(-1e30) * x` is NaN when `x = inf`; `mask ? p * x : 0` is always 0
   (the probe puts inf at always-masked positions and the result is bitwise 0).
3. **"Written before the wait" is only a request: ptxas moves part of the independent work back after the wait loop.** The probe writes 32 `ex2` before an
   `asm volatile` + `"memory"` try_wait loop: in PTX all 32 / 32 are before `mbarrier.try_wait` (the front end did not move them),
   but in SASS only 18 remain before the first `TRYWAIT`, and the other 14 `MUFU.EX2` were scheduled by ptxas after the wait loop. Writing `ex2` as
   `asm volatile` also gives 18 — volatile constrains only the front end, not ptxas's scheduling. Not written before the wait: 0.
   Accept by how many actually sit before the wait instruction in SASS; do not infer it from the source position.
4. Even hoisting only part of it pays: with 1 consumer warpgroup and a producer beat of 200 ns, t(no hoist)/t(hoist) = 1.081
   (684 → 632 ns / beat). Without beat control, when the segment itself is the bottleneck, it is 0.949 instead: the hoisted values live across the wait and take extra registers, so do not hoist then.
5. Hoist only work **already protected by an earlier synchronization** (the scalars here are written at kernel start followed by `__syncthreads`); the awaited data
   (`gslot` here) must be read after the wait.
6. `float4` reads need 16-byte alignment (`__align__(16)`), otherwise an illegal address or wrong reads.

## Sources

- PTX ISA: *Parallel Synchronization and Communication Instructions: `mbarrier.try_wait`*;
  *Floating Point Instructions: `ex2`*; <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide / *Inline PTX Assembly in CUDA* (semantics of `asm volatile` and the `"memory"` clobber):
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>
- GPU-certified by the probe (direct run on a B200 GPU node) —
  `PROBE PASS max_diff=2.536e-07 branchfree_masked_elementwise_pass/numeric masked_exact_zero=True`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/no_per_element_branch bra=[11, 11, 11, 11, 11] bssy=[1, 1, 1, 1, 1] fsel=[0, 32, 32, 32, 32] predicated_if=273 predicated_vec=17`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/vec_lds lds_sel_vec=18 lds128=16 lds_if=97 lds_sel_cpp=18 lds128_sel_cpp=16`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/hoist_order_ptx ptx_hoist=32 ptx_hoist_v=32 ptx_vec=0`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/ptxas_sinks_part_across_wait sass_hoist=18 sass_hoist_v=18 sass_vec=0`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/sel_time ratio=1.1141`;
  `PROBE PASS max_diff=n/a branchfree_masked_elementwise_pass/hoist_time ratio=1.0812`.
