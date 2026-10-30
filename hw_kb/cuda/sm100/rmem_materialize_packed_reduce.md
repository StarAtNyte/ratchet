# rmem_materialize_packed_reduce (in-thread row reduction: a four-chain packed `add.rn.f32x2` sum, ptxas fusing a 2-input max chain into `FMNMX3`; "copy into a new array first" is a no-op)

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS, after an earlier first round; the probe is not included in this release).**
> Certification covers: numerics and SASS of the four-chain packed sum, "copy into a new local array first, then reduce" not changing SASS, a 2-input `fmaxf` chain becoming 3-input
> `FMNMX3` in SASS, and a hand-written 3-input `max.f32` not giving fewer instructions. **Uncertified**: the impact of the reduction segment on whole-kernel time (the probe does no timing).
>
> One file per hardware interface. Operator-agnostic: mechanism only. For packed f32x2 spellings see [`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md).

## Purpose

A thread holds a row of 128 fp32 (outputs of upstream FFMAs, kept in registers) to be reduced to one sum / one maximum. On sm_100a:

- **Sum**: with 4 `add.rn.f32x2` chains (each chain consumes one pair of adjacent elements per step), 128 elements need only 63 `FADD2` + 1 `FADD`;
  scalar 4 chains take 131 `FADD`. Whole-kernel SASS 248 vs 320 instructions.
- **Maximum**: just write a 2-input `fmaxf` chain — ptxas itself fuses two adjacent maxes into one 3-input `FMNMX3` (124 pairwise maxes → 62 `FMNMX3`).
  A hand-written PTX 3-input `max.f32 d, a, b, c` is correct, but gives no fewer.

## When needed (trigger)

- Summing a long row within a thread takes most of the FP issue (strings of `FADD` in SASS), and the elements already sit in registers in pairs.
- You want to hand-write a 3-input max to save max instructions: first check whether SASS already has `FMNMX3` — most likely it already does.
- You want to avoid `MOV`s when packing by "first copying the row into a new array (so the register pairs line up)": under CUDA this step changes nothing (gotcha 3).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
#define E 128
__device__ __forceinline__ uint64_t pk(float a, float b) {
  uint64_t r; asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ void upk(uint64_t v, float& a, float& b) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v));
}
__device__ __forceinline__ uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t d; asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b)); return d;
}

// sum: 4 packed chains, 8 elements per round (requires E % 8 == 0); SASS 63 FADD2 + 1 FADD, 0 MOV
__device__ __forceinline__ float row_sum_packed4(const float (&v)[E]) {
  uint64_t acc[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) acc[c] = pk(v[2 * c], v[2 * c + 1]);
#pragma unroll
  for (int j = 1; j < E / 8; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[c] = add2(acc[c], pk(v[8 * j + 2 * c], v[8 * j + 2 * c + 1]));
  uint64_t a = add2(add2(acc[0], acc[1]), add2(acc[2], acc[3]));
  float x, y; upk(a, x, y);
  return x + y;
}

// maximum: a plain 2-input fmaxf chain; ptxas produces FMNMX3 (4 chains: 62 FMNMX3 + 3 FMNMX)
__device__ __forceinline__ float row_max4(const float (&v)[E]) {
  float a[4] = {v[0], v[1], v[2], v[3]};
#pragma unroll
  for (int i = 4; i < E; ++i) a[i % 4] = fmaxf(a[i % 4], v[i]);
  return fmaxf(fmaxf(a[0], a[1]), fmaxf(a[2], a[3]));
}
```

## Gotchas

1. **The packed sum changes the order of additions.** 4 chains × two lanes per chain = 8 partial sums, merged pairwise at the end; not bitwise identical to a serial sum
   (relative error against fp64 in the probe: serial 6.2e-7, packed 1.6e-7). When bitwise agreement with some fixed order is required, do not switch.
2. **Packing straight from FFMA outputs needs no `MOV`.** In the probe all 128 elements are results of upstream `FFMA`s, and packing them to feed `FADD2` gives 0 `MOV` in SASS:
   ptxas itself allocates adjacent elements as aligned register pairs.
3. **"Copy into a new local array first, then reduce" is a no-op.** Reducing directly vs first `m[i] = v[i]` then reducing: identical SASS opcode sequences (248 = 248 instructions).
   Do not add this step "to line up the register pairs".
4. **A 2-input max chain is already 3-input.** An `fmaxf` chain is `FMNMX3` in SASS; a hand-written `max.f32 d, a, b, c` (3-input PTX) assembles and is correct
   (including the edge triples with `-inf` and positive / negative zero), but the max instruction count is 66 vs 65, no gain.
   A single-accumulator serial `fmaxf` chain is fused just the same (63 `FMNMX3` + 1 `FMNMX`), so there is no need to split into multiple chains to form triples either.
5. **NaN semantics**: `fmaxf` / `max.f32` return the non-NaN operand; `torch.maximum` propagates NaN. Keep NaN out of comparison inputs, or handle it separately.

## Sources

- PTX ISA: *Floating Point Instructions: `add` (`.f32x2`)*, *`max` (three-input form)* — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Math API: `fmaxf` — https://docs.nvidia.com/cuda/cuda-math-api/index.html
- GPU-certified by the probe (plus a three-input max variant):
  `PROBE PASS max_diff=6.201e-07 materialize/numeric` (serial 6.2e-07, chain4 2.2e-07, packed4 / packed4_mat 1.6e-07, max exact);
  `PROBE PASS max_diff=n/a materialize/packed_in_sass FADD2=63 FADD=1`;
  `PROBE PASS max_diff=n/a materialize/mat_noop instrs_direct=248 instrs_mat=248 mov_direct=0 mov_mat=0`;
  `PROBE PASS max_diff=0.000e+00 materialize/max2_fuses_fmnmx3 FMNMX3_4chain=62 FMNMX_all_4chain=65 FMNMX3_1chain=63 FMNMX_all_1chain=64`;
  `PROBE PASS max_diff=0.000e+00 materialize/max3_ptx_no_gain edge_ok=True max_instrs_3in_ptx=66 max_instrs_2in_fmaxf=65`.
  Fact lines: scalar serial 312 instructions (128 `FADD`), scalar 4 chains 320 (131 `FADD`), packed 4 chains 248.
