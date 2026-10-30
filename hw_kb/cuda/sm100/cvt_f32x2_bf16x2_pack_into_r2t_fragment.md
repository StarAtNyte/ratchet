# cvt_f32x2_bf16x2_pack_into_r2t_fragment (an adjacent f32 pair → one `cvt.rn.bf16x2.f32` → one 32-bit word → directly one column slot of `tcgen05.st`)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cell `cvtpack`; the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the DSL's recast view / TensorSSA wrapping is DSL-specific and is not carried over here.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Architecture scope: sm100 (the conversion instruction exists on every generation; this entry covers how it feeds the 32-bit columns of `tcgen05.st`).

## Purpose

`tcgen05.st` moves data by 32-bit columns, two bf16 per column. To land a row of f32 in TMEM as bf16 (e.g. the A operand of the next MMA, see
[`tmem_r2t_recast_fragment`](tmem_r2t_recast_fragment.md)), the shortest path in hardware is PTX `cvt.rn.bf16x2.f32 d, a, b`:
**two f32 in, one 32-bit word out** (`a` into the high half, `b` into the low half), one SASS `F2FP.BF16.F32.PACK_AB`. That word is one source register of `tcgen05.st`,
written whole, never touching half registers. In CUDA C++ the same instruction can also be obtained from `__floats2bfloat162_rn(lo, hi)` (`cuda_bf16.h`,
`.x` = low half).

## When needed (trigger)

- The output of an element-wise piece of code must enter TMEM as bf16 (or any register group consumed in 32-bit slots: a `tcgen05.st` source, an `stmatrix` source, an `st.shared.v4` source);
- in a SASS census the `F2FP` count == elements per thread, accompanied by an equal number of `PRMT` (the degenerate shape of converting element by element and then inserting half words), or you are unsure which shape your spelling lowered to.

## Minimal runnable example

The spelling matches the certification probe (TMEM row → 32 f32 → 16 packed words → written back to another 16 columns of TMEM → read back and checked):

```cpp
__device__ inline uint32_t pack_bf16x2(float lo, float hi) {     // lo -> low half (the first bf16 in memory)
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
uint32_t x[32];
tm_ld32x32b_x32(trow, x);                                        // trow = tbase + ((32*warp) << 16)
asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
uint32_t p[16];
#pragma unroll
for (int j = 0; j < 16; ++j)                                     // slot j = source elements (2j, 2j+1)
  p[j] = pack_bf16x2(__uint_as_float(x[2 * j]), __uint_as_float(x[2 * j + 1]));
tm_st32x32b_x16(trow + 64, p);                                   // one packed word = one 32-bit column
asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
```

Machine-checkable criterion (`cuobjdump -sass` census of the kernel carrying this section): `F2FP.BF16.F32.PACK_AB` count == elements per thread / 2; `PRMT` == 0 within the packing section;
`LDL == STL == 0`. The probe's readings are under Sources.

## Gotchas

- **Half order: in `cvt.rn.bf16x2.f32 d, a, b`, a is the high half.** To land (src[2j], src[2j+1]) as (low half, high half), write the asm sources as `hi, lo`. Getting it backwards
  **raises no error and produces no inf/nan**: swapping the two halves of every word = swapping adjacent column pairs, so the downstream result is wrong but finite. The probe pins this down with a bitwise comparison.
- **Rounding is RNE (`.rn`)**: bitwise identical to the host-side bf16 rounding "+0x7FFF + LSB, then truncate" (the probe checks with non-integer inputs); only switching to `.rz` diverges.
- **The source pair must be adjacent and in the consumer's column order**: the j-th 32-bit column ↔ source elements (2j, 2j+1). If the source register order is not column order (e.g. a fragment from `.16x256b`),
  first find the right pairs from that readback instruction's thread-value map (see [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md)).
- **Writing half words element by element is different code**: converting one bf16 first and then inserting it into half of a word (`PRMT` / shift-or) doubles the instruction count under register-budget pressure; without pressure ptxas may pair them itself.
  Judge by the footprint (`F2FP` count), not by the source spelling.
- **After `tcgen05.st`, `tcgen05.wait::st` is required** before the same thread reads back or hands off to another role (plus `tcgen05.fence::before_thread_sync` + a sync).

## Sources

- PTX ISA: `cvt` (`cvt.rn.bf16x2.f32 d, a, b`: a → high half, b → low half), `tcgen05.st` / `tcgen05.wait::st` (32-bit column addressing) —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUDA Math API `__floats2bfloat162_rn` (`cuda_bf16.h`).
- GPU-certification probe (`cvtpack`), verdict
  `PROBE PASS max_diff=0.0`: `C4-cvt-bf16x2-into-tmem-slot` (2048 packed words round-tripped through `tcgen05.st` / `ld`, 0 mismatches against host RN-even,
  non-integer inputs). SASS (carrying kernel, 32 f32 per thread): `F2FP.BF16.F32.PACK_AB` 16, `PRMT` 0, `LDL` / `STL` 0.
