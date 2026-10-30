# tmem_chunk_e4m3_quant_chain (chunked TMEM readback → per-chunk amax → UE8M0 → e4m3 packing → 256-bit write: compute the scale byte with three integer instructions instead of `div.rn.f32`, making the whole segment branch-free straight-line code)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the probe is not included in this release; verdicts and SASS readings under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the readings were re-taken on the CUDA spelling.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Architecture scope: sm100. For the ordering edges between readback and face writes / fence / rendezvous see
> [`tmem_ld_epilogue_issue_order`](tmem_ld_epilogue_issue_order.md).

## Purpose

One chunk (32 f32, 32 columns of one row per thread) must do:

```
x[0..31]  <- tcgen05.ld.32x32b.x32
amax      <- max |x|
e         <- UE8M0 byte: the smallest s = 2^(e-127) with amax / s <= 448 (the largest finite e4m3 value)
q[i]      <- x[i] * 2^(127-e)                    (multiply by a power of 2, exact)
w[0..7]   <- cvt.rn.satfinite.e4m3x2.f32 x16 + word assembly
st.global.v8.b32 [row + 32*k], w                 (one 256-bit write)
```

Two hardware facts decide how short this chain can be:

1. **`div.rn.f32` is not one instruction**: IEEE correctly rounded division in SASS is `MUFU.RCP` + Newton iteration + an `FCHK` check + a conditional branch to a slow-path subroutine
   (`CALL`). One division per chunk = one basic-block boundary per chunk, and ptxas schedules only within basic blocks.
2. **"÷448, then round up to a power of 2" has an exact division-free equivalent**: 448 = 1.75 · 2⁸. Let `b` be the bit pattern of `amax`:
   **`e = ((b + 0x1FFFFF) >> 23) − 8`**, then clamp to `[0, 254]` (a mantissa > 0x600000, i.e. > 1.75, carries into the exponent field).
   The inverse scale is assembled directly from bits: the bit pattern of `2^(127−e)` is `(254 − e) << 23`. It is bitwise identical to "`div.rn`, take the exponent, +1 if the mantissa is non-zero" on the domain where `amax/448` is a normal f32
   (`div.rn` never rounds the next representable number above `448·2^k` back to `2^k`).

## When needed (trigger)

- A per-chunk epilogue segment (readback → amax → scale byte → narrow-type packing → write) is on the critical path, and SASS has a group of `MUFU.RCP … FCHK … BRA`
  or a `CALL` to the division subroutine per chunk;
- you want the next chunk's `tcgen05.ld` to land under this chunk's arithmetic, but find the SASS cut into one segment per chunk by the division.

## Minimal runnable example

Same spelling as the certification probe:

```cpp
__device__ __forceinline__ uint32_t ue8m0_int(float amax, float* inv) {    // no division
  int e = (int)((__float_as_uint(amax) + 0x1FFFFFu) >> 23) - 8;            // 448 = 1.75 * 2^8
  e = e < 0 ? 0 : (e > 254 ? 254 : e);
  *inv = __uint_as_float((uint32_t)(254 - e) << 23);                        // 2^(127-e)
  return (uint32_t)e;
}
__device__ __forceinline__ uint16_t e4m3x2(float lo, float hi) {           // lo -> low byte
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
// per chunk:
uint32_t x[32];
tm_ld32x32b_x32(trow + 32 * c, x);
asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
float amax = 0.f;
for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(__uint_as_float(x[j])));
float inv; const uint32_t e = ue8m0_int(amax, &inv);
uint32_t w[8];
for (int j = 0; j < 8; ++j)
  w[j] = e4m3x2(__uint_as_float(x[4*j]) * inv, __uint_as_float(x[4*j+1]) * inv)
       | ((uint32_t)e4m3x2(__uint_as_float(x[4*j+2]) * inv, __uint_as_float(x[4*j+3]) * inv) << 16);
asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
             :: "l"(q + row * 128 + c * 32), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]) : "memory");
sf[row * 4 + c] = (uint8_t)e;
```

**Issuing the next chunk's readback early** (separating `ld` and `wait`, issuing the next chunk's `ld` before this chunk's arithmetic) is also numerically certified: issue chunk c+1's `tcgen05.ld` before computing chunk c,
and `tcgen05.wait::ld` at the end of chunk c (it waits for **all** `tcgen05.ld` previously issued by this thread). This is not a proven speed-up lever; the timing is only a fact line, see Sources.

## Gotchas

1. **`cvt.rn.satfinite.e4m3x2.f32 d, a, b`: a goes into the high byte**; reversing it swaps adjacent elements, finite but wrong numbers.
2. **`.satfinite` must not be dropped**: without it, rounding of |x·inv| beyond 448 produces NaN; with it, values clamp to ±448. With e computed as above, `|x·inv| ≤ 448` always holds,
   but values right on the rounding boundary still rely on it as a safety net.
3. **Chunks with amax = 0**: the integer formula gives e = 0 (clamped at the lower bound), payload all 0; consistent with the division spelling (one of the probe's boundary chunks).
4. **`st.global.v8.b32` needs 32 B alignment**; one chunk of 32 e4m3 is exactly 32 B.
5. **Once the whole segment is division-free, whether ptxas has already hoisted this chunk's `LDTM` in SASS** must be counted yourself (`cuobjdump -sass`); hoisting explicitly in the source only pins the order.

## Sources

- PTX ISA: `div.rn.f32`, `cvt.rn.satfinite.e4m3x2.f32`, `st.global.v8.b32` (256-bit vector write, sm100), `tcgen05.ld` / `tcgen05.wait::ld` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe (three variants `div` / `int` / `intpf`, payload and scale bytes checked bitwise against the host reference,
  including boundary chunks with amax = 0, exactly 448·2^k, and slightly above and below it), verdict
  `PROBE PASS max_diff=0.0`: `C5-quant-div` / `C5-quant-int` / `C5-quant-intpf` (16384 payload bytes, 512 scale bytes, 0 mismatches).
  SASS: the division version has `FCHK` 4 and 4 more `CALL`s than the integer version (one slow-path subroutine per chunk); the integer and early-issue versions have `FCHK` 0, and their key instruction sequences are identical item by item (`LDTM ×2` at the start: ptxas already hoisted the next chunk's readback by itself);
  one `STG.E.ENL2.256` per chunk. Toy timing (4 chunks, `%clock64`): division 1363–1435, integer 752–784, early issue 767–827 cycles — early issue is not faster.
