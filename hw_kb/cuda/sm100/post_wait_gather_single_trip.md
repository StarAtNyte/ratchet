# post_wait_gather_single_trip (an "issue the whole batch" gather after a dependency wait: each chunk's 33 independent loads in one round trip, instead of one trip for metadata, a reduction, then another trip for data)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS` (`pdl-gather-numeric` max_diff=0, `pdl-gather-sass-batched`, `pdl-gather-one-trip-timing` max_diff=1.20338).**
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

After `griddepcontrol.wait` (PDL) or any dependency wait, a warp must read, per chunk, one per-lane piece of metadata (a pair of f32) **and** 32 rows of data gathered at lane-broadcast
addresses, then do a warp reduction (`SHFL.BFLY` / `FMNMX`) + weighted accumulation. This entry covers how to issue these 33 loads as one batch and pay for only one
round trip; if the predicate or address of the data loads depends on the reduction result, it is necessarily two trips.

## When needed (trigger)

- In SASS there is still a string of `LDG` **after** the first `SHFL.BFLY` / `FMNMX` (the data-row loads land after the reduction chain). That is the criterion; the source shape is only a suspicion.
- The predicate of the data loads is taken from the reduction result (e.g. "skip this row when its weight is 0"): that is a true dependency, necessarily two trips.

## Minimal runnable example

```cpp
asm volatile("griddepcontrol.wait;" ::: "memory");          // only once, before the first load that reads producer data
for (int ch = 0; ch < nch; ++ch) {                          // #pragma unroll 1
  const int p = ch * 32 + lane, valid = p < n;
  const int slot = slot_of(row, p);                         // your address function (pure arithmetic)
  const int key = valid ? slot : -1;                        // predicate folded into the sign bit: only 1 shfl per row
  int keys[32];
  for (int i = 0; i < 32; ++i) keys[i] = __shfl_sync(0xffffffffu, key, i);   // compute the 32 broadcasts into registers first
  float buf[34];
  float2 ab = ld_v2_if(meta + slot, valid);                 // @p ld.global.cg.v2.f32, disabled reads give 0
  buf[0] = ab.x; buf[1] = ab.y;
  for (int i = 0; i < 32; ++i) buf[2 + i] = ld_f32_if(x + (size_t)keys[i] * 32 + lane, keys[i]);  // key < 0: no memory access
  // dependency token: the first consumer waits for all 34 values of this batch; Z is a kernel parameter that is 0 at runtime
  uint32_t t = 0;
  for (int i = 0; i < 34; ++i) t ^= __float_as_uint(buf[i]);
  t &= (uint32_t)Z;
  float mx = __uint_as_float(__float_as_uint(valid ? buf[0] : -3.0e38f) | t);
  for (int o = 1; o < 32; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float coef = valid ? (buf[0] - mx) * buf[1] + 1.0f : 0.0f;
  for (int i = 0; i < 32; ++i) acc += __shfl_sync(0xffffffffu, coef, i) * buf[2 + i];
}
// ld_f32_if: asm volatile("{ .reg .pred p; setp.ge.s32 p, %2, 0; mov.b32 %0, 0; @p ld.global.cg.f32 %0, [%1]; }"
//                         : "=f"(x) : "l"(addr), "r"(key));
// The producer does asm volatile("griddepcontrol.launch_dependents;") after its last global write (position: gotcha 3 of pdl_griddepcontrol_launch;
// this entry's probe puts the producer's at the start, which does not affect visibility after the wait); the consumer is launched with cudaLaunchKernelEx +
// cudaLaunchAttributeProgrammaticStreamSerialization = 1.
```

## Gotchas

1. **Source order ≠ issue order**: in the probe the same batch of loads written before the reduction but without a token (`k_notoken`) has only 5 `LDG`s before the first BFLY/FMNMX in SASS and
   28 after — ptxas sank them, and the timing is even slower than the batched shape with the token (see gotcha 5). What reliably constrains ptxas is a **data dependency**.
2. **Z in the token must be a runtime kernel parameter** (a constant 0 / `x*0` gets folded); keep the token in the integer domain (bit-pattern OR); a floating-point `x + 0*t` contaminates the result on inf/NaN.
3. **Do not let the loads' predicate or address depend on the reduction result**: the probe's `k_naive` (data-load predicate taken from `coef`) has 1 `LDG` before the first BFLY in SASS and 32 after.
4. **Fold the predicate into the sign bit of the broadcast value**, and load with `@p ld` (after `mov 0`) instead of `if`: an out-of-bounds lane's address is out of bounds, but the predicate is off, so there is no memory access.
5. **Probe timing** (148 CTAs x 4 warps, 160 chunks per row, median of 9):

   | Table size | naive (two trips) | batched + token | batched without token |
   |---|---|---|---|
   | 8 MB (L2-resident) | 225.0 µs | 206.8 µs (naive / batched = 1.088) | 226.1 µs |
   | 512 MB (far larger than L2) | 412.8 µs | 343.0 µs (naive / batched = **1.203**) | 451.9 µs |

   One trip instead of two saves one round trip; the more expensive the round trip (data not in L2), the larger the gain; L2-resident it is only about 9%.
6. **Timing must be device-bound**: each launch must do enough work, otherwise you measure launch overhead.
7. **Put the wait only once**, before the first load that reads producer data; work that does not read the producer's output (row-index scans, address broadcasts) may go before the wait.

## Sources

- PTX ISA: `griddepcontrol`, the `.cg` modifier of `ld` and predicated execution, `shfl.sync`: <https://docs.nvidia.com/cuda/parallel-thread-execution/>
- CUDA C++ Programming Guide, Programmatic Dependent Launch (`cudaLaunchAttributeProgrammaticStreamSerialization`;
  `cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion` are the runtime API for the same mechanism):
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization>
- GPU-certification probe — `PROBE PASS max_diff=0` (the three variants agree with the host reference through the PDL
  producer → consumer path), `PROBE PASS max_diff=n/a` (`pdl-gather-sass-batched`: all 33 `LDG`s of the batched shape before the first
  BFLY/FMNMX, 0 `LDL/STL`), `PROBE PASS max_diff=1.20338` (`pdl-gather-one-trip-timing`: naive / batched ≥ 1.15 when the table is far larger than L2;
  max_diff is this time ratio).
