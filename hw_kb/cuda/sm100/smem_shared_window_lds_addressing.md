# smem_shared_window_lds_addressing (dynamic smem reads/writes lowering to `LDS` / `LDS.128` / `STS` rather than generic-addressing `LD.E` / `ST.E`: do not align the base address through a `uintptr_t` round trip)

> **Certification source: B200 GPU-certified in CUDA mode (SM clock locked at 1500 MHz; the probe is not included in this release;
> numerics + both SASS contracts all PROBE PASS).** Certification covers: numerics of four addressing spellings over the same dynamic smem (the integer-rounding anti-pattern / spellings A / B / C),
> per-kernel SASS read/write instruction counts, and per-hop cycles of a single-lane pointer chase. Whole kernel in a real kernel: changing only this one base-address line (anti-pattern → spelling A) was 4.2–6.7% faster in the same run (see Sources).
>
> One file per hardware interface. Operator-agnostic: it covers only "how to let the compiler know this pointer points to shared". Architecture scope: sm100 (the mechanism itself is architecture-independent).

## Purpose

Make C++ dereferences that read/write dynamic smem lower to `LDS` / `LDS.128` / `STS` (shared state space) rather than generic-addressing `LD.E` / `LD.E.128` / `ST.E`.
Generic reads take the LSU's generic path, and their waits are booked on the **long scoreboard** (the same class of scoreboard as gmem reads).

The principle (one sentence): the compiler infers the shared state space for a generic pointer only when it can trace the pointer's **definition chain** all the way back to a `__shared__` variable (pointer arithmetic throughout).
After one `uintptr_t` integer operation (`+`, `&`, round-up alignment) the definition chain is broken, and every pointer derived from it can only issue `LD` / `ST` as a generic pointer.

## When needed (trigger)

- SASS has `LD.E` / `LD.E.64` / `LD.E.128` / `ST.E*` whose addresses come from pointers derived from an `extern __shared__` array.
  (Real gmem reads are `LDG.E*`; `LD.E*` without the G is generic addressing.)
- `long_scoreboard` is high in the stall ledger, while the booked rows depend only on values in smem (e.g. the `IMAD.WIDE` of "take an index from smem → build a gmem address").
- The source has this spelling: the `extern __shared__` base is **converted to an integer** and aligned back, and the sub-regions are derived from that pointer:
  `T* base = (T*)(((uintptr_t)dyn + A - 1) & ~(uintptr_t)(A - 1));`

## Minimal runnable example

```cpp
#include <stdint.h>
__device__ __forceinline__ uint32_t smem_u32(const void* p) {      // 32-bit shared address
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Anti-pattern: integer rounding -> every read/write derived from base is LD.E / LD.E.128 / ST.E
__global__ void bad(/*...*/) {
  extern __shared__ uint8_t dyn_a[];
  uint8_t* base = reinterpret_cast<uint8_t*>(
      (reinterpret_cast<uintptr_t>(dyn_a) + (ALIGN - 1)) & ~uintptr_t(ALIGN - 1));
  int* T = reinterpret_cast<int*>(base);                  // T[i] -> LD.E
}

// Spelling A: compute the alignment from the 32-bit shared address, but add it to the original pointer (pointer arithmetic, chain unbroken)
__global__ void good_offset(/*...*/) {
  extern __shared__ uint8_t dyn_b[];
  uint8_t* base = dyn_b + ((ALIGN - (smem_u32(dyn_b) & (ALIGN - 1))) & (ALIGN - 1));
  int*    T = reinterpret_cast<int*>(base);               // T[runtime_i] -> LDS
  float4* F = reinterpret_cast<float4*>(base + F_OFF);    // F[j]         -> LDS.128 (F_OFF a multiple of 16)
}

// Spelling B: declare the alignment directly, no runtime rounding at all
__global__ void good_align(/*...*/) {
  extern __shared__ __align__(ALIGN) uint8_t dyn_c[];
  int* T = reinterpret_cast<int*>(dyn_c);                 // -> LDS
}

// Spelling C: when you already have the 32-bit shared address (the one used for ldmatrix / TMA / mbarrier), issue ld.shared / st.shared directly,
//             regardless of how the generic pointer was obtained (in the probe spelling C deliberately keeps the anti-pattern base, and reads/writes are still LDS / STS)
__device__ __forceinline__ int ld_shared_b32(uint32_t a) {
  int v; asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ float4 ld_shared_v4f32(uint32_t a) {     // a must be 16 B aligned
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_shared_b32(uint32_t a, int v) {
  asm volatile("st.shared.b32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
```

Acceptance (machine-readable): `cuobjdump -sass <module .so>`, count read/write instructions per kernel: in the kernel carrying this code, generic reads `LD` / `LD.E*` should be 0 and `LDS*` / `LDS.128` > 0;
`ST.E*` correspondingly becomes `STS*`.

## Gotchas

1. **Integer arithmetic is the only break point, not `reinterpret_cast` itself**. A `reinterpret_cast<int*>` / `<float4*>` between pointers does not break the inference; a `uintptr_t` round trip does.
   One bad base contaminates **every** sub-region derived from it; fixing that one place fixes them all.
2. **Spelling A computes the remainder from the 32-bit shared address** (`smem_u32(dyn) & (ALIGN-1)`); do not assume the dynamic smem base is aligned, nor that it is not.
   The probe read only the base remainder after spelling B declared the alignment (0, in a kernel that also has a 148 B static `__shared__` array); the remainder of the original base without declared alignment was not read separately by the probe.
3. **Whether spelling B is really aligned: prove it once at runtime by reading `smem_u32(dyn) & (ALIGN-1)`**. SW128 / TMA / UMMA faces require 1024 B alignment, and misalignment computes wrong numbers (not tested by this probe; alignment requirements in the `tma_*` / `umma_raw_desc_issue` entries).
4. **Spelling C's asm carries a `"memory"` clobber**, and still needs the usual `__syncthreads()` / `__syncwarp()` against earlier C++ writes; `asm volatile` provides no cross-thread ordering.
   The address of `ld.shared.v4` must be 16 B aligned.
5. **`smem_u32()` is correct under any spelling**: `cvta.to.shared` computes the right shared offset from a generic pointer too, so
   `ldmatrix` / `cp.async` / TMA / mbarrier, which take a 32-bit address anyway, are unaffected by the anti-pattern — only ordinary reads/writes issued through C++ dereferences (`p[i]`, `*p`) are affected.
   This is also why in the same kernel `LDSM` is fine while the sub-region read through C++ indexing became `LD.E`.
6. **Static `__shared__` arrays are unaffected**: in the same kernel, even when the dynamic region takes the anti-pattern, static arrays are still `LDS` / `STS`;
   so "there is LDS" in SASS does not prove the dynamic region is fixed; look at the few reads of the dynamic region (e.g. whether `LD.E.128` became `LDS.128`).
7. **The cost is not in the unloaded single-hop latency**: on a dependency chain with a single lane and no gmem background traffic, a generic `LD` is only 3 cycles per hop slower than `LDS` (32.0 vs 29.0, see Sources).
   The whole-kernel readings (see Sources) are far larger than this difference, and the probe did not isolate where it comes from (inference: the waits of generic reads are booked on the long scoreboard, the same class as in-flight gmem reads,
   so they wait longer when there is gmem traffic; not tested).
8. Whether the inference still holds when a derived pointer goes through a non-inlined function parameter, or is stored into smem / a local array and fetched back — **not tested**; when you hit it, go by SASS counts, or switch to spelling C.

## Sources

- PTX ISA: Generic Addressing, State Spaces (`.shared`), `ld` / `st`, `cvta` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide: `__shared__` / dynamic shared memory, `__cvta_generic_to_shared` —
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>; Inline PTX Assembly (`asm volatile`, the `"memory"` clobber) —
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>
- CUDA Binary Utilities (`cuobjdump -sass`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probe (256 threads, 1 CTA; the four spellings read the same int table as a runtime-index gather,
  then read one `float4`; plus a single-lane pointer chase of 4096 hops, timed with `clock64`, minimum of 5). Verdicts:
  numerics `PROBE PASS max_diff=0.0`, both SASS contracts `PROBE PASS max_diff=n/a`. Readings:
  - the numerics of all four spellings agree (`max_diff=0.0`); spelling B's base remainder `base_mod1024=0`.
  - SASS (per-kernel counts): anti-pattern `genLD=9 genST=8 LDS=1 LDS128=0 STS=1` (the reads include `LD.E`, `LD.E.128`; the remaining 1 `LDS` / `STS` is the static array);
    spellings A / B / C all `genLD=0 genST=0 LDS=10 LDS128=1 STS=9`. Pointer-chase kernel: anti-pattern `genLD=2 genST=5 LDS=0`, spelling A `genLD=0 genST=0 LDS=2 STS=5`.
  - pointer chase per hop: anti-pattern (generic `LD.E`) 32.0 cycles, spelling A (`LDS`) 29.0 cycles.
- Whole-kernel reading in a real kernel (the same mechanism in a real kernel, changing only the dynamic smem base line: anti-pattern → spelling A): measured alternately in the same job, the 3 input shapes were 6.65% / 6.72% / 4.19% faster
  (combined +5.85%; removing it alone from a version stacked with two other changes was 3.90% slower; noise 0.5–0.9%). Landing this one line alone after certification, cross-job whole kernel 0.0365 → 0.0342 / 0.0343 ms.
  Before the change, in that kernel's stall ledger the `IMAD.WIDE` row consuming the smem index held 66% of long_scoreboard.
