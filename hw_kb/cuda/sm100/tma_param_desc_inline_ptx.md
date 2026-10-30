# tma_param_desc_inline_ptx (when issuing tensor-map TMA from inline PTX, `[tmap]` takes the address of a `__grid_constant__` kernel parameter: no gmem copy, no tensormap fence)

> **Certification source: CUDA-mode GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-ctrl2d`, `tma-gmemdesc`, `tma-ptx-param-desc`).**
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

The tensor-map operand of `cp.async.bulk.tensor` is the **generic address** of the 128 B object residing in the `.param` / `.const` / `.global` state space.
A host-encoded `CUtensorMap` passed into the kernel as a `const __grid_constant__ CUtensorMap` parameter —
`reinterpret_cast<uint64_t>(&tm)` is exactly that address: written before launch, unchanged for the whole launch, it needs **no** tensormap
proxy fence. The other route is to `cudaMemcpy` the descriptor into a piece of gmem and, in the kernel, first do
`fence.proxy.tensormap::generic.acquire.gpu [p], 128;` before using it — one more copy to manage and one more fence, needed only when the descriptor must be
rewritten on the device or there is one per CTA.

## When needed (trigger)

- Writing inline-PTX TMA and needing the `[tmap]` address; or your PTX has `fence.proxy.tensormap` while the descriptor is actually fixed before
  launch.
- The descriptor does not change during the whole launch. To modify fields on the device (`tensormap.replace`) or to have one per CTA, take the gmem route with the
  acquire fence.

## Minimal runnable example

```cpp
#include <cuda.h>
extern "C" __global__ void k(const __grid_constant__ CUtensorMap tm /* must be const + __grid_constant__ */, ...) {
  const uint64_t tmap = reinterpret_cast<uint64_t>(&tm);     // PTX: mov.u64 + cvta.param.u64
  if (threadIdx.x == 0) asm volatile("prefetch.tensormap [%0];" :: "l"(tmap) : "memory");  // optional
  // ... mbarrier init / arm ...
  // cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dst], [%tmap, {c0, c1}], [mbar];
}
// host: CUtensorMap tm; cuTensorMapEncodeTiled(&tm, ...); k<<<g, b, smem>>>(tm, ...);   // passed by value

// Control: descriptor in gmem (use only when it must be rewritten on the device)
extern "C" __global__ void k_gmem(const CUtensorMap* tmg, ...) {
  if (threadIdx.x == 0)
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(tmg) : "memory");
  // afterwards [tmap] = (uint64_t)tmg
}
```

## Gotchas

1. **Without `__grid_constant__` it is no longer a param-space address.** When an ordinary by-value parameter has its address taken, the compiler copies it into local memory,
   and `&tm` becomes a local address. The probe's `nogc` cell (the same kernel without `__grid_constant__`): the launch reports `an illegal memory access was encountered` (the PTX shows `st.local` ×8 and `cvta.local`: the descriptor was copied into local memory, and TMA got a local address).
2. **`__grid_constant__` requires the parameter to be `const`-qualified**, and the kernel must not write it.
3. **Do not mix the two routes**: after switching to the parameter address, delete the gmem slot, the copy and the tensormap fence together; accept by the PTX:
   probe readings — the `__grid_constant__` parameter route: `fence.proxy.tensormap` 0 occurrences, `cvta.param` 1, `prefetch.tensormap` 1; the gmem copy route: `fence.proxy.tensormap` 1, `cvta.param` 0. In SASS the gmem route has one extra `UTMACCTL.IV` and one extra `UTMACMDFLUSH` (the parameter route has only the prefetch's `UTMACCTL.PF`).
4. **The numerics cannot show which copy of the descriptor the hardware read** (the two routes are bitwise identical); only the PTX definition chain (`cvta.param`) and the fence count can.

## Sources

- PTX ISA, `cp.async.bulk.tensor` (state space and generic addressing of the tensor-map operand), `prefetch.tensormap`,
  `fence.proxy.tensormap`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide, `__grid_constant__` and "Using TMA to transfer multi-dimensional arrays" (`CUtensorMap`
  passed in as a `const __grid_constant__` parameter): <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA Driver API, `cuTensorMapEncodeTiled`: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- GPU-certification probe: `PROBE PASS max_diff=0.0`: the `ctrl2d` (parameter address) and `gmemdesc` (gmem copy + acquire fence) cells land bitwise identical bytes; PTX contract `tma-ptx-param-desc` in gotcha 3; the `nogc` cell records facts only (gotcha 1).
