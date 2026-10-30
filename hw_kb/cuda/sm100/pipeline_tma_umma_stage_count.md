# pipeline_tma_umma_stage_count (the depth S of an operand ring: when to deepen it, how deep, the smem cap)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> The certification uses a ring with a **1D `cp.async.bulk` producer + generic-load consumers** (16 KB per stage, 148 CTAs streaming
> 310 MB from DRAM), proving the "depth hides producer latency" curve and the per-CTA smem cap. **Uncertified**: the same curve with tensor-map TMA +
> a tcgen05.mma consumer side, the 2-CTA pair version, the best depth for any specific GEMM.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For how to write the ring itself see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md).

## Purpose

In an S-stage operand ring, the depth decides how many beats the producer (TMA / bulk copy) can lead the consumer (MMA / compute). Each extra stage costs
one stage of operand bytes + two 8-byte mbarriers; whether to add stages, and how many, depends on "is the producer really stuck on empty".

Measured on B200 (148 CTAs × (1 producer warp + 4 consumer warps), 16 KB per stage, 128 stages per CTA, median of 15 in the same job):

| S | Dynamic smem | Time | Effective bandwidth |
|---|---|---|---|
| 1 | 16 KB | 150.0 µs | 2069 GB/s |
| 2 | 32 KB | 86.8 µs | 3576 GB/s |
| 3 | 48 KB | 68.3 µs | 4547 GB/s |
| 4 | 64 KB | **64.0 µs** | 4852 GB/s |
| 6 | 96 KB | **63.4 µs** | 4894 GB/s |
| 8 | 128 KB | 64.2 µs | 4833 GB/s |
| 12 | 192 KB | 68.3 µs | 4545 GB/s |

That is: it saturates at about 64 KB in flight per SM (4 stages × 16 KB); deeper is no faster, and S=12 is even slightly slower.
**The depth's gain curve is "until saturation", not "the deeper the better".**

## When needed (trigger)

- The producer really is stuck on this ring's empty: high `long_scoreboard` in the stall ledger, compute / tensor metrics not saturated,
  consumers waiting on full in the timeline. Then estimate the depth by "bytes in flight ≈ bandwidth × latency", and A/B the candidate depths in the same job.
- You want to free smem for other uses (larger tiles, epilogue buffers): stages beyond the saturation point can be taken back.
- A 1 CTA/SM kernel: going shallower **does not buy** occupancy, it only saves smem.

## Minimal runnable example

```cpp
// Taken from the certification probe: the depth is a template constant and every derived quantity is computed from it
constexpr int STAGE_BYTES = 16384;
constexpr int NCONS = 128;

template <int S>
__global__ void __launch_bounds__(NCONS + 32, 1) k_stream(const int4* __restrict__ src, long long* out, int K) {
  extern __shared__ __align__(1024) unsigned char dsmem[];   // S * STAGE_BYTES bytes of operand ring
  __shared__ __align__(8) uint64_t full[S];                  // barrier table length follows S: 2*S of them
  __shared__ __align__(8) uint64_t empty[S];
  // ... init full(count 1) / empty(count NCONS), fence.mbarrier_init, __syncthreads ...
  // producer (1 thread): wait empty[st] (phase^1) -> arrive.expect_tx(full[st], STAGE_BYTES)
  //                      -> cp.async.bulk one whole stage -> advance (st, phase)
  // consumers (128 threads): wait full[st] (phase) -> read one whole stage -> arrive empty[st] -> advance
}

template <int S>
int launch(const int4* src, long long* out, int K, int grid) {
  const int dyn = S * STAGE_BYTES;                            // smem bytes are derived from S too
  cudaError_t e = cudaFuncSetAttribute(k_stream<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
  if (e != cudaSuccess) return (int)e;                        // beyond the cap it is refused right here
  k_stream<S><<<grid, NCONS + 32, dyn>>>(src, out, K);
  return (int)cudaGetLastError();
}

// per-CTA cap: cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev)
```

## Gotchas

1. **The per-CTA cap is 232448 B (227 KB), dynamic + static combined.** Measured
   `cudaDevAttrMaxSharedMemoryPerBlockOptin = 232448`; `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`
   accepts 232448 and refuses 232449; a launch using all 232448 bytes of dynamic smem works. Beyond 48 KB this attribute must be set first,
   otherwise the launch fails outright. Count candidate depths down from "the deepest that fits".
2. **Static smem draws on the same budget, and may be larger than you compute**: the S=12 kernel declares only a 192 B barrier table,
   yet `cudaFuncGetAttributes().sharedSizeBytes` reads 1024 B. Compute the headroom from this reading, not by hand.
3. **Write every derived quantity as a function of S** (ring bytes, barrier table length 2·S, the smem attribute, byte self-checks), so a depth change is
   one line; the `expect_tx` byte count of one stage is independent of S and must not change with it.
4. **Two rings, two tables.** Merging the barriers of the operand ring and another ring (e.g. accumulator double buffering) into one table with hand-cut offsets
   hard-codes the depth; when the table is one slot short, the two rings share one 8-byte word, with no numerical symptom, only intermittent hangs or stale reads.
5. **Deepening beyond saturation is a negative gain**: measured S=12 is 7.7% slower than S=6 (68.3 vs 63.4 µs). Depth is not "as deep as fits".
6. **Differences between adjacent depths must be measured as several variants in the same job**: S=4/6/8 are within 1.3%, which cross-job node differences can swamp.

## Sources

- CUDA C++ Programming Guide: *Shared Memory* (dynamic smem beyond 48 KB needs a `cudaFuncSetAttribute`
  opt-in), the per-block smem cap of Compute Capability 10.0, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Runtime API: `cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)`,
  `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`, `cudaFuncGetAttributes`,
  https://docs.nvidia.com/cuda/cuda-runtime-api/
- PTX ISA: *cp.async.bulk*, *mbarrier*, https://docs.nvidia.com/cuda/parallel-thread-execution/
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 pipeline_tma_umma_stage_count/D1-exact-all-depths`,
  `PROBE PASS max_diff=n/a pipeline_tma_umma_stage_count/D2-optin-smem-ceiling optin=232448 set_optin=0 set_optin_plus1=1 launch_at_optin=0 static_smem_S12=1024`,
  `PROBE PASS max_diff=n/a pipeline_tma_umma_stage_count/D3-depth-hides-producer-latency t1_over_best=2.36`.
