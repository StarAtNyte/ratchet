# launch_identity_stamp (launch identity stamp: have the kernel itself report "which instance is running, launched with what configuration")

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS; the probe is not included in this release).**
> This is an **instrumentation recipe**, not a performance lever: it makes no kernel faster; it only turns "what the host thinks it launched" and "what actually ran on the GPU"
> into a table that can be compared cell by cell. Along the way it certifies the semantics of the cluster-related special registers (`%nclusterid` ≠ `%cluster_nctaid`)
> and the reading of `%dynamic_smem_size`.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

A templated kernel (`template<int STAGES, ...>`) + host dispatch that picks the instance at runtime + the cluster /
dynamic smem configuration of `cudaLaunchKernelEx` — if any of the three layers differs from what you think, the result may still be numerically right, only **what is measured is not the variant you meant to measure**
(e.g. A/B variants actually running the same instance). The identity stamp has each CTA write **the constants compiled into the executable** and **the special registers at launch**
into a small gmem table prefilled with −1, and the host compares it cell by cell against the expected table.

## When needed (trigger)

- An A/B comparison reads "almost exactly the same", and you suspect both variants dispatched to the same template instance / the same cubin.
- You changed compile-time / launch-time parameters such as stage count, tile shape, cluster shape or dynamic smem bytes and want to confirm they really took effect.
- Writing a cluster kernel and unsure which register gives "the number of clusters" and which "the cluster shape".

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
#define SREG(name, out) asm volatile("mov.u32 %0, %%" name ";" : "=r"(out))

template <int VARIANT, int STAGES>              // stamp compile-time constants: they are hard-coded in the executable, not intentions passed in
__global__ void __launch_bounds__(128) k_stamp(int* tab) {   // tab: [ROWS, 16], prefilled -1 by the host
  extern __shared__ __align__(16) unsigned char dsmem[];
  if (threadIdx.x != 0) return;
  uint32_t ncx, ntx, cnx, cny, cnz, cnr, crank, nclx, ncly, nclz, dyn;
  SREG("nctaid.x", ncx);  SREG("ntid.x", ntx);
  SREG("cluster_nctaid.x", cnx); SREG("cluster_nctaid.y", cny); SREG("cluster_nctaid.z", cnz); // cluster shape
  SREG("cluster_nctarank", cnr);                 // CTAs per cluster
  SREG("cluster_ctarank", crank);                // this CTA's rank within the cluster
  SREG("nclusterid.x", nclx); SREG("nclusterid.y", ncly); SREG("nclusterid.z", nclz);           // number of clusters
  SREG("dynamic_smem_size", dyn);                // dynamic smem bytes of this launch
  const int lin = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  int* r = tab + lin * 16;                       // one row per linear CTA id: rows never written stay -1 = grid assertion
  r[0] = VARIANT; r[1] = STAGES; r[2] = (int)ncx; r[3] = (int)ntx;
  r[4] = (int)cnx; r[5] = (int)cny; r[6] = (int)cnz; r[7] = (int)cnr; r[8] = (int)crank;
  r[9] = (int)nclx; r[10] = (int)ncly; r[11] = (int)nclz; r[12] = (int)dyn; r[13] = 0x5EED;
}

// host: cluster + dynamic smem launch
auto kern = k_stamp<1, 7>;
cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);  // > 48 KiB must be enabled first
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3(1, 96, 1); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 65536; cfg.stream = 0;
cudaLaunchAttribute attr[1];
attr[0].id = cudaLaunchAttributeClusterDimension;
attr[0].val.clusterDim.x = 1; attr[0].val.clusterDim.y = 8; attr[0].val.clusterDim.z = 1;
cfg.attrs = attr; cfg.numAttrs = 1;
cudaLaunchKernelEx(&cfg, kern, tab_ptr);
```

The expected table lives in the host test script (not in the module under test) and is compared cell by cell; comparing one variant's output against another variant's expected table
must be judged a mismatch — that proves the table itself discriminates.

## Gotchas

1. **`%nclusterid` is "the number of clusters", `%cluster_nctaid` is "the cluster shape".** Measured with grid (1,96,1), cluster (1,8,1):
   `%cluster_nctaid` = (1,8,1), `%nclusterid` = (1,12,1), `%cluster_nctarank` = 8; grid (60,1,1), cluster (2,1,1):
   `%cluster_nctaid` = (2,1,1), `%nclusterid` = (30,1,1). The two names differ by one word; swapping them reads "8" as "12".
   `%cluster_ctarank` numbers CTAs in linear order within the cluster (in the two examples above it equals `blockIdx.y % 8` and `blockIdx.x % 2` respectively).
2. **Stamp compile-time constants, not runtime arguments.** A `stages` argument passed into the kernel only reports the host's intention; template parameters / `constexpr`
   are the values actually compiled into this executable.
3. **`%dynamic_smem_size` equals the launch's `dynamicSmemBytes` exactly.** Measured 65536, 40960, and even an unaligned 1000 B reads back 1000
   (identical across all CTAs of a launch). Beyond 48 KiB you must first call `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, n)`,
   otherwise the launch fails.
4. **One row per linear CTA id + a −1 sentinel = a free grid assertion.** Give the table a few more rows than the grid; the extra rows must still be −1, and rows left unwritten also
   stay −1, so both kinds of grid error are caught by the cell-by-cell comparison.
5. **A cluster launch requires each grid dimension to be divisible by the cluster shape**, and the cluster shape can only be given via
   `cudaLaunchAttributeClusterDimension` of `cudaLaunchKernelEx` (or compile-time `__cluster_dims__`); the `<<<>>>` syntax has no such parameter.
6. **Do not stamp timing runs.** Compile it out with a template switch (`if constexpr (STAMP)`); use the stamped run only to check identity.

## Sources

- PTX ISA: *Special Registers* (`%nctaid`, `%ntid`, `%cluster_nctaid`, `%cluster_nctarank`, `%cluster_ctarank`,
  `%nclusterid`, `%dynamic_smem_size`) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Runtime API: `cudaLaunchKernelEx`, `cudaLaunchConfig_t`, `cudaLaunchAttributeClusterDimension`,
  `cudaFuncSetAttribute` / `cudaFuncAttributeMaxDynamicSharedMemorySize` — https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
- CUDA C++ Programming Guide: Thread Block Clusters — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 stamp/V1-rows` (96 rows exact cell by cell, extra rows at the end of the table still −1),
  `PROBE PASS max_diff=0.0 stamp/V2-rows` (60 rows), `PROBE PASS stamp/gate-catches`,
  `PROBE PASS stamp/cluster-sregs V1_cluster_nctaid=[1, 8, 1] V1_nclusterid=[1, 12, 1] V2_cluster_nctaid=[2, 1, 1] V2_nclusterid=[30, 1, 1]`,
  `PROBE PASS stamp/dyn-smem-exact V1=[65536] V2=[40960]`; fact line: a 1000 B launch reads back 1000.
