# tma_flat_restage_box_rank (at most 256 elements per box dimension: a contiguous run longer than 256 elements is split into two dimensions `(256, run/256)`; the address function is unchanged, the descriptor gains one dimension)

> **Certification source: CUDA-mode GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-flat-run-rank3`, `tma-box-limit`, `tma-sass-rank-one-issue`).**
>
> One file per hardware interface. Operator-agnostic: mechanism only. For cutting a dimension when a stage spans several swizzle atoms see
> [`tma_3d_operand_box_2cta`](tma_3d_operand_box_2cta.md).

## Purpose

When the smem byte order a downstream instruction wants is not the image of any box of the original tensor, one delivery shape is: rearrange the bytes once, by the target address function, into a
**flat gmem staging area**, after which each stage's delivery degenerates into "a trivial copy of one contiguous run". The only descriptor
constraint that then shows up is **`boxDim` ≤ 256 per dimension (counted in elements, not bytes)**. If the run is longer, split it into two dimensions `(256, run/256)`: dim0 = 256
contiguous elements, dim1 stride = 256 × element bytes, box `{256, run/256}` — the address function `256·j + i` is bytewise identical to "one-dimensional run, stride 1",
at the cost of one more map dimension (rank 3 when the flat area is further indexed by block; SASS reads `UTMALDG.3D`).

## When needed (trigger)

- You wrote a flat descriptor and want some box dimension > 256: `cuTensorMapEncodeTiled` returns an error outright (gotcha 1).
- The current shape rearranges the target face in smem once per stage, with the rearrangement on the producer's critical path, and you want to switch to rearranging once into a gmem
  staging area and then delivering trivially with TMA (the gain belongs to that rearrangement; this entry certifies only the delivery step).
- You thought you built a 2D descriptor, but SASS shows `.3D` (or the other way round): the rank can be read back in SASS.

## Minimal runnable example

```cpp
// flat staging area: C1 blocks x C0 segments x RUN 16-bit elements, segment (c0, c1) starts at (c1*C0 + c0) * RUN. RUN = 2048.
constexpr int RUN = 2048, C0 = 4, C1 = 3;
static_assert(RUN % 256 == 0, "split factor must divide the run");
cuuint64_t dims[3]    = {256, (cuuint64_t)C0 * (RUN / 256), C1};
cuuint64_t strides[2] = {256 * 2, (cuuint64_t)RUN * C0 * 2};   // every non-innermost stride is a multiple of 16 B
cuuint32_t box[3]     = {256, RUN / 256, 1};                    // one load moves a whole run (4096 B)
cuuint32_t estr[3]    = {1, 1, 1};
cuTensorMapEncodeTiled(&tmf, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, scratch, dims, strides, box, estr,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// segment (c0, c1): coordinates {0, c0 * (RUN/256), c1} — the first is always 0 (the box covers the whole dimension)
//   cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
//       [dst], [tmap, {0, c0*8, c1}], [mbar];      expect_tx = RUN * 2
```

## Gotchas

1. **256 is elements, not bytes.** Encoder readings from the probe: without swizzle, a 16-bit box `{256, 8}` is accepted and `{512, 1}` refused; an 8-bit box `{256, 1}` is accepted and `{512, 1}` refused (512 B is refused as well, showing the limit is in elements, not bytes); a row dimension of 512 is likewise refused.
2. **The split needs exact strides.** Only `(256, run/256)` with strides `(1, 256)` elements is bytewise identical to the one-dimensional run; any other stride silently changes
   the address function — neither the encoder nor ptxas reports it; only the numbers go wrong.
3. **Each call site gets one extra coordinate that is always 0** (the box covers all of dim0).
4. **The rank is visible only in SASS**: the probe's flat staging-area map (rank 3) reads `UTMALDG.3D` ×1.
5. **The flat area's rearrangement must match the consumer's face byte for byte**: assert the premise "each stage is exactly one contiguous run" on the host side;
   do not assume it.

## Sources

- CUDA Driver API, `cuTensorMapEncodeTiled` (boxDim ≤ 256 and non-zero per dimension; globalStrides multiples of 16 B;
  `boxDim[0] × element bytes` a multiple of 16 B): <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- PTX ISA, tile mode and coordinate semantics of `cp.async.bulk.tensor`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe: `PROBE PASS max_diff=0.0`: a flat area with RUN = 2048, C0 = 4, C1 = 3; one `.3d` load of segment (2, 1) lands 4096 B bytewise identical to the source segment; encoder limits in gotcha 1 (contract `tma-box-limit`).
