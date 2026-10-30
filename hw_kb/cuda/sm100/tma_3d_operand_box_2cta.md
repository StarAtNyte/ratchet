# tma_3d_operand_box_2cta (a stage spanning n SW128 atoms along the contiguous dimension: a rank-2 map needs one `UTMALDG.2D` per atom; cutting the contiguous dimension into a `(64, n)` rank-3 map moves it all with one `UTMALDG.3D`)

> **Certification source: CUDA-mode GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-ctrl3d`, `tma-pair3d`, `tma-sass-rank-one-issue`, `tma-box-limit`).**
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the ≤ 256 elements per box dimension limit and flat staging areas see
> [`tma_flat_restage_box_rank`](tma_flat_restage_box_rank.md); for pair accounting see
> [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md).

## Purpose

An SW128 tensor map requires **box inner dimension × element bytes ≤ 128 B** (64 elements for 16-bit types, = one swizzle atom).
So when a stage's operand spans n atoms along the contiguous dimension (K for K-major) (`BK = 64·n`), a rank-2 map can only issue one per atom.
Cut the contiguous dimension into two in the descriptor — dim0 = the 64 elements within an atom (contiguous), dim1 = rows, dim2 = which atom (stride 128 B) —
write the box as `{64, rows, n}`, and **one** `cp.async.bulk.tensor.3d` moves the whole stage; the bytes landing in smem are bitwise identical to n rank-2
loads (destination addresses `rows × 128 B` apart): the box is laid into smem in dim0 → dim1 → dim2 order, which is exactly
the face "n atoms stacked atom by atom". In CUDA the rank is an explicit parameter of `cuTensorMapEncodeTiled`, and the PTX rank suffix
(`.2d` / `.3d`) must match it.

## When needed (trigger)

- The contract / design table requires **one** `UTMALDG.3D.*` per operand per stage, while your SASS has n
  `UTMALDG.2D.*` per operand per stage (`BK` spans n SW128 atoms).
- You want to open the SW128 box inner dimension to 128 16-bit elements (256 B): the encoder refuses it outright (gotcha 1); this entry's cut is then the way.
- **Anti-trigger**: using it as a feed-side speed-up. The byte count does not change; only the producer's per-stage issue count does. Issue-side readings: a pair producer merging each operand's multiple rank-2 loads
  into one rank-3 (halving the loads per beat) took the wrap segment (wait returns → next wait starts) from 210.9 → 122.4 cycles (see
  [`mbarrier_ring_unrolled_const_addr`](mbarrier_ring_unrolled_const_addr.md)), but the whole kernel gained nothing: on a pair GEMM, gate 0.0136 → 0.0136 ms (flat);
  on another single-CTA tcgen05 GEMM, a same-job A/B had rank-3 1.00% slower than rank-2; on a third kernel, cross-job +0.85% —
  when the producer's issue count is not on the critical path, saving half of the TMA instructions buys no time.

## Minimal runnable example

```cpp
// X: R rows x K columns of 16-bit, K contiguous (row pitch K*2 B). BK = 128 = two SW128 atoms.
// rank-2 (control): dims {K, R}, strides {K*2}, box {64, 64} -> one .2d per atom per stage
// rank-3 (this entry): dims {64, R, K/64}, strides {K*2 (row), 128 (next atom)}, box {64, 64, 2}
cuuint64_t dims[3]    = {64, (cuuint64_t)R, (cuuint64_t)(K / 64)};
cuuint64_t strides[2] = {(cuuint64_t)K * 2, 128};      // dim2's stride may be smaller than dim1's
cuuint32_t box[3]     = {64, 64, 2};
cuuint32_t estr[3]    = {1, 1, 1};
cuTensorMapEncodeTiled(&tm3, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box, estr,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

__device__ __forceinline__ void tma3d(uint32_t dst, uint64_t tmap, int c0, int c1, int c2, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4}], [%5];"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(c2), "r"(mbar) : "memory");
}
// one stage: coordinates {0, row0, k_stage * 2}; expect_tx is still 64 x 128 x 2 B = 16 KB (same as two .2d)
tma3d(stage_base /*1024 B aligned*/, (uint64_t)&tm3, 0, row0, kt * 2, full_bar);
// pair pipeline: add .cta_group::2 to the string (-> UTMALDG.3D.2CTA); the accounting rules are unchanged
```

## Gotchas

1. **Under SW128 the box inner dimension is at most 128 B.** Encoder readings from the probe: SW128 + 16-bit, box `{64, 8}` accepted, `{128, 8}` (256 B) returns `CUDA_ERROR_INVALID_VALUE`. To move two atoms with one load you can only cut the dimension, not widen the box.
2. **The PTX rank suffix must match the map's rank**: a rank-3 map goes with `.3d` and three coordinates.
3. **Neither the smem face nor `expect_tx` changes.** The order in which a rank-3 box is laid into smem is exactly n atoms stacked atom by atom, and the byte count is the same;
   in the probe one rank-3 load and two rank-2 loads land bitwise identical bytes.
4. **The cut sub-dimension = the number of elements of one swizzle atom along the contiguous dimension**: 64 for SW128 + 16-bit; 32 / 16 for SW64 / SW32 (uncertified).
5. **Accept by SASS counts and suffixes**: two rank-2 loads read `UTMALDG.2D` ×2, one rank-3 load reads `UTMALDG.3D` ×1; the pair version reads `UTMALDG.3D.2CTA` ×1. The numerics cannot show the rank.
6. **When accepting by count, first confirm the unroll factor**: if the producer loop is unrolled, divide the static count by the unroll factor.

## Sources

- CUDA Driver API, `cuTensorMapEncodeTiled` (tensorRank, boxDim, globalStrides; box inner dimension ≤ 128 B with SWIZZLE_128B):
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- PTX ISA, `cp.async.bulk.tensor` (`.1d`–`.5d` matching the tensor map's rank):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe: `PROBE PASS max_diff=0.0`: for the same 64-row x 128-column 16-bit tile (two SW128 atoms), `ctrl2d` (two rank-2 loads), `ctrl3d` (map dims {64, 256, 2}, strides {256 B, 128 B}, box {64, 64, 2}, one load) and `pair3d` (the same plus `.cta_group::2`, delivered to the leader) all land bytes equal to the SW128 expectation; SASS contract `tma-sass-rank-one-issue` and encoder contract `tma-box-limit` passed in the same round.
- Issue-side readings: a separate GPU-certification probe of the producer issue segment —
  `PROBE PASS max_diff=n/a tma_producer_issue_segment/R3` (wrap segment 210.9 → 122.4 with the load count halved).
