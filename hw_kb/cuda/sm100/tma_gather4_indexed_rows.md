# tma_gather4_indexed_rows (`.tile::gather4` of tensor-map TMA: one instruction moves 4 rows at 4 arbitrary row coordinates into an SW128 face)

> **Certification source: CUDA-mode GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-g4`, `tma-g4oob`, `tma-g4pair`, `tma-sass-gather4`).**
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

The feed is a row-by-row gather by an index table (row numbers come from another table, rows are not contiguous in gmem), and the destination is an SW128 operand face the MMA reads directly.
A plain `.tile` box can only frame contiguous rows; `.tile::gather4` carries **1 column coordinate + 4 unrelated row coordinates** in one instruction, landing 4 rows
(each one box wide, 64 16-bit elements = 128 B, 512 B in total) into the 4 128 B rows starting at the destination address according to the descriptor's swizzle,
with completion booked into the mbarrier's tx-count as usual. Same bytes, 1/4 of the issue count of "one box per row".

## When needed (trigger)

- The feed is an indirect row-by-row gather, each row a contiguous run of bytes, and the destination face is a swizzled MMA operand face.
- The one-row-box-per-row spelling has a single issuing thread serially issuing hundreds per round.
- Already writing this PTX and stuck on: how many rows the box has / coordinate order / whose barrier books the pair's bytes / whether out-of-range rows hang.
- **Anti-trigger**: the rows are contiguous anyway; use a plain `.tile` multi-row box.

## Minimal runnable example

```cpp
// host: rank-2 map, dim0 = contiguous dimension (D elements), dim1 = row number; the box is **one row**: {64, 1}
cuuint64_t dims[2] = {D, NROWS}, strides[1] = {(cuuint64_t)D * 2};
cuuint32_t box[2] = {64, 1}, estr[2] = {1, 1};
cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box, estr,
                       CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                       CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

__device__ __forceinline__ void gather4(uint32_t dst, uint64_t tmap, int c0, int r0, int r1, int r2, int r3,
                                        uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(mbar) : "memory");
}
// face: NRUN segments (64 columns each) x NSLOT slots (one slot = one 128 B row); slot s / segment c at (c*NSLOT + s)*128 B (base 1024 B aligned)
if (threadIdx.x == 0) {                                   // raw PTX has no built-in election: one thread issues
  mbar_arrive_expect_tx(bar, NRUN * NSLOT * 128);         // 512 B each; out-of-range rows are counted too
  for (int c = 0; c < NRUN; ++c)
    for (int b = 0; b < NSLOT / 4; ++b)
      gather4(face + (c * NSLOT + 4 * b) * 128,           // linear (unswizzled) destination offset
              (uint64_t)&tm, c * 64, idx[4*b], idx[4*b+1], idx[4*b+2], idx[4*b+3], bar);
}
// pair (cluster (2,1,1)): append .cta_group::2 to the string, the barrier is mapa(bar, 0), only the leader arms the sum of both CTAs
```

## Gotchas

1. **The box must be one row `{64, 1}`.** Each of the four row coordinates moves this one-row box once. The probe's `g4box4` cell (box `{64, 4}`, same
   instruction): the launch reports `an illegal instruction was encountered`. One map cannot serve both `.tile` (multi-row box) and `.tile::gather4`.
2. **The destination address is the face's linear offset; the swizzle is derived by the descriptor from the destination address.** The 4 rows of one gather4 cover only half of the SW128
   8-row cycle, and 4 rows landing in the second half are just as exact — do not apply the swizzle again yourself. When the descriptor has no swizzle the rows land linearly
   (probe cell `g4noswz`: 4 rows land byte by byte in the linear layout), which does not match the SW128 face's address function.
3. **Out-of-range row coordinates raise no error, land zeros and are still booked.** Probe cell `g4oob` (row coordinates −1, NROWS, 100000, including one instruction with all four rows out of range):
   out-of-range rows are all 0, the rest bytewise exact, no fault, the barrier flips as usual (tx counted in full). So sentinel indices can be issued as they are, and `expect_tx` is still 512 B per instruction; do not change the issue count to skip invalid rows.
4. **Three things for pair accounting**: each instruction carries `.cta_group::2` with the barrier pointing at the leader (`mapa(bar, 0)`); only the leader arms, bytes =
   the sum of both CTAs; the arm is visible before the peer's first delivery (one cluster rendezvous after init + arm). Probe cell `g4pair`: both CTAs' 48 rows x 512 columns bytewise exact, the leader flips.
5. **Only one thread in a warp may execute this asm** (otherwise 32× the deliveries).
6. **Issue is a single-thread serial cost**: probe reading: one thread issuing 96 gather4 back to back takes about 12900 SM cycles (about 134 cycles each) (just a reading, not a pricing).

## Sources

- PTX ISA, `cp.async.bulk.tensor` (the `.tile::gather4` load mode, `.cta_group`, `.mbarrier::complete_tx::bytes`):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Driver API, `cuTensorMapEncodeTiled`: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- GPU-certification probe: `PROBE PASS max_diff=0.0`: 48 index rows (including one duplicate) x 512 bf16 columns, an SW128 face of 8 segments x 48 slots, 96 gather4; `g4`, `g4oob`, `g4pair` bytewise exact; SASS reads `UTMALDG.2D.GATHER4` (pair version `UTMALDG.2D.GATHER4.2CTA`). `g4box4` / `g4noswz` record facts only.
