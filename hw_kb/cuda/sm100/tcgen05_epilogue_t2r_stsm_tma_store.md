# tcgen05_epilogue_t2r_stsm_tma_store (assembling the whole path of the accumulator out of TMEM: `tcgen05.ld` → convert to bf16 in registers → smem staging face (SW128 / SW64 image) → `fence.proxy.async` → rendezvous → TMA store → bulk-group accounting)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the helper atom selection / partitioning errors there are DSL-specific, replaced here by hand-written addresses and PTX, and the readings were re-taken.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the main loop see [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md);
> for warp-private boxes with contiguous lanes (M-major) see [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md);
> for the conversion instruction see [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md);
> for the ordering edges between readback and fence / rendezvous see [`tmem_ld_epilogue_issue_order`](tmem_ld_epilogue_issue_order.md).

## Purpose

The accumulator lives in TMEM (128 lanes × 32-bit columns), and each of the four legs from TMEM to gmem has one hardware constraint:

| Leg | Hardware fact | CUDA / PTX spelling |
|---|---|---|
| ① TMEM → registers | `tcgen05.ld.32x32b.xN`: thread t (warp w) = lane `32*(w%4)+t`, gets N consecutive columns; `.16x256b`: m16n8 C fragment shape (see gotcha 5) | the destination registers may be used only after `tcgen05.wait::ld` |
| ② conversion | a pair of f32 → one bf16x2 word | `cvt.rn.bf16x2.f32 d, hi, lo` (the first source goes to the high half) |
| ③ registers → smem staging face | the TMA store reads a face **with the swizzle image**; N-major (C's columns contiguous): thread = one row of the face, one `st.shared.v4` per 16 B | face 1024 B aligned; SW128 image: the c-th 16 B block of row r is at `r*128 + ((c ^ (r%8)) * 16)` |
| ④ smem → gmem | `cp.async.bulk.tensor` goes through the async proxy; generic-proxy writes must first be fenced over; the store is a bulk async-group, not an mbarrier | every writing thread `fence.proxy.async.shared::cta` → rendezvous (`bar.sync id, 128`) → one thread issues the store + `cp.async.bulk.commit_group` → `cp.async.bulk.wait_group.read N` before reusing the face, `wait_group 0` before exit |

## When needed (trigger)

- The accumulator is already computed correctly (fp32 readback bitwise equal), and you want to switch to the real epilogue "convert to bf16 → staging face → TMA store";
- the TMA store writes out stale data / half a tile: missing `fence.proxy.async` or a missing rendezvous (gotcha 2);
- the 16 B blocks of each output row are permuted: the face's swizzle image disagrees with the tensor map's `CU_TENSOR_MAP_SWIZZLE_*` (gotcha 3);
- a staging stage gets overwritten / results are intermittently wrong: no `wait_group.read` before reusing the face (gotcha 4).

## Minimal runnable example

N-major output C (bf16, `[M][64]`), 4 readback warps, one `128 × 64` staging face per CTA (one SW128 atom row wide); the spelling matches the certification probe:

```cpp
// host: tensor map and face use the same swizzle
CUtensorMap mc = make_map_2d(dC, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, M, 64, /*box*/64, 128, CU_TENSOR_MAP_SWIZZLE_128B);

__device__ inline uint32_t sw128(uint32_t off) { return off ^ (((off >> 7) & 7u) << 4); }   // 16 B block number ^= row % 8

// kernel: face = 1024 B-aligned shared address; tbase = TMEM base; r = 32*warp + lane
uint32_t v[64];
tm_ld32x32b_x32(tbase + ((uint32_t)(32 * warp) << 16) + 0,  v);        // (1) two readbacks, then one wait
tm_ld32x32b_x32(tbase + ((uint32_t)(32 * warp) << 16) + 32, v + 32);
asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
uint32_t p[32];
for (int j = 0; j < 32; ++j)                                          // (2) even columns go into the low half
  p[j] = pack_bf16x2(__uint_as_float(v[2 * j]), __uint_as_float(v[2 * j + 1]));
for (int c = 0; c < 8; ++c)                                           // (3) 8 writes of 16 B = one row of 64 bf16
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(face + sw128(r * 128 + c * 16)), "r"(p[4*c]), "r"(p[4*c+1]), "r"(p[4*c+2]), "r"(p[4*c+3]) : "memory");
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");          // (4) every writing thread, after its last write
asm volatile("bar.sync 1, 128;" ::: "memory");                        //    all rows written
if (threadIdx.x == 0) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :: "l"(reinterpret_cast<uint64_t>(&mc)), "r"(face), "r"(0), "r"(m0) : "memory");   // coords = {column, row}
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");           // before exit / releasing the face: everything has landed
}
```

With a multi-block pipeline (S slots in one face), the thread that issues the store does `cp.async.bulk.wait_group.read N` after the commit (`N ≤ S − 2`, see
[`tmem_ld_epilogue_issue_order`](tmem_ld_epilogue_issue_order.md) gotcha 3); the other threads inherit slot safety from the next rendezvous.

## Gotchas

1. **The lane segment of a TMEM readback is set by the warp number mod 4**; the high 16 address bits are the lane, the low 16 the column: `tbase + ((32*(warp%4)) << 16) + col`.
2. **When the generic proxy writes smem and the async proxy (TMA) reads it, `fence.proxy.async.shared::cta` then a rendezvous are required in between.** Missing the fence = occasional stale data;
   missing the rendezvous = the store is issued before other warps have finished writing. One fence per thread per block, after the block's last write.
3. **The face's swizzle image must match the tensor map's swizzle**: SW128 = 16 B block number XOR address bits [7,10) (row % 8, row width 128 B);
   the row width inside the box must equal the swizzle width (SW128 → 128 B = 64 bf16). The probe also certified the "linear face + `CU_TENSOR_MAP_SWIZZLE_NONE`" setting:
   both settings write the same gmem; the only difference is write-side bank conflicts (gotcha 6).
4. **A TMA store is a bulk async-group**: `commit_group` / `wait_group.read N` (wait until smem has been read, the face can be reused) / `wait_group N` (wait until the writes are complete).
   `.read` does not wait for gmem to land, so the `wait_group 0` before exit cannot be omitted; bulk-group accounting is **per thread**, so wait only on the thread that issued the store.
5. **The thread-value map of `tcgen05.ld.16x256b` is the m16n8 C fragment shape**: register i lands at lane `+ t/4 + 8*((i>>1)&1)`, column `+ 2*(t%4) + (i&1) + 8*(i>>2)`
   (x2; the probe contract `C3-ld16x256b-map` checked this map register by register). This fragment shape is isomorphic to `stmatrix`, which is the precondition for an M-major (contiguous-lane) staging face
   to be written with matrix atoms; see [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md).
6. **A staging face without swizzle compiles, runs and is numerically all right — it just has write-side bank conflicts**: with thread = row and a 128 B row pitch, the threads within each of the 8
   phases of one `st.shared.v4` land on the same group of banks. The conflict amount is in the NCU fact lines under Sources; a missing swizzle raises no error; accept by `bankconf_st`.
7. **`tcgen05.wait::ld` before releasing the accumulator** (then `tcgen05.fence::before_thread_sync` + a sync), otherwise the next tile's MMA may overwrite columns not yet fully read.
8. **The coords order is `{innermost dimension, outer dimension}`**: `[tmap, {col, row}]`, consistent with the dims order of `cuTensorMapEncodeTiled`.

## Sources

- PTX ISA: `tcgen05.ld` (lane–column assignment of `.32x32b` / `.16x256b`), `tcgen05.wait::ld`, `cvt.rn.bf16x2.f32`, `st.shared`,
  `fence.proxy.async`, `cp.async.bulk.tensor` (`.global.shared::cta.bulk_group`, swizzle modes), `cp.async.bulk.commit_group` /
  `wait_group{.read}` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUDA driver `cuTensorMapEncodeTiled` (`CU_TENSOR_MAP_SWIZZLE_128B` / `_NONE`); CUTLASS `cute::SM90_TMA_STORE_2D`,
  `cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b32x` / `SM100_TMEM_LOAD_16dp256b2x` (same PTX spelling).
- GPU-certification probe verdict
  `PROBE PASS max_diff=0.0`: `C1-nmajor-sw128-tma-store` / `C1-nmajor-linear-tma-store` (two launches, bf16 output bitwise equal to the payload),
  `C3-ld16x256b-map` (1024 register slots, 0 mismatches). NCU fact lines (the same kernel with only the face image changed): SW128 face `bankconf_st` 0, smem write wavefronts 532;
  linear face `bankconf_st` 3584, wavefronts 4116; staging-write phase `%clock64` median 123 vs 436 cycles.
