# smem_sw64_row_face_store_width (SWIZZLE_64B staging face with lane = row and a 64 B row pitch: each lane must write a full 16 B per store; narrow stores give 4-way conflicts)

> One file per hardware interface. Operator-agnostic; mechanism only. The language surface is CUDA C++ / inline PTX; the address algebra does not depend on the frontend.
>
> **Status**: ✅ Verified (certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0`, certification round 1; this entry uses its contracts C3 (the face bytes written by both
> `st.shared.u32` and `st.shared.v4.b32` equal the swizzle image computed by numpy, max_diff 0.0) and C4 (NCU `bankconf_st`: 144 for u32 stores, 0 for v4 stores)).
>
> **Certified surface**: 4 warps each write one 2048 B face (32 rows × 64 B), face 1024 B aligned; only write-side conflicts and the bytes written were measured — no TMA store attached, no speed measured.

## Purpose

A staging face of 16-bit data laid out as **lane = row, 64 B row pitch, `SWIZZLE_64B`** (32 16-bit elements per row, exactly one 64 B TMA box row),
handed to a TMA store once written. On such a face **every lane must write a full 16 B chunk with each smem store** (`st.shared.v4.b32`):
per-lane 32-bit narrow stores have 4-way bank conflicts, one 16 B store has none; both forms write exactly the same bytes, and the TMA descriptor needs no change.

## When needed (trigger)

- The staging face's write-side `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` is clearly higher than expected, and the store pattern is one row per lane, one `st.shared.u32`
  (or narrower) per store.
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st` / store instruction count ≈ 4, while you assumed each narrow store is 1 wavefront.

## Minimal runnable example

Same pattern as the certification probe (`sw64` and the two store loops). `w[]` holds the 16 already-packed 32-bit words of this row (32 16-bit elements).

```cuda
// SWIZZLE_64B: the 16 B chunk index (address bits [4,6)) is XORed with address bits [7,9)
__device__ __forceinline__ uint32_t sw64(uint32_t off) { return off ^ (((off >> 7) & 3u) << 4); }

// face: shared address of this warp's staging face (__cvta_generic_to_shared), 1024 B aligned; lane = row, 64 B row pitch
// Narrow stores (4-way conflicts): 16 stores of 4 B per lane
#pragma unroll
for (uint32_t j = 0; j < 16; ++j)
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(face + sw64(lane * 64 + j * 4)), "r"(w[j]) : "memory");

// 16 B stores (conflict-free): 4 stores of 16 B per lane, writing exactly the same bytes as above
#pragma unroll
for (uint32_t c = 0; c < 4; ++c)
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(face + sw64(lane * 64 + c * 16)),
                    "r"(w[4*c]), "r"(w[4*c+1]), "r"(w[4*c+2]), "r"(w[4*c+3]) : "memory");
```

Acceptance: write-side `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` is 0; the face bytes are bit-identical to before the change (or the downstream TMA store's result is unchanged).

Probe readings (4 warps, each writing one 2048 B face):

| Store form | `bankconf_st` | smem st wavefronts / store instructions |
|---|---|---|
| `st.shared.u32` | 144 | 208 / 52 (4 each) |
| `st.shared.v4.b32` | **0** | 64 / 16 (4 each = the ideal for a 16 B store) |

## Gotchas

1. **SW64 only permutes 16 B chunks; it does nothing to the words inside a chunk.** With lane = row and a 64 B row pitch, one row covers only 16 banks; an even and an odd row together make the 32
   banks. The XOR mask comes from address bits [7,9), i.e. `(lane >> 1) & 3`, and only permutes among 4 chunks. The bank of the j-th warp-wide `st.shared.u32`
   = `16·(lane & 1) + (j ⊕ 4·((lane >> 1) & 3))`, which takes only 2 × 4 = 8 distinct values; spread over 32 lanes that is a 4-way conflict
   (probe reading: 4 wavefronts per store, see the table above).
   With 16 B stores, every 8 lanes form one 128 B phase, and `(lane & 1, (lane >> 1) & 3)` tiles exactly the 8 16 B slots, so there is no conflict.
2. **Generalisation (derived from the address algebra above; the probe only tested SW64)**: swizzle permutes at 16 B granularity, so with lane = row each lane must write a full 16 B per store
   for zero conflicts. To keep narrow stores you would have to make lanes map to columns rather than rows within one instruction, which first requires a transpose of the data across lanes.
3. **v4 stores require 16 B address alignment.** `sw64` only flips bits [4,6), so it does not break 16 B alignment; the face base alignment must match the phase in which TMA interprets the swizzle — the probe used
   1024 B alignment (another certification of the same requirement: [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md) gotcha 5).
   Lower alignments were not tested by the probe.
4. **A lower conflict reading does not mean the whole kernel got faster.** On the full kernel where this was found, switching to 16 B stores (together with a change in the same package that removed local memory) cut the kernel's
   `bankconf_st` to about 1/3 of before (the remaining write-side conflicts came from f32 staging stores that were not changed), yet device time moved by only −1.2%, inside the loop's 1.5% noise band.

## Sources

- CUDA C++ Programming Guide, Shared Memory / bank conflicts section (32 banks of 4 B each; 128-bit accesses are split into quarter-warp phases):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x
- PTX ISA: the vector form `.v4` of `st.shared` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- TMA swizzle mode: `CU_TENSOR_MAP_SWIZZLE_64B` (CUDA Driver API `cuTensorMapEncodeTiled`).
- NCU metrics: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`,
  `smsp__sass_inst_executed_op_shared_st.sum`.
- Certification probe: `PROBE PASS max_diff=0.0`
  (round 1, C3 / C4). C1 / C2 of the same probe: see [`rmem_array_runtime_index_local_frame`](rmem_array_runtime_index_local_frame.md).
- Related mainline entry: [`umma_smem_operand_sts128_publish`](umma_smem_operand_sts128_publish.md) (address algebra of hand-written `STS.128` on a K-major UMMA operand face:
  swizzle only XORs at 16 B granularity within a row).
