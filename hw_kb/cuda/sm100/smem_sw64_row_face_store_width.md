# smem_sw64_row_face_store_width (a SWIZZLE_64B staging face with lane = row and a 64 B row pitch: each lane writes a full 16 B at a time; word-by-word narrow writes give a 4-way conflict)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cells `sw64 u32` / `sw64 v4`; the probe is not included in this release;
> verdicts and NCU readings under Sources). The same-named entry in the CuTe DSL tree was already in the CUDA spelling; the byte and conflict readings here were re-taken on this repo's CUDA probe.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

A staging face for 16-bit data laid out as **lane = row, 64 B row pitch, `SWIZZLE_64B`** (32 16-bit elements per row = one 64 B TMA box row), handed to a
TMA store once written. On such a face **every smem write of each lane must fill a whole 16 B block** (`st.shared.v4.b32`): per-lane narrow 32-bit writes have bank conflicts,
16 B writes at once do not; both spellings lay down exactly the same bytes, and the tensor map does not change.

## When needed (trigger)

- The staging face's write-side `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` is clearly above expectation, and the spelling is one row per lane, `st.shared.u32` (or narrower) each time;
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` / write instructions > 1 while you thought each narrow write was 1 wavefront.

## Minimal runnable example

Same spelling as the certification probe. `w[]` is this row's 16 already-packed 32-bit words (32 16-bit elements):

```cpp
// SWIZZLE_64B: the 16 B block number (address bits [4,6)) XOR address bits [7,9)
__device__ __forceinline__ uint32_t sw64(uint32_t off) { return off ^ (((off >> 7) & 3u) << 4); }

// face: shared address of this warp's staging face (__cvta_generic_to_shared), 1024 B aligned; lane = row, 64 B row pitch
// narrow writes (conflicting): 16 writes of 4 B per lane
#pragma unroll
for (uint32_t j = 0; j < 16; ++j)
  asm volatile("st.shared.u32 [%0], %1;" :: "r"(face + sw64(lane * 64 + j * 4)), "r"(w[j]) : "memory");

// 16 B writes (conflict-free): 4 writes of 16 B per lane, laying down exactly the same bytes as above
#pragma unroll
for (uint32_t c = 0; c < 4; ++c)
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(face + sw64(lane * 64 + c * 16)), "r"(w[4*c]), "r"(w[4*c+1]), "r"(w[4*c+2]), "r"(w[4*c+3]) : "memory");
```

Acceptance: write-side `bankconf_st` is 0; the face bytes are bitwise identical to before the change (or the downstream TMA store's result is unchanged). The probe's readings are under Sources.

## Gotchas

1. **SW64 permutes only 16 B blocks; it does not touch the words inside a block.** With lane = row and a 64 B row pitch, one row covers only 16 banks, and an odd and an even row together make 32 banks;
   the XOR mask comes from address bits [7,9), i.e. `(lane >> 1) & 3`, permuting only among 4 blocks. The bank number of the j-th warp-level `st.shared.u32`
   = `16·(lane & 1) + (j ⊕ 4·((lane >> 1) & 3))` has only 8 distinct values, so spread over 32 lanes it is a 4-way conflict.
   With 16 B written at once, each group of 8 lanes is one 128 B phase, and `(lane & 1, (lane >> 1) & 3)` covers the 8 16 B slots exactly.
2. **Generalization (derived from address algebra; the probe tested only SW64)**: the swizzle permutes at 16 B granularity, so with lane = row each lane must write a full 16 B each time for the conflicts to be 0.
3. **v4 writes need 16 B-aligned addresses**: `sw64` flips only bits [4,6), so 16 B alignment is preserved; the face base is 1024 B aligned (consistent with the phase in which TMA interprets the swizzle).
4. **Lower conflict readings do not mean a faster whole kernel**: do a same-job A/B on your own kernel before switching.

## Sources

- CUDA C++ Programming Guide, Shared Memory / bank conflicts (32 banks of 4 B each; 128-bit accesses are phased by 8 threads):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x
- PTX ISA: the vector form `.v4` of `st.shared` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Public infra API: `CU_TENSOR_MAP_SWIZZLE_64B` of the CUDA driver's `cuTensorMapEncodeTiled`.
- NCU metrics: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`,
  `smsp__inst_executed_op_shared_st.sum`.
- GPU-certification probe (`sw64 u32` / `sw64 v4`), verdict
  `PROBE PASS max_diff=0.0`: `E1-sw64-face-u32` / `E1-sw64-face-v4` (the 2048 words laid down by both spellings all equal the host SW64 image).
  NCU (4 warps each writing a 2048 B face, 64 repetitions): `st.shared.u32` 4096 write instructions, 16384 wavefronts (4 each), `bankconf_st` 12288;
  `st.shared.v4.b32` 1024 write instructions, 4096 wavefronts (4 each = the ideal for 16 B writes), `bankconf_st` 0. `%clock64`: 258.8 vs 75.5 cycles per repetition.
