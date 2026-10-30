# umma_smem_operand_sts128_publish (a bf16 UMMA operand computed in registers: hand-written `st.shared.v4` into a K-major SW128 face → `fence.proxy.async.shared::cta` → handed to the MMA-issuing thread → `tcgen05.mma` reads it through a descriptor)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cell `sts`; the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the DSL spelling that derives addresses from layout objects is not carried over; the address formula is given directly here.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Same family: for writing the B face with `stmatrix` see site B of
> [`smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`](smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128.md); for descriptors see
> [`umma_raw_desc_issue`](umma_raw_desc_issue.md).

## Purpose

The UMMA's A (or B) operand is not moved in by TMA but **computed by threads in registers** (f32 → bf16x2 pairwise conversion). For `tcgen05.mma` to read it through the same K-major
SWIZZLE_128B descriptor, the threads must write a face with **exactly the same byte image that TMA `CU_TENSOR_MAP_SWIZZLE_128B` would land**:

```
face 1024 B aligned; the c-th 16 B block of row m (elements k = 8c .. 8c+7) is at
    off(m, c) = m * 128 + ((c ^ (m % 8)) * 16)          // SW128: 16 B block number XOR address bits [7,10)
```

One row of 64 bf16 = 8 `st.shared.v4.b32`. Publishing after the writes has only one hard constraint: **every writing thread does `fence.proxy.async.shared::cta` after its own last write**
(UMMA goes through the async proxy, `st.shared` through the generic proxy), and only then synchronizes with the MMA-issuing thread.

## When needed (trigger)

- The operand is produced in registers by a warpgroup (element-wise transforms, online quantization, the result of an upstream MMA) and must feed the next UMMA;
- the MMA reads stale data / partly stale data: missing `fence.proxy.async`; results misplaced in groups of 8 rows: the swizzle image is wrong (the block number not XORed with row % 8).

## Minimal runnable example

Single CTA, the A face (128 × 64 bf16) written by threads, the B face landed by TMA; same spelling as the certification probe:

```cpp
__device__ inline uint32_t sw128(uint32_t off) { return off ^ (((off >> 7) & 7u) << 4); }

const int m = threadIdx.x;                                     // thread = one row of the face (128 threads)
#pragma unroll
for (int c = 0; c < 8; ++c) {
  uint32_t w[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) w[q] = pack_bf16x2(a(m, 8 * c + 2 * q), a(m, 8 * c + 2 * q + 1));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
               :: "r"(sA + sw128(m * 128 + c * 16)), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
}
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // every writing thread, after its last write
asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
__syncthreads();                                               // or: __syncwarp per warp + one lane arriving on a full mbarrier
if (warp == 0) {
  mbar_wait(tma_bar, 0);                                       // the B face's TMA
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (elect_one()) {
    for (int kk = 0; kk < 4; ++kk)                             // A / B use the same kind of SW128 descriptor, K step low word +2
      umma_f16_64(tbase, (uint64_t(HI) << 32) | (desc_lo(sA, 16) + 2 * kk),
                         (uint64_t(HI) << 32) | (desc_lo(sB, 16) + 2 * kk), IDESC, kk != 0);
    tc_commit(accb);
  }
  __syncwarp();
}
```

## Gotchas

- **The address depends only on the row**: the XOR source bits [7,10) come from `m*128`, and the in-block offset `c*16 < 128` does not touch them; so when a thread stays on one row, `sA + m*128` and the XOR amount
  `(m%8)<<4` are thread constants, and only "XOR the block number" remains per block in the loop. Whether the address is recomputed in the loop body: look at the `LOP3` / `SHF` before each `STS.128` in SASS.
- **The face must be 1024 B aligned** (the swizzle phase is counted from a 1024 B boundary) for the descriptor's base offset field to be 0.
- **Position of the fence**: after this thread's **last** `st.shared` and before the sync, once per thread per round; inserted between writes it covers only the writes before it.
- **When a warp arrives only once** (mbarrier publication): `__syncwarp()` first, then pick one lane to arrive; the barrier's count = the number of writing warps;
  every writing thread must fence itself; it is not enough for only the arriving lane to fence.
- **The face-writing role must wait for the MMA to finish reading before reusing the face**: the MMA-issuing thread does `tcgen05.commit` to a "face empty" mbarrier, and the face-writing role waits on it first in the next round.

## Sources

- PTX ISA: `st.shared`, `fence.proxy.async`, `tcgen05.mma` (smem descriptor, the SWIZZLE_128B address transform), `tcgen05.fence`,
  `cvt.rn.bf16x2.f32` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUDA driver `CU_TENSOR_MAP_SWIZZLE_128B` (the same image the threads write); CUTLASS `cute::UMMA::SmemDescriptor`.
- GPU-certification probe: cell `sts`, verdict
  `PROBE PASS max_diff=0.0`: `D1-operand-sts128-sw128` (the A face written by 128 threads with 8 `STS.128` each, the B face by TMA, bitwise exact
  over two launches). SASS: 8 `STS.128`, the publication segment `MEMBAR` + `FENCE.VIEW.ASYNC.S` + `BAR.SYNC`.
