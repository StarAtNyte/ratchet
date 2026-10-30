# umma_raw_desc_issue (issuing `tcgen05.mma.kind::f16` from inline PTX: hand-assembled 64-bit smem descriptors + 32-bit instruction descriptor; K step = descriptor low word +2; the accumulate bit is a predicate operand; one `elect.sync` per beat wraps all of the beat's MMAs and the commit)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; all readings were re-taken on the CUDA spelling, and the numbers there are not carried over.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Neighboring entries: [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) (whole-machine assembly, CTA pair),
> [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md) (unrolling by the stage count),
> [`umma_mainloop_wrap_segment_issue_floor`](umma_mainloop_wrap_segment_issue_floor.md) (per-k-block resident readings: three tiers, rolled / unrolled / single thread).

## Purpose

In CUDA mode there is no "MMA atom" computing descriptors for you: the two 64-bit **smem descriptors** for A / B and the one 32-bit
**instruction descriptor (idesc)** that a `tcgen05.mma` needs must be assembled by hand from the PTX ISA bit fields. This entry gives every field for a single CTA (`cta_group::1`), A / B both K-major bf16,
faces landed by TMA with `CU_TENSOR_MAP_SWIZZLE_128B`, plus the shape of the issue loop:

- **compute each operand face's base descriptor once outside the loop**; inside the loop only integer additions of "stage offset + constant" remain;
- **one step along K (K16 = 32 B) within the same 128 B swizzle atom row (64 bf16) = descriptor low word +2** (the start-address field is in units of 16 B);
  the hardware applies the swizzle from the address bits, and no other field changes;
- **the accumulate bit is the instruction's predicate operand** (`enable-input-d`): pass 0 for the first instruction of the whole accumulation, 1 for the rest;
- per beat: **the whole warp waits on the full mbarrier**, `tcgen05.fence::after_thread_sync`, then **one** thread selected by `elect.sync` issues all of the beat's MMAs
  and the `tcgen05.commit` (commit tracks only the tcgen05 async operations previously issued by the thread that issues it, so the MMA and the commit must be the same thread).

## When needed (trigger)

- You are writing the first UMMA in `kernel.cu`, or already have the TMA feed and TMEM allocation and need to hook the MMA chain on;
- results wrong in whole blocks / all NaN / only the first 16 columns right: first check the bit fields below (SBO, version bit, swizzle mode, N>>3 / M>>4, K step +2);
- results doubled after the first k-block, or carrying leftovers from the last launch: the accumulate predicate is wrong (the first one must pass 0).

**Not a speed-up lever.** How the descriptors are passed (two 32-bit registers assembled with `mov.b64` / one 64-bit `"l"` operand) gives numerics of 0.0 for both in the probe,
with per-beat cycles in the fact lines under Sources; to shorten the per-beat issue chain, look at [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md) first.

## Minimal runnable example

Conditions: single CTA, `BM = 128`, K tile = 64 (one 128 B atom row), A face `128 × 64` bf16, B face `BN × 64` bf16, each stage's face base
1024 B aligned (TMA box = `{64, rows}`, `CU_TENSOR_MAP_SWIZZLE_128B`); fp32 accumulator in TMEM. The spelling matches the certification probe line by line.

```cpp
// fk-cuda-cflags: -lineinfo
// smem descriptor (PTX ISA "Shared memory descriptor"), K-major SWIZZLE_128B:
//   lo: [0,14) start address>>4   [16,30) LBO>>4 (unused for swizzled K-major; fill with 16 B)
//   hi: [32,46) SBO>>4 = 1024 B (8 rows x 128 B)  [46,48) version = 1 (required on sm100)
//       [49,52) base offset (0 when the face is 1024 B aligned)  [61,64) layout: 2 = SWIZZLE_128B (4 = 64B, 6 = 32B, 0 = none)
__host__ __device__ inline uint32_t desc_lo(uint32_t saddr, uint32_t lbo) {
  return ((saddr >> 4) & 0x3FFFu) | (((lbo >> 4) & 0x3FFFu) << 16);
}
__host__ __device__ inline uint32_t desc_hi(uint32_t sbo, uint32_t layout) {
  return ((sbo >> 4) & 0x3FFFu) | (1u << 14) | (layout << 29);
}
// instruction descriptor kind::f16: [4,6) D format 1 = f32; [7,10) A format, [10,13) B format: 0 = f16, 1 = bf16;
//   bit 15 / 16 = 0: A / B K-major; [17,23) N>>3; [24,29) M>>4
__host__ __device__ constexpr uint32_t idesc_bf16_f32(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | (uint32_t(N >> 3) << 17) | (uint32_t(M >> 4) << 24);
}
__device__ inline void umma_f16(uint32_t d_tmem, uint32_t alo, uint32_t ahi, uint32_t blo, uint32_t bhi,
                                uint32_t idesc, uint32_t acc) {
  asm volatile("{\n\t.reg .pred p;\n\t.reg .b64 da, db;\n\tsetp.ne.b32 p, %6, 0;\n\t"
               "mov.b64 da, {%1, %2};\n\tmov.b64 db, {%3, %4};\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], da, db, %5, p;\n\t}"
               :: "r"(d_tmem), "r"(alo), "r"(ahi), "r"(blo), "r"(bhi), "r"(idesc), "r"(acc) : "memory");
}
__device__ inline uint32_t elect_one() {
  uint32_t pred;
  asm volatile("{\n\t.reg .b32 rx;\n\t.reg .pred px;\n\telect.sync rx|px, %1;\n\tselp.b32 %0, 1, 0, px;\n\t}"
               : "=r"(pred) : "r"(0xffffffffu));
  return pred;
}

// ---- MMA warp (TMEM base tbase already published as in tmem_alloc_ptr_handoff_no_named_barrier)
const uint32_t a_lo0 = desc_lo(sA, 16), b_lo0 = desc_lo(sB, 16);   // outside the loop: stage 0's low words
const uint32_t HI = desc_hi(1024, 2);                              // the high word shared by all stages and all K steps
constexpr uint32_t IDESC = idesc_bf16_f32(128, BN);
for (int k = 0; k < KT; ++k) {
  const int s = k % S;
  mbar_wait(full0 + 8 * s, (k / S) & 1);                          // the whole warp waits
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (elect_one()) {                                              // one election region per beat: MMA and commit on the same thread
    const uint32_t a_s = a_lo0 + s * (A_BYTES >> 4), b_s = b_lo0 + s * (B_BYTES >> 4);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)                                // 64 / 16 = 4 instructions; K step within an atom row = low word +2
      umma_f16(tbase, a_s + 2 * kk, HI, b_s + 2 * kk, HI, IDESC, (k | kk) != 0);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(empty0 + 8 * s) : "memory");             // release stage s to the producer
  }
  __syncwarp();
}
if (elect_one())
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(accf) : "memory");
```

## Gotchas

- **The start-address field is only 14 bits, in units of 16 B**: `(addr >> 4) & 0x3FFF` covers the 256 KB shared window; take the address from
  `__cvta_generic_to_shared(p)` (a 32-bit address inside the shared window), not a generic pointer. With only additions in the loop, staying inside the window never carries into the LBO field.
- **The version bit (bit 46) must be 1 on sm100.** CUTLASS's `cute::UMMA::SmemDescriptor` calls it `version_`; a descriptor missing it is not in the sm100 format.
- **The low word +2 K step holds only within one 128 B atom row.** When the K tile spans several atoms (e.g. 128 bf16 = two boxes), stepping across atoms uses the byte distance between the two boxes
  >> 4, not another +2; box bases must be 1024 B aligned, otherwise the base offset field is no longer 0.
- **SBO = 1024 B is the swizzle period of 8 rows × 128 B**; LBO is not used for 128 B-swizzled K-major. Switching to 64B / 32B swizzle requires changing SBO, the layout
  field and the atom row width as well; this entry did not certify those modes.
- **The idesc M field is M>>4 and the N field is N>>3**; for `kind::f16`, A / B format 0 = f16, 1 = bf16, and bits 4–5 are the D format (1 = f32).
  The probe certified N = 64 and N = 256 at M = 128.
- **The accumulate bit is a per-instruction predicate**: if the first MMA passes 1, leftovers in TMEM are added into the result. The probe catches this class of error by prefilling the output with NaN and launching three times in the same process.
- **`asm volatile` + `"memory"`**: an MMA / commit with no outputs and no `volatile` gets deleted as dead code; without `"memory"` it gets moved across the mbarrier wait.
- **The whole warp waits, one thread issues**: the `mbarrier.try_wait` loop is executed by the whole warp, and `elect.sync` requires all threads of the warp to arrive (after the wait,
  `__syncwarp()` before the next beat). The commit is written inside the election region; **do not** wrap it in another `elect.sync`.
- **Two ways of passing descriptors**: two 32-bit registers `mov.b64 da, {lo, hi}` and one 64-bit `"l"` operand (`(uint64_t(hi) << 32) | lo`)
  give numerics of 0.0 for both in the probe; the per-beat cycle difference is in the fact lines under Sources; choose by your own SASS / timing.

## Sources

- PTX ISA: `tcgen05.mma` (the `enable-input-d` predicate operand), Shared memory descriptor, Instruction descriptor (`kind::f16`),
  `tcgen05.commit`, `tcgen05.fence`, `elect.sync` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
- Public infra API: CUTLASS `cute::UMMA::SmemDescriptor` / `cute::UMMA::InstrDescriptor` (`include/cute/arch/mma_sm100_desc.hpp`,
  bit fields consistent with this entry); CUDA driver API `cuTensorMapEncodeTiled` (`CU_TENSOR_MAP_SWIZZLE_128B`).
- GPU-certification probe: verdict
  `PROBE PASS max_diff=0.0` (contracts of the same run: `A1-exact-v64` (N = 64), `A1-exact-v256` (N = 256), `A1-exact-v64d64` (64-bit `"l"` descriptor),
  three launches each, NaN prefill, bitwise equal to the host fp32 reference). Fact lines: the two ways of passing descriptors have the same cycles per k-block (resident 295.3 / 295.3, streaming 324.0 / 324.0);
  in SASS the descriptor high word is `UMOV 0x40004040`, the K step is `UIADD3 …, 0x2 / 0x4 / 0x6`, and idesc (N = 64) is `UMOV 0x8100490`.
