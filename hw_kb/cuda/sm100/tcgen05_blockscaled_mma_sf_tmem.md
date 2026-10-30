# tcgen05_blockscaled_mma_sf_tmem (block-scaled UMMA `kind::mxf8f6f4.block_scale`: scale factors go from a 512 B atom in smem → `tcgen05.cp.32x128b.warpx4` → TMEM; each MMA picks the byte for its own K block via the TMEM address + the SF id in idesc)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the helper / layout objects there, and the sections on fp4 container faces and pair upload, were not run on this repo's CUDA probe and are not carried over (see gotcha 7).
>
> One file per hardware interface. Operator-agnostic: mechanism only. The main-loop skeleton, descriptors, TMEM allocation and readback are as in [`umma_raw_desc_issue`](umma_raw_desc_issue.md) /
> [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md); this entry covers only the extra pieces.

## Purpose

`tcgen05.mma.kind::mxf8f6f4.block_scale` scales one row of A and one column of B for each block of **K = 32** with one UE8M0 byte (a power of 2):
`D[m][n] += Σ_kb 2^(sfa[m][kb]−127) · 2^(sfb[n][kb]−127) · Σ_{k∈kb} A[m][k]·B[n][k]`. The extra hardware facts:

| Step | Fact | CUDA / PTX spelling |
|---|---|---|
| where SF lives | the MMA reads SF from **TMEM**; the instruction carries the TMEM addresses of SFA / SFB | `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [d], adesc, bdesc, idesc, [tsfa], [tsfb], p;` |
| SF layout in smem | one **512 B atom** per 128 rows × 4 K blocks: `byte(m, kb) = 16*(m%32) + 4*(m/32) + kb` | the host rearranges the naive `(rows, K/32)` SF into this atom once; in the kernel, TMA or a plain copy into smem |
| how SF gets into TMEM | `tcgen05.cp.cta_group::1.32x128b.warpx4`: 32 rows × 16 B replicated to 4 lane segments | the source is an smem descriptor (SWIZZLE_NONE, SBO = 128 B); issued by the MMA-issuing thread before the first MMA |
| which byte each MMA picks | one TMEM column holds the bytes of 4 K blocks; the SF id field of idesc picks the byte | idesc: `b_sf_id` [4,6), `a_sf_id` [29,31), scale format bit 23 (1 = UE8M0); the top 2 bits of the TMEM address carry the same value (see gotcha 3) |
| instruction shape | K = 32 for `mxf8f6f4`; e4m3 operands are 1 byte | one 128 B atom row = 128 e4m3 = 4 MMAs; the K step is still descriptor low word +2 (32 B) |

## When needed (trigger)

- You need to issue an FP8 (e4m3 / e5m2) × FP8 UMMA with UE8M0 block scaling and do not know how to lay out SF, how to get it into TMEM, or how each instruction points at its own K block;
- numerics **finite but wrong in blocks** (the error period aligned with 32 / 128 rows or 4 K blocks): check the byte order of the SF atom, the `tcgen05.cp` descriptor, and whether the SF id follows the K block.

## Minimal runnable example

Single CTA, M = N = 128, K = 128 (one k-tile, 4 MMAs); A / B are e4m3, K-major, TMA `SWIZZLE_128B` (box `{128 B, 128 rows}`,
`CU_TENSOR_MAP_DATA_TYPE_UINT8`); SFA / SFB one 512 B atom each in smem. The spelling matches the certification probe:

```cpp
// host: naive sf[m][kb] -> atom
for (int m = 0; m < 128; ++m) for (int kb = 0; kb < 4; ++kb) atomA[16 * (m % 32) + 4 * (m / 32) + kb] = sfa[m * 4 + kb];

// idesc (the block-scale variant of PTX ISA "Instruction descriptor"): A / B format E4M3 = 0; N>>3 @17; UE8M0 @23; M>>4 @24
__host__ __device__ constexpr uint32_t idesc_mxf8(int m, int n, uint32_t asf, uint32_t bsf) {
  return (bsf << 4) | (uint32_t(n >> 3) << 17) | (1u << 23) | (uint32_t(m >> 4) << 24) | (asf << 29);
}

// kernel: sf[0] / sf[1] are the two 512 B atoms (smem, 128 B aligned), already fence.proxy.async'd + synced;
// the A / B faces are landed by TMA (alo / blo are their descriptor low words); the MMA-issuing warp first waits on their full mbarrier
mbar_wait(full_ab, 0);
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
if (elect_one()) {                                          // the thread that issues the MMAs
  const uint32_t sfa_col = tbase + 256, sfb_col = tbase + 260;           // the accumulator occupies columns 0..127
  const uint32_t sdhi = desc_hi(128, 0);                                  // SWIZZLE_NONE, SBO = 128 B (8 rows x 16 B)
  const uint64_t da_sf = (uint64_t(sdhi) << 32) | desc_lo(smem_u32(sf[0]), 128);
  const uint64_t db_sf = (uint64_t(sdhi) << 32) | desc_lo(smem_u32(sf[1]), 128);
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfa_col), "l"(da_sf) : "memory");
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(sfb_col), "l"(db_sf) : "memory");
  for (int kb = 0; kb < 4; ++kb) {
    const uint32_t ta = sfa_col | (uint32_t(kb) << 30), tb = sfb_col | (uint32_t(kb) << 30);   // top 2 address bits = byte number
    const uint64_t da = (uint64_t(HI) << 32) | (alo + 2 * kb), db = (uint64_t(HI) << 32) | (blo + 2 * kb);
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;\n\t}"
                 :: "r"(tbase), "l"(da), "l"(db), "r"(idesc_mxf8(128, 128, kb, kb)), "r"((uint32_t)(kb != 0)), "r"(ta), "r"(tb)
                 : "memory");
  }
  tc_commit(accb);
}
```

## Gotchas

1. **Byte order of the SF atom**: row m splits into `m%32` (16 B row index) and `m/32` (which 4-byte group within the row), and the K block is the byte index within the group. After the `32x128b.warpx4`
   replication, byte kb of column `m/32` of TMEM lane `32q + m%32` is `sf[m][kb]` (one copy in each of the 4 lane segments).
2. **The source of `tcgen05.cp` is a descriptor, not an address**: the SWIZZLE_NONE K-major canonical form is "core matrices of 8 rows × 16 B"; 32 rows = 4 core matrices,
   SBO = 128 B; there is only one 16 B column, so LBO has no effect. The atom's smem address is 16 B aligned (the probe uses 128 B). When smem was written by the generic proxy,
   `fence.proxy.async.shared::cta` + a sync are needed before `tcgen05.cp`.
3. **The SF id picks the byte**: each MMA's `a_sf_id` / `b_sf_id` (idesc) = its K block number `kb % 4`; the TMEM column address of SF is the same for all 4 K blocks.
   The spelling certified by the probe matches CUTLASS: the top 2 address bits are also written with the same byte number (`cute::UMMA::make_runtime_instr_desc_block_scaled` takes the SF id from the top 2 address bits).
   The other two addressing schemes (top 2 bits cleared with idesc still set, and moving the column per K block with SF id always 0) only produce fact lines, see Sources.
4. **`tcgen05.cp` and the MMA are issued in order by the same thread**: it goes through the tcgen05 async chain, cp first and MMA after, with no extra wait needed.
5. **K step**: e4m3's K = 32 = 32 B, the same as bf16's K16 — descriptor low word +2; one 128 B atom row = 4 MMAs = 4 SF bytes = one TMEM column.
6. **D is fp32**; the block-scale idesc has no D format field (bits 4–5 are `b_sf_id`), so do not reuse the idesc function for `kind::f16`.
7. **Shapes this entry did not certify**: smem container faces for fp4 (e2m1) operands, `kind::mxf4` / `mxf4nvf4`, scales in E4M3 format, SF upload for a CTA pair. Write your own probe before using them.

## Sources

- PTX ISA: `tcgen05.mma` (`.kind::mxf8f6f4`, `.block_scale`, TMEM layout of scale factors), `tcgen05.cp` (`.32x128b.warpx4`),
  Instruction descriptor (block-scale variant), Shared memory descriptor — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
- Public infra API: CUTLASS `cute::UMMA::InstrDescriptorBlockScaled`, `cute::UMMA::make_runtime_instr_desc_block_scaled`,
  `cute::SM100_MMA_MXF8F6F4_SS`, `cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta`, `cutlass::detail::Sm1xxBlockScaledBasicChunk` (SF atom
  `((32,4),(32,4)):((16,4),(0,1))`) — all in the public CUTLASS headers; their PTX spelling and bit fields agree with this entry.
- GPU-certification probe (`c0` contract, `c1` / `c2` facts), verdict
  `PROBE PASS max_diff=0.0`: `F1-blockscaled-mxf8f6f4-ue8m0` (e4m3 × e4m3, one UE8M0 per (row, 32 K), SF taking 2^{-1,0,1},
  bitwise equal to the host reference over two launches; the same output differs from an "unscaled" reference by 94.5, showing the scale really takes effect). Fact lines: top 2 address bits cleared with the SF id written only in idesc (`c1`) is likewise bitwise exact;
  moving the column per K block with SF id always 0 (`c2`) reports `misaligned address` at runtime.
