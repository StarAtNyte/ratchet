# tmem_r2t_recast_fragment (writing a bf16 operand from registers back into TMEM as the next UMMA's A: packed words → `tcgen05.st.32x32b` → `tcgen05.mma … [a_tmem] …`; the lane / column / half-word layout of A in TMEM)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cells `ts` / `tsswap`; the probe is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree. The DSL entry's constraints ①② (the copy atom's element type, the Repetition depth) are shape checks of the DSL front end
> that do not exist in the CUDA spelling and are not carried over; constraint ③ (the register peak is decided by the source's last consumer) belongs to register allocation, see
> [`lowtier_role_tmem_readout_spill_ladder`](lowtier_role_tmem_readout_spill_ladder.md).
>
> One file per hardware interface. Operator-agnostic: mechanism only. Architecture scope: sm100.

## Purpose

`tcgen05.mma` allows the A operand to reside in TMEM (the `[a_tmem]` form of `kind::f16`), with B still through an smem descriptor. The typical shape is "read an f32 tile from TMEM →
element-wise work in registers → convert to bf16 and write back to TMEM → the next MMA takes it straight from TMEM". TMEM is addressed by 32-bit columns, so the unit written back is a **packed word**:

- row m of A lives in TMEM **lane m** (with M = 128, each of 4 warps writes its own 32 lanes);
- the c-th 32-bit column holds `A[m][2c]` (**low half**) and `A[m][2c+1]` (high half) — consistent with "even elements go into the low half" of
  [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md);
- one K16 MMA consumes 8 columns; the A address of the kk-th K16 step = A base column + `8*kk`.

The probe also ran the counter-example "even elements go into the high half" as a fact line (see Sources); a reversed layout is the kind that gives entirely wrong numbers without any error.

## When needed (trigger)

- The code has both tcgen05 and a bf16 register fragment that must land in TMEM, with the next MMA reading it from TMEM;
- a TS-form MMA result has whole rows wrong / adjacent columns wrong in pairs: check the half-word order and the column step of the K step first.

## Minimal runnable example

Single CTA, M = 128, N = 64, K = 64 (4 MMAs); A written to TMEM columns 128..159, B landed by TMA as an SW128 face. Same spelling as the certification probe:

```cpp
// each thread = one row of A, m = 32*warp + lane
uint32_t pk[32];
#pragma unroll
for (int c = 0; c < 32; ++c) pk[c] = pack_bf16x2(a_row[2 * c], a_row[2 * c + 1]);   // even k goes into the low half
tm_st32x32b_x32(tbase + ((uint32_t)(32 * warp) << 16) + 128, pk);                  // lane m, columns 128..159
asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
__syncthreads();                                                                     // all 4 warps have written
if (warp == 0) {
  mbar_wait(full_b, 0);                                                              // MMA may be issued only after the B face's TMA has landed
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (elect_one()) {
    for (int kk = 0; kk < 4; ++kk) {
      const uint64_t db = (uint64_t(desc_hi(1024, 2)) << 32) | (desc_lo(sB, 16) + 2 * kk);
      asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                   "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}"
                   :: "r"(tbase), "r"(tbase + 128 + 8 * kk), "l"(db), "r"(idesc_bf16_f32(128, 64)), "r"((uint32_t)(kk != 0)) : "memory");
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(accb) : "memory");
  }
  __syncwarp();
}
```

## Gotchas

- **After writing, `tcgen05.wait::st`, then `before_thread_sync` → sync → `after_thread_sync`, before the MMA may read it**: 4 warps write TMEM and one thread issues the MMA;
  this is a cross-thread handoff, with the same fence rules as [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md).
- **Half-word order**: even k in the low half. Reversing it (the `tsswap` counter-example) raises no error; the results are finite but wrong.
- **The K step is 8 columns** (16 bf16 = 8 32-bit columns), not 16; when A and D are in the same TMEM allocation, do not let their column ranges overlap.
- **A warp can only write its own lane segment**: warp `w`'s `tcgen05.st` writes lanes `32*(w%4)..+31`, so an M = 128 A needs 4 warps with different remainders writing together.
- **The TMEM A face of a CTA pair (`cta_group::2`) is not certified by this entry**: in a pair the lane layout of A differs from the single-CTA one; write your own probe before using it.

## Sources

- PTX ISA: `tcgen05.mma` (the `[a-tmem]` form with A in TMEM), `tcgen05.st` / `tcgen05.wait::st`, Matrix fragments for `tcgen05.mma`
  (the layout of A in Tensor Memory) — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
- Public infra API: CUTLASS `cute::SM100_MMA_F16BF16_TS` (`include/cute/arch/mma_sm100_umma.hpp`, same PTX spelling).
- GPU-certification probe: cells `ts` / `tsswap`, verdict
  `PROBE PASS max_diff=0.0`: `D3-operand-tmem-a` (M 128 × N 64 × K 64, A into TMEM via `tcgen05.st`, 4 TS-form MMAs, bitwise exact
  over two launches). Fact line: the same kernel with the half-word order reversed (even k in the high half) gives `max_diff = 30` with no error. SASS: writing A is one `STTM` and 32 `F2FP.BF16.F32.PACK_AB`.
