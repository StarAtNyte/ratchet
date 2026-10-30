# tcgen05_umma_issue_cta_pair (assembling the whole UMMA main loop, single CTA and CTA pair: TMA → mbarrier ring → `tcgen05.mma` → commit → TMEM readback; the pair's `.cta_group::2` TMA hits the leader's barrier, commit multicasts)

> **Certification source: re-certified on a GPU probe in CUDA mode** (GEMM and pair probes, not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the error index there belongs to the DSL front end and is not carried over; the readings were re-taken on the CUDA spelling.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For descriptor bit fields see [`umma_raw_desc_issue`](umma_raw_desc_issue.md),
> for TMEM base-address publication see [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md),
> for the finishing side (conversion / staging face / TMA store) see [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md).

## Purpose

Blackwell UMMA has five coupled shape constraints; get any one wrong and the symptom shows up somewhere else (hang, half the result, all wrong):

| Step | Hardware fact | CUDA / PTX spelling |
|---|---|---|
| smem operand faces | UMMA reads the faces through smem descriptors; the swizzle is a descriptor field | faces landed by TMA `CU_TENSOR_MAP_SWIZZLE_128B`, face base 1024 B aligned; descriptor layout field = 2 |
| issue and completion | MMA is asynchronous; completion via `tcgen05.commit` arriving on an mbarrier | S mbarriers each for full / empty (count 1); producer `arrive.expect_tx` + TMA, the MMA warp waits on full and commits to empty |
| accumulator | lives in TMEM, addressed by 32-bit columns, alloc before use | `tcgen05.alloc` issued by one warp, the base address written into an smem word; readback with `tcgen05.ld.32x32b`, warp `w` can access only lanes `32*(w%4)..+31` |
| pair operands | a `cta_group::2` MMA is issued by the leader and reads **both CTAs' smem** (same offset); D lands in each CTA's own TMEM | each CTA loads its own 128 rows of A and N/2 rows of B; both CTAs' TMAs use the `.cta_group::2` form with the completion signal hitting **the leader's** full barrier (`mapa` to rank 0) |
| pair completion | `tcgen05.commit` by default arrives only on the **issuing CTA's** own barrier | `tcgen05.commit.cta_group::2…multicast::cluster [bar], 0b11`: the barrier at the same offset arrives in both CTAs |

## When needed (trigger)

- You are writing your first UMMA on sm100, or moving from a single CTA to `cta_group::2`. **First run this entry's example in your job** (the geometry is constant),
  get it bitwise exact, then swap in your own tiles and warp roles one item at a time.
- Pair shape: **the leader's accumulator full flips, while the same-named barrier on the follower never flips** (the follower's epilogue warp hangs, the job hits the timeout) —
  the commit is not multicast (gotcha 3; the probe's negative contract is exactly this phenomenon).
- Pair shape: **the full barrier never fills** — the follower's TMA did not hit the leader's barrier, or `expect_tx` counts too many (e.g. both CTAs arm) (gotcha 2).
  Conversely, `expect_tx` counting too few (the leader counts only its own half) **does not hang**: the phase completes early once the local half lands, and the MMA reads half a face, silently wrong
  (see [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md) gotcha 4; this probe has no negative control for this cell).
- The second launch in the same process hangs / reads leftovers from the last one: TMEM was not deallocated, the final cluster barrier is missing, or the first accumulate predicate was passed 1.

## Minimal runnable example

### Single CTA (192 threads: warp 0 TMA, warp 1 TMEM allocation + MMA, warps 2–5 readback)

For the main loop and descriptors see the example in [`umma_raw_desc_issue`](umma_raw_desc_issue.md) (the same probe); producer and readback:

```cpp
// producer (warp 0, one elected thread): needs no TMEM base address, does not wait for allocation
if (elect_one()) {
  for (int k = 0; k < KT; ++k) {
    const int s = k % S;
    if (k >= S) mbar_wait(empty0 + 8 * s, ((k / S) - 1) & 1);
    mbar_arrive_expect_tx(full0 + 8 * s, A_BYTES + B_BYTES);       // the bytes of this stage's two faces
    tma_load_2d(sA + s * A_BYTES, &tmapA, full0 + 8 * s, k * 64, m0);  // coords = {K element, row}
    tma_load_2d(sB + s * B_BYTES, &tmapB, full0 + 8 * s, k * 64, n0);
  }
}
// readback (warps 2..5; sub = warp % 4 decides the lane segment)
mbar_wait(accf, 0);
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
const int sub = warp % 4, row = 32 * sub + lane;
for (int c = 0; c < BN / 32; ++c) {
  uint32_t v[32];
  tm_ld32x32b_x32(tbase + ((uint32_t)(32 * sub) << 16) + c * 32, v);   // address: lane in the high 16 bits, column in the low 16 bits
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  // v[j] = fp32 bits of C[m0 + row][n0 + 32c + j]
}
```

`tma_load_2d` = `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [tmap, {c0, c1}], [bar];`,
with the tensor map passed into the kernel as `const __grid_constant__ CUtensorMap` and its address taken as the 64-bit operand.

### CTA pair (cluster (2,1,1), pair tile 256 × N, 128 rows per CTA)

```cpp
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(192, 1) pair_k(...) {
  const uint32_t rank = cluster_ctarank();                       // mov.u32 %0, %cluster_ctarank;
  const int m0 = pairid * 256 + rank * 128;                      // this CTA's 128 rows of A
  const int nb = n0 + rank * (N / 2);                            // this CTA's N/2 rows of B
  // ... thread 0 inits full/empty/accf (count 1) + fence.mbarrier_init.release.cluster
  cluster_arrive_relaxed(); cluster_wait();                      // rendezvous 1: init visible to the whole cluster
  if (warp == 1) { tm_alloc2(holder, 128); tm_relinquish2(); tc_fence_before(); }   // .cta_group::2, same-numbered warp in both CTAs
  cluster_arrive(); cluster_wait();                              // rendezvous 2 (release/acquire): publish the base-address word
  tc_fence_after();
  const uint32_t tbase = holder;

  if (warp == 0 && elect_one()) {                                // both CTAs run the producer
    for (int k = 0; k < KT; ++k) {
      const int s = k % S;
      if (k >= S) mbar_wait(empty0 + 8 * s, ((k / S) - 1) & 1);   // waits on its own CTA's empty
      const uint32_t fb = mapa_shared(full0 + 8 * s, 0);          // the leader's full[s] (cluster address)
      if (rank == 0) mbar_arrive_expect_tx(full0 + 8 * s, 2 * (A_BYTES + B_BYTES));   // both CTAs' bytes
      tma_load_2d_cg2(sA + s * A_BYTES, &tmapA, fb, k * 64, m0);  // .cta_group::2 form
      tma_load_2d_cg2(sB + s * B_BYTES, &tmapB, fb, k * 64, nb);
    }
  } else if (warp == 1 && rank == 0) {                           // only the leader waits on full, issues and commits
    const uint32_t IDESC = idesc_bf16_f32(256, N), HI = desc_hi(1024, 2);
    for (int k = 0; k < KT; ++k) {
      const int s = k % S;
      mbar_wait(full0 + 8 * s, (k / S) & 1);
      tc_fence_after();
      if (elect_one()) {
        const uint32_t a_s = desc_lo(sA + s * A_BYTES, 16), b_s = desc_lo(sB + s * B_BYTES, 16);
        for (int kk = 0; kk < 4; ++kk)                           // the leader's own smem offsets; the hardware reads the peer at the same offsets
          umma2_f16(tbase, (uint64_t(HI) << 32) | (a_s + 2 * kk), (uint64_t(HI) << 32) | (b_s + 2 * kk), IDESC, (k | kk) != 0);
        tc_commit2_mc(empty0 + 8 * s, 0x3);                       // empty[s] arrives in both CTAs
      }
      __syncwarp();
    }
    if (elect_one()) tc_commit2_mc(accf, 0x3);                    // accf arrives in both CTAs
    __syncwarp();
  } else if (warp >= 2) {                                        // each CTA reads 128 rows x N columns of its own TMEM
    mbar_wait(accf, 0); tc_fence_after();
    // ... same readback as the single CTA; result row = m0 + 32*(warp%4) + lane
    tc_fence_before();
  }
  __syncthreads();
  cluster_arrive(); cluster_wait();                              // finish: the peer may still be reading this CTA's smem / receiving commits
  if (warp == 1) { tc_fence_after(); tm_dealloc2(tbase, 128); }
}
```

PTX spellings (matching the probe):
`tcgen05.mma.cta_group::2.kind::f16 [d], adesc, bdesc, idesc, p;`,
`cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [tmap, {c0, c1}], [bar];`,
`tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [bar], mask16;`,
`tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [holder], ncols;`, `tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, ncols;`,
`mapa.shared::cluster.u32 r, addr, 0;`, `barrier.cluster.arrive{.relaxed}.aligned;` / `barrier.cluster.wait.aligned;`.

## Gotchas

1. **In a pair each CTA loads only half of A and of B**: A is split by M (rank r loads the r-th 128 rows), B by N (rank r loads the r-th N/2 rows);
   the MMA's idesc describes the whole pair's shape (M = 256, N = the whole N), the descriptors carry only the leader's own smem offsets, and both CTAs' faces must be at the same offset.
   The probe also computed a "B halves swapped" reference as a debugging fact line (see Sources).
2. **Only the leader's full barrier is used**: both CTAs' TMAs use the `.cta_group::2` form with the mbarrier operand given as **the leader's** cluster address
   (`mapa.shared::cluster` to rank 0; rank 0 `mapa`-ing itself also gets its own address); `arrive.expect_tx` is issued only by the leader, with the byte count = the sum of both CTAs'
   face bytes. The follower's complete_tx may arrive before the leader's expect_tx; that is allowed (within one phase the tx count goes negative, then positive).
3. **The commit must multicast**: `tcgen05.commit.cta_group::2…[bar]` without `multicast::cluster` arrives only on the issuing CTA's own barrier.
   The probe's negative contract: in the same kernel with only the multicast removed from the accumulator commit, the leader's half is exact while the follower's half stays NaN and its bounded wait gives up.
   The empty barrier likewise: each CTA's producer waits on **its own** CTA's empty[s], so the empty commit also multicasts `0b11`.
4. **Both CTAs run the producer; only the leader runs the MMA**: a leader guard on the TMA issue = full never fills; removing the leader guard from the MMA = both CTAs issue MMAs.
5. **TMEM allocation is paired**: `tcgen05.alloc.cta_group::2` is executed by the **same-numbered** warp in both CTAs, with the holder at the same smem offset;
   dealloc also uses `.cta_group::2`, after the final cluster barrier.
6. **Do not omit the final cluster barrier**: the leader's MMA reads the follower's smem and its commit writes the follower's barrier; if either CTA exits first,
   the peer's accesses land on a CTA that no longer exists.
7. **The readback lane segment is set by the warp number mod 4**: warp `w`'s `tcgen05.ld` can access only lanes `32*(w%4)..+31`, so the 4 readback warps must cover all 4 remainders
   (the probe uses warps 2..5).
8. **Wrap the MMA and commit in `elect.sync`**; without it there are 32 duplicate MMAs / commits.
9. **Differences between the single-CTA shape and the pair**: `cta_group::1` MMA / commit / alloc, TMA in the plain `shared::cluster` form hitting this CTA's barrier,
   `expect_tx` = this CTA's bytes, no `mapa`, no multicast, no final cluster barrier (dealloc after `__syncthreads()` is enough).

## Sources

- PTX ISA: `tcgen05.mma` (`.cta_group::2`), `tcgen05.commit` (`.multicast::cluster`), `tcgen05.alloc` / `dealloc` /
  `relinquish_alloc_permit`, `tcgen05.ld`, `cp.async.bulk.tensor` (`.cta_group::2`), `mbarrier`, `mapa`, `barrier.cluster` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUDA driver `cuTensorMapEncodeTiled`; CUDA C++ `__cluster_dims__`; CUTLASS `cute::SM100_TMA_2SM_LOAD_2D`,
  `cutlass::arch::umma_arrive_multicast_2x1SM`, `cute::TMEM::Allocator2Sm` (PTX spelling consistent with this entry).
- GPU-certification probes (GEMM and pair), verdict
  `PROBE PASS max_diff=0.0`: single CTA `A1-exact-v64` / `A1-exact-v256`; pair `B1-pair-exact` (cluster (2,1,1), M = 256, N = 128,
  K = 512, bitwise exact over three launches); negative contract `B3-plain-commit-starves-peer` (accumulator commit without multicast: leader half 65536/65536 exact, follower half 65536/65536 stays NaN,
  its bounded wait gives up). Fact line: a reference computed with "B halves swapped" differs from the output by 89 — confirming that rank r loads the r-th N/2 rows.
