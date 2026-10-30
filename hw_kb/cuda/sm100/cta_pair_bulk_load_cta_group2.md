# cta_pair_bulk_load_cta_group2 (`.cta_group::2` tensor-map TMA load: each of the two CTAs issues one, both book completion on the leader's single mbarrier)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-pair2d`, `tma-pair_remote`, `tma-sass-cta-group-suffix` etc. all passed in the same round).** The example is inline PTX + `cuTensorMapEncodeTiled`; the readings were taken on the CUDA spelling.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the downstream "who may wait / should the last round still peek" see
> [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md); for whether each operand should be multicast or non-multicast see
> [`tma_2cta_nonmulticast_operand_load`](tma_2cta_nonmulticast_operand_load.md).

## Purpose

A `tcgen05` `cta_group::2` MMA is carried out jointly by a CTA pair (two CTAs whose `%cluster_ctarank` differs only in the lowest bit; the one with low bit 0 is the leader)
and issued only by the leader. The feed side therefore has to answer: **who books the bytes of the two halves of the pair.** The `.cta_group`
modifier on `cp.async.bulk.tensor` decides whether a load is **allowed** to deliver its completion signal (complete_tx) to the peer CTA's mbarrier:

| Modifier | Where mbar may be | Consequence of redirecting to the peer |
|---|---|---|
| omitted (= `.cta_group::1`) | must be in the same CTA as the destination smem | undefined behavior per the PTX ISA; what this probe measured is in gotcha 1 |
| `.cta_group::2` | the same CTA **or** its peer | this is what it is for |

What `.cta_group::2` opens up is **where the accounting goes**, not how much is moved: one instruction still moves only one box into **the issuer's own** smem. So the pair's
two halves of bytes are made up by the two CTAs **issuing one each**, both delivering completion to the leader's same barrier; the leader arms the sum of the two,
and only the leader waits.

## When needed (trigger)

- Writing a `cta_group::2` UMMA pipeline and wanting one full barrier to book both halves of the pair (without an extra layer of cross-CTA rendezvous).
- Symptom: the pair pipeline hangs at the **first** consume point (job killed on timeout, not an illegal instruction, not wrong numbers), while the same code runs under
  `cluster = 1` — first check whether the load's PTX string has `.cta_group::2`, then check the barrier pointer.
- There is already a layer of "count = 2 per slot" cross-CTA rendezvous barriers (each CTA arms its own half, flips locally, then rendezvous once more):
  that is the compensating shape for not using `.cta_group::2`. Mechanical predicate: in the same kernel's SASS the tcgen05 family carries `.2CTA` while
  `UTMALDG` does not.
- **Anti-trigger**: a `cta_group::1` pipeline where the two CTAs each hold independent operands and each issue their own MMA — each arms and waits on its own; do not add `.cta_group::2`.

## Minimal runnable example

```cpp
// fk-cuda-cflags: (none; sm_100a is supplied by the operator tree's builder)
#include <cuda.h>
// host: rank-2 map, dims[0] is the contiguous dimension; box 64 columns x 64 rows, SW128 (box inner dimension 64 x 2 B = 128 B)
//   cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims{K, R}, strides{K*2},
//                          box{64, 64}, elemStrides{1, 1}, INTERLEAVE_NONE, SWIZZLE_128B, L2_128B, OOB_FILL_NONE);
// kernel parameter: const __grid_constant__ CUtensorMap tm (address rules in tma_param_desc_inline_ptx)

__device__ __forceinline__ uint32_t mapa_u32(uint32_t addr, uint32_t rank) {   // local offset -> the same offset in CTA rank
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank)); return r;
}
__device__ __forceinline__ void tma2d_cg2(uint32_t dst, uint64_t tmap, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
               " [%0], [%1, {%2, %3}], [%4];"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}

// per CTA: tile 16 KB = two SW128 atoms (K direction 0..63 / 64..127), base 1024 B aligned
const uint32_t rank = cluster_ctarank();            // mov.u32 %0, %cluster_ctarank;
if (threadIdx.x == 0) {
  mbar_init(bar, 1);                                // mbarrier.init.shared::cta.b64
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  if (rank == 0) mbar_arrive_expect_tx(bar, 2 * TILE_B);   // only the leader arms, bytes = the whole pair
}
__syncthreads();
cluster_sync();   // barrier.cluster.arrive.release + wait.acquire: the leader's init + arm visible before any delivery
if (threadIdx.x == 0) {                             // both CTAs issue (no leader guard on the producer side)
  const uint32_t mb = mapa_u32(bar, 0);             // completion delivered to the leader's barrier
  tma2d_cg2(tile,        (uint64_t)&tm, 0,  row0, mb);
  tma2d_cg2(tile + 8192, (uint64_t)&tm, 64, row0, mb);
}
if (rank == 0) { while (!mbar_try_wait(bar, 0)) {} }   // only the leader waits; its flip => both halves have landed
// the follower's own full barrier is never armed and must not be waited on (see cta_pair_consumer_peek)
```

The probe's acceptance: smem prefilled with sentinels, and both CTAs' 16 KB tiles equal byte for byte the expectation under the SW128 address function (`max_diff = 0`).

## Gotchas

1. **Omitting `.cta_group::2` and only redirecting the barrier pointer to the leader.** It compiles and reports no error. The PTX ISA makes it undefined behavior;
   this probe's `pair1_remote` cell (follower without the modifier, delivering to the leader's barrier, the leader arming only the follower's bytes)
   measured: the leader's barrier **never flipped** within the bounded wait (flip=0), while the bytes in the follower's own smem were right — the bytes landed, the completion signal was not booked at the leader. **Check the modifier in the PTX string first, then the pointer.**
2. **"The barrier flipped" does not mean "the remote half was booked".** The tx-count records only byte counts, not their origin; as long as the armed bytes do not exceed what the waiter
   delivered itself, local deliveries alone fill it. The most discriminating cell is one where the waiter **delivers not a single byte** and only the remote one remains: the probe's
   `pair_remote` cell (the leader issues no load, arms only the follower's 16 KB, the follower delivers to the leader with `.cta_group::2`)
   : the barrier flips and the follower's 16 KB is exact byte for byte — the remote complete_tx really was booked on the leader's barrier.
3. **A leader guard on the producer side = deadlock.** The follower does not issue its half and the leader's barrier is forever half the bytes short. Delivery is the business of both CTAs;
   waiting is the leader's alone.
4. **`expect_tx` must equal the sum of the bytes actually issued by all CTAs of the pair for this slot**, computed from the same tile-size constant; do not count the
   layout and copy numbers by hand. A phase completes when "pending arrivals reach zero and tx-count reaches zero": with **too few** counted (e.g. the leader arms only its own half),
   the bytes that land first complete the phase early — the waiter reads an unfinished slot without any error, the late complete_tx is booked into the next phase, and the next slot likewise
   flips early (silent wrong numbers; gotcha 2 is a special case of it); with **too many** counted (e.g. both CTAs arm) the tx-count never reaches zero and it hangs.
   This probe has no "too few / too many" negative cells; these two are derived from the PTX phase-completion rule.
5. **The first slot's arm must be visible before the peer's first delivery.** In the steady state the empty barrier covers it, in the prologue nothing does: the probe puts
   one cluster rendezvous (release / acquire) after init + arm and only then issues. The failure without this step is intermittent (it depends on the two CTAs' relative progress).
6. **Raw PTX has no built-in election: a load may be issued by only one thread.** If the whole warp executes this asm, that is 32 deliveries and 32× the bytes.
7. **Accept by the SASS suffix, not the numbers.** Writing `.cta_group::2` or not may be numerically identical (gotcha 2); only the opcode tells them apart:
   without the modifier the two read `UTMALDG.2D`, with `.cta_group::2` they read `UTMALDG.2D.2CTA` (`UTMALDG.3D.2CTA` with a rank-3 map).
8. **`.cta_group::2` needs `sm_100a`** (an architecture-specific target); the same instruction does not compile for a non-`a` target.

## Sources

- PTX ISA, `cp.async.bulk.tensor` (the `.cta_group` clause: with `::1` mbar must be in the same CTA as the destination smem, with `::2` it may be in the
  peer CTA; a remote barrier needs `.shared::cluster`), the definition of a CTA pair, `mapa`, `mbarrier`, `barrier.cluster`,
  `fence.mbarrier_init`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Driver API, `cuTensorMapEncodeTiled`: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- CUDA C++ Programming Guide, Distributed Shared Memory / Thread Block Clusters (`cudaLaunchKernelEx` +
  `cudaLaunchAttributeClusterDimension`): <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- The wrapper for the same mechanism in the public CUTLASS / CuTe C++ headers: `cute::SM100_TMA_2SM_LOAD` (this entry did not certify the wrapper path).
- GPU-certification probe — `PROBE PASS max_diff=0.0`. One process per spelling, cluster (2,1,1), smem prefilled with sentinels, bounded waits; `ctrl2d` (each arms and waits on its own, default modifier), `pair2d` (both CTAs issue `.cta_group::2`, delivering to the leader, the leader arming 32 KB), `pair_remote` (only the follower issues, the leader arms 16 KB and delivers not a single byte itself), `pair3d` all exact byte for byte with the leader flipping; `pair1_remote` (no modifier, redirected to the leader) records only the fact: the leader does not flip. SASS contract `tma-sass-cta-group-suffix`: `UTMALDG.2D` / `UTMALDG.2D.2CTA` / `UTMALDG.2D.MULTICAST.2CTA` each where expected.
