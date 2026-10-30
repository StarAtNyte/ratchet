# tma_2cta_nonmulticast_operand_load (which load string each operand uses in a `cta_group::2` pipeline: one shared across pairs uses `.multicast::cluster`, one belonging only to this pair uses `.cta_group::2` without multicast)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0.0` (`tma-pair2d`, `tma-mcast_self`, `tma-sass-cta-group-suffix`).**
>
> One file per hardware interface. Operator-agnostic: mechanism only. Prerequisite: [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md)
> (why the bytes of the pair's two halves are booked on the leader's single full barrier). The descriptor rank (`UTMALDG.<r>D`) is a separate
> knob, see [`tma_3d_operand_box_2cta`](tma_3d_operand_box_2cta.md).

## Purpose

In a `cta_group::2` pipeline with a cluster shaped `(2·m, n, 1)`, the two operands differ in "who needs these bytes":

| Operand | Shared across pairs? | PTX string | SASS |
|---|---|---|---|
| the one shared by several pairs along some mode of the cluster | yes | `...complete_tx::bytes.multicast::cluster.cta_group::2 [dst], [tmap, {..}], [mbar], ctaMask;` | `UTMALDG.<r>D.MULTICAST.2CTA` |
| the one belonging only to this pair (both operands when the cluster has only one pair) | no | `...complete_tx::bytes.cta_group::2 [dst], [tmap, {..}], [mbar];` | `UTMALDG.<r>D.2CTA` |

The latter can also be written as the multicast string with a `ctaMask` that "contains only itself" — it compiles, the bytes are bitwise identical and the accounting is right — but it is **a different
instruction**. Only the SASS suffix tells the two spellings apart.

## When needed (trigger)

- The contract / design table says an operand is `UTMALDG.*.2CTA` (non-multicast), but SASS has only `UTMALDG.*.MULTICAST.2CTA`:
  the typical cause is the two operands sharing one multicast issue function.
- The multicast mask has only its own bit (the corresponding cluster mode has size 1 after the pair split).
- **Anti-trigger**: both operands really are shared across pairs — both should be multicast; does not apply to `cta_group::1` pipelines.

## Minimal runnable example

```cpp
// belongs only to this pair: no multicast, with .cta_group::2, barrier delivered to the leader (mapa to rank & ~1)
__device__ __forceinline__ void tma2d_cg2(uint32_t dst, uint64_t tmap, int c0, int c1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
               " [%0], [%1, {%2, %3}], [%4];" :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(mbar) : "memory");
}
// shared across pairs: .multicast::cluster comes before .cta_group::2, ctaMask is a 16-bit operand ("h" constraint)
__device__ __forceinline__ void tma2d_mc_cg2(uint32_t dst, uint64_t tmap, int c0, int c1, uint32_t mbar,
                                             uint16_t cta_mask) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
               ".multicast::cluster.cta_group::2 [%0], [%1, {%2, %3}], [%4], %5;"
               :: "r"(dst), "l"(tmap), "r"(c0), "r"(c1), "r"(mbar), "h"(cta_mask) : "memory");
}
// producer (executed by both CTAs, issued by a single thread):
//   const uint32_t mb = mapa_u32(full_bar, rank & ~1u);
//   tma2d_mc_cg2(sA_stage, (uint64_t)&tmA, k0, m0, mb, mask_A);   // the shared path
//   tma2d_cg2  (sB_stage, (uint64_t)&tmB, k0, n0, mb);            // the path belonging only to this pair: no mask
// expect_tx, the consumer side and the MMA side do not change: the bytes landing in each CTA are the same.
```

## Gotchas

1. **One issue function feeding both operands.** The shared path uses the multicast string, the other path uses it too out of convenience, with a mask containing only itself — no error,
   no wrong numbers, only `.MULTICAST` left behind in SASS. Split it into two issue functions.
2. **Dropping `.cta_group::2` too when switching to the non-multicast string.** That is `cta_group::1`, and in a pair pipeline the follower's half cannot be booked on the
   leader's barrier (see gotcha 1 of the prerequisite entry). Non-multicast ≠ 1-CTA.
3. **Thinking that switching strings requires changing `expect_tx` or the consumer's arrival count.** It does not: the bytes each CTA receives and "who writes my slot" are both unchanged.
4. **Acceptance can only look at SASS.** Probe readings: the `.cta_group::2` string without multicast → `UTMALDG.2D.2CTA` ×2; the same load written as the multicast string with a `ctaMask` containing only itself → `UTMALDG.2D.MULTICAST.2CTA` ×2; in both cells the landed bytes are bitwise identical to the expectation.
5. **ctaMask is a 16-bit operand** (pass a `uint16_t` with the `"h"` constraint); it comes after `[mbar]` and before the optional cache policy.

## Sources

- PTX ISA, `cp.async.bulk.tensor` (`.multicast::cluster` and `ctaMask`, the semantics and writing order of the two `.cta_group` modifiers):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Driver API, `cuTensorMapEncodeTiled`: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- Wrappers of the same family in the public CUTLASS / CuTe C++ headers: `cute::SM100_TMA_2SM_LOAD`, `cute::SM100_TMA_2SM_LOAD_MULTICAST` (this wrapper path is uncertified).
- GPU-certification probe — `PROBE PASS max_diff=0.0`: in both the `pair2d` and `mcast_self` cells (cluster (2,1,1), both CTAs issue, both deliver to the leader's barrier, the leader arms the sum of both CTAs) the landed bytes are bitwise identical to the SW128 expectation, the leader flips, and the follower's own full barrier does not complete; SASS suffixes as in gotcha 4. A real multi-CTA mask across pairs was not tested by this probe (the cluster has only one pair); the PTX spelling of that row is given per the PTX ISA, uncertified.
