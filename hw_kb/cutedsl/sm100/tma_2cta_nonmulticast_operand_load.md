# tma_2cta_nonmulticast_operand_load (which load op each operand should use in a `cta_group::2` pipeline: operands shared across pairs go multicast, operands that belong only to this pair go through the non-multicast `.2CTA`)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> Prerequisite entry: [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md) (why the pair's
> two halves of bytes must be accounted by the leader's single full barrier, and what `.cta_group::2` unlocks). This entry answers only one question that one did not:
> **when the two operands have different sharing scopes, which load op should each operand use.** The descriptor rank
> (the r in `UTMALDG.<r>D`) is a separate knob, see [`tma_3d_operand_box_2cta`](tma_3d_operand_box_2cta.md).

## Purpose

A `cta_group::2` UMMA pipeline with a cluster shaped `(2·m, n, 1)` (a V=2 pair, then m along M and n along N).
The two operands differ in "who needs these bytes":

| Operand | Who needs it | Shared across pairs? | Op to use | SASS (`cache_ops` axis) |
|---|---|---|---|---|
| the one shared by several pairs along some cluster mode (e.g. A in cluster `(2,2,1)` with the two pairs laid out along N) | CTAs at the same V position in different pairs | **yes** | `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` + `mcast_mask` | `UTMALDG.<r>D.MULTICAST.2CTA` |
| the one that belongs only to this pair (its cluster mode has size 1 after the V split) | only this CTA | **no** | `CopyBulkTensorTileG2SOp(CtaGroup.TWO)`, **no** `mcast_mask` | `UTMALDG.<r>D.2CTA` |

The latter can also be written with the multicast op plus a "self-only" mask — it compiles, the values are right, and the barrier accounting is right too — but
that is **a different instruction** (the `.MULTICAST` encoding with a single self bit in the mask). The two spellings are bit-identical in value; only the SASS suffix
tells them apart. When the cluster has only one pair (`(2,1,1)`), **both** operands are of the latter kind.

## When needed (trigger)

- **Contract trigger**: the design document / contract table says an operand is `UTMALDG.*.2CTA` (non-multicast), but the census `cache_ops`
  only has `UTMALDG.*.MULTICAST.2CTA`. Typical cause: the host side built only **one** load op and
  `make_tiled_tma_atom_A` / `_B` share it.
- **Structural trigger**: the mode `m` in `create_tma_multicast_mask(..., mcast_mode=m)` has size 1 in `cta_layout_vmnk`
  — that mask holds only the self bit, and this load does not need multicast; hand-writing
  `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` + mask in cluster `(2,1,1)` is the same situation.
- **Anti-trigger**: both operands really are shared across pairs — then both should be the multicast op. Also not applicable to `cta_group=1` pipelines
  (there the non-multicast op is the default `CopyBulkTensorTileG2SOp()`).

## Mechanism

1. **`.cta_group::2` and multicast are orthogonal.** The former decides that the completion signal may land on the pair peer's (leader's)
   barrier; the latter decides how many CTAs' smem the bytes are written into. So an operand that "belongs only to this pair" wants
   `.cta_group::2` **without** multicast: switch the op to `CopyBulkTensorTileG2SOp(CtaGroup.TWO)`.
2. **Keep using the barrier pointer the pipeline hands back.** As in the multicast version, pass the local
   full barrier handed back by `PipelineTmaUmma` (the `.barrier` of the handle returned by `acquire_and_advance()`) directly to `tma_bar_ptr=`. The certification probe's
   C1 is written exactly this way: the leader arms for the whole pair's bytes, only the leader waits, the kernel completes without hanging and the values are bit-exact —
   the follower half's completion really was accounted on the leader's barrier. An explicit `map_dsmem_ptr` to the leader is **uncertified**
   in this entry: in the build used for certification `cute.arch` has no `map_dsmem_ptr` (probe C3 hits
   `AttributeError: module 'cutlass.cute.arch' has no attribute 'map_dsmem_ptr'` at trace time). Before using it in a hand-written form, check
   `hasattr(cute.arch, "map_dsmem_ptr")` first.
3. **The accounting does not change by a character.** `tx_count` is still "bytes of this CTA's A+B faces for one stage × pair size"; switching multicast on or off
   does not change the number of bytes each CTA receives. The `PipelineTmaUmma` consumer arrival count
   (`size(vmnk[2]) + size(vmnk[1]) - 1`) is also unchanged.
4. **The producer issues in both CTAs**, no leader guard (same as gotcha 2 of the prerequisite entry).

## Minimal runnable example (CuTe DSL)

Host side: **each** operand gets its own op (this one spot is the whole mechanism).

```python
from cutlass.cute.nvgpu import cpasync, tcgen05

cta_layout_vmnk = cute.tiled_divide(cute.make_layout(CLUSTER_MNK), (tiled_mma.thr_id,))
# CLUSTER_MNK = (2, 2, 1): V=2 (pair), two pairs laid out along N -> A is shared across both pairs, B belongs only to this pair

op_a = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)   # multicast across pairs
op_b = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)            # non-multicast, .2CTA

a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op_a, a, a_one_stage_layout, MMA_TILER_MNK, tiled_mma, cta_layout_vmnk.shape)
b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
    op_b, b, b_one_stage_layout, MMA_TILER_MNK, tiled_mma, cta_layout_vmnk.shape)
```

Device side: partitioning unchanged; B's `cute.copy` **drops `mcast_mask`**.

```python
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a, coord_vmnk[2], cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
    cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
tBsB, tBgB = cpasync.tma_partition(
    tma_atom_b, coord_vmnk[1], cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
    cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
mcast_a = cpasync.create_tma_multicast_mask(cta_layout_vmnk, coord_vmnk, mcast_mode=2)

# producer warp, executed by both CTAs (no leader guard)
ab_empty = ab_producer.acquire_and_advance()
cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)],
          tma_bar_ptr=ab_empty.barrier, mcast_mask=mcast_a)
cute.copy(tma_atom_b, tBgB[(None, ab_empty.count)], tBsB[(None, ab_empty.index)],
          tma_bar_ptr=ab_empty.barrier)                      # no mcast_mask
```

`tx_count`, consumer arrival count, the MMA side, the commit mask: **all unchanged**.

**Let the helper pick the op** (equivalent, and cannot fall into gotcha 1):

```python
import cutlass.utils.blackwell_helpers as sm100_utils
op_a = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_MNK, tiled_mma.thr_id)
op_b = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_MNK, tiled_mma.thr_id)
# measured: (2,2,1) -> A = CopyBulkTensorTileG2SMulticastOp(TWO), B = CopyBulkTensorTileG2SOp(TWO)
#           (2,1,1) -> both A and B are CopyBulkTensorTileG2SOp(TWO) (SASS has only UTMALDG.*.2CTA, no MULTICAST)
# for the stream that got the non-multicast op, cute.copy gets no mcast_mask.
```

## Gotchas

1. **One op feeding both atoms.** `tma_load_op = CopyBulkTensorTileG2SMulticastOp(TWO)` then shared by A and B —
   no error, no wrong values, B just stays on the `.MULTICAST` encoding. When changing it, **split it into two op variables** (or leave it to the helper).
2. **Op switched, `mcast_mask=` still there.** Change in pairs: "switch the op ⇒ delete that copy's `mcast_mask` argument".
3. **Dropping `.2CTA` along with it.** `CopyBulkTensorTileG2SOp()` (default `CtaGroup.ONE`) in a pair pipeline
   is the prerequisite entry's number-one hang form: the follower half's bytes are not accounted on the leader's barrier. Non-multicast ≠ 1-CTA.
4. **Thinking that switching op requires changing the tx / arrival counts.** It does not. The bytes landing in each CTA and "who writes my slot" are unchanged.
5. **Accept on the SASS, not the values.** The two spellings are bit-identical in value (probe C0 vs. C1), so acceptance can only read `cache_ops`:
   a `UTMALDG.*.2CTA` without `MULTICAST` appears, while the shared stream is still `UTMALDG.*.MULTICAST.2CTA`.
6. **Switching op does not change the descriptor rank.** Both streams are `.2D` in this entry's certified readings. Trying to bump to `.3D` along the way with an `(M, K, 1)` tensor
   does not work: in probe C2 the static extent-1 L mode obtained from torch `unsqueeze(-1)` is folded away by the constructor, and it still reads
   `UTMALDG.2D.2CTA`. How to actually get `.3D`: see [`tma_3d_operand_box_2cta`](tma_3d_operand_box_2cta.md).
7. **Multicast disappearing when the cluster is down to one pair is a footprint of the geometry, not of this knob.** When the cluster goes from `(2,2,1)` to
   `(2,1,1)`, the helper picks the non-multicast `.2CTA` op for both streams (see the second probe under "Sources"), and `MULTICAST` disappears from `cache_ops`
   accordingly; the ms change of that edit belongs to the cluster / grid / tile shape and cannot be used to price "non-multicast load".

## Sources

- Certification probe (a GPU probe, not included in this release) — **Status: ✅ Verified**
  (`PROBE PASS max_diff=0.0`, certification round 2; three contracts all passed: T0 compile / T1 C1, C2 bit-identical to the fp32 reference /
  T2 SASS divergence). Variants: C0 both streams on the multicast op (SASS only `UTMALDG.2D.MULTICAST.2CTA`);
  C1 B switched to the non-multicast `.2CTA` op, local pipeline barrier pointer (`UTMALDG.2D.2CTA` + `UTMALDG.2D.MULTICAST.2CTA`,
  max_diff 0.0); C2 = C1 with an `(M,K,1)` tensor from torch `unsqueeze(-1)` (TMA coordinate tensor `(512,256,1):(1@1,1@0,0)`,
  still `.2D`, max_diff 0.0); C3 = C1 with an explicit `map_dsmem_ptr` to the leader (`AttributeError` at trace time, no cubin,
  fact recorded only); C4 helper picks the ops (cluster `(2,2,1)`: A multicast, B non-multicast, SASS same as C1, max_diff 0.0).
  Geometry: pair MMA `256×128×16`, tiler `(256,128,64)`, cluster `(2,2,1)`, small-integer fp16 inputs, fp32 accumulation.
- In cluster `(2,1,1)` the helper picks the non-multicast `.2CTA` op for both streams: another probe,
  `PROBE PASS max_diff=0.0`, SASS reads `UTMALDG.2D.2CTA` /
  `UTMALDG.3D.2CTA`, neither with `MULTICAST`.
- Public API: `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`, `CopyBulkTensorTileG2SMulticastOp`,
  `cpasync.tma_partition`, `cpasync.create_tma_multicast_mask`, `cutlass.cute.nvgpu.make_tiled_tma_atom_A` / `_B`,
  `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A` / `_B`, `cutlass.cute.nvgpu.tcgen05.CtaGroup`,
  `cutlass.pipeline.PipelineTmaUmma`.
- PTX ISA, `cp.async.bulk.tensor` (the respective semantics of the `.multicast::cluster` and `.cta_group` modifiers):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
