# tma_multicast_cluster (SM90 cluster launch + TMA multicast)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry covers a set of four: **the launch-side spelling of the cluster topology**, **construction of
> the multicast TMA atom**, **the box split of `tma_partition`**, and **the expect-tx / arrive counts of
> the cluster mbarrier**. WGMMA / warp specialization / swizzle are not included (each has its own entry).

## Purpose

With cluster = `[N,1,1]` (N CTAs forming one hardware cluster), one gmem tile can be **fetched 1/N by each**
CTA in the cluster and then **multicast inter-CTA** by the TMA unit into the smem of every CTA in the
cluster. The gmem read volume of the same data drops to 1/N (L2/HBM bandwidth), while every CTA still gets
the complete tile. Typical use: the A side of a GEMM/conv multicast along N (the same activation rows fed
to the CTAs of different N blocks), the B side multicast along M; when only one side multicasts, the other
uses `num_multicast=1`.

## When needed (trigger)

- The milestone froze `cluster=[N,1,1]` but the delivered tree's launch says `cluster=[1,1,1]` and all
  atoms are `num_multicast=1` — both places must change together; changing only one is the same as
  changing nothing (gotcha 1).
- Profiler symptoms appearing together: `tensor_pct` clearly below the performance baseline, `stall_long_sb`
  elevated, `warp_active_pct` low, `dram_read` elevated — the producer cannot keep the consumer fed; first
  check whether multicast is really on.
- Any scenario that needs "two CTAs reading the same data" (activation-stationary, split-N).

## Minimal runnable example

Example: cluster `[2,1,1]`, A-side multicast (`num_multicast=2`). The four parts together form the complete
path; getting any single part wrong hangs or silently loses data.

### 1) Host side: multicast TMA atom

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync

CLUSTER = 2                       # number of CTAs in the cluster
NUM_MULTICAST = CLUSTER

# smem layout: slice off the stage mode before passing it to the atom builder (single-stage view)
smem_one = cute.slice_(a_smem_layout_staged, (None, None, 0))

# <<< multicast has its own op class >>> passing num_multicast != 1 to the plain CopyBulkTensorTileG2SOp
# raises a ValueError on the host side right away:
#   "expects num_multicast to be 1 for non multicast G2S copies"
tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SMulticastOp(),   # not CopyBulkTensorTileG2SOp
    a,                            # gmem tensor
    smem_one,                     # single-stage smem layout (swizzled or a bare layout)
    (TILE_M, TILE_K),             # cta_tiler
    num_multicast=NUM_MULTICAST,  # <<< must equal the cluster extent in that direction
)

# The non-multicast side stays as before (and the op class must be switched back too):
tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), b, smem_b_one, (TILE_N, TILE_K),
    num_multicast=1)
```

The official CUTLASS hopper dense_gemm picks the op class this way, one of two by `mcast_dim == 1`
(`_make_tma_atoms_and_tensors`).

### 2) Host/kernel side: `cta_layout_vmnk` and pipeline counts

```python
# the (M,N) layout of the cluster, with a V mode prepended:
#   V = the 2-SM mode of tcgen05, **always 1 on sm90**
#   M/N = the two axes of the cluster
cta_layout_mnk  = cute.make_layout((CLUSTER_M, CLUSTER_N, 1))
cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
# with A multicast along N and B along M: for cluster=[2,1,1] to multicast A, take
#   (CLUSTER_M, CLUSTER_N) = (1, 2) -> cta_layout_vmnk = (1, 1, 2, 1)

num_mcast_ctas_a = CLUSTER_N      # number of CTAs the A side multicasts to
num_mcast_ctas_b = CLUSTER_M      # number of CTAs the B side multicasts to
mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1     # = 1 when neither side multicasts

mainloop_pipeline = pipeline.PipelineTmaAsync.create(
    barrier_storage=storage.mbar.data_ptr(),
    num_stages=AB_STAGE,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    # <<< multiply by mcast_size by hand >>> consumer_release does a REMOTE arrive to every
    # CTA in the cluster, so each empty barrier receives, per stage,
    #   (number of warps calling consumer_release) x mcast_size arrivals.
    consumer_group=pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_consumer_warps * mcast_size),
    # <<< tx_count is not multiplied >>> this CTA issues only 1/N of the box; the rest is multicast into
    # this CTA's barrier by the peer CTAs, adding up to exactly one complete box.
    tx_count=tma_copy_bytes,
    cta_layout_vmnk=cta_layout_vmnk,   # pass (1,1,1,1) without a cluster
)
```

When `size(cta_layout_vmnk) > 1`, `create()` **internally** already issues
`mbarrier_init_fence()` + `cluster_arrive_relaxed()` + `cluster_wait()`
(`agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)` in `pipeline/sm90.py`;
that helper contains both the arrive and the wait). Therefore:

- **do not** hand-write another `sync_threads()` or `cluster_arrive_relaxed/cluster_wait`;
- `create()` must be reached by **all threads**, i.e. placed **before** the producer/consumer branches;
- to separate the arrive from the wait (doing unrelated work such as partitioning in between), pass
  `defer_sync=True` and then call `pipeline_init_arrive(...)` / `pipeline_init_wait(...)` yourself, in
  order — this is the route the official dense_gemm takes.

### 3) Kernel side: mcast_mask and the box split of `tma_partition`

```python
# this CTA's linear rank in the cluster
cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
cta_coord_vmnk = cta_layout_vmnk.get_hier_coord(cta_rank)

# mask: which mode to multicast along. In VMNK mode 2 = N (A side), mode 1 = M (B side).
# Do not hand-write (1 << CLUSTER) - 1: it is only accidentally right when "the whole cluster is one
# multicast group", and silently wrong when the cluster is 2-D.
a_mcast_mask = cpasync.create_tma_multicast_mask(
    cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2)

# <<< where the box is split >>> cta_coord = this CTA's coordinate in the multicast group,
# cta_layout = the multicast group's layout. Passing (0, cute.make_layout(1)) makes every CTA
# issue the same slice: the barrier never receives a full box -> hang.
a_cta_layout = cute.make_layout(
    cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a,
    cta_coord_vmnk[2],            # A along N; the B side uses cta_coord_vmnk[1]
    a_cta_layout,
    cute.group_modes(sA, 0, 2),
    cute.group_modes(gA, 0, 2),
)

# producer: the multicast copy must carry mcast_mask, and the mask must be a cutlass Integer
# (create_tma_multicast_mask returns Int16); a Python int is rejected by unpack.
mainloop_pipeline.producer_acquire(pstate)
cute.copy(tma_atom_a,
          tAgA[(None, pstate.count)],
          tAsA[(None, pstate.index)],
          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate),
          mcast_mask=a_mcast_mask)
cute.copy(tma_atom_b,                       # the non-multicast side carries no mask
          tBgB[(None, pstate.count)], tBsB[(None, pstate.index)],
          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(pstate))
mainloop_pipeline.producer_commit(pstate)
```

The consumer side is **word for word the same** as the no-cluster version (`consumer_wait` /
`consumer_release`): the remote arrive is done inside the pipeline by `dst_rank` + `is_signalling_thread` —
lanes `0..cluster_size-1` of each warp each send one arrive to one CTA, the other lanes send none. So the
arrive count depends only on "**the number of warps calling consumer_release**", not on the number of
threads in a warp.

### 4) Launch side

```python
kernel(...).launch(
    grid=[grid_m, grid_n, 1],      # every dimension must be an integer multiple of the corresponding cluster dimension
    block=[NUM_THREADS, 1, 1],
    cluster=[CLUSTER_M, CLUSTER_N, 1],   # size must == size(cta_layout_vmnk)
    stream=stream)
```

## Gotchas

1. **`num_multicast` and `cluster` must change together.**
   `num_multicast=2` + `cluster=[1,1,1]`: the two CTAs are not in the same cluster, the multicast target
   does not exist — illegal/hang. `num_multicast=1` + `cluster=[2,1,1]`: runs and gives the right result,
   but each CTA fetches its own copy and **not a single byte of bandwidth is saved** — this is the typical
   silent scene of "thought the cluster was already on", only discoverable through the profiler
   (dram_read / tensor_pct).

2. **The V mode of `cta_layout_vmnk` is always 1 on sm90; the cluster axes go into M/N.**
   The correct construction is `(1, *cta_layout_mnk.shape)`, i.e. `(1, CLUSTER_M, CLUSTER_N, 1)`.
   Writing the cluster extent into V (e.g. `(2,1,1,1)`) misaligns both the pipeline's mcast-mode decision
   and the mode numbering of `create_tma_multicast_mask`.

3. **mcast_mask must be computed by `create_tma_multicast_mask` / `make_layout_image_mask`.**
   It requires both `cta_layout_vmnk` and `cta_coord_vmnk` to be **rank 4**, otherwise a host-side
   ValueError. The return value is `Int16`; the multicast unpack of `cute.copy` explicitly checks
   `isinstance(mcast_mask, Integer)`, and passing a Python int raises
   "expects a multicast mask to be provided via the mcast_mask kw argument".

4. **The `cta_coord`/`cta_layout` of `tma_partition` are the box splitter.**
   On the multicast path you must pass this CTA's coordinate within the multicast group and that group's
   layout; keeping the unicast `(0, cute.make_layout(1))` makes the N CTAs issue the same slice, the
   transaction barrier only ever receives 1/N of the bytes, and the first `consumer_wait` hangs forever.

5. **`tx_count` is not multiplied by N; the consumer arrive count is.** The two go in opposite directions
   and are easy to get both wrong at once:
   - `tx_count` = the bytes of one complete box (this CTA issues half, the peer CTA multicasts the other
     half); multiply by N and it never fills → hang.
   - consumer arrive = `number of warps × mcast_size`; without the multiplication the empty barrier flips
     early and the producer overwrites a stage still being read → silent data race.

6. **`PipelineTmaAsync.create()` carries its own cluster handshake and must be executed by everyone.**
   Putting it inside a branch such as `if warp_idx == 0:` = some threads skip the cluster barrier → hang.
   Conversely, adding another hand-written `cluster_arrive_relaxed/cluster_wait` after `create()` is
   redundant (and getting the pair wrong is a new deadlock source).

7. **Every dimension of `grid` must be an integer multiple of the corresponding cluster dimension.** If it
   does not divide, an incomplete cluster is left over and the behaviour is undefined. When the tile count
   does not divide, pad to a multiple, or fall back to `cluster=[1,1,1]`.

8. **sm_90a target.** Cluster instructions (`block_idx_in_cluster`, `cluster_arrive_relaxed`, multicast
   TMA) are rejected by ptxas under `sm_90` (no `a`). `CUTE_DSL_ARCH=sm_90a` is required
   (same as the warp_specialization entry).

9. **Multicast requires the target smem offsets to be the same in every CTA of the cluster.** The same
   kernel with the same `SmemAllocator` order satisfies this naturally; if the allocation order changes in a
   branch on `cta_rank`, data is written to the wrong offset of the peer CTA (silent wrong data).

10. **`CopyBulkTensorTileG2SMulticastOp()` can only be constructed inside a DSL context.**
    Its `__post_init__` reads `BaseDSL._get_dsl().get_arch_enum()` for an arch check; constructing it at
    module level (outside any `@cute.jit`) fails right away.

## Sources

- **CuTe DSL / CUTLASS official sources (the 4.4.2 python package agrees with the 4.6.1 examples)**:
  - op-class choice and `num_multicast`: `cutlass/cute/nvgpu/cpasync/helpers.py`
    `make_tiled_tma_atom` (multicast branch `isinstance(op, CopyBulkTensorTileG2SMulticastOp)`);
    usage in `_make_tma_atoms_and_tensors` of
    `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py`
  - `cta_layout_vmnk = (1, *cta_layout_mnk.shape)`, `mcast_size = a + b - 1`,
    `consumer_arrive_cnt = mcast_size * num_warps`: the kernel body of the same dense_gemm.py
  - cluster handshake: `cutlass/pipeline/sm90.py` `PipelineTmaAsync.create`
    (the `defer_sync=False` branch calls `agent_sync(Agent.ThreadBlockCluster, is_relaxed=True)`)
    and `cutlass/pipeline/helpers.py` `agent_sync` (arrive + wait as a pair)
  - lane selection of the remote arrive: `pipeline/sm90.py`
    `PipelineTmaAsync.init_empty_barrier_arrive_signal` (`tidx % 32 < cluster_size`,
    `dst_rank = tidx % cluster_size`) and `consumer_release`
  - mask and split: `cpasync/helpers.py` `create_tma_multicast_mask` (rank-4 check),
    `tma_partition`; `cpasync/copy.py`
    `CopyBulkTensorTileG2SMulticastNonExecTrait.unpack` (`mcast_mask` must be an Integer)
  - a bare layout (no swizzle) can also build a TMA atom:
    `examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/tma_v0.py`
- **GPU certification**: ⬜ to be filled in once a GPU probe passes.
