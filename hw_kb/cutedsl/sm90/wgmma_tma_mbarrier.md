# wgmma_tma_mbarrier (CuTeDSL: TMA load → mbarrier pipeline → WGMMA SS main loop)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers the minimal
> interface trio of the SM90 main loop: `tma_load` (bulk async G2S) + `mbarrier` (producer/consumer sync) +
> `wgmma` (warpgroup asynchronous MMA). On Hopper the three form one data path; written separately they
> could not give a runnable example, so they are merged into one entry.

## Purpose

Move two fp16 tiles from gmem into 128B-swizzled smem with bulk asynchronous TMA, use an mbarrier (wrapped
by `cutlass.pipeline.PipelineTmaAsync`) for the producer/consumer handshake, then have one warpgroup (128
threads) issue WGMMA SS directly from smem (A and B both in smem), accumulating into registers.
This is the foundation path of the main loop of any "smem-resident + tensor core" kernel on SM90.

## When needed (trigger)

- To saturate SM90 compute with the tensor cores: the plain `cp.async` + `mma.sync` path cannot feed them;
  you need TMA (one instruction moves a whole tile, no register transit) + WGMMA (warpgroup-level async
  MMA that reads smem directly).
- Symptom: early kernel stage — there is not yet any working minimal TMA/WGMMA example, or the probe keeps
  getting stuck on the argument shapes of `make_tiled_tma_atom` / `tma_partition` / mbarrier phase /
  `cute.gemm`.
- Dataflow: gmem tile → multi-stage smem ring buffer → tensor cores; the consumer must wait for "all bytes of
  this stage have arrived" before issuing the MMA.

## Minimal runnable example

The complete path with a single CTA, a single warpgroup, 2 k-tiles and 2 smem stages. No warp
specialization, no cluster, no persistent scheduler (those are higher-level structures with their own
entries). GPU-certified in the target environment (see "Sources").

### 1) Host side: TiledMma, staged smem layout, TMA atom

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup

TILE_M, TILE_N, TILE_K = 64, 64, 64   # the WGMMA atom's M-mode must = 64 (see gotcha 1)
AB_STAGE = 2                           # number of stages of the smem ring buffer

# --- TiledMma: single atom (1,1,1), tiler_mn=(64, TILE_N), SS form ---
# the major mode is derived from the tensor layout; for fp16 k-major the atom K-mode = 16,
# i.e. TILE_K=64 contains 4 k-blocks (the main loop below issues cute.gemm per k-block).
a_layout = cutlass.utils.LayoutEnum.from_tensor(a)     # a: cute.Tensor (M,K) fp16
b_layout = cutlass.utils.LayoutEnum.from_tensor(b)     # b: (N,K)
tiled_mma = sm90_utils.make_trivial_tiled_mma(
    cutlass.Float16, cutlass.Float16,
    a_layout.sm90_mma_major_mode(), b_layout.sm90_mma_major_mode(),
    cutlass.Float32, (1, 1, 1), tiler_mn=(64, TILE_N),
)

# --- staged 128B-swizzled smem layout: atom -> tile_to_shape to (tile, tile_k, stage) ---
# get_smem_layout_atom picks the swizzle level (SW128 etc.) from (layout, dtype, major-mode length).
a_atom = warpgroup.make_smem_layout_atom(
    sm90_utils.get_smem_layout_atom(a_layout, cutlass.Float16, TILE_K),
    cutlass.Float16)
a_smem_layout_staged = cute.tile_to_shape(
    a_atom, (TILE_M, TILE_K, AB_STAGE), order=(0, 1, 2))   # k-major: order=(0,1,2)
# B likewise, shape=(TILE_N, TILE_K, AB_STAGE)

# --- TMA atom: the smem layout given to the atom builder must have the stage mode sliced off first ---
smem_layout_1stage = cute.slice_(a_smem_layout_staged, (None, None, 0))
tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),        # bulk G2S without multicast
    a, smem_layout_1stage, (TILE_M, TILE_K),  # gmem tensor, single-stage smem layout, tile shape
    num_multicast=1,
)
# The return value is a pair: atom + a "coordinatised" gmem tensor. The kernel must use tma_tensor_a,
# not the original a (TMA addresses through the descriptor; only the coordinate tensor carries the right mapping).
```

### 2) SharedStorage: mbarrier array and aligned smem buffers

```python
@cute.struct
class SharedStorage:
    # PipelineTmaAsync needs two mbarriers per stage, full + empty, each an Int64
    mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGE * 2]
    sA: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float16, cute.cosize(a_smem_layout_staged)],
        1024]   # TMA destination buffers need large alignment (at least 128B; 1024 is safe)
    sB: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float16, cute.cosize(b_smem_layout_staged)],
        1024]
```

### 3) Kernel side: pipeline creation, TMA partition, producer/consumer

```python
@cute.kernel
def kernel(tma_atom_a, mA, tma_atom_b, mB, mC, tiled_mma,
           a_smem_layout_staged, b_smem_layout_staged):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:                              # descriptor prefetch (optional optimisation)
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # TMA transaction bytes per stage = one stage of A + one stage of B (fed to expect_tx)
    tma_copy_bytes = cute.size_in_bytes(
        cutlass.Float16, cute.slice_(a_smem_layout_staged, (None, None, 0))
    ) + cute.size_in_bytes(
        cutlass.Float16, cute.slice_(b_smem_layout_staged, (None, None, 0)))

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # --- mbarrier pipeline: PipelineTmaAsync wraps the full/empty phase flips ---
    # producer_group: TMA is issued by a single representative thread (leader chosen internally);
    # the arrive count of consumer_group = number of consumer warps (single warpgroup = 4).
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
        num_stages=AB_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
        tx_count=tma_copy_bytes,                   # bytes expected to arrive per stage
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),   # no cluster
    )

    # swizzled smem tensors: outer is the shape layout, inner is the swizzle; pass them separately
    sA = storage.sA.get_tensor(a_smem_layout_staged.outer,
                               swizzle=a_smem_layout_staged.inner)
    sB = storage.sB.get_tensor(b_smem_layout_staged.outer,
                               swizzle=b_smem_layout_staged.inner)

    # tile the gmem (this CTA takes row block m=0; the k dimension is left None for the main loop to iterate)
    gA = cute.local_tile(mA, (TILE_M, TILE_K), (0, None))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (0, None))

    # --- TMA partition: group the first two modes on both smem and gmem first ---
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),        # no cluster: cta_crd=0, layout=1
        cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

    # --- WGMMA fragments: SS form, A/B are both smem descriptors ---
    thr_mma = tiled_mma.get_slice(tidx)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    acc = cute.make_fragment(
        thr_mma.partition_C(cute.make_identity_tensor((TILE_M, TILE_N))).shape,
        cutlass.Float32)

    cute.arch.sync_threads()   # the mbarrier init must be visible to the whole CTA before use

    # --- producer: issue TMA. Each time acquire -> copy (with tma_bar_ptr) -> commit -> advance ---
    if warp_idx == 0:
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, AB_STAGE)
        for _ in range(K_TILES):
            mainloop_pipeline.producer_acquire(producer_state)   # wait for the stage to be empty
            cute.copy(tma_atom_a,
                      tAgA[(None, producer_state.count)],        # k-tile number count
                      tAsA[(None, producer_state.index)],        # stage number index
                      tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                          producer_state))
            cute.copy(tma_atom_b, tBgB[(None, producer_state.count)],
                      tBsB[(None, producer_state.index)],
                      tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                          producer_state))
            mainloop_pipeline.producer_commit(producer_state)    # a NOP under TMA
            producer_state.advance()

    # --- consumer: wait for all bytes -> issue WGMMA per k-block -> release the stage ---
    consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, AB_STAGE)
    acc.fill(0.0)
    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)   # accumulate into acc instead of overwriting
    warpgroup.fence()                                 # register fence before WGMMA
    num_k_blocks = cute.size(tCrA, mode=[2])          # TILE_K/atom_k (fp16: 4)
    for _ in range(K_TILES):
        mainloop_pipeline.consumer_wait(consumer_state)          # wait for the full phase
        for kb in cutlass.range_constexpr(num_k_blocks):
            cute.gemm(tiled_mma, acc,
                      tCrA[(None, None, kb, consumer_state.index)],
                      tCrB[(None, None, kb, consumer_state.index)],
                      acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)          # release only after all WGMMAs of this stage have retired
        mainloop_pipeline.consumer_release(consumer_state)       # arrive empty
        consumer_state.advance()
    # acc now holds this thread's C fragment; for landing it in smem/gmem see the epilogue-type entries
    # (flat_tv_r2s_cfrag / cfrag_epilogue_stmatrix).
```

launch: `kernel(...).launch(grid=[...], block=[128,1,1], cluster=[1,1,1], stream=...)`;
when `SharedStorage` goes through `cute.struct` there is no need to compute the `smem=` byte count by hand.

## Gotchas

1. **The M-mode of the WGMMA atom must be 64.** Passing `tiler_mn=(128, N)` to `make_trivial_tiled_mma`
   with a single atom raises `OpError: expects the M-mode to be 64`. A tile M=128 is obtained by stacking
   atoms along M with `atom_layout_mnk=(2,1,1)` (2 warpgroups = 256 threads), not by widening the atom.
2. **The TMA atom builder only takes a single-stage smem layout.** You must first
   `cute.slice_(staged, (None, None, 0))` and then pass it to `make_tiled_tma_atom`; passing the 3-mode
   layout with the stage directly gives a shape mismatch.
3. **The kernel must use the coordinate tensor returned by `make_tiled_tma_atom`** (`tma_tensor_a` in the
   example), not the original gmem tensor — TMA addresses through the descriptor; the original tensor
   pointer is meaningless in the kernel.
4. **`group_modes(t, 0, 2)` is required before `tma_partition`**, grouping (tile_i, tile_j) into one mode;
   on both the smem and the gmem side. Forgetting it reports a rank/profile mismatch.
5. **`cute.copy` must carry `tma_bar_ptr=`.** TMA's completion signal goes through the mbarrier's
   `expect_tx/arrive`; without the barrier pointer the consumer never sees the full phase (hang).
   With `PipelineTmaAsync` take the pointer from `producer_get_barrier(state)`.
6. **`tx_count` = the total bytes expected by that barrier for one stage** (A+B together,
   `tma_copy_bytes` in the example). Too low releases early (reading data that has not arrived, silent
   numerical error); too high hangs.
7. **There must be a CTA sync between mbarrier initialisation and use** (`cute.arch.sync_threads()`; with a
   cluster `cluster_arrive_relaxed/cluster_wait`). Only some threads do the init; using it without a sync is
   a race.
8. **`count` and `index` of the pipeline state are two different quantities**: `count` is the monotonic
   k-tile counter (which gmem tile), `index` is the ring stage index (which smem stage). Index the gmem side
   with `count` and the smem side with `index`; mixing them up reads the wrong tile or writes the wrong
   stage. When reusing the state across multiple tiles, `reset_count()` at the start of each tile.
9. **The order of the WGMMA trio is fixed**: `warpgroup.fence()` before writing acc; `commit_group()` after
   each batch of `cute.gemm`; `wait_group(N)` before releasing that stage (N = number of batches allowed in
   flight, conservatively 0). Calling `consumer_release` before `wait_group` lets the producer overwrite
   smem that WGMMA is still reading. Accumulation semantics need an explicit
   `tiled_mma.set(warpgroup.Field.ACCUMULATE, True)`, otherwise each `cute.gemm` overwrites acc.
10. **For fp16 SS, `cute.gemm` is issued per k-block**: one stage of TILE_K=64 contains
    `num_k_blocks = cute.size(tCrA, mode=[2])` atom-Ks (4 for fp16), unrolled with
    `cutlass.range_constexpr`, fragment index `(None, None, kb, stage_index)`.
11. **The producer/consumer counts must match the CooperativeGroup**: the consumer arrive count = the
    number of warps that actually call `consumer_release` (single-warpgroup fused form = 4; in a real
    warp-specialized kernel it is mcast_size × number of mma warpgroups × 4). With a wrong count the empty
    phase never flips.
12. **The non-warp-specialized (fused) form is deadlock-free only when `AB_STAGE >= number of k-tiles`**:
    the producer segment runs before the consumer segment; if there are fewer stages than k-tiles,
    `producer_acquire` waits for a release that never comes. A real main loop has far more k-tiles than
    stages and must use warp specialization (DMA warp and MMA warpgroup advancing in parallel, see the
    `warp_specialization` entry).
13. **The file must not contain `from __future__ import annotations` (PEP 563).** That import turns all
    `@cute.struct` field annotations and `@cute.kernel`/`@cute.jit` parameter annotations into strings,
    while CuTeDSL builds types from **the annotation objects themselves**; the symptom is
    `cute.compile` raising
    `TypeError: Struct element only support struct/array/base_dsl scalar, but got
    "cute.struct.MemRange[...]"` (the value is a string). Every Python file containing CuTeDSL
    struct/kernel definitions must drop this line (measured on GPU).

## Sources

- GPU-certified by a GPU probe (the probe is not included in this release; single CTA, 2 k-tiles,
  2 stages, integer fp16 inputs vs an exact fp32 reference) — **passed on H100:
  `PROBE PASS max_diff=0.0`**.
- Official reference: cutlass examples `examples/python/CuTeDSL/hopper/dense_gemm.py` (same API surface).
