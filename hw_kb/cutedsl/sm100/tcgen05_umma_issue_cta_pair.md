# tcgen05_umma_issue_cta_pair (the full assembly of a tcgen05 UMMA mainloop: smem face → TMA → issue → commit → TMEM readback, single CTA and CTA pair; the multicast form of a hand-written pair commit)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry answers one question: **in CuTe DSL, from "build the atom" to "read the accumulator back to gmem", which API does
> each step of a `tcgen05.mma` call, and what shape must each tensor have?** The example below is a complete GEMM mainloop that is
> bit-exact on the GPU, in two forms: `CtaGroup.ONE` (single CTA) and `CtaGroup.TWO` (a CTA pair in cluster (2,1,1)).
>
> It is not a "reference" to rewrite at will; it is a **path proven to pass the verifier**: copy it and get it running first, then
> swap in your own geometry one place at a time. Nine common errors are indexed by their error text in "When needed", each mapped
> to one line of the example.

## Purpose

Blackwell UMMA has five mutually coupled "shape constraints"; get any one wrong and the error surfaces somewhere else:

| Stage | Hardware fact | Corresponding shape in the DSL |
|---|---|---|
| smem operand face | UMMA reads operands through an **smem descriptor**; the swizzle mode is a field of the descriptor | the tensor's layout must be **affine**; the swizzle hangs on the **pointer** (`allocate_tensor(layout=L.outer, swizzle=L.inner)`), you cannot use the composed layout as the tensor layout |
| operand partitioning | one instruction consumes one (MMA_MN × MMA_K) block; a k-tile is several instructions | the tensor fed to `cute.gemm` must have profile `(MMA, MMA_MN, MMA_K[, STAGE])` — produced by `tiled_mma.make_fragment_A/B(smem tensor)`, not the (mn, k) face itself |
| accumulator | lives in TMEM, addressed by 32-bit columns, allocate before use | `tiled_mma.make_fragment_C(partition_shape_C(tiler_mn))` gives the layout, `TmemAllocator` gives the pointer, `cute.make_tensor` combines them |
| issue and completion | MMA is asynchronous; completion is signalled by `tcgen05.commit` onto an mbarrier | `PipelineTmaUmma` (TMA full / commit empty) and `PipelineUmmaAsync` (acc full / empty) already wrap the commit; `release()`/`commit()` is it |
| pair | a `cta_group::2` MMA is issued by the leader and reads both sides' smem; TMA completion lands on the leader's barrier | `get_slice(rank)`, `cta_layout_vmnk`, `tx_count × pair`, the `is_leader_cta` guard, `tail()` — all in the example |

## When needed (trigger)

**Look up by the error you see** (each row has the correct form in the example below):

| Error / symptom | What you did | Corresponding line in the example |
|---|---|---|
| `Expected affine layout for operand A, but got composed layout` | built the smem tensor **directly** from the composed layout returned by `make_smem_layout_a` | `smem.allocate_tensor(element_type=…, layout=a_smem_layout.outer, byte_alignment=128, swizzle=a_smem_layout.inner)` |
| `expected input tensor to have the profile (MMA, MMA_MN, MMA_K, …)` | fed `cute.gemm` the (mn, k) smem face, or something you `zipped_divide`d yourself | `tCrA = tiled_mma.make_fragment_A(sA)`, then `tCrA[(None, None, k_block, stage)]` |
| `NVVM backend compilation failed` (a libNVVM paragraph, no reason given) | hand-written `MmaF16BF16Op(...)` with the argument order / kwargs bound wrong, or an operand tensor of the wrong shape that got past the front end | use the example's **positional** order: `(ab_dtype, acc_dtype, inst_mnk, cta_group, OperandSource.SMEM, a_major, b_major)`; copy the whole kernel and get it running before changing anything |
| `operand #0 does not dominate this use … tiled_mma.set(…)` | `tiled_mma.set` sits in one runtime region and `gemm` uses it in another | `set(ACCUMULATE, True)` right after `cute.gemm`, both inside the `range_constexpr` body within `if warp_idx == 0:` |
| `range_constexpr should be preprocessed by preprocessor` | wrote `cutlass.range_constexpr` inside an **undecorated** helper | use it only inside `@cute.kernel` / `@cute.jit` bodies; a helper is either decorated or turns its static loops into plain `range` ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 5) |
| `pack_shape(): incompatible function arguments … Invoked with types: tuple` | a hand-written layout / tiler / coordinate contains elements the DSL does not accept (a misplaced `None`, nested tuples, Python objects) | faces, atoms and fragments all come from helpers (`make_smem_layout_a/b`, `make_tiled_tma_atom_A/B`, `make_fragment_*`, `partition_shape_C`); the tiler is a plain int tuple |
| first launch is fine, **the second launch of the same compiled entry never returns** | the pair form is missing its teardown: producer without `tail()`, TMEM dealloc not going through the pair handshake, or the leader exiting first | `ab_producer.tail()` / `acc_producer.tail()` + `TmemAllocator(is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=…)` + `tmem.free(ptr)`; 3 back-to-back launches of the same compiled entry are certified |
| the pair's full barrier never fills | the load op is not built on `CtaGroup.TWO`, or the pipeline did not get `cta_layout_vmnk`, or `tx_count` is not multiplied by the pair size | `CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)` + `PipelineTmaUmma.create(…, tx_count=bytes × size(vmnk[0]), cta_layout_vmnk=…)`; mechanism in [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md) |
| **hand-written mbarrier pair**: the leader's acc / S full flips, **the follower's barrier of the same name never flips** (the follower's consumer warp hangs, the whole job hits the timeout) | the leader issued a plain `tcgen05.commit(mbar)` — it only arrives on the **issuing CTA's own** mbarrier; nobody arrives on the follower's barrier | `tcgen05.commit(mbar_ptr, mask_self \| mask_peer, CtaGroup.TWO)` (see "Multicast of a hand-written pair commit") — the pipeline objects' `commit()`/`release()` issue exactly this internally |
| `make_tmem_copy` rejects the TMEM readback, or it reads the wrong columns | the subtile column count does not match `Ld32x32bOp(Repetition.xN)` | `Repetition.x64` ⇔ 64 fp32 columns per subtile; `SUBTILE_CNT = accumulator columns // 64` |

**Structural trigger** (look here before running anything): you are about to write your first UMMA on sm100; or you already have TMA
supply and TMEM allocation and are about to hook up the MMA chain; or you plan to use `cta_group::2`. **First run this entry's example
verbatim in your job** (its geometry is parameterised; changing four integers gives your tile), and only after you get `max_diff=0.0`
swap in your own faces, geometry and warp roles one item at a time.

## Minimal runnable example

Below is the complete kernel + host in the **CTA pair form** (the differences for the single-CTA form are listed after the code). A and
B are both K-major (rows, K) bf16, C is (M, N) fp32; one cluster computes one `TILER_MNK[0] × TILER_MNK[1]` output tile.

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

IO_DTYPE, ACC_DTYPE, C_DTYPE = cutlass.BFloat16, cutlass.Float32, cutlass.Float32
INST_MNK = (256, 128, 16)        # pair form M=256 (128 rows per CTA); single-CTA form uses (128, N, 16)
TILER_MNK = (256, 128, 64)       # one k-tile = 64 deep = 4 instructions of K=16
CLUSTER_MNK = (2, 1, 1)
AB_STAGES, ACC_STAGE, THREADS = 2, 1, 128
SUBTILE_CNT = INST_MNK[1] // 64  # Ld32x32b x64: 64 columns per readback subtile

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_dealloc_mbar: cutlass.Int64          # pair form: TMEM release handshake
    tmem_holding_buf: cutlass.Int32

@cute.kernel
def kernel_pair(tiled_mma: cute.TiledMma,
                tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
                tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
                mC_mnl: cute.Tensor,
                a_smem_layout: cute.ComposedLayout, b_smem_layout: cute.ComposedLayout,
                cta_layout_vmnk: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
    mma_coord_vmnk = (bidx % cute.size(cta_layout_vmnk, mode=[0]),
                      bidx // cute.size(cta_layout_vmnk, mode=[0]), bidy, None)
    mma_coord_mnk = mma_coord_vmnk[1:]

    # (1) smem face: split the composed layout at allocation — the affine part is the layout, the swizzle hangs on the pointer
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(element_type=IO_DTYPE, layout=a_smem_layout.outer,
                              byte_alignment=128, swizzle=a_smem_layout.inner)
    sB = smem.allocate_tensor(element_type=IO_DTYPE, layout=b_smem_layout.outer,
                              byte_alignment=128, swizzle=b_smem_layout.inner)
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    # (2) barriers: one stage = bytes of this CTA's two faces × pair size (both CTAs' TMA land on the leader)
    num_tma_copy_bytes = (cute.size_in_bytes(IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2]))
                          + cute.size_in_bytes(IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
                          ) * cute.size(cta_layout_vmnk, mode=[0])
    num_tma_producer = cute.size(cta_layout_vmnk.shape[2]) + cute.size(cta_layout_vmnk.shape[1]) - 1
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer),
        tx_count=num_tma_copy_bytes, barrier_storage=storage.ab_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 cute.size(cta_layout_vmnk, mode=[0]) * THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        cta_layout_vmnk=cta_layout_vmnk).make_participants()

    # (3) partitioning: gmem tile → MMA partition; smem face → MMA fragment (this step yields (MMA, MMA_MN, MMA_K, STAGE))
    gA = cute.local_tile(mA_mkl, TILER_MNK, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, TILER_MNK, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, TILER_MNK, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(mma_coord_vmnk[0])       # rank within the pair: this CTA's half
    tCgA, tCgB, tCgC = thr_mma.partition_A(gA), thr_mma.partition_B(gB), thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)                    # (MMA, MMA_M, MMA_K, STAGE)
    tCrB = tiled_mma.make_fragment_B(sB)                    # (MMA, MMA_N, MMA_K, STAGE)
    tCtAcc = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(TILER_MNK[:2]))  # (MMA, MMA_M, MMA_N)

    tAsA, tAgA = cpasync.tma_partition(tma_atom_a, cta_in_cluster_coord_vmnk[2],
                                       cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
                                       cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
    tBsB, tBgB = cpasync.tma_partition(tma_atom_b, cta_in_cluster_coord_vmnk[1],
                                       cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
                                       cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
    mcast_a = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2)
    mcast_b = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1)

    # (4) TMEM: pair-mode allocation + swap the pointer into the accumulator tensor
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                               barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS),
                               is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
                               two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    # (5) readback plan: (128 lanes × 64 columns) subtiles, Ld32x32b x64 per thread
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // SUBTILE_CNT),)
    tCtAcc_epi, gC_epi = cute.zipped_divide(tCtAcc, epi_tiler), cute.zipped_divide(tCgC, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC, tDgC = tmem_thr_copy.partition_S(tCtAcc_epi), tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, ACC_DTYPE)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, C_DTYPE)

    # (6) mainloop: both CTAs load their own half; only the leader waits on full, issues, and commits
    is_leader_cta = mma_coord_vmnk[0] == 0
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        if is_leader_cta:
            acc_producer.acquire_and_advance()
        for _ in cutlass.range(num_k_tiles, prefetch_stages=AB_STAGES - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier, mcast_mask=mcast_a)
            cute.copy(tma_atom_b, tBgB[(None, ab_empty.count)], tBsB[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier, mcast_mask=mcast_b)
            if is_leader_cta:
                ab_full = ab_consumer.wait_and_advance()
                for k_block_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    k_block_coord = (None, None, k_block_idx, ab_full.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_full.release()                 # = tcgen05.commit → empty[stage]
        if is_leader_cta:
            acc_producer.commit()                 # = tcgen05.commit → acc full

    # (7) epilogue: each CTA reads back the accumulator of its own 128 lanes
    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(C_DTYPE))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # (8) teardown: exit only after draining (a leader that exits early invalidates the peer's DSMEM view)
    if warp_idx == 0:
        ab_producer.tail()
        if is_leader_cta:
            acc_producer.tail()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

@cute.jit
def host_pair(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(IO_DTYPE, ACC_DTYPE, INST_MNK, tcgen05.CtaGroup.TWO,
                              tcgen05.OperandSource.SMEM,
                              tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, TILER_MNK, a.element_type, AB_STAGES)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, TILER_MNK, b.element_type, AB_STAGES)
    a_one, b_one = cute.select(a_smem_layout, mode=[0, 1, 2]), cute.select(b_smem_layout, mode=[0, 1, 2])
    cta_layout_vmnk = cute.tiled_divide(cute.make_layout(CLUSTER_MNK), (tiled_mma.thr_id,))
    op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(op, a, a_one, TILER_MNK, tiled_mma,
                                                                cta_layout_vmnk.shape)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(op, b, b_one, TILER_MNK, tiled_mma,
                                                                cta_layout_vmnk.shape)
    grid_shape = cute.round_up(cute.ceil_div((*c.layout.shape, 1), (TILER_MNK[0] // 2, *TILER_MNK[1:])),
                               CLUSTER_MNK)
    kernel_pair(tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor, c,
                a_smem_layout, b_smem_layout, cta_layout_vmnk
                ).launch(grid=grid_shape, block=[THREADS, 1, 1], cluster=CLUSTER_MNK)

# host side: torch tensor → DSL tensor (K-major, leading_dim=1)
# a_t = from_dlpack(a, assumed_align=32).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=K)
# c_t = ...mark_compact_shape_dynamic(mode=1, divisibility=N)
# compiled = cute.compile(host_pair, a_t, b_t, c_t); compiled(a_t, b_t, c_t)   # repeated launches are certified
```

The **single-CTA form** differs from the above in only six places: `INST_MNK = (128, N, 16)`; `CtaGroup.ONE` (switch both the MMA
op and the TMA op); the TMA op is `cpasync.CopyBulkTensorTileG2SOp(CtaGroup.ONE)` and `cute.copy` gets no `mcast_mask`;
`tma_partition(atom, 0, cute.make_layout(1), …)`; `make_tiled_tma_atom_A/B` without the sixth argument;
`thr_mma = tiled_mma.get_slice(0)`. `PipelineTmaUmma.create` / `PipelineUmmaAsync.create` get no `cta_layout_vmnk`,
`TmemAllocator` gets no `is_two_cta` / dealloc mbar, `SharedStorage` drops `tmem_dealloc_mbar`, the mainloop has no `is_leader_cta`
guard and no `tail()`, `grid_shape = cute.ceil_div((*c.layout.shape, 1), TILER_MNK[:2])`, and `launch` gets no `cluster`.

**Multicast of a hand-written pair commit** (when you do not use `PipelineUmmaAsync` and manage mbarriers yourself). Hardware fact:
`tcgen05.commit` by default only arrives on the mbarrier in the smem of **the CTA that issued it**; in the pair form the MMA is issued
only by the leader, so if the follower's consumer waits on the barrier of the same name **in its own CTA**, the commit must be
**multicast** to both CTAs. The spelling (identical to what `PipelineAsyncUmma` does internally):

```python
# kernel top (both CTAs compute it): mask = self | peer, taken with create_tma_multicast_mask on the V mode (mode 0)
mask_self = cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=0)
peer_coord_vmnk = (cta_in_cluster_coord_vmnk[0] ^ 1, *cta_in_cluster_coord_vmnk[1:])
mask_peer = cpasync.create_tma_multicast_mask(cta_layout_vmnk, peer_coord_vmnk, mcast_mode=0)
commit_mask = mask_self | mask_peer                      # both sides are 0b11 in cluster (2,1,1)

# barrier: one in each CTA's own smem, count = 1; fence after init, then pass a cluster barrier so the peer sees the init
if warp_idx == 0:
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(storage.acc_full_mbar.ptr, 1)
cute.arch.mbarrier_init_fence()
cute.arch.cluster_arrive_relaxed(); cute.arch.cluster_wait()   # every thread in both CTAs must pass it

# after the leader issues the last gemm: one commit that arrives on both CTAs' acc_full_mbar (same smem offset)
if is_leader_cta:
    with cute.arch.elect_one():
        tcgen05.commit(storage.acc_full_mbar.ptr, commit_mask, tcgen05.CtaGroup.TWO)

# the consumers on both sides each wait on their own:
cute.arch.mbarrier_wait(storage.acc_full_mbar.ptr, phase)     # or mbarrier_try_wait for a bounded wait
```

All three are required: `mask` (omitted = only to self), `CtaGroup.TWO` (must match the MMA's cta_group), `elect_one`.

**Geometry readings printed at certification** (use them to check what your own faces look like when copying):

| Form | inst / tiler | A face per CTA (one stage) | B face per CTA (one stage) | `cta_layout_vmnk` |
|---|---|---|---|---|
| single CTA | (128,256,16) / (128,256,64) | `S<3,4,3> o 0 o ((128,16),1,4):((64,1),0,16)` | `((256,16),1,4)` same swizzle | — |
| pair | (256,256,16) / (256,256,64) | `((128,16),1,4)` — of the pair's 256 rows each CTA loads 128 | `((128,16),1,4)` — N=256, each CTA loads half | `((2),1,1,1):((1),0,0,0)` |
| pair | (256,128,16) / (256,128,64) | `((128,16),1,4)` | `((64,16),1,4)` — N=128, each CTA loads 64 | same as above |

In the pair form **each CTA loads only half of A and half of B**, and UMMA reads both sides' smem directly; `tx_count` is therefore
"bytes of this CTA's two half-faces × 2", accounted once on the leader's barrier.

## Gotchas

1. **A composed layout cannot be a tensor layout.** `make_smem_layout_a/b` returns `S<…> o 0 o (…)`; split it at `allocate_tensor` into
   `layout=L.outer, swizzle=L.inner`. Same bytes, same addresses — the swizzle just moves from the layout to the pointer; both the MMA descriptor and TMA accept this form.
2. **Operand fragments must come from `make_fragment_A/B(smem tensor)`.** It regroups the (mn, k, stage) face into `(MMA, MMA_MN, MMA_K, STAGE)`;
   a shape you `zipped_divide` yourself may fail the profile check even with identical extents, and in the pair form only the fragment knows the "each CTA loads only half" semantics.
3. **`MmaF16BF16Op` takes positional arguments**, in the order `(ab_dtype, acc_dtype, inst_mnk, cta_group, a_src, a_major, b_major)`. The decorator erases the signature,
   so binding by name cannot see the parameter list; a mis-bound construction passes the front end and dies in libNVVM.
4. **`set(ACCUMULATE)` and `gemm` in the same region.** The example puts them in the `range_constexpr` body inside `if warp_idx == 0:`;
   changing `tiled_mma` in one region and using it in another is where `does not dominate` comes from. For each new tile (persistent kernels), `set(ACCUMULATE, False)` before its first `gemm`.
5. **`range_constexpr` is only legal inside a decorated body** (`const_expr` deciding on trace-time Python values works as usual in an undecorated helper); a helper is either `@cute.jit` or does only trace-time Python arithmetic with plain `range` for static loops (see [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 1 / gotcha 5).
6. **Pair accounting lives in `cta_layout_vmnk`, not in your hands.** Without that argument `PipelineTmaUmma.create(..., cta_layout_vmnk=…)` silently degrades to
   single-CTA accounting; `tx_count` must be multiplied by `size(vmnk[0])`; the TMA op is built on `CtaGroup.TWO` (the example uses the multicast op; in cluster (2,1,1) the mask contains only self, equivalent to "each loads its own").
   The four requirements and the hang fingerprints are in [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md); who may `wait` is in [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md).
7. **The producer runs in both CTAs, the consumer only in the leader.** Putting a leader guard on the TMA issue = deadlock; removing the guard from `wait_and_advance`/`gemm`/`commit` = asking a phase that never flips.
8. **The three sources of a hang on the second launch** all have their fix in the example: `ab_producer.tail()` (drains the arrives the producer ran ahead with), `acc_producer.tail()`,
   and the pair dealloc handshake of `TmemAllocator(is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=…)` (`pipeline.sync` before `free`).
9. **A hand-written `tcgen05.commit` must be wrapped in `with cute.arch.elect_one():`** (otherwise 32 redundant commits); with the pipeline objects' `release()`/`commit()` you need not care — they already elect.
10. **Readback subtile width = `Repetition`.** `Ld32x32bOp(Repetition.x64)` is 64 fp32 columns per thread, so the subtile column count must be 64;
    `SUBTILE_CNT = columns // 64`. Switching to x128 makes one segment 128 columns.
11. **torch → DSL tensors must be marked with a dynamic layout**: `from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=1).mark_compact_shape_dynamic(mode=1, divisibility=K)`,
    otherwise the TMA descriptor is built on a static shape and every new set of tensors recompiles.

12. **A hand-written commit on a pair must be multicast.** `tcgen05.commit(mbar)` only arrives on the issuing CTA's own barrier; when the follower waits on the barrier of the same name
    in its own CTA you must use `tcgen05.commit(mbar, mask_self | mask_peer, CtaGroup.TWO)` (how to build the mask: "Multicast of a hand-written pair commit" above).
    Certification readings: the multicast form in cluster (2,1,1), `mask=3` on both sides — all 8 CTAs flip, whole-machine bit-exact; the same kernel **without the mask**:
    leader half exact, follower half all zeros, and 4096 rounds of `try_wait` never see the flip (that is the hang); a commit written as `CtaGroup.ONE` while the MMA is `TWO`
    does not even compile (`NVVM backend compilation failed`, no reason given).

## Sources

- Certification probe (a GPU probe, not included in this release) — **Status: ✅ GPU-certified**, final verdict
  `PROBE PASS max_diff=0.0` (passed on the first round of this certification); four contracts all passed in the same round: `U0-lowers` — five variants
  (single CTA (128,256)/(128,128), pair (256,256)/(256,128), pair launched 3 times back to back via `cute.compile`) all compile to launchable kernels;
  `U1-1cta-exact` 0.0 / `U2-pair-exact` 0.0 — A, B are integer bf16 in [-2, 2), C is fp32, **bit-equal** to the fp32 `einsum` reference
  (not within tolerance); `U3-relaunch` 0.0 — three launches of the same compiled entry all return and are all exact. Each variant runs in its own subprocess with its own 240 s budget;
  the subprocess uses `sys.executable` (parent and child share one interpreter).
- Certification probe (multicast of a hand-written pair commit) — **Status: ✅ GPU-certified**, final verdict `PROBE PASS max_diff=0.0` (round 3 of this certification; round 1 was a misuse of `cute.printf` in the probe itself;
  round 2 already passed the positive case and only had a flag index missing grid.y); three contracts all passed in the same round: `M0-lowers`; `M1-pair-mcast-commit-exact` 0.0 —
  `tcgen05.commit(mbar, mask_self | mask_peer, CtaGroup.TWO)`, the hand-written mbarriers in both CTAs' own smem flip, C bit-equal to the fp32 reference;
  `M2-plain-commit-starves-peer` — the same kernel with only the mask removed: leader half exact and flipped, follower half all zeros, 4096 rounds of bounded `try_wait` never see the flip
  (negative contract: this path is exactly "the follower's barrier never flips"). Fact line: a `CtaGroup.ONE` commit (with or without mask) fails to compile in the libNVVM backend.
- Public API: `cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op` / `CtaGroup` / `OperandSource` / `OperandMajorMode` / `Field` /
  `Ld32x32bOp` / `Repetition` / `make_tmem_copy`; `cutlass.utils.blackwell_helpers.make_smem_layout_a` / `make_smem_layout_b`;
  `cutlass.cute.nvgpu.make_tiled_tma_atom_A` / `make_tiled_tma_atom_B`; `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp` /
  `CopyBulkTensorTileG2SMulticastOp` / `tma_partition` / `create_tma_multicast_mask` / `prefetch_descriptor`; `cutlass.cute.nvgpu.tcgen05.commit` (`mask` / `cta_group`);
  `cute.arch.mbarrier_init` / `mbarrier_init_fence` / `mbarrier_try_wait` / `cluster_arrive_relaxed` / `cluster_wait` / `elect_one`;
  `cutlass.pipeline.PipelineTmaUmma` / `PipelineUmmaAsync` / `CooperativeGroup` / `NamedBarrier` / `sync`;
  `cutlass.utils.TmemAllocator` (`allocate` / `wait_for_alloc` / `retrieve_ptr` / `relinquish_alloc_permit` / `free`);
  `cute.TiledMma.make_fragment_A/B/C` / `partition_shape_C` / `get_slice` / `set`; `cute.gemm`; `cute.arch.block_idx_in_cluster`.
- Official docs: CUTLASS CuTe DSL documentation https://docs.nvidia.com/cutlass/latest/ (the Blackwell tutorial GEMM series; this entry's example condenses its
  single-CTA and 2-CTA tutorials into one tile); PTX ISA chapters on `tcgen05.mma` / `tcgen05.commit` / `tcgen05.alloc`.
