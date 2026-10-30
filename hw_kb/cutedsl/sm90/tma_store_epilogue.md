# tma_store_epilogue (CuTeDSL: WGMMA C-frag → stmatrix R2S → proxy fence → TMA store)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers the minimal interface set of
> the SM90 epilogue store path: `stmatrix`-family matrix stores (registers → swizzled epi SMEM) +
> `fence_proxy(async_shared)` + NamedBarrier ordering + `tma_store` (bulk async S2G) + `PipelineTmaStore` throttling.
> As with the main-loop trio ([`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)): on Hopper the four form one data path;
> written separately they could not give a runnable example, so they are merged into one entry. The same kind of
> mechanism at the CUDA/PTX level is [`cfrag_epilogue_stmatrix`](cfrag_epilogue_stmatrix.md) (this entry is its
> CuTeDSL counterpart).

## Purpose

Write the WGMMA FP32 accumulator fragment (spread over the registers of 1–2 warpgroups) back to gmem at high
throughput: reduce precision in registers → write into a swizzled epi SMEM ring buffer in one go with a
stmatrix-family copy atom (removing `op_st` bank conflicts) → generic→async proxy fence + an all-hands NamedBarrier →
one warp issues a bulk TMA S2G (writes merged into whole-tile transactions) → `PipelineTmaStore` throttles the ring
reuse. This is the standard foundation path of "the output end of a tensor-core kernel" on SM90.

## When needed (trigger)

- The main loop already accumulates into registers with WGMMA, and the C tile must be written back to gmem, while
  per-coordinate scalar stores straight into gmem have become the tail bottleneck (NCU: fragmented `l1tex` write
  transactions, or a high epilogue share).
- NCU reports high `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` (when writing SMEM staging through a
  non-stmatrix path).
- Symptom: the probe/kernel keeps getting stuck on `make_tiled_copy_S` / `retile` / fence order /
  a hung TMA store (single-lane issue).
- Dataflow: register C-frag → multi-stage SMEM ring buffer → gmem, with the write side (stmatrix, generic proxy) and
  the read side (TMA, async proxy) in different proxies, requiring an explicit fence + barrier.
- **Or**: this path already works and the numbers are right, but ncu's
  `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum` is non-zero with symmetric loads and stores — go to §4; the
  staging fragment has most likely been copied as a full-size copy of the accumulator.

## Minimal runnable example

Single CTA, 2 MMA warpgroups (256 threads, no DMA warpgroup, no big branch), C tile 128×128 n-major fp16, epi subtile
(128,32)×4, epi ring of 2 stages. The main loop is stripped out: acc is faked in registers with static identity
coordinates (the correct fix of gotcha (d) of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md)).

### 1) Host side: TiledMma, epi SMEM layout, S2G TMA atom

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup

CTA_M, CTA_N = 128, 128
EPI_M, EPI_N = 128, 32     # cooperative epi_tile = (min(128,M), min(32,N))
EPI_STAGE = 2              # ring depth (2/4 both fine; the mechanism does not depend on depth)
THREADS = 256              # 2 MMA warpgroups

# --- TiledMma: the same object as the main loop (it defines the epilogue's C-frag layout) ---
# cooperative form: atom_layout=(2,1,1) stacks 2 atoms along M, tiler_mn=(64, CTA_N).
tiled_mma = sm90_utils.make_trivial_tiled_mma(
    cutlass.Float16, cutlass.Float16,
    cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
    cutlass.Float32, (2, 1, 1), tiler_mn=(64, CTA_N),
)

# --- epi SMEM layout: the atom picks the swizzle level from C's layout/dtype/major length ---
# n-major C: major-mode length = EPI_N; staged to (EPI_M, EPI_N, EPI_STAGE).
c_layout_enum = cutlass.utils.LayoutEnum.from_tensor(c)     # c: cute.Tensor (M,N) fp16
epi_atom = warpgroup.make_smem_layout_atom(
    sm90_utils.get_smem_layout_atom(c_layout_enum, cutlass.Float16, EPI_N),
    cutlass.Float16)
epi_smem_layout_staged = cute.tile_to_shape(
    epi_atom, (EPI_M, EPI_N, EPI_STAGE), order=(0, 1, 2))   # n-major: order=(0,1,2)

# --- S2G TMA atom: single-stage layout + epi tile shape (same as gotcha 2 on the load side) ---
epi_layout_1stage = cute.slice_(epi_smem_layout_staged, (None, None, 0))
tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(),      # S2G has no multicast, do not pass num_multicast
    c, epi_layout_1stage, (EPI_M, EPI_N),
)
# the kernel must use tma_tensor_c (the coordinate tensor), not the original c (same as gotcha 3 on the load side).

# --- NamedBarrier for all epilogue threads (constructed on the host, used in the kernel via self) ---
epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
```

### 2) SharedStorage: only the epi buffer; PipelineTmaStore has no mbarrier storage

```python
@cute.struct
class SharedStorage:
    sC: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float16,
                             cute.cosize(epi_smem_layout_staged)],
        1024]
# Note: PipelineTmaStore only wraps bulk-group commit/wait, NOT mbarriers -- it takes no SMEM.
```

### 3) Kernel side: the r2s copy mechanism, the fixed eight-step order, the TMA store throttle ring

```python
@cute.kernel
def kernel(tma_atom_c, mC, tiled_mma, epi_smem_layout_staged):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_c)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sC = storage.sC.get_tensor(epi_smem_layout_staged.outer,
                               swizzle=epi_smem_layout_staged.inner)

    # --- TMA store throttle pipeline: a thin wrapper of bulk-group commit/wait ---
    tma_store_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=EPI_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
    )

    # --- gmem side zipped-divided by epi tile and fed straight to tma_partition (no more group_modes) ---
    gC_epi = cute.zipped_divide(mC, (EPI_M, EPI_N))   # ((EPI_M,EPI_N), (RestM,RestN))
    bSG_sD, bSG_gD = cpasync.tma_partition(
        tma_atom_c, 0, cute.make_layout(1),
        cute.group_modes(sC, 0, 2),    # the smem side still groups the first two modes
        gC_epi,                        # the gmem side is already in zipped form
    )
    # hierarchical gmem coordinates of the subtiles: row-major rest layout + get_hier_coord
    epi_tile_shape = gC_epi.shape[1]
    epi_tile_layout = cute.make_layout(epi_tile_shape,
                                       stride=(epi_tile_shape[1], 1))
    num_epi = cute.size(gC_epi, mode=[1])

    # acc: the C-frag produced by the main loop (here static per-thread (m,n) coordinates show where its shape comes from)
    thr_mma = tiled_mma.get_slice(tidx)
    tCcC = thr_mma.partition_C(cute.make_identity_tensor((CTA_M, CTA_N)))
    acc = cute.make_rmem_tensor(tCcC.shape, cutlass.Float32)
    # ... acc filled by the WGMMA main loop (see the wgmma_tma_mbarrier entry) ...

    # --- r2s copy mechanism: three-layer construction, fixed order ---
    # (1) pick the SMEM store op by C layout/dtype (fp16 output -> stmatrix family)
    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
        c_layout_enum, elem_ty_d=cutlass.Float16, elem_ty_acc=cutlass.Float32)
    # (2) stmatrix.x4 atom + tiled_mma -> value grouping of the C-frag
    copy_atom_C = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(c_layout_enum.is_m_major_c(), 4),
        cutlass.Float16)
    tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
    # (3) store op as the atom, expanded into a tiled copy by the C-atom's tiling
    tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)

    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)   # consumer-internal thread index (see gotcha 4)
    tRS_sD = thr_copy_r2s.partition_D(sC)           # (R2S,R2S_M,R2S_N,PIPE_D)
    tRS_rAcc = tiled_copy_r2s.retile(acc)           # acc reordered into copy-value order

    # staging registers per subtile (one block at acc precision + one at output precision)
    rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
    tRS_rD_layout = cute.make_layout(rD_shape[:3])
    tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.Float32)
    tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, cutlass.Float16)
    size_tRS_rD = cute.size(tRS_rD)

    # --- epilogue main loop: fixed eight-step order (see gotcha 5; for where the throttle goes see gotcha 13) ---
    for epi_idx in cutlass.range_constexpr(num_epi):
        # 1. take this subtile's value segment from the retiled acc
        for epi_v in cutlass.range_constexpr(size_tRS_rD):
            tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
        # 2. reduce precision in registers (whole-segment load->convert->store)
        tRS_rD_out.store(tRS_rD.load().to(cutlass.Float16))
        # 3. throttle: wait for the previous round's store to finish reading this slot. Place it right next to the overwrite it protects,
        #    not right next to the store it drains (gotcha 13). Issue it unconditionally -- in the first round there is no in-flight
        #    bulk-group and wait_group.read returns immediately; no epi_idx>0 special case needed.
        if warp_idx == 0:
            tma_store_pipeline.producer_acquire()   # wait_group(EPI_STAGE-1)
        # 4. rendezvous: the throttle has confirmed; only now may everyone overwrite this ring slot
        epilog_sync_barrier.arrive_and_wait()
        # 5. stmatrix writes this stage (generic proxy)
        epi_buffer = epi_idx % EPI_STAGE
        cute.copy(tiled_copy_r2s, tRS_rD_out,
                  tRS_sD[(None, None, None, epi_buffer)])
        # 6. generic->async proxy fence: only then can TMA see the stmatrix writes
        cute.arch.fence_proxy("async.shared", space="cta")
        # 7. all-hands rendezvous: TMA may read only after every writer has fenced
        epilog_sync_barrier.arrive_and_wait()
        # 8. a single warp (a whole warp! see gotcha 6) issues the TMA store + commit, then goes straight to the back edge --
        #    the next round's steps 1-2 (and, in a persistent kernel, the next tile's main loop)
        #    run in the shadow of this store draining
        gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
        if warp_idx == 0:
            cute.copy(tma_atom_c, bSG_sD[(None, epi_buffer)],
                      bSG_gD[(None, gmem_coord)])
            tma_store_pipeline.producer_commit()

    tma_store_pipeline.producer_tail()   # called by everyone: drain all stores before the kernel exits
```

> There are still 2 NamedBarriers per round (steps 4 and 7), and `producer_acquire` is still once per subtile — **the
> order changed, the counts did not**.

launch: `kernel(...).launch(grid=[...], block=[THREADS,1,1], cluster=[1,1,1], stream=...)`;
when `SharedStorage` goes through `cute.struct` there is no need to compute the `smem=` byte count by hand.

### 4) The register-cost invariant of the staging fragment (the half `max_diff` cannot see) 🟡 certification in progress

The two staging fragments of §3 (`tRS_rD` / `tRS_rD_out`) are sized **per epi subtile**: their shape comes from
`partition_S(sC)`, and `sC` is the ring buffer of the **epi subblock**, so each round of `for epi_idx in
range(num_epi)` handles only `size(acc)/num_epi` elements. This is not a stylistic preference; it is the **cost
premise** of this path:

```
size(tRS_rD) * num_epi == size(acc)        # elements per thread, an identity
```

**The degenerate case (the most easily miscopied part of this entry)**: when the destination SMEM tile is the whole
MMA tile (`num_epi == 1` — typically when this SMEM is not an epilogue ring but **the operand buffer of the next
MMA**), the same three allocation lines produce "a **full-size f32 copy** of the accumulator + a **full-size f16
copy**", i.e. `E + E/2` registers per thread (`E` = accumulator elements per thread; a 128×128 tile / 256 threads ⇒
`E = 64` ⇒ 96 registers). The numbers are as correct as ever, and `max_diff` is 0 as ever — **the certification probe
of §3 is structurally blind to it** (it only checks numbers). That is why this section exists.

#### Invariants (checkable against the code item by item)

1. **Staging must be subtile-sized**: `size(staging) * num_epi == size(acc)`.
   If the equation does not hold, the subtile form has been copied onto the whole block.
2. **Declare it inside the consuming segment**: staging is a **temporary**, not state. Placing it outside the pair /
   main loop declares to ptxas "this block must reappear every round", and the bill is charged at its allocated size
   (for the accounting see mechanism 3 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)).
3. **When the output-precision values are already in hand, no staging at all**: if the upstream elementwise segment
   has already packed an f16 fragment for some RS WGMMA, that fragment is a legal source for the copy — `retile` it
   directly (the spelling in the next subsection), instead of going through another f32 staging + a whole-block
   `.store(.load().to(...))`.
4. **Acceptance looks at spill, not `max_diff`**: the criterion of this invariant is ncu's
   `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum` **== 0**, not the numbers. The numbers of the two forms are
   **bit-for-bit identical** (both numerical lines of this section's probe are `0.0`), so any probe that only checks
   `max_diff` is structurally blind to it.

#### An f16 C-frag already in hand: the exact spelling for retiling it directly

```python
# product of the upstream elementwise segment: the A operand (f16) the next RS WGMMA will read, which must stay live anyway
rP = cute.make_fragment_like(acc, cutlass.Float16)
for i in cutlass.range_constexpr(cute.size(acc)):
    rP[i] = cutlass.Float16(acc[i])
...
tRS_sD = thr_copy_r2s.partition_D(sC)              # destination: partition the [staged] tensor as before
cute.copy(tiled_copy_r2s, tiled_copy_r2s.retile(rP),
          tRS_sD[(None, None, None, stage)])       # source: retile the existing f16 directly
```

Not a word changes on the destination side (first `partition_D` the staged tensor, then slice by the stage index —
gotchas 2 / 3); only the **source** changes from "gather into staging → whole-block cvt" to
`retile(<existing f16 fragment>)`. `retile` only switches to another value-order view and produces no movement
instruction, so this path's staging cost is **0 registers**.

> **Why this spelling deserves its own certification**: an earlier workspace probe used it and
> failed, but that round also wrote the **destination** as a rank-2 `cute.slice_` of the staged layout (rather than
> partitioning and then slicing the stage), so the two variables were not separated; the next probe round
> switched the destination back to the §3 form and in passing switched the source back to f32 staging too. So the
> question "can the source be `retile(f16)`" stayed unanswered, and students could only copy the version with staging.
> This section's probe keeps only one variable (destination fixed to the certified form, only the source switched).

#### Machine-readable criterion (this section's certification probe)

The certification probe (not included in this release): the same kernel and the same payload, differing only in the copy's
source (DIRECT = `retile(rP)`; STG = §3's f32 staging), a 128×128 f16 single-stage SMEM tile, 256 threads, 96
simultaneously live int32 of ballast per thread (built per gotcha 10 of [`warp_specialization`](warp_specialization.md):
single seed, forward serial chain production, reverse-order XOR-multiply chain consumption, carried in a Python list
rather than `cute.make_rmem_tensor`). Six contracts:

- each form on its own `max_diff == 0` (the bits written into `sP` are exactly the same);
- each form's ballast checksum equal bit for bit to the Python reference (**the payload is real**, not an experiment
  emptied by optimisation);
- `spill(DIRECT) == 0`;
- **`cost(STG)` strictly worse**: more spill, or (when both fit) a higher static `REG`.
  The criterion is written as "strictly worse" rather than "STG must spill" because whether the cost lands in spill or
  in static allocation depends on how close the payload is to the budget, while **whether the cost exists** does not.


## Gotchas

1. **The S2G atom builder only takes a single-stage smem layout** (same as gotcha 2 of
   [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) on the load side), and the kernel must use the returned coordinate
   tensor `tma_tensor_c` (same as gotcha 3). The S2G `make_tiled_tma_atom` **does not take** `num_multicast`.
2. **The two sides of `tma_partition` have asymmetric forms**: the smem side needs `group_modes(sC, 0, 2)`; the gmem
   side takes the result of `zipped_divide(gC_tile, epi_tile)` — it is already in zipped form and is **not** grouped
   again. The gmem index of a subtile is not the linear `epi_idx` but the hierarchical coordinate
   `make_layout(rest_shape, stride=(rest_shape[1],1)).get_hier_coord(epi_idx)`; using `epi_idx` directly gives a rank
   mismatch or writes the wrong block.
3. **The three-layer r2s construction has a fixed order**: `sm90_get_smem_store_op(c_layout, d_ty, acc_ty)` (picks
   stmatrix or a degraded st automatically by output dtype/layout) → `make_copy_atom(StMatrix8x8x16bOp(
   is_m_major, 4), c_dtype)` + `make_tiled_copy_C_atom(·, tiled_mma)` (hangs the value grouping on the MMA's C-frag
   layout) → `make_tiled_copy_S(store_op, C_atom_tiling)`. The write side and the TMA read side need no hand-matched
   bijection — `partition_D(sC)` and the TMA's smem layout share the same `epi_smem_layout_staged`, and consistency is
   guaranteed by the layout object itself. **Do not** try to bypass this mechanism by hand-writing per-element scalar
   stores into swizzled SMEM and then assembling TMA (GPU evidence: the workspace d31 probe took that detour and failed
   certification for 12 rounds).
4. **The thread index of `get_slice` is the consumer-internal index**: with a DMA warpgroup it must be
   `tidx - num_dma_threads`; `tiled_mma.get_slice` likewise numbers by the set of threads it covers. Using the global
   tidx shifts everything (silent numerical error).
5. **The proxy order cannot be swapped (INV9 class)**: stmatrix is the generic proxy, TMA the async proxy.
   `copy(r2s)` → `fence_proxy(async_shared, shared_cta)` → NamedBarrier →
   TMA store + `producer_commit` + `producer_acquire` → **a second** NamedBarrier.
   Missing the fence = TMA reads stale data; missing the first barrier = only the store warp's writes are visible;
   missing the second barrier = the next round's stmatrix overwrites a stage the in-flight TMA is still reading (silent
   numerical error).
6. **The TMA store must be issued by a whole warp**. CuTeDSL's TMA copy atom does a warp-level elect internally
   (membermask = whole warp); issuing from a single lane with `if tidx == 0:` is UB → the GPU hangs and
   `cuda.synchronize` never returns (GPU evidence: workspace d31 probe round 3).
   Correct form `if warp_idx == K:`; the commit/wait of non-issuing lanes hold empty bulk-groups and return immediately.
7. **`PipelineTmaStore` has no mbarrier storage**: `create(num_stages, producer_group)` does not take
   `barrier_storage` and takes no SharedStorage. `producer_acquire()` is the
   `cp.async.bulk.wait_group(num_stages-1)` throttle (≤ N-1 groups in flight); the final `producer_tail()` is called by
   **all** threads to drain.
8. **A NamedBarrier's `barrier_id` starts at 1** (0 is reserved for `sync_threads`); `num_threads` = all threads taking
   part in the epilogue, and both `arrive_and_wait` calls must be executed **without divergence** by every
   participating thread (inside a branch only some threads take = hang).
9. **The DSL trap of a large else-less `if` branch** (GPU evidence: workspace d31 probe round 10, the DSL error landing
   on the if header, the body's first statement never executed): an else-less if that **binds for the first time** a
   large number of DSL objects (smem tensors, partitions, fragments) inside the branch body is rejected by the region
   liveness analysis of the if-executor. Safe forms: the producer-first `if/else` used by the certification probe (both
   branches non-empty), or — what this entry's probe does — not putting the epilogue body into a big branch at all (all
   threads take part throughout, with only the TMA issue inside a small warp-if).
10. **File level: no `from __future__ import annotations`** (PEP 563 turns `@cute.struct`/`@cute.kernel` annotations
    into strings, and `cute.compile` raises a TypeError; same as gotcha 13 of
    [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)).
11. **The staging of §3 is a subtile-sized temporary, not "a block this path must have"** (see §4).
    Copying it onto a whole tile with `num_epi == 1` and hoisting it outside the main loop gives an accumulator-sized
    f32 copy + an f16 copy, both resident. **The symptom is spill, not wrong values**: `max_diff` is still 0, and ncu's
    local ld/st counts rise symmetrically (the signature of a whole fragment round-tripping once per pair).
12. **`max_diff` is structurally blind to this entry's cost side**. The certification probe of §3 only checks numbers,
    so "copied §3 and the probe passed" does not imply "the copied form fits at your tile size". Before going into a
    production kernel, check §4's equation `size(staging) * num_epi == size(acc)` and include the spill count in
    acceptance.
13. **Place the throttle right next to the overwrite it protects, not right next to the store it drains.**
    `producer_acquire()` (= `cp.async.bulk.wait_group.read num_stages-1`) protects **the next overwrite of that ring
    slot** — stmatrix must not write into this smem while TMA is still reading it. But it naturally gets written after
    the store ("issue, then wait"), so **all the work sandwiched in between is dragged into waiting with it**: in §3 of
    this entry that is the next round's steps 1-2 (fetching values + reducing precision), in a persistent kernel it is
    **the next tile's whole main loop**. Moving the acquire to just before the next round's overwrite (step 3 of §3)
    leaves the protection relation unchanged word for word, while the work in between runs in the shadow of the store
    draining.
    - **Issue unconditionally; do not write `if epi_idx > 0`**: in the first round there is no in-flight bulk-group and
      `wait_group.read` returns immediately; tile boundaries are thereby covered automatically too (the first round
      waits on the previous tile's last store), and a special case would actually open a vacuum at the tile boundary.
    - **Counts unchanged, only the order changes**: `producer_acquire` is still once per subtile, the NamedBarrier still
      twice per round. Machine-readable self-check = the number of `cp.async.bulk.wait_group` in the PTX is the same
      under both placements; if the count dropped, the throttle was deleted rather than moved, which is wrong (see C4 of
      the certification probe below).
    - **`producer_tail()` as before**: it is the closing drain before the kernel exits, orthogonal to this gotcha, and
      must be kept under any placement (in the PTX census it is the one extra instruction).
    - **Counter-trigger**: the deeper the ring, the longer the slot reuse interval, the less the wait actually happens
      anyway, and the gain of the placement narrows accordingly; `num_stages=1` (single buffer) is the tier with the
      largest gain. This is a **lever**, not a **default** — changing the placement requires a same-job before/after
      comparison.
14. **The rest-mode structures of `retile(source)` and `partition_D(destination)` do not necessarily match, and
    `cute.copy` compares mode by mode**. Both sides carry the same total number of elements per thread (the same N atom
    applications), but split differently: `tiled_copy.retile(rSrc)` keeps the **source fragment's** mode structure,
    `thr_copy.partition_D(sDst)` keeps the **destination smem tile's** mode structure. When they disagree, trace time
    fails hard:

        ValueError: Expected source and destination tensors to have the same size
                    in mode-1, but got 1 and 8

    In the construction of §3 the two sides happen to match (the source is a staging fragment built from the shape of
    `partition_S(sC)`), so copying §3 never hits this; **you hit it when the source becomes an existing fragment whose
    shape is decided elsewhere** (e.g. the branch of §"An f16 C-frag already in hand: retile it directly"). The fix is
    to rebuild the source's iterator as a tensor with **the destination's shape** — a pure view, zero movement, zero
    staging registers:

    ```python
    dst = thr_copy_r2s.partition_D(sDst)
    src = cute.make_tensor(rSrc.iterator, cute.make_layout(dst.shape))
    cute.copy(tiled_copy_r2s, src, dst)          # both sides the same shape, the per-mode comparison passes
    ```

    **Refutation record (same-round GPU sweep, do not try again)**: folding the rest modes of each side into one mode
    with `cute.group_modes(t, 1, rank)` does **not** fix it — it still raises the same mode-1 rejection.
    Two other candidates (gathering into a staging fragment of the destination's shape, pairing by static (m, d)
    coordinates) work but each costs **+32 registers**; once the zero-cost view above works there is no reason to pick
    them.
    ⚠️ The view reinterpretation bypasses this layer of mode-structure checking, so it **must be separately proven to be
    a bijection** — two marginal ramps (a row ramp + a column ramp, using small integers exact in bf16, with the strict
    criterion `max_diff == 0.0`), see §4 of [`gpu_probe_ladder`](gpu_probe_ladder.md).

15. **The smem layout atom of the staging face has a price: dropping the swizzle removes the prologue's local round
    trip, but the whole kernel gets slower.**
    This face is written by `stmatrix` and read by the TMA store, and **no wgmma descriptor reads it**, so the swizzle
    is not part of an operand contract, and the atom can be switched between `warpgroup.SmemLayoutAtomKind.K_SW128` and
    `K_INTER` (the landing layout, the `stmatrix` destination view and the TMA-store box must all come from the same
    atom).
    - **What `K_INTER` buys**: on a swizzled face each thread's address per store is `base XOR f(op)` — a set of
      loop-invariant registers computed once and resident across the whole persistent loop; under `K_INTER` it is
      `base + compile-time offset`, folded into the `STS` immediate. Static census 5 `LDL` + 5 `STL` / 24 B frame → 0 + 0
      / 0 B, numbers bit-for-bit unchanged.
    - **What it costs**: two adjacent evaluations differing only in this atom — `K_INTER` → `K_SW128`: `bankconf_st`
      94010 → 66250, `tensor_pct` 70.98 → 76.33, `stall_barrier` 1.62 → 1.33, while
      `spill_ld / spill_st` 0 / 0 → 163840 / 21120 sectors; gate 0.23543 → 0.21981 ms (−6.63%).
      That is: the once-per-tile local round trip came back, and the time went down.
    - **Load-bearing condition and rule**: in a structure with a persistent tile loop, a store once per tile, and
      `LDL/STL` = 0 in the steady-state body to begin with, the swizzle is worth about 6–7%. `spill == 0` is not the
      objective function: first use line-number attribution to confirm whether the spill lands in the prologue /
      epilogue or in the steady-state body; the few that land in a once-per-tile boundary segment are not worth trading
      write-side bank conflicts for.
      The dossier's `rings` lost two families at this atom switch, which is moving along
      (gotcha 16 of [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)).

## Sources

- **The throttle placement of §3 (gotcha 13) — GPU-certified** by a throttle-placement probe
  (132 CTAs × 256 threads, C tile 128×128 n-major fp16, epi
  subtile (128,32)×4, **ring depth 1**, 8 tiles per CTA; acc faked from static identity coordinates folded with the tile
  number, so every subtile's payload differs and a store landing in the wrong window shows up as wrong values). The
  three variants **differ only in the throttle position**: `after` (the placement before the move) / `deferred` (§3's
  current placement) / `none` (**negative control**: throttle deleted entirely, all barriers kept).
  All five contracts passed in the same round:
  - C1/C2 `after` and `deferred` each **bit-for-bit equal** to the host reference (`max_diff=0`, zero errors over
    17301504 elements) ⇒ moving the position does not break the protection relation;
  - C3 `none` must be **wrong** — measured `max_diff=2023`, 11475192 elements wrong ⇒ in this shape the hazard really
    does occur, so C2 is "tested protection" rather than "protection that never fired".
    **Without this item, C2 is empty**;
  - C4 PTX `cp.async.bulk.wait_group` counts `after=5`, `deferred=5`, `none=1`, i.e. the two protected variants are
    equal and each has `num_epi=4` more than the control, and the control's remaining 1 is exactly the
    `producer_tail()` every variant keeps ⇒ the throttle was **moved**, not **deleted** (otherwise C5 could get a free
    win by deleting the wait);
  - C5 same-job directionality: over 31 timed launches, `deferred` is **faster on both order statistics, median and
    minimum** (med 0.0694 vs 0.0737 ms, min 0.0692 vs 0.0732 ms). The quantified gain is anchored in the operator-side
    same-job before/after comparison, not in this micro-benchmark.
  Two independent re-runs agreed (C5 median ratio 1.0627 / 1.0648, C3 wrong-element count 11475192 / 11426912).
- **DSL version**: the spellings of §1/§3 are calibrated against **CUTLASS-DSL 4.6.1** (the probe above actually ran on
  that version). The earlier spelling **no longer runs** on that version; three hard drifts were fixed together:
  `cute.make_fragment` → `cute.make_rmem_tensor`;
  `cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta)`
  → `cute.arch.fence_proxy("async.shared", space="cta")` (the two symbols `ProxyKind` / `SharedSpace` no longer exist;
  `kind`/`space` now take string literals);
  `warpgroup.OperandMajorMode` → `cute.nvgpu.OperandMajorMode` (the former still exists but is deprecated).
  Before copying this entry, check the installed version with the API-drift measurement method of
  [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) — **"✅ Verified" is said of one version; it is not permanent**.
- **"Host side" of §1 = the `@cute.jit` entry, not module scope**: `make_smem_layout_atom` /
  `tile_to_shape` / `make_trivial_tiled_mma` / `make_tiled_tma_atom` all go through `make_swizzle`, which parses MLIR
  types and needs a live Context. Called from ordinary Python they raise
  `An MLIR function requires a Context but none was provided`. Likewise the `SharedStorage` of §2 must be built inside
  `@cute.jit` through an **ordinary** factory function (`cute.cosize` is a traced query); writing `global` directly in
  the traced function body hits `UNSUP_GLOBAL` — pass it through a module-level mutable container instead.
- GPU-certified by the main certification probe (single CTA, 256 threads,
  128×128 fp16 C, 4 subtiles × 2-stage ring, static identity acc vs an exact torch reference).
- Official reference: cutlass examples `examples/python/CuTeDSL/hopper/dense_gemm_persistent.py`
  (same API surface).
- Failure side evidence (the empirical source of gotchas 3/6/9): the GPU failure fingerprints of rounds 3/10 of an
  earlier workspace epilogue probe.
- **§4 (the register cost of staging)**:
  - The invariant itself is not a new mechanism; it writes down the **premise** of the construction §3 already has:
    the staging's shape comes from `partition_S(sC)`, and `sC` is the **epi subblock** buffer — i.e. the staging size is
    defined by the epi tile, **not by the accumulator**. With `num_epi == 1` the two are equal, and this path
    degenerates into "copying the accumulator".
  - Accounting convention (live ranges rather than an allocation list, `W` usually the largest term):
    mechanism 3 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md);
    ballast construction (how to build a payload with a lower bound on liveness that ptxas cannot dissolve):
    gotcha 10 of [`warp_specialization`](warp_specialization.md).
  - The criterion uses ncu's local ld/st sectors + `launch__registers_per_thread`, not the disassembly channel.
  - GPU certification: the staging-cost probe — 🟡 certification in progress.
- **Gotcha 14**: GPU-certified by an epilogue probe — four candidate source spellings swept in
  the same round; only the zero-cost view reinterpretation both lowers and is the identity on (packed row, column),
  `PROBE PASS max_diff=0.0  (bijection of the shipped r2s spelling
  (C-frag -> F2FP -> STSM -> swizzled smem -> TMA store -> gmem))`; the same construction was double-checked on both
  edges by an epilogue ping-pong probe with
  `PROBE PASS max_diff=0  # S1-r2s-row-ramp` / `# S2-r2s-col-ramp`.
- **Gotcha 15 (the price of the staging face's layout atom)**: dossier census and NCU of the KEEP evaluation (eval job
  number not entered in the status record), axis `rings` `["stride=0x1", "stride=0x1ff,wrap=sel", "stride=0x2"]` →
  `["stride=0x1"]` (moving along), gate 0.23543 → 0.21981 ms; static census probe
  (each row in its own subprocess + its own cache, `K_INTER` row 0 + 0
  / 0 B, control row 5 + 5 / 24 B, numbers bit-for-bit identical).
  ⚠️ **The probe's PASS evidence comes from an earlier iteration**: `PROBE PASS (max_diff=0.0) — 4/4 PASS` came two
  iterations before the KEEP, and the probe file was not changed after that and was carried verbatim into this KEEP's
  tree (the archived file is byte-identical to the version on that tree); i.e. "the same probe holds on this tree" is
  given by file identity, not by re-running it on this tree.
  Status: ✅ KEEP-certified.
