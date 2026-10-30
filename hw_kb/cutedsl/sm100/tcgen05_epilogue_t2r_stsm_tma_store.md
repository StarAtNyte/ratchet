# tcgen05_epilogue_t2r_stsm_tma_store (the full assembly for getting a UMMA accumulator out of TMEM: TMEM → registers → convert → smem staging face → TMA store)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **finishing-side** counterpart of [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md): that entry reads the accumulator
> back into registers with `Ld32x32b` and then `autovec_copy`s it straight into gmem — enough to certify the main loop, not enough for a real epilogue.
> This entry covers only **the four-leg path from TMEM to gmem**: which helper to call, what profile each leg's tensors must have, and where the fence and barrier between the two copies go:
> `choice of the epi tile → tmem_load atom and t2r partitioning → smem store atom and r2s
> partitioning (`retile`) → staging-face layout and TMA store partitioning → async-group accounting`. The main-loop skeleton is identical to that entry and is not repeated.
>
> Likewise it is not a "reference" but **a path that has passed the verifier**: the atom of every one of the four legs is picked by a DSL helper
> from (CTA tile, output majorness, element width), and **both sides of a partitioning always start from the same tiled copy** —
> five common errors are indexed by error text in the "When needed" section, each corresponding to one line of the example.

## Purpose

Blackwell's accumulator lives in TMEM (128 datapaths × 32-bit columns); getting it out of TMEM involves four mutually coupled shape constraints:

| Stage | Hardware fact | Corresponding DSL form |
|---|---|---|
| Sub-tile | `tcgen05.ld` reads 16 or 32 datapaths × some columns at a time; 4 epilogue warps split M | `sm100_utils.compute_epilogue_tile_shape(cta_tile_mnk, use_2cta, layout_c, c_dtype)` gives a by-mode tiler; **do not pick it by hand** (a hand-picked tile cannot determine the atom; the atom is derived from the tile) |
| TMEM → registers | `tcgen05.ld.32x32b` / `16x256b` / `16x128b` …; the set of columns each thread gets is fixed by the atom | `get_tmem_load_op(cta_tile_mnk, layout_c, c_dtype, acc_dtype, epi_tile, use_2cta)` → `tcgen05.make_tmem_copy(atom, tAcc_epi[(None,None,0,0)])` → `thr.partition_S(flat_divide(tAcc, epi_tile))` |
| Registers → smem | `stmatrix` (16-bit destination, transposed per 8×8 matrix) or plain `st.shared`, depending on the thread-value ownership of the t2r atom | `get_smem_store_op(layout_c, c_dtype, acc_dtype, tiled_copy_t2r)` → `cute.make_tiled_copy_D(atom_r2s, tiled_copy_t2r)` → `thr.partition_D(sC)`; on the register side `tiled_copy_r2s.retile(rC)` |
| smem → gmem | the TMA bulk-tensor store reads a **swizzled** staging face; generic-proxy writes must first be fenced to the async proxy | `make_smem_layout_epi(c_dtype, layout_c, epi_tile, stages)` (swizzle attached to the pointer); `cpasync.make_tiled_tma_atom(CopyBulkTensorTileS2GOp(), c, slice_(L,(None,None,0)), epi_tile)`; `fence_proxy("async.shared", space="cta")` + named barrier + `PipelineTmaStore` |

## When needed (trigger)

**Look up the error you see** (each row has the correct spelling in the example below):

| Error / symptom | What you did | Corresponding line in the example |
|---|---|---|
| `Expected source and destination tensors to have the same rank, but got 2 and 3` | the two sides of r2s were not partitioned from the same tiled copy: register side used the t2r fragment, smem side a hand-written `zipped_divide` | register side always `tRS_rC = tiled_copy_r2s.retile(tTR_rC)`, smem side always `thr_copy_r2s.partition_D(sC)` |
| `Expected source and destination tensors to have the same size in mode-1, but got 1 and 2` | built r2s with `make_tiled_copy_S` (applying t2r's thread layout with the registers as the source) | `tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)` — r2s's **D** (smem) inherits t2r's thread-value layout |
| `'cute.copy' op expects same size in restAtomVRank, but got srcShape … dstShape …` | the src / dst side of r2s has one rest mode too many or too few (hand-written tiler, or an smem face not produced by `make_smem_layout_epi`) | the face comes from `make_smem_layout_epi(..., epi_tile, stages)` → `(EPI_M, EPI_N, STAGE)`; `tRS_sC[(None, None, None, buf)]` takes one stage |
| `no candidate atom built a TMEM copy against the accumulator segment` / `make_tmem_copy` rejects | a hand-picked `Ld16x256bOp(x2)` or similar does not match the datapath / column count of the epi tile | do not hand-pick the atom: `get_tmem_load_op(...)` derives `Ld32x32b`/`Ld16x256b` and the `Repetition` from (cta_tile, major, width, epi_tile) |
| `'cp_async_bulk_commit_group' not found` / `no cute.arch.fence_proxy` | searched for the name in the wrong module | they are all in **`cute.arch`**: `cute.arch.fence_proxy("async.shared", space="cta")`, `cute.arch.cp_async_bulk_commit_group()`, `cute.arch.cp_async_bulk_wait_group(n, read=True)`, `cute.arch.fence_view_async_tmem_load()`; with `pipeline.PipelineTmaStore`, commit / acquire / tail are already wrapped |
| the TMA store writes stale data / half a tile | issued TMA right after the smem writes, without `fence_proxy` + barrier | `cute.copy(tiled_copy_r2s, …)` → `fence_proxy("async.shared", space="cta")` → `epi_bar.arrive_and_wait()` → one warp issues `cute.copy(tma_atom_c, …)` |
| a staging stage gets overwritten / results intermittently wrong | the next sub-tile writes into a stage the previous sub-tile's TMA is still reading | `c_pipeline.producer_commit()` + `c_pipeline.producer_acquire()` (= `wait_group(stages-1, read=True)`) right after the TMA store; `producer_tail()` before exit |
| hand-configured `Ld16x256bOp(Repetition.x1)` + `StMatrix8x8x16bOp(·, 4)`: compiles and runs, C values misplaced in blocks | the stmatrix matrix count exceeds the registers delivered by one read-back (gotcha 19) | x1 only pairs with `num_matrices` ∈ {1, 2}; or let the helper choose the atom |
| `Can not pick tmem_rep based on cta tile shape and tmem atom.` (`get_tmem_load_op`); or with a hand-written atom, at run time `an illegal memory access was encountered` | the block width is not a multiple of 8 | use a block width that is a multiple of 8 (gotcha 19) |

**Structural triggers** (look before you run anything): your accumulator is already correct (read back to gmem bit-exact) and you are about to switch to a real epilogue that
"converts to bf16 / goes through smem / TMA store"; or you are in the pair form and each CTA must write out its own 128 rows; or the output is **M-major** (column-major) rather than N-major.
**First run this entry's example verbatim in your job** (the geometry is parameterised), get `max_diff=0.0`, and only then swap in your own tile, warp roles and stage count.

## Minimal runnable example

Only the lines that **differ** from [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) are listed; copy the `…` parts from that entry.
The output C is bf16 (M, N); **both N-major (`leading_dim=1`) and M-major (`leading_dim=0`) are certified**;
each CTA writes its own `CTA_TILE_M × TILE_N` block (pair form = half of the pair tile).

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

IO_DTYPE, ACC_DTYPE, C_DTYPE = cutlass.BFloat16, cutlass.Float32, cutlass.BFloat16
INST_MNK, TILER_MNK, CLUSTER_MNK = (256, 224, 16), (256, 224, 64), (2, 1, 1)   # pair; single CTA uses (128, N, 16) / (1,1,1)
CTA_TILE_MNK = (TILER_MNK[0] // CLUSTER_MNK[0], TILER_MNK[1], TILER_MNK[2])    # each CTA's own block: (128, 224, 64)
AB_STAGES, ACC_STAGE, C_STAGES, THREADS = 2, 1, 2, 128
_ST = {}                                       # holds only C's majorness (a Python enum); static layouts must be passed as kernel parameters, see gotcha 10

# ---- three partitioning helpers: plain Python functions, inlined when called in the kernel; both sides always start from the same tiled copy ----
def epi_partition_t2r(tidx, tCtAcc, tCgC, epi_tile):
    copy_atom_t2r = sm100_utils.get_tmem_load_op(CTA_TILE_MNK, _ST["c_layout"], C_DTYPE, ACC_DTYPE, epi_tile, TWO_CTA)
    tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)         # (EPI_M, EPI_N, RestM, RestN)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)                                # (T2R, T2R_M, T2R_N, RestM, RestN)
    gC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
    tTR_gC = thr_copy_t2r.partition_D(gC_epi)
    tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, ACC_DTYPE)
    return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, gC_epi

def epi_partition_r2s(tiled_copy_t2r, tTR_rC, tidx, sC):
    copy_atom_r2s = sm100_utils.get_smem_store_op(_ST["c_layout"], C_DTYPE, ACC_DTYPE, tiled_copy_t2r)
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)      # the D side (smem) inherits t2r's thread layout
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_copy_r2s.partition_D(sC)                                        # (R2S, R2S_M, R2S_N, STAGE)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)                                       # register side: no partition, retile
    return tiled_copy_r2s, tRS_rC, tRS_sC

def epi_partition_tma(tma_atom_c, gC_epi, sC):
    bSG_sC, bSG_gC = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1),
                                           cute.group_modes(sC, 0, 2), cute.group_modes(gC_epi, 0, 2))
    return bSG_sC, bSG_gC                                                        # ((V), STAGE) / ((V), RestM, RestN)

@cute.kernel
def kernel(tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl,
           tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,                       # the four extra parameters:
           a_smem_layout, b_smem_layout,
           c_smem_layout_staged: cute.ComposedLayout, epi_tile: cute.Tile,       #   static layouts enter the kernel as parameters
           cta_layout_vmnk):
    …                                                                            # coordinates, warp_idx, mma_coord as in that entry
    smem = cutlass.utils.SmemAllocator(); storage = smem.allocate(SharedStorage)
    sA, sB = …                                                                   # as in that entry
    sC = smem.allocate_tensor(element_type=C_DTYPE, layout=c_smem_layout_staged.outer,   # (EPI_M, EPI_N, STAGE)
                              byte_alignment=1024, swizzle=c_smem_layout_staged.inner)
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_c)                                  # A/B prefetch as in that entry
    ab_producer, ab_consumer = …; acc_producer, acc_consumer = …                 # as in that entry
    c_pipeline = pipeline.PipelineTmaStore.create(                              # bulk async-group accounting, no mbarrier
        num_stages=C_STAGES, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS))
    epi_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=THREADS)          # id 1 belongs to the TMEM allocator
    gA, gB, gC, thr_mma, tCgA, tCgB, tCgC, tCrA, tCrB, tCtAcc = …                # as in that entry; tCgC = this CTA's (128, N)
    tmem …; tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)                  # as in that entry

    tiled_copy_t2r, tTR_tAcc, tTR_rAcc, gC_epi = epi_partition_t2r(tidx, tCtAcc, tCgC, epi_tile)
    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, C_DTYPE)
    tiled_copy_r2s, tRS_rC, tRS_sC = epi_partition_r2s(tiled_copy_t2r, tTR_rC, tidx, sC)
    bSG_sC, bSG_gC = epi_partition_tma(tma_atom_c, gC_epi, sC)

    … main loop as in that entry …                                               # leader issues, acc_producer.commit()

    # ---- epilogue: each sub-tile takes the four-leg path ----
    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    tTR_tAcc_g = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))            # (V, M, N, (RestM, RestN))
    bSG_gC_g = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))                  # (V, (RestM, RestN))
    for st in cutlass.range(cute.size(tTR_tAcc_g.shape, mode=[3])):
        cute.copy(tiled_copy_t2r, tTR_tAcc_g[(None, None, None, st)], tTR_rAcc)  # (1) TMEM -> registers
        tRS_rC.store(tiled_copy_r2s.retile(tTR_rAcc).load().to(C_DTYPE))       # (2) convert (in the r2s view)
        buf = st % C_STAGES
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, buf)])       # (3) registers -> smem stage
        cute.arch.fence_proxy("async.shared", space="cta")                       #    generic -> async proxy
        epi_bar.arrive_and_wait()                                                #    all epilogue threads done writing
        if warp_idx == 0:
            cute.copy(tma_atom_c, bSG_sC[(None, buf)], bSG_gC_g[(None, st)])     # (4) TMA store one sub-tile
            c_pipeline.producer_commit()                                         #    cp.async.bulk.commit_group
            c_pipeline.producer_acquire()                                        #    wait_group(C_STAGES-1, read): stage reusable
        epi_bar.arrive_and_wait()
    cute.arch.fence_view_async_tmem_load()                                       # before releasing the accumulator
    acc_full.release()
    … tail() / pipeline.sync(barrier_id=1) / tmem.free(tmem_ptr) as in that entry …
    c_pipeline.producer_tail()                                                   # wait_group(0): all stores landed before exit

@cute.jit
def host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    … tiled_mma / a_smem_layout / b_smem_layout / cta_layout_vmnk / TMA atoms of A, B as in that entry …
    c_layout = utils.LayoutEnum.from_tensor(c)                                   # ROW_MAJOR (N contiguous) or COL_MAJOR (M contiguous)
    epi_tile = sm100_utils.compute_epilogue_tile_shape(CTA_TILE_MNK, TWO_CTA, c_layout, C_DTYPE)
    c_smem_layout_staged = sm100_utils.make_smem_layout_epi(C_DTYPE, c_layout, epi_tile, C_STAGES)
    epi_smem_layout = cute.slice_(c_smem_layout_staged, (None, None, 0))
    c_tma_atom, c_tma_tensor = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, epi_tile)
    _ST["c_layout"] = c_layout                                                   # an enum can be read through a global; a layout cannot
    kernel(tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor, c_tma_atom, c_tma_tensor,
           a_smem_layout, b_smem_layout, c_smem_layout_staged, epi_tile, cta_layout_vmnk
           ).launch(grid=…, block=[THREADS, 1, 1], cluster=CLUSTER_MNK)

# host side: N-major C = from_dlpack(c, assumed_align=16).mark_layout_dynamic(leading_dim=1)
#            M-major C = torch.zeros(N, M).t() -> from_dlpack(...).mark_layout_dynamic(leading_dim=0)
```

**Geometry readings printed during certification** (when copying, check what your own atoms and partitions look like; C bf16, accumulator fp32, 4 warps per CTA):

| Form / per-CTA tile | C major | `epi_tile` | t2r atom (`get_tmem_load_op`) | r2s atom (`get_smem_store_op`) | Staging face (one stage, `make_smem_layout_epi`) | Sub-tiles |
|---|---|---|---|---|---|---|
| single CTA (128,128) | N (`leading_dim=1`) | `(128:1, 32:1)` | `Ld32x32bOp` ×32: TV `(32,1024):(0,1)` → `(32,32):(32,1)`, 32 columns per thread | plain `st.shared` (ThrID `1:0`, not stmatrix) | `S<2,4,3> o 0 o ((8,16),(32,1)):((32,256),(1,0))`, 8 KiB | 4 |
| single CTA (128,128) | M (`leading_dim=0`) | `(128:1, 32:1)` | `Ld16x256bOp` ×4: TV `(32,512):(0,1)` → `((4,8),(2,2,4)):((2,32),(1,256,8))` | `StMatrix8x8x16bOp(transpose, 4)`: ThrID `32:1`, TV Dst `(32,8):(8,1)` | `S<3,4,3> o 0 o ((64,2),(8,4)):((1,2048),(64,512))`, 8 KiB | 4 |
| pair (256,224) → per CTA (128,224) | N | `(128:1, 32:1)` | same `Ld32x32bOp` ×32 | same plain store | same `S<2,4,3>` | 7 |
| pair (256,224) → per CTA (128,224) | M | `(128:1, 32:1)` | same `Ld16x256bOp` ×4 | same stmatrix ×4 | same `S<3,4,3>` | 7 (`cute.compile` launched twice in a row, both 0.0) |

Partition readings (M-major, pair): `tTR_tAcc = (((32,16),1),2,1,1,7)`, `tTR_rAcc = (((2,2,4),1),2,1)`, `tRS_rC = ((8,2),2,1):((1,8),16,0)`,
`tRS_sC = ((8,2),2,1,(1,2))`, `bSG_sC = ((2048,2),(1,2))`, `bSG_gC = (((64,32),2),1,7)`; N-major: `tTR_rAcc = ((32,1),1,1)`,
`tRS_rC = ((1,32),1,1)`, `bSG_sC = ((4096,1),(1,2))`. **Two fact rows**: (1) with the same certified partitioning, hand-picking the tile as `(128,16)` (M-major, pair) also
lowers and is exact — `Ld16x256bOp` ×2 + stmatrix ×4, 4 KiB staging, 14 sub-tiles; so "rank mismatch / mode-1 mismatch" is not the tile's fault but the partitioning's,
yet the tile should still be left to the helper (it picks by 4096 elements / iteration and the TMA alignment floor). (2) Swapping only `make_tiled_copy_D` for `make_tiled_copy_S`: `tRS_rC`
becomes `((8,(2,2,16)),1,1)`, `tRS_sC` becomes `((8,(4,16)),2,1,(1,2))`, and `cute.copy` reports `Expected source and destination tensors to have the
same size in mode-1, but got 1 and 2`.

## Gotchas

1. **None of the four legs' atoms is hand-picked.** The tile is set by `compute_epilogue_tile_shape`, the t2r atom is derived by `get_tmem_load_op` from
   (CTA tile, C's majorness, C / accumulator width, tile), and the r2s atom is derived by `get_smem_store_op` from **t2r's tiled copy**.
   Hand-picking an `Ld16x256bOp(x2)` and then matching a stmatrix to it is where "rank mismatch / mode-1 mismatch / restAtomVRank mismatch" come from.
2. **r2s uses `make_tiled_copy_D`, not `_S`.** The register values are already distributed by t2r's thread layout; r2s applies the same thread layout to the smem (D side),
   and the register side only needs `retile`. Applying it the other way around (`_S`) is off by a factor of 2 in mode-1.
3. **The staging face is the three-mode (EPI_M, EPI_N, STAGE) from `make_smem_layout_epi`, swizzle attached to the pointer.** Same discipline as the A/B faces:
   `allocate_tensor(layout=L.outer, swizzle=L.inner)`; build the TMA atom from the one stage `slice_(L, (None, None, 0))`.
4. **The generic proxy writes smem and the async proxy (TMA) reads it; in between you must `fence_proxy("async.shared", space="cta")` and then barrier.**
   Missing fence = occasional stale data; missing barrier = the store is issued before other warps finish writing.
5. **The TMA store is a bulk async-group, not an mbarrier.** `PipelineTmaStore`'s `producer_commit` (commit_group) /
   `producer_acquire` (`wait_group(stages-1, read=True)`) / `producer_tail` (`wait_group(0)`) are all there is;
   `read=True` only waits until "smem has been read", not until gmem has landed — so `producer_tail()` before exit cannot be omitted.
   **The two wait forms and where each applies**: `cute.arch.cp_async_bulk_wait_group(n)` without `read=` is the write-completion form (PTX `cp.async.bulk.wait_group N`),
   `read=True` emits `cp.async.bulk.wait_group.read N`; in the certification probe, the SASS of the write-completion form has exactly one extra instruction over the read-completion form, a `CCTL.IVALL`. After changing the wait form,
   go by the PTX text (grep for both spellings; the exit path should no longer contain the one without `.read`), not by source comments — if only one call site was changed and another default-form wait is still there,
   it compiles, the numerics are all right, and the pricing is a noop. "Cannot be omitted before exit" means the smem source must have been read before the CTA exits; for this drain **the read-completion form is enough**: in the certification probe
   the warp that issues the stores exits the CTA after only `wait_group(0, read=True)`, and the gmem read back after host sync is bit-correct (negative control: overwriting
   the staging face before the wait shows poison values on read-back). "Does not wait for gmem to land" matters only when **the same grid reads this gmem back again** (a later phase of the same kernel, or another CTA synchronised via a flag / atomic):
   before that read, the issuing thread must `wait_group(0)` (write completion) and then publish the signal — this case is not covered by the probe; write it per PTX ISA semantics.
   Whole kernel: both times only the wait form was changed, nothing got faster (same job read completion 0.01821 ms vs write completion 0.01808 ms; on another tree switching the exit drain to read completion, gate 0.0729 → 0.0743 ms);
   no gain is claimed.
6. **An M-major C takes a different set of atoms.** For the same tile, N-major gets `Ld32x32b` + plain `st.shared`, M-major gets
   `Ld16x256b` + `StMatrix8x8x16b(transpose)` — decided by `LayoutEnum.from_tensor(c)` (`leading_dim`);
   on the host, an M-major torch tensor must be marked `leading_dim=0`.
7. **In the pair form the epilogue is one per CTA.** `tCgC = thr_mma.partition_C(gC)` (`get_slice(rank)`) is already this CTA's
   128 rows; `CTA_TILE_MNK` uses "half of the pair tile", not the MMA's M=256, otherwise the datapath count computed by `get_tmem_load_op` is wrong.
8. **`fence_view_async_tmem_load()` before releasing the accumulator.** `tcgen05.ld` is asynchronous; `release()` without the fence and the next tile's MMA
   may overwrite columns that have not finished being read.
9. **Names in `cute.arch`**: `fence_proxy(kind, space=)`, `cp_async_bulk_commit_group()`, `cp_async_bulk_wait_group(n, read=)`,
   `fence_view_async_shared()`, `fence_view_async_tmem_load()` / `_store()`. Searching other modules by name will not find them.
10. **Static layouts computed on the host (`epi_tile`, the staging face) must be passed as kernel parameters**, not smuggled into the kernel through globals / closures: that gives
    `'cute.composed_get_outer' op using value defined outside the region … required by region isolation constraints`
    (an ICE worded "This is a bug in the DSL", but it is actually yours). Python enums / ints can be read through globals. Also, inside `@cute.jit`, `if` / `or` on **Python values**
    are staged too: `NameError … Variables used inside staged control flow must be defined before the control flow
    region`, `DSLNotImplemented: <class 'NoneType'> is not supported` — write `if cutlass.const_expr(...)`, assign before the branch,
    and use only plain int constants in the condition (see [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)).
11. **A staging face without swizzle compiles, runs and is numerically correct — the stmatrix write side just has an 8-way conflict by construction.** A flat row-major face of 16 rows × 128 columns bf16
    has a 256 B row pitch, so the 8 rows of one `stmatrix` 8×8 group land on the same 4 banks, and every staging write is an 8-way conflict.
    KEEP readings on the same tree, same tile, with only this one face changed from swizzle NONE to SW128: `bankconf_st` 16.67M → 0.39M,
    `smem_wavefronts_st` 18.79M → 2.52M, `tensor_pct` 76.14 → 81.09, `stall_long_sb` 22.73 → 18.72 (same-package reading:
    `regs_per_thread` 58 → 74, not isolated). **The gate's ms is not load-bearing for this KEEP**: device window 0.3836 → 0.4259 ms (+11.03%)
    while the per-call window 0.4057 → 0.3855 ms (−4.98%); the two windows move in opposite directions, and cross-job absolute ms on this tree carries ±20% node drift
    (see [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md) gotcha 5); this gotcha's evidence is the per-kernel counters.
    Because a missing swizzle **raises no error**, write it as a contract self-check: fail if `L.inner is None` (that is exactly what `swizzle=L.inner` passes at face allocation),
    and give the probe a separate non-numeric "face has swizzle" contract, otherwise "trace succeeded" holds just as well for an unswizzled face.
12. **Once a face has swizzle, there is no "second view".** Without swizzle, the same bytes can be wrapped in another row-major `[m][n]` layout and handed to the
    TMA store ((n, m) and (m, n) land at the same address); swizzle permutes the 128 B blocks within a row, so "same bytes, another mode tree" is no longer the identity —
    a second view addresses swizzled bytes as linear bytes, and the store writes out a C whose 128 B blocks within each row are permuted. The correct composition changes three places together:
    (1) build the descriptor from the single-slot slice **of the staging face itself**: `make_tiled_tma_atom(CopyBulkTensorTileS2GOp(), mC_frame,
    slice_(L, (None, None, 0)), epi_tile, 1)`; (2) present the gmem side in **the same frame**: when MMA M = C's n, `mC_frame =
    make_tensor(pC, make_layout((n_cols, m_total), stride=(1, n_cols)))` — the same row-major C bytes read in the staging face's mode order;
    after `zipped_divide(gC_frame, epi_tile)` the per-segment coordinates are transposed accordingly into `(n_block, m_seg)`; (3) in the kernel, `tma_partition(atom, 0,
    make_layout(1), group_modes(sC, 0, rank−1), gC)` reads **the tensor stmatrix wrote**, without building another view.
    The certification probe's payload is each cell's own absolute coordinates (`partition_D(make_identity_tensor(...))`): a permutation from inconsistent swizzle on the two sides
    is read out directly, not merely detected. The descriptor rank does not change because of this (the `UTMASTG.2D` on the `cache_ops` axis is the same before and after).
13. **Choose `LayoutEnum` by the MMA's frame, not by the torch tensor's row-major-ness.** When the MMA is issued with M = the output's contiguous dimension (a transposed block of C in TMEM),
    the epilogue's two op selectors and the staging-face builder are all keyed on `LayoutEnum.COL_MAJOR` — even though the gmem C is row-major
    `[m][n]`. This is the M-major row of gotcha 6: `Ld16x256b` t2r + transposed `stmatrix` + an `S<3,4,3>` face, and in the pair form each CTA writes its own
    n-block. A hand-assembled substitute uses the **MN-major** 128 B atom: `make_smem_layout_atom(SmemLayoutAtomKind.MN_SW128, bf16)` then
    `tile_to_shape(atom, (EPI_M, EPI_N, STAGES), order=(0, 1, 2))` (the K-major family is for operand faces; using it here puts you in the wrong frame).
    One parameter name: DSL 4.6.2's `make_smem_layout_epi` spells the element-type parameter `epi_dtype`.

14. **When stores are issued by a dedicated warp and the face is written by another role, what `wait_group(S−1, read=True)` releases is the face of the **previous** block,
    not this block's.** `PipelineTmaStore`'s acquire is accounting within one warp (gotcha 5); if the face has to be handed back to another role via mbarrier,
    you must translate that accounting into arrives yourself:
    - Open S copies of the face: `make_smem_layout_epi(dtype, layout, epi_tile, n_sub * S)`, slot `s` = the contiguous run of sub-tiles
      `[s·n_sub, (s+1)·n_sub)`; the TMA atom is still built from one sub-tile, and the descriptor does not change with S; S mbarriers each for `FULL` / `EMPTY`.
    - The store-issuing warp, per block c: wait `FULL[c % S]` (parity `(c // S) % 2`) → issue `UTMASTG` → `commit_group()` →
      `wait_group(S−1, read=True)` → **arrive on `EMPTY[(c−1) % S]`** (only when `c ≥ 1`). `wait_group(S−1)` only guarantees that groups
      older than this one have finished reading smem, while this block's store is still in flight, so what can be released is exactly the previous block's face.
    - The face-writing role waits `EMPTY[c % S]` with parity `(c // S − 1) % 2`, only when `c ≥ S`; the last block's face is never released (nobody waits for it).
    - **Mixed paths**: if there is a branch that does not go through TMA (a tail block written via LSU under a row predicate), it commits no group of its own, so it must
      `wait_group(0, read=True)` and then arrive, otherwise the per-slot pattern "one arrive per generation" is broken by the tail block.
    Certification readings (KEEP evaluation, eval job id not recorded; S = 1 → 2): contract axis `rings` `[stride=0x1, stride=0x3f]` →
    `[stride=0x1, stride=0x1,slots=536870911]` (the slot index became the run-time `c % 2`; `slots=` is the reading of the 0x1fffffff
    mask on the index chain, not 2 — for how to read it see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11); gate
    0.20533 → 0.20733 ms (+0.97%) — **in this structure** (the store-issuing warp lives in the 24-register yielding region, bundled with removing one
    item-level named-barrier rendezvous) the ms is not load-bearing, and the slot bookkeeping (`c % S`, `(c−1) % S`, sub-tile base address, two drain
    paths) landing in the 24-register tier took `spill_st_sectors` 9,024 → 24,896: before adding bookkeeping to a yielding warp, see
    [`warp_specialization`](warp_specialization.md) split self-selection 3.

15. **Writing gmem directly (per-thread `STG.128`) and "staging face + bulk tensor store" are two write-out forms of the same read-back; the presence or absence of
    `UTMASTG` in `cache_ops` is their footprint, and on a tree whose LSU write path is shared with other roles, the direct write is slower.**
    - **Direct-write spelling**: `cute.autovec_copy(rB, cute.make_tensor((mO.iterator + off).align(16), cute.make_layout(CHUNK)))`,
      where `off` is a run-time element offset (one contiguous run per thread). `.align(16)` is an assertion, not rounding — it is the sole basis on which `autovec_copy` chooses
      `st.global.v4.b32` (`STG.E.128`); without it, the per-element fallback **still compiles, runs and is numerically correct**, just with 8× the store
      count, so the probe must have a separate non-numeric contract checking for `st.global.v4.b32` in the PTX (contract `S9-vec` of a GPU probe, not included in this release;
      `S9-store` / `S9-relaunch` byte-exact). Removed together with it: the staging-face view, the store atom / descriptor, `fence_view_async_shared`,
      the 128-thread named rendezvous, `commit_group` / `wait_group`.
    - **Staging-face spelling (the `st.shared` variant of this entry's main line, thread = face row)**: a 128-bit `CopyUniversalOp` atom,
      `make_tiled_copy_tv(atom, make_layout((ROWS, 1), stride=(1, ROWS)), make_layout((1, CHUNK)))`, thread t writes row t of the face;
      `partition_D` the face's `(row, col_in_chunk, chunk)` view once per **compile-time** chunk coordinate and hoist it out of the item loop (adding a run-time offset
      to the partitioned iterator carries the swizzle into an address no longer covered by the 128-bit alignment fact); build the descriptor from the same face's `(row, col, 1)` view
      (`make_composed_layout(L.inner, 0, view)`, the "no second view" of gotcha 12); then `fence_view_async_shared` → 128-thread
      named rendezvous → one warp `cute.copy(atom, ss[None, 0], gs)` + `commit_group` (**not** wrapped in `elect_one`: the bulk tensor copy
      does its own election internally; wrapping it deadlocks) → `wait_group(0, read=True)` only on the issuing thread. A second probe (contracts
      `S13-face` / `S13-store` / `S13-relaunch`) certified this path byte-exact on 32-column chunks, including face reuse between two items separated by a drain.
    - **Readings (same tree, two KEEPs, opposite directions)**: face → direct write: `cache_ops` loses `UTMASTG.5D`, `nbar_widths` loses `[128]`,
      gate 1.0667 → 1.13 ms (**+5.93%**), `tensor_pct` 53.67 → 50.65, `stall_long_sb` 3.18 → 3.51, `smem_dynamic` unchanged
      (the face stays for the inbound load). Direct write → face: `cache_ops` gets `UTMASTG.5D` back, `nbar_widths` gets `[128]` back, gate 0.9963 → 0.9746 ms
      (**−2.18%**), `tensor_pct` 59.89 → 61.81, `stall_barrier` 1.76 → 1.81. Both are same-package readings (the former also moved the accumulator release
      to right after the last read-back batch, the latter also moved the issue point of the inbound load and removed one in-register lane transpose); recorded, not split.
    - **Load-bearing conditions**: per thread per item 8 chunks × 4 `STG.128` (64 B contiguous runs), 128 threads → 4 096 `STG.128` per item per CTA
      on the LSU / L1 / L2 write path, which at the same time carries exchange-face traffic, gather landings and the operand faces' `STS`; switching to face + TMA store
      is 512 smem wavefronts plus one TMA-engine read. 1 CTA/SM, one resident wave, pair UMMA. **How to read it**: a missing
      `UTMASTG` in `cache_ops` is not "one hop saved"; on a tree where the write path is already shared, it adds back 4 096 LSU instructions; conversely, it is not a universal −2% either.

16. **One face time-shared by an inbound TMA load and an outbound TMA store: the drain waits only on the store-issuing thread, the other warps inherit the ordering through
    the next load's completion chain; a tail item with no next load adds one named rendezvous.**
    - `cp_async_bulk_wait_group(0, read=True)` is written inside `if warp_idx == 0`: what `read=True` waits for is exactly "the face has been read", i.e. the event that lets
      the next inbound load write this face, so only the thread that issues the next load needs to wait for it — right after the drain it issues load(item+2).
    - Before writing the face next time, the other three warps wait on the mbarrier "the consumer of load(item+1) has read the face"; that comes after the load, and the load after the drain,
      so no `STS` can overtake an in-flight store.
    - The exception is the last item: there is no load(item+1), so there is no such edge to inherit — warp 0 has already drained its own stores at the bottom of the previous body, and the top of the tail body
      adds one `barrier(BAR_EPI, 128)` to tell warps 1–3. Steady state: one rendezvous per item; the tail body two; once per CTA per launch.
    - **Rendezvous before the store, drain after the rendezvous.** The other way round (drain before the rendezvous) makes all four warps wait for the drain: on the same tree
      `stall_barrier` reads 2.22 vs 1.81.

19. **The repetition count of `Ld16x256b` and the matrix count of `stmatrix` must match, and the block width must be a multiple of 8 — hand-configure them wrong and neither compilation nor execution complains; the writes are just misplaced.**
    - **Matching rule.** One `Ld16x256b` repetition = per warp 16 rows × 8 columns fp32 = 4 fp32 per thread = 2 32-bit registers after converting to 16-bit
      = two 8×8 matrices. So `Ld16x256bOp(Repetition.xR)` with `StMatrix8x8x16bOp(transpose, n)` requires `n ≤ 2R`: **x1 can only pair with `n` ∈ {1, 2}**,
      only from x2 up can it pair with 4. `Repetition` comes only in powers of 2; the block width W just has to be a multiple of `8R` (`make_tmem_copy` tiles the atom along columns `W / 8R` times;
      block widths that are odd multiples of 8 use x1, tiled an odd number of times). r2s is still `make_tiled_copy_D(atom, t2r)` + `retile`.
    - **Certification readings** (M-major C, 128 lanes, 4 epilogue warps, transposed stmatrix): x1 + `n = 2` (block width an odd multiple of 8), x2 + `n = 2`, x2 + `n = 4`,
      x4 + `n = 4` are all bit-exact; **x1 + `n = 4` compiles, runs, `max_diff = 1831`** (a matrix stitched across two atoms, values not matching the TMEM columns), caught only by a numeric comparison;
      with a block width that is not a multiple of 8, both candidates (x1 + `n = 1` / `n = 2`) fail at run time with `CUDA error: an illegal memory access was encountered`.
    - **The helper did not pick an illegal combination in the measurements**: `get_tmem_load_op` / `get_smem_store_op` (`COL_MAJOR`, epi tile `(128, W)`) give x1 + `n = 2` for odd multiples of 8
      and x2 + `n = 4` for odd multiples of 16 (including 16 itself); when the block width is not a multiple of 8, `get_tmem_load_op` reports
      `Can not pick tmem_rep based on cta tile shape and tmem atom.`. For narrow blocks (8 columns, x1), in both frames — single CTA 128 rows and pair-form 64 rows + half-N split —
      and both C majors, the helper picks `StMatrix8x8x16b … number of matrices = 2`, and the SASS is item-for-item identical to the explicit atom (`STSM.*.2` ×2, `STS.U16` 0,
      the other 3 plain `STS` are setup writes). **Illegal combinations only appear with hand-configured atoms**; if you hand-configure, write `n ≤ 2R` as a trace-time `assert`, and add a SASS contract
      (`STSM > 0`, `STS.U16 == 0`).
    - **When the gmem column count is not a multiple of the block width, do not write a mask**: issue full-width TMA stores for every block as usual; columns beyond the descriptor's tensor shape are dropped (in both of the probe's column tails — the last block
      having only some valid columns, and a whole block out of bounds — the guard rows were not written). This assumes the shape in the descriptor is the real column count (a `mark_layout_dynamic` C, or compiled with the real shape).
      When a block's starting point in gmem is not an integer multiple of W (blocks of different widths concatenated in one row), do not use `start // W` as the `local_tile` block index (it rounds down and writes to the wrong place);
      first `cute.domain_offset((0, start), mC)`, then `local_tile(…, (None, 0))`.
    - **Whole-kernel readings**: one same-job A/B switching the epilogue to a wider odd-multiple-of-8 block (x1 + `n = 2`) read +0.96% / +8.0% on two workloads (slower, not kept);
      replacing the helper-chosen r2s with an explicit atom (SASS unchanged) — gate 0.0164 → 0.0165 ms, flat. The block width is a tuning knob; this gotcha only covers "which combinations are legal".

## Sources

- Certification probe — **status: ✅ GPU-certified**, final verdict
  `PROBE PASS max_diff=0.0` (certification round 4; the first three rounds were the probe itself hitting gotcha 10 in three forms,
  unrelated to the epilogue composition), all four contracts passed in the same round: `T0-lowers` — E1..E4 all compile to launchable kernels; `T1-1cta-exact` 0.0 — single CTA (128,128),
  both N-major / M-major C; `T2-pair-exact` 0.0 — pair (256,224), both C; `T3-relaunch` 0.0 — M-major pair launched twice via `cute.compile`.
  A/B drawn from bf16 integers {−1,0,1} so that |sum| ≤ 256 is exact in bf16; C is bf16, **bit-equal** to an fp32 `einsum` reference. Each variant in its own subprocess
  (`sys.executable`), 240 s budget; the two negative variants N1 (hand-picked tile) / N2 (`make_tiled_copy_S`) only produce fact rows.
- Public interfaces: `cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape` / `get_tmem_load_op` / `get_smem_store_op` /
  `make_smem_layout_epi`; `cutlass.cute.nvgpu.tcgen05.make_tmem_copy` / `Ld32x32bOp` / `Ld16x256bOp`;
  `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom` / `CopyBulkTensorTileS2GOp` / `tma_partition`;
  `cute.make_tiled_copy_D` / `TiledCopy.retile` / `flat_divide` / `group_modes` / `slice_`;
  `cutlass.pipeline.PipelineTmaStore` / `NamedBarrier`; `cutlass.utils.LayoutEnum.from_tensor`;
  `cute.arch.fence_proxy` / `cp_async_bulk_commit_group` / `cp_async_bulk_wait_group` / `fence_view_async_tmem_load`.
- Official docs: CUTLASS CuTe DSL documentation https://docs.nvidia.com/cutlass/latest/ (the epilogue section of the Blackwell GEMM example; this entry's example is
  its three `epilog_*_copy_and_partition` partitioning helpers and epilogue main loop reduced to one tile); PTX ISA sections on `tcgen05.ld` /
  `stmatrix` / `cp.async.bulk.tensor` / `fence.proxy.async`.
- **Certification probe for gotchas 11–13** —
  **status: ✅ GPU-certified**, final verdict `PROBE PASS max_diff=0.0` (after 2 rounds; in round 1 both contracts FAILed together because the parameter name of `make_smem_layout_epi`
  was not guessed as `epi_dtype`). Two contracts: `C1-face` (non-numeric) — `make_smem_layout_epi(bf16, COL_MAJOR, (128, 16), 2)`
  gives a three-mode `(MMA_M, MMA_N, SLOT)` composed layout, the single-slot slice is accepted by `make_tiled_tma_atom` against the C in the MMA frame, and the face **has**
  swizzle; `C2-store` (numeric) — pair form TMEM → `Ld16x256b` t2r → bf16 pack → transposed stmatrix into the swizzled slot → TMA
  store, payload = each cell's absolute coordinates (slot 0 stores lane = n, slot 1 stores column = m); every row of the read-back C equals `n % 128` / `m − 16`,
  and the n-blocks of both CTAs of the pair are all bit-equal.
- **KEEP evaluation for gotcha 11** (NCU dossier census + gate window; eval job id not in the status record): only the staging face's swizzle changed NONE → SW128,
  with the descriptor and gmem frame changed per gotcha 12; axis `rings` 3 → 4 entries (`stride=0x1..0x3` → `..0x4`, rings untouched, codegen followed),
  `cache_ops` unchanged, gate 0.3836 → 0.4259 ms (per-call window 0.4057 → 0.3855 ms, the two windows in opposite directions, ms not load-bearing),
  `bankconf_st` 16.67M → 0.39M, `smem_wavefronts_st` 18.79M → 2.52M.
- **KEEP evaluation for gotcha 14** (NCU dossier census + gate window, eval job id not recorded; the KEEP eval passed correctness): output face
  1 → 2 copies + `wait_group(1, read=True)` + an `EMPTY` arrive lagging one block, bundled with removing one item-level named-barrier rendezvous and
  reading item parameters one round early: axis `rings` `[stride=0x1, stride=0x3f]` → `[stride=0x1, stride=0x1,slots=536870911]`,
  `cache_ops` / `supply` / `commit` unchanged, gate 0.20533 → 0.20733 ms, `spill_ld/st_sectors` 25,760 / 9,024 →
  27,104 / 24,896 (same-package readings, not separable; the store-side increase is attributed by per-line information to the bookkeeping in the yielding region).
- **KEEP evaluations for gotcha 15** (NCU dossier census + gate window, the second kind of evidence in this index; eval job ids of both KEEPs not in the status record):
  face → direct write: `cache_ops` `["UTMALDG.2D.GATHER4.2CTA", "UTMALDG.5D.2CTA", "UTMASTG.5D"]` → `["UTMALDG.2D.GATHER4.2CTA",
  "UTMALDG.5D.2CTA"]`, `nbar_widths` `[[64], [128], [512]]` → `[[64], [512]]`, gate 1.0667 → 1.13 ms; direct write → face: both axes back
  with the same readings in reverse, gate 0.9963 → 0.9746 ms; correctness PASS both times. One round for each of the two probes: S9 (`S9-lowers` /
  `S9-store` / `S9-relaunch` / `S9-vec`) and S13 (`S13-lowers` / `S13-face` / `S13-store` /
  `S13-relaunch`); verdicts transcribed from the change records of the respective KEEPs (probe receipts carry no job id).
- **Readings for gotcha 16**: a `stall_barrier` census on the same KEEP tree (form with drain before the rendezvous 2.22, drain after the rendezvous 1.81).
- **Certification probes for gotcha 19 (two)**:
  - Probe 1 — **status: ✅ GPU-certified**,
    `PROBE PASS max_diff=0.0` (certification round 1): `N0-lowers` (8 subprocesses compile and launch), `N1-{plain,split}-{M,N}` each 0.0
    (the explicit `StMatrix8x8x16bOp(·, 2)` path is bit-equal, cells outside the sub-block keep their sentinel), `N2-footprint` (`STSM > 0`, `STS.U16 == 0`, other STS no more than the helper path);
    fact rows: all 8 cells (2 frames × 2 majors × helper / explicit) `rc=0` and bit-equal, (STSM, STS.U16, other STS) all (2, 0, 3), the helper's r2s atom is
    `StMatrix8x8x16b … number of matrices = 2` (transposed for M-major).
  - Probe 2 (geometry taken from a specific problem shape) — **status: ✅ GPU-certified**, `PROBE PASS max_diff=0.0`
    (certification round 2; round 1 paired x1 with `n = 4`, i.e. the misplacement described above): five block partitions bit-exact (two gmem column tails, guard rows not written), `N6` with a block width not a multiple of 8: both candidates
    hit an illegal memory access at run time, `N8` x1 + `n = 4` compiles and runs, `max_diff = 1831`; helper fact rows: see gotcha 19.
- **Certification probe for the two wait forms of gotcha 5** — **status: ✅ GPU-certified**,
  `PROBE PASS max_diff=0.0` (certification round 1, all six contracts passed in the same round): four numeric forms (write-completion baseline / exit after waiting only for read completion /
  overwrite the face as soon as it has been read / per-block commit with per-block release), 6 payload groups each, bit-equal, poison and sentinel counts 0; negative control (releasing the face before the wait): 11691 poisoned cells; `W-ptx` —
  the write-completion form has only `wait_group 0`, the read-completion form only `wait_group.read 0`; `W-sass` — the write-completion form has exactly one extra instruction over the read-completion form, `CCTL.IVALL`.
