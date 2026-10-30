# wgmma_ss_min_gemm (bf16 WGMMA SS minimal GEMM: no TMA/mbarrier, including swizzled smem layout construction)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is the
> **decoupled form** of [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md): the WGMMA SS GEMM is pulled out
> of the TMA/mbarrier pipeline and made to work on its own — smem filled with ordinary stores, synced
> with a CTA barrier, the GEMM fed directly from any `local_tile` view. It also gives a standalone
> construction recipe for swizzled smem layouts (covering the `smem_alloc_swizzle` need in the index).

## Purpose

Get a bf16 SS GEMM (A and B both in smem, fp32 accumulation) working on its own without introducing
TMA/mbarrier: `local_tile` gmem view → cooperative generic stores fill swizzled smem → async-proxy
fence → WGMMA → write accumulator fragments back to gmem. The point is to **decouple the "GEMM math
path" from the "data-movement pipeline"** — certify the GEMM alone first, then attach the TMA pipeline
(the two are orthogonal, each replaceable).

## When needed (trigger)

- A kernel/probe keeps getting stuck on `cute.gemm` / `partition_A/B` / fragment shapes and you want a
  minimal GEMM reproduction environment without a pipeline state machine.
- Symptom: WGMMA result all 0 / garbage with no error (most likely the async-proxy fence is missing,
  see gotcha 1); NCU `tensor_pct = 0` (WGMMA never executed successfully).
- You need to feed the GEMM directly from the operator's existing stride/`local_tile` view rather than
  first building a TMA descriptor as in the full-pipeline example.
- When you only need the smem swizzle layout construction (removing bank conflicts, meeting the
  core-matrix layout constraint of the WGMMA SS descriptor), host-side step 2 of this entry can be cited
  on its own.

## Minimal runnable example

A single kernel computes `C = A @ Bᵀ`: A `(M, K)` and B `(N, K)` both bf16 k-major (last dimension
stride-1), C `(M, N)` fp32. tile = (128, 128, 128), one (m, n) tile per CTA, blocks taken with
`local_tile`; 256 threads (2 warpgroups). One tile covers all of K (no k iteration, no stages).
GPU-certified in the target environment (see "Sources").

### 1) Host side: TiledMma (tile_m=128 → 2 warpgroups stack atoms along M)

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum

TILE_M, TILE_N, TILE_K = 128, 128, 128
NUM_WG = TILE_M // 64          # the WGMMA atom M-mode is always 64 -> tile_m=128 needs 2 warpgroups
NUM_THREADS = NUM_WG * 128     # 256

tiled_mma = sm90_utils.make_trivial_tiled_mma(
    cutlass.BFloat16, cutlass.BFloat16,
    warpgroup.OperandMajorMode.K,   # A k-major (row-major, last dimension contiguous)
    warpgroup.OperandMajorMode.K,   # B k-major
    cutlass.Float32,
    atom_layout_mnk=(NUM_WG, 1, 1),  # stack atoms along M, do not widen the atom
    tiler_mn=(64, TILE_N),
)
```

### 2) Host side: swizzled smem layout (i.e. the smem_alloc_swizzle recipe)

```python
# get_smem_layout_atom(layout enum, dtype, length of the major direction) picks the
# SW32/SW64/SW128 level automatically from major length x dtype width (bf16 x 128 -> 128B swizzle).
# The third argument is the element count of the major mode: pass tile_k for k-major.
smem_atom = warpgroup.make_smem_layout_atom(
    sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, TILE_K),
    cutlass.BFloat16)
# Non-staged (single buffer): tile_to_shape straight to the whole tile; k-major uses order=(0, 1).
# (The staged version appends a stage mode and uses order=(0, 1, 2), see the wgmma_tma_mbarrier entry.)
sA_layout = cute.tile_to_shape(smem_atom, (TILE_M, TILE_K), order=(0, 1))
sB_layout = cute.tile_to_shape(smem_atom, (TILE_N, TILE_K), order=(0, 1))

@cute.struct
class SharedStorage:                 # 1024B aligned; without TMA/mbarrier no barrier array is needed
    sA: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sA_layout)], 1024]
    sB: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, cute.cosize(sB_layout)], 1024]
```

### 3) Kernel side: fill smem → fence → WGMMA → write back

```python
@cute.kernel
def kernel(mA, mB, mC, tiled_mma, sA_layout, sB_layout, SharedStorage):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # take blocks with local_tile from any stride view (k has only one tile -> coordinate 0)
    gA = cute.local_tile(mA, (TILE_M, TILE_K), (bx, 0))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (by, 0))
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (bx, by))

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # swizzled smem tensor: pass outer (shape layout) and inner (swizzle) separately
    sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
    sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)

    # --- fill smem with cooperative generic stores (simplest, probe-grade; production uses TMA/cp.async) ---
    # Address with explicit 2-D logical coordinates: the swizzle lives inside the layout, the code only sees logical coordinates (gotcha 6).
    for i in cutlass.range_constexpr((TILE_M * TILE_K) // NUM_THREADS):
        e = i * NUM_THREADS + tidx
        sA[(e % TILE_M, e // TILE_M)] = gA[(e % TILE_M, e // TILE_M)]
    for i in cutlass.range_constexpr((TILE_N * TILE_K) // NUM_THREADS):
        e = i * NUM_THREADS + tidx
        sB[(e % TILE_N, e // TILE_N)] = gB[(e % TILE_N, e // TILE_N)]

    # --- key: between generic stores -> WGMMA (async proxy) there must be a fence + barrier (gotcha 1) ---
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    # --- SS fragments: A/B sliced by warpgroup slot; C sliced by thread (gotcha 3) ---
    warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
    wg_layout = cute.make_layout(NUM_WG, stride=128)
    wg_mma = tiled_mma.get_slice(wg_layout(warp_group_idx))
    tSrA = wg_mma.make_fragment_A(wg_mma.partition_A(sA))
    tSrB = wg_mma.make_fragment_B(wg_mma.partition_B(sB))
    acc = cute.make_fragment(
        wg_mma.partition_shape_C((TILE_M, TILE_N)), cutlass.Float32)

    # --- WGMMA: create a new mma_atom to change ACCUMULATE (gotcha 2); issue per k-block (gotcha 4) ---
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, False)      # first k-block does the zero-init
    for kb in cutlass.range_constexpr(cute.size(tSrA.shape[2])):
        cute.gemm(mma_atom, acc, tSrA[None, None, kb], tSrB[None, None, kb], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)

    # --- write back: C fragments sliced by thread, element by element straight to gmem (probe-grade; efficient path: see the epilogue entry) ---
    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC)
    for i in cutlass.range_constexpr(cute.size(acc)):
        tCgC[i] = acc[i]
```

launch: `kernel(...).launch(grid=[M//TILE_M, N//TILE_N, 1], block=[NUM_THREADS, 1, 1])`;
smem is accounted automatically by `cute.struct` (no manual setting needed even above 48KB).

## Gotchas

1. **After generic (non-TMA) stores write smem, and before WGMMA reads it, there must be
   `cute.arch.fence_view_async_shared()` + a barrier**. WGMMA reads smem through the async proxy, and
   generic-proxy stores are not automatically visible to it; the symptom of a missing fence is **no
   error but a result of 0/garbage** (silent error). When the writer is the whole CTA use
   `cute.arch.sync_threads()` as the barrier (`sync_warp()` only within a warp). Filling smem with TMA
   does not need this fence (TMA itself is in the async proxy).
2. **Do not `set(ACCUMULATE, ...)` directly on tiled_mma inside a loop** — compilation fails with
   `operand #0 does not dominate this use`. First create a new atom with
   `cute.make_mma_atom(tiled_mma.op)`, set on the atom, and pass the atom (not tiled_mma) to `cute.gemm`.
   Zero-init idiom: `ACCUMULATE=False` before the first k-block, True afterwards; this saves `acc.fill(0)`.
3. **Slice slots of fragments**: SS operands (A/B smem descriptors) are sliced by the **warpgroup's first
   thread** — `tiled_mma.get_slice(cute.make_layout(NUM_WG, stride=128)(warp_group_idx))`; the C
   accumulator / write-back is sliced by **thread**, `get_slice(tidx)`. The atom M-mode is always 64;
   tile_m=128 stacks atoms along M via `atom_layout_mnk=(2,1,1)` (2 warpgroups, 256 threads) (same as
   wgmma_tma_mbarrier gotcha 1).
4. **The atom K-mode of bf16 k-major = 16**: TILE_K=128 contains
   `cute.size(tSrA.shape[2])` = 8 k-blocks; `cutlass.range_constexpr` issues `cute.gemm` block by block;
   without stages the fragment index is `[None, None, kb]` (one dimension fewer than the staged version).
5. **The swizzle level is determined by the third argument of `get_smem_layout_atom` (element count of
   the major direction)**: bf16×128 → 128B swizzle. Do not hand-write a flat layout (e.g.
   `(128,128):(128,1)`) and feed it to WGMMA directly — the SS descriptor requires core-matrix structure,
   and an illegal layout reports a shape error at `make_fragment_*`/`cute.gemm` or silently computes wrong.
6. **Always address swizzled smem with logical coordinates** (the swizzle is encapsulated in the
   layout). Note that after `tile_to_shape` the shape is hierarchical (e.g. ((8,16),(64,2))); when
   linearising indices by hand it is easy to get the colex order wrong — explicit 2-D coordinates are the
   safest for cooperative fills.
7. **Order of the WGMMA trio**: `warpgroup.fence()` before writing acc; `commit_group()` after issuing a
   batch; `wait_group(0)` before consuming acc (write-back) (same as wgmma_tma_mbarrier gotcha 9).
8. **The file must not contain `from __future__ import annotations`** (same as wgmma_tma_mbarrier
   gotcha 13: it turns the `@cute.struct`/kernel annotations into strings and `cute.compile` raises a
   TypeError).

## Sources

- Public infra module (quack `sm90_utils.py`):
  - `make_smem_layout` L14-38 (non-staged: `tile_to_shape(atom, shape, order=(0,1))`)
  - `gemm` L149-173 (the dominance gotcha of creating a new mma_atom to change ACCUMULATE,
    fence/commit/wait order, per-k-block loop)
  - `partition_fragment_ABC` L217-245 (`partition_shape_C` builds acc,
    `make_fragment_A(partition_A(sA))` SS form)
- GPU-certified by a GPU probe (the probe is not included in this release; 2×2 CTA tile grid,
  bf16 integer inputs vs an exact fp32 reference, NaN prefill to catch uncovered coordinates).
- Official reference: cutlass examples `examples/python/CuTeDSL/hopper/dense_gemm.py` (same API surface).
