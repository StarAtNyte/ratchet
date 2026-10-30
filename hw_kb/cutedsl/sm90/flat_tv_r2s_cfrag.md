# flat_tv_r2s_cfrag (conflict-free 128-bit flat-TV R2S of the WGMMA C-frag)

> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose

Write one WGMMA FP32 C accumulator fragment (`acc`, spread over the registers of the warpgroup's 128
threads) into smem in one go, with **one 128-bit STS per thread (4 consecutive fp32)** and **the 32 STS
of a warp hitting 32 different banks → zero bank conflicts**; then read it back from smem into
registers through **the same byte-order bijection**. The core idea: give up "partition smem in the
WGMMA C-frag's native (m,d) shape" and instead use a **flat (slot_size, num_wg) smem layout + a flat TV
copy**, so the thread↔byte mapping depends only on the linear offset and not on WGMMA's quad-pair
interleave. As long as the write side (K-writer) and the read side (K-reader) instantiate **exactly the
same** TV bijection, the round trip is the identity.

## When needed (trigger)

- NCU reports high `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` (when storing a WGMMA C-frag
  to smem). Typical symptom: writing a row-major `(slot_m, slot_d)` smem buffer with
  `make_tiled_copy_C(atom, tiled_mma)` (WGMMA-frag native partition), NCU shows ~50% of shared stores
  taking N-way conflicts (in one observed case: 11.6-way, 51.8% of 12.1M STS).
- Degenerate fallback: one 32-bit STS per cell (per-cell atom). This splits one 128-bit transaction
  into 4 32-bit ones — 1/4 throughput and even heavier conflicts (in the same case ~72.7M residual
  shared-store bank conflicts).
- Dataflow shape: one warpgroup's C-frag must land in smem and then be read back by another piece of
  code (another warp of the same kernel, or a later postprocess kernel) — both ends must share the
  same layout.

## Minimal runnable example

The mechanism's three pieces: **flat smem layout** + **flat TV copy (write)** + **same-layout retile
(read)**. No specific operator algorithm embedded; it only demonstrates the "C-frag ↔ smem" round trip.

```python
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils_basic

# --- shape parameters (operator-agnostic) -----------------------------------
# TILE_M, D: the C-frag's logical (m, d); NUM_WG: number of participating warpgroups (>=1).
# Constraints: (TILE_M * D) % NUM_WG == 0, and slot_size % 4 == 0 (4 fp32 per thread).
# *** INV(WGMMA atom M-mode=64): the M-mode of the WGMMA atom of one warpgroup (128 threads,
#     NUM_WG=1) must be 64. So a single WG covers 64 rows of C-frag (tiler_mn=(64,D),
#     atom_layout=(1,1,1)). A 128-row tile needs atom_layout=(2,1,1), i.e. 2
#     warpgroups = 256 threads; a 128-wide atom cannot be used (it triggers OpError: expects the
#     M-mode to be 64, but got 128). This example takes the single-WG representative shape TILE_M=64. ***
TILE_M, D, NUM_WG = 64, 64, 1
WG_THREADS = 128                       # one warpgroup = 128 threads
slot_size  = TILE_M * D // NUM_WG      # fp32 count per WG slot

# --- (1) flat smem layout: (slot_size, num_wg[, stage]) ---------------------
# Column-major: stride=(1, slot_size[, slot_size*num_wg]). One contiguous slot per wg.
# Key: it does **not** carry the WGMMA frag's quad interleave; purely linear -> the STS offset is set by TV alone.
sdQacc_layout = cute.make_layout((slot_size, NUM_WG), stride=(1, slot_size))

# --- (2) flat TV copy: make_tiled_copy_tv((128, num_wg), val=4) -------------
# 128-bit universal copy atom; thread-layout=(128, num_wg); val-layout=4 fp32.
# => each thread gets 4 **consecutive** fp32 = one 128-bit STS; the start offsets of the
#    32 threads of a warp differ by 4 fp32 = 16 bytes = one bank boundary -> 32 different banks, 0 conflicts.
ELEMS_128 = 128 // Float32.width        # = 4
r2s_tiled_copy = cute.make_tiled_copy_tv(
    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32,
                        num_bits_per_copy=128),
    cute.make_layout((WG_THREADS, NUM_WG)),   # thread layout
    cute.make_layout(ELEMS_128),              # value layout = 4 contiguous fp32
)

@cute.kernel
def _r2s_roundtrip(mOut: cute.Tensor,
                   sdQacc_layout: cute.Layout, tiled_mma: cute.TiledMma):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    sdQacc = smem.allocate_tensor(Float32, sdQacc_layout, byte_alignment=1024)

    # ---- build a WGMMA C-frag (in a real kernel it comes from WGMMA; in the verification probe it
    #      is filled in registers with "each element's own logical (m,d) coordinate", no gmem read
    #      -> the frag shape is fully static) ----
    # *** gotcha (d): autovec_copy requires src/dst to be **statically shaped**. Loading from a gmem
    #     partition_C brings problem-size dynamic modes into acc -> autovec_copy raises
    #     "expects source and destination tensors to be statically shaped".
    #     Fix: use partition_C of make_identity_tensor((TILE_M, D)) to get the static (m,d)
    #     coordinate of each of this thread's registers, fill the ramp directly in registers,
    #     and drop the gmem read entirely. ***
    thr_mma = tiled_mma.get_slice(tidx)
    cId = cute.make_identity_tensor((TILE_M, D))         # static coordinates
    tCcId = thr_mma.partition_C(cId)                     # static C-frag coordinates
    acc = cute.make_fragment(tCcId.shape, Float32)       # WGMMA partition_C registers
    for i in cutlass.range_constexpr(cute.size(tCcId)):
        m, d = tCcId[i]
        acc[i] = Float32(m * D + d)                      # acc[m,d] = m*D+d

    # ---- write: C-frag -> flat smem (one 128-bit STS per thread) -------------
    thr = r2s_tiled_copy.get_slice(tidx)
    tdQ_smem = thr.partition_D(sdQacc[None, None, 0] if sdQacc.layout.rank == 3
                               else sdQacc)
    # *** gotcha (a): do not .retile the flat layout into a C-frag — use make_fragment_like +
    #     a linear reinterpret instead. acc.iterator is the same register file; just put the TV shape on it. ***
    tdQ_layout = cute.make_layout(tdQ_smem.shape)
    tdQ_rmem   = cute.make_tensor(acc.iterator, tdQ_layout)   # linear reinterpret
    cute.autovec_copy(tdQ_rmem, tdQ_smem)                     # 128-bit STS
    cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                          space=cute.arch.SharedSpace.shared_cta)
    cute.arch.barrier()                                       # read only after the write is visible

    # ---- read: flat smem -> registers (the **same** TV bijection) ------------
    s2r_thr = r2s_tiled_copy.get_slice(tidx)                  # same TV -> same byte order
    tdQ_smem_r = s2r_thr.partition_S(
        sdQacc[None, None, 0] if sdQacc.layout.rank == 3 else sdQacc)
    rdQ = cute.make_fragment_like(tdQ_smem_r, Float32)
    cute.autovec_copy(tdQ_smem_r, rdQ)                        # 128-bit LDS
    # rdQ now holds the C-frag values in TV order; reinterpret back to acc.layout (the write side's
    # m,d view), then scatter back to mOut at the **same** logical (m,d) coordinates used for the fill,
    # to check. Dynamic gmem only gets per-coordinate scalar stores and is never fed to autovec_copy (see gotcha d).
    acc_back = cute.make_tensor(rdQ.iterator, acc.layout)
    for i in cutlass.range_constexpr(cute.size(tCcId)):
        m, d = tCcId[i]
        mOut[m, d] = acc_back[i]                              # mOut[m,d] == m*D+d
```

Key points (why it is conflict-free):
- **Flat layout** (linear stride, no quad interleave) + **val=4 consecutive fp32**: the STS of thread
  `t` in a warp lands at fp32 offsets `4*t … 4*t+3`; adjacent threads are 16 bytes apart, exactly
  covering 32 banks → 0 conflicts, and it is a true 128-bit transaction (not 4×32-bit).
- **Same TV for write and read**: `make_tiled_copy_tv((128, num_wg), val=4)` is instantiated at both
  ends with **the same object / the same parameters** → the thread↔byte bijection is byte-for-byte
  identical → the round trip is the identity.

## Gotchas

- **(0) The WGMMA atom's M-mode must be 64.** If `tiled_mma` is built from a single atom with
  `tiler_mn=(128, D)`, the atom's M-mode becomes 128 → `make_trivial_tiled_mma` raises
  `OpError: expects the M-mode to be 64, but got 128` in `warpgroup.mma __post_init__`.
  A tile with M=128 is achieved by **stacking 2 atoms along M in atom_layout**
  (`atom_layout_mnk=(2,1,1)`, i.e. 2 warpgroups = 256 threads), **not** by a 128-wide atom. A single
  warpgroup (128 threads, NUM_WG=1) can only cover **64 rows** of C-frag: `atom_layout=(1,1,1)` +
  `tiler_mn=(64, D)`. This mechanism's R2S↔read bijection does not depend on the atom's M-mode
  (STS count per thread = slot_size/128, the atom repeats automatically), so the verification probe
  can take the single-WG TILE_M=64.
- **(a) `.retile` of the flat layout → C-frag rank error.** The flat `(slot_size,num_wg)` and the
  WGMMA C-frag `(MMA, MMA_M, MMA_N)` differ in rank/nesting; `thr_copy.retile(acc)` errors on the shape
  mismatch or silently misplaces data. Correct approach:
  `cute.make_tensor(acc.iterator, cute.make_layout(tdQ_smem.shape))` — put the TV's flat shape on the
  same register file (`acc.iterator`) as a **linear reinterpret**; when a dtype conversion is needed,
  pair it with `cute.make_fragment_like`.
- **(b) Variable scope in CuTe DSL dynamic control flow.** The write side is often wrapped in an
  `if const_expr(...)` / a dynamic `mbarrier_wait` branch (e.g. selecting `sdQacc[None,None,dq_stage]`
  by stage). Loop/branch invariants such as `tdQ_smem` / `tdQ_layout` **must be hoisted before the
  if/loop**; binding them for the first time inside a branch makes the DSL lose track of them, or gives
  each branch a different instance → the round-trip bijection is inconsistent. The observed fix hoisted
  `tdQsdQaccum_const` / `tdQ_layout_const` out of the main loop and only re-selected the stage slice
  inside the branch.
- **(c) K-write and K-read must be the same bijection, otherwise the byte order is scrambled.** If
  either end uses a different thread-layout, a different val count, or one end uses
  `make_tiled_copy_C(atom, mma)` (WGMMA native) while the other uses `make_tiled_copy_tv(...)` (flat),
  the byte order is scrambled and the values are silently wrong (no error). Every later reader (even a
  separate postprocess kernel) must use **exactly the same** `make_tiled_copy_tv((128, num_wg), val=4)`
  + the same `(slot_size, num_wg)` layout. **Verify with a GPU round-trip probe** (known values → R2S →
  read back → compare against numpy); the bijection holds only if `max_diff==0`.
- **(d) `autovec_copy` requires both src and dst to be statically shaped.** If the verification probe
  loads `acc` from the gmem `partition_C(mIn)`, the problem-size **dynamic modes** are brought into the
  frag and `cute.autovec_copy(tCgIn, acc)` throws
  `ValueError: autovec_copy expects source and destination tensors to be statically shaped`.
  This bijection does **not** need real input: use `partition_C` of
  `make_identity_tensor((TILE_M, D))` to get the **static** `(m, d)` coordinate of each of this
  thread's registers, fill the ramp directly in registers (`acc[m,d]=m*D+d`), and drop the gmem read
  entirely; after reading back, store to `mOut` **element by element with scalar stores** at the same
  static coordinates. Result: every tensor fed to `autovec_copy` (`acc` / the flat TV register view /
  the smem partition / `rdQ`) has a compile-time static shape, and the only dynamic one, `mOut`, only
  gets per-coordinate stores and never enters `autovec_copy`.

## Sources

- GPU-certified: round-trip probe (known ramp → flat-TV R2S → read back through the same bijection →
  compare against the numpy reference, `max_diff==0`); the probe shape is the one described in gotchas
  (c)/(d); status in `index.md` (✅ Verified).
- Public infra: CuTe DSL `make_tiled_copy_tv` / `autovec_copy` (`cutlass/cute/core.py`).
