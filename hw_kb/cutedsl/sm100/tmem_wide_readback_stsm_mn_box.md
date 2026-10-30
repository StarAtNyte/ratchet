# tmem_wide_readback_stsm_mn_box (TMEM accumulator readback → 16-bit → a warp-private C staging box **contiguous along the TMEM lane direction**, written with a matrix atom instead of element-wise STS)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **warp-private box** variant of gotcha 6 / gotcha 13 (the M-major row) of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md).
> There the staging face is shared by 4 warps, one warp issues the store, with a named barrier in between;
> here each warp has its own small box (`[32 lanes × 16 columns]`), writes it, fences it and issues the TMA store itself, **with no cross-warp rendezvous**.
> The hardware constraints of the TMEM → registers → smem stretch are the same in both forms; this entry only pins down the spelling that runs on the small box.

## Purpose

The accumulator in TMEM is `lane × column` (lane = the MMA's M). When the output C in gmem has **M as its contiguous dimension** (swap-AB / the MMA's M
lands on the output's contiguous dimension), the staging box must be contiguous along lanes (`(32 lane, 16 col):(1, 32)`). Then the **shape** of the register fragment
decides the instruction width of the smem write:

| t2r atom | What each thread gets | Widest write usable on a lane-contiguous box | Write instructions per warp per 1 KiB box |
|---|---|---|---|
| `tcgen05.Ld32x32bOp(Repetition.x16)` | thread = one lane, 16 **columns** (the 16 elements of one box row, 64 B apart from each other) | only element-wise 16-bit `st.shared` (`STS.U16`); one 128-bit write needs the values of 8 **adjacent lanes**, and they sit in 8 different threads | 16 STS |
| `tcgen05.Ld16x256bOp(Repetition.x2)` | m16n8 C fragment shape: two lanes 8 apart × two adjacent columns × several column groups | `stmatrix ... .trans` (`StMatrix8x8x16bOp(transpose=True, 4)`): one instruction writes 4 8×8 matrices, each matrix row = 8 adjacent lanes = 16 B | 2 STSM |

(The last column matches the static SASS counts of the certification probe: element-wise baseline 64 `STS.U16` = 4 blocks per warp × 16; stmatrix variant 16
`STSM.16.MT88.4` = 4 blocks × 2 × two compile-time branches, see gotcha 4.)

So "replacing the epilogue's element-wise STS with a 128-bit / matrix atom" **is not a matter of swapping one store instruction; the TMEM readout atom has to change first**.
The r2s tiled copy is derived from t2r (`make_tiled_copy_D(stsm_atom, t2r)`), so the thread-value bijection of the two stretches is the same.

## When needed (trigger)

> **This entry certifies only the spelling and the values, not performance, and there is already one opposite reading.** In a same-job A/B on a complete GEMM kernel (same mechanism:
> `Ld16x256b` + transposed stmatrix writing warp-private SW64 boxes; cold L2, device median, all three variants bit-exact), the stmatrix version
> was **slower** than the element-wise `STS.U16` version: +95.8% without readout prefetch, +36.3% with the next block's readout prefetched (readings in "Sources"). Where the slowdown comes from was not located.
> So a high `scalar_sts_in_loop` and many `STS.U16` are **not by themselves a reason to change the atom**; before changing, run a same-job A/B on your own kernel, and land it only if the A/B is faster.

- You have already decided to remove the element-wise writes (e.g. a same-job A/B shows the epilogue write side really is on the critical path), the epilogue readout is `Ld32x32b`,
  and the box's contiguous dimension is the TMEM lane;
- You want to write `STS.128` but find no 8 lane-adjacent values in registers to assemble — that is the first row of the table above: **change the readout atom**, do not go and do a shuffle transpose.

Not applicable: the box's contiguous dimension is the TMEM **column** (N-major C) — then the 16 columns per thread of `Ld32x32b` are already contiguous, and a plain 128-bit
store suffices (see the staging-face spelling of gotcha 15 of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md)).

## Minimal runnable example

8 warps; warp `w` reads TMEM lane quadrant `w % 4` and column half `w // 4`; each warp has two private 1 KiB boxes, double-buffered.
Only the lines that differ from the element-wise version are listed (TMEM allocation: see gotcha 9; `tma_partition` and bulk-group accounting are the same as the element-wise version). The spelling matches the certification probe line for line.

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05, warp

F16, F32 = cutlass.Float16, cutlass.Float32
BOX_N, BOX_T, STG = 32, 16, 32 * 16                   # box = 32 lane x 16 col, lane-contiguous, 1 KiB
NUM_WARPS = 8; NBUF = 2 * NUM_WARPS

# ---- host (@cute.jit): the box swizzle and the TMA descriptor come from the same composed layout ----
box_sw = cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0,          # 64 B swizzle: each box row is 64 B
                                   cute.make_layout((BOX_N, BOX_T), stride=(1, BOX_N)))
tma_c, tma_c_tensor = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileS2GOp(), c_mn, box_sw, (BOX_N, BOX_T))      # c_mn: C in the MMA frame (lane, col), lane-contiguous
# box_sw is also passed in as a kernel argument (a static layout cannot enter the kernel through a global)

# ---- kernel ----
sC = smem.allocate_tensor(element_type=F16,
                          layout=cute.make_layout((BOX_N, BOX_T, NBUF), stride=(1, BOX_N, STG)),
                          byte_alignment=1024, swizzle=box_sw.inner)       # swizzle attached to the pointer
quad, half = warp_idx % 4, warp_idx // 4
et = quad * 32 + lane                                    # t2r / r2s both sliced in the 128-thread frame
tiles = cute.flat_divide(plane, (128, BOX_T))            # plane = TMEM (128 lane, cols)
t2r = tcgen05.make_tmem_copy(
    cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), F32),  # not Ld32x32b
    tiles[(None, None, 0, 0)])
thr = t2r.get_slice(et)
tT = thr.partition_S(tiles)
rA = cute.make_rmem_tensor(thr.partition_D(cute.make_identity_tensor((128, BOX_T))).shape, F32)
rH = cute.make_rmem_tensor(rA.shape, F16)

r2s = cute.make_tiled_copy_D(                            # D side inherits t2r's thread-value layout
    cute.make_copy_atom(warp.StMatrix8x8x16bOp(True, 4), F16), t2r)
thr_r = r2s.get_slice(et)
# 128-lane frame -> 8 private boxes: lane = (lane within quadrant, quad), warp w's box = 2*w + b
frame = cute.make_tensor(sC.iterator, cute.make_layout(
    ((BOX_N, 4), BOX_T, 2, 2), stride=((1, 2 * STG), BOX_N, 8 * STG, STG)))
tRS00 = thr_r.partition_D(frame[(None, None, 0, 0)])    # all four (column half, buffer) slices cut at compile time
tRS01 = thr_r.partition_D(frame[(None, None, 0, 1)])
tRS10 = thr_r.partition_D(frame[(None, None, 1, 0)])
tRS11 = thr_r.partition_D(frame[(None, None, 1, 1)])

gC = cute.local_tile(tma_c_tensor, (BOX_N, BOX_T), (None, None))
bSG_sC, bSG_gC = cpasync.tma_partition(tma_c, 0, cute.make_layout(1),
                                       cute.group_modes(sC, 0, 2), cute.group_modes(gC, 0, 2))

for i in cutlass.range_constexpr(CHUNKS):                # column chunks per warp
    cute.copy(t2r, tT[(None, None, None, 0, half * CHUNKS + i)], rA)        # TMEM -> registers
    rH.store(rA.load().to(F16))
    sb = 2 * warp_idx + (i % 2)
    cute.arch.cp_async_bulk_wait_group(1, read=True)    # the store from two blocks ago has finished reading this box
    cute.arch.sync_warp()
    if cutlass.const_expr(i % 2 == 0):                   # buffer index: compile time
        if half == 0:                                    # column half: a warp-uniform runtime branch picks a compile-time slice
            cute.copy(r2s, r2s.retile(rH), tRS00)        # 2 x STSM.16.MT88.4
        else:
            cute.copy(r2s, r2s.retile(rH), tRS10)
    else:
        if half == 0:
            cute.copy(r2s, r2s.retile(rH), tRS01)
        else:
            cute.copy(r2s, r2s.retile(rH), tRS11)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_warp()
    cute.copy(tma_c, bSG_sC[(None, sb)], bSG_gC[(None, quad, half * CHUNKS + i)])
    cute.arch.cp_async_bulk_commit_group()
cute.arch.cp_async_bulk_wait_group(0, read=False)
cute.arch.fence_view_async_tmem_load()                   # before releasing the accumulator / dealloc
```

## Gotchas

1. **An `Ld32x32b` fragment has no wide write on a lane-contiguous box** (the general form of the same criterion: gotcha 1 of
   [`smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`](smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128.md)).
   Thread = one lane, and a 16 B segment of the box is 8 adjacent lanes — in 8 different threads, so no single-thread 128-bit write can be assembled;
   it can only fall back to element-wise `STS.U16` (it compiles and the values are all right; the probe's baseline variant is this form). This is derived from data ownership;
   the probe did not compile each of `.align(16)` / `autovec_copy` / 128-bit `CopyUniversalOp`. To get a wide write, **first change the readout atom to `Ld16x256b`**.
2. **r2s must be derived from t2r**: `make_tiled_copy_D(StMatrix8x8x16bOp(True, n), t2r)`, and on the register side always `r2s.retile(rH)`.
   Hand-writing a stmatrix tiled copy and then matching registers to it does not guarantee the bijections agree (the error of the `make_tiled_copy_S` form: see
   gotcha 2 of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md)).
3. **`transpose=True` is because the box's contiguous dimension is the lane (the fragment's row), not the column.** Acceptance needs a payload whose coordinates can be read back (see the payload design of the probe in Sources),
   not symmetric or constant data — a symmetric payload cannot reveal a transpose error inside an 8×8 block. (The `transpose=False` variant was not run by the probe.)
4. **The offset of each warp's private box comes in through slicing, not runtime pointer arithmetic.** In `frame` the warp's quadrant offset is brought in by **thread slicing** (the second-level
   stride `2*STG` of the lane mode); the column half `half` and the double buffer `b` are compile-time slices (the four `tRS`), selected at runtime by a warp-uniform `if` — that is the certified spelling.
   A runtime whole-block offset like `sC.iterator + runtime * STG` was not tested by this probe; a spelling of the same family was rejected by IR verification for insufficient alignment on the 128-bit write side
   (see gotcha 10 of [`consumer_warpgroup_pingpong_parity`](consumer_warpgroup_pingpong_parity.md)). On the TMA store side,
   `bSG_sC[(None, sb)]` with a runtime `sb` is certified to work.
5. **With swizzle, the box and the descriptor use the same composed layout.** A box row is 64 B ⇒ 64 B swizzle (`Swizzle<2,4,3>`, acting on byte addresses,
   attached to the pointer); `make_tiled_tma_atom` takes the same composed layout, and `tma_partition` reads the tensor that stmatrix wrote.
   The r2s `frame` and the TMA-side `(32,16,NBUF)` view map the same coordinate to the same **linear** offset, and the swizzle acts on the address, so the two agree
   (this is not the same thing as the "second view with permuted modes" of gotcha 12 of [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md)).
   The box base is allocated 1024 B aligned (the probe's spelling), so that the swizzle phase at the start of each 1 KiB box matches what the descriptor sees.
6. **Why swizzle (computed from addresses; the probe did not measure banks)**: in the transposed form the 8 rows of one 8×8 matrix are 8 columns of the box,
   64 B apart, which modulo 128 B fall on only 2 16 B slots ⇒ a 4-way conflict per matrix; the 64 B swizzle XORs the 16 B block index of row r with `(r>>1)&3`,
   so the 8 rows fall on 8 different 16 B slots. Numerically both boxes are correct (probe W2 / W3); conflicts must be checked with `bankconf_st`.
7. **Bulk-group accounting as usual**: each warp does its own `commit_group`, `wait_group(1, read=True)` before writing a box (double buffer), and
   `wait_group(0)` before exit / release. `cute.copy(tma_c, …)` is called by the whole warp, not wrapped in `elect_one`.
8. **Acceptance looks at the instruction family, not only ms**: `STSM.16.MT88.4` appears in SASS and `STS.U16` drops to zero (how probe contract W4 reads it). **Do not require every `STS*` to be zero**:
   setup such as TMEM address staging and barriers leaves a few plain STS (certified reading: the SW64 stmatrix variant has 2 `STS` + 1 `STS.U8`),
   and the element-wise baseline likewise has 3 besides `STS.U16`; what to compare is `STS.U16` and "the other plain STS are no more than the baseline". A contract written as "all STS = 0" FAILs on this point
   (the third certification round was exactly this: stmatrix reading (STSM 16, STS 3), all three value contracts passed) — the contract was wrong, not the mechanism. The dossier's `scalar_sts_in_loop` should drop accordingly — that is only an instruction footprint and does not mean ms will drop (see "When needed").
9. **`TmemAllocator`'s column counter is maintained at trace time; `allocate` must not be wrapped in a runtime `if warp_idx == …:`**.
   `allocate` / `relinquish_alloc_permit` / `free` already let only the `allocator_warp_id` warp issue the instruction internally; all warps can call them directly
   (that is how the certification probe is written: `allocate` / `wait_for_alloc` / `retrieve_ptr` / `relinquish_alloc_permit` / `free` all at kernel top level).
   With `allocate(n)` in one runtime `if` block and `free` in another, `free` sees a counter of 0:
   without a column count it raises `num_columns must be between 32 and 512, and must be pow of 2, but got 0`, and with an explicit `free(ptr, n)` it raises
   `num_columns must be less than or equal to num_allocated_columns` — passing the column count explicitly **cannot** bypass the check.
   (The form with alloc and free in the same runtime `if` block was not tested by this probe.) Another certified reading and fix for the same counter problem:
   gotcha 8 of [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md).

## Sources

- Certification probe (a GPU probe, not included in this release) —
  Status: ✅ **GPU-certified**, `PROBE PASS max_diff=0.0` (fourth round of this certification). Five contracts passed in the same round: `W0-lowers` (the three variants compile and launch),
  `W1-sts-exact` (`Ld32x32b` + element-wise baseline), `W2-stsm-exact` (`Ld16x256b` + transposed stmatrix, flat box),
  `W3-sw64-exact` (same, 64 B swizzle box, launched twice in a row onto fresh NaN outputs), `W4-footprint` (both stmatrix variants have STSM > 0 in SASS,
  `STS.U16` = 0, other plain STS no more than the baseline). SASS readings (STSM, STS.U16, other STS): element-wise baseline (0, 64, 3), flat-box stmatrix (16, 0, 3),
  SW64-box stmatrix (16, 0, 3); SW64 variant in full: `LDTM.16dp256bit.x2` ×8, `STSM.16.MT88.4` ×16, `UTMASTG.2D` ×4.
  Payload `T[l, c] = l + (c%16)/16` (c < 64) / `(c−64) + (l%16)/16` (c ≥ 64), exact in fp16; the two halves together pin both the lane and the column coordinate.
  The first two rounds hit the two errors of gotcha 9 (none of the three variants compiled); the third round was the contract-spelling problem of gotcha 8.
- Opposite reading (performance, not a certification probe): a same-job A/B on a complete GEMM kernel (M=248, N=28672, K=4096, fp16), CUPTI kernel duration,
  L2 flushed before every launch, the three variants interleaved in the same job, each bit-exact twice on integer inputs: element-wise `STS.U16` 52.03 µs; `Ld16x256b` + transposed stmatrix writing
  SW64 private boxes 101.89 µs (+95.8%); same plus prefetching the next block's readout 70.91 µs (+36.3%).
- Certification coverage: only fp16, a lane-contiguous `[32 lanes × 16 columns]` box, 8 warps / 1 CTA, TMEM 128 lanes × 128 columns; the stmatrix variant takes the first of
  (Repetition, num_matrices) = (2, 4) → (1, 4) → (2, 2) that lowers, and actually got (2, 4); the other two were not run.
  No MMA mainloop (TMEM filled with the payload by `St32x32b`), so the accumulator FULL / EMPTY handshake is not involved.
- Public interfaces: `cutlass.cute.nvgpu.tcgen05.make_tmem_copy` / `Ld16x256bOp` / `Ld32x32bOp` / `Repetition`;
  `cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp`; `cute.make_tiled_copy_D` / `TiledCopy.retile` / `make_composed_layout` /
  `make_swizzle`; `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom` / `CopyBulkTensorTileS2GOp` / `tma_partition`;
  `cutlass.utils.TmemAllocator` (`allocate` / `wait_for_alloc` / `retrieve_ptr` / `relinquish_alloc_permit` / `free`);
  `cute.arch.fence_proxy` / `cp_async_bulk_commit_group` / `cp_async_bulk_wait_group` / `fence_view_async_tmem_load`.
- Official documentation: PTX ISA `tcgen05.ld` (lane–column assignment of `.16x256b` / `.32x32b`), `stmatrix` (`.trans`, `.x4`),
  `cp.async.bulk.tensor` (swizzle modes) — https://docs.nvidia.com/cuda/parallel-thread-execution/ ;
  CUTLASS CuTe DSL documentation https://docs.nvidia.com/cutlass/latest/ .
