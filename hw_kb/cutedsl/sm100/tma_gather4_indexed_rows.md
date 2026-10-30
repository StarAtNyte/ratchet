# tma_gather4_indexed_rows (tensor-map TMA `.tile::gather4`: one instruction moves 4 arbitrary indexed rows into a swizzled operand face)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; inline PTX ([`ptx_escape`](ptx_escape.md) form (2)). CuTe DSL 4.6.x's
> `cutlass.cute.nvgpu.cpasync` has **no** copy op for this instruction (gotcha 9).
>
> | | |
> |---|---|
> | **Knob** | `cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes[.cta_group::2] [dst], [tmap, {c0, r0, r1, r2, r3}], [mbar];` — one instruction moves **4 arbitrary rows** (`r0..r3` are four unrelated row coordinates, `c0` is the column coordinate along the contiguous dimension) × one box width (64 bf16 elements = 128 B), 512 B in total. The tensor map's box is **one row**: `boxDim = (64, 1)` (gotcha 1). Each stage is issued serially by one thread with a single `expect_tx(Σ bytes)` |
> | **Footprint** | Same byte count, **1/4** of the instructions of the "one box per row" form (the geometry certified here: 768 → 192 instructions / CTA / round). The dossier census axis `cache_ops` (verbatim dotted opcode) reads **`["UTMALDG.2D.GATHER4.2CTA"]`** on the pair form with `.cta_group::2` (a KEEP census from another tree, see gotcha 11); **adding `.L2::cache_hint` or not leaves this opcode unchanged** — the policy operand does not show in the census |
> | **Load-bearing condition** | Issue itself costs time, and it is single-thread serial: 192 instructions / CTA / round issued by one thread take **1.765 µs (≈ 9.2 ns / instruction)**; split across two warps, one face each, the round period goes **2.03 → 1.41 µs**; split across four warps it **gets no faster** (period ratio 0.675, worse than the 0.585 of two warps). On the same tree, swapping only "768 one-row boxes" for "192 gather4", with bytes / faces / accounting untouched: device window **7.0209 → 5.8345 ms** (same-job paired timing ratio 0.1331 → 0.1601, higher is faster) |
>
> **Certification coverage**: two mutually independent GPU probes, each byte-exact, covering both ways of obtaining the descriptor (copied out of the DSL's TMA atom; host-side encoding),
> both single-CTA and CTA-pair accounting, both K-major and MN-major operand faces, and out-of-bounds row coordinates; the multicast variant has one more byte-exact reading (gotcha 7).

## Purpose

The supply is a **row-by-row gather driven by an index table** (row numbers come from another table; the rows are not contiguous in gmem), while the destination is a **swizzled MMA operand face**
(the kind of SW128 face `make_smem_layout_a/b` produces). A plain `.tile` box of tensor-map TMA can only frame **contiguous** rows,
so without gather4 there are only two ways to write it: one 1-row box per row (`CopyBulkTensorTileG2SOp`, one instruction per 128 B row),
or one non-tensor-map 1D bulk per row ([`cp_async_bulk_1d_mbarrier_gather`](cp_async_bulk_1d_mbarrier_gather.md),
no swizzle). One `.tile::gather4` instruction lands 4 arbitrary rows into the face with the descriptor's swizzle,
and completion is recorded into the mbarrier's tx-count as usual — same bytes, a quarter of the instructions.

## When needed (trigger)

- The supply is an indirect row-by-row gather, each row is one contiguous byte run in gmem, and the destination face is a swizzled face the MMA reads directly.
- The producer issues from a single thread (the TMA convention), and the profile shows the MMA waiting on supply: the instruction count itself is the bottleneck
  (the one-box-per-row form, hundreds of instructions per round issued serially by one thread).
- You are already writing this PTX and are stuck on: how many rows the box should have / in which order the modifiers go / whose barrier the pair's bytes are recorded on /
  whether an out-of-bounds index will hang.

**Anti-trigger**: the rows are already contiguous; a plain `.tile` box (multi-row) is enough and gather4 is not needed.

## Minimal runnable example

A skeleton extracted from the two certification probes: a CTA pair (`cluster = (2,1,1)`); each CTA loads its own 48 indexed rows × 512 dims
(bf16) into its own K-major SW128 face, and every box is recorded on **the leader's** barrier. The descriptor takes the path "the DSL's TMA atom
is used only to build the descriptor, which is copied into gmem and handed to the PTX" (the other path, host-side encoding, is in gotcha 6).

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.address_space import AddressSpace
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

ATT = llvm.AsmDialect.AD_ATT
G4 = ("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
      ".mbarrier::complete_tx::bytes.cta_group::2"     # order: see gotcha 2; single CTA: drop .cta_group::2
      " [$0], [$1, {$2, $3, $4, $5, $6}], [$7];")

@dsl_user_op
def gather4(dst, tmap, c0, r0, r1, r2, r3, mbar, *, loc=None, ip=None):
    # Form (2): address operands only, side effects only. dst / mbar are Int32 smem window addresses
    # (Pointer.toint()), tmap is the descriptor's Int64 address, all five coordinates are Int32.
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip), cutlass.Int64(tmap).ir_value(loc=loc, ip=ip),
         cutlass.Int32(c0).ir_value(loc=loc, ip=ip),
         cutlass.Int32(r0).ir_value(loc=loc, ip=ip), cutlass.Int32(r1).ir_value(loc=loc, ip=ip),
         cutlass.Int32(r2).ir_value(loc=loc, ip=ip), cutlass.Int32(r3).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        G4, "r,l,r,r,r,r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# ---- host (inside @cute.jit): a TMA atom only to build the descriptor -------------------------
#   gX is given "contiguous dimension first": mode 0 = the 512 dims contiguous within a row, mode 1 = row number.
#   Then the descriptor's mode order is the PTX coordinate order {c0, r0..r3}; no reordering to think about.
#   box = 64 dims × 1 row = 128 B = exactly one 128 B swizzle row, carrying the operand face's own swizzle.
#   The atom's own cta_group is ONE (TWO is byte-exact too, see gotcha 2): pair accounting is a modifier on the **instruction**, not a descriptor property.
D, NROWS = 512, 32768
gX = cute.make_tensor(mX.iterator, cute.make_layout((D, NROWS), stride=(1, D)))
box = cute.make_composed_layout(face_layout.inner, 0,              # face_layout: make_smem_layout_b(...)
                                cute.make_layout((64, 1), stride=(1, 64)))
tma_atom = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE), gX, box, (64, 1), 1)[0]
#   mDesc: an int64 gmem buffer, one 16-word (128 B) slot per CTA, passed into the kernel.

# ---- kernel: each CTA copies the atom's descriptor into its own slot, then acquires -------------
mgr = utils.TensorMapManager(utils.TensorMapUpdateMode.GMEM, 128)
dptr = mgr.get_tensormap_ptr(mDesc.iterator + 16 * cute.arch.block_idx()[0], AddressSpace.gmem)
mgr.init_tensormap_from_atom(tma_atom, dptr, 0)          # copied by one thread in warp 0
cute.arch.barrier()
mgr.fence_tensormap_initialization()
cpasync.fence_tma_desc_acquire(dptr)

# pair accounting (gotcha 5): only the leader arms, bytes = the whole pair; the arm must be visible before the peer's first landing
TX = 2 * 48 * 512 * 2                                     # two CTAs × 48 rows × 512 dims × 2 B
if warp_idx == 0:
    if rank == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, TX)
cute.arch.barrier()
cute.arch.cluster_arrive()
cute.arch.cluster_wait()

if warp_idx == 1:
    bar = cute.arch.map_dsmem_ptr(mbar, 0).toint()        # both CTAs record on the leader's barrier
    base = cute.recast_ptr(sFace.iterator, None, dtype=cutlass.BFloat16).toint()
    tmap = dptr.toint()
    with cute.arch.elect_one():                           # raw PTX has no built-in election (gotcha 4)
        for c in cutlass.range_constexpr(NRUN):           # the face is organised in runs of 64 dims
            for b in cutlass.range_constexpr(NSLOT // 4): # every 4 consecutive slots = one gather4
                dst = base + 2 * (b * 4 * SLOT_STRIDE + c * RUN_STRIDE)   # unswizzled linear offset (gotcha 3)
                k = [mIdx[rank * NSLOT + 4 * b + u] for u in range(4)]
                gather4(dst, tmap, c * 64, k[0], k[1], k[2], k[3], bar)
```

`NSLOT` (this CTA's row count, 48), `NRUN` (512 / 64 = 8), `SLOT_STRIDE` (element distance between two adjacent rows in the face, = 64) and
`RUN_STRIDE` (element distance between adjacent runs) are all read from the face's layout at trace time — a slot must be exactly one 128 B row
(`SLOT_STRIDE == 64`) for four consecutive slots to form the contiguous 512 B destination segment of one gather4; if that relation cannot be read out, do not issue.
Single-CTA form: drop `.cta_group::2` from the instruction, use the CTA's own `mbar` as the barrier, count only its own bytes in `TX`, and no cluster rendezvous is needed.

## Gotchas

1. **The tensor map's box must be one row: `boxDim = (64, 1)`.** With the same gather4 instruction, a box of `(64, 4)` (assuming that since one instruction
   moves 4 rows the box should be 4 rows) fails at launch with `an illegal instruction was encountered`, in all three gather cells without exception;
   `(64, 1)` is byte-exact. The gather4 convention is: the box is **one row**, and each of the four row coordinates in the instruction moves it once.
   The same tensor with the same swizzle, in plain `.tile` mode with a 4-row box and contiguous row coordinates, is also byte-exact — the two modes have different box
   conventions, and one map cannot serve both modes.
2. **Modifier order and the atom's `CtaGroup` are not levers; what changes the landed bytes is the descriptor's swizzle.** A spelling sweep of five
   candidates, all compiled and all run: `.mbarrier::complete_tx::bytes.cta_group::2` (the PTX ISA's written order, used by this example)
   and the reversed `.cta_group::2.mbarrier::complete_tx::bytes` are both byte-exact; building the descriptor's TMA atom with
   `CtaGroup.ONE` or `TWO` gives the same result (pair accounting is a modifier on the **instruction**; the descriptor does not carry it); without `.cta_group::2`,
   each CTA recording its own bytes on its own barrier, is byte-exact too. **The only cell that landed wrong is a descriptor without swizzle**: the rows land in a
   linear (unswizzled) layout; the data itself is complete but does not match the SW128 face's address function (gotcha 3). So when the numbers are wrong, first check that the descriptor's
   swizzle matches the face's swizzle; do not spend rounds on modifier order.
3. **The destination address is the face's linear (unswizzled) offset; the swizzle is derived by the descriptor from the destination address.** The SW128 swizzle cycles every 8 rows,
   and the 4 rows of one gather4 cover only half a cycle block; the certification readings show that 4 rows landing in the second half-block are as exact as those in the first half —
   the hardware computes the swizzle from the **destination address**, not from the row index inside the box. So write dst as `base + linear element offset × 2`,
   and do not apply the swizzle a second time yourself.
4. **Raw PTX has no built-in election; wrap it in `cute.arch.elect_one()` yourself.** The DSL's TMA copy atom elects internally
   (see [`elect_one`](elect_one.md)); inline PTX does not: every certified form issues inside `elect_one()`. Without the wrap, all 32 lanes of the warp
   issue once each and the barrier receives 32× the bytes — this point is an inference, not tested on its own (a wrong tx in a real kernel
   shows up as a hang, see gotcha 5).
5. **Pair accounting has three requirements; missing any one is a hang or undefined behaviour:** every instruction carries `.cta_group::2` and points its barrier operand at
   **the leader's** barrier (`cute.arch.map_dsmem_ptr(mbar, 0)`); only the leader arms, with `expect_tx` = the whole pair's
   bytes; the leader's arm must be visible to the peer before the peer's first landing (CTA barrier + `cluster_arrive / cluster_wait`
   placed outside all warp branches). Violations of the third are **intermittent**; see
   [`cta_pair_bulk_load_cta_group2`](cta_pair_bulk_load_cta_group2.md) gotcha 5.
6. **The host-side descriptor encoding path** (equivalent to the example above, also byte-exact): use `cuda.bindings.driver.cuTensorMapEncodeTiled`
   to encode a rank-2 map on the host (dim 0 = contiguous dimension, dim 1 = row number; `CU_TENSOR_MAP_SWIZZLE_128B`; `boxDim = [64, 1]`;
   `elementStrides = [1, 1]`), copy the returned 128 bytes (`ctypes.string_at(int(tm.getPtr()), 128)`) into an int64
   device tensor, issue `fence.proxy.tensormap::generic.acquire.gpu [$0], 128;` on it once in the kernel, then use its address as
   `tmap`. These bindings are **typed**: `globalDim` / `globalStrides` need `cuuint64_t`, `boxDim` / `elementStrides`
   need **`cuuint32_t`**; the function has **no output parameter** and returns `(CUresult, CUtensorMap)`. Passing an extra output argument as in the C signature gives
   `TypeError: cuTensorMapEncodeTiled() takes exactly 11 positional arguments (12 given)`; passing a bare int for `boxDim` gives
   `TypeError: Cannot convert int to cuda.bindings.driver.cuuint32_t` — the two probe rounds before certification died at these two points respectively.
7. **Multicast variant** (this KB has one byte-exact reading; not in the example above): append `.multicast::cluster` to the modifiers and a 16-bit `ctaMask`
   at the end of the operands; the certified spelling declares a 16-bit register in the asm block and converts into it:
   `{ .reg .b16 m; cvt.u16.u32 m, $8; cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1, {$2, $3, $4, $5, $6}], [$7], m; }`
   (constraint string `r,l,r,r,r,r,r,r,r`). Certified form: the two CTAs each issue half the rows with `ctaMask = 0b11`, the whole face lands in both CTAs,
   and **each CTA arms the whole face's bytes on its own barrier** (not pair accounting; this cell did not carry `.cta_group::2`);
   in the same round-period measurement set it brought the single-thread form's period down to 0.551.
8. **Out-of-bounds row coordinates do not error: they land zeros and are still counted.** For a row coordinate < 0 or ≥ the row count, that row lands as 0, with no fault, and that row's
   bytes **are still** counted into the barrier's tx (likewise for an instruction with all four rows out of bounds) — `expect_tx` is still computed at "512 B per instruction". So sentinel
   indices (e.g. −1) can be issued as is, relying on the zero row plus later masking; do not change the instruction count to "skip invalid rows", which would make tx vary with each round's
   number of valid rows (see [`cp_async_bulk_1d_mbarrier_gather`](cp_async_bulk_1d_mbarrier_gather.md) gotcha 1).
9. **DSL version**: CuTe DSL 4.6.x and 4.7.x have no gather4 copy op in `cpasync` (a 4.6.x docstring mentions a
   gather4 op name, but the class does not exist), only the underlying MLIR op; from 4.8.0 there is `cpasync.CopyBulkTensor2DGather4G2SOp`
   (including `CtaGroup.TWO` and the multicast variant), used together with `make_tiled_tma_atom(..., gmem_coord_tensor=...)` — **this KB has not certified
   that DSL path**; before switching versions, check the bytes with a probe first.
10. **Issue is a single-thread serial cost, and spreading it has a limit.** See "Load-bearing condition" in the header: two warps issuing one face each bring the period down to 0.585–0.69×,
    four warps get no faster. Another certified form has 4 warps each contribute one elected thread that issues its own share of the rows (with 256-bit index loads);
    it is certified for byte correctness only, not priced. Reading the indices is itself on the serial chain: the LDS chain that reads the indices out of the smem band
    takes ≥ 15% of the single-thread period (removing it brings the period down to 0.827×); for batched index prefetch see [`gather_index_lane_owned`](gather_index_lane_owned.md).
11. **The spelling of `.L2::cache_hint` on the tensor form (KEEP-certified; a knob whose value is invisible).** The modifier goes **last**
    (after `.cta_group::2`), and the policy is the **last operand** (after `[mbar]`; the cta-mask slot before it belongs only to the
    `.multicast` form, and the non-multicast form has no such slot):

    ```python
    G4_HINT = ("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
               ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
               " [$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;")     # constraint string "r,l,r,r,r,r,r,r,l,~{memory}"
    # $8: Int64 policy descriptor, built once per issuing thread at the role entry, outside the loop, constant for the whole launch:
    #   "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;"  constraint "=l", has_side_effects=False
    #   (result-form asm, written like form (3) of cp_async_bulk_1d_mbarrier_gather; 1.0 = all accesses of the instruction use this priority)
    ```

    | | |
    |---|---|
    | **Knob** | The modifier + operand above; with and without are two independent asm strings, and only the selected one reaches ptxas |
    | **Footprint** | `cache_ops` is `["UTMALDG.2D.GATHER4.2CTA"]` before and after — **the census cannot see it**. What can be checked is the PTX text: count of `.L2::cache_hint` == count of `tile::gather4`, `createpolicy` appears once; the setting without it has neither, with the same gather4 count. The contract axis `rings` is rewritten in the same round from `stride=0x400100` to `stride=0x400100,wrap=sel`, zero barrier changes, follows along |
    | **Load-bearing condition** | In the structure "1 CTA/SM, `dram_pct` ≈ 7.8% (not bandwidth-bound), `tensor_pct` ≈ 63.6, 96 gather4 per chunk", `evict_last` reads gate **0.997 → 1.0004 ms (+0.34%, no value)**, `dram_bytes_read` 663688544 → 659009664 (−0.7%), `dram_pct` 7.88 → 7.83, `stall_long_sb` 5.2 → 5.15. These are this tree's readings: what the same policy bit is worth on a bandwidth-bound tree, this entry has no reading for |

    How to check correctness: the policy bit does not change the delivered values, so the contract is written as **the two settings, with and without, are bit-identical** (including out-of-bounds sentinel rows and the padded tail block),
    plus one sensitivity control (the same compiled artifact with a shifted index table; the output must change) — otherwise an error like "the policy operand landed in a coordinate slot",
    invisible at the opcode level, has nothing to catch it. The 64-bit descriptor takes one register pair in the issuing role: on the certified tree the issuing role stays at
    `LDL == STL == 0` at the 96-instruction setting. The `evict_normal` setting exists only as a control, with no ms reading of its own.
12. **Moving issue from a 4-warp role to 3 warps of another role (24 → 32 instructions per warp, total 96 unchanged).**
    Each issuing warp takes every third row group (`r = wp + 3g`, g = 0..7); destination addresses, `expect_tx`, pair accounting, and the
    issue order "index reads above the stage-empty wait, gather4 below it" are all unchanged; the three warps share one issue body parameterised by warp index, and the other duties
    (on this tree, bitmap generation and queue lookup) hang in the same body as warp-uniform branches, so gather4 appears only once
    in the binary and the role's register usage is the maximum over the duties, not their sum. Footprint: dossier `supply_counts`
    `copies 24` → `copies 32` (once per issue body), `addr-source loads 1` → `4`, `merge.shfl_in_loop` 24 → 32,
    `loops` `{copy-producer: 4, scalar: 25}` → `{copy-producer: 13, scalar: 13}`, `edges`
    `[[scalar, scalar]]` → `[[copy-producer, copy-producer]]`; `supply` is still
    `addr-source loads clustered-at-top; prefetched`. Readings: gate **0.9993 → 0.9844 ms (−1.49%)**,
    `tensor_pct` 63.63 → 64.35, `stall_long_sb` 5.15 → 5.47, `stall_barrier` 0.17 → 0.27, spill 0 / 0 unchanged
    — **in this structure** (1 CTA/SM, issue not on the critical path: the timeline readings on the same tree before the move were issuing role busy 0 /
    period 2375, elementwise role busy 1127 / period 2417) going from 4 to 3 issuing warps does not hurt the period; this step frees a whole
    warpgroup, and −1.49% sits at the edge of that tree's noise band, so it is not taken as the price of the issue move itself. Verify the landed bytes with
    "one stage bit-exact + sentinel rows zero + the stage after wrap-around bit-exact".

## Sources

- PTX ISA, "Data Movement and Conversion Instructions": `cp.async.bulk.tensor` (`.tile::gather4` load mode,
  `.cta_group`, `.multicast::cluster` and `ctaMask`, `.mbarrier::complete_tx::bytes`), `fence.proxy.tensormap`,
  `mbarrier.arrive.expect_tx`; CUDA Driver API `cuTensorMapEncodeTiled`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols: `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`,
  `cpasync.make_tiled_tma_atom`, `cpasync.fence_tma_desc_acquire`, `cpasync.CopyBulkTensor2DGather4G2SOp` (from 4.8.0,
  uncertified), `cutlass.utils.TensorMapManager` / `TensorMapUpdateMode.GMEM` (`get_tensormap_ptr`,
  `init_tensormap_from_atom`, `fence_tensormap_initialization`), `cute.arch.map_dsmem_ptr`, `cute.arch.elect_one`,
  `cute.arch.mbarrier_arrive_and_expect_tx`, `cute.arch.cluster_arrive / cluster_wait`,
  `cutlass.utils.blackwell_helpers.make_smem_layout_b`, `cuda.bindings.driver.cuTensorMapEncodeTiled`,
  `cutlass._mlir.dialects.llvm.inline_asm`. General rules for the inline-asm channel are in [`ptx_escape`](ptx_escape.md).
- Certification probe (host-side encoding + pair accounting + out-of-bounds rows; a GPU probe, not included in this release) —
  **Status: ✅ GPU-certified**, `PROBE PASS max_diff=0`, all five contracts passed in the same round: `S30-desc` (plain `.tile`
  4-row box, contiguous rows → a real 48-row × 512-dim SW128 face), `S30-g4` (`.tile::gather4`, `boxDim[1] = 1`, 96 instructions filling the same face,
  indices a random permutation with one duplicate value), `S30-g4v` (the same, landing in an MN-major 96-row × 128-dim face), `S30-g4pair` (two CTAs,
  `.cta_group::2`, everything recorded on the leader's barrier, only the leader arms the whole pair's bytes), `S30-g4oob` (seven out-of-bounds row coordinates,
  one instruction with all four rows out of bounds: those rows are 0, the rest exact, no fault, the barrier flips as usual); every cell requires the leader's bounded
  `mbarrier_try_wait` to actually flip (`flip = 1`). The previous round of the same probe read the illegal instruction of gotcha 1.
- Certification probe (descriptor from the DSL atom + pair accounting) —
  **Status: ✅ GPU-certified**, `PROBE PASS max_diff=0.0`, all six contracts passed in the same round: `S3-offsets` (the destination-address
  offset model matches the library-produced swizzle face element by element, one 4-row group is one contiguous 512 B segment, trace time), `S3-stage-tma` (control: `.cta_group::2`
  whole-stage plain TMA), `S3-ptx-gather4` (single-thread gather4 issue), `S3-ptx-tile` (control: one 128 B `.tile` box per
  row), `S3-ring-4thr` (4 warps with one elected thread each, two 256-bit index loads per thread for 16 indices, all of the pair's instructions recorded on
  one `expect_tx` of the leader), `S3-ring-mask` (the valid-bit mask produced along the way by the same threads reads back exact). The spelling sweep in the same job:
  all five candidates compiled; both modifier orders, the atom's
  `CtaGroup` ONE / TWO, and no `.cta_group::2` with separate accounting per CTA are all byte-exact; only the cell whose descriptor carries no swizzle lands in a
  linear layout (gotcha 2).
- Multicast reading: cell `S42-mcast-g4` of another probe, `PROBE PASS max_diff=0`
  (the whole face exact in both CTAs, each barrier flipped). The four other cells of the same run test MMA forms, are judged FAIL,
  and are unrelated to this entry; this entry takes only this cell's reading, which is why gotcha 7 says "one reading" and it is not in the example.
- Issue-rate readings (two probes): the S47 probe
  (three cells `PROBE PASS`;
  single-thread issue of 192 instructions 1.765 µs, round period 2.033 µs; two warps one face each 1.408 µs; per-cell byte and instruction-count self-checks passed);
  the S46 probe (two warps period ratio 0.585, four warps 0.675
  i.e. no further scaling, multicast 0.551, index LDS chain removed 0.827; the four-warp cell is FAIL by its own verdict, and the reading is the conclusion).
- Landing readings (KEEP, only the supply channel swapped): before (device 7.0209 ms, paired ratio 0.1331); after
  (5.8345 ms, 0.1601); confirmation run (5.8361 ms).
- Gotcha 11 (`.L2::cache_hint`): GPU probe, after two rounds
  `PROBE PASS (max_diff=768.0) — 6/6 PASS` (job id not recorded; 768.0 is the reading of the sensitivity-control cell, whose verdict is
  `max_diff > 0`): the payload, scale bytes and two statistics of the with / without settings are bit-identical over 148 work items (`max_diff 0.0`,
  including 6 out-of-bounds sentinels and the padded tail block), in the PTX the count of `.L2::cache_hint` == the count of gather4 and `createpolicy` appears once,
  `LDL == STL == 0`, the real shape issued twice on the whole machine to completion. In the previous round two numeric-contract cells hit
  `ValueError` because the readback channel was unreachable on that tree, unrelated to the spelling. KEEP eval (NCU dossier census + gate window; eval job id not in the status record): axis `rings`
  `[…, "stride=0x400100"]` → `[…, "stride=0x400100,wrap=sel"]`, `cache_ops` unchanged, gate 0.997 → 1.0004 ms.
  PTX ISA: `createpolicy`, `.L2::cache_hint` of `cp.async.bulk.tensor`.
- Gotcha 12 (moving the issuing warps): GPU probe, after three rounds
  `PROBE PASS (max_diff=0.0) — 6/6 PASS` (job id not recorded); both earlier failures were on the probe's readback side (round one: the readback channel's
  switch combination was rejected; round two: scored against a wrong face-address model, reading 0.30–0.62), not on the issue side. KEEP eval (eval job id not in the status
  record): for axis `rings` see the fourth-tree paragraph of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11,
  gate 0.9993 → 0.9844 ms.
