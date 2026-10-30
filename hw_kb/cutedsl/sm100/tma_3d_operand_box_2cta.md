# tma_3d_operand_box_2cta (the rank `<r>D` of a TMA descriptor follows the mode structure of the gmem view: how to get `.3D`, and why a larger box or an extra size-1 mode does not give it)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> Prerequisite entries: [`tma_2cta_nonmulticast_operand_load`](tma_2cta_nonmulticast_operand_load.md) (which load op
> each operand should use in a pair pipeline — that is a different knob), [`tma_flat_restage_box_rank`](tma_flat_restage_box_rank.md)
> (the rank can only be read back from SASS; a three-mode tensor whose last mode has stride 0 reads `.2D`). This entry answers one
> question only: **the contract requires `UTMALDG.3D.*` / `UTMASTG.3D`, your census shows `.2D` — what do you change?**

## Purpose

The r in `UTMALDG.<r>D` / `UTMASTG.<r>D` is the rank of the tensor map. CuTe's TMA constructors
(`make_tiled_tma_atom` / `make_tiled_tma_atom_A/_B`) do not let you write r directly; they derive it from **the mode
structure of the gmem tensor view you hand them**. The rules measured on the GPU (each has a SASS reading, see "Sources"):

| gmem view | r read back |
|---|---|
| Plain rank-2 tensor (the contiguous dimension is one flat mode), **regardless of** how many swizzle atoms each copy spans along the contiguous dimension: operand `BK = 64` / `128` (1 / 2 SW128 atoms), store tile `(64,64)` / `(64,128)` row-major, `(128,64)` column-major | **2D** |
| The contiguous dimension split into a hierarchical mode `(64, n/64):(1, 64)` (64 = the number of 16-bit elements in one SW128 atom), other modes unchanged — the same bytes, the same address function, and in layout algebra it could be coalesced back into one mode | **3D** (operands, both paths `UTMALDG.3D.MULTICAST.2CTA` / `UTMALDG.3D.2CTA`; store row-major / column-major `UTMASTG.3D`; the single-CTA-MMA load ops `CopyBulkTensorTileG2SOp(CtaGroup.ONE)` / `CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)` give `UTMALDG.3D` / `UTMALDG.3D.MULTICAST`, **without** the `.2CTA` suffix). Same whether the host tensor has a static layout or a `mark_layout_dynamic(leading_dim=1)` dynamic layout |
| The extent of the third mode L is **dynamic**: on the host, `permute(1, 2, 0)` of a contiguous `(L, M, K)` tensor gives `(M, K, L)`, then `from_dlpack(...).mark_layout_dynamic(leading_dim=1)` | **3D**, also for L = 1; with L = 2 the values are bit-exact (the L index is honoured) |
| The extent of the third mode is a **static 1**, whatever its stride: `(M, K, 1)` assembled from a rank-2 tensor inside `@cute.jit` (prints `(?,?,1):(?{i64},1,?{i64})`, non-zero stride); torch `unsqueeze(-1)` followed directly by `from_dlpack` (no `mark_layout_dynamic`) | **2D** (the mode is folded away) |

The address function does not change, so the device-side `cute.copy`, the box, the byte count and `expect_tx` all stay as they are and the
values are bit-identical; what changes is only the rank of the descriptor and, in the case below, **how many TMAs are issued per stage**.

### Instruction count when one stage spans n swizzle atoms along the contiguous dimension

Under SW128 the inner box dimension of a rank-2 map is at most one atom wide (64 elements of a 16-bit type = 128 B). So with `BK = 64·n` (n > 1),
under a plain rank-2 view **the same** `cute.copy(tma_atom, gmem_stage, smem_stage, ...)` is expanded into one instruction per atom:

| SASS per operand per stage (`BK = 128`, n = 2, SW128 fp16, producer loop `unroll=1`) | plain rank-2 view | contiguous dimension split as `(64, K/64):(1, 64)` |
|---|---|---|
| multicast path (`CopyBulkTensorTileG2SMulticastOp(CtaGroup.TWO)`) | 2 × `UTMALDG.2D.MULTICAST.2CTA` | 1 × `UTMALDG.3D.MULTICAST.2CTA` |
| non-multicast path (`CopyBulkTensorTileG2SOp(CtaGroup.TWO)`) | 2 × `UTMALDG.2D.2CTA` | 1 × `UTMALDG.3D.2CTA` |
| single-CTA-MMA non-multicast load op (`CopyBulkTensorTileG2SOp(CtaGroup.ONE)`) | 2 × `UTMALDG.2D` | 1 × `UTMALDG.3D` |

After the split the map is rank 3 and the box is `(64, rows, n)`; the n atoms sit on the third dimension, and one instruction moves this operand for a whole stage.
Static host layout and `mark_layout_dynamic` dynamic layout read the same; the smem layout (the face from `make_smem_layout_a/_b` is already
"n atoms stacked atom by atom"), `tx_count` and the device code are untouched, and the values are bit-identical. Only n = 2 is certified; n > 2 follows by the same rule, uncertified.

## When needed (trigger)

- The contract / design table says `UTMALDG.3D.[MULTICAST.]2CTA` or `UTMASTG.3D`, and the census `cache_ops` shows `.2D`.
- You wrapped the gmem tensor as `(M, K, 1)` to get `.3D`, and SASS still shows `.2D`.
- The contract requires **one** `UTMALDG.3D.*` per operand per stage, while your SASS has n `UTMALDG.2D.*` per operand per stage
  (`BK` spans n SW128 atoms): splitting the contiguous dimension fixes count and rank together (see the section above).
- You already switched one operand to `.3D` and the census still has one `.2D`: the remaining path's view was not split (gotcha 9).
- **Anti-trigger**: growing BK or the epi tile width to turn `.2D` into `.3D` — measured **ineffective** (still 2D), and you pay
  for it in smem / stage count for nothing.

> **This entry certifies only the spelling, the SASS footprint and the values, not performance.** Whole-kernel readings (only the gmem view changed; box, `expect_tx`, stage count all unchanged):
> on one tree (instructions per stage halved) device median 0.0124 → 0.0124 ms (±0%), NCU `stall_long_sb` across the two evals
> 13.84 → 15.60; on another tree (the single-CTA-MMA op) two landings 0.0090 → 0.0091 ms (+1.1%) and
> 0.0087 → 0.0089 ms (+2.3%). All three are flat to slightly slower, and the patches containing them were later reverted for lack of gain. On a third tree (instructions per stage halved)
> gate 0.0552 → 0.0558 ms (+1.09%, single reads from two different jobs; the same-job control baseline itself drifted by about 3% across two jobs on the same machine, and the same-job normalised reading actually favours it by 2.0%),
> recorded as flat within noise. Halving the instruction count shortens the producer's issue segment per stage;
> the bytes moved are unchanged. Do it when the contract requires `.3D` and accept it by count and rank — **do not treat it as a supply-side speed-up**.

## Minimal runnable example

### A. Split the contiguous dimension (host tensor untouched, only the view changes; inside `@cute.jit`)

```python
def split_contig(t, mode):
    # Split the unit-stride mode `mode` of rank-2 tensor t into (64, n/64):(1, 64); same bytes, same address function
    shp, std = t.shape, t.layout.stride
    n = shp[mode]                                   # requires n % 64 == 0
    sub = ((64, n // 64), (1, 64))
    if mode == 1:
        lay = cute.make_layout((shp[0], sub[0]), stride=(std[0], sub[1]))
    else:
        lay = cute.make_layout((sub[0], shp[1]), stride=(sub[1], std[1]))
    return cute.make_tensor(t.iterator, lay)

# Operands (K-major, SW128, fp16, BK stays 64):
a_in = split_contig(a, 1)             # a: (M, K) K-contiguous -> (M, (64, K/64)):(K, (1, 64))
b_in = split_contig(b, 1)
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op_a, a_in, a_one_stage_layout, TILER_MNK, tiled_mma, cta_layout_vmnk.shape)   # -> UTMALDG.3D.*
b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
    op_b, b_in, b_one_stage_layout, TILER_MNK, tiled_mma, cta_layout_vmnk.shape)   # -> UTMALDG.3D.*
# Device side unchanged: the kernel does local_tile / tma_partition on the a_tma_tensor / b_tma_tensor returned by the constructor.
# The same code with BK = 64·n in TILER_MNK: per operand per stage, n UTMALDG.2D.* become 1 UTMALDG.3D.*.
# Single-CTA MMA: swap op_a / op_b for the two CtaGroup.ONE ops (the multicast path still passes cta_layout_vmnk.shape); the split is the same.

# TMA store (16-bit output, epi tile along the contiguous dimension = 64):
lay    = utils.LayoutEnum.from_tensor(c)                      # take the majorness from the original tensor first
c_in   = split_contig(c, 1 if lay == utils.LayoutEnum.ROW_MAJOR else 0)
staged = sm100_utils.make_smem_layout_epi(c_dtype, lay, (64, 64), C_STAGES)
one    = cute.slice_(staged, (None, None, 0))
tma_st, tma_c = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c_in, one, (64, 64))  # -> UTMASTG.3D
# device: local_tile / tma_partition on tma_c (not the original c); cute.copy has the same shape as in 2D.
```

### B. Dynamic L mode (host side changes the tensor's form; the kernel gets one more index)

```python
from cutlass.cute.runtime import from_dlpack

# host (torch side): the rank-3 view comes from a permute of a contiguous (L, M, K) tensor; stride of (M, K, L) = (K, 1, M*K)
A3 = A.view(1, M, K).permute(1, 2, 0)            # A: (M, K) contiguous -> (M, K, 1), L-mode extent dynamic
w = lambda t: from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
# ...hand w(A3) / w(B3) / w(C3) to @cute.jit, then make_tiled_tma_atom_A/_B / make_tiled_tma_atom as usual

# kernel: tile the whole tensor; after partition there is an extra last mode RL; supply the L index when taking a tile
gA = cute.local_tile(mA, cute.slice_(MMA_TILER, (None, 0, None)), (None, None, None))  # (BM, BK, RM, RK, RL)
tAsA, tAgA = cpasync.tma_partition(tma_atom_a, coord, cta_layout,
                                   cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3))
tAgA_t = tAgA[(None, m_tile, None, l)]           # (TMA, RK); l = 0 when L = 1
cute.copy(tma_atom_a, tAgA_t[(None, k)], tAsA[(None, slot)], tma_bar_ptr=bar)
# store side likewise: gT = local_tile(mC_tma, (EPI_M, EPI_N), (None, None, None)) -> (EPI_M, EPI_N, RM, RN, L),
#              cute.copy(atom_c, bS[(None, buf)], bG[(None, rm, rn, l)])
```

Both forms only change the host-side view (B also adds one L index in the kernel); `expect_tx` / `tx_count` are unchanged.

## Gotchas

1. **Trying to raise the rank with an extra size-1 mode.** A mode with static extent 1 is folded away by the constructor, and a non-zero stride does not save it (see row four of the
   "Purpose" table). For `.3D`, either split the contiguous dimension into a hierarchical mode (example A) or make the third mode's extent dynamic (example B).
   Other constructions (other permutes, `mark_compact_shape_dynamic` combinations, etc.) were not tested; go by the census and **do not assume**.
2. **Trying to raise the rank with a larger BK / epi tile width.** Measured ineffective: under a plain rank-2 gmem view `BK = 128` and stores `(64,128)` /
   `(128,64)` are all still `.2D`. What you pay for nothing: smem per stage doubles ⇒ stage count roughly halves
   ([`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md)), and a hand-written `tx_count` constant must change or it hangs.
   Moreover, under a plain view `BK = 128` is **2** `.2D` instructions per operand per stage (one per atom, see the end of "Purpose").
3. **Acceptance can only look at SASS.** The 2D and 3D forms move the same bytes to the same addresses; `max_diff` cannot see r;
   the contract is that `.3D` appears and `.2D` disappears in the census `cache_ops`.
4. **Take the majorness before splitting.** The certified form calls `LayoutEnum.from_tensor` on the original tensor first, then uses it to decide whether to split mode 1
   (row-major) or mode 0 (column-major); taking the majorness on the hierarchical view is uncertified — do not rely on it.
5. **On the device side use the tensor returned by the constructor.** In the certified form of example A the kernel takes the coordinate tensor returned by `make_tiled_tma_atom*`,
   not the original 2D tensor; mixing the two views is uncertified.
6. **The split sub-mode size = the number of elements of one swizzle atom along the contiguous dimension.** This entry certifies only 64 for SW128 + 16-bit;
   SW64 / SW32 atoms are 32 / 16 16-bit elements wide, so the sub-mode size changes with the swizzle — uncertified.
7. **In example B the L index is the last mode after partition.** After `tma_partition` the shape is `(TMA, RM, RK, RL)`; taking a tile
   must be written `(None, m, None, l)`; on the store side likewise `(None, rm, rn, l)`.
8. **No special handling when the row count is not a multiple of the tile.** Example A only splits the contiguous dimension; the row dimension keeps its original (possibly dynamic) extent;
   out-of-bounds rows are zero-filled by TMA as usual, those bytes are counted into complete_tx as usual, and `tx_count` still counts the full box — no hang. Certified forms:
   in a pair pipeline the MMA-N-side non-multicast operand (`make_tiled_tma_atom_B` + `CopyBulkTensorTileG2SOp(CtaGroup.TWO)`, each CTA a box of half
   MMA-N) with the last tile out of bounds — the corresponding output columns are exactly 0; single-CTA MMA + cluster multicast with the multicast path's rows out of bounds — the out-of-bounds region reads back 0;
   single-CTA MMA non-multicast op with the MMA-N-side path having fewer rows than the box — out-of-bounds columns exactly 0 under both the `.2D` and the `.3D` view.
   Do not change `tx_count` or add predicates because you switched to `.3D`.
9. **Split the two operands separately and accept each one.** If only one path is split the census shows one `.3D` + one `.2D` (e.g. A split, B not →
   `UTMALDG.3D.2CTA` + `UTMALDG.2D.2CTA`). `cache_ops` is the deduplicated opcode set; looking only for "`.3D` appeared" misses the unsplit path;
   check each tensor handed to `make_tiled_tma_atom_A` / `_B` / `make_tiled_tma_atom`.
10. **When accepting by count, confirm the unroll factor first.** How to read "instructions per stage": with the producer loop at `cutlass.range(..., unroll=1)`,
   the **static count** of each `UTMALDG` opcode in SASS is the number issued per stage; if the loop is unrolled, multiply by the unroll factor.

## Sources

- Certification probe (splitting the contiguous dimension; BK / epi width ineffective; a GPU probe, not included in this release) —
  **✅ `PROBE PASS max_diff=0.0`** (certification round 4). Readings (each variant compiled in its own process → SASS read →
  run → compared element-wise with the reference):
  - G0 pair GEMM, `BK = 64`, plain `(M,K)`: `UTMALDG.2D.2CTA` + `UTMALDG.2D.MULTICAST.2CTA`, max_diff 0.0 against the fp32 reference.
  - G1 pair GEMM, `BK = 128`, plain gmem: still `.2D`, max_diff 0.0.
  - G2 pair GEMM, `BK = 64`, K viewed as `(64, K/64):(1, 64)`: TMA coordinate tensor `(512,(64,8)):(1@1,(1@0,1@2))`,
    `UTMALDG.3D.2CTA` + `UTMALDG.3D.MULTICAST.2CTA`, max_diff 0.0.
  - S0 / S1 / S2 store (generic writes to a swizzle face → `fence_proxy` → single-thread TMA store), row-major `(64,64)` / row-major `(64,128)` /
    column-major `(128,64)`, plain gmem: `UTMASTG.2D`, bit-identical.
  - S5 / S6 store, row-major / column-major `(64,64)`, contiguous dimension viewed as `(64, n/64):(1, 64)`: `UTMASTG.3D`, bit-identical.
  - Contract rows: T0-lowers PASS (n/a), T1-baseline-exact 0.0, T2-operand-3d `['G2']` 0.0, T3-store-3d `['S5','S6']` 0.0.
- Certification probe (dynamic L mode; static extent-1 folded away) —
  **✅ `PROBE PASS max_diff=0.0`** (certification round 1, all four contract rows pass). Readings: permute-built L=1 pair GEMM
  (A view `(?,?,?):(?{i64},1,?{i64})`) `UTMALDG.3D.2CTA`, values bit-identical; permute-built store `UTMASTG.3D`, bit-identical;
  rank-2 control `UTMALDG.2D.2CTA` / `UTMASTG.2D`; static `(M,K,1)` assembled inside `@cute.jit` (`(?,?,1):(?{i64},1,?{i64})`)
  `UTMALDG.2D.2CTA`; permute-built L=2 `UTMALDG.3D.2CTA`, values bit-identical.
- Static `(M,K,1)` from torch `unsqueeze(-1)` reads `.2D`: variant C2 of the certification probe of the `tma_2cta_nonmulticast_operand_load` entry
  (`PROBE PASS max_diff=0.0`).
- Certification probe (instructions per stage at `BK = 128`, end of "Purpose", gotcha 10) —
  **✅ `PROBE PASS max_diff=0.0`** (certification round 2, all four contract rows pass). Pair GEMM, A on the multicast pair op, B on the non-multicast
  pair op, producer loop `unroll=1`, each variant compiled in its own process → static SASS count → run → compared with the fp32 reference:
  - P (plain rank-2 view, dynamic layout): 2 × `UTMALDG.2D.MULTICAST.2CTA` + 2 × `UTMALDG.2D.2CTA`, max_diff 0.0.
  - H (K viewed as `(64, K/64):(1, 64)`, static layout) / HD (same, `mark_layout_dynamic(leading_dim=1)`): 1 × `UTMALDG.3D.MULTICAST.2CTA`
    + 1 × `UTMALDG.3D.2CTA` each, max_diff 0.0.
  - Contract rows: T0-runs PASS (n/a), T1-exact 0.0, T2-rank3 PASS (n/a), T3-one-issue-per-operand-per-stage PASS (n/a).
- Certification probe (single-CTA MMA + cluster multicast, row two of the "Purpose" table; rows out of bounds, gotcha 8)
  — **✅ `PROBE PASS max_diff=0.0`** (certification round 1). Load ops
  `CopyBulkTensorTileG2SMulticastOp(CtaGroup.ONE)` / `CopyBulkTensorTileG2SOp(CtaGroup.ONE)`, via `make_tiled_tma_atom_A` / `_B`:
  - A-multicast and B-multicast forms, rank-2 view: `UTMALDG.2D` + `UTMALDG.2D.MULTICAST`; K viewed as `(64, K/64):(1, 64)` (dynamic layout,
    coordinate tensor `(?,(64,?)):(1@1,(1@0,1@2))`): `UTMALDG.3D` + `UTMALDG.3D.MULTICAST`; max_diff 0.0 for all four.
  - Same split, static layout (`(100,(64,8)):(512,(1,64))`): likewise `UTMALDG.3D` + `UTMALDG.3D.MULTICAST`, max_diff 0.0.
  - Multicast path with 100 rows and a 128 tile: the out-of-bounds region reads back 0 (`oob_region_absmax=(0.0, 0.0)`).
- Certification probe (pair pipeline MMA-N-side non-multicast operand, tail tile out of bounds, gotcha 8 / gotcha 9)
  — **✅ `PROBE PASS max_diff=0.0`** (certification round 2, all four contract rows pass). B via
  `make_tiled_tma_atom_B` + `CopyBulkTensorTileG2SOp(CtaGroup.TWO)`, each CTA a box of half MMA-N, B's row count not a multiple of MMA-N:
  - X2 (cluster `(2,1,1)`, A split, B not): `UTMALDG.3D.2CTA` + `UTMALDG.2D.2CTA`, max_diff 0.0, out-of-bounds columns `tail_max=0.0`.
  - X3 (same, B split too, coordinate tensor `(?,(64,?)):(1@1,(1@0,1@2))`): only `UTMALDG.3D.2CTA`, max_diff 0.0, `tail_max=0.0`.
  - Y3 (cluster `(2,2,1)`, A multicast, B non-multicast, both split): `UTMALDG.3D.2CTA` + `UTMALDG.3D.MULTICAST.2CTA`, max_diff 0.0, `tail_max=0.0`.
  - Contract rows: T0-lowers PASS (n/a), T1-2d-baseline 0.0, T2-half-box-3d 0.0, T3-with-mcast-peer 0.0.
- Certification probe (single-CTA MMA non-multicast op, instructions per stage at `BK = 128`, row three of the table at the end of "Purpose"; rows out of bounds, gotcha 8):
  **✅ `PROBE PASS max_diff=0.0`**
  (certification round 1, all three contract rows pass). Both operands are `CopyBulkTensorTileG2SOp(CtaGroup.ONE)`, via `make_tiled_tma_atom_A` / `_B`,
  producer loop `unroll=1`, each variant compiled in its own process → static SASS count → run → compared with the fp32 reference:
  - P0 (plain rank-2 view, static layout): 4 static `UTMALDG`, all `.2D`, max_diff 0.0, out-of-bounds columns 0.0.
  - P1 (K viewed as `(64, K/64):(1, 64)`, static layout, coordinate tensor `(…,(64,…)):(1@1,(1@0,1@2))`): 2 static `UTMALDG`, all `.3D`, max_diff 0.0, out-of-bounds columns 0.0;
    smem layout identical to P0.
  - P2 (same split, `mark_layout_dynamic` dynamic layout): likewise 2 `.3D`, max_diff 0.0, out-of-bounds columns 0.0.
  - Contract rows: T0-lowers PASS (n/a), T1-baseline 2D x4 0.0, T2-3d-one-box static x2 0.0 (P2 is an informational row only).
- Public interfaces: `cutlass.cute.nvgpu.make_tiled_tma_atom_A` / `_B`, `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom`,
  `CopyBulkTensorTileG2SOp`, `CopyBulkTensorTileG2SMulticastOp`, `CopyBulkTensorTileS2GOp`, `cpasync.tma_partition`,
  `cute.make_layout`, `cute.make_tensor`, `cute.local_tile`,
  `cutlass.utils.blackwell_helpers.make_smem_layout_a` / `_b` / `make_smem_layout_epi`, `cutlass.utils.LayoutEnum`,
  `cutlass.cute.runtime.from_dlpack(...).mark_layout_dynamic`,
  `tcgen05.CtaGroup`, `cutlass.cute.nvgpu.cpasync.create_tma_multicast_mask`.
- CUDA Driver API, `cuTensorMapEncodeTiled` (tensorRank, boxDim; with SW128 the inner box dimension ≤ 128 B; out-of-bounds element fill):
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- PTX ISA, `cp.async.bulk.tensor` (the `.1d`–`.5d` rank modifier matches the tensor-map rank):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
