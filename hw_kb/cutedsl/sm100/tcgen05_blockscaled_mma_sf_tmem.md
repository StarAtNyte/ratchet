# tcgen05_blockscaled_mma_sf_tmem (block-scaled UMMA: scale factors go TMA → smem → `tcgen05.cp` → TMEM, each instruction selects its own SF columns; the smem face of fp4 operands)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the block-scaled branch of [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md):
> that entry covers one `kind::f16` UMMA from construction to read-back; this entry covers only **the extra pieces** —
> how to build the `kind::mxf8f6f4` atom, the three-leg path of scale factors from gmem to TMEM, how each MMA instruction points at the SF columns of
> its own K=32 block, and **what a 4-bit operand looks like in smem under an 8-bit MMA kind**.
> The main-loop skeleton (face allocation, fragments, pipeline, TMEM allocation, read-back, pair form) is identical to that entry and is not repeated.

## Purpose

`tcgen05.mma.kind::mxf8f6f4` (and `kind::mxf4` / `kind::mxf4nvf4`) scales each row of A and each column of B, per **K=32** block (16 for mxf4nvf4),
by one UE8M0 / E4M3 scale factor. Hardware facts:

| Stage | Hardware fact | Corresponding DSL form |
|---|---|---|
| Where the SF lives | MMA reads SF from **TMEM** (the `idesc` carries the TMEM addresses of SFA / SFB) | `tiled_mma.set(Field.SFA, tCtSFA[(None, None, kblock)].iterator)`, same for `SFB`; set once before every `cute.gemm` |
| How SF gets into TMEM | `tcgen05.cp` (smem → TMEM, asynchronous, uses the MMA completion mechanism) | `Cp4x32x128bOp` copy atom + `make_s2t_copy(atom, tmem_sf_tensor)` + `get_s2t_smem_desc_tensor`; the `cute.copy` is issued by the MMA issuer |
| SF form in smem | a hardware-mandated blocked layout: one atom = 32 rows × 4 K blocks, row groups interleaved across 4 atoms | `make_smem_layout_sfa/sfb(tiled_mma, tiler, sf_vec_size, stages)`; TMEM side `make_tmem_layout_sfa/sfb(...)` |
| SF form in gmem | the TMA box must be able to move it into the blocked layout in one go | the gmem tensor is organised by the layout from `tile_atom_to_shape_SF((mn, k, l), sf_vec_size)`; the host reorders the plain (mn, k/32) SF into this layout **once** |
| Instruction shape | K of `kind::mxf8f6f4` is **32** (`kind::mxf4`: 64) | `make_blockscaled_trivial_tiled_mma(a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, (M, N))` fills K itself, k-tile = 4 × 32 = 128 |
| 4-bit operands | `kind::mxf8f6f4` reads operands as **8-bit containers**; a 4-bit type is either unpacked into the container by TMA or goes through the dense face of `kind::mxf4` | see the section "The smem face of 4-bit operands" (decided by the certification readings) |

## When needed (trigger)

- You need to issue a block-scaled UMMA (`Float8E4M3FN` × `Float8E4M3FN` with UE8M0 scales, or `Float4E2M1FN` × `Float8E4M3FN`),
  and do not know what to pass to `tiled_mma.set(Field.SFA/SFB, …)` or how the SF gets to TMEM.
- You get `pack_shape(): incompatible function arguments … Invoked with types: tuple`, profile errors, affine errors — as in that entry, first get this entry's example
  running verbatim, then change the geometry.
- Numerics are **all wrong but nothing hangs** (calc_diff ≈ 0.5–1 for every segment, independent of K): check the smem face form of 4-bit operands, the SF byte ↔ K block correspondence, and the host-side blocked reordering.
- **Only half of the pair's result is right**: check the `cta_group` of the SF upload (see "Pair form").

## Minimal runnable example

Only the lines that **differ** from that entry are listed; copy the `…` parts from it. A and B are both K-major (rows, K); SF is (rows, K/32) UE8M0 bytes.

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

A_DTYPE, B_DTYPE = cutlass.Float8E4M3FN, cutlass.Float8E4M3FN     # or A = cutlass.Float4E2M1FN, see the next section
SF_DTYPE, SF_VEC = cutlass.Float8E8M0FNU, 32
INST_MN = (128, 128)                 # pair form (256, N)
TILER = (INST_MN[0], INST_MN[1], 32 * 4)   # mxf8f6f4 instruction K=32, four per k-tile

@cute.jit
def host_bs(a_ptr: cute.Pointer, b_ptr: cute.Pointer, sfa_ptr: cute.Pointer,
            sfb_ptr: cute.Pointer, c_ptr: cute.Pointer):
    m, n, k, l = M, N, K, 1
    a_tensor = cute.make_tensor(a_ptr, cute.make_layout((m, k, l), stride=(k, 1, m * k)))
    b_tensor = cute.make_tensor(b_ptr, cute.make_layout((n, k, l), stride=(k, 1, n * k)))
    c_tensor = cute.make_tensor(c_ptr, cute.make_layout((m, n, l), stride=(n, 1, m * n)))
    # gmem view of the SF: the atom layout tiled over (mn, k, l) -- the host-side bytes are already arranged by it (see below)
    sfa_tensor = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, SF_VEC))
    sfb_tensor = cute.make_tensor(sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, SF_VEC))

    tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        A_DTYPE, B_DTYPE, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        SF_DTYPE, SF_VEC, tcgen05.CtaGroup.ONE, INST_MN)          # positional args; it fills K=32
    cta_layout_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (tiled_mma.thr_id,))

    a_smem = sm100_utils.make_smem_layout_a(tiled_mma, TILER, A_DTYPE, AB_STAGES)
    b_smem = sm100_utils.make_smem_layout_b(tiled_mma, TILER, B_DTYPE, AB_STAGES)
    sfa_smem = blockscaled_utils.make_smem_layout_sfa(tiled_mma, TILER, SF_VEC, AB_STAGES)   # plain Layout
    sfb_smem = blockscaled_utils.make_smem_layout_sfb(tiled_mma, TILER, SF_VEC, AB_STAGES)
    a_one, b_one = cute.slice_(a_smem, (None, None, None, 0)), cute.slice_(b_smem, (None, None, None, 0))
    sfa_one, sfb_one = cute.slice_(sfa_smem, (None, None, None, 0)), cute.slice_(sfb_smem, (None, None, None, 0))

    op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    tma_a, tma_ta = cute.nvgpu.make_tiled_tma_atom_A(op, a_tensor, a_one, TILER, tiled_mma, cta_layout_vmnk.shape)
    tma_b, tma_tb = cute.nvgpu.make_tiled_tma_atom_B(op, b_tensor, b_one, TILER, tiled_mma, cta_layout_vmnk.shape)
    tma_sfa, tma_tsfa = cute.nvgpu.make_tiled_tma_atom_A(op, sfa_tensor, sfa_one, TILER, tiled_mma,
                                                        cta_layout_vmnk.shape, internal_type=cutlass.Int16)
    tma_sfb, tma_tsfb = cute.nvgpu.make_tiled_tma_atom_B(op, sfb_tensor, sfb_one, TILER, tiled_mma,
                                                        cta_layout_vmnk.shape, internal_type=cutlass.Int16)
    tx_bytes = (cute.size_in_bytes(A_DTYPE, a_one) + cute.size_in_bytes(B_DTYPE, b_one)
                + cute.size_in_bytes(SF_DTYPE, sfa_one) + cute.size_in_bytes(SF_DTYPE, sfb_one)
                ) * cute.size(tiled_mma.thr_id.shape)
    ...  # kernel(...).launch(grid=..., block=[128,1,1], cluster=(1,1,1))

@cute.kernel
def kernel_bs(tiled_mma, tma_a, mA, tma_b, mB, tma_sfa, mSFA, tma_sfb, mSFB, mC,
              a_smem: cute.ComposedLayout, b_smem: cute.ComposedLayout,
              sfa_smem: cute.Layout, sfb_smem: cute.Layout, cta_layout_vmnk: cute.Layout,
              tx_bytes: cutlass.Constexpr):
    ...                                                   # (1)(2) of that entry: faces, pipeline
    sSFA = smem.allocate_tensor(element_type=SF_DTYPE, layout=sfa_smem, byte_alignment=128)   # the SF face is not composed
    sSFB = smem.allocate_tensor(element_type=SF_DTYPE, layout=sfb_smem, byte_alignment=128)
    gSFA = cute.local_tile(mSFA, cute.slice_(TILER, (None, 0, None)), (None, None, None))
    gSFB = cute.local_tile(mSFB, cute.slice_(TILER, (0, None, None)), (None, None, None))
    tCgSFA, tCgSFB = thr_mma.partition_A(gSFA), thr_mma.partition_B(gSFB)   # SF uses the same partitioning as A/B
    tAsSFA, tAgSFA = cpasync.tma_partition(tma_sfa, 0, cute.make_layout(1),
                                           cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
    tAsSFA, tAgSFA = cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)  # the blocked layout has 0-stride modes
    ...  # same for SFB

    # (4') TMEM: SFA columns right after the accumulator, then SFB columns
    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
    sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc), dtype=SF_DTYPE)
    tCtSFA = cute.make_tensor(sfa_tmem_ptr, blockscaled_utils.make_tmem_layout_sfa(tiled_mma, TILER, SF_VEC, sfa_one))
    sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
                                   + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=SF_DTYPE)
    tCtSFB = cute.make_tensor(sfb_tmem_ptr, blockscaled_utils.make_tmem_layout_sfb(tiled_mma, TILER, SF_VEC, sfb_one))

    # (4'') smem -> TMEM SF copy: atom + s2t tiled copy + smem descriptor view
    copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF_DTYPE)
    tCsSFA_c, tCtSFA_c = cute.filter_zeros(sSFA), cute.filter_zeros(tCtSFA)
    s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_c)
    thr_s2t_sfa = s2t_sfa.get_slice(0)
    tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(s2t_sfa, thr_s2t_sfa.partition_S(tCsSFA_c))
    tCtSFA_s2t = thr_s2t_sfa.partition_D(tCtSFA_c)
    ...  # same for SFB

    # (6') main loop: wait full -> two tcgen05.cp -> four MMAs, each first writing its own K block's SF columns into the idesc
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for k_tile in cutlass.range(k_tile_cnt, prefetch_stages=AB_STAGES - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(tma_a, ...); cute.copy(tma_b, ...)
            cute.copy(tma_sfa, tAgSFA[(None, ab_empty.count)], tAsSFA[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            cute.copy(tma_sfb, tBgSFB[(None, ab_empty.count)], tBsSFB[(None, ab_empty.index)], tma_bar_ptr=ab_empty.barrier)
            ab_full = ab_consumer.wait_and_advance()
            stage = (None, None, None, None, ab_full.index)
            cute.copy(s2t_sfa, tCsSFA_s2t[stage], tCtSFA_s2t)          # tcgen05.cp
            cute.copy(s2t_sfb, tCsSFB_s2t[stage], tCtSFB_s2t)
            for kb in cutlass.range(cute.size(tCrA, mode=[2]), unroll_full=True):
                tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kb)].iterator)
                tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kb)].iterator)
                cute.gemm(tiled_mma, tCtAcc, tCrA[(None, None, kb, ab_full.index)],
                          tCrB[(None, None, kb, ab_full.index)], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
        acc_empty.commit()
    ...                                                   # (7)(8) of that entry: read-back, teardown
```

**The host reorders the plain SF into the blocked layout** (once, outside the timed window; `(mn, k/32)` UE8M0 bytes → the form the kernel's TMA reads):

```python
@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(sf_ref_ptr: cute.Pointer, sf_mma_ptr: cute.Pointer,
                                   mn: int, sf_k: int, l: int, mma_shape: tuple):
    permuted_shape = tuple(mma_shape[i] for i in (3, 4, 1, 5, 2, 0))
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))
    sf_ref = cute.make_tensor(sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k)))
    sf_mma = cute.group_modes(cute.group_modes(cute.make_tensor(sf_mma_ptr, cute_layout), 0, 3), 1, 3)
    for i in cutlass.range(cute.size(sf_ref)):
        c = sf_ref.layout.get_hier_coord(i)
        sf_mma[c] = sf_ref[c]

# mma_shape = (l, ceil(mn/128), ceil(sf_k/4), 32, 4, 4); destination buffer = contiguous uint8 of this shape;
# source = (l, mn, sf_k) contiguous; both sides use make_ptr(cutlass.Float8E8M0FNU, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
```

## The smem face of 4-bit operands (`Float4E2M1FN` × `Float8E4M3FN`, `kind::mxf8f6f4`)

This is **the only place in this entry where the helper defaults must not be followed**, and the failure mode is **compiles, runs to completion, numerics all wrong**:

| Face | How to build it | Size of one stage (128 rows × K 128) | Certification reading |
|---|---|---|---|
| **Dense face (DSL default, forbidden)** | `make_smem_layout_a(tiled_mma, tiler, Float4E2M1FN, st)` → `S<2,4,3>` (K_SW64); TMA without `internal_type` | 8 KiB | compiles, launches, returns, **`max_diff = 1598`, almost every element wrong** |
| **8-bit container face (correct)** | `make_smem_layout_a(tiled_mma, tiler, cutlass.Uint8, st)` → `S<3,4,3>` (K_SW128); `allocate_tensor(element_type=Uint8, layout=.outer, swizzle=.inner)`; TMA `make_tiled_tma_atom_A(op, a_e2m1_tensor, a_one_u8, tiler, tiled_mma, vmnk.shape, internal_type=cutlass.Uint8)`; fragment directly `tiled_mma.make_fragment_A(sA_u8)` | 16 KiB | **`max_diff = 0.0`** |

- The gmem-side tensor is still typed **`Float4E2M1FN`** with its layout built in elements (`make_layout((m, k, l), stride=(k, 1, m*k))`, k is the element count, not bytes;
  even k in the low nibble); `internal_type=Uint8` makes TMA unpack each 4-bit element into 1 byte in smem, and MMA reads it as an 8-bit container.
- **`tx_count` counts packed bytes**: one stage of the container face is 16 KiB, but the TMA completion signal only counts 8 KiB (packed bytes). A barrier armed with 16 KiB **never fills up**
  (that level timed out in the certification). Spelling: `a_bytes = cute.size_in_bytes(Uint8, a_one) // 2`.
- Both operands E2M1 (`kind::mxf4`) does not go through this section: that is an op the helper selects on its own, and the dense face is its native form.

## SF upload in the pair form (`CtaGroup.TWO`)

In a pair, **B is split along N with each CTA loading one half, but SFB is not split**: each CTA's accumulator is its own 128 rows × the whole N, so each CTA needs the scale factors of the whole N.
The official composition gives SFB **its own** (1-CTA) tiled MMA and tiler, uses it for tile / partition / TMA, and multicasts to both sides of the pair:

```python
# host
tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
    A_DTYPE, B_DTYPE, K, K, SF_DTYPE, SF_VEC, tcgen05.CtaGroup.ONE,
    (INST_MN[0] // 2, cute.round_up(INST_MN[1], 128)))            # (M/pair, N rounded up to 128)
TILER_SFB = (INST_MN[0] // 2, round_up(INST_MN[1], 128), K_TILE)
cta_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout(CLUSTER_MNK), (tiled_mma_sfb.thr_id,))   # V=1
sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(CLUSTER_MNK, tiled_mma.thr_id)   # pair -> Multicast(TWO)
tma_sfb, tma_tsfb = cute.nvgpu.make_tiled_tma_atom_B(sfb_op, sfb_tensor, sfb_one, TILER_SFB, tiled_mma_sfb,
                                                    cta_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
# kernel
coord_sfb = cta_layout_sfb_vmnk.get_flat_coord(cute.arch.block_idx_in_cluster())
gSFB = cute.local_tile(mSFB, cute.slice_(TILER_SFB, (0, None, None)), (None, None, None))
tCgSFB = tiled_mma_sfb.get_slice(rank).partition_B(gSFB)
sfb_cta_layout = cute.make_layout(cute.slice_(cta_layout_sfb_vmnk, (0, None, 0, 0)).shape)
tBsSFB, tBgSFB = cpasync.tma_partition(tma_sfb, coord_sfb[1], sfb_cta_layout,
                                       cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
mcast_sfb = cpasync.create_tma_multicast_mask(cta_layout_sfb_vmnk, coord_sfb, mcast_mode=1)
cute.copy(tma_sfb, ..., tma_bar_ptr=ab_empty.barrier, mcast_mask=mcast_sfb)    # both CTAs issue
```

SFA uses the same partitioning as A (each CTA its own 128 rows); the TMEM layouts of `tCtSFA/tCtSFB` still use **the main tiled_mma and the main tiler**;
the `tcgen05.cp` atom is built as **`Cp4x32x128bOp(CtaGroup.TWO)`** and **issued only by the leader after full has been waited on** (it writes TMEM for both sides of the pair;
an atom built with `CtaGroup.ONE` does not even get through libNVVM on a pair, reporting the reasonless `NVVM backend compilation failed`);
`tx_count` is still "the sum of this CTA's four faces' bytes × 2" (certification reading: at inst (256,128), e4m3 A 16384 + half-width B 8192 + SFA 512 + SFB 512 = 25600, ×2 = 51200;
with A as an fp4 container face, A counts as packed 8192, ×2 = 34816). Building the SFB TMA atom directly from the main tiled_mma is rejected on the host:
`expected top-level shape equivalence between the SMEM layout and the CTA V-map` (this error = SFB did not use its own 1-CTA partitioning).
Pair geometry printed during certification: main `tiled_mma` ThrID `2:1`, TV-B `(2,(64,32))` (each CTA loads half of N), B face `S<3,4,3> o 0 o ((64,32),1,4,2)`;
`cta_layout_sfb_vmnk = ((1),2,1,1):((0),1,0,0)` (V=1, M direction 2 = multicast to both sides of the pair), `sfb_op = CopyBulkTensorTileG2SMulticastOp`.

## Gotchas

1. **`make_blockscaled_trivial_tiled_mma` takes positional arguments**, in the order `(a_dtype, b_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, (M, N))`;
   it selects the op by `(a_dtype, b_dtype, sf_vec_size)`: both E2M1 → `kind::mxf4` (K=64, vec 32) / `mxf4nvf4` (K=64, vec 16);
   otherwise → `kind::mxf8f6f4` (K=32). **The k-tile depth is its returned `shape_mnk[2]` times 4**; do not hard-code 64.
2. **The gmem / smem / TMEM layouts of the SF are all produced by helpers** (`tile_atom_to_shape_SF` / `make_smem_layout_sfa/sfb` /
   `make_tmem_layout_sfa/sfb`); they contain 0-stride modes, so `cute.filter_zeros` before TMA partitioning and s2t partitioning.
3. **The SF TMA atom needs `internal_type=cutlass.Int16`** (as in the example); the SF face is a plain `Layout`, so `allocate_tensor` takes `layout=` directly, with no `.outer/.inner`.
4. **Before every `cute.gemm`, `set(Field.SFA/SFB, tCtSF[(None, None, kb)].iterator)`** — what you pass is the `.iterator` of a TMEM tensor slice, not the tensor itself;
   the K block `kb` is the same index as the `kb` in `tCrA[(None, None, kb, stage)]`.
5. **`tcgen05.cp` is issued by the MMA issuer after full has been waited on and before the first MMA**; it rides the same `tcgen05` asynchronous chain as the MMA and needs no extra barrier.
6. **SF byte ↔ K block**: one UE8M0 byte scales 32 K; a 1×128 scale must be replicated into 4 bytes across the 4 K=32 blocks. Periodic wrong values = this step is wrong.
7. **fp32 accumulation**; when C is output as bf16/fp16, convert in the epilogue.

## Sources

- Certification probe (a GPU probe, not included in this release) — **status: ✅ GPU-certified**, final verdict
  `PROBE PASS max_diff=0.0` (certification round 2; round 1 already passed the four single-CTA cells, while the three pair cells were rejected at the construction of the SFB TMA atom),
  all seven contracts passed in the same round: `S0-lowers` — all five variants W1/W2/W3/P1/P3 compile to launchable kernels; `S1-mxf8-exact` 0.0 — e4m3×e4m3, UE8M0 vec 32,
  SF via TMA→smem→`tcgen05.cp`→TMEM, `set(Field.SFA/SFB)` per instruction, **bit-equal** to a dequantized fp32 reference; `S2-fp4-container-exact` 0.0 —
  A=e2m1 through the Uint8 container face + TMA `internal_type=Uint8`, tx counted in packed bytes; `S3-fp4-dense-face-is-wrong` — the same kernel with only the A face swapped for the DSL-default
  e2m1 dense face compiles, launches, returns, `max_diff=1598` (negative contract: this path **runs but is wrong**); `S4-tx-counts-packed-bytes` — the same container face with tx counted as
  16 KiB of container bytes: the barrier never flips, 90 s timeout (negative contract); `S5-pair-mxf8-exact` 0.0 / `S6-pair-fp4-container-exact` 0.0 — cluster (2,1,1),
  `CtaGroup.TWO`, SFB through its own 1-CTA tiled MMA + multicast, leader uploads with `Cp4x32x128bOp(CtaGroup.TWO)`. Fact row: on a pair, swapping the upload atom for
  `Cp4x32x128bOp(CtaGroup.ONE)` fails in the libNVVM backend (`NVVM backend compilation failed`, no reason given) — that reasonless error can also be this.
  Each variant in its own subprocess (`sys.executable`) with its own budget; A/B drawn from the integers {−2..2}, SF from 2^{0,1,2}, so the reference is exact.
- Public interfaces: `cutlass.utils.blackwell_helpers.make_blockscaled_trivial_tiled_mma`; `cutlass.utils.blockscaled_layout.tile_atom_to_shape_SF` /
  `make_smem_layout_sfa` / `make_smem_layout_sfb` / `make_tmem_layout_sfa` / `make_tmem_layout_sfb`;
  `cutlass.cute.nvgpu.tcgen05.Cp4x32x128bOp` / `make_s2t_copy` / `get_s2t_smem_desc_tensor` / `find_tmem_tensor_col_offset` / `Field.SFA` / `Field.SFB` /
  `MmaMXF8F6F4Op`; `cute.nvgpu.make_tiled_tma_atom_A/B(..., internal_type=…)`; `cute.filter_zeros`; `cute.recast_ptr`; `cutlass.cute.runtime.make_ptr`.
- Official docs: CUTLASS CuTe DSL documentation https://docs.nvidia.com/cutlass/latest/ (Blackwell block-scaled tutorial GEMM);
  cuBLAS documentation "block scaling factors layout"; PTX ISA `tcgen05.mma` kinds and the scale factor sections, `tcgen05.cp`.
