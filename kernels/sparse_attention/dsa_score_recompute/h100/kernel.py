"""dsa_score_recompute kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py and README.md, do not change):

  dsa_score_recompute(q_attn, k_attn, lse, q_idx, k_idx, weights,
                      topk_indices, sm_scale, topk_length=None)
      -> (target, predict)

    q_attn       : torch.Tensor (1, S_Q, H_Q, D_QK)    cuda bfloat16
    k_attn       : torch.Tensor (1, S_KV, D_QK)        cuda bfloat16
    lse          : torch.Tensor (1, S_Q, H_Q)          cuda float32  KV-only natural-log LSE (given)
    q_idx        : torch.Tensor (1, S_Q, H_IDX, D_IDX) cuda bfloat16
    k_idx        : torch.Tensor (1, S_KV, D_IDX)       cuda bfloat16
    weights      : torch.Tensor (1, S_Q, H_IDX)        cuda bfloat16  (indexer scale folded in)
    topk_indices : torch.Tensor (1, S_Q, TOPK)         cuda int32    invalid family: -1
    sm_scale     : float                                attention scale (d_qk**-0.5)
    topk_length  : torch.Tensor (1, S_Q) cuda int32 | None

    target  : torch.Tensor (1, S_Q, TOPK) cuda float32
              (sum_h exp(q_attn·k_topk·sm_scale - lse_h)), L1-normalized over
              the valid prefix; EXACT ZEROS outside it
    predict : torch.Tensor (1, S_Q, TOPK) cuda float32
              softmax over the valid prefix of sum_h relu(q_idx·k_topk)·w_h;
              EXACT ZEROS outside it

  fp32 accumulation everywhere; the pad-zero clause is a seam contract
  (chains/dsa_bwd/manifest.toml) — the consumer contracts pads against
  log(predict), so a nonzero pad is a live gradient leak.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc. Delegating the math to torch or to any reference library reachable in
the image fails the harness (eval/bench.py scans this file; the banned-token
list is in its fence). JIT-compile once and cache the compiled kernel across
calls (the bench times repeated calls with rotating buffers).

--------------------------------------------------------------------------
Build state: M0 architecture complete — this tree is the M0 gate submission.
Nothing below changed in this revision; the construction ran in three steps,
each with its own whole-machine smoke, and the decisions landed as:

  D0.1  / D0.2  / D0.14  two CuTe DSL kernels (attn -> target, idx ->
        predict) launched serially from the wrapper with zero preprocessing
        kernels, 1 CTA = 1 query token, grid 32768, runtime n_tiles.
  D0.3  / D0.4   attn stationarity (64-head Q half-slab resident x 2 passes,
        K re-gathered per pass, head sums crossing the pass boundary in
        smem) and the exp2 numeric contract (sm_scale.log2e host-folded,
        lse.log2e in-kernel per pass, masking at the P/s layer, den + eps).
  D0.5  / D0.15  256 threads = WG0 copy-producer + WG1 mma-consumer, one
        role branch after a shared prelude, no setmaxnreg (compiler-default
        register allocation, spill-free target, 1 CTA/SM by the smem bound).
  D0.6  / D0.7   HGMMA 64x64x16 bf16->f32 SS (M = 64 key, N = 64 head,
        K = 512 / 128) over the SW128 K-major smem map: 201728 B for attn,
        54272 B for idx.
  D0.8  / D0.9 / D0.10  two-phase predicated cp.async.cg 128-bit K gather,
        one symmetric named-barrier rendezvous per tile serving as both full
        and empty signal, and the per-tile issue order ARRIVE -> 32 HGMMA ->
        wait_group(0) -> elementwise segment -> BAR.
  D0.11  the resident Q half-slab (attn) and Qi slab (idx) move by TMA bulk
        against a single mbarrier with `arrive.expect_tx` — 8 boxes /
        expect_tx(65536) per attn pass, 2 boxes / expect_tx(16384) for idx,
        each box a 64 head x 64 d = 8 KiB SW128 column block.
  D0.12 / D0.13  head sums merged inside the loop (16 in-thread adds + quad
        butterfly + predicated scalar STS), denominator once in the epilogue,
        pads stored as exact zeros; the idx kernel carries the same skeleton
        at 64 heads x 128 d, single pass, with the raw-domain row max and the
        all-invalid-row zero branch in its softmax epilogue.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# ---------------------------------------------------------------- constants --
LOG2E = 1.4426950408889634
NTHREADS = 256           # D0.5: WG0 copy-producer + WG1 mma-consumer
WG = 128
KTILE = 64               # D0.2/D0.6: 64 keys per tile
H_HALF = 64              # D0.3: resident Q half-slab = 64 heads
H_Q = 128
NPASS = H_Q // H_HALF    # D0.3: 2 passes, K streamed in full on each
D_QK = 512
H_IDX = 64
D_IDX = 128
MAX_TOPK = 1024
PARK = -1.0e30           # finite park value (never -inf: (-inf)-(-inf) = NaN)
NEG_TEST = -1.0e29

# named barriers (D0.9).  0xa is consumer-internal (128), the rest are the
# symmetric 256-thread rendezvous points.
BAR_EPI = 10             # 0xa: consumer warpgroup epilogue reduction
BAR_PASS = 11            # 0xb: once per pass — Q half-slab + lse are in
BAR_RING = 12            # 0xc: once per tile — full signal AND empty signal
BAR_EXIT = 13            # 0xd: exit rendezvous

# copy geometry: one row is `nlane * ncp` 128-bit segments; `ngrp` rows are
# filled in parallel by the 128 producer threads, `nrow` rounds cover 64 rows.
ATTN_CP = (4, 16, 8, 8)  # 512 bf16 rows: 64-thread atom, 32 copies/thread/tile
IDX_CP = (8, 8, 16, 1)   # 128 bf16 rows: 16-thread atom, 8 copies/thread/tile

# D0.11 TMA geometry: one box = 64 head rows x 64 d = one 8 KiB SW128 column
# block (the swizzle atom is 8 rows x 128 B, and 64 bf16 = 128 B).
TMA_BOX = 64
NBOX_Q = D_QK // TMA_BOX          # 8 boxes / pass
NBOX_QI = D_IDX // TMA_BOX        # 2 boxes / CTA
QBYTES = H_HALF * D_QK * 2        # 65536: the attn expect_tx
QIBYTES = H_IDX * D_IDX * 2       # 16384: the idx expect_tx


# ------------------------------------------------------------------ API drift --
# hw_kb/cutedsl/sm90/cute_dsl_trace_scope.md gotcha 4: `cute.make_fragment(shape, dtype)`
# is spelled `cute.make_rmem_tensor` on some DSL builds.  Resolve, do not guess.
_MK_RMEM = getattr(cute, "make_fragment", None)
if _MK_RMEM is None:
    _MK_RMEM = cute.make_rmem_tensor


# hw_kb/cutedsl/sm90/softmax_segment_ilp.md (b): `cute.arch.fmax` is the DSL spelling
# where it exists; where it does not, the certified PTX escape.  Either way one
# `FMNMX`, never the `if v > m` form (that is FSETP + FSEL per hop).
_FMAX = getattr(cute.arch, "fmax", None)
if _FMAX is None:
    from cutlass.cutlass_dsl import T as _T

    @dsl_user_op
    def _FMAX(a, b, *, loc=None, ip=None) -> cutlass.Float32:
        return cutlass.Float32(llvm.inline_asm(
            _T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;", "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# -------------------------------------------------- non-bulk group completion --
# `cute.arch` carries only the bulk (TMA) commit/wait on this build; the
# non-bulk cp.async group ops go through the certified inline-PTX escape
# (hw_kb/cutedsl/sm90/cp_async.md section 3, hw_kb/cutedsl/sm90/ptx_escape.md form 1).
@dsl_user_op
def _cpa_commit_asm(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _cpa_wait_asm(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


_ARCH_COMMIT = getattr(cute.arch, "cp_async_commit_group", None)
_ARCH_WAIT = getattr(cute.arch, "cp_async_wait_group", None)


def _cpa_commit():
    if _ARCH_COMMIT is None:
        _cpa_commit_asm()
    else:
        _ARCH_COMMIT()


def _cpa_wait(n):
    if _ARCH_WAIT is None:
        _cpa_wait_asm(n)
    else:
        _ARCH_WAIT(n)


# ------------------------------------------------------- shared memory maps --
# D0.7 resource-topology contract.  Sizes are the contract; the offsets follow
# the allocator.  attn: 8 + 512 + 256 + 4096 -> pad to 1024 -> 5120, + 64 KiB Q
# + 128 KiB K ring = 201728 B  -> 1 CTA/SM.
@cute.struct
class AttnStorage:
    sBar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
    sRed: cute.struct.MemRange[cutlass.Float32, WG]
    sLse: cute.struct.MemRange[cutlass.Float32, H_HALF]
    sSums: cute.struct.MemRange[cutlass.Float32, MAX_TOPK]
    sQ: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, H_HALF * D_QK], 1024]
    sK: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, 2 * KTILE * D_QK], 1024]


# idx: 8 + 512 + 128 + 4096 -> pad to 1024 -> 5120, + 16 KiB Qi + 32 KiB K ring
# = 54272 B -> 3 CTA/SM.
@cute.struct
class IdxStorage:
    sBar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 1], 8]
    sRed: cute.struct.MemRange[cutlass.Float32, WG]
    sW: cute.struct.MemRange[cutlass.BFloat16, H_IDX]
    sScores: cute.struct.MemRange[cutlass.Float32, MAX_TOPK]
    sQi: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, H_IDX * D_IDX], 1024]
    sKi: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, 2 * KTILE * D_IDX], 1024]


# ----------------------------------------------- trace-time object builders --
# Called from inside a traced `@cute.jit` entry ONLY: `make_smem_layout_atom` /
# `get_smem_layout_atom` reach `ir.Type.parse` for the swizzle type and
# `make_trivial_tiled_mma` / `make_copy_atom` build MLIR types, so at module
# scope (no MLIR Context) they raise "An MLIR function requires a Context but
# none was provided" (hw_kb/cutedsl/sm90/cute_dsl_trace_scope.md).  The struct above
# keeps plain element counts; the asserts below pin the swizzled cosize back
# onto those counts before anything is allocated against them.
def _check_box_view(flat, boxed, nhead, depth):
    """Trace-time: the (nhead, depth) MMA view of the Q slab and the
    (nhead, 64, nbox) TMA view must address the SAME bytes -- box i has to be
    the i-th 8 KiB column block, or the TMA lands Q somewhere the descriptor
    read side does not look.  Both come from `tile_to_shape` of the same SW128
    atom with the m-tiles fastest, so they agree by construction; a mismatch
    would only ever surface as silently wrong Q data on the GPU, so pin it
    here.  Skipped (not failed) if the layout is not callable on this build.
    """
    bad = None
    try:
        f, b = flat.outer, boxed.outer
        ds = list(range(TMA_BOX)) + list(range(0, depth, 7))
        for h in range(nhead):
            for d in ds:
                if f((h, d)) != b((h, d % TMA_BOX, d // TMA_BOX)):
                    bad = (h, d)
                    break
            if bad is not None:
                break
    except Exception:                       # no callable layout on this build
        return
    if bad is not None:
        raise AssertionError("Q TMA box view != MMA view at %r" % (bad,))


def _smem_layouts(depth, nhead):
    """SW128 K-major (D0.7): 2-stage K ring + the resident Q slab.

    `sqb` is the SAME Q bytes seen as `nbox` boxes of (nhead, 64) -- the shape
    D0.11's TMA moves.  `cute.slice_(sqb, (None, None, 0))` is the single-box
    layout `make_tiled_tma_atom` wants (KB wgmma_tma_mbarrier gotcha 2).
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR,
                                        cutlass.BFloat16, depth),
        cutlass.BFloat16)
    sk = cute.tile_to_shape(atom, (KTILE, depth, 2), order=(0, 1, 2))
    sq = cute.tile_to_shape(atom, (nhead, depth), order=(0, 1))
    sqb = cute.tile_to_shape(atom, (nhead, TMA_BOX, depth // TMA_BOX),
                             order=(0, 1, 2))
    assert cute.cosize(sk) == 2 * KTILE * depth, cute.cosize(sk)
    assert cute.cosize(sq) == nhead * depth, cute.cosize(sq)
    assert cute.cosize(sqb) == nhead * depth, cute.cosize(sqb)
    _check_box_view(sq, sqb, nhead, depth)
    return sk, sq, sqb


def _g2s(nthr):
    """D0.8: cp.async.cg 128-bit (LDGSTS.E.BYPASS.128), `.cg` bypasses L1."""
    g2s_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16, num_bits_per_copy=128)
    return cute.make_tiled_copy_tv(g2s_atom, cute.make_layout(nthr),
                                   cute.make_layout(8))


def _mma(nhead):
    """D0.6: HGMMA.64x64x16.F32.BF16, SS mode, M = 64 key / N = 64 head."""
    return sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        cutlass.Float32, atom_layout_mnk=(1, 1, 1), tiler_mn=(KTILE, nhead))


# ------------------------------------------------------------ producer bodies --
@cute.jit
def _gather_tile(sD, mSrc, mIdx, bidx, base, grp, glane, tiled_g2s, ntk,
                 s_kv: cutlass.Constexpr, cap: cutlass.Constexpr,
                 nrow: cutlass.Constexpr, ngrp: cutlass.Constexpr,
                 nlane: cutlass.Constexpr, ncp: cutlass.Constexpr):
    """D0.8: two-phase predicated gather of one 64-key tile into `sD`.

    Phase 1 issues the whole batch of index LDGs (clamped subscript, no
    dependent copy behind any of them) and reduces each to a Boolean live
    flag; phase 2 walks the rows issuing either the 128-bit cp.async.cg group
    or a vector 16 B zero fill.  Keeping the two phases apart is the D0.8
    ordering contract: no LDGSTS is tight-chained onto its own LDG
    (hw_kb/cutedsl/sm90/gather_index_prefetch_predicated.md).

    `slot < ntk` (not `< tl`) is deliberate — rows between tl and the tile
    boundary carry real data and are masked later at the P/s layer (D0.4).
    """
    idxs = []
    lives = []
    for rr in cutlass.range_constexpr(nrow):
        slot = base + grp + rr * ngrp
        slot_c = cutlass.Int32(0)
        if slot < cap:
            slot_c = slot
        iv = cutlass.Int32(mIdx[(bidx, slot_c)])
        idxs.append(iv)
        lives.append((slot < ntk) & (iv >= 0) & (iv < s_kv))
    for rr in cutlass.range_constexpr(nrow):
        drow = sD[(grp + rr * ngrp, None)]
        if lives[rr]:
            for t in cutlass.range_constexpr(ncp):
                thr_c = tiled_g2s.get_slice(t * nlane + glane)
                cute.copy(tiled_g2s,
                          thr_c.partition_S(mSrc[(idxs[rr], None)]),
                          thr_c.partition_D(drow))
        else:
            chunks = cute.flat_divide(drow, (8,))
            for t in cutlass.range_constexpr(ncp):
                chunks[(None, t * nlane + glane)].fill(0.0)


@cute.jit
def _tma_slab(tma_atom, tSrc, sDb, mbar, row_tile, warp_idx, nhead,
              tx: cutlass.Constexpr, nbox: cutlass.Constexpr):
    """D0.11: move one resident Q slab with `nbox` TMA boxes against a single
    mbarrier.

    `arrive.expect_tx` is a single-thread instruction, so it goes inside
    `elect_one` (KB elect_one); the TMA copies themselves must NOT be wrapped
    in `elect_one` (the copy atom elects internally -- wrapping deadlocks), so
    the whole of warp 0 issues them under a warp-uniform predicate.  One
    arrive + `tx` bytes completes the phase; the caller waits on it.
    """
    gS = cute.local_tile(tSrc, (nhead, TMA_BOX), (row_tile, None))
    tSs, tSg = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sDb, 0, 2), cute.group_modes(gS, 0, 2))
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, tx)
        for b in cutlass.range_constexpr(nbox):
            cute.copy(tma_atom, tSg[(None, b)], tSs[(None, b)],
                      tma_bar_ptr=mbar)


# --------------------------------------------------------------- attn kernel --
# Operand views (D0.14 through-connection):
#   tQ   (S_Q*H_Q, D_QK) bf16 — the TMA COORDINATE tensor for q_attn (never the
#                               raw gmem tensor: TMA addresses via descriptor).
#                               Row token*H_Q + h; pass p takes the 64-row tile
#                               `token*NPASS + p`.
#   mK   (S_KV,    D_QK) bf16 — cp.async.cg gather source, one 1 KiB row per key
#   mLse (S_Q,     H_Q)  f32  — read straight from the input, no contiguous /
#                               dtype copy (D0.1); ·log2e in-kernel per pass (D0.4)
@cute.kernel
def _attn_kernel(tma_atom, tQ, mK, mLse, mIdx, mTl, mOut, tiled_g2s, tiled_mma,
                 sk_layout, sq_layout, sqb_layout,
                 topk: cutlass.Constexpr, cscale: cutlass.Constexpr,
                 s_kv: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --- shared prelude (before the single role branch; warp_specialization) --
    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(AttnStorage)
    sSums = st.sSums.get_tensor(cute.make_layout(MAX_TOPK))
    sRed = st.sRed.get_tensor(cute.make_layout(WG))
    sLse = st.sLse.get_tensor(cute.make_layout(H_HALF))
    # two views of the same 64 KiB: the MMA descriptor reads (H_HALF, D_QK),
    # the TMA writes (H_HALF, 64, NBOX_Q).  _check_box_view pins them equal.
    sQ = st.sQ.get_tensor(sq_layout.outer, swizzle=sq_layout.inner)
    sQb = st.sQ.get_tensor(sqb_layout.outer, swizzle=sqb_layout.inner)
    sK = st.sK.get_tensor(sk_layout.outer, swizzle=sk_layout.inner)
    mbar = st.sBar.data_ptr()

    tl = mTl[bidx]
    n_tiles = (tl + (KTILE - 1)) // KTILE
    ntk = n_tiles * KTILE

    # D0.5: warp 0 also does the TMA descriptor prefetch and the mbarrier init.
    # D0.11: ONE mbarrier for the Q arrival, arrive count 1 (the elected
    # producer thread), 2 phases = 2 passes.  Init must be fenced and separated
    # from first use by a CTA sync (KB elect_one gotcha 8).
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()

    cute.arch.sync_threads()

    # --- the one role branch (D0.5): WG0 = copy-producer, WG1 = mma-consumer --
    if tidx < WG:
        nrow, ngrp, nlane, ncp = ATTN_CP
        grp = tidx // nlane
        glane = tidx % nlane

        for p in cutlass.range_constexpr(NPASS):
            # D0.11: Q half-slab p by TMA (8 boxes, one expect_tx(64 KiB)); the
            # pass's 64 lse values ride the producer's own LDG -> STS.  The
            # producer waits on the mbarrier too, so the lse move and the pass
            # rendezvous both sit behind Q's arrival.  Safe against the pass-1
            # overwrite: the consumer's last BAR 0xc of pass p-1 lands after
            # its wgmma wait_group(0), so nothing is still reading sQ here.
            _tma_slab(tma_atom, tQ, sQb, mbar, bidx * NPASS + p, warp_idx,
                      H_HALF, QBYTES, NBOX_Q)
            if tidx < H_HALF:
                sLse[tidx] = mLse[(bidx, p * H_HALF + tidx)]
            cute.arch.mbarrier_wait(mbar, p)
            cute.arch.barrier(barrier_id=BAR_PASS, number_of_threads=NTHREADS)

            # --- prologue: warm tile 0 into stage 0, then rendezvous (D0.10) --
            _gather_tile(sK[(None, None, 0)], mK, mIdx, bidx, 0, grp, glane,
                         tiled_g2s, ntk, s_kv, topk, nrow, ngrp, nlane, ncp)
            _cpa_commit()
            _cpa_wait(0)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)

            # --- steady state: gather tile i+1 into stage (i+1)%2 -------------
            # lookahead = 1 tile = ring depth 2; BAR 0xc is both the full
            # signal for tile i and the empty signal for stage (i+1)%2 (D0.9).
            for i in cutlass.range(n_tiles):
                _gather_tile(sK[(None, None, (i + 1) % 2)], mK, mIdx, bidx,
                             (i + 1) * KTILE, grp, glane, tiled_g2s, ntk,
                             s_kv, topk, nrow, ngrp, nlane, ncp)
                _cpa_commit()
                _cpa_wait(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=BAR_RING,
                                  number_of_threads=NTHREADS)

        cute.arch.barrier(barrier_id=BAR_EXIT, number_of_threads=NTHREADS)
    else:
        tc = tidx - WG
        warp = tc // 32
        lane = tc % 32
        # D0.6 wgmma f32 accumulator map (certified bit-exact by the step-2
        # probe): element e of thread (warp, lane) is
        #   row(key)  = warp*16 + lane//4 + 8*((e%4)//2)
        #   col(head) = (lane%4)*2 + (e%2) + 8*(e//4)
        # -> 2 keys x 16 heads per thread, the 4 lanes of a quad splitting the
        # 64 heads of a key, which is why the head sum is 16 in-thread adds
        # plus a quad butterfly (D0.12).
        rb = warp * 16 + lane // 4
        cb = (lane % 4) * 2
        leader = (lane % 4) == 0

        wgm = tiled_mma.get_slice(0)                    # A/B are warpgroup-scope
        dB = wgm.make_fragment_B(wgm.partition_B(sQ))   # loop invariant
        acc = _MK_RMEM(wgm.partition_shape_C((KTILE, H_HALF)), cutlass.Float32)
        atom = cute.make_mma_atom(tiled_mma.op)
        nk_blk = cute.size(
            wgm.make_fragment_A(wgm.partition_A(sK[(None, None, 0)])).shape[2])

        for p in cutlass.range_constexpr(NPASS):
            cute.arch.barrier(barrier_id=BAR_PASS, number_of_threads=NTHREADS)
            # D0.11: the consumer waits on the Q mbarrier as well -- it cannot
            # issue MMA against a half-arrived B operand.  The mbarrier
            # completion is what makes the TMA's async-proxy writes visible to
            # the wgmma descriptor read, so no extra fence is owed here.
            cute.arch.mbarrier_wait(mbar, p)

            # D0.4: lse·log2e is pre-multiplied once per pass, in-kernel (no
            # preprocessing kernel), and stays resident for the 16 columns
            # this thread owns.
            lse2 = []
            for j in cutlass.range_constexpr(16):
                lse2.append(sLse[cb + (j % 2) + 8 * (j // 2)] * LOG2E)

            cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)
            for i in cutlass.range(n_tiles):
                # 0  ARRIVE (wgmma fence) — no wait at the loop top: tile i was
                #    handed over by the PREVIOUS round's BAR 0xc (D0.10).
                stg = cute.arch.make_warp_uniform(i % 2)
                dA = wgm.make_fragment_A(
                    wgm.partition_A(sK[(None, None, stg)]))
                warpgroup.fence()
                # 1  32 x HGMMA.64x64x16 SS; first step scale-d = 0, the
                #    accumulator never carries across tiles (D0.6).
                atom.set(warpgroup.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(nk_blk):
                    cute.gemm(atom, acc, dA[None, None, kb],
                              dB[None, None, kb], acc)
                    atom.set(warpgroup.Field.ACCUMULATE, True)
                # 2  commit -> wait_group 0: the chain is fully drained before
                #    the elementwise segment (depth 0 is the D0.10 contract;
                #    D1.5 is the lever that raises it).
                warpgroup.commit_group()
                warpgroup.wait_group(0)

                # 3  elementwise segment: FFMA(acc·c − lse2) -> exp2 -> column
                #    sum -> quad butterfly -> predicated scalar STS (D0.4/D0.12).
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(32):
                    pv = cute.math.exp2(acc[e] * cscale
                                        - lse2[(e // 4) * 2 + (e % 2)],
                                        fastmath=True)
                    if (e // 2) % 2 == 0:
                        s0 = s0 + pv
                    else:
                        s1 = s1 + pv
                s0 = s0 + cute.arch.shuffle_sync_bfly(s0, 1)
                s0 = s0 + cute.arch.shuffle_sync_bfly(s0, 2)
                s1 = s1 + cute.arch.shuffle_sync_bfly(s1, 1)
                s1 = s1 + cute.arch.shuffle_sync_bfly(s1, 2)

                # pass 2 reads back what pass 1 left (D0.3: the head sums are
                # what crosses the pass boundary, and smem is the only legal
                # exchange surface for it -- D0.12's `smem` ruling).
                k0 = i * KTILE + rb
                k1 = k0 + 8
                if p == 0:
                    if leader & (k0 < tl):
                        sSums[k0] = s0
                    if leader & (k1 < tl):
                        sSums[k1] = s1
                else:
                    if leader & (k0 < tl):
                        sSums[k0] = sSums[k0] + s0
                    if leader & (k1 < tl):
                        sSums[k1] = sSums[k1] + s1

                # 4  release stage i%2 and take delivery of tile i+1 (D0.9).
                cute.arch.barrier(barrier_id=BAR_RING,
                                  number_of_threads=NTHREADS)

        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)

        # --- epilogue (D0.12): 8 keys per thread at stride 128, mask at the
        # score layer, denominator once, pad stored as an exact zero. ---
        nk = topk // WG
        vals = []
        part = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(nk):
            key = j * WG + tc
            iv = mIdx[(bidx, key)]
            sv = cutlass.Float32(0.0)
            if (key < tl) & (iv >= 0):
                sv = sSums[key]
            vals.append(sv)
            part = part + sv

        sRed[tc] = part
        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)
        for k in cutlass.range_constexpr(7):
            stp = 64 >> k
            if tc < stp:
                sRed[tc] = sRed[tc] + sRed[tc + stp]
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)

        # den == 0 rows: 0 * (1/1e-10) is an exact zero, never a NaN (D0.4).
        inv = 1.0 / (sRed[0] + 1.0e-10)
        for j in cutlass.range_constexpr(nk):
            key = j * WG + tc
            mOut[(bidx, key)] = vals[j] * inv

        cute.arch.barrier(barrier_id=BAR_EXIT, number_of_threads=NTHREADS)


def _make_attn_entry(s_q, topk, cscale, s_kv):
    """Entry factory: the shape selectors are CLOSED OVER, never entry formals.

    A `Constexpr` formal on a `@cute.jit` entry is baked into the artifact but
    is absent from its runtime parameter table, so passing it again at call
    time shifts every later argument by one slot and `stream` lands in a slot
    the executor tries to turn into a tensor pointer -- "cannot be converted to
    pointer", raised at launch with a perfectly clean compile.  `Constexpr` on
    the *kernel* is unaffected.
    (hw_kb/cutedsl/sm90/uniform_reg_budget.md, hw_kb/cutedsl/sm90/wgmma_desc_uniform.md #6)
    """
    # distinct qualnames: the two entries carry the SAME tensor signature, and
    # a compile-cache / dump-name key built from the signature alone would let
    # the second compile hand back the first artifact (same KB entries).
    # cscale = sm_scale·log2e is folded on the HOST (D0.4) and rides in as a
    # closed-over constant rather than a preprocessing kernel (D0.1 keeps
    # kernel_count == 2); it is part of the compile-cache key.
    @cute.jit
    def _attn_entry(mQ, mK, mLse, mIdx, mTl, mOut, stream):
        sk_layout, sq_layout, sqb_layout = _smem_layouts(D_QK, H_HALF)
        # D0.11: one G2S bulk-tensor atom over the (S_Q*H_Q, D_QK) view; the
        # box is one SW128 column block.  The kernel takes the coordinate
        # tensor the builder returns, never mQ itself.
        tma_atom, tQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ,
            cute.slice_(sqb_layout, (None, None, 0)), (H_HALF, TMA_BOX))
        _attn_kernel(tma_atom, tQ, mK, mLse, mIdx, mTl, mOut,
                     _g2s(D_QK // 8), _mma(H_HALF), sk_layout, sq_layout,
                     sqb_layout, topk, cscale, s_kv).launch(
            grid=[s_q, 1, 1], block=[NTHREADS, 1, 1], stream=stream)
    return _attn_entry


# ---------------------------------------------------------------- idx kernel --
# Operand views (same discipline as the attn kernel):
#   tQi (S_Q*H_IDX, D_IDX) bf16 — the TMA coordinate tensor for the Qi slab
#                                 (D0.13), 256 B rows, single pass, 2 boxes
#   mKi (S_KV,      D_IDX) bf16 — cp.async.cg gather source (D0.13 producer)
#   mW  (S_Q,       H_IDX) bf16 — weights read straight from the input, no
#                                 dtype copy (D0.1); unpacked to f32 in-kernel
@cute.kernel
def _idx_kernel(tma_atom, tQi, mKi, mW, mIdx, mTl, mOut, tiled_g2s, tiled_mma,
                sk_layout, sq_layout, sqb_layout,
                topk: cutlass.Constexpr, s_kv: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(IdxStorage)
    sScores = st.sScores.get_tensor(cute.make_layout(MAX_TOPK))
    sRed = st.sRed.get_tensor(cute.make_layout(WG))
    sW = st.sW.get_tensor(cute.make_layout(H_IDX))
    sQi = st.sQi.get_tensor(sq_layout.outer, swizzle=sq_layout.inner)
    sQib = st.sQi.get_tensor(sqb_layout.outer, swizzle=sqb_layout.inner)
    sKi = st.sKi.get_tensor(sk_layout.outer, swizzle=sk_layout.inner)
    mbar = st.sBar.data_ptr()

    tl = mTl[bidx]
    n_tiles = (tl + (KTILE - 1)) // KTILE
    ntk = n_tiles * KTILE

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom)
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()

    cute.arch.sync_threads()

    if tidx < WG:
        nrow, ngrp, nlane, ncp = IDX_CP
        grp = tidx // nlane
        glane = tidx % nlane

        # single pass — 64 heads x 128 d fit resident (D0.13); D0.11 form:
        # 2 TMA boxes, one expect_tx(16 KiB), once per CTA.
        _tma_slab(tma_atom, tQi, sQib, mbar, bidx, warp_idx,
                  H_IDX, QIBYTES, NBOX_QI)
        if tidx < H_IDX:
            sW[tidx] = mW[(bidx, tidx)]
        cute.arch.mbarrier_wait(mbar, 0)
        cute.arch.barrier(barrier_id=BAR_PASS, number_of_threads=NTHREADS)

        _gather_tile(sKi[(None, None, 0)], mKi, mIdx, bidx, 0, grp, glane,
                     tiled_g2s, ntk, s_kv, topk, nrow, ngrp, nlane, ncp)
        _cpa_commit()
        _cpa_wait(0)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)

        for i in cutlass.range(n_tiles):
            _gather_tile(sKi[(None, None, (i + 1) % 2)], mKi, mIdx, bidx,
                         (i + 1) * KTILE, grp, glane, tiled_g2s, ntk,
                         s_kv, topk, nrow, ngrp, nlane, ncp)
            _cpa_commit()
            _cpa_wait(0)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)

        cute.arch.barrier(barrier_id=BAR_EXIT, number_of_threads=NTHREADS)
    else:
        tc = tidx - WG
        warp = tc // 32
        lane = tc % 32
        rb = warp * 16 + lane // 4
        cb = (lane % 4) * 2
        leader = (lane % 4) == 0
        zero = cutlass.Float32(0.0)

        wgm = tiled_mma.get_slice(0)
        dB = wgm.make_fragment_B(wgm.partition_B(sQi))
        acc = _MK_RMEM(wgm.partition_shape_C((KTILE, H_IDX)), cutlass.Float32)
        atom = cute.make_mma_atom(tiled_mma.op)
        nk_blk = cute.size(
            wgm.make_fragment_A(wgm.partition_A(sKi[(None, None, 0)])).shape[2])

        cute.arch.barrier(barrier_id=BAR_PASS, number_of_threads=NTHREADS)
        cute.arch.mbarrier_wait(mbar, 0)          # D0.11: Qi has landed
        # weights unpacked to f32 once (the indexer scale is already folded in
        # -- multiplying by sm_scale here would be the §1.6 #4 bug).
        wv = []
        for j in cutlass.range_constexpr(16):
            wv.append(sW[cb + (j % 2) + 8 * (j // 2)].to(cutlass.Float32))

        cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)
        for i in cutlass.range(n_tiles):
            stg = cute.arch.make_warp_uniform(i % 2)
            dA = wgm.make_fragment_A(wgm.partition_A(sKi[(None, None, stg)]))
            warpgroup.fence()
            atom.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(nk_blk):   # K = 128 -> 8 HGMMA
                cute.gemm(atom, acc, dA[None, None, kb],
                          dB[None, None, kb], acc)
                atom.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)

            s0 = cutlass.Float32(0.0)
            s1 = cutlass.Float32(0.0)
            for e in cutlass.range_constexpr(32):
                rv = _FMAX(acc[e], zero)                  # relu, f32 (FMNMX)
                cv = rv * wv[(e // 4) * 2 + (e % 2)]
                if (e // 2) % 2 == 0:
                    s0 = s0 + cv
                else:
                    s1 = s1 + cv
            s0 = s0 + cute.arch.shuffle_sync_bfly(s0, 1)
            s0 = s0 + cute.arch.shuffle_sync_bfly(s0, 2)
            s1 = s1 + cute.arch.shuffle_sync_bfly(s1, 1)
            s1 = s1 + cute.arch.shuffle_sync_bfly(s1, 2)

            k0 = i * KTILE + rb
            k1 = k0 + 8
            if leader & (k0 < tl):
                sScores[k0] = s0
            if leader & (k1 < tl):
                sScores[k1] = s1

            cute.arch.barrier(barrier_id=BAR_RING, number_of_threads=NTHREADS)

        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)

        # --- epilogue (D0.13): row max in the raw domain over the valid
        # prefix, exp2 with the log2e fold, pad stored as an exact zero. ---
        nk = topk // WG
        vals = []
        oks = []
        m = cutlass.Float32(PARK)
        for j in cutlass.range_constexpr(nk):
            key = j * WG + tc
            iv = mIdx[(bidx, key)]
            ok = (key < tl) & (iv >= 0)
            sv = cutlass.Float32(PARK)
            if ok:
                sv = sScores[key]
            vals.append(sv)
            oks.append(ok)
            m = _FMAX(m, sv)

        sRed[tc] = m
        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)
        for k in cutlass.range_constexpr(7):
            stp = 64 >> k
            if tc < stp:
                sRed[tc] = _FMAX(sRed[tc], sRed[tc + stp])
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)

        mrow = sRed[0]
        # all-invalid row: park guard keeps exp2 underflowing to a clean zero
        mu = cutlass.Float32(0.0)
        if mrow > NEG_TEST:
            mu = mrow

        evs = []
        part = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(nk):
            ev = cute.math.exp2((vals[j] - mu) * LOG2E, fastmath=True)
            evs.append(ev)
            part = part + ev

        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)
        sRed[tc] = part
        cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)
        for k in cutlass.range_constexpr(7):
            stp = 64 >> k
            if tc < stp:
                sRed[tc] = sRed[tc] + sRed[tc + stp]
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=WG)

        inv = 1.0 / (sRed[0] + 1.0e-30)
        for j in cutlass.range_constexpr(nk):
            key = j * WG + tc
            o = cutlass.Float32(0.0)
            if oks[j]:
                o = evs[j] * inv
            mOut[(bidx, key)] = o

        cute.arch.barrier(barrier_id=BAR_EXIT, number_of_threads=NTHREADS)


def _make_idx_entry(s_q, topk, s_kv):
    """Same closure discipline as `_make_attn_entry`."""
    @cute.jit
    def _idx_entry(mQi, mKi, mW, mIdx, mTl, mOut, stream):
        sk_layout, sq_layout, sqb_layout = _smem_layouts(D_IDX, H_IDX)
        tma_atom, tQi = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQi,
            cute.slice_(sqb_layout, (None, None, 0)), (H_IDX, TMA_BOX))
        _idx_kernel(tma_atom, tQi, mKi, mW, mIdx, mTl, mOut,
                    _g2s(D_IDX // 8), _mma(H_IDX), sk_layout, sq_layout,
                    sqb_layout, topk, s_kv).launch(
            grid=[s_q, 1, 1], block=[NTHREADS, 1, 1], stream=stream)
    return _idx_entry


# ------------------------------------------------------------------ host side --
_COMPILED = {}
_STREAMS = {}
_FULL_TL = {}


def _stream():
    """torch's current stream as a CUstream (the bench times events on it)."""
    raw = torch.cuda.current_stream().cuda_stream
    got = _STREAMS.get(raw)
    if got is None:
        try:
            import cuda.bindings.driver as _cuda
        except ImportError:                                  # older bindings
            from cuda import cuda as _cuda
        got = _cuda.CUstream(raw)
        _STREAMS[raw] = got
    return got


def _compiled(s_q, topk, cscale, s_kv, attn_args, idx_args):
    # cscale is a closed-over constant, so it belongs in the cache key: two
    # calls at the same shape with a different sm_scale must not share an
    # artifact that baked the first one in.  s_kv is the gather bound.
    key = (s_q, topk, cscale, s_kv)
    got = _COMPILED.get(key)
    if got is None:
        # entry args are tensors + stream ONLY -- see _make_attn_entry
        got = (cute.compile(_make_attn_entry(s_q, topk, cscale, s_kv),
                            *attn_args),
               cute.compile(_make_idx_entry(s_q, topk, s_kv), *idx_args))
        _COMPILED[key] = got
    return got


def dsa_score_recompute(q_attn, k_attn, lse, q_idx, k_idx, weights,
                        topk_indices, sm_scale, topk_length=None):
    s_q = topk_indices.shape[1]
    topk = topk_indices.shape[2]
    s_kv = k_attn.shape[1]
    dev = topk_indices.device

    if topk % WG != 0 or topk > MAX_TOPK:
        raise ValueError("topk %d unsupported (need a multiple of %d, <= %d)"
                         % (topk, WG, MAX_TOPK))
    # the depths / head counts are baked into the smem map (D0.7), the MMA K
    # extent (D0.6) and the pass count (D0.3)
    if q_attn.shape[3] != D_QK or q_idx.shape[3] != D_IDX:
        raise ValueError("depths (%d, %d) unsupported (built for %d / %d)"
                         % (q_attn.shape[3], q_idx.shape[3], D_QK, D_IDX))
    if q_attn.shape[2] != H_Q or q_idx.shape[2] != H_IDX:
        raise ValueError("head counts (%d, %d) unsupported (built for %d / %d)"
                         % (q_attn.shape[2], q_idx.shape[2], H_Q, H_IDX))

    target = torch.empty((1, s_q, topk), device=dev, dtype=torch.float32)
    predict = torch.empty((1, s_q, topk), device=dev, dtype=torch.float32)

    if topk_length is None:
        tl_key = (s_q, topk, dev.index)
        tl = _FULL_TL.get(tl_key)
        if tl is None:
            tl = torch.full((s_q,), topk, device=dev, dtype=torch.int32)
            _FULL_TL[tl_key] = tl
    else:
        tl = topk_length.view(s_q)

    # Q is flattened to the (token*H + h, D) 2D view the TMA descriptor is
    # built over (D0.11 / D0.13): one 64-row tile per (token, pass), and the
    # box's 64-column extent stays a separate descriptor axis because it is a
    # strict divisor of D.  Everything else keeps its natural 2D shape.  No
    # contiguous / dtype copy anywhere on this path -- D0.1 keeps the launch
    # count at exactly 2 kernels.
    mQ = from_dlpack(q_attn.view(s_q * H_Q, D_QK), assumed_align=16)
    mK = from_dlpack(k_attn.view(-1, D_QK), assumed_align=16)
    mLse = from_dlpack(lse.view(s_q, H_Q), assumed_align=16)
    mQi = from_dlpack(q_idx.view(s_q * H_IDX, D_IDX), assumed_align=16)
    mKi = from_dlpack(k_idx.view(-1, D_IDX), assumed_align=16)
    mW = from_dlpack(weights.view(s_q, H_IDX), assumed_align=16)
    mIdx = from_dlpack(topk_indices.view(s_q, topk), assumed_align=16)
    mTl = from_dlpack(tl, assumed_align=16)
    mTarget = from_dlpack(target.view(s_q, topk), assumed_align=16)
    mPredict = from_dlpack(predict.view(s_q, topk), assumed_align=16)

    stream = _stream()
    cscale = float(sm_scale) * LOG2E          # D0.4 host fold, one f32 constant
    attn_args = (mQ, mK, mLse, mIdx, mTl, mTarget, stream)
    idx_args = (mQi, mKi, mW, mIdx, mTl, mPredict, stream)
    attn, idx = _compiled(s_q, topk, cscale, s_kv, attn_args, idx_args)
    attn(*attn_args)
    idx(*idx_args)
    return target, predict
