"""conv2d_wgrad kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):
  conv2d_wgrad(x, dy) -> dw
    x:  torch.Tensor (N, C, H, W) cuda bfloat16, channels_last memory format
    dy: torch.Tensor (N, K, P, Q) cuda bfloat16, channels_last memory format
    dw: torch.Tensor (K, C, R, S) cuda bfloat16, channels_last memory format
  N=128, C=256, H=W=28, K=256, R=S=3, stride=1, padding=1, dilation=1,
  groups=1, no bias  ->  P=Q=28.  fp32 accumulation inside.

channels_last means the underlying memory is NHWC / NPQK / KRSC; cute-DSL code
should treat the innermost (contiguous) dimension as the channel.

This is the WEIGHT-GRADIENT of the convolution: every dw element is a sum of
N*P*Q = 100,352 products — the reduction axis is the huge one (images x output
positions), while the output is tiny (K*C*R*S = 589,824 elements).  As an
implicit GEMM: dw[K, C*R*S] = dy^T[K, N*P*Q] @ im2col(x)[N*P*Q, C*R*S] with
M=256, N=2304, reduction depth 100,352.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc.  Delegating the math to torch / library conv, grad, or matmul calls fails
the harness (eval/bench.py scans kernel.py for the forbidden call families).
JIT-compile once and cache the compiled kernel across calls (the bench times
repeated calls).
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

# ------------------------------------------------------------------ geometry
# Problem (shape A, frozen by the harness).
NIMG, CCH, HW = 128, 256, 28          # N, C, H = W = P = Q
KCH, RF, SF = 256, 3, 3
PAD = 1

# D0.4 tile shape: bM = M (all 256 dY channels), bN = 128 (one c-half of one
# (r, s) tap), bK = 64 reduction points per stage.
BM, BN, BK = 256, 128, 64
# D0.6 K-axis geometry: bK is built from RECT spatial rectangles of BH x BW
# output positions inside a single image.  4x4 is the only (BH, BW) that
# divides 28, stays strictly below it (so the halo really is out of bounds on
# each axis) and makes BH*BW a multiple of the WGMMA k-step 16.
BH, BW = 4, 4
RECT = BK // (BH * BW)                # 4 rectangles per stage
RPI = (HW // BH) * (HW // BW)         # 49 rectangles per image
RCOL = HW // BW                       # 7 rectangle columns per image

# D0.1/D0.3 work decomposition: 18 j-tiles x 7 k-segments = 126 CTAs, one wave.
JTILES = (CCH * RF * SF) // BN        # 18
SEGS = 7
GRID = JTILES * SEGS                  # 126
KTILES = (NIMG * HW * HW) // SEGS // BK   # 224 stages per segment
RECTS_PER_SEG = KTILES * RECT         # 896
BACKEDGES = (KTILES - 2) // 2         # 111 back-edges, 2 stages each

# D0.5 warp roles / register split.
NUM_WG = 2                            # consumer warpgroups
CONS_THREADS = NUM_WG * 128           # 256
NTHREADS = CONS_THREADS + 128         # 384 = 1 producer WG + 2 consumer WG
PROD_REGS, CONS_REGS = 40, 232        # 128*40 + 256*232 = 64512 <= 65536

# D0.7 cluster: ctaid pairs (2t, 2t+1) are the two c-halves of the same (r, s)
# tap, so both CTAs of a cluster want the SAME dY tile -> A is multicast, B is
# not.  126 = 63 clusters, and 18 j-tiles per segment is even, so a cluster
# never straddles a k-segment.  The cluster axis is the GEMM's N axis (the two
# CTAs differ only in c_half), i.e. mode 1 of cta_layout_mnk.
CLUSTER = 2
# Each consumer warp of EVERY cluster CTA signals one empty-barrier arrival to
# each CTA in the cluster, so a slot is refilled only after both consumers of
# the multicast are done with it.
EMPTY_ARRIVES = (CONS_THREADS // 32) * CLUSTER   # 16

# D0.9 accumulator organisation: 2 chains per consumer WG, each chain a
# 64x128 fp32 accumulator (64 regs/thread) -> 2 x 2 x 64 rows = 256 = bM.
CHAIN = 2
MCH = BM // CHAIN                     # 128 rows per chain (64 per warpgroup)
CHUNK_ELEMS = (MCH // NUM_WG) * BN    # 8192 fp32 per (wg, chain) chunk
CHUNK_BYTES = CHUNK_ELEMS * 4         # 32768 B

# D0.8 smem budget.  One mbarrier per stage covers A and B together.
STAGE = 4
TX_BYTES = (BM * BK + BN * BK) * 2    # 49152 = 0xc000
SA_ELEMS = BM * BK * STAGE            # bf16
SB_ELEMS = BN * BK * STAGE            # bf16
# mbarrier + parameter face is padded to exactly 2048 B so that
# smem_dynamic == STAGE*49152 + 32768 + 2048.
MBAR_SLOTS = 2 * STAGE                # Int64
PAD_SLOTS = (2048 - MBAR_SLOTS * 8) // 8

# ------------------------------------------------------------------ M1 levers
# The flow levers of M1 (D1.1 stage count, D1.2 WGMMA wait depth, D1.3 smem
# swizzle) are pinned here rather than left implicit in the code.  They share
# one budget -- the smem identity STAGE*49152 + 34816 = smem_dynamic <= 232448
# -- and one knob: the ring's release lag.  Exactly one commit group stays in
# flight per back-edge, so the consumer holds exactly one slot and the producer
# sees STAGE-1 free ones; every lever below is scored against that lag.
#
# D1.1 stage count S = STAGE = 4.  This is the hard cap of the ring, not a
#   preference: S=5 needs 5*49152 + 34816 = 280576 B against a 232448 B
#   per-CTA dynamic smem limit.  The other admissible value, S=3, moves the
#   wrong way -- the M0 gate measured stall_long_sb 6.37 against the
#   reference's 4.17, i.e. the consumer is already waiting on data, and S=3
#   would cut the producer lead from 3 stages to 2.
#
# D1.2 WGMMA wait depth n = 1.  A back-edge consumes 2 stages, commits 2
#   groups and releases 2 slots.  At the wait there are 3 groups outstanding,
#   so n=1 retires exactly the 2 whose slots are about to be handed back and
#   keeps the newest group in flight over the releases and the next
#   consumer_wait.  n=2 retires only one group per back-edge, so releasing two
#   slots would hand the producer a slot whose WGMMA is still reading it;
#   staying safe costs one more slot of lag and drops the producer lead to 1
#   stage.  n=0 drains the tensor pipe at each of the 112 back-edges.
WAIT_DEPTH = 1
# D1.3 swizzle mode = 128B on both faces.  A and B are both MN-major bf16 with
#   major modes of BM=256 / BN=128 elements = 4096 / 2048 bits, both multiples
#   of the 1024-bit SW128 period, and the atom's 8-row period divides the 16
#   rows one TMA box occupies (D0.6).  The M0 gate measured bankconf_st 0 and
#   smem_wavefronts_st 129024 -- exactly the structural lower bound of D0.8 --
#   so the narrower 64B mode has no conflict left to recover.  Both faces must
#   use the same mode on the TMA and the WGMMA descriptor side.
SWIZZLE_MODE = warpgroup.SmemLayoutAtomKind.MN_SW128

# scratch ABI (D0.2, frozen): [18 j-tiles][4 chunks][8192 fp32], each block
# contiguous, block bytes laid out in the chunk's flat-TV order.
SCRATCH_BLOCKS = JTILES * 4


# --------------------------------------------------------- inline PTX (D0.13)
# The non-tensor cp.reduce.async.bulk S2G ADD is not wrapped by DSL 4.4.2.
@dsl_user_op
def _bulk_reduce_add_f32(dst_addr, src_addr, nbytes, *, loc=None, ip=None):
    """dst_addr: Int64 (global), src_addr: Int32 (shared::cta).  One thread."""
    llvm.inline_asm(
        None,
        [dst_addr.ir_value(loc=loc, ip=ip),
         src_addr.ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_commit_group(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_group_read(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [],
                    "cp.async.bulk.wait_group.read {};".format(int(n)), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# ------------------------------------------------------------------- layouts
def _layouts():
    """MN-major smem allocation + the TMA / WGMMA views composed onto it.

    A (dY) is M-major and B (X) is N-major: dY's memory order is NPQK and X's
    is NHWC, so the GEMM's M and N axes are the contiguous ones on both sides.
    The buffer is allocated once in the MN-major form that is legal for both
    TMA and WGMMA, and every other view is a composition of it (SWIZZLE_MODE,
    the D1.3 lever, rides along on all of them, which is what keeps the TMA
    and WGMMA descriptor sides in the same mode by construction).  Transposer
    strides are the colex coefficients of the allocation domain
    (idx = mn + k*MN + s*MN*bK) written in the reading view's coordinate
    order, so a rectangle rc lands on k = rc*16 + h*BW + w.
    MUST be called from inside @cute.jit.
    """
    bf16 = cutlass.BFloat16
    atom_a = warpgroup.make_smem_layout_atom(SWIZZLE_MODE, bf16)
    atom_b = warpgroup.make_smem_layout_atom(SWIZZLE_MODE, bf16)
    base_a = cute.tile_to_shape(atom_a, (BM, BK, STAGE), order=(1, 0, 2))
    base_b = cute.tile_to_shape(atom_b, (BN, BK, STAGE), order=(1, 0, 2))
    a_tma = cute.composition(base_a, cute.make_layout(
        (1, BH, BW, BM, RECT * STAGE),
        stride=(0, BW * BM, BM, 1, BH * BW * BM)))
    a_mma = cute.composition(base_a, cute.make_layout(
        (MCH, BK, STAGE, CHAIN), stride=(1, BM, BM * BK, MCH)))
    b_tma = cute.composition(base_b, cute.make_layout(
        (1, BH, BW, BN, RECT * STAGE),
        stride=(0, BW * BN, BN, 1, BH * BW * BN)))
    return base_a, base_b, a_tma, a_mma, b_tma


def _storage():
    @cute.struct
    class SharedStorage:
        mbar: cute.struct.MemRange[cutlass.Int64, MBAR_SLOTS]
        pad: cute.struct.MemRange[cutlass.Int64, PAD_SLOTS]
        sA: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, SA_ELEMS], 1024]
        sB: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, SB_ELEMS], 1024]
        sEpi: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, CHUNK_ELEMS], 1024]
    return SharedStorage


def _tiled_mma():
    return sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.MN,      # A (dY) is M-major
        warpgroup.OperandMajorMode.MN,      # B (X)  is N-major
        cutlass.Float32,
        atom_layout_mnk=(NUM_WG, 1, 1),     # 2 warpgroups stacked along M
        tiler_mn=(64, BN))                  # the WGMMA atom M-mode is always 64


def _epi_tv():
    """Flat-TV bijection for the C-fragment <-> staging face round trip:
    one 128-bit STS per thread, zero bank conflicts."""
    f32 = cutlass.Float32
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), f32,
                            num_bits_per_copy=128),
        cute.make_layout((128, 1)),
        cute.make_layout(128 // f32.width))


def _epi_layout():
    return cute.make_layout((CHUNK_ELEMS, 1), stride=(1, CHUNK_ELEMS))


# --------------------------------------------------------------- main kernel
@cute.kernel
def _wgrad_main(tma_a: cute.CopyAtom, mY: cute.Tensor,
                tma_b: cute.CopyAtom, mX: cute.Tensor,
                mS: cute.Tensor, tiled_mma: cute.TiledMma,
                base_b: cute.ComposedLayout,
                a_tma_l: cute.ComposedLayout, a_mma_l: cute.ComposedLayout,
                b_tma_l: cute.ComposedLayout,
                tv: cute.TiledCopy, epi_l: cute.Layout,
                tx: cutlass.Constexpr[int],
                Storage: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # D0.3 tile map: linear decode, no lookup table.
    j_tile = bidx % JTILES
    seg = bidx // JTILES
    tap = j_tile // 2
    c_half = j_tile % 2
    r = tap // SF
    s = tap % SF

    # D0.7 cluster geometry.  bM = M makes the two CTAs of a cluster share the
    # whole A (dY) tile, so one TMA fills both smem rings; B stays per-CTA.
    cta_layout_mnk = cute.make_layout((1, CLUSTER, 1))
    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cluster_crd = cta_layout_mnk.get_flat_coord(cta_rank)
    a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_crd,
                                               mode=1)
    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    pipe = pipeline.PipelineTmaAsync.create(
        barrier_storage=st.mbar.data_ptr(), num_stages=STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 EMPTY_ARRIVES),
        tx_count=tx,
        cta_layout_vmnk=cute.make_layout((1, 1, CLUSTER, 1)))

    # Every view of the smem allocation is built HERE, before the warp-role
    # branch: a `cute.struct` instance is not a flattenable dynamic expression,
    # so calling a method on it inside a dynamic `if` makes the DSL try to
    # yield the storage object through the scf.if region and fail to trace.
    # Only the addresses cross the branch, and they are dead on the producer
    # path, so the register split is unaffected.
    sA_tma = st.sA.get_tensor(a_tma_l.outer, swizzle=a_tma_l.inner)
    sB_tma = st.sB.get_tensor(b_tma_l.outer, swizzle=b_tma_l.inner)
    sA_mma = st.sA.get_tensor(a_mma_l.outer, swizzle=a_mma_l.inner)
    sB_mma = st.sB.get_tensor(base_b.outer, swizzle=base_b.inner)
    sEpi = st.sEpi.get_tensor(epi_l)
    src_addr = sEpi.iterator.toint()
    # consumer thread ctid maps to tv thread ctid % 128 == tidx % 128
    tS = tv.get_slice(tidx % 128).partition_D(sEpi)
    tS_l = cute.make_layout(tS.shape)
    cute.arch.sync_threads()

    if warp_idx < 4:
        # ---------------------------------------------------- producer WG
        cute.arch.setmaxregister_decrease(PROD_REGS)
        # D1.4 issue granularity: the 8 TMA copies of a stage are spread over
        # the 4 producer warps, one rectangle (its A copy and its B copy) per
        # warp, instead of all 8 on warp 0.  A stage's issue window is the
        # ~1e3 cycles the consumers spend on its 16 WGMMA, and a lone warp has
        # no ILP to hide the latency of its own dependent address chain, so the
        # per-stage producer path is cut from 4 coordinate decodes + 8 copies
        # to 1 decode + 2 copies.  Coordinates stay RECOMPUTED per stage rather
        # than incrementally advanced: one rect per warp is 4 divisions by
        # compile-time constants and carries no loop-borne dependency.
        #
        # Only warp 0 acquires.  producer_acquire's expect-tx is elected per
        # WARP (MbarrierArray.arrive_and_expect_tx wraps elect_one), so four
        # callers would arm the barrier with 4 x 49152 B.  The named barrier
        # publishes the acquired slot to the other three warps before they
        # write into it.  Nothing is needed on the way out: a slot is
        # re-acquired only after its full barrier completed, which already
        # implies every warp's copy into it landed.
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
        # The (r-1, s-1) window shift is a loop invariant of the whole kernel
        # (D0.3), so im2col degenerates into one origin move; H and W stay
        # separate descriptor axes, so the halo is a real per-axis
        # out-of-bounds and the TMA unit writes the zeros itself.
        mXo = cute.domain_offset((0, r - PAD, s - PAD, 0), mX)
        ps = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, STAGE)
        rid = seg * RECTS_PER_SEG + warp_idx
        for kt in range(KTILES):
            if warp_idx == 0:
                pipe.producer_acquire(ps)
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            bar = pipe.producer_get_barrier(ps)
            rr = rid + kt * RECT
            n = rr // RPI
            rem = rr % RPI
            hb = rem // RCOL
            wb = rem % RCOL
            slot = ps.index * RECT + warp_idx
            gA = cute.local_tile(mY, (1, BH, BW, BM), (n, hb, wb, 0))
            # A is multicast: tma_partition splits the dY box across the
            # cluster, each CTA issues its share with the full mask, so one
            # gmem read lands in both CTAs' smem (D0.7).
            tAsA, tAgA = cpasync.tma_partition(
                tma_a, cluster_crd[1], a_cta_layout,
                cute.group_modes(sA_tma, 0, 4),
                cute.group_modes(gA, 0, 4))
            cute.copy(tma_a, tAgA, tAsA[(None, slot)], tma_bar_ptr=bar,
                      mcast_mask=a_mcast_mask)
            gB = cute.local_tile(mXo, (1, BH, BW, BN), (n, hb, wb, c_half))
            tBsB, tBgB = cpasync.tma_partition(
                tma_b, 0, cute.make_layout(1),
                cute.group_modes(sB_tma, 0, 4),
                cute.group_modes(gB, 0, 4))
            cute.copy(tma_b, tBgB, tBsB[(None, slot)], tma_bar_ptr=bar)
            pipe.producer_commit(ps)
            ps.advance()
        # A CTA must not exit while its cluster peer can still multicast into
        # its smem or signal its mbarriers.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
    else:
        # --------------------------------------------------- consumer WGs
        cute.arch.setmaxregister_increase(CONS_REGS)
        ctid = tidx - 128
        wg_idx = cute.arch.make_warp_uniform(ctid // 128)

        wg_mma = tiled_mma.get_slice(cute.make_layout(NUM_WG, stride=128)(wg_idx))
        tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA_mma))
        tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB_mma))
        accs = [cute.make_fragment(wg_mma.partition_shape_C((MCH, BN)),
                                   cutlass.Float32)
                for _ in range(CHAIN)]
        for ch in cutlass.range_constexpr(CHAIN):
            accs[ch].fill(0.0)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        nkb = cute.size(tCrA, mode=[2])
        warpgroup.fence()
        # D0.11 issue order: every back-edge consumes 2 K-stages = 16 WGMMA in
        # 2 commit groups, and carries exactly one deferred wait that leaves
        # one group in flight; the ring slots are released only after it, so
        # the address / ring bookkeeping runs in the WGMMA flight shadow.  No
        # other synchronisation point sits inside the loop.  The relation is
        # frozen; only its depth is the D1.2 lever (WAIT_DEPTH), and the
        # wait_group(0) below is not that lever -- it is the full drain D0.11
        # requires before the epilogue.
        cs = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGE)
        rs = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGE)
        pipe.consumer_wait(cs)
        for ch in cutlass.range_constexpr(CHAIN):
            for kb in cutlass.range_constexpr(nkb):
                cute.gemm(tiled_mma, accs[ch],
                          tCrA[(None, None, kb, cs.index, ch)],
                          tCrB[(None, None, kb, cs.index)], accs[ch])
        warpgroup.commit_group()
        cs.advance()
        for _ in range(BACKEDGES):
            pipe.consumer_wait(cs)
            for ch in cutlass.range_constexpr(CHAIN):
                for kb in cutlass.range_constexpr(nkb):
                    cute.gemm(tiled_mma, accs[ch],
                              tCrA[(None, None, kb, cs.index, ch)],
                              tCrB[(None, None, kb, cs.index)], accs[ch])
            warpgroup.commit_group()
            cs.advance()
            pipe.consumer_wait(cs)
            for ch in cutlass.range_constexpr(CHAIN):
                for kb in cutlass.range_constexpr(nkb):
                    cute.gemm(tiled_mma, accs[ch],
                              tCrA[(None, None, kb, cs.index, ch)],
                              tCrB[(None, None, kb, cs.index)], accs[ch])
            warpgroup.commit_group()
            cs.advance()
            warpgroup.wait_group(WAIT_DEPTH)
            pipe.consumer_release(rs)
            rs.advance()
            pipe.consumer_release(rs)
            rs.advance()
        pipe.consumer_wait(cs)
        for ch in cutlass.range_constexpr(CHAIN):
            for kb in cutlass.range_constexpr(nkb):
                cute.gemm(tiled_mma, accs[ch],
                          tCrA[(None, None, kb, cs.index, ch)],
                          tCrB[(None, None, kb, cs.index)], accs[ch])
        warpgroup.commit_group()
        cs.advance()
        warpgroup.wait_group(0)
        pipe.consumer_release(rs)
        rs.advance()
        pipe.consumer_release(rs)
        rs.advance()
        # Same exit guard as the producer branch: every thread of the cluster
        # arrives, so no CTA tears down its smem while its peer is still in the
        # main loop (D0.7).
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        # --------------------------------------------- epilogue (D0.13)
        # Per chunk: C-fragment -> staging face in flat-TV order (one 128-bit
        # STS per thread), then one non-tensor cp.reduce.async.bulk .add.f32
        # atomically adds the whole 32768 B block into the fp32 scratch.  No
        # bf16 conversion, no TMA store, no semaphore spin in this kernel.
        for cidx in cutlass.range_constexpr(4):
            owner = cidx // 2
            ch = cidx % 2
            if wg_idx == owner:
                cute.autovec_copy(
                    cute.make_tensor(accs[ch].iterator, tS_l), tS)
            # generic-proxy STS -> async-proxy (bulk engine) read
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=1, number_of_threads=CONS_THREADS)
            if ctid == 0:
                gblk = mS[(j_tile * 4 + cidx, None)]
                _bulk_reduce_add_f32(gblk.iterator.toint(), src_addr,
                                     CHUNK_BYTES)
                _bulk_commit_group()
                _bulk_wait_group_read(0)
            cute.arch.barrier(barrier_id=1, number_of_threads=CONS_THREADS)


# ----------------------------------------------------------- finalize kernel
@cute.kernel
def _wgrad_finalize(mS: cute.Tensor, mDW: cute.Tensor,
                    tiled_mma: cute.TiledMma, tv: cute.TiledCopy,
                    epi_l: cute.Layout):
    """One CTA per (j_tile, chunk): read the fp32 partial sums back through the
    same flat-TV bijection the main kernel wrote them with, convert to bf16 at
    the single conversion point of the whole operator, write dw, and zero the
    scratch block for the next call (D0.2's clearing invariant)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    j_tile = bidx // 4
    cidx = bidx % 4
    wg = cidx // 2
    ch = cidx % 2

    gS = cute.make_tensor(mS[(bidx, None)].iterator, epi_l)
    thr_tv = tv.get_slice(tidx)
    tG = thr_tv.partition_S(gS)
    rF = cute.make_fragment_like(tG, cutlass.Float32)
    cute.autovec_copy(tG, rF)
    zF = cute.make_fragment_like(tG, cutlass.Float32)
    zF.fill(0.0)
    cute.autovec_copy(zF, tG)

    thr_mma = tiled_mma.get_slice(wg * 128 + tidx)
    cId = thr_mma.partition_C(cute.local_tile(
        cute.make_identity_tensor((BM, BN)), (MCH, BN), (ch, 0)))
    jbase = j_tile * BN
    for i in cutlass.range_constexpr(cute.size(cId)):
        m, nn = cId[i]
        mDW[(m, jbase + nn)] = rF[i].to(cutlass.BFloat16)


# ------------------------------------------------------------------- hosts
@cute.jit
def _jit_wgrad(mY: cute.Tensor, mX: cute.Tensor, mS: cute.Tensor,
               dw_addr: cutlass.Int64, stream):
    """Both launches of one conv2d_wgrad call live in ONE compiled host
    function (D2.3): the two kernels of D0.2 are unchanged, but the Python
    side crosses into compiled code once instead of twice, so the argument
    walk, the packed-arg build and the capi trampoline are paid once."""
    base_a, base_b, a_tma_l, a_mma_l, b_tma_l = _layouts()
    one_a = cute.slice_(a_tma_l, (None, None, None, None, 0))
    one_b = cute.slice_(b_tma_l, (None, None, None, None, 0))
    tma_a, tY = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(), mY, one_a, (1, BH, BW, BM),
        num_multicast=CLUSTER)
    tma_b, tX = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mX, one_b, (1, BH, BW, BN))
    tiled_mma = _tiled_mma()
    tv = _epi_tv()
    epi_l = _epi_layout()
    _wgrad_main(tma_a, tY, tma_b, tX, mS, tiled_mma, base_b,
                a_tma_l, a_mma_l, b_tma_l, tv, epi_l,
                TX_BYTES, _storage()).launch(
        grid=[GRID, 1, 1], block=[NTHREADS, 1, 1], cluster=[CLUSTER, 1, 1],
        min_blocks_per_mp=1, stream=stream)
    # dw is a fresh allocation on every call, so it enters as an ADDRESS rather
    # than as a dlpack tensor.  A cached cute tensor view would hold the dlpack
    # capsule, hence the buffer, alive: the caching allocator would then have
    # to hand out a different address next call, which turns the view cache
    # into a permanent miss and grows without bound.  Only the base pointer is
    # dynamic here -- the layout is the compile-time constant of section 1.2
    # (dw KRSC-contiguous == row-major [256, 2304], column j = (r*S+s)*C + c).
    mDW = cute.make_tensor(
        cute.make_ptr(cutlass.BFloat16, dw_addr, cute.AddressSpace.gmem,
                      assumed_align=16),
        cute.make_layout((KCH, RF * SF * CCH), stride=(RF * SF * CCH, 1)))
    _wgrad_finalize(mS, mDW, tiled_mma, tv, epi_l).launch(
        grid=[SCRATCH_BLOCKS, 1, 1], block=[128, 1, 1], stream=stream)


# --------------------------------------------------------------- host entry
# D2.3 host launch tax.  The gate and the target are per-call END-TO-END
# windows: bench.py records a CUDA event, calls conv2d_wgrad, records again,
# so every microsecond of Python between the two records is measured device
# time's equal.  Everything that does not depend on this call's data is
# therefore built once and reused across calls -- the JIT module (which
# carries both TMA descriptor constructions and both launches), the cute
# tensor views of x / dy / scratch, and the CUstream wrapper.  Reuse is keyed
# on each tensor's device ADDRESS, never on an assumption that the address is
# stable: a pointer that moves simply misses and that one view is rebuilt.
_MACH = {}                       # (device index, dtype) -> machine
_XV, _YV = {}, {}                # data_ptr -> cute tensor view
_STREAM = [-1, None]             # raw stream handle -> CUstream wrapper
_DW_SHAPE = (KCH, CCH, RF, SF)


def _wrap_stream(raw):
    try:
        from cuda import cuda as _cuda
        return _cuda.CUstream(raw)
    except Exception:
        pass
    try:
        from cuda.bindings import driver as _drv
        return _drv.CUstream(raw)
    except Exception:
        return raw


def _raw_stream_slow(devi):
    return torch.cuda.current_stream(devi).cuda_stream


# torch's own fast path for "the current stream as a raw handle" -- one C call,
# no Stream object built.  Fall back to the object route if it is absent.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", _raw_stream_slow)


def _stream(devi):
    raw = _raw_stream(devi)
    if raw != _STREAM[0]:
        _STREAM[0] = raw
        _STREAM[1] = _wrap_stream(raw)
    return _STREAM[1]


def _build(x, dy, devi, key):
    dev = x.device
    # A view cache entry is only meaningful for the machine it was built
    # against, and a device address alone does not identify a device.
    _XV.clear()
    _YV.clear()
    scratch = torch.zeros((SCRATCH_BLOCKS, CHUNK_ELEMS), device=dev,
                          dtype=torch.float32)
    cS = from_dlpack(scratch, assumed_align=16)
    probe = torch.empty(_DW_SHAPE, device=dev, dtype=x.dtype,
                        memory_format=torch.channels_last)
    run = cute.compile(
        _jit_wgrad,
        from_dlpack(dy.permute(0, 2, 3, 1), assumed_align=16),
        from_dlpack(x.permute(0, 2, 3, 1), assumed_align=16),
        cS, probe.data_ptr(), _stream(devi))
    mach = (dev, x.dtype, scratch, cS, run)
    _MACH[key] = mach
    return mach


def _view(cache, t, addr):
    v = cache.get(addr)
    if v is None:
        if len(cache) > 4:
            cache.clear()
        # channels_last memory is NHWC / NPQK: permute to the memory order so
        # the innermost (contiguous) mode really is the channel.
        v = from_dlpack(t.permute(0, 2, 3, 1), assumed_align=16)
        cache[addr] = v
    return v


def conv2d_wgrad(x, dy):
    devi = x.get_device()
    key = (devi, x.dtype)
    mach = _MACH.get(key)
    if mach is None:
        mach = _build(x, dy, devi, key)
    dev, dt, _scratch, cS, run = mach

    xp = x.data_ptr()
    cX = _XV.get(xp)
    if cX is None:
        cX = _view(_XV, x, xp)
    yp = dy.data_ptr()
    cY = _YV.get(yp)
    if cY is None:
        cY = _view(_YV, dy, yp)

    # dw is allocated straight in its final logical form: (K, C, R, S) with
    # channels_last memory is KRSC-contiguous, which IS the row-major
    # [256, 2304] matrix the finalize kernel writes, so the returned tensor
    # needs no view/permute wrapper on the way out.
    out = torch.empty(_DW_SHAPE, device=dev, dtype=dt,
                      memory_format=torch.channels_last)
    run(cY, cX, cS, out.data_ptr(), _stream(devi))
    return out
