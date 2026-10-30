"""mla_paged_prefill — the student's kernel file (the only file the student edits).

Multi-head Latent Attention prefill forward over a paged KV cache (page size 1,
i.e. one token per page) with a causal mask, the FlashInfer-Bench definition
mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 (captured from DeepSeek-V3 / R1
incremental prefill at tensor-parallel 8): 16 query heads; every head shares
ONE latent KV row per token — the 512-wide compressed latent `ckv` (it is both
the key's "nope" half AND the value) plus a 64-wide rotary key part `kpe`;
bf16 in / bf16 out; ragged batches described by qo_indptr / kv_indptr /
kv_indices. The campaign point is one real FlashInfer-Bench workload (18
sequences, 15883 query tokens, 15937 KV tokens, page ids verbatim from the
trace). The signature below is the card's frozen contract; the student
implements the body in CuTe DSL; the signature itself may not change.

Semantics (sequence b owns query rows q0 = qo_indptr[b] .. q1 = qo_indptr[b+1]
and the KV rows page_j = kv_indices[kv_indptr[b] + j] for 0 <= j < kv_len, in
that order; q_len = q1 - q0, kv_len = kv_indptr[b+1] - kv_indptr[b], and
prefix = kv_len - q_len >= 0 is the number of already-cached tokens in front
of this chunk; per query head h; fp32 chain):

    K[j] = concat(ckv_cache[page_j, 0, :], kpe_cache[page_j, 0, :])     (576)
    V[j] = ckv_cache[page_j, 0, :]                                      (512)
    Q[i] = concat(q_nope[q0 + i, h, :], q_pe[q0 + i, h, :])             (576)
    s[i, j] = sm_scale * <Q[i], K[j]>     for j <= prefix + i  (causal), else masked
    m = max_j s[i, j];  p[j] = exp(s[i, j] - m);  l = sum_j p[j]
    out[q0 + i, h, :] = (sum_j p[j] * V[j]) / l          (bf16 on output)
    lse[q0 + i, h]    = log2(sum_j exp(s[i, j])) = (m + ln(l)) / ln(2)   (fp32, base-2)

Pages a sequence does not list must never be read (the correctness cases fill
every unreferenced page with NaN). Query lengths, prefixes and page ids are
arbitrary: the trace has 1-token chunks next to 4000-token ones, prefixes from
0 to 1800, pages shared by several sequences, stride runs and long contiguous
runs — assume nothing about order, contiguity, uniqueness or multiples of a
tile. A query tile may straddle sequence boundaries.
"""

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op, T

try:
    from cutlass.cute.runtime import from_dlpack
except ImportError:  # pragma: no cover - DSL version drift
    from_dlpack = cute.runtime.from_dlpack

BF16 = cutlass.BFloat16
F32 = cutlass.Float32
I32 = cutlass.Int32

# cute.make_fragment was renamed make_rmem_tensor (cute_dsl_trace_scope gotcha 4)
_rmem = getattr(cute, "make_rmem_tensor", None) or getattr(cute, "make_fragment")

# ---------------------------------------------------------------------------
# M0 architecture constants (D0.5 / D0.6 / D0.7 / D0.12 — frozen resource
# topology; the levers D1.x may move are STAGES and the wait depth only).
#
# D0.16 — M0 delivery: the topology below is FROZEN at this commit.  Additive
# only from here; M1+ tunes levers, it does not retile.
#
#   signature      mla_paged_prefill, eight in / two out          (unchanged)
#   grid / block   ceil(TOTAL_Q/16) x H = 993 x 16 = 15888 / 128  (D0.5, D0.6)
#   smem           static 0, dynamic 61504 <= 65536               (D0.12)
#                  Q face 18432 resident | KV 2 stages 36864 | 4 S partials
#                  4096 at fixed stride | merged S 1024 | P 512 | m/l/r 64 ea.
#   rendezvous     5 blocking full-CTA bar.sync per KV-tile trip  (D0.12/D0.13)
#   in-flight      1 cp.async group (= STAGES 2), wait targets the PREVIOUS
#                  round's delivery, prologue warms one beat           (D0.13b)
#   handoff        [SYNC: named-barrier] symmetric rendezvous, no mbarrier
#   supply         [GATHER: prefetched] page ids clustered ahead of the fill
#   phase          [PHASE: none] one consumer phase, 4 unspecialised warps
#   registers      64 f32 O accumulators + 8 f32 QK accumulators / thread
#
# D0.17 — what the M0 gate deliberately does NOT check, so a later reader does
# not read the absence as an omission:
#   * per-CTA trip count.  D0.15 (3) narrows the KV loop to the causal
#     frontier, so the measured trip count MUST fall below the reference's
#     ~150 scan-everything figure — that drop is the point, not a regression,
#     and it drags dram_bytes_read / smem_wavefronts_ld down with it.
#   * absolute performance.  vs_ref_device is recorded as M1's starting
#     reading; M0 sets no ms line.
#   * the three things the reference got right (ckv stored once, 576 = 4x144
#     tail-free split-K, cp.async double buffer with a one-beat prologue) have
#     no separate gate line: their fingerprints already sit inside the smem
#     budget, the rendezvous/in-flight counts and the grid check.
#
# M0 GATE — submitted on this tree.  Source-side self-check of the checklist
# items that are readable without a profiler (the ncu/sass-derived ones and the
# five correctness cases are the gate run's to measure):
#   1  whole-machine smoke certificate: notes/probe_m0_gate_pre.py, PASS —
#      C1 lowered and the launch COMPLETED at TOTAL_Q 15883 / 18 sequences /
#      H 16 / 989669 pages / grid 993x16, alongside N1..N7 (out, base-2 lse and
#      global cosine distance against an fp64 chain at sm_scale
#      0.1352337747812271 AND 0.07 from one artefact, every unlisted page NaN).
#   4  block = [NT, 1, 1] = 128x1x1                      (_mla_launch)
#   5  grid = [ceil(total_q/16), heads, 1] = 993x16       (_mla_launch)
#   6  smem_static 0; smem_dynamic = Smem.size_in_bytes() = 61504 <= 65536
#      (16384 qn + 2048 qp + 32768 kc + 4096 kp + 4096 sp + 1024 ss + 512 pp
#       + 3 x 256 for rm/rl/rr) — the whole budget is the dynamic allocation.
#   8  5 blocking full-CTA sync_threads per KV-tile trip (rz1..rz5), all inside
#      the trip body; the prologue's two are outside the loop.
#   9  in-flight cp.async groups = 1: one unconditional commit_group per trip
#      and cp_async_wait_group(1), i.e. the wait targets the PREVIOUS round's
#      delivery; the fill is CopyG2SOp at 128 bits per copy (LDGSTS.E.128, the
#      .ca form — no bypass atom is constructed anywhere in this file).
#  10  both chains are warp-level mma.sync m16n8k16 (_qk_tile, _pv_chunk); no
#      scalar fallback path exists for either.
#  13  no absolute performance line; vs_ref_device is recorded as M1's start.
# ---------------------------------------------------------------------------
NT = 128                # threads / CTA = 4 warps, no specialisation   (D0.7)
NWARP = 4
TQ = 16                 # TILE_Q, one m16n8k16 M atom                  (D0.5)
TKV = 16                # TILE_KV                                      (D0.5)
DC = 512                # D_CKV  (key nope half AND value)
DP = 64                 # D_KPE  (key rotary part)
KC_W = DC // NWARP      # 128 ckv reduction dims per warp              (D0.8)
KP_W = DP // NWARP      # 16  kpe reduction dims per warp -> 144 total
NW = DC // NWARP        # 128 output d columns per warp                (D0.9)
NCHUNK = 4              # PV accumulator chunks -> 64 f32 / thread
NCOL = NW // NCHUNK     # 32 d columns per chunk
STAGES = 2              # KV double buffer                            (D0.11)

LOG2E = 1.4426950408889634
PARK = -1.0e30          # finite park value, not -inf   (branch_free_select_mask)
NEG_TEST = -1.0e29
L_FLOOR = 1.0e-30


def _seg(site, row=0):
    """Timeline probe hook (driver-instrumented arm only); a no-op here."""
    return None


# --------------------------------------------------------------- PTX rungs ---
# Deterministic instruction classes for the element-wise segment (D0.15 (2)(4),
# D2.3): predicate + FSEL for the mask, MUFU.EX2 for the base-2 exponential.

@dsl_user_op
def _sel_le_i32(a, b, x, y, *, loc=None, ip=None) -> cutlass.Float32:
    """a <= b ? x : y  (a, b int32; x, y f32) -> ISETP + FSEL, no branch."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.le.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _sel_gt_f32(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """x > y ? a : b  (all f32) -> FSETP + FSEL, no branch (empty-row guard)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.gt.f32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _fmaxf(x, y, *, loc=None, ip=None) -> cutlass.Float32:
    """max(x, y) -> one FMNMX."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _pow2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """2**x -> one MUFU.EX2 (the softmax chain lives in the base-2 domain)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """base-2 logarithm -> one MUFU.LG2 (the lse contract is base-2)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _rcp(x, *, loc=None, ip=None) -> cutlass.Float32:
    """1/x -> one MUFU.RCP (the epilogue's single division per row)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# ------------------------------------------------------------- smem budget ---
# D0.12: Q face 18432 B resident + KV double stage 36864 B + S/P/stat workspace.

@cute.struct
class Smem:
    qn: cute.struct.Align[cute.struct.MemRange[BF16, TQ * DC], 1024]
    qp: cute.struct.Align[cute.struct.MemRange[BF16, TQ * DP], 1024]
    kc: cute.struct.Align[cute.struct.MemRange[BF16, STAGES * TKV * DC], 1024]
    kp: cute.struct.Align[cute.struct.MemRange[BF16, STAGES * TKV * DP], 1024]
    sp: cute.struct.Align[cute.struct.MemRange[F32, NWARP * TQ * TKV], 1024]
    ss: cute.struct.Align[cute.struct.MemRange[F32, TQ * TKV], 1024]
    pp: cute.struct.Align[cute.struct.MemRange[BF16, TQ * TKV], 1024]
    rm: cute.struct.Align[cute.struct.MemRange[F32, TQ], 256]
    rl: cute.struct.Align[cute.struct.MemRange[F32, TQ], 256]
    rr: cute.struct.Align[cute.struct.MemRange[F32, TQ], 256]


def _mma():
    return cute.make_tiled_mma(warp.MmaF16BF16Op(BF16, F32, (16, 8, 16)), (1, 1, 1))


def _ldsm_a(mma):
    """A operand (M, K), K contiguous -> non-transposed member, tiler 16x16 = 4."""
    return cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma)


def _ldsm_b(mma):
    """B operand (N, K), K contiguous -> non-transposed member, tiler 8x16 = 2."""
    return cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BF16), mma)


def _ldsm_bt(mma):
    """B operand (N, K), N contiguous -> transposed member, tiler 8x16 = 2."""
    return cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BF16), mma)


def _swap(t):
    """(a, b) block -> (b, a) view: pure relabel, the contiguous mode is unchanged.

    The one ckv face is read twice: K-contiguous as QK^T's B operand (key nope
    half) and, through this relabel, N-contiguous as P.V's B operand (value) —
    D0.4 (1) "ckv is stored once".
    """
    return cute.make_tensor(t.iterator.align(16), cute.select(t.layout, [1, 0]))


# ------------------------------------------------------- the two MMA chains ---
# Both live in their own @cute.jit helper so an isolated probe can drive the
# shipped chain itself instead of transcribing it (gpu_probe_ladder gotcha 4).

@cute.jit
def _qk_tile(qn4, qp4, kc4, kp4, sSp, accS, frA, frB, cCs, wid, lane,
             mma, tcA, tcB):
    """D0.8: this warp's 144 reduction dims of Q@K^T, then publish the partial.

    Nine chained m16n8k16 steps (8 ckv k-blocks of 16 + 1 kpe k-block), both
    operands fed by ldmatrix from smem, accumulating in one f32 fragment; the
    16x16 partial then lands in this warp's slot of the 4-deep S partial
    buffer (fixed stride apart, D0.4 (2)).
    """
    thrA = tcA.get_slice(lane)
    thrB = tcB.get_slice(lane)
    accS.fill(0.0)
    for kb in cutlass.range_constexpr(KC_W // 16):
        ta = qn4[(None, None, 0, wid * (KC_W // 16) + kb)]
        tb = kc4[(None, None, 0, wid * (KC_W // 16) + kb)]
        cute.copy(tcA, thrA.partition_S(ta), thrA.retile(frA))
        cute.copy(tcB, thrB.partition_S(tb), thrB.retile(frB))
        cute.gemm(mma, accS, frA, frB, accS)
    ta = qp4[(None, None, 0, wid)]
    tb = kp4[(None, None, 0, wid)]
    cute.copy(tcA, thrA.partition_S(ta), thrA.retile(frA))
    cute.copy(tcB, thrB.partition_S(tb), thrB.retile(frB))
    cute.gemm(mma, accS, frA, frB, accS)
    for i in cutlass.range_constexpr(cute.size(accS)):
        m, n = cCs[i]
        sSp[(wid, m, n)] = accS[i]


@cute.jit
def _pv_chunk(sKcs, frP, frV, acc, col_blk, lane, mma, tcBT):
    """D0.9: NCOL output d columns of P.V on the tensor core.

    The value operand is the SAME ckv face QK^T just read K-contiguous, now
    relabelled N-contiguous and fed by the transposed ldmatrix member — D0.4
    (1) "ckv is stored once, read two ways".
    """
    thrBT = tcBT.get_slice(lane)
    vt = _swap(cute.flat_divide(sKcs, (TKV, NCOL))[(None, None, 0, col_blk)])
    cute.copy(tcBT, thrBT.partition_S(vt), thrBT.retile(frV))
    cute.gemm(mma, acc, frP, frV, acc)


# --------------------------------------------------------------- KV supply ---

@cute.jit
def _fill_kv(mCkv, mKpe, mIdx, sKc, sKp, kv_base, kv_len, j0, tidx, tvC, tvP):
    """D0.11: one KV tile (16 rows x 576) gmem -> smem through cp.async.

    Nine LDGSTS.E.128 per thread (8 x ckv + 1 x kpe = 18432 B / tile). The page
    ids are ALL loaded first, clustered ahead of the fill, so no LDGSTS waits on
    "this row's index just came back" — the [GATHER: prefetched] bit order.
    Rows past the end of the sequence are clamped onto the sequence's last
    listed page, so an unlisted (NaN) page is never touched.
    """
    tpr_c = DC * 2 // 16                 # 64 lanes cover one 1024 B ckv row
    teams_c = NT // tpr_c                # 2 row teams -> 8 rows each
    thrC = tvC.get_slice(tidx % tpr_c)
    team_c = tidx // tpr_c

    tpr_p = DP * 2 // 16                 # 8 lanes cover one 128 B kpe row
    thrP = tvP.get_slice(tidx % tpr_p)
    team_p = tidx // tpr_p               # 16 row teams -> one row each

    last = kv_len - 1

    # --- index phase: every page id first (D0.11 bit order) ---
    pids = []
    for i in cutlass.range_constexpr(TKV // teams_c):
        j = j0 + i * teams_c + team_c
        jc = j
        if j > last:
            jc = last
        pids.append(I32(mIdx[kv_base + jc]))
    jp = j0 + team_p
    jpc = jp
    if jp > last:
        jpc = last
    pid_p = I32(mIdx[kv_base + jpc])

    # --- fill phase: 9 cp.async per thread, one open group ---
    for i in cutlass.range_constexpr(TKV // teams_c):
        r = i * teams_c + team_c
        cute.copy(tvC, thrC.partition_S(mCkv[(pids[i], None)]),
                  thrC.partition_D(sKc[(r, None)]))
    cute.copy(tvP, thrP.partition_S(mKpe[(pid_p, None)]),
              thrP.partition_D(sKp[(team_p, None)]))


@cute.jit
def _fill_q(mQn, mQp, sQn, sQp, q_start, qmax, head, tidx, tvC, tvP):
    """D0.2: the Q face lands ONCE in the prologue and stays resident.

    Rows past TOTAL_Q (only the last flat tile has any) are clamped onto the
    last query row; they belong to no sequence, so they stay parked all the way
    through and are never stored.
    """
    tpr_c = DC * 2 // 16
    teams_c = NT // tpr_c
    thrC = tvC.get_slice(tidx % tpr_c)
    team_c = tidx // tpr_c
    tpr_p = DP * 2 // 16
    thrP = tvP.get_slice(tidx % tpr_p)
    team_p = tidx // tpr_p

    for i in cutlass.range_constexpr(TQ // teams_c):
        r = i * teams_c + team_c
        g = q_start + r
        gc = g
        if g > qmax:
            gc = qmax
        cute.copy(tvC, thrC.partition_S(mQn[(gc, head, None)]),
                  thrC.partition_D(sQn[(r, None)]))
    gp = q_start + team_p
    gpc = gp
    if gp > qmax:
        gpc = qmax
    cute.copy(tvP, thrP.partition_S(mQp[(gpc, head, None)]),
              thrP.partition_D(sQp[(team_p, None)]))


# ------------------------------------------------------------- the kernel ----

@cute.kernel
def _mla_kernel(mQn: cute.Tensor, mQp: cute.Tensor, mCkv: cute.Tensor,
                mKpe: cute.Tensor, mQo: cute.Tensor, mKvp: cute.Tensor,
                mIdx: cute.Tensor, mOut: cute.Tensor, mLse: cute.Tensor,
                sm_scale: F32, total_q: cutlass.Constexpr,
                nseq: cutlass.Constexpr, mma: cute.TiledMma,
                tcA: cute.TiledCopy, tcB: cute.TiledCopy, tcBT: cute.TiledCopy,
                tvC: cute.TiledCopy, tvP: cute.TiledCopy):
    tidx, _, _ = cute.arch.thread_idx()
    bx, head, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    wid = cute.arch.warp_idx()

    _seg("cta.entry")

    # sm_scale is honoured inside the kernel; log2(e) is folded into the same
    # constant so the whole softmax chain is base-2 (D0.3 (5), D0.15 (4)).
    cscale = sm_scale * F32(LOG2E)

    st = cutlass.utils.SmemAllocator().allocate(Smem)
    sQn = st.qn.get_tensor(cute.make_layout((TQ, DC), stride=(DC, 1)))
    sQp = st.qp.get_tensor(cute.make_layout((TQ, DP), stride=(DP, 1)))
    sKc = st.kc.get_tensor(cute.make_layout((STAGES, TKV, DC),
                                            stride=(TKV * DC, DC, 1)))
    sKp = st.kp.get_tensor(cute.make_layout((STAGES, TKV, DP),
                                            stride=(TKV * DP, DP, 1)))
    sSp = st.sp.get_tensor(cute.make_layout((NWARP, TQ, TKV),
                                            stride=(TQ * TKV, TKV, 1)))
    sS = st.ss.get_tensor(cute.make_layout((TQ, TKV), stride=(TKV, 1)))
    sP = st.pp.get_tensor(cute.make_layout((TQ, TKV), stride=(TKV, 1)))
    sM = st.rm.get_tensor(cute.make_layout(TQ))
    sL = st.rl.get_tensor(cute.make_layout(TQ))
    sR = st.rr.get_tensor(cute.make_layout(TQ))

    q_start = I32(bx * TQ)
    q_end = q_start + TQ
    if q_end > total_q:
        q_end = I32(total_q)

    # ---- prologue: Q face + row statistics -------------------------------
    _seg("prologue.qfill.begin")
    _fill_q(mQn, mQp, sQn, sQp, q_start, I32(total_q - 1), head, tidx, tvC, tvP)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)          # the Q path drains completely
    cute.arch.fence_view_async_shared()
    _seg("prologue.qfill.end")

    if tidx < TQ:
        sM[tidx] = F32(PARK)
        sL[tidx] = F32(0.0)
        sR[tidx] = F32(1.0)

    # ---- register state ---------------------------------------------------
    thr = mma.get_slice(lane)
    thrA = tcA.get_slice(lane)

    # Every name bound here lives in the kernel frame for the whole trace; a
    # staged `if`/`for` region that re-binds one of them must keep its
    # structure (CONTAINER_STRUCTURE_CHANGED otherwise), so the loop bodies
    # below use names that appear nowhere else (cute_dsl_const_expr_if).
    accS = _rmem(thr.partition_shape_C((TQ, TKV)), F32)
    accs = []
    for nb0 in cutlass.range_constexpr(NCHUNK):
        ac = _rmem(thr.partition_shape_C((TQ, NCOL)), F32)
        ac.fill(0.0)
        accs.append(ac)

    # the C map of one 32-column chunk doubles as the address map of the
    # epilogue and as the row map of the accumulator rescale: an m16n8k16 C
    # fragment touches exactly two rows per lane (i % 4 in {0,1} -> r_lo,
    # {2,3} -> r_hi), so the rescale needs two smem reads, not 64.
    cCs = thr.partition_C(cute.make_identity_tensor((TQ, TKV)))
    cCh = thr.partition_C(cute.make_identity_tensor((TQ, NCOL)))
    r_lo, _c0 = cCh[0]
    r_hi, _c2 = cCh[2]

    qn4 = cute.flat_divide(sQn, (TQ, 16))
    qp4 = cute.flat_divide(sQp, (TQ, 16))

    frA = thr.make_fragment_A(thr.partition_A(qn4[(None, None, 0, 0)]))
    frB = thr.make_fragment_B(thr.partition_B(qn4[(None, None, 0, 0)]))
    frP = thr.make_fragment_A(thr.partition_A(sP))
    frV = thr.make_fragment_B(thr.partition_B(
        _swap(cute.flat_divide(sKc[(0, None, None)], (TKV, NCOL))[(None, None, 0, 0)])))

    cute.arch.sync_threads()

    # ---- per-sequence sub-blocks of this flat q-tile (D0.5) ---------------
    # A 16-row tile may straddle sequence boundaries; every row of the tile
    # belongs to exactly one sequence, so at most TQ sub-blocks.  b0 = the
    # sequence owning q_start.
    b0 = I32(0)
    for _scan in cutlass.range(nseq - 1):
        if mQo[b0 + 1] <= q_start:
            b0 = b0 + 1

    tripc = I32(0)

    for sb in cutlass.range(TQ):
        bb = b0 + sb
        bbc = bb
        if bb > nseq - 1:
            bbc = I32(nseq - 1)
        q0b = mQo[bbc]
        q1b = mQo[bbc + 1]
        if bb > nseq - 1:
            q0b = q_end                      # force an empty sub-block
        kv_base = mKvp[bbc]
        kv_len = mKvp[bbc + 1] - kv_base
        prefix = kv_len - (q1b - q0b)

        lo = q0b
        if lo < q_start:
            lo = q_start
        hi = q1b
        if hi > q_end:
            hi = q_end
        r0 = lo - q_start
        r1 = hi - q_start

        # D0.15 (3): tile-level narrowing of the KV loop bound by the causal
        # frontier of this sub-block's LAST query row (the diagonal tile is
        # still handled exactly by the element-wise mask).
        kv_hi = prefix + (q_start + r1 - 1 - q0b) + 1
        if kv_hi > kv_len:
            kv_hi = kv_len
        ntile = (kv_hi + (TKV - 1)) // TKV
        if r1 <= r0:
            ntile = I32(0)

        # prologue: prefetch tile 0 one beat ahead (D0.11 / D0.13(b))
        if ntile > 0:
            _fill_kv(mCkv, mKpe, mIdx, sKc[(0, None, None)], sKp[(0, None, None)],
                     kv_base, kv_len, I32(0), tidx, tvC, tvP)
            cute.arch.cp_async_commit_group()

        stg = I32(0)
        for t in cutlass.range(ntile):
            nstg = 1 - stg
            j0 = t * TKV

            # 1-5: next tile's page ids + fill, commit, wait on the PREVIOUS
            # round's delivery.  The only thing in flight across the body is
            # the next KV tile's cp.async (D0.13(a)).
            _seg("kv.fill.begin", tripc)
            if t + 1 < ntile:
                _fill_kv(mCkv, mKpe, mIdx,
                         sKc[(nstg, None, None)], sKp[(nstg, None, None)],
                         kv_base, kv_len, j0 + TKV, tidx, tvC, tvP)
            cute.arch.cp_async_commit_group()
            _seg("kv.fill.end", tripc)

            _seg("kv.wait.begin", tripc)
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_view_async_shared()
            _seg("kv.wait.end", tripc)

            _seg("rz1.begin", tripc)
            cute.arch.sync_threads()                       # rendezvous #1
            _seg("rz1.end", tripc)

            # 7-8: QK^T, warp-level mma.sync fed by ldmatrix, split-K 144 dims
            # per warp, chained accumulate, then publish this warp's partial.
            _seg("qk.begin", tripc)
            kc4 = cute.flat_divide(sKc[(stg, None, None)], (TKV, 16))
            kp4 = cute.flat_divide(sKp[(stg, None, None)], (TKV, 16))
            _qk_tile(qn4, qp4, kc4, kp4, sSp, accS, frA, frB, cCs, wid, lane,
                     mma, tcA, tcB)
            _seg("qk.end", tripc)

            _seg("rz2.begin", tripc)
            cute.arch.sync_threads()                       # rendezvous #2
            _seg("rz2.end", tripc)

            # 10: the 4-way merge, inside the loop body (D0.10)
            _seg("merge.begin", tripc)
            for e in cutlass.range_constexpr((TQ * TKV) // NT):
                i = e * NT + tidx
                rr = i // TKV
                cc = i % TKV
                sS[(rr, cc)] = ((sSp[(0, rr, cc)] + sSp[(1, rr, cc)])
                                + (sSp[(2, rr, cc)] + sSp[(3, rr, cc)]))
            _seg("merge.end", tripc)

            _seg("rz3.begin", tripc)
            cute.arch.sync_threads()                       # rendezvous #3
            _seg("rz3.end", tripc)

            # 12: the element-wise segment — branch-free causal mask, running
            # max in the RAW domain, one FFMA folding cscale into MUFU.EX2.
            _seg("softmax.begin", tripc)
            if tidx < TQ:
                # visible columns of this row: j <= prefix + i and j < kv_len;
                # rows outside this sub-block get an empty frontier.
                lim = prefix + (q_start + tidx - q0b)
                if lim > kv_len - 1:
                    lim = kv_len - 1
                if tidx < r0:
                    lim = I32(-1)
                if tidx > r1 - 1:
                    lim = I32(-1)
                climit = lim - j0

                sv = []
                mm = F32(PARK)
                for cj in cutlass.range_constexpr(TKV):
                    x = _sel_le_i32(I32(cj), climit, sS[(tidx, cj)], F32(PARK))
                    sv.append(x)
                    mm = _fmaxf(mm, x)
                m_old = sM[tidx]
                m_new = _fmaxf(m_old, mm)
                alpha = _pow2((m_old - m_new) * cscale)
                mu = _sel_gt_f32(m_new, F32(NEG_TEST), m_new * cscale, F32(0.0))
                lsum = F32(0.0)
                for cj in cutlass.range_constexpr(TKV):
                    pv = _pow2(sv[cj] * cscale - mu)
                    lsum = lsum + pv
                    sP[(tidx, cj)] = BF16(pv)
                sM[tidx] = m_new
                sL[tidx] = sL[tidx] * alpha + lsum
                sR[tidx] = alpha
            _seg("softmax.end", tripc)

            _seg("rz4.begin", tripc)
            cute.arch.sync_threads()                       # rendezvous #4
            _seg("rz4.end", tripc)

            # 14: accumulator rescale — two smem reads per lane
            _seg("rescale.begin", tripc)
            f_lo = sR[r_lo]
            f_hi = sR[r_hi]
            for nb in cutlass.range_constexpr(NCHUNK):
                acc = accs[nb]
                for i in cutlass.range_constexpr(cute.size(acc)):
                    if cutlass.const_expr(i % 4 < 2):
                        acc[i] = acc[i] * f_lo
                    else:
                        acc[i] = acc[i] * f_hi
            _seg("rescale.end", tripc)

            # 15: P.V on the tensor core — the SAME ckv face, read a second
            # way (N-contiguous relabel + transposed ldmatrix member).
            _seg("pv.begin", tripc)
            cute.copy(tcA, thrA.partition_S(sP), thrA.retile(frP))
            for nb in cutlass.range_constexpr(NCHUNK):
                _pv_chunk(sKc[(stg, None, None)], frP, frV, accs[nb],
                          wid * NCHUNK + nb, lane, mma, tcBT)
            _seg("pv.end", tripc)

            _seg("rz5.begin", tripc)
            cute.arch.sync_threads()                       # rendezvous #5
            _seg("rz5.end", tripc)

            stg = nstg
            tripc = tripc + 1

        cute.arch.cp_async_wait_group(0)

    # ---- epilogue: one normalisation per row, bf16 narrowing, base-2 lse ---
    _seg("epilogue.begin", tripc)
    inv_lo = _rcp(_fmaxf(sL[r_lo], F32(L_FLOOR)))
    inv_hi = _rcp(_fmaxf(sL[r_hi], F32(L_FLOOR)))
    g_lo = q_start + r_lo
    g_hi = q_start + r_hi

    if g_lo < total_q:
        for nb in cutlass.range_constexpr(NCHUNK):
            acc = accs[nb]
            for i in cutlass.range_constexpr(cute.size(acc)):
                if cutlass.const_expr(i % 4 < 2):
                    _rl, cn = cCh[i]
                    mOut[(g_lo, head, wid * NW + nb * NCOL + cn)] = \
                        BF16(acc[i] * inv_lo)
    if g_hi < total_q:
        for nb in cutlass.range_constexpr(NCHUNK):
            acc = accs[nb]
            for i in cutlass.range_constexpr(cute.size(acc)):
                if cutlass.const_expr(i % 4 >= 2):
                    _rh, cn = cCh[i]
                    mOut[(g_hi, head, wid * NW + nb * NCOL + cn)] = \
                        BF16(acc[i] * inv_hi)

    if tidx < TQ:
        gl = q_start + tidx
        if gl < total_q:
            mLse[(gl, head)] = (sM[tidx] * cscale
                                + _lg2(_fmaxf(sL[tidx], F32(L_FLOOR))))
    _seg("epilogue.end", tripc)


@cute.jit
def _mla_launch(mQn: cute.Tensor, mQp: cute.Tensor, mCkv: cute.Tensor,
                mKpe: cute.Tensor, mQo: cute.Tensor, mKvp: cute.Tensor,
                mIdx: cute.Tensor, mOut: cute.Tensor, mLse: cute.Tensor,
                sm_scale: F32, total_q: cutlass.Constexpr,
                nseq: cutlass.Constexpr, heads: cutlass.Constexpr):
    mma = _mma()
    atom = cute.make_copy_atom(cpasync.CopyG2SOp(), BF16, num_bits_per_copy=128)
    tvC = cute.make_tiled_copy_tv(atom, cute.make_layout(DC * 2 // 16),
                                  cute.make_layout(8))
    tvP = cute.make_tiled_copy_tv(atom, cute.make_layout(DP * 2 // 16),
                                  cute.make_layout(8))
    _mla_kernel(mQn, mQp, mCkv, mKpe, mQo, mKvp, mIdx, mOut, mLse,
                sm_scale, total_q, nseq, mma,
                _ldsm_a(mma), _ldsm_b(mma), _ldsm_bt(mma), tvC, tvP).launch(
        grid=[(total_q + TQ - 1) // TQ, heads, 1],
        block=[NT, 1, 1],
        smem=Smem.size_in_bytes())


# --------------------------------------------------------------- host side ---

_CACHE = {}


def _dl(x):
    return from_dlpack(x, assumed_align=16)


def mla_paged_prefill(
    q_nope: torch.Tensor,      # [TOTAL_Q, H=16, D_CKV=512] bf16
    q_pe: torch.Tensor,        # [TOTAL_Q, H=16, D_KPE=64]  bf16
    ckv_cache: torch.Tensor,   # [NUM_PAGES, 1, 512] bf16  (key nope half AND value)
    kpe_cache: torch.Tensor,   # [NUM_PAGES, 1, 64]  bf16  (key rotary part)
    qo_indptr: torch.Tensor,   # [B + 1] int32, query-row offsets per sequence
    kv_indptr: torch.Tensor,   # [B + 1] int32, page-list offsets per sequence
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] int32, page ids
    sm_scale: float,           # softmax scale (campaign: 0.1352337747812271)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [TOTAL_Q, 16, 512] bf16, lse [TOTAL_Q, 16] fp32 base-2 log-sum-exp)."""
    total_q = q_nope.shape[0]
    heads = q_nope.shape[1]
    npages = ckv_cache.shape[0]
    nseq = qo_indptr.shape[0] - 1

    out = torch.empty((total_q, heads, DC), dtype=torch.bfloat16,
                      device=q_nope.device)
    lse = torch.empty((total_q, heads), dtype=torch.float32, device=q_nope.device)

    ckv2 = ckv_cache.view(npages, DC)
    kpe2 = kpe_cache.view(npages, DP)

    args = (_dl(q_nope), _dl(q_pe), _dl(ckv2), _dl(kpe2), _dl(qo_indptr),
            _dl(kv_indptr), _dl(kv_indices), _dl(out), _dl(lse),
            F32(sm_scale))
    key = (total_q, heads, npages, nseq, kv_indices.shape[0])
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(_mla_launch, *args, total_q, nseq, heads)
        _CACHE[key] = fn
    # The Constexpr tail (total_q / nseq / heads) is baked into the compiled
    # artefact by cute.compile and must NOT be handed to the compiled callable
    # again: the compiled call takes the dynamic arguments only.
    fn(*args)
    return out, lse
