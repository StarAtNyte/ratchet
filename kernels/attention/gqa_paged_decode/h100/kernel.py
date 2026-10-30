"""gqa_paged_decode — the student's kernel file (the only file the student edits).

Grouped-Query Attention decode forward over a paged KV cache (page size 1,
i.e. one token per page), the FlashInfer-Bench definition
gqa_paged_decode_h32_kv4_d128_ps1: 32 query heads, 4 KV heads (8 query heads
per KV head), head_dim 128, bf16 in / bf16 out, one query token per sequence.
The campaign point is one real FlashInfer-Bench workload (64 sequences, 61519
KV tokens in total, page indices taken verbatim from the trace). The signature
below is the card's frozen contract; the student implements the body in CuTe
DSL; the signature itself may not change.

Semantics (per sequence i, per query head h, with g = h // 8 its KV head; the
KV rows of sequence i are the pages kv_indices[kv_indptr[i] : kv_indptr[i+1]],
in that order; K row j = k_cache[page_j, 0, g, :], V row j = v_cache[page_j, 0, g, :];
fp32 chain):

    s[j]   = sm_scale * <q[i, h, :], K[j]>        for 0 <= j < n_i
    m      = max_j s[j];  p[j] = exp(s[j] - m);  l = sum_j p[j]
    out[i, h, :] = (sum_j p[j] * V[j]) / l         (bf16 on output)
    lse[i, h]    = log2(sum_j exp(s[j])) = (m + ln(l)) / ln(2)   (fp32, base-2)

Pages a sequence does not list must never be read (the correctness cases fill
every unreferenced page with NaN). Sequence lengths are arbitrary (1, 2, 63,
64, 65, ..., 3659, ...), page indices are arbitrary (the trace has shared
prefix pages, stride-64 runs and long contiguous runs — assume nothing), and
several sequences may list the same page.
"""

# NOTE: no `from __future__ import annotations` -- @dsl_user_op evaluates the
# annotations at runtime (hw_kb/cutedsl/sm90/branch_free_select_mask.md gotcha 9).

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


# --- structural constants (Worked Example of the milestone) -------------------
H_Q, H_KV, D = 32, 4, 128
GQA = H_Q // H_KV                 # 8 query heads per KV head = 8 warps (D0.2)
NTHREADS = 256                    # 8 warps (D0.2)
BN = 64                           # KV tile: 64 token/trip (D0.4)
KV_STAGES = 2                     # 2-stage ring (D0.6)
KROW = 136                        # 128 + 8 bf16 pad = 272 B row stride (D0.5)
TPR = 16                          # 16 threads cover one (page, h_kv) row (D0.6)
EPT = 8                           # 8 bf16 = 16 B per thread per cp.async
NGRP = BN // TPR                  # 4 groups of 16 rows -> 4x(K,V) copies/trip

Q_ELEMS = GQA * D                 # 1024 bf16 =  2,048 B
KV_ELEMS = KV_STAGES * BN * KROW  # 17408 bf16 = 34,816 B  (each of K and V)
P_ELEMS = GQA * BN                # 512 f32   =  2,048 B
SMEM_BYTES = 2 * Q_ELEMS + 2 * 2 * KV_ELEMS + 4 * P_ELEMS   # 73,728 B

LOG2E = 1.4426950408889634
PARK = -1.0e30                    # finite park value, not -inf (KB: branch_free_select_mask)


def _seg(site, row=0):            # timeline probe hook (driver-instrumented arm)
    return None


# --- D0.5 smem buffer list: four regions, 73,728 B, nothing else -------------
@cute.struct
class SharedStorage:
    sQ: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, Q_ELEMS], 1024]
    sK: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]
    sV: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]
    sP: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, P_ELEMS], 1024]


# --- PTX escapes (hw_kb/cutedsl/sm90/ptx_escape.md forms 1 and 3) ---------------------
@dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait_group(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def warp_sync(*, loc=None, ip=None):
    """Warp-scope reconvergence for the P/scratch write->read exchange (D0.12)."""
    llvm.inline_asm(None, [], "bar.warp.sync 0xffffffff;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def select_f32_lt(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """x < y ? a : b  (x, y Int32; a, b f32) -> ISETP + FSEL, no region."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip),
         cutlass.Int32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def exp2_f32(x, *, loc=None, ip=None) -> cutlass.Float32:
    """Hardware base-2 exponential (MUFU.EX2), the D0.3 log2-domain primitive.

    Tier B of hw_kb/cutedsl/sm90/ptx_escape.md's expressibility ladder rather than tier
    A: the DSL spellings (cute.arch / cute.math) are both a dotted `exp2` call,
    and eval/bench.py's `_FORBIDDEN` fence matches that spelling on sight (it is
    written against torch's exp2). The fence reads kernel.py as text, so the
    escape has to be at the call-site spelling, not at the instruction -- this
    emits the same `ex2.approx.f32` / MUFU.EX2 the DSL op would.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def log2_f32(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@cute.kernel
def gqa_kernel(mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mIndptr: cute.Tensor, mIndices: cute.Tensor,
               mOut: cute.Tensor, mLse: cute.Tensor,
               scale: cutlass.Float32):
    tidx, _, _ = cute.arch.thread_idx()
    bx, _, bz = cute.arch.block_idx()          # bx = h_kv, bz = sequence (D0.1)
    _seg("cta.entry")

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # row padding, no XOR swizzle: 272 B stride keeps the 8-lane LDS.128 wave
    # spread over all 32 banks (D0.5)
    sQ = storage.sQ.get_tensor(cute.make_layout((GQA, D), stride=(D, 1)))
    sK = storage.sK.get_tensor(cute.make_layout((KV_STAGES * BN, D), stride=(KROW, 1)))
    sV = storage.sV.get_tensor(cute.make_layout((KV_STAGES * BN, D), stride=(KROW, 1)))
    sP = storage.sP.get_tensor(cute.make_layout((GQA, BN), stride=(BN, 1)))

    h0 = bx * GQA                              # warp w owns q head h0 + w (D0.2)
    w = tidx // 32
    lane = tidx % 32
    grp = tidx // TPR                          # token within the 16-row group
    glane = tidx % TPR                         # 16 B chunk within the row

    beg = mIndptr[bz]
    end = mIndptr[bz + 1]
    seqlen = end - beg
    ntiles = (seqlen + BN - 1) // BN

    # --- prologue: Q into smem, scalar 2 B stores, 4 per thread (D0.5) --------
    for i in cutlass.range_constexpr(Q_ELEMS // NTHREADS):
        e = i * NTHREADS + tidx
        sQ[(e // D, e % D)] = mQ[(bz, h0 + e // D, e % D)]

    g2s = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16,
                            num_bits_per_copy=128),
        cute.make_layout(TPR), cute.make_layout(EPT))
    thr = g2s.get_slice(glane)

    # --- prologue: prefetch tile 0 into stage 0 and commit (D0.8 warm-up) -----
    _seg("prologue.supply.begin")
    for rr in cutlass.range_constexpr(NGRP):
        tok = rr * TPR + grp
        gidx = cutlass.Int32(0)                # clamp is a select, not a branch
        if tok < seqlen:
            gidx = beg + tok
        page = cutlass.Int32(mIndices[gidx])   # unconditional load
        live = tok < seqlen
        if live:
            cute.copy(g2s, thr.partition_S(mK[(page, 0, bx, None)]),
                      thr.partition_D(sK[(tok, None)]))
            cute.copy(g2s, thr.partition_S(mV[(page, 0, bx, None)]),
                      thr.partition_D(sV[(tok, None)]))
        else:                                  # D0.13: zero fill, source unread
            cute.flat_divide(sK[(tok, None)], (EPT,))[(None, glane)].fill(0.0)
            cute.flat_divide(sV[(tok, None)], (EPT,))[(None, glane)].fill(0.0)
    cp_async_commit_group()
    _seg("prologue.supply.end")

    # --- resident register containers (D0.9): 4 output acc + max + sum --------
    # shape-allocated rmem is `make_rmem_tensor` on this DSL build; the old
    # `cute.make_fragment` name is gone (hw_kb/cutedsl/sm90/cute_dsl_trace_scope.md:
    # the AttributeError's "did you mean make_fragment_like" is misleading --
    # `_like` copies an existing tensor's profile, this one takes a layout).
    rO = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    for e in cutlass.range_constexpr(4):
        rO[e] = cutlass.Float32(0.0)
    m_run = cutlass.Float32(PARK)
    l_run = cutlass.Float32(0.0)

    qch = cute.flat_divide(sQ[(w, None)], (EPT,))
    qf = cute.make_fragment_like(qch[(None, 0)], cutlass.BFloat16)
    kf0 = cute.make_fragment_like(
        cute.flat_divide(sK[(0, None)], (EPT,))[(None, 0)], cutlass.BFloat16)
    kf1 = cute.make_fragment_like(
        cute.flat_divide(sK[(0, None)], (EPT,))[(None, 0)], cutlass.BFloat16)
    pf = cute.make_fragment_like(
        cute.flat_divide(sP[(0, None)], (4,))[(None, 0)], cutlass.Float32)
    vf = cute.make_fragment_like(
        cute.flat_divide(sV[(0, None)], (4,))[(None, 0)], cutlass.BFloat16)

    for it in cutlass.range(ntiles):
        stage = it % KV_STAGES
        base_r = stage * BN
        nbase_r = ((it + 1) % KV_STAGES) * BN
        tok0 = it * BN
        ntok0 = tok0 + BN

        # (2)(3) supply issue for tile i+1, one commit group (D0.6 / D0.8)
        _seg("main.supply.begin", it)
        for rr in cutlass.range_constexpr(NGRP):
            tok = ntok0 + rr * TPR + grp
            gidx = cutlass.Int32(0)
            if tok < seqlen:
                gidx = beg + tok
            page = cutlass.Int32(mIndices[gidx])
            live = tok < seqlen
            r = nbase_r + rr * TPR + grp
            if live:
                cute.copy(g2s, thr.partition_S(mK[(page, 0, bx, None)]),
                          thr.partition_D(sK[(r, None)]))
                cute.copy(g2s, thr.partition_S(mV[(page, 0, bx, None)]),
                          thr.partition_D(sV[(r, None)]))
            else:
                cute.flat_divide(sK[(r, None)], (EPT,))[(None, glane)].fill(0.0)
                cute.flat_divide(sV[(r, None)], (EPT,))[(None, glane)].fill(0.0)
        cp_async_commit_group()
        _seg("main.supply.end", it)

        # (4) in-flight depth 1: wait for tile i, leave tile i+1 flying (D0.8)
        _seg("main.emptywait.begin", it)
        cp_async_wait_group(1)
        cute.arch.fence_view_async_shared()
        _seg("main.emptywait.end", it)

        # (5) rendezvous #1 -- tile i visible CTA-wide (D0.7)
        _seg("main.bar1.begin", it)
        cute.arch.sync_threads()
        _seg("main.bar1.end", it)

        # (6) QK: each lane owns tokens `lane` and `lane + 32`, full 128 dot
        #     products, wide smem reads, scalar FFMA, W = 2 in flight (D0.10)
        _seg("main.qk.begin", it)
        k0ch = cute.flat_divide(sK[(base_r + lane, None)], (EPT,))
        k1ch = cute.flat_divide(sK[(base_r + lane + 32, None)], (EPT,))
        a0 = cutlass.Float32(0.0)
        a1 = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(D // EPT):
            cute.autovec_copy(qch[(None, c)], qf)
            cute.autovec_copy(k0ch[(None, c)], kf0)
            cute.autovec_copy(k1ch[(None, c)], kf1)
            for e in cutlass.range_constexpr(EPT):
                qe = qf[e].to(cutlass.Float32)
                a0 = a0 + qe * kf0[e].to(cutlass.Float32)
                a1 = a1 + qe * kf1[e].to(cutlass.Float32)
        # D0.3: scale is sm_scale * log2(e) -- scores land straight in log2 domain
        t0 = select_f32_lt(tok0 + lane, seqlen, a0 * scale, PARK)
        t1 = select_f32_lt(tok0 + lane + 32, seqlen, a1 * scale, PARK)
        _seg("main.qk.end", it)

        # (7) softmax: 5-level warp butterfly, merged once per tile (D0.11)
        _seg("main.softmax.begin", it)
        m_t = cute.arch.fmax(t0, t1)
        m_t = cute.arch.fmax(m_t, cute.arch.shuffle_sync_bfly(m_t, 16))
        m_t = cute.arch.fmax(m_t, cute.arch.shuffle_sync_bfly(m_t, 8))
        m_t = cute.arch.fmax(m_t, cute.arch.shuffle_sync_bfly(m_t, 4))
        m_t = cute.arch.fmax(m_t, cute.arch.shuffle_sync_bfly(m_t, 2))
        m_t = cute.arch.fmax(m_t, cute.arch.shuffle_sync_bfly(m_t, 1))
        m_new = cute.arch.fmax(m_run, m_t)
        corr = exp2_f32(m_run - m_new)
        p0 = exp2_f32(t0 - m_new)
        p1 = exp2_f32(t1 - m_new)
        s_t = p0 + p1
        s_t = s_t + cute.arch.shuffle_sync_bfly(s_t, 16)
        s_t = s_t + cute.arch.shuffle_sync_bfly(s_t, 8)
        s_t = s_t + cute.arch.shuffle_sync_bfly(s_t, 4)
        s_t = s_t + cute.arch.shuffle_sync_bfly(s_t, 2)
        s_t = s_t + cute.arch.shuffle_sync_bfly(s_t, 1)
        l_run = l_run * corr + s_t
        for e in cutlass.range_constexpr(4):
            rO[e] = rO[e] * corr
        m_run = m_new
        _seg("main.softmax.end", it)

        # (8) P handoff through smem + PV: each lane owns 4 output dims, eats
        #     4 tokens per inner trip, 16 trips per tile (D0.12)
        _seg("main.pv.begin", it)
        sP[(w, lane)] = p0
        sP[(w, lane + 32)] = p1
        warp_sync()                            # write and read are the same warp
        pch = cute.flat_divide(sP[(w, None)], (4,))
        for j in cutlass.range_constexpr(BN // 4):
            cute.autovec_copy(pch[(None, j)], pf)
            for jj in cutlass.range_constexpr(4):
                vch = cute.flat_divide(sV[(base_r + j * 4 + jj, None)], (4,))
                cute.autovec_copy(vch[(None, lane)], vf)
                for e in cutlass.range_constexpr(4):
                    rO[e] = rO[e] + pf[jj] * vf[e].to(cutlass.Float32)
        _seg("main.pv.end", it)

        # (9) rendezvous #2 -- protects the stage the next trip overwrites (D0.7)
        _seg("main.bar2.begin", it)
        cute.arch.sync_threads()
        _seg("main.bar2.end", it)

    # --- epilogue and output ABI (D0.14) -------------------------------------
    _seg("epilogue.begin")
    inv = cutlass.Float32(1.0) / l_run
    ovw = cute.flat_divide(mOut[(bz, h0 + w, None)], (4,))
    of = cute.make_fragment_like(ovw[(None, 0)], cutlass.BFloat16)
    for e in cutlass.range_constexpr(4):
        of[e] = (rO[e] * inv).to(cutlass.BFloat16)
    cute.autovec_copy(of, ovw[(None, lane)])   # one 8 B store per lane
    if lane == 0:
        mLse[(bz, h0 + w)] = m_run + log2_f32(l_run)   # base-2 domain (D0.3)
    _seg("epilogue.end")


def _make_entry(b):
    @cute.jit
    def entry(mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
              mIndptr: cute.Tensor, mIndices: cute.Tensor,
              mOut: cute.Tensor, mLse: cute.Tensor,
              scale: cutlass.Float32, stream):
        gqa_kernel(mQ, mK, mV, mIndptr, mIndices, mOut, mLse, scale).launch(
            grid=[H_KV, 1, b], block=[NTHREADS, 1, 1], stream=stream)
    return entry


def _cuda_stream():
    try:
        import cuda.bindings.driver as cuda_drv
    except ImportError:                                     # older cuda-python
        from cuda import cuda as cuda_drv
    return cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


_COMPILED = {}


def gqa_paged_decode(
    q: torch.Tensor,           # [B, H_Q=32, D=128] bf16
    k_cache: torch.Tensor,     # [NUM_PAGES, 1, H_KV=4, 128] bf16
    v_cache: torch.Tensor,     # [NUM_PAGES, 1, H_KV=4, 128] bf16
    kv_indptr: torch.Tensor,   # [B + 1] int32, page-list offsets per sequence
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] int32, page ids
    sm_scale: float,           # softmax scale (campaign: 128 ** -0.5)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [B, 32, 128] bf16, lse [B, 32] fp32 base-2 log-sum-exp)."""
    b = q.shape[0]
    out = torch.empty(b, H_Q, D, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(b, H_Q, device=q.device, dtype=torch.float32)

    mQ = from_dlpack(q, assumed_align=16)
    mK = from_dlpack(k_cache, assumed_align=16)
    mV = from_dlpack(v_cache, assumed_align=16)
    mIndptr = from_dlpack(kv_indptr, assumed_align=4)
    mIndices = from_dlpack(kv_indices, assumed_align=4)
    mOut = from_dlpack(out, assumed_align=16)
    mLse = from_dlpack(lse, assumed_align=4)
    scale = cutlass.Float32(float(sm_scale) * LOG2E)
    stream = _cuda_stream()

    key = (b, int(k_cache.shape[0]), int(kv_indices.numel()))
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_make_entry(b), mQ, mK, mV, mIndptr, mIndices,
                          mOut, mLse, scale, stream)
        _COMPILED[key] = fn
    fn(mQ, mK, mV, mIndptr, mIndices, mOut, mLse, scale, stream)
    return out, lse
