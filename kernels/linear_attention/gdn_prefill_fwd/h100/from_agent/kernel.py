"""gdn_prefill_fwd — the student's kernel (CuTe DSL, sm_90a).

Frozen signature (see README.md):

    gdn_prefill_fwd(q[N,16,128] bf16, k[N,16,128] bf16 (L2-normed),
                    v[N,32,128] bf16, a[N,32] fp32, beta[N,32] fp32,
                    scale: float)
        -> (o[N,32,128] bf16, final_state[1,32,128,128] fp32, V-major)

The recurrence, per v-head h (which reads q/k head h // 2), with S : [d, d]
(K-major: S[i, j] contracts the key index) starting at zero, for t = 0 .. N-1:

    S      <- a[t,h] * S                                   # decay
    S      <- S + beta[t,h] * k[t] (v[t,h] - S^T k[t])^T   # delta-erase + write
    o[t,h]  = S^T (scale * q[t])                           # inclusive readout

`final_state` is the state after token N-1, fp32, unscaled, and is reported
**V-major**: `final_state[0, h, v, k] = S[k, v]`. Gate and state math are fp32
end to end; io is bf16. `k` arrives L2-normalised.

Structure (two kernels, one CUDA graph replay per call):

  p1      auxiliary kernel, grid (16, 128) = one CTA per (qk-head, chunk),
          128 threads. Produces the chunk-local quantities that do not depend
          on the chunk's entry state: the gate records, P, T = (I + L)^-1, and
          the ready flag, for BOTH v-heads of its qk-head (D2.2: q.k^T and
          k.k^T depend on the qk-head alone, so one CTA shares the two GEMMs
          and the two operand tiles between the pair).
  chain3  main kernel, 64 CTAs in clusters of 4, 160 threads. Spins on p1's
          flags, consumes T/P/gate, carries the state across the 128 chunks,
          and writes o / final_state.

The two run concurrently on two streams inside the graph, but chain3's node is
created first (D2.3) so the scheduler places its clusters before p1's CTAs.

Chunk algebra (chunk = 64 tokens, lg_j = sum_{t<=j} log2 a_t, gamma_j = 2^lg_j,
Gamma = gamma_63, K/Q/V the 64x128 row blocks, S_0 the entry state):

    L[j][i] = beta_j * 2^(lg_j - lg_i) * (k_i . k_j)        i < j
    T       = (I + L)^-1
    P[i][j] = scale * 2^(lg_i - lg_j) * (q_i . k_j)         j <= i
    Bu      = T (D_beta V - D_{beta.gamma} K S_0)
    o       = D_{scale.gamma} Q S_0 + P Bu
    S_64    = Gamma S_0 + K^T D_{Gamma/gamma} Bu
"""

import threading

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.utils import LayoutEnum

# ----------------------------------------------------------------- shape / plan
NTOK = 8192
HQK = 16
HV = 32
DIM = 128
CHUNK = 64
NCHUNK = NTOK // CHUNK              # 128
NREC = NCHUNK * HV                  # 4096 (chunk, v-head) records
GATE_ROWS = 5                       # fp32 scalars recorded per token

P1_THREADS = 128                    # 4 warps
P1_CTAS_PER_SM = 2                  # D2.2 cost clause ("4 CTA/SM 可能降为 2-3"):
                                    # the launch bound is also the register cap
                                    # — min_blocks_per_mp=4 caps ptxas at
                                    # 65536/(128*4) = 128 regs/thread, and the
                                    # doubled D2.2 CTA does not fit in it: p1
                                    # compiled with STACK=248 B, LDL=58, STL=35
                                    # (graph NCU spill_ld_sectors 1966080,
                                    # dram_bytes_write 132.4 -> 193.3 MB,
                                    # device 0.27364 -> 0.28357 ms).
                                    # Same-job 4-arm A/B, every arm one textual
                                    # substitution off this line and bitwise
                                    # equal to base in o and final_state
                                    # (notes/probe_d22_spill.py), cubin
                                    # res-usage + shipped-graph median:
                                    #   4    reg=128 stack=248 LDL=58 STL=35
                                    #        graph 286.4 us  p1 solo 149.1 us
                                    #   3    reg=168 stack=88  LDL=21 STL=16
                                    #        graph 270.9 us  p1 solo 129.9 us
                                    #   2    reg=221 stack=0   LDL=0  STL=0
                                    #        graph 276.9 us  p1 solo 134.8 us
                                    #   none reg=231 stack=0   LDL=0  STL=0
                                    #        graph 277.4 us  p1 solo 135.0 us
                                    # 3 has the fastest graph but still spills;
                                    # the standing gate line is
                                    # spill_ld_sectors == 0, so the pick is the
                                    # fastest spill-free arm, 2 (-3.30% graph).
                                    # Dropping the bound entirely is the same
                                    # occupancy with 10 more registers and no
                                    # guarantee, so the bound stays explicit.
P1_BOX = 64                         # one K_SW128 box = 64 bf16 = 128 B per row
P1_NBOX = DIM // P1_BOX             # 2 boxes side by side along d
FACE_PITCH = 72                     # 64 + 8 pad: keeps the C-fragment stores and
                                    # the ldmatrix rows off a single bank column

C3_CTAS = 64                        # D0.4: ctaid & 1 = d_v half, ctaid >> 1 = head
C3_THREADS = 160                    # D0.5: 1 consumer warpgroup + 1 producer warp
C3_CLUSTER = 4
SPIN_LIMIT = 4000000                # bounded flag spin: never turns into a hang
NANOS = 32                          # D1.5 lever 3: flag-spin backoff gear 0x20.
                                    # Same-job 7-arm A/B (notes/probe_d15_wait_
                                    # gears.py): graph median 287.9 / 287.9 /
                                    # 288.8 / 288.1 us at 0x20 / 0x40 / 0x80 /
                                    # 0x100, outputs bitwise equal across arms.
                                    # (reference backs off with NANOSLEEP 0x40)

NSTAGE = 3                          # D0.7: slot = chunk mod 3
                                    # D1.6 lever 2: kept at 3. The NSTAGE = 2
                                    # arm of notes/probe_d16_depth_stage.py
                                    # measured graph median 308.7 us vs 288.2
                                    # (+7.12%), solo 209.5 vs 188.1 us, output
                                    # bitwise equal - nothing in v1 wants the
                                    # freed smem, so 2 is a control only.
                                    # D1.6 lever 3: swizzle stays at the 128 B
                                    # starting point (K_SW128), the only
                                    # conflict-free atom for the 64x64 bf16
                                    # boxes per
                                    # hw_kb/cutedsl/sm90/ldmatrix_wideK_conflict_free_face
                                    # - not armed.
GATE_W = GATE_ROWS * CHUNK          # 320 fp32 = 1280 B per record
GATE_PAD = 512                      # fp32 per stage slot: keeps the slot 2 KiB
TILE_B = CHUNK * CHUNK * 2          # 8192 B, one box
TX_BYTES = 7 * TILE_B + GATE_W * 4  # 58624 = 0xe500, D0.7's expect_tx
WAIT_BUDGET = 400000                # bounded mbarrier wait

EPI_BUF = 2                         # D0.11: the o staging is double buffered
EPI_BAR = 1                         # NamedBarrier 0x1; 0x0 is sync_threads'

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32


def _seg(site, row=0):
    """Timeline-probe hook. The driver rewrites this in the probe arm only."""
    return None


# ------------------------------------------------------------------ PTX escapes
# hw_kb/cutedsl/sm90/ptx_escape.md form 2/3: instructions the DSL does not export, or
# exports only behind a deprecation. Pure ones carry has_side_effects=False.


@dsl_user_op
def _ex2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """2**x, hardware approximation (MUFU.EX2)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """base-2 logarithm, hardware approximation (MUFU.LG2)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.ftz.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _mk_sel(cmp_op):
    """x <cmp> y ? a : b, on int32 predicates -> one ISETP + one FSEL."""

    @dsl_user_op
    def _sel(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
        return cutlass.Float32(llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(x).ir_value(loc=loc, ip=ip),
             cutlass.Int32(y).ir_value(loc=loc, ip=ip),
             cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.%s.s32 p, $1, $2;"
            "\n\tselp.f32 $0, $3, $4, p;\n}" % cmp_op,
            "=f,r,r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    return _sel


_sel_le = _mk_sel("le")     # x <= y ? a : b
_sel_ge = _mk_sel("ge")     # x >= y ? a : b
_sel_gt = _mk_sel("gt")     # x >  y ? a : b
_sel_eq = _mk_sel("eq")     # x == y ? a : b


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait_all(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.wait_group 0;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _ld_acquire_u32(addr, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _st_release_u32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
               cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;", "l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _st_relaxed_u32(addr, val, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
               cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "st.relaxed.gpu.global.u32 [$0], $1;", "l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None):
    llvm.inline_asm(
        None, [cutlass.Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;", "r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _isel_eq(x, y, a, b, *, loc=None, ip=None) -> cutlass.Int32:
    """x == y ? a : b, on int32 — one ISETP + one SEL."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip),
         cutlass.Int32(y).ir_value(loc=loc, ip=ip),
         cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;"
        "\n\tselp.b32 $0, $3, $4, p;\n}", "=r,r,r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _mbar_try_wait(addr, phase, *, loc=None, ip=None) -> cutlass.Int32:
    """hw_kb/cutedsl/sm90/mbarrier_wait_encoding form (e'): one parity attempt -> 1/0.
    has_side_effects AND the ~{memory} clobber, or the attempt is hoisted out
    of the retry loop and the wait never terminates."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
        "selp.b32 $0, 1, 0, p;\n}", "=r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _bulk_g2s(dst_smem, src_gmem, nbytes, bar_smem, *, loc=None, ip=None):
    """D0.7's gate record: the 1D bulk load. The bare PTX is a per-thread
    instruction with no internal elect (hw_kb/cutedsl/sm90/s2g_bulk_reduce gotcha 3),
    so the caller does the electing."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src_gmem).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
         cutlass.Int32(bar_smem).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global."
        "mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def _bar_wait(bar, phase):
    """Bounded parity wait. A wrong arrive count or a wrong phase gives up
    instead of wedging the whole eval job; the give-up term rides on the
    attempt so the loop condition stays a single comparison.

    D1.5 levers 1 and 2 keep this non-blocking (e') tight spin at both the
    consumer full-wait and the producer empty-wait call sites. Same-job A/B
    against `cute.arch.mbarrier_wait` (compiler retry loop + NANOSLEEP
    suspend), notes/probe_d15_wait_gears.py, graph median vs this form:
    consumer blocking +4.07%, producer blocking +0.13%, both +4.44%."""
    addr = bar.toint()
    n = I32(0)
    ok = _mbar_try_wait(addr, phase)
    while ok == 0:
        n = n + 1
        ok = _mbar_try_wait(addr, phase) + n // WAIT_BUDGET


# ------------------------------------------------------------------ smem layout
# 33808 B: two 64x128 bf16 operand tiles + 2 x 128 fp32 scalars (one set per
# v-head of the pair) + the four warp carries of the gate scan. The two 64x72
# bf16 triangular faces are second readings of the q and the k range, taken
# after both operand tiles are dead (D2.2: one face per v-head, still no extra
# smem — the pair's two faces are 9216 B each and q + k is 32768 B).


@cute.struct
class P1Smem:
    q: cute.struct.Align[cute.struct.MemRange[BF16, CHUNK * DIM], 1024]
    k: cute.struct.Align[cute.struct.MemRange[BF16, CHUNK * DIM], 1024]
    sc: cute.struct.MemRange[F32, 2 * 2 * CHUNK]
    tot: cute.struct.MemRange[F32, 4]


# The readings of the two smem ranges, in element units. Built inside the
# kernel body: layouts are trace-time objects, there is nothing to pass in.
#
#   w3      (r, c_in_box, box) of a 64x128 operand face — cp.async destination
#   a4      (m, k, row_block, k_block) A operand over the face
#   b3      (n, k, k_block) B operand over the face, k contiguous
#   face    (r, c, f) of the 64x72 triangular face of v-head f of the pair
#   fa16    (m, k, rb, cb, f)     16x16 A operand block, k contiguous
#   fb16    (n, k, cb, rb, f)     16x16 B operand block, n contiguous
#   fa32    (m, k, rb16, cb32, f) 16x32 A operand
#   fb32    (n, k, cb32, rb32, f) 32x32 B operand, n contiguous
#   sc      (f, i): lg[0:64] then beta[64:128] of v-head f of the pair
#
# The face mode's stride is the whole 64x128 operand tile, so face 0 lives on
# the dead q range and face 1 on the dead k range; the index is dynamic (it is
# the warp's half of the pair), which is why it is a layout mode and not two
# tensors.


def _p1_faces():
    """hw_kb/cutedsl/sm90/ldmatrix_wideK_conflict_free_face: a 64x128 bf16 K-contiguous
    face has a 256 B row pitch, so the eight row addresses of every ldmatrix
    8x8 phase land in one 16 B bank group — an 8-way conflict on *every*
    operand fetch, invisible to correctness. Storing the same bytes as two
    K_SW128 boxes (128 B pitch, 16 B chunks XOR-permuted by row mod 8) costs
    zero extra smem and makes each phase one wavefront.

    The three views are all `composition`s of the one face, so the write side
    and both read sides address the same logical cells. Their right operands
    live in the face's colex coordinate space (index = r + 64 * c), NOT in
    memory strides: "row moves by 1, column moves by M" (gotcha 1)."""
    atom = warpgroup.make_smem_layout_atom(
        warpgroup.SmemLayoutAtomKind.K_SW128, BF16)
    face = cute.tile_to_shape(atom, (CHUNK, DIM), order=(0, 1))
    w3 = cute.composition(face, cute.make_layout(
        (CHUNK, P1_BOX, P1_NBOX), stride=(1, CHUNK, CHUNK * P1_BOX)))
    a4 = cute.composition(face, cute.make_layout(
        (16, 16, CHUNK // 16, DIM // 16),
        stride=(1, CHUNK, 16, 16 * CHUNK)))
    b3 = cute.composition(face, cute.make_layout(
        (CHUNK, 16, DIM // 16), stride=(1, CHUNK, 16 * CHUNK)))
    return w3, a4, b3


# ------------------------------------------------------------- acc -> A operand


# The three helpers below are deliberately UNDECORATED: they contain no DSL
# control flow, only trace-time Python loops over static sizes. The
# preprocessor does not enter undecorated callees, so a plain `range` stays a
# real Python loop and the constexpr-in-helper trap never comes up
# (hw_kb/cutedsl/sm90/warp_mma_f16 gotcha 11, cp_async gotcha 7b).


def _chain_a(thr, acc, m_dim, n_dim):
    """hw_kb/cutedsl/sm90/warp_mma_f16 section 4: the fp32 accumulator of one
    `mma.sync` is the A operand of the next, with no smem round trip. Only the
    grouping differs, so the destination fragment is written through a strided
    view in the accumulator's own linear order."""
    op_a = thr.make_fragment_A(thr.partition_shape_A((m_dim, n_dim)))
    cv = cute.size(acc, mode=[0])
    av = cute.size(op_a, mode=[0])
    ratio = av // cv
    m_rest = cute.size(acc, mode=[1])
    k_rest = cute.size(acc, mode=[2]) // ratio
    view = cute.make_tensor(
        op_a.iterator,
        cute.make_layout((cv, m_rest, ratio, k_rest),
                         stride=(1, av, cv, av * m_rest)))
    for i in range(cute.size(acc)):
        view[i] = BF16(acc[i])
    return op_a


def _gemm_ldm(mma, tc_a, tc_b, lane, s_a, s_b, acc):
    """One `cute.gemm` with both operands fetched by ldmatrix."""
    thr = mma.get_slice(lane)
    thr_a, thr_b = tc_a.get_slice(lane), tc_b.get_slice(lane)
    fr_a = thr.make_fragment_A(thr.partition_A(s_a))
    fr_b = thr.make_fragment_B(thr.partition_B(s_b))
    cute.copy(tc_a, thr_a.partition_S(s_a), thr_a.retile(fr_a))
    cute.copy(tc_b, thr_b.partition_S(s_b), thr_b.retile(fr_b))
    cute.gemm(mma, acc, fr_a, fr_b, acc)


def _gemm_chained(mma, tc_b, lane, acc_prev, s_b, acc, m_dim, n_prev):
    """Second step of a two-step product: A comes from `acc_prev`."""
    thr = mma.get_slice(lane)
    thr_b = tc_b.get_slice(lane)
    op_a = _chain_a(thr, acc_prev, m_dim, n_prev)
    fr_b = thr.make_fragment_B(thr.partition_B(s_b))
    cute.copy(tc_b, thr_b.partition_S(s_b), thr_b.retile(fr_b))
    cute.gemm(mma, acc, op_a, fr_b, acc)


# ---------------------------------------------------------------- the p1 kernel


@cute.kernel
def _k_p1(mQ: cute.Tensor, mK: cute.Tensor, mA: cute.Tensor, mB: cute.Tensor,
          mT: cute.Tensor, mP: cute.Tensor, mG: cute.Tensor,
          flag_base: cutlass.Int64, scale: cutlass.Float32,
          mma, tc_a, tc_bn, tc_bt):
    tidx, _, _ = cute.arch.thread_idx()
    qh, chunk, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    wid = cute.arch.warp_idx()
    _seg("p1.entry")

    # D2.2: the CTA is a (qk-head, chunk), and it serves the two v-heads
    # 2*qh and 2*qh+1. Both GEMMs (q.k^T and k.k^T) and both operand tiles
    # depend on the qk-head alone, so they are computed once here instead of
    # once per v-head; everything downstream of them (the gate scan, the two
    # mask/scale passes, the inverse, T) is per v-head. Warps 0-1 carry
    # v-head 2*qh and warps 2-3 carry 2*qh+1 wherever the work splits by
    # v-head; all four warps take 16 rows of each shared object.
    t0 = chunk * CHUNK
    rec0 = chunk * HV + 2 * qh          # the pair's records are rec0, rec0+1
    fidx = wid // 2                     # this warp's v-head within the pair
    sub = wid % 2                       # this warp's half of that v-head
    head = 2 * qh + fidx
    rec = rec0 + fidx

    ml = cute.make_layout
    w3, a4, b3 = _p1_faces()

    st = cutlass.utils.SmemAllocator().allocate(P1Smem)
    sQw = st.q.get_tensor(w3.outer, swizzle=w3.inner)
    sKw = st.k.get_tensor(w3.outer, swizzle=w3.inner)
    sQa = st.q.get_tensor(a4.outer, swizzle=a4.inner)
    sKa = st.k.get_tensor(a4.outer, swizzle=a4.inner)
    sKb = st.k.get_tensor(b3.outer, swizzle=b3.inner)
    sSc = st.sc.get_tensor(ml((2, 2 * CHUNK), stride=(2 * CHUNK, 1)))
    sTot = st.tot.get_tensor(ml(4))

    # ---- seg0a: cp.async the two 64x128 operand faces (16 B per thread) -----
    # Eight lanes cover one 128 B row inside one box, which is the write form
    # the face recipe certifies; the row index is a dynamic team coordinate.
    cp_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    cp_atom = cute.make_copy_atom(cp_op, BF16, num_bits_per_copy=128)
    cp_tiled = cute.make_tiled_copy_tv(cp_atom, cute.make_layout(8),
                                       cute.make_layout(8))
    cp_thr = cp_tiled.get_slice(tidx % 8)
    team = tidx // 8
    l_grow = ml((P1_BOX, P1_NBOX), stride=(1, P1_BOX))
    for i in cutlass.range_constexpr(CHUNK // 16):
        r = i * 16 + team
        gQr = cute.make_tensor(mQ[(t0 + r, qh, None)].iterator, l_grow)
        gKr = cute.make_tensor(mK[(t0 + r, qh, None)].iterator, l_grow)
        for box in cutlass.range_constexpr(P1_NBOX):
            cute.copy(cp_tiled, cp_thr.partition_S(gQr[(None, box)]),
                      cp_thr.partition_D(sQw[(r, None, box)]))
            cute.copy(cp_tiled, cp_thr.partition_S(gKr[(None, box)]),
                      cp_thr.partition_D(sKw[(r, None, box)]))
    _cp_async_commit()

    # ---- seg0b: log-domain gate scan, entirely in fp32 ----------------------
    # Ratios are always exp2 of a difference of prefix sums: a 64-step product
    # of a ~ U(0,1) underflows fp32, a difference of logs does not.
    #
    # D2.1 lever 2 (p1 per-CTA latency): the prefix sum is D0.13's warp scan —
    # five `shfl.up` steps and one cross-warp carry through smem — in place of
    # every thread walking all 64 log terms out of smem (64 dependent LDS +
    # FADD + SELP per thread, and a second CTA rendezvous to protect the
    # in-place overwrite of the term array). Every thread runs the same five
    # steps so the shuffles stay outside any branch. The tree order sums the
    # same fp32 terms in a different association than the serial walk, which
    # is a rounding difference, not a definition change.
    #
    # D2.2: the two warps that used to rescan the pair's other v-head's tokens
    # redundantly now scan that v-head for real — warps 0-1 scan v-head 2*qh,
    # warps 2-3 scan 2*qh+1, each pair's two warps carrying tokens 0-31 and
    # 32-63 and meeting through its own two `sTot` slots. Every one of the 128
    # threads then owns exactly one (v-head, token) gate record column, so the
    # scan and the record stores lose their `tidx < CHUNK` predicate.
    tt = tidx % CHUNK
    bj = mB[(t0 + tt, head)]
    v = _lg2(mA[(t0 + tt, head)])
    for d in cutlass.range_constexpr(5):
        # `mask_and_clamp` is a *lower* bound for shfl.up: the DSL default of
        # WARP_SIZE-1 fails the predicate on every lane, which then reads its
        # own register instead of lane-2^d. 0 is the whole-warp scan form.
        y = cute.arch.shuffle_sync_up(v, 1 << d, mask_and_clamp=0)
        v = v + _sel_ge(lane, 1 << d, y, F32(0.0))
    sSc[(fidx, CHUNK + tt)] = bj
    if lane == 31:
        sTot[wid] = v                               # this warp's 32-token sum
    _seg("p1.scan.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.scan.bar.end")

    w0 = sTot[2 * fidx]
    tot = w0 + sTot[2 * fidx + 1]
    pre = v + _sel_eq(sub, 1, w0, F32(0.0))
    sSc[(fidx, tt)] = pre
    gj = _ex2(pre)
    mG[(rec, 0, tt)] = bj                           # beta_j
    mG[(rec, 1, tt)] = bj * gj                      # beta_j * gamma_j
    mG[(rec, 2, tt)] = _ex2(tot - pre)              # Gamma / gamma_j
    mG[(rec, 3, tt)] = scale * gj                   # scale * gamma_j
    mG[(rec, 4, tt)] = _ex2(tot)                    # Gamma

    _seg("p1.supply.wait.begin")
    _cp_async_wait_all()
    cute.arch.sync_threads()
    _seg("p1.supply.wait.end")

    # ---- seg1a: P = mask(scale * 2^(lg_i - lg_j) * (q_i . k_j)) -------------
    # D2.2: one q.k^T for the pair. The 64x64 product is v-head independent —
    # only the mask/scale pass that turns it into P is per v-head — so the
    # eight k-steps run once and the accumulator is read twice.
    w16 = wid * 16
    acc_p = cute.make_rmem_tensor(
        mma.get_slice(lane).partition_shape_C((16, CHUNK)), F32)
    acc_p.fill(0.0)
    for kb in cutlass.range_constexpr(DIM // 16):
        _gemm_ldm(mma, tc_a, tc_bn, lane,
                  sQa[(None, None, wid, kb)], sKb[(None, None, kb)], acc_p)

    cCrd = mma.get_slice(lane).partition_C(
        cute.make_identity_tensor((16, CHUNK)))
    # D2.1 lever 2: the C fragment's N-rest mode only moves the column, so the
    # two row scalars each thread needs are the same for all 8 atoms — read
    # once, ahead of the whole mask pass, and read each atom's column scalars
    # as a group of independent LDS before the group's stores rather than one
    # scalar at a time inside the store stream.
    # The C partition is ((2,2), M_rest, N_rest) and M_rest is 1 here, so the
    # linear element index is e = i + 4*g and only g moves the column.
    assert cute.size(acc_p, mode=[0]) == 4 and cute.size(acc_p, mode=[1]) == 1
    NAT = cute.size(acc_p, mode=[2])
    crd0 = [cCrd[i] for i in range(4)]
    prow = [w16 + crd0[i][0] for i in range(4)]
    for f in cutlass.range_constexpr(2):
        plg = [sSc[(f, prow[i])] for i in range(4)]
        gPr = mP[(rec0 + f, None, None)]
        for g in cutlass.range_constexpr(NAT):
            pcol = [cCrd[4 * g + i][1] for i in range(4)]
            pcl = [sSc[(f, pcol[i])] for i in range(4)]
            for i in cutlass.range_constexpr(4):
                val = scale * _ex2(plg[i] - pcl[i]) * acc_p[4 * g + i]
                gPr[(prow[i], pcol[i])] = BF16(
                    _sel_ge(prow[i], pcol[i], val, F32(0.0)))

    # ---- seg1b: L = mask(beta_j * 2^(lg_j - lg_i) * (k_i . k_j)) -----------
    # Also one GEMM for the pair. It is issued before any store into the
    # triangular faces, because those faces are the q and the k operand tile:
    # the barrier below is the one point where both tiles are provably dead.
    acc_l = cute.make_rmem_tensor(
        mma.get_slice(lane).partition_shape_C((16, CHUNK)), F32)
    acc_l.fill(0.0)
    for kb in cutlass.range_constexpr(DIM // 16):
        _gemm_ldm(mma, tc_a, tc_bn, lane,
                  sKa[(None, None, wid, kb)], sKb[(None, None, kb)], acc_l)
    _seg("p1.gemm1.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.gemm1.bar.end")

    fp = FACE_PITCH
    ff = CHUNK * DIM                    # face 0 on the q tile, face 1 on k
    fRC = st.q.get_tensor(ml((CHUNK, fp, 2), stride=(fp, 1, ff)))
    fA16 = st.q.get_tensor(
        ml((16, 16, 4, 4, 2), stride=(fp, 1, 16 * fp, 16, ff)))
    fB16 = st.q.get_tensor(
        ml((16, 16, 4, 4, 2), stride=(1, fp, 16, 16 * fp, ff)))
    fA32 = st.q.get_tensor(
        ml((16, 32, 4, 2, 2), stride=(fp, 1, 16 * fp, 32, ff)))
    fB32 = st.q.get_tensor(
        ml((32, 32, 2, 2, 2), stride=(1, fp, 32, 32 * fp, ff)))

    # Two mask/scale passes over the one k.k^T, one per v-head of the pair,
    # each into its own face. Same gather as the P pass, and here it also
    # breaks an alias stall: the face and the scalars are both smem, so a
    # scalar load issued between two face stores cannot be reordered ahead of
    # them.
    for f in cutlass.range_constexpr(2):
        lbt = [sSc[(f, CHUNK + prow[i])] for i in range(4)]
        llg = [sSc[(f, prow[i])] for i in range(4)]
        for g in cutlass.range_constexpr(NAT):
            lcol = [cCrd[4 * g + i][1] for i in range(4)]
            lcl = [sSc[(f, lcol[i])] for i in range(4)]
            for i in cutlass.range_constexpr(4):
                val = lbt[i] * _ex2(llg[i] - lcl[i]) * acc_l[4 * g + i]
                fRC[(prow[i], lcol[i], f)] = BF16(
                    _sel_gt(prow[i], lcol[i], val, F32(0.0)))
    _seg("p1.face.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.face.bar.end")

    # ---- seg2: the eight 16x16 diagonal blocks, two per warp ---------------
    # Column c of (I + A)^-1 by forward substitution; lane c owns the column,
    # the A entries are warp-uniform broadcast reads.
    # D2.2: there are two faces and still four warps, so each warp inverts the
    # two diagonal blocks of the 32x32 half it owns — which are exactly the
    # blocks its own seg3 consumes, so this level no longer needs a CTA-wide
    # rendezvous behind it (seg3 reads T_ii, L_ij, T_jj of one half only).
    for bb in cutlass.range_constexpr(2):
        o16 = sub * 32 + bb * 16
        xs = [F32(0.0)] * 16
        for r in cutlass.range_constexpr(16):
            s = F32(0.0)
            for m in cutlass.range_constexpr(r):
                s = s + F32(fRC[(o16 + r, o16 + m, fidx)]) * xs[m]
            xs[r] = _sel_eq(r, lane, F32(1.0), -s)
        cute.arch.sync_warp()
        if lane < 16:
            for r in cutlass.range_constexpr(16):
                fRC[(o16 + r, o16 + lane, fidx)] = BF16(xs[r])
    # The stores above are lane-predicated and seg3's ldmatrix is warp-wide,
    # so the level still ends on a warp fence — just not on a CTA-wide one.
    _seg("p1.lvl0.bar.begin")
    cute.arch.sync_warp()
    _seg("p1.lvl0.bar.end")

    # ---- seg3: 16 -> 32. Each 32x32 diagonal half is [[T_jj, 0], [L_ij, T_ii]],
    # so its corner block is T_{i,j} = -(T_ii L_ij) T_jj, i = 2p+1, j = 2p.
    # The four halves (two faces x two warps) are disjoint and each warp reads
    # its L_ij before overwriting it.
    ii = 2 * sub + 1
    jj = 2 * sub
    thr = mma.get_slice(lane)
    a1 = cute.make_rmem_tensor(thr.partition_shape_C((16, 16)), F32)
    a1.fill(0.0)
    _gemm_ldm(mma, tc_a, tc_bt, lane, fA16[(None, None, ii, ii, fidx)],
              fB16[(None, None, jj, ii, fidx)], a1)
    a2 = cute.make_rmem_tensor(thr.partition_shape_C((16, 16)), F32)
    a2.fill(0.0)
    _gemm_chained(mma, tc_bt, lane, a1, fB16[(None, None, jj, jj, fidx)], a2,
                  16, 16)
    c2 = thr.partition_C(cute.make_identity_tensor((16, 16)))
    for e in cutlass.range_constexpr(cute.size(a2)):
        rr, cc = c2[e]
        fRC[(ii * 16 + rr, jj * 16 + cc, fidx)] = BF16(-a2[e])

    # seg4/5 reads both halves of its face (T11 from warp sub 0, T22 from warp
    # sub 1), so this level does need the rendezvous. All four warps now hold
    # data at every level, so it is the plain CTA-wide one again.
    _seg("p1.lvl1.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.lvl1.bar.end")

    # ---- seg4/5: 32 -> 64. The lower-left 32x32 corner block, two warps per
    # face, 16 output rows each: Z = -(T22 A21) T11, with A21 still holding L.
    b1 = cute.make_rmem_tensor(thr.partition_shape_C((16, 32)), F32)
    b1.fill(0.0)
    _gemm_ldm(mma, tc_a, tc_bt, lane, fA32[(None, None, 2 + sub, 1, fidx)],
              fB32[(None, None, 0, 1, fidx)], b1)
    b2 = cute.make_rmem_tensor(thr.partition_shape_C((16, 32)), F32)
    b2.fill(0.0)
    _gemm_chained(mma, tc_bt, lane, b1, fB32[(None, None, 0, 0, fidx)], b2,
                  16, 32)
    c3 = thr.partition_C(cute.make_identity_tensor((16, 32)))
    # A21 is this level's B operand and this level's destination: both warps
    # of the face must be done loading before either writes.
    _seg("p1.lvl2.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.lvl2.bar.end")
    for e in cutlass.range_constexpr(cute.size(b2)):
        rr, cc = c3[e]
        fRC[(32 + sub * 16 + rr, cc, fidx)] = BF16(-b2[e])

    # ---- seg6: publish T (32 rows per warp, two records per CTA) -----------
    _seg("p1.store.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.store.bar.end")
    gTr = mT[(rec, None, None)]
    for i in cutlass.range_constexpr(32):
        row = sub * 32 + i
        for jj2 in cutlass.range_constexpr(2):
            col = lane + 32 * jj2
            gTr[(row, col)] = fRC[(row, col, fidx)]

    # ---- seg7: release both records, then raise both flags -----------------
    # P is written by all four warps and T / the gate records by the two warps
    # of their v-head, so the release is CTA-wide for both flags.
    _seg("p1.flag.bar.begin")
    cute.arch.sync_threads()
    _seg("p1.flag.bar.end")
    cute.arch.fence_acq_rel_gpu()
    if tidx < 2:
        _st_release_u32(flag_base + cutlass.Int64(rec0 + tidx) * 4, 1)
    _seg("p1.publish")


@cute.jit
def _h_p1(mQ: cute.Tensor, mK: cute.Tensor, mA: cute.Tensor, mB: cute.Tensor,
          mT: cute.Tensor, mP: cute.Tensor, mG: cute.Tensor,
          flag_base: cutlass.Int64, scale: cutlass.Float32, stream):
    op = warp.MmaF16BF16Op(BF16, F32, (16, 8, 16))
    mma = cute.make_tiled_mma(op, (1, 1, 1))
    # num_matrices follows the tiler's element count: A (16,16) = 4, B (8,16) = 2
    tc_a = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma)
    tc_bn = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), BF16), mma)
    tc_bt = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), BF16), mma)
    # D2.2: grid.x is the qk-head, not the v-head. The dispatch order is
    # unchanged in kind — x fastest, so the hardware still launches
    # chunk-major and the flags come up in chain3's consumption order.
    _k_p1(mQ, mK, mA, mB, mT, mP, mG, flag_base, scale,
          mma, tc_a, tc_bn, tc_bt).launch(
        grid=[HQK, NCHUNK, 1], block=[P1_THREADS, 1, 1],
        min_blocks_per_mp=P1_CTAS_PER_SM, stream=stream)


# --------------------------------------------------------------- chain3 buffers
# D0.7's buffer list, 194608 B <= the 195648 B envelope. The ranges are ordered
# so every 1024-aligned MemRange lands on an already-aligned offset.
#
#   q, k   64 tok x 128 d bf16 per stage = 2 boxes each   3 x 16 KiB  (x2)
#   v      64 tok x 64 v bf16 per stage  = 1 box          3 x  8 KiB
#   t, p   64 x 64 bf16 per stage        = 1 box each     3 x  8 KiB  (x2)
#   g      gate record 1280 B, padded to a 2 KiB slot     3 x  2 KiB
#   ost    the o staging double buffer (D0.11)                      16 KiB
#   mbar   full[3] + empty[3]


@cute.struct
class C3Smem:
    q: cute.struct.Align[cute.struct.MemRange[BF16, CHUNK * DIM * NSTAGE], 1024]
    k: cute.struct.Align[cute.struct.MemRange[BF16, CHUNK * DIM * NSTAGE], 1024]
    v: cute.struct.Align[
        cute.struct.MemRange[BF16, CHUNK * CHUNK * NSTAGE], 1024]
    t: cute.struct.Align[
        cute.struct.MemRange[BF16, CHUNK * CHUNK * NSTAGE], 1024]
    p: cute.struct.Align[
        cute.struct.MemRange[BF16, CHUNK * CHUNK * NSTAGE], 1024]
    g: cute.struct.Align[cute.struct.MemRange[F32, GATE_PAD * NSTAGE], 1024]
    ost: cute.struct.Align[cute.struct.MemRange[BF16, 2 * CHUNK * CHUNK], 1024]
    mbar: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int64, 2 * NSTAGE], 1024]


def _c3_layouts():
    """D0.7's staged smem views. 64 bf16 of contiguous mode = 1024 bits, so the
    atom is K_SW128 and one (64, 64) box is a contiguous 8 KiB swizzled tile —
    which is what lets the TMA box view and the wgmma operand view alias byte
    for byte over the same range."""
    al = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, CHUNK),
        BF16)
    box_l = cute.tile_to_shape(al, (CHUNK, CHUNK), order=(0, 1))
    # the canonical k-major operand view of a q/k stage: (tok, d, stage)
    sQ_l = cute.tile_to_shape(al, (CHUNK, DIM, NSTAGE), order=(0, 1, 2))
    sV_l = cute.tile_to_shape(al, (CHUNK, CHUNK, NSTAGE), order=(0, 1, 2))
    # the TMA box view of the same bytes: mode 2 runs over 2*stage + box, which
    # is exactly the canonical view's (d // 64) + 2*stage.
    sQb_l = cute.tile_to_shape(al, (CHUNK, CHUNK, 2 * NSTAGE), order=(0, 1, 2))
    # chain2 reads the same k bytes MN-major (hw_kb/cutedsl/sm90/wgmma_mn_major_operand
    # section 3): the write view's (t, d, s) colex domain index is
    # t + d*64 + s*64*128, so the transposer for (d, t, s) has stride
    # (64, 1, 8192).
    sKmn_l = cute.composition(sQ_l, cute.make_layout(
        (DIM, CHUNK, NSTAGE), stride=(CHUNK, 1, CHUNK * DIM)))
    # the v half block read back as (v, tok, s) — the C-fragment orientation of
    # chain0's accumulator. Same colex recipe: sV_l's (t, v, s) index is
    # t + v*64 + s*64*64.
    sVt_l = cute.composition(sV_l, cute.make_layout(
        (CHUNK, CHUNK, NSTAGE), stride=(CHUNK, 1, CHUNK * CHUNK)))
    sG_l = cute.make_layout((GATE_ROWS, CHUNK, NSTAGE),
                            stride=(CHUNK, 1, GATE_PAD))
    return sQ_l, sQb_l, sV_l, sKmn_l, sVt_l, sG_l, box_l


# ---------------------------------------------- the consumer's five MMA chains
# D0.1's chunk algebra in transposed form (accumulator rows = v, columns = token
# or k), D0.6's operand ledger (every chain's A comes from the previous chain's
# fp32 accumulator, repacked in place; every B is an smem descriptor), and
# D0.9's issue order (chain0+chain1 in flight together, the `t` segment hidden
# behind chain1). Wait depths after D1.3 + D1.4: LE1 (chain0) / LE1 (chain1,
# chain0b in flight) / LE0 (chain0b) / LE1 (chain1b, chain2 in flight) / LE0
# (chain2, in the caller, after the epilogue). Five chains, five commit groups.
#
#   chain0   (K S_0)^T  = S^T . K^T          A = S^T  [64 v, 128 k]  B = k box
#   chain1   (Q S_0)^T  = S^T . Q^T          A = S^T                 B = q box
#   chain0b  Bu^T       = t . T^T            A = t    [64 v, 64 tok] B = T
#   chain2   S^T       += (D_{G/g} Bu)^T . K A = Bu^T . (G/g)        B = k, MN
#   chain1b  o^T       += Bu^T . P^T         A = Bu^T                B = P


def _pack_pairs(src_pv, dst_pv):
    """fp32 C fragment -> bf16 RMEM A fragment, two adjacent columns per cvt.

    hw_kb/cutedsl/sm90/wgmma_interleave gotcha 10: the m64nNk16 accumulator fragment and
    the m64k16 RMEM A fragment own the same cells in the same linear order, so
    the conversion is a pairwise cvt over two flat views of the two iterators
    and never crosses a lane boundary."""
    for j in range(cute.size(src_pv, mode=[1])):
        v2 = cute.arch.cvt_f32x2_bf16x2(src_pv[(None, j)].load())
        dst_pv[(None, j)].store(cute.TensorSSA(v2, (2,), BF16))


class _C3Regs:
    """Trace-time holder for the consumer's fragments, views and extents."""

    __slots__ = ("sVt", "l2r", "thr_l", "cC", "acc_s", "acc_o", "acc_c", "rV",
                 "frA_s", "frA_x", "frA_g", "frB_k", "frB_q", "frB_t", "frB_p",
                 "frB_n", "pv_s_src", "pv_s_dst", "pv_c_src", "pv_x_dst",
                 "pv_g_dst", "pv_o_src", "pv_o_dst", "nk_d", "nk_c", "ngrp",
                 "ncol", "epi_d", "epi_s")


def _c3_setup(mma_ss, mma_st, tw, sQ, sK, sKmn, sVt, sT, sP, l2r):
    """D0.6's register budget, allocated once outside the chunk loop.

    64 (S^T) + 32 (o^T) + 32 (chain0 C, reused by chain0b) + 32 (bf16 S^T)
    + 16 (t, reused by the unscaled Bu pack) + 16 (the scaled Bu pack)
    + 16 (v^T). The A fragments of chain0b and chain1b share one allocation:
    chain0b has retired by the time chain1b's operand is packed."""
    R = _C3Regs()
    wg = mma_ss.get_slice(0)            # SS descriptors are warpgroup uniform
    wg2 = mma_st.get_slice(0)
    thr = mma_ss.get_slice(tw)          # the C fragment is per thread
    R.sVt = sVt
    R.l2r = l2r
    R.thr_l = l2r.get_slice(tw)
    R.cC = thr.partition_C(cute.make_identity_tensor((CHUNK, CHUNK)))
    R.acc_s = cute.make_rmem_tensor(wg2.partition_shape_C((CHUNK, DIM)), F32)
    R.acc_o = cute.make_rmem_tensor(wg.partition_shape_C((CHUNK, CHUNK)), F32)
    R.acc_c = cute.make_rmem_tensor(wg.partition_shape_C((CHUNK, CHUNK)), F32)
    R.rV = cute.make_rmem_tensor(wg.partition_shape_C((CHUNK, CHUNK)), BF16)
    R.frA_s = wg.make_fragment_A(wg.partition_shape_A((CHUNK, DIM)))
    R.frA_x = wg.make_fragment_A(wg.partition_shape_A((CHUNK, CHUNK)))
    R.frA_g = wg2.make_fragment_A(wg2.partition_shape_A((CHUNK, CHUNK)))
    R.frB_k = wg.make_fragment_B(wg.partition_B(sK))
    R.frB_q = wg.make_fragment_B(wg.partition_B(sQ))
    R.frB_t = wg.make_fragment_B(wg.partition_B(sT))
    R.frB_p = wg.make_fragment_B(wg.partition_B(sP))
    R.frB_n = wg2.make_fragment_B(wg2.partition_B(sKmn))
    ml = cute.make_layout
    n2 = cute.size(R.acc_s) // 2
    n1 = cute.size(R.acc_c) // 2
    R.pv_s_src = cute.make_tensor(R.acc_s.iterator, ml((2, n2)))
    R.pv_s_dst = cute.make_tensor(R.frA_s.iterator, ml((2, n2)))
    R.pv_c_src = cute.make_tensor(R.acc_c.iterator, ml((2, n1)))
    R.pv_x_dst = cute.make_tensor(R.frA_x.iterator, ml((2, n1)))
    R.pv_g_dst = cute.make_tensor(R.frA_g.iterator, ml((2, n1)))
    # D0.11's o pack lands in the v^T fragment: v^T is read at the top of the
    # next chunk's `t` segment and is dead from there to the end of this one,
    # so the epilogue's bf16 copy of o^T costs no registers.
    R.pv_o_src = cute.make_tensor(R.acc_o.iterator, ml((2, n1)))
    R.pv_o_dst = cute.make_tensor(R.rV.iterator, ml((2, n1)))
    R.nk_d = cute.size(R.frA_s.shape[2])        # 8 k-steps over d_k = 128
    R.nk_c = cute.size(R.frA_x.shape[2])        # 4 k-steps over the 64 tokens
    R.ngrp = cute.size(R.acc_c) // 4
    # D1.2: the column index of each group of four C elements does not depend
    # on the chunk, so the 8 gate-record column indices are computed once here
    # and every gate load in the body is `base + slot + ncol[g]` off them.
    R.ncol = [R.cC[4 * g][1] for g in range(R.ngrp)]
    # ACCUMULATE is preset once; zero-init is `fill(0.0)`, which is the form
    # that traces inside a dynamic loop (wgmma_interleave gotchas 4/5).
    mma_ss.set(warpgroup.Field.ACCUMULATE, True)
    mma_st.set(warpgroup.Field.ACCUMULATE, True)
    return R


def _c3_chains(R, mma_ss, mma_st, sG, s, c=0):
    """One chunk of the state recurrence, D0.9's segment order.

    The gate scalars are per-column (per-token) factors. A thread's 32 C
    fragment elements sit on 2 rows and 16 columns in groups of four
    `(r0,n) (r0,n+1) (r1,n) (r1,n+1)`, so one group needs two scalar reads per
    vector, not four (hw_kb/cutedsl/sm90/ldmatrix_stmatrix section 4b)."""

    # ---- segment E: S^T -> bf16 A, then chain0 and chain1 back to back ------
    # D1.6 lever 1 (arm `vhoistE`): the `t` segment's v^T ldmatrix is issued
    # here, ahead of both chains, instead of after segment A's wait. v is stage
    # data covered by the full barrier at the top of the chunk body, so it never
    # depended on chain0; issuing it at the top of segment E puts the LDSM
    # latency under the widest window in the chunk. Measured in
    # notes/probe_d16_depth_stage.py (one job, six arms back to back, all
    # bitwise equal to base): graph median 284.6 us vs base 288.2 us (-1.25%),
    # chain3 solo 184.0 vs 188.1 us; the same load hoisted only to just before
    # segment A's wait (`vhoistA`) measured +1.47%.
    cute.copy(R.l2r, R.thr_l.partition_S(R.sVt[(None, None, s)]),
              R.thr_l.retile(R.rV))
    _pack_pairs(R.pv_s_src, R.pv_s_dst)
    R.acc_c.fill(0.0)
    R.acc_o.fill(0.0)
    warpgroup.fence()
    # D1.6 lever 1, second candidate (`split0`): chain0 issued as two commit
    # groups (k-steps 0-3, 4-7) measured +0.31% graph median, and combined with
    # the v^T hoist (`both`) -1.01% vs the hoist's -1.25% - so chain0 stays one
    # commit group and the wait-depth ladder stays LE1/LE1/LE0/LE1/LE0.
    for kb in range(R.nk_d):
        cute.gemm(mma_ss, R.acc_c, R.frA_s[(None, None, kb)],
                  R.frB_k[(None, None, kb, s)], R.acc_c)
    warpgroup.commit_group()
    _seg("c3.chain0.commit", c)
    warpgroup.fence()
    for kb in range(R.nk_d):
        cute.gemm(mma_ss, R.acc_o, R.frA_s[(None, None, kb)],
                  R.frB_q[(None, None, kb, s)], R.acc_o)
    warpgroup.commit_group()
    _seg("c3.chain1.commit", c)

    # ---- D1.2: segment A's gate scalars are issued here, ahead of the wait
    # that opens the segment, so the LDS latency lands inside chain0's flight
    # window and the values are in registers when the FMULs need them. beta_j
    # (row 0) for the 16 columns this thread owns, taken as 8 adjacent pairs
    # (LDS.64), plus Gamma (row 4) for D1.3's decay: 17 registers live across
    # the wait. beta_j.gamma_j (row 1) stays inside the segment - hoisting it
    # too would double that cost.
    bet = [(sG[(0, n, s)], sG[(0, n + 1, s)]) for n in R.ncol]
    gam = sG[(4, 0, s)]

    # ---- segment A: retire chain0 only; chain1 stays in flight over the
    # elementwise work. t = beta_j . v^T - (beta_j gamma_j) . (K S_0)^T.
    _seg("c3.chain0.wait.begin", c)
    warpgroup.wait_group(1)
    _seg("c3.chain0.wait.end", c)
    for g in range(R.ngrp):
        n0 = R.ncol[g]
        b0, b1 = bet[g]
        w0 = sG[(1, n0, s)]
        w1 = sG[(1, n0 + 1, s)]
        for sub in range(4):
            i = 4 * g + sub
            bb = b0 if sub % 2 == 0 else b1
            ww = w0 if sub % 2 == 0 else w1
            R.acc_c[i] = bb * F32(R.rV[i]) - ww * R.acc_c[i]
    # D1.3: the state decay S^T x Gamma (64 in-place FMUL) leaves segment C for
    # chain1's flight window. Legal: chain0 / chain1 read the bf16 copy packed
    # in segment E, not this fp32 accumulator, and chain2 - its only other
    # writer - is not issued until segment C.
    for i in range(cute.size(R.acc_s)):
        R.acc_s[i] = R.acc_s[i] * gam
    _pack_pairs(R.pv_c_src, R.pv_x_dst)

    # ---- segment B: chain0b over the freed accumulator, chain1 still in flight
    R.acc_c.fill(0.0)
    warpgroup.fence()
    for kb in range(R.nk_c):
        cute.gemm(mma_ss, R.acc_c, R.frA_x[(None, None, kb)],
                  R.frB_t[(None, None, kb, s)], R.acc_c)
    warpgroup.commit_group()
    _seg("c3.chain0b.commit", c)
    # D1.2: scale.gamma_j (row 3) issues before the wait that retires chain1.
    sgm = [(sG[(3, n, s)], sG[(3, n + 1, s)]) for n in R.ncol]
    # D1.3: o_inter is chain1's accumulator, so its scaling can only start once
    # chain1 has retired - one more wait here (chain0b stays in flight) buys a
    # window for the 32 FMUL that used to sit exposed at the top of segment C.
    _seg("c3.chain1.wait.begin", c)
    warpgroup.wait_group(1)
    _seg("c3.chain1.wait.end", c)
    for g in range(R.ngrp):
        q0, q1 = sgm[g]
        for sub in range(4):
            i = 4 * g + sub
            R.acc_o[i] = R.acc_o[i] * (q0 if sub % 2 == 0 else q1)
    # D1.2: segment C's Gamma/gamma_j (row 2) issues before segment C's wait,
    # after row 3's scalars have died - 16 registers live across that wait.
    gog = [(sG[(2, n, s)], sG[(2, n + 1, s)]) for n in R.ncol]

    # ---- segment C: drain chain0b, then the two Bu packs --------------------
    _seg("c3.chain0b.wait.begin", c)
    warpgroup.wait_group(0)
    _seg("c3.chain0b.wait.end", c)
    _pack_pairs(R.pv_c_src, R.pv_x_dst)         # chain1b's A: Bu^T, unscaled
    for g in range(R.ngrp):
        r0, r1 = gog[g]
        for sub in range(4):
            i = 4 * g + sub
            R.acc_c[i] = R.acc_c[i] * (r0 if sub % 2 == 0 else r1)
    _pack_pairs(R.pv_c_src, R.pv_g_dst)         # chain2's A: Bu^T . (Gamma/g)

    # D1.4: chain1b and chain2 no longer share a commit group. wait_group counts
    # groups, not GEMMs (hw_kb/cutedsl/sm90/wgmma_deferred_wait gotcha 1), so the only
    # way to retire o^T while the state update is still running is to give each
    # chain its own group, oldest first: chain1b, then chain2.
    warpgroup.fence()
    for kb in range(R.nk_c):
        cute.gemm(mma_ss, R.acc_o, R.frA_x[(None, None, kb)],
                  R.frB_p[(None, None, kb, s)], R.acc_o)
    warpgroup.commit_group()
    _seg("c3.chain1b.commit", c)
    warpgroup.fence()
    for kb in range(R.nk_c):
        cute.gemm(mma_st, R.acc_s, R.frA_g[(None, None, kb)],
                  R.frB_n[(None, None, kb, s)], R.acc_s)
    warpgroup.commit_group()
    _seg("c3.chain2.commit", c)

    # ---- segment D: retire chain1b only. o^T is final, so the epilogue that
    # follows this call runs inside chain2's flight window; the caller drains
    # chain2 afterwards, before the stage release (k box MN-major is chain2's
    # B operand, so the release may not move ahead of that drain).
    # The wait sits *after* chain2's commit, never before it: with only one
    # group in flight `wait_group(1)` is already satisfied and would not wait
    # at all (same entry, gotcha 3).
    _seg("c3.chain1b.wait.begin", c)
    warpgroup.wait_group(1)
    _seg("c3.chain1b.wait.end", c)


# ------------------------------------------------------- the epilogue (D0.11)
# o^T (64 v, 64 tok) fp32 -> bf16 pack -> STSM x4 into an 8 KiB staging slot ->
# proxy fence -> NamedBarrier 0x1 -> TMA store straight into o; the staging is
# double buffered and the reuse is throttled so that one store group stays in
# flight. final_state comes off the S^T accumulator by plain 64-bit STG: the CTA
# holds S^T[v half block, :] and `final_state[0, h, v, k] = S[k, v]` is exactly
# that, row major, so there is no transpose on that path.
#
# Orientation. o is contiguous in d_v (`o[tok, h, v]`) so the staging slot has
# to be v-contiguous, and the C fragment's M mode *is* v — the publish therefore
# reads the slot as an m-major C tile and takes the **transposed** stmatrix
# member. Per hw_kb/cutedsl/sm90/stsm_transpose_exchange the transposed member does not
# transpose the logical matrix (face[m, n] reads back the same either way); it
# only picks which mode is physically contiguous, and one LayoutEnum has to
# drive all four call-sites (member / value grouping / atom family / order) or
# the change is silently a no-op (that entry's gotcha 2).


def _c3_epi_layouts():
    """The o staging slot, twice, plus the (v, tok) view the stmatrix writes.

    The slot is allocated exactly like every other box — K_SW128 over (tok, v),
    v contiguous — so the TMA store atom takes the same single-stage layout the
    load side already uses. The publish view is its colex transposer: the staged
    domain index of (tok, v, buf) is tok + 64*v + 4096*buf, so (v, tok, buf)
    carries stride (64, 1, 4096), the same recipe as sVt_l. The two views are
    byte-identical to the MN_SW128 face the official epilogue would allocate
    (the two atom families share one swizzle and differ only by transposition,
    stsm_transpose_exchange section 'one LayoutEnum drives four places')."""
    al = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, CHUNK),
        BF16)
    sO_l = cute.tile_to_shape(al, (CHUNK, CHUNK, EPI_BUF), order=(0, 1, 2))
    sOt_l = cute.composition(sO_l, cute.make_layout(
        (CHUNK, CHUNK, EPI_BUF), stride=(CHUNK, 1, CHUNK * CHUNK)))
    return sO_l, sOt_l


def _c3_epi_r2s(mma_ss):
    """The r2s three-layer build (hw_kb/cutedsl/sm90/tma_store_epilogue section 3).

    COL_MAJOR = m-major C, because the publish view's M mode (v) is the
    contiguous one; the same enum feeds the store op (which emits the
    instruction) and the C atom (which groups the registers)."""
    E = LayoutEnum.COL_MAJOR
    return cute.make_tiled_copy_S(
        sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
        cute.make_tiled_copy_C_atom(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4),
                                BF16),
            mma_ss))


def _c3_epi_setup(R, r2s, tw, sOt):
    """Bound once outside the chunk loop, like the rest of D0.6's plan. The
    source is the already-live v^T fragment, re-viewed in copy-value order, so
    this path carries no staging fragment at all (tma_store_epilogue section 4:
    a whole-tile f32 + f16 staging pair is the shape to avoid when num_epi
    is 1).

    The source shape has to come from `partition_S`, not from `retile`. This
    atom's per-thread value count is 8 while the C fragment holds 32, and
    `retile` keeps the surplus inside mode 0 (`((8, 4), 1, 1)`) instead of
    handing it to the rest modes, so `cute.copy` sees rest 1 against the
    destination's rest 4 and refuses. The remedy is the one
    hw_kb/cutedsl/sm90/flat_tv_r2s_cfrag gotcha (a) gives for the same symptom: a
    linear reinterpret of the same registers under the partition's own shape
    (`make_tensor(<frag>.iterator, make_layout(shape))`). It is value-identical
    to the certified staging form - `retile` is a pure regroup of a compact
    fragment, so `retile(x)[v] == x[v]` and section 3's
    `tRS_rD[v] = tRS_rAcc[v]` gather is the identity we are folding away."""
    thr = r2s.get_slice(tw)
    R.epi_d = thr.partition_D(sOt)
    R.epi_s = cute.make_tensor(R.rV.iterator, cute.make_layout(
        cute.shape(thr.partition_S(sOt))[:3]))


@cute.jit
def _c3_epi_o(R, r2s, tma_o, bSG_s, bSG_g, col, c, b, w0):
    """One chunk's o publish. Fixed order, none of it commutes: pack, throttle,
    rendezvous, stmatrix, generic->async fence, rendezvous, store.

    D1.4: the whole body runs inside chain2's flight window. The operand set is
    disjoint from every in-flight accumulator — it reads o^T (retired by the
    `wait_group(1)` that ends the chains helper) and writes the v^T fragment and
    the staging slot, never S^T or chain2's A fragment
    (hw_kb/cutedsl/sm90/wgmma_deferred_wait gotcha 2). The two NamedBarriers and the
    proxy fence do not drain the wgmma pipeline, which is the point: they are
    what used to sit exposed after the full drain.

    The throttle sits against the overwrite it protects (the stmatrix below),
    not against the store it drains (tma_store_epilogue gotcha 13), and the
    store is issued by a whole warp because the copy atom elects internally
    (gotcha 6).

    `@cute.jit` because `w0` is a runtime Boolean: only a decorated function
    gets its AST rewritten, and a plain `if` on a dynamic value in an
    undecorated helper raises PHASE_DYNAMIC_TO_STATIC_BOOL. Under an open
    insertion point the decorator does not emit a call - it inlines the
    preprocessed body - so the holder object still crosses the boundary."""
    _pack_pairs(R.pv_o_src, R.pv_o_dst)
    _seg("c3.store.throttle.begin", c)
    if w0:
        cute.arch.cp_async_bulk_wait_group(EPI_BUF - 1, read=True)
    cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
    _seg("c3.store.throttle.end", c)
    cute.copy(r2s, R.epi_s, R.epi_d[(None, None, None, b)])
    cute.arch.fence_view_async_shared()
    _seg("c3.store.bar.begin", c)
    cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=128)
    _seg("c3.store.bar.end", c)
    if w0:
        cute.copy(tma_o, bSG_s[(None, b)], bSG_g[(None, c, col)])
        cute.arch.cp_async_bulk_commit_group()
    _seg("c3.store.issue", c)


def _c3_epi_state(R, mma_st, tw, mFS, row):
    """The last chunk's S^T accumulator, straight to final_state[0, h, v, k].

    A C-fragment group is `(r0,n) (r0,n+1) (r1,n) (r1,n+1)`, so element pairs
    sit on adjacent k and are adjacent in the destination too: autovec_copy
    picks that up as 64-bit stores. The row offset is a multiple of 512 B but
    a dynamic offset drops the static alignment attribute, so restore it."""
    gS = cute.make_tensor((mFS.iterator + row * DIM).align(16),
                          cute.make_layout((CHUNK, DIM), stride=(DIM, 1)))
    cute.autovec_copy(R.acc_s, mma_st.get_slice(tw).partition_C(gS))


# ---------------------------------------------------------------- chain3 kernel
# Part 4 of the M0 group closes the data flow: the consumer's two accumulators
# now reach o and final_state through D0.11's epilogue.


@cute.kernel
def _k_chain3(tma_q: cute.CopyAtom, gQc: cute.Tensor,
              tma_k: cute.CopyAtom, gKc: cute.Tensor,
              tma_v: cute.CopyAtom, gVc: cute.Tensor,
              tma_t: cute.CopyAtom, gTc: cute.Tensor,
              tma_p: cute.CopyAtom, gPc: cute.Tensor,
              tma_o: cute.CopyAtom, gOc: cute.Tensor, mFS: cute.Tensor,
              mG2: cute.Tensor, flag_base: cutlass.Int64,
              sQ_l: cute.ComposedLayout, sQb_l: cute.ComposedLayout,
              sV_l: cute.ComposedLayout, sKmn_l: cute.ComposedLayout,
              sVt_l: cute.ComposedLayout, sG_l: cute.Layout,
              sO_l: cute.ComposedLayout, sOt_l: cute.ComposedLayout,
              mma_ss, mma_st, l2r, r2s):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    wid = tidx // 32
    rank = cute.arch.block_idx_in_cluster()
    _seg("c3.entry")

    # D0.4 map: ctaid & 1 = d_v half, ctaid >> 1 = v-head, qk-head = v-head // 2.
    # The cluster of 4 is therefore {2 v-heads sharing a qk-head} x {2 halves}:
    # ranks 0/1 share one head, ranks 2/3 the other, all four share qh.
    head = bidx // 2
    half = bidx % 2
    qh = bidx // 4

    st = cutlass.utils.SmemAllocator().allocate(C3Smem)
    sQ = st.q.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
    sQb = st.q.get_tensor(sQb_l.outer, swizzle=sQb_l.inner)
    sK = st.k.get_tensor(sQ_l.outer, swizzle=sQ_l.inner)
    sKb = st.k.get_tensor(sQb_l.outer, swizzle=sQb_l.inner)
    sKmn = st.k.get_tensor(sKmn_l.outer, swizzle=sKmn_l.inner)
    sV = st.v.get_tensor(sV_l.outer, swizzle=sV_l.inner)
    sVt = st.v.get_tensor(sVt_l.outer, swizzle=sVt_l.inner)
    sT = st.t.get_tensor(sV_l.outer, swizzle=sV_l.inner)
    sP = st.p.get_tensor(sV_l.outer, swizzle=sV_l.inner)
    sG = st.g.get_tensor(sG_l)
    sO = st.ost.get_tensor(sO_l.outer, swizzle=sO_l.inner)
    sOt = st.ost.get_tensor(sOt_l.outer, swizzle=sOt_l.inner)
    sGbase = st.g.data_ptr()
    mbar = st.mbar.data_ptr()

    # ---- D0.8 prologue: mbarrier init, then the cluster handshake -----------
    # empty[s] gathers at whoever supplied that stage, deduplicated. rank 0
    # carries q/k for the whole cluster, so all four arrive on it (4 CTAs x 4
    # warps = 16); rank 2 carries T/P for its pair, so ranks 2/3 arrive on it
    # (8); every CTA fetches its own v and gate, which leaves only ranks 1/3
    # needing an arrive on themselves (4 each) — ranks 0/2 are covered by the
    # arrives they already receive. 16 / 8 / 4 / 4 for ranks 0 / 2 / 1 / 3 is
    # the milestone's reading (a).
    ecnt = 4 * _isel_eq(rank, 0, 4, _isel_eq(rank, 2, 2, 1))
    if wid == 4:
        with cute.arch.elect_one():
            # D2.1 lever 1 (chain3 prologue): pull the six tensormaps into the
            # descriptor cache while the prologue rendezvous and the first
            # flag spin are still running, so the first TMA of chunk 0 does
            # not pay the cold descriptor fetch on top of the p1 wait.
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_t)
            cpasync.prefetch_descriptor(tma_p)
            cpasync.prefetch_descriptor(tma_o)
            for s in cutlass.range_constexpr(NSTAGE):
                cute.arch.mbarrier_init(mbar + s, 1)
                cute.arch.mbarrier_init(mbar + NSTAGE + s, ecnt)
    cute.arch.mbarrier_init_fence()
    _seg("c3.prologue.bar.begin")
    cute.arch.sync_threads()
    _seg("c3.prologue.bar.end")
    _seg("c3.prologue.cluster.begin")
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
    _seg("c3.prologue.cluster.end")

    # ---- gmem tiling. Every tensor is read as rank 2 with the head folded
    # into the column mode, so every TMA box is a plain (64, 64) tile and the
    # column-tile index carries the head: 2*qh + box for q/k, 2*head + half
    # for v, and T/P are indexed by the record alone.
    gQ = cute.group_modes(cute.local_tile(gQc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    gK = cute.group_modes(cute.local_tile(gKc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    gV = cute.group_modes(cute.local_tile(gVc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    gT = cute.group_modes(cute.local_tile(gTc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    gP = cute.group_modes(cute.local_tile(gPc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    tQs, tQg = cpasync.tma_partition(tma_q, 0, cute.make_layout(1),
                                     cute.group_modes(sQb, 0, 2), gQ)
    tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1),
                                     cute.group_modes(sKb, 0, 2), gK)
    tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1),
                                     cute.group_modes(sV, 0, 2), gV)
    tTs, tTg = cpasync.tma_partition(tma_t, 0, cute.make_layout(1),
                                     cute.group_modes(sT, 0, 2), gT)
    tPs, tPg = cpasync.tma_partition(tma_p, 0, cute.make_layout(1),
                                     cute.group_modes(sP, 0, 2), gP)
    # the o store: the column tile carries (v-head, d_v half) the same way
    gO = cute.group_modes(cute.local_tile(gOc, (CHUNK, CHUNK), (None, None)),
                          0, 2)
    tOs, tOg = cpasync.tma_partition(tma_o, 0, cute.make_layout(1),
                                     cute.group_modes(sO, 0, 2), gO)

    # D0.6's register plan, bound once outside every branch so the fragments
    # dominate both the warp-role branch and the chunk loop.
    R = _c3_setup(mma_ss, mma_st, tidx % 128, sQ, sK, sKmn, sVt, sT, sP, l2r)
    _c3_epi_setup(R, r2s, tidx % 128, sOt)
    R.acc_s.fill(0.0)                   # the chunk-0 entry state is zero

    # ================================================ producer warp (D0.10)
    if wid == 4:
        for c in cutlass.range(NCHUNK):
            s = c % NSTAGE
            rnd = c // NSTAGE
            # Round 0 waits parity 1 on a fresh barrier, which is satisfied at
            # once (the producer-phase-1 convention, written out so that no
            # phase bit is loop-carried).
            _seg("c3.empty.wait.begin", c)
            _bar_wait(mbar + NSTAGE + s, 1 - rnd % 2)
            _seg("c3.empty.wait.end", c)

            rec = c * HV + head
            addr = flag_base + rec.to(cutlass.Int64) * 4
            _seg("c3.flagspin.begin", c)
            sp = I32(0)
            fv = _ld_acquire_u32(addr)
            # the give-up term rides on the loaded value so the loop condition
            # stays a single comparison
            while fv == 0:
                _nanosleep(NANOS)
                sp = sp + 1
                fv = _ld_acquire_u32(addr) + sp // SPIN_LIMIT
            _seg("c3.flagspin.end", c)

            # the async proxy has to see p1's release before the TMA reads it
            cute.arch.fence_proxy("async.shared", space="cta")
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + s, TX_BYTES)

            # D0.8 reading (a): rank 0 carries q/k for the whole cluster, the
            # even rank carries T/P for its pair, every CTA fetches its own v
            # half block and its own gate record. 7 boxes + 1280 B land in
            # every CTA, which is what expect_tx was told.
            if rank == 0:
                for j in cutlass.range_constexpr(2):
                    cute.copy(tma_q, tQg[(None, c, 2 * qh + j)],
                              tQs[(None, 2 * s + j)], tma_bar_ptr=mbar + s,
                              mcast_mask=cutlass.Int16(0xF))
                    cute.copy(tma_k, tKg[(None, c, 2 * qh + j)],
                              tKs[(None, 2 * s + j)], tma_bar_ptr=mbar + s,
                              mcast_mask=cutlass.Int16(0xF))
            if rank % 2 == 0:
                pm = cutlass.Int16(_isel_eq(rank, 0, 3, 12))
                cute.copy(tma_t, tTg[(None, rec, 0)], tTs[(None, s)],
                          tma_bar_ptr=mbar + s, mcast_mask=pm)
                cute.copy(tma_p, tPg[(None, rec, 0)], tPs[(None, s)],
                          tma_bar_ptr=mbar + s, mcast_mask=pm)
            cute.copy(tma_v, tVg[(None, c, 2 * head + half)], tVs[(None, s)],
                      tma_bar_ptr=mbar + s)

            # the gate record: one 1280 B 1D bulk copy. Both addresses are
            # 16 B-aligned (rec * 1280 and s * 2048), but a dynamic offset
            # drops the static alignment attribute, so restore it explicitly.
            gaddr = (mG2.iterator + rec * GATE_W).align(16).toint()
            saddr = (sGbase + s * GATE_PAD).align(16).toint()
            baddr = (mbar + s).toint()
            with cute.arch.elect_one():
                _bulk_g2s(saddr, gaddr, GATE_W * 4, baddr)
            _seg("c3.supply.issue", c)

    # ============================================== consumer warpgroup (D0.9)
    if wid < 4:
        col = 2 * head + half            # the o column tile this CTA owns
        w0 = wid == 0                    # the store-issuing warp
        for c in cutlass.range(NCHUNK):
            s = c % NSTAGE
            rnd = c // NSTAGE
            _seg("c3.full.wait.begin", c)
            _bar_wait(mbar + s, rnd % 2)
            _seg("c3.full.wait.end", c)

            _c3_chains(R, mma_ss, mma_st, sG, s, c)
            # D1.4: the epilogue runs with chain2 in flight (the chains helper
            # returns having retired chain1b only), and the drain of chain2 is
            # the loop body's last blocking point before the stage release. It
            # also keeps every group off the back edge
            # (hw_kb/cutedsl/sm90/wgmma_backedge_pending_group path 1).
            _c3_epi_o(R, r2s, tma_o, tOs, tOg, col, c, c % EPI_BUF, w0)
            _seg("c3.chain2.wait.begin", c)
            warpgroup.wait_group(0)
            _seg("c3.chain2.wait.end", c)

            eb = mbar + NSTAGE + s
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(eb, peer_cta_rank_in_cluster=0)
                if rank >= 2:
                    cute.arch.mbarrier_arrive(eb, peer_cta_rank_in_cluster=2)
                if rank % 2 == 1:
                    cute.arch.mbarrier_arrive(eb,
                                              peer_cta_rank_in_cluster=rank)
            _seg("c3.release", c)

        # the state after token N-1 is what is in the accumulator once the loop
        # has run out, so final_state leaves the body rather than branching
        # inside it; then drain the stores still reading the staging slots.
        _c3_epi_state(R, mma_st, tidx % 128, mFS, head * DIM + half * CHUNK)
        _seg("c3.store.drain.begin")
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        _seg("c3.store.drain.end")

    # D0.3 exit reset: both CTAs of a pair clear the same 128 flags, which is
    # idempotent, but only after the cluster barrier — otherwise one CTA can
    # clear a word its peer has not read yet, and no CTA may retire while a
    # peer can still arrive into its mbarriers.
    _seg("c3.exit.cluster.begin")
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    _seg("c3.exit.cluster.end")
    if tidx < NCHUNK:
        _st_relaxed_u32(flag_base + cutlass.Int64(tidx * HV + head) * 4, 0)


@cute.jit
def _h_chain3(tq: cute.Tensor, tk: cute.Tensor, tv: cute.Tensor,
              tT: cute.Tensor, tP: cute.Tensor, tG: cute.Tensor,
              to: cute.Tensor, tfs: cute.Tensor,
              flag_base: cutlass.Int64, stream):
    # rank-2 views: the head lives in the column mode, so every box is a plain
    # (64, 64) tile of a 2D tensor and there are no singleton modes anywhere.
    mQ2 = cute.make_tensor(tq.iterator, cute.make_layout(
        (NTOK, HQK * DIM), stride=(HQK * DIM, 1)))
    mK2 = cute.make_tensor(tk.iterator, cute.make_layout(
        (NTOK, HQK * DIM), stride=(HQK * DIM, 1)))
    mV2 = cute.make_tensor(tv.iterator, cute.make_layout(
        (NTOK, HV * DIM), stride=(HV * DIM, 1)))
    mT2 = cute.make_tensor(tT.iterator, cute.make_layout(
        (NREC * CHUNK, CHUNK), stride=(CHUNK, 1)))
    mP2 = cute.make_tensor(tP.iterator, cute.make_layout(
        (NREC * CHUNK, CHUNK), stride=(CHUNK, 1)))
    mG2 = cute.make_tensor(tG.iterator, cute.make_layout(
        (NREC, GATE_W), stride=(GATE_W, 1)))
    # o keeps the same column-folded reading as v, so the o box is the same
    # (64 tok, 64 d_v) tile and its column tile is 2*v_head + half. final_state
    # folds (head, v) into the row: row = head*128 + v, k contiguous.
    mO2 = cute.make_tensor(to.iterator, cute.make_layout(
        (NTOK, HV * DIM), stride=(HV * DIM, 1)))
    mFS = cute.make_tensor(tfs.iterator, cute.make_layout(
        (HV * DIM, DIM), stride=(DIM, 1)))

    sQ_l, sQb_l, sV_l, sKmn_l, sVt_l, sG_l, box_l = _c3_layouts()
    sO_l, sOt_l = _c3_epi_layouts()

    # D0.6: every chain's A is an RMEM fragment, so a_source = RMEM and A is
    # k-major by construction (the ISA only lets B be transposed in that form).
    # Two tilers cover the five chains: (64, 64) with a k-major B for chain0 /
    # chain1 / chain0b / chain1b, and (64, 128) with an MN-major B for chain2,
    # whose reduction runs along the token mode of the k box.
    mj = cute.nvgpu.OperandMajorMode
    mma_ss = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, mj.K, mj.K, F32, atom_layout_mnk=(1, 1, 1),
        tiler_mn=(CHUNK, CHUNK), a_source=warpgroup.OperandSource.RMEM)
    mma_st = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, mj.K, mj.MN, F32, atom_layout_mnk=(1, 1, 1),
        tiler_mn=(CHUNK, DIM), a_source=warpgroup.OperandSource.RMEM)
    # v^T into the C-fragment order of chain0's accumulator. The v box is
    # allocated with the v mode contiguous and the fragment's M mode is v, so
    # this is the transposed member (hw_kb/cutedsl/sm90/ldmatrix_stmatrix, the C-read
    # rows of the member table); the tiler is the whole 64x64 C tile, hence
    # make_tiled_copy_C and four matrices.
    l2r = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), BF16), mma_ss)
    # D0.11's publish side, over the same C-fragment tiling as chain1b
    r2s = _c3_epi_r2s(mma_ss)

    # The op class and the mask have to change together
    # (hw_kb/cutedsl/sm90/tma_multicast_cluster gotcha 1); num_multicast stays 1
    # because the issuing CTA sends a whole box, not a 1/N slice — the mask,
    # not any box splitting, is what makes this a multicast.
    op_mc = cpasync.CopyBulkTensorTileG2SMulticastOp()
    tma_q, gQc = cpasync.make_tiled_tma_atom(op_mc, mQ2, box_l,
                                             (CHUNK, CHUNK), num_multicast=1)
    tma_k, gKc = cpasync.make_tiled_tma_atom(op_mc, mK2, box_l,
                                             (CHUNK, CHUNK), num_multicast=1)
    tma_t, gTc = cpasync.make_tiled_tma_atom(op_mc, mT2, box_l,
                                             (CHUNK, CHUNK), num_multicast=1)
    tma_p, gPc = cpasync.make_tiled_tma_atom(op_mc, mP2, box_l,
                                             (CHUNK, CHUNK), num_multicast=1)
    # v is this CTA's own half block: never multicast
    tma_v, gVc = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mV2, box_l, (CHUNK, CHUNK),
        num_multicast=1)
    # the store atom takes one stage of the staging layout, never the staged one
    # (hw_kb/cutedsl/sm90/tma_store_epilogue gotcha 1), and S2G has no multicast arg.
    tma_o, gOc = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mO2,
        cute.slice_(sO_l, (None, None, 0)), (CHUNK, CHUNK))

    _k_chain3(tma_q, gQc, tma_k, gKc, tma_v, gVc, tma_t, gTc, tma_p, gPc,
              tma_o, gOc, mFS, mG2, flag_base,
              sQ_l, sQb_l, sV_l, sKmn_l, sVt_l, sG_l, sO_l, sOt_l,
              mma_ss, mma_st, l2r, r2s).launch(
        grid=[C3_CTAS, 1, 1], block=[C3_THREADS, 1, 1],
        cluster=[C3_CLUSTER, 1, 1], stream=stream)


# ----------------------------------------------------------------- host driver

_LOCK = threading.Lock()
_WS = None
_COMPILED = None
_CACHE = {}
_ORDER = []
_CACHE_MAX = 6


def _cu(stream):
    import cuda.bindings.driver as cuda_drv
    return cuda_drv.CUstream(stream.cuda_stream)


def _workspace():
    """Persistent cross-kernel workspace: T, P, the gate records and the flag
    array. Allocated once and zeroed once; chain3 resets the flags on exit, so
    there is no fill node in the graph."""
    global _WS
    if _WS is None:
        dev = torch.device("cuda")
        wT = torch.empty((NREC, CHUNK, CHUNK), device=dev, dtype=torch.bfloat16)
        wP = torch.empty((NREC, CHUNK, CHUNK), device=dev, dtype=torch.bfloat16)
        wG = torch.empty((NREC, GATE_ROWS, CHUNK), device=dev,
                         dtype=torch.float32)
        wF = torch.zeros((NREC,), device=dev, dtype=torch.int32)
        _WS = {
            "T": wT, "P": wP, "G": wG, "F": wF,
            "tT": from_dlpack(wT, assumed_align=16),
            "tP": from_dlpack(wP, assumed_align=16),
            "tG": from_dlpack(wG, assumed_align=16),
            "fbase": cutlass.Int64(wF.data_ptr()),
        }
    return _WS


def _compiled(ws, tq, tk, tv, ta, tb, to, tfs, sc, stream):
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = (
            cute.compile(_h_p1, tq, tk, ta, tb, ws["tT"], ws["tP"], ws["tG"],
                         ws["fbase"], sc, stream),
            cute.compile(_h_chain3, tq, tk, tv, ws["tT"], ws["tP"], ws["tG"],
                         to, tfs, ws["fbase"], stream),
        )
    return _COMPILED


class _Entry:
    __slots__ = ("graph", "o", "state", "hold")


def _build(q, k, v, a, beta, scale):
    ws = _workspace()
    tq = from_dlpack(q, assumed_align=16)
    tk = from_dlpack(k, assumed_align=16)
    tv = from_dlpack(v, assumed_align=16)
    ta = from_dlpack(a, assumed_align=16)
    tb = from_dlpack(beta, assumed_align=16)

    sc = cutlass.Float32(float(scale))
    cur = torch.cuda.current_stream()

    ent = _Entry()
    ent.o = torch.zeros((NTOK, HV, DIM), device=q.device, dtype=torch.bfloat16)
    ent.state = torch.zeros((1, HV, DIM, DIM), device=q.device,
                            dtype=torch.float32)
    to = from_dlpack(ent.o, assumed_align=16)
    tfs = from_dlpack(ent.state, assumed_align=16)
    c_p1, c_c3 = _compiled(ws, tq, tk, tv, ta, tb, to, tfs, sc, _cu(cur))

    # Hold the inputs so their allocations cannot be recycled under the graph's
    # captured pointers.
    ent.hold = (q, k, v, a, beta, tq, tk, tv, ta, tb, to, tfs)

    side = torch.cuda.Stream()

    def _issue_p1(cu):
        c_p1(tq, tk, ta, tb, ws["tT"], ws["tP"], ws["tG"], ws["fbase"], sc,
             cu)

    def _issue_c3(cu):
        c_c3(tq, tk, tv, ws["tT"], ws["tP"], ws["tG"], to, tfs, ws["fbase"],
             cu)

    # The warm-up pass puts both kernels on one stream, so it has to keep the
    # producer first: chain3 alone on a stream would spin on flags nobody sets.
    warm = torch.cuda.Stream()
    warm.wait_stream(cur)
    with torch.cuda.stream(warm):
        _issue_p1(_cu(warm))
        _issue_c3(_cu(warm))
    cur.wait_stream(warm)
    torch.cuda.synchronize()

    # D2.3: chain3 is launched on the capture stream and p1 on the forked side
    # stream, so chain3's kernel node is created first and the CTA scheduler
    # places the 16 clusters (1 CTA/SM, 194608 B smem) before p1's 2048-CTA
    # flood takes the SMs. The two nodes stay concurrent — both fork off the
    # same event and rejoin at the end, neither waits on the other — and p1
    # reads nothing chain3 produces, so the order cannot deadlock.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cap = torch.cuda.current_stream()
        ev = torch.cuda.Event()
        ev.record(cap)
        side.wait_event(ev)
        _issue_c3(_cu(cap))
        _issue_p1(_cu(side))
        ev2 = torch.cuda.Event()
        ev2.record(side)
        cap.wait_event(ev2)
    ent.graph = g
    return ent


def gdn_prefill_fwd(q, k, v, a, beta, scale):
    key = (q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(),
           beta.data_ptr(), float(scale))
    with _LOCK:
        ent = _CACHE.get(key)
        if ent is None:
            ent = _build(q, k, v, a, beta, scale)
            _CACHE[key] = ent
            _ORDER.append(key)
            while len(_ORDER) > _CACHE_MAX:
                _CACHE.pop(_ORDER.pop(0), None)
    ent.graph.replay()
    return ent.o, ent.state
