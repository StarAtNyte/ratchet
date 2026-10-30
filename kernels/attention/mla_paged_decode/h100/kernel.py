"""mla_paged_decode — the student's kernel file (the only file the student edits).

Multi-head Latent Attention decode forward over a paged compressed-KV cache
(page size 1, i.e. one token per page), the FlashInfer-Bench definition
mla_paged_decode_h16_ckv512_kpe64_ps1 (DeepSeek-V3/R1 decode after weight
absorption, tensor parallel 8): 16 query heads per rank, a 512-wide latent
("nope") component plus a 64-wide rotary ("pe") component, bf16 in / bf16 out,
one query token per sequence. The campaign point is one real FlashInfer-Bench
workload (64 sequences, 75145 KV tokens in total, page indices taken verbatim
from the trace). The signature below is the card's frozen contract; the student
implements the body in CuTe DSL; the signature itself may not change.

Semantics (per sequence i, per query head h; the KV rows of sequence i are the
pages kv_indices[kv_indptr[i] : kv_indptr[i+1]], in that order; latent row
Kc[j] = ckv_cache[page_j, 0, :] (512), rotary row Kp[j] = kpe_cache[page_j, 0, :]
(64); the SAME latent row is the value; fp32 chain):

    s[j]   = sm_scale * ( <q_nope[i, h, :], Kc[j]> + <q_pe[i, h, :], Kp[j]> )
    m      = max_j s[j];  p[j] = exp(s[j] - m);  l = sum_j p[j]
    out[i, h, :] = (sum_j p[j] * Kc[j]) / l           (bf16 on output, 512 wide)
    lse[i, h]    = log2(sum_j exp(s[j])) = (m + ln(l)) / ln(2)   (fp32, base-2)

All 16 heads of a sequence share the same KV rows (MLA has ONE latent KV head).
Pages a sequence does not list must never be read (the correctness cases fill
every unreferenced page with NaN). Sequence lengths are arbitrary (1, 2, 63, 64,
65, ..., 9197, ...), page indices are arbitrary (the trace has shared-prefix
pages, stride-16 runs and long contiguous runs — assume nothing), several
sequences may list the same page, and the page pool is large (the campaign pool
holds 989669 pages, ~1.1 GB of cache).

--------------------------------------------------------------------------
Structure (milestone M0 — architecture, under construction):

  D0.1  two kernels inside the window: `step` (one CTA = one KV chunk of one
        sequence x all 16 heads -> partial) and `combine` (one CTA = one (b, h)
        -> out/lse).  No zero-fill kernel: out / partials are torch.empty and
        are fully overwritten.
  D0.2  inter-CTA split = equal-token chunking + device-side self-scheduling.
        Host uses SHAPES ONLY (T = kv_indices.numel()); the device recomputes
        chunk_indptr from kv_indptr with a block scan and binary/linear search
        for its own (sequence, chunk).  No device->host sync in the window.
  D0.3  intermediate ABI: base-2 log domain (scale2 = sm_scale * log2 e folded
        on the host), partial_out fp32 and ALREADY NORMALISED per chunk,
        partial_lse fp32 base-2, nobody zero-fills.
  D0.4  intermediate-value ledger: partial_out / partial_lse cross the two
        kernels through a global workspace, same stream, no event.
  D0.5  bM = 16 = all 16 heads (MLA has one latent KV head, so the head mode IS
        the m16n8k16 M mode), bN = 32 KV rows per tile.  Phase S splits the 576
        depth over warps (C1: w0-w3 take 128 latent each, w4 the 64 rotary,
        w5-w7 idle -> five partial-sum contributors), phase O splits the 512
        output dims over all 8 warps (64 each).  Resident in registers for the
        whole KV loop: the per-warp Q depth slice (A fragments) and the O
        accumulator; the KV ring, the S partial sums and P go through smem.
  D0.6  smem: ring 74752 B (2 stages x 32 rows x 576 bf16 on a 584-element row
        stride) + qs 18688 + ss 12800 + p 1280 + stat 192 + idx 256 + meta 1284
        = 109252 B.
        Write side 16 B/thread cp.async.cg (LDGSTS.E.BYPASS.128); read side is
        the matrix atom throughout -- LDSM x4 non-transposed for Q and P, LDSM
        x2 non-transposed for K as B, LDSM x2 TRANSPOSED for V as B (its
        contiguous mode is the output dim = the N mode).  The S partial sums go
        out on the C-frag bijection (64-bit STS) and come back as 128-bit LDS.
  D0.7  step block = 256 = 8 warps, convoy, no producer/consumer specialisation.
  D0.8  register budget: O accumulator 32 words + Q A-fragments 32 words resident;
        the S accumulator, the K/V/P fragments and the merge segment's containers
        are fixed containers allocated once outside the KV loop.  The 128
        registers/thread ceiling is an ENTRY-POINT property and is asked for as
        one: block 256 + min_blocks_per_mp 2.
  D0.9  issue order: body i issues tile i+1, waits on the tile the PREVIOUS body
        issued (tile i), then consumes tile i; 1-tile prologue warm-up; wait
        depth N = 1 (the D1.1 starting point).
  D0.10 five blocking rendezvous per loop body: #1 after the wait, #2 after the
        S partial sums are published, #3 after the running max is published,
        #4 after P and l are published, #5 after the consume.
  D0.11 indirect supply = prefetched: the page ids of tile j+2 are fetched by
        body j with a `cp.async` of their own (issued after rendezvous #1, in a
        commit group of its own) into a 2-stage smem index ring; body j+1's wait
        retires them and body j+2 issues its KV cp.async off them.  The id chain
        never touches a register and the loop body issues no index LDG at all,
        so no LDGSTS ever waits on its own row's index.
  D0.12 supply side: rows past the chunk end are zero-filled in place, never
        issued and never clamped onto another row (an unlisted page must not be
        touched, a listed one must not be counted twice).  Consume side: the
        invalid columns are forced to -inf in registers before the exp2.
  D0.13 combine: grid (B, H), block 128, 4 output dims per thread.

Construction steps 1-4 of the M0 sequence deliver the spine, the KV supply ring,
the two MMA phases with the cross-warp merge and the base-2 softmax, and the
index prefetch ring -- the last high-risk runtime mechanism before the gate.
Each step gets its own whole-machine smoke.
"""

from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op


# ----------------------------------------------------------------- constants

H = 16
D_CKV = 512
D_KPE = 64
LOG2E = 1.4426950408889634          # log2(e) -- folded into sm_scale host-side (D0.3)

NCTA_TARGET = 264                   # 132 SM x 2 CTA/SM (D0.2)
STEP_THREADS = 256                  # D0.7: block = 256 = 8 warps, convoy, no specialisation
# D0.8: 128 registers/thread = 65536 / (2 CTA/SM x 256 threads).  That ceiling is
# a property of the ENTRY POINT, not of how tidily the body is written: ptxas
# holds a kernel to it only when both launch-bound attributes are on the entry,
# and `nvvm.minctasm` is emitted ONLY when `min_blocks_per_mp > 0` (`nvvm.reqntid`
# comes free from `launch(block=...)`).  Leave it at the default 0 and the
# allocator takes whatever it likes -- the 2 CTA/SM the smem ledger (D0.6, 109252
# B of the 114688 B two-CTA allowance) was sized around then silently never
# materialises, with no error anywhere.  There is no `setmaxnreg` in this kernel
# (D0.7 has no warp specialisation), so a plain entry-point bound is the whole
# mechanism and cannot fight a per-role budget.
MIN_CTA_PER_SM = 2
COMBINE_THREADS = 128               # D0.13
B_MAX = 96                          # widest batch the chunk_indptr block scan covers
SCAN_PAD = 64                       # zero pad in front of the scan array (branch-free reads)
SCAN_STEPS = 7                      # Hillis-Steele offsets 1,2,...,64 -> covers 128 entries
STEP_SCAN_N = SCAN_PAD + STEP_THREADS + 1
COMB_SCAN_N = SCAN_PAD + COMBINE_THREADS + 1

# --- D0.5 / D0.6: the KV supply ring -----------------------------------------
# bN = 32 KV rows per tile; one ring row holds the latent 512 and the rotary 64
# of the SAME page id side by side, so the S phase sees one 576-deep K row.
BN = 32
RING_STAGES = 2                     # D0.9 needs both stages live (one filling, one read)
ROW_W = D_CKV + D_KPE               # 576 bf16 of payload per ring row
# D2.1 lever, M0 value = explicit row pad (the lever's third candidate) rather
# than the 128 B XOR swizzle: 8 bf16 of pad puts the row stride at 584 elements
# = 292 fp32 words = 4 (mod 32), so the eight row addresses an `ldmatrix` issues
# land on banks 4r (r = 0..7) -- every one of the 32 banks exactly once, on the
# cp.async write side and on BOTH LDSM members of the read side.  A plain
# (non-composed) layout also keeps `flat_divide` / `select` on the subtile views
# inside the certified envelope.  The frozen thing in D0.6 is the atom, and the
# atom is unchanged.
ROW_PAD = 8
ROW_S = ROW_W + ROW_PAD             # 584 bf16 = 1168 B physical row stride
RING_BYTES = RING_STAGES * BN * ROW_S * 2        # 74752 B

# --- D0.5: the two MMA phases -------------------------------------------------
# Phase S (S = Q.K^T): the 576 depth is split over warps -- w0..w3 take 128 of
# the latent depth each, w4 takes the 64 rotary, w5..w7 idle (candidate C1: five
# partial-sum contributors, the minimum of the live fork).  Each contributor
# runs 4 zero-init n8 accumulator chains (4 x n8 = bN 32).
# Phase O (O += P.V): the 512 output dims are split over warps -- 64 dims each,
# 8 chains of n8, K = 32 = bN.  O stays in registers across the whole KV loop.
MMA_M, MMA_N, MMA_K = 16, 8, 16     # warp-level mma.sync atom (bM = 16 = all heads)
WARPS = STEP_THREADS // 32          # 8
DEPTH_CHUNK = 64                    # one A/B slice = 64 deep (9 of them per row)
N_DEPTH_CHUNKS = ROW_W // DEPTH_CHUNK           # 9
S_CONTRIB = 5                       # w0..w4 (C1)
O_DIMS_PER_WARP = D_CKV // WARPS    # 64
N_CHAIN_S = BN // MMA_N             # 4 n8 chains in the S phase
KB_O = BN // MMA_K                  # 2 k-blocks in the O phase

# --- D0.6: the rest of the smem ledger ---------------------------------------
# qs   Q staging, 16 heads x 576 bf16 on the ring's row stride (same pad, so the
#      same LDSM bank argument holds).
# ss   the S partial-sum region: S_CONTRIB slots x 16 heads x 40 fp32.  40 = 32
#      + 8 pad; a 40-word row stride is 8 (mod 32), which is exactly what the
#      C-frag's 64-bit stores need (lane l writes rows l/4, l/4+8 at columns
#      2*(l%4)) to cover 32 distinct banks per phase.  This is the reference's
#      own 16x40 shape.
# p    P (probabilities) 16 x 40 bf16; a 40-element row stride is 20 (mod 32)
#      words, so the 8 row addresses of the O-phase A-side LDSM hit
#      20r mod 32 = 0,20,8,28,16,4,24,12 -- all 32 banks.
# stat cross-warp softmax statistics: running m, running l, and the rescale
#      factor 2^(m_old - m_new), 16 fp32 each.
QS_ROWS = H
QS_BYTES = QS_ROWS * ROW_S * 2                   # 18688 B
S_ROW = BN + 8                                   # 40 fp32 per head row
SS_BYTES = S_CONTRIB * H * S_ROW * 4             # 12800 B
P_ROW = BN + 8                                   # 40 bf16 per head row
P_BYTES = H * P_ROW * 2                          # 1280 B
STAT_BYTES = 3 * H * 4                           # 192 B
NEG_BIG = -3.0e38                                # the -inf stand-in (D0.12)
MRG_THREADS = 128                                # merge / softmax segment width
MRG_PER_HEAD = MRG_THREADS // H                  # 8 threads per head
MRG_COLS = BN // MRG_PER_HEAD                    # 4 columns per thread (128-bit)

# cp.async is per-thread and tops out at 128 bit = 16 B = 8 bf16.  A row is
# covered by ROW_BYTES/16 lanes; the remaining threads form row teams.  Keeping
# THREADS_PER_ROW > 1 is what puts two adjacent lanes in each 32 B sector.
CP_ELEMS = 8
CKV_TPR = (D_CKV * 2) // 16         # 64 lanes cover one 1024 B latent row
CKV_TEAMS = STEP_THREADS // CKV_TPR  # 4 row teams
CKV_TRIPS = BN // CKV_TEAMS          # 8 rows per team, a constexpr trip count
KPE_TPR = (D_KPE * 2) // 16         # 8 lanes cover one 128 B rotary row
KPE_TEAMS = STEP_THREADS // KPE_TPR  # 32 row teams
KPE_TRIPS = BN // KPE_TEAMS          # 1 row per team
WAIT_N = 1                          # D0.9 issue-order relation, D1.1 starting point
#   With D0.11's index feed mounted, a body commits TWO groups -- {KV tile j}
#   then {idx tile j+2} -- so wait(1) retires "the previous body's KV tile and
#   the ids it fetched" and leaves this body's KV tile in flight.  D1.1's ladder
#   moves in units of a body's pair, not of single groups.

# --- D0.11: the index ring ----------------------------------------------------
# 2 stages x 32 int32 = 256 B.  Stage j % 2 holds the page ids of tile j, so at
# the top of body j both live stages are full: j % 2 = tile j (issued now) and
# (j + 1) % 2 = tile j+1 (issued next body).  Body j re-fills stage j % 2 with
# the ids of tile j+2 AFTER rendezvous #1, i.e. after every thread has finished
# reading that stage for tile j -- which is what lets 2 stages carry a 2-body
# lead.  The fill is itself a cp.async in the same group stream as the KV
# supply, so the id chain never occupies a register and never issues an LDG:
# ids for tile j+2 are committed in body j, retired by body j+1's wait, read in
# body j+2.  The holding surface is smem (the D0.11 starting point): a thread
# issues cp.async for 8 different rows of the tile, so a lane-owned or
# register-resident form -- the D1.3 candidates -- would need a broadcast per
# row rather than one broadcast LDS.
IDX_STAGES = 2
IDX_BYTES = IDX_STAGES * BN * 4                  # 256 B


def _seg(site, row=0):
    """Timeline-probe hook. The driver rewrites this body on the probe arm."""
    return None


# ------------------------------------------------------- PTX escapes (form 3)
# `ex2.approx.f32` / `lg2.approx.f32` are the fp32 hardware transcendentals
# (SASS MUFU.EX2 / MUFU.LG2) and `max.f32` is FMNMX -- the instruction forms the
# milestone's D2.3 asks for.  They are emitted directly so no DSL-version
# spelling drift can silently push the softmax segment onto a slow path.

@dsl_user_op
def _ex2f(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _lg2f(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _fmaxf(a, b, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _shfl_xor(x, mask: cutlass.Constexpr, *, loc=None, ip=None) -> cutlass.Float32:
    """Butterfly exchange inside the warp (SASS SHFL.BFLY).  The row reductions
    of the softmax segment are 8 lanes wide and lane-aligned, so the segment
    mask 0x1f (whole warp) with a 1/2/4 butterfly reduces exactly inside each
    aligned group of 8."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "shfl.sync.bfly.b32 $0, $1, %d, 0x1f, 0xffffffff;" % int(mask), "=f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# `cute.arch` only exports the bulk (TMA) commit/wait pair; the non-bulk ones are
# operand-less per-thread instructions, so the shortest version-proof spelling is
# inline PTX.  has_side_effects=True is mandatory: a void asm without it is dead-
# code-eliminated and the symptom is "the data never arrived", not an error.

@dsl_user_op
def _cp_commit(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _cp_wait(n, *, loc=None, ip=None):
    """Block until at most `n` of this thread's committed groups are in flight."""
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _cp_async_idx(dst, src, nbytes, *, loc=None, ip=None):
    """One int32 gmem -> smem `cp.async` (D0.11's index-ring write side).

    cp-size 4 forces `.ca`: `.cg` is a 16 B-only cache mode, and the index ring
    cannot use 16 B copies because a chunk's first row is `kv_indptr[b] +
    c * CHUNK`, an arbitrary runtime offset into `kv_indices` -- a 16 B copy
    would need the source 16 B aligned and nothing makes it so.  `.ca` and `.cg`
    copies may share a commit group.

    `nbytes` is the PTX src-size operand: 4 copies the id, 0 writes 4 zero bytes
    and reads NOTHING from `src` (so a row past the chunk end costs no gmem
    traffic).  The address must still be a legal one either way.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4, $2;", "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# ------------------------------------------------------------- step kernel

@cute.kernel
def _step_kernel(
    mIndptr: cute.Tensor,      # [B+1]  int32
    mIndices: cute.Tensor,     # [T]    int32
    mCkv: cute.Tensor,         # [NUM_PAGES, 1, 512] bf16
    mKpe: cute.Tensor,         # [NUM_PAGES, 1, 64]  bf16
    mQn: cute.Tensor,          # [B, 16, 512] bf16
    mQp: cute.Tensor,          # [B, 16, 64]  bf16
    mPo: cute.Tensor,          # [W_max, 16, 512] fp32   partial_out (normalised)
    mPl: cute.Tensor,          # [W_max, 16]      fp32   partial_lse (base-2)
    scale2: cutlass.Float32,   # sm_scale * log2(e)      (D0.3)
    chunk: cutlass.Constexpr,
    nb: cutlass.Constexpr,     # batch size B
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    w = cutlass.Int32(bidx)
    lane = cute.arch.lane_idx()
    wid = cute.arch.warp_idx()

    _seg("step.entry")

    # --- D0.6 smem ---------------------------------------------------------
    # One ring buffer of (stage, row, 576) bf16 on a 584-element row stride.
    # `sKv` / `sPe` are two views of the SAME rows: columns [0, 512) are the
    # latent part (which is also V) and columns [512, 576) the rotary part, so
    # one page id lands as one row.
    smem = cutlass.utils.SmemAllocator()
    sRing = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((RING_STAGES, BN, ROW_W),
                         stride=(BN * ROW_S, ROW_S, 1)),
        byte_alignment=1024)
    sKv = cute.make_tensor(sRing.iterator,
                           cute.make_layout((RING_STAGES, BN, D_CKV),
                                            stride=(BN * ROW_S, ROW_S, 1)))
    # .align(16): slicing may leave the static alignment at the element width,
    # and both copies here are 128-bit (16 B on both sides is a hard requirement).
    sPe = cute.make_tensor(sRing[(None, None, D_CKV)].iterator.align(16),
                           cute.make_layout((RING_STAGES, BN, D_KPE),
                                            stride=(BN * ROW_S, ROW_S, 1)))

    # Q staging (16 heads x 576, nope ++ pe on the ring's row stride), the S
    # partial-sum slots, the P face and the cross-warp softmax statistics.
    sQ = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((H, ROW_W), stride=(ROW_S, 1)),
                              byte_alignment=1024)
    sQn = cute.make_tensor(sQ.iterator,
                           cute.make_layout((H, D_CKV), stride=(ROW_S, 1)))
    sQp = cute.make_tensor(sQ[(None, D_CKV)].iterator.align(16),
                           cute.make_layout((H, D_KPE), stride=(ROW_S, 1)))
    sS = smem.allocate_tensor(cutlass.Float32,
                              cute.make_layout((S_CONTRIB, H, BN),
                                               stride=(H * S_ROW, S_ROW, 1)),
                              byte_alignment=128)
    sP = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((H, BN), stride=(P_ROW, 1)),
                              byte_alignment=128)
    sStat = smem.allocate_tensor(cutlass.Float32, cute.make_layout((3, H)),
                                 byte_alignment=16)
    # D0.11: the index ring.  32 consecutive int32 per stage, so the KV issue
    # loops read it as a broadcast LDS (all 64 lanes of a cp.async row team want
    # the same page id) with no bank conflict.  The stride is written out: a
    # bare `make_layout((2, 32))` is COMPACT COLUMN-MAJOR (stride (1, 2)), which
    # would interleave the two stages word by word -- harmless while every
    # access goes through the layout, fatal now that the write side is a
    # `cp.async` taking a raw byte address (word (s, r) must be at 4*(32s + r)).
    sIdx = smem.allocate_tensor(cutlass.Int32,
                                cute.make_layout((IDX_STAGES, BN),
                                                 stride=(BN, 1)),
                                byte_alignment=16)

    # --- D0.2 device-side schedule: chunk_indptr block scan (no host sync) ---
    sCip = smem.allocate_tensor(cutlass.Int32,
                                cute.make_layout(STEP_SCAN_N),
                                byte_alignment=16)

    # --- D0.5 / D0.6 read side: one warp-level MMA atom, three LDSM members --
    # bM = 16 = all 16 heads, so a single m16n8k16 atom covers the head mode and
    # a warp owns a whole tile of it.  Members follow the face's contiguous mode:
    # Q (M, K) and P (M, K) are depth/KV contiguous -> non-transposed x4;
    # K as B = (N = KV row, K = depth) is depth contiguous -> non-transposed x2;
    # V as B = (N = out dim, K = KV row) is dim contiguous, i.e. N contiguous
    # -> TRANSPOSED x2.
    mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32,
                          (MMA_M, MMA_N, MMA_K)), (1, 1, 1))
    tcA = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16), mma)
    tcBs = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), cutlass.BFloat16), mma)
    tcBo = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), cutlass.BFloat16), mma)
    thr_mma = mma.get_slice(lane)          # warp-level MMA slices on the LANE
    thrA = tcA.get_slice(lane)
    thrBs = tcBs.get_slice(lane)
    thrBo = tcBo.get_slice(lane)

    # cp.async.cg: .BYPASS (skip L1) is the load-cache mode the ring wants, and
    # .cg admits only 16 B copies -- both rows are whole multiples of 16 B.
    cp_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    cp_atom = cute.make_copy_atom(cp_op, cutlass.BFloat16, num_bits_per_copy=128)
    tiled_ck = cute.make_tiled_copy_tv(cp_atom,
                                       cute.make_layout(CKV_TPR),
                                       cute.make_layout(CP_ELEMS))
    tiled_pe = cute.make_tiled_copy_tv(cp_atom,
                                       cute.make_layout(KPE_TPR),
                                       cute.make_layout(CP_ELEMS))
    thr_ck = tiled_ck.get_slice(tidx % CKV_TPR)
    thr_pe = tiled_pe.get_slice(tidx % KPE_TPR)
    ck_team = tidx // CKV_TPR
    pe_team = tidx // KPE_TPR
    # zero source for the tail rows (D0.12 supply side, see the loop)
    zck = cute.make_fragment_like(thr_ck.partition_D(sKv[(0, 0, None)]),
                                  cutlass.BFloat16)
    zpe = cute.make_fragment_like(thr_pe.partition_D(sPe[(0, 0, None)]),
                                  cutlass.BFloat16)
    zck.fill(0.0)
    zpe.fill(0.0)

    sCip[tidx] = cutlass.Int32(0)
    if tidx + STEP_THREADS < STEP_SCAN_N:
        sCip[tidx + STEP_THREADS] = cutlass.Int32(0)
    cute.arch.sync_threads()

    if tidx < nb:
        li = mIndptr[tidx + 1] - mIndptr[tidx]
        sCip[SCAN_PAD + 1 + tidx] = (li + (chunk - 1)) // chunk
    cute.arch.sync_threads()

    # Hillis-Steele inclusive scan.  The SCAN_PAD zeros in front make every read
    # in range for every thread, so the scan carries no branch at all.
    for s in cutlass.range_constexpr(SCAN_STEPS):
        off = 1 << s
        v = sCip[SCAN_PAD + tidx - off]
        cute.arch.sync_threads()
        sCip[SCAN_PAD + tidx] = sCip[SCAN_PAD + tidx] + v
        cute.arch.sync_threads()

    n_work = sCip[SCAN_PAD + nb]

    if w < n_work:
        # which (sequence, chunk) is this work item?  chunk_indptr is strictly
        # increasing (every sequence lists at least one page), so the count of
        # entries <= w is the sequence index.
        bsel = cutlass.Int32(0)
        for i in cutlass.range_constexpr(nb):
            bsel = bsel + cutlass.Int32(sCip[SCAN_PAD + 1 + i] <= w)
        c = w - sCip[SCAN_PAD + bsel]

        seq_beg = mIndptr[bsel]
        seq_end = mIndptr[bsel + 1]
        row_beg = seq_beg + c * chunk
        row_cap = row_beg + chunk
        row_end = row_cap + (seq_end - row_cap) * cutlass.Int32(seq_end < row_cap)
        n_rows = row_end - row_beg

        n_tiles = (n_rows + (BN - 1)) // BN

        # ---- prologue: stage Q, then park it in registers for the whole loop
        # (D0.5 residency (1): the per-warp Q depth slice is a loop invariant,
        # which is the whole point of splitting S over the depth dimension).
        for t in cutlass.range_constexpr(H // CKV_TEAMS):
            r = t * CKV_TEAMS + ck_team
            cute.copy(tiled_ck,
                      thr_ck.partition_S(mQn[(bsel, r, None)]),
                      thr_ck.partition_D(sQn[(r, None)]))
        if pe_team < H:
            cute.copy(tiled_pe,
                      thr_pe.partition_S(mQp[(bsel, pe_team, None)]),
                      thr_pe.partition_D(sQp[(pe_team, None)]))
        # D0.11 prologue: seed BOTH index stages, stage 0 with tile 0's ids and
        # stage 1 with tile 1's.  The steady state always issues off ids fetched
        # two bodies earlier, so the first TWO tiles' ids have to come from
        # outside the loop -- that pair of fetches is the prologue cost this
        # decision prices.  They ride the group the Q staging already commits,
        # and the barrier below (the one Q staging already needs) publishes them.
        # Thread `tidx` owns ring word `tidx` of both stages; addresses are byte
        # offsets off the two bases because `cp.async` takes raw addresses.
        gidx_b = mIndices.iterator.toint()               # gmem base, Int64
        sidx_b = sIdx.iterator.toint()                   # shared window, Int32
        if tidx < BN:
            rg0 = row_beg + tidx
            v0 = cutlass.Int32(rg0 < row_end)
            _cp_async_idx(sidx_b + tidx * 4,
                          gidx_b + cutlass.Int64(rg0 * v0) * 4, v0 * 4)
            rg1 = rg0 + BN
            v1 = cutlass.Int32(rg1 < row_end)
            _cp_async_idx(sidx_b + (BN + tidx) * 4,
                          gidx_b + cutlass.Int64(rg1 * v1) * 4, v1 * 4)
        _cp_commit()
        _cp_wait(0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        # C1 depth assignment, branch free: w0..w3 own the two 64-deep latent
        # chunks 2w and 2w+1, w4 owns chunk 8 (the rotary 64) twice with its
        # second Q fragment zeroed, w5..w7 sit out the S phase entirely.  Making
        # w4's shape identical to the others is what keeps ONE unrolled S body
        # instead of a per-warp branch around the whole tile loop.
        lt4 = cutlass.Int32(wid < 4)
        dc0 = lt4 * (2 * wid) + (1 - lt4) * (N_DEPTH_CHUNKS - 1)
        dc1 = lt4 * (2 * wid + 1) + (1 - lt4) * (N_DEPTH_CHUNKS - 1)
        lt4f = cutlass.Float32(lt4)

        q9 = cute.flat_divide(sQ, (H, DEPTH_CHUNK))          # (16, 64, 1, 9)
        qf0 = q9[(None, None, 0, dc0)]
        frQ0 = thr_mma.make_fragment_A(thr_mma.partition_A(qf0))
        cute.copy(tcA, thrA.partition_S(qf0), thrA.retile(frQ0))
        qf1 = q9[(None, None, 0, dc1)]
        frQ1 = thr_mma.make_fragment_A(thr_mma.partition_A(qf1))
        cute.copy(tcA, thrA.partition_S(qf1), thrA.retile(frQ1))
        for i in cutlass.range_constexpr(cute.size(frQ1)):
            frQ1[i] = cutlass.BFloat16(cutlass.Float32(frQ1[i]) * lt4f)

        # ---- D0.5 residency (2): the O accumulator, 8 n8 chains = 64 output
        # dims = 32 fp32 words per thread, alive across the whole KV loop.
        accO = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((H, O_DIMS_PER_WARP)), cutlass.Float32)
        accO.fill(0.0)
        cCo = thr_mma.partition_C(
            cute.make_identity_tensor((H, O_DIMS_PER_WARP)))
        # ---- D0.8: fixed containers, allocated once outside the KV loop and
        # overwritten per tile.  Shapes come from a dummy partition at constant
        # coordinates, so they are exactly the profiles `cute.gemm` and
        # `retile` expect, and no unrolled iteration opens a fresh alloca.
        kt0 = sRing[(0, None, None)]
        ks0 = cute.flat_divide(kt0, (MMA_N, DEPTH_CHUNK))[(None, None, 0, 0)]
        accS = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((H, MMA_N)), cutlass.Float32)
        frK0 = thr_mma.make_fragment_B(thr_mma.partition_B(ks0))
        frK1 = thr_mma.make_fragment_B(thr_mma.partition_B(ks0))
        p0 = cute.flat_divide(sP, (H, MMA_K))[(None, None, 0, 0)]
        frP = thr_mma.make_fragment_A(thr_mma.partition_A(p0))
        v0 = cute.flat_divide(kt0, (MMA_K, O_DIMS_PER_WARP))[(None, None, 0, 0)]
        vb0 = cute.make_tensor(v0.iterator.align(16),
                               cute.select(v0.layout, [1, 0]))
        frV = thr_mma.make_fragment_B(thr_mma.partition_B(vb0))
        # fixed containers for the merge / softmax segment (allocated once,
        # overwritten per tile -- no per-tile alloca, no register cliff)
        srg = cute.make_rmem_tensor(cute.make_layout(MRG_COLS), cutlass.Float32)
        stg = cute.make_rmem_tensor(cute.make_layout(MRG_COLS), cutlass.Float32)
        prg = cute.make_rmem_tensor(cute.make_layout(MRG_COLS), cutlass.BFloat16)
        mh = tidx // MRG_PER_HEAD                    # this thread's head
        mc = tidx % MRG_PER_HEAD                     # its column group
        mcol = mc * MRG_COLS

        if tidx < H:
            sStat[(0, tidx)] = cutlass.Float32(NEG_BIG)   # running max m
            sStat[(1, tidx)] = cutlass.Float32(0.0)       # running sum l
        cute.arch.sync_threads()

        # --- D0.9 issue order.  Iteration j issues tile j and consumes tile
        # j-1, i.e. exactly "body i issues tile i+1, waits on the tile the
        # previous body issued, consumes tile i" with the 1-tile prologue folded
        # into j = 0 -- one issue site, one consume site.  The extra trailing
        # iteration issues nothing (every row is past row_end) and therefore
        # commits an EMPTY group, which keeps every thread's in-flight group
        # count aligned so the last real tile is still covered by wait(N).
        for j in cutlass.range(n_tiles + 1):
            st = j % RING_STAGES
            base_g = row_beg + j * BN

            # D0.12 supply side: a row past the chunk end is never issued.
            # Clamping it onto another listed row would count that row twice;
            # clamping onto an unlisted one would read a poisoned page.  It is
            # zero-filled in place rather than merely skipped: with a real
            # consumer the stale bytes of that ring slot become a V operand, and
            # 0 * (whatever was there two tiles ago) is only harmless if
            # "whatever" is finite -- an uninitialised smem word is not.  Zeroed
            # rows give s = 0 (then masked to -inf on the consume side) and
            # V = 0, both exact.
            for t in cutlass.range_constexpr(CKV_TRIPS):
                r = t * CKV_TEAMS + ck_team
                rg = base_g + r
                if rg < row_end:
                    pg = sIdx[(j % IDX_STAGES, r)]      # D0.11: fetched at j-2
                    cute.copy(tiled_ck,
                              thr_ck.partition_S(mCkv[(pg, 0, None)]),
                              thr_ck.partition_D(sKv[(st, r, None)]))
                else:
                    # only the LAST REAL tile has rows to zero; the trailing
                    # empty body (which exists purely to keep the in-flight
                    # group count aligned) must not pay a full tile of stores.
                    if base_g < row_end:
                        cute.autovec_copy(zck,
                                          thr_ck.partition_D(sKv[(st, r, None)]))
            for t in cutlass.range_constexpr(KPE_TRIPS):
                r = t * KPE_TEAMS + pe_team
                rg = base_g + r
                if rg < row_end:
                    pg = sIdx[(j % IDX_STAGES, r)]      # D0.11: fetched at j-2
                    cute.copy(tiled_pe,
                              thr_pe.partition_S(mKpe[(pg, 0, None)]),
                              thr_pe.partition_D(sPe[(st, r, None)]))
                else:
                    if base_g < row_end:
                        cute.autovec_copy(zpe,
                                          thr_pe.partition_D(sPe[(st, r, None)]))
            _cp_commit()
            _seg("step.kv.commit", j)

            # Wait on the tile the PREVIOUS iteration issued, never on the one
            # just issued: at j the in-flight groups are {KV tile j-1, idx tile
            # j+1, KV tile j} -- the index feed below interleaves one group of
            # its own per body -- and wait(1) retires everything but the newest,
            # i.e. exactly the previous body's KV tile plus the ids that body
            # fetched.  At j = 0 there is a single group in flight, so this is a
            # no-op and nothing is consumed.
            _seg("step.kv.wait.begin", j)
            _cp_wait(WAIT_N)
            _seg("step.kv.wait.end", j)
            cute.arch.fence_view_async_shared()
            # D0.10 rendezvous #1: cp.async retires per thread, so only a
            # barrier says "the whole stage is readable by everybody".
            _seg("step.bar1.begin", j)
            cute.arch.sync_threads()
            _seg("step.bar1.end", j)

            # --- D0.11 index prefetch: tile j+2's ids into stage j % 2 --------
            # Rendezvous #1 above is the licence to write this stage: every
            # thread has already read it for tile j's KV issue.  The fetch is a
            # cp.async, so nothing about the id chain lives in a register and
            # the body issues NO index LDG at all (the machine-readable form of
            # [GATHER: prefetched] is "no index LDG after the first LDGSTS in
            # the loop body"; here the count is zero for the whole body).  Its
            # own commit puts it one group AHEAD of the next body's KV group:
            # body j's group stream is {KV tile j} then {idx tile j+2}, so
            # body j+1's wait(WAIT_N=1) -- which is the wait that retires KV
            # tile j for the consume -- retires these ids too, one full body
            # before body j+2 issues off them.  The gmem latency of the fetch is
            # therefore spanned by the rest of this body's consume segment plus
            # the whole of body j+1: no LDGSTS ever waits on its own row's id.
            # Out-of-range rows go out with src-size 0: 4 zero bytes, no read.
            if tidx < BN:
                ist = (j + 2) % IDX_STAGES              # = j % 2: 2 stages, lead 2
                rgn = base_g + 2 * BN + tidx
                vn = cutlass.Int32(rgn < row_end)
                _cp_async_idx(sidx_b + (ist * BN + tidx) * 4,
                              gidx_b + cutlass.Int64(rgn * vn) * 4, vn * 4)
            _cp_commit()

            cst = (j + RING_STAGES - 1) % RING_STAGES
            cbase = (j - 1) * BN
            nv = n_rows - cbase
            nv = nv + (BN - nv) * cutlass.Int32(nv > BN)      # min(nv, BN)
            nv = nv * cutlass.Int32(j > 0)                    # prologue: nothing yet

            if nv > 0:
                kTile = sRing[(cst, None, None)]              # (32, 576)
                ks = cute.flat_divide(kTile, (MMA_N, DEPTH_CHUNK))
                #                                          -> (8, 64, 4, 9)

                # ---- phase S: S = Q . K^T, depth split over w0..w4 ----------
                _seg("step.s.begin", j)
                if wid < S_CONTRIB:
                    ssw = sS[(wid, None, None)]               # (16, 32) fp32
                    sn = cute.flat_divide(ssw, (H, MMA_N))    # (16, 8, 1, 4)
                    for n in cutlass.range_constexpr(N_CHAIN_S):
                        accS.fill(0.0)
                        kf0 = ks[(None, None, n, dc0)]        # (8 kv, 64 depth)
                        cute.copy(tcBs, thrBs.partition_S(kf0),
                                  thrBs.retile(frK0))
                        cute.gemm(mma, accS, frQ0, frK0, accS)
                        kf1 = ks[(None, None, n, dc1)]
                        cute.copy(tcBs, thrBs.partition_S(kf1),
                                  thrBs.retile(frK1))
                        cute.gemm(mma, accS, frQ1, frK1, accS)
                        # publish this chain: the C-frag holds two adjacent n
                        # per lane, so partition_C + autovec is a 64-bit STS on
                        # the 40-word row stride = 32 distinct banks per phase.
                        cute.autovec_copy(
                            accS, thr_mma.partition_C(sn[(None, None, 0, n)]))
                _seg("step.s.end", j)

                # D0.10 rendezvous #2: the merge reads all five contributions.
                _seg("step.bar2.begin", j)
                cute.arch.sync_threads()
                _seg("step.bar2.end", j)

                # ---- merge + row max (128 threads, 8 per head, 4 cols each) --
                _seg("step.mrg.begin", j)
                if tidx < MRG_THREADS:
                    for k in cutlass.range_constexpr(MRG_COLS):
                        srg[k] = cutlass.Float32(0.0)
                    for s in cutlass.range_constexpr(S_CONTRIB):
                        cute.autovec_copy(
                            cute.local_tile(sS[(s, mh, None)], (MRG_COLS,), (mc,)),
                            stg)
                        for k in cutlass.range_constexpr(MRG_COLS):
                            srg[k] = srg[k] + stg[k]
                    # D0.3 base-2 domain + D0.12 consume-side mask: the columns
                    # past the tail are forced to -inf so 2^(s - m) is exactly 0.
                    tmx = cutlass.Float32(NEG_BIG)
                    for k in cutlass.range_constexpr(MRG_COLS):
                        vk = cutlass.Int32(mcol + k < nv)
                        srg[k] = (srg[k] * scale2 * cutlass.Float32(vk)
                                  + cutlass.Float32(1 - vk)
                                  * cutlass.Float32(NEG_BIG))
                        tmx = _fmaxf(tmx, srg[k])
                    tmx = _fmaxf(tmx, _shfl_xor(tmx, 1))
                    tmx = _fmaxf(tmx, _shfl_xor(tmx, 2))
                    tmx = _fmaxf(tmx, _shfl_xor(tmx, 4))
                    if mc == 0:
                        mo = sStat[(0, mh)]
                        mn = _fmaxf(mo, tmx)
                        sStat[(0, mh)] = mn
                        sStat[(2, mh)] = _ex2f(mo - mn)   # rescale for O and l
                _seg("step.mrg.end", j)

                # D0.10 rendezvous #3: the running max is a CTA-level quantity.
                _seg("step.bar3.begin", j)
                cute.arch.sync_threads()
                _seg("step.bar3.end", j)

                # ---- softmax + publish P and l -----------------------------
                _seg("step.sm.begin", j)
                if tidx < MRG_THREADS:
                    mn = sStat[(0, mh)]
                    rs = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(MRG_COLS):
                        pk = _ex2f(srg[k] - mn)
                        rs = rs + pk
                        prg[k] = cutlass.BFloat16(pk)
                    rs = rs + _shfl_xor(rs, 1)
                    rs = rs + _shfl_xor(rs, 2)
                    rs = rs + _shfl_xor(rs, 4)
                    if mc == 0:
                        sStat[(1, mh)] = sStat[(1, mh)] * sStat[(2, mh)] + rs
                    cute.autovec_copy(
                        prg, cute.local_tile(sP[(mh, None)], (MRG_COLS,), (mc,)))
                _seg("step.sm.end", j)

                # D0.10 rendezvous #4: every warp reads the whole 16 x 32 P.
                _seg("step.bar4.begin", j)
                cute.arch.sync_threads()
                _seg("step.bar4.end", j)

                # ---- phase O: O = O * 2^(m_old - m_new) + P . V -------------
                _seg("step.o.begin", j)
                for i in cutlass.range_constexpr(cute.size(accO)):
                    hm, dnm = cCo[i]
                    accO[i] = accO[i] * sStat[(2, hm)]
                pk2 = cute.flat_divide(sP, (H, MMA_K))        # (16, 16, 1, 2)
                ko = cute.flat_divide(kTile, (MMA_K, O_DIMS_PER_WARP))
                #                                          -> (16, 64, 2, 9)
                for kb in cutlass.range_constexpr(KB_O):
                    pf = pk2[(None, None, 0, kb)]             # (16 head, 16 kv)
                    cute.copy(tcA, thrA.partition_S(pf), thrA.retile(frP))
                    vf = ko[(None, None, kb, wid)]            # (16 kv, 64 dim)
                    # B = V^T recorded as (N = dim, K = kv): pure relabel, the
                    # contiguous mode is still the dim = N mode -> transposed
                    # LDSM member (tcBo).
                    vb = cute.make_tensor(vf.iterator.align(16),
                                          cute.select(vf.layout, [1, 0]))
                    cute.copy(tcBo, thrBo.partition_S(vb), thrBo.retile(frV))
                    cute.gemm(mma, accO, frP, frV, accO)
                _seg("step.o.end", j)

            # D0.10 rendezvous #5: nobody may refill this stage until every warp
            # is done reading it (the next iteration issues tile j+1 into stage
            # (j + 1) % 2 == cst).
            _seg("step.bar5.begin", j)
            cute.arch.sync_threads()
            _seg("step.bar5.end", j)

        # ---- epilogue: partial_out = Otilde_k (already normalised, D0.3),
        # partial_lse = m + log2(l) in the base-2 domain.
        if tidx < H:
            lh = sStat[(1, tidx)]
            mPl[(w, tidx)] = sStat[(0, tidx)] + _lg2f(lh)
            sStat[(2, tidx)] = cutlass.Float32(1.0) / lh
        cute.arch.sync_threads()
        gpo = cute.flat_divide(mPo[(w, None, None)], (H, O_DIMS_PER_WARP))
        for i in cutlass.range_constexpr(cute.size(accO)):
            hm, dnm = cCo[i]
            accO[i] = accO[i] * sStat[(2, hm)]
        cute.autovec_copy(accO,
                          thr_mma.partition_C(gpo[(None, None, 0, wid)]))

    _seg("step.exit")


# ---------------------------------------------------------- combine kernel

@cute.kernel
def _combine_kernel(
    mIndptr: cute.Tensor,      # [B+1] int32
    mPo: cute.Tensor,          # [W_max, 16, 512] fp32
    mPl: cute.Tensor,          # [W_max, 16]      fp32
    mOut: cute.Tensor,         # [B, 16, 512] bf16
    mLse: cute.Tensor,         # [B, 16]      fp32
    chunk: cutlass.Constexpr,
    nb: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    b = cutlass.Int32(bx)
    h = cutlass.Int32(by)

    _seg("combine.entry")

    smem = cutlass.utils.SmemAllocator()
    sCip = smem.allocate_tensor(cutlass.Int32,
                                cute.make_layout(COMB_SCAN_N),
                                byte_alignment=16)

    sCip[tidx] = cutlass.Int32(0)
    if tidx + COMBINE_THREADS < COMB_SCAN_N:
        sCip[tidx + COMBINE_THREADS] = cutlass.Int32(0)
    cute.arch.sync_threads()

    if tidx < nb:
        li = mIndptr[tidx + 1] - mIndptr[tidx]
        sCip[SCAN_PAD + 1 + tidx] = (li + (chunk - 1)) // chunk
    cute.arch.sync_threads()

    for s in cutlass.range_constexpr(SCAN_STEPS):
        off = 1 << s
        v = sCip[SCAN_PAD + tidx - off]
        cute.arch.sync_threads()
        sCip[SCAN_PAD + tidx] = sCip[SCAN_PAD + tidx] + v
        cute.arch.sync_threads()

    beg = sCip[SCAN_PAD + b]
    end = sCip[SCAN_PAD + b + 1]
    n_k = end - beg

    # pass 1 -- the chunk-wise base-2 log-sum-exps share one running max
    mx = cutlass.Float32(-3.0e38)
    for k in cutlass.range(n_k):
        mx = _fmaxf(mx, mPl[(beg + k, h)])

    # pass 2 -- sec 1.2 merge: out = sum_k Otilde_k * 2^(lse_k - L), L = lse of
    # the whole sequence.  partial_out is ALREADY normalised per chunk (D0.3),
    # so the weights are exactly 2^(lse_k - mx) / l.
    acc = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    acc.fill(0.0)
    rp = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
    l = cutlass.Float32(0.0)
    for k in cutlass.range(n_k):
        wk = _ex2f(mPl[(beg + k, h)] - mx)
        l = l + wk
        cute.autovec_copy(cute.local_tile(mPo[(beg + k, h, None)], (4,), (tidx,)), rp)
        for j in cutlass.range_constexpr(4):
            acc[j] = acc[j] + wk * rp[j]

    inv = cutlass.Float32(1.0) / l
    ro = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
    for j in cutlass.range_constexpr(4):
        ro[j] = cutlass.BFloat16(acc[j] * inv)
    cute.autovec_copy(ro, cute.local_tile(mOut[(b, h, None)], (4,), (tidx,)))

    if tidx == 0:
        mLse[(b, h)] = mx + _lg2f(l)

    _seg("combine.exit")


# ------------------------------------------------------------------ launcher

@cute.jit
def _launch(
    mIndptr, mIndices, mCkv, mKpe, mQn, mQp, mPo, mPl, mOut, mLse,
    scale2: cutlass.Float32,
    chunk: cutlass.Constexpr,
    nb: cutlass.Constexpr,
    n_step_cta: cutlass.Constexpr,
    stream,
):
    _step_kernel(mIndptr, mIndices, mCkv, mKpe, mQn, mQp, mPo, mPl,
                 scale2, chunk, nb).launch(
        grid=[n_step_cta, 1, 1], block=[STEP_THREADS, 1, 1],
        min_blocks_per_mp=MIN_CTA_PER_SM, stream=stream)
    _combine_kernel(mIndptr, mPo, mPl, mOut, mLse, chunk, nb).launch(
        grid=[nb, H, 1], block=[COMBINE_THREADS, 1, 1], stream=stream)


# --------------------------------------------------------------- host side

_COMPILED = {}
_STREAM_CTOR = None


def _cu_stream():
    """torch's current stream as the driver handle the DSL launcher wants."""
    global _STREAM_CTOR
    ptr = torch.cuda.current_stream().cuda_stream
    if _STREAM_CTOR is None:
        try:
            from cuda.bindings import driver as _drv
            _STREAM_CTOR = _drv.CUstream
        except ImportError:
            from cuda import cuda as _drv          # older cuda-python layout
            _STREAM_CTOR = _drv.CUstream
    return _STREAM_CTOR(ptr)


def mla_paged_decode(
    q_nope: torch.Tensor,      # [B, H=16, D_CKV=512] bf16
    q_pe: torch.Tensor,        # [B, H=16, D_KPE=64] bf16
    ckv_cache: torch.Tensor,   # [NUM_PAGES, 1, 512] bf16  (latent K == V)
    kpe_cache: torch.Tensor,   # [NUM_PAGES, 1, 64] bf16   (rotary K part)
    kv_indptr: torch.Tensor,   # [B + 1] int32, page-list offsets per sequence
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] int32, page ids
    sm_scale: float,           # softmax scale (campaign: 0.13523…, from the trace)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [B, 16, 512] bf16, lse [B, 16] fp32 base-2 log-sum-exp)."""
    b = q_nope.shape[0]
    assert b <= B_MAX, f"batch {b} exceeds the scan width {B_MAX}"

    # --- D0.2: SHAPES ONLY.  No .item(), no .cpu(), no value-dependent branch:
    # a single device->host sync here would drain the 50-call device window.
    t_tot = kv_indices.numel()
    per = NCTA_TARGET - b
    if per < 1:
        per = 1
    raw = (t_tot + per - 1) // per
    chunk = ((raw + 31) // 32) * 32
    if chunk < 32:
        chunk = 32
    # sum_i ceil(L_i / chunk) <= b + t_tot // chunk holds for every page list
    # with L_i >= 1, so this is a safe (and tight) bound on the work items.
    n_step_cta = b + t_tot // chunk

    dev = q_nope.device
    # D0.3 / D0.4: nobody zero-fills.  Every (w, h, :) slot is fully overwritten
    # by exactly one step CTA, and combine fully overwrites out / lse.
    partial_out = torch.empty((n_step_cta, H, D_CKV), dtype=torch.float32, device=dev)
    partial_lse = torch.empty((n_step_cta, H), dtype=torch.float32, device=dev)
    out = torch.empty((b, H, D_CKV), dtype=torch.bfloat16, device=dev)
    lse = torch.empty((b, H), dtype=torch.float32, device=dev)

    a_indptr = from_dlpack(kv_indptr, assumed_align=16)
    a_indices = from_dlpack(kv_indices, assumed_align=16)
    a_ckv = from_dlpack(ckv_cache, assumed_align=16)
    a_kpe = from_dlpack(kpe_cache, assumed_align=16)
    a_qn = from_dlpack(q_nope, assumed_align=16)
    a_qp = from_dlpack(q_pe, assumed_align=16)
    a_po = from_dlpack(partial_out, assumed_align=16)
    a_pl = from_dlpack(partial_lse, assumed_align=16)
    a_out = from_dlpack(out, assumed_align=16)
    a_lse = from_dlpack(lse, assumed_align=16)

    scale2 = cutlass.Float32(float(sm_scale) * LOG2E)   # D0.3: base-2 domain
    stream = _cu_stream()

    key = (b, t_tot, int(ckv_cache.shape[0]), chunk, n_step_cta)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_launch, a_indptr, a_indices, a_ckv, a_kpe, a_qn, a_qp,
                          a_po, a_pl, a_out, a_lse, scale2,
                          chunk, b, n_step_cta, stream)
        _COMPILED[key] = fn
    fn(a_indptr, a_indices, a_ckv, a_kpe, a_qn, a_qp, a_po, a_pl, a_out, a_lse,
       scale2, stream)

    return out, lse
