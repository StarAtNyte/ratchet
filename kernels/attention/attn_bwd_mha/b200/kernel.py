"""flash_attn_bwd — the student's kernel file (the ONLY file the student edits).

Dense multi-head causal attention BACKWARD on Blackwell (sm_100a): given the
forward's output `o` and log-sum-exp `lse`, compute dq / dk / dv from `do`.
Plain attention — one kv head per q head (no GQA), no paging, no dropout,
no sliding window, no softcap. Layout is the FlashAttention public layout
(batch, seqlen, heads, headdim). The signature below is the card's frozen
contract; the student implements the body in CuTe DSL; the signature itself
may not change.

  flash_attn_bwd(q, k, v, o, do, lse) -> (dq, dk, dv)

    q, k, v : torch.Tensor (B, N, H, D) cuda bfloat16
    o       : torch.Tensor (B, N, H, D) cuda bfloat16   forward output (given)
    do      : torch.Tensor (B, N, H, D) cuda bfloat16   gradient wrt o
    lse     : torch.Tensor (B, H, N)    cuda float32    forward log-sum-exp
                                                       (natural log, of the
                                                       scaled, masked scores)
    dq, dk, dv : torch.Tensor (B, N, H, D) cuda bfloat16

  B=8, N=4096, H=16, D=128; softmax scale = 1/sqrt(D); causal (query i attends
  keys j <= i); fp32 accumulation inside.

Semantics (per batch b, head h; fp32 chain; i, j index the sequence):

    S[i, j] = scale * <q[i], k[j]>          for j <= i, else -inf
    P[i, j] = exp(S[i, j] - lse[i])         (lse given: P rows sum to 1)
    dv[j]   = sum_i P[i, j] * do[i]
    dP[i,j] = <do[i], v[j]>
    delta[i]= <do[i], o[i]>
    dS[i,j] = P[i, j] * (dP[i, j] - delta[i])
    dq[i]   = scale * sum_j dS[i, j] * k[j]
    dk[j]   = scale * sum_i dS[i, j] * q[i]

The arithmetic on q / k / v / o / do / lse / dq / dk / dv is the student's
CuTe DSL kernels and nothing else: delegating any of it to torch / SDPA /
flash_attn / quack / triton / autograd, compiling C++/CUDA on the side, or
loading a library kernel fails the harness (eval/bench.py scans this file).
Host-side torch is fine for allocation and integer scheduling metadata.
JIT-compile once and cache the compiled kernel across calls (the bench times
repeated calls on rotating buffers).

--------------------------------------------------------------------------
Build state (M0 construction order, all six steps landed; step 2a landed the
TMEM leg, 2b the TMA supply leg, 2c the MMA leg, 3 the softmax segment's tmem
side, 4 the dS cross-CTA exchange and chain3 dQ, 5 the dQ reduction path, 6 the
dK/dV epilogue and D0.7's setmaxreg split, with the cross-trip software
pipeline and the segment's activity budget, D0.13 x D0.14, across 5 and 6):

  preprocess  : real form  (dpsum / lse_log2 / dq_accum zeroing)   [D0.3, D0.4]
                  - 16 B granules on all three streams, the row sum
                    a 16-lane SHFL.BFLY, the -inf guard an FSEL  [D2.1]
  main        : resource topology + operand supply + all five chains, with the
                softmax segment feeding all three of their intermediates
                  - 16 warp roles by warp_idx                      [D0.7]
                  - 512-column 2-CTA TMEM allocation + column map  [D0.6]
                  - the 231,424 B arena as D0.8's buffer list, the
                    seven operand faces held as swizzled tiles     [D0.8]
                  - TMA producer: K / V / K_dsplit prologue once,
                    Q_row / dO_row / dO_col / Q_col / lse / dpsum
                    single-stage per trip, pair-accounted mbarriers [D0.10]
                  - causal trip structure m in [2p, 32)            [D0.5]
                  - chain0 S^T = K . Q_row^T and chain2 dP^T =
                    V . dO_row^T, cta_group::2 UMMA, 8 K-steps
                    each, accumulators in TMEM at 0x000 / 0x100    [D0.6, D0.9]
                  - chain1 dK += dS^T . Q_col and chain4 dV +=
                    P^T . dO_col: D0.9's two RS rows, A read out
                    of TMEM at 0x100 / 0x000 as packed bf16 with
                    no smem copy, B an MN-major smem face, f32
                    accumulators live across trips at 0x180 /
                    0x080                                          [D0.6, D0.9]
                  - the softmax segment on warps 0-7: LDTM S^T ->
                    branch-free causal select, its predicate a
                    two-word threshold bitmap built once per lane
                    per trip -> exp2 in the log2
                    domain -> bf16 pack -> STTM P^T; LDTM dP^T ->
                    P * (dP - dpsum) -> STTM dS^T, with the four
                    tmem handoff barriers 0x40 / 0x48 / 0x50 /
                    0x58 tying it to the MMA warp                  [D0.11, D0.4]
                  - the same dS, transposed into smem and exchanged
                    across the pair: the warpgroup holding my Q
                    rows stores the local half of 0x2c400, the one
                    holding the peer's stores 0x20400, and a
                    16 KB `UBLKCP.S.S` with a remote expect-tx
                    lands it in the peer's face; the relay warp
                    turns 0x158 into the count-2 0x160            [D0.11, D0.9]
                  - chain3 dQ = dS . K_dsplit, M128xN128xK16
                    cta_group::2 with both operands MN-major smem
                    descriptors, D = tmem 0x40-0x7f behind 0x60
                    and 0x98, releasing the face with a multicast
                    0x68                                           [D0.6, D0.9]
                  - the dQ-writer on warps 8-11: LDTM the 64 f32
                    columns at 0x40, arrive 0x98, then 8 x [all 128
                    lanes store their own 32 B of one 4 KB chunk
                    (two column quads) into the 4-slot ring ->
                    proxy fence -> BAR(0x4) -> elected
                    `cp.reduce.async.bulk...add.f32` into the
                    quad-major f32 workspace -> commit -> wait_group
                    (3, read) -> BAR(0x4)]                         [D0.12, D0.3]
                  - the cross-trip pipeline: a prologue that warms
                    up one beat (S(0), dP(0), dV(0)) and a body
                    whose trip i issues S(i+1), dK(i), dP(i+1),
                    dQ(i), dV(i+1), each of the eight waits naming
                    its own tile, every operand face released by
                    the commit of the chain that read it and no
                    rendezvous left in the trip                    [D0.13]
                  - the segment's activity budget: both f32
                    operands read in two 32-column slices so that
                    only P's fp32 line is live across segments,
                    and ONE bf16 staging serving the STTM and the
                    smem row alike                                 [D0.14]
                  - every f32 -> bf16 rounding in the kernel (P^T,
                    dS^T and the dK / dV epilogue) taken as ONE
                    convert of the whole 64-value line, which is
                    the packed `F2FP.BF16.F32.PACK_AB` form        [D0.4]
                  - the dK / dV epilogue: after the loop the two
                    accumulators are declared FINAL (0x70 / 0x78,
                    multicast), each softmax warpgroup LDTMs its
                    64 d columns of one of them, scales dK by
                    softmax_scale, rounds to bf16 and lands it in
                    a (128 KV, 64 d) K-major staging face aliased
                    onto the V / K regions, and one warp per
                    warpgroup stores that box with a bulk-tensor
                    TMA store                                      [D0.15, D0.4]
                  - D0.7's register split, warpgroup granular:
                    136 for warps 0-11, 104 for 12-15, which is
                    the whole 2048-warp-reg file at 1 CTA/SM       [D0.7]
  postprocess : real form: 32 x 16 B per thread out of D0.3's block
                (thread = TMEM lane, 512 contiguous bytes per warp
                per instruction) -> x scale -> bf16 -> a 32 KB
                rotation-swizzled staging tile -> 16 B per thread, 16
                threads per q row, into (B, N, H, D)      [D2.2, D0.3, D0.4]

All five chains now run on real operands: the two RS chains read the P^T / dS^T
the segment writes in TMEM, and chain3 reads the exchanged dS out of smem
(D0.9's one smem row — the only intermediate that changes orientation and
crosses to the peer). dQ leaves the kernel on D0.12's path — the accumulator is
drained into the ring and reduced at the L2 slice, with no SM-side atomics and no
per-element store — so the main kernel now produces dq end to end (preprocess
zeroes the workspace, the reduce-add accumulates into it, postprocess scales and
rounds it). The operand faces are released by the commits of the chains that read
them (D0.13), so step 2b's per-trip cluster rendezvous is gone and the TMA warp
runs two tiles ahead on Q_row / dO and one tile ahead on Q_col / lse / dpsum.
All three outputs are now written: dq through D0.12's reduction and the
postprocess kernel, dk / dv through D0.15's epilogue, which drains tmem 0x180 /
0x080 into the K / V regions (empty from the last S^T / dP^T chain onwards) and
stores them with four `UTMASTG.4D` boxes per pair. No smem is opened for it and
the arena is still 231,424 B.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutils
import cutlass.utils.blackwell_helpers as bh
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


# --- frozen campaign point ----------------------------------------------------
_B, _N, _H, _D = 8, 4096, 16, 128
_BLK_M = 128          # Q rows per block (preprocess / postprocess / main tile)
_BLK_N = 128          # KV rows per CTA in the main kernel (pair covers 256)
_D_HALF = 64          # d columns owned by one CTA of the pair
_NBLK = _N // _BLK_M  # 32 blocks along the sequence

_PRE_THREADS = 256
_POST_THREADS = 128
_MAIN_THREADS = 512

_SCALE = 1.0 / math.sqrt(_D)
_LOG2E = 1.4426950408889634
_SCALE_LOG2 = _SCALE * _LOG2E

# (B, N, H, D) strides, in elements.
_STRIDE_B = _N * _H * _D
_STRIDE_N = _H * _D
_STRIDE_H = _D

# dq_accum (D0.3): one 64 KB run per (b, h, m_block), laid out QUAD-MAJOR /
# LANE-INNER --
#     [cta_rank 2][column quad 16][tmem lane 128][4 f32]
# -- so the innermost 16 B is ONE lane's four adjacent accumulator columns, a
# quad is 2 KB (all 128 lanes of it), this CTA's 64 columns are its 16 quads =
# 32 KB, and the pair's two ranks are the 64 KB run. The axes are chain3's
# accumulator axes verbatim, which is what makes every consumer of this block an
# atom rather than a gather: the writer's 4 KB chunk is two whole quads
# ([2][128 lane][4]) and is copied byte for byte, and the postprocess reader
# takes one (rank, quad)'s 16 B per thread with thread = lane, so a warp's 32
# lanes are 512 contiguous bytes.
# The SEMANTICS come from chain3's MEASURED lane map (see the D0.12 block
# below): rank = the Q half, lane = (d half, q row within 64), column = d within
# the half, i.e.
#     q = 64*rank + lane % 64,     d = 64*(lane // 64) + col
# and the run covers q rows [128*m_block, +128) x all 128 d of one (b, h).
_ACC_LANES = 128                                   # chain3's TMEM lane count
_ACC_VEC = 4                                       # f32 per lane per quad, 16 B
_ACC_QUADS = _D_HALF // _ACC_VEC                   # 16 column quads per CTA
_ACC_QUAD_STRIDE = _ACC_LANES * _ACC_VEC           # 512 f32 = 2 KB
_ACC_RANK_STRIDE = _ACC_QUADS * _ACC_QUAD_STRIDE   # 8192 f32 = 32 KB
_ACC_TILE = 2 * _ACC_RANK_STRIDE                   # 16384 f32 = 64 KB
_ACC_ELEMS = _B * _H * _NBLK * _ACC_TILE           # 67,108,864 f32 = 256 MiB

_NEG_BIG = -3.0e38                           # -inf sentinel for the lse guard
_PARK = -1.0e30                              # masked score park value, FINITE
                                             # (hw_kb branch_free_select_mask:
                                             # PARK * scale_log2 still exp2's to
                                             # 0 and never makes a NaN)

# --- D0.11: the softmax segment's split of the accumulator --------------------
# The 8 softmax warps are two warpgroups; each covers all 128 TMEM lanes (a warp
# reaches lanes [32*(warp%4), +32)) and owns 64 of the accumulator's 128 columns,
# which is D0.7's "每 lane 64 列". A packed bf16 column pair is one 32-bit column,
# so a warpgroup's 64 f32 columns become 32 32-bit columns.
_WG_COLS = 64
_HALF_COLS = _WG_COLS // 2    # D0.14: the transient operands are read in two
                              # 32-column slices, so only 32 of them are live
                              # at a time while P stays fp32 across the segment
_WG_THREADS = 128             # one warpgroup, which is also the TMEM lane count
_SOFTMAX_THREADS = 256        # warps 0-7, the width of the segment's named bar
_SOFTMAX_BAR = 3              # D0.11's BAR.SYNC 0x3
_PAIR_MASK = 0b11             # both CTAs of the cluster-2 pair

# --- D0.12: the dQ reduction path ---------------------------------------------
# chain3's accumulator is split across the pair by M (MEASURED): CTA `rank` holds
# q rows [64*rank, +64) over ALL 128 d, as 128 TMEM lanes x 64 columns —
#     lane <  64 -> q = 64*rank + lane,      d = col
#     lane >= 64 -> q = 64*rank + lane - 64, d = col + 64
# which is exactly D0.3's (rank, lane, column) axis triple, so nothing has to be
# regrouped on the way out: this CTA's 8192 f32 are the eight 4 KB chunks
# [rank][quads 2c, 2c+1][128 lanes][4] of its own 32 KB rank half, chunk c sits
# at rank*32 KB + c*4 KB of the run, and the staging slot the bulk reduce reads
# is those same 4 KB byte for byte.
# The slot is written by ALL 128 threads, two 16 B stores each (lane t lands at
# quad*0x800 + t*16), which is D0.12's "每槽 128 线程全员各写 32 B" — the 16
# lanes x 256 B row-major form is §4's forbidden zone (16-way bank conflict,
# 112 idle lanes).
_WRITER_THREADS = 128         # warps 8-11 — also the TMEM lane count
_WRITER_BAR = 4               # D0.12's BAR.SYNC 0x4, 128 wide
_RING_SLOTS = 4               # the staging ring at 0x34800 is 4 slots
_CHUNK_QUADS = 2              # one slot / one bulk reduce = 2 column quads
_CHUNK_F32 = _CHUNK_QUADS * _ACC_QUAD_STRIDE      # 1024 f32 = 4 KB
_CHUNK_BYTES = 4 * _CHUNK_F32
_N_CHUNK = _ACC_QUADS // _CHUNK_QUADS   # 8 chunks = this CTA's 8192 f32 / trip
# --- D1.3: the ring's depth and in-flight tier --------------------------------
# D0.12's starting point is 4 slots with `<= 3` in flight (the reference's
# `DEPBAR.LE SB0, 0x3`); the candidates are {2 slots / <=1, 4 slots / <=3,
# 4 slots / <=2} and 8 slots is out by arithmetic (+16 KB over D0.8's 231,424 B
# against a 232,448 B ceiling). Slot count and in-flight depth are ONE lever:
# the slot is recycled by chunk c+4, and `cp.async.bulk.wait_group.read n` after
# chunk c's reduce is what certifies chunk c-n has READ its slot, so the pair
# (slots, n) is safe exactly when n <= slots - 1 — 4/<=2 retires chunk c-2 where
# 4/<=3 retires c-3, one reduce earlier, at the cost of one less reduce
# overlapping the next chunk's stores. This is the rung this tree takes: the
# writer warpgroup is a masked role (its `WAIT 0x90` spin is the largest
# MASKED stall, not an exposed one), so what the tier buys is the SLOT being
# free earlier for the store side, not a shorter reduce pipeline.
_RING_INFLIGHT = 2            # D1.3's "4 slots / <= 2" rung (was 3)

# --- D1.1: the MMA warp's wait-placement tier ---------------------------------
# D0.13 froze the masking RELATION (which tile each of the eight waits names) and
# left the DEPTH — how many chains are already in flight when a wait is taken —
# as this lever. The relation is the same under every tier: only the ORDER of the
# trip's last two blocks moves, and with it the in-flight depth at their waits.
#
#   "a"  chain3 dQ(i) then chain4 dV(i+1)  — D0.13's transcribed order, in-flight
#        depth vector 0,0,1,1,2,3,3,4: 0x60 / 0x160 are taken with three chains up
#        (S(i+1), dK(i), dP(i+1)) and 0x48 — the reference's largest exposure —
#        with four.
#   "b"  chain4 dV(i+1) then chain3 dQ(i) — 0x48 taken with three chains up and
#        0x60 / 0x160 with four; 0x68 (dS smem EMPTY) and 0x90 (dQ FULL) both flip
#        one chain later, so the segment's stage E and the writer start later too.
#
# Both tiers issue the same five chains against the same accumulators and the same
# barriers, so the numerics are identical and the two are a same-job A/B on device
# ms + tensor_pct. Candidate (c) of D1.1 (one more prologue warm-up beat, dV(0)
# pushed into the body's first trip) moves the PROLOGUE rather than the trip body
# and is not part of this knob.
_WAIT_TIER = "a"
_TIER_ORDER = (3, 4) if _WAIT_TIER == "a" else (4, 3)

# --- D2.1: the preprocess kernel's access shape -------------------------------
# Three jobs, all of them pure streaming, so the shape of every access is the
# whole decision: 270.6 MB read + 223.6 MB written per call is a bandwidth
# budget, and an atom narrower than 16 B pays for it in issue slots.
#   * o / dO -> dpsum. The granule is 8 bf16 = 16 B, a q row is 16 of them, and
#     the row's 16 granules are given to 16 CONSECUTIVE threads — so a warp's
#     instruction covers two whole rows as two 256 B runs, and the row sum is a
#     4-step butterfly inside a 16-lane segment instead of a per-thread walk of
#     the row. 256 threads x 16 rows per step x 8 steps = the block's 128 rows,
#     16 `LDG.E.128` and 32 `SHFL.BFLY` + 32 `FADD` per thread.
#   * dq_accum zeroing. The run is 64 KB per (b, h, m_block) and the zeroing is
#     blind to D0.3's block layout (it writes every byte), so it is 16
#     `STG.E.128` of a zero granule per thread, a warp covering 512 contiguous
#     bytes.
#   * lse -> lse_log2. One f32 per q row, so 128 of the 256 threads, and the
#     -inf guard is a predicate + `FSEL`, not a branch.
_PRE_GRAN = 8                                  # bf16 per 16 B granule
_PRE_ROW_GRAN = _D // _PRE_GRAN                # 16 granules per q row
_PRE_ROWS_STEP = _PRE_THREADS // _PRE_ROW_GRAN  # 16 rows per step
_PRE_STEPS = _BLK_M // _PRE_ROWS_STEP          # 8 steps
_PRE_BFLY = 4                                  # log2(16): the butterfly depth
_PRE_STEP_GRAN = _PRE_ROWS_STEP * _H * _PRE_ROW_GRAN   # granules between steps
_PRE_ZERO_STEPS = _ACC_TILE // _ACC_VEC // _PRE_THREADS  # 16 STG.E.128/thread

# --- D2.2: the postprocess kernel's layout transform --------------------------
# The read side is D0.3's block read with thread = lane (32 x 16 B per thread,
# stride 2 KB, 512 contiguous bytes per warp per instruction); the write side is
# the (B, N, H, D) row order. The two differ by a transpose between the lane
# axis (= q) and the quad axis (= d), so the tile goes through smem: 128 q rows
# x 16 granules of 8 bf16 = 32 KB, scaled and rounded before it is staged so the
# read-back is one 16 B granule per thread.
_POST_GRAN = 8                        # bf16 per 16 B granule
_POST_ROW_GRAN = _D // _POST_GRAN     # 16 granules per q row
_POST_ROWS_STEP = _POST_THREADS // _POST_ROW_GRAN     # 8 rows per store step
_POST_STEPS = _BLK_M // _POST_ROWS_STEP               # 16 steps
_SM_POST = _BLK_M * _D * 2            # 32,768 B of bf16 staging


# --- D0.7: warp roles (16 warps, dispatched by warp_idx = tid >> 5) -----------
_WARP_SOFTMAX = 0     # warps 0-7   softmax / dS element-wise + dK/dV epilogue
_WARP_WRITER = 8      # warps 8-11  dQ-writer (tmem -> smem ring -> bulk reduce)
_WARP_MMA = 12        # warp 12     MMA issue (only the leader CTA issues)
_WARP_TMA = 13        # warp 13     TMA producer
_WARP_RELAY = 14      # warp 14     dS exchange relay
_WARP_IDLE = 15       # warp 15     idle after barrier init

# --- D0.7 + D0.14 r3: the setmaxreg split ------------------------------------
# `setmaxnreg` is WARPGROUP granular and `.aligned` — all four warps of a
# warpgroup execute the same instruction with the same immediate — so the split
# can only be per warpgroup, not per warp: the reference's observed 136 / 136 /
# 104 / 104 / 24 / 104 has three different values inside warps 12-15 and is a
# per-warp READING, not a shape a single instruction can take. The warpgroup
# form of the same budget is
#     WG0, WG1 (softmax 0-7)  152      D0.14 r3: +16 over r0, margin 28 at peak 124
#     WG2      (writer  8-11) 104      D0.14 r3: cut to measured peak 68 + headroom
#     WG3      (12-15)        104      MMA 82 / TMA 6 / relay 2 / idle
# Budget: 8 x 152 + 4 x 104 + 4 x 104 = 1216 + 416 + 416 = 2048 = 65536/32, i.e.
# the whole register file at 1 CTA/SM (the reference spends 1968 of the same
# 2048). r0 (136/136/104) was the starting point; the segment already reads both
# statistics in place (r1) and takes its transient operands in two 32-column
# halves (r2, `_HALF_COLS`), and the spill counters on this tree are still not
# zero — so r3 re-cuts the writer WG from 136 to its measured peak 68 + headroom
# and gives the freed 32 per WG2 warp — 128 registers — to WG0 + WG1, which is
# +16 per softmax warp on the one role that spills.
_REG_SOFTMAX = 152    # warps 0-7   (D0.14 r3)
_REG_WRITER = 104     # warps 8-11  (D0.14 r3)
_REG_SERVICE = 104    # warps 12-15

# --- D0.6: TMEM column map (512 columns x 128 lanes, all of TMEM) -------------
# One CTA of the pair owns 128 lanes; a column is 32 bit, so a bf16 tile packs
# two values per column and covers half the columns of its f32 producer.
_TMEM_COLS = 512
_TMEM_LANES = 128
_TMEM_LANE_STRIDE = 65536    # lane stride of a TMEM tensor, in 32-bit columns

_TMEM_ST = 0x000      # S^T   f32  128 cols  chain0 D (zero-init each iter)
_TMEM_PT = 0x000      # P^T   bf16  64 cols  packed over S^T's first half
_TMEM_DQ = 0x040      # dQ    f32   64 cols  chain3 D, aliases S^T's 2nd half
_TMEM_DV = 0x080      # dV    f32  128 cols  chain4 D, accumulated across iters
_TMEM_DP = 0x100      # dP^T  f32  128 cols  chain2 D (zero-init each iter)
_TMEM_DST = 0x100     # dS^T  bf16  64 cols  packed over dP^T's first half
_TMEM_DK = 0x180      # dK    f32  128 cols  chain1 D, accumulated across iters

# --- D0.6: the MMA atom the five chains are issued on -------------------------
# One instruction is M256 x N128 x K16 bf16 -> f32 with the accumulator in TMEM,
# issued by the LEADER only and reading both CTAs' smem (`cta_group::2`): the
# pair's 256 rows are this CTA's 128 plus the peer's 128, and N=128 is likewise
# split 64/64, which is exactly how D0.8 slices the faces (this CTA's 128 KV
# rows as A, its 64 Q / dO rows as B). The chain's K extent is the head dim, so
# one chain = 128/16 = 8 instructions.
_MMA_K = 16
_MMA_INST = (2 * _BLK_N, _BLK_M, _MMA_K)      # (256, 128, 16)
_MMA_TILER = (2 * _BLK_N, _BLK_M, _D)         # k-tile = the whole head dim
# chain3 dQ is the one chain with a different shape: M = the q block's 128 rows
# (64 per CTA), N = the head dim (64 d columns per CTA), and a K extent of the
# PAIR's 256 KV rows — which is why its A operand needs the peer's dS half.
_MMA_INST3 = (_BLK_M, _D, _MMA_K)             # (128, 128, 16)


# --- D0.8: the 231,424 B dynamic smem arena -----------------------------------
# Byte sizes and order ARE D0.8's buffer list; the offsets below are the ones
# that list assigns, and the allocator reproduces them by construction (every
# region is a multiple of its successor's alignment).
_SM_CTRL = 1024               # 0x00000  28 mbarriers + tmem pointer slot
_SM_Q_ROW = 16384             # 0x00400  this CTA's 64 Q rows x 128 d
_SM_K = 32768                 # 0x04400  this CTA's 128 KV rows x 128 d
_SM_V = 32768                 # 0x0c400  this CTA's 128 KV rows x 128 d
_SM_DO_COL = 16384            # 0x14400  128 dO rows x this CTA's 64 d
_SM_Q_COL = 16384             # 0x18400  128 Q rows x this CTA's 64 d
_SM_DO_ROW = 16384            # 0x1c400  this CTA's 64 dO rows x 128 d
_SM_DS_SEND = 16384           # 0x20400  dS^T[128 KV][peer's 64 Q] send staging
_SM_K_DSPLIT = 32768          # 0x24400  the pair's 256 KV x this CTA's 64 d
_SM_DS = 32768                # 0x2c400  dS[64 Q][256 KV] = local half + peer
_SM_LSE = 512                 # 0x34400  128 f32, single stage
_SM_DPSUM = 512               # 0x34600  128 f32, single stage
_SM_DQ_RING = 16384           # 0x34800  4 slots x 4 KB, dQ reduction staging

_SMEM_BYTES = (_SM_CTRL + _SM_Q_ROW + _SM_K + _SM_V + _SM_DO_COL + _SM_Q_COL
               + _SM_DO_ROW + _SM_DS_SEND + _SM_K_DSPLIT + _SM_DS + _SM_LSE
               + _SM_DPSUM + _SM_DQ_RING)      # 231,424 B, cap is 232,448

# Control region: the mbarrier table at its D0.8 byte offsets, with the tmem
# pointer slot sitting at 0x120 between the two runs. Barrier index = byte
# offset / 8, which is how the D0.10 / D0.11 / D0.12 tables name them.
_N_BAR_LO = 20                # 0x000 .. 0x098
_N_BAR_HI = 8                 # 0x128 .. 0x160
_TMEM_SLOT_OFF = 0x120
_BAR_HI_OFF = 0x128

_BAR_Q_FULL = 0               # 0x00  Q_row FULL   (prologue also carries K)
_BAR_Q_EMPTY = 1              # 0x08  Q_row EMPTY
_BAR_DO_FULL = 2              # 0x10  dO FULL      (prologue also carries V)
_BAR_DO_EMPTY = 3             # 0x18  dO EMPTY
_BAR_LSE_FULL = 4             # 0x20  lse_log2 FULL
_BAR_LSE_EMPTY = 5            # 0x28  lse_log2 EMPTY (count 8, one per softmax)
_BAR_DPS_FULL = 6             # 0x30  dpsum FULL
_BAR_DPS_EMPTY = 7            # 0x38  dpsum EMPTY (count 8)
_BAR_S_FULL = 8               # 0x40  S^T FULL   (tmem; MMA commit -> softmax)
_BAR_P_RDY = 9                # 0x48  P^T READY  (count 16, both CTAs' softmax
                              #       warps -> the LEADER; MMA waits it)
_BAR_DP_FULL = 10             # 0x50  dP^T FULL  (tmem; MMA commit -> softmax)
_BAR_DS_RDY = 11              # 0x58  dS^T READY (count 16, same shape as 0x48)
_BAR_S_DONE = 12              # 0x60  S^T CONSUMED (count 16): tmem 0x40-0x7f
                              #       may be reused by chain3's dQ
_BAR_DS_EMPTY = 13            # 0x68  dS smem EMPTY (chain3's multicast commit
                              #       -> both CTAs' softmax warps)
_BAR_DV_FINAL = 14            # 0x70  dV accumulator FINAL (MMA commit -> epi)
_BAR_DK_FINAL = 15            # 0x78  dK accumulator FINAL (MMA commit -> epi)
_BAR_DV_EMPTY = 16            # 0x80  dV epilogue staging EMPTY (step 6)
_BAR_DK_EMPTY = 17            # 0x88  dK epilogue staging EMPTY (step 6)
_BAR_DQ_FULL = 18             # 0x90  dQ accumulator FULL (chain3's MULTICAST
                              #       commit: each CTA reads its own TMEM half,
                              #       so a leader-only arrive hangs the follower)
_BAR_DQ_EMPTY = 19            # 0x98  tmem 0x40-0x7f drained (count 8 = 4 writer
                              #       warps x 2 CTAs -> the leader; MMA waits it)
_BAR2_DEALLOC = 0             # 0x128 tmem dealloc rendezvous (count 32)
_BAR2_QC_FULL = 1             # 0x130 Q_col FULL
_BAR2_QC_EMPTY = 2            # 0x138 Q_col EMPTY
_BAR2_KD_FULL = 3             # 0x140 K_dsplit FULL (prologue only)
_BAR2_KD_EMPTY = 4            # 0x148 K_dsplit EMPTY (loop tail)
_BAR2_DS_PEER = 6             # 0x158 peer's dS half landed (remote expect-tx)
_BAR2_DS_DONE = 7             # 0x160 dS exchanged (count 2, relay -> leader)


# --- D0.8 / D0.10: the seven TMA operand faces --------------------------------
# A tcgen05 smem operand tile is shaped (MN, K), and which axis is contiguous in
# gmem picks the gmem view the descriptor is built on:
#   K-major  (Q_row / dO_row / K / V): MN = rows, K = d -> view (N, D, H, B),
#       tiler (rows, 128).
#   MN-major (Q_col / dO_col / K_dsplit): MN = d, K = rows. d is STILL the
#       contiguous axis (it is mode 0 of an MN-major tile) -> view (D, N, H, B),
#       tiler (64, rows). A (rows, 64) tiler on the K view would ask TMA for a
#       transpose of a d-contiguous tensor, which no box does.
# (major, mn, k) per face; `mn` is mode 0 of the smem tile, `k` is mode 1.
_FACE = {
    "q_row":    ("K",  64, _D),
    "do_row":   ("K",  64, _D),
    "k":        ("K",  _BLK_N, _D),
    "v":        ("K",  _BLK_N, _D),
    "q_col":    ("MN", _D_HALF, _BLK_M),
    "do_col":   ("MN", _D_HALF, _BLK_M),
    "k_dsplit": ("MN", _D_HALF, 2 * _BLK_N),
    # The two dS faces are the same kind of tile but have NO TMA descriptor:
    # they are written by the softmax warps' stores and by the cluster bulk
    # copy, and read by chain3's UMMA descriptor (D0.9's one smem row). MN-major
    # with mn = the 64 Q rows one CTA owns, so one lane's 64 Q values at a fixed
    # KV row are one contiguous 128 B swizzle row.
    "ds":       ("MN", 64, 2 * _BLK_N),   # 0x2c400 dS[my 64 Q][the pair's 256]
    "ds_send":  ("MN", 64, _BLK_N),       # 0x20400 dS[peer's 64 Q][my 128 KV]
    # D0.15's epilogue staging / TMA-store box: this CTA's 128 KV rows x the 64
    # d columns ONE warpgroup owns. K-major, so a lane's 64 bf16 at its own KV
    # row are one 128 B swizzle row — the same shape as half of the K / V face,
    # which is exactly the region it is aliased onto (D0.15 reading (a): K / V
    # have no reader left once the last S^T / dP^T chain has retired, and the
    # smem budget is full, so the epilogue opens no buffer of its own).
    "dkv":      ("K", _BLK_N, _D_HALF),   # 16 KB; two of them tile one face
}
_FACE_ORDER = (0, 1)          # d-tiles outer for K-major; the canonical fit

# --- D1.4: the dS exchange face's swizzle mode --------------------------------
# The lever is the MODE only, at a FIXED atom on both sides of the face: the
# write side stays the softmax segment's 128 B row store and the read side stays
# chain3's MN-major UMMA descriptor, because "只钉 swizzle 不钉 atom 无效" — a
# scalarised path costs multiples of the wavefronts a mode ever moves. Candidates
# {SW128, SW64, SW32}; the mode rides the SAME `_face_layout` the descriptor and
# the allocation are both built from, so descriptor swizzle bits, arena offsets
# and store addressing move together by construction and cannot disagree.
# Scope: the two dS faces ("ds" at 0x2c400 and "ds_send" at 0x20400). Every
# other face is a TMA operand whose box was frozen in D0.8 and is not this
# lever's subject.
# Value: SW128 — D0.8's starting point. The switch to a narrower mode is
# D1.4's own trigger (`bankconf_st > 2x` the reference's 2,544,993 while
# `smem_wavefronts_st` stays in band = the CONFLICT branch of §5), and the
# store side's conflict counter is inside that band on this tree, so the
# starting point stands and the two narrower modes are not taken.
_DS_SWZ = "SW128"
_DS_FACES = ("ds", "ds_send")


def _swz_kind(major, mode):
    """The layout atom kind for (major, swizzle mode), by the DSL's own name."""
    nm = "%s_%s" % ("MN" if major == "MN" else "K", mode)
    kind = getattr(tcgen05.SmemLayoutAtomKind, nm, None)
    if kind is None:
        raise RuntimeError("no smem layout atom kind %r on the installed DSL" % nm)
    return kind


# expect-tx bytes, per pair, derived from the face sizes (never hand-written:
# one hand-written byte count diverges from the layout when a tile changes, and
# divergence shows up as a hang or as a barrier that flips early).
_TX_Q = 2 * _SM_Q_ROW                              # 0x08000
_TX_DO = 2 * (_SM_DO_ROW + _SM_DO_COL)             # 0x10000
_TX_QC = 2 * _SM_Q_COL                             # 0x08000
_TX_KD = 2 * _SM_K_DSPLIT                          # 0x10000
_TX_PRO_Q = 2 * (_SM_K + _SM_Q_ROW)                # 0x18000 (K + Q_row(m_min))
_TX_PRO_DO = 2 * (_SM_V + _SM_DO_ROW + _SM_DO_COL)  # 0x20000 (V + dO(m_min))
_TX_STAT = 4 * _BLK_M                              # 512 B, per CTA


def _seg(site, row=0):
    """Timeline-probe hook. The driver rewrites this body on the probe arm;
    on the shipped tree it is a no-op that costs nothing."""
    return None


def _f32(x):
    return x.to(cutlass.Float32)


def _bf16(x):
    return x.to(cutlass.BFloat16)


def _addr(p):
    """The integer address of a tensor / smem array, for inline-PTX operands."""
    fn = getattr(p, "toint", None)
    if fn is not None:
        return fn()
    return p.iterator.toint()


def _face_layout(tag):
    """The swizzled smem layout of one face, BUILT IN THE CALLER'S REGION.

    Call this once per region that needs the layout (host, for the descriptor;
    kernel, for `allocate_tensor`) rather than building it once and passing it
    across: `tile_to_shape` on a swizzled atom materialises a composed layout as
    an SSA value of the region that traced it, and reading its swizzle back from
    another region is a `composed_get_inner ... defined outside the region` ICE.
    The arguments are Python constants, so the two builds are the same layout
    and descriptor and allocation agree.
    """
    major, mn, k = _FACE[tag]
    # D1.4: the two dS faces take the lever's mode, every other face the frozen
    # SW128 of D0.8's buffer list.
    kind = _swz_kind(major, _DS_SWZ if tag in _DS_FACES else "SW128")
    atom = tcgen05.make_smem_layout_atom(kind, cutlass.BFloat16)
    return cute.tile_to_shape(atom, (mn, k), order=_FACE_ORDER)


def _alloc_face(smem, tag, align=1024):
    """DEVICE side: allocate one face, with the composed layout SPLIT.

    `_face_layout` is `swizzle o offset o layout`. Handing the whole thing to
    `allocate_tensor` builds a position-independent swizzle layout, and the
    tcgen05 verifier refuses a composed layout for an A/B operand — the UMMA
    descriptor carries the swizzle as a field of its own and wants it on the
    ITERATOR. Splitting it here is that form, and it is byte-identical: the
    address function, the arena offsets and the host-side TMA descriptors are
    all exactly what step 2b delivered against.
    """
    lay = _face_layout(tag)
    return smem.allocate_tensor(cutlass.BFloat16, lay.outer,
                                byte_alignment=align, swizzle=lay.inner)


def _mma_view(s, lay):
    """The same bytes, regrouped into the profile `make_fragment_A/B` reads.

    An operand face is `((MN_cta, K_inst), MMA_MN, MMA_K, STAGE)`; D0.8's face
    is the same address function grouped as (mn, k), and `make_fragment_A` does
    not re-tile — it reads mode 0 AS the instruction's value mode. So the face
    has to arrive already in that grouping. The swizzle rides on `s.iterator`
    (see `_alloc_face`), so re-viewing keeps it.
    """
    return cute.make_tensor(s.iterator, lay.outer)


def _chain3_layouts(tiled, want):
    """DEVICE side (trace time): chain3's A / B operand layouts and how many
    passes one trip takes over its K = 256.

    D0.6 reads the reference's dQ chain as "每迭代 2 趟 × 8 步" — two K=128
    passes — but flags it as an inference. Both readings are the same product
    over the same 32 KB faces, and which one CUTLASS's layout builder agrees
    with is a property of the installed library, so it is settled by asking:
    the one-pass K=256 form is taken when its A layout covers the whole face
    (`want` = the face's cosize), and D0.6's two-pass form otherwise. The pass
    offset is a whole number of swizzle periods, so the second pass's address
    function is the first's, shifted.
    """
    for ktile in (2 * _BLK_N, _BLK_N):
        n_pass = (2 * _BLK_N) // ktile
        tiler = (_MMA_INST3[0], _MMA_INST3[1], ktile)
        try:
            la = bh.make_smem_layout_a(tiled, tiler, cutlass.BFloat16, 1)
            lb = bh.make_smem_layout_b(tiled, tiler, cutlass.BFloat16, 1)
        except Exception:
            continue
        if cute.cosize(la.outer) * n_pass == want:
            return la, lb, n_pass
    raise RuntimeError("chain3: no A layout covers the dS face (cosize %d)"
                       % want)


def _resolve_commit():
    """Pick the `tcgen05.commit` call form from the INSTALLED signature, once,
    at import time. Resolved here rather than in the kernel body so that the
    callable the kernel invokes carries no control flow of its own (the DSL
    preprocessor only rewrites decorated sources)."""
    import inspect
    commit = tcgen05.commit
    try:
        sig = inspect.signature(commit)
        params = [(n, p) for n, p in sig.parameters.items()
                  if n not in ("loc", "ip")]
    except (TypeError, ValueError):
        params = []
    var_only = all(p.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD)
                   for _, p in params)
    if not params or var_only:
        # a decorator ate the signature: nothing can be read off it, so take
        # the plain form and let the op's own default pick the cta group.
        def _commit(bar, cg):
            tcgen05.commit(bar)
        return _commit
    by_name = dict(params)
    # any parameter other than the barrier and the cta group that is required
    # (carries no default) has to be supplied; the only one in this family is
    # the cta mask, and the pair's mask is both CTAs.
    extra = dict((n, 0b11) for n, p in params[1:]
                 if "mask" in n and p.default is inspect.Parameter.empty)
    cg_param = by_name.get("cta_group")
    if cg_param is None:
        def _commit(bar, cg, _extra=extra):
            tcgen05.commit(bar, **_extra)
    elif cg_param.kind is inspect.Parameter.POSITIONAL_ONLY:
        def _commit(bar, cg, _extra=extra):
            tcgen05.commit(bar, *list(_extra.values()), cg)
    else:
        def _commit(bar, cg, _extra=extra):
            tcgen05.commit(bar, cta_group=cg, **_extra)
    return _commit


_COMMIT = _resolve_commit()


@dsl_user_op
def _commit_mcast_ptx(mbar_addr, *, loc=None, ip=None):
    """`tcgen05.commit` in its MULTICAST form: one arrive into the mbarrier at
    this offset in EVERY CTA of the pair (hw_kb `ptx_escape`, tier B).

    The mask is built inside the asm block so the call needs no 16-bit operand
    type (`.b16` is the ctamask's width in the ISA). The caller elects one
    thread: tcgen05 commit is on hw_kb `elect_one`'s must-elect list.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 m;\n\tmov.u16 m, %d;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster"
        ".multicast::cluster.b64 [$0], m;\n}" % _PAIR_MASK,
        "r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _resolve_commit_mcast():
    """The pair-multicast arrive, in whichever spelling the installed DSL has.

    D0.11's 0x40 / 0x50 rows are waited by the softmax warps of BOTH CTAs — each
    CTA's 128 TMEM lanes are its own 128 KV rows, and only that CTA can LDTM
    them — while the MMA is issued by the leader alone. The plain `commit` only
    arrives the ISSUING CTA's mbarrier (hw_kb `cta_pair_consumer_peek`), so the
    follower would wait a phase nothing ever flips: a hang, not wrong numbers.
    Both arms below are the same instruction; the order is the probe ladder's
    (H-dsl before H-ptx).
    """
    import inspect
    try:
        params = [(n, p) for n, p in
                  inspect.signature(tcgen05.commit).parameters.items()
                  if n not in ("loc", "ip")]
    except (TypeError, ValueError):
        params = []
    names = [n for n, _ in params]
    kw = dict((n, _PAIR_MASK) for n, _ in params[1:] if "mask" in n)
    if kw:
        if "cta_group" in names:
            kw["cta_group"] = tcgen05.CtaGroup.TWO

        def _commit(bar, _kw=kw):
            tcgen05.commit(bar, **_kw)
        return _commit

    def _commit(bar):
        _commit_mcast_ptx(_addr(bar))
    return _commit


_COMMIT_MCAST = _resolve_commit_mcast()


def _resolve_exp2():
    """`exp2` as the installed DSL spells it (D0.4: the main loop's only
    transcendental, one `MUFU.EX2` per element)."""
    import inspect
    for holder, nm in ((getattr(cute, "math", None), "exp2"),
                       (cute.arch, "exp2"), (cute.arch, "ex2")):
        if holder is None:
            continue
        f = getattr(holder, nm, None)
        if f is None:
            continue
        try:
            has_fm = "fastmath" in inspect.signature(f).parameters
        except (TypeError, ValueError):
            has_fm = False
        if has_fm:
            return lambda x, _f=f: _f(x, fastmath=True)
        return lambda x, _f=f: _f(x)
    raise RuntimeError("no exp2 spelling found on the installed DSL")


_EXP2 = _resolve_exp2()


# --- D1.5: producer lead and tensormap prefetch -------------------------------
# The producer's lead itself is NOT a lever here: D0.8's arena is full, so every
# face is single stage (D0.10, `rings = []`), which caps the lead at one tile for
# Q_col / lse / dpsum and two for Q_row / dO — and the statistics' segment is
# pinned to the head of the body by D0.10 form A, which is an M0 contract and
# not in this decision's candidate domain. What IS this decision's is the cold
# cost of the descriptors: nine tensormaps are read out of gmem the first time
# each is used, and the first use of most of them is the prologue's own issue,
# where nothing is in flight to cover the miss.
#   `UTMACCTL.PF` = one prefetch per descriptor, issued by the TMA warp before
#   its first expect-tx, so the nine faults are taken as one burst that overlaps
#   the prologue's cluster rendezvous instead of one at a time in front of each
#   copy. No smem, no barrier, no register: the only cost is the instructions.
# Candidates {9 (all), 5 (the body's four faces + K_dsplit, i.e. the ones a trip
# touches), 0 (no prefetch)}, taken as a COUNT over `_PREFETCH_ORDER` below,
# which is ordered body-faces first exactly so the rungs nest.
# Value: 9 — the reference's own prologue prefetches nine.
_PREFETCH_TMAPS = 9
# body faces (4) -> the trip-resident K_dsplit -> prologue-only K / V -> the two
# epilogue store descriptors.
_PREFETCH_ORDER = ("q_row", "do_row", "q_col", "do_col", "k_dsplit",
                   "k", "v", "store_dk", "store_dv")


def _resolve_prefetch_desc():
    """`prefetch.tensormap` as the installed DSL spells it (hw_kb
    `tcgen05_umma_issue_cta_pair`: `cpasync.prefetch_descriptor(atom)`)."""
    for holder, nm in ((cpasync, "prefetch_descriptor"),
                       (cute, "prefetch_descriptor"),
                       (cute.arch, "prefetch_descriptor")):
        f = getattr(holder, nm, None)
        if f is not None:
            return lambda atom, _f=f: _f(atom)
    f = getattr(cute.arch, "prefetch", None)
    if f is not None:
        return lambda atom, _f=f: _f(atom, tensormap=True)
    raise RuntimeError(
        "no tensormap-prefetch spelling found on the installed DSL "
        "(cpasync.prefetch_descriptor / cute.arch.prefetch(tensormap=True))")


_PREFETCH_DESC = _resolve_prefetch_desc()


def _resolve_named_bar():
    """`bar.sync <id>, <n>` over a subset of the CTA — the segment's rendezvous
    is 256 threads wide (warps 0-7), not the whole block.

    There is no fallback on purpose: `sync_threads()` issued by 256 of the 512
    threads is a hang, and a hang is the one failure the milestone's triage
    cannot localise. If the spelling ever moves, fail loudly at trace time.
    """
    import inspect
    f = getattr(cute.arch, "barrier", None)
    if f is not None:
        try:
            ps = list(inspect.signature(f).parameters)
        except (TypeError, ValueError):
            ps = []
        if "barrier_id" in ps and "number_of_threads" in ps:
            return lambda i, n: f(barrier_id=i, number_of_threads=n)
    raise RuntimeError(
        "no width-carrying named barrier on the installed DSL "
        "(cute.arch.barrier(barrier_id=, number_of_threads=))")


_NAMED_BAR = _resolve_named_bar()


def _resolve_proxy_fence():
    """The smem async-proxy fence: what makes a generic `STS` visible to the
    UMMA descriptor read and to the bulk-copy engine.

    Without it the dS face is a race that happens to pass on a quiet machine —
    the softmax warps write it with ordinary stores and both of its readers go
    through the async proxy.
    """
    import inspect
    arch = cute.arch
    f = getattr(arch, "fence_proxy", None)
    if f is not None:
        PK = getattr(arch, "ProxyKind", None)
        kind = None
        if PK is not None:
            for nm in ("async_shared", "async_", "async"):
                kind = getattr(PK, nm, None)
                if kind is not None:
                    break
        try:
            params = list(inspect.signature(f).parameters)
        except (TypeError, ValueError):
            params = []
        SS = getattr(arch, "SharedSpace", None)
        space = getattr(SS, "shared_cta", None) if SS is not None else None
        if kind is not None and "space" in params and space is not None:
            return lambda: f(kind, space=space)
        if kind is not None:
            return lambda: f(kind)
    for nm in ("fence_view_async_shared", "fence_proxy_async_shared",
               "fence_async_shared"):
        g = getattr(arch, nm, None)
        if g is not None:
            return g
    raise RuntimeError("no smem async-proxy fence on the installed DSL")


_PROXY_FENCE = _resolve_proxy_fence()


def _resolve_vec_copy():
    """rmem -> smem, VECTORIZED: one lane's 64 bf16 are one 128 B swizzle row of
    a dS face (8 x `STS.128`), and one writer lane's 64 f32 are the 256
    contiguous bytes of a ring row (16 x `STS.128`).

    There is no scalar fallback on purpose — D0.8's atom line ("atom 是断崖不是
    旋钮"): a scalar `STS` path multiplies `smem_wavefronts_st` and would be a
    silent downgrade of the buffer list's own contract rather than a knob.
    """
    f = getattr(cute, "autovec_copy", None)
    if f is not None:
        return f
    nv = getattr(cute, "nvgpu", None)
    for holder in (nv, getattr(nv, "common", None)):
        op = getattr(holder, "CopyUniversalOp", None) if holder else None
        if op is None:
            continue

        # the atom is built inside the caller, i.e. inside the traced region:
        # `make_copy_atom` emits IR and has no MLIR context of its own here.
        # Its element type is the source's, so the one resolver serves the bf16
        # dS row and the f32 ring row alike.
        def _cp(s, d, _op=op):
            cute.copy(cute.make_copy_atom(_op(), s.element_type,
                                          num_bits_per_copy=128), s, d)
        return _cp
    raise RuntimeError("no vectorizing rmem->smem copy on the installed DSL")


_VEC_COPY = _resolve_vec_copy()


def _resolve_copy128():
    """A copy whose width is PINNED at 128 bit: the ring / staging stores and the
    postprocess kernel's gmem load and gmem store.

    A width inferred from the operand is the element's, not the granule's — the
    same 16 B of traffic issued as four `LDG.E` instead of one `LDG.E.128`
    (D0.8's atom line, read on the postprocess side: D2.2's census is 32
    `LDG.E.128` per thread, and the addresses are 16 B aligned by construction —
    the quad IS four f32). Falls back to the autovec spelling when the universal
    op is not on this DSL build.

    The atom is only half of it; the other half is how the operand is SPELLED.
    `make_tensor(t.iterator + <dynamic off>, layout(4))` drops the declared
    alignment to the element's 4 B and the verifier refuses the atom outright
    (`S ptr alignment (32 bits) does not meet requirement (128 bits)`); the
    spelling that carries the granule's alignment is a (numel/n, n) view built
    ONCE from the UNOFFSET iterator and then sliced — see `_gvec`.
    """
    nv = getattr(cute, "nvgpu", None)
    for holder in (nv, getattr(nv, "common", None) if nv is not None else None):
        op = getattr(holder, "CopyUniversalOp", None) if holder else None
        if op is None:
            continue

        def _cp(s, d, _op=op):
            cute.copy(cute.make_copy_atom(_op(), s.element_type,
                                          num_bits_per_copy=128), s, d)
        return _cp
    return _VEC_COPY


_COPY128 = _resolve_copy128()


def _acc_tile_base(b, h, m_blk):
    """D0.3: the first f32 of the (b, h, m_block) run of dq_accum.

    One expression, used by the producer (preprocess zeroing), the accumulator
    (the dQ-writer's reduce) and the consumer (postprocess) alike — three
    call-sites that disagree is dq wrong, not dq slow.
    """
    return ((b * _H + h) * _NBLK + m_blk) * _ACC_TILE


def _acc_chunk_off(rank, c):
    """D0.3 / D0.12: chunk `c` of pair rank `rank`, relative to the run's base.

    `base + rank*32 KB + chunk*4 KB` — the chunk is two whole column quads, so
    it is a CONTIGUOUS 4 KB of the block and the bulk reduce does no shuffling.
    """
    return rank * _ACC_RANK_STRIDE + c * _CHUNK_F32


def _gvec_view(t, numel, n):
    """A gmem tensor re-viewed as (numel / n, n) — `n` contiguous elements per
    row — built from the UNOFFSET iterator, so every row of it is `n`-element
    aligned by construction. Taken ONCE, outside the loops that slice it."""
    return cute.make_tensor(t.iterator,
                            cute.make_layout((numel // n, n), stride=(n, 1)))


def _gvec(flat, v):
    """DEVICE side: granule `v` of a `_gvec_view`, as a copy operand. This is the
    spelling `_COPY128` needs on gmem (an offset iterator is not)."""
    return flat[(v, None)]


def _ring_layout():
    """The dQ staging ring, in the shape it is used through: 4 slots of
    [2 column quads][128 lanes][4 f32]. One slot IS one workspace chunk."""
    return cute.make_layout(
        (_RING_SLOTS, _CHUNK_QUADS, _ACC_LANES, _ACC_VEC),
        stride=(_CHUNK_F32, _ACC_QUAD_STRIDE, _ACC_VEC, 1))


def _ring_store_chunk(sRing, slot, lane, line, views, staging, c):
    """DEVICE side: this thread's 32 B of chunk `c`, into ring slot `slot`.

    Two 16 B stores at `slot + quad*0x800 + lane*16`, one per quad of the
    chunk — all 128 threads live, a warp's 32 lanes covering 512 contiguous
    bytes, so the store is bank-conflict free and the slot is filled by exactly
    two instructions per thread (D0.12's `4 warp x 2 条 x 8 槽 = 64 STS.128`).
    `views[q]` is the 4-register window of the accumulator line at quad `q` when
    this DSL build will offset register storage, and `staging` is the fallback
    copy for when it will not.
    """
    # plain `range` / plain `if`: this helper is not a rewritten body, it is
    # ordinary Python running at trace time (the `_issue` convention).
    for hq in range(_CHUNK_QUADS):
        qd = c * _CHUNK_QUADS + hq
        dst = sRing[(slot, hq, lane, None)]
        if views[qd] is not None:
            _COPY128(views[qd], dst)
        else:
            for j in range(_ACC_VEC):
                staging[j] = line[_ACC_VEC * qd + j]
            _COPY128(staging, dst)


def _post_swz(dg, q):
    """DEVICE side: the PHYSICAL granule index of the logical granule `dg` in row
    `q` of D2.2's staging tile.

    The tile is 128 q rows x 16 granules of 8 bf16, and the transform's two
    sides walk it along different axes: the write side has the 8 lanes of a
    phase on 8 DIFFERENT q rows at the SAME granule, the read side has 16 lanes
    on the SAME row at 16 different granules. An unswizzled 256 B row makes the
    first of those an 8-way bank conflict (all 8 lanes land on the same four
    banks). The rotation `(dg + q) % 16` is a bijection within each row, so both
    phases cover a full 256 B row and neither repeats a bank; padding cannot do
    the same job here, because a pad that fixes the write side breaks the read
    side's 16 B alignment. A rotation rather than an XOR because `+` and `%` are
    the two operators every build has on a dynamic Int32.
    """
    return (dg + q) % _POST_ROW_GRAN


def _post_tile_layout():
    """D2.2's staging tile as it is used through: 128 q rows x 16 granules x 8
    bf16, so one slice IS one 16 B granule and the swizzle is an index."""
    return cute.make_layout((_BLK_M, _POST_ROW_GRAN, _POST_GRAN),
                            stride=(_D, _POST_GRAN, 1))


def _rmem_line(n, dtype):
    """A flat register array of `n` elements (the 64 bf16 of one swizzle row).
    DEVICE side; the two spellings differ only in how the extent is given."""
    try:
        return cute.make_rmem_tensor((n,), dtype)
    except Exception:
        return cute.make_rmem_tensor(cute.make_layout(n), dtype)


@dsl_user_op
def _mapa(addr, cta, *, loc=None, ip=None) -> cutlass.Int32:
    """`mapa.shared::cluster`: my shared address, as seen by CTA `cta` of the
    cluster. This is `map_dsmem_ptr` spelled so that one call serves an mbarrier
    and a swizzled operand face alike (the DSL helper wants a typed pointer;
    the bulk copy wants a plain 32-bit address)."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(cta).ir_value(loc=loc, ip=ip)],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _expect_tx_at(mbar_addr, tx, *, loc=None, ip=None):
    """`mbarrier.arrive.expect_tx` on an address that is REMOTE: D0.11's 0x158
    row is armed by the SENDER on the receiver's barrier. One instruction sets
    both the arrive and the transaction count, which is why that row's count is
    1 while its tx is the copy's whole 16 KB."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(tx).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [$0], $1;",
        "r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_s2s(dst_addr, src_addr, nbytes, mbar_addr, *, loc=None, ip=None):
    """`UBLKCP.S.S`: one bulk copy from my smem into the PEER's, completing on
    the peer's mbarrier. Per-thread instruction with no internal election (hw_kb
    `elect_one`), so the caller elects exactly one thread."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.shared::cta"
        ".mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_red_add_f32(dst_gmem, src_smem, nbytes, *, loc=None, ip=None):
    """`UBLKRED.G.S.ADD.F32.RN`: D0.12's reduction, performed at the L2 slice.

    D0.12's `[EXPR:]` ladder settles on this tier-B escape: the DSL's cpasync
    surface carries only the TENSOR reduce (`CopyReduceBulkTensorTileS2GOp`,
    which lowers to `UTMAREDG` — the备胎, not this), and D0.3's blocked
    workspace is addressed as flat bytes rather than as a tile of a tensor.
    Per-thread instruction with no internal election (hw_kb `elect_one`), so the
    caller elects exactly one thread of the writer warpgroup; the source must be
    `.shared::cta`, and an ordinary `STS` into it is invisible to the async
    proxy until fenced.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_gmem).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_smem).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
        " [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_commit_ptx(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_read_ptx(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.wait_group.read %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_all_ptx(n, *, loc=None, ip=None):
    """`cp.async.bulk.wait_group N` WITHOUT `.read`: waits for the copies to
    have LANDED, not merely to have released their source.

    The ring's discipline only needs `.read` (D0.12), but an epilogue TMA store
    is the last thing the CTA does — there is nobody left to be ordered against,
    so what has to be true before the CTA retires is that the bytes are in gmem.
    """
    llvm.inline_asm(None, [], "cp.async.bulk.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _setmaxnreg_ptx(action, n, *, loc=None, ip=None):
    """`setmaxnreg.{inc,dec}.sync.aligned.u32 <imm>` — D0.7's register split.

    The count is an ISA IMMEDIATE (24..256, a multiple of 8), so it is baked
    into the asm text rather than passed as an operand. Executed by all four
    warps of a warpgroup under a warpgroup-uniform predicate.
    """
    llvm.inline_asm(None, [],
                    "setmaxnreg.%s.sync.aligned.u32 %d;" % (action, int(n)), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _resolve_setmaxreg():
    """The warpgroup register realloc pair, in whichever spelling the installed
    DSL carries; the PTX escape (hw_kb `ptx_escape`, tier B) is the fallback.

    Both directions exist because ptxas hands the kernel ONE static allocation
    (with 512 threads and the arena forcing 1 CTA/SM that is 65536/512 = 128):
    the warpgroups that want more than it issue `.inc`, the one that wants less
    issues `.dec`, and the CTA's total stays inside the register file.
    """
    arch = cute.arch
    inc = None
    dec = None
    for nm in ("warpgroup_reg_alloc", "setmaxnreg_alloc", "alloc_reg"):
        f = getattr(arch, nm, None)
        if f is not None:
            inc = f
            break
    for nm in ("warpgroup_reg_dealloc", "setmaxnreg_dealloc", "dealloc_reg"):
        f = getattr(arch, nm, None)
        if f is not None:
            dec = f
            break
    if inc is not None and dec is not None:
        return inc, dec
    return ((lambda n: _setmaxnreg_ptx("inc", n)),
            (lambda n: _setmaxnreg_ptx("dec", n)))


_REG_INC, _REG_DEC = _resolve_setmaxreg()


# --- D1.6: TMA-side mbarrier wait encoding (form e / e') ----------------------
# D1.6's candidate domain is {suspending (the starting point) / non-blocking
# try-wait + explicit spin}, chosen PER ROLE. This lever takes the TMA producer
# warp's EMPTY waits — the 5 in the body and the 2 in the prologue's second
# beat. That warp owns 6 registers and issues nothing but addresses and copies,
# so keeping it resident between attempts costs an issue slot the roles it feeds
# are not competing for, while every suspend it avoids is a wake-up latency paid
# on the supply's critical path once per face per trip (D0.10: the producer's
# whole job is to stay ahead, and `lse EMPTY` is the reference's most-waited
# barrier at 308/CTA).
# Every OTHER wait keeps the suspending encoding: the MMA warp's operand FULLs
# and the two handoff barriers, the segment's four, the writer's and the relay's
# — those roles hold 82 / 152 / 104 registers and share the SM with 15 other
# warps, which is exactly the case hw_kb `mbarrier_wait_encoding` gotcha 6 says
# spinning loses. arrive / init counts are untouched (gotcha 8).
# Form (e): a `@cute.jit` helper retrying `mbarrier_try_wait` in an explicit
# while loop (the loop has to sit in a traced scope — gotcha 5), with form (e')
# — the inline-PTX twin of the same instruction — for a build whose arch face
# does not export it.
_arch_try_wait = getattr(cute.arch, "mbarrier_try_wait", None)

if _arch_try_wait is not None:
    @cute.jit
    def _tma_empty_wait(bar, phase):
        ok = _arch_try_wait(bar, phase)
        while ok == 0:
            ok = _arch_try_wait(bar, phase)
else:
    @dsl_user_op
    def _mbar_try_wait_i32(mbar_addr, phase, *, loc=None, ip=None):
        """One parity attempt -> 1 satisfied / 0 not. `has_side_effects` AND the
        memory clobber are both required: a pure asm with a return value can be
        hoisted out of the `while`, and the symptom of that is an endless spin,
        not a wrong number."""
        return cutlass.Int32(llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
             cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.b32 $0, 1, 0, p;\n}",
            "=r,r,r,~{memory}",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @cute.jit
    def _tma_empty_wait(bar, phase):
        addr = bar.toint()
        ok = _mbar_try_wait_i32(addr, phase)
        while ok == 0:
            ok = _mbar_try_wait_i32(addr, phase)


def _resolve_bulk_group():
    """`cp.async.bulk.commit_group` / `.wait_group.read N`, in whichever
    spelling the installed DSL carries.

    `.read` is the whole of D0.12's ring discipline: it retires a group once its
    SOURCE has been read, which is what says slot (c mod 4) may be refilled by
    chunk c+4. Waiting on full completion instead would serialise the writer
    against L2."""
    import inspect
    c = getattr(cute.arch, "cp_async_bulk_commit_group", None)
    w = getattr(cute.arch, "cp_async_bulk_wait_group", None)
    if c is not None and w is not None:
        try:
            ps = list(inspect.signature(w).parameters)
        except (TypeError, ValueError):
            ps = []
        if "read" in ps:
            return c, (lambda n: w(n, read=True)), (lambda n: w(n, read=False))
        return c, (lambda n: w(n)), (lambda n: w(n))
    return _bulk_commit_ptx, _bulk_wait_read_ptx, _bulk_wait_all_ptx


_BULK_COMMIT, _BULK_WAIT, _BULK_WAIT_ALL = _resolve_bulk_group()


def _slice_in_order(id_t, ix, n, what):
    """TRACE TIME: element j of this fragment is column j of its slice.

    D0.14's whole economy rests on it. If the value order IS the column order
    then one flat array indexed by the Q COLUMN serves every consumer of the
    segment's output at once — the exp2 result that has to stay fp32 from
    segment B to segment D, the bf16 packing the STTM writes, and the 128 B
    swizzle row the smem store writes — and the segment carries ONE staging
    instead of one per consumer. If it were a permutation, each consumer would
    need its own gather (+32 to +64 registers, i.e. the spill D0.14 forbids).

    There is no fallback on purpose (the `_resolve_named_bar` rule): a silently
    reordered store writes plausible numbers at the wrong Q columns, which is a
    correctness bug with no mechanical symptom. Fail loudly at trace time.
    """
    crd0 = id_t[ix(0)]
    ok = isinstance(crd0[1], int)
    if ok:
        ok = all(isinstance(id_t[ix(j)][1], int)
                 and int(id_t[ix(j)][1]) == j for j in range(n))
    if not ok:
        raise RuntimeError(
            "%s: the tcgen05 copy fragment's value order is not the column "
            "order on this DSL build (first coord %s)" % (what, str(crd0)))
    return True


def _one_mask_word(n, what):
    """TRACE TIME: a transient slice is exactly one 32-bit mask word wide.

    D1.2's causal predicate is a THRESHOLD BITMAP — one register per slice, bit c
    testing column c — so it closes only while a slice carries `_HALF_COLS` == 32
    columns; a wider slice needs a second word per half and the `1 << j`
    immediates would wrap silently. Same no-fallback rule as `_slice_in_order`,
    and for the same reason it lives out here rather than in the kernel: the
    kernel's body is staged, and a `raise` may only run in plain Python
    (`UNSUP_EARLY_EXIT` otherwise).
    """
    if n != _HALF_COLS:
        raise RuntimeError(
            "%s: %d columns per slice, the causal bitmap wants %d "
            "(one 32-bit word)" % (what, n, _HALF_COLS))
    return True


def _frag_index(t):
    """A trace-time index builder for element j of a partitioned copy fragment.

    The tcgen05 copy family reports its value mode either nested — `((E, R),
    MMA...)`, one repetition per instruction — or flat, depending on the atom,
    and the two want `((j, 0),) + tail` and `(j,) + tail` respectively. Reading
    it off the shape rather than hard-coding one spelling is what lets the
    segment change repetition (x64 -> two x32, D0.14) without the index form
    becoming a second, silent assumption.
    """
    tail = (0,) * (cute.rank(t) - 1)
    if isinstance(t.shape[0], tuple):
        return lambda j: ((j, 0),) + tail
    return lambda j: (j,) + tail


def _pick(a, b):
    """`a` if this DSL build gave one, else the fallback `b`. Module level so
    the choice is plain Python: a trace-time selection written as an `if` in a
    decorated body is source-rewritten, and the two arms here are objects, not
    control flow."""
    if a is None:
        return b
    return a


def _frag_view(t, off, lay):
    """Half of a flat register line, wearing the copy fragment's own layout, or
    None if this DSL build will not offset register storage.

    It is what lets the dQ-writer's two `Ld32x32b.x32` land directly in the two
    halves of the 64-wide line the ring store reads (D0.14: "two LDTM.x32 = 64
    columns taken in one go"), instead of through a 32-register staging plus a
    copy loop.
    """
    try:
        return cute.make_tensor(t.iterator + off, lay)
    except Exception:
        return None


def _flat_view(t, n):
    """A flat n-element view of a register fragment, or None if this DSL build
    will not re-view register storage.

    The packed bf16 the STTM writes and the 64 bf16 of one dS swizzle row are
    the same values in the same order (D0.14), so the smem store reads the
    packing's own registers rather than a second copy of them; the flat view is
    only how a vectorizing copy is told those 64 values are one line.
    """
    try:
        return cute.make_tensor(t.iterator, cute.make_layout(n))
    except Exception:
        return None


@dsl_user_op
def _cvt_bf16x2(lo, hi, *, loc=None, ip=None) -> cutlass.Int32:
    """One pair of f32 -> one packed bf16x2 word, `lo` in the LOW half.

    hw_kb `cvt_f32x2_bf16x2_pack_into_r2t_fragment`, form (a): SASS
    `F2FP.BF16.F32.PACK_AB`, one instruction per pair and no `PRMT`. The PTX is
    `cvt.rn.bf16x2.f32 d, a, b` with `a` landing in the HIGH half, so the
    operands are written `$0, $2, $1`; spelling them in source order is the
    silent bug the kb names (every word's halves swap = adjacent columns swap,
    finite numbers at the wrong places).
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
         cutlass.Float32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _pack_pairs(dst_i32, ix, src_f32, n_pair):
    """D0.4 (3): a line of f32 rounded to bf16 a PAIR at a time, each pair
    written as one whole 32-bit slot of the destination staging.

    The unit is the slot, not the element and not the line. Per element
    (`r[j] = x.to(BFloat16)` into a bf16 recast face) is one convert per value
    plus one half-register read-modify-write, twice the instructions; per line
    (`dst.store(src.load().to(bf16))`) reaches the same `PACK_AB` count but is
    one value-domain operation over the whole source, so every element of the
    source stays live until the store retires. In D0.14's segment D that is
    P's 64 fp32 plus the 32 packed registers coexisting at the segment's peak,
    which is exactly the margin the budget does not have (kb
    `cvt_f32x2_bf16x2_pack_into_r2t_fragment`, gotcha "整行 TensorSSA ... 活跃区
    不对"). Pairwise, each pair dies at its own convert and the writes
    interleave with the reads.

    Slot k is source elements (2k, 2k+1) — the two bf16 of the 32-bit column
    the STTM writes and the two adjacent Q columns the smem row carries, in
    that order. DEVICE side, trace time: plain `range`, no region.
    """
    for k in range(n_pair):
        dst_i32[ix(k)] = _cvt_bf16x2(src_f32[2 * k], src_f32[2 * k + 1])


def _pack_dst(flat, a_face, a_ix, line):
    """The 32-bit slot face the dS row / epilogue staging is packed through:
    the STTM fragment's own face when the flat bf16 view of it took (one
    staging serves both consumers, D0.14), the separate bf16 line's otherwise.

    Module level so the choice stays plain Python (`_pick`'s rule), and the
    line's recast is taken only on the build that needs it — the fragment's own
    f32 -> i32 recast is same-width and always available, a bf16 line's is not.
    """
    if flat is not None:
        return a_face, a_ix
    b = cute.recast_tensor(line, cutlass.Int32)
    return b, _frag_index(b)


def _imm(v):
    """A Python bit pattern as the signed value an Int32 constant accepts.

    Bit 31's mask is a negative Int32 (hw_kb `threshold_bitmap_predicate_select`
    gotcha 3), so the column-63 test would otherwise overflow the constant.
    """
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


@dsl_user_op
def _ones_from(t, *, loc=None, ip=None) -> cutlass.Int32:
    """A 32-bit word whose bit c is set iff `c >= t`; t <= 0 -> all ones,
    t >= 32 -> zero.

    D0.5's causal predicate over a lane's run of columns is a THRESHOLD, not a
    per-column flag: the valid set is a suffix `c >= t`. hw_kb
    `threshold_bitmap_predicate_select` — one word per lane per trip, the
    per-column work being a bit test that ptxas folds into `R2P` + `FSEL`
    instead of the `VIADD` + `ISETP` + `FSEL` a per-column comparison costs.
    `max.s32` clamps the low end and PTX's own shift rule ("shift amounts
    greater than the register width are clamped") the high end, so neither end
    needs a comparison. The shift MUST be done here in PTX: LLVM's `shl` is
    poison past the register width (gotcha 1).
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [cutlass.Int32(t).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 tl;\n\t.reg .b32 m;\n\t"
        "max.s32 tl, $1, 0;\n\tmin.s32 tl, tl, 32;\n\t"
        "mov.b32 m, 0xffffffff;\n\tshl.b32 $0, m, tl;\n}",
        "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _sel_bit(cond, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """`cond != 0 ? a : b` — one `setp` + one `selp`, no scf region.

    The bitmap's consumer: the flag stays a PREDICATE and never becomes a float
    multiplicand (hw_kb `branch_free_select_mask` form (a)), and the predicate's
    source is a single-bit test of one register (`w & (1 << c)`, gotcha 4) —
    which is the shape ptxas needs to pack seven of them into one `R2P`.
    `setp` and `selp` must stay in the SAME asm block: predicates do not cross
    block boundaries.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(cond).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;"
        "\n\tselp.f32 $0, $2, $3, p;\n}",
        "=f,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _gview_k(t):
    return cute.make_tensor(t.iterator, cute.make_layout(
        (_N, _D, _H, _B), stride=(_STRIDE_N, 1, _STRIDE_H, _STRIDE_B)))


def _gview_mn(t):
    return cute.make_tensor(t.iterator, cute.make_layout(
        (_D, _N, _H, _B), stride=(1, _STRIDE_N, _STRIDE_H, _STRIDE_B)))


def _build_face(src, tag):
    """HOST side: (tma atom, tma coord tensor) for one face.

    `CtaGroup.TWO` on the load op is the whole of the pair mechanism: it is what
    lets a completion signal be deposited in the peer's mbarrier. Without it the
    `map_dsmem_ptr` at the issue site is UB and the barrier never fills (the
    symptom is a hang, not wrong numbers).
    """
    major, mn, k = _FACE[tag]
    view = _gview_mn(src) if major == "MN" else _gview_k(src)
    op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
    return cpasync.make_tiled_tma_atom(op, view, _face_layout(tag), (mn, k), 1)


def _build_store(dst, tag):
    """HOST side: (tma atom, tma coord tensor) for one epilogue STORE box.

    The mirror image of `_build_face`, with two differences that both follow
    from the direction: the op is `CopyBulkTensorTileS2GOp` (smem -> gmem), and
    it carries NO cta group — a store is per-CTA bookkeeping on the bulk
    async-group scoreboard, not an mbarrier transaction, so there is no pair
    accounting and nothing to arm (hw_kb `tcgen05_epilogue_t2r_stsm_tma_store`).
    """
    major, mn, k = _FACE[tag]
    view = _gview_mn(dst) if major == "MN" else _gview_k(dst)
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), view, _face_layout(tag), (mn, k))


def _partition(atom, g, s, tag):
    """DEVICE side: fuse both ends' box modes and partition once per face."""
    major, mn, k = _FACE[tag]
    gz = cute.zipped_divide(g, (mn, k))
    ts, tg = cpasync.tma_partition(atom, 0, cute.make_layout(1),
                                   cute.group_modes(s, 0, 2), gz)
    return (atom, ts, tg)


def _issue(part, coord, bar):
    """One TMA copy. `coord` names the tile in the view's own mode order.

    No lane predicate: the DSL copy atom elects internally, and wrapping it in
    `elect_one` deadlocks. The caller's guard is warp-granular.
    """
    atom, ts, tg = part
    dst = ts if cute.rank(ts) == 1 else ts[None, 0]
    cute.copy(atom, tg[None, coord], dst, tma_bar_ptr=bar)


def _issue_store(part, coord):
    """One TMA store — the same partition read the other way round.

    No `tma_bar_ptr`: a bulk-tensor STORE completes on the issuing thread's
    async-group scoreboard (`commit_group` / `wait_group`), not on an mbarrier,
    which is why the epilogue needs no FULL row of its own.
    """
    atom, ts, tg = part
    src = ts if cute.rank(ts) == 1 else ts[None, 0]
    cute.copy(atom, src, tg[None, coord])


@dsl_user_op
def _bulk_g2s_512(dst_smem, src_gmem, mbar_smem, *, loc=None, ip=None):
    """`UBLKCP.S.G` for one 512 B row-statistic tile (non-tensor bulk copy).

    The DSL's cpasync surface has no non-tensor `cp.async.bulk` G2S, so this is
    the tier-B escape. It is a per-thread instruction with no internal election,
    so unlike the TMA atoms above the CALLER must elect exactly one thread.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src_gmem).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar_smem).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [$0], [$1], 512, [$2];",
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _sel_le_f32(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """`x <= y ? a : b` — one FSETP + one FSEL, no scf region.

    D2.1's lse guard: a fully masked row's lse is -inf, and the branch form of
    the guard puts a whole scf region (and a reconvergence point) around a
    single `FSEL` on all 128 rows. `setp.le.f32` is ORDERED, so a NaN comparand
    takes the `b` arm — the same arm the finite rows take, which is the arm that
    keeps the NaN visible instead of quietly turning it into a zero.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.le.f32 p, $1, $2;"
        "\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _shfl_bfly_f32(x, mask, *, loc=None, ip=None) -> cutlass.Float32:
    """`shfl.sync.bfly.b32`: lane `l` reads lane `l ^ mask`, whole warp live.

    D2.1's row reduction. The 16 lanes that share a q row are a CONTIGUOUS
    16-lane segment of the warp, and `j = lane ^ mask` for mask in 1/2/4/8
    never flips lane bit 4 — so the exchange cannot leave the segment and the
    clamp operand stays at the full-warp `0x1f` (a segmask would be a no-op
    here). `mask` is a python int, so it is a PTX immediate and costs no
    register; the membermask is 0xffffffff because every thread of the block
    reaches this in uniform control flow.

    hw_kb `ptx_escape` form 3 (a returning intrinsic), with
    `has_side_effects=True` because a warp-convergent op must not be sunk,
    hoisted or duplicated across the reduction's steps.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "shfl.sync.bfly.b32 $0, $1, %d, 0x1f, 0xffffffff;" % int(mask),
        "=f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# --- D2.1: the preprocess kernel's three streams, as trace-time helpers --------
# Module level, so their unrolled bodies are ordinary python (the `_issue`
# convention): plain `range`, plain list building, no construct the kernel
# source preprocessor has to rewrite.

def _pre_load_granules(gO8, gDO8, v_base):
    """DEVICE side: this thread's 16 `LDG.E.128` — one o granule and one dO
    granule per step — ALL issued before any of them is read.

    Each step gets its own pair of register lines rather than reusing one, so
    the 16 loads carry no false dependency between them and the whole batch is
    in flight at once (16 B x 16 x 256 threads = 64 KB per CTA). That is the
    working set the reference's 80 registers/thread is made of.
    """
    ro, rd = [], []
    for st in range(_PRE_STEPS):
        v = v_base + st * _PRE_STEP_GRAN
        a = _rmem_line(_PRE_GRAN, cutlass.BFloat16)
        c = _rmem_line(_PRE_GRAN, cutlass.BFloat16)
        _COPY128(_gvec(gO8, v), a)
        _COPY128(_gvec(gDO8, v), c)
        ro.append(a)
        rd.append(c)
    return ro, rd


def _pre_zero_tile(gAcc4, tile_v, tidx):
    """DEVICE side: this CTA's 64 KB run of dq_accum, as 16 `STG.E.128` of a
    zero granule per thread. Nobody else clears the workspace — a missed zero
    makes the dQ reduce-add accumulate garbage and dq is wrong every call."""
    z4 = _rmem_line(_ACC_VEC, cutlass.Float32)
    for j in range(_ACC_VEC):
        z4[j] = cutlass.Float32(0.0)
    for i in range(_PRE_ZERO_STEPS):
        _COPY128(z4, _gvec(gAcc4, tile_v + i * _PRE_THREADS + tidx))


def _pre_rowsums(ro, rd):
    """DEVICE side: one row sum per step, in this thread's lane of the segment.

    Per step: the 8-term dot product of the thread's two granules (two
    independent FFMA chains so the 8 products are not one dependent line), then
    the 4-step butterfly that folds the 16 lanes of the row into every lane of
    it. All 16 lanes end up holding the row sum; lane 0 of the segment is the
    one that stores it.
    """
    out = []
    for st in range(_PRE_STEPS):
        e0 = cutlass.Float32(0.0)
        e1 = cutlass.Float32(0.0)
        for j in range(0, _PRE_GRAN, 2):
            e0 = e0 + _f32(ro[st][j]) * _f32(rd[st][j])
            e1 = e1 + _f32(ro[st][j + 1]) * _f32(rd[st][j + 1])
        acc = e0 + e1
        for s in range(_PRE_BFLY):
            acc = acc + _shfl_bfly_f32(acc, 1 << s)
        out.append(acc)
    return out


def _pre_store_sums(mDpsum, b, h, row_q, sums):
    """DEVICE side: the 8 `STG.E` of the segment-leader lanes, one per step."""
    for st in range(_PRE_STEPS):
        mDpsum[(b, h, row_q + st * _PRE_ROWS_STEP)] = sums[st]


# --- preprocess ---------------------------------------------------------------
# D0.3 producer: dpsum = rowsum(o * do) and lse_log2 = lse * log2(e) (-inf -> 0),
# plus the dq_accum workspace zeroing (nobody else clears it; a missed zero makes
# the reduce-add accumulate garbage and dq is wrong every call).
# D0.4: the log2-domain statistic is produced HERE, so the main loop's exp2 is a
# single FFMA + MUFU.EX2 with no base change of its own.
# D2.1 gives all three streams their access shape: 16 B granules everywhere, the
# row reduction a 16-lane butterfly rather than a per-thread walk of the row, and
# the -inf guard a predicate. The kernel touches no smem and no tensor core — it
# is a bandwidth job (270.6 MB in, 223.6 MB out per call), so what it spends per
# byte is the whole of it.

@cute.kernel
def _preprocess_kernel(
    mO: cute.Tensor,        # (B, N, H, D) bf16
    mDO: cute.Tensor,       # (B, N, H, D) bf16
    mLSE: cute.Tensor,      # (B, H, N)    f32
    mDpsum: cute.Tensor,    # (B, H, N)    f32   out
    mLseLog2: cute.Tensor,  # (B, H, N)    f32   out
    mDqAcc: cute.Tensor,    # (ACC_ELEMS,) f32   out (zeroed)
):
    tidx, _, _ = cute.arch.thread_idx()
    m_blk, h, b = cute.arch.block_idx()

    # D2.1's three streams, all taken through 16 B granules. The two gmem views
    # are built ONCE from the unoffset iterator, which is the spelling that
    # carries the granule's alignment into the copy atom (`_gvec_view`); the
    # three tensors are contiguous (B, N, H, D) / (ACC_ELEMS,) runs, so a flat
    # granule index is the whole address arithmetic.
    gO8 = _gvec_view(mO, _B * _N * _H * _D, _PRE_GRAN)
    gDO8 = _gvec_view(mDO, _B * _N * _H * _D, _PRE_GRAN)
    gAcc4 = _gvec_view(mDqAcc, _ACC_ELEMS, _ACC_VEC)

    # this thread's place in the o / dO read: `lane` is which of the row's 16
    # granules it owns, `row0` which of the step's 16 rows. tidx is
    # row0 * 16 + lane, so the 16 lanes of a row are consecutive threads (one
    # instruction = two whole 256 B rows per warp) and a butterfly over masks
    # 1/2/4/8 stays inside the row.
    lane = tidx % _PRE_ROW_GRAN
    row0 = tidx // _PRE_ROW_GRAN
    row_q = m_blk * _BLK_M + row0
    v_base = ((b * _N + row_q) * _H + h) * _PRE_ROW_GRAN + lane

    # (1) the block's o / dO, all 16 loads in flight before the first is read.
    ro, rd = _pre_load_granules(gO8, gDO8, v_base)

    # (2) zero this CTA's 64 KB run of dq_accum while those loads are in the
    # air. The zeroing is the one consumer of the run that is blind to D0.3's
    # block layout — it writes every byte of it — so it survives the layout
    # unchanged; the address is still taken through the shared helper so the
    # workspace's three call-sites (producer / accumulator / consumer) cannot
    # drift apart.
    _pre_zero_tile(gAcc4, _acc_tile_base(b, h, m_blk) // _ACC_VEC, tidx)

    # (3) dpsum = rowsum(o * dO): 8 dot products + 8 butterflies, then the
    # segment leaders store their 8 rows.
    sums = _pre_rowsums(ro, rd)
    if lane == 0:
        _pre_store_sums(mDpsum, b, h, row_q, sums)

    # (4) lse_log2 = lse * log2(e), with the fully masked rows (-inf) parked at
    # 0. One f32 per q row, so the first 128 threads, and the guard is a
    # predicate + FSEL rather than a region.
    if tidx < _BLK_M:
        row = m_blk * _BLK_M + tidx
        l = mLSE[(b, h, row)]
        mLseLog2[(b, h, row)] = _sel_le_f32(
            l, cutlass.Float32(_NEG_BIG), cutlass.Float32(0.0),
            l * cutlass.Float32(_LOG2E))


# --- main (M0 construction step 5: topology + supply + chains + softmax + dQ) -
# D0.2 shape: grid (N/128, H, B) with x = KV block, cluster (2,1,1) along x so a
# CTA pair covers 256 KV rows; block = 512 threads (D0.7's 16 warps).
# D0.6: all 512 TMEM columns are allocated for the pair (is_two_cta=True) and
#       released at a cluster-wide rendezvous.
# D0.5: the causal trip structure (pair p covers KV [256p, 256p+256), q blocks
#       run m = 2p .. 31, first trip's supply peeled into the prologue) and the
#       per-element predicate select that masks the scores.
# D0.8: the arena is carved as its buffer list, and the seven operand faces are
#       held as swizzled bf16 tiles at their listed offsets — the same layouts
#       the host built the TMA descriptors on, which is also what the UMMA smem
#       descriptors are built on.
# D0.10: one TMA producer warp per CTA; K / V / K_dsplit in the prologue, and
#       Q_row / dO_row / dO_col / Q_col / lse / dpsum single-stage per trip. The
#       operand faces are pair-accounted (`CtaGroup.TWO`, leader arms the pair's
#       bytes once, both CTAs issue, only the leader waits); lse / dpsum are the
#       one exception — non-tensor bulk copies, each CTA arming its own 512 B.
# D0.11: the tmem-side handoff — the MMA warp's two chain commits are multicast
#       so both CTAs' softmax warps can wait them, and the segment answers with
#       the two count-16 rows the leader's MMA warp waits before its RS chains —
#       plus the dS exchange: the segment's registers become an MN-major smem
#       face (local half) and a send staging (the peer's Q rows), one 16 KB
#       `UBLKCP.S.S` per CTA per trip arms and fills the PEER's 0x158, and the
#       relay warp turns that into the count-2 0x160 chain3 waits.
# D0.12: dQ leaves TMEM on the writer warps (8-11) — one LDTM of the 64 f32
#       columns at 0x40, a 4-slot x 4 KB smem ring, and eight
#       `cp.reduce.async.bulk...add.f32` of 4 KB each into D0.3's blocked f32
#       workspace, so the cross-KV-block reduction happens at the L2 slice with
#       no SM-side atomic and no per-element store. 0x90 (multicast) and 0x98
#       (count 8) are the accumulator's FULL / EMPTY pair.

@cute.kernel
def _main_kernel(
    mLseLog2: cute.Tensor,
    mDpsum: cute.Tensor,
    mDqAcc: cute.Tensor,
    a_sdk: cute.CopyAtom, g_sdk: cute.Tensor,
    a_sdv: cute.CopyAtom, g_sdv: cute.Tensor,
    a_qr: cute.CopyAtom, g_qr: cute.Tensor,
    a_dr: cute.CopyAtom, g_dr: cute.Tensor,
    a_qc: cute.CopyAtom, g_qc: cute.Tensor,
    a_dc: cute.CopyAtom, g_dc: cute.Tensor,
    a_k: cute.CopyAtom, g_k: cute.Tensor,
    a_v: cute.CopyAtom, g_v: cute.Tensor,
    a_kd: cute.CopyAtom, g_kd: cute.Tensor,
):
    _seg("main.entry")
    tidx, _, _ = cute.arch.thread_idx()
    n_blk, h, b = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- D0.7's register split, before anything else -----------------------
    # Warpgroup granular and warpgroup uniform: `warp < 8` is the two softmax
    # warpgroups, `8 <= warp < 12` the writer warpgroup, `warp >= 12` the
    # service one, so every warp of a warpgroup takes the same arm with the same
    # immediate, which is what `.aligned` requires (D0.14 r3 splits what used to
    # be one 136 arm over warps 0-11 into 152 for the segment and 104 for the
    # writer). It has to come first: the counts are a property of the warp from
    # here on, and an `.inc` issued after the segment has already spilled buys
    # nothing.
    if warp < _WARP_WRITER:
        _REG_INC(_REG_SOFTMAX)
    elif warp < _WARP_MMA:
        _REG_DEC(_REG_WRITER)
    else:
        _REG_DEC(_REG_SERVICE)

    # ---- D0.8 arena, in the buffer list's own order ------------------------
    smem = cutils.SmemAllocator()
    mbar = smem.allocate_array(cutlass.Int64, _N_BAR_LO, byte_alignment=16)
    smem.allocate_array(cutlass.Int32, (_TMEM_SLOT_OFF - 8 * _N_BAR_LO) // 4,
                        byte_alignment=16)
    tmem_holding = smem.allocate_array(cutlass.Int32, 2, byte_alignment=16)
    mbar_hi = smem.allocate_array(cutlass.Int64, _N_BAR_HI, byte_alignment=8)
    smem.allocate_array(
        cutlass.Int32, (_SM_CTRL - _BAR_HI_OFF - 8 * _N_BAR_HI) // 4,
        byte_alignment=8)
    sQr = _alloc_face(smem, "q_row")
    sK = _alloc_face(smem, "k")
    sV = _alloc_face(smem, "v")
    sDOc = _alloc_face(smem, "do_col")
    sQc = _alloc_face(smem, "q_col")
    sDOr = _alloc_face(smem, "do_row")
    # dS send staging (0x20400): dS^T[my 128 KV][the peer's 64 Q], written by
    # the warpgroup that holds the peer's Q columns, read by the bulk copy.
    sDSs = _alloc_face(smem, "ds_send")
    sKd = _alloc_face(smem, "k_dsplit")
    # dS (0x2c400): chain3's A — my 64 Q rows x the pair's 256 KV. The k-block
    # my own KV rows own is written here directly; the other one arrives from
    # the peer's staging over DSMEM.
    sDS = _alloc_face(smem, "ds")
    sLse = smem.allocate_array(cutlass.Float32, _BLK_M, byte_alignment=128)
    sDps = smem.allocate_array(cutlass.Float32, _BLK_M, byte_alignment=128)
    # dQ reduction ring (0x34800): 4 slots x 4 KB (D0.12). UNSWIZZLED on
    # purpose — its reader is a bulk reduce over raw bytes, so the iterator's
    # integer IS the hardware address — and held as a tensor with the
    # (slot, quad, lane, 4) layout it is used through, so the view and the 4 KB
    # the engine reduces into D0.3's chunk are the same bytes by construction.
    sRing = smem.allocate_tensor(cutlass.Float32, _ring_layout(),
                                 byte_alignment=1024)

    # ---- D0.2 / D0.5 linear map -------------------------------------------
    rank = n_blk % 2                    # leader = even rank of the pair
    peer = 1 - rank
    pair = n_blk // 2
    kv_row0 = n_blk * _BLK_N            # = 256 * pair + 128 * rank
    m_min = 2 * pair                    # first q block that can see this pair
    n_iter = _NBLK - m_min              # 32 - 2p trips, the first one peeled

    # ---- D0.10 barrier table (the supply half; 0x40..0x98 belong to later
    # steps) + D0.6's TMEM allocation, both before the init fence -----------
    if warp == _WARP_SOFTMAX:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + _BAR_Q_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_Q_EMPTY, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DO_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DO_EMPTY, 1)
            cute.arch.mbarrier_init(mbar + _BAR_LSE_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_LSE_EMPTY, 8)
            cute.arch.mbarrier_init(mbar + _BAR_DPS_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DPS_EMPTY, 8)
            cute.arch.mbarrier_init(mbar + _BAR_S_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DP_FULL, 1)
            # D0.11's three count-16 rows: 8 softmax warps per CTA x 2 CTAs, all
            # arriving the LEADER's copy through a cluster address.
            cute.arch.mbarrier_init(mbar + _BAR_P_RDY, 16)
            cute.arch.mbarrier_init(mbar + _BAR_DS_RDY, 16)
            cute.arch.mbarrier_init(mbar + _BAR_S_DONE, 16)
            # the dS smem face's own EMPTY: chain3's multicast commit flips it
            # in BOTH CTAs, so each CTA's copy takes one arrive per trip.
            cute.arch.mbarrier_init(mbar + _BAR_DS_EMPTY, 1)
            # D0.15's two full/empty pairs. The FINAL rows take the leader's
            # MULTICAST commit (both CTAs' softmax warps drain their OWN 128
            # TMEM lanes, so a leader-only arrive hangs the follower); the
            # EMPTY rows take all 16 softmax warps of the pair, pre-arrived in
            # the prologue so the MMA warp's single acquire passes straight
            # through (D0.11: "首次 acquire 直接通过").
            cute.arch.mbarrier_init(mbar + _BAR_DV_FINAL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DK_FINAL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DV_EMPTY, 16)
            cute.arch.mbarrier_init(mbar + _BAR_DK_EMPTY, 16)
            # D0.12's two rows: 0x90 takes chain3's multicast commit (one arrive
            # in each CTA), 0x98 takes the 8 writer warps of the pair.
            cute.arch.mbarrier_init(mbar + _BAR_DQ_FULL, 1)
            cute.arch.mbarrier_init(mbar + _BAR_DQ_EMPTY, 8)
            cute.arch.mbarrier_init(mbar_hi + _BAR2_QC_FULL, 1)
            cute.arch.mbarrier_init(mbar_hi + _BAR2_QC_EMPTY, 1)
            cute.arch.mbarrier_init(mbar_hi + _BAR2_KD_FULL, 1)
            cute.arch.mbarrier_init(mbar_hi + _BAR2_KD_EMPTY, 1)
            # 0x158 takes the PEER's remote expect-tx (count 1, tx = the whole
            # 16 KB); 0x160 takes both relay warps' arrives into the leader.
            cute.arch.mbarrier_init(mbar_hi + _BAR2_DS_PEER, 1)
            cute.arch.mbarrier_init(mbar_hi + _BAR2_DS_DONE, 2)
    if warp == _WARP_MMA:
        cute.arch.alloc_tmem(_TMEM_COLS, tmem_holding, is_two_cta=True)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=True)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding)
    _seg("main.tmem.ready")

    # ---- D0.14: one scope per warp role ------------------------------------
    # Every role's operand algebra and every role's trip loop live INSIDE that
    # role's own branch from here on. The previous shape — all five roles'
    # fragments built in the kernel's common scope, one shared trip loop with
    # role branches inside it — makes every role's registers live in every warp
    # for the whole kernel, because a value defined before the loop and used
    # inside it is live across the loop on every path. Their sum is not a
    # budget any allocation can meet: the segment's P line (64 f32) + its
    # operand slice (32) + its bf16 packing (32) + the writer's dQ line (64) is
    # 192 registers of ARRAYS alone, before a single descriptor, against the 128
    # per thread that 512 threads x 1 CTA/SM leaves (65,536 / 512). What ptxas
    # does with the difference is spill it, every trip, in every warp.
    # D0.14's account is a PER-ROLE account — softmax 124, writer 68, MMA 82,
    # TMA 6, relay 2 — and this is what makes those four numbers the four
    # separate live sets they are read as, rather than one sum: a role's arrays
    # now die at the end of its branch, and the peak the allocator sees is the
    # max over roles, not the total. Nothing about D0.13's issue order, the
    # barrier table, the waits or their target tiles changes; the roles were
    # already skewed against each other and already synchronised only through
    # mbarriers and warp-scoped named barriers, so splitting the one loop into
    # four gives every role the same trip sequence it ran before.
    # ---- D0.5 / D0.13 trip loop --------------------------------------------
    # Single stage (D0.8's budget leaves no room for a ring), so the refill of a
    # face can only start once its consumer has released it; D0.13 buys the
    # depth a ring would have bought by RELEASING EARLY instead — a face is free
    # the moment the chain that reads it has been issued and committed, not when
    # the trip ends.
    # All five chains run here — S^T reads K and Q_row and dP^T reads V and
    # dO_row by smem descriptor, dK reads tmem 0x100 against Q_col and dV reads
    # tmem 0x000 against dO_col (D0.9's two RS rows), and dQ reads the exchanged
    # dS face against K_dsplit (D0.9's smem row).
    # Every role runs n_iter trips, but they are SKEWED against each other and
    # that skew is the decision: trip i of the MMA warp issues the chains of two
    # different tiles, trip i of the segment works on tile i, and trip i of the
    # TMA warp fetches tile i+2 (Q_row / dO) and tile i+1 (Q_col / lse / dpsum).
    # The three hazards the old per-trip rendezvous used to cover are covered by
    # ISSUE ORDER plus one wait each instead:
    #   S(i+1) overwrites P^T(i)  -> dV(i) was issued last trip, in order;
    #                                the segment is past S(i) because dV(i)
    #                                waited 0x48 flip i.
    #   dP(i+1) overwrites dS^T(i)-> dK(i) was issued earlier this trip.
    #   dQ(i) overwrites S^T(i+1)'s second half -> 0x60 flip i (the segment's
    #                                LDTM of S(i+1)) and 0x98 flip i (the
    #                                writer's drain of dQ(i-1)).

    # ---- D0.10 prologue: K / V / K_dsplit once + the first trip's stream ----
    # K rides Q_row's barrier and V rides dO's (one arm of 0x18000 / 0x20000 for
    # the pair), so the loop below sees one completion per trip on every barrier
    # and the phase of trip i is simply i & 1.
    _seg("tma.entry")
    if warp == _WARP_TMA:
        # ---- D1.5: the nine tensormaps, prefetched before the first arm ------
        # First statement of the producer's prologue and ahead of every
        # expect-tx: the descriptors' gmem reads are then in flight across the
        # arming and the cluster rendezvous below, which is the only stretch of
        # this kernel where the TMA warp has nothing else to wait on. The two
        # store descriptors are in the list even though the SEGMENT issues
        # them — the fetch warms one cache for the whole CTA, and the epilogue
        # has no window of its own to hide a cold descriptor in.
        _atoms = {"q_row": a_qr, "do_row": a_dr, "q_col": a_qc, "do_col": a_dc,
                  "k_dsplit": a_kd, "k": a_k, "v": a_v,
                  "store_dk": a_sdk, "store_dv": a_sdv}
        for _tag in _PREFETCH_ORDER[:_PREFETCH_TMAPS]:
            _PREFETCH_DESC(_atoms[_tag])
        if rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar + _BAR_Q_FULL, _TX_PRO_Q)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar + _BAR_DO_FULL, _TX_PRO_DO)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_hi + _BAR2_QC_FULL, _TX_QC)
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_hi + _BAR2_KD_FULL, _TX_KD)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar + _BAR_LSE_FULL, _TX_STAT)
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar + _BAR_DPS_FULL, _TX_STAT)
    # The arm has to be visible to BOTH issuers before either copy. In steady
    # state the EMPTY handshake gives that ordering; the prologue has no such
    # handshake, so it pays one cluster rendezvous.
    _seg("tma.prologue.arm.begin")
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    _seg("tma.prologue.arm.end")

    # ======== warp 13: the TMA producer ========
    if warp == _WARP_TMA:
        # ---- one partition per face, outside every predicate (layout algebra) --
        p_qr = _partition(a_qr, g_qr, sQr, "q_row")
        p_dr = _partition(a_dr, g_dr, sDOr, "do_row")
        p_qc = _partition(a_qc, g_qc, sQc, "q_col")
        p_dc = _partition(a_dc, g_dc, sDOc, "do_col")
        p_k = _partition(a_k, g_k, sK, "k")
        p_v = _partition(a_v, g_v, sV, "v")
        p_kd = _partition(a_kd, g_kd, sKd, "k_dsplit")

        # Every FULL barrier of an operand face lives in the LEADER's smem: that is
        # what pair accounting means, and it is where both CTAs deposit.
        lb_q = cute.arch.map_dsmem_ptr(mbar + _BAR_Q_FULL, 0)
        lb_do = cute.arch.map_dsmem_ptr(mbar + _BAR_DO_FULL, 0)
        lb_qc = cute.arch.map_dsmem_ptr(mbar_hi + _BAR2_QC_FULL, 0)
        lb_kd = cute.arch.map_dsmem_ptr(mbar_hi + _BAR2_KD_FULL, 0)

        _issue(p_k, (n_blk, 0, h, b), lb_q)
        _issue(p_qr, (2 * m_min + rank, 0, h, b), lb_q)
        _issue(p_v, (n_blk, 0, h, b), lb_do)
        _issue(p_dc, (rank, m_min, h, b), lb_do)
        _issue(p_dr, (2 * m_min + rank, 0, h, b), lb_do)
        _issue(p_qc, (rank, m_min, h, b), lb_qc)
        _issue(p_kd, (rank, pair, h, b), lb_kd)
        # `off_pro` and nothing else: a name bound inside `with elect_one()` is
        # NOT scoped to that block — the context manager is a plain runtime
        # object, so its SSA value lives inside the `scf.if` region it opens
        # while the Python name leaks into this branch's scope. If that name is
        # also assigned inside the trip loop below, the loop rewriter sees it as
        # live-in and makes it a loop-carried value whose init and yield are
        # both defined in a child region — which is an SSA dominance error, not
        # a numerics bug, and it takes the whole kernel's lowering with it. The
        # prologue and the body used to sit in different regions, so the two
        # `off`s never met; one scope per role (below) put them in the same one.
        # Every elect_one binding in the trip loops is spelled uniquely for the
        # same reason.
        with cute.arch.elect_one():
            off_pro = cutlass.Int64((b * _H + h) * _N + m_min * _BLK_M) * 4
            _bulk_g2s_512(_addr(sLse), _addr(mLseLog2) + off_pro,
                          _addr(mbar + _BAR_LSE_FULL))
            _bulk_g2s_512(_addr(sDps), _addr(mDpsum) + off_pro,
                          _addr(mbar + _BAR_DPS_FULL))

        # ---- D0.10 / D0.13: the supply's matching warm-up ----------------------
        # Q_row and dO are consumed one tile AHEAD of the trip index (the body's
        # trip i runs S / dP / dV of tile i+1), so their producer runs two tiles
        # ahead of the trip: tile 0 went out above, tile 1 goes out here as soon as
        # the prologue's chains release the faces, and the loop below issues tile
        # i+2. Q_col / lse / dpsum keep the one-tile lead — their consumers (chain1
        # and the segment) are on tile i during trip i.
        # One segment per face, each waiting its OWN empty — the same shape the body
        # takes below, and for the same reason: a segment that waits two empties at
        # once cannot re-arm the first face until the second one's producer has run.
        _seg("tma.pro2.empty.begin")
        _tma_empty_wait(mbar + _BAR_Q_EMPTY, 0)
        _seg("tma.pro2.empty.end")
        if rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar + _BAR_Q_FULL, _TX_Q)
        _seg("tma.pro2.issue.begin")
        _issue(p_qr, (2 * (m_min + 1) + rank, 0, h, b), lb_q)
        _seg("tma.pro2.issue.end")
        _seg("tma.pro2.do.begin")
        _tma_empty_wait(mbar + _BAR_DO_EMPTY, 0)
        _seg("tma.pro2.do.end")
        if rank == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar + _BAR_DO_FULL, _TX_DO)
        _issue(p_dc, (rank, m_min + 1, h, b), lb_do)
        _issue(p_dr, (2 * (m_min + 1) + rank, 0, h, b), lb_do)

        for i in cutlass.range(n_iter):
            ph_t = i % 2
            # ---- D0.10 / D0.13: supply, at the two leads the order requires -----
            # No rendezvous here any more: step 2b's "sync the CTA, sync the
            # cluster, then arrive the EMPTY barriers by hand" stub is exactly the
            # all-hands rendezvous D0.10's symptom table warns about, and it also
            # pinned the producer to the slowest consumer of the trip. Every face
            # is released by the commit of the chain that read it (0x08 off chain0,
            # 0x138 off chain1, 0x18 off chain4, all multicast so the follower's
            # producer wakes too), and the TMA warp's only synchronisation is the
            # EMPTY wait in front of each re-arm. Q_row / dO are consumed by
            # S(i+1) / dP(i+1) / dV(i+1) in THIS trip, so their producer runs two
            # tiles ahead (tile i+2); Q_col / lse / dpsum keep the one-tile lead.
            # D0.10 形态 A, 统计量段首发: FIVE serial segments, each waiting
            # exactly ONE empty, in the order `lse -> dpsum -> Q_col -> Q_row ->
            # dO_col + dO_row`. The order is a contract, not a formatting choice.
            # What it pins is a RELATION, not a spelling: neither statistic's
            # EMPTY wait may sit behind an EMPTY that an MMA chain's retirement
            # gates. The three operand faces are exactly those EMPTYs —
            #   Q_col : consumed by dK(i+1); released by chain1(i)        -> 0x138
            #   Q_row : consumed by S(i+2);  released by chain0(i)        -> 0x08
            #   dO    : consumed by dP(i+2) / dV(i+2); by chain4(i)       -> 0x18
            # — while the two statistics are released by the softmax warps
            # themselves, the moment they are done broadcasting the tile's
            # values (D0.13 s1 / s2):
            #   lse   : consumed by the segment's tile i+1                -> 0x28
            #   dpsum : consumed by the segment's tile i+1                -> 0x38
            # Putting either statistic after an operand face serialises the
            # segment's own start behind a chain that the segment feeds:
            # segment(i+1) waits lse FULL -> lse waits Q_col EMPTY -> Q_col
            # waits dK(i) -> dK(i) waits dS(i) from segment(i). Hoisting both
            # statistics to the head of the body cuts that link by construction
            # and leaves the lead the producer has on them bounded only by the
            # segment's own release position, not by any chain's retirement.
            # Five waits per trip either way, which is D0.11's TMA budget exactly.
            m1 = m_min + i + 1
            m2 = m_min + i + 2
            if i + 1 < n_iter:
                _seg("tma.lse.empty.begin", i)
                _tma_empty_wait(mbar + _BAR_LSE_EMPTY, ph_t)
                _seg("tma.lse.empty.end", i)
                # The TMA atoms must NOT be wrapped in an election; the bulk
                # copies must be.
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + _BAR_LSE_FULL, _TX_STAT)
                    off_lse = cutlass.Int64(
                        (b * _H + h) * _N + m1 * _BLK_M) * 4
                    _bulk_g2s_512(_addr(sLse), _addr(mLseLog2) + off_lse,
                                  _addr(mbar + _BAR_LSE_FULL))
            if i + 1 < n_iter:
                _seg("tma.dps.empty.begin", i)
                _tma_empty_wait(mbar + _BAR_DPS_EMPTY, ph_t)
                _seg("tma.dps.empty.end", i)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + _BAR_DPS_FULL, _TX_STAT)
                    off_dps = cutlass.Int64(
                        (b * _H + h) * _N + m1 * _BLK_M) * 4
                    _bulk_g2s_512(_addr(sDps), _addr(mDpsum) + off_dps,
                                  _addr(mbar + _BAR_DPS_FULL))
            if i + 1 < n_iter:
                _seg("tma.qc.empty.begin", i)
                _tma_empty_wait(mbar_hi + _BAR2_QC_EMPTY, ph_t)
                _seg("tma.qc.empty.end", i)
                if rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_hi + _BAR2_QC_FULL, _TX_QC)
                _issue(p_qc, (rank, m1, h, b), lb_qc)
            if i + 2 < n_iter:
                _seg("tma.q.empty.begin", i)
                _tma_empty_wait(mbar + _BAR_Q_EMPTY, 1 - ph_t)
                _seg("tma.q.empty.end", i)
                if rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + _BAR_Q_FULL, _TX_Q)
                _issue(p_qr, (2 * m2 + rank, 0, h, b), lb_q)
            if i + 2 < n_iter:
                _seg("tma.do.empty.begin", i)
                _tma_empty_wait(mbar + _BAR_DO_EMPTY, 1 - ph_t)
                _seg("tma.do.empty.end", i)
                if rank == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + _BAR_DO_FULL, _TX_DO)
                _issue(p_dc, (rank, m2, h, b), lb_do)
                _issue(p_dr, (2 * m2 + rank, 0, h, b), lb_do)
            _seg("tma.supply.end", i)

    # ======== warp 12: the MMA issue ========
    if warp == _WARP_MMA:
        # ---- D0.6 chains 0 and 2: operands, fragments, accumulators -------------
        # D0.14's MMA row, read the way the decision spells it: the 67 descriptor
        # mirrors are LOOP-INVARIANT state of this warp — "若 student 的 MMA warp
        # 超 104，先查描述符是否在循环内重建". Every face this warp reads is single
        # stage (D0.8 has no ring), so the ten fragments below carry the same
        # addresses on every trip and are built ONCE, outside the trip loop and
        # outside every predicate; the prologue's three chains read the same ten,
        # so the warm-up builds nothing of its own either. The trip body is then
        # nothing but D0.13's eight waits, five chains of `cute.gemm` and seven
        # commits.
        # The alternative — each chain building its own operands inside the loop —
        # was measured on this tree and is worse on both axes it was supposed to
        # help: per-trip descriptor construction sits on the MMA warp's issue path
        # (device 1.4899 -> 1.7319 ms, tensor_pct 57.38 -> 51.78 at equal
        # correctness), and it does not close the register account either. Probe
        # `probe_d014_spill3` reads the in-loop tree's local-memory traffic
        # end to end: the PTX front end has no depot at all (all three entries
        # `depot=0B ld.local=0 st.local=0`), so every access is ptxas' own, and
        # SASS carries 7 LDL + 7 STL — the STL at `_alloc_face`'s
        # `allocate_tensor` (the faces' own pointers, spilled at their definition
        # in the kernel's common scope) and all 7 LDL inside `warp == _WARP_MMA`,
        # at the chain0 / chain1 / chain2 operand sites that rebuild from them.
        # The softmax and writer roles contribute none, so what is left of the
        # spill account is entirely this role's operand algebra: building it in
        # the loop moves which values ptxas spills (face pointers instead of
        # descriptors), not whether it spills.
        tiled_mma = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, _MMA_INST, tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        l_a = bh.make_smem_layout_a(tiled_mma, _MMA_TILER, cutlass.BFloat16, 1)
        l_b = bh.make_smem_layout_b(tiled_mma, _MMA_TILER, cutlass.BFloat16, 1)
        # A = this CTA's 128 KV rows of the pair's M=256; B = this CTA's 64 Q (dO)
        # rows of N=128. Both chains are K-major on both operands and share the two
        # layouts, because K and V carry the same face and so do Q_row and dO_row.
        frg_k = tiled_mma.make_fragment_A(_mma_view(sK, l_a))
        frg_v = tiled_mma.make_fragment_A(_mma_view(sV, l_a))
        frg_qr = tiled_mma.make_fragment_B(_mma_view(sQr, l_b))
        frg_dr = tiled_mma.make_fragment_B(_mma_view(sDOr, l_b))
        acc_layout = tiled_mma.make_fragment_C(
            tiled_mma.partition_shape_C(_MMA_INST[:2])).layout
        # TMEM pointer arithmetic is in 32-bit columns, so the column map's own
        # numbers are the offsets: S^T at 0x000, dP^T at 0x100 (D0.6).
        t_st = cute.make_tensor(tmem_ptr + _TMEM_ST, acc_layout)
        t_dp = cute.make_tensor(tmem_ptr + _TMEM_DP, acc_layout)
        n_kstep = cute.size(frg_k, mode=[2])       # 8 instructions per chain

        # ---- D0.9's two RS rows: chains 1 and 4, A read straight out of TMEM ----
        # "RS" on tcgen05 means the A operand is the TMEM the producing chain's
        # accumulator already sits in, read as packed bf16 — no smem round trip, no
        # P / dS^T copy (D0.9 forbids one outright). One geometry settles both:
        #
        #   chain4 dV : dV[KV][d] += P^T [KV][Q] . dO_col[Q][d]   A = tmem 0x000
        #   chain1 dK : dK[KV][d] += dS^T[KV][Q] . Q_col [Q][d]   A = tmem 0x100
        #
        # `OperandSource.TMEM`, A K-major (k contiguous inside a 32-bit column pair),
        # B an MN-major smem face — this CTA's 64 d columns of the pair's N=128 —
        # and an f32 accumulator that, unlike S^T / dP^T, lives ACROSS trips.
        tiled_rs = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, _MMA_INST, tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))
        l_b_mn = bh.make_smem_layout_b(tiled_rs, _MMA_TILER, cutlass.BFloat16, 1)
        frg_dc = tiled_rs.make_fragment_B(_mma_view(sDOc, l_b_mn))
        frg_qc = tiled_rs.make_fragment_B(_mma_view(sQc, l_b_mn))
        # The A operand is built from the SHAPE `partition_shape_A` reports, over a
        # bf16 recast of the TMEM pointer: a TMEM tensor has no smem descriptor to
        # hand `make_fragment_A`, and the register face of a TMEM operand is the
        # lane/column map itself — ((128,16),1,8):((65536,1),0,16), i.e. lane = this
        # CTA's KV row, k contiguous in bf16 units. Two bf16 per 32-bit column is
        # why P^T covers 0x00-0x3f where its f32 producer covered 0x00-0x7f.
        a_shape = tiled_rs.partition_shape_A((_MMA_TILER[0], _MMA_TILER[2]))
        a_frg_layout = tiled_rs.make_fragment_A(a_shape).layout
        a_pt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + _TMEM_PT, dtype=cutlass.BFloat16),
            a_frg_layout)
        a_dst = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + _TMEM_DST, dtype=cutlass.BFloat16),
            a_frg_layout)
        # An smem fragment carries a STAGE mode; a TMEM operand has nothing to
        # stage, so the A fragment may come back rank 3 or rank 4. Rank is a
        # trace-time integer, so this selection costs nothing at run time.
        a_staged = cute.rank(a_pt) == 4
        acc_rs_layout = tiled_rs.make_fragment_C(
            tiled_rs.partition_shape_C(_MMA_INST[:2])).layout
        t_dv = cute.make_tensor(tmem_ptr + _TMEM_DV, acc_rs_layout)
        t_dk = cute.make_tensor(tmem_ptr + _TMEM_DK, acc_rs_layout)
        n_kstep_rs = cute.size(frg_dc, mode=[2])   # 8 instructions per chain

        # ---- D0.9's smem row: chain3 dQ = dS . K_dsplit -------------------------
        # The one chain whose A operand changes orientation and crosses to the peer.
        # The segment holds dS TRANSPOSED (a TMEM lane is one of MY KV rows, a TMEM
        # column is one Q row of the q block), while chain3 wants A = dS[Q][KV] over
        # the PAIR's 256 KV — a real transpose plus a cross-CTA concatenation, which
        # is why D0.9 forces this one operand through smem while P^T / dS^T stay in
        # TMEM. Shape is D0.6's: M128 x N128 x K16, so the pair splits A by M (64 Q
        # rows each, which is exactly the 64-row face D0.8 gives each CTA) and the
        # accumulator by N (each CTA's 128 lanes hold all 128 Q rows of its own 64 d
        # columns — the map the probe MEASURED with a self-describing product).
        # Both operands are MN-major: A's contiguous axis is the 64 Q rows, B's the
        # 64 d columns.
        tiled_dq = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, _MMA_INST3, tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.MN, tcgen05.OperandMajorMode.MN))
        l_a3, l_b3, n_pass3 = _chain3_layouts(
            tiled_dq, cute.cosize(_face_layout("ds").outer))
        # One view per pass; the pass offset is in bf16 elements (16 KB each), which
        # is a whole number of swizzle periods.
        a3 = [cute.make_tensor(sDS.iterator + p * (64 * _BLK_N), l_a3.outer)
              for p in range(n_pass3)]
        b3 = [cute.make_tensor(sKd.iterator + p * (_D_HALF * _BLK_N), l_b3.outer)
              for p in range(n_pass3)]
        frg_a3 = [tiled_dq.make_fragment_A(v) for v in a3]
        frg_b3 = [tiled_dq.make_fragment_B(v) for v in b3]
        acc_dq_layout = tiled_dq.make_fragment_C(
            tiled_dq.partition_shape_C(_MMA_INST3[:2])).layout
        t_dq = cute.make_tensor(tmem_ptr + _TMEM_DQ, acc_dq_layout)
        n_kstep3 = cute.size(frg_a3[0], mode=[2])

        _seg("mma.entry")
        # The dS face starts out empty, and 0x68 is the only thing the segment
        # waits before writing it: one arrive per CTA here makes trip i's wait
        # phase i & 1 with no first-trip special case.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar + _BAR_DS_EMPTY)
        # K_dsplit is loaded once for the whole kernel, so its FULL barrier is
        # awaited once, here, on the leader (the only CTA eligible to wait a pair
        # barrier).
        _seg("mma.kdsplit.begin")
        if rank == 0:
            cute.arch.mbarrier_wait(mbar_hi + _BAR2_KD_FULL, 0)
        _seg("mma.kdsplit.end")

        # ---- D0.13 prologue: warm up one beat ----------------------------------
        # The body's trip i issues the chains of TWO tiles — S / dP / dV of tile
        # i+1 and dK / dQ of tile i — so tile 0's three leading chains have to be
        # issued before the body starts. That is the whole of the prologue: S(0),
        # dP(0), dV(0), each committing the same barriers it commits in the body,
        # so every row's flip k belongs to tile k with no first-trip special case.
        # The EMPTY commits are what the supply pipelines on: 0x08 released here
        # lets the TMA warp fetch Q_row(1) while the body is still on tile 0.
        if rank == 0:
            _seg("mma.pro.q.begin")
            cute.arch.mbarrier_wait(mbar + _BAR_Q_FULL, 0)
            _seg("mma.pro.q.end")
            _seg("mma.pro.chain0.begin")
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(n_kstep):
                cute.gemm(tiled_mma, t_st, frg_k[(None, None, kb, 0)],
                          frg_qr[(None, None, kb, 0)], t_st)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                _COMMIT_MCAST(mbar + _BAR_S_FULL)
                _COMMIT_MCAST(mbar + _BAR_Q_EMPTY)
            _seg("mma.pro.chain0.end")

            _seg("mma.pro.do.begin")
            cute.arch.mbarrier_wait(mbar + _BAR_DO_FULL, 0)
            _seg("mma.pro.do.end")
            _seg("mma.pro.chain2.begin")
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(n_kstep):
                cute.gemm(tiled_mma, t_dp, frg_v[(None, None, kb, 0)],
                          frg_dr[(None, None, kb, 0)], t_dp)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                _COMMIT_MCAST(mbar + _BAR_DP_FULL)
            _seg("mma.pro.chain2.end")

            # dV(0) is the pair's first write of 0x80-0xff, so it is the one
            # instruction of chain4 that carries D0.6's `UP0` overwrite; every
            # later trip accumulates into it.
            _seg("mma.pro.pready.begin")
            cute.arch.mbarrier_wait(mbar + _BAR_P_RDY, 0)
            _seg("mma.pro.pready.end")
            _seg("mma.pro.chain4.begin")
            tiled_rs.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(n_kstep_rs):
                if cutlass.const_expr(a_staged):
                    cute.gemm(tiled_rs, t_dv, a_pt[(None, None, kb, 0)],
                              frg_dc[(None, None, kb, 0)], t_dv)
                if cutlass.const_expr(not a_staged):
                    cute.gemm(tiled_rs, t_dv, a_pt[(None, None, kb)],
                              frg_dc[(None, None, kb, 0)], t_dv)
                tiled_rs.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                _COMMIT_MCAST(mbar + _BAR_DO_EMPTY)
            _seg("mma.pro.chain4.end")

        for i in cutlass.range(n_iter):
            ph_m = i % 2
            _seg("mma.operands.begin", i)
            if rank == 0:
                # D0.13's issue order. Trip i issues five chains belonging to
                # TWO tiles — S(i+1), dK(i), dP(i+1), dQ(i), dV(i+1) — so that
                # the segment's element-wise work on tile i+1 runs inside the
                # flight of dK(i) + dP(i+1) + dQ(i) instead of in front of them.
                # Each wait names a different tile, and that is the contract:
                # #1 0x00 tile i+1, #2 0x98 dQ(i-1), #4 0x130 tile i, #5 0x58
                # tile i, #7 0x10 tile i+1, #9 0x60 tile i+1, #10 0x160 tile i,
                # #12 0x48 tile i+1. The tile each wait names is the frozen part;
                # the DEPTH at which it is taken is D1.1's lever `_WAIT_TIER`,
                # which orders the trip's last two blocks (chain3 / chain4).
                # At tier "a" the in-flight depth before those eight waits is
                # 0,0,1,1,2,3,3,4 — nothing is ever drained inside the trip,
                # because every commit now goes to the consumer that waits it
                # (the operand faces' EMPTY rows) rather than back to the issuer.
                #
                # 0x98 guards tmem 0x40-0x7f, which chain0 overwrites as S^T's
                # second half and chain3 rewrites as its own accumulator
                # (D0.6's alias), so it is waited every trip — including the
                # last one, where no chain0 is issued but chain3 still is.
                _seg("mma.dqempty.begin", i)
                cute.arch.mbarrier_wait(mbar + _BAR_DQ_EMPTY, ph_m)
                _seg("mma.dqempty.end", i)

                # chain0: S^T(i+1) = K . Q_row^T, 8 x M256xN128xK16, D = tmem
                # 0x000, zero-initialised by the first instruction carrying
                # ACCUMULATE=False (D0.6: the accumulator is not live across
                # trips, unlike dV / dK). Its commit releases Q_row two tiles
                # ahead of the TMA warp's next load.
                if i + 1 < n_iter:
                    cute.arch.mbarrier_wait(mbar + _BAR_Q_FULL, 1 - ph_m)
                    _seg("mma.chain0.begin", i)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(n_kstep):
                        cute.gemm(tiled_mma, t_st, frg_k[(None, None, kb, 0)],
                                  frg_qr[(None, None, kb, 0)], t_st)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        _COMMIT_MCAST(mbar + _BAR_S_FULL)
                        _COMMIT_MCAST(mbar + _BAR_Q_EMPTY)
                    _seg("mma.chain0.end", i)

                # chain1: dK(i) += dS^T . Q_col, A = tmem 0x100 (dS^T's bf16
                # packing over dP^T's first half), B = Q_col MN-major. The A
                # operand is the segment's output of THIS tile, so the chain
                # waits 0x58 (dS^T READY, count 16: every softmax warp of both
                # CTAs) — the second of the reference's two exposed waits. The
                # accumulator is LIVE ACROSS TRIPS, so only the first trip
                # carries ACCUMULATE=False (D0.6's `UP0` overwrite); the flag is
                # a runtime predicate of the instruction, not a descriptor
                # field, which is why a dynamic value is its natural spelling.
                cute.arch.mbarrier_wait(mbar_hi + _BAR2_QC_FULL, ph_m)
                _seg("mma.dsready.begin", i)
                cute.arch.mbarrier_wait(mbar + _BAR_DS_RDY, ph_m)
                _seg("mma.dsready.end", i)
                _seg("mma.chain1.begin", i)
                tiled_rs.set(tcgen05.Field.ACCUMULATE, i > 0)
                for kb in cutlass.range_constexpr(n_kstep_rs):
                    if cutlass.const_expr(a_staged):
                        cute.gemm(tiled_rs, t_dk, a_dst[(None, None, kb, 0)],
                                  frg_qc[(None, None, kb, 0)], t_dk)
                    if cutlass.const_expr(not a_staged):
                        cute.gemm(tiled_rs, t_dk, a_dst[(None, None, kb)],
                                  frg_qc[(None, None, kb, 0)], t_dk)
                    tiled_rs.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    _COMMIT_MCAST(mbar_hi + _BAR2_QC_EMPTY)
                _seg("mma.chain1.end", i)

                # chain2: dP^T(i+1) = V . dO_row^T, same form, D = tmem 0x100.
                # It overwrites dS^T(i)'s bf16 columns, which is safe by ISSUE
                # ORDER and not by a barrier: dK(i) above is already in the
                # pipe and tcgen05 retires a CTA's instructions in order
                # (D0.9's cost line).
                if i + 1 < n_iter:
                    cute.arch.mbarrier_wait(mbar + _BAR_DO_FULL, 1 - ph_m)
                    _seg("mma.chain2.begin", i)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(n_kstep):
                        cute.gemm(tiled_mma, t_dp, frg_v[(None, None, kb, 0)],
                                  frg_dr[(None, None, kb, 0)], t_dp)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        _COMMIT_MCAST(mbar + _BAR_DP_FULL)
                    _seg("mma.chain2.end", i)

                # D1.1's tier: the trip's last two blocks, issued in the order
                # `_TIER_ORDER` names. Everything above this point is fixed by
                # D0.13 — chain0 and chain2 must bracket chain1 (dP^T(i+1)
                # overwrites dS^T(i)'s columns and is safe only BEHIND dK(i) in
                # issue order), and chain1 must follow chain0 so that S(i+1) is
                # the oldest chain in flight. chain3 and chain4 are the two that
                # commute: they touch disjoint TMEM (0x40-0x7f vs 0x00-0x3f),
                # disjoint smem faces and disjoint barriers, so swapping them
                # changes only which of 0x48 and 0x60/0x160 is taken deeper into
                # the trip. The loop is unrolled at trace time, so exactly one
                # arm of each test is emitted and tier "a" traces byte for byte
                # what the straight-line D0.13 order traced.
                for slot in cutlass.range_constexpr(2):
                    chain = _TIER_ORDER[slot]

                    # chain3: dQ(i) = dS . K_dsplit, A = the exchanged dS face at
                    # 0x2c400 (MN-major), B = K_dsplit, D = tmem 0x40-0x7f. Two
                    # waits left to take here, and they guard different things:
                    # 0x60 says the softmax warps are past their LDTM of S^T(i+1)
                    # — the tile chain0 just produced, whose second half this
                    # accumulator ALIASES — and 0x160 says both CTAs' faces carry
                    # the peer's dS half of tile i (the local half and the tmem
                    # dS^T rode 0x58, waited before chain1). The accumulator is
                    # not live across trips — every trip computes the dQ of a
                    # different q block — so ACCUMULATE is False on the first
                    # instruction of every trip, unlike dK / dV.
                    if cutlass.const_expr(chain == 3):
                        _seg("mma.sdone.begin", i)
                        cute.arch.mbarrier_wait(mbar + _BAR_S_DONE, ph_m)
                        _seg("mma.sdone.end", i)
                        _seg("mma.dsdone.begin", i)
                        cute.arch.mbarrier_wait(mbar_hi + _BAR2_DS_DONE, ph_m)
                        _seg("mma.dsdone.end", i)
                        _seg("mma.chain3.begin", i)
                        tiled_dq.set(tcgen05.Field.ACCUMULATE, False)
                        for p in cutlass.range_constexpr(n_pass3):
                            for kb in cutlass.range_constexpr(n_kstep3):
                                cute.gemm(tiled_dq, t_dq,
                                          frg_a3[p][(None, None, kb, 0)],
                                          frg_b3[p][(None, None, kb, 0)], t_dq)
                                tiled_dq.set(tcgen05.Field.ACCUMULATE, True)
                        # 0x68 releases the face in BOTH CTAs (each one's softmax
                        # warps write their own copy next trip), so the commit is
                        # multicast — and so is 0x90, for the same reason on the
                        # other side: each CTA's writer warps drain their own half
                        # of the accumulator and can only be woken by an arrive in
                        # their own smem.
                        with cute.arch.elect_one():
                            _COMMIT_MCAST(mbar + _BAR_DS_EMPTY)
                            _COMMIT_MCAST(mbar + _BAR_DQ_FULL)
                        _seg("mma.chain3.end", i)

                    # chain4: dV(i+1) += P^T . dO_col, A = tmem 0x000 (P^T's bf16
                    # packing over S^T's first half), B = dO_col MN-major. Waits
                    # 0x48 (P^T READY, tile i+1) — the reference's LARGEST
                    # exposure. Under tier "a" it is the trip's last wait and is
                    # taken with four chains in flight, which is the point of
                    # putting it there. Its commit releases dO, whose two faces
                    # chain2 and chain4 have both now read.
                    if cutlass.const_expr(chain == 4):
                        if i + 1 < n_iter:
                            _seg("mma.pready.begin", i)
                            cute.arch.mbarrier_wait(mbar + _BAR_P_RDY, 1 - ph_m)
                            _seg("mma.pready.end", i)
                            _seg("mma.chain4.begin", i)
                            tiled_rs.set(tcgen05.Field.ACCUMULATE, True)
                            for kb in cutlass.range_constexpr(n_kstep_rs):
                                if cutlass.const_expr(a_staged):
                                    cute.gemm(tiled_rs, t_dv,
                                              a_pt[(None, None, kb, 0)],
                                              frg_dc[(None, None, kb, 0)], t_dv)
                                if cutlass.const_expr(not a_staged):
                                    cute.gemm(tiled_rs, t_dv,
                                              a_pt[(None, None, kb)],
                                              frg_dc[(None, None, kb, 0)], t_dv)
                                tiled_rs.set(tcgen05.Field.ACCUMULATE, True)
                            with cute.arch.elect_one():
                                _COMMIT_MCAST(mbar + _BAR_DO_EMPTY)
                            _seg("mma.chain4.end", i)
            _seg("mma.operands.end", i)

        # ---- D0.15: the accumulators are final ---------------------------------
        # Both rows are committed by the LEADER's MMA warp and multicast, because
        # each CTA drains its own half of the pair's accumulator and can only be
        # woken by an arrive in its own smem. A `tcgen05.commit` signals after every
        # instruction this CTA has issued has completed, so the flip of 0x70 is also
        # the statement that chain0 / chain2 are retired and the K / V faces have no
        # reader left — which is the whole of D0.15 reading (a)'s safety argument
        # for staging the epilogue on top of them.
        if rank == 0:
            _seg("mma.dvempty.begin")
            cute.arch.mbarrier_wait(mbar + _BAR_DV_EMPTY, 0)
            _seg("mma.dvempty.end")
            with cute.arch.elect_one():
                _COMMIT_MCAST(mbar + _BAR_DV_FINAL)
            _seg("mma.dkempty.begin")
            cute.arch.mbarrier_wait(mbar + _BAR_DK_EMPTY, 0)
            _seg("mma.dkempty.end")
            with cute.arch.elect_one():
                _COMMIT_MCAST(mbar + _BAR_DK_FINAL)

    # ======== warp 14: the dS exchange relay ========
    if warp == _WARP_RELAY:
        for i in cutlass.range(n_iter):
            ph_r = i % 2
            # ---- D0.7's relay warp: one job, 24 registers ----------------------
            _seg("relay.wait.begin", i)
            cute.arch.mbarrier_wait(mbar_hi + _BAR2_DS_PEER, ph_r)
            _seg("relay.wait.end", i)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(
                    cute.arch.map_dsmem_ptr(mbar_hi + _BAR2_DS_DONE, 0))

    # ======== warps 0-7: the softmax / dS segment, then the dK / dV epilogue ========
    if warp < _WARP_WRITER:
        # The DSMEM leg needs a PLAIN integer base for each dS face, and a swizzled
        # iterator is not obliged to hand one back. `sLse` is the first UNSWIZZLED
        # array after the two faces and every region in between is a 1 KB multiple
        # asking at most 1 KB of alignment, so the allocator cannot pad between them
        # and its own arithmetic is D0.8's fixed distances subtracted from sLse.
        a_ds = _addr(sLse) - cutlass.Int32(_SM_DS)
        a_send = _addr(sLse) - cutlass.Int32(_SM_DS + _SM_K_DSPLIT + _SM_DS_SEND)

        # ---- D0.11: the softmax segment's view of TMEM -------------------------
        # Geometry is D0.6's: a TMEM lane is one of THIS CTA's 128 KV rows and a
        # TMEM column is one Q row of the 128-row q block, columns 0-63 being the
        # leader's Q rows and 64-127 the follower's (that is how D0.8 slices the
        # Q_row / dO_row faces, tile `2m + rank`). Two warpgroups split the 128
        # columns 64/64 and each covers all 128 lanes.
        #
        # The read is `Ld32x32b.x32` over 32 f32 columns, TWICE per operand, and the
        # write back is `St32x32b.x32` over the 32 32-bit columns the 64 packed
        # values occupy (hw_kb `tmem_r2t_recast_fragment`'s W = E*T/32 with E=64,
        # T=16). The readback of a TMEM round trip cannot by itself prove the
        # coordinate map — t2r and r2t are the same op family and cancel each
        # other's permutation — which is why the certifying probe pinned it against
        # a live UMMA producer and an absolute-coordinate mask instead.
        #
        # D0.14 is why the read is sliced. D0.13's interleave forces P's fp32 copy
        # to stay live from segment B to segment D (the reference does not recompute
        # it and does not read it back from TMEM), so the segment's non-accumulator
        # working set is P's 64 registers plus whatever transient is being read;
        # a 64-wide transient puts the peak at 128+ and the budget is 124. Two
        # 32-column slices put it at 96 + addresses, which is the whole of the
        # step: the accumulators never leave TMEM, so registers only ever hold one
        # slice of one operand plus P.
        #
        # All 512 threads build this algebra (it is trace-time, and `wg`/`lt` are
        # runtime values); only warps 0-7 ever issue the copies.
        wg = tidx // _WG_THREADS             # warpgroup: 0 / 1 for warps 0-7
        lt = tidx % _WG_THREADS              # thread in the WG == the TMEM lane
        col0 = wg * _WG_COLS                 # this WG's first f32 column
        c32_0 = wg * (_WG_COLS // 2)         # ... and its first 32-bit column
        t_s_h = [cute.make_tensor(
            tmem_ptr + _TMEM_ST + col0 + hf * _HALF_COLS, cute.make_layout(
                (_BLK_N, _HALF_COLS), stride=(_TMEM_LANE_STRIDE, 1)))
            for hf in range(2)]
        t_dp_h = [cute.make_tensor(
            tmem_ptr + _TMEM_DP + col0 + hf * _HALF_COLS, cute.make_layout(
                (_BLK_N, _HALF_COLS), stride=(_TMEM_LANE_STRIDE, 1)))
            for hf in range(2)]
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32),
            t_s_h[0])
        thr_l = t2r.get_slice(lt)
        src_s = [thr_l.partition_S(t) for t in t_s_h]
        src_dp = [thr_l.partition_S(t) for t in t_dp_h]
        id_l = thr_l.partition_D(
            cute.make_identity_tensor((_BLK_N, _HALF_COLS)))
        r_h = cute.make_rmem_tensor(id_l.shape, cutlass.Float32)   # ONE slice
        n_h = cute.size(r_h)
        # Index tails: a partitioned fragment's trailing modes are all singletons,
        # and their rank is a trace-time integer.
        ix_l = _frag_index(r_h)
        ix_i = _frag_index(id_l)
        _slice_in_order(id_l, ix_i, n_h, "segment t2r slice")
        # P (then dS) lives here, indexed by the Q COLUMN of this warpgroup's 64.
        r_p = _rmem_line(_WG_COLS, cutlass.Float32)
        # The packed bf16 faces: P^T over S^T's first half, dS^T over dP^T's.
        t_pt_wg = cute.make_tensor(tmem_ptr + _TMEM_PT + c32_0, cute.make_layout(
            (_BLK_N, _WG_COLS // 2), stride=(_TMEM_LANE_STRIDE, 1)))
        t_dst_wg = cute.make_tensor(tmem_ptr + _TMEM_DST + c32_0, cute.make_layout(
            (_BLK_N, _WG_COLS // 2), stride=(_TMEM_LANE_STRIDE, 1)))
        r2t = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition.x32, tcgen05.Unpack.NONE),
                cutlass.Float32),
            t_pt_wg)
        thr_s = r2t.get_slice(lt)
        dst_pt = thr_s.partition_D(t_pt_wg)
        dst_dst = thr_s.partition_D(t_dst_wg)
        r_pk32 = cute.make_rmem_tensor(thr_s.partition_S(t_pt_wg).shape,
                                       cutlass.Float32)
        r_pk = cute.recast_tensor(r_pk32, cutlass.BFloat16)
        n_pk = cute.size(r_pk32)
        # The same storage wearing its 32-bit slot face: `n_pk` slots, slot k
        # holding Q columns (2k, 2k+1). This is the face the packing writes
        # (kb `cvt_f32x2_bf16x2_pack_into_r2t_fragment`: the convert's product
        # is a whole word, so the word is the addressing unit); `cute.copy`
        # below is still handed the f32 face the copy atom was built on.
        pk_i32 = cute.recast_tensor(r_pk32, cutlass.Int32)
        ix_p32 = _frag_index(pk_i32)
        id_s = thr_s.partition_D(
            cute.make_identity_tensor((_BLK_N, _WG_COLS // 2)))
        _slice_in_order(id_s, _frag_index(id_s), n_pk, "segment r2t")
        C_LOG2 = cutlass.Float32(_SCALE_LOG2)
        PARK = cutlass.Float32(_PARK)
        # D0.5's causal predicate, as the threshold it is. A thread's 64 values
        # are one Q-contiguous line at ONE of its KV rows — the same property
        # the segment's smem dS row already rests on (the transpose below is
        # free precisely because the row does not move) — so over the run of
        # columns the valid set is the suffix `q >= kv`, i.e. `c >= t` with
        #     t = (kv_row0 + my row) - (q_row0 + col0)
        # and t moving once per trip, not once per column. The row itself is
        # trip-invariant, so its absolute value is hoisted out of the loop here;
        # `_slice_in_order` above has already pinned element j to column j,
        # which is what makes the column index a compile-time constant and the
        # bit tests a run of immediates against one register.
        kv_abs = kv_row0 + id_l[ix_i(0)][0]
        _one_mask_word(n_h, "segment t2r slice")   # one word per half
        # D0.14: the smem row is the PACKING's own registers. A lane's 64 bf16 dS
        # values at its KV row are one 128 B swizzle row of a dS face AND the 32
        # 32-bit columns the STTM writes — the same values in the same Q-column
        # order (both fragments checked in order above) — so the segment carries one
        # bf16 staging, not one per consumer. `r_line` is only the fallback for a
        # DSL build that will not re-view register storage.
        pk_line = _flat_view(r_pk, _WG_COLS)
        r_line = _rmem_line(_WG_COLS, cutlass.BFloat16)
        ds_line = _pick(pk_line, r_line)
        # When the flat view took, ONE pass of `_pack_pairs` (D0.4 (3)) fills
        # the STTM's register face and the smem row at the same time, because
        # they are the same 64 values in the same order. When it did not, the
        # two are separate storage and the STTM face is packed a second time.
        ds_i32, ix_d32 = _pack_dst(pk_line, pk_i32, ix_p32, r_line)
        pk_ok = pk_line is not None

        # D0.11's count-16 rows live in the leader's smem too: both CTAs' softmax
        # warps arrive them, and the leader's MMA warp is the only waiter.
        lb_prdy = cute.arch.map_dsmem_ptr(mbar + _BAR_P_RDY, 0)
        lb_dsrdy = cute.arch.map_dsmem_ptr(mbar + _BAR_DS_RDY, 0)
        lb_sdone = cute.arch.map_dsmem_ptr(mbar + _BAR_S_DONE, 0)
        # ... and so do D0.15's two staging EMPTY rows.
        lb_dvempty = cute.arch.map_dsmem_ptr(mbar + _BAR_DV_EMPTY, 0)
        lb_dkempty = cute.arch.map_dsmem_ptr(mbar + _BAR_DK_EMPTY, 0)

        # D0.15's staging rows start out empty too: the K / V faces are not epilogue
        # staging until the loop ends, so the MMA warp's one acquire of each has to
        # pass without a producer. 8 softmax warps x 2 CTAs = the rows' count of 16.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(lb_dvempty)
            cute.arch.mbarrier_arrive(lb_dkempty)

        for i in cutlass.range(n_iter):
            ph_s = i % 2
            # ---- D0.11 / D0.14: the softmax segment (warps 0-7) ----------------
            # Per trip, in D0.11's order, with D0.14's slicing:
            #   WAIT lse(0x20) -> WAIT S(0x40) -> 2 x [LDTM 32 cols of S^T ->
            #   select(causal) -> FFMA(S * scale_log2 - lse_log2) -> exp2 -> P]
            #   -> arrive 0x60 (after the second LDTM), 0x28 -> BAR(0x3)
            #   -> pack bf16 -> STTM P^T -> fence -> arrive 0x48
            #   WAIT dpsum(0x30), dP(0x50) -> 2 x [LDTM 32 cols of dP^T ->
            #   (dP - dpsum) * P in place] -> arrive 0x38 -> BAR(0x3)
            #   -> pack bf16
            #   -> WAIT 0x68 -> STTM dS^T -> fence -> STS the SAME packed bf16 as
            #   one dS row (local half | send staging) -> proxy fence -> BAR(0x3)
            #   -> arrive 0x58
            #   -> [elected] remote expect-tx on the peer's 0x158 + UBLKCP.S.S 16 KB
            # That is 3 BAR(0x3) and 5 blocking mbarrier waits per warpgroup per
            # trip (0x20, 0x40, 0x30/0x50, 0x68), inside D0.11's rendezvous budget
            # of 5 + 4. Live registers across it: P's 64 fp32 from the first exp2
            # to the last pack, one 32-wide slice of whichever operand is being
            # read, and the 32 the packing occupies — D0.14's account, with the
            # accumulators all still in TMEM.
            # D0.4 puts the whole of it in the log2 domain: preprocess already
            # produced lse_log2, `scale_log2` is a host-folded constant, and the
            # score arrives UNSCALED out of chain0, so one FFMA + one MUFU.EX2 per
            # element is the entire exp path and there is no base change anywhere.
            # The mask is D0.5's: a per-element predicate select on the absolute
            # (q, kv) pair, no diagonal / interior split and no branch.
            _seg("softmax.stats.begin", i)
            cute.arch.mbarrier_wait(mbar + _BAR_LSE_FULL, ph_s)
            _seg("softmax.stats.end", i)
            _seg("softmax.s.begin", i)
            cute.arch.mbarrier_wait(mbar + _BAR_S_FULL, ph_s)
            q_row0 = (m_min + i) * _BLK_M
            # The trip's whole causal mask: ONE threshold and TWO words per
            # lane, computed before either half is read. Bit c of `w_msk[0]` is
            # this lane's column c (of the warpgroup's 64), bit c of `w_msk[1]`
            # its column 32 + c — the second word is the SAME `_ones_from` on
            # `t - 32`, the shift's clamping absorbing which word the threshold
            # falls in, so there is no branch and no second comparison (hw_kb
            # `threshold_bitmap_predicate_select` gotcha 2). D0.14 segment A's
            # "mask 谓词位图，不展开成寄存器" is these two registers: the
            # per-column predicate is never materialised, only tested.
            t_msk = kv_abs - (q_row0 + col0)
            w_msk = (_ones_from(t_msk), _ones_from(t_msk - _HALF_COLS))
            # D0.14 segment A+B, twice over 32 columns: read a slice, mask it,
            # fold scale and lse in the log2 domain, exp2 it, and leave the
            # result in P's fp32 line. The slice's registers die at the end of
            # each half, so the peak is P's 64 plus one slice's 32.
            for hf in cutlass.range_constexpr(2):
                cute.copy(t2r, src_s[hf], r_h)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(hf == 1):
                    # All of S^T is out of TMEM: 0x40-0x7f may be reused by
                    # chain3's dQ accumulator (D0.6's alias), which is what
                    # waits this row. D0.13's trip i waits the arrive of trip
                    # i+1's segment, so the FIRST trip's arrive is skipped and
                    # one is issued after the loop instead — that is what keeps
                    # every flip of this row waited exactly once, in order.
                    if i > 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(lb_sdone)
                base = hf * _HALF_COLS
                w = w_msk[hf]
                for j in cutlass.range_constexpr(n_h):
                    col = col0 + base + j
                    # One bit test + one select per column, and nothing else:
                    # the column number is a compile-time constant folded into
                    # the mask immediate, so the `base + c` address arithmetic
                    # the absolute-coordinate form needed (one `VIADD` per
                    # column) is gone with it.
                    sc = _sel_bit(w & _imm(1 << j), r_h[ix_l(j)], PARK)
                    r_p[base + j] = _EXP2(sc * C_LOG2 - sLse[col])
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mbar + _BAR_LSE_EMPTY)
            # Every softmax thread of the CTA must be past its LDTM before any
            # of them packs: P^T aliases S^T's FIRST half, so warpgroup 1's
            # 32-bit columns 32-63 are warpgroup 0's f32 columns 64-127's
            # destination and warpgroup 0's f32 source columns at the same time.
            _NAMED_BAR(_SOFTMAX_BAR, _SOFTMAX_THREADS)
            # D0.4 (3): P^T is packed a pair at a time, one convert per 32-bit
            # column of the STTM fragment. P's line has to survive this (it is
            # segment D's multiplicand), so what the pairwise form buys here is
            # the packing's own 32 registers filling one slot at a time instead
            # of the whole 64-wide source being read as one value first.
            _pack_pairs(pk_i32, ix_p32, r_p, n_pk)
            cute.copy(r2t, r_pk32, dst_pt)
            cute.arch.fence_view_async_tmem_store()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(lb_prdy)
            _seg("softmax.s.end", i)

            _seg("softmax.dp.begin", i)
            cute.arch.mbarrier_wait(mbar + _BAR_DPS_FULL, ph_s)
            cute.arch.mbarrier_wait(mbar + _BAR_DP_FULL, ph_s)
            # D0.14 segment D, the peak segment, twice over 32 columns:
            # dS = P * (dP - dpsum), fp32 (D0.4 (4)), computed IN PLACE over P's
            # line so that no third array exists. dpsum is indexed by the Q row,
            # which is the COLUMN of the transposed accumulator. Masked elements
            # carry P = 0 and so contribute nothing to dK / dQ.
            for hf in cutlass.range_constexpr(2):
                cute.copy(t2r, src_dp[hf], r_h)
                cute.arch.fence_view_async_tmem_load()
                base = hf * _HALF_COLS
                for j in cutlass.range_constexpr(n_h):
                    r_p[base + j] = (r_h[ix_l(j)]
                                     - sDps[col0 + base + j]) * r_p[base + j]
            # D0.13 s2: the dpsum stage is free the instant this warp's LAST
            # broadcast read of it has landed — which is here, the subtraction
            # above being its only consumer. Releasing it any later (at the
            # segment's closing rendezvous, next to 0x58) costs the producer a
            # whole segment of lead: 0x38 is the only thing the TMA warp waits
            # before re-filling the stage for tile i+1, and the stage is single
            # (D0.10), so a late arrive is a late FULL is a late segment start.
            # Counterpart of s1's arrive on 0x28 after the exp2 line above.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mbar + _BAR_DPS_EMPTY)
            # Every softmax thread must be past its LDTM of dP^T before any of
            # them writes dS^T back over its first half — the same aliasing the
            # P^T packing has, one warpgroup's destination being the other's
            # source.
            _NAMED_BAR(_SOFTMAX_BAR, _SOFTMAX_THREADS)
            # D0.4 (3) again, for dS^T — and this is the site the pairwise form
            # is for: P dies here, so each pair's two fp32 are dead the instant
            # their word exists. Read as one line instead, all 64 stay live
            # until the last word is written and the segment's peak carries 64 +
            # 32 at once, which is D0.14's margin of 12 exceeded. `ds_line` is
            # the smem row's source and is the STTM fragment's own storage
            # whenever the flat view took, so one pass serves both; the second
            # pass only runs on a build where they are separate.
            _pack_pairs(ds_i32, ix_d32, r_p, n_pk)
            if cutlass.const_expr(not pk_ok):
                _pack_pairs(pk_i32, ix_p32, r_p, n_pk)
            _seg("softmax.dsempty.begin", i)
            cute.arch.mbarrier_wait(mbar + _BAR_DS_EMPTY, ph_s)
            _seg("softmax.dsempty.end", i)
            cute.copy(r2t, r_pk32, dst_dst)
            cute.arch.fence_view_async_tmem_store()
            # ---- D0.9's smem row: the same dS, transposed into the face -----
            # A thread's 64 values are one Q-contiguous line at ONE of my KV
            # rows, so the transpose is free: the face is MN-major, the line is
            # its mode 0, and the store is 8 x STS.128 across one swizzle row.
            # Which face it lands in is the warpgroup: WG `rank` holds MY Q
            # rows (accumulator columns 64*rank .. +64) and writes the local
            # half of my own face at the k-block my KV rows own; WG `1-rank`
            # holds the PEER's Q rows and writes the send staging, which the
            # bulk copy lands in the peer's face at that same k-block.
            _seg("softmax.dsx.begin", i)
            if wg == rank:
                _VEC_COPY(ds_line, sDS[(None, rank * _BLK_N + lt)])
            else:
                _VEC_COPY(ds_line, sDSs[(None, lt)])
            # a generic STS is not visible to the async proxy the UMMA
            # descriptor and the bulk engine read through
            _PROXY_FENCE()
            with cute.arch.elect_one():
                # 0x58 certifies BOTH halves of D0.11's row: tmem 0x100-0x13f
                # and the local half of the smem face. Its count is 16 — every
                # softmax warp of the PAIR arrives it — so a warp may arrive as
                # soon as its OWN STTM and its OWN store have been fenced; the
                # row cannot flip until all sixteen have. Arriving it behind the
                # segment's closing rendezvous instead costs the MMA warp the
                # slowest warp's remaining segment at exposure point 2 (0x58,
                # 69 spins/CTA) for a guarantee the row already carries.
                cute.arch.mbarrier_arrive(lb_dsrdy)
            # What the rendezvous IS still needed for: warp 0's bulk copy below
            # reads the send staging, which the OTHER warpgroup wrote.
            _NAMED_BAR(_SOFTMAX_BAR, _SOFTMAX_THREADS)
            # ---- the cross-CTA leg: 16 KB of DSMEM, no gmem round trip ------
            # The sender arms the RECEIVER's 0x158 (one instruction carries the
            # arrive and the whole 16 KB transaction count) and then issues the
            # copy against it; the receiver's relay warp turns "it landed" into
            # the count-2 signal the leader's MMA warp waits before chain3.
            if warp == _WARP_SOFTMAX:
                with cute.arch.elect_one():
                    pb = _mapa(_addr(mbar_hi + _BAR2_DS_PEER), peer)
                    _expect_tx_at(pb, _SM_DS_SEND)
                    _bulk_s2s(_mapa(a_ds, peer) + _SM_DS_SEND * rank,
                              a_send, _SM_DS_SEND, pb)
            _seg("softmax.dsx.end", i)
            _seg("softmax.dp.end", i)

        # D0.13's last trip issues chain3 dQ(N-1) behind a 0x60 whose producer —
        # the segment's LDTM of S(N), a tile that does not exist — never runs. The
        # reference closes the row the same way: the segment skips the arrive on its
        # first trip and issues one here, so the row flips exactly N times, the MMA
        # warp waits all N in order, and no phase is ever consumed out of turn.
        _seg("softmax.sdone.tail.begin")
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(lb_sdone)
        _seg("softmax.sdone.tail.end")

        # D0.14: the epilogue's own algebra is built HERE, after the trip
        # loop, not before it. Its views are dead for every one of the
        # n_iter trips, and anything defined ahead of the loop is live
        # across it — which is the whole reason this kernel's roles now
        # carry their own scopes (see the dispatch note above).
        # ---- D0.15: the dK / dV epilogue's three views -------------------------
        # (1) TMEM. dV at 0x080 and dK at 0x180 are M256xN128 `cta_group::2`
        #     accumulators, so each CTA holds its OWN 128 KV rows over all 128 d as
        #     128 lanes x 128 f32 columns — lane = KV row, column = d. That is the
        #     same geometry the segment reads S^T / dP^T through, so the epilogue
        #     reuses the segment's own slice atom and thread slice: warpgroup `wg`
        #     takes the 64 d columns at `col0`, in two 32-column halves.
        # (2) smem. The staging is the K / V faces themselves (D0.15 reading (a)):
        #     both are K_SW128 (128, 128) bf16, i.e. exactly two of the (128, 64)
        #     "dkv" boxes laid end to end, and 64 columns of bf16 = 16 KB is a whole
        #     number of swizzle periods, so the half at element offset 8192 wears
        #     the same address function as the half at 0. dV stages over V, dK over
        #     K; neither has a reader left once its FINAL row has flipped, because
        #     the commit that flips it is ordered behind every S^T / dP^T chain.
        # (3) gmem. One TMA store box per warpgroup per tensor: coordinate
        #     (n_blk, wg, h, b) on the (N, D, H, B) view = KV rows [128*n_blk, +128)
        #     x d columns [64*wg, +64). Two stores per tensor per CTA, which is
        #     D0.15's `UTMASTG.4D` census.
        t_dv_h = [cute.make_tensor(
            tmem_ptr + _TMEM_DV + col0 + hf * _HALF_COLS, cute.make_layout(
                (_BLK_N, _HALF_COLS), stride=(_TMEM_LANE_STRIDE, 1)))
            for hf in range(2)]
        t_dk_h = [cute.make_tensor(
            tmem_ptr + _TMEM_DK + col0 + hf * _HALF_COLS, cute.make_layout(
                (_BLK_N, _HALF_COLS), stride=(_TMEM_LANE_STRIDE, 1)))
            for hf in range(2)]
        src_epi = [[thr_l.partition_S(t) for t in t_dv_h],
                   [thr_l.partition_S(t) for t in t_dk_h]]
        l_epi = _face_layout("dkv")
        s_epi = [[cute.make_tensor(sV.iterator + w * (_BLK_N * _D_HALF),
                                   l_epi.outer) for w in range(2)],
                 [cute.make_tensor(sK.iterator + w * (_BLK_N * _D_HALF),
                                   l_epi.outer) for w in range(2)]]
        p_epi = [[_partition(a_sdv, g_sdv, s_epi[0][w], "dkv") for w in range(2)],
                 [_partition(a_sdk, g_sdk, s_epi[1][w], "dkv") for w in range(2)]]
        # dV is NOT scaled and dK is (D0.4's scale-position clause: dQ's scale is
        # postprocess's, dK's is here, dV has none).
        SC_DK = cutlass.Float32(_SCALE)

        # ---- D0.15: the dK / dV epilogue on the softmax warps ------------------
        # Per tensor: `WAIT FINAL -> 2 x LDTM.x32 of this warpgroup's 64 f32 columns
        # -> (dK only) x softmax_scale -> F2FP -> 8 x STS.128 into the staging face
        # -> proxy fence -> BAR(0x3, 256) -> [one warp per WG] UTMASTG.4D -> commit
        # -> arrive the staging EMPTY row`, and one `wait_group 0` at the end so no
        # store is still reading smem when the CTA retires. No per-element `STG`
        # anywhere: the path out is a matrix atom plus a TMA box, which is what
        # keeps `smem_wavefronts_st` on D0.8's atom line.
        # The 64 bf16 a lane produces are one 128 B swizzle row of a K-major face at
        # its own KV row, so the staging store is the same single vectorized copy
        # the dS row uses, through the same one bf16 staging line (D0.14).
        for which in cutlass.range_constexpr(2):
            nm = ("dv", "dk")[which]
            _seg("epi.%s.final.begin" % nm)
            cute.arch.mbarrier_wait(
                mbar + (_BAR_DV_FINAL, _BAR_DK_FINAL)[which], 0)
            _seg("epi.%s.final.end" % nm)
            _seg("epi.%s.read.begin" % nm)
            # The two TMEM halves land in the segment's own fp32 line — dead
            # since the last trip — and D0.4's scale for dK is applied there, in
            # fp32, BEFORE the rounding. The line is then packed once (D0.4 (3),
            # the same `_pack_pairs` the two in-loop sites use): the staging
            # store wants 64 bf16 in Q-column order, which is what the slots
            # hold, two adjacent columns each.
            for hf in cutlass.range_constexpr(2):
                cute.copy(t2r, src_epi[which][hf], r_h)
                cute.arch.fence_view_async_tmem_load()
                base = hf * _HALF_COLS
                if cutlass.const_expr(which == 0):
                    for j in cutlass.range_constexpr(n_h):
                        r_p[base + j] = r_h[ix_l(j)]
                if cutlass.const_expr(which == 1):
                    for j in cutlass.range_constexpr(n_h):
                        r_p[base + j] = r_h[ix_l(j)] * SC_DK
            _pack_pairs(ds_i32, ix_d32, r_p, n_pk)
            if wg == 0:
                _VEC_COPY(ds_line, s_epi[which][0][(lt, None)])
            else:
                _VEC_COPY(ds_line, s_epi[which][1][(lt, None)])
            _PROXY_FENCE()
            _NAMED_BAR(_SOFTMAX_BAR, _SOFTMAX_THREADS)
            _seg("epi.%s.read.end" % nm)
            _seg("epi.%s.store.begin" % nm)
            # One warp per warpgroup issues its half tile. The TMA atom elects
            # internally, so the copy must NOT be wrapped in `elect_one` — and
            # the group bookkeeping must not be either: `commit_group` /
            # `wait_group` track the ISSUING THREAD's own copies, and a separate
            # election is not obliged to land on the same lane the atom picked.
            # Running them warp-wide makes the lane that issued the store the
            # lane that commits and waits it; every other lane commits an empty
            # group, which retires at once.
            if warp % 4 == 0:
                if wg == 0:
                    _issue_store(p_epi[which][0], (n_blk, 0, h, b))
                else:
                    _issue_store(p_epi[which][1], (n_blk, 1, h, b))
                _BULK_COMMIT()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(
                    (lb_dvempty, lb_dkempty)[which])
            _seg("epi.%s.store.end" % nm)
        # `.read` would only say the staging has been read; what has to be true
        # before this CTA lets go of its smem and its slot is that the bytes are
        # in gmem, so the tail wait is the plain one.
        _seg("epi.drain.begin")
        if warp % 4 == 0:
            _BULK_WAIT_ALL(0)
        _seg("epi.drain.end")

    # ======== warps 8-11: the dQ writer ========
    if warp >= _WARP_WRITER:
        if warp < _WARP_MMA:
            # ---- D0.12: the dQ-writer's view of the same TMEM ---------------
            # chain3's accumulator is 128 lanes x 64 f32 columns at 0x40, and
            # its (lane, column) -> (q, d) map is the one MEASURED at the head
            # of this file. It is read with the segment's own slice atom
            # (D0.14's "two LDTM.x32 = 64 columns taken in one go", peak 68):
            # two copies land straight in the two halves of one flat 64-f32
            # line, and - the fragment being in column order - the four
            # registers at 4*quad ARE the 16 B this lane owns in column quad
            # `quad`, which is what the ring slot wants from it. So the store is
            # 16 x 16 B per trip with no gather and no second line, and the ring
            # row a thread used to own alone is now the 128 lanes' 2 KB quad.
            # The atom is BUILT HERE rather than shared with the segment: it is
            # the same `Ld32x32b.x32` over the same (128 lane, 32 column) shape,
            # so the slice is identical, but a fragment built in the common
            # scope is live in all 16 warps (D0.14's scope note above).
            lt = tidx % _WG_THREADS              # thread in the WG == TMEM lane
            t_dq_h = [cute.make_tensor(
                tmem_ptr + _TMEM_DQ + hf * _HALF_COLS, cute.make_layout(
                    (_TMEM_LANES, _HALF_COLS), stride=(_TMEM_LANE_STRIDE, 1)))
                for hf in range(2)]
            t2r = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
                    cutlass.Float32),
                t_dq_h[0])
            thr_l = t2r.get_slice(lt)
            src_dq = [thr_l.partition_S(t) for t in t_dq_h]
            id_l = thr_l.partition_D(
                cute.make_identity_tensor((_TMEM_LANES, _HALF_COLS)))
            r_h = cute.make_rmem_tensor(id_l.shape, cutlass.Float32)
            n_h = cute.size(r_h)
            ix_l = _frag_index(r_h)
            _slice_in_order(id_l, _frag_index(id_l), n_h, "writer t2r slice")
            r_dq = _rmem_line(_D_HALF, cutlass.Float32)
            dq_half = [_frag_view(r_dq, hf * _HALF_COLS, r_h.layout)
                       for hf in range(2)]
            # One 4-register window per column quad - the ring store's source.
            # The fallback staging is only for a DSL build that will not offset
            # register storage; it costs 4 registers and one copy, never a
            # gather.
            _LAY4 = cute.make_layout(_ACC_VEC)
            dq_quad = [_frag_view(r_dq, _ACC_VEC * qd, _LAY4)
                       for qd in range(_ACC_QUADS)]
            r_q4 = _rmem_line(_ACC_VEC, cutlass.Float32)
            # The ring is read by the bulk engine as raw bytes, so its base is
            # taken once, outside the loop, as a plain integer address; so is
            # the workspace's (the reduce's address source is loop-invariant but
            # for the trip's q block).
            a_ring = _addr(sRing)
            a_dqacc = _addr(mDqAcc)

            # ... and so does D0.12's 0x98: both CTAs' writer warps release the same
            # 2-CTA accumulator, and the leader's MMA warp is the only waiter.
            lb_dqempty = cute.arch.map_dsmem_ptr(mbar + _BAR_DQ_EMPTY, 0)

            # Same for D0.12's 0x98: the dQ columns start out drained, so one arrive per
            # writer warp per CTA (8 = the row's count) completes its phase 0 before
            # trip 0 and every trip's wait is phase i & 1 with no first-trip case.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(lb_dqempty)

            for i in cutlass.range(n_iter):
                ph_w = i % 2
                # ---- D0.12: the dQ-writer (warps 8-11) -----------------------------
                # Per trip: `WAIT 0x90 -> LDTM 64 f32 columns -> arrive 0x98 -> 8 x
                # [STS one 4 KB chunk into the ring -> proxy fence -> BAR(0x4) ->
                # elected UBLKRED 4 KB + commit + wait_group(`_RING_INFLIGHT`,
                # read) -> BAR(0x4)]`,
                # which is D0.12's 16 BAR(0x4) + 1 blocking wait per trip exactly.
                # 0x98 is arrived the instant the fragment is in registers, not at the
                # end of the ring: the tmem columns are free from that point and chain3
                # of the next trip is the thing waiting for them.
                # The ring slot is `chunk mod 4` and nothing waits on it directly —
                # `wait_group(n, read)` after chunk c's reduce is what says chunk c-n
                # has READ its slot before the chunk that recycles it (c-n+4) refills
                # it, across trips as well as within one (no per-trip drain: a ring
                # that only works when emptied is not a ring). The rung is D1.3's
                # `_RING_INFLIGHT = 2` on 4 slots, which is safe for the same reason
                # every n <= 3 is: slot (c mod 4) is next written by chunk c+4, and
                # by then chunk c's reduce is n=2 groups behind the head.
                # The 4 KB destination is D0.3's blocked address, derived from
                # the accumulator map: this CTA owns q rows [64*rank, +64) of BOTH d
                # halves, so its 8 chunks are two 16 KB runs.
                _seg("writer.dqfull.begin", i)
                cute.arch.mbarrier_wait(mbar + _BAR_DQ_FULL, ph_w)
                _seg("writer.dqfull.end", i)
                for hf in cutlass.range_constexpr(2):
                    if cutlass.const_expr(dq_half[hf] is not None):
                        cute.copy(t2r, src_dq[hf], dq_half[hf])
                    if cutlass.const_expr(dq_half[hf] is None):
                        cute.copy(t2r, src_dq[hf], r_h)
                        for j in cutlass.range_constexpr(n_h):
                            r_dq[hf * _HALF_COLS + j] = r_h[ix_l(j)]
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(lb_dqempty)
                _seg("writer.ring.begin", i)
                m_cur = m_min + i
                for c in cutlass.range_constexpr(_N_CHUNK):
                    slot = c % _RING_SLOTS
                    # every lane writes its own 32 B of the slot: two 16 B
                    # stores, one per column quad of the chunk
                    _ring_store_chunk(sRing, slot, lt, r_dq, dq_quad, r_q4, c)
                    # a generic STS is invisible to the async proxy the bulk
                    # engine reads the slot through
                    _PROXY_FENCE()
                    _NAMED_BAR(_WRITER_BAR, _WRITER_THREADS)
                    if warp == _WARP_WRITER:
                        with cute.arch.elect_one():
                            off_red = cutlass.Int64(
                                _acc_tile_base(b, h, m_cur)
                                + _acc_chunk_off(rank, c)) * 4
                            _bulk_red_add_f32(
                                a_dqacc + off_red,
                                a_ring + cutlass.Int32(slot * _CHUNK_BYTES),
                                cutlass.Int32(_CHUNK_BYTES))
                            _BULK_COMMIT()
                            _BULK_WAIT(_RING_INFLIGHT)
                    _NAMED_BAR(_WRITER_BAR, _WRITER_THREADS)
                _seg("writer.ring.end", i)

            # No slot of the ring may still be under an async read when the CTA exits
            # and its smem is handed to the next one (D0.12): only the thread that
            # issued the reduces tracks their group, so only it drains.
            _seg("writer.drain.begin")
            if warp == _WARP_WRITER:
                with cute.arch.elect_one():
                    _BULK_WAIT(0)
            _seg("writer.drain.end")

    # Nothing of the pair's TMEM may be released while the peer still holds a
    # view of it (D0.11's 0x128 rendezvous; the mbarrier form of it arrives with
    # the chains, a cluster barrier is the whole of it here).
    _seg("main.dealloc.begin")
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
    if warp == _WARP_MMA:
        cute.arch.dealloc_tmem(tmem_ptr, _TMEM_COLS, is_two_cta=True)
    _seg("main.exit")


# --- postprocess (D2.2) -------------------------------------------------------
# D0.3's consumer: read the blocked f32 dq_accum run for (b, h, m_block), undo
# the (rank, lane, column) -> (q, d) split, apply softmax_scale (D0.4 puts dQ's
# scale here) and round to bf16 in the (B, N, H, D) output layout.
#
# Both ends of that are ATOM decisions, not spellings:
#   * the READ is D0.3's block read with thread = TMEM lane — 32 x `LDG.E.128`
#     per thread at stride 0x800, so a warp's 32 lanes are 512 CONTIGUOUS bytes
#     per instruction. The row-major reading of this block (thread = q row,
#     128 scalar `LDG` down a 256 B row) is §4's forbidden zone: same bytes,
#     8x the instructions and no coalescing at all.
#   * the WRITE is one 16 B granule per thread with 16 threads per q row, so a
#     store instruction covers two whole 256 B rows of (B, N, H, D).
# The two disagree by a transpose (lane = q, quad = d), which is what the 32 KB
# bf16 staging tile is for; `_post_swz`'s rotation is what keeps both sides
# bank-conflict free. Both gmem legs are sliced out of a `_gvec_view`, which is
# the only spelling of "16 B at a dynamic element offset" the installed DSL will
# accept for a 128-bit atom.

@cute.kernel
def _postprocess_kernel(
    mDqAcc: cute.Tensor,    # (ACC_ELEMS,) f32
    mDq: cute.Tensor,       # (B, N, H, D) bf16  out
):
    tidx, _, _ = cute.arch.thread_idx()
    m_blk, h, b = cute.arch.block_idx()

    smem = cutils.SmemAllocator()
    sT = smem.allocate_tensor(cutlass.BFloat16, _post_tile_layout(),
                              byte_alignment=1024)

    gAcc4 = _gvec_view(mDqAcc, _ACC_ELEMS, _ACC_VEC)
    gDq8 = _gvec_view(mDq, _B * _N * _H * _D, _POST_GRAN)

    # this tile's first 16 B granule, in the read view's granule units
    tile_v = _acc_tile_base(b, h, m_blk) // _ACC_VEC
    s = cutlass.Float32(_SCALE)
    r4 = [_rmem_line(_ACC_VEC, cutlass.Float32) for _ in range(_CHUNK_QUADS)]
    r_in = _rmem_line(_POST_GRAN, cutlass.BFloat16)
    r_out = _rmem_line(_POST_GRAN, cutlass.BFloat16)

    # ---- read: thread = TMEM lane, 16 B per (rank, quad) -------------------
    # D0.3's semantics: this thread is lane `tidx`, so it carries q row
    # `64*rank + tidx % 64` at d columns `64*(tidx // 64) + 4*quad + j`. Two
    # adjacent quads are the 8 consecutive d of one output granule, which is why
    # the scale and the bf16 rounding happen here — the staging tile is then
    # read back one granule per thread instead of two. The lane's 16 B of quad
    # `quad` sit at (rank*16 + quad)*0x800 + lane*16, so a warp's 32 lanes are
    # 512 contiguous bytes per instruction.
    q_lo = tidx % _D_HALF
    dh = tidx // _D_HALF
    for r in cutlass.range_constexpr(2):
        q = r * _D_HALF + q_lo
        for gi in cutlass.range_constexpr(_POST_ROW_GRAN // 2):
            for t in cutlass.range_constexpr(_CHUNK_QUADS):
                v = (tile_v + r * (_ACC_RANK_STRIDE // _ACC_VEC)
                     + (_CHUNK_QUADS * gi + t) * (_ACC_QUAD_STRIDE // _ACC_VEC)
                     + tidx)
                _COPY128(_gvec(gAcc4, v), r4[t])
            for t in cutlass.range_constexpr(_CHUNK_QUADS):
                for j in cutlass.range_constexpr(_ACC_VEC):
                    r_in[_ACC_VEC * t + j] = _bf16(r4[t][j] * s)
            dg = dh * (_POST_ROW_GRAN // 2) + gi
            _COPY128(r_in, sT[(q, _post_swz(dg, q), None)])

    cute.arch.sync_threads()

    # ---- write: 16 threads per q row, one 16 B granule each ----------------
    dg = tidx % _POST_ROW_GRAN
    row0 = tidx // _POST_ROW_GRAN
    for st in cutlass.range_constexpr(_POST_STEPS):
        row = st * _POST_ROWS_STEP + row0
        _COPY128(sT[(row, _post_swz(dg, row), None)], r_out)
        v8 = (((b * _N + m_blk * _BLK_M + row) * _H + h) * _POST_ROW_GRAN + dg)
        _COPY128(r_out, _gvec(gDq8, v8))


# --- window body (D0.1: three launches, one per call, in order) ---------------

@cute.jit
def _bwd_jit(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mDO: cute.Tensor,
    mLSE: cute.Tensor,
    mDq: cute.Tensor,
    mDk: cute.Tensor,
    mDv: cute.Tensor,
    mDpsum: cute.Tensor,
    mLseLog2: cute.Tensor,
    mDqAcc: cute.Tensor,
):
    _preprocess_kernel(mO, mDO, mLSE, mDpsum, mLseLog2, mDqAcc).launch(
        grid=[_NBLK, _H, _B],
        block=[_PRE_THREADS, 1, 1],
    )
    # D0.8's faces, one TMA descriptor each. Q and dO appear twice — once row
    # sliced for the S^T / dP^T chains, once d-column sliced for the dK / dV
    # chains — and K a third time as K_dsplit for the dQ chain; that is what
    # buys all five chains an smem descriptor read with no transpose.
    a_qr, g_qr = _build_face(mQ, "q_row")
    a_dr, g_dr = _build_face(mDO, "do_row")
    a_qc, g_qc = _build_face(mQ, "q_col")
    a_dc, g_dc = _build_face(mDO, "do_col")
    a_k, g_k = _build_face(mK, "k")
    a_v, g_v = _build_face(mV, "v")
    a_kd, g_kd = _build_face(mK, "k_dsplit")
    # D0.15's two store boxes: (128 KV rows, 64 d) on the same (N, D, H, B) view
    # the K-major load faces use, so one CTA's output is two boxes side by side
    # in d and the pair's is four.
    a_sdk, g_sdk = _build_store(mDk, "dkv")
    a_sdv, g_sdv = _build_store(mDv, "dkv")
    _main_kernel(mLseLog2, mDpsum, mDqAcc,
                 a_sdk, g_sdk, a_sdv, g_sdv,
                 a_qr, g_qr, a_dr, g_dr, a_qc, g_qc, a_dc, g_dc,
                 a_k, g_k, a_v, g_v, a_kd, g_kd).launch(
        grid=[_N // _BLK_N, _H, _B],
        block=[_MAIN_THREADS, 1, 1],
        cluster=[2, 1, 1],
        smem=_SMEM_BYTES,
    )
    _postprocess_kernel(mDqAcc, mDq).launch(
        grid=[_NBLK, _H, _B],
        block=[_POST_THREADS, 1, 1],
        smem=_SM_POST,
    )


_COMPILED = None


def _run(q, k, v, o, do, lse, compile_only: bool = False):
    """The timed window body. Returns the three outputs plus the intermediates
    of D0.3's ledger (probes read them; `flash_attn_bwd` drops them)."""
    global _COMPILED

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dpsum = torch.empty(_B, _H, _N, device=q.device, dtype=torch.float32)
    lse_log2 = torch.empty(_B, _H, _N, device=q.device, dtype=torch.float32)
    dq_accum = torch.empty(_ACC_ELEMS, device=q.device, dtype=torch.float32)

    args = [
        from_dlpack(t, assumed_align=16)
        for t in (q, k, v, o, do, lse, dq, dk, dv, dpsum, lse_log2, dq_accum)
    ]
    if _COMPILED is None:
        _COMPILED = cute.compile(_bwd_jit, *args)
    if not compile_only:
        _COMPILED(*args)
    return dq, dk, dv, dpsum, lse_log2, dq_accum


def flash_attn_bwd(
    q: torch.Tensor,     # (B, N, H, D) bf16
    k: torch.Tensor,     # (B, N, H, D) bf16
    v: torch.Tensor,     # (B, N, H, D) bf16
    o: torch.Tensor,     # (B, N, H, D) bf16  forward output
    do: torch.Tensor,    # (B, N, H, D) bf16  gradient wrt o
    lse: torch.Tensor,   # (B, H, N)    fp32  forward log-sum-exp (natural log)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dq, dk, dv), each (B, N, H, D) bf16."""
    dq, dk, dv, _, _, _ = _run(q, k, v, o, do, lse)
    return dq, dk, dv
