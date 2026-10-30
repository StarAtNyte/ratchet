"""gdn_prefill_varlen — the student's kernel file (the only file the student edits).

Gated Delta Net (GDN) prefill, the FlashInfer-Bench definition
`gdn_prefill_qk4_v8_d128_k_last` — Qwen3-Next linear-attention layers at tensor
parallel 4: 4 query heads and 4 key heads, 8 value heads (GVA: each q/k head is
shared by two value heads), head size 128, bf16 in / bf16 out, variable-length
batching through cu_seqlens, and a per-sequence fp32 recurrent state carried in the
k-last layout [N, H, V, K]. The campaign point is one real contest workload
(a9540651: 8192 tokens over 20 sequences, the heaviest of the definition's 100
contest workloads). The signature below is the card's frozen contract; the student
implements the body in CuTe DSL; the signature itself may not change.

Semantics (per sequence s spanning tokens [cu_seqlens[s], cu_seqlens[s+1]), per
value head h; q and k are shared across the two value heads of their group, i.e.
q_exp = q.repeat_interleave(2, dim=1), and the recurrence carries the state as
S [K=128, V=128], which is the TRANSPOSE of the [V, K] layout the state tensors
use; fp32 chain):

    g[t, h]    = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))     fp32
    beta[t, h] = sigmoid(b[t, h])                                         fp32

    S_0      = state[s, h].T                              # [V, K] -> [K, V]
    per token t, in order:
        S'    = g[t, h] * S
        v_old = k_exp[t, h] @ S'                           # [1, V]
        v_new = beta[t, h] * v[t, h] + (1 - beta[t, h]) * v_old
        S     = S' - k_exp[t, h]^T @ v_old + k_exp[t, h]^T @ v_new
        out[t, h] = scale * (q_exp[t, h] @ S)              # bf16 on output
    new_state[s, h] = S.T                                  # [K, V] -> [V, K], fp32

The sequences are independent; a zero-length sequence contributes nothing and keeps
its incoming state. `state` is an INPUT and must come back unmodified — the updated
state is the second return value, a freshly allocated tensor. Sequence lengths are
arbitrary (the contest workloads range from 6 to 8192 tokens in a batch, with
individual streams from 1 to several thousand), so the schedule may not assume a
multiple of any tile size.

--------------------------------------------------------------------------------
M1 status: D1.3 the store off the chain, D1.4 the item prologue off the chain

D1.3.  The output face is double buffered and warp 11 no longer drains its
store group before releasing a face: it commits the `UTMASTG` pair of chunk c,
waits the group count down to one (so chunk c's store is still in flight) and
releases the face of chunk c - 1.  WG B's B5 therefore waits on a release that
was published while warp 11 was a chunk behind, and the store round trip is
gone from the chunk period.  The tail-chunk path stores through the LSU and
commits no group of its own, so it drains to zero instead of to one, which
keeps the per-slot release exactly one arrive per generation whatever the tail
chunks do.

D1.4.  (1) The 288-thread item rendezvous is removed.  It was the only thing
serialising WG B's 64 KiB state load / write-back against WG A's and the issue
warp's state-independent work for the next item; the chunk counter that keys
every phase is a pure function of the item sequence, so the three roles need no
meeting to agree on it, and the runahead is bounded by the protocol itself
(`ACC_A full` of chunk cg is only emitted after step 8 of cg - 1, which waits
on WG B's B1 and B4 of cg - 1).  (2) `cu_seqlens` -- the one data-dependent
address load in the kernel -- is read a whole item ahead into a loop-carried
register in each of the three roles, so only the first item of a CTA pays its
latency.  (3) The next item's 64 KiB state plane is L2-prefetched an item early
(`prefetch.global.L2`, four per lane over its own 512 B v row).

--------------------------------------------------------------------------------
D1.1 lookahead d = 1 + the D1.2 depths it needs

The state-independent segment of chunk c + 1 -- UMMA groups (a) K.K^T and
(b) Q.K^T, and the whole WG A segment (mask, 64 x 64 triangular inverse, P)
that drains their accumulators -- now runs inside chunk c's state chain.  The
issue warp emits groups (a)/(b) of chunk c + 1 straight after step 8 (group
(i)) of chunk c, waiting on no commit of chunk c, so ACC_A / ACC_B of chunk
c + 1 are published a whole chain early and WG A's 549-instruction segment
overlaps chunk c's B4 -> B1 -> (d) -> B2 instead of sitting in front of it.
Steps 3-8 of chunk c + 1 still sit behind chunk c's c14..c16.  The critical
path per chunk is now B1 -> (d) -> B2 -> (g) -> B4 -> (h)(i).

Two D1.2 depths are what make the window legal rather than a race, and they
land in the same commit: the staging faces s0 / s1 are double buffered
(WG A writes L of chunk c + 1 while groups (g)/(h) of chunk c still read T / P
of chunk c), and the Q tile ring goes 1 -> 2 (group (b) of chunk c + 1 reads
Q_{c+1} while group (e) of chunk c has not yet released Q_c).  `T ready` and
`P ready` become per-staging-slot mbarriers for the same reason -- they are the
one producer/consumer pair the lookahead lets run a generation apart, and a
single barrier would flip parity twice and hang the issue warp.  K stays at two
slots (exactly enough: K_c lives to group (i), K_{c+1} is read by group (a) of
the lookahead) and V stays single; those two remaining notches are the D1.2
single-variable A/B rounds.

--------------------------------------------------------------------------------
Implementation status (M0 construction sequence, complete; M0 Gate passed)

The whole chunk recurrence of D0.2 is on the machine: every one of the seven
UMMA groups of D0.9, the five WG B segments of D0.11, the full WG A segment of
D0.10 (mask, triangular inverse, P) and the D0.12 issue order that ties them
together.  Per chunk the MMA warp issues (a) K.K^T, (b) Q.K^T, (d) (K S_0)^T,
(e) (Q S_0)^T, (g) X.T^T, (h) U.P^T and (i) U~^T.K = 44 instructions closed by
16 `tcgen05.commit`s; WG B carries S32 -> S16 + gamma_C decay (B1), X (B2),
the ACC_O column scaling (B3), the U16 / U16s pair (B4) and the O epilogue (B5);
WG A turns ACC_A into the masked L, inverts (I + L) and turns ACC_B into P.

Step 7 lands the real D0.10 triangular inverse: `_tri_inverse` below blocks
(I + L) 16 x 16 and closes it in two levels of block doubling, every product a
`cute.gemm` on a 16 x 16 x 16 warp-level MMA (`HMMA.16816`) whose operands and
result move through the staging faces on `ldmatrix` / `ldmatrix.trans` /
`stmatrix`. Only the four 16 x 16 diagonal blocks are inverted by scalar
forward substitution, and that runs in registers with the already-final row
broadcast by `shfl` -- no scalar `LDS`/`STS` touches a 64 x 64 face.

The D0.7 warpgroup register split is mounted (one `setmaxnreg` per warpgroup,
before any fragment is live).
"""

from __future__ import annotations

from typing import Tuple

import torch

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.layout import LayoutEnum
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

# ---------------------------------------------------------------------------
# Problem geometry (D0.1 / D0.2): frozen by the operator definition.
# ---------------------------------------------------------------------------
HQ = 4            # query heads
HK = 4            # key heads
HV = 8            # value heads; v-head h uses q/k head h // 2 (GVA)
DIM = 128         # head size
CHUNK = 64        # token chunk (D0.2)
# D0.10: the triangular inverse is blocked 16 x 16 (INV_N x INV_N of them) and
# combined in two levels of block doubling, 16 -> 32 -> 64.
INV_B = 16
INV_N = CHUNK // INV_B

# D0.1 / D0.7: one persistent wave of 148 CTAs, 384 threads = 3 warpgroups.
MAX_CTAS = 148
THREADS = 384
WARPGROUP = 128

# D0.7 register split (`setmaxnreg`, one value per warpgroup): WG A 224, WG B
# 248, the four specialist warps 32. 128 * (224 + 248 + 32) = 64512 = 168 * 384,
# i.e. exactly the per-thread budget `reqntid = 384` leaves ptxas when it targets
# 1 CTA/SM -- the occupancy the whole-TMEM allocation of D0.3 already pins.
#
# The triple is the minimum-spill point of an eight-candidate sweep run on the
# delivered tree, all at the same sum 504 per 128-thread role (the sum is not a
# free axis: at 65536 the two `.inc` roles wait for pool the `.dec` role never
# frees and the kernel hangs -- hw_kb `warp_specialization` pit 4). Static
# SASS spill instructions per `USETMAXREG` region, every candidate launching and
# numerically identical (max_diff 0.004075):
#
#   split        LDL STL  prelude(168)  dec        inc(A)   inc(B)
#   24/224/256    10   9     0/1        24: 6/8    224: 2/0  256: 2/0   <- before
#   32/216/256     7  11     0/1        32: 2/9    216: 3/1  256: 2/0
#   40/208/256     7  11     0/1        40: 2/9    208: 3/1  256: 2/0
#   56/192/256    24  19     1/2        56: 4/2    192:16/15 256: 3/0
#   32/224/248     7  11     0/1        32: 2/9    224: 3/1  248: 2/0   <- now
#   40/224/240     7  11     0/1        40: 2/9    224: 3/1  240: 2/0
#   24/232/248    11  18     0/1        24: 6/16   232: 3/1  248: 2/0
#   24/240/240    11  18     0/1        24: 6/16   240: 3/1  240: 2/0
#
# Reading: the specialists' 24 is the binding quota -- 14 of the 19 sites sit in
# the `.dec` region and the six LDL among them drop to two as soon as the
# specialists get 32 (their branch head carries `hq`, `n_seq` and D1.4's
# loop-carried `cu_seqlens` pair across all four warps). 32 and 40 measure
# identically, so the donor keeps the surplus: WG B gives the 8 and stays clean
# at 248, WG A holds the 224 it needs (at 192 its own region takes 31 sites).
# No split in the sweep reaches spill == 0; the residue is a live-set fact, not
# a quota fact.
REG_WG_A = 224
REG_WG_B = 248
REG_SPECIALIST = 32

# D0.9 group (a)/(b) shape. The K/Q/V smem faces are built as this MMA's
# A-operand face (M = 64 tokens, K = 128 features, K-major, SW128), so the tiles
# the TMA lands are already the operand layout the UMMA groups will read.
MMA_TILER = (CHUNK, CHUNK, DIM)
# D0.9 the M = 128 geometries: (d)/(e) 128x64x128, (g)/(h) 128x64x64 (same op,
# different K extent), (i) 128x128x64 with the B operand MN-major.
TILER_KS = (DIM, CHUNK, DIM)
TILER_UP = (DIM, CHUNK, CHUNK)
TILER_SU = (DIM, DIM, CHUNK)

# D0.8 ring depths, at the D1.2 setting that the D1.1 lookahead d = 1 requires.
# K lives from group (a) of chunk c to group (i) of chunk c and is read one more
# time in the lookahead window, so two slots are exactly enough (the third slot
# is the remaining D1.2 notch). Q is now read by group (b) of chunk c+1 while
# group (e) of chunk c has not yet released chunk c's copy, so it must be two.
# V is still released mid-chunk by B2 and stays single.
K_SLOTS = 2
Q_SLOTS = 2
V_SLOTS = 1

# D1.2 staging groups: with d = 1 WG A writes T/P for chunk c+1 while groups (g)
# and (h) of chunk c are still reading chunk c's faces, so the two 64 x 64 bf16
# exchange faces s0 / s1 are double buffered (s0/s1 and s0'/s1', +16 KiB).
STAGE_SLOTS = 2

# D1.3 output face depth. The M0 form was one 16 KiB face drained by warp 11
# before it was released, which put a whole `UTMASTG` round trip between WG B's
# B5 of chunk c and its B5 of chunk c + 1. With two faces and a `wait_group 1`
# the store of chunk c is still in flight when warp 11 releases the face of
# chunk c - 1, so the release WG B waits on at chunk c + 1 is one that warp 11
# has already published (+16 KiB).
OUT_SLOTS = 2

# D0.4 / D0.10: the gating array is double buffered (warp 10 may run one chunk
# ahead of WG A), which is also the second copy D1.1's lookahead needs.
GATE_SLOTS = 2
GATE_FIELDS = 5
# D0.11 (G5): the array WG B reads is a *field-major* face array
# `[slot][field][token]`, not D2.3's token-major record `[slot][token][field]`.
# Every one of WG B's reads of it is a *column* read -- element e of a fragment
# wants the scalar of its own token -- and a vector load can only serve that if
# the four adjacent tokens of one field are adjacent in smem, which is exactly
# what the token-major record cannot do: with an odd field count its consecutive
# tokens sit 20 B apart, neither contiguous nor 16 B aligned, and no amount of
# load-combining recovers it (hw_kb `branchfree_masked_elementwise_pass`,
# variant "[row-lane x col] fragment" §(a)). Each field is now one row of
# CHUNK = 64 fp32 = 256 B, so every row start is 16 B aligned. D2.3's bank
# argument was about the *record*, and it moves with the fields WG A reads,
# which keep their own two faces (`sGatePair` / `sGateCol`) untouched.
#
# Two of the five fields are stored in the form the consumer wants rather than
# the form the producer computes, which is the cheap half of the same recipe:
# warp 10 does K = 64 extra FMUL/FNEG per chunk (2 per lane) so that 128 WG B
# lanes do none.
GATE_BETA = 0           # beta_t                (B2)
GATE_NGAM = 1           # -gamma_t, negated     (B2: one fma2, not mul + sub)
GATE_SGAM = 2           # scale * gamma_t       (B3: `scale` premultiplied)
GATE_LAM = 3            # lambda_t              (B4)
GATE_GC = 4             # gamma_C, at token 0 of its row (B1)

# D0.10 lever (2) of hw_kb `branchfree_masked_elementwise_pass`: the structure of
# the accumulator fragment WG A's two mask segments walk, i.e. which of the 32
# elements one thread owns share a row (and therefore share `G_i` and `beta_i`)
# and which have consecutive columns (and therefore share a vector read of
# `G_j`). It is a static property of the atom + tile + thread slice
# (`Ld16x256b` x8 over the 64 x 64 plane of `_mma_ab`), never of the data, and
# gotcha 6 is explicit that it must be *read off* the identity partition rather
# than guessed -- `notes/probe_d010_scalar_batch.py` dumps every thread's
# (row, col) table from the very constructions below (`_mma_ab()` ->
# `partition_C(identity)` -> `flat_divide` -> `thr.partition_D`) and derives
# these two tables from it on the host. Measured shape: two rows per thread
# (r and r + 8) 16 elements each, and 16 runs of two consecutive columns whose
# first column is even (so a `ld.shared.v2` of the column face is 8 B aligned).
# The same run stepped at four columns does not exist (the probe tried L = 4
# first and it did not hold), which is why the column reads are `v2` and not
# `v4`. 96 scalar `LDS` per chunk per thread become 2 + 16 = 18 vector ones.
ACC_ROW_GROUPS = ((0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29),
                  (2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31))
ACC_COL_RUNS = tuple((2 * k, 2 * k + 1) for k in range(16))

# D0.11 (G5): the same thing for the two accumulator fragments *WG B* walks,
# which live on the (DIM, CHUNK) = 128 x 64 plane of `_mma_m128` and therefore
# have a different structure from WG A's 64 x 64 `_mma_ab` table above -- the
# table does not transfer, and gotcha 6 says to read it off rather than guess.
# `notes/probe_d011_wgb_colvec.py` dumps both from the very constructions below
# (`partition_D` of `tiled_ks` / `tiled_acc32` over `idPlane`):
#   `ks`  (`Ld16x256b` x8, B2's ACC_KS read): four rows per thread, 64 elements,
#         32 runs of two consecutive columns whose first column is even -> one
#         `ld.shared.v2` per run (8 B aligned). `v4` does *not* hold here.
#   `col` (`Ld32x32b` x32, B3's in-place ACC_O and B4's ACC_U read): one row per
#         thread, all 64 of its columns, 16 runs of four consecutive columns
#         starting on a multiple of four -> one `ld.shared.v4` per run. Every
#         lane owns the same 64 columns, so these reads are pure broadcasts.
KS_COL_RUNS = tuple((2 * k, 2 * k + 1) for k in range(32))
COL_COL_RUNS = tuple(tuple(range(4 * k, 4 * k + 4)) for k in range(16))
# How many runs of a segment's column scalars are read *above* that segment's
# accumulator wait. Reading the whole face there would be the cleanest form of
# hw_kb `branchfree_masked_elementwise_pass` §(c) lever 3, but it puts 64 fp32
# across the wait, and B2's segment budget (D0.11: {64 ACC_KS fp32 + 32 V bf16
# pairs + column scalars} <= ~110 regs) has no room for that. At 4 the pass runs
# strip-wise instead -- run q + GATE_STRIPS is read before run q is multiplied --
# so at most five runs of scalars are ever live while every vector load still
# leads its consumer by four runs, and the first four lead it by the whole wait.
GATE_STRIPS = 4

# D2.1: the granularity of the B1 -> group (d) handover, i.e. how many pieces the
# 128-column state pass of segment B1 is split into. The state pass is 4 `x32`
# TMEM steps of 32 columns each (`TMEM_REP`), so the legal values of the
# milestone's candidate set are 1 (one whole), 2 (the M0 starting point: two
# halves, 2 steps each) and 4 (one step each). Two numbers move with it: the
# issue warp's group (d) is cut into the same number of pieces (8 k-blocks / segs
# each waiting on its own segment barrier), and B1's peak fp32 live set is
# 128 / S16_SEGS state registers plus the bf16 pack of the same piece.
S16_SEGS = 4
J_PER_SEG = 4 // S16_SEGS    # `x32` state steps per segment
KB_PER_SEG = 8 // S16_SEGS   # group (d) k-blocks per segment
# The producer side (B1) is written over `S16_SEGS`; the consumer side (group
# (d)) is written out one wait per piece, because the timeline probe only
# rewrites `_seg` markers whose site is a literal string. Changing the knob
# therefore means editing that unrolled block too -- this assert is the reminder.
assert S16_SEGS == 4, "group (d) below is unrolled for four segments"

# D0.8 / D0.12 / D0.14 mbarrier table. Producer side arrives once (one elected
# lane, or one `tcgen05.commit`); the "consumed" side is arrived by the 128
# threads of the consuming warpgroup.
BAR_K_FULL = 0          # + slot   <- TMA expect_tx           (1)
BAR_K_EMPTY = 2         # + slot   <- commit after group (i)  (1)
BAR_Q_FULL = 4          # + slot   <- TMA expect_tx           (1)
BAR_Q_EMPTY = 6         # + slot   <- commit after group (e)  (1)
BAR_V_FULL = 8          #          <- TMA expect_tx           (1)
BAR_V_EMPTY = 9         #          <- WG B (B2)             (128)
BAR_GATE_FULL = 10      # + slot   <- warp 10 elected lane     (1)
BAR_GATE_EMPTY = 12     # + slot   <- WG A + WG B            (256)
BAR_ACC_A_FULL = 14     #          <- group (a) commit c1      (1)
BAR_ACC_A_EMPTY = 15    #          <- WG A                   (128)
BAR_ACC_B_FULL = 16     #          <- group (b) commit c2      (1)
BAR_ACC_B_EMPTY = 17    #          <- WG A                   (128)
# D2.1: the S16 handover is per segment of the state pass, `S16_SEGS` of them.
BAR_S16_SEG = 18        # + seg    <- WG B (B1, one segment) (128)
BAR_S16_EMPTY = 22      #          <- commit after group (e)   (1)
BAR_ACC_KS_FULL = 23    #          <- group (d) commit c3      (1)
BAR_X_READY = 24        #          <- WG B (B2)              (128)
# D1.1: T / P are the one producer-consumer pair the lookahead lets run a whole
# generation apart (WG A publishes chunk c+1 before the MMA warp has consumed
# chunk c), so their ready signals are per staging slot -- a single barrier
# would flip parity twice and hang the consumer.
BAR_T_READY = 25        # + slot   <- WG A (inverse)         (128)
BAR_P_READY = 27        # + slot   <- WG A (P)               (128)
BAR_ACC_O_FULL_E = 29   #          <- group (e) commit c4      (1)
BAR_ACC_O_SCALED = 30   #          <- WG B (B3)              (128)
BAR_ACC_U_FULL = 31     #          <- group (g) commit c8      (1)
BAR_ACC_O_FULL_H = 32   #          <- group (h) commit c11     (1)
BAR_U16_READY = 33      #          <- WG B (B4)              (128)
BAR_U16S_READY = 34     #          <- WG B (B4)              (128)
BAR_S32_DEC = 35        #          <- WG B (B1)              (128)
BAR_S32_FULL = 36       #          <- group (i) commit c14     (1)
BAR_ACC_O_EMPTY = 37    #          <- WG B (B5)              (128)
# D1.3: both output-face signals are per slot. "Face full" is the ordinary
# producer signal; "face empty" is now published one chunk late (warp 11 arrives
# for chunk cg - 1 after the `wait_group 1` of chunk cg), which is exactly what
# lets the store of chunk cg stay in flight.
BAR_OUT_FULL = 38       # + slot   <- WG B (B5, face written) (128)
BAR_OUT_EMPTY = 40      # + slot   <- warp 11 (face read)       (1)
MBAR_COUNT = 42

# D0.10: the WG A internal rendezvous (id 0x2, width 128) that separates the
# phases of the L -> T -> P sequence on the shared staging faces. Five blocking
# meets per chunk, one under the budget of six.
NBAR_WGA = 2
# D0.10: reserved for a half-warpgroup meet (two ids, one per half of WG A, each
# 64 threads wide).  The phase 1 -> phase 2 seam of `_tri_inverse` was cut this
# way and is not any more -- the dependency graph of the inverse does not in fact
# split along the 16-wide output column slab a warp owns, see the note at that
# seam -- so nothing arrives on these two ids today.
NBAR_WGA_L = 3
NBAR_WGA_R = 4
HALF_WG = WARPGROUP // 2

# D0.3 / D0.4: one `tcgen05.alloc` of the whole 512-column block per CTA, held for
# the CTA's lifetime, and the D0.4 column plan it carries.
TMEM_COLS = 512
S32_COL = 0               # S32 = state fp32 [v x k], columns [0, 128)
S16_COL = 128             # S16 = state bf16 A-operand copy, [128, 192)
ACC_A_COL = 192           # K.K^T fp32, M = 64,          [192, 256)
ACC_B_COL = 256           # Q.K^T fp32, M = 64,          [256, 320)
ACC_KS_COL = 320          # (K S_0)^T -> U^T,            [320, 384)
ACC_O_COL = 384           # (Q S_0)^T -> O^T,            [384, 448)
X16_COL = 448             # X bf16 -> U16 bf16,          [448, 480)
U16S_COL = 480            # lambda-scaled U bf16,        [480, 512)
# TMEM addresses are (lane << 16) | column, and pointer arithmetic on a TMEM
# pointer counts 32-bit columns -- so a [lane, column] tensor has this stride.
TMEM_LANE_STRIDE = 65536
# `Ld/St32x32b` with Repetition.x32 tiles the 128 columns of S32 into 4 steps of
# 32 columns, i.e. the "LDTM x4 / STTM x4 per state pass" of D0.11 segment B1.
TMEM_REP = 32


def _seg(site, row=0):
    """Timeline-probe marker. The driver rewrites this body on the probe arm."""
    return None


# ---------------------------------------------------------------------------
# Construction aid (probe only): the per-chunk stage dump.
#
# `_DEBUG_SINK` is None on every production path, and the production compile is
# traced with `dbg_on = False`, so not one of the writes below exists in the
# kernel that runs. A probe sets it to a `[DBG_PLANES, DIM, DIM]` fp32 tensor,
# which selects a second compile whose only difference is that chunk 0 of item 0
# writes every intermediate of D0.10 / D0.11 out under the identity coordinates
# of the very fragment the production path already indexes with.
# ---------------------------------------------------------------------------
_DEBUG_SINK = None
DBG_PLANES = 13
DBG_ACC_A = 0     # k_i . k_j                     (64 x 64)
DBG_L = 1         # L face as the inverse reads it (64 x 64)
DBG_ACC_B = 2     # q_t . k_i                     (64 x 64)
DBG_T = 3         # T = (I + L)^-1 face           (64 x 64)
DBG_P = 4         # P face                        (64 x 64)
DBG_KS = 5        # (K S_0)^T                     (128 x 64)
DBG_V = 6         # V^T through the LDSM.MT read  (128 x 64)
DBG_X = 7         # X = beta (V - gamma K S_0)    (128 x 64)
DBG_O_E = 8       # (Q S_0)^T before the scaling  (128 x 64)
DBG_U = 9         # U^T                           (128 x 64)
DBG_US = 10       # lambda-scaled U^T             (128 x 64)
DBG_O_H = 11      # O^T after group (h)           (128 x 64)
DBG_GATE = 12     # rows 0..4 = G, beta, gamma, lambda, gamma_C


# ---------------------------------------------------------------------------
# D0.5 gating math. The whole chain is fp32 and lives in log space; the only
# transcendentals the hardware has are `ex2.approx` and `lg2.approx`, so every
# `exp` / `log` below is expressed in base two.
# ---------------------------------------------------------------------------
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """log2(x) for x in [1, 2].

    The A rung of the expressibility ladder (hw_kb `ptx_escape`) is the DSL's
    own math module; this pinned build exports `cute.math.exp2` but its `log`
    family is not part of the readable surface, so this is the B rung: the one
    PTX instruction the hardware has (`MUFU.LG2`).
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _prefetch_l2(addr, *, loc=None, ip=None):
    """`prefetch.global.L2 [addr]` -- D1.4 (3), the B rung of the ladder.

    The A rung (`cute.arch.prefetch(cache_level='L2')`) does not lower on this
    toolchain, so this is the one PTX instruction the hardware has; its SASS
    footprint is `CCTL.E.PF2`. `has_side_effects` is mandatory -- a prefetch has
    no result, so without it the whole thing is dead code (hw_kb `ptx_escape`
    gotcha 4), and the address is a gmem `Int64`, constraint `l` (gotcha 3).
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sel_lt_s32(xi, yi, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """`xi < yi ? a : b` -- one `ISETP` plus one `FSEL`, no basic block.

    The branch-free spelling of a per-element mask (hw_kb
    `branchfree_masked_elementwise_pass` §(a)). The DSL's own `if` is an `scf`
    region: whether it becomes an `FSEL` or E real branches is ptxas' size
    heuristic to make, and inside a branch body the neighbouring elements'
    `LDS`/`MUFU` cannot interleave. Selecting on the *product* (rather than
    parking the exponent) makes the masked elements exactly zero no matter
    whether the discarded factor overflowed -- `selp` is a register pick, not
    arithmetic, so an `inf` or `NaN` in the taken-away operand cannot leak.
    """
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Int32(xi).ir_value(loc=loc, ip=ip),
                cutlass.Int32(yi).ir_value(loc=loc, ip=ip),
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $1, $2;\n"
            "\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _lds_v2(addr):
    """One 64-bit shared load (8 B aligned address) -> two `Float32`.

    The batched half of hw_kb `branchfree_masked_elementwise_pass` §(a): the
    scalars a fragment element needs are read once per *group* of elements that
    share them, not once per element. No `dsl_user_op` decorator -- the result
    is a struct type parsed at trace time, which needs the live MLIR context of
    the tracing thread. `has_side_effects=True` plus the `memory` clobber are
    mandatory (gotcha 4): the operands of the asm (an smem base plus a constant
    offset) are available from the first instruction of the kernel, so a pure
    asm could be hoisted above the `mbarrier` wait that publishes these bytes
    and would read the previous chunk's values -- silently, and only on the
    first trip.
    """
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [cutlass.Int32(addr).ir_value()],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), r, [k]))
            for k in range(2)]


def _lds_v4(addr):
    """One 128-bit shared load (16 B aligned address) -> four `Float32`."""
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [cutlass.Int32(addr).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [cutlass.Float32(llvm.extractvalue(T.f32(), r, [k]))
            for k in range(4)]


def _fma2_pair(a0, a1, b0, b1, c0, c1):
    """`(a0 * b0 + c0, a1 * b1 + c1)` in one `FFMA2`.

    The packed half of hw_kb `branchfree_masked_elementwise_pass`, variant
    "[row-lane x col] fragment" §(c): the two adjacent columns of a fragment run
    are one register pair on both sides, so the per-element factor multiply of a
    column-scaling pass costs one instruction per *pair*. All three operands are
    pairs here (the KB spells a broadcast-`b` form as well); `$0` is the low
    half, i.e. the lower column, matching the field order a `ld.shared.v4`
    returns, so no transposition is needed anywhere. Not a `dsl_user_op` -- the
    struct return type is parsed at trace time and the decorator cannot describe
    a two-value result. `has_side_effects=False` is correct and wanted: this is a
    pure function of its registers, and an unused result should vanish. The asm
    text is spelled at the call rather than hoisted to a module constant on
    purpose: a name looked up from the enclosing scope inside a helper that runs
    under a staged loop is the shape the tracer rejects, and the known-good
    `_lds_v2` above has none.
    """
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [cutlass.Float32(v).ir_value() for v in (a0, a1, b0, b1, c0, c1)],
        "{\n\t.reg .b64 pa, pb, pc, t;\n\t"
        "mov.b64 pa, {$2, $3};\n\t"
        "mov.b64 pb, {$4, $5};\n\t"
        "mov.b64 pc, {$6, $7};\n\t"
        "fma.rn.ftz.f32x2 t, pa, pb, pc;\n\t"
        "mov.b64 {$0, $1}, t;\n}",
        "=f,=f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (cutlass.Float32(llvm.extractvalue(T.f32(), r, [0])),
            cutlass.Float32(llvm.extractvalue(T.f32(), r, [1])))


def _mul2_pair(a0, a1, b0, b1):
    """`(a0 * b0, a1 * b1)` in one `FMUL2` -- the same recipe, one step."""
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [cutlass.Float32(v).ir_value() for v in (a0, a1, b0, b1)],
        "{\n\t.reg .b64 pa, pb, t;\n\t"
        "mov.b64 pa, {$2, $3};\n\t"
        "mov.b64 pb, {$4, $5};\n\t"
        "mul.rn.ftz.f32x2 t, pa, pb;\n\t"
        "mov.b64 {$0, $1}, t;\n}",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (cutlass.Float32(llvm.extractvalue(T.f32(), r, [0])),
            cutlass.Float32(llvm.extractvalue(T.f32(), r, [1])))


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None) -> cutlass.Float32:
    """One scalar shared load -- the L = 1 fallback of the run table."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _exp(x):
    return cute.math.exp2(x * LOG2E, fastmath=True)


def _exp2(x):
    """`2^x` -- one `MUFU.EX2`, no scaling `FMUL`.

    For a chain whose argument is already carried in units of log2 (D0.10 step 1
    keeps the WG A copies of `G` pre-scaled by LOG2E at the producer), the
    `x * LOG2E` of `_exp` is a per-element multiply of a loop-variant value and
    cannot be folded by the compiler.
    """
    return cute.math.exp2(x, fastmath=True)


def _softplus(z):
    """log(1 + e^z), the stable form: max(z, 0) + log1p(e^-|z|)."""
    az = cute.arch.fmax(z, -z)
    return cute.arch.fmax(z, cutlass.Float32(0.0)) + _lg2(
        cutlass.Float32(1.0) + _exp(-az)) * LN2


def _sigmoid(x):
    return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + _exp(-x))


def _ge_mask(a, b):
    """1.0 when the integers a >= b, else 0.0 -- branch free.

    `a - b` is exact in fp32 for the small integers used here (lane indices), so
    clamping `a - b + 1` into [0, 1] is an exact predicate, and it keeps the
    warp-wide scan below free of divergent control flow.

    This is the "0 / 1 float times a value" form that hw_kb
    `branchfree_masked_elementwise_pass` gotcha 2 rejects for *per element*
    masks -- and it stays, because its one remaining call site is not one. The
    argument there is `lane` against a trace-time constant, so the whole
    `IADD`/`I2F`/`FADD`/`FMNMX`/`FMNMX` chain is loop invariant and is hoisted
    out of the chunk loop once and for all; only the `FMUL` is paid per trip,
    and a `setp` + `selp` inline asm -- opaque to ptxas, so recomputed on every
    trip -- would cost an `ISETP` per level to save a register that is not
    scarce. The per-element sites, where the predicate really is data dependent
    and the chain really would be paid per element, use `_sel_lt_s32`.
    """
    d = cutlass.Float32(a - b) + cutlass.Float32(1.0)
    return cute.arch.fmin(cute.arch.fmax(d, cutlass.Float32(0.0)),
                          cutlass.Float32(1.0))


def _eq_mask(a, b):
    """1.0 when the integers a == b, else 0.0 -- branch free.

    Used to seed the unit diagonal of the triangular inverse without a
    dynamically indexed register write (which would sink the column into local
    memory). Loop invariant for the same reason as `_ge_mask` above.
    """
    d = cutlass.Float32(a - b)
    ad = cute.arch.fmax(d, -d)
    return cutlass.Float32(1.0) - cute.arch.fmin(ad, cutlass.Float32(1.0))


# ---------------------------------------------------------------------------
# D0.9 MMA op constructors. Positional arguments only (hw_kb
# `tcgen05_umma_issue_cta_pair` gotcha 3).
# ---------------------------------------------------------------------------
def _mma_ab():
    """Groups (a)/(b): M = 64 tokens, N = 64 tokens, K = 128 features, A/B smem."""
    return cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        cutlass.BFloat16, cutlass.Float32, (CHUNK, CHUNK, 16),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))


def _mma_m128():
    """Groups (d)/(e)/(g)/(h): M = 128 features, N = 64 tokens, A from TMEM.

    One op serves all four: they differ only in the K extent (128 for the state
    groups, 64 for the T/P groups), which lives in the fragment, not the atom.
    """
    return cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        cutlass.BFloat16, cutlass.Float32, (DIM, CHUNK, 16),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))


def _mma_su():
    """Group (i): M = 128 features, N = 128 features, K = 64 tokens.

    The B operand is the very same K tile bytes as groups (a)/(b)/(d), read
    through an MN-major descriptor (N = feature, K = token).
    """
    return cute.make_tiled_mma(tcgen05.MmaF16BF16Op(
        cutlass.BFloat16, cutlass.Float32, (DIM, DIM, 16),
        tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
        tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN))


# ---------------------------------------------------------------------------
# D0.10 step 4: the 64 x 64 unit lower triangular inverse, `mma.sync` bypass.
#
# The atoms are all warp level and all derived from one 16 x 16 x 16 tiled MMA,
# so every fragment in the routine below is that MMA's own thread-value map and
# the exchange faces are read / written by the matrix atoms (`LDSM.16.M88.4`,
# `LDSM.16.MT88.4`, `STSM.16.M88.4`) rather than element by element.
#
# The permutation `(16, 16, 16)` is what turns the m16n8k16 atom into a tile
# that covers a whole 16 x 16 block: two `HMMA.16816` per `cute.gemm`, an A
# fragment of one `ldmatrix.x4`, a B fragment of one `ldmatrix.x4.trans` and a
# C fragment of one `stmatrix.x4` (hw_kb
# `smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`, station B).
# ---------------------------------------------------------------------------
def _inv_atoms():
    hmma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        (1, 1, 1), (INV_B, INV_B, INV_B))
    # A is (M, K) with K contiguous on the face -> the plain member; B is
    # (N, K) read out of a row-major face, i.e. N contiguous -> the transposed
    # member. Both tiles are 16 x 16 bf16 = four 8 x 8 matrices.
    ld_a = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4),
                            cutlass.BFloat16), hmma)
    ld_b = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4),
                            cutlass.BFloat16), hmma)
    # The C fragment goes back to a row-major face at the same (m, n) -> the
    # plain `stmatrix` member.
    st_c = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 4),
                            cutlass.BFloat16), hmma)
    return hmma, ld_a, ld_b, st_c


# The four fragment movers of the routine below.  They are module level (and so
# resolve out of the module globals, not out of a closure cell) on purpose:
# `_tri_inverse` is `@cute.jit`, so its dynamic `if` bodies become regions, and
# a region may not call a nested function that captured a non-function value.
# They hold no control flow of their own, so plain Python `range` over a
# trace-time extent is the constexpr loop here -- they are never preprocessed.
def _inv_load_a(ld_a, thr_la, rA, blk):
    cute.copy(ld_a, thr_la.partition_S(blk), ld_a.retile(rA))


def _inv_load_b(ld_b, thr_lb, rB, blk):
    cute.copy(ld_b, thr_lb.partition_S(blk), ld_b.retile(rB))


def _inv_zero_c(rC):
    for e in range(cute.size(rC)):
        rC[e] = cutlass.Float32(0.0)


def _inv_store_c(st_c, thr_sc, rC, rCb, blk, sgn):
    s = cutlass.Float32(sgn)
    for e in range(cute.size(rC)):
        rCb[e] = cutlass.BFloat16(s * rC[e])
    cute.copy(st_c, st_c.retile(rCb), thr_sc.partition_D(blk))


def _inv_frags(atoms, lane):
    """The register state of `_tri_inverse`, built once per role, not per call.

    `_tri_inverse` is inlined at its call site, so an `alloca` written inside it
    is an `alloca` inside WG A's chunk loop, and an `alloca` in a staged loop
    body is not guaranteed to be promoted out of local memory (hw_kb
    `warp_specialization` pit 7 / pit 6).  Building them here -- above every
    loop, inside the role branch -- keeps the whole routine in registers.
    """
    hmma = atoms[0]
    thr_h = hmma.get_slice(lane)
    id16 = cute.make_identity_tensor((INV_B, INV_B))
    return (
        cute.make_rmem_tensor(thr_h.partition_A(id16).shape, cutlass.BFloat16),
        # The second A fragment. Every phase below runs two products back to
        # back where the *second* one's A operand is a block that is already
        # final before the first product starts; a second fragment is what lets
        # its `ldmatrix` be issued ahead of the first product instead of behind
        # it, so the load latency overlaps the first `HMMA` pair and its
        # `stmatrix` rather than sitting in front of its consumer.  Four
        # registers per thread.
        cute.make_rmem_tensor(thr_h.partition_A(id16).shape, cutlass.BFloat16),
        cute.make_rmem_tensor(thr_h.partition_B(id16).shape, cutlass.BFloat16),
        cute.make_rmem_tensor(thr_h.partition_C(id16).shape, cutlass.Float32),
        cute.make_rmem_tensor(thr_h.partition_C(id16).shape, cutlass.BFloat16),
        cute.make_rmem_tensor((8,), cutlass.Float32),
        cute.make_rmem_tensor((8,), cutlass.Float32),
        cute.make_rmem_tensor((8,), cutlass.BFloat16),
    )


@cute.jit
def _tri_inverse(wS0, wS1, atoms, frags, wrp, lane):
    """T = (I + L)^{-1} on the two staging faces; four warps, three phases.

    `wS0` is the 64 x 64 bf16 face holding the strictly lower L (its strictly
    *upper* blocks are structurally zero and are never read, so they are this
    routine's inter-level scratch -- the D0.4 "the inter-segment exchange
    reuses the staging faces" line); `wS1` receives T.

    With 16 x 16 blocks M_pq of M = I + L, and using

        [[A, 0], [C, B]]^{-1} = [[A^-1, 0], [-B^-1 C A^-1, B^-1]]

    at two levels of doubling:

      phase 0  warp w inverts the diagonal block M_ww at two 8-wide levels:
               its two 8 x 8 diagonal sub-blocks A / B are solved *at the same
               time* by a column-parallel forward substitution held entirely in
               registers (one lane per row, the already-final row broadcast by
               `shfl`, lanes of A broadcasting from A's rows and lanes of B
               from B's), and the 8 x 8 off-diagonal sub-block C is closed by
               two `mma.sync` products; the strictly upper part of the block
               row of T is zeroed.
      phase 1  warps 0 / 1 close the two 32 x 32 diagonal blocks:
               W = M_10 T_00 then T_10 = -T_11 W, four `HMMA.16816` each;
               every warp closes the phase by issuing the `ldmatrix` pair of
               the 64 level's j = 0 product, whose operands are already final.
      phase 2  all four warps close the 64 level, one 16 x 16 output block per
               warp per step: W2 = M_LL T_AA then T_LL = -T_BB W2, eight
               `HMMA.16816` each.

    Phase 1 keeps its intermediate inside one warp (the W block a warp writes
    is the W block it reads back), so that exchange costs a `sync_warp` and not
    a warpgroup rendezvous.  Phase 2's does cross warps -- that is the price of
    spreading the level over four of them -- so it meets: four blocking meets
    below, all of them the warpgroup-wide `NBAR_WGA`.  The phase 1 -> phase 2
    one was tried as a *half* warpgroup meet on `NBAR_WGA_L` / `NBAR_WGA_R`, on
    the grounds that nothing step 1 of phase 2 reads or writes crosses between
    the warps that own output column slab 0 and those that own slab 1 -- true
    of step 1's own operands, but not of the step-2 operand load that sits
    immediately after that seam (see the note there).  It is back to full width.
    The pre-meet operand issues below are GPU-graded by
    `notes/probe_m0_step7_tri_inverse.py`: all five stages PASS with the
    readings of the preceding forms of this routine unchanged
    (`I1-identity 0.0`, `I2-subdiag 0.0`, `I4-blockdiag 3.2196e-3`,
    `I3-random 4.5776e-4` rel).
    """
    hmma, ld_a, ld_b, st_c = atoms

    # 16 x 16 block views of the two faces. `bX` is indexed (row block, column
    # block) and is the A-operand frame; `bXt` is the (column, row) relabel --
    # the same bytes, the swizzle still on the pointer -- and is the B-operand
    # frame, where element (n, k) of block (nb, kb) is face[16 kb + k][16 nb + n].
    bL = cute.flat_divide(wS0, (INV_B, INV_B))
    bT = cute.flat_divide(wS1, (INV_B, INV_B))
    bLt = cute.flat_divide(
        cute.make_tensor(wS0.iterator, cute.select(wS0.layout, mode=[1, 0])),
        (INV_B, INV_B))
    bTt = cute.flat_divide(
        cute.make_tensor(wS1.iterator, cute.select(wS1.layout, mode=[1, 0])),
        (INV_B, INV_B))
    # 8 bf16 = 16 B = the SW128 swizzle's own unit: the widest vector that is
    # contiguous on a swizzled face, and the granule phase 0 moves rows in.
    vL = cute.flat_divide(wS0, (1, 8))
    vT = cute.flat_divide(wS1, (1, 8))

    rA, rA2, rB, rC, rCb, mrow, trow, vec = frags
    thr_la = ld_a.get_slice(lane)
    thr_lb = ld_b.get_slice(lane)
    thr_sc = st_c.get_slice(lane)

    # -- phase 0: the four 16 x 16 diagonal inverses ------------------------
    # The 16 x 16 block splits the same way the 32 and 64 levels do, one level
    # lower:  M_ww = [[A, 0], [C, B]]  with A, B unit lower 8 x 8, so
    #
    #     M_ww^-1 = D^-1 - D^-1 . C16 . D^-1,      D = [[A, 0], [0, B]],
    #
    # exactly because C16 = [[0, 0], [C, 0]] squares to zero.  The two 8 x 8
    # substitutions that build D^-1 are independent, so they run in the same
    # seven steps instead of one behind the other: the serial `shfl -> FFMA`
    # chain of this phase is 7 links deep, not 15, and carries 28 broadcasts
    # per lane instead of 120.  The correction is the two matrix products
    # above, on the 16 x 16 atoms phases 1 and 2 already use -- and the A
    # operand of the first of them is the raw L block rather than C16, see the
    # note at its `ldmatrix` below.
    #
    # `notes/probe_m0_step7_tri_inverse.py` grades this form on GPU against a
    # numpy inverse; its `I4-blockdiag` stage exists for this phase in
    # particular (L nonzero only inside the 16 x 16 diagonal blocks, so phases 1
    # and 2 multiply zeros and cannot mask a phase-0 error):
    #   I0-face n/a, I1-identity 0.0, I2-subdiag 0.0,
    #   I4-blockdiag 3.22e-3, I3-random 4.58e-4 (rel, bar 2e-2 -- the
    #   correction rides two bf16 rounding steps).
    # Those five readings have now been taken three times -- on the four-warp
    # phase 2 plus the `rA2` preloads, and again on a three-segment re-cut of
    # the 32 and 64 levels (measured slower, so the cut below is back to the
    # two-phase one) -- and came back unchanged to the last bit every time,
    # which is the expected signature of this class of edit: they
    # move *which warp* issues a product, *when* an `ldmatrix` goes out and
    # which products are dropped for multiplying a structurally zero block,
    # never the per-output-block operand pair, the k order or the accumulator
    # init, so any different number would have been an indexing or race error
    # rather than rounding.
    #
    # Lane l takes row i = l % 16 of its warp's block; the two halves of the
    # warp run the same scan on their own copy so that every lane reaches the
    # `shfl` (the broadcast source is the same row inside each half).
    i = lane % INV_B
    hlf = lane // INV_B
    r0 = wrp * INV_B
    hb = i // 8              # which 8 x 8 diagonal sub-block owns this row
    ii = i % 8               # the row inside that sub-block
    # The A operand of the first correction product below, issued here -- at the
    # very top of the phase, above the `mrow` load, the seven-level substitution
    # and the `sync_warp` -- because it depends on nothing this routine does.
    #
    # It used to be C16, the L diagonal block with its two 8 x 8 diagonal
    # triangles zeroed *in place*, and that zeroing is what used to chain this
    # `ldmatrix` behind a warp-wide store pass and a `sync_warp`.  The zeroing
    # is unnecessary.  Feeding the raw block L_ww = [[A_str, 0], [C, B_str]]
    # instead of C16 = [[0, 0], [C, 0]] gives
    #     Y = L_ww . D^-1 = [[A_str A^-1, 0     ], [C A^-1    , B_str B^-1    ]]
    #     Z = D^-1 . Y    = [[A^-1 A_str A^-1, 0], [B^-1 C A^-1, B^-1 B_str B^-1]]
    # so Z differs from D^-1 C16 D^-1 only in the two 8 x 8 *diagonal*
    # sub-blocks -- exactly the bytes the trow store at the end of this phase
    # overwrites unconditionally.  The lower-left 8 x 8, the only part of Z
    # that survives, is B^-1 C A^-1 either way; the upper-right 8 x 8 is
    # structurally zero either way, because L_ww is strictly lower triangular
    # (step 2 writes the face through `_sel_lt_s32(col, row, ...)`, whose
    # else-arm is an exact `0.0`) and D^-1 is block diagonal.  Whatever the
    # diagonal sub-blocks of Z come out as -- including an overflow to inf,
    # since every output element is a separate register -- is dead: the
    # `stmatrix` of the second product is followed by the trow store, and the
    # first read of this block by any other phase is past the WG A meet that
    # closes phase 0.
    #
    # So the operand is untouched input: final at the WG A rendezvous the
    # caller took before entering, and nothing in this routine writes the
    # diagonal blocks of the L face.  One `STS.128` per lane (duplicated
    # across the two warp halves, i.e. a same-address store conflict) goes with
    # it, and the `LDSM` latency now runs under the whole substitution.
    _inv_load_a(ld_a, thr_la, rA, bL[(None, None, wrp, wrp)])
    # Only the diagonal sub-block's own row is needed in registers now; C goes
    # through the matrix products, not through the scan.
    cute.autovec_copy(vL[(0, None, r0 + i, wrp * 2 + hb)], vec)
    for e in cutlass.range_constexpr(8):
        mrow[e] = cutlass.Float32(vec[e])
    # The one part of the T row that genuinely changes every chunk and is
    # written by nothing else: the 8 x 8 sub-block of the *diagonal* block that
    # is off the diagonal for this row (the upper-right one for the rows of A,
    # the lower-left one for the rows of B).  The products below read the whole
    # 16 x 16 diagonal block as D = blockdiag(A^-1, B^-1), so those bytes must
    # be zero; the lower-left one carries -B^-1 C A^-1 from the previous chunk,
    # so it cannot be left alone.  The rest of the row used to be zeroed here
    # too -- four more `STS.128` per lane per chunk, in the most latency-bound
    # block WG A has -- and does not need to be: the strictly *lower* blocks of
    # the row are rewritten whole by phases 1 and 2, the diagonal block by the
    # second product's `stmatrix` plus the trow store, and the strictly *upper*
    # blocks are only ever written with zeros, so they are zeroed once per
    # staging slot in WG A's prologue instead of once per chunk.
    # This store and the trow store below now land in *different* 8-wide column
    # blocks of the same row, and each lane issues both of them, so the
    # `sync_warp` that used to order the two passes against each other is gone
    # as well -- the two warp halves still write the same bytes with the same
    # values, which is a same-address conflict but not a hazard.
    for e in cutlass.range_constexpr(8):
        vec[e] = cutlass.BFloat16(0.0)
    cute.autovec_copy(vec, vT[(0, None, r0 + i, wrp * 2 + (1 - hb))])
    # T row ii starts as e_ii (the unit diagonal is implicit: the face holds
    # the strictly lower L, never the diagonal), then rows are eliminated in
    # order.  The broadcast source is row m *of this row's own sub-block*, so
    # the A lanes and the B lanes advance together.
    for k in cutlass.range_constexpr(8):
        trow[k] = _eq_mask(k, ii)
    src0 = hlf * INV_B + hb * 8
    for m in cutlass.range_constexpr(7):
        # Row m is final at step m; every row below it takes -M[ii][m] times it,
        # and every row at or above it takes nothing.  That second clause used
        # to be spelled `_ge_mask(ii, m + 1) * mrow[m]`, and the mask was
        # redundant: `mrow[m]` *is* L[ii][m] inside this 8 x 8 diagonal
        # sub-block, and L is strictly lower triangular by construction -- step
        # 2 writes the face through `_sel_lt_s32(col, row, ...)`, whose
        # else-arm is an exact `0.0`, so every entry with m >= ii is exactly
        # zero in the bytes this row was loaded from (the in-place zeroing of
        # the triangles happens after the load and does not enter here).  The
        # guard therefore multiplied by 1.0 wherever it mattered and by 0.0
        # wherever the operand was already 0.0.  Dropping it takes one `FMUL`
        # per level -- seven off the front of the deepest dependent chain WG A
        # has -- and the seven loop-invariant mask floats that `_ge_mask` kept
        # in registers across the whole chunk loop with it, with nothing added.
        # Bit-exact: `0.0 * x == 0.0` and `1.0 * x == x` for the finite `x`
        # here, so the substitution sees the same numbers it did before.
        act = mrow[m]
        for k in cutlass.range_constexpr(m + 1):
            trow[k] = trow[k] - act * cute.arch.shuffle_sync(trow[k], src0 + m)
    # Publication of D^-1, branch free: the diagonal 8 x 8 sub-block of this
    # row takes the row just solved.  Both halves issue it, with identical
    # bytes, because both halves solved the same row.  The single `sync_warp`
    # below is what makes the whole 16 x 16 diagonal block final for the
    # `ldmatrix` pair of the correction: this store and the zero store issued
    # before the substitution are the only two writes to it, and between them
    # they cover it.
    for e in cutlass.range_constexpr(8):
        vec[e] = cutlass.BFloat16(trow[e])
    cute.autovec_copy(vec, vT[(0, None, r0 + i, wrp * 2 + hb)])
    cute.arch.sync_warp()
    # The correction, warp private, on one of the six structurally zero upper
    # blocks of the L face: (0, 3) / (0, 2) / (1, 3) / (1, 2) for warps 0..3,
    # all of them dead until phases 1 and 2 write their own scratch there.
    sr = wrp // 2
    sc = 3 - wrp % 2
    # Y = L_ww . D^-1.  The second product's A is D^-1 itself, which the
    # `sync_warp` above has just made final and which nothing between here and
    # there touches (the first product writes the *L* face), so its `ldmatrix`
    # runs underneath the first `HMMA` pair, its `stmatrix` and the `sync_warp`
    # that follows them instead of in front of its own consumer.  The first
    # product's A (L_ww) went out at the top of the phase.
    _inv_load_a(ld_a, thr_la, rA2, bT[(None, None, wrp, wrp)])
    _inv_load_b(ld_b, thr_lb, rB, bTt[(None, None, wrp, wrp)])
    _inv_zero_c(rC)
    cute.gemm(hmma, rC, rA, rB, rC)
    _inv_store_c(st_c, thr_sc, rC, rCb, bL[(None, None, sr, sc)], 1.0)
    cute.arch.sync_warp()
    # T_ww = D^-1 - D^-1 . Y.  The product's upper-right 8 x 8 is structurally
    # zero and its lower-left 8 x 8 is the answer; its two diagonal 8 x 8 are
    # the dead terms the un-zeroed A operand adds.  The store lands the whole
    # 16 x 16 block and the two 8 x 8 diagonals are written back on top of it,
    # which is what makes those terms dead.
    _inv_load_b(ld_b, thr_lb, rB, bLt[(None, None, sc, sr)])
    _inv_zero_c(rC)
    cute.gemm(hmma, rC, rA2, rB, rC)
    _inv_store_c(st_c, thr_sc, rC, rCb, bT[(None, None, wrp, wrp)], -1.0)
    cute.arch.sync_warp()
    for e in cutlass.range_constexpr(8):
        vec[e] = cutlass.BFloat16(trow[e])
    cute.autovec_copy(vec, vT[(0, None, r0 + i, wrp * 2 + hb)])
    cute.arch.barrier(barrier_id=NBAR_WGA, number_of_threads=WARPGROUP)

    # -- phase 1: the two 32 x 32 diagonal blocks --------------------------
    # This region was cut a second way -- the 32 level and the 64 level's
    # step 1 merged and re-cut into three segments of four block products per
    # warp, dropping the four products that multiply a structurally zero block
    # and shortening the longest dependent run through the region from six
    # products to five.  It measured 6.8% *slower* on `t_ns` than the two-phase
    # cut below even though both of its static figures improved, so the cut is
    # back to the two-phase one.  What that re-cut also did, and what the
    # figures do not show, is give the region three divergent `wrp < 2`
    # regions, each with a different instruction stream on the two halves of
    # WG A, where the form below has one (phase 1) and runs phase 2 as a single
    # stream over all four warps; and it put warps 2 / 3's `ldmatrix` and
    # `stmatrix` traffic alongside the 32 level, which is the region's critical
    # path.  Only the pre-meet operand issue at the end of phase 1 survives
    # from it, in the branch-free form below.
    kb0 = wrp % 2      # this warp's W2 row block / T_LL output row block
    slab = wrp // 2    # this warp's 16-wide output column slab
    if wrp < 2:
        ba = wrp * 2
        bb = ba + 1
        # W = M_10 . T_00, parked in the structurally zero block (ba, bb) of L.
        # Same preload as phase 0: T_11 is a phase-0 output and nothing in this
        # phase writes it, so its `ldmatrix` goes out with the first product's.
        _inv_load_a(ld_a, thr_la, rA, bL[(None, None, bb, ba)])
        _inv_load_a(ld_a, thr_la, rA2, bT[(None, None, bb, bb)])
        _inv_load_b(ld_b, thr_lb, rB, bTt[(None, None, ba, ba)])
        _inv_zero_c(rC)
        cute.gemm(hmma, rC, rA, rB, rC)
        _inv_store_c(st_c, thr_sc, rC, rCb, bL[(None, None, ba, bb)], 1.0)
        cute.arch.sync_warp()
        # T_10 = -T_11 . W
        _inv_load_b(ld_b, thr_lb, rB, bLt[(None, None, bb, ba)])
        _inv_zero_c(rC)
        cute.gemm(hmma, rC, rA2, rB, rC)
        _inv_store_c(st_c, thr_sc, rC, rCb, bT[(None, None, bb, ba)], -1.0)
    # The j = 0 operand pair of phase 2's step 1 waits on nothing this phase
    # produces: its A is the untouched input block M_LL[kb0, 0] of L and its B
    # is T[(0, slab)], which is phase-0 output (T[(0,0)]) or one of the
    # structurally zero blocks phase 0 zeroed (T[(0,1)]) -- and phase 1 writes
    # only L (0,1) / (2,3) and T (1,0) / (3,2).  Issuing the two `ldmatrix`
    # here, on all four warps and outside the branch above, puts their latency
    # inside the meet instead of in front of phase 2's first product: warps
    # 2 / 3 are waiting at the meet anyway, and on warps 0 / 1 the two loads
    # are issued after their last use of `rA` / `rB`, so nothing of the 32
    # level's chain moves.  The two fragments (8 registers) stay live across
    # the meet; WG A's budget is `REG_WG_A`.
    _inv_load_a(ld_a, thr_la, rA, bL[(None, None, 2 + kb0, 0)])
    _inv_load_b(ld_b, thr_lb, rB, bTt[(None, None, slab, 0)])
    # This seam is warpgroup wide, and it has to be -- the half-warpgroup form
    # it replaces (`NBAR_WGA_L` / `NBAR_WGA_R`, 64 threads each, selected by
    # `wrp < 2`) let warp 3 read a block warp 1 had not necessarily written.
    #
    # Step 1's *own* operands do split along the halves, which is what that
    # form was reasoning from: step 1 reads M_LL (untouched input), T (0, slab)
    # and T (1, slab); the slab-0 pair (warps 0 / 1) needs an ordering against
    # warp 0's phase-1 output T (1,0) and the slab-1 pair (warps 2 / 3) reads
    # only phase-0 output T (1,1) and the zeroed upper block T (0,1).  The write
    # side splits too: W2 is parked in L (kb0, 2 + slab), i.e. (0,2)/(1,2) for
    # the slab-0 pair and (0,3)/(1,3) for the slab-1 pair.
    #
    # What does *not* split is the first instruction after this meet.  Step 2's
    # kb = 0 A operand, T (2 + kb0, 2), is loaded there rather than after the
    # meet between the two steps, and for kb0 = 1 that block is T (3,2) -- warp
    # 1's phase-1 output.  The warps with kb0 = 1 are warps 1 and 3, one in each
    # half, so warp 3's `ldmatrix` of it was ordered against warp 1's `stmatrix`
    # of it by nothing at all: `NBAR_WGA_R` joins warps 2 / 3 only, and warp 3
    # reaches it immediately (it does no phase-1 work), so it would issue that
    # load while warp 1 is still in the 32 level.  The meet below orders the
    # *consumer* (`cute.gemm`) but not the load, and a load is where the smem
    # read happens.  The dependency is inherent to the 2 x 2 warp grid: step 1's
    # consumers of T (1,0) are the slab-0 warps (a row of the grid) and step 2's
    # consumers of T (3,2) are the kb0 = 1 warps (a column), so whichever way
    # the four warps are split into two halves, one of the two phase-1 outputs
    # crosses the split.  This is the third of the four meets inside the
    # routine; WG A takes five per chunk in all, the sixth having been the
    # redundant one in front of `P ready` (see the caller).
    cute.arch.barrier(barrier_id=NBAR_WGA, number_of_threads=WARPGROUP)

    # -- phase 2: the 64 level, one 16 x 16 output block per warp ----------
    # The 64 level is a 32 x 32 output built from two 32x32x32 products, and at
    # the 16 x 16 granularity of the atom that is eight independent block
    # products per step -- four in each of the two steps.  Both steps are
    # therefore spread over all four warps (one output block each) instead of
    # over two warps (two output blocks each): the same eight `cute.gemm` calls
    # and the same sixteen `ldmatrix`, but two warps deep instead of four, and
    # warps 2 / 3 stop idling at the phase-1 rendezvous while warps 0 / 1 carry
    # the whole level.  The price is that W2 now crosses warps -- warp
    # (2 s + kb) produces the block that warps (2 s) and (2 s + 1) both read
    # -- so the `sync_warp` between the two steps becomes a WG A meet.
    # Both A operands of step 2 are phase-0 / phase-1 outputs that step 1 does
    # not touch (step 1 writes the L face, step 2 reads T blocks
    # (2 + kb0, 2 + kb)), so neither of their two `ldmatrix` is left in front of
    # its consumer: the kb = 0 one is issued here, ahead of step 1's products,
    # and the kb = 1 one just below, ahead of the meet.
    _inv_load_a(ld_a, thr_la, rA2, bT[(None, None, 2 + kb0, 2)])
    # step 1: W2[kb0, slab] = sum_j M_LL[kb0, j] . T_AA[j, slab], parked in
    # the free upper-right 32 x 32 of L at block (kb0, 2 + slab) -- the same
    # four scratch blocks the two-warp form used, one per warp not two.  The
    # j = 0 pair is already in registers (issued before the meet above), so the
    # first product goes out with no load in front of it and the j = 1 pair's
    # `ldmatrix` runs underneath it.
    _inv_zero_c(rC)
    cute.gemm(hmma, rC, rA, rB, rC)
    _inv_load_a(ld_a, thr_la, rA, bL[(None, None, 2 + kb0, 1)])
    _inv_load_b(ld_b, thr_lb, rB, bTt[(None, None, slab, 1)])
    cute.gemm(hmma, rC, rA, rB, rC)
    # Step 2's kb = 1 A operand, T (2 + kb0, 3), is not just free of the meet
    # below -- it is final before the meet above: T (3,3) is phase-0 output and
    # T (2,3) is one of the strictly upper blocks the caller zeroed per staging
    # slot, and step 1 writes only the L face.  So it goes out here, in the same
    # place the j = 0 pair of step 1 goes out one phase up: after the last use
    # of `rA`, so the 64 level's chain does not move, and ahead of the meet, so
    # its `LDSM` latency is spent inside the rendezvous and the store below
    # rather than in front of its own consumer.  Four registers live across the
    # meet, next to `rA2`'s four; WG A's budget is `REG_WG_A`.
    _inv_load_a(ld_a, thr_la, rA, bT[(None, None, 2 + kb0, 3)])
    _inv_store_c(st_c, thr_sc, rC, rCb, bL[(None, None, kb0, 2 + slab)], 1.0)
    cute.arch.barrier(barrier_id=NBAR_WGA, number_of_threads=WARPGROUP)
    # step 2: T_LL[kb0, slab] = -sum_kb T_BB[kb0, kb] . W2[kb, slab].  Both A
    # operands are already in registers (`rA2` from before step 1, `rA` from
    # before the meet), so the only loads left here are the two W2 blocks, which
    # genuinely have to wait for the partner warp.
    _inv_zero_c(rC)
    _inv_load_b(ld_b, thr_lb, rB, bLt[(None, None, 2 + slab, 0)])
    cute.gemm(hmma, rC, rA2, rB, rC)
    _inv_load_b(ld_b, thr_lb, rB, bLt[(None, None, 2 + slab, 1)])
    cute.gemm(hmma, rC, rA, rB, rC)
    _inv_store_c(st_c, thr_sc, rC, rCb, bT[(None, None, 2 + kb0, slab)], -1.0)
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.barrier(barrier_id=NBAR_WGA, number_of_threads=WARPGROUP)


# ---------------------------------------------------------------------------
# Device kernel
# ---------------------------------------------------------------------------
@cute.kernel
def _gdn_kernel(
    tma_k: cute.CopyAtom, gmemK: cute.Tensor,   # [T, DIM, HK] bf16 TMA view
    tma_q: cute.CopyAtom, gmemQ: cute.Tensor,   # [T, DIM, HQ] bf16 TMA view
    tma_v: cute.CopyAtom, gmemV: cute.Tensor,   # [T, DIM, HV] bf16 TMA view
    tma_o: cute.CopyAtom, gmemO: cute.Tensor,   # [DIM, T, HV] bf16 TMA store view
    mState: cute.Tensor,    # [N, HV, DIM, DIM] fp32, k-last, read-only
    mALog: cute.Tensor,     # [HV] fp32
    mA: cute.Tensor,        # [T, HV] bf16
    mDtBias: cute.Tensor,   # [HV] fp32
    mB: cute.Tensor,        # [T, HV] bf16
    mCuSeq: cute.Tensor,    # [N + 1] int64
    mOut: cute.Tensor,      # [T, HV, DIM] bf16
    mNewState: cute.Tensor,  # [N, HV, DIM, DIM] fp32, k-last
    mDbg: cute.Tensor,      # probe-only stage sink; = mNewState when dbg_on is False
    k_layout: cute.ComposedLayout,
    q_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    stage_layout: cute.ComposedLayout,
    kb_layout: cute.ComposedLayout,
    qb_layout: cute.ComposedLayout,
    kmn_layout: cute.ComposedLayout,
    out_layout: cute.ComposedLayout,
    epi_tile: cute.Tile,
    scale: cutlass.Float32,
    n_item: cutlass.Int32,
    tx_bytes: cutlass.Constexpr,
    n_sub: cutlass.Constexpr,
    epi_n: cutlass.Constexpr,
    dbg_on: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    _seg("cta.prologue.enter")

    # D0.7: warpgroup role split. wg 0 = WG A (gating mask / inverse / P),
    # wg 1 = WG B (TMEM math, state load/store, epilogue), wg 2 = the four
    # single-warp specialists: warp 8 = UMMA issue, warp 9 = TMA load,
    # warp 10 = gating scalars, warp 11 = output store (step 6).
    # The role predicate is derived from the warp-uniform warp index, not from
    # `tidx`: `setmaxnreg` is `.sync.aligned` and ptxas has to see that all four
    # warps of a warpgroup take the same side of the branch (hw_kb
    # `warp_specialization` pit 3). `lane_in_wg` stays a per-thread quantity --
    # it indexes copy partitions, it is not a predicate.
    wg = warp_idx // 4
    lane_in_wg = tidx % WARPGROUP

    # -----------------------------------------------------------------------
    # D0.4 smem plan at the D1.2 depths: K ring x2, Q x2, V x1 (16 KiB each),
    # two groups of the two 8 KiB staging faces, the double-buffered gating
    # arrays, the mbarrier table. 101,888 + 16 KiB (Q) + 16 KiB (staging) B.
    # -----------------------------------------------------------------------
    smem = utils.SmemAllocator()
    mbar = smem.allocate_array(cutlass.Int64, MBAR_COUNT, byte_alignment=16)
    # D0.14: the one word `tcgen05.alloc` writes the TMEM base address into.
    tmem_hold = smem.allocate_array(cutlass.Int32, 1, byte_alignment=16)
    sK = smem.allocate_tensor(element_type=cutlass.BFloat16,
                              layout=k_layout.outer, byte_alignment=128,
                              swizzle=k_layout.inner)
    sQ = smem.allocate_tensor(element_type=cutlass.BFloat16,
                              layout=q_layout.outer, byte_alignment=128,
                              swizzle=q_layout.inner)
    sV = smem.allocate_tensor(element_type=cutlass.BFloat16,
                              layout=v_layout.outer, byte_alignment=128,
                              swizzle=v_layout.inner)
    # D0.4 staging s0 / s1 (8 KiB per group): the two 64 x 64 bf16 exchange
    # faces WG A produces into. Both are laid out as the B-operand face of the
    # M = 128, K = 64 groups (g)/(h), so the UMMA descriptor reads them without
    # a second view: s0 carries L first and then P (the inverse consumes L
    # before P overwrites it), s1 carries T. D1.2 gives each of them
    # STAGE_SLOTS groups so that WG A's chunk c+1 faces do not collide with the
    # chunk c faces groups (g)/(h) are still reading.
    sS0 = smem.allocate_tensor(element_type=cutlass.BFloat16,
                               layout=stage_layout.outer, byte_alignment=1024,
                               swizzle=stage_layout.inner)
    sS1 = smem.allocate_tensor(element_type=cutlass.BFloat16,
                               layout=stage_layout.outer, byte_alignment=1024,
                               swizzle=stage_layout.inner)
    # D0.4 out tile: the D0.13 output face, (EPI_M = 128 v, EPI_N = 32 tokens,
    # n_sub) MN_SW128 -- the `stmatrix` transpose writes it and the TMA store
    # reads it. The swizzle rides on the pointer and there is no second view of
    # these bytes (hw_kb `tcgen05_epilogue_t2r_stsm_tma_store` 12). D1.3 gives
    # it OUT_SLOTS copies (16 KiB each): the sub-tile mode carries both, so slot
    # `os` is the contiguous run of sub-tiles [os * n_sub, (os + 1) * n_sub).
    sOut = smem.allocate_tensor(element_type=cutlass.BFloat16,
                                layout=out_layout.outer, byte_alignment=1024,
                                swizzle=out_layout.inner)
    # D0.11 (G5): the gating array WG B reads, double buffered and *field
    # major*: [slot][field][token], stride (320, 64, 1), 2560 B in total -- the
    # same size D2.3's token-major record had, re-cut so that the four adjacent
    # tokens of one field are four adjacent words. Every WG B read of it is a
    # column read (see `KS_COL_RUNS` / `COL_COL_RUNS`), so this is the only
    # layout that lets them be vector loads at all. Each field row is
    # CHUNK * 4 = 256 B, a multiple of 16, so every row -- in every slot -- is
    # 16 B aligned and a `ld.shared.v4` of a four-column run is legal.
    #
    # The write side pays for it: warp 10's lane l writes tokens 2l and 2l + 1
    # of each field, i.e. words 2l and 2l + 1 of a row, which is the same 2-way
    # conflict any 64-word row takes from a 32-lane warp. That is 8 stores in
    # the publishing warp against the 256 -> 96 shared loads it saves in each of
    # WG B's 128 lanes.
    sGateB = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((GATE_SLOTS, GATE_FIELDS, CHUNK),
                                stride=(GATE_FIELDS * CHUNK, CHUNK, 1)),
        byte_alignment=16)
    # D0.10 lever (2): the two faces WG A's mask segments read, shaped for the
    # vector loads rather than for a field-major sweep (hw_kb
    # `branchfree_masked_elementwise_pass` (c)). `G_t` is not a field of the
    # array above at all -- WG B never reads it -- and lives here instead, in
    # two copies:
    #   `sGatePair`  (G_t, beta_t) interleaved at an 8 B stride, so the *row*
    #                scalars of an element group come back in one `LDS.64`;
    #   `sGateCol`   G_t alone, column contiguous, so the *column* scalars of a
    #                two-column run come back in one `LDS.64` (the run start is
    #                an even token, hence 8 B aligned).
    # One copy cannot serve both roles: a row read wants (G, beta) adjacent and
    # a column read wants (G_t, G_t+1) adjacent. 1.5 KiB of smem buys
    # 96 -> 18 shared loads per chunk per thread across 128 threads. These two
    # faces are WG A's; the field-major array above is WG B's, and the two sets
    # of readers never touch each other's copy.
    sGatePair = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((GATE_SLOTS, CHUNK, 2),
                                stride=(2 * CHUNK, 2, 1)),
        byte_alignment=16)
    sGateCol = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((GATE_SLOTS, CHUNK), stride=(CHUNK, 1)),
        byte_alignment=16)

    # D0.14 CTA prologue: the mbarrier table is initialised once, then the one
    # all-hands NamedBarrier of the whole kernel.
    if warp_idx == 0:
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(K_SLOTS):
                cute.arch.mbarrier_init(mbar + BAR_K_FULL + i, 1)
                cute.arch.mbarrier_init(mbar + BAR_K_EMPTY + i, 1)
            for i in cutlass.range_constexpr(Q_SLOTS):
                cute.arch.mbarrier_init(mbar + BAR_Q_FULL + i, 1)
                cute.arch.mbarrier_init(mbar + BAR_Q_EMPTY + i, 1)
            cute.arch.mbarrier_init(mbar + BAR_V_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_V_EMPTY, WARPGROUP)
            for i in cutlass.range_constexpr(GATE_SLOTS):
                cute.arch.mbarrier_init(mbar + BAR_GATE_FULL + i, 1)
                cute.arch.mbarrier_init(mbar + BAR_GATE_EMPTY + i, 2 * WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_ACC_A_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_A_EMPTY, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_ACC_B_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_B_EMPTY, WARPGROUP)
            for i in cutlass.range_constexpr(S16_SEGS):
                cute.arch.mbarrier_init(mbar + BAR_S16_SEG + i, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_S16_EMPTY, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_KS_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_X_READY, WARPGROUP)
            for i in cutlass.range_constexpr(STAGE_SLOTS):
                cute.arch.mbarrier_init(mbar + BAR_T_READY + i, WARPGROUP)
                cute.arch.mbarrier_init(mbar + BAR_P_READY + i, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_ACC_O_FULL_E, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_O_SCALED, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_ACC_U_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_O_FULL_H, 1)
            cute.arch.mbarrier_init(mbar + BAR_U16_READY, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_U16S_READY, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_S32_DEC, WARPGROUP)
            cute.arch.mbarrier_init(mbar + BAR_S32_FULL, 1)
            cute.arch.mbarrier_init(mbar + BAR_ACC_O_EMPTY, WARPGROUP)
            for i in cutlass.range_constexpr(OUT_SLOTS):
                cute.arch.mbarrier_init(mbar + BAR_OUT_FULL + i, WARPGROUP)
                cute.arch.mbarrier_init(mbar + BAR_OUT_EMPTY + i, 1)

    # D0.14: tensormap prefetch, issued by the warps that drive the TMA.
    if warp_idx == 9:
        cpasync.prefetch_descriptor(tma_k)
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_v)
    if warp_idx == 11:
        cpasync.prefetch_descriptor(tma_o)

    # D0.3 / D0.14: one `tcgen05.alloc` of the whole 512-column block, held for
    # the CTA's lifetime, issued by the first warp of WG B (the warpgroup that
    # owns the TMEM traffic). `alloc_tmem` is a warp-wide instruction -- all 32
    # lanes must reach it, so no `elect_one` here.
    if warp_idx == 4:
        cute.arch.alloc_tmem(TMEM_COLS, tmem_hold, is_two_cta=False)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # -----------------------------------------------------------------------
    # D0.3 TMEM view of the state: lane = v (128 lanes), column = k. That is
    # byte-for-byte the k-last `[V, K]` plane of `state` / `new_state`, so the
    # item prologue and the item epilogue are a straight vector copy with no
    # transpose -- and it is also the A-operand orientation the transposed UMMA
    # groups of D0.9 want.
    # -----------------------------------------------------------------------
    tmem_ptr = cute.arch.retrieve_tmem_ptr(
        cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_hold)
    tS32 = cute.make_tensor(
        tmem_ptr + S32_COL,
        cute.make_layout((DIM, DIM), stride=(TMEM_LANE_STRIDE, 1)))
    # The bf16 operand copy S16: 128 bf16 per lane packed into 64 32-bit
    # columns. The fp32 face below is the one the r2t path writes; the MMA
    # reads the same bytes through its own A fragment layout (see frS16).
    tS16f32 = cute.make_tensor(
        tmem_ptr + S16_COL,
        cute.make_layout((DIM, DIM // 2), stride=(TMEM_LANE_STRIDE, 1)))
    tX16f32 = cute.make_tensor(
        tmem_ptr + X16_COL,
        cute.make_layout((DIM, CHUNK // 2), stride=(TMEM_LANE_STRIDE, 1)))
    tU16sf32 = cute.make_tensor(
        tmem_ptr + U16S_COL,
        cute.make_layout((DIM, CHUNK // 2), stride=(TMEM_LANE_STRIDE, 1)))

    # The lane-indexed TMEM copies of D0.11 (B1, B3, B4) and D0.13. All are
    # built on [lane, column] tiles, so the register-side and gmem-side
    # partitions of a pair index the same (v, k) elements.
    tiled_r2t = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(TMEM_REP),
                               tcgen05.Unpack.NONE),
            cutlass.Float32),
        tS32)
    tiled_t2r = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(TMEM_REP)),
            cutlass.Float32),
        tS32)
    # The bf16 pack partner of `tiled_t2r`: W = E * T / 32 = 16 for bf16 out of
    # an E = 32 source (hw_kb `tmem_r2t_recast_fragment`, the depth equation).
    tiled_s16 = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16, tcgen05.Unpack.NONE),
            cutlass.Float32),
        tS16f32)
    tiled_x16 = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8, tcgen05.Unpack.NONE),
            cutlass.Float32),
        tX16f32)
    tiled_u16 = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16, tcgen05.Unpack.NONE),
            cutlass.Float32),
        tX16f32)
    tiled_u16s = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16, tcgen05.Unpack.NONE),
            cutlass.Float32),
        tU16sf32)
    # TMEM is addressed per warp sub-partition: warp w of the CTA reaches lanes
    # 32*(w % 4) .. +31, so WG B's four warps cover the 128 v rows on the four
    # TMEM sub-partitions exactly when the slice index is the rank inside the
    # warpgroup.
    thr_r2t = tiled_r2t.get_slice(lane_in_wg)
    thr_t2r = tiled_t2r.get_slice(lane_in_wg)
    thr_s16 = tiled_s16.get_slice(lane_in_wg)
    thr_x16 = tiled_x16.get_slice(lane_in_wg)
    thr_u16 = tiled_u16.get_slice(lane_in_wg)
    thr_u16s = tiled_u16s.get_slice(lane_in_wg)
    tRS_tS = thr_r2t.partition_D(tS32)      # r2t destination: S32
    tLS_tS = thr_t2r.partition_S(tS32)      # t2r source:      S32

    # -----------------------------------------------------------------------
    # D0.9 groups (a)/(b) and the WG A path that drains their accumulators.
    #
    # Both groups are 64 x 64 x 128 bf16 -> f32 with A and B out of smem,
    # K-major; group (a) reads the one K face through both descriptors and
    # group (b) reads Q as A and the same K face as B. acc[m, n] is therefore
    # sum_d A[m, d] B[n, d]: ACC_A[i, j] = k_i . k_j and ACC_B[t, i] = q_t . k_i.
    # -----------------------------------------------------------------------
    mma_ab = _mma_ab()
    # D1.1: the ACCUMULATE field is carried by *two pre-set atoms* per op rather
    # than by `set()` on one. `tiled_mma.set()` mutates the object, so the DSL
    # has to thread the atom through every enclosing region as a loop-carried /
    # yielded value; once the lookahead put a `set()` inside the staged
    # `if c + 1 < n_chunk` nested in the chunk `for`, the mutated atom was
    # defined in the `if` region and yielded by the `for` -- "operand #0 does
    # not dominate this use", an IR verification ICE with no line of user code
    # to blame. Building the False / True variants once here, at kernel scope,
    # takes the atoms out of every region's carried set: a `cute.gemm` simply
    # picks the variant it needs, and nothing mutates anything.
    mma_ab_n = _mma_ab()
    mma_ab_n.set(tcgen05.Field.ACCUMULATE, False)
    mma_ab_y = _mma_ab()
    mma_ab_y.set(tcgen05.Field.ACCUMULATE, True)
    acc_layout = mma_ab.make_fragment_C(
        mma_ab.partition_shape_C((CHUNK, CHUNK))).layout
    tAccA = cute.make_tensor(tmem_ptr + ACC_A_COL, acc_layout)
    tAccB = cute.make_tensor(tmem_ptr + ACC_B_COL, acc_layout)
    # The (64, 64) plane of each accumulator. An M = 64 accumulator occupies 16
    # of the 32 TMEM datapaths in each of the four sub-partitions, which is why
    # the read-back atom below is the 16-datapath family (hw_kb
    # `smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`, gotcha 1) -- and it
    # is also the atom whose fragment shape has an `stmatrix` partner, so the
    # same registers go straight out to the staging faces.
    planeA = cute.flat_divide(tAccA[((None, None), 0, 0)],
                              (CHUNK, CHUNK))[(None, None, 0, 0)]
    planeB = cute.flat_divide(tAccB[((None, None), 0, 0)],
                              (CHUNK, CHUNK))[(None, None, 0, 0)]
    tiled_acc_t2r = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32),
        planeA)
    thr_acc_t2r = tiled_acc_t2r.get_slice(tidx)
    # Every register element's absolute (row, col) inside the 64 x 64 tile:
    # the identity first through the MMA's own C partition (the accumulator's
    # frame), then through the same t2r slice the values come out of.
    idAcc = cute.flat_divide(
        mma_ab.get_slice(0).partition_C(
            cute.make_identity_tensor((CHUNK, CHUNK)))[((None, None), 0, 0)],
        (CHUNK, CHUNK))[(None, None, 0, 0)]
    cAcc = thr_acc_t2r.partition_D(idAcc)
    tAccA_src = thr_acc_t2r.partition_S(planeA)
    tAccB_src = thr_acc_t2r.partition_S(planeB)
    # D0.10 lever (2): the shared-window base addresses of the two gating faces,
    # hoisted out of the chunk loop -- the vector loads below address them by
    # hand (`.v2` has no DSL spelling), slot offset included per chunk.
    gate_pair_base = cutlass.Int32(sGatePair.iterator.toint())
    gate_col_base = cutlass.Int32(sGateCol.iterator.toint())
    # D0.11 (G5): the same for WG B's field-major face array. Its rows are
    # addressed by hand below for the same reason -- `.v2` / `.v4` have no DSL
    # spelling -- with the slot and the field folded into the base per segment.
    gate_b_base = cutlass.Int32(sGateB.iterator.toint())

    # r2s: the same thread-value map, 16-bit destination, so the helper picks
    # the `stmatrix` family. The destination faces are relabelled flat -- a row
    # of the face is exactly one 128 B SW128 atom, so (N, K):(K, 1) is a pure
    # relabel and the swizzle stays on the pointer.
    tiled_r2s = cute.make_tiled_copy_D(
        sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, cutlass.BFloat16, cutlass.Float32,
            tiled_acc_t2r),
        tiled_acc_t2r)
    thr_r2s = tiled_r2s.get_slice(tidx)
    # D1.2: the staging groups are the outermost (last) mode of the staged smem
    # layout, so group g starts CHUNK * CHUNK elements = 8 KiB in -- a whole
    # number of SW128 swizzle periods (1 KiB), which is what makes this flat
    # relabel of a swizzled pointer legal per group as well as per element.
    wS0 = cute.make_tensor(
        sS0.iterator,
        cute.make_layout((CHUNK, CHUNK, STAGE_SLOTS),
                         stride=(CHUNK, 1, CHUNK * CHUNK)))
    wS1 = cute.make_tensor(
        sS1.iterator,
        cute.make_layout((CHUNK, CHUNK, STAGE_SLOTS),
                         stride=(CHUNK, 1, CHUNK * CHUNK)))

    # The A / B fragments of groups (a) and (b). The K face carries the ring
    # stage mode; Q is single slot.
    frKa = mma_ab.make_fragment_A(sK)
    frKb = mma_ab.make_fragment_B(sK)
    frQa = mma_ab.make_fragment_A(sQ)
    n_kblk = cute.size(frKa, mode=[2])

    # -----------------------------------------------------------------------
    # D0.9 the M = 128 groups. A lives in TMEM (the op's own constraint makes it
    # K-major, which is exactly the D0.4 column plan: lane = feature, columns
    # walk K); B is an smem face read through a descriptor. Each A fragment is
    # the MMA's own layout from a *shape*, placed on a bf16 view of the TMEM
    # pointer -- the pointer arithmetic is in 32-bit columns and happens before
    # the recast (hw_kb `tmem_r2t_recast_fragment`, last gotcha).
    # -----------------------------------------------------------------------
    mma_m128 = _mma_m128()
    mma_su = _mma_su()
    # Same two-variant treatment as groups (a)/(b) above.
    mma_m128_n = _mma_m128()
    mma_m128_n.set(tcgen05.Field.ACCUMULATE, False)
    mma_m128_y = _mma_m128()
    mma_m128_y.set(tcgen05.Field.ACCUMULATE, True)
    mma_su_y = _mma_su()
    mma_su_y.set(tcgen05.Field.ACCUMULATE, True)
    accKS_layout = mma_m128.make_fragment_C(
        mma_m128.partition_shape_C((DIM, CHUNK))).layout
    tAccKS = cute.make_tensor(tmem_ptr + ACC_KS_COL, accKS_layout)
    tAccO = cute.make_tensor(tmem_ptr + ACC_O_COL, accKS_layout)
    tS32acc = cute.make_tensor(
        tmem_ptr + S32_COL,
        mma_su.make_fragment_C(mma_su.partition_shape_C((DIM, DIM))).layout)

    frS16 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + S16_COL, dtype=cutlass.BFloat16),
        mma_m128.make_fragment_A(mma_m128.partition_shape_A((DIM, DIM))).layout)
    frX16 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + X16_COL, dtype=cutlass.BFloat16),
        mma_m128.make_fragment_A(
            mma_m128.partition_shape_A((DIM, CHUNK))).layout)
    frU16s = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + U16S_COL, dtype=cutlass.BFloat16),
        mma_su.make_fragment_A(mma_su.partition_shape_A((DIM, CHUNK))).layout)

    # The B faces. The K tile bytes are read through three descriptors: K-major
    # for groups (a)/(b) (64 x 64 x 128) and (d) (128 x 64 x 128), and MN-major
    # for group (i) (128 x 128 x 64). The swizzle rides on the pointer, so a
    # second view of the same buffer is just a second layout.
    frKd = mma_m128.make_fragment_B(
        cute.make_tensor(sK.iterator, kb_layout.outer))
    frQe = mma_m128.make_fragment_B(
        cute.make_tensor(sQ.iterator, qb_layout.outer))
    frKi = mma_su.make_fragment_B(
        cute.make_tensor(sK.iterator, kmn_layout.outer))
    frT = mma_m128.make_fragment_B(sS1)
    frP = mma_m128.make_fragment_B(sS0)

    # -----------------------------------------------------------------------
    # D0.11 WG B partitions. Segments B2 and B5 read a 128 x 64 accumulator
    # with the 16-datapath atom, because B2 has to meet the V face element by
    # element and only that fragment shape has an `ldmatrix` partner (hw_kb
    # `smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`, gotcha 1).
    # -----------------------------------------------------------------------
    planeKS = tAccKS[((None, None), 0, 0)]
    planeO = tAccO[((None, None), 0, 0)]
    idPlane = cute.make_identity_tensor((DIM, CHUNK))
    tiled_ks = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32),
        planeKS)
    thr_ks = tiled_ks.get_slice(lane_in_wg)
    cKS = thr_ks.partition_D(idPlane)
    # Station A of the exchange-face recipe: the V tile is (64 token,
    # 128 feature) with the feature contiguous inside each of its two SW128
    # boxes, so the (feature, token) view has mode 0 contiguous and the
    # transposed `ldmatrix` member is the one that lands the same (v, t) the
    # TMEM fragment holds.
    sVt = cute.make_tensor(
        sV.iterator,
        cute.make_layout(((CHUNK, DIM // CHUNK), CHUNK),
                         stride=((1, CHUNK * CHUNK), CHUNK)))
    tiled_vt = cute.make_tiled_copy_D(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4),
                            cutlass.BFloat16),
        tiled_ks)
    thr_vt = tiled_vt.get_slice(lane_in_wg)

    tiled_acc32 = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(TMEM_REP)),
            cutlass.Float32),
        planeO)
    tiled_acc32w = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(TMEM_REP),
                               tcgen05.Unpack.NONE),
            cutlass.Float32),
        planeO)
    thr_acc32 = tiled_acc32.get_slice(lane_in_wg)
    thr_acc32w = tiled_acc32w.get_slice(lane_in_wg)
    cCol = thr_acc32.partition_D(idPlane)

    # -----------------------------------------------------------------------
    # D0.11 B5 + D0.13: the output epilogue, the four-leg path of hw_kb
    # `tcgen05_epilogue_t2r_stsm_tma_store`. The MMA issues O transposed
    # (M = 128 = v, which is the *contiguous* dimension of `output`), so both
    # op selectors and the face builder are keyed on `COL_MAJOR` -- the MMA's
    # frame, not the torch tensor's (that KB entry, gotcha 13). None of the
    # three atoms is hand-picked: the epi tile comes from the helper, the
    # t2r atom from the tile, the r2s atom from the t2r tiled copy (which for
    # M-major bf16 out of an fp32 accumulator is `STSM.16.MT88.4`, the
    # [v, t] -> [t, v] transpose D0.11 segment B5 names).
    # -----------------------------------------------------------------------
    epiO = cute.flat_divide(planeO, epi_tile)      # (EPI_M, EPI_N, 1, n_sub)
    tiled_o_t2r = tcgen05.make_tmem_copy(
        sm100_utils.get_tmem_load_op(
            (DIM, CHUNK, CHUNK), LayoutEnum.COL_MAJOR, cutlass.BFloat16,
            cutlass.Float32, epi_tile, False),
        epiO[(None, None, 0, 0)])
    thr_o_t2r = tiled_o_t2r.get_slice(lane_in_wg)
    tTR_tO = thr_o_t2r.partition_S(epiO)
    tTR_cO = thr_o_t2r.partition_D(cute.flat_divide(idPlane, epi_tile))
    tiled_o_r2s = cute.make_tiled_copy_D(
        sm100_utils.get_smem_store_op(
            LayoutEnum.COL_MAJOR, cutlass.BFloat16, cutlass.Float32,
            tiled_o_t2r),
        tiled_o_t2r)
    thr_o_r2s = tiled_o_r2s.get_slice(lane_in_wg)
    tRS_sO = thr_o_r2s.partition_D(sOut)

    # D0.1: static grid-stride dispatch over items i in [0, N*HV).
    # grid = min(MAX_CTAS, N*HV) on the host, so every CTA owns at least one item
    # and this trip count is >= 1 whenever there is any work at all.
    n_round = (n_item - bidx + gdim - 1) // gdim

    # D0.14: the ring phase is a CTA-lifetime count, not a per-item one — the
    # barriers are never reset between items, so every phase below is derived
    # from this running chunk counter. Each role now keeps its own copy of it
    # (`cc` below): the three copies are advanced by the same `n_chunk` in the
    # same item order, so they hold the same value in every role at every
    # rendezvous — the counter is a pure function of the item sequence, not a
    # shared variable.

    # -----------------------------------------------------------------------
    # D0.7 register split (hw_kb `warp_specialization`). Four things have to be
    # true together or the split is silently a no-op: (1) the dec/inc pair, (2)
    # `min_blocks_per_mp=1` on the launch -- without it ptxas cannot pin the
    # entry register count and drops every `setmaxnreg` (SASS `USETMAXREG == 0`,
    # that entry's pit 1), (3) the dec/inc as the *first* statement of the one
    # role branch, with the whole shared prelude above it, and (4) each role's
    # own item loop living inside its own branch, so no shared code is ever
    # reached on a path that has already passed a `.dec` (pit 11, forms A and
    # B). The heavy register state of each role -- the accumulator and face
    # fragments -- is built inside its branch as well (pit 6): a fragment made
    # above the branch stays live across the specialists' `.dec` point and is
    # forced into the 24-register region.
    #
    #   128 * 24 + 128 * 224 + 128 * 256 = 64512 <= 65536
    #
    # `setmaxnreg` is `.sync.aligned`, so the predicate is the warpgroup-uniform
    # `warp_idx // 4` and never an `elect_one` (pit 3), and no barrier sits
    # between a `.dec` and the `.inc` that waits on it (pit 5).
    # -----------------------------------------------------------------------
    if wg == 2:
        cute.arch.setmaxregister_decrease(REG_SPECIALIST)
        cc = cutlass.Int32(0)

        # warp 11's tail-chunk read side of the output face: 8 bf16 (16 B) is
        # the SW128 swizzle's own unit, so a vector that wide is contiguous
        # after the permutation and 16 of them cover one token row of 128 v.
        # Built once here rather than per item: an `alloca` that sits inside a
        # staged loop body is not guaranteed to be promoted out of local memory
        # (hw_kb `warp_specialization` pit 7), and local traffic is exactly what
        # this decision is measured on.
        fOut = cute.flat_divide(sOut, (8, 1))      # (8, 1, 16, EPI_N, n_sub)
        rTail = cute.make_rmem_tensor((8,), cutlass.BFloat16)
        # warp 10's two-token gating quartet (a_t, b_t) x (t, t + 1), for the
        # same reason as `rTail`: an `alloca` written inside the chunk loop is
        # not guaranteed to be promoted, and a local-memory array inside the
        # 24-register region is exactly what this decision is measured on.
        rg = cute.make_rmem_tensor((4,), cutlass.Float32)

        # D1.4 (2): `cu_seqlens` read one item ahead. The pair belonging to the
        # item a role is about to start is the only data-dependent address load
        # in the whole kernel, and in the M0 form it sat at the top of the item
        # with everything the item does waiting behind its `LDG` latency. Here
        # each iteration loads the *next* item's pair into a loop-carried
        # register, so the latency is covered by a whole item of work, and only
        # the first CTA item still pays it. No branch and no smem: the index is
        # clamped to the last sequence so the tail iteration's (unused) load
        # stays inside `cu_seqlens`.
        n_seq = n_item // HV
        nx_lo = cutlass.Int32(mCuSeq[bidx // HV])
        nx_hi = cutlass.Int32(mCuSeq[bidx // HV + 1])

        for r in cutlass.range(n_round):
            item = bidx + r * gdim
            s = item // HV
            h = item % HV
            hq = h // 2                     # GVA: v-head h reads q/k head h // 2

            _seg("item.prologue.begin", r)
            t_lo = nx_lo
            t_hi = nx_hi
            s_nx = cutlass.min((item + gdim) // HV, n_seq - 1)
            nx_lo = cutlass.Int32(mCuSeq[s_nx])
            nx_hi = cutlass.Int32(mCuSeq[s_nx + 1])
            _seg("item.prologue.end", r)

            n_tok = t_hi - t_lo
            n_chunk = (n_tok + CHUNK - 1) // CHUNK

            # -------------------------------------------------------------------
            # warp 9 — the TMA producer (D0.8). Order per chunk is K -> Q -> V,
            # each group `empty-wait -> elect -> arrive.expect_tx -> 2 x UTMALDG`
            # (the two SW128 boxes of the tile are the one copy the atom issues).
            # The K empty-wait is against the slot's release two chunks back, which
            # is what hides the K load under the previous chunk; Q and V wait on
            # the previous chunk. Rows past the end of the tensor are zero-filled
            # by the TMA, which is the tail-chunk contract D0.6 rests on.
            # -------------------------------------------------------------------
            if warp_idx == 9:
                # D0.8 [GATHER: none]: the gmem side of the supply is affine. The TMA
                # coordinate origin is shifted onto this item once (token `t_lo`, head
                # `hq` / `h`), so inside the chunk loop the only varying coordinate is
                # the chunk index — no address-dependent load in the loop body.
                gK = cute.group_modes(
                    cute.flat_divide(cute.domain_offset((t_lo, 0, hq), gmemK),
                                     (CHUNK, DIM)), 0, 2)
                gQ = cute.group_modes(
                    cute.flat_divide(cute.domain_offset((t_lo, 0, hq), gmemQ),
                                     (CHUNK, DIM)), 0, 2)
                gV = cute.group_modes(
                    cute.flat_divide(cute.domain_offset((t_lo, 0, h), gmemV),
                                     (CHUNK, DIM)), 0, 2)
                tKsK, tKgK = cpasync.tma_partition(tma_k, 0, cute.make_layout(1),
                                                   cute.group_modes(sK, 0, 3), gK)
                tQsQ, tQgQ = cpasync.tma_partition(tma_q, 0, cute.make_layout(1),
                                                   cute.group_modes(sQ, 0, 3), gQ)
                tVsV, tVgV = cpasync.tma_partition(tma_v, 0, cute.make_layout(1),
                                                   cute.group_modes(sV, 0, 3), gV)

                for c in cutlass.range(n_chunk):
                    cg = cc + c
                    ks = cg % K_SLOTS

                    _seg("supply.kring.empty.begin", cg)
                    if cg >= K_SLOTS:
                        cute.arch.mbarrier_wait(mbar + BAR_K_EMPTY + ks,
                                                (cg // K_SLOTS - 1) % 2)
                    _seg("supply.kring.empty.end", cg)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + BAR_K_FULL + ks, tx_bytes)
                    cute.copy(tma_k, tKgK[(None, c, 0, 0)], tKsK[(None, ks)],
                              tma_bar_ptr=mbar + BAR_K_FULL + ks)

                    qs = cg % Q_SLOTS
                    _seg("supply.q.empty.begin", cg)
                    if cg >= Q_SLOTS:
                        cute.arch.mbarrier_wait(mbar + BAR_Q_EMPTY + qs,
                                                (cg // Q_SLOTS - 1) % 2)
                    _seg("supply.q.empty.end", cg)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + BAR_Q_FULL + qs, tx_bytes)
                    cute.copy(tma_q, tQgQ[(None, c, 0, 0)], tQsQ[(None, qs)],
                              tma_bar_ptr=mbar + BAR_Q_FULL + qs)

                    _seg("supply.v.empty.begin", cg)
                    if cg >= V_SLOTS:
                        cute.arch.mbarrier_wait(mbar + BAR_V_EMPTY, (cg - 1) % 2)
                    _seg("supply.v.empty.end", cg)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar + BAR_V_FULL, tx_bytes)
                    cute.copy(tma_v, tVgV[(None, c, 0, 0)], tVsV[(None, 0)],
                              tma_bar_ptr=mbar + BAR_V_FULL)

            # -------------------------------------------------------------------
            # warp 10 — the gating scalars (D0.5). The whole chain is fp32 and in
            # log space: lane l owns tokens 2l and 2l+1, a five-round warp scan of
            # the per-lane pair sum gives the inclusive prefix G_t, and lane 31's
            # total is broadcast as G_C. Padding tokens (t >= cu_seqlens[s+1]) get
            # l_t = 0 and beta_t = 0, which is exactly what makes them drop out of
            # L, X, U and the state update (§1.3). The four arrays plus gamma_C are
            # the cross-role contract: WG A reads G and beta, WG B reads beta,
            # gamma, lambda and gamma_C.
            # -------------------------------------------------------------------
            if warp_idx == 10:
                lane = cute.arch.lane_idx()
                neg_expA = -_exp(mALog[h])
                dt_bias = mDtBias[h]
                for c in cutlass.range(n_chunk):
                    cg = cc + c
                    gb = cg % GATE_SLOTS
                    _seg("gate.empty.begin", cg)
                    if cg >= GATE_SLOTS:
                        cute.arch.mbarrier_wait(mbar + BAR_GATE_EMPTY + gb,
                                                (cg // GATE_SLOTS - 1) % 2)
                    _seg("gate.empty.end", cg)

                    base = t_lo + c * CHUNK
                    t0 = base + lane * 2
                    # The tail chunk runs past the sequence and may run past the
                    # end of `a` / `b` entirely, so the loads live under the
                    # predicate rather than being masked after the fact.
                    for e in cutlass.range_constexpr(4):
                        rg[e] = cutlass.Float32(0.0)
                    if t0 < t_hi:
                        rg[0] = neg_expA * _softplus(
                            cutlass.Float32(mA[(t0, h)]) + dt_bias)
                        rg[1] = _sigmoid(cutlass.Float32(mB[(t0, h)]))
                    if t0 + 1 < t_hi:
                        rg[2] = neg_expA * _softplus(
                            cutlass.Float32(mA[(t0 + 1, h)]) + dt_bias)
                        rg[3] = _sigmoid(cutlass.Float32(mB[(t0 + 1, h)]))

                    v = rg[0] + rg[2]
                    for d in cutlass.range_constexpr(5):
                        off = 1 << d
                        # `mask_and_clamp = 0` is load-bearing for `.up`: the
                        # wrapper's default is WARP_SIZE - 1, whose segment field
                        # makes every source lane index fail the bound, so the
                        # shuffle hands each lane back its *own* value and the scan
                        # degenerates to v * 2^(rounds this lane passes the mask).
                        # The DSL's own warp prefix sum passes 0 for the same
                        # reason (`utils/grouped_gemm_persistent_tile_scheduler.py`
                        # `_prefix_sum`).
                        v = v + cute.arch.shuffle_sync_up(
                            v, off, mask_and_clamp=0) * _ge_mask(lane, off)
                    g_c = cute.arch.shuffle_sync(v, 32 - 1)
                    g1 = v
                    g0 = v - rg[2]

                    # D0.10 lever (2): G_t goes to the two faces WG A reads
                    # vectorially instead of to field 0 of the gating array --
                    # the (G, beta) pair for a row read, G alone for a column
                    # read. Both are published by the same `GATE_FULL` arrival
                    # below and released by the same `GATE_EMPTY`, so the gating
                    # protocol is untouched. beta stays in field 1 as well: WG B
                    # reads it one token at a time, where a pair read buys
                    # nothing.
                    #
                    # The two WG A faces carry `G * LOG2E`, not `G`: WG A's only
                    # use of them is `exp(G_i - G_j) = exp2(LOG2E*G_i -
                    # LOG2E*G_j)`, and scaling here turns that per-element chain
                    # from FADD -> FMUL -> MUFU.EX2 into FADD -> MUFU.EX2. The
                    # scaling costs this warp two FMULs per chunk (one per token
                    # it owns, each stored twice) and saves WG A 32 per thread.
                    # Both faces are private to WG A -- the field-major array
                    # below keeps the unscaled fields WG B reads -- so nothing
                    # else sees the change of units. Numerically it moves one
                    # rounding from the difference to the two operands, i.e. an exponent error
                    # of ~6e-8*|G| instead of ~6e-8*|G_i - G_j|; both are orders
                    # below the bf16 rounding that L and P take immediately
                    # after.
                    g0b = g0 * LOG2E
                    g1b = g1 * LOG2E
                    sGatePair[(gb, lane * 2, 0)] = g0b
                    sGatePair[(gb, lane * 2, 1)] = rg[1]
                    sGatePair[(gb, lane * 2 + 1, 0)] = g1b
                    sGatePair[(gb, lane * 2 + 1, 1)] = rg[3]
                    sGateCol[(gb, lane * 2)] = g0b
                    sGateCol[(gb, lane * 2 + 1)] = g1b
                    # D0.11 (G5): WG B's fields, field major, and two of them
                    # pre-shaped for their consumer -- `-gamma_t` so that B2's
                    # `V - gamma_t * acc` is a single `fma`, and `scale *
                    # gamma_t` so that B3 has no `scale` multiply of its own.
                    # Four FMUL/FNEG per lane per chunk here against 64 FMUL
                    # per lane per chunk in each of WG B's 128 lanes; the
                    # product `scale * gamma_t` is formed in fp32 exactly where
                    # B3 used to form it, so the value is unchanged.
                    gam0 = _exp(g0)
                    gam1 = _exp(g1)
                    lam0 = _exp(g_c - g0)
                    lam1 = _exp(g_c - g1)
                    sGateB[(gb, GATE_BETA, lane * 2)] = rg[1]
                    sGateB[(gb, GATE_BETA, lane * 2 + 1)] = rg[3]
                    sGateB[(gb, GATE_NGAM, lane * 2)] = -gam0
                    sGateB[(gb, GATE_NGAM, lane * 2 + 1)] = -gam1
                    sGateB[(gb, GATE_SGAM, lane * 2)] = scale * gam0
                    sGateB[(gb, GATE_SGAM, lane * 2 + 1)] = scale * gam1
                    sGateB[(gb, GATE_LAM, lane * 2)] = lam0
                    sGateB[(gb, GATE_LAM, lane * 2 + 1)] = lam1
                    if lane == 0:
                        sGateB[(gb, GATE_GC, 0)] = _exp(g_c)
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar + BAR_GATE_FULL + gb)
                    if cutlass.const_expr(dbg_on):
                        if item == 0:
                            if c == 0:
                                # field 0 (G_t) now lives in the pair face, in
                                # units of log2 -- back to natural log for the
                                # dump so the debug view keeps its meaning.
                                mDbg[(DBG_GATE, 0, lane * 2)] = \
                                    sGatePair[(gb, lane * 2, 0)] * LN2
                                mDbg[(DBG_GATE, 0, lane * 2 + 1)] = \
                                    sGatePair[(gb, lane * 2 + 1, 0)] * LN2
                                # Fields 1..4 keep the *stage* meaning they had
                                # before D0.11 (G5) reshaped the array -- beta,
                                # gamma, lambda, gamma_C -- and are dumped from
                                # the registers rather than re-read from smem,
                                # where two of them now live negated / scaled.
                                mDbg[(DBG_GATE, 1, lane * 2)] = rg[1]
                                mDbg[(DBG_GATE, 1, lane * 2 + 1)] = rg[3]
                                mDbg[(DBG_GATE, 2, lane * 2)] = gam0
                                mDbg[(DBG_GATE, 2, lane * 2 + 1)] = gam1
                                mDbg[(DBG_GATE, 3, lane * 2)] = lam0
                                mDbg[(DBG_GATE, 3, lane * 2 + 1)] = lam1
                                if lane == 0:
                                    mDbg[(DBG_GATE, 4, 0)] = _exp(g_c)

            # -------------------------------------------------------------------
            # warp 11 — the output store (D0.13). Per chunk: wait for the face WG
            # B's B5 wrote, then either two `UTMASTG` boxes (the whole 64-token
            # chunk, 8 KiB each) or — on the tail chunk, where a box would write
            # past `cu_seqlens[s+1]` into the next sequence's rows, which belong to
            # a different CTA — a row-predicated vector store out of the same face.
            # The store group is drained before the face is released (M0 starting
            # point; the `wait_group.read 1` of D1.3 is the lever).
            # -------------------------------------------------------------------
            if warp_idx == 11:
                # D0.13: the store side of the same [GATHER: none] discipline. The
                # output frame is (v, token, head) -- the face's own mode order, which
                # is the only view of those bytes the swizzled face allows -- shifted
                # onto this item once, so the chunk loop varies only the sub-tile index.
                gO = cute.group_modes(
                    cute.flat_divide(cute.domain_offset((0, t_lo, h), gmemO),
                                     epi_tile), 0, 2)
                bSG_sO, bSG_gO = cpasync.tma_partition(tma_o, 0, cute.make_layout(1),
                                                       cute.group_modes(sOut, 0, 2),
                                                       gO)
                # The tail chunk's plain-store view of the same rows: (1 token, 8 v)
                # tiles of this item's `output[:, h, :]` plane.
                gOt = cute.flat_divide(mOut[(None, h, None)], (1, 8))

                lane11 = cute.arch.lane_idx()
                tail_vb = lane11 % 16          # which 8-wide v block of the row
                tail_rr = lane11 // 16         # which of the two rows of a pass
                for c in cutlass.range(n_chunk):
                    cg = cc + c
                    base = t_lo + c * CHUNK
                    # D1.3: the face this chunk was written into, and the face of
                    # chunk cg - 1, which is the one this iteration releases.
                    os_ = cg % OUT_SLOTS
                    op_ = (cg + OUT_SLOTS - 1) % OUT_SLOTS
                    sub0 = os_ * n_sub

                    _seg("store.out.full.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_OUT_FULL + os_,
                                            (cg // OUT_SLOTS) % 2)
                    _seg("store.out.full.end", cg)

                    if base + CHUNK <= t_hi:
                        for st in cutlass.range_constexpr(n_sub):
                            cute.copy(tma_o, bSG_sO[(None, sub0 + st)],
                                      bSG_gO[(None, 0, c * n_sub + st, 0)])
                        cute.arch.cp_async_bulk_commit_group()
                        # D1.3: `wait_group 1` leaves this chunk's store in
                        # flight and only guarantees that every *earlier* group
                        # has finished reading its face -- which is precisely the
                        # face released below.
                        _seg("store.drain.begin", cg)
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                        _seg("store.drain.end", cg)
                        # Async groups are per thread and the TMA op issues from
                        # one elected lane, so only that lane's wait is the real
                        # one; this is what makes the release below warp-wide.
                        cute.arch.sync_warp()
                    else:
                        # 32 lanes = 2 token rows x 16 v-blocks of 8; 32 passes
                        # cover the 64 rows of the face, each under its own row
                        # predicate. 8 bf16 = 16 B = the SW128 swizzle unit.
                        # This path commits no group of its own, so the lagging
                        # release below needs the outstanding groups drained in
                        # full rather than down to one.
                        _seg("store.drain.begin", cg)
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        _seg("store.drain.end", cg)
                        for p in cutlass.range_constexpr(CHUNK // 2):
                            tl = p * 2 + tail_rr
                            if base + tl < t_hi:
                                cute.autovec_copy(
                                    fOut[(None, 0, tail_vb, tl % epi_n,
                                          sub0 + tl // epi_n)],
                                    rTail)
                                cute.autovec_copy(
                                    rTail,
                                    gOt[(0, None, base + tl, tail_vb)])
                        cute.arch.sync_warp()

                    # D1.3: the release lags one chunk, in both branches, so the
                    # arrive pattern per slot stays "once per generation" however
                    # the tail chunks fall. The face of the very last chunk the
                    # CTA stores is never released -- nothing waits on it.
                    if cg >= 1:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(mbar + BAR_OUT_EMPTY + op_)

            # -------------------------------------------------------------------
            # warp 8 — the UMMA issue warp (D0.9 / D0.12 table (a)). Seven groups,
            # 44 instructions, 16 `tcgen05.commit`s per chunk. A commit retires only
            # after the tensor core has consumed the descriptors of the instructions
            # before it, so it doubles as the "operand bytes are free" signal: the
            # K and Q ring releases and the "S16 consumed" release are commits, and
            # they cost the issue warp no wait. The warp never waits on a group it
            # just issued -- the only waits are the eight in the table.
            #
            # D1.1 lookahead d = 1. The state-independent groups (a)/(b) of chunk
            # c + 1 are issued at the end of chunk c's body, immediately after step
            # 8, without waiting on any commit of chunk c. That publishes ACC_A /
            # ACC_B of chunk c + 1 while chunk c's state chain (B4 -> B1 -> (d) ->
            # B2) is still in flight, so WG A's whole segment (mask, 64 x 64
            # inverse, P) for chunk c + 1 runs underneath it and `T ready` is
            # already set when (g) of chunk c + 1 issues. The window costs no extra
            # TMEM: WG A drains ACC_A / ACC_B of chunk c in its steps 2/3, so the
            # `empty` waits inside are the same single-slot handshake as before,
            # only satisfied earlier. What it does cost is the second staging group
            # (D1.2): WG A writes L of chunk c + 1 while groups (g)/(h) of chunk c
            # have not yet read T / P of chunk c.
            # -------------------------------------------------------------------
            if warp_idx == 8:
                # Every operand this helper touches is an explicit parameter: the
                # DSL rejects a nested function that captures a traced value from
                # the enclosing scope when it is called from inside a staged
                # `if` / `for` (SCOPE_CLOSURE_CAPTURE), and both call sites below
                # are staged. Only module-level names (`_seg`, the BAR_* / *_SLOTS
                # constants, `cute` / `cutlass` / `tcgen05`) are read as globals.
                def _issue_ab(cg, mbar, mma_ab_n, mma_ab_y, n_kblk,
                              tAccA, tAccB, frKa, frKb, frQa):
                    """Steps 1-2 of the table for the chunk with global index cg."""
                    ks = cg % K_SLOTS
                    qs = cg % Q_SLOTS

                    # step 1: (a) K.K^T -> ACC_A
                    _seg("mma.kring.full.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_K_FULL + ks,
                                            (cg // K_SLOTS) % 2)
                    _seg("mma.kring.full.end", cg)
                    _seg("mma.acca.empty.begin", cg)
                    if cg >= 1:
                        cute.arch.mbarrier_wait(mbar + BAR_ACC_A_EMPTY,
                                                (cg - 1) % 2)
                    _seg("mma.acca.empty.end", cg)
                    for kb in cutlass.range_constexpr(n_kblk):
                        cute.gemm(mma_ab_n if kb == 0 else mma_ab_y,
                                  tAccA, frKa[(None, None, kb, ks)],
                                  frKb[(None, None, kb, ks)], tAccA)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_A_FULL)

                    # step 2: (b) Q.K^T -> ACC_B
                    _seg("mma.q.full.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_Q_FULL + qs,
                                            (cg // Q_SLOTS) % 2)
                    _seg("mma.q.full.end", cg)
                    _seg("mma.accb.empty.begin", cg)
                    if cg >= 1:
                        cute.arch.mbarrier_wait(mbar + BAR_ACC_B_EMPTY,
                                                (cg - 1) % 2)
                    _seg("mma.accb.empty.end", cg)
                    for kb in cutlass.range_constexpr(n_kblk):
                        cute.gemm(mma_ab_n if kb == 0 else mma_ab_y,
                                  tAccB, frQa[(None, None, kb, qs)],
                                  frKb[(None, None, kb, ks)], tAccB)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_B_FULL)

                # D1.1 pipeline prologue: chunk 0 of this item gets its (a)/(b)
                # before the loop, so that every body iteration ends one chunk
                # ahead. The lookahead does not cross the item boundary -- warp 8
                # meets WG A and WG B at the item rendezvous.
                if n_chunk >= 1:
                    _issue_ab(cc, mbar, mma_ab_n, mma_ab_y, n_kblk,
                              tAccA, tAccB, frKa, frKb, frQa)

                for c in cutlass.range(n_chunk):
                    cg = cc + c
                    ks = cg % K_SLOTS
                    qs = cg % Q_SLOTS
                    ss = cg % STAGE_SLOTS

                    # steps 3/4: (d) (K S_0)^T -> ACC_KS, issued in `S16_SEGS`
                    # pieces (D2.1) so that WG B's B1 can hand over the first
                    # 128 / S16_SEGS columns of S16 before it has packed the
                    # rest. Piece p covers k-blocks [p * KB_PER_SEG,
                    # (p + 1) * KB_PER_SEG) -- 16 k values per k-block, 8
                    # S16 columns -- which is exactly the state the p-th
                    # segment barrier publishes. The wait sites are written out
                    # one per piece because the timeline probe only rewrites
                    # `_seg` calls whose site is a literal string.
                    _seg("mma.s16q0.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_S16_SEG + 0, cg % 2)
                    _seg("mma.s16q0.end", cg)
                    for kb in cutlass.range_constexpr(KB_PER_SEG):
                        cute.gemm(mma_m128_n if kb == 0 else mma_m128_y,
                                  tAccKS, frS16[(None, None, kb)],
                                  frKd[(None, None, kb, ks)], tAccKS)
                    _seg("mma.s16q1.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_S16_SEG + 1, cg % 2)
                    _seg("mma.s16q1.end", cg)
                    for kb in cutlass.range_constexpr(KB_PER_SEG, 2 * KB_PER_SEG):
                        cute.gemm(mma_m128_y, tAccKS, frS16[(None, None, kb)],
                                  frKd[(None, None, kb, ks)], tAccKS)
                    _seg("mma.s16q2.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_S16_SEG + 2, cg % 2)
                    _seg("mma.s16q2.end", cg)
                    for kb in cutlass.range_constexpr(2 * KB_PER_SEG,
                                                      3 * KB_PER_SEG):
                        cute.gemm(mma_m128_y, tAccKS, frS16[(None, None, kb)],
                                  frKd[(None, None, kb, ks)], tAccKS)
                    _seg("mma.s16q3.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_S16_SEG + 3, cg % 2)
                    _seg("mma.s16q3.end", cg)
                    for kb in cutlass.range_constexpr(3 * KB_PER_SEG,
                                                      4 * KB_PER_SEG):
                        cute.gemm(mma_m128_y, tAccKS, frS16[(None, None, kb)],
                                  frKd[(None, None, kb, ks)], tAccKS)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_KS_FULL)

                    # step 5: (e) (Q S_0)^T -> ACC_O; its commits also release the
                    # Q slot and declare S16 consumed (last reader of both).
                    _seg("mma.acco.empty.begin", cg)
                    if cg >= 1:
                        cute.arch.mbarrier_wait(mbar + BAR_ACC_O_EMPTY,
                                                (cg - 1) % 2)
                    _seg("mma.acco.empty.end", cg)
                    for kb in cutlass.range_constexpr(8):
                        cute.gemm(mma_m128_n if kb == 0 else mma_m128_y,
                                  tAccO, frS16[(None, None, kb)],
                                  frQe[(None, None, kb, qs)], tAccO)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_O_FULL_E)
                        tcgen05.commit(mbar + BAR_Q_EMPTY + qs)
                        tcgen05.commit(mbar + BAR_S16_EMPTY)

                    # step 6: (g) X.T^T -> ACC_U (the ACC_KS columns, now free)
                    _seg("mma.x.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_X_READY, cg % 2)
                    _seg("mma.x.end", cg)
                    _seg("mma.t.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_T_READY + ss,
                                            (cg // STAGE_SLOTS) % 2)
                    _seg("mma.t.end", cg)
                    for kb in cutlass.range_constexpr(4):
                        cute.gemm(mma_m128_n if kb == 0 else mma_m128_y,
                                  tAccKS, frX16[(None, None, kb)],
                                  frT[(None, None, kb, ss)], tAccKS)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_U_FULL)

                    # step 7: (h) U.P^T accumulated onto the scaled ACC_O
                    _seg("mma.p.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_P_READY + ss,
                                            (cg // STAGE_SLOTS) % 2)
                    _seg("mma.p.end", cg)
                    _seg("mma.u16.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_U16_READY, cg % 2)
                    _seg("mma.u16.end", cg)
                    _seg("mma.acco.scaled.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_ACC_O_SCALED, cg % 2)
                    _seg("mma.acco.scaled.end", cg)
                    for kb in cutlass.range_constexpr(4):
                        cute.gemm(mma_m128_y, tAccO, frX16[(None, None, kb)],
                                  frP[(None, None, kb, ss)], tAccO)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_ACC_O_FULL_H)

                    # step 8: (i) U~^T.K accumulated onto the decayed S32; its
                    # commit also releases the K ring slot (last reader).
                    _seg("mma.s32dec.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_S32_DEC, cg % 2)
                    _seg("mma.s32dec.end", cg)
                    _seg("mma.u16s.begin", cg)
                    cute.arch.mbarrier_wait(mbar + BAR_U16S_READY, cg % 2)
                    _seg("mma.u16s.end", cg)
                    for kb in cutlass.range_constexpr(4):
                        cute.gemm(mma_su_y, tS32acc, frU16s[(None, None, kb)],
                                  frKi[(None, None, kb, ks)], tS32acc)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar + BAR_S32_FULL)
                        tcgen05.commit(mbar + BAR_K_EMPTY + ks)

                    # D1.1 lookahead: steps 1-2 of chunk c + 1, issued right after
                    # step 8 without waiting on any commit of chunk c. Steps 3-8 of
                    # chunk c + 1 still sit behind chunk c's c14..c16 at the top of
                    # the next iteration, so the only thing that moves is the pair
                    # of state-independent groups -- and with them WG A's whole
                    # segment for chunk c + 1, which now runs underneath chunk c's
                    # B4 -> B1 -> (d) -> B2 instead of in front of it. The two
                    # `empty` waits inside are already satisfied at this point: WG A
                    # released ACC_A / ACC_B of chunk c in its steps 2/3, long
                    # before it published T / P.
                    if c + 1 < n_chunk:
                        _issue_ab(cg + 1, mbar, mma_ab_n, mma_ab_y, n_kblk,
                                  tAccA, tAccB, frKa, frKb, frQa)

            # D1.4 (1): the 288-thread item rendezvous that used to sit here is
            # gone. It was the one thing that made WG B's state load of item
            # r + 1 (64 KiB through the LSU) and its state write-back of item r
            # serialise against WG A's and the issue warp's state-independent
            # work for the new item's chunk 0. Nothing else rested on it: every
            # cross-role handoff is an mbarrier keyed on the CTA-lifetime chunk
            # counter, which is a pure function of the item sequence and
            # therefore identical in all three roles without a meeting, and the
            # runahead each role can build is already bounded by the protocol
            # itself -- WG A cannot begin chunk cg before `ACC_A full` of cg,
            # which the issue warp only emits after step 8 of cg - 1, which
            # waits on WG B's B1 and B4 of cg - 1. So no role can get two chunks
            # ahead of WG B, which is the margin every two-slot barrier here
            # (the gating arrays above all) needs.
            cc = cc + n_chunk

        # D0.13: `wait_group(0, read=True)` only says the face has been
        # read, not that the bytes have landed, so the CTA drains its
        # store groups for real before it exits (hw_kb
        # `tcgen05_epilogue_t2r_stsm_tma_store` gotcha 5).
        if warp_idx == 11:
            cute.arch.cp_async_bulk_wait_group(0, read=False)

        # D0.14 CTA teardown, specialist copy. The teardown used to sit after
        # the role branch, where all 384 threads rejoined: a join below a `.dec`
        # is compiled at the 24-register budget on the specialists' path and
        # forces anything live across it into local memory (hw_kb
        # `warp_specialization` pit 11, form B -- its footprint was `STL` in the
        # `.dec` region paired with `LDL` in the region after the last
        # `USETMAXREG`). `sync_threads` is a CTA-wide barrier, so one copy per
        # branch is the same rendezvous: all 384 threads arrive exactly once.
        cute.arch.sync_threads()
        _seg("cta.epilogue.exit")

    elif wg == 0:
        cute.arch.setmaxregister_increase(REG_WG_A)
        cc = cutlass.Int32(0)

        rAcc = cute.make_rmem_tensor(cAcc.shape, cutlass.Float32)
        rAccB = cute.make_rmem_tensor(cAcc.shape, cutlass.Float32)
        rFace = cute.make_rmem_tensor(cAcc.shape, cutlass.BFloat16)
        # Elements of the accumulator fragment one thread owns -- the trip
        # count of both mask passes and the length of the hoisted factor list.
        E_ACC = cute.size(rAcc)
        # D0.10 step 4: the warp-level MMA and its three matrix atoms.
        inv_atoms = _inv_atoms()
        inv_wrp = lane_in_wg // 32
        inv_lane = lane_in_wg % 32
        inv_frags = _inv_frags(inv_atoms, inv_lane)

        # D0.10 step 4, hoisted out of the chunk loop: the strictly upper
        # 16 x 16 blocks of the T face carry the zeros of a unit lower
        # triangular T, and `_tri_inverse` never writes anything but zeros into
        # them -- phase 0 writes the diagonal blocks, phase 1 T (1,0) / (3,2)
        # and phase 2 T (2,0) / (2,1) / (3,0) / (3,1), each of them whole.
        # They were being re-zeroed every chunk by four `STS.128` per lane
        # sitting in phase 0's opening block, the most latency-bound stretch WG
        # A has.  Zeroing every staging slot once here -- above every loop, and
        # ordered against chunk 0's first write of the face by the `NBAR_WGA`
        # rendezvous WG A takes before entering the routine -- leaves the
        # per-chunk pass with the one store that genuinely changes (the 8 x 8
        # of the diagonal block that is off the diagonal for this row).  This
        # zeroes the whole face rather than just the six upper blocks so that
        # it stays one straight-line pass with no `wrp`-dependent branch; the
        # rest of the face is rewritten in full every chunk anyway.
        inv_vec = inv_frags[7]
        for e in cutlass.range_constexpr(8):
            inv_vec[e] = cutlass.BFloat16(0.0)
        for sl in cutlass.range_constexpr(STAGE_SLOTS):
            vT_pro = cute.flat_divide(wS1[(None, None, sl)], (1, 8))
            for cb in cutlass.range_constexpr(2 * INV_N):
                cute.autovec_copy(
                    inv_vec,
                    vT_pro[(0, None,
                            inv_wrp * INV_B + inv_lane % INV_B, cb)])

        # -------------------------------------------------------------------
        # WG A — mask, triangular inverse, P (D0.10). Each accumulator is read
        # once with the 16-datapath `Ld16x256b` atom (the fragment shape that
        # has an `stmatrix` partner), turned into its bf16 exchange face element
        # by element under the identity coordinates of that same read, and
        # written out with `STSM.16.M88.4` -- no scalar `STS.U16` touches a
        # 64 x 64 face.
        #
        #   L[i, j] = beta_i * exp(G_i - G_j) * (k_i . k_j)   for j < i
        #   T       = (I + L)^{-1}                            unit lower
        #   P[t, i] = scale  * exp(G_t - G_i) * (q_t . k_i)   for i <= t
        #
        # Every exponent argument is <= 0 (G is a non-increasing prefix of
        # log g), so the factors are in (0, 1] and may only underflow to zero,
        # which is the mathematically correct limit. Both accumulators are
        # released back to the MMA warp as soon as their values are in
        # registers, before any of the element math.
        # -------------------------------------------------------------------
        # D1.4 (2): the same one-item-ahead `cu_seqlens` read as the specialist
        # branch; WG A keeps its own copy because the three roles no longer meet.
        n_seq = n_item // HV
        nx_lo = cutlass.Int32(mCuSeq[bidx // HV])
        nx_hi = cutlass.Int32(mCuSeq[bidx // HV + 1])

        for r in cutlass.range(n_round):
            item = bidx + r * gdim
            s = item // HV
            h = item % HV

            _seg("item.prologue.begin", r)
            t_lo = nx_lo
            t_hi = nx_hi
            s_nx = cutlass.min((item + gdim) // HV, n_seq - 1)
            nx_lo = cutlass.Int32(mCuSeq[s_nx])
            nx_hi = cutlass.Int32(mCuSeq[s_nx + 1])
            _seg("item.prologue.end", r)

            n_tok = t_hi - t_lo
            n_chunk = (n_tok + CHUNK - 1) // CHUNK

            for c in cutlass.range(n_chunk):
                cg = cc + c
                gb = cg % GATE_SLOTS
                # D1.2: the staging group this chunk's L / T / P live in. The
                # group is free without any extra signal: WG A cannot start
                # chunk cg before `ACC_A full` of chunk cg, and under D1.1 that
                # group (a) is issued after step 8 of chunk cg - 1, hence after
                # groups (g)/(h) of chunk cg - 2 -- the last readers of this
                # group's previous tenant.
                ss = cg % STAGE_SLOTS
                fS0 = wS0[(None, None, ss)]
                fS1 = wS1[(None, None, ss)]
                tS0_dst = thr_r2s.partition_D(fS0)

                _seg("wga.gate.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_GATE_FULL + gb,
                                        (cg // GATE_SLOTS) % 2)
                _seg("wga.gate.full.end", cg)

                # step 1, hoisted above the accumulator waits: the 32 factors
                #     f[e] = exp(G_row(e) - G_col(e))
                # depend only on the gating array, which the wait just above
                # published -- nothing in this chain touches an accumulator.
                # ptxas never moves work across an `mbarrier` wait loop, so
                # where the chain sits in the source *is* its schedule: issued
                # here, its FADD -> MUFU.EX2 -> FMUL latency runs inside the
                # `ACC_A full` wait below instead of in front of the consumer.
                # One f serves both masks: step 2 and step 5 read the same
                # fragment coordinates out of the same identity tensor, so
                # element e has the same (row, col) -- and therefore the same
                # exponent -- in both (hw_kb
                # `branchfree_masked_elementwise_pass` §(c) and gotcha 11).
                #
                # lever (2): the scalars come in per *group* of elements that
                # share them, not per element. `ACC_ROW_GROUPS` / `ACC_COL_RUNS`
                # are the measured fragment structure; one `LDS.64` per row
                # group returns that row's (G_i, beta_i) pair, one per column
                # run returns the two G_j of the run. 18 shared loads where the
                # per-element form issued 96, and the same 32 `MUFU.EX2`.
                # Only the two factor lists built below stay live across the
                # wait; the raw scalars die into them.
                pair_b = gate_pair_base + gb * (2 * CHUNK * 4)
                col_b = gate_col_base + gb * (CHUNK * 4)
                g_row = [None] * E_ACC
                b_row = [None] * E_ACC
                g_col = [None] * E_ACC
                for grp in ACC_ROW_GROUPS:
                    v = _lds_v2(pair_b + cAcc[grp[0]][0] * 8)
                    for e in grp:
                        g_row[e] = v[0]
                        b_row[e] = v[1]
                for run in ACC_COL_RUNS:
                    v = _lds_v2(col_b + cAcc[run[0]][1] * 4)
                    for k, e in enumerate(run):
                        g_col[e] = v[k]
                # "gate array consumed" (D0.10 step 1), released here rather
                # than at the bottom of the chunk: the 18 vector loads above are
                # WG A's only reads of the gate faces -- everything below works
                # from `g_row` / `b_row` / `g_col`, which are registers now --
                # so warp 10 can refill this slot a whole chunk earlier than it
                # used to.  Nothing sinks past the arrive to read the slot after
                # it: `_lds_v2` is `has_side_effects`, and `mbarrier.arrive`
                # carries release semantics, which orders prior shared loads
                # ahead of it in ptxas as well.  WG B holds the other half of
                # this barrier's 2 x `WARPGROUP` arrivals and is unchanged.
                cute.arch.mbarrier_arrive(mbar + BAR_GATE_EMPTY + gb)
                # lever (3), completed for both factors: neither mask pass keeps
                # a multiply of its own. The shared exponential is formed once
                # (gotcha 11: step 2 and step 5 read the same fragment
                # coordinates, so element e has the same exponent in both) and
                # both row scalars are folded into it here, before the `ACC_A
                # full` wait:
                #     lfac = beta_i * exp(G_i - G_j)     for L
                #     pfac = scale  * exp(G_i - G_j)     for P
                # ptxas will not move work across an `mbarrier` wait loop (hw_kb
                # `branchfree_masked_elementwise_pass` §(c)/gotcha 8), so where
                # these multiplies sit in the source *is* their schedule: issued
                # here, all 64 FMULs run inside the ACC_A latency and each mask
                # pass is left with one FMUL, one FSEL and one convert per
                # element. `scale * fac[e] * rAccB[e]` was left-associative, so
                # `pfac[e] * rAccB[e]` is bit-identical -- a pure schedule move.
                # The exponent argument arrives pre-scaled by LOG2E from warp 10,
                # so the chain is FADD -> MUFU.EX2 with no `x * LOG2E` FMUL.
                # `lfac` dies in step 2 and `pfac` in the P pass, which now sits
                # in front of the inverse rather than behind it -- so neither
                # factor list reaches the inverse at all, and what WG A carries
                # through it is the 32 bf16 of `rFace`.
                lfac = [None] * E_ACC
                pfac = [None] * E_ACC
                for e in cutlass.range_constexpr(E_ACC):
                    f_e = _exp2(g_row[e] - g_col[e])
                    lfac[e] = b_row[e] * f_e
                    pfac[e] = scale * f_e

                # step 2: ACC_A -> L -> staging s0
                _seg("wga.acca.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_A_FULL, cg % 2)
                _seg("wga.acca.full.end", cg)
                cute.copy(tiled_acc_t2r, tAccA_src, rAcc)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(mbar + BAR_ACC_A_EMPTY)
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rAcc)):
                                mDbg[(DBG_ACC_A, cAcc[e][0], cAcc[e][1])] = \
                                    rAcc[e]
                # L[i, j] = beta_i f_ij (k_i . k_j) for j < i, else 0 -- with the
                # beta multiply hoisted, one `FMUL` and one `FSEL` per element,
                # straight-line.
                for e in cutlass.range_constexpr(E_ACC):
                    rFace[e] = cutlass.BFloat16(_sel_lt_s32(
                        cAcc[e][1], cAcc[e][0],
                        lfac[e] * rAcc[e],
                        cutlass.Float32(0.0)))
                cute.copy(tiled_r2s, tiled_r2s.retile(rFace), tS0_dst)

                # step 3, issued in front of the WG A rendezvous rather than
                # after it: nothing between the `stmatrix` above and the meet
                # below depends on ACC_B, so the `LDTM` latency runs underneath
                # the store, the proxy fence and the wait for the other three
                # warps instead of in series after them -- and the MMA warp gets
                # `ACC_B empty` one rendezvous earlier, which is what gates
                # group (b) of the next chunk. The accumulator is only read
                # here; its element pass follows the meet.
                _seg("wga.accb.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_B_FULL, cg % 2)
                _seg("wga.accb.full.end", cg)
                cute.copy(tiled_acc_t2r, tAccB_src, rAccB)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(mbar + BAR_ACC_B_EMPTY)
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rAccB)):
                                mDbg[(DBG_ACC_B, cAcc[e][0], cAcc[e][1])] = \
                                    rAccB[e]
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.barrier(barrier_id=NBAR_WGA,
                                  number_of_threads=WARPGROUP)
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            if lane_in_wg < CHUNK:
                                for i in cutlass.range_constexpr(CHUNK):
                                    mDbg[(DBG_L, i, lane_in_wg)] = \
                                        cutlass.Float32(fS0[(i, lane_in_wg)])

                # step 5, arithmetic half, moved across the drain into the
                # inverse's opening block.
                # P[t, i] = scale f_ti (q_t . k_i) for i <= t, else 0. It reuses
                # the exponential of step 1 (same fragment coordinates as the L
                # mask, gotcha 11) with `scale` already folded into it by the
                # hoisted chain, and drops the branch: keep-if col <= row ==
                # not(row < col), so the selp takes `0` when row < col. One
                # FMUL, one FSEL and one convert per element.
                #
                # Nothing in it touches the staging faces -- the operands are
                # `rAccB`, which the `LDTM` above has already read out of TMEM,
                # and `pfac`, which the hoisted chain built before the `ACC_A
                # full` wait -- so the only reason it used to sit *after* the
                # inverse is that its `stmatrix` overwrites the L face the
                # inverse is still reading. The store, the publication fence and
                # `P ready` therefore stay where they were; only the 32 `FMUL`,
                # 32 `FSEL` and 16 packing converts move up here, into the same
                # basic block as phase 0 of `_tri_inverse` (the block runs from
                # the rendezvous above to phase 0's first `sync_warp`). Two
                # things follow:
                #   * That block is the most latency-bound stretch of WG A: a
                #     seven-level `shfl -> FFMA` forward substitution whose
                #     level m offers only m + 1 independent operations, so the
                #     warp spends most of it waiting on shuffle latency with
                #     nothing to issue. 96 mutually independent instructions are
                #     exactly what fills it, and ptxas interleaves them freely
                #     because it is one block (hw_kb: intra-block order is
                #     ptxas', only order *across* a wait is the source's).
                #   * The live set carried through the whole inverse drops from
                #     `pfac` + `rAccB` -- 64 fp32 registers -- to the 32 bf16 of
                #     `rFace`, i.e. 16. The inverse's own frames (`rA`, `rA2`,
                #     `rB`, `rC`, `rCb`, `mrow`, `trow`, `vec`) get 48 registers
                #     of room back, and the WG A tail after `T ready` shrinks to
                #     the `stmatrix` alone.
                # `rFace` is free to be rewritten here: its L contents were
                # consumed by the `stmatrix` of step 2, upstream of the ACC_B
                # wait. The element values are unchanged -- same operand pair,
                # same select, same rounding.
                for e in cutlass.range_constexpr(E_ACC):
                    rFace[e] = cutlass.BFloat16(_sel_lt_s32(
                        cAcc[e][0], cAcc[e][1],
                        cutlass.Float32(0.0),
                        pfac[e] * rAccB[e]))

                # step 4: T = (I + L)^{-1} -- the 16 x 16 blocked, two-level
                # `mma.sync` inverse of D0.10, four blocking WG A meets (the
                # last of which is also the publication fence for the T face;
                # the third is the one that lets the 64 level run on four warps
                # instead of two).
                _seg("wga.inv.begin", cg)
                _tri_inverse(fS0, fS1, inv_atoms, inv_frags, inv_wrp, inv_lane)
                _seg("wga.inv.end", cg)
                cute.arch.mbarrier_arrive(mbar + BAR_T_READY + ss)
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            if lane_in_wg < CHUNK:
                                for i in cutlass.range_constexpr(CHUNK):
                                    mDbg[(DBG_T, i, lane_in_wg)] = \
                                        cutlass.Float32(fS1[(i, lane_in_wg)])

                # step 5, store half: P -> staging s0. This is the part that
                # genuinely has to wait -- the inverse has only now released the
                # L face and its six upper scratch blocks, and this `stmatrix`
                # writes over them. The arithmetic that used to be here is in
                # the inverse's opening block, above; `rFace` has held P since
                # then.
                #
                # No WG A meet stands between the store and `P ready`: the
                # only consumer of the P face is the MMA warp, and
                # `BAR_P_READY` is initialised with `WARPGROUP` arrivals, so
                # the mbarrier *is* the rendezvous -- it flips only once all
                # 128 threads have arrived, and each of those arrivals carries
                # release semantics over that thread's own `stmatrix`, which
                # the `fence.proxy.async` above has already pushed into the
                # async proxy.  A named barrier in front of it adds a second,
                # redundant meet of the same 128 threads.  This is *not* true
                # of the last meet inside `_tri_inverse`: that one is an
                # anti-dependency (it keeps this store off the L face while
                # other warps are still reading it), not a publication.
                # WG A's blocking meets per chunk go from six to five -- one
                # under the D0.10 budget.
                cute.copy(tiled_r2s, tiled_r2s.retile(rFace), tS0_dst)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(mbar + BAR_P_READY + ss)
                if cutlass.const_expr(dbg_on):
                    # The stage dump reads the whole face from every lane, which
                    # the arrive above does not order; the probe pays for its
                    # own meet.
                    cute.arch.barrier(barrier_id=NBAR_WGA,
                                      number_of_threads=WARPGROUP)
                    if item == 0:
                        if c == 0:
                            if lane_in_wg < CHUNK:
                                for i in cutlass.range_constexpr(CHUNK):
                                    mDbg[(DBG_P, i, lane_in_wg)] = \
                                        cutlass.Float32(fS0[(i, lane_in_wg)])

            # D1.4 (1): no item rendezvous -- WG A walks straight into the next
            # item's chunk 0 while WG B is still writing back the state of this
            # one (see the note at the issue warp's copy).
            cc = cc + n_chunk

        # D0.14 CTA teardown, WG A's copy (see the specialist branch).
        cute.arch.sync_threads()
        _seg("cta.epilogue.exit")

    else:
        cute.arch.setmaxregister_increase(REG_WG_B)
        cc = cutlass.Int32(0)

        rKS = cute.make_rmem_tensor(cKS.shape, cutlass.Float32)
        rV = cute.make_rmem_tensor(cKS.shape, cutlass.BFloat16)
        rCol = cute.make_rmem_tensor(cCol.shape, cutlass.Float32)
        rO = cute.make_rmem_tensor(
            tTR_cO[(None, None, None, 0, 0)].shape, cutlass.Float32)
        rOb = cute.make_rmem_tensor(
            tTR_cO[(None, None, None, 0, 0)].shape, cutlass.BFloat16)
        tRS_rO = tiled_o_r2s.retile(rOb)
        rS16 = cute.make_rmem_tensor(
            thr_s16.partition_S(tS16f32).shape, cutlass.Float32)
        rS16b = cute.recast_tensor(rS16, cutlass.BFloat16)
        rX = cute.make_rmem_tensor(
            thr_x16.partition_S(tX16f32).shape, cutlass.Float32)
        rXb = cute.recast_tensor(rX, cutlass.BFloat16)
        rU = cute.make_rmem_tensor(
            thr_u16.partition_S(tX16f32).shape, cutlass.Float32)
        rUb = cute.recast_tensor(rU, cutlass.BFloat16)
        rUs = cute.make_rmem_tensor(
            thr_u16s.partition_S(tU16sf32).shape, cutlass.Float32)
        rUsb = cute.recast_tensor(rUs, cutlass.BFloat16)
        # The register side of the S32 pair. A TMEM copy names its register
        # face with the *opposite* partitioner of the side the TMEM tensor
        # sits on: `partition_D` for the load, `partition_S` for the store
        # (hw_kb `tmem_r2t_recast_fragment`). `tLS_tS`, the t2r *source*,
        # is the TMEM-side view and is 32x too large to size a fragment.
        # Both directions are `x32` on the same tile, so one fragment
        # serves the load and the store alike.
        rState = cute.make_rmem_tensor(
            thr_t2r.partition_D(tS32).shape, cutlass.Float32)
        # The two gmem-side ends of the same [V, K] plane -- the item prologue's
        # `state` load and the item epilogue's `new_state` store. Their shapes
        # come from the identity tensor of the plane rather than from `gS` /
        # `gNS`, whose offsets are per-item: the fragment has to be built here,
        # above the item loop, because an `alloca` inside a staged loop body is
        # not guaranteed to be promoted out of local memory (hw_kb
        # `warp_specialization` pit 7) and these two are 128 fp32 each.
        idState = cute.make_identity_tensor((DIM, DIM))
        rS = cute.make_rmem_tensor(
            thr_r2t.partition_S(idState).shape, cutlass.Float32)
        rSo = cute.make_rmem_tensor(
            thr_t2r.partition_D(idState).shape, cutlass.Float32)

        # D0.11 (G5): the element counts the two run tables are cut for. The
        # tables are a static property of atom + tile + slice, so a mismatch is
        # a trace-time error, not a wrong answer at run time -- and a wrong
        # table is silent (it reads the right *kind* of scalar for the wrong
        # column), so it is worth one assert each.
        E_KS = cute.size(rKS)
        E_COL = cute.size(rCol)
        assert E_KS == 2 * len(KS_COL_RUNS), "KS_COL_RUNS does not cover rKS"
        assert E_COL == 4 * len(COL_COL_RUNS), "COL_COL_RUNS does not cover rCol"

        tS16_dst = thr_s16.partition_D(tS16f32)
        tX16_dst = thr_x16.partition_D(tX16f32)
        tU16_dst = thr_u16.partition_D(tX16f32)
        tU16s_dst = thr_u16s.partition_D(tU16sf32)
        tKS_src = thr_ks.partition_S(planeKS)
        tO_src = thr_acc32.partition_S(planeO)
        tO_dst = thr_acc32w.partition_D(planeO)
        tU_src = thr_acc32.partition_S(tAccKS[((None, None), 0, 0)])

        # -------------------------------------------------------------------
        # WG B — the five TMEM segments of D0.11. 128 lanes = the v dimension.
        # -------------------------------------------------------------------
        # D1.4 (2): WG B's copy of the one-item-ahead `cu_seqlens` read.
        n_seq = n_item // HV
        nx_lo = cutlass.Int32(mCuSeq[bidx // HV])
        nx_hi = cutlass.Int32(mCuSeq[bidx // HV + 1])

        for r in cutlass.range(n_round):
            item = bidx + r * gdim
            s = item // HV
            h = item % HV

            _seg("item.prologue.begin", r)
            t_lo = nx_lo
            t_hi = nx_hi
            s_nx = cutlass.min((item + gdim) // HV, n_seq - 1)
            nx_lo = cutlass.Int32(mCuSeq[s_nx])
            nx_hi = cutlass.Int32(mCuSeq[s_nx + 1])
            _seg("item.prologue.end", r)

            n_tok = t_hi - t_lo
            n_chunk = (n_tok + CHUNK - 1) // CHUNK

            # D0.14 item prologue, state load: WG B reads the k-last `[V, K]` fp32
            # plane of this item straight into S32, one lane per v row, 128 fp32 per
            # thread (16 x 256-bit `ld.global`, the row is 512 B contiguous) in 4
            # `St32x32b` steps. `partition_S` of the r2t copy names, for this thread,
            # exactly the (v, k) elements its `partition_D` will land in S32, so the
            # gmem side needs no index arithmetic of its own.
            _seg("item.state_load.begin", r)
            gS = mState[(s, h, None, None)]
            tRS_gS = thr_r2t.partition_S(gS)
            cute.autovec_copy(tRS_gS, rS)
            cute.copy(tiled_r2t, rS, tRS_tS)
            # Without this the state is not visible to any later TMEM reader.
            cute.arch.fence_view_async_tmem_store()
            _seg("item.state_load.end", r)

            # D1.4 (3): L2 prefetch of the *next* item's state plane, issued
            # just behind this item's own state load so it cannot delay it. The
            # plane is the one 64 KiB the item prologue pulls through the LSU,
            # and with the rendezvous gone it is the only part of the prologue
            # still in front of B1. Each of the 128 lanes owns one 512 B v row,
            # so four `prefetch.global.L2` per thread cover the whole plane at
            # 128 B granularity, a whole item early. The A rung
            # (`cute.arch.prefetch`) does not lower on this toolchain, so this is
            # the B rung (hw_kb `ptx_escape`): footprint `CCTL.E.PF2`.
            pf_item = cutlass.min(item + gdim, n_item - 1)
            pf_base = mState[(pf_item // HV, pf_item % HV, None, None)] \
                .iterator.toint() + cutlass.Int64(lane_in_wg * (DIM * 4))
            for pfi in cutlass.range_constexpr(4):
                _prefetch_l2(pf_base + (pfi * 128))

            for c in cutlass.range(n_chunk):
                cg = cc + c
                gb = cg % GATE_SLOTS

                # --- B1: S32 -> S16 (S16_SEGS pieces) + the gamma_C decay ---
                _seg("wgb.s32.full.begin", cg)
                if cg >= 1:
                    cute.arch.mbarrier_wait(mbar + BAR_S32_FULL, (cg - 1) % 2)
                _seg("wgb.s32.full.end", cg)
                _seg("wgb.s16.empty.begin", cg)
                if cg >= 1:
                    cute.arch.mbarrier_wait(mbar + BAR_S16_EMPTY, (cg - 1) % 2)
                _seg("wgb.s16.empty.end", cg)
                _seg("wgb.gate.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_GATE_FULL + gb,
                                        (cg // GATE_SLOTS) % 2)
                _seg("wgb.gate.full.end", cg)

                # D0.11 "让位阶梯" rung r1, now at the D2.1 setting
                # S16_SEGS = 4: the state passes through in four batches of 32
                # columns (one `Ld32x32b` each) instead of two of 64. B1 is the
                # peak segment of WG B (the budget line is "non-accumulator
                # working set <= 158 regs"), and 128 state fp32 + the bf16 pack
                # temporaries + the branch's long-lived addresses sat close
                # enough to the 256-register ceiling that ptxas spilled the
                # *long-lived* values -- the footprint was `STL` near the top of
                # the region and one `LDL` reload at the head of every later
                # segment (B2 / B3 / B4 / B5 / item epilogue). Two halves left
                # 15,616 / 15,424 spill ld/st sectors behind; a batch is now
                # 32 fp32 + its 16 packed bf16 registers, and the fp32 live
                # range of batch sg ends at its write-back, before batch sg + 1
                # loads. Each batch publishes its own `S16 seg` barrier, which
                # is what lets the issue warp start group (d) after a quarter of
                # the pass instead of a half; "S32 decayed" still arrives once,
                # after the last batch is written back.
                gamma_c = sGateB[(gb, GATE_GC, 0)]
                for sg in cutlass.range_constexpr(S16_SEGS):
                    for j in cutlass.range_constexpr(sg * J_PER_SEG,
                                                     (sg + 1) * J_PER_SEG):
                        cute.copy(tiled_t2r, tLS_tS[None, None, j],
                                  rState[None, None, j])
                    cute.arch.fence_view_async_tmem_load()
                    for j in cutlass.range_constexpr(sg * J_PER_SEG,
                                                     (sg + 1) * J_PER_SEG):
                        rS16b[None, None, j].store(
                            rState[None, None, j].load().to(cutlass.BFloat16))
                        cute.copy(tiled_s16, rS16[None, None, j],
                                  tS16_dst[None, None, j])
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.mbarrier_arrive(mbar + BAR_S16_SEG + sg)
                    # The same registers, decayed and put back -- this batch's
                    # fp32 live range ends here, before the next batch loads.
                    for j in cutlass.range_constexpr(sg * J_PER_SEG,
                                                     (sg + 1) * J_PER_SEG):
                        rSj = rState[None, None, j]
                        for e in cutlass.range_constexpr(cute.size(rSj)):
                            rSj[e] = rSj[e] * gamma_c
                        cute.copy(tiled_r2t, rState[None, None, j],
                                  tRS_tS[None, None, j])
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar + BAR_S32_DEC)

                # --- B2: X = beta_t (V - gamma_t (K S_0)) -> X16 ------------
                # D0.11 (G5), hw_kb `branchfree_masked_elementwise_pass`
                # variant "[row-lane x col] fragment": this segment used to read
                # two scalars out of the gating array *per element* (128 `LDS`)
                # and spend three FP ops per element on them. The column scalars
                # now come in one `ld.shared.v2` per column run per face (the
                # `ks` fragment's runs are two columns long, so `v2` is the
                # widest legal read), the first `GATE_STRIPS` runs of each face
                # ahead of the `ACC_KS full` wait so their latency runs inside
                # it, and the arithmetic is one `FFMA2` plus one `FMUL2` per
                # column pair. `-gamma_t` arrives already negated from warp 10,
                # which is what turns `V - gamma_t * acc` into a single fma.
                # The gate slot these reads touch was published by the `GATE
                # full` wait at the top of the chunk, i.e. by a strictly earlier
                # barrier -- the precondition for hoisting them at all (gotcha
                # 8) -- and `_lds_v2` is `has_side_effects`, so none of them can
                # sink past a later wait either.
                gate_b_slot = gate_b_base + gb * (GATE_FIELDS * CHUNK * 4)
                ks_beta_b = gate_b_slot + GATE_BETA * (CHUNK * 4)
                ks_ngam_b = gate_b_slot + GATE_NGAM * (CHUNK * 4)
                s_beta = [None] * E_KS
                s_ngam = [None] * E_KS
                for q in cutlass.range_constexpr(GATE_STRIPS):
                    run = KS_COL_RUNS[q]
                    off = cKS[run[0]][1] * 4
                    vb = _lds_v2(ks_beta_b + off)
                    vg = _lds_v2(ks_ngam_b + off)
                    for k, e in enumerate(run):
                        s_beta[e] = vb[k]
                        s_ngam[e] = vg[k]
                _seg("wgb.accks.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_KS_FULL, cg % 2)
                _seg("wgb.accks.full.end", cg)
                _seg("wgb.v.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_V_FULL, cg % 2)
                _seg("wgb.v.full.end", cg)
                cute.copy(tiled_ks, tKS_src, rKS)
                cute.arch.fence_view_async_tmem_load()
                cute.copy(tiled_vt, thr_vt.partition_S(sVt),
                          tiled_vt.retile(rV))
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rKS)):
                                mDbg[(DBG_KS, cKS[e][0], cKS[e][1])] = rKS[e]
                                mDbg[(DBG_V, cKS[e][0], cKS[e][1])] = \
                                    cutlass.Float32(rV[e])
                # Strip-wise: the scalars of run q + GATE_STRIPS are issued
                # before run q is multiplied, so every vector load leads its
                # consumer by four runs and at most five runs' worth of scalars
                # are live at once -- the half-hoist of gotcha 9, which is what
                # B2's ~110-register segment budget affords next to the 64
                # ACC_KS fp32 and the 32 V bf16 pairs.
                for q in cutlass.range_constexpr(len(KS_COL_RUNS)):
                    # `const_expr`, not a plain `if`: the test is a Python bool,
                    # but a bare `if` in a traced body is lifted into an
                    # `scf.if` region whatever its test is, and the `LDS` values
                    # this one produces are consumed *after* it -- outside the
                    # region that defines them. Wrapping the test keeps the
                    # branch where it belongs, at trace time.
                    if cutlass.const_expr(q + GATE_STRIPS < len(KS_COL_RUNS)):
                        nrun = KS_COL_RUNS[q + GATE_STRIPS]
                        noff = cKS[nrun[0]][1] * 4
                        vb = _lds_v2(ks_beta_b + noff)
                        vg = _lds_v2(ks_ngam_b + noff)
                        for k, e in enumerate(nrun):
                            s_beta[e] = vb[k]
                            s_ngam[e] = vg[k]
                    e0, e1 = KS_COL_RUNS[q]
                    x0, x1 = _fma2_pair(s_ngam[e0], s_ngam[e1],
                                        rKS[e0], rKS[e1],
                                        cutlass.Float32(rV[e0]),
                                        cutlass.Float32(rV[e1]))
                    rKS[e0], rKS[e1] = _mul2_pair(s_beta[e0], s_beta[e1],
                                                  x0, x1)
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rKS)):
                                mDbg[(DBG_X, cKS[e][0], cKS[e][1])] = rKS[e]
                rXb.store(rKS.load().to(cutlass.BFloat16))
                cute.copy(tiled_x16, rX, tX16_dst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar + BAR_X_READY)
                cute.arch.mbarrier_arrive(mbar + BAR_V_EMPTY)

                # --- B3: ACC_O *= scale * gamma_t, in place ----------------
                # The `col` fragment is one whole row per lane, so its runs are
                # four columns long and every lane wants the *same* 64 column
                # scalars: 16 broadcast `ld.shared.v4` replace 64 scalar `LDS`,
                # and the `scale *` of the old per-element factor is gone into
                # the face warp 10 publishes. Same strip-wise hoist as B2.
                col_sgam_b = gate_b_slot + GATE_SGAM * (CHUNK * 4)
                s_sgam = [None] * E_COL
                for q in cutlass.range_constexpr(GATE_STRIPS):
                    run = COL_COL_RUNS[q]
                    v = _lds_v4(col_sgam_b + cCol[run[0]][1] * 4)
                    for k, e in enumerate(run):
                        s_sgam[e] = v[k]
                _seg("wgb.acco.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_O_FULL_E, cg % 2)
                _seg("wgb.acco.full.end", cg)
                cute.copy(tiled_acc32, tO_src, rCol)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rCol)):
                                mDbg[(DBG_O_E, cCol[e][0], cCol[e][1])] = \
                                    rCol[e]
                for q in cutlass.range_constexpr(len(COL_COL_RUNS)):
                    # Trace-time branch, as in B2.
                    if cutlass.const_expr(q + GATE_STRIPS < len(COL_COL_RUNS)):
                        nrun = COL_COL_RUNS[q + GATE_STRIPS]
                        v = _lds_v4(col_sgam_b + cCol[nrun[0]][1] * 4)
                        for k, e in enumerate(nrun):
                            s_sgam[e] = v[k]
                    run = COL_COL_RUNS[q]
                    for (e0, e1) in ((run[0], run[1]), (run[2], run[3])):
                        rCol[e0], rCol[e1] = _mul2_pair(
                            s_sgam[e0], s_sgam[e1], rCol[e0], rCol[e1])
                cute.copy(tiled_acc32w, rCol, tO_dst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar + BAR_ACC_O_SCALED)

                # --- B4: U^T -> U16 (plain) and U16s (lambda scaled) -------
                # Same fragment as B3, so the same 16 broadcast `v4` reads --
                # this time of the lambda face -- and the same strip-wise hoist
                # over the `ACC_U full` wait. The four leading runs stay live
                # across the plain U16 pack as well, which is 16 registers.
                col_lam_b = gate_b_slot + GATE_LAM * (CHUNK * 4)
                s_lam = [None] * E_COL
                for q in cutlass.range_constexpr(GATE_STRIPS):
                    run = COL_COL_RUNS[q]
                    v = _lds_v4(col_lam_b + cCol[run[0]][1] * 4)
                    for k, e in enumerate(run):
                        s_lam[e] = v[k]
                _seg("wgb.accu.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_U_FULL, cg % 2)
                _seg("wgb.accu.full.end", cg)
                cute.copy(tiled_acc32, tU_src, rCol)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rCol)):
                                mDbg[(DBG_U, cCol[e][0], cCol[e][1])] = rCol[e]
                rUb.store(rCol.load().to(cutlass.BFloat16))
                cute.copy(tiled_u16, rU, tU16_dst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar + BAR_U16_READY)
                for q in cutlass.range_constexpr(len(COL_COL_RUNS)):
                    # Trace-time branch, as in B2.
                    if cutlass.const_expr(q + GATE_STRIPS < len(COL_COL_RUNS)):
                        nrun = COL_COL_RUNS[q + GATE_STRIPS]
                        v = _lds_v4(col_lam_b + cCol[nrun[0]][1] * 4)
                        for k, e in enumerate(nrun):
                            s_lam[e] = v[k]
                    run = COL_COL_RUNS[q]
                    for (e0, e1) in ((run[0], run[1]), (run[2], run[3])):
                        rCol[e0], rCol[e1] = _mul2_pair(
                            s_lam[e0], s_lam[e1], rCol[e0], rCol[e1])
                if cutlass.const_expr(dbg_on):
                    if item == 0:
                        if c == 0:
                            for e in cutlass.range_constexpr(cute.size(rCol)):
                                mDbg[(DBG_US, cCol[e][0], cCol[e][1])] = rCol[e]
                rUsb.store(rCol.load().to(cutlass.BFloat16))
                cute.copy(tiled_u16s, rUs, tU16s_dst)
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar + BAR_U16S_READY)

                # --- B5: O out to the D0.13 output face ---------------------
                # `Ld16x256b` x n_sub out of ACC_O, bf16 pack, then
                # `STSM.16.MT88.4` into the swizzled face -- the transpose
                # that turns the accumulator's [v, t] into the [t, v] the
                # output tile (and the TMA store box) wants. No scalar STS
                # touches the face.
                _seg("wgb.accoh.full.begin", cg)
                cute.arch.mbarrier_wait(mbar + BAR_ACC_O_FULL_H, cg % 2)
                _seg("wgb.accoh.full.end", cg)
                # D1.3: the face of chunk cg is the one chunk cg - OUT_SLOTS
                # used, and warp 11 released it one chunk after it stored it --
                # i.e. while it was working on chunk cg - OUT_SLOTS + 1. The
                # `UTMASTG` round trip of the previous chunk is no longer in
                # front of this segment.
                os_ = cg % OUT_SLOTS
                sub0 = os_ * n_sub
                _seg("wgb.out.empty.begin", cg)
                if cg >= OUT_SLOTS:
                    cute.arch.mbarrier_wait(mbar + BAR_OUT_EMPTY + os_,
                                            (cg // OUT_SLOTS - 1) % 2)
                _seg("wgb.out.empty.end", cg)
                for st in cutlass.range_constexpr(n_sub):
                    cute.copy(tiled_o_t2r, tTR_tO[(None, None, None, 0, st)],
                              rO)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(dbg_on):
                        if item == 0:
                            if c == 0:
                                cSt = tTR_cO[(None, None, None, 0, st)]
                                for e in cutlass.range_constexpr(
                                        cute.size(rO)):
                                    mDbg[(DBG_O_H, cSt[e][0], cSt[e][1])] = \
                                        rO[e]
                    tRS_rO.store(
                        tiled_o_r2s.retile(rO).load().to(cutlass.BFloat16))
                    cute.copy(tiled_o_r2s, tRS_rO,
                              tRS_sO[(None, None, None, sub0 + st)])
                cute.arch.mbarrier_arrive(mbar + BAR_ACC_O_EMPTY)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(mbar + BAR_OUT_FULL + os_)

                cute.arch.mbarrier_arrive(mbar + BAR_GATE_EMPTY + gb)


            # D0.13 state write-back: S32 -> registers (4 x `Ld32x32b`) -> 16 x
            # 256-bit `st.global` into the k-last `new_state[s, h]` plane, one lane
            # per v row, zero transpose. The M0 starting point is the plain store;
            # the L1 no-allocate modifier of the reference is the D2.4 lever.
            _seg("item.state_store.begin", r)
            if n_chunk >= 1:
                cute.arch.mbarrier_wait(mbar + BAR_S32_FULL,
                                        (cc + n_chunk - 1) % 2)
            gNS = mNewState[(s, h, None, None)]
            tLS_gNS = thr_t2r.partition_D(gNS)
            cute.copy(tiled_t2r, tLS_tS, rSo)
            cute.arch.fence_view_async_tmem_load()
            cute.autovec_copy(rSo, tLS_gNS)
            _seg("item.state_store.end", r)

            # D1.4 (1): no item rendezvous. The state write-back above and the
            # state load at the top of the next iteration are ordered inside
            # WG B by program order alone -- they are the same 128 threads, each
            # touching its own TMEM lane -- and every other role is held off S32
            # by `S32 full` / `S16 empty` as before.
            cc = cc + n_chunk

        # D0.14 CTA teardown, WG B's copy: the rendezvous that says every role
        # has left TMEM alone, then the release itself (warp 4 holds the
        # allocation). It lives *inside* the role branch -- see the teardown
        # note in the specialist branch.
        cute.arch.sync_threads()
        if warp_idx == 4:
            cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS, is_two_cta=False)
        _seg("cta.epilogue.exit")


# ---------------------------------------------------------------------------
# Host path (D0.15): one launch, one cached compile, T and N dynamic.
# ---------------------------------------------------------------------------
@cute.jit
def _gdn_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mState: cute.Tensor,
    mALog: cute.Tensor,
    mA: cute.Tensor,
    mDtBias: cute.Tensor,
    mB: cute.Tensor,
    mCuSeq: cute.Tensor,
    mOut: cute.Tensor,
    mNewState: cute.Tensor,
    mDbg: cute.Tensor,
    scale: cutlass.Float32,
    n_item: cutlass.Int32,
    grid_x: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    # D0.9 group (a)/(b): bf16 x bf16 -> f32, cta_group::1, both operands
    # K-major out of smem. Its A-side face geometry is the one the TMA lands.
    tiled_mma = _mma_ab()
    k_layout = sm100_utils.make_smem_layout_a(tiled_mma, MMA_TILER,
                                              cutlass.BFloat16, K_SLOTS)
    q_layout = sm100_utils.make_smem_layout_a(tiled_mma, MMA_TILER,
                                              cutlass.BFloat16, Q_SLOTS)
    v_layout = sm100_utils.make_smem_layout_a(tiled_mma, MMA_TILER,
                                              cutlass.BFloat16, V_SLOTS)
    one_stage = cute.select(k_layout, mode=[0, 1, 2])

    # D0.4 staging s0 / s1: the B-operand face of the M = 128, N = 64, K = 64
    # groups (g)/(h), i.e. a 64 (= N, token) x 64 (= K, index) bf16 SW128 face
    # the UMMA descriptor reads directly.
    mma_up = _mma_m128()
    stage_layout = sm100_utils.make_smem_layout_b(
        mma_up, TILER_UP, cutlass.BFloat16, STAGE_SLOTS)
    # The same K / Q tile bytes as B operands of the M = 128 groups: K-major
    # (N = token, K = feature) for (d)/(e), MN-major (N = feature, K = token)
    # for (i). SW128 makes the two descriptors read the same bytes.
    kb_layout = sm100_utils.make_smem_layout_b(
        mma_up, TILER_KS, cutlass.BFloat16, K_SLOTS)
    qb_layout = sm100_utils.make_smem_layout_b(
        mma_up, TILER_KS, cutlass.BFloat16, Q_SLOTS)
    kmn_layout = sm100_utils.make_smem_layout_b(
        _mma_su(), TILER_SU, cutlass.BFloat16, K_SLOTS)

    # D0.8: one tile = 64 tokens x 128 features = 16 KiB, landed as two SW128
    # boxes of 64 x 64 (inner extent 128 B) by a single atom, so the expect_tx
    # byte count of a tile is the whole 16 KiB.
    tx_bytes = cute.size_in_bytes(cutlass.BFloat16, one_stage)

    op = cpasync.CopyBulkTensorTileG2SOp()
    tma_k, gmemK = cute.nvgpu.make_tiled_tma_atom_A(op, mK, one_stage,
                                                    MMA_TILER, tiled_mma)
    tma_q, gmemQ = cute.nvgpu.make_tiled_tma_atom_A(op, mQ, one_stage,
                                                    MMA_TILER, tiled_mma)
    tma_v, gmemV = cute.nvgpu.make_tiled_tma_atom_A(op, mV, one_stage,
                                                    MMA_TILER, tiled_mma)

    # D0.13 output epilogue. The MMA issues O transposed (M = 128 = v, the
    # contiguous dimension of `output`), so the epilogue frame is (v, token)
    # with v contiguous, and both op selectors and the face builder are keyed
    # on `COL_MAJOR` -- the MMA's frame, not the torch tensor's row-major
    # [t, h, v] (hw_kb `tcgen05_epilogue_t2r_stsm_tma_store` gotcha 13). The
    # gmem side is presented in that same frame, because the swizzled face has
    # no second view (gotcha 12): `output` seen as (v, token, head).
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        (DIM, CHUNK, CHUNK), False, LayoutEnum.COL_MAJOR, cutlass.BFloat16)
    epi_mn = cute.product_each(cute.shape(epi_tile))
    n_sub = (DIM // epi_mn[0]) * (CHUNK // epi_mn[1])
    # D1.3: `n_sub * OUT_SLOTS` sub-tiles, i.e. OUT_SLOTS whole faces back to
    # back. The TMA atom is still built off one sub-tile of the face, so the
    # descriptor is unchanged by the depth.
    out_layout = sm100_utils.make_smem_layout_epi(
        cutlass.BFloat16, LayoutEnum.COL_MAJOR, epi_tile, n_sub * OUT_SLOTS)
    # `out_layout.inner` is the swizzle, and it is load-bearing: a face without
    # one compiles, runs and is numerically right, only 8-way bank conflicted
    # on the `stmatrix` side, so no numerical verdict would catch its loss
    # (that KB entry, gotcha 11). `notes/probe_m0_step6_epilogue.py` carries it
    # as a standing non-numerical contract.
    mOutF = cute.make_tensor(mOut.iterator,
                             cute.select(mOut.layout, mode=[2, 0, 1]))
    tma_o, gmemO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mOutF,
        cute.slice_(out_layout, (None, None, 0)), epi_tile)

    _gdn_kernel(
        tma_k, gmemK, tma_q, gmemQ, tma_v, gmemV, tma_o, gmemO,
        mState, mALog, mA, mDtBias, mB, mCuSeq, mOut, mNewState, mDbg,
        k_layout, q_layout, v_layout, stage_layout,
        kb_layout, qb_layout, kmn_layout, out_layout, epi_tile,
        scale, n_item, tx_bytes, n_sub, epi_mn[1],
        _DEBUG_SINK is not None,
    ).launch(
        grid=[grid_x, 1, 1],
        block=[THREADS, 1, 1],
        # D0.7 / hw_kb `warp_specialization` pit 1: `nvvm.minctasm` is only
        # emitted when this is > 0, and without it ptxas cannot determine the
        # entry register count, so it drops every `setmaxnreg` (info-level
        # C7508, which the DSL's JIT never surfaces) and compiles all three
        # warpgroups at the uniform 65536 / 384 = 168. It does not change the
        # occupancy the rest of the design already pins to 1 CTA/SM.
        min_blocks_per_mp=1,
        stream=stream,
    )


_COMPILED: dict = {}


def _dyn(t: torch.Tensor, stride_order, align: int = 16):
    """DSL view of a contiguous torch tensor with mode 0 (T or N) dynamic."""
    return (
        from_dlpack(t, assumed_align=align)
        .mark_compact_shape_dynamic(mode=0, stride_order=stride_order)
    )


def _tma_view(t: torch.Tensor):
    """(T, H, D) -> (T, D, H): the (M, K, L) form the A-side TMA atom wants.

    A permute is a view, not a copy — the feature dimension stays the
    contiguous one, which is what makes the 128 B inner box of D0.8 legal.
    """
    p = t.permute(0, 2, 1)
    return _dyn(p, p.dim_order())


def gdn_prefill_varlen(
    q: torch.Tensor,           # [T, HQ=4, D=128] bf16
    k: torch.Tensor,           # [T, HK=4, D=128] bf16
    v: torch.Tensor,           # [T, HV=8, D=128] bf16
    state: torch.Tensor,       # [N, HV=8, D, D] fp32, k-last [N, H, V, K]; READ-ONLY
    A_log: torch.Tensor,       # [HV] fp32, learned log decay
    a: torch.Tensor,           # [T, HV] bf16, input-dependent decay drive
    dt_bias: torch.Tensor,     # [HV] fp32, learned decay bias
    b: torch.Tensor,           # [T, HV] bf16, update-gate drive
    cu_seqlens: torch.Tensor,  # [N + 1] int64, cumulative lengths, cu_seqlens[-1] == T
    scale: float,              # output scale (campaign: 0.08838834764831843 = 1/sqrt(128))
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (output [T, 8, 128] bf16, new_state [N, 8, 128, 128] fp32 k-last)."""
    n_tok = q.shape[0]
    n_seq = state.shape[0]

    # D0.15: `empty`, not `zeros` — a memset would be a second launch.
    output = torch.empty((n_tok, HV, DIM), dtype=torch.bfloat16, device=q.device)
    new_state = torch.empty((n_seq, HV, DIM, DIM), dtype=torch.float32, device=state.device)

    n_item = n_seq * HV
    grid_x = max(1, min(MAX_CTAS, n_item))

    tensors = (
        _tma_view(q),
        _tma_view(k),
        _tma_view(v),
        # D0.14 / D0.13: the state planes are declared 32-byte aligned so that
        # `autovec_copy` may take its 256-bit cap -- 16 vector accesses per
        # thread per plane instead of 32, which is the access width D0.14
        # (load) and D0.13 (store) name. The claim is checked host-side by
        # `from_dlpack` (a torch allocation is 512 B aligned) and every
        # per-thread offset inside a plane is a 512 B row stride.
        _dyn(state, (0, 1, 2, 3), align=32),
        from_dlpack(A_log, assumed_align=16),
        _dyn(a, (0, 1)),
        from_dlpack(dt_bias, assumed_align=16),
        _dyn(b, (0, 1)),
        _dyn(cu_seqlens, (0,), align=8),
        _dyn(output, (0, 1, 2)),
        _dyn(new_state, (0, 1, 2, 3), align=32),
        # The probe sink, or -- with the dump traced away -- a second view of a
        # tensor the kernel already has, so the production launch allocates
        # nothing extra and the parameter is never read.
        _dyn(_DEBUG_SINK, (0, 1, 2)) if _DEBUG_SINK is not None
        else _dyn(new_state, (0, 1, 2, 3), align=32),
    )
    scalars = (cutlass.Float32(scale), cutlass.Int32(n_item), cutlass.Int32(grid_x))
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    key = (q.device.index, _DEBUG_SINK is not None)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_gdn_launch, *tensors, *scalars, stream)
        _COMPILED[key] = fn
    fn(*tensors, *scalars, stream)

    return output, new_state
