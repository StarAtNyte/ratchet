"""fp8_gemm — the student's kernel file (the only file the student edits).

Dense FP8 GEMM (Fprop, NT layout), fine-grained scaling, fixed campaign point
m4096_n4096_k4096. The signature below is the card's frozen contract; the
student implements the body in CuTe DSL; the signature itself may not change.

Semantics: C = dequant(A) @ dequant(B)^T computed blockwise --

    C[i, j] = sum over 128-deep K blocks kb of
              a_sf[i, kb] * b_sf[j // 128, kb] * ( A[i, kb*128:(kb+1)*128]
                                                 . B[j, kb*128:(kb+1)*128] )

with the block partial sums accumulated in true fp32: Hopper's FP8 tensor-core
accumulator keeps only ~14 mantissa bits, so partial sums MUST be promoted to
fp32 registers at most every 128 into K (two-level accumulation) or the
correctness court fails at K=4096.

Scales arrive in the plain layouts documented below; the kernel may repack
them internally (a one-time cached repack is legitimate -- layout preparation
is not the contest, the GEMM is). Output is bf16.

NOTE (hw_kb wgmma_tma_mbarrier gotcha 13 / wgmma_ss_min_gemm gotcha 8):
this file must NOT carry `from __future__ import annotations` -- PEP 563 would
stringify the @cute.struct / @cute.kernel annotations and cute.compile throws.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum


# --------------------------------------------------------------------------
# M0 Gate ledger (closing round; no delivery in this commit -- comment only).
#
# Read against the NCU signature of the M0 package commit 9480047, which the
# driver evaluated whole (the M0 sub-steps carried no gate of their own, D0.13).
#
#   Correctness
#     calc_diff <= 1e-3 vs the fp32 dequant matmul oracle ......... PASS
#
#   Architecture conformance (each vs the value this file declares)
#     D0.1   kernel_count = 1, dominant_time_pct 100.0 >= 99.0 .... PASS
#     D0.1/4 grid_size = 132 ................................... PASS
#     D0.5   block_size = 384 .................................. PASS
#     D0.6   smem 0 static + 217088 dynamic = 217088 B, inside
#            216192 +/- 2048 = [214144, 218240] ................. PASS
#     D0.5/9 spill_ld_sectors = spill_st_sectors = 0 ............ PASS
#     D0.1/12 dram_bytes_write 24244640 <= 1.30 x 22234880
#            = 28905344 (1.090x, so no split-K traffic) ......... PASS
#
#   Mechanism verification (separates "the mechanism runs" from "it happened
#   to produce the right numbers")
#     overlap floor    tensor_pct 71.15 >= 65.0 ................. PASS
#     read-side path   bankconf_ld 372.625 <= 1000 .............. PASS
#
#   Not gated here, by the milestone's own text: stall_barrier (advisory
#   attribution only), subtile interleave (D0.11), multicast operand
#   ownership (D0.7), MMA atom shape, the 40/232 split, peak live 200.
#
#   No absolute performance line at M0 by construction.  For the record the
#   package measured device_median_ms 0.114 against target_ms 0.1029 in the
#   device window; closing that gap is M1-M3 lever work.
#
# All nine gated predicates hold, so from this commit the signature and the
# D0.1-D0.12 resource-topology contract are frozen: M1+ moves levers only.
# Gate predicates stop binding here -- planned lever drift in M1+ (e.g. the
# smem byte count moving with STAGE) is not a breach.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# M1 / D1.1 -- mbarrier wait-encoding gear.  Gear taken: (b) non-blocking
# try + explicit tight spin.  Starting point was (a) the blocking suspend
# face; this commit moves off it.
#
# Scope is the producer->consumer supply chain only, i.e. the two waits that
# sit on the k-loop critical path:
#
#   producer empty wait  (l.~361)  producer_try_acquire spin, then the
#     satisfied token is handed back to producer_acquire(pstate, tok).  The
#     call cannot be dropped on this side: for PipelineTmaAsync it also
#     issues the full barrier's arrive_and_expect_tx.  With the token only
#     the wait is skipped, so the blocking wait survives in the binary behind
#     a runtime branch -- static NANOSLEEP count does not go to zero here.
#   consumer full wait   (l.~454)  consumer_try_wait spin replaces
#     consumer_wait outright (it only ever waits), so this side leaves no
#     blocking wait in the binary at all.
#
# Untouched, deliberately: every arrive / release / expect-tx count and
# thread set; the epilogue store_pipe acquire (once per output tile, off the
# supply path, D0.12 territory); the SFB-publish and epilogue named barriers.
# The D0.7 topology is unchanged -- same STAGE mbarrier pairs, same full
# arrive 1 / empty arrive 16, same cluster scope, same smem and registers.
# Phases are each caller's own state.phase.
#
# GPU probe before delivery, notes/probe_d11_spin_wait.py, 3/3 PASS: the
# non-blocking face resolves on the PipelineTmaAsync class this file
# instantiates and its producer_acquire takes a token; the shipped entry
# still traces (both dynamic `while tok == 0` loops lower to scf.while
# rather than folding at trace time); end-to-end calc_diff 1.387779e-06 at
# the campaign point, which is also the no-deadlock verdict -- a swallowed
# token or a mis-ordered spin hangs instead of reaching that line.
#
# D1.1 is decided by a same-shape A/B on device median + stall_barrier, and
# the driver's eval of this commit is the (b) arm.  The (a) arm is the M0
# gate round, iter 1: device_median_ms 0.114, stall_barrier 5.12,
# tensor_pct 71.31, spill 0/0.  Whichever arm is faster is the gear; a loss
# here means reverting to (a), not a defect in the topology.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# M1 / D1.2 -- pipeline stage depth.  Candidate domain {2, 3}.  The A/B is
# complete and this commit docks the lever at the winner, STAGE = 3.
#
# Both arms, same topology, same D1.1 gear (b), driver eval, campaign point:
#   STAGE 3 (iter 2)  device_median_ms 0.1056  tensor_pct 77.85
#                     stall_barrier 4.79  smem_dynamic 217088  spill 0/0
#                     bankconf_st 23028  bankconf_ld 1.75
#   STAGE 2 (iter 3)  device_median_ms 0.1247  tensor_pct 67.22
#                     stall_barrier 5.62  smem_dynamic 166912  spill 0/0
#                     bankconf_st 91207  bankconf_ld 627.25
# The D1.2 criterion is device median + tensor_pct + smem_dynamic; the first
# two move together and both favour 3 by far more than run-to-run spread
# (+18.1% device time, -10.6 pts tensor_pct), so 3 is the docked position and
# the depth-2 arm is spent.
#
# STAGE is a single constant and every consumer of it is parametric, so the
# move back is that one line.  Restored by it, relative to the depth-2 arm:
#   sA   BM*BK*STAGE   65536 ->  98304 B
#   sB   BN*BK*STAGE   32768 ->  49152 B
#   sSFA BM*STAGE       2048 ->   3072 B
#   mbar STAGE*2 Int64     32 ->     48 B  (3 mbarrier pairs again)
#   smem/CTA          166048 -> 216240 B
# and, through make_pipeline_state / PipelineTmaAsync(num_stages=STAGE), the
# modulus of both pipeline states -- producer run-ahead is again at most 3
# k-blocks (D0.9's run-ahead bound), which is what the M0 segment ledger was
# written against.
#
# Deliberately unchanged across both arms and this one: the buffer list
# itself (D0.6 -- D staging is still its own 65536 B, not aliased onto the
# A/B ring), tile geometry 256x128x128, the 40/232 register split, the
# arrive/release counts and thread sets, cluster scope, multicast masks, and
# the D1.1 gear (b) at both wait points.  Single-variable lever, both ways.
#
# Docking at 3 also returns smem_dynamic to the M0 gate band and leaves
# kernel_count 1 / grid_size 132 / block_size 384 as they were in every
# round so far (216240 B and 166048 B are both over half the 233472 B per-SM
# budget, so occupancy was 1 CTA/SM in both arms).
#
# 4 is not in the candidate domain: 4*(32768+16384) + 65536 D staging =
# 262144 B > 227 KiB, and fitting it would require aliasing D staging onto
# the A/B ring, i.e. editing the D0.6 buffer list, which M1+ may not do.
# The domain is therefore exhausted with the starting point winning, which
# is the form D1.2 explicitly says does not trigger the escalation clause
# (that clause fires only on 2 beating 3).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# M1 Gate ledger (closing round; submitted alone, no delivery bundled -- this
# commit is comment only).
#
# Read against the NCU signature of the docked M1 configuration, commit
# 625c786: D1.1 gear (b) non-blocking try + tight spin, D1.2 STAGE = 3, the
# D0.x topology untouched.  That is the program this file currently is; the
# diff carrying this ledger adds no executable statement.
#
#   Correctness
#     calc_diff <= 1e-3 vs the fp32 dequant matmul oracle ......... PASS
#
#   Performance ladder (device window, the gate window per 5.2)
#     device_median_ms 0.1059 <= 0.1285 .......................... PASS
#     (the ladder's 0.1285 is tensor_pct 65 -- the M0 floor -- converted to
#      an equivalent time; M1 asks only that the supply-chain levers leave
#      that lower bound.  For the record: reference 0.1029 ms, so the docked
#      arm sits 1.029x of it, and the e2e window reads 0.1256 ms.)
#
#   Mechanism verification line (catches "levers moved but the overlap did
#   not" -- an end-to-end PASS with this line short may not close M1)
#     tensor_pct 77.42 >= 70.0 ................................... PASS
#     (M0 closed at 71.15 / 71.31 with gear (a); the two docked levers put
#      it at 77.42, i.e. the end-to-end improvement is attributable to the
#      overlap the levers were supposed to buy, not to something else.
#      Reference is 81.18.)
#
#   Register economy
#     spill_ld_sectors = 0, spill_st_sectors = 0 ................. PASS
#
#   Architecture conformance lines (all held every round of M1)
#     kernel_count = 1, dominant_time_pct 100.0 .................. PASS
#     grid_size = 132 ............................................ PASS
#     block_size = 384 ........................................... PASS
#
# D1.3 (wgmma wait_group depth 0 vs 1) is NOT executed, on its own trigger
# clause: it runs "only once D1.1 and D1.2 are both docked AND the M1 gate is
# still short", because its expected value is negative -- the decision buys a
# falsification reading, not speed, and it predicts spill != 0 (192 + 64 =
# 256 > the 232 math-side budget) plus a slowdown.  D1.1 and D1.2 are docked,
# but the gate above is not short: every line passes, with device median
# 0.1059 against a 0.1285 ceiling and tensor_pct 77.42 against a 70.0 floor.
# The second conjunct is false, so the A/B does not fire and depth stays at
# 0.  The forbidden-zone row for wait_group >= 1 therefore stands unchanged
# and un-retried -- D1.3 remains its only planned execution entrance, still
# unspent, should a later milestone find the gate short.
#
# Nothing here is escalated to meta: D1.2's forced-escalation trigger fires
# only on 2 beating 3 (3 won), and the ladder is met, so no INFEASIBLE or
# short-gate report is due.
#
# M1 closes with the supply-chain lever positions frozen for M2+: wait
# encoding (b), STAGE 3, wgmma wait depth 0.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# M2 package -- D2.1 + D2.2 + D2.3 + D2.4, delivered as one commit.
#
# The four are bundled because they draw on one shared account, and the
# milestone says single A/Bs of them mask each other: the *issue slots and
# registers of the math segment* (D0.9's segment ledger -- non-accumulator
# working set 8 words, ~32 words of headroom, 192 of the math side's 232
# registers already spoken for by 128 promotion + 64 wgmma outputs).  The
# knobs they share, and their positions after this commit:
#
#   register file placement: warp-invariant bookkeeping goes to the uniform
#     file (UR*) via cute.arch.make_warp_uniform, so it does not draw on the
#     per-thread setmaxnreg allowance.  Marked seeds: the tile-index chain
#     (ctile / m_block / n_block) in both the producer and the consumer, and
#     both pipeline stage indices.  warp_idx / cta_rank / wg_idx were already
#     marked in M0.  Thread-variable quantities are out of scope by
#     construction -- they physically cannot enter UR.
#   per-thread residency: exactly one new resident register, the row-scale
#     base sSFA_thr (D2.4 (a)).  That is the whole per-thread cost of the
#     package; D0.9's headroom absorbs it and spill must stay 0.
#   k-loop body content: after this commit the body's addressing is one
#     resident base + one uniform stage displacement + compile-time
#     immediates.  No tid-derived multiply / shift / mask is regenerated per
#     k-block per subtile -- D2.4's static self-check.
#   payload shape: 4 scale loads per k-block per math warpgroup, the
#     reference count; 2 FMUL per subtile for the scale products; ACC_VALS
#     FFMA per subtile for the promotion.  Nothing per-element is re-fetched
#     or re-multiplied.
#   frozen and untouched: the D0.x topology (buffer list, tile 256x128x128,
#     40/232 split, arrive/release counts and thread sets, cluster scope and
#     multicast masks) and the M1 lever positions (wait encoding (b),
#     STAGE 3, wgmma wait depth 0).
#
# Per decision, what is in the executable diff:
#
#   D2.1 (uniform register file).  make_warp_uniform on the tile-index chain
#     in both warp roles -- these are the TMA coordinates -- and on both
#     stage indices.  Each stage index is marked *at its use site*, on the
#     far side of the D1.1 spin region: the producer's pidx after the
#     try-acquire spin, the consumer's st after the try-wait spin.  A marking
#     does not survive an scf region boundary (hw_kb wgmma_desc_uniform
#     gotcha 2), so a single marking at the seed would have been dropped
#     exactly where it matters.  The consumer's st is also the wgmma
#     descriptor *slice coordinate*, which is the marking hw_kb
#     wgmma_desc_uniform r1 prescribes (the descriptor itself is not
#     markable).
#
#   D2.2 (row-scale read form).  The two rows a thread owns in the m64nNk32
#     C-fragment (r0 and r0+8) are read per pair through one base with an
#     immediate displacement, and the scale product a_sf*b_sf is hoisted out
#     of the FFMA body -- 2 FMUL per subtile, not one per element.  A literal
#     64-bit pair load is not available on top of this: the pair's two rows
#     are 8 rows = 32 B apart in the D0.3 MN-major SFA payload, not adjacent,
#     and D0.3 is frozen M0 contract.  The load count is the same 4 per
#     k-block per warpgroup either way.
#
#   D2.3 (promotion instruction form).  The segment is 2 FMUL + ACC_VALS
#     FFMA, each FFMA reading the wgmma output register and the hoisted scale
#     product and writing the promotion accumulator in place.  No
#     intermediate copy, so the conversion terminates the upstream live range
#     (D0.9).  Broadcast axis is M, which is hw_kb cfrag_elementwise_fastpath
#     case (b): the row scalars hoist out of the unrolled body and it reduces
#     to 1 FFMA per element.
#
#   D2.4 (loop-invariant bookkeeping).  (a) the row-scale smem base, the only
#     tid-derived quantity the math segment needs, is folded into a base
#     pointer once before the k-loop via _rebase (cute.domain_offset); (b)
#     the body reads it as sSFA_thr[((h, j), st)] with h, j compile-time
#     constants, so the four addresses are that base plus the
#     +0x0 / +0x20 / +0x200 / +0x220 immediate quartet (pair = 8 rows = 0x20
#     B, half = 128 rows = 0x200 B) -- the same quartet the reference uses;
#     (c) stage advance and the empty release move one uniform pointer, no
#     per-thread base is rebuilt in the body.  D2.4 is the complement of
#     D2.1 and D2.2: it moves *address computation* out of the k-loop, where
#     D2.1 only chooses a register file for warp-invariants and D2.2 only
#     fixes the scale payload shape.
#
# GPU layout-probe before delivery (notes/probe_d24_sfa_hoist.py, 3
# contracts, all PASS): C1 cute.domain_offset resolves on this build and is
# the spelling _rebase takes; C2 the hoisted view and the direct indexing it
# replaces are the same bijection -- an isolated kernel stages an SFA-shaped
# buffer through the direct spelling and reads it back through the hoisted
# one at a runtime uniform stage, checked against a numpy regather, max_diff
# 0.0, which is what rules out the 64-row-periodic mis-address (§1 diagnosis
# list #4) that a wrong hoist produces instead of a crash; C3 the shipped
# entry with the package applied traces, compiles and is numerically right
# end to end at the campaign point against an fp32 dequant-matmul reference,
# calc_diff 1.388931e-06 -- this is what covers the make_warp_uniform
# markings, which are assertions and so fail as wrong numbers, not as a
# compile error.
#
# What the probe deliberately does not cover, and where it is read instead:
# whether any of this is faster (driver's device median) and whether the
# bookkeeping actually landed in UR (M2 Gate's stall-sampling line, the
# thread-variable integer/address opcode family at <= 4.5% of in-loop stall
# samples, anchor 2.12% -- read from ncu, not from a probe).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# M2 Gate ledger (closing round; submitted alone, no delivery bundled -- this
# commit is comment only).
#
# Read against the NCU signature of the M2 package commit 40fea0f, which is
# the program this file currently is (D2.1 + D2.2 + D2.3 + D2.4 applied, the
# D0.x topology and the M1 lever positions -- wait encoding (b), STAGE 3,
# wgmma wait depth 0 -- untouched).  The diff carrying this ledger adds no
# executable statement.
#
#   Correctness
#     calc_diff <= 1e-3 vs the fp32 dequant matmul oracle ......... PASS
#
#   Performance ladder (device window, the gate window per 5.2)
#     device_median_ms 0.1060 <= 0.1144 .......................... PASS
#     (0.1144 is reference 0.1029 x 81.18/73, i.e. the tensor_pct 73 floor
#      converted to an equivalent time.  For the record the docked M1 config
#      already read 0.1055 and the package reads 0.1060: the package is flat
#      to M1 within the +/-1% noise band, so its account is carried by this
#      gate, not by a per-decision delta.  e2e window reads 0.1260 ms.)
#
#   Mechanism verification line (the D2.4 decision line) ......... DEFERRED
#     In-loop stall samples with the thread-variable integer/address opcode
#     family (IMAD / IADD3 / VIADD / LEA / ULEA / LOP3 / ULOP3 / SHF /
#     UIADD3) at <= 4.5%, anchored on the reference's 2.12% (ULEA 1.34 +
#     UIADD3 0.78) [MEASURED, job 832284].  Not read this round: the line
#     needs `ncu --set full` stall sampling (smsp__pcsamp_warp_stall_reasons*)
#     resolved per SASS line, folded by opcode, and restricted to main-loop
#     samples.  The per-round profiler digest carries only the three
#     aggregate stall shares (stall_barrier / stall_long_sb / stall_mio) with
#     no opcode or SASS-line resolution, and the probe channel returns only
#     PROBE verdict lines -- neither can produce the numerator or the
#     main-loop-restricted denominator this line is defined on.  That is the
#     acquisition-unavailable case the line's own failure clause names, so it
#     records DEFERRED with M3 as the named receiver, and this round is
#     adjudicated by the tensor_pct backstop below, exactly as the clause
#     prescribes.
#
#     Four-state admission, the three conditions:
#       (1) reference meets it in the same envelope: 2.12% at the same shape,
#           same hardware, same smem/register envelope [MEASURED, job
#           832284].  So the line is not INFEASIBLE -- 6.4's boundary puts it
#           on the other side.
#       (2) local lever + mechanism account: the lever this line judges
#           (D2.4) is spent -- there is no second position of it to measure,
#           and the reason the line is open is the missing instrument, not a
#           lever that failed to respond.  What is checkable without the
#           instrument is D2.4's own static self-check, and it holds in the
#           shipped source: the k-loop body's only smem-scale addressing is
#           sSFA_thr[((h, j), st)] with h, j compile-time constants over the
#           _rebase'd base, i.e. one resident base + one uniform stage
#           displacement + the +0x0/+0x20/+0x200/+0x220 immediate quartet;
#           no tid-derived multiply / shift / mask is regenerated per k-block
#           per subtile.  Static self-check is weaker evidence than the stall
#           census (it cannot see what the backend re-materialises), which is
#           why the line defers rather than closes.
#       (3) named receiving milestone: M3, whose gate must give a final
#           ruling on every deferred line and may not close with one open.
#     Falsification clause: if the M3 round obtains the census and the family
#     reads > 4.5%, D2.4 did not land as written and the end-to-end reading
#     must be re-attributed before M2 may be closed.  Per the line's own
#     text, a red reading here is a form signal -- it does not trigger REVERT
#     of the package.
#
#   tensor_pct backstop (sanity, and this round's substitute judge)
#     tensor_pct 77.55 >= 73.0 ................................... PASS
#     (M1 closed at 77.76, reference is 81.18.  The milestone marks this line
#      as unable to distinguish "did it" from "did nothing" -- it was already
#      satisfied at the M2 entry -- so it is recorded as a backstop, not as
#      evidence that D2.1-D2.4 landed.)
#
#   Register economy
#     spill_ld_sectors = 0, spill_st_sectors = 0 ................. PASS
#     (D2.4's one new resident register, the sSFA_thr base, fits D0.9's
#      headroom as its cost clause predicted.)
#
#   Read-side path
#     bankconf_ld 2.125 <= 1000 .................................. PASS
#     (D2.2's criterion; reference 0.  bankconf_st 25580 is not gated until
#      M3 and sits well under that gate's 1.5 x 68132.25 = 102198.)
#
#   Architecture conformance (held every round since M0)
#     kernel_count = 1, dominant_time_pct 100.0, grid_size 132,
#     block_size 384, smem 0 + 217088 B ......................... PASS
#
# Five gated lines PASS, one DEFERRED to M3 under its own clause.  M2 closes
# with the math-segment forms frozen for M3: uniform-file bookkeeping, paired
# row-scale read at 4 LDS per k-block per warpgroup, promotion at 2 FMUL +
# ACC_VALS FFMA in place, and the thread-variable base hoisted out of the
# k-loop.  M3 opens with D3.1, and carries the deferred stall-census line.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# D3.1 ledger -- epilogue path refinement.  Two parts: (i) the atom
# calibration, which this round confirms and records without changing a
# selection, and (ii) the store-pipeline throttle placement, which is this
# round's executable change.  Both are read against the M2-closing signature
# at 86fd0a1: device_median_ms 0.1055, tensor_pct 77.40, bankconf_st
# 25444.25, bankconf_ld 3.375, spill 0/0, smem 0 + 217088 B, grid 132,
# block 384, kernel_count 1.
#
# (i) swizzle mode and STSM member selection (hw_kb tma_store_epilogue §3
#     "r2s 三层构造顺序固定" / cfrag_epilogue_stmatrix 机制 B).  The single
#     LayoutEnum C_LAYOUT = ROW_MAJOR (mC is [M, N] row-major, i.e. n-major
#     C) drives all four call sites, and they are consistent by construction
#     rather than by hand:
#       1. make_d_smem_layout: get_smem_layout_atom(C_LAYOUT, BFloat16, BN)
#          -- n-major C takes the K branch, and the major-mode width is
#          BN * 16 = 2048 bits, a multiple of the 1024-bit SW128 period, so
#          the selection is K_SW128, the widest swizzle the atom contract
#          offers.  K_SW64 / K_SW32 / K_INTER are the narrower members of the
#          same domain; taking one of them would only shrink the XOR period
#          below the 8-row x 16 B group an STSM addresses, which is the
#          conflict-free condition itself.
#       2. sm90_get_smem_store_op(C_LAYOUT, elem_ty_d=BFloat16,
#          elem_ty_acc=Float32) -- the 16-bit output width selects
#          StMatrix8x8x16bOp(is_m_major_c(), 4), i.e. the x4 member; the
#          non-16-bit fall-back is CopyUniversalOp (scalar store), which is
#          the "机制 B 缺失" form the KB's trigger section names.
#       3. StMatrix8x8x16bOp(C_LAYOUT.is_m_major_c(), 4) in the C-atom that
#          carries the value grouping -- same member, same transpose flag
#          (False for n-major C, so no transposing STSM variant).
#       4. the TMA store descriptor, built from a slice of the very same
#          sD layout object, so the write-side XOR bijection and the
#          descriptor's swizzle are the same object, not two spellings that
#          have to be matched (KB gotcha 2 / gotcha 3).
#     Measured position of the selection: bankconf_st 25444.25 against the
#     reference's 68132.25 (0.37x), bankconf_ld 3.375.  The pairing is left
#     where the atom contract puts it; no member is changed this round.
#
# (ii) store-pipeline throttle placement -- the executable change.  Before:
#      per tile the epilogue ran STSM -> fence -> barrier -> {TMA store,
#      commit, acquire} -> barrier, so `wait_group.read 0` sat between the
#      store and the tile-loop back edge and every CTA drained its 64 KiB
#      store to completion before the next tile's mainloop was allowed to
#      start.  After: the acquire is the first thing the *next* tile's
#      epilogue does, immediately ahead of that tile's STSM writes, followed
#      by the rendezvous that publishes it to the other seven math warps.
#      The hazard being guarded is unchanged and so is its guard -- sD is a
#      single stage (D0.6, frozen), so the only overwrite hazard is STSM
#      against the previous tile's in-flight TMA read, and `wait_group.read
#      0` + a barrier is correct anywhere between the previous commit and
#      the next STSM.  What moves is which work sits inside the shadow: the
#      NKB k-blocks of tile i+1 now execute with tile i's store in flight.
#      Invariants held: two named barriers per tile (D0.12's count), both
#      executed by all MATH_THREADS undiverged; one bulk-group commit per
#      tile from a whole warp (gotcha 6); producer_tail unchanged as the
#      drain of the last tile's store; sSFB is not touched between the two
#      barriers, so the tile-i+1 SFB publish still cannot race a tile-i
#      k-loop read.  No smem, register, mbarrier, arrive-count or launch
#      geometry change: this is a reordering of three statements.
# --------------------------------------------------------------------------

# M0 architecture constants (D0.1 / D0.4 / D0.5 / D0.6 / D0.7) -- frozen.
#
# M0 ships as one package; the budgets and knobs its decisions share are:
#
#   smem 216240 B / CTA (band 216192 +/- 2048, so 1 CTA/SM) -- sA 98304 @0,
#     sB 49152 @98304, sD 65536 @147456, sSFA 3072 @212992, sSFB 128 @216064,
#     mbar 48 @216192.  D0.4 (tile) x D0.6 (partition) x D0.7 (STAGE) x D0.12
#     (D staging not aliased onto the A/B ring) all draw on this one number;
#     it is what pins STAGE to 3 and forbids 4.
#   registers 40/232 via setmaxnreg (128*40 + 256*232 = 64512 <= 65536).
#     D0.2 (128 fp32 promotion) + D0.9 (64 wgmma outputs, one set reused by
#     both subtile chains) = 192 of the math side's 232; that residue is what
#     forces wait_group(0) in D0.9 and leaves ~8 words for addressing.
#   synchronisation: STAGE mbarrier pairs (full arrive 1, empty arrive
#     8 warps x 2 CTAs = 16, cluster-scoped) carry every streaming handoff
#     (D0.7); named barriers appear only outside the k-loop -- 1 for the SFB
#     publish (D0.8) and 2 in the epilogue (D0.12).  Per k-block per math WG
#     that leaves exactly 3 blocking rendezvous (1 full wait + 2 wait_group),
#     which is the convoy budget of D0.10.
#   tile geometry: BMxBNxBK 256x128x128 (D0.4) fixes MMA_M=128 per half, the
#     subtile = half*2 + wg_id interleave (D0.11), the 256-entry SFA payload
#     (D0.3), and the 256x128 single-shot TMA store (D0.12).
#
# Lever positions: STAGE=3 (D1.2, docked at its starting point after the
# {2,3} A/B -- 2 was tried in iter 3 and lost), wait_group(0) (D1.3,
# starting point), 40/232 (D3.3, starting point), grouped rasterisation
# group=16 (D3.2, starting point).  D1.1 has
# also moved off its starting point -- both mainloop wait points now take the
# non-blocking try + tight-spin face instead of the blocking suspend face;
# see the two call sites below.  D3.1's swizzle/STSM pairing stays at the
# atom contract's selection (K_SW128 + StMatrix x4, non-transposing); its
# store-throttle placement has moved -- the epilogue's `wait_group.read 0`
# now guards the *next* tile's STSM rather than the tile-loop back edge.
# --------------------------------------------------------------------------
M_DIM, N_DIM, K_DIM = 4096, 4096, 4096

BM, BN, BK = 256, 128, 128          # D0.4 output tile / k-block depth
MMA_M = 128                         # 2 wgmma atoms (64 each) stacked over M
M_HALVES = BM // MMA_M              # 2 -> subtile = half*2 + wg_id (D0.11)
STAGE = 3                           # D0.7 pipeline depth (lever D1.2)
SF_BLK = 128                        # fine-grained scaling block
NKB = K_DIM // BK                   # 32 k-blocks per output tile

NUM_MATH_WG = 2
MATH_THREADS = NUM_MATH_WG * 128    # 256  (warps 0-7)
PRODUCER_WARP = 8                   # warps 8-11 = producer warpgroup (D0.5)
NUM_THREADS = 384
PRODUCER_REGS = 40                  # D0.5 setmaxnreg split (lever D3.3)
MATH_REGS = 232                     # 128*40 + 256*232 = 64512 <= 65536

CLUSTER = 2                         # D0.7 cluster 2x1x1, A/SFA multicast
NUM_SM = 132
NUM_CLUSTERS = NUM_SM // CLUSTER    # 66

MBLOCKS = M_DIM // BM               # 16
NBLOCKS = N_DIM // BN               # 32
NPAIRS = NBLOCKS // CLUSTER         # 16 cluster-wide N slots
CTILES = MBLOCKS * NPAIRS           # 256 cluster-tiles (512 CTA tiles)
RAST_GROUP = 16                     # D3.2 grouped rasterisation, group = 16

ACC_VALS = (MMA_M * BN) // MATH_THREADS   # 64 fp32 per thread per subtile

_SFA_ELEMS = BM * STAGE             # 768 fp32  (3 KiB)
_SFB_ELEMS = NKB                    # 32 fp32   (128 B)

# api drift (hw_kb cute_dsl_trace_scope gotcha 4): make_fragment -> make_rmem_tensor
_MAKE_FRAG = getattr(cute, "make_fragment", None)
if _MAKE_FRAG is None:
    _MAKE_FRAG = cute.make_rmem_tensor


# api drift (D2.4): folding a runtime element offset into a tensor's base
# pointer.  `cute.domain_offset(coord, t)` is the documented spelling (hw_kb
# tma_ndim_oob_zerofill uses it to translate a tensor's coordinate origin);
# older/leaner builds only expose iterator arithmetic.  Both compute the same
# pointer, so resolve once at import and let the probe certify the mapping.
_HAS_DOMAIN_OFFSET = hasattr(cute, "domain_offset")


def _rebase(t, coord, lin):
    """Iterator for `t` translated to `coord` (== linear element offset `lin`).

    The whole point of the call is *when* it happens: used outside a loop it
    turns a per-iteration `base + f(tid)` into a base register computed once
    (D2.4 (a)).  `lin` must equal `t.layout(coord)`; it is only the fallback
    spelling of the same offset.
    """
    if _HAS_DOMAIN_OFFSET:
        return cute.domain_offset(coord, t).iterator
    return t.iterator + lin


# api drift: the generic->async-proxy shared fence has several spellings across
# DSL versions (hw_kb cp_async_mbarrier_ring gotcha 11 says to getattr-probe).
# cutlass-dsl 4.6.1 takes *string literals* ("async.shared", space="cta") and
# its ProxyKind.from_str raises TypeError on an enum; older builds hang the
# enums off cute.arch.  All candidates emit `fence.proxy.async.shared::cta`.
def _resolve_async_proxy_fence():
    """-> (nullary callable emitting the fence, human-readable spelling)."""
    fp = getattr(cute.arch, "fence_proxy", None)
    if fp is not None:
        kind_enum = getattr(cute.arch, "ProxyKind", None)
        if kind_enum is None:                      # 4.6.x string-literal API
            return ((lambda: fp("async.shared", space="cta")),
                    "cute.arch.fence_proxy('async.shared', space='cta')")
        kind = kind_enum.async_shared
        space_enum = getattr(cute.arch, "SharedSpace", None)
        if space_enum is not None:                 # older enum API
            space = space_enum.shared_cta
            return ((lambda: fp(kind, space=space)),
                    "cute.arch.fence_proxy(ProxyKind.async_shared, "
                    "space=SharedSpace.shared_cta)")
        return ((lambda: fp(kind)),
                "cute.arch.fence_proxy(ProxyKind.async_shared)")
    for nm in ("fence_view_async_shared", "fence_proxy_async_shared_cta",
               "fence_proxy_async", "fence_view_async_shared_cta"):
        f = getattr(cute.arch, nm, None)
        if f is not None:
            return (lambda f=f: f()), "cute.arch.%s()" % nm
    raise RuntimeError(
        "no async-proxy shared fence found; cute.arch exposes: %s"
        % sorted(a for a in dir(cute.arch)
                 if "fence" in a.lower() or "proxy" in a.lower()))


try:
    _ASYNC_PROXY_FENCE, _ASYNC_PROXY_FENCE_DESC = _resolve_async_proxy_fence()
except Exception as _e:          # keep import alive so probes can report it
    _ASYNC_PROXY_FENCE_DESC = "UNRESOLVED: %s" % _e

    def _ASYNC_PROXY_FENCE(_err=_e):
        raise _err


C_LAYOUT = LayoutEnum.ROW_MAJOR     # mC is [M, N] row-major (n-major C)


# --------------------------------------------------------------------------
# shared constructors: the kernel and the layout probes instantiate the very
# same objects through these, so a probe can never drift from the shipped path.
# --------------------------------------------------------------------------
def make_tiled_mma():
    """D0.4: wgmma m64n128k32 e4m3xe4m3 -> fp32, 2 atoms stacked over M."""
    return sm90_utils.make_trivial_tiled_mma(
        cutlass.Float8E4M3FN, cutlass.Float8E4M3FN,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        cutlass.Float32, (NUM_MATH_WG, 1, 1), tiler_mn=(64, BN),
    )


def make_ab_smem_layouts():
    ab_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            LayoutEnum.ROW_MAJOR, cutlass.Float8E4M3FN, BK),
        cutlass.Float8E4M3FN)
    return (cute.tile_to_shape(ab_atom, (BM, BK, STAGE), order=(0, 1, 2)),
            cute.tile_to_shape(ab_atom, (BN, BK, STAGE), order=(0, 1, 2)))


def make_d_smem_layout():
    """D0.6 buffer 3: the 256x128 bf16 D staging, swizzled, single stage."""
    epi_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(C_LAYOUT, cutlass.BFloat16, BN),
        cutlass.BFloat16)
    return cute.tile_to_shape(epi_atom, (BM, BN, 1), order=(0, 1, 2))


def make_sfa_smem_layout():
    """D0.3: MN-major row scales -- one contiguous 1024 B TMA payload/stage."""
    return cute.make_layout((1, BM, STAGE), stride=(BM, 1, BM))


def make_r2s_copy(tiled_mma):
    """D0.12: STSM-family reg->smem store atom laid on the MMA C-frag tiling."""
    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
        C_LAYOUT, elem_ty_d=cutlass.BFloat16, elem_ty_acc=cutlass.Float32)
    copy_atom_c = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(C_LAYOUT.is_m_major_c(), 4),
        cutlass.BFloat16)
    return cute.make_tiled_copy_S(
        copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma))


@cute.struct
class SharedStorage:
    """D0.6 smem partition table -- the contract. Declaration order fixes the
    buffer sequence A, B, D, SFA, SFB, mbar; the three STAGE-sized buffers
    (sA, sB, sSFA) scale with the D1.2 lever, the rest do not.
      STAGE 3: A(0) B(98304) D(147456) SFA(212992) SFB(216064) mbar(216192),
               total 216240 B -- the M0 gate reading, and current: D1.2
               docked the lever here after the A/B.
      STAGE 2: A(0) B(65536) D(98304) SFA(163840) SFB(165888) mbar(166016),
               total 166048 B -- the losing arm, iter 3.
    Both are > 233472/2, so 1 CTA/SM either way (D0.1 persistent 132)."""
    sA: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float8E4M3FN, BM * BK * STAGE], 1024]
    sB: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float8E4M3FN, BN * BK * STAGE], 1024]
    sD: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, BM * BN], 1024]
    sSFA: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, _SFA_ELEMS], 1024]
    sSFB: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, _SFB_ELEMS], 16]
    mbar: cute.struct.MemRange[cutlass.Int64, STAGE * 2]


@cute.kernel
def fp8_gemm_kernel(
    tma_a: cute.CopyAtom, mA: cute.Tensor,
    tma_b: cute.CopyAtom, mB: cute.Tensor,
    tma_sfa: cute.CopyAtom, mSFA: cute.Tensor,
    tma_c: cute.CopyAtom, mC: cute.Tensor,
    mSFB: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tx_count: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---------------- shared prelude (before the single role branch) --------
    # hw_kb warp_specialization gotcha 11: everything shared lives *before*
    # setmaxnreg, and there is exactly one role branch in the whole kernel.
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_a)
        cpasync.prefetch_descriptor(tma_b)
        cpasync.prefetch_descriptor(tma_sfa)
        cpasync.prefetch_descriptor(tma_c)

    # MLIR region isolation: a (composed) layout built in the *host* @cute.jit
    # region is an SSA value of that region, and `.outer`/`.inner` taken here
    # lower to `cute.composed_get_outer` on a value defined outside this
    # region -> "ICE IR Verification Failed".  Every layout the kernel body
    # touches is therefore constructed *inside* the kernel region, from the
    # same shared constructors the host uses for its TMA atoms.
    sA_layout, sB_layout = make_ab_smem_layouts()
    sD_layout = make_d_smem_layout()
    sSFA_layout = make_sfa_smem_layout()

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
    sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
    sD = storage.sD.get_tensor(sD_layout.outer, swizzle=sD_layout.inner)
    sSFA = storage.sSFA.get_tensor(sSFA_layout)
    sSFB = storage.sSFB.get_tensor(cute.make_layout(_SFB_ELEMS))

    # D0.7: one mbarrier pair per stage; empty is cluster-scoped and takes
    # 8 consumer warps x 2 CTAs = 16 arrivals.
    cta_layout_vmnk = cute.make_layout((1, 1, CLUSTER, 1))
    mainloop = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mbar.data_ptr(),
        num_stages=STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, (MATH_THREADS // 32) * CLUSTER),
        tx_count=tx_count,
        cta_layout_vmnk=cta_layout_vmnk,
    )

    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cta_coord_vmnk = cta_layout_vmnk.get_hier_coord(cta_rank)
    mcast_mask = cpasync.create_tma_multicast_mask(
        cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2)
    mcast_layout = cute.make_layout(
        cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape)

    gA = cute.zipped_divide(mA, (BM, BK))          # ((BM,BK),(MBLOCKS,NKB))
    gB = cute.zipped_divide(mB, (BN, BK))          # ((BN,BK),(NBLOCKS,NKB))
    gSFA = cute.zipped_divide(mSFA, (1, BM))       # ((1,BM),(NKB,MBLOCKS))
    gC = cute.zipped_divide(mC, (BM, BN))          # ((BM,BN),(MBLOCKS,NBLOCKS))

    tAsA, tAgA = cpasync.tma_partition(
        tma_a, cta_coord_vmnk[2], mcast_layout,
        cute.group_modes(sA, 0, 2), gA)
    tSsS, tSgS = cpasync.tma_partition(
        tma_sfa, cta_coord_vmnk[2], mcast_layout,
        cute.group_modes(sSFA, 0, 2), gSFA)
    tBsB, tBgB = cpasync.tma_partition(
        tma_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 2), gB)
    bSG_sD, bSG_gD = cpasync.tma_partition(
        tma_c, 0, cute.make_layout(1),
        cute.group_modes(sD, 0, 2), gC)

    # D0.1/D3.2: persistent grouped rasterisation over cluster-tiles.
    cluster_id = bidx // CLUSTER
    num_tiles = (CTILES - cluster_id + NUM_CLUSTERS - 1) // NUM_CLUSTERS

    # ---------------- the one role branch (D0.5) ---------------------------
    if warp_idx >= PRODUCER_WARP:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)

        if warp_idx == PRODUCER_WARP:
            pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, STAGE)
            for it in cutlass.range(num_tiles, unroll=1):
                # D2.1: the whole tile-index chain is warp-invariant (it is fed
                # by bidx and the loop counter only), so mark it at the seed and
                # it stays on the uniform datapath -- these are the TMA
                # coordinates, i.e. exactly the "TMA 坐标" clause.
                ctile = cute.arch.make_warp_uniform(
                    cluster_id + it * NUM_CLUSTERS)
                # D3.2 grouped rasterisation, group = RAST_GROUP: with
                # MBLOCKS == RAST_GROUP == 16 the group map is exactly
                # m-fastest (one group covers every m-block).
                m_block = cute.arch.make_warp_uniform(ctile % MBLOCKS)
                n_block = cute.arch.make_warp_uniform(
                    (ctile // MBLOCKS) * CLUSTER + cta_rank)

                for kb in cutlass.range(NKB, unroll=1):
                    # D1.1 lever, non-blocking face: spin on the empty
                    # barrier with producer_try_acquire, then hand the
                    # satisfied token back.  producer_acquire is *not*
                    # droppable on this side -- for PipelineTmaAsync it also
                    # issues the full barrier's arrive_and_expect_tx, and
                    # skipping that sets no transaction count = deadlock
                    # (hw_kb mbarrier_wait_encoding gotcha 2).  With the token
                    # only the wait is skipped; the expect-tx arrive still
                    # happens.
                    tok = mainloop.producer_try_acquire(pstate)
                    while tok == 0:
                        tok = mainloop.producer_try_acquire(pstate)
                    mainloop.producer_acquire(pstate, tok)
                    bar = mainloop.producer_get_barrier(pstate)
                    # D2.1 / D2.4 (c): the stage index is warp-invariant but it
                    # is read back *after* the spin region above, and a marking
                    # does not survive an scf region boundary (hw_kb
                    # wgmma_desc_uniform gotcha 2) -- re-mark it at the use
                    # site so the three smem destinations advance by one
                    # uniform pointer instead of three per-thread ones.
                    pidx = cute.arch.make_warp_uniform(pstate.index)
                    cute.copy(tma_a,
                              tAgA[(None, (m_block, kb))],
                              tAsA[(None, pidx)],
                              tma_bar_ptr=bar, mcast_mask=mcast_mask)
                    cute.copy(tma_sfa,
                              tSgS[(None, (kb, m_block))],
                              tSsS[(None, pidx)],
                              tma_bar_ptr=bar, mcast_mask=mcast_mask)
                    cute.copy(tma_b,
                              tBgB[(None, (n_block, kb))],
                              tBsB[(None, pidx)],
                              tma_bar_ptr=bar)
                    mainloop.producer_commit(pstate)
                    pstate.advance()
    else:
        cute.arch.setmaxregister_increase(MATH_REGS)

        wg_idx = cute.arch.make_warp_uniform(tidx // 128)
        wg_mma = tiled_mma.get_slice(
            cute.make_layout(NUM_MATH_WG, stride=128)(wg_idx))

        # D0.11: partition_A over a 256-row A tile with a 128-row MMA tile
        # puts half h / warpgroup w on subtile h*2 + w -- the interleaved
        # assignment (WG0 owns rows [0,64) and [128,192)).
        sA_h = [cute.local_tile(sA, (MMA_M, BK, STAGE), (h, 0, 0))
                for h in range(M_HALVES)]
        tCrA = [wg_mma.make_fragment_A(wg_mma.partition_A(sA_h[h]))
                for h in range(M_HALVES)]
        tCrB = wg_mma.make_fragment_B(wg_mma.partition_B(sB))

        acc_shape = wg_mma.partition_shape_C((MMA_M, BN))
        acc = _MAKE_FRAG(acc_shape, cutlass.Float32)
        promo = [_MAKE_FRAG(acc_shape, cutlass.Float32)
                 for _ in range(M_HALVES)]

        # wgmma m64nNk32 C-fragment: this thread owns rows r0 and r0+8 of its
        # 64-row atom; value v carries the +8 row when (v//2)%2 == 1.
        r0 = (tidx // 128) * 64 + ((tidx % 128) // 32) * 16 + (tidx % 32) // 4

        # D2.4 (a)+(b) / D2.2: the row-scale read, hoisted into one base.
        #
        # `r0` is the only tid-derived quantity the math segment needs, and it
        # is loop-invariant in k.  Fold it into a base pointer here, once, and
        # give that pointer a layout whose remaining modes are the four scales
        # this thread reads per k-block:
        #
        #     ((half, pair), stage)  strides ((MMA_M, 8), BM)
        #
        # so the k-loop body indexes it as sSFA_thr[((h, j), st)] with h and j
        # compile-time constants and st the (uniform) stage.  The address that
        # reaches LDS is therefore `one resident base register + a uniform
        # stage displacement + an immediate`, and the immediates are exactly
        # the +0x0 / +0x20 / +0x200 / +0x220 quartet (4 B elements: pair = 8
        # rows = 0x20 B, half = 128 rows = 0x200 B).  Nothing tid-derived is
        # recomputed inside the loop -- that is D2.4's static self-check.
        #
        # D2.2's payload clause is met at 4 LDS per k-block per warpgroup, the
        # reference count, with no per-element scatter.  A literal 64-bit load
        # per pair is *not* available on top of it: the two rows of a pair are
        # 8 rows = 32 B apart in the D0.3 MN-major SFA payload, not adjacent,
        # and D0.3 is frozen M0 contract.  The pair mode above is the same idea
        # expressed as one base + one immediate, which is what the reference
        # itself does.
        sSFA_thr = cute.make_tensor(
            _rebase(sSFA, (0, r0, 0), r0),
            cute.make_layout(((M_HALVES, 2), STAGE), stride=((MMA_M, 8), BM)))

        mma_first = cute.make_mma_atom(tiled_mma.op)
        mma_first.set(warpgroup.Field.ACCUMULATE, False)
        mma_acc = cute.make_mma_atom(tiled_mma.op)
        mma_acc.set(warpgroup.Field.ACCUMULATE, True)

        # D0.12 epilogue machinery
        tiled_copy_r2s = make_r2s_copy(tiled_mma)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        sD_h = [cute.local_tile(sD, (MMA_M, BN, 1), (h, 0, 0))
                for h in range(M_HALVES)]
        tRS_sD = [thr_r2s.partition_D(sD_h[h]) for h in range(M_HALVES)]
        # the copy source must carry the copy's (R2S, R2S_M, R2S_N) value
        # order, not the wgmma C-frag order -- take the shape from the
        # partition itself.  The whole 128x128 half is one epi subtile
        # (num_epi == 1), so the pack buffer holds the half's 64 values.
        rD_shape = cute.shape(thr_r2s.partition_S(sD_h[0]))
        rD_layout = cute.make_layout(rD_shape[:3])

        store_pipe = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, MATH_THREADS))
        sfb_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=MATH_THREADS)
        epi_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=MATH_THREADS)

        cstate = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGE)

        for it in cutlass.range(num_tiles, unroll=1):
            # D2.1: same warp-invariant tile-index chain as the producer's.
            ctile = cute.arch.make_warp_uniform(cluster_id + it * NUM_CLUSTERS)
            m_block = cute.arch.make_warp_uniform(ctile % MBLOCKS)
            n_block = cute.arch.make_warp_uniform(
                (ctile // MBLOCKS) * CLUSTER + cta_rank)

            # D0.8: SFB staging + CTA-wide publish (one rendezvous per tile).
            if tidx < NKB:
                sSFB[tidx] = mSFB[(n_block, tidx)]
            sfb_bar.arrive_and_wait()

            for h in cutlass.range_constexpr(M_HALVES):
                promo[h].fill(0.0)

            for kb in cutlass.range(NKB, unroll=1):
                # D1.1 lever, non-blocking face: consumer_wait does nothing
                # but wait, so the tight spin replaces it outright (hw_kb
                # mbarrier_wait_encoding (b)) -- unlike the producer side this
                # leaves no blocking wait in the binary at all.  The phase is
                # cstate's own (gotcha 4); the loop sits directly in the
                # @cute.kernel body so the dynamic Boolean lowers to scf.while
                # rather than being folded at trace time (gotcha 5).
                tok = mainloop.consumer_try_wait(cstate)
                while tok == 0:
                    tok = mainloop.consumer_try_wait(cstate)
                # D2.1 / D2.4 (c): re-mark the stage index on the far side
                # of the spin region.  It feeds both the wgmma descriptor slice
                # coordinates below (hw_kb wgmma_desc_uniform r1: mark the
                # *slice coordinate*, the descriptor itself cannot be marked)
                # and the row-scale displacement, so losing its uniformity
                # costs 2 dwords per descriptor per issue plus a per-thread
                # address chain per k-block.
                st = cute.arch.make_warp_uniform(cstate.index)
                bscale = sSFB[kb]

                for h in cutlass.range_constexpr(M_HALVES):
                    # D2.2 / D2.4 (b): one resident base, immediate offsets.
                    # 2 FMUL per subtile, hoisted clear of the 64 FFMA below.
                    s0 = sSFA_thr[((h, 0), st)] * bscale
                    s1 = sSFA_thr[((h, 1), st)] * bscale

                    warpgroup.fence()
                    cute.gemm(mma_first, acc,
                              tCrA[h][(None, None, 0, st)],
                              tCrB[(None, None, 0, st)], acc)
                    for kk in cutlass.range_constexpr(BK // 32 - 1):
                        cute.gemm(mma_acc, acc,
                                  tCrA[h][(None, None, kk + 1, st)],
                                  tCrB[(None, None, kk + 1, st)], acc)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)

                    # D2.3: the promotion segment at its instruction floor.
                    # 2 FMUL (above, the scale products) + ACC_VALS FFMA, each
                    # FFMA reading the wgmma output register and the hoisted
                    # scale product and writing the promotion accumulator *in
                    # place* -- no intermediate copy, so the conversion
                    # terminates the upstream live range (D0.9).  The broadcast
                    # axis is M here, which is the case hw_kb
                    # cfrag_elementwise_fastpath (b) covers: the row scalars
                    # hoist out of the fully unrolled inner body and it reduces
                    # to 1 FFMA per element with no per-element re-fetch.
                    ph = promo[h]
                    for v in cutlass.range_constexpr(ACC_VALS):
                        # trace-time select: v is a python int here, so this
                        # is a plain conditional expression, not staged flow.
                        sc = s0 if (v // 2) % 2 == 0 else s1
                        ph[v] = ph[v] + acc[v] * sc

                mainloop.consumer_release(cstate)
                cstate.advance()

            # D0.12 / D3.1: pack -> STSM -> proxy fence -> barrier ->
            # 1 TMA store, with the store-pipeline throttle carried to the
            # head of the *next* tile's epilogue (see below).
            #
            # D3.1, throttle placement.  sD is a single stage (D0.6, frozen),
            # so the only overwrite hazard is this tile's STSM against the
            # previous tile's in-flight TMA read of the same buffer.  The
            # guard for that hazard is `wait_group.read 0` + a rendezvous,
            # and it is correct anywhere between the previous commit and
            # this STSM.  Placing it here -- ahead of the pack, i.e. on the
            # far side of the k-loop that has just run -- means the k-loop
            # of tile i+1 executes while tile i's 64 KiB store is still in
            # flight, instead of the CTA draining the store at the tile
            # boundary before the next mainloop may start.  The barrier
            # count is unchanged at two per tile (hw_kb tma_store_epilogue
            # gotcha 5: the second rendezvous of the classic seven-step
            # order is exactly this guard, kept, only relocated); the
            # arrive/commit counts, the thread sets (all MATH_THREADS,
            # undiverged) and the single-warp issue (gotcha 6) are also
            # unchanged.  `producer_acquire` is a no-op on the first tile
            # (no bulk group outstanding), and the final store is drained by
            # the unchanged `producer_tail` below.
            if warp_idx == 0:
                store_pipe.producer_acquire()
            epi_bar.arrive_and_wait()

            # the fp32->bf16 conversion terminates the promotion live range
            # (D0.9): the pack buffer is the only epilogue staging.
            for h in cutlass.range_constexpr(M_HALVES):
                tRS_rAcc = tiled_copy_r2s.retile(promo[h])
                rD = _MAKE_FRAG(rD_layout.shape, cutlass.BFloat16)
                for v in cutlass.range_constexpr(cute.size(rD_layout)):
                    rD[v] = cutlass.BFloat16(tRS_rAcc[v])
                cute.copy(tiled_copy_r2s, rD,
                          tRS_sD[h][(None, None, None, 0)])
            _ASYNC_PROXY_FENCE()
            epi_bar.arrive_and_wait()
            if warp_idx == 0:
                cute.copy(tma_c, bSG_sD[(None, 0)],
                          bSG_gD[(None, (m_block, n_block))])
                store_pipe.producer_commit()

        store_pipe.producer_tail()

    # cluster-scoped barriers stay live until every peer CTA is done.
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()


@cute.jit
def fp8_gemm_jit(mA: cute.Tensor, mSFA: cute.Tensor,
                 mB: cute.Tensor, mSFB: cute.Tensor,
                 mC: cute.Tensor, stream):
    # host-region copies, used only to build the TMA atoms.  The kernel builds
    # its own inside its own region (see the note at the smem allocation).
    tiled_mma = make_tiled_mma()
    sA_layout, sB_layout = make_ab_smem_layouts()
    sD_layout = make_d_smem_layout()
    sSFA_layout = make_sfa_smem_layout()

    a_one = cute.slice_(sA_layout, (None, None, 0))
    b_one = cute.slice_(sB_layout, (None, None, 0))
    d_one = cute.slice_(sD_layout, (None, None, 0))
    sfa_one = cute.slice_(sSFA_layout, (None, None, 0))

    tma_a, tA = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(), mA, a_one, (BM, BK),
        num_multicast=CLUSTER)
    tma_sfa, tSFA = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(), mSFA, sfa_one, (1, BM),
        num_multicast=CLUSTER)
    tma_b, tB = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mB, b_one, (BN, BK),
        num_multicast=1)
    tma_c, tC = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mC, d_one, (BM, BN))

    tx_count = (cute.size_in_bytes(cutlass.Float8E4M3FN, a_one)
                + cute.size_in_bytes(cutlass.Float8E4M3FN, b_one)
                + cute.size_in_bytes(cutlass.Float32, sfa_one))

    fp8_gemm_kernel(
        tma_a, tA, tma_b, tB, tma_sfa, tSFA, tma_c, tC, mSFB,
        tiled_mma, tx_count,
    ).launch(
        grid=[NUM_SM, 1, 1],
        block=[NUM_THREADS, 1, 1],
        cluster=[CLUSTER, 1, 1],
        min_blocks_per_mp=1,      # hw_kb warp_specialization gotcha 1
        stream=stream,
    )


# --------------------------------------------------------------------------
# host entry
# --------------------------------------------------------------------------
_STATE = {}


def _cuda_stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        import cuda.bindings.driver as _drv
    except ImportError:
        from cuda import cuda as _drv
    return _drv.CUstream(raw)


def _build(a, a_sf, b, b_sf):
    # D0.3: one-time cached repack of the row scales into MN-major [K/128, M].
    a_sf_t = a_sf.t().contiguous()
    c = torch.empty(a.shape[0], b.shape[0], device=a.device,
                    dtype=torch.bfloat16)

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mSFA = from_dlpack(a_sf_t, assumed_align=16)
    mSFB = from_dlpack(b_sf, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)

    stream = _cuda_stream()
    compiled = cute.compile(fp8_gemm_jit, mA, mSFA, mB, mSFB, mC, stream)
    return compiled, (mA, mSFA, mB, mSFB, mC), a_sf_t, c


def fp8_gemm(
    a: torch.Tensor,      # [M, K]         float8_e4m3fn, row-major   M=K=4096
    a_sf: torch.Tensor,   # [M, K/128]     fp32, per-token 1x128 scales
    b: torch.Tensor,      # [N, K]         float8_e4m3fn, row-major   N=4096
    b_sf: torch.Tensor,   # [N/128, K/128] fp32, per-128x128-block scales
) -> torch.Tensor:
    """Returns c [M, N] bf16 = dequant(a) @ dequant(b)^T."""
    key = (a.data_ptr(), a_sf.data_ptr(), b.data_ptr(), b_sf.data_ptr(),
           tuple(a.shape), tuple(b.shape))
    ent = _STATE.get(key)
    if ent is None:
        ent = _build(a, a_sf, b, b_sf)
        _STATE.clear()
        _STATE[key] = ent
    compiled, args, _a_sf_t, c = ent
    compiled(*args, _cuda_stream())
    return c
