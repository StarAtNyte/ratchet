"""fp4_grouped_gemm — the student's kernel file (the only file the student edits).

M-grouped FP8 x FP4 GEMM on Blackwell (B200, sm_100a): MoE Fprop, NT layout,
contiguous grouping, micro-scaled operands — FP8 (e4m3) activations with
per-token 1x128 scales and FP4 (e2m1) expert weights with per-row 1x32 scales,
both scale kinds UE8M0 (exact powers of two), bf16 output. Fixed campaign point
g8_em4096_n4096_k4096. The signature below is the card's frozen contract; the
student implements the body in CuTe DSL; the signature itself may not change.

Semantics: the M axis is a concatenation of G=8 expert segments. Row i belongs
to expert m_indices[i] (int32; -1 marks an alignment-padding row). For every
valid row i with g = m_indices[i] >= 0 and every output column j:

    C[i, j] = sum over k of  A_deq[i, k] * B_deq[g, j, k]

    A_deq[i, k]    = float(a[i, k])                      * a_sf[i, k // 128]
    B_deq[g, j, k] = e2m1(nibble_k of b[g, j, k // 2])   * b_sf[g, j, k // 32]

with the products accumulated in fp32 (the tensor core's block-scaled MMA
already applies both scales per 32-deep K block and accumulates in fp32; a
kernel that keeps partial sums in a narrower format fails the correctness
court at K=4096). FP4 packing: element k = 2j sits in the LOW nibble of
b[g, :, j], element k = 2j+1 in the HIGH nibble; e2m1 code = sign bit 3 +
magnitude index into {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Scales arrive as fp32 tensors
whose values are exact powers of two (UE8M0-representable), i.e. the exponent
byte is the whole scale.

Grouping contract: every expert segment's start is aligned to ALIGN rows (see
eval/bench.py ALIGN — the reference library's recommended contiguous-layout
alignment on this architecture), so an ALIGN-row M block never straddles two
experts; the trailing block of a segment may contain padding rows
(m_indices == -1), and an expert may own zero rows. Output rows for padding
are UNCHECKED — compute them, zero them, or skip them. Scales, weights and
indices arrive in the plain layouts documented below; the kernel may repack
them internally and may derive block-level scheduling metadata from m_indices
(e.g. a per-block expert table) — a one-time cached repack is legitimate:
layout preparation is not the contest, the grouped GEMM is. Output is bf16.

-------------------------------------------------------------------------------
Build state: M0 gate submission — every M0 decision (D0.1-D0.14) is mounted at
MOUNT_SHIPPED and the whole-machine smoke certificate this row needs is on the
lineage: `notes/probe_m0_gate_smoke.py` read PROBE PASS on the shipped tree with
all five rungs green — C1-smoke (campaign shape lowers, launches, completes, and
relaunches through the same compiled entry on cloned buffers), C4-launch and
C5-court (the ragged geometry, 16 m-blocks against the 148-wide grid with a
zero-row and a one-row expert, i.e. the tail wave that used to hang), C2-court
(the frozen court from eval/bench.py on the campaign case) and C3-rotate (the
same data through fresh pointers is bitwise identical, so the D0.2 identity
cache is not answering a stale key).  Every M0 mechanism is on the machine in
the milestone's real form -- cta_group::2 TMEM pair allocation, the pair TMA
pipeline, the tcgen05.commit multicast and the TMA writeback; the two-loss
fallback (`cta_group::1`) shape was never mounted.  This commit changes no
executable code: it is the gate row, submitted alone.  What is here:

  S1 (already smoked at the real campaign shape)
    D0.5  grid 148x1x1, block 256 (8 warps), cluster (2,1,1), dynamic smem at
          the D0.7 budget, CTA i walks tile linear ids i, i+148, ...  The trip
          count is the compile-time constant ceil(num_tiles / GRID) on every
          CTA, with the out-of-range iterations of the tail wave MASKED (the
          producer clamps its gmem indices to tile 0 and only the epilogue's TMA
          store is predicated off) rather than skipped: a run-time trip count
          gives the two halves of a pair different depths whenever num_tiles is
          not a multiple of GRID, and the shallower half walks past a barrier the
          other half is still waiting on -- which is the ragged-shape hang
          `notes/probe_m0_tail_bisect.py` isolated (148-block campaign and a
          synthetic 37-block exact-multiple shape both completed; the 16-block
          ragged shape hung) and C4/C5 above now certify fixed.
    D0.6  warp-role dispatch (w0 TMA producer on BOTH CTAs, w1 MMA issuer on
          the leader only, w2 TMEM alloc/dealloc, w4-7 epilogue) and the
          cta_group::2 TMEM allocation handshake (512 columns), run through
          `cutlass.utils.TmemAllocator` with every call unguarded on both
          halves of the pair -- `tcgen05.alloc.cta_group::2` is asymmetric
          rather than leader-only, and the leader-only spelling hung on the
          SECOND launch [notes/probe_m0_tmem_pair2.py, rung T1-a2].
    D0.13 inter-CTA rasterization: 16-wide super-group tile -> (m_block,
          n_block).  Because bidx and GRID are both even, a pair's in_group
          index is always even, so the two CTAs of a pair always land on the
          same m_block and on adjacent n_blocks with the even one on the
          leader.  That is what lets each CTA fetch exactly its own half of the
          pair's MMA operands with a per-CTA tiler and no MMA-side partition on
          the delivery path.

  S2 (operand supply + MMA)
    D0.4  tcgen05 cta_group::2 kind::mxf8f6f4 MMA, M=256 x N=224 x K=32, four
          instructions per 128-deep k-block; MMA-A = wgt, MMA-B = act; fp32
          accumulator in TMEM as a 128(n) x 224(m) C-transpose block.
    D0.7  smem topology: NSTAGE stages of {act 14336, wgt 16384, sf_act 1024,
          sf_wgt 512} bytes per CTA, TMA on the write side, UMMA descriptor /
          UTCCP on the read side.  The buffer list and the atoms are the frozen
          M0 contract; the stage COUNT is the D1.3 lever (M0 start 6).
    D0.8  the fp4 weight face is a 1 byte/element container plane (Uint8 smem
          layout, TMA internal_type=Uint8 -> U4_UNPACK_U8, expect-tx counted on
          the 8192 PACKED bytes, never on the 16384 container bytes).
    D0.9  both scale factors reach smem already in the blocked SF layout and go
          to TMEM with tcgen05.cp; no transpose warp, no sf_ready barrier.
    D0.10 NSTAGE-deep full/empty pipeline: one full barrier per stage, armed by the
          LEADER only and for the whole pair's bytes, with BOTH CTAs delivering
          into it (that is the entire point of CopyBulkTensorTileG2SOp(
          CtaGroup.TWO); without it the redirected completion signal is UB and
          the barrier simply never fills).  empty is released by the issuer's
          tcgen05.commit multicast to both halves.  The expert id is read once
          per tile, before the first empty-wait.
          The two main faces are carried by TWO views of one buffer: the helper
          face (affine part as the layout, swizzle on the pointer) is what the
          MMA fragments are built from, and a `(rows, BLOCK_K, stage)` view of
          the same swizzled bytes is what the TMA descriptor is built from,
          because the descriptor builder checks the layout's TOP-LEVEL shape
          against the CTA V-map and the helper's MMA mode tree is not it.  That
          view is what `notes/probe_m0_s2_face.py` certified byte-exact on both
          halves of a pair.

  S3 (accumulator handover + epilogue)
    D0.11 accumulator double buffering: tile t accumulates into TMEM columns
          (t & 1) * 224, acc_full is one tcgen05.commit from the issuer at the
          end of the tile, acc_empty is counted across all 256 epilogue threads
          of the pair and is waited for at the TOP of the next-but-one tile.
          That is what puts the whole epilogue of tile t inside the MMA chain of
          tile t+1.
    D0.12 epilogue on warps 4-7 of BOTH CTAs: 14 segments per tile, each 16 m
          rows x the CTA's own 128 n columns.  Per segment: wait for the staging
          slot (one store still in flight), LDTM the 16 fp32 this thread owns
          out of TMEM, pack to bf16, transposing-stmatrix into the staging slot
          (the accumulator is a C-transpose block, so the transpose happens here
          rather than through 16 strided scalar stores), async-proxy fence, then
          one TMA store of the whole 16 x 128 segment.  The staging ring runs
          ACROSS the tile boundary, so the slot wait is unconditional.
          The read-back FRAME is the one `notes/probe_m0_s3_t2r7.py` certified
          (PROBE PASS max_diff=0.0, both axes, both halves of the pair): a plain
          rank-2 (128 lane, 16 column) TMEM tile, `layout_d = COL_MAJOR` in the
          MMA's own frame, and both tiled copies picked by MEASURING what they
          hand a thread rather than by name (see `_r2s_frame`).

  S4
    D0.1  one CuTe DSL kernel, one launch per call; the only other thing inside
          the window is the `torch.empty` that allocates c.  Every layout
          preparation is behind the D0.2 cache, so no call but the first of a
          given input identity does any of it.
    D0.2  host contract: the blocked scale planes and the block -> expert table
          are produced once per INPUT IDENTITY and held (`_prepare`), and the
          DSL entry is compiled once per launch GEOMETRY (`_COMPILED`) -- a call
          with different buffers is a new pointer value, never a recompile.
    D0.3  dataflow and stationarity: one 128(n) x 224(m) output block per CTA,
          K = 4096 streamed through smem at BLOCK_K = 128 while the fp32
          accumulator stays in TMEM for the whole K, output written exactly once
          (no split-K, no reduction), padding rows computed and stored like any
          other row.
    D0.14 issue order, wait-target relations and the segment live set: the three
          role loops below are the event order of the contract (producer reads
          the expert id before its first empty-wait; the issuer waits acc_empty
          at the TOP of the tile so its MMA chain runs under the previous tile's
          epilogue; the epilogue's per-segment wait is on the store that last
          read THIS slot), every k-block loop is `unroll=1` so no role expands
          32 k-block coordinates into registers at once, and the bf16 pack
          terminates the fp32 fragment's live range.

  After the M0 gate (this step)
    D0.7  the epilogue staging plane is moved onto the 128 B SWIZZLE the
          topology contract names as its starting point (`_epi_face`).  D0.7
          pins each buffer's ATOM and leaves the swizzle mode adjustable from
          M1 on; the gate row shipped this one plane unswizzled and measured
          the cost -- bankconf_st 16.67M inside smem_wavefronts_st 18.79M,
          against the reference's 3.9M inside 6.93M and a contract line of
          12.6M total.  Under a plain row-major plane the 16 rows of a segment
          are 256 B apart, so the eight rows one stmatrix writes land on the
          same four banks.  The swizzle also takes away the buffer's second
          view: the descriptor is now built from the staging face itself and
          the gmem side is presented in the same (n, m) MMA frame.  Certified
          end to end by `notes/probe_m1_epi_swizzle.py`.
    D1.1  wait ENCODING (this step), one constant: `WAIT_SPIN`.  All four of the
          kernel's in-loop pipeline waits -- producer on empty[s], issuer on
          full[s], issuer on acc_empty[b], epilogue on acc_full[b] -- move from
          the blocking face (compiler-generated retry around a try-wait carrying
          a suspend hint; the warp sleeps and is woken) to the non-blocking face
          (`mbarrier_try_wait` with the retry loop written out; the warp stays
          resident).  Nothing else moves: same barriers, same phases, same
          arrive/release counts, same order.  Both producer-side acquires are
          still CALLED with the satisfied token, because on the pair pipeline
          the acquire is also where the leader arms the transaction count.
    D1.4  the two tile-boundary exposures on the delivery side, plus the store
          wait's depth as a named knob.  (a) `STORE_WAIT_DEPTH` now carries the
          epilogue's per-segment `cp_async_bulk_wait_group(read=True)` depth; it
          ships at 1, the double-buffered starting point, with 0 (no staging
          double buffer) reachable as the lever's control arm.  (b) the per-tile
          expert-table LDG moves one tile AHEAD: tile 0's entry is read once in
          the producer prologue and tile t+1's at the k-loop TAIL of tile t,
          under the last stage's four TMAs, carried forward in a register -- so
          the top of a tile reaches its first empty-wait with no naked gmem read
          in front of it.  Coordinates for t+1 are computed there too, with the
          same tail-wave clamp the in-loop form used, so a masked iteration
          still delivers and the pipeline depths stay balanced.  (c) all five
          tensormaps (act, wgt, both scale planes, C) are prefetched from warp 0
          before the TMEM allocation handshake, so the first tile's deliveries
          do not pay a cold descriptor fetch and the handshake's cluster barrier
          covers the fetch latency.
    D1.2  where the barrier is first ASKED, one constant: `PEEK_AHEAD`.  The two
          in-loop pipeline waits that run 32x per tile -- producer on empty[s],
          issuer on full[s] -- move their probe from the point of use to the
          TAIL of the previous iteration, and the head of the next iteration
          consumes that token instead of reading the barrier again.  Each of the
          four call-sites is walked past the KB's two INDEPENDENT guards.
          Eligibility: the consumer probes sit inside `is_issuer` because full[s]
          is armed only on the leader, while the producer probes stay unguarded
          because empty[s] is arrived on both halves -- the guards point in
          opposite directions and reversing them is a deadlock, not a slowdown.
          Existence: each tail probe carries a flat "there is another wait"
          counter, false exactly once per kernel call, so no probe ever asks a
          phase that will not flip (`mbarrier.try_wait` bills that question at
          the hardware suspend timeout, not at one instruction).  The two
          once-per-tile acc waits are audited and left at the point of use: they
          are reached unsatisfied by construction, so a probe issued earlier
          only moves the suspend earlier.
    D1.3  stage count, one constant: `NSTAGE`.  The candidate domain {6, 5, 4}
          is measured end to end (6 -> 0.3756 ms, 5 -> 0.378 ms, 4 -> 0.400 ms)
          and docks at 6, the smem ceiling: the reading is monotone, so the
          starting point is the optimum and the smem_dynamic architecture line
          binds again at 202752 B.
    D0.11 the accumulator pipeline's commit is re-pointed at the handle
          `acquire_and_advance` hands back (`acc_producer.commit(mm_acc_tok)`).
          The bare `commit()` it replaces resolved against the participant's
          own state, which that same call had already advanced one stage, so
          acc_full was armed for the NEXT buffer: the epilogue of every even
          tile was released a whole tile late and the double buffer of D0.11
          was in effect a single one.  It is invisible to correctness and to
          the trace by construction and only shows as double-peaked, tile-
          alternating acc waits, which is what the timeline table reads
          (`mma.acc.begin` -> `mma.acc.end` 1765 cycles of wait against an
          18253-cycle issuer period [MEASURED, timeline @1c8c6eb8b4]).  The
          consumer side already released on its handle and both buffer indices
          already came from handles, so this is the single call-site that broke
          the invariant [hw_kb/cutedsl/sm100/pipeline_umma_async_handle_commit.md].

Two structural rules the file is built around, both earned from failures:

  * NOTHING but raw pointers crosses the host/trace boundary.  `cute.make_tensor`
    and `cute.make_layout` emit into the current MLIR context, and the host
    thread has none; calling them there fails inside the binding layer as
    `pack_shape(): incompatible function arguments ... Invoked with types:
    tuple` -- an argument error about the DEFAULTED context parameter, naming
    neither the context nor the call site.  The host builds `make_ptr` values
    (runtime objects, not IR); every layout is static geometry rebuilt inside
    the entry point.  This is also the shape D0.2 wants: one compile per
    geometry, and a later call with different buffers is a new pointer value.
  * DSL helpers are called through `_call_named`, never positionally and never
    on the strength of `inspect.signature`: this build wraps its entry points in
    `(*args, **kwargs)` wrappers that declare no parameters, so a signature-led
    binder passes an empty argument list and the callee reports every argument
    as missing.  That error message is the authority on this build's parameter
    names, and the binder is a fixed point over it.  What makes the inference
    safe is not the binder but `_shapes`' check of the four operand planes
    against the D0.7 byte contract.
"""

import importlib
import inspect
import re
import types

import torch

# Module scope on purpose.  `cute.compile` resolves the annotations of the entry
# point it is handed (`pC: cute.Pointer`) against that function's MODULE globals,
# so the DSL names have to be importable from there -- a `cute` bound only
# inside the builder raises NameError at compile time, well after tracing looks
# like it should have worked.  For the same reason there is no
# `from __future__ import annotations` here: annotations stay real objects.
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass.cute.nvgpu import cpasync, tcgen05

# --- shipped topology contract (D0.5 / D0.7 / D0.11 / D0.13) -----------------
# Values come from the milestone and are frozen at the M0 gate.
BLOCK_M = 224              # act rows per tile (= the grouping ALIGN, 7 x 32)
BLOCK_N = 128              # wgt rows per CTA; MMA M = 256 across the pair
BLOCK_K = 128              # 4 x (K=32) MMA per k-block
SF_VEC = 32                # scale factor vector length seen by the MMA
# D1.3 stage-count sweep, candidate domain {6, 5, 4}.  6 is the M0 starting
# point and the smem ceiling in one: 7 stages need 234796 B against the 232448 B
# per-CTA limit, so the domain only goes DOWN.  Supply granularity (BLOCK_K 128
# per stage) equals consumption granularity (one k-block eats a whole stage in
# four K=32 MMA), which is what makes the count a free lever rather than a shape
# change -- every byte count, barrier array, face layout and TMA view below is
# derived from this constant, so the three arms are the same tree with one
# number changed.  The domain is now fully measured:
#   6 -> device 0.3756 ms, tensor_pct 81.21, smem 202752 B [MEASURED, iter 12]
#   5 -> device 0.378  ms, tensor_pct 81.74, smem 170496 B [MEASURED, iter 13]
#   4 -> device 0.400  ms, smem 138240 B                   [MEASURED, iter 14]
# Monotone in the stage count and 6 wins, which is the decision's stated
# expectation: fewer stages = less TMA latency hidden ahead of the issuer, and
# the smaller concurrent footprint buys nothing here because the super-group
# rasterization already holds the working set far inside L2.  D1.3 therefore
# DOCKS at 6, the smem ceiling, and this constant does not move again.
NSTAGE = 6                 # operand stages (smem ceiling; D1.3 docked here)
GRID = 148                 # one wave over 148 SMs
CTA_THREADS = 256          # 8 warps
CLUSTER_M = 2              # cta_group::2 pair along M
SUPER_GROUP = 16           # rasterization super-group width (D2.2 lever)
TMEM_COLS = 512            # 2 x 224 accumulator columns + SF columns
ACC_COLS = 224             # one accumulator buffer (D0.11 gives it two)
# TMEM scale-factor columns sit above both accumulator buffers.  The reference
# puts its three tcgen05.cp destinations at columns 448 / 452 / 456
# [MEASURED, run1 section A8], i.e. 4 columns apart and starting right after
# 2 x 224; the same placement here keeps the total under the 512 ceiling.
TMEM_SFA_COL = CLUSTER_M * ACC_COLS
TMEM_SFB_COL = TMEM_SFA_COL + 4
ACC_BUFFERS = 2            # D0.11: tile t writes buffer t & 1

# D0.12 epilogue geometry.  A segment is 16 m rows x the whole 128-wide n block;
# 224 / 16 = 14 segments per tile, which is the reference's segment count.  The
# staging buffer holds two of them so the pack of segment k runs inside the TMA
# store of segment k-1.
EPI_THREADS = 128          # warps 4-7
EPI_SEG_M = 16
EPI_SEGS = BLOCK_M // EPI_SEG_M          # 14
EPI_SLOTS = 2
EPI_BAR_ID = 8             # named barrier, 128 threads wide

# D1.4(a) store-wait depth: how many bulk store-groups may still be in flight
# when the epilogue is allowed to overwrite the staging slot it is about to
# fill.  Depth 1 is the double-buffered starting point -- the pack of segment k
# runs under the store of segment k-1, and only the store that last read THIS
# slot has to have drained.  Depth 0 drains every store before the next pack and
# is the lever's control arm (no double buffer), kept reachable by one constant
# rather than by editing the call site.  The drain after the tile loop is a
# different call and stays at 0: it has to see an empty queue, not a shallow one.
STORE_WAIT_DEPTH = 1

# The TMEM allocator's retrieve barrier: CTA-wide (all 256 threads read the base
# address the allocating warp published), so it may not share EPI_BAR_ID, which
# is 128 wide and only the epilogue half arrives on.
TMEM_BAR_ID = 1

# D1.1 wait encoding.  The same mbarrier and the same phase parity have two
# encodings in this DSL, and they differ only in what the warp does while the
# barrier has not flipped:
#   False -- the blocking face (`mbarrier_wait`): a try-wait carrying a written-in
#            suspendTimeHint, the retry loop generated by the compiler, the warp
#            suspended and woken when the barrier flips.  Every wait pays one
#            wake-up latency, and that latency appears in no throughput counter.
#   True  -- the non-blocking face (`mbarrier_try_wait`) with the retry loop
#            written here: the warp stays resident and re-reads the barrier.
# The blocking face is the M0 starting point (it is the reference's shape).  The
# spin face is not free -- it holds an issue slot and pays one smem barrier read
# per retry -- so which one wins is a measurement, not a derivation
# [hw_kb/cutedsl/sm100/mbarrier_wait_encoding.md gotcha 6].  This knob is what makes the
# two arms the same tree with one constant changed.
WAIT_SPIN = True

# D1.2 consumer/producer PEEK placement.  Orthogonal to WAIT_SPIN: that constant
# picks how a wait is encoded, this one picks WHEN the barrier is first asked.
#   False -- ask at the point of use: the first `try_*` of a wait is the first
#            instruction of the wait itself.
#   True  -- ask one unit of work early: the probe for wait i+1 is issued at the
#            TAIL of iteration i (the KB's fused-single-loop shape), and the head
#            of iteration i+1 consumes that token, so a wait that is already
#            satisfied -- which is the steady state on the consumer side, the
#            producer runs up to NSTAGE ahead -- costs no barrier read at all.
# A peek is only sound where BOTH of the KB's guards hold, and they are
# independent [hw_kb/cutedsl/sm100/cta_pair_consumer_peek.md]:
#   (1) ELIGIBILITY -- on a cta_group::2 pair the full barrier is armed only on
#       the leader, so every consumer-side call (the non-blocking probe included)
#       must sit inside the leader predicate.  The producer side is the exact
#       opposite: `try_acquire` must run on BOTH halves or the follower stops
#       delivering and the leader's barrier never fills.
#   (2) EXISTENCE -- a peek exists to hand a token to the NEXT wait.  Where there
#       is no next wait, the probe asks a phase that never flips, and
#       `mbarrier.try_wait` is potentially blocking: that question is billed at
#       the hardware's suspend timeout, not at one instruction.  So the tail
#       probe carries an explicit "there is another wait" predicate, which is
#       false exactly once per kernel call, at the last k-block of the last tile.
# Peeking requires the spin encoding (the token hand-back is what consumes it),
# so this constant is read as `PEEK_AHEAD and WAIT_SPIN`.
PEEK_AHEAD = True

# D0.7 per-stage bytes; declared here because the launch has to reserve the
# dynamic smem before the epilogue staging buffer itself lands (S3).
ACT_STAGE_BYTES = 14336    # 112 rows x 128 B e4m3 (half of 224 per CTA)
WGT_STAGE_BYTES = 16384    # 128 rows x 128 B Uint8 container face (D0.8)
SFA_STAGE_BYTES = 1024     # blocked sf_act, 224 rows x 4 B
SFB_STAGE_BYTES = 512      # blocked sf_wgt, 128 rows x 4 B
EPI_STAGE_BYTES = 8192     # epilogue staging, 2 x 4096 B
SMEM_TAIL_BYTES = 1024     # mbarriers + TMEM base slot
SMEM_BYTES = (
    EPI_STAGE_BYTES
    + NSTAGE * (ACT_STAGE_BYTES + WGT_STAGE_BYTES + SFA_STAGE_BYTES + SFB_STAGE_BYTES)
    + SMEM_TAIL_BYTES
)                          # 8192 + NSTAGE*32256 + 1024, under the 232448 B
                           # per-CTA ceiling: 202752 B at 6 stages, 170496 B at
                           # 5, 138240 B at 4.  This is what the launch reserves
                           # and what ncu reads back as smem_dynamic.

# D0.8: the weights are DELIVERED packed (two e2m1 per byte) into a container
# plane that is twice as wide.  expect-tx has to count the packed bytes; the
# container's 16384 leaves the barrier one half short forever (forbidden zone).
WGT_PACKED_STAGE_BYTES = WGT_STAGE_BYTES // 2

# Mount levels for the M0 build order.  The shipped path is always the highest
# one; the lower ones exist so a probe can put the run-time-only mechanisms on
# the machine ONE AT A TIME and smoke each -- "trace-invisible, explodes on
# launch" mechanisms must not first appear together.  A ladder only localizes a
# fault if consecutive rungs differ by ONE mechanism, so the operand faces, the
# pipeline object and the delivery are three rungs, not one.
MOUNT_SKEL = 1             # S1: launch + roles + TMEM handshake + tile loop
MOUNT_PIPE = 2             # + operand faces, smem allocation, pipeline object
MOUNT_TMA = 3              # + tma_partition and the four deliveries per stage
MOUNT_MMA = 4              # + tcgen05.cp scale upload and the 4 MMA per k-block
MOUNT_ACC = 5              # + acc double buffer and the acc_full/acc_empty pair
MOUNT_EPI = 6              # + LDTM -> F2FP -> transposing stmatrix into staging
MOUNT_STORE = 7            # + the TMA store of the staging slot to C
MOUNT_SHIPPED = MOUNT_STORE


def _seg(site, row=0):
    """Timeline-probe hook: a no-op in the shipped kernel; the probe arm of the
    driver rewrites the body to stamp a timestamp for `site` at row `row`."""
    return None


# ---------------------------------------------------------------------------
# Build-time binding helpers.
#
# Several Blackwell entry points below have moved parameter names and argument
# order across cutlass-dsl builds, and the failure mode of guessing is a
# positional call that binds silently to the wrong slot.  Binding BY NAME at
# build time (plain Python, before any tracing) turns that into a loud error.
# ---------------------------------------------------------------------------

def _blackwell_helpers():
    """`blackwell_helpers` merged with `blockscaled_layout`: the scale-factor
    layout builders have lived in either module depending on the build."""
    mods = []
    for name in ("cutlass.utils.blackwell_helpers",
                 "cutlass.utils.blockscaled_layout",
                 "cutlass.utils"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    if not any(hasattr(m, "make_smem_layout_a") for m in mods):
        raise RuntimeError("no cutlass Blackwell helper module exports make_smem_layout_a")

    class _Merged:
        __name__ = "+".join(m.__name__ for m in mods)

        def __getattr__(self, item):
            for m in mods:
                fn = getattr(m, item, None)
                if fn is not None:
                    return fn
            raise AttributeError(item)

    return _Merged()


def _resolve(name, *modules):
    for m in modules:
        fn = getattr(m, name, None)
        if fn is not None:
            return fn
    raise RuntimeError(f"{name!r} not found in any of the offered modules")


# ---------------------------------------------------------------------------
# `cp.async.bulk` group commit / wait for the TMA store.
#
# `cpasync.cp_async_bulk_commit_group` does not exist on this image, and the two
# calls are not optional: with no commit the store group never closes, so the
# wait can never retire and the epilogue deadlocks; with no wait the staging
# slot is recycled under a store still reading it.  So (a) search by name across
# every module that has carried them, and (b) if no module has, emit the PTX
# directly -- these are the no-operand, side-effect-only shape of the inline-asm
# escape, which has no address and therefore no swizzle exposure at all.
#
# Every candidate name below carries `bulk` or `tma_store` on purpose.  The
# plain `cp_async_commit_group` / `cp_async_wait_group` are a DIFFERENT group
# (the LDGSTS one); binding to those would compile, launch, and wait on a group
# the TMA stores never join.
_BULK_COMMIT_NAMES = ("cp_async_bulk_commit_group", "cpasync_bulk_commit_group",
                      "async_bulk_commit_group", "bulk_commit_group",
                      "cp_async_bulk_commit", "tma_store_commit_group")
_BULK_WAIT_NAMES = ("cp_async_bulk_wait_group", "cpasync_bulk_wait_group",
                    "async_bulk_wait_group", "bulk_wait_group",
                    "tma_store_wait_group")

# Filled in by `_bulk_group_ops`; read by the smoke probe so the trajectory
# records which of the two paths the image actually took.
_BULK_BIND = {"commit": "unbound", "wait": "unbound"}


def _bulk_modules():
    mods = [cute.arch, cpasync, cute]
    for name in ("cutlass.cute.nvgpu.cpasync.helpers",
                 "cutlass.cute.arch",
                 "cutlass.cute.nvgpu.common"):
        try:
            mods.append(importlib.import_module(name))
        except ImportError:
            pass
    return mods


def _dsl_user_op():
    """The decorator that lets a plain python function emit IR inside a trace.
    Its home module has moved across DSL builds, so search rather than import.
    """
    for mod in ("cutlass.cutlass_dsl", "cutlass.base_dsl.dsl", "cutlass"):
        try:
            fn = getattr(importlib.import_module(mod), "dsl_user_op", None)
        except ImportError:
            fn = None
        if fn is not None:
            return fn
    return None


def _asm_emitter():
    """`(text, loc, ip) -> None`: the no-operand, side-effect-only inline-asm
    shape.  No address crosses it, so no swizzle exposure either."""
    llvm = importlib.import_module("cutlass._mlir.dialects.llvm")

    def _asm(text, loc, ip):
        llvm.inline_asm(None, [], text, "", has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    return _asm


def _ptx_bulk_group_ops():
    """The PTX escape: `cp.async.bulk.commit_group` / `.wait_group[.read] N`."""
    dsl_user_op = _dsl_user_op()
    if dsl_user_op is None:
        raise RuntimeError(
            "no cp.async.bulk group commit/wait in %s and no dsl_user_op to "
            "emit the PTX with; the TMA store epilogue cannot close its store "
            "group at all" % [getattr(m, "__name__", m)
                              for m in _bulk_modules()])
    _asm = _asm_emitter()

    @dsl_user_op
    def commit(*, loc=None, ip=None):
        _asm("cp.async.bulk.commit_group;", loc, ip)

    @dsl_user_op
    def wait(n, read=True, *, loc=None, ip=None):
        # `n` is the group count and is an immediate in the ISA, so it must be a
        # build-time python int -- which every call site here passes.
        _asm("cp.async.bulk.wait_group%s %d;" % (".read" if read else "",
                                                 int(n)), loc, ip)

    return commit, wait


def _bulk_group_ops():
    """(commit, wait) with `wait(n, read=...)` as the calling convention."""
    found = {}
    for key, names in (("commit", _BULK_COMMIT_NAMES),
                       ("wait", _BULK_WAIT_NAMES)):
        for name in names:                     # name first, module second: the
            for m in _bulk_modules():          # canonical spelling wins over a
                fn = getattr(m, name, None)    # near-miss in an earlier module
                if fn is not None:
                    found[key] = fn
                    _BULK_BIND[key] = "%s.%s" % (
                        getattr(m, "__name__", str(m)), name)
                    break
            if key in found:
                break

    if "commit" not in found or "wait" not in found:
        # Both or neither: a DSL commit paired with a PTX wait would be fine,
        # but mixing is one more thing to have been wrong about for no gain.
        commit, wait = _ptx_bulk_group_ops()
        _BULK_BIND["commit"] = _BULK_BIND["wait"] = "ptx:cp.async.bulk"
        return commit, wait

    commit, wait = found["commit"], found["wait"]
    if not _accepts(wait, "read"):
        # No `.read` modifier available: the plain wait also drains the writes,
        # which is stronger than the slot recycle needs.  Slower, never wrong.
        _raw = wait
        _BULK_BIND["wait"] += "(no-read)"

        def wait(n, read=True, _raw=_raw):
            return _raw(n)

    return commit, wait


# ---------------------------------------------------------------------------
# The generic-proxy -> async-proxy fence for the TMA store.
#
# The epilogue writes the staging slot with stmatrix (the GENERIC proxy) and the
# TMA store reads it through the ASYNC proxy.  Without a fence between them the
# store may read the slot's previous contents: stale output, not an error, and
# no correctness probe that runs the two in the same order can tell the
# difference.  `cute.arch.fence_proxy` + `ProxyKind` is not on this image (the
# S3 smoke died in exactly that build-time guard), so bind the same way the bulk
# group ops are bound: search names, then emit the PTX.
#
# `fence.proxy.async.shared::cta` is the CTA-scoped member of the family; the
# stmatrix writes and the store that reads them are both in this CTA, so the
# wider `fence.proxy.async` (device scope) is only slower, never more correct.
_PROXY_FENCE_NAMES = ("fence_view_async_shared", "fence_proxy_async_shared_cta",
                      "fence_proxy_async_shared", "fence_proxy_async",
                      "fence_async_proxy", "fence_view_async_shared_cta")
_PROXY_KIND_NAMES = ("async_shared", "async_", "async", "ASYNC_SHARED", "ASYNC")

# Read by the smoke probe, same as `_BULK_BIND`: which spelling this image took
# is otherwise invisible in the trajectory.
_PROXY_BIND = {"fence": "unbound"}


def _ptx_async_proxy_fence():
    dsl_user_op = _dsl_user_op()
    if dsl_user_op is None:
        raise RuntimeError(
            "no async-proxy fence in %s and no dsl_user_op to emit the PTX "
            "with; the stmatrix writes into the staging slot would be "
            "invisible to the TMA store's async proxy"
            % [getattr(m, "__name__", m) for m in _bulk_modules()])
    _asm = _asm_emitter()

    @dsl_user_op
    def fence(*, loc=None, ip=None):
        _asm("fence.proxy.async.shared::cta;", loc, ip)

    return fence


def _needs_args(fn):
    """Whether introspection can SEE a required parameter of `fn`.

    A blind signature answers False -- the DSL's wrappers are varargs-only and
    "cannot see it" is not evidence that one exists.  Used to reject a
    same-named entry point that is not the no-operand fence we want.
    """
    for cand in _unwrap_candidates(fn):
        try:
            params = inspect.signature(cand).parameters
        except (TypeError, ValueError):
            continue
        required = [p for p, v in params.items()
                    if v.default is inspect.Parameter.empty
                    and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
                    and p not in ("self", "cls", "loc", "ip")]
        if params:
            return bool(required)
    return False


def _async_proxy_fence():
    """A zero-argument callable that fences generic -> async proxy."""
    for name in _PROXY_FENCE_NAMES:        # name first, module second, as in
        for m in _bulk_modules():          # `_bulk_group_ops`
            fn = getattr(m, name, None)
            if fn is None or _needs_args(fn):
                continue
            _PROXY_BIND["fence"] = "%s.%s" % (
                getattr(m, "__name__", str(m)), name)
            return fn

    # The kind-taking spelling, if this image has it.
    raw = getattr(cute.arch, "fence_proxy", None)
    kinds = getattr(cute.arch, "ProxyKind", None)
    kind = None
    if raw is not None and kinds is not None:
        for n in _PROXY_KIND_NAMES:
            kind = getattr(kinds, n, None)
            if kind is not None:
                break
    if kind is not None:
        space = getattr(getattr(cute.arch, "SharedSpace", None), "shared_cta",
                        None)
        if space is not None and _accepts(raw, "space"):
            _PROXY_BIND["fence"] = "cute.arch.fence_proxy(kind, space=)"

            def fence(raw=raw, kind=kind, space=space):
                return raw(kind, space=space)
        else:
            _PROXY_BIND["fence"] = "cute.arch.fence_proxy(kind)"

            def fence(raw=raw, kind=kind):
                return raw(kind)

        return fence

    fence = _ptx_async_proxy_fence()
    _PROXY_BIND["fence"] = "ptx:fence.proxy.async.shared::cta"
    return fence

# signature is `(*args, **kwargs)`.  `inspect.signature` then reports NO named
# parameters at all, and a binder that trusts it passes NOTHING -- which the
# callee reports as every argument being missing, from inside a trace, with the
# call site buried under library frames.  So introspection is only the first
# guess here; the authority is the callee's own TypeError, which names the
# parameters it actually wants.  Binding is a fixed point over that message.
_MISSING_PAT = re.compile(
    r"missing\s+(?:\d+\s+)?required\s+(?:positional\s+|keyword-only\s+)?"
    r"argument(?:\(s\)|s)?\s*:?\s*(.+)", re.I)
_UNEXPECTED_PAT = re.compile(r"unexpected keyword argument '([^']+)'")


def _ident_list(text):
    out = []
    for chunk in re.split(r",| and ", text):
        name = chunk.strip().strip("'\"` \t").rstrip(".")
        if re.fullmatch(r"[A-Za-z_]\w*", name):
            out.append(name)
    return out


def _unwrap_candidates(fn, depth=3):
    """`fn` plus every plain function reachable from it through `__wrapped__`,
    `__func__` or its closure cells, nearest first.

    The DSL decorates its entry points with `(*args, **kwargs)` wrappers, so
    `inspect.signature(fn)` describes the WRAPPER and reports no named
    parameters at all.  The decorated function is still reachable from the
    wrapper's closure, and it is the only thing that knows the OPTIONAL
    parameter names.  That distinction is the whole reason this exists: a
    REQUIRED parameter nobody passes is reported loudly by the callee, while an
    optional one nobody passes silently takes its default -- and
    `cta_layout_vmnk` defaulting to None turns a cta_group::2 pipeline into a
    cta_group::1 pipeline that arms the pair's byte count and then waits
    forever (KB: cta_pair_bulk_load_cta_group2, "现成流水缺 cta_layout_vmnk 时
    静默退化成 ONE").
    """
    out, seen, queue = [], set(), [(fn, 0)]
    while queue:
        f, d = queue.pop(0)
        if id(f) in seen or d > depth:
            continue
        seen.add(id(f))
        out.append(f)
        nxt = [g for g in (getattr(f, "__wrapped__", None),
                           getattr(f, "__func__", None)) if g is not None]
        for cell in (getattr(f, "__closure__", None) or ()):
            try:
                v = cell.cell_contents
            except ValueError:                   # an empty cell (recursive def)
                continue
            if inspect.isfunction(v) or inspect.ismethod(v):
                nxt.append(v)
        for g in nxt:
            queue.append((g, d + 1))
    # A wrapper copies `__name__` from what it wraps, so a same-named candidate
    # is the decorated function and a differently-named one is some unrelated
    # helper that happened to be in the closure.
    want = getattr(fn, "__name__", None)
    out.sort(key=lambda f: 0 if getattr(f, "__name__", None) == want else 1)
    return out


def _declared(fn):
    """(named parameters, accepts **kwargs) as far as introspection can tell.

    Returns `(None, True)` only when NO candidate in the unwrap chain declares a
    named parameter -- i.e. when introspection has genuinely told us nothing.
    """
    for cand in _unwrap_candidates(fn):
        try:
            params = inspect.signature(cand).parameters
        except (TypeError, ValueError):
            continue
        named = [p for p, v in params.items()
                 if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)
                 and p not in ("self", "cls")]
        if named:
            return named, any(v.kind == v.VAR_KEYWORD for v in params.values())
    return None, True


def _call_named(fn, _widen=True, _require=(), **pool):
    """Call `fn`, filling each of its parameters from `pool` BY PARAMETER NAME.

    `pool` may carry several spellings of the same value; only the names the
    callee actually wants get passed.  A required parameter with no entry in the
    pool raises here, naming it, rather than shifting every later positional
    argument -- and rather than silently calling with an empty argument list.

    When introspection is blind, satisfying only the parameters the error names
    is NOT enough: those are the REQUIRED ones, and a call that omits an
    OPTIONAL one silently takes its default -- `num_stages` defaulting to 1
    where the topology contract says 6 is a miscompile that nothing downstream
    reports.  So a blind bind is followed by a widening pass that offers each
    remaining pool entry and keeps the ones the callee accepts.  `_widen=False`
    turns that off for callees that EMIT IR, which must be called exactly once;
    those callees instead name what they may not be bound without in
    `_require`, which is checked against the kwargs that were actually passed.
    """
    label = getattr(fn, "__name__", repr(fn))
    named, _var_kw = _declared(fn)
    kwargs = {p: pool[p] for p in (named or ()) if p in pool}

    seen = []
    result, bound = None, False
    for _ in range(8):
        try:
            result, bound = fn(**kwargs), True
            break
        except TypeError as exc:
            msg = str(exc)
            seen.append(msg.splitlines()[0][:200])
            m = _MISSING_PAT.search(msg)
            wanted = _ident_list(m.group(1)) if m else []
            unexpected = _UNEXPECTED_PAT.findall(msg)
            filled = [p for p in wanted if p not in kwargs and p in pool]
            dropped = [p for p in unexpected if p in kwargs]
            if not filled and not dropped:
                # Either the TypeError comes from inside the callee (nothing to
                # rebind) or it names a parameter nobody offered.  Both are for
                # the caller to fix, and both need the name to be readable.
                named_in_error = wanted or ["<the error names no parameter>"]
                not_offered = [p for p in wanted if p not in pool]
                raise RuntimeError(
                    "%s: cannot bind arguments. wants=%s not-offered=%s "
                    "passed=%s offered=%s errors=%s"
                    % (label, named_in_error, not_offered, sorted(kwargs),
                       sorted(pool), seen)) from exc
            for p in filled:
                kwargs[p] = pool[p]
            for p in dropped:
                del kwargs[p]
    if not bound:
        raise RuntimeError(
            "%s: argument binding did not converge; errors=%s" % (label, seen))

    if _widen and named is None:
        for p, v in pool.items():
            if p in kwargs:
                continue
            try:
                wider = fn(**dict(kwargs, **{p: v}))
            except BaseException:                         # noqa: BLE001
                continue                     # the callee has no such parameter
            kwargs[p], result = v, wider

    # The bookkeeping above says what was OFFERED; this says what was PASSED.
    # A silently-defaulted optional is the failure mode the whole binder exists
    # to prevent, and it is invisible in the result object, so the call sites
    # that have a topology contract riding on an optional name it here.
    unbound = [p for p in _require if p not in kwargs]
    if unbound:
        raise RuntimeError(
            "%s: bound WITHOUT %s -- those carry a topology contract and their "
            "defaults are silent. declared=%s passed=%s offered=%s"
            % (label, unbound, named, sorted(kwargs), sorted(pool)))
    _BOUND[label] = sorted(kwargs)
    return result


# What `_call_named` actually passed, per callee.  Read by the build-time probe:
# "which parameters reached the callee" is otherwise unobservable.
_BOUND: dict = {}


def _accepts(fn, pname):
    """Whether `fn` takes a parameter called `pname`.

    Unreadable or varargs-only signatures answer True: "introspection cannot
    see it" is not evidence of absence, and the alternative is refusing to build
    a call that would in fact have worked.
    """
    named, var_kw = _declared(fn)
    if named is None:
        return True
    return pname in named or var_kw


def _drop_stage(layout_staged):
    """Drop the trailing stage mode of a staged smem layout.

    The arity has to come from the layout: slicing `make_smem_layout_a`'s 4-mode
    result with a 3-element coordinate is the documented "weakly congruent"
    rejection (forbidden zone), not a spelling variant.
    """
    rank = cute.rank(layout_staged)
    return cute.slice_(layout_staged, (None,) * (rank - 1) + (0,))


def _group_but_stage(t):
    """Collapse every mode of a staged smem tensor except the stage mode, which
    is the shape `tma_partition` wants on the destination side."""
    return cute.group_modes(t, 0, cute.rank(t.layout) - 1)


def _stage_slice(t, stage):
    """Select one stage of a tensor whose trailing mode is the stage mode.

    The arity comes from the tensor, for the same reason `_drop_stage` takes it
    from the layout.  A `tma_partition` result is rank 2 and takes `(None,
    stage)`; an s2t partition of a blocked SF face is rank 5 and the same
    two-element coordinate is the "expects `coord` and shape of view are weakly
    congruent" rejection.
    """
    rank = cute.rank(t.layout)
    return t[(None,) * (rank - 1) + (stage,)]


# --- composed smem layouts: affine part vs swizzle ---------------------------
# `make_smem_layout_a/b` return a COMPOSED layout (`S<3,4,3> o 0 o (...)`).  A
# composed layout may not be a tensor's layout: `make_fragment_A/B` rejects it
# as "Expected affine layout for operand A, but got composed layout".  The
# certified form splits it at allocation time -- the affine part is the layout,
# the swizzle rides on the POINTER -- which is the same bytes and the same
# addresses, just with the swizzle moved to where the MMA descriptor and the TMA
# descriptor both expect to find it.

def _affine_of(layout):
    return getattr(layout, "outer", layout)


def _swizzle_of(layout):
    return getattr(layout, "inner", None)


# --- the tiler-shaped second view of an operand face (D0.7 / D0.8) -----------
# The generic TMA builder checks the smem layout's TOP-LEVEL shape against the
# CTA V-map, and what it wants there is `(rows, BLOCK_K[, stage])`.  The helper
# face is the same bytes under the same 128 B swizzle but carries the MMA's mode
# tree (`((rows, 32), 1, 4)`), whose top-level shape is a different thing; the
# builder rejects it.  So the face is described to the TMA through a second view
# with the mode tree the builder wants.
#
# Both address functions are identical -- view coordinate (r, k) lands at
# r*128 + k, and helper coordinate (r, k % 32) of sub-block k // 32 lands at
# r*128 + (k % 32) + (k // 32)*32 -- which is why one buffer can carry both.
# `_shapes` checks that (byte count, and that the two views carry the SAME
# swizzle) rather than trusting the arithmetic.
#
# Certified end to end by `notes/probe_m0_s2_face.py`: the pair delivers through
# this view under a leader-armed pair transaction and every byte of both CTAs'
# act half-tiles reads back correct (PROBE PASS max_diff=0.0).

def _smem_atom_kind(names=("K_SW128", "K_SW128_LINEAR", "SW128_K")):
    """(`make_smem_layout_atom`, the first named 128 B atom kind this build has).

    The default names are the K-major family the operand faces want; the
    epilogue staging plane asks for the MN-major spelling of the same 128 B
    swizzle, because its contiguous mode is the MMA's M.
    """
    mk = kinds = None
    for name in ("cutlass.cute.nvgpu.warpgroup", "cutlass.cute.nvgpu.warp",
                 "cutlass.cute.nvgpu", "cutlass.utils.blackwell_helpers"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        mk = mk or getattr(mod, "make_smem_layout_atom", None)
        kinds = kinds or getattr(mod, "SmemLayoutAtomKind", None)
    if mk is None or kinds is None:
        return None, None
    for kn in names:
        kind = getattr(kinds, kn, None)
        if kind is not None:
            return mk, kind
    return None, None


def _face_view(dtype, rows, nstage, helper):
    """`(rows, BLOCK_K[, nstage])`-shaped view of one operand face.

    Two constructions, in the order the probe tried them.  `atomkind` asks the
    library for its own K-major 128 B atom and tiles it to shape, so it cannot
    disagree with the library about what SW128 means; `remode` lifts the swizzle
    off the helper layout itself, which cannot disagree with the helper by
    construction but depends on that layout exposing its swizzle under a name
    this build uses.  MUST be called from inside a trace.
    """
    shape = (rows, BLOCK_K) if nstage is None else (rows, BLOCK_K, nstage)
    mk, kind = _smem_atom_kind()
    if mk is not None:
        return cute.tile_to_shape(mk(kind, dtype), shape,
                                  order=tuple(range(len(shape))))
    swz = _swizzle_of(helper)
    if swz is None:
        raise RuntimeError(
            "no make_smem_layout_atom/SmemLayoutAtomKind in any nvgpu module, "
            "and the helper face exposes no swizzle to re-compose either")
    stride = ((BLOCK_K, 1) if nstage is None
              else (BLOCK_K, 1, rows * BLOCK_K))
    flat = cute.make_layout(shape, stride=stride)
    for fn in ("make_composed_layout", "composition", "make_swizzle_layout"):
        f = getattr(cute, fn, None)
        if f is None:
            continue
        for args in ((swz, 0, flat), (swz, flat)):
            try:
                return f(*args)
            except Exception:                              # noqa: BLE001
                pass
    raise RuntimeError("no cute composer accepted (swizzle[, offset], layout)")


# --- blocked scale-factor addressing (D0.9) ---------------------------------
# The tcgen05.cp source plane is CUTLASS's blocked SF atom: a 128-row,
# 4-K-block chunk of 512 bytes in which the byte for (row r, K=32 block j) sits
# at (r % 32) * 16 + ((r // 32) % 4) * 4 + j, with whole 128-row chunks 512
# bytes apart.  The host repack writes the bytes straight at those offsets, so
# the TMA feeding this plane is a flat contiguous copy and no SF tile ever has
# to be a box of the plain (M, K/32) scale tensor -- which it could not be,
# since an M block starts at a multiple of 224 while the atom is 128 rows tall.

def _sf_offsets(rows, k_sub):
    """(row, K=32 sub-block, byte offset) for every byte of one blocked SF tile."""
    return [(r, j,
             (r % 32) * 16 + ((r // 32) % 4) * 4 + j + (r // 128) * 512)
            for r in range(rows) for j in range(k_sub)]


def _sf_tile_bytes(rows, k_sub):
    """Size of one blocked SF tile.  Rounded up to whole 128-row atoms: a
    224-row tile is two atoms (1024 B) even though its last atom is only
    three quarters used, which is exactly the 1024 B sf_act slot the topology
    contract asks for."""
    return ((rows + 127) // 128) * 128 * k_sub


# --- the epilogue read-back frame (D0.12), as certified ----------------------
# `notes/probe_m0_s3_t2r7.py` drove this chain -- live pair-mode TMEM -> LDTM ->
# pack -> transposing stmatrix -> staging -> read back -- with each cell's own
# absolute coordinate as the payload, and got PROBE PASS max_diff=0.0 on both
# axes and on both halves of the pair.  Four things in it are NOT interchangeable
# with the spelling one would write first, and each cost a probe round:
#
#   1. the TMEM segment is a PLAIN RANK-2 tile, (128 lanes, 16 columns) built
#      straight on the TMEM pointer -- not a `zipped_divide` tile of the
#      accumulator layout.  The divide's nested mode tree partitions into a
#      register frame the store side then will not accept.
#   2. `layout_d` is COL_MAJOR, read in the MMA's frame: mode 0 of the epilogue
#      tile is the MMA's M (= the TMEM lane = n of C) and mode 1 is the MMA's N
#      (= the TMEM column = m of C).  The staging slot reaches its row-major
#      [m][n] face by being COLUMN-major here -- same bytes, same addresses, and
#      that identity is what the TMA store on the far side reads through.
#   3. the register -> staging tiled copy is built by `make_tiled_copy_D`.
#      `make_tiled_copy_S` inherits the reference copy's SOURCE TV layout, which
#      for a TMEM copy is the tcgen05 datapath face: it hands a thread 512
#      elements for a 16-element fragment and the copy does not even trace.
#   4. the register-side spelling handed to `cute.copy` (`tiled.retile`,
#      `thr.retile`, or the fragment as-is) is likewise not decidable by name:
#      one of them matches the total size and mis-matches the atom's value mode.
#
# So 3 and 4 are decided the way the probe decided them: build every candidate,
# measure what it hands a thread, take the one that matches the fragment, and
# fail loudly if none does.  All of it is trace-time python -- these are plain
# functions called FROM a staged body, which is why they may raise at all.

def _flat(x, depth=0):
    """A (possibly nested) cute shape or stride as a flat list of leaves.

    The nesting comes out of the library's own constructors and is not knowable
    in advance, so it is walked rather than indexed.
    """
    if depth > 8:
        return [x]
    if isinstance(x, (tuple, list)):
        out = []
        for e in x:
            out.extend(_flat(e, depth + 1))
        return out
    try:
        parts = list(x)
    except Exception:                                      # noqa: BLE001
        return [x]
    if len(parts) == 1 and parts[0] is x:
        return [x]
    out = []
    for e in parts:
        out.extend(_flat(e, depth + 1))
    return out


def _layout_shape_stride(layout):
    """(flat shape, flat stride) of a layout, as python ints.

    Both the attribute and the free-function spellings are tried: getting this
    wrong mis-strides the TMEM tile silently rather than loudly.
    """
    out = []
    for attr, fn in (("shape", getattr(cute, "shape", None)),
                     ("stride", getattr(cute, "stride", None))):
        got = None
        for take in (lambda: getattr(layout, attr),
                     lambda: fn(layout) if fn is not None else None):
            try:
                got = take()
            except Exception:                              # noqa: BLE001
                got = None
            if got is not None:
                break
        if got is None:
            raise RuntimeError(f"cannot read {attr} off {layout!r}")
        vals = []
        for v in _flat(got):
            try:
                vals.append(int(v))
            except Exception:                              # noqa: BLE001
                raise RuntimeError(
                    f"{attr} of {layout!r} is not static: {v!r}")
        out.append(vals)
    return out[0], out[1]


def _top_modes(layout, attr):
    """The TOP-LEVEL modes of a layout's shape or stride, as a tuple.

    `_layout_shape_stride` flattens; appending a mode needs the nesting kept,
    because mode 0 of an accumulator fragment is itself a pair.  A rank-1
    layout whose attribute does not iterate is wrapped rather than expanded.
    """
    got = getattr(layout, attr)
    try:
        return tuple(got)
    except Exception:                                      # noqa: BLE001
        return (got,)


def _size_of(t, mode=None):
    """`cute.size` of a tensor, or of one of its top-level modes, as an int.

    Mode 0 is the atom's value mode, which is the half of the shape the IR
    verifier compares first; None means the number could not be read.
    """
    try:
        return int(cute.size(t) if mode is None else cute.size(t, mode=[mode]))
    except Exception:                                      # noqa: BLE001
        return None


def _epi_layout_d():
    """The major order both epilogue op selectors are keyed on (COL_MAJOR)."""
    enum = _resolve("LayoutEnum", cutlass.utils, _blackwell_helpers(), cute,
                    cutlass)
    layout_d = getattr(enum, "COL_MAJOR", None)
    if layout_d is None:
        raise RuntimeError(
            "LayoutEnum has no COL_MAJOR; members="
            + str([m for m in dir(enum) if m[:1].isupper()][:12]))
    return layout_d


def _epi_face(sm100, layout_d):
    """The D0.12 staging plane as a SWIZZLED, MMA-frame, 3-mode face.

    Modes are (MMA M = n of C, MMA N = m of C, slot) -- the frame both epilogue
    op selectors and both partitions are keyed on -- and the plane carries the
    128 B swizzle D0.7 names as this buffer's starting point.  The swizzle is
    the whole point: under a plain row-major plane the 16 rows of a segment are
    256 B apart, so the eight rows of one stmatrix group land on the same four
    banks and every staging write is an 8-way conflict by construction (the M0
    gate read bankconf_st 16.67M of 18.79M store wavefronts, against the
    reference's 3.9M of 6.93M).

    `make_smem_layout_epi` is the library's own builder for this plane; the
    hand-built fallback tiles the library's MN-major 128 B atom to the same
    shape, so it cannot disagree with the library about what SW128 means.

    Certified end to end by `notes/probe_m1_epi_swizzle.py` (PROBE PASS
    max_diff=0.0): the face is accepted by `make_tiled_tma_atom` against an
    MMA-frame C, and live pair-mode TMEM -> t2r -> pack -> transposing stmatrix
    -> swizzled slot -> TMA store puts every cell where the frame says, with
    each cell's own absolute coordinate as the payload (a chunk permutation
    from a swizzle the two sides disagree on is read off, not merely detected).
    MUST be called from inside a trace.
    """
    epi_tile = (BLOCK_N, EPI_SEG_M)
    fn = getattr(sm100, "make_smem_layout_epi", None)
    if fn is not None:
        return _call_named(
            fn,
            element_type=cutlass.BFloat16, elem_ty=cutlass.BFloat16,
            elem_ty_d=cutlass.BFloat16, element_type_d=cutlass.BFloat16,
            dtype=cutlass.BFloat16, dtype_d=cutlass.BFloat16,
            d_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16,
            epi_dtype=cutlass.BFloat16, epi_elem_ty=cutlass.BFloat16,
            epi_element_type=cutlass.BFloat16,
            layout=layout_d, layout_d=layout_d, layout_D=layout_d,
            d_layout=layout_d, c_layout=layout_d, epi_layout=layout_d,
            epi_tile=epi_tile, epilogue_tile=epi_tile, epi_tiler=epi_tile,
            tile=epi_tile, tile_shape=epi_tile,
            stage=EPI_SLOTS, stages=EPI_SLOTS, num_stages=EPI_SLOTS,
            epi_stage=EPI_SLOTS, num_epi_stage=EPI_SLOTS)
    mk, kind = _smem_atom_kind(("MN_SW128", "MN_SW128_LINEAR", "SW128_MN"))
    if mk is None:
        raise RuntimeError(
            "this build has neither make_smem_layout_epi nor an MN-major "
            "128 B SmemLayoutAtomKind, so the epilogue staging plane cannot "
            "be given the swizzle D0.7 asks for")
    return cute.tile_to_shape(
        mk(kind, cutlass.BFloat16), (BLOCK_N, EPI_SEG_M, EPI_SLOTS),
        order=(0, 1, 2))


def _r2s_frame(atom, t2r, s_slot, tidx, want):
    """(tiled copy, thread copy, staging partition) for register -> staging.

    Whichever constructor hands a thread exactly `want` elements -- `want` being
    the per-thread fragment size, taken from the t2r copy's DESTINATION
    partition.  MUST be called from inside a trace.
    """
    measured, chosen = [], None
    for name in ("make_tiled_copy_D", "make_tiled_copy_S"):
        fn = getattr(cute, name, None)
        if fn is None:
            measured.append(f"{name}: absent on this build")
            continue
        try:
            tiled = fn(atom, t2r)
            thr = tiled.get_slice(tidx)
            staging = thr.partition_D(s_slot)
        except Exception as e:                             # noqa: BLE001
            measured.append(f"{name}: raised {' '.join(str(e).split())[:120]}")
            continue
        got = _size_of(staging)
        measured.append(f"{name}: {got} per thread (mode0 {_size_of(staging, 0)})")
        if chosen is None and got == want:
            chosen = (tiled, thr, staging)
    if chosen is None:
        raise RuntimeError(
            f"no register -> staging copy constructor hands a thread {want} "
            f"elements (the t2r fragment's size): " + " | ".join(measured))
    return chosen


def _r2s_src(tiled, thr, staging, frag):
    """The name of the register-side spelling `cute.copy` will accept.

    Accepted only on an exact match of BOTH the per-thread size and the atom's
    value mode: those are the two numbers the IR verifier compares, and a
    candidate matching only the first is a 256-element view of a 16-element
    fragment.
    """
    want, want0 = _size_of(staging), _size_of(staging, 0)
    opts, measured = [], []
    for name, obj in (("tiled", tiled), ("thr", thr)):
        fn = getattr(obj, "retile", None)
        if fn is None:
            measured.append(f"{name}.retile: absent")
            continue
        try:
            opts.append((name, fn(frag)))
        except Exception as e:                             # noqa: BLE001
            measured.append(
                f"{name}.retile: raised {' '.join(str(e).split())[:120]}")
    opts.append(("as-is", frag))
    for name, out in opts:
        measured.append(f"{name}: {_size_of(out)} (mode0 {_size_of(out, 0)})")
        if _size_of(out) == want and _size_of(out, 0) == want0:
            return name
    raise RuntimeError(
        f"no register-side spelling matches the staging partition "
        f"(size {want}, mode0 {want0}): " + " | ".join(measured))


def _apply_r2s_src(spelling, tiled, thr, frag):
    """The spelling `_r2s_src` chose, applied to one fragment."""
    if spelling == "as-is":
        return frag
    return (tiled if spelling == "tiled" else thr).retile(frag)


# --- derived static geometry ------------------------------------------------
# Plain python ints, so they are legal at import time and land in the traced
# code as trace-time constants.
ACT_ROWS_PER_CTA = BLOCK_M // CLUSTER_M          # 112
K_SUB = BLOCK_K // SF_VEC                        # 4 K=32 blocks per k-block
SF_ACT_BYTES = _sf_tile_bytes(BLOCK_M, K_SUB)    # 1024
SF_WGT_BYTES = _sf_tile_bytes(BLOCK_N, K_SUB)    # 512
# A single TMA box mode may not exceed 256 elements, so the flat SF run is
# carried as (run / 256, 256) with stride (256, 1) -- same address function,
# one more descriptor mode.
SF_ACT_SUB = SF_ACT_BYTES // 256
SF_WGT_SUB = SF_WGT_BYTES // 256
# Whole-pair transaction bytes for the one full barrier the leader arms.
# Derived, not copied: the weight face is charged at its PACKED size and
# everything else at its container size, and BOTH CTAs deliver into it.
PER_CTA_TX = (ACT_STAGE_BYTES + WGT_PACKED_STAGE_BYTES
              + SF_ACT_BYTES + SF_WGT_BYTES)
PAIR_TX = CLUSTER_M * PER_CTA_TX
MMA_TILER = (CLUSTER_M * BLOCK_N, BLOCK_M, BLOCK_K)


# ---------------------------------------------------------------------------
# Device side.
# ---------------------------------------------------------------------------

def _shapes():
    """Every cute-level object the host trace and the device trace share: the
    tiled MMA, the four smem operand faces, their flat TMA views, the two TMEM
    scale-factor layouts, the accumulator layout and the s2t copy atom.

    MUST be called from inside a trace (`@cute.jit` / `@cute.kernel` body).
    Every constructor below emits into the CURRENT MLIR context, and plain
    python -- e.g. the body of the builder that returns the entry point -- has
    none.  Outside a trace even a fully static shape tuple is rejected, by the
    binding layer rather than by the DSL, as

        TypeError: pack_shape(): incompatible function arguments ...
                   Invoked with types: tuple

    which names neither the context nor the call site that needed one.  So this
    is a function called from two traced sites rather than a block of builder
    code: the objects are rebuilt once per trace, which is free (all of it is
    static metadata) and keeps every value inside the region that uses it.

    It stays undecorated on purpose: it carries no runtime control flow, so the
    preprocessor has nothing to rewrite, and a `@cute.jit` helper would put its
    results behind a call boundary instead of the caller's own scope.
    """
    sm100 = _blackwell_helpers()
    S = types.SimpleNamespace()

    # --- the tiled MMA and every layout derived from it ---------------------
    # MMA-A = wgt (e2m1) so that the 256-row M axis lands on n, away from the
    # 224-row act axis; MMA-B = act (e4m3), N = 224.
    #
    # The parameter names are the ones this build's own binder reported as
    # missing when it was handed nothing: a_dtype, b_dtype, a_leading_mode,
    # b_leading_mode, sf_dtype, sf_vec_size, cta_group, mma_tiler_mn.  The
    # "major_mode" spellings are kept in the pool because they are what the
    # published examples use and the binder only passes what is asked for.
    S.tiled_mma = _call_named(
        _resolve("make_blockscaled_trivial_tiled_mma", sm100),
        a_dtype=cutlass.Float4E2M1FN,
        b_dtype=cutlass.Float8E4M3FN,
        a_leading_mode=tcgen05.OperandMajorMode.K,
        b_leading_mode=tcgen05.OperandMajorMode.K,
        a_major_mode=tcgen05.OperandMajorMode.K,
        b_major_mode=tcgen05.OperandMajorMode.K,
        a_major=tcgen05.OperandMajorMode.K,
        b_major=tcgen05.OperandMajorMode.K,
        sf_dtype=cutlass.Float8E8M0FNU,
        sf_vec_size=SF_VEC,
        cta_group=tcgen05.CtaGroup.TWO,
        mma_tiler_mn=(MMA_TILER[0], MMA_TILER[1]),
        mma_tiler=(MMA_TILER[0], MMA_TILER[1]),
        tiler_mn=(MMA_TILER[0], MMA_TILER[1]),
    )

    # D0.8: the weight plane is a Uint8 CONTAINER, not an E2M1 dense plane.
    # The dense plane is numerically wrong on this MMA kind (forbidden zone) and
    # the container is what makes the descriptor advance 32 B per K=32 block.
    S.wgt_smem = _call_named(
        _resolve("make_smem_layout_a", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        a_dtype=cutlass.Uint8, dtype=cutlass.Uint8, elem_ty=cutlass.Uint8,
        element_type=cutlass.Uint8,
        num_stages=NSTAGE, stages=NSTAGE, num_stage=NSTAGE)
    S.act_smem = _call_named(
        _resolve("make_smem_layout_b", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        b_dtype=cutlass.Float8E4M3FN, dtype=cutlass.Float8E4M3FN,
        elem_ty=cutlass.Float8E4M3FN, element_type=cutlass.Float8E4M3FN,
        num_stages=NSTAGE, stages=NSTAGE, num_stage=NSTAGE)
    S.sf_wgt_smem = _call_named(
        _resolve("make_smem_layout_sfa", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        sf_vec_size=SF_VEC, sf_dtype=cutlass.Float8E8M0FNU,
        num_stages=NSTAGE, stages=NSTAGE, num_stage=NSTAGE)
    S.sf_act_smem = _call_named(
        _resolve("make_smem_layout_sfb", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        sf_vec_size=SF_VEC, sf_dtype=cutlass.Float8E8M0FNU,
        num_stages=NSTAGE, stages=NSTAGE, num_stage=NSTAGE)

    # One stage of each operand face: the TMA box side, and the single-stage
    # blocked SF planes the TMEM layout builders are calibrated against.
    S.wgt_smem_one = _drop_stage(S.wgt_smem)
    S.act_smem_one = _drop_stage(S.act_smem)
    S.sf_wgt_smem_one = _drop_stage(S.sf_wgt_smem)
    S.sf_act_smem_one = _drop_stage(S.sf_act_smem)

    # The tiler-shaped second view of the two main faces (see `_face_view`).
    # One stage for the TMA descriptor, all stages for the destination tensor.
    S.act_view_one = _face_view(
        cutlass.Float8E4M3FN, ACT_ROWS_PER_CTA, None, S.act_smem)
    S.act_view = _face_view(
        cutlass.Float8E4M3FN, ACT_ROWS_PER_CTA, NSTAGE, S.act_smem)
    S.wgt_view_one = _face_view(cutlass.Uint8, BLOCK_N, None, S.wgt_smem)
    S.wgt_view = _face_view(cutlass.Uint8, BLOCK_N, NSTAGE, S.wgt_smem)

    # D0.12 epilogue staging: two slots of 16 m rows x 128 n bf16, carried on
    # the SWIZZLED MMA-frame face (see `_epi_face`).
    #
    # ONE buffer and ONE view, unlike the operand faces.  A swizzle is a
    # permutation of the 128 B chunks within a row, so "same bytes, different
    # mode tree" stops being a free identity the moment the plane carries one:
    # the row-major [m][n] second view this buffer used to hand the TMA store
    # would address the swizzled bytes as if they were not.  The descriptor is
    # therefore built from THIS face, and the gmem side is presented in the
    # same frame -- C as (n, m) -- which is the same bytes of the same
    # row-major C read with the mode order the staging plane uses.
    S.epi_smem = _epi_face(sm100, _epi_layout_d())
    S.epi_smem_one = _drop_stage(S.epi_smem)

    # The pair's cta layout, built the certified way rather than spelled out:
    # `size(vmnk, mode=[0])` is what every pair-aware constructor below reads to
    # decide that this is a cta_group::2 machine (KB tcgen05_umma_issue_cta_pair).
    S.cta_layout_vmnk = cute.tiled_divide(
        cute.make_layout((CLUSTER_M, 1, 1)), (S.tiled_mma.thr_id,))

    # The four operand planes are built by helpers whose parameter names this
    # build does not expose, so the binder infers them.  What makes that safe is
    # not the binding bookkeeping but this check: every plane's byte count is
    # pinned by the D0.7 topology contract, and a mis-bound stage count, dtype
    # or tiler moves it.  The SF planes additionally have to agree with the host
    # repack, which writes bytes at analytic offsets -- disagree and every byte
    # lands in the wrong place, which the court sees as a K-periodic calc_diff
    # rather than as a build error.  All four are reported together: one round
    # per wrong plane is one round too many.
    wrong = []
    for name, lay, want in (
            ("act   (MMA-B, 112 rows x 128 B x %d)" % NSTAGE,
             S.act_smem, NSTAGE * ACT_STAGE_BYTES),
            ("wgt   (MMA-A, 128 rows x 128 B container x %d)" % NSTAGE,
             S.wgt_smem, NSTAGE * WGT_STAGE_BYTES),
            ("act view (TMA side, %d stages)" % NSTAGE,
             S.act_view, NSTAGE * ACT_STAGE_BYTES),
            ("wgt view (TMA side, %d stages)" % NSTAGE,
             S.wgt_view, NSTAGE * WGT_STAGE_BYTES),
            ("sf_wgt (blocked, one stage)", S.sf_wgt_smem_one, SF_WGT_BYTES),
            ("sf_act (blocked, one stage)", S.sf_act_smem_one, SF_ACT_BYTES),
            ("epi staging (%d slots x %d x %d bf16)"
             % (EPI_SLOTS, EPI_SEG_M, BLOCK_N),
             S.epi_smem, EPI_STAGE_BYTES // 2)):
        got = int(cute.cosize(lay))
        if got != want:
            wrong.append("%s: %d B, contract says %d B" % (name, got, want))
    # The two views of each main face are only interchangeable if they carry the
    # SAME swizzle -- equal byte counts alone would let a K_SW64 view sit on a
    # K_SW128 buffer, which reads back permuted bytes and shows up in the court
    # as a wrong answer rather than as a build error.
    for name, helper, view in (("act", S.act_smem, S.act_view),
                               ("wgt", S.wgt_smem, S.wgt_view)):
        a, b = _swizzle_of(helper), _swizzle_of(view)
        if str(a) != str(b):
            wrong.append("%s: helper face swizzle %s but TMA view swizzle %s"
                         % (name, a, b))
    # The staging plane's swizzle is checked on its own, because an unswizzled
    # face would build, compile, run and answer correctly -- and leave the
    # store-side bank conflicts exactly where the M0 gate measured them.
    if _swizzle_of(S.epi_smem) is None:
        wrong.append("epi staging: the face carries NO swizzle (%s)"
                     % (S.epi_smem,))
    if wrong:
        raise RuntimeError(
            "smem topology contract (D0.7) violated by the helper-built "
            "layouts: " + " | ".join(wrong))

    # Flat byte views of the same two buffers -- the TMA destination side.
    # Mode order is the certified one (KB tma_flat_restage_box_rank): the run is
    # carried as `(run // 256, 256)` with stride `(256, 1)`, i.e. the 256-element
    # box mode is the CONTIGUOUS one and it is the second mode, not the first.
    # The address function is identical either way, which is exactly why the
    # order has to be taken from the card rather than from arithmetic: what the
    # descriptor encoder rejects is a box shape, not an address function.
    S.sf_act_flat = cute.make_layout((SF_ACT_SUB, 256, NSTAGE),
                                     stride=(256, 1, SF_ACT_BYTES))
    S.sf_wgt_flat = cute.make_layout((SF_WGT_SUB, 256, NSTAGE),
                                     stride=(256, 1, SF_WGT_BYTES))
    S.sf_act_flat_box = cute.make_layout((SF_ACT_SUB, 256), stride=(256, 1))
    S.sf_wgt_flat_box = cute.make_layout((SF_WGT_SUB, 256), stride=(256, 1))

    S.acc_layout = S.tiled_mma.make_fragment_C(
        S.tiled_mma.partition_shape_C((MMA_TILER[0], MMA_TILER[1]))).layout
    # D0.12 read-back plan.  The accumulator is a C-transpose block -- TMEM lane
    # = n (MMA-A = wgt), TMEM column = m (MMA-B = act) -- so one epilogue segment
    # is 16 COLUMNS of all 128 lanes, and 224 / 16 = 14 of them cover the tile.
    # The load op is the library's own choice for this (tile, layout, dtype,
    # 2cta) combination rather than a hand-picked `Ld32x32bOp`: the op and the
    # register frame the store side inherits have to come from the same
    # selector, which is what probe_m0_s3_t2r7.py certified.  Either way each of
    # the 128 threads ends up with 16 fp32, which is exactly the segment's
    # per-thread live set in the D0.14 budget.
    #
    # One segment as the certified RANK-2 TMEM tile.  Its strides are read off
    # the accumulator's own layout rather than assumed: `(lane_stride,
    # col_stride)` is what makes this tile the same addresses the MMA wrote.
    # `acc_col_stride` is what converts a column count into that arithmetic.
    acc_shape, acc_stride = _layout_shape_stride(S.acc_layout)
    S.acc_lanes = acc_shape[0]
    S.acc_col_stride = acc_stride[1]
    S.acc_seg_layout = cute.make_layout(
        (S.acc_lanes, EPI_SEG_M), stride=(acc_stride[0], acc_stride[1]))
    # Both the accumulator double buffer and the epilogue segment are reached by
    # INDEXING A MODE of a tensor that starts at the TMEM base, not by pointer
    # arithmetic on that base.  A TMEM pointer carries the base's alignment (16
    # B from `retrieve_tmem_ptr`), and adding a RUNTIME offset to it drops that
    # to the element width; the `tcgen05.ld` atom this epilogue is built on
    # (`tmem_load<f32, 16 DP, 256 bit, x2>`) requires at least 2 columns (8 B),
    # so a run-time-offset pointer fails IR verification outright:
    #     'cute.copy' op ... require tmem memref aligned at 2 cols (8B) at
    #     least, but got tmem align 4 bytes
    # The buffer index and the segment index therefore live as extra MODES here
    # and are applied with a slice at the use site.
    _acc_sh = _top_modes(S.acc_layout, "shape")
    _acc_st = _top_modes(S.acc_layout, "stride")
    S.acc_rank = len(_acc_sh)
    S.acc_layout_all = cute.make_layout(
        _acc_sh + (ACC_BUFFERS,),
        stride=_acc_st + (ACC_COLS * S.acc_col_stride,))
    # (lane, column-in-segment, segment, buffer): the same addresses, cut the
    # way the epilogue walks them.
    S.acc_seg_all = cute.make_layout(
        (S.acc_lanes, EPI_SEG_M, EPI_SEGS, ACC_BUFFERS),
        stride=(acc_stride[0], acc_stride[1],
                EPI_SEG_M * S.acc_col_stride, ACC_COLS * S.acc_col_stride))
    if S.acc_lanes != BLOCK_N:
        raise RuntimeError(
            "the accumulator is not the C-transpose block the topology "
            "contract describes: %d TMEM lanes, D0.4 says %d (lane = n)"
            % (S.acc_lanes, BLOCK_N))
    # The epilogue tile in the MMA's frame: (M of the MMA = n of C, N of the
    # MMA = m of C).  Both op selectors and both partitions read it this way.
    S.epi_tile = (BLOCK_N, EPI_SEG_M)
    S.epi_layout_d = _epi_layout_d()
    S.t2r_atom = _call_named(
        _resolve("get_tmem_load_op", sm100),
        cta_tile_shape_mnk=MMA_TILER, cta_tile_shape=MMA_TILER,
        cta_tile=MMA_TILER, tile_shape_mnk=MMA_TILER,
        layout_d=S.epi_layout_d, layout_D=S.epi_layout_d,
        d_layout=S.epi_layout_d,
        elem_ty_d=cutlass.BFloat16, element_type_d=cutlass.BFloat16,
        dtype_d=cutlass.BFloat16, d_dtype=cutlass.BFloat16,
        elem_ty_acc=cutlass.Float32, element_type_acc=cutlass.Float32,
        dtype_acc=cutlass.Float32, acc_dtype=cutlass.Float32,
        epi_tile=S.epi_tile, epilogue_tile=S.epi_tile, epi_tiler=S.epi_tile,
        use_2cta_instrs=True, use_2cta=True, is_2cta=True,
        use_two_cta=True, two_cta=True)
    S.tmem_sf_wgt = _call_named(
        _resolve("make_tmem_layout_sfa", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        sf_vec_size=SF_VEC, sf_dtype=cutlass.Float8E8M0FNU,
        smem_layout=S.sf_wgt_smem_one, sf_smem_layout=S.sf_wgt_smem_one,
        sfa_smem_layout=S.sf_wgt_smem_one)
    S.tmem_sf_act = _call_named(
        _resolve("make_tmem_layout_sfb", sm100),
        tiled_mma=S.tiled_mma,
        mma_tiler_mnk=MMA_TILER, mma_tiler=MMA_TILER, tiler=MMA_TILER,
        cta_tile_shape_mnk=MMA_TILER,
        sf_vec_size=SF_VEC, sf_dtype=cutlass.Float8E8M0FNU,
        smem_layout=S.sf_act_smem_one, sf_smem_layout=S.sf_act_smem_one,
        sfb_smem_layout=S.sf_act_smem_one)
    S.s2t_atom = cute.make_copy_atom(_s2t_op(), cutlass.Float8E8M0FNU)
    return S


# The TMEM pair protocol, as certified by `notes/probe_m0_tmem_pair2.py`.
#
# `tcgen05.alloc.cta_group::2` is ASYMMETRIC, not leader-only: rank 0 runs the
# FIND_AND_SET retry loop and pushes the result into the peer's DSMEM, and rank 1
# executes the same instruction and only waits for that push.  A leader-only
# alloc is therefore half a protocol -- it completes the FIRST launch (nothing
# has to be waited for yet) and then the NEXT launch's allocation of all 512
# columns has nothing to release it, which is the relaunch hang the probe
# bisected to MOUNT_SKEL.  The certified rung (T1-a2, three launches through one
# compiled entry) is `cutlass.utils.TmemAllocator` with `is_two_cta=True` and
# every call UNGUARDED so both halves of the pair execute it.
#
# These are the names that rung checked on this build; a missing one has to be a
# loud build-time python error, not a silent default inside a staged body.
_TMEM_INIT_MUST = ("alloc_result_dst_smem_ptr", "barrier_for_retrieve",
                   "is_two_cta", "two_cta_tmem_dealloc_mbar_ptr")
_TMEM_METHODS = ("allocate", "wait_for_alloc", "retrieve_ptr",
                 "relinquish_alloc_permit", "free")


def _tmem_allocator():
    """`cutlass.utils.TmemAllocator`, checked against the certified contract."""
    cls = getattr(cutlass.utils, "TmemAllocator", None)
    if cls is None:
        raise RuntimeError(
            "cutlass.utils has no TmemAllocator; the cta_group::2 TMEM pair "
            "protocol needs it -- the raw leader-only alloc/dealloc hangs on "
            "the second launch [probe_m0_tmem_pair2.py]")
    declared, _var_kw = _declared(cls)
    missing = [p for p in _TMEM_INIT_MUST if p not in (declared or ())]
    if declared is not None and missing:
        raise RuntimeError(
            "TmemAllocator.__init__ lacks %s (declared=%s); without those the "
            "allocation is not the pair form" % (missing, declared))
    absent = [m for m in _TMEM_METHODS if not hasattr(cls, m)]
    if absent:
        raise RuntimeError("TmemAllocator has no %s" % absent)
    return cls


_COMPILED: dict = {}


def _build_entry(num_m_blocks, num_n_blocks, num_k_blocks, num_experts, m_total,
                 super_group=SUPER_GROUP, mount=MOUNT_SHIPPED):
    """Build the shape-specialized DSL entry point for one launch geometry.

    Everything closed over here is a plain python int or a python-level binding
    decision: the integer divisions in the rasterization land as multiply-shift
    chains rather than as division instructions, and no cute object is built at
    this level -- builder code runs outside any MLIR context, where every cute
    constructor fails (see `_shapes`).
    """
    num_tiles = num_m_blocks * num_n_blocks
    n_cols = num_n_blocks * BLOCK_N
    k_depth = num_k_blocks * BLOCK_K
    group_span = num_m_blocks * super_group      # tiles per full super-group
    tiles_per_cta = (num_tiles + GRID - 1) // GRID
    # A tail exists when the tile count is not a multiple of the grid.  It never
    # splits a pair: num_tiles is even (num_n_blocks is) and GRID is even, so the
    # per-CTA trip count changes only at an even bidx.
    has_tail = (num_tiles % GRID) != 0
    has_hw_rank = hasattr(cute.arch, "block_idx_in_cluster")

    # TMEM alloc / retrieve / release all go through one library object (D0.6 /
    # D0.11); resolved and contract-checked here, where a missing spelling is a
    # loud python error rather than an attribute lookup in a staged body.
    _tmem_alloc_cls = _tmem_allocator()
    _named_bar = getattr(pipeline, "NamedBarrier", None)
    if _named_bar is None:
        raise RuntimeError(
            "cutlass.pipeline has no NamedBarrier; the TMEM allocator needs one "
            "to publish the base address to the whole CTA")

    # D1.1: the spin arm needs three spellings on this build -- the participants'
    # non-blocking probe, the token parameter on the blocking call the probe's
    # result is handed back to, and the arch primitive the probe lowers to.  The
    # token hand-back is not an optimization: on `PipelineTmaUmma` the producer's
    # acquire ALSO arms the pair's transaction count, so the call has to happen
    # whether or not the wait did [KB mbarrier_wait_encoding gotcha 2].  Checked
    # HERE because a missing spelling inside the traced body is an attribute
    # error with no line of this file in it -- and because `raise` in a staged
    # body is rejected outright by the preprocessor, not at the raise.
    spin_wait = bool(WAIT_SPIN)
    if spin_wait:
        missing = []
        if not hasattr(cute.arch, "mbarrier_try_wait"):
            missing.append("cute.arch.mbarrier_try_wait")
        for cls_name, probe, blocking, tok_param in (
                ("PipelineProducer", "try_acquire", "acquire_and_advance",
                 "try_acquire_token"),
                ("PipelineConsumer", "try_wait", "wait_and_advance",
                 "try_wait_token")):
            cls = getattr(pipeline, cls_name, None)
            if cls is None:
                missing.append("cutlass.pipeline." + cls_name)
                continue
            if not hasattr(cls, probe):
                missing.append("%s.%s" % (cls_name, probe))
            blocking_fn = getattr(cls, blocking, None)
            if blocking_fn is None:
                missing.append("%s.%s" % (cls_name, blocking))
                continue
            named, var_kw = _declared(blocking_fn)
            if named is not None and not var_kw and tok_param not in named:
                missing.append("%s.%s(%s=) -- declared %s"
                               % (cls_name, blocking, tok_param, list(named)))
        if missing:
            raise RuntimeError(
                "WAIT_SPIN is on but this build lacks %s; the spin encoding "
                "needs the non-blocking probe and the token hand-back, so set "
                "WAIT_SPIN = False to run the blocking arm instead" % missing)

    # D1.2: the peek arm reuses exactly the spellings the spin arm just checked
    # -- the probe and the token parameter -- so it has no build check of its
    # own; it only needs the spin arm to be on, because the point of a peek is
    # that its token is handed to the blocking call instead of a fresh probe.
    peek_ahead = bool(PEEK_AHEAD) and spin_wait

    mount_pipe = mount >= MOUNT_PIPE
    mount_tma = mount >= MOUNT_TMA
    mount_mma = mount >= MOUNT_MMA
    mount_acc = mount >= MOUNT_ACC
    mount_epi = mount >= MOUNT_EPI
    mount_store = mount >= MOUNT_STORE

    # Epilogue entry points, bound at build time for the same reason as `_free`:
    # a missing spelling has to be a loud python error out here, not an
    # attribute lookup inside a staged body where `raise` is rejected outright.
    sm100 = _blackwell_helpers()
    _tmem_copy = _resolve("make_tmem_copy", tcgen05)
    # The s2t SOURCE side is not the partitioned smem tensor itself but a
    # descriptor view of it: `tcgen05.cp` addresses smem through an MMA-style
    # descriptor, and the partition alone carries plain addresses.
    _s2t_desc = _resolve("get_s2t_smem_desc_tensor", tcgen05)
    _r2s_op = _resolve("get_smem_store_op", sm100)
    _store_op = _resolve("CopyBulkTensorTileS2GOp", cpasync)
    _store_commit, _store_wait = _bulk_group_ops()
    _fence_async_proxy = _async_proxy_fence()
    # `LayoutEnum` names the major order the stmatrix selector is keyed on, read
    # in the MMA's frame -- COL_MAJOR, which is what makes the accumulator's
    # C-transpose block come out as a row of C in the staging slot.  Bound here
    # for the same reason as `_free`: a missing member has to be a loud python
    # error out here rather than an attribute lookup in a staged body.
    _layout_d = _epi_layout_d()

    mount_mma = mount >= MOUNT_MMA

    # TMA atom factory, bound and checked HERE rather than inside the traced
    # entry.  The DSL's AST pass rejects `raise` anywhere in a staged body
    # (UNSUP_EARLY_EXIT) -- it never evaluates the condition, so a guard that is
    # false at trace time still kills the compile.  Build-time python is the
    # only place a precondition may be asserted.
    _tma_atom = _resolve("make_tiled_tma_atom", cpasync)
    # D1.4(c): the tensormap prefetch.  Bound out here for the same reason as
    # every other spelling above -- inside a staged body a missing name is an
    # attribute error with no line of this file in it.
    _desc_prefetch = _resolve("prefetch_descriptor", cpasync)
    if not _accepts(_tma_atom, "internal_type"):
        raise RuntimeError(
            "make_tiled_tma_atom has no internal_type parameter; the fp4 "
            "weight face needs U4_UNPACK_U8 -- the plain format lands as a "
            "dense 64 B row and the MMA reads garbage (forbidden zone)")

    @cute.kernel
    def gemm_kernel(
        tma_act: cute.CopyAtom,
        gAct: cute.Tensor,      # ((112,128), (rest_m, rest_k))
        tma_wgt: cute.CopyAtom,
        gWgt: cute.Tensor,      # ((128,128), (rest_n, rest_k, G))
        tma_sf_act: cute.CopyAtom,
        gSFAct: cute.Tensor,    # ((sub,256), (1, 1, KB, MB))
        tma_sf_wgt: cute.CopyAtom,
        gSFWgt: cute.Tensor,    # ((sub,256), (1, 1, KB, NB, G))
        mBlkExp: cute.Tensor,   # int32, (num_m_blocks,)
        tma_c: cute.CopyAtom,
        gC: cute.Tensor,        # ((128,16), (rest_n, rest_seg)) bf16, MMA frame
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tidx // 32)

        rank = cutlass.Int32(bidx % CLUSTER_M)
        if cutlass.const_expr(has_hw_rank):
            rank = cutlass.Int32(cute.arch.block_idx_in_cluster())
        is_leader = cutlass.Boolean(rank == 0)

        # --- D0.6 roles ------------------------------------------------------
        # Wrap each predicate as a Staged (Boolean) value: the operands of ==
        # / >= / & are cutlass.Int32 / cutlass.Boolean, which support Python
        # operator overloads and can return a Python bool at trace time.
        # cutlass.Boolean(...) ensures the DSL sees a runtime SSA value and
        # lowers each `if` as a proper scf.if rather than a Python branch.
        is_producer = cutlass.Boolean(warp_id == 0)                  # both CTAs of the pair
        is_issuer = cutlass.Boolean((warp_id == 1) & is_leader)         # leader issues, peer idles
        # D0.6 keeps warp 2 on the TMEM alloc/dealloc handshake, but the guard
        # is no longer written here: the allocator object is told
        # `allocator_warp_id=2` and elects that warp itself, while the calls
        # around it are CTA-wide (see the allocation block below).
        is_epilogue = cutlass.Boolean(warp_id >= 4)                  # warps 4-7, 128 threads
        is_epi_lead = cutlass.Boolean(warp_id == 4)                  # the store-issuing warp
        epi_tid = tidx - (CTA_THREADS - EPI_THREADS)

        # --- smem topology (D0.7) -------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        if cutlass.const_expr(mount_pipe):
            # Built HERE, inside the device trace, from the same function the
            # host trace calls: a layout is an MLIR value of the region that
            # created it, so each region builds its own copy of this (entirely
            # static) metadata rather than reaching into the other's.
            S = _shapes()
            # The two main faces are allocated in SPLIT form -- affine layout,
            # swizzle on the pointer.  Handing `allocate_tensor` the composed
            # layout instead makes `make_fragment_A/B` reject the result as
            # "Expected affine layout for operand A, but got composed layout".
            sAct = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=_affine_of(S.act_smem), byte_alignment=1024,
                swizzle=_swizzle_of(S.act_smem))
            sWgt = smem.allocate_tensor(
                element_type=cutlass.Uint8,
                layout=_affine_of(S.wgt_smem), byte_alignment=1024,
                swizzle=_swizzle_of(S.wgt_smem))
            # The TMA's view of those same bytes: same swizzled pointer, the
            # `(rows, BLOCK_K, stage)` mode tree the descriptor builder's V-map
            # check wants.  `_shapes` has already checked that the two views
            # agree on both the byte count and the swizzle.
            sActTma = cute.make_tensor(sAct.iterator, _affine_of(S.act_view))
            sWgtTma = cute.make_tensor(sWgt.iterator, _affine_of(S.wgt_view))
            # Both scale planes are allocated as FLAT byte runs: the TMA writes
            # them linearly and the tcgen05.cp reads them back through a second
            # view carrying the blocked layout.  One buffer, two views -- which
            # is only sound because the blocked layout is compact, checked in
            # `_shapes`.
            sSFActFlat = smem.allocate_tensor(
                cutlass.Uint8, S.sf_act_flat, byte_alignment=1024)
            sSFWgtFlat = smem.allocate_tensor(
                cutlass.Uint8, S.sf_wgt_flat, byte_alignment=1024)
            # Split form again, for the same reason as the operand faces: the
            # affine part is the layout and the 128 B swizzle rides on the
            # pointer, which is where both the stmatrix partition and the store
            # descriptor expect to find it.
            sEpi = smem.allocate_tensor(
                cutlass.BFloat16, _affine_of(S.epi_smem), byte_alignment=1024,
                swizzle=_swizzle_of(S.epi_smem))
        # The pair's dealloc mbarrier and the slot the allocation result is
        # published into.  Both are allocated OUTSIDE the mount guard and in a
        # fixed order, so they sit at the same smem offset in each half of the
        # pair -- which is what makes the peer's DSMEM arrival land on the
        # leader's copy of the barrier rather than on whatever is at that offset.
        tmem_dealloc_mbar = smem.allocate_array(
            cutlass.Int64, 1, byte_alignment=16)
        tmem_holding = smem.allocate_array(cutlass.Int32, 1, byte_alignment=16)
        if cutlass.const_expr(mount_pipe):
            mbar_storage = smem.allocate_array(
                cutlass.Int64, 2 * NSTAGE, byte_alignment=16)
            acc_mbar_storage = smem.allocate_array(
                cutlass.Int64, 2 * ACC_BUFFERS, byte_alignment=16)
            # tx_count is the WHOLE PAIR's bytes and cta_layout_vmnk carries
            # V=2: without the latter the pipeline silently degrades to
            # CtaGroup.ONE and arms a barrier that only ever receives half of
            # what it waits for.  `make_participants` is the certified shape of
            # this object (KB tcgen05_umma_issue_cta_pair): the tokens it hands
            # back already carry the leader-mapped barrier pointer and the
            # commit is already elected, so neither is done by hand here.
            ab_producer, ab_consumer = _call_named(
                pipeline.PipelineTmaUmma.create,
                _widen=False,          # this one EMITS IR: call it exactly once
                _require=("barrier_storage", "num_stages", "tx_count",
                          "cta_layout_vmnk"),
                barrier_storage=mbar_storage,
                num_stages=NSTAGE,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                tx_count=PAIR_TX,
                cta_layout_vmnk=S.cta_layout_vmnk,
            ).make_participants()
            # D0.11: acc_full is one tcgen05.commit from the issuer; acc_empty
            # counts EVERY epilogue thread of BOTH halves of the pair -- 128 x 2
            # = 256.  One thread short of that and the issuer waits forever.
            acc_producer, acc_consumer = _call_named(
                pipeline.PipelineUmmaAsync.create,
                _widen=False,
                _require=("barrier_storage", "num_stages", "cta_layout_vmnk"),
                barrier_storage=acc_mbar_storage,
                num_stages=ACC_BUFFERS,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, CLUSTER_M * EPI_THREADS),
                cta_layout_vmnk=S.cta_layout_vmnk,
            ).make_participants()

        # --- D1.4(c) tensormap prefetch --------------------------------------
        # All five descriptors this kernel ever uses, pulled into the
        # descriptor cache before any tile runs.  A tensormap is read by the TMA
        # unit out of gmem on first use; without this the first delivery of the
        # first tile -- the one nothing is overlapping yet -- pays five cold
        # fetches in series with the prologue.  Issued from warp 0 only (the
        # KB's shape: 32 redundant prefetches per CTA, no lane election) and
        # placed BEFORE the TMEM allocation handshake so the cluster barrier
        # that handshake ends with is what covers the fetch latency.  The store
        # descriptor is prefetched here too even though warp 4 is what issues
        # the store: the cache is per-SM, not per-warp.
        if cutlass.const_expr(mount_tma):
            if is_producer:
                _desc_prefetch(tma_act)
                _desc_prefetch(tma_wgt)
                _desc_prefetch(tma_sf_act)
                _desc_prefetch(tma_sf_wgt)
                _desc_prefetch(tma_c)

        # --- TMEM pair allocation (D0.6 / D0.11) -----------------------------
        # Certified form [notes/probe_m0_tmem_pair2.py, rung T1-a2: three
        # launches through one compiled entry at the shipped launch geometry].
        # The allocator object is built BEFORE `mbarrier_init_fence` because
        # with `is_two_cta=True` its constructor is what emits the dealloc
        # mbarrier's initialization -- the fence and the cluster barrier that
        # make that initialization visible have to come after it.
        #
        # Every call below is UNGUARDED: `tcgen05.alloc.cta_group::2` is issued
        # by a warp in EACH half of the pair (rank 0 searches and pushes, rank 1
        # waits for the push), and `allocator_warp_id=2` is what keeps D0.6's
        # TMEM warp on the instruction inside the library.  `wait_for_alloc` and
        # `free` synchronize the whole CTA through the named barrier below, so
        # all 256 threads have to reach them.
        tmem_bar = _call_named(
            _named_bar, _widen=False, _require=("barrier_id",),
            barrier_id=TMEM_BAR_ID,
            num_threads=CTA_THREADS, number_of_threads=CTA_THREADS)
        # `num_allocated_columns` is deliberately NOT in this pool.  It is the
        # allocator's RUNNING TOTAL of columns already handed out, not a
        # capacity to declare: seeding it with 512 and then asking `allocate`
        # for 512 more counts the one allocation twice and the allocator
        # refuses the pair at trace time with "total allocated columns must be
        # less than or equal to 512".  The certified rung
        # [notes/probe_m0_tmem_pair2.py, T1-a2] binds only the five names below
        # and lets the total start at the default.
        tmem_alloc = _call_named(
            _tmem_alloc_cls, _widen=False,
            _require=("alloc_result_dst_smem_ptr", "is_two_cta",
                      "two_cta_tmem_dealloc_mbar_ptr"),
            alloc_result_dst_smem_ptr=tmem_holding,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=2,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar)
        _seg("tmem.alloc.pair.begin")
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        _call_named(tmem_alloc.allocate, _widen=False,
                    _require=("num_columns",), num_columns=TMEM_COLS)
        tmem_alloc.wait_for_alloc()
        _seg("tmem.alloc.pair.end")
        tmem_ptr = _call_named(tmem_alloc.retrieve_ptr, _widen=False,
                               dtype=cutlass.Float32,
                               element_type=cutlass.Float32)

        # --- D0.5 persistence + D0.13 rasterization --------------------------
        # Always use the compile-time constant trip count so every pipeline's
        # tail() sees a static depth.  Out-of-range iterations (tail wave) are
        # masked in the producer (gmem indices clamped to tile 0) and in the
        # epilogue (TMA store gated on ep_in_range); acquire/release calls
        # remain unconditional so barrier depths stay balanced on both halves.
        n_iters = cutlass.Int32(tiles_per_cta)

        if cutlass.const_expr(not mount_tma):
            # S1 / +faces+pipeline shape: each role folds the schedule it would
            # have driven into a live value so the dispatch cannot be optimized
            # away.  The pipeline object, when it exists at this level, is built
            # and left unused -- "does the pipeline CONSTRUCT" is the question
            # this rung is for, and the answer must not be entangled with
            # whether the deliveries partition.
            acc = cutlass.Int32(0)
            for t in cutlass.range(n_iters):
                _seg("sched.tile.begin", t)
                idx = bidx + GRID * t
                in_range = idx < num_tiles
                group = idx // group_span
                in_group = idx - group * group_span
                m_block = in_group // super_group
                n_block = group * super_group + (in_group - m_block * super_group)
                if is_producer & in_range:
                    acc = acc + m_block * BLOCK_M + n_block * BLOCK_N
                if is_issuer & in_range:
                    acc = acc + m_block
                _seg("sched.tile.end", t)
            if acc == -1:
                # A sink the compiler cannot prove dead, writing somewhere that
                # exists at every mount level.  The predicate never fires.
                tmem_holding[0] = cutlass.Int32(0)
        else:
            tAsA, tAgA = cpasync.tma_partition(
                tma_act, 0, cute.make_layout(1), _group_but_stage(sActTma), gAct)
            tBsB, tBgB = cpasync.tma_partition(
                tma_wgt, 0, cute.make_layout(1), _group_but_stage(sWgtTma), gWgt)
            tFsF, tFgF = cpasync.tma_partition(
                tma_sf_act, 0, cute.make_layout(1),
                _group_but_stage(sSFActFlat), gSFAct)
            tEsE, tEgE = cpasync.tma_partition(
                tma_sf_wgt, 0, cute.make_layout(1),
                _group_but_stage(sSFWgtFlat), gSFWgt)

            # --- producer: warp 0 on BOTH CTAs (D0.10) ----------------------
            # Guarding the delivery side with is_leader is a deadlock, not a
            # slowdown: the follower would stop delivering its half and the
            # leader's barrier would never fill.
            if is_producer:
                # D1.4(b): the expert entry is now read one tile AHEAD.  The
                # index arithmetic of tile t+1 and its single LDG are done at
                # the k-loop tail of tile t -- under the last stage's four TMAs
                # -- and the value is carried into the next iteration in a
                # register, so the top of a tile reaches its first empty-wait
                # with nothing outstanding in front of it.  Previously this LDG
                # sat between `prod.tile.begin` and the tile's first
                # `producer_acquire`, where its latency is exposed once per
                # tile on the delivery warp.
                #
                # Tile 0's entry is read HERE, in the producer prologue: one LDG
                # for the whole life of the CTA, covered by the TMEM allocation
                # handshake that precedes this point.  The three names below are
                # loop-carried (assigned in the body, live before it), so each
                # is pinned to `cutlass.Int32` on both sides -- the initializer
                # and the tail -- rather than left to whatever the tensor read
                # and the division chain each happen to produce.
                pr_idx0 = cutlass.Int32(bidx)
                pr_clamped0 = cutlass.Int32(pr_idx0 * (pr_idx0 < num_tiles))
                pr_group0 = pr_clamped0 // group_span
                pr_in_group0 = pr_clamped0 - pr_group0 * group_span
                pr_m_block = cutlass.Int32(pr_in_group0 // super_group)
                pr_n_block = cutlass.Int32(
                    pr_group0 * super_group
                    + (pr_in_group0 - pr_m_block * super_group))
                pr_expert = cutlass.Int32(mBlkExp[pr_m_block])
                # D1.2 producer peek, site 1 of 2 (the prologue probe).  It is
                # UNGUARDED by design: empty[s] is arrived on BOTH halves of the
                # pair, so both halves are eligible to ask, and a leader guard
                # here is the deadlock in the KB's table (the follower would stop
                # delivering its half of the pair's bytes).  Guard (2) is
                # satisfied by construction: this token is consumed by the very
                # first acquire, which exists whenever the CTA runs at all.
                # Binding by CALLING rather than by a dummy is deliberate -- the
                # name has to exist before the two loops to be carried out of
                # them, and calling is what pins it to the probe's own type
                # instead of to whatever `Boolean(False)` happens to be.
                if cutlass.const_expr(peek_ahead):
                    pr_peek = ab_producer.try_acquire()
                for pr_t in cutlass.range(n_iters):
                    _seg("prod.tile.begin", pr_t)
                    pr_act_tile = pr_m_block * CLUSTER_M + rank
                    for pr_kb in cutlass.range(num_k_blocks, unroll=1):
                        _seg("prod.stage.begin", pr_t)
                        # The token's barrier is ALREADY the leader's: the
                        # pipeline was created with cta_layout_vmnk carrying
                        # V=2, which is what makes the pair's accounting -- one
                        # barrier, both CTAs' bytes -- the pipeline's job rather
                        # than a hand-rolled map_dsmem_ptr at every call site.
                        #
                        # D1.1 site 1 of 4: producer waits empty[s].  Under the
                        # spin arm the wait is this retry loop and the acquire
                        # is still called -- with the satisfied token, so its
                        # wait is skipped and its OTHER half, the leader-side
                        # `arrive_and_expect_tx` that arms the pair's byte
                        # count, still runs.  Dropping the call instead is the
                        # deadlock in KB mbarrier_wait_encoding gotcha 2.
                        if cutlass.const_expr(peek_ahead):
                            # The probe for THIS acquire was issued at the tail
                            # of the previous iteration (or in the prologue), so
                            # in the common case the loop below runs zero times
                            # and the acquire costs no barrier read.  When the
                            # peek came back false the retry loop is exactly the
                            # non-peek arm's, re-using the same name.
                            while pr_peek == 0:
                                pr_peek = ab_producer.try_acquire()
                            pr_tok = ab_producer.acquire_and_advance(pr_peek)
                        elif cutlass.const_expr(spin_wait):
                            pr_ok = ab_producer.try_acquire()
                            while pr_ok == 0:
                                pr_ok = ab_producer.try_acquire()
                            pr_tok = ab_producer.acquire_and_advance(pr_ok)
                        else:
                            pr_tok = ab_producer.acquire_and_advance()
                        pr_s = pr_tok.index
                        pr_bar = pr_tok.barrier
                        cute.copy(tma_act, tAgA[(None, (pr_act_tile, pr_kb))],
                                  tAsA[(None, pr_s)], tma_bar_ptr=pr_bar)
                        cute.copy(tma_wgt, tBgB[(None, (pr_n_block, pr_kb, pr_expert))],
                                  tBsB[(None, pr_s)], tma_bar_ptr=pr_bar)
                        cute.copy(tma_sf_act, tFgF[(None, (0, 0, pr_kb, pr_m_block))],
                                  tFsF[(None, pr_s)], tma_bar_ptr=pr_bar)
                        cute.copy(tma_sf_wgt,
                                  tEgE[(None, (0, 0, pr_kb, pr_n_block, pr_expert))],
                                  tEsE[(None, pr_s)], tma_bar_ptr=pr_bar)
                        # D1.2 producer peek, site 2 of 2 (the steady-state tail
                        # probe), placed AFTER the four deliveries so nothing
                        # this iteration still owes the pipeline sits behind a
                        # potentially-blocking probe.  Guard (2) written out as a
                        # FLAT wait counter rather than a disjunction of the two
                        # loop bounds -- both bounds are python ints closed over
                        # by the builder, so the whole predicate folds to one
                        # imad and one compare, and it is false exactly once per
                        # call: at the last k-block of the last tile, where
                        # `ab_producer.tail()`, not
                        # another acquire, is what follows.  Skipping the
                        # assignment leaves the carried token at its last value;
                        # that value is never read again, because the only
                        # reader is an acquire that no longer happens.
                        if cutlass.const_expr(peek_ahead):
                            pr_more = (pr_t * num_k_blocks + pr_kb + 1
                                       < tiles_per_cta * num_k_blocks)
                            if pr_more:
                                pr_peek = ab_producer.try_acquire()
                        _seg("prod.stage.end", pr_t)
                    # k-loop tail: tile t+1's coordinates and its one LDG, with
                    # the last stage's four TMAs still in flight to cover them.
                    # Tail wave: an OOB next-tile index clamps to tile 0 exactly
                    # as the in-loop index used to, so the read stays in bounds
                    # and the deliveries of a masked iteration still fill the
                    # barrier (the epilogue is what drops the store).
                    pr_nxt = bidx + GRID * (pr_t + 1)
                    pr_clamped = cutlass.Int32(pr_nxt * (pr_nxt < num_tiles))
                    pr_group = pr_clamped // group_span
                    pr_in_group = pr_clamped - pr_group * group_span
                    pr_m_block = cutlass.Int32(pr_in_group // super_group)
                    pr_n_block = cutlass.Int32(
                        pr_group * super_group
                        + (pr_in_group - pr_m_block * super_group))
                    pr_expert = cutlass.Int32(mBlkExp[pr_m_block])
                    _seg("prod.tile.end", pr_t)
                # Drains the arrives the producer is ahead by.  Without it the
                # SECOND launch of the same compiled entry hangs -- which in a
                # bench that calls the kernel twenty times is every call but the
                # first (KB tcgen05_umma_issue_cta_pair gotcha 8).
                ab_producer.tail()

            # --- consumer: warp 1 on the LEADER only (D0.4 / D0.9) ----------
            # The MMA frame is built OUTSIDE the role predicate, which is the
            # KB's certified shape (tcgen05_umma_issue_cta_pair step (3): the
            # fragments are made at kernel top level and consumed inside
            # `if warp_idx == 0:`).
            #
            # What a dynamic `if` carries is NOT "every free variable of the
            # region".  The DSL's region analysis builds the carried list from
            # exactly two sources: names ASSIGNED anywhere in the body that are
            # also live in the enclosing scope, plus names used as the RECEIVER
            # of a method call in the body -- receivers are marked because a
            # python object is mutable by default, so the region has to be
            # handed a runtime copy of it.  Reading an outer name costs
            # nothing; calling a method on it is what conscripts it.
            #
            # `S` is the shared-metadata namespace: a plain python object whose
            # attributes include a `LayoutEnum`, so it has no runtime form at
            # all.  Any `S.<x>.method(...)` inside a role region is therefore
            # `error[TYPE_DYNAMIC_EXPR_UNSUPPORTED] ... (a LayoutEnum)`,
            # reported AT THE `if` and naming a value the `if` never mentions.
            # Which side of the predicate the frame is built on does not enter
            # into it -- `make_fragment_A` is itself a call on `S.tiled_mma`,
            # so moving the block inside the guard moved the fault with it.
            # The rule this code follows: inside a role region `S` may only be
            # READ; the MMA handle is re-bound to a body-local name in the
            # innermost region that calls `.set()` on it (a name first bound in
            # the body is dropped from the carried list, and a receiver that is
            # also assigned in the body is removed from it).
            if cutlass.const_expr(mount_mma):
                # The operand fragments come from the smem FACE, not from a
                # partition of it: `make_fragment_A/B` is what reshapes the
                # (mn, k, stage) face into the (MMA, MMA_MN, MMA_K, STAGE)
                # profile `cute.gemm` checks, and in pair mode it is also
                # the only thing that knows "this CTA holds half".  Feeding
                # it a self-partitioned tensor is the documented route to
                # `expected input tensor to have the profile (MMA, MMA_MN,
                # MMA_K, ...)` at the gemm.
                tCrA = S.tiled_mma.make_fragment_A(sWgt)
                tCrB = S.tiled_mma.make_fragment_B(sAct)
                tCtSFWgt = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + TMEM_SFA_COL,
                                    dtype=cutlass.Float8E8M0FNU),
                    S.tmem_sf_wgt)
                tCtSFAct = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + TMEM_SFB_COL,
                                    dtype=cutlass.Float8E8M0FNU),
                    S.tmem_sf_act)
                # The second view of the two scale buffers: same bytes the
                # TMA wrote, read through the blocked layout UTCCP wants.
                sSFWgt = cute.make_tensor(
                    cute.recast_ptr(sSFWgtFlat.iterator,
                                    dtype=cutlass.Float8E8M0FNU),
                    S.sf_wgt_smem)
                sSFAct = cute.make_tensor(
                    cute.recast_ptr(sSFActFlat.iterator,
                                    dtype=cutlass.Float8E8M0FNU),
                    S.sf_act_smem)
                # The blocked SF layouts the helpers produce carry 0-stride
                # modes (the same UE8M0 byte is read by four K=32 blocks).
                # Both sides of the s2t copy have to be filtered before
                # they are partitioned -- the TMEM side because the tiled
                # copy is BUILT from it, the smem side at partition time.
                tCtSFWgt_z = cute.filter_zeros(tCtSFWgt)
                tCtSFAct_z = cute.filter_zeros(tCtSFAct)
                s2t_wgt = tcgen05.make_s2t_copy(S.s2t_atom, tCtSFWgt_z)
                s2t_act = tcgen05.make_s2t_copy(S.s2t_atom, tCtSFAct_z)
                thr_wgt = s2t_wgt.get_slice(0)
                thr_act = s2t_act.get_slice(0)
                sf_wgt_src = _s2t_desc(
                    s2t_wgt, thr_wgt.partition_S(cute.filter_zeros(sSFWgt)))
                sf_wgt_dst = thr_wgt.partition_D(tCtSFWgt_z)
                sf_act_src = _s2t_desc(
                    s2t_act, thr_act.partition_S(cute.filter_zeros(sSFAct)))
                sf_act_dst = thr_act.partition_D(tCtSFAct_z)

            # The full barrier is armed only on the leader, so every consumer
            # call -- blocking or not -- has to sit inside this predicate.
            if is_issuer:
                # D1.2 consumer peek, site 1 of 2 (the prologue probe).  Guard
                # (1) is this `if`: `is_issuer` is `warp 1 AND leader`, and full
                # is armed only on the leader, so a probe one nesting level out
                # would be asking the follower's copy of a barrier that is never
                # arrived -- the "diffuse through the whole main loop, priced at
                # the suspend timeout" half of the KB's table.  Guard (2) holds
                # by construction: the first full-wait always exists.
                if cutlass.const_expr(peek_ahead):
                    mm_peek = ab_consumer.try_wait()
                for mm_t in cutlass.range(n_iters):
                    _seg("mma.tile.begin", mm_t)
                    # D0.11: tile t accumulates into buffer t & 1, and may not
                    # start until the epilogue of tile t-2 has drained that same
                    # buffer.  The wait is at the TOP of the tile, so the MMA
                    # chain of tile t runs under the epilogue of tile t-1.
                    if cutlass.const_expr(mount_acc):
                        # D1.1 site 2 of 4: issuer waits acc_empty[b].  Once per
                        # tile, and the one wait whose latency lands directly on
                        # the MMA chain's start -- nothing of tile t issues until
                        # it retires.  Bracketed by its own timeline sites: it is
                        # the only MMA-role wait that is not inside the k-loop,
                        # so without a pair of its own it reads as k-loop time.
                        _seg("mma.acc.begin", mm_t)
                        # D1.2 audit, not peeked.  acc_empty is arrived by all
                        # 256 epilogue threads of the pair onto the LEADER's
                        # barrier and waited on by the leader's issuer, so this
                        # probe's eligibility guard is the enclosing `is_issuer`
                        # -- same column of the KB table as full[s].  It is left
                        # at the point of use: it runs once per tile, and unlike
                        # full[s] it is a wait that is genuinely not satisfied
                        # when it is reached (the epilogue of tile t-2 is still
                        # running), so asking earlier only moves the suspend
                        # earlier.  A peek prices at zero where the token would
                        # come back false anyway.
                        if cutlass.const_expr(spin_wait):
                            mm_acc_ok = acc_producer.try_acquire()
                            while mm_acc_ok == 0:
                                mm_acc_ok = acc_producer.try_acquire()
                            mm_acc_tok = acc_producer.acquire_and_advance(
                                mm_acc_ok)
                        else:
                            mm_acc_tok = acc_producer.acquire_and_advance()
                        _seg("mma.acc.end", mm_t)
                        # Buffer `index` of the double buffer, taken as a MODE
                        # of the tensor at the TMEM base rather than as an
                        # offset on the base pointer: a run-time pointer offset
                        # drops the pointer's alignment to the element width
                        # (see `acc_layout_all` in `_shapes`).
                        tCtAcc = cute.make_tensor(
                            tmem_ptr, S.acc_layout_all)[
                                (None,) * S.acc_rank + (mm_acc_tok.index,)]
                    elif cutlass.const_expr(mount_mma):
                        tCtAcc = cute.make_tensor(tmem_ptr, S.acc_layout)
                    for mm_kb in cutlass.range(num_k_blocks, unroll=1):
                        _seg("mma.stage.begin", mm_t)
                        # D1.1 site 3 of 4: issuer waits full[s], 32 times per
                        # tile -- the hottest wait in the kernel.  The consumer
                        # side has no second job the way the producer's acquire
                        # does, so the token hand-back here only skips the
                        # blocking wait.
                        if cutlass.const_expr(peek_ahead):
                            # Steady state: the producer runs up to NSTAGE ahead,
                            # so the stage this iteration consumes was already
                            # full when the previous iteration probed it and the
                            # retry loop runs zero times.  That is the whole
                            # lever -- 32 barrier reads per tile become 32 token
                            # tests on a value already in a register.
                            while mm_peek == 0:
                                mm_peek = ab_consumer.try_wait()
                            mm_tok = ab_consumer.wait_and_advance(mm_peek)
                        elif cutlass.const_expr(spin_wait):
                            mm_ok = ab_consumer.try_wait()
                            while mm_ok == 0:
                                mm_ok = ab_consumer.try_wait()
                            mm_tok = ab_consumer.wait_and_advance(mm_ok)
                        else:
                            mm_tok = ab_consumer.wait_and_advance()
                        mm_s = mm_tok.index
                        if cutlass.const_expr(mount_mma):
                            # The MMA handle under a BODY-LOCAL name.  The
                            # instruction descriptor is mutated below (SFA/SFB/
                            # ACCUMULATE), and the receiver of a method call in
                            # a region body is conscripted into that region's
                            # carried list -- `S.tiled_mma.set(...)` would carry
                            # `S` itself, which is a plain python namespace with
                            # a `LayoutEnum` in it and no runtime form.  A name
                            # ASSIGNED in the same body is dropped from the
                            # carried list instead, so the alias is what keeps
                            # `S` read-only in here.  Same object either way:
                            # the rebind is trace-time python, not a copy.
                            mma_h = S.tiled_mma
                            # tcgen05.cp: wgt SF (128 rows, one atom) and act SF
                            # (224 rows, two atoms) up to their TMEM columns.
                            cute.copy(s2t_wgt, _stage_slice(sf_wgt_src, mm_s),
                                      sf_wgt_dst)
                            cute.copy(s2t_act, _stage_slice(sf_act_src, mm_s),
                                      sf_act_dst)
                            for kp in cutlass.range_constexpr(K_SUB):
                                mma_h.set(tcgen05.Field.SFA,
                                          _sf_slice(tCtSFWgt, kp).iterator)
                                mma_h.set(tcgen05.Field.SFB,
                                          _sf_slice(tCtSFAct, kp).iterator)
                                # ACCUMULATE is cleared exactly once per TILE,
                                # and it is set in the SAME region as the gemm
                                # that reads it: setting it in the enclosing
                                # tile-loop body instead is the "operand does
                                # not dominate this use" fault.  Only the very
                                # first instruction of the tile overwrites.
                                if cutlass.const_expr(kp == 0):
                                    mma_h.set(tcgen05.Field.ACCUMULATE,
                                              mm_kb > 0)
                                else:
                                    mma_h.set(tcgen05.Field.ACCUMULATE,
                                              True)
                                cute.gemm(mma_h, tCtAcc,
                                          tCrA[(None, None, kp, mm_s)],
                                          tCrB[(None, None, kp, mm_s)],
                                          tCtAcc)
                        mm_tok.release()
                        # D1.2 consumer peek, site 2 of 2 (the steady-state tail
                        # probe).  It sits after `wait_and_advance`, so the
                        # participant's state already points at the NEXT stage
                        # and this probe asks about that stage -- the token it
                        # produces is consumed by the head of the next iteration
                        # and by nothing else, which is what keeps a
                        # never-satisfied token from ever reaching a wait (the
                        # one way this class of bug turns from slow into a
                        # deadlock).  Guard (1): still inside `is_issuer`.
                        # Guard (2): written out, false exactly once per call.
                        if cutlass.const_expr(peek_ahead):
                            mm_more = (mm_t * num_k_blocks + mm_kb + 1
                                       < tiles_per_cta * num_k_blocks)
                            if mm_more:
                                mm_peek = ab_consumer.try_wait()
                        _seg("mma.stage.end", mm_t)
                    # The accumulator of this tile is complete once the last MMA
                    # retires; commit is what tells the epilogue so.  It has to
                    # be issued ON THE HANDLE `acquire_and_advance` handed back
                    # at the top of the tile: that call advanced the
                    # participant's own state one stage, so a bare `commit()`
                    # resolves against the NEXT buffer and arms acc_full[(t+1)&1]
                    # instead of acc_full[t&1] -- the epilogue of an even tile is
                    # then released only by the commit of tile t+1, one whole
                    # tile late, which halves the double buffer back to a single
                    # one.  It does not misnumber anything (tcgen05.commit
                    # tracks ALL prior tcgen05 work, so the data the epilogue
                    # eventually reads is complete) and it does not deadlock at
                    # an even tile count, so correctness and the trace both stay
                    # silent; it shows up only as the issuer's acc_empty wait and
                    # the epilogue's acc_full wait going double-peaked in
                    # alternation [hw_kb/cutedsl/sm100/pipeline_umma_async_handle_commit.md,
                    # GPU-certified job 1789416076].  `mm_acc_tok.index` already
                    # picks the TMEM columns above, so this is the one call-site
                    # that was reading the participant state instead of the
                    # handle -- both sides now name the same buffer sequence.
                    if cutlass.const_expr(mount_acc):
                        acc_producer.commit(mm_acc_tok)
                    _seg("mma.tile.end", mm_t)
                if cutlass.const_expr(mount_acc):
                    acc_producer.tail()

            # --- epilogue: warps 4-7 on BOTH CTAs (D0.11 / D0.12) -----------
            # Each CTA reads back its OWN 128 accumulator lanes, so this side
            # carries no leader guard: the pair's two halves own two adjacent
            # n-blocks of the same m-block and write disjoint columns of C.
            if cutlass.const_expr(mount_acc):
                # Same scope rule as the MMA frame above: the read-back frame is
                # built OUTSIDE `if is_epilogue`, because the stmatrix atom is
                # selected by `LayoutEnum.COL_MAJOR` and carries it, and a value
                # bound inside a dynamic `if` has to be expressible at runtime.
                # Pure metadata for every warp that does not use it, except the
                # two register fragments -- which is the third legal shape in
                # the KB note: allocate the fragment before the region, write
                # its contents inside.
                # The read-back frame is built once, off buffer 0's first
                # segment; only the TMEM ADDRESS differs between buffers and
                # between segments, and `partition_S` is pure indexing, so
                # the same thread slice re-partitions whichever tile it is
                # handed.  The tile is the certified plain rank-2 one -- 128
                # lanes x 16 columns -- not a divide of the accumulator.
                tSeg0 = cute.make_tensor(tmem_ptr, S.acc_seg_layout)
                t2r = _tmem_copy(S.t2r_atom, tSeg0)
                thr_t2r = t2r.get_slice(epi_tid)
                # The register fragment is sized from the DESTINATION
                # partition: `partition_S` of a TMEM copy is the tcgen05
                # datapath face and is NOT a per-thread element count.  What
                # this does hold is the 16 fp32 of the D0.14 segment budget.
                rAcc = cute.make_rmem_tensor(
                    thr_t2r.partition_D(tSeg0).shape, cutlass.Float32)
                if cutlass.const_expr(mount_epi):
                    # STSM with transpose: the fragment holds a COLUMN of
                    # the C-transpose block (one n, 16 m), and the staging
                    # slot wants a ROW of C (16 m, all n).  The transposing
                    # stmatrix is what turns that around inside the warp
                    # instead of through 16 scalar strided stores.
                    r2s_atom = _call_named(
                        _r2s_op,
                        layout_d=_layout_d, layout_D=_layout_d,
                        d_layout=_layout_d,
                        elem_ty_d=cutlass.BFloat16,
                        element_type_d=cutlass.BFloat16,
                        dtype_d=cutlass.BFloat16, d_dtype=cutlass.BFloat16,
                        elem_ty_acc=cutlass.Float32,
                        element_type_acc=cutlass.Float32,
                        dtype_acc=cutlass.Float32,
                        acc_dtype=cutlass.Float32,
                        cta_tile_shape_mnk=MMA_TILER,
                        epi_tile=S.epi_tile,
                        tiled_tmem_load=t2r, tiled_copy_t2r=t2r,
                        tiled_copy=t2r, copy_t2r=t2r)
                    # Both the constructor and the register-side spelling
                    # are chosen by MEASUREMENT against the fragment; see
                    # `_r2s_frame` for why neither is decidable by name.
                    r2s, thr_r2s, tSt0 = _r2s_frame(
                        r2s_atom, t2r, sEpi[None, None, 0], epi_tid,
                        _size_of(rAcc))
                    rC = cute.make_rmem_tensor(rAcc.shape, cutlass.BFloat16)
                    r2s_src = _r2s_src(r2s, thr_r2s, tSt0, rC)
                if cutlass.const_expr(mount_store):
                    # The store side reads the SAME tensor the stmatrix wrote.
                    # No second view: with a swizzle on the plane, a second
                    # mode tree over the same bytes is a chunk permutation, not
                    # an identity, so the descriptor was built from this face
                    # and gC arrives in its frame (n-tile, m-segment).
                    tCsC, tCgC = cpasync.tma_partition(
                        tma_c, 0, cute.make_layout(1),
                        _group_but_stage(sEpi), gC)

                if is_epilogue:
                    for ep_t in cutlass.range(n_iters):
                        # The two thread slices under body-local names, for the
                        # same reason as `mma_h` in the MMA chain: `partition_S`
                        # / `partition_D` are METHOD CALLS, and a receiver bound
                        # outside the region is conscripted into the region's
                        # carried list.  A `ThrCopy` built off the stmatrix atom
                        # carries the `LayoutEnum` that selected that atom, so
                        # carrying one is `TYPE_DYNAMIC_EXPR_UNSUPPORTED` at the
                        # `if`.  Rebinding here (a name assigned in the body is
                        # dropped from the list) keeps both slices read-only
                        # names of the enclosing scope.
                        t2r_thr = thr_t2r
                        _seg("epi.tile.begin", ep_t)
                        ep_idx = bidx + GRID * ep_t
                        # Tail wave: OOB iterations still drain acc_empty and
                        # run the LDTM/pack/stmatrix path (register work only),
                        # but skip the TMA store -- ep_in_range gates it below.
                        ep_in_range = ep_idx < num_tiles
                        ep_clamped = cutlass.Int32(ep_idx * ep_in_range)
                        ep_group = ep_clamped // group_span
                        ep_in_group = ep_clamped - ep_group * group_span
                        ep_m_block = ep_in_group // super_group
                        ep_n_block = ep_group * super_group + (
                            ep_in_group - ep_m_block * super_group)
                        _seg("epi.acc.begin", ep_t)
                        # D1.1 site 4 of 4: epilogue waits acc_full[b].  The
                        # longest-held wait of the three roles (the epilogue of
                        # tile t-1 finishes well inside the MMA chain of tile t)
                        # and the one where the spin arm's cost side is largest:
                        # 128 threads re-reading one barrier instead of sleeping.
                        # Which way that trades is the reading this arm buys.
                        # D1.2 audit, not peeked.  acc_full is armed by the
                        # issuer's `tcgen05.commit` MULTICAST to both halves of
                        # the pair, so unlike full[s] both CTAs' epilogues are
                        # eligible to ask and no leader guard belongs here --
                        # the KB's eligibility column is about which barrier is
                        # armed where, not about "consumer side always guards".
                        # Left at the point of use for the same reason as
                        # acc_empty: once per tile, and not satisfied when
                        # reached (it is the handoff from the MMA chain that
                        # just ended).
                        if cutlass.const_expr(spin_wait):
                            ep_ok = acc_consumer.try_wait()
                            while ep_ok == 0:
                                ep_ok = acc_consumer.try_wait()
                            ep_tok = acc_consumer.wait_and_advance(ep_ok)
                        else:
                            ep_tok = acc_consumer.wait_and_advance()
                        _seg("epi.acc.end", ep_t)
                        # Every segment of both buffers as ONE tensor at the
                        # TMEM base; the buffer and the segment are mode
                        # indices, not pointer offsets (see `acc_seg_all`).
                        tSegAll = cute.make_tensor(tmem_ptr, S.acc_seg_all)
                        for ep_seg in cutlass.range_constexpr(EPI_SEGS):
                            ep_slot = ep_seg % EPI_SLOTS
                            if cutlass.const_expr(mount_store):
                                # Wait until at most one store is still in
                                # flight, i.e. until the store that last read
                                # THIS slot has drained.  Unconditional on
                                # purpose: the staging ring runs across the tile
                                # boundary too, so segments 0 and 1 of tile t
                                # are the ones racing segments 12 and 13 of tile
                                # t-1.  With nothing pending the wait is free.
                                # Depth is the D1.4(a) lever, `STORE_WAIT_DEPTH`;
                                # 1 (double-buffered) is its starting point and
                                # the reference's, 0 is the control arm.
                                # Row = a counter that rises for the whole life
                                # of the CTA, not the segment index: a row that
                                # restarts every tile reads as a wrap and only
                                # the last tile's segments survive the table.
                                _seg("epi.slot.begin", ep_t * EPI_SEGS + ep_seg)
                                if is_epi_lead:
                                    _store_wait(STORE_WAIT_DEPTH, read=True)
                                cute.arch.barrier(
                                    barrier_id=EPI_BAR_ID,
                                    number_of_threads=EPI_THREADS)
                                _seg("epi.slot.end", ep_t * EPI_SEGS + ep_seg)
                            # Segment `seg` is the same rank-2 tile 16 TMEM
                            # columns further along.
                            tSeg = tSegAll[
                                (None, None, ep_seg, ep_tok.index)]
                            cute.copy(t2r, t2r_thr.partition_S(tSeg), rAcc)
                            cute.arch.fence_view_async_tmem_load()
                            if cutlass.const_expr(mount_epi):
                                r2s_thr = thr_r2s
                                # The pack terminates the fp32 fragment's live
                                # range: no fp32 copy survives it (D0.14 seg A).
                                rC.store(rAcc.load().to(cutlass.BFloat16))
                                cute.copy(
                                    r2s,
                                    _apply_r2s_src(r2s_src, r2s, thr_r2s, rC),
                                    r2s_thr.partition_D(sEpi[None, None, ep_slot]))
                            if cutlass.const_expr(mount_store):
                                # Makes the stmatrix writes visible to the async
                                # proxy the TMA reads through; without it the
                                # store can read the PREVIOUS contents of the
                                # slot, which is stale output, not an error.
                                _fence_async_proxy()
                                cute.arch.barrier(
                                    barrier_id=EPI_BAR_ID,
                                    number_of_threads=EPI_THREADS)
                                # One warp issues.  The copy atom elects a lane
                                # for itself, so wrapping it would strand the
                                # other 31; the commit is left unelected so that
                                # every lane keeps a self-consistent group
                                # counter -- the issuing lane's groups hold one
                                # store each and the rest are empty, which is
                                # what makes the wait above mean what it says
                                # without depending on which lane got elected.
                                # OOB tail iterations skip the store; the staging
                                # slot is still filled (stmatrix ran above) and
                                # the wait above is still counted, so the slot
                                # ring and the store-group counter stay balanced.
                                if is_epi_lead & ep_in_range:
                                    cute.copy(
                                        tma_c, tCsC[(None, ep_slot)],
                                        tCgC[(None, (ep_n_block,
                                                     ep_m_block * EPI_SEGS
                                                     + ep_seg))])
                                    _store_commit()
                        # Every one of the 128 threads has to arrive: the
                        # acc_empty count is 256 across the pair and one missing
                        # arrive stalls the issuer forever.
                        ep_tok.release()
                        _seg("epi.tile.end", ep_t)
                    if cutlass.const_expr(mount_store):
                        # Drain before the closing cluster barrier: a store still
                        # in flight is reading a staging slot this CTA is about
                        # to leave.
                        if is_epi_lead:
                            _store_wait(0, read=True)

        # --- drain: TMEM release, then the closing cluster barrier -----------
        # The permit is relinquished after the tile loop and before the release,
        # in the order the certified rung ran: a CTA that has relinquished may
        # not allocate again, so it may not sit between the constructor and the
        # allocation.  The release itself is unguarded (both halves of the pair
        # execute it) and preceded by a CTA-wide named barrier so that no thread
        # is still reading the accumulator when its columns go back.
        tmem_alloc.relinquish_alloc_permit()
        _seg("tmem.dealloc.begin")
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        cute.arch.barrier(barrier_id=TMEM_BAR_ID,
                          number_of_threads=CTA_THREADS)
        _call_named(tmem_alloc.free, _widen=False, _require=("tmem_ptr",),
                    tmem_ptr=tmem_ptr, num_columns=TMEM_COLS)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        _seg("tmem.dealloc.end")

    @cute.jit
    def entry(
        pAct: cute.Pointer,
        pWgt: cute.Pointer,
        pSFAct: cute.Pointer,
        pSFWgt: cute.Pointer,
        pBlkExp: cute.Pointer,
        pC: cute.Pointer,
    ):
        # Every gmem view is built HERE, inside the trace, from raw pointers.
        # Building them on the host is what the previous revision did, and a
        # cute constructor on a thread with no MLIR context on its stack fails
        # inside the binding layer as
        #
        #     TypeError: pack_shape(): incompatible function arguments.
        #       supported: pack_shape(shape: object, context: Context|None = None)
        #       Invoked with types: tuple
        #
        # -- an argument error about the argument the CALLER never passed, which
        # names neither the context nor the call site.  Handing the entry point
        # nothing but `cute.runtime.make_ptr` pointers also gives D0.2 the shape
        # it wants: the compiled entry is keyed by geometry, and a later call
        # with different buffers is a new pointer value, not a recompile.
        mAct = cute.make_tensor(
            pAct, cute.make_layout((m_total, k_depth), stride=(k_depth, 1)))
        # wgt reinterpreted as (N, K, G) e2m1: two elements per delivered byte,
        # low nibble first, which is the element order the Uint8 container plane
        # already wants -- no nibble swap anywhere (D0.8).
        mWgt = cute.make_tensor(
            pWgt, cute.make_layout((n_cols, k_depth, num_experts),
                                   stride=(k_depth, 1, n_cols * k_depth)))
        # Both scale planes are flat byte runs of whole blocked SF tiles; a TMA
        # box mode may not exceed 256 elements, so the run is carried as
        # (run/256, 256) with the contiguous mode second -- the box order the
        # certified flat-restage form uses.
        mSFAct = cute.make_tensor(
            pSFAct, cute.make_layout(
                (SF_ACT_SUB, 256, num_k_blocks, num_m_blocks),
                stride=(256, 1, SF_ACT_BYTES, SF_ACT_BYTES * num_k_blocks)))
        mSFWgt = cute.make_tensor(
            pSFWgt, cute.make_layout(
                (SF_WGT_SUB, 256, num_k_blocks, num_n_blocks, num_experts),
                stride=(256, 1, SF_WGT_BYTES, SF_WGT_BYTES * num_k_blocks,
                        SF_WGT_BYTES * num_k_blocks * num_n_blocks)))
        mBlkExp = cute.make_tensor(
            pBlkExp, cute.make_layout(num_m_blocks, stride=1))
        # C in the MMA frame: (MMA M = n of C, MMA N = m of C), n contiguous.
        # The buffer itself is the ordinary row-major C[m][n] the caller gets
        # back; this is the same bytes read with the mode order the swizzled
        # staging plane uses, which is what lets one descriptor serve both.
        mC = cute.make_tensor(
            pC, cute.make_layout((n_cols, m_total), stride=(1, n_cols)))

        # CtaGroup.TWO on every delivery op: it is what permits a completion
        # signal to land on the peer CTA's (here: the leader's) barrier.  Same
        # call shape as CtaGroup.ONE, invisible in review, visible only as the
        # .2CTA suffix in SASS -- and its absence is a hang, not a wrong answer.
        op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        mk = _tma_atom

        # The host trace's own copy of the shared metadata; the kernel builds
        # its own from the same function (see `_shapes`).
        S = _shapes()

        tma_act, gAct = mk(op, mAct, S.act_view_one,
                           (ACT_ROWS_PER_CTA, BLOCK_K), 1)
        tma_wgt, gWgt = mk(op, mWgt, S.wgt_view_one,
                           (BLOCK_N, BLOCK_K), 1, internal_type=cutlass.Uint8)
        tma_sf_act, gSFAct = mk(op, mSFAct, S.sf_act_flat_box,
                                (SF_ACT_SUB, 256), 1)
        tma_sf_wgt, gSFWgt = mk(op, mSFWgt, S.sf_wgt_flat_box,
                                (SF_WGT_SUB, 256), 1)
        # D0.12 writeback.  One descriptor, box = a whole 128 n x 16 m segment
        # of C in the staging plane's own frame, built from the one-slot slice
        # of the swizzled face itself; the store side needs no cta_group (each
        # CTA writes its own n-block).
        tma_c, gCt = _tma_atom(_store_op(), mC, S.epi_smem_one,
                               (BLOCK_N, EPI_SEG_M), 1)

        gemm_kernel(
            tma_act, cute.zipped_divide(gAct, (ACT_ROWS_PER_CTA, BLOCK_K)),
            tma_wgt, cute.zipped_divide(gWgt, (BLOCK_N, BLOCK_K)),
            tma_sf_act, cute.zipped_divide(gSFAct, (SF_ACT_SUB, 256)),
            tma_sf_wgt, cute.zipped_divide(gSFWgt, (SF_WGT_SUB, 256)),
            mBlkExp,
            tma_c, cute.zipped_divide(gCt, (BLOCK_N, EPI_SEG_M)),
        ).launch(
            grid=[GRID, 1, 1],
            block=[CTA_THREADS, 1, 1],
            cluster=[CLUSTER_M, 1, 1],
            smem=SMEM_BYTES,
        )

    return entry


def _s2t_op():
    """The smem -> TMEM copy op for scale factors (tcgen05.cp)."""
    for name in ("Cp4x32x128bOp", "Cp32x128bOp", "Cp128x128bOp", "Cp4x256bOp"):
        op = getattr(tcgen05, name, None)
        if op is None:
            continue
        try:
            return op(tcgen05.CtaGroup.TWO)
        except TypeError:
            return op()
    raise RuntimeError(
        "no tcgen05 smem->TMEM copy op found; looked for Cp4x32x128bOp and kin")


def _sf_slice(t, kp):
    """The scale-factor columns belonging to K=32 sub-block `kp`.

    The sub-block lives in the SF TMEM layout's trailing mode, so the coordinate
    is built from the layout's own rank rather than assuming one.
    """
    rank = cute.rank(t.layout)
    if rank < 3:
        return t
    return t[(None,) * (rank - 1) + (kp,)]


# ---------------------------------------------------------------------------
# Host side.
# ---------------------------------------------------------------------------

def _exponent_bytes(sf: torch.Tensor) -> torch.Tensor:
    """UE8M0 exponent byte of an exact power of two: the fp32 exponent field."""
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8)


def _sf_index_tensors(rows, k_sub, device):
    table = _sf_offsets(rows, k_sub)
    mk = lambda col: torch.tensor([t[col] for t in table], device=device,
                                  dtype=torch.long)
    return mk(0), mk(1), mk(2)


def _pack_sf_act(a_sf, num_m_blocks, num_k_blocks, k_sub):
    """act scales -> one blocked SF tile per (m_block, k_block).

    A token's 1x128 scale covers four K=32 blocks, so the same byte is
    replicated across the four sub-block slots (D0.9 / D0.2).
    """
    dev = a_sf.device
    eb = _exponent_bytes(a_sf)                       # [M_total, K/128]
    row_ix, _, off_ix = _sf_index_tensors(BLOCK_M, k_sub, dev)
    cos = _sf_tile_bytes(BLOCK_M, k_sub)
    m_base = torch.arange(num_m_blocks, device=dev, dtype=torch.long) * BLOCK_M
    grow = (m_base.view(-1, 1) + row_ix.view(1, -1)).clamp_(max=eb.shape[0] - 1)
    vals = eb[grow]                                  # [MB, rows*k_sub, KB]
    out = torch.zeros((num_m_blocks, num_k_blocks, cos), dtype=torch.uint8, device=dev)
    out[:, :, off_ix] = vals.permute(0, 2, 1)
    return out


def _pack_sf_wgt(b_sf, num_n_blocks, num_k_blocks, k_sub):
    """wgt scales -> one blocked SF tile per (expert, n_block, k_block)."""
    dev = b_sf.device
    g = b_sf.shape[0]
    eb = _exponent_bytes(b_sf).view(g, num_n_blocks, BLOCK_N, num_k_blocks, k_sub)
    row_ix, sub_ix, off_ix = _sf_index_tensors(BLOCK_N, k_sub, dev)
    cos = _sf_tile_bytes(BLOCK_N, k_sub)
    # Two advanced indices separated by a slice: torch moves their joint axis to
    # the front, so the result is [rows*k_sub, G, NB, KB].
    sel = eb[:, :, row_ix, :, sub_ix]
    out = torch.zeros((g, num_n_blocks, num_k_blocks, cos),
                      dtype=torch.uint8, device=dev)
    out[:, :, :, off_ix] = sel.permute(1, 2, 3, 0)
    return out


# --- D0.2: the one-time repack, keyed by input identity ----------------------
# The three artifacts below are pure functions of `(a_sf, b_sf, m_indices)` and
# of the launch geometry, so they are produced once per input identity and held.
# What makes that sound is the KEY: a data pointer alone is not an identity,
# because torch's caching allocator hands a freed tensor's address to the next
# tensor of the same size -- two different a_sf can key alike and the second one
# would silently be given the first one's scales.  So the key carries shape,
# stride, dtype, device and `_version` (torch's in-place write counter) as well,
# and the ENTRY holds a reference to the source tensors, which is what keeps the
# allocator from reissuing the address while the entry is alive.
#
# `_version` only sees torch's own in-place writes; a raw-pointer write behind
# torch's back is outside the contract (D0.2) and the bench does not do one.

_PREPARED: dict = {}
# The perf window rotates across four buffer sets and the two correctness cases
# are two more geometries; eight entries hold all of it, at roughly 9 MB each
# (4.9 MB of blocked sf_act + 4.2 MB of blocked sf_wgt + 592 B of expert table).
_PREPARED_MAX = 8


def _tensor_identity(t: torch.Tensor):
    return (int(t.data_ptr()), tuple(t.shape), tuple(t.stride()),
            str(t.dtype), str(t.device), int(getattr(t, "_version", 0)))


def _prepare(a_sf, b_sf, m_indices, num_m_blocks, num_n_blocks, num_k_blocks):
    """The blocked scale planes and the block -> expert table for these inputs."""
    key = (_tensor_identity(a_sf), _tensor_identity(b_sf),
           _tensor_identity(m_indices),
           num_m_blocks, num_n_blocks, num_k_blocks)
    hit = _PREPARED.get(key)
    if hit is not None:
        return hit

    prep = types.SimpleNamespace(
        # The block -> expert table samples the first row of every 224-row
        # block, which the alignment contract guarantees is a valid row; the
        # clamp is defensive only.
        blk_expert=m_indices[::BLOCK_M].to(torch.int32).clamp_min(0).contiguous(),
        sf_act=_pack_sf_act(a_sf, num_m_blocks, num_k_blocks, K_SUB),
        sf_wgt=_pack_sf_wgt(b_sf, num_n_blocks, num_k_blocks, K_SUB),
        # Held so that no source tensor's address can be recycled under a live
        # cache entry.  This is the correctness red line of D0.2, not a
        # lifetime convenience.
        sources=(a_sf, b_sf, m_indices))
    # The host repack lays the blocked SF bytes out at the analytic offsets the
    # device views assume; if its tile size ever disagreed with the DSL's own SF
    # plane every byte would land in the wrong place, which the court sees as a
    # K-periodic calc_diff rather than as a build error.  `_shapes` asserts that
    # correspondence inside the trace; this asserts the host half of it.
    for name, packed, want in (("sf_act", prep.sf_act, SF_ACT_BYTES),
                               ("sf_wgt", prep.sf_wgt, SF_WGT_BYTES)):
        if int(packed.shape[-1]) != want:
            raise RuntimeError(f"{name} tile is {int(packed.shape[-1])} B, "
                               f"the device view reads {want} B")

    while len(_PREPARED) >= _PREPARED_MAX:
        _PREPARED.pop(next(iter(_PREPARED)))
    _PREPARED[key] = prep
    return prep


def fp4_grouped_gemm(
    a: torch.Tensor,          # [M_total, K]        float8_e4m3fn, row-major (K = 4096)
    a_sf: torch.Tensor,       # [M_total, K/128]    fp32, per-token 1x128 UE8M0 scales (powers of two)
    b: torch.Tensor,          # [G, N, K/2]         int8, packed e2m1 pairs, row-major per expert (G=8, N=4096; NT: C_g = A_g @ B_g^T)
    b_sf: torch.Tensor,       # [G, N, K/32]        fp32, per-row 1x32 UE8M0 scales (powers of two)
    m_indices: torch.Tensor,  # [M_total]           int32, expert id per row; -1 = padding row
) -> torch.Tensor:
    """Returns c [M_total, N] bf16; row i = dequant(a[i]) @ dequant(b[m_indices[i]])^T.

    D0.1's window: ONE kernel, ONE launch.  Besides the launch the only work in
    here is the `torch.empty` that allocates c and two dict lookups -- the
    blocked scale planes and the expert table come from the D0.2 cache, and the
    compiled entry from the geometry-keyed compile cache.  The first call on a
    given set of inputs pays the repack; the bench's 20 warmup iterations over
    four rotating buffers pay it four times in total.
    """
    from cutlass.cute.runtime import make_ptr

    m_total, k = int(a.shape[0]), int(a.shape[1])
    g, n = int(b.shape[0]), int(b.shape[1])
    if n % BLOCK_N != 0:
        raise ValueError(f"N={n} is not a multiple of BLOCK_N={BLOCK_N}")
    if k % BLOCK_K != 0:
        raise ValueError(f"K={k} is not a multiple of BLOCK_K={BLOCK_K}")
    num_m_blocks = (m_total + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = n // BLOCK_N
    num_k_blocks = k // BLOCK_K
    if num_n_blocks % SUPER_GROUP != 0:
        raise ValueError(
            f"num_n_blocks={num_n_blocks} is not a multiple of "
            f"SUPER_GROUP={SUPER_GROUP}; the device path assumes full super-groups")

    # --- D0.2: expert table + blocked scales, cached on input identity -------
    prep = _prepare(a_sf, b_sf, m_indices,
                    num_m_blocks, num_n_blocks, num_k_blocks)

    # D0.1: the only allocation inside the timed window.  `empty`, not `zeros`:
    # every row of c is covered by exactly one tile (num_m_blocks * 224 >=
    # m_total and the store's box is clamped by the descriptor at the ragged
    # edge), so a pre-zeroed buffer would only hide a missing writeback.
    c = torch.empty((m_total, n), device=a.device, dtype=torch.bfloat16)

    # --- what crosses into the trace: pointers, nothing else ----------------
    # `make_ptr` is a host-side runtime constructor (it builds a JIT argument,
    # not IR), so it is legal here; `cute.make_tensor` / `cute.make_layout` are
    # not -- they emit into the current MLIR context and there is none on this
    # thread.  The layouts they would have carried are all static geometry, so
    # they are rebuilt inside the entry point from the closed-over ints.
    p_act = make_ptr(cutlass.Float8E4M3FN, a.data_ptr(),
                     cute.AddressSpace.gmem, assumed_align=16)
    p_wgt = make_ptr(cutlass.Float4E2M1FN, b.data_ptr(),
                     cute.AddressSpace.gmem, assumed_align=32)
    p_sf_act = make_ptr(cutlass.Uint8, prep.sf_act.data_ptr(),
                        cute.AddressSpace.gmem, assumed_align=16)
    p_sf_wgt = make_ptr(cutlass.Uint8, prep.sf_wgt.data_ptr(),
                        cute.AddressSpace.gmem, assumed_align=16)
    p_blk = make_ptr(cutlass.Int32, prep.blk_expert.data_ptr(),
                     cute.AddressSpace.gmem, assumed_align=16)
    p_c = make_ptr(cutlass.BFloat16, c.data_ptr(),
                   cute.AddressSpace.gmem, assumed_align=16)

    # D0.2's other half: ONE compile per launch GEOMETRY.  The key carries no
    # pointer, so a call on different buffers of the same geometry -- which is
    # every call of the rotating perf window -- reuses this entry and only
    # rebinds the six pointer arguments.
    key = (num_m_blocks, num_n_blocks, num_k_blocks, g, SUPER_GROUP, m_total, n,
           MOUNT_SHIPPED)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _build_entry(num_m_blocks, num_n_blocks, num_k_blocks, g, m_total),
            p_act, p_wgt, p_sf_act, p_sf_wgt, p_blk, p_c)
        _COMPILED[key] = compiled
    compiled(p_act, p_wgt, p_sf_act, p_sf_wgt, p_blk, p_c)
    return c
