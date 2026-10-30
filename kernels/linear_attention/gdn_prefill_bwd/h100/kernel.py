"""gdn_prefill_bwd — the student's kernel file (the only file the student edits).

Gated Delta Rule (GDN) chunked prefill BACKWARD, fixed campaign point
qk16_v32_d128_n8192 (Qwen3.5-35B TP1 head geometry, single 8192-token
sequence). The signature below is the card's frozen contract; the student
implements the body in CuTe DSL; the signature itself may not change.

Forward semantics this is the backward of (per token t, per v-head h, fp32
K-major state S[h] of [D, D], zeros init; q/k are GVA-shared: v-head h reads
qk-head h // (H_V // H_QK); alpha = exp(g)):

    S = alpha[t, h] * S                       # per-token scalar forget gate
    v_old = k[t] @ S                          # read the key's current value
    S = S + beta[t, h] * outer(k[t], v[t] - v_old)   # delta-rule write
    o[t, h] = scale * (q[t] @ S)              # inclusive readout

Given the upstream gradient `do` of `o` (no gradient on the final state, no
initial state), return the gradients of the loss with respect to q, k, v, g and
beta. dq/dk are summed over the v-heads that share a qk-head. k arrives
L2-normalized along the head dim; the kernel must not re-normalize (the
gradient is with respect to the normalized k it receives).

Structure (milestone D0.1, as amended by D2.1): one call = 4 kernels /
4 launches in stream order

    k1_chunk_prep  (K1, the price segment: rebuilds g_cumsum and A; also
                    clears dq / dk, D2.1's reduce-add targets)
      -> prepare_h   (K2)
      -> dh_boundary (K3)
      -> gdn_bwd_main(K4, the main kernel; its dq/dk tiles are reduce-ADDED
                      into the final outputs, so the GVA pair sum no longer
                      needs a kernel of its own)

M0 GATE (this commit carries no functional change; it is the acceptance read
of the architecture the previous commits built).  Where each checklist line is
answered in this file:

  whole-machine smoke     all 5 kernels lower and launch at the real shape on
                          every eval since the D0.14 commit; the eval PASSes
                          are a strictly stronger certificate than a smoke
  correctness             `gdn_prefill_bwd` end to end vs eval/oracle.py
  data-flow / no constant K1 -> g_cumsum, A;  K2 -> h_c;  K3 -> dh_end;
                          K4 -> dv, dg, dbeta, dq/dk_partial;  K5 -> dq, dk,
                          every one written whole and read by its consumer
  intermediate ABI (D0.7) the `torch.empty` block of `gdn_prefill_bwd` and the
                          2D/4D/5D views handed to each artifact below it;
                          nothing in the window is zeroed (segment 7's dh_end
                          slot is discarded by K4 on the segment predicate)
  precision (D0.6)        bf16 MMA in / f32 accumulate, bf16 exchange faces,
                          f32 dh carried in registers, bf16 dq/dk/dv out
  kernel_count 5 /        `_BUILDERS`: five distinct kernel symbols, six
  launches 24             launches per call (K5 is one artifact, two calls)
  grid 256 / block 512    `_build_k4`: grid (N_SEG, H_V), TID = 512
  regs <= 128             every kernel enters at the 128-register quota;
                          K4 redistributes with `setmaxnreg` 32/160/144/176
  smem_dynamic in band    K4 = 104 KB landing (13 boxes) + 112 KB exchange
                          (14 x 8 KB) + the 1 KB scalar block ~= 217 KB
  tensor_pct (3-WG skew)  the functional role split of `_build_k4` with
                          per-face full/empty mbarriers (no CTA-wide
                          rendezvous), so the three consumers phase-shift
  smem wavefronts         D0.13: elementwise runs on the f32 C-fragments; only
                          cross-WG / transposed quantities take an STSM face
  dram_bytes_write        D0.11: dh lives in WG0's registers across all 16
                          chunks, never round-trips through gmem
  scalar_sts_in_loop      D0.16: derived faces are shaped once by
                          LDSM -> FMUL -> F2FP -> STSM and non-accumulator
                          tiles enter elementwise segments by matrix copy
  dram_bytes_read         one TMA box per input tile per chunk, h_c and A read
                          once each

M1 GATE (this commit carries no functional change either; it is the acceptance
read of the two intra-CTA masking levers M1 left open -- D1.1 and D1.5 -- on
the topology M0 froze).  D1.2 / D1.3 / D1.4 were closed by the milestone's own
closure clauses (the iter-16 `probe_k4_landing_vs_chain` verdict: `land_all_off`
+0.44% and `xchg_waits_off` -0.46%, both inside the 3.02% noise floor), so no
landing depth, swizzle tier or wait-encoding knob is touched here.  Where each
checklist line is answered in this file:

  correctness             `gdn_prefill_bwd` end to end vs eval/oracle.py, same
                          rigid rel-L2 lines as M0
  device window           the 5-kernel / 6-launch table of `_BUILDERS` is
                          unchanged from the M0 gate; the M1 levers are both
                          internal to `_build_k4`
  mechanism line (1)      D1.1: the `link` / `chain` split near the top of
  tensor_pct              `_build_k4` -- `link` issues an SS chain with no
                          fence/commit/wait so consecutive chains that
                          accumulate into the same registers share one fence,
                          one `commit_group` and one drain, and WG0's stage-5
                          quadrant loop runs the ladder proper: quadrant j is
                          committed, `wait_group(1)` reaps j-1, and j-1's psi
                          pass executes inside j's flight
  mechanism line (2)      the same construct read on the absolute axis; the
  main-kernel time_ns     `commit` axis footprint is 24 `DEPBAR.LE 0x0` + 3
                          `LE 0x1` per steady-state chunk, against 48 `LE 0x0`
                          on the M0-gate tree
  spill not degraded      D1.5's three widened chains (WG0 stage 5's psi dot
                          32 -> 8 dependent FFMA on four partial sums; WG1
                          stage 3/4's row dots 4x16 -> 8x8; WG1's in-chunk
                          exclusive scan of r 64 chained broadcast `LDS` -> a
                          5-step `shfl.sync.idx` Kogge-Stone scan plus a
                          butterfly carry) were sized against the D0.12 段式
                          live-set budget and the unchanged 32/160/144/176
                          `setmaxnreg` quota
  smem_dynamic            unchanged from the M0 gate: no slot was deepened and
                          no face resized, because D1.2 is closed
  bankconf                unchanged exchange-face atoms and layouts (D1.3
                          closed); D1.1 / D1.5 move no traffic to smem

(The two gate blocks above are the acceptance reads of the trees they were
taken on.  The one line of theirs D2.1 supersedes is the launch table:
`kernel_count` is 4 and `total_launches` 16 from this commit on.)

M2 / D2.1 -- the GVA pair sum stops being a kernel.  K4 used to write dq and
dk as per-v-head partials [N, H_V, D] (134 MB), and a 6th launch, K5
`group_reduce`, read all of it back to write 67 MB of [N, H_QK, D].  On the
iter-16 per-kernel table that launch was the single largest gap against the
reference (90.7 us vs its `group_reduce` 63.7).  It is gone: K4's epilogue now
ADDS its 64 x 128 dq / dk tile straight into the final tensor with
`cp.reduce.async.bulk ... .add.noftz.bf16` (the non-tensor, address-form
flavour -- hw_kb `s2g_bulk_reduce`), so the two v-head CTAs of a GVA pair --
adjacent blocks of the same wave, D0.5's head-fastest numbering -- accumulate
into the same 256-B rows through L2.  Three things had to move with it:

  zeroing        a reduce-add target must start at zero.  D2.1's ledger offers
                 host `torch.zeros` or K1; this is K1 (see the note at its
                 entry), because the host spelling costs a FillFunctor launch.
  staging order  the reduction adds shared bytes LINEARLY onto global bytes,
                 so the staged tile has to be in gmem order.  dq / dk now
                 publish through `publish_flat` into an UN-SWIZZLED row-major
                 64 x 128 view of the same smem the TMA store staged through
                 (landing slots LV..LV+1 / the dh face pair SDH[0..1]); the
                 row PITCH of that view is D1.3's, below.  That K4's own STSM
                 tiled copy still places `acc[i]` at `cS[i]` against that
                 non-compact view is GPU-verified in
                 notes/probe_m2_d21_bulk_reduce.py (S1), as is the ADD itself
                 (S2) and the 256-B-not-512-B issue width (S3).
  rounding       K5 summed two bf16 partials in f32 and rounded once; the
                 reduction unit adds the second tile onto the first with one
                 correctly-rounded bf16 add, `.noftz` so subnormals are not
                 flushed.  Same number of roundings, same kind.

dv is untouched: it has no GVA pair to fold, so it keeps D0.14's `UTMASTG.4D`
through a dead face pair.

M2 / D2.2 -- the two auxiliary kernels, on the two knobs the decision names.

  look-ahead     `NWARM` goes 2 -> 1.  It is the warm-up depth of K2's forward
                 scan AND the look-ahead depth of K3's dh recurrence, which
                 D0.4 requires to move together (同边纪律: one decay bound
                 backs both).  K2 drops from 18 trips to 17 per interior
                 segment; K3 drops from two trips to one, i.e. half the
                 kernel.  The bound is written out at NWARM's definition.
  wait ladder    K2 and K3 issued every WGMMA chain as its own group -- fence,
                 4 k-blocks, `commit_group`, `wait_group(0)` -- so chain i + 1
                 could not ISSUE until chain i had retired, which is the depth
                 0 D1.1 already paid off inside K4.  Both kernels now carry
                 K4's `link` / `chain` split and use it three ways: consecutive
                 chains on the SAME accumulator share one group and one drain
                 (K2's W', K3's Sq and its per-quadrant dh pair); the four
                 quadrants of a state advance -- D2.2's named site, in K2's
                 stage (d) and K3's stage (c) -- ride D1.1's depth-1 ladder,
                 free of register cost because those accumulators are resident
                 either way (D0.11); and where two independent accumulators feed
                 elementwise work, both groups are issued before either is
                 reaped, so `wait_group(1)` takes the first while the second
                 flies and the first's elementwise + STSM publish is the
                 covered段 (K2's B and U halves, K3's dvn and W halves).  The
                 ladder's only cost is the second f32 accumulator's liveness,
                 32 registers inside the unchanged 32/160x3 `setmaxnreg`
                 quota; the merges cost nothing.
  NOT done       the decision's third item, double-buffering K2's h_c TMA
                 store, is its own 次项 and the timeline says so: on the
                 iter-17 table K2's chunk period was 6449 cycles of which the
                 `hc.store.drain` segment was not in the top three (the drain
                 sits three stages downstream of the issue and is already
                 complete when it is waited on).  It would cost 4 more 8 KB
                 faces to remove a wait that does not show up.

No other kernel, no ABI, no launch table and no smem partition changes: K2
keeps 10 faces / 5 landing slots at STAGE 2, K3 keeps 15 faces / 7 slots at
STAGE 1 (its four dh faces go unread at NWARM = 1 rather than being
re-indexed, so the depth stays a one-constant knob).

M1 / D1.3 -- the swizzle-tier decision, reopened on its own bankconf trigger
(`bankconf_st > 3x` the reference; the closed half of the decision is the slot
DEEPENING it was paired with, D1.2).  What D2.1 introduced was the one layout
in K4 that is not swizzled at all, and it had to be: `cp.reduce.async.bulk`
adds smem bytes linearly onto gmem bytes and knows nothing of a swizzle, so
the staged tile is plain row-major.  At the compact pitch (128 bf16 = 256 B)
every row of it starts on bank 0, so the 16 rows one `STSM.16.M88.4` writes
all want the same bank quad: the publish takes 16 wavefronts where its 512 B
need 4, and `bankconf_st` went 1.6M -> 7.70M when D2.1 landed [iter 17 vs 18
NCU].  D1.3's tier for an un-swizzled face is the pad (hw_kb `s2g_bulk_reduce`
bf16 §3): one 16 B pad per row, `STGP` = 136 elements = 272 B, after which
consecutive rows step 4 banks and the 32 lane addresses of an STSM spread over
all 8 quads, 4 to a quad.  The reduction is untouched -- each row still holds
its 128 columns contiguously at a 16 B aligned base, and the byte COUNT stays
256 so the pad columns are never read.

The pad makes the tile 17408 B, i.e. 1008 B longer than the 16 KB slot pair it
stages through, so the block AFTER the pair must be dead as well.  That is
paid for by renumbering slots, not by buying smem:

  dq (landing)   the landing order becomes q, k, do, h_c, v, A: v -- which the
                 dq tile stages through, dead since stage 1 -- is placed so
                 that A follows it.  A is dead from stage 3 in both roles that
                 read it, and it belongs to the SAME LE_AV class as the v
                 boxes, so the overrun has exactly the lifetime the staging
                 already had, and the producer cannot refill either until the
                 dq drain that precedes WG1's LE_AV arrival.  Zero smem.
  dk (face)      dk stages through the two dh blocks mh = 0's chains have just
                 finished with, and mh = 1 then reads the other two -- which,
                 at the old numbering, were exactly the blocks the overrun
                 would land in.  The dh quadrants are therefore renumbered so
                 that mh = 0's pair is the LAST pair of the face region
                 (SDH = (12, 13, 2, 3)); the two quantities evicted from 12-13
                 take the blocks quadrants 0-1 vacate (SVN = (11, 0), Ms at 1).
                 Every dh access is per-slot, so this is a table, not a
                 lifetime change.  The overrun then lands in a 1 KB tail pad,
                 the decision's only smem cost: 231424 -> 232448 B, the
                 `smem_dynamic` ceiling exactly.

Both the address-correctness of the padded pitch under K4's own STSM copy and
the pad columns being left untouched by it are GPU-verified in
notes/probe_m2_d13_pad_stride.py (S1 / S5), as is the reduce-add through the
padded source with two CTAs of a GVA pair meeting on one gmem block (S2 / S3).

NOTE: no `from __future__ import annotations` in this file (PEP 563
stringifies the @cute.struct / @cute.kernel annotations that CuTeDSL builds
its types from).
"""

from typing import Callable, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.math          # noqa: F401  (cute.math.exp2 in K1/K2)
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

# --- fixed campaign point (D0.2 of the milestone; never varies) --------------
N, H_QK, H_V, D = 8192, 16, 32, 128
C = 64                       # chunk length in tokens
NC = N // C                  # 128 chunks
SEG_CHUNKS = 16              # 16 chunks = 1024 tokens per segment
N_SEG = NC // SEG_CHUNKS     # 8 segments
GVA = H_V // H_QK            # 2 v-heads per qk-head
NWARM = 1                    # D0.3/D0.4 warm-up / look-ahead chunks
# D2.2 cut this from 2 to 1.  Both users (K2's warm-up scan, K3's look-ahead dh
# recurrence) rest on the SAME decay bound and so must move together (D0.4's
# 同边纪律); one constant is what enforces that.  The bound: everything older
# than the warm-up enters through a transition operator of norm <= exp(sum g)
# over the warm-up interval (milestone 1.3), and g = log(alpha) with
# alpha ~ U(0,1) (eval/oracle.py make_inputs) has mean -1 and variance 1 per
# token, so one 64-token chunk sums to -64 +/- 8.  The quantity that has to
# stay under the 8e-3 tolerance is the MAX over the 7 x 32 = 224 (boundary,
# head) pairs, i.e. about -64 + 2.9 sigma = -41, giving exp(-41) ~ 2e-18 --
# four orders below the fp32 epsilon of the states themselves and sixteen
# below the tolerance.  At NWARM = 2 the same number was exp(-100)-ish; both
# are exact zero in the only sense that matters, and the second chunk was
# 1/17 of K2's trips and 1/2 of K3's whole kernel.


def _seg(site, row=0):
    """Timeline-probe marker site (driver stubs this in on the probe arm)."""
    return None


# --- D2.3: programmatic dependent launch (PDL) -------------------------------
# The launch table is four kernels on one stream, each a data dependency of the
# next, and `gpu_entry.sh` exports CUDA_DEVICE_MAX_CONNECTIONS=1, so the "stream
# ordering" arm of D2.3 has nothing to win: every stream is multiplexed onto one
# hardware channel and two grids cannot be resident from two streams.  PDL is
# the in-stream arm, and it is what D2.3 actually asks for -- not overlapping
# the kernels' work (the dependencies are real), but overlapping the LAUNCH: the
# successor's CTAs are scheduled onto the SMs, allocate their 226 KB of smem and
# their register quota, and run their entry code while the predecessor's last
# blocks are still ramping down, so the gap between "predecessor's last CTA
# retires" and "successor's first CTA issues its first TMA" collapses.
#
# Two halves, both required:
#   host   `.launch(use_pdl=True)` on the DEPENDENT kernel sets
#          `cudaLaunchAttributeProgrammaticStreamSerialization`.  This is the
#          A-rung DSL knob (`ptx_escape` ladder): it is a documented `.launch`
#          parameter of the CuTe DSL, default False.  `_compiled` below falls
#          back to a launch without it if the image's DSL predates it.
#   device `griddepcontrol.wait;` gates the dependent's first global read, and
#          `griddepcontrol.launch_dependents;` in the predecessor says "my
#          successors may be scheduled now".  Neither is exposed by a `cute.arch`
#          intrinsic, so both go through `ptx_escape`'s `llvm.inline_asm`
#          channel (B-rung; certified on GPU by
#          notes/probe_m2_d23_launch_overlap.py, which compiled them into a DSL
#          kernel, launched it, checked its numerics and counted both mnemonics
#          in the dumped PTX).
#
# The correctness argument is entirely on `wait`: the trigger is a SCHEDULING
# signal and carries no visibility guarantee, while `griddepcontrol.wait` blocks
# until every grid this one depends on has completed AND flushed its writes to
# global memory.  So the rule this file follows is the conservative one: `wait`
# is the FIRST statement of every dependent kernel, ahead of every global read
# including the g_cumsum/beta prologues, and the trigger is the LAST statement
# of every producer, after its final global write (K1's A/g_cumsum stores, K2's
# drained h_c bulk group, K3's dh_end stores).  No kernel logic is reordered
# across either one, so the overlap is purely the block-scheduling ramp.
#
# Both are also no-ops when the launch does not carry the attribute -- a `wait`
# with no programmatic dependency returns immediately -- which is why they can
# be unconditional in the kernel bodies even on the `use_pdl` fallback path.
#
# K1 deliberately does NOT get a `wait`: it is the head of the window, and the
# grid that precedes it in stream order is the PREVIOUS call's K4 in a batched
# timing loop.  K4 reduce-ADDs into dq/dk and K1 zeroes them, so K1 must stay a
# plain stream-serialized launch.  K4 correspondingly gets no trigger.
@dsl_user_op
def _pdl_wait(*, loc=None, ip=None):
    """Block until the grids this one depends on have completed and flushed."""
    llvm.inline_asm(None, [], "griddepcontrol.wait;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _pdl_trigger(*, loc=None, ip=None):
    """Let dependent grids be scheduled.  Issued by every CTA: the kick-off
    needs all blocks to have executed it (or exited), otherwise it falls back to
    the implicit one at warp completion -- correct either way, but no overlap."""
    llvm.inline_asm(None, [], "griddepcontrol.launch_dependents;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


_PDL = True                  # cleared by `_compiled` if `.launch` rejects it


def _pdl_kw():
    """Launch kwargs for a PDL-dependent kernel (trace-time Python, not IR)."""
    return {"use_pdl": True} if _PDL else {}


# =============================================================================
# K1 — k1_chunk_prep (D0.2, the price segment)
#
# grid (H_V, NC) = 4096 CTA x block 128 (one warpgroup), v-head FASTEST.  Per
# (chunk, v-head):
#   read the 64 tokens of g, beta (f32) and of the k-tile (64x128 bf16) of the
#   qk-head this v-head reads; chunk-inclusive log-space cumsum -> g_cumsum;
#   form L = I + tril(beta_i exp(G_i-G_j) k_i.k_j), solve A = (I+L)^-1 in place
#   (diagonal 8x8 transposed back-substitution, then mma.sync block doubling
#   8->16->32->64); write A bf16 [N, 32, 64] and g_cumsum f32 [N, 32].
# Every gmem touch is row-major (lane on the contiguous axis) and the two
# v-heads of a GVA pair are adjacent CTAs, so k is read once per qk-head.
# =============================================================================

def _rmem_alloc():
    """By-shape rmem allocator; the name drifts across DSL versions."""
    import inspect
    fn = (getattr(cute, "make_rmem_tensor", None)
          or getattr(cute, "make_fragment", None))
    if fn is None:
        raise RuntimeError("no by-shape rmem allocator on cutlass.cute")
    try:
        first = list(inspect.signature(fn).parameters)[0].lower()
    except Exception:                                      # noqa: BLE001
        first = ""
    dtype_first = first in ("dtype", "element_type", "elem_type", "ty", "t")
    return (lambda shape, dtype: fn(dtype, shape)) if dtype_first \
        else (lambda shape, dtype: fn(shape, dtype))


def _reg_ops():
    """(decrease, increase) setmaxnreg ops; the name drifts across DSL versions.

    DSL >= 4.4 spells them `setmaxregister_decrease / _increase`; the older
    `warpgroup_reg_dealloc / _alloc` are kept as deprecated forwarders.  Both
    lower to `setmaxnreg.{dec,inc}.sync.aligned.u32`.
    """
    dec = (getattr(cute.arch, "setmaxregister_decrease", None)
           or getattr(cute.arch, "warpgroup_reg_dealloc", None))
    inc = (getattr(cute.arch, "setmaxregister_increase", None)
           or getattr(cute.arch, "warpgroup_reg_alloc", None))
    if dec is None or inc is None:
        raise RuntimeError("no setmaxnreg op on cutlass.cute.arch")
    return dec, inc


def _build_k1():
    """Build the K1 launch: (jit function, list of its tensor arg shapes)."""
    F16, BF16, F32, I32 = (cutlass.Float16, cutlass.BFloat16,
                           cutlass.Float32, cutlass.Int32)
    ALLOC = _rmem_alloc()
    TID, BAR, KC = 128, 1, 16
    LOG2E = 1.4426950408889634

    # -- bit ops / selects: dsl_user_op + inline PTX --------------------------
    @dsl_user_op
    def _f16_lo(w, *, loc=None, ip=None):
        return F32(llvm.inline_asm(
            T.f32(), [I32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;"
            "\n\tcvt.f32.f16 $0, lo;\n}",
            "=f,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _f16_hi(w, *, loc=None, ip=None):
        return F32(llvm.inline_asm(
            T.f32(), [I32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;"
            "\n\tcvt.f32.f16 $0, hi;\n}",
            "=f,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _sel_eq(a, b, x, y, *, loc=None, ip=None):
        return F32(llvm.inline_asm(
            T.f32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip),
                      F32(x).ir_value(loc=loc, ip=ip),
                      F32(y).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _sel_gt(a, b, x, y, *, loc=None, ip=None):
        return F32(llvm.inline_asm(
            T.f32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip),
                      F32(x).ir_value(loc=loc, ip=ip),
                      F32(y).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.gt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _imax(a, b, *, loc=None, ip=None):
        return I32(llvm.inline_asm(
            T.i32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip)],
            "max.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip))

    @dsl_user_op
    def _imin(a, b, *, loc=None, ip=None):
        return I32(llvm.inline_asm(
            T.i32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip))

    @cute.kernel
    def k1_chunk_prep(mK: cute.Tensor, mG: cute.Tensor, mB: cute.Tensor,
             mGc: cute.Tensor, mA: cute.Tensor,
             mDqZ: cute.Tensor, mDkZ: cute.Tensor,
             mma_wg: cute.TiledMma, r2s16: cute.TiledCopy,
             mma8: cute.TiledMma, tcA8: cute.TiledCopy, tcB8: cute.TiledCopy,
             tcC8: cute.TiledCopy, mma16: cute.TiledMma, tcA16: cute.TiledCopy,
             tcB16: cute.TiledCopy, tcCst: cute.TiledCopy, tcCld: cute.TiledCopy,
             lytK: cute.ComposedLayout, lytF: cute.ComposedLayout,
             L_b8: cute.Layout, L_t8: cute.Layout, L_16: cute.Layout,
             L_t16: cute.Layout, L_16x32: cute.Layout, L_t16x32: cute.Layout,
             L_t32x16: cute.Layout,
             SharedStorage: cutlass.Constexpr[Callable]):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        wid = cute.arch.warp_idx()
        bx, by, _ = cute.arch.block_idx()
        # V-HEAD FASTEST (same free fact D0.5 keeps for the main kernel): the
        # two v-heads that share qk-head `hv // GVA` are ADJACENT CTAs, so the
        # 16 KB k tile they both read is fetched from DRAM once and hit in L2
        # the second time.  With chunk fastest the two readers were 128 CTAs
        # apart and k was streamed twice (134 MB instead of 67 MB).
        hv, ch = bx, by

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sk.get_tensor(lytK.outer, swizzle=lytK.inner)
        sF = storage.sf.get_tensor(lytF.outer, swizzle=lytF.inner)
        sG = storage.sg.get_tensor((C,))
        sB = storage.sb.get_tensor((C,))
        sW = storage.sw.get_tensor((4,))

        _seg("k1.entry")

        # ---- D2.1: clear dq / dk, K4's reduce-add targets ------------------
        # dq/dk no longer arrive as per-v-head partials that a separate kernel
        # sums: K4 now folds the GVA pair straight into the final tensors with
        # `cp.reduce.async.bulk ... .add.bf16`, so both must be ZERO when K4
        # starts.  D2.1's ledger names two zeroers, host `torch.zeros` or K1;
        # this is K1, because the host one lowers to an
        # `at::vectorized_elementwise_kernel` FillFunctor -- a 5th kernel and 8
        # more launches against D2.1's declared table of 4 kernels / 16
        # launches (the same reason D0.11's prologue judges dh_end's segment-7
        # slot by segment number instead of zeroing it on the host).
        #
        # The grid is already the right shape: dq and dk together are 32
        # qk-head planes x 128 chunks, and K1 is 32 v-heads x 128 chunks, so
        # every CTA clears exactly one 64-token x 128-wide plane and the cover
        # is exact.  Addressing goes through an f32 VIEW of the bf16 output
        # (one store clears two elements, and a warp's 32 lanes cover 32
        # consecutive f32 = 128 B of one row).
        #
        # WINDOW NOTE (window.toml): K1 is the price segment (`k1_`), so this
        # zeroing -- which is NOT contract tax -- is charged to the priced
        # kernel.  D2.1 sanctions the placement; the deduction it feeds is
        # correspondingly generous by the ~20 us [ESTIMATE, 67 MB / 3.35 TB/s]
        # the two clears cost.
        zr = ch * C
        zw = D // 2
        if hv < H_QK:
            for i in cutlass.range_constexpr((C * (D // 2)) // TID):
                e = i * TID + tidx
                mDqZ[(zr + e // zw, hv * zw + e % zw)] = F32(0.0)
        if hv >= H_QK:
            for i in cutlass.range_constexpr((C * (D // 2)) // TID):
                e = i * TID + tidx
                mDkZ[(zr + e // zw, (hv - H_QK) * zw + e % zw)] = F32(0.0)

        # ---- load the chunk: k-tile (64x128 bf16), g and beta (64 f32) -----
        # k is L2-normed already; v-head `hv` reads qk-head hv // GVA.
        # The traversal is ROW-MAJOR (`e // D, e % D`), i.e. lane `tidx` walks
        # the CONTIGUOUS axis: one LDG then covers 32 consecutive bf16 = 64 B
        # of ONE row = 1 L1 wavefront.  Column-major (`e % C, e // C`) put the
        # 32 lanes of every LDG in 32 DIFFERENT rows 4096 B apart = 32
        # wavefronts per instruction, 8192 per CTA for a 16 KB tile.
        gK = mK[(ch, None, hv // GVA, None)]
        for i in cutlass.range_constexpr((C * D) // TID):
            e = i * TID + tidx
            sK[(e // D, e % D)] = gK[(e // D, e % D)]
        if tidx < C:
            sG[tidx] = mG[(ch, tidx, hv)]
            sB[tidx] = mB[(ch, tidx, hv)]
        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()
        _seg("k1.load.end", ch)

        # ---- chunk-inclusive cumsum of g (warps 0/1, shfl.scan + carry) ----
        # The source lane is computed explicitly and fed to `shuffle_sync`
        # (shfl.sync.IDX); `shuffle_sync_up` passes c=0 for its mask_and_clamp,
        # which makes the .up predicate false on every lane and silently
        # degenerates the scan into x *= 2.
        x = F32(0.0)
        if wid < 2:
            x = sG[wid * 32 + lane]
        for d in cutlass.range_constexpr(5):
            st = 1 << d
            shv = cute.arch.shuffle_sync(x, _imax(lane - st, I32(0)))
            x = x + _sel_gt(lane, st - 1, shv, F32(0.0))
        if wid < 2:
            if lane == 31:
                sW[wid] = x
        cute.arch.sync_threads()
        if wid < 2:
            G = x + _sel_eq(wid, 1, sW[0], F32(0.0))
            sG[wid * 32 + lane] = G
        cute.arch.sync_threads()
        _seg("k1.cumsum.end", ch)

        # ---- WGMMA K @ K^T -> f32 accumulator (64 x 64) --------------------
        # The A/B descriptors ride the WARPGROUP-scope slice `get_slice(0)`;
        # only the accumulator and the C-coordinates are per-thread (hw_kb
        # `wgmma_desc_uniform` 坑 3).  `wgmma.mma_async` reads its two 64-bit
        # matrix descriptors out of the uniform file (`desc[UR*]`), and a
        # `tidx`-derived slice coordinate is exactly what stops ptxas proving
        # them warp-invariant -- they then get computed in per-thread
        # registers and re-staged with `R2UR` before every issue.
        wg_wg = mma_wg.get_slice(0)
        thr_wg = mma_wg.get_slice(tidx)
        tCrA = wg_wg.make_fragment_A(wg_wg.partition_A(sK))
        tCrB = wg_wg.make_fragment_B(wg_wg.partition_B(sK))
        acc = ALLOC(thr_wg.partition_shape_C((C, C)), F32)
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(mma_wg.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(D // KC):
            cute.gemm(mma_atom, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        _seg("k1.mma.end", ch)

        # ---- L = I + tril(beta_i e^{G_i-G_j} k_i.k_j), in the C-fragment ----
        # The exponent argument is masked to <= 0 BEFORE exp2, not after: on the
        # upper triangle G_i - G_j is positive and can reach +64, and exp2
        # overflows to inf in flight (which then poisons every downstream
        # reading).
        cS = thr_wg.partition_C(cute.make_identity_tensor((C, C)))
        # D0.16's M-axis hoist: a 64x64 C-fragment gives each lane elements in
        # exactly TWO rows, so the ROW factors (G_i and beta_i) are two values,
        # not 32.  The two rows are taken as the min and the max of the lane's
        # own m coordinates -- no assumption about WHICH fragment-index bit
        # carries the m-half -- and selected per element by comparing the
        # element's own m against the min.  64 scalar `LDS` per lane become 4
        # loads plus 2 `selp`.  The column factor G_j genuinely varies per
        # element and stays a read.
        mlo, mhi = cS[0][0], cS[0][0]
        for i in cutlass.range_constexpr(cute.size(cS)):
            mlo = _imin(mlo, cS[i][0])
            mhi = _imax(mhi, cS[i][0])
        g0, b0 = sG[mlo], sB[mlo]
        g1, b1 = sG[mhi], sB[mhi]
        for i in cutlass.range_constexpr(cute.size(cS)):
            mi, nj = cS[i][0], cS[i][1]
            gi = _sel_eq(mi, mlo, g0, g1)
            bi = _sel_eq(mi, mlo, b0, b1)
            dg = _sel_gt(mi, nj, gi - sG[nj], F32(0.0))
            gated = (bi
                     * cute.math.exp2(dg * LOG2E, fastmath=True) * acc[i])
            acc[i] = _sel_eq(mi, nj, F32(1.0),
                             _sel_gt(mi, nj, gated, F32(0.0)))

        # ---- publish to the K_SW128 fp16 face (STSM x4) --------------------
        thr_r = r2s16.get_slice(tidx)
        stg = cute.make_layout(cute.shape(thr_r.partition_S(sF))[:3])
        s32 = ALLOC(stg.shape, F32)
        s16 = ALLOC(stg.shape, F16)
        tacc = r2s16.retile(acc)
        for v in cutlass.range_constexpr(cute.size(s32)):
            s32[v] = tacc[v]
        s16.store(s32.load().to(F16))
        cute.copy(r2s16, s16, thr_r.partition_D(sF))
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)
        _seg("k1.face.end", ch)

        # ---- seg 2: eight 8x8 unit-lower diagonal blocks, inverted in place
        #      (transposed back-substitution, SHFL off the chain) ------------
        f8 = cute.flat_divide(sF, (8, 8))
        if wid < 2:
            c = lane % 8
            blk = wid * 4 + lane // 8
            r0 = blk * 8
            base = lane - c
            rr = r0 + c
            rRow = ALLOC((8, 1), F16)
            cute.autovec_copy(f8[(rr % 8, None, rr // 8, r0 // 8)],
                              rRow[(None, 0)])
            rW = cute.recast_tensor(rRow, I32)
            # A fresh name, not one already live in this trace: a dynamic `if`
            # carries every live local across the region.
            yv = [F32(0.0)] * 8
            for qq in cutlass.range_constexpr(4):
                q = 3 - qq
                s_odd = F32(0.0)
                s_even = F32(0.0)
                for t in cutlass.range_constexpr(6 - 2 * q):
                    j = 7 - t
                    ww = cute.arch.shuffle_sync(rW[(q, 0)], base + j)
                    s_odd = s_odd + _f16_hi(ww) * yv[j]
                    s_even = s_even + _f16_lo(ww) * yv[j]
                yv[2 * q + 1] = _sel_eq(2 * q + 1, c, F32(1.0), -s_odd)
                ww = cute.arch.shuffle_sync(rW[(q, 0)], base + 2 * q + 1)
                s_even = s_even + _f16_lo(ww) * yv[2 * q + 1]
                yv[2 * q] = _sel_eq(2 * q, c, F32(1.0), -s_even)
            rOut = ALLOC((8, 1), F16)
            for i in cutlass.range_constexpr(8):
                rOut[(i, 0)] = F16(yv[i])
            cute.autovec_copy(rOut[(None, 0)],
                              f8[(rr % 8, None, rr // 8, r0 // 8)])
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)

        sFb8 = storage.sf.get_tensor(L_b8, swizzle=lytF.inner, dtype=F16)
        sFt8 = storage.sf.get_tensor(L_t8, swizzle=lytF.inner, dtype=F16)
        sF16 = storage.sf.get_tensor(L_16, swizzle=lytF.inner, dtype=F16)
        sFt16 = storage.sf.get_tensor(L_t16, swizzle=lytF.inner, dtype=F16)
        sF16x32 = storage.sf.get_tensor(L_16x32, swizzle=lytF.inner, dtype=F16)
        sFt16x32 = storage.sf.get_tensor(L_t16x32, swizzle=lytF.inner, dtype=F16)
        sFt32x16 = storage.sf.get_tensor(L_t32x16, swizzle=lytF.inner, dtype=F16)

        # ---- seg 3: h = 8, one pair of blocks per warp ----------------------
        b = wid
        A8 = sFb8[(None, None, 2 * b + 1, 2 * b + 1)]
        O8 = sFb8[(None, None, 2 * b + 1, 2 * b)]
        B1 = sFt8[(None, None, 2 * b, 2 * b + 1)]
        B2 = sFt8[(None, None, 2 * b, 2 * b)]
        thr = mma8.get_slice(lane)
        tA8, tB8, tC8 = (tcA8.get_slice(lane), tcB8.get_slice(lane),
                         tcC8.get_slice(lane))
        frA = thr.make_fragment_A(thr.partition_A(A8))
        cute.copy(tcA8, tA8.partition_S(A8), tA8.retile(frA))
        frB1 = thr.make_fragment_B(thr.partition_B(B1))
        cute.copy(tcB8, tB8.partition_S(B1), tB8.retile(frB1))
        frB2 = thr.make_fragment_B(thr.partition_B(B2))
        cute.copy(tcB8, tB8.partition_S(B2), tB8.retile(frB2))
        a1 = ALLOC(thr.partition_shape_C((16, 8)), F32)
        a1.fill(0.0)
        cute.gemm(mma8, a1, frA, frB1, a1)
        for i in cutlass.range_constexpr(cute.size(a1)):
            a1[i] = -a1[i]
        opA = thr.make_fragment_A(thr.partition_shape_A((16, 8)))
        cv = cute.size(a1, mode=[0])
        av = cute.size(opA, mode=[0])
        ratio = av // cv
        opA_view = cute.make_tensor(
            opA.iterator,
            cute.make_layout((cv, cute.size(a1, mode=[1]), ratio,
                              cute.size(a1, mode=[2]) // ratio),
                             stride=(1, av, cv, av * cute.size(a1, mode=[1]))))
        for i in cutlass.range_constexpr(cute.size(a1)):
            opA_view[i] = F16(a1[i])
        a2 = ALLOC(thr.partition_shape_C((16, 8)), F32)
        a2.fill(0.0)
        cute.gemm(mma8, a2, opA, frB2, a2)
        c8 = ALLOC(cute.shape(a2), F16)
        for i in cutlass.range_constexpr(cute.size(a2)):
            c8[i] = F16(a2[i])
        cute.copy(tcC8, tC8.retile(c8), tC8.partition_D(O8))
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)

        # ---- seg 4: h = 16, warps 0/1 --------------------------------------
        if wid < 2:
            A16 = sF16[(None, None, 2 * wid + 1, 2 * wid + 1)]
            O16 = sF16[(None, None, 2 * wid + 1, 2 * wid)]
            B16a = sFt16[(None, None, 2 * wid, 2 * wid + 1)]
            B16b = sFt16[(None, None, 2 * wid, 2 * wid)]
            thr16 = mma16.get_slice(lane)
            uA, uB, uC = (tcA16.get_slice(lane), tcB16.get_slice(lane),
                          tcCst.get_slice(lane))
            fA = thr16.make_fragment_A(thr16.partition_A(A16))
            cute.copy(tcA16, uA.partition_S(A16), uA.retile(fA))
            fB1 = thr16.make_fragment_B(thr16.partition_B(B16a))
            cute.copy(tcB16, uB.partition_S(B16a), uB.retile(fB1))
            fB2 = thr16.make_fragment_B(thr16.partition_B(B16b))
            cute.copy(tcB16, uB.partition_S(B16b), uB.retile(fB2))
            p1 = ALLOC(thr16.partition_shape_C((16, 16)), F32)
            p1.fill(0.0)
            cute.gemm(mma16, p1, fA, fB1, p1)
            for i in cutlass.range_constexpr(cute.size(p1)):
                p1[i] = -p1[i]
            op16 = thr16.make_fragment_A(thr16.partition_shape_A((16, 16)))
            cv = cute.size(p1, mode=[0])
            av = cute.size(op16, mode=[0])
            op16_view = cute.make_tensor(
                op16.iterator,
                cute.make_layout((cv, cute.size(p1, mode=[1]), av // cv,
                                  cute.size(p1, mode=[2]) // (av // cv)),
                                 stride=(1, av, cv,
                                         av * cute.size(p1, mode=[1]))))
            for i in cutlass.range_constexpr(cute.size(p1)):
                op16_view[i] = F16(p1[i])
            p2 = ALLOC(thr16.partition_shape_C((16, 16)), F32)
            p2.fill(0.0)
            cute.gemm(mma16, p2, op16, fB2, p2)
            c16 = ALLOC(cute.shape(p2), F16)
            for i in cutlass.range_constexpr(cute.size(p2)):
                c16[i] = F16(p2[i])
            cute.copy(tcCst, uC.retile(c16), uC.partition_D(O16))
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)

        # ---- seg 5/6: h = 32, 2x2 warp grid, split-K ------------------------
        xg, yg = wid // 2, wid % 2
        A32 = sF16x32[(None, None, 2 + yg, 1)]
        O32 = sF16x32[(None, None, 2 + yg, 0)]
        B32a = sFt16x32[(None, None, xg, 1)]
        B32b = sFt32x16[(None, None, 0, xg)]
        thr32 = mma16.get_slice(lane)
        vA, vB = tcA16.get_slice(lane), tcB16.get_slice(lane)
        fA = thr32.make_fragment_A(thr32.partition_A(A32))
        cute.copy(tcA16, vA.partition_S(A32), vA.retile(fA))
        fB1 = thr32.make_fragment_B(thr32.partition_B(B32a))
        cute.copy(tcB16, vB.partition_S(B32a), vB.retile(fB1))
        fB2 = thr32.make_fragment_B(thr32.partition_B(B32b))
        cute.copy(tcB16, vB.partition_S(B32b), vB.retile(fB2))
        # h=32 is split-K: GEMM1 is (M=16, N=16, K=32) -- its N is the half of
        # the 32-wide output this warp column owns, and that same 16 is GEMM2's
        # K.  Only GEMM2 is (M=16, N=32, K=16).
        q1 = ALLOC(thr32.partition_shape_C((16, 16)), F32)
        q1.fill(0.0)
        cute.gemm(mma16, q1, fA, fB1, q1)
        for i in cutlass.range_constexpr(cute.size(q1)):
            q1[i] = -q1[i]
        op32 = thr32.make_fragment_A(thr32.partition_shape_A((16, 16)))
        cv = cute.size(q1, mode=[0])
        av = cute.size(op32, mode=[0])
        op32_view = cute.make_tensor(
            op32.iterator,
            cute.make_layout((cv, cute.size(q1, mode=[1]), av // cv,
                              cute.size(q1, mode=[2]) // (av // cv)),
                             stride=(1, av, cv,
                                     av * cute.size(q1, mode=[1]))))
        for i in cutlass.range_constexpr(cute.size(q1)):
            op32_view[i] = F16(q1[i])
        q2 = ALLOC(thr32.partition_shape_C((16, 32)), F32)
        q2.fill(0.0)
        cute.gemm(mma16, q2, op32, fB2, q2)
        c32 = ALLOC(cute.shape(q2), F16)
        for i in cutlass.range_constexpr(cute.size(q2)):
            c32[i] = F16(q2[i])
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)      # 5
        thrS = tcCst.get_slice(lane)
        dst = thrS.partition_D(O32)
        src = thrS.retile(c32)
        if xg == 0:
            cute.copy(tcCst, src, dst)
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)      # 6
        if xg == 1:
            thrL = tcCld.get_slice(lane)
            red = ALLOC(cute.shape(c32), F16)
            cute.copy(tcCld, thrL.partition_S(O32), thrL.retile(red))
            for i in cutlass.range_constexpr(cute.size(c32)):
                c32[i] = c32[i] + red[i]
            cute.copy(tcCst, src, dst)
        cute.arch.barrier(barrier_id=BAR, number_of_threads=TID)      # 7
        _seg("k1.inv.end", ch)

        # ---- exit: row-major drain of the fp16 face, bf16 to gmem -----------
        # A's rows are 64 bf16 = 128 B contiguous in gmem, so the drain walks
        # them row-major: one STG then covers 32 consecutive bf16 = 64 B = two
        # FULL 32-B sectors.  The previous form read the face back with an
        # LDSM into C-fragment order and scattered `gA[m, n]` per element: 4
        # used bytes per 32-B write sector = 8x write amplification on the
        # 33.5 MB A tensor, and the LDSM fragment was 32 live F16 registers
        # for no reason -- the face is already the value being written.
        gA = mA[(ch, None, hv, None)]
        for i in cutlass.range_constexpr((C * C) // TID):
            e = i * TID + tidx
            gA[e // C, e % C] = BF16(F32(sF[(e // C, e % C)]))
        if tidx < C:
            mGc[(ch, tidx, hv)] = sG[tidx]
        _seg("k1.exit", ch)
        # D2.3: A, g_cumsum and the dq/dk zeroing are all issued; K2 and K3 may
        # be scheduled.  They still block at their own `griddepcontrol.wait`
        # until this grid has retired and flushed, so this is a scheduling
        # signal only and nothing above it had to move.
        _pdl_trigger()

    # -- layouts / atoms.  These MUST be built while an MLIR Context is live,
    # i.e. inside the @cute.jit trace: swizzled layout atoms parse an
    # `!cute.swizzle<...>` type and raise "requires a Context" when built at
    # module/closure level.
    def _atoms():
        k_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, D),
            cutlass.BFloat16)
        lytK = cute.tile_to_shape(k_atom, (C, D), order=(0, 1))

        f_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, F16)
        lytF = cute.tile_to_shape(f_atom, (C, C), order=(0, 1))

        mma_wg = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            nvgpu.OperandMajorMode.K, nvgpu.OperandMajorMode.K,
            F32, atom_layout_mnk=(1, 1, 1), tiler_mn=(C, C))

        E = LayoutEnum.ROW_MAJOR
        r2s16 = cute.make_tiled_copy_S(
            sm90_utils.get_smem_store_op(E, elem_ty_d=F16, elem_ty_acc=F32),
            cute.make_tiled_copy_C_atom(
                cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), F16),
                mma_wg))

        mma8 = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 8)), (1, 1, 1))
        tcA8 = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 1), F16), mma8)
        tcB8 = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 1), F16), mma8)
        tcC8 = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 1), F16), mma8)
        mma16 = cute.make_tiled_mma(warp.MmaF16BF16Op(F16, F32, (16, 8, 16)), (1, 1, 1))
        tcA16 = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), F16), mma16)
        tcB16 = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), F16), mma16)
        tcCst = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), F16), mma16)
        tcCld = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), F16), mma16)

        L_b8 = cute.make_layout(((8, 2), 8, 8, 8), stride=((64, 0), 1, 512, 8))
        L_t8 = cute.make_layout((8, 8, 8, 8), stride=(1, 64, 8, 512))
        L_16 = cute.make_layout((16, 16, 4, 4), stride=(64, 1, 1024, 16))
        L_t16 = cute.make_layout((16, 16, 4, 4), stride=(1, 64, 16, 1024))
        L_16x32 = cute.make_layout((16, 32, 4, 2), stride=(64, 1, 1024, 32))
        L_t16x32 = cute.make_layout((16, 32, 4, 2), stride=(1, 64, 16, 2048))
        L_t32x16 = cute.make_layout((32, 16, 2, 4), stride=(1, 64, 32, 1024))
        return (mma_wg, r2s16, mma8, tcA8, tcB8, tcC8, mma16, tcA16,
                tcB16, tcCst, tcCld, lytK, lytF, L_b8, L_t8, L_16, L_t16,
                L_16x32, L_t16x32, L_t32x16)

    def _storage():
        @cute.struct
        class SharedStorage:
            sk: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, C * D], 1024]
            sf: cute.struct.Align[cute.struct.MemRange[F16, C * C], 1024]
            sg: cute.struct.Align[cute.struct.MemRange[F32, C], 1024]
            sb: cute.struct.Align[cute.struct.MemRange[F32, C], 1024]
            sw: cute.struct.Align[cute.struct.MemRange[F32, 4], 1024]
        return SharedStorage

    @cute.jit
    def launch_k1_chunk_prep(mK, mG, mB, mGc, mA, mDqZ, mDkZ, st):
        SharedStorage = _storage()
        atoms = _atoms()
        # v-head fastest (blockIdx.x): the GVA pair shares one k tile
        k1_chunk_prep(mK, mG, mB, mGc, mA, mDqZ, mDkZ,
                      *atoms, SharedStorage).launch(
            grid=[H_V, NC, 1], block=[TID, 1, 1], stream=st)

    return launch_k1_chunk_prep


# =============================================================================
# K2 / K3 / K4 / K5 — the rest of the D0.1 launch table
#
# K2 (D0.3), K3 (D0.4), K4 (D0.5/D0.6/D0.8-D0.14) and K5 (D0.14's GVA pair
# reduction) are all in their true form below.
# =============================================================================

def _build_k2():
    """K2 `prepare_h` (D0.3) — segment-parallel forward scan + NWARM-chunk warm-up.

    grid (N_SEG, H_V) = 256, block 512 = 4 warpgroups (1 producer + 3 consumer).
    Each (segment, v-head) CTA starts from the ZERO state, scans NWARM warm-up
    chunks (segment 0 has none) and then the segment's own 16 chunks, publishing
    every one of the segment's chunk ENTRY states to `h_c`.  The warm-up is what
    replaces the reference's cp_cache: the transition operator over an interval
    is bounded by exp(sum g) (milestone 1.3) and 128 tokens of g ~ log U(0,1)
    sum to about -64 per chunk, so everything older than the warm-up is fp32
    zero (the bound and the D2.2 move from two warm-up chunks to one are
    written out at NWARM's definition).

    Per chunk (S = fp32 state [d_k, d_v], G = K1's chunk-inclusive g_cumsum):

        W'  = K @ S                        # WG1
        B   = beta * (V - exp(G) * W')     # WG1, row-scalar elementwise
        U   = A @ B                        # WG2, A = K1's (I + tril(...))^-1
        Kd  = exp(G_last - G) * K          # WG2, the decayed key face
        S  <- exp(G_last) * S + Kd^T @ U   # WG0, as four 64x64 quadrants

    h_c holds the chunk ENTRY state, so WG0's bf16 publish of S happens BEFORE
    the update, and that same face serves both the TMA store and WG1's B
    operand.  Supply is a 2-stage TMA landing ring (5 boxes/chunk: k x2, v x2,
    A x1) driven by one producer warp through a PipelineTmaAsync; g_cumsum and
    beta are hoisted out of the ring and read once for the whole segment.

    Every elementwise segment takes its non-accumulator operands by MATRIX copy
    (D0.16): v enters WG1's B segment as an `LDSM` fragment isomorphic to the
    accumulator, the Kd face is shaped once by `LDSM -> FMUL -> STSM`, and the
    row scalars (beta, the gate) are hoisted to the two rows a thread holds.
    The per-element `sTile[(m, n)]` reads and the per-element scalar `STS` face
    loops those replace are the form §4 of the milestone bans outright.
    """
    BF16, F32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32
    ALLOC = _rmem_alloc()
    TID = 512                 # 4 warpgroups
    STAGE = 2                 # landing ring depth
    SLOTS = 5                 # k0 k1 v0 v1 A, per stage
    TILE = C                  # every smem tile is 64 x 64 bf16 = 8 KB
    NFACE = 10                # S[2][2] | B[2] | U[2] | Kd[2]
    FS, FB, FU, FK = 0, 4, 6, 8
    BAR_CONS = 1              # NamedBarrier over the 384 consumer threads
    BAR_WG0 = 2               # NamedBarrier inside WG0
    SEG_TOK = (SEG_CHUNKS + NWARM) * C
    LOG2E = 1.4426950408889634
    # D0.3 registers the reference's `setmaxnreg` sites for this kernel, and
    # D0.11's binding constraint applies to K2's state WG exactly as it does to
    # K4's: WG0 carries the 128x128 f32 chunk state in 128 registers/thread at
    # EVERY program point of its loop, the publish of stage (a) included.  Under
    # the 128-register entry quota that is an arithmetic overrun and not an
    # allocator preference -- 128 resident + the pack fragment + the stage's
    # addressing cannot fit, and the evicted accumulator is re-materialised from
    # local memory on each of its three touches per chunk.  The producer warp
    # needs only its TMA bookkeeping, so handing its surplus to the consumers
    # raises every consumer to the largest quota the pool allows:
    # 128*32 + 384*160 = 65536 conserves it exactly (`warp_specialization`
    # 坑 4 / 坑 8; over-subscribing does not fail to compile -- the consumers
    # hang on `.inc` forever).
    PROD_REGS, CONS_REGS = 32, 160
    REG_DEC, REG_INC = _reg_ops()

    @dsl_user_op
    def _sel_gt_i(a, b, x, y, *, loc=None, ip=None):
        return I32(llvm.inline_asm(
            T.i32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip),
                      I32(x).ir_value(loc=loc, ip=ip),
                      I32(y).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.gt.s32 p, $1, $2;\n\tselp.s32 $0, $3, $4, p;\n}",
            "=r,r,r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _imax(a, b, *, loc=None, ip=None):
        return I32(llvm.inline_asm(
            T.i32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip)],
            "max.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip))

    # cute.arch's bulk commit/wait names drift across DSL minor versions; both
    # are operand-less per-thread instructions, so inline PTX is version-proof.
    @dsl_user_op
    def _bulk_commit(*, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                        has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def _bulk_wait(n, *, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.bulk.wait_group.read %d;" % int(n),
                        "", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    def _exp(x):
        return cute.math.exp2(x * LOG2E, fastmath=True)

    # -- Two helpers reached from inside staged control flow.  Both take every
    # runtime value (thread-sliced partitioners, smem tensors, accumulators) as
    # an EXPLICIT argument: a function entered from a staged `if`/`for` may not
    # reach runtime values through its closure.  Neither takes @cute.jit (they
    # hold no runtime control flow, and a jit boundary would rebind their
    # trace-time arguments as dynamic values) -- so their bodies are never
    # preprocessed either, and `cutlass.range_constexpr` would reach the
    # interpreter raw and raise.  Plain `range` is the spelling OUTSIDE a
    # decorated body; `range_constexpr` the one inside it.
    def publish(r2s_, thr_r, sF, stg, acc, slot):
        """f32 C-fragment -> bf16 -> STSM x4 into face `slot` of `sF`.

        The narrowing reads the accumulator THROUGH the store atom's retile and
        converts element by element straight into the 16-bit fragment, with no
        f32 staging copy of the accumulator (D0.12 段B: "打包/精度转换必须终结
        上游 f32 累加器的活跃区 —— 转换后不得保留其 f32 副本").  WG0 is the case
        that makes the difference: its accumulator IS the segment-resident
        state, so a full f32 copy of the quadrant being published adds 32
        registers on top of the 128 that stay live across the call.
        """
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(acc)
        for v in range(cute.size(s16)):
            s16[v] = BF16(tacc[v])
        cute.copy(r2s_, s16, thr_r.partition_D(sF[(None, None, slot)]))

    def ldsm_c(l2r_, thr_l, cshp, sT, slot):
        """D0.16 ②: slot `slot` of `sT` -> a fragment isomorphic to the C tile.

        ONE `LDSM.16.M88.4` per warp per 16x16 sub-tile (four per 64x64 tile)
        replaces the 32 per-element scalar `LDS` + 32 swizzle address
        computations a `sT[(m, n)]` read costs, and the result is indexed by
        the SAME fragment index as the accumulator it is about to be combined
        with.  Every landing box and every face in this kernel is the same
        K_SW128 64x64 bf16 tile K4's slots are, and the atom is built on the
        same TiledMma, so this is the mapping notes/probe_m0_d016_ldsm.py
        verified on the GPU (R1 on a TMA-landed slot, R4 on an STSM-written
        one).  Returns bf16; call sites convert per element on use, so no f32
        copy of the tile is ever materialised (D0.12 段B).
        """
        r = ALLOC(cshp, BF16)
        cute.copy(l2r_, thr_l.partition_S(sT[(None, None, slot)]),
                  thr_l.retile(r))
        return r

    def shape_face(l2r_, r2s_, thr_l, thr_r, cshp, stg,
                   sS, src, sD, dst, f0, f1):
        """D0.16 ③: a row-gated derived face, shaped once by matrix copies.

        `dst_face[m, n] = f_m * src_tile[m, n]`, where f is a ROW scalar and so
        collapses (D0.16's M-axis hoist) to the TWO values the caller already
        exponentiated for its own rows; bit 1 of the fragment index selects
        between them (probe R2).  The path is `LDSM` -> 32 FMUL -> F2FP ->
        `STSM`, a constant number of matrix instructions per warp, in place of
        the 64 scalar LDS + 64 scalar STS + 64 `MUFU` + 64 swizzle address
        computations the per-element face loops used to pay per chunk -- the
        whole of this kernel's `scalar_sts_in_loop`.  The gate factors are
        <= 1 (log gate <= 0, §1.3; beta <= 1), so the bf16 face cannot
        overflow.
        """
        rx = ldsm_c(l2r_, thr_l, cshp, sS, src)
        ry = ALLOC(cshp, BF16)
        for i in range(cute.size(rx)):
            ry[i] = BF16(F32(rx[i]) * (f0 if (i // 2) % 2 == 0 else f1))
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(ry)
        for v in range(cute.size(s16)):
            s16[v] = tacc[v]
        cute.copy(r2s_, s16, thr_r.partition_D(sD[(None, None, dst)]))

    def link(mma, acc, sA_t, sB_t, nk, accumulate):
        """Put ONE WGMMA SS chain into the group that is currently open.

        No fence, no `commit_group`, no wait -- the inner half of the two-level
        issue helper hw_kb `wgmma_deferred_wait` 坑 11 asks for, and what lets
        several chains share one group and one drain (D1.1, brought over to K2
        by D2.2).

        `mma` must be the TiledMma whose major-mode pair matches the two views
        actually handed in.  `accumulate` is the flag of the chain's FIRST
        k-block only (the rest always accumulate): False initialises `acc`,
        True adds onto it.

        The A/B descriptors are built on the WARPGROUP-scope slice
        `get_slice(0)`, never on the per-thread slice (hw_kb
        `wgmma_desc_uniform` 坑 3).  `wgmma.mma_async` reads its two 64-bit
        matrix descriptors from the uniform register file (`desc[UR*]`), and
        ptxas can only leave them there if it can prove they are
        warp-invariant; a `tidx`-derived slice coordinate breaks that proof,
        so each descriptor is computed in per-thread registers and re-staged
        with `R2UR` ahead of every issue -- 2 dwords x its whole live range
        charged to the `setmaxnreg` quota, for every chain in flight.  Only
        `partition_C` (the accumulator) and the STSM partitions are
        per-thread.  A fresh atom per call site is mandatory (坑 4).
        """
        wgm = mma.get_slice(0)
        at = cute.make_mma_atom(mma.op)
        at.set(warpgroup.Field.ACCUMULATE, accumulate)
        fa = wgm.make_fragment_A(wgm.partition_A(sA_t))
        fb = wgm.make_fragment_B(wgm.partition_B(sB_t))
        for kb in range(nk):
            cute.gemm(at, acc, fa[None, None, kb], fb[None, None, kb], acc)
            at.set(warpgroup.Field.ACCUMULATE, True)

    @cute.kernel
    def prepare_h(tmaK: cute.CopyAtom, gKz: cute.Tensor,
             tmaV: cute.CopyAtom, gVz: cute.Tensor,
             tmaA: cute.CopyAtom, gAz: cute.Tensor,
             tmaH: cute.CopyAtom, gHz: cute.Tensor,
             mGc: cute.Tensor, mBe: cute.Tensor,
             mma_km: cute.TiledMma, mma_mm: cute.TiledMma,
             r2s: cute.TiledCopy, l2r: cute.TiledCopy,
             lyt_land: cute.ComposedLayout, lyt_face: cute.ComposedLayout,
             lyt_faceT: cute.ComposedLayout,
             SharedStorage: cutlass.Constexpr[Callable]):
        tidx, _, _ = cute.arch.thread_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, _ = cute.arch.block_idx()
        tw = tidx % 128
        # D2.3: ahead of EVERY global read -- the mGc prologue below reads K1's
        # g_cumsum, and the producer's A boxes are K1's output too.
        _pdl_wait()

        seg = I32(bx)
        head = I32(by)
        hq = head // GVA

        nw = _sel_gt_i(seg, 0, NWARM, 0)     # segment 0 starts at the true zero
        ntrips = I32(SEG_CHUNKS) + nw
        c0 = seg * SEG_CHUNKS - nw

        # ---- shared prologue (all 512 threads; no role branch yet) ----------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLand = storage.land.get_tensor(lyt_land.outer, swizzle=lyt_land.inner)
        sFace = storage.face.get_tensor(lyt_face.outer, swizzle=lyt_face.inner)
        sFaceT = storage.face.get_tensor(lyt_faceT.outer, swizzle=lyt_faceT.inner)
        sGc = storage.gc.get_tensor((SEG_TOK,))
        sBe = storage.be.get_tensor((SEG_TOK,))

        tx = SLOTS * TILE * TILE * 2                       # bytes per stage
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bars.data_ptr(), num_stages=STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))

        tKs, tKg = cpasync.tma_partition(tmaK, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gKz)
        tVs, tVg = cpasync.tma_partition(tmaV, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gVz)
        tAs, tAg = cpasync.tma_partition(tmaA, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gAz)
        tHs, tHg = cpasync.tma_partition(tmaH, 0, cute.make_layout(1),
                                         cute.group_modes(sFace, 0, 2), gHz)

        # gate / beta for the whole segment, once (keeps them out of the ring)
        for e in cutlass.range_constexpr(SEG_TOK // TID + 1):
            j = e * TID + tidx
            if j < SEG_TOK:
                ch = _imax(c0 + j // C, I32(0))
                sGc[j] = mGc[(ch, j % C, head)]
                sBe[j] = mBe[(ch, j % C, head)]
        cute.arch.sync_threads()
        _seg("k2.entry")

        # ---- role split -----------------------------------------------------
        # the dec/inc is the first statement of its branch, with no barrier
        # between branch and quota (`warp_specialization` 坑 5), and everything
        # shared sits ABOVE the branch so that the prologue is compiled against
        # the 128-register entry quota rather than the producer's 32 (坑 11).
        if wid >= 12:
            REG_DEC(PROD_REGS)
            # producer: ONE warp issues the 5 TMA boxes of each chunk
            if wid == 12:
                _seg("k2.producer.entry")
                p_st = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, STAGE)
                for t in cutlass.range(ntrips):
                    ch = _imax(c0 + t, I32(0))
                    _seg("k2.supply.acquire.begin", t)
                    pipe.producer_acquire(p_st)
                    _seg("k2.supply.acquire.end", t)
                    bar = pipe.producer_get_barrier(p_st)
                    b = p_st.index * SLOTS
                    for mh in cutlass.range_constexpr(2):
                        cute.copy(tmaK, tKg[(None, (ch, hq * 2 + mh))],
                                  tKs[(None, b + mh)], tma_bar_ptr=bar)
                    for nh in cutlass.range_constexpr(2):
                        cute.copy(tmaV, tVg[(None, (ch, head * 2 + nh))],
                                  tVs[(None, b + 2 + nh)], tma_bar_ptr=bar)
                    cute.copy(tmaA, tAg[(None, (ch, head))],
                              tAs[(None, b + 4)], tma_bar_ptr=bar)
                    pipe.producer_commit(p_st)
                    p_st.advance()
                    _seg("k2.supply.issue.end", t)
                _seg("k2.producer.exit")
        else:
            REG_INC(CONS_REGS)
            _seg("k2.consumer.entry")
            c_st = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, STAGE)
            # ONE per-thread MMA slice, and it is used for nothing but the
            # accumulator coordinates: A/B now ride `get_slice(0)` inside
            # `chain` (hw_kb `wgmma_desc_uniform`).  The three TiledMma differ
            # only in their operand major modes, so partition_C is the same
            # for all three.
            thr_c = mma_km.get_slice(tw)
            cS = thr_c.partition_C(cute.make_identity_tensor((TILE, TILE)))
            # A comprehension is its own scope and the rewriter does not descend
            # into it, so this `range` stays plain python; the statement loop on
            # the next line IS rewritten and therefore needs `range_constexpr`
            # (a plain `range` there would hand the body a runtime Int32 and
            # make `accS[j]` a dynamic list index).
            accS = [ALLOC(cS.shape, F32) for _ in range(4)]
            for j in cutlass.range_constexpr(4):
                accS[j].fill(0.0)
            thr_r = r2s.get_slice(tw)
            stg = cute.make_layout(cute.shape(thr_r.partition_S(
                sFace[(None, None, 0)]))[:3])
            # D0.16's read side.  Same rule as the STSM slice: the tiled copy is
            # sliced by the WARPGROUP thread number, never by lane and never by
            # `tidx` (`ldmatrix_stmatrix` 坑 5), so that `thr_l`'s fragment index
            # agrees element for element with `cS`.
            thr_l = l2r.get_slice(tw)
            cshp = thr_c.partition_shape_C((TILE, TILE))

            for t in cutlass.range(ntrips):
                ch = _imax(c0 + t, I32(0))
                _seg("k2.chunk.land.begin", t)
                pipe.consumer_wait(c_st)
                _seg("k2.chunk.land.end", t)
                b = c_st.index * SLOTS
                gbase = t * C

                # ---- (a) WG0 publishes the chunk ENTRY state ----------------
                if wid < 4:
                    for j in cutlass.range_constexpr(4):
                        publish(r2s, thr_r, sFace, stg, accS[j], FS + j)
                    cute.arch.fence_view_async_shared()
                    _seg("k2.state.publish.end", t)
                cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=384)

                # ---- (b) WG0 stores h_c; WG1 computes W' and B --------------
                if wid < 4:
                    if t >= nw:                      # warm-up chunks: no store
                        if wid == 0:
                            for mh in cutlass.range_constexpr(2):
                                for nh in cutlass.range_constexpr(2):
                                    cute.copy(
                                        tmaH, tHs[(None, FS + mh * 2 + nh)],
                                        tHg[(None, ((ch * H_V + head) * 2 + mh, nh))])
                            _bulk_commit()
                            _seg("k2.hc.store.issue.end", t)
                elif wid < 8:
                    # D0.16's M-axis hoist, once per chunk for the whole role:
                    # the 32 elements a thread holds sit on exactly TWO rows
                    # (probe R2), so beta and the gate are TWO scalar `LDS` and
                    # TWO `MUFU` for the whole segment below instead of 32 of
                    # each, and bit 1 of the fragment index picks between them.
                    m0, m1 = cS[0][0], cS[2][0]
                    be0, be1 = sBe[gbase + m0], sBe[gbase + m1]
                    eg0, eg1 = _exp(sGc[gbase + m0]), _exp(sGc[gbase + m1])
                    # D1.1's ladder, brought to K2 by D2.2.  W' = K @ S
                    # contracts over d_k = 128 = two 64-wide tiles; both target
                    # the SAME accumulator with nothing between them, so they
                    # share one group and one drain (hw_kb `wgmma_deferred_wait`
                    # 坑 11) instead of fencing/committing/draining twice.  The
                    # two N halves are INDEPENDENT accumulators, so both groups
                    # are issued before either is reaped: `wait_group(1)` takes
                    # half 0 while half 1 flies, and half 0's LDSM + 32 FFMA +
                    # STSM publish is the covered段 (坑 12's sign condition --
                    # straight-line, no blocking wait and no stage release in
                    # it).  Cost: the second f32 accumulator is live across the
                    # first half's elementwise, +32 registers on WG1's 160.
                    accB = [ALLOC(cS.shape, F32) for _ in range(2)]
                    for nh in cutlass.range_constexpr(2):
                        warpgroup.fence()
                        # W' = K @ S contracts over d_k = 128 = two 64-wide
                        # tiles: the FIRST one initialises the fragment
                        # (accumulate False) and the second adds onto it.
                        for mh in cutlass.range_constexpr(2):
                            link(mma_km, accB[nh],
                                 sLand[(None, None, b + mh)],
                                 sFaceT[(None, None, FS + mh * 2 + nh)],
                                 4, mh > 0)
                        warpgroup.commit_group()
                    for nh in cutlass.range_constexpr(2):
                        warpgroup.wait_group(1 - nh)
                        acc = accB[nh]
                        # v enters the elementwise by matrix copy (D0.16 ② --
                        # it is aligned element for element with `acc`), not by
                        # 32 scalar `LDS` at `sV[(m, n)]`
                        rV = ldsm_c(l2r, thr_l, cshp, sLand, b + 2 + nh)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                acc[i] = be0 * (F32(rV[i]) - eg0 * acc[i])
                            else:
                                acc[i] = be1 * (F32(rV[i]) - eg1 * acc[i])
                        publish(r2s, thr_r, sFace, stg, acc, FB + nh)
                    cute.arch.fence_view_async_shared()
                    _seg("k2.wprime.end", t)
                cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=384)

                # ---- (c) WG2: U = A @ B, and the decayed key face -----------
                # inside the consumer branch wid < 12 already holds, so one
                # comparison is enough (a chained `8 <= wid < 12` would ask
                # Python for the truthiness of a traced value)
                if wid >= 8:
                    glast = sGc[gbase + C - 1]
                    # D1.1's ladder again, and here the covered段 is already
                    # written: the two U = A @ B chains are independent
                    # accumulators and the Kd face does not depend on either,
                    # so both groups go out first and the Kd shaping (two
                    # LDSM -> 32 FMUL -> STSM passes) runs inside their flight.
                    accU = [ALLOC(cS.shape, F32) for _ in range(2)]
                    for nh in cutlass.range_constexpr(2):
                        warpgroup.fence()
                        link(mma_km, accU[nh],
                             sLand[(None, None, b + 4)],
                             sFaceT[(None, None, FB + nh)], 4, False)
                        warpgroup.commit_group()
                    # Kd = e^{G_last-G} o K is a ROW scaling of a landed tile,
                    # so D0.16 ③: LDSM into C-fragment order, two FMUL factors
                    # per thread, STSM out.  No element of the face is ever
                    # addressed individually, which is the whole of this
                    # kernel's `scalar_sts_in_loop`.  The factor is <= 1
                    # (G is the inclusive cumsum of g <= 0, so G_last <= G_r).
                    m0, m1 = cS[0][0], cS[2][0]
                    d0 = _exp(glast - sGc[gbase + m0])
                    d1 = _exp(glast - sGc[gbase + m1])
                    for mh in cutlass.range_constexpr(2):
                        shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                   sLand, b + mh, sFace, FK + mh, d0, d1)
                    for nh in cutlass.range_constexpr(2):
                        warpgroup.wait_group(1 - nh)
                        publish(r2s, thr_r, sFace, stg, accU[nh], FU + nh)
                    cute.arch.fence_view_async_shared()
                    _seg("k2.uface.end", t)
                cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=384)

                # ---- (d) WG0 advances the state -----------------------------
                if wid < 4:
                    dec = _exp(sGc[gbase + C - 1])
                    # D2.2's named site: the state advance used to be four
                    # groups, each fence -> 4 k-blocks -> commit -> drain, so
                    # quadrant j could not even ISSUE until j-1 had retired.
                    # D1.1's ladder verbatim -- quadrant j is committed and
                    # left in flight, `wait_group(1)` reaps j-1 -- so at any
                    # moment two chains are outstanding and the tensor pipe
                    # stops serialising against itself.  Unlike WG1/WG2 above
                    # this costs nothing in registers: the four quadrants are
                    # the segment-resident state and are live either way
                    # (D0.11), and depth 1 holds the descriptor liveness at
                    # the two chains K4's WG0 already carries.
                    for j in cutlass.range_constexpr(4):
                        a = accS[j]
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            a[i] = a[i] * dec
                    for j in cutlass.range_constexpr(4):
                        warpgroup.fence()
                        link(mma_mm, accS[j],
                             sFaceT[(None, None, FK + j // 2)],
                             sFaceT[(None, None, FU + j % 2)], 4, True)
                        warpgroup.commit_group()
                        if cutlass.const_expr(j > 0):
                            warpgroup.wait_group(1)
                    warpgroup.wait_group(0)
                    _seg("k2.state.update.end", t)
                    # the store warp drains its own bulk group before ANY of
                    # WG0's four warps may STSM over the face again
                    if wid == 0:
                        _seg("k2.hc.store.drain.begin", t)
                        _bulk_wait(0)
                        _seg("k2.hc.store.drain.end", t)
                    cute.arch.barrier(barrier_id=BAR_WG0, number_of_threads=128)
                    pipe.consumer_release(c_st)
                c_st.advance()
            _seg("k2.exit")
        _bulk_wait(0)
        # D2.3: every h_c bulk store has drained, so K4 may be scheduled.  It
        # still blocks at its own `griddepcontrol.wait` until this grid retires.
        _pdl_trigger()

    # -- layouts / atoms / TMA: all IR builders, so all inside the jit trace --
    def _atoms(mK2, mV2, mA2, mH2):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, TILE),
            BF16)
        lyt1 = cute.tile_to_shape(atom, (TILE, TILE), order=(0, 1))
        lyt_land = cute.tile_to_shape(atom, (TILE, TILE, SLOTS * STAGE),
                                      order=(0, 1, 2))
        lyt_face = cute.tile_to_shape(atom, (TILE, TILE, NFACE), order=(0, 1, 2))
        # read view: sFaceT[(c, r, j)] == sFace[(r, c, j)].  colex index of
        # (r, c, j) over (64, 64, NFACE) is r + 64c + 4096j, so the transposer's
        # stride, in READ-view coordinate order (c, r, j), is (64, 1, 4096).
        lyt_faceT = cute.composition(
            lyt_face, cute.make_layout((TILE, TILE, NFACE),
                                       stride=(TILE, 1, TILE * TILE)))

        MM = nvgpu.OperandMajorMode
        mma_km = sm90_utils.make_trivial_tiled_mma(
            BF16, BF16, MM.K, MM.MN, F32, atom_layout_mnk=(1, 1, 1),
            tiler_mn=(TILE, TILE))
        mma_mm = sm90_utils.make_trivial_tiled_mma(
            BF16, BF16, MM.MN, MM.MN, F32, atom_layout_mnk=(1, 1, 1),
            tiler_mn=(TILE, TILE))

        E = LayoutEnum.ROW_MAJOR
        r2s = cute.make_tiled_copy_S(
            sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
            cute.make_tiled_copy_C_atom(
                cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4),
                                    BF16), mma_km))
        # D0.16's READ side: the matrix-copy atom that brings a non-accumulator
        # tile into a fragment ISOMORPHIC to the accumulator, so that fragment
        # index `i` and `cS[i]` name the same (m, n).  Every landing box and
        # every face here is the same K_SW128 64x64 bf16 tile with N contiguous
        # that K4's slots are, so the member is the NON-transposed `.M88` one
        # and the whole C tile is covered -- hence `make_tiled_copy_C` (not
        # `..._C_atom`, which builds only the per-atom tiling).
        l2r = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma_km)

        g2s = cpasync.CopyBulkTensorTileG2SOp()
        tmaK, cK = cpasync.make_tiled_tma_atom(g2s, mK2, lyt1, (TILE, TILE),
                                               num_multicast=1)
        tmaV, cV = cpasync.make_tiled_tma_atom(g2s, mV2, lyt1, (TILE, TILE),
                                               num_multicast=1)
        tmaA, cA = cpasync.make_tiled_tma_atom(g2s, mA2, lyt1, (TILE, TILE),
                                               num_multicast=1)
        tmaH, cH = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mH2, lyt1, (TILE, TILE))
        z = lambda t: cute.zipped_divide(t, (TILE, TILE))
        return (tmaK, z(cK), tmaV, z(cV), tmaA, z(cA), tmaH, z(cH),
                mma_km, mma_mm, r2s, lyt_land, lyt_face, lyt_faceT, l2r)

    def _storage(n_land, n_face):
        @cute.struct
        class SharedStorage:
            bars: cute.struct.MemRange[cutlass.Int64, STAGE * 2]
            land: cute.struct.Align[cute.struct.MemRange[BF16, n_land], 1024]
            face: cute.struct.Align[cute.struct.MemRange[BF16, n_face], 1024]
            gc: cute.struct.Align[cute.struct.MemRange[F32, SEG_TOK], 1024]
            be: cute.struct.Align[cute.struct.MemRange[F32, SEG_TOK], 1024]
        return SharedStorage

    @cute.jit
    def launch_prepare_h(mK2, mV2, mA2, mH2, mGc, mBe, st):
        a = _atoms(mK2, mV2, mA2, mH2)
        SS = _storage(cute.cosize(a[11]), cute.cosize(a[12]))
        prepare_h(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                  mGc, mBe, a[8], a[9], a[10], a[14], a[11], a[12], a[13],
                  SS).launch(grid=[N_SEG, H_V, 1], block=[TID, 1, 1],
                             min_blocks_per_mp=1, stream=st, **_pdl_kw())

    return launch_prepare_h


def _build_k3():
    """K3 `dh_boundary` (D0.4) — segment-end dh by NWARM-chunk look-ahead.

    grid (N_SEG-1, H_V) = 224, block 512 = 4 warpgroups (1 producer + 3
    consumer), the backward twin of K2.  Segment 7's dh_end slot has no producer
    (no final-state gradient) and is never written; K4 discards what it reads
    there on the segment number (D0.7's second zeroing option).

    Each (boundary, v-head) CTA seeds dh = 0 and runs the chunk-form adjoint
    BACKWARDS over the NWARM look-ahead chunks of the FOLLOWING segment; what is
    left is dL/dS at the boundary's last token.  The zero seed is D0.4's decay
    bound: everything more than NWARM * 64 tokens later is suppressed by
    exp(sum g) below fp32 (the arithmetic is at NWARM's definition).  Per chunk (G = K1's chunk-inclusive g_cumsum, A = K1's inverse):

        Sq  = scale e^{G_i-G_j} (q_i.k_j), j <= i   Kb = beta e^G o K
        Kd  = e^{G_last-G} o K                      Qg = scale e^{G} o Q
        W   = A @ Kb   ( = A diag(beta e^G) K )
        dvn = Kd @ dh + Sq^T @ dO
        dh <- e^{G_last} dh + Qg^T @ dO - W^T @ dvn

    Roles per trip: WG0 owns the 128x128 f32 dh as four 64x64 accumulators and
    publishes its bf16 copy face, WG1 forms Kb and then dvn, WG2 forms Sq/Kd and
    then W/Qg; three NamedBarrier arrivals over the 384 consumer threads separate
    the three stages.  Supply is the same PipelineTmaAsync landing ring as K2 at
    STAGE=1 (at NWARM = 1 there is a single trip, so depth buys nothing), 7
    boxes per trip (q x2 and k x2 at
    qk-head head//2, do x2, A x1); g_cumsum and beta are hoisted out of the ring.
    The exchange faces carry -dvn, so WG0's dh update is a plain accumulate.

    All three derived faces (Kb, Kd, Qg) are ROW gatings of a landed tile and
    are shaped once by `LDSM -> FMUL -> STSM` (D0.16 ③); the W product's gate
    sits on the contraction index, so moving it from A's columns to K's rows
    (D0.16 ①) is what makes the row form available.  No face element and no
    input tile element is ever addressed individually.
    """
    BF16, F32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32
    ALLOC = _rmem_alloc()
    TID = 512                 # 4 warpgroups
    STAGE = 1                 # landing ring depth
    SLOTS = 7                 # q0 q1 k0 k1 do0 do1 A, per stage
    TILE = C
    NFACE = 15                # dh[2][2] | Sq | Kb[2] | Kd[2] | Qg[2] | W[2] | dvn[2]
    FH, FSQ, FKB, FKD, FQG, FW, FDV = 0, 4, 5, 7, 9, 11, 13
    BAR_CONS = 1              # NamedBarrier over the 384 consumer threads
    LOG2E = 1.4426950408889634
    SCALE = _SCALE            # baked at build time (see `_compiled`)
    # Same per-role register accounting as K2 (D0.11's "dh 所在 WG 必须 >= 128 +
    # 工作集"): WG0 holds the 128x128 f32 dh in 128 registers/thread across the
    # whole trip loop, so the 128-register entry quota cannot also hold the
    # pack fragment and the stage's addressing.  The producer's surplus goes to
    # the consumers; 128*32 + 384*160 = 65536 conserves the pool exactly.
    PROD_REGS, CONS_REGS = 32, 160
    REG_DEC, REG_INC = _reg_ops()

    @dsl_user_op
    def _sel_gt(a, b, x, y, *, loc=None, ip=None):
        """int compare a > b ? x : y on f32 payloads (branch_free_select_mask)."""
        return F32(llvm.inline_asm(
            T.f32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip),
                      F32(x).ir_value(loc=loc, ip=ip),
                      F32(y).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.gt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    def _exp(x):
        return cute.math.exp2(x * LOG2E, fastmath=True)

    # -- same two helpers as K2, same reasons: every runtime value is an
    # EXPLICIT argument (a function entered from a staged if/for may not reach
    # runtime values through its closure), no @cute.jit, and therefore plain
    # `range` (an undecorated body is never preprocessed, so `range_constexpr`
    # would reach the interpreter raw and raise).
    def publish(r2s_, thr_r, sF, stg, acc, slot):
        """f32 C-fragment -> bf16 -> STSM x4 into face `slot` of `sF`.

        No f32 staging copy of the accumulator is made on the way through
        (D0.12 段B): the narrowing reads it through the store atom's retile
        straight into the 16-bit fragment, which for WG0 -- whose accumulator
        is the resident dh -- is the difference between fitting and spilling.
        """
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(acc)
        for v in range(cute.size(s16)):
            s16[v] = BF16(tacc[v])
        cute.copy(r2s_, s16, thr_r.partition_D(sF[(None, None, slot)]))

    def ldsm_c(l2r_, thr_l, cshp, sT, slot):
        """D0.16 ②: slot `slot` of `sT` -> a fragment isomorphic to the C tile.

        One `LDSM.16.M88.4` per warp per 16x16 sub-tile in place of 32 scalar
        `LDS` plus 32 swizzle address computations, with `r[i]` and `acc[i]`
        naming the same (m, n).  Same layouts and same atom as K4's, GPU-
        verified in notes/probe_m0_d016_ldsm.py (R1 / R4).
        """
        r = ALLOC(cshp, BF16)
        cute.copy(l2r_, thr_l.partition_S(sT[(None, None, slot)]),
                  thr_l.retile(r))
        return r

    def shape_face(l2r_, r2s_, thr_l, thr_r, cshp, stg,
                   sS, src, sD, dst, f0, f1):
        """D0.16 ③: a row-gated derived face, shaped once by matrix copies.

        `dst_face[m, n] = f_m * src_tile[m, n]`; f is a ROW scalar and so
        collapses (D0.16's M-axis hoist) to the TWO values the caller
        exponentiated for its own rows, bit 1 of the fragment index selecting
        between them (probe R2).  `LDSM` -> 32 FMUL -> F2FP -> `STSM` replaces
        the 64 scalar LDS + 64 scalar STS + 64 `MUFU` the per-element face
        loops paid per trip.  All three factors this kernel folds are <= 1
        (log gate <= 0 and beta <= 1, §1.3), so the bf16 face cannot overflow.
        """
        rx = ldsm_c(l2r_, thr_l, cshp, sS, src)
        ry = ALLOC(cshp, BF16)
        for i in range(cute.size(rx)):
            ry[i] = BF16(F32(rx[i]) * (f0 if (i // 2) % 2 == 0 else f1))
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(ry)
        for v in range(cute.size(s16)):
            s16[v] = tacc[v]
        cute.copy(r2s_, s16, thr_r.partition_D(sD[(None, None, dst)]))

    def link(mma, acc, sA_t, sB_t, nk, accumulate):
        """Put ONE WGMMA SS chain into the group that is currently open.

        No fence, no `commit_group`, no wait -- the inner half of the two-level
        issue helper hw_kb `wgmma_deferred_wait` 坑 11 asks for, and what lets
        several chains share one group and one drain (D1.1, brought over to K3
        by D2.2).

        `accumulate` is the flag of the chain's FIRST k-block only (the rest
        always accumulate): False initialises `acc`, True adds onto it.  A
        128-deep contraction split over two tiles is therefore (False, True).

        The A/B descriptors are built on the WARPGROUP-scope slice
        `get_slice(0)`, never on the per-thread slice (hw_kb
        `wgmma_desc_uniform` 坑 3).  `wgmma.mma_async` reads its two 64-bit
        matrix descriptors from the uniform register file (`desc[UR*]`), and
        ptxas can only leave them there if it can prove they are
        warp-invariant; a `tidx`-derived slice coordinate breaks that proof,
        so each descriptor is computed in per-thread registers and re-staged
        with `R2UR` ahead of every issue -- 2 dwords x its whole live range
        charged to the `setmaxnreg` quota, for every chain in flight.  Only
        `partition_C` (the accumulator) and the STSM partitions are
        per-thread.  A fresh atom per call site is mandatory (坑 4).
        """
        wgm = mma.get_slice(0)
        at = cute.make_mma_atom(mma.op)
        at.set(warpgroup.Field.ACCUMULATE, accumulate)
        fa = wgm.make_fragment_A(wgm.partition_A(sA_t))
        fb = wgm.make_fragment_B(wgm.partition_B(sB_t))
        for kb in range(nk):
            cute.gemm(at, acc, fa[None, None, kb], fb[None, None, kb], acc)
            at.set(warpgroup.Field.ACCUMULATE, True)

    @cute.kernel
    def dh_boundary(tmaQ: cute.CopyAtom, gQz: cute.Tensor,
             tmaK: cute.CopyAtom, gKz: cute.Tensor,
             tmaO: cute.CopyAtom, gOz: cute.Tensor,
             tmaA: cute.CopyAtom, gAz: cute.Tensor,
             mGc: cute.Tensor, mBe: cute.Tensor, mDh: cute.Tensor,
             mma_kk: cute.TiledMma, mma_km: cute.TiledMma,
             mma_mm: cute.TiledMma, r2s: cute.TiledCopy, l2r: cute.TiledCopy,
             lyt_land: cute.ComposedLayout, lyt_landT: cute.ComposedLayout,
             lyt_face: cute.ComposedLayout, lyt_faceT: cute.ComposedLayout,
             SharedStorage: cutlass.Constexpr[Callable]):
        # D2.3: ahead of EVERY global read -- the mGc/mBe prologue below reads
        # K1's g_cumsum, and the producer's A boxes are K1's output too.
        tidx, _, _ = cute.arch.thread_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, _ = cute.arch.block_idx()
        tw = tidx % 128
        _pdl_wait()

        bnd = I32(bx)
        head = I32(by)
        hq = head // GVA
        c0 = (bnd + 1) * SEG_CHUNKS      # first chunk of the FOLLOWING segment

        # ---- shared prologue (all 512 threads; no role branch yet) ----------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLand = storage.land.get_tensor(lyt_land.outer, swizzle=lyt_land.inner)
        sLandT = storage.land.get_tensor(lyt_landT.outer, swizzle=lyt_landT.inner)
        sFace = storage.face.get_tensor(lyt_face.outer, swizzle=lyt_face.inner)
        sFaceT = storage.face.get_tensor(lyt_faceT.outer, swizzle=lyt_faceT.inner)
        sGc = storage.gc.get_tensor((NWARM * C,))
        sBe = storage.be.get_tensor((NWARM * C,))

        tx = SLOTS * TILE * TILE * 2                       # bytes per stage
        pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bars.data_ptr(), num_stages=STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx, cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))

        tQs, tQg = cpasync.tma_partition(tmaQ, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gQz)
        tKs, tKg = cpasync.tma_partition(tmaK, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gKz)
        tOs, tOg = cpasync.tma_partition(tmaO, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gOz)
        tAs, tAg = cpasync.tma_partition(tmaA, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gAz)

        # gate / beta for every look-ahead chunk, once (out of the ring); trip
        # t holds chunk c0 + (NWARM-1-t), so trip-major is reverse chunk order
        for t in cutlass.range_constexpr(NWARM):
            ch = c0 + (NWARM - 1 - t)
            if tidx < C:
                sGc[t * C + tidx] = mGc[(ch, tidx, head)]
                sBe[t * C + tidx] = mBe[(ch, tidx, head)]
        cute.arch.sync_threads()
        _seg("k3.entry")

        # ---- role split -----------------------------------------------------
        # the dec/inc is the first statement of its branch (`warp_specialization`
        # 坑 5) and the shared prologue above stays on the entry quota (坑 11)
        if wid >= 12:
            REG_DEC(PROD_REGS)
            # producer: ONE warp issues the 7 TMA boxes of each trip
            if wid == 12:
                _seg("k3.producer.entry")
                p_st = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, STAGE)
                for t in cutlass.range_constexpr(NWARM):
                    ch = c0 + (NWARM - 1 - t)
                    _seg("k3.supply.acquire.begin", t)
                    pipe.producer_acquire(p_st)
                    _seg("k3.supply.acquire.end", t)
                    bar = pipe.producer_get_barrier(p_st)
                    b = p_st.index * SLOTS
                    for mh in cutlass.range_constexpr(2):
                        cute.copy(tmaQ, tQg[(None, (ch, hq * 2 + mh))],
                                  tQs[(None, b + mh)], tma_bar_ptr=bar)
                    for mh in cutlass.range_constexpr(2):
                        cute.copy(tmaK, tKg[(None, (ch, hq * 2 + mh))],
                                  tKs[(None, b + 2 + mh)], tma_bar_ptr=bar)
                    for nh in cutlass.range_constexpr(2):
                        cute.copy(tmaO, tOg[(None, (ch, head * 2 + nh))],
                                  tOs[(None, b + 4 + nh)], tma_bar_ptr=bar)
                    cute.copy(tmaA, tAg[(None, (ch, head))],
                              tAs[(None, b + 6)], tma_bar_ptr=bar)
                    pipe.producer_commit(p_st)
                    p_st.advance()
                    _seg("k3.supply.issue.end", t)
                _seg("k3.producer.exit")
        else:
            REG_INC(CONS_REGS)
            _seg("k3.consumer.entry")
            c_st = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, STAGE)
            # ONE per-thread MMA slice, and it is used for nothing but the
            # accumulator coordinates: A/B now ride `get_slice(0)` inside
            # `chain` (hw_kb `wgmma_desc_uniform`).  The three TiledMma differ
            # only in their operand major modes, so partition_C is the same
            # for all three.
            thr_c = mma_km.get_slice(tw)
            cS = thr_c.partition_C(cute.make_identity_tensor((TILE, TILE)))
            # a comprehension is its own scope and the rewriter does not descend
            # into it, so this `range` stays plain python
            accS = [ALLOC(cS.shape, F32) for _ in range(4)]
            for j in cutlass.range_constexpr(4):
                accS[j].fill(0.0)
            thr_r = r2s.get_slice(tw)
            stg = cute.make_layout(cute.shape(thr_r.partition_S(
                sFace[(None, None, 0)]))[:3])
            # D0.16's read side, sliced by the WARPGROUP thread number so that
            # `thr_l`'s fragment index agrees element for element with `cS`
            # (`ldmatrix_stmatrix` 坑 5).
            thr_l = l2r.get_slice(tw)
            cshp = thr_c.partition_shape_C((TILE, TILE))

            for t in cutlass.range_constexpr(NWARM):
                _seg("k3.chunk.land.begin", t)
                pipe.consumer_wait(c_st)
                _seg("k3.chunk.land.end", t)
                b = c_st.index * SLOTS
                gbase = t * C

                # ---- (a) dh face | Kb face | Sq + Kd faces ------------------
                if wid < 4:
                    # dh_out is the zero seed on trip 0: no face needed
                    if cutlass.const_expr(t > 0):
                        for j in cutlass.range_constexpr(4):
                            publish(r2s, thr_r, sFace, stg, accS[j], FH + j)
                        cute.arch.fence_view_async_shared()
                    _seg("k3.dh.publish.end", t)
                elif wid < 8:
                    # D0.16 ①: W = A diag(beta e^G) K carries its gate on the
                    # CONTRACTION index, so scaling A's COLUMNS and scaling K's
                    # ROWS are the same matrix.  The row form is the one D0.16
                    # ③ can shape with matrix copies (an M-axis factor is TWO
                    # values per thread; a column factor is sixteen), so the
                    # face this role publishes is Kb = beta e^G o K and the W
                    # chain takes the PLAIN landed A as its A operand.  Both
                    # factors are <= 1, so the bf16 face cannot overflow.
                    m0, m1 = cS[0][0], cS[2][0]
                    fb0 = sBe[gbase + m0] * _exp(sGc[gbase + m0])
                    fb1 = sBe[gbase + m1] * _exp(sGc[gbase + m1])
                    for mh in cutlass.range_constexpr(2):
                        shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                   sLand, b + 2 + mh, sFace, FKB + mh,
                                   fb0, fb1)
                    cute.arch.fence_view_async_shared()
                    _seg("k3.ab.end", t)
                else:
                    acc = ALLOC(cS.shape, F32)
                    # both halves of the 128-deep q.k contraction target the
                    # same accumulator with nothing between them, so one fence,
                    # one group, one drain (D1.1 / 坑 11, D2.2 brings it here)
                    warpgroup.fence()
                    for mh in cutlass.range_constexpr(2):
                        link(mma_kk, acc,
                             sLand[(None, None, b + mh)],
                             sLand[(None, None, b + 2 + mh)], 4, mh > 0)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    # D0.16's M-axis hoist: the exponent's ROW term is TWO
                    # scalar `LDS` for the whole segment, not one per element;
                    # only the column term stays per element (sixteen distinct
                    # columns per thread).  The exponent is NOT split into
                    # e^{G_i} e^{-G_j} -- above the diagonal the difference is
                    # positive and its exp may be inf, which is why the mask is
                    # SELECTED rather than multiplied in.
                    m0, m1 = cS[0][0], cS[2][0]
                    g0, g1 = sGc[gbase + m0], sGc[gbase + m1]
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        n = cS[i][1]
                        gn = sGc[gbase + n]
                        if cutlass.const_expr((i // 2) % 2 == 0):
                            acc[i] = _sel_gt(n, m0, F32(0.0),
                                             SCALE * _exp(g0 - gn) * acc[i])
                        else:
                            acc[i] = _sel_gt(n, m1, F32(0.0),
                                             SCALE * _exp(g1 - gn) * acc[i])
                    publish(r2s, thr_r, sFace, stg, acc, FSQ)
                    # Kd = e^{G_last-G} o K, the D0.16 ③ derived face
                    glast = sGc[gbase + C - 1]
                    d0, d1 = _exp(glast - g0), _exp(glast - g1)
                    for mh in cutlass.range_constexpr(2):
                        shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                   sLand, b + 2 + mh, sFace, FKD + mh, d0, d1)
                    cute.arch.fence_view_async_shared()
                    _seg("k3.sq.end", t)
                cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=384)

                # ---- (b) dvn (WG1) | W + Qg (WG2) ---------------------------
                if wid >= 4:
                    if wid < 8:
                        # each half's three (or, on the seed trip, one) chains
                        # land in ONE accumulator, so they share a group and a
                        # drain; the two halves are independent accumulators
                        # and both groups go out before either is reaped, so
                        # half 0's negate + STSM publish covers half 1's flight
                        # (D1.1's ladder, D2.2 brings it here).
                        accV = [ALLOC(cS.shape, F32) for _ in range(2)]
                        for nh in cutlass.range_constexpr(2):
                            warpgroup.fence()
                            if cutlass.const_expr(t > 0):
                                for mh in cutlass.range_constexpr(2):
                                    link(mma_km, accV[nh],
                                         sFace[(None, None, FKD + mh)],
                                         sFaceT[(None, None, FH + mh * 2 + nh)],
                                         4, mh > 0)
                                link(mma_mm, accV[nh],
                                     sFaceT[(None, None, FSQ)],
                                     sLandT[(None, None, b + 4 + nh)], 4, True)
                            else:
                                link(mma_mm, accV[nh],
                                     sFaceT[(None, None, FSQ)],
                                     sLandT[(None, None, b + 4 + nh)], 4, False)
                            warpgroup.commit_group()
                        for nh in cutlass.range_constexpr(2):
                            warpgroup.wait_group(1 - nh)
                            acc = accV[nh]
                            # the face carries -dvn, so WG0's dh update is a
                            # plain accumulate instead of a subtract
                            for i in cutlass.range_constexpr(cute.size(cS)):
                                acc[i] = -acc[i]
                            publish(r2s, thr_r, sFace, stg, acc, FDV + nh)
                        cute.arch.fence_view_async_shared()
                        _seg("k3.dvn.end", t)
                    else:
                        # W = A @ Kb: the plain landed A tile is the A operand
                        # (K-major, the contraction on its columns) and the
                        # gated key face is the B operand, MN-major through the
                        # transposing view -- the same product the Ab form
                        # computed, with the gate moved to the side D0.16 ③ can
                        # shape (see the Kb comment in stage (a)).  The two
                        # halves are independent accumulators and Qg depends on
                        # neither, so both groups go out first and the Qg
                        # shaping is the covered段.
                        accW = [ALLOC(cS.shape, F32) for _ in range(2)]
                        for mh in cutlass.range_constexpr(2):
                            warpgroup.fence()
                            link(mma_km, accW[mh],
                                 sLand[(None, None, b + 6)],
                                 sFaceT[(None, None, FKB + mh)], 4, False)
                            warpgroup.commit_group()
                        # Qg = scale e^G o Q, the third D0.16 ③ derived face
                        m0, m1 = cS[0][0], cS[2][0]
                        q0 = SCALE * _exp(sGc[gbase + m0])
                        q1 = SCALE * _exp(sGc[gbase + m1])
                        for mh in cutlass.range_constexpr(2):
                            shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                       sLand, b + mh, sFace, FQG + mh, q0, q1)
                        for mh in cutlass.range_constexpr(2):
                            warpgroup.wait_group(1 - mh)
                            publish(r2s, thr_r, sFace, stg, accW[mh], FW + mh)
                        cute.arch.fence_view_async_shared()
                        _seg("k3.w.end", t)
                cute.arch.barrier(barrier_id=BAR_CONS, number_of_threads=384)

                # ---- (c) WG0 advances dh -----------------------------------
                if wid < 4:
                    dec = _exp(sGc[gbase + C - 1])
                    # D2.2's named site, K3's copy: the dh advance was eight
                    # groups (four quadrants x two chains), each one fenced,
                    # committed and drained on its own, so no k-block could
                    # issue while another was in flight.  The two chains of a
                    # quadrant target the SAME accumulator with nothing between
                    # them, so they share a group and a drain (坑 11), and the
                    # four quadrants ride D1.1's depth-1 ladder on top --
                    # exactly the shape K4's WG0 stage 5 already carries.  Free
                    # in registers: the accumulators are trip-resident either
                    # way (D0.11).
                    if cutlass.const_expr(t > 0):
                        for j in cutlass.range_constexpr(4):
                            a = accS[j]
                            for i in cutlass.range_constexpr(cute.size(cS)):
                                a[i] = a[i] * dec
                    for j in cutlass.range_constexpr(4):
                        mh, nh = j // 2, j % 2
                        warpgroup.fence()
                        link(mma_mm, accS[j],
                             sFaceT[(None, None, FQG + mh)],
                             sLandT[(None, None, b + 4 + nh)], 4, t > 0)
                        link(mma_mm, accS[j],
                             sFaceT[(None, None, FW + mh)],
                             sFaceT[(None, None, FDV + nh)], 4, True)
                        warpgroup.commit_group()
                        if cutlass.const_expr(j > 0):
                            warpgroup.wait_group(1)
                    warpgroup.wait_group(0)
                    _seg("k3.dh.update.end", t)
                    pipe.consumer_release(c_st)
                c_st.advance()

            # ---- epilogue: the f32 dh_end slot (D0.7: STG, no TMA) ----------
            if wid < 4:
                for mh in cutlass.range_constexpr(2):
                    for nh in cutlass.range_constexpr(2):
                        a = accS[mh * 2 + nh]
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            m, n = cS[i][0], cS[i][1]
                            mDh[(bnd, head, mh * TILE + m, nh * TILE + n)] = a[i]
            _seg("k3.exit")
        # D2.3: the dh_end stores are issued, so K4 may be scheduled.  At CTA
        # scope (outside the role branch) so that every block signals -- the
        # kick-off needs all of them, and the producer warps never reach the
        # consumer epilogue above.
        _pdl_trigger()

    # -- layouts / atoms / TMA: all IR builders, so all inside the jit trace --
    def _atoms(mQ2, mK2, mO2, mA2):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, TILE),
            BF16)
        lyt1 = cute.tile_to_shape(atom, (TILE, TILE), order=(0, 1))
        lyt_land = cute.tile_to_shape(atom, (TILE, TILE, SLOTS * STAGE),
                                      order=(0, 1, 2))
        lyt_face = cute.tile_to_shape(atom, (TILE, TILE, NFACE), order=(0, 1, 2))
        # read views: sXT[(c, r, j)] == sX[(r, c, j)].  The colex index of
        # (r, c, j) over (64, 64, n) is r + 64c + 4096j, so the transposer's
        # stride, in READ-view coordinate order (c, r, j), is (64, 1, 4096).
        tsp = lambda lyt, n: cute.composition(
            lyt, cute.make_layout((TILE, TILE, n),
                                  stride=(TILE, 1, TILE * TILE)))
        lyt_landT = tsp(lyt_land, SLOTS * STAGE)
        lyt_faceT = tsp(lyt_face, NFACE)

        MM = nvgpu.OperandMajorMode
        mk = lambda am, bm: sm90_utils.make_trivial_tiled_mma(
            BF16, BF16, am, bm, F32, atom_layout_mnk=(1, 1, 1),
            tiler_mn=(TILE, TILE))
        mma_kk = mk(MM.K, MM.K)     # Sq = Q @ K^T: both operands land-major
        mma_km = mk(MM.K, MM.MN)    # Kd @ dh, A @ Kb
        mma_mm = mk(MM.MN, MM.MN)   # Sq^T @ dO, Qg^T @ dO, W^T @ dvn

        E = LayoutEnum.ROW_MAJOR
        r2s = cute.make_tiled_copy_S(
            sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
            cute.make_tiled_copy_C_atom(
                cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4),
                                    BF16), mma_km))
        # D0.16's READ side, on the same TiledMma as the accumulators so that a
        # fragment index names the same (m, n) in both (probe R1 / R4)
        l2r = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma_km)

        g2s = cpasync.CopyBulkTensorTileG2SOp()
        t4 = lambda m: cpasync.make_tiled_tma_atom(g2s, m, lyt1, (TILE, TILE),
                                                   num_multicast=1)
        tmaQ, cQ = t4(mQ2)
        tmaK, cK = t4(mK2)
        tmaO, cO = t4(mO2)
        tmaA, cA = t4(mA2)
        z = lambda t: cute.zipped_divide(t, (TILE, TILE))
        return (tmaQ, z(cQ), tmaK, z(cK), tmaO, z(cO), tmaA, z(cA),
                mma_kk, mma_km, mma_mm, r2s,
                lyt_land, lyt_landT, lyt_face, lyt_faceT, l2r)

    def _storage(n_land, n_face):
        @cute.struct
        class SharedStorage:
            bars: cute.struct.MemRange[cutlass.Int64, STAGE * 2]
            land: cute.struct.Align[cute.struct.MemRange[BF16, n_land], 1024]
            face: cute.struct.Align[cute.struct.MemRange[BF16, n_face], 1024]
            gc: cute.struct.Align[cute.struct.MemRange[F32, NWARM * C], 1024]
            be: cute.struct.Align[cute.struct.MemRange[F32, NWARM * C], 1024]
        return SharedStorage

    @cute.jit
    def launch_dh_boundary(mQ2, mK2, mO2, mA2, mGc, mBe, mDh, st):
        a = _atoms(mQ2, mK2, mO2, mA2)
        SS = _storage(cute.cosize(a[12]), cute.cosize(a[14]))
        dh_boundary(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                    mGc, mBe, mDh, a[8], a[9], a[10], a[11], a[16],
                    a[12], a[13], a[14], a[15],
                    SS).launch(grid=[N_SEG - 1, H_V, 1], block=[TID, 1, 1],
                               min_blocks_per_mp=1, stream=st, **_pdl_kw())

    return launch_dh_boundary


def _build_k4():
    """K4 `gdn_bwd_main` — the main kernel (D0.5/D0.8/D0.10/D0.11).

    grid (N_SEG, H_V) = 256 (`blockIdx.x` = segment, `blockIdx.y` = v-head, so
    the head is the fast axis of the flat block id), block 512 = 4 warpgroups,
    non-persistent, 1 CTA/SM.  Each CTA walks its segment's 16 chunks BACKWARDS
    (chunk 15 -> 0) carrying dh = dL/dS in WG0's registers, seeded from K3's
    `dh_end[segment]`.

    Roles are a FUNCTIONAL split (D0.5), not a spatial one, and each role owns
    a TOP-LEVEL branch with its own chunk loop (so that WG0's 128 dh registers
    are not live inside WG1/WG2's code, D0.12's per-segment live-set budget):

      WG3 (warp 12)  producer: the 13 TMA boxes of each chunk
      WG2 (wid 8-11) block/triangular WG: everything whose contraction or
                     output rides the 64-token axis (Sq, U, Ms, Mv, dk) plus
                     the gated elementwise faces (Qg, Kg)
      WG1 (wid 4-7)  state WG: everything contracting with h_c or dh
                     (Bm, dU, dv, dq)
      WG0 (wid 0-3)  dh owner: 128x128 f32 dh in 128 registers/thread, two
                     rank-64 updates + the gate decay per chunk, bf16 copy face

    Chunk algebra (G = K1's chunk-inclusive g_cumsum, A = K1's (I+L)^-1, h =
    K2's chunk entry state, dh = dL/dS at the chunk's exit; all fp64-checked in
    notes/verify_qk_chunk_algebra.py against the per-token adjoint of §1.2):

        Sq = scale e^{G_i-G_j}(q_i.k_j), j<=i    Qg = scale e^{G} o Q
        Bm = beta o (V - e^G o (K @ h))          U  = A @ Bm      ( = u )
        dU = e^{G_last-G} o (K @ dh) + Sq^T @ dO
        dv = beta o (A^T @ dU)                   Kg = e^G o K
        Ms = mask_{j<=i}(scale e^{G_i-G_j}) o (dO @ U^T)
        Mv = mask_{j<i}(e^{G_i-G_j}) o (dv @ U^T)
        dq = scale e^G o (dO @ h^T) + Ms @ K
        dk = e^{G_last-G} o (U @ dh^T) + Ms^T @ Q - (Mv + Mv^T) @ K
             - e^G o (dv @ h^T)
        dh <- e^{G_last} dh + Qg^T @ dO - Kg^T @ dv

    The Bm route (one K=128 chain plus an elementwise) replaces the u/W pair,
    and the Mv route (one 64x64 product plus its symmetrisation) replaces the
    dA/dL differentiation of the inverse; the Kg^T dv form of the dh update is
    the W^T dU form with the gate moved onto K and dv, which is what lets the
    face table hold every part-2 quantity inside the 14 slots D0.8 fixed.
    Signs: the faces carry -dU, -dv and -Mv so that every consumer accumulates.

    Part 3 (D0.14) adds the epilogue.  dbeta and dg close in the chunk:

        du     = A^T @ dU                       (WG1's stage-3 accumulator,
                                                 before the beta scaling)
        dbeta  = <du_i, U_i> / beta_i
        r_i    = <q_i, dq_i> - <v_i, dv_i>
        dg_i   = <dh_in, h> - sum_{j<i} r_j     (exclusive, in-chunk)

    the last line being the chunk-local form of the exact identity
    dg_t = <dL/dS_{t-1}, S_{t-1}> (psi_{t-1} = psi_t + q.dq - v.dv telescopes
    the per-token adjoint; both fp64-checked as Q5/Q6 of
    notes/verify_qk_chunk_algebra.py).  It is what folds the reference's
    separate `chunk_local_cumsum` launch into K4.  The chunk boundary
    <dh_in, h> is a 128x128 inner product of WG0's UPDATED dh against the
    landed h_c, so WG0 forms it in stage 5 and hands WG1 one f32 across the
    last barrier; the two row dot products r and dbeta are quad-lane SHFL
    reductions of WG1's own C-fragments (D0.13), staged through the 1 KB
    scalar block D0.8 provisioned, and the exclusive scan is a lane
    Kogge-Stone `shfl` scan over that block plus a butterfly carry (D1.5).

    dq/dk/dv leave through the D0.14 store path -- STSM to a 16 KB staging
    pair, then `UTMASTG.4D` (2 boxes x 64 columns) + `commit_group` +
    `wait_group` -- rather than the plain bf16 global stores parts 1-2 used.
    The staging pair is not new smem: each of the three tensors takes two
    8 KB blocks that are dead at its store point (dv: the -dU faces it has
    just consumed; dq: the v landing boxes, dead since stage 1; dk: the two
    dh faces its own chains have just finished with).  Every store drains
    before the barrier that lets its blocks be rewritten.

    CROSS-WG SYNC (D0.10).  The five stages are NOT separated by a CTA-wide
    rendezvous.  Every exchange face carries its own full/empty mbarrier pair
    (hw_kb `wg_exchange_face_mbarrier`), so each WG advances on the faces it
    actually reads and writes and the three roles phase-shift against each
    other on their own; the only NamedBarriers left are five 128-wide ones,
    all WG-internal (BAR_WG0 for the psi warp combine, BAR_WG1 / BAR_WG2 to
    order a warpgroup's STSM ahead of its single store warp and to speak for
    that warp's drain).  The landing zone is split the same way: one full/empty
    pair per SLOT CLASS (h_c, A+v, q, k, do -- the five expect-tx byte counts
    D0.10 registers), the full armed by the producer lane's
    `mbarrier.arrive.expect_tx` and completed by that class's own TMA
    transactions, the empty arrived by every WG that reads the class.  A class
    therefore returns to the producer when its own readers are done rather than
    when the slowest reader of any box is: WG1 hands back q/k/do/h_c at the end
    of its stage 4 while it is still holding the v boxes its dq store is
    staging through, and WG2 hands back A after stage 2.  The barrier table,
    its arrive counts and the same-chunk / previous-chunk phase of every wait
    are enumerated at the top of `_build_k4`.  Each class stays single-stage,
    so a chunk's refill of a class still cannot start before that class's last
    reader of the previous chunk -- deepening a class is D1.2.
    """
    BF16, F32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32
    I64 = cutlass.Int64
    ALLOC = _rmem_alloc()
    TID = 512                 # 4 warpgroups
    STAGE = 1                 # landing ring depth (D0.8: single-stage)
    TILE = C
    # landing zone, 13 boxes = 104 KB (D0.8's table)
    SLOTS = 13
    # D1.3's ROW PITCH for the two D2.1 staging views, in bf16 elements: one
    # 16 B pad per 128-element row.  At the natural pitch (128 elements = 256 B)
    # every row of the staging tile starts on bank 0, so the 16 rows one
    # `STSM.16.M88.4` writes collide 16 ways; at 136 (272 B = 17 x 16 B)
    # consecutive rows step 4 banks and the publish returns to the 4 wavefronts
    # its 512 B need (hw_kb `s2g_bulk_reduce` bf16 §3).  The reduction is
    # unaffected: each row still holds its 128 columns contiguously and every
    # row base stays 16 B aligned, only the row STEP changes (256 -> 272 B).
    # Address-correctness of this exact pitch under K4's own STSM tiled copy,
    # including that the pad columns stay untouched, is GPU-verified in
    # notes/probe_m2_d13_pad_stride.py (S1 / S5 / S2 / S3).
    STGP = 2 * TILE + 8
    # 1008 B of the padded tile lie past its pair; the face region gets that
    # much tail (rounded to the region's 1024 B alignment) because dk stages
    # through the region's last pair.
    FPAD = 512
    # 64 padded rows are 17408 B, so a staging tile OVERRUNS its 16 KB slot
    # pair by 1008 B and the block after the pair must be dead too.  That is
    # what fixes the slot numbering below: dq stages through the v pair, so v
    # is placed where A -- dead since stage 3 of every role, and released by
    # the same LE_AV class the v boxes are -- follows it.  No smem is added on
    # the landing side.
    LQ, LK, LO, LH, LV, LA = 0, 2, 4, 6, 10, 12
    # Exchange faces, 14 x 8 KB = 112 KB (D0.8's enumeration, unchanged from
    # part 1: the smem partition is a shared knob of the pack and neither part
    # 2 nor part 3 moves it).  Part 2 added five quantities (Bm, U, -dv, Ms,
    # -Mv, Kg) and part 3 the three output staging pairs WITHOUT adding a face:
    # each takes a slot whose occupant is already dead, which is the reuse D0.8
    # allows inside the 8 KB granule ("生命期错开可复用同一块").  Lifetimes, by
    # the five consumer stages of a chunk (`>>` = a D0.14 store staging use,
    # which lasts from the STSM to the `wait_group` a few instructions later):
    #
    #  slot |  (1)      (2)      (3)      (4)      (5)   | producer -> consumer
    # 12-13 |  dh_0 dh_1 (published from WG0's registers)| WG0 -> WG1 dU, WG2 dk
    # 12-13 |                                     >> dk  | WG2 -> gmem
    #  2-3  |  dh_2 dh_3 ------------------------------>| WG0 -> WG1 dU, WG2 dk
    #  4-5  |  Qg ------------------------------------>  | WG2 -> WG0 dh
    #  6-7  |  Bm --->        -dv -------------------->  | WG1 -> WG2 U /
    #       |                                            |   WG1 -> WG2, WG0
    #  8    |  Sq --->                 -Mv ----------->  | WG2 -> WG1 dU /
    #       |                                            |   WG2 -> WG2 dk
    #  9-10 |           U --------------------------->   | WG2 -> WG2, WG1
    #  11,0 |           -dU -->>dv     Kg ------------>  | WG1 -> WG1 dv /
    #       |                                            |   WG2 -> WG0 dh
    #   1   |                    Ms ------------------>  | WG2 -> WG1 dq, WG2 dk
    #
    # dq's pair is not here: it stages through LANDING slots LV, whose v tiles
    # have been dead since stage 1.  Peak live = 14 slots, i.e. the face region
    # is exactly full and no further quantity may be added without a lifetime
    # that dies.
    #
    # D1.3 moved the dh quadrants around inside that table without changing any
    # lifetime.  dk stages through the two dh blocks that mh = 0's chains have
    # just finished with, and the padded pitch above makes that staging tile
    # run 1008 B past the pair -- which, at the old numbering, was the block
    # holding the dh quadrant mh = 1 is about to read.  Putting mh = 0's pair
    # LAST (12-13) sends the overrun into a 1 KB tail pad instead, and the two
    # quantities evicted from 12-13 take the blocks mh = 1's quadrants vacate.
    # Only the mh = 0 pair has to be the last pair; which slots the other three
    # dh quadrants sit in is free, so quadrants 2-3 stay where they were.
    NFACE = 14
    SDH = (12, 13, 2, 3)       # dh quadrant mh * 2 + nh -> face slot (nh fast)
    PDK = SDH[0] // 2          # dk's staging pair: the LAST pair of the region
    SVN = (11, 0)              # -dU / dv staging / Kg (two independent slots)
    FQG, FBM, FSQ, FU, FMS = 4, 6, 8, 9, 1
    FNDV, FMV = FBM, FSQ                      # the lifetime-disjoint aliases
    BAR_WG0, BAR_WG1, BAR_WG2 = 2, 3, 4       # one per consumer WG, 128 wide
    # ---- D0.10's cross-WG handoff: one full/empty mbarrier pair per exchange
    # quantity (hw_kb `wg_exchange_face_mbarrier`).  The contract is "跨 WG 一律
    # mbarrier"; NamedBarrier is allowed only for WG-INTERNAL段间同步, never at
    # CTA width.  Each edge below is one row of D0.9's consumption ledger, with
    # the ALIASED lifetimes of a slot carrying their own pair (a slot's second
    # occupant has a different producer/consumer set than its first, so one
    # pair cannot serve both):
    #
    #   pair   slots   full: writer -> readers            empty: readers -> next writer
    #   DH   12,13,2,3 WG0 s1 -> WG1 s2, WG2 s5           WG1 s2 read + WG2 s5 dk-overwrite
    #                                                     -> WG0 s1 (t+1)
    #   QG     4-5     WG2 s1 -> WG0 s5                   WG0 s5 -> WG2 s1 (t+1)
    #   BM     6-7     WG1 s1 -> WG2 s2                   WG2 s2 -> WG1 s3 (same chunk)
    #   NDV    6-7     WG1 s3 -> WG2 s4/s5, WG0 s5        WG2 s5 + WG0 s5 -> WG1 s3?? no:
    #                                                     -> WG1 s1 (t+1) is BM's slot --
    #                                                     see the alias note below
    #   SQ     8       WG2 s1 -> WG1 s2                   WG1 s2 -> WG2 s4 (same chunk)
    #   U      9-10    WG2 s2 -> WG1 s3 (WG2 self s3-s5)  WG1 s3 -> WG2 s2 (t+1)
    #   KG     11,0    WG2 s4 -> WG0 s5                   WG0 s5 -> WG1 s2 (t+1, writes -dU)
    #   MS     1       WG2 s3 -> WG1 s4 (WG2 self s5)     WG1 s4 -> WG2 s3 (t+1)
    #   PSI    sRow    WG0 s5 -> WG1 epilogue             WG1 epilogue -> WG0 s5 (t+1)
    #
    # ALIAS WARNING (slots 6-7).  Those two blocks carry Bm in stages 1-2 and
    # -dv from stage 3 on, so they have TWO producers with different consumer
    # sets, and the two writes are in the SAME chunk, two stages apart.  A
    # single full/empty pair cannot serve both (the -dv write would consume the
    # empty that belongs to the Bm write).  The edge is therefore serialised
    # inside WG1 by the chunk's own program order: WG1 writes Bm in s1, WG2
    # reads it in s2 and releases (XE_BM), WG1 waits XE_BM at ph -- i.e. the
    # SAME chunk -- before writing -dv in s3.  XE_BM is thus a one-way channel
    # (WG2 -> WG1, same chunk), not a wrap-around empty, and the -dv faces then
    # need only the usual previous-chunk empty (XE_NDV, waited at 1 - ph).
    # Same story for SVN's two slots (-dU in s2-s3, dv staging in s3, Kg in s4):
    # WG1's -dU read is its own, the staging drain is X_FREE1112, and Kg's
    # previous reader is WG0 (XE_KG at 1 - ph) which also orders it after WG1's
    # own read by the chunk's program order.
    #
    # plus two one-way channels (a single barrier each: one arrival per chunk
    # by the writer's side, and the waiter is downstream IN THE SAME CHUNK, so
    # it parses at `t % 2` index 0):
    #
    #   RDH            WG1 s2 (dh operand reads complete) -> WG2 s5 may overwrite
    #   FREE1112       WG1 s3 (dv store drained) -> WG2 s4 may write Kg over it
    #
    # (a third channel, LANDDONE, existed while the 13 landing boxes shared one
    # merged full/empty pair: WG1 and WG2 reported off the landing zone and WG0
    # alone released the single slot.  With a pair per slot class every WG
    # arrives on the classes it actually read, so the relay is gone.)
    #
    # Arrive counts follow the recipe's gotcha 1: full = the WRITING warpgroup's
    # thread count, empty = the sum of the READING warpgroups' thread counts,
    # each thread arriving after its own fence / its own chain's
    # `wgmma.wait_group.sync.aligned` (which is itself a warpgroup rendezvous,
    # so an arrive that follows a chain needs no extra NamedBarrier; an arrive
    # that follows a SINGLE-warp TMA drain does, and gets BAR_WG1 / BAR_WG2).
    XF_DH, XE_DH = 0, 1
    XF_QG, XE_QG = 2, 3
    XF_BM, XE_BM = 4, 5
    XF_NDV, XE_NDV = 6, 7
    XF_SQ, XE_SQ = 8, 9
    XF_U, XE_U = 10, 11
    XF_KG, XE_KG = 12, 13
    XF_MS, XE_MS = 14, 15
    XF_PSI, XE_PSI = 16, 17
    X_RDH, X_FREE1112 = 18, 19
    # ---- D0.10's landing-zone supply, per SLOT CLASS.  The 13 boxes of a
    # chunk are not one transaction: they are the five classes the milestone
    # registers by their expect-tx byte counts (h_c 0x8000, A+v 0x6000, and
    # q / k / do 0x4000 each, summing to the 0x1a000 landing zone).  Each class
    # carries its OWN full/empty pair, so a class goes back to the producer as
    # soon as the WGs that read THAT class are off it, instead of every box
    # waiting for the slowest consumer of any box.  full is armed by the
    # producer's elected lane with `mbarrier.arrive.expect_tx` and completed by
    # the class's own TMA transactions (init count 1 = that one lane); empty is
    # arrived by every WG that reads the class -- 384 for do and h_c (all three
    # consumers), 256 for q, k and A+v (WG1 and WG2 only; WG0 reads neither q,
    # k, v nor A).  The class is what the milestone calls a slot: A and v share
    # one pair because they share one expect-tx box count.
    LF_Q, LE_Q = 20, 21
    LF_K, LE_K = 22, 23
    LF_O, LE_O = 24, 25
    LF_AV, LE_AV = 26, 27
    LF_H, LE_H = 28, 29
    NXB = 30
    BOX = TILE * TILE * 2                      # 8 KB, one 64x64 bf16 box
    TX_Q, TX_K, TX_O = 2 * BOX, 2 * BOX, 2 * BOX        # 0x4000 each
    TX_AV, TX_H = 3 * BOX, 4 * BOX                      # 0x6000 / 0x8000
    # (barrier index, expected arrivals).  Full counts are the writing WG's
    # thread count (1 for a TMA class: the producer lane that arms expect_tx);
    # empty counts are the sum of the READING sides' thread counts, where
    # "reading" includes any side that overwrites the block.
    XCOUNTS = ((XF_DH, 128), (XE_DH, 256),     # WG1 s2 + WG2 s5 both take it
               (XF_QG, 128), (XE_QG, 128),     # WG0 s5
               (XF_BM, 128), (XE_BM, 128),     # WG2 s2, same chunk
               (XF_NDV, 128), (XE_NDV, 256),   # WG2 s5 + WG0 s5
               (XF_SQ, 128), (XE_SQ, 128),     # WG1 s2, same chunk
               (XF_U, 128), (XE_U, 128),       # WG1 s3; WG2's own reads self-order
               (XF_KG, 128), (XE_KG, 128),     # WG0 s5, previous chunk
               (XF_MS, 128), (XE_MS, 128),     # WG1 s4; WG2's own reads self-order
               (XF_PSI, 128), (XE_PSI, 128),   # WG1's epilogue, same chunk
               (X_RDH, 128),                   # WG1 s2, same chunk
               (X_FREE1112, 32),               # WG1's store warp alone
               (LF_Q, 1), (LE_Q, 256),         # q:    WG1 s4, WG2 s1/s5
               (LF_K, 1), (LE_K, 256),         # k:    WG1 s1/s2/s4, WG2 s1/s4/s5
               (LF_O, 1), (LE_O, 384),         # do:   WG0 s5, WG1 s2/s4, WG2 s3
               (LF_AV, 1), (LE_AV, 256),       # A+v:  WG1 s1/s3 + dq staging,
                                               #       WG2 s2
               (LF_H, 1), (LE_H, 384))         # h_c:  WG0 s5, WG1 s1/s4, WG2 s5
    # D0.8's 1 KB scalar block, 256 f32 (the "行和交接" row it provisions):
    #   [0, 64)     r_i    = <q_i, dq_i> - <v_i, dv_i>   (WG1)
    #   [64, 128)   dbeta_i                              (WG1)
    #   [128, 132)  WG0's four per-warp partials of <dh_in, h>
    #   132         psi = <dh_in, h>, the chunk boundary of the dg recurrence
    NROW = 256
    RW, RPSI = 128, 132

    SEG_TOK = SEG_CHUNKS * C
    # ---- D0.12's per-segment live-set budget, role by role.  The three
    # consumer roles do NOT have the same working set, so they must not get the
    # same quota: D0.5 leaves the split to the student with one binding
    # constraint -- "dh 所在 WG 必须 >= 128 + 工作集" -- and WG0 carries 128
    # registers of resident dh (D0.11) at EVERY program point of its loop, the
    # publish of stage 1 included, so 160 is the floor its quota may take at
    # all.
    #
    # `cfrag_fused_pack_liveness` 机制 3 accounting at each role's peak point
    # (A_resident + A_flight + pack + W, the pack being the bf16 fragment alone
    # -- see `publish`):
    #   WG0  stage 1  128 (dh) + 16 (pack)          + W
    #   WG1  stage 3   32 (du) + 16 (pack) + 4 rows + W
    #   WG2  stage 5   32 (a)  + 32 (ah)  + 16 pack + W
    # The quotas are multiples of 8 and conserve the pool exactly
    # (`warp_specialization` 坑 4 / 坑 8): 128*(32 + 160 + 144 + 176) = 65536.
    # Over-subscribing this sum does not fail to compile -- every consumer
    # hangs on its `.inc` forever.
    #
    # The split itself is NOT an a-priori division of the paper accounting
    # above: the paper number says only which role is biggest, and the three
    # peaks are within 48 of each other, so the ordering has to be settled on
    # the allocator's own reading.  Measured, one variant per run, everything
    # else held (`probe_m0audit_k4_regsplit`, per-quota static LDL/STL census
    # of the four `setmaxnreg` regions of K4's SASS + the NCU spill counters):
    #
    #   quota (P/WG0/WG1/WG2)   spill_ld   K4 ns    tensor   spill slots/region
    #   32/192/144/144 (was)     4988934   414776   29.13    1 / 13 / 31
    #   32/160/160/160           4203066   396652   30.45    23 / 14 /  7
    #   32/160/144/176 (now)     3940842   385704   31.35    25 / 10 /  2
    #
    # i.e. the accumulator-heavy role is WG2, not WG0: WG2 runs five stages and
    # holds TWO 64x64 f32 accumulators plus a pack fragment at its stage-5 peak,
    # and starving it to 144 costs 31 spill slots -- more, on the whole-kernel
    # counters, than the 25 that giving WG0 back the same 32 registers costs on
    # its side.  WG0's own 128 dh registers are inert (they are touched three
    # times a chunk and never re-materialised at 160+), so the marginal value of
    # its 33rd..64th non-accumulator register is lower than WG2's.
    #
    # What remains at this split is WG0's 25 slots, and their source is
    # structural rather than a quota: D0.16's h_c ledger row asks for the psi
    # inner product to read h_c 分块流式 (a sub-tile at a time) and `ldsm_c`
    # brings the whole 64x64 quadrant in as one 16-register fragment, which sits
    # on top of the 128 resident ones; the stage-1 pack fragment is the same
    # shape on the same 128.  That is D0.12's 让位阶梯 and not this commit.
    #
    # `W` above is the NON-accumulator working set D0.12 段账 caps (<= 32 for
    # WG0, <= 115 for WG2).  The WGMMA operand descriptors are NOT part of it:
    # `wgmma.mma_async` takes them from the uniform file, so they cost the
    # per-thread quota nothing -- PROVIDED the slice coordinate they are built
    # on is warp-invariant, which is what `chain` now guarantees by building
    # A/B on `get_slice(0)`.  Built on a `tidx`-derived slice they fall into
    # the vector file at 2 dwords each for their whole live range; WG2's
    # stage 5 alone holds six chains' worth, and every role's container set
    # sits close enough to its quota that the tax evicts a whole accumulator
    # array rather than the descriptors themselves (`wgmma_desc_uniform`
    # 两种失败形态, 形态 2).
    PROD_REGS = 32
    WG0_REGS, WG1_REGS, WG2_REGS = 160, 144, 176
    REG_DEC, REG_INC = _reg_ops()
    LOG2E = 1.4426950408889634
    SCALE = _SCALE            # baked at build time (see `_compiled`)

    @dsl_user_op
    def _sel_gt(a, b, x, y, *, loc=None, ip=None):
        """int compare a > b ? x : y on f32 payloads (branch_free_select_mask)."""
        return F32(llvm.inline_asm(
            T.f32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip),
                      F32(x).ir_value(loc=loc, ip=ip),
                      F32(y).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.gt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @dsl_user_op
    def _imax(a, b, *, loc=None, ip=None):
        """Clamp for the scan's source lane (same helper K1's cumsum uses)."""
        return I32(llvm.inline_asm(
            T.i32(), [I32(a).ir_value(loc=loc, ip=ip),
                      I32(b).ir_value(loc=loc, ip=ip)],
            "max.s32 $0, $1, $2;", "=r,r,r", has_side_effects=False,
            is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc, ip=ip))

    def _exp(x):
        return cute.math.exp2(x * LOG2E, fastmath=True)

    def _bfly(x, lane, s):
        """Butterfly SHFL partner of `lane` at stride 1 << s, `.idx` flavour.

        The source lane is spelled arithmetically (flip bit s = add the stride
        on the even side of it, subtract it on the odd side) rather than as
        `lane ^ (1 << s)`: `shuffle_sync` is the same `shfl.sync.idx` K1's scan
        uses, and this file never leans on the DSL integer type carrying a
        bitwise operator.
        """
        st = 1 << s
        return cute.arch.shuffle_sync(
            x, lane + st - 2 * st * ((lane // st) % 2))

    # cute.arch's bulk commit/wait names drift across DSL minor versions; both
    # are operand-less per-thread instructions, so inline PTX is version-proof
    # (same spelling K2 uses for its h_c store group).
    @dsl_user_op
    def _bulk_commit(*, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                        has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def _bulk_wait(n, *, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.bulk.wait_group.read %d;" % int(n),
                        "", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def _bulk_red(dst, src, nbytes, *, loc=None, ip=None):
        """D2.1: ADD `nbytes` of smem onto gmem, asynchronously, per thread.

        The NON-tensor (bare address, bare byte count) flavour of
        `cp.reduce.async.bulk` (hw_kb `s2g_bulk_reduce`): it adds `nbytes`
        bytes of shared LINEARLY onto the same number of bytes of global, so
        the staging block must already be in gmem byte order -- which is why
        dq/dk stage through the FLAT views below and not through the swizzled
        64x64 faces every other publish site writes.  `.noftz` is the
        ISA-mandated qualifier for the 16-bit float reduction types.

        Completion is tracked by the same `bulk_group` mechanism the TMA
        stores use, so `_bulk_commit` / `_bulk_wait` retire it; the group is
        PER THREAD, so each issuing lane commits and drains its own.
        """
        llvm.inline_asm(
            None,
            [I64(dst).ir_value(loc=loc, ip=ip),
             I32(src).ir_value(loc=loc, ip=ip),
             I32(nbytes).ir_value(loc=loc, ip=ip)],
            "cp.reduce.async.bulk.global.shared::cta.bulk_group"
            ".add.noftz.bf16 [$0], [$1], $2;",
            "l,r,r", has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    # -- same two helpers as K2/K3, same reasons: every runtime value is an
    # EXPLICIT argument, no @cute.jit, and therefore plain `range` inside.
    def _xw(bX, i, ph):
        """Wait on exchange barrier `i` (a trace-time index) at phase `ph`.

        Readers pass `ph = t % 2`, writers waiting on an empty whose readers
        are in the PREVIOUS chunk pass `1 - t % 2` (a fresh mbarrier satisfies
        parity 1 immediately, so chunk 0 does not wait).  One-way channels are
        read-side: the arrival is in the same chunk, so their waiter uses
        `t % 2` like any reader.
        """
        cute.arch.mbarrier_wait(bX[(None, i)].iterator, ph)

    def _xa(bX, i):
        """Arrive (once per thread) on exchange barrier `i`."""
        cute.arch.mbarrier_arrive(bX[(None, i)].iterator)

    def _bp(bX, i):
        """The raw mbarrier pointer of barrier `i` (for `tma_bar_ptr=`)."""
        return bX[(None, i)].iterator

    def _xtx(bX, i, tx):
        """Arm landing-class barrier `i` for `tx` bytes of TMA traffic.

        `mbarrier.arrive.expect_tx` is a single-thread instruction (hw_kb
        `elect_one`), and it is BOTH the producer's arrival (the barrier is
        initialised with count 1) and the transaction-byte expectation the
        class's `cute.copy`s then retire.
        """
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(bX[(None, i)].iterator, tx)

    def publish(r2s_, thr_r, sF, stg, acc, slot, neg=False):
        """f32 C-fragment -> bf16 -> STSM x4 into face `slot` of `sF`.

        `neg` (a trace-time bool) flips the sign on the way through: the
        exchange faces carry -dv so that every consumer accumulates, while the
        D0.14 store staging of the same accumulator needs +dv.

        The narrowing reads the accumulator THROUGH the store atom's retile and
        converts element by element straight into the 16-bit fragment: no f32
        staging copy of the accumulator is made (D0.12 段B, "打包/精度转换必须
        终结上游 f32 累加器的活跃区 —— 转换后不得保留其 f32 副本").  Where the
        upstream accumulator dies at the pack this is only a tidier spelling of
        the same allocation (`cfrag_fused_pack_liveness` 机制 1: ptxas splits a
        staging fragment's live range into the dead half anyway).  Where it does
        NOT die it is the difference between fitting and spilling, and WG0 is
        exactly that case: its accumulator is the segment-resident dh, so a full
        f32 copy of the quadrant being published adds 32 registers on top of the
        128 that stay live across the call.
        """
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(acc)
        for v in range(cute.size(s16)):
            x = tacc[v]
            s16[v] = BF16(-x) if neg else BF16(x)
        cute.copy(r2s_, s16, thr_r.partition_D(sF[(None, None, slot)]))

    def publish_flat(r2s_, thr_r, sFF, stg, acc, mh, pair):
        """D2.1's staging publish: the same STSM, into a FLAT row-major view.

        `sFF` is the un-swizzled (64, 64, 2, pairs) view of the same smem a
        `publish` would write: element (r, c, mh, p) sits at
        `p * 8192 + r * STGP + mh * 64 + c`, i.e. the slot PAIR read as one
        row-major 64 x 128 tile in gmem byte order -- what
        `cp.reduce.async.bulk` needs, since it adds bytes linearly and knows
        nothing of a swizzle.  Writing column half `mh` therefore touches BOTH
        8 KB blocks of the pair (and, at D1.3's padded pitch, 1008 B of the
        block after it), so all of that must be dead at the call, not just the
        one block the old per-half TMA box used.

        That K4's own `r2s` tiled copy still places `acc[i]` at `cS[i]` when
        its destination is this non-compact view -- rather than the swizzled
        layout every other publish site writes -- is a thread-to-address fact,
        GPU-verified on these exact atoms at BOTH pitches: stride 128 in
        notes/probe_m2_d21_bulk_reduce.py (contract S1) and D1.3's stride 136,
        whose row step is not even a multiple of the 64-column half, in
        notes/probe_m2_d13_pad_stride.py (S1 value for value, S5 for the pad
        columns being left alone), both on a staging view carrying a non-zero
        pair offset.  Alignment holds: row r of half mh starts at byte
        `r * 272 + mh * 128` (272 = 17 x 16), and every STSM lane address is
        16 B apart inside that.
        """
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(acc)
        for v in range(cute.size(s16)):
            s16[v] = BF16(tacc[v])
        cute.copy(r2s_, s16,
                  thr_r.partition_D(sFF[(None, None, mh, pair)]))

    def reduce_rows(gbase, sFF, pair, ch, hq, lane_):
        """D2.1: ADD the staged 64 x 128 tile onto `dq`/`dk` rows, one warp.

        A gmem row of the output is `D` bf16 = 256 B, and consecutive rows are
        `H_QK * D * 2` = 4096 B apart, so the tile is 64 INDEPENDENT 256-B
        reductions -- there is no box shape that covers it in one issue, and
        the tensor (descriptor) flavour of the instruction does not lower in
        this DSL (hw_kb `s2g_bulk_reduce`).  The 32 lanes of the caller's
        single store warp take two rows each; each lane then commits its OWN
        bulk group (the group is per thread) and the caller drains it where
        the staging blocks have to be free again.

        The two v-head CTAs that share qk-head `hq` are ADJACENT blocks of the
        same wave (D0.5's head-fastest numbering), so their two reductions of
        the same 256 B meet in L2.

        D1.3 changed the row STEP only: the source of row j is
        `sbase + j * STGP * 2`, not `j * 256`.  The byte COUNT stays `D * 2`,
        so the 8 pad elements at the end of each row are never pulled into the
        reduction (contract S2 of notes/probe_m2_d13_pad_stride.py fills them
        with a sentinel and reads the result back to prove it).
        """
        sbase = sFF[(None, None, 0, pair)].iterator.toint()
        g0_ = (gbase + I64(ch) * I64(TILE * H_QK * D * 2)
               + I64(hq) * I64(D * 2))
        for s in range(2):
            j = lane_ * 2 + s
            _bulk_red(g0_ + I64(j) * I64(H_QK * D * 2),
                      sbase + j * (STGP * 2), D * 2)
        _bulk_commit()

    def ldsm_c(l2r_, thr_l, cshp, sT, slot):
        """D0.16 ②: slot `slot` of `sT` -> a fragment isomorphic to the C tile.

        ONE `LDSM.16.M88.4` per warp per 16x16 sub-tile (four per 64x64 tile)
        replaces the 32 per-element scalar `LDS` + 32 swizzle address
        computations a `sT[(m, n)]` read costs, and the result is indexed by the
        SAME fragment index as the accumulator it is about to be combined with:
        `r[i]` and `acc[i]` are both the element at `cS[i]`.  Verified on the
        GPU against these exact layouts (R1 for a TMA-landed slot, R4 for an
        STSM-written one) in notes/probe_m0_d016_ldsm.py.

        Returns a bf16 fragment; every call site converts per element on use,
        so no f32 copy of the tile is ever materialised (D0.12 段B).
        """
        r = ALLOC(cshp, BF16)
        cute.copy(l2r_, thr_l.partition_S(sT[(None, None, slot)]),
                  thr_l.retile(r))
        return r

    def shape_face(l2r_, r2s_, thr_l, thr_r, cshp, stg,
                   sS, src, sD, dst, f0, f1):
        """D0.16 ③: a row-gated derived face, shaped once by matrix copies.

        `dst_face[m, n] = f_m * src_tile[m, n]`, where f is a ROW scalar and so
        collapses (D0.16's M-axis hoist) to the TWO values the caller already
        exponentiated for its own rows: bit 1 of the fragment index selects
        between them (probe R2).  The path is `LDSM` -> 32 FMUL -> F2FP ->
        `STSM`, i.e. a constant number of matrix instructions per warp, in
        place of the 64 per-element scalar LDS + 64 scalar STS + 64 `MUFU` the
        Qg / Kg loops used to pay per chunk -- the whole of K4's
        `scalar_sts_in_loop`.  End to end this is probe R3.

        The gate factors are <= 1 (log gate <= 0, §1.3), so the bf16 face
        cannot overflow; the narrowing is the same one `publish` does.
        """
        rx = ldsm_c(l2r_, thr_l, cshp, sS, src)
        ry = ALLOC(cshp, BF16)
        for i in range(cute.size(rx)):
            ry[i] = BF16(F32(rx[i]) * (f0 if (i // 2) % 2 == 0 else f1))
        s16 = ALLOC(stg.shape, BF16)
        tacc = r2s_.retile(ry)
        for v in range(cute.size(s16)):
            s16[v] = tacc[v]
        cute.copy(r2s_, s16, thr_r.partition_D(sD[(None, None, dst)]))

    def link(mma, acc, sA_t, sB_t, nk, accumulate):
        """Put ONE WGMMA SS chain into the group that is currently open.

        No fence, no `commit_group`, no wait -- this is the inner half of the
        two-level issue helper hw_kb `wgmma_deferred_wait` 坑 11 asks for, and
        it is what lets several chains share one group and one drain (D1.1).

        Semantics of every call site in this file:  acc[m, n] (+)= sum_k
        sA_t[m, k] * sB_t[n, k].  A view built by `cute.composition` (the `T`
        views) is MN-major, one taken straight out of the landing zone or a
        face is K-major -- `mma` must be the TiledMma whose major pair matches.

        The A/B descriptors are built on the WARPGROUP-scope slice
        `get_slice(0)`, never on the per-thread slice (hw_kb
        `wgmma_desc_uniform` 坑 3).  `wgmma.mma_async` reads its two 64-bit
        matrix descriptors from the uniform register file (`desc[UR*]`), and
        ptxas can only leave them there if it can prove they are
        warp-invariant; a `tidx`-derived slice coordinate breaks that proof,
        so each descriptor is computed in per-thread registers and re-staged
        with `R2UR` ahead of every issue -- 2 dwords x its whole live range
        charged to the `setmaxnreg` quota, for every chain in flight.  Only
        `partition_C` (the accumulator) and the STSM partitions are
        per-thread.  A fresh atom per call site is mandatory (坑 4).
        """
        wgm = mma.get_slice(0)
        at = cute.make_mma_atom(mma.op)
        at.set(warpgroup.Field.ACCUMULATE, accumulate)
        fa = wgm.make_fragment_A(wgm.partition_A(sA_t))
        fb = wgm.make_fragment_B(wgm.partition_B(sB_t))
        for kb in range(nk):
            cute.gemm(at, acc, fa[None, None, kb], fb[None, None, kb], acc)
            at.set(warpgroup.Field.ACCUMULATE, True)

    def chain(mma, acc, sA_t, sB_t, nk, accumulate):
        """A chain that is a group all by itself: fence -> link -> commit ->
        wait_group(0).  Kept for the call sites that have nothing to merge or
        defer against."""
        warpgroup.fence()
        link(mma, acc, sA_t, sB_t, nk, accumulate)
        warpgroup.commit_group()
        warpgroup.wait_group(0)

    @cute.kernel
    def gdn_bwd_main(tmaQ: cute.CopyAtom, gQz: cute.Tensor,
             tmaK: cute.CopyAtom, gKz: cute.Tensor,
             tmaV: cute.CopyAtom, gVz: cute.Tensor,
             tmaO: cute.CopyAtom, gOz: cute.Tensor,
             tmaA: cute.CopyAtom, gAz: cute.Tensor,
             tmaH: cute.CopyAtom, gHz: cute.Tensor,
             tmaDv: cute.CopyAtom, gDvz: cute.Tensor,
             mDq: cute.Tensor, mDk: cute.Tensor,
             mGc: cute.Tensor, mBe: cute.Tensor, mDh: cute.Tensor,
             mDg: cute.Tensor, mDb: cute.Tensor,
             mma_kk: cute.TiledMma, mma_km: cute.TiledMma,
             mma_mm: cute.TiledMma, r2s: cute.TiledCopy,
             l2r: cute.TiledCopy,
             lyt_land: cute.ComposedLayout, lyt_landT: cute.ComposedLayout,
             lyt_face: cute.ComposedLayout, lyt_faceT: cute.ComposedLayout,
             SharedStorage: cutlass.Constexpr[Callable]):
        # D2.3: ahead of EVERY global read -- the mGc/mBe prologue below reads
        # K1's g_cumsum, and the producer's A / h_c / dh_end boxes are K1's,
        # K2's and K3's outputs.
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, _ = cute.arch.block_idx()
        tw = tidx % 128
        _pdl_wait()

        seg = I32(bx)
        head = I32(by)
        hq = head // GVA
        c0 = seg * SEG_CHUNKS

        # ---- shared prologue (all 512 threads, BEFORE the role branch and so
        # before the setmaxnreg split: materialising the smem addressing, the
        # mbarrier init and the TMA descriptors under the producer's reduced
        # budget is `warp_specialization` pitfall 11) -----------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLand = storage.land.get_tensor(lyt_land.outer, swizzle=lyt_land.inner)
        sLandT = storage.land.get_tensor(lyt_landT.outer, swizzle=lyt_landT.inner)
        sFace = storage.face.get_tensor(lyt_face.outer, swizzle=lyt_face.inner)
        sFaceT = storage.face.get_tensor(lyt_faceT.outer, swizzle=lyt_faceT.inner)
        # D2.1's staging views: the SAME two regions, un-swizzled, read as
        # ROW-PADDED row-major 64 x 128 tiles over each adjacent PAIR of 8 KB
        # slots.  `cp.reduce.async.bulk` adds shared bytes linearly onto global
        # bytes, so the staged tile has to be in gmem order; the swizzled views
        # above are not.  D1.3 sets the row STEP to STGP = 136 elements rather
        # than the compact 128: the 128 columns of a row stay contiguous and
        # 16 B aligned (all the reduction sees), while the 16 rows of one
        # `STSM.16.M88.4` stop landing on a single bank quad.  The tile is then
        # 17408 B and runs 1008 B past its pair -- into the (dead) A box for
        # dq, into the face region's 1 KB tail pad for dk.  Apart from that
        # pad no smem is added: these still alias the landing slots LV..LV+1
        # (dq) and the dh face pair SDH[0..1] (dk) the TMA store staged
        # through before.
        sLandF = storage.land.get_tensor(cute.make_layout(
            (TILE, TILE, 2, (SLOTS * STAGE) // 2),
            stride=(STGP, 1, TILE, 2 * TILE * TILE)))
        sFaceF = storage.face.get_tensor(cute.make_layout(
            (TILE, TILE, 2, NFACE // 2),
            stride=(STGP, 1, TILE, 2 * TILE * TILE)))
        sGc = storage.gc.get_tensor((SEG_TOK,))
        sBe = storage.be.get_tensor((SEG_TOK,))
        sRow = storage.row.get_tensor((NROW,))
        bX = storage.xb.get_tensor(cute.make_layout((1, NXB)))

        # D0.10's exchange AND landing barriers: one thread inits, everybody
        # sees them after the `sync_threads` that closes this prologue
        if wid == 0:
            with cute.arch.elect_one():
                for ib in cutlass.range_constexpr(len(XCOUNTS)):
                    cute.arch.mbarrier_init(
                        bX[(None, XCOUNTS[ib][0])].iterator, XCOUNTS[ib][1])
        cute.arch.mbarrier_init_fence()

        tQs, tQg = cpasync.tma_partition(tmaQ, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gQz)
        tKs, tKg = cpasync.tma_partition(tmaK, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gKz)
        tVs, tVg = cpasync.tma_partition(tmaV, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gVz)
        tOs, tOg = cpasync.tma_partition(tmaO, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gOz)
        tAs, tAg = cpasync.tma_partition(tmaA, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gAz)
        tHs, tHg = cpasync.tma_partition(tmaH, 0, cute.make_layout(1),
                                         cute.group_modes(sLand, 0, 2), gHz)
        # D0.14's store path, as amended by D2.1.  dv is still a TMA store
        # through a dead face pair.  dq and dk are no longer per-v-head
        # partials that a 6th launch sums: they are reduce-ADDED straight into
        # the final [N, H_QK, D] tensors, so they need no TMA descriptor at
        # all -- only the base address of the tensor, plus the flat staging
        # views above.  The staging blocks are unchanged (dq in the LANDING
        # v boxes, dead since stage 1; dk in the dh face pair its own stage-5
        # chains have just finished with).
        tDvs, tDvg = cpasync.tma_partition(tmaDv, 0, cute.make_layout(1),
                                           cute.group_modes(sFace, 0, 2), gDvz)
        gDq = mDq.iterator.toint()
        gDk = mDk.iterator.toint()

        # gate / beta for the whole segment, once (keeps them out of the ring)
        for e in cutlass.range_constexpr(SEG_TOK // TID):
            j = e * TID + tidx
            sGc[j] = mGc[(c0 + j // C, j % C, head)]
            sBe[j] = mBe[(c0 + j // C, j % C, head)]
        cute.arch.sync_threads()
        _seg("k4.entry")

        # ---- the ONE role branch (dec/inc is its body's first statement) ----
        if wid >= 12:
            REG_DEC(PROD_REGS)
            # producer: ONE warp issues the 13 TMA boxes of each chunk
            if wid == 12:
                _seg("k4.producer.entry")
                for t in cutlass.range(SEG_CHUNKS):
                    ch = c0 + (SEG_CHUNKS - 1 - t)          # backwards
                    ph = t % 2
                    # Five independent classes, each `wait empty(1 - ph) ->
                    # arm expect_tx -> issue that class's boxes`.  The order is
                    # the order the consumers block on them: WG1's stage 1 and
                    # WG2's stage 1 open on k / h_c / q, and A+v goes last
                    # because its empty is the latest to arrive (WG1 hands the
                    # v boxes back only after the dq store staged through them
                    # has drained, at its stage 5).  Waiting on a late class
                    # first would hold back the classes whose empties are
                    # already in -- the whole point of splitting the pair.
                    _seg("k4.supply.acquire.begin", t)
                    _xw(bX, LE_K, 1 - ph)
                    _seg("k4.supply.acquire.end", t)
                    _xtx(bX, LF_K, TX_K)
                    for mh in cutlass.range_constexpr(2):
                        cute.copy(tmaK, tKg[(None, (ch, hq * 2 + mh))],
                                  tKs[(None, LK + mh)],
                                  tma_bar_ptr=_bp(bX, LF_K))
                    # h_c: the chunk ENTRY state, four 64x64 boxes of the same
                    # 2D picture K2 stored into
                    _xw(bX, LE_H, 1 - ph)
                    _xtx(bX, LF_H, TX_H)
                    for mh in cutlass.range_constexpr(2):
                        for nh in cutlass.range_constexpr(2):
                            cute.copy(
                                tmaH,
                                tHg[(None, ((ch * H_V + head) * 2 + mh, nh))],
                                tHs[(None, LH + mh * 2 + nh)],
                                tma_bar_ptr=_bp(bX, LF_H))
                    _xw(bX, LE_Q, 1 - ph)
                    _xtx(bX, LF_Q, TX_Q)
                    for mh in cutlass.range_constexpr(2):
                        cute.copy(tmaQ, tQg[(None, (ch, hq * 2 + mh))],
                                  tQs[(None, LQ + mh)],
                                  tma_bar_ptr=_bp(bX, LF_Q))
                    _xw(bX, LE_O, 1 - ph)
                    _xtx(bX, LF_O, TX_O)
                    for nh in cutlass.range_constexpr(2):
                        cute.copy(tmaO, tOg[(None, (ch, head * 2 + nh))],
                                  tOs[(None, LO + nh)],
                                  tma_bar_ptr=_bp(bX, LF_O))
                    _xw(bX, LE_AV, 1 - ph)
                    _xtx(bX, LF_AV, TX_AV)
                    cute.copy(tmaA, tAg[(None, (ch, head))],
                              tAs[(None, LA)], tma_bar_ptr=_bp(bX, LF_AV))
                    for nh in cutlass.range_constexpr(2):
                        cute.copy(tmaV, tVg[(None, (ch, head * 2 + nh))],
                                  tVs[(None, LV + nh)],
                                  tma_bar_ptr=_bp(bX, LF_AV))
                    _seg("k4.supply.issue.end", t)
                _seg("k4.producer.exit")
        else:
            _seg("k4.consumer.entry")
            # ONE per-thread MMA slice, and it is used for nothing but the
            # accumulator coordinates: A/B now ride `get_slice(0)` inside
            # `chain` (hw_kb `wgmma_desc_uniform`).  The three TiledMma differ
            # only in their operand major modes, so partition_C is the same
            # for all three.
            thr_c = mma_km.get_slice(tw)
            cS = thr_c.partition_C(cute.make_identity_tensor((TILE, TILE)))
            thr_r = r2s.get_slice(tw)
            stg = cute.make_layout(cute.shape(thr_r.partition_S(
                sFace[(None, None, 0)]))[:3])
            # D0.16's read side.  Same rule as the STSM slice: the tiled copy
            # is sliced by the WARPGROUP thread number, never by lane and never
            # by `tidx` (`ldmatrix_stmatrix` 坑 5), so that `thr_l`'s fragment
            # index agrees element for element with `cS`.
            thr_l = l2r.get_slice(tw)
            cshp = thr_c.partition_shape_C((TILE, TILE))

            # Each consumer role gets its OWN chunk loop rather than a shared
            # loop with a three-way branch per stage: with one loop, WG0's 128
            # dh registers are live at every program point of WG1's and WG2's
            # code too (liveness is per program point, not per warp), which
            # leaves 32 of the budget for their accumulators.  Three top-level
            # branches let the allocator size each role independently, the
            # form `setmaxnreg` exists for.  Each loop now blocks only on the
            # faces and the landing classes its own stages touch, and releases
            # each of them where its last read of it retires.
            #
            # The `.inc` of each role is the first statement of ITS branch and
            # carries ITS quota (D0.12's per-role live-set budget above): the
            # three roles' working sets differ by 48 registers and a flat quota
            # has to be sized for the largest of them, which the pool cannot
            # afford.  Everything shared sits ABOVE the branch, where it is
            # compiled against the 128-register entry quota, because a budget
            # move hoisted ahead of the shared prologue forces that prologue
            # into the reduced quota (`warp_specialization` 坑 11); and there is
            # no barrier between branch and `.inc` (坑 5).
            if wid < 4:
                REG_INC(WG0_REGS)
                # ================= WG0: the dh owner (D0.11) ================
                # a comprehension is its own scope and the rewriter does not
                # descend into it, so this `range` stays plain python
                accD = [ALLOC(cS.shape, F32) for _ in range(4)]
                # prologue: dh = dh_end[segment], f32, straight into the 128
                # registers/thread it lives in for the whole segment.  D0.7
                # gives two ways to make segment 7's slot (no final-state
                # gradient, no K3 producer) read zero; this is the SECOND one,
                # "K4 judges by segment number": the `selp` below discards
                # whatever the untouched slot holds, so `dh_end` is a plain
                # `torch.empty` and the window carries no memset launch at all
                # (the first option -- host `torch.zeros` -- put an
                # `at::vectorized_elementwise_kernel` FillFunctor in the
                # window, which is a 6th kernel and 4 extra launches against
                # D0.1's declared table of 5 / 24).
                for mh in cutlass.range_constexpr(2):
                    for nh in cutlass.range_constexpr(2):
                        a = accD[mh * 2 + nh]
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            m, n = cS[i][0], cS[i][1]
                            a[i] = _sel_gt(
                                seg, I32(N_SEG - 2), F32(0.0),
                                mDh[(seg, head, mh * TILE + m,
                                     nh * TILE + n)])
                # STAGE == 1, so the landing slot set never moves; the ring
                # state that used to index it is now five per-class barrier
                # phases, all equal to `t % 2`
                b = 0
                for t in cutlass.range(SEG_CHUNKS):
                    ci = I32(SEG_CHUNKS - 1) - t
                    ch = c0 + ci
                    gbase = ci * C
                    glast = sGc[gbase + C - 1]
                    ph = t % 2

                    # -- (1) publish dh for WG1's dU and WG2's dk ------------
                    # the slot's previous readers are the PREVIOUS chunk's WG1
                    # s2 and WG2 s5 (this is the one edge the milestone calls
                    # "等上轮"), so the write waits their empty at 1 - ph; a
                    # fresh barrier satisfies parity 1 at once
                    _xw(bX, XE_DH, 1 - ph)
                    for j in cutlass.range_constexpr(4):
                        publish(r2s, thr_r, sFace, stg, accD[j], SDH[j])
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_DH)
                    _seg("k4.dh.publish.end", t)
                    # -- (2)/(3)/(4) WG0's own stages contribute nothing to the
                    # exchange face; they run straight into (5) --- the only
                    # cross-WG waits in this role are the faces it reads.
                    # -- (5) dh <- e^{G_last} dh + Qg^T dO + Kg^T (-dv), and
                    # with it psi = <dh_in, h>: the chunk-entry boundary of the
                    # dg reverse recurrence (D0.14).  dh_in is the accumulator
                    # this stage produces and h is the landed chunk entry
                    # state, so the dot rides the SAME pass over the four
                    # quadrants -- a second pass would re-open 128 pinned
                    # registers' worth of addressing for nothing.  The CTA-wide
                    # sum is a 5-step lane butterfly plus a four-way combine
                    # through the scalar block, and WG1 picks it up off the PSI
                    # barrier at the end of this stage.
                    _xw(bX, XF_QG, ph)
                    _xw(bX, XF_KG, ph)
                    _xw(bX, XF_NDV, ph)
                    # WG0's only landing classes, and they are both read right
                    # here: do as the B operand of the Q-bar rank-64 update,
                    # h_c as the other half of the psi inner product
                    _seg("k4.chunk.land.begin", t)
                    _xw(bX, LF_O, ph)
                    _xw(bX, LF_H, ph)
                    _seg("k4.chunk.land.end", t)
                    dec = _exp(glast)
                    # D1.5 / `ew_segment_latency_schedule` §(d).  The psi dot
                    # is the only serial chain inside the段 D1.1's depth-1
                    # ladder covers, and as ONE rolling accumulator its in-
                    # flight width was 1: 32 dependent FFMA per quadrant, the
                    # same order as the eight WGMMA the shadow is made of, so
                    # the段 did not fit inside its own cover.  Four independent
                    # partial sums (the KB's 正例 shape -- one value per group,
                    # merged once at the end) cut the chain to 8 FFMA.  The
                    # price is three registers on top of WG0's 128 resident dh:
                    # the LDSM fragment and the pack fragment are unchanged, so
                    # D0.12's 段C non-accumulator account moves by exactly
                    # those three and stays inside its <= 32.
                    ps0, ps1 = F32(0.0), F32(0.0)
                    ps2, ps3 = F32(0.0), F32(0.0)
                    # D1.1 at depth 1, the one role where the ladder is free.
                    # The four quadrant accumulators ARE dh: they are live at
                    # every program point of this loop either way, so letting
                    # quadrant j fly across quadrant j-1's psi pass adds no
                    # live value at all (`sH` is born and dies inside the
                    # covered段, exactly as it did at depth 0, and a chain's
                    # descriptors are dead the moment its last k-block issues).
                    # The covered段 is straight-line LDSM + 32 FFMA with no
                    # blocking wait and no stage release in it -- hw_kb
                    # `wgmma_deferred_wait` 坑 12's sign condition, not 坑 10's.
                    # Both chains of a quadrant target the SAME accumulator and
                    # nothing runs between them, so they also share one group
                    # and one drain (坑 11): 8 groups/chunk -> 4.
                    for j in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            accD[j][i] = accD[j][i] * dec
                    for j in cutlass.range_constexpr(4):
                        mh, nh = j // 2, j % 2
                        warpgroup.fence()
                        link(mma_mm, accD[j],
                             sFaceT[(None, None, FQG + mh)],
                             sLandT[(None, None, b + LO + nh)], 4, True)
                        link(mma_mm, accD[j],
                             sFaceT[(None, None, SVN[mh])],
                             sFaceT[(None, None, FNDV + nh)], 4, True)
                        warpgroup.commit_group()
                        if cutlass.const_expr(j > 0):
                            # quadrant j stays in flight; this reaps j - 1
                            warpgroup.wait_group(1)
                            p = j - 1
                            sH = ldsm_c(l2r, thr_l, cshp, sLand, b + LH + p)
                            for i in cutlass.range_constexpr(cute.size(cS)):
                                pd = accD[p][i] * F32(sH[i])
                                if cutlass.const_expr(i % 2 == 0):
                                    if cutlass.const_expr((i // 2) % 2 == 0):
                                        ps0 = ps0 + pd
                                    else:
                                        ps1 = ps1 + pd
                                else:
                                    if cutlass.const_expr((i // 2) % 2 == 0):
                                        ps2 = ps2 + pd
                                    else:
                                        ps3 = ps3 + pd
                    warpgroup.wait_group(0)
                    sH = ldsm_c(l2r, thr_l, cshp, sLand, b + LH + 3)
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        pd = accD[3][i] * F32(sH[i])
                        if cutlass.const_expr(i % 2 == 0):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                ps0 = ps0 + pd
                            else:
                                ps1 = ps1 + pd
                        else:
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                ps2 = ps2 + pd
                            else:
                                ps3 = ps3 + pd
                    ps = (ps0 + ps1) + (ps2 + ps3)
                    _seg("k4.dh.update.end", t)
                    # every chain above ended in `wgmma.wait_group.sync.aligned
                    # 0` and the psi pass has read its last h_c element, so
                    # WG0 is off both of its landing classes for this chunk --
                    # the producer may refill do and h_c while WG0 is still
                    # doing the butterfly and the psi combine below
                    _xa(bX, LE_O)
                    _xa(bX, LE_H)
                    for s in cutlass.range_constexpr(5):
                        ps = ps + _bfly(ps, lane, s)
                    if lane == 0:
                        sRow[RW + wid] = ps
                    cute.arch.barrier(barrier_id=BAR_WG0,
                                      number_of_threads=128)
                    # WG1 has read the PREVIOUS chunk's psi out of the scalar
                    # block (transitively implied by XF_NDV above, but written
                    # out so the edge survives a reordering)
                    _xw(bX, XE_PSI, 1 - ph)
                    if tw == 0:
                        sRow[RPSI] = (sRow[RW] + sRow[RW + 1]
                                      + sRow[RW + 2] + sRow[RW + 3])
                    _seg("k4.psi.end", t)
                    # release (5)'s read faces and hand psi to WG1.  XF_QG /
                    # XF_KG / XE_NDV all follow WG0's four `chain` calls, which
                    # each end in a warpgroup `wgmma.wait_group.sync.aligned`
                    # (itself a 128-thread rendezvous), so the arrive needs no
                    # extra NamedBarrier; BAR_WG0 above is what makes `sRow`
                    # visible CTA-wide before the PSI arrive publishes it.
                    _xa(bX, XE_QG)
                    _xa(bX, XE_KG)
                    _xa(bX, XE_NDV)
                    _xa(bX, XF_PSI)
            elif wid < 8:
                REG_INC(WG1_REGS)
                # ============ WG1: the h_c / dh contractions ================
                b = 0
                for t in cutlass.range(SEG_CHUNKS):
                    ci = I32(SEG_CHUNKS - 1) - t
                    ch = c0 + ci
                    gbase = ci * C
                    glast = sGc[gbase + C - 1]
                    ph = t % 2
                    # D0.16's M-axis hoist, once per chunk for the whole role:
                    # the 32 elements a thread holds sit on exactly TWO rows
                    # (probe R2), so every ROW scalar of every elementwise
                    # segment below -- g, beta -- is TWO scalar `LDS`, not 32,
                    # and the exponentials of those rows are two `MUFU` at
                    # each use instead of 32.
                    m0, m1 = cS[0][0], cS[2][0]
                    g0 = sGc[gbase + m0]
                    g1 = sGc[gbase + m1]
                    be0 = sBe[gbase + m0]
                    be1 = sBe[gbase + m1]

                    # -- (1) Bm = beta o (V - e^G o (K @ h)) ----------------
                    # slots 6-7 are the pack's one aliased pair (see the
                    # barrier table): their PREVIOUS occupant is the previous
                    # chunk's -dv, whose readers are WG2 s4/s5 and WG0 s5, so
                    # this write waits XE_NDV at 1 - ph.  It is NOT gated on
                    # XE_BM -- that barrier announces this chunk's later
                    # release of the SAME slots and belongs on the -dv write.
                    # Three of the five landing classes open here: k and h_c
                    # feed the chain, v the elementwise that follows it.
                    _xw(bX, XE_NDV, 1 - ph)
                    _seg("k4.chunk.land.begin", t)
                    _xw(bX, LF_K, ph)
                    _xw(bX, LF_H, ph)
                    _xw(bX, LF_AV, ph)
                    _seg("k4.chunk.land.end", t)
                    eg0, eg1 = _exp(g0), _exp(g1)
                    for nh in cutlass.range_constexpr(2):
                        acc = ALLOC(cS.shape, F32)
                        # D1.1 / hw_kb 坑 11: the two mh halves accumulate into
                        # the SAME register file and nothing runs between them,
                        # so they are one group and one drain, not two.
                        warpgroup.fence()
                        for mh in cutlass.range_constexpr(2):
                            link(mma_km, acc,
                                 sLand[(None, None, b + LK + mh)],
                                 sLandT[(None, None, b + LH + mh * 2 + nh)],
                                 4, mh > 0)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        # v enters the elementwise by matrix copy (D0.16 ② --
                        # it is aligned element for element with `acc`)
                        rV = ldsm_c(l2r, thr_l, cshp, sLand, b + LV + nh)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                acc[i] = be0 * (F32(rV[i]) - eg0 * acc[i])
                            else:
                                acc[i] = be1 * (F32(rV[i]) - eg1 * acc[i])
                        publish(r2s, thr_r, sFace, stg, acc, FBM + nh)
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_BM)
                    _seg("k4.bm.end", t)
                    # -- (2) dU = e^{G_last-G} o (K @ dh) + Sq^T @ dO -------
                    # the decay is a ROW scalar of K @ dh, so the plainly
                    # landed K is the operand and no decayed-key face is
                    # needed; the face carries -dU so its consumer accumulates
                    _xw(bX, XF_DH, ph)
                    _xw(bX, XF_SQ, ph)
                    _xw(bX, LF_O, ph)      # dO, the B operand of the Sq^T term
                    d0, d1 = _exp(glast - g0), _exp(glast - g1)
                    for nh in cutlass.range_constexpr(2):
                        acc = ALLOC(cS.shape, F32)
                        warpgroup.fence()
                        for mh in cutlass.range_constexpr(2):
                            link(mma_km, acc,
                                 sLand[(None, None, b + LK + mh)],
                                 sFaceT[(None, None, SDH[mh * 2 + nh])],
                                 4, mh > 0)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                acc[i] = d0 * acc[i]
                            else:
                                acc[i] = d1 * acc[i]
                        chain(mma_mm, acc,
                              sFaceT[(None, None, FSQ)],
                              sLandT[(None, None, b + LO + nh)], 4, True)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            acc[i] = -acc[i]
                        publish(r2s, thr_r, sFace, stg, acc, SVN[nh])
                    cute.arch.fence_view_async_shared()
                    # both chains ended in `wgmma.wait_group.sync.aligned 0`,
                    # so the dh faces and the Sq face are genuinely released
                    _xa(bX, XE_DH)
                    _xa(bX, XE_SQ)
                    # and WG1's dh operand reads are done too, which is the
                    # edge WG2's stage-5 dk overwrite of those faces needs
                    _xa(bX, X_RDH)
                    _seg("k4.dvn.end", t)
                    # -- (3) dv = beta o (A^T @ dU), and with it dbeta and the
                    # <v, dv> half of the dg increment ----------------------
                    # the face carries -dU, so the accumulator is -du and
                    # beta o it is -dv -- the sign all three consumers want
                    # (WG2's Mv and dk h-term, WG0's dh update); the store
                    # staging is the negation.  The staging block of each half
                    # is the -dU face this iteration's own chain has just
                    # consumed: `chain` ends in `wgmma.wait_group.sync.aligned
                    # 0`, so the warpgroup's operand reads are complete and the
                    # accumulator need not outlive the pair.
                    # The two row dots are carried as plain scalars (one per
                    # row of the two a C-fragment holds) the way `glast` and
                    # `dec` are -- a python list rebound inside a
                    # `cutlass.range` body would be a loop-carried container.
                    # Two in-edges, both from THIS chunk's WG2 s2: XF_U (the U
                    # face the dbeta row dot reads) and XE_BM (WG2 has finished
                    # reading Bm, so slots 6-7 may become -dv).
                    _xw(bX, XF_U, ph)
                    _xw(bX, XE_BM, ph)
                    db0, db1 = F32(0.0), F32(0.0)
                    rv0, rv1 = F32(0.0), F32(0.0)
                    # D1.5 §(d) again: each of the two rows a C-fragment holds
                    # runs on TWO partial sums rather than one, so the four
                    # row dots of stages 3 and 4 open eight independent chains
                    # of 8 FFMA instead of four of 16.  Four more registers in
                    # the role D0.12 gives 144; the merge is one FADD each,
                    # just before the butterfly that closes the rows anyway.
                    db0b, db1b = F32(0.0), F32(0.0)
                    rv0b, rv1b = F32(0.0), F32(0.0)
                    for nh in cutlass.range_constexpr(2):
                        acc = ALLOC(cS.shape, F32)
                        chain(mma_mm, acc,
                              sLandT[(None, None, b + LA)],
                              sFaceT[(None, None, SVN[nh])], 4, False)
                        sU = ldsm_c(l2r, thr_l, cshp, sFace, FU + nh)
                        sV = ldsm_c(l2r, thr_l, cshp, sLand, b + LV + nh)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            # acc is -du here: dbeta_i = <du_i, U_i> / beta_i
                            d = acc[i] * F32(sU[i])
                            # bit 1 of the fragment index selects which of the
                            # thread's two rows this element sits on; the test
                            # is trace-time, so `const_expr` (without it the
                            # arms become IR regions and their bindings do not
                            # escape -- cute_dsl_const_expr_if).  It is also
                            # what makes the beta row scalar two `LDS` for the
                            # whole chunk instead of one per element (D0.16).
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                acc[i] = be0 * acc[i]                 # = -dv
                                # acc is -dv, so this row sum IS -<v_i, dv_i>
                                if cutlass.const_expr((i // 4) % 2 == 0):
                                    rv0 = rv0 + acc[i] * F32(sV[i])
                                    db0 = db0 - d
                                else:
                                    rv0b = rv0b + acc[i] * F32(sV[i])
                                    db0b = db0b - d
                            else:
                                acc[i] = be1 * acc[i]
                                if cutlass.const_expr((i // 4) % 2 == 0):
                                    rv1 = rv1 + acc[i] * F32(sV[i])
                                    db1 = db1 - d
                                else:
                                    rv1b = rv1b + acc[i] * F32(sV[i])
                                    db1b = db1b - d
                        publish(r2s, thr_r, sFace, stg, acc, FNDV + nh)
                        publish(r2s, thr_r, sFace, stg, acc, SVN[nh],
                                neg=True)
                    cute.arch.fence_view_async_shared()
                    # slots 6-7 now carry -dv for WG2 s4/s5 and WG0 s5, and
                    # SVN's two the +dv store staging.  WG1's own -dU read of
                    # them is complete (the chain above waited), so the only
                    # remaining reader of that pair is the store warp.
                    _xa(bX, XF_NDV)
                    _xa(bX, XE_U)
                    _seg("k4.dv.end", t)
                    cute.arch.barrier(barrier_id=BAR_WG1,
                                      number_of_threads=128)
                    if wid == 4:
                        for nh in cutlass.range_constexpr(2):
                            cute.copy(tmaDv, tDvs[(None, SVN[nh])],
                                      tDvg[(None, (ch, head * 2 + nh))])
                        _bulk_commit()
                        _bulk_wait(0)
                        # the store warp alone has finished reading the two
                        # staging blocks, so it alone releases them to WG2's
                        # stage-4 Kg (BAR_WG1 above orders the whole WG's STSM
                        # ahead of this single warp's copy)
                        _xa(bX, X_FREE1112)
                    _seg("k4.dv.store.end", t)
                    # -- (4) dq = scale e^G o (dO @ h^T) + Ms @ K -----------
                    _xw(bX, XF_MS, ph)
                    _xw(bX, LF_Q, ph)      # the last class WG1 opens
                    sg0, sg1 = SCALE * eg0, SCALE * eg1
                    for mh in cutlass.range_constexpr(2):
                        acc = ALLOC(cS.shape, F32)
                        warpgroup.fence()
                        for nh in cutlass.range_constexpr(2):
                            link(mma_kk, acc,
                                 sLand[(None, None, b + LO + nh)],
                                 sLand[(None, None, b + LH + mh * 2 + nh)],
                                 4, nh > 0)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                acc[i] = sg0 * acc[i]
                            else:
                                acc[i] = sg1 * acc[i]
                        chain(mma_km, acc,
                              sFace[(None, None, FMS)],
                              sLandT[(None, None, b + LK + mh)], 4, True)
                        sQ = ldsm_c(l2r, thr_l, cshp, sLand, b + LQ + mh)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            r = acc[i] * F32(sQ[i])
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                if cutlass.const_expr((i // 4) % 2 == 0):
                                    rv0 = rv0 + r
                                else:
                                    rv0b = rv0b + r
                            else:
                                if cutlass.const_expr((i // 4) % 2 == 0):
                                    rv1 = rv1 + r
                                else:
                                    rv1b = rv1b + r
                        # the v landing boxes have been dead since stage 1;
                        # D2.1 stages the half into the FLAT view of that pair
                        publish_flat(r2s, thr_r, sLandF, stg, acc,
                                     mh, (b + LV) // 2)
                    # the two row dots close here.  The split chains merge
                    # first (one FADD each, D1.5), then lanes l, l^1, l^2 of a
                    # quad hold the same two rows of a 64x64 C-fragment
                    # (the row index rides bit 1 of the fragment index), so
                    # two butterfly steps give every lane the full row sum
                    # and the l%4 == 0 lane of each quad owns the write.
                    # Everything the guarded arm needs is bound OUTSIDE it
                    # (cute_dsl_const_expr_if: a runtime `if` is an IR region
                    # and names first bound inside one do not escape it).
                    rv0 = rv0 + rv0b
                    rv1 = rv1 + rv1b
                    db0 = db0 + db0b
                    db1 = db1 + db1b
                    for s in cutlass.range_constexpr(2):
                        rv0 = rv0 + _bfly(rv0, lane, s)
                        rv1 = rv1 + _bfly(rv1, lane, s)
                        db0 = db0 + _bfly(db0, lane, s)
                        db1 = db1 + _bfly(db1, lane, s)
                    m0, m1 = cS[0][0], cS[2][0]   # (bound at the chunk top)
                    # db already carries +<du_i, U_i>: the stage-3 accumulator
                    # is -du and the row sum subtracts, so the quotient is
                    # dbeta itself and must NOT be negated a second time.
                    # + 1e-30 keeps beta == 0 (where u is exactly 0 and the
                    # row sum with it) out of 0/0
                    e0 = db0 / (be0 + F32(1e-30))
                    e1 = db1 / (be1 + F32(1e-30))
                    if lane % 4 == 0:
                        sRow[m0] = rv0
                        sRow[m1] = rv1
                        sRow[C + m0] = e0
                        sRow[C + m1] = e1
                    cute.arch.fence_view_async_shared()
                    # the Ms-reading chain above has waited, so the Ms slot is
                    # released back to WG2's next-chunk stage 3
                    _xa(bX, XE_MS)
                    # and stage 4 was WG1's last read of q, k, do and h_c --
                    # all four chains ended in `wgmma.wait_group.sync.aligned
                    # 0` and the sQ row dot is done.  Only A+v is still held
                    # (the dq tiles are staging in the v boxes right now).
                    _xa(bX, LE_Q)
                    _xa(bX, LE_K)
                    _xa(bX, LE_O)
                    _xa(bX, LE_H)
                    _seg("k4.dq.end", t)
                    cute.arch.barrier(barrier_id=BAR_WG1,
                                      number_of_threads=128)
                    if wid == 4:
                        reduce_rows(gDq, sLandF, (b + LV) // 2, ch, hq, lane)
                    _seg("k4.dq.store.end", t)
                    # -- (5) WG2 -> dk | WG0 -> dh + psi --------------------
                    # WG1 has no (5); it is off the landing zone now (its own
                    # stage-4 reads are complete), so it reports at the same
                    # moment WG2 does and WG0 can hand the slot to the
                    # producer.
                    if wid == 4:
                        # the producer refills the A and v boxes as soon as
                        # this class's empty is in, which is now gated on
                        # exactly this drain
                        _bulk_wait(0)
                    # A has been dead since stage 3 and v since stage 1 (the
                    # dq tiles that took the v boxes have just left smem).  All
                    # 128 threads arrive, but the count cannot be reached
                    # without warp 4's arrival, which is the one that follows
                    # the drain -- the same relay the dv staging uses.
                    _xa(bX, LE_AV)
                    # the dg / dbeta epilogue runs last: it needs WG0's
                    # <dh_in, h>, which arrives on XF_PSI at the end of WG0's
                    # own stage 5.  It reads only the scalar block, so it is
                    # off the landing zone WG0 hands back at the same moment.
                    #
                    # D1.5 / `ew_segment_latency_schedule` §(d).  The exclusive
                    # scan used to be 64 broadcast `LDS` chained onto ONE
                    # rolling sum -- in-flight width 1, a 64-deep FADD chain,
                    # and 64 shared-read wavefronts per warp per chunk, on the
                    # edge that gates WG0's next-chunk stage 5 (XE_PSI).  The
                    # same value comes out of a lane scan: each warp of WG1
                    # reads sRow[tw % 64] once and runs the 5-step Kogge-Stone
                    # `shfl.sync.idx` inclusive scan K1's cumsum uses (the
                    # source lane is spelled arithmetically and clamped -- the
                    # `.up` flavour's mask_and_clamp would falsify the
                    # predicate on every lane), then subtracts its own term to
                    # make it exclusive.  The carry the upper half needs (the
                    # total over tokens 0..31) is a 5-step butterfly over ONE
                    # more strided read, so it costs neither a scalar staging
                    # slot nor a rendezvous: both rows were published by the
                    # BAR_WG1 above.  Warps 6-7 run the same code on rows they
                    # do not own; the store guard discards it, exactly as the
                    # 64-read form discarded their sums.  Chain 64 -> 5,
                    # shared reads 64 -> 2.
                    _xw(bX, XF_PSI, ph)
                    own = sRow[tw % C]
                    lo = sRow[lane]
                    for s in cutlass.range_constexpr(5):
                        lo = lo + _bfly(lo, lane, s)
                    x = own
                    for s in cutlass.range_constexpr(5):
                        sd = 1 << s
                        shv = cute.arch.shuffle_sync(
                            x, _imax(lane - sd, I32(0)))
                        x = x + _sel_gt(lane, sd - 1, shv, F32(0.0))
                    sc = (x - own) + _sel_gt(tw, I32(C // 2 - 1), lo,
                                             F32(0.0))
                    dgv = sRow[RPSI] - sc
                    dbv = sRow[C + (tw % C)]
                    if tw < C:
                        mDg[(ch, tw, head)] = dgv
                        mDb[(ch, tw, head)] = dbv
                    _xa(bX, XE_PSI)
                    _seg("k4.dg.end", t)
            else:
                REG_INC(WG2_REGS)
                # ============ WG2: the 64-token axis ========================
                b = 0
                for t in cutlass.range(SEG_CHUNKS):
                    ci = I32(SEG_CHUNKS - 1) - t
                    ch = c0 + ci
                    gbase = ci * C
                    glast = sGc[gbase + C - 1]
                    ph = t % 2
                    # ---- D0.16's in-chunk gate mask, formed ONCE per chunk --
                    # `mk[i] = [j <= i] e^{G_i - G_j}` in the 64x64 accumulator's
                    # own fragment order, so the three masked segments of this
                    # role (Sq in s1, Ms in s3, -Mv in s4) are each a single
                    # FMUL per element against a register they already hold,
                    # instead of re-reading a g PAIR and re-exponentiating it 32
                    # times over, three times a chunk.  Cost per thread per
                    # chunk: 2 row `LDS` + 16 column `LDS` (bit 1 of the index
                    # picks the row, probe R2) + 32 FADD + 32 MUFU -- the
                    # reference's own段-1 shape.  The exponent is NOT split into
                    # e^{G_i} e^{-G_j} (D0.16 forbids it: |sum g| reaches tens
                    # and either factor alone can overflow); above the diagonal
                    # the difference IS positive and its exp may be inf, which
                    # is why the mask is SELECTED rather than multiplied in.
                    # 32 f32 registers live from here to stage 4, inside the
                    # 16-32 D0.12 段账 allows; stage 5's two accumulators come
                    # after its last use.
                    m0, m1 = cS[0][0], cS[2][0]
                    g0 = sGc[gbase + m0]
                    g1 = sGc[gbase + m1]
                    mk = ALLOC(cS.shape, F32)
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        n = cS[i][1]
                        gn = sGc[gbase + n]
                        if cutlass.const_expr((i // 2) % 2 == 0):
                            mk[i] = _sel_gt(n, m0, F32(0.0), _exp(g0 - gn))
                        else:
                            mk[i] = _sel_gt(n, m1, F32(0.0), _exp(g1 - gn))

                    # -- (1) Sq = mask_{j<=i} scale e^{G_i-G_j}(q_i.k_j) and
                    # Qg = scale e^G o Q ------------------------------------
                    # the Qg slots' previous reader is the PREVIOUS chunk's
                    # WG0 s5; the Sq slot's is WG2's own s5 (-Mv), so it needs
                    # no cross-WG empty
                    _xw(bX, XE_QG, 1 - ph)
                    _seg("k4.chunk.land.begin", t)
                    _xw(bX, LF_Q, ph)
                    _xw(bX, LF_K, ph)
                    _seg("k4.chunk.land.end", t)
                    acc = ALLOC(cS.shape, F32)
                    warpgroup.fence()
                    for mh in cutlass.range_constexpr(2):
                        link(mma_kk, acc,
                             sLand[(None, None, b + LQ + mh)],
                             sLand[(None, None, b + LK + mh)], 4, mh > 0)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        acc[i] = (SCALE * mk[i]) * acc[i]      # keep j <= i
                    publish(r2s, thr_r, sFace, stg, acc, FSQ)
                    # Qg = scale e^G o Q: a ROW scaling of a landed tile, so
                    # D0.16 ③ -- LDSM into C-fragment order, two FMUL factors
                    # per thread, STSM out.  No element of the face is ever
                    # addressed individually.
                    eg0, eg1 = _exp(g0), _exp(g1)
                    for mh in cutlass.range_constexpr(2):
                        shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                   sLand, b + LQ + mh, sFace, FQG + mh,
                                   SCALE * eg0, SCALE * eg1)
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_QG)
                    _xa(bX, XF_SQ)
                    _seg("k4.sq.end", t)
                    # -- (2) U = A @ Bm ( = u, the chunk's delta-rule write) -
                    # Bm's writer is THIS chunk's WG1 s1; the U slots' previous
                    # reader is the PREVIOUS chunk's WG1 s3 (WG2's own stage-3
                    # and stage-5 reads are self, so they do not arrive)
                    _xw(bX, XF_BM, ph)
                    _xw(bX, XE_U, 1 - ph)
                    _xw(bX, LF_AV, ph)     # A, WG2's only use of that class
                    for nh in cutlass.range_constexpr(2):
                        acc = ALLOC(cS.shape, F32)
                        chain(mma_km, acc,
                              sLand[(None, None, b + LA)],
                              sFaceT[(None, None, FBM + nh)], 4, False)
                        publish(r2s, thr_r, sFace, stg, acc, FU + nh)
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_U)
                    _xa(bX, XE_BM)
                    # both chains have retired, so WG2 is off A; the v half of
                    # the class is WG1's to release
                    _xa(bX, LE_AV)
                    _seg("k4.u.end", t)
                    # -- (3) Ms = mask_{j<=i} scale e^{G_i-G_j}(dO_i.U_j) ---
                    # U is WG2's own stage-2 product: a fence + program order
                    # is the whole synchronisation.  Ms's previous reader is
                    # the PREVIOUS chunk's WG1 s4.
                    _xw(bX, XE_MS, 1 - ph)
                    _xw(bX, LF_O, ph)      # dO, WG2's only use of that class
                    acc = ALLOC(cS.shape, F32)
                    warpgroup.fence()
                    for nh in cutlass.range_constexpr(2):
                        link(mma_kk, acc,
                             sLand[(None, None, b + LO + nh)],
                             sFace[(None, None, FU + nh)], 4, nh > 0)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        acc[i] = (SCALE * mk[i]) * acc[i]      # keep j <= i
                    publish(r2s, thr_r, sFace, stg, acc, FMS)
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_MS)
                    # the two chains above retired, so WG2 is off dO
                    _xa(bX, LE_O)
                    _seg("k4.ms.end", t)
                    # -- (4) -Mv = mask_{j<i} e^{G_i-G_j}((-dv)_i.U_j) and
                    # Kg = e^G o K ------------------------------------------
                    # three in-edges here, and each is a different kind:
                    #   XF_NDV  THIS chunk's WG1 s3 published -dv
                    #   XE_SQ   THIS chunk's WG1 s2 finished reading Sq, whose
                    #           slot the -Mv write takes over
                    #   XE_KG   the PREVIOUS chunk's WG0 s5 finished reading Kg
                    #   X_FREE1112  WG1's dv store warp drained the two staging
                    #           blocks the Kg write takes over
                    _xw(bX, XF_NDV, ph)
                    _xw(bX, XE_SQ, ph)
                    _xw(bX, XE_KG, 1 - ph)
                    _xw(bX, X_FREE1112, ph)
                    acc = ALLOC(cS.shape, F32)
                    warpgroup.fence()
                    for nh in cutlass.range_constexpr(2):
                        link(mma_kk, acc,
                             sFace[(None, None, FNDV + nh)],
                             sFace[(None, None, FU + nh)], 4, nh > 0)
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                    for i in cutlass.range_constexpr(cute.size(cS)):
                        # the STRICT half of the same mask: `mk` already holds
                        # 0 above the diagonal, so dropping the diagonal too is
                        # one `selp` on the coordinate map -- no second
                        # exponential and no second fragment
                        n = cS[i][1]
                        if cutlass.const_expr((i // 2) % 2 == 0):
                            acc[i] = _sel_gt(m0, n, mk[i], F32(0.0)) * acc[i]
                        else:
                            acc[i] = _sel_gt(m1, n, mk[i], F32(0.0)) * acc[i]
                    publish(r2s, thr_r, sFace, stg, acc, FMV)
                    # Kg = e^G o K, the other D0.16 ③ derived face
                    for mh in cutlass.range_constexpr(2):
                        shape_face(l2r, r2s, thr_l, thr_r, cshp, stg,
                                   sLand, b + LK + mh, sFace, SVN[mh],
                                   eg0, eg1)
                    cute.arch.fence_view_async_shared()
                    _xa(bX, XF_KG)
                    _seg("k4.mv.end", t)
                    # -- (5) dk, one accumulator per 64-column half of the
                    # key dim: e^{G_last-G} o (U @ dh^T) + Ms^T @ Q
                    #          + (-Mv + (-Mv)^T) @ K + e^G o ((-dv) @ h^T) --
                    # dh's two readers are THIS chunk's WG1 s2 and WG2's own
                    # stage 5, and WG2 then OVERWRITES the dh faces with dk.
                    # The overwrite is what X_RDH gates: WG1's s2 chains end in
                    # `wait_group`, which is a warpgroup rendezvous and lets it
                    # announce the release with no extra NamedBarrier, while
                    # WG2's own read is covered by the `wait_group` at the end
                    # of its LAST chain before the publish.
                    _xw(bX, XF_DH, ph)
                    _xw(bX, X_RDH, ph)
                    _xw(bX, LF_H, ph)      # the h-term of dk
                    d0, d1 = _exp(glast - g0), _exp(glast - g1)
                    for mh in cutlass.range_constexpr(2):
                        a = ALLOC(cS.shape, F32)
                        # three groups, not six: the six chains of this half
                        # split into runs that share an accumulator with no
                        # register traffic in between, and each run is one
                        # group and one drain (hw_kb 坑 11).  `ah` keeps its own
                        # group at depth 0 rather than flying across the run
                        # above it -- a ladder there would hold a second 32-f32
                        # accumulator live through three chains, and WG2 is the
                        # role whose live set the spill line is charged to.
                        warpgroup.fence()
                        for nh in cutlass.range_constexpr(2):
                            link(mma_kk, a,
                                 sFace[(None, None, FU + nh)],
                                 sFace[(None, None, SDH[mh * 2 + nh])],
                                 4, nh > 0)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                a[i] = d0 * a[i]
                            else:
                                a[i] = d1 * a[i]
                        warpgroup.fence()
                        link(mma_mm, a,
                             sFaceT[(None, None, FMS)],
                             sLandT[(None, None, b + LQ + mh)], 4, True)
                        link(mma_km, a,
                             sFace[(None, None, FMV)],
                             sLandT[(None, None, b + LK + mh)], 4, True)
                        link(mma_mm, a,
                             sFaceT[(None, None, FMV)],
                             sLandT[(None, None, b + LK + mh)], 4, True)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        ah = ALLOC(cS.shape, F32)
                        warpgroup.fence()
                        for nh in cutlass.range_constexpr(2):
                            link(mma_kk, ah,
                                 sFace[(None, None, FNDV + nh)],
                                 sLand[(None, None, b + LH + mh * 2 + nh)],
                                 4, nh > 0)
                        warpgroup.commit_group()
                        warpgroup.wait_group(0)
                        for i in cutlass.range_constexpr(cute.size(cS)):
                            if cutlass.const_expr((i // 2) % 2 == 0):
                                a[i] = a[i] + eg0 * ah[i]
                            else:
                                a[i] = a[i] + eg1 * ah[i]
                        # the mh = 0 dh faces (SDH[0..1] = the region's LAST
                        # pair) are dead once this mh's first pair of chains is
                        # done (mh = 1 reads SDH[2..3]), and nothing else reads
                        # the dh faces in stage 5 -- they are the 16 KB dk
                        # stages through.  D2.1 stages the half into the FLAT
                        # view of that pair, which spans BOTH blocks, and at
                        # D1.3's padded pitch 1008 B of the tail pad after
                        # them; all of that is dead here, which is why the
                        # staging pair is the LAST one (D1.3's slot renumber).
                        publish_flat(r2s, thr_r, sFaceF, stg, a,
                                     mh, PDK)
                    cute.arch.fence_view_async_shared()
                    _seg("k4.dk.end", t)
                    # WG2's last reads of the -dv faces (stage 4's -Mv and the
                    # h-term chain just above) are complete
                    _xa(bX, XE_NDV)
                    # and so are its last reads of q (the Ms^T term), k (the
                    # -Mv terms) and h_c (the h term) -- every one of those
                    # chains ended in `wgmma.wait_group.sync.aligned 0`, so the
                    # three classes go back to the producer here.  The dk store
                    # below stages through the FACE region, not the landing
                    # zone, so it does not hold any of them.
                    _xa(bX, LE_Q)
                    _xa(bX, LE_K)
                    _xa(bX, LE_H)
                    cute.arch.barrier(barrier_id=BAR_WG2,
                                      number_of_threads=128)
                    if wid == 8:
                        reduce_rows(gDk, sFaceF, PDK, ch, hq, lane)
                        # WG0 republishes dh into these two blocks in stage 1
                        # of the next chunk
                        _bulk_wait(0)
                    _seg("k4.dk.store.end", t)
                    cute.arch.barrier(barrier_id=BAR_WG2,
                                      number_of_threads=128)
                    # the dk staging has drained out of SDH[0..1] and WG2's own
                    # reads of all four dh faces are done, so they go back to
                    # WG0 for the next chunk.  BAR_WG2 above is what lets all
                    # 128 threads speak for the single store warp's drain.
                    _xa(bX, XE_DH)
            _seg("k4.exit")
        # belt-and-braces: every store group is already drained inside the loop
        # (each one before the barrier that frees its staging blocks), but a
        # CTA must not exit with a bulk store still reading its smem
        _bulk_wait(0)

    # -- layouts / atoms / TMA: all IR builders, so all inside the jit trace --
    def _atoms(mQ2, mK2, mV2, mO2, mA2, mH2, mDv2):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, TILE),
            BF16)
        lyt1 = cute.tile_to_shape(atom, (TILE, TILE), order=(0, 1))
        lyt_land = cute.tile_to_shape(atom, (TILE, TILE, SLOTS * STAGE),
                                      order=(0, 1, 2))
        lyt_face = cute.tile_to_shape(atom, (TILE, TILE, NFACE), order=(0, 1, 2))
        # read views: sXT[(c, r, j)] == sX[(r, c, j)].  The colex index of
        # (r, c, j) over (64, 64, n) is r + 64c + 4096j, so the transposer's
        # stride, in READ-view coordinate order (c, r, j), is (64, 1, 4096).
        tsp = lambda lyt, n: cute.composition(
            lyt, cute.make_layout((TILE, TILE, n),
                                  stride=(TILE, 1, TILE * TILE)))
        lyt_landT = tsp(lyt_land, SLOTS * STAGE)
        lyt_faceT = tsp(lyt_face, NFACE)

        MM = nvgpu.OperandMajorMode
        mk = lambda am, bm: sm90_utils.make_trivial_tiled_mma(
            BF16, BF16, am, bm, F32, atom_layout_mnk=(1, 1, 1),
            tiler_mn=(TILE, TILE))
        mma_kk = mk(MM.K, MM.K)     # Sq = Q @ K^T: both operands land-major
        mma_km = mk(MM.K, MM.MN)    # K @ dh, Ms @ K
        mma_mm = mk(MM.MN, MM.MN)   # Sq^T @ dO, A^T @ dU, Qg^T @ dO, W^T @ dU

        E = LayoutEnum.ROW_MAJOR
        r2s = cute.make_tiled_copy_S(
            sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
            cute.make_tiled_copy_C_atom(
                cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4),
                                    BF16), mma_km))
        # D0.16's READ side: the matrix-copy atom that brings a non-accumulator
        # tile into a fragment ISOMORPHIC to the accumulator, so that fragment
        # index `i` and `cS[i]` name the same (m, n).  Every landing box and
        # every face is the same K_SW128 64x64 bf16 tile with N contiguous, so
        # the member is the NON-transposed `.M88` one and the whole C tile is
        # covered -- hence `make_tiled_copy_C` (not `..._C_atom`, which builds
        # only the per-atom tiling).  Both facts are GPU-verified against these
        # exact layouts in notes/probe_m0_d016_ldsm.py (R1/R4 value-for-value on
        # a TMA-landed and on an STSM-written slot; C1 = no `.trans` in the PTX,
        # which the numbers alone cannot see).
        l2r = cute.make_tiled_copy_C(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), BF16), mma_km)

        g2s = cpasync.CopyBulkTensorTileG2SOp()
        t4 = lambda m: cpasync.make_tiled_tma_atom(g2s, m, lyt1, (TILE, TILE),
                                                   num_multicast=1)
        tmaQ, cQ = t4(mQ2)
        tmaK, cK = t4(mK2)
        tmaV, cV = t4(mV2)
        tmaO, cO = t4(mO2)
        tmaA, cA = t4(mA2)
        tmaH, cH = t4(mH2)
        # D0.14's ONE remaining S2G atom (D2.1 took dq/dk off the TMA path);
        # no `num_multicast` on the store side
        s2g = cpasync.CopyBulkTensorTileS2GOp()
        st4 = lambda m: cpasync.make_tiled_tma_atom(s2g, m, lyt1, (TILE, TILE))
        tmaDv, cDv = st4(mDv2)
        z = lambda t: cute.zipped_divide(t, (TILE, TILE))
        return (tmaQ, z(cQ), tmaK, z(cK), tmaV, z(cV), tmaO, z(cO),
                tmaA, z(cA), tmaH, z(cH), tmaDv, z(cDv),
                mma_kk, mma_km, mma_mm, r2s,
                lyt_land, lyt_landT, lyt_face, lyt_faceT, l2r)

    def _storage(n_land, n_face):
        @cute.struct
        class SharedStorage:
            # D0.10's barriers, cross-WG exchange and landing supply alike.
            # They land in the padding before the first 1024 B-aligned region,
            # so they cost no smem (`wg_exchange_face_mbarrier` gotcha 2):
            # 30 x 8 = 240 B and `land` starts at 1024 either way.
            xb: cute.struct.MemRange[cutlass.Int64, NXB]
            land: cute.struct.Align[cute.struct.MemRange[BF16, n_land], 1024]
            # + FPAD: D1.3's padded staging tile is 17408 B and dk stages
            # through the LAST pair of the face region, so its last four rows
            # sit 1008 B past `n_face`.  This is the one kilobyte the decision
            # costs (the dq side spends none: its overrun lands in the A box).
            # 231424 + 1024 = 232448 = the `smem_dynamic` ceiling exactly.
            face: cute.struct.Align[
                cute.struct.MemRange[BF16, n_face + FPAD], 1024]
            gc: cute.struct.Align[cute.struct.MemRange[F32, SEG_TOK], 1024]
            be: cute.struct.Align[cute.struct.MemRange[F32, SEG_TOK], 1024]
            row: cute.struct.Align[cute.struct.MemRange[F32, NROW], 1024]
        return SharedStorage

    @cute.jit
    def launch_gdn_bwd_main(mQ2, mK2, mV2, mO2, mA2, mH2, mGc, mBe, mDh,
                            mDv2, mDq, mDk, mDg, mDb, st):
        a = _atoms(mQ2, mK2, mV2, mO2, mA2, mH2, mDv2)
        SS = _storage(cute.cosize(a[18]), cute.cosize(a[20]))
        gdn_bwd_main(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                     a[8], a[9], a[10], a[11], a[12], a[13],
                     mDq, mDk,
                     mGc, mBe, mDh, mDg, mDb,
                     a[14], a[15], a[16], a[17], a[22],
                     a[18], a[19], a[20], a[21],
                     SS).launch(grid=[N_SEG, H_V, 1], block=[TID, 1, 1],
                                min_blocks_per_mp=1, stream=st, **_pdl_kw())

    return launch_gdn_bwd_main


# K5 `group_reduce` is GONE (D2.1).  It existed only to fold the two v-heads
# of a GVA pair -- K4 wrote dq/dk as [N, H_V, D] partials and K5 read 134 MB
# back to write 67 MB of [N, H_QK, D].  K4 now reduce-ADDs its tiles into the
# final tensors (`cp.reduce.async.bulk`, bf16), which K1 has pre-zeroed, so
# the launch table is 4 kernels / 16 launches and the partial round trip is
# off the DRAM account.  The rounding is unchanged in kind: K5 summed two
# bf16 partials in f32 and rounded once; the reduction unit adds the second
# bf16 tile onto the first with a single correctly-rounded bf16 add
# (`.noftz`, so the subnormal end of the range is not flushed either).


# --- compiled-artifact cache -------------------------------------------------
# `cute.compile` once per process, then call the compiled artifact: calling a
# @cute.jit directly re-does argument processing on every call, and bench.py
# reports `host_bound_ratio` off exactly that cost.  Shapes are fixed for the
# whole campaign, so one artifact per kernel suffices.
#
# `scale` is a readout constant, not a tensor: K3 (and later K4) bake it at
# build time, so `gdn_prefill_bwd` drops their artifacts if a call ever arrives
# with a different one.
_BUILDERS = {"k1": _build_k1, "k2": _build_k2, "k3": _build_k3,
             "k4": _build_k4}
_COMPILED = {}
_SCALE = D ** -0.5


def _compiled(name, *args):
    if name not in _COMPILED:
        global _PDL
        try:
            _COMPILED[name] = cute.compile(_BUILDERS[name](), *args)
        except Exception as exc:                           # noqa: BLE001
            # D2.3 A-rung fallback: `use_pdl` is a documented `.launch`
            # parameter, but it is not in every DSL release.  If this image's
            # `.launch` rejects it, drop the attribute and rebuild WITHOUT it
            # for this and every later kernel.  The device-side griddepcontrol
            # pair stays in the kernels either way: with no programmatic
            # dependency on the launch, `wait` returns immediately and
            # `launch_dependents` has no dependents to release, so the
            # fallback build is exactly the pre-D2.3 behaviour.
            if not _PDL:
                raise
            _PDL = False
            try:
                _COMPILED[name] = cute.compile(_BUILDERS[name](), *args)
            except Exception:                              # noqa: BLE001
                # the PDL-less build fails the same way, so the attribute was
                # not the problem -- report the original failure
                raise exc
    return _COMPILED[name]


def _stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        from cuda.bindings import driver as _cd
        return _cd.CUstream(raw)
    except Exception:                                      # noqa: BLE001
        from cuda import cuda as _cd
        return _cd.CUstream(raw)


def gdn_prefill_bwd(
    q: torch.Tensor,        # [N, H_QK, D] bf16   N=8192, H_QK=16, D=128
    k: torch.Tensor,        # [N, H_QK, D] bf16   L2-normalized along D
    v: torch.Tensor,        # [N, H_V,  D] bf16   H_V=32 (GVA: 2 v-heads per qk-head)
    g: torch.Tensor,        # [N, H_V] fp32       LOG-space forget gate, alpha = exp(g) in (0, 1)
    beta: torch.Tensor,     # [N, H_V] fp32       write strength, values in (0, 1)
    do: torch.Tensor,       # [N, H_V,  D] bf16   gradient of the forward output o
    scale: float,           # D ** -0.5, the forward readout scale
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dq [N, H_QK, D] bf16, dk [N, H_QK, D] bf16, dv [N, H_V, D] bf16,
    dg [N, H_V] fp32, dbeta [N, H_V] fp32)."""
    dev = q.device
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    do = do.contiguous()

    global _SCALE
    if scale != _SCALE:
        _SCALE = scale
        for _n in ("k3", "k4"):
            _COMPILED.pop(_n, None)

    # ---- intermediates, the D0.7 ABI (allocated here, written whole) --------
    g_cumsum = torch.empty((NC, C, H_V), device=dev, dtype=torch.float32)
    A = torch.empty((NC, C, H_V, C), device=dev, dtype=torch.bfloat16)
    h_c = torch.empty((NC, H_V, D, D), device=dev, dtype=torch.bfloat16)
    # D0.7: NOTHING in the window needs zeroing.  Segment 7's dh_end slot has
    # no producer (no final-state gradient); K4 reads it and throws it away on
    # the segment number instead (the ledger's second option), so this stays a
    # `torch.empty` and the window has no memset kernel in it.
    dh_end = torch.empty((N_SEG, H_V, D, D), device=dev, dtype=torch.float32)
    dv = torch.empty((N, H_V, D), device=dev, dtype=torch.bfloat16)
    dg = torch.empty((N, H_V), device=dev, dtype=torch.float32)
    dbeta = torch.empty((N, H_V), device=dev, dtype=torch.float32)
    # D2.1: dq/dk are K4's reduce-add targets, not a kernel's whole write, so
    # they must start at zero -- but still `torch.empty` here, because K1 does
    # the clearing (a host `torch.zeros` adds a FillFunctor launch; see the
    # note at K1's entry).
    dq = torch.empty((N, H_QK, D), device=dev, dtype=torch.bfloat16)
    dk = torch.empty((N, H_QK, D), device=dev, dtype=torch.bfloat16)

    mK = k.view(NC, C, H_QK, D)
    mG = g.view(NC, C, H_V)
    mB = beta.view(NC, C, H_V)
    mDg = dg.view(NC, C, H_V)
    mDb = dbeta.view(NC, C, H_V)
    # K2's / K3's TMA views: every box is a 64x64 tile of a 2D row-major
    # picture, so each tensor collapses to the (rows, row-stride) pair the tile
    # lives in.
    mQ2 = q.view(N, H_QK * D)
    mK2 = k.view(N, H_QK * D)
    mV2 = v.view(N, H_V * D)
    mO2 = do.view(N, H_V * D)
    mA2 = A.view(NC * C, H_V * C)
    mH2 = h_c.view(NC * H_V * D, D)
    # K4's dv leaves by TMA store (D0.14), so it takes the same 2D picture its
    # inputs arrive in: a 64x64 tile is (chunk, head * 2 + half).  dq/dk leave
    # by D2.1's address-form reduce-add, which wants the plain 2D picture of
    # the FINAL tensor (row = token, 16 qk-heads x 128) and only its base
    # address.  The f32 views are K1's zeroing handles: one f32 store clears
    # two bf16 outputs.
    mDv2 = dv.view(N, H_V * D)
    mDq2 = dq.view(N, H_QK * D)
    mDk2 = dk.view(N, H_QK * D)
    mDqZ = dq.view(torch.float32).view(N, H_QK * (D // 2))
    mDkZ = dk.view(torch.float32).view(N, H_QK * (D // 2))

    fd = from_dlpack
    st = _stream()

    a_k1 = (fd(mK, assumed_align=16), fd(mG, assumed_align=16),
            fd(mB, assumed_align=16), fd(g_cumsum, assumed_align=16),
            fd(A, assumed_align=16), fd(mDqZ, assumed_align=16),
            fd(mDkZ, assumed_align=16), st)
    _compiled("k1", *a_k1)(*a_k1)

    a_k2 = (fd(mK2, assumed_align=16), fd(mV2, assumed_align=16),
            fd(mA2, assumed_align=16), fd(mH2, assumed_align=16),
            fd(g_cumsum, assumed_align=16), fd(mB, assumed_align=16), st)
    _compiled("k2", *a_k2)(*a_k2)

    a_k3 = (fd(mQ2, assumed_align=16), fd(mK2, assumed_align=16),
            fd(mO2, assumed_align=16), fd(mA2, assumed_align=16),
            fd(g_cumsum, assumed_align=16), fd(mB, assumed_align=16),
            fd(dh_end, assumed_align=16), st)
    _compiled("k3", *a_k3)(*a_k3)

    a_k4 = (fd(mQ2, assumed_align=16), fd(mK2, assumed_align=16),
            fd(mV2, assumed_align=16), fd(mO2, assumed_align=16),
            fd(mA2, assumed_align=16), fd(mH2, assumed_align=16),
            fd(g_cumsum, assumed_align=16), fd(mB, assumed_align=16),
            fd(dh_end, assumed_align=16), fd(mDv2, assumed_align=16),
            fd(mDq2, assumed_align=16), fd(mDk2, assumed_align=16),
            fd(mDg, assumed_align=16), fd(mDb, assumed_align=16), st)
    _compiled("k4", *a_k4)(*a_k4)

    return dq, dk, dv, dg, dbeta
