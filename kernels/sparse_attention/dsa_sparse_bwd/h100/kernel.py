"""dsa_sparse_bwd kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py and README.md, do not change):

  dsa_sparse_bwd(q, kv, out, dout, lse, attn_sink, topk_idxs, sm_scale,
                 topk_length=None) -> (dq, dkv, d_sink)

    q           : torch.Tensor (S_Q, H_Q, D_QK)  cuda bfloat16
    kv          : torch.Tensor (S_KV, D_QK)      cuda bfloat16  (K === V, flat)
    out         : torch.Tensor (S_Q, H_Q, D_V)   cuda bfloat16  fwd output (given)
    dout        : torch.Tensor (S_Q, H_Q, D_V)   cuda bfloat16  grad wrt out
    lse         : torch.Tensor (S_Q, H_Q)        cuda float32   KV-only natural-log LSE
    attn_sink   : torch.Tensor (H_Q,)            cuda float32
    topk_idxs   : torch.Tensor (S_Q, TOPK)       cuda int32     invalid family: -1 or >= S_KV
    sm_scale    : float
    topk_length : torch.Tensor (S_Q,) cuda int32 | None         left-prefix truncation

    dq     : torch.Tensor (S_Q, H_Q, D_QK) cuda bfloat16   deterministic reduction
    dkv    : torch.Tensor (S_KV, D_QK)     cuda bfloat16   fp32 accumulation, bf16 out-cast
    d_sink : torch.Tensor (H_Q,)           cuda float32

  fp32 accumulation end to end; sentinel family masked as a family;
  topk_length truncates the left prefix only; empty row (tl == 0) contributes
  exactly zero gradient; the sink enters through the softmax denominator as
  the fwd contract folded it, and d_sink follows from attn_sink/dout.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc. Delegating the math to a library reachable in the image fails the harness
(eval/bench.py scans this file; the banned-token list is in its fence).
JIT-compile once and cache the compiled kernel across calls (the bench times
repeated calls with rotating buffers).

--------------------------------------------------------------------------
M0 architecture state
--------------------------------------------------------------------------
Pipeline (D0.1): 5 launches on one stream, dependency = stream order.

  K0a  d_sink   zero fill        (accumulate-into target of K1)
  K0b  dkv_ws   zero fill        (accumulate-into target of K2)
  K1   preprocess                Delta / lse2 / d_sink
  K2   main                      dq / dkv_ws
  K3   cast                      dkv_ws f32 -> dkv bf16

Intermediate tensor ABI (D0.3) / precision (D0.4):

  delta  (S_Q, H_Q) f32   rowsum(O . dOut), host empty, K1 -> K2
  lse2   (S_Q, H_Q) f32   log2e * logaddexp(lse, sink), host empty, K1 -> K2
  d_sink (H_Q,)     f32   host zeros (K0a), K1 red.global.add.f32
  dkv_ws (S_KV,D_QK) f32  host zeros (K0b), K2 accumulates, K3 casts
  dq     (S_Q,H_Q,D_QK) bf16 host empty, K2 writes every element exactly once

The dkv_ws column order is the student's own choice (the contract is only
"fp32 accumulate -> zeroed -> one cast").  D0.19 spends it on the write-back
address account: (S_KV, D_QK) row-major, but inside each CH = 64 wide dKV
chunk logical column n is stored at slot

  PERM(n) = 4*((n%8)//2) + (n//8)%4 + 16*(n%2) + 32*((n//8)//4)

(a bijection of [0, 64) onto itself; K3 applies the same forward map when it
reads).  Two properties are bought here, and the second one is what the
reduction actually costs:

  * a lane's four registers 4t apart (t = 0..3) land on four CONSECUTIVE f32
    slots, so the write-back is `red.global.add.v4.f32` rather than four times
    as many scalar reductions;
  * the four LANES of a warp that share an accumulator row (they differ only
    in n0 = 2*((tid%32)%4)) land on four ADJACENT 16-byte granules, so one
    warp's reduction instruction covers 16 whole 32-byte L2 sectors instead of
    half-writing 32 of them.  Per-thread the offsets then step 0x40 inside a
    row, which is the reference's own stride.

K2 work decomposition / resource topology (D0.2, D0.5-D0.13):

  grid    (S_Q, H_Q/64, 1), block 256 = 2 warpgroups.  One CTA owns one
          (query token, 64-head half); the token's shared top-k list is the
          only inner axis, so M = 64 query heads (GQA pack factor 64).
  smem    Q tile 64x512 bf16 and dO tile 64x512 bf16 RESIDENT (loaded once per
          CTA, never swapped); KV block 64x512 bf16 SINGLE-BUFFERED (one
          gather per block, next block reuses the slot); P and dS 64x64 bf16
          single-slot staging; 128 f32 row scalars + 64 i32 gathered row ids.
  tiles   KV block = 64 rows x 512 d; the dQ d-axis is split in half, WG0 owns
          dq[:, 0:256] and WG1 owns dq[:, 256:512], one 64x128 accumulator per
          chain so each warpgroup carries 128 accumulator registers.
  roles   no dedicated producer warp.  WG0 = KV gather + S/dP + softmax-bwd +
          P/dS publish + dq[:, 0:256]; WG1 = dKV chunks + dkv_ws reduction +
          dq[:, 256:512].
  chains  S = Q.KV^T and dP = dO.KV^T (A k-major smem desc, B k-major smem
          desc, 64x64x16, 32 k-blocks); dQ += dS.KV (A = dS k-major, B = a
          MN-major VIEW of the same KV bytes, 64x128x16, 2 chains per WG);
          dKV chunk = P^T.dO + dS^T.Q (A = MN-major view of P/dS, B = MN-major
          view of dO/Q, 64x64x16, 8 chunks x 2 chains).  Every transposed
          operand is a second `get_tensor` on the SAME MemRange through a
          host-side `cute.composition` transposer — one allocation, two views,
          no extra smem.
  order   (D0.12) WG0 issues S and dP back to back and runs the exp2/P-pack
          segment under the dP chain's flight (wait_group(1)), then packs dS
          under a drained pipe (wait_group(0)).  WG1 keeps two chunk chains in
          flight (wait_group(2)) and retires the previous chunk's dkv_ws
          reduction in that shadow.  WG1's own two dQ chains are NOT a tail
          block: chain j is issued between chunk pairs, right after pair
          DQ_AT[j] = 0, 1, and is retired by the `wait_group(2)` the NEXT pair
          performs anyway -- a dQ chain committed after pair c is older in the
          commit FIFO than pair c+1, so the wait that retires pair c retires it
          too, at no drain of its own.  That is what lets the KV handoff go out
          at pair KV_AT = 2 instead of at the body tail, with FIVE chunk pairs
          still to run: WG1 has no reader of the KV block left after its dQ
          chains retire (chunks 2..7 touch only P/dS/dO/Q), so WG0's block i+1
          gather + S + dP runs UNDER WG1's block i rather than after all of it.
          The single-buffered gather cannot be hidden inside WG0 (no cross-block
          prefetch), so this is the only place it can hide.
          (D1.4) both placements are two pairs earlier than the first build's
          (chains after pairs 2/3, credit after pair 4).  Moving the credit
          alone does not pay -- WG0 cannot use a slot it is not ready to fill --
          so it goes in with D1.3 below, which is what shortens WG0's side of
          that window; the pair is the supply window (G3), and its benefit is
          the two halves together.
          (D1.1) three waits/credits moved WITHOUT changing any depth, wait
          target or masking relation, and all three INSIDE one body -- nothing
          is carried across the loop back edge: WG0's dQ drain stays at the
          body tail but now sits AFTER the next step's row-id publish, so that
          publish's index load runs under the chains; WG1's dQ chain is issued
          BEFORE the retired chunk's REDG write-back instead of after it (same
          commit-FIFO slot); WG1's 0x3/0x4 credits are hoisted over the LAST
          chunk's write-back, which reads only registers and gmem, not those
          two smem slots.  The earlier form of this decision made WG0's drain
          loop-carried (taken at the top of the next body); that put an
          in-flight group on the back edge and ptxas answered by serializing
          every HGMMA in the kernel -- `commit` axis `LE 0x0` x72 in BOTH
          consumer loops with `LE 0x1`/`LE 0x2` gone, tensor 37.3 -> 26.6 %,
          stall_barrier 3.45 -> 7.89, spill still 0, device 67.2 -> 93.7 ms.
          That is path 1 of hw_kb/cutedsl/sm90/wgmma_backedge_pending_group.md and its
          rule is followed here: drain before the back edge (form (a)).
  scale   the softmax sees sm_scale * <q, kv>, so the chain rule puts one
          sm_scale on the whole dS path (dq and the dS^T.Q half of dkv) and
          none on the P^T.dO half.  It is folded into the dS pack, which is the
          one site every dS consumer goes through, so it is paid exactly once.

  sync    (D0.14) 7 named barriers + barrier 0.  Every CROSS-warpgroup edge is
          a one-way publish/handoff -- one side `bar.arrive`s and walks on, the
          other side `bar.sync`s -- not a symmetric all-hands rendezvous:

            0x0 256  CTA rendezvous, prologue and epilogue only
            0x1 256  WG0 arv (P written + fence)  -> WG1 sync at body top
            0x2 256  WG0 arv (dS written + fence) -> WG1 sync after its first
                                                     P^T.dO chain is issued
            0x3 256  WG1 arv (body tail)          -> WG0 sync before writing P
            0x4 256  WG1 arv (body tail)          -> WG0 sync before writing dS
            0x5 256  WG1 arv (first wait with two -> WG0 sync before its dQ
                     chunk pairs in flight)          chains: phase gate
            0x8 128  intra-WG0: this WG's cp.async has landed
            0x9 256  WG1 arv MID BODY, at the wait that retires its dQ chains
                     (chunk pair 4 of 8)          -> WG0 sync at body top:
                                                     "KV + row-id slot free"

          Blocking rendezvous per KV block: WG0 = 5 (0x9, 0x8, 0x3, 0x4, 0x5),
          WG1 = 2 (0x1, 0x2); arrive-only: WG0 = 2 (0x1, 0x2), WG1 = 4 (0x5,
          0x9, 0x3, 0x4).  0x5 does double duty as the intra-WG0 fence WG0's
          dQ chains need before reading the dS tile all 128 lanes co-wrote,
          which is what keeps WG0's blocking count at 5 rather than 6.
  phase   (D0.16) WG1 runs about half a body behind WG0 and the two never use
          the tensor pipes in lockstep.  0x1/0x2/0x5 are "this trip"; 0x3/0x4/
          0x9 are "last trip" (they release the buffers block i-1 used).  0x9
          leaves WG1 in MID body and 0x3/0x4 at its tail, and the difference is
          the overlap window: everything WG0 can do on the strength of the KV
          block alone (gather, S, dP, exp2/P-pack) proceeds against WG1's
          second half, and only the P/dS publish waits for WG1 to finish.  The
          stagger is set up in the prologue: WG1 seeds one credit each on
          0x3/0x4/0x9 and then does nothing until WG0 publishes P(0), while
          WG0's first body consumes those seeded credits.  The row-id slab is
          double buffered for the same reason -- WG0 publishes block i+1's ids
          at the tail of body i into the other slot while WG1 is still reading
          block i's, so the steady-state body spends no barrier on them (0x9
          at body top is their visibility fence).
  supply  (D0.15 / D0.17) the KV block is gathered by `cp.async`
          (`LDGSTS.E.BYPASS.128`, `cache_mode=GLOBAL`): a 1024-byte KV row is
          64 lanes x 128 bit, a warpgroup is two row teams, so each thread
          issues KV_TRIPS = 32 copies per block.  `[GATHER: prefetched]` -- the
          block's 64 row ids are fetched in one batch a whole body ahead,
          sanitized (both sentinel families and the `pos >= tl` tail collapse
          to -1) and published through the same smem slab WG1 reads, so no
          copy waits on its own index.  (D1.3) their CONSUMPTION is batched to
          match, and the batch is held LANE-OWNED: each lane reads exactly ONE
          id out of the slab (lane L of every warp holds the team's row L, 32
          consecutive i32 slots per warp = one conflict-free transaction) and
          the burst broadcasts row i with `shfl.sync.idx.b32`.  The burst is
          therefore not 32 serial `LDS` -> compare -> branch -> `LDGSTS` chains
          standing between the 0x9 handoff and the first copy: the smem round
          trip is paid once per block, and what stands in front of each copy is
          a register shuffle.  A team's rows are a contiguous segment
          (team*32 + i) rather than interleaved, which is what makes the team's
          ids one lane-owned run.  An invalid row is never dereferenced;
          its smem row is written zero, which makes S, dP and dq see an exact
          zero.
  tail    (D0.18) the CTA walks its ceil(tl/64) blocks TAIL FIRST: step 0 is
          the partial block, steps 1.. are the full ones.  The `pos < tl` test
          is therefore evaluated exactly once, in the prologue publish that
          feeds the peeled step; a steady-state step's publish covers a block
          that is full by construction and tests only `does the next step
          exist`, one warp-uniform compare.  A zero-filled row costs one
          `STS.128` per lane (the gather's own TV layout, universal atom), not
          KV_EPT scalar stores.  `idx[t, tl:]` is never addressed: out-of-range
          positions are clamped away by a select before the index load, and
          WG1 skips the reduction of any row whose id came back -1.

  publish (D0.9) every fp32 C-frag that has to reach smem goes through
          StMatrix, never through per-element stores: P and dS each leave in 4
          `STSM.16.M88.4` (32 bf16 per thread, 8 per x4 store) and each dQ
          chain's 64x128 accumulator in 8 more, landing in its own 128-column
          window of the staging slab.  One `LayoutEnum` (ROW_MAJOR, the enum
          the faces were allocated with) drives the store op and the
          `StMatrix8x8x16bOp` register grouping together, so the bytes STSM
          lands are cell for cell the bytes the scalar stores landed and every
          MN-major consumer keeps reading through its transposed view.  The two
          row-broadcast scalars (lse2, Delta) are hoisted into registers by the
          same account: a 64xN C-frag gives each thread exactly two rows, m0
          and m0+8, so four `LDS` outside the block loop plus a trace-time
          select on the register index replace one `LDS` per element.
  epi     (D0.19) dKV write-back = `red.global.add.v4.f32` (REDG.E.ADD.F32x4),
          8 per chunk per thread = 64 per KV block, addressed off two row-id
          reads per block rather than one per element; dQ write-out = the two
          warpgroups publish their f32 accumulators into the dead Q tile and
          the whole 64x512 bf16 slab leaves in 16 128-bit stores per thread.
          The same 128-bit slab atom, read backwards, fills the resident Q and
          dO tiles in the prologue.  All of it rests on GPU-confirmed
          bijections: the flat-register -> (m, n) map of `partition_C` for a
          64xN warpgroup accumulator, the STSM landing for both the standalone
          64x64 face and the 128-column window of a 64x512 slab, and the fact
          that `partition_S` / `partition_D` of one `make_tiled_copy_tv` slice
          agree across the 128B-swizzled smem tile and the gmem slab.

Supply and the epilogue are still at their conservative floor where the
mechanism is a runtime one: the Q and dO tiles are filled by ordinary vector
copies rather than by TMA, and the dq slab leaves by vector stores rather than
by `UTMASTG`, so there are no 0x6/0x7 intra-WG epilogue barriers.  Register
allocation is left to the compiler.  Those mechanisms are separate decisions
and go on one at a time, each followed by a whole-machine smoke, because they
are the trace-blind / launch-fatal ones.

Calling convention (hw_kb/cutedsl/sm90/cfrag_fused_pack_liveness.md gotcha "probe
side" 1, cfrag_elementwise_fastpath.md "call convention"): a `cutlass.Constexpr`
parameter is baked in by `cute.compile` and must NOT be passed again to the
compiled object — the argument lists of `cute.compile(fn, *args)` and
`compiled(*args)` have to match item for item, and an arity mismatch kills the
process before anything is measured.  Therefore no entry here takes a
Constexpr parameter at all: every compile-time constant is a module-level
global, read while tracing, and set by the host right before `cute.compile`
under the same lock that fills the JIT cache.
"""

import threading
from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

LOG2E = 1.4426950408889634

# K1 shape knobs (D1.2).  A block owns K1_TTOK CONSECUTIVE tokens and all of
# their heads; warp w takes head hg*K1_NWARP + w, and the head-group index hg
# is the INNER walk, so at every step the block's 8 warps read 8 adjacent rows
# and the block streams one contiguous slab of `out` (and one of `dout`)
# strictly ascending.  K1_UNR head groups are in flight at a time.
K1_NWARP = 8
K1_TTOK = 8
K1_UNR = 4
K1_EPT = 8        # bf16 elements per 128-bit load

# --- K2 tile geometry (D0.2 / D0.6, frozen by the milestone) ---------------
H = 64            # M mode: query heads packed per CTA
KVB = 64          # KV rows per block
DQK = 512         # d_qk == d_v
DQN = 128         # dQ chain N mode (2 chains per warpgroup -> 256 d per WG)
CH = 64           # dKV chunk width (8 chunks per KV block)
K2_BLOCK = 256    # 2 warpgroups

# --- KV gather geometry (D0.15): one 128-bit cp.async per thread per trip ---
# ROW_BYTES = DQK*2 = 1024 -> KV_TPR = 1024/16 = 64 threads cover one KV row;
# a warpgroup (128 threads) is KV_TEAMS = 2 row teams, so each thread issues
# KV_TRIPS = KVB/KV_TEAMS = 32 LDGSTS per KV block.
KV_EPT = 8                                   # bf16 elements in 128 bit
KV_TPR = (DQK * 2) // 16                     # 64
KV_TEAMS = (K2_BLOCK // 2) // KV_TPR         # 2
KV_TRIPS = KVB // KV_TEAMS                   # 32
# D1.3 holds a team's row ids LANE-OWNED (lane L of a warp holds the team's
# row L) and broadcasts them with `shfl.sync.idx.b32`, which is a 1:1 map only
# while a team's trip count is exactly one warp wide.
assert KV_TRIPS == 32

# --- named barrier ids (D0.14) ---------------------------------------------
# Every CROSS-warpgroup barrier here is a one-way publish/handoff, not a
# symmetric all-hands rendezvous: one side `barrier_arrive`s (non-blocking,
# publishes and walks on) and the other side `barrier`s (blocks).  Both sides
# contribute 128 arrivals, so the width is the full 256.  id 0 is reserved for
# `sync_threads` (whole-CTA rendezvous) and must not be reused
# (hw_kb/cutedsl/sm90/mma_sync_subtile_broadcast_splitk.md gotcha 7).
BAR_P = 1     # WG0 -> WG1   "P(i) is in smem"
BAR_DS = 2    # WG0 -> WG1   "dS(i) is in smem"
BAR_PF = 3    # WG1 -> WG0   "P slot free"   (WG1 finished reading block i-1)
BAR_DSF = 4   # WG1 -> WG0   "dS slot free"
BAR_PH = 5    # WG1 -> WG0   phase gate (staggers the two WGs' tensor use)
BAR_GA = 8    # intra-WG0, 128 wide: this WG's cp.async landed
BAR_KV = 9    # WG1 -> WG0   "KV slot + row-id slot free"  (serialization point)

K3_BLOCK = 256

# --- trace-time globals ----------------------------------------------------
# Read while tracing, written by the host under _LOCK immediately before the
# matching cute.compile.  They are never read at call time, so the compiled
# object stays valid for its shape once the cache holds it.
_G_K1_NWARP = K1_NWARP
_G_K1_DV = DQK
_G_K1_TTOK = K1_TTOK
_G_K1_NHG = 1
_G_K1_UNR = K1_UNR
_G_K1_EPT = K1_EPT
_G_K1_GX = 1

_G_K2_SKV = 1
_G_K2_CSCALE = 1.0
_G_K2_SCALE = 1.0
_G_K2_GX = 1
_G_K2_GY = 1

_G_K3_DQK = DQK
_G_K3_BLOCK = K3_BLOCK
_G_K3_CH = CH
_G_K3_GX = 1


def _setg(**kw):
    globals().update(kw)


def _rmem_alloc():
    """by-shape rmem allocator, name+arg-order taken from the live DSL.

    cute_dsl_trace_scope gotcha 4: `cute.make_fragment(shape, dtype)` is gone
    in some versions and the interpreter's `make_fragment_like` suggestion is
    the wrong replacement; the by-shape one is `cute.make_rmem_tensor`.
    """
    import inspect
    fn = (getattr(cute, "make_rmem_tensor", None)
          or getattr(cute, "make_fragment", None))
    if fn is None:
        raise RuntimeError("no by-shape rmem allocator on cutlass.cute")
    try:
        first = list(inspect.signature(fn).parameters)[0].lower()
    except Exception:
        first = ""
    dtype_first = first in ("dtype", "element_type", "elem_type", "ty", "t")

    def alloc(shape, dtype):
        if dtype_first:
            return fn(dtype, shape)
        return fn(shape, dtype)
    return alloc


_ALLOC = _rmem_alloc()


def _make_rs_mma(a_major, b_major, n):
    """Tiled MMA whose A operand comes from REGISTERS (D0.11 `[SOURCE: RS]`).

    The default `sm90_utils.make_trivial_tiled_mma` builds the SS form (both
    operands are matrix descriptors).  Which spelling asks for the RS form is
    version dependent -- the helper may or may not forward an operand-source
    argument, and `warpgroup.MmaF16BF16Op`'s positional order is not fixed
    across DSL minors -- so the cascade is tried here, at trace time, and the
    first spelling that builds wins.  This is the cascade
    notes/probe_rs_dq_chain.py ran on the GPU (its contract A), and the whole
    cost is host side: nothing survives into the kernel but the atom.
    """
    import inspect
    bf, f32 = cutlass.BFloat16, cutlass.Float32
    RM = warpgroup.OperandSource.RMEM
    errs = []
    try:
        params = list(inspect.signature(
            sm90_utils.make_trivial_tiled_mma).parameters)
    except Exception as exc:                                # DSL version drift
        params = []
        errs.append("helper signature unreadable: %r" % (exc,))

    for k in ("a_source", "a_src", "a_operand_source", "operand_source"):
        if k not in params:
            continue
        try:
            return sm90_utils.make_trivial_tiled_mma(
                bf, bf, a_major, b_major, f32, atom_layout_mnk=(1, 1, 1),
                tiler_mn=(H, n), **{k: RM})
        except Exception as exc:
            errs.append("helper %s -> %r" % (k, exc))

    cands = (
        lambda: warpgroup.MmaF16BF16Op(bf, f32, (H, n, 16), RM,
                                       a_major, b_major),
        lambda: warpgroup.MmaF16BF16Op(bf, bf, f32, (H, n, 16), RM,
                                       a_major, b_major),
        lambda: warpgroup.MmaF16BF16Op(bf, f32, (H, n, 16), a_src=RM,
                                       a_major_mode=a_major,
                                       b_major_mode=b_major),
        lambda: warpgroup.MmaF16BF16Op(bf, f32, (H, n, 16), a_source=RM,
                                       a_major_mode=a_major,
                                       b_major_mode=b_major),
    )
    for build in cands:
        try:
            op = build()
        except Exception as exc:
            errs.append("op ctor -> %r" % (exc,))
            continue
        try:
            return cute.make_tiled_mma(op, (1, 1, 1))
        except Exception as exc:
            errs.append("make_tiled_mma -> %r" % (exc,))
    raise RuntimeError("no RS mma spelling worked: " + " | ".join(errs))


# ---------------------------------------------------------------------------
# inline-PTX escapes (hw_kb/cutedsl/sm90/ptx_escape.md form 2 / 3)
# ---------------------------------------------------------------------------

def _f32ty():
    try:
        return T.f32()
    except Exception:                                   # DSL version drift
        from cutlass._mlir import ir
        return ir.F32Type.get()


def _i32ty():
    try:
        return T.i32()
    except Exception:                                   # DSL version drift
        from cutlass._mlir import ir
        return ir.IntegerType.get_signless(32)


@dsl_user_op
def _fmax(a, b, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        _f32ty(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _fabs(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        _f32ty(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "abs.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        _f32ty(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ex2(x, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        _f32ty(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _shfl_bfly(x, lane_mask, *, loc=None, ip=None) -> cutlass.Float32:
    """shfl.sync.bfly.b32 with an immediate lane mask (full-warp membermask).

    lane_mask is a trace-time python int, so it is baked into the asm text and
    the op needs no second operand."""
    return cutlass.Float32(llvm.inline_asm(
        _f32ty(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "shfl.sync.bfly.b32 $0, $1, %d, 0x1f, 0xffffffff;" % int(lane_mask),
        "=f,f",
        # convergent: side-effect-free would let it be hoisted/sunk across the
        # warp-uniform control flow it is only correct inside of
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _shfl_idx_i32(x, lane, *, loc=None, ip=None) -> cutlass.Int32:
    """shfl.sync.idx.b32 with an immediate source lane (full-warp membermask).

    The gather's index supply (D1.3): a team's 32 row ids are held one per lane
    and this broadcasts row `lane`'s id to the whole warp, so the copy burst
    stands behind a register shuffle instead of behind 32 smem loads.  `lane`
    is a trace-time python int and is baked into the asm text.  Marked
    side-effecting for the same reason `_shfl_bfly` is: it is convergent, and a
    pure op could be hoisted or sunk across the warp-uniform control flow it is
    only correct inside of."""
    return cutlass.Int32(llvm.inline_asm(
        _i32ty(),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, %d, 0x1f, 0xffffffff;" % int(lane),
        "=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _cp_async_commit(*, loc=None, ip=None):
    """cp.async.commit_group — packs THIS thread's outstanding cp.async into
    one group.  `cute.arch` only carries the bulk (TMA) spelling of the pair
    and its name drifts across DSL minor versions; the non-bulk two take no
    operands, so inline PTX is both shortest and version-proof
    (hw_kb/cutedsl/sm90/cp_async.md §3).  `has_side_effects=True` is load bearing: a
    void inline asm without it is dead-code eliminated and the symptom is
    silently unfilled smem, not an error (that KB's gotcha 2)."""
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _cp_async_wait(n, *, loc=None, ip=None):
    """cp.async.wait_group N — wait until at most N committed groups are in
    flight.  Per-thread retirement only; cross-thread visibility still needs
    the fence + barrier after it (cp_async.md gotcha 5)."""
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _red_add_f32(addr, val, *, loc=None, ip=None):
    """red.global.add.f32 [addr], val  — fire-and-forget fp32 reduction."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
         cutlass.Float32(val).ir_value(loc=loc, ip=ip)],
        "red.global.add.f32 [$0], $1;", "l,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _red_add_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """red.global.add.v4.f32 [addr], {v0,v1,v2,v3}  (D0.19).

    One 16-byte reduction instead of four 4-byte ones: same reduced bytes, a
    quarter of the L2 reduction transactions and a quarter of the address
    arithmetic.  `addr` must be 16-byte aligned; the workspace column order
    (see `_k2_kernel`) is what makes the four values a contiguous run.
    `cute.arch.atomic_add` is scalar-only, so the vector form is inline PTX.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
         cutlass.Float32(v0).ir_value(loc=loc, ip=ip),
         cutlass.Float32(v1).ir_value(loc=loc, ip=ip),
         cutlass.Float32(v2).ir_value(loc=loc, ip=ip),
         cutlass.Float32(v3).ir_value(loc=loc, ip=ip)],
        "red.global.add.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# K1 — preprocess: Delta, lse2, d_sink   (D0.1 / D0.3 / D0.4 / D1.2)
#
# D1.2 streams `out` and `dout` instead of sampling them.  The previous walk
# gave a warp one FIXED head and marched it down the token axis, so a block's
# 8 KiB of live rows repeated every 128 KiB and the ~1000 blocks resident at
# once had their 8 KiB pieces spread across the whole 4.3 GB tensor.  Here a
# block owns K1_TTOK consecutive tokens and ALL of their heads, with the head
# group as the inner walk, so the block reads its two slabs strictly ascending
# and consecutive blocks are consecutive in memory.  The row itself is loaded
# through a 128-bit copy atom rather than 32 two-byte strided loads, and
# K1_UNR head groups are in flight before any is consumed.
# ---------------------------------------------------------------------------

@cute.kernel
def _k1_kernel(mOut, mDout, mLse, mSink, mDelta, mLse2, mDsink,
               tiled_ld: cute.TiledCopy):
    nwarp = _G_K1_NWARP
    d_v = _G_K1_DV
    ttok = _G_K1_TTOK
    nhg = _G_K1_NHG
    unr = _G_K1_UNR

    tidx, _, _ = cute.arch.thread_idx()
    bx, _by, _ = cute.arch.block_idx()

    lane = tidx % 32
    w = tidx // 32
    t0 = bx * ttok

    # D1.2 supply: a row (d_v bf16) is the tile of `tiled_ld`, whose (thread,
    # value) layout is 32 lanes x K1_EPT elements, so `partition_S` of the
    # lane's slice turns one row into d_v/(32*EPT) 128-bit loads -- 2 per row
    # at d_v = 512, hence 4 `LDG.128` per row-pair (out + dout) per lane, in
    # place of the 32 two-byte 32-lane-strided loads the scalar walk issued.
    # The fragment shape comes from the partition of an identity tensor, so it
    # is the copy's own bijection and not a second hand-rolled one.
    thr = tiled_ld.get_slice(lane)
    frg = cute.make_layout(cute.shape(
        thr.partition_S(cute.make_identity_tensor((d_v,))))).shape
    rO = []
    rD = []
    for _u in cutlass.range_constexpr(unr):
        rO.append(_ALLOC(frg, cutlass.BFloat16))
        rD.append(_ALLOC(frg, cutlass.BFloat16))
    nel = cute.size(rO[0])

    # This warp owns one head per head group for the WHOLE block, so both the
    # sink scalar and the d_sink partial are register resident: nhg loads here
    # and nhg `red.global.add.f32` at the end, instead of one atomic per
    # (token, head).
    # `ds` is an rmem fragment, not a python list of SSA values: it is written
    # inside the DYNAMIC token loop, and only memory survives that region
    # boundary (a name rebound in the body would have to be a loop-carried
    # value, which a subscript store is not).  Its index is a compile-time
    # constant everywhere, so it stays in registers.
    ds = _ALLOC(cute.make_layout(nhg).shape, cutlass.Float32)
    sink2 = []
    for hg in cutlass.range_constexpr(nhg):
        sink2.append(mSink[hg * nwarp + w] * LOG2E)
        ds[hg] = cutlass.Float32(0.0)

    for i in cutlass.range(ttok):
        t = t0 + i
        for g0 in cutlass.range_constexpr(nhg // unr):
            # every row's loads are issued before any of them is consumed:
            # `unr` head groups x (out, dout) x d_v/(32*EPT) slices are in
            # flight at once, which is what keeps the DRAM pipe fed.
            for u in cutlass.range_constexpr(unr):
                hu = (g0 * unr + u) * nwarp + w
                cute.copy(tiled_ld, thr.partition_S(mOut[(t, hu, None)]), rO[u])
                cute.copy(tiled_ld, thr.partition_S(mDout[(t, hu, None)]),
                          rD[u])

            for u in cutlass.range_constexpr(unr):
                hg = g0 * unr + u
                hu = hg * nwarp + w

                # Delta = rowsum(O . dOut), fp32 accumulation (D0.4), over four
                # independent partials so the lane's product chain is not one
                # serial dependence.
                p = []
                for _k in cutlass.range_constexpr(4):
                    p.append(cutlass.Float32(0.0))
                for v in cutlass.range_constexpr(nel):
                    p[v % 4] = p[v % 4] + (rO[u][v].to(cutlass.Float32)
                                           * rD[u][v].to(cutlass.Float32))
                acc = (p[0] + p[1]) + (p[2] + p[3])

                # warp all-reduce, butterfly: every lane ends with the row sum
                for s in cutlass.range_constexpr(5):
                    acc = acc + _shfl_bfly(acc, 1 << s)
                delta = acc

                # lse2 = log2e*logaddexp(lse, sink) in the exp2 domain,
                # overflow-safe: m2 + log2(1 + exp2(-|a2 - sink2|))
                # (lse = +inf -> lse2 = +inf)
                a2 = mLse[(t, hu)] * LOG2E
                m2 = _fmax(a2, sink2[hg])
                dd = _fabs(a2 - sink2[hg])
                lse2 = m2 + _lg2(cutlass.Float32(1.0)
                                 + _ex2(cutlass.Float32(0.0) - dd))

                # d_sink[h] -= sigmoid(sink - lse)*Delta = exp2(sink2-lse2)*Delta
                ds[hg] = ds[hg] - _ex2(sink2[hg] - lse2) * delta

                if lane == 0:
                    mDelta[(t, hu)] = delta
                    mLse2[(t, hu)] = lse2

    if lane == 0:
        base = mDsink.iterator.toint()
        for hg in cutlass.range_constexpr(nhg):
            _red_add_f32(base + cutlass.Int64(hg * nwarp + w) * 4, ds[hg])


@cute.jit
def _k1_jit(mOut, mDout, mLse, mSink, mDelta, mLse2, mDsink, stream):
    tiled_ld = cute.make_tiled_copy_tv(
        cute.make_copy_atom(nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                            num_bits_per_copy=16 * _G_K1_EPT),
        cute.make_layout(32), cute.make_layout(_G_K1_EPT))
    _k1_kernel(mOut, mDout, mLse, mSink, mDelta, mLse2, mDsink,
               tiled_ld).launch(
        grid=[_G_K1_GX, 1, 1],
        block=[_G_K1_NWARP * 32, 1, 1],
        stream=stream)


# ---------------------------------------------------------------------------
# K2 — main kernel (D0.2, D0.5 - D0.13)
# ---------------------------------------------------------------------------

def _k2_storage(cq, ckv, cdo, cp, cds):
    """@cute.struct lives in a plain helper CALLED FROM the jit — the class may
    be defined outside a traced body but `cute.cosize` needs a live context
    (hw_kb/cutedsl/sm90/wgmma_mn_major_operand.md gotcha 1)."""
    @cute.struct
    class SS:
        sQ: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cq], 1024]
        sKV: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, ckv], 1024]
        sDO: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cdo], 1024]
        sP: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cp], 1024]
        sDS: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, cds], 1024]
        sSc: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 128], 1024]
        # two row-id slots: WG0 publishes block i+1's ids at the tail of body
        # i while WG1 is still reading block i's (D0.16 phase contract).
        sIx: cute.struct.Align[
            cute.struct.MemRange[cutlass.Int32, 2 * KVB], 1024]
    return SS


@cute.kernel
def _k2_kernel(mQ: cute.Tensor, mKV: cute.Tensor, mDO: cute.Tensor,
               mIdx: cute.Tensor, mTl: cute.Tensor,
               mLse2: cute.Tensor, mDelta: cute.Tensor,
               mDq: cute.Tensor, mWs: cute.Tensor,
               mma_s: cute.TiledMma, mma_dq: cute.TiledMma,
               mma_dqr: cute.TiledMma, mma_kv: cute.TiledMma,
               l_q: cute.ComposedLayout, l_kv: cute.ComposedLayout,
               l_do: cute.ComposedLayout, l_p: cute.ComposedLayout,
               l_ds: cute.ComposedLayout, l_qt: cute.ComposedLayout,
               l_kvt: cute.ComposedLayout, l_dot: cute.ComposedLayout,
               l_pt: cute.ComposedLayout, l_dst: cute.ComposedLayout,
               tiled_kv: cute.TiledCopy, tiled_zf: cute.TiledCopy,
               tiled_st: cute.TiledCopy,
               r2s_p: cute.TiledCopy, r2s_q: cute.TiledCopy,
               SS: cutlass.Constexpr[Callable]):
    s_kv = _G_K2_SKV
    cscale = _G_K2_CSCALE
    sscale = _G_K2_SCALE

    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    wg = cute.arch.make_warp_uniform(tidx // 128)
    t128 = tidx % 128
    h0 = by * H

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    sQ = st.sQ.get_tensor(l_q.outer, swizzle=l_q.inner)
    sKV = st.sKV.get_tensor(l_kv.outer, swizzle=l_kv.inner)
    sDO = st.sDO.get_tensor(l_do.outer, swizzle=l_do.inner)
    sP = st.sP.get_tensor(l_p.outer, swizzle=l_p.inner)
    sDS = st.sDS.get_tensor(l_ds.outer, swizzle=l_ds.inner)
    sSc = st.sSc.get_tensor(cute.make_layout(128))
    sIx = st.sIx.get_tensor(cute.make_layout(2 * KVB))

    # --- prologue: Q / dO resident tiles + the two row-scalar vectors (D0.5)
    # Both tiles are filled through the 128-bit slab atom (D0.9 write atom):
    # one `partition_S` / `partition_D` pair off the SAME slice, 16 LDG.128 +
    # STS.128 per thread per tile, in place of 256 two-byte load/store pairs.
    # It is the dq write-out's copy read backwards, so the (thread, element)
    # bijection is the one that pass already relies on.
    thr_fill = tiled_st.get_slice(tidx)
    gQ = cute.local_tile(mQ[(bx, None, None)], (H, DQK), (by, 0))
    gDO = cute.local_tile(mDO[(bx, None, None)], (H, DQK), (by, 0))
    cute.copy(tiled_st, thr_fill.partition_S(gQ), thr_fill.partition_D(sQ))
    cute.copy(tiled_st, thr_fill.partition_S(gDO), thr_fill.partition_D(sDO))
    if tidx < H:
        sSc[tidx] = mLse2[(bx, h0 + tidx)]
        sSc[H + tidx] = mDelta[(bx, h0 + tidx)]

    tl = mTl[bx]
    nblk = (tl + (KVB - 1)) // KVB

    # --- D0.18: the TAIL block is step 0, and the truncation is paid HERE ---
    # The CTA walks ceil(tl/64) KV blocks in the order  T, 0, 1, ... T-1  with
    # T = nblk - 1 the partial one, so the peeled first step is the only step
    # whose rows can sit past `tl`.  This prologue publishes exactly that step's
    # row ids, and it is the one place a per-row `pos < tl` test is evaluated:
    # rows at or past tl collapse to -1 and are zero filled in the peel.  Every
    # later step covers a FULL block, so its publish (body tail) carries no
    # per-row tl mask at all -- just one warp-uniform "does this step exist".
    #
    # Position is clamped by a select before the index load, so `idx[t, tl:]`
    # is never addressed (the load falls back to slot 0), and tl == 0 (nblk ==
    # 0, hence a negative position) takes the same path and gathers nothing.
    # Every later block's ids are published one body ahead into the other slot,
    # so the steady-state body carries no barrier of its own for them: the 0x9
    # handoff at body top is what makes them visible to all 128 lanes of WG0.
    if tidx < KVB:
        p0 = (nblk - 1) * KVB + tidx
        live0 = (p0 >= 0) & (p0 < tl)
        sf0 = cutlass.Int32(0)
        if live0:
            sf0 = p0
        ix0 = mIdx[(bx, sf0)]
        v0 = cutlass.Int32(-1)
        if live0 & (ix0 >= 0) & (ix0 < s_kv):
            v0 = ix0
        sIx[tidx] = v0

    # generic store -> async proxy read needs BOTH (wgmma_ss_min_gemm gotcha 1)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()                               # D0.14 barrier 0x0

    # transposed read views: second get_tensor on the SAME MemRange
    sQt = st.sQ.get_tensor(l_qt.outer, swizzle=l_qt.inner)
    sKVt = st.sKV.get_tensor(l_kvt.outer, swizzle=l_kvt.inner)
    sDOt = st.sDO.get_tensor(l_dot.outer, swizzle=l_dot.inner)
    sPt = st.sP.get_tensor(l_pt.outer, swizzle=l_pt.inner)
    sDSt = st.sDS.get_tensor(l_dst.outer, swizzle=l_dst.inner)

    wgm_s = mma_s.get_slice(0)
    wgm_q = mma_dq.get_slice(0)
    wgm_qr = mma_dqr.get_slice(0)
    wgm_k = mma_kv.get_slice(0)
    cS = mma_s.get_slice(t128).partition_C(cute.make_identity_tensor((H, KVB)))
    cQ = mma_dq.get_slice(t128).partition_C(cute.make_identity_tensor((H, DQN)))
    cK = mma_kv.get_slice(t128).partition_C(cute.make_identity_tensor((KVB, CH)))

    if wg == 0:
        # ------------------------------------------------------------------
        # WG0: KV gather, S / dP, softmax-bwd, P/dS publish, dq[:, 0:256]
        # ------------------------------------------------------------------
        accQ = [_ALLOC(cQ.shape, cutlass.Float32),
                _ALLOC(cQ.shape, cutlass.Float32)]
        for j in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(cute.size(accQ[j])):
                accQ[j][i] = cutlass.Float32(0.0)
        accS = _ALLOC(cS.shape, cutlass.Float32)
        accP = _ALLOC(cS.shape, cutlass.Float32)

        # --- the two row-broadcast scalars, hoisted into registers ----------
        # (hw_kb/cutedsl/sm90/wgmma_cfrag_publish_stsm.md mechanism B, m-indexed form.)
        # lse2 and Delta are broadcast along N, and for a 64xN warpgroup
        # accumulator a thread's registers cover exactly TWO rows -- register i
        # sits at m0 when (i % 4) < 2 and at m0 + 8 otherwise.  So the whole
        # broadcast is FOUR `LDS` here, outside the block loop, plus a
        # trace-time select on the register index, instead of one `LDS` per
        # element per block (which is what pins `smem_wavefronts_ld` on the
        # smem-gather floor: it cannot be merged by ptxas).
        m0_s, _n0_s = cS[0]
        m1_s, _n1_s = cS[2]
        lse0 = sSc[m0_s]
        lse1 = sSc[m1_s]
        del0 = sSc[H + m0_s]
        del1 = sSc[H + m1_s]

        # --- D0.9 publish path: ONE staging fragment, reused by P and by dS -
        # `partition_D` gives the STSM destination inside the face and `retile`
        # aliases the accumulator in the atom's register grouping; the staging
        # fragment is the bf16 buffer STSM reads from.  The accumulator is
        # converted straight INTO it rather than through an f32 copy of itself:
        # the f32 copy would be 32 more registers live on top of an accumulator
        # that is still needed (accS carries P into the dS expression), and this
        # kernel runs at 251 of 255 registers with `spill == 0` on the gate
        # line (D0.13 non-accumulator working set).  P and dS publish at
        # different points of the body, so one fragment serves both.
        thr_pub = r2s_p.get_slice(t128)
        tPub_P = thr_pub.partition_D(sP)
        tPub_DS = thr_pub.partition_D(sDS)
        tAccS = r2s_p.retile(accS)
        pub_l = cute.make_layout(cute.shape(thr_pub.partition_S(sP))[:3])
        pub16 = _ALLOC(pub_l.shape, cutlass.BFloat16)

        # --- D0.11 [SOURCE: RS]: dS leaves the accumulator ONCE, as bf16, and
        # that one register home has two readers ---------------------------
        # `dsRaw` is the bf16 pack of the dS C-frag, allocated in the
        # accumulator's own (thread, cell) order.  Two views alias it:
        #
        #   dsFrg  = the A operand of WG0's dQ chains, read straight out of
        #            registers by WGMMA.  A 64xN warpgroup C-frag and a 64xK
        #            A-fragment with K == N own the same cells per thread in
        #            the same linear register order (flat index i = 8*kb + j,
        #            j the ((2,2,2)) mode of k-block kb), so the A view is a
        #            pure layout alias -- no repack, no shuffle, no smem round
        #            trip.  The element TYPE is what makes it an alias and not
        #            a reinterpret: `cute.gemm` takes A as bf16, so the pack
        #            has to have happened before the view is taken (aliasing
        #            the f32 accumulator itself is rejected by the verifier).
        #   tRS_ds = the same registers in the StMatrix atom's grouping, which
        #            is what feeds the publish staging for WG1 (WG1 cannot
        #            reach these registers -- they are per warpgroup -- so its
        #            copy still goes through smem, D0.11's second dS row).
        #
        # So the softmax-bwd tail converts 32 values once and both consumers
        # read that result, instead of one conversion per consumer.
        dsRaw = _ALLOC(cS.shape, cutlass.BFloat16)
        tAcA = wgm_qr.partition_A(cute.make_identity_tensor((H, KVB)))
        dsFrg = cute.make_tensor(dsRaw.iterator, cute.make_layout(tAcA.shape))
        tRS_ds = r2s_p.retile(dsRaw)

        # gather partitioning (D0.15): the lane's slot inside a KV row is the
        # only thing the TiledCopy slices on; the team number only picks WHICH
        # rows this thread walks.  Both partition_S and partition_D come from
        # this ONE slice, so the two sides share a bijection by construction
        # (cp_async.md gotcha 6).
        slot_kv = t128 % KV_TPR                 # this lane's 16 B slot in a row
        thr_kv = tiled_kv.get_slice(slot_kv)
        team = t128 // KV_TPR

        # Zero fill for an invalid row (D0.18), one `STS.128` per lane per row
        # rather than KV_EPT scalar `STS.U16`.  The zero fragment is built ONCE
        # here, outside the block loop, and its shape is taken from the
        # destination partition rather than written by hand.  `tiled_zf` uses
        # the SAME (thread, value) layout as the gather, so the lane fills
        # exactly the 16 bytes its `cp.async` would have written: coverage of
        # the row (64 lanes x 8 elements, each once) is the gather's own
        # bijection, not a second hand-rolled one.
        thr_zf = tiled_zf.get_slice(slot_kv)
        zfrag = _ALLOC(cute.make_layout(
            cute.shape(thr_zf.partition_D(sKV[(0, None)]))).shape,
            cutlass.BFloat16)
        for i in cutlass.range_constexpr(cute.size(zfrag)):
            zfrag[i] = cutlass.BFloat16(0.0)

        for blk in cutlass.range(nblk):
            cur = (blk % 2) * KVB
            nxt = ((blk + 1) % 2) * KVB

            # (1/5) 0x9 -- WG1 publishes "KV slot + row-id slot free" once its
            # block i-1 dQ chains have retired; the prologue seeds one credit.
            # This is also the intra-WG0 fence that makes the row ids written
            # at the tail of body i-1 visible to all 128 lanes.
            cute.arch.barrier(barrier_id=BAR_KV, number_of_threads=K2_BLOCK)

            # --- gather one KV block into the single slot (D0.5 / D0.15) ----
            # `[GATHER: prefetched]`: the block's row ids were pulled in ONE
            # batch of index loads a whole body earlier and live in the smem
            # slab WG1 also reads, so a row's copy never waits on "its own"
            # index coming back from gmem.  Both sentinel families (-1 and
            # >= s_kv) and -- in the peeled step only -- the `pos >= tl` tail
            # were collapsed into the single value -1 there, so the burst below
            # carries one row-level predicate instead of three compares per
            # row, and no arithmetic on `tl` at all (D0.18).
            #
            # D1.3: the CONSUMPTION of those ids is batched too, and the batch
            # is held LANE-OWNED.  The form that lost was one `LDS` per row
            # inside the burst: every copy sat behind its own `LDS` -> compare
            # -> branch chain and the 32 chains ran strictly serially between
            # the 0x9 handoff and the first `LDGSTS` -- a SHORT scoreboard
            # wait, which `stall_long_sb` does not see.  The form that ALSO
            # lost is the warp-uniform register batch (32 `LDS` back to back at
            # the top, all 128 lanes reading the same 32 slots): ptxas does not
            # keep such a batch, it sinks each load back down to the branch
            # that consumes it, so the machine code is the serial form again
            # (`hw_kb/cutedsl/sm90/gather_index_lane_owned.md`, gotcha 1 -- batch depth
            # has no semantics).  What survives register allocation is one id
            # per LANE: lane L of every warp loads the team's row L (32
            # consecutive i32 slots = one conflict-free transaction per warp),
            # and the burst broadcasts row i out of lane i with
            # `shfl.sync.idx.b32`.  The team's smem round trip is then paid
            # ONCE per block instead of 32 times per thread, the working set
            # grows by ONE 32-bit register rather than KV_TRIPS of them (D0.13
            # segment A), and what stands in front of each `LDGSTS` is a
            # register shuffle instead of a memory load.  The team's rows are a
            # CONTIGUOUS segment (team*KV_TRIPS + i, not the interleaved
            # i*KV_TEAMS + team), which is what makes lane L's slot the team's
            # row L.
            r0 = team * KV_TRIPS
            my_id = sIx[cur + r0 + (t128 % 32)]
            for i in cutlass.range_constexpr(KV_TRIPS):
                r = r0 + i
                row = _shfl_idx_i32(my_id, i)
                if row >= 0:
                    cute.copy(tiled_kv,
                              thr_kv.partition_S(mKV[(row, None)]),
                              thr_kv.partition_D(sKV[(r, None)]))
                else:
                    # An invalid row is never dereferenced; its smem row is
                    # written 0, and S/dP/dq all see an exact zero through it
                    # (dq's contribution goes through `dS . kv` with kv == 0,
                    # and WG1 skips the row's reduction entirely).  After
                    # D0.18 the only rows that land here in a steady-state
                    # block are sentinels -- the `pos >= tl` tail is confined
                    # to the peeled first step.
                    cute.copy(tiled_zf, zfrag,
                              thr_zf.partition_D(sKV[(r, None)]))
            _cp_async_commit()
            _cp_async_wait(0)
            # cp.async retires per thread; the fence publishes the writes to
            # the async proxy the WGMMA descriptors read through, and the
            # barrier is what makes one thread's rows visible to the others.
            # (2/5) 0x8 is 128 wide: only WG0 gathers, so only WG0 has to meet
            # here -- WG1 is held off the same bytes by 0x9 instead.
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=BAR_GA,
                              number_of_threads=K2_BLOCK // 2)

            # --- S = Q . KV^T then dP = dO . KV^T, issued back to back ------
            tSB = wgm_s.make_fragment_B(wgm_s.partition_B(sKV))
            tSA = wgm_s.make_fragment_A(wgm_s.partition_A(sQ))
            warpgroup.fence()
            a0 = cute.make_mma_atom(mma_s.op)
            a0.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(cute.size(tSA.shape[2])):
                cute.gemm(a0, accS, tSA[None, None, kb], tSB[None, None, kb],
                          accS)
                a0.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            tPA = wgm_s.make_fragment_A(wgm_s.partition_A(sDO))
            warpgroup.fence()
            a1 = cute.make_mma_atom(mma_s.op)
            a1.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(cute.size(tPA.shape[2])):
                cute.gemm(a1, accP, tPA[None, None, kb], tSB[None, None, kb],
                          accP)
                a1.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            # exp2 / P-pack runs under the dP chain's flight (D0.12).  The pack
            # is in place: it terminates the live range of the S accumulator
            # (D0.13 segment B).
            warpgroup.wait_group(1)
            for i in cutlass.range_constexpr(cute.size(accS)):
                # trace-time select on the register index; a DSL `if` here
                # would be a runtime branch (that KB's gotcha 6)
                a = lse0 if (i % 4) < 2 else lse1
                accS[i] = _ex2(accS[i] * cscale - a)
            # pack P into the bf16 staging fragment while the dP chain is still
            # in flight -- only the STSM itself has to wait for the slot
            for v in cutlass.range_constexpr(cute.size(pub16)):
                pub16[v] = tAccS[v].to(cutlass.BFloat16)

            # (3/5) 0x3 -- P slot free (WG1 finished reading block i-1's P).
            cute.arch.barrier(barrier_id=BAR_PF, number_of_threads=K2_BLOCK)
            cute.copy(r2s_p, pub16, tPub_P)             # STSM.16.M88.4 x4
            # `arrive` only orders SHARED accesses, and only the ones written
            # before it (hw_kb/cutedsl/sm90/softmax_segment_ilp.md gotcha "STS before
            # arrive"): the stores above and the async-proxy fence must both
            # precede it, everything else may float.
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(barrier_id=BAR_P,
                                     number_of_threads=K2_BLOCK)

            # dS = sm_scale * P * (dP - Delta), in place, once the dP chain
            # lands.  The score the softmax sees is sm_scale * <q, kv>, so the
            # chain rule puts one sm_scale on the whole dS path (dq and the
            # dS^T.Q half of dkv) and NONE on the P^T.dO half -- folding it
            # into the dS pack pays it exactly once, here.
            warpgroup.wait_group(0)
            for i in cutlass.range_constexpr(cute.size(accP)):
                d = del0 if (i % 4) < 2 else del1
                accP[i] = sscale * accS[i] * (accP[i] - d)
            # the single bf16 pack of dS (D0.11): it terminates the dS
            # accumulator's live range and is what BOTH consumers read -- the
            # register A operand below and, through the atom-grouped view, the
            # staging fragment the STSM publish stores from.
            for i in cutlass.range_constexpr(cute.size(accP)):
                dsRaw[i] = accP[i].to(cutlass.BFloat16)
            for v in cutlass.range_constexpr(cute.size(pub16)):
                pub16[v] = tRS_ds[v]

            # (4/5) 0x4 -- dS slot free.
            cute.arch.barrier(barrier_id=BAR_DSF, number_of_threads=K2_BLOCK)
            cute.copy(r2s_p, pub16, tPub_DS)            # STSM.16.M88.4 x4
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(barrier_id=BAR_DS,
                                     number_of_threads=K2_BLOCK)

            # (5/5) 0x5 -- phase gate: WG1 has issued its first two chunk
            # pairs, so the two warpgroups' tensor use is staggered rather than
            # stacked.  Under the RS form the chains' A operand is a register
            # home, so this barrier no longer doubles as the fence that used to
            # guard the smem dS descriptor; the only smem operand left is B,
            # the KV tile, published by this warpgroup's own gather and already
            # fenced at 0x8.
            cute.arch.barrier(barrier_id=BAR_PH, number_of_threads=K2_BLOCK)

            # --- dq[:, 0:256] += dS . KV  (A from REGISTERS, D0.11 [SOURCE: RS])
            # A is `dsFrg`, the layout alias of the bf16 dS pack that is still
            # live in this warpgroup's registers; B stays the MN-major view of
            # the KV bytes.  The chain therefore reads no smem for A at all --
            # the values never make the round trip out to the dS tile and back
            # through a descriptor, which is the whole point of the ledger's RS
            # ruling.  The register home must not be overwritten before the
            # group retires (WGMMA reads register operands asynchronously): the
            # drain below is inside the same body, and the next write to it is
            # the next block's pack, well after that.
            for j in cutlass.range_constexpr(2):
                tQB = wgm_qr.make_fragment_B(wgm_qr.partition_B(
                    cute.local_tile(sKVt, (DQN, KVB), (j, 0))))
                warpgroup.fence()
                aq = cute.make_mma_atom(mma_dqr.op)
                aq.set(warpgroup.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(cute.size(dsFrg.shape[2])):
                    cute.gemm(aq, accQ[j], dsFrg[None, None, kb],
                              tQB[None, None, kb], accQ[j])
                warpgroup.commit_group()

            # --- publish the NEXT step's row ids into the OTHER slot --------
            # WG1 is still reading slot `cur`; the write lands in `nxt` and is
            # made visible by the 0x9 sync at the top of the next body, so it
            # costs no barrier of its own.
            #
            # D1.1: this segment is placed BETWEEN the dQ issue above and the
            # drain below, so its gmem index load and its `STS` run inside the
            # chains' flight instead of behind their drain.  It is the only
            # work in this body that is independent of both dQ operands (it
            # touches neither `accQ` nor the `dsRaw` register home, and it
            # writes the row-id slot WG1 is NOT reading), which is what makes
            # it legal to put under the chains -- and, per
            # hw_kb/cutedsl/sm90/wgmma_backedge_pending_group.md, keeping the drain on
            # THIS side of the back edge is what keeps the groups from rolling
            # into the next trip and serializing every HGMMA in the kernel.
            #
            # D0.18: step `blk + 1` runs physical block `blk`, which is a FULL
            # block whenever it exists -- its 64 positions are all < tl by
            # construction (blk <= nblk - 2  =>  blk*64 + 63 < (nblk-1)*64 <
            # tl), so no position is tested against tl here.  The only test is
            # `live1`, one warp-uniform "does that step exist"; when it is
            # false the slot is never read and the clamped load keeps the
            # address inside the top-k row, so `idx[t, tl:]` is not addressed.
            if t128 < KVB:
                live1 = (blk + 1) < nblk
                p1 = blk * KVB + t128
                sf1 = cutlass.Int32(0)
                if live1:
                    sf1 = p1
                ix1 = mIdx[(bx, sf1)]
                v1 = cutlass.Int32(-1)
                if live1 & (ix1 >= 0) & (ix1 < s_kv):
                    v1 = ix1
                sIx[nxt + t128] = v1

            # The dQ chains retire here, before the back edge: the next body's
            # gather overwrites the KV slot they read as B, and the loop head
            # must be reached with nothing in flight (a group carried across
            # the back edge makes the accumulator loop-carried, and ptxas then
            # serializes EVERY HGMMA in the function -- measured: `commit` axis
            # collapsing to `LE 0x0` x72 with spill still 0).
            warpgroup.wait_group(0)

        # --- WG1 is done with the resident Q / dO tiles; sQ becomes staging --
        # Each dQ chain's 64x128 C-frag is published into its own 128-column
        # WINDOW of the 64x512 slab through the same StMatrix path as P/dS
        # (8 `STSM.16.M88.4` per chain), not one scalar store per element.
        cute.arch.sync_threads()
        thr_dq = r2s_q.get_slice(t128)
        dq_l = cute.make_layout(cute.shape(thr_dq.partition_S(
            cute.local_tile(sQ, (H, DQN), (0, 0))))[:3])
        dq16 = _ALLOC(dq_l.shape, cutlass.BFloat16)
        for j in cutlass.range_constexpr(2):
            tAccQ = r2s_q.retile(accQ[j])
            for v in cutlass.range_constexpr(cute.size(dq16)):
                dq16[v] = tAccQ[v].to(cutlass.BFloat16)
            cute.copy(r2s_q, dq16, thr_dq.partition_D(
                cute.local_tile(sQ, (H, DQN), (0, j))))
    else:
        # ------------------------------------------------------------------
        # WG1: dKV chunks + dkv_ws reduction, dq[:, 256:512]
        # ------------------------------------------------------------------
        accQ = [_ALLOC(cQ.shape, cutlass.Float32),
                _ALLOC(cQ.shape, cutlass.Float32)]
        for j in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(cute.size(accQ[j])):
                accQ[j][i] = cutlass.Float32(0.0)
        accK = [_ALLOC(wgm_k.partition_shape_C((KVB, CH)), cutlass.Float32),
                _ALLOC(wgm_k.partition_shape_C((KVB, CH)), cutlass.Float32)]
        ws_base = mWs.iterator.toint()

        # --- D0.12 issue order: the dQ chains are NOT a tail block ----------
        # WG1's two dQ chains sit BETWEEN chunk pairs, and the KV handoff goes
        # out in mid body -- both positions are the frozen ones:
        #
        #   dQ chain j    after chunk pair DQ_AT[j]   (the 1st and 2nd pairs)
        #   0x9 "KV free" after the DEPBAR that follows chunk pair KV_AT
        #
        # The point is the commit-group FIFO.  A dQ chain committed after pair
        # c is OLDER than pair c+1, so the `wait_group(2)` that pair c+1 does
        # anyway -- the one whose only stated job is retiring pair c so its
        # dkv_ws reduction can go out -- retires the dQ chain as a side effect,
        # at no extra drain.  Two pairs later the chain is provably done and
        # the KV slot can be released while six of the eight chunks are still
        # to come.  Releasing it at the body tail instead (which is what the
        # first build did) makes 0x9 wait on the WHOLE of WG1's body, and then
        # WG0's gather + S + dP -- the half body that is supposed to hide under
        # WG1's chunks -- runs with WG1 parked on 0x1 and the tensor pipes idle.
        #
        # D1.4: both positions move two pairs EARLIER -- chains after pairs
        # 0 and 1, credit after pair 2 -- so that five chunk pairs, not three,
        # remain to cover WG0's next gather + S + dP.  The FIFO relation is
        # untouched (chain j is still committed after pair DQ_AT[j] and before
        # pair DQ_AT[j] + 1, and the credit still rests on a `wait_group(2)`
        # that has already drained both chains: chain 0 retires at pair 1's
        # wait, chain 1 at pair 2's, and the credit goes out after that same
        # wait).  Its operands are live this early: B is the KV tile, published
        # by WG0's gather before 0x1, and A is the dS tile, whose 0x2 handoff
        # is taken inside pair 0.
        NCH = DQK // CH                              # 8 dKV chunks per KV block
        KV_AT = min(2, NCH - 1)
        DQ_AT = (max(KV_AT - 2, 0), max(KV_AT - 1, 0))

        # --- dKV write-back address account (D0.19), hoisted ---------------
        # Flat register i of a 64x64 warpgroup accumulator sits at
        #     m = 16*(t/32) + (t%32)/4 + 8*((i/2)%2)
        #     n = 2*((t%32)%4) + (i%2) + 8*(i/4)
        # so a thread's 32 values live in exactly TWO rows, m0 and m0+8, and
        # in one row they are the 16 columns n0+b+8*k (b = i%2, k = i/4).
        # The workspace stores logical column n of a CH-wide chunk at slot
        #     PERM(n) = 4*((n%8)//2) + (n//8)%4 + 16*(n%2) + 32*((n//8)//4)
        # under which that lane's four registers 4 apart become 4 CONSECUTIVE
        # slots, so the chunk retires in 8 `red.global.add.v4.f32` rather than
        # 32 scalar reductions -- and, since n0 carries the WEIGHT-4 field
        # rather than the weight-8 one, the four lanes of a warp that share an
        # accumulator row land on four ADJACENT 16-byte granules instead of on
        # four granules 64 bytes apart.  That is the whole point of this
        # ordering rather than the naive one: the reduction is a read-modify-
        # write of 32-byte L2 sectors, and a warp instruction whose lanes tile
        # 64 contiguous bytes per row touches 16 sectors and fills every one of
        # them, where the strided arrangement touches 32 and half-writes each,
        # i.e. exactly twice the L2 reduction traffic for the same bytes.
        # Per thread the offsets step 0x40 inside a row, so the loop nest below
        # walks (row, kh, b) to issue them in that ascending order.
        # The row ids are per-thread block constants: two smem reads per KV
        # block instead of 32 per chunk.
        m0_k, n0_k = cK[0]
        m1_k, _n1_k = cK[2]

        # Seed one credit on each WG1 -> WG0 handoff so WG0's very first body
        # finds the KV / P / dS slots free.  This is the prologue half of the
        # ping-pong stagger (D0.16): WG1 itself does nothing until WG0 has
        # published P(0), so it enters the loop half a body behind.
        cute.arch.barrier_arrive(barrier_id=BAR_KV, number_of_threads=K2_BLOCK)
        cute.arch.barrier_arrive(barrier_id=BAR_PF, number_of_threads=K2_BLOCK)
        cute.arch.barrier_arrive(barrier_id=BAR_DSF, number_of_threads=K2_BLOCK)

        for blk in cutlass.range(nblk):
            cur = (blk % 2) * KVB

            # (1/2) 0x1 -- "P(i) is in smem".
            cute.arch.barrier(barrier_id=BAR_P, number_of_threads=K2_BLOCK)

            # the whole block's row-id account: WG0 published this slot at the
            # tail of its body i-1, ahead of the 0x1 arrive above.
            r0_k = sIx[cur + m0_k]
            r1_k = sIx[cur + m1_k]

            tKA_p = wgm_k.make_fragment_A(wgm_k.partition_A(sPt))
            tKA_d = wgm_k.make_fragment_A(wgm_k.partition_A(sDSt))
            # A of the dQ chains is the dS tile, K-major -- the same descriptor
            # for both chains, so it is built once outside the chunk walk.
            tQA = wgm_q.make_fragment_A(wgm_q.partition_A(sDS))
            for c in cutlass.range_constexpr(NCH):
                acc = accK[c % 2]
                tKB_o = wgm_k.make_fragment_B(wgm_k.partition_B(
                    cute.local_tile(sDOt, (CH, H), (c, 0))))
                warpgroup.fence()
                b0 = cute.make_mma_atom(mma_kv.op)
                b0.set(warpgroup.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tKA_p.shape[2])):
                    cute.gemm(b0, acc, tKA_p[None, None, kb],
                              tKB_o[None, None, kb], acc)
                    b0.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()

                # (2/2) 0x2 -- "dS(i) is in smem", waited on AFTER the first
                # chunk's P^T.dO chain has been issued: that chain only needs
                # P, so WG1 spends WG0's dS pack inside a WGMMA flight instead
                # of inside a barrier.
                if cutlass.const_expr(c == 0):
                    cute.arch.barrier(barrier_id=BAR_DS,
                                      number_of_threads=K2_BLOCK)

                tKB_q = wgm_k.make_fragment_B(wgm_k.partition_B(
                    cute.local_tile(sQt, (CH, H), (c, 0))))
                warpgroup.fence()
                b1 = cute.make_mma_atom(mma_kv.op)
                b1.set(warpgroup.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(cute.size(tKA_d.shape[2])):
                    cute.gemm(b1, acc, tKA_d[None, None, kb],
                              tKB_q[None, None, kb], acc)
                warpgroup.commit_group()

                # Retire chunk c-1 with chunk c's two chains still in flight.
                # In the two iterations that follow a dQ chain there are five
                # groups outstanding rather than four (pair c-1, the dQ chain,
                # pair c), and this same wait keeps the newest two -- so it
                # retires the dQ chain along with pair c-1, which is what makes
                # the KV release below free.
                warpgroup.wait_group(2)
                if cutlass.const_expr(c == 1):
                    # phase gate, released at the first wait that actually has
                    # two chunk pairs in flight -- WG0 may now start its dQ
                    # chains against this warpgroup's dKV chains.
                    cute.arch.barrier_arrive(barrier_id=BAR_PH,
                                             number_of_threads=K2_BLOCK)
                if cutlass.const_expr(c == KV_AT):
                    # 0x9 -- "KV slot + row-id slot free", mid body (D0.14 /
                    # D0.16: WG1 r17 middle, after the DEPBAR that retires its
                    # dQ chains).  Chain 1 was committed after pair KV_AT - 1
                    # and chain 0 one pair before that, so the wait above --
                    # which keeps only pair KV_AT's own two groups -- has
                    # drained both, and this warpgroup has no reader of the KV
                    # block left; chunks KV_AT..NCH-1 only touch P/dS/dO/Q,
                    # and after D1.4 that is FIVE of the eight pairs.  WG0's
                    # block i+1
                    # gather + S + dP therefore runs under them instead of
                    # after all of them, which is the whole point of the
                    # ping-pong: the single-buffered gather cannot be hidden
                    # inside WG0, only behind the other warpgroup.
                    cute.arch.barrier_arrive(barrier_id=BAR_KV,
                                             number_of_threads=K2_BLOCK)

                # --- dq[:, 256:512] += dS . KV, one chain per position ------
                # Issued here, between chunk pairs, and never drained on its
                # own: the next pair's `wait_group(2)` retires it (see above).
                # B is the MN-major view of the KV bytes this warpgroup is
                # about to release; A is the dS tile, whose slot stays held
                # until the body tail (0x4), so both operands are live here.
                #
                # D1.1: issued BEFORE the retired chunk's write-back, not
                # after.  Its commit-FIFO position is unchanged (after pair c,
                # before pair c+1), so the KV release above still rests on a
                # wait that has drained both chains; what changes is that the
                # write-back segment below -- an index-to-address chain and 8
                # reduction instructions, the `[GATHER: serial]` stall site --
                # now runs with this chain in flight instead of in front of it.
                for j in cutlass.range_constexpr(2):
                    if cutlass.const_expr(c == DQ_AT[j]):
                        tQB = wgm_q.make_fragment_B(wgm_q.partition_B(
                            cute.local_tile(sKVt, (DQN, KVB), (2 + j, 0))))
                        warpgroup.fence()
                        aq = cute.make_mma_atom(mma_dq.op)
                        aq.set(warpgroup.Field.ACCUMULATE, True)
                        for kb in cutlass.range_constexpr(
                                cute.size(tQA.shape[2])):
                            cute.gemm(aq, accQ[j], tQA[None, None, kb],
                                      tQB[None, None, kb], accQ[j])
                        warpgroup.commit_group()

                if cutlass.const_expr(c >= 1):
                    prev = accK[(c - 1) % 2]
                    for v1 in cutlass.range_constexpr(2):       # row +0 / +8
                        for kh in cutlass.range_constexpr(2):   # n-tiles
                            for b in cutlass.range_constexpr(2):    # col parity
                                reg = b + 2 * v1 + 16 * kh
                                row = r0_k if v1 == 0 else r1_k
                                if row >= 0:
                                    off = (row * DQK + (c - 1) * CH
                                           + 2 * n0_k + 16 * b + 32 * kh)
                                    _red_add_v4(
                                        ws_base + cutlass.Int64(off) * 4,
                                        prev[reg], prev[reg + 4],
                                        prev[reg + 8], prev[reg + 12])

            warpgroup.wait_group(0)

            # --- release the P / dS slots WG0 is about to overwrite ---------
            # Only these two go out here.  Every chunk reads P and dS, so the
            # last chunk's chains are their last reader and the drain above is
            # what makes the release legal; the KV slot went out at chunk
            # KV_AT, since nothing after it reads KV.
            #
            # D1.1: the arrives sit between that drain and the LAST chunk's
            # write-back rather than behind it.  The write-back reads only
            # accumulator registers and gmem -- it is not a reader of either
            # smem slot -- so hoisting the two credits over it hands WG0 its
            # next body's P/dS slots one reduction segment earlier, and that
            # segment then overlaps WG0's restart instead of delaying it.
            cute.arch.barrier_arrive(barrier_id=BAR_PF,
                                     number_of_threads=K2_BLOCK)
            cute.arch.barrier_arrive(barrier_id=BAR_DSF,
                                     number_of_threads=K2_BLOCK)

            last = accK[(NCH - 1) % 2]
            for v1 in cutlass.range_constexpr(2):
                for kh in cutlass.range_constexpr(2):
                    for b in cutlass.range_constexpr(2):
                        reg = b + 2 * v1 + 16 * kh
                        row = r0_k if v1 == 0 else r1_k
                        if row >= 0:
                            off = (row * DQK + (DQK - CH)
                                   + 2 * n0_k + 16 * b + 32 * kh)
                            _red_add_v4(ws_base + cutlass.Int64(off) * 4,
                                        last[reg], last[reg + 4],
                                        last[reg + 8], last[reg + 12])

        cute.arch.sync_threads()               # sQ / sDO retired -> staging
        thr_dq = r2s_q.get_slice(t128)
        dq_l = cute.make_layout(cute.shape(thr_dq.partition_S(
            cute.local_tile(sQ, (H, DQN), (0, 2))))[:3])
        dq16 = _ALLOC(dq_l.shape, cutlass.BFloat16)
        for j in cutlass.range_constexpr(2):
            tAccQ = r2s_q.retile(accQ[j])
            for v in cutlass.range_constexpr(cute.size(dq16)):
                dq16[v] = tAccQ[v].to(cutlass.BFloat16)
            cute.copy(r2s_q, dq16, thr_dq.partition_D(
                cute.local_tile(sQ, (H, DQN), (0, 2 + j))))

    # --- dq epilogue (D0.19): one 128-bit-per-lane pass over the staged tile
    # Both warpgroups have packed their f32 accumulators into the dead Q tile
    # above; this moves the whole 64x512 bf16 slab out in 16 vector stores per
    # thread instead of 128 two-byte ones.  The TiledCopy's thread layout gives
    # one warp 256 consecutive columns of one row (512 contiguous gmem bytes)
    # and each lane an 8-element = 16-byte granule, which the 128B swizzle
    # keeps contiguous in smem, so `partition_S` and `partition_D` are the same
    # bijection on both sides.  Every element of the slab is written exactly
    # once, tl == 0 rows included (their accumulators are the zeros they were
    # initialised to).
    cute.arch.sync_threads()
    gDq = cute.local_tile(mDq[(bx, None, None)], (H, DQK), (by, 0))
    thr_st = tiled_st.get_slice(tidx)
    cute.copy(tiled_st, thr_st.partition_S(sQ), thr_st.partition_D(gDq))


@cute.jit
def _k2_jit(mQ, mKV, mDO, mIdx, mTl, mLse2, mDelta, mDq, mWs, stream):
    bf = cutlass.BFloat16
    f32 = cutlass.Float32
    KM = warpgroup.OperandMajorMode.K
    MN = warpgroup.OperandMajorMode.MN

    # swizzle tier follows the CONTIGUOUS mode width: 512 bf16 and 64 bf16
    # both land on 128B swizzle.
    atomD = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf, DQK), bf)
    atomK = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf, KVB), bf)

    l_q = cute.tile_to_shape(atomD, (H, DQK), order=(0, 1))
    l_kv = cute.tile_to_shape(atomD, (KVB, DQK), order=(0, 1))
    l_do = cute.tile_to_shape(atomD, (H, DQK), order=(0, 1))
    l_p = cute.tile_to_shape(atomK, (H, KVB), order=(0, 1))
    l_ds = cute.tile_to_shape(atomK, (H, KVB), order=(0, 1))

    # transposers: shape = READ view, stride = colex coefficients of the WRITE
    # view's domain, reordered to the read view's order.
    l_qt = cute.composition(l_q, cute.make_layout((DQK, H), stride=(H, 1)))
    l_kvt = cute.composition(l_kv, cute.make_layout((DQK, KVB), stride=(KVB, 1)))
    l_dot = cute.composition(l_do, cute.make_layout((DQK, H), stride=(H, 1)))
    l_pt = cute.composition(l_p, cute.make_layout((KVB, H), stride=(H, 1)))
    l_dst = cute.composition(l_ds, cute.make_layout((KVB, H), stride=(H, 1)))

    mma_s = sm90_utils.make_trivial_tiled_mma(
        bf, bf, KM, KM, f32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, KVB))
    mma_dq = sm90_utils.make_trivial_tiled_mma(
        bf, bf, KM, MN, f32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, DQN))
    # RS form: same geometry as mma_dq but A comes from registers, not from a
    # smem descriptor.  WG0's dQ chains take it, because their A operand is the
    # dS pack that warpgroup just produced and still holds (D0.11
    # [SOURCE: RS]).  WG1's dQ chain keeps the SS form: it did not produce dS,
    # and registers are per warpgroup, so its only path to those values is the
    # smem tile.
    mma_dqr = _make_rs_mma(KM, MN, DQN)
    mma_kv = sm90_utils.make_trivial_tiled_mma(
        bf, bf, MN, MN, f32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, CH))

    # D0.15 / D0.17: the KV gather atom.  cp.async is the only G2S path that
    # takes a RUNTIME row id (the address is computed in registers), which is
    # what a top-k gather needs and what no TMA descriptor can express.
    # `cache_mode=GLOBAL` is `.cg` -> `LDGSTS.E.BYPASS` (D0.17): with ~214 KB
    # of the 256 KB L1/smem array handed to smem there is almost no L1 left
    # for `.ca` to allocate a line in, so every copy queues on that path;
    # bypassing straight to L2 is a latency account, not a bandwidth one.
    kv_op = cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)
    tiled_kv = cute.make_tiled_copy_tv(
        cute.make_copy_atom(kv_op, bf, num_bits_per_copy=128),
        cute.make_layout(KV_TPR), cute.make_layout(KV_EPT))

    # D0.18 zero fill: the SAME (thread, value) layout as the gather, with a
    # 128-bit universal atom instead of the async one, so an invalid row costs
    # one `STS.128` per lane instead of KV_EPT scalar stores and lands on
    # exactly the bytes that lane's `cp.async` would have filled.
    tiled_zf = cute.make_tiled_copy_tv(
        cute.make_copy_atom(nvgpu.CopyUniversalOp(), bf,
                            num_bits_per_copy=128),
        cute.make_layout(KV_TPR), cute.make_layout(KV_EPT))

    # D0.9 / D0.19 slab atom: 8 rows x 32 lane-groups of 8 bf16.  One warp
    # covers 256 consecutive columns of one row = 512 contiguous gmem bytes,
    # and 8 bf16 is one 16-byte granule, which the 128B swizzle keeps
    # contiguous in smem.  Tiled over (64, 512) that is 16 vector copies per
    # thread for a whole 64x512 tile.  Used in BOTH directions: gmem -> smem
    # for the resident Q / dO fill (LDG.128 + STS.128 per lane, in place of a
    # 2-byte LDG/STS pair per element) and smem -> gmem for the dq write-out.
    tiled_st = cute.make_tiled_copy_tv(
        cute.make_copy_atom(nvgpu.CopyUniversalOp(), bf,
                            num_bits_per_copy=128),
        cute.make_layout((8, 32), stride=(32, 1)),
        cute.make_layout((1, 8), stride=(8, 1)))

    # D0.9 publish atoms (hw_kb/cutedsl/sm90/wgmma_cfrag_publish_stsm.md, mechanism A):
    # a warpgroup's fp32 C-frag leaves for smem as bf16 through StMatrix --
    # `STSM.16.M88.4`, one per 8 bf16 per thread, so 4 per 64x64 face (P, dS)
    # and 8 per 64x128 dQ window -- instead of one scalar `STS.U16` per element.
    # ONE LayoutEnum drives both call sites (that KB's gotcha 9):
    # `get_smem_store_op` emits the instruction and `StMatrix8x8x16bOp(
    # is_m_major, 4)` groups the registers; changing only one leaves the SASS
    # unchanged or makes `cute.copy` reject the value count.  ROW_MAJOR is the
    # same enum the faces above were allocated with (N contiguous, order=(0,1)),
    # so the non-transposed member is the right one and every MN-major consumer
    # keeps reading through its transposed READ view -- the bytes STSM lands are
    # the bytes the scalar stores landed, cell for cell.
    E = LayoutEnum.ROW_MAJOR
    store_op = sm90_utils.get_smem_store_op(E, elem_ty_d=bf, elem_ty_acc=f32)
    atom_c = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), bf)
    r2s_p = cute.make_tiled_copy_S(
        store_op, cute.make_tiled_copy_C_atom(atom_c, mma_s))
    r2s_q = cute.make_tiled_copy_S(
        store_op, cute.make_tiled_copy_C_atom(atom_c, mma_dq))

    SS = _k2_storage(cute.cosize(l_q), cute.cosize(l_kv), cute.cosize(l_do),
                     cute.cosize(l_p), cute.cosize(l_ds))
    _k2_kernel(mQ, mKV, mDO, mIdx, mTl, mLse2, mDelta, mDq, mWs,
               mma_s, mma_dq, mma_dqr, mma_kv,
               l_q, l_kv, l_do, l_p, l_ds, l_qt, l_kvt, l_dot, l_pt, l_dst,
               tiled_kv, tiled_zf, tiled_st, r2s_p, r2s_q,
               SS).launch(grid=[_G_K2_GX, _G_K2_GY, 1],
                          block=[K2_BLOCK, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# K3 — cast: dkv_ws f32 -> dkv bf16   (D0.3 / D0.4)
# ---------------------------------------------------------------------------

@cute.kernel
def _k3_kernel(mWs, mDkv):
    d_qk = _G_K3_DQK
    block = _G_K3_BLOCK
    ch = _G_K3_CH

    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    for k in cutlass.range_constexpr(d_qk // block):
        c = k * block + tidx
        # undo K2's workspace column order (D0.3 leaves the order to the
        # student; D0.19 picks the one that makes the reduction vectorisable
        # AND sector-filling): inside one `ch`-wide chunk, logical column n
        # lives at slot PERM(n) = 4*((n%8)//2) + (n//8)%4 + 16*(n%2) +
        # 32*((n//8)//4).  This walk is indexed by the LOGICAL column, so it
        # applies the same forward map K2 wrote through -- the map is a
        # bijection but not an involution, so the expression here has to be
        # PERM itself, not its inverse read backwards.  `ch == 1` is the
        # identity fallback for a d the chunking does not divide (K2 itself
        # only accepts d == 512), and PERM(0) == 0 keeps it exact.
        n = c % ch
        s = c - n + (4 * ((n % 8) // 2) + (n // 8) % 4
                     + 16 * (n % 2) + 32 * ((n // 8) // 4))
        mDkv[(bx, c)] = mWs[(bx, s)].to(cutlass.BFloat16)


@cute.jit
def _k3_jit(mWs, mDkv, stream):
    _k3_kernel(mWs, mDkv).launch(
        grid=[_G_K3_GX, 1, 1], block=[_G_K3_BLOCK, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# host side: JIT cache + launch
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_CACHE = {}


def _stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        from cuda.bindings import driver as _cd
        return _cd.CUstream(raw)
    except Exception:
        from cuda import cuda as _cd
        return _cd.CUstream(raw)


def _t(x):
    return from_dlpack(x, assumed_align=16)


def _compiled(key, jit_fn, args, setup):
    """Compile once per key (tracing under `setup`'s globals), then return the
    compiled object.  `args` is the FULL and ONLY argument list — it carries no
    Constexpr entry, so compile and call take exactly the same items."""
    fn = _CACHE.get(key)
    if fn is None:
        with _LOCK:
            fn = _CACHE.get(key)
            if fn is None:
                setup()
                fn = cute.compile(jit_fn, *args)
                _CACHE[key] = fn
    return fn


def _fit(want, n):
    """Largest `v <= want` that divides `n`, walking down by halves.

    Every shape in the delivery list is a power of two, so the halving walk is
    exact there and `_fit` is the identity at shape A (128 heads -> 8 warps,
    512 d -> 256 threads, 32768 tokens -> 32 per block).  It exists so that no
    kernel can be handed a degenerate grid at the reduced court point or in a
    probe."""
    v = min(want, n)
    while v > 1 and n % v != 0:
        v //= 2
    return v


def _k1_args(out, dout, lse, attn_sink, delta, lse2, d_sink, stream):
    return (_t(out), _t(dout), _t(lse), _t(attn_sink),
            _t(delta), _t(lse2), _t(d_sink), stream)


def _k1_setup(s_q, h_q, d_v):
    nw = _fit(K1_NWARP, h_q)
    tt = _fit(K1_TTOK, s_q)
    nhg = h_q // nw
    _setg(_G_K1_NWARP=nw, _G_K1_DV=d_v, _G_K1_TTOK=tt, _G_K1_NHG=nhg,
          _G_K1_UNR=_fit(K1_UNR, nhg), _G_K1_EPT=_fit(K1_EPT, d_v // 32),
          _G_K1_GX=s_q // tt)


def _k2_args(q, kv, dout, idx, tl, lse2, delta, dq, ws, stream):
    return (_t(q), _t(kv), _t(dout), _t(idx), _t(tl),
            _t(lse2), _t(delta), _t(dq), _t(ws), stream)


def _k2_setup(s_q, h_q, s_kv, sm_scale):
    _setg(_G_K2_SKV=s_kv, _G_K2_CSCALE=float(sm_scale) * LOG2E,
          _G_K2_SCALE=float(sm_scale),
          _G_K2_GX=s_q, _G_K2_GY=h_q // H)


def _k3_setup(s_kv, d_qk):
    _setg(_G_K3_DQK=d_qk, _G_K3_BLOCK=_fit(K3_BLOCK, d_qk),
          _G_K3_CH=(CH if d_qk % CH == 0 else 1), _G_K3_GX=s_kv)


def _run_k1(out, dout, lse, attn_sink, delta, lse2, d_sink, stream=None):
    s_q, h_q, d_v = out.shape
    assert d_v % 32 == 0                       # one warp walks a row in D/32 steps
    stream = _stream() if stream is None else stream
    args = _k1_args(out, dout, lse, attn_sink, delta, lse2, d_sink, stream)
    fn = _compiled(("k1", s_q, h_q, d_v), _k1_jit, args,
                   lambda: _k1_setup(s_q, h_q, d_v))
    fn(*args)


def _run_k2(q, kv, dout, idx, tl, lse2, delta, dq, ws, sm_scale, stream=None):
    s_q, h_q, d_qk = q.shape
    s_kv, topk = kv.shape[0], idx.shape[1]
    assert d_qk == DQK and h_q % H == 0 and topk % KVB == 0
    stream = _stream() if stream is None else stream
    args = _k2_args(q, kv, dout, idx, tl, lse2, delta, dq, ws, stream)
    fn = _compiled(("k2", s_q, h_q, d_qk, s_kv, topk), _k2_jit, args,
                   lambda: _k2_setup(s_q, h_q, s_kv, sm_scale))
    fn(*args)


def _run_k3(dkv_ws, dkv, stream=None):
    s_kv, d_qk = dkv_ws.shape
    stream = _stream() if stream is None else stream
    args = (_t(dkv_ws), _t(dkv), stream)
    fn = _compiled(("k3", s_kv, d_qk), _k3_jit, args,
                   lambda: _k3_setup(s_kv, d_qk))
    fn(*args)


def precompile(s_q, h_q, d_qk, d_v, s_kv, topk, sm_scale, device="cuda"):
    """Lower every kernel in the delivery list at a given shape without
    launching anything (compile-only contract, hw_kb gpu_probe_ladder §7)."""
    f = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
    b = lambda *s: torch.zeros(*s, device=device, dtype=torch.bfloat16)
    st = _stream()
    # keep the torch buffers alive for the whole tracing pass
    keep = dict(q=b(s_q, h_q, d_qk), kv=b(s_kv, d_qk),
                o=b(s_q, h_q, d_v), do=b(s_q, h_q, d_v), lse=f(s_q, h_q),
                sink=f(h_q), delta=f(s_q, h_q), lse2=f(s_q, h_q),
                dsink=f(h_q), dq=b(s_q, h_q, d_qk), ws=f(s_kv, d_qk),
                dkv=b(s_kv, d_qk),
                idx=torch.zeros(s_q, topk, device=device, dtype=torch.int32),
                tl=torch.zeros(s_q, device=device, dtype=torch.int32))
    _compiled(("k1", s_q, h_q, d_v), _k1_jit,
              _k1_args(keep["o"], keep["do"], keep["lse"], keep["sink"],
                       keep["delta"], keep["lse2"], keep["dsink"], st),
              lambda: _k1_setup(s_q, h_q, d_v))
    _compiled(("k2", s_q, h_q, d_qk, s_kv, topk), _k2_jit,
              _k2_args(keep["q"], keep["kv"], keep["do"], keep["idx"],
                       keep["tl"], keep["lse2"], keep["delta"], keep["dq"],
                       keep["ws"], st),
              lambda: _k2_setup(s_q, h_q, s_kv, sm_scale))
    _compiled(("k3", s_kv, d_qk), _k3_jit,
              (_t(keep["ws"]), _t(keep["dkv"]), st),
              lambda: _k3_setup(s_kv, d_qk))
    return keep


def dsa_sparse_bwd(q, kv, out, dout, lse, attn_sink, topk_idxs, sm_scale,
                   topk_length=None):
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    dev = q.device
    stream = _stream()

    # --- outputs + intermediate tensor ABI (D0.3) --------------------------
    dq = torch.empty((s_q, h_q, d_qk), device=dev, dtype=torch.bfloat16)
    dkv = torch.empty((s_kv, d_qk), device=dev, dtype=torch.bfloat16)
    delta = torch.empty((s_q, h_q), device=dev, dtype=torch.float32)
    lse2 = torch.empty((s_q, h_q), device=dev, dtype=torch.float32)
    d_sink = torch.zeros(h_q, device=dev, dtype=torch.float32)          # K0a
    dkv_ws = torch.zeros((s_kv, d_qk), device=dev, dtype=torch.float32)  # K0b
    if topk_length is None:
        topk_length = torch.full((s_q,), topk_idxs.shape[1], device=dev,
                                 dtype=torch.int32)

    # --- K1 / K2 / K3, stream-ordered (D0.1) -------------------------------
    _run_k1(out, dout, lse, attn_sink, delta, lse2, d_sink, stream)
    _run_k2(q, kv, dout, topk_idxs, topk_length, lse2, delta, dq, dkv_ws,
            sm_scale, stream)
    _run_k3(dkv_ws, dkv, stream)

    return dq, dkv, d_sink
