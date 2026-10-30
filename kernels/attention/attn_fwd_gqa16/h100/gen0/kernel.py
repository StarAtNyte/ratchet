"""attn_fwd kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):
  attn_fwd(q, k, v) -> o
    q: torch.Tensor (B, H_Q, S, D) cuda bfloat16
    k: torch.Tensor (B, H_KV, S, D) cuda bfloat16
    v: torch.Tensor (B, H_KV, S, D) cuda bfloat16
    o: torch.Tensor (B, H_Q, S, D) cuda bfloat16
  Causal masking, softmax scale = 1/sqrt(D), GQA mapping
  h_kv = h_q // (H_Q // H_KV), fp32 softmax/accumulation inside.

M0 architecture build, assembled in the milestone's construction order; each
step is smoke-tested at the real shape before the next mechanism goes on:
  1 host skeleton + TMA descriptor construction + role branch            [done]
  2 TMA loads + mbarrier pipeline (D0.8 / D0.9), consumer on stub        [done]
  3 both MMA chains + RS packing (D0.3 / D0.6)                           [done]
  4 softmax + mask + online merge + deferred rescale
    (D0.5 / D0.12 / D0.13)                                               [done]
  5 epilogue: STSM -> aliased smem -> TMA store (D0.14)                   [done]
  6 issue order / release points and ping-pong signalling (D0.10, D0.11) [done]

Every runtime-control-flow helper below carries its own `@cute.jit`: the CuTe
preprocessor rewrites only the body of the function it decorates and does not
descend into callees, so a `range_constexpr` loop or a runtime `if` in an
undecorated helper never gets staged and dies at trace time.

The math this file mounts:

  D0.6  both WGMMA chains, and the P operand travels in REGISTERS.  chain0
        S = Q.K^T is SS (Q and K read by smem descriptor, no LDS/LDSM); chain1
        O += P.V is RS -- chain0's fp32 accumulator is packed in place to bf16
        (cvt.f32x2.bf16x2, 32 F2FP) and handed to chain1 as its A operand, so P
        never makes a smem round trip.  V is consumed through a second, MN-major
        view of the very same V MemRange (no transposed copy, no extra smem).
  D0.5  bf16 in / fp32 accumulate; m, l and the online rescale all fp32; the
        exponential is exp2 with log2(e)/sqrt(D) pre-folded into ONE constant,
        so each element pays a single FFMA into exp2 (the row's max*scale is
        hoisted out of the element loop).
  D0.12 one single K-tile loop with an unconditional causal predicate per
        element (no masked/unmasked loop variants); the row coordinate is
        decoded from the D0.2 packed row (row % 16 = head, row // 16 = seq).
  D0.13 deferred rescale: chain1 is issued against the UNCORRECTED O
        accumulator (PV lags S by one K-tile), and O is fixed up with 64 FMUL
        only after the group drains.  Row max / row sum are merged once per
        iteration, inside the loop, with a 2-step warp butterfly over the quad
        of lanes that share a row.

The two consumer warpgroups split the 128-row M-tile 64/64 through
atom_layout_mnk=(2,1,1), so each of them carries one 64x128 chain0 accumulator
(64 regs), one 64x128 chain1 accumulator (64 regs) and the 64-element bf16 P
fragment (32 regs).

  D0.14 the epilogue: the final 1/l rescale is folded into the same multiply
        that feeds the bf16 conversion (no fp32 staging block), the bf16
        fragment IS the StMatrix8x8x16b x4 source, and it is stored into the
        dead Q buffer (D0.7 alias) and flown out by a TMA S2G through the
        mirror image of the Q descriptor.  Q.EMPTY is released at that reuse
        point, one instruction before the first byte of Q is overwritten.

M1 levers, adjudicated by A/B on the device window and left at the arm that
measured faster:
  D1.2 mbarrier wait encoding = the BLOCKING face (consumer_wait /
       producer_acquire) at all 8 wait sites.  The non-blocking spin arm
       (consumer_try_wait + tight retry loop, producer_acquire fed a
       pre-satisfied token) was mounted and measured at 0.2671 ms device
       against 0.2539 ms for this arm, with tensor_pct 62.14 vs 65.90 and
       stall_long_sb 1.34 vs 0.80 -- the resident spin buys nothing at
       1 CTA/SM and costs issue slots the consumers' elementwise segment
       needs, so the blocking encoding stands.
  D1.3 producer issue organisation = ONE warp issues every TMA (Q, then K and
       V interleaved in a single loop), warp 0 keeping the D0.14 O store.
       The 4-warp split (one supply pipeline per warp) was measured in the
       same bundle and is on the losing side of the same A/B; fork F2 keeps
       its originally selected edge.

M2 register economy (companion group, one commit):
  D2.1 the integer bookkeeping chain is seeded from blockIdx + constants only
       and marked warp-uniform once at its entry.
  D2.2 every container (acc_s, acc_o, tOrP, rO, the row-statistic fragments)
       is allocated once outside the K-tile loop and overwritten per tile, and
       the F2FP packing is in place -- nothing in the loop body is rebound.
  D2.3 in-flight width W = EW_W = 4 on both reductions of the elementwise
       segment (see EW_W).
  D2.4 the segment is already at its short form: MN views of the C fragments
       are pure layout compositions, the row scalars are hoisted out of the
       element loop and broadcast by SHFL rather than reloaded from smem, the
       exponential is one FFMA + one fastmath exp2, and the packed bf16 P
       feeds chain1 directly as an RS operand.
  D2.5 the WGMMA descriptors are seeded from wg_slot, a warp-uniform slice
       coordinate, so make_fragment_A/B produce descriptors on the uniform
       path; the mark is on the coordinate, not on the descriptor.

Construction step 6, the last piece of the {D0.10, D0.11, D0.13} companion
group, is now mounted:
  D0.11 the two consumer warpgroups run an asymmetric one-way ping-pong on two
        static inbox ids (PP_BARRIER_ID + w, 256 participants).  A warpgroup
        blocks only on its OWN inbox (`arrive_and_wait`) and signals only the
        PEER's (non-blocking arrive), so the two element-wise segments are
        mutually excluded rather than symmetrically met; a single priming
        arrive from WG1 before the steady loop makes WG0 the leader, and the
        peeled first beat supplies the offset.  Blocking rendezvous per
        consumer warpgroup per trip = 1 (the baton), plus the loop-tail and
        epilogue meets once each.
  D0.10 the steady-state trip now follows the tabled order step for step; the
        row that moved is 9, `arrive(K.EMPTY[p])`, from immediately after
        wait_group(1) to after the element-wise segment it is specified to
        follow.  The deferred-wait relation itself (wait_group(1), softmax
        inside chain1's flight, V.EMPTY after wait_group(0)) was already in
        place from step 4.

M2 Gate (closing row, submitted alone; no decision bundled, no executable
change -- the AST below is byte-identical to commit c2044a0, the tree the
driver last evaluated in full, so every reading here is taken on these exact
kernel bytes).
  Roster: D2.1, D2.2, D2.3, D2.4, D2.5 -- the whole companion group, mounted
  as one commit at iter 8 and re-mounted unchanged at iter 13 on top of the
  D1.2/D1.3 revert; D2.3's W is EW_W = 4, D2.2/D2.4/D2.5 required no edit
  beyond the marks already described above.
  Lines, read against the iter-14 evaluation of these bytes:
    correctness                                          PASS
    spill_ld_sectors = spill_st_sectors = 0              0 / 0        PASS
    tensor_pct >= 55                                     66.32        PASS
    smem_wavefronts_ld <= 26214                          0            PASS
    device median <= 0.32 ms                             0.2528 ms    PASS
  Companion-group settlement (the group's performance account is cleared once,
  here, per the gate's own clause): device median at the M1 gate was 0.2745 ms;
  the iter-8 group commit read 0.2873 ms, a declared in-group regression that
  the ledger carried rather than adjudicated; the epilogue re-mount at iter 9
  read 0.2803 ms with smem_wavefronts_st 265346 -> 262144 and bankconf_st
  3202 -> 0; D1.4 at iter 10 read 0.2543 ms; the iter-13 re-mount of the group
  on the reverted M1 arms read 0.2531 ms and iter 14 reads 0.2528 ms.  The
  group closes at 0.2528 ms, below the 0.32 ms line by 21%, with spill at 0
  and regs_per_thread 168 against the 240-register consumer quota.
  Deferred lines carried out of M2: none.
  Budget/knobs frozen at this gate, unchanged from M0: 164864 B dynamic smem,
  0 B static, 384 threads / 3 warpgroups, register split 24/240, tile
  128x128x128, grid 1024, KV_STAGE = 2, wg_wait depth N = 1, wait encoding =
  blocking, producer issue organisation = single warp, D1.4 heavy-first
  decode, EW_W = 4.
  Next phase: M3 (D3.1 epilogue write-side swizzle / STSM member, D3.2 TMA box
  swizzle width).

M3 exchange surface, fork F1 adjudicated (D3.1 and D3.2 are the two sides of
activated fork F1 and are coupled: changing one side alone computes the wrong
number, so they move as a set through the single SMEM_CONTIG knob):
  Arm swap executed and measured.  The alternate arm SW64 (SMEM_CONTIG = 32,
  32 contiguous bf16 elements = 64 B per swizzle period) was shipped at iter 16
  across all four coupled call sites -- the Q and K/V gmem->smem TMA
  descriptors, the SS WGMMA descriptors chain0 reads Q and K through, the
  MN-major composed view chain1 reads V through, and the STSM destination /
  O TMA store source in the epilogue.  This commit moves SMEM_CONTIG back to
  128, restoring the SW128 incumbent on the same four call sites.  Box shapes
  never moved in either direction: Q / O stay (GQA, S_PER_TILE, D) and K / V
  stay (BN, D), i.e. the D0.8 split is frozen.  The StMatrix member never moved
  either (StMatrix8x8x16b x4, non-transposed, selected by
  LayoutEnum.ROW_MAJOR.is_m_major_c()); the C-fragment tiling and the r2s
  tiled copy are rebuilt on whichever atom the knob selects, by construction.
  F1 ledger -- two arms, both correct at the real shape:
    SW128 (iter 15): device 0.2527 ms, time_ns 256789, tensor_pct 66.36,
      bankconf_st/ld 0/0, smem_wavefronts_st/ld 262144/0,
      dram_bytes_write 17446400.
    SW64  (iter 16): device 0.2542 ms, time_ns 259883, tensor_pct 65.78,
      bankconf_st/ld 0/0, smem_wavefronts_st/ld 262144/0,
      dram_bytes_write 18807467.
  The device-window gap is 0.594%, i.e. INSIDE the resolution band b = 0.009,
  so under the delivery discipline it carries no evidentiary weight on which
  arm is faster and may not by itself settle the fork.  The fork is therefore
  settled on the co-adjudication quantities, which come off the same NCU metric
  table and carry no host tax or cross-job wall-clock drift: time_ns 256789 vs
  259883 (1.20% apart, outside the band, SW128 lower -- and SW128's worst of
  three reads, 256789, is still below SW64's only read) and tensor_pct 66.36 vs
  65.78 (SW128 higher).  Both point the same way.  Secondary mechanism reading
  in the same direction: dram_bytes_write rises 7.80% under SW64.
  Verdict: original arm MAINTAINED, the SW64 alternate is falsified; F1 closes
  on SW128 and the exchange surface is restored to it.
  Why the fork had nothing to buy: both of D3.1's trigger conditions are absent
  on the incumbent -- the conflict form (bankconf_st > 0) reads 0, and the
  volume form reads smem_wavefronts_st 262144 / smem_wavefronts_ld 0, which is
  exact parity with the reference signature, so no swizzle-width or STSM-member
  arm can improve either metric; it can only tie or lose.  The remaining
  registered arm (no swizzle) is dominated on that same ground: removing the
  swizzle can add bank conflicts, never remove conflicts that are already 0.
  Shared budget and knobs, all unchanged by this commit: 164864 B dynamic
  smem, 0 B static, 384 threads / 3 warpgroups, register split 24/240, tile
  128x128x128, grid 1024, KV_STAGE = 2, 10 mbarriers, wg_wait depth N = 1,
  blocking wait encoding, single-warp producer issue, D1.4 heavy-first decode,
  EW_W = 4.  The only value that moves is SMEM_CONTIG, 32 -> 128.

Re-mount onto the rewound workspace base (this commit).  The workspace tree
was reset to commit 0b3ef1b, the iter-13 tree, which predates two deliveries
the ledger above already records as shipped and kept; both are restored here
so the tree matches the last tree the driver evaluated in full (iter 17):
  D0.10 / D0.11 -- M0 construction step 6: the two static ping-pong inboxes
    (PP_BARRIER_ID + w, 256 participants, asymmetric one-way placement), the
    WG1 priming arrive outside the trip loop, and the D0.10 tabled trip order
    whose moved row is 9, arrive(K.EMPTY[p]), sitting after the element-wise
    segment rather than immediately after wait_group(1).
  D3.1 / D3.2 -- the adjudicated F1 arm as a named constant: SMEM_CONTIG, the
    single knob that drives all four coupled call sites, at 128 (SW128).  The
    iter-13 base spells the same surface as the bare literal D, so this half of
    the re-mount is a naming change and the executable surface is identical.
    The knob is additionally pinned to a literal here, where the evaluated tree
    read it from ATTN_FWD_SMEM_CONTIG with 128 as the default -- the fork is
    closed, so the arm is no longer selectable at run time.
No other value moves: tile shape, stage counts, smem map, mbarrier roster,
register split, grid, wait encoding, producer organisation, EW_W and the D1.4
decode are all as tabulated in the M2 gate block above.

NOTE: no `from __future__ import annotations` in this file (PEP 563 stringifies
@cute.struct / @cute.kernel annotations and breaks cute.compile).
"""

import math
import os

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

# ---------------------------------------------------------------- shape A ----
B, H_Q, H_KV, S, D = 2, 16, 1, 4096, 128
GQA = H_Q // H_KV                     # 16 query heads per kv head

# D0.3 tile shape: bM = bN = bK = D = 128 (D is not blocked).
BM, BN = 128, 128
# D0.2 pack_GQA: an M-tile is 16 heads x 8 sequence positions, head = fast axis
# (row % 16 = head, row // 16 = sequence offset inside the tile).
S_PER_TILE = BM // GQA                # 8
NUM_M_TILES = (S * H_Q) // BM         # 512
GRID = B * NUM_M_TILES                # 1024  (D0.2 architecture-conformance line)

# D0.4 warp roles: 3 warpgroups = 1 producer (24 regs) + 2 consumers (240 regs).
NUM_WG = 3
NUM_THREADS = NUM_WG * 128            # 384   (D0.4 architecture-conformance line)
PRODUCER_REGS = 24
CONSUMER_REGS = 240
CONSUMER_WG = 2
CONSUMER_THREADS = CONSUMER_WG * 128
CONSUMER_WARPS = CONSUMER_THREADS // 32   # 8 = EMPTY-side arrive count (D0.9)

# D0.14 epilogue rendezvous.  barrier_id 0 is reserved for sync_threads.
#   id 1 = epilogue rendezvous, 288 threads = 2 consumer WGs + the store warp
#   id 2 = loop-tail rendezvous, 256 threads = the 2 consumer WGs only; it is
#          what proves BOTH warpgroups have retired every chain0 (= every read
#          of sQ) before the Q buffer is recycled as the O staging buffer.
EPI_BARRIER_ID = 1
LOOP_TAIL_BARRIER_ID = 2
STORE_WARP = 0
EPI_THREADS = CONSUMER_THREADS + 32   # 288  (D0.11 signal table)

# D0.11 ping-pong inboxes.  One static id per consumer warpgroup: PP_BARRIER_ID
# + w is warpgroup w's INBOX -- the only barrier w ever blocks on, and the only
# one the OTHER warpgroup ever signals.  That asymmetric placement (one side
# arrives, the other side only waits) is the contract; the DSL wants constant
# ids, so the dynamic id of the signal table is spelled as the two-static-id
# branch the decision's implementation note allows.  Participants = 256 = the
# two consumer warpgroups, producer excluded: WG w's own 128 threads block in
# `arrive_and_wait` and the peer's 128 supply the other half of the count.
PP_BARRIER_ID = 8
PP_THREADS = CONSUMER_THREADS         # 256

# D0.7 smem budget, closed to the byte:
#   mbarrier page 1024 | V stage0 32768 | V stage1 32768 | Q 32768
#                      | K stage0 32768 | K stage1 32768   = 164864 B
KV_STAGE = 2
MBAR_SLOTS = 128                      # 128 * 8 B = 1024 B mbarrier page
Q_ELEMS = BM * D                      # 16384 bf16 = 32768 B
KV_ELEMS = BN * D * KV_STAGE          # 32768 bf16 = 65536 B
SMEM_BYTES = MBAR_SLOTS * 8 + (KV_ELEMS + Q_ELEMS + KV_ELEMS) * 2        # 164864

# D0.8 TMA transaction sizes (one full tile per stage, per buffer).
Q_BYTES = BM * D * 2                  # 32768
KV_BYTES = BN * D * 2                 # 32768

# D0.9 mbarrier page map.  10 slots, 8 B apart, in the order
#   [Q.FULL, Q.EMPTY, K.FULL0, K.FULL1, K.EMPTY0, K.EMPTY1,
#    V.FULL0, V.FULL1, V.EMPTY0, V.EMPTY1]
# laid out as a (2, 5) column-major view so that each pipeline gets a
# contiguous run of Int64 slots: Q = column 0 (2 slots), K = columns 1..2
# (4 slots), V = columns 3..4 (4 slots).
MBAR_Q_COL = 0
MBAR_K_COL = 1
MBAR_V_COL = 3

# D0.5 the ONE folded constant: log2(e) / sqrt(D).  Every S element pays a
# single FFMA (`s * SOFTMAX_SCALE - row_max_scaled`) on the way into exp2;
# there is no separate 1/sqrt(D) multiply and no base conversion.
SOFTMAX_SCALE = math.log2(math.e) / math.sqrt(D)

# D3.1 / D3.2 (adjudicated fork F1): the ONE swizzle-width knob.  The third
# argument of get_smem_layout_atom is the CONTIGUOUS-mode extent in elements
# and the atom family is picked from `extent * dtype.width` bits:
#   >=1024 -> SW128, >=512 -> SW64, >=256 -> SW32, else INTER (no swizzle).
# bf16 therefore reads 128/64 -> SW128, 32 -> SW64, 16 -> SW32, 8 -> INTER.
# Both sides of the exchange surface hang off this single value: the read-side
# Q/K/V descriptors (D3.2) and the epilogue write-side STSM destination (D3.1)
# share `smem_atom`, so the swizzle family can never go out of sync between
# them.  Tile shape, stage count, buffer bytes and mbarrier slots do not move.
#
# Fork F1's alternate arm SW64 (32) was shipped and measured at iter 16; this
# commit restores the SW128 incumbent (128).  The device-window gap between the
# arms is 0.594%, inside the resolution band, so the fork was settled on the
# co-adjudication quantities off the same NCU table -- time_ns 256789 (SW128)
# vs 259883 (SW64), tensor_pct 66.36 vs 65.78 -- which both favour SW128.  See
# the F1 ledger in the module docstring for the full reading.
# F1 is settled, so the knob is pinned to the surviving arm rather than left
# on an environment escape hatch: the exchange surface is now a compile-time
# constant on every one of its four coupled call sites.
SMEM_CONTIG = 128

# D2.3 in-flight width W of the elementwise segment: the number of mutually
# independent element chains the segment's two reductions (row max, row sum)
# are carried on.  W is a property of the SOURCE dependency graph -- ptxas
# reorders and renames but never rewrites it, so a single rolling accumulator
# (W = 1) leaves the consumer's register slack unspendable no matter how large
# it is.  Candidate ladder {1, 4, 8}; this is the first rung above the M0
# starting point.  The cost is W-1 extra live fp32 per reduction, priced
# against the consumer role's 240-register setmaxnreg quota (static allocation
# 168, spill 0), and n_col must stay divisible by W (32 % 4).
EW_W = 4


@cute.struct
class SharedStorage:
    # Field order mirrors the D0.7 layout map (mbarrier page, V0/V1, Q, K0/K1).
    mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, MBAR_SLOTS], 1024]
    sV: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]
    sQ: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, Q_ELEMS], 1024]
    sK: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]


def _as_bf16(v):
    """Float32 -> BFloat16, tolerant of the DSL's cast spelling."""
    f = cutlass.Float32(v)
    if hasattr(f, "to"):
        return f.to(cutlass.BFloat16)
    return cutlass.BFloat16(f)


def _as_f32(v):
    """BFloat16 -> Float32, tolerant of the DSL's cast spelling."""
    if hasattr(v, "to"):
        return v.to(cutlass.Float32)
    return cutlass.Float32(v)


# Which float-max spelling the installed DSL surface offers (the probe prints
# it, so a missing symbol reads as "resolved to select" rather than a crash).
_FMAX_FN = None
_FMAX_IMPL = "select"
for _mod, _name in ((cute, "math.fmax"), (cute, "math.max"),
                    (cute, "arch.fmax"), (cutlass, "max")):
    _obj = _mod
    for _part in _name.split("."):
        _obj = getattr(_obj, _part, None)
        if _obj is None:
            break
    if _obj is not None:
        _FMAX_FN, _FMAX_IMPL = _obj, _name
        break


# Same resolver pattern for the reciprocal: the epilogue wants the fast
# approximate reciprocal (MUFU.RCP), not a full-precision div.rn.
_RCP_FN = None
for _mod, _name in ((cute, "math.rcp"), (cute, "arch.rcp_approx"),
                    (cute, "math.reciprocal")):
    _obj = _mod
    for _part in _name.split("."):
        _obj = getattr(_obj, _part, None)
        if _obj is None:
            break
    if _obj is not None:
        _RCP_FN = _obj
        break


def _rcp(x):
    if _RCP_FN is not None:
        return _RCP_FN(x)
    return cutlass.Float32(1.0) / x


@cute.jit
def _fmax_select(a, b):
    """Fallback when the DSL exposes no fmax: a decorated select.

    The `if` here is on a RUNTIME value, so it only becomes staged control flow
    (scf.if / select) because this helper carries its own @cute.jit -- the
    preprocessor does not descend into undecorated callees.
    """
    r = a
    if b > a:
        r = b
    return r


if _FMAX_FN is None:
    _FMAX_FN, _FMAX_IMPL = _fmax_select, "select(@cute.jit)"


def _fmax(a, b):
    return _FMAX_FN(a, b)


def _acc_mn_layout(acc_layout):
    """((2,2,V), MMA_M, MMA_N) -> (rows, cols) view of the same registers.

    Pure layout composition (no data movement): mode 0's second `2` is the
    row offset (m, m+8) inside the WGMMA C atom, everything else is a column.
    """
    acc_cm = cute.make_layout(acc_layout.shape)
    shape = ((acc_cm.shape[0][1], acc_cm.shape[1]),
             (acc_cm.shape[0][0], *acc_cm.shape[0][2:], acc_cm.shape[2]))
    stride = ((acc_cm.stride[0][1], acc_cm.stride[1]),
              (acc_cm.stride[0][0], *acc_cm.stride[0][2:], acc_cm.stride[2]))
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


@cute.jit
def _gemm(tiled_mma, acc, tA, tB, stage, n_kb: cutlass.Constexpr):
    """One staged WGMMA chain: fence -> n_kb k-blocks -> commit (no wait).

    The staged B operand is sliced *inside* the cute.gemm call (4-mode index);
    binding it to a name first breaks tracing inside a dynamic loop body.
    """
    warpgroup.fence()
    for kb in cutlass.range_constexpr(n_kb):
        cute.gemm(tiled_mma, acc,
                  tA[(None, None, kb)], tB[(None, None, kb, stage)], acc)
    warpgroup.commit_group()


@cute.jit
def _softmax(acc_mn, c_mn, row_max, row_sum, row_corr, row_spos,
             n_row: cutlass.Constexpr, n_col: cutlass.Constexpr, kt_first_col):
    """D0.12 mask + D0.5 exp2 + D0.13 online merge, in place on chain0's acc.

    On entry acc_mn holds S for this K-tile; on exit it holds P = exp2(S*scale
    - m_new*scale) with masked positions at exactly 0, row_max/row_sum are
    advanced to this tile, and row_corr holds exp2((m_old - m_new)*scale) for
    the deferred O fixup.
    """
    for r in cutlass.range_constexpr(n_row):
        # D0.12: one unconditional predicate per element, single loop variant.
        # `col_limit` is the last unmasked column of this row in this K-tile.
        col_limit = row_spos[r] - kt_first_col
        for c in cutlass.range_constexpr(n_col):
            if c_mn[r, c][1] > col_limit:
                acc_mn[r, c] = cutlass.Float32(float("-inf"))

        # D0.13: row max merged inside the loop body -- element-wise first,
        # then a 2-step butterfly across the 4 lanes of the quad that share
        # this row (the WGMMA C atom splits a row over lanes l, l^1, l^2, l^3).
        # D2.3: the element-wise part runs on EW_W independent partials that
        # are merged once at the end, instead of one rolling max (the previous
        # form was a single n_col-1 deep FMAX chain, every step waiting on the
        # previous step's result).  Chain depth per row: 31 -> 7 + 3.
        mx_w = []
        for w in cutlass.range_constexpr(EW_W):
            mx_w.append(acc_mn[r, w])
        for g in cutlass.range_constexpr(1, n_col // EW_W):
            for w1 in cutlass.range_constexpr(EW_W):
                mx_w[w1] = _fmax(mx_w[w1], acc_mn[r, g * EW_W + w1])
        mx = mx_w[0]
        for w2 in cutlass.range_constexpr(1, EW_W):
            mx = _fmax(mx, mx_w[w2])
        mx = _fmax(mx, cute.arch.shuffle_sync_bfly(mx, 1))
        mx = _fmax(mx, cute.arch.shuffle_sync_bfly(mx, 2))
        m_old = row_max[r]
        mx = _fmax(mx, m_old)
        row_max[r] = mx

        # D0.5: the row's max*scale is hoisted, so the element loop is
        # 1 FFMA + 1 exp2 and nothing else.
        mx_scaled = mx * SOFTMAX_SCALE
        corr = cute.math.exp2(m_old * SOFTMAX_SCALE - mx_scaled, fastmath=True)
        row_corr[r] = corr

        # D2.3 again, on the row sum: the exp2 calls were already independent
        # of each other, but every one of them fed the SAME rolling sum, so the
        # segment closed on an n_col deep FADD chain that no amount of register
        # slack could interleave.  EW_W partial sums, merged once at the end.
        sum_w = []
        for w3 in cutlass.range_constexpr(EW_W):
            sum_w.append(cutlass.Float32(0.0))
        for g2 in cutlass.range_constexpr(n_col // EW_W):
            for w4 in cutlass.range_constexpr(EW_W):
                c2 = g2 * EW_W + w4
                p = cute.math.exp2(acc_mn[r, c2] * SOFTMAX_SCALE - mx_scaled,
                                   fastmath=True)
                acc_mn[r, c2] = p
                sum_w[w4] = sum_w[w4] + p
        acc_sum = sum_w[0]
        for w5 in cutlass.range_constexpr(1, EW_W):
            acc_sum = acc_sum + sum_w[w5]
        acc_sum = acc_sum + cute.arch.shuffle_sync_bfly(acc_sum, 1)
        acc_sum = acc_sum + cute.arch.shuffle_sync_bfly(acc_sum, 2)
        row_sum[r] = row_sum[r] * corr + acc_sum


@cute.jit
def _rescale_o(acc_o_mn, row_corr,
               n_row: cutlass.Constexpr, n_col: cutlass.Constexpr):
    """D0.13 deferred fixup: 64 FMUL, one per O accumulator register."""
    for r in cutlass.range_constexpr(n_row):
        corr = row_corr[r]
        for c in cutlass.range_constexpr(n_col):
            acc_o_mn[r, c] = acc_o_mn[r, c] * corr


def _one_way_arrive(bar):
    """Non-blocking arrive on a NamedBarrier -- the ARRIVE-only half of D0.11.

    `arrive_and_wait` is a symmetric meet and would couple the two warpgroups
    the wrong way round (both block, both release together); the phase contract
    needs one side to signal without blocking.  The spelling is resolved at
    trace time so the file does not depend on which face this DSL build
    exposes: the NamedBarrier method when it has one, otherwise the raw
    `cute.arch` primitive it wraps.
    """
    fn = getattr(bar, "arrive", None)
    if fn is not None:
        fn()
    else:
        cute.arch.barrier_arrive(barrier_id=bar.barrier_id,
                                 number_of_threads=bar.num_threads)


@cute.jit
def _pp_recv(cwg, pp0, pp1):
    """D0.11 step 1 -- take the baton: block on THIS warpgroup's own inbox.

    The dynamic barrier id of the signal table, spelled as the two-static-id
    branch (the decision's implementation note declares the two equivalent:
    what has to survive is the asymmetric placement, not a register-held id).
    """
    if cwg == 0:
        pp0.arrive_and_wait()
    else:
        pp1.arrive_and_wait()


@cute.jit
def _pp_send(cwg, pp0, pp1):
    """D0.11 step 6 -- hand the baton over: arrive on the PEER's inbox only."""
    if cwg == 0:
        _one_way_arrive(pp1)
    else:
        _one_way_arrive(pp0)


@cute.jit
def _pack_p(src_pairs, dst_pairs, n_pairs: cutlass.Constexpr):
    """D0.6 RS hand-off: chain0's fp32 acc -> chain1's bf16 A operand.

    The two fragments have the same compact column-major element order, so the
    conversion is a pure in-register pass (32 x cvt.f32x2.bf16x2), no shuffle
    and no smem round trip.
    """
    for j in cutlass.range_constexpr(n_pairs):
        pv2 = cute.arch.cvt_f32x2_bf16x2(src_pairs[(None, j)].load())
        dst_pairs[(None, j)].store(
            cute.TensorSSA(pv2, (2,), cutlass.BFloat16))


@cute.kernel
def _attn_kernel(tma_q, gmem_q, tma_k, gmem_k, tma_v, gmem_v, tma_o, gmem_o,
                 tiled_mma_qk, tiled_mma_pv, tiled_copy_r2s,
                 sq_layout, skv_layout, sq_tma_layout, svt_layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- shared prelude: everything before the ONE role branch (D0.4 rule) ----
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_k)
        cpasync.prefetch_descriptor(tma_v)
        cpasync.prefetch_descriptor(tma_o)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sQ_tma = storage.sQ.get_tensor(sq_tma_layout.outer, swizzle=sq_tma_layout.inner)
    sQ = storage.sQ.get_tensor(sq_layout.outer, swizzle=sq_layout.inner)
    sK = storage.sK.get_tensor(skv_layout.outer, swizzle=skv_layout.inner)
    sV = storage.sV.get_tensor(skv_layout.outer, swizzle=skv_layout.inner)
    # D0.6: second view of the *same* V bytes, transposed to (D, BN, stage) so
    # chain1 can reduce along BN.  Costs no smem and no copy.
    sVt = storage.sV.get_tensor(svt_layout.outer, swizzle=svt_layout.inner)

    # ---- D0.9: three independent mbarrier pipelines over one 1024 B page ----
    mbar_t = storage.mbar.get_tensor(
        cute.make_layout((2, MBAR_SLOTS // 2), stride=(1, 2)))
    q_pipe = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_t[(None, MBAR_Q_COL)].iterator,
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 CONSUMER_WARPS),
        tx_count=Q_BYTES,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
    k_pipe = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_t[(None, MBAR_K_COL)].iterator,
        num_stages=KV_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 CONSUMER_WARPS),
        tx_count=KV_BYTES,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
    v_pipe = pipeline.PipelineTmaAsync.create(
        barrier_storage=mbar_t[(None, MBAR_V_COL)].iterator,
        num_stages=KV_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread,
                                                 CONSUMER_WARPS),
        tx_count=KV_BYTES,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )

    # D0.2 flat grid decode: blockIdx -> (b, m), with the D1.4 static
    # permutation.  Both keys are power-of-two, so the whole decode is one
    # shift + one and; NUM_M_TILES - 1 folds into a compile-time constant.
    # b is the fast axis and m runs DESCENDING, so the flat CTA order is
    # globally sorted by K-loop count n(m) = (8m+7)//BN + 1 (32 down to 1),
    # each length appearing once per batch.  Bijective on [0, GRID).
    # D2.1: the whole integer bookkeeping chain (b_idx, m_tile, n_kt, and the
    # tile coordinates they feed) is seeded from blockIdx and compile-time
    # constants only -- tidx enters at the consumption point (row_spos) and
    # nowhere earlier, so the chain stays warp-uniform and is priced against
    # the uniform register file rather than the 240-register setmaxnreg quota.
    # The marker is placed once, at the chain's entry (it is a hint: bidx is
    # already provably warp-invariant, but the mark does not depend on ptxas
    # proving it and survives the scf.if / @cute.jit boundaries downstream).
    bidx_u = cute.arch.make_warp_uniform(bidx)
    b_idx = bidx_u % B
    m_tile = (NUM_M_TILES - 1) - bidx_u // B
    # D0.12 causal range: the M-tile's last sequence position is
    # m*8 + 7, so it needs K-tiles 0 .. that position's tile (max 32).
    n_kt = (m_tile * S_PER_TILE + (S_PER_TILE - 1)) // BN + 1

    # D0.8 supply path: 100% TMA.
    gQ = cute.zipped_divide(gmem_q, (GQA, S_PER_TILE, D))
    gK = cute.zipped_divide(gmem_k, (BN, D))
    gV = cute.zipped_divide(gmem_v, (BN, D))

    tQsQ, tQgQ = cpasync.tma_partition(
        tma_q, 0, cute.make_layout(1), cute.group_modes(sQ_tma, 0, 3), gQ)
    tKsK, tKgK = cpasync.tma_partition(
        tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), gK)
    tVsV, tVgV = cpasync.tma_partition(
        tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), gV)

    # D0.14 write side: the O tile leaves through the SAME 4D box the Q tile
    # came in through (mO and mQ have identical shape and mode order), and its
    # smem source IS the Q buffer -- sQ_tma, the very tensor the Q load landed
    # in.  No extra smem is budgeted for the staging (D0.7 alias clause).
    gO = cute.zipped_divide(gmem_o, (GQA, S_PER_TILE, D))
    tOsO, tOgO = cpasync.tma_partition(
        tma_o, 0, cute.make_layout(1), cute.group_modes(sQ_tma, 0, 3), gO)

    # Bulk-group throttle for the store.  It carries no mbarrier, so it costs
    # nothing in the D0.7 smem map; only the store warp is in its group.
    tma_store_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
    )
    epi_barrier = pipeline.NamedBarrier(
        barrier_id=EPI_BARRIER_ID, num_threads=EPI_THREADS)
    loop_tail_barrier = pipeline.NamedBarrier(
        barrier_id=LOOP_TAIL_BARRIER_ID, num_threads=CONSUMER_THREADS)
    # D0.11: consumer WG w's inbox.  Built in the shared prelude with every
    # other synchronisation object (D0.4 rule), used only inside the consumer
    # branch; the producer never touches either one.
    pp_barrier0 = pipeline.NamedBarrier(
        barrier_id=PP_BARRIER_ID, num_threads=PP_THREADS)
    pp_barrier1 = pipeline.NamedBarrier(
        barrier_id=PP_BARRIER_ID + 1, num_threads=PP_THREADS)

    cute.arch.sync_threads()

    # ---- the single role branch; dec/inc is the first statement of each body --
    if warp_idx < 4:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        # F2 selected edge (D0.4): one warp issues every TMA.
        if warp_idx == 0:
            q_pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            q_pipe.producer_acquire(q_pstate)
            cute.copy(tma_q,
                      tQgQ[(None, (0, m_tile, 0, b_idx))],
                      tQsQ,
                      tma_bar_ptr=q_pipe.producer_get_barrier(q_pstate))
            q_pipe.producer_commit(q_pstate)

            k_pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, KV_STAGE)
            v_pstate = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, KV_STAGE)
            for kt in cutlass.range(n_kt):
                k_pipe.producer_acquire(k_pstate)
                cute.copy(tma_k,
                          tKgK[(None, (kt, 0, b_idx))],
                          tKsK[(None, k_pstate.index)],
                          tma_bar_ptr=k_pipe.producer_get_barrier(k_pstate))
                k_pipe.producer_commit(k_pstate)
                k_pstate.advance()

                v_pipe.producer_acquire(v_pstate)
                cute.copy(tma_v,
                          tVgV[(None, (kt, 0, b_idx))],
                          tVsV[(None, v_pstate.index)],
                          tma_bar_ptr=v_pipe.producer_get_barrier(v_pstate))
                v_pipe.producer_commit(v_pstate)
                v_pstate.advance()

            # D0.14: the store is issued by a WHOLE warp (the TMA copy atom
            # elects internally; a single-lane issue is UB and hangs).  This
            # warp is the 32-thread half of the 288-thread epilogue rendezvous.
            epi_barrier.arrive_and_wait()
            cute.copy(tma_o, tOsO, tOgO[(None, (0, m_tile, 0, b_idx))])
            tma_store_pipeline.producer_commit()
            tma_store_pipeline.producer_tail()
    else:
        cute.arch.setmaxregister_increase(CONSUMER_REGS)

        # ---- setup: everything allocated / bound exactly once (no fragment,
        # ---- no atom, no rebinding inside the dynamic loop body) ------------
        cidx = tidx - 128                                   # 0 .. 255
        cwg = cute.arch.make_warp_uniform(cidx // 128)       # 0 or 1
        wg_slot = cute.make_layout(CONSUMER_WG, stride=128)(cwg)
        wg_qk = tiled_mma_qk.get_slice(wg_slot)
        wg_pv = tiled_mma_pv.get_slice(wg_slot)

        # D0.6: chain0 operands are smem descriptors (SS), chain1's A is RMEM.
        tSrQ = wg_qk.make_fragment_A(wg_qk.partition_A(sQ))
        tSrK = wg_qk.make_fragment_B(wg_qk.partition_B(sK))
        tOrP = wg_pv.make_fragment_A(wg_pv.partition_shape_A((BM, BN)))
        tOrV = wg_pv.make_fragment_B(wg_pv.partition_B(sVt))

        acc_s = cute.make_fragment(wg_qk.partition_shape_C((BM, BN)),
                                   cutlass.Float32)
        acc_o = cute.make_fragment(wg_pv.partition_shape_C((BM, D)),
                                   cutlass.Float32)

        QK_KB = cute.size(tSrQ, mode=[2])          # 8 k-blocks of 16
        PV_KB = cute.size(tOrP, mode=[2])          # 8 k-blocks of 16
        tiled_mma_qk.set(warpgroup.Field.ACCUMULATE, True)
        tiled_mma_pv.set(warpgroup.Field.ACCUMULATE, True)

        # (rows, cols) views of the two C fragments + their static coordinates.
        thr_qk = tiled_mma_qk.get_slice(cidx)
        thr_pv = tiled_mma_pv.get_slice(cidx)
        cS = thr_qk.partition_C(cute.make_identity_tensor((BM, BN)))
        cO = thr_pv.partition_C(cute.make_identity_tensor((BM, D)))
        acc_s_mn = cute.make_tensor(acc_s.iterator, _acc_mn_layout(acc_s.layout))
        acc_o_mn = cute.make_tensor(acc_o.iterator, _acc_mn_layout(acc_o.layout))
        cS_mn = cute.make_tensor(cS.iterator, _acc_mn_layout(cS.layout))
        cO_mn = cute.make_tensor(cO.iterator, _acc_mn_layout(cO.layout))
        N_ROW = cute.size(acc_s_mn, mode=[0])      # 2 rows per thread
        N_COL_S = cute.size(acc_s_mn, mode=[1])    # 32 S columns per thread
        N_COL_O = cute.size(acc_o_mn, mode=[1])    # 32 O columns per thread

        # D0.6 RS hand-off views (built once, reused every tile).
        P_PAIRS = cute.size(tOrP) // 2             # 32
        p_src = cute.make_tensor(acc_s.iterator, cute.make_layout((2, P_PAIRS)))
        p_dst = cute.make_tensor(tOrP.iterator, cute.make_layout((2, P_PAIRS)))

        # D0.14 write path, bound once (nothing here is rebound in a loop).
        # rO is the ONLY epilogue staging register block: the bf16 fragment the
        # STSM atom reads directly.  There is no fp32 staging copy -- the O tile
        # is the whole epi tile (num_epi == 1), so an fp32 + bf16 staging pair
        # would be a full-size duplicate of the accumulator.
        rO = cute.make_fragment_like(acc_o, cutlass.BFloat16)
        O_PAIRS = cute.size(acc_o) // 2            # 32
        o_src = cute.make_tensor(acc_o.iterator, cute.make_layout((2, O_PAIRS)))
        o_dst = cute.make_tensor(rO.iterator, cute.make_layout((2, O_PAIRS)))
        thr_copy_r2s = tiled_copy_r2s.get_slice(cidx)   # consumer-local index
        # dest = the dead Q buffer (D0.7 alias); source = the SAME registers the
        # F2FP wrote.  `retile` keeps the C fragment's mode split and
        # `partition_D` keeps the SMEM tile's, so the two disagree on mode
        # count even though they carry the same 64 bf16 per thread; the probe
        # sweep (notes/probe_m0_step5_epilogue.py, contract 1) measured all four
        # candidate spellings against a row/column-ramp bijection reference and
        # only this one -- view the fragment's registers directly on the
        # destination's shape -- is both accepted by `cute.copy` and the
        # identity on (packed row, D column).
        tRS_sO = thr_copy_r2s.partition_D(sQ)
        tRS_rO = cute.make_tensor(rO.iterator,
                                  cute.make_layout(tRS_sO.shape))

        row_max = cute.make_fragment(N_ROW, cutlass.Float32)
        row_sum = cute.make_fragment(N_ROW, cutlass.Float32)
        row_corr = cute.make_fragment(N_ROW, cutlass.Float32)
        row_spos = cute.make_fragment(N_ROW, cutlass.Int32)
        for r0 in cutlass.range_constexpr(N_ROW):
            row_max[r0] = cutlass.Float32(float("-inf"))
            row_sum[r0] = cutlass.Float32(0.0)
            # D0.2 packed row -> sequence position (row // 16 = seq offset).
            row_spos[r0] = m_tile * S_PER_TILE + cS_mn[r0, 0][0] // GQA
        acc_o.fill(0.0)

        k_cstate = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, KV_STAGE)
        v_cstate = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, KV_STAGE)

        q_cstate = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1)
        q_pipe.consumer_wait(q_cstate)
        # Q.EMPTY is released only at the epilogue's alias reuse point (D0.7),
        # one instruction before the first byte of the Q buffer is overwritten.

        # ---- prologue: K-tile 0's chain0 is peeled (no PV partner yet), which
        # ---- is what puts chain1 one tile behind chain0 for the whole loop ---
        k_pipe.consumer_wait(k_cstate)
        acc_s.fill(0.0)
        _gemm(tiled_mma_qk, acc_s, tSrQ, tSrK, k_cstate.index, QK_KB)
        warpgroup.wait_group(0)
        k_pipe.consumer_release(k_cstate)
        k_cstate.advance()
        _softmax(acc_s_mn, cS_mn, row_max, row_sum, row_corr, row_spos,
                 N_ROW, N_COL_S, 0)
        _pack_p(p_src, p_dst, P_PAIRS)

        # ---- D0.11 priming: the peeled first beat has already offset the two
        # ---- warpgroups; this single one-way arrive from WG1 puts the first
        # ---- baton in WG0's inbox and makes WG0 the leader.  It is issued
        # ---- unconditionally (never inside the trip loop), so a CTA whose
        # ---- n_kt is 1 -- the steady loop then runs zero trips -- simply
        # ---- leaves it unclaimed instead of waiting on a partner that never
        # ---- comes.  On bar 8 that leaves exactly one arrive outstanding at
        # ---- CTA exit; nothing waits on bar 8 again, and the loop-tail (2)
        # ---- and epilogue (1) meets are on other ids.
        if cwg == 1:
            _one_way_arrive(pp_barrier0)

        # ---- steady state: chain0(tile j) and chain1(tile j-1) are both in
        # ---- flight; wait_group(1) retires chain0 only, so the softmax of
        # ---- tile j runs inside chain1's shadow (D0.10 / D0.13).  The trip is
        # ---- issued in D0.10's tabled order, step for step: take baton, K
        # ---- FULL, chain0, V FULL, chain1, hand baton over, wait_group(1),
        # ---- softmax, K.EMPTY, wait_group(0), V.EMPTY, 64-FMUL fixup.
        # ---- Both warpgroups run n_kt-1 trips over the same m_tile, so the
        # ---- receive/hand-over counts match trip for trip.
        for it in cutlass.range(n_kt - 1):
            kt = it + 1
            _pp_recv(cwg, pp_barrier0, pp_barrier1)            # 1 take baton

            k_pipe.consumer_wait(k_cstate)                     # 2 K.FULL[p]
            acc_s.fill(0.0)
            _gemm(tiled_mma_qk, acc_s, tSrQ, tSrK, k_cstate.index, QK_KB)  # 3

            v_pipe.consumer_wait(v_cstate)                     # 4 V.FULL[p]
            _gemm(tiled_mma_pv, acc_o, tOrP, tOrV, v_cstate.index, PV_KB)  # 5

            # 6: the baton goes over BEFORE this warpgroup's element-wise
            # segment, which is the whole point -- the peer issues its two MMA
            # chains while we hold the SFU, so the two softmax segments never
            # overlap (D0.11), on top of each one already hiding inside its own
            # chain1 (D0.10).
            _pp_send(cwg, pp_barrier0, pp_barrier1)            # 6 hand over

            warpgroup.wait_group(1)                            # 7 chain0 only
            _softmax(acc_s_mn, cS_mn, row_max, row_sum, row_corr, row_spos,
                     N_ROW, N_COL_S, kt * BN)                  # 8 softmax

            # 9: K.EMPTY is released here, after the element-wise segment --
            # not at chain0's last read of sK.  Releasing later is never a
            # hazard, only a throttle: it is what keeps the producer from
            # overrunning the two-deep K ring while the tile is still resident.
            k_pipe.consumer_release(k_cstate)                  # 9 K.EMPTY[p]
            k_cstate.advance()

            warpgroup.wait_group(0)                            # 10 drain PV
            v_pipe.consumer_release(v_cstate)                  # 11 V.EMPTY[p]
            v_cstate.advance()
            # Only now is chain1 retired: P may be overwritten and O fixed up.
            _pack_p(p_src, p_dst, P_PAIRS)
            _rescale_o(acc_o_mn, row_corr, N_ROW, N_COL_O)     # 12 64 FMUL

        # ---- epilogue: the last PV, which the steady state never issued -----
        v_pipe.consumer_wait(v_cstate)
        _gemm(tiled_mma_pv, acc_o, tOrP, tOrV, v_cstate.index, PV_KB)
        warpgroup.wait_group(0)
        v_pipe.consumer_release(v_cstate)
        v_cstate.advance()

        # ---- D0.14 epilogue: F2FP -> STSM -> aliased smem -> TMA store -----
        # (1) Both consumer WGs have now drained every chain0, so nobody reads
        #     sQ any more and the buffer may be recycled.
        loop_tail_barrier.arrive_and_wait()
        q_pipe.consumer_release(q_cstate)

        # (2) The last rescale: 2 MUFU.RCP per thread, folded into the same
        #     multiply that feeds the bf16 conversion (no fp32 staging block).
        for r1 in cutlass.range_constexpr(N_ROW):
            inv_l = _rcp(row_sum[r1])
            for c3 in cutlass.range_constexpr(N_COL_O):
                acc_o_mn[r1, c3] = acc_o_mn[r1, c3] * inv_l
        _pack_p(o_src, o_dst, O_PAIRS)

        # (3) STSM: one matrix store per thread into the swizzled buffer; the
        #     destination's swizzle family is D3.1's knob (see SMEM_CONTIG).
        cute.copy(tiled_copy_r2s, tRS_rO, tRS_sO)

        # (4) generic proxy (STSM) -> async proxy (TMA) fence, then (5) the
        #     288-thread rendezvous that lets the store warp read what we wrote.
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                              space=cute.arch.SharedSpace.shared_cta)
        epi_barrier.arrive_and_wait()


@cute.jit
def _attn_fwd_jit(mQ, mK, mV, mO, stream):
    # D0.7 / D0.8 / D3.2: swizzled k-major tiles.  SMEM_CONTIG is the atom's
    # contiguous extent in bf16 elements and therefore the swizzle family of
    # the whole exchange surface; the (BN, D) / (GQA, S_PER_TILE, D) box shapes
    # handed to make_tiled_tma_atom below are the frozen D0.8 split and do not
    # move with it.
    smem_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16,
                                        SMEM_CONTIG),
        cutlass.BFloat16)
    sq_layout = cute.tile_to_shape(smem_atom, (BM, D), order=(0, 1))
    skv_layout = cute.tile_to_shape(smem_atom, (BN, D, KV_STAGE), order=(0, 1, 2))

    # The Q tile's M mode is the packed (head, seq-in-tile) pair; the TMA box has
    # to see them as two separate modes, so re-split mode 0 of the (BM, D) tile
    # into (GQA, S_PER_TILE) without touching the swizzle or the contiguous mode.
    sq_tma_layout = cute.composition(
        sq_layout, cute.make_layout((GQA, S_PER_TILE, D), stride=(1, GQA, BM)))
    skv_tma_layout = cute.slice_(skv_layout, (None, None, 0))
    # D0.6: MN-major read view of the same V bytes.  sV's (n, d, s) domain has
    # colex index n + d*BN + s*BN*D, so the (d, n, s) transposer's strides are
    # (BN, 1, BN*D).  Same bytes, same swizzle, no copy.
    svt_layout = cute.composition(
        skv_layout,
        cute.make_layout((D, BN, KV_STAGE), stride=(BN, 1, BN * D)))

    tma_q, gmem_q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mQ, sq_tma_layout,
        (GQA, S_PER_TILE, D), num_multicast=1)
    tma_k, gmem_k = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mK, skv_tma_layout,
        (BN, D), num_multicast=1)
    tma_v, gmem_v = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mV, skv_tma_layout,
        (BN, D), num_multicast=1)
    # D0.14: O leaves through the mirror image of the Q descriptor (same 4D
    # packed box, same swizzled single-stage layout).  S2G takes no multicast.
    tma_o, gmem_o = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mO, sq_tma_layout,
        (GQA, S_PER_TILE, D))

    # D0.3 / D0.6: chain0 S = Q.K^T, both operands k-major in smem (SS).
    # chain1 O += P.V, A from registers (RS), B = V read MN-major.
    # atom_layout_mnk=(2,1,1) splits the 128-row M-tile 64/64 over the two
    # consumer warpgroups; the WGMMA atom's M mode is always 64.
    tiled_mma_qk = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        cutlass.Float32,
        atom_layout_mnk=(CONSUMER_WG, 1, 1),
        tiler_mn=(64, BN),
    )
    tiled_mma_pv = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
        cutlass.Float32,
        atom_layout_mnk=(CONSUMER_WG, 1, 1),
        tiler_mn=(64, D),
        a_source=warpgroup.OperandSource.RMEM,
    )

    # D0.14 r2s construction, three layers, order fixed:
    #   store op picked from (C layout, out dtype, acc dtype) -> STSM family
    #   -> StMatrix8x8x16b x4 atom hung on chain1's C-fragment tiling
    #   -> tiled copy whose source order is exactly that C fragment.
    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
        LayoutEnum.ROW_MAJOR, elem_ty_d=cutlass.BFloat16,
        elem_ty_acc=cutlass.Float32)
    copy_atom_C = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(
            LayoutEnum.ROW_MAJOR.is_m_major_c(), 4),
        cutlass.BFloat16)
    tiled_copy_r2s = cute.make_tiled_copy_S(
        copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma_pv))

    _attn_kernel(tma_q, gmem_q, tma_k, gmem_k, tma_v, gmem_v, tma_o, gmem_o,
                 tiled_mma_qk, tiled_mma_pv, tiled_copy_r2s,
                 sq_layout, skv_layout, sq_tma_layout, svt_layout).launch(
        grid=[GRID, 1, 1],
        block=[NUM_THREADS, 1, 1],
        min_blocks_per_mp=1,      # without this ptxas silently drops setmaxnreg
        stream=stream,
    )


def _current_stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        from cuda.bindings import driver as _drv
        return _drv.CUstream(raw)
    except Exception:
        from cuda import cuda as _drv
        return _drv.CUstream(raw)


_COMPILED = None


def attn_fwd(q, k, v):
    global _COMPILED
    o = torch.empty_like(q)

    # gmem views in cute mode order.  Q/O: (H, S, D, B); K/V: (S, D, B).
    mQ = from_dlpack(q.permute(1, 2, 3, 0), assumed_align=16)
    mK = from_dlpack(k.squeeze(1).permute(1, 2, 0), assumed_align=16)
    mV = from_dlpack(v.squeeze(1).permute(1, 2, 0), assumed_align=16)
    mO = from_dlpack(o.permute(1, 2, 3, 0), assumed_align=16)
    stream = _current_stream()

    if _COMPILED is None:
        dump = os.environ.get("ATTN_FWD_DUMP_DIR")
        if dump:
            _COMPILED = cute.compile(
                _attn_fwd_jit, mQ, mK, mV, mO, stream,
                options="--dump-dir=%s --keep-cubin --keep-ptx" % dump)
        else:
            _COMPILED = cute.compile(_attn_fwd_jit, mQ, mK, mV, mO, stream)

    _COMPILED(mQ, mK, mV, mO, stream)
    return o
