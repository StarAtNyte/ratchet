"""conv2d_dgrad kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):
  conv2d_dgrad(dy, w) -> dx
    dy: torch.Tensor (N, K, P, Q) cuda bfloat16, channels_last memory format
    w:  torch.Tensor (K, C, R, S) cuda bfloat16, channels_last memory format
    dx: torch.Tensor (N, C, H, W) cuda bfloat16, channels_last memory format
  N=128, C=256, H=W=56, K=256, R=S=3, stride=2, padding=1, dilation=1,
  groups=1, no bias  ->  P=Q=28.  fp32 accumulation inside.

channels_last means the underlying memory is NPQK / KRSC / NHWC; cute-DSL code
should treat the innermost (contiguous) dimension as the channel.

This is the INPUT-GRADIENT of a STRIDED convolution: dx(n, c, h, w) sums
dy(n, k, p, q) * w(k, c, r, s) over exactly the taps with h = 2p + r - 1 and
w = 2q + s - 1.  Because of stride 2 the valid (r, s) tap set depends on the
parity class (h%2, w%2): the four classes see 1/2/2/4 of the 9 taps, i.e. the
operator decomposes into four smaller convolutions writing interleaved output
positions.  As an implicit GEMM the output side is huge (N*H*W = 401,408 rows
by C = 256 columns) while the reduction is shallow and parity-irregular.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc.  Delegating the math to torch / library conv, transposed-conv, grad, or
matmul calls fails the harness (eval/bench.py scans kernel.py for the
forbidden call families).  JIT-compile once and cache the compiled kernel
across calls (the bench times repeated calls).

-----------------------------------------------------------------------------
IMPLEMENTATION (milestone M0 architecture, decisions D0.1-D0.17)

One kernel, no pre/post pass, every dx element written exactly once (D0.1).

Decomposition (D0.2).  h = 2*hh + h0 inside parity class h0, and h = 2p+r-1
gives p = hh + d with r = h0 + 1 - 2d; d ranges over {0} for h0 = 0 and over
{1, 0} for h0 = 1.  So for a FIXED tap the map hh -> p is the identity shifted
by a compile-time d, each class is a dense implicit GEMM over its own taps
({1,2,2,4} of them, k depth {256,512,512,1024}), and no tap is ever masked at
the consumer: p = P only happens on the d = 1 halo, where the TMA descriptor's
per-axis bounds check supplies hardware zeros.

Tiling (D0.5).  bM 256 dx pixels x bN 128 channels x bK 64.  The 256 pixels are
the rectangle (n, hh, ww) = (16, 4, 4) of the class grid: 4 | 28 and 16 | 128,
so 8*7*7 = 392 M-tiles x 2 C-tiles = 784 tiles per class, 3,136 total, zero
waste.  Flattened, a tile's A operand is a contiguous 256x64 k-major block.

Data path (D0.10 / D0.11).  A(dy) and B(w) enter smem by rank-4 `.tile` TMA and
are read by WGMMA straight out of smem (SS form, no LDSM); the accumulators
leave through stmatrix into a swizzled epi buffer and out by rank-4 TMA store
into the parity-strided view of dx.  Each staging buffer therefore carries
several layout views of the SAME bytes, built by `cute.composition` of a flat
colex transposer onto the swizzled `tile_to_shape` allocation.  The layout
handed to `make_tiled_tma_atom` is built from a SINGLE-STAGE `tile_to_shape`
and composed once, never sliced out of the staged composition: the staged view
is `composition(ComposedLayout, Layout)`, and slicing that is the one spelling
of this buffer the GPU rejected.

Pipeline (D0.6-D0.9, D0.14-D0.16; rungs D1.1/D1.2).  384 threads = 1 producer
warpgroup (40 registers, warp 0 issues the TMAs) + 2 consumer warpgroups (232
registers, 2 accumulator chains each, split along M).  S = STAGE = 4 stage ring
(D1.1).  Steady state is 8 WGMMA per k-block per warpgroup, one commit_group,
then wait_group(WG_WAIT) so the next k-block's addressing runs under the chains
still in flight; the tile tail drains fully before the epilogue (D0.14 B, depth
0, contract not rung).  WG_WAIT = 2 (D1.2) means the consumer holds 3 of the 4
stages and releases k-block `count - WG_WAIT` -- the newest stage no live WGMMA
is still sourcing -- so the producer runs S - (WG_WAIT + 1) = 1 stage ahead.
That coupling is why D1.1 and D1.2 are one delivery.  k loop is unroll = 1
(D1.3, UNROLL) with the first max(1, WG_WAIT) k-blocks peeled (D1.3,
PEEL_FIRST): D0.16's peel, extended to the depth because a k-block cannot
release a stage until WG_WAIT groups sit behind it, and the tile-tail drain
hands those last WG_WAIT stages back.  Register split stays 40/232 (D1.4).
Ring positions are COMPUTED from one monotone k-block counter rather than carried
in a mutable pipeline state (see `state_at`), because a state advanced inside the
persistent tile loop leaves its fields defined in that region.

M1 RUNGS D1.3 / D1.4.  Both are held at their D0.16 / D0.7 starting points, and
both triggers are read off the one profiler block this lineage has measured on
the M0 device half [MEASURED, iter 5: 0.2945 ms end-to-end, time_ns 210,848].

  D1.3  unroll {1, 2}, peel {on, off}; held at 1 and on.  The rung's trigger for
        unroll = 2 is a CONJUNCTION -- `warp_active_pct` far under the
        reference's 17.68% AND `tensor_pct` not met.  Measured warp_active_pct
        13.30 is under it, but measured tensor_pct 68.69 is over the reference's
        own 61.89% [MEASURED, job 734407] and over both the M1 (>= 40%) and M3
        (>= 55%) lines, so the second conjunct is false and the trigger does not
        fire.  The reverse trigger (spill -> fall back to 1) has nothing to fall
        back from: spill is 0 + 0.  Milestone §5.2 additionally names a low
        `warp_active_pct` as the expected shape of warp specialization at 1
        CTA/SM rather than as a defect, so the one fired half is not evidence to
        act on.  Peel stays on because off is not reachable from here at all:
        PEEL = max(1, WG_WAIT) and D1.2 sits at depth 2 (see the assert).
  D1.4  {(40,232) start, (32,240), (48,224), (24,240)}; held at (40,232).  The
        rung's trigger is `spill_ld_sectors + spill_st_sectors` over D0.15's
        upper line of 850,680; measured is 0 + 0, i.e. zero, and
        `regs_per_thread` 168 sits mid-window of the gate's [150, 180].  There is
        no spill to relieve in either direction.

Holding is the delivery, not a deferral: the trigger conditions are the rungs'
own criteria and they are measured, not assumed.  Both rungs are now named
constants with the illegal peel/depth combination asserted, so the next round
that does want to move one moves a value rather than an expression.

Schedule (D0.12).  grid = 132 = SM count, persistent: each CTA walks
tile = blockIdx + i*132 within each parity class, decoding (i_n, i_p, i_q,
c_tile) by integer division — no index table anywhere.

Cluster (D0.13).  CTAs launch as clusters of 2 along x.  132 = 66 clusters, and
because c_tile is the fastest axis of the tile decode while GRID is even, the
two CTAs of a cluster carry the SAME (i_n, i_p, i_q) and the two different
c_tiles at every step of the persistent loop: tile = bidx + i*132 keeps the
parity of bidx, so ranks 0/1 of a cluster see c_tile 0/1 of one M-tile.  That is
D0.13's C-axis pairing — the pair's A(dy) operand is byte-identical and only
B(w) differs — and it is what a dy multicast would ride on.  The pairing and the
cluster shape are the frozen half of D0.13; the multicast OPERAND is the
[INFERRED] half, carries [VERIFY: none], sits outside the M0 gate and is
re-opened as the M3 lever D3.2, so the loads here stay plain per-CTA TMA and the
mainloop pipeline stays CTA-local (cta_layout_vmnk = (1,1,1,1)): switching them
on means cross-CTA empty-barrier signalling, which nothing in reach certifies
and which fails as a hang rather than as a number.

Host path (D0.17).  The harness times ONE conv2d_dgrad(dy, w) call between its
own event records, so every host-side action this function takes is inside the
measured number: wall = device + residual, and the kernel's own levers move
only the device term.  That residual is over budget -- the last read carrying a
profiler block put `median_ms x 1e6 - time_ns` at 294,500 - 210,848 = 83,652 ns
against the 24,040 ns this decision owns -- so the host path is where the gap
is, and D0.17 is what governs it.

D0.17 governs it in a fixed order: instrument, then route, then move exactly
the one bucket the ledger names.  The instrument is `notes/residual_probe.py`,
which takes the gate's own per-call window W and a pipelined window D in the
same process on the same commit and splits W - D into R_alloc / R_launch /
R_cache / R_graph / R_sync / R_other plus an added R_issue.  It landed and was
certified on GPU in the pure measurement round D0.17 C mandates as the first
commit; that round moved no bucket.

THIS round moves one, and it is R_launch: the alloc-to-issue prologue.  In the
plain path that prologue was, every call, three permutes, four strided parity
views of dx, six `from_dlpack` exports and the six-element argument list, then
the compile key and the dict probe -- MILESTONE D0.17 B item 5 ("per-call 的
python 侧参数打包") and item 3 ("每调用查询/重取 JIT 编译产物") verbatim, sitting
in a window whose whole non-kernel content measures 83,652 ns.  The other five
buckets are ruled out by construction rather than by preference: R_graph is 0
(this path has no capture or replay), R_cache is 1 (the compile key carries no
pointer, so it cannot miss in steady state), R_sync is the empty-body floor of
an event pair and cannot be tens of microseconds, and R_alloc is a warm
caching-allocator `torch.empty` -- which leaves R_launch and R_issue, and only
R_launch has a cure that does not go through CUDA graph capture, the machinery
this lineage already measured as a regression.

The move is a lookup keyed on the operand identity (base address, shape,
stride, dtype, device of both operands), holding the compiled artifact and the
packed arguments.  It carries one consequence that is stated rather than
hidden: a packed cute tensor owns the dlpack capsule of the torch tensor it
wraps, so caching the dx side of the pack pins that buffer.  Allocating a fresh
dx per call AND caching its arguments is therefore self-defeating -- every
previous dx stays alive, the allocator returns a new address every call, and
the cache misses every call, which is D0.17's own "R_cache < 1: the key carries
a per-call quantity" row and the most likely reading of the 0.2945 -> 0.3003 ms
that the un-ledgered iteration-6 argument cache produced.  So the output
buffers are owned by `_build` and rotated (`_DX_SLOTS` = 2, consecutive results
are distinct tensors), which is D0.17's R_alloc cure taken as the precondition
for the R_launch one.  The kernel writes every dx element exactly once (D0.1),
so a reused buffer needs no clear.

WHAT THIS ROUND'S LEDGER IS.  After the move the steady per-call window holds
exactly the key tuple, one dict get, and `fn(*args)` -- the terminal issue,
where the DSL packs the six-tensor ABI and re-encodes six tensormaps, since
`make_tma_atom` runs in the host half of the jit function.  So the harness's own
`residual_raw` before and after this commit reads out the moved buckets in the
gate's own units: before = R_alloc + R_launch + R_issue + R_sync = 83,652 ns
[MEASURED, iter 5 profiler block], after = R_issue + R_sync.  Their difference
is R_alloc + R_launch, measured by the instrument the gate itself uses rather
than by a self-reported print.  If that difference is inside the noise floor,
R_issue is the whole residual, the next round owns it, and the two host-path
levers left are the ones that reach inside the compiled artifact.

One note for the instrument: `notes/residual_probe.py` counts R_cache off
`_COMPILED`, which after this commit is consulted only on the cold path.  Its
R_cache term must be rebound to `_CALL` before it is run again.

-----------------------------------------------------------------------------
M0 GATE.  The M0 chart D0.1-D0.16 landed together; D0.17 owns the residual and
governs the host path, which this round changes and nothing else does.  Every
line below is a property of the device half -- launch geometry, smem partition,
register split, tile coverage, MAC count, TMA form -- and the device half is
byte-identical to the commit that measured 0.2945 ms: the diff of this round is
confined to `_build` / `conv2d_dgrad` and the module docstring, does not touch
`_dgrad_jit`, `_dgrad_kernel` or any layout/atom builder, and passes the same
six cute tensors to the same compiled artifact.  So the statically checkable
gate lines below are unchanged by construction rather than by re-derivation.

Two build repairs, both already committed, are what made the file runnable and
neither moved the pipeline's shape, order or hand-off.  (1) `range_constexpr` is
a preprocessor marker and is not rewritten inside a dynamic control-flow region,
so the three compile-time loops -- all of which sit inside the dynamic
producer/consumer split -- reached the runtime stub and raised; they are now
spelled as module-level tuple iteration (CLASS_IDX / CHAIN_IDX / EPI_RND_IDX).
(2) Pipeline positions were carried in mutable `PipelineState` objects advanced
inside the persistent tile loop, which made their fields SSA values defined
inside an scf.for region and read from a sibling region -- "operand #0 does not
dominate this use" at IR verification.  Positions are now the closed form of a
monotone k-block counter, constructed where used (see `state_at`).

notes/probe_f_whole_kernel.py cleared all nine of its rungs on the GPU against
the device half of this file: taps / coverage / budget / stream on the tables,
then pack, compile, launch, form, and numerics at
max|out - ref32| / max|ref32| = 0.00286152 against an fp32 nine-tap einsum
reference built inside the probe, versus the 2e-2 of §5.1.  That is the probe's
own reference, not the harness verdict.  The statically checkable half of the
gate closes below; the rest is read off the run.

  smem_dynamic  pipe 64 B | sA 4x32 KiB = 131,072 | sB 4x16 KiB = 65,536 |
                sD 4x8 KiB = 32,768, each 1024-aligned  ->  230,400 B, inside
                the [196,608, 232,448] window, and smem_static == 0 because
                every buffer comes from the SmemAllocator (D0.8).
  block_size    (1 + 2) x 128 = 384 (D0.6).      grid_size 132 (D0.12).
  registers     40x128 + 232x256 = 64,512 of 65,536 per CTA (D0.7).
  write-once    tile coverage is exact, not merely sufficient: CTA bidx walks
                ceil((784 - bidx)/132) tiles, which sums to 784 over the 132
                CTAs, so the four classes tile dx's four parity sublattices
                with no overlap and no remainder -- every dx element is the
                target of exactly one TMA store (D0.1).
  MAC count     4 classes x their own {1,2,2,4} taps, never 9-with-mask, so the
                tensor-pipe active time line is a per-class dense 1.000x by
                construction rather than by tuning (D0.2, the candidate-D trap).
  D0.11 form    the dy/w loads and the dx store are rank-4 `.tile` TMA, not the
                reference's 5D im2col descriptors.  This is not the SM-side
                gather fallback either: the stride-2 scatter is carried by the
                four parity-strided dx views plus the constexpr tap shifts, so
                no address expansion runs on the SM.  Nothing is claimed exempt
                on that account -- dram_bytes_read is measured as written.
  residual      `median_ms x 1e6 - time_ns <= 24,040 ns` is the one gate line
                that closes on neither static arithmetic nor a probe: it is a
                property of the timed window, so only a full harness run reads
                it.  It is open -- 83,652 ns at the last read carrying a
                profiler block -- and D0.17 owns it.  Per §7, a kernel whose
                mechanism lines all hold and whose `time_ns` is already under
                the reference's own NCU 255,296 ns [MEASURED, job 734407]
                routes any remaining end-to-end gap here and not to a kernel
                lever -- the M1 attribution rule says the same thing for the
                levers that come next.

-----------------------------------------------------------------------------
M1 GATE.  The M1 chart is complete: D1.1 (stage count S) with D1.2 (segment-A
wait depth) as one delivery, then D1.3 (k-loop unroll degree, first-block peel)
with D1.4 (producer/consumer register split) as the next.  This row is the gate
alone -- no decision, no functional change; the diff against the previous
commit is this docstring and nothing else.

WHAT M1 MOVED AND WHAT IT HELD.  Exactly one lever value changed across the
whole milestone: D1.2's segment-A wait depth, 1 -> 2 (WG_WAIT).  S stayed 4,
unroll stayed 1, peel stayed on, the split stayed (40, 232) -- each held
against its own trigger read off measured counters, not assumed; the D1.3 /
D1.4 paragraph above records both trigger evaluations, and D1.1's smem/stall
triggers were evaluated the same way.  No M1 rung touches the topology M0 Gate
froze: tile, WG division, exchange-surface atoms and the masking relations are
untouched by a wait depth.

B GROUP (architecture conformance), closable by arithmetic here because the two
rungs that could have moved a launch parameter -- D1.1 through smem, D1.4
through the split -- were both held:

  smem_dynamic  unchanged at 230,400 B.  D1.1's licence to drift this number is
                unused because S did not move; 230,400 <= 232,448 holds with
                margin and smem_static == 0 (every buffer from the allocator).
  cta_per_sm    1.  230,400 B exceeds half the 227 KiB per-SM allowance so two
                CTAs cannot co-reside, and grid 132 == SM count is what the
                persistent schedule wants (D0.12).
  block_size    384; grid_size 132; 40x128 + 232x256 = 64,512 regs/CTA, i.e.
                regs_per_thread 168, mid-window of the gate's [150, 180].
  MAC count     the parity decomposition is untouched.  A wait depth changes
                the order in which issued WGMMA groups are awaited, not how
                many issue, so the tensor-active-time line's input -- the MAC
                count -- is the same one M0 Gate measured.  No lever in M1 can
                move it; if it moves, that is a violation, not a drift.

C GROUP, read off this run.  The wait-depth move is a device change, so these
numbers cannot be carried over: the profiler block [MEASURED, iter 5: 0.2945 ms
end-to-end, time_ns 210,848, tensor_pct 68.69, stall_long_sb 4.14,
warp_active_pct 13.30] was taken at WG_WAIT = 1.  It is this gate's baseline,
not its evidence.  Measured against the M1 lines that baseline sat: end-to-end
0.2945 <= 0.36 ms; tensor_pct 68.69 >= 40; stall_long_sb 4.14 <= 9.30
(= 2.0 x 4.65 [MEASURED, job 734407]); tensor_pct/100 x time_ns = 144,832 <=
200,000 ns.  Whether depth 2 still clears all four is exactly what this row
measures.  Correctness is rigid and binary and is likewise read off the run.

THE LINE KNOWN TO BE OPEN, AND WHO OWNS IT.  `median_ms x 1e6 - time_ns <=
24,040 ns` read 83,652 ns at that same baseline.  It is not an M1 lever's line.
M1's own attribution rule states the routing in advance: with `time_ns <=
335,960` the gap belongs to D0.17, and 210,848 is under that bound by 1.59x, so
no further round may be spent on S, wait depth, unroll or the register split on
account of it.  D0.17 has since moved two buckets out of the timed window
(R_alloc and R_launch, at the commit immediately before D1.1+D1.2), and this
gate row is the first full harness run to carry a profiler block since that
move -- so its `residual_raw` is simultaneously this gate's residual line and
the after-reading of that move, whose before-reading is the 83,652 ns above.
"""

from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

# ---- problem constants (shape D) -------------------------------------------
N, C, H, W = 128, 256, 56, 56
K, R, S = 256, 3, 3
P, Q = 28, 28

# ---- architecture constants (M0) -------------------------------------------
BM, BN, BK = 256, 128, 64          # D0.5 tile
BPN, BP, BQ = 16, 4, 4             # (n, hh, ww) rectangle, BPN*BP*BQ = BM
EK = K // BK                       # 4 k-sub-blocks per tap
STAGE = 4                          # D0.8 starting point, D1.1 rung
EPI_M, EPI_N = 64, 64              # one chain x half the channels -> 8 KiB
EPI_STAGE = 2                      # epi ring depth per consumer warpgroup

# D1.2: segment-A tail wait depth (D0.14's `wg_wait`, rungs {0, 1, 2}).  It is
# the depth NUMBER only -- the issue order of D0.14 is frozen and untouched.
# The consumer holds WG_WAIT + 1 stages of the A/B ring (the k-block in flight
# plus the WG_WAIT it has not drained), so it shares D1.1's smem budget: the
# producer's run-ahead is STAGE - (WG_WAIT + 1) stages, which is why D1.1 and
# D1.2 are one atomic delivery.  Segment B (the tile tail before the epilogue)
# stays a full drain at depth 0 -- that is D0.14's contract, not a rung.
WG_WAIT = 2

# D1.3: k-loop unroll degree (rungs {1, 2}) and the first-block peel switch
# (rungs {on, off}).  Both are held at D0.16's starting point, 1 and on.
#
# UNROLL acts on ONE loop: the consumer's steady k-block loop.  The producer's
# per-tile issue sequence is len(taps)*EK <= 16 boxes of compile-time extent and
# is already fully unrolled at trace time, so the rung has no second call site.
#
# PEEL_FIRST is not independent of D1.2: PEEL = max(1, WG_WAIT), because a
# k-block cannot release a stage until WG_WAIT groups sit behind it, so the
# steady body's `consumer_release(count - WG_WAIT)` names a k-block of the
# PREVIOUS tile unless the first WG_WAIT are peeled.  Turning the peel off is
# therefore legal only at depth 0 -- asserted below rather than left implicit,
# since the illegal combination fails as a race, not as an error.  That shared
# knob is why D1.3 lands with D1.4 and reads against D1.2's delivery.
UNROLL = 1
PEEL_FIRST = True
PEEL = max(1, WG_WAIT) if PEEL_FIRST else 0
assert PEEL_FIRST or WG_WAIT == 0

WG_THREADS = 128
PRODUCER_WARPS = WG_THREADS // 32                   # warps 0-3 are the producer
NUM_CONSUMER_WG = 2
NTHREADS = (1 + NUM_CONSUMER_WG) * WG_THREADS       # 384  (D0.6)
PRODUCER_REGS, CONSUMER_REGS = 40, 232              # D0.7 split, D1.4 rung
# D1.4 rungs, all at the same CTA total of 64,512 of 65,536: (40, 232) is the
# starting point, {(32, 240), (48, 224), (24, 240)} the alternatives.  Held at
# (40, 232): the rung's trigger is spill over D0.15's upper line, and the split
# is what feeds the consumer's 128 accumulator words (D0.9) plus segment A's
# <= 30 non-accumulator words (D0.15) -- the same budget D1.3's peel depth draws
# on, which is why the two are one delivery.  Moving either direction spends a
# measured-zero problem: consumer-up would take registers from the producer that
# nothing shows it can spare, consumer-down compresses the segment-A allowance
# that D1.2's depth 2 just widened the live range of.
assert PRODUCER_REGS * WG_THREADS + CONSUMER_REGS * NUM_CONSUMER_WG * WG_THREADS \
    <= 65536
CHAINS_PER_WG = 2                                   # D0.9
NCHUNK = NUM_CONSUMER_WG * CHAINS_PER_WG            # 4 accumulator chains / CTA
CHAIN_M = BM // NCHUNK                              # 64 = the WGMMA atom M-mode
GRID = 132                                          # D0.12
CLUSTER = 2                                         # D0.13, cluster = (2,1,1)
assert GRID % CLUSTER == 0                          # 66 whole clusters

M_TILES = (N // BPN) * (P // BP) * (Q // BQ)        # 392
C_TILES = C // BN                                   # 2
TILES_PER_CLASS = M_TILES * C_TILES                 # 784
CLASSES = ((0, 0), (0, 1), (1, 0), (1, 1))
NSLOT = EPI_STAGE * NUM_CONSUMER_WG                 # 4 epi slots
EPI_ROUNDS = CHAINS_PER_WG * (BN // EPI_N)          # 4 rounds/WG, 8 boxes/tile
EPI_PIX = EPI_M // (BP * BQ)                        # 4 n-values per epi box

A_BOX = (BPN, BP, BQ, BK)
B_BOX = (BK, 1, 1, BN)
D_BOX = (EPI_PIX, BP, BQ, EPI_N)


def class_taps(h0, w0):
    """Parity class -> its tap list, as (dh, dw, r, s).

    p = hh + dh with r = h0 + 1 - 2*dh (same on the w axis), so a tap is a
    constexpr shift of the class grid, not a slide.  Tap counts: 1/2/2/4.
    """
    dhs = [0] if h0 == 0 else [1, 0]
    dws = [0] if w0 == 0 else [1, 0]
    return tuple((dh, dw, h0 + 1 - 2 * dh, w0 + 1 - 2 * dw)
                 for dh in dhs for dw in dws)


TAPS = tuple(class_taps(h0, w0) for h0, w0 in CLASSES)

# k-blocks per tile of each class ({1,2,2,4} taps x EK) and, since the persistent
# loop runs class after class, the position of a class's first k-block in the
# CTA's monotone k-block stream: count = ntile*KBASE[cls] + i*NKB[cls] + j.
NKB = tuple(len(t) * EK for t in TAPS)                  # (4, 8, 8, 16)
KBASE = tuple(sum(NKB[:i]) for i in range(len(NKB)))    # (0, 4, 12, 20)
assert sum(NKB) == 36 and all(n % STAGE == 0 for n in NKB)
# D1.1 x D1.2 joint budget: the consumer holds WG_WAIT + 1 stages, so the ring
# must leave the producer at least one to fill (deadlock line), and every tile
# must be long enough to contain the peel.
assert 0 <= WG_WAIT <= 2 and STAGE >= WG_WAIT + 2 and min(NKB) >= PEEL

# ---- compile-time loop indices ---------------------------------------------
# The ONLY spelling of "unroll me at trace time" that this kernel may use inside
# the @cute.kernel body: iterate one of these module-level tuples.
#
# `cutlass.range_constexpr` is a preprocessor MARKER, and the preprocessor does
# not rewrite it inside a dynamic control-flow region -- the marker then reaches
# its runtime stub and raises "range_constexpr should be preprocessed by
# preprocessor".  All three compile-time loops in this kernel sit inside the
# dynamic `if warp_idx ...` producer/consumer split (two of them additionally
# inside a dynamic `cutlass.range` body), which is exactly that case.
#
# A tuple iterator is nesting-independent, which is why it is the spelling that
# survives: it is not a marker at all, so it is left alone whether or not the
# preprocessor visits the region, and either way the body runs as ordinary
# Python at trace time -- fully unrolled, induction variable a real Python int.
# That last property is load-bearing where the variable indexes a Python
# container of traced values (`dsts[cls]`, `TAPS[cls]`, `accs[ch]`), which a
# variable degraded into a dynamic scf.for value could not do.  Plain
# `range(...)` is NOT interchangeable here: it IS a shape the preprocessor
# rewrites, into a dynamic loop, wherever it does run.
#
# Dynamic loops stay spelled `cutlass.range(...)` and are unaffected.
CLASS_IDX = tuple(range(len(CLASSES)))              # 4 parity classes
CHAIN_IDX = tuple(range(CHAINS_PER_WG))             # 2 accumulator chains / WG
EPI_RND_IDX = tuple(range(EPI_ROUNDS))              # 4 epilogue rounds / WG
PEEL_IDX = tuple(range(PEEL))                       # peeled k-blocks (D1.2)
TAIL_IDX = tuple(range(WG_WAIT))                    # stages the drain frees
# Depth 0 has nothing in flight to defer to, so each peeled k-block drains and
# releases in place.  Spelled as an empty/singleton tuple rather than a
# `const_expr` branch for the reason above: a marker inside this dynamic region
# is exactly the shape the preprocessor leaves alone.
PEEL_DRAIN = (0,) if WG_WAIT == 0 else ()


# ---- pipeline position: computed, never carried -----------------------------
# A `pipeline.PipelineState` holds three traced scalars (count, index, phase) and
# `advance()` MUTATES them, so the object is a loop-carried variable: after one
# trip through a `cutlass.range` body its fields are SSA values that were defined
# INSIDE that scf.for region.  Reading them afterwards -- in the next tile, or in
# the next parity class's loop -- is a use from a sibling region, which the MLIR
# verifier rejects with "operand #0 does not dominate this use" (the DSL can only
# carry a mutation it can SEE textually in the loop body it is rewriting; here the
# advance sat inside an undecorated helper, and even in the body it would have to
# survive the unrolled per-class loop nest above it).
#
# So this kernel never advances a state.  A circular buffer's position is a closed
# form of one monotone counter -- index = count % STAGE, and the phase bit flips
# once per lap -- and every count here is itself a closed form of `ntile`, the
# tile-loop induction variable and compile-time constants (KBASE / NKB / tap and
# k-sub-block offsets).  Each state is therefore CONSTRUCTED where it is used,
# out of values that dominate that point, and producer and consumer agree on which
# stage a k-block owns by computing the same formula rather than by stepping in
# lockstep.  `phase0` is read off a fresh `make_pipeline_state`, so the phase
# convention comes from the library rather than from an assumption here.
class _PState:
    """count/index/phase, the three fields the pipeline API reads off a state."""

    def __init__(self, stages, count, index, phase):
        self.stages = stages
        self.count = count
        self.index = index
        self.phase = phase


def state_at(phase0, count):
    """The state a fresh state of phase `phase0` reaches after `count` advances.

    Prefers the library's own `PipelineState` and falls back to `_PState`, but
    only after checking that the constructor stored the three fields where this
    call put them: a different argument order would otherwise be a silent
    stage/phase permutation (a hang or a race, not an error).
    """
    st = _PState(STAGE, count, count % STAGE, (phase0 + count // STAGE) % 2)
    real_cls = getattr(pipeline, "PipelineState", None)
    if real_cls is not None:
        try:
            real = real_cls(STAGE, st.count, st.index, st.phase)
        except Exception:                                        # noqa: BLE001
            return st
        if (real.count is st.count and real.index is st.index
                and real.phase is st.phase):
            return real
    return st


# =============================================================================
# layouts: one allocation per buffer, two views of the same bytes
# =============================================================================
def make_a_layouts(stage=STAGE):
    """A(dy): one allocation, three views of the same bytes, + the atom layout.

    `alloc_view` is the swizzled k-major `tile_to_shape` (BM, BK, stage).  A
    composition's inner layout indexes the outer one in COLEX order, so on the
    (BM, BK, stage) domain the logical index is m + k*BM + s*BM*BK regardless of
    alloc_view's own strides; the other two views are `cute.composition`
    transposers onto that index space and therefore address identical bytes:

      mma_view (CHAIN_M, BK, NCHUNK, stage)   m = chunk*CHAIN_M + m0
      tma_view (BPN, BP, BQ, BK, stage)       m = i0*BP*BQ + i1*BQ + i2

    `tma_one` is what the TMA atom builder gets: the SAME box composed onto a
    single-stage `tile_to_shape` rather than `cute.slice_` of `tma_view`.  The
    two denote the same layout -- stage 0 of the staged allocation is the
    unstaged one -- but only this spelling is certified.  `slice_` of a plain
    ComposedLayout is the KB recipe; `tma_view` is a composition OF a
    ComposedLayout, and slicing that is the construction the GPU rejected.

    The chunk mode exists because a WGMMA atom's M-mode is 64 while bM is 256:
    partitioning A over the (BM, BK) tensor would leave the four 64-row chunks
    in the fragment's M-rest mode, and picking one with an integer coordinate
    collapses the operand to rank 1 -- which `cute.gemm` rejects against a
    rank-3 accumulator ("`c` must have rank <= 2 when `a` has rank 1").
    The chunk mode is sliced off the SMEM TENSOR (a dynamic offset, `wg` and
    `ch` picking one 64-row chunk) BEFORE `partition_A`, so what the MMA sees
    is the (CHAIN_M, BK, stage) rank-3 tensor of the certified recipe and the
    fragment slice `(None, None, kk, stage)` is its rank-2 (V, M) operand.
    One fragment per chain = one smem descriptor per chain.
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, BK),
        cutlass.BFloat16)
    alloc_view = cute.tile_to_shape(atom, (BM, BK, stage), order=(0, 1, 2))
    mma_view = cute.composition(
        alloc_view,
        cute.make_layout((CHAIN_M, BK, NCHUNK, stage),
                         stride=(1, BM, CHAIN_M, BM * BK)))
    tma_view = cute.composition(
        alloc_view,
        cute.make_layout((BPN, BP, BQ, BK, stage),
                         stride=(BP * BQ, BQ, 1, BM, BM * BK)))
    tma_one = cute.composition(
        cute.tile_to_shape(atom, (BM, BK), order=(0, 1)),
        cute.make_layout(A_BOX, stride=(BP * BQ, BQ, 1, BM)))
    return alloc_view, mma_view, tma_view, tma_one


def make_b_layouts(stage=STAGE):
    """B(w): allocated as TMA must drop it (k rows, C contiguous), read MN-major.

    Allocation domain (BK, BN, stage) -> colex k + n*BK + s*BK*BN.
      TMA box (BK, 1, 1, BN, stage): r/s have extent 1, hence stride 0.
      WGMMA   (BN, BK, stage): MN-major, stride (BK, 1, BK*BN).
    `tma_one` is the same box on a single-stage allocation (see make_a_layouts).
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, BN),
        cutlass.BFloat16)
    alloc_view = cute.tile_to_shape(atom, (BK, BN, stage), order=(0, 1, 2))
    tma_view = cute.composition(
        alloc_view,
        cute.make_layout((BK, 1, 1, BN, stage), stride=(1, 0, 0, BK, BK * BN)))
    mma_view = cute.composition(
        alloc_view, cute.make_layout((BN, BK, stage), stride=(BK, 1, BK * BN)))
    tma_one = cute.composition(
        cute.tile_to_shape(atom, (BK, BN), order=(0, 1)),
        cute.make_layout(B_BOX, stride=(1, 0, 0, BK)))
    return alloc_view, tma_view, mma_view, tma_one


def make_epi_layouts(nslot=NSLOT):
    """D(dx): swizzled (EPI_M, EPI_N, nslot) for stmatrix + rank-4 box for TMA.

    EPI_N = 64 bf16 = 128 B is the SW128 atom width; a TMA box whose innermost
    extent disagrees with the descriptor's swizzle width silently misorders.
    The slot mode carries EPI_STAGE per consumer warpgroup, so the two stage
    their halves of a round concurrently under one rendezvous.
    `tma_one` is the same box on a single-slot allocation (see make_a_layouts).
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, EPI_N),
        cutlass.BFloat16)
    stsm_view = cute.tile_to_shape(atom, (EPI_M, EPI_N, nslot), order=(0, 1, 2))
    tma_view = cute.composition(
        stsm_view,
        cute.make_layout((EPI_PIX, BP, BQ, EPI_N, nslot),
                         stride=(BP * BQ, BQ, 1, EPI_M, EPI_M * EPI_N)))
    tma_one = cute.composition(
        cute.tile_to_shape(atom, (EPI_M, EPI_N), order=(0, 1)),
        cute.make_layout(D_BOX, stride=(BP * BQ, BQ, 1, EPI_M)))
    return stsm_view, tma_view, tma_one


def make_tma_atom(op, src, one, box, **kw):
    """`make_tiled_tma_atom` on the single-stage layout the builders hand back."""
    return cpasync.make_tiled_tma_atom(op, src, one, tuple(box), **kw)


def make_tiled_mma():
    """WGMMA 64x128x16 bf16/f32, A k-major, B MN-major (D0.3 / D0.9)."""
    return sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
        cutlass.Float32, (1, 1, 1),
        tiler_mn=(CHAIN_M, BN))


# =============================================================================
# device-side pieces (plain functions, traced into the kernel)
#
# These are UNDECORATED, so the DSL preprocessor never rewrites their source:
# it only transforms the AST of the @cute.kernel / @cute.jit body itself.  Two
# consequences are load-bearing here:
#   * `cutlass.range_constexpr` is a preprocessor marker, not a runtime object,
#     so in a helper it would reach its stub and raise.  A helper's compile-time
#     loop is therefore spelled with plain `range`, which here means the same
#     thing -- a Python-level loop executed while tracing, i.e. fully unrolled.
#     Note this is the OPPOSITE of the rule inside the kernel body, where
#     `range` is a shape the preprocessor rewrites into a dynamic scf.for and
#     compile-time loops must iterate a module-level tuple (see CLASS_IDX).
#     The two spellings are not interchangeable and neither one is universal.
#   * a branch on a DYNAMIC predicate cannot live in a helper at all (there is
#     no `if` to rewrite into a region).  Hence the epilogue is split into a
#     register half and a smem half with the `warp_idx` election between them,
#     in the kernel, where the preprocessor can see it.
# A third rule is about what these helpers may CONSTRUCT rather than how they
# loop, and it applies wherever they are called from a `cutlass.range` body:
# that body is traced into an scf.for region, and only scalar-ish values
# (Int32 / bool / pipeline state) can be yielded across its boundary.  So no
# helper reached from inside the tile loop builds a fragment or an mma atom --
# every fragment is allocated once in the kernel's setup and only mutated in
# place, ACCUMULATE is preset once on the shared tiled_mma, and the
# accumulators are cleared with `fill(0.0)` rather than with a fresh atom
# carrying ACCUMULATE=False.
# Library calls (cute.copy, pipeline methods, ...) are unaffected: they carry
# their own decorators and do not rely on the caller being preprocessed.
# =============================================================================
def decode_tile(tile):
    """tile in [0, 784) -> (i_n, i_p, i_q, c_tile), integer division only (D0.12).

    c_tile is the fastest axis, which is what makes the cluster pairing of D0.13
    hold: GRID is even and a CTA walks tile = bidx + i*GRID, so tile keeps the
    parity of bidx forever and tile // C_TILES is the same for bidx and bidx^1.
    Ranks 0 and 1 of a cluster therefore hold the two C-tiles of one M-tile at
    every step, sharing A(dy) and differing only in B(w).  The pair also runs the
    same number of steps: ntile switches from 6 to 5 at bidx = 124, an even
    boundary, so it never splits a cluster.
    """
    c_tile = tile % C_TILES
    m = tile // C_TILES
    i_q = m % (Q // BQ)
    m1 = m // (Q // BQ)
    i_p = m1 % (P // BP)
    i_n = m1 // (P // BP)
    return i_n, i_p, i_q, c_tile


def produce_tile(mainloop, phase0, count0, tma_a, mDy, tma_b, mW, sA_tma, sB_tma,
                 i_n, i_p, i_q, c_tile, taps):
    """Issue one tile's k-blocks: for each tap, EK boxes of A and B.

    `count0` is this tile's first k-block in the CTA's monotone k-block stream, so
    tap t's k-sub-block ks is k-block count0 + t*EK + ks and its stage/phase are
    that number's closed form (see `state_at`): no state is carried in or out.
    """
    for t in range(len(taps)):
        dh, dw, r, s = taps[t]
        # The whole stride-2 tap mechanism: shift P and Q INDEPENDENTLY by a
        # constexpr offset so each stays a bounds-checked descriptor axis and
        # the p == P halo is hardware zero.
        gA = cute.local_tile(cute.domain_offset((0, dh, dw, 0), mDy),
                             A_BOX, (i_n, i_p, i_q, None))
        gB = cute.local_tile(mW, B_BOX, (None, r, s, c_tile))
        tAs, tAg = cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA_tma, 0, 4), cute.group_modes(gA, 0, 4))
        tBs, tBg = cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB_tma, 0, 4), cute.group_modes(gB, 0, 4))
        for ks in range(EK):
            ps = state_at(phase0, count0 + (t * EK + ks))
            mainloop.producer_acquire(ps)
            bar = mainloop.producer_get_barrier(ps)
            cute.copy(tma_a, tAg[(None, ks)], tAs[(None, ps.index)],
                      tma_bar_ptr=bar)
            cute.copy(tma_b, tBg[(None, ks)], tBs[(None, ps.index)],
                      tma_bar_ptr=bar)
            mainloop.producer_commit(ps)


def make_mma_fragments(tiled_mma, tid_wg, wg, sA_mma, sB_mma):
    """Per-chain A fragments (one smem descriptor each) + the shared B fragment.

    The chunk mode of the A view is sliced off HERE, on the smem tensor, so
    `partition_A` sees the (CHAIN_M, BK, stage) rank-3 tensor of the certified
    recipe.  `wg` is dynamic; slicing a smem tensor by it is a pointer offset,
    not a register-array index (which would spill, warp_specialization gotcha 7).
    """
    thr_mma = tiled_mma.get_slice(tid_wg)
    tCrAs = [
        tiled_mma.make_fragment_A(
            thr_mma.partition_A(
                sA_mma[(None, None, wg * CHAINS_PER_WG + ch, None)]))
        for ch in range(CHAINS_PER_WG)
    ]
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_mma))
    acc_shape = thr_mma.partition_C(
        cute.make_identity_tensor((CHAIN_M, BN))).shape
    return thr_mma, tCrAs, tCrB, acc_shape


def issue_kblock(tiled_mma, tCrAs, tCrB, accs, stage_idx):
    """One k-block = 8 WGMMA (2 chains x 4 atom-K) under ONE commit_group.

    Both chains read the same smem stage, so they must land in the same group:
    `wait_group(WG_WAIT)` then retires a whole k-block at a time and its stage
    becomes releasable.  Splitting them would let the wait release a stage the
    second chain is still reading, at any depth.

    Operand profile is the (V,M) x (V,N) => (V,M,N) dispatch: each chain has its
    own rank-3 A fragment, so `(None, None, kk, stage)` is rank 2 (M-rest = 1)
    rather than the rank-1 operand `cute.gemm` rejects.

    `tiled_mma` is passed to `cute.gemm` DIRECTLY, with ACCUMULATE preset once
    at kernel setup.  Every call site of this helper is traced inside a dynamic
    `cutlass.range` body, and a per-call-site `cute.make_mma_atom` is not a
    yieldable type there (it raises at trace time); the fresh-atom
    ACCUMULATE False->True idiom is straight-line-only.  Zeroing is therefore
    `acc.fill(0.0)` in the peeled prologue, which is also what D0.16's
    accumulator-clear step names.
    """
    n_ksub = cute.size(tCrAs[0], mode=[2])
    warpgroup.fence()
    for ch in range(CHAINS_PER_WG):
        for kk in range(n_ksub):
            cute.gemm(tiled_mma, accs[ch],
                      tCrAs[ch][(None, None, kk, stage_idx)],
                      tCrB[(None, None, kk, stage_idx)], accs[ch])
    warpgroup.commit_group()


def consume_kblock(mainloop, phase0, count, tiled_mma, tCrAs, tCrB, accs):
    """One steady-state k-block: wait, issue, retire k-block `count - WG_WAIT`.

    The caller owns the loop (dynamic `cutlass.range` bodies live in the kernel
    itself, never in a helper) and the peel/drain around it.  `count` is this
    k-block's position in the CTA's monotone stream; the stage it consumes and the
    stage it hands back are both closed forms of it, so nothing is carried across
    the loop boundary.

    `wait_group(WG_WAIT)` leaves WG_WAIT groups in flight, so the newest stage a
    release may name is `count - WG_WAIT`: the groups still flying are reading
    the WG_WAIT stages above it, and handing one of those back would let the
    producer's TMA overwrite smem a live WGMMA is sourcing.  Releasing exactly at
    the retirement edge is also what makes the depth pay -- the stage goes back
    the instant its last reader retires, while later k-blocks keep the tensor
    pipe busy (KB sm90/wgmma_deferred_wait section 4).
    """
    cs = state_at(phase0, count)
    mainloop.consumer_wait(cs)
    issue_kblock(tiled_mma, tCrAs, tCrB, accs, cs.index)
    warpgroup.wait_group(WG_WAIT)    # retire everything older than WG_WAIT
    # ... then hand the retired k-block's stage back
    mainloop.consumer_release(state_at(phase0, count - WG_WAIT))


def epi_round_stage(tiled_r2s, tRS_sD, tRS_rD, tRS_rD_out, accs, wg, rnd):
    """Epilogue round, register half: acc -> bf16 -> stmatrix -> epi smem.

    Round rnd covers chain `rnd // 2` and channel half `rnd % 2`, and lands in
    this warpgroup's slot `rnd % EPI_STAGE` of the shared epi ring.  `wg` is
    dynamic, so the slot index is a pointer offset, not a register-array index.
    """
    size_rD = cute.size(tRS_rD)
    ch, nh = rnd // (BN // EPI_N), rnd % (BN // EPI_N)
    slot = wg * EPI_STAGE + rnd % EPI_STAGE
    tRS_rAcc = tiled_r2s.retile(accs[ch])
    for v in range(size_rD):
        tRS_rD[v] = tRS_rAcc[nh * size_rD + v]
    tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
    cute.copy(tiled_r2s, tRS_rD_out, tRS_sD[(None, None, None, slot)])


def epi_round_store(tma_d, mDx, sD_tma, rnd, i_n, i_p, i_q, c_tile):
    """Epilogue round, smem half: BOTH warpgroups' boxes out by rank-4 TMA.

    Called by one elected warp for the whole CTA, so it walks g = 0, 1 itself
    and reads the slot each warpgroup filled in `epi_round_stage`.
    """
    ch, nh = rnd // (BN // EPI_N), rnd % (BN // EPI_N)
    for g in range(NUM_CONSUMER_WG):
        gD = cute.local_tile(
            mDx, D_BOX,
            (i_n * (BPN // EPI_PIX) + g * CHAINS_PER_WG + ch,
             i_p, i_q, c_tile * (BN // EPI_N) + nh))
        bSs, bSg = cpasync.tma_partition(
            tma_d, 0, cute.make_layout(1),
            cute.group_modes(sD_tma, 0, 4), cute.group_modes(gD, 0, 4))
        cute.copy(tma_d, bSs[(None, g * EPI_STAGE + rnd % EPI_STAGE)], bSg)


def make_epi_copy(tiled_mma, tid_wg, sD_stsm):
    """The r2s three-layer construction + its staging fragments."""
    c_enum = LayoutEnum.ROW_MAJOR        # dx tile is (pixel, channel)
    copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
        c_enum, elem_ty_d=cutlass.BFloat16, elem_ty_acc=cutlass.Float32)
    copy_atom_C = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(c_enum.is_m_major_c(), 4),
        cutlass.BFloat16)
    tiled_r2s = cute.make_tiled_copy_S(
        copy_atom_r2s, cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma))
    thr_r2s = tiled_r2s.get_slice(tid_wg)
    tRS_sD = thr_r2s.partition_D(sD_stsm)
    rD_layout = cute.make_layout(cute.shape(thr_r2s.partition_S(sD_stsm))[:3])
    tRS_rD = cute.make_fragment(rD_layout.shape, cutlass.Float32)
    tRS_rD_out = cute.make_fragment(rD_layout.shape, cutlass.BFloat16)
    return tiled_r2s, tRS_sD, tRS_rD, tRS_rD_out


def make_shared_storage(a_l, b_l, epi_l):
    @cute.struct
    class SharedStorage:
        pipe: cute.struct.MemRange[cutlass.Int64, 2 * STAGE]
        sA: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_l)], 1024]
        sB: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_l)], 1024]
        sD: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_l)], 1024]

    return SharedStorage


# =============================================================================
# the kernel
# =============================================================================
@cute.kernel
def _dgrad_kernel(tma_a, mDy, tma_b, mW,
                  tma_d0, mDx0, tma_d1, mDx1, tma_d2, mDx2, tma_d3, mDx3,
                  tiled_mma,
                  a_mma_l: cute.ComposedLayout, a_tma_l: cute.ComposedLayout,
                  b_mma_l: cute.ComposedLayout, b_tma_l: cute.ComposedLayout,
                  epi_stsm_l: cute.ComposedLayout, epi_tma_l: cute.ComposedLayout,
                  SharedStorage: cutlass.Constexpr[Callable],
                  tx: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA_tma = storage.sA.get_tensor(a_tma_l.outer, swizzle=a_tma_l.inner)
    sA_mma = storage.sA.get_tensor(a_mma_l.outer, swizzle=a_mma_l.inner)
    sB_tma = storage.sB.get_tensor(b_tma_l.outer, swizzle=b_tma_l.inner)
    sB_mma = storage.sB.get_tensor(b_mma_l.outer, swizzle=b_mma_l.inner)
    sD_stsm = storage.sD.get_tensor(epi_stsm_l.outer, swizzle=epi_stsm_l.inner)
    sD_tma = storage.sD.get_tensor(epi_tma_l.outer, swizzle=epi_tma_l.inner)

    mainloop = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.pipe.data_ptr(),
        num_stages=STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        # every consumer warp calls consumer_release
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, NUM_CONSUMER_WG * WG_THREADS // 32),
        tx_count=tx,
        # CTA-local: this CTA's barriers are signalled only by its own producer
        # and its own consumers, so the cluster launch adds no cross-CTA
        # dependency here (D0.13's multicast half is D3.2's, see the docstring).
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
    epi_bar = pipeline.NamedBarrier(
        barrier_id=1, num_threads=NUM_CONSUMER_WG * WG_THREADS)

    # mbarrier init must be visible CTA-wide BEFORE the producer/consumer split
    # (the consumer's setmaxnreg .inc must not sit behind a barrier only the
    # producer reaches).  CTA-wide is also ENOUGH under the D0.13 cluster: no
    # peer CTA touches these barriers, so this stays sync_threads rather than a
    # cluster arrive/wait pair.
    cute.arch.sync_threads()

    # tiles this CTA owns within one class: tile = bidx + i*GRID
    ntile = (TILES_PER_CLASS - bidx + GRID - 1) // GRID

    if warp_idx < PRODUCER_WARPS:
        # ---------------- producer warpgroup (D0.6 / D0.7) ------------------
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        if warp_idx == 0:
            # phase convention comes from the library's own initial state; the
            # position within the ring is computed, never advanced (see state_at).
            p_phase0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, STAGE).phase
            for cls in CLASS_IDX:
                for i in cutlass.range(ntile, unroll=1):
                    p_n, p_p, p_q, p_c = decode_tile(bidx + i * GRID)
                    produce_tile(mainloop, p_phase0,
                                 ntile * KBASE[cls] + i * NKB[cls],
                                 tma_a, mDy, tma_b, mW,
                                 sA_tma, sB_tma, p_n, p_p, p_q, p_c, TAPS[cls])
    else:
        # ---------------- consumer warpgroups (D0.9 / D0.14 / D0.15) --------
        cute.arch.setmaxregister_increase(CONSUMER_REGS)
        wg = cute.arch.make_warp_uniform(tidx // WG_THREADS) - 1
        tid_wg = tidx - (wg + 1) * WG_THREADS      # consumer-internal thread id

        thr_mma, tCrAs, tCrB, acc_shape = make_mma_fragments(
            tiled_mma, tid_wg, wg, sA_mma, sB_mma)
        # ACCUMULATE is preset ONCE here, in straight-line code, and tiled_mma
        # goes to `cute.gemm` unwrapped: the k-blocks are issued from inside
        # dynamic `cutlass.range` bodies, where building a per-call-site mma
        # atom is not a yieldable type.  Zeroing is `fill(0.0)` per tile below.
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
        # 2 chains split along M (D0.9): 2 WG x 2 chains x 64 rows = bM 256.
        accs = [cute.make_fragment(acc_shape, cutlass.Float32)
                for _ in CHAIN_IDX]

        tiled_r2s, tRS_sD, tRS_rD, tRS_rD_out = make_epi_copy(
            tiled_mma, tid_wg, sD_stsm)
        store_pipe = pipeline.PipelineTmaStore.create(
            num_stages=EPI_STAGE,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, NUM_CONSUMER_WG * WG_THREADS))

        c_phase0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, STAGE).phase
        dsts = ((tma_d0, mDx0), (tma_d1, mDx1), (tma_d2, mDx2), (tma_d3, mDx3))

        for cls in CLASS_IDX:
            tma_d, mDx = dsts[cls]
            nkb = NKB[cls]
            for i in cutlass.range(ntile, unroll=1):
                c_n, c_p, c_q, c_c = decode_tile(bidx + i * GRID)
                # this tile's first k-block in the CTA's monotone k-block stream
                base = ntile * KBASE[cls] + i * nkb
                # first k-blocks peeled (D0.16 + D1.2): the accumulators are
                # cleared here, in the prologue, and the first WGMMA batches go
                # out before the steady-state loop is entered.  PEEL = max(1,
                # WG_WAIT) of them, because a k-block cannot release any stage
                # until WG_WAIT groups are behind it; at depth <= 1 this is
                # D0.16's single peeled k-block unchanged.
                for ch in CHAIN_IDX:
                    accs[ch].fill(0.0)
                for p in PEEL_IDX:
                    csp = state_at(c_phase0, base + p)
                    mainloop.consumer_wait(csp)
                    issue_kblock(tiled_mma, tCrAs, tCrB, accs, csp.index)
                    for _ in PEEL_DRAIN:
                        warpgroup.wait_group(0)
                        mainloop.consumer_release(csp)
                # steady k-block loop -- the one call site of D1.3's UNROLL rung.
                for j in cutlass.range(nkb - PEEL, unroll=UNROLL):
                    consume_kblock(mainloop, c_phase0, base + PEEL + j,
                                   tiled_mma, tCrAs, tCrB, accs)
                warpgroup.wait_group(0)   # tile tail is a full drain (D0.14 B)
                # the drain retired the WG_WAIT k-blocks the steady loop was
                # still holding; hand their stages back before the epilogue.
                for p in TAIL_IDX:
                    mainloop.consumer_release(
                        state_at(c_phase0, base + nkb - WG_WAIT + p))
                # epilogue (D0.10 / D0.14 C).  Fixed seven-step order per
                # round: r2s copy, generic->async proxy fence, rendezvous,
                # whole-warp TMA store + commit + throttle, rendezvous.  The
                # loop and the election live HERE, not in a helper, because
                # only a preprocessed body can branch on a dynamic warp_idx.
                for rnd in EPI_RND_IDX:
                    epi_round_stage(tiled_r2s, tRS_sD, tRS_rD, tRS_rD_out,
                                    accs, wg, rnd)
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta)
                    epi_bar.arrive_and_wait()
                    if warp_idx == PRODUCER_WARPS:   # first warp of consumer 0
                        epi_round_store(tma_d, mDx, sD_tma, rnd,
                                        c_n, c_p, c_q, c_c)
                        store_pipe.producer_commit()
                        store_pipe.producer_acquire()
                    epi_bar.arrive_and_wait()   # throttle seen before reuse
        store_pipe.producer_tail()


@cute.jit
def _dgrad_jit(mDy, mW, mDx0, mDx1, mDx2, mDx3):
    a_alloc_l, a_mma_l, a_tma_l, a_one = make_a_layouts()
    b_alloc_l, b_tma_l, b_mma_l, b_one = make_b_layouts()
    epi_stsm_l, epi_tma_l, epi_one = make_epi_layouts()
    tiled_mma = make_tiled_mma()

    tma_a, tDy = make_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mDy,
                               a_one, A_BOX)
    tma_b, tW = make_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mW,
                              b_one, B_BOX)
    tma_d0, tDx0 = make_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx0,
                                 epi_one, D_BOX)
    tma_d1, tDx1 = make_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx1,
                                 epi_one, D_BOX)
    tma_d2, tDx2 = make_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx2,
                                 epi_one, D_BOX)
    tma_d3, tDx3 = make_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), mDx3,
                                 epi_one, D_BOX)

    tx = (BM * BK + BN * BK) * 2                 # A + B bytes of ONE stage
    SharedStorage = make_shared_storage(a_alloc_l, b_alloc_l, epi_stsm_l)

    _dgrad_kernel(tma_a, tDy, tma_b, tW,
                  tma_d0, tDx0, tma_d1, tDx1, tma_d2, tDx2, tma_d3, tDx3,
                  tiled_mma, a_mma_l, a_tma_l, b_mma_l, b_tma_l,
                  epi_stsm_l, epi_tma_l, SharedStorage, tx).launch(
                      grid=[GRID, 1, 1], block=[NTHREADS, 1, 1],
                      cluster=[CLUSTER, 1, 1],      # D0.13; grid stays in CTAs
                      min_blocks_per_mp=1)


# =============================================================================
# host entry
# =============================================================================
_COMPILED = {}          # (shapes, dtype, device) -> compiled artifact
_CALL = {}              # operand identity -> [slot, fn, dx buffers, arg sets]
_CALL_CAP = 4           # distinct operand identities kept resident
_DX_SLOTS = 2           # rotating dx buffers: consecutive calls never alias


def _pack(dy, w, dx):
    """channels_last torch tensors -> the six cute tensors the kernel takes."""
    dxv = dx.permute(0, 2, 3, 1)                       # (N, H, W, C)
    args = [from_dlpack(dy.permute(0, 2, 3, 1), assumed_align=16),   # (N,P,Q,K)
            from_dlpack(w.permute(0, 2, 3, 1), assumed_align=16)]    # (K,R,S,C)
    for h0, w0 in CLASSES:
        args.append(from_dlpack(dxv[:, h0::2, w0::2, :], assumed_align=16))
    return args


def _build(dy, w):
    """Everything the per-call window used to rebuild, built once per identity.

    A packed cute tensor owns the dlpack capsule of the torch tensor it wraps,
    so caching the dx side of `_pack` PINS that buffer: allocating a fresh dx
    per call and caching its arguments would keep every previous dx alive, hand
    the allocator a new address every time, and miss the cache on every call.
    The buffers are therefore owned here and rotated, which is D0.17's own
    R_alloc cure (reuse the output buffer) taken because it is the precondition
    for the R_launch one, not as a second change.  `_DX_SLOTS` = 2 keeps two
    consecutive results distinct tensors; the kernel writes every dx element
    exactly once (D0.1), so a reused buffer needs no clear.
    """
    dxs, argss = [], []
    for _ in range(_DX_SLOTS):
        dx = torch.empty((N, C, H, W), device=dy.device, dtype=dy.dtype,
                         memory_format=torch.channels_last)
        dxs.append(dx)
        argss.append(_pack(dy, w, dx))
    ckey = (tuple(dy.shape), tuple(w.shape), dy.dtype, dy.device.index)
    fn = _COMPILED.get(ckey)
    if fn is None:
        fn = cute.compile(_dgrad_jit, *argss[0])
        _COMPILED[ckey] = fn
    return [0, fn, tuple(dxs), tuple(argss)]


def conv2d_dgrad(dy, w):
    # D0.17, bucket R_launch: the alloc-to-issue prologue is looked up, not
    # rebuilt.  The key carries base address, shape, stride, dtype and device of
    # both operands -- a key missing any of those would hand a stale descriptor
    # to a different tensor, which is a correctness fault, not a slow path.
    key = (dy.data_ptr(), w.data_ptr(), dy.shape, w.shape,
           dy.stride(), w.stride(), dy.dtype, w.dtype, dy.device.index)
    ent = _CALL.get(key)
    if ent is None:
        if len(_CALL) >= _CALL_CAP:
            _CALL.clear()
        ent = _CALL[key] = _build(dy, w)
    i = ent[0] ^ 1
    ent[0] = i
    ent[1](*ent[3][i])
    return ent[2][i]
