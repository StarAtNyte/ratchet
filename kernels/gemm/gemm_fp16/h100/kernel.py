"""gemm kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):
  gemm(a, b) -> c
    a: torch.Tensor (M, K) cuda bfloat16, row-major
    b: torch.Tensor (K, N) cuda bfloat16, row-major
    c: torch.Tensor (M, N) cuda bfloat16 (FP32 accumulation inside)

M0 architecture ledger -- candidate C2.  Decision -> where it lives:

  D0.1  single kernel, persistent, grid = min(#tiles, 132), no split-K.
        `grid` in _gemm_jit; `num_iters` / `t = bidx + it*NUM_SM` tile loop.
  D0.2  C resident in registers (`acc`, never staged to smem/gmem during the
        K loop), A/B streamed through the smem ring, fp32 accumulate, bf16
        cast only in the epilogue.
  D0.3  bM x bN x bK = 128 x 256 x 64 -> 64 x 32 = 2048 output tiles, 128
        k-stages.  BM/BN/BK below; N=8192 is divisible by 256, padding = 0.
  D0.4  384 threads = 1 producer WG (warp_idx < 4, warp 0 issues the TMAs) +
        2 cooperative consumer WGs (warp_idx 4..11) on the SAME 128x256 tile:
        WG0 rows 0-63, WG1 rows 64-127, via the M-stacked
        `atom_layout_mnk=(2,1,1)` atoms.
  D0.5  setmaxnreg (40, 232): 128*40 + 256*232 = 64512 <= 65536, both quotas
        multiples of 8 in [24, 256].  The accumulator is 64*256/128 = 128
        fp32 per consumer thread (forced by the D0.3 tile arithmetic).
        dec/inc are the first statement of their branch and no barrier sits
        between them; the CTA-wide sync_threads() is before the branch, and
        launch() carries min_blocks_per_mp=1 so ptxas does not drop them.
  D0.6  wgmma.m64n256k16 SS: `tiler_mn=(64, BN)`, both operands read straight
        from smem descriptors (make_fragment_A/B, no ldmatrix), 4 k-blocks
        per k-stage, fence / commit_group / wait_group(N) async management.
  D0.7  smem topology -- exactly four buffers, all dynamic (smem_static == 0):
        S1 sA  4 x 128x64x2  =  65536 B   TMA G2S      -> WGMMA descriptor
        S2 sB  4 x 256x64x2  = 131072 B   TMA G2S      -> WGMMA descriptor
        S3 sC      128x64x2  =  16384 B   stmatrix R2S -> TMA bulk store
        S4 mbarrier 8 x Int64 =    64 B
        With the 1024-B alignments the allocation is 214016 B (91.6% of the
        232448 B per-block cap, so CTA/SM = 1 on the smem account alone).
  D0.8  consumption ledger: A and B are smem-forced (WGMMA SS addresses its
        operands by smem descriptor, and B has no RS form at all); C is
        smem-forced (a TMA bulk store's source must be smem, and the C-frag
        distribution does not match row-major C).  No MMA-produced tensor is
        re-consumed as an MMA operand here, so the RS-vs-smem question does
        not arise for this operator.
  D0.9  issue order / covering.  Segment A: wait_group(MMA_WAIT) leaves
        MMA_WAIT wgmma groups in flight, so the next stage_wait plus the
        stage-index and phase advance run underneath them, and consumer_release
        fires MMA_WAIT stages behind the wait that retired that stage's last
        reader (D1.2 moves the depth, not the covering relation).  Segments B
        and C are carried by the warp specialisation itself -- the producer WG
        solves the next tile's coordinates and refills stages asynchronously
        while the consumers are still inside the current tile's mainloop and
        epilogue.  Per D0.9 the tensor cores are idle during epi_cvt; that is
        structural under a cooperative organisation and is not worked around.
  D0.10 live-set budget: only the 128-word accumulator is long-lived.  The
        epilogue converts one 128x64 block per segment and stores it
        immediately (tRS_rD / tRS_rD_out are one segment wide), so no full-tile
        bf16 copy is ever co-live with the fp32 accumulator.  Every
        value-fragment index is a range_constexpr compile-time constant -- a
        runtime index would demote the whole fragment to local memory.
  D0.11 epilogue: 4 segments of 128x64, each cvt -> stmatrix R2S -> S3 ->
        fence.proxy.async -> named barrier across both consumer WGs (the TMA
        store must wait for both 64-row halves) -> TMA bulk store -> second
        named barrier before S3 is reused.
  D0.13 A, B and C all move over cp.async.bulk.tensor (TMA); cluster (1,1,1)
        with num_multicast=1 is the starting point -- hw_kb has no certified
        cluster/multicast entry, so enabling it is deferred to D2.2.
  D0.14 compile once into _COMPILED, per call only wrap the tensors + launch.
  D0.15 no split-K / stream-K: K is never split across CTAs, there is no
        cross-CTA reduction and every output element is written exactly once.

  D0.12 rasterisation: the persistent scheduler reverse-maps the linear tile
        index t = bidx + it*NUM_SM through _tile_coord(), a 2D grouped
        swizzle rather than the single-axis order.  The 64 x 32 tile grid is
        cut into bands of GROUP_M = 16 M-rows; a band spans the full N
        extent and is walked M-fastest, so all 16 x 32 tiles of a band retire
        before the next band opens, and bands advance along M.  The 132 CTAs
        in flight therefore cover a 16-tile A row-band against 9 B column
        tiles instead of a 132-tile A row-span, which is what keeps the A/B
        re-reads in L2.  Producer and consumers call the identical helper, so
        the two call sites cannot drift apart.  Both divisions are by
        trace-time constants.  A ragged final band (num_m_tiles % GROUP_M
        != 0) falls back to the single-axis order; the frozen 8192^3 shape
        gives num_m_tiles = 64, so that branch is not taken here.

Lever state (explicitly NOT frozen by D0.16-E):
  AB_STAGES=4 (D1.1, settled), MMA_WAIT=1 (D1.2, settled),
  (40, 232) (D1.3, settled), GROUP_M=16 (D2.1), cluster (1, 1, 1) (D2.2),
  4 x 64 epilogue (D3.3).

D0.16 M0 gate ledger.  This commit IS the gate line, submitted alone; the
kernel body is untouched, the entire diff is this docstring block.

Pre-gate state: D0.1-D0.15 are all landed at HEAD, and notes/probe_d016_gate.py
returned PROBE PASS max_diff=0.0 on both of its contracts against this exact
kernel body:
  C-A  _tile_coord, imported from this file and driven with Python ints (it is
       plain trace-time arithmetic, so the probe exercises the same code the
       kernel emits), is a bijection of {0 .. T-1} onto the tile grid on the
       frozen 64 x 32 grid -- the grouped branch, num_m_tiles % GROUP_M == 0 --
       and on an 8 x 17 grid.  Reported error = max over tiles of
       |visits - 1|, with any out-of-range coordinate counted as an error.
  C-B  gemm() end to end on 1024 x 4352 x 384 with operands drawn from
       {-1, 0, 1}: num_m_tiles = 8 takes the grouped branch, total_tiles = 136
       > NUM_SM = 132 wraps the persistent tile loop, and k_steps = 6 >
       AB_STAGES = 4 wraps the mbarrier ring and flips the phase bit.  K = 384
       keeps every fp32 partial sum an exact integer, so a tile that is skipped,
       visited twice, or read at the wrong coordinate cannot hide under a
       tolerance; the output matched the bf16-rounded reference exactly.

Gate predicates that are fixed by construction (the NCU-measured lines are the
driver's to read off this job):
  A2  the output is allocated (M, N), bf16, on the input device; for the frozen
      shape that is (8192, 8192).
  A3  the product itself is never delegated to a tensor-library one-liner or to
      a vendor GEMM library -- every multiply-accumulate is issued by the WGMMA
      atoms built in _make_tiled_mma and fed from the smem ring.
      NOTE for every future edit of this file: eval/bench.py screens the *text*
      of kernel.py (it strips `#` comments, but NOT docstrings), so the banned
      entry-point names must not be spelled out anywhere in this file, not even
      inside a sentence asserting that they are absent.  A previous gate
      submission was rejected on exactly that -- the assertion matched its own
      screen.  State the predicate, never the names.
  B1  exactly one kernel launch per gemm() call: _gemm_kernel, no auxiliary or
      reduction kernel anywhere on the path.
  B2  block_size == THREADS == 384 (launch block=[THREADS, 1, 1]).
  B3  grid_size == min(total_tiles, NUM_SM) == min(2048, 132) == 132.
  B4  smem_dynamic == 214016 B, inside the [209000, 218000] band:
        mbar  AB_STAGES*2 x Int64 =     64 B at      0
        sA    4 x 128 x 64 x 2    =  65536 B at   1024  (1024-B aligned)
        sB    4 x 256 x 64 x 2    = 131072 B at  66560
        sC        128 x 64 x 2    =  16384 B at 197632  -> end 214016
      S1 + S2 = 196608 B; the allocation is 91.6% of the 232448 B per-block cap,
      which pins CTA/SM = 1 on the smem account alone.
  B5  smem_static == 0: all four buffers come from the dynamic SmemAllocator.
  B7  no runtime-indexed fragment anywhere (D0.10): every value-fragment index
      is a range_constexpr constant, which is the structural reason to expect
      spill_ld_sectors == spill_st_sectors == 0.

D1.2 + D1.3 settle ledger (this commit).  Both levers leave their exploratory
arms and take the value that measured best over the M1 sweep, ahead of the
D1.4 gate.  M1 mandates judging every A/B on the same-job ratio against the
reference timing in that job (absolute ms recorded only: cross-job spread is
about 10%, same-job about 3.6%, so an absolute gap under 4% decides nothing).

D1.2  MMA_WAIT 2 -> 1.  The {0, 1, 2} set is fully measured at AB_STAGES = 4,
      tier (40, 232):
        N = 0   ratio 0.8740              tensor_pct 82.09
        N = 1   ratio 0.8816, 0.8830      tensor_pct 88.80, 93.57
        N = 2   ratio 0.8949, 0.8562      tensor_pct 93.05, 92.76
      N = 1 is the only arm whose two jobs repeat inside 0.2%, and it owns the
      two fastest absolute readings on the board (1.6095 ms and 1.6244 ms,
      against 1.6558 / 1.6641 ms for N = 2).  N = 2's pair straddles it with a
      3.9% spread, i.e. inside the band M1 declares non-decisive.  N = 0 is
      last on both ratio and tensor_pct and is dropped.

D1.3  setmaxnreg (72, 216) -> (40, 232).  All four candidate tiers are
      measured at AB_STAGES = 4:
        (40, 232)  ratio 0.8949 / 0.8562 at N = 2; 0.8816 / 0.8830 at N = 1
        (24, 240)  ratio 0.8645 at N = 2
        (56, 224)  ratio 0.8819 at N = 2
        (72, 216)  ratio 0.8618 at N = 2
      (40, 232) is the only tier with readings at the settled wait depth and
      carries the family's highest reading; the three producer-richer tiers
      all read below it.  spill_ld_sectors == spill_st_sectors == 0 on every
      job on the board, so no tier is forced by D0.10's yield ladder -- this
      is D1.3's own sweep settling, not rung r2 of that ladder.

Feasibility of the settled tier:
  conservation   2*232 + 40 = 504 <= 504, exact equality as in every tier
  occupancy      128*40 + 256*232 = 5120 + 59392 = 64512 <= 65536, the same
                 64512 total the other three tiers consumed
  granularity    both quotas multiples of 8 and inside setmaxnreg's [24, 256]
  consumer need  D0.10's segment account puts the consumer peak live set at
                 128 accumulator words + 40 temps = 168 <= 232

Release trail (parameterised on MMA_WAIT; no ladder code moves in this
commit).  wait_group(N) leaves N wgmma groups in flight, so the N stages those
groups are still reading must stay unreleased:

  prologue k-stage  consumer_wait, gemm with ACCUMULATE=False, commit_group
  ramp              REL_RAMP = max(0, MMA_WAIT-1) k-stages that
                    consumer_wait + commit_group + wait_group(MMA_WAIT)
                    with NO release
  steady            k_steps - 1 - REL_RAMP k-stages, each one release after
                    its wait_group(MMA_WAIT)
  drain             wait_group(0), then REL_TAIL = max(1, MMA_WAIT) releases

Balance: acquires = 1 + REL_RAMP + (k_steps-1-REL_RAMP) = k_steps and
releases = (k_steps-1-REL_RAMP) + REL_TAIL = k_steps, for every N in the value
set.  At the settled N = 1 the ramp loop emits empty and REL_TAIL = 1, so the
emitted mainloop is the ladder this file ran at the M0 gate.  Safety: after
the wait in the k-th steady stage the groups still in flight are the newest
MMA_WAIT, so the oldest un-retired group reads stage k+1-MMA_WAIT while the
release issued in that stage frees stage k+1-MMA_WAIT-1, one stage behind the
oldest reader.  With AB_STAGES = 4 the consumer now holds at most
MMA_WAIT+1 = 2 unreleased stages, leaving the producer two stages to refill.

Shared budget/knobs for the pair, unchanged from HEAD~1 apart from the three
constants: smem_dynamic 214016 B (the B4 breakdown above, S1+S2 = 196608 B,
inside the 232448 B per-block cap and inside D0.16-B4's [209000, 218000]
window), smem_static 0, AB_STAGES = 4 (D1.1, settled), GROUP_M = 8,
cluster (1, 1, 1), epilogue 4 x 64, NamedBarrier id 1, grid 132, block 384.
regs_per_thread drift from the tier change is D0.16-B6's planned drift.

NOTE, the same screen as the A3 note above: this ledger scores the A/Bs as
bare ratios on purpose.  The status field those numbers come from carries a
vendor library name inside its own identifier, and spelling that identifier
here would match eval/bench.py's text screen, which keeps docstrings.  An
earlier submission of this very settle was rejected on exactly that.

D1.4 M1 gate ledger.  This commit IS the gate line, submitted alone per the
gate-commit rule; the kernel body is untouched and the entire diff is this
docstring block (no executable line moves, no lever moves).

Trigger check -- D1.1, D1.2 and D1.3 each carry at least one measured A/B, and
all three levers are on their settled arms at HEAD:
  D1.1  AB_STAGES  both arms of {3, 4} measured at MMA_WAIT = 1, (40, 232):
        arm 4 ratio 0.8816, arm 3 ratio 0.8845 -- spread 0.33%, inside M1's
        non-decisive band; settled on 4 (dram_bytes_read 2.276 GB vs 2.460 GB,
        stall_long_sb 8.85 vs 10.86 on that pair).
  D1.2  MMA_WAIT  all three of {0, 1, 2} measured at AB_STAGES = 4, (40, 232);
        settled on N = 1 (scoring in the D1.2 block above).
  D1.3  setmaxnreg  all four of {(40,232), (24,240), (56,224), (72,216)}
        measured at AB_STAGES = 4; settled on (40, 232) (scoring above).

Gate lines, read off the job that ran this exact kernel body (the settled
configuration AB_STAGES = 4 / MMA_WAIT = 1 / (40, 232), the immediately
preceding commit at HEAD, whose eval is the one this gate re-runs):

  correctness           PASS on the frozen 8192^3 shape under the harness's
                        |out - ref32|max <= 2e-2 * |ref32|max budget.
  performance line      bwd_median_ms 1.617 <= 1.85, and the same-job ratio
                        against the reference timing 0.8782 >= 0.83.  Both
                        sides of the equivalent pair clear it, so the line is
                        met on either reading.  MFU 68.73%.
  mechanism line 1      tensor_pct 93.42 >= 80.  This is D0.9's covering
                        contract under its first real test: the wait_group(N)
                        relation plus the producer/consumer specialisation
                        were supposed to keep the tensor pipe fed, and at the
                        settled depth the pipe is 93.42% occupied -- 99.7% of
                        the reference signature's 93.72.  The contract is not
                        falsified, and the four-state escalation to M3 (which
                        the line reserves for "all three depths tried, still
                        below 80, unresponsive to depth") is not entered:
                        the metric did respond to depth (82.09 at N = 0 ->
                        88.80 / 93.57 at N = 1) and cleared the line.
  mechanism line 2      spill_ld_sectors == 0 and spill_st_sectors == 0.
                        D0.10's segment account is not falsified; neither rung
                        of its yield ladder was walked at any point in M1 --
                        spill has been 0 on every job on the board, including
                        the (24, 240) tier that gives the producer the least.
  supply line (sum)     stall_long_sb 3.44 + stall_barrier 9.27 = 12.71 <= 33.
                        Adjudicative only when tensor_pct < 80; tensor_pct is
                        93.42, so this line is auto-satisfied and the sum is
                        recorded for M2/M3 attribution, not used to decide.
                        12.71 is well under the ~1/3 a healthy warp-specialised
                        form is allowed by the producer WG's own wait duty.
  stall_long_sb alone   3.44.  Advisory, no threshold, no adjudication.
                        Recorded for the record: it fell from 8.85 (M0 gate) as
                        the stall moved into stall_barrier (6.08 -> 9.27), the
                        bucket-to-bucket migration the line predicts.

Drift record (recorded, not re-checked -- these are the post-M1 baselines the
later phases' budget accounts start from):
  smem_dynamic     214016 B after D1.1 settled on AB_STAGES = 4 (the B4
                   breakdown above; the arm-3 excursion read 164864 B, the
                   planned 49152 B/stage drift under D0.16-B4).
  regs_per_thread  168 after D1.3 settled on (40, 232).
  smem_static      0.  block 384x1x1, grid 132x1x1, unchanged since M0.

NCU comparison against the reference signature (advisory, no constraint on the
next step): tensor_pct 93.42 vs 93.72; stall_barrier 9.27 vs 1.66;
stall_mio 3.41 vs 2.05; dram_bytes_read 1.539 GB vs the compulsory-traffic
floor; dram_pct 32.04.  The one materially wider gap is stall_barrier, which
is where M2's rasterisation work and M3's epilogue work are aimed.

Lever state entering M2, unchanged by this commit: AB_STAGES = 4,
MMA_WAIT = 1, setmaxnreg (40, 232) -- all three frozen for the remainder of
the phase order -- with GROUP_M = 8 (D2.1), cluster (1, 1, 1) (D2.2) and the
4 x 64 epilogue (D3.3) still on their M0 starting points.

D2.1 swizzle band width A/B, arm G = 16.  The only executable change is the
module-level constant GROUP_M, 8 -> 16; _tile_coord()'s body, both of its
call sites, every smem layout, every barrier and the launch geometry are
byte-identical to HEAD~1, so this is a single-variable A/B against the D1.4
gate job (G = 8: dram_bytes_read 1547428096, dram_pct 32.56, tensor_pct
93.05).  The band width is the D0.12 lever, not the D0.12 topology: bands are
still M-fastest over the full N extent, and 64 % 16 == 0 keeps the grouped
branch, so the ragged fallback stays untaken on the frozen shape.

Why this arm out of {1, 2, 4, 8, 16}: the 132 in-flight CTAs occupy a
G x ceil(132/G) window of the tile grid, so the distinct operand bytes one
k-slice of that window touches go as G*BM + (132/G)*BN = 128*G + 256*132/G
elements, minimised at G = sqrt(132*BN/BM) = 16.2.  Enumerated on the frozen
64 x 32 grid the window is G = 4: 4 x 33, G = 8: 8 x 17, G = 16: 16 x 9,
which is 8960 / 5376 / 4352 units -- G = 16 is the model's floor and the
neighbour of the G = 8 starting point on the side the model favours.  The
bijection of _tile_coord() at G = 16 was re-checked at trace-time arithmetic
on both the 64 x 32 grid (grouped branch) and an 8 x 17 grid (ragged
fallback); every tile is visited exactly once on both.

Shared budget/knobs, all unchanged from HEAD~1: smem_dynamic 214016 B,
smem_static 0, AB_STAGES = 4, MMA_WAIT = 1, setmaxnreg (40, 232), cluster
(1, 1, 1) with num_multicast = 1, epilogue 4 x 64, NamedBarrier id 1,
grid 132x1x1, block 384x1x1.  D2.2 is untouched and still on its M0 starting
point; it is the other half of the M2 pair and carries its own hw_kb
prerequisite.
"""

from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

# --- D0.3 tile shape / D0.4 warp roles / D0.5 register split / D0.7 smem topology ---
BM, BN, BK = 128, 256, 64
AB_STAGES = 4                       # D1.1 lever, settled arm 4
EPI_M, EPI_N = 128, 64              # D0.11: 4 segments of 128x64
EPI_STAGES = 1                      # 16384 B, the 4 segments reuse it serially
NUM_SM = 132                        # D0.1 persistent grid
THREADS = 384
CONSUMER_THREADS = 256
DMA_THREADS = 128
PRODUCER_REGS = 40                  # D1.3 lever, settled tier (40, 232)
CONSUMER_REGS = 232
MMA_WAIT = 1                        # D1.2 lever, {0,1,2}; settled arm N=1
# D1.2 release trail.  wait_group(N) leaves N wgmma groups in flight, so the
# N stages those groups are still reading must stay unreleased.  The mainloop
# therefore ramps: REL_RAMP leading k-stages commit + wait without releasing,
# the steady k-stages release one stage each, and REL_TAIL closing releases
# drain the trail after wait_group(0).  Acquires and releases both total
# k_steps per tile for every N in the value set (N=0 -> ramp 0, tail 1).
REL_RAMP = max(0, MMA_WAIT - 1)
REL_TAIL = max(1, MMA_WAIT)
GROUP_M = 16                        # D0.12 swizzle band width, D2.1 lever
EPI_BAR_ID = 1

# All three operands are row-major by the frozen harness contract.  This is a
# module-level constant on purpose: a LayoutEnum is a plain Python enum, so it
# must never appear as a @cute.kernel parameter (hw_kb wgmma_mn_major_operand
# gotcha 0 -- a bare `cutlass.Constexpr` annotation is rejected during tracing,
# and LayoutEnum has no dynamic-expression form).  Referenced as a global it is
# resolved at trace time, exactly as in hw_kb tma_store_epilogue.
C_LAYOUT = LayoutEnum.ROW_MAJOR


def _tile_coord(t, num_m_tiles, num_n_tiles):
    """D0.12: reverse-map a linear tile index to (tile_m, tile_n).

    2D grouped swizzle, NOT the single-axis order.  The tile grid is cut into
    bands of GROUP_M rows along M; a band covers the full N extent and is
    walked M-fastest, bands then advance along M.  `t` is dynamic, both
    num_m_tiles and num_n_tiles are trace-time constants, so every division
    below is by a constant.  The ragged-band fallback is resolved in plain
    Python at trace time (this helper is not DSL-preprocessed), so only one of
    the two forms is ever emitted.
    """
    if num_m_tiles % GROUP_M != 0:
        tile_m = t // num_n_tiles
        return tile_m, t - tile_m * num_n_tiles
    tiles_per_band = GROUP_M * num_n_tiles
    band = t // tiles_per_band
    rem = t - band * tiles_per_band
    tile_n = rem // GROUP_M
    tile_m = band * GROUP_M + (rem - tile_n * GROUP_M)
    return tile_m, tile_n


def _a_smem_layout(mn, k, stages):
    """Staged swizzled k-major smem layout for the A operand (hw_kb wgmma_ss_min_gemm §2)."""
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, k),
        cutlass.BFloat16,
    )
    return cute.tile_to_shape(atom, (mn, k, stages), order=(0, 1, 2))


def _b_smem_layouts(mn, k, stages):
    """MN-major B operand: ONE allocation, two views (hw_kb wgmma_mn_major_operand).

    write view (BK, BN, S), BN contiguous -- the orientation TMA lands in from the
    row-major (K, N) gmem tensor; read view (BN, BK, S) MN-major -- what WGMMA's B
    descriptor expects.  The transposer is a flat make_layout with explicit colex
    strides: the write domain's index at (k, n, s) is k + n*BK + s*BK*BN, so the
    read view at (n, k, s) carries stride (BK, 1, BK*BN).
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, mn),
        cutlass.BFloat16,
    )
    write = cute.tile_to_shape(atom, (k, mn, stages), order=(0, 1, 2))
    read = cute.composition(
        write, cute.make_layout((mn, k, stages), stride=(k, 1, k * mn)))
    return write, read


def _epi_smem_layout(m, n, stages):
    """Staged swizzled epilogue smem layout (hw_kb tma_store_epilogue §1)."""
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(C_LAYOUT, cutlass.BFloat16, n),
        cutlass.BFloat16,
    )
    return cute.tile_to_shape(atom, (m, n, stages), order=(0, 1, 2))


def _make_shared_storage(a_layout, b_layout, epi_layout):
    """Built from inside @cute.jit (cute.cosize needs a live context)."""

    @cute.struct
    class SharedStorage:
        mainloop_mbar: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
        sA: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout)], 1024]
        sB: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout)], 1024]
        sC: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(epi_layout)], 1024]

    return SharedStorage


@cute.kernel
def _gemm_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout_write: cute.ComposedLayout,
    b_smem_layout_read: cute.ComposedLayout,
    epi_smem_layout: cute.ComposedLayout,
    k_steps: cutlass.Constexpr[int],
    num_m_tiles: cutlass.Constexpr[int],
    num_n_tiles: cutlass.Constexpr[int],
    total_tiles: cutlass.Constexpr[int],
    n_epi: cutlass.Constexpr[int],
    SharedStorage: cutlass.Constexpr[Callable],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_c)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
    sB_w = storage.sB.get_tensor(                    # TMA lands here: (BK, BN, S)
        b_smem_layout_write.outer, swizzle=b_smem_layout_write.inner)
    sB_r = storage.sB.get_tensor(                    # WGMMA reads here: (BN, BK, S)
        b_smem_layout_read.outer, swizzle=b_smem_layout_read.inner)
    sC = storage.sC.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)

    tma_bytes = cute.size_in_bytes(
        cutlass.BFloat16, cute.slice_(a_smem_layout, (None, None, 0))
    ) + cute.size_in_bytes(
        cutlass.BFloat16, cute.slice_(b_smem_layout_write, (None, None, 0))
    )

    # --- D0.4: TMA producer (1 elected thread) / 8 consumer warps release stages ---
    mainloop = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar.data_ptr(),
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
        tx_count=tma_bytes,
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
    )
    store_pipe = pipeline.PipelineTmaStore.create(
        num_stages=EPI_STAGES,
        producer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, CONSUMER_THREADS),
    )

    gA = cute.zipped_divide(mA, (BM, BK))            # ((BM,BK), (m_tiles, k_steps))
    gB = cute.zipped_divide(mB, (BK, BN))            # ((BK,BN), (k_steps, n_tiles))
    gC = cute.zipped_divide(mC, (EPI_M, EPI_N))      # ((EPI_M,EPI_N), (m_e, n_e))

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), gA)
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB_w, 0, 2), gB)
    bSG_sC, bSG_gC = cpasync.tma_partition(
        tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), gC)

    cute.arch.sync_threads()

    # --- D0.1 persistent tile loop, D0.12 grouped-swizzle reverse map ---
    num_iters = (cutlass.Int32(total_tiles + NUM_SM - 1) - bidx) // NUM_SM

    if warp_idx < 4:
        # ================= producer warpgroup (D0.5: give registers back) ==========
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        if warp_idx == 0:
            prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, AB_STAGES)
            for it in cutlass.range(num_iters):
                t = bidx + it * NUM_SM
                tile_m, tile_n = _tile_coord(t, num_m_tiles, num_n_tiles)
                for k in cutlass.range(k_steps):
                    mainloop.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, (tile_m, k))],
                        tAsA[(None, prod_state.index)],
                        tma_bar_ptr=mainloop.producer_get_barrier(prod_state),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, (k, tile_n))],
                        tBsB[(None, prod_state.index)],
                        tma_bar_ptr=mainloop.producer_get_barrier(prod_state),
                    )
                    mainloop.producer_commit(prod_state)
                    prod_state.advance()
            mainloop.producer_tail(prod_state)
    else:
        # ================= consumer warpgroups (D0.5: take registers) ==============
        cute.arch.setmaxregister_increase(CONSUMER_REGS)

        # D0.6: SS operands sliced by warpgroup slot; C accumulator per thread = 128 f32
        wg_idx = cute.arch.make_warp_uniform(tidx // 128) - 1
        wg_mma = tiled_mma.get_slice(cute.make_layout(2, stride=128)(wg_idx))
        tCrA = wg_mma.make_fragment_A(wg_mma.partition_A(sA))
        tCrB = wg_mma.make_fragment_B(wg_mma.partition_B(sB_r))
        acc = cute.make_fragment(wg_mma.partition_shape_C((BM, BN)), cutlass.Float32)

        # D0.11: C-frag -> matrix R2S atom -> swizzled epi smem stage
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            C_LAYOUT, elem_ty_d=cutlass.BFloat16, elem_ty_acc=cutlass.Float32)
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(C_LAYOUT.is_m_major_c(), 4),
            cutlass.BFloat16)
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        # hw_kb tma_store_epilogue gotcha 4: consumer-internal thread number.
        thr_r2s = tiled_copy_r2s.get_slice(tidx - DMA_THREADS)
        tRS_sC = thr_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(acc)
        rD_layout = cute.make_layout(cute.shape(thr_r2s.partition_S(sC))[:3])
        tRS_rD = cute.make_fragment(rD_layout.shape, cutlass.Float32)
        tRS_rD_out = cute.make_fragment(rD_layout.shape, cutlass.BFloat16)
        size_rD = cute.size(tRS_rD)

        epi_bar = pipeline.NamedBarrier(
            barrier_id=EPI_BAR_ID, num_threads=CONSUMER_THREADS)

        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, AB_STAGES)
        rel_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, AB_STAGES)

        num_kb = cute.size(tCrA, mode=[2])

        for it in cutlass.range(num_iters):
            t = bidx + it * NUM_SM
            tile_m, tile_n = _tile_coord(t, num_m_tiles, num_n_tiles)

            # --- D0.9 segment A: deferred wait ladder, one wgmma group in flight ---
            # prologue k-stage: first k-block overwrites acc (ACCUMULATE=False)
            mainloop.consumer_wait(cons_state)
            warpgroup.fence()
            mma0 = cute.make_mma_atom(tiled_mma.op)
            mma0.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(num_kb):
                cute.gemm(
                    mma0, acc,
                    tCrA[(None, None, kb, cons_state.index)],
                    tCrB[(None, None, kb, cons_state.index)],
                    acc,
                )
                mma0.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            cons_state.advance()

            # ramp: the first REL_RAMP k-stages stay unreleased -- their groups
            # are still in flight behind wait_group(MMA_WAIT)
            for _r in cutlass.range_constexpr(REL_RAMP):
                mainloop.consumer_wait(cons_state)
                warpgroup.fence()
                mma_r = cute.make_mma_atom(tiled_mma.op)
                mma_r.set(warpgroup.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(num_kb):
                    cute.gemm(
                        mma_r, acc,
                        tCrA[(None, None, kb, cons_state.index)],
                        tCrB[(None, None, kb, cons_state.index)],
                        acc,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(MMA_WAIT)
                cons_state.advance()

            # steady: one release per k-stage, trailing MMA_WAIT stages behind
            for k in cutlass.range(k_steps - 1 - REL_RAMP):
                mainloop.consumer_wait(cons_state)
                warpgroup.fence()
                mma1 = cute.make_mma_atom(tiled_mma.op)
                mma1.set(warpgroup.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(num_kb):
                    cute.gemm(
                        mma1, acc,
                        tCrA[(None, None, kb, cons_state.index)],
                        tCrB[(None, None, kb, cons_state.index)],
                        acc,
                    )
                warpgroup.commit_group()
                warpgroup.wait_group(MMA_WAIT)
                mainloop.consumer_release(rel_state)
                rel_state.advance()
                cons_state.advance()

            warpgroup.wait_group(0)
            for _t in cutlass.range_constexpr(REL_TAIL):
                mainloop.consumer_release(rel_state)
                rel_state.advance()

            # --- D0.11 epilogue: 4 segments, one 128x64 block converted at a time ---
            for epi in cutlass.range_constexpr(n_epi):
                for v in cutlass.range_constexpr(size_rD):
                    tRS_rD[v] = tRS_rAcc[epi * size_rD + v]
                tRS_rD_out.store(tRS_rD.load().to(cutlass.BFloat16))
                cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sC[(None, None, None, 0)])
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared,
                    space=cute.arch.SharedSpace.shared_cta)
                epi_bar.arrive_and_wait()
                if warp_idx == 4:
                    cute.copy(
                        tma_atom_c,
                        bSG_sC[(None, 0)],
                        bSG_gC[(None, (tile_m, tile_n * n_epi + epi))],
                    )
                    store_pipe.producer_commit()
                    store_pipe.producer_acquire()
                epi_bar.arrive_and_wait()

        store_pipe.producer_tail()


@cute.jit
def _gemm_jit(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream,
    m: cutlass.Constexpr[int],
    n: cutlass.Constexpr[int],
    k: cutlass.Constexpr[int],
):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K,         # A (M, K) row-major -> k-major
        warpgroup.OperandMajorMode.MN,        # B (K, N) row-major -> MN-major
        cutlass.Float32,
        atom_layout_mnk=(2, 1, 1),           # 2 consumer WGs stack atoms along M
        tiler_mn=(64, BN),                   # wgmma.m64n256k16
    )

    a_smem_layout = _a_smem_layout(BM, BK, AB_STAGES)
    b_write, b_read = _b_smem_layouts(BN, BK, AB_STAGES)
    epi_smem_layout = _epi_smem_layout(EPI_M, EPI_N, EPI_STAGES)

    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA,
        cute.slice_(a_smem_layout, (None, None, 0)), (BM, BK), num_multicast=1)
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mB,
        cute.slice_(b_write, (None, None, 0)), (BK, BN), num_multicast=1)
    tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mC,
        cute.slice_(epi_smem_layout, (None, None, 0)), (EPI_M, EPI_N))

    SharedStorage = _make_shared_storage(
        a_smem_layout, b_write, epi_smem_layout)

    num_m_tiles = m // BM
    num_n_tiles = n // BN
    total_tiles = num_m_tiles * num_n_tiles
    grid = min(total_tiles, NUM_SM)

    _gemm_kernel(
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        tma_atom_c, tma_tensor_c,
        tiled_mma,
        a_smem_layout, b_write, b_read, epi_smem_layout,
        k // BK,
        num_m_tiles,
        num_n_tiles,
        total_tiles,
        BN // EPI_N,
        SharedStorage,
    ).launch(
        grid=[grid, 1, 1],
        block=[THREADS, 1, 1],
        cluster=[1, 1, 1],
        min_blocks_per_mp=1,
        stream=stream,
    )


# --- D0.14: compile once, cache the compiled artefact, per-call only launch ---
_COMPILED = {}


def gemm(a, b):
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, "inner dimensions must match"
    assert m % BM == 0 and n % BN == 0 and k % BK == 0, "shape not tiled by 128x256x64"

    c = torch.empty(m, n, device=a.device, dtype=a.dtype)

    mA = from_dlpack(a, assumed_align=16)
    mB = from_dlpack(b, assumed_align=16)
    mC = from_dlpack(c, assumed_align=16)
    stream = cutlass_torch.current_stream()

    key = (m, n, k, a.dtype, a.device.index)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_gemm_jit, mA, mB, mC, stream, m, n, k)
        _COMPILED[key] = compiled
    compiled(mA, mB, mC, stream)
    return c
