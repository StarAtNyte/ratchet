# ew_segment_latency_schedule (intra-segment scheduling of an elementwise segment: who owns it, and what buys it)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **orthogonal dimension** of [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md):
> that one governs "which instructions, and how many, this segment uses"; this one governs "in what **order and
> spacing** the same batch of instructions is issued", and — this entry's core conclusion — **that order is not
> yours to control; it belongs to ptxas, and ptxas buys it with register headroom**. The register ledger itself
> is in [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md).

## Purpose

When an elementwise block **is waiting on its own dependencies** (the NCU stall profile is dominated by
`short_scoreboard` + `mio_throttle`, or — when the segment has no MIO producers — `wait` alongside
`dispatch_stall`/`misc`), rather than dominated by `not_selected` + `math_pipe_throttle` (issuing), answer two
questions:

1. **Whose problem is it** — "the source wrote producers right next to consumers", or "ptxas has no headroom to
   pull them apart"?
2. **Which knob to turn** — this entry gives machine-read criteria, and gives **the only knob that actually
   works**.

The hardware has three different things to deal with. The first two share the MIO path on SM90; the third is not on
MIO at all, and its criteria surface is completely different:

| Category | Who is on this path | Dominant NCU stall when it fails | Who can move it |
|---|---|---|---|
| **Latency**: the result doesn't come back that fast | results of `LDS`/`LDSM`/`STS` (shared) and `MUFU.*` (SFU: `EX2`/`RCP`/`SQRT`…) are accounted by the **short scoreboard** | `stall_short_scoreboard` | **ptxas's list scheduler**, at the cost of registers (this entry) |
| **Throughput**: this path can't issue fast enough | same as above; the MIO instruction queue is full | `stall_mio_throttle` | reduce the **instruction count** of the MIO class / switch paths (see the two `cfrag_*` entries) |
| **Fixed latency**: the result takes a few cycles to become usable | `FFMA`/`FMUL`/`FADD`/`IMAD`/`IADD3`/`LEA`/`LOP3`/`SHF`/`MOV`/`PRMT`… — not on MIO, no scoreboard, dependencies accounted by **stall counts hard-coded at compile time** | `stall_wait` | **the same knob**: ptxas's list scheduling + the source's **in-flight width** (§(a′) + §(d1)) |

The third row is this entry's **second producer family**, with a criteria surface completely different from the
first two rows: it is in neither `short_scoreboard` nor `mio_throttle`; NCU records it under `wait`. **The
definition of `wait` is "waiting on a fixed-latency execution dependency", not "waiting on MMA"** — the two share
one counter and cannot be separated by stall category, only by bucketing in the segment ledger (the
differential clause of item 2 of "When needed"). For an elementwise block of pure FMA / integer bookkeeping /
address arithmetic, `short_scoreboard` and `mio_throttle` are both single digits, and looking for it through the
entries of the first two rows **will certainly not find it**.

If a warp issues `MUFU.EX2` and **immediately** issues its consumer, it has to stand and wait out the whole SFU
latency; if there are several unrelated instructions in between, the latency is filled with its own work. The
only free independent work within a segment is the segment's other elements — but **interleaving them is something
ptxas does by itself, and more thoroughly than source order**: it has to hold the intermediate values of `k`
in-flight elements at once, so the interleaving depth is priced directly by **the number of available registers**.

**But ptxas only reorders and renames; it does not change the dependency graph.** The upper bound of what it can
interleave is the number of element chains that **are already parallel** in the source — this entry calls that
quantity the **in-flight width `W`** (the number of mutually independent element chains open at the same time in
the segment). `W` is a property of the source; the compiler cannot invent it: if the source strings a segment's
elements onto the same rolling quantity, `W` is 1, and no amount of free registers buys interleaving. So the same
knob has **two directions**, decided by "the segment's live-value peak vs the budget":

| Segment live-value peak vs budget | Problem | Direction |
|---|---|---|
| at the budget (or `LDL/STL > 0`) | demand too large; ptxas is forced to flatten the chains | **free up headroom** (§(b)) |
| clearly below the budget | width too small; the headroom sits idle and ptxas has nothing to interleave | **spend headroom to buy width** (§(d)) |

The two branches are opposite; measure first, then act — don't choose by intuition.

**An intuition this entry corrects**: "hoisting `LDS`/`MUFU` in batches in the source to pull them away from their
consumers" is a **no-op** at the CuTe DSL level. The DSL emits NVVM/PTX, and between PTX and SASS ptxas reorders
from scratch; as long as it has headroom, element-major source and hand-blocked source compile to **the same SASS**.
(Measured by this entry's probe: the element-major variant and the 8-element stage-major variant of the same segment,
under a 168-register cap, both compiled to `REG=147`, with **median distance 28 for both** from `MUFU.EX2 →` the
first consumer (had the source order landed it would be 1-2), and a time difference of 1.1% — below measurement
noise. Deeper blocking was slightly slower.)

## When needed (trigger)

When the segment's dominant stall is `short_scoreboard` / `mio_throttle` / `wait` (**not** `barrier` (waiting on
sync) / `long_scoreboard` (waiting on gmem)), judge in this order:

1. **First rule out occupancy**: when there are enough warps on the same SMSP, other warps cover the latency for you,
   and the whole entry does not apply (check `sm__warps_active` first). Low occupancy (1-4 warps per SMSP) is the
   precondition for this entry, and is exactly the norm for large-budget warp-specialized kernels.
2. **Then split into three branches by "what producers the segment has"** (the three have different cures; taking
   the wrong branch burns a round for nothing):

   | The segment's SASS has | Stall shape | Go to |
   |---|---|---|
   | an MMA rendezvous point, and **the rendezvous bucket itself is over budget** | `wait` alone takes most of it, `short_sb`/`mio` both low | [`wgmma_deferred_wait`](wgmma_deferred_wait.md), not this entry |
   | `MUFU.*` / `LDS` | `wait` / `short_scoreboard` / `mio_throttle` about 1/4 each, none over ~40% | **this entry**, anchored on the `MUFU.EX2` of §(a) |
   | **no** `MUFU.*`, no `LDS`, no MMA (pure FMA / integer bookkeeping / address arithmetic) | `wait` **alongside** `dispatch_stall`/`misc`/`not_selected`, `short_scoreboard` and `mio_throttle` both single digits | **this entry's fixed-latency branch**, anchored per §(a′), widened per §(d1) |

   - The middle branch (the mixed form) already **sits** in the shadow of a stretch of in-flight MMAs
     (machine-read confirmation: the `WARPGROUP.DEPBAR.LE` immediate in effect at the segment start ≥ 1; read the
     immediate at the start of each elementwise segment with `transcribe()` + `sched_verdict()` of
     `notes/sched_live_tools.py`) — the shadow is already laid, the `wait` share is "the rent of the shadow itself",
     and there is no tier left to deepen it; the real question is **whether this segment fills** it, which is
     exactly §(a) below. **Routing the mixed form as a whole to deferred-wait is a misreading**: that recipe has no
     row to read once the depth is already ≥1, while this entry's criteria were never run.

   **Differential clause: `wait` inside the segment ≠ waiting on MMA.** The two share one counter and the stall
   category itself cannot separate them; they are separated by **bucketing the segment ledger**: cut the kernel's
   dynamic time into buckets by SASS position (MMA rendezvous bucket = `WARPGROUP.DEPBAR` / `BAR.SYNC` / mbarrier
   wait points; elementwise bucket; other) and compare `t_ns` bucket by bucket against the baseline (when a
   comparable control implementation exists, use both sides' segment ledgers from **the same capture, at the same
   segment position**):

   - **The rendezvous bucket itself is over budget** ⇒ it is a wait-depth/supply problem ⇒ leave this entry, go to
     `wgmma_deferred_wait`.
   - **The rendezvous bucket is at parity with the baseline (or even lower), and the excess falls in the
     elementwise bucket** ⇒ the `wait` in that bucket is exposed by **this segment's own ALU / address dependency
     chains** and has nothing to do with in-flight MMAs. Routing to deferred-wait then is a **misroute**: that recipe
     has no row to read once the rendezvous bucket is at parity, and a round buys nothing.

   Machine-read backing (this entry's certification probe): a pure FFMA segment with **not a single MMA, barrier or
   shared/global memory access** still has `wait` as the overwhelmingly dominant term (measured
   `wait ≥ 4 × (barrier + long_scoreboard)`, and at the same time `wait ≥ 4 × (short_scoreboard + mio_throttle)`).
   **The `wait` reading by itself carries no "waiting on MMA" information**; taking it as MMA evidence is misreading
   the counter's definition.
3. **Then measure this segment's actual schedule** (§(a) below): read three numbers together — the **median
   distance** `dist_med` from `MUFU.EX2` to its first consumer in SASS, the **segment's live-value peak**
   `live_peak` (a backward def-use over the segment's SASS), and the **budget** of the warpgroup this segment
   belongs to (without warp specialization = the per-thread cap derived from launch bounds; with setmaxnreg = that
   role's `setmaxnreg.inc` immediate — **not** the `REG:` of `cuobjdump -res-usage`, see the gotcha "`REG:` is not
   this segment's demand in a warp-specialized kernel"):

   | `dist_med` | `live_peak` vs budget | Reading | What to change |
   |---|---|---|---|
   | ≥ 8 | any | ptxas has already interleaved it all the way for you | **this segment has no latency hole to fill**. If stalls are still high it is a true throughput floor; go to the two `cfrag_*` entries to reduce MIO counts, or raise occupancy |
   | 1-4 | **at** the budget (or `LDL/STL > 0`) | ptxas was forced by registers to flatten the chains | **free up headroom** (§(b)) |
   | 1-4 | **clearly below** the budget, and `LDL/STL == 0` | first rule out the segment being cut into basic blocks / not fully unrolled (gotchas 3, 4); if both are clean it is **insufficient in-flight width**, with headroom sitting idle unspent | **spend headroom to buy width** (§(d)) |

   **A fixed-latency segment (the third branch) reads the same table, with two changes**: the anchor becomes the
   fixed-latency arithmetic class of §(a′) (the `dist_med` thresholds `≥8` / `1-4` are unchanged); the last row has
   only one exit left — this segment has neither MIO counts to reduce nor a second path to switch to, so when
   `dist_med` falls in `1-4` **the only cure is widening** (§(d1)). The `dist_med ≥ 8` row reads differently here:
   a fixed latency is only a few cycles to cover, and `≥ 8` already covers it; if the segment is still slow then,
   look at occupancy or outside the segment, and don't keep widening inside it.

4. **Don't take instruction count as this segment's price**: in an MIO-bound segment the non-MIO instructions are
   not on the critical path, and deleting them is almost free of effect (marginal price ≈ 0). Before acting, run one
   delete-instructions pricing experiment to confirm this, so you don't go back and forth between two mutually
   exclusive routes.

## Minimal runnable example

### (a) Measure the actual schedule: `dist_med` + `live_peak` + budget

First get the SASS. The DSL's on-disk artifacts change with versions; chain three routes on "whichever works
first", don't bet on one:

1. Compile once with `CUTE_DSL_DUMP_DIR=<dir>` (plus `CUTE_DSL_KEEP_IR/KEEP_PTX/KEEP_CUBIN=1`), filter files under
   `<dir>` and `CUTE_DSL_CACHE_DIR` by ELF magic (`\x7fELF`) → `cuobjdump -sass`; get registers separately from
   `cuobjdump -res-usage` (the `REG:` field).
2. When what lands is PTX (the file has `.target sm_XX`), run `ptxas -arch=sm_XX -O3 -o x.cubin` yourself, then
   `cuobjdump -sass` — note this route uses **your own** ptxas options, and its ordering conclusions are comparable
   with the DSL's internal compile only at the same `-O` level.
3. If neither, go through the profiler: `ncu --set full --launch-count 1 -f -o rep <run>`, then
   `ncu --import rep.ncu-rep --page source --print-source sass`. No on-disk artifact needed, at the cost of running
   the kernel once more.

The three routes have different disassembly formats (cuobjdump has `/*addr*/` comments, NCU a bare hex address
column); when comparing variants, always use text obtained by **the same route**, and the parser must recognise
both formats (`parse_sass` of `notes/sass_tools.py`). Once you have the text:

```python
# MUFU.EX2 -> the first instruction reading its destination register as a source; take the median distance
import re
_DEST = re.compile(r"^(R\d+)\s*,")
dists = []
for i, cur in enumerate(ins):                       # ins = parse_sass(txt).instructions
    if not cur.opcode.startswith("MUFU.EX2"):
        continue
    m = _DEST.match(cur.operands)
    if not m:
        continue
    pat = re.compile(r"\b" + m.group(1) + r"\b")
    for j in range(i + 1, min(i + 400, len(ins))):
        ops = ins[j].operands
        if pat.search(ops.split(",", 1)[1] if "," in ops else ops):
            dists.append(j - i)
            break
dists.sort()
assert dists, "not a single MUFU.EX2 in the segment -- switch the anchor, see §(a′) (an empty list is not dist_med = 0)"
dist_med = dists[len(dists) // 2]
```

When comparing variants use **ratios** rather than absolute counts: `(FFMA+FMUL+FADD)/EX2`, `F2FP/EX2`, `LDS/EX2`.
Loop unrolling on the ptxas side multiplies the absolute counts as a whole by `U` (`U = EX2 count / elements per
thread`, which should be a positive integer); ratios are unaffected. Every comparison must also pin
`LDL == STL == 0`, otherwise you are comparing two different things.

The third number — the **segment's live-value peak** — needs only one backward def-use over the same SASS; for
decoding register **ranges** (accumulator width / `.xN` / width suffixes) use the repo's certified decoder directly,
don't hand-roll it:

```python
import sched_live_tools as slt          # notes/sched_live_tools.py
# body = the segment's instruction slice (e.g. first MUFU.EX2 .. last MUFU.EX2)
live, live_peak = set(), 0
for ins in reversed(body):
    defs, uses, _guessed = slt.reg_accesses(ins)
    live -= defs
    live |= uses
    live_peak = max(live_peak, len(live))
```

Compare `live_peak` with the "budget", not with `REG:`. How to get the budget: without warp specialization it is the
cap derived from launch bounds (the formula in §(b)); with warp specialization it is that role's `setmaxnreg.inc`
immediate — `setmaxnreg.inc.sync.aligned.u32 <N>;` in PTX, `USETMAXREG ... INC` in SASS. The `REG:` of
`cuobjdump -res-usage` only reports **the per-thread allocation of the whole kernel**; it cannot see redistribution
between roles.

### (a′) Switch the anchor to fixed-latency producers: how to measure `dist_med` for a segment without `MUFU.EX2`

§(a) anchors on `MUFU.EX2` because in that class of segment the SFU result is the only long-latency producer. In a
segment of pure FMA / integer bookkeeping / address arithmetic the `EX2` count is **0**, and the `dists` returned by
the §(a) code is an **empty list** — note that **an empty list is not `dist_med = 0`**: 0 falls into no row of the
reading table (neither `≥8` nor `1-4`), and reading it as 0 judges the segment "nothing to work with" and silently
drops it from the table. **The measurement only changes the anchor's opcode set; everything else is word for word
the same**:

```python
# Fixed-latency producer class: not on MIO, no scoreboard, dependencies accounted by stall counts hard-coded at compile time.
FIXED_LAT = {"FFMA", "FMUL", "FADD", "FSEL", "FSETP", "IMAD", "IADD3", "LEA",
             "LOP3", "SHF", "MOV", "PRMT", "SEL", "F2FP", "F2F", "I2F", "F2I",
             "ISETP", "IMNMX", "FMNMX", "HADD2", "HFMA2", "HMUL2"}

def first_consumer_dists(body, anchor):
    # anchor instruction -> first instruction reading its destination register as a source; distance in issue slots.
    # Double the body before walking: only then does a consumer landing on the next round's anchor read the real
    # cross-back-edge distance; without doubling, the anchors at the end of the segment fall straight out of the slice and bias the sample.
    n, dbody, dists = len(body), body + body, []
    for i in range(n):
        if dbody[i].opcode.split(".", 1)[0] not in anchor:
            continue
        defs, _ = du(dbody[i])                    # du() is the def-use decoding of §(a)
        defs = {r for r in defs if r[0] == "R"}
        if len(defs) != 1:                        # multi-destination / no-destination instructions are not anchors
            continue
        for j in range(i + 1, i + n):
            if defs & du(dbody[j])[1]:
                dists.append(j - i)
                break
    return sorted(dists)

dists = first_consumer_dists(body, FIXED_LAT)
assert dists, "not a single anchor-class instruction in this segment -- the measurement does not apply, it is not dist_med = 0"
dist_med = dists[len(dists) // 2]
```

Usage rules:

1. **Look at the sample count before the median.** `len(dists) == 0` is the signal "switch the anchor", not a
   reading. Every time you report `dist_med`, report `n_anchor` with it, otherwise an empty list masquerades as a
   very small distance.
2. **When a segment mixes wide arithmetic with a narrow prerequisite chain, the median lies.** If a segment has
   both a large batch of mutually independent arithmetic (large distances) and a short serial scalar / address chain
   (distances 1-2), the median is pulled up by the bulk, while **the segment's time is priced by that narrow
   chain**. So read the **distribution**, not the median: print the histogram of `dists` (or the 25th percentile);
   "a large batch of big distances + a small cluster of 1-2" is the fingerprint of "wide arithmetic + a narrow
   prerequisite chain", and what needs widening is that prerequisite chain, not the arithmetic that was already wide.
3. **Take the anchor set as the intersection with the opcodes actually present in the segment**; don't apply the
   whole `FIXED_LAT` unconditionally; if the segment also has `MUFU` / `LDS`, it is the second branch (the mixed
   form), go back to §(a)'s anchor.
4. **The set of ratios with `EX2` as the denominator in §(a) changes its denominator in this branch to the anchor
   class count** (or a variant-independent quantity such as "elements per thread"): `(FFMA+FMUL+FADD)/elements`,
   `LDS/elements`. The denominator is only there to absorb ptxas's unroll factor `U`; when the anchor changes, the
   denominator must change with it, otherwise the "only the order/form changed" gate (§(c) item 1) is always true on
   this branch and stops nothing.

### (b) The only lever: free up register headroom for the scheduler

Headroom is not abstract — it is "cap − actual demand", and both ends can be moved:

**Raise the cap.** The per-thread register cap is decided by launch bounds, not by how many variables you write:

```python
# CuTe DSL: block + min_blocks_per_mp emit `.reqntid <block>` + `.minnctapersm <M>`,
# from which ptxas sets the cap at 65536 / (block * M), rounded down to the allocation granularity (8).
kernel(...).launch(
    grid=[num_sm, 1, 1], block=[128, 1, 1],
    min_blocks_per_mp=3,        # 128 threads: 65536/384 -> 168 reg/thread
    smem=Storage.size_in_bytes(), stream=stream)
# min_blocks_per_mp=4 -> 128 reg/thread; =5 -> 96 reg/thread
```

> The cap is a **compile-time** constraint: when `grid` actually only fits 1 CTA/SM, `min_blocks_per_mp` does not
> change the occupancy on the machine, only ptxas's budget. This is exactly how to measure "scheduler headroom"
> separately from "occupancy": fix `grid == number of SMs` and sweep only `min_blocks_per_mp` — the machine stays the
> same, the compiler changes.
>
> In a warp-specialized kernel there is a second route: `cute.arch.setmaxregister_increase/_decrease` hands the
> producer warps' quota over to the consumer warps (see [`uniform_reg_budget`](uniform_reg_budget.md),
> [`warp_specialization`](warp_specialization.md)).

**Reduce demand.** Every resident live value of the segment saved (upstream accumulators + downstream values already
produced + the non-accumulator working set) adds a notch of interleaving depth for ptxas. Per-segment live-set
accounting is in [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md); the typical technique is to pack/write
back upstream values as soon as they are consumed, rather than letting them span the whole segment.

**Its price is real money**, on both sides: push the cap down ⇒ ptxas first flattens the schedule, then spills, and
once `LDL/STL` appear the segment has at once more instructions, more MIO traffic, and lost interleaving depth. So
"freeing headroom" and "adding live values" are two sides of the same coin: any rewrite that keeps `k` more values
live in the segment must first confirm that `REG` is not at the cap and `LDL/STL` are still 0, otherwise what you buy
is a net negative.

### (c) Regression gates (use them together)

1. **Only the order/form changed**: the three ratios `(FFMA+FMUL+FADD)/EX2`, `F2FP/EX2`, `LDS/EX2` differ by ≤ 2%
   across variants, and `LDL == STL == 0`. (On the fixed-latency branch the segment has no `EX2`; the denominator
   becomes the anchor class count or elements per thread — see §(a′) rule 4.)
2. **Actual schedule**: read `dist_med` and `REG:` per the table of §(a); the direction of the change must be
   `dist_med` rising **and** `REG` not at the cap.
3. **Pay-off**: segment time (NCU segment-level `t_ns`) drops. When the segment is less than two or three tenths of
   the total, end-to-end drowns the signal; accept at the segment level, with end-to-end only as final confirmation.
4. **Double reading of the segment ledger (mandatory on the fixed-latency branch)**: anchor acceptance on the
   **same-position segment ledger** — in the same capture, cut the kernel's dynamic time into buckets by SASS position
   (MMA rendezvous bucket / elementwise bucket / other), and compare two numbers of **the same bucket** before and
   after; both must move: the bucket's `t_ns` drops **and** the bucket's `wait` share drops.
   - Looking only at end-to-end: drowned out by other buckets (and also cancelled by regressions in other buckets,
     reading as "no effect").
   - Looking only at the `wait` share: fooled by "swapping one kind of waiting for another" (the share moved, the time
     did not).
   - When a comparable control implementation exists, both sides' segment ledgers must come from **the same job, at
     the same segment position**; comparing across jobs lets clock / input differences eat the signal. A cell on the
     control side that was never measured must not enter the ledger — a cell "derived" by multiplying the control's
     instruction count by your own side's regression slope is not a measurement, and it assigns a whole residual to
     the wrong place.

### (d) The opposite direction: spend idle headroom to buy in-flight width

`live_peak` clearly below the budget while `dist_med` is still 1-4 means the headroom is sitting there idle but ptxas
has nothing to work with. It has nothing to work with because **there is no independent work to interleave**: the
only things that can be interleaved are **mutually independent** element chains within the segment, and that number
(the in-flight width `W`) is the dependency graph hard-coded by the source; the compiler can reorder, not rewrite.

**How wide is needed (lower bound)**:

```
W* ≈ ceil( end-to-end latency of the per-element dependency chain / cycles per element the chain occupies on its tightest pipe )
```

Take the denominator on the tightest pipe: one warp-level `MUFU` (SFU has 4 lanes per SM sub-partition) occupies the
SFU for **8 cycles**, `LDS` occupies the LSU for several cycles; the numerator adds up the **dependency latency** of
every hop on the chain (`MUFU` and `LDS` are the big ones, `FFMA`/`F2FP` small). The longer the chain and the more it
hangs on MIO, the larger `W*`.

**How wide is possible (upper bound)**:

```
W ≤ (budget − segment resident ledger) / live values in flight per element
```

The per-segment accounting of the resident ledger (in-flight accumulators + resident accumulators + upstream residue
+ the non-accumulator working set) is in [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md). When `W*`
exceeds this upper bound, the segment's latency **cannot be covered** under the current budget — then go back to §(b)
and free headroom first, or raise occupancy; don't force the widening.

**How to widen (the spelling at the DSL level)** — the goal is to have more elements **simultaneously** in the
"produced, not yet fully consumed" state:

```python
# Counter-example: W = 1. All elements strung onto the same rolling quantity; one chain, ptxas has nothing to interleave.
acc = init
for i in cutlass.range_constexpr(NELEM):
    acc = f(acc, x[i])            # step i's input depends on step i-1's output

# Example: W = k. k independent chains open at once, merged once at the end.
acc = [init for _ in range(k)]                     # Python list of SSA values
for i in cutlass.range_constexpr(NELEM // k):
    for j in cutlass.range_constexpr(k):           # these k steps are mutually independent
        acc[j] = f(acc[j], x[i * k + j])
out = reduce_pairs(acc)                            # the merge is done only once
```

> The snippet above is inside `range_constexpr` (pure Python at trace time; the list holds SSA values). Putting the
> same form into a **runtime** loop (`cutlass.range`), the loop-carried state must be carried in a form the DSL can
> track — named scalars, or an rmem fragment + static indices (the variants of both of this entry's probe sets are
> written this way), see [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md).

Three rewrites that also count as "widening":

- **Don't string elements onto the same rolling quantity** (the same running max / running sum / the same temporary
  slot reused over and over); give each group its own value. Leave the merge to be done once at the end of the
  segment.
- **Don't close a live range mid-chain and immediately reopen it**: packing/writing back a batch of values and then
  reading the same batch back in the next statement is equivalent to pushing `W` back down to the size of one group.
  Either go all the way in one pass, or really let it leave this segment.
- **Fully unroll**: `cutlass.range_constexpr`, values in a Python list of SSA values. A fragment with runtime indices
  falls into local memory and turns the segment into a heavy `LDL/STL` user (gotchas); back edges are also scheduling
  boundaries (gotchas).

**Advance in steps, measuring at each step**: double `W` one step at a time, measuring `dist_med` / `live_peak` /
`LDL,STL` / segment time. If `dist_med` stops rising, or `LDL/STL` go from 0 to positive, or `STACK > 0` — any of
these — step back to the previous level. `W*` is finite: stop once the latency is covered; going wider only pushes
live values toward the budget cap and then falls off the §(b) end.

**Two neighbouring routes explicitly forbidden on this branch** (both measured ≈0 for this form; taking them burns a
round for nothing):

1. **Reordering instructions / hand-blocking in the source** — certified zero gain (gotcha 1), and it doesn't change
   the dependency graph at all: the criterion is "did the number of independent chains change after the rewrite";
   if not, it is only a reordering.
2. **Cutting instruction counts in a covered MIO-bound segment** — run one delete-instructions pricing experiment
   before acting (item 4 of "When needed"); if the ceiling is less than a third of the gap, don't take this route —
   those instructions were never on the critical path.

### (d1) The fixed-latency producer branch: `W*` is single-digit, but lacking it still costs a whole multiple

The segments of this branch have no SFU and no shared; every hop on the chain is just an ordinary ALU instruction.
Plug the two numbers of this kind of chain into §(d)'s lower-bound formula:

```
numerator:   end-to-end latency of the per-element dependency chain = hops on the chain × fixed latency per hop (most
             arithmetic instructions on SM90 are on the order of a few cycles; the official docs give the "about 4 clock cycles" tier)
denominator: cycles per element the chain occupies on its tightest pipe = 1
             (FP32 is 128 lanes per SM / 4 SMSPs = one warp-level FFMA per SMSP per cycle)
=>   W* ≈ fixed latency per hop ÷ occupancy cycles per hop ≈ 4-8
     (when every hop on the chain is fixed-latency, the hop count cancels between numerator and denominator: `W*` does not
      depend on chain length, only on "how many cycles one hop waits" and "how many can issue per cycle". A step that strings
      two instructions together (e.g. `FMUL` then `FFMA`) counts as two hops, doubling `W*`.)
```

By contrast, the `W*` of SFU / shared chains is in the tens (a single `MUFU` occupies the SFU for 8 cycles, with high
latency on top). Two consequences, both to be followed:

1. **Widening is extremely cheap**: the resident live values of 4-8 chains are usually readily available within the
   headroom; there is no need to go through §(b) first.
2. **The gain of widening has a hard ceiling**: going wider past `W*` is pure waste — it only pushes live values
   toward the budget cap and then falls off the §(b) end. Follow §(d)'s steps, and stop when `dist_med` stops rising.

**But the cost of "insufficient width" is a whole multiple, not a few percent.** The same batch of instructions, the
same counts, the same constants — changing only the dependency graph from 1 chain to 4 or more makes a difference of
several times in time — this entry's probe measured exactly this on three variants with **instruction counts equal
one for one** (see "Sources"). So on this branch:

- **Don't use "this segment is all cheap ALU and not many instructions" as a reason to rule it out.** The price of a
  fixed-latency segment is set by the **dependency graph**, not by the count; with the same count and different
  widths, times can differ several-fold.
- **Don't take "this batch of arithmetic is already mutually independent" as "this segment is wide enough".** The
  segment's width is set by **the narrowest time window**: if a batch of mutually independent arithmetic all depends
  on the same serial scalar / address prerequisite chain, the in-flight width is 1 while that chain runs, and the whole
  segment's time is priced by it. Read the distribution of `dists` per §(a′) rule 2 to find that chain and widen it
  (split it into mutually independent sub-chains, or move it as a whole to the previous round to run in parallel with
  this round's wide arithmetic).
- **After widening, accept at the segment level per §(c) item 4**: it only counts as paid off when both the bucket's
  `t_ns` and the bucket's `wait` share drop.

## Gotchas

- **Reordering instructions in the source and expecting it to land in SASS.** This is the most expensive pitfall of
  this entry, because it looks most like "I'm doing scheduling optimisation". ptxas has its own list scheduling and it
  is register-pressure sensitive: **with ample headroom it interleaves for you** (so your rewrite gains nothing),
  **with little headroom it squashes your hoisting back** (so your rewrite gains nothing, for the opposite reason, and
  the extra live values introduced by blocking make spilling worse). The two cases can only be told apart by
  `dist_med` + `REG:`, not by reading the source; in neither case is the cure the order.
- **Reading high `short_scoreboard` as "time to reorder".** It only says "waiting on MIO-class results". First locate
  it with the table of item 3 of "When needed": when `dist_med ≥ 8` and `REG` has headroom, the segment is already
  pressed against its throughput floor and no ordering will move it — that is a matter for the two `cfrag_*` entries
  (reduce counts) or for occupancy.
- **The segment is cut into basic blocks.** `bar.sync` / `fence_view_async_shared` / mbarrier waits / runtime branches
  all cut the segment into several basic blocks, ptxas's scheduling window resets at each boundary, and `dist_med`
  collapses to 1-2 with it. Hoist all synchronisation the segment needs to before the whole segment.
- **The segment is not fully unrolled.** In an inner loop with a runtime trip count, the back edge is also a
  scheduling boundary; static bounds always use `cutlass.range_constexpr`. (This holds for **any** intra-segment
  scheduling, independent of this entry's conclusion.)
- **Fragments with dynamic indices.** Holding SSA values in a Python list at trace time is safe (the values themselves
  are registers); a fragment with runtime indices falls into local memory and turns this segment straight into a heavy
  `LDL/STL` user.
- **Reading `REG:` as the "real demand".** With launch bounds on, `REG:` is **the actual allocation under that cap**,
  not the unconstrained demand; to measure the unconstrained demand, compile a separate version without
  `min_blocks_per_mp` (see [`uniform_reg_budget`](uniform_reg_budget.md), [`wgmma_desc_uniform`](wgmma_desc_uniform.md)).
- **`REG:` is not this segment's demand in a warp-specialized kernel.** `cuobjdump -res-usage` only reports **the
  per-thread allocation of the whole kernel**; the inter-role redistribution of `setmaxnreg` is runtime behaviour it
  cannot see. Using it to judge "does this segment have idle headroom" counts the consumer role's headroom as the
  producer's, misreading both sides. The criterion is the **segment's live-value peak** (§(a)'s backward def-use)
  against **that role's `setmaxnreg.inc` immediate**.
- **Reading §(b)'s warning as "the fewer live values the better".** Saving live values only pays when they are **at
  the budget**; with idle headroom the same warning is backwards — then fewer live values just leave the headroom
  idle. The direction is decided by `live_peak` vs budget (the two-row table in "Purpose"), not by intuition; both
  branches have measured prices.
- **Treating "widening" as "blocking".** Blocking reorders the same batch of already independent elements by stage,
  and ptxas normalises it away (this entry's first gotcha); widening **changes the dependency graph** so that more
  independent chains are open at once. One-line criterion: did the **number of independent chains** change after the
  rewrite? If not, it is only a reordering; don't expect a gain.
- **Width is not "the more the better".** Once `W` exceeds `W*` (the lower-bound formula of §(d)) the latency is
  already covered, and further widening only pushes live values toward the budget cap and then falls off the §(b) end
  (first the schedule is flattened, then spills). Widen in steps, measuring `dist_med` / `LDL,STL` / segment time at
  each step, and step back as soon as any of them gets worse.
- **Reading `wait` inside the segment as "waiting on MMA".** `wait` is defined as "waiting on a **fixed-latency**
  execution dependency"; waiting on MMA is just one kind of dependency that lands in this counter. A pure ALU segment
  with no MMA, no barrier and no memory access at all can still be overwhelmingly dominated by `wait` (measured by this
  entry's probe). The two are separated not by stall category but by **bucketing the segment ledger**: only when the
  rendezvous bucket itself is over budget go to [`wgmma_deferred_wait`](wgmma_deferred_wait.md); when the rendezvous
  bucket is already at parity with the baseline and the excess falls in the elementwise bucket, going to that recipe
  is a misroute, and a round buys nothing.
- **The `EX2` anchor returns an empty list, read as `dist_med = 0`.** An empty list means "the measurement does not
  apply, switch the anchor", not a very small distance; `0` falls into no row of the reading table, so the segment is
  silently judged "nothing to work with". Report `dist_med` together with `n_anchor` (§(a′) rule 1).
- **Widening a fixed-latency segment by the width tiers of SFU segments.** The `W*` of a fixed-latency chain is
  **single-digit** (§(d1)); widening it to the tens of chains of an SFU segment, every chain past `W*` only pushes live
  values toward the budget cap and then falls off the §(b) end (first the schedule is flattened, then spills).
- **"This batch of arithmetic is already mutually independent, so the segment is wide enough".** The segment's width
  is set by the narrowest time window: if a batch of mutually independent arithmetic all depends on the same serial
  scalar / address prerequisite chain, the in-flight width is 1 while that chain runs. The criterion is the
  **distribution** of `dists` (§(a′) rule 2), not the median.
- **Using "the instructions are all cheap ALU and not many" as a reason to rule out a fixed-latency segment.** The
  price on this branch is set by the dependency graph: with instruction counts equal one for one and only the width
  changed, times can differ by whole multiples (this entry's probe C4 + C6 are exactly this comparison).
- **Don't use this entry as a substitute for an occupancy problem**: when there are enough warps per SMSP, sibling
  warps cover the latency, and this entry's gain converges to 0.

## Sources

- Semantics of NCU stall reasons (`short_scoreboard` = scoreboard dependency waiting on MIO-class results, including
  shared and SFU; `mio_throttle` = MIO instruction queue full; `math_pipe_throttle` / `not_selected` = throughput
  side): Nsight Compute Kernel Profiling Guide, "Warp Scheduler States"
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>.
- The per-SM-sub-partition throughput ratio of SFU / MUFU vs FMA (`MUFU` 4 lanes per cycle vs FP32 32 lanes per
  cycle, i.e. one warp-level `MUFU` occupies the SFU for 8 cycles): CUDA C Programming Guide, the "Arithmetic
  Instructions" table (compute capability 9.0 row)
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>.
- Launch bounds → per-thread register cap (`.maxntid`/`.reqntid` + `.minnctapersm` are constraints when ptxas
  allocates registers): PTX ISA, "Performance-Tuning Directives"
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives>; for the
  semantics of ptxas's `-maxrregcount` see the CUDA Binary Utilities / nvcc manual.
- PTX spelling and semantics of `setmaxnreg` (`setmaxnreg.inc.sync.aligned.u32 <N>` is a warpgroup-level
  redistribution of the per-thread quota, a different thing from the compile-time cap set by
  `.reqntid`/`.minnctapersm`): PTX ISA, "Miscellaneous Instructions: setmaxnreg"
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg>.
- Public infra API symbols: `cutlass.range_constexpr`, `cute.math.exp2(..., fastmath=True)`,
  `kernel(...).launch(..., min_blocks_per_mp=...)`, `cute.arch.setmaxregister_increase/_decrease`; the SASS parsing
  tool `notes/sass_tools.py`, register-range decoding with `reg_accesses` / `transcribe` / `sched_verdict` of
  `notes/sched_live_tools.py`.
- GPU-certification probe (not included in this release) — the same elementwise
  block, **two axes crossed**: source issue order (element-major `CP=1` / 8-element stage-major `CP=8`) × ptxas
  register cap (`min_blocks_per_mp` 3/4/5 ⇒ 168/128/96). In all six cells `grid == number of SMs`, and the machine-side
  occupancy is constantly 1 CTA/SM (one active warp per SMSP), so the only things moving are "source order" and
  "compiler budget". Four contracts:
  **A** each of the six cells reproduces the numpy reference;
  **B** at the loosest cap, the ratios `(FFMA+FMUL+FADD)/EX2`, `F2FP/EX2`, `LDS/EX2` of the two orders differ by ≤2%
  and `LDL/STL == 0` (⇒ the order axis is pure order);
  **C1** at the loosest cap, the **element-major** order (producers right next to consumers by construction) has a
  median distance from `MUFU.EX2 →` the first consumer ≥ 8 (had the source order landed it would be 1-2 ⇒ ptxas
  normalised the source order away);
  **C2** at any cap, source blocking buys no time (`t(CP=1)/t(CP=8) < 1.03`);
  **D** register headroom is the lever, and it has a price: `REG` drops with the cap, and the tightest tier is ≥5%
  slower than the loosest.
- GPU-certification probe (the two-row direction table of "Purpose", the `live_peak` criterion of §(a), the widening
  branch of §(d), and the three accompanying new gotchas):
  the same elementwise block, **only the
  dependency graph changed**: 1024 `MUFU.EX2` per thread spread over `W ∈ {1, 4, 8, 32}` mutually independent chains
  (1024/W steps per chain). `grid == number of SMs`, 128 threads/CTA, `min_blocks_per_mp=3` ⇒ exactly one active warp
  per SMSP, cap 168 reg/thread, so the only thing moving is "the source's in-flight width". Five contracts:
  **A** each of the four cells reproduces the numpy reference (a distinct fixed point per slot: land in the wrong slot
  and the answer is wrong);
  **B** the axis is pure width — all four cells have exactly 1024 `EX2`, `(FFMA+FMUL+FADD)/EX2` differ by ≤5%,
  `LDL == STL == STACK == 0` (no cell is secretly paying for spills);
  **C** ptxas cannot invent width: `dist_med(W=1) ≤ 4` (nothing to reorder, producers right next to consumers) while
  `dist_med(W=32) ≥ 8`;
  **D** this width has a price: `t(W=1) / min(t(W>1)) ≥ 1.5`;
  **E** two instruments can read this gap: `REG(W=1)` is ≥48 below the cap (the idle headroom is visible, not
  imagined), and `REG` and the segment's `live_peak` each rise by ≥32 from `W=1` to `W=32` (the width spends exactly
  this headroom).
  ✅ **These five contracts are GPU-certified** (`PROBE PASS max_diff=1.19209e-07`) — the widening branch of §(d), the
  two-row direction table of "Purpose" and the `live_peak` criterion of §(a) are all measured conclusions.
- Semantics of the NCU stall reason `wait` ("Warp was stalled waiting on a **fixed latency execution dependency**") —
  it is a different counter from `short_scoreboard` (scoreboard dependency waiting on MIO-class results) and
  `barrier` (waiting on a rendezvous); `wait` itself **does not distinguish** whether this fixed-latency dependency
  comes from an ALU chain or elsewhere: Nsight Compute Kernel Profiling Guide, "Warp Scheduler States"
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>.
- The magnitude of fixed latency ("the latency of most arithmetic instructions is typically 4 clock cycles") and FP32
  at 128 lanes per SM per cycle (÷4 SMSPs = one warp-level FFMA per SMSP per cycle); the `W*` of §(d1) is the ratio of
  these two numbers: CUDA C Programming Guide, "Maximize Instruction-Level Parallelism" / "Arithmetic Instructions"
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-instruction-level-parallelism>
  and <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>.
- Public infra API symbols (added for this branch): `cutlass.range(..., unroll=1)`, `cutlass.range_constexpr`,
  `kernel(...).launch(..., min_blocks_per_mp=...)`; the NCU per-class stall ratio
  `smsp__average_warps_issue_stalled_<class>_per_issue_active.ratio`.
- GPU-certification probe (the third row of the table in "Purpose", the third branch and differential clause of item 2
  of "When needed", §(a′), §(c) item 4, §(d1), and the five accompanying new gotchas):
  the same elementwise block, **only the width axis of the dependency graph
  changed**: 96 fixed-latency steps `f = f*0.5 + c` per thread per round (one `FFMA` each), spread over
  `W ∈ {1, 4, 16}` mutually independent chains. The segment has **no** `MUFU`, no shared/global memory access, no MMA,
  no barrier — the defining condition of this branch, and the only way to make the "`wait` ≠ waiting on MMA"
  differential provable. `grid == number of SMs`, 128 threads/CTA ⇒ exactly one active warp per SMSP (no sibling warp
  to cover its latency). The step is a contraction (fixed point `f* = 2c`, the error halving each step), so although
  the three variants have different chain lengths they share one closed-form reference, and rounding does not
  accumulate. Eight contracts:
  **C1-C3** each of the three cells reproduces the closed-form fixed point (a distinct `c` per slot ⇒ land in the wrong
  slot and the answer is wrong);
  **C4** the axis is pure width — equal `FFMA` counts in the three cells (the same integer unroll factor `U`),
  `MUFU == 0`, MIO class == 0, MMA/barrier class == 0, `LDL == STL == 0`;
  **C5** the old anchor gives an empty list on this segment (`MUFU.EX2` sample count 0 in all three cells ⇒ the reading
  of §(a) is undefined), while the anchor of §(a′) separates the two ends (`dist_med(W=1) ≤ 4`,
  `dist_med(W=16) ≥ 8`);
  **C6** width has a price: `t(W=1) / min(t(W=4), t(W=16)) ≥ 1.5` (counts equal one for one, only the dependency
  graph differs);
  **C7** stall categories: `wait ≥ 4 × (short_scoreboard + mio_throttle)` **and**
  `wait ≥ 4 × (barrier + long_scoreboard)` ⇒ the two categories of the existing §(d) entry point are absent on this
  segment, and `wait` dominance has nothing to do with in-flight MMAs;
  **C8** the share leaves with width: `wait(W=1) ≥ 2 × wait(W=16)` ⇒ the acceptance signal of §(c) item 4 is real and
  moves with this lever.
  ✅ **These eight contracts are GPU-certified** (all eight PROBE lines PASS in the same round; numeric contracts
  `PROBE PASS max_diff=9.632110598012389e-08`, non-numeric contracts `max_diff=n/a`) — the anchor-switch measurement
  of §(a′), the magnitude of `W*` and "insufficient width costs a whole multiple" of §(d1), and the differential clause
  "`wait` dominance in the segment ≠ waiting on MMA" are all measured conclusions, not reasoning.
