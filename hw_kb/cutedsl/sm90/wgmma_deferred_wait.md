# wgmma_deferred_wait (WGMMA asynchronous depth scheduling: a commit/wait ladder hides the elementwise segment inside an in-flight GEMM)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers the **asynchronous completion
> model** of SM90 WGMMA (`wgmma.fence` / `commit_group` / `wait_group(N)`) and the "deferred-wait ladder" idiom:
> how to issue a GEMM→elementwise→GEMM chain so that the elementwise segment really overlaps the in-flight WGMMA,
> and **how to verify in a machine-readable way that the overlap really binds** (deferred wait written in the source ≠
> real overlap in SASS).
> Prerequisite: the basic WGMMA path is in [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md).

## Purpose

WGMMA is a warpgroup-level **asynchronous** instruction: it does not block after issue; `commit_group` packs all
previously uncommitted wgmmas into one group, and `wait_group(N)` waits until "≤ N groups in flight". Changing the
wait from "`wait_group(0)` right after every GEMM" to a **ladder of deferrals** (the first with `wg_wait=-1`, no wait;
after the second is committed, `wait_group(1)` reaps only the first) lets the register/elementwise segment between
the two GEMMs execute during the second GEMM's tensor-core time — this is the lever that moves `tensor_pct` (tensor
pipe duty cycle) from "all non-tensor time exposed every iteration" towards the reference level.

## When needed (trigger)

- The main loop is a GEMM→elementwise→GEMM→… chain, and NCU shows
  `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`
  (tensor_pct below) clearly below a reference on the same foundation, while spill=0, the stall lines (barrier/long_sb)
  are healthy and DRAM is not the bottleneck — i.e. "the same tensor workload, purely non-tensor time exposed every
  iteration".
- Deferred wait has already been written in the source, but after landing tensor_pct barely moves (Δ≈0): the wait did
  not land after the elementwise segment at SASS level, or the overlap was silently serialised by a register
  dependency — locate it with this entry's "machine-readable footprint" section.
- Counter-symptom: if stall/spill/DRAM has a dominant lesion, fix that first; this entry is the right remedy only when
  "everything else is healthy and only the tensor density is low".

## Minimal runnable example

### 1) The canonical issue form of each call site (fence → new atom → per k → commit → optional wait)

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup

@cute.jit
def wg_gemm(tiled_mma, acc, tA, tB, wg_wait: cutlass.Constexpr[int],
            zero_init: cutlass.Constexpr[bool] = True):
    warpgroup.fence()                       # must fence after generic code has touched acc
    # build a new atom per call site (do not reuse one across call sites and change its attributes:
    # once control flow intervenes it triggers MLIR "operand #0 does not dominate this use")
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)  # first k-block overwrites instead of zeroing
    for kb in cutlass.range_constexpr(cute.size(tA, mode=[2])):
        cute.gemm(mma_atom, acc, tA[(None, None, kb)], tB[(None, None, kb)], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()                # previously uncommitted wgmmas = one group
    if cutlass.const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)       # wait until <= wg_wait groups are in flight
```

`wg_wait` semantics (this is the `mma_*(..., wg_wait=?)` parameter common in issue helpers):
`-1` = commit only, no wait; `1` = `wait_group(1)` after the commit — **reaps every earlier group except the one just
committed**, which stays in flight; `0` = drain everything.

### 2) The deferred-wait ladder (the minimal chain of two GEMMs + one elementwise segment)

```python
# G1: in flight after issue
wg_gemm(tiled_mma, acc1, tA1, tB1, wg_wait=-1)
# G2: wait_group(1) after commit -- reaps only G1; G2 stays in flight
wg_gemm(tiled_mma, acc2, tA2, tB2, wg_wait=1)

# elementwise segment: acc1 is now retired and can be read/written freely; acc2 is in flight and must not be touched.
# this segment runs during G2's tensor-core time (exactly the overlap we want).
for i in cutlass.range_constexpr(cute.size(acc1)):
    out_frag[i] = f(acc1[i])

warpgroup.wait_group(0)                     # reap G2; only then may acc2 be touched
consume(acc2)
```

A multi-GEMM chain rolls by the same rule (a sliding window of depth 1): **after each GEMM is committed,
`wait_group(1)` reaps the previous one, and the register segment right after it only touches the accumulator just
retired**; the bottom of the main loop may omit `wait_group(0)` — the next iteration's `wait_group(1)` naturally reaps
the leftover in-flight group, and a full drain is needed only in the epilogue (or before releasing an smem stage that
GEMM reads).

### 3) Mid-chain landing in smem to feed the next WGMMA (the P→smem→GEMM form)

When the elementwise result must go through smem as an operand of the next WGMMA, **use an async-proxy fence + barrier
for visibility, not `wait_group(0)` as a catch-all** (the latter drains the whole wgmma pipeline and zeroes the
overlap):

```python
copy_r2s(frag_f16, s_operand)               # R2S (stmatrix / TV copy both fine)
cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                      space=cute.arch.SharedSpace.shared_cta)   # generic -> async proxy visibility
bar.arrive_and_wait()                       # all threads consuming that smem meet
wg_gemm(tiled_mma2, acc3, t_operand, tB3, wg_wait=1)   # the ladder keeps rolling
```

### 4) Where to release a pipeline stage (when combined with an mbarrier pipeline)

Put `consumer_release(stage)` **right after the wait that reaped "the last GEMM reading that stage"**, releasing
immediately — the following GEMM is still in flight, so TMA can refill that stage during tensor-core time:

```python
wg_gemm(..., wg_wait=1)          # this wait(1) reaped the GEMM that read stage X
pipeline.consumer_release(state) # give stage X back immediately; do not delay until after wait_group(0)
```

## Machine-readable footprint of whether the overlap binds (acceptance criteria; none can be skipped)

A deferred wait at the source level only "requests" overlap; you must confirm it really happens with one of the
footprints below:

1. **Single-knob wait-flip A/B timing** (the criterion of this entry's certification probe, the most direct): compile
   the same kernel twice, **the only difference being that one wait argument at G2's call site** (`wait_group(1)`
   deferred vs `wait_group(0)` full drain). The deferred version must be clearly faster (probe threshold
   `serial / overlap ≥ 1.05`; measured 1.101 / 1.103 / 1.104 over three GPU re-runs, stable to ±0.002).
   The two binaries differ by this single token, so the time saved can only come from the wait depth itself — this is
   the clean footprint of "deferred wait causally in effect".
   **But do not use a ratio footprint to rule on "how much of E was hidden" (overlap magnitude)** — this entry's probe
   proved with seven GPU rounds that a fair magnitude anchor cannot be built at the source level, in either direction:
   - **A fake-serial baseline (the "E fully exposed" anchor) cannot be built.** What decides overlap is the real data
     dependency (RAW) on the accumulator, not where the wait sits in the source, and even pinning RAW in both
     directions does not close every leak:
     - upstream leak: if the baseline drains early right after G1 and retires acc1, an elementwise segment depending
       only on acc1 gets hoisted by ptxas above G2's drain, and the baseline overlaps too (with E>G the measured ratio
       inverted to 0.824, while the overlap side was in fact fully bound).
     - downstream leak: pinning upstream alone is not enough — if nothing after E has a RAW dependency on E, the only
       reverse constraint is a register WAR (the next G1 overwriting acc1), and **WAR does not pin the schedule**:
       ptxas reads acc1 early into temporaries, rotates the loop, and sinks E's chain into the shadow of the **next
       iteration's** in-flight group (measured: only ~1/3 of E exposed).
     - after pinning RAW in both directions (the chain tail writes back into acc, the next GEMM with
       `ACCUMULATE=True` reads it at its first k-block; numerically neutral trick: with acc pre-zeroed the chain tail
       writes `y_end - c`, exactly +0.0 at run time, an fp subtraction the compiler cannot fold) there is **still
       hardware-level residual overlap**: wgmma completion is asynchronous, and independent ALU code next to the drain
       point overlaps the in-flight group on its own (measured: serial still hides about half of E).
   - **A no-E anchor (the "2G pure tensor pipe" anchor) cannot be built either**: with the elementwise segment removed
     and no cross-iteration register dependency, the `wait_group(1)` ladder pipelines deeply across iterations and the
     tensor pipe rushes close to peak — measured ~2.0us/iter, more than twice as fast as any cadence with E
     (≥4.1us/iter, reproduced over three rounds). The no-E stream and the ladder with E are not an isomorphic control,
     and ratios like `overlap/noew` have no verdict value.
   - A structurally isomorphic control (changing only the per-element chain length, every other token identical)
     measures "the marginal cost of lengthening E", mixed with issue-bandwidth / register-budget contention effects —
     treat it as diagnostic data, not as pass/fail.
   So the division of labour is: judge **whether it binds** with the wait-flip A/B (this item) + SASS / tensor_pct (the
   next two items); judge **how much is left to squeeze** with NCU cycles per iteration against the performance
   baseline, never with ratios to synthetic anchors. For the safe form of elementwise ballast see gotcha 8
   (medium-sized straight-line code; to lengthen its duration stack latency with a serial dependency chain, not
   instruction count, and never with a runtime inner loop).
2. **SASS check**: after `cuobjdump --dump-sass`, find the wgmma batch (`HGMMA.*`) and its wait (`wait_group` lowers
   to a DEPBAR-type wait instruction whose count is N). Criterion: the FFMA/EXP2/conversion instruction block of the
   elementwise segment must sit **after the last HGMMA and before the wait that reaps it**; if ptxas sank these
   instructions below the wait, the overlap did not bind.
3. **NCU**: tensor_pct goes up (this is the mechanism-line metric — the decision landed with tensor_pct Δ≈0 means not
   bound); at the same time the main kernel's cycles per iteration (`sm__cycles_active` / iteration count) should go
   down, and no stall line should get worse (overlap does not change the workload, only the exposure).
4. **The `commit` axis of the production cubin (dossier census, KEEP-certified)**: a histogram of the immediates of
   `WARPGROUP.DEPBAR.LE <n>` counted per loop region. The exact footprint of one level of `wait_group(1)` ladder landing
   is **exactly one more `LE 0x1` and one fewer `LE 0x0` in that region** (the reading of one KEEP: consumer main loop
   `DEPBAR {LE 0x0 x12}` → `{LE 0x0 x11, LE 0x1 x1}`, other regions unchanged). This footprint only proves that **the
   wait depth made it into the cubin**, not that it hid anything — the ruling is still the wait-flip A/B of item 1 and
   the tensor_pct of item 3 (see gotcha 10).

## Gotchas

1. **`wait_group` counts groups, not GEMMs.** Group boundaries are drawn by `commit_group`: if two logical GEMMs share
   one `commit_group`, they form one group, and `wait_group(1)` cannot reap only the earlier one — **every GEMM you
   want to reap on its own must have its own `commit_group`**.
2. **Touching an in-flight accumulator = numerical UB + silent serialisation.** If the elementwise segment reads/writes
   an accumulator not yet reaped by a wait (including seemingly harmless reads through a reshape view, f32→f16
   conversion, stores to gmem/smem), at best the numbers are wrong, at worst ptxas materialises register interlocks /
   early waits in SASS to stay correct — the source looks overlapped, while in SASS the whole segment stalls behind the
   GEMM. The operand set of the overlapped segment must be strictly disjoint from **every** in-flight accumulator.
3. **The wait one line off zeroes everything.** `wait_group(1)` must come **after the second GEMM is committed**: if
   placed before its commit (when G1 is the only in-flight group), the `wait_group(1)` condition is already met and it
   does not wait → reading acc1 afterwards is a race. The order is always commit(G2) → wait(1) → touch acc1.
4. **Reusing the same mma atom across call sites and changing its `ACCUMULATE` attribute**: as soon as control flow
   (if/loop) separates the call sites, MLIR reports `operand #0 does not dominate this use`.
   The canonical approach is a new `cute.make_mma_atom(tiled_mma.op)` per call site (example 1).
5. **`warpgroup.fence()` is not a wait.** It only declares "generic code's accesses to the accumulator registers are
   ordered", and never drains the pipe; it is needed before every wgmma batch (especially right after generic code
   wrote that acc); leaving it out gets reordered into errors, an extra one is harmless. Do not expect it to do
   wait_group's job, and do not substitute a wait for it.
6. **Using the wrong sync primitive for the smem transit segment.** Writing an elementwise result to smem and feeding it
   to WGMMA needs `fence_proxy(async_shared)` + a barrier (example 3); using `wait_group(0)` "to sync along the way"
   drains the whole wgmma pipe and kills every overlap on the chain. Conversely, with only the barrier and no proxy
   fence, WGMMA (the async proxy) may read stale smem — a silent numerical error.
7. **Releasing a stage too early or too late are both wrong.** Before the wait that reaps its reader → the producer
   overwrites smem being read (wrong numbers); delayed until after the full drain → the TMA refill cannot overlap the
   tensor cores (latency lost for nothing). The rule is in example 4.
8. **The elementwise segment inside the in-flight window must be "medium-sized straight-line code" — anti-patterns in
   both directions silently destroy the overlap (GPU measurements with the same probe, numbers all right, overlap
   zeroed).**
   - **A huge fully unrolled block**: 4096 straight-line FFMAs (≈64KB of SASS) make the front end / i-cache the
     bottleneck, and the serial/overlap timing ratio collapses from 1.114 (fully bound) to 0.998.
   - **An inner loop with a runtime trip count**: rewriting the same elementwise work as a dynamic 40-step loop per
     element (tiny loop body, zero i-cache pressure) gives a timing ratio of 1.020 — **an in-flight wgmma group does
     not survive dynamic control flow between the commit and the deferred wait / inside the in-flight window**; it gets
     silently serialised. Note this applies only to **inner** dynamic loops: the main loop's own back edge rolling
     with an in-flight group (depth-1 sliding window, no drain at the loop bottom) was measured fully bound (1.114), and
     is safe.
   - Measured safe forms: fully unrolled straight-line code ≤ ~2K instructions (~32KB SASS) confirmed bound; stringing
     each element's chain with `y*0.0 + acc[i]` into **one serially dependent FFMA chain** (~1.3K instructions,
     ~21KB) also confirmed bound (with E>G the overlap side's median == G+E, G2 fully hidden inside E) — to lengthen
     the elementwise segment's duration, stack time by latency with a dependency chain, not by adding instructions or
     introducing dynamic loops.
9. **Declaring it done without checking the footprint.** A source ladder ≠ SASS overlap: register pressure, unexpected
   fragment reuse and compiler sinking can all silently defeat it. After every change to the wait structure, check it
   per the "machine-readable footprint" section (A/B timing is the cheapest, tensor_pct the final judge).
10. **`LE 0x1` appearing in SASS ≠ an exposed drain was covered.** The full reading of one KEEP: one level of ladder
    opened on a chain pair where "the later GEMM's operands do not depend on the earlier one's accumulator" — removing
    the earlier one's `wait_group(0)`, moving the sole reader segment of its accumulator wholesale after the later
    GEMM's issue, putting `wait_group(1)` there to reap the earlier one, and keeping `wait_group(0)` at the loop bottom
    so no group crosses the back edge (path 1 of [`wgmma_backedge_pending_group`](wgmma_backedge_pending_group.md)).
    The `commit` axis showed the footprint (item 4), the numbers were right, spill 0 on both sides, yet the gate window
    went 0.5016 → 0.5062 ms (**+0.92%, slower**), tensor_pct 10.33 → 10.21, stall_barrier 1.60 → 1.51,
    stall_long_sb 3.08 → 3.15. By the rule of item 3, tensor_pct Δ≈0 means "not bound / nothing to hide".
    The structural condition this time: the reader segment moved into the in-flight window **starts with a blocking
    mbarrier wait** and ends with `stmatrix` + async-proxy fence + mbarrier arrive; the in-flight window covered an
    elementwise segment + the issue of the next GEMM. Under this structure this level of ladder is worth −0.9%.
    Discipline: for ladder changes **do the single-knob wait-flip A/B first** (item 1), then land; if the segment moved
    into the window contains a blocking wait, the A/B is all the more mandatory — its wait time is not the tensor
    cores' business.


11. **Two data-independent chains can be merged into one commit group: one `warpgroup.fence()` → chain 1 → chain 2 →
    one `commit_group()` → one `wait_group(0)`.** Three preconditions must all hold: the A operand is the same set of
    **read-only** registers (written once before either chain is issued), B are different smem descriptors (different
    rings), and the accumulators are different. **No** second `fence` between the two chains — a fence orders generic
    code's register writes against async-proxy reads, and nobody writes the A registers in between. Split the helper
    into two levels, "put one chain into the currently open group" (no fence / no commit) and "fence + chain + commit";
    a chain issued on its own still goes through the latter. Footprint: one fewer `WARPGROUP.DEPBAR` per chunk in
    steady state (the `commit` axis moves along), one fewer handover pair.
    **⊗ clause**: if this pair of warpgroups runs a ping-pong (taking the baton with `cute.arch.barrier` on its own
    NamedBarrier id, passing it with `cute.arch.barrier_arrive` on the other's id, never arriving on an id it blocks
    on), the baton points must be **coarsened** together with the merge — take the baton once before the fence, pass it
    once after the commit. A baton pass left between the two chains releases the other side's SYNC before this side's
    chain 2 has been issued, both halves issue into the tensor pipe at once, and the merge's gain is handed straight
    back as contention.
    Chain selection rule: merge only pairs that **add no live registers** (the shared A packing and both accumulators
    already resident); pairs that need a second A packing (+16 registers / thread) are excluded. The reading of one
    KEEP: `BAR.SYNC` per chunk in steady state 5 → 4, gate window 0.4459 → 0.435 ms (−2.44%); the cost side is chain
    2's B-ring full wait moving earlier to the top of the merged segment (that stage is held for half a chain longer,
    the release point unchanged). Structural condition: consumer at the 192-register tier, two 64-row RS chains sharing
    one A packing, each with its own 64×64 f32 accumulator, in a ping-pong pair with the sibling warpgroup.

12. **Opening the same level of ladder on "two mutually independent chains" (the later one's operands do not depend on
    the earlier one's accumulator) flips the sign of the cost.** The KEEP of gotcha 10 was a chain where "the later
    GEMM's operands **depend** on the earlier one's accumulator" — opening one level of ladder moved the reader segment
    into the in-flight window and was worth **−0.9%** (slower), because the moved segment starts with a blocking
    mbarrier wait. **The same ladder opened on a pair of independent chains** (the later GEMM's A/B operands unrelated
    to the earlier one's accumulator, the two chains `commit`ting as two groups of their own), moving chain A's reclaim
    point up to `wait_group(1)` and letting chain B stay in flight across a whole elementwise + publish segment (which
    contains **no** blocking wait), is worth **+0.03%** (neutral) — the `commit` axis footprint appeared as expected:
    **the one `mma-consumer` ring where the ladder happened** `{LE 0x0 x11}` → `{LE 0x0 x10, LE 0x1 x1}`, and **this one
    ring by itself** is the complete footprint of item 4 (exactly one more `LE 0x1`, one fewer `LE 0x0`). In the same
    round two other rings went `{LE 0x0 x2}` → `{LE 0x0 x1}`; that is another role merging its two halves into one
    (moving along, see below), **do not add it into this account** — adding it would count "three fewer `LE 0x0`",
    which does not match item 4. Read the footprint **per ring**, not as a total over the whole kernel.
    In the same round `tensor_pct` 25.35 → 25.2 (Δ≈0.15pp, i.e. not bound / nothing to hide) — the same rule as gotcha 6
    of [`wgmma_backedge_pending_group`](wgmma_backedge_pending_group.md): **the gain is capped by the duration of "the
    segment moved into the window"**, not by the difference between 0 and 1 itself. The reading is written into the
    structural condition: the segment covered by the window is pure elementwise + publish (no blocking wait), and in
    the same commit **another consumer role** merged its two half accumulations into one call per gotcha 11 (two
    chains committed back to back, one `wait_group(0)` at the end) — **the changes are one package and the reading
    cannot be split** (the `{LE 0x0 x2}`→`{LE 0x0 x1}` of the second/third rings of the `commit` axis is that role's
    two-halves merge moving along, not part of the depth ladder).
    Discipline as in gotcha 10: single-knob wait-flip A/B first, then land, and **before touching the wait, first ask
    "does the segment moved into the window contain a blocking wait / a stage return"** — if yes, it is gotcha 10's
    direction (slower); only if not, try it.
    KEEP in an eval job (job number not entered in the status record), the `commit` axis readings as described, gate
    0.9362 → 0.9365 ms.

13. **Gotcha 11 (merge) applied across all roles + one level of gotcha-12-direction ladder opened in a single role: read
    one criterion per ring and one for the kernel total on the `commit` axis.** The reading of one KEEP (verbatim):
    seven `mma-consumer` rings
    `{LE 0x0 x14}`, `{LE 0x0 x18}`, `{LE 0x0 x2}`, `{LE 0x0 x2}`, `{LE 0x0 x2}`, `{LE 0x0 x2}`,
    `{LE 0x0 x8}` → `{LE 0x0 x1, LE 0x1 x3}`, `{LE 0x0 x12}`, `{LE 0x0 x1}`, `{LE 0x0 x1}`,
    `{LE 0x0 x1}`, `{LE 0x0 x2}`, `{LE 0x0 x6}`; kernel total **48 `LE 0x0` → 24 `LE 0x0`
    + 3 `LE 0x1`**. Gate 0.7528 → 0.7478 ms (−0.66%, a cross-job reading, within the node noise band).
    - **Total halved = gotcha 11's merge applied everywhere**: once each run of "chains that are consecutive,
      accumulate into the same set of registers, with no register traffic in between" shares one fence / one
      `commit_group` / one drain, `LE 0x0` goes from "one per chain" to "one per run"; the reductions per ring are
      uneven (14 → 4, 18 → 12, 2 → 1, 8 → 6, one ring 2 → 2 unchanged) — per ring you can only read "how many runs of
      this role were merged"; whether the merge was applied everywhere must be read from the total. The merge is
      liveness-neutral: a merged run is just the instruction stream of a longer k loop, and the `setmaxnreg` tier bill
      does not move a cent.
    - **A ring reading `{LE 0x0 x1, LE 0x1 x(N−1)}` = a depth-1 ladder across an N-step fully unrolled loop**: step j
      issues + commits, `wait_group(1)` reaps j−1 and consumes j−1; the loop-bottom `wait_group(0)` reaps the last step.
      With N = 4 that is exactly 3 `LE 0x1` + 1 `LE 0x0`, with **no other** `DEPBAR` in that ring (the "one `LE 0x1`" of
      gotcha 12 is the special case N = 2).

    The two-level issue helper (the pair gotcha 11 needs; descriptors on the warpgroup-scope slice, see
    [`wgmma_desc_uniform`](wgmma_desc_uniform.md)):

    ```python
    def link(mma, acc, sA_t, sB_t, nk, accumulate):
        """Put one SS chain into the currently open group: no fence / no commit / no wait."""
        wgm = mma.get_slice(0)
        at = cute.make_mma_atom(mma.op)                    # new atom per call site (gotcha 4)
        at.set(warpgroup.Field.ACCUMULATE, accumulate)
        fa = wgm.make_fragment_A(wgm.partition_A(sA_t))
        fb = wgm.make_fragment_B(wgm.partition_B(sB_t))
        for kb in cutlass.range_constexpr(nk):
            cute.gemm(at, acc, fa[None, None, kb], fb[None, None, kb], acc)
            at.set(warpgroup.Field.ACCUMULATE, True)

    def chain(mma, acc, sA_t, sB_t, nk, accumulate):
        """A chain forming its own group; call sites with nothing to merge or defer still go through it."""
        warpgroup.fence()
        link(mma, acc, sA_t, sB_t, nk, accumulate)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
    ```

    An N-step depth-1 ladder (two chains per step accumulating into the same accumulator, merged into one group):

    ```python
    for j in cutlass.range_constexpr(N):            # generic-code preprocessing of the accumulators hoisted above the ladder as a whole
        for i in cutlass.range_constexpr(cute.size(acc[j])):
            acc[j][i] = acc[j][i] * dec
    for j in cutlass.range_constexpr(N):
        warpgroup.fence()
        link(mma, acc[j], A1[j], B1[j], nk, True)
        link(mma, acc[j], A2[j], B2[j], nk, True)   # same accumulator, no register traffic in between -> same group (gotcha 11)
        warpgroup.commit_group()
        if cutlass.const_expr(j > 0):
            warpgroup.wait_group(1)                 # j in flight; reap j-1
            consume(acc[j - 1])                     # straight-line code: one LDSM + 32 FFMA; no blocking wait, no stage return
    warpgroup.wait_group(0)
    consume(acc[N - 1])
    ```

    Generic-code preprocessing of the accumulators (the `* dec` above) is **hoisted above the ladder as a whole**: once
    the ladder is open, between any step's fence and its reap there are other accumulators in flight, and touching an
    accumulator with non-wgmma instructions inside the pipelined segment triggers the whole-function `DEPBAR`
    insertion of [`wgmma_backedge_pending_group`](wgmma_backedge_pending_group.md).
    **Conditions for it to be free** (only one role met them this time; the other roles stayed at depth 0): the N
    accumulators are already state resident across the whole loop (the ladder adds no live values), temporaries
    created in the covered segment live and die inside it (a chain's descriptors die after its last k-block is issued),
    the segment has no blocking wait and no stage return (the sign condition of gotcha 12, not of gotcha 10).
    Counter-example: the ladder would make a second 32-register f32 accumulator stay resident across three chains, and
    that role is exactly the one the spill line is accounted against — leave it at depth 0.
    **Load-bearing condition**: −0.66% is the reading of the package "merge applied across all roles (48 → 24) + one
    role's N = 4 ladder", with the live set at every program point the same as the depth-0 tree (register tiers
    unchanged); the two cannot be split, and it is within the cross-job noise band. This KEEP certifies the **footprint**
    on the `commit` axis (merge applied everywhere = total halved, ladder = `{LE 0x0 x1, LE 0x1 x3}`), **not the
    price**: what halving the wait count is worth on a tree "whose register tiers are already maxed out, with every
    change constrained to an unchanged live set" needs a same-job A/B, let alone the price of how much segment the
    ladder hid (judging binding still goes through items 1 and 3 of the machine-readable footprint).

14. **Keep the producer-driven full waits inside the ping-pong window (after taking the baton); do not hoist them above
    taking the baton.**
    Two consumer warpgroups form a ping-pong pair (the ⊗ clause of gotcha 11: take the baton with `cute.arch.barrier` on
    one's own NamedBarrier id, pass it with `cute.arch.barrier_arrive` on the other's id, exactly one blocking
    rendezvous per steady-state block), and the steady-state block has the form `barrier(take) → mbarrier_wait(ring A
    full) → chain 1 → mbarrier_wait(ring B full) → chain 2 → barrier_arrive(pass)`. The knob is the position of these
    two full waits relative to taking the baton; two KEEPs flipped it once in each direction:
    - **moved out** (both waits hoisted above taking the baton, the window covering only the two MMA chains): gate
      0.2317 → 0.2386 ms (**+2.98%, slower**), tensor_pct 71.31 → 69.44, stall_barrier 1.10 → 1.23, stall_mio 0.16 →
      0.21, smem_wavefronts_st 288583 → 312470;
    - **moved back** (the same pair of waits back after taking the baton; the same submission also carried other,
      themselves unchanged, already-landed items): 0.2386 → 0.2339 ms (−1.97%), tensor_pct 69.44 → 71.2, stall_barrier
      1.23 → 1.10, the other readings back to before the move.
    The direction is the opposite of gotcha 10, but the two "windows" are not the same thing: the `wait_group` in-flight
    window of gotcha 10 covers the in-flight time of this WG's tensor pipe (moving a blocking wait in makes it slower),
    while the ping-pong window here covers the time this WG holds exclusive issue rights (moving a blocking wait out
    makes it slower). Both are only readings, with no timeline isolation; before touching either, do a same-job A/B.
    **Machine-readable footprint**: no axis in the dossier **encodes this position specifically**, but a group of axes
    tracks it strictly in both directions — the `stride=0x400` of `rings` (disappears when moved out, reappears when
    moved back, gotcha 15 of [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)), `mbar_per_role.spin` 8 ↔ 4,
    `mbar_tiers` `{CONST: 10, RING: 2}` ↔ `{CONST: 11, RING: 1}`, `mma-consumer→spin` disappearing / reappearing in
    `edges`; `nbar_verdicts` / `nbar_widths` unchanged all three times (`publish-handoff … width=256` always present,
    `[[256], [256], [256]]`). This group of axes **can only serve as a tripwire** ("something around the wait placement
    moved"), neither as a load-bearing signal nor for pricing: they cannot distinguish "the wait moved" from "codegen
    changed tiers along with it", and the control separating the two was not done (see gotcha 15). What really needs
    checking after landing is the order of the mbarrier spin (`SYNCS.PHASECHK`) relative to `BAR.SYNC` in the block
    loop's SASS `[to be checked]`.
    **Load-bearing condition**: 2 consumer WGs in ping-pong, a single copy warp feeding a 2-stage ring (blocking
    mbarrier wait form), one SS chain + one RS chain per block, an in-block `wait_group(1)` ladder, consumer at the
    240-register tier; under this structure moving out is worth +2.98%. The **one-time** full wait on the resident
    operand slot at the first block of a tile goes before taking the baton, and the last chain of the tile's last block
    is not wrapped in a window (blocking rendezvous per tile = number of blocks) — that is a different rule, landed with
    the package of gotcha 11 of
    [`persistent_tile_loop_ring_continuity`](persistent_tile_loop_ring_continuity.md), with no separate reading.

## Sources

- The canonical issue helper (fence → new atom → per k → commit → optional wait, and the "new atom" comment):
  quack/sm90_utils.py `gemm`/`gemm_zero_init`/`gemm_w_idx` L149–214.
- Semantics of `wgmma.fence/commit_group/wait_group`: PTX ISA "Asynchronous warpgroup
  level matrix multiply-accumulate instructions".
- GPU-certified by a GPU probe (the probe is not included in this release) — contract 1 numerical
  (the serial/overlap variants under the ladder discipline vs an exact reference, measured max_diff=0.0) + contract 2
  causal (single-knob wait-flip A/B: `serial/overlap ≥ 1.05`, measured 1.10×, stable to ±0.002 over three re-runs;
  overlap-magnitude ratios are only printed as diagnostics — for the anchor traps in both directions see item 1 of the
  "machine-readable footprint").
- KEEP certification of the production cubin contract axis (footprint item 4 + gotcha 10): KEEP in an eval job (job
  number not entered in the status record), axis `commit` `mma-consumer: DEPBAR {LE 0x0 x12}` → `{LE 0x0 x11, LE 0x1 x1}`
  (the second consumer region `{LE 0x0 x1}` unchanged), gate 0.5016 → 0.5062 ms, tensor_pct 10.33 → 10.21, spill_ld 0 → 0.
- Merged commit group ⊗ coarsened ping-pong baton (gotcha 11): KEEP in an eval job (job number not entered in the
  status record), steady-state `BAR.SYNC` per chunk 5 → 4, worth −2.44%; the `rings` / `commit` axes flip with codegen
  (see gotcha 11 of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md)).
- Gotcha 13 (merge applied everywhere + one role's N = 4 depth-1 ladder): KEEP in an eval job (job number not entered in
  the status record), axis `commit` seven rings `{LE 0x0 x14}, {LE 0x0 x18}, {LE 0x0 x2} ×4, {LE 0x0 x8}` →
  `{LE 0x0 x1, LE 0x1 x3}, {LE 0x0 x12}, {LE 0x0 x1} ×3, {LE 0x0 x2}, {LE 0x0 x6}` (total 48 `LE 0x0`
  → 24 `LE 0x0` + 3 `LE 0x1`), gate 0.7528 → 0.7478 ms (within the noise band; footprint certification, not pricing).
  The spelling of the two-level helper and the ladder loop is taken from that KEEP's change record.
- Gotcha 14 (full wait placement inside the ping-pong window): dossier censuses + gate windows of two KEEP evaluations
  (eval job numbers not entered in the status record), `rings` / `mbar_per_role` / `mbar_tiers` / `edges` moving along,
  `nbar_*` unchanged; wait placement does not change the numbers, correctness is carried by the whole-kernel gate, no
  `max_diff` probe.
