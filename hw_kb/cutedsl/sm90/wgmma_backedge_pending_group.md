# wgmma_backedge_pending_group (an in-flight wgmma group crossing a dynamic loop back edge: the fingerprint of ptxas whole-function serialisation, acceptance and safe forms)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is the **boundary clause** of gotcha 8
> of [`wgmma_deferred_wait`](wgmma_deferred_wait.md), "the main loop's back edge rolling with an in-flight group is
> safe": when ptxas gives up wgmma pipeline analysis and serialises **every single** HGMMA **in the whole kernel**
> (each HGMMA immediately followed by a `WARPGROUP.DEPBAR.LE gsb0, 0x0`), the only place it can be seen (SASS), and the
> machine-readable acceptance that is mandatory before carrying an in-flight group across a back edge. Prerequisite:
> `commit_group` / `wait_group(N)` semantics in [`wgmma_deferred_wait`](wgmma_deferred_wait.md).

## Purpose

ptxas performs a **whole-function** pipeline analysis of `wgmma.mma_async`: each stretch "from `wgmma.fence` / the
first wgmma to the `wait_group` that reaps it" is a pipeline stage. It must prove that within the stage no
**non-wgmma instruction reads or writes the accumulator registers (or A register operands) of that stage's wgmmas**,
that the stage does not cross a function boundary, and that there are enough registers to hold the stage's
accumulators. When it cannot prove this, it does not degrade locally, but gives up asynchronous pipelining for **all**
wgmmas **in that function**: after every HGMMA it inserts a `WARPGROUP.DEPBAR` at the level of `wait_group 0`, the
tensor cores execute one instruction at a time with zero overlap, and other warpgroups / other chains are hit as well.
**The numbers are not wrong** (ptxas preserves the semantics exactly by serialising) — this is a silent performance
disease, not a correctness disease.

The mechanism facts of this entry:

1. **A single touch of the accumulator inside the stage is enough**: the certification probe's `early_read` variant
   only adds one read of `acc[0]` (a global store) between the chain's `commit_group` and `wait_group(0)`, with not a
   word changed elsewhere, and the whole kernel's wgmmas get serialised (readings in "Sources").
2. **Carrying across the back edge itself (the control-flow form) is not the trigger**: `drain` (`wait_group(0)` before
   the back edge) and `carry` (tail `wait_group(1)`, reaped by `wait_group(0)` at the top of the trip, last-trip guard
   `if it + 1 < n` + `else` drain, prologue chain not drained) have almost identical SASS wait axes (D/H 0.156 vs
   0.156), with no serialisation.
3. **When an in-flight group crosses the back edge, the instruction touching the accumulator may be one the compiler
   put there itself**: an in-flight group crossing the back edge means the accumulator is a loop-carried value at the
   loop header (the prologue chain and the tail chain merge there), and the pipeline stage spans the back edge, the
   loop header and all code at the top of the trip up to the reaping wait. The phi copies appearing in this stretch
   (`mov` of accumulator registers), a zero-fill `mov 0` sunk below `wgmma.fence`, the DAG scheduler hoisting the first
   reader above the wait… are all "non-wgmma instructions touching the accumulator inside the stage", and none of them
   are in your DSL source. The difference between a real main loop (warp specialisation, `setmaxnreg` budget,
   mbarrier spin at the top of the trip, a second RS chain in flight, 176+ registers) and a minimal probe is exactly
   the difference in these compiler decisions — so **carrying across the back edge must be accepted per cubin with
   SASS**, not judged safe by reasoning. The ladder readings of this entry's probe (see "Sources") record which level
   of form reproduced the serialisation in an isolated kernel.

## When needed (trigger)

- You just changed some `wait_group(0)` at the tail of the main loop to `wait_group(1)` (wanting the barrier wait /
  the first register work at the top of the trip to be covered by the in-flight MMA), correctness still PASSES, but
  device time gets **dramatically** worse (double-digit percent), NCU `sm__pipe_tensor_cycles_active` drops by double
  digits of pp, and `stall_barrier` rises.
- You just moved some chain's `warpgroup.wait_group(0)` (**depth unchanged**, still a drain) from the end of the body to
  **the top of the next trip** (wanting the back edge, the handoff barrier at the top of the trip and the next step's
  index publication to run under the chain), and added a final drain after the loop. The depth value is unchanged, but
  the group crosses the back edge all the same — symptoms as in the previous item: ms worse by double digits,
  `tensor_pct` down by double digits of pp, `stall_barrier` doubled, spill still 0 (readings in the "second kind of
  knob" section of "Field footprint").
- The SASS wgmma wait axis (the immediate sequence of `WARPGROUP.DEPBAR`) changes from "1–2 `LE 0x0`/`0x1` per chain"
  to "`LE 0x0` × the number of HGMMAs in the chain" — in every consumer ring, including chains you **did not** change.
- **There is no compile-time warning to rely on**: the CuTe DSL compile path does not surface ptxas's advisory; only
  a SASS census can tell (see "Machine-readable acceptance").

## Minimal runnable example

The pieces below differ only in one tail line / one read. `issue_slab(acc, stg, lo)` stands for 8 `cute.gemm` calls on
the same accumulator; `consume(acc)` is the first register segment at the top of the trip that reads the accumulator.
All fragments / atoms are bound once outside the loop (for the dynamic-loop discipline see
[`wgmma_interleave`](wgmma_interleave.md)).

### (a) Safe form — drain before the back edge (gear 0); establish this baseline first

```python
# prologue: chain(0)
acc.fill(0.0); warpgroup.fence()
issue_slab(acc, 0, lo=True); issue_slab(acc, 0, lo=False)
warpgroup.commit_group()
warpgroup.wait_group(0)

for it in cutlass.range(n_trip):
    consume(acc)                                  # first reader at the top of the trip
    if it + 1 < n_trip:                           # last-trip exit: no more issues
        stg = (it + 1) % NSTAGE
        acc.fill(0.0); warpgroup.fence()
        issue_slab(acc, stg, lo=True);  warpgroup.commit_group()
        warpgroup.wait_group(1)                   # the previous group retires (the stage it read can be returned here)
        issue_slab(acc, stg, lo=False); warpgroup.commit_group()
        warpgroup.wait_group(0)                   # <- drain before the back edge: the loop header never has anything in flight
    else:
        warpgroup.wait_group(0)
```

### (b) The always-wrong form — touching the accumulator inside the stage (the probe's positive control)

```python
        issue_slab(acc, stg, lo=False); warpgroup.commit_group()
        scratch[(bx, lane)] = acc[0]              # <- read the accumulator after the commit, before the wait
        warpgroup.wait_group(0)
```
One read, and every HGMMA in the whole kernel is followed by a `DEPBAR.LE 0x0`. The numbers are still right. In a real
kernel this read is usually not one you wrote (see point 3 of "Purpose").

### (c) Carrying across the back edge — allowed, but must be ruled on by "machine-readable acceptance" every time

```python
# prologue: chain(0) committed without waiting, left for the top of the trip to reap
...; warpgroup.commit_group()

for it in cutlass.range(n_trip):
    wait_flag()                                   # barrier wait at the top of the trip covered by the in-flight group
    warpgroup.wait_group(0)                       # reap the group left by the previous trip / the prologue
    consume(acc)
    if it + 1 < n_trip:
        ...                                       # the two-slab issue as in (a)
        warpgroup.wait_group(1)                   # <- slab 2 stays in flight, crossing the back edge
    else:
        warpgroup.wait_group(0)
```
In an isolated kernel this form is clean (`carry` D/H 0.156). After putting it into a real main loop, run the
"machine-readable acceptance" first, and if D/H fails, fall back to (a); the gain is only the short stretch between
the top of the trip and the first reader anyway.

## Machine-readable acceptance (mandatory after every change to wait depth)

```bash
cuobjdump -sass <kernel>.cubin | grep -cE '\bHGMMA'                      # H
cuobjdump -sass <kernel>.cubin | grep -cE 'WARPGROUP\.DEPBAR'            # D
cuobjdump -sass <kernel>.cubin | grep -oE 'WARPGROUP\.DEPBAR\.LE [^,]+, 0x[0-9a-f]+' | sort | uniq -c
```

- **Criterion**: `D / H ≤ 0.25`, and no stretch of "≥ 8 consecutive HGMMAs each immediately followed by a
  `DEPBAR.LE … 0x0`". The healthy form is 1–2 `DEPBAR`s at the end of each chain (`0x1` reaping the previous group,
  `0x0` draining).
- **Pathological form**: `D / H ≈ 1.0`, immediates almost all `0x0`, and **every** chain (including unchanged ones)
  the same.
- **Positive-form criterion (alongside the two above; neither can be skipped)**: list the non-zero depths this cubin
  **should** keep (e.g. one consumer ring covering an elementwise segment under the next chain with `LE 0x1`, another
  ring keeping two groups in flight with `LE 0x2`), and assert one by one that they still appear in the
  `WARPGROUP.DEPBAR` histogram. The fingerprint of serialisation is exactly "all non-zero immediates disappear
  together"; the whole-cubin D/H ratio and the "≥ 8 consecutive" criterion can report the disease but cannot say which
  chain's covering relation was lost; the positive form is one line per chain.
- **The census target is the shipped kernel's own cubin**: compiled through the kernel file's own compile entry with
  all compile-time constants of the real shape (shrinking only the grid immediate dimension), with `cuobjdump -sass`
  reading the cubin on disk; a clean reading from a model kernel is zero evidence for the real main loop (gotcha 3).
  Run this census **before** the GPU eval (in the same probe as the numerical contract), so a single change of wait
  position does not first burn a whole-kernel eval before showing up on the contract axis.
- For an automated version see this entry's certification probe's `census()`: counting `HGMMA`, `WARPGROUP.DEPBAR`
  and the immediate histogram, and the fraction "the instruction after an HGMMA is a DEPBAR". The probe's
  `ptx_census()` gives an extra PTX-level clue: within the two kinds of text window `wgmma.fence` → `wgmma.wait_group`
  and "loop-header label → first `wait_group`", count touches of the accumulator registers (the union of the first
  operands of all `wgmma.mma_async`) by non-wgmma instructions, and `mov`s between accumulator registers — non-zero
  means touches the compiler put inside a stage.

## Field footprint: the healthy / serialised readings of the `commit` axis on a main-loop cubin (KEEP-certified)

The driver's SASS dossier records one `commit` axis per consumer ring: the immediate histogram of `WARPGROUP.DEPBAR`
plus the `LDGDEPBAR` count. Below are readings of **the same kernel** (three-warpgroup specialisation, two consumer
replicas each with one SS chain + one RS chain, 168 registers/thread) before and after two one-line patches. The two
patches turned completely different knobs, yet produced **verbatim identical** pathological readings — this axis can
only tell you "serialised"; triage needs a second reading (see below).

Healthy reading (drain before the back edge, RS A-fragment resident in registers):

```
mma-consumer: DEPBAR {LE 0x0 x1, LE 0x1 x1} LDGDEPBAR x0
mma-consumer: DEPBAR {LE 0x0 x1} LDGDEPBAR x0
mma-consumer: DEPBAR {LE 0x0 x1} LDGDEPBAR x0
mma-consumer: DEPBAR {LE 0x0 x2} LDGDEPBAR x0
mma-consumer: DEPBAR {LE 0x1 x1} LDGDEPBAR x0
```

Serialised reading (both paths end up here):

```
mma-consumer: DEPBAR {LE 0x0 x16} LDGDEPBAR x0     # x4
mma-consumer: DEPBAR {LE 0x0 x8} LDGDEPBAR x0      # x2
```

How to read it: the whole `LE 0x1` axis disappears; each ring's `LE 0x0` count equals that ring's HGMMA count (16-HGMMA
chain → x16, 8-HGMMA chain → x8); the number of rings goes from 5 to 6 (two chains that used to share one reaping point
are split into per-instruction waits each).

### Path 1: an in-flight group crossing the back edge (form (c) put into a real main loop)

- **Knob (DSL)**: the `warpgroup.wait_group(0)` reaping the last slab at the loop tail → `wait_group(1)` (the last slab
  stays in flight across the back edge); right after `cute.arch.mbarrier_wait(...)` at the top of the trip a
  `warpgroup.wait_group(0)` as the reaping point (the last position before the accumulator's first reader); the
  prologue chain drops its own `wait_group(0)` and is reaped by the same wait at the top of the trip. The last-trip
  guard's `else` branch keeps `wait_group(0)`.
- **Footprint**: the `commit` axis changes from the healthy reading to the serialised reading (above); NCU, same job:
  `sm__pipe_tensor_cycles_active` 54.71 → 43.82 %, `stall_barrier` 1.85 → 3.58,
  **`spill_ld / spill_st` 0 → 0** (no register spill — this is a touch inside the stage, not a resource shortage).
- **Load-bearing condition**: three-warpgroup specialisation, a rotated chain (the next trip's SS chain issued at the
  tail of this trip while another RS chain is still in flight), an mbarrier spin at the top of the trip between the
  back edge and the reaping wait. Under this structure the gate window went 2.2493 → 2.9735 ms (+32%). After falling
  back to form (a) (tail `wait_group(0)`, the prologue chain draining itself, no reaping at the top of the trip) the
  axis reading and the ms were restored. The specific instruction touching inside the stage was not isolated on its own
  (code reading attributes it to the carried accumulator's `fill(0.0)` falling into the pipeline stage after the carry,
  `[to be checked]`); what is certain is only "spill is 0, so it is not path 2".

### Path 2: the RS A-fragment loses its register home (a spill turns the A operand into an LDL-defined register)

- **Knob (DSL)**: on an RS A-fragment already used as a wgmma A operand (packed bf16 pairs), open another
  `cute.recast_tensor(<the fragment's rmem view>, cutlass.Int32)` view, and after that chain's `wait_group(0)` drain,
  do an in-place RMW of the 16 32-bit slots through this view (inline PTX `mul.rn.bf16x2`). Spelling and gotchas in
  mechanism 1.4 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md).
- **Footprint**: the `commit` axis likewise becomes the serialised reading (x16 ×4 + x8 ×2, verbatim identical to
  path 1); NCU, same job:
  **`spill_ld` 0 → 1,699,968, `spill_st` 0 → 1,699,968 sectors**,
  `sm__pipe_tensor_cycles_active` 56.15 → 44.37 %, `stall_barrier` 1.69 → 3.64,
  `regs_per_thread` 168 → 168 (the budget did not move; the allocator drove the fragment into local memory, so the
  wgmma's A registers are defined by LDL — exactly one source of "a non-wgmma instruction defining the A operand" in
  this entry).
- **Load-bearing condition**: a consumer budget of 168 registers/thread, the consumer live set close to the budget
  ceiling. Under this structure the gate window went 2.18 → 2.9641 ms (+36%). After taking the second view out of the
  trace (not constructed outside a `const_expr` branch) the spill went to zero, the axis was restored, 2.1712 ms.
  **In an isolated probe the same spelling PASSES the `no_spill` contract** — the register home is the result of
  whole-function allocation, and a minimal reproduction does not carry the real live set.

### Triage: the same axis; look at `spill_ld / spill_st` first

| `commit` axis | `spill_ld / st` | Lesion | Action |
|---|---|---|---|
| serialised reading | 0 | non-wgmma touch inside a pipeline stage (path 1) | check wait-depth changes / touches of the accumulator and A registers inside the stage; fall back to form (a) |
| serialised reading | ≠ 0 | A-fragment / accumulator lost its register home (path 2) | check for a second type view on the fragment, runtime indices, the live set; **do not touch the wait depth** |

### The second kind of knob for path 1: depth unchanged, drain point crossing the back edge (KEEP-certified)

A different tree from "Path 1" above: **two** consumer warpgroups each with its own dynamic main loop (two
`mma-consumer` rings in the dossier), 72 HGMMAs per ring; what was moved is a pair of **RS-form** chains (A operand
resident in registers, B a single-buffered smem slot that the next trip's gather overwrites, so draining this pair
before the gather is mandatory).

- **Knob (DSL)**: this pair of chains is issued at the end of the body, and their `warpgroup.wait_group(0)` (**depth 0
  unchanged**) moved from the end of the body to **the top of the next trip** — after the cross-role handoff barrier
  at the top of the trip, before the gather — with a final `wait_group(0)` added after the loop. The changer's intent
  was to let the barrier wait at the top of the trip, the next step's index publication and the back edge run under the
  chain; the RS A operand's register home is rewritten only further downstream in that trip, so register lifetimes do
  not force an early drain either. Two other in-body moves in the same commit (see (d) in the next section) do not
  involve the back edge.
- **Footprint** (driver dossier `commit` axis, two consumer rings, verbatim):

  ```
  before:  mma-consumer: DEPBAR {LE 0x0 x1, LE 0x2 x8} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x2, LE 0x1 x1} LDGDEPBAR x1
  after:   mma-consumer: DEPBAR {LE 0x0 x72} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x72} LDGDEPBAR x1
  ```

  Read as above: each ring's `LE 0x0` count = that ring's HGMMA count (72), and `LE 0x1` (one ring's window covering an
  elementwise segment under the next chain) and `LE 0x2` (the other ring's two-groups-in-flight window) disappear
  entirely — **including the ring of the warpgroup that was not changed**. NCU, same job: `tensor_pct` 37.26 → 26.59 %,
  `stall_barrier` 3.45 → 7.89, `stall_long_sb` 2.38 → 1.80, **`spill_ld / spill_st` 0 → 0**, `regs_per_thread` 244 →
  243 (triage table: spill 0 → path 1, not a resource shortage).
- **Load-bearing condition**: two consumer warpgroups each holding a dynamic main loop, the carried chains in RS form,
  and a final segment outside the loop that reads the accumulator. Under this structure the gate window went 67.1738 →
  93.4879 ms (+39%). After putting the drain point back at the end of the body (the form of (d) in the next section)
  the axis reading was restored verbatim, gate 93.4879 → 66.9123 ms. The specific touching instruction inside the
  stage was likewise not isolated (`[to be checked]`); what is certain is that "the depth value did not change"
  provides no protection at all — whether a group crosses the back edge depends only on **whether there is an
  in-flight group at the back edge**, not on the immediate of `wait_group`.

### Safe alternative form (d): reordering within the body — put independent work between the commit and the end-of-body drain

To cover one more piece of work under the chain without letting the group cross the back edge, move that work
**into the same body**, after this pair of chains' `commit_group()` and before the end-of-body `wait_group(0)`; the
drain point itself does not move.

```python
for it in cutlass.range(n_trip):
    handoff_wait()                                # cross-role wait at the top of the trip: nothing in flight, wait freely
    gather(stage)                                 # overwrite the B slot: the previous trip drained at the end of its body, legal
    ...                                           # this trip's other chains / elementwise segments
    warpgroup.fence()
    issue_rs_chain(acc, a_regs, b_smem=stage)     # A resident in registers, B reads the smem slot
    warpgroup.commit_group()
    publish_next_ids(slot=other)                  # <- independent work: gmem index load + predicate arithmetic + STS
    warpgroup.wait_group(0)                       # <- the drain point stays at the end of the body: never in flight at the back edge
```

- **Work put under the chain must satisfy three conditions**: it neither reads nor writes the in-flight chain's
  accumulator; it does not rewrite the RS chain's A register home; the smem it writes is neither an operand of the
  in-flight chain nor a slot another role is reading right now (the example writes **another** index slot, made
  visible by the handoff at the top of the next trip, with no extra barrier). If any of the three fails, it is a "touch
  inside the stage" — back to this entry's pathological form.
- **Footprint**: the `commit` axis returns to the pre-change reading (the two "before" lines of the previous section),
  and the census probe shows D/H ≤ 0.25, no "≥ 8 consecutive" stretch, and both `LE 0x1` and `LE 0x2` present (the
  positive-form criterion).
- **Load-bearing condition and value**: under the same structure, (d) together with the two other in-body moves of the
  same commit (one chain issued before a gmem write-back instead, and the release credits of two smem slots moved
  before that write-back — both touching only registers and gmem, not reading those two slots) was worth only 67.1738 →
  66.9123 ms (−0.4%) relative to the unchanged tree: the reordering can only cover the small piece that was moved in
  (one index load + a few STS). This is another reading of the same rule as gotcha 6 — the ceiling of both back-edge
  carrying and in-body reordering is "the short stretch between the top of the trip and the first reader"; the gain is
  priced by segment length, and a double-digit ms change can only come from breaking the form, never from getting it
  right.


### The third reading for path 1: the back edge of a persistent tile loop + issue under a runtime predicate (KEEP-certified)

Yet another tree: a consumer role's **persistent tile loop** (outer; back edge = tile boundary) wrapping a
steady-state inner loop, three `mma-consumer` rings in the dossier.

- **Knob (DSL)**: at the tile tail, after committing this tile's last chain, load the next tile's A fragment and issue
  the next tile's first chain under **the same `warpgroup.fence()`** (two `commit_group`s), with the end-of-body
  `wait_group(0)` changed to `warpgroup.wait_group(1)` — reaping this tile's chain and letting the next tile's chain fly
  over this tile's epilogue (elementwise + packing + `stmatrix` + TMA store) and the back edge; the top of the next trip
  does `wait_group(0)` directly. The first tile is peeled before the loop; **the last tile switches the whole hoist off
  with the runtime predicate `i + 1 < n_rounds`, falling back to `wait_group(0)`**.
- **Both the argument and the self-check before landing read "clean"**: the epilogue in the window touches only the
  accumulator of the retired group and its packed fragment, not the in-flight group's accumulator or A fragment; all
  twelve contracts of the pre-landing probe passed, but its SASS census only measured the inner steady-state body
  (GMMA 16, `LDL/STL` 0).
- **Footprint** (`commit` axis verbatim):

  ```
  before:  mma-consumer: DEPBAR {LE 0x0 x1, LE 0x1 x1} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x1} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x1} LDGDEPBAR x0
  after:   mma-consumer: DEPBAR {LE 0x0 x16} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x16} LDGDEPBAR x0
           mma-consumer: DEPBAR {LE 0x0 x8} LDGDEPBAR x0
  ```

  NCU, same job: HGMMA 32 → 40, those immediately followed by `LE 0x0` 3 → **40 / 40** (ratio 0.094 → 1.0),
  `tensor_pct` 79.11 → 52.16, `stall_barrier` 1.27 → 4.7, `spill_ld / spill_st` 0 → 0 (triage table: path 1), gate
  0.214 → 0.31928 ms. **The boolean bit named `wgmma_ptxas_serialized` in the same profiler output reads 0** —
  acceptance recognises only the D/H ratio and the `commit` axis, not that bit.
- **Withdrawal = form (d)**: issue the **previous** tile's epilogue at the top of the **next** tile's body, between
  that tile's first chain's `commit_group` and its `wait_group(0)` (`if i > 0`, with one more copy after the loop for
  the last tile), the prologue left in place, and the loop's last action a drain. The `commit` axis returned to
  `{LE 0x0 x1, LE 0x1 x1} / {LE 0x0 x1} / {LE 0x0 x2}`, D/H 0.125, gate 0.31928 → 0.21839 ms; but relative to the tree
  **before both changes** it is 0.214 → 0.21839 ms, `tensor_pct` 79.11 → 78.36 — form (d) is again worth nothing here
  (gotcha 6).
- **The same carry is clean on a later tree**: the issue made **unconditional** and the end-of-body wait depth a
  **compile-time constant** (the last tile issues one extra dead chain whose result nobody reads, reading a stage
  nobody fills any more, collected by the drain after the loop; only the two data wait points hang under
  `i + 1 < n_rounds`), with the first tile still supplied by the peel before the loop, so the loop body has a single
  commit form. The no-serialise contract of the cross-tile probe (not included in this release) PASSES line by line
  (GMMA 40, `LE 0x1` ×2, `LE 0x0` ×3), KEEP evaluation D/H = 3 / 40, `tensor_pct` 84.4 → 84.86, gate 0.20329 →
  0.20098 ms; the same probe's paired timing priced it as null (−0.17% / −0.11%, noise floor of a byte-identical copy
  −0.03% [−0.50, +1.67]). Other changes also separate the two trees, **so whether "issue under a runtime predicate /
  two branches with two wait depths" is the trigger was not isolated on its own** `[INFERRED]`; what is certain is only
  that the same source intent gives D/H = 1.0 in one spelling and 0.075 in the other.

## Gotchas

1. **The blast radius is the whole function, not that one chain.** One ill-formed pipeline stage makes every
   warpgroup and every MMA chain in the kernel wait instruction by instruction — do not look for the cause in "chains
   that were not changed"; check the wait axis first.
2. **Compilation will not tell you.** On the DSL path ptxas's advisory is not available; put the SASS census on the
   acceptance checklist of every change to wait depth; a correctness PASS cannot replace it.
3. **"Clean in an isolated probe" does not mean "clean in the main loop".** Instructions touching the accumulator can
   come from register allocation (phi copies), constant sinking and scheduling; they depend on the register pressure
   and control flow of the whole stretch of code, which a minimal reproduction does not carry. An in-flight group
   crossing the back edge is safe only once it passes acceptance on **the target cubin**.
4. **While in flight, touch neither the accumulator nor the A register operand.** An RS-form A fragment (P packed into
   registers) must not be rewritten before its wgmma group retires; hoisting "the next trip's packing" above the
   in-flight group triggers serialisation just the same.
5. **The relative position of `fill(0.0)` and `wgmma.fence` is decided by the compiler.** `acc.fill(0.0)` written
   before `fence()` in the DSL source does not guarantee that the `mov 0` in PTX is before `wgmma.fence` too; when an
   in-flight group crosses the back edge, the whole top of the trip lies between the zero-fill and the previous group's
   reaping, giving sinking more room. Acceptance looks at the cubin, not the source order.
6. **The gain is small anyway.** What back-edge carrying can cover is only the short stretch between the top of the
   trip and the first reader (one barrier wait + a little register work); first establish the baseline with (a), then
   try (c), and rule with the acceptance criteria.

7. **One `commit` axis, two causes.** "A `LE 0x0` after every HGMMA" can be caused by a touch inside the stage (spill
   0), or by the A-fragment spilling to local memory (spill non-zero — the A operand defined by LDL). Look at
   `spill_ld / spill_st` before choosing a direction, or you will waste a round flipping wait depths (the triage table
   of the "Field footprint" section).
8. **A carry that passes machine-readable acceptance can still lose — when that wait also delays a stage return.**
   Another chain of the same kernel flipped its pre-back-edge `wait_group(0)` into a carry (form (c)),
   and the tail-carry probe passed all four contracts (D/H within the line at both tiers, no long
   HGMMA-DEPBAR stretch, numbers PASS), a clean cubin; but the carry delayed the return of the stage that chain reads to
   the top of the next trip, the ring on the producer path had no cross-trip lookahead, device 2.2617 → 2.4774 ms
   (−9.5%), tensor 54.35 → 50.31 %.
   Rule: a carry can only cover the short stretch between the top of the trip and the first reader; if this wait point
   also returns a stage, the cost of delaying the return is usually larger than the stretch covered. **Only wait points
   that release no stage at all** are worth trying a carry on.

9. **With the `wait_group` immediate unchanged, a group can still be carried across the back edge.** Whether there is a
   carry depends only on whether there is an in-flight group at the back edge: moving a depth-0 drain from the end of
   the body to the top of the next trip is, in ptxas's eyes, the same thing as flipping the end-of-body `wait_group(0)`
   to `wait_group(1)` (the "second kind of knob" section, both consumer rings at x72 together). "Depth / wait target /
   covering relation all unchanged" is not a safety argument; write the trigger condition on the acceptance checklist
   as "the position of the drain point relative to the back edge changed", not "the wait depth changed".
10. **When one commit has several wait / credit moves, serialisation comes from only one of them, and a whole-kernel
    eval cannot tell which.** Put the SASS census (including the positive-form criterion) and the numerical contract in
    the same probe, run before the eval: which move crossed the back edge is visible at a glance on the `commit` axis,
    with no need to burn a whole-kernel eval and then back out one move at a time.

11. **A constructive argument that "the window only touches operands of the retired group" is not enough, and neither
    is a census of the inner steady-state body.** A carry across the tile back edge was argued before landing and passed
    all twelve probe contracts, yet the whole-kernel evaluation read all 40 / 40 HGMMAs serialised (the "third reading"
    section). Two disciplines: ① the pre-landing SASS contract measures `WARPGROUP.DEPBAR.LE 0x0 / HGMMA` over **the
    whole function**, not a local count of some loop body — serialisation is whole-function, and a local body's "GMMA
    16, `LDL/STL` 0" is blind to it; ② make the number of in-flight groups at every `wait_group(N)` a **compile-time
    fact**: at boundary tiles prefer issuing one extra dead chain nobody reads and draining uniformly after the loop,
    rather than choosing with a runtime predicate between "issue + `wait_group(1)`" and "no issue + `wait_group(0)`" (a
    dynamic wait depth is not expressible anyway; issuing only one group and then `wait_group(1)` lets later segments
    read an accumulator still in flight). The causality of ② was not isolated on its own `[INFERRED]`, but it is the
    only spelling that read a clean cubin under the same intent.

## Sources

- Semantics of `wgmma.fence` / `wgmma.commit_group` / `wgmma.wait_group` and the constraint "the accumulator of an
  asynchronous wgmma must not be accessed before the wait": PTX ISA, Asynchronous warpgroup level matrix
  multiply-accumulate instructions (https://docs.nvidia.com/cuda/parallel-thread-execution/).
- CuTe DSL API symbols: `cutlass.cute.nvgpu.warpgroup.fence / commit_group / wait_group`,
  `warpgroup.OperandSource.RMEM`, `cutlass.range` (dynamic trip count), `cutlass.const_expr`,
  `cute.gemm`, `cute.arch.mbarrier_init / mbarrier_arrive / mbarrier_wait`,
  `cute.arch.warpgroup_reg_alloc / warpgroup_reg_dealloc`,
  `cutlass.utils.hopper_helpers.make_trivial_tiled_mma`.
- GPU-certified probe — nine variants in the same job, each in its
  own subprocess; contracts: **numeric** (every variant == the exact reference, `max_diff == 0.0`), **serial-sass**
  (`drain` has `WARPGROUP.DEPBAR / HGMMA ≤ 0.25` and `early_read ≥ 0.9`), **serial-time**
  (same-job paired `ms(early_read) / ms(drain) ≥ 1.15`). The D/H and times of the ladder variants `carry`,
  `carry_spin`, `drain_pv`, `carry_pv`, `carry_pv_spin`, `carry_pv_spin_wide`, `carry_pv_spin_wide_reg` are INFO
  readings, to be backfilled here after certification.
  First-round readings (three-variant version): `drain` D/H 0.156, `carry` (conditional tail + back-edge carry) 0.156,
  unconditional carry 0.125, time ratios 0.98 / 0.97 — the control-flow form alone does not trigger. Status: ✅ GPU-certified (PROBE PASS max_diff=0.0, all three contracts PASS).
- Contract-axis readings on the main-loop cubin (the "Field footprint" section): from the NCU/SASS dossier census of
  every driver evaluation, one each before and after the two one-line patches; the tail-carry probe
  (four contracts instrument / sass-drain / sass-carry / numeric PASS) is the certification probe of gotcha 8. Status: ✅ KEEP-certified.
- Main-loop cubin contract-axis readings of the "second kind of knob" and of form (d): NCU/SASS dossier censuses of
  driver evaluations, one before and one after the drain point crossed the back edge and one after putting it back at
  the end of the body; the cubin of form (d) was censused by a wait-census probe through the shipped
  kernel's own compile entry — `PROBE PASS` (6/6: numeric `max_diff=0.0557904`, repeated launches bit-for-bit
  identical, `WARPGROUP.DEPBAR / HGMMA ≤ 0.25`, no "≥ 8 consecutive HGMMAs each followed by a DEPBAR" stretch, both
  covering depths `LE 0x1` and `LE 0x2` still present). Status: ✅ KEEP-certified.
- The "third reading" section and gotcha 11: dossier censuses of the two KEEP evaluations, serialisation and
  withdrawal (eval job numbers not entered in the status record), `commit` axis three rings
  `{LE 0x0 x1, LE 0x1 x1} / {LE 0x0 x1} / {LE 0x0 x1}` → `{LE 0x0 x16} ×2 + {LE 0x0 x8}`
  → `{LE 0x0 x1, LE 0x1 x1} / {LE 0x0 x1} / {LE 0x0 x2}`, gate 0.214 → 0.31928 → 0.21839 ms;
  the clean-carry readings come from the cross-tile probe (`PROBE PASS`, the
  no-serialise / census-stable / located / role-split contracts all passed in the same round, numerical line
  `max_diff=2.601962e-06`) and its KEEP evaluation.
  ⚠️ **The certified version is the third version of the probe within the same round**: the first two failed earlier in the same
  iteration — the first version 5/6 (stuck on `C4-body-lines`: KV main loop LDL/STL == 0, NANOSLEEP == 0, GMMA ≥ 12),
  the second version 6/7 (stuck on `C6-region-sound`: the attribution must cover ≥ 80% of the dominant function,
  `cons.body` must contain 12..24 GMMAs and two DEPBAR depths). **Whether the criteria were relaxed until they passed
  or corrected was not checked by a control in this entry**, so the attribution of the trigger stays `[INFERRED]`, and
  the next run must recheck the criteria's strength itself before citing it.
  Status: ✅ KEEP-certified (trigger attribution `[INFERRED]`).
