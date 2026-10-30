# per_slot_landing_depth (**per-slot** stage depth of a single-level landing zone: which slots are deep and which are shallow within one landing)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is the **depth-dimension
> complement** of [`pipeline_runtime_subring`](pipeline_runtime_subring.md) and
> [`cp_async_mbarrier_ring`](cp_async_mbarrier_ring.md): those two cover "N sub-rings + one independent
> pipeline stream each"; this one covers "**uneven stage depth across the slots of the same landing
> zone**" — some data one deep, some two blocks deep, because within the bounds the smem envelope allows
> they differ in sensitivity to latency.

## Purpose

On the **single-level TMA landing zone of a fused main kernel**, different operand streams differ in
sensitivity to "supply arrival latency": one stream is a critical prerequisite of the next MMA block —
the earlier it arrives, the earlier compute starts; another stream's consumption point is far from its
`full` wait (it gets its first reader only after other segments), so its need to be covered by stage
depth is much weaker. Give the former 2 stages and the latter 1 stage, so the smem budget is spent where
**depth can pay off**, instead of giving every stream an extra deep pocket (that only takes more smem
without waiting less).

## When needed (trigger)

- Structural symptom: a fused main kernel with several operand streams sharing one smem envelope
  boundary (`smem_dynamic` already a few hundred B under the cap); in NCU some stream's `tensor_pct` is
  low-to-mid and `stall_barrier` is healthy, but `smem_dynamic` does not allow "deepen every stream by
  one level".
- **Trigger criterion (distance axis)**: the streams differ in **the distance from their `full` wait
  point to their first consuming GEMM**. A landing zone with uniform distances (every slot read at the
  top of the trip) has nothing to split into deep and shallow, and this entry does not apply — "deepen
  all slots by one level" was ruled a negative gain on that structure because smem fills up while the
  waits do not shrink, not because depth is not a lever.
- **Key prerequisite (granularity axis, orthogonal to the previous one)**: **consumer atom granularity =
  supply stage granularity** — one stage is exactly one bite on the consumer side. When the
  granularities differ, the depth knob couples with granularity and this entry degenerates into the
  granularity-coupling exception; don't use it. The two criteria say two different things: the distance
  axis decides **which slots should be deep**, the granularity axis decides **whether this entry can be
  used at all**.
- Direct profiler signal: deep/shallow comparison — changing "all slots 1-stage" to "two deep slots one
  level deeper, the other four kept at 1-stage" gave `tensor_pct` 22.19 → 26.79 (+4.6pp),
  `smem_dynamic` 206848 → 231424 B (1024 B under the cap), gate 0.9676 → 0.8482 ms (−12.3%).

## Minimal runnable example

> The code below is **pseudocode**: variable names come from a concrete tree, and `C` / `DH` / `NST` are
> that tree's tile constants (chunk length / head dim / stage count of the deep slots). API names are
> the real CuTe DSL names (`cute.tile_to_shape` / `cute.composition` / `cute.local_tile`); shapes and
> orders must be redetermined for another tree.

### 1) Classify the dataflow: which slots should be deep, which shallow

First order the streams by the distance "`full` wait point → the operand's first consuming GEMM", then:

- **Deep-stage slots**: streams read by the first GEMM **immediately** after the `full` wait point —
  they are critical prerequisites, and one more level of depth keeps two adjacent blocks in flight
  (the producer supplies one beat ahead), hiding the "idle wait for it to arrive".
- **Shallow-stage (1-stage) slots**: streams whose first reader is **further downstream in the trip**
  (after several independent segments) — their arrival latency is already covered by the earlier
  segments, and one more level only takes more smem.

### 2) Staged layout + transposer for deep slots; move the `full` wait of shallow slots down

```python
# (a) separate layout for deep slots: stage mode OUTERMOST (one whole block per stage, sliceable across stages)
l_k2 = cute.tile_to_shape(atom_128b, (C, DH, NST), order=(0, 1, 2))
l_a2 = cute.tile_to_shape(atom_64b,  (C, C,  NST), order=(0, 1, 2))

# (b) MN-major transposer of the deep slots: the stride of the stage dim = **the element count of that slot's whole tile**
#     (not some generic constant: k's tile is C×DH, A's tile is C×C, see gotcha 3)
t_k2 = cute.composition(l_k2, cute.make_layout((DH, C, NST), stride=(C, 1, C * DH)))
t_a2 = cute.composition(l_a2, cute.make_layout((C,  C, NST), stride=(C, 1, C * C)))

# (c) the `full` wait of shallow slots: not at the top of the trip, but right before the first segment of the trip that really reads it
for it in cutlass.range(n_iter):
    ...                                       # first do the segments that depend only on deep slots
    pipe_shallow.consumer_wait(st_shallow)    # ← moved here, not the top of the trip
    first_reader_of_shallow(...)
```

### 3) Two producer states: shallow slots feed **this block**, deep slots feed **the next block** (one beat ahead)

The producer side must give different streams different lead, otherwise "one level deeper" only takes
more smem while the producer still feeds on the same beat. **Direction-neutral**: whether block indices
ascend or descend is up to the algorithm (in a backward-recurrence kernel `c` decreases); this entry
only prescribes "deep slots feed **the next block**, and the guard is **there is a next block**":

```python
for it in cutlass.range(n_iter):
    c = chunk_of(it)          # ascending it, or descending c1 - it — decided by the algorithm, irrelevant here

    # ps1 (depth 1): shallow slot, feeds **this block** c
    pipe_shallow.producer_acquire(st_ps1)
    copy(tma_atom, g_shallow[c], s_shallow[(None, st_ps1.index)])
    pipe_shallow.producer_commit(st_ps1)
    st_ps1.advance()

    # ps2 (depth 2): deep slot, feeds **the next block** (one beat ahead)
    pipe_deep.producer_acquire(st_ps2)                 # unconditional
    if it + 1 < n_iter:                                # issue the copy only if there is a next block
        copy(tma_atom, g_deep[next_chunk_of(it)], s_deep[(None, st_ps2.index)])
    pipe_deep.producer_commit(st_ps2)                  # unconditional
    st_ps2.advance()                                   # unconditional
# Prologue: before entering the loop, issue the deep slot of the **first block** once (the warm-up beat)
```

- **Only the deep-slot copy carries the `it + 1 < n_iter` predicate**; the shallow-slot copy is issued
  unconditionally on this block.
- **acquire / commit / advance are all unconditional** (the predicate wraps only the copy, not the state
  advance).

### 4) Two consumer states: the top of the trip opens on the deep slot, the shallow slot's wait is inserted between segments

```python
for it in cutlass.range(n_iter):
    stg = st_c2.index                                # runtime stage index of the deep slot
    pipe_deep.consumer_wait(st_c2)
    oK = cute.local_tile(sK, (C, DH), (0, 0, stg))   # ← stage coordinate **last** (gotcha 4)
    seg_a_only_deep(oK)                              # segment depending only on the deep slot
    pipe_shallow.consumer_wait(st_c1)                # ← the shallow slot waits only here
    seg_b_shallow()
    ...
```

The math segments of the same warpgroup are **split into two sub-segments** — "depends only on deep
slots" and "also depends on shallow slots" — with the shallow slot's `full` wait moved before the second
sub-segment and the deep-slot segment run first.

## Gotchas

1. **Don't treat "+1 for all slots" as the same thing.** The deep-slot gain comes from "the producer one
   beat ahead + the consumer reading it first at the top of the trip"; +1 on a shallow slot only takes
   more smem (its `full` wait was already downstream of the top of the trip), and the net effect is
   filling smem without removing a single wait. When smem is close to the cap, first count "which waits
   are on the critical path", then allocate depth.
2. **Deep/shallow switching is a correctness-sensitive structure, not a pure performance setting.** Two
   producer states, two consumer states, each slot's own phase count — if the `% STAGE` depth differs on
   either side, the other side's wait points are misaligned. Such mismatches **all show up as
   silent-or-hang** (reading stale data or deadlock); CPU reasoning cannot decide them, a GPU probe is
   required (same as [`pipeline_runtime_subring`](pipeline_runtime_subring.md) gotcha 2). This entry's
   certification probe was written for exactly this (see "Sources"): its first round read the
   fingerprint "of the five outputs only dg is out of tolerance", which is exactly a phase/race issue,
   not a wrong computation.
3. **The deep slot's transposer stage stride = "the element count of that slot's whole tile", not some
   dimension within the block.** If a slot's tile is `C×DH`, take `C*DH`; if `C×C`, take `C*C` — **each
   slot computes its own**, and two deep slots on the same tree usually have different stage strides.
   The shallow MN-major transposer transposes within the block (its stage stride is an in-block
   dimension); once a deep slot gets a stage dimension, the stride must change to span the whole block;
   otherwise the stage slice reads the **neighbouring block's** bytes — finite, plausible, but wrong. For
   any "stage mode outermost" layout, the stage-dim stride of the transposer/view must follow the outer
   mode.
4. **The deep slot's operand view uses a runtime stage slice, with the stage coordinate last**:
   `cute.local_tile(sK, (C, DH), (0, 0, stg))` — the stage mode is laid out outermost, so the matching
   `local_tile` coordinate goes **last**, not first. A fixed rank-2 view + multiple stages = always reading
   stage 0, and correctness silently breaks once depth > 1.
5. **The producer's predicate wraps only the copy, not the state advance.** acquire / commit / advance
   are unconditional; only the deep slot's copy is wrapped in the "there is a next block" predicate.
   Wrapping advance in the predicate as well makes the state stall and the next round's stage index drift
   (deadlock).
6. **The prologue must warm up the deep slot's beat** (issue the first block's deep slot once first),
   otherwise on the first trip the deep slot has no "previous beat" available, and the consumer's
   `consumer_wait` at the top of the first trip waits on an empty slot.
7. **When the consumer is split into sub-segments, the end of the deep-slot segment is the shallow slot's
   wait point.** Put the shallow slot's `consumer_wait` after independent work that "reads neither the
   deep nor the shallow slot", otherwise it either blocks the critical path too early or re-exposes the
   shallow slot's arrival latency.

## Sources

- This entry is the **depth**-dimension dual of
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md) (N sub-rings, one independent pipeline stream
  each): that one splits streams by warpgroup, this one splits depth by each stream's latency
  sensitivity. The stage / phase semantics of TMA `PipelineTmaAsync`, the arrive-count budget of
  `producer_acquire` / `consumer_wait`, and "mismatches all show up as silent-or-hang" are all taken
  from that certified entry and its probe (`PROBE PASS`×3).
- **GPU-certified**: a KEEP that changed the stage counts from (1,1,1,1,1,1) to (1,1,2,1,1,2) (two deep
  slots). Three load-bearing pieces of evidence, **not including the contract axis**:
  - **smem ledger (structural closure)**: `smem_dynamic` 206848 → 231424 B, a difference of 24576 B =
    one extra block for each of the two deep slots (`C×DH×2` = 16 KiB and `C×C×2` = 8 KiB), matching
    "+1 stage for each of the two slots" **byte for byte**; and only 1024 B remain to the 232448 B cap —
    the boundary of the depth allocation is hard.
  - **Independent max_diff probe** (not included in this release; `PROBE PASS max_diff=0.0037`, 5/5
    contracts). It tests exactly this entry's three things: the stage
    stride of the deep slot's transposer (read it wrong and you read the neighbouring block), runtime
    stage slices feeding the WGMMA descriptor, and the producer one beat ahead + two sets of barrier
    phases.
  - **Whole-machine readings**: gate 0.9676 → 0.8482 ms (−12.3%), `tensor_pct` 22.19 → 26.79.
- **No trustworthy footprint on the contract axis (important)**: in that KEEP the dossier's `rings`
  reading did change from `[]` to `["stride=0x1,wrap=sel"]`, but **in the same round**
  `loops / mbar_objs / mbar_tiers / mbar_per_role / edges / itc_candidates` all moved. Per the reading of
  [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) **gotcha 12**, the appearance/disappearance
  of `rings` is a follow-on effect of register allocation, and only "the value changed **and** the count
  axes did not move" counts as a footprint — this round does not satisfy that, so **the reading is not
  used as certification evidence**. (The driver's kb-axis picked it out, which is exactly the kind of
  misjudgement gotcha 12 describes.)
- **Not a general −12.3%**: it is the sum of three things on the same tree — "deep slots supplied ahead +
  the consumer reading the deep slots first at the top of the trip + shallow-slot waits moved down".
  Carrying conditions: a fused single-level landing zone, 6 streams, consumer atom granularity = supply
  stage granularity, `smem_dynamic` at the cap. Also, the register-side `spill_ld/st` were non-zero in
  that round (left over since the compression of the previous phase) — the deep/shallow switch's
  contribution to the register ledger is **not isolated** ([UNVERIFIED-BOUNDARY]).
