# pipeline_tma_umma_stage_count (the **stage-count knob** of a `PipelineTmaUmma` operand ring: which derived quantities one constant drives, how the per-CTA smem limit is accounted, which dossier axes can / cannot see it, under what conditions it is load-bearing)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the counterpart of [`ring_depth_lever`](../sm90/ring_depth_lever.md) (sm90 tree, not in the read set of an sm100 run) on
> **DSL pipeline objects**: that one covers a hand-written `MemRange` + `i % STAGE` ring; this one covers the ring whose index / phase is maintained by
> `cutlass.pipeline.PipelineTmaUmma` itself — **which** derived quantities a depth change lands on, how the smem account is settled, which dossier readings are the footprint of depth and which merely follow codegen,
> plus the readings of one three-point sweep (including its noise band).
>
> The certification evidence is **KEEP evaluations of contract-axis changes** (the second kind of evidence in this index): three patches that each changed only one
> constant; their NCU dossier census and gate windows are listed verbatim in the "Machine-read footprint" section below.

## Purpose

In Blackwell's TMA → smem → UMMA main loop, the stage count of the operand ring decides how many k-blocks the producer (the warp issuing TMA)
can run ahead of the MMA issuer: each extra stage hides one more stretch of TMA latency in front of the issuer;
the cost is the full operand-face bytes of each stage.

## When needed (trigger)

- In the main loop the MMA issuer stalls on the **full wait** (on the timeline, the producer's TMA completion lands after the issuer's
  wait starts; `stall_long_sb` high while tensor occupancy is not full) — **first prove that the binding edge is supply**, then touch the depth.
- Conversely: smem is at the ceiling, another buffer needs room, and you want to make the operand ring shallower to free bytes.
- Or you simply want to turn "the stage count" from literals scattered around into one sweepable knob — that is the main point of this entry's example.

## Minimal runnable example

The spelling comes down to one sentence: **all derived quantities are computed from the same module constant**; changing the depth = changing this one number.
Extracted below from the patch of a three-point sweep (operator logic removed; the main-loop skeleton follows
[`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md)):

```python
import cutlass, cutlass.cute as cute, cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils

NSTAGE = 6                 # operand ring depth: the only knob (this initial value is back-computed from the limit per gotcha 2, not copied)
ACC_BUFFERS = 2            # accumulator ring depth: another ring, another barrier table, independent of NSTAGE

# (1) smem account: fixed part + NSTAGE × bytes per stage + tail (barrier table + misc).
#     This number is what is passed as smem= at launch, and also the smem_dynamic read back by ncu.
#     Bytes per stage = the sum of the bytes of all operand faces of one stage, set by the face layouts and independent of NSTAGE:
#     compute it face by face with cute.size_in_bytes(dtype, single-stage face); do not write a literal (hard-coded, it is no longer a knob).
PER_STAGE_BYTES = sum(cute.size_in_bytes(dt, L1) for dt, L1 in operand_faces_one_stage)
FIXED_BYTES = ...                                # non-ring buffers (e.g. epilogue staging), independent of NSTAGE
TAIL_BYTES = ...                                 # barrier table + misc, independent of NSTAGE
SMEM_BYTES = FIXED_BYTES + NSTAGE * PER_STAGE_BYTES + TAIL_BYTES
assert SMEM_BYTES <= 232448                      # sm100 per-CTA opt-in dynamic smem limit (227 KiB)

# (2) faces: the helper's last positional argument is the stage count; the last mode of a face is the stage mode
def build_faces(tiled_mma, tiler_mnk, a_dtype, b_dtype):
    a_smem = sm100_utils.make_smem_layout_a(tiled_mma, tiler_mnk, a_dtype, NSTAGE)   # (…, STAGE)
    b_smem = sm100_utils.make_smem_layout_b(tiled_mma, tiler_mnk, b_dtype, NSTAGE)
    # the TMA descriptor only looks at one stage: slice off the trailing stage mode (compute the arity from the rank, do not hard-code it)
    def one_stage(L):
        return cute.slice_(L, (None,) * (cute.rank(L) - 1) + (0,))
    return a_smem, b_smem, one_stage(a_smem), one_stage(b_smem)

@cute.kernel
def kernel(...):
    smem = cutlass.utils.SmemAllocator()
    ...                                            # faces allocated by layout.outer / swizzle=layout.inner (same as the previous entry)
    # (3) barrier table: NSTAGE Int64 each for full + empty; the accumulator ring gets its own table
    ab_mbar = smem.allocate_array(cutlass.Int64, 2 * NSTAGE, byte_alignment=16)
    acc_mbar = smem.allocate_array(cutlass.Int64, 2 * ACC_BUFFERS, byte_alignment=16)
    # (4) pipeline objects: num_stages is NSTAGE; tx_count is the bytes of "one stage" (multiplied by the pair size in the pair form),
    #     it does not change with NSTAGE
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=ab_mbar, num_stages=NSTAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=PAIR_TX, cta_layout_vmnk=cta_layout_vmnk).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=acc_mbar, num_stages=ACC_BUFFERS, ...).make_participants()

    # (5) steady state: the stage coordinate is read from the handle; there is no `% NSTAGE` anywhere in the code
    for kb in cutlass.range(num_k_blocks, unroll=1):
        tok = ab_consumer.wait_and_advance()       # wait on full[state.index], hand back a handle
        s = tok.index                              # the stage consumed this time
        ...                                        # issue MMA with stage s's faces
        tok.release()                              # empty[s] (on the handle, see pipeline_umma_async_handle_commit)
```

**Changing the depth touches only the `NSTAGE` line**, and five things follow automatically: the declared smem bytes (`smem=` at launch),
`allocate_array(Int64, 2 * NSTAGE)`, `PipelineTmaUmma.create(num_stages=NSTAGE)`, the stage argument of each face helper
and their multi-stage TMA destination views, and the byte self-check against the stage count. **Unchanged** are:
`tx_count` (the bytes of one stage), the accumulator ring (`ACC_BUFFERS` and its own barrier table), the TMEM column count,
threads and warp roles, the wait encoding and the peek position.

## Machine-read footprint (what to look at after changing the depth, and what not)

On one tree (sm100 pair UMMA, `cta_group::2`, 1 CTA/SM, one resident wave, supply granularity = consumption granularity),
three KEEPs that **changed only this one constant**, with the readings of their eval jobs verbatim — **the eval job numbers of these three evaluations were not recorded in the status
record, so this entry has no clickable job pointer**; for the certification boundary see "Sources":

| NSTAGE | `smem_dynamic` | dossier `mbar_objs` | `mbar_tiers` | contract axis `rings` | gate (device window) | same-job reference ratio | `tensor_pct` | `stall_long_sb` | `stall_mio` |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 202752 | 24 | CONST 20 / RING 4 | 4: `stride=0x1..0x4,wrap=sel` | 0.3756 ms | 0.9554 | 81.21 | 17.03 | 0.24 |
| 5 | 170496 | 21 | CONST 18 / RING 3 | 3: `stride=0x1..0x3,wrap=sel` | 0.378 ms (+0.64%) | 0.9527 | 81.74 | 17.62 | 0.09 |
| 4 | 138240 | 18 | CONST 16 / RING 2 | 2: `stride=0x1..0x2,wrap=sel` | 0.400 ms (+5.82%) | 0.901 | 81.39 | 18.8 | 0.08 |

| Look at | Reading |
|---|---|
| `smem_dynamic` | **the only direct footprint**: the difference between two adjacent rows is the face bytes per stage, matching formula (1) byte for byte |
| count axes `mbar_objs` / `mbar_tiers` | 3 objects per stage, 2 per stage in the CONST tier, 1 per stage in the RING tier — linear in depth |
| launch reports no smem overflow, no hang | all three points PASS correctness; depth has only two failure classes, "over the limit" and "phase error", see gotcha 4 |

| **Do not look at** | Why |
|---|---|
| the number of entries on the contract axis `rings` | on this tree it happens to be NSTAGE − 2 with strides `0x1..0x(NSTAGE−2)`, so it **looks** like it encodes depth; but on the same tree a patch that changed no ring at all (only swapped the swizzle of the epilogue staging face) pushed it from 3 entries to 4. It is a codegen fingerprint of the phase / index arithmetic chain ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11); read depth from the two rows above |
| `cache_ops`, `nbar_*`, `supply`, `merge` | they did not move at all across the three points — a depth change should not touch them; if they move, the spelling has hard-coded the depth somewhere (gotcha 1) |

## The same knob on a hand-written mbarrier ring (no pipeline object; the ring computes `cg % DEPTH` and parity itself)

`PipelineTmaUmma` maintains index / phase for you; on a hand-written mbarrier ring (single CTA, `setmaxnreg` multi-role, the supply
warp issues TMA, the MMA issuer waits on full, and after consuming does `tcgen05.commit` to empty), the depth enters the code in **only
four places**; all of them must be written as functions of the depth constant for a depth change to be "change only one constant":

```python
K_SLOTS = 3                                   # operand ring depth: the only knob (change it and everything below follows)
Q_SLOTS = 2                                   # each other ring has its own constant, independent

# (1) barrier table: each offset is derived from the previous offset + the depth of the ring it indexes, and the table length follows.
#     With a table written as literals, every depth change needs edits in two unrelated places; a table one entry shorter than the slots it indexes
#     means two rings share the same 8 B word -- an error of this kind has no numerical symptom at all.
BAR_K_FULL = 0                                # + slot   <- TMA expect_tx (1 arrive)
BAR_K_EMPTY = BAR_K_FULL + K_SLOTS            # + slot   <- the consumer side's tcgen05.commit (1)
BAR_Q_FULL = BAR_K_EMPTY + K_SLOTS
BAR_Q_EMPTY = BAR_Q_FULL + Q_SLOTS
...                                           # all later non-ring barriers are derived from the previous + 1
MBAR_COUNT = <last offset> + <its slot count>  # the table length is derived, not counted
mbar = smem.allocate_array(cutlass.Int64, MBAR_COUNT, byte_alignment=16)

# (2) faces: the helper's stage argument is the depth (sm100_utils.make_smem_layout_a(..., K_SLOTS)),
#     and every view that uses it as the B operand also carries K_SLOTS stages; the TMA descriptor is still built from a single stage.

# (3) slot index and parity: shared by the three K sites (supply issues TMA, MMA waits on full, commit to empty)
slot = cg % K_SLOTS
#     the supply side's empty wait: parity (cg // K_SLOTS - 1) % 2, and only waits when cg >= K_SLOTS
#     -- the first K_SLOTS chunks skip the empty wait; that is the prologue's warm-up beats, no separate prologue needed;
#     the MMA side's full wait: parity (cg // K_SLOTS) % 2.
```

**With the depth-independent spelling, the diff contains only the `K_SLOTS` line** (plus comments): the barrier table, the faces' stage count, the slot index,
both parities and the prologue guard are all functions of it; the ring shape (one full / empty pair per slot, the stage mode of the same staged tensor
indexed in three ways, the issue order of each ring), the arrive counts and the depths of other rings are all untouched. The cost is one more
copy of that operand's face bytes per extra stage (16 KiB in this example) + one more pair of mbarriers.

**Certification readings (KEEP evaluation, NCU dossier census + gate window; eval job number not recorded; single CTA, hand-written mbarriers,
`setmaxnreg` with three roles, 1 CTA/SM)** — operand ring 2 → 3, only one constant changed:

| Axis | Before | After |
|---|---|---|
| contract axis `rings` | `[stride=0x1, stride=0x1,slots=536870911]` | `[stride=0x1,slots=536870911, stride=0x2]` |
| count axes `mbar_objs` / `mbar_tiers` / `mbar_per_role` | follow (one more barrier pair) | follow |
| gate (device window) | 0.12804 ms | 0.12602 ms (−1.58%) |

Reading: `rings` gains a `stride=0x2` entry and the earlier bare `stride=0x1` disappears; **no `slots=3` appears**
— consistent with the conclusion above that "`rings` does not encode depth" (the readings of a third tree, summarised in
[`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11); the footprint of depth is, as before, the barrier object count
and the dynamic smem bytes.

**Load-bearing conditions (worth −1.58% this time, only under the structure below)**: the consumer side uses this operand from the chunk's first MMA group
to its last, then reads it once more in the next chunk's lookahead window, so depth 2 is the minimum that runs; depth 3
switches the empty wait on which the supply warp blocks for chunk c + 2 from "the commit after chunk c's last MMA group"
to "the one of chunk c − 1" (long since flipped), so this path's TMA is issued as soon as the previous loop iteration frees a slot on the other ring
— the supply warp runs two chunks ahead of the MMA issuer instead of one. **This reading nearly coincides with the noise floor of same-job
ablations on the same tree (about 1.5%)**; its direction matches the structural argument above, but it is not enough to price on its own; on the same tree,
deepening other rings (those on whose empty the supply warp does not wait) read as zero increment (change record: Q's third slot
"reads as zero increment") — depth is load-bearing only when "the supply side really blocks on this ring's empty", the same shape as the conclusion of gotcha 6.

**Certification boundary [UNVERIFIED-BOUNDARY]**: what this section's KEEP certification pins down is the **spelling** (depth enters only four places, the diff is down to one constant, correctness PASS) and
the **footprint** (`rings` shows no `slots=3`, the barrier object count follows). Whether "depth 3 is worth it" was **not ruled on** for this tree: −1.58% is inside the noise floor of same-job ablations;
the build chain this KEEP belonged to was later abandoned as a whole, and the code with which the same tree finally met its bar (> 1.05x) **kept this ring at depth 2**
and did not put 3 back. The readings can only show that "deepening does not hurt correctness and is not seen by the `rings` accounting"; they must not be used as a prior that "deepening a hand-written ring = gain"; back-compute the depth
from your own smem limit as described above and sweep multiple variants in the same job.

**Second set of certification readings (KEEP evaluation; another tree: pair UMMA, 512 threads with four roles, a hand-written mbarrier table, 1 CTA/SM)** —
what was deepened this time is not the TMA operand ring but **the smem face written by a register role and read by UMMA as the A operand**, 1 → 2 slots:

| Axis | Before | After |
|---|---|---|
| `smem_dynamic` | 186904 | 199208 (+12288 = the second face, +16 = two barriers) |
| count axes `mbar_objs` / `mbar_tiers` | 30 / `{CONST: 29, RING: 1}` | 33 / `{CONST: 32, RING: 1}` |
| contract axis `rings` | `[stride=0x1]` | `[stride=0x1,slots=9]` (no `slots=2`) |
| gate (device window) | 1.2415 ms | 1.2474 ms (+0.48%, in the same package as a fast path of the elementwise segment, not separable) |
| `tensor_pct` / `stall_long_sb` / `stall_barrier` | 44.04 / 6.34 / 2.11 | 44.14 / 6.12 / 2.01 |

This face differs from the TMA operand-ring spelling in three ways, all of them the form actually shipped on the certified tree:

- **The slot index is the parity of the global chunk counter, not the in-tile ordinal.** When the number of chunks per tile is odd, taking the slot by in-tile ordinal
  puts the two adjacent chunks on either side of a tile boundary into the same face.
- **The two slots are two base addresses under the same single-face layout (`base + face element count × slot`), not a stage mode added to the layout.**
  The face is a compile-time operand of the MMA: the read side is written per slot as a branch in the MMA issue warp, both branches issuing the same
  instructions on their own base address; on the write side the store view takes the slot as a leading dynamic mode. Both sides read the same face-element-count constant, so the offsets are not computed separately.
- **After deepening, "the face is empty" no longer implies "the previous chunk's MMA has landed".** If the same role also has a path that requires the previous
  chunk's MMA to be complete (on this tree, a read-modify-write of the accumulator TMEM), it must **wait for the flip of the other slot itself**.
  An mbarrier wait does not consume the phase, so sharing this edge with the chunk that already waits on that flip costs nothing.

**Load-bearing conditions**: under the structure "the role writing this face is busy 1120 per chunk, all its waits total 247 (timeline readings of the same tree before deepening,
in the tool's own units)" it is worth +0.48% (same package, no value); deepening only has something to save when the write side really blocks on this face's empty,
the same shape as the previous set of readings and as gotcha 6.

## Gotchas

1. **"Only the depth changed" is a checkable assertion.** With the depth-independent spelling the diff contains only one constant; the derived quantities (smem bytes,
   barrier table length, `num_stages`, the faces' stage argument, the multi-stage views, the byte self-check) are all functions of it. Whenever the diff
   shows a change to `tx_count`, the register level, a wait site or an arrive count, the spelling is not depth-independent. The three sweep points all having
   `regs_per_thread` 74 and identical `cache_ops` / `nbar_widths` / `supply` is the reading of this assertion.
2. **The limit is 232448 B/CTA; the candidate range can only go down from the limit.** Use formula (1) to compute the first depth that crosses this line;
   the level just before it is this tree's ceiling and also the starting point of the sweep; below that is the candidate range. Compute this row first, then decide which points to sweep
   (the three adjacent depths in the table above were chosen this way).
3. **Two rings, two barrier tables, independent of each other.** The operand ring (`PipelineTmaUmma`, `2 * NSTAGE` Int64) and
   the accumulator ring (`PipelineUmmaAsync`, `2 * ACC_BUFFERS`) each get their own `allocate_array`; changing the depth of one leaves the other's
   table, tx and commit untouched. Merging the two tables into one region and slicing offsets by hand is the typical way to hard-code the depth.
4. **On a 1 CTA/SM tree, the bytes freed by a shallower ring buy nothing.** At all three points grid and block are unchanged, and even the shallowest level's
   `smem_dynamic` exceeds half the SM capacity, so occupancy does not move; going shallower leaves only the cost with no gain, unless another buffer
   is waiting for those bytes.
5. **The noise band of one reading (written down so +0.64% is not taken as a signal).** Two confirmation jobs of the same tree read the same-job
   reference ratio as 0.95 and 1.12 (18% apart); one node once read a device ms 20.8% higher than the KEEP reading and was
   rejected by the loop as a suspicious node. So the +0.64% of 6 → 5 is inside the noise band; this entry's ordering conclusion rests on the 6.5% difference
   between 4 and 6 and on the same-job reference ratio moving in the same direction (0.9554 → 0.9527 → 0.901). To distinguish two adjacent depths at ±1%,
   you need multiple variants in the same job, not cross-job comparison of ms.
6. **One load-bearing reading (what it was worth under which conditions).** Structural conditions: pair UMMA, 1 CTA/SM, one resident wave, rasterisation already keeping
   the working set inside L2, supply granularity = consumption granularity (one stage feeds exactly one consumption). Under these conditions depth is **monotonic**:
   each stage fewer hides one stretch of TMA latency less; `stall_long_sb` in the table above rises point by point and device ms moves with it; **the smaller
   concurrency footprint bought by going shallower is worth nothing here** (the working set is in L2 anyway). **This is this tree's reading, not a general percentage,
   much less the answer to "what depth to pick"** — back-compute the candidate range from your own tree's limit per gotcha 2 and sweep the order yourself: on trees where the mid-stage role's
   own segment length limits the lead, depth-type levers price at zero ([`ring_depth_lever`](../sm90/ring_depth_lever.md) gotcha 6).
7. **When going back to depth 6 is in the same package as another fix, the reading cannot be split.** 4 → 6 shipped together with "the accumulator ring's commit put back on the handle"
   read 0.400 → 0.3481 ms (−12.97%), `tensor_pct` 81.39 → 89.2; per the table above, the depth's own share is only
   0.400 → 0.3756, and the rest belongs to [`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md).
   Same-package readings are recorded, not split.

8. **The depth of a hand-written ring is also "change only one constant", but that has to be guaranteed by the offset chain of the barrier table.** A pipeline object writes the
   barrier table length as `2 * NSTAGE` and that is it; if a hand-written ring's table uses literal offsets, the depth is hard-coded both in the table and in the slot-index
   arithmetic, and a table one entry short means two rings share one 8 B word, with no numerical symptom. The spelling and one 2 → 3 reading are in the previous section.

## Sources

- **CUDA C++ Programming Guide** — the *Compute Capabilities* table: for compute capability 10.x the maximum dynamic shared memory per thread block that can be
  opted into is 227 KiB (= 232448 B)
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications>
- **CuTe DSL public API symbols**: `cutlass.pipeline.PipelineTmaUmma.create(barrier_storage, num_stages, producer_group,
  consumer_group, tx_count, cta_layout_vmnk)` / `.make_participants()`; `cutlass.pipeline.PipelineUmmaAsync.create`;
  `.index` of the handle returned by `PipelineConsumer.wait_and_advance` / `PipelineProducer.acquire_and_advance`;
  `cutlass.utils.blackwell_helpers.make_smem_layout_a / make_smem_layout_b` (last argument = stage count);
  `cutlass.utils.SmemAllocator.allocate_array`; `cute.slice_` / `cute.rank`.
- **KEEP evaluations** (NCU dossier census + gate window, the second kind of evidence in this index; eval job numbers not recorded in the status record):
  three patches that changed only `NSTAGE`, 6 → 5 → 4: axis `rings` 4 → 3 → 2 entries, `mbar_objs` 24 → 21 → 18,
  `smem_dynamic` 202752 → 170496 → 138240 B, gate 0.3756 → 0.378 → 0.400 ms; all three points PASS correctness.
  Going back to the deepest level shipped in the same package as the accumulator-ring handle fix (gate 0.400 → 0.3481 ms).
- **Certification boundary (this entry has no probe and no job pointer)**: depth is a single constant and numerically invisible, so it goes through the index's
  second kind of evidence (KEEP census of contract-axis changes), **not** probe certification; and the eval job numbers of these three KEEPs were not recorded in the status
  record (a gap on the harness side). The only job number that could be pointed to is the watermark confirmation of the same-package landing — it only proves that landing's
  reading and is not evidence for this three-point sweep, so it is **not written into this entry**: this entry therefore does not pass the driver's
  machine check that "a KB entry needs a probe verdict or a job pointer to count as a price", and that is intentional — **this entry cannot be used to price a reopen**; it is a reading recipe.
- **KEEP evaluation, hand-written mbarrier ring form** (NCU dossier census + gate window, the second kind of evidence in this index; eval job number
  not recorded): on a tree with a single CTA, `setmaxnreg` three roles and 1 CTA/SM, operand ring 2 → 3, the diff only one constant (the barrier
  table offsets all switched to derived expressions in the same package): axis `rings` `[stride=0x1, stride=0x1,slots=536870911]` →
  `[stride=0x1,slots=536870911, stride=0x2]`, `mbar_objs` / `mbar_tiers` / `mbar_per_role` follow,
  gate 0.12804 → 0.12602 ms (−1.58%, nearly coinciding with that tree's same-job ablation noise floor of about 1.5%); correctness PASS. **[UNVERIFIED-BOUNDARY]**: the build chain this KEEP
  belonged to was later abandoned, and in the code with which the same tree finally met its bar this ring is still at depth 2 — this section certifies the spelling and the footprint, not the gain of depth 3 (see "Certification boundary" in the section above).
  `cute.arch.mbarrier_wait` / `mbarrier_arrive` / `cutlass.utils.SmemAllocator.allocate_array` are public API symbols.
- **KEEP evaluation, register-role-written A operand face 1 → 2 slots** (NCU dossier census + gate window; eval job number not recorded in the status record;
  in the same package as the elementwise segment's fast path, correctness PASS): axis `rings` `["stride=0x1"]` → `["stride=0x1,slots=9"]`,
  gate 1.2415 → 1.2474 ms; `smem_dynamic` 186904 → 199208, `mbar_objs` 30 → 33.
- Related entries: [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) (how the ring is built),
  [`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md) (commit / release on the handle),
  [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11 (reading the `rings` axis),
  [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md) (another form in which depth-type levers price at zero).
