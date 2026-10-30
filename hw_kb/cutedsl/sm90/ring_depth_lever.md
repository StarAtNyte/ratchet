# ring_depth_lever (the **depth knob** of a hand-built mbarrier ring: which line to change, how the smem bill settles, which dossier axes see it / do not see it, how to validate non-power-of-2 depths, when it is load-bearing)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry gives the complete
> account of one thing on an "already built per-stage mbarrier ring": **changing only the depth** —
> DSL spelling, machine-readable footprint, validation form, and one KEEP reading that was not
> load-bearing ("worthless" is itself a transferable reading — so the next run does not make ring
> depth its prime suspect again).

## Purpose

In a hand-built mbarrier pipeline (one full / empty `MemRange` pair per ring, stage index
`i % X_STAGE`, phase `(i // X_STAGE) % 2`), move the stage count of a ring as a lever on the producer's
lead: deepening lets the producer run a few more chunks without being blocked by empty; making it
shallower frees smem for other rings.

## When needed (trigger)

- The consumer side stalls on the **full wait** of some ring (timeline: the producer segment's arrive
  lands after the consumer's wait has started; or `stall_barrier` / mbarrier-wait readings concentrate
  on this ring) — **first prove that this is the binding edge**, then change the depth. The illusion
  that a depth change costs nothing and risks nothing comes from it being "one constant"; its cost is
  in the smem bill, and its benefit exists only on the binding edge.
- Conversely: when smem is at the ceiling and room is needed, first make shallower the small-table
  rings that are **not on the binding edge**.

## Minimal runnable example

When the spelling is depth-agnostic, changing the depth = changing one constant per ring + settling the
smem bill once; below is the minimal form of such a "depth-agnostic" spelling (extracted from a KEEP
change, operator logic removed):

```python
# One compile-time depth per ring. Changing depth touches only these two lines; index / phase / read views stay.
A_STAGE, B_STAGE = 3, 3           # changed from 2 / 5
CHUNK = 64
f16 = cutlass.Float16

@cute.struct
class Storage:
    # barrier arena: STAGE Int64s each for full / empty per ring; it sits before the 1024 B-aligned
    # faces below, entirely inside the alignment padding, so its bytes do not enter the smem bill (gotcha 2)
    b_af: cute.struct.MemRange[cutlass.Int64, A_STAGE]
    b_ae: cute.struct.MemRange[cutlass.Int64, A_STAGE]
    b_bf: cute.struct.MemRange[cutlass.Int64, B_STAGE]
    b_be: cute.struct.MemRange[cutlass.Int64, B_STAGE]
    sA: cute.struct.Align[cute.struct.MemRange[f16, CHUNK * CHUNK * A_STAGE], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, B_STAGE * CHUNK], 128]

# The face layout carries a trailing stage mode: element (r, c, s) -> r*CHUNK + c + s*CHUNK*CHUNK.
# Every operand read view of the face (get_tensor(layout, swizzle=, dtype=)) carries the same trailing mode;
# a third stage just lengthens this mode, no address arithmetic changes.
a_l = cute.tile_to_shape(atom_c, (CHUNK, CHUNK, A_STAGE), order=(0, 1, 2))
b_af = storage.b_af.get_tensor(cute.make_layout((1, A_STAGE)))
b_ae = storage.b_ae.get_tensor(cute.make_layout((1, A_STAGE)))

def _bar(bt, idx):                       # the idx-th pointer of a (1, N) barrier tensor
    return bt[(None, idx)].iterator

@cute.jit
def _bar_wait(bar, phase):               # non-blocking try-wait + explicit tight spin
    ok = cute.arch.mbarrier_try_wait(bar, phase)
    while ok == 0:
        ok = cute.arch.mbarrier_try_wait(bar, phase)

# Steady state: index and phase are depth-agnostic expressions
for i in cutlass.range(n_chunks):
    s = i % A_STAGE
    p = (i // A_STAGE) % 2
    _bar_wait(_bar(b_af, s), p)          # consumer: wait for chunk i's full
    ...                                   # read stage s of sA (the view carries the trailing mode; coordinate s is a runtime value)
    cute.arch.mbarrier_arrive(_bar(b_ae, s))   # release empty when done; the arrive count does not change with depth
```

The smem bill (must be settled whenever depth changes):

```
   face 1 × depth × bytes per stage + face 2 × ... + small-table ring × ...  ==  dynamic smem
   dynamic smem  ≤  shared_mem_config_size (the launch-side opt-in ceiling; may be exactly equal)
```

The bill of one KEEP: two 64×64 16-bit face rings 2 → 3 (+16,384 B), more than the 14,336 B of headroom
at the time, so two small-table rings 5 → 3 (−2,048 B) went in as the enabler of the same change;
dynamic smem 186,368 → 200,704 B **== `shared_mem_config_size`**, at the ceiling to the byte without
an error.

## Machine-readable footprint (what to look at after a depth change, and what not)

| Look at | Reading |
|---|---|
| dynamic smem bytes | the only direct footprint (186,368 → 200,704 in the example) |
| dossier count axes such as `mbar_objs` | the number of barrier objects changes with the sum of STAGEs (46 → 42 barriers in the example) |
| launch: no smem-over-limit error, no hang | see "Validation form" |

| **Do not look at** | Why |
|---|---|
| the dossier contract axis `rings` | its reading on the `stride=0x1` family is a codegen fingerprint of the phase / index arithmetic chain and **does not encode depth**: in the example the depth 2→3 / 5→3 reading went from bare `stride=0x1` → `stride=0x1,slots=253`, exactly the same reading as a segment rewrite on the same tree that **changed no ring at all**; no `slots=3` appears anywhere. For the reading rule see gotcha 11 of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) |
| register tier, arrive count, wait target, release point | under a depth-agnostic spelling **none of these should change**; if one does, the spelling was not depth-agnostic (gotcha 5) |

## Validation form (both kinds of launch-only failure must be targeted explicitly)

A depth change only brings two kinds of failure, neither of them numerical: smem over the limit
(launch refused) and wrong phase (hang / reading the wrong stage, though the numbers may happen to be
right). One KEEP's whole-kernel probe (not included in this release;
`PROBE PASS max_diff=n/a`) targets one kind per step, in three steps:

1. **A minimal shape where not a single chunk wraps** (N=128 in the example): an abort here can only be
   the smem limit;
2. **A shape whose chunk count is not a multiple of the odd depth** (N=1024 = 16 chunks in the example;
   16 is not a multiple of 3): the depth-3 rings and the depth-2 rings wrap on **different chunks**, so the
   `(i // STAGE) % 2` phases of the rings **fall out of lockstep** — an all-even depth set (2 / 2 / 2) or
   the 2-and-5 combination never exercised this within 16 chunks;
3. The production shape, the same compiled artifact run twice (the phase state is reset across launches).

## Gotchas

1. **Rings that are waited on together must be deepened together.** If a role waits on the empty of
   **two** rings before starting a chunk, deepening only one leaves the other's "release of chunk j−2"
   edge binding as before, and the deepened one wastes smem. That is why the example's two output rings
   went 2 → 3 together.
2. **The barrier arena stays out of the smem bill only if it lands in alignment padding.** When
   `MemRange[Int64, STAGE]` sits before a 1024 B-aligned face, a few dozen barriers (42 × 8 B in the
   example) sit entirely inside the padding opened by `Align`, and a depth change does not change the
   total bytes; move the arena after the faces and the bill must be recomputed.
3. **Being at the ceiling is fine; one byte over is a launch failure.** Dynamic smem may be exactly
   == `shared_mem_config_size` (200,704 B in the example); the margin left for the enabler (making
   another ring shallower) must be computed to the byte.
4. **Non-power-of-2 depths are not a dead end, but the phase must be exercised separately.**
   `i % 3` / `(i // 3) % 2` give an integer sequence from the compiler, and correctness is fine; the
   problem is the **test shape** — running only one chunk, or a chunk count that is a common multiple of
   all depths, keeps the phases in lockstep forever, and a wrong stage stride can stay hidden (step 2 of
   the validation form).
5. **"Nothing but the depth changed" is a checkable assertion, not a feeling.** Under a depth-agnostic
   spelling: the register tier is unchanged, every arrive count is unchanged, the wait targets / release
   points / chains / phase contracts are unchanged, and every read view of a face carries the trailing
   stage mode. Any change in the diff beyond these means the spelling hard-codes the depth somewhere
   (typical: some read view used `flat_divide` to cut out a single-stage face, see gotchas 1 / 13 of
   [`tri_inverse_block_doubling`](tri_inverse_block_doubling.md)).
6. **A reading that was not load-bearing (recorded so nobody has to guess it again).** Structural
   conditions: four warpgroups (one copy producer, one per-chunk serial middle role writing two output
   rings, two consumer warpgroups reading them); the middle role waits on the empty of both output rings
   before each chunk, and among the candidates these two rings were exactly the ones that "most directly
   limit the middle role's lead". Deepening 2 → 3 (small-table rings 5 → 3 to balance the bill): gate
   window 0.435 → 0.4379 ms (**+0.67%, worthless**). Under this structure ring depth is not a lever: how
   far the middle role can lead is set by its own segment length, and one more stage would not fill up.
   First prove with timeline / stall readings that the full wait is the binding edge, then spend these 16 KB.

## Sources

- **PTX ISA**: *Parallel Synchronization and Communication Instructions* — `mbarrier` phase
  semantics (the phase flips when the arrive count is reached; `try_wait` queries by parity)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
- **CuTe DSL public API symbols**: `cute.struct.MemRange` / `cute.struct.Align` / `MemRange.get_tensor(layout, swizzle=, dtype=)`,
  `cute.tile_to_shape`, `cute.arch.mbarrier_try_wait` / `mbarrier_arrive`, `cutlass.range`.
- **KEEP evaluation** (NCU dossier census + gate window; the eval job number was not entered in the status
  record): `mbar_objs` and dynamic smem change with depth (46 → 42 barriers; smem exactly ==
  `shared_mem_config_size`), the `rings` axis reading is the same as with no ring change, value +0.67%;
  the whole-kernel probe gives
  `PROBE PASS max_diff=n/a` on the no-wrap shape, the 16-chunk shape and the production shape ×2.
- Related entries: [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (gotcha 11: reading the `rings` axis),
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md), [`gather_producer_lead_stack`](gather_producer_lead_stack.md)
  (on one deep ring, "fill a stage, wait only on that stage's empty" — a timing lever other than depth),
  [`handoff_sync_choice`](handoff_sync_choice.md).
