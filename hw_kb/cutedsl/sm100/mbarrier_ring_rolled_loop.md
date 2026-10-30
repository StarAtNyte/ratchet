# mbarrier_ring_rolled_loop (the **rolled main loop** of an S-slot full/empty mbarrier ring: stage index and phase bit are loop-carried run-time values, barrier address `base + stage`, one loop body per role; `cutlass.range(n)` without `unroll=1` gets partially unrolled)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status: ✅ Verified (only the numerics and the static PTX / SASS shape were verified; no time gain is certified)** -- eight independent GPU probes,
> covering the pipeline-object form (single CTA, and on a CTA pair with producer / MMA split across two warps), the hand-written counter form, the mask form, the hand-written CTA-pair form,
> three advance spellings when S is not a power of 2 (gotcha 16), and the effect of carrying every issue operand on the "wait → issue" dependence chain (gotcha 17); verdicts and readings are in Sources.
> **The same mechanism's readings on seven whole-kernel trees are both positive and negative, mostly flat** (see the table at the end of "When needed"; the whole-kernel readings of gotchas 16 and 17 are in those gotchas); do a same-job A/B on your own tree before switching spellings.
>
> Neighbouring entries: the opposite of this entry -- unrolling the MMA warp's K loop by the stage count, with the stage number a compile-time constant -- is
> [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md); the two are the two ends of the same axis.
> How to choose the stage count see [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md); how to encode the wait itself
> (suspend / spin) see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md); which handle commit / release lands on see
> [`pipeline_umma_async_handle_commit`](pipeline_umma_async_handle_commit.md).
> A rolled main loop measured a few cycles longer on the wrap segment than an unrolled one, see [`umma_mainloop_wrap_segment_issue_floor`](umma_mainloop_wrap_segment_issue_floor.md).

## Purpose

An S-slot operand ring is, in hardware, two sets of smem mbarriers: `full[s]` (the TMA bytes have arrived → the consumer may read) and `empty[s]`
(the consumer has finished reading → the producer may overwrite), usually placed in the same `Int64` array (the first S are full, the next S empty, matching
the layout of `PipelineTmaUmma(barrier_storage=…)`). Which barrier the i-th k-block uses and which parity it waits on is a function of i:

```
stage  = i mod S              (when S is a power of 2 this can also be written i & (S-1))
parity = (i div S) mod 2      (inverted again when the producer waits on empty)
```

**The rolled spelling** keeps this function at run time: each role's steady state has only one loop, `stage` / `phase` are loop-carried
`Int32` (or computed from the loop counter), and the barrier address is `base + stage` on every beat. In PTX / SASS every wait site, every TMA and every
commit appears only once, regardless of the trip count.

**The unrolled spelling** (stage became a constant at trace time or compile time): every copy gets its own constant stage, an immediate barrier address, and its own
wait / arrive / copy. Two routes to it --

- written by you: a warm-up section with `cutlass.range_constexpr(S)`, an MMA chain unrolled by S (the latter is the deliberate trade-off of
  [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md));
- done by the compiler: `cutlass.range(n)` without an `unroll` argument carries no unroll annotation on the loop, and the back end may unroll it on its own.
  What the "default" control variants in the probes read is **partial unrolling by a fixed factor** (e.g. 4 times rolled's TMA count,
  2 times its MMA count), with the same counts at two different trip counts -- not "one copy per k-block", but not one copy either.

## When needed (trigger)

- The main loop is written `for k in cutlass.range(n):` (without `unroll=1`), and `n` is a compile-time constant (a Python int, or from a static shape
  `cute.size(...)`) -- the most common case of partial unrolling; a run-time trip count does not guarantee no unrolling either.
- In the compiled output, the counts of one role's `mbarrier.try_wait` (PTX) / `SYNCS.PHASECHK` (SASS), `cp.async.bulk.tensor` /
  `UTMALDG`, `tcgen05.mma` are an integer multiple of "the amount for one k-block" rather than exactly one copy.
- Before the loop there is a "first fill all S stages" warm-up with `range_constexpr(S)`, followed by a steady-state loop starting from S:
  outside the loop there are S copies of `SYNCS.ARRIVE.TRANS64` with constant addresses and a multiplied number of `UTMALDG`.
- The dossier census's `mbar_tiers` are all CONST, with far more objects than `2·S + a few`; usable as a clue, but **not as a criterion** (gotcha 11).

**These triggers do not guarantee a gain.** Same-job readings of the same mechanism on seven whole-kernel trees (gate is the device window; one tree per row; the whole-kernel readings of gotcha 17 are in gotcha 17):

| Shape of the tree | Change | Reading |
|---|---|---|
| tree 1: compile-time k-block count, several MMAs per beat, whole kernel about 9 µs | `unroll=1` added on both the producer and MMA sides (two independent landings) | gate 0.0087 → 0.009 ms (+3.45%, slower), identical both times; that eval's same-job ratio 1.0151 → 0.9895; going back to the default reads 0.009 → 0.0088 ms. **Slower** |
| tree 1 (same as above) | sweeping `unroll` ∈ {default, 4, 2, 1} **only** on the producer side, MMA side kept at default | gate 0.0087–0.0088 ms unmoved; NCU kernel time 13596 / 13680 / 13420 / 13188 ns, lowest with `unroll=1` |
| tree 2: same kind, pair UMMA | `unroll=1` on both sides | gate 0.0098 → 0.0098 ms; same-job ratio 0.9804 → 0.9845. **Flat** |
| tree 3: compile-time trip count, unrolled warm-up section | warm-up changed from `range_constexpr(S)` to an `unroll=1` loop, steady state `i % S` (compile-time trip count) → mask form + run-time trip count (with other changes in the same package) | gate 0.0249 → 0.025 ms (+0.40%); the whole chain of changes was rolled back as "same-job ratio 1.0318 → 1.0278, no gain within 1.5%". **Flat** |
| tree 4: ring shape changed in the same package when switching the hand-written ring to rolled | `PipelineTmaUmma` → hand-written rolled ring, in the same package as one ring-shape change | gate 0.0369 → 0.0361 ms (−2.17%). **A same-package reading that cannot be split**; it cannot be credited to rolled |
| tree 5: pair UMMA, several workloads | MMA side unrolled by S (stage a compile-time constant) vs run-time stage, same-job A/B | unrolled / run-time = 1.1647 (one workload), 1.0012 / 0.9983 / 0.9993 (the other three); geometric mean 1.0385. **Unrolled is 16% slower on one workload, flat on the rest** |
| tree 6: hand-written ring, warm-up and wind-down unrolled slot by slot by S (`range_constexpr(S)`), several workloads | both sides switched to a `PipelineTmaUmma` run-time ring (warm-up and main loop both `unroll=1` run-time loops, `tail()` replacing the S wind-down waits), same-job A/B | run-time ring / unrolled, geometric mean over 4 workloads: back to back +0.22% (slower direction), with an L2 flush before each launch +0.04%; individual workloads −0.25% … +0.70%; outputs bit-identical. The eval gate that landed afterwards 0.0151 → 0.0152 ms, same-job ratio 1.0569 → 1.0478 (within 1.5%), rolled back with the patch it belonged to. **Flat** |
| an earlier version of tree 6 (the same hand-written ring; S not a power of 2 on the priced workload) | only `k % S` / `k // S` replaced by (B)'s carried registers: producer only / MMA only / both sides | same-job pricing 0.00% / +0.27% / +0.22% (positive = faster; noise of that pricing 1.06%). **Flat** |

Also: on the tree recorded in [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md), fully unrolling the whole ring on the MMA side reads
0.0095 ms and the run-time stage reads 0.0091 ms (full unrolling pushed per-thread registers from 25 to 178).

How to read it: the cost of rolled falls on **the MMA issue side, which must issue several MMAs back to back per beat** (a few more index / phase computations per beat); the producer side rolled
never lost in the table above. The cost of unrolling is code size and registers (especially when fully unrolled). Which end wins is decided by the tree: **sweep by role separately** (producer and
MMA each rolled / partially unrolled / fully unrolled), timing each level same-job.

## Minimal runnable example

### (A) The pipeline-object form: the only thing to do is keep the outer loop rolled

`PipelineState` itself is three `Int32` `(index, phase, count)`, and `advance()` generates "`index += 1`; back to zero at S, flipping phase".
Whether it can stay in registers depends only on whether the loop outside is unrolled:

```python
import cutlass, cutlass.cute as cute, cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05

# NSTAGE: ring depth, chosen per pipeline_tma_umma_stage_count; barrier storage = MemRange[Int64, 2 * NSTAGE]
ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
    num_stages=NSTAGE,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, N_CONS),  # single CTA: 1
    tx_count=STAGE_TX_BYTES,                    # the byte sum of all operand faces of one stage
    barrier_storage=st.ab_mbar.data_ptr(),      # take the pointer at the kernel's top level, see gotcha 9
    cta_layout_vmnk=cta_layout_vmnk,            # single CTA: None
    defer_sync=True,
).make_participants()

if warp_idx == TMA_WARP:
    for k in cutlass.range(k_blocks, unroll=1):              # ← unroll=1: keep it rolled
        h = ab_prod.acquire_and_advance()                    # wait on empty[h.index] (run-time index) + expect_tx
        cute.copy(tma_a, tAgA[(None, h.count)], tAsA[(None, h.index)], tma_bar_ptr=h.barrier)
        cute.copy(tma_b, tBgB[(None, h.count)], tBsB[(None, h.index)], tma_bar_ptr=h.barrier)
    ab_prod.tail()                                           # wait the leading stages back before exiting (gotcha 7)

if warp_idx == MMA_WARP:
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
    for k in cutlass.range(k_blocks, unroll=1):              # ← rolled as well
        h = ab_cons.wait_and_advance()                       # wait on full[h.index]
        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):   # the few MMAs within a stage: compile-time unrolling is exactly what is wanted
            kc = (None, None, kb, h.index)                   # slice the stage mode with a run-time index -> descriptor base added at run time
            cute.gemm(tiled_mma, tAcc, tCrA[kc], tCrB[kc], tAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        h.release()                                          # lands on the handle, not ab_cons.release()
```

**The CTA-pair version of (A)** (probe 6 in Sources): both loop bodies unchanged word for word, with only three additions -- `create(..., cta_layout_vmnk=…)` (pass `None` for a single CTA),
`tx_count` multiplied by the pair size (expect_tx issued only by the leader, decided inside the object), and the MMA loop wrapped in `if is_leader_cta:` (the producer loop runs on both CTAs,
the same number of times). After barrier init, first `cluster_arrive_relaxed()` + `cluster_wait()`, then enter any role (gotcha 12). When producer and MMA are split across two warps,
and the MMA warp itself also takes part in the accumulator readback (it is one of the consumers of `PipelineUmmaAsync`), its `acc_producer.tail()` goes after its own
`acc_full.release()` -- the order probe 6 uses; the other way round waits on its own release not yet issued, `[INFERRED]` self-deadlock, untested.

### (B) The hand-written form: stage / phase as your own loop-carried registers

```python
Int32, Int64 = cutlass.Int32, cutlass.Int64

@cute.struct
class Storage:                                     # defined outside the @cute.kernel
    mbar: cute.struct.MemRange[Int64, 2 * NSTAGE]  # full[0..S-1] then empty[0..S-1]; an Int64 field is naturally 8 B aligned

st = cutlass.utils.SmemAllocator().allocate(Storage)
full0 = st.mbar.data_ptr()                         # Int64 element pointer: +1 = the next barrier (8 B)
empty0 = full0 + NSTAGE

if warp_idx == INIT_WARP:                          # init runs only once; unrolling it in the prologue is right (gotcha 12)
    with cute.arch.elect_one():
        for s in cutlass.range_constexpr(NSTAGE):
            cute.arch.mbarrier_init(full0 + s, 1)  # 1 arrive (expect_tx) + the TMA's tx bytes
            cute.arch.mbarrier_init(empty0 + s, 1) # 1 release (an elected commit or one lane's arrive)
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()                           # in cluster / pair form replace with cluster_arrive_relaxed + cluster_wait

if warp_idx == TMA_WARP:
    stage = Int32(0)                               # typed as Int32 outside the loop (gotcha 5)
    phase = Int32(1)                               # the producer waits on empty from parity 1: returns immediately on a fresh barrier (gotcha 4)
    for k in cutlass.range(k_blocks, unroll=1):
        cute.arch.mbarrier_wait(empty0 + stage, phase)          # the only wait site, address computed at run time
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full0 + stage, STAGE_TX_BYTES)
        cute.copy(tma_a, tAgA[(None, k)], tAsA[(None, stage)], tma_bar_ptr=full0 + stage)
        cute.copy(tma_b, tBgB[(None, k)], tBsB[(None, stage)], tma_bar_ptr=full0 + stage)
        stage = stage + 1
        if stage == NSTAGE:                        # dynamic if: give Int32 in both branches
            stage = Int32(0)
            phase = phase ^ 1
    for _ in cutlass.range(NSTAGE, unroll=1):      # drain: wait the last NSTAGE empties back (gotcha 7)
        cute.arch.mbarrier_wait(empty0 + stage, phase)
        stage = stage + 1
        if stage == NSTAGE:
            stage = Int32(0)
            phase = phase ^ 1

if warp_idx == MMA_WARP:
    stage = Int32(0)
    phase = Int32(0)                               # the consumer waits on full from parity 0
    for k in cutlass.range(k_blocks, unroll=1):
        cute.arch.mbarrier_wait(full0 + stage, phase)
        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cute.gemm(tiled_mma, tAcc, tCrA[(None, None, kb, stage)], tCrB[(None, None, kb, stage)], tAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(empty0 + stage)         # MMA has finished reading this stage -> give it back to the producer
        # an ordinary load consumer: cute.arch.sync_warp(); then one lane cute.arch.mbarrier_arrive(empty0 + stage)
        stage = stage + 1
        if stage == NSTAGE:
            stage = Int32(0)
            phase = phase ^ 1
```

Equivalent spelling: the producer also starts from `phase = Int32(0)` and passes `phase ^ 1` when waiting on `empty` (the probe certified both).

**Mask variant** (when NSTAGE is a power of 2): maintain no carried variables; compute from the loop counter `s = k & (NSTAGE - 1)`,
`ph = (k >> LOG2_NSTAGE) & 1` (the producer additionally `^ 1`). The probe read the same SASS instruction count as (B) and the same in-ring address shape
(`ULEA URx, <stage>, <base>, 0x3`, i.e. base + stage·8). When NSTAGE is not a power of 2 use (B); `k % S` / `k // S` have the same semantics,
and division by a constant S is turned into multiply-high + shift before PTX (there is no `div.` / `rem.` in PTX); what each beat gains is this sequence, see gotcha 16.

### (C) Differences of the CTA-pair (`cta_group::2`) form from a single CTA

Different only where the ring is concerned (for partitioning, TMEM and readback see [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md)):

```python
# the producers of both CTAs run; expect_tx is issued only by the leader, once, with the bytes summed over both CTAs
if is_leader:
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(full0 + stage, STAGE_TX_BYTES * 2)
cute.copy(tma_a, tAgA[(None, k)], tAsA[(None, stage)], tma_bar_ptr=full0 + stage)   # both sides pass their own CTA's full0 + stage

# MMA runs only on the leader; the release must be multicast to the empty[stage] of both CTAs (each side's producer waits on the one in its own CTA)
commit_mask = mask_self | mask_peer               # cpasync.create_tma_multicast_mask(..., mcast_mode=0), one call each
with cute.arch.elect_one():
    tcgen05.commit(empty0 + stage, commit_mask, tcgen05.CtaGroup.TWO)
```

### Acceptance: how to confirm it really is rolled (without relying on the dossier axis)

Compile the same kernel once each with **two different trip counts** (for a compile-time trip count change the constant; for a run-time one change the shape and recompile), dump, and count:

```bash
export CUTE_DSL_KEEP_PTX=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir> CUTE_DSL_CACHE_DIR=<new directory>   # a new cache directory keeps a JIT hit from skipping the dump
nvdisasm -c <dir>/*.cubin > k.sass             # without nvdisasm, cuobjdump -sass
```

| Reading | rolled | unrolled |
|---|---|---|
| PTX `mbarrier.try_wait`, SASS `SYNCS.PHASECHK` counts | equal at the two trip counts | a multiple of rolled (with partial unrolling they may also be equal at the two trip counts, so look at the next two rows too) |
| PTX `cp.async.bulk.tensor` / SASS `UTMALDG`, `tcgen05.mma` / `UTC*MMA` counts | exactly the amount for one k-block | an integer multiple of the amount for one k-block |
| `SYNCS.ARRIVE.TRANS64` outside the loop (straight-line stretch) | 0 (only the prologue arms of other barriers remain) | = S when the warm-up is unrolled |
| PTX `.pragma "nounroll"` | one per `unroll=1` loop | 0 |
| recompiled with a different S (trip count unchanged) | the number of `mbarrier.try_wait` / `expect_tx` sites in the loop unchanged; `mbarrier.init` +2 per added stage (one-time inits in the prologue, gotcha 12) | the sites in the loop grow with S |

Note that one `mbarrier_wait` lowers to **several** `SYNCS.PHASECHK` (a fast try + a retry loop), so the total PHASECHK count is not the number of wait sites;
the criterion is "unchanged with the trip count" plus "TMA / MMA counts = the amount for one k-block", not some absolute count.

## Gotchas

1. **`cutlass.range(n)` without `unroll` ≠ rolled.** Without it the loop carries no unroll annotation and the back end may unroll on its own (the probes read partial
   unrolling by a fixed factor); `unroll=1` attaches `llvm.loop.unroll.disable`, which shows up as `.pragma "nounroll"` in PTX. `unroll_full=True` is the opposite direction.
   `[read from source, not probed]`: `unroll=0` generates `count = 0` without attaching disable; for rolled, write `1`.
   The producer and MMA sides must **each** be written; change only one side and the other is still unrolled code (the per-role sweep above does exactly this).
2. **Python `range(...)` / `cutlass.range_constexpr(...)` unroll at trace time.** The outer stage loop must never be written with them; they are only for
   small loops known at compile time **inside** a stage (the few MMAs within a stage) and for initializing the S barriers in the prologue.
3. **The warm-up section is not needed.** "First fill all S stages with `range_constexpr(S)`, then enter the steady state" generates S copies of expect_tx / copy with constant addresses outside the loop,
   and the steady-state loop is a second copy of the code. The producer waits on `empty` from parity 1, so the first S waits on fresh barriers return immediately -- one loop starting from 0
   is the warm-up and the steady state at the same time.
4. **The initial phase is part of the ring protocol.** The producer waits on `empty` from parity **1** (or `phase = 0` paired with `phase ^ 1`), the consumer waits on `full`
   from parity **0**; the Producer / Consumer of `pipeline.make_pipeline_state` are exactly these two initial values. Both starting from 0: the producer blocks forever
   on the first beat. The phase flip may only hang on the beat where stage returns to 0; flipping on every beat or forgetting to flip either deadlocks or releases early and reads the previous lap's stale data
   (wrong values without a hang, most easily exposed when the trip count is not a multiple of S).
5. **Loop-carried variables must be declared outside the loop as `Int32(...)`, and the wraparound branch must also assign `Int32(0)`.** Written as a Python `0`, `stage + 1` is treated at trace time
   as a Python int (constant-folded), or the types disagree at the merge point of the dynamic `if` and it errors; a variable that first appears inside the loop body is not carried to the next trip.
6. **Barrier pointer arithmetic counts elements.** `MemRange[Int64, …].data_ptr()` is an Int64 element pointer; `+ stage` advances `8·stage` bytes;
   writing `+ 8 * stage` jumps to another barrier (silent misalignment or a hang).
7. **tail / drain cannot be omitted.** The producer leads by at most S beats; after the loop ends, every slot must be waited back once more (`ab_prod.tail()` or S hand-written `empty`
   waits) before exiting. `prod.tail()` advances the state S−1 times at trace time and then does **one** acquire, adding only one wait site, so it works as is in a rolled ring.
   `[INFERRED]` omitting it in cluster / pair form lets the peer's async commit hit the smem of a CTA that has already exited.
8. **Two roles and three index uses must be the same (stage, parity) bijection.** The producer's expect_tx / copy destination / `empty` wait and the consumer's
   `full` wait / smem view / release all come from the same advance rule; the TMA **source** coordinate uses the k-block number (`k` or `h.count`), the smem **destination** and
   the MMA fragments use the stage (`stage` or `h.index`). If any of the three still says a constant `s`, it reads another slot (wrong values, no hang); using `k` on the smem side
   writes some other smem out of bounds when `k ≥ S`.
9. **Inside a dynamic `if` you cannot take the field pointer of a `@cute.struct` object.** Writing
   `st.mbar.data_ptr()` inside a run-time branch like `if warp_idx == …:` fails at trace time with `DSLRuntimeError: The 'if' statement encountered a user-defined Python object …`;
   take the barrier pointer into a local variable at the kernel's top level first (the same gotcha is also in `umma_issue_chain_precomputed_desc`).
10. **Allocate the barrier array with `@cute.struct` + `MemRange`.** In the CuTe DSL version installed in the GPU image,
    `SmemAllocator.allocate_array(…, byte_alignment=…)` fails at trace time with
    `TypeError: SmemAllocator.allocate_array() got an unexpected keyword argument 'byte_alignment'` (read by the probe);
    a `MemRange[Int64, 2 * NSTAGE]` field + `allocate(Storage)` + `.data_ptr()` gets the base, and its 8 B alignment satisfies the mbarrier requirement.
11. **The dossier axes `rings` / `mbar_tiers` can neither read nor prove rolled.** On the toy kernel the barrier base is derived from
    `SR_CgaCtaId` via `SmemAllocator`, so almost all sites read UNRESOLVED; the only RING label of the mask form (`stride=0x1,slots=25`) reads the loop-counter chain,
    the carried-variable form has not a single RING, and the two have identical SASS. On a whole kernel `rings` sprouts one `wrap=sel` only when both sides are rolled, and stays `[]`
    when only the producer is rolled; when `PipelineTmaUmma` was switched to a hand-written rolled ring, `rings` went from one entry to `[]` instead. Accept only by the two-trip-count counts above;
    for the interpretation of that axis as a whole see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11.
12. **Unrolling the init in the prologue by S is right.** Init sites are supposed to have constant addresses; what must become a ring are the wait / arrive / copy sites **in the loop**.
    So the number of `mbarrier.init` sites grows linearly with S (one pair per stage; probe 6 at three values of S: 9 / 11 / 17, while the wait sites in the loop stay the same), and it cannot be used as an unrolling criterion.
    In cluster / pair form the init must be visible to the cluster before any remote arrive: `mbarrier_init` → `mbarrier_init_fence()` →
    `cluster_arrive_relaxed()` + `cluster_wait()` (all threads, all CTAs), and only then issue the first TMA or commit.
13. **A trip count not divisible by S needs no tail loop.** A run-time ring handles the remainder naturally (the probes covered non-divisible trip counts); only a spelling statically unrolled by S needs
    two stretches of code, "whole laps + a tail starting from stage 0".
14. **The two rules of the pair form: miss one and it hangs.** expect_tx is issued only by the leader, wrapped in `elect_one`, with bytes summed over both CTAs (a follower arriving again
    makes this phase complete early; the follower's `full[s]` is never waited on and can just be left alone); the releasing `tcgen05.commit` must carry `mask_self | mask_peer`
    and `CtaGroup.TWO`; without the mask it arrives only at the leader, and the follower's producer never gets released.
15. **Rolled is not automatically faster.** What it changes is code size, the per-beat index / phase arithmetic and where the barrier address comes from; it does not change the ring depth or the
    semantics of any wait. The probes give only structural and numeric facts; for gains see the table at the end of "When needed", both positive and negative.
16. **When S is not a power of 2, there are three spellings of the (stage, phase) advance; all are correct and none is faster.** Computing `i % S` / `(i // S) & 1` from the item number;
    (B)'s carried registers + a dynamic `if stage == S` wraparound; or carried registers + a purely arithmetic advance (only subtraction, shift, and, xor; no comparison, no division):

    ```python
    def ring_step(stage, phase):                   # S is a Python int; stage must be in [0, S-1]
        lt = ((stage - (S - 1)) >> 31) & 1         # 1 when not in the last slot, 0 in the last slot
        return (stage + 1) & (Int32(0) - lt), phase ^ (lt ^ 1)
    ```

    Certification probe (a single-CTA copy ring, one non-power-of-2 S; the consumer's per-item wrap segment = wait returns → next wait starts, timestamped with `%clock64`): all three spellings bit-correct, the same total SASS count;
    wrap segment computed ≈ 124, `if` wraparound ≈ 114, pure arithmetic ≈ 124 cycles/item -- **the `if` wraparound is the shortest**, and pure arithmetic is not faster than computing.
    None of the three has `div.` / `rem.` in PTX (constant division has been eliminated), so "does the PTX have div" cannot tell the spellings apart; what can is the select instructions:
    PTX `selp.` computed 3 / `if` wraparound 6 / pure arithmetic 0, SASS `SEL` + `USEL` likewise 3 / 6 / 0 -- the spelling with the most select instructions is actually the fastest; the select count is not a price tag.
    The pure-arithmetic spelling has two further structural uses: it has no branch, so it can be written in an ordinary Python helper function (a dynamic `if` is preprocessed only inside a `@cute.kernel` / `@cute.jit` body,
    and written into a helper it fails at trace time); producer / consumer / several loop stretches share the same function, and (stage, phase) continues directly across stretches (the probe covered
    a consumer walking the same ring in two stretches, a rate-limited producer + release by `tcgen05.commit`, and two other non-power-of-2 S). The `& 1` after `>> 31` makes `lt` 0 / 1 whether the shift is arithmetic or logical;
    without it, and lowered to an arithmetic shift (the usual case for a signed `Int32`), `lt = −1`, and `0 − lt` is no longer an all-ones mask (derived; the probe ran only the spelling with `& 1`). Whole-kernel reading (a `cta_group::1` UMMA tree, the `% S` / `// S` computation on both the producer and MMA sides replaced together by
    carried registers + the purely arithmetic advance, in the same package as one change to the MMA-side commit spelling): gate 0.024 → 0.024 ms, no visible change at two significant digits (resolution about 4%); no separate same-job A/B was done.
17. **Carrying only stage / phase while issue still goes through DSL slices leaves the "wait returns → first issue" dependence chain fully intact.** `cute.copy(…, tAsA[(None, stage)])`,
    `cute.gemm(…, tCrA[(None, None, kb, stage)], …)` and `full0 + stage` compute addresses from stage **after** the wait (something like `ULEA`), and that chain hangs on the operands of the
    issue instructions. To zero it, **every** operand the issue uses must be made a loop-carried `Int32`: the two barrier addresses (+8 bytes), the smem descriptor low word
    (in 16 B units, stage stride / 16; encoding see [`umma_raw_desc_issue`](umma_raw_desc_issue.md)) or the TMA destination smem address (in bytes), the TMA coordinates (in elements);
    advanced **after** this beat's issue, all reset to their base values and the phase flipped when `stage == S`; the wait, `expect_tx`, TMA, UMMA and commit all switched to inline PTX taking `Int32` addresses
    (a slice is a Python object with no IR value and cannot be carried across a `for`). Certified readings (a single-CTA TMA → UMMA ring, one non-power-of-2 S; counting from each `SYNCS.PHASECHK*` along the fall-through path
    to the first `UTCHMMA` / `UTMALDG`, and doing a backward register slice from the issue instruction's operand registers; the number of non-move instructions in the slice = the dependence chain length):

    | Spelling | MMA-side dependence chain | MMA-side raw window count | producer-side dependence chain | producer-side raw window count |
    |---|---|---|---|---|
    | computing `% S` / `// S` from the loop counter + DSL slices | 2 | 16 | 1 | 12 |
    | carrying only stage / phase + DSL slices | 2 | 17 | 1 | 8 |
    | all issue operands carried + inline PTX | **0** | 17 | **0** | 9 |

    The accumulated results of all three spellings are bit-equal to numpy. **The raw window count is not the acceptance quantity**: ptxas fills the basic block after the wait with instructions unrelated to issue -- the next beat's advance, the back-edge comparison, constant `UMOV` --
    and the carried version's window even has 1 more; what is cut is only the two `ULEA` on the issue operands (one on the producer side).
    Flat on the whole kernel (consistent with gotcha 15): the lever pricing before the rewrite was 0.50% slower with it added and 0.58% faster with it removed (three workloads combined; one workload alone 2.12% slower, beyond the noise,
    one 0.75% faster); after landing on the MMA side, gate 0.03245 → 0.0324 ms;
    producer-side same-job A/B 0.03764 → 0.03761 ms (0.9993); warm-up rewritten the same way, same-job A/B 0.03766 → 0.03773 ms (1.0020). Before changing it, first confirm that the few instructions after the wait returns
    really are on the critical path ([`umma_mainloop_wrap_segment_issue_floor`](umma_mainloop_wrap_segment_issue_floor.md)).
    The same idea applied to barrier wait / arrive and data-slot reads/writes (address and parity computed within the previous beat's body and loop-carried) is in [`ring_index_lookahead`](ring_index_lookahead.md); its whole-kernel readings are likewise at the edge of the noise.

## Sources

- **PTX ISA**: mbarrier (`mbarrier.init` / `mbarrier.arrive.expect_tx` / `mbarrier.try_wait.parity`, phase completion and parity semantics):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>;
  `tcgen05.commit`: <https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-commit>;
  `cp.async.bulk.tensor`; `.pragma "nounroll"`: <https://docs.nvidia.com/cuda/parallel-thread-execution/#descriptions-pragma-nounroll>.
- **CuTe DSL public API symbols**: `cutlass.range(..., unroll=, unroll_full=)`, `cutlass.range_constexpr`;
  `cutlass.pipeline.PipelineTmaUmma.create` / `PipelineAsync.create` / `PipelineUmmaAsync.create`, `make_participants`, `acquire_and_advance` /
  `wait_and_advance` / `release` / `commit` / `tail`, the handle's `index` / `count` / `barrier`, `PipelineState`
  (`index` / `phase` / `advance`), `make_pipeline_state`; `cute.struct.MemRange`, `cutlass.utils.SmemAllocator.allocate`;
  `cute.arch.mbarrier_init` / `mbarrier_init_fence` / `mbarrier_wait` / `mbarrier_arrive` / `mbarrier_arrive_and_expect_tx` /
  `elect_one` / `cluster_arrive_relaxed` / `cluster_wait`; `cutlass.cute.nvgpu.tcgen05.commit` (`mask` / `cta_group`);
  `cpasync.create_tma_multicast_mask` / `tma_partition`.
- **GPU certification probes** (eight, all verifying only numerics and static shape; the 7th also reports wrap-cycle info lines; the probes are not included in this release):
  1. Three variants pipeline-object rolled / hand-written rolled / default `cutlass.range`, two compile-time trip counts,
     an artificial per-beat delay on the consumer forcing the producer to really go around and wait on `empty`. `PROBE PASS max_diff=0.0` (5/5):
     values and stage sequences of the three variants bit-exact; PTX `try_wait` / SASS `PHASECHK` the same at both trip counts -- pipeline rolled 3 / 6, hand-written rolled 2 / 4
     (SASS 176 lines); the default control 10 / 20 (SASS 287 lines), also the same at both trip counts (partial unrolling).
  2. The pipeline-object / hand-written forms, run-time K, five trip counts (from less than one lap to more than four, including non-divisible),
     each launched twice. `PROBE PASS max_diff=0.0` (3/3). PTX: pipeline `tma=2 mma=4 try_wait=6 expect_tx=2 commit=2 nounroll=2`;
     hand-written `tma=2 mma=4 try_wait=6 expect_tx=1 commit=2 nounroll=3`; control (compile-time trip count, without `unroll=1`)
     `tma=8 mma=8 try_wait=10 expect_tx=5 commit=3 nounroll=0`.
  3. Mask form / carried-variable form / a "`range_constexpr(S)` warm-up + steady state" control,
     single-CTA UMMA, several sets of trip counts + re-entry. `PROBE PASS max_diff=0.0` + structure line `PROBE PASS max_diff=n/a`:
     SASS instruction counts 360 / 360 / 440; total `SYNCS.ARRIVE.TRANS64` 1 / 1 / S+1, of which in the straight-line stretch 0 / 0 / S; `UTMALDG` of both rolled forms equals the amount for one k-block, the warm-up control has S k-blocks' worth more;
     `TRYWAIT` sites 9 in all three; PTX `try_wait` 6, `nounroll` 2, the same in all three. The previous round (same probe) read gotcha 11's RING label readings.
  4. The hand-written pair form (B)+(C), two trip counts (neither a multiple of S, one of them launched three times in a row).
     `PROBE PASS max_diff=0.0` (round 2, 2/2): the counts the same at both trip counts -- `PHASECHK` 23 (of which `TRYWAIT` 16),
     `UTMALDG` 4, `ARRIVE` 4, `UTC*MMA` = the count for one k-block, SASS 472 instructions; the control `PipelineTmaUmma` + default `cutlass.range`:
     `PHASECHK` 32 (`TRYWAIT` 22), `UTMALDG` 16, `ARRIVE` 8, `UTC*MMA` as above, SASS 616 instructions, numerics likewise exact.
     Round 1 FAILed for using the absolute PHASECHK count as the criterion (last paragraph of "Acceptance").
  5. Form (B), no MMA, single CTA with two warps,
     S ∈ {4, 3, 5, 6} × {lane-arrive release, elected `tcgen05.commit` release} × {consumer slowed, producer slowed}, 16 k-blocks (not divisible by 3 / 5 / 6),
     plus a static PTX census. `PROBE PASS max_diff=0.0` (5/5): the four numeric levels bit-equal; the run-time ring with S=3 → 2
     `mbarrier.try_wait`, S=6 → 2, the control statically unrolled by S=4 → 8.
  6. The CTA-pair version of form (A) (the paragraph "The CTA-pair version of (A)"), the producer warp running on both CTAs,
     the MMA warp only on the leader, each one `unroll=1` loop, K a run-time extent; S ∈ {3, 4, 7} each run for several laps (3 and 7 not dividing the trip count), plus one level with S = 7 and
     a trip count of less than one lap; each level launched 3 times in a row from the same compiled entry. `PROBE PASS max_diff=0.0` (round 2, 3/3): C bit-equal to the fp32 reference
     at all four levels; PTX `mbarrier.try_wait` 7, `expect_tx` 2, the same at all four levels, `mbarrier.init` 9 / 11 / 17. Round 1 reported `AttributeError` at all four levels because the TMEM holder was written with `.ptr`
     (the same gotcha see [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 10).
  7. Gotcha 16: a copy ring in form (B) (TMA producer + copy consumer, no MMA), a non-power-of-2 S,
     40 items, the consumer walking the same ring in two stretches with the state continuing across stretches; three advance spellings × one S, with the pure-arithmetic spelling additionally run with a rate-limited consumer, a rate-limited producer + release by `tcgen05.commit`,
     and another non-power-of-2 S; plus a PTX / SASS census. `PROBE PASS max_diff=0.0` (round 2, 7/7): the six numeric levels bit-exact;
     the census shows no `div.` / `rem.` in any of the three, `selp.` 3 / 6 / 0, SASS `SEL` + `USEL` 3 / 6 / 0, total count 192 for all; wrap readings (info lines, not criteria)
     123.8 / 113.7 / 124.5 cycles/item (cold). Round 1 FAILed because the census contract assumed "the computing spelling must have `div` / `rem` in PTX"; after switching to the select instructions as the fingerprint, round 2 passed.
  8. Gotcha 17: a single-CTA TMA → UMMA ring (`cta_group::1`, K-major f16, 128 B swizzle), one non-power-of-2 S;
     the three spellings (computing + DSL slices / carrying only stage·phase + DSL slices / all issue operands carried + inline PTX) each compiled once and run once;
     `PROBE PASS max_diff=0.0` (round 2, 5/5): the accumulated results of all three levels bit-equal to numpy; MMA-side dependence chain 2 / 2 / 0, window 16 / 17 / 17;
     producer-side dependence chain 1 / 1 / 0, window 12 / 8 / 9. Round 1 4/5, the 4th contract FAILed.
  The `TypeError` of gotcha 10 was read in round 1 of probe 1; the error of gotcha 9 is in the certification record of `umma_issue_chain_precomputed_desc`.
- **Whole-kernel readings** (the table at the end of "When needed"): all are each tree's own eval / same-job A/B timings, not independent certifications; they serve only as warnings in both directions, not as pricing.
