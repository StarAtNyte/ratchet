# cp_async_mbarrier_ring (signal `cp.async` completion straight to an mbarrier: non-blocking arrive + ring-depth lookahead)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry follows [`cp_async`](cp_async.md): that one covers **how to issue** LDGSTS (atom / partitioning
> / runtime row indices / 16 B alignment); this one only covers **how to announce that it has landed**. The
> two cover the two ends of the same batch of instructions; the `commit_group` / `wait_group` given in
> `cp_async.md` §3 is one way of announcing, this entry gives the other — and their consequences in a
> pipeline differ by an order of magnitude.

## Purpose

`cp.async` (LDGSTS) has **two mutually independent completion mechanisms**:

| Mechanism | Instruction | Who waits | Does the issuing thread block? |
|---|---|---|---|
| group completion | `cp.async.commit_group` / `cp.async.wait_group N` | the issuing thread itself | **yes** (`wait_group` is a blocking instruction) |
| mbarrier completion | `cp.async.mbarrier.arrive[.noinc].shared::cta.b64 [mbar]` | any thread waiting on this mbarrier | **no** |

The semantics of the second instruction: "**once all `cp.async` previously issued by this thread have
landed**, do one arrive-on on `mbar` on this thread's behalf". It moves "waiting for landing" from the
**issuing thread** to **the hardware + the consuming threads**; the issuing thread continues with the next
instruction as soon as it executes this one. So the producer can issue the stages of the whole ring depth in
one go, and the gmem round trips of N stages overlap each other; ring depth costs only smem, not time.

Writing the same thing with group completion necessarily degenerates into:

```
issue stage -> commit_group -> wait_group(0) -> whole-group rendezvous -> arrive
```

`wait_group` blocks ⇒ **only one stage is in flight at any moment** ⇒ every stage exposes the whole gmem round
trip on the critical path. Lookahead depth drops to zero, and the ring is left with only its smem cost.

## When needed (trigger)

- **All three NCU readings hold at once** (this is the diagnostic signature of this entry, not a guess):
  1. **absolute tensor busy time** such as `sm__pipe_tensor_cycles_active` is roughly equal to the
     reference (± a few %) — the MMA amount and shape are right; it is not "computing too much";
  2. but `tensor_pct` is only about half of the reference — the **idle gaps** of the tensor pipe are the
     difference;
  3. `stall_long_sb` (long scoreboard) is significantly above the reference, while `dram_bytes_*` is within
     band and the absolute value of `stall_barrier` is negligible.
  ⇒ "can't feed it" rather than "computing too much", and not a sync-shape problem (the differential for the
  sync shape is in [`handoff_sync_choice`](handoff_sync_choice.md): there the dominant stall is
  `stall_barrier`; the differential for the issue-organisation shape is in
  [`gather_index_prefetch`](gather_index_prefetch.md): there stall_long_sb is around ten times higher and in
  SASS fetches interleave one by one with address arithmetic).
- **Source-shape symptom** (visible without a profiler): in the producer's main loop a
  `cp_async_wait_group(0)` or any whole-warpgroup rendezvous (`NamedBarrier.arrive_and_wait()`) sits between
  "issue this stage" and "announce this stage is full".
- **Construction symptom**: you want to manage a `cp.async` ring with `PipelineAsync`, find that its
  `producer_commit` is a plain `mbarrier.arrive` that flips the barrier **at issue time**, and so you drain
  first and then commit. This means the wrong pipeline class was chosen (see gotcha 3).
- **Reverse exclusion**: when the transfer is a regular block (contiguous tile / fixed stride / im2col
  sliding window), don't use `cp.async`; use TMA — TMA has a hardware tx-count path, recipes in
  [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) / [`pipeline_runtime_subring`](pipeline_runtime_subring.md).
  **The tx-count mechanism of those two is completely unusable for `cp.async`**: `cp.async` produces no
  transaction count; it can only use this entry's arrive-on-completion.

## Minimal runnable example

Two spellings, the same instruction. The geometry is just a size label: a ring of `NSTAGE` stages, each
stage `ROWS` rows × `RELEM` elements, `NPROD` producer threads / `NCONS` consumer threads. The issue side
(copy atom / TiledCopy / partitioning / runtime row indices) follows [`cp_async`](cp_async.md) throughout and
is not repeated here.

### A) Packaged spelling: `PipelineCpAsync` (preferred)

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

# --- Shared prologue: `.create` ends with a whole-CTA rendezvous, so it must come before the role branch and
# --- be reached by every thread (and before the setmaxnreg dec, see warp_specialization gotcha 11).
kv_pipe = pipeline.PipelineCpAsync.create(
    barrier_storage=storage.bars.data_ptr(),      # MemRange[Int64, 2*NSTAGE]
    num_stages=NSTAGE,
    # ↓↓↓ the easiest line of this entry to get wrong: producer group size = **the number of threads that
    #     issue cp.async and commit**, not 1. `.noinc` does not add to the count at issue time, the barrier's
    #     initial arrive count is this number, and each thread's copies landing deducts 1 (see gotcha 1).
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NPROD),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NCONS),
)

if tidx < NPROD:                                   # producer warpgroup
    p_st = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, NSTAGE)
    for t in cutlass.range(NTILE):
        kv_pipe.producer_acquire(p_st)             # wait for empty[stage]
        issue_stage(..., p_st.index)               # see cp_async.md: all LDGSTS of this stage
        kv_pipe.producer_commit(p_st)              # -> cp.async.mbarrier.arrive.noinc
        p_st.advance()                             # non-blocking: issue the next stage right away
else:                                              # consumer warpgroup
    c_st = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, NSTAGE)
    for t in cutlass.range(NTILE):
        kv_pipe.consumer_wait(c_st)                # wait **stage by stage**, not for all at once
        consume_stage(..., c_st.index)
        kv_pipe.consumer_release(c_st)             # release stage by stage, not all at once at the end
        c_st.advance()
```

The producer loop has **no** `cp_async_commit_group` / `cp_async_wait_group` / rendezvous of any kind. That is
the whole difference.

### B) Raw spelling: direct `cute.arch` calls (when building your own barriers)

When you need to cut barrier sub-ranges yourself, or a pipeline class doesn't fit, the mechanism is these
three lines:

```python
# Init: the arrive count of full = number of producer threads that will arrive
if warp_idx == 0:
    for s in cutlass.range_constexpr(NSTAGE):
        cute.arch.mbarrier_init(bars + s,           NPROD)   # full[s]
        cute.arch.mbarrier_init(bars + NSTAGE + s,  NCONS)   # empty[s]
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()

# producer: wait for empty -> issue -> asynchronous arrive -> next round immediately
cute.arch.mbarrier_wait(bars + NSTAGE + p_st.index, p_st.phase)
issue_stage(..., p_st.index)
cute.arch.cp_async_mbarrier_arrive_noinc(bars + p_st.index)   # <<< this entry's recipe
p_st.advance()

# consumer: wait for full -> read -> plain arrive to release
cute.arch.mbarrier_wait(bars + c_st.index, c_st.phase)
consume_stage(..., c_st.index)
cute.arch.mbarrier_arrive(bars + NSTAGE + c_st.index)
c_st.advance()
```

Get phases from `pipeline.make_pipeline_state(...)` (producer starts at `phase=1`, consumer at `phase=0`,
`advance()` flips automatically at the end); don't hand-roll them — a hand-rolled phase error **shows up as a
deadlock, not a wrong result**. Pointer arithmetic `bars + idx` accepts a **runtime** `idx`
(`cute.Pointer.__add__`), which is the same opening [`pipeline_runtime_subring`](pipeline_runtime_subring.md)
uses to cut sub-rings.

## Gotchas

1. **`.noinc` means "no pre-increment", so the arrive count must be budgeted in advance.** PTX's
   `cp.async.mbarrier.arrive` by default **increments** the mbarrier's pending count by 1 **at execution
   time** and decrements it by 1 when the copy lands; with `.noinc` it does **not** increment and only
   decrements on landing — so that decrement must be absorbed up front by the initial value of
   `mbarrier_init`. **The DSL exposes only `.noinc`** (`cute.arch.cp_async_mbarrier_arrive_noinc`;
   `PipelineCpAsync` uses it internally too), so there is only one rule:

   > the arrive count of `full[s]` == the **number of producer threads** that arrive on `full[s]`.

   In the packaged spelling this number is the `N` of `CooperativeGroup(Agent.Thread, N)`. Copying the TMA
   pipeline's `CooperativeGroup(pipeline.Agent.Thread)` (default size=1, correct for TMA because it is
   single-thread issue) gives count=1: **the barrier flips as soon as the first thread's copies land**, and the
   consumer reads smem that the other N-1 threads have not finished moving. No error, no deadlock — it is
   **silently partially correct**: wrong in per-thread blocks, as hard to find as the "mostly right" shape of
   [`cp_async`](cp_async.md) gotcha 5. Conversely, writing the count too large (e.g. by warp count instead of
   thread count) is a **deadlock**: the barrier is never filled. Common ground: **neither mismatch is a
   numeric bug**; CPU reasoning cannot decide them, and after a change you must re-verify on the GPU.

2. **Every producer thread must commit; never wrap it in `elect_one` / `if tidx == 0`.**
   `cp.async.mbarrier.arrive` tracks "the cp.async previously issued by **this thread**"; it is a per-thread
   instruction, in the same class as LDGSTS itself. If only one thread arrives, only that thread's 16 B are
   tracked and all the rest are untracked. The dividing line is the same as [`cp_async`](cp_async.md) gotcha 3
   and [`elect_one`](elect_one.md) gotcha 6: **divide by "is this instruction one per thread or one per CTA",
   not by "is it a synchronisation primitive"**.

3. **`PipelineAsync` ≠ `PipelineCpAsync` — choosing the wrong one forces you to cancel out the mechanism by
   hand.** The two classes have different producer-side `PipelineOp`s, so `producer_commit` lands on different
   instructions:

   | Class | producer op | Instruction issued by `producer_commit` | When the barrier flips |
   |---|---|---|---|
   | `PipelineAsync` | `AsyncThread` | `mbarrier.arrive` | **at issue time** (data still on the way) |
   | `PipelineCpAsync` | `AsyncLoad` | `cp.async.mbarrier.arrive.noinc` | **at landing time** |

   Managing a `cp.async` ring with `PipelineAsync`, `producer_commit` releases the consumer before the data
   has landed, so you can only force the data to land yourself with `wait_group(0)` + a whole-WG rendezvous
   before the commit — **giving back exactly all the overlap this recipe is meant to earn**. That is not
   "being a bit conservative"; it replaces the mechanism with its opposite. `PipelineAsync` is for handoffs
   where both sides are ordinary threads (consumer→consumer), not for cp.async supply.

4. **API discoverability trap: `cp_async_mbarrier_arrive_noinc` is not in the `__all__` of
   `cute/arch/__init__.py`.** It enters the namespace via `from .mbar import *`; attribute access
   (`cute.arch.<name>`) works, but **`from cutlass.cute.arch import *` doesn't pick it up, and no api-audit based
   on an `__all__` / `dir()` whitelist finds it either**. The conclusion "the DSL lacks this capability, only raw
   inline PTX works" usually arises exactly this way (same as the reverse record about `CopyG2SOp` noted in the
   Sources of [`cp_async`](cp_async.md): search scope too narrow → wrongly concluding the DSL has no such
   interface → degrading to a fallback). One line checks it at runtime:
   ```python
   import cutlass.cute as cute, cutlass.pipeline as pipeline
   print(hasattr(cute.arch, "cp_async_mbarrier_arrive_noinc"))   # expect True
   print(hasattr(pipeline, "PipelineCpAsync"))                   # expect True
   ```

5. **The arrive must immediately follow this stage's copies, and come before the next stage's copies.** The
   instruction tracks "**all** previous" cp.async, so writing "issue stage k, issue stage k+1, arrive k, arrive
   k+1" is not wrong (`full[k]` only waits longer, the conservative direction), but `full[k]` is then held back
   by stage k+1 landing — **the lookahead eats itself, and the readings look almost as if there were no
   overlap**. The cost of a wrong order is performance, not correctness, so it can only be found through the
   machine-read footprint and timing; there will be no error.

6. **The consumer side must acquire / release stage by stage.** `consumer_wait`ing all `NSTAGE` stages at once
   at the top of the loop body and releasing them all at once at the end is equivalent to shrinking the ring
   capacity to "exactly one trip's consumption" — **lookahead depth drops to zero**; fixing the producer side
   is then wasted, and gather and compute still strictly alternate. The criterion is direct: the total ring
   capacity must be **strictly greater** than one trip of the loop body's consumption, otherwise the ring is
   topologically incapable of lookahead.

7. **`.create()` ends with a whole-CTA rendezvous (`agent_sync(Agent.ThreadBlock)`).** Put inside a role branch
   ⇒ only half the threads show up ⇒ deadlock. It must come before the single role branch, together with the
   rest of the shared prologue (in a warp-specialized kernel also before `setmaxregister_decrease`,
   [`warp_specialization`](warp_specialization.md) gotcha 11). To defer and manage it yourself, pass
   `defer_sync=True` and add `mbarrier_init_fence()` + a rendezvous yourself.

8. **Don't leave a `cp_async_wait_group(0)` in "just to be safe".** It is a blocking instruction; leaving it
   in means the mechanism is not in place. This entry's machine-read criterion exists for exactly this (next
   item). Likewise, this mechanism **does not need** `cp.async.commit_group`: group counting and mbarrier
   tracking are two independent mechanisms, and when only the latter is used, not a single one of the former
   needs to be issued.
8b. **The "in-flight depth" is itself a tunable knob, not "the deeper the better".** The previous item says
    leaving `wait_group` in "means the mechanism is not in place", but its context is **refusing to use
    arrive-on-completion when you should** — not enough overlap yet. Conversely, when **overlap is no longer the
    bottleneck and issue bursts are**, `wait_group(0)`'s "only one group in flight at a time" is a **deliberate
    supply pacer**, not a degeneration. The choice between the two mechanisms is **by bottleneck**, not by
    principle:

    | Bottleneck | Use | Mechanism |
    |---|---|---|
    | supply latency exposed on the critical path (tensor idle gaps) | arrive-on-completion (this entry) | the whole ring depth in flight at once |
    | the producer issues a full group in one burst and **disrupts** the consumers / L2 ports of the same CTA (`mio_throttle` / `lg_throttle` high, `tensor_pct` low with no supply wait) | commit → `wait_group(0)` → arrive (depth 1) | a group is issued, drained, and only then signals `full`; the SM is fed one group at a time |

    The **sign** of depth: depth 1 (`wait_group(0)`) turns "supply" from "how deep can it overlap" into "feed
    one bite at a time" — the cost is that each group's gmem round trip is exposed on the critical path (the
    degeneration described at the start of this entry), the gain is that **issue bursts are flattened**: no
    longer does one instant slam every stage's LDGSTS into the LSU / L2 queues at once. This direction has been
    measured as a positive gain once: changing arrive-on-completion's 3 slots in flight back to
    `commit_group → wait_group(0) → fence → mbarrier_arrive` (depth 1), on a tree where "each group's gather
    is already a full burst of 16 × LDGSTS.128 and the consumers and L2 ports are disrupted", the gate window
    went 10.2516 → 10.0912 ms (−1.56%). This reading is **the sum of two changes in one commit**: depth 1 (P1)
    landed bundled with "split one row's gather from one `cute.copy` site into 4 slices and insert the
    address-source load between the slices" (P2), with no reading for P1 alone or P2 alone; a third candidate
    (empty-arrive phase, P3) was built and measured on its own, 10.4091 ms (a 3.15% regression against
    10.0912), and does not enter the contract. Index prefetch itself is in
    [`gather_index_prefetch`](gather_index_prefetch.md); inserting gaps between slices has no entry of its own.
    **The reading varies with the tree; the criterion is the bottleneck**: first read NCU's
    `mio_throttle`/`lg_throttle` and `tensor_pct`, then decide whether to go deeper or shallower; do not
    default to "the more overlap the better".
    P2's machine-read footprint is an extra line `addr-source loads interleaved; prefetched` on the `supply`
    axis (the existing `clustered-at-top; prefetched` is still there): **between** two `cp.async` issues,
    insert an instruction unrelated to the fetched values but **with side effects** (`has_side_effects=True`)
    (e.g. an `ld.shared.b32` of the index needed next time, or an `LDS`). **[INFERRED]** Why it helps — "it
    opens a gap in the LSU's receive queue so the burst does not fill it with every LDGSTS within one clock
    cycle; the point is not the value of that load but the gap it opens" — is one reading of the summed
    readings, with no single-variable probe; start a probe before citing this explanation.

    **Certification** (KEEP evaluation, with the dossier's `commit` / `supply` axes + the gate window as
    evidence): besides the existing `mma-consumer: DEPBAR {LE 0x0 x2, LE 0x1 x1} LDGDEPBAR x0` (×8, unchanged
    before and after), the `commit` axis **gained** `copy-producer: DEPBAR {} LDGDEPBAR x1` (×5, the
    machine-read footprint of the producer group draining at depth 1); the `supply` axis gained
    `addr-source loads interleaved; prefetched`; gate window 10.2516 → 10.0912 ms (−1.56%). Depth 1 and slice
    gap insertion landed bundled, and the reading cannot be split; what it proves is "on a tree where the gather
    is already a full burst and supply and waits are already tight, this bundle still has a non-zero positive
    gain", not a general −1.56%, let alone the price of depth 1 alone. Pricing it alone requires a same-job
    A/B. The correctness contracts of P1 / P2 (slice bijection, depth 1 does not hang) are in the
    certification probe (not included in this release; `PROBE PASS` 11/11).

    The reverse readings (depth 1 → 3) on two trees where supply was not the bottleneck were zero to slightly
    negative; together with the −1.56% above they support only one **[INFERRED]** sentence: choose the
    in-flight depth by bottleneck — reduce to 1 when issue bursts are the bottleneck, deepen when supply
    exposure is. All three readings were bundled with other changes and all were within the node noise band;
    they are not single-variable evidence; do a same-job A/B before citing them. Machine-read footprint: on a
    tree with a back edge, the disappearance / appearance of the producer row (`DEPBAR {} LDGDEPBAR`) on the
    `commit` axis; on a fully unrolled tree this axis cannot be read — anchor on `cache_ops` or on the PTX
    counts of `cp.async.mbarrier.arrive` / `cp.async.wait_group` (gotcha 9).

9. **Machine-read footprint (can be copied directly as gate lines).** Correct values do not mean the mechanism
   is in effect — reverting to the blocking form computes correctly too, so acceptance must look at the issue
   footprint (for how to obtain the instruments, and the discipline "take the OR of three instruments; if all
   are mute, report an instrument failure", see [`cp_async`](cp_async.md) gotcha 1):
   ```python
   # PTX: mechanism present == there is an arrive and the producer does not drain
   n_arrive = len(re.findall(r"cp\.async\.mbarrier\.arrive", ptx))     # must be >= 1
   n_drain  = len(re.findall(r"cp\.async\.wait_(?:group|all)",  ptx))  # must be == 0
   n_copy16 = len(re.findall(r"cp\.async\.(?:ca|cg)\.shared",   ptx))  # must be >= 1

   # SASS: arrive-on-LDGSTS is its own mnemonic family (separable from ordinary mbarrier arrives)
   fam = set(re.findall(r"\b(?:LDGSTS|ARRIVES)[A-Za-z0-9_.]*", sass))
   hit = [m for m in fam if "LDGSTSBAR" in m or ("ARRIVE" in m and "LDGSTS" in m)]
   # mechanism present -> hit non-empty; blocking form -> hit empty, and DEPBAR appears (where wait_group lands)
   ```
   Note that `\bLDGSTS\b` (the copy itself) and `LDGSTSBAR...` (the arrive) must be counted separately: the
   latter is immediately followed by letters, so `\b...\b` will not miscount. **Don't hard-code the mnemonic as
   one string** to assert on — match by the family regex above; on a mismatch, first print `fam` to see the
   actual spelling, then change the criterion.

10. **Failure-mode classification table (classify first, then debug; don't debug a deadlock as a numeric
    bug).**

    | Symptom | Most likely cause |
    |---|---|
    | deadlock / job timeout with no output | arrive count too large; phase relabelling inconsistent between the two sides; `.create()` inside a role branch |
    | result wrong in per-thread blocks, the rest correct | arrive count too small (gotcha 1); only one thread commits (gotcha 2) |
    | result all 0 / all sentinel | the consumer is WGMMA and the async-proxy fence is missing (gotcha 11); or commit flips before the data lands (gotcha 3) |
    | values all correct but `tensor_pct` unmoved | mechanism not in place (gotchas 8/9); or arrive in the wrong position (gotcha 5); or the consumer waits for everything at the top (gotcha 6) |

11. **The async-proxy fence when the consumer is WGMMA is outside this entry's certified scope.** This entry's
    certification probe uses ordinary smem loads as the consumer (generic proxy), so it **only** certifies "this
    path is correct without a fence". WGMMA reads smem through the async proxy, and "a missing fence raises no
    error, the B operand is 0/garbage" recorded in [`cp_async`](cp_async.md) gotcha 9 /
    [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) gotcha 1 still holds on that path. The safe practice is to add a
    `fence.proxy.async.shared::cta` on the producer side **before** the arrive (the DSL has two spellings,
    `cute.arch.fence_view_async_shared` / `cute.arch.fence_proxy`; probe with `getattr` for which one exists);
    it is not a blocking instruction and does not eat overlap; **but whether it is still required after
    switching to this mechanism has no separate A/B in this entry**, marked [UNVERIFIED-BOUNDARY]; whoever hits
    it first adds a probe.

12. **Producer tail (`producer_tail`).** Before the producer exits, walk the ring once more with
    `producer_acquire`, so the kernel does not end with arrive signals still pending that nobody waits for.
    `PipelineAsync.producer_tail(state)` already packages this; in the raw spelling, wait on each `empty[s]`
    once more. This entry's probe did not make it a contract (smem is reclaimed only after all threads of the
    same CTA exit, so it does not trigger in this form), but constructions that reuse barriers across kernels or
    dealloc early must add it.

## Sources

- **PTX ISA**: the arrive-on-completion semantics of `cp.async.mbarrier.arrive{.noinc}.shared{::cta}.b64
  [mbar]`, and the rule for `.noinc` that "the pending count is not incremented at execution time, so that
  decrement must be accounted for in the mbarrier initialisation"; the group-completion semantics of
  `cp.async.commit_group` / `cp.async.wait_group N` (control); the phase convention of
  `mbarrier.try_wait.parity`.
- **Public infra API symbols** (the interface surface the student has to call anyway):
  `cutlass.cute.arch.cp_async_mbarrier_arrive_noinc`, `cutlass.cute.arch.mbarrier_init` /
  `mbarrier_init_fence` / `mbarrier_wait` / `mbarrier_arrive`, `cutlass.pipeline.PipelineCpAsync` /
  `PipelineAsync` / `CooperativeGroup` / `Agent.Thread` / `PipelineOp.AsyncLoad` / `PipelineOp.AsyncThread` /
  `make_pipeline_state` / `PipelineUserType` / `NamedBarrier`. The two-row comparison table of gotcha 3 is read
  directly from the `MbarrierArray.arrive` dispatch in `cutlass/pipeline/helpers.py`
  (`AsyncLoad -> arrive_cp_async_mbarrier`, `AsyncThread -> arrive_mbarrier`) and `PipelineCpAsync.create` in
  `cutlass/pipeline/sm90.py`; the arrive-count rule of gotcha 1 is read from `self.arrive_count = self.cg.size`
  + `mbarrier_init` in `MbarrierArray.__init__` of the same file.
- **Certified recipes of the same family**: [`cp_async`](cp_async.md) (the whole issue side: atom /
  partitioning / alignment / instrument acquisition / verdict discipline),
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md) (runtime-sliced barrier sub-ranges, phase
  relabelling, the relation between arrive count and deadlock), [`warp_specialization`](warp_specialization.md)
  gotcha 11 (the shared prologue must come before the role branch and the `setmaxnreg` dec),
  [`handoff_sync_choice`](handoff_sync_choice.md) (rendezvous vs mbarrier selection and triad differential),
  [`gather_index_prefetch`](gather_index_prefetch.md) (the supply-type triad; division of labour with this
  entry: that one governs "issue organisation", this one "completion announcement").
- **GPU-certified** (main probe) — three variants of the same kernel
  (`RAW` raw `cp_async_mbarrier_arrive_noinc` / `PIPE` `PipelineCpAsync` / `DRAIN` `PipelineAsync` +
  `wait_group(0)` + rendezvous + single-thread commit), with geometry and arithmetic identical word for word,
  differing only in how the producer announces a stage is full; 256 MB source, runtime non-monotonic repeatable
  row indices, 4-stage ring, 64 trips, 1 CTA/SM. Eight contracts C1-C8: each of the three variants bit-equal to
  the int32 reference (the control group must be the same function, otherwise the timing comparison is
  meaningless), the PTX of RAW/PIPE has `cp.async.mbarrier.arrive` with zero `cp.async.wait_group`, DRAIN's PTX
  is exactly the opposite, RAW's SASS has the arrive-on-LDGSTS mnemonic while DRAIN has 0, and
  `t(DRAIN)/t(RAW) ≥ 1.05` and `t(DRAIN)/t(PIPE) ≥ 1.05`.
