# handoff_sync_choice (choosing the primitive for cross-role supply handoff: all-party rendezvous vs mbarrier pipeline)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is a
> **reading/selection recipe**: it teaches "when which synchronisation primitive is mandatory, and what
> the machine-read symptoms of a wrong choice look like"; the runnable paths of the two primitives are
> in the certified entries pointed to at the end, and this entry does not repeat their code.

## Purpose

For **streaming supply** from producer→consumer in a warp-specialised kernel (the producer keeps moving
a KV/operand stream stage by stage, the consumer consumes stage by stage), the choice of
synchronisation primitive decides whether supply latency can be hidden behind compute.

## When needed (trigger)

- **NCU triad**: low `tensor_pct` + low `dram_pct` + high `stall_barrier`. The three must be read
  together: when the two sides wait on each other, compute and bandwidth sit idle **at the same time**
  — reading only the first misdiagnoses "compute is too weak", reading only the second misdiagnoses
  "bandwidth is too weak"; actually the handoff protocol has serialised the two sides.
- **SASS criterion**: dense `BAR.SYNC` in the mainloop region (on the order of ~ per trip × per handoff
  point), while mbarrier-form `SYNCS.PHASECHK` / `DEPBAR` are absent → the handoff is an all-party
  rendezvous.

## Mechanism

| | NamedBarrier all-party rendezvous | mbarrier stage-by-stage pipeline |
|---|---|---|
| Semantics | released only when threads of **both** sides have arrived | phase flip, one side arrives / one side waits |
| Supply latency | **serialised into the critical path**: the producer waits for the consumer to finish computing before it dares overwrite, the consumer waits for the producer to finish moving before it can start computing | the producer runs N stages ahead; transfer latency hides behind the consumer's compute |
| Legitimate use | **one-off / low-frequency handoffs** (a constant number of rendezvous per kernel, epilogue meeting points) | **streaming supply** (the default choice) |

The cost of a wrong choice is an order of magnitude: a measured kernel whose streaming supply was
built as a rendezvous (in-house census probe, not included in this release) showed `BAR.SYNC`×38 in
the mainloop region, `tensor_pct` 15%, `stall_barrier` 16%, `dram_pct` 9% — the full triad, end-to-end
several times off the healthy form, while no single metric read on its own pointed at the sync
protocol.

## Minimal runnable example

No code repeated; two certified paths can be used directly:

- The mbarrier pipeline trio (TMA G2S → `PipelineTmaAsync` → WGMMA SS):
  [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)
- A per-WG independent pipeline with sub-rings cut at runtime (each consumer WG has its own full/empty
  stream): [`pipeline_runtime_subring`](pipeline_runtime_subring.md)

## Gotchas

1. **Don't overcorrect**: a one-off handoff that happens a constant number of times per kernel (e.g. a
   single large TMA load, an epilogue rendezvous) is perfectly legitimate with NamedBarrier — building
   a pipeline for it is a pure complexity tax. The criterion is **frequency × whether it is on the
   steady-state supply path**, not whether the primitive itself is good or bad.
2. **`stall_barrier` alone in the triad is not enough to convict**: kernels with dense algorithmic
   rendezvous points also show barrier stalls. Conviction needs the triad + the SASS-shape criterion
   (the rendezvous primitive appears on the **steady-state supply** path).
3. **Pick the mainloop region as the diagnosis window**: one-off rendezvous in the prologue/epilogue
   don't count; the census is bounded by the MMA envelope.

4. **On a handoff surface, "rendezvous + exit read-modify-write" is a pair that must be deleted
   together.** When the producer does one more read-modify-write pass over the same surface before
   delivering it (read back / scale / narrow / write back), it must first rendezvous all parties so
   every warp's stores are visible; if the consumer can consume the unmodified surface directly and
   fold that step into its own operand formation, then once the RMW is gone the rendezvous has no
   reason to exist either — a per-thread `fence_view_async_shared()` + mbarrier full arrive (count =
   all producing threads) already carries the same ordering, and a stronger one. One KEEP reading:
   inter-segment rendezvous per chunk 5 → 4, gate 0.435 → 0.4173 ms (−4.07%, bundled with the
   downstream switch to fp16 operands, not separable). The spelling and probe evidence are in
   [`tri_inverse_block_doubling`](tri_inverse_block_doubling.md) gotcha 14.

5. **Before adding another pair of NamedBarrier slots to two ping-pong consumer WGs to "stagger" an
   elementwise segment, read the phase column first — if the two segments already don't intersect,
   the slot pair is pure cost.** The knob: one pair of 256-wide slots in the steady-state body (128
   blocking + 128 arrive-only, the same asymmetry as the MMA slots); a WG blocks on its own slot when
   entering the segment and only arrives on the peer's slot when leaving, and one WG seeds once at role
   entry; blocking waits per body 5 → 6, mbarrier table, registers and smem all unchanged. It was landed
   on the strength of the trigger reading "the length of this segment + loop top + issue exceeds the
   peer's MMA window", while on the same timeline the two WGs' segments **already did not intersect**
   (segment length about 900, gap about 1300 cycles, the two WGs' segment lengths differing by only 10).
   Readings: gate 0.21361 → 0.2189 ms (+2.48%), `tensor_pct` 78.99 → 76.90, `stall_barrier` 1.27 →
   1.44; after removing it 0.2189 → 0.21381 ms and all three readings recovered. The dossier's `rings`
   going to zero / recovering as a whole when adding / removing is a follow-on effect
   ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 16).
   Rule: a rendezvous only buys something when the two sides **would really collide**; "one side's
   serial segment is longer than the other side's in-flight window" does not imply a collision — what
   you need before adding a rendezvous is an overlap reading on the phase column.

## Sources

- **PTX ISA**: `barrier{.cta}.sync` (named barrier, all-arrive semantics) in the *Parallel
  Synchronization and Communication Instructions* chapter, and `mbarrier.arrive` /
  `mbarrier.try_wait` / phase semantics (§9.7.13).
- **CuTe DSL public API**: `cutlass.pipeline.PipelineTmaAsync` (per-stage mbarrier pipeline,
  producer_acquire/consumer_wait/consumer_release), `cutlass.pipeline.NamedBarrier`
  (`arrive_and_wait`).
- **GPU-measured criteria** (in-house probes, two rounds of SASS census + NCU) — the triad readings and the SASS-shape criterion of the rendezvous-form
  kernel come from this measurement; the contrasting pipeline-form signature (`PHASECHK`/`DEPBAR`
  dominant, barrier stalls within band) comes from architecture-signature measurements on the same
  hardware.
- KEEP reading of gotcha 4 (eval job number not recorded): inter-segment rendezvous per chunk 5 → 4,
  value −4.07%, the `rings` axis flips with codegen; operand-path probe `PROBE PASS`.
- KEEP readings of gotcha 5 (one add / one removal, eval job numbers not recorded): axis `rings` three
  families ↔ `[]`, gate 0.21361 → 0.2189 → 0.21381 ms, `stall_barrier` 1.27 → 1.44 → 1.27. Status:
  ✅ KEEP-certified.
