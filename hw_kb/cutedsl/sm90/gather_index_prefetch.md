# gather_index_prefetch (batched index prefetch for indirect supply: don't let each row's fetch wait for its own index round trip)

> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose

Issue organisation for gather-style supply streams (`x[idx[i]]` — the index array must be read from
gmem before you know what to fetch): **decouple across rows** the "index load" from the "fetch by
address", so that a row-by-row serial chain does not expose a full gmem round-trip latency on
**every row**.

## When needed (trigger)

- **NCU triad (supply-bound)**: `stall_long_sb` many times above the reference + low `tensor_pct` +
  `stall_barrier` and DRAM bytes **within band**. It is the differential diagnosis of the
  sync-topology triad (barrier as the dominant term) — both have low tensor, and the dominant stall
  term tells them apart. `stall_long_sb` is the wait on a register-level gmem load: LDGSTS/TMA
  (asynchronous into smem) do not produce it — **the index load is exactly what produces it**.
- **SASS fingerprint**: fetch instructions in single-issue form — `LDGSTS` (or a per-row copy)
  interleaved one by one with address arithmetic (`LD STS LD STS ...`, each its own cluster) instead
  of a batched burst; the row address's dependency chain traces back to a gmem register load issued
  just before.

## Mechanism

In the serial form each row's critical path = `LDG idx[i]` (round trip of hundreds of cycles) →
address arithmetic → `LDGSTS row i`: **row i's fetch is stalled by its own index round trip**, so n
rows eat n exposed latencies. After decoupling:

1. **Batched index prefetch**: the indices of a batch of rows are first loaded into registers/UR in
   one go (vectorised/unrolled) — the index round trip of a batch pays the latency **once**;
2. **Cross-row pipelining**: while computing row i's address and issuing row i's fetch, the indices
   of rows i+1..i+k are already in flight or have arrived — the `LDGSTS` issue stream is no longer
   interrupted by index arrival;
3. When the index is a warp-invariant integer it can go into the uniform file (see
   [`uniform_reg_budget`](uniform_reg_budget.md)) and does not use the per-thread budget;
4. Invalid rows (sentinel indices) are handled with predication, without breaking the batched form.

Pricing of batch depth k: k × index-width registers buy away (n−1) exposed latencies — usually a
single-digit number of registers for an order of magnitude more issue density.

## Minimal runnable example

The certified path for row-by-row gather (cp.async per row, runtime non-monotonic repeatable indices)
is in [`cp_async`](cp_async.md); this entry's increment on top of it is the **issue organisation**:

```python
# Anti-pattern (serial chain): each row waits for its own index
for i in cutlass.range_constexpr(ROWS):
    r = mIdx[base + i]                 # LDG round trip ~hundreds of cycles
    copy_row(tma_or_cpasync, src_row=r, dst_slot=i)   # stalled behind the previous row

# This entry's form: batched prefetch + cross-row decoupling
idxs = []
for i in cutlass.range_constexpr(ROWS):     # issue the whole batch of indices first
    idxs.append(mIdx[base + i])             # loads are mutually independent, overlap in flight
for i in cutlass.range_constexpr(ROWS):     # issue fetches back to back once they arrive
    copy_row(tma_or_cpasync, src_row=idxs[i], dst_slot=i)
```

(`ROWS` is the batch depth; a deeper form software-pipelines the two loops into "prefetch batch j+1 ‖
issue batch j".)

When the row set carries invalid markers (sentinel indices / a runtime valid length) and invalid rows
must be 0 in smem, the **predicated DSL spelling** of the two-loop form above (clamped select +
unconditional load + live flag → `if live` back-to-back issue / `STS.128` zero fill) is in
[`gather_index_prefetch_predicated`](gather_index_prefetch_predicated.md).

## Gotchas

1. **The compiler will not decouple for you**: with a per-row `idx→copy` dependency chain in the
   source, the scheduler can only reorder within a single row — cross-row decoupling must be written
   into the source structure (two loops / software pipelining).
2. **Don't use bytes as the criterion**: the serial chain does not read a single extra byte (DRAM
   bytes within band is exactly its signature); it is slow only through exposed latency. The bytes
   line catches reuse problems; the `stall_long_sb` line is the one that catches this entry.
3. Batch depth is bounded by the register budget — putting the indices into UR (when warp-invariant)
   bypasses the per-thread quota; when the index has a lane component it lands in the regular file
   per the `+ tidx` rule of [`uniform_reg_budget`](uniform_reg_budget.md), so count it in the budget
   before choosing the depth.
4. Orthogonal to the handoff protocol: the mbarrier pipeline
   ([`handoff_sync_choice`](handoff_sync_choice.md)) governs the producer↔consumer handoff; this entry
   governs the producer's **own** issue density — when "the protocol is right but the producer is
   the slow source" (barrier within band, long_sb off the charts), this entry is what to fix.

## Sources

- **PTX ISA**: `ld.global` (register-level load and scoreboard dependency semantics),
  the `cp.async` / *Asynchronous copy* chapter (asynchronous copies do not occupy the register
  scoreboard).
- **Official NCU metric**: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`
  (this harness's alias `stall_long_sb`; Kernel Profiling Guide, warp stall reasons).
- **GPU-certified measured criteria** (in-house three-probe series + A/B):
  the serial form measured `stall_long_sb` at 12.7× the reference (8.91 vs 0.7) while
  barrier/bytes/spill were all within band and tensor was 31%; sleep-quantum A/B (`timeout_ns`
  10ms→2µs, the SASS immediate flip was confirmed) device 4.0642→4.0634ms **-0.0%** — ruling out the
  wait form itself and pinning it on issue organisation; the single-issue `LD STS`×64 interleaved
  skeleton is the source of the SASS fingerprint.
