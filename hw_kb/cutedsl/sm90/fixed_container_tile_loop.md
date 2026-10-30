# fixed_container_tile_loop (fixed-container loop across tiles: allocate the register working set once, overwrite it tile by tile)

> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose

The construction that keeps a flash-shaped working set (a resident large accumulator + a small
accumulator rebuilt every tile + packed fragments) at **peak = sum of the containers** with zero spill
across a tile loop under a tight `setmaxnreg` budget.

## When needed (trigger)

- The mainloop's fp working set exceeds the `setmaxnreg` budget, and `LDL/STL` show up inside the MMA
  envelope;
- or when fixing spill by "release the upstream early after conversion" doesn't work — this entry
  gives another GPU-certified route.

## Mechanism

**Zero spill comes from "an exact total ≤ budget + container reuse", not from "releasing midway".**

1. **Containers are created once outside the tile loop**: a fully unrolled list of SSA values
   (compile-time indices) or a resident fragment; each tile's new data **overwrites the same
   container** (accumulators are reloaded with clear-accum semantics).
2. **Keeping the fp32 originals alive until the re-read point is legitimate**: after a bf16 copy is
   packed, the fp32 original lives until the deferred rescale re-reads it — as long as the total of
   "large accumulator + small accumulator + packed copy + scalars" is within budget, the three being
   live together is not an accident, it is the design. Do not release for the sake of releasing.
3. The compilation pipeline (DSL→LLVM→ptxas) is **faithful** to this shape: straight-line code
   overwriting tile by tile gets its registers allocated once and reused across the loop (GPU
   certification in "Sources" — budget 216, container sum 176 words, 8 tiles, `LDL/STL == 0` and the
   values bit-exact; the control variant with +64 non-rematerialisable live values spills 70
   instructions, proving the budget really bites and the test is not too loose).

## Minimal runnable example

Skeleton (prototype of the certification probe's F variant; the probe is not included in this
release):

```python
rO = [cutlass.Float32(0.0) for _ in range(128)]     # resident accumulator: created once outside the loop
for t in cutlass.range_constexpr(T):
    p = s                                            # per-tile serial production chain
    rP = []
    for i in cutlass.range_constexpr(32):
        p = p + ...                                  # overwrite semantics: does not read the old rP
        rP.append(p)
    rS = [rP[i].to(cutlass.BFloat16) for i in range(32)]     # pack
    for k in cutlass.range_constexpr(128):
        rO[k] = rO[k] + rS[k % 32].to(cutlass.Float32)       # consume
    for i in cutlass.range_constexpr(32):            # deferred rescale: re-read the fp32 originals
        rO[(i*4+t) % 128] = rO[(i*4+t) % 128] + \
            (rP[i] * cutlass.Float32(0.25)).to(cutlass.BFloat16).to(cutlass.Float32)
    s = rP[31] - rP[30]                              # cross-tile data dependency (serialises the tiles)
```

## Gotchas

1. **"Packing in place lowers the peak" has been falsified** — see the falsification record in
   [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md). Don't expect a conversion step to
   free registers for you; only the total-budget design pays off.
2. **The spill counter is a whole-kernel sum; partition first, then assign blame**: each
   `setmaxnreg` role zone of a warp-specialised kernel has its own budget (e.g. 72/216). Measured
   lesson: in a kernel whose mainloop was at 211/216 with zero spill inside the envelope, all 133
   `LDL/STL` were in the **address bookkeeping of the 72-register producer zone** (once per CTA,
   negligible performance share) — blaming by the total would burn the whole budget on a zone that is
   not the lesion. Always start with a position histogram of `LDL/STL` (partitioned by the MMA
   envelope and the dec/inc markers).
3. **A fully unrolled tile loop pays an address tax**: once 8 trips are unrolled into straight-line
   code, the scheduler keeps the address vectors of different trips alive together (measured
   addr/integer temporaries +~20 words). When the container sum approaches the budget limit, keep a
   real loop (`cutlass.range`) or halve the unroll.
4. When a container is an SSA list, indices must all be compile-time (`range_constexpr`); a runtime
   index demotes the fragment into local memory and pollutes both variants alike (see the
   construction discipline of [`uniform_reg_budget`](uniform_reg_budget.md)).

## Sources

- **CuTe DSL public API**: `cutlass.range_constexpr` (host-side unrolling), scalar
  `.to(cutlass.BFloat16/Float32)` conversion, `cute.arch.setmaxregister_increase/_decrease`
  (role-zone budgets; certified spelling in [`warp_specialization`](warp_specialization.md)).
- **GPU-certified**: the certification probe `PROBE PASS` ×5
  contracts — the setmaxnreg four-piece set measured present in PTX; F (container form) and C (+64
  independent live values control) both match a bit-exact numpy model exactly (1/16 grid-point
  arithmetic + bf16 RNE software emulation); `LDL/STL(F) == 0`, `LDL/STL(C) == 70`.
- Partitioned reading, measured with a separate GPU probe.
