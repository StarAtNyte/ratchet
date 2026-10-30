# tile_boundary_wait_placement_prep_in_flight (wait placement at the tile boundary: the next tile's preparation runs during the flight of the previous tile's MMA chain)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**, warp-specialized persistent kernels with hand-written mbarriers.
>
> This entry records **one knob + its footprint + the conditions under which it is load-bearing**, not a percentage:
>
> | | |
> |---|---|
> | **Knob** | One role has two pieces of work per tile: a **prep segment** (gmem reads + elementwise work + one `tcgen05.st` writing the operand into TMEM) and a **finish segment** (the epilogue of the previous tile's accumulator). Two changes in one package: ① the loop runs `n_tiles + 1` passes, pass k = prep(k) then finish(k − 1), instead of prep(k) → finish(k); ② the "operand TMEM is free" mbarrier wait moves from the top of the pass to **immediately before `tcgen05.st`**, so all the gmem reads and elementwise work above it no longer queue behind the previous tile's last MMA |
> | **Footprint** | The barrier table (objects, counts, multicast, remote ownership) and every wait's phase expression are unchanged character for character; only the program points move — so the dossier has **no load-bearing axis**: `mbar_tiers` `{CONST: 48}` → `{CONST: 47, RING: 1}` and contract axis `rings` `[]` → `["stride=0x1"]` are both codegen side effects of the address chain ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11). What can be read is in the profiler and the timeline: `tensor_pct` 36.69 → 40.33, `stall_long_sb` 5.47 → 4.48, this role's per-tile period 48692 → 43774 (busy 4344 → 716, wait 41072 → 32932, in the timeline tool's own units) |
> | **Load-bearing condition** | Under the structure "1 CTA/SM, pair UMMA, a long prep segment (16 × 256-bit gmem reads + a few hundred elementwise instructions per thread), operand TMEM held by the previous tile's MMA chain until its last instruction, the MMA issuer waiting for 'operand ready' at the tile head", the gate window went **1.4387 → 1.3086 ms (−9.04%)**. On trees with a short prep segment, or where the operand TMEM frees up early anyway, this step has nothing to hide |

## Purpose

In a persistent loop, one role both prepares the **next** tile and finishes the **previous** one, and the last step of prep writes a resource
that the in-flight MMA chain is still reading (operand TMEM). Where the "wait until the resource is free" wait is placed decides whether the part
of prep that does **not** touch that resource — gmem reads, elementwise work, side results stored to smem — queues **after** the previous tile's MMA chain or runs
**during its flight**. ptxas does not move instructions across a wait; execution order is entirely determined by source order
(item ③ of [`branchfree_masked_elementwise_pass`](branchfree_masked_elementwise_pass.md) is the same fact in the form it takes in an
elementwise segment).

## When needed (trigger)

- The top of some role's loop body is an mbarrier wait for "the previous tile releases the resource", immediately followed by gmem reads and
  register arithmetic, while the instructions that actually write the resource are at the end of the segment.
- In the timeline / stall ledger: wait takes most of that role's period; the MMA issuer waits for "operand ready" at the tile head;
  `tensor_pct` is low while the barrier table checks out item by item (the wait-target relations are right; the placement just serialises prep behind).
- The per-tile "prep → finish" order puts the next tile's prep after this tile's finish, and the finish in turn waits for this tile's
  last MMA.

**Anti-trigger**: the prep segment is only a few instructions; or another role is busier and the period is not set by this wait (read the per-role busy first).

## Minimal runnable example

Skeleton (the role on the certified tree, with the operator-specific arithmetic replaced by comments):

```python
# pass k: prep(k) first, finish(k-1) after; n_tiles + 1 passes, the first pass has no finish, the last has no prep.
# one copy of the code per segment — not "one copy for the prologue + one for the loop body".
for k in cutlass.range(n_tiles + 1):
    tile_prep = first_tile + k * TILE_STRIDE          # each segment's tile index is hoisted above both regions
    tile_epi = tile_prep - TILE_STRIDE
    par_prep = k % 2
    par_epi = (k + 1) % 2                             # == (k - 1) % 2

    if k < n_tiles:
        # ---- prep(k): none of this touches operand TMEM; it runs during the flight of tile k-1's MMA chain ----
        for c in cutlass.range_constexpr(N_LOADS):
            cute.copy(g2r_atom, gX[(tile_prep, row(c), None)], rStage)   # gmem -> registers
            ...                                                          # elementwise work, pack into the r2t fragment
        sSide[(par_prep, tidx)] = side_result                            # side result stored to smem

        # ---- the wait sits right before the first instruction that writes the resource ----
        cute.arch.mbarrier_wait(mbar + BAR_OPERAND_TMEM_FREE, par_prep)
        for j in cutlass.range_constexpr(N_SUBTILES):
            cute.copy(r2t, rFrag[(None, 0, j)], tOperand[(None, 0, j)])  # tcgen05.st
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(mbar + BAR_OPERAND_READY, 0)           # publish after the st and the fence
        cute.arch.mbarrier_arrive(mbar + BAR_SIDE_READY)

    if k > 0:
        # ---- finish(k-1): wait for tile k-1's accumulator to be final, read back, write out, release the accumulator ----
        ...
```

Phase accounting: `BAR_OPERAND_TMEM_FREE` flips once per tile and phase 0 is pre-arrived by the prologue, so pass k's prep waits on
`k % 2` — pass 0 gets through on the pre-arrive, later passes on tile k − 1's flip. `BAR_OPERAND_READY` is arrived once per prep segment
= once per tile; the MMA issuer still waits on the same expression at the head of tile k, it is just that the phase has already flipped by the time it gets there.
No new ring is introduced: pass k's prep only waits on events inside tile k − 1, and the finish after it only waits on tile k − 1's own completion.

## Gotchas

1. **The barrier table must not change; only program points do.** The correctness argument for this step rests entirely on "objects, counts, multicast, remote ownership and phase
   expressions are all the same as before". Changing any count or phase along the way makes it a different change, and the failure mode is a hang.
2. **The two segments no longer share "this pass's tile index".** Prep uses tile k, finish uses tile k − 1: every per-tile input (positions,
   angles, per-tile scalars) is read separately by each segment; the parity of smem slots reused by parity (the double buffer for side results) follows the
   **prep tile**, not the pass index.
3. **Publish after `tcgen05.st` and its fence.** As soon as "operand ready" flips, the MMA issuer reads the operand face;
   put the arrive after `fence_view_async_tmem_store()`.
4. **Write the two segments inline in two `if` regions; don't factor them into local functions.** Recorded on the certified tree: calling a closure
   that captures runtime values inside a dynamic `if` region is rejected by the DSL preprocessor with `SCOPE_CLOSURE_CAPTURE`, and every statement of both segments uses
   that role's fragments. Hoist things like the tile index above both regions; neither region carries values out
   ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md), [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)).
5. **Once the wait has moved, everything computed above it must stay live across the wait.** The packed r2t fragment and the elementwise intermediates
   occupy registers during the wait. On the certified tree `spill_ld/st_sectors` read 2090832 / 1587072 →
   2095568 / 1582336 across this step (flat); for a role whose register tier is at the edge, count first with the region method of
   [`warp_specialization`](warp_specialization.md) before moving.
6. **The dossier's `rings` / `mbar_tiers` flips are not this step's footprint.** See the "Footprint" row in the header: zero ring changes; the flips are side effects.
   Accept on `tensor_pct`, per-role periods and the gate window.
7. **`bankconf_st` changes.** In the same KEEP, `bankconf_st` 1185218 → 2162100 while `bankconf_ld` stayed flat
   (8151487 → 8244148), with neither the smem face nor the store instructions changed — this entry has no attribution for why; trees with a hard line on store-side conflicts should check it too.

## Sources

- PTX ISA: `mbarrier.try_wait` / `mbarrier.arrive`, `tcgen05.st`, `tcgen05.fence`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL public API symbols: `cutlass.cute.arch.mbarrier_wait` / `mbarrier_arrive`,
  `cutlass.cute.arch.fence_view_async_tmem_store`, `cute.copy`, `cutlass.range` / `cutlass.range_constexpr`.
- KEEP evaluation (NCU dossier census + gate window; eval job id not recorded in the status record; the confirmation run for this reading read
  1.3159 ms): axis `rings` `[]` → `["stride=0x1"]`, gate 1.4387 → 1.3086 ms; `mbar_tiers`
  `{CONST: 48}` → `{CONST: 47, RING: 1}`, the other count axes and `cache_ops` / `nbar_*` / `supply` / `edges` unchanged;
  `tensor_pct` 36.69 → 40.33, `stall_long_sb` 5.47 → 4.48, `stall_barrier` 1.82 → 1.71, `smem_dynamic`
  186904 unchanged. Timeline readings before / after the same change (this role): period 48692 → 43774, busy 4344 → 716,
  wait 41072 → 32932.
- GPU probe in the same round (not included in this release): `PROBE PASS (max_diff=0.0008544921875) — 6/6 PASS` (whole-machine numeric
  contract; job id not recorded).
