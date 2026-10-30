# persistent_tile_loop_ring_continuity (tile loop of a persistent CTA: mbarrier rings continuous across tiles + per-tile operand slot handshake + the next tile's slot issued from inside this tile's supply ring)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers, when turning a
> warp-specialized TMA/mbarrier kernel from "one CTA per tile, exit when done" into "one CTA per SM, walking several
> tiles by a static schedule", **how the synchronisation objects survive tile boundaries**: ring indices / phases are
> derived from a running count over the CTA's lifetime, the operand reloaded per tile goes into slots with a
> full/empty handshake, and once the slot count is raised to 2 the next tile's load is issued from inside this tile's
> supply ring. How the register working set is reused across tiles is in
> [`fixed_container_tile_loop`](fixed_container_tile_loop.md); this entry covers only the smem side and the mbarrier side.

## Purpose

- **grid = number of SMs** (the host reads it from the device properties), and each CTA walks several tiles by a
  static snake table;
- the supply ring (one full / empty mbarrier pair per stage) is **not reset** at tile boundaries: stage index and
  phase come from a running count that each side maintains on its own and never clears;
- the operand reloaded per tile (called X below) changes from "loaded once in the prologue" into slots with an
  X-full / X-empty handshake; with 2 slots, the next tile's X is issued as soon as this tile's supply ring has issued
  its R-th stage, using the remaining stages to cover its arrival latency.

## When needed (trigger)

- The number of work tiles is far larger than the number of SMs, and each CTA does only one tile: every tile pays a
  CTA prologue (descriptor prefetch, mbarrier initialisation, exposed load of the first operand, `setmaxnreg`
  tiering);
- when tile weight varies linearly with the tile number (e.g. KV length varying with position under a causal mask),
  a simple-stride CTA assignment is unbalanced; a snake (reversing direction every other round) makes the summed
  weight of two adjacent rounds a constant independent of the CTA number;
- structural conditions: one copy-producer warp + several consumer warpgroups, TMA into smem, mbarrier full/empty
  rings. **A shape where each CTA runs only one tile cannot detect any of this entry's errors** (see gotcha 1).

## Minimal runnable example

The skeleton below is extracted from a GPU-certified tree (operator logic removed; `X` is the operand reloaded per
tile, `Y` is the operand flowing through the ring per stage; `STAGE` ring depth, `X_SLOTS` slot count, `ARRIVE`
consumer arrive count).

### 1) Host: grid and static schedule, depending only on `bidx` and compile-time constants

```python
SM_COUNT = torch.cuda.get_device_properties(dev).multi_processor_count
GRID = min(SM_COUNT, N_WORK)          # one CTA per SM; the probe pins it small through the same entry point

def _n_rounds(cta):                   # rounds in which this CTA has work: divisions written as 0/1 indicators, no branch
    n_full = N_WORK // GRID
    rem = N_WORK - n_full * GRID
    off_last = cta if n_full % 2 == 0 else (GRID - 1) - cta   # which end of the snake the last short round starts from
    return n_full + (rem + (GRID - 1) - off_last) // GRID

def _tile_work(i, cta):               # round i -> tile number t (snake: even rounds forward, odd rounds reversed)
    off = cta + (i & 1) * ((GRID - 1) - 2 * cta)
    t = i * GRID + off
    return t, n_stages_of(t)          # number of stages this tile streams through the ring (operator-defined)
```

`cta = cute.arch.make_warp_uniform(bidx)`; the chain has no lane component at all, so the whole bookkeeping lands in
the uniform register file and takes nothing from the consumer's `setmaxnreg` quota
([`uniform_reg_budget`](uniform_reg_budget.md)).

### 2) mbarrier table: every buffer uses the same "base + slot/stage" addressing

```python
MB_X_FULL = 0                          # + slot
MB_X_EMPTY = MB_X_FULL + X_SLOTS
MB_Y_FULL = MB_X_EMPTY + X_SLOTS       # + stage
MB_Y_EMPTY = MB_Y_FULL + STAGE
NUM_MBAR = 2 * X_SLOTS + 2 * STAGE

@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, NUM_MBAR]
    sX: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(x_staged)], 1024]
    sY: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(y_staged)], 1024]

# the slot is the slowest (last) mode of the staged layout, exactly the same shape as the ring's stage
x_staged = cute.tile_to_shape(atom, (ROWS, COLS, X_SLOTS), order=(0, 1, 2))
y_staged = cute.tile_to_shape(atom, (ROWS, COLS, STAGE),   order=(0, 1, 2))

if warp_idx == 0:
    with cute.arch.elect_one():
        for s in cutlass.range_constexpr(X_SLOTS):
            cute.arch.mbarrier_init(mbar[(None, MB_X_FULL + s)].iterator, 1)       # TMA expect-tx
            cute.arch.mbarrier_init(mbar[(None, MB_X_EMPTY + s)].iterator, 1)      # the one thread that issues the store
        for s in cutlass.range_constexpr(STAGE):
            cute.arch.mbarrier_init(mbar[(None, MB_Y_FULL + s)].iterator, 1)
            cute.arch.mbarrier_init(mbar[(None, MB_Y_EMPTY + s)].iterator, ARRIVE)
cute.arch.mbarrier_init_fence()
```

### 3) Every view split off a staged tensor carries the slot as its last mode, sliced per tile by the runtime slot number

```python
def _slice_stage(t, s):                # last-mode slice = pure address arithmetic, no container rebuilt in the loop
    r = cute.rank(t)
    return t[tuple([None] * (r - 1) + [s])]

tXsX, tXgX = cpasync.tma_partition(tma_x, 0, cute.make_layout(1), cute.group_modes(sX_tma, 0, 4), gX)
tSrX = thr_mma.partition_A(sX)         # (.., X_SLOTS): WGMMA A fragment
tRS_sO = thr_r2s.partition_D(sX)       # (.., X_SLOTS): alias the epilogue lands back into this tile's own slot
```

### 4) Producer: the running count `run` derives ring index/phase; the next tile's X is issued from inside this tile's supply ring

```python
def _x_issue(rnd):                     # round rnd's X goes into slot rnd % X_SLOTS
    slot = rnd % X_SLOTS
    ph = (rnd // X_SLOTS + 1) % 2      # the producer leads by one phase: each slot's first use falls straight through on a new barrier
    cute.arch.mbarrier_wait(mbar[(None, MB_X_EMPTY + slot)].iterator, ph)
    with cute.arch.elect_one():        # expect-tx is a single-thread instruction
        cute.arch.mbarrier_arrive_and_expect_tx(mbar[(None, MB_X_FULL + slot)].iterator, X_BYTES)
    cute.copy(tma_x, tXgX[(None, coord_of(rnd))], tXsX[(None, slot)],      # the copy is issued by the whole warp
              tma_bar_ptr=mbar[(None, MB_X_FULL + slot)].iterator)

_x_issue(cutlass.Int32(0))             # the CTA's first X is issued in the prologue
run = cutlass.Int32(0)                 # stage running count over the CTA's lifetime, never cleared
for i in cutlass.range(n_rounds):
    t, n_st = _tile_work(i, cta)
    for j in cutlass.range(n_st):
        m = run + j
        idx = m % STAGE
        ph = (m // STAGE + 1) % 2
        cute.arch.mbarrier_wait(mbar[(None, MB_Y_EMPTY + idx)].iterator, ph)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar[(None, MB_Y_FULL + idx)].iterator, Y_BYTES)
        cute.copy(tma_y, tYgY[(None, coord_of(t, j))], tYsY[(None, idx)],
                  tma_bar_ptr=mbar[(None, MB_Y_FULL + idx)].iterator)
        if i + 1 < n_rounds:
            if j == R - 1:             # first R stages of this tile issued -> issue the next tile's X
                _x_issue(i + 1)
    if i + 1 < n_rounds:
        if n_st < R:                   # short-tail tile: the loop never reached R-1, the same issue goes after the loop
            _x_issue(i + 1)            # the two sites are mutually exclusive, exactly one runs (gotcha 6)
    run = run + n_st                   # advanced at the top level of the loop body: it is the value carried by the back edge
```

### 5) Consumer: the same running count; per tile only the accumulator and statistics are reset, and this tile's slot is released at the tail

```python
run = cutlass.Int32(0)
for i in cutlass.range(n_rounds):
    t, n_st = _tile_work(i, cta)
    slot = i % X_SLOTS
    x_ph = (i // X_SLOTS) % 2
    acc.fill(0.0); reset(stats)                                   # the only per-tile state
    cute.arch.mbarrier_wait(mbar[(None, MB_X_FULL + slot)].iterator, x_ph)
    for j in cutlass.range(n_st):
        m = run + j
        idx = m % STAGE
        ph = (m // STAGE) % 2                                     # the consumer phase gets no +1
        cute.arch.mbarrier_wait(mbar[(None, MB_Y_FULL + idx)].iterator, ph)
        ...  # read stage idx of tYsY, compute
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar[(None, MB_Y_EMPTY + idx)].iterator)
    run = run + n_st
    # epilogue: O lands back into the alias of this tile's own slot, the store is issued by one warp; the slot is returned only after it is read
    cute.arch.barrier(barrier_id=BAR_C, number_of_threads=CONSUMER_THREADS)   # this tile's X is dead
    cute.copy(tiled_r2s, rO_src, _slice_stage(tRS_sO, slot))
    cute.arch.fence_view_async_shared()
    cute.arch.barrier(barrier_id=BAR_C, number_of_threads=CONSUMER_THREADS)   # STSM visible to the async proxy
    if warp_idx == STORE_WARP:                                    # warp-aligned predicate (the copy atom elects within the whole warp internally)
        cute.copy(tma_o, tOsO[(None, slot)], tOgO[(None, coord_of(t))])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)         # the slot can go back to the producer only after smem has been read
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(mbar[(None, MB_X_EMPTY + slot)].iterator)
```

Why the producer waiting on X-empty inside tile t in §4 cannot form a cycle: it waits on slot
`(t+1) % 2 == (t-1) % 2`, which is released by **the epilogue of tile t−1** on the consumer side, and that epilogue
needs nothing more from the blocked supply warp (all of tile t−1's Y has long been issued). The supply warp can stall
there, but cannot close a loop.

## Machine-readable footprint (what to look at after landing, and what not)

| Look at | Reading (certified tree, one CTA per tile → persistent, two steps) |
|---|---|
| dossier `grid_size` | 1024 → 132 (= number of SMs) |
| dynamic smem | 164864 → 197632 B (second slot +32 KiB; 12 mbarriers, 96 B, land in the 1024 B alignment padding) |
| `mbar_objs` / `mbar_tiers` | the table grows from 10 objects to 12 (census reading 9 → 11, tier `{CONST: 9}` → `{CONST: 7, RING: 3, THREAD: 1}`) |
| `nbar_verdicts` / `nbar_widths` | the producer leaves the epilogue rendezvous: `symmetric rendezvous … width=288` and `[288]` disappear, only 256-wide remains; final state (gotcha 11) `[[256], [256], [256]]` → `[[128, 256], [256]]`, `symmetric rendezvous … width=256` disappears |
| `loops` | every role gets one more real back edge (the tile loop): `mma-consumer` 7 → 4 + `scalar` 1 (segments previously unrolled per tile merged into the loop body) |
| certified cubin invariants | `LDL+STL == 0` (slot slicing is address arithmetic), HGMMA count unchanged, DEPBAR histogram still contains `LE 0x1` |

| **Do not look at** | Why |
|---|---|
| contract axis `rings` | the two steps read `["stride=0x1,wrap=sel"]` → `[]` → `["stride=0x1", "stride=0x1,slots=9", "stride=0xff8"]`, while in the same rounds `loops / mbar_objs / mbar_tiers / mbar_per_role / edges` all moved and `cache_ops` did not move at all — per gotchas 12 / 14 of [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) record it as codegen moving along; not a basis |
| `cache_ops` | this entry changes no copy instruction, so it is always `UTMALDG.3D / UTMALDG.4D / UTMASTG.4D`; it not moving does **not** mean the structure did not change (gotcha 14) |

## Load-bearing condition (readings from one run, not generic percentages)

Structure: 1 copy-producer warp + 2 consumer warpgroups, 2-stage Y ring, 7–8 tiles per CTA, each tile's store still
waited on **on the spot** (`cp_async_bulk_wait_group(0, read=True)` on the epilogue's critical path).

| Step | Gate window | Other readings |
|---|---|---|
| one-slot persistent (tile loop + continuous rings + X-full/X-empty handshake + store moved to the consumer side) | 0.2421 → 0.2402 ms (−0.78%) | tensor 68.85 → 69.27%, regs 168 unchanged, spill 0 |
| second slot + the next tile's X issued after this tile's R-th stage | 0.2424 → 0.2284 ms (−5.78%) | tensor 69.27 → 72.22%, `stall_long_sb` 1.01 → 0.69 |

One-slot persistent alone is worth almost nothing: the prologue saved per tile is eaten by "the X load exposed at the
top of the tile + the store wait on the epilogue's critical path"; the gain is cashed in only after the second slot
hides X's load inside the whole tile's supply window. The two steps are one ⊗ package; the first step's reading is
not the package's reading.

## Gotchas

1. **One side resets and the other does not = hang, and the first tile does not show it.** If ring indices / phases
   are cleared at tile boundaries (or the two sides advance differently), if slot phases do not alternate, or if the
   one-time pre-arrive of a ping-pong is repeated every tile, it all shows up as a kernel that never retires. The
   certification shape must have **several tiles per CTA, wrapping the ring many times**: pin the CTA count small
   through the same entry point (a probe hook such as a host-side `SM_FORCE`), and cover **both parities** of the
   last short round (`n_full` odd / even decides which end of the snake the short round starts from). A one-CTA-one-tile
   shape detects nothing for this entry (gotcha 5 of [`pipeline_runtime_subring`](pipeline_runtime_subring.md)).
2. **Three sources of silently wrong answers, each with its own instrument.** Schedule table not a bijection
   (tiles missed / duplicated) → fill the destination with NaN first, so a missed tile stays NaN instead of averaging
   into plausible numbers; per-tile state not reset (accumulator / row statistics folding t−1 into t) → find the
   largest error by row; it lands at the positions of that CTA's second and later tiles; X-empty released before the
   store has finished reading the alias region → X(t+1) overwrites bytes the store is still moving. All three are
   **finite, plausible, and races** — write the contract as 12 launches with the same input, a NaN fill each time,
   all within tolerance and bit-for-bit identical.
3. **"Two slots" and "one slot indexed twice" are indistinguishable at run time.** A slot mode you forgot to enlarge
   makes the two slots alias; the kernel still runs and a single launch may happen to be right. The contract must read
   the traced layout's `cute.cosize` × element bytes: the X buffer must be exactly slot count × single-tile bytes, and
   total smem must fall in the expected window.
4. **The arrive count of X-empty is 1, not the number of consumer warps.** The slot is returned by the one thread
   that "issued the store and waited for it to be read" (arrive inside `elect_one`); copying Y-empty's 8 / 16 is a
   deadlock that never fills. When each WG has its own store warp, the count = number of store warps (gotcha 11).
5. **The scheduling bookkeeping must carry no lane component.** `_n_rounds` / `_tile_work` take only `bidx` and
   compile-time constants and are written with divisions rather than branches, so the whole chain stays in uniform
   registers; mixing in `tidx` even once pushes it into the consumer's `setmaxnreg` quota.
6. **Write `min(R, n_st)` as two mutually exclusive issue sites, not as a value computed in a runtime branch.** A name
   bound inside a runtime if region cannot leave that region ([`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md));
   the site inside the loop hits only when `n_st >= R`, the site after the loop only when `n_st < R`, and exactly one
   of them executes.
7. **expect-tx under `elect_one`, the TMA copy issued by the whole warp.** The DSL copy atom elects internally with the
   whole warp as the member mask; wrapping the copy in `elect_one` too leaves the other threads of the warp outside a
   block they are waiting in (gotcha 6 of [`elect_one`](elect_one.md) / gotcha 6 of
   [`tma_store_epilogue`](tma_store_epilogue.md)). Likewise on the store side: the predicate that issues the store must
   be warp-aligned (`warp_idx == k`), not `tidx == k`.
8. **After the producer leaves the epilogue rendezvous, the width must change with it.** Once the store moves to the
   consumer side, the producer is no longer a party to the rendezvous, and the 288-wide CTA rendezvous that included
   the producer must become a 256-wide consumer rendezvous; not changing the width = hang. (A transitional form; the
   final state deletes both of these rendezvous entirely, see gotcha 11.)
9. **The ping-pong pre-arrive happens once per CTA, not once per tile.** The arrive / sync counts between the two
   consumer warpgroups are continuous across tile boundaries (tile t's last body arrive pairs with tile t+1's first
   body sync); repeating the pre-arrive every tile is a counting error, not a no-op.
10. **Advance the running count at the top level of the loop body.** Put `run = run + n_st` at the outermost level of
    the tile loop body so it is unambiguously the value carried by the back edge; hiding it in a `const_expr` branch
    can make the producer and consumer sides disagree under some smoke tier.
11. **Each consumer warpgroup finishes the tile tail on its own: no cross-WG signal at all in the epilogue, and the two
    256-wide symmetric rendezvous are deleted entirely.** The "256-wide consumer rendezvous" of gotcha 8 is a
    transitional form; in the final state each WG touches only its own half of the rows:
    - the last lane merge of the row reduction + `MUFU.RCP` + normalisation + 16-bit packing are all done in
      registers, and `stmatrix` writes only this WG's own row region in the current operand slot (offset = WG index ×
      half-block bytes). The precondition is that the half-block byte count is an integer multiple of the 128 B swizzle
      period (half block = 4096 elements in this example), so the two halves share no byte — each can write its own
      without a rendezvous;
    - `cute.arch.fence_view_async_shared()` + a **128-wide, intra-WG** NamedBarrier (each WG with its own id) makes
      the generic-proxy writes of this WG's 128 threads visible to the warp that issues the store; missing either is a
      silent numerical error, not a crash;
    - this WG's store warp (`warp_idx % 4 == 0`, a warp-aligned predicate, gotcha 7) issues its own half block with a
      **second** store descriptor (box covering only a half block, coordinate offset = WG index × half-block rows),
      and then walks straight into the next tile's first-block window;
    - draining the previous tile's store (`cp_async_bulk_wait_group(0, read=True)`) moves to **the head of the next
      round**, after the first chain of the first block has been issued, and only after draining does it arrive on
      the slot's empty. So the slot-empty arrive count = number of consumer WGs (the "= 1" of gotcha 4 is the bill
      for the single-store-thread form; with one store warp per WG it becomes 2); the store of the CTA's last round
      is drained separately outside the tile loop;
    - the tile's first block follows the same window protocol as steady-state blocks (take the baton → ring full →
      chain → pass the baton, just without the second chain), and the last chain of the tile's last block is **not**
      wrapped in a window: the number of blocking rendezvous per tile is exactly the number of blocks, the two WGs
      never meet at a tile boundary, and the phase difference established by the ping-pong carries straight across
      tile boundaries (the cross-tile count of gotcha 9 is cashed in here).
    **Machine-readable footprint**: in `nbar_verdicts`, `symmetric rendezvous (all sides SYNC -> lockstep) width=256`
    disappears and `publish-handoff … width=256` → `width=128/256`; `nbar_widths` `[[256], [256], [256]]` →
    `[[128, 256], [256]]`; `mma-consumer→mma-consumer` disappears from `edges`; `mbar_objs` stays 12 (only the arrive
    count changes); the disappearance of `stride=0x400` from `rings` is codegen moving along (gotcha 15 of
    [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)).
    **Load-bearing condition**: the whole tile-boundary de-lockstep package landed at once (including the first-block
    window protocol, the deferred drain, empty count 2 etc.; it cannot be split), gate 0.2262 → 0.2198 ms (−2.83%),
    tensor_pct 73.09 → 74.92, stall_barrier 1.23 → 1.07, stall_mio 0.15 → 0.07; **`smem_wavefronts_st` 263709 →
    382225 (+45%)** — a cost-side reading within the same KEEP, which the student's record attributed to the two WGs'
    `stmatrix` bursts no longer being aligned by the rendezvous, not separately isolated; when reading this axis, do not
    take it as a regression. Structure: persistent with two slots, 2 consumer WGs in ping-pong, a single copy warp,
    2-stage ring.


## Sources

- **PTX ISA**: *Parallel Synchronization and Communication Instructions* — `mbarrier` phase-parity semantics and
  `expect-tx`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>;
  `cp.async.bulk.wait_group.read`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group>
- **CuTe DSL public API symbols**: `cute.tile_to_shape`, `cute.struct.MemRange` / `cute.struct.Align`,
  `cutlass.cute.nvgpu.cpasync.tma_partition` / `prefetch_descriptor`, `cute.arch.mbarrier_init` /
  `mbarrier_init_fence` / `mbarrier_wait` / `mbarrier_arrive` / `mbarrier_arrive_and_expect_tx` /
  `elect_one` / `make_warp_uniform` / `barrier` / `fence_view_async_shared` /
  `cp_async_bulk_commit_group` / `cp_async_bulk_wait_group`, `cutlass.range` / `range_constexpr`,
  `torch.cuda.get_device_properties(...).multi_processor_count`.
- **GPU-certified probes (one per step; the probes are not included in this release)**:
  - one-slot persistent: six contracts all passed in the same round — C1 CTA count pinned to 40 (3–4 tiles per CTA, wrapping many times)
    and C2 pinned to 25 (5–6 tiles, the other parity of the short round) within 2e-2 of the fp32 reference; C3
    production shape grid 132 `PROBE PASS max_diff=0.00311646`; C4 12 launches with NaN fill bit-for-bit identical;
    C5 `LDL+STL=0`, HGMMA 32, DEPBAR histogram contains `LE 0x1`; C6 schedule table is a bijection + snake load
    spectrum (against a plain stride).
  - second slot + earlier issue point: seven contracts all passed
    in the same round — C1–C6 as above (C3 `PROBE PASS max_diff=0.00311646`, the first whole-kernel launch of the
    197632 B map), C7 traced smem map: the X buffer is exactly two single tiles, `NUM_MBAR == 12`, total within the
    [196608, 199680] window.
- **KEEP evaluation** (NCU dossier census + gate window): the eval job number of one-slot persistent was not entered in
  the status record; the second slot was KEEP in an eval job (paired confirmation window, device 0.2284 ms,
  consistent with the gate reading), axis `rings` `[]` → `["stride=0x1", "stride=0x1,slots=9", "stride=0xff8"]`
  (moving along, not a basis), gate 0.2424 → 0.2284 ms.
- Gotcha 11 (independent epilogue per WG): dossier census + gate window of the KEEP evaluation (the eval job number was
  not entered in the status record), axes `nbar_verdicts` / `nbar_widths` / `edges` as recorded in gotcha 11;
  correctness is carried by the whole-kernel gate's correctness check, with no separate probe.
- Related entries: [`fixed_container_tile_loop`](fixed_container_tile_loop.md) (cross-tile reuse on the register side),
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md) (phase mismatches all show up as hangs),
  [`per_slot_landing_depth`](per_slot_landing_depth.md) (producer one beat ahead + the predicate wrapping only the copy),
  gotcha 10 of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (the repeated-launch contract),
  [`gpu_probe_ladder`](gpu_probe_ladder.md) (one rung per process, report a hang as a hang).
