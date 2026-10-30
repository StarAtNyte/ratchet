# gather_index_lane_owned (index supply for a gather producer: one per lane + `shfl.sync.idx` broadcast; no pinned uniform batch, no smem relay)

> **Certification source: the sm90 tree (the `shfl.sync.idx` broadcast form, GPU-probe certified) · two forms re-checked on sm100, both certified by KEEP census:
> lane-owned + lane-quad `shuffle_sync` assembling TMA gather4 coordinates (see the "sm100 form" section), and gather4 row coordinates owned by lanes in
> groups (see gotcha 6). The pricing readings are still only the sm90 ones.**
>
> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **positive recipe** after [`gather_index_prefetch`](../sm90/gather_index_prefetch.md) (the issue organisation of prefetches) and
> "pinning the indices into a uniform batch" were refuted by measurement:
> when a warp must gather N rows by index and the indices themselves come from one gmem load, **spread the N indices over
> N lanes, one each**, and at copy time broadcast the index of row i with `shfl.sync.idx`. Each thread spends only
> **1 register**; there is no batch for ptxas to break up or spill, and the index round trip is paid only once across the N rows.
> Same-job single-variable measurement **+5.6% ratio / −5.4% time**, producer spills 20 → 12 LDL/STL.

## Purpose

A gather-type producer (the top-k KV of sparse attention, MoE token routing, any `x[idx[i]]`)
must know the row number before moving each row, and the row numbers live in another tensor. Of three ways of holding them, only one neither
spills nor serialises under CuTe DSL:

| Spelling | Registers per thread | Measured (this entry's certified tree) | Problem |
|---|---|---|---|
| **uniform batch** (32 lanes read the same address, `make_warp_uniform`, batch depth 4) | 2×4 sought but not obtained | 3.394 ms (baseline) | ptxas does not put it in the uniform file; landing per-thread it spills: LDL/STL 20/20, `tight-chained 7` |
| uniform batch depth 16 (a whole stage) | — | **cubin identical to batch depth 4** | ptxas breaks the batch back into per-row loads; the depth parameter is empty |
| **one per lane + `shfl.sync.idx`** | 1 | **3.211 ms (−5.4%)** | LDL/STL 12/12, producer exposure 357 → 305 µs, spin exposure 1.124 → 0.996 ms |
| gmem→smem relay (one LDG → STS → named barrier → LDS, the reference's form) | 0 | 3.530 ms (**+4.0%**) | one LDS before every copy (284 of them), `short_scoreboard` rises from 0 to 38% of producer stalls, spin exposure 1.245 ms |

## When needed (trigger)

1. The producer's addresses come from **load results** (not an affine chain of `bidx`/constants): the uniformisation of `uniform_reg_budget` mechanism 1
   does not work for it -- the batch measurably cannot be pinned (rows 1/2 of the table above are that reading).
2. The dossier's supply axis reads `tight-chained copies > 0` (a copy's address is used right after it comes back from memory) or the producer segment
   has LDL/STL; raising the batch depth `G` leaves the cubin unchanged or makes spills jump.
3. The number of rows N a warp gathers is ≤ 32 (one row per lane fits). When N > 32, each lane holds ⌈N/32⌉,
   still far fewer than a batch.

## Minimal runnable example (CuTe DSL + one line of inline PTX)

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

@dsl_user_op
def shfl_idx(v, lane, *, loc=None, ip=None) -> cutlass.Int32:
    """shfl.sync.idx.b32: broadcast lane `lane`'s 32-bit value to the whole warp."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(v).ir_value(loc=loc, ip=ip),
         cutlass.Int32(lane).ir_value(loc=loc, ip=ip)],
        "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=r,r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

# In the producer's per-stage fill block (one warp = one thread team, responsible for ROWS rows):
lane = tidx % 32
my_row = lane % ROWS                       # lane i is responsible for row i's index (i < ROWS)
my_idx = mIdx[(token, col + my_row * ROW_TEAMS)] & kv_wrap   # one LDG per lane, 1 register
for i in cutlass.range_constexpr(ROWS):
    ki = shfl_idx(my_idx, i)               # row i's index: broadcast, no register batch
    cute.copy(tiled_kv,
              thr.partition_S(mKV[(ki, slab, None)]),
              thr.partition_D(sKV[(i * ROW_TEAMS + team, None, st)]))
```

If the DSL side has an idx broadcast like `cute.arch.shuffle_sync`, it can be used directly; the inline PTX above is because
cutlass-dsl 4.6.1, current when this entry was certified, exported only `shuffle_sync_bfly`.

## Machine-read footprint (acceptance criteria)

1. **SASS counts**: `SHFL.IDX` == rows per stage × number of stages (16×4 = 64 here); producer segment
   `LDG.E` drops from one per row to one per warp (68 → 8); `LDL`/`STL` drop (20 → 12). `HGMMA` and
   `LDGSTS` unchanged (a producer-only change).
2. **Same-job A/B timing**: compared with the uniform-batch version, device ms must drop significantly (measured −5.4%, jitter over three rounds
   < 0.03%).
3. **Role exposure ledger**: copy-producer exposure ns drops (357 → 305 µs), the `long_scoreboard` share drops
   (60 → 55%); spin exposure drops with it (1.124 → 0.996 ms).
4. **The change in how the supply axis reads is normal**: addr-source loads 87 → 18 (the spill refills are gone), the shape goes from
   `interleaved` to `clustered-at-top`, tight-chained 7 → 11. **In this form these two readings
   are not a problem**: the address chain is now `SHFL → LDGSTS` back to back, and the tool counts the copies after SHFL as tight-chained;
   but SHFL is a register operation (a few cycles), not a gmem round trip. Judge by timing and LDL/STL, not by this axis.

## sm100 form: lane-owned indices, lane-quad `shuffle_sync` assembling TMA gather4 coordinates directly (certified by KEEP census)

The supply role issues `UTMALDG.2D.GATHER4` (one instruction moving 4 rows by 4 row coordinates + 1 column coordinate), with the coordinates coming from
the index table. In the sm90 form a lane holds one index and broadcasts it to the whole warp at copy time; here it goes one step further: **the lane that issues the copy is
a neighbour of the lanes holding the coordinates** -- lane v holds the index of row v of the window, and v's lane quad `(v & ~3) .. +3` holds exactly
the 4 rows that gather4 needs, so four `shuffle_sync` assemble them into 4 compile-time register slots of the issuing lane.
Per lane per stage: one scalar index load, one gather4 issue site, one `vote.ballot` producing the valid bitmap.

### Knob (how to write it in the DSL)

```python
# One scalar index load per lane. Mark the inline asm has_side_effects=True: this is the next stage's prefetch,
# and marked as a pure read it would be sunk to its first use -- back onto the gather's own dependence chain (prefetch distance zero).
IDX1_ASM = "ld.global.nc.L1::no_allocate.L2::256B.b32 $0, [$1];"
IDX1_CONS = "=r,l"
def _idx1(m_idx, off):
    addr = m_idx.iterator.toint() + cutlass.Int64(off) * cutlass.Int64(4)
    return cutlass.Int32(llvm.inline_asm(T.i32(), [cutlass.Int64(addr).ir_value()],
                                         IDX1_ASM, IDX1_CONS, has_side_effects=True,
                                         is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))

def _clamp_row(v):                    # branch-free clamp to [0, n_rows), out of range -> row 0 (validity goes through the bitmap separately)
    v = v - (v & (v >> 31))           # negative (including INT32_MIN) -> 0
    return v & ((v - n_rows) >> 31)   # >= n_rows -> 0

def _valid_bit(v):                    # 1 iff 0 <= v < n_rows, no wraparound at either end
    nneg = ~(v >> 31)
    return (nneg & (((v & nneg) - n_rows) >> 31)) & 1

def _quad_coords(cl, v):              # the issuing lane takes its quad's 4 coordinates: the source lane is per-lane, i is static
    src = (v >> 2) << 2
    return [cute.arch.shuffle_sync(cl, src + i) for i in range(4)]

# The supply warp's role body (all 32 lanes run it; only the bitmap byte and the arrive stay on lane 0)
v = lane % NWIN                                   # the window row this lane holds = the unit number it may issue
issue = ((v - lo) | (hi - 1 - v) | (NWIN - 1 - lane)) >= 0   # loop-invariant per-lane issue predicate: one signed comparison
cur = _idx1(mIdx, first_stage_off + v)            # stage 0's index, loaded outside the loop
for s in cutlass.range(n_stages):
    nxt = _idx1(mIdx, next_stage_off(s) + v)      # (1) the next stage's index, issued at the very top of the body
    cl = _clamp_row(cur)                          # (2) clamp + quad coordinates both before the ring wait
    crd = _quad_coords(cl, v)
    cute.arch.mbarrier_wait(b_empty + slot, ph)   # (3) the slot's previous generation has been consumed
    if issue:                                     # (4) one issue site, one loop-invariant predicate
        _gather4(dst_run, tmap, (col0, *crd), b_full_addr)
    ball = cute.arch.vote_ballot_sync(_valid_bit(cur) != 0)   # (5) the bitmap after the issue
    if lane == 0:
        # <write ball's nibble into the bitmap face>
        cute.arch.mbarrier_arrive(b_mask_full + slot)
    cur = nxt                                     # (6) rotate: one scalar iter_arg per lane
```

The form it replaces: per supply warp one elected lane holding 16 indices (two 256-bit `v8` loads), walking an unrolled
window of 16 units, 13 issue sites with run-time guards, and 16 scalar iter_args rotating across the ring's back edge.

### Footprint (NCU dossier census + gate window, the same KEEP)

| | Before (elected lane) | After (lane-owned) |
|---|---|---|
| contract axis `rings` | `["stride=0x38ad0"]` | `["stride=0x1", "stride=0x1,wrap=sel", "stride=0x2,wrap=sel", "stride=0x38ad0"]` |
| `cache_ops` | contains `UTMALDG.2D.GATHER4.2CTA` | unchanged |
| gate (device window) | 1.1778 ms | **1.013 ms (−13.99%)** |
| `tensor_pct` | 46.21 | 58.77 |
| `stall_long_sb` | 2.22 | 3.87 |
| `stall_barrier` | 0.29 | 0.39 |
| `dram_pct` | 9.16 | 11.28 |
| `regs_per_thread` / spill | 128 / 0 | 128 / 0 |
| `smem_dynamic` | 232 248 | 232 248 (unchanged) |

The three extra `rings` entries are fingerprints of the index chain after the role body left `if lane == 0`
([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11, the fifth tree), not rings.

### Load-bearing conditions

Structural conditions: sm100 pair UMMA, 7 supply warps per CTA, each warp one 16-row window shard (64 gather4 units per stage,
each unit 4 rows × 64 columns = 512 B, all recorded on one `expect_tx` of the leader), a 4-slot ring,
prefetch distance 1 stage, supply warps living on the 80-register yielding tier. Under these conditions −13.99%, `tensor_pct`
+12.5 points: what was removed is 16 serial clamps on one lane + 13 guarded issue sites + 16 iter_args across the back edge,
replaced by one clamp in parallel over 32 lanes, 4 shuffles and one issue site. **This is this tree's reading** (the noise
floor of that tree's ablation is 1.62%), not a general percentage.

**Certification boundary [UNVERIFIED-BOUNDARY]**: the build chain carrying that KEEP was later abandoned as a whole (the code of the same tree that finally met the bar
still supplies in the elected-lane form). This section certifies the spelling, the footprint and its reading on that tree; it does not certify the gain of moving it alone
to another tree.

## Gotchas

1. **Raising the uniform batch depth to get prefetch**: the cubin does not change (measured: batch depths 4 and 16 have the same sha). ptxas breaks the uniform
   batch back into one load next to each consumption point; the depth parameter has no semantics. To prefetch, change the holding scheme, not the depth.
2. **Copying the reference's smem relay**: the reference (LDG→STS→LDS) is right in its own kernel, but moved to this tree it is
   **4% slower**: one more LDS before every copy, and the producer's `short_scoreboard` share rises from 0 to 38%.
   The reference can use it because its producer has something else covering the LDS latency; "the reference writes it this way" in the dossier is not
   "you should write it this way too" -- the shape must be priced (see `[PRICED:]`).
3. **Taking the supply axis going from `interleaved` to `clustered-at-top` as a regression**: see footprint item 4.
4. **When N > 32 rows, letting only the first N%32 lanes hold indices**: each lane holds ⌈N/32⌉, and the broadcast is `shfl_idx(my_idx[i // 32], i % 32)`.
5. Ceiling: this entry removes the producer's **index** round trip, but 1.0 ms of spin remains -- that is the round trip of the gather data itself
   (the long_scoreboard of 64 LDGSTS), which needs a deeper ring / earlier stage issue, not this entry.
6. **The form feeding `.tile::gather4` on sm100: each holding lane takes a group of 4 row coordinates.** One gather4 needs 4 row coordinates
   ([`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md)), so the holding unit changes from "one index" to "a group of four":
   lane g (g < number of groups) reads the 4 int32 of group g with **one** 128-bit gmem read into a 4-element rmem tensor, and at the issue site
   `cute.arch.shuffle_sync` (`shfl.sync.idx.b32`) broadcasts these 4 coordinates to the elected issuing lane. Three disciplines:
   - **The broadcast is executed by the whole warp, outside `elect_one`** (`shfl.sync` needs the whole warp to participate); the result is a
     dead value on non-issuing lanes; the gather4 itself is inside `elect_one`.
   - **Issue order unchanged**: the index read is **above** the stage-empty wait, the broadcast and the gather4 **below** it. Hoisting the broadcast above the wait too
     puts the whole batch of coordinates back on the register account as "living across the wait", which is equivalent to going back to batch holding.
   - **Sentinel rules are evaluated per lane**: the tests for running past the end of the index table / past the valid length are the same set of predicates, each computed on its own holding lane.

   | Axis | elected lane holds a batch of 24 | lane-owned (6 holding lanes × 4) | moved again to 3 warps (8 holding lanes × 4) |
   |---|---|---|---|
   | dossier `supply` | `addr-source loads hoisted; prefetched` | `addr-source loads clustered-at-top; prefetched` | same as left |
   | `supply_counts` | 24 rows "copies 1, no in-loop address-source loads" | `addr-source loads 1 …; copies 24; tight-chained copies 0` | `addr-source loads 4 …; copies 32; tight-chained copies 0` |
   | `merge.shfl_in_loop` | -- (no such key) | 24 | 32 |
   | `loops.copy-producer` | 28 | 3 | 13 |

   **Load-bearing conditions**: the first landing came in the same package as "switch the wait encoding to an explicit spin", gate 1.2812 → 1.3169 ms (+2.79%), a same-package reading
   that cannot be separated (for the spin half see [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 12). Afterwards the wait encoding and
   the register tiers were both switched back, leaving only this entry's tree, which differs from the pre-landing tree **only in this one change**; the readings of the two respective eval jobs
   (cross-job, not a same-job A/B):

   | | batch holding | lane holding |
   |---|---|---|
   | gate (device window) | 1.2812 ms | 1.2348 ms (−3.62%) |
   | `tensor_pct` | 42.91 | 43.99 |
   | `stall_long_sb` | 4.34 | 6.34 |
   | `spill_ld / st_sectors` | 588240 / 9472 | 3799504 / 3286272 |

   Under the structure "1 CTA/SM, 4 issuing warps each issuing 24 gather4, issuing role on the 128 tier" ms improves, while **spill sectors
   and `stall_long_sb` both get worse** -- the issuing role's own static reading is `LDL 2 / STL 0` (per-role attribution taken from the tree that also carried the spin
   encoding), so the extra spills are not in the issuing role's body, and this entry did not trace them to source lines. So this set of readings can only say "lane holding is worth −3.6% on this tree and is not free";
   it cannot say it saves spills; a tree with a hard spill limit must recount after landing. The issuing role can then yield registers: dropping to the 96 tier still gives
   `LDL == STL == 0`. `supply` going from `hoisted` to `clustered-at-top` reads per footprint item 4; it is not a regression.
7. **(sm100 form) The inline-asm load used for prefetch must be marked `has_side_effects=True`.** It is the next stage's index,
   issued at the very top of the body; marked as a pure read, the compiler sinks it to its first use, the prefetch distance drops from 1 stage to zero, and the gather's
   coordinates hang behind a gmem round trip again.
8. **(sm100 form) The shuffle's source lane is computed per lane, the index i is static.** `shuffle_sync(cl, ((v >> 2) << 2) + i)`,
   with i over `range(4)`: the four coordinates land in four compile-time register slots. Selecting from a register array with a run-time index = the whole array
   demoted to local memory, and `spill_*_sectors != 0` at the gate.
9. **(sm100 form) The whole warp runs the shuffle, and the issue predicate folds in `lane < NWIN`.** `shfl.sync` needs the full member mask;
   the upper half warp reads from the same 16 source lanes and does not issue; the three-part interval test is written as one signed comparison
   `((v - lo) | (hi - 1 - v) | (NWIN - 1 - lane)) >= 0`, loop-invariant, decided once per kernel.
10. **(sm100 form) Clamp and shuffle before the ring wait, the bitmap after the issue.** After the ring wait only the issue itself remains;
   the bitmap's consumer arrives one round later, so placing it before the issue only lengthens the issue chain.

## Sources

- ✅ Verified: certified by a GPU probe (the probe is not included in this release; based on a gather-attention tree that had already passed
  first-green)
  (H100, cutlass-dsl 4.6.1 image): four variants × three rounds,
  `PROBE PASS lane delta_vs_base=5.56% gates=ok`, `PROBE PASS smem delta_vs_base=-3.92%
  gates=ok`, `depth16 gates=cubin_same_as_base` (exactly gotcha 1's conclusion). All four versions `ROLE-OK … tensor=`
  + `PMTL-OK`.
- Price tags: `[PRICED: +5.6% ratio / −5.4% ms (lane vs uniform batch)]`;
  `[PRICED: smem relay −3.9% (slower)]`; `[PRICED: uniform batch depth 4→16 = 0 (cubin identical)]`.
- Public interfaces: PTX `shfl.sync.idx.b32`; `cutlass._mlir.dialects.llvm.inline_asm` (see
  [`ptx_escape`](ptx_escape.md)); `cutlass.cute.arch.make_warp_uniform` (this entry does **not** depend on it).
- Gotcha 6 (sm100, gather4 row coordinates held by lanes): GPU probe,
  `PROBE PASS (max_diff=0.00036487803405893526) — 5/5 PASS` (the same probe as for the wait encoding: numerics against an fp64 control,
  12 repeated launches on the same input bit-identical, a real shape launched whole-machine to completion with no holes; the job number was not recorded); the form with eight holding lanes
  is covered by a separate gather probe, `PROBE PASS (max_diff=0.0) — 6/6 PASS` (one stage bit-exact, sentinel rows
  zero, padded tail block zero). Two KEEP evaluations (NCU dossier census + gate window; the eval job numbers were not recorded in the status record): the first, axis
  `rings` `["stride=0x1"]` → `[]`, `supply` `hoisted` → `clustered-at-top`, gate 1.2812 → 1.3169 ms (same package);
  the second, `supply_counts` copies 24 → 32, gate 0.9993 → 0.9844 ms (same package). The pair of readings differing only in this one change
  (1.2812 / 1.2348 ms) are each taken from the eval job of the respective KEEP; the latter's confirmation run reads 1.2434 ms. Public interfaces:
  `cutlass.cute.arch.shuffle_sync`, `cutlass.cute.arch.elect_one`.
- ✅ **KEEP certification of the sm100 form** (contract-axis change, this index's second class of evidence; the eval job number was not recorded in the status record):
  axis `rings` `["stride=0x38ad0"]` → `["stride=0x1", "stride=0x1,wrap=sel", "stride=0x2,wrap=sel", "stride=0x38ad0"]`,
  `cache_ops` unchanged, gate 1.1778 → 1.013 ms, correctness PASS. The same-tree GPU probe cited by that KEEP's change record:
  a lane-supply probe (on that build chain), contracts `S13-lane-walk` (lane residency over the whole persistent schedule /
  the four shuffle coordinates / the ballot word / the group nibble / the issue predicate vs a numpy walk, max_diff 0, including item boundaries, ragged tails and
  both ends of int32), `S13-lane-g4` (the ring slots of a real (2,1,1) cluster issued by 7 warps × 9–10 lanes each and recorded on one
  `expect_tx` of the leader, max_diff 0, the leader's bounded wait sees the tx close), `S13-lane-mask` (0 bad bytes); verdicts transcribed from the change
  record; the probe receipt carried no job number.
- Public interfaces of the sm100 form: PTX `ld.global.nc.L1::no_allocate.L2::256B`, `shfl.sync`, `vote.ballot.sync`,
  `cp.async.bulk.tensor … .tile::gather4` (issue spelling see [`ptx_escape`](ptx_escape.md)); `cute.arch.shuffle_sync`,
  `cute.arch.vote_ballot_sync`, `cute.arch.mbarrier_wait` / `mbarrier_arrive`.
