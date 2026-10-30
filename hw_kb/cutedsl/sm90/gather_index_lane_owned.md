# gather_index_lane_owned (index supply for a gather producer: each lane holds one + `shfl.sync.idx` broadcast; no pinned uniform batch, no smem staging)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **positive recipe** that follows [`gather_index_prefetch`](gather_index_prefetch.md)
> (issue organisation of the prefetch) and [`uniform_index_batch`](uniform_index_batch.md) (how to
> measure that a uniform batch cannot be pinned): when a warp has to gather N rows by index and the
> indices themselves come from one gmem load, **spread the N indices over N lanes, one each**, and
> broadcast row i's index with `shfl.sync.idx` at copy time. Each thread spends only **1 register**;
> there is no batch for ptxas to break up or spill; the index round trip is paid once across the N rows.
> Single-variable measurement in the same job: **+5.6% ratio / −5.4% time**, producer spill 20 → 12
> LDL/STL.

## Purpose

A gather-type producer (sparse attention's top-k KV, MoE token routing, any `x[idx[i]]`) must know the
row number before moving each row, and the row numbers live in another tensor. Of three ways to hold
them, only one neither spills nor serialises under CuTe DSL:

| Form | Registers per thread | Measured (this entry's certification tree) | Problem |
|---|---|---|---|
| **uniform batch** (32 lanes read the same address, `make_warp_uniform`, batch depth 4) | 2×4 sought but not obtained | 3.394 ms (baseline) | ptxas does not put it in the uniform file; it lands per-thread and spills: LDL/STL 20/20, `tight-chained 7` |
| uniform batch depth 16 (a whole stage) | — | **cubin identical to batch depth 4** | ptxas breaks the batch back into per-row loads; the depth parameter is empty |
| **one per lane + `shfl.sync.idx`** | 1 | **3.211 ms (−5.4%)** | LDL/STL 12/12, producer exposure 357 → 305 µs, spin exposure 1.124 → 0.996 ms |
| gmem→smem staging (one LDG → STS → named barrier → LDS; the reference's form) | 0 | 3.530 ms (**+4.0%**) | one LDS before every copy (284 of them), `short_scoreboard` rises from 0 to 38% of producer stalls, spin exposure 1.245 ms |

## When needed (trigger)

1. The producer's address comes from a **load result** (not an affine chain of `bidx`/constants):
   mechanism 1 of `uniform_reg_budget` (uniformisation) does not work for it, and `uniform_index_batch`
   measured that the batch cannot be pinned.
2. The dossier's supply axis reads `tight-chained copies > 0` (the copy's address is used right after
   it returns from memory) or the producer segment has LDL/STL; raising the batch depth `G` shows an
   unchanged cubin or a jump in spill.
3. A warp gathers N ≤ 32 rows (one row per lane fits). For N > 32 each lane holds ⌈N/32⌉, still far
   fewer than a batch.

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

# Inside each stage-fill block of the producer (one warp = one thread-team, responsible for ROWS rows):
lane = tidx % 32
my_row = lane % ROWS                       # lane i owns row i's index (i < ROWS)
my_idx = mIdx[(token, col + my_row * ROW_TEAMS)] & kv_wrap   # one LDG per lane, 1 register
for i in cutlass.range_constexpr(ROWS):
    ki = shfl_idx(my_idx, i)               # row i's index: broadcast, no register batch
    cute.copy(tiled_kv,
              thr.partition_S(mKV[(ki, slab, None)]),
              thr.partition_D(sKV[(i * ROW_TEAMS + team, None, st)]))
```

If the DSL side has an idx broadcast such as `cute.arch.shuffle_sync`, it can be used directly; the
inline PTX above is used because the cutlass-dsl 4.6.1 this entry was certified with only exports
`shuffle_sync_bfly`.

## Machine-read footprint (acceptance criteria)

1. **SASS counts**: `SHFL.IDX` == rows per stage × number of stages (here 16×4 = 64); producer-segment
   `LDG.E` drops from one per row to one per warp (68 → 8); `LDL`/`STL` drop (20 → 12). `HGMMA` and
   `LDGSTS` unchanged (producer-only change).
2. **Same-job A/B timing**: device ms must drop significantly against the uniform-batch version
   (measured −5.4%, three-round jitter < 0.03%).
3. **Role exposure ledger**: copy-producer exposed ns drops (357 → 305 µs), the `long_scoreboard` share
   drops (60 → 55%); spin exposure drops with it (1.124 → 0.996 ms).
4. **A changed reading on the supply axis is normal**: addr-source loads 87 → 18 (the spill refills
   are gone), the shape changes from `interleaved` to `clustered-at-top`, tight-chained 7 → 11.
   **These two readings are not a problem in this form**: the address chain is now `SHFL → LDGSTS`
   back to back, and the tool counts the copy after the SHFL as tight-chained; but SHFL is a register
   operation (a few cycles), not a gmem round trip. Judge by timing and LDL/STL, not by this axis.

## Gotchas

1. **Raising the uniform batch depth to get prefetch**: the cubin does not change (measured: batch
   depth 4 and 16 have the same sha). ptxas breaks the uniform batch back into one load next to each
   consumer; the depth parameter has no semantics. To prefetch, change how the indices are held, not
   the depth.
2. **Copying the reference's smem staging**: the reference (LDG→STS→LDS) is right in its own kernel,
   but transplanted onto this tree it is **4% slower**: one extra LDS before every copy, and the
   producer's `short_scoreboard` share rises from 0 to 38%. The reference can afford it because
   something else in its producer covers the LDS latency; "the reference writes it this way" in the
   dossier does not mean "you should write it this way too" — a form has to be priced (see `[PRICED:]`).
3. **Treating the supply axis changing from `interleaved` to `clustered-at-top` as a regression**:
   see footprint 4.
4. **For N > 32 rows, letting only the first N%32 lanes hold indices**: each lane holds ⌈N/32⌉, and
   the broadcast is `shfl_idx(my_idx[i // 32], i % 32)`.
5. Ceiling: this entry removes the producer's **index** round trip, but 1.0 ms of spin remains — that
   is the round trip of the gather data itself (long_scoreboard of the 64 LDGSTS), which needs a
   deeper ring / earlier stage issue, not this entry.

## Sources

- ✅ Verified: certified by a GPU probe (the probe is not included in this release; based on a
  gather-attention tree that had passed first-green)
  (H100, cutlass-dsl 4.6.1 image): four variants × three rounds,
  `PROBE PASS lane delta_vs_base=5.56% gates=ok`, `PROBE PASS smem delta_vs_base=-3.92%
  gates=ok`, `depth16 gates=cubin_same_as_base` (which is exactly gotcha 1's conclusion). All four
  versions `ROLE-OK … tensor=` + `PMTL-OK`.
- Price tags: `[PRICED: +5.6% ratio / −5.4% ms (lane vs uniform batch)]`;
  `[PRICED: smem staging −3.9% (slower)]`; `[PRICED: uniform batch depth 4→16 = 0 (cubin identical)]`.
- Public interfaces: PTX `shfl.sync.idx.b32`; `cutlass._mlir.dialects.llvm.inline_asm` (see
  [`ptx_escape`](ptx_escape.md)); `cutlass.cute.arch.make_warp_uniform` (this entry does **not** depend
  on it).
