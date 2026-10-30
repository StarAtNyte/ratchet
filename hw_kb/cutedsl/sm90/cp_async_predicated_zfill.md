# cp_async_predicated_zfill (src-size zero fill of `cp.async` + a runtime-truncated trip count)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry is the **predication** patch to [`cp_async`](cp_async.md) (issue/partitioning/group completion of a
> row-wise gather): that one assumes "every row is a valid row"; this one handles "the row set mixes in invalid
> rows, and the number of valid rows is a runtime value". The calling convention (`@dsl_user_op` / `AD_ATT` / `$n`
> / constraint string) follows [`ptx_escape`](ptx_escape.md).
> **Spelling ruling**: the load-bearing form of this entry is the inline-PTX src-size register form of §1 (tier
> B), **and the destination address must be written as "one swizzled base address per thread + PTX immediate
> offsets"** (§2 / gotcha 8). The DSL's own `cute.copy(..., pred=)` (tier A) also zero-fills numerically, but
> under the producer budget reduced by `setmaxnreg` it is a **spill cliff** (gotcha 15); the cause of the cliff is
> not the instruction spelling but **each copy materialising its own destination address** — a spelling that
> only switches to asm while still computing addresses one by one (the B0 control of gotcha 15) spills exactly
> as much as tier A. Before using tier A, read "Expressibility ladder" and "Self-check: locating spills".

## Purpose

The PTX form of `cp.async` has a fourth operand, **`src-size`**:

```
cp.async.cg.shared.global [dst], [src], cp-size, src-size;
```

The hardware reads `src-size` bytes from `src`, and the remaining `cp-size - src-size` bytes of the destination
are **written as 0 by the hardware**. `src-size` can be a **register**, so "is this row valid" can be decided at
runtime: valid → 16, invalid → 0. An invalid row thus gets 16 B of all zeros, **produces no source read** (no
sector request for that row on the L2 side), and still goes through the same issue stream (the same commit group /
the same `cp.async.mbarrier.arrive`). The second mechanism that goes with it is writing the trip count of the issue
loop as a **runtime value** (`cutlass.range(n_trips)`, with `n_trips` derived from the number of valid rows), so
the whole invalid tail is never issued at all. Together they replace the form "clamp invalid rows to some legal
row + mask to zero again on the consumer side".

## When needed (trigger)

- The row set carries **invalid markers** (sentinel values in the index array, such as `-1`; or a "valid row
  count" scalar smaller than the capacity), and the current practice is to clamp invalid rows to some legal row,
  move it anyway, and have the consumer side wipe out its contribution with a flag/mask.
- Symptom 1 (**sensitive to the operating point**): when the number of valid rows shrinks (a high share of
  sentinels, a valid length only half the capacity), the time **does not drop accordingly**, and may even read ~2×
  the baseline built with all rows valid — because the trip count is Constexpr, the invalid rows' `LDGSTS` are
  issued anyway and occupy L2 and the long scoreboard just the same; on an "all valid" construction this is
  completely invisible.
- Symptom 2: on the profiler, the supply hotspot is amplified by invalid rows (`stall_long_sb` up, `tensor_pct`
  down), while `dram_bytes_read` is unchanged (the clamp target row is in L2; it is a latency ledger, not a
  bandwidth ledger, the same analysis as the cache-qualifier section of [`cp_async`](cp_async.md)).
- Symptom 3: a clamp target row was specially chosen so that "invalid-row data is finite and produces no NaN" —
  the zero fill directly gives the finite value 0, and this concern disappears.
- Symptom 4 (**already zero-filling, but with tier A `pred=`**): the producer role shows local memory accesses after
  `setmaxnreg.dec` — `STACK` > 0 in `cuobjdump -res-usage`, `LDL`/`STL` > 0 in SASS, non-zero NCU
  `spill_ld/spill_st` sectors, `stall_long_sb` raised while `dram_bytes` is unchanged; when attributed by source
  line, the hottest `STL` lands on the `cute.copy(..., pred=)` line in the fill loop, and the `LDL` family is
  dominated by long-scoreboard. This is the signature of gotcha 15: switch to the tier-B "base + immediate" spelling
  of §1 + §2, rather than touching the register split; and don't just replace `cute.copy` with asm while keeping
  per-copy address computation — that step leaves the spill unchanged (the B0 reading of gotcha 15).
- Reverse exclusion:
  - Regular row sets (tile / stride / sliding window) → out-of-bounds zero fill goes through TMA's descriptor
    boundary check ([`tma_ndim_oob_zerofill`](tma_ndim_oob_zerofill.md)), not this entry.
  - Invalid rows **need not** occupy space in smem (the consumer side loops by row count and never touches them)
    → only the "runtime trip count" half of this entry is needed; the zero-fill half can be dropped.
  - What you need is "skip, don't write" rather than "write 0" → that is the predicated-skip form of
    `cute.copy(..., pred=)`, see the tier-A ruling ("Expressibility ladder" below).

## Minimal runnable example

The geometry is only a size label (the same formulas as [`cp_async`](cp_async.md)): source `(256, 512)` bf16
row-major, destination smem `(32, 512)`; 128 threads → `THREADS_PER_ROW = 64`, `ROW_TEAMS = 2`; `n_valid` (a
runtime Int32) rows are valid, and `-1` in the index array marks an invalid row.

### 1) Instruction: src-size as a register operand (`ptx_escape` form 2)

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def cp_async_16_zfill(dst_smem_addr, src_gmem_addr, src_bytes, *, loc=None, ip=None):
    """dst: Int32 (shared::cta window address); src: Int64 (global);
    src_bytes: Int32 -- 16 = copy, 0 = destination 16 B all zeros, source not read."""
    llvm.inline_asm(
        None,
        [dst_smem_addr.ir_value(loc=loc, ip=ip),
         src_gmem_addr.ir_value(loc=loc, ip=ip),
         src_bytes.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",                                   # order = $0,$1,$2; types per ptx_escape gotcha 3
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

A producer with a reduced budget uses its **immediate-offset** variant — the same instruction, with both addresses
written as `[$base + IMM]`, where `IMM` is a Python int baked directly into the PTX string (the same technique as
`int(n)` in `cp.async.wait_group %d` of [`ptx_escape`](ptx_escape.md)):

```python
@dsl_user_op
def cp_async_16_zfill_imm(dst_base, src_base, src_bytes, dst_off, src_off, *, loc=None, ip=None):
    """dst_base: Int32 shared window address (one per thread, already swizzled); src_base: Int64 (one per row);
    dst_off / src_off: Python int immediates. dst_off must be a multiple of 1024 (see §2)."""
    dst_off, src_off = int(dst_off), int(src_off)
    llvm.inline_asm(
        None,
        [dst_base.ir_value(loc=loc, ip=ip),
         src_base.ir_value(loc=loc, ip=ip),
         src_bytes.ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0+%d], [$1+%d], 16, $2;" % (dst_off, src_off),
        "r,l,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

No copy has any register "of its own" any more: the only live values are one smem base per thread, one Int64 source
address per row and one src-size per row. `[reg+imm]` is a legal form of a PTX address operand, which ptxas
compiles directly into `LDGSTS [R+imm]`.

`commit_group` / `wait_group` are exactly the same as [`cp_async`](cp_async.md) §3; when switching to the
arrive-on-completion of [`cp_async_mbarrier_ring`](cp_async_mbarrier_ring.md), a zero-filled copy also counts as
one landing.

### 2) Issue loop: runtime trip count + one src-size per row

```python
g2s  = cute.make_tiled_copy_tv(                       # partitioning is still done by the DSL; only the instruction changes
    cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                        cutlass.BFloat16, num_bits_per_copy=128),
    cute.make_layout(THREADS_PER_ROW), cute.make_layout(ELEMS_PER_THREAD))
thr  = g2s.get_slice(tidx % THREADS_PER_ROW)
team = tidx // THREADS_PER_ROW

n_trips = (n_valid + ROW_TEAMS - 1) // ROW_TEAMS      # runtime value
for t in cutlass.range(n_trips):                      # scf.for, not range_constexpr
    r  = t * ROW_TEAMS + team                         # destination row (dynamic)
    j  = mIdx[r]                                      # source row index (dynamic, -1 = invalid)
    ok = (j >= 0) & (r < n_valid)
    row    = cutlass.Int32(0)                         # clamp to any legal row: the address must be legal, but it is not read
    nbytes = cutlass.Int32(0)                         # src-size 0 -> 16 B of zeros
    if ok:                                            # runtime if (written directly in the kernel body)
        row    = j
        nbytes = cutlass.Int32(16)
    src = thr.partition_S(mSrc[(row, None)]).iterator.toint()   # Int64
    dst = thr.partition_D(sBuf[(r,   None)]).iterator.toint()   # Int32 (flat row-major)
    cp_async_16_zfill(dst, src, nbytes)

cp_async_commit_group()
cp_async_wait_group(0)
cute.arch.fence_view_async_shared()
cute.arch.sync_threads()
```

Result contract (exactly these classes are checked row by row by the certification probe):

| Row | smem content |
|---|---|
| `r < n_valid` and `mIdx[r] >= 0` | `src[mIdx[r]]` |
| issued, but `mIdx[r] < 0` or `r >= n_valid` (the tail of the last trip) | **all 0** |
| `r >= n_trips * ROW_TEAMS` (not issued) | **untouched** (the probe pre-fills sentinels to prove it was not touched) |

### 2b) Address discipline for a reduced-budget producer: base computed once, all offsets immediates

The flat form of §2 computes one `dst` per copy — in a 128-thread producer with `setmaxnreg.dec 72`, if there is
also a runtime body loop outside the issue loop (the same destination rows refilled once per body), these
destination addresses are **invariant across the body loop**, and the compiler hoists them all out of the loop and
keeps them live: 64 copies per thread per body = 64 live registers, which a budget of 72 cannot hold, so it spills
straight away (gotcha 15's measurement: tier A and the per-copy-address B0 were both `STACK 96`, `LDL 24 / STL 24`,
with every `STL` on an address line). The certified form:

```python
# Inside the role branch, outside the body loop: one base address per smem slot. Row = this thread's team row, box 0, this
# lane's 16 B chunk, addressed through the "unswizzled linear view", then the 128 B swizzle applied once.
la = thr.partition_D(sA_lin[(rteam, None, 0)]).iterator.toint()
dst0a = la ^ ((la >> 3) & 0x70)                     # bits[4,7) ^= bits[7,10)
# The two sets of immediates have different strides and cannot share one constant:
#   the destination tile is (BN, DBOX, NBOX), boxes a whole BN×DBOX slot apart -> SLOT_BYTES = BN * DBOX * 2 = 8192
#   the source is consecutive boxes within one row                             -> BOX_BYTES  = DBOX * 2      = 128
#   row stride ROW_BYTES = DBOX * 2 = 128; two adjacent rows of this thread are ROW_TEAMS rows apart

for body in cutlass.range(n_body):                  # runtime body count
    ...                                             # index / src-size decision as in §2
    for j in cutlass.range_constexpr(ROWS_PT):      # this thread's j-th row: drow = rteam + ROW_TEAMS*j
        src_row = thr.partition_S(mSrc[(row_j, 0, None)]).iterator.toint()   # one Int64 per row
        for box in cutlass.range_constexpr(NBOX):
            cp_async_16_zfill_imm(dst0a, src_row, nbytes_j,
                                  j * ROW_TEAMS * ROW_BYTES + box * SLOT_BYTES,  # destination: j*2048 + box*8192
                                  box * BOX_BYTES)                               # source:      box*128
```

Why immediates can bypass the swizzle: the 128 B swizzle only touches bits[0,10) of the byte address (both fields of
the XOR are inside it); the destination addresses of this thread's other (j, box) differ from the base only by
`j * ROW_TEAMS * 128 B` (`ROW_TEAMS = 16` → multiples of 2 KB) and `box * 8 KB`, all **multiples of 1 KB**, and adding
them does not change bits[0,10), so `swizzle(base + off) == swizzle(base) + off`. The condition is that `ROW_TEAMS` is
a multiple of 8 (each of the thread's rows sits at the same position within an 8-row swizzle group); otherwise the
row offset steps into bits[7,10), the immediate form does not hold, and you fall back to one base per (j) (still
`ROWS_PT` registers rather than 64).

### 3) Three-in-one criteria (the mechanism really works, rather than the values happening to be right)

1. **Values**: fill the clamp target rows of invalid rows with a "poison value" (non-zero, non-sentinel) — the zero
   fill is correct iff what is read back is 0, not the poison value (poison = the source was read = degenerated into
   a plain cp.async) and not the sentinel (sentinel = not issued = predicated skip).
2. **Footprint (differential, classified by family)**: in `cuobjdump -sass` the `LDGSTS` count > 0, and every
   `LDGSTS` has, besides the two **bracketed address operands**, one more **bare operand** — a runtime predicate
   (`P#` / `!P#`) or a register (`R#`) — which is the encoding of src-size going through a register/predicate; a
   control group of the same kernel with src-size replaced by the constant 16 (a plain `cp.async`) has only the two
   bracketed operands. **Don't match the mnemonic literally on `ZFILL`**: the current toolchain emits
   `LDGSTS.E.BYPASS.128` (no `ZFILL` suffix) + a bare predicate operand, and a literal match reports 0 even with all
   values correct (another instance of [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 1).
   **Forms that do not count as src-size**: a leading `@P#` guard predicate (that is "not issued", not zero fill),
   a bare `PT` (constant true = folded into 16). For how to get the cubin see [`cp_async`](cp_async.md) gotcha 1
   (env-var + `options=` double sink). The control group must also **read the poison value** — proving that criterion
   1's poison channel can really see a degenerated copy, rather than reading 0 only because the clamp row happened
   not to be read.
3. **No source read (differential)**: compare NCU `lts__t_sectors_srcunit_tex_op_read.sum` over **four** launches —
   the kernel under test (src-size register) and the constant-16 control kernel, each run once with "all rows valid"
   and once with "a quarter of the rows valid" (invalid destination row r clamped to **that CTA's own r-th poison
   row** — one private clamp row per (CTA, row), never shared), all four launches in the same profiler window with
   exactly the same address stream. The criteria are all **ratios**, with no absolute sector gate:
   - `zfill_q / ctrl_q ≤ 0.5`: same address stream, differing only in src-size — this is the direct evidence that
     "src-size=0 issues no L2 request"; if the hardware still read the source, the two would be equal and the ratio
     ≈ 1 (the clamp row is an L2 **hit**, but hits are counted as read sectors too).
   - `zfill_q / zfill_all ≤ 0.5`: the same in the form of two launches of the same kernel.
   - `zfill_all / ctrl_all ∈ [0.8, 1.25]`: calibration — with src-size all 16 the two kernels read the same bytes;
     otherwise the two cubins are not "different only in src-size".
   - `ctrl_q / ctrl_all ∈ [0.8, 1.25]`: **control integrity** — the constant-16 kernel reads the same volume whatever
     the index set (each sector requested exactly once). Failing this = the control group's clamp rows were merged by
     L1TEX (the second form of gotcha 14), in which case the denominator of `zfill_q / ctrl_q` is distorted and the
     mechanism criterion does not hold.
   - The control group's quarter launch must read the poison value (its address stream really lands on the clamp
     rows), and the tested group's quarter launch must have all values correct.
   `dram_bytes_read` is not a proxy for this criterion (hits are not counted in DRAM). For the criterion's **geometry
   discipline** see gotcha 14. Expected reading (kept as an INFO line): `zfill_q` = the valid-row payload + the sectors
   of the index array itself; a single extra sector means invalid rows issued requests.

## Gotchas

1. **`src-size` must be ≤ `cp-size`, and `cp-size` is an immediate.** This entry certifies only the two settings
   `16 / 0`; intermediate values (8, 4) are also legal per the PTX ISA (the first `src-size` bytes copied, the rest
   zero-filled) but are uncertified.
2. **The address must still be a legal address.** With src-size = 0 the hardware does not read the source, but the
   operand must still be a well-formed global address — clamping to row 0 (or any resident row) is enough; it only
   carries "a legal address", not "data"; don't pass a wild pointer because "it won't be read anyway".
3. **Zero fill ≠ predicated skip.** The predicated form of `cute.copy(..., pred=)` in C++ CuTe semantics **does not
   issue** (the destination is untouched); only the src-size / `ignore-src` form writes 0. For "invalid rows are 0 in
   smem" (no masking on the consumer side), use this entry; for "invalid rows don't touch smem", use the predicated
   skip. The difference in what is read back is exactly "sentinel vs 0" of §3 criterion 1. Which of the two the DSL's
   `pred=` actually lowers to for `CopyG2SOp` is in the tier-A ruling of "Expressibility ladder" below.
4. **When the trip count is written as a runtime value, don't use `range_constexpr`.** `cutlass.range(n_trips)`
   accepts a dynamic Int32 (traced as `scf.for`); `range_constexpr` only takes a Python int. Cost: the loop body is no
   longer fully unrolled, and address hoisting and DEPBAR tiers differ from the Constexpr version — establish the
   mechanism per this entry first; unrolling/pipeline depth belong to the levers of
   [`guard_predicate_issue_distance`](guard_predicate_issue_distance.md) /
   [`midchain_stage_wait`](midchain_stage_wait.md).
5. **Variables assigned inside a dynamic `if` must be declared outside first** (`row = cutlass.Int32(0)`, then
   `if ok: row = j`), otherwise the SSA join has no value for the else branch. The same applies to `nbytes`.
6. **src-size is a per-thread operand, but the decision must be uniform per row.** The 64 lanes of the same row
   must compute the same `ok` — base the decision on row-level quantities (index, row number, `n_valid`) and don't
   mix in lane-level quantities; otherwise one row is half 0 and half data, which numerically looks like the "wrong
   in per-thread blocks" of [`cp_async`](cp_async.md) gotcha 5.
7. **Unissued tail rows are dirty.** Rows with `r >= n_trips * ROW_TEAMS` hold the previous content in smem; the
   consumer side must either truncate by the same `n_trips`, or pad the capacity to a multiple of `ROW_TEAMS` so the
   last trip's zero fill covers them. The probe pre-fills sentinels precisely to tell "not written" from "written 0".
8. **A swizzled destination buffer needs the swizzle applied by hand** ([`ptx_escape`](ptx_escape.md) gotcha 5): §2
   of this entry takes `dst` with `iterator.toint()` of a flat row-major view. When the destination is a 128 B
   swizzle layout (a ComposedLayout of the `tile_to_shape(make_smem_layout_atom(get_smem_layout_atom(...)), ...)`
   kind), the **certified form** is: take the address through an **unswizzled linear view** of the same bytes
   (`storage.sX.get_tensor(layout.outer)`, without passing `swizzle=`), then apply
   `dst = dl ^ ((dl >> 3) & 0x70)` (byte address bits[4,7) ^= bits[7,10)) in DSL integer arithmetic and hand it to the
   asm; the consumer side reads as usual through the view with `swizzle=layout.inner`. `iterator.toint()` directly on
   a swizzled view gives the linear address, and the data lands in the wrong bank rows. **Do this XOR only once per
   thread** (§2b: base + immediates); don't redo it on every copy — destination addresses XORed per copy cannot be
   merged by the compiler with the following addition, and are hoisted out of the loop as 64 independent loop
   invariants and all kept live (the B0 reading of gotcha 15). Both forms, flat destination and 128 B swizzle
   destination, are GPU-certified (by a zero-fill probe and a register-cost probe respectively).
9. **The cache qualifier is independent of src-size.** The example writes `.cg` (SASS `BYPASS`) following the
   load-bearing condition of [`cp_async`](cp_async.md); switching to `.ca` or adding `.L2::256B` does not affect the
   zero-fill semantics.
10. **`has_side_effects=True` and 16 B alignment on both sides**: same as [`cp_async`](cp_async.md) gotchas 2/4;
    missing the former gets the whole asm DCE'd and reads back all sentinels; unmet alignment is a runtime address
    error.
11. **Certification-script discipline**: the parent process uses only the standard library, DSL code goes in child
    processes/submodules, and `@cute.kernel` and `@cute.jit` are defined at module top level — the clauses of
    [`cp_async`](cp_async.md) gotchas 7b/12 are applied as-is in this entry's probe (the DSL source embedded as a
    string and written out as a separate module). **But not all-or-nothing**: only when the first rung produced
    nothing should "no verdict line" be printed; when later rungs die on instruments (dump directory, ncu slot),
    contracts already completed print `PROBE PASS/FAIL` as usual, and dead contracts print
    `PROBE FAIL max_diff=n/a <contract> instrument: ...` — otherwise one ncu slot error swallows both the numeric
    and the footprint contracts that already passed.
12. **For an NCU comparison, move both tensor allocation and read-back outside the profiler window.**
    `--profile-from-start off` + `--launch-count N` counts **any** kernel inside the window: if the window also
    contains a `torch.full` / `.to(bfloat16)`, or even a `.float()` read-back (also an elementwise kernel), PyTorch
    kernels get mixed into the first N launches and the last launch under test is never sampled. Practice: each
    launch uses its own output tensor, and all read-backs go after `profiler.stop()`; add `-k regex:<kernel name>`
    to ncu so stray kernels cannot take slots; the criterion requires the kernel names of all launches to be the same
    and not a torch kernel name, and all launch names are printed to the log.
13. **The "bare operand" family classification of criterion 2 must be validated on the control group in the same
    round.** Looking only at the 3-operand form of the cubin under test is not enough — compile the constant-src-size
    control kernel in the same round and confirm it falls back to the 2-operand form, so that the classifier has
    discriminating power ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 3: count the target form
    and the failure form separately).
14. **NCU read sectors must not have an absolute gate; each (CTA, row) must read its own source row — including
    clamp rows.** Concurrently in-flight `LDGSTS.BYPASS` requests with the same address are merged **before**
    reaching L2 (on the L1TEX side), and L2 read sectors count only one copy. Both forms have been measured:
    - **Shared across CTAs**: when 256 CTAs all read the same 32 KB source, `lts__t_sectors_srcunit_tex_op_read.sum`
      is only about 1/4 of the payload (256 × 32 KB / 32 B = 262144) (measured 68938).
    - **Clamp row shared within a CTA**: when the control kernel (constant 16) clamps all 24 invalid rows of one CTA
      to the same poison row, only about 28% of these 24 repeated reads reach L2 (quarter-valid launch measured
      121240, instead of the 263168 that "every row reads" would give) — so `zfill_q / ctrl_q` comes out at 0.549,
      and the kernel under test is judged FAIL although it issued not a single extra sector (`zfill_q` = valid
      payload 65536 + indices 1024 = 66560, exactly).
    Meanwhile, on kernels whose (CTA, row) addresses do not overlap, the same metric matches
    `sm__sass_data_bytes_mem_global_op_ldgsts` at 32 B/sector. When writing the criterion: (a) block the source by
    CTA, each CTA getting a run of data rows **plus a run of poison rows as long as the capacity**, with invalid
    destination row r clamped to its own r-th poison row, so valid/clamp addresses never overlap anywhere in the
    grid; (b) use as the denominator the reading of the **constant-16 control kernel on the same address stream**,
    not a computed payload, and add `ctrl_q / ctrl_all ∈ [0.8, 1.25]` to prove the control group was not merged;
    (c) print absolute values only as INFO lines for the record. A side observation on the clamp form itself: the L2
    traffic of a plain `cp.async` clamped to a single row is not "one copy per row" but partially merged in L1TEX —
    so its cost is a long-scoreboard latency ledger, and `dram_bytes_read` / L2 sectors both underestimate it
    (consistent with symptom 2 of "When needed").

15. **Tier A `pred=` is a spill cliff under the producer budget reduced by `setmaxnreg` — this entry prices it, and
    makes "tier B + base/immediate addressing" the mandatory spelling. The cause of the cliff is the
    materialisation of destination addresses, not the instruction spelling.** For the same producer geometry (128
    threads `setmaxnreg.dec 72`, a runtime body loop, 8 row indices prefetched per body, 4 delivery phases, 16 16 B
    `cp.async` per thread per phase, a 128 B swizzle destination ring + full/empty mbarriers), the three spellings
    differ only in the copy lines:
    | Spelling | Destination address | Reading (`cuobjdump -res-usage` / SASS) |
    |---|---|---|
    | tier A `cute.copy(..., pred=)` | computed per copy by the DSL + swizzle applied | `STACK 96`, `LDL 24 / STL 24` |
    | B0: asm src-size, per-copy `iterator.toint()` + per-copy XOR | computed per copy | **the same** `STACK 96`, `LDL 24 / STL 24` |
    | B1: asm src-size, one swizzled base per slot + PTX immediates (§2b) | once | `LDL + STL == 0` (certification contract) |
    All three have `LDGSTS` as `LDGSTS.E.BYPASS.128` + a bare predicate operand (the same family as §3 criterion 2) —
    the price difference is not in the instruction.
    Per-line attribution (`nvdisasm --print-line-info`) puts every `STL` of A/B0 on the **address lines** (B0's
    hand-written XOR line and `partition_D` line; tier A's `cute.copy` line), and the `LDL`s on the copy and arrive
    lines: the destination addresses are invariant across the body loop, the compiler hoists all 64 out of the loop
    and keeps them resident in registers, and 72 cannot hold them. Rules:
    - **A gather in a producer under a reduced budget is always written as B1** (§1's `cp_async_16_zfill_imm` + §2b
      addressing); tier A is only allowed in roles without `setmaxnreg.dec`, or, if used there, must first pass
      "Self-check: locating spills" with 0.
    - **Just replacing `cute.copy` with asm is not a fix**: computing addresses per copy is still B0, reading the same
      as tier A. What gets fixed is the addressing.
    - **When the spill is on the producer side, do not shrink the producer's register split** — that is the wrong
      direction; nor touch the consumer segment's live-set ledger first. First attribute `LDL/STL` to source lines per
      "Self-check: locating spills"; if the lines are the copy / address lines of the fill loop, it is this entry, and
      changing the addressing is enough.
    - Certification readings (GPU probe, not included in this release; verdicts in "Sources").
16. **The `setmaxnreg` four-piece set comes before any spill reading.** If any of `.reqntid` + `.minnctapersm` +
    `dec` + `inc` is missing, the producer was not compiled under 72 at all, and `LDL/STL == 0` proves nothing
    ([`uniform_reg_budget`](uniform_reg_budget.md) self-check 0, [`warp_specialization`](warp_specialization.md)
    gotcha 1). This entry's probe writes the four-piece set into the cliff contract as a precondition, and both
    cubins must have all four.

## Self-check: locating spills (before any register-ladder step)

A spill reading can only identify a spelling once it lands on a **source line**. Three steps, all machine-read, all
on the dumped cubin (dump method: the double sink of [`cp_async`](cp_async.md) gotcha 1):

```bash
export CUTE_DSL_LINEINFO=1          # set before compiling; line numbers go into the cubin, zero performance cost
export CUTE_DSL_CACHE_DIR=/tmp/dslcache_li   # a fresh cache directory: cache hit = no compile = no dump
cuobjdump -res-usage <dump>.cubin | grep -E 'REG|STACK|SPILL'   # STACK > 0 means there is a local frame
nvdisasm -c --print-line-info <dump>.cubin \
  | awk '/\/\/## File/{loc=$0} /[^A-Za-z0-9_](LDL|STL)[^A-Za-z0-9_]/{print loc}' | sort | uniq -c | sort -rn | head
```

Word boundaries can only be written with POSIX character classes: `\<` `\>` are GNU awk extensions, and on mawk (the
default awk of the job image) and macOS awk they **silently match nothing** — a machine-read self-check reporting
"no spill" when in fact the regex never took effect is exactly the kind of mistake this section guards against.
First verify the pipeline produces output on a dump known to contain `LDL`, then trust its 0.

How to read it:
- The hottest `STL`/`LDL` lines land on **the copy line or address line inside the fill loop** (tier A
  `cute.copy(..., pred=)`; or `iterator.toint()` / the hand-written XOR line in the asm form) → gotcha 15; switch to
  §2b's "base once + immediates" addressing, then rerun this self-check and require `LDL + STL == 0`.
- They land in the shared prologue before the role branch → [`warp_specialization`](warp_specialization.md) gotcha
  11 (the prologue crosses the dec point).
- They land inside a consumer segment → only then is it the turn of the live-set ledger / split ladder.
- **Ordering discipline**: before any ladder step that "changes the producer/consumer split" or "changes the live set
  within a segment", do these three steps first and write the attributed lines into the build record; a spill fix
  without line attribution is guesswork.

## Expressibility ladder (the A/B/C of [`ptx_escape`](ptx_escape.md))

- **A (DSL knob)**: `cute.copy` has a `pred=` parameter (the varlen utilities of public infra spell it
  `cute.copy(atom, src, dst, pred=<Boolean rmem tensor>)`, with the predicate tensor built over the partition's rest
  dimension and `cutlass.Boolean` elements). Observed on the GPU: a predicate tensor of shape `(1,)` compiles, and for
  `CopyG2SOp` lowers to **zero fill** (invalid rows read back 0, not the sentinel), with SASS of the same shape as
  tier B (`LDGSTS.E.BYPASS.128` + a bare predicate operand); the signature of `CopyG2SOp.__init__` has **no**
  src-size / zfill knob (only `cache_mode`). **Pricing**: in a producer after `setmaxnreg.dec`, tier A is a spill cliff
  (gotcha 15: under the same geometry tier A has `LDL + STL > 0`, B1 `== 0`). The root cause is that tier A has no
  means of expressing "base + immediate" — the DSL computes one swizzled destination address per copy, and the
  compiler keeps them resident as 64 loop invariants. So the ruling on tier A is "semantically usable, unusable under
  budget" — allowed only in roles without a reduced budget, and only after passing "Self-check: locating spills".
- **B (inline PTX)**: §1 of this entry, src-size through an explicit Int32 register (16 / 0; other settings
  expressible). Two addressing variants: B0 (each copy addressed through the linear view of §2 / gotcha 8 + a per-copy
  XOR) and B1 (§2b: one swizzled base per slot, everything else PTX immediates `[$base+IMM]`). The footprint of both is
  `LDGSTS` + a bare src-size operand (§3 criterion 2). **B1 is the certified form and the mandatory spelling for a
  reduced-budget producer; B0 under budget spills just like tier A (gotcha 15 table)**, and is only usable in simple
  forms with no outer body loop where destination addresses do not form loop invariants (§2's one-shot issue loop).
- C does not apply.

## Sources

- PTX ISA `cp.async`: the `src-size` operand ("copy src-size bytes, zero-fill the rest of the destination") and the
  `ignore-src` predicate form
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async>; the
  group completion semantics of `cp.async.commit_group` / `cp.async.wait_group`.
- Public infra API symbols: CUTLASS `cutlass::arch::cp_async_zfill<SizeInBytes, CacheOperation>`
  (`cutlass/arch/memory_sm80.h`, the same form `src_in_bytes = pred ? SizeInBytes : 0`) and CuTe's
  `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` copy trait (`cute/arch/copy_sm80.hpp`); CuTe DSL `cute.copy(..., pred=)`,
  `cutlass.Boolean`, `cute.make_rmem_tensor`, `cutlass.range` (dynamic trip), `cpasync.CopyG2SOp(cache_mode=...)`,
  `cutlass._mlir.dialects.llvm.inline_asm` + `cutlass.cutlass_dsl.dsl_user_op`.
- Certified recipes of the same family: [`cp_async`](cp_async.md) (partitioning / group completion / cubin double
  sink / certification discipline), [`ptx_escape`](ptx_escape.md) (calling convention, ladder, swizzle addresses),
  [`cp_async_mbarrier_ring`](cp_async_mbarrier_ring.md) (arrive-on-completion),
  [`tma_ndim_oob_zerofill`](tma_ndim_oob_zerofill.md) (hardware zero fill of regular blocks, this entry's reverse
  exclusion), [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) (footprint criteria classified by family,
  target/failure forms counted separately).
- GPU-certified (zfill probe) — three contracts `zfill_numeric` (poison clamp rows +
  sentinel pre-fill + runtime `n_valid=21` over a 32-row capacity: valid rows, zero-filled rows, the tail rows of the
  last trip, unissued rows — four classes checked row by row) / `zfill_footprint` (differential SASS census: every
  `LDGSTS` of the cubin under test carries a bare predicate/register operand, the constant-16 control cubin has only
  the two address operands, the control group's values read the poison) / `zfill_no_src_read` (differential ratios
  of NCU L2 read sectors over four launches: tested / control × all valid / quarter valid, the source blocked by CTA,
  one private clamp row per (CTA, row), the control group's volume independent of the index set), plus an `EXPR A=`
  verdict line. GPU verdict of round 6: `PROBE PASS max_diff=0.0` (all three contracts PASS).
- GPU-certified (register pricing, gotchas 15/16 + the swizzled destination addressing of §2b / gotcha 8 +
  "Self-check: locating spills") — four contracts
  `zfill_regcost_numeric_asm_imm` (B1 on a 72/216 split, 8 runtime bodies, a 128 B swizzle two-slot ring, equal row by
  row to a numpy gather: valid rows / both sentinel families / prefix-truncated rows all 0 without reading the poison
  rows / no unwritten rows) / `zfill_regcost_numeric_asm` (B0 control, same contract) / `zfill_regcost_numeric_dsl`
  (tier A, same contract) / `zfill_regcost_cliff` (all three cubins have the `setmaxnreg` four-piece set + every
  `LDGSTS` carries a bare src-size operand + **B1 `LDL+STL == 0` and tier A `LDL+STL > 0` holding at the same time in
  the same round**, with B0's `LDL+STL` printed as a control reading on the same line), with INFO lines carrying the
  REG/STACK of `cuobjdump -res-usage` and the per-source-line LDL/STL attribution of `nvdisasm --print-line-info`.
  GPU verdict of round 1 (only tiers A and B0): both tiers' values all PASS, cliff FAIL — the two cubins read exactly
  the same (`REG 168 / STACK 96 / LDL 24 / STL 24`, all 64 `LDGSTS` with a bare predicate operand), with all
  attributed lines on address lines; this is the measured source of gotcha 15's "the cause is in the addressing, not
  the spelling". GPU verdict of round 2 (B1 added): `PROBE PASS max_diff=0.0` — all four contracts
  PASS: B1's values equal the numpy gather row by row, and the cliff contract holds (B1 `LDL+STL == 0`, tier A `> 0`,
  the four-piece set present, every `LDGSTS` carrying a bare src-size operand). From then on this entry is **✅
  Verified** (both probes PASS).
