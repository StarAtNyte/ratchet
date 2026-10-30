# gather_index_prefetch_predicated (two-phase issue for a predicated gather: index phase "clamp + unconditional load + live flag" → copy phase "back-to-back cp.async by live / whole-row 16 B vector zero fill")

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **predicated spelling in CuTe DSL** of [`gather_index_prefetch`](gather_index_prefetch.md)
> (the **principle** of batched index prefetch + cross-row decoupling): the row set mixes in invalid rows
> (sentinel indices / beyond a runtime valid length), and invalid rows must be 0 in smem. It answers two
> questions that entry leaves open:
> (1) what source shape a "per-row `load → copy` wrapped in an `if`" must be turned into (a two-phase split,
> **effective on its own**: on the production kernel, with the zero fill still scalar, changing only the
> index phase earned +13.84%); and under what condition spilling scatters the clustered loads again (when the
> live set saturates to `STACK ≠ 0` — the spill variant of the certification probe, not the production
> kernel);
> (2) how to write the zero fill of invalid rows so it is `STS.128` rather than 64 `STS.U16` per lane — the
> smem traffic ledger, and under spill conditions a prerequisite of (1).
> It is the sister entry of [`cp_async_predicated_zfill`](cp_async_predicated_zfill.md) (src-size zero fill,
> branch-free): that one folds the "zero fill" into `cp.async` itself; this one keeps the branched form "no
> copy for invalid rows, issue a vector zero store instead". The index phase of the two is **identical row by
> row** (clamping select, unconditional load, Boolean live), but that one is a runtime `scf.for` with one row
> per trip, issuing the copy directly after `j = mIdx[r]` in the same trip, **without this entry's cross-row
> batching**; the combination "batched two-phase indices + src-size zero-fill copy phase" **has not been
> certified by any probe** (see gotchas 7 / 8).

## Purpose

In gather-type supply (`smem[r] = src[idx[r]]`, with `idx` read from gmem at runtime), each thread is
responsible for several rows, each row several 16 B `cp.async`, and each row carries a **row-level
predicate** (valid / invalid). This entry gives the **source structure** that makes ptxas cluster "the index
loads of a batch of rows" at the top of the body and issue the "copies / zero fills" back to back afterwards,
two vector spellings of the zero fill, and one live-set trap that can overturn the whole thing.

## When needed (trigger)

- **Source shape (main criterion)**: one `if slot < n_valid: idx = mIdx[slot]` per row (the load inside the
  `if` region), immediately followed in the same loop iteration by `if live: cute.copy(...)×k`. The 4 rows
  unrolled by `range_constexpr` are 4 stretches of straight-line code, each with its own branch, and each
  stretch's copy addresses depend on the load it just issued.
- **SASS fingerprint (sufficient, not necessary)**: in the kernel text, `LDG.E` (32-bit index loads) and
  `LDGSTS` **interleave row by row** — `LDG · LDGSTS×8 · LDG×3 · … · LDGSTS×8 …`; count with `cuobjdump -sass`
  "the `LDG`s that still appear after the first `LDGSTS`" (`ldg_after`); ≥ 1 is certainly the serial form,
  the target form = 0. **But `ldg_after = 0` does not mean healthy**: in the certification probe the
  non-spilling per-row `if` form (`serial_vec`) had ptxas cluster the 4 `LDG`s at the top of the body by itself
  (`ldg_after = 0`, `REG 207`), yet it was still **7.7%** slower than the two-phase form (80.94 / 75.17 µs,
  `instr` 496 vs 472 — the extra is per-row branch and predicate overhead). In the certification evidence
  `ldg_after ≥ 1` only appears in the spill variant. So when you see `ldg_after = 0`, still check the source
  shape and the instruction count.
- **Spill fingerprint (only when the live set saturates; the cause of `ldg_after ≥ 1`)**:
  `cuobjdump -res-usage` reports `REG:255` with non-zero `STACK`, and SASS has dozens of `STL`/`LDL` between
  the `LDG` and the first `LDGSTS` (`LDG · STL×93 · LDL×8 · LDG · LDGSTS×8 · LDG×3 …`). The production kernel
  (`REG 255`, `spill 0`) does not have this fingerprint; the certification probe's scalar-zero-fill variant
  (zero fill in every block, 46% invalid rows) does.
- **NCU symptom**: `stall_long_sb` many times over the line while `dram_bytes` is within band (the index round
  trip is a latency ledger, not a bandwidth ledger, same as [`gather_index_prefetch`](gather_index_prefetch.md)
  gotcha 2); a rising barrier share is downstream (the consumer waits longer at the handoff barrier), not the
  lesion.
- **Zero-fill symptom**: `smem_wavefronts_st` over the line, `bankconf_st` over the line by orders of
  magnitude, `stall_mio` rising, while the number of bytes written to smem has not changed — per-bf16 scalar
  zero stores are 64 `STS.U16` per lane per row, and ptxas **will not** merge them into vector stores for you.
- **Measured pricing (same gather-attention tree, same-job single variable)**: clustering the index loads at
  the top of the body, on its own, **+13.84%** (independent); the upper bound from ablating the index
  dependency entirely **+13.25%** — the two readings corroborate each other: almost all of the main kernel's
  ~16% excess is this chain. Switching the zero fill to vectors on that tree cost +1.05% time (within noise),
  while `bankconf_st` 10× and `smem_wavefronts_st` 1.59× dropped back into band.

## Mechanism

1. **The loads must land in a stretch of straight-line code with no branches.** With `idx = mIdx[slot]`
   written inside `if slot < n:`, the load sits in a region only that row enters (a DSL runtime `if` = a basic
   block boundary); row i's `LDGSTS` address depends on row i's `LDG` result, and row i+1's `LDG` is queued
   after row i's branch — whether ptxas will hoist it across the branch depends on whether it can if-convert
   the whole stretch and has free registers; in the production kernel with a saturated live set it **did
   not** (the +13.84% above is that bill). Don't leave this to compiler discretion: split it into two phases.
2. **The two-phase split.** Index phase: the 4 indices are loaded **unconditionally** (when a slot may exceed
   the capacity of the index array, clamp with a **select** — "an `if` containing a single assignment" such as
   `slot_c = 0; if slot < CAP: slot_c = slot` is if-converted into a `SEL`, not a branch), the live flag is
   computed with Boolean algebra, and both are appended to Python lists. Copy phase: `if lives[rr]:` branches —
   by now the predicate is already in a register, so the branch waits on no scoreboard; there are no loads
   between the 4 branches, and the copy stream is continuous. SASS of the certified form: `LDG×5 · LDS×3 ·
   LDGSTS×8 · STS.128×8 · LDS×3 · LDGSTS×8 · STS.128×8 …` — ptxas predicates the copies and zero fills into one
   stretch of straight-line code, with no `BRA` between the 4 rows.
3. **The two-phase split is effective on its own; spilling is the only condition that undoes it.** Measured
   on the production kernel (same tree, same job, single variable, `zero_vec` not in the stack): with the zero
   fill **still scalar `STS.U16`**, changing only the index phase earned **+13.84%**, and that tree was
   `REG 255 / spill 0` — no spill, so the clustering holds. The certification probe's scalar-zero-fill
   variant is different: it zero-fills in every block with 46% invalid rows, the 93 loop-invariant addresses
   of the swizzled view are hoisted as a whole → `REG:255 / STACK:376`, and to free registers ptxas **sinks 3
   of the 4 `LDG`s already clustered at the top of the body below the first `LDGSTS`** (`ldg_after = 3`,
   exactly like the per-row form). So the order is **conditional**: when `-res-usage` reports `STACK ≠ 0` and
   the `STL`s fall between the `LDG` and the first `LDGSTS`, fix the zero fill first, then look at the LDG
   order; when `STACK = 0`, the two-phase split and the zero fill are two independent bills, each priced on
   its own.
4. **Vector spellings of the zero fill (both certified to become `STS.128`)**:
   (a) a `TiledCopy` of a 128-bit `CopyUniversalOp` atom (the same `(thr, val)` layout as the gather), whose
   source is a zero rmem fragment **built once outside the loop** (shape taken from the destination
   partition) and whose destination is the partition of the swizzled smem view; (b) a shorter spelling:
   `cute.flat_divide(sKV[(r, None)], (8,))` cuts a row into 16 B chunks, and `chunks[(None, c)].fill(0.0)`
   lands directly as one `STS.128`. The two have the same SASS shape (`STS.128×32`, `REG 208`, `STACK 0`) and
   the same time.

## Minimal runnable example

The geometry is only a size label: source `(n_src, 512)` bf16 row-major (1 KB/row), destination smem
`(64, 512)` 128 B-swizzle K-major face; 128 threads = 16 row teams × 8 lanes; per thread 4 rows × 8 chunks ×
16 B. `mIdx` is `(n_cta, CAP)` int32, with `-1` / `>= n_src` as sentinels; `tl` (runtime) is the valid prefix
length.

```python
g2s_atom  = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                                cutlass.BFloat16, num_bits_per_copy=128)
tiled_g2s = cute.make_tiled_copy_tv(g2s_atom, cute.make_layout(64), cute.make_layout(8))

grp, glane = tidx // 8, tidx % 8
tl = mTl[bidx]

for n in cutlass.range(n_blk):                       # runtime block count
    nb = n_blk - 1 - n
    # ---- phase 1: 4 indices + 4 lives, straight-line code, no branches ----
    idxs, lives = [], []
    for rr in cutlass.range_constexpr(4):
        r    = grp + rr * 16
        slot = nb * 64 + r
        slot_c = cutlass.Int32(0)                    # the clamp is a select, not a branch
        if slot < CAP:
            slot_c = slot
        idx  = cutlass.Int32(mIdx[(bidx, slot_c)])   # unconditional load
        live = (slot < tl) & (idx >= 0) & (idx < n_src)   # row-level predicate, same value across the 8 lanes
        idxs.append(idx)
        lives.append(live)
    # ---- phase 2: 8 back-to-back cp.async if live, otherwise 8 STS.128 zero fills ----
    for rr in cutlass.range_constexpr(4):
        r = grp + rr * 16
        if lives[rr]:
            for t in cutlass.range_constexpr(8):
                thr_c = tiled_g2s.get_slice(t * 8 + glane)
                cute.copy(tiled_g2s, thr_c.partition_S(mSrc[(idxs[rr], None)]),
                                     thr_c.partition_D(sKV[(r, None)]))
        else:
            chunks = cute.flat_divide(sKV[(r, None)], (8,))   # 64 16 B chunks
            for t in cutlass.range_constexpr(8):
                chunks[(None, t * 8 + glane)].fill(0.0)        # one STS.128
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()                          # or a 128-wide named barrier
```

Another zero-fill spelling of the same shape (when you want to reuse a `TiledCopy` partition):

```python
uni_atom  = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                                num_bits_per_copy=128)
tiled_uni = cute.make_tiled_copy_tv(uni_atom, cute.make_layout(64), cute.make_layout(8))
thr0  = tiled_uni.get_slice(glane)                   # built once outside the loop
zl    = cute.make_layout(cute.shape(thr0.partition_D(sKV[(0, None)])))
zfrag = cute.make_rmem_tensor(zl.shape, cutlass.BFloat16)
zfrag.fill(0.0)
...
            for t in cutlass.range_constexpr(8):     # the else branch of phase 2
                thr_c = tiled_uni.get_slice(t * 8 + glane)
                cute.copy(tiled_uni, zfrag, thr_c.partition_D(sKV[(r, None)]))
```

**Anti-pattern** (the same kernel with only the two phases merged back into one loop and the zero fill
written per element):

```python
for rr in cutlass.range_constexpr(4):
    live, idx = cutlass.Boolean(False), cutlass.Int32(0)
    if slot < tl:                                     # the load is inside the region
        idx = cutlass.Int32(mIdx[(bidx, slot)])
        if idx >= 0:
            if idx < n_src:
                live = cutlass.Boolean(True)
    if live:  ...copies...                            # this row's copy waits for this row's index
    else:
        for c in range_constexpr(64):
            sKV[(r, base + c)] = cutlass.BFloat16(0.0)   # ×64/lane: STS.U16, addresses hoisted as a whole -> spill
```

### Machine-read footprint (acceptance criteria)

| Reading | Target form (two-phase + vector zero fill) | Anti-pattern |
|---|---|---|
| `cuobjdump -sass`: number of `LDG` after the first `LDGSTS` | **0** (all 4 `LDG`s at the top of the body) | 3 (interleaved row by row) |
| run-length fingerprint of the kernel text | `LDGx5 LDSx3 LDGSTSx8 STS.128x8 …`, no `BRA` between rows | `LDG STLx93 LDLx8 LDG LDSx3 LDGSTSx8 LDGx3 BRA …` |
| `STS.128` / `STS.U16` | `STS.128 = 32`, `STS.U16 = 0` | `STS.U16 = 256` (64 per lane per row) |
| `-res-usage` | `REG 208 · STACK 0`, `LDL = STL = 0` | `REG 255 · STACK 376`, `LDL = STL = 93` |
| A/B time with the same index phase, only the zero fill changed (1 CTA/SM) | 1.00 | **1.567** (117.77 µs / 75.17 µs, certification round; round 1: 1.560) |

A third form — per-row `if` load + vector zero fill (`serial_vec`, no spill): `ldg_after = 0`, `REG 207 ·
STACK 0`, `STS.128 = 32`; its fingerprint has the **same shape** as the target form, `instr` 496 vs 472,
time **1.077** (80.94 / 75.17 µs). It is the "all footprints green but still 7.7% slower" tier: the criteria
must include `instr` and time.

`LDGDEPBAR` (commit) starts with `LDG`; classify it before `LDG` in the census, otherwise it necessarily sits
after an `LDGSTS` and counts the target form as ≥ 1 too (the per-family classification of
[`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)).

## Gotchas

1. **Whether a scalar zero fill wrecks the index clustering: look at `STACK`.** A per-element scalar zero
   store is not just `STS.U16 ×64/lane/row`: the address of every element of the swizzled view is a **loop
   invariant** of the runtime block loop, and when the live set is saturated ptxas hoists 93 such addresses
   out of the loop as a whole → `REG:255 / STACK:376` → spill traffic inserted between the `LDG` and the
   `LDGSTS`, sinking the already-clustered index loads — this is the reading of the certification probe's
   scalar variant (zero fill in every block, 46% invalid rows). On the production kernel (`REG 255` but
   `spill 0`) the same scalar zero fill did **not** trigger it: changing only the index phase with the zero
   fill untouched still earned +13.84%. Rule: `-res-usage` reports `STACK ≠ 0` and the `STL`s fall between the
   `LDG` and the first `LDGSTS` → fix the zero fill first; otherwise the two-phase split is effective on its
   own, and the zero fill is accounted separately on its own smem traffic ledger (`smem_wavefronts_st` /
   `bankconf_st`).
2. **A load written inside an `if` = leaving the decoupling to compiler discretion.** "Load inside
   `if slot < n_valid:`" looks like merely saving one out-of-bounds read, but the cost is chaining the round
   trips of the 4 rows — unless ptxas happens to if-convert the whole stretch and has registers to spare. Leave
   the out-of-bounds problem to the select clamp; **the predicate goes into the live flag, not into the
   load**. The same applies to nested regions such as `if idx >= 0: if idx < n_src:` — replace them with
   Boolean algebra `&`.
3. **The clamp must be in select form**: `slot_c = Int32(0); if slot < CAP: slot_c = slot` (the region contains
   only one pure register assignment, which ptxas if-converts into `SEL`). A region containing a memory access,
   such as `slot_c = mIdx[...]`, is a branch again. Variables assigned inside a dynamic `if` must be declared
   outside first (same as [`cp_async_predicated_zfill`](cp_async_predicated_zfill.md) gotcha 5).
4. **Append to Python lists only at the top level of `range_constexpr`.** Passing Int32 / Boolean between two
   `range_constexpr` loops via lists is the certified form; writing the append inside a runtime `if` region
   runs into "a first binding inside a region does not escape" of
   [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md).
5. **Build the zero fragment (spelling (a)) once outside the loop.** `make_rmem_tensor + fill` inside each
   row's `else` rematerialises it every row; the shape must be taken from the **destination partition**
   (`cute.shape(thr.partition_D(...))`), not hand-written as `(8,)` — the partition shape is a nested form
   like `((8), 1)`, and a hand-written shape is rejected by `cute.copy`'s per-mode comparison. Spelling (b)'s
   `.fill(0.0)` has no such problem; the chunk view carries its own shape.
6. **live must be a row-level quantity.** The 8 lanes of the same row must compute the same live (using only
   slot / idx / tl / n_src, with no lane component mixed in), otherwise one row is half copied, half zero
   filled.
7. **The copy phase's branch diverges within the warp** (one warp = 4 row teams, each with its own live). In
   the certified form ptxas predicates the two paths into one stretch of straight-line code (`LDGSTS×8 ·
   STS.128×8`, no `BRA`), at the cost of running both paths once each, far less than one gmem round trip. To
   avoid even the predication you can switch to the sister entry's src-size zero fill (`cp.async … 16,
   $nbytes`, `nbytes=0` for invalid rows) — but note that what that entry certified is a **per-row**
   `scf.for` (one row per trip, load→copy within the same trip); the combination "this entry's batched
   two-phase indices + that entry's src-size copy phase" **has not been certified by a probe**; to use it,
   start a probe first (it is in the gap queue).
8. **Don't use the `cute.copy` form in a producer after `setmaxnreg.dec`.** This entry's example uses the
   DSL's `cute.copy` to take addresses one by one, which is fine in a role with ≥ 200 registers (certified form
   `REG 208`, `LDL/STL == 0`); in a producer with a reduced budget it is a spill cliff — and the probe's spill
   variant in gotcha 1 demonstrates that spilling wrecks the index clustering along with it. The candidate way
   out is the "base + immediate" addressing of [`cp_async_predicated_zfill`](cp_async_predicated_zfill.md)
   §2b, but its combination with this entry's batched index phase is **uncertified** (same as gotcha 7); this
   entry has no readings for the form under a reduced budget.
9. **A batch depth of 4 indices = 4 registers, negligible in a 256 budget**; for deeper batches or tighter
   budgets, switch to the "each lane holds one + `shfl.idx`" of
   [`gather_index_lane_owned`](gather_index_lane_owned.md). Neither affects the copy phase.
10. **Don't use bytes as the criterion, and don't look only at values.** The serial chain does not read one
    extra byte, and the values are all correct; only the SASS order, `-res-usage`'s `STACK` and
    `stall_long_sb` can see it. After the change, rerun `cuobjdump -sass` to count "the `LDG`s after an
    `LDGSTS`" and check that `STL/LDL` are 0.
11. **The handoff is unchanged.** `commit_group / wait_group(0) / fence_view_async_shared / barrier` are the
    same as [`cp_async`](cp_async.md) §3; the zero fill is an ordinary STS, covered by the same
    `fence_view_async_shared` + barrier, and the consumer side needs no extra mask.

## Sources

- **PTX ISA**: `ld.global` (register-level loads and scoreboard dependencies), the `cp.async` / *Asynchronous
  copy* chapter (asynchronous copy addresses come from registers, completion goes through `commit_group` /
  `wait_group`), the vector form `.v4.b32` of `st.shared`, `st.local` / `ld.local` (spill traffic)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>.
- **Official NCU metrics**: `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active`
  (`stall_long_sb`), `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` (`smem_wavefronts_st`),
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` (`bankconf_st`).
- **Public infra API symbols**: `cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)`,
  `cute.nvgpu.CopyUniversalOp`, `cute.make_copy_atom` / `cute.make_tiled_copy_tv` / `TiledCopy.get_slice` /
  `partition_S` / `partition_D`, `cute.flat_divide`, `cute.make_rmem_tensor` + `Tensor.fill`,
  `cute.arch.cp_async_commit_group` / `cp_async_wait_group` / `fence_view_async_shared`, `cutlass.range` /
  `cutlass.range_constexpr`, `cutlass.Boolean`; `cuobjdump -sass` / `-res-usage`.
- **Certified recipes of the same family**: [`gather_index_prefetch`](gather_index_prefetch.md) (principle and
  triad), [`cp_async`](cp_async.md) (partitioning / group completion),
  [`cp_async_predicated_zfill`](cp_async_predicated_zfill.md) (branch-free zero fill + reduced-budget
  addressing), [`gather_index_lane_owned`](gather_index_lane_owned.md) (holding deep batches),
  [`guard_predicate_issue_distance`](guard_predicate_issue_distance.md) (the general rule that branches are not
  speculated across), [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) (region scope).
- **GPU-certification probe** (not included in this release) — five builds of the same kernel
  differing only in two source knobs (index phase serial / clustered × zero fill scalar / vec / `.fill`), with
  each axis's A/B holding the other knob at the recipe value; eight contracts: four numeric (the two recipe
  spellings + two control variants, all equal row by row to a numpy gather, invalid rows all 0, read back by
  another thread) + `gather_pred_recipe_footprint` (both recipe spellings: `LDG ≥ 4` all before the first
  `LDGSTS`, `LDGSTS ≥ 32`, `STS.128 ≥ 32`, `STS.U16 = 0`, `LDL = STL = 0`) + `gather_pred_fill_footprint`
  (scalar variant `STS.U16 ≥ 256`) + `gather_pred_fill_time` (same-index-phase A/B ≥ 1.25) +
  `gather_pred_index_axis` (A/B with the same vector zero fill: the two-phase form is no slower than the per-row
  form; the per-row variant's own `ldg_after` is recorded as a reading in the verdict line). The readings of
  the first round (both variants used the scalar zero fill for the index-phase A/B) are the evidence for
  gotcha 1's spill variant: both variants were `REG 255 / STACK 376 / STL 93 / ldg_after 3`, time ratio 1.005.
  Certification-round readings: `clustered_vec` / `clustered_fillapi` 75.17 / 75.62 µs (`REG 208 · STACK 0 ·
  ldg_after 0 · instr 472`), `serial_vec` 80.94 µs (`REG 207 · ldg_after 0 · instr 496`, index_axis 1.077),
  `clustered_scalar` 117.77 µs (`REG 255 · STACK 376 · ldg_after 3`, fill_time 1.567), `serial_scalar`
  118.02 µs; landing on the production kernel (zero fill scalar→vector bundled with the two-phase indices)
  72.06 → 63.21 ms. Status: ✅ Verified — `PROBE PASS max_diff=0.0` (round 2 of this
  certification, all eight contracts passed).
