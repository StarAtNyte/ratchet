# tmem_ld_epilogue_issue_order (instruction order of `tcgen05.ld` / `wait::ld` / conversion / `st.shared` / fence / rendezvous in a chunked epilogue: which edges are hard, what ptxas has already moved for you, and what source reordering read on the whole kernel)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**. For how the four stages fit together (atom, partitioning, staging face, descriptor) see
> [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md); for the register peak of the read-back segment see
> [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md). This entry only covers instruction order **once everything already fits together correctly**:
> which ordering edges are hard, what ptxas has already moved for you between rendezvous, and what "interleave face writes and conversion per piece", "source rotation" and "replace the face rendezvous with an mbarrier" read in SASS and on the whole kernel.
>
> **Status: ✅ Verified (spelling and values; gains not certified)** — five independent GPU probes (not included in this release), all bit-exact:
> - Hard ordering edges + `%clock64` breakdown + three wrap orders: `PROBE PASS max_diff=0.0` (round 1, R0–R4);
> - Drain readings of an in-flight `tcgen05.ld` across `st.shared` / `fence.proxy.async` / `bar.sync`: `PROBE PASS max_diff=0.0` (round 2, R0–R5);
> - Four orders interleaving face writes with the next chunk's conversion per piece + SASS counts: `PROBE PASS max_diff=0.0` (round 1, R0–R6);
> - Source rotation and SASS order of small-group "read-back → global store": `PROBE PASS max_diff=0.0` (round 4);
> - Two rendezvous forms for a shared face: `PROBE PASS max_diff=0.0` (round 1, R0–R4).
>
> **Read this sentence first**: every reordering above has its spelling certified in a probe, but **none ever read a gain on a whole kernel** (readings under "When needed").
> The reason is in Purpose §2: ptxas already moves `LDTM` on its own between two `bar.sync`. Read the SASS before touching the source.

## Purpose

### §1 Hard ordering edges of one epilogue chunk

One chunk of a chunked epilogue is: read back → wait for the read-back → convert to the narrow type → write the staging slot → cross proxy → rendezvous → one warp issues the TMA store → return the slot.
Chunk period = rendezvous wait + "wrap" (each warp's own instruction stream between this chunk's `bar.sync` returning and the next chunk's `bar.sync` being issued).
Within the wrap only the order can move, and the order is constrained by these edges:

| step | instruction | must come after | what it blocks |
|---|---|---|---|
| (1) read back | `tcgen05.ld` (`cute.copy(t2r, tmem_src, rmem)`) | the wait for accumulator full | the destination registers cannot be read before (2) |
| (2) wait for read-back | `tcgen05.wait::ld` (`cute.arch.fence_view_async_tmem_load()`) | (1) | **all `tcgen05.ld` this thread issued before it**, regardless of which |
| (3) convert | `cvt` / `F2FP` (`rmem.load().to(F16)`) | (2) | — |
| (4) write slot | `st.shared` / `stmatrix` | (3); **and** the previous round's TMA store from this slot has finished reading (the accounting of (8)) | — |
| (5) cross proxy | `fence.proxy.async.shared::cta` (`cute.arch.fence_proxy("async.shared", space="cta")`) | this thread's **last** (4) of this chunk | (6) |
| (6) rendezvous | `bar.sync id, N` (`NamedBarrier(...).arrive_and_wait()`) | (5) | (7) |
| (7) issue store | `cp.async.bulk.tensor` S2G + `cp.async.bulk.commit_group` (one warp) | (6) | — |
| (8) return slot | `cp.async.bulk.wait_group.read N` (only in the warp that issues stores) | (7) | the next (4) writing this slot (passed to other warps through one (6)) |

### §2 What these edges become in SASS (the root of every reordering conclusion)

- `tcgen05.wait::ld` is **not a standalone fence instruction** in SASS; it is a scoreboard wait attached to the first instruction that reads a destination register.
  So between two `bar.sync`, ptxas moves `LDTM` **in both directions**:
  - Small-group "read-back → global store" chain (no rendezvous between groups): source with per-group `ld; wait; st` and a hand-written rotation of two fragments compile to **item-for-item identical** order —
    `LDTM LDTM STG STG LDTM LDTM …`, with the number of `LDTM`s before the k-th `STG` being `[2, 2, 4, 4, …, G, G]`.
  - Read-back written to a shared face, source "issue the next chunk's ld before writing this chunk's face" vs sequential "ld → wait → write face": the SASS instruction lists are exactly the same,
    and the LDTM lead is 2 in both.
  - Source places the next chunk's ld right after this chunk's `wait::ld`: in SASS that `LDTM` is sunk **after** this chunk's `STS` and `MEMBAR.ALL.CTA` +
    `FENCE.VIEW.ASYNC.S` (the lowering of `fence.proxy.async`), before `BAR.SYNC` (the probe's SASS fact line).
- **The only edge observed never to be crossed is `BAR.SYNC`**: in the serial form the next chunk's `LDTM` always stays after `BAR.SYNC` → `UTMASTG` →
  `DEPBAR.LE` (same job).
- Corollary: **source reordering between two rendezvous often does not change SASS** (and when it does, only the part ptxas would have moved anyway); what reliably changes SASS is "moving an instruction across a rendezvous"
  (e.g. moving an ld from the top of the next chunk to before this chunk's rendezvous changed the SASS accordingly) or "changing the number of rendezvous".

### §3 Does the tail of a chunk drain an in-flight `LDTM`?

The probe's drain shape: `stamp a; ld; stamp b; X; stamp c; wait::ld; stamp d`; X drained the ld ⇔ `(c − b)` grows to the order of the read-back latency and `(d − c)` ≈ 0.
Median readings of the store-issuing warp (cycles):

| X | across X (c − b) | wait::ld (d − c) |
|---|---|---|
| none | 111 | 102 |
| `fence.proxy.async` | 118 | 107 |
| `bar.sync` (128 threads) | 110 | 112 |
| `st.shared` (one fp16 piece of the previous round) | 339 | 47 |
| all three | 490 | 35 |

How to read it: neither `fence.proxy.async` nor `bar.sync` **drains** an in-flight `LDTM` (the wait segment is unchanged); `st.shared` takes time itself and hides more than half of the read-back latency,
but no X brings the wait segment to zero — **no drain was observed**.

### §4 Probe readings of the two reorderings (one CTA per SM, epilogue owns the SM; look at ratios only)

- **Order within the wrap** (2 read-back pieces per chunk, store-issuing warp, median cycles over the middle chunks):
  serial wrap 243 / chunk period 276; no reordering, including moving the `wait_group`, was shorter. Every wrap segment is 20–77 cycles; none dominates.
- **Interleaving face writes with the next chunk's conversion per piece**: coarse chunk period (median over the middle chunks) write face then convert 387 / interleaved 402 /
  interleaved with the next chunk's ld after the interleave 407 / write one piece first, then wait 403 cycles; per launch 10.47 / 10.50 / 10.49 / 10.74 µs.
  "Number of `F2FP`s between the first and last `STS`" per segment: the write-then-convert source has the most in the middle segments, and the three interleaved spellings actually have 1–6 fewer
  — **the source that does not interleave is the one ptxas interleaves most**.
- **Face rendezvous** (median per unit): sequential write + mbarrier rendezvous 0.895 µs, full-width named barrier rendezvous 0.811 µs (fastest, see §5).

### §5 Rendezvous on a shared face: mbarrier where only the store-issuing warp waits vs a full-width named barrier

Alternative spelling: each writing warp does `fence.proxy.async` → `sync_warp` → `elect_one` arrives on `face_full[buf]` (count = number of writing warps),
**does not wait**, and goes straight to read the next chunk; only the store-issuing warp waits on `face_full`; the writers' only blocking point is `face_empty[buf]` before reusing the face.
For the FULL / EMPTY accounting and parity of the mbarriers see [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md) gotcha 14.
Certified readings: with an even writer load and one store-issuing thread, **the full-width named barrier is faster than the mbarrier spelling** (0.811 vs 0.872 µs/unit);
in SASS the named-barrier version has `BAR.SYNC` 6 / `SYNCS.ARRIVE` 3 / `TRYWAIT` 6, the mbarrier version 3 / 6 / 12 — the writers execute twice as many mbarrier instructions.

## When needed (trigger)

- You plan any of "cross-chunk rotation / interleaving face writes with conversion / replacing the rendezvous with an mbarrier", or on the timeline the epilogue's
  per-chunk period has a wrap longer than the rendezvous wait, and ablation has already ruled out "TMEM read-back latency", "the rendezvous itself" and "`STS` bank conflicts".
- In static SASS every `LDTM` is immediately followed by an `F2FP` reading its destination register, and you suspect the read-back latency is on the critical path.

**Whole-kernel readings (all from same-job A/Bs or gates after the same mechanism landed in a real kernel; not one is a gain)** — before landing, do a same-job A/B on your own kernel:

| landed form | structure | readings |
|---|---|---|
| 11 rounds of wrap reordering (conversion done early, `wait_group` delayed, piece interleaving, forcing `F2FP` to stay before the fence, etc.) | epilogue serial after the main loop | gate 0.017–0.0172 ms each round, best 0.017 ms, no round faster; in the round after interleaving, the same-job A/B of three variants base/variant 1.0016 / 1.0020 / 0.9998 |
| three face-rendezvous forms (mbarrier where only the store-issuing warp waits / full-width named barrier / store-issuing warp `bar.sync`, the others `bar.arrive`) | each CTA writes several shared-face chunks | 0.9970 / 0.9972 (relative to the mbarrier form, within noise) |
| source rotation of small-group read-back | read-back stored directly to global | SASS **item-for-item identical** to the serial form — a no-op change |


## Minimal runnable example

The serial loop body of the certification probe (`S` staging slots, `INFL` store groups in flight, `NPC` read-back pieces per chunk, `CH` chunks;
`ld(c)` = this chunk's `NPC` calls to `cute.copy(t2r, …)`, returning a list of f32 register fragments; `cvt` / `slot_piece` are plain Python helpers):

```python
assert INFL <= S - 2                                   # slot reuse condition (gotcha 3), checked at trace time
for c in cutlass.range_constexpr(CH):
    slot = c % S
    cur = ld(c)                                        # (1) issue this chunk's ld at the top of the chunk
    cute.arch.fence_view_async_tmem_load()             # (2) wait::ld
    hs = [cvt(r) for r in cur]                         # (3)
    for j in cutlass.range_constexpr(NPC):
        cute.autovec_copy(hs[j], thr.partition_D(slot_piece(slot, j)))   # (4)
    cute.arch.fence_proxy("async.shared", space="cta")  # (5) after this chunk's last write, before the rendezvous, once per chunk
    epi_bar.arrive_and_wait()                          # (6)
    if warp_idx == 0:
        cute.copy(tma_c, bSG_sC[(None, slot)], bSG_gC[(None, cta, c)])   # (7)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(INFL, read=True)             # (8)
if warp_idx == 0:
    cute.arch.cp_async_bulk_wait_group(0, read=False)  # drain before exit
```

`%clock64` stamps (for breaking down the wrap; the address is baked into a constant at trace time, and only threads with `gate == 0` write):

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def clk_store(gate, addr, *, loc=None, ip=None):
    llvm.inline_asm(None,
        [cutlass.Int32(gate).ir_value(loc=loc, ip=ip), cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b64 t;\n\tsetp.eq.b32 p, $0, 0;\n"
        "\tmov.u64 t, %clock64;\n\t@p st.global.u64 [$1], t;\n}",
        "r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
# stamp for chunk c, step k: clk_store(lane | cta, BASE + ((warp * CH + c) * NSITE + k) * 8)
```

Acceptance (both require reading SASS; values cannot show order):
- **Has ptxas already moved it for you**: between rendezvous, count "the number of `LDTM`s already issued before the k-th write" or the "`LDTM` lead" — if it is already ≥ 2, a source rotation is a no-op change.
- **Did the interleave survive**: count "the number of `F2FP`s between the first and last `STS`" and compare with the **non-interleaved** source — the non-interleaved source itself may have more (§4).

Getting SASS: `CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>` (one process and one directory per variant), then `nvdisasm -c <cubin>` or `cuobjdump -sass`.

## Gotchas

1. **`tcgen05.wait::ld` is not selective and has no "wait up to the k-th" counted form** (unlike `cp.async.wait_group N`).
   Any `tcgen05.ld` issued before this chunk's wait is waited on together by this chunk's wait.
2. **Do not drop the per-chunk `wait::ld`.** The probe's fact variant "no `wait::ld` per chunk, one wait after the loop" happened to be numerically correct (median chunk period 362 cycles),
   but that is the SASS scoreboard covering for it, not something a PTX spelling should rely on; the PTX ISA defines `tcgen05.ld` as asynchronous, with completion observed by `tcgen05.wait::ld`. The acceptance payload must encode coordinates (something like `m + (n % 16) / 16`); a constant payload cannot reveal a read-too-early error.
3. **The safety condition for slot reuse depends only on which two rendezvous the `wait_group` falls between.** If the store-issuing warp executes
   `wait_group(N, read)` after chunk c and before chunk c+1's rendezvous, then slot `(c+2) % S`, written by other warps after chunk c+1's rendezvous, needs group `c+2−S` to have finished reading, i.e. **`N ≤ S − 2`**.
   Moving this wait from "after commit" to "before the next chunk's own face write" keeps the condition (the probe certified both positions); moving it **after** the next chunk's rendezvous breaks it.
4. **`wait_group` runs only on the store-issuing warp** — bulk-group accounting is per thread; the other warps inherit slot safety through the next rendezvous.
5. **`fence.proxy.async` once per thread per chunk, positioned after this chunk's last write and before the rendezvous**; it cannot be inserted between pieces (it then only covers the first pieces,
   TMA reads stale bytes — silently wrong; a bit-exact comparison catches it). When a shared face uses an mbarrier rendezvous, every writing thread must fence, then `sync_warp`, then one elected thread arrives.
6. **Stamps have a cost; only break down ratios.** Each stamp is one `st.global` + one `%clock64`; 5–8 stamps in the wrap lengthen it, so do not compare absolute stamped values with unstamped ms;
   `%clock64` is only comparable within one SM. `has_side_effects=True` only orders the stamps among themselves; pure register instructions still move across stamps — the "cvt segment" reading is approximate.
7. **Loops inside helpers use Python `range`** (unrolled at trace time); `cutlass.range_constexpr` may only be written in the body of a `@cute.kernel` / `@cute.jit` function;
   calling it in a plain function raises at compile time `DSLRuntimeError: range_constexpr should be preprocessed by preprocessor.`
8. **Three details of the shared-face mbarrier rendezvous**: the count of `face_full` = number of writing warps (each warp arrives once via `elect_one`); the face parity is computed from the **global face index**
    (`g = u · NF + j`, `buf = g & 1`, `par = (g >> 1) & 1`; with an odd number of faces per unit it cannot be written as a constant), and writers wait on `face_empty` with `par ^ 1`;
    `cute.copy(tma_c, …)` is called by the whole warp, not wrapped in `elect_one`.

## Sources

- PTX ISA: `tcgen05.ld` / `tcgen05.wait::ld` (Tensor Memory section), `fence.proxy.async`, `cp.async.bulk.commit_group` /
  `cp.async.bulk.wait_group.read`, `bar.sync`, `mbarrier`, `%clock64` — https://docs.nvidia.com/cuda/parallel-thread-execution/
- Public API symbols: `tcgen05.Ld32x32bOp` / `Repetition` / `make_tmem_copy`, `cute.arch.fence_view_async_tmem_load`,
  `cute.arch.fence_proxy`, `cute.arch.cp_async_bulk_commit_group` / `cp_async_bulk_wait_group`, `cute.arch.mbarrier_init` /
  `mbarrier_arrive` / `mbarrier_wait`, `cute.arch.elect_one` / `sync_warp`, `pipeline.NamedBarrier`, `cute.autovec_copy`,
  `cpasync.make_tiled_tma_atom(CopyBulkTensorTileS2GOp(), …)` / `tma_partition`, `cutlass._mlir.dialects.llvm.inline_asm`.
- Certification probes (five):
  - Chunk wrap reorder — `PROBE PASS max_diff=0.0` (round 1): R0 compiles, R1–R3 three wrap orders bit-equal to numpy, R4 stamps monotonic.
  - In-flight TMEM ld, one deep — `PROBE PASS max_diff=0.0` (round 2): R0 compiles, R1 drain family with five X, R2–R4 serial / ld right after wait / ld after cvt bit-equal, R5 stamps monotonic; the fact variant "no wait per chunk" is recorded as readings only.
  - STS / cvt piece interleave — `PROBE PASS max_diff=0.0` (round 1): R0 compiles, R1–R4 four orders bit-equal, R5 stamps monotonic, R6 SASS parse.
  - TMEM read-back → STG rotation — `PROBE PASS max_diff=0.0` (round 4): both forms write bit-equal output; the serial form's SASS already hoists `LDTM` across `wait::ld`; the two forms' orders are item-for-item identical.
  - TMEM read-out face, batched rendezvous — `PROBE PASS max_diff=0.0` (round 1): three spellings bit-equal, LDTM lead ≥ 2.
