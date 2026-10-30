# cp_async (non-TMA per-thread `cp.async` / LDGSTS: row-wise sparse gmem→smem gather)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry and the TMA G2S of [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) are **two parallel supply
> paths**: TMA moves "regular blocks a descriptor can describe", this entry moves "rows selected by runtime
> indices".

## Purpose

Move **several whole rows selected by runtime indices** from gmem into smem, without a register round trip and
without a TMA descriptor. `cp.async` (`LDGSTS` in SASS) is a **per-thread** asynchronous copy instruction: the
address is computed in registers, so "which row to move" can be any runtime value — non-monotonic, repeatable,
no sorting needed. At most **128 bits** (16 B) per thread per instruction; completion is managed by the group
counts of `commit_group` / `wait_group`.

## When needed (trigger)

- The **row set of the supply is only known at runtime** (index arrays, page tables, row numbers after
  compaction), with no static box coordinates to write into a TMA descriptor.
- Symptom: you want TMA but find you need "several boxes landing in the same buffer" — every certified TMA
  multi-box loop in the KB is **one box per pipeline stage**, and stuffing N non-adjacent rows into one stage is
  not among TMA's certified forms.
- Symptom: filling smem element by element with plain `sX[i] = gX[j]` (the `CopyUniversalOp` path) computes
  correctly, but the transfer goes through a register round trip (LDG→STS) and occupies the issuing thread, so
  you don't get "transfer overlapped with compute".
- Reverse exclusion: when the row set is regular (contiguous tile / fixed stride / im2col sliding window),
  **don't** use this entry; use TMA (`wgmma_tma_mbarrier` / `tma_im2col` / `tma_ndim_oob_zerofill`).

## Minimal runnable example

Transfer geometry (the concrete numbers in this example are only **size labels**; the formulas follow): source
`(256, 512)` bf16 row-major, gathered by 32 runtime row numbers into a `(32, 512)` smem buffer; 128 threads.

```
bytes per row ROW_BYTES = 512 * 2 = 1024               -> row starts naturally 16 B-aligned
each thread moves 128 bit = 16 B = 8 bf16              -> ELEMS_PER_THREAD = 8
covering one row needs THREADS_PER_ROW = ROW_BYTES / 16 = 64  -> one "row team" per row
128 threads / 64 = ROW_TEAMS = 2                       -> each team walks ROWS/2 = 16 rows
```

In general: `THREADS_PER_ROW = ROW_BYTES / 16`. When `THREADS_PER_ROW <= NTHREADS`, split into teams by
`ROW_TEAMS = NTHREADS / THREADS_PER_ROW`, each team walking its own batch of rows (this example's form); when
`THREADS_PER_ROW > NTHREADS`, the other way round: one row is issued in `THREADS_PER_ROW / NTHREADS` passes. In
both cases **the row indices are runtime values and the pass counts are constexpr**.

### 1) copy atom and TiledCopy (sync/async differ only by one op)

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass.cute.nvgpu import cpasync

# 128 bit is the per-instruction limit of cp.async; bf16 -> 8 elements
op   = cpasync.CopyG2SOp()            # async: LDGSTS
# op = cute.nvgpu.CopyUniversalOp()   # sync control: plain LDG+STS (top-level namespace)
atom  = cute.make_copy_atom(op, cutlass.BFloat16, num_bits_per_copy=128)
tiled = cute.make_tiled_copy_tv(atom,
                                cute.make_layout(THREADS_PER_ROW),  # thr layout
                                cute.make_layout(ELEMS_PER_THREAD)) # val layout
```

### 2) Kernel side: per-row slicing + partitioning + issue

```python
smem = cutlass.utils.SmemAllocator()
sKV = smem.allocate_tensor(
    cutlass.BFloat16,
    cute.make_layout((ROWS, ROW_ELEMS), stride=(ROW_ELEMS, 1)),
    byte_alignment=128)

# the copy is sliced by "lane within the row"; the team number only decides which rows this thread walks (both are dynamic values)
thr  = tiled.get_slice(tidx % THREADS_PER_ROW)
team = tidx // THREADS_PER_ROW

for i in cutlass.range_constexpr(ROWS_PER_TEAM):     # pass count constexpr
    r = i * ROW_TEAMS + team                         # destination row number (dynamic)
    cute.copy(tiled,
              thr.partition_S(mSrc[(mIdx[r], None)]),  # source row: runtime index -> rank-1 slice
              thr.partition_D(sKV[(r, None)]))         # destination row: rank-1 slice
    if i == ROWS_PER_TEAM // 2 - 1:
        cp_async_commit_group()                        # commit in two batches (see §3)

cp_async_commit_group()
cp_async_wait_group(1)      # allow at most 1 group in flight (this one line is what a multi-stage pipeline is)
cp_async_wait_group(0)      # drain
cute.arch.fence_view_async_shared()
cute.arch.sync_threads()    # cp.async retires per thread — cross-thread visibility relies on this barrier
```

### 3) Group completion: `commit_group` / `wait_group` (inline PTX)

This family in `cute.arch` only has the **bulk (TMA)** versions `cp_async_bulk_commit_group` /
`cp_async_bulk_wait_group`, and their naming differs between DSL minor versions (the same situation as
[`s2g_bulk_reduce`](s2g_bulk_reduce.md)). The two non-bulk ones are **operand-less per-thread instructions**;
inline PTX is the shortest and has no version risk:

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

@dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def cp_async_wait_group(n, *, loc=None, ip=None):
    """Wait until "the number of committed groups in flight <= n"."""
    llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(n), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

Semantics (PTX ISA): `commit_group` bundles all `cp.async` previously issued by **this thread** into one group;
`wait_group N` waits until the number of groups in flight ≤ N. A multi-stage pipeline is just "commit once after
issuing each stage, `wait_group(stages-1)` before consuming" — the stage count is a lever, not part of this
recipe.

## Gotchas

1. **Correct values do not mean the mechanism is in effect.** The scalar fallback path (`CopyUniversalOp`, or
   even per-element `sX[i]=gX[j]`) gives exactly the same result as cp.async. The only thing that tells them apart
   is the **issue footprint**. And **for obtaining the footprint itself, two certified recipes in the KB give
   contradictory methods** — the most time-consuming pitfall of this entry, so lay the contradiction out first:
   - the acceptance section of [`warp_specialization`](warp_specialization.md) certifies
     `cute.compile(..., options=f"--dump-dir={d} --keep-cubin --keep-ptx")`;
   - the "acceptance" section of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) and the "getting the
     cubin" gotcha of [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md) both record `[MEASURED]`:
     **the same `options=` string is mute**, and the only working path is to set `CUTE_DSL_KEEP_CUBIN=1` +
     `CUTE_DSL_DUMP_DIR=<dir>` **before** `import cutlass`.

   Both are ✅, so **don't bet on one** — feed both sinks in one compile, and take the OR of all instruments as
   the criterion. (Note that `CUTE_DSL_KEEP_PTX` has **no source anywhere** in the whole tree; only `KEEP_CUBIN`
   does; setting it is harmless, but don't treat it as a working path.)
   ```python
   os.environ["CUTE_DSL_DUMP_DIR"]   = env_dir    # must come before import cutlass
   os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
   import cutlass, cutlass.cute as cute           # <- only import afterwards
   ...
   # one compile, two sinks; opt_dir is driven by options=, env_dir by the environment variables
   compiled = cute.compile(jit_fn, *args,
                           options="--dump-dir=%s --keep-cubin --keep-ptx" % opt_dir)
   # scan both directories, taking only files newly produced by this compile (by mtime)
   # *.ptx should contain 16-byte cp.async copies (line ending `, 16;`)
   # cuobjdump -sass *.cubin | grep -c LDGSTS   -> should be > 0
   ```
   **Compiling only once** matters: `cfrag_fused_pack_liveness` records that dump file names are built only from
   the tensor signature, so a second compile in the same process **overwrites in place** and may hit the compile
   cache directly (in which case the second contract passes vacuously). For comparisons, use separate processes.

   Each of the three instruments (PTX / SASS / NVVM IR) may be missing in some image (no `cuobjdump`, PTX not
   written to disk, the jit object not exposing IR), so **take the OR of the three as the criterion**, and **first
   assert that at least one instrument is alive**: with the dump directory empty, "no cp.async found" and "the
   instruments are mute" look exactly the same, and the latter must be reported as an instrument failure, not a
   mechanism failure (same as the ptxas return-code pitfall in `warp_specialization`). Printing which root
   directories were actually searched and which files were found is the cheapest single line that tells the two
   apart.
2. **Missing `has_side_effects=True` → commit/wait are DCE'd.** A void inline asm without it is deleted
   entirely; the symptom is **reading back all sentinels** ("the data never arrived"), not an error. Pre-fill smem
   with a sentinel value outside the data domain so that such silent failures become visible.
3. **Never wrap cp.async in `elect_one` or `if tidx == 0`.** It is a per-thread instruction; every thread must
   issue its own 16 B; wrapping it means only 1 thread (`elect_one` is warp-scoped → 1 per warp) issues, and the
   result is "only 1/64 of a row is real data, the rest is sentinel". The "raw PTX must be wrapped" of
   [`elect_one`](elect_one.md) gotcha 6 is about **bulk/TMA** single-issue instructions and does not apply here —
   the line is drawn by "is this instruction one per thread or one per CTA", not by "is it inline PTX".
4. **16 B alignment is a hard requirement, on both sides.** `num_bits_per_copy=128` requires both source and
   destination addresses to be 16 B-aligned: `ROW_BYTES % 16 == 0`, the byte count of the smem row stride
   `% 16 == 0`, the gmem base 16 B-aligned (host side `from_dlpack(t, assumed_align=16)`). When the row length is
   not an integer multiple of 16 B, drop to `num_bits_per_copy=32` (or pad the rows to a 16 B boundary); don't
   expect the hardware to split it for you.
5. **`wait_group` only guarantees this thread's copies have arrived; the barrier guarantees everyone else's.**
   The symptom of missing `sync_threads()` is not all wrong, but **each thread only sees the rows it moved
   itself** — a "mostly right, wrong in per-thread blocks" result, much harder to track down than all wrong.
6. **`partition_S` / `partition_D` must come from the partition object of the same `get_slice`**, and `cute.copy`
   takes a `TiledCopy` (not a `CopyAtom`). When the two sides' partitions come from different slices (e.g. one
   side uses `tidx`, the other `tidx % THREADS_PER_ROW`), the two TV bijections disagree; no shape error is
   produced, but data is **moved to the wrong places**.
7. **Pass counts use `range_constexpr`, row numbers use dynamic values.** Writing the loop bound as a dynamic
   quantity prevents unrolling and address hoisting; confine the dynamism to the single place `mIdx[r]`.
7b. **`range_constexpr` can only be written in the body of a function directly decorated with `@cute.kernel` /
   `@cute.jit`.** It is not an ordinary Python generator but a marker for a **rewrite the preprocessor does on
   the decorated function's source**; the preprocessor only looks at the source of the decorated function itself
   and **does not enter the bodies of plain Python helpers it calls**. Moving the loop of §2 into an undecorated
   helper (e.g. to share one copy of the code between the sync/async paths) and calling it from the kernel is a
   hard compile-time failure:

       Detail: range_constexpr should be preprocessed by preprocessor.

   The identifying feature of this pitfall is that **the sync control group dies too** — it contains not a single
   `cp.async`, so "the control group doesn't compile either" proves the problem is in the trace structure, not in
   the async mechanism or the values. Likewise, `@cute.kernel` / `@cute.jit` definitions belong at **module top
   level**, not nested inside factory functions evaluated at runtime. The cost is that the sync/async paths each
   unroll their own copy of the loop — that duplication is necessary. **Pure setup code with no DSL control
   flow** (building `CopyAtom`/`TiledCopy`, `SmemAllocator` allocations, `@dsl_user_op` inline-PTX wrappers) is
   fine to leave in helpers: the dividing line is **loops and control flow cannot be moved out, setup can**.
   Refinement (GPU-verified, see [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)): "cannot be moved out"
   precisely means cannot be moved into an **undecorated** function — a helper carrying its own `@cute.jit` can
   be called normally from the traced body and its control flow is still rewritten; that entry also generalises
   it to all runtime control flow (a bare `if` on a runtime value raises `PHASE_DYNAMIC_TO_STATIC_BOOL`, same
   fix).
   Also note: `range_constexpr` **fully unrolls** the loop body, so the pass count translates directly into
   compile time. Scaling the pre-fill/read-back pass counts up to hundreds turns compilation itself into a
   timeout risk (also showing up as "no verdict" rather than wrong values); keep the geometry small.
8. **Swizzled destination buffer**: when the destination is switched to a swizzled tensor built with
   `get_smem_layout_atom` + `tile_to_shape`, the code of this section is unchanged — `partition_D` acts directly on
   the swizzled tensor, with the swizzle encapsulated in the layout (same as
   [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) gotcha 6). **This entry is GPU-certified only for a flat row-major
   destination**; when switching to a swizzled destination, check once yourself: the swizzle scrambles the
   positions of 16 B chunks, not the bytes within a chunk, so a 128-bit copy still lands on a single legal address
   ([INFERRED]), but if the innermost contiguous run of the destination layout is shorter than 16 B, that premise
   is gone.
9. **When the consumer is WGMMA, add `fence_view_async_shared()`** (already in this entry's example). WGMMA reads
   smem through the async proxy, and the pitfall is that missing it **raises no error, and the result is
   0/garbage** (`wgmma_ss_min_gemm` gotcha 1). This probe's consumer is an ordinary load, so here it is only
   certified as "harmless to add and the path is correct"; there is no separate A/B of "is it required when WGMMA
   consumes".
10. **Cache qualifiers (`.ca` / `.cg`, whether to bypass L1) — certified, see the section "Cache qualifiers"
    below.** The argument-less `CopyG2SOp()` is the DSL default `.ca` (SASS bare `LDGSTS.E`);
    `cache_mode=LoadCacheMode.GLOBAL` is what gives `.cg` (SASS `LDGSTS.E.BYPASS`). Print the knob name at runtime
    before writing it, don't go from memory (names like `CG` / `L2` do not exist):
    ```python
    import inspect
    print(inspect.signature(cpasync.CopyG2SOp.__init__))
    print([n for n in dir(cpasync) if "Cache" in n or "cache" in n])
    ```

## Cache qualifiers (certified)

`cp.async` has two mutually independent cache bits, each taking one suffix in SASS; the DSL exposes only one of
them:

| PTX | SASS suffix | DSL spelling | Notes |
|---|---|---|---|
| `.ca` (default, allocates through L1) | none | `cpasync.CopyG2SOp()` = `CopyG2SOp(cache_mode=LoadCacheMode.ALWAYS)` | cp-size may be 4 / 8 / 16 B; can be **mixed in the same group** with large `.cg` chunks (see the section "Mixed cache modes" below) |
| `.cg` (bypasses L1, L2 only) | `.BYPASS` | `cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)` | cp-size **must be 16 B** (PTX ISA); 4 / 8 B copies can only use `.ca`; the constructor argument is right in the signature; members `ALWAYS / GLOBAL / LAST_USE / NONE / STREAMING` |
| `.L2::256B` (L2 prefetch width, `.level::prefetch_size`) | `.LTC256B` | **none** — only via the inline PTX of [`ptx_escape`](ptx_escape.md) form 2; spelling in the section "`.L2::256B` prefetch width" below | independent of both the `.cg` and src-size bits; its value varies with the tree: ≈ 0 on a tree where `.cg` was just switched on, −0.81% bundled with spin waits on a tree where supply/waits are already tight (next section) |

**Load-bearing condition (when `.cg` is worth something)**: when smem takes most of the shared L1/smem array
(256 KB) (≥ ~200 KB), `.ca`'s "allocate a line in L1 first, then fill" path has almost no L1 to allocate, and
every `LDGSTS` queues on that path; `.cg` skips it. The symptom is high `stall_long_sb`, low `tensor_pct`, while
`dram_bytes_read` is **unchanged or even slightly up** — it is a latency ledger, not a bandwidth ledger, so
**acceptance is anchored on the dossier's `cache_ops` axis** (`LDGSTS.E.BYPASS` appearing at the issue points),
not on proxies like `dram_bytes_read`. On a row-wise gather kernel with 213 KB of smem, this one constructor
argument was worth 22% of device time; on a tree of the same family where handshake stalls still dominated it was
worth only ~4% — the number varies with the tree, the condition does not.

**Certification** (a GPU probe, not included in this release; three interleaved rounds in the same
job, correctness + footprint gate + NCU): `PROBE PASS dsl_cg` (`LoadCacheMode.GLOBAL`, footprint
`LDGSTS.E.BYPASS.128 x128`), `PROBE PASS asm_cg_A` / `asm_l2_A` (inline PTX `.cg` / `.cg.L2::256B`, footprint
`LDGSTS.E.BYPASS[.LTC256B].128 x128`), `PROBE FAIL asm_cg_B` (linear smem address without the swizzle applied:
footprint right, data wrong — [`ptx_escape`](ptx_escape.md) gotcha 5).

### `.L2::256B` prefetch width (certified: KEEP evaluation = footprint on the dossier's `cache_ops` axis + gate window)

**The knob**. The DSL's `LoadCacheMode` has no such member; it can only be changed in the inline PTX string (form
2). Per the ISA grammar the qualifier goes after the two state spaces and before the operand list, orthogonal to
`.cg` (the cache-mode bit) and the `src-size` zero-fill operand
([`cp_async_predicated_zfill`](cp_async_predicated_zfill.md)) — changing it touches neither of those two, nor the
operands and constraint string:

```python
# only one extra qualifier in the PTX string; the $0/$1 addresses, $2 = src-size register, and the "r,l,r" constraint string are all unchanged
"cp.async.cg.shared.global.L2::256B [$0+%d], [$1+%d], 16, $2;" % (dst_off, src_off)
#                        ^^^^^^^^^^ .level::prefetch_size
```

**Footprint** (dossier `cache_ops` axis, before → after, verbatim):

| Before | After |
|---|---|
| `LDGSTS.E.BYPASS.128` | `LDGSTS.E.BYPASS.LTC256B.128` |

`.BYPASS` unchanged, the `src-size` operand unchanged, just one extra `.LTC256B` suffix. Note that when the
contract signature compares the cache **mode** class it strips `.LTC<n>B`, so this footprint is only visible in
the **raw reading** of the `cache_ops` axis, and the contract-axis diff will not report a `!` for it on its own —
acceptance looks at the raw line.

**Load-bearing conditions**. It is a latency-side knob, invisible in values ([`ptx_escape`](ptx_escape.md)
gotcha 7), and can only be priced by the same-job before/after gate window, while its value varies with the tree:

| State of the tree | Reading |
|---|---|
| `.cg` just switched on from `.ca`, L1 just squeezed out by smem (previous section's probe) | adding `.L2::256B` on top of `.cg`: +0.5%, ≈ 0 within the noise band |
| `.cg` already in place, indices prefetched across bodies, four-phase half-slot single-buffer handoff, 1 CTA/SM (smem 214 KB) × 12 warps, landed in the same patch bundle as the mbarrier **spin wait encoding** ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) form (e)) | gate window 2.0927 → 2.0757 ms (−0.81%); `dram_bytes_read` 1.921e9 → 1.908e9; `stall_long_sb` 4.27 → 4.24; `tensor_pct` 61.56 → 62.12 |

The two knobs of the second row landed in one bundle, and the reading **cannot be split** to give `.L2::256B` on
its own; what it proves is "on a tree where `.cg` has long been in place and supply and waits are already tight,
this bit still has a non-zero reading", not "a general −0.8%". Pricing it alone requires a same-job A/B.

**Certification**: KEEP evaluation — correctness passes (the qualifier does not change the value path), NCU
dossier census of the `cache_ops` axis before and after as above, gate window as above.

### Mixed cache modes: one `.ca` scalar mixed into a large `.cg` chunk (same commit group, same completion signal)

`.ca` / `.cg` is not a **mutually exclusive** "pick one per copy" choice: a thread can, in **the same open cp.async
group**, issue both 16 B `.cg` chunks (K-row gather / ordinary bulk rows) and one 4 B `.ca` scalar (a per-row fp32
factor or the like), and the group completion signal (`commit_group` / `wait_group`, or
`cp.async.mbarrier.arrive.noinc`) covers both. The `cache_ops` axis therefore shows both `LDGSTS.E.BYPASS.128`
(the chunks) and `LDGSTS.E` (the 4 B scalar). The two suffixes each say their own thing: `.BYPASS` is the
footprint of `.cg` (bypass L1), `.128` is the 16 B width — a bare `LDGSTS.E` has neither `.BYPASS` (it is `.ca`,
does not bypass L1) nor a width suffix (32-bit). **Don't** read "no `.BYPASS`" as a consequence of the width: a
16 B `.ca` has no `.BYPASS` either (table above, "SASS suffix: none").

**Why `.ca`, and why it is useful**: a 4 B scalar going through `.ca` is not a choice but an ISA constraint —
PTX's `cp.async.cg` only accepts cp-size 16, so 4 / 8 B copies can only be spelled `.ca`; the completion signal's
coverage is independent of the cache mode, and every `cp.async` in the group is counted by the same
`commit_group` / arrive. The real gain is in "joining the same group": a per-row factor that would otherwise be
fetched separately (LDG→STS + its own wait) now "rides" on the chunk group and is counted by the group completion
signal on landing, saving one round trip and one separate drain. `.cg` meanwhile lets the 16 B chunks bypass L1
(see the load-bearing condition in "Cache qualifiers" above). **The two cache modes do not interfere**: the same
thread, the same group, both modes coexisting is legal, and both lines are in the footprint.

**Spelling** (two inline PTX instructions, each doing its own thing, joining the same group):

```python
@dsl_user_op
def cp_async_ca_4b(dst_smem, src_global, *, loc=None, ip=None):
    """4-byte `.ca`: a single f32 element gmem→smem. Joins the calling thread's open cp.async group."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip),   # shared-window address (Int32)
         cutlass.Int64(src_global).ir_value(loc=loc, ip=ip)],# gmem byte address (Int64)
        "cp.async.ca.shared::cta.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

# 16 B chunks go through .cg (DSL CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL) or inline .cg)
...
cp_async_ca_4b(di_dst, di_src)      # 4 B .ca scalar
cute.copy(tiled_cg, ...)            # 16 B .cg chunk
cp_async_commit_group()             # both in the same group, covered together by the group completion signal
cp_async_wait_group(0)
```

**Pitfalls**:
- The 4 B copy has **neither `.128` nor `.BYPASS`** (the former because it is 32-bit, the latter because it is
  `.ca`) — on the `cache_ops` axis it is registered as bare `LDGSTS.E`, while the chunks are
  `LDGSTS.E.BYPASS.128`. When checking the footprint look for both; don't search only for `.BYPASS`; and don't
  write `.cg` for the 4 B one — the PTX ISA does not allow it (`.cg`'s cp-size is 16 only).
- The completion signal **covers the 4 B scalar too**: the exact moment the 4 B `.ca` lands need not coincide with
  the chunks, but the group completion semantics guarantee release only when "all copies of the group have
  landed" — so the dI scalar needs no separate drain. This requires its `cp.async` to really be in the same commit
  group (issued before the `commit_group`).
- Address types each to their own: gmem uses `Int64` (`Pointer.toint()`), smem uses `Int32` (the shared-window
  address; the DSL rejects generic 64-bit pointers, see [`ptx_escape`](ptx_escape.md)).

**Certification**: a footprint certification, not a performance certification — the same mechanism was KEPT once
on each of two trees of the same operator; both times the `cache_ops` axis went from containing only
`LDGSTS.E.BYPASS.128` to containing both `LDGSTS.E` (4 B `.ca`) and `LDGSTS.E.BYPASS.128` (16 B `.cg`), and
correctness passed both times (the value path unchanged); gate window the first time 26.1272 → 26.2268 ms (+0.38%,
bundled with an arrive-on-completion change), the second time 12.5636 → 12.703 ms (+1.11%, bundled
with a lane-owned index change) — **slightly negative both times, and not separable either time**.
This section only proves "the two cache modes in one group are legal, footprint as above", not how many
milliseconds it is worth.

11. **There must be no `from __future__ import annotations` in the file** (same as `wgmma_ss_min_gemm`
    gotcha 8).
12. **The parent process of a certification script must never import the layer under test.** This is not a GPU
    mechanism but a pitfall of **certification method**; this recipe burned 16 probe rounds on it, and it is
    written here so the next recipe does not repeat it.

    The driver's feedback channel is **asymmetric** (see the probe feedback path of
    `src/forgekernel/loop/driver.py` and `parse.py`):
    - `PROBE FAIL` → only `max_diff=...` is passed back; **the log tail is discarded**, the traceback is invisible;
    - `NO VERDICT` (not a single `PROBE` line in the log) → the last 25 lines / 2000 characters of the log are
      passed back;
    - `max_diff=unreported` with **no reason clause** = every verdict line is FAIL, and not one line carries a
      float.

    Hence two rules:
    - **The parent process uses only the standard library.** Push `import cutlass` / `torch` / `numpy`,
      `@dsl_user_op` decorators, and any module-level DSL evaluation (things like `cutlass.BFloat16.width`) into
      function bodies of the child process. When the parent imports the layer under test at module level, an import
      failure **kills the whole probe before the first rung runs**, and it then falls into the trap below.
    - **If a rung died, don't print a verdict line.** Printing `PROBE FAIL max_diff=n/a` for a rung that never ran
      turns a diagnosable crash with your own hands into an undiagnosable numeric failure: it is classified as FAIL
      → the tail is discarded → the next round only knows "some number is wrong", and goes looking for a numeric
      bug that does not exist. The right way is **all or nothing**: if any rung produced no result, print not a
      single `PROBE` line, and instead print a compact diagnosis at the end of the log (it must fit in 25 lines),
      so it is classified as NO VERDICT and the tail is passed back. The reverse — "dead rungs silent, live rungs
      print PASS" — **must never be done**: it would be a false certification.
    - Also: `gpu_entry.sh` has `TASK_TIMEOUT=600`, and the sum of the subprocess timeouts of all rungs must be
      **clearly less** than 600 s, otherwise the last rung is certain to be KILLed, again showing up as no result.

13. **For a `.cg` row-wise gather, the lane→row mapping decides L2 sector efficiency: a form where one lane owns
    a whole row and walks it in 16 × 16 B makes every `LDGSTS` use only half of each 32 B sector it touches,
    doubling L2 read sector requests.** The `THREADS_PER_ROW` of this entry's minimal example (one row fetched
    jointly by several adjacent lanes) exists for exactly this; don't degenerate it to `layout(1)` for the sake of
    "one row per lane, simple address chain".

    - **Mechanism**: one warp-level `LDGSTS.128` is 32 lanes × 16 B; which 32 B sector each lane lands in is decided
      by the thr/val layout. thr `layout(1)` + val `layout(row elements/slice)` = one row per lane, adjacent lanes a
      whole row (256 B) apart, so each lane's 16 B covers only **half** of its sector; `.cg` bypasses L1, so the
      other half has no L1 line to hit and is requested from L2 again by the same lane's next instruction —
      `lts__t_sectors_op_read` ≈ 2× the payload. Under `.ca`, L1 absorbs this refetch, so the pitfall **appears
      only with `.cg`**, and `.cg` is exactly the mode to use when smem squeezes out L1 (the load-bearing condition
      of the section above): the two recommendations only hold together.
    - **Correct form**: adjacent lanes land on the two 16 B halves of the same sector. With a 256 B row width = 16
      lanes jointly fetch one whole row, one warp instruction fetches 2 rows, and one lane owns "rows of this group
      ÷ number of row teams" rows: the tiler is exactly one row,
      `make_tiled_copy_tv(atom, cute.make_layout(16), cute.make_layout(8))`, `get_slice(tidx % 16)` selects the
      16 B column, `tidx // 16` selects the row family. The `LDGSTS` count per thread, bytes, `.cg` and the
      swizzled destination are all **unchanged** — just a redistribution of the same copies; the minimal change of
      2 lanes per sector (secpair) already gets most of the gain, and 16 lanes per whole row is the upper bound.
    - **Symptom / acceptance** (read straight from NCU): `lts__t_sectors_op_read` ≈ 2× the theoretical payload;
      `lts__throughput` close to saturation with `xbar2lts` as the top breakdown item; high `mio_throttle` on the
      consumer side — the producer's `LDGSTS` and the consumer's `LDS`/`STSM`/`UBLKRED` share the MIO issue ports
      of the four sub-partitions and the SM↔L2 port, the extra sector requests queue them behind, and the consumer
      is taxed **without explicitly waiting for the gather results**. It is an **exposure** ledger, not a
      **volume** ledger: instruction count, tensor cycles and DRAM bytes do not move.
    - **Certification** (single variable: only the lane→row mapping changes, `LDGSTS` count ratio 1.00, values
      identical): pricing probe (base / secpair / row16
      alternating in the same job, NCU clock + counters) — row16 `lts__t_sectors_op_read` **0.58×**
      (theoretical lower bound 0.5×), NCU clock **+17%**, with `stall_mio` and `l2_requests` dropping sharply in the
      same direction. Correctness contract probe (11/11 PASS): the gathered rows are **bit-equal** to `source table[clamp(idx, 0)]` (a raked val
      layout, swapped column/row-family roles, a swizzle that does not follow the row slice would all push it away
      from 0), and the 4 B `.ca` scalar in the same group is still covered by the group completion signal. The
      landing KEEP on that tree: −13.7%.
    - **Cost**: each lane holds several row indices instead of 1 (a few more `LDS`, smem wavefronts go up); spill
      rises when the producer's register quota is tight. The gain above already includes this bill; on a tree with
      registers already at the limit, work out the register split first.

## Sources

- **Public infra module** (a vendored CuTeDSL utility library, not operator source:
  `chains/dsa/vendor_cudnn/cudnn/deepseek_sparse_attention/utils/copy.py:22-43`, `tiled_copy_1d` /
  `tiled_copy_2d`): `cpasync.CopyG2SOp()` and `cute.nvgpu.CopyUniversalOp()` go through **the same**
  `make_copy_atom(op, dtype, num_bits_per_copy=...)` + `make_tiled_copy_tv(atom, thr_layout, val_layout)`
  construction — in the source it is the single line `copy_op = cpasync.CopyG2SOp() if is_async else
  cute.nvgpu.CopyUniversalOp()` (`is_async` swaps only this one op object), plus the 128-bit cap of
  `num_copy_bits = min(128, num_copy_elems * dtype.width)`.
  ⚠️ **Reverse record**: not finding `CopyG2SOp` by searching `hw_kb/` makes one believe "the DSL has no non-bulk
  G2S atom; only raw PTX works" (a student's notes in one run recorded exactly that, and switched
  to inline PTX `cp.async.cg.shared::cta.global` on that basis). That conclusion came from **too narrow a search
  scope**: `cpasync` indeed only has bulk **tensor reduction** ops ([`s2g_bulk_reduce`](s2g_bulk_reduce.md)
  gotcha 1 is about that), but the **copy** op `CopyG2SOp` does exist. When searching for an interface, search the
  vendored infra too.
- **PTX ISA**: the 16 B copy of `cp.async` and the group completion semantics of `cp.async.commit_group` /
  `cp.async.wait_group N`.
- **Certified recipes of the same family**: [`ptx_escape`](ptx_escape.md) (the **general rules** of the inline-PTX
  calling convention and the expressibility ladder: `@dsl_user_op` / `AD_ATT` + `$n` / `has_side_effects=True`
  mandatory, the swizzled smem address pitfall, footprint checks), [`s2g_bulk_reduce`](s2g_bulk_reduce.md) §1,
  [`elect_one`](elect_one.md) gotchas 2/6 (warp scope, the line between per-thread and single-issue instructions),
  [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) gotchas 1/6/8 (async-proxy fence, logical coordinates for swizzled
  smem, `__future__` annotations).
- **GPU-certified** (gather probe) — three rungs: R0 sync-op control group (proves
  the partitioning/index/smem/read-back scaffolding works), R1 `cp.async` row-wise gather (32 rows, runtime
  non-monotonic repeatable indices, two committed batches + `wait_group(1)/(0)`, compared element by element
  against numpy `src[idx]`), R2 issue footprint (16 B `cp.async` in PTX, `LDGSTS` in SASS, or `nvvm.cp.async.*` in
  NVVM IR, taking the OR of the three, ruling out the scalar fallback; one compile feeds both the env-var and
  `options=` dump sinks — see gotcha 1's record of the contradiction — and when all three instruments are mute it
  reports an **instrument failure**, not a mechanism failure). Each rung runs in its own subprocess, and the verdict
  lines carry rung labels, so a failure can report itself as which rung, and whether it is a numeric problem or
  never ran at all.
- **GPU-certified (gotcha 13)**: pricing probe (three forms
  alternating in the same job, NCU clock + `lts__t_sectors_op_read` / `l2 requests` / `mio_throttle` counts) and
  correctness probe (G an independent kernel reading back the
  gather destination slot through the same swizzled view as the WGMMA descriptor / T closed-form solution on a
  micro geometry / X frozen criterion on the evaluation geometry / S full-size replay, one subprocess per rung).
