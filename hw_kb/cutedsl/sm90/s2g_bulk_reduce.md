# s2g_bulk_reduce (`cp.reduce.async.bulk` S2G ADD: whole-block atomic reduction smem → gmem)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers the bulk path on SM90 for
> "several CTAs accumulating fp32 into the same block of gmem": the **non-tensor** `cp.reduce.async.bulk` (no
> descriptor, no swizzle constraint) + the staging smem form it requires + the fence / commit / wait / reuse order.
> It and [`tma_store_epilogue`](tma_store_epilogue.md) (S2G **overwrite**) are the two end points of the same data
> path: that one writes the tile down, this one **adds** the tile down. The staging half directly reuses the flat TV
> bijection of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md).

## Purpose

**Add a block of SMEM element by element** into the same block of gmem addresses, moved in one instruction by the
bulk copy engine:

```
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32  [dstMem], [srcMem], size;
```

One instruction covers `size` bytes (measured in this entry: 16384 B = 4096 fp32), with the semantics of one
`.relaxed.gpu` read-modify-write addition per element on the gmem side. It replaces "one `red.global.add.f32` /
`atomicAdd` per thread": for the same bytes, the issue count collapses from "several per thread" to **one per CTA**,
and it takes no LSU issue slots (it goes through the bulk engine, recorded in SASS as `UBLKRED.G.S.ADD.F32.RN`).

## When needed (trigger)

- Dataflow: the fp32 accumulation results of several CTAs must land in **the same gmem region** (split-K / several
  CTAs sharing one accumulation scratch / any "adding partial sums"), and the gmem side must accumulate, not overwrite.
- NCU / SASS symptoms: a high share of non-MMA cycles with normal `dram_bytes`, `spill = 0`, `bankconf = 0` — i.e. the
  byte volume is right but the **issue count** is wrong. Seeing swathes of `RED.E.ADD.F32` (per-thread scalar or v2
  atomics) in SASS instead of `UBLKRED.G.S.ADD.F32.RN` is the machine-readable signature of this interface being
  absent. Note: accepting by a byte band such as `dram_bytes_write` **cannot detect** this difference — both forms
  move the same bytes.
- Reverse trigger (when **not** to use it): with only one writer and no accumulation needed on the gmem side, use the
  ordinary TMA store of [`tma_store_epilogue`](tma_store_epilogue.md), not the reduction variant.

### Two variants; pick the right one first

| | tensor variant | **non-tensor variant (this entry)** |
|---|---|---|
| PTX | `cp.reduce.async.bulk.tensor.Nd...` | `cp.reduce.async.bulk...` |
| CuTe DSL | `cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)` + `make_tiled_tma_atom` | **not wrapped by the DSL**, needs inline PTX |
| gmem addressing | TMA descriptor (multi-dimensional coordinates) | raw address + byte count (one-dimensional) |
| smem layout constraint | must be a descriptor-legal layout (swizzle level, box shape) | **none**: any layout, byte order = gmem order |
| Applies to | gmem tile with multi-dimensional strides, needing TMA for bounds/swizzle | the gmem target block is **contiguous**, and you want to pick the staging layout yourself |

The reason to pick the non-tensor variant is **layout freedom**: it adds the `size` bytes of smem in linear order, as
is, onto the `size` bytes of gmem, so the staging buffer can be the **flat TV layout** of
[`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md) — the spelling with one 128-bit STS per thread and zero bank conflicts.
The price: the element order in the gmem scratch is **TV order** rather than logical (m, n) order, and the reader
must read back with **the same** TV bijection (when the scratch is your own intermediate, that is not a cost, just a
convention). Conversely, the tensor variant forces staging through a descriptor-legal swizzled 2-D layout, and when an
fp32 C-fragment is written into such a layout `get_smem_store_op` can only degrade into a 32-bit scalar store (SM90's
`stmatrix` exists only in 16-bit form), which is where the bank conflicts come from.

## Minimal runnable example

Single warpgroup (128 threads), fp32, `TILE_M=64`, `D=64` → 4096 fp32 = 16384 B per slot.
`NUM_CTA` CTAs accumulate into the same gmem block. The main loop is stripped out: `acc` is faked in registers with
static identity coordinates (as in gotcha d of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md)).

### 1) The three PTX instructions the DSL does not wrap

The `llvm.inline_asm` spelling copies the form cutlass itself uses in `cute/nvgpu/cpasync/helpers.py` to emit
`fence.proxy.tensormap::generic.*`: `@dsl_user_op` handles `loc`, and operands use `Pointer.toint()`
(**gmem → Int64 / smem → Int32**, see `cute/core.py::Pointer.toint`).
The general rules of the calling convention (placeholders / constraint string / side effects / swizzled smem addresses
/ footprint check) and the ladder "check the op signature before writing asm" are in [`ptx_escape`](ptx_escape.md);
this entry only covers these three instructions themselves.

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def bulk_reduce_add_f32(dst_addr, src_addr, nbytes, *, loc=None, ip=None):
    """dst_addr: Int64 (global); src_addr: Int32 (shared::cta window address).
    nbytes a multiple of 16, both addresses 16-byte aligned. Issued by a **single thread**."""
    llvm.inline_asm(
        None,
        [dst_addr.ir_value(loc=loc, ip=ip),
         src_addr.ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[$0], [$1], $2;",
        "l,r,r",                                  # order matches $0,$1,$2
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bulk_commit_group(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bulk_wait_group_read(n, *, loc=None, ip=None):
    """`.read`: wait until all groups except the most recent n have **finished reading the source SMEM** (!= finished writing gmem)."""
    llvm.inline_asm(None, [],
                    "cp.async.bulk.wait_group.read {};".format(int(n)), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

### 2) Staging: flat layout + flat TV (copied from flat_tv_r2s_cfrag)

```python
F32 = cutlass.Float32
TILE_M, D, NUM_WG = 64, 64, 1
SLOT   = TILE_M * D // NUM_WG          # 4096 fp32
NBYTES = SLOT * 4                      # 16384 B -- a multiple of 16

# a linear slot, no swizzle, no quad interleave: smem byte order == gmem byte order.
flat_l = cute.make_layout((SLOT, NUM_WG), stride=(1, SLOT))

# 4 consecutive fp32 per thread = one 128-bit STS, a warp spans 32 banks -> 0 conflicts.
tv = cute.make_tiled_copy_tv(
    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32,
                        num_bits_per_copy=128),
    cute.make_layout((128, NUM_WG)),   # thread layout
    cute.make_layout(128 // F32.width))  # value layout = 4 fp32
```

SharedStorage uses `cute.struct`, alignment ≥ 16 (the example takes 1024, which also satisfies the customary alignment
of the TMA family):

```python
@cute.struct
class SharedStorage:
    sflat: cute.struct.Align[cute.struct.MemRange[F32, cute.cosize(flat_l)], 1024]
```

### 3) Kernel: staging → fence → barrier → single-thread issue → commit → wait → barrier

```python
@cute.kernel
def kernel(mScratch: cute.Tensor, tiled_mma: cute.TiledMma,
           tv: cute.TiledCopy, flat_l: cute.Layout,
           SharedStorage: cutlass.Constexpr, num_issue: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sflat = storage.sflat.get_tensor(flat_l)      # no swizzle -> no swizzle=

    # acc is produced by the main loop's WGMMA; here static (m, d) coordinates illustrate where its shape comes from.
    thr_mma = tiled_mma.get_slice(tidx)
    cId = thr_mma.partition_C(cute.make_identity_tensor((TILE_M, D)))
    acc = cute.make_fragment(cId.shape, F32)

    # hoist every loop invariant out of the issue loop (flat_tv gotcha b). Addresses taken only once.
    thr_tv   = tv.get_slice(tidx)
    tS       = thr_tv.partition_D(sflat)
    tR       = cute.make_tensor(acc.iterator, cute.make_layout(tS.shape))
    dst_addr = mScratch.iterator.toint()          # Int64 (gmem)
    src_addr = sflat.iterator.toint()             # Int32 (smem window)

    for issue in cutlass.range_constexpr(num_issue):
        # ... fill acc / main-loop output ...
        cute.autovec_copy(tR, tS)                 # one 128-bit STS per thread
        # the generic-proxy STS must be visible to the async proxy, and all lanes must have written,
        # before the single issuing thread may fire. Both are required.
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                              space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier()
        if tidx == 0:                             # a single thread, not a whole warp (see gotcha 3)
            bulk_reduce_add_f32(dst_addr, src_addr, NBYTES)
            bulk_commit_group()
            bulk_wait_group_read(0)               # the source SMEM may now be reused
        cute.arch.barrier()                       # let everyone know about the above (see gotcha 5)
```

launch: `kernel(...).launch(grid=[NUM_CTA,1,1], block=[128,1,1])`;
the gmem target block is the same address range shared by the `NUM_CTA` CTAs — exactly the accumulating semantics
wanted.

### 4) Read side: read back with the same TV bijection

The gmem scratch is in **TV order**, so the reader (a later segment of the same kernel, or a separate postprocess
kernel) must use **exactly the same** `tv` and `(SLOT, NUM_WG)` layout:

```python
gflat = cute.make_tensor(mScratch.iterator,
                         cute.make_layout((SLOT, NUM_WG), stride=(1, SLOT)))
tG = tv.get_slice(tidx).partition_S(gflat)        # static shape -> autovec_copy accepts it
rF = cute.make_fragment_like(tG, F32)
cute.autovec_copy(tG, rF)
acc_back = cute.make_tensor(rF.iterator, acc.layout)   # linear reinterpret back into the (m,d) view
```

## Gotchas

1. **CuTe DSL only wraps the tensor variant.** The `__all__` of `cpasync` only has
   `CopyReduceBulkTensorTileS2GOp` (corresponding to `atom_make_non_exec_tiled_tma_reduce` in `make_tiled_tma_atom`);
   there is **no** non-tensor S2G bulk reduce copy op. A descriptor-free whole-block reduction can only go through
   inline PTX. Do not look for `cp_reduce_async_bulk` in `cute.arch` — it only has `cp_async_bulk_commit_group` /
   `cp_async_bulk_wait_group` (and naming differs between DSL minor versions; inline PTX actually carries no version
   risk).
2. **Alignment and granularity**: `size` a multiple of 16, `dstMem` and `srcMem` both 16-byte aligned (the general
   requirement of the `cp.async.bulk` family). The smem side relies on `cute.struct.Align[..., N]`; the gmem side on the
   tile itself being contiguous with a start offset that is an integer multiple of 16 B. Violations raise no error —
   wrong data moved / silent errors.
3. **Single-thread issue, not a whole warp.** This is the opposite of gotcha 6 of
   [`tma_store_epilogue`](tma_store_epilogue.md), and the reason is in that very gotcha: the DSL's TMA **copy atom**
   does a warp-level elect internally (membermask = whole warp), so that path must be `if warp_idx == K:`; here it is
   **raw PTX**, `cp.reduce.async.bulk` itself is a per-thread instruction, and issuing it from several threads means
   several reductions (the numbers simply double, a silent error). Use `if tidx == 0:` (or pick one lane with
   `cute.arch.elect_one()`).
4. **`.relaxed.gpu`, not "ordered".** Each element's read-modify-write is a GPU-scope relaxed atomic (safe to
   accumulate across CTAs, no lost updates), but there is **no ordering guarantee across CTAs at all**, and the
   floating-point summation order is nondeterministic. To get `max_diff == 0` in validation, use values that are
   "exact under any summation order" (e.g. integers all < 2^24); comparing random fp32 for bitwise equality is asking
   for flakiness.
5. **Reusing the staging buffer takes two steps, not one.** `wait_group.read 0` only lets **the one issuing thread**
   know the source SMEM has been read; the other threads are still in the previous round. To issue a second time on the
   same buffer, there must be another `cute.arch.barrier()` after the wait, otherwise other threads' STS overwrite
   bytes the bulk engine is still reading (silent error).
   Full order: `STS → fence_proxy(async_shared, shared_cta) → barrier → [single thread] reduce +
   commit_group + wait_group.read → barrier`.
6. **`.read` vs without `.read`.** `cp.async.bulk.wait_group.read N` waits for "the source SMEM has been read" (the
   buffer can be reused); without `.read` it waits for "the whole group is complete" (visible on the gmem side too).
   `.read` is enough for reusing the scratch; for **others** to read the gmem result, a stronger visibility means is
   needed (naturally satisfied across a kernel boundary). Before the kernel exits, at least wait out the in-flight
   groups, otherwise the engine is still reading when the SMEM is reclaimed.
7. **The gmem scratch is in TV order, not logical (m, n) order.** This is the price of the layout freedom bought by
   choosing the non-tensor variant. The writer and reader must instantiate **the same**
   `make_tiled_copy_tv((128, NUM_WG), val=4)` and the same `(SLOT, NUM_WG)` layout (as in gotcha c of
   [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md)); a different spelling on either end scrambles silently. If this gmem
   block must be read **externally** in its natural layout, flat TV staging cannot be used — only then fall back to the
   tensor variant (and accept the bank conflicts of fp32 staging).
8. **There is no "zero-conflict stmatrix path" for fp32 C-fragment → swizzled smem.** SM90's `stmatrix` exists only in
   16-bit form, so `hopper_helpers.get_smem_store_op(..., elem_ty_d=Float32, ...)` falls back to `CopyUniversalOp`
   (32-bit scalar store). Writing a WGMMA fp32 C-frag through its native `partition_C` into swizzled 2-D staging puts a
   warp's 32 stores on only a few banks (the quad interleave of the C-frag makes 32 lanes cover only a narrow strip of
   8 rows × 4 columns) — that is where "after switching to bulk reduction, `bankconf_st` exploded from 0 to tens of
   millions" comes from. Zero conflicts require switching to **flat TV**, and only the non-tensor variant can take flat
   TV.
9. **The usual inline-asm mines**: the constraint string order must correspond one-to-one with `$0/$1/$2` (`"l,r,r"`);
   `has_side_effects=True` must not be omitted (otherwise it gets DCE'd, showing up as "nothing happened"); the asm
   text uses `$` placeholders (cutlass uses `AsmDialect.AD_ATT` + `$0` style), and `%0` fails to parse.
10. **File level: no `from __future__ import annotations`** (same as gotcha 13 of
    [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) / gotcha 10 of [`tma_store_epilogue`](tma_store_epilogue.md)).

## Serial-path form of high-volume issue (multiple chains / predicates / clustered addressing / payload granularity)

The minimal example above only covers the **single-shot** form "one CTA, one staging buffer, one `UBLKRED`". A real
kernel often has **dozens to hundreds** of `UBLKRED`s per reduction (one per row), and their **issue organisation** —
i.e. "what is allowed to appear between two adjacent `UBLKRED`s" — is the other side that decides performance. The
four lessons of this section all transfer across operators; the readings in parentheses come from round-by-round
restructuring of one bulk-reduce write-back.

### A) How to dispatch many reductions: split by warp, not one thread serially

`cp.reduce.async.bulk` is a **single-thread-issue** instruction (gotcha 3), but **several** `UBLKRED`s can be issued by
**several threads** each issuing its own: split the address space to be reduced evenly by thread / by warp, **each one
issued by exactly one thread**, each with its own `commit_group` + its own drain. The serial form (one thread issuing
everything) gates the issue of the whole reduction chain on a single thread, with `stall` proportional to the issue
count; after splitting by warp into N parallel 32-deep chains, the issue count is unchanged but **the issue window is
spread over N warps**. There is only one dispatch rule: **every reduced row is issued exactly once by exactly one
thread** (the partition neither misaligned nor duplicated); only then are the numbers the same as serial.

- The parallelism (4 chains 32 deep vs 1 chain 128 deep) is an **issue-organisation** question, not a numerical one;
  four chains whose address partitions cover `[0, BN)` without overlap are correct.
- **Each warp closes its own bulk groups and drains its own groups**: `commit_group` / `wait_group` are per-thread, so
  each chain's writer commits and waits (`.read`) on its own; do not share one wait across warps — that gates the
  drain on the slowest chain.
- **The lever is "how many warps are issuing", not "whether the loop is rolled / whether addresses are vectorised".**
  On the same baseline two changes were each KEEP once: (a) still one warp per consumer, sites collapsed into a static
  station inside a rolled runtime loop, addresses broadcast per trip with `shfl.sync.idx`, firing under a per-lane
  predicate in the asm — the serial path was fixed cleanly, yet the reading was slightly negative (the issue window is
  still gated on two threads, and the loop and per-trip `SHFL` are net cost); (b) four warps per consumer each driving
  one fully unrolled 32-deep chain, each closing its own groups, without fixing a single site, got about half (§A of
  this section). Split warps first, then fix the serial path (§B / §C are the second step on top of (b)).

### B) Block invalid rows with an **instruction predicate**, not a branch

The set of reduced rows often carries invalid markers (out-of-bounds rows / `idx < 0` / masked rows). Writing "skip
invalid rows" as a positive branch `if ridx >= 0:` puts an `ISETP→BRA` of its own in front of every `UBLKRED`, and
ptxas **does not hoist an issue across a basic-block boundary** — so every pair of UBLKREDs on the issue path has a
dependency chain (`LDS→ISETP→BRA`) between them. Write the guard as **an instruction predicate inside the same asm
block**:

```python
@dsl_user_op
def bulk_reduce_add_f32_pred(dst_addr, src_addr, nbytes, idx, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
         cutlass.Int32(idx).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "setp.ge.s32 p, $3, 0;\n\t"
        "@p cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[$0], [$1], $2;\n}",
        "l,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

Effect: between two adjacent `UBLKRED`s **only address arithmetic remains** — no LDS, no ISETP, no BRA. Predicate
off == not issued, an invalid row contributes exactly 0, and its (never dereferenced) target address does not matter.
The numbers are the same as the branch form, but the issue path no longer has basic-block boundaries, and `UBLKRED`s
can issue back to back.

### C) **Cluster-prefetch** the target addresses (row indices) at the head of the chain

Each `UBLKRED`'s destination address comes from a runtime row number (the gmem target row); read **all the row
indices this chain needs at once** at the head of the chain (a row of mutually independent `LDG/LDS` that can pipeline
with each other), rather than "reading one right before each reduction". This is the same rule as
[`gather_index_prefetch`](gather_index_prefetch.md) (batch-prefetch the indices, decouple fetching from addressing),
only applied to the **destination** addresses of bulk-reduce. One clustered read at the chain head replaces "an index
round trip per row" with "one batch read + pure address arithmetic".

### D) Payload granularity: does one `UBLKRED` reduce a **whole row** or a **half row** (staging shared across warpgroups)

The payload's byte count decides **who writes a staging block**:

- **Half row /** each warpgroup reduces only the half it wrote: staging is per-warpgroup private scratch (written and
  reduced by itself, no cross-WG sync needed). But two passes (one half row each) are needed to cover a row.
- **Whole row**: one warpgroup reduces **the whole row**, necessarily reading the half written by the peer (the other
  warpgroup) → staging becomes a **cross-WG resource**, the barriers around it widen from per-WG width (128) to
  **cross-WG width** (256), and the sync points must guarantee "the peer's half row has been written, and my own pass
  has finished reading" before the next pass overwrites it. Widening barriers + rearranging meetings is the
  **correctness** cost of a whole-row payload.

Both have **the same byte volume** (the bytes per reduction are unchanged); they differ in issue count (half row: two
per row vs whole row: one per row) and staging ownership. Reading how many bytes one `UBLKRED` covers in SASS, or the
`merge`/barrier-width axes of the dossier, tells which form it is; **do not** check by `dram_bytes_write` — both forms
move the same bytes.

### E) The issue point of the reduction can be **deferred into the next block's flight**

The issue point of bulk-reduce need not stay at "the tail of the current block, right where staging was written".
Move it **into the flight of the next block's TMA/MMA chain**, so its **acceptance latency** (the time for the TMA
engine to accept the issue) is covered by the next block's flight, rather than exposed bare between "staging written →
end of block". The last block's issue moves into the epilogue. This is the **write-back-side counterpart** of the idea
of [`midchain_stage_wait`](midchain_stage_wait.md), "hide the wait inside an in-flight chain": what is waited on is no
longer the MMA's completion but **the reduction's acceptance**. When moving it, the sync points move too: the meeting
that used to mean "staging has been read" must guarantee that **all** readers have finished before it is overwritten,
and the drain (`wait_group.read`) must also follow the issue. **The correctness failure mode is not wrong numbers but a
race or a hang** — so the A/B must go through a race probe with "repeated launches on the same input", not a single
launch.

**Verified sources**: the machine-readable footprints and race probes of the five items above (split by warp, block
invalid rows with a predicate, clustered addressing, whole-row vs half-row payload, deferring the issue into the next
block's flight), certified by two GPU probes (not included in this release):
a reduce-move/payload probe (predicated issue + whole-row payload + clustered
addressing / deferred issue, `PROBE PASS` 15/15; race probe 12 launches, the two outputs not going through bulk-reduce
bit-for-bit identical, the drift of the bulk-reduce output at reordering scale) and
a gather-cadence probe (`PROBE PASS` 11/11).

## bf16 variant (`cp.reduce.async.bulk` per-row ADD, non-contiguous gmem tile)

The minimal example and the five "high-volume issue" items above all use f32 (`UBLKRED.G.S.ADD.F32.RN`). bf16
(`UBLKRED.G.S.ADD.BF16.RN`, asm suffix `.add.bf16` or `.add.noftz.bf16`) is **another dtype** of the same interface,
with two constraints of its own — these two do not exist for f32 and are bf16-specific:

### 1) PTX suffix: `.noftz` is **required** on 16-bit floating-point reduction types

The bf16 reduction uses `.add.noftz.bf16`. **`.noftz` is a required qualifier on 16-bit floating-point reduction
types** (PTX ISA: 16-bit floating-point reduction types must state flush-to-zero semantics explicitly), not a matter
of luck "trying which one lowers" — treating it as a spelling to try produces an instruction that should simply have
been written right. What does need the expressibility ladder is a **different** problem (measured with this tool: all
six spellings of the tensor reduction variant `cpasync.CopyReduceBulkTensorTileS2GOp` crash inside `cute.compile`,
judged not to lower in this DSL; the non-tensor variant went through on the first try). So the order is: **first
write the suffix the ISA requires, and only if it is not accepted, go to rung B of the ladder and try other
spellings**.

```python
@dsl_user_op
def bulk_reduce_add_bf16(dst_addr, src_addr, nbytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
```

### 2) When the target gmem block is **not contiguous**: one `UBLKRED` reduces **one row**, not one block

The f32 "single-shot" form assumes a contiguous target block (see the "whole-row payload" of §1); the common bf16
target is **per-token rows** (contiguous within a row, with a head stride across rows) — then a (64×128) tile is not one
contiguous address range, and only **one reduction per row** is possible: 128 bf16 per row = 256 B, 64 rows per tile =
64 `UBLKRED`s. The issue organisation copies §A of "high-volume issue" (split by warp, **each chain issued by exactly
one thread, each committing / draining / closing its own groups**). **The number of reductions, the issuer of each
chain and the number of reduced rows are all readings; re-read them on a new tree** — the measured form of this entry
is: **one producer warp (32 lanes)**, 64 rows split evenly over lanes, 2 rows per lane, **each lane committing its own
bulk group and draining it itself** (not "4 warps with 16 rows each", and not "lane 0 issuing 16 rows" — both violate
§A's single-issuer rule and conflict with this entry's gotcha 3, "single-thread issue"). **Do not** check by
`dram_bytes_write` — per-row and whole-block reduction move the same bytes.

### 3) The non-tensor reduction adds smem bytes **linearly** to gmem: the staging layout must match the gmem order

The non-tensor variant adds `size` bytes **as is**, in storage order (§"Two variants; pick the right one first"), so if
the staging block is a swizzled 2-D face (the 128B swizzle path used to remove bank conflicts), the linear address read
by `iterator.toint()` is unrelated — swizzled byte order ≠ the gmem target row order, and the reduction "lines up"
wrongly. A bf16 per-row reduction requires staging in **natural row-major + row pad**: when the row byte count is not a
multiple of the swizzle's 128B, pad so rows stay contiguous between each other and the publishing side's
`STSM.16.M88.4` instructions fill 4 wavefronts. **The reason for the pad is not "rounding up to a multiple"** — the row
itself (128 bf16 = 256 B) already sits on the natural swizzle's 128B grid; the pad is there to **scatter the publishing
side's banks**: one `STSM.16.M88.4` writes 16 rows, and with a row pitch of exactly 256 B these 16 rows **land on the
same bank quad**, one instruction producing a 16-way conflict. **Adding 16 B per row** (`STGP = DH + 8` = 136 elements
= 272 B) staggers adjacent rows by 4 banks, and publishing returns to the 4 wavefronts/instruction of the swizzled
form; meanwhile each row's 128 columns stay contiguous and 16 B aligned, and the reduction side's linear order is
unaffected. **Do not feed a swizzled face's address straight into the non-tensor reduction** — it is an interface with
"no descriptor, no swizzle constraint" (§"Two variants"), any layout will do, but only a linear order you picked
yourself lines up with gmem.

**Verified sources**: the three items above come from a landing that changed store-out from "TMA overwrite at head
granularity + a separate reduction afterwards" to "per-row bulk-reduce ADD inside the kernel" (`PROBE PASS
max_diff=0.0`, all four contracts passed: preferred bf16 / CTAs racing on two target addresses bit-for-bit identical
over 4 launches / the raced points exactly 2.0 per element / the asm ladder can be LOWERED). Dossier `cache_ops` axis
readings: before the change `["UTMALDG.3D", "UTMALDG.4D", "UTMASTG.3D"]`, after the change
`["UBLKRED.G.S.ADD.BF16.RN", "UTMALDG.3D", "UTMALDG.4D", "UTMASTG.3D"]`
(the `UBLKRED` family entering `cache_ops` is defined by `_CACHE_FAMS` of [`orch_tools`](ptx_sass_census_criteria.md);
`UBLKRED.G.S.ADD.BF16.RN` is the **only** signature of `UBLKRED`'s appearance, independent of back edges, visible even
on a fully unrolled tree). Gate window 0.9766 → 0.9593 ms (−1.77%),
**load-bearing condition**: the kernel deleted the whole fp32 intermediate workspace and one kernel (5 launches → 3),
and the staging smem changed from a swizzled face to row-major + pad — not the price of the reduction interface alone,
but the sum of "reduction interface + staging rearrangement + deleting the after-the-fact reduction kernel"; on the
register side `spill_st` was 93696 sectors, non-zero, and the reduction's contribution to the register bill was not
isolated ([UNVERIFIED-BOUNDARY], the same discipline as §"Verified sources").
(KEEP in an eval job (job number not entered in the status record), the `cache_ops` axis readings above.)
**Dossier contract-axis readings** (as footprints, not generic percentages): the multi-chain form (b) of §A (1 chain
128 deep → 4 chains 32 deep) has **no** footprint on the contract axes — on a fully unrolled tree
without back edges, `loops` / `commit` / `supply` are all blind (gotcha 11 of
[`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)); the "first appearance" of `commit` rows belongs to the back
edge introduced by the rolled form (a), not to a footprint of the reduction mechanism. The whole-row payload + deferred
issue of §D/E changed the `mma-consumer` row on the `commit` axis from `{LE 0x0 x1, LE 0x1 x1}` to
`{LE 0x0 x2, LE 0x1 x1}` (in-flight MMA depth down one tier; the `cache_ops` axis did **not** move this round, and its
`LDGSTS.E` is the landing of the mixed cache modes of [`cp_async`](cp_async.md) from an earlier round). **The register
bill of the same KEEP**: `regs_per_thread` pinned at 168 unchanged, while `spill_ld_sectors` / `spill_st_sectors` jumped
from 0 to 2,097,152 (0 the round before, and staying in the tree from then on up to the best), `stall_long_sb` 4.16 →
6.03 — what each of whole-row payload / cross-WG staging / deferred issue pushed into local memory was **not
isolated**, and this section's gate gain was obtained with this spill on board. **[UNVERIFIED-BOUNDARY]** Before
adopting §D/E, look at your own tree's spill reading; on a tree whose registers are already at the ceiling, the
whole-row payload's staging and the live ranges lengthened by deferring the issue may pay in spill first before any
gain, so first split this bill with a same-job A/B.
Row indices, barrier widths and the W split are **readings**; on a new tree re-read them by `mio_throttle` / issue
count.
(The job numbers of the KEEP evaluations are in the index row.)

### 4) The staging view can **alias** an existing swizzled face: take a second view of the same `MemRange` without `swizzle=`, and the publishing STSM writes into it unchanged

The row-major staging of §3 need not open new smem. The **pair of adjacent** 8 KB 64×64 swizzled slots that used to
serve as TMA-store staging (the pair already dead at the publish point), `get_tensor`'d again with a plain layout, is a
64×128 bf16 tile in gmem order:

```python
# (r, c_in_half, half, pair): each pair is 16 KB = two adjacent slots, read as row-major 64x128; element (r, c, mh, p)
# at p*8192 + r*128 + mh*64 + c -- row pitch 256 B, exactly the byte order of one 128-bf16 row in gmem
lyt_flat = cute.make_layout((64, 64, 2, NPAIR), stride=(2 * 64, 1, 64, 2 * 64 * 64))
sFace  = storage.face.get_tensor(lyt_face.outer, swizzle=lyt_face.inner)   # the original swizzled view, as before
sFaceF = storage.face.get_tensor(lyt_flat)                                  # flat view of the same bytes
```

**Not a word changes on the publishing side**: the r2s tiled copy of [`tma_store_epilogue`](tma_store_epilogue.md),
`make_tiled_copy_S(get_smem_store_op(ROW_MAJOR, BF16, F32), make_tiled_copy_C_atom(StMatrix8x8x16bOp(False, 4), mma))`,
just gets a **column half** of the flat view as destination, `thr_r.partition_D(sFaceF[(None, None, mh, pair)])`, with
`acc[i]` landing at `(cS[i][0], mh*64 + cS[i][1])`. The destination view's row pitch is 128 elements rather than 64 (a
non-compact rest mode), and each lane's STSM address is still 16 B aligned (row r of column half `mh` starts at
`r*256 + mh*128`). This is a "thread → address" fact, ruled on by GPU on a view with a **non-zero pair offset** (probe
S1, `max_diff=0.0`; the kind of question only a GPU can answer, as in gotcha 2 of
[`ldmatrix_stmatrix`](ldmatrix_stmatrix.md)).

**⚠ A column half writes the whole slot pair, not one slot**: in the flat view the 64 rows of column half `mh` span
both 8 KB blocks of the pair (rows 0–31 in the first block, rows 32–63 in the second), so the call site requires **both
slots to be dead**; the old precondition "one TMA box per half" needing only that one slot dead is not enough.

Issue (one store warp, 32 lanes × 2 rows, each lane with its own bulk group; same as §2):

```python
sbase = sFaceF[(None, None, 0, pair)].iterator.toint()            # Int32 (smem)
g0 = gbase + I64(row0) * I64(ROW_PITCH_B) + I64(col0 * 2)           # Int64 (gmem); row pitch = the target tensor's row pitch
if wid == STORE_WARP:
    for s in cutlass.range_constexpr(2):
        j = lane * 2 + s
        bulk_reduce_add_bf16(g0 + I64(j) * I64(ROW_PITCH_B), sbase + j * 256, 256)   # 256 B = one row, not 512
    bulk_commit_group()                                             # per-thread group
# the drain (wait_group.read) goes before this slot pair is next overwritten (the two steps of gotcha 5), not necessarily right after the issue
```

Probe S2/S3: two CTAs each add their own tile into the same row block, and the result = prefill + X₀ + X₁ bit for bit
(ADD rather than overwrite, the lane dispatch covering each of the 64 rows once); the adjacent column block is still
the prefill constant (256 B per issue, no spill-over into the neighbouring block). S4: in the PTX every
`cp.reduce.async.bulk` carries `.add.noftz.bf16`.

**The pad is a bank-level refinement, not a correctness precondition**: the 16 B row pad of §3 scatters the STSM's
banks; the flat view without pad at a 256 B row pitch (the one above) is numerically correct and can land first, with
the pad priced separately as a later tier.

**The target must be pre-zeroed, and the zeroing can be charged to an upstream kernel**: the reduction adds rather
than writes, so the first write into the target tensor must be 0. The host-side spelling `torch.zeros` adds one fill
launch; put the zeroing into an earlier kernel of the same pipeline (through an f32 view, one plane per CTA), and the
launch table does not grow.

**Footprint** (dossier `cache_ops` axis, verbatim): `["UTMALDG.2D", "UTMASTG.2D"]` →
`["UBLKRED.G.S.ADD.BF16.RN", "UTMALDG.2D", "UTMASTG.2D"]`. `UTMASTG.2D` **stays** because the same kernel's third output
has no pair to fold into and still goes through a TMA store; the reduction path itself needs no descriptor at all (only
the target tensor's base address), and the descriptors and `tma_partition` of the two folded stores disappear with it.
`loops` / `mbar_per_role` / `edges` moved along in the same round and are not counted.

**Load-bearing condition**: gate window 0.7422 → 0.7258 ms (−2.21%, a cross-job reading at the edge of the node noise
band, not a same-job A/B), the price of the sum "deleting a separate pairwise-sum kernel (2 launches) + two per-partial
intermediate tensors + moving the target zeroing into an upstream kernel + zero new staging smem (aliasing the dead
TMA-store slot pair)", not the price of the reduction instruction itself; on a tree whose smem budget is already at the
ceiling (231424 / 232448 B), "zero new" is a condition for landing at all rather than a gain. The rounding form is
unchanged: one correctly rounded bf16 add, corresponding to the original single rounding after the f32 sum. This KEEP
certifies the footprint on the `cache_ops` axis (`UBLKRED` made it into the cubin, the descriptors of the two folded
stores disappeared), not the price of the reduction path; for a price, first do a same-job A/B.

## Sources

- PTX ISA, `cp.reduce.async.bulk` (non-tensor variant): form
  `cp.reduce.async.bulk.dst.src.bulk_group.op.type [dstMem], [srcMem], size;`,
  `.dst={.global}` / `.src={.shared::cta}` / `.op={.add,.min,.max,.inc,.dec,.and,.or,.xor}`,
  from PTX ISA 8.0 and SM_90; each reduced element has `.relaxed.gpu` semantics.
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk>
  ; the corresponding libcu++ wrapper signature (`cuda::ptx::cp_reduce_async_bulk(space_global_t, space_shared_t,
  op_add_t, float*, const float*, uint32_t)`) is at
  <https://nvidia.github.io/cccl/unstable/libcudacxx/ptx/instructions/cp_reduce_async_bulk.html>.
- `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group{.read}`: same PTX ISA,
  "Bulk asynchronous copy completion" subsection.
- The CuTe DSL interface surface (public infra modules, not operator source):
  - `cutlass/cute/nvgpu/cpasync/copy.py` — `CopyReduceBulkTensorTileS2GOp` (the tensor variant, the basis of
    gotcha 1); `cutlass/cute/nvgpu/cpasync/__init__.py::__all__` has no non-tensor reduction op.
  - `cutlass/cute/nvgpu/cpasync/helpers.py::fence_tma_desc_acquire` /
    `cp_fence_tma_desc_release` — what this entry's `llvm.inline_asm` calling form (operands, constraint string,
    `AD_ATT` + `$n` placeholders, `has_side_effects`) is copied from.
  - `cutlass/cute/core.py::Pointer.toint` — gmem/generic → `Int64`, everything else (smem) → `Int32`.
  - `cutlass/utils/hopper_helpers.py::get_smem_store_op` (old name `sm90_get_smem_store_op`)
    — gives `StMatrix8x8x16bOp` only when `elem_ty_d.width == 16`, otherwise `CopyUniversalOp`; the basis of gotcha 8.
- The staging half (flat layout + flat TV + linear reinterpret + reading back with the same bijection) entirely reuses
  the certified entry [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md).
- GPU-certified by the certification probe
  (8 CTAs × 64×64 fp32 slots accumulating into the same gmem block, constant prefill to verify ADD rather than
  overwrite, read back with the same TV bijection and compared against a torch reference; rungA issues twice in a row
  on the same staging buffer).
- **bf16 §4 (aliasing a dead swizzled TMA-store slot pair as flat staging)**: GPU-certified
  by a separate probe (same mma / same r2s publish atom; S1 publishing into a
  non-compact flat view and reading back with scalar LDS bit-for-bit equal, S2 two CTAs reducing into the same row
  block = prefill + X₀ + X₁, S3 the neighbouring block still prefill, all three `PROBE PASS max_diff=0.0`; S4 every
  `cp.reduce.async.bulk` in the PTX carries `.add.noftz.bf16`).
  KEEP certification of the production cubin contract axis: KEEP in an eval job (job number not entered in the status
  record), axis `cache_ops`
  `["UTMALDG.2D", "UTMASTG.2D"]` → `["UBLKRED.G.S.ADD.BF16.RN", "UTMALDG.2D", "UTMASTG.2D"]`,
  gate 0.7422 → 0.7258 ms (cross-job, at the edge of the noise band; footprint certification, not pricing).
