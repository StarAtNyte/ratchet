# cvt_f32x2_bf16x2_pack_into_r2t_fragment (an adjacent f32 pair → one packed bf16x2 word → written directly into a 32-bit slot of the r2t fragment)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**. The conversion instruction itself exists in every generation; this entry is about how it connects to
> the 32-bit column fragment consumed by `tcgen05.st`, so it lives under the sm100 tree.
>
> This entry answers one question only: **for a row of f32 in registers that must land as bf16 in an r2t fragment of 32-bit elements,
> how to assemble the conversion step so that it is "one f32 pair → one `F2FP.BF16.F32.PACK_AB` → one whole-word write".**
> Three spellings, each with a different machine-read footprint:
>
> | Spelling | SASS footprint (N f32 per thread) | Live range |
> |---|---|---|
> | (1) element-wise `x.to(BFloat16)` written into half-registers of the bf16 recast face | **N F2FP + N PRMT** (one conversion per element + one half-register read-modify-write) -- the shape measured under budget pressure, **[UNVERIFIED-BOUNDARY]**, see "Certification boundary" | ends element by element |
> | (2) whole-row TensorSSA `dst.store(src.load().to(BFloat16))` | N/2 PACK_AB | **the whole source row is live until the store** (under budget pressure = spill) |
> | (3) **this entry: pairwise conversion, one 32-bit value written per slot** | **N/2 PACK_AB, 0 PRMT** | each pair ends as soon as it is converted |
>
> The difference between (1) and (3) is the instruction count (2x the conversions + as many PRMT); the difference between (2) and (3) is not in the instruction count but in the live range --
> this is a corollary of constraint (3) of [`tmem_r2t_recast_fragment`](tmem_r2t_recast_fragment.md),
> and this entry is its positive spelling for the "conversion" step.

## Certification boundary [UNVERIFIED-BOUNDARY]

What this entry's GPU certification pins down is **the forward footprint and bit-exactness of spelling (3)** (both entry points (a)/(b) give N/2 `PACK_AB`, 0 `PRMT`,
0 spill, going through a real r2t fragment → TMEM → readback of the same columns). The "N F2FP + N PRMT" of spelling (1) in the table is **not**
pinned by a probe contract: the certification probe's element-wise stage (the same spelling (1), a small kernel with N = 64 per thread and REG 50) compiled to a footprint identical to spelling (3)
-- without budget pressure ptxas pairs two adjacent conversions into one `PACK_AB` by itself, `PRMT == 0`. N + N has been measured only
**under budget pressure in a large kernel** (the diagnostic reading when this entry was triggered); the boundary (how much pressure, which live-range shape makes ptxas give up pairing)
has not been isolated. So the machine-read criterion judges the **footprint**, not the spelling: only counting `F2FP == N` means the degradation of (1); element-wise code that already shows `PACK_AB == N/2`
need not be rewritten for this; only under budget pressure rearrange the live ranges per spelling (3).

## Purpose

`tcgen05.st` moves data by 32-bit columns, two bf16 per column; the register fragment feeding it therefore has f32/32-bit
elements (the copy atom is built on F32, see `tmem_r2t_recast_fragment` constraint (1)). To convert an f32 source to bf16
and land it in this fragment, the shortest hardware path is PTX `cvt.rn.bf16x2.f32 d, a, b`: **two f32 in, one
32-bit word out** (`a` into the high half, `b` into the low half), one `F2FP.BF16.F32.PACK_AB` in SASS. This word is
one slot of the fragment, written as a whole word, never touching half-registers.

## When needed (trigger)

**Structural trigger**: the output of an element-wise stretch must enter TMEM as a narrow type (or any fragment consumed by 32-bit slots),
the source is an f32 register row, and the stretch is under register budget pressure.

**Symptom triggers**:

| Symptom | Hit |
|---|---|
| In the SASS census the number of `F2FP` == elements per thread (not elements / 2), with as many `PRMT` | spelling (1): element-wise conversion + half-register read-modify-write |
| The number of `F2FP` is already elements / 2, but this stretch's `spill_ld/st` (NCU `l1tex__t_sectors_pipe_lsu_mem_local_op_*`) or ptxas `bytes stack frame` rises with it | spelling (2): whole-row SSA stretches the source row's live range to the full |
| After switching spellings the numerics are right and it is not slower, but you do not know which of (1)(2)(3) you wrote | count once with the "Machine-read criterion" below |

**Anti-trigger**: when the destination is a 16-bit smem face (`stmatrix` / vector `st.shared`), this entry's conversion spelling
still applies (likewise "one pair → one word"), but slot addressing is no longer that of the r2t fragment and is outside this entry's scope.

## Minimal runnable example

Both spellings turn "an f32 pair" into "one 32-bit value" and then write the slot; they differ only in the conversion's entry point.
**Spelling (a)** goes through inline PTX, returns `Int32`, and writes into the fragment's Int32 recast face;
**spelling (b)** goes through the DSL-exported `cute.arch.cvt_f32x2_bf16x2`, which takes a two-element f32 TensorSSA;
the returned two-element bf16 value is stored into column k of a "(2, n) view" on the fragment's bf16 recast face.
Both store into the same f32 fragment (`r_pk32`), and what is handed to `cute.copy` is still the f32 face.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

F32, BF16, I32 = cutlass.Float32, cutlass.BFloat16, cutlass.Int32

# ---- spelling (a): inline PTX, one f32 pair -> one Int32 word ---------------------------
@dsl_user_op
def cvt_bf16x2_f32(lo, hi, *, loc=None, ip=None) -> cutlass.Int32:
    """(lo, hi) -> one bf16x2 word, lo in the low half.
    PTX `cvt.rn.bf16x2.f32 d, a, b` puts a in the high half, so the operands are written `$0, $2, $1`."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [F32(lo).ir_value(loc=loc, ip=ip), F32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def frag_index(t):
    """Index of the fragment's j-th element: ((j, 0), 0, ...) when the value mode is nested, (j, 0, ...) when flat.
    Read it from the shape, do not hard-code it -- the index shape changes when the Repetition depth changes."""
    tail = (0,) * (cute.rank(t) - 1)
    if isinstance(t.shape[0], tuple):
        return lambda j: ((j, 0),) + tail
    return lambda j: (j,) + tail


# The following is inside the kernel body (at trace time); r2t / thr_s / tP are built as in the example of tmem_r2t_recast_fragment:
#   r2t   = tcgen05.make_tmem_copy(cute.make_copy_atom(
#               tcgen05.St32x32bOp(tcgen05.Repetition.x32, tcgen05.Unpack.NONE), F32), tP)
#   thr_s = r2t.get_slice(tidx)
# r_src is this thread's f32 source row (N elements, N even), r_pk32 is the f32 fragment r2t consumes.
def pack_pairs_into_slots(r_src, r_pk32, n_src):
    r_i32 = cute.recast_tensor(r_pk32, I32)      # the 32-bit integer face of the same registers
    ix = frag_index(r_i32)
    for k in range(n_src // 2):                  # plain Python loop: unrolled at trace time
        r_i32[ix(k)] = cvt_bf16x2_f32(r_src[2 * k], r_src[2 * k + 1])
    # afterwards: cute.copy(r2t, r_pk32, thr_s.partition_D(tP)); fence_view_async_tmem_store()


# ---- spelling (b): the DSL-exported pairwise conversion, writing a (2, n) view of the bf16 face ----------------------
def pack_pairs_via_dsl(r_src, r_pk32, n_src):
    n2 = n_src // 2
    src2 = cute.make_tensor(r_src.iterator, cute.make_layout((2, n2)))   # column k = (2k, 2k+1)
    dst2 = cute.make_tensor(cute.recast_tensor(r_pk32, BF16).iterator,
                            cute.make_layout((2, n2)))                    # the bf16 face of the same storage
    for k in range(n2):
        v = cute.arch.cvt_f32x2_bf16x2(src2[(None, k)].load())   # returns a raw ir vector, not a TensorSSA
        dst2[(None, k)].store(cute.TensorSSA(v, (2,), BF16))      # wrap it in a TensorSSA before the store (gotcha 1)
```

Two disciplines on the calling shape:

- **The packing helper is undecorated and uses a Python `range`** -- it runs at trace time inside the kernel body and
  has no run-time control flow, so it opens no `if`/`for` region in the kernel and no name gets bound inside a region and stuck
  there (see [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)). Decorating it with `@cute.jit`
  and using `range_constexpr` also works, but is unnecessary.
- **What you write is the Int32 / bf16 recast face; what you hand to `cute.copy` is the f32 face**. The three faces share one storage,
  and the constraint that the atom is built on F32 (`tmem_r2t_recast_fragment` (1)) does not change because you write through the integer face.

### Machine-read criterion (used for both certification and regression)

Do a SASS census of the kernel carrying this stretch (`CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_DUMP_DIR` to keep the cubin,
`nvdisasm -c` / `cuobjdump -sass`; `cuobjdump -res-usage` to read `LOCAL`):

```
number of F2FP.BF16.F32.PACK_AB == elements per thread / 2     (== elements ⇒ degraded to spelling (1))
number of PRMT (inside the packing stretch) == 0              (> 0 ⇒ half-register read-modify-writes mixed in)
LDL == STL == 0, LOCAL == 0                                   (> 0 ⇒ a live-range problem; first check whether it is spelling (2))
```

## Gotchas

- **The return value of `cvt_f32x2_bf16x2` cannot be `store`d directly.** It returns a raw two-element bf16 ir vector (`Vector`);
  `dst2[(None, k)].store(cute.arch.cvt_f32x2_bf16x2(...))` fails at build time with
  `ValueError: Expected TensorSSA, but got Vector`; it must be wrapped with `cute.TensorSSA(v, (2,), BF16)` before the store.
  In the certification probe's sweep of store spellings, the direct store is the only candidate that fails to build; wrapped, it is bit-exact (the spelling of §(b)).
- **Half order: in `cvt.rn.bf16x2.f32 d, a, b`, a is the high half.** To have (src[2k], src[2k+1]) land as (lo, hi)
  (little-endian, 2k is the first bf16 in memory, and also the half the consumer reads as "column 2k" within the 32-bit column), the asm must be written
  `$0, $2, $1`. Writing `$0, $1, $2` **raises no error and produces no inf/nan**: swapping the two halves of every word = swapping adjacent columns,
  so the downstream result is wrong but finite. The certification probe pins this down with a bit-by-bit comparison.
- **Writing the bf16 recast face element by element is not "another spelling of the same thing".** Under budget pressure it is, in SASS, one `F2FP` per element
  plus one `PRMT` (stuffing the new half into the old word), more than doubling the instruction count (in a small kernel ptxas pairs them automatically, see "Certification boundary"); and it requires the fragment to be addressable by half-register,
  which in itself pushes the compiler toward scalarization. `F2FP == number of elements` is the machine-read footprint of this degradation.
- **The whole-row TensorSSA conversion also gives PACK_AB, but the live range is wrong.** `dst.store(src.load().to(BF16))`
  is a whole-row operation on a value domain: the whole source row is live until the store, and every consumer of the source row is pushed after the store.
  On a path with register headroom it compiles to the same cost as the pairwise spelling (what the gotcha "write granularity is neutral" of `tmem_r2t_recast_fragment`
  measures is exactly this); under budget pressure, the pairwise spelling lets ptxas kill each source pair right after its conversion and interleave conversions with
  downstream writes, while the whole-row spelling does not. **When there are spills, look at this first, before anything else.**
- **The slot unit is the 32-bit word, and the index is read from the shape.** Slot k ↔ source elements (2k, 2k+1). The fragment's value mode
  may be the nested `((E, 1), 1, ...)` (one Repetition per instruction) or flat, and the two index shapes differ;
  `frag_index` discriminates from the shape, so changing the Repetition depth needs no rewrite. Hard-coding one shape is a second silent assumption.
- **Source pairs must be adjacent, even in number, and consistent with the consumer's column order.** The premise of pairwise conversion is that (2k, 2k+1) are exactly the two halves of the downstream 32-bit
  column k; if the source row's element order is not column order (e.g. a fragment produced by some retile, with a non-compact layout),
  first confirm the order per the "same bijection at both call sites" of `tmem_r2t_recast_fragment`, then pair.
  The DSL-exported `cvt_f32x2_bf16x2` takes only a two-element TensorSSA; when the source fragment is not compact the `(2, n)` view cannot extract
  adjacent pairs, and only the scalar-pair form of inline PTX (spelling (a)) remains.
- **The (2, n) view requires a DSL build that allows reinterpreting register storage.** Spelling (b)'s `dst2` / `src2` are
  `make_tensor(t.iterator, make_layout((2, n)))`; some builds reject re-viewing a register iterator.
  If rejected, use spelling (a): it needs only the recast face and per-slot indices, no re-view.
- **The cvt rounding is RNE (`.rn`)**, the same as the default rounding of `.to(BFloat16)`, so the three spellings are bit-identical;
  only switching to `.rz` or another rounding makes them diverge numerically.

## Sources

- PTX ISA, the `cvt` instruction section (`cvt.rn.bf16x2.f32 d, a, b`: a → high half, b → low half, `.rn` rounding)
  and `tcgen05.st` in the "TensorCore 5th Generation Instructions" chapter (32-bit column addressing)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols: `cute.arch.cvt_f32x2_bf16x2`, `cute.recast_tensor`,
  `cute.make_rmem_tensor`, `cute.make_tensor(iterator, layout)`, `TensorSSA.load/store/to`,
  `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`,
  `cutlass.cute.nvgpu.tcgen05.make_tmem_copy / St32x32bOp / Ld32x32bOp / Repetition / Unpack`,
  `cute.arch.fence_view_async_tmem_store / fence_view_async_tmem_load`.
- Compile-time observation channel: `CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_DUMP_DIR=<dir>` to keep the cubin,
  `nvdisasm -c` / `cuobjdump -sass` to count `F2FP` / `PRMT` / `LDL` / `STL`, `cuobjdump -res-usage` to read `LOCAL`.
- GPU certification probe (the probe is not included in this release)
  -- **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, passed at the first try in round 1 of this certification, all six contracts pass in the same round; **the N + N footprint of spelling (1) is [UNVERIFIED-BOUNDARY]**, see "Certification boundary"). The six contracts: `per_element_reference_bits` (spelling (1) is bit-equal to the host RNE reference,
  pinning the reference to the DSL's own rounding); `pair_ptx_bits` (spelling (a) bit-equal, lo = the even element);
  `pair_ptx_footprint` (the kernel of spelling (a): F2FP == elements / 2, PRMT == 0, LDL == STL == 0, LOCAL == 0);
  `pair_api_bits` / `pair_api_footprint` (spelling (b): an in-process sequential sweep of store spellings -- a direct `store(api(...))`
  fails to build with `Expected TensorSSA, but got Vector`; `store(cute.TensorSSA(v, (2,), BF16))` is bit-exact and is the winner,
  the sweep stops there, and the two candidates `Int32(v)` writing the integer slot / two scalar calls were not tried; the winning kernel's footprint is F2FP == N/2, PRMT == 0);
  `r2t_slot_roundtrip` (the pairwise-packed words written into the partition_S fragment of a real `St32x32bOp.x32` → TMEM →
  read back with `Ld32x32bOp.x32`, landing in the same 32-bit columns as the host reference packing).
