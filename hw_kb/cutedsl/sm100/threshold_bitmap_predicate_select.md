# threshold_bitmap_predicate_select (threshold mask: one bitmap per lane per pass → bit-test predicates → `FSEL`, not a per-column `ISETP`)

> **Certification source: sm100, this tree** — ✅ Verified: certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=3.947611147100029e-07` (round 2 of this certification; in round 1 the census
> tool's `BRA` count included the self-jump trap after `EXIT` in the loop body; the mechanism itself was unchanged, see §(d) item 4).
>
> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the sequel to [`branch_free_select_mask`](branch_free_select_mask.md). That one pins down "a mask is a select, not arithmetic",
> and its spelling is **one segment per column** of `setp + selp` asm; its predicate source is "one already-existing valid bit per column".
> This entry handles another, more common valid set: **within one thread's K columns, the valid columns are decided by a threshold**
> (contiguous runs like `c >= t` / `c < t` — triangular / banded masks are all this). Here a per-column comparison is the wrong form:
> one `ISETP` per column plus one `VIADD` to materialise "the absolute column index of column c" turns the mask into 3K instructions.
> The right form computes the threshold **only once** per lane per pass and generates bitmap words; each column's select is driven by a **bit test**,
> and ptxas folds consecutive bit tests into `R2P` (one instruction loading the low 7 bits of a register into 7 predicate registers)
> + predicated `FSEL`.

## Purpose

Turn "is column c valid" from **one integer comparison per column** into **one bitmap per pass + one bit test per column**, so that
the mask lands in SASS as `R2P` + `FSEL` (about 1.2K instructions) rather than `VIADD` + `ISETP` + `FSEL` (3K instructions).
Not a single `FSEL` is saved — what is saved are the two instructions that feed the predicate.

## When needed (trigger)

The SASS census (loop body, one thread holding K columns) shows all of:

1. **`ISETP` ≈ K**: one integer comparison per column — the predicates are compared column by column.
2. **`VIADD` (or `IADD3`) ≈ K**: one integer add per column — one side of the comparison is `base + c`, with `c` a compile-time constant and
   `base` a runtime value; their sum is materialised into a register outside the asm block (asm operands can only be registers; immediates cannot fold in).
3. **`FSEL` ≈ K, `R2P == 0`**: the select is already in predicate form (not `I2F` arithmetic), but the predicates do not come from a bitmap.
4. The valid set is a **contiguous run**: for a fixed lane and pass, `valid(c)` is monotone in `c` (a prefix or a suffix) —
   this is the precondition for generating the bitmap in one go; scattered valid bits (an independent flag per column) still go through
   [`branch_free_select_mask`](branch_free_select_mask.md) §(a).

All four together = "per-column comparison form". If only 1 and 3 appear without 2 (both sides of the comparison are existing registers), it is still
this entry's target: those K `ISETP`s are what must go.

## Mechanism: instruction accounting of the three forms (per pass, per thread, K columns)

| Form | Once per pass | Per column | Total (K=64) | Notes |
|---|---|---|---|---|
| **per-column comparison** (to be removed) | — | `VIADD`(base+c) + `ISETP` + `FSEL` | ≈ 192 | the `_ge` form's `setp.ge.s32 p, base+c, t` |
| **bitmap + per-bit LOP3** | bitmap words ≈ 2×4 instructions | `LOP3` (with predicate output) + `FSEL` | ≈ 136 | degraded reading when ptxas does not synthesise `R2P` |
| **bitmap + `R2P`** (this entry's target) | bitmap words ≈ 2×4 instructions | `FSEL`; 1 `R2P` per 7 columns (+1 `SHF` to shift in the next group) | ≈ 90 | the certification probe's readings are under "Sources" |

Key points:

- **The threshold is computed only once per lane per pass.** `t = threshold − column_base` is one `IADD`; after that nothing that is independent of the column index
  appears per column any more.
- **The bitmap comes free from PTX's shift clamping**: `shl.b32 d, a, b` yields 0 when `b ≥ 32` (PTX specifies that shift amounts
  greater than the register width are clamped to the width), so `0xffffffff << clamp(t, 0, 32)` is directly "bit c set ⇔ c ≥ t",
  all ones for `t ≤ 0`, all zeros for `t ≥ 32`, **no second comparison needed**. The second word is computed again from `t − 32`.
- **Each column is left with only a bit test.** `w & (1 << c)` non-zero means valid, fed into `setp.ne` → `selp`; ptxas sees a run of single-bit tests on the same
  register, which is exactly the input shape from which it synthesises `R2P`. `R2P` can only load the low 7 bits at a time
  (P0–P6), so between every 7 columns there is an extra `SHF` shifting the next group down — ptxas inserts it itself,
  no need to write it by hand.
- **Prefix / strict inequalities are variants of the same bitmap**: `c < t` is `~ones_from(t)` (one `LOP3`, or swap the select's
  two operands); `c > t` is `ones_from(t + 1)`; `c ≤ t` is `~ones_from(t + 1)`.

## Minimal runnable example

### (a) Bitmap word: `ones_from(t)` — bit c set ⇔ c ≥ t (t any Int32, clamped at both ends)

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


@dsl_user_op
def ones_from(t, *, loc=None, ip=None) -> cutlass.Int32:
    """32-bit word with bit c set iff c >= t. t < 0 -> all ones; t >= 32 -> 0.
    max.s32 handles negatives; PTX shl clamps shift amounts >= 32 to 32, so t >= 32 gives 0
    without a second comparison."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [cutlass.Int32(t).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .s32 tl;\n\t.reg .b32 m;\n\t"
        "max.s32 tl, $1, 0;\n\tmin.s32 tl, tl, 32;\n\t"
        "mov.b32 m, 0xffffffff;\n\tshl.b32 $0, m, tl;\n}",
        "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def select_f32(cond, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """cond != 0 ? a : b  (cond: Int32)  -> one FSEL; branch_free_select_mask §(a)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(cond).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.f32 $0, $2, $3, p;\n}",
        "=f,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
```

**The shift must be done in PTX**: in the DSL, `Int32(-1) << t` lowers to an LLVM `shl`, and a shift amount ≥ 32 is poison,
so the result is undefined when `t` is out of range (gotcha 1). The four integer instructions run once per pass and are not in the per-column account.

### (b) Per pass: one threshold, two words, K column bit tests + select (plain Python helper, unrolled at trace time)

```python
PARK = -1.0e30                      # finite park value (branch_free_select_mask gotcha 3)
K = 64                              # number of columns this thread holds (two 32-bit words)


def _imm(v):
    """Python int -> a signed value acceptable as an Int32 constant (the mask for bit 31 is negative)."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def mask_row(x, t, park):
    """x: K f32 values (value of column c); t: this lane's threshold for this pass (Int32, column index relative to this thread's first column).
    Returns K select results. No decorator: unrolled only at trace time, no runtime control flow."""
    lo = ones_from(t)               # columns 0..31
    hi = ones_from(t - 32)          # columns 32..63
    out = []
    for c in range(K):
        w = lo if c < 32 else hi
        out.append(select_f32(w & _imm(1 << (c % 32)), x[c], park))
    return out


# in the loop body (per pass):
#   t = threshold - column_base           # one IADD, once per lane per pass
#   s = mask_row(x, t, PARK)              # two ones_from + K bit-test selects
#   for c in range(K):
#       p[c] = exp2(s[c] * C - m)         # one FFMA feeding EX2 (branch_free_select_mask §(b))
```

`column_base` is the absolute column index of this thread's first column (a runtime value) and `threshold` is this lane's absolute threshold for this pass;
after subtracting the two, only the compile-time constant `c` is left of the column index, and it all folds into the mask immediates.

### (c) Single-asm-block spelling of the same mechanism (one block per 8 columns, column indices are immediates inside the block)

When you want to take the pairing of bit test and select entirely out of LLVM's hands (e.g. `w & mask` gets rewritten by some other transform,
or you want exactly two PTX instructions per column), use an 8-output block:

```python
from cutlass._mlir import ir

_S8 = "!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>"


def _asm8_text(bit0):
    lines = ["{", "\t.reg .pred p0, p1, p2, p3, p4, p5, p6, p7;", "\t.reg .b32 t;"]
    for k in range(8):                       # outputs $0..$7, word $8, x $9..$16, park $17
        lines.append("\tand.b32 t, $8, 0x%08x;" % (1 << (bit0 + k)))
        lines.append("\tsetp.ne.b32 p%d, t, 0;" % k)
        lines.append("\tselp.f32 $%d, $%d, $17, p%d;" % (k, 9 + k, k))
    lines.append("}")
    return "\n".join(lines)


def select8(word, xs, park, bit0):
    """Bits bit0..bit0+7 of word drive the selects of xs[0..7]; returns 8 Float32. No decorator."""
    r = llvm.inline_asm(
        ir.Type.parse(_S8),
        [cutlass.Int32(word).ir_value()] + [cutlass.Float32(v).ir_value() for v in xs]
        + [cutlass.Float32(park).ir_value()],
        _asm8_text(bit0), "=f,=f,=f,=f,=f,=f,=f,=f,r,f,f,f,f,f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return [cutlass.Float32(llvm.extractvalue(T.f32(), r, [k])) for k in range(8)]
```

The operand numbering rule for struct results is in [`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md)
(results take `$0..$7` first, the first input is `$8`).

### (d) Acceptance: SASS census (loop body, one thread K columns; u = unroll factor seen by the census, taken as `MUFU.EX2 / K`)

1. `FSEL == K·u` (not a single select is dropped); the `MUFU.EX2` count is unchanged.
2. `ISETP ≤ 8·u` (only loop guards and the like remain); `VIADD + IADD3 ≤ 24·u` (column indices are no longer materialised per column).
3. `R2P ≥ 8·u`: ptxas synthesised the bit tests into register→predicate moves; no `R2P` with `LOP3` ≈ K·u is the degraded "bitmap +
   per-bit LOP3" reading (still K instructions fewer than per-column comparison, but not this entry's target form).
4. `LDL == STL == 0`; the loop body has no `BRA` / `BSSY` other than the guard + back edge (after `EXIT` there is always a
   `BRA <own address>` teardown trap; it does not belong to the loop body, exclude it by address when taking the census).
5. Numerics: **bit-identical** to the per-column comparison form — the predicate values of the two forms are exactly the same and the two `FSEL` operands are the same,
   so there is no rounding difference. The difference from the numpy reference comes only from the downstream `ex2.approx`.

## Gotchas

1. **The bitmap shift is unsafe to write in the DSL.** `-1 << t` / `(1 << t) - 1` are LLVM poison when `t ≥ 32`
   (not 0, not all ones — "anything"), and likewise for `t < 0`; do it only in PTX, with `max.s32` clamping the lower bound and
   `shl.b32`'s clamping rule clamping the upper bound (§(a)). `min.s32 tl, tl, 32` spells out the upper bound only so readers need not
   look up the ISA; removing it leaves the readings unchanged.
2. **The second word uses `t − 32`, not a split like `t >> 5`.** The same `ones_from` is called twice with inputs differing only by
   a constant; the clamping absorbs the "which word does t fall in" branch — no `if`, no region.
3. **The mask for bit 31 is negative.** `1 << 31` as an Int32 constant must be written as `-2147483648` (`_imm` in §(b)),
   otherwise constant construction overflows; in PTX write hex `0x80000000` (§(c)).
4. **Test each column with `w & (1 << c)`, not `(w >> c) & 1`.** The latter costs an extra `SHF` per column, and ptxas synthesising
   `R2P` wants the "same register, different single-bit masks" shape. A grouped right shift (every 8 columns `w >> 8g`, then test
   bits 0..7) is an equivalent spelling; whether to let it appear is decided by the certification readings (see "Sources").
5. **asm block boundaries are opaque only to LLVM; they do not exist for ptxas.** Inline asm text goes verbatim into the PTX file, and ptxas sees
   a flat stream of `and` / `setp` / `selp` — `R2P` is synthesised by ptxas at the SASS level, **PTX has no `r2p`
   instruction**, you can neither write it nor prevent it. So §(b) and §(c) give ptxas the same input; the only difference is whether LLVM
   can see the `and` (§(b) it can, so it may be CSE'd / reordered; §(c) it cannot).
6. **Predicates do not cross asm blocks** (same as `branch_free_select_mask`): `setp` and `selp` must be in the same block;
   a block-scoped `.reg .pred p` gets its own copy per call site, so names never clash.
7. **`has_side_effects=False`.** Both ops are pure functions; marking them as having side effects pins `ones_from` next to each column
   and blocks ptxas's scheduling, and `R2P` synthesis also depends on ptxas being able to schedule 7 `FSEL`s together.
8. **`x` must be a loop-invariant register value, and the threshold must really change every pass.** The probe ensures this with a dynamic pass count + reading
   the threshold from gmem every pass; if the threshold is a constant visible at compile time, the optimiser folds the bitmap into constants, the census reads 0
   `R2P`, and the mechanism was never actually exercised.
9. **Column indices are relative to this thread's first column.** After `t = threshold − column_base`, `c ∈ [0, K)`; if the absolute column index stays
   in each column (`base + c ≥ threshold`), you are back to the per-column comparison form — those K `VIADD`s are exactly its fingerprint.
10. **Unroll factor.** The census counts the whole kernel; if the optimiser unrolls the pass loop u times, every threshold scales by u (§(d));
    `MUFU.EX2 / K` is the most stable way to read u.
11. **A class lever only pays off when the segment is issue-bound** (`branch_free_select_mask` gotcha 11): what is saved are issue slots
    (about 2K instructions per pass per thread); a latency-bound segment's clock does not move; check the §(d) footprint first, then the clock.

## Sources

- PTX ISA: `shl` ("Shift amounts greater than the register width N are clamped to N")
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl>;
  `max` / `min` (integer) <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max>;
  `and` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and>;
  `setp` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>;
  `selp` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>.
- CUDA Binary Utilities, Instruction Set Reference (`R2P` "Move Register To Predicate Register",
  `P2R`, `FSEL`, `ISETP` as SASS mnemonics)
  <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference>.
- NVVM/LLVM inline asm constraint letters (`r` / `f` / `=f`) and struct multi-results
  <https://llvm.org/docs/NVPTXUsage.html>; LLVM `shl` out-of-range shift is poison
  <https://llvm.org/docs/LangRef.html#shl-instruction>.
- Public infra API symbols: `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.cutlass_dsl.T`,
  `cutlass._mlir.dialects.llvm.inline_asm` / `llvm.extractvalue` (`asm_dialect=llvm.AsmDialect.AD_ATT`),
  `cutlass._mlir.ir.Type.parse`, `cutlass.Int32(...).ir_value(loc=, ip=)` /
  `cutlass.Float32(...).ir_value(loc=, ip=)`, `cutlass.range` / `cutlass.range_constexpr`,
  `cute.make_rmem_tensor`.
- GPU certification probe — the same computation
  (64 columns / thread, dynamic pass count, threshold and offset read from gmem per pass per lane, select → one FFMA → `ex2.approx`
  → accumulate) in five forms, one process each: `cmp_base` (per-column `setp.ge.s32 base+c, t`), `and_sel` (§(b)),
  `grp8_sel` (every 8 columns first `w >> 8g`, then test bits 0..7), `asm8_sel` (§(c)), `native_sel`
  (`arith.cmpi` + `arith.select`, no asm); five contracts judged in the same round: **bitmap_words** (`ones_from` word-exact for
  t ∈ [−40, 87]), **numeric** (the winner vs. a float64 numpy reference, relative error ≤ 1e-4),
  **baseline_census** (`cmp_base` has `ISETP ≥ 64`, i.e. the census can tell the per-column comparison form apart),
  **bitmap_census** (the winner meets §(d) items 1, 2, 4), **r2p_synth** (the winner has `R2P ≥ 8`).
  Winner = the form that passes both numerics and footprint with the most `R2P` and the least mask overhead; the winner in the certification readings is **`and_sel` (§(b))**,
  `asm8_sel` (§(c)) has an instruction-for-instruction identical footprint and ties right behind it, `grp8_sel` has the extra `SHF`s of the grouped right shift, `native_sel` has no `R2P`
  (`LOP3 ≈ K`, i.e. the degraded "bitmap + per-bit LOP3" reading). Status: **✅ Verified**
  — `PROBE PASS max_diff=3.947611147100029e-07` (round 2; all five contracts passed in the same round;
  `max_diff` is the winning form's relative error against the float64 reference, coming from the downstream `ex2.approx`).
