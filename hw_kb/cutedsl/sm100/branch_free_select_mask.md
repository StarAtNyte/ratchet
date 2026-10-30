# branch_free_select_mask (the hardware form of an element-wise mask: predicate + `FSEL`, not "convert the flag bit to float and do arithmetic")

> **Certification source: the sm90 tree · not re-checked on sm100.**
>
> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the sequel to [`softmax_segment_ilp`](softmax_segment_ilp.md): that one prices the row-reduction segment by
> MIO count and instruction class, and gives one class lever, `fmax` → `FMNMX`. This entry adds the class lever for **the other half**
> of the segment -- the mask. The product of the `fl.to(Float32)` step in that entry's §(c) example is `I2F`, which is exactly the form this entry
> eliminates; since this entry's certification, the mask spelling of that example follows this entry.

## Purpose

An element-wise segment must "keep or discard" scores according to **one valid bit per column**. In hardware this is one
`FSEL` (predicated select): `s = p ? x : PARK`. CuTe DSL exports no select primitive; an `if` statement is a
region, and whether it lowers to `FSEL` is up to the optimizer. This entry gives a **deterministic spelling** -- the `setp` + `selp.f32` escape hatch
inside a `@dsl_user_op` (form 3 of [`ptx_escape`](ptx_escape.md)) -- and
the segment shape that comes with it: the row max is done in the **raw domain**, and the scale constant is folded into the `FFMA` that feeds `EX2`.

## When needed (trigger)

In the SASS census (loop body, one thread holding R rows × K columns) all of the following appear at once:

1. **`I2F` ≥ K**: one integer→float conversion per column -- the flag bit has been `.to(Float32)`-ed.
2. **`FSEL` ≈ 0** (only a few stray ones from other guards remain): the mask does not go through predicated selection.
3. **What feeds `MUFU.EX2` is a separate `FADD`** (`s - m`), not an `FFMA`: the scale was already multiplied into `s` at the mask step,
   so pass 3 has only the subtraction left to do.
4. The segment's main stall items are `short_scoreboard` / `wait`, and the segment sits on a serial handshake chain (the other side is waiting for it).

All four together = the mask goes through the "I2F form": per column `I2F` + `FMUL`(cv) + `FFMA`(bv), per element
`FFMA`(x·cv + bv), and in pass 3 per element `FADD` + `MUFU`. (Count `I2F` by opcode prefix: on sm_90 a
32-bit integer→float may also be spelled `I2FP.F32.U32`.)
**Item 1 not appearing does not mean the select form is already in place**: if the valid bit is a single-bit register value the optimizer can see through
(an `& 1` on a word straight out of `LDS`), `.to(Float32)` may be lowered to predicated `FMUL`/`FFMA` --
`I2F == 0` and only a few stray `FSEL`, yet the two arithmetic instructions per column and the separate `FADD` of pass 3 remain untouched,
and the clock is still ~5% worse (first-round reading of the certification probe). When the valid bit comes out through `prmt`/inline asm or a byte read (opaque to the optimizer),
`I2F` appears, one per column. Judge the form by items 2 and 3; item 1 is only supporting evidence.

`I2F` and `MUFU` are in the same throughput tier
(the CUDA Programming Guide's arithmetic-instruction throughput table: 32-bit integer→float conversion and `exp2f` are both 16 results per SM per clock,
`FFMA` is 128), so K `I2F` are load on **the same pipe**, stacked directly on the R·K `EX2`, not free ALU work.

## Mechanism: the instruction accounting of the three forms (per trip, per thread, R rows × K columns)

| Form | Per column | Per element (mask) | Row-max domain | Pass 3 per element | Notes |
|---|---|---|---|---|---|
| **I2F form** (to eliminate) | bit extraction + `I2F` + `FMUL`(cv) + `FFMA`(bv) | `FFMA`(x·cv + bv) | scaled domain | `FADD`(s − m) + `MUFU.EX2` | `I2F` and `EX2` share a pipe |
| **select form** (this entry) | bit extraction (`LOP3`, often folded with `setp` into one `LOP3` with a predicate output) | `FSEL`(p ? x : PARK) | **raw domain** | `FFMA`(s·C − m·C) + `MUFU.EX2` | one extra `FMUL` per row (m·C) |
| **run-time `if` form** (DSL native) | bit extraction + `ISETP` | up to the optimizer: `FSEL`, or a `BSSY`/`BRA` region | raw domain | same as above | the certification probe's `ifsel` variant only gives readings, **no guarantee** |

There are only two key points:

- **The mask is a selection, not arithmetic**. The valid bit is the source of a predicate; do not turn it into 0.0/1.0 and multiply.
  One `selp.f32` parks an invalid column at a finite large negative number (of the `-1e30` kind, not `-inf`, see gotcha 3).
- **Defer the scaling to the `FFMA` before `EX2`**. Do the row max on the raw scores (`FMNMX` is insensitive to a monotone scaling;
  `max(x)·C = max(x·C)` differs only by rounding), multiply by `C` once per row to get `m·C`, then
  `p = exp2(fma(s, C, −m·C))` -- the `FFMA` feeds `MUFU` directly, and the separate `FADD` in pass 3 disappears.

## Minimal runnable example

### (a) Two select primitives (PTX escape hatch, pure functions, no side effects)

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


@dsl_user_op
def select_f32(cond, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """cond != 0 ? a : b   (cond: Int32; any nonzero integer counts as true)  -> one FSEL"""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(cond).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.f32 $0, $2, $3, p;\n}",
        "=f,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def select_f32_gt(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """x > y ? a : b   (all fp32)  -> FSETP + FSEL, no region"""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.gt.f32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
```

The `.reg .pred p` inside `{ ... }` is block-scoped: each call site gets its own copy, so calling it any number of times in one kernel
never clashes on names. Inline asm can only pass `r`/`f` class operands, and a predicate cannot be passed across asm blocks,
so the comparison and the selection must be written in **the same** asm block.

### (b) Segment shape: select mask → raw-domain row max → one FFMA into EX2

```python
C = 0.7213475                  # scale constant (a compile-time constant or uniform; see gotcha 4)
PARK = -1.0e30                 # a finite large negative number, not -inf (gotcha 3)
NEG_TEST = -1.0e29
fmax = cute.arch.fmax          # if absent, see the max.f32 escape hatch of ptx_escape

h = sFlags[flag_word_index]    # this thread's K valid bits for this trip (one Int32 word)
for c in cutlass.range_constexpr(K):
    bit = h & (1 << c)         # nonzero means valid; do not .to(Float32)
    for r in cutlass.range_constexpr(R):
        s[(r, c)] = select_f32(bit, x[(r, c)], PARK)        # FSEL, raw domain

for r in cutlass.range_constexpr(R):
    m = s[(r, 0)]
    for c in cutlass.range_constexpr(K - 1):
        m = fmax(m, s[(r, c + 1)])                            # FMNMX, raw domain
    m = fmax(m, cute.arch.shuffle_sync_bfly(m, 1))
    m = fmax(m, cute.arch.shuffle_sync_bfly(m, 2))
    m_old = m_run[r]
    m = fmax(m, m_old)
    m_run[r] = m
    alpha = cute.math.exp2((m_old - m) * C, fastmath=True)   # subtract first, then multiply (gotcha 5)
    mc = m * C                                                # one FMUL per row
    mu = select_f32_gt(m, NEG_TEST, mc, 0.0)                  # empty-row guard, also a select (gotcha 3)
    lsum = cutlass.Float32(0.0)
    for c in cutlass.range_constexpr(K):
        pv = cute.math.exp2(s[(r, c)] * C - mu, fastmath=True)   # one FFMA feeds EX2
        lsum = lsum + pv
    ...
```

### (c) Acceptance: SASS counts (loop body), not end-to-end

1. `I2F == 0` (before the change ≥ K, counting `I2F`/`I2FP` by prefix; if the valid bit is a register bit the optimizer can see through, it may already be 0 before the change);
   `FSEL ≥ R·K` (≈ 0 before the change); no `BRA`/`BSSY` in the loop body apart from the back edge.
2. `FADD` drops by ≥ three quarters of R·K (the `s − m` of pass 3 disappears; the row-sum `FADD` remain);
   `FADD + FFMA + FMUL + I2F` in total drops by ≥ 2·K.
3. The number of `MUFU.EX2` is unchanged: the exp2 work itself did not move; only the class of the instructions feeding it changed.
4. `LDL == STL == 0`.
5. Numerics: not bit-equal to the old form (`FFMA` contraction and the scaled domain differ, one rounding each),
   relative error on the order of 1e-6; relative error against the numpy reference ≤ 2e-3 (the accumulation of `ex2.approx`).

## Gotchas

1. **`.to(Float32)` is that `I2F`**. Any spelling of "turn the valid bit into 0.0/1.0 and multiply" brings
   it back -- including the seemingly cheaper `s * fl + (fl − 1) * M`. The valid bit can only be used as a predicate.
   The optimizer can occasionally fold the conversion of a see-through 0/1 value into predicated arithmetic (`I2F` disappears), but that only changes
   the spelling; the `FADD` of pass 3 and the two arithmetic instructions per column remain -- items 2 and 3 of "When needed" are the criterion.
2. **A run-time `if` is not a spelling of select**. `if bit == 0: s = PARK` is an scf region in the DSL;
   in some versions / some contexts the optimizer folds it into `ISETP` + `FSEL`, in other cases it stays as
   `BSSY`/`BRA`; even when folded into `FSEL` it is two instructions (comparison + selection), whereas in the `selp` form the comparison is often merged
   with the bit extraction. For determinism use §(a). The readings of the certification probe's `ifsel` variant are in the `EXPR A=` line of Sources.
3. **Use a finite park value, and guard empty rows**. Parking at `-inf`, a row with no valid columns gives
   `(-inf) − (-inf) = NaN`; parking at `-1e30`, a row with no valid columns gives `exp2(0) = 1` --
   every discarded column gets full weight. The guard is one more select: `mu = (m > NEG_TEST) ? m·C : 0`,
   after which `exp2(s·C − 0)` underflows to 0 on the park value. Do not write this step as a run-time `if` either.
4. **`C` must be a constant or warp-uniform**. It is used once per element; if it is an ordinary register value,
   the second operand of `FFMA` occupies a register (no big deal), but if it must be recomputed every trip that is K extra pieces of work.
   A compile-time constant is best (`FFMA` takes an immediate / constant-bank operand directly).
5. **Use `(m_old − m) · C` for the rescale factor; do not write `fma(m_old, C, −m·C)`**. `m·C` is already rounded,
   so when the two are equal the fused form leaves a residue of ~ulp(m·C); at the park value (order 1e30) that residue is
   of order 1e22, and `exp2` goes straight to inf / 0. Subtracting first and then multiplying gives an exact 0 when `m_old == m`.
6. **Constraint string and placeholders**: the output `=f` comes first and maps to `$0`; an integer predicate source uses `r` (Int32), fp32 uses `f`;
   placeholders are `$n`, not `%n` ([`ptx_escape`](ptx_escape.md) gotchas 2, 3).
7. **`has_side_effects=False`**. select is a pure function; marking it side-effecting blocks CSE and scheduling
   (when two rows share one column's predicate, ptxas merges the `setp` into one -- provided it can move freely).
8. **Do not compare the extracted 0/1 again**. `bit = h & (1 << c)` is true when nonzero, and `setp.ne.s32 p, bit, 0`
   takes it directly; `(h >> c) & 1` costs one more `SHF` and tempts you to `.to(Float32)`.
9. **There must be no file-level `from __future__ import annotations`** (`@dsl_user_op` evaluates annotations at run time).
10. **Read the flag words that feed the predicates before the blocking wait**. They are `LDS`; if the barrier guarding them is not the one you are about to
    block on, put the read before the block, so that the `LDS` latency falls inside the wait instead of being scheduled on the other side's critical
    path (`bar.sync` only orders the shared memory accesses of its own participants, see
    [`softmax_segment_ilp`](softmax_segment_ilp.md) §(a)).
11. **A class lever only pays off when the segment is issue-bound**. What it saves is issue slots: about 2·K + R·K
    instructions per trip per thread (K `I2F` + 2K per-column arithmetic + R·K `FADD`). If the segment is latency-bound -- each
    scheduler has only 1–2 warps, and the row reduction is written as a 15-deep serial `FMNMX` / `FADD` chain -- these instructions were
    hidden in their own stalls anyway, and the clock does not move at all (certification probe round 2: 1 CTA/SM + a serial chain,
    `t(I2F form)/t(select form) = 0.993`, even though the SASS footprint had changed as expected). Only after turning the reduction chain into a tree
    (likewise 15 instructions, 4 deep) and giving each scheduler ≥ 4 warps did the same pair of forms separate on the clock.
    Order of judgement: first look at the SASS footprint (§(c) items 1–3) to confirm the form changed, then at the clock; if the clock does not move, first check whether
    the segment is latency-bound; do not roll back the form.
    Spelling of the tree reduction: **put it in an undecorated ordinary Python helper function** (unrolled at trace time, emitting only
    15 instructions). If "`while len(v) > 1: v = [op(v[2i], v[2i+1]) ...]`" is written directly in the
    `@cute.kernel` body, the DSL's AST rewrite treats it as a run-time loop, and the loop-carried value changing from `list<16>`
    to `list<8>` reports `TYPE_UNSTABLE_JOIN` (exactly the compile failure of the certification probe's round 3).

## Sources

- PTX ISA: `selp` (select by predicate, `selp.f32 d, a, b, p`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>;
  `setp` (comparison writing a predicate, `.ne.s32` / `.gt.f32`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>;
  `max` (a single `max.f32`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>;
  for block-scoped register declarations in inline asm (`{ .reg .pred p; ... }`) see PTX ISA "Statements / Scope"
  and the CUDA "Inline PTX Assembly" documentation
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>.
- CUDA C++ Programming Guide, the "Arithmetic Instructions" throughput table (32-bit integer→float conversion and
  `exp2f` in the same tier, 16 per SM per clock; `FFMA` 128)
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>.
- NVVM/LLVM inline asm constraint letters (`r` / `f` / `=f`) <https://llvm.org/docs/NVPTXUsage.html>.
- Public infra API symbols: `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.cutlass_dsl.T`,
  `cutlass._mlir.dialects.llvm.inline_asm` (`asm_dialect=llvm.AsmDialect.AD_ATT`),
  `cutlass.Int32(...).ir_value(loc=, ip=)` / `cutlass.Float32(...).ir_value(loc=, ip=)`,
  `cute.arch.fmax`, `cute.arch.shuffle_sync_bfly`, `cute.math.exp2(..., fastmath=True)`,
  `cute.make_rmem_tensor`, `cutlass.range_constexpr`, `cutlass.const_expr`.
- GPU certification probe (the probe is not included in this release) -- the same computation (2 rows × 16 columns /
  thread, a quad sharing two rows, an online row reduction over 1024 trips, both row max and row sum as 4-deep tree reductions,
  2 CTA/SM × 256 threads = 4 warps per scheduler (issue-bound geometry, gotcha 11); the per-trip increment is a
  loop-carried fp32 register rather than `Float32(it)` (the latter is an `I2F` on the path common to all variants);
  three quads parked over the whole segment to exercise the empty-row guard; the valid bits are
  a flag plane of one byte per column, 4 words per trip, with column pairs extracted through a `prmt.b32` escape hatch -- opaque to the optimizer,
  so that the `I2F` footprint is deterministic) in three variants
  `i2f` / `sel` / `ifsel`, differing only in the path from valid bit to score, with five contracts judged in the same round: **numeric** each of the three variants
  reproduces the float64 numpy reference (relative error ≤ 2e-3, park rows `l == acc == 0`); **no_spill**
  `LDL == STL == 0`; **sel_footprint** `sel` loop body `I2F == 0 ∧ FSEL ≥ 32 ∧` branch-free,
  `i2f` loop body `I2F ≥ 16 ∧ FSEL ≤ 4`; **fold_into_ex2** `FADD(i2f) − FADD(sel) ≥ 24 ∧
  (FADD+FFMA+FMUL+I2F) total drop ≥ 32 ∧ equal MUFU counts (±2)`; **form_time**
  `t(i2f)/t(sel) ≥ 1.03`. It also prints the two ladder verdict lines `EXPR A=` (readings of how the run-time `if` variant lowers) / `EXPR B=` (`selp`
  footprint) and the maximum ulp distance between the `i2f`↔`sel` outputs. Status: **✅ Verified** (GPU PROBE PASS, max_diff=7.46e-06, all five contracts pass).
