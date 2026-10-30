# branch_free_select_mask (the hardware form of an elementwise mask: predicate + `FSEL`, not "convert the flag bit to float and do arithmetic")

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the sequel to [`softmax_segment_ilp`](softmax_segment_ilp.md): that one prices the
> row-reduction segment in MIO counts and instruction classes, and gives one class lever,
> `fmax` → `FMNMX`. This entry adds the **other half** of the segment's class levers — the mask. The
> product of the `fl.to(Float32)` step in that entry's §(c) example is an `I2F`, which belongs to the
> form this entry eliminates; with this entry certified, that example's mask spelling follows this
> entry.

## Purpose

An elementwise segment has to "keep or void" scores according to **one valid bit per column**. In
hardware this is one `FSEL` (predicated select): `s = p ? x : PARK`. CuTe DSL exports no select
primitive; an `if` statement is a region, and whether it lowers to `FSEL` is up to the optimiser. This
entry gives a **deterministic spelling** — the `setp` + `selp.f32` escape hatch inside a `@dsl_user_op`
([`ptx_escape`](ptx_escape.md) form 3) — and the segment form that comes with it: the row max is done
in the **raw domain**, and the scale constant is folded into the `FFMA` that feeds `EX2`.

## When needed (trigger)

In the SASS census (loop body, one thread holding R rows × K columns) all of these appear together:

1. **`I2F` ≥ K**: one integer→float conversion per column — the flag bit has been `.to(Float32)`ed.
2. **`FSEL` ≈ 0** (only a stray few left, from other guards): the mask does not go through predicated
   select.
3. **What feeds `MUFU.EX2` is a standalone `FADD`** (`s - m`), not an `FFMA`: the scale was already
   multiplied into `s` at the mask step, and pass 3 has only the subtraction left to do.
4. The segment's dominant stall is `short_scoreboard` / `wait`, and the segment sits on a serial
   handshake chain (the other side is waiting for it).

These four together = the mask takes the "I2F form": per column `I2F` + `FMUL`(cv) + `FFMA`(bv), per
element `FFMA`(x·cv + bv), per element in pass 3 `FADD` + `MUFU`. (Count `I2F` by opcode prefix: on
sm_90 a 32-bit integer→float may also be spelled `I2FP.F32.U32`.)
**Item 1 not appearing does not mean it is already the select form**: if the valid bit is a
single-bit register value the optimiser can see through (`& 1` on a word straight out of `LDS`),
`.to(Float32)` may be lowered to predicated `FMUL`/`FFMA` — `I2F == 0` and `FSEL` only a stray few, but
the two arithmetic instructions per column and pass 3's standalone `FADD` are untouched, and the clock
is still ~5% off (first-round reading of the certification probe). When the valid bit is extracted via
`prmt`/inline asm or a byte read (opaque to the optimiser), `I2F` shows up one per column. Judge the
form by items 2 and 3; item 1 is only corroborating evidence.

`I2F` and `MUFU` are in the same throughput tier (CUDA Programming Guide arithmetic-instruction
throughput table: 32-bit integer→float conversion and `exp2f` are both 16 results per SM per clock,
`FFMA` is 128), so K `I2F` are a **same-pipe** load stacked directly on R·K `EX2`, not free ALU work.

## Mechanism: the instruction ledger of the three forms (per trip, per thread, R rows × K columns)

| Form | Per column | Per element (mask) | Row-max domain | Per element in pass 3 | Notes |
|---|---|---|---|---|---|
| **I2F form** (to be eliminated) | bit extract + `I2F` + `FMUL`(cv) + `FFMA`(bv) | `FFMA`(x·cv + bv) | scaled domain | `FADD`(s − m) + `MUFU.EX2` | `I2F` and `EX2` share a pipe |
| **select form** (this entry) | bit extract (`LOP3`, often folded with `setp` into one `LOP3` with a predicate output) | `FSEL`(p ? x : PARK) | **raw domain** | `FFMA`(s·C − m·C) + `MUFU.EX2` | one extra `FMUL` per row (m·C) |
| **runtime `if` form** (DSL native) | bit extract + `ISETP` | up to the optimiser: `FSEL`, or a `BSSY`/`BRA` region | raw domain | same as above | the certification probe's `ifsel` variant only reports a reading, **no guarantee** |

There are only two points:

- **A mask is a selection, not arithmetic.** The valid bit is the source of a predicate; do not turn it
  into 0.0/1.0 and multiply. One `selp.f32` parks invalid columns at a large finite negative number
  (something like `-1e30`, not `-inf`, see gotcha 3).
- **Defer the scaling to the `FFMA` right before `EX2`.** Do the row max on the raw scores (`FMNMX` is
  insensitive to monotone scaling; `max(x)·C = max(x·C)` up to rounding), multiply by `C` once per row
  to get `m·C`, then `p = exp2(fma(s, C, −m·C))` — the `FFMA` feeds `MUFU` directly, and the standalone
  `FADD` in pass 3 disappears.

## Minimal runnable example

### (a) Two select primitives (PTX escape hatch, pure functions, no side effects)

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


@dsl_user_op
def select_f32(cond, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """cond != 0 ? a : b   (cond: Int32; any non-zero integer counts as true)  -> one FSEL"""
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

The `.reg .pred p` inside `{ ... }` is block-scoped: each call site gets its own copy, so it never
clashes however many times it is called in one kernel. Inline asm can only pass `r`/`f`-class operands,
and a predicate cannot be passed across asm blocks, so the comparison and the selection must be written
in **the same** asm snippet.

### (b) Segment form: select mask → raw-domain row max → one FFMA into EX2

```python
C = 0.7213475                  # scale constant (compile-time constant or uniform; see gotcha 4)
PARK = -1.0e30                 # large finite negative number, not -inf (gotcha 3)
NEG_TEST = -1.0e29
fmax = cute.arch.fmax          # if missing, see the max.f32 escape hatch in ptx_escape

h = sFlags[flag_word_index]    # this thread's K valid bits for this trip (one Int32 word)
for c in cutlass.range_constexpr(K):
    bit = h & (1 << c)         # non-zero means valid; do not .to(Float32)
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

1. `I2F == 0` (≥ K before the change, counting `I2F`/`I2FP` by prefix; if the valid bit is a register
   bit the optimiser can see through, it may already be 0 before the change); `FSEL ≥ R·K` (≈ 0
   before); no `BRA`/`BSSY` in the loop body apart from the back edge.
2. `FADD` drops by ≥ three quarters of R·K (pass 3's `s − m` disappears; the row-sum `FADD` stays);
   `FADD + FFMA + FMUL + I2F` together drop by ≥ 2·K.
3. The `MUFU.EX2` count is unchanged: the exp2 work itself did not move, only the class of instruction
   feeding it.
4. `LDL == STL == 0`.
5. Numerics: not bit-equal to the old form (`FFMA` contraction and the scaled domain differ, one
   rounding each), relative error on the order of 1e-6; relative error against the numpy reference
   ≤ 2e-3 (accumulation of `ex2.approx`).

## Gotchas

1. **`.to(Float32)` is that `I2F`.** Any spelling that "turns the valid bit into 0.0/1.0 and
   multiplies" brings it back — including the seemingly cheaper `s * fl + (fl − 1) * M`. The valid bit
   may only be used as a predicate. The optimiser can occasionally fold the conversion of a 0/1 value it
   sees through into predicated arithmetic (`I2F` disappears), but that only changes the spelling: pass
   3's `FADD` and the two arithmetic instructions per column are still there — items 2 and 3 of "When
   needed" are the criterion.
2. **A runtime `if` is not a spelling of select.** `if bit == 0: s = PARK` is an scf region in the DSL;
   in some versions / some contexts the optimiser folds it into `ISETP` + `FSEL`, in others it stays as
   `BSSY`/`BRA`; even when folded into `FSEL` it is two instructions (compare + select), whereas in the
   `selp` form the comparison is often merged with the bit extract. For determinism use §(a). The
   reading of the certification probe's `ifsel` variant is in the `EXPR A=` line of "Sources".
3. **Park with a finite number, and guard empty rows.** Parking at `-inf`, a row with no valid columns
   gets `(-inf) − (-inf) = NaN`; parking at `-1e30`, a row with no valid columns gets `exp2(0) = 1` —
   every voided column gets full weight. The guard is one more select: `mu = (m > NEG_TEST) ? m·C : 0`,
   after which `exp2(s·C − 0)` underflows to 0 on the park value. Don't write this step as a runtime
   `if` either.
4. **`C` must be a constant or warp-uniform.** It is used once per element; if it is an ordinary
   register value, `FFMA`'s second operand takes a register (harmless), but if it has to be recomputed
   every trip that adds K instructions of work. A compile-time constant is best (the `FFMA` carries an
   immediate / constant-bank operand directly).
5. **Use `(m_old − m) · C` for the rescale factor, don't write `fma(m_old, C, −m·C)`.** `m·C` has
   already been rounded, so when the two are equal the fused form leaves a ~ulp(m·C) residue; on the
   park value (order 1e30) that residue is on the order of 1e22, and `exp2` outputs inf / 0 outright.
   Subtract-then-multiply is exactly 0 when `m_old == m`.
6. **Constraint string and placeholders**: the output `=f` comes first and maps to `$0`; an integer
   predicate source uses `r` (Int32), fp32 uses `f`; placeholders are `$n`, not `%n`
   ([`ptx_escape`](ptx_escape.md) gotchas 2, 3).
7. **`has_side_effects=False`.** Select is a pure function; marking it side-effecting blocks CSE and
   scheduling (when two rows share one column's predicate, ptxas merges the `setp` into one — provided
   it can move freely).
8. **Don't compare the 0/1 from the bit extract again.** `bit = h & (1 << c)` is true when non-zero,
   and `setp.ne.s32 p, bit, 0` consumes it directly; `(h >> c) & 1` costs an extra `SHF` and tempts you
   into `.to(Float32)`.
9. **No `from __future__ import annotations` at file level** (`@dsl_user_op` evaluates annotations at
   runtime).
10. **Read the flag words that feed the predicates before the blocking wait.** They are `LDS`; if the
    barrier guarding them is not the one you are about to block on, put the reads before the block so
    the `LDS` latency lands inside the wait instead of on the other side's critical path (`bar.sync`
    only orders the shared accesses of its own participants, see
    [`softmax_segment_ilp`](softmax_segment_ilp.md) §(a)).
11. **A class lever only pays off when the segment is issue-bound.** What it saves is issue slots: about
    2·K + R·K instructions per trip per thread (K `I2F` + 2K per-column arithmetic + R·K `FADD`). If the
    segment is latency-bound — only 1–2 warps per scheduler, and the row reduction written as a 15-deep
    serial `FMNMX` / `FADD` chain — these instructions were already hidden in their own stalls, and the
    clock does not move at all (second round of the certification probe: 1 CTA/SM + serial chains,
    `t(I2F form)/t(select form) = 0.993`, while the SASS footprint had already changed as expected).
    Only after changing the reduction chain into a tree (the same 15 instructions, 4 deep) and giving
    each scheduler ≥ 4 warps did the same pair of forms separate on the clock.
    Order of judgement: first check the SASS footprint (§(c) items 1–3) to confirm the form changed, then
    look at the clock; if the clock does not move, first check whether the segment is latency-bound —
    do not revert the form.
    Spelling of the tree reduction: **put it in an undecorated plain Python helper** (unrolled at trace
    time, emitting only 15 instructions). If
    "`while len(v) > 1: v = [op(v[2i], v[2i+1]) ...]`" is written directly in the `@cute.kernel` body,
    the DSL's AST rewrite treats it as a runtime loop, and the loop-carried value changing from
    `list<16>` to `list<8>` raises `TYPE_UNSTABLE_JOIN` (this is exactly the compile failure of the
    certification probe's third round).

## Sources

- PTX ISA: `selp` (select by predicate, `selp.f32 d, a, b, p`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>;
  `setp` (compare and write a predicate, `.ne.s32` / `.gt.f32`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>;
  `max` (single `max.f32`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>;
  for block-scoped register declarations in inline asm (`{ .reg .pred p; ... }`) see PTX ISA
  "Statements / Scope" and the CUDA "Inline PTX Assembly" document
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>.
- CUDA C++ Programming Guide, "Arithmetic Instructions" throughput table (32-bit integer→float
  conversion and `exp2f` in the same tier, 16 per SM per clock; `FFMA` 128)
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>.
- NVVM/LLVM inline asm constraint letters (`r` / `f` / `=f`) <https://llvm.org/docs/NVPTXUsage.html>.
- Public infra API symbols: `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.cutlass_dsl.T`,
  `cutlass._mlir.dialects.llvm.inline_asm` (`asm_dialect=llvm.AsmDialect.AD_ATT`),
  `cutlass.Int32(...).ir_value(loc=, ip=)` / `cutlass.Float32(...).ir_value(loc=, ip=)`,
  `cute.arch.fmax`, `cute.arch.shuffle_sync_bfly`, `cute.math.exp2(..., fastmath=True)`,
  `cute.make_rmem_tensor`, `cutlass.range_constexpr`, `cutlass.const_expr`.
- GPU-certified probe (not included in this release) — three variants
  `i2f` / `sel` / `ifsel` of the same computation (2 rows × 16 columns per thread, a quad sharing two
  rows, online row reduction over 1024 trips, row max and row sum both 4-deep tree reductions,
  2 CTA/SM × 256 threads = 4 warps per scheduler (issue-bound geometry, gotcha 11); the per-trip
  increment is a loop-carried fp32 register rather than `Float32(it)` (the latter would be an `I2F` on
  the path common to all variants); three quads parked as a whole to exercise the empty-row guard; the
  valid bits are a flag plane of one byte per column, 4 words per trip, with column pairs extracted via
  a `prmt.b32` escape hatch — opaque to the optimiser, so the `I2F` footprint is deterministic) that
  differ only in the path from valid bit to score, with five contracts judged in the same round:
  **numeric** each of the three variants reproduces the float64 numpy reference (relative error
  ≤ 2e-3, park rows `l == acc == 0`); **no_spill** `LDL == STL == 0`; **sel_footprint** `sel` loop
  body `I2F == 0 ∧ FSEL ≥ 32 ∧` branch-free, `i2f` loop body `I2F ≥ 16 ∧ FSEL ≤ 4`; **fold_into_ex2**
  `FADD(i2f) − FADD(sel) ≥ 24 ∧ (FADD+FFMA+FMUL+I2F) total drop ≥ 32 ∧ equal MUFU counts (±2)`;
  **form_time** `t(i2f)/t(sel) ≥ 1.03`. It also prints two ladder-verdict lines, `EXPR A=` (how the
  runtime-`if` variant lowers) / `EXPR B=` (the `selp` footprint), and the maximum ulp distance between
  the `i2f`↔`sel` outputs. Status: **✅ Verified** (GPU PROBE PASS, max_diff=7.46e-06, all five
  contracts passed).
