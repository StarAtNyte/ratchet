# mbarrier_try_wait_fastpath_layout (how an mbarrier wait is laid out in SASS: ptxas outlines a plain retry loop but not a hand-written "fast path + spin"; footprints of the blocking form and of a `test_wait` spin)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; readings come from sm100 SASS (`nvdisasm -c`).
>
> **Status: ✅ Verified (layout contracts + numerics)** — certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=5.961e-08` (certification round 2; the four forms are bit-identical, the `rsqrt.approx` error bound holds,
> and both layout contracts, "plain retry loop = outlined" and "blocking form = inline_backedge, hand-written fast path = inline_forward", passed in the same round).
> Supplementary readings for the receiver-side `test_wait` spin / blocking form come from a second probe, the receiver-tail probe,
> `PROBE PASS max_diff=5.961e-08` (round 1).
> **No whole-kernel gain is certified**: both whole-kernel readings are inside the noise (see "When needed").

This entry only covers **what layout the same wait gets in SASS**. The axis "suspend or stay resident and spin when the wait is not yet satisfied" is in
[`mbarrier_wait_encoding`](mbarrier_wait_encoding.md); the two are orthogonal.

## Purpose

For an acquire wait on the same mbarrier and the same phase there are five common spellings in CuTe DSL / inline PTX. The synchronisation semantics are identical
(all `.acquire.cta`); the difference is **where ptxas puts the retry and how many wait instructions stay on the straight-line path**:

| Spelling | Source | SASS layout (measured) |
|---|---|---|
| dsl | `ok = cute.arch.mbarrier_try_wait(mbar, ph)` + `while ok == 0:` retry | **outlined** |
| asm_loop | one asm: `L: try_wait.parity p; @!p bra L;` | **outlined** |
| asm_fast | one asm: `try_wait.parity p; @p bra DONE; S: try_wait.parity p; @!p bra S; DONE:` | inline_forward |
| blocking | `cute.arch.mbarrier_wait(mbar, ph)` | inline_backedge (with `NANOSLEEP.SYNCS`; in this entry's two skeletons. In [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (f) the same spelling has out-of-line wait sites in another kernel) |
| test_spin | one asm: `mbarrier.test_wait.parity` + DSL `while` retry | inline_backedge (`SYNCS.PHASECHK.TRANS64`, no `.TRYWAIT`) |

- **outlined**: the straight-line path has only one `SYNCS.PHASECHK.TRANS64.TRYWAIT P0` and one `@!P0 BRA`; the branch target is a retry block after the first
  `EXIT` (inside the block: `TRYWAIT` + a `@!P0 BRA` self-loop + a `BRA` back to the join point). When the data has already arrived, this stretch costs only one
  `TRYWAIT` plus one not-taken branch, and the following `LDS` issues right after.
- **inline_forward** (hand-written fast path): ptxas keeps both `TRYWAIT`s as written: `TRYWAIT P0` → `@P0 BRA` (over the second one) →
  `TRYWAIT P0` → `@!P0 BRA` (only this one jumps to the outlined block). The success path takes **one extra jump**.
- **inline_backedge** (blocking form): `TRYWAIT P0` → `@!P0 NANOSLEEP.SYNCS 0x989680` → `@!P0 SYNCS.PHASECHK.TRANS64 P0` →
  `@!P0 BRA` back to the first instruction — the retry back edge stays on the straight-line path, and each failure carries a suspend hint.
- **test_spin**: on the straight-line path, `SYNCS.PHASECHK.TRANS64 P0` (a pure query, not `.TRYWAIT`) + an `@!P0 BRA` back edge.

Conclusion (layout only): **ptxas outlines a plain try_wait retry loop** (a DSL `while` or a single-asm self-loop; the loop body is moved after `EXIT`);
**a hand-written fast path + spin is not merged**. This entry gives no evidence of a gain that would justify changing the spelling for the layout (see "When needed").

## When needed (trigger)

- Before rewriting a wait as "fast path + outlined spin": first check the current SASS with the classifier of gotcha 6 below; if it is already outlined there is nothing to cash in.
- The current spelling is `cute.arch.mbarrier_wait`, the wait site has an `@!P0 NANOSLEEP.SYNCS` back edge on the straight line in SASS, and you want to know what the layout becomes after switching to a try_wait loop.
- **Whole-kernel readings (both inside the noise; no gain is claimed)**:
  - one whole-kernel lever pricing: replacing the blocking `mbarrier_wait` with a single-asm try_wait self-loop (the asm_loop form),
    alone **−3.18%** (negative = slower), removed from the full set 0.00%, noise band of that run 7.61%, ruled null;
  - one whole-kernel same-job A/B: hand-written fast path (inline_forward) against the DSL loop (outlined), cold start −0.04% (iqr 0.0320),
    warm +0.29% (fast path slower, iqr 0.0077, slower in 22 of 24 rounds; the negative sign at cold start = fast path faster), all magnitudes inside the iqr.
  In other words: the layout difference cannot be measured in whole-kernel timing; if you ship it, do a paired re-measurement in the same job.
- **Anti-triggers**:
  - Already outlined: changing the spelling will not change the SASS any further.
  - Time is mostly spent waiting for the slowest peer (wait start → wait end is much larger than wait end → result): the layout only affects the fast path, no measurable gain.
  - Switching to a pure-query `test_wait` spin: its footprint is likewise inline_backedge (a `SYNCS.PHASECHK.TRANS64` back edge on the straight line), and there is no whole-kernel reading.

## Minimal runnable example

```python
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
ATT = llvm.AsmDialect.AD_ATT

# (a) DSL try_wait retry loop. ptxas outlines it (outlined)
ok = cute.arch.mbarrier_try_wait(mbar, phase)      # mbar = sMbar.iterator
while ok == 0:
    ok = cute.arch.mbarrier_try_wait(mbar, phase)

# (b) the equivalent inline-PTX spelling (for when it must be combined with other asm blocks); also outlined
@dsl_user_op
def wait_loop(mbar_addr, parity, *, loc=None, ip=None):
    """mbar_addr is a 32-bit smem address (.iterator.toint())."""
    llvm.inline_asm(
        None, [Int32(mbar_addr).ir_value(loc=loc, ip=ip), Int32(parity).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n"
        "WAIT_L:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra WAIT_L;\n\t}",
        "r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# Contrast: try_wait; @p bra DONE; SPIN: try_wait; @!p bra SPIN; DONE:
# The measured SASS is inline_forward (both TRYWAITs on the straight line), see gotcha 4.
# Read smem after the wait with an asm that carries ~{memory} (see gotcha 2).
```

## Certification skeletons and numerics

Both probe skeletons are "in-cluster `st.async` + mbarrier `complete_tx` exchange → wait → read back the smem slots → fixed addition tree →
`rsqrt.approx`", with only the wait spelling changed; skeleton A has a longer read-back stretch after the wait, skeleton B a shorter one.

- All spellings give bit-identical results; the `rsqrt.approx` relative error against float64 is ≤ 5.961e-8.
- Layout is judged with the classifier from the "Purpose" table (gotcha 6); on both skeletons every spelling has the same layout.
- Both probes also read in-segment `%clock` cycles (single INFO reads); this entry does not rank spellings by them.
  Launch-loop timing (skeleton A, the four at 9.002–9.050 µs/call) is swamped by launch overhead and **cannot resolve a layout difference**.

## Gotchas

1. **Put PTX labels inside `{ }`**: a label is scoped to its block, so inlining the same asm several times does not duplicate names. Without braces, the second
   inlining in the same function is a duplicate label and ptxas errors out.
2. **An asm's `~{memory}` / `has_side_effects=True` only constrains LLVM; ptxas can still rewrite the shape of memory accesses.** Writing the
   `ld.shared` after the wait as an asm with `~{memory}` is what keeps the reads from being hoisted above the wait (the probe's SASS contract: all reads after `TRYWAIT`).
   But ptxas will still **merge** adjacent, 16 B-aligned scalar reads: in skeleton B, several independent `ld.shared.f32` asms produce SASS identical, instruction for instruction, to the version
   written as `ld.shared.v4` (both become `LDS.128`). Do not count on "one asm per read" to preserve the number or width of reads.
   `try_wait` / `test_wait` without a qualifier = `.acquire.cta`; after success, the data accounted by `complete_tx` in this phase is visible;
   do not change it to `.relaxed`.
3. **Pass parity as a register operand** (`"r"`) and flip it per phase every time the mbarrier is reused; written as an immediate it can only wait on one phase.
4. **Block layout is decided by ptxas per wait site; there is no "cold block" annotation at the PTX level.** A plain self-loop (`L: try_wait; @!p bra L`) is
   rotated by ptxas itself into "inlined once + outlined back edge"; a hand-written fast path + spin, on the contrary, is **not** merged. Moreover "DSL loop = outlined"
   **does not hold at every wait site**: for a second DSL try_wait loop immediately following in the same kernel (waiting on another mbarrier), the SASS is
   `TRYWAIT` → `@P0 BRA` (skip) → `TRYWAIT` → `@!P0 BRA` (outlined block), i.e. one extra `TRYWAIT` on the straight-line path
   (SASS output of one job, INFO). Inspect the SASS of every critical wait site separately.
   Another certification ([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) section (f), one wait call site per warp) found that the
   straight-line wait sites of one kernel are likewise a few outlined and most in place, consistent with this entry's "decided per site"; in that certification the DSL blocking form (`cute.arch.mbarrier_wait`)
   also showed "fast-path `TRYWAIT` → `@!P BRA` into a retry block after EXIT" — i.e. **the blocking form can also go out-of-line**;
   in this entry's two skeletons the blocking form happens to be inline_backedge in both. Both sets of readings stand; go by the per-site readings of your own SASS.
5. **`cute.arch.mbarrier_wait` is inline_backedge with `NANOSLEEP.SYNCS` in this entry's two skeletons** (in the sm100 certification of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (f)
   the same spelling also showed an out-of-line fast path — again decided per site, see gotcha 4). Before changing the spelling, read the whole-kernel readings under "When needed" —
   both were inside the noise; no gain has ever been cashed in.
6. **Acceptance method**: `cute.compile(..., options="--dump-dir=<d> --keep-cubin")`, `nvdisasm -c <cubin>`, parse the `.L_x_N:`
   labels and `` BRA `(.L_x_N) `` targets: a predicated `BRA` within ≤ 3 instructions after the first `TRYWAIT` whose target lies after the first `EXIT` and whose target block contains
   `TRYWAIT` = outlined; target ≤ the first `TRYWAIT` = inline_backedge; anything else = inline_forward. Then confirm whole-kernel time with paired timing in the same job.
7. **The ending `%clock` read must be predicated on the result** (e.g. `setp.nan.f32 p, x, x; @!p mov.u32 t, %clock;`); otherwise ptxas may schedule it before
   the reduction and the in-segment cycle count reads low.

## Sources

- PTX ISA: `mbarrier.try_wait.parity` (default `.acquire.cta` semantics, the `waitComplete` predicate, suspendTimeHint),
  `mbarrier.test_wait.parity`, `bra`, predicated execution `@p`, labels and statement-block scope, `ld.shared`, `%clock`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`nvdisasm`, SASS `SYNCS.PHASECHK`, `BRA`, `EXIT`, `NANOSLEEP`, `LDS`):
  https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `cute.arch.mbarrier_try_wait`, `cute.arch.mbarrier_wait`, `cute.arch.mbarrier_init`,
  `cute.arch.mbarrier_init_fence`, `cute.arch.mbarrier_arrive_and_expect_tx`, `cutlass.utils.SmemAllocator.allocate_tensor`,
  `llvm.inline_asm` (`dsl_user_op`), `cute.compile(options="--dump-dir=… --keep-cubin")`.
- GPU certification probes: the fast-path layout probe (✅ `PROBE PASS max_diff=5.961e-08`, round 2);
  the receiver-tail probe (✅ `PROBE PASS max_diff=5.961e-08`, round 1).
