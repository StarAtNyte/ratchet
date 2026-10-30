# ldg_issue_order_pin (pinning every `LDG` before the first FP consumer: the dependence-token and single-trip-loop pins and the cost of each)

> One entry per hardware interface. Operator-agnostic; mechanism only.

> **Status**: ✅ Verified. Each of the two pins has one GPU certification, on the same toy shape (several groups of 128-bit `LDG` per thread):
> - dependence token: `PROBE PASS max_diff=0.00391` (round 2 of that certification; the numeric and SASS contracts both pass in the same round);
> - single-trip loop (keeping `.nc`): `PROBE PASS max_diff=0.00391` (round 2; the numeric and SASS contracts both pass in the same round).
>
> **What is certified**: only **the numerics and the static SASS order** (whether all `LDG` are scheduled before the first FP instruction after the first `LDG`, `LDL/STL` is 0,
> and for the single-trip loop additionally `.CONSTANT` and no `DEPBAR`). The per-CTA cycle readings in the toy are only a single-run INFO. **The whole-kernel gain is a mixed reading** (see "Whole-kernel readings of the same mechanism"):
> whether to pin, and with which pin, depends on whether ptxas actually sinks loads in your SASS.

## Purpose
A thread issues several mutually independent groups of global loads at the start and only then begins floating-point work. To use fewer registers, ptxas
may move the `LDG` of later groups backward, past the first FP consumer of the earlier group, even past a `BAR`. Once moved,
the moved loads are issued only after the earlier group's data has **arrived** and its consumers are running, so an extra serial memory round trip appears.
This entry gives two spellings that pin every `LDG` before the first FP instruction at the SASS level, the cost of each, and several spellings that **look like they pin but measurably do not**.

| Pin | How it constrains ptxas | Cost |
|---|---|---|
| A dependence token | xor the result words of the later groups → AND with a parameter that is 0 at run time → OR into **every** consumer of the early-consumed group | the early-consumed group waits until every load has landed (a landing gate); one extra integer instruction per word |
| C single-trip loop | write every load into **the same** inline PTX block, wrapped in a backward-branching loop that runs only once | ptxas peels / duplicates the loop, multiplying the static `LDG` count; more registers than not pinning (on one instance of the same whole kernel C used about 18% more than not pinning, A somewhat more still) |

The construction of the dependence token itself (xor / AND with a run-time 0 parameter / OR) is the same as the
`fence` of [`post_wait_gather_single_trip`](post_wait_gather_single_trip.md); that entry covers "a batch of gathers after a dependent wait pays only one round trip", this entry covers "the issue position of the load groups at the start" and the trade-off between the two pins.

## When needed (trigger)
- **The criterion looks at SASS, not source.** In the linear SASS (`cuobjdump -sass`), some `LDG` has an index greater than that of the first
  `FADD`/`FMUL`/`FFMA`/`MUFU`/`HFMA*` **after the first `LDG`** (FP instructions scheduled before all `LDG` are address / initialization arithmetic, e.g. the floating-point reciprocal path of an integer division;
  they cannot consume load data and do not count). The common shape is the early-consumed group's `LDG` crowded at the start and the other `LDG` landing in the middle of the FP chain, or around a `BAR`.
- The source already writes all loads first; marking the load asm `has_side_effects=True`, moving the later-used groups to the very top of the source, or stuffing all loads
  into the same asm statement leaves the SASS completely unchanged (see gotcha 1).
- The register count is clearly below what "all load results live at the same time" needs, meaning ptxas is trading issue position for registers.
- **Conversely, if no `LDG` sinks in the SASS, do not pin.** After an arithmetic rewrite of the same kernel ptxas may no longer sink loads; then any pin is pure cost
  (whole-kernel reading: not pinning was 2.5% faster, see below). Recount every time you change the arithmetic after the loads.

## Minimal runnable example

### A dependence token: the token enters every consumer of the early-consumed group
```python
from cutlass import Float32, Int32, Int64
import cutlass, cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
ATT = llvm.AsmDialect.AD_ATT

@dsl_user_op
def ld128(addr, *, loc=None, ip=None):            # one LDG.E.128, 4 result words
    s = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4), [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), s, [i], loc=loc, ip=ip)) for i in range(4)]

@dsl_user_op
def dep_tok(z, words, *, loc=None, ip=None):
    """(xor of all words) & z. z is a kernel parameter that is 0 at run time, so the value is always 0;
    ptxas cannot prove it, so every instruction that uses it must be scheduled after all loads of words."""
    txt = "{\n\t.reg .b32 a;\n\tmov.b32 a, $2;\n"
    for k in range(1, len(words)):
        txt += "\txor.b32 a, a, $%d;\n" % (2 + k)
    txt += "\tand.b32 $0, a, $1;\n}"
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(z).ir_value(loc=loc, ip=ip)] + [Int32(v).ir_value(loc=loc, ip=ip) for v in words],
        txt, "=r,r," + ",".join(["r"] * len(words)),
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))

@dsl_user_op
def widen_t(u, tok, *, loc=None, ip=None):
    """bf16x2 word → two f32, ORing the token into the integer bit pattern first (tok == 0, the bits are unchanged).
    The token lives in the integer domain: a floating-point x + 0*t would be polluted by inf/NaN, and might be folded away."""
    s = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(u).ir_value(loc=loc, ip=ip), Int32(tok).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 v;\n\tor.b32 v, $2, $3;\n\tshl.b32 $0, v, 16;\n\t"
        "and.b32 $1, v, 0xffff0000;\n\t}",
        "=f,=f,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=ATT,
        loc=loc, ip=ip)
    return (Float32(llvm.extractvalue(T.f32(), s, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), s, [1], loc=loc, ip=ip)))

# kernel body (Z: Int32 is a kernel parameter; the host passes 0):
#   early = [w for a in EARLY_ADDRS for w in ld128(a)]   # the group consumed first
#   late  = [w for a in LATE_ADDRS for w in ld128(a)]    # the groups used later
#   tok   = dep_tok(Z, late)                    # depends on every result word of the later groups
#   xs    = [h for u in early for h in widen_t(u, tok)]   # every early-consumed word passes through the token once
```

### C single-trip loop: every load in the same asm block, keeping `.nc`
```python
@dsl_user_op
def ld_v4_pinned(addrs, z, *, loc=None, ip=None):
    """Issue one ld.global.nc.v4.b32 per address in addrs, all wrapped in a backward-branching loop that runs once.
    z is a kernel parameter that is 0 at run time (Int32). Returns the 4 result words of each address, in the order of addrs."""
    n = len(addrs)
    nout = 4 * n
    txt = "{\n\t.reg .pred p;\n\t.reg .b32 c;\n\tmov.b32 c, $%d;\nPIN_LOOP:\n\t" % (nout + n)
    for g in range(n):
        txt += "ld.global.nc.v4.b32 {$%d, $%d, $%d, $%d}, [$%d];\n\t" % (
            4 * g, 4 * g + 1, 4 * g + 2, 4 * g + 3, nout + g)
    txt += "sub.s32 c, c, 1;\n\tsetp.ge.s32 p, c, 0;\n\t@p bra PIN_LOOP;\n}"
    s = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * nout),
        [Int64(a).ir_value(loc=loc, ip=ip) for a in addrs] + [Int32(z).ir_value(loc=loc, ip=ip)],
        txt, ",".join(["=r"] * nout + ["l"] * n + ["r"]),
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
    w = [Int32(llvm.extractvalue(T.i32(), s, [i], loc=loc, ip=ip)) for i in range(nout)]
    return [w[4 * g:4 * g + 4] for g in range(n)]

# kernel body: groups = ld_v4_pinned(ADDRS, Z)   # ADDRS = every load address
# then consume each group's words directly: no token, and no ORing in of any other group's words
```
c starts at 0; after subtracting 1, `setp.ge` is false and the loop body executes only once; but this is real backward-branching control flow, so ptxas cannot move the `LDG` in the loop body
behind the consumers after the loop.

## Gotchas
1. **These spellings do not pin** (same toy shape, measured on the GPU; each has about one third of the `LDG` sunk past the first FP instruction, the same as not pinning):
   source order (later-used groups written first), marking the load asm `has_side_effects=True`, stuffing every load into one asm statement (without a loop),
   all plain weak loads, all `.nc` plus `bar.warp.sync`, a separate asm conditional branch after the loads as a "basic-block fence",
   all loads written into one asm block followed, inside the same asm block, by a **forward** branch conditioned on Z, a side-effecting `%clock64` read after the loads.
   Inline asm has no boundaries once in PTX, and ptxas splits and reorders it by its own scheduling; on this entry's toy what constrains it is **data dependence** and
   **a backward-branching loop** (the two spellings below). Two other structural fences have been certified on another shape (a batch of loads feeding MMA directly from registers): `membar.cta` after the group (weak loads),
   and putting the whole load group **inside a block skipped by a run-time predicate** (the branch before the loads, the loads inside the block; not the same as "a forward branch after the loads" above),
   see [`ldg_issue_pin_structural_fence`](ldg_issue_pin_structural_fence.md); neither was tested on this entry's toy.
2. **Z must be a run-time kernel parameter.** Written as a constant 0, `x * 0`, or any expression LLVM / ptxas can prove is 0, the token dependence is folded away,
   the single-trip loop is folded into straight-line code, and the SASS reverts to what it was.

A (dependence token)
- A1. **The token must enter every consumer of the early-consumed group, not just the first.** ORing the token into one word constrains only that one word, and ptxas can compute the other words first.
  The probe's "token into one word only" variant happened to line up on this toy as well (0 sunk), but that was the scheduler's choice, not something the dependence guarantees.
- A2. **It is a landing gate.** Every consumer of the early-consumed group waits until the later groups have **all landed**, plus a stretch of xor chain; if you want "compute whatever arrives first", A contradicts it; switch to C.

C (single-trip loop)
- C1. **Every load must be in the same asm block.** Wrapping only some of them in the loop leaves the loads outside the loop still sunk; in the whole kernel the variant "some groups in the loop, the rest outside"
  has `LDG` sunk past FP and also the worst timing (see below).
- C2. **Mark the asm `has_side_effects=True`** (the probe does; whether it would be deleted / moved without it was not tested separately).
- C3. **ptxas peels / duplicates the loop.** In the toy the static `LDG` became 3 times the original count, with several extra `BRA`, and only one copy runs at run time;
  the acceptance check counts "the `LDG` count is an integer multiple of the original". In the same whole kernel, the static `LDG` of both instances also became 3 times the original count.
- C4. Operand numbering: outputs first, addresses next, z last; change them together when the number of groups changes. The label is written inside the asm's `{ }`; whether inlining the same labelled asm block several times in one kernel
  causes duplicate names was not tested (the probe and the whole kernel each have only one call site).

Common
- **The register cost is real.** Once pinned, the result words of the later-used groups are live from the entry all the way to their consumers. After the change, first confirm that `LDL/STL` is still 0.
- **Accept on SASS first, then on same-job timing.** Get the cubin: `CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>` (give every variant its own
  `CUTE_DSL_CACHE_DIR`, or a cache hit will not give you the new cubin), `cuobjdump -sass <cubin>`, split by `Function :`, and count only your own kernel.
  Criterion: every `LDG` index is less than the index of "the first FP instruction after the first `LDG`", the `LDG` count is unchanged (an integer multiple for C), 0 `LDL/STL`;
  for C, additionally every `LDG` carries `.CONSTANT` and there is no `DEPBAR` between the `LDG` and the first FP instruction.

## Certified readings (toy shape: several groups of 128-bit `LDG` per thread, some groups consumed before a block reduction)
SASS (the fraction of all `LDG` sunk past "the first FP instruction after the first `LDG`") and the per-CTA median cycles (single-run INFO):

| Variant | Run | Fraction sunk | Cycles |
|---|---|---|---|
| unpinned (source order) | token probe | 1/3 | 2468 |
| `has_side_effects=True` / single asm statement | token probe | 1/3 | 2458 / 2444 |
| A, token into one word only | token probe | 0 | 2329 |
| A, token into every word | token probe | 0 | 2322 |
| C single-trip loop | loop probe | 0 (static `LDG` 3 times the original count) | 2405 |

- A with the token into every word: about 12% more instructions, 0 `LDL/STL`.
- Same-job unpinned / pinned cycle ratio: A 1.063 (dependence-token certification run), 1.060 (single-trip-loop certification run); C 1.029. This toy measures the duration of the whole stretch, not how long the first consumer of the early-consumed group waited; single readings, with no significance certification.
  **Do not use it to rank the two pins**; look at the whole-kernel readings in the next section.

## Whole-kernel readings of the same mechanism (a latency-bound straight-line row-reduction kernel, same-job pairing, geometric mean over 6 shapes)
All ratios below are "control time / variant time": > 1 means the variant is faster. The control is the A then in use in the whole kernel.
- **When ptxas really sinks loads, pinning is the big win.** Unpinned, one instance had `LDG` sunk past the first FP instruction. With A pinning it,
  the ratio of the "token removed" variant is 0.903 (i.e. unpinned is about 11% slower); an earlier run looking only at the 4 small-batch shapes gave 0.973.
- **When ptxas no longer sinks loads, pinning is pure cost.** After an arithmetic rewrite after the loads of the same kernel, one instance had 0 sunk even unpinned; the ratio of removing A was then
  1.026, 1.025 / 1.026.
- **C should be used only on the instance that sinks.** Both instances switched to C: 0.983 / 0.987. C only on the instance that sinks `LDG`
  when unpinned, the other instance unpinned: 1.031; on the shapes covered by that instance, C against A is 1.009 / 1.013,
  while unpinned against A is 0.991 / 0.988. Partially in the loop (some groups in, the rest outside): 0.868.

## Sources
- PTX ISA: the `.nc` qualifier of `ld`, the `.b32` forms of `xor` / `and` / `or`,
  `setp` / `bra` / predicated execution, `.reg` and labels inside a `{ }` block: https://docs.nvidia.com/cuda/parallel-thread-execution/ ;
  register constraints of inline PTX: https://docs.nvidia.com/cuda/inline-ptx-assembly/
- CUDA Binary Utilities (`cuobjdump -sass`): https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `llvm.inline_asm` (`dsl_user_op`, `has_side_effects`), `cute.compile`,
  environment variables `CUTE_DSL_KEEP_CUBIN` / `CUTE_DSL_DUMP_DIR` / `CUTE_DSL_CACHE_DIR`.
- Certified by two GPU probes (the probes are not included in this release):
  dependence token (`PROBE PASS max_diff=0.00391`),
  single-trip loop keeping `.nc` (`PROBE PASS max_diff=0.00391`).
