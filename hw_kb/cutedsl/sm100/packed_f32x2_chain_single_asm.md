# packed_f32x2_chain_single_asm (sm100)

> One file per hardware interface. Operator-agnostic: mechanism only.
> Architecture scope: **sm100 (Blackwell) only.** The `.f32x2` ALU forms do not
> exist on sm90; nothing here transfers.
> Sibling entry: `sm100/rmem_materialize_packed_reduce` covers the *single*
> packed op reached through the DSL and its register-placement precondition.
> This entry covers the case that one does not: a **chain** of dependent packed
> steps.

## Purpose

sm100 adds `add` / `sub` / `mul` / `fma` in a `.f32x2` form that retires two
fp32 lanes in one instruction, operating on an aligned 64-bit register pair.
A chain of these is reachable only through inline PTX, and how you *spell* the
chain decides both whether it compiles and what it costs: write the whole chain
— temporaries, broadcast constants, every packed step — inside **one** inline-asm
block, and let only plain f32 scalars cross the constraint boundary.

The opposite spelling (one asm block per packed instruction, packed values
travelling between blocks as `l`-constrained i64 values, constants pre-packed
outside) is numerically identical but structurally worse: each block is an
optimisation barrier the compiler cannot schedule across, and each crossing
re-imposes the aligned-pair constraint at a point you did not choose.

## When needed (trigger)

- You have an arithmetic chain of **two or more dependent steps** that you want
  to run two lanes at a time.
- You already have a working single-step packed helper (pack, one instruction,
  unpack, all in one block) and the obvious generalisation — call it N times —
  either loses its saving to register moves or stops compiling.
- Symptom: **NVVM dies with no diagnostic** the moment a chain helper is called
  from inside a real loop body, while the same helper compiles standalone and
  every variant that removes only the call compiles. That signature is a spelling
  problem; see Gotchas.
- Symptom: a packed instruction whose source operand is a hex-float literal, or
  a braced list of them.
- You are about to price a packed chain against a single hardware instruction
  that does the same job, and you need the instruction count to be a property of
  the mechanism rather than of your spelling.

## Minimal runnable example

### The shape

```python
import struct
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

F32 = cutlass.Float32
_PAIR_STRUCT = "!llvm.struct<(f32, f32)>"


def _f32(v: float) -> float:
    """Round a Python float to the nearest float32 value."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


def _hexf(v: float) -> str:
    """PTX hex-float literal for an exact float32 value."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", v))[0]


# Whatever your chain's constants are. Generic values here on purpose.
K = [_f32(v) for v in (1.2345679, 0.31830987, -0.6931472, 2.7182817)]

# ONE block. Operand numbering: a 2-field struct result takes $0 and $1, so the
# two f32 inputs are $2 and $3.
_CHAIN_ASM = (
    "{\n\t"
    ".reg .f32 fk0, fk1, fk2, fk3;\n\t"
    ".reg .b64 a, k0, k1, k2, k3, t;\n\t"
    # pack once, on entry
    "mov.b64 a, {$2, $3};\n\t"
    # Packed ALU ops take REGISTER operands only -- there is no immediate form.
    # Each broadcast constant is materialised HERE, inside the block, so it
    # never has to be carried in across a constraint.
    f"mov.f32 fk0, {_hexf(K[0])};\n\t" "mov.b64 k0, {fk0, fk0};\n\t"
    f"mov.f32 fk1, {_hexf(K[1])};\n\t" "mov.b64 k1, {fk1, fk1};\n\t"
    f"mov.f32 fk2, {_hexf(K[2])};\n\t" "mov.b64 k2, {fk2, fk2};\n\t"
    f"mov.f32 fk3, {_hexf(K[3])};\n\t" "mov.b64 k3, {fk3, fk3};\n\t"
    # The chain: packed instructions back to back on block-local .b64
    # registers. All four constants stay live across all six steps; no packed
    # value is ever handed back to the DSL and re-read.
    "add.rn.ftz.f32x2 t, a, k0;\n\t"
    "fma.rn.ftz.f32x2 t, t, a, k1;\n\t"
    "fma.rn.ftz.f32x2 t, t, a, k2;\n\t"
    "fma.rn.ftz.f32x2 t, t, a, k3;\n\t"
    "sub.rn.ftz.f32x2 t, t, k0;\n\t"
    "fma.rn.ftz.f32x2 t, t, a, k1;\n\t"
    # unpack once, on exit
    "mov.b64 {$0, $1}, t;\n\t"
    "}\n"
)


def packed_chain(x0, x1):
    """Two lanes through a 6-step packed chain -> (lo, hi) as plain Float32.

    Undecorated on purpose: it returns a PAIR, which no `dsl_user_op` return
    annotation can describe, and it needs none -- there is no control flow in
    here for the preprocessor to rewrite.
    """
    r = llvm.inline_asm(
        ir.Type.parse(_PAIR_STRUCT),
        [F32(x0).ir_value(), F32(x1).ir_value()],
        _CHAIN_ASM, "=f,=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return (F32(llvm.extractvalue(T.f32(), r, [0])),
            F32(llvm.extractvalue(T.f32(), r, [1])))
```

Call it from a kernel exactly like a scalar helper — the packed-ness is invisible
on the DSL side, because it never leaves the block:

```python
@cute.kernel
def k_chain(mIn: cute.Tensor, mOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    i = tidx * 2
    y0, y1 = packed_chain(mIn[i], mIn[i + 1])
    mOut[i] = y0
    mOut[i + 1] = y1
```

### Directed rounding is a suffix, not an instruction

The packed forms carry the same rounding-mode suffixes as their scalar
counterparts (`.rn` / `.rm` / `.rz` / `.rp`). A step that needs a directed
rounding gets it for free:

```python
_ADD_RM_ASM = (
    "{\n\t"
    ".reg .b64 a, b, t;\n\t"
    "mov.b64 a, {$2, $3};\n\t"
    "mov.b64 b, {$4, $5};\n\t"
    "add.rm.ftz.f32x2 t, a, b;\n\t"        # round toward -inf, both lanes
    "mov.b64 {$0, $1}, t;\n\t"
    "}\n"
)
```

This is the difference between a chain that spends an instruction on a separate
conversion and one that does not, and it is worth checking before you accept an
instruction count.

### The build self-check

Before pricing a packed chain against whatever single instruction it is meant to
displace, **count** — and count the spelling you intend to ship:

```
issue slots per element = (instructions in the block, excluding `.reg` declarations) / 2
```

Divide by 2 because the block retires two elements. The example above is 16
instructions for 2 elements, of which 6 are the packed ALU chain and 8 are
one-off constant materialisation that does not scale with chain length — so the
marginal cost of the chain proper is `6 / 2 = 3.00` ALU slots per element.

Two rules for reading that number:

- **Count the one-block spelling, not the multi-block one.** The same arithmetic
  written as N single-instruction helpers costs the extra `mov.b64` pack/unpack
  pairs at every boundary, and those are a property of your spelling, not of the
  mechanism. Pricing the multi-block form and concluding the mechanism loses is
  a measurement of the wrong thing.
- **PTX is the upper bound, SASS is the answer.** `mov` instructions here are
  largely register-allocator constraints rather than work; ptxas removes the ones
  it can. If the count decides a decision, read it back from a SASS census
  (`smem_traffic_triage` and `sm90/ptx_sass_census_criteria` for the
  method) instead of trusting the PTX line count.

## Gotchas

- **Packed ALU ops take register operands only.** There is no immediate-operand
  form of `add/sub/mul/fma .f32x2`. The braced vector syntax `{a, b}` is a `mov`
  operand form, not a general source-operand form, so `{0f…, 0f…}` written
  directly into a packed instruction is not the same thing as a materialised
  pair. Build the constant explicitly: `mov.f32` the hex literal into an `.f32`
  register, then `mov.b64` it into a `.b64` pair.

- **"Compiles standalone, dies in the loop, no diagnostic" is a spelling
  signature, not a toolchain limit.** A packed chain helper can assemble in a
  small standalone translation unit and then kill the backend with no message
  once the same helper is inlined into a large loop body — and the variants that
  remove only the call will compile, which reads like proof that the mechanism
  is unavailable. It is not proof. The one-block spelling above is GPU-certified
  (see Sources). Re-read the operands and the block structure before closing a
  decision as `toolchain-limit`.

- **Do not let packed values cross the constraint boundary.** Handing a packed
  value back to the DSL as an `l`-constrained i64 and reading it into the next
  block is numerically identical (certified: `const_inblock_vs_hoist`,
  max_diff = 0.0) — so this is purely a codegen choice, and one block wins it.
  Inline asm is opaque to the compiler: every extra block is a scheduling barrier,
  and every crossing re-imposes the aligned-pair constraint at a boundary you did
  not choose. Hoisting a *packed constant* out of a loop is the right move when
  the chain is one instruction; it stops being the right move the moment the
  chain has steps, because then the constants can live in the block instead.

- **Operand numbering with a struct result.** A 2-field struct result consumes
  `$0` and `$1`, so the first *input* is `$2`. Getting this wrong is an assembler
  error when the numbers run off the end and silent operand aliasing when they do
  not.

- **The constraint letter picks the PTX register class, not the LLVM type.**
  `f` → `.f32`, `r` → `.b32`/`.s32`, `l` → `.b64`. The letter does not have to
  agree with the struct field type: `mov.b32` / `mov.b64` bridges bit-identically,
  so if the last instruction producing your result writes an integer register,
  `=r` is the correct letter even though the field is `f32`.

- **`.ftz` must match on both sides of any comparison.** With `.ftz`, denormal
  inputs and results flush to zero. A reference computed without flushing will
  disagree on exactly the lanes that land in the denormal range — which is a
  handful of lanes out of thousands, i.e. it looks like a mystery rather than
  like a modifier. Keep intermediates in normal range, or drop `.ftz` on both
  sides.

- **There is no `max.f32x2` / `min.f32x2` on this ISA.** NVVM rejects both for
  `sm_100a`. A chain that needs a clamp has to take that one step scalar, which
  forces an unpack and a repack in the middle of the block — so decide where the
  clamp goes while you are designing the chain, not after. For the reduction
  case, `sm100/rmem_materialize_packed_reduce` covers the 3-input `nvvm.fmax`
  route.

- **Reusing one scratch `.f32` register for several constants.** PTX virtual
  registers are mutable, so `mov.f32 fk, …; mov.b64 k0, {fk, fk}; mov.f32 fk, …`
  is legal — but it serialises the materialisations on a false dependency for no
  benefit. Declare one `.f32` scratch per constant; they cost nothing after
  ptxas.

- **A single packed op does not need any of this.** `cute.arch.add_packed_f32x2`
  is on the DSL surface for exactly that case; use it, and read
  `sm100/rmem_materialize_packed_reduce` for the register-placement precondition
  that decides whether it pays. This entry earns its keep only at chain length
  ≥ 2.

## Sources

- PTX ISA, "Alternate Floating Point Instructions" (`add` / `sub` / `mul` / `fma`
  in the `.f32x2` form, rounding-mode and `.ftz` modifiers)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-instructions>
- PTX ISA, "Data Movement and Conversion Instructions: `mov`" (the braced vector
  operand form, and that it is a `mov` form)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov>
- Inline PTX Assembly in CUDA, "Constraints" (register-class letters `f` / `r` /
  `l`, and that the letter selects the PTX register class)
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints>
- CuTe DSL / cutlass public API symbols: `cutlass._mlir.dialects.llvm.inline_asm`,
  `llvm.extractvalue`, `llvm.AsmDialect.AD_ATT`, `cutlass._mlir.ir.Type.parse`,
  `cutlass.cutlass_dsl.T`, `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.Float32`,
  `cutlass.Int64`, `cute.arch.add_packed_f32x2`.
- Sibling entries: `sm100/rmem_materialize_packed_reduce` (single packed op via
  the DSL, plus the 3-input `nvvm.fmax` route for clamps),
  `ptx_escape` (escaping and brace handling in inline-PTX strings),
  `cute_dsl_trace_scope` (why a pair-returning helper stays undecorated).
- GPU certification probe (the probe is not included in this release)
  — **status: ✅ verified on GPU, `PROBE PASS max_diff=0.0`**
  (round 1 of this certification; aggregate reading, all three contracts passing
  in the same run). Three contracts: `chain_single_asm` (6-instruction chain,
  four simultaneously-live in-block packed constants, vs a step-by-step fp32
  reference, tolerance 1e-5 — so the packed chain and the scalar fp32 chain came
  back bit-identical, not merely within tolerance); `const_inblock_vs_hoist`
  (in-block constants vs hoisted i64 operands, bit-identical — which is the
  measurement behind this entry's central claim: the two spellings are a pure
  codegen choice, so anything you price between them is schedule, never
  numerics); `rmode_rm_packed` (`add.rm.ftz.f32x2` equals the largest float32
  ≤ the exact sum, and splits from `.rn` on a non-trivial number of lanes — the
  rounding-mode suffix is honoured in the packed form).
