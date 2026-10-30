# cute_dsl_const_expr_if (compile-time branch: `cutlass.const_expr`)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is the **branch-side**
> counterpart of [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md): that one covers "which code the
> preprocessor rewrites", this one covers "where the scope boundary of the rewritten region lies".

## Purpose

A CuTe DSL kernel is traced Python: the preprocessor walks the AST of `@cute.jit` / `@cute.kernel` and
decides, **statement by statement**, whether each control-flow statement is "evaluated on the spot in
the Python interpreter" or "lowered to IR and left to execute at runtime". For `if` / `elif` / `else` /
`while`, **the default is the latter** — even if the condition is a Python constant, the statement is
lowered into a runtime region.

`cutlass.const_expr(<condition>)` flips that one statement to the former: the condition is evaluated by
Python at trace time, only the selected branch enters the trace, the `if` reverts to an ordinary Python
branch, and bindings inside the branch are visible outward under ordinary Python scoping. It is the
**branch-side paired interface** of `cutlass.range_constexpr` (the loop side, compile-time unrolling) —
since DSL 4.1.0 the default of `if`/`while` flipped from "compile time" to "runtime", and `const_expr` is
the only annotation entry point (`dynamic_expr` was deprecated in the same version; see the official
changelog).

## When needed (trigger)

Three symptoms, all pointing at the same thing:

1. **`NameError: name '<x>' is not defined`, with a stack like
   `ast_helpers.ir_loop → if_region_N → if_executor.if_execute`.**
   `<x>` is **first bound** inside some `if` branch but referenced outside the branch (or in another
   branch). The official limitation clause reads "Values originating in control flow body are not
   available outside the control flow" — a branch is an independent scope, and names newly created
   inside the region **do not escape**. Note that this is orthogonal to the intuition "Python only
   executes the True branch, so at gear 0 these names are never evaluated at all": the error does not
   come from whether the branch is executed, it comes from **the reference point lying outside the
   region**.
2. **The condition is clearly a compile-time switch** (`_DEPTH >= 1`, `X is None`, a tile-size
   comparison, a `cutlass.Constexpr` parameter), yet a branch or the code of both branches is visible
   in SASS/PTX.
3. **`break` / `continue` / `return` / `raise` are rejected inside a control-flow body** (MLIR SSA
   requires every path to yield consistently), and the error message itself suggests switching to
   `const_expr` — when the condition is static, that is the official way out.

The criterion (the only one): **a condition made only of Python constants / module globals /
`Constexpr` parameters / compile-time shapes → must be wrapped in `const_expr`; a condition containing
runtime values such as `tidx`/`bidx`/tensor reads → must not be wrapped** (wrapping is rejected: feeding a
dynamic value into `const_expr` is illegal).

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute

_GEAR = 1          # compile-time gear switch: a module global read at trace time

@cute.kernel
def k(mIn: cute.Tensor, mOut: cute.Tensor, mAux: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    x = mIn[i]

    # (a) Loop side: compile-time unrolling; bindings in the body **can** escape (it is plain Python unrolling).
    for u in cutlass.range_constexpr(2):
        z = x * cutlass.Float32(float(u))
    y = z                                   # legal

    # (b) Branch side: the condition is a compile-time constant -> must be wrapped in const_expr.
    #     Both call-sites use **the same** condition; only then can gear-specific quantities survive across sites.
    for t in cutlass.range(4):              # runtime IR loop
        if cutlass.const_expr(_GEAR >= 1):
            w = x * cutlass.Float32(4.0)    # gear-specific temporary, first bound here
            mAux[i] = cutlass.Float32(1.0)  # gear-specific side effect
        mOut[i] = mOut[i] + x
        if cutlass.const_expr(_GEAR >= 1):
            mOut[i] = mOut[i] + w           # harvested across call-sites: visible only with the same condition

    # (c) if/else binding once in each branch, used outside: must also be wrapped, otherwise `b` cannot leave the region.
    if cutlass.const_expr(_GEAR >= 1):
        b = x * cutlass.Float32(2.0)
    else:
        b = x - cutlass.Float32(3.0)
    mOut[i] = mOut[i] + b + y

    # (d) A true runtime condition: do not wrap. Wrapping is rejected.
    if tidx < 32:
        mOut[i] = mOut[i] * cutlass.Float32(0.5)
```

Remove the `cutlass.const_expr(...)` of (b) / (c), change nothing else, and `cute.compile` dies at trace
time with `NameError: name 'w' is not defined` / `name 'b' is not defined` — this is exactly the
single-variable control the certification probe runs (see "Sources").

**A name used across branches has only three legal ways out**:
1. The condition is a compile-time quantity → wrap it in `const_expr`, folding the whole thing into
   ordinary Python (preferred);
2. The condition is a runtime quantity → bind the name unconditionally once **before** the region (give
   it a neutral initial value), and inside the region only **re-assign** it, never bind it for the
   first time;
3. The name carries data → put it in a fragment / smem allocated before the region; inside the region
   only write contents, don't create new names.

## Gotchas

1. **Loops and branches are asymmetric — this is the easiest point to be fooled by.**
   Bindings inside a `range_constexpr` body escape (plain Python unrolling); bindings inside an `if`
   branch do not (region scope). The two constructs behave differently in the same kernel body; "my
   `range_constexpr` here has always been fine, so `if` should be the same" is the wrong model.
2. **Both call-sites of a gear switch must use the same condition.** If the binding site is wrapped and
   the harvest site is not (or vice versa), the two sites live in different scopes — at best a
   `NameError`, at worst harvesting a quantity from the other gear. In terms of spelling: for
   gear-specific objects (a dedicated pipeline state, a dedicated temporary plane), **the creating site
   and the using site must either sit in the same `const_expr` region or each use a literally identical
   `const_expr` condition**.
3. **`const_expr` only accepts things evaluable at trace time.** Wrapping a runtime value is illegal
   (`if cutlass.const_expr(dyn == 10)` is rejected); conversely, `for i in range(n)` is still lowered
   to an IR loop even if `n` is a Python int — "Python constant" does not mean "compile-time control
   flow"; the annotation does.
4. **Module-global conditions are read at trace time.** When a global is used as a gear switch, the
   assignment must happen **before the first `cute.compile`**; changing the global in the same process
   and compiling the same entry again may get the previous gear's cubin straight back from the JIT
   cache. Practice on the probe/driver side: one subprocess per gear, and isolate `CUTE_DSL_CACHE_DIR`
   per gear.
5. **`const_expr` is not `cutlass.Constexpr`.** The latter is a parameter **annotation** (declaring the
   parameter a compile-time quantity, see [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 3);
   the former is a statement-level **condition annotation**. Typical combination: a `Constexpr`
   parameter comes in, and `const_expr` tests it.
6. **Inside a runtime region a variable's type must not change**, and `break`/`continue`/`return`/
   `raise` are not allowed. When you hit either of these, first ask whether the condition is static —
   if it is, `const_expr` solves it in one step, with no need to change the algorithm.
7. **A folded-away branch lands nothing.** The branch of `const_expr(False)` produces no code, and side
   effects inside it (writes to an auxiliary plane, counter increments) do not happen — this has been
   GPU-certified as "at gear 0 the auxiliary plane stays 0 element by element". Conversely, the branch of
   a bare `if` is a region function, and where its side effects land is decided by the lowered region;
   don't infer it from "Python won't take this branch".

## Sources

- Official docs (mechanism and limitation clauses, the run-time/compile-time comparison table, the
  `break/continue/return` limitation, "Values originating in control flow body are not available
  outside the control flow"):
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html>
- Official changelog (since 4.1.0 `if`/`while` are lowered to runtime by default and need an explicit
  `const_expr`; `dynamic_expr` deprecated):
  <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html>
- Public infra API symbols: `cutlass.const_expr`, `cutlass.range_constexpr`, `cutlass.range`,
  `cutlass.Constexpr`, `cute.compile`.
- GPU-certified probe (not included in this release) — **`PROBE PASS max_diff=0.0`,
  all six contracts C1–C6 pass in the same round**. Two shapes (K1: a single-branch `if` inside a
  dynamic `cutlass.range`, first binding `w` inside the branch, harvested by **another** region; K2: a
  kernel-top-level `if/else`, each branch binding `b`, read after the region) each have two copies,
  `ce`/`plain`, and **the parent side reads its own source back and asserts that the two copies differ
  only by the single wrapper `cutlass.const_expr(`**, so the verdict difference has exactly one
  independent variable:
  C1 the `const_expr` version at gear 0 is bit-equal to numpy **and the folded-away branch's
  auxiliary-plane write never landed** (the maximum of the two deviations is reported, so a folded
  branch that still emitted code could not hide); C2 at gear 1 likewise bit-equal and `ref_sep > 0`
  (the two gears' reference values really differ, so C1/C2 are not the same assertion stated twice),
  i.e. harvesting `w` across regions and `z` escaping from the `range_constexpr` body both hold; C3 K1
  with the wrapper removed fails at trace time with `NameError: 'w'`; C4/C5 K2's two planes are each
  bit-equal and the reference values are separable; C6 K2 with the wrapper removed fails at trace time
  with `NameError: 'b'`. C3 is inside a dynamic loop and C6 at kernel top level ⇒ this lowering is not
  a property of the loop context. Each variant runs in its own subprocess + `CUTE_DSL_CACHE_DIR`
  isolated per variant (the gear is a trace-time module global, and a shared cache would hand back the
  previous gear's cubin — empirical evidence for gotcha 4).
