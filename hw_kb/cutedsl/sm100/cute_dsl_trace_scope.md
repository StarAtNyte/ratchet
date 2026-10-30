# cute_dsl_trace_scope (six trace-scope gotchas: the decorator boundary, closure capture, Constexpr transcription, API drift, `range_constexpr` in an undecorated helper, subscript stores through an object attribute)

> **Certification source: the sm90 tree · partially re-checked on sm100** (gotchas 5 and 6 certified on sm100; gotchas 1–4 not re-checked).
>
> One entry per hardware interface. Operator-agnostic; mechanism only. This entry generalises [`cp_async`](../sm90/cp_async.md) gotcha 7b
> from `range_constexpr` to **all runtime control flow**, plus two probe-engineering facts.
> Names of the form `sm90/xxx` are entries of the Hopper tree and **are not in the read set of an sm100 run**: the loop hands
> a run only the one tree `hw_kb/cutedsl/<kb_arch>/` (`loop.toml [loop] kb_arch`). The names are written out so that you
> know they exist; every reading in them was taken on Hopper and **is not a conclusion for this architecture**; if you need
> one, re-earn it here.

## Purpose

A CuTe DSL kernel is **traced Python**: which code the preprocessor rewrites and which values are visible at
trace time is decided by the **decorator boundary**, contrary to Python-semantics intuition. This entry records four
boundary facts about compile-time hard failures / false positives, all verified on the GPU as "one cut, the verdict flips".

## When needed (trigger)

When writing / changing any DSL kernel that has helper functions, runtime branches, or a cross-version migration; and
when writing a certification probe for such a kernel.

## Mechanism and gotchas

1. **Every helper that contains runtime control flow must carry its own `@cute.jit`.** The preprocessor that rewrites
   `if`/`for` into staged control flow **only walks the source of the decorated function itself**; it does not descend
   into the ordinary Python helpers it calls. A comparison against a runtime value in an undecorated helper
   (`if m1 >= s_q:`, `m1` derived from `bidx`) fails at trace time with:

       PHASE_DYNAMIC_TO_STATIC_BOOL: Runtime value where a Python bool is required

   The fix is one cut: add `@cute.jit` to the helper. A decorated helper can be called normally from a traced body --
   the control flow does **not** have to be inlined into the kernel body (this refines cp_async 7b's "the loop cannot
   be moved out": it cannot be moved out into an *undecorated* function; once decorated, it can).
   Trap shape: if such a helper is introduced only by **build-phase** commits (pre-green, no whole-machine eval), the
   defect can lie dormant for several rounds -- "the transcribed probe compiles" is not "the shipped file compiles",
   see gotcha 3.
   **Corollary (removing one reason not to add the decorator)**: adding `@cute.jit` to a helper does **not** cut its
   writes into register fragments **owned by the caller** -- the decorated helper writes into the fragment passed in
   as an argument, and after it returns the caller sees the written values. So "afraid of losing writes across the jit
   boundary" is not a reason to keep a helper undecorated; the two forms have the same semantics on this point, and
   only the former is rewritten correctly by the preprocessor.
   (GPU-certified: on the same path, the whole-machine bijection contract with the decorated helper gives
   `max_diff == 0.0`, and every value on that path is written by the decorated helper into the caller's fragment.)

2. **SCOPE_CLOSURE_CAPTURE: a function called inside a staged `if`/`for` must not closure-capture runtime values.**
   When trace enters a function from staged control flow, runtime values in its closure (tensors, indices)
   are not available -- they must be passed explicitly as arguments. When writing helpers, always passing explicitly is safest.

3. **Constexpr parameters must keep their annotation verbatim when transcribed.** A parameter declared as
   `cutlass.Constexpr` at the kernel entry, if **re-spelled** as an unannotated parameter in a probe/wrapper,
   is bound as dynamic by `cute.compile(wrapper, ...)` -- and a dynamic value cannot be fed into a
   `Constexpr` parameter: a hard compile-time failure. This is a "transcription-induced defect": the shipped entry itself
   has no problem. **A probe always `cute.compile`s the shipped entry as a whole** (together with its real
   constexpr/dynamic argument split); do not transcribe; the verdict of a transcribed version can be either a false
   negative or a false positive for the shipped file.

4. **Measure API drift; do not trust the error message's nearest suggestion.** Example: `cute.make_fragment(shape, dtype)`,
   which allocates a register tensor by shape/layout, has been removed in some DSL versions and reports

       AttributeError: module 'cutlass.cute' has no attribute 'make_fragment'.
       Did you mean: 'make_fragment_like'?

   The nearest suggestion is misleading -- `make_fragment_like` means "modelled on an existing tensor"; the new name
   for allocation by shape is **`cute.make_rmem_tensor`** (after the rename the same probe flips from FAIL to PASS).
   `cute.make_fragment` in this KB's older recipe examples reads as `cute.make_rmem_tensor` on such versions.
   **How to enumerate all drift in one pass (api-audit)**: walk every dotted DSL name that appears in the artifact and
   try to resolve it, reporting every unresolvable name in one round -- but **strip comments and strings** first before
   scanning (comments recording old names create false positives), and the audit is a text lint, so it may only use the
   `note:` diagnostic channel and **must not occupy a PROBE verdict line** (a check that cannot fail on the artifact
   must not be able to fail the verdict -- name resolution on the traced path is already covered by the compile contract).

5. **In an undecorated helper: `const_expr` works, `range_constexpr` does not.** An ordinary Python helper called from
   the kernel body (without `@cute.jit` / `@cute.kernel`) is not rewritten by the preprocessor. `cutlass.const_expr`
   is a real function; it folds trace-time Python values (ints / bools in the closure) as usual;
   `cutlass.range_constexpr`, however, is a **preprocessor marker**; when the preprocessor has not consumed it, it is
   itself a trace-time exception:

       DSLRuntimeError: ... range_constexpr should be preprocessed by preprocessor.

   So in the same helper, every `const_expr` gate before the `range_constexpr` works, and only the rung that reaches
   that loop dies -- it looks like "the construct this rung adds is broken", but it is the loop spelling. Fix:
   use a plain `range` under a static upper bound (it unrolls the same as `range_constexpr`), or add `@cute.jit` to the
   helper (gotcha 1). It costs one GPU round.
6. **Inside a staged region, do not subscript-store a tensor through a Python object's attribute; bind it to a plain local first.**
   `obj.face[(r, c)] = v` written inside a `for` / `if` that lowers to a region fails at trace time with:

       error[TYPE_DYNAMIC_EXPR_UNSUPPORTED]: A value carried through this `for` is a
       plain Python value (Meta value) (a `<your object class>`) that cannot be turned into a
       Runtime value (Staged value)

   The AST transform resolves the subscript store to its **root name** `obj`, sees it referenced inside the staged
   region, and tries to make this Python object, which has no IR value, a loop-carried value. It has nothing to do with
   the loop bound or with what the object holds. Fix: `face = obj.face` before entering any staged `for` / `if`, and
   only subscript locals (or parameters) inside the region.
   The same defect lies dormant just as well in a staged `if` such as `if kh == 0:`; it simply has not been reached yet.

## Sources

GPU certification probes from a run on an fp8 scoring-type operator (the probes are not included in this release):
- Gotchas 1, 3, 4: an end-to-end compile probe (three contracts compile + rail + values;
  after the one `@cute.jit` cut, `PROBE PASS max_diff≈5.3e-2` at fp8 tolerance; the api-audit false positive and the
  strip-comments fix are recorded in that probe's docstring) and a second compile probe
  (the falsification of the R2 transcription trap is on record: the same Constexpr spelling passes on the shipped path and dies on the transcribed path).
- Gotcha 2: a sub-ring probe (the explicit-argument form passes on the GPU); see also
  `sm90/pipeline_runtime_subring`.
- The corollary of gotcha 1 (writes still reach the caller's fragment after decoration): GPU-certified by an epilogue probe --
  `PROBE PASS max_diff=0.0  (bijection ... C-frag -> ... -> gmem)`;
  every write on that path happens inside a helper decorated with `@cute.jit`, into a fragment held by the caller.
- How to organise the probe side to get several facts back in one round (the discipline of this entry's gotchas 3/4, generalised):
  [`gpu_probe_ladder`](gpu_probe_ladder.md).
- Gotchas 5, 6: a QK probe (after 6 rounds `PROBE PASS max_diff=0.0` × 3). Round 1 died with
  gotcha 6's error text (after rebinding to a local variable the next round died some other way, i.e. that defect was gone); round 4 died with gotcha 5's
  error text (in the same helper the `const_expr` gates had worked in every earlier rung; after switching to a plain `range` the next round reached
  `cute.copy`). Both are on record in per-version docstrings / inline comments.
