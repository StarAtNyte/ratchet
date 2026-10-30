# cute_dsl_trace_scope (four trace-scope pitfalls: decorator boundary, closure capture, Constexpr re-spelling, API drift)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry generalises
> [`cp_async`](cp_async.md) gotcha 7b from `range_constexpr` to **all runtime control flow**, plus two
> probe-engineering facts.

## Purpose

A CuTe DSL kernel is **traced Python**: which code the preprocessor rewrites and which values are
visible at trace time is decided by the **decorator boundary**, contrary to Python-semantics intuition.
This entry collects four boundary facts about compile-time hard failures / false positives, all
verified on the GPU as "one cut, the verdict flips".

## When needed (trigger)

When writing/changing any DSL kernel that has helper functions, runtime branches, or is migrated
across versions; and when writing certification probes for such kernels.

## Mechanism and gotchas

1. **Every helper that contains runtime control flow must carry its own `@cute.jit`.** The
   preprocessor that rewrites `if`/`for` into staged control flow **only walks the decorated
   function's own source**; it does not descend into plain Python helpers it calls. A comparison on a
   runtime value inside an undecorated helper (`if m1 >= s_q:`, with `m1` derived from `bidx`) fails
   at trace time with:

       PHASE_DYNAMIC_TO_STATIC_BOOL: Runtime value where a Python bool is required

   The fix is one cut: add `@cute.jit` to the helper. A decorated helper can be called normally from a
   traced body — the control flow does **not** have to be inlined into the kernel body (this refines
   cp_async 7b's "the loop cannot be moved out": it cannot be moved into an *undecorated* function;
   once decorated, it can).
   Trap shape: if such a helper is introduced only by **build-phase** commits (pre-green, no
   whole-machine eval), the defect can lie dormant for several rounds — "the transcribed probe
   compiles" does not mean "the shipped file compiles", see gotcha 3.
   **Corollary (removes one reason not to add the decorator)**: adding `@cute.jit` to a helper does
   **not** cut off its writes into register fragments **owned by the caller** — a decorated helper
   writes into a fragment passed in as an argument, and after it returns the caller sees the written
   values. So "fear of losing writes across the jit boundary" is not a reason to keep a helper
   undecorated; the two forms have the same semantics on this point, and only the former is rewritten
   correctly by the preprocessor.
   (GPU-certified: on the same path, after decorating the helper the whole-machine bijection contract
   gives `max_diff == 0.0`, and every value on that path is written into the caller's fragment by
   the decorated helper.)

2. **SCOPE_CLOSURE_CAPTURE: a function called inside a staged `if`/`for` must not closure-capture
   runtime values.** When a function is entered from staged control flow at trace time, runtime
   values in its closure (tensors, indices) are unavailable — they must be passed explicitly as
   arguments. Always passing arguments explicitly when writing helpers is the safest.

3. **Constexpr parameters must keep their annotation verbatim when re-spelled.** If a kernel entry
   parameter declared `cutlass.Constexpr` is **re-spelled** in a probe/wrapper as an unannotated
   parameter, `cute.compile(wrapper, ...)` binds it as dynamic — and a dynamic value cannot be fed into
   a `Constexpr` parameter, a hard compile-time failure. This is "a defect produced by the
   re-spelling": the shipped entry itself has no problem. **A probe always `cute.compile`s the shipped
   entry as a whole** (with its real constexpr/dynamic argument split); do not re-spell it — a verdict
   from a re-spelled version can be a false negative or a false positive for the shipped file.

4. **Measure API drift; don't trust the nearest suggestion in the error message.** Example:
   `cute.make_fragment(shape, dtype)`, which allocates a register tensor by shape/layout, has been
   removed in some DSL versions, reporting

       AttributeError: module 'cutlass.cute' has no attribute 'make_fragment'.
       Did you mean: 'make_fragment_like'?

   The nearest suggestion is misleading — `make_fragment_like` means "imitate an existing tensor"; the
   new name for allocation by shape is **`cute.make_rmem_tensor`** (after the rename the same probe
   flips from FAIL to PASS). `cute.make_fragment` in earlier recipe examples of this KB reads as
   `cute.make_rmem_tensor` on those versions.
   **How to enumerate all drift in one go (api-audit)**: walk every dotted DSL name that appears in
   the artifact and try to resolve it, reporting every unresolvable name in one round — but first
   **strip comments and strings** before scanning (comments recording old names create false
   positives), and since the audit is a text lint it may only use the `note:` diagnostic channel and
   **must not occupy the PROBE verdict line** (a check that cannot fail on the artifact must not be
   able to fail the verdict — name resolution on the traced path is already covered by the compile
   contract).

## Sources

GPU-certified probes from an fp8-scoring operator (the probes are not included in this release):
- Gotchas 1, 3, 4: a compile end-to-end probe (three contracts: compile + rail + values; after
  the `@cute.jit` cut `PROBE PASS max_diff≈5.3e-2` at fp8 tolerance; the api-audit false positive and
  the strip-comments fix are recorded in that probe's docstring) and a second compile probe
  (the falsification of the R2 re-spelling trap is on record: the same Constexpr spelling passes on
  the shipped path and dies on the re-spelled path).
- Gotcha 2: a subring probe (the explicit-argument form GPU PASS), see also
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md).
- Corollary of gotcha 1 (after decoration it can still write into the caller's fragment): GPU-certified
  by an epilogue probe —
  `PROBE PASS max_diff=0.0  (bijection ... C-frag -> ... -> gmem)`;
  every write on that path happens inside a `@cute.jit`-decorated helper, into a fragment held by the
  caller.
- How to organise the probe side to get several facts back in one round (generalising the discipline
  of gotchas 3/4 of this entry): [`gpu_probe_ladder`](gpu_probe_ladder.md).
