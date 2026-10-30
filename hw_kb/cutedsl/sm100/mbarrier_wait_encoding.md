# mbarrier_wait_encoding (two encodings of an mbarrier wait: suspending wait <-> non-blocking try/test + explicit spin)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Certification source: sm90; the phase-parity item has been reproduced on sm100;
> sm100 also has one whole-tree comparison of the two encodings (gotcha 12); everything else is not re-checked.**
> There was once a record claiming that "on sm100, `mbarrier_wait` and `test_wait` use opposite phase conventions"; if that
> held, the symptom of the mismatch would be a silent deadlock (process sleeping + GPU at 100%, no error, no line number). **It has
> been disproved by measurement on Blackwell**: same barrier, same phase, the blocking and non-blocking forms agree -- waiting on
> phase 1 returns immediately, waiting on phase 0 hangs, both encodings give the same answer. That item of this entry applies as is.
> The remaining items still only have Hopper readings. Section (f) (how the dossier reads the wait-site addressing, and the
> retry-loop layout) has its own sm100 certification.
> Section (g) (one-shot peek: the difference between `try_wait` and `test_wait`, and the inline-PTX stand-in when `test_wait` is
> absent) and gotcha 14 (timestamps cannot measure a `try_wait` suspension) have their own sm100 certification;
> gotcha 14 additionally cites readings from two other sm100 certifications (see that gotcha).
>
> This entry covers only **how the wait is encoded**, not how the pipeline is built. Setting up the ring / arrive count / phase
> is in `wgmma_tma_mbarrier` and `pipeline_runtime_subring` (sm90 tree, not in an sm100 run's read set; readings from Hopper);
> "should this be an mbarrier rather than an all-hands rendezvous" is in [`handoff_sync_choice`](../sm90/handoff_sync_choice.md).
> The three are orthogonal: with the right primitive and the right pipeline, the wait encoding can still be the wrong setting.

## Purpose

For the same mbarrier and the same phase, CuTe DSL offers **two wait encodings**. Their synchronization semantics are identical
(both are acquire, both honour the same phase parity -- **the phase-parity item has been reproduced on sm100 (see the header); the
other items only have Hopper readings**); the only difference is **what this warp does while the wait is not yet satisfied**:

- **Suspending**: the hardware puts the warp to sleep and wakes it when the barrier flips;
- **Non-blocking + explicit spin**: returns a `Boolean`; the warp stays resident and retries on its own.

When the wait sits on the critical path, the wake-up latency of "go to sleep, then get woken" is added to the total time **once
per wait**, and it is invisible in every throughput metric (tensor / dram / bytes) -- it only shows up in the stall ledger as the
sleep family.

## When needed (trigger)

- **The sleep family has a significant share of the stall ledger**, and the lines hit by SASS `NANOSLEEP.SYNCS`, traced back,
  are exactly the `producer_acquire` / `consumer_wait` call sites (the stall reason on those lines is usually dominated by
  `long_scoreboard` -- the warp hangs on an asynchronous completion).
- On the structural side, all of the following hold: the mbarrier pipeline **is already built** (not a rendezvous form), the
  wait-target relationship is right (it waits for the beat it should wait for), and the tensor metrics match the structure --
  i.e. **nothing other than the wait site itself explains it**.
  This is how this entry is told apart from two other look-alike symptoms:
  - Supply issue badly organized -> `stall_long_sb` high **and** tensor low **and** loads interleaved one by one with address
    arithmetic (see [`gather_index_prefetch`](../sm90/gather_index_prefetch.md));
  - Wrong primitive -> dense `BAR.SYNC` while `PHASECHK`/`DEPBAR` are absent
    (see [`handoff_sync_choice`](../sm90/handoff_sync_choice.md)).
- **Anti-trigger (read before acting)**: when waits rarely actually happen (the producer is far enough ahead, stages are deep
  enough), the two encodings show no measurable difference; and spinning takes issue slots plus one smem barrier read per retry,
  so with many concurrent warps on the SM it can be **slower**. This is a lever that must be A/B'd within the same job, not a
  default.

## Mechanism: two API surfaces over the same barrier

| | Suspending (default) | Non-blocking + explicit spin |
|---|---|---|
| pipeline entry | `pipeline.consumer_wait(state)`<br>`pipeline.producer_acquire(state)` | `pipeline.consumer_try_wait(state)`<br>`pipeline.producer_try_acquire(state)` |
| sync-object entry | `MbarrierArray.wait(index, phase)` | `MbarrierArray.try_wait(index, phase)` |
| arch entry | `cute.arch.mbarrier_wait(bar, phase)` | `cute.arch.mbarrier_try_wait(bar, phase)`<br>`cute.arch.mbarrier_test_wait(bar, phase)` |
| Return value | `None` (loops internally until satisfied) | `Boolean` token (the result of **one** attempt) |
| Lowering | Goes through the **try-wait** family (`mbarrier.try_wait*`) with a hard-coded suspendTimeHint; the retry loop is **generated by the compiler** | Goes through `mbarrier.try_wait*` / `mbarrier.test_wait*`, **no hint**; the loop is **written by you** |
| When not satisfied | Suspends the warp (`NANOSLEEP` appears in SASS) | Returns `false`; the warp stays resident |

`cute.arch.mbarrier_wait` hard-codes suspendTimeHint to `10_000_000` (10 ms). That is an **upper bound on the suspension**, not
the wait time -- the hardware wakes the warp when the barrier flips. The crux is not the number but the fact that this path
encodes "waiting" as **sleep-wake**: every wait pays one wake-up latency. The non-blocking surface has no such segment.

Whether the hint shows up as the 4th operand in the final PTX or is absorbed by the backend into the retry loop it generates is an
NVVM implementation choice; **do not use the operand count as the criterion**. There are two stable, dependable discriminators:
at the **PTX level** the two paths land on **mutually exclusive instruction families** (the blocking surface emits only
`mbarrier.try_wait*`, `mbarrier_test_wait` emits only `mbarrier.test_wait*`); at the **SASS level** only the blocking surface
leaves a `NANOSLEEP` suspension footprint. The certification probe prints the measured census of both levels item by item; read
it from there.

The difference between `mbarrier_test_wait` and `mbarrier_try_wait` is **whether the hardware may briefly suspend**:
`test_wait` is a pure query (never suspends), `try_wait` lets the implementation suspend for an implementation-defined short time.
Both return a `Boolean`, both carry acquire semantics, and both require you to write the retry loop yourself.

## Minimal runnable example

Three sections, ordered from least to most invasive. `pipe` is any `cutlass.pipeline.PipelineAsync` subclass
(including `PipelineTmaAsync`); `cstate` / `pstate` are states built with `make_pipeline_state`.

### (a) Zero-change probe-first: the token form

Natively supported by the DSL: first make **one** non-blocking attempt, and fall back to the suspending wait only if it fails.
The common already-ready case pays no suspension at all. Smallest change; try this first.

```python
tok = pipe.consumer_try_wait(cstate)
pipe.consumer_wait(cstate, tok)     # token true -> the internal if_generate skips the blocking wait
```

### (b) Consumer-side tight spin (replace the suspension entirely)

`consumer_wait` does nothing but wait, so once the spin is satisfied it can be **dropped entirely**.

```python
tok = pipe.consumer_try_wait(cstate)
while tok == 0:                      # dynamic Boolean -> scf.while; see gotcha 5
    tok = pipe.consumer_try_wait(cstate)
# here the full barrier is ready; consume stage `cstate.index` directly
...
pipe.consumer_release(cstate)
cstate.advance()
```

### (c) Producer-side tight spin (the token **must** be handed back)

`producer_acquire` does more than wait: for `PipelineTmaAsync` it is also responsible for the `arrive_and_expect_tx` on the full
barrier. So after spinning, the producer side cannot simply skip it -- hand the satisfied token back; the wait is skipped and the
rest still happens.

```python
tok = pipe.producer_try_acquire(pstate)
while tok == 0:
    tok = pipe.producer_try_acquire(pstate)
pipe.producer_acquire(pstate, tok)   # wait skipped; the expect_tx arrive still happens
# issue this stage's TMA / write this stage's smem
...
pipe.producer_commit(pstate)
pstate.advance()
```

### (d) The barest layer: spin directly on the barrier owned by the pipeline

When you need pure-query semantics (not even a short suspension), take the barrier pointer from the pipeline and issue
`mbarrier_test_wait` yourself. **Which pointer to take: see gotcha 1.** The DSL build used by the sm100 certification job does not
export `cute.arch.mbarrier_test_wait` (calling it directly raises `AttributeError` at trace time); `getattr` first, and when it is
absent use the inline-PTX stand-in from (g).

```python
# consumer waits on the full barrier
fbar = pipe.producer_get_barrier(cstate)          # index taken from your own state
ok = cute.arch.mbarrier_test_wait(fbar, cstate.phase)
while ok == 0:
    ok = cute.arch.mbarrier_test_wait(fbar, cstate.phase)

# producer waits on the empty barrier
ebar = pipe.consumer_get_barrier(pstate)
ok = cute.arch.mbarrier_test_wait(ebar, pstate.phase)
while ok == 0:
    ok = cute.arch.mbarrier_test_wait(ebar, pstate.phase)
```

### (e) Hand-allocated barriers: a `try_wait` spin in a `@cute.jit` helper (drop-in replacement for `cute.arch.mbarrier_wait`)

When the barrier is not a pipeline object but a raw pointer initialized by hand with `mbarrier_init` in `SharedStorage`, there is
no `consumer_try_wait` layer; replace `cute.arch.mbarrier_wait(bar, phase)` directly with a helper that carries its own
`@cute.jit` (gotcha 5), call site by call site, arguments unchanged:

```python
@cute.jit
def bar_wait_spin(bar, phase):
    ok = cute.arch.mbarrier_try_wait(bar, phase)   # try family: hardware may briefly suspend, no hint
    while ok == 0:
        ok = cute.arch.mbarrier_try_wait(bar, phase)

# call site: cute.arch.mbarrier_wait(bar_full, par)  ->  bar_wait_spin(bar_full, par)
```

Works on both sides (producer waiting on empty, consumer waiting on full): a hand-made barrier has no expect_tx hidden inside
`producer_acquire` (gotcha 2), so the producer side can be replaced wholesale too; arrive / init counts stay untouched (gotcha 8).

**(e') Inline-PTX stand-in when the arch surface is absent** (same instruction, same semantics; when the build does not export
`cute.arch.mbarrier_try_wait`, probe once with `getattr` and then choose):

```python
@dsl_user_op
def mbar_try_wait_i32(mbar_addr, phase, *, loc=None, ip=None) -> cutlass.Int32:
    # One parity attempt -> 1 satisfied / 0 not satisfied. The operand is an Int32 shared-window address
    # (Pointer.toint()), not a generic 64-bit pointer (the DSL refuses the conversion).
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
        "selp.b32 $0, 1, 0, p;\n}",
        "=r,r,r,~{memory}",          # ~{memory} + has_side_effects: see below
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

_arch_try_wait = getattr(cute.arch, "mbarrier_try_wait", None)
if _arch_try_wait is not None:
    @cute.jit
    def bar_wait_spin(bar, phase):
        ok = _arch_try_wait(bar, phase)
        while ok == 0:
            ok = _arch_try_wait(bar, phase)
else:
    @cute.jit
    def bar_wait_spin(bar, phase):
        addr = bar.toint()
        ok = mbar_try_wait_i32(addr, phase)
        while ok == 0:
            ok = mbar_try_wait_i32(addr, phase)
```

You need **both** `has_side_effects=True` **and** the `~{memory}` clobber: this asm has a return value, and relying on "it has a
result so it will not be DCE'd" is not enough -- **[INFERRED]** a pure-function-shaped asm without the clobber can be hoisted out
of the `while` body, after which every trip compares the same stale register, and the symptom is **spinning forever** (a hang,
not a wrong result). The certified tree passed gotcha 10's repeated-launch contract with both in place; the control variant
without the clobber was never run, so before citing "removing it hangs", launch a probe first.

### (f) **Addressing** of the wait site and **retry-loop layout** (two more axes, orthogonal to suspend / spin)

The first five sections only answer "what does the warp do while the wait is unsatisfied". The same wait has two further,
independent aspects, each with its own footprint in SASS and in the dossier. This section covers how the dossier reads them and
their SASS layout; **it is not a performance lever**.

- **Where the address comes from (dossier reading).** For a hand-made barrier table where "each warp waits on its own one", the
  most convenient spelling is `cute.arch.mbarrier_wait(table + warp, ph)`. It has **one** call site, the address is runtime
  pointer arithmetic, SASS carries only one immediate offset (the table's first entry), and the warp's share sits in a register.
  The static census reads such an address **unstably**: it may read it as an extra THREAD-tier object (on top of the NW CONST
  objects at the init / arrive end, `mbar_objs` = NW + 1, with no edge to the CONST objects), or the def-chain may fold it onto
  the CONST object of the table's first entry (object count unchanged, but the identity is wrong). Neither reading tells you
  which barrier it waits on. This is purely a census-reading issue, not extra barriers; do not change the wait addressing just
  to make the dossier axis look nice.

- **Where the retry loop goes (ptxas decides; the source is not a lever).** In PTX, `cute.arch.mbarrier_wait` is just an
  `mbarrier.try_wait.parity` loop with a local label (hint `10000000`); hand-writing inline PTX of the same shape gives the same
  PTX. The SASS layout of the retry loop is decided by ptxas per wait site, and both forms appear in the same kernel:
  - **out-of-line**: a fast path `SYNCS.PHASECHK.TRANS64.TRYWAIT P, [..]` in straight-line code, immediately followed by
    `@!P BRA` jumping forward to a retry block **after the last `EXIT`**; the retry block holds `TRYWAIT` + `PHASECHK`
    + a back edge, using the same immediate offset;
  - **in place**: the wait site itself is a small spin loop left in the middle of the straight-line code.
  Among a set of wait sites in the certification probe (one call site per warp), of the wait instructions in straight-line code
  only 1 went out-of-line and 12 were in-place spins; the SASS census of the DSL blocking surface and of the equivalent inline
  label loop was identical.
  Which layout each spelling (DSL try_wait retry loop / blocking surface / hand-written fast path) lands on is in
  [`mbarrier_try_wait_fastpath_layout`](mbarrier_try_wait_fastpath_layout.md) (no whole-kernel gain has ever materialized).

  So if you want to "move the retry loop into a separate out-of-line region", there is no deterministic switch at the CuTe DSL /
  inline PTX level (the spelling influences ptxas's choice, see the entry linked above, but it is still decided per wait site and
  must be read in SASS site by site).
  One thing to know for the census: in the retry block after EXIT, the def-chain of the address register is traced backwards in
  textual order and often hits an unrelated definition, ending UNRESOLVED; some wait sites in straight-line code do the same (the
  base register is reused for another address in a sibling branch, e.g. the `IMAD.WIDE` of a gmem pointer, which the backward
  trace hits first). These sites are excluded by the axes that count only resolved objects: `mbar_objs` /
  `mbar_tiers` / `mbar_per_role` / `edges`. So "the dossier has no spin role / is missing an edge to the spin" may just mean the
  retry block was not resolved; **it does not mean the source lacks a retry loop**.

**sm100 certification readings** (`PROBE PASS max_diff=0.0`; NW warps x NW barriers, each warp waits on its own,
warp 0 arrives after loading from gmem, so the first attempt necessarily fails and the retry path is executed): for the runtime
address `table + warp`, under both spellings (DSL blocking surface and inline label loop) the straight-line wait sites carry only
1 immediate (the table's first entry); this time it was read as "def-chain folded onto the CONST of the table's first entry",
`mbar_objs` = NW, `mbar_tiers` = {CONST: NW} -- not the THREAD tier; the object count shows nothing wrong, only the number of
immediates in the wait instructions reveals it. The retry block (after EXIT) read as UNRESOLVED, confirming the second bullet.

arrive / init / count / phase all stay untouched (gotcha 8).

### (g) One-shot peek: `try_wait` waits a while for you, only `test_wait` is a pure query

The sections above are all "wait until satisfied". Another usage is to **probe only once**: take a token first, do work that does
not depend on this barrier in between, and right before actually using the data decide from the token whether to block ((a) is
the special case: hand it back immediately after probing):

```
tok = peek(bar, ph)            # probe once
...                            # work that does not depend on the data bar protects
if tok == 0: wait(bar, ph)     # before using the data: token true -> skip; the blocking-wait line always stays
```

There are two candidates for `peek`, with different semantics: `mbarrier.test_wait` returns immediately; `mbarrier.try_wait`
(and `consumer_try_wait` / `producer_try_acquire` / `cute.arch.mbarrier_try_wait`, which land on it) may wait for an
implementation-defined time limit while the phase is incomplete, and returns true if the arrive comes during that time. sm100
certification readings (SM locked at 1500 MHz; per CTA one writer warp + one peeking warp; after the two
rendezvous, the writer delays D cycles, writes the data, then arrives; the peeking warp probes once right after the rendezvous and
blocks if the token is false; 128 peeks per setting):

| Setting | `try_wait` peek returns true | `test_wait` peek returns true |
|---|---|---|
| Phase already complete (peeking warp idles 6000 cycles before probing) | 128/128 | 128/128 |
| D = 1000 / 4000 | 128/128 | 0/128 |
| D = 20000 / 100000 / 1000000 | 0/128 | 0/128 |

In all six settings, both encodings' "peek + token-gated fallback" produced bit-exact data. In other words: on this
implementation a `try_wait`-form peek is a **bounded wait** -- the bound falls between 4000 and 20000 cycles; two other sm100
certifications give the same bound: on a ring, peeking the next beat before this beat's commit while supply is throttled, the
`try_wait` peek costs 7819-7823 cycles per beat and `test_wait` 101 cycles; asking about a phase that never
completes, one `try_wait` takes 5184 ns. Both were measured at SM 1500 MHz; about 7.8k cycles ~= 5.2 µs. Whether
it scales with time or with cycles at other clocks was not measured.

**Which line this wait lands on**: the result predicate of `try_wait` is variable-latency; subsequent instructions unrelated to it
keep issuing, and the wait lands on the **first instruction that uses the predicate** (gotcha 14). In the DSL, the `if
try_wait(...)` / the line converting the result to `Int32` usually follows the peek immediately, so with the usual spelling all the
work after that line waits for it -- the 7819-7823 cycles above were stamped after the `if`. To keep the work after the peek from
being delayed, use `test_wait`.

**Entry point for `test_wait`**: the DSL build used by the sm100 certification job does not export `cute.arch.mbarrier_test_wait`
(`getattr` returns `None`); the whole `test_wait` column of the table above went through the inline-PTX stand-in below (same shape
as (e'), only the instruction changes):

```python
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

@dsl_user_op
def test_wait_i32(mbar_addr, phase, *, loc=None, ip=None) -> cutlass.Int32:
    # One pure query -> 1 complete / 0 incomplete. The operand is an Int32 shared-window address (Pointer.toint())
    return cutlass.Int32(llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.test_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
        "selp.b32 $0, 1, 0, p;\n}",
        "=r,r,r,~{memory}",                  # has_side_effects + ~{memory}: both are needed (see (e'))
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

_arch_test = getattr(cute.arch, "mbarrier_test_wait", None)

if _arch_test is not None:                   # choose once at the Python level, no branch inside the jit body
    @cute.jit
    def peek_test(bar, phase):               # returns an Int32 token: 1 true / 0 false
        r = cutlass.Int32(0)
        if _arch_test(bar, phase):
            r = cutlass.Int32(1)
        return r
else:
    @cute.jit
    def peek_test(bar, phase):
        return test_wait_i32(bar.toint(), phase)

# Usage (hand-made barrier; bar is a pointer in smem initialized with mbarrier_init, ph is the caller's own phase)
tk = peek_test(bar, ph)
# ... work that does not depend on the data bar protects ...
if tk == 0:
    cute.arch.mbarrier_wait(bar, ph)         # token false: block as usual (semantics unchanged)
# only here read the data bar protects
```

Barriers owned by a pipeline can be peeked the same way: take the pointer per gotcha 1 (consumer waiting on full uses
`producer_get_barrier(state)`, producer waiting on empty uses `consumer_get_barrier(state)`), and hand the token back to
`consumer_wait(state, tok)` / `producer_acquire(state, tok)` -- these two calls only look at whether the token is true (the
certification probe covers only the hand-made barrier form).

Key points:
- **A false token from `test_wait` is the normal case.** A pure query always returns false before the arrive (every late setting
  in the table above is 0/128); whether the early peek helps depends on whether the "other work" can cover the remaining wait,
  not on the fraction of true tokens.
- **The token and the wait it is handed back to must be the same `(index, phase)`**; wrong phase -> the token is always true and
  skips a wait that was needed (wrong data), or always false (just slow).
- **`test_wait` inside a `while` is (d)'s pure-query spin**, which is a different thing from "probe once and move on".
- **No whole-kernel gain has ever been read, and the `try_wait` form was slower within the same job every time.** A one-beat early
  peek has been landed on MMA / supply roles in three sm100 trees: the `test_wait` form read within noise, same-job or cross-job;
  on the MMA side the `try_wait` form read slower within the same job every time -- peeking after the in-beat commit 0.9982 (> 1
  means faster); per-beat / seam `try_wait` early peeks placed before the accumulator commit had time ratios 1.0072 / 1.0132, and
  `test_wait` at the same position 0.9982 / 0.9984; in another tree, an unguarded last beat placed before the accumulator commit
  was 8.5% / 8.7% slower with `try` on both sides and level with `test` on both sides (sources of these readings: "When needed" in
  [`umma_consumer_full_peek_before_commit`](umma_consumer_full_peek_before_commit.md)). Default to `test_wait` for one-shot peeks
  (the supply-side participant form has another same-job break-even reading, see
  [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) gotcha P5). On one of those
  trees the `test_wait` early peek landed in the same package as other tile-seam reorderings; the same-job baseline ratio went
  1.0245 -> 1.0039 (cross-job, the median of three confirmations of that tree's best code without the early peek was 1.0088),
  and the whole package was later rolled back. An early peek is a form, not a lever; A/B it within the same job before and after
  landing. Where in the beat to place the MMA-side early peek: see
  [`umma_consumer_full_peek_before_commit`](umma_consumer_full_peek_before_commit.md).

## Gotchas

1. **The barrier accessor's name says "who signals", not "who waits".**
   `producer_get_barrier(state)` gives the **full** barrier (the one the producer arrives on), which is exactly what the
   consumer waits on; `consumer_get_barrier(state)` gives the **empty** barrier, which the producer waits on. Both take
   `index` from the **caller's own** state. Swapping them = deadlock (waiting on a phase that never flips) or early release
   (reading a stage that is not fully written).

2. **On the producer side, `producer_acquire` cannot be deleted outright.**
   For `PipelineTmaAsync`, `producer_acquire` = two things: "wait on empty" + "`arrive_and_expect_tx` on the full barrier".
   Spinning without calling it = the transaction count is never set = the full barrier never fills = deadlock. The correct
   spelling is (c)'s token hand-back; if you really want to do it by hand you must add
   `cute.arch.mbarrier_arrive_and_expect_tx(pipe.producer_get_barrier(pstate), tx)` yourself.
   The consumer side has no such issue (`consumer_wait` only waits).

3. **The token form leaves a dead branch in the binary.**
   The skip in `producer_acquire(state, tok)` is a **runtime** `scf.if`, not resolved at compile time: the blocking wait's code
   is still in SASS, and **the static `NANOSLEEP` count does not drop to zero**. Acceptance must use **dynamic** readings (the
   sleep family's stall share / end-to-end time); using the static count as the criterion leads to the false conclusion "the
   switch did not take". Only dropping `consumer_wait` entirely on the consumer side as in (b) brings the static count to zero.

4. **Use the caller's own `state.phase` for the phase; do not make one up.**
   `make_pipeline_state(PipelineUserType.Producer, stages)` has initial phase **1** (on the first lap, empty counts as already
   empty), Consumer has **0**; `state.advance()` flips the phase only when the index wraps to 0. When hand-writing a spin, just
   copy `state.phase`; maintaining your own phase is the main source of off-by-one errors, and a wrong phase manifests as a
   deadlock or reading a stale stage, not as an error.

5. **`while tok == 0:` must sit in a scope the preprocessor can enter.**
   That is, inside the body of a `@cute.kernel` function, or inside a helper that **carries its own `@cute.jit`**. Written in an
   undecorated plain helper, the dynamic `Boolean` is evaluated as a Python bool at trace time ->
   `PHASE_DYNAMIC_TO_STATIC_BOOL`. See [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md).

6. **Spinning is not free; it must be A/B'd within the same job.**
   Spinning takes issue slots plus one smem barrier read per retry. When waits are sparse (deep stages, producer far ahead) or
   there are many concurrent warps on the SM, the execution resources the suspending form gives up are worth more than the
   saved wake-up latency. This entry is a **lever**, not a **default**: switching encodings requires a same-job before/after
   comparison; you cannot pick the setting from the symptom alone.

7. **`mbarrier_conditional_try_wait(cond, bar, phase)` is not a spin primitive.**
   It means "attempt once only if `cond` is true, otherwise return `True` directly", used to put one attempt under a predicate;
   using it as a loop body "succeeds" and releases immediately when `cond` is false. To spin, use `mbarrier_try_wait` /
   `mbarrier_test_wait` with an explicit `while`.

8. **Do not change the arrive side while you are at it.** This entry touches only the **wait** side. Changing the counts or the
   number of issuing threads of arrive / release / expect_tx means deadlock or silent errors, and is fully orthogonal to the
   wait encoding.

9. **A spin encoding rewrites the dossier's mbarrier census readings; the structure did not change.** For the same set of
   hand-made barriers at fixed addresses with no stage-index arithmetic, switching only the waits from suspending to (e)'s spin,
   the NCU dossier census read:
   `spin_loops` 14 -> 27; mbarrier objects UNRESOLVED 5 -> 0, tier `{CONST: 12}` ->
   `{CONST: 10, RING: 10}`; contract axis `rings` `[]` -> 10 x `stride=0xe000`; the set of inter-role edges was rewritten as
   well. The source has no ring at all (each barrier is a fixed object in `SharedStorage`); the address chain the compiler
   generated for the explicit retry loop was read by static chain-tracing as "initial value + increment stride".
   Reading: when `rings` / `mbar_tiers` flip after a wait-encoding change, check the source first -- does the barrier have
   stage-index arithmetic? If not, record it as codegen drift: neither a prime suspect nor evidence that "a ring was built".
   Which axes flip is **a property of the tree**, not of the encoding: on a tree whose slot indices are compile-time constants
   and whose barrier addresses are fixed offsets, the same switch to (e) left `rings` / `mbar_tiers` untouched; what flipped
   were `loops` and the per-region counts of `commit` / `supply` -- the hand-written `while` loops were the first back edges in
   that tree ([`ptx_sass_census_criteria`](../sm90/ptx_sass_census_criteria.md) gotcha 11). The two cases have only one
   thing in common: what a wait-encoding change leaves in the dossier is **back edges**, not new synchronization objects.

10. **The correctness contract for switching the wait encoding is not one launch; it is "repeated launches on the same input +
    bitwise comparison".**
    Spins written wrong fail in three ways -- wrong phase, the attempt hoisted out of the loop, and a lost acquire letting the
    consumer read a half-written slot -- the first two hang, the third is a **data race**: a single launch may pass by luck.
    Write the contract as the same input launched N times in one process (N=12 is enough), each passing the judge, with
    **outputs that are deterministic by construction bitwise identical** (quantities accumulated in a fixed order within each
    CTA; outputs of unordered cross-CTA reductions are not bound by this, only report drift). The same-job control variant uses
    the suspending form; when the control and experimental variants both die, suspect the probe's copy scaffolding before the
    encoding ([`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 4).


11. **Readings of the `stride=0x1` family on the `rings` axis (bare `stride=0x1` / `slots=253` / `slots=9` /
    `wrap=sel`) are codegen fingerprints of the phase / index arithmetic chain; they do not encode ring depth.** The census
    reads an AND mask on the barrier address chain as `slots = (mask + 1) / stride`, and a SEL as `wrap=sel`
    (definition in the tool `orch_tools.resolve_addr`): on a stride-1 chain `slots=253` is just the mask 0xfc, and
    `slots=9` is the mask 0x8 -- neither is the depth of any ring. Readings from four KEEPs on the same tree:
    - A whole-segment rewrite of a middle role (zero mbarrier source changes, all ring depths unchanged):
      `[slots=9, wrap=sel]` -> `[slots=253, slots=9, wrap=sel]`;
    - Two wgmma chains merged into one commit group (rings unchanged): the `slots=253` one -> bare `stride=0x1`;
    - **Actually changing depth** (two rings 2 -> 3, two rings 5 -> 3): bare -> `slots=253` -- **the same** reading as the first
      item; no `slots=3` appeared;
    - Hand-off exit segment deleted + one chain switched to fp16 operands (ring depths unchanged, one empty count 128 -> 384):
      `[bare, slots=9, wrap=sel]` -> `[bare, slots=253, slots=9, wrap=sel]`.
    Seven later KEEPs on the same tree pin down "the flips have nothing to do with wait sites" even harder (each reading comes
    from that KEEP's own eval job; job ids in "Sources"):
    - One SS wgmma chain, together with its element-wise drain, moved from role A to role B (A loses two mbarrier wait sites,
      B gains one; ring depths and arrival counts unchanged): `[bare, slots=253, slots=9, wrap=sel]` -> `[bare, slots=9]`;
      `slots=253` and `wrap=sel` disappear **together**; the same move with the new fragment borrowing a dead fragment's
      registers (spill 7,022,592 -> 0, instruction order unchanged) reads **unchanged**; moving it back brings the four entries
      back as they were.
    - **Pure register-side** rewrites inside one role, zero mbarrier source changes: narrowing the drain to delete the f32
      staging fragment -> the four entries shrink to `[slots=9]`; then keeping 32 f32 factors resident across a chain ->
      `[slots=9, wrap=sel]`; landing the three items as one package reads **the same** as landing "the third item alone" (both
      `[slots=9, wrap=sel]`).
    - **Code AST unchanged, only two `setmaxnreg` setting constants changed** (104/192 -> 120/184):
      `[slots=9, wrap=sel]` -> `[bare, slots=9]`.
    - Adding only one 128-thread named-barrier rendezvous (zero mbarrier changes): the reading is the same as the baseline's four
      entries.
    Reading: record every change in this family's count and members as codegen drift -- it moves even with **register
    settings** -- neither a prime suspect, nor usable to prove "the depth made it into the cubin" or "added/removed wait sites
    made it into the cubin".
    The footprint of a depth change is in the dynamic smem byte count and the barrier object count; how to read them is in
    [`ring_depth_lever`](../sm90/ring_depth_lever.md).
    **Readings from a second tree (sm100 pair UMMA, rings maintained by `PipelineTmaUmma` / `PipelineUmmaAsync` objects,
    `wrap=sel` family)** -- on this tree the family **appears** to encode depth, which is exactly why it must be written down:
    - Operand ring 6 -> 5 -> 4 -> 6, one constant changed each time: `[0x1, 0x2, 0x3, 0x4]` -> `[0x1, 0x2, 0x3]` ->
      `[0x1, 0x2]` -> the four entries come back; the count is exactly NSTAGE - 2 and the strides increase one by one;
    - **Rings unchanged**, only the epilogue staging face switched from swizzle NONE to SW128 (the stmatrix partitioning and the
      TMA store frame rewritten accordingly): three entries -> four (an extra `stride=0x4,wrap=sel`), `mbar_objs` 23 -> 24,
      `edges` `[]` -> two inter-role edges -- not one character of mbarrier source changed.
    Since the same reading is pushed by both depth and non-depth changes, "count = depth - 2" is just a codegen coincidence of
    this tree. Read depth from `smem_dynamic` (202752 / 170496 / 138240) and `mbar_objs` (24 / 21 / 18);
    see [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md).
    **Readings from a third tree (sm100 single CTA, hand-written mbarriers + three `setmaxnreg` roles)**:
    - Landing the `setmaxnreg` four-piece set (one branch per role, each role's item loop living inside its own branch, TMEM
      round trips batched by column), zero mbarrier source changes: `[]` -> `[stride=0x1, stride=0x3f]` -- the address chain
      previously failed to resolve as RING, and only after the branch reordering became "increment inside the loop";
    - **Actually changing depth** (output face 1 -> 2 copies, `FULL` / `EMPTY` each becoming 2, slot index becoming the runtime
      `c % 2`): `stride=0x3f` -> `stride=0x1,slots=536870911` -- `slots` is the reading of the 0x1fffffff mask on the index chain
      (`(mask + 1) / 1`); **no `slots=2` appeared**;
    - **Changing depth for real once more** (hand-written operand ring 2 -> 3, one constant changed, with the barrier-table
      offsets switched to a derived formula in the same package; spelling in the hand-written ring section of
      [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md)):
      `[stride=0x1, stride=0x1,slots=536870911]` -> `[stride=0x1,slots=536870911, stride=0x2]` --
      an extra `stride=0x2`, bare `stride=0x1` gone, **no `slots=3` appeared** (the build chain this KEEP belonged to was later
      abandoned, and the passing code of the same tree still has depth 2 on that ring; the reading serves only as a footprint,
      see the certification boundary in the hand-written ring section of `pipeline_tma_umma_stage_count`);
    - **Pure register-side + smem-face rewrite, zero mbarrier source changes, register settings unchanged** (one role's
      element-wise segment rewritten per the [`branchfree_masked_elementwise_pass`](branchfree_masked_elementwise_pass.md)
      variant: scalar face switched to field-major, element-wise scalar reads switched to vector reads, scalar multiplies
      switched to packed pairs, face reads hoisted before the wait):
      `[stride=0x1, stride=0x1,slots=536870911]` -> `[stride=0x1]` -- the `slots=536870911` of the double-buffered output face
      disappeared, while not one line of the output face's ring code changed.
    Three trees, one conclusion: the family does not encode depth, and when depth really changes it does not report it either;
    a register-side rewrite with zero mbarrier changes can still make an existing reading disappear.
    **Readings from a fourth tree (sm100 pair UMMA, 512 threads in four warpgroups, hand-written mbarrier table +
    `setmaxnreg`, 1 CTA/SM)** -- nine KEEPs on the same tree, `rings` flipped nine times, each reading from that KEEP's own
    eval job:

    | What changed | Ring / barrier source | `rings` before -> after | gate window (ms) |
    |---|---|---|---|
    | One role's trip order reordered, one wait moved down ([`tile_boundary_wait_placement_prep_in_flight`](tile_boundary_wait_placement_prep_in_flight.md)) | unchanged | `[]` -> `[stride=0x1]` | 1.4387 -> 1.3086 |
    | 26 waits switched to explicit spins + index switched to lane-owned (same package) | wait side only | `[stride=0x1]` -> `[]` | 1.2812 -> 1.3169 |
    | **Only the `setmaxnreg` immediates changed** (8 registers moved from one role to another) | unchanged | `[]` -> `[stride=0x1]` | 1.3169 -> 1.3203 |
    | Warp-uniform fast path added to the element-wise segment + one output face 1 -> 2 slots (same package, **a 2-deep ring really added**) | +2 barriers | `[stride=0x1]` -> `[stride=0x1,slots=9]` | 1.2415 -> 1.2474 |
    | **Only the `setmaxnreg` immediates changed** (32 registers change role) | unchanged | `[stride=0x1,slots=9]` -> `[stride=0x1,wrap=sel]` | 1.0564 -> 0.9888 |
    | Static stepping replaced by a cluster launch control queue (**a 4-deep ring really added**, [`cluster_launch_control_work_queue`](cluster_launch_control_work_queue.md)) | +8 barriers | `[stride=0x1,wrap=sel]` -> `[stride=0x1,slots=25, stride=0x1,slots=9, stride=0x30a6f,wrap=mask, stride=0x400100]` | 0.9888 -> 0.997 |
    | `.L2::cache_hint` added to gather4 (one modifier + one operand) | unchanged | `stride=0x400100` -> `stride=0x400100,wrap=sel`, the other three unchanged | 0.997 -> 1.0004 |
    | Supply issue moved from 4 warps of one role to 3 warps of another role | waiter changes, count unchanged | previous row's four -> `[stride=0x-1,slots=-25, stride=0x1,slots=9, stride=0x2,wrap=mask, stride=0x3,wrap=mask, stride=0x30a6f,wrap=mask]` | 0.9993 -> 0.9844 |
    | Element-wise segment spread over two warpgroups, the counts of five barrier groups doubled | counts only | `stride=0x30a6f,wrap=mask` -> `stride=0x3126f,wrap=mask`, the other four unchanged | 0.9844 -> 1.0548 |

    This table adds four things beyond the first three trees:
    - **Wide-stride entries (`stride=0x30a6f` / `0x400100` / `0x3126f`) are drift too.** Their "stride" is a constant on the
      address chain; a patch changing only the cache-policy bits and a patch changing only a barrier count each rewrote one of
      them.
    - **Negative readings are legitimate output.** `stride=0x-1,slots=-25` is the same `stride=0x1,slots=25` read after the
      issuing body changed warps (increment became decrement), not two rings.
    - **Neither of the two really-added rings was read as its own depth.** The 2-deep one reads `slots=9`; for the 4-deep one no
      entry reads `slots=4`.
    - **ms has nothing to do with the flips.** Across the nine, gate ranged from -9.04% to +7.15%; of the two "only `setmaxnreg`
      immediates changed", one was +0.26% and the other -6.40%, and `rings` flipped both times.
    **Readings from a fifth tree (sm100 pair UMMA, 512 threads, hand-written mbarrier table + three `setmaxnreg` settings;
    `wrap=mask` and large-stride families)**:
    - Two consumer WGs alternating by round parity + three count-4 hand-off barriers (mbarrier table 27 -> 30 rows, spelling in
      [`consumer_warpgroup_pingpong_parity`](consumer_warpgroup_pingpong_parity.md)): `[]` ->
      `[stride=0x10200,wrap=mask, stride=0x10200,wrap=mask]`; **the same package** re-landed on another baseline: `[]` ->
      `[stride=0x1]` -- the same source change, two readings;
    - The readback segment split between two WGs by TMEM column half (one warp-uniform scalar shifts both the TMEM window and the
      gmem address; one barrier's arrive count 8 -> 16, no new barrier): the two `0x10200` -> `[stride=0x38ad0]`;
    - The supply role switched from an elected lane to lane-owned (zero mbarrier-table changes, zero wait-site changes, see the
      sm100 section of [`gather_index_lane_owned`](gather_index_lane_owned.md)): `[stride=0x38ad0]` ->
      `[stride=0x1, stride=0x1,wrap=sel, stride=0x2,wrap=sel, stride=0x38ad0]` -- all three new readings come from the index
      chain after the role body left `if lane == 0`;
    - Deleting one 512 B smem face (a statistics face from 5 columns to 4, `smem_dynamic` 232 248 -> 231 736), zero ring and
      barrier changes: `stride=0x38ad0` -> `stride=0x388d0`, the difference is exactly 0x200 = 512 B -- **a single large-stride
      reading is an absolute smem address that got into the address chain**; it moves when the face moves, and is not any ring;
    - Write-out form direct write <-> staging face + TMA store (`UTMASTG.5D` disappears / reappears in `cache_ops`, see
      [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md) gotcha 15): `rings` is `[]` both times,
      unchanged.
    **Readings from a sixth tree (sm100 pair UMMA + gather4, 512 threads in four warpgroups, hand-written mbarrier table + four
    `setmaxnreg` settings, 1 CTA/SM; across the four KEEPs the barrier table, ring depths and wait encoding were all unchanged;
    the only thing touching an mbarrier was the first row moving one S-free arrive to after the second TMEM readback)** -- on
    this tree **only** `rings` and the count axes that follow it moved; the other contract axes (`cache_ops` / `edges` /
    `nbar_*` / `supply`) did not move once in four:

    | What changed (all on the register side of one role) | `rings` before -> after | Count axes that drifted along | gate window (ms) |
    |---|---|---|---|
    | Spill removal in the consumer segment: TMEM readback split by column half ([`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md)) + temporary table deleted + `setmaxnreg` settings redistributed, `spill_ld/st_sectors` 7602176/6181696 -> 0/0 | `[stride=0x1, stride=0x2e24f,wrap=mask]` -> `[stride=0x1]` | `mbar_tiers` RING 2 -> 1, one extra THREAD 1 | 1.2072 -> 1.06555 |
    | Element-wise segment switched to packed arithmetic (FADD2 / FFMA2 / 3-input FMNMX, [`softmax_segment_ilp`](softmax_segment_ilp.md)) + mask switched to bitmap-predicate select | `[stride=0x1]` -> `[stride=0x1, stride=0x1,wrap=sel]` | RING 1 -> 2, THREAD gone | 1.02155 -> 0.9721 |
    | Mask select re-encoded: 48 FSEL -> 24 PRMT sign-replicated keep-words + one LOP3 per pair (wait sites and segment structure unchanged) | previous row's two -> plus `stride=0x2b24f,wrap=mask` | **`mbar_objs` 37 -> 38**, RING 2 -> 3, `mbar_unresolved` 65 -> 61 | 0.9565 -> 0.9809 |
    | Previous row's keep-words deleted, replaced by FSEL inside a warp-uniform branch | three -> the two of two rows up | the four count axes **return item by item** to the readings of two rows up | 0.9809 -> 0.9532 |

    This table adds three things:
    - **`mbar_objs` can also be pushed by a pure register-side rewrite.** The third row's diff contains no mbarrier call at all (no
      new barrier, no wait-encoding change, and not the gotcha-9 form either), yet `mbar_objs` went +1, and -1 when reverted --
      when checking the barrier table with `mbar_objs`, its +/-1 is not evidence; check the table in the source.
    - **Wide-stride `wrap=mask` entries confirmed again to be smem addresses.** The two wide-stride readings `0x2e24f` and
      `0x2b24f` differ by 0x3000 = 12288 B, exactly the difference in `smem_dynamic` between the two points in time
      (189344 -> 177056). On the same address chain, it moves as soon as the face layout moves.
    - **Flips are reversible, and occur with ms moving in the same or the opposite direction.** The third and fourth rows are a
      round trip and the readings come back as they were; across the four, gate ranged from -11.73% to +2.55%.
      The first row's -11.73% came in the same package as the spill elimination and cannot be credited to `rings`.
    Six trees, one conclusion: the family does not encode depth, and when depth really changes it does not report it either; a
    register-side rewrite with zero mbarrier changes can still make an existing reading disappear (or appear); a single reading
    with a stride close to the smem limit is an address, and the `wrap=mask` family moves with the branch shape of the role body.

12. **One whole-tree reading on sm100: explicit spinning lost to suspending, and lost on spills.** The same tree (512 threads /
    16 warps / 1 CTA/SM, 26 dynamic wait sites, 3 of them inside the element-wise chain with 48 live f32 columns crossing the
    wait) switched every wait from `cute.arch.mbarrier_wait` to a tight `mbarrier_try_wait` spin in a `@cute.jit` helper
    (barrier objects, phase expressions, call sites and the arrive side all untouched), one eval before and one after:

    | | Suspending | Explicit spin |
    |---|---|---|
    | gate (device window) | 1.2812 ms | 1.3169 ms (+2.79%, same package as switching the index to lane-owned) |
    | `spill_ld / st_sectors` | 588240 / 9472 | 54262224 / 36454592 |
    | `stall_long_sb` / `stall_barrier` / `stall_mio` | 4.34 / 1.71 / 0.03 | 6.46 / 2.08 / 0.07 |
    | `tensor_pct` | 42.91 | 41.84 |
    | dossier `mbar_objs` / `mbar_unresolved` | 48 / 25 | 4 / 95 |
    | dossier `loops.copy-producer` / `spin_loops` / `edges` | 28 / 71 / 2 entries | 3 / 46 / `[]` |

    After that, only the wait encoding was switched back to suspending (lane ownership of the index kept, register settings back
    to their original values): gate 1.2348 ms, spill 3799504 / 3286272, `mbar_objs` 30 / `mbar_unresolved` 61. Two readings:
    - **The readings point at registers, not just issue slots.** Spill sectors differ by one to two orders of magnitude between
      the two encodings and drop back when switching back to suspending. That tree's records attribute it to the back edge of
      the hand-written `while` landing inside the element-wise chain and cutting the batch of fragment elements live across the
      wait `[INFERRED: no isolation experiment; the same package also changed how the index is held -- for that half see
      [`gather_index_lane_owned`](gather_index_lane_owned.md) gotcha 6]`. The actionable part does not depend on this
      attribution: on a tree whose wait sites sit **inside the element-wise chain**, count `LDL` / `STL` once before and once
      after switching to spin -- this is the second item to check besides gotcha 6's cost surface.
    - **Spinning makes the dossier unable to recognize barrier objects.** `mbar_objs` 48 -> 4, `mbar_unresolved` 25 -> 95,
      `edges` emptied -- not one object was missing from the barrier table. This is the third form of gotcha 9 (there the objects
      are read as RING, here they fail to resolve); on a tree with a spin encoding, do not check the barrier table with
      `mbar_objs` / `edges`; check the table in the source and the PTX text instead.

13. **A "table base + warp" wait makes the dossier misidentify the barrier objects.** No extra barriers were built; the same table
    is read as two kinds of thing through two kinds of addressing: the immediate offsets at the init / arrive end read as NW CONST
    objects; the runtime offset at the wait end either reads as 1 extra THREAD object (`mbar_objs` = NW + 1, `mbar_tiers` gains a
    THREAD tier with no edge to the CONST objects, and `edges` shrinks too), or is folded onto the CONST object of the table's
    first entry (right count, wrong identity). Both readings have appeared on different trees. To decide which barrier a wait
    waits on, look at the immediate and the register source of the SASS wait instruction; do not draw conclusions from the
    `mbar_objs` / `mbar_tiers` counts; the retry block after EXIT often reads UNRESOLVED ((f), second bullet). As gotcha 9 /
    gotcha 12 said, changing the **encoding** also rewrites these axes; when both axes move together, do not credit every reading
    to the addressing.

14. **Two timestamps around one `try_wait` cannot measure its wait.** The result predicate of `try_wait` goes through the
    variable-latency scoreboard, so the following `%clock64` / `%globaltimer` read, which has no data dependency on it, issues
    first; "read the clock before and after the peek" therefore reads the issue interval, not the wait. The same phenomenon in
    three sm100 certifications:
    - this entry's section (g) certification: in the two settings where the `try_wait` peek returns true while the writer only arrives after about
      1.07k / 4.07k cycles, the median `%clock64` difference around the peek is 6 cycles in both (`test_wait` is also 6);
    - the `umma_consumer_full_peek_before_commit` certification: a lone `try_wait` bracketed on a freshly initialized barrier reads 2 cycles; in the same probe, with the end
      stamp placed after `if try_wait(...)` (control-dependent on the predicate), each not-ready peek reads 7819-7823 cycles;
    - the `cta_pair_consumer_peek` certification: asking about a phase that never completes, with the end stamp guarded by the predicate it reads 5184 ns;
      the same bracket with an independent timer read reads 0 ns.
    To measure one peek, the end timestamp must depend on the predicate (convert it to `Int32` first, then use it as a PTX
    predicate guarding the timer read, or place the read after an `if`).
    Likewise, in a profile this wait lands on the first instruction that reads the predicate, not on `SYNCS.PHASECHK` itself.

## Sources

- **PTX ISA**: *Parallel Synchronization and Communication Instructions* --
  semantics of `mbarrier.test_wait` / `mbarrier.try_wait` and the `suspendTimeHint` operand,
  phase parity and acquire semantics:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait>
- **CuTe DSL public API symbols** (the interface surface the student calls when writing code anyway):
  `cutlass.cute.arch.mbarrier_wait` / `mbarrier_try_wait` / `mbarrier_test_wait` /
  `mbarrier_conditional_try_wait` / `mbarrier_arrive_and_expect_tx`;
  `consumer_wait` / `consumer_try_wait` / `producer_acquire` / `producer_try_acquire` /
  `producer_get_barrier` / `consumer_get_barrier` / `producer_commit` /
  `consumer_release` of `cutlass.pipeline.PipelineAsync` and `PipelineTmaAsync`; `cutlass.pipeline.make_pipeline_state`
  (Producer initial phase = 1). All of these can be read directly in the source of the installed `cutlass` wheel.
- **GPU-certified probe** (the probe is not included in this release)
  -- 132 CTAs x 64 threads, 2-stage ring, 8192 handshakes (`unroll=1`, each wait site appears statically only once),
  a same-job three-variant A/B/C (`block` / `spin` / `test`, **differing only in the wait encoding**), six contracts:
  - C1-C3: each of the three variants equals the closed-form sum `Σ(i+1+lane)` (relative error 0) -- the non-blocking spin is a
    correct drop-in, the token hand-back is reliable, and the signal-side mapping of `producer_get_barrier` /
    `consumer_get_barrier` (gotcha 1) holds;
  - C4: the two PTX surfaces land on **mutually exclusive families** -- `block` emits only the try family, `test` only the test
    family, `spin` the try family (the hint-operand census is printed only, not a criterion);
  - C5: SASS suspension footprint `NANOSLEEP(block) ≥ 1` and strictly greater than the pure-query variant;
  - C6: direction of the same-job handshake latency -- `block` is **slower on both order statistics, median and minimum**
    (the quantified gain is anchored in the same-job before/after comparison on the operator side, not in this microbenchmark).
  The verdict follows the index status row.
- **KEEP evaluation** (NCU dossier census + gate window; same patch package as the `.L2::256B` prefetch width, see the
  `.L2::256B` section of [`cp_async`](../sm90/cp_async.md)): 9 wait sites (the producer's four-phase half-slot empty
  waits x4, the consumer's two full half-slots / mask plane / one-shot Q wait) all switched to (e), 1 CTA/SM
  (smem 214 KB) x 12 warps. gate window 2.0927 -> 2.0757 ms (-0.81%, same-package reading, not separable);
  `stall_barrier` 2.26 -> 2.20; `stall_mio` 0.23 -> 0.52 (the spin's smem barrier reads -- gotcha 6's cost surface now has a
  reading); `tensor_pct` 61.56 -> 62.12. Census readings in gotcha 9.
- **KEEP evaluation, second landing** (another operator tree, all 13 dynamic wait sites switched to (e)/(e'), in the same package
  as one intra-warp merge): gate -0.72%, same-package reading, not separable, within the noise band; the
  contract-axis flips were all back-edge drift (gotcha 9, second paragraph), and not one arrive / commit changed. Correctness
  contract (gotcha 10's repeated-launch form): a GPU probe with 14 contracts, all PASS (12 launches on the same input all pass
  the judge, two deterministic outputs bitwise identical).
- **Reading of the `stride=0x1` family on the `rings` axis (gotcha 11)**: the family's readings are codegen fingerprints of the
  phase / index arithmetic chain (`slots=(mask+1)/stride`; derivation in `resolve_addr` of the tool
  `skills/orchestration/orch_tools.py`), **encoding neither ring depth nor the addition/removal of wait sites**. Census over
  eleven KEEPs on the same tree: a segment rewrite with zero ring changes, depth 2->3 / 5->3, one SS wgmma chain moved between
  roles and back together with its wait sites, pure register-side rewrites (deleting the f32 staging, 32 resident factors), only
  two `setmaxnreg` setting constants changed with the AST unchanged -- the family's count and members flip with register
  settings and register-side rewrites, and flip even with zero mbarrier source changes. Reading rule: neither a prime suspect nor
  usable to prove "the depth made it into the cubin"; the footprint of depth is in the dynamic smem byte count and the barrier
  object count ([`ring_depth_lever`](../sm90/ring_depth_lever.md)).
  **Four KEEPs on the second tree (sm100 pair UMMA)** (eval job ids not recorded in the status
  record): operand ring 6 -> 5 -> 4 -> 6 reads 4 -> 3 -> 2 -> 4 entries (gate 0.3756 -> 0.378 -> 0.400 -> 0.3481 ms, the last in the
  same package as a handle fix); the staging-face swizzle patch with zero ring changes reads 3 -> 4 entries (gate 0.3836 ->
  0.4259 ms). **Four KEEPs on the third tree (sm100 hand-written mbarriers + three `setmaxnreg` roles)** (eval job ids not
  recorded): landing the four-piece set `[]` -> `[stride=0x1, stride=0x3f]`
  (gate 0.19544 -> 0.20533 ms); output face 1 -> 2 copies reads `stride=0x3f` -> `stride=0x1,slots=536870911`, no `slots=2`
  (gate 0.20533 -> 0.20733 ms); hand-written operand ring 2 -> 3 reads `[stride=0x1, stride=0x1,slots=536870911]` ->
  `[stride=0x1,slots=536870911, stride=0x2]`, no `slots=3` (gate 0.12804 -> 0.12602 ms; that chain was later abandoned and the
  passing code still has depth 2 on that ring, footprint only); one role's pure register-side +
  smem-face rewrite with zero mbarrier changes reads `[stride=0x1, stride=0x1,slots=536870911]` -> `[stride=0x1]` (gate 0.12595
  -> 0.11789 ms; the ms belongs to the rewrite itself and has nothing to do with `rings`).
- **The nine KEEPs on the fourth tree (gotcha 11's last table) and gotcha 12's whole-tree readings** (NCU dossier census + gate
  window; eval job ids not recorded in the status record): axis strings and gate readings as in the tables. Three confirmation
  readings: the confirmation run for the trip-reordering row reads
  1.3159 ms; the confirmation run for switching the wait encoding back to suspending reads 1.2434 ms; the confirmation
  run for the row where 32 registers change role reads 0.9859 ms. Gotcha 12's correctness contract (gotcha 10's
  repeated-launch form): a GPU probe, `PROBE PASS (max_diff=0.00036487803405893526) — 5/5 PASS` (12
  launches on the same input bitwise identical, the real shape issued to completion on the whole machine).
- **GPU-certified probe for (f) / gotcha 13**
  -- whole-machine CTAs x NW warps, each warp waits on its own barrier (count 32; warp 0 arrives after loading from gmem,
  so the first attempt necessarily fails and the retry path is executed); compares the DSL blocking surface and the equivalent
  inline label loop for correctness, SASS immediates and the dossier reading.
  **✅ Verified: `PROBE PASS max_diff=0.0`** (certification round 4 for this section of the entry).
- **GPU-certified probe for (g) / gotcha 14**
  (the entry name `mbarrier_peek_try_vs_test` in the probe's header comment is the name at certification time; it has been
  merged into section (g) of this entry) -- 32 CTAs x 2 warps x 4 rounds, one mbarrier with count = 32; the writer warp
  delays D cycles after the rendezvous, writes 32 known values, then arrives; the peeking warp probes once, blocks if the token is
  false, then reads the data; six settings (phase already complete + D ∈ {1e3, 4e3, 2e4, 1e5, 1e6}) x two encodings, each in its
  own subprocess. **✅ Verified: `PROBE PASS max_diff=0.0`** (certification round 2; all seven contracts
  passed in the same round: both encodings' data bitwise correct in all six settings; with the phase already complete both peeks
  are true 128/128; the `test_wait` peek is true 0/128 at D = 2e4 / 1e5 / 1e6). The `try_wait` token readings (128/128 at
  D = 1e3 / 4e3, 0/128 later) and the timestamp readings are same-job fact lines, not contracts.
  The two further readings cited in gotcha 14 come from the certification probes of
  [`umma_consumer_full_peek_before_commit`](umma_consumer_full_peek_before_commit.md)
  and [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md).
- PTX ISA `mbarrier.test_wait` / `mbarrier.try_wait` (the former non-blocking; the latter potentially blocking, may suspend until
  the phase completes or a system-dependent time limit):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>;
  public interfaces `Pointer.toint()`, `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`.
- **KEEP evaluations of the fifth tree in gotcha 11** (NCU dossier census + gate window, the second kind of evidence for this
  index; eval job ids not recorded in the status record):
  the `rings` readings of the five KEEPs, in the order above: `[]` -> two `0x10200,wrap=mask` (gate 1.1692 -> 1.1647 ms), `[]` ->
  `[stride=0x1]` (1.1062 -> 1.1183 ms), -> `[stride=0x38ad0]` (1.1647 -> 1.1778 ms), -> four entries (1.1778 -> 1.013 ms),
  `0x38ad0` -> `0x388d0` (1.0004 -> 0.997 ms); the write-out form twice `[]`, unchanged (1.0667 -> 1.13 ms, 0.9963 -> 0.9746 ms).
  All correctness PASS.
- **KEEP evaluations of the sixth tree in gotcha 11** (NCU dossier census + gate window, the second kind of evidence for this
  index; eval job ids not recorded in the status record):
  the `rings` readings of the four KEEPs, in table order: `[stride=0x1, stride=0x2e24f,wrap=mask]` -> `[stride=0x1]` (gate
  1.2072 -> 1.06555 ms), `[stride=0x1]` -> `[stride=0x1, stride=0x1,wrap=sel]` (1.02155 -> 0.9721 ms), -> plus
  `stride=0x2b24f,wrap=mask` (0.9565 -> 0.9809 ms, `mbar_objs` 37 -> 38), -> back to two entries (0.9809 -> 0.9532 ms,
  `mbar_objs` 38 -> 37). All correctness PASS; `smem_dynamic` at the two points in time 189344 / 177056 B.
