# gpu_probe_ladder (instrument engineering for multi-contract GPU probes: process isolation / self-contained digest / fact–verdict split / candidate sweep / two-marginal bijection)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is a **reading
> recipe**: it teaches no mechanism; it only teaches how to build the probe that carries the verdict so
> that one GPU round brings back **more than one** fact.

## Purpose

A GPU round is the most expensive resource in this pipeline, and one round can only ask once. The
**structure** of the probe decides how much information the round brings back: the same submission can
return "11 FAILs, 1 fact", or "5 verdicts + a measured table of the candidate space". This entry
collects the constructions that turn the former into the latter, all verified on the GPU in the form
"add one cut, the next round's verdict flips".

## When needed (trigger)

When writing any multi-contract / multi-rung GPU certification probe. And after these symptoms appear:

- A round comes back **all FAIL while carrying only one fact** (every rung died at the step they share);
- Some rung has "no verdict", and the stretch of output holding the reason has already scrolled out of
  the retained log tail;
- The verdict goes round in circles on questions like "which spelling fits", **burning one GPU round per
  guess**.

## Minimal runnable example

### 1) One process per rung

```python
# (stage, budget seconds, sanitizer allowed?, verdict tags owned by this stage)
PLAN = [
    ("S",     130, True,  ["S1-row-ramp", "S2-col-ramp"]),
    ("L0",     50, True,  ["L0-path-lowers"]),
    ("E",      90, True,  ["E1-row-ramp", "E2-col-ramp"]),
    ("whole", 210, False, ["C1-smoke", "C2-numeric", "C3-footprint"]),
]

def _run(stage, timeout, sanitize=False):
    cmd = [sys.executable, os.path.abspath(__file__), stage]      # re-exec itself
    if sanitize:
        cmd = ["compute-sanitizer", "--tool", "memcheck"] + cmd
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as te:                        # hang = timeout
        return "timeout", (te.stdout or b"").decode("utf-8", "replace")
```

The child talks to the parent only through prefixed lines, and the parent routes them by prefix:

```
[[BEGIN]]<tag>     # printed before touching CUDA; distinguishes "never started" from "died silently after starting"
[[V]]tag|ok|md     # verdict (the parent turns it into PROBE PASS/FAIL)
[[E]]tag|reason    # the reason for a verdict
[[N]]...           # fact (produces no verdict)
```

### 2) Self-contained DIGEST: fold the cause of death into the last printed block

```python
def _postmortem(stage, rc, dt, body, keep=14):
    lines = body.splitlines()
    head = ["stage %r exit=%s after %.0fs, %d bytes / %d lines"
            % (stage, rc, dt, len(body), len(lines))]
    if not lines:
        head.append("  child produced NO OUTPUT AT ALL "
                    "(process never reached its first print)")
        return head
    if not any(ln.startswith("[[BEGIN]]") for ln in lines):
        head.append("  no [[BEGIN]] marker: died before the stage body "
                    "(import / interpreter / launcher level)")
    interesting = [ln for ln in lines
                   if ln.startswith(("[[E]]", "[[parent]]", "note:"))
                   or re.search(r"Error|error:|Traceback|assert|CUDA|"
                                r"Exception|Killed|Aborted", ln)]
    for ln in (interesting[-keep:] if interesting else lines[-keep:]):
        head.append("  | " + ln[:200])
    return head
```

The parent calls it for every stage that "has tags but no verdicts" and **folds the returned lines into
the digest**; the digest is the last thing printed in the whole run, so the cause of death survives log
truncation by construction.

### 3) Candidate sweep instead of round-by-round guessing (compress N rounds into 1)

Some questions are not "does the mechanism work" but "which acceptable view spelling fits" — they have
no answer you can reason out, only one you can measure. Don't guess one per round:

```python
_CANDS = ("shipped", "spelling-A", "spelling-B", "reinterpret", "flat-dst")

# (a) first print the layout of every candidate as a fact (including those that don't lower)
def attempt(f):
    try:
        t = f()
        return "shape=%s layout=%s" % (t.shape, t.layout)
    except Exception as e:                    # failing to lower is a fact too
        return "<%s: %s>" % (type(e).__name__, str(e).splitlines()[0][:90])

nrec("shape  spelling-A  = %s" % attempt(lambda: tiled.retile(rSrc)))
nrec("shape  partition_D = %s" % attempt(lambda: thr.partition_D(sDst)))

# (b) one subprocess per candidate, all running the same round-trip reference
for cand in _CANDS:
    for mode in ("row", "col"):
        md[mode] = one(cand, mode)          # exception text is recorded as a fact too
    nrec("cand %-22s row=%-42s col=%s" % (cand, md["row"], md["col"]))
    if cand == "shipped":
        vrec("bijection", md["row"] == 0.0 and md["col"] == 0.0, ...)
        #  ^^ only the shipped candidate is a verdict
```

**The key is in the last two lines**: non-shipped candidates always go through the fact channel. So
"learning which spelling is right" never itself reads as `PROBE FAIL`, and one sweep round brings back
the measured table of the whole candidate space.

### 4) Two-marginal ramps pin the bijection

One marginal cannot pin a bijection: a row ramp is blind to column permutations, a column ramp is blind
to row permutations, and **a mapping that preserves both marginals at once is the identity**.

```python
# batch 0 = row ramp (value = packed row index); batch 1 = column ramp (value = column index)
# both ramps are integers < 256 -> exactly representable in bf16 -> the criterion is a strict 0.0, not a tolerance
ref[0, r, c] = float(r)
ref[1, r, c] = float(c)
verdict("bijection", max_diff_row == 0.0 and max_diff_col == 0.0)
```

### 5) Pay for the sanitizer only on faults with "no source line"

```python
NO_LINE_FINGERPRINT = re.compile(             # CUDA-level causes of death: carry no source line by nature
    r"illegal memory access|unspecified launch failure|misaligned address",
    re.IGNORECASE)
py_level = "Traceback (most recent call last)" in body
line_less = (rc != "timeout") and not py_level and bool(
    NO_LINE_FINGERPRINT.search(body)          # illegal memory access / launch failure
    or (isinstance(rc, int) and rc != 0)
    or missing)                               # tags but no verdicts = silent death
if sanitizable and line_less and any_failed:
    retry.append((stage, tags))               # rerun only the **lowest** failing rung
```

## Gotchas

1. **A per-rung `try/except` is not isolation; it only looks like it.** A sticky CUDA context error, an
   illegal memory access or a hang is **not a Python exception** — it takes down every later rung in the
   same process. Empirical shape: all five rungs report `no verdict emitted (stage rc=0)` while there is
   only one real cause of death. **The granularity of isolation must be the process** (the parent
   re-execs itself + `argv[1]` selects the stage), not a `try` block.
2. **Infer verdicts from "was there output", not from the exit code.** A rung may have `rc=0` yet emit
   no verdict at all. The parent must go by the tag list: `missing = [t for t in tags if t not in seen]`;
   every missing one is recorded as FAIL + reason and triggers a postmortem.
3. **The log tail gets truncated, and the reason is by default exactly in the half that gets cut.** The
   parent's echo of the child's output is printed **before** the digest, so when you need it, it has just
   scrolled away. The fix is §2: **fold** the cause of death into the digest. Also distinguish three ways
   of dying — "no output at all" (the process never reached its first print), "no `[[BEGIN]]` marker"
   (died at the import/interpreter/launcher level), "marker but no verdict" (died inside the stage body).
   The next action for each is completely different.
4. **A ladder cannot locate a fault in the step all its rungs share.** If every rung and the shipped file
   write that shared step with the same spelling, the N FAILs that come back carry only **one** fact — the
   ladder's whole resolution is wasted. Two fixes; pick one per situation:
   - the shared step **calls the shipped file's own constructor** instead of re-spelling it (same as
     [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 3);
   - or, if the shared step itself is the unknown, **promote it to the swept axis** (§3).
5. **A shape/spelling rejection is not a "does the mechanism work" problem; don't debug it as one.** For
   example `Expected source and destination tensors to have the same size in mode-1,
   but got 1 and 8` asks "which acceptable source view fits this destination" — chased by guessing it
   **burns one GPU round per round**, and each round rules out only one candidate. Switch to the sweep of
   §3: run them all in one round.
6. **Don't let "learning" read as FAIL.** Non-shipped candidates in a sweep, layout prints, api-audit and
   other observations always go through the fact channel; only contracts occupy the `PROBE` verdict line.
   Otherwise the structure of 10 PASS and 1 FAIL in one round loses its meaning — someone reading the log
   cannot tell "the kernel is broken" from "I just learned that 3 of 4 candidates don't lower". Same
   discipline as [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 4 for api-audit.
7. **Split compile-only contracts and launch contracts into two.** A failure of "can it lower" throws a
   Python exception with a **line number**; a failure of "can it launch" has **no line number**. With two
   contracts, which one the FAIL lands on is itself a piece of locating information; merged into one, the
   two kinds of fault are indistinguishable. Side benefit of a compile-only contract: it **cannot hang**,
   so it can safely be placed anywhere.
8. **One marginal cannot pin a bijection.** A row ramp cannot see column permutations, a column ramp
   cannot see row permutations. Run both, and choose values that are **small integers exactly
   representable in bf16** (< 256), so the criterion is a strict `max_diff == 0.0` — not a tolerance that
   needs explaining.
9. **A sanitizer is only worth it for faults with no source line, and only affordable for small rungs.**
   A Python traceback already names the line; **a hang is not memcheck's domain** (a wrong mbarrier
   arrive-count or phase waits forever rather than faulting, and that can only be located by the process
   split of §1). The 10–100× instrumentation slowdown is only affordable on single-CTA small rungs; **a
   whole-machine stage is never sanitized**; within budget, rerun only the **lowest** failing rung (higher
   rungs are most likely its downstream).
10. **The parent needs a global deadline, not just per-stage timeouts.** When the budget runs out, record
    the remaining tags as "skipped: out of job budget" and print the digest as usual — a probe cut off
    by the job system for timing out brings back zero facts, the most expensive kind of failure.

## Sources

- This repo's GPU certification (gotchas 1, 2, 3, 4, 5, 6, 8, 9): an epilogue ping-pong GPU probe
  (the probe is not included in this release) — all eleven contracts pass in the same round,
  including `PROBE PASS max_diff=0  # S1-r2s-row-ramp` / `PROBE PASS max_diff=0  # S2-r2s-col-ramp`
  (the two-marginal pair of §4, the shipped candidate found by the sweep), `PROBE PASS max_diff=n/a  # L0-store-path-lowers`
  (the compile-only contract of §7), `PROBE PASS max_diff=n/a  # C1-whole-machine-smoke`.
  The failure shapes of gotchas 1/3 (all five rungs `no verdict emitted (stage rc=0)` in the same round,
  the reason already scrolled out of the log tail) and of gotchas 4/5 (every rung hitting the same mode-1
  shape rejection) are recorded in that probe's per-version docstrings.
- This repo's GPU certification (gotchas 5, 6, 8): a second epilogue probe — a
  four-candidate sweep settled it in one round, after which `PROBE PASS max_diff=0.0  (bijection ...)`
  serves as the regression contract, in the same round as `PROBE PASS max_diff=n/a  (smoke: ... lowers + launch completes)` and
  `PROBE PASS max_diff=0.009644746780395508  (whole kernel vs fp32 math SDPA)`.
- How to write the verdict line itself (PTX/SASS count contracts) without wronging the kernel:
  [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md).
- The concrete spelling settled by the sweep (the source view of the epilogue r2s):
  [`tma_store_epilogue`](tma_store_epilogue.md) gotcha 14.
