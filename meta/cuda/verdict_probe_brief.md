## Verdict experiments (optional, budget {MAX_VERDICT_PROBE_ROUNDS} GPU-job rounds)

**Before** giving the final diagnosis, you may run bounded experiments on real hardware to decide between hypotheses. Once a diagnosis lands, the granularity at which it
gets tested is a whole segment of student iterations (hours); a verdict experiment costs only one GPU job (minutes).
When the evidence is not enough to decide, experiment first, then diagnose.

**Protocol**: write a self-contained python script to `notes/<name>.py` (inside the student workspace; it travels to the node with the job tarball
and runs there), then make **this one turn end with a single line on its own, without writing cause_class**. This run's kernel is single-file
CUDA C++: write the variants to compare as `notes/<name>_<variant>.cu` (a copy of the kernel file + switches/instrumentation); the script compiles and loads them with the operator tree's own
builder — `import sys; sys.path.insert(0, "eval"); from leaderboard import load_cuda_entry` —
with the same compile options as the kernel; the first compile of the CUTLASS headers is heavy (about half a minute per variant), so count the number of variants against the job clock:

```
PROBE_REQUEST: notes/<name>.py
```

The driver commits, submits the GPU job, and feeds back to you the log tail (your experiment's readings) and the verdict-line audit; you can then
send the next experiment or give the final diagnosis. Once the budget is used up you must diagnose from the evidence you have. After the diagnosis the driver resets the
student workspace to its pre-probe state (the scripts are archived on the `-metaprobe` branch) — probe files are **not** left to the
student, so do not let the diagnosis or the patch cite their paths.

**When a job is worth spending**:
- two or more cause hypotheses are compatible with the existing trajectory/NCU evidence, and choosing the wrong direction costs a whole segment of iterations;
- you are about to propose a **structural/foundation** patch (invalidating many completed decisions) — falsify the cheaper explanations first;
- the trajectory shows several REVERT_PATCH / "patch had no gain" in a row — first suspect that **the best watermark is poisoned by an outlier**:
  re-measure the current best commit in the same job before concluding "the direction is wrong";
- you suspect some milestone contract **never really landed** (e.g. an exclusive region / hiding relation rewritten by the compiler) — verify it down to SASS.

**Experiment design rules**:
1. **Pair within one job, read only ratios**: node bimodality makes absolute values drift ±6% across jobs, wider than most single-round gains. All
   variants (control included) must be measured alternately in the same job, and conclusions are read only from ratios.
2. **One variable per variant, pre-registered criteria**: for each hypothesis, have the script itself print one line
   `PROBE PASS/FAIL max_diff=<reading>` (write `max_diff=n/a` for a non-numerical contract) — the verdict is put in the
   log by the script, not by your after-the-fact interpretation. **One contract per line**: the driver records line by line (verdict + reading verbatim + end-of-line comment);
   a reading can be a number, `n/a` or a `k=v,...` string; a control contract that fails to run affects only its own line
   and does not wipe a decisive contract's PASS into a FAIL. In the end-of-line comment, name the contract and whether it is control or decisive.
3. **PASS and FAIL are both evidence**: a FAIL in a verdict experiment is not a failed experiment, it is a falsified hypothesis — often
   the most valuable result. Write the verdict into the final diagnosis's `evidence` field (cite the probe name + reading).
4. **Measure only, do not build**: a probe is a measuring instrument — add switches/instrumentation/re-measurements on the existing student code; do not deliver
   a new implementation, let alone include any ORACLE implementation-body code (the script goes into the probe job's snapshot of the student tree).
   **Measured numbers** of the reference may be used (legitimate anchors).
5. The script may call tools inside the job image (ncu, SASS dump, etc.) for attribution, not only for timing.

**Verdict experiments without a script (ablation)**: when the hypothesis is "some mechanism already on the tree (a barrier layer, a pipeline segment, a
stage) is currently costing how much", you need not write a measurement script yourself — write the ablation as JSON:

```
PROBE_REQUEST: notes/<name>.json
```

The file format is the same as the pricing round's `ablations` (`{"ablations": [{"name", "decisions", "hypothesis",
"edits": [{"old", "new"}]}]}`; each entry removes that mechanism or degrades it to a naive form, `old` occurs exactly once
in the kernel, at most 4 entries). The driver runs it with a frozen recipe: base and each ablation alternate in the same job, reading the NCU kernel clock, and **wrong
results are allowed** (what the ablated kernel computes does not matter, only the time). ncu runs first; a timeout marks it `hung` and it is not touched again (no
bench, no later rounds), and the ablation segment has its own total budget; **an ablation that deadlocks still costs one ncu timeout**, so if it can be written as a degraded form
(a coarser synchronization that still runs), do not write it as a deletion that hangs. A reading Δ that is positive and above the noise floor = faster without it = an upper bound on that mechanism's current
cost (hypothesis holds); within the band = not a major item; Δ negative beyond noise = slower without it = it is helping (hypothesis falsified). Only a positive,
above-noise Δ can price an R1 (`reopen_price: <Δ> via ablation:<name>`), and it is not the gain of any fix. The hypothesis "the reference has no such wait layer"
cannot be tested against the reference in a blackbox run; ablation is the only way to answer it on the student side.
