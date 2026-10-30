# Meta-Agent (teacher) — Finish-time KB harvest round: turn expensive probe wins and contract-axis KEEPs into recipes (single turn, no GPU)

> Trigger one (kb-tripwire): in this run some student probe burned ≥ the threshold number of GPU rounds before its final `PROBE PASS`.
> The student won, so the inner loop never stopped as stuck and meta never had a chance to diagnose
> G3 — but an N-round pyrrhic victory exposes exactly a knowledge gap that very nearly got stuck, and the lesson is worth no less than a G3.
> Without a harvest, these lessons stay scattered across probe notes and commit messages and evaporate with the run;
> the next run that hits the same kind of construct steps into it again at full price.
>
> Trigger two (contract-axis change): some KEEP patch changed a **contract axis** of the student's SASS dossier (the driver takes a census once in the NCU job
> of every evaluated iter: loop roles, mbarrier objects/layering, ring shape, NamedBarrier
> rulings, cross-role edges, DEPBAR in-flight depth, reduction placement, supply issue order, cache-policy bits) and did so without any
> probe tug-of-war — a cheap win that KEEPs in one round. Knowledge of this kind used to have no channel into the KB at all. An observed case:
> a one-line patch changing a cp.async cache mode was worth only a few percent on an earlier tree, evaporated with the run, and was priced at 22% on another tree.
> Cheap does not mean unimportant: its value changes with the tree, so **what gets recorded is "knob + SASS footprint + under which conditions it carries load"**,
> not that one percentage.

## How this differs from the G3 certification path (important)

The G3 path (`kb_recipe_prompt.md`) is a GPU certification loop driven by the driver: you write a recipe + a new probe,
and it counts as certified only once it runs to PASS on the GPU. **This round runs no GPU job at all**: the winning probe has already returned
`PROBE PASS` on real hardware, and **that run itself is the certification evidence**. Your only job is to write it down — a recipe that cites that
probe as its source (probe path + number of rounds + verdict line) counts as verified.

## Inputs (paths injected at runtime)

- `POLICY` = `{META_MILESTONE_PATH}` — the milestone/KB rules.
- `KB_DIR` = `{KB_DIR}` — the hardware knowledge base root. The template is `{KB_DIR}/TEMPLATE.md`,
  the listing is `{KB_DIR}/{KB_ARCH}/index.md`.
- `WORKSPACE` = `{WORKSPACE}` — the student's operator workspace. **The raw material of the battles is here**:
  - the probe file itself (`WORKSPACE/notes/<name>.py` and the `notes/<name>.cu` it compiles) — the comments of its final form
    usually record the failed hypotheses and corrections of the earlier rounds;
  - `git -C {WORKSPACE} log --oneline -- notes/<name>.py` — the per-round commit messages
    record each round's pitfall and that round's fix;
  - the kernel construct certified by the probe (the probe docstring names the kernel file and line numbers).
- Appended at the end of this prompt is the **probe battle list** (for each battle: the probe path, the GPU rounds burned, the final verdict line,
  the tail of the winning job log) and/or the **contract-axis change list** (each KEEP that changed a contract axis: iter, decision id,
  sha, the readings of the changed axes in the previous census and in this one, the gate window's before/after ms, the student's change record).
  Only one of the two lists may be present.

## What you deliver

For every battle / every contract-axis change in the lists, judge whether its lesson **generalizes** (operator-agnostic hardware / CUDA C++ /
PTX mechanism knowledge), then:

1. **Generalizes → write/update a recipe**:
   - A new mechanism gets a new file `{KB_DIR}/{KB_ARCH}/<interface>.md`, strictly following the sections of `{KB_DIR}/TEMPLATE.md`
     (Purpose / When needed / Minimal runnable example / Gotchas / Sources). The minimal example is **extracted from the winning probe**
     (it is already a GPU-verified minimal exercise), with the operator-specific logic abstracted away.
   - A pitfall an existing recipe is missing → **append an item** to that recipe's "Gotchas"; do not open a duplicate file.
   - Register/update the corresponding row in `{KB_DIR}/{KB_ARCH}/index.md`, with status ✅ Verified, and state the evidence:
     `(probe <path>, PROBE PASS after N rounds, job <winning job number>)`.
2. **Does not generalize** (pure operator logic, a purely local typo) → skip it, and give the reason in the final output.
3. **How to write contract-axis change items** (a different evidence surface from probe battles):
   - Certification evidence = that KEEP's own NCU job (the axis readings of the two dossier censuses before and after + the eval's gate-window ms);
     no `max_diff` probe is needed or exists — knobs such as cache-policy bits and DEPBAR depth are **numerically invisible**,
     and were kept out of recipes precisely because "only numerical probes count as certification". The index's evidence parenthesis reads
     `(KEEP in job <the eval job number of that KEEP>, axis <axis> <before>→<after>, gate <ms_before>→<ms_after>)`
     — the iter number, decision id and sha given to you in the list are inputs for locating things, **and are not written into any KB file**.
   - The recipe body records three things: the **knob** (how it is written in CUDA C++ / inline PTX and which SASS footprint it corresponds to, taken from the change
     record and the axis readings — the footprint is the evidence; if the change record does not make the spelling clear, mark it `[to be checked]`), the **footprint** (the axis readings before and after
     the change, verbatim), and the **load-bearing condition** (this run's ms change is just one reading: write "worth X% under <this run's structural conditions>",
     not a universal percentage). If an existing recipe already lists it as an "outside the certified scope" pitfall, turn it into a
     certified recipe and rewrite the old pitfall item to point to the new section.
   - When one KEEP moved several axes at once, write them up separately per axis; if an axis reading changed but the ms did not move, record it anyway ("no value" is itself
     a transferable reading — it keeps the next run from taking it as the prime suspect again).

## Hard rules

- **Recipes are operator-agnostic**: only hardware / CUDA C++ / PTX mechanisms, never copy the operator's algorithmic logic (L0).
- **Write only what has evidence**: every claim must point back to a probe verdict, a job log or a per-round commit message;
  adding from memory a mechanism not in the inputs = contamination; better to leave it out.
- **Unverified rows of the recon table do not enter the KB**: rows with `VERIFY none`, and rows that say "may not be expressible" without an `[EXPR:]`
  verdict, have already been stored separately by the driver as `evidence/gap_queue.md` and `docs/kb-provenance/gaps/`
  (for humans; not in any agent's read set). They are the recon's readings of the reference's structure, not mechanism knowledge;
  they **must not** be written into `{KB_DIR}` as a `[pending certification]` pitfall, a `⏳` placeholder or in any other form — the KB's only entrance is
  a probe-certified recipe.
- **Do not send PROBE_REQUEST**: this round has no GPU budget and does not need one — certification has already happened.
  A recipe that needs new evidence to stand does not belong in this round: mark it `[pending certification]` in the recipe's "Sources" section, or simply skip it.
- Experiment narrative does not go into any file of `{KB_DIR}` (index.md included): run names/dates, iter numbers, commit/cubin shas
  are never written (kb_lint machine-checks this; a violation is sent back once this round, and the next loop start refuses); the evidence parenthesis carries only the job number,
  probe path and readings; lessons are phrased generically; the narrative belongs in the MR description and `docs/kb-provenance/`.

## Output format (the driver parses this)

After writing all files, end with one line per recipe:

```
RECIPE_RECORDED: hw_kb/cuda/{KB_ARCH}/<name>.md
```

(Several recipes, several lines; appending a pitfall to an existing recipe also counts as one recipe — write that recipe's path.)
If you judge that there is no generalizable lesson, output as the last line:

```
NO_RECIPE: <one-sentence reason>
```
