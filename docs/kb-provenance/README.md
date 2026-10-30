# Provenance archive for knowledge-base entries (kb-provenance)

> **Written for people, not for agents.** This directory is **never in any agent's read set**: the
> student's L0 readable list holds only the knowledge base itself (`hw_kb/cutedsl/`, `hw_kb/cuda/`), and the
> teacher's inputs do not include it either. The reasons follow.
>
> **In this release the directory holds only this file.** The provenance records and certification
> probes of the entries that ship with the knowledge base are not included; each entry keeps its
> certification status and readings. The runs you start write their provenance here as described
> below.

## Why it is kept apart

The knowledge base is the one artifact **every later operator's student reads**. An entry's "source"
has two readers with two purposes, and their safety boundaries differ:

- **People (reviewers / maintainers)**: checking that a recipe is not made up -- they need to see
  "extracted from which line of which reference implementation, and what the reference binary's
  measured footprint is".
- **The student agent**: getting past "I don't know how to use this interface" -- it needs only the
  mechanism, a minimal example and the certification status.

Writing a reference implementation's **file names / function names / line numbers / paraphrased content
/ measured footprints** into an entry hands the reference solution to the same operator's student
through the knowledge base's side door: an exact symbol name (an internal function name of some public
repository, say) is also a retrieval key into the base model's pretraining memory, with more bandwidth
than any paraphrase. Blackbox mode's premise that the reference cannot be read would no longer hold.

So the evidence chain is split in two layers by reader:

| Where | What | Readers |
|------|------|------|
| the "source" of a knowledge-base entry (e.g. `hw_kb/<language>/<arch>/<entry>.md`) | certification status, probe path, links to official docs, **API symbols** of public infrastructure | student + people |
| `docs/kb-provenance/<entry>.md` (this directory; `docs/kb-provenance/cuda/` for the CUDA knowledge base) | where in the reference implementation it came from (file / function / line), paraphrased content, the reference's measured footprint, the experiment narrative | people only |

"API symbols of public infrastructure" means the interface surface the student calls anyway when
writing code (public functions of cutlass / CuTe DSL / quack, PTX instruction names) -- a tool's
symbols may be cited; what must be kept out are the symbols of the **reference solution**.

## What writes here

- **Recipe certification**: when the teacher certifies a recipe, it writes the narrated provenance
  into the file here with the entry's name (see `meta/kb_recipe_prompt.md`).
- **The end-of-run knowledge-base write-up** (at the end of a run, or `forge harvest`): likewise.
- **The gap queue** `gaps/<op>.md`: generated mechanically by the driver from the candidate table when
  reconnaissance completes and at the finish -- rows marked `VERIFY none`, and rows that say "may not
  be expressible" but carry no `[EXPR:]` verdict. They are reconnaissance's reading of the reference's
  structure, not mechanism knowledge, and **never enter the knowledge base** (neither as a `[pending certification]` slot nor as a
  placeholder row); to verify one, start a separate probe, and its
  evidence comes here as usual. The queue is regenerated on every run and is not held by the "refuse to
  start while unharvested" gate.

A person reviews the files here and commits them together with the matching knowledge-base entries
(see [docs/kb-harvest.md](../kb-harvest.md)).

## Rules

- Inside an entry, a source **may be pointed to but not paraphrased**; the reference solution's symbol
  names / file names / line numbers all belong in this directory.
- A file written here has the same name as the entry it backs.
- Mechanical gate: `src/forgekernel/loop/kb_lint.py` scans the knowledge base for banned tokens at two
  points, loop start and recipe certification. A hit at certification goes back to the teacher to fix
  first; if it still fails, the entry is moved out of the student-readable tree (into `kb_quarantine/`
  at the repository root). A hit at start refuses the start (`tests/test_kb_lint.py` guards this too).
  A source-mode run also derives tokens mechanically from the resolved reference tree (file names + def
  names), so protection does not depend on a hand-kept list.
- This directory is not subject to the lint -- it is where that content is meant to live.
