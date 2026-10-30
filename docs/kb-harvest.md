# Knowledge-base harvest: how certified recipes get into the repository

> **Written for people, not for agents.** This document lives in `docs/` and not in `hw_kb/` on
> purpose: the knowledge-base directories are in the student's and the teacher's read sets (the L0
> "readable" list of `meta/student_prompt.md`, the inputs of `meta/kb_recipe_prompt.md`), and putting
> the harness's operating procedure there would push into the agents' context something they have
> neither the right nor the need to take part in.

## The problem: no code commits the main repository for you

When the teacher certifies a recipe (diagnosis class G3, hardware interface unknown: the student
cannot use some hardware interface), it writes it **straight into the main repository's working tree**, into the knowledge-base
directory (`hw_kb/cutedsl/` for a CuTe DSL run, `hw_kb/cuda/` for a CUDA run; narrated provenance goes into
`docs/kb-provenance/`). But every git operation of the driver is confined to the student's workspace --
no code commits the main repository's copy.

So a freshly certified recipe is just an uncommitted file: one `git checkout` and it is gone, after the
GPU probe has already been paid for.

**Which checkout it lands in also matters**: the knowledge base's location follows where ForgeKernel is
installed (the source tree an editable install points at), not the current directory. Start a run from
another worktree and the recipe may land in another tree's `hw_kb/`.

## What happens at the two ends of a run

**Before the start**: when the knowledge-base directory (and `docs/kb-provenance/`, except `gaps/`; the
CUDA and CuTe DSL modes each guard their own directories) has uncommitted changes, a new run refuses
to start (exit 3), so that the previous run's unharvested recipes do not get buried under the next
run. Commit or discard them, then `forge run` again. Resuming a run that has already started is not
held by this gate. When the loop is run directly on a hand-made tree (`forge <op> loop`),
`--allow-dirty-kb` starts it on a dirty knowledge base.

**At the end**: the terminal prints a `[harvest]` list (the uncommitted files left in the knowledge-base
and provenance directories, with **absolute repository paths**); the run's `report.md` lists them too,
under "New knowledge-base entries". The recipes stay in the working tree for a person to review and
then commit or discard -- until then, the next new run is still held by the gate above.

**When a run did not reach its finish**: a run stopped before its finish (a killed session, an
interruption) never does the end-of-run knowledge-base write-up. `forge harvest .artifacts/<op>_<id>`
does it afterwards: one teacher turn, no GPU job, no confirmation; what it writes is likewise left for
a person to review.

## What to look at when reviewing a recipe

A recipe's only gate is **a probe written by the same agent** -- it shows the mechanism runs on the
GPU, not that the recipe belongs in the knowledge base.

What to check is **not** "does it contain code": the knowledge base is the only one of the three
artifacts allowed to contain code, and the exemption exists **precisely because** it is
operator-agnostic. Check the other two axes:

1. **A recipe must describe a hardware mechanism only, never an operator's algorithm disguised as a
   mechanism** (L0). The knowledge base is the one artifact every later operator's student reads, so a
   recipe that smuggles in an algorithm contaminates every operator after it.
2. **A source may be pointed to, never paraphrased**: file names, function names, line numbers,
   paraphrased content and measured footprints of a reference implementation (readable source or
   blackbox alike) all go to `docs/kb-provenance/<entry>.md` (in no agent's read set); the reasons are
   in that directory's README. The mechanical lint (`src/forgekernel/loop/kb_lint.py`, run at loop
   start and when a recipe is certified; an entry that fails at certification is moved out of the
   student-readable tree into `kb_quarantine/` at the repository root) catches leaks of proper names;
   **a narrated paraphrase without proper names gets past the lint, and only a human reader catches
   it** -- that is the blind spot of this review axis.

**Note that the probe script is not in the knowledge-base directory**: it is under `notes/` in the
student's workspace, committed on the workspace's `-kbprobe` branch, and the next certification
overwrites it.

## Settings

| Switch | Effect |
|---|---|
| `--allow-dirty-kb` (`forge <op> loop`) | start even though the knowledge base has uncommitted changes |
| `forge harvest .artifacts/<op>_<id>` | do the knowledge-base write-up of a run that did not reach its finish |
