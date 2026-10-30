# ForgeKernel: distilling operator milestones (design rationale)

> **This is the original design rationale**: why the system has the structure it has, and where its
> invariants come from. The implementation differs in details, marked "In the implementation" below;
> the launch surface (tasks, runs, the command line) is defined by [v1-launch.md](v1-launch.md), and how
> a milestone is written by [`meta_milestone.md`](../../meta_milestone.md) and
> [`docs/milestones-template.md`](../milestones-template.md).
>
> The goal: build a **meta agent** (the *teacher* in the implementation) that, given a reference
> operator, distils a **milestone**. The milestone drives the inner student loop, so that another agent
> (the *student*) that **cannot see the reference's code** reproduces the reference's performance
> smoothly, on **the same hardware and the given test cases**. In essence: a **performance-equivalent
> natural-language rendering** of the reference operator.

---

## 0. Core definition

**Operational equivalence**: on the same hardware and the same test cases, an agent that cannot see the
reference's code reproduces the reference's correctness and performance from the milestone (the
performance criterion is a threshold / ratio, not an exact reproduction).

A milestone is not "the operator's code in prose" but **the structure of the optimization path that
leads to the operator** -- a **smooth curriculum** that even a weaker agent can climb step by step. What
it distils is "which decisions matter, in what order, and what triggers each", not "what the code
looks like".

---

## 1. Three artifacts and the leak boundary

The system's knowledge is cut into three parts, and **the boundaries are the core of leak prevention**:

| Artifact | Operator-specific? | Contains code? | Content | Who iterates on it |
|---|---|---|---|---|
| **per-op milestone** | yes | partly (contract / interface fragments only, **no kernel implementation body**) | the operator's architecture decisions / tiling·pipeline parameters / target (natural language + numbers) | the outer meta loop (often, burns GPU) |
| **hardware knowledge base (KB)** | **no** | **yes (example code)** | how to use hardware interfaces: tensor cores / TMA / mbarrier / WGMMA / swizzle… | accumulated across operators (rarely) |
| **meta-milestone** | no | no | the policy and vocabulary for distilling an operator into a milestone (two-bucket layering, DAG, ⊗, the dead ends harvested from trajectories) | people + a few operator samples (very rarely) |

> **Invariant L0 (the leak boundary)**: a milestone gives "the starting point + the decisions + the
> parameters", **never the end point**. "How to issue a WGMMA / how to write a TMA descriptor" is
> hardware knowledge and belongs to the operator-agnostic KB; "how this operator's 5 GEMMs connect, how
> the softmax backward is computed" is algorithm and goes into the milestone as natural-language
> decisions. Architecture decisions and parameters (tile_m, the number of stages, the warp role split,
> ...) **may** be given outright -- they are decisions, not code.
>
> **A milestone gives no kernel implementation body** (M0 included): short contract / interface-shaped
> fragments may be given, and the architecture is given as "decision + parameter values". **The red line
> is the implementation body of the reference's optimized kernel** -- M0 delivers the final
> architecture, and giving its code would hand over the answer. The details are in `meta_milestone.md`
> §4; how to write a milestone is in `docs/milestones-template.md`.

> **Invariant L1 (the meta agent is operator-agnostic)**: the meta agent's **harness code and prompts
> must carry zero operator-specific information**. Everything operator-specific (the milestone, the
> trajectory, the reference's source, the shapes, the target) may enter only as **run-time input**
> (files / paths / configuration), **never** hard-coded into the harness or the system prompt.
> - The test: `flash_attn` / `5-GEMM` / `softmax` / `tile_m` / any operator name or operator-structure
>   word appearing in the meta harness / prompts = a violation. The meta agent may meet these only in
>   the input files it opens.
> - The consequence: switching operators, the meta harness / prompts work **without a single change**;
>   needing a change to work = L1 broken.
> - This is the mechanical guarantee of generalisation: what generalises is "the teacher who knows how to
>   prepare a lesson" (meta-milestone + harness), and working on one operator must never let that
>   operator's information seep into the teacher itself.

---

## 2. Overall architecture: two nested loops (isomorphic)

| Dimension | Inner = student loop | Outer = meta loop |
|---|---|---|
| **What is optimized** | kernel code | **the per-op milestone** |
| **Metric** | latency / MFU (geometric mean over all shapes) | **how well the student reproduces the reference's performance** |
| **Fixed input (decision graph)** | the per-op milestone | **the meta-milestone** |
| **Working memory / trajectory** | the inner round records (in the implementation: `student/status.jsonl`) | the outer round records (in the implementation: `meta_status.jsonl`) |
| **Diagnostic signal** | the NCU profile | **the whole inner trajectory** (round records + NCU + git log + curve smoothness) |
| **Playbook** | NCU symptom → code fix | trajectory symptom → milestone change + KB example |
| **Dead-end table** | performance regressions | milestone-change regressions (a change that made the student **worse**) |
| **R0 discipline** | one minimal cut at a time, NCU-gated | **one minimal milestone edit at a time, transfer-gated** |
| **One iteration unit** | one decision Dx.y | one milestone defect fixed |

The outer loop's "NCU" is the optimization trajectory the inner loop produces. Both loops advance top
down in phase order (see §4.3).

---

## 3. The inner student loop

The inner loop is a student loop that knows nothing about any particular milestone (its state is in the
round records, its decision graph in the milestone; the harness generates its configuration for every
run). Because the goal is "execute a known good path smoothly" rather than "search without the code",
three design points matter:

1. **Relaxed revert**: a performance regression midway does **not** trigger a strict revert. The
   student is executing a path the meta agent has validated, and some decisions are scaffolding for
   later gains (a `⊗` atomic group, say, regresses when enabled alone). A strict revert would keep it
   from crossing the valley in between.
2. **Correctness is still a hard gate**: any step that breaks correctness must be stopped -- it is the
   only reliable line between "the decision was wrong" and "the decision was right but is not fast yet".
3. **The knowledge base can be consulted**: when the student cannot write some hardware interface, it
   looks up the KB's example code instead of waiting for the milestone to give code.

> A side effect (for the meta agent): with no revert, the trajectory keeps steps that "regressed but
> were kept". The meta agent can no longer rely on the student's REVERT marks to tell right from wrong,
> and reads the auxiliary signals of §4.2 instead.

---

## 4. The outer meta loop

### 4.1 Reading the trajectory: batch (default)

The meta agent reads **the whole trajectory once the inner loop plateaus or finishes a segment** (not
real-time streaming). It is cheaper, attribution is cleaner, and it is less likely to mistake "the
student is incapable" for "the milestone is underdetermined".

### 4.1bis What to feed it: layered (a mechanical digest + raw artifacts on demand), not everything at once, not pre-summarised by an LLM

"Give it everything" = the round records plus every round's git diff + full job logs + NCU CSVs (MBs, and
not scalable over operators × iterations). Two layers instead:

- **Layer 1 (the default input) = a structured trajectory digest, extracted mechanically (a parser, not
  an LLM)**, with a fixed operator-agnostic schema:
  ```
  target/achieved/gap
  curve:       [{iter, metric, verdict}]
  decisions:   [{iter, id, change, correctness, metric_before/after, verdict, binding_constraint}]
  regressions: [{id, symptom, iter, retry_precondition}]   # dead-end table
  profiler:    [{iter, job, binding_constraint, key_metrics, conclusion}]
  milestone_ref + raw_artifact_paths
  ```
- **Layer 2 (drill down on demand) = raw artifacts as paths only** (the full round records / git / NCU
  CSVs / logs); the meta agent pulls them **only when a load-bearing conclusion needs checking**.

**Why not pre-summarise with an LLM**: the summary's blind spots become the meta agent's blind spots. An
observed failure mode: midway through a trajectory a wrong conclusion was drawn ("an occupancy wall,
only a retile helps") and corrected only two rounds later to register spill -- a meta agent given only
the summary frozen at that point would have inherited the misjudgement. Mechanical extraction only
rearranges **fields that are already structured** and reinterprets nothing, so no load-bearing signal is
lost; an LLM pre-summary adds a filter with judgement of its own, which is dangerous.

> **A hard consequence for the inner loop**: for Layer 1 to be extracted reliably, **the inner round
> records must be structured** (decision id / metric before-after / verdict / per-round
> binding_constraint / dead-end table / NCU findings) -- free text does not parse reliably. The
> `binding_constraint` field is especially important: without it, diagnosis has to read through
> thousands of lines of raw logs to reconstruct it; with it, the case is clear at a glance.
>
> In the implementation: `status.jsonl` holds one record per round with **objective facts only** (the
> technique and change applied, correctness, readings, NCU) and **not** the student's subjective
> analysis, so the meta agent diagnoses from measured hardware facts rather than from the student's
> narrative; the Layer-1 digest is built mechanically by `build_digest()` in
> `src/forgekernel/inner/trajectory.py`.

> The meta agent's two other inputs are separate from the trajectory: **the milestone** (needed for both
> diagnosis and patching) and **the reference's source** (needed only for R2, adding an optimization
> decision; teacher only).

### 4.2 Diagnosis: classifying where the student is stuck (two classes, G and R)

> The operational version is `meta/diagnose_prompt.md`; the policy's single source of truth is
> `meta_milestone.md` §7.
> **G, guidance failure** (the student cannot move): G1 underdetermined / G2 too coarse /
> G3 hardware interface unknown / G4 student noise.
> **R, reach failure** (the student follows correctly but tops out below the target): R1 foundation
> ceiling / R2 missing optimization decision.

| Symptom (read from the trajectory) | Root cause | Fix | Where it lands |
|---|---|---|---|
| searching blindly / several steps all reverted / no decision gives a direction | G1 underdetermined (the milestone) | add an abstract decision / add a trigger condition | per-op milestone |
| steps too big; the model keeps getting the same decision wrong | G2 too coarse (beyond the model's ability) | **split** that milestone step | per-op milestone |
| knows what to do but cannot write it (does not know a hardware interface) | G3 hardware interface unknown (operator-agnostic) | add **example code** to the knowledge base | hardware KB |
| correctness wrong again and again / wrong mental model (the interface is known) | G4 student noise | tolerate / retry, leave the milestone alone | -- |
| follows correctly but tops out early; levers within the phase exhausted; a foundation value is pinned | R1 foundation ceiling | loosen the foundation decision / parameter (a structural patch) | per-op milestone (foundation) |
| follows all the way but falls short; the reference has it and the milestone lacks it | R2 missing optimization decision | add that upper-layer decision | per-op milestone |

Auxiliary diagnostic signals (replacing the REVERT marks):
- **whether correctness held throughout** (hard)
- **whether the final performance reaches the target**
- **the smoothness of the performance curve**: a deep dip followed by recovery / topping out early ⇒ the
  foundation milestones are in the wrong order or a prerequisite decision is missing (a strong signal
  for R1)

> Validated early on a real inner trajectory: a fresh meta agent, given only operator-agnostic prompts,
> correctly diagnosed R1 (a pinned tile parameter capping performance through spill), located the
> shallowest foundation decision, and proposed the direction of a structural patch.

### 4.3 Edit preference: foundation first (phase order + time-dependent cost)

The cost of changing the d-th milestone ≈ the inner-loop work already stacked on top of it. Hence:
- **Early in the meta loop**: nothing is stacked yet, **changing the foundation (M0/M1) is cheapest**,
  and the foundation sets the global ceiling (the inner loop is locally optimal, and a good foundation is
  what lets it climb to the best performance) → get the structural foundation right in one go now.
- **Late in the meta loop**: the foundation is validated, and touching it again would throw away a lot
  → switch to **additive / refining** patches on the upper layers (M2+).

> The rule: the meta loop follows the M0→M5 phase order itself -- **lock the foundation milestones
> first, then refine upward**. Going back to change an early milestone late = wasted work in both loops;
> avoid it as far as possible (spend the early structural window on it instead).

### 4.4 Patch types and restart scope

| Patch | Effect on the inner loop | Cost |
|---|---|---|
| **additive** (add a decision at the end of the DAG) | earlier KEEP decisions stay valid; **resume with prefix reuse** | low |
| **structural** (change a DAG edge / a ⊗ group / a foundation parameter) | the downstream decisions that depend on it are void; **partial / full restart** from there | high |

Dependency tracking (§6) decides how deep a structural patch has to rerun. An early structural patch on
the foundation is cheap because little sits downstream -- which is exactly what makes "foundation first"
workable.

> In the implementation: a patch's outcome is judged on the inner trajectory segment after it and
> recorded in `meta_status.jsonl`; a patch that brought no attributable improvement is recorded as
> `REVERT_PATCH` and rolled back, and the next diagnosis reads this memory so that a direction that
> already regressed is not proposed again (`src/forgekernel/meta_trajectory.py`).

### 4.5 No big refactors

A milestone is a **smooth curriculum**: each step is "get one architecture decision right", so that a
weaker agent solves a complex task step by step. So a milestone **should contain no large refactors or
repeated attempts**. Concretely: whatever tile / pipeline the reference uses, the meta agent reads it
from the source and **writes it straight into the foundation milestone as a given parameter**; the
student has the right configuration from round 1, and **there is no mid-course refactor such as a
retile**.

---

## 5. Scope and leak prevention

**The only leak red line is L0.** Shape parameters may be hard-coded -- that is not a leak, it is an
honest statement of a narrow scope. The leak question then collapses to one thing: the disclosure
boundary.

- **The disclosure boundary (the red line)**: the target may be given, architecture decisions and
  parameters may be given, short contract / interface fragments may be given; **no kernel implementation
  body may be given (M0 included)**, and **least of all the reference's optimized implementation body**
  (L0, see §1). A milestone may target specific shapes. The Worked Example is not a "suspect zone"; it
  is **the label of this milestone's range of validity**.

- **Coverage = the shapes given this time**: a run is given several shapes, performance is judged on the
  geometric mean, and the milestone's scope is exactly those shapes. A target shape outside that scope
  (changing head_dim flips the decision set, say) → **a new task with its own milestone** for that
  regime.
  - **Generalisation = a family of milestones tiling the whole shape space**; once all target shapes are
    covered, the operator has generalised over shapes.

- **Distil faithfully, cover by tiling**: better one correct narrow milestone than one wrong wide one.
  Abstraction (shape-free) is a bonus, not the goal -- so MDL pressure is no longer a hard constraint,
  only a weak preference for "abstract where you can".

---

## 6. Prefix reuse

Prefix reuse lets the inner loop continue from the point of change after a structural or additive
patch, instead of restarting from 0. It needs:

1. **A checkpoint definition**: after each inner decision, the triple
   `(milestone version, round-record snapshot, code git commit)` is a checkpoint, indexed by decision
   id.
2. **Milestone versioning**: each meta patch produces a milestone version, with a record of "which
   decision d this version changed relative to the previous one".
3. **A decision dependency graph**: for each inner decision, which milestone entries (decision ids) it
   depends on. A patch that hits entry X ⇒ every checkpoint after d*, the earliest decision depending on
   X, is void; resume from the last valid checkpoint before d*.
4. **A resume entry point**: the inner loop can "continue from a given checkpoint (code commit + record
   prefix)", not only start from the baseline.

> In the implementation: `src/forgekernel/resume/` (`compute_resume` computes the resume point and the
> invalidated checkpoints from the entries a patch hit).

---

## 7. The path to generalisation

- **A per-op milestone is single-use**: one per operator, with no attempt at cross-operator reuse.
- **What is reused across operators is the meta-milestone + the KB**:
  - a new operator → the meta agent distils a new milestone with the same meta-milestone;
  - where something does not fit → the fix is folded back into the **meta-milestone** (not into a
    per-op milestone);
  - a new hardware interface is needed → it goes into the **KB** (operator-agnostic, benefiting every
    later operator automatically).
- In one sentence: **what generalises is "the teacher who knows how to prepare a lesson"
  (meta-milestone + KB), not "a universal lesson plan" (any one milestone).**

---

## 8. File layout

```
meta_milestone.md            # cross-operator: the policy for writing milestones (two buckets / DAG / ⊗ / dead ends harvested from trajectories)
docs/milestones-template.md  # cross-operator: the milestone meta-template (writing rules)
meta/                        # teacher and student prompts (meta/cuda/: CUDA C++)
hw_kb/{cutedsl,cuda}/        # cross-operator: the hardware-interface knowledge base (operator-agnostic), by GPU architecture
tasks/<op>/                  # the task definition the user writes
.artifacts/<op>_<id>/operators/<op>/
  milestone.md               # per-op: written by the distillation step, refined by the patch steps
  student/status.jsonl       # the inner round records
  meta_status.jsonl          # the outer round records
  eval/                      # timing and correctness (every shape × every input group)
.artifacts/<op>_<id>/ws/     # the student's workspace (the kernel code)
```

---

## 9. How it was bootstrapped

The work started from one operator as the gold standard (an attention backward kernel): an
expert-written milestone, the matching source and an inner trajectory of a dozen-odd rounds were at
hand -- a ready-made single-operator development / evaluation bench. The order was:

- **Phase 0 -- validate the consuming side (cheapest; a go/no-go gate)**: give the existing hand-written
  milestone to a student that cannot see the operator's code, run the inner loop, and measure how far
  it reproduces -- "the reproduction ceiling of a good milestone". If even the hand-written version
  cannot carry the student, fix the consuming side first (most likely by filling in the KB).
- **Phase 1 -- the L0 boundary**: mechanically keep the student from reading the reference's source
  paths.
- **Phase 2 -- prefix reuse (§6)**: make the inner loop checkpointable and resumable.
- **Phase 3 -- the meta extraction side**: the meta agent generates a milestone from the source, compared
  with the hand-written one (structurally: does it capture the same decisions / ⊗; operationally: does
  the student do as well with the generated one). The hand-written version is the gold standard for the
  meta agent's output.
- **Phase 4 -- close the outer loop**: the meta agent reads the trajectory → diagnoses → patches
  foundation first → resumes with prefix reuse. The student's failures naturally rediscover dead
  ends, which the meta agent harvests into the dead-end table.

---

## 10. Risks

- **Reachability of the target**: even an expert-written milestone may not hit the target exactly
  (when it deliberately forbids a retile, for example). A distilled milestone should avoid that trap
  because "parameters may be given", but the equivalence criterion should still be a threshold, not
  an exact reproduction.
  > In the implementation: the target is a speed ratio over the reference (`[target]` in
  > `task.toml`, 1.2x by default), checked against a same-job confirmation.
- **Outer-loop cost**: every outer iteration = one segment of the inner loop (GPU hours). Prefix reuse
  (§6) is the lifeline for cost.
  > In the implementation: prefix reuse is `src/forgekernel/resume/` (§6).
- **Credit assignment**: the student model and budget must be fixed for a performance gap to be
  attributable to the milestone rather than to the student's ability.
  > In the implementation: both are written in the task (`[agent.student]`, `[budget]`), and a run
  > cannot override them.
- **Granularity vs. capability**: the finer the milestone, the easier each step but the more steps. A
  weaker student can force a finer milestone; the optimal granularity is an implicit knob of the meta
  loop. It is not tuned actively.
- **KB cold start**: phase 0 is likely to expose many missing hardware-interface examples, which need
  to be filled in bulk first.
  > In the implementation: the repository ships a knowledge base for Hopper and Blackwell (`hw_kb/cutedsl/`,
  > `hw_kb/cuda/`), and each run's certified findings are added to it (`forge harvest`).
