# skills/

The tools the agents use, as in ForgeTrain: one directory per skill holding a `SKILL.md`
(what it is, how an agent uses it) and its scripts. Before each agent turn the loop mirrors
every skill into the tree the agent works in (`.claude/skills/<name>/`) and keeps it out of
the diff via `.git/info/exclude`.

GPU jobs run what is committed in a tree, so the driver also seeds the scripts a job needs
into the tree's `notes/` and commits them there: a probe imports them bare
(`import sass_tools`), a frozen recipe runs them as `python3 notes/<tool>.py`. The harness
imports the same files as `forgekernel.skills.<module>` to read their verdict lines.

| skill | what it reads |
|---|---|
| [sass](sass/SKILL.md) | a kernel's SASS: structure, MMA operand sources, schedule and register liveness, per-phase construction, launch shape |
| [stall](stall/SKILL.md) | measured per-instruction stall samples: where the time leaks |
| [timeline](timeline/SKILL.md) | when it leaks: PM-sampling phase timelines and `%clock64` segment stamps |
| [orchestration](orchestration/SKILL.md) | sync topology and supply order, per-iteration census, per-role exposure |
| [lever-pricing](lever-pricing/SKILL.md) | measured prices of candidate levers by ablation |
| [job-env-check](job-env-check/SKILL.md) | whether a GPU job's `python3` can run the tools at all |
| [ncu-trace](ncu-trace/SKILL.md) | NCU metrics across iterations, for the teacher |
