# CuTe DSL hardware knowledge base (hw_kb/cutedsl)

> This is the knowledge base for **CuTe DSL mode**. CUDA mode (loop language `cuda-sol`, single-file
> CUDA C++ student) reads and writes a separate, independent tree, `hw_kb/cuda/` (see its README and
> `src/forgekernel/loop/lang_mode.py`).

An **operator-agnostic** library of hardware-interface examples. It is the **only** one of the three
artifacts that is allowed to contain code — because "how to issue an MMA / how to write a DMA
descriptor" is a hardware mechanism, not the algorithm of some operator.

## Invariants

- **Hardware mechanisms only, never operator algorithms.** "How to drive the tensor core" ✅; "how this
  operator's 5 GEMMs are wired together" ❌. The latter is a decision of the per-op milestone (natural
  language, no code).
- Every entry is self-contained, minimal, runnable, and carries its own sources. When the student is
  stuck on how to use an interface, it looks here **instead of** waiting for the milestone to hand it code.
- Example code must come from a **verifiable source** (official docs / public infra / a minimal example
  we ran ourselves); nothing is invented. The source goes into the entry's "Sources" field.
- **The evidence chain is split into two layers by reader**: the "Sources" section of an entry is read
  by the student and may cite only three kinds of thing — official docs / public-infra API symbols /
  this repo's probes. The reference implementation's (oracle's) file names, function names, line
  numbers, paraphrased content and measured footprints all go to `docs/kb-provenance/<entry name>.md`
  (for human review; never in an agent read set). The rationale and the full discipline are in
  `docs/kb-provenance/README.md`; the mechanical gate is `src/forgekernel/loop/kb_lint.py`.

## Layout

```
hw_kb/cutedsl/
  TEMPLATE.md        # entry template
  sm90/
    index.md         # SM90 (Hopper) interface list + completion status
    <interface>.md   # one per interface (following TEMPLATE)
  sm100/
    index.md         # SM100 (Blackwell) list; the entry point is the "look up by symptom first" table
    <interface>.md
```

**One tree per architecture; there is no shared tree.** An entry belongs to a tree by **what it depends
on**, not by where it was written: as soon as a rule names a mechanism specific to one architecture (the
WGMMA mainloop, the register-resident accumulator, the `setmaxnreg` quota zones), it belongs to that tree
only. Conversely, an entry that names no hardware unit (DSL algebra, probe and census methodology,
warp-level arithmetic) is kept **once in each tree** — a deliberate duplication: it saves the few rounds
that every new architecture would otherwise spend re-earning it. A transcribed copy carries a
`Certification source` line saying on which architecture its readings were taken; **location is not
provenance — that line is.**

**The read set is enforced by the loop, not by convention.** A run receives exactly one tree,
`hw_kb/cutedsl/<kb_arch>/`: `KB:<entry>` references resolve only inside that tree, the three leak lints (start /
certification / harvest) and the quarantine scan only that tree, and the write-side prompts register new
recipes in that tree's `index.md`. `kb_arch` comes from `[loop] kb_arch` in `operators/<op>/loop.toml`
(or `forge loop --kb-arch`), default `sm90`; the tree it points to must exist and have an `index.md`,
otherwise the run is refused at start (exit 14). Opening a tree for a new architecture = create the
directory + write an `index.md` following the rules above; no code change needed.

## Fill strategy

The KB is **demand-driven**: in phase 0 the student runs an existing milestone, and whichever interface
it gets stuck on is the entry that gets filled first. Do not enumerate and fill everything up front
(pre-filling = overfitting to one operator's usage). Each architecture's `index.md` keeps its own
to-be-filled list.
