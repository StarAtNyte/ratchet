# CUDA hardware knowledge base (hw_kb/cuda)

ForgeKernel's **CUDA mode** (loop language `cuda-sol`: the student writes a single-file CUDA C++ `kernel.cu`,
the operator tree compiles it with nvcc + the CUTLASS headers) has its own hardware knowledge base. It is **fully separate** from the CuTe DSL mode's `hw_kb/cutedsl/`:
CUDA runs read and write only this tree, DSL runs read and write only `hw_kb/cutedsl/` (`src/forgekernel/loop/lang_mode.py`).
The mechanism knowledge on both sides can be the same, the spelling differs — DSL entries teach `cute.*` calls, these teach CUDA C++ / CUTLASS (CuTe C++) /
inline PTX — so the two do not share a tree.

## Invariants (same as `hw_kb/cutedsl/`)

- **Hardware mechanisms only, no operator algorithms.** "How to issue a `tcgen05.mma`" ✅; "how this operator's 5 GEMMs are wired" ❌.
- Every entry is self-contained, minimal and runnable, and carries its own sources. Example code comes from **verifiable sources** (PTX ISA / CUDA Programming Guide /
  public CUTLASS headers / this repo's GPU-certification probes); nothing is made up.
- **The evidence chain is split in two layers by reader**: the student reads an entry's "Sources", which may cite only official docs / public infra API symbols /
  this repo's probes; the reference implementation's file names, function names, line numbers, paraphrases of its content and measured footprints all go to
  `docs/kb-provenance/cuda/<entry name>.md` (for human review, not in the agent read set). The mechanical gate is the same as for `hw_kb/cutedsl/`:
  `src/forgekernel/loop/kb_lint.py`.
- **Only GPU-certified material enters**: a probe returns `PROBE PASS`, or (for numerically invisible knobs) a KEEP's own
  NCU dossier census shows the contract axis moved. Entries of the DSL tree are **not copied over as they are** — their readings were taken on the DSL spelling.
  For the same mechanism to enter here, the example must be rewritten in CUDA and the readings re-taken on a CUDA probe (G3 certification in a CUDA run,
  or a dedicated CUDA probe job); the DSL side's readings and conclusions are only hypotheses to be verified and must not be carried over.

## Layout

```
hw_kb/cuda/
  TEMPLATE.md        # entry template (example code is CUDA C++ / inline PTX)
  sm100/
    index.md         # SM100 (Blackwell) list; the entry point is the "Look up by symptom first" table
    <interface>.md
```

**One tree per architecture.** `kb_arch` comes from `[loop] kb_arch` in `operators/<op>/loop.toml` (or
`forge loop --kb-arch`); CUDA mode resolves it under `hw_kb/cuda/<kb_arch>/`, and the tree must exist and have an `index.md`,
otherwise the run is refused at start (exit 14). Opening a tree for a new architecture = create the directory + write `index.md`.

## Initial contents

- **Methodology** (how to build a GPU probe, how to triage smem traffic): transcribed from `hw_kb/cutedsl/sm100/`, each with a `Certification source` line.
- **Seed mechanism entries**: translated mechanism by mechanism from `hw_kb/cutedsl/sm100/` into CUDA C++ / inline PTX, each re-certified on B200 with a CUDA probe
  before entering. Entries that do not hold under CUDA, bring no gain, or are specific to the DSL front end were not taken.
- Later entries grow one by one from G3 certification in CUDA runs and from the finish-time harvest.

## Filling strategy

Demand-driven: whichever interface the student gets stuck on gets its entry first (`meta/cuda/kb_recipe_prompt.md`); expensive probe wins
and KEEPs that changed a contract axis are written up as recipes in the finish round (`meta/cuda/kb_harvest_prompt.md`). No up-front enumeration.
