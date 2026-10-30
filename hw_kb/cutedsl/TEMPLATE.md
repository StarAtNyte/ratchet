# <interface name> (e.g. WGMMA / TMA load / mbarrier)

> One file per hardware interface. Operator-agnostic; mechanism only.

## Purpose
What hardware-level problem this interface solves (one sentence).

## When needed (trigger)
Which student need / symptom calls for it (profiler symptom / dataflow shape).

## Minimal runnable example
```python
# Minimal, self-explanatory example. From a verifiable source; nothing invented.
# Shows only "how to use this interface"; embeds no specific operator's algorithm.
```

## Gotchas
- What error a misuse raises / where it goes silently wrong (e.g. alignment requirements, phase
  counting, drain timing).

## Sources

**The student reads this section; only three kinds of citation are allowed** (a tool's symbols may be
cited, the reference solution's may not):

1. Official documentation links (PTX ISA / CUDA docs / official NVIDIA forum answers);
2. **API symbols** of public infra (the interface surface of public modules such as cutlass / CuTe DSL /
   quack, i.e. what the student has to call when writing code anyway);
3. The path and verdict (`PROBE PASS ...`) of a GPU-certified probe in this repo.

**Must not appear**: the reference implementation's (oracle's, whether open- or closed-source) file
names, function names, line numbers, paraphrased content or measured footprints, nor the words
"oracle / teacher / reference implementation" — all such narrative evidence chains go to
`docs/kb-provenance/<this entry name>.md` (for human review; never in any agent read set), and the entry
keeps only one line:

```
- Detailed provenance archive: docs/kb-provenance/<entry>.md (for human review; not in any agent read set).
```

The mechanical gate `src/forgekernel/loop/kb_lint.py` scans for banned tokens at three points — loop
start, recipe certification into the KB, and publish: on a hit at certification the entry is first
bounced back for revision and quarantined if it still fails; a hit at harvest refuses the publish.
