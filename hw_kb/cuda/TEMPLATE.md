# <interface name> (e.g. tcgen05.mma / TMA load / mbarrier)

> One file per hardware interface. Operator-agnostic: mechanism only. CUDA mode: examples are CUDA C++ / CUTLASS (CuTe C++) / inline PTX.

## Purpose
What hardware-level problem this interface solves (one sentence).

## When needed (trigger)
Which student need / symptom calls for it (profiler symptom / SASS footprint / dataflow shape / compiler error text).

## Minimal runnable example
```cpp
// Minimal, self-contained example. From a verifiable source; nothing made up.
// Show only "how to use this interface"; do not embed any specific operator's algorithm.
// Write inline PTX with the full constraint string and modifiers; put required compile flags on a `// fk-cuda-cflags:` line.
```

## Gotchas
- What error a misuse raises / where it goes silently wrong (e.g. alignment requirements, descriptor fields, phase counting, memory order and fences, drain timing).

## Sources

**The student reads this section; only three kinds of citation are allowed** (a tool's symbols may be cited, a reference solution's may not):

1. Official documentation links (PTX ISA / CUDA C++ Programming Guide / official answers on the NVIDIA forums);
2. **API symbols** of public infra (CUDA runtime / driver API, the interface surface of the public CUTLASS and CuTe C++ headers,
   i.e. what the student's code has to call anyway);
3. Paths and verdicts of this repo's GPU-certification probes (`PROBE PASS ...`).

**Forbidden**: the reference implementation's (the oracle's, open or closed source) file names, function names, line numbers, paraphrases of its content,
measured footprints, and the words "oracle / teacher / reference implementation" — all such narrated provenance goes to
`docs/kb-provenance/cuda/<this entry's name>.md` (for human review, in no agent read set); the entry keeps only one line:

```
- Detailed provenance: docs/kb-provenance/cuda/<entry>.md (for human review, not in the agent read set).
```

The mechanical gate `src/forgekernel/loop/kb_lint.py` scans for banned tokens at three points: loop start, recipe
certification into the KB, and publish. A hit at certification is bounced back for revision first and the entry is quarantined if it still fails; a hit at harvest refuses the publish.
