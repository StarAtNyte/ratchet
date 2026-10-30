# mbarrier_try_wait_fastpath_layout (how ptxas places the retry block of an mbarrier wait: outlined / inline_forward / inline_backedge)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: numerical correctness of six source spellings, the SASS layout class of each wait point (nvcc 13.1 / sm_100a), clock64 readings.
> The layouts were measured **in a single-wait-point kernel**; in larger kernels ptxas decides wait point by wait point, so use the conclusion as "look at your own SASS first".
> This entry is **not a performance lever**: the spellings' fast paths differ by 0–5 cycles; the whole-kernel gain is uncertified.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For which PTX encoding a wait uses see
> [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md); this entry covers only where the retry block goes when the same encoding lands in SASS.

## Purpose

Recognize the three shapes of a wait point when reading SASS / the stall ledger, and know that a "hand-written fast path" is unnecessary under CUDA:

| Layout | SASS shape | Extra on the success path |
|---|---|---|
| **outlined** | the straight-line code has only one `SYNCS.PHASECHK.TRANS64.TRYWAIT P0` + `@!P0 BRA <address after EXIT>`; the retry block sits after EXIT at the end of the function | 0 (one predicated branch that is not taken) |
| **inline_forward** | `TRYWAIT P0` → `@P0 BRA past` → a second `TRYWAIT` → `@!P0 BRA <after EXIT>` | one **taken** branch |
| **inline_backedge** | wait instruction + `@!P0 BRA <itself>`, the back edge right on the straight-line code | 0, but the retry spins in place |

Measured on B200 (nvcc 13.1, one single-wait-point kernel per spelling):

| Source spelling | Layout | Unblocked: start waiting → result (cycles, median) | Blocked: peer arrive → result (cycles, median) |
|---|---|---|---|
| label loop with a suspend hint | outlined (`NANOSLEEP.SYNCS` in the retry block) | 85 | 101 |
| hint-free asm self-loop `L: try_wait p; @!p bra L;` | outlined | 85 | 101 |
| C++ `while (!try_wait_once(...)) {}` | outlined | 86 | 102 |
| hand-written "fast path first, then spin" | **inline_forward** (not merged) | 90 | 106 |
| test_wait self-loop | inline_backedge (plain `PHASECHK`, no `.TRYWAIT`) | 141.5 | 146 (p10 109 / p90 190) |
| two consecutive C++ `while` waits | the first outlined, the second TRYWAIT right after it on the straight line | 130 (both waits together) | 146 |

## When needed (trigger)

- Reading SASS and seeing code blocks with `TRYWAIT` after EXIT at the end of the function, or a timeline tool marking that address range as unattributable:
  that is the outlined retry block, not dead code, and not "the wait got lost".
- Wanting to "hand-write a fast path" for an mbarrier wait to save a few instructions: **do not write it**. ptxas already outlines a plain try_wait loop,
  so the success path is just one untaken branch; a hand-written fast path instead lowers to inline_forward, 5 cycles more.
- Choosing between a test_wait self-loop and a try_wait loop: the try_wait loop gets the result about 55 cycles faster (85 vs 141.5).

## Minimal runnable example

```cpp
// Taken from the certification probe: recommended spelling (ptxas outlines it automatically)
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@!P1 bra LAB_WAIT;\n"
      "}\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(bar))), "r"(parity) : "memory");
}

// Not recommended: a hand-written fast path — measured to lower to inline_forward (one extra taken branch on the success path)
//   "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
//   "@P1 bra DONE;\n"
//   "LAB_WAIT:\n"
//   "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
//   "@!P1 bra LAB_WAIT;\n"
//   "DONE:\n"

// Checking the layout: cuobjdump -sass <module>.so, find the first predicated BRA within 6 instructions after SYNCS.PHASECHK,
// and see whether its target is after the last EXIT (outlined), before the wait point (inline_backedge), or in between (inline_forward).
```

## Gotchas

1. **The layout is decided per wait point; there is no source switch.** The same PTX loop may be laid out differently in different kernels / positions;
   rewriting the source to make the layout "look nice" is not an optimization goal — the layouts' fast paths differ by only 0–5 cycles.
2. **A hinted loop is outlined under CUDA too**: `NANOSLEEP.SYNCS` is in the retry block after EXIT, and the straight-line code has
   only one TRYWAIT. So "is there a NANOSLEEP in the straight-line code" cannot identify the encoding; count together with the function tail.
3. **Read the end timestamp under a predicate that depends on the result** (`t1 = (v == SENTINEL) ? 0 : clock64()`), otherwise ptxas
   can hoist the `CS2R SR_CLOCKLO` above the wait and what you measure is 0. In the probe the SASS window shows this CS2R really is
   predicated with `@P0`.
4. **Isolate labels inside asm with a `{ }` scope**: no name clash when the same wait function is inlined several times (do not rely on `%=`).
5. **Use a register operand for parity**, otherwise there is one asm per phase and the number of wait points doubles.
6. **A numeric probe cannot see the layout**: the six spellings read back bitwise identical values; the only criteria are the SASS classification + clock64.

## Sources

- PTX ISA: *mbarrier.try_wait / mbarrier.test_wait*, https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities: `cuobjdump -sass`, https://docs.nvidia.com/cuda/cuda-binary-utilities/
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 mbarrier_try_wait_fastpath_layout/L1-values-all-forms`,
  `PROBE PASS max_diff=n/a mbarrier_try_wait_fastpath_layout/L2-every-site-classified`,
  `PROBE PASS max_diff=n/a mbarrier_try_wait_fastpath_layout/L3-nanosleep-only-hinted nanosleep=1/0/0/0/0/0`.
