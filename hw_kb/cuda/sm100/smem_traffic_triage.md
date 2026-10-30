# Triaging smem traffic over the line into two types (wavefronts × bankconf four quadrants)

> **Certification source: methodology transcribed from the CuTe DSL mode knowledge base (the same-named entry in `hw_kb/cutedsl/sm100`, obtained on sm90, not re-checked on sm100),
> not re-checked in CUDA mode.** The triage uses only NCU signatures and is independent of the kernel language.

**This is a reading recipe (a diagnosis method), not a mechanism recipe**: it prescribes how to write the causal table of a health criterion,
and the **first triage step** of an audit facing smem traffic over the line, so that a "volume disease" is not given a "conflict cure".

## Why it exists

An observed failure mode: a milestone health criterion had only a single branch, "wavefronts over the line → check the stmatrix
swizzle". The student measured write-side wavefronts ≈ 2× the reference and read-side ≈ 1.8×, but `bankconf_st`
only a few dozenths of the reference's. Tuning the swizzle tier per the prescription gained nothing; the real lesion was in the feed path (an operand
that should have been fed from registers to the MMA took a detour through smem). The single-branch causal table steered the audit's first cut wrong, at the cost of a whole plateau round +
the budget of a structural rebuild.

## Triage matrix

Anchored on the reference's NCU signature for the same shape (`smem_wavefronts_st/ld`, `bankconf_st/ld`):

| wavefronts | bankconf | Type | Pathology | Lever |
|---|---|---|---|---|
| over the line | also over (≥3× the reference) | **conflict-type** | wrong swizzle mode / contiguous dimension, wrong `stmatrix` / `ldmatrix` member (including `.trans`), the same data replayed over and over | swizzle tier and address XOR bits, matrix-atom member; the write side and read side must use the same swizzle |
| over the line | clean (≤ the reference's order of magnitude) | **volume-type** | the **number of byte transfers** itself has grown: scalarized / narrow instructions split up, the same data landed repeatedly, an extra relay (an operand that should be fed straight from registers lands in smem and is read back) | first check whether the operand source was built as decided (in SASS, does the MMA's operand come from registers or an smem descriptor); then check the access granularity (have 128-bit wide instructions degraded into `STS.U16` / `LDS.U16`) and repeated writes |
| within target | over the line | conflict-type (mild) | more replays but normal volume | same levers as conflict-type; lower priority than other over-the-line signals |
| within target | within target | healthy | — | — |

**Read and write sides inflated together in similar proportions** is a strong signal of the "extra relay" within volume-type: one extra trip = one extra write +
one extra read, so both bills grow together; pure write-side degradation (split instructions / repeated writes) blows up only the st side.

## Requirements on the milestone (the template A.5 clause "two branches for smem-traffic health criteria")

Whenever an `smem_wavefronts`/`bankconf` criterion is set, the causal table must give a lever for each of the two branches; missing one makes it
a single-branch causal table — the machine check does not verify this (prose is hard to lint); it is enforced by the distill obligations and by comparison with this recipe.

Other failure modes of the same family: scalar accesses to an exchange face eating most of the issue budget (an extreme case of volume-type); read-side bankconf
a hundred times the reference's while the criterion watched only the write side (the four quadrants must be checked on both sides).

## Sources

- This repo: the triage matrix (NCU signature readings) of the CuTe DSL mode knowledge base's `hw_kb/cutedsl/sm100/smem_traffic_triage.md`.
