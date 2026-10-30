# Two kinds of smem traffic overrun (wavefronts × bankconf quadrants)

> **Certification source: sm90 campaign · not re-checked on sm100.**

**This is a diagnostic recipe (a reading method), not a mechanism recipe**: it prescribes how to
write the cause table of a health criterion, and the **first triage step** an audit takes when
smem traffic is over the line, so that a "volume disease" is not treated with "conflict medicine".

## Why this exists

An observed failure mode: the milestone's health criterion had a single branch, "wavefronts over
the line → check the STSM swizzle". The student measured write-side wavefronts ≈ 2× the reference
and read-side ≈ 1.8×, but `bankconf_st` was only a few percent of the reference's. Tuning the
swizzle level as prescribed gained nothing; the real lesion was in the operand-feed path (an
operand that should go RS made a detour through smem and back, see [[mma_operand_source_verify]]).
The single-branch cause table pointed the audit's first cut the wrong way, at a cost of one full
plateau round plus the budget of one structural rebuild.

## Reading matrix

Anchor on the reference's NCU signature at the same shape (`smem_wavefronts_st/ld`, `bankconf_st/ld`):

| wavefronts | bankconf | Kind | Pathology | Lever |
|---|---|---|---|---|
| over the line | also over the line (≥3× reference) | **conflict** | wrong swizzle family / contiguous-mode extent, wrong store member chosen, the same data replayed again and again | swizzle level, `LayoutEnum`/atom member (see the four coupled sites in [[stsm_transpose_exchange]]) |
| over the line | clean (≤ the reference's order of magnitude) | **volume** | the **number of byte trips** itself grew: scalarisation / wide instructions split into narrow ones, the same data landed more than once, an extra transit hop (the RS ruling row was not built → lands in smem and is read back) | first check whether the `[SOURCE: RS]` row was built (the `mma_a_reg` census, see [[mma_operand_source_verify]]); then check access granularity (whether 128-bit wide instructions degraded) and duplicate writes |
| on target | over the line | conflict (mild) | more replay but normal volume | same levers as conflict; lower priority than other over-the-line signals |
| on target | on target | healthy | — | — |

**Both sides inflated at a similar ratio** is a strong signal of the "extra transit hop" case of
the volume kind: one extra trip = one extra write + one extra read, so both bills grow together; a
pure write-side degradation (split instructions / duplicate writes) only blows up the st side.

## What the milestone must contain (the template A.5 clause "two-branch health criterion for smem traffic")

Whenever a `smem_wavefronts`/`bankconf` criterion is set, its cause table must give a lever for
each of the two branches; missing either one makes it a single-branch cause table. The
machine-read validation does not check this (prose is hard to lint); it is enforced by the distill
obligations against this recipe.

Failure modes of the same family: scalar atoms on an exchange face eating most of the issue budget
(an extreme case of the volume kind); read-side bankconf a hundred times the reference's while the
criterion only watched the write side (the four quadrants must be checked on both sides).
