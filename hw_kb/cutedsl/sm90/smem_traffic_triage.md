# Two-type triage of an smem-traffic breach (wavefronts × bankconf, four quadrants)

**This is a triage recipe (a diagnosis method), not a mechanism recipe**: it prescribes how to
write the cause table of a health criterion, and the **first triage step** an audit takes when smem
traffic is over the line, so a "volume disease" is not given the "conflict medicine".

## Why it exists

An observed failure mode: the milestone's health criterion only wrote a single branch,
"wavefronts over the line → check the STSM swizzle". The student measured write-side wavefronts ≈
2× the reference and read-side ≈ 1.8×, yet `bankconf_st` was tens of times lower than the
reference's. Retuning the swizzle level as prescribed gained nothing; the real lesion was in the
operand-feed path (an operand that should go RS took a round trip through smem, see
[[mma_operand_source_verify]]). A single-branch cause table steers the audit's first cut in the
wrong direction, at the cost of a whole plateau round plus the budget of one structural rebuild.

## Triage matrix

Anchor on the reference's NCU signature at the same shape (`smem_wavefronts_st/ld`, `bankconf_st/ld`):

| wavefronts | bankconf | Type | Pathology | Lever |
|---|---|---|---|---|
| over the line | over the line too (≥3× reference) | **conflict type** | wrong swizzle family / wrong contiguous-mode extent, wrong store member; the same data is replayed over and over | swizzle level, `LayoutEnum`/atom member (see the four coupled sites in [[stsm_transpose_exchange]]) |
| over the line | clean (≤ the reference's order of magnitude) | **volume type** | the number of **byte movements** itself grew: scalarisation / narrow instructions split into many, the same data landing more than once, an extra transit hop (the RS ruling row was not built → land in smem and read back) | first check whether the `[SOURCE: RS]` row was built (`mma_a_reg` census, see [[mma_operand_source_verify]]); then check access granularity (did 128-bit wide instructions degrade?) and duplicate writes |
| on target | over the line | conflict type (mild) | more replays but normal volume | same levers as the conflict type; lower priority than the other over-the-line signals |
| on target | on target | healthy | — | — |

**Both read and write sides inflated by a similar ratio** is a strong signal for the "extra transit
hop" variant of the volume type: one extra hop = one extra write + one extra read, so both bills grow
together; a pure write-side degradation (split instructions / duplicate writes) only blows up the st side.

## Requirement on the milestone (template A.5 clause "two-branch smem-traffic health criterion")

Whenever an `smem_wavefronts`/`bankconf`-type criterion is set, its cause table must give a lever for
each of the two branches; missing either one makes it a single-branch cause table. The machine
validator does not check this (prose is hard to lint); it is enforced through the distill obligation
and by checking against this recipe.

Other failure modes of the same family: a scalar atom on the exchange face eating most of the issue
budget (an extreme case of the volume type); read-side bankconf a hundred times the reference's while
the criterion only watched the write side (the four quadrants must be checked on both sides).
