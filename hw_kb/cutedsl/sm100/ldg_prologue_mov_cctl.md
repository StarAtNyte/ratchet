# ldg_prologue_mov_cctl (register-operand prologue: where the MOVs before the first HMMA come from, and where the CCTL prefetches land)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe, not included in this release; `PROBE PASS max_diff=3.13e-05`, round 2 of this certification; all four contracts pass).
> **What is certified**: only the SASS shape (number of zero-fill writes, number of early copies, where the LDG / CCTL land) and the numerics; **no gain is certified**; the whole-kernel readings are under "When needed".

## Purpose

In the prologue of "global → register → `mma.sync`" (from the kernel entry to the first `HMMA`, called segment P below),
besides `LDG` and address arithmetic, SASS shows two kinds of MOV:

1. **Zero-fill writes**: the "`mov.b32 r, 0` first, then `@p ld`" of a predicated load -- every 256-bit predicated load costs 8 register writes
   (`MOV R, RZ` / `CS2R R, SRZ`), all scheduled before that LDG, occupying issue slots of segment P.
2. **Fragment-assembly copies**: the destination of `LDG.*.256` is 8 consecutive aligned registers; the A operand of
   `mma.sync.m16n8k16` needs 4 consecutive aligned registers in SASS. If the 4 registers of an A quadruple come from **two different loads** (2 each),
   ptxas can only copy them into a new quadruple with `MOV` / `IMAD.MOV.U32 R, RZ, RZ, Rsrc`.
   The source of the copy is a load result, so it is **the first to eat the long scoreboard**; if it is scheduled before
   the last LDG / CCTL of segment P, in-order issue stalls right there and every later load and prefetch is pushed back.

This entry answers: how to remove the zero-fill writes wholesale; and which spellings make the position of the copies and of the CCTLs visible and controllable in SASS.

## When needed (trigger)

- Segment P's stalls are dominated by `long_scoreboard` / `lg_throttle`, and the top stall rows are `IMAD.MOV` / `MOV` in segment P
  (not HMMA).
- Segment P has more MOVs than the "structural floor of fragment assembly" by a whole multiple of 8 (= number of predicated loads × 8).
- Prefetches (`prefetch.global.L2*` → `CCTL`) trail all the LDGs as a string, and you want to interleave them between the LDGs.
- **Whole-kernel readings** (the whole kernel where this entry was found): clamping per spelling 1 + moving the prefetches between the load groups, the
  same-job whole-kernel probe showed the warm-L2 geometric mean 6.134 → 5.591 µs (−8.9%); spreading the prefetches further gave cold-L2 8.431 → 7.933 µs. But five consecutive evaluations
  gave gate 0.0080–0.0083 ms, not beating the 0.0079 ms from before the change, and the patch was withdrawn for no gain on the same-job ratio. **The spelling is certified; no whole-kernel gate gain was seen**; price it same-job on your own kernel before using it.

## Minimal runnable example

Certified readings (6 variants of the same probe skeleton: two groups of `LDG.E.ENL2.256.CONSTANT` in segment P (some of them predicated loads); each variant against the fp64 reference `max_diff ≤ 3.13e-05`, padding columns not written; STL = LDL = 0):

| Variant | Zero-fill writes in segment P | Early copies | LDG position in segment P | CCTL position in segment P |
|---|---|---|---|---|
| `zf` (predicate + zero-fill, prefetch after the groups) | yes (8 per predicated load) | **yes** | two stretches | end of group |
| `clamp` (clamped, no predicate) | **0** | 0 | one contiguous stretch | — |
| `clamp_pfblk` (clamped + prefetch in its own guarded block) | 0 | 0 | one contiguous stretch | **none in segment P** (sunk past the first HMMA) |
| `clamp_pfin` (clamped + prefetch written inside the group) | 0 | 0 | spread out | **all after the last LDG** |
| `zf_pfin` (zero-fill + prefetch written inside the group) | yes (8 per predicated load) | 0 | two stretches | sandwiched between the two groups |
| `clamp_pfin_p` (clamped + a predicate that is always true at run time, no zero-fill + prefetch inside the group) | 0 | 0 | one contiguous stretch | **all before the first LDG** |

Key points: clamping removes the zero-fill writes wholesale (→ 0), and the early copies also go to 0 (the copies sink by themselves to after the last LDG,
right against the first HMMA); the prefetch position is decided by the other instructions in the group, and the three in-group spellings give three different landing spots.

Below are the spellings within the certified scope.

**Spelling 1: replace zero-fill with an address clamp** -- valid only when "the columns / rows computed from out-of-bounds rows are never written back" (an out-of-bounds row affects only its own part of the output,
and that part is blocked by the store guard). Out-of-bounds lanes read a legal row instead (clamped to M−1), and the load carries no predicate:

```python
r    = my_row                           # the row this lane's load reads
rc   = r - I32(r >= M) * (r - (M - 1))  # pure-arithmetic clamp to M−1 (no branch)
addr = base + I64(rc) * ROW_BYTES + kofs
# inside the group write a plain "ld.global.nc.v8.b32 {...}, [$a+off];", with no more mov.b32 $i, 0 and @p
# the store side keeps the `if m < M:` guard -- this is the precondition for the clamp to be correct
```

**Spelling 2: write the prefetch inside the load group** -- **not guaranteed** to sit between LDGs in SASS (see gotcha 4); once written you must look at the SASS (one inline asm, wrapped in a real run-time guard branch, same as
[`ldg_issue_pin_structural_fence`](ldg_issue_pin_structural_fence.md) spelling B):

```python
# body:
#   { .reg .pred q; setp.ne.b32 q, $G, 0;
#     @!q bra SKIP${:uid};
#     ld.global.nc.v8.b32 {$0..$7},  [$X+0];
#     ld.global.nc.v8.b32 {$8..$15}, [$X+OFF1];
#     prefetch.global.L2 [$PF+0];          # inserted between LDGs
#     ld.global.nc.v8.b32 {...}, [$Y+...];
#     ...
#   SKIP${:uid}: }
```

**Spelling 3: the prefetch as its own guarded block**, placed after all load groups:
`{ .reg .pred q; setp.ne.b32 q, $G, 0; @!q bra PF${:uid}; prefetch.global.L2 [$0]; ... PF${:uid}: }`,
`has_side_effects=True`.

### Acceptance (SASS; the numerics cannot show it)

`CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>`, clear the dump directory before compiling each variant → `cuobjdump -sass`,
take the instructions before the first `HMMA` as segment P, and count:

- Zero-fill writes: `MOV R, RZ` / `MOV R, 0x0` / `IMAD.MOV.U32 R, RZ, RZ, RZ` count 1 each, `CS2R R, SRZ` counts 2;
- LDG result copies: `MOV` / `IMAD.MOV` whose source register falls inside some `LDG.*.256 Rn` of segment P (covering Rn..Rn+7);
  those scheduled **before** the last LDG / CCTL of segment P count as "early copies";
- whether `CCTL` sits between LDGs (look per variant, see gotcha 4); STL / LDL must be 0.

## Gotchas

1. **The clamp is correct only when the out-of-bounds columns are discarded**. An out-of-bounds lane reads real data and multiplies out nonzero garbage; once the store-side
   `m < M` guard is removed, wrong values are written. When out-of-bounds rows must take part in the computation as **true zeros** (e.g. out of bounds along the K reduction direction),
   the clamp cannot be used.
2. **Zero-fill writes cannot be moved after the load**: `mov 0` and `@p ld` write the same group of registers (WAW), so ptxas can only schedule them before the LDG.
3. **Fragment-assembly copies are a structural floor, not something you can delete**: when an A quadruple spans two loads, every k-step must copy;
   all you can control is how close it is to the first HMMA.
4. **PTX source order does not pin the CCTL position**: even with `prefetch.global.L2` written between two
   `ld.global.nc.v8.b32` in the same asm, ptxas may still schedule every `CCTL` after the last `LDG` (it schedules freely between loads and prefetches
   that have no data dependence). Measured: the same spelling keeps the prefetch between LDGs in a "load group with predicated zero-fill",
   and in a "clamped, unpredicated" load group the prefetches sink wholesale to the end of the group -- whether they get interleaved depends on the other instructions in the group, not on
   where you write them. When relying on "prefetches inserted between LDGs relieve `lg_throttle`", count the CCTL positions with the acceptance script above after every change;
   do not assume source order = SASS order. Measured (see the table above): prefetch inside a clamped group → all after the LDGs;
   prefetch inside a zero-fill group → sandwiched between the two LDG groups; clamp + always-true predicate → all hoisted **before** the first LDG.
5. **A prefetch in its own guarded block may leave segment P entirely**: in `clamp_pfblk` the CCTLs are scheduled after the first HMMA -- that is good for
   "do not block the HMMA", but if your intent was to issue the prefetch as early as possible to cover the next stage, it has actually been delayed.
6. **The early copies come from the zero-fill group, not from the copy itself**: under the same fragment assembly, `zf` has early copies and every clamped variant has 0 --
   the zero-fill writes and predicated loads lengthen the LDG stretch, and ptxas interleaves some copies into it; once the zero-fill is gone you no longer need to move copies by hand.
7. Judging "CCTL sits between LDGs" in the acceptance check by `min(CCTL) < max(LDG)` also counts "all before the LDGs" as true;
   to tell the two apart, print the position lists directly.

## Sources

- PTX ISA: the vector form of `ld.global.nc` (`.v8.b32`), predicated execution, `prefetch.global.L2`, the operand fragment layout of
  `mma.sync.aligned.m16n8k16`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- API symbols of public infra: `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`.
- Certification probe: `PROBE PASS max_diff=3.13e-05`.
