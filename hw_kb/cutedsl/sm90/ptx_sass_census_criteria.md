# ptx_sass_census_criteria (machine-readable footprint criteria: writing a PTX/SASS census that does not pass off an instrument fault as a kernel defect)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is a **reading recipe**: it does not
> teach how to use any mechanism, only how to write the criterion itself that "proves a mechanism really landed in
> the instructions".

## Purpose

Many SM90 mechanisms (async copies, barriers, packed conversions, matrix copy atoms, register reallocation) are
**numerically invisible** — used wrongly they may still compute right, just slowly or in a degraded way. The only
criterion is the issue footprint: count the opcodes in the dump. The dangerous failure mode of such criteria is not a
miss, but **judging in the wrong direction**:

> `count == 0` can mean "the mechanism is missing", or "the census counted wrong".
> In a criterion `assert count >= N`, these two look **exactly the same**.

And their costs are completely asymmetric: the former requires changing the kernel, the latter changing the probe.
Guessing the wrong direction = burning a GPU round for nothing, and the next round will likely keep changing the kernel
in the wrong direction. This entry writes "making the two distinguishable" as a construction you can copy.

## When needed (trigger)

- When writing a certification contract for any mechanism that is "invisible at trace time, known to work only once
  issued";
- **Symptom (the main trigger of this entry)**: a PTX/SASS count contract FAILs **on its own**, while the numerical
  contracts of the same round (bijection / whole-kernel correctness) PASS. The numerical contract goes through exactly
  the path under suspicion ⇒ the path works ⇒ **treat it as an instrument suspect first; do not touch the kernel**.

## Minimal runnable example

### 1) Histogram first: print the facts before talking about the criterion

Before the criterion, unconditionally print **every** opcode of the family with its count. An unexpected spelling is
thereby **visible** instead of silently counting as zero.

```python
import re

_FAMILY = re.compile(r"\bcvt\.[A-Za-z0-9_.]+")     # replace with the stem of the family you are checking

def hist(blob):
    """Each distinct opcode of this family in the dump -> count."""
    h = {}
    for op in _FAMILY.findall(blob):
        h[op] = h.get(op, 0) + 1
    return h

def fmt(h, cap=16):
    items = sorted(h.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
    return " ".join("%s=%d" % kv for kv in items) or "<none>"

note("cvt opcodes: %s" % fmt(hist(blob)))          # a channel for facts, not a verdict
```

### 2) Classify by "family + qualifiers", not by literal mnemonic; count the failure form separately

```python
def n_packed(h):        # target form: any cvt with bf16x2, whatever the modifiers drift to
    return sum(c for op, c in h.items() if "bf16x2" in op)

def n_scalar(h):        # failure form: per-element scalar conversion. Counted separately, never merged into the above
    return sum(c for op, c in h.items()
               if "bf16x2" not in op and re.search(r"\.bf16\.f32$", op))
```

The same approach on the barrier family is "classify by qualifiers" rather than text matching:

```python
# the leading \b excludes `mbarrier` exactly (m and b are both word chars, no boundary inside "mbarrier")
OPC_ANY = re.compile(r"\b(?:barrier|bar)((?:\.[A-Za-z0-9_:]+)*)")

def classify(q):
    """qualifier string -> 'sync' (blocking) | 'arrive' (one-way) | None (not a CTA rendezvous)."""
    if ".warp" in q or ".cluster" in q or ".red" in q:
        return None                       # warp mask / cluster scope: operands mean something else
    if ".arrive" in q:
        return None if ".sync" in q else "arrive"
    return "sync" if ".sync" in q else None

inv = {}
for m in OPC_ANY.finditer(blob):
    inv[m.group(0)] = inv.get(m.group(0), 0) + 1
note("barrier-family inventory: "
     + (", ".join("%s x%d" % kv for kv in sorted(inv.items())) or "(none)"))
```

### 3) Operands may be registers: resolve through `mov` **per file**

```python
OPC_SITE = re.compile(
    # up to `;` if there is one, otherwise to end of line -- the dump of an inline-asm body may have no terminator
    r"\b(?:barrier|bar)((?:\.[A-Za-z0-9_:]+)*)[ \t]+([^;{}\n]*)")

sync_ids, arrive_ids = [], []
for path in ptx_files:
    txt = open(path, "r", errors="replace").read()
    # build the table per FILE, never merged across dumps: virtual register names are module-local, a global table would cross wires
    movs = {}
    for r, v in re.findall(r"mov\.[bsu]32\s+(%[A-Za-z0-9_]+)\s*,\s*"
                           r"(-?\d+|0[xX][0-9a-fA-F]+)\s*;", txt):
        iv = int(v, 0)
        movs[r] = None if (r in movs and movs[r] != iv) else iv   # give up on conflict
    for q, ops in OPC_SITE.findall(txt):
        kind = classify(q)
        if kind is None:
            continue
        first = ops.split(",")[0].strip().rstrip(";")
        bid = (int(first, 0) if re.fullmatch(r"-?\d+|0[xX][0-9a-fA-F]+", first)
               else movs.get(first))                              # None = cannot be resolved
        (sync_ids if kind == "sync" else arrive_ids).append(bid)
```

### 4) SASS cross-check: shares none of PTX's spelling assumptions

```python
for op, a in re.findall(r"\bBAR\.(SYNC|ARV|RED)[A-Z_.0-9]*\s+([^;,\n]+)", sass):
    a = a.strip()
    if not re.fullmatch(r"0[xX][0-9a-fA-F]+|\d+", a):
        continue                          # register-form id: cannot be attributed; skip rather than count as wrong
    ...
```

### 5) Layered criterion strength + print a basis line

Compute all three criteria, take the **OR**, and print "which one carried the verdict":

```python
ok_ptx  = pp_sync  >= 2 and pp_arrive  >= 2      # strong: the one-way sites really are on the target id
ok_sass = sass_seen and sass_sync >= 2 and sass_arrive >= 2   # strong: same conclusion on the SASS side
ok_mech = inv_arrive >= 1 and inv_sync >= 2      # weak: opcode-level tally, does not touch operands
note("verdict basis: ptx-ids=%s sass-ids=%s mechanism=%s "
     "(opcode tally blocking=%d one-way=%d)"
     % (ok_ptx, ok_sass, ok_mech, inv_sync, inv_arrive))
verdict("asymmetric-baton", ok_ptx or ok_sass or ok_mech)
```

**Legitimacy condition of the weak criterion (if it does not hold, it is letting things slide)**: it must still reject
the thing it exists to reject — in this example the weak criterion must reject a "symmetric rendezvous", and a
symmetric rendezvous **issues only blocking barriers and has no one-way arrive to issue**, so `inv_arrive == 0` must
fail. Degrading a one-way arrive into an arrive-and-wait (exactly the regression the check is meant to catch) also
drives the one-way count to 0. When designing a weak criterion, write the counter-example first and verify that it
really is rejected.

### 6) The forensic fallback for a zero count: compile that helper on its own

```python
if n_packed(hist(blob)) == 0:
    # the main dump has the target form under no spelling at all, while the numerical contract says the values are right
    # -> compile the helper alone (compile-only, never issued, so it cannot hang or fault),
    #    and pin down in a clean environment what it actually emits. Runs only when the main count is 0; healthy rounds pay nothing.
    micro_hist = compile_alone(helper)
    note("pack_alone opcodes: %s" % fmt(micro_hist))
```

## Gotchas

1. **Literal mnemonic matching is a criterion that is "correct once".** The toolchain changes spellings, and when it
   does you get no notice — you get a 0. Two independent pieces of evidence:
   - barriers: the criterion was written as `bar|barrier` followed by `.sync`/`.arrive`, while current NVPTX emits the
     ISA-8.x `barrier.cta.{sync,arrive}.aligned` form; the census saw zero sites and reported "mechanism missing";
   - packed conversion: the criterion hard-coded the single literal `cvt.rn.bf16x2.f32`, while what was emitted was
     `cvt.rn.satfinite.bf16x2.f32`; count 0, again reported as "mechanism missing".
   Both times the numerical contracts of the same round PASSED. **The fix is always "family + qualifier
   classification"**, so drift in modifiers (`.satfinite` / `.relu` / `.aligned` / `.cta`) does not affect the
   classification.
2. **`count == 0`: suspect the instrument first, not the kernel.** Next to the criterion, the same round must print
   the full opcode histogram (numeric families) or the raw matching lines (site families) as **facts**. Without it,
   every round's zero count means guessing the operand form all over again; with it, the operand form is in the log
   after the first round.
3. **Count the target form and the failure form separately; never merge the families.** A per-element scalar
   conversion is not a synonym of a packed conversion; it is **the reason this criterion exists** — merging it into
   the same count amounts to judging the failure mode as a pass.
4. **PTX operands may be registers rather than immediates** (barrier ids especially). To attribute them you must build
   a per-file `mov.u32 %r, imm` lookup table; **merging across dumps crosses wires** (virtual register names are
   module-local), and on a same-name conflict the entry should be set to `None` (give up the attribution) rather than
   taking the last assignment.
5. **Do not assume statements have a terminator.** In some dumps the inline-asm body has no `;`. Operand parsing must
   be "up to `;` if there is one, otherwise to end of line"; a spelling that only accepts `;` zeroes the whole site
   table — and that zero is indistinguishable from the zero of "mechanism missing" (one concrete cause of gotcha 2).
6. **`\b(?:barrier|bar)` happens not to match `mbarrier`** (m and b are both word chars; no boundary inside the word).
   When counting CTA barriers that is what you want; **when counting mbarriers this regex gives you 0 all the way**.
7. **A strong criterion may be unreachable; design the degradation path in advance.** Barrier id attribution can fail
   on both the PTX and the SASS side (`ptx-ids=False sass-ids=False`) while the mechanism itself is fine. The approach
   is the three-layer OR + basis line of §5: **a weak criterion never impersonates a strong one**, and the log states
   who carried the verdict this round. Without a basis line, a PASS cannot answer "was the strong property proved, or
   did it fall back to the weak one".
8. **The value of the SASS cross-check is that it shares no spelling assumption with PTX**, and ids are mostly
   carried as immediates; but register forms exist too — when you meet one, **skip the site**; do not count "cannot be
   resolved" as "not on the target id".
9. **Zero-count forensics must be compile-only.** Compiling that helper alone is the shortest way to pin down the
   spelling in a clean environment, and a compile-only contract **cannot hang or fault** — it can be put
   unconditionally and safely in the failure branch. Made the default, every round pays the compile time for nothing.
10. **A separate dump directory per variant, emptied at the start of every round.** Leftover dumps mix the previous
    round's footprint into this round's histogram, producing a wrong direction that is harder to spot than "zeroed".
    Same as the two-variant isolation discipline of [`uniform_reg_budget`](uniform_reg_budget.md) and
    [`wgmma_desc_uniform`](wgmma_desc_uniform.md).
11. **Axes counted per loop region (`commit` / `supply`) presuppose "a back edge exists": on a fully unrolled kernel
    they are always `[]`, and any change that introduces a back edge makes footprints that were already in the
    instruction stream "appear out of nowhere".** Regions are spanned by SASS back edges; `DEPBAR` / `LDGDEPBAR` /
    address-source `LDS` in straight-line code belong to no region. So when `loops` is `{}`, these two axes reading
    `[]` only means nobody counted; when within one KEEP `loops` flips from `{}` to non-empty (a rolled runtime loop, a
    hand-written `while` spin wait), `commit` / `supply` become whole rows, the diff reports them as changed axes, and
    even when the change touched only one role the other role's rows appear too (region partitioning is a property
    after ptxas scheduling). How to read it: on a KEEP where `loops` flips, record the "appearance" of these two axes
    as codegen moving along, compare against the previous tree's raw SASS, and if the multisets are the same it is not
    a depth change; only when the DSL change really touched `commit_group` / `wait_group` / mbarrier arrive is an
    **increase or decrease** in the `commit` rows a footprint.
    `cache_ops` is counted over the whole kernel and does not depend on regions; on a fully unrolled tree only it can
    see the cache bits / width / completion mode of cp.async — use it as the anchor.
12. **The `rings` axis records "present/absent" as a footprint, but** what it measures is the self-increment
    fingerprint of the mbarrier address def-chain — **change the register allocation and the address chain may detour
    through another set of registers, so the self-increment is unchanged but the ring disappears from the axis**.
    Each `rings` entry is a ring fingerprint traced by `orch_tools.resolve_addr`
    (`stride=0x…,slots=…` / `wrap=sel` etc.; `_ring_key` drops the base and keeps only stride/slots/wrap — a
    different smem offset with the same structure must diff clean). The premise of the RING tier is "the address chain
    passes through an **in-loop self-increment**" (`IADD3` adding to itself + `LOP3` and-mask or `SEL` wrap), and the
    intermediate registers of that chain are chosen by the **allocator**. With the same DSL source, changing the
    `setmaxnreg` tiers once (moving 16 registers from one role to another) can perfectly well make the allocator
    rehome the barrier address base to another register, so the two-step walk of the def-chain changes, the RING
    decision cannot read it, and the `rings` axis goes from `["stride=0x1,wrap=sel"]` to `[]` — **while the ring
    structure (number of slots, increment stride, wrap mode) has not moved a finger**. How to read it: the
    **appearance/disappearance** of `rings` is codegen moving along (just as `loops` flipping moves along with back
    edges), not a mechanism footprint; to really say "the ring structure changed", check that the ring's mbarrier
    objects are **still** in `mbar_objs / mbar_tiers / mbar_per_role` (the count axes move too, and `mbar_per_role` is
    "how many mbarrier objects a role gets" — it drifts with register allocation, not with the ring itself). **When
    within the same KEEP `rings` disappears while `mbar_objs / mbar_per_role` change in step**, you can almost conclude
    it is the allocator moving along, not a ring-structure change; only a change in the **values** of `rings` (stride /
    slots values, or from empty to non-empty, with the count axes unmoved) counts as a footprint. Observed form: a pure
    register-bill change of the `setmaxnreg` tiers of **two consumer roles**, 152 → 136 and 160 → 176
    (producer and the other role unchanged, total 512), took `rings` from
    `["stride=0x1,wrap=sel"]` to `[]`, while `mbar_objs / mbar_tiers / mbar_per_role /
    loops / edges` all moved (all moving along) — no ring change at all in the register bill, yet the gate went −3.97%.
13. **Besides the "blindness on fully unrolled trees" of the `commit` / `supply` axes and the "register drift" of
    gotcha 12, there is a third point — `cache_ops` is the only axis independent of regions and of register allocation;
    use it as the **anchor** of mechanism footprints.**
    When judging "did this change move the mechanism or is it just codegen moving along", look at the three axes
    separately: `cache_ops` moved = mechanism confirmed (it is counted over the whole kernel, with no address
    def-chain); `commit` moved = possibly a back edge introduced (gotcha 11) or a serialisation trap; `rings` moved =
    possibly register drift (gotcha 12) or a real ring change. When all three axes move but `cache_ops` does not,
    first treat it as "the structure did not change; the statistics surface did"; only a move in `cache_ops` (or in the
    values of `rings`) is worth opening a probe for.

14. **`cache_ops` not moving ≠ the structure did not change: it anchors only copy instructions, not sync / scheduling
    structure.** The reading of gotcha 13 (three axes moved, `cache_ops` did not → first treat it as "the statistics
    surface changed") has a premise — that the candidate change is a copy mechanism. Turning a one-CTA-one-tile kernel
    into a persistent one (grid 1024 → 132, one more real back edge per role, ring index / phase derived from a running
    count instead), then widening the per-tile operand to two slots (mbarrier table 10 → 12, smem +32 KiB): in both
    steps `cache_ops` stays `UTMALDG.3D / UTMALDG.4D / UTMASTG.4D`, while the structure is changed beyond recognition.
    The readings of these two steps on `rings`: `["stride=0x1,wrap=sel"]` → `[]` (in the same round `loops`
    `mma-consumer` 7 → 4 + `scalar` 1, `mbar_objs` 17 → 9, `mbar_tiers` `{CONST:16, RING:1}` →
    `{CONST:9}`, `nbar_widths` lost `[288]`), then `[]` → `["stride=0x1", "stride=0x1,slots=9",
    "stride=0xff8"]` (in the same round `mbar_objs` 9 → 11, `mbar_tiers` → `{CONST:7, RING:3, THREAD:1}`,
    `edges` gained a `copy-producer → copy-producer` self-loop). The attribution of `stride=0xff8` is unproven (whether
    it is the address-chain fingerprint of alternating two slots; no SASS control), so per gotcha 12 it is recorded as
    moving along. The footprint of such changes is in the dossier's **non-contract readings**: `grid_size`,
    `smem_dynamic`, the **direction** of `mbar_objs` (same direction as the increase/decrease of the source's mbarrier
    table), the width that disappears from `nbar_verdicts` / `nbar_widths` (the role that left the rendezvous in the
    source), the extra level in `loops`; correctness can only come from a probe with many tiles, many laps + repeated
    launches ([`persistent_tile_loop_ring_continuity`](persistent_tile_loop_ring_continuity.md)).
    So the reading of the three axes gets one more sentence: **`cache_ops` not moving only rules out "the copy
    mechanism changed", not "the sync / scheduling structure changed"; for the latter, check the three
    source-checkable quantities grid / smem / mbarrier table.**

15. **The moving-along family of `rings` is not limited to `stride=0x1`: wide-stride entries (of the `stride=0x400`
    kind) also flip with "wait placement / rendezvous form", and are not structural changes of the data supply ring.**
    Gotcha 12 records `stride=0x1,wrap=sel` disappearing under `setmaxnreg` tiering; this adds **another family** — a
    `stride=0x400` ring entry (wide stride, never part of the `stride=0x1` phase/index arithmetic family) appears /
    disappears with two kinds of **sync reordering**, while the data-supply mbarrier ring (depth / slots / stride, the
    `stride=0x1,slots=9` entry always present in the readings) has not moved a finger:
    - **moving the producer-driven data waits of a ping-pong consumer out of / into the SYNC window** (hoisting
      `mbarrier_wait(K_FULL)` / `mbarrier_wait(V_FULL)` above `barrier(PP_BAR+wg)`, so that the window holds only the
      two MMA chains): `["stride=0x1,slots=9", "stride=0x400"]` ↔
      `["stride=0x1,slots=9"]` — moved out, `stride=0x400` disappears (value +2.98%); moved back, it reappears
      (value −1.97%); the same axis flips once in each direction on two adjacent KEEPs;
    - **an independent epilogue per WG (de-lockstep) replacing two 256-wide symmetric NamedBarrier rendezvous**:
      `["stride=0x1,slots=9", "stride=0x400"]` → `["stride=0x1,slots=9"]` (value −2.83%).
    All three flips came with simultaneous changes in `mbar_tiers / mbar_per_role / edges` (the epilogue one also with
    that wide rendezvous disappearing from `nbar_verdicts / nbar_widths`). **The reading strength here is weaker than
    gotcha 12; do not conflate the two**: the control of gotcha 12 is a **pure `setmaxnreg` bill change** — not a single
    ring in the source moved and the axis dropped, so moving along is **proven**; each of the three flips here came
    with a real sync-structure change (wait placement / wide rendezvous removal), so "the count axes moved in step" is
    **equally** compatible with "codegen moving along" and with "the axis faithfully recorded the thing that was
    changed", and the control that separates the two (wait placement unchanged, only flipping the register tiers; or
    the reverse) **was not done**. So this entry draws only two conclusions **supported by the readings**: ①
    `stride=0x400` is not the data supply ring — the data-supply mbarrier ring (the `stride=0x1,slots=9` entry) did not
    move a finger in all three, and `stride=0x1` is not the only family that flips; do not assume it is a real ring just
    because "it is not the known phase fingerprint"; ② the ms value comes from **the thing that was reordered** —
    moving the waits out of the window so the ping-pong window covers only the in-flight MMA chains (see
    [`wgmma_deferred_wait`](wgmma_deferred_wait.md)), removing the wide rendezvous = epilogue de-lockstep —
    **not from the `rings` flip**; the load-bearing footprint is in `nbar_widths` (the removed wide rendezvous) and in
    the SASS position of the waits. How the next run should use it: the appearance/disappearance of `stride=0x400` can
    serve as a **tripwire** (it tracks wait placement strictly in both directions, showing "something moved around
    here"), but **not as a load-bearing signal, and not as a pricing basis**; to use it qualitatively, first add the
    control above.


16. **The control gotcha 15 lacked: adding / removing only one pair of NamedBarriers (not a single object of the
    mbarrier table touched) zeroes all three `rings` families entirely, and restores them verbatim.** On the same tree
    with a frozen ring structure (two levels of K/V supply rings, 10 mbarrier objects, smem map byte count unchanged in
    every row below), `rings` flipped eight times over eight KEEPs, and not once was a ring changed:

    | The thing changed (zero change to ring source) | `rings` before → after | Gate window |
    |---|---|---|
    | a staging face read without a descriptor switched smem layout atom (`K_INTER` → `K_SW128`) | `[0x1, 0x1ff·wrap=sel, 0x2]` → `[0x1]` | 0.23543 → 0.21981 ms |
    | two full wait points in the loop body changed to try-wait spins | `[0x1]` → `[0x1, 0x1ff·wrap=sel]` | 0.21719 → 0.2183 ms |
    | the smem→RMEM A-fragment load atom: `CopyUniversalOp` → `LdMatrix8x8x16bOp(False, 4)` | previous row → plus `0x2` | 0.2183 → 0.21298 ms |
    | the same atom switched back to a scalar copy + one empty arrive moved one segment later | three families → `[0x1]` | 0.20421 → 0.20477 ms |
    | in-flight wgmma group carried across the tile back edge, the whole function serialised | three families → `[]` | 0.214 → 0.31928 ms |
    | carry withdrawn, replaced by reordering within the body | `[]` → `[0x1, 0x1ff·wrap=sel]` | 0.31928 → 0.21839 ms |
    | a pair of 256-wide NamedBarrier slots added in the steady-state body (blocking waits per body 5 → 6) | three families → `[]` | 0.21361 → 0.2189 ms |
    | that pair of slots removed (issued code back to before the addition, plus only one semantics-free timeline marker) | `[]` → three families | 0.2189 → 0.21381 ms |

    (In the table `0x1` = `stride=0x1`, `0x1ff·wrap=sel` = `stride=0x1ff,wrap=sel`, `0x2` = `stride=0x2`,
    three families = all three present.) Conclusions that can be drawn:
    - **The last two rows are the control**: adding and removing touch only NamedBarriers; mbarrier table, ring depth,
      stage / phase arithmetic and smem map all stay unchanged, yet the three families disappear entirely and come back
      verbatim — the moving-along of the wide-stride family (here `0x1ff,wrap=sel`) and of the `0x2` family is
      therefore **proven**, no longer just "compatible" as in gotcha 15.
    - **`rings == []` is not a severity reading**: a +49% whole-function serialisation and a +2.5% extra rendezvous read
      the same.
    - **`stride=0x2` is not the footprint of `LDSM`**: in the first row it disappears with the layout atom of a face, and
      that row touched no copy atom; its flipping in step with the atom in rows three / four is just more moving along.
    - **The real footprint of every row is in other readings**: serialisation shows in the `commit` axis and
      `WARPGROUP.DEPBAR / HGMMA` ([`wgmma_backedge_pending_group`](wgmma_backedge_pending_group.md)); the load atom
      shows in the SASS `LDS` / `LDSM` counts and `spill_ld / spill_st` (gotcha 13 of
      [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md)); the face's atom shows in `bankconf_st` (gotcha 15 of
      [`tma_store_epilogue`](tma_store_epilogue.md)); the extra rendezvous shows in the blocking waits per body and
      `stall_barrier` (gotcha 5 of [`handoff_sync_choice`](handoff_sync_choice.md)).

## Sources

- Official opcode surface: the `barrier`/`bar` family of the PTX ISA (from ISA 8.x `barrier.cta.*.aligned` is the
  recommended spelling, `bar.*` the historical alias) and the modifier syntax of the `cvt` family.
- This repo's GPU certification (gotcha 1 barrier side, 2, 4, 5, 6, 7, 8): an epilogue ping-pong GPU probe
  (the probe is not included in this release) →
  `PROBE PASS max_diff=n/a  # C4-d011-asymmetric-baton`, with the same round's log carrying
  `C4 verdict basis: ptx-ids=False sass-ids=False mechanism=True
  (opcode tally blocking=9 one-way=3)` — the strong criteria unreachable on both sides, the weak criterion carrying
  the verdict, which is the direct evidence for gotcha 7; the `bar.sync %r780, %r781;` form printed in the same
  round's `C4 ptx raw |` lines is the raw fact behind gotcha 4 (register-form operands).
- Gotcha 11 (region-counted axes presuppose back edges): dossier censuses of four KEEP evaluations, no `max_diff` probe — this is a reading recipe, and the evidence is the
  dossier's before/after readings.
- Gotcha 14 (`cache_ops` anchors only copy mechanisms; the two-step readings of going persistent + the second slot):
  dossier censuses of KEEP evaluations (the first step's eval job number was not entered in the status record; the
  second step was KEEP in an eval job, paired confirmation window, gate 0.2424 → 0.2284 ms), correctness carried by
  two persistent-loop probes each passing fully in the same round
  (`PROBE PASS max_diff=0.00311646`).
- This repo's GPU certification (gotcha 1 conversion side, 2, 3, 9): a second epilogue probe →
  `PROBE PASS max_diff=n/a  (ptx footprint of the shipped entry; failed checks = none)`,
  with the same round's histogram `cvt opcodes: cvt.rn.satfinite.bf16x2.f32=96 cvt.s16.s8=10 ...` and the
  footprint line `cvt_pack=96 cvt_scalar=0` both on record — the finished form with target and failure forms listed
  separately.
- Gotcha 15 (the wide-stride `stride=0x400` ring also moves along; ping-pong wait placement / epilogue de-lockstep
  trigger the flips): dossier censuses of three KEEP evaluations (job numbers not entered in the status record), axis
  `rings` `["stride=0x1,slots=9", "stride=0x400"]` ↔ `["stride=0x1,slots=9"]`, gate windows respectively
  0.2317 → 0.2386 / 0.2386 → 0.2339 / 0.2262 → 0.2198 ms; no `max_diff` probe — this is a reading recipe, and the
  evidence is the dossier's before/after readings.
- Gotcha 16 (three families flipping eight times, the NamedBarrier add / remove control): before/after dossier census
  readings of each of eight KEEP evaluations (eval job numbers not entered in the status record), axis strings and
  gate windows copied verbatim into the table in the gotcha; no `max_diff` probe — a reading recipe, the evidence is
  the dossier readings. Status: ✅ KEEP-certified.
- The probe structure that goes with the criteria (fact/verdict split, self-contained digest):
  [`gpu_probe_ladder`](gpu_probe_ladder.md).
