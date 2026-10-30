# wgmma_desc_uniform (where WGMMA operand descriptors and partition chains land in registers: pinning the long-lived descriptors of `make_fragment_A/B` into the uniform file)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **second domain of application** of mechanism 1 of [`uniform_reg_budget`](uniform_reg_budget.md).
> That entry pins the r1 yield ladder ("keep bookkeeping warp-invariant and it lands in UR") on **scalar integer
> bookkeeping chains**, with the criterion "is there a `+ tidx` in the seed". But the long-lived integer quantities
> that really fill a warpgroup GEMM are not hand-written bookkeeping chains; they are the **smem operand
> descriptors** produced by `make_fragment_A/B`, and the partition/address chains derived from
> `TiledMma / TiledCopy.get_slice(...)` — their "seed" is not an arithmetic expression but a **slice coordinate**, and
> the remedy given by the `cute.make_fragment` gotcha (bare SSA scalars + a Python list) does not apply to this
> mandatory API. This entry fills in that part.

## Purpose

SM90's `wgmma.mma_async` does not take an smem pointer directly; it takes a **64-bit matrix descriptor**
(start address / leading byte offset / stride byte offset / matrix base offset / swizzle mode packed into one
`.u64`, PTX ISA §9.7.16.6.1). In CuTe DSL this descriptor is produced by

```python
wg = tiled_mma.get_slice(0)                       # A/B are warpgroup-scoped
dA = wg.make_fragment_A(wg.partition_A(sA_slice))  # -> operand view in smem_desc form
dB = wg.make_fragment_B(wg.partition_B(sB_slice))
```

Semantically it is a quantity **of which the whole warpgroup has one copy** (that is what `get_slice(0)` means: A/B
are not sliced per thread). The hardware is designed the same way — in SASS the descriptor operand of WGMMA is
written `desc[UR*]`, **read from the uniform register file**.

So the same descriptor lands in two completely different ledgers, depending on whether ptxas **can prove** it
warp-invariant:

| Form of the descriptor's slice coordinate | Where the descriptor lands | Eats the per-thread `setmaxnreg` quota? |
|---|---|---|
| compile-time constant / derived directly from `bidx`, `warp_idx()` | `UR*` | **no** |
| obtained only after an smem/gmem round trip, across `scf.if`, across a function boundary, or after mixing in `tidx` | first lands in `R*`, then moved into UR | **yes, 2 dwords per descriptor × lifetime** |

Note the "**then moved into** UR" in the second row — it is not optional. WGMMA's descriptor operand can **only** be
read from the uniform file (`desc[UR*]`), so even when the descriptor is computed in regular registers, ptxas must move
it across files **before every issue**, one `R2UR` per dword. The failure form therefore has an extremely sharp
machine-readable signature:

> **`R2UR` count ÷ WGMMA count**. With the descriptors staying on the uniform path this ratio is ≈ 0;
> fallen into regular registers it is **≥ 2** (this entry's probe measured **4** — 2 operands × 2 dwords each,
> everything moved again before every WGMMA).

This ratio is more useful than "how many extra registers": it does not depend on launch bounds, nor on calibrating the
working set; one `grep -c` reads it, and it is **normalised**, comparable across operators. **It is this entry's
trigger criterion itself** (see "When needed") — not corroboration, not an extra.

A main loop keeps 4-6 descriptors resident (the A/B of several GEMM chains, the B of multi-buffered / partner
blocks…); **falling out of the uniform file makes 8-12 per-thread registers appear out of nowhere in the live set of
the whole loop**. On a bill where `setmaxnreg` has pinned the consumer, this is usually exactly the order of magnitude
by which the headroom is crossed — showing up as `spill_ld / spill_st` jumping from 0 to tens of millions of sectors
while `REG:` does not move at all.

But beware a counter-intuitive point, where this entry is most easily shut out by its own self-check conditions:
**the volume of the spill has nothing to do with the volume of the cause.** Where these dozen-odd dwords of tax land
and what they push out depends on how much headroom the quota has left — with wide headroom, what lies in the frame is
the descriptors themselves (8-12 dwords); with headroom already at the edge, what gets pushed out is **the whole
container array** (frame ≈ container sum, tens to over a hundred dwords). The two forms have exactly the same cause and
remedy; only their volume readings are opposite. That is why the trigger criterion below reads only the ratio.

## When needed (trigger)

**There is only one trigger criterion**, machine-readable, normalised, independent of any working-set calibration:

> **`R2UR count ÷ WGMMA count ≥ 2` ⇒ the descriptors fell out of the uniform file; this entry is the remedy.**
> Ratio ≈ 0 ⇒ the descriptors were on the uniform path throughout; the cause is not here; read no further.
> (For extraction see "Acceptance of r1"; mnemonics must be extracted by opcode position — a whole-line grep counts
> `UR4`/`URZ` in.)

The first three items are **prerequisite self-checks**: if they fail, the ratio above cannot be read meaningfully,
because what you hold is not a kernel "whose quota really took effect". They are **not** parallel trigger conditions —
once they pass, go read the ratio, and the ratio decides:

1. `setmaxnreg` really takes effect — the quartet `.reqntid N` + `.minnctapersm 1` + `dec` + `inc` all present
   (gotcha 1 of [`warp_specialization`](warp_specialization.md) / self-check 0 of [`uniform_reg_budget`](uniform_reg_budget.md)).
2. Redo the bill per mechanism 3 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md):
   `peak = A_flight + A_resident + upstream residual + W`, and measure `W`.
3. Clear the `+ tidx` out of the **scalar** bookkeeping chains per mechanism 1 of [`uniform_reg_budget`](uniform_reg_budget.md).

> ⚠️ An early version of this entry used "the spill volume divided back comes to 8-12 dwords per thread, and
> `spill_ld / spill_st` ≈ number of loop trips" as a 4th trigger condition. **That was wrong, and has been demoted to a
> corroborating reading**: it only holds under form 1 of the table below and is necessarily reversed under form 2, so
> it shuts out a kernel that really has this disease — divided back it is several hundred dwords/thread, `ld/st` lands
> near 1, both shape checks fail, while in the same SASS `R2UR/GMMA` is a clean positive. The order of the criteria is
> itself what makes or breaks the diagnosis here.

### Two failure forms; the volume criterion recognises only the first

| | Form 1: what spills is the **descriptors** | Form 2: what spills is the **containers** |
|---|---|---|
| Premise | the container set still has headroom to the quota | the container set is already at the quota |
| What lies in the frame | those 4-6 descriptors | the whole accumulator/container array (**frame ≈ container sum**) |
| dwords/thread divided back | 8-12 | tens to over a hundred, equal to the container sum |
| `spill_ld / spill_st` | ≈ number of loop trips (stored once, reloaded every trip) | ≈ 1 (stored once and read once per trip) |
| Volume shape criterion | positive | **negative (reversed)** |
| **`R2UR / GMMA`** | **≥ 2** | **≥ 2** |

The causal chain of form 2 must be spelled out, because it is "a small tax pushing out a big frame": the descriptor tax
is only a dozen-odd dwords, but it pushes a container set already at the edge out of the `setmaxnreg` quota; when the
register allocator is forced to make room, **the smallest unit it can give up as a whole is an array** — an accumulator
fragment is an array, not a pile of independent scalars — so what lies in the frame is the whole accumulator. Divide
that frame back for a shape check and the reading is necessarily "not 8-12 dwords", **while the cause is still the
descriptor tax**. Telling it apart needs no reasoning: with the same code and the same quota, adding/removing only the
single `make_warp_uniform` mark makes the frame jump between 0 and "the container sum" (C6/C8 of this entry's probe).

So the order is: **first read `R2UR/GMMA` to characterise, then use volume to explain what is in the frame**, not the
other way round with volume as the gate.

### Once judged a match, there are two legitimate actions at the same level (not exclusive; do the first first)

1. **r1: re-mark `make_warp_uniform` on the slice coordinates** (see below). Zero instruction cost, removing the tax
   itself directly. This is the only **fix**; the other only stops the bleeding.
2. **The quota itself is a lever**: raise the consumer's `setmaxnreg.inc` target by one tier (an integer multiple of
   8), subject to `producer threads × producer quota + consumer threads × consumer quota ≤ 65536`
   (gotcha 4 of [`warp_specialization`](warp_specialization.md)). The tax remains; you just move the boundary out of
   the way. It is also the **discriminating experiment** between forms 1/2: under form 2 the spill changes
   monotonically with the quota (direct evidence of a budget constraint), under form 1 it often does not move at all
   (that reload is structural).
   > Precondition: this tier must be a **declared movable lever** in your milestone. If some quota value is frozen by an
   > architecture contract, moving it is not an action this entry authorises — then only r1 remains.

> Conversely, **this entry does not solve**: the accumulators themselves overrunning
> ([`wgmma_deferred_wait`](wgmma_deferred_wait.md)), too many fp intermediates (UR is an integer datapath), or a whole
> array demoted to local by indexing a fragment with a runtime index
> (gotcha 7 of [`warp_specialization`](warp_specialization.md); that is an addressing problem, not a live-range one).
> The common signature of these three is `R2UR / GMMA ≈ 0`: if the ratio does not read positive, do not spend time on
> this entry.

## Minimal runnable example

### r1: mark `make_warp_uniform` on the **slice coordinate**, not on the descriptor

The descriptor itself is the return value of `make_fragment_A/B`; you cannot mark it; what you can mark is the
**runtime slice coordinate** fed to `partition_A/B`. The rule is:

> **Every runtime coordinate entering `partition_A/B` must, before the descriptor is built, be a value freshly marked by
> `cute.arch.make_warp_uniform`** — especially when it is obtained "after crossing something ptxas cannot see through"
> (an smem/gmem round trip, the result of an `scf.if` branch, an argument passed across an `@cute.jit` function
> boundary).

```python
@cute.kernel
def k(..., mma: cute.TiledMma, ...):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- slice coordinates: re-mark them [immediately] after obtaining them ----------------------------
    # warp_idx comes marked out of the box, but after derivation + crossing a branch/function boundary once,
    # the mark does not follow -- re-mark once where it is used as a slice coordinate, at zero cost.
    wg = cute.arch.make_warp_uniform(warp_idx // 4 - 1)
    par = cute.arch.make_warp_uniform(1 - wg)

    # ---- descriptors: A/B use get_slice(0) (warpgroup scope), only C uses get_slice(tid) --
    wgm = mma.get_slice(0)
    dA = wgm.make_fragment_A(wgm.partition_A(sA[(None, None, wg)]))
    dB = wgm.make_fragment_B(wgm.partition_B(sB[(None, None, par)]))
    acc = cute.make_rmem_tensor(wgm.partition_shape_C((M, N)), cutlass.Float32)

    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, True)
    for t in cutlass.range(NTRIP):          # dA/dB are loop invariants, resident across the whole loop
        warpgroup.fence()
        for kb in cutlass.range_constexpr(cute.size(dA.shape[2])):
            cute.gemm(atom, acc, dA[None, None, kb], dB[None, None, kb], acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
```

Counter-example (the same code, missing only those two re-marks):

```python
    wg = warp_idx // 4 - 1        # <<< a derived value; after crossing a branch/function boundary ptxas cannot prove invariance
    par = 1 - wg
    ...                            # the descriptors fall back into R* wholesale, 2 dwords each x the whole loop
```

**The cost comparison is machine-readable** (GPU measurements of this entry's probe, see "Sources"): the two variants
differ only in that single `make_warp_uniform` mark and are bit-for-bit identical numerically; the unmarked side
**needs `REG(P) - REG(U)` more per-thread registers** (measured: 5 descriptors priced at **+20**), the count of
**operations** on the uniform datapath collapses accordingly, and at the same time `R2UR` jumps from single digits to
**4 per WGMMA** — the descriptors are computed in R* and moved again into UR before every issue.

### Acceptance of r1 (counting U* instructions; not doing it is the same as not doing the change)

Mnemonics must be extracted **by opcode position** (the first token after the instruction-address comment). A sloppy
whole-line grep counts **operand names** such as `UR4` / `URZ` as uniform instructions, giving away a large slice of
the pass mark:

```bash
cuobjdump -sass a.cubin > s.txt
opc() { sed -nE 's#.*/\*[0-9a-f]{4,}\*/[[:space:]]+(@!?[A-Z0-9_]+[[:space:]]+)?([A-Z][A-Z0-9_.]*).*#\2#p' s.txt | cut -d. -f1; }

# 1) [main criterion] number of vector->uniform moves, divided by the WGMMA count
opc | grep -cE '^R2UR$'                 # numerator
opc | grep -cE 'GMMA$'                  # denominator
#    ~0        descriptors computed on the uniform path throughout and kept in UR -- the form we want
#    >=2/each  descriptors computed in R*, moved wholesale into UR before every issue -- exactly this entry's disease

# 2) descriptors really are read from the uniform file (hardware form, true on both sides; counting none = you are not reading real SASS)
grep -cE '[A-Z]*GMMA.*desc\[UR' s.txt

# 3) descriptor arithmetic really happens on the uniform path: the positive case should clearly exceed the counter-example
opc | grep -E '^U' | grep -vcE '^ULDC$'
#    ULDC does not count -- kernel-parameter constant-bank reads, present in every kernel (same as uniform_reg_budget "Acceptance").
#    R2UR/S2UR/VOTEU/REDUX do not count either -- they are **cross-file moves**, not uniform operations;
#    counting them makes the criterion **flip sign** (see gotcha 8, where the first version of this entry fell).

# 4) absolute-value ruling
cuobjdump -res-usage a.cubin        # REG / STACK
opc | grep -cE '^(LDL|STL)$'        # must be 0
```

**Do not enable launch bounds when measuring the "price".** `.reqntid N` + `min_blocks_per_mp>0` pins `REG:`
(384 threads ⇒ 168); then REG does not vary with the working set and cannot be used as a criterion
("Acceptance" of [`uniform_reg_budget`](uniform_reg_budget.md)). To read directly "how many registers this spelling is
worth", run the same code in a probe with **128 threads, one warpgroup, no `min_blocks_per_mp`**: the ceiling becomes
`65536/128`, `REG:` is the real per-thread demand, and `REG(counter-example) − REG(positive case)` is the price tag —
**no need to calibrate the working set to the edge of the budget to force a spill**. The calibrate-to-the-edge route
(how things were done before this entry's probe) depends on guessing the base chain depth, and a wrong guess gives the
empty experiment "neither side spills".

### r2: the next tier of the yield ladder — **one resident base address per buffer; do not hoist N sliced views out of the loop**

A/B descriptors derived from `get_slice(0)` can go into UR via r1; **partitions derived from `get_slice(tid)`
(`partition_C` / `TiledCopy.partition_S/D`) cannot** — by definition they are one address per thread, and UR cannot
hold them. For this tier the action is not "switch files" but "reduce the count":

```python
    # good: partition the whole tensor once, keep the varying mode in the partition, consume the runtime coordinate only [at the call site]
    tDg = thr_copy.partition_D(gDst)                # (CPY, ..., STAGE)
    for t in cutlass.range(NTRIP):
        cute.copy(tc, tSrc, tDg[(None, None, t)])   # only 1 base address resident

    # bad: hoisting N already-sliced views out of the loop, N per-thread base addresses resident across the whole loop
    tDg_s = [thr_copy.partition_D(gDst[(None, None, s)])
             for s in cutlass.range_constexpr(NSTAGE)]
    for t in cutlass.range_constexpr(NTRIP):
        cute.copy(tc, tSrc, tDg_s[t])
```

The same discipline holds for descriptors too, and is the idiomatic form of an SS main loop: **keep modes such as
stage / partner block in the fragment and index `dA[(None, None, kb, stage)]` at the call site**, rather than building
one descriptor per stage and hoisting it out.

> ⚠️ **This section was not separately ruled on by this entry's probe.** The two ingredients of r2 each have an existing
> source: "per-thread partitions cannot go into UR" is a definitional consequence of `get_slice(tid)`; "indexing a
> fragment with a runtime index ⇒ the whole array demoted to local memory" is gotcha 7 of
> [`warp_specialization`](warp_specialization.md). ptxas applies strength reduction to a set of addresses with a
> **constant stride**, hoisting the per-thread offset out for sharing, so the real price of hoisting may be far below
> `N × 2` — to attribute spills to it, **first measure it with the "measuring the price" method above**; do not draw
> conclusions from the form
> (for a record of the same kind of refutation see [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md):
> packing in place does not lower the peak).

## Gotchas

1. **`make_warp_uniform` is an assertion, not a conversion.** It only tells the compiler "this value is the same
   everywhere in the warp"; it does not turn a quantity that really varies by lane into a uniform one. A wrong mark
   raises no error — it **silently computes wrong**. Before marking, ask yourself: is this coordinate really equal on
   all 32 lanes of the same warp?
   (warpgroup index, stage index, block index: yes. lane/quad index: no.)
2. **The mark does not follow derivations, and in particular does not cross function boundaries or `scf.if`.** The
   DSL's `warp_idx()` comes marked out of the box, but once `warp_idx // 4 - 1` is passed as an argument into another
   `@cute.jit` helper, or goes through a role branch once (`if warp_idx < 4: ... else: ...`), the chain of proof ptxas
   holds is broken by the time it reaches the descriptor construction point.
   **Re-mark once at the point where the slice coordinate is used** — it generates no instruction; there is no reason
   to skip it.
3. **A/B use `get_slice(0)`, C uses `get_slice(tid)`; getting it backwards breaks both ends.** Building A/B
   descriptors with `get_slice(tid)` turns a warpgroup quantity into a per-thread quantity with your own hands
   (same as gotcha 3 of [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)); building the accumulator/write-back partition with
   `get_slice(0)` makes 128 threads write the same addresses.
4. **A descriptor is not "data in a fragment"; do not carry it with `cute.make_fragment`.**
   On smem operands `make_fragment_A/B` returns a **descriptor-form view**, not a block of rmem storage; `.load()`ing
   it elsewhere, or building another fragment to hold it, forcibly turns a quantity that could stay in UR into
   per-thread storage (same origin as "do not put uniform quantities into `cute.make_fragment`" in
   [`uniform_reg_budget`](uniform_reg_budget.md)).
5. **`REG:` is a constant under launch bounds; do not report progress with it.** Conversely, **to measure the price
   you must remove the launch bounds** (see "measuring the price" above). Each reading has its single correct use;
   mixing them gives the two illusions "I changed it and nothing happened" or "whatever I change it's the same".
6. **For an A/B, each variant gets its own process and its own dump directory.** The dump file name is built only from
   the jit function's tensor signature; the `Constexpr` variant selector is not in the name: compiling two variants in
   one process ⇒ the second cubin overwrites the first in place, and same signature = same compile-cache key. The
   variant selector must be captured by a **closure** in an entry factory, not passed as a `Constexpr` parameter of the
   `@cute.jit` entry (it shifts by one at launch and raises `cannot be converted to pointer`, while compilation is
   completely normal) — same as "Acceptance" of [`uniform_reg_budget`](uniform_reg_budget.md).
7. **The control must itself compute correctly.** A spill / extra registers is a performance defect, not a numerical
   one; if the control computes wrong, "it is more expensive" carries no information. The two variants of this entry
   are semantically identical (the **values** of the slice coordinates are equal per lane; ptxas just cannot prove it),
   so they share the same reference — which is also why this A/B is clean.
8. **When counting uniform instructions, "cross-file moves" are not uniform operations — counting them flips the
   criterion's sign.** Two kinds of things must be removed from the count:
   - `ULDC`: kernel-parameter constant-bank reads, present in every kernel; counting them gives away a free pass;
   - `R2UR` / `S2UR` / `VOTEU` / `REDUX`: these are **vector ↔ uniform moves**, not operations on the uniform datapath.
     **It is precisely the failing side that moves the most** (descriptors computed in R* and moved into UR before
     every WGMMA issue).
   > **Refutation record**: the first version of this entry's probe added `R2UR`/`S2UR` into the same counter as `U*`,
   > so the criterion "the positive case should have more uniform instructions" read **positive case 90 /
   > counter-example 298** on GPU — completely backwards, and the contract judged FAIL. In the same round `REG` differed
   > by +20 and `R2UR` 1 vs 288; the mechanism actually held very cleanly, **the only thing wrong was the counter
   > counting the symptom as the cure**.
   > After splitting into two counters, "uniform operations" and "cross-file moves", both are strong signals in their
   > own direction.
   There is also a pure syntax trap: mnemonics must be extracted **by opcode position** (see "Acceptance" above);
   a whole-line grep for `U[A-Z]*` counts **operand names** like `UR4`/`URZ`.
9. **UR slots are limited.** External reverse-engineering material gives sm_75–sm_90 `UR0..UR62` + `URZ` per warp
   (`[external RE material, unverified by this entry]`). Descriptors are only a few, enough; but if there are also
   dozens of levels of uniform bookkeeping chains, ptxas **silently** puts what does not fit back into regular
   registers — the symptom is "everything is marked and it still spills"; counting `U*` instructions then shows the
   chain did not all get in.
10. **Do not infer the size of the cause from the size of the frame.** "Over a hundred dwords spilled, so the cause is
    something of over a hundred dwords" is the most expensive wrong inference of this entry. The register allocator
    does not retail by the dword: **a fragment is an array, either staying in registers whole or going into the frame
    whole**, so as soon as a tax of a dozen-odd dwords crosses the quota line, what falls into the frame is **all** of
    the array it pushed out. Two disciplines follow:
    - characterise only by `R2UR / GMMA`; use volume only to explain **what is in the frame** (divided back ≈ the size
      of some container ⇒ form 2; ≈ the sum of the descriptors ⇒ form 1).
    - remove/add `make_warp_uniform` for a same-quota A/B and watch the frame jump between 0 and the container sum —
      the only reading that pins the causality in a single step, faster than any volume arithmetic.
11. **Quota sweeps go in steps of 8, and `inc` must really be an "increase".** The target of `setmaxnreg.inc` must be
    above the baseline given by the launch bounds (`65536 / threads per CTA`), otherwise it is not an increase and does
    not behave the way you think. Sweeping one tier up and down (±8) is much stronger than a single point: a single
    reading cannot tell "the quota is not the constraint" from "this tier is still not enough"
    (same as the ladder discipline of [`uniform_reg_budget`](uniform_reg_budget.md)).
12. **Landing r1 without a positive trigger reading: the mark is not free — the full reading of one KEEP.** In a
    consumer main loop at the `setmaxnreg` 192 tier, the loop counter was marked once with `make_warp_uniform`, seven
    ring-stage indices (`i % X_STAGE`) were derived from it and each marked at its definition, and seven mbarrier phases
    (`(i // X_STAGE) % 2`) were **hoisted from the `_wait` call sites to the top of the loop** as seven marked SSA
    values — these indices seed five wgmma B descriptor slice coordinates, eight mbarrier pointers and the LDSM/STSM
    partition coordinates, and the assertion itself is true (gotcha 1). `R2UR / GMMA` was **not** read before landing.
    Readings of the same eval: gate window 0.4922 → 0.5137 ms (**+4.37%, slower**), `spill_ld_sectors` **0 →
    393,216**, `spill_st_sectors` 1,835,008 → 2,228,224, tensor_pct 10.51 → 10.07; the dossier's `rings` axis gained an
    extra bare `stride=0x1` beyond `[stride=0x1,slots=9, stride=0x1,wrap=sel]` (some mbarrier address chain resolved as
    a stride-1 self-increment without wrap evidence). Two actions were mixed in one change — the mark itself, and
    hoisting seven phases from the call sites to the loop top (each live range stretched from "one div/mod" to "loop
    top → that wait point") — and the readings cannot be split. Transferable disciplines:
    - **Read the ratio first; if the ratio ≈ 0, do not touch it** ("When needed"). r1's gain comes only from the one
      disease "descriptors fell out of UR"; without the disease, the mark can only be zero or negative.
    - **Do not hoist phases / indices to the loop top just "to make them one uniform SSA value"**: that lengthens live
      ranges, and on a quota at the edge it converts directly into `LDL/STL`. Re-mark at the **point of use** (r1's own
      words), without moving the definition.
    - **A bare `stride=0x1` ring fingerprint is not a footprint of the mark**: on the same tree a change that touched
      neither mbarriers nor marks (the gotcha "Shortening the chain does not make the whole kernel faster" of
      [`tri_forward_subst_shfl`](tri_forward_subst_shfl.md)) also made it appear. It only says the address chain's
      codegen changed, not that it got better or worse.
    - Side evidence from the same tree: marking only the two role-fork seeds as uniform and changing nothing else, the
      device window went 0.5014 → 0.6096 ms (as reported by a pre-flight census probe, not included in this release).
      **Marks must be measured; they cannot be treated as harmless annotations** ("Acceptance" of
      [`uniform_reg_budget`](uniform_reg_budget.md)).

## Sources

- **Hardware facts (WGMMA matrix descriptor)**: PTX ISA
  [Asynchronous Warpgroup Level Matrix Instructions — Matrix Descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)
  (`desc-a` / `desc-b` of `wgmma.mma_async` are 64-bit `.u64` descriptors encoding start address /
  LBO / SBO / matrix base offset / swizzle mode; the instruction is warpgroup-scoped).
- **Hardware facts (uniform datapath / per-warp `UR*` file)**: see the public reverse-engineering material listed in
  "Sources" of [`uniform_reg_budget`](uniform_reg_budget.md)
  (there is no official specification of SASS semantics; marked `[external RE material]`). **"WGMMA's descriptor
  operands are read from the uniform register file" was measured directly by this entry's probe** (the descriptor
  operand of `GMMA` instructions is written `desc[UR*]`).
- **CuTe DSL API surface**: `cutlass.cute.atom` — `TiledMma.get_slice` /
  `ThrMma.partition_A` / `partition_B` / `partition_C` / `partition_shape_C`,
  `MmaAtom.make_fragment_A` / `make_fragment_B` (`_cute_ir.mma_make_fragment`,
  `MmaOperand.A|B`); `cutlass.cute.nvgpu.warpgroup` — `OperandSource.SMEM`
  (`MmaFragKind.smem_desc`) / `OperandMajorMode` / `Field.ACCUMULATE`;
  `cutlass.cute.arch.make_warp_uniform` (`cutlass/cute/arch/elect.py`, which describes itself as
  "compiler hint indicating that the specified value is invariant across all threads
  in the warp", lowered to `cute_nvgpu.arch_make_warp_uniform`), `cutlass.cute.arch.warp_idx`
  (itself `make_warp_uniform(tid // WARP_SIZE)`).
- **Adjacent certified entries**: [`uniform_reg_budget`](uniform_reg_budget.md) (r1 for scalar bookkeeping chains,
  the `setmaxnreg` self-check quartet, A/B discipline), [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)
  (`get_slice(0)` vs `get_slice(tidx)` for SS operands, swizzled smem layout construction),
  [`warp_specialization`](warp_specialization.md) (gotchas 1 / 7 / 11),
  [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) (per-segment live-set accounting).
- **Official reference**: cutlass examples `examples/python/CuTeDSL/hopper/dense_gemm.py` (same API surface).
- **GPU-certified probe** (+ a subprocess worker; not included in this release) — a single warpgroup (128 threads,
  **no** `min_blocks_per_mp`, so `REG:` is the real per-thread demand), bf16 WGMMA SS,
  five loop-invariant descriptors (`sA[..., s0]` / `sB[..., s0]` / `sA[..., s1]` / `sB[..., s1]` /
  `sB[..., s2]`), with the slice coordinates fed by a gmem load `mSel[bidx, tidx]` that is **addressed per thread but
  takes the same value across lanes** — ptxas cannot prove invariance, while the `make_warp_uniform` assertion is true,
  so the two variants are semantically identical and share one reference. Each variant runs in its own subprocess with
  its own dump directory.
  Seven contracts: U and the control each equal to the fp32 reference bit for bit, `STACK == 0` on both sides (the REG
  reading is pure demand rather than a spill artefact), `REG(control) − REG(U) ≥ 8`, `uALU(U) − uALU(control) ≥ 16`
  (neither `ULDC` nor `R2UR`/`S2UR`/`VOTEU`/`REDUX` counted — the former gives a free pass, the latter are moves not
  operations, and counting them flips the criterion's sign, see gotcha 8), **`R2UR(control) ≥ 2 × GMMA count` and
  `R2UR(U) ≤ GMMA count / 8`** (the normalised cross-file move criterion), and on the U side every `GMMA`'s descriptor
  operands taken from `UR`.
  - **Certification status: ✅ GPU-certified** — `PROBE PASS max_diff=0.0`, all seven contracts passed in the same round.
- **GPU-certified probe (form 2 / trigger criterion)** (+ a device-side worker) — two tiled MMAs sharing one warpgroup
  (`(64,256)` + `(64,64)`, accumulators totalling **160 dwords/thread**, resident across the whole loop), fed by five
  loop-invariant descriptors; the A/B again differs only in one `make_warp_uniform` mark. Two phases:
  "price": a single warpgroup, **no** `min_blocks_per_mp`, `REG:` is the real per-thread demand, and
  `REG(P) − REG(U)` is the calibration-free price tag; "ws": 1 producer + 2 consumer warpgroups with
  `setmaxnreg`, a quota ladder `ceil8(REG(U)) + {0, 8, 16}`, finding the tier where "U fits and P does not".
  Ten contracts: the quartet present, both variants each equal to the fp32 reference **bit for bit** (integer
  operands, errors are not rounding), both price variants `STACK == 0` with `REG(P) − REG(U) ≥ 8`, the cubins of all
  variants mutually distinct, **a boundary tier exists** (at the same quota `STACK(U) == 0` while `STACK(P) > 0` ⇒ the
  only remaining independent variable of the causality is the mark), **the volume criterion reads negative at that
  tier** while **`R2UR/GMMA` is ≥ 2 on the P side and ≤ 0.5 on the U side**, and `desc[UR..]` countable on both sides.
  - **Certification status: ✅ GPU-certified** — `PROBE PASS max_diff=0.0`, all ten contracts C1-C10 passed in the same round.
    Form 2 is therefore a **measured form** rather than a conjecture: with the same code and the same quota tier,
    adding/removing only one `make_warp_uniform` mark makes `STACK` jump between 0 and non-zero (C6), while at that tier
    **the volume self-check reads negative and `R2UR/GMMA` is still ≥ 2 on the P side / ≤ 0.5 on the U side** (C8/C9) —
    the change "only one trigger criterion, the ratio; volume demoted to corroboration" thus has GPU backing.
- KEEP certification of the production cubin contract axis (gotcha 12): KEEP in an eval job (job number not entered in
  the status record), axis `rings` `[stride=0x1,slots=9, stride=0x1,wrap=sel]` → `[stride=0x1, stride=0x1,slots=9, stride=0x1,wrap=sel]`,
  gate 0.4922 → 0.5137 ms, spill_ld 0 → 393,216, spill_st 1,835,008 → 2,228,224, tensor_pct 10.51 → 10.07.
  The pre-flight census probe (`R2UR/GMMA`, U* counts and the per-`USETMAXREG`-tier
  LDL/STL split for the three variants head / duc / du) was run; its readings were not archived with the KEEP record.
