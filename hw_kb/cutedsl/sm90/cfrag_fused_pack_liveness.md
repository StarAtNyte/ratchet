# cfrag_fused_pack_liveness (the **live-range** form of a C-fragment elementwise segment: fused consumption + N-axis broadcast scalars)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **live-range patch** to [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md).
> That one covers "for the same number of elements, how to use the fewest instructions" (per element 1 FFMA + 1
> MUFU.EX2 + 0.5 F2FP); **this one covers how to get this segment's register ledger right, and how to fetch the
> scalars when the broadcast axis falls on the N axis**.
>
> ⚠️ This entry is **not** a recipe for "change the spelling to eliminate spill". The two common spellings of an
> elementwise segment (in-place packing vs a separate fragment, fetching scalars per pair vs per element) were
> measured to **compile to the same footprint** — read the next section first, and don't attribute spill to the
> packing spelling.
>
> ⚠️ **Read-side volume (`smem_wavefronts_ld`) is a different matter; see mechanism 2b.** Both smem fetch forms of
> mechanism 2 (per element / per pair) are pinned by the same **dynamic** lower bound; to actually push the read side
> down, the segment must stop touching smem per element — move the scalar vector into registers once per warp, then
> broadcast with `shfl.sync.idx`.

## Two things this entry does **not** claim (falsified on the GPU, `[MEASURED]`)

First nail down two claims that sound reasonable but do not hold when measured, so nobody tunes by them:

1. **"Packing in place lowers the peak" — does not hold.** The peak falls in segment A: the fp32 accumulators of two
   MMA chains (64 + 64) are alive there at the same time, and the second is **deliberately left in flight** (see
   mechanism 3); no packing spelling can change that. By segment B, the upstream accumulator is **already dead in
   both spellings** — the "separate `rP`" spelling also moves the data away pair by pair, and the elements of `acc_s`
   likewise lose their readers pair by pair. ptxas does live-range splitting on fully unrolled fragments, so the two
   source forms land on **the same register allocation**. Measured: the `(spill, REG, LDS)` of FUSED and BLOCK =
   `(0, 168, 32)`, **exactly the same**. A reading of the same kind on a production tree (a KEEP): in
   a role at the 104 tier, changing two slot-exit faces from "mask channel → f32 staging fragment → batched `.to()` →
   `cute.copy`" to "read the accumulator through the store atom's `retile(acc)`, write straight into a 16-bit
   fragment with pairwise `cvt`" (fused bit-equal to staged, probe in "Sources"), deleting the declaration of 32 f32
   staging registers: gate 0.4115 → 0.4141 ms (+0.6%), `spill_ld/st` 0/0 → 132,096/2,048. **Deleting the staging
   fragment did not save a single register, and even added a little spill** — staging only lives within those few
   narrowing instructions, and ptxas had long since cut it into other dead zones.
2. **"Fetching scalars per element costs 4× the LDS instructions of fetching per pair" — does not hold as a static
   footprint.** Under full unrolling with `range_constexpr`, every scalar address is "base + compile-time constant",
   and ptxas merges the per-element gather straight into **the same 32** instructions as the pair-vector fetch.
   Measured: `LDS = 32` for both.

⇒ So **the static footprint (`cuobjdump`) cannot tell these two spellings apart**: the two source forms compile to
the same instruction stream, so their **dynamic** read-side volume must be equal too. Read-side volume can only be
measured with ncu's `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`, not with the LDS count in SASS.

### A **retraction** (the original text extrapolated from the static footprint to dynamic volume, overreaching) `[MEASURED]`

This entry once said "high `smem_wavefronts_ld` is not a trigger of this entry / has no verified causal link to
reading scalars per element", and recorded one implementation's 4.18× read-side overrun as "attribution falsified,
cause not located". **That judgement is void.** The only part of it that holds is "there is no difference **between
the per-element and per-pair spellings**"; it was wrongly read as "**fetching scalars per element from smem** has, in
itself, no causal link to wavefront volume". The correct causality is (given and GPU-certified by mechanism 2b):

> Each execution of an LDS produces **at least 1 smem load wavefront** (a wavefront is a pipeline cycle the LSU
> spends serving one request; ≥ 1 even without bank conflicts). Therefore **any** form that fetches scalars per
> element / per pair inside the segment has its read-side volume per warp-iteration **held up from below** by its
> own static count:
>
> ```
> smem_wavefronts_ld / warp-iteration  ≥  LDS issued by the segment per warp-iteration  ≥  NSTAT × NELEM/4
> ```
>
> ptxas's merging can only bring per-element **down to** this lower bound, **never below** it. So there is causality
> between "read-side overrun" and "fetching row scalars from smem inside the segment" — it just **cannot be crossed**
> by the per-element/per-pair distinction in spelling; it takes a whole different tier of mechanism (mechanism 2b) to
> move it.

What is left of this entry, then? **Accounting** (mechanism 3), **the lower bound of the smem fetch form**
(mechanism 2), and **the tier that escapes that bound** (mechanism 2b). In-place packing (mechanism 1) is still
recommended as a **spelling** — it costs nothing extra and makes the accounting easier — but it is not a
register-saving lever; **don't treat it as a spill-eliminating action**.

## Purpose

WGMMA's fp32 accumulator (C-fragment) is large: a 64×N warpgroup accumulator over 128 threads is `64*N/128` fp32
registers/thread (N=128 → **64 registers/thread**). The elementwise segment converts it to fp16 to feed the next MMA
or write back to smem. This entry covers the segment's **live range**:

1. **How to compute the peak correctly** (mechanism 3) — account by "program point" rather than by "fragment list",
   with the upstream accumulator counted by its **residue** rather than at full size. The typical consequence of
   getting it wrong is an inflated sum (e.g. 288 vs a budget of 240), leading to the misjudgement "the MMA overlap must
   be cut", when the overlap is exactly the item that should not be cut.
2. **How to fetch broadcast scalars** (mechanism 2) — when the broadcast axis falls on the accumulator's **N** axis
   (transposed accumulator), the row hoist of `cfrag_elementwise_fastpath` **does not hold**; one 64-bit fetch per pair
   keeps the resident scalars at a constant 2, and lands on the merging lower bound by construction.
3. **The in-place packing spelling** (mechanism 1) — the recommended spelling that costs nothing extra and makes the
   accounting easier. **But it does not save registers** (see "Two things this entry does not claim" at the top).
4. **How to push this segment's smem read-side volume down to the information-theoretic lower bound** (mechanism
   2b) — the broadcast-axis scalars exist only once for the whole warp; staging them into registers once and
   broadcasting with `shfl.sync.idx` drops the read side from "≥ 1 LDS per 4 elements" to "NSTAT × JMAX LDS per
   warp-iteration". This is the only lever that moves `smem_wavefronts_ld` when it is over the line.

## When needed (trigger)

- `spill_ld / spill_st` (ncu `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`) is **non-zero**, while both knobs
  of [`warp_specialization`](warp_specialization.md) (`setmaxregister_*` + `min_blocks_per_mp`) are confirmed in
  effect (PTX has `.minnctapersm` + `setmaxnreg`). Still spilling with the split done right ⇒ the cause is the
  **working set**, not the budget allocation. Then **redo the accounting** with mechanism 3 (in particular, measure
  `W`), rather than first changing the packing spelling — changing the spelling was measured not to move the
  footprint.
- The sum of the fragments' **allocation sizes** already exceeds the budget (e.g. 7 fragments totalling 288 registers
  vs a budget of 240). **This sum is almost certainly inflated**: it counts each fragment at full size and as alive
  throughout, whereas the upstream accumulator actually has only a single-digit residue at the peak point. Recompute
  per mechanism 3 first, then judge whether it is really over budget.
- **Transposed accumulator**: the row scalars the elementwise segment subtracts/multiplies are indexed along the
  accumulator's **N** axis rather than the M axis. The hoist spelling of
  [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md) §(b) (`for r: stat = mStat[row]; for c: ...`) **does
  not hold** then, and copying it degenerates into per-element re-fetches.

- **`smem_wavefronts_ld` over the line** (triaged as the **volume type** per
  [`smem_traffic_triage`](smem_traffic_triage.md): wavefronts over the line while `bankconf_ld` is clean, and the st
  side not rising with it ⇒ purely the read side). Then **first compute one number**:
  `smem_wavefronts_ld / (CTA-iterations × consumer warps per CTA)`. If this "per warp-iteration" number falls around
  `NSTAT × NELEM/4` or above, the lesion is this segment fetching row scalars per element/per pair from smem —
  **change it per mechanism 2b; don't change the packing spelling** (the two spellings of mechanisms 1/2 give the same
  number here, and changing them gains nothing).

> Note the **direction** of the triggers: `smem_wavefronts_ld` over the line triggers **mechanism 2b**, not mechanisms
> 1/2. Rewriting per element ↔ per pair compiles to the same instruction stream, with identically zero gain on this
> metric (see item 2 of the falsification record at the top).

## Mechanism 1: per-fragment consumption → in-place packing → ending the upstream live range

### 1.1 In-place packing is legal (not "frugal use" — provably safe)

The fp32 elements `2k`/`2k+1` of the C-fragment are packed into one 32-bit fp16 pair. Writing this pair back into
**the k-th 32-bit slot of the same fragment** (`recast_tensor(acc, Int32)`) is always safe:

```
read acc[2k], acc[2k+1]  ->  write slot k
k <= 2k, with equality only at k = 0, and at k = 0 the read happens before the write
```

So traversing `k` in ascending order, the write pointer always stays behind the read pointer. The packed result
occupies the **first half** of the fragment's slots with the same base — so it can still be aliased through the same
`iterator` as the A-operand of the next RS WGMMA (bijection in
[`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md) §(c)).

```python
p_i32 = cute.recast_tensor(acc, cutlass.Int32)     # 64 32-bit slots, aliasing the same registers
p_f16 = cute.recast_tensor(acc, cutlass.Float16)   # 128 fp16 slots, for downstream per-element read-back

for k in cutlass.range_constexpr(cute.size(acc) // 2):
    p_i32[k] = cvt_f16x2_f32(f(acc[2 * k]), f(acc[2 * k + 1]))   # in place
# At this point: acc's fp32 semantics have all ended; only the first 32 slots carry the fp16 result
```

The contrasting form (the example of `cfrag_elementwise_fastpath` §(c)):

```python
rP = cute.make_fragment(acc_fA.shape, cutlass.Float16)   # a separate full-size fragment
for i in ...: acc[i] = f(acc[i])                          # compute the whole block first
for k in ...: rP_i32[k] = cvt_f16x2_f32(acc[2*k], acc[2*k+1])   # then pack the whole block
```

Intuition says "the full-size fp32 and the growing fp16 coexist at the same program point" here. **Measured, that is
not so** `[MEASURED]`: the second loop, as it reads `acc[2k]`, also takes away its last reader; ptxas
splits the live ranges per element, the fp32 of `acc` and the fp16 of `rP` **shrink and grow in step**, and their sum
never exceeds E. The two spellings therefore land on the same allocation (`spill/REG/LDS` all equal).

⇒ **The reason for the in-place spelling is not saving registers** but: one fragment name fewer, no need on the books
to explain "are E fp32 and E/2 fp16 alive at the same time", and the packed result shares the base with the upstream,
so aliasing it as the A-operand needs no second `iterator`. These are readability and accounting gains, not footprint
gains.

### 1.2 A checkable form of "must not keep a full fp32 copy"

A criterion you can check against the code line by line: **after the elementwise segment ends, the upstream
accumulator's fp32 values must have no readers left**. If downstream still needs that value, let it read the
**packed fp16** (`p_f16[i]`, one `H2F`); don't keep the fp32 for it — keeping one fp32 element costs 1 register,
reading it back costs 1 instruction, and multiplied by the element count, the former is the one that lands on the
peak.

### 1.3 When the destination is smem: interleave by blocks, don't accumulate the whole block and then copy once

If the consumer of the packed result is an **R2S copy** (the `STSM` family) rather than an RS WGMMA, there is no
constraint that "the whole block must be alive at once": **pack a block, store a block**, and the registers only need
to carry the width of one copy atom — F2FP and STSM interlock block by block, rather than "all F2FP → one
`cute.copy` issued as a batch". **An RS WGMMA's A-operand has no such freedom**: the whole A-fragment must stay alive
until the corresponding MMA retires; that is a bill that must be paid, so write it into the budget.

### 1.4 Boundary (GPU-measured): don't open a second type view on a fragment that has already served as a wgmma A operand to do an after-the-fact RMW

The aliasing of 1.1 holds only **in one direction, once**: fp32 accumulator → in-place packing → the packed result
aliased through the same `iterator` as the A operand of an RS WGMMA. The following extension was **measured not to
hold**:

```python
# tOrP: the packed bf16 RS A-fragment, which has already fed one wgmma chain
p_w = cute.recast_tensor(cute.make_tensor(tOrP.iterator, cute.make_layout((NPAIR,))),
                         cutlass.Int32)            # a second type view, aliasing the same registers as tOrP
...
warpgroup.wait_group(0)                            # the chain has drained; the registers are "writable"
for j in cutlass.range_constexpr(NPAIR):
    p_w[j] = mul_bf16x2(p_w[j], factor_w[j % NROW])   # correct the packed result per pair after the fact (inline PTX mul.rn.bf16x2)
```

In an isolated probe (one warpgroup, the kernel's own two MMA atoms; the GPU probes of this entry are not included in this release) all five contracts
PASS, including `no_spill` (LDL == STL == 0) and "shorter SASS". Put into a real consumer mainloop at 168
registers/thread with the live set close to the cap:

| Reading | No second view | Second view + after-the-fact RMW |
|---|---|---|
| `spill_ld / spill_st` sectors | 0 / 0 | 1,699,968 / 1,699,968 |
| `regs_per_thread` | 168 | 168 |
| `sm__pipe_tensor_cycles_active` | 56.15 % | 44.37 % |
| `commit` axis (per consumer ring) | `{LE 0x0 x1, LE 0x1 x1}` … | `{LE 0x0 x16}` ×4 + `{LE 0x0 x8}` ×2 |
| gate window | 2.18 ms | 2.9641 ms (+36%) |

Mechanism reading: the allocator drove this fragment out to local memory, so wgmma's A registers are defined by
LDL, and ptxas gives up pipelining the wgmma for **the whole function** (a `WARPGROUP.DEPBAR.LE 0x0` after every
HGMMA, including chains that were never touched; criteria and triage in the "On-machine footprint" section, path 2, of
[`wgmma_backedge_pending_group`](wgmma_backedge_pending_group.md)). The 32 ALU instructions saved cannot buy back 3.4 M
spill sectors.

After taking the second view out of the trace (building it, together with its factor array, only inside a
`cutlass.const_expr` branch), spill returned to zero, the `commit` axis recovered, 2.1712 ms. The tier "build the view
but do no RMW" has not been measured separately, `[to be checked]`; the two measured ends are "view + RMW" and
"neither".

Rule: when P needs a per-row correction **after** packing, use the form "correct the fp32 source → repack" (paying 32
FMUL + 16 F2FP extra, while the fragment keeps its register home), rather than multiplying in place on the packed
result; **an isolated probe's `no_spill` does not transfer** — the register home is the result of whole-function
allocation, and only the target cubin's NCU `spill_ld / st` counts.

## Mechanism 2: fetching scalars when the broadcast axis falls on the N axis

### 2.1 First get the C-fragment's linear order right (verified on the GPU by this entry's probe, `[MEASURED]`: zero violations over 4096 groups)

The linear element order of the per-thread fragment given by `partition_C` is **`(n_pair fastest, then m_pair, then
v)`**:

| Linear index | Relation |
|---|---|
| `4v`, `4v+1` | same M, **two adjacent N** (`n` and `n+1`) |
| `4v+2`, `4v+3` | **the same pair of N**, the other M |

Two corollaries, both depending only on this order:

- The two elements of one packed pair **span two adjacent N** (so for one pair the N-axis scalar is **two** values,
  not one broadcast value — which is exactly why the row hoist fails under transposition);
- Two adjacent pairs (`2v` and `2v+1`) **share the same pair of N scalars** (they differ only in M).

⇒ **Only one 64-bit scalar fetch is needed per 4 elements** (two adjacent N, 8-byte aligned, because a pair's
starting N is always even). This is the **lower bound** of fetching, and what this entry's probe verifies is that it
lands on the lower bound (`LDS == 2 × NELEM/4`, half for each of the two statistics) rather than degenerating into two
32-bit gathers.

> Note: **this is not a static advantage over the per-element spelling.** Under full unrolling ptxas merges the
> per-element gather to the same count (see item 2 of "Two things this entry does not claim" at the top). The value of
> writing per pair is that it lands on the lower bound **by construction**, independent of whether ptxas can prove the
> addresses affine — once a runtime component (pipeline stage, predicated branch) mixes into the index, merging is no
> longer guaranteed, while the pair form is unaffected.

### 2.2 Spelling

```python
# Pair view of the scalar vector: mode 0 = the two adjacent N a pair spans, mode 1 = selects the pair
sStP = cute.make_tensor(sStat.iterator,
                        cute.make_layout((2, LEN // 2), stride=(1, 2)))
v2 = cute.make_fragment(2, cutlass.Float32)        # live scalar registers: always 2

for v in cutlass.range_constexpr(cute.size(acc) // 4):
    # one 64-bit fetch feeds 4 elements
    cute.autovec_copy(sStP[None, tCcC[4 * v][1] // 2], v2)
    for sub in cutlass.range_constexpr(2):
        k = 2 * v + sub
        p_i32[k] = cvt_f16x2_f32(g(acc[2 * k],     v2[0]),
                                 g(acc[2 * k + 1], v2[1]))
```

`tCcC = thr_mma.partition_C(cute.make_identity_tensor((M, N)))`; `tCcC[i][1]` is element i's global N coordinate (a
runtime value; using it as a **slice index** is legal; using it as a **register-array index** drives the whole array
into local memory, see [`warp_specialization`](warp_specialization.md) gotcha 7).

### 2.3 Two things not to do

- **Don't prefetch into one big fragment by N coordinate**: each thread has `2·V·MMA_N` distinct N coordinates (for an
  N=128 warpgroup accumulator → 32), and two statistics means 64 resident registers — more expensive than what is
  saved. The form above keeps the resident scalars at **2**, at the cost of only one LDS per 4 elements.
- **Don't assume the M-axis hoist can simply be turned into an N-axis hoist**: when `for r: stat = mStat[r]` becomes
  `for c: stat = mStat[c]`, the inner loop body no longer accesses adjacent fragment elements contiguously (N is the
  fastest axis), and packed pairs span outer iterations. Only the two-level `(v, sub)` spelling above keeps the pair
  whole at the innermost level, so it can both hoist and pack.

> Up to here, mechanism 2 only guarantees **landing on the smem-gather lower bound**. What that bound itself is, and how
> to escape it, is in the next section.

## Mechanism 2b: move the scalars out of smem entirely — one staging per warp + `shfl.sync.idx` broadcast

> Certification status: ✅ **GPU-verified** (probe
> → all five contracts A-E PASS, final verdict `PROBE PASS max_diff=0.0`). Numeric equivalence, the pair structure, the
> two gather variants landing above the smem-gather lower bound, `WARP_BCAST`'s `wpwi` falling below half of that
> bound, `SHFL` going up and `LDS` dropping to the order of `NSTAT × JMAX` in SASS — all measured, not derived.

### 2b.1 It solves what mechanism 2 cannot

Mechanism 2's form (one 64-bit LDS per 4 elements) is already **the smem-gather lower bound**:

```
smem_wavefronts_ld / warp-iteration  ≥  LDS issued by the segment per warp-iteration  =  NSTAT × NELEM/4
```

(`NSTAT` = number of row-scalar vectors the segment subtracts/multiplies, `NELEM` = accumulator elements per thread.
Each execution of an LDS is at least 1 wavefront, even without bank conflicts.) An N=128 warpgroup accumulator with two
row scalars ⇒ lower bound = 2 × 64/4 = **32 wavefronts/warp-iteration**, and the per-element spelling after ptxas
merging **also only reaches this line**. Going further down on a read-side overrun requires switching mechanisms.

**How to measure (the criterion upgraded from "static LDS count" to the operational definition of this dynamic
quantity)**:

```
wpwi = l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
       ÷ (CTA count × consumer warps per CTA × inner iteration count)
```

The denominator is **the number of times the segment is executed** — neither the warp count nor the iteration count;
leaving out either factor gives a number that cannot be compared with the lower bound above. There are two
construction requirements for the measurement; missing either makes the number fake:

- **The address must contain a runtime component** (stage / slot rotation), otherwise the fetch is hoisted out of the
  loop;
- **Adjacent iterations must not read the same address.** ptxas can unroll dynamic loops with small trip counts, and
  two reads of the same address with no write in between make the second a legal CSE — a gather variant then measures
  **below its own lower bound**, looking as if the mechanism did not hold. The probe has each iteration read **its own
  slot** (`SLOTS == ITER`) to close this off.

### 2b.2 Mechanism: the broadcast-axis scalars exist only **once** for the whole warp

The N coordinates of the C-fragment within one warp cover the whole N axis (4 distinct N starting points per lane ×
steps of 8 per group), so **what one warp needs is exactly the whole scalar vector** — not a small piece per lane.
Hence:

1. **staging**: spread the whole vector over the warp's registers as "pair index `p` goes to lane `p % 32`, register
   slot `p // 32`". Each staging instruction is one `LDS.64` (32 lanes × 8 B = 256 B = 2 wavefronts), `JMAX =
   (N/2)/32` of them per scalar vector.
2. **broadcast**: the pair of adjacent N scalars each 4-element group needs is fetched with two `shfl.sync.idx` from
   the lane holding it.

Cost table (per warp-iteration):

| Item | Mechanism 2 (per-pair gather) | Mechanism 2b (warp broadcast) |
|---|---|---|
| LDS count | `NSTAT × NELEM/4` (32 in this example) | `NSTAT × JMAX` (4 in this example) |
| smem read wavefronts | ≥ 32 | `NSTAT × JMAX × 2` = 8 = **the information-theoretic lower bound for reading the whole vector** (2 × 128 × 4 B / 128 B) |
| SHFL.IDX count | 0 | `2 × NSTAT × NELEM/4` (64 in this example) |
| resident scalar registers | 2 | `2 × JMAX × NSTAT` (8 in this example) |

SHFL goes through the MIO/branch unit rather than the LSU; **what changes is the pipeline**, not just the name of the
cost — so this step is a net gain on a read-side (volume-type) overrun, with `stall_mio` and instruction count in the
cost column.

### 2b.3 The key: register indices must be compile-time constants; **put every runtime component into the lane operand**

This entry's gotcha "don't index a register array with a runtime value" collides head-on here: `p = q // 2` is a
runtime value, and `rStat[...]` is a register array. It works because of an arithmetic fact —

for the WGMMA C-fragment, the pair index of the `v`-th 4-element group is

```
p(v, lane) = 4v + c ,      c = lane % 4 ∈ [0, 4)
```

so `p >> 5 = (4v) >> 5` — **independent of the lane, and known at trace time** (because `4 | 4v` and the in-group
offset `c < 4`, the whole group lies within one 32-wide lane block). Therefore:

- **the register slot `j = (4v) // 32` is a compile-time constant** → a legal register-array index that does not go to
  local memory;
- **the source lane `p % 32` is runtime and differs per lane** → and `shfl`'s source-lane operand **is allowed to
  differ per lane anyway**, which is exactly where the runtime component gets placed.

This invariant must be **verified on the GPU** (this entry's probe, contract B), not taken on intuition — it depends
on the C-fragment linear order (§2.1), and must be re-verified when the tiler / atom layout changes.

### 2b.4 Spelling (exact)

```python
NPAIR = N // 2                 # number of pairs of the scalar vector
JMAX  = NPAIR // 32            # pairs held per lane (must divide evenly; N=128 -> 2)
lane  = tidx % 32

# pair view of the same smem: mode 0 = the two adjacent N one pair spans
sPair = cute.make_tensor(sStat.iterator,
                         cute.make_layout((2, NPAIR), stride=(1, 2)))
rStat = cute.make_fragment((2, JMAX), cutlass.Float32)      # 2*JMAX resident

# ---- staging: once per warp, NSTAT × JMAX LDS.64 --------------------------
for j in cutlass.range_constexpr(JMAX):
    cute.autovec_copy(sPair[None, 32 * j + lane], rStat[None, j])

# ---- broadcast: two SHFL.IDX per 4-element group --------------------------
for v in cutlass.range_constexpr(cute.size(acc) // 4):
    q   = tCcC[4 * v][1] // 2        # runtime pair index (tCcC = partition_C(identity))
    j   = (4 * v) // 32              # compile-time constant: independent of the lane (§2b.3)
    src = q % 32                     # runtime, differs per lane -- legal
    s0 = cute.arch.shuffle_sync(rStat[(0, j)], src)   # even N
    s1 = cute.arch.shuffle_sync(rStat[(1, j)], src)   # odd N
    for sub in cutlass.range_constexpr(4):
        i = 4 * v + sub
        acc[i] = g(acc[i], s0 if sub % 2 == 0 else s1)
```

`cute.arch.shuffle_sync(value, offset, mask=0xffffffff, mask_and_clamp=31)` (`shuffle_sync_op` in
`cutlass/cute/arch/nvvm_wrappers.py`, `kind=nvvm.ShflKind.idx`; `cutlass/cute/arch/__init__.py` exports
`shuffle_sync` / `_up` / `_down` / `_bfly`). If that DSL version does not export it, the equivalent raw PTX (for the
calling convention and the ladder "check the signature first, then write asm" see [`ptx_escape`](ptx_escape.md)):

```python
llvm.inline_asm(T.f32(), [val_f32, src_i32],
                "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=f,f,r", ...)
#  4th operand c = 31 -> segmask 0 / clamp 31 (all 32 lanes as one segment)
#  5th operand = membermask, the whole warp participates
```

### 2b.5 Preconditions and pitfalls

- **`NPAIR % 32 == 0`**. When it does not divide evenly, the last staging LDS has lanes out of bounds, and while the
  lane independence of `p >> 5` still holds, the range of `j` must be computed from the largest `v`. Work it out before
  using this on accumulators with N < 64.
- **`32 * j + lane` must be a pair index, not an element index**: the 8 B alignment of staging comes from "a pair's
  starting N is always even" (the premise verified in §2.1); writing an element index degenerates into two 32-bit
  gathers, and wavefronts rise instead of falling.
- **Staging must stay inside the loop body** (if the scalar vector rotates by stage/slot). With a runtime slot
  component in the address, the compiler won't hoist it; conversely, if you hoist staging out of the loop while the slot
  changes, that is a **bug**, not an optimisation.
- **Don't replace the M-axis hoist with `shfl`.** When the broadcast axis falls on the M axis, the row hoist of
  `cfrag_elementwise_fastpath` §(b) is already one scalar per row and cheaper; this entry only targets the **transposed
  accumulator** (broadcast axis on N).
- **The whole warp must be present.** The membermask of `shfl.sync` is the whole warp; if this segment is written
  inside a divergent predicate (e.g. a tail block where only some lanes participate), narrow the mask yourself,
  otherwise it is undefined behaviour.

## Mechanism 3: per-segment live-set accounting (turning the above into numbers you can reconcile)

The peak is not at the **start** of the elementwise segment but at the point where its own temporaries are most
numerous and the upstream accumulator is **not yet fully dead**. So account by "program point" rather than by
"fragment list":

```
peak point = the segment over which some MMA chain is [deliberately left in flight] (see wgmma_deferred_wait)
           = the full accumulator of the in-flight chain  A_flight
           + accumulators resident across the whole loop  A_resident
           + the [residue] of the upstream accumulator at that point  A_residue   <-- mechanism 1 directly reduces this term
           + the non-accumulator working set W (fp temporaries + addresses/bookkeeping)  <-- mechanism 2 directly reduces this term
           ≤ budget
```

Two value-taking rules when reconciling:

**Count the upstream accumulator by its [residue], not at full size.** With mechanism 1 in effect, ptxas chops up
the upstream live ranges, and the residue at the peak point usually collapses to single digits — **and ptxas does this
by itself**: the two source forms of this entry's probe (in-place packing / separate fragment) land on the same
allocation, so the residue collapse is **not** a gain earned by some spelling; don't treat "switch to in-place
packing" as an action that reduces the residue. Its use is **the right value to take when accounting**: counting at
full size gives an inflated sum, leading to the misjudgement "the overlap must be cut".

What can really be moved are the other two terms: `A_flight` (whether to overlap, see
[`wgmma_deferred_wait`](wgmma_deferred_wait.md)) and `W` (the non-accumulator working set) — `W` is often the largest
term, larger than any single accumulator. When the peak is over budget, measure `W` first; don't change the packing
spelling first.

> **How to reduce `W` once measured**: this entry stops here (it only teaches accounting). The hardware action for
> reducing `W` is a separate entry — [`uniform_reg_budget`](uniform_reg_budget.md): warp-invariant **integer**
> quantities such as addresses/bookkeeping can go into the per-warp uniform register file and not eat the per-thread
> quota of `setmaxnreg`; a single `+ tidx` mixed into the seed flips the whole chain back to regular registers. fp
> temporaries cannot go into UR; that part can only be reduced by shortening live ranges.

Accounting discipline (written into the milestone's topology contract, and also used in code review):
- For each fragment record its **live range** (from the defining instruction to the last reader), not its allocation
  size;
- For each MMA chain "left in flight", count its full accumulator in **every segment** it covers;
- W is measurable: `W = peak − Σ(accumulators alive at that point)`, not estimated.

## Acceptance (machine-read; skipping it is as good as not doing it)

- **Spill must be 0**: the `LOCAL:` / `STACK:` fields of `cuobjdump -res-usage <dump>.cubin`, or ncu's local ld/st
  sectors. The cubin **cannot** be obtained from `cute.compile(..., options="--dump-dir=…")` — the compile result object
  of the current stack does not expose the cubin, and that options string is not a working path either; the only
  usable route is to set `CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_DUMP_DIR=<dir>` **before importing cutlass**, and take the
  `.cubin` from that directory after compiling (see the gotcha of the same name in
  [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md)).
- **Don't use the static footprint of an A/B to prove "the form is better"** `[MEASURED]`. Two variants of
  the same kernel (in-place packing + pair fetch vs whole block + separate fragment + per-element fetch) measured
  **exactly the same** `(spill, REG, LDS)` (`(0, 168, 32)` vs `(0, 168, 32)`). So:
  - **The criterion is "does it fit", not "which is smaller"** — measure whether the **absolute value of `spill` is 0**;
    don't compare the difference between two variants.
  - `REG:` is **always unusable** here: 384 threads + `min_blocks_per_mp=1` ⇒ the static cap is fixed at 168 by launch
    bounds and does not vary with the working set (same as the gotcha "`regs_per_thread` cannot tell whether this entry
    was done").
  - To do an A/B, the control group is informative only under **a load that can fail**: raise "in-flight accumulators +
    resident accumulators" close to 240 before measuring; otherwise both fit easily, and the resulting "equal" only says
    the load is too light.
- **The two variants must each have their own process and their own dump directory** `[MEASURED]`. Dump
  file names are built only from the **tensor signature** of the jit function (measured shape
  `cutlass__probe_jit_Tensorgmem_o_12864641_….sm_90a.cubin`), and **the `Constexpr` variant selector does not enter the
  name**. So compiling two variants in the same process:
  - the second cubin **overwrites the first in place**, and "the difference of file sets before and after compiling" is
    empty ⇒ the second variant's footprint reads as `None`, and the A/B contract is judged negative not by the code but by
    `None` (one probe run was exactly this: FUSED read `spill=0 / reg=168 / LDS=32`, BLOCK read nothing at all);
  - worse, the same signature is also the same **compile cache key**, so it cannot be ruled out that the second
    `cute.compile` directly returned the first variant's compiled function — in which case the second variant's numeric
    contract passed **vacuously**.
  Process isolation removes both at once: no shared cache, no same-name overwrite, and each variant's dump directory
  holds only its own `.cubin`, so newest-wins fetching suffices, with no set difference and no betting on mtime.
- **In an A/B, first prove the control group is really worse**: reporting only "my version has spill=0" is empty …
  **— this item has been falsified by this entry's own probe, see the previous item**. On this entry's two forms, the
  control group is not worse; the only usable criterion is absolute spill.
- **LDS count**: `cuobjdump -sass <cubin> | grep -c '\bLDS'`. With mechanism 2 in effect it should be `2 × elements/4`
  (1/4 of the element count for each of the two statistics). **But equalling this number does not mean "written per
  pair"** — the per-element spelling after ptxas merging gives the same number (see item 2 at the top). This item only
  confirms that it **did not degenerate** into two 32-bit gathers (which would give 2× the lower-bound count); it cannot
  be used to judge the source form.
- **The criterion for read-side volume is dynamic, not the static LDS count** (this entry's earlier wording overreached
  here and has been retracted). Measure `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`, **normalised per
  warp-iteration**:

  ```
  wpwi = smem_wavefronts_ld / (CTA-iterations × consumer warps per CTA)
  ```

  Two lines, read by mechanism tier:
  - `wpwi ≳ NSTAT × NELEM/4` ⇒ still at the **smem-gather lower bound**. Spelling changes per mechanisms 1/2 have
    **identically zero gain** on it (the same instruction stream); only mechanism 2b moves it.
  - Mechanism 2b in effect ⇒ `wpwi ≈ NSTAT × JMAX × 2` (the information-theoretic lower bound of reading the whole scalar
    vector), with `SHFL` appearing in SASS and the LDS count dropping to the order of `NSTAT × JMAX`.
  The static footprint can only give a **lower bound** (count × 1), never an upper bound; conversely, `wpwi` below the
  gather lower bound is sufficient evidence that "the mechanism really changed".
- Values: the packing bijection silently misplaces things the moment it is touched, so both A/B variants must
  **simultaneously** be bit-equal to a trivial reference.

## Gotchas

- **In-place packing must traverse pairs in ascending order.** Descending or random order overwrites element `2k`
  before reading it, with no error — the result is fp16 bit patterns read as fp32, wildly wrong values.
- **Use both directions of `recast_tensor`**: write the packed result through the `Int32` view (a 32-bit slot = one
  pair), read back per element downstream through the `Float16` view (128 slots). Both views **alias the same
  registers** as the fp32 fragment and produce no move instructions.
- **Don't index a register array with a runtime value.** The N coordinate `tCcC[i][1]` may only be a **tensor slice**
  index (`sStP[None, q // 2]`); as soon as it indexes a `cute.make_fragment`, ptxas drives the whole array to local
  memory, and the spill count explodes regardless of the budget. Mechanism 2b is the **legal workaround** for this
  pitfall: put the runtime component into `shfl`'s source-lane operand and use only the compile-time constant `j` for
  the register slot (§2b.3) — not an exception, just a different place to put the runtime component.
- **For `autovec_copy` to become one 64-bit LDS, the source must be truly contiguous and aligned**: the mode-0 stride
  of the pair view `make_layout((2, L//2), stride=(1, 2))` must be 1, and the pair start index must be even. Writing
  `stride=(L//2, 1)` turns it into two 32-bit gathers — no fewer instructions, plus an extra layer of address
  arithmetic.
- **This entry and `warp_specialization` are conditions in series**: when `setmaxnreg` is not in effect (missing
  `min_blocks_per_mp`), the consumer has only 168, and doing this entry right still spills; conversely, with 240 in
  hand, the wrong form still spills. **First use whether PTX has `.minnctapersm` to tell which one it is**; don't tune
  the two things mixed together.
- **`regs_per_thread` cannot tell whether this entry was done.** It is a CTA average, and with an asymmetric split it
  gives the same number as "no split"; the only criterion is the spill count.
- **Don't serialise the two MMA chains for the sake of "seeming to have fewer fragments".** The 64 registers of the
  in-flight chain are the **price** of overlap, not waste; the right action is to reduce `A_residue` and `W`, not to
  cancel the overlap (cancelling it costs more tensor utilisation than it saves, see
  [`wgmma_deferred_wait`](wgmma_deferred_wait.md)).
- **`recast_tensor` aliasing is free, but only once, in the packing direction.** Opening another `Int32` view on a
  fragment that has already served as a wgmma A operand to do an after-the-fact RMW makes the whole fragment spill to
  local memory under a real register budget, and serialises the whole function's wgmma along with it (the table of
  §1.4). The legal form for correcting the packed result is correcting the fp32 source and repacking; both views must be
  built only inside the `const_expr` branch that uses them.
- **Registers freed by a transient fragment cannot pay for a resident quantity: account by peak, not by count.** A
  32-element f32 factor fragment computed in the first elementwise segment, carried across a chain of 8 wgmma + its
  drain + two mbarrier waits, and reused in the second elementwise segment (saving 32 `MUFU.EX2`/thread/iteration), in a
  role at the 104 tier: `spill_ld/st` 132,096/2,048 → 950,272/623,104, gate 0.4141 → 0.488 ms (+17.9%), `tensor_pct`
  12.57 → 10.68 (KEEP). The 32 staging registers just "freed" by the previous item are exactly that
  number — but staging lives only within one narrowing, while the resident factors span the whole chain; the two peaks
  are not at the same point, **equal counts ≠ cancelling peaks**. Under this structural condition (role at the 104 tier,
  accumulators already filling the headroom while the chain is in flight), recomputing `MUFU.EX2` 32 times is cheaper
  than keeping 32 registers resident; residency can only be positive when the role has ≥ 32 of **peak** headroom while
  the chain is in flight. Moving registers over from the paired pole role (tiers 104/192 → 120/184) only halved the spill
  (520,192/520,192) while the gate got worse (0.4873 → 0.5059, KEEP) — see the transfer granularity
  of [`warp_specialization`](warp_specialization.md) gotcha 8 and gotcha 12.
- **Temporaries borrow the registers of dead fragments: `cute.make_tensor(dead_frag.iterator, layout)`, not a separate
  fragment.** Within the same role body, accumulators and exit-face fragments that have already been packed / written
  out at some point and will only be reopened in the next round by a wgmma (`ACCUMULATE=False`) or STSM have dead
  registers at that point; the new segment's temporary accumulators, f32 staging and 16-bit exit faces can each land
  directly on those registers with `cute.make_tensor(<dead fragment>.iterator, <this segment's layout>)` (asserting equal
  element counts at trace time), with the instruction order unchanged. Measured on the same code: parallel declarations
  → spill 7,022,592/7,022,592 (KEEP); borrowing dead fragments → spill 0 (KEEP).
  This is the zero-cost spelling when mechanism 3's ledger has "new segment ≤ upstream dead fragments"; when the ledger
  does not balance (the new segment is larger than all dead fragments combined), it cannot help. It **only balances the
  register ledger**: the borrowed segment's own busy time is still added to the role's cycles
  ([`warp_specialization`](warp_specialization.md) gotcha 12).

## Sources

- **Official / public infra**: `recast_tensor` / `make_tensor` / `autovec_copy` of CuTe DSL `cutlass/cute/core.py`;
  `setmaxregister_*` of `cutlass/cute/arch/nvvm_wrappers.py`; PTX ISA `cvt.rn.f16x2.f32` (packing semantics: the first
  source operand goes into the high half, so the asm is written `$0, $2, $1`).
- **Measured footprints and cost-magnitude data** (live-range breakdown, schedule order, spill cost, magnitude
  measurements of read-side volume, etc.) are not reproduced in this entry. Of these, the attribution "reading scalars
  per element vs per pair makes a difference" has been falsified by this entry's probe (the two spellings compile
  identically); the real cause of the read-side volume overrun has been located as **the smem-gather lower bound
  itself** (the dimension of ≥1 LDS per 4 elements), with mechanism **2b** as the way out and the dynamic `wpwi` as the
  criterion.
- **API source of mechanism 2b**: `shuffle_sync_op` of `cutlass/cute/arch/nvvm_wrappers.py` (signature
  `(value, offset, mask=FULL_MASK, mask_and_clamp=WARP_SIZE-1, kind)`, lowering to `nvvm.shfl_sync`; 32-bit issued
  directly, 64-bit split into two), `cutlass/cute/arch/__init__.py` exports `shuffle_sync` / `shuffle_sync_up` / `_down`
  / `_bfly`; PTX ISA `shfl.sync.idx.b32 d, a, b, c, membermask` (low 5 bits of `c` = clamp, high bits = segmask). "Each
  execution of an LDS is at least 1 smem wavefront" = the definition of `l1tex__data_pipe_lsu_wavefronts_*` (the number of
  pipeline cycles the LSU spends serving one request, a different quantity from the replay count of
  `..._bank_conflicts_*`).
- **GPU-certification probe**: a 384-thread CTA (1
  producer stub + 2 consumer WGs, 24/240), two WGMMA chains with the second deliberately left in flight, plus a
  64-register **resident accumulator block** (seeded from gmem, not with `fill(0.0)`: a zero fill would be
  constant-folded, and those 64 registers would never be held across this segment). FUSED (in-place packing + N-axis
  pair-vector fetch) and BLOCK (whole block + separate full-size fragment + per-element gather) differ only in these two
  places of this entry, **with each variant running in its own subprocess + its own dump directory** (reason in
  "Acceptance"). Six contracts: the C-fragment linear-order assumption (adjacency **+ pair start N even**, the latter
  being the premise of a 64-bit aligned fetch), each of the two variants bit-equal to the numpy reference, the two
  cubins each readable and different from each other, **the budget ledger** (in-flight 64 + resident 64 + residue + W
  fit into 240, `spill(FUSED) == 0`), **the fetch lower bound** (`LDS(FUSED) ≤ 2 × elements/4`).
  - A correction to the contract design: two contracts were once written as lexicographic `spill/REG` FUSED < BLOCK and
    `LDS(FUSED) < LDS(BLOCK)`, and in one probe run **both came out tied** (`(0,168,32)` vs `(0,168,32)`). That is not a
    measurement mistake but the conclusion — the peak is fixed by "the fp32 accumulators of two chains alive at once",
    independent of the packing spelling; the static LDS count is flattened by ptxas's merging. The two contracts were
    changed accordingly into criteria on **absolute values**.
  - Numbers read on the GPU: linear-order assumption zero violations over 4096 groups, both variants numerically
    `max_diff = 0`, `spill = 0 / REG = 168 / LDS = 32` (the same for both variants) `[MEASURED]`.
  - **Certification conclusion**: all six contracts PASS on the GPU, final verdict `PROBE PASS max_diff=0.0`. Mechanisms
    1 / 2 / 3 = ✅ Verified (marked accordingly in index.md).
- **GPU-certification probe of mechanism 2b**
  — one warpgroup (128 threads / 1 CTA), a real C-fragment from `partition_C` (M=64, N=128, 64 elements per thread → 16
  4-element groups, `j` ranging over {0,1}), two row-scalar vectors, a **dynamic** loop of `ITER=8` (`cutlass.range`),
  each iteration reading **its own slot** (`SLOTS == ITER`; the address has a runtime component ⇒ hoisting is impossible,
  and adjacent iterations use different addresses ⇒ no cross-iteration CSE under any unroll factor, see "How to measure"
  in §2b.1). The three variants differ only where the scalar is fetched: `GATHER_ELEM` (per-element LDS) /
  `GATHER_PAIR` (mechanism 2's per-pair 64-bit) / `WARP_BCAST` (mechanism 2b). Each variant has **its own subprocess +
  its own dump directory + its own compile-cache directory + the variant compiled into the tensor signature** (the
  `mTag` shape), isolating the pitfall "the Constexpr selector enters neither the cubin name nor the compile cache key";
  the subprocesses run under `ncu --profile-from-start off`, with each subprocess opening the profiling window on **its
  single launch** (JIT compilation left outside the window), so the CSV has only one kernel row. The reference values
  presuppose no C-fragment layout: the device writes back **its own (m, n) coordinates**, and the host rebuilds the
  numpy reference from those coordinates, so a wrong layout assumption is caught only by contract B and cannot quietly
  change the reference. Five contracts: A the three variants bit-equal to the numpy reference; B the pair structure of
  the 4-element groups, pair start N even, and the lane independence of `q >> 5`, with zero violations per lane, and the
  warp's N coordinates covering the whole axis; C the `wpwi` of both gather variants ≥ the lower bound
  `NSTAT × NELEM/4 = 32`; D `WARP_BCAST`'s `wpwi` strictly below half of that bound and ≤ 1/3 of both gathers; E in
  `WARP_BCAST`'s SASS `SHFL ≥ 2 × groups` and `1 ≤ LDS ≤ NSTAT × JMAX × 2`.
  - **Two engineering constraints on the probe side** (unrelated to the mechanism, but without them no verdict comes
    back): ① `cutlass.Constexpr` parameters are passed only to `cute.compile` and **must not be passed again to
    compiled(...)** (the calling convention of a compiled CuTe DSL kernel) — passing them is an arity error, the subprocess
    dies before any measurement, and all five contracts report errors together with `max_diff=n/a`, looking like a
    mechanism problem when it is not; this probe therefore puts the variant selector in a **module-level global**, so the
    argument lists of `cute.compile(fn, *args)` and `compiled(*args)` are identical. ② `gpu_entry.sh` SIGKILLs at
    `TASK_TIMEOUT`, and a killed job has **not a single verdict line** (= NO VERDICT, much worse than a FAIL that names
    names); the parent therefore keeps its own books, trims subprocess timeouts to the remaining budget, and reports what
    does not fit as a FAIL with a reason. FAIL lines uniformly end with a parenthesis stating the contract name and the
    measured value.
  - **Certification conclusion**: all five contracts A-E PASS on the GPU, final verdict `PROBE PASS max_diff=0.0` — the
    three variants bit-equal to the numpy reference (A), the pair structure and the lane independence of `q >> 5` with
    zero violations (B), the `wpwi` of both gather variants ≥ the lower bound 32 (C), `WARP_BCAST`'s `wpwi` strictly
    below half of the bound and ≤ 1/3 of both gathers (D), `WARP_BCAST`'s SASS `SHFL ≥ 2 × groups` and
    `1 ≤ LDS ≤ NSTAT × JMAX × 2` (E). Mechanism 2b = ✅ Verified (marked accordingly in index.md); from then on "the
    read-side lower bound = 1 LDS per 4 elements" holds only for forms that **keep fetching scalars from smem**.
- Readings of §1.4: an isolated probe (five contracts numeric_form0 / numeric_form1 / mapping_discriminates /
  no_spill / shorter_form1 PASS, proving the spelling itself is legal and does not spill in isolation) + the NCU/SASS
  dossier census of the driver evaluation (two versions of the same kernel with/without the second view: spill, tensor,
  `commit` axis, gate window). Status: ✅ KEEP-certified (boundary clause).
- **fused narrowing == staged narrowing** (the premise of the reading "deleting staging saves no registers"): GPU probe
  (four contracts
  `PROBE PASS max_diff=0.0`: the fp16 face and the bf16 face each bit-equal to the wide-precision result of the same
  algebra (rounded to the target format), and each bit-equal to the old path via f32 staging); deleting staging was KEPT
  (spill from 0 to positive, gate +0.6%).
- KEEPs for **resident factors / borrowing dead fragments / tier transfer**: 32 resident factors (spill
  reappears, gate +18%); parallel declarations (seven-digit spill) → borrowing dead fragments
  (spill 0); tier transfer (the receiving side's spill halved, gate slower).
