# cfrag_elementwise_fastpath (the shortest instruction form of an elementwise segment over the WGMMA C-fragment)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> ⚠️ **This entry only governs instruction count — not the live range, and not read-side volume.** The
> row hoist of §(b) below assumes the broadcast axis falls on the accumulator's **M** axis (not true for a
> transposed accumulator), and the example in §(c) allocates a separate **full-size** fp16 fragment and
> "computes the whole block, then packs the whole block" — **copying those two places verbatim spills
> under a 240-register budget.** The live-range form (in-place packing that ends the upstream fp32,
> pair-vector fetches of N-axis broadcast scalars, per-segment live-set accounting) is in
> [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md); the two entries must be satisfied
> **together**.
>
> ⚠️ **This entry's "≤ 3 instructions/element" is not a read-side volume criterion.** When the broadcast
> axis falls on the **N** axis (transposed accumulator), fetching the row scalar from smem per element
> **fits exactly into this budget** (1 LDS + 1 FFMA + 1 EX2 = 3/element), but its `smem_wavefronts_ld` is
> held up by a lower bound of `NSTAT × NELEM/4` — **the instruction-count criterion cannot see this
> term**. The correct way out for that term is mechanism **2b** of
> [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) (one staging per warp + `shfl.sync.idx`
> broadcast), with the criterion switched to the dynamic `wpwi`. See §(e).
>
> ⚠️ **This entry also does not govern the issue order of these instructions.** A segment whose
> per-element counts and live range are on target can still be stuck as a whole on
> `short_scoreboard`/`mio_throttle` (producers right next to consumers, each MIO result read the moment it
> returns). For the order-and-spacing dimension see
> [`ew_segment_latency_schedule`](ew_segment_latency_schedule.md).

## Purpose
Get an elementwise segment over the WGMMA f32 accumulator (C-fragment) — "scale + subtract a
row-broadcast statistic + `exp2`/multiply-add + f32→f16 conversion feeding the next RS WGMMA" — emitted
by ptxas as the shortest SASS block: **per element ≈ 1 FFMA + 1 MUFU.EX2 + 0.5 F2FP (pair-packed
conversion)**, with zero fragment rearrangement throughout (every view is a pure layout composition over
the same registers).

## When needed (trigger)
- `sm__pipe_tensor_cycles_active` share is below the reference, and a wait-flip A/B shows "draining all
  GEMMs is actually fastest" — i.e. the elementwise block itself (not the pipeline wait) is exposed as
  the critical path.
- In SASS the blocks between HGMMA batches are significantly longer than ~3 inst/element: several
  `F2F`/`H2F`/`I2F`/`IMNMX`/`MOV`/`LDG` per element (scalar conversions, re-fetching the statistic per
  element, per-element integer ops all leave this footprint).
  - **`SHFL` is not in this footprint list** (an earlier version listed it; it has been removed). When the
    broadcast axis falls on the M axis it is indeed waste (a row hoist is cheaper); when the broadcast
    axis falls on the **N** axis, `SHFL` is exactly the footprint of the **correct way out** — mechanism
    2b trades `NSTAT × NELEM/4` `LDS` for `2 × NELEM/4` `SHFL.IDX`; **what changes is the pipeline, not
    the name**, and the static instruction count is roughly flat (measured difference < 1
    instruction/element). Treating `SHFL` as a symptom per the old list would rule out the only option
    that can cut read-side volume (see §(e)).
- There is already a bijection feeding an f16 fragment to the RS WGMMA, but the conversion/statistic
  code is written scalar by scalar.

## Minimal runnable example

### (a) Two zero-cost views (pure layout composition, no data movement)

```python
# MN view: ((2,2,V), MMA_M, MMA_N) -> rows x columns ((2,MMA_M), (2,V,MMA_N))
# Source: quack/layout_utils.py:168-205 (public infra)
def convert_layout_acc_mn(acc_layout):
    acc_cm = cute.make_layout(acc_layout.shape)          # reference col-major
    shape = ((acc_cm.shape[0][1], acc_cm.shape[1]),
             (acc_cm.shape[0][0], *acc_cm.shape[0][2:], acc_cm.shape[2]))
    stride = ((acc_cm.stride[0][1], acc_cm.stride[1]),
              (acc_cm.stride[0][0], *acc_cm.stride[0][2:], acc_cm.stride[2]))
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))

# frgA view: ((2,2,V), MMA_M, MMA_N) -> A-operand order ((2,2,2), MMA_M, (V/2, MMA_N))
# Source: quack/layout_utils.py:207-250
def convert_layout_acc_frgA(acc_layout):
    div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
    l = cute.logical_divide(acc_layout, ((None, None, div), None, None))
    return cute.make_layout(
        ((l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
         l.shape[1], (l.shape[0][2][1], l.shape[2])),
        stride=((l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1], (l.stride[0][2][1], l.stride[2])))
```

### (b) Row-statistic hoist + fully unrolled inner loop (the inner body is just FFMA + EX2)

> ⚠️ The code below **holds only when the broadcast axis falls on the M axis**. With a transposed
> accumulator (broadcast axis on N) a row hoist gets no reuse and degenerates into per-element smem
> re-fetches. **Do not stop at "fetch it with a different spelling" then** — mechanism 2 of
> [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) (64-bit fetch per pair) compiles to the
> same instruction stream as the per-element form with identical read-side volume; **it only pins you to
> the smem-gather lower bound**; actually cutting the read side requires mechanism **2b** (per-warp
> staging + `shfl.sync.idx`).
> In other words: M axis → §(b) here; N axis → mechanism 2b, with mechanism 2 only the lower-bound
> reference for 2b.

```python
acc_mn  = cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))
tCcC_mn = cute.make_tensor(tCcC.iterator, convert_layout_acc_mn(tCcC.layout))
# tCcC = thr_mma.partition_C(cute.make_identity_tensor((M, N))): static coordinates
for r in cutlass.range_constexpr(cute.size(acc_mn, mode=[0])):
    stat = mStat[tCcC_mn[r, 0][0]]        # row statistic: fetched ONCE per row (hoist)
    for c in cutlass.range_constexpr(cute.size(acc_mn, mode=[1])):
        acc_mn[r, c] = cute.math.exp2(acc_mn[r, c] * SCALE - stat,
                                      fastmath=True)   # 1 FFMA + 1 MUFU.EX2
```
(The inner loop can also be fully unrolled with `cutlass.range(..., unroll_full=True)`, equivalent to
`range_constexpr`. An inner loop with a runtime trip count serialises the in-flight WGMMAs — see
[`wgmma_deferred_wait`](wgmma_deferred_wait.md) run C — so static bounds are always fully unrolled.)

### (c) Pack f32→f16 (1 F2FP per pair) + feed the RS WGMMA with zero rearrangement

> ⚠️ The `rP = cute.make_fragment(acc_fA.shape, Float16)` below **allocates a separate full-size
> fragment** that coexists with acc's fp32 at the same program point. The instruction count is right, the
> live range is wrong. In-place packing (writing back into acc's own slots, ending the upstream fp32 on the
> spot) is mechanism 1 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md).

```python
@dsl_user_op
def cvt_f16x2_f32(a, b, *, loc=None, ip=None) -> cutlass.Int32:
    # inline PTX wrapper: cvt.rn.f16x2.f32 (packing semantics per the PTX ISA; general calling convention in ptx_escape form 3)
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))

acc_fA = cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))
rP     = cute.make_fragment(acc_fA.shape, cutlass.Float16)
rP_i32 = cute.recast_tensor(rP, cutlass.Int32)
for i in cutlass.range_constexpr(cute.size(rP_i32)):
    rP_i32[i] = cvt_f16x2_f32(acc_fA[2 * i], acc_fA[2 * i + 1])

# Alias directly as the RS WGMMA's A operand (no rearrangement): a compact
# col-major fragment of frgA shape has the same memory order as a compact col-major
# one of acc shape, so the same registers can be reinterpreted with the mma's A-fragment shape:
tAcA  = thr_mma2.partition_A(cute.make_identity_tensor((M, K2)))
rP_mma = cute.make_tensor(rP.iterator, cute.make_layout(tAcA.shape))
warpgroup.fence()                       # an RMEM A operand requires a fence before issue
for kb in cutlass.range_constexpr(cute.size(rP_mma, mode=[2])):
    cute.gemm(mma_atom2, acc2, rP_mma[(None, None, kb)],
              tBrB2[(None, None, kb)], acc2)
```

### (d) Machine-read SASS footprint (certification/regression criteria)

**First state the scope of the criteria clearly: (d) is a latency/instruction criterion; it measures
"how many instructions the same number of elements took". It has no resolving power over read-side
volume (`smem_wavefronts_ld`) or over the register live range.** The broadcast-axis scalar term must be
**carved out separately** per §(e) and must not be merged into these three counts.

- `MUFU.EX2` count == elements per thread (exactly 1 per element, fastmath SFU path).
- `F2FP` count == elements per thread / 2 (packing succeeded; == element count means it degenerated into
  scalar conversions).
- Non-HGMMA instructions between HGMMA batches / element count ≤ ~3 — **with the fetch instructions of
  the broadcast-axis scalars already subtracted from this ratio** (see §(e): that term is checked
  separately by (e)'s dynamic criterion, not by count). ≥6 means a per-element slow path has crept in.

### (e) The broadcast-axis scalar term: carve it out, switch the criterion to the dynamic `wpwi`

The three counts of §(d) cannot police this term, because it is **inside the budget but not in the
budget's units**:

| Form | Fetch instructions | §(d)'s ledger | Actual read-side volume |
|---|---|---|---|
| M axis, row hoist (§(b)) | one per **row** | ~0/element | negligible |
| N axis, per element / per pair from smem (mechanism 2) | `NSTAT × NELEM/4` LDS | **≤3/element, passes** | `wpwi ≥ NSTAT × NELEM/4`, **at the lower bound** |
| N axis, per-warp staging + `shfl` (mechanism 2b) | `NSTAT × JMAX` LDS + `2 × NELEM/4` SHFL | **flat with the previous row** (difference < 1/element), and carries `SHFL` | `wpwi ≈ NSTAT × JMAX × 2` (information-theoretic lower bound) |

⇒ **The static instruction count cannot separate the two**: it gives the per-element gather a pass (it
falls into the same budget envelope as the §(b) row), and the only static signal that does separate
them — the presence of `SHFL` — points at the **correct way out**. This term is not "the criterion lacks
precision"; it is **the wrong units**. So its regression criterion is not a count but

```
wpwi = l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
       ÷ (CTA-iterations × consumer warps per CTA × times this segment is executed)

wpwi ≳ NSTAT × NELEM/4   ⇒ still at the smem-gather lower bound; rewrites per mechanisms 1/2 are identity with zero gain,
                            the only lever that moves it is mechanism 2b of cfrag_fused_pack_liveness
wpwi ≈ NSTAT × JMAX × 2  ⇒ 2b in effect (SHFL goes up in SASS, LDS drops to the order of NSTAT × JMAX)
```

`NSTAT` = number of row-scalar vectors the segment subtracts/multiplies, `NELEM` = accumulator elements
per thread, `JMAX = (N/2)/32` = pairs held per lane. The mechanism, exact spelling and preconditions
(`NPAIR % 32 == 0`, staging must not be hoisted out, the whole warp must be present) are in
[`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) §2b.

**Order in which to use the criteria** (so the two criteria do not cancel each other out): first use
§(d) to confirm the element body is the shortest form (EX2 / F2FP / rest ≤3), then use (e)'s `wpwi` to
check the broadcast-axis scalar term separately. **§(d) passing ≠ this segment is acceptable**; only both
passing is.

## Gotchas
- **`cvt.rn.f16x2.f32 d, a, b` puts a in the high half and b in the low half.** The asm operands must be
  written `$0, $2, $1` so that (src[2i], src[2i+1]) land as (lo, hi) — on little-endian, 2i is the first
  f16 in memory. Writing `$0, $1, $2` makes the values wildly wrong immediately (the downstream GEMM
  blows up entirely).
- **`cvt.rn.bf16x2.f32 d, a, b` has the same order as f16x2: a high half, b low half; the asm is likewise
  `$0, $2, $1`.** The `cute.arch.cvt_f32x2_bf16x2` exported by this DSL version takes a two-element
  TensorSSA; if the source is a fragment produced by a store atom's `retile(acc)` (its layout is the
  atom's, not compact), then fetching scalars pairwise by index has to go through the scalar-pair form of
  inline PTX (the same spelling as for fp16, only the instruction name changes). Getting the half order
  backwards **raises no error and produces no inf/nan**: swapping the halves of each 32-bit word =
  swapping adjacent column pairs, so the downstream result is wrong but finite and everything looks
  normal. GPU-certified (by a GPU probe, not included in this release): four contracts `PROBE PASS max_diff=0.0` — the bf16 face being bit-equal to the wide-
  precision result of the same algebra (rounded to bf16) pins the half order; fused is bit-equal to the
  old path via f32 staging.
- **Narrowing pairwise directly saves neither registers nor time.** The same code with the f32 staging
  fragment deleted and pairwise `cvt` writing straight into the 16-bit fragment: KEEP,
  gate +0.6%, spill went from 0 to positive. It is a choice of **spelling**, not a register lever — for
  the accounting see item 1 of "two things this entry does not claim" in
  [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md).
- Packing requires an even number of source elements; when the atom's N dimension V=N/8 is odd, frgA's
  div can only be 1 (handled in the layout conversion).
- **Per-scalar `F16(x)` doubles the block length**: 1 conversion per element + pack moves; an F2FP count
  == element count is the machine-read footprint of this degeneration.
- **Per-element slow paths in the inner loop dominate the block length**: re-fetching the statistic per
  element (LDG), integer max/I2F, coordinate lookups, selects — each one is multiplied by the element
  count. Hoist row-broadcast quantities into one scalar in the row loop; terms that must stay in the inner
  loop are folded into a single FFMA in the `exp2` argument.
- **Don't use "≤3 instructions/element" to give a pass to per-element smem scalar fetches on the N
  axis** — this is the most commonly misused point of this entry. `1 LDS + 1 FFMA + 1 EX2 = 3/element`
  **fits exactly into the budget**, and (d) will pass it, while its `smem_wavefronts_ld` sits right at the
  `NSTAT × NELEM/4` lower bound, and ptxas's coalescing **can only reach, never go below** that line. The
  instruction-count criterion **has no resolving power** on this term (not imprecise — the wrong units):
  attach `wpwi` separately per §(e).
- **Nor use the presence of `SHFL` to judge negatively.** Under N-axis broadcast, `SHFL` is the
  footprint of mechanism 2b, i.e. what the correct answer looks like; it moves the cost from the LSU to the
  MIO/branch unit (the cost column is `stall_mio` and instruction count), and is a net gain when read-side
  volume is over the line. `SHFL` is only waste when the broadcast axis falls on the **M** axis and a row
  hoist was possible.
- Without `fastmath=True` it takes the non-SFU slow path (several instructions instead of 1 MUFU.EX2).
- If the same f32 value goes through **two** elementwise segments (the second still needs it), keep it in
  acc's f32 registers to pass it along; converting to f16 and expanding back with `H2F` = 2 extra
  conversions per element for nothing.
- If the f16 fragment has **two consumers** (RS WGMMA + R2S copy), both call-sites must use the same
  element-order bijection (both starting from frgA order; R2S uses `thr_r2s.retile(rP)`); building the
  order separately on each side silently misaligns.
- The mn / frgA views alias the same registers: once the conversion is done, acc can be reused/overwritten
  immediately, no copy needed.
- **Getting the cubin when checking the machine-read footprint**: the object returned by `cute.compile`
  on the current stack (`CudaDialectJitCompiledFunction`) exposes neither `_state` nor `__cubin__`; to get
  SASS you must set `CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_DUMP_DIR=<dir>` **before compiling**, take the
  `.cubin` from the dump directory after compiling (distinguish multiple compiles by mtime), then run
  `cuobjdump -sass`.

## Sources
- Public infra: `quack/layout_utils.py:168-250` (the two layout conversions).
- PTX ISA `cvt.rn.f16x2.f32` (packing semantics: a→high half, b→low half).
- Source of §(e) and of the two `SHFL`-related revisions:
  [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) §2b (mechanism, exact spelling, the
  operational definition of `wpwi` and the two measurement-construction requirements), GPU-certified; the
  definition of the ncu metric `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` (the number of
  pipeline cycles the LSU spends serving one request; each execution of an LDS is at least 1 wavefront).
- Certification probe of §(e) (**✅ GPU-certified: `PROBE PASS max_diff=1.19209e-07`, all four
  contracts A-D pass**)
  — one warpgroup (128 threads / 1 CTA), a real `partition_C` C-fragment (M=64, N=128, 64 elements per
  thread), one row-scalar vector (`NSTAT=1`), an 8-iteration outer loop of `cutlass.range(ITER)`, each
  iteration reading its own slot (`SLOTS == ITER`; the address has a runtime component ⇒ the fetch stays
  in the loop and cannot be hoisted; adjacent iterations use different addresses ⇒ no cross-iteration CSE
  is possible under any unroll factor). The three variants differ only where the scalar is fetched:
  `HOISTED` (fetched once per iteration, i.e. the M-axis form blessed by §(b), used as the **measured
  calibration baseline** of the budget envelope) / `PER_ELEM` (per-element LDS on the N axis, the form
  §(d) lets through, the same spelling as this repo's per-element segments) / `WARP_BCAST` (mechanism
  2b). The loop body is taken from the largest back-edge interval in SASS, and **every per-element count
  is divided by the loop-body copy count SASS reports itself**, `U = EX2/NELEM` (no `unroll=` kwarg is
  assumed, and ptxas-side unrolling cannot quietly multiply the per-element counts either). Each variant
  has its own subprocess + its own cubin dump directory + its own compile-cache directory, and the
  variant number is also put into the tensor signature (the extent of `mTag`), so the on-disk JIT cache
  cannot serve one variant's code for another; `ncu --profile-from-start off`, with the profiler window
  enclosing only that one launch (JIT compilation outside the window), and when the ncu side fails it
  falls back to rerunning without ncu so contracts that do not need ncu are not dragged down. The
  reference values presuppose no C-fragment layout: the device writes back its own (m, n) coordinates, and
  the host rebuilds the numpy reference from them; scalars for adjacent n are never equal (step 7 mod 9),
  so broadcasting from the wrong lane / the wrong half of a pair cannot pass silently.
  Four contracts: **A** all three variants reproduce their respective reference values (numeric); **B**
  in all three loop bodies `MUFU.EX2` is exactly `== NELEM × U` (exactly 1 per element, with the same `U`
  for all three) ⇒ **(d)'s exact-count item treats the three forms identically**; **C** `PER_ELEM`'s
  instructions/element do not exceed `HOISTED + 1.5` (inside the same "≤ ~3" envelope) and differ from
  `WARP_BCAST` by ≤ 1 per element (**the budget cannot separate these two**), while `SHFL` appears only
  on the `WARP_BCAST` side and `≥ 2 × NELEM/4` ⇒ **the only static signal that separates them points at
  the correct way out**; **D** `wpwi(PER_ELEM) ≥ NSTAT × NELEM/4` (at the lower bound),
  `wpwi(WARP_BCAST) ≤ half of that` (lower bound broken), and `wpwi(HOISTED) ≤ wpwi(WARP_BCAST)`
  (control group: the metric really measures the scalar-read term) ⇒ **the dynamic quantity separates
  the three, in the right order**.
- **Calling-convention pitfalls when writing the probe (an observed failure: the first two rounds of this
  entry's probe died here, unrelated to the mechanism)**: `cutlass.Constexpr` parameters are baked into
  the compiled artifact by `cute.compile` and **must not** be passed again when calling the compiled
  result (the argument lists of `cute.compile(fn, *args)` and `compiled(*args)` must match item for
  item); variant selectors must go through module-level globals read at trace time. Likewise there is no
  evidence that `cutlass.range(...)` has an `unroll=` kwarg on this stack; device-side trace-time branches
  always go through `cutlass.const_expr(...)`, and a ternary expression is turned into a device select by
  the DSL's AST processing — use a Python list with a Python int index instead.
- **GPU-certification probe for the bf16 half order + fused == staged**:
  four contracts
  `PROBE PASS max_diff=0.0`; the "direct pairwise narrowing saves no registers" reading = a KEEP evaluation.
