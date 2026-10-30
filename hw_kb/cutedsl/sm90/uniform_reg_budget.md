# uniform_reg_budget (moving the **non-accumulator working set W** into the uniform register file: warp invariants do not eat the per-thread budget)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is the **downstream action entry** of mechanism 3 of
> [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md). That entry splits the peak into
> `A_flight + A_resident + A_residual + W ≤ budget` and points out that **`W` is usually the largest term** (measured
> on the reference `W = 97`, larger than any single accumulator), but it **only teaches how to measure W, not how to
> push W down**. This entry covers only the one hardware fact for pushing W down: **SM90 has a second register file
> (the uniform register file, UR); integer quantities that are invariant within a warp live there and do not take
> the per-thread quota that `setmaxnreg` gives you.**

## Purpose

The SM90 register quota is **per thread**: a consumer warpgroup getting 216 means 216 **regular registers (R)**. But
since Turing (sm_75) the hardware also has an **integer-only uniform datapath** with a set of **per-warp** uniform
registers (`UR*` in SASS); one UR costs only `1/32` of a regular register in occupancy, and is effectively **not
charged** in the `setmaxnreg` bill.

So the same batch of "address arithmetic / loop bookkeeping / stage index / descriptor offset" values lands in two
completely different ledgers, depending on **whether it is a warp invariant**:

| Form of the value | Lands in | Eats the 216 quota? |
|---|---|---|
| derived from `bidx` / `warp_idx` / compile-time constants, the same in every lane of the warp | `UR*` (uniform file) | **no** |
| has `tidx` / `lane_idx()` folded in (even a single addition) | `R*` (regular file) | **yes, one copy per thread** |

**One `+ tidx` is enough to flip a whole bookkeeping chain from "free" to "charged against 216"** — the cheapest
cause of a W overrun, and the easiest to commit unintentionally. This entry pins that boundary down as a
machine-readable criterion.

## When needed (trigger)

Self-check in order; **if either of the first two fails, do not read mechanism 1** (the cause is not here, and
changing it will not move anything):

1. **First prove that `setmaxnreg` really takes effect** (otherwise the consumer never got 216, and talking about
   pushing W down is empty).
   Machine-readable criterion (PTX + SASS, see "Self-check 0" below): `.reqntid <N>` + `.minnctapersm 1` +
   `setmaxnreg.dec…<producer>` + `setmaxnreg.inc…<consumer>` **all four present together**.
   Missing `.minnctapersm` ⇒ ptxas silently drops `setmaxnreg`
   (gotcha 1 of [`warp_specialization`](warp_specialization.md)); all four present but still spilling ⇒ check
   gotcha 11 first (budget move placed too early).
2. **Then redo the bill per mechanism 3 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)**, in
   particular measuring `W = peak − Σ(accumulators live at that point)`. The accumulator terms are the price that must
   be paid (in-flight chains are the cost of overlap, not waste).
3. If at this point `spill_ld / spill_st > 0` still, and `W` is clearly larger than "what the fp temporaries
   themselves should need" ⇒ the excess is most likely **address/bookkeeping quantities that could have been warp
   invariants but were written per-thread**. That is this entry.

> Conversely, **this entry does not solve**: the accumulators themselves overrunning (that needs cutting overlap or
> changing the tile, see [`wgmma_deferred_wait`](wgmma_deferred_wait.md)), or too many fp intermediates
> (UR is an **integer** datapath; floating-point values cannot go there).

## Self-check 0: was `setmaxnreg` silently dropped (machine-readable, do this first)

```python
import os
# must come before import cutlass (the same-named gotcha in cfrag_elementwise_fastpath.md):
os.environ["CUTE_DSL_DUMP_DIR"]   = d
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_KEEP_PTX"]   = "1"
import cutlass, cutlass.cute as cute
...
cute.compile(jit_fn, *args)      # after compiling, take the .ptx / .cubin from d/
```

```bash
grep -c '\.reqntid 384'            *.ptx     # generated automatically by launch(block=[384,1,1])
grep -c '\.minnctapersm 1'         *.ptx     # generated only when min_blocks_per_mp>0  <<< the number-one gotcha
grep -cE 'setmaxnreg\.dec\S* 72'   *.ptx
grep -cE 'setmaxnreg\.inc\S* 216'  *.ptx
cuobjdump -sass *.cubin | grep -cE '\bLDL\b|\bSTL\b'      # the absolute criterion for spills, must be 0
```

All four hit and `LDL/STL` still > 0 ⇒ the budget arrived, the **working set** does not fit ⇒ read on.
`.minnctapersm` missing ⇒ stop here and go back to gotcha 1 of [`warp_specialization`](warp_specialization.md);
any change from this entry would be wasted.

(`launch__registers_per_thread` **cannot** be used to decide this: it is a CTA average, and `128×72 + 256×216` gives
the same number as "168 for everyone".)

## Mechanism 1: keep bookkeeping warp-invariant and it lands in UR (GPU-verified)
### 1.1 Criterion (measured on GPU by this entry's probe)

The same integer bookkeeping chain of depth `P`, **differing only in whether the seed has `+ tidx`**, every other
instruction identical:

| Variant | Seed of the bookkeeping chain | Peak demand | Landing (verified) | Spill (verified) |
|---|---|---|---|---|
| U | derived from `bidx` (warp-invariant) | per-thread `B`, bookkeeping chain in UR | `u_alu = 83`, `S2UR = 1` | `LDL/STL == 0` |
| P | the same chain `+ tidx` (per-thread) | per-thread `B + P` | `u_alu = 0`, `S2UR = 0` | `LDL/STL > 0` |

**The "Landing" column** (`[MEASURED]`): that single `+ tidx` in the seed knocked the whole bookkeeping
chain off the uniform datapath — the U side counts 83 uniform ALU instructions + the entry
`S2UR UR?, SR_CTAID.X`, the P side has **none at all**. That is this entry's mechanism claim.

**The "Spill" column** did not yet hold in that round: in that round both variants had `LDL/STL` = 0,
and the control did not spill at all (because the two chains were allowed to stagger their live ranges, not because
the mechanism failed — see the sinking gotcha in "Acceptance"). After the consumption of the two chains was
**interleaved** onto the same `acc`, pinning the peak to `B + P` by the shape of the graph, this entry's probe got
`PROBE PASS max_diff=0.0` on GPU with all six contracts passed — C4 `LDL/STL(U) == 0` and C5 `LDL/STL(P) > 0` held at
the same time. That is, both "one `+ tidx` flips the chain to per-thread" and "that flip crosses exactly the 216 line"
**are GPU-verified**.

Numerically both variants equal their own int32 reference bit for bit (a spill is a performance defect, not a
numerical one; the control must really have computed something, or the experiment is empty) — this is verified.

### 1.2 Spelling

```python
@cute.kernel
def k(...):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # warp_idx itself has already gone through make_warp_uniform (in the DSL source warp_idx = MWU(tid // 32))
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    ...
    # ---- bookkeeping / address arithmetic: fed only by warp invariants ----
    base = cute.arch.make_warp_uniform(bidx * STRIDE_C + 13)   # explicit mark (see 1.3)
    for i in cutlass.range_constexpr(P):
        base = (base ^ (i + 7)) * M + (i + 3)   # the whole chain stays in UR, takes nothing from 216
        ...
    # ---- mix the lane dimension in only at the [last step] ----
    val = mIn[base + tidx]        # address = uniform base + per-thread offset
```

Counter-example (the same chain with the lane mixed in at the first step → the whole chain becomes per-thread):

```python
    base = bidx * STRIDE_C + 13 + tidx        # <<< one + tidx flips the whole chain into R*
    for i in cutlass.range_constexpr(P):
        base = (base ^ (i + 7)) * M + (i + 3)
```

**The rule in one sentence: push `tidx / lane_idx()` to the point of consumption, not into the seed.**

> **This section only covers scalar bookkeeping chains.** For WGMMA operand descriptors (`make_fragment_A/B`) and
> the long-lived partition/address chains derived from `TiledMma / TiledCopy.get_slice(...)`, the "seed" is a
> **slice coordinate** rather than an arithmetic expression, and the point you can mark is different too (mark the
> coordinates fed to `partition_A/B`, not the descriptor) — see [`wgmma_desc_uniform`](wgmma_desc_uniform.md).
Every level of the bookkeeping chain is fed only by `bidx`, `warp_idx`, compile-time constants, and the previous
level; the per-thread component appears only in the **last addition**.

### 1.3 `make_warp_uniform` is a mark, not a guarantee

`cute.arch.make_warp_uniform(x)` is only **a hint to the compiler** (DSL 4.6.1
`cutlass/cute/arch/elect.py` L23-42, lowered to `cute_nvgpu.arch_make_warp_uniform`).
It does not change the value, and cannot turn a quantity that really varies by lane into a uniform one.

- For quantities derived from `bidx`, ptxas **can already** prove warp invariance (`S2UR SR_CTAID.X`), and they go
  into UR with or without the mark — this entry's probe **cannot** decide, and did not try to decide, "whether the
  mark is necessary".
- The value of the mark is that **it does not depend on ptxas's proving ability**: once a component whose invariance
  ptxas cannot prove gets mixed into the chain (a runtime stage index, a predicated branch result across `scf.if`, a
  value round-tripped through smem), the proof breaks and the whole chain falls back to regular registers. At that
  point the mark is your only handle.
- Hence the recommendation: **mark once at the entry of the chain**, and at acceptance count the `U*` instructions in
  SASS to confirm it really went into UR — do not announce that W dropped based on the source form alone.

## Mechanism 2: yield ladder r2 (update in place, keep no historical values)

The second-cheapest action on the same bill: loop-carried scalars (running max / running sum / pointer / phase) are
**updated in place**; do not keep historical values alive.

```python
    # good: liveness is always 1 (an SSA loop-carried value; the DSL makes it an iter_arg of scf.for)
    m = ...
    for ...:
        m = op(m, x)

    # bad 1: keeping the previous round's value until the end of this round to compute a correction -> liveness 2, multiplied by the row count
    m_old = m
    m     = op(m, x)
    ...                       # m_old used only after a long stretch
    corr  = f(m_old, m)       # <<< stretches m_old's live range across the whole stretch

    # bad 2: accumulating historical values into a fragment, indexed by a [runtime] index
    hist = cute.make_fragment(N, cutlass.Float32)
    hist[stage] = m           # <<< stage is a runtime value => the whole array is demoted to local memory
```

The two bad spellings have different mechanisms; keep them apart:

- **Bad 1 is a live-range problem**: the extra register count = number of historical values × number of rows,
  charged against 216. The fix is to compute `corr` right next to the update of `m` so `m_old` dies immediately.
- **Bad 2 is an addressing problem** (gotcha 7 of [`warp_specialization`](warp_specialization.md)): registers have no
  indirect addressing; as soon as a runtime index appears, ptxas puts **the whole array** into local memory and the
  `spill` count explodes independently of the budget. Always traverse with `cutlass.range_constexpr` (compile-time
  constant indices); hoist quantities that change across iterations into scalars.

> ⚠️ **This section was not separately ruled on by this entry's probe**. The mechanism of bad 2 comes from gotcha 7 of
> [`warp_specialization`](warp_specialization.md) (the same ptxas behaviour, recorded in that entry's context); bad 1
> is a definitional consequence of SSA live ranges, with no independent hardware assertion. It is listed here because
> it is the next tier of action after yield ladder r1 (mechanism 1), not because this entry verified it on GPU.

## Acceptance (machine-readable; not doing it is the same as not doing the change)

- **The criterion is an absolute value, not a difference between variants**: measure whether the **absolute value of
  `LDL/STL` is 0** (`cuobjdump -sass <cubin> | grep -cE '\bLDL\b|\bSTL\b'`), or ncu's
  `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`.
  `REG:` is fixed by launch bounds under `.reqntid N` + `min_blocks_per_mp=1` (384 threads ⇒ 168), **does not vary
  with the working set, and cannot be used as a criterion** (same as "Acceptance" of
  [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)).
- **Confirm the chain really went into UR and was not rematerialised away**:
  `cuobjdump -sass <cubin> | grep -cE '\bU(IADD3|IMAD|LOP3|SHF|MOV|PRMT|LEA|SEL)\b'`
  should be of the same order as the chain depth. **`ULDC` does not count** — it is a constant-bank read of kernel
  parameters, present in every kernel; using it as evidence gives away a free pass.
- **For an A/B, each variant gets its own process and its own dump directory**. The dump file name is built only from
  the jit function's **tensor signature**; the `Constexpr` variant selector **is not in the name**: compiling two
  variants in one process ⇒ the second cubin overwrites the first in place, and same signature = same compile-cache
  key, so you cannot rule out the second `cute.compile` simply returning the first variant (`[MEASURED]`
  went wrong exactly this way).
- **The variant selector must be captured by a closure, not passed as a `Constexpr` parameter of the `@cute.jit`
  entry**. A constexpr is baked into the artifact but **is not in the artifact's runtime parameter list**; passing it
  again when calling the compiled artifact shifts every later argument by one, `stream` lands in the slot the executor
  wants to convert to a tensor pointer, and it raises `[Internal Error] ... cannot be converted to pointer`.
  **Compilation is completely normal** (the emission footprint looks healthy), only the launch dies — so this gotcha
  is easily misread as a numerical/mechanism problem. Spelling: an entry factory `make_entry(flag)` closes over the
  flag, and the entry signature keeps only **tensors + stream**; a `Constexpr` on a **kernel** parameter is unaffected.
  `[MEASURED, round 1 of this entry's probe = NO VERDICT]`
- **The control must run under a load that can fail**: raise the per-thread base working set close to the budget
  before measuring. "Equal" when both sides fit comfortably only says the load is too light, not that the form does
  not matter.
- **The two chains must really be live at the same time; source order does not count**. In an A/B the base chain and
  the bookkeeping chain are naturally **two independent DAGs** (the bookkeeping chain is fed only by `bidx`/constants
  and does not depend on the base chain — exactly what the mechanism requires). Consume one completely before the
  other, and the scheduler **may legally sink the whole bookkeeping chain** below the base chain's consumption, so the
  peak liveness is only `max(B, P)` rather than `B + P`, both variants fit ⇒ the control does not spill, and the
  experiment has no teeth. You **cannot** pin it by making the bookkeeping chain depend on the base chain (that would
  make the bookkeeping chain per-thread in both variants, and the comparison disappears on the spot). The only legal
  carrier is the **accumulator**: **interleave** the consumption of the two chains onto the same serial `acc`
  (`w[P-1], v[B-1], w[P-2], v[B-2], …`), so `acc` touches `w` at step 0 and `v` only at step `B-1`; the last `w` must
  then be live before the first `v` is consumed, and neither can sink below the other — the peak `B + P` is decided
  by **the shape of the graph**, not by the order written in the source.
  `[MEASURED: before interleaving C5 failed — both variants had LDL/STL = 0]`
- **Both variants must each equal the reference bit for bit**: a spill is a performance defect, not a numerical one;
  if the control computes wrong, "it is worse" carries no information.

## Gotchas

- **An `if` in a kernel body is not a host Python `if` — even when the condition is a Python `bool`**. The DSL's AST
  rewrite turns **every** `if` in a jit/kernel body into a traced region (stack frames named `if_region_N` /
  `then_block_N`), and **both branches are traced** regardless of the condition's value. So with `range_constexpr`
  fully unrolled, a boundary check that looks purely host-side

  ```python
  for i in cutlass.range_constexpr(max(B, P)):
      if i < P:                      # <<< becomes a traced region; the next line is traced even when i >= P
          acc = f(acc, w[P - 1 - i]) # <<< negative index overflow => IndexError: list index out of range
  ```

  raises `IndexError` at **trace time** and generates not a single instruction. The symptom is **not a single verdict
  line** in the job (NO VERDICT), very easily misread as a compile failure or a numerical problem. Two fixes:
  `if cutlass.const_expr(i < P):` marks the check as compile-time; or **write no `if` at all** and split the bounded
  loop into two unconditional `range_constexpr` loops (this entry's probe uses the latter — which also keeps `scf.if`
  out of the generated DAG, which the liveness argument assumed anyway).
  `[MEASURED, round 3 of this entry's probe = NO VERDICT]`
- **One `+ tidx` is enough**. If **any** level of the chain mixes in a lane component, everything from that level on
  (and before it, whatever is consumed only by it) flips to regular registers. When reviewing code, look at **the seed
  and the inputs of every level**, not the last line.
- **UR is an integer datapath**. fp temporaries (softmax `m`/`l`, scaling factors) **cannot go** into UR; for them
  only the mechanism 2 tier of action is available.
- **The number of URs is limited** (RE material: sm_75–sm_90 have `UR0..UR62` + `URZ` per warp, 64 slots in total
  `[external reverse-engineering material, unverified by this entry]`). A bookkeeping chain a few dozen levels deep is
  fine; several hundred warp invariants do not fit, and ptxas puts the overflow back into regular registers —
  **silently**, the symptom being "it is all uniform and still spills". Counting `U*` instructions then shows the
  chain did not all get in.
- **A round trip through smem destroys the invariance proof**. Write a warp invariant into smem and read it back, and
  the value read back is an ordinary load result to ptxas, no longer uniform (unless you mark it again). Bookkeeping
  quantities handed between producer and consumer are especially easy to lose this way.
- **Do not put uniform quantities into `cute.make_fragment`**. A fragment is per-thread storage; once inside it is one
  copy per thread, and an alloca that SROA refuses to promote lands straight in local memory. Carry warp invariants as
  **bare SSA scalars + Python lists** (same origin as gotcha 10 of [`warp_specialization`](warp_specialization.md)).
- **Code after `.dec` is compiled against the reduced quota**. Computing the bookkeeping chain **before** the role
  branch (entry quota `floor(65536/blockDim)`) is usually enough, but if those values are still live after the branch
  they straddle the `.dec` point (gotchas 6 / 11). Uniform quantities are not bound by this (they are not in the
  `setmaxnreg` bill), which is itself a side benefit of making bookkeeping uniform.
- **Do not report progress with `launch__registers_per_thread` or the `REG:` difference between two variants**; see
  "Acceptance".

## Sources

- **Hardware facts (uniform datapath / uniform register file)**: the integer-only scalar datapath and per-warp `UR*`
  file introduced in Turing (sm_75); regular instructions can read UR, uniform instructions essentially only touch UR;
  one UR costs about `1/32` of a regular register in occupancy. Typical SASS forms `S2UR UR4, SR_CTAID.X`,
  `ULDC.64 UR4, c[0x0][0x210]` (Hopper's kernel-parameter bank). **There is no official specification of SASS
  semantics**; the above is public reverse-engineering material
  ([JEB SASS reversing notes](https://www.pnfsoftware.com/blog/reversing-nvidia-cuda-sass-code/),
  [redplait: uniform registers & wide loading](http://redplait.blogspot.com/2025/07/sass-instructions-uniform-registers.html),
  [PTXAS RE reference: UR usage in async copy / TMA](https://gh.evko.io/crucible-notes/ptxas/sass-isa/async-copy.html)),
  marked `[external RE material]`; this entry **asserts only the effects actually measured by the probe below**, and
  asserts nothing about the number of UR slots.
- **CuTe DSL spelling**: `cutlass/cute/arch/elect.py` L23-42 `make_warp_uniform`
  (lowered to `cute_nvgpu.arch_make_warp_uniform`, whose comment describes itself as "compiler hint indicating that the
  specified value is invariant across all threads in the warp");
  `cutlass/cute/arch/nvvm_wrappers.py` L322-334 `warp_idx` (itself
  `make_warp_uniform(tid // WARP_SIZE)`), L1299-1319 `setmaxregister_increase/_decrease`;
  exports in `cutlass/cute/arch/__init__.py` L28 / L103-104. (The source tree read locally is 4.6.1;
  4.4.2 has the same names in the same places.)
- **"W is the largest term" and this entry's positioning**: the measured decomposition of mechanism 3 of
  [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) (`peak 232 / budget 240 / upstream residual 7 / W = 97`
  `[MEASURED]`). This entry does not repeat that entry's bookkeeping method; it only follows on
  from it.
- **This entry's mechanism does not come from any benchmarked operator implementation** (the benchmark target is a
  closed binary with no source to read). The field signals that triggered this entry (an implementation under test
  with `spill_ld 1.17e7 / spill_st 1.97e6`, `tensor_pct` 0.23×, `stall_barrier` 10.1× the reference) are used only to
  convey the order of magnitude of the cost, **not as the source of the mechanism**.
- **GPU-certified probe** (not included in this release)
  — a 384-thread CTA (1 producer WG `.dec 72` + 2 consumer WGs `.inc 216`,
  `128×(168−72) = 256×(216−168) = 12288` exactly balanced, so `.inc` does not block),
  `min_blocks_per_mp=1`. Two integer chains in the consumer, both built per gotcha 10 of
  [`warp_specialization`](warp_specialization.md) to guarantee a lower bound on liveness
  (single seed + forward serial production + reverse-order consumption + an XOR between accumulator and multiply,
  ruling out rematerialisation/reordering/distributivity; plus **the consumption of the two chains interleaved onto the
  same `acc`**, ruling out one chain sinking wholesale below the other — see "Acceptance"):
  per-thread base chain depth `B = 184` (slightly below 216) + bookkeeping chain depth `P = 40`, both chains fully live
  at the start of consumption, so the per-thread demand = `B` (U) or `B + P = 224` (control). The two variants **differ
  only in the `+ tidx` in the bookkeeping chain's seed**, each running in its own subprocess + its own dump directory.
  (`P` is deliberately held at 40: the external RE material gives about 63 UR slots per warp; a chain deeper than that
  would not fit even with the mechanism holding and would silently fall back to regular registers — the test would
  then measure UR capacity rather than the invariance mechanism.)
  The interleave is issued by **two unconditional `range_constexpr` loops** (steps 0..P−1 consume v+w, steps P..B−1
  consume only v), leaving no `if` in the kernel body — for the reason see the first item of "Gotchas".
  Six contracts: the `setmaxnreg 72/216` quartet present (`.reqntid 384` + `.minnctapersm 1` + dec + inc),
  U and the control each equal to the int32 reference bit for bit, `LDL/STL(U) == 0`, `LDL/STL(control) > 0`,
  U's SASS uniform-ALU instruction count ≥ `P/2` (`ULDC` not counted).
  Both dump sinks (`options=--dump-dir` and the environment variables) are fed at once, unioned and deduplicated by
  content hash (apart from `ULDC`, C6 is a `>=` threshold, and counting the same cubin twice gives away a free pass).
  - **Certification status: ✅ GPU-certified** — `PROBE PASS max_diff=0.0`, all six contracts listed above passed
    (including C4 `LDL/STL(U) == 0` and C5 `LDL/STL(control) > 0` holding together in the same round).
