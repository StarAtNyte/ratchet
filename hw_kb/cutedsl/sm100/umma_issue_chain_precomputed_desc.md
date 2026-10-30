# umma_issue_chain_precomputed_desc (the per-step issue chain of the MMA warp on a UMMA operand ring: unroll the K loop by the stage count so the stage index becomes a compile-time constant, deleting the stage bookkeeping after the wait)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Prerequisites: [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) (how the whole UMMA mainloop is assembled),
> [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (how the wait is encoded; orthogonal to this entry).
>
> **Status: ✅ Verified (only numerics and static SASS chain length were verified)** — certified by a GPU probe (this entry's probes are not included in this release), `PROBE PASS max_diff=0.0` (first certification, round 4);
> a separately written second probe `PROBE PASS max_diff=0.0` (second certification, round 2; all 8 contracts passed in the same round, additionally testing partial unrolling and the "MMAs per ring lap" limit).
> **This is not a certification of a time win; the whole-kernel readings on both trees were flat or slower** (see "When needed").
> The section "Two rings with different stage counts: unroll by the common period" has a third probe: `PROBE PASS max_diff=0.0` (round 1, four variants bit-exact),
> which likewise certifies only numerics and static SASS counts; that section's one whole-kernel landing was flat (no gain, see that section).

## Purpose

In the K loop of a TMA → UMMA operand ring, the MMA warp has to do per step: wait on `full[s]`, issue `MMA_K_PER_BLOCK`
`tcgen05.mma` instructions, `tcgen05.commit` to `empty[s]`, then advance stage / phase.

The standard spelling is `h = consumer.wait_and_advance(); cute.gemm(..., tCrA[(None, None, kk, h.index)], ...); h.release()`.
Its stage index is a runtime value, so after the wait it has to compute:

- the stage base address (`UIMAD/ULEA`, i.e. `index × stage_stride`);
- the empty barrier address;
- the stage wrap-around (`UISETP + USEL`);
- the phase flip (`USEL + LOP3`).

**The mechanism of this entry: unroll the K loop by `NSTAGE` so the stage index becomes a compile-time constant `s`.** After unrolling:

- the descriptor offset of `(s, kk)` and the addresses of `full[s] / empty[s]` all fold into immediates, and the stage wrap-around disappears too;
- the only runtime per-step state left is the phase bit, which flips only once every `NSTAGE` steps (`ph ^= 1`), placed at the tail of the outer loop.

Probe readings (same for both geometries): worst-step chain length (instruction count from `SYNCS` to the first `UTC*MMA` + non-MMA instructions sandwiched between this step's MMAs)
25 → 10, of which wait-to-first-MMA 16 → 7; average instructions per loop-body step 35.0 → 12.75. Most of the remaining 10 are
uniform-register spills: after unrolling, all `(s, kk)` descriptors are pre-placed in URs, and at `NSTAGE = 4` there are already not enough URs, so per outer iteration 4 `MOV.SPILL` (UR → R)
+ 4 `R2UR.FILL` + 4 `R2UR` land on the chains of steps 2 and 3 (see the gotcha "the cost of unrolling").

**Applicability limit: the number of MMAs per ring lap (`NSTAGE × MMA_K_PER_BLOCK`) decides whether full unrolling shortens or lengthens the chain** (second certification,
a warp-specialized single-CTA kernel, both geometries with the same MMAs per step; instruction counting as above). After full unrolling every `(s, kk)` A/B descriptor becomes
loop-invariant, is hoisted out of the loop and lives in uniform registers; as many pairs as a lap needs, that many URs:

| MMAs per lap | runtime stage (B) worst step | full unroll (U) worst step | `R2UR` / `*SPILL*` count in U's MMA loop |
|---|---|---|---|
| 16 | 25 | **14** (2 after wait + 12 between MMAs) | 12 / 6 |
| 32 | 25 | **49** (25 after wait + 24 between MMAs) | 150 / 74 |

At 32 per lap the descriptors overflow the URs, every MMA of every step has to move them back from R first (`R2UR.FILL` / `R2UR`), and the chain is about 2× the runtime spelling —
**do not fully unroll when a lap reaches 32 MMAs**; use the "partial unroll" from the gotchas below, and accept it by counting SASS as usual.
(The 16-per-lap cell and the first certification's "25 → 10" are two different probe kernels: same direction, different numbers; both read UR spills.)

**Negative results (measured on GPU)**:
- First certification, round 3: precomputing the next step's `idx / phase / descriptor low word` after the commit in source (loop-carried `Int32` +
  branch-free bit ops), **together with** writing the 4 MMAs back-to-back as inline PTX in one `elect_one` region, made the chain longer (27–28 per step vs 25 for the original spelling).
- The second certification separated the two and only did "precompute next step's stage / phase after the commit" (no inline PTX): under both geometries the worst per-step chain was
  **25 = the original 25**; it only moved instructions from between the MMAs to after the wait (after wait 16 → 18, between MMAs 9 → 7), without saving a single one.

ptxas reorders these mutually independent uniform ALU ops back to after the wait and before the MMAs by its own heuristics. Source order does not decide SASS order; the instructions have to be **deleted** — moving them does nothing.

## When needed (trigger)

> For the opposite spelling (a rolled ring with stage / phase kept at runtime) and the same-job reading from another tree where "unrolling the MMA side by S" was 16% slower on one workload,
> see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md); the two entries are the two ends of the same axis, A/B'd per role.

- In SASS, in the K loop body, after `SYNCS.PHASECHK` and before the first `UTC*MMA`, and between several `UTC*MMA`,
  sits stage bookkeeping such as `UIMAD/ULEA` (stage base), `UISETP/USEL` (wrap-around), `LOP3` (phase).
- The timeline shows that the part of each MMA-warp step "outside the wait" is clearly larger than the wait itself, and the ablation "issue fewer UMMAs per step" gives a speedup above noise.

**These triggers do not guarantee a gain.** The only whole-kernel landing was on a single-CTA-issue GEMM tree where all the triggers above held (several UMMAs per step;
the "issue only 1 per step" ablation gave +5.21%, noise floor 2.52%):

| Form | device median | registers per thread |
|---|---|---|
| before landing (runtime stage) | 0.0091 ms | 25 |
| fully unrolled by the whole ring (NSTAGE) | 0.0095 ms (**slower**) | 178 |
| unrolled every 4 steps (`base + u`) | 0.0091 ms | 25 |
| unrolled every 8 steps (producer also unrolled every 4) | 0.0093 ms | 36 |

The same round also tried peeling the first / last group, moving the probe before the wait, and unrolling the producer by 4 / 8 steps; all readings were within 0.0091–0.0093 ms;
the whole series of changes was finally rolled back as a whole on "same-job ratio 0.9947 → 0.9939, no gain within 1.5%".

After the second certification, on another tree (more MMAs per step than the tree above; worst per-step chain of the runtime-stage spelling 23) a
same-job pricing was done (each variant in its own subprocess, all numerically correct; timed both with cold L2 and hot L2):

| MMA-warp loop form | worst per-step chain | `R2UR` / `*SPILL*` in MMA loop | cold / hot timing vs runtime stage |
|---|---|---|---|
| runtime stage (baseline) | 23 | 3 / 0 | — |
| unrolled every 2 steps | 33 | 4 / 0 | +0.12% / +0.05% (positive = faster) |
| unrolled every 4 steps | 42 | 8 / 2 | +0.12% / −0.04% |
| unrolled every 8 steps | 73 | 351 / 64 | +0.12% / −0.00% |
| fully unrolled | 89 | 398 / 94 | +0.12% / +0.06% |

On this tree **even unrolling every 2 steps lengthened the chain**, and all timings were within ±0.12%. On trees with many MMAs per step and per lap, the bookkeeping saved by unrolling does not pay for
the descriptor moves. **Before switching, count SASS on your own tree and do a same-job A/B.**

**Not the right remedy when**:

- the MMA itself is the bottleneck (tensor pipe already saturated);
- the full wait takes most of each step — that is a producer / bandwidth problem, see
  [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md).
- the descriptor low word `UIADD3` / high word `UMOV` per MMA: the `gdesc` operand requires lo/hi in a pair of consecutive URs, one pair per MMA; this entry cannot remove them.

## Minimal runnable example

Conditions: single CTA, `cta_group::1`, A/B both K-major fp16, fp32 accumulator in TMEM. Producer, barriers, TMEM allocation and readback
are the same as in [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) (the producer keeps using `acquire_and_advance + cute.copy`);
only the MMA warp is shown here. The probe ran `NSTAGE = 4`, 4 MMAs per step (K tile 64 / instruction K 16), several steps along K (the stage ring wraps several times).

```python
NSTAGE = 4                                   # operand-ring stage count
MMA_K_PER_BLOCK = TILE_K // INST_K           # MMAs per step (e.g. 64 // 16 = 4)
assert K_BLOCKS % NSTAGE == 0                # see "Gotchas": the remainder must be handled separately

# ab_pipe = pipeline.PipelineTmaUmma.create(num_stages=NSTAGE, ..., barrier_storage=storage.ab_mbar_ptr.data_ptr())
# its barrier_storage holds NSTAGE full barriers first, then NSTAGE empty barriers.
# The pointers must be taken from the struct at kernel top level (outside any dynamic if), see "Gotchas"
bar_full0 = storage.ab_mbar_ptr.data_ptr()
bar_empty0 = storage.ab_mbar_ptr.data_ptr() + NSTAGE

if warp_idx == WARP_MMA:
    acc_slot = acc_producer.acquire_and_advance()
    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)       # first MMA of the first step: D = A*B
    ph = cutlass.Int32(0)                                # the consumer waits on full starting at phase 0
    for kr in cutlass.range(K_BLOCKS // NSTAGE, unroll=1):
        for s in cutlass.range_constexpr(NSTAGE):        # s is a Python int: a compile-time constant
            cute.arch.mbarrier_wait(bar_full0 + s, ph)
            for kk in cutlass.range_constexpr(MMA_K_PER_BLOCK):
                kc = (None, None, kk, s)                 # stage mode indexed by a constant -> offset folds into an immediate
                cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty0 + s)           # release stage s to the producer
        ph = ph ^ 1                                      # flips only once every NSTAGE steps
    acc_slot.commit()                                    # accumulator full -> epilogue
```

## Gotchas

- **`K_BLOCKS % NSTAGE != 0`**: the outer loop covers only whole groups. The remaining steps need a tail segment after the loop with `range_constexpr`: stages start at 0,
  phase uses `ph` as of the end of the loop. When K_BLOCKS is a runtime value, the tail has to fall back to the runtime-stage spelling. (The first two probes only ran the divisible case;
  a peeled straight-line tail is numerically certified in the probe of the section "Two rings with different stage counts": slot / parity computed as constants from the global item index known at trace time.)
- **The cost of unrolling is code size and registers**: the loop body is `NSTAGE × MMA_K_PER_BLOCK` MMAs plus NSTAGE wait-retry stubs, and
  all `NSTAGE × MMA_K_PER_BLOCK` descriptor pairs live in uniform registers. At `NSTAGE = 4` with 4 per step (same for both geometries) the probe already read
  UR spills: per iteration 4 `MOV.SPILL` + 4 `R2UR.FILL`, sandwiched between MMAs and after the wait — the chain still dropped from 25 to 10, but acceptance must count these
  spill instructions; at 32 MMAs per lap the spills stretch the chain to about 2× the original spelling (the limit table in "Purpose"). In the first table of "When needed", full unrolling by the whole ring
  pushed registers per thread from 25 to 178 and made the whole kernel slower.
- **Partial unrolling (unroll only U steps) is numerically certified but does not guarantee a shorter chain**: the outer loop runs `K_BLOCKS // U` times, and each iteration **computes directly** from the outer counter
  `base = (kr & (G - 1)) * U`, `ph = (kr >> log2(G)) & 1` (`G = NSTAGE // U`; requires NSTAGE to be a multiple of U and G a power of 2);
  the inner `range_constexpr(U)` uses stage `base + u`. There is no stage / phase state carried across iterations, but `base` is a runtime value, so the descriptor offsets
  do not fold completely. Second certification with U = 2: bit-exact against the reference in both geometries, no `R2UR` / `*SPILL*` in the MMA loop, worst per-step chain 24 / 22 (runtime spelling 25) —
  only 1–3 shorter; on the tree of the second table in "When needed", unrolling every 2 steps went 23 → 33 instead. Chain length varies with geometry; count the SASS yourself.

  ```python
  G = NSTAGE // U                                  # Python int, power of 2
  LOG2_G = G.bit_length() - 1
  for u2_k in cutlass.range(K_BLOCKS // U, unroll=1):
      u2_trip = cutlass.Int32(u2_k)                # convert to Int32 first, then do bit ops
      u2_base = (u2_trip & (G - 1)) * U
      u2_ph = (u2_trip >> LOG2_G) & 1
      for u in cutlass.range_constexpr(U):
          cute.arch.mbarrier_wait(bar_full0 + (u2_base + u), u2_ph)
          for kk in cutlass.range_constexpr(MMA_K_PER_BLOCK):
              kc = (None, None, kk, u2_base + u)
              cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
              tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
          with cute.arch.elect_one():
              tcgen05.commit(bar_empty0 + (u2_base + u))
  ```
- **Give local variables in the partially unrolled loop new names**: a name already assigned elsewhere in the same kernel function (even in another `const_expr` branch)
  (e.g. `ph`, `s`), when reassigned inside a dynamic `cutlass.range` loop body, is treated by the preprocessor as a loop-carried variable; since it does not exist before the loop in this branch, tracing raises
  `` DSLRuntimeError: `ph` is None prior to this `for`, and update to Int32 inside of this `for` is not supported. ``
  (read in round 1 of the second certification). Fix: use names unique to this branch (the `u2_*` above), and convert the outer counter with `cutlass.Int32(...)` before the bit ops.
- **When the phase flips**: each stage is used exactly once per outer-loop iteration, so the phase flips once per iteration and all stages share the same `ph`.
  If you hand-write a separate producer loop, the producer's initial phase for waiting on empty must match the `PipelineTmaUmma` convention; if the producer keeps using
  `acquire_and_advance` you need not worry.
- **Wrap the commit in `with cute.arch.elect_one():`**: commit tracks the tcgen05 async operations issued earlier by the thread that issues it.
  `cute.gemm` goes through the DSL MMA atom and needs no elect of your own.
- **Ring barrier addresses**: `PipelineTmaUmma`'s `barrier_storage` is 2×NSTAGE Int64; the first NSTAGE are full, the last NSTAGE are empty.
  Pointer arithmetic is in elements, so write `data_ptr() + s` / `data_ptr() + NSTAGE + s`.
- **A `@cute.struct` object cannot be referenced inside a dynamic `if`**: writing `storage.xxx.data_ptr()` inside a runtime branch like `if warp_idx == ...:`
  throws at trace time `DSLRuntimeError: The 'if' statement encountered a user-defined Python object, which cannot be automatically
  converted into an dynamic expression.` (read in certification round 2). Take the barrier pointers into locals at kernel top level first, and use only those pointers inside the branch.
- **Pass the TMEM holder field itself**: writing `.ptr` throws `AttributeError: '_Pointer' object has no attribute 'ptr'` (read in certification round 1);
  this gotcha is collected in [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 10.
- **Always accept by SASS**: count the instructions between `SYNCS` and the first `UTC*MMA` of each step, plus those sandwiched between MMAs. Moving things around or hand-ordering in source
  may be reordered back by ptxas (see the negative results above).
- This entry changes only the MMA warp's loop form; producer, barrier counts, TMEM layout and readback are unchanged; orthogonal to the wait encoding (blocking / non-blocking / look-ahead).

## Two rings with different stage counts: unroll by the common period

The spelling above assumes a single ring with one slot consumed per step. When the MMA warp takes `r_X` slots per step from ring X and `r_Y` slots from ring Y,
and the stage counts differ, `S_X ≠ S_Y`, unrolling by `NSTAGE` cannot make the slot indices of both rings constant at once. The unroll length has to be the common period of the two rings:

```
P_X = S_X // gcd(S_X, r_X)      # steps until ring X returns to the same slot
P_Y = S_Y // gcd(S_Y, r_Y)
G   = lcm(P_X, P_Y)             # unroll length in steps; one group = G steps, GI = G * r items (when both rings take the same number per step)
LAPS_X = GI // S_X              # laps ring X makes in one group (an integer)
```

The `l`-th item of a group (a Python int) sits in slot `l % S_X` of ring X, the same in every group. Parity is `(l // S_X) & 1` XORed with a group-level bit:
when `LAPS_X` is even that bit is always 0 and the whole parity is constant; when `LAPS_X` is odd it flips once per group, and that is the ring's only loop-carried state (one `Int32`).
Worked example (not the probe geometry): `S_X = 3, r = 2` → `P_X = 3`; `S_Y = 2, r = 2` → `P_Y = 1`; `G = 3`, `GI = 6`;
X makes 2 laps per group (parity fully constant), Y makes 3 laps per group (carries one bit). When the total step count is not a multiple of `G`, peel the remainder into a straight-line tail after the loop
(global item index `NG * GI + l` is known at trace time; `NB` must be a compile-time constant); "things done only on the last step" become unconditional code when they fall in the tail.

```python
import math
G  = math.lcm(SA // math.gcd(SA, IPB), SB // math.gcd(SB, IPB))   # Python int; each ring takes IPB per step
GI = G * IPB
LAPS_A, LAPS_B = GI // SA, GI // SB
NG, REM = NB // G, NB % G                       # NB is a compile-time constant

if warp_idx == WARP_MMA:
    u_apar = cutlass.Int32(0)                   # group-level parity bit, typed before the loop
    u_bpar = cutlass.Int32(0)
    for u_g in cutlass.range(NG, unroll=1):
        for u_l in cutlass.range_constexpr(GI):  # u_l is a Python int → slot index constant
            u_as, u_bs = u_l % SA, u_l % SB
            cute.arch.mbarrier_wait(b_full + u_bs, u_bpar ^ ((u_l // SB) & 1))
            cute.arch.mbarrier_wait(a_full + u_as, u_apar ^ ((u_l // SA) & 1))
            for u_kk in cutlass.range_constexpr(MMA_PER_ITEM):
                if cutlass.const_expr(u_l == 0 and u_kk == 0):
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, u_g != 0)
                cute.gemm(tiled_mma, acc, tCrA[(None, None, u_kk, u_as)], tCrB[(None, None, u_kk, u_bs)], acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():
                tcgen05.commit(a_empty + u_as)
                tcgen05.commit(b_empty + u_bs)
        if cutlass.const_expr(LAPS_B % 2 == 1):
            u_bpar = u_bpar ^ 1
        if cutlass.const_expr(LAPS_A % 2 == 1):
            u_apar = u_apar ^ 1
    for t_l in cutlass.range_constexpr(REM * IPB):   # peeled tail: slot / parity all constant
        t_c = NG * GI + t_l
        cute.arch.mbarrier_wait(b_full + (t_c % SB), (t_c // SB) & 1)
        cute.arch.mbarrier_wait(a_full + (t_c % SA), (t_c // SA) & 1)
        for t_kk in cutlass.range_constexpr(MMA_PER_ITEM):
            if cutlass.const_expr(NG == 0 and t_l == 0 and t_kk == 0):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            cute.gemm(tiled_mma, acc, tCrA[(None, None, t_kk, t_c % SA)], tCrB[(None, None, t_kk, t_c % SB)], acc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        with cute.arch.elect_one():
            tcgen05.commit(a_empty + (t_c % SA))
            tcgen05.commit(b_empty + (t_c % SB))
    with cute.arch.elect_one():
        tcgen05.commit(acc_full)                # after the tail: unconditional
```

Key points and gotchas:

- **G is counted in steps, not `lcm(S_X, S_Y)`**: `lcm(S_X, S_Y)` is the common period in items; `P = S // gcd(S, r)` gives steps directly (in the worked example `lcm(3, 2) = 6` items = 3 steps).
- **The parity of a ring with an odd lap count is not constant**: if you drop the group-level bit, from the second group on that ring's waits land on the wrong phase — waiting on an already-completed phase returns immediately and reads data that has not landed (silent error),
  or waits on a phase that never comes (hang). Use `LAPS % 2` at trace time to decide whether to carry it; alternatively double the unroll length to `2G` so all lap counts become even, doubling code size and descriptor count again.
- **Peel the remainder into a tail; do not skip items with a runtime `if` inside the unrolled group** (the item index would become a runtime value again). The tail can also be peeled at the head: the first MMA's `ACCUMULATE=False`
  becomes a constant, but "the last step's" work goes back into the loop and needs a test.
- **The producer need not be unrolled too**: both sides see the same (slot, parity) sequence; in the probe the producer stayed rolled (`c % S`, `(c // S) & 1`), mixed with the unrolled consumer, bit-exact numerics.
- Give loop-local variables unique prefixes per the gotcha above (`u_*` / `t_*` in the example); wrap the group-level parity-bit flips in `const_expr`, so the bit of the ring with an even lap count is never assigned.

**Probe readings (third probe; one rolled control — both rings carry (slot, parity), branch-free advance — plus unrolling at three total step counts, two with a tail and one without;
all four bit-identical numerically)**:

| Quantity | rolled control | unrolled (three total step counts) |
|---|---|---|
| instructions per UMMA within the MMA span | 24.69 | 23.96 – 24.45 |
| `R2UR` per UMMA | 4.7 | 5.3 – 6.1 |
| spill / fill | 0 / 0 | 0 / 0 |
| `SYNCS.PHASECHK` address form | `[R+URZ]`, `[R+URZ+imm]` | `[R+URZ+imm]`, `[R+URZ]` |
| MMA-loop cycles per item | 514.1 | 482.2 – 493.6 |

How to read it: after unrolling, issue instructions per UMMA are almost the same as the rolled control, with slightly more `R2UR`; both address forms are "general-register base + immediate", so **you cannot tell from the address form whether the loop was unrolled**.
The pair with the same total step count (unrolled vs rolled) has 482.2 vs 514.1 cycles per item, about 6% fewer; the other two unrolled variants have different total step counts and are not directly comparable. This probe did not isolate the limiting source and is not a pricing. Numerics are the section's only certification.

**Whole-kernel reading (one tree, one landing)**: on a single-CTA tree, with the MMA warp unrolled per this section (with a peeled tail), gate 0.0439 ms, same-job ratio 0.9971;
the pre-landing tree re-measured 5 times same-job on the same node: 0.0438 ms, ratio 0.9958–1.0098. **Flat**, no gain, rolled back with the patch it was in. Do not use it as a speedup recipe;
before switching, count SASS on your own tree and do a same-job A/B.

## Sources

- PTX ISA: `tcgen05.mma`, `tcgen05.commit`, `mbarrier.try_wait` —
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma
- Public infra API:
  - `cutlass.pipeline.PipelineTmaUmma` (`barrier_storage` layout: full first, empty after; `make_participants`)
  - `cute.arch.mbarrier_wait`, `cute.arch.elect_one`, `cutlass.cute.nvgpu.tcgen05.commit`
  - `cutlass.range_constexpr`, `cute.gemm`, `tcgen05.Field.ACCUMULATE`
  - `utils.TmemAllocator`
- GPU certification probe: ✅ `PROBE PASS max_diff=0.0`, round 4. Compares the runtime-stage spelling (B) with the unrolled spelling (U), two geometries
  (one of them 128×64×16 instruction, 128×64×64 tile), A/B drawn from {-1, 0, 1} so the result is integral, bit-exact required, the same compiled entry launched twice.
  Four contracts all passed in the same round: `C0-lowers`, `C1-dsl-exact` 0.0, `C2-unroll-exact` 0.0, `C3-chain` (worst step in both geometries B=25 → U=10).
  U's 4 steps (wait-to-first-MMA, instructions sandwiched between MMAs) = (1,1) (2,8) (7,0) (3,1); in the two shortest steps only 1 instruction (`UMOV` / `UPLOP3`) sits between the 4 `UTCHMMA`,
  and most instructions of steps 2 and 3 are UR `MOV.SPILL` / `R2UR.FILL` / `R2UR`. The negative result is read from the `C3-chain` FAIL of round 3 (the previous version of the same probe).
- Second GPU certification probe (a separately written warp-specialized single-CTA kernel: one TMA warp, one MMA warp): ✅ `PROBE PASS max_diff=0.0`,
  round 2. Four MMA-warp loop spellings B (`wait_and_advance` / `release`), P (branch-free precompute of next step's stage / phase after the commit),
  U (fully unrolled by NSTAGE), U2 (unrolled every 2 steps, base / phase computed from the outer counter), two geometries (16 / 32 MMAs per lap), all bit-equal to numpy.
  8 contracts all passed in the same round: `C0-lowers`, `C1-dsl-exact` 0.0, `C2-unroll-exact` 0.0, `C3-unroll2-exact` 0.0, `C4-small-ring-unroll-shorter`
  (16 per lap: U 14 < B 25), `C5-precompute-exact` 0.0, `C6-move-does-not-shorten` (P 25 ≥ B 25, both geometries),
  `C7-big-ring-unroll-regresses` (32 per lap: U 49 > B 25). U2's chain (24 / 22) is a fact row, not a contract.
  In round 1 U2 failed to trace in both geometries; the reading is the gotcha "give local variables in the partially unrolled loop new names".
- Third GPU certification probe (the section "Two rings with different stage counts"; single CTA, `cta_group::1`, producer rolled, MMA warp unrolled per that section):
  ✅ `PROBE PASS max_diff=0.0`, round 1. Four variants (rolled control + three total step counts unrolled by the common period: two with a peeled tail, one without),
  integer data bit-exact against numpy, four contracts all passed in the same round; SASS counts and cycles per item are information rows, not contracts.
