# warp_specialization (warpgroup role split + `setmaxnreg` register partitioning, sm_100a)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers only **how the CTA's register budget is moved between
> warpgroups, how to accept the move by machine reading, and how to choose the split levels yourself**. The paths that move **data** between roles
> (TMA + mbarrier pipeline, UMMA issue) are in their own entries.
>
> **Status: ✅ GPU-certified** — `PROBE PASS max_diff=0.0` (certified by a GPU probe, not included in this release;
> all eight contracts PASS; readings in "Sources").

## Purpose

On sm_100 the register file of one SM is still **65536 32-bit registers**, and one CTA may hold all of them.
ptxas gives each thread a **static** limit from `.reqntid N`: `floor(65536 / N)` rounded down to a multiple of 8.
A CTA of three warpgroups (384 threads) therefore gives everyone **168** — not enough for the math warpgroups that hold several
accumulator / fragment sets at once, while single-warp specialized roles that only issue TMA / UMMA descriptors cannot use theirs at all.

`setmaxnreg` redistributes this budget at **runtime**: the giving warpgroup does `.dec`, the taking warpgroup does `.inc`,
with conservation `Σ(warpgroup threads × that warpgroup's quota) ≤ 65536`, each quota **a multiple of 8,
within [24, 256]**. Typical three roles: `128×24 + 128×224 + 128×256 = 64512`.
**The profiler's `registers_per_thread` still reports 168** (it is the CTA average); the taking warpgroups
actually get 224 / 256.

## When needed (trigger)

- CTA ≥ 256 threads, asymmetric warpgroup roles (some move data / issue instructions, some compute), and the computing side's
  register working set > `floor(65536 / blockDim)`.
- **Decisive symptom: `spill_ld / spill_st` (ncu `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`)
  is nonzero while `launch__registers_per_thread` looks "normal" (= 168).** A drop in `tensor_pct`,
  a rise in `stall_long_scoreboard` (`LDL` is a long-scoreboard load), growth in `dram_bytes_write`
  (spill stores leak to DRAM) are all downstream effects, not independent causes.
- `registers_per_thread` **cannot** tell you whether the split took effect: "24 + 224 + 256" and "no split at all,
  everyone at 168" give the same number. The only machine-readable quantities that tell them apart are the spill count (must be exactly 0 when the split is right)
  and the presence of `USETMAXREG` in SASS (see "Acceptance").
- **dec/inc are already in the code and `spill > 0` remains**: do not stop at gotcha 1; per "Acceptance", cut the SASS into regions at
  `USETMAXREG`, see which region the spills fall in, then compare against the three causes in gotcha 1 / gotcha 11 / gotcha 4 —
  their profiler symptoms are identical; only the SASS region readings and the statement order distinguish them.

## Minimal runnable example

All four pieces are required: **① dec/inc inside the kernel**, **② `min_blocks_per_mp` on the launch side**,
**③ placement of dec/inc (dominating their respective role bodies)**, **④ acceptance on the compiled artifacts**. Doing only ①② still spills
massively (gotcha 11); doing only ①③ is as good as not writing it (gotcha 1).

**Placement rule (one sentence)**: the shared prelude (smem allocation / mbarrier init / TMEM alloc / construction of descriptors and
pipeline objects / whole-CTA sync) all goes **before** the **single** role branch; dec/inc is the **first** statement of each
branch body; **the role body (including its own persistent item loop) lives entirely inside the branch, and the roles never rejoin
after the branch**.

```python
WG          = 128
THREADS     = 3 * WG          # two math warpgroups + one specialized warpgroup (4 single-warp roles)
REG_SPEC    = 24              # giver: a multiple of 8, >= 24
REG_A       = 224             # taker A
REG_B       = 256             # taker B     128*(24+224+256) = 64512 <= 65536

@cute.kernel
def _kernel(...):
    tidx, _, _ = cute.arch.thread_idx()
    # use a warp-uniform quantity as the predicate: setmaxnreg is .sync.aligned, the whole warpgroup must arrive together (gotcha 3)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    wg = warp_idx // 4

    # ---- shared prelude: compiled at the [entry quota 168], before any dec ----
    smem = utils.SmemAllocator()
    ...                           # mbarrier init / TMEM alloc / TMA partition / pipeline objects
    cute.arch.sync_threads()      # the whole-CTA sync is also before the branch

    # ---- the single role branch: dec/inc is the first statement of each branch body, the role body lives entirely inside ----
    if wg == 2:
        cute.arch.setmaxregister_decrease(REG_SPEC)
        # the 4 single-warp specialized roles split by warp_idx [inside here]: the dec was already issued by the whole warpgroup together
        if warp_idx == 8:  ...    # e.g. UMMA issue loop
        if warp_idx == 9:  ...    # e.g. TMA supply loop
        ...
    elif wg == 0:
        cute.arch.setmaxregister_increase(REG_A)
        for item in ...:          # this role's own persistent loop, living inside the branch
            ...                   # heavy register state is built [inside] the branch (gotcha 6)
    else:
        cute.arch.setmaxregister_increase(REG_B)
        for item in ...:
            ...

@cute.jit
def _launch(..., stream):
    _kernel(...).launch(
        grid=grid, block=[THREADS, 1, 1],
        min_blocks_per_mp=1,      # <<< without it, ptxas silently drops dec/inc (gotcha 1)
        stream=stream)
```

**Counter-example A (gotcha 11, form one: budget move separated from the role bodies)** — three early `if`s only move the budget, and the shared prelude
lands after the dec:

```python
    if wg == 2: cute.arch.setmaxregister_decrease(REG_SPEC)
    if wg == 0: cute.arch.setmaxregister_increase(REG_A)
    if wg == 1: cute.arch.setmaxregister_increase(REG_B)
    smem = utils.SmemAllocator()      # <<< the prelude is now after the dec: ptxas gives this straight-line code
    ...                               #     only a quota of 24 (any warp may reach here)
    cute.arch.sync_threads()
    for item in ...:                  # counter-example B (gotcha 11, form two): loop outside, branch inside;
        if wg == 2: ...               # the roles rejoin at the top of every iteration — the shared code at the join
        if wg == 0: ...               # is likewise compiled at 24
```

DSL spelling: `cute.arch.setmaxregister_decrease(n)` / `cute.arch.setmaxregister_increase(n)`
(lowers to `nvvm.setmaxregister`, PTX `setmaxnreg.{dec,inc}.sync.aligned.u32 <imm>;`,
SASS `USETMAXREG.*`). The old names `warpgroup_reg_dealloc / warpgroup_reg_alloc` are `@deprecated`.
`launch(min_blocks_per_mp=1)` generates `nvvm.minctasm` → PTX `.minnctapersm 1`.

### Acceptance (skipping it is as good as not doing it)

Keep the cubin / PTX at compile time, then count instructions. All three steps are machine reads, all implemented in the probe:

```python
import os
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"      # must be before import cutlass
os.environ["CUTE_DSL_KEEP_PTX"]   = "1"
os.environ["CUTE_DSL_DUMP_DIR"]   = d
import cutlass, cutlass.cute as cute
compiled = cute.compile(_launch, *args)       # .ptx / .cubin appear in d/
```

```
# (1) all four pieces present: in the PTX
grep -E '\.reqntid|\.minnctapersm|setmaxnreg\.(dec|inc)' d/*.ptx
#     .reqntid 384   .minnctapersm 1   setmaxnreg.dec…24   setmaxnreg.inc…224   setmaxnreg.inc…256
# (2) ptxas did not drop them: USETMAXREG count in SASS == number of dec/inc statements (3 here); == 0 means gotcha 1
nvdisasm -c d/*.cubin | grep -c USETMAXREG
# (3) zero spills
nvdisasm -c d/*.cubin | grep -cE '\bLDL\b|\bSTL\b'     # must be 0
cuobjdump -res-usage d/*.cubin                           # REG:168 (entry quota) LOCAL:0 (check STACK too)
```

**Locating spills (when (3) is nonzero)**: cut the SASS into regions at the `USETMAXREG` instructions — region 0 is the code before the first
`USETMAXREG` (the prelude, compiled at the entry quota 168), and each subsequent `USETMAXREG` opens a
new region (compiled at that instruction's quota) — and count `LDL` / `STL` per region:

| `LDL/STL` fall in | Cause | Fix |
|---|---|---|
| region 0 (prelude) | the prelude itself > 168 (rare) | slim the prelude: recompute addresses on demand, do not materialize address arrays (gotcha 7) |
| after `.dec`, before the next `USETMAXREG` | the giver's quota is too small, **or** shared code fell into the giver's region (gotcha 11) | check the statement order first; if the role body itself really exceeds 24, add 8 per "Choosing the split" |
| after some `.inc` | that taker's working set > quota | reduce that role's live set, or move 8 from a role with headroom (choosing the split) |
| everywhere, `USETMAXREG == 0` | gotcha 1 | add `min_blocks_per_mp` |
| a few `STL` in the `.dec` region, and matching `LDL` in the join region **after** the last `USETMAXREG` | there is still a join after the branch (wrap-up / teardown placed after the role branch, gotcha 11 form C) | keep one copy of the wrap-up per role, leave no statements after the branch |
| in the `.dec` region, with source lines all in the role branch's **shared head** (item scalars, the carried next-item index, `grid_dim`) | the giver's quota is too small: these values live across the whole branch and are read by every warp of the branch | choosing the split, step 3: giver +8, taken from a taker with headroom |
| several `STL` at one place in some `.inc` region, then **exactly one `LDL` at the head of each subsequent segment** | no segment is over budget by itself: branch-level long-lived values are pushed out by one peak segment of the same branch and pay again on every use (gotcha 14) | shrink that peak segment (batch by columns); the split cannot move it |

**Down to source lines (the region table says "which quota", this step says "which statement")**: set `CUTE_DSL_LINEINFO=1` before compiling, then
`nvdisasm --print-line-info -c d/*.cubin` — the SASS is interleaved with `//## File "kernel.py", line N` markers; attribute each
`LDL/STL` to the nearest marker above it and aggregate by (region, source line). Region cut points can also be recognized directly by mnemonic:
`USETMAXREG.DEALLOC.CTAPOOL` is `.dec`, `USETMAXREG.TRY_ALLOC.CTAPOOL` is `.inc`. When line info did not make it into the cubin,
attribution is empty and the region method works as usual (both steps are implemented in the regsplit GPU certification probe).

### Choosing the split yourself (do not blindly apply 24 / 224 / 256)

The levels are **free variables under constraints**; the right levels are derived backwards from "acceptance (3) == 0", not from rule-of-thumb numbers:

1. Constraints: each level a multiple of 8, ∈ [24, 256]; `Σ(threads × level) ≤ threads × entry quota` — the pool is the share
   allocated to the CTA at launch by the entry quota (384 threads: 384 × 168 = **64512**), **not 65536**; going up to 65536 hangs
   on `.inc` (gotcha 13). With three 128-thread roles the sum of per-role levels ≤ 504,
   and +8 on one role takes exactly −8 from another (same granularity); −8 per thread on a 256-thread role = +16 on a 128-thread role,
   and "only +8 for a 128-thread role" needs −4 on the 256-thread side, which is **not encodable** (gotcha 8).
2. First measure with the region method which region spills and how much (the `LDL+STL` count is a lower-bound indicator of how far the live set exceeds the quota).
3. Take 8 from a role **with no spills and a level > 24** and give it to the spilling role, recompile, recount; repeat until every region is 0.
   The giving role (specialized warps) is **not** automatically fine at 24: it issues UMMA descriptors / TMA coordinates / bookkeeping
   indices, and a single-warp role's loop body can spill just the same — if it spills, give it 32 / 40, taken from the 256 side.
4. Nothing left to take (Σ at the cap and some region still spills) ⇒ not a split problem; that role's live set must shrink (fragment segmentation,
   deferred materialization, per-iteration hoisting of scalars); go back to the respective entries.
5. Also check the cycle ledger when choosing whom to take from: taking registers from a **busier** role to give to the one waiting on it costs more than the receiver saves (gotcha 12).

## Gotchas

1. **No `min_blocks_per_mp` → `setmaxnreg` is silently dropped (the number-one gotcha).** ptxas executes `setmaxnreg` only when it can **statically
   determine the per-thread register count at entry**; otherwise it drops the instruction and leaves only an info-level warning
   `(C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to determine register
   count at entry` — the DSL's JIT does not surface it to the Python layer. So **the code looks right, but ptxas compiles at a uniform
   168 and the takers spill massively**. In the DSL the entry register count is pinned by two nvvm attributes: `nvvm.reqntid`
   (generated automatically from `launch(block=...)`) and `nvvm.minctasm` (generated **only when** `min_blocks_per_mp > 0`;
   default 0 = not generated). `min_blocks_per_mp=1` is enough and does not change the occupancy semantics.
   Machine-readable footprint: **`USETMAXREG == 0`** in SASS while all the `setmaxnreg` are in the PTX.
2. **Do not "help" it with `-maxrregcount` / a lower launch bound.** That is a different route (pinning the entry count directly),
   and stacked with `setmaxnreg` they conflict (known forms: adding a bound hangs at the first MMA; removing it spills badly).
   Pick one: `reqntid + minctasm + setmaxnreg`, or global maxrregcount; do not mix.
3. **The predicate must be warpgroup-uniform, and all 4 warps of the warpgroup must execute the same instruction.**
   `setmaxnreg` is `.sync.aligned`: use `cute.arch.make_warp_uniform(cute.arch.warp_idx())`
   to get a warp-uniform value visible to the compiler and compare on `warp_idx // 4`; **do not** use a per-thread `tidx` predicate,
   do not put it inside `elect_one` / `if lane == 0`, and **do not** issue the dec separately per single-warp role (the 4 specialized warps of a warpgroup
   first `.dec` together, then split work by `warp_idx` inside the branch).
4. **`.inc` blocks, and oversubscription deadlocks, with no error and no spill.** The increase waits until the CTA pool really has free registers, which are handed over
   by the `.dec` of other warpgroups of the same CTA. Over-counting (e.g. `(32 + 232 + 256) × 128 = 66560`) leaves the takers
   all stuck on `.inc` waiting on each other — the kernel is killed on timeout without a single error line in the log. Do the arithmetic before compiling. The arithmetic's upper bound
   is not 65536 but the entry allocation `threads × entry quota` (gotcha 13).
5. **dec/inc go first in their respective branches, with no barrier in between.** The taker's `.inc` must wait for the giver's `.dec`
   to happen first; a barrier between them that both sides participate in (or the taker waiting, before its `.inc`, on an mbarrier that the giver only
   reaches afterwards) is a circular wait. Put the whole-CTA `sync_threads()` **before the branch**.
6. **Build heavy register state inside the branch.** If accumulators / fragments are built before the branch and still read after it, their live range
   spans the giver's `.dec` point; if ptxas cannot prove them dead, they spill in the 24 region.
7. **Getting 224 does not mean you can use it carelessly: a dynamically indexed register array is demoted to local as a whole.** If a fragment is indexed with a runtime
   `Int32`, ptxas can only put the whole array in local memory, and the spill count explodes regardless of the budget.
   Iterate fragments with `cutlass.range_constexpr` always; hoist quantities that change across iterations into scalars.
8. **Quota rounding and exchange granularity**: `floor(65536 / blockDim)` rounded down to a multiple of 8 — 384 → 168,
   256 → 256 (capped), 512 → 128. Budget moves between roles balance at a granularity of "threads × 8" (see "Choosing the split" 1).
9. **Requires `sm_100a` (a family / arch-specific target), not `sm_100`.** cutlass's C++ wrapper is guarded by
   `__CUDA_ARCH_FEAT_SM100_ALL` (or the family macro), and on non-`a` targets the whole instruction is `#if`'d out; on the DSL side
   `CUTE_DSL_ARCH=sm_100a`.
10. **To verify this mechanism, first prove the control group really spills.** Building a working set with a lower bound on liveness must block, all at once:
    (i) rematerialization — when every value can be traced to one global load, ptxas would rather reissue the `LDG`, so the whole chain is fed by only
    **one** seed; (ii) reordering — produce a serial chain and consume it as a serial chain in **reverse** order, the two orders opposite, so all N values are live
    at the instant `v[N-1]` is produced; (iii) reassociation / distributivity — integer `*` distributes over `+`, and reverse Horner gets folded into forward
    streaming with only 2 live registers; insert an **XOR** between the accumulator and the multiply (inline PTX `xor.b32`, which neither LLVM nor
    ptxas can fold). Keep the heavy values in a Python list (SSA, fully unrolled with compile-time indices), not in a `make_fragment`
    array (an unpromoted alloca is itself local memory).
11. **The budget move does not dominate the role bodies → shared code lands in the giver's quota and spills massively anyway.** This is the top reason for "gotcha 1 is all
    done right and it still spills"; the symptoms are the same as gotcha 1, but `USETMAXREG` **is** in the SASS — `setmaxnreg`
    took effect, just too early / too fragmented. ptxas cuts the kernel into budget regions at `setmaxnreg`: a piece of code that is after some `.dec` on
    **any** path is compiled at that `.dec`'s quota. Two common forms:
    **A. Early `if`s only move the budget**, and the prelude (smem / mbarrier / TMEM alloc / descriptors) is built afterwards — the prelude is
    straight-line code run by all threads, most of its values are still live for the takers, so the giver is forced to materialize the whole prelude in 24;
    **B. Loop outside, branch inside** (the persistent item loop rejoins all roles at the top of every iteration to read item parameters / wait on a
    barrier) — the shared code at the join is likewise compiled at 24, and every iteration crosses a budget-region boundary.
    Same fix for both: prelude first, **one** branch, dec/inc first in the branch body, **each role's own item loop
    living inside its own branch**, no rejoin after the branch.
    **C. Wrap-up after the branch** (CTA teardown: `sync_threads` + TMEM dealloc placed after the role branch, all threads
    rejoining there) — what remains after forms A and B are fixed. Certified reading: `LDL=15 STL=5`, regions
    `{dec: 0/2, inc: 4/3, inc: 9/0, end: 2/0}` — the 2 `STL` in the `.dec` region pair with the 2 `LDL` in the join region after the last
    `USETMAXREG`. Fix: put **one copy of the wrap-up in each branch** (`sync_threads` is a CTA-level barrier; all
    threads still arrive once each), leave no statements after the branch; in the next round the `STL` in the `.dec` region dropped to zero.
12. **Moving work between roles: look at both sides' busy time, not at who is waiting.** The cycle of a warp-specialized kernel is set by the busiest role;
    a role with a high wait share only shows that it is waiting on others — moving work onto it makes it busier but does not shorten the cycle; moving work onto a role busier
    than it lengthens the cycle by the amount moved. Judge by per-role busy cycles (the timeline's busy / wait decomposition).
    (This gotcha's reading comes from a KEEP evaluation on an sm90 tree; the mechanism is architecture-independent; not re-measured separately on sm100.)
13. **The upper bound of `Σ(threads × level)` is the entry allocation `threads × entry quota`, not 65536; going up to 65536 hangs on `.inc`.**
    `.inc` only takes from the pool this CTA was given at launch, pool = entry quota × threads (384 threads:
    168 × 384 = 64512), 1024 less than the whole register file. GPU readings (regsplit probe, same job,
    four variants each in its own subprocess with its own 900 s budget): 24/224/256 (Σ = 64512) compiles, launches, numerically correct;
    32/224/256, 48/208/256, 72/184/256 (Σ = 65536) all three compile to `USETMAXREG = 3`, all **report no error,
    do not spill, sit out the full 900 s and get killed** — the takers wait on `.inc` for a pool chunk the giver can never hand over; that is gotcha 4's form,
    just occurring at the cell where "the arithmetic looks exactly legal". Three 128-thread roles: sum of levels ≤ 504; on top of 24/224/256,
    "+8 for someone" is already a transfer, not headroom. When sweeping, pin the sum at 504 and only move between roles.
14. **"A few `STL` + then one `LDL` at the head of each segment" does not mean those segments are over budget; long-lived values are pushed out by a peak segment; the lever is shrinking the
    peak segment, not the split.** Certified reading: taker at level 256, `LDL=15 STL=5` all in the `.inc` region, line info places the
    `LDL` at the first statement of each subsequent segment and at the item epilogue — what gets pushed out are branch-level long-lived values (hoisted partitioned tensors,
    item scalars, mbarrier bases), and what pushes them is a TMEM round-trip segment that reads all 128 fp32 into registers at once, then packs / scales / writes back.
    Fix: **batch this segment by columns** — two batches of 64 columns each (two `Ld32x32b`), each batch
    t2r → `fence_view_async_tmem_load()` → consume → r2t, with this batch's fp32 live range ending before the next batch loads;
    when the publish granularity is already cut by half-segments, no barrier / arrive / phase change is needed between batches. Peak fp32 128 → 64.
    Eval reading landed in the same package as the four pieces: `spill_ld/st_sectors` 856,704 / 415,808 → 25,760 / 9,024
    (30×, not zeroed — the remainder is in the shared head of the `.dec` region, see the acceptance table).
15. **The levels are load-bearing only when "the receiver is the role where the spills are"; attribute statically first, then spend eval.** (KEEP certification, 512 threads = four
    128-thread warpgroups, entry quota 128, sum of the four levels pinned at 512.) The change is only the immediates of four `setmaxnreg`; the `USETMAXREG`
    count and positions are unchanged — numerically invisible, and no load-bearing axis on the dossier either (`rings` flipped both times, as a follower, see
    [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11, fourth tree). Read the footprint from `spill_ld/st_sectors`
    and the per-region `LDL` / `STL` counts. Two KEEPs on the same tree that changed only immediates:

    | Levels (four roles) | What was moved | `spill_ld / st_sectors` | gate window |
    |---|---|---|---|
    | 184/128/72/128 → 192/120/72/128 | 8 registers, from one non-spilling role to another non-spilling role | 54262224 / 36454592 → unchanged | 1.3169 → 1.3203 ms (+0.26%, no value) |
    | 184/128/72/128 → 184/96/72/160 | 32 registers, from a role with headroom to **the role where the spills are** | 4653056 / 3279168 → **0 / 0** | 1.0564 → 0.9888 ms (−6.40%), `tensor_pct` 55.85 → 62.76 |

    The difference between the two is not how much was moved but **to whom**. The region method (this entry's "Acceptance" section) gave the answer before the first one:
    compile each of four candidate levels once, cut regions at `USETMAXREG` and count `LDL / STL` —

    | Candidate | Total | role 0 | role 1 | role 2 | role 3 | outside role bodies |
    |---|---|---|---|---|---|---|
    | 184/128/72/128 | 44/38 | 2/0 | 2/0 | 2/0 | 23/28 | 15/10 |
    | 192/120/72/128 | 44/38 | 2/0 | 2/0 | 2/0 | 23/28 | 15/10 |
    | 176/128/80/128 | 54/48 | 3/9 | 2/0 | 2/0 | 23/28 | 24/11 |
    | 168/128/88/128 | 74/65 | 14/17 | 2/0 | 2/0 | 19/30 | 37/18 |

    — the bulk of the spills is in role 3, and all four candidates pin role 3 at 128: **no level in the candidate set can touch the role where the spills are**,
    and per-level eval can only read noise (the first row of the table above is exactly that). Three rules for reading it: run the candidate set through the region table first; if the spilling role's level is the same in all
    candidates ⇒ the candidate set is wrong, add a level first and then sweep; choose as giver a role that **measurably has headroom** (`LDL 2 / STL 0`, and still
    0 after lowering its level), not a role that "looks idle"; a role that starts spilling below some level (role 0 in the table above: 184 → 176 → 168 reads
    2/0 → 3/9 → 14/17) cannot be a giver. One compile per candidate, no GPU timing: the static table of four levels is cheaper than four evals.
16. **Two warpgroups sharing one role body, each `setmaxnreg`'d to a different level: the lower level spills.** A role body parameterized by warpgroup
    number is entered by two warpgroups, with `setmaxnreg` written as a branch on the warpgroup number inside the body (one side `.dec 96`, the other
    `.inc 160`). Landing reading (that step also **halved** each thread's working set): `spill_ld/st_sectors` 0 / 0 →
    21037056 / 12087040, gate 0.9844 → 1.0548 ms (+7.15%), `tensor_pct` 64.35 → 60.12. Then changing only immediates:
    184/120/72/136 reads spill 0 / 0, 1.0045 ms; 184/128/72/128 reads 0 / 0, 1.0056 ms. The tree's record explains it as
    "a shared code region is allocated only once, at the lower of the two levels" `[INFERRED: consistent with the readings, no isolating experiment]`.
    The actionable part: once a role body is shared, the levels of its two entrants must be set **together**; the extra registers of the higher level are of no use to this body
    (184/120/72/136 and 184/128/72/128 read flat); when spreading a role body onto a warpgroup that previously belonged to another role,
    land the level re-split in the same package as the spreading, or accept and record the spill reading of the intermediate step.
17. **`launch(..., max_number_threads=...)` replaces `.reqntid`, and ptxas no longer sizes registers by the actual `block`.** The DSL auto-generates `nvvm.reqntid` from
    `block` only when `max_number_threads` is not given; when it is given, it only emits PTX `.maxntid X, Y, Z` and no `.reqntid`, regardless of how many threads are actually launched. It compiles, runs, numerics all correct,
    it just spills. Certified readings (the same toy kernel, the same `block`, one process and one dump directory per variant): with only `block` (PTX `.reqntid`) REG 167, no spills (INFO);
    `block` + `min_blocks_per_mp=1` → PTX `.reqntid` + `.minnctapersm 1`, no `.maxntid` / `.maxnreg`, REG 167, `LDL = STL = 0`;
    the same `block` with `max_number_threads=[1024, 1, 1]` instead (no `min_blocks_per_mp`) → PTX has only `.maxntid 1024, 1, 1`, REG 32, `LDL / STL` 133 / 133,
    numerics still bit-correct. `floor(65536 / 1024) = 64` is only the theoretical cap from `.maxntid`; it actually got 32, below that (whether because `.minnctapersm` is missing is untested). Three layers of acceptance: `grep -E '\.(reqntid|maxntid|minnctapersm|maxnreg)\b' *.ptx`, the
    `REG` of `cuobjdump -res-usage`, the `LDL` / `STL` counts in SASS. **`LOCAL:0` from `-res-usage` does not mean no spills**: at the REG 32 level above `LOCAL` is still 0 while `LDL/STL` is
    133 / 133 — trust the SASS counts (and NCU `spill_st / spill_ld`). Whole kernel: this gotcha itself is untested on a whole kernel (the whole kernel where it was discovered did not use `max_number_threads`);
    that whole kernel landed `min_blocks_per_mp=1` in the same package as another memory-access rewrite, so the reading cannot be attributed separately.

## Sources

- **PTX ISA**: *Miscellaneous Instructions: `setmaxnreg`* (`setmaxnreg.{inc,dec}.sync.aligned.u32 imm`,
  imm ∈ [24, 256] and a multiple of 8, all threads of the warpgroup execute the same instruction; `.inc` blocks until the pool has enough free registers)
  — https://docs.nvidia.com/cuda/parallel-thread-execution/ .
- **cutlass C++ wrapper**: `warpgroup_reg_alloc<N>` / `warpgroup_reg_dealloc<N>` in `include/cutlass/arch/reg_reconfig.h`;
  `CUDA_CTA_RECONFIG_ACTIVATED` is defined under
  `__CUDA_ARCH_FEAT_SM90_ALL` / `__CUDA_ARCH_FEAT_SM100_ALL` (and the SM101/103/120 families)
  — the basis of gotcha 9.
- **CuTe DSL spelling**: `cutlass.cute.arch.setmaxregister_increase / setmaxregister_decrease`
  (old names `warpgroup_reg_alloc / warpgroup_reg_dealloc` are `@deprecated`),
  `cutlass.cute.arch.make_warp_uniform`, `cutlass.cute.arch.warp_idx`;
  `KernelLauncher.launch(**kwargs)` → `LaunchConfig.min_blocks_per_mp`;
  `_generate_kernel_attrs` generates `nvvm.minctasm` only when `min_blocks_per_mp > 0`, and `nvvm.reqntid`
  is generated automatically from the block size — the DSL-side basis of gotcha 1.
- **C7508 / launch-bounds conflict (gotchas 1, 2)**: official answers on the NVIDIA developer forums
  https://forums.developer.nvidia.com/t/setmaxnreg-ignored-unable-to-determine-register-count-at-entry/361145
  and https://forums.developer.nvidia.com/t/hopper-launch-bounds-and-setmaxnreg-conflicts/317408 .
- **`.inc` blocking and oversubscription deadlock (gotcha 4)**: https://forums.developer.nvidia.com/t/about-setmaxnreg/356190 .
- **Compiled-artifact observation channel**: `CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_KEEP_PTX=1` + `CUTE_DSL_DUMP_DIR`,
  `nvdisasm -c` / `cuobjdump -sass` to count `LDL` / `STL` / `USETMAXREG`, `cuobjdump -res-usage` to read
  `REG` / `LOCAL` (the same machine-read criterion channel as [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md)).
- **GPU certification probe**: 384-thread CTA
  (24 / 224 / 256, Σ = 64512), takers with 192 simultaneously live int32 per thread (> 168, < 224), prelude with
  56 (> 24, < 168), working set built per gotcha 10 (inline-PTX XOR chain). **Four compiled variants differing in only two knobs
  compute the same function**; PASS only when all eight contracts pass:
  `uniform_control_spills` (no setmaxnreg, no bound: `LDL+STL > 0`, first proving the control really spills),
  `fused_numerics` (single branch + prelude first + bound: all 384 threads exactly equal to numpy lane by lane, `.inc` did not deadlock),
  `fused_no_spill` (`LDL == STL == 0`, `LOCAL == 0`), `fused_reconfig_in_sass` (`USETMAXREG ≥ 3`),
  `nobound_numerics` / `nobound_reconfig_dropped` (same as fused but without bound: `USETMAXREG == 0` and spills, gotcha 1),
  `split_numerics` / `split_prelude_spills` (three early `if`s move the budget, prelude after, then the role branch: spills, gotcha 11).
  The region method (`LDL/STL` between each `USETMAXREG`) is printed as a fact row with every variant.
  **Status: ✅ `PROBE PASS max_diff=0.0`** (certification round 1). GPU readings (sm_100a,
  `.reqntid 384` / `.minnctapersm 1` / dec 24 / inc 224 / inc 256, REG=168):
  `uniform_control_spills` LDL=56 STL=56 USETMAXREG=0; `fused_no_spill` / `fused_reconfig_in_sass`
  LDL=0 STL=0 LOCAL=0 USETMAXREG=3 (LDL/STL all 0 in the three regions, instrs=1808); `nobound_reconfig_dropped`
  LDL=56 STL=56 USETMAXREG=0 (the same function with only `min_blocks_per_mp` removed: ptxas drops all the `setmaxnreg` = gotcha 1);
  `split_prelude_spills` LDL=379 STL=379 USETMAXREG=3, region reading `{dec:0/0, inc:0/0, inc:0/0, end:379/379}`
  (all spills in the join region after the last `USETMAXREG`, instrs=2584 = the gotcha 11 footprint: dec/inc present, still spilling);
  all three numerics contracts exactly equal to numpy lane by lane (max_diff=0.0), no deadlock on `.inc`.
- **Gotcha 11 form C, gotcha 13, gotcha 14 and the line-info attribution in acceptance: per-round GPU probe + KEEP evaluation**. Regsplit probe
  (the four pieces on a real three-role kernel; contracts R1 PTX four pieces / R2 `USETMAXREG ≥ 3` /
  R3 `LDL == STL == LOCAL == 0` + whole-machine smoke on five shapes, same job each round): round 1 R1, R2 and smoke all passed, R3
  `LDL=15 STL=5`, regions `{dec: 0/2, inc: 4/3, inc: 9/0, end: 2/0}` (the form C footprint); round 2, after one wrap-up copy per branch,
  the `STL` in the `.dec` region dropped to zero with the total unchanged, and `CUTE_DSL_LINEINFO=1` attribution placed the `LDL` at the head of each subsequent segment (the gotcha 14 footprint);
  round 4 aggregated by (region, source line): 14 of 19 in the `USETMAXREG.DEALLOC.CTAPOOL` region, with source lines all in the role branch's shared head;
  round 5: the three Σ = 65536 variants hung for 900 s, the one Σ = 64512 variant launched and was numerically correct (gotcha 13; reading table at the top of the probe file).
  **KEEP evaluation of the four pieces + column batching in one package** (NCU dossier census + gate window, eval job id not recorded): SASS `USETMAXREG`
  0 → 3, `regs_per_thread` 168 unchanged, `spill_ld/st_sectors` 856,704 / 415,808 → 25,760 / 9,024, gate
  0.19544 → 0.20533 ms (+5.06% — **in this structure**: 1 CTA/SM, three roles 24/224/256, the giver's `.dec` region still spills;
  ms is not load-bearing, the value of the four pieces is read on the spill sectors); contract axis `rings` `[]` → `[stride=0x1, stride=0x3f]`, zero mbarrier
  source changes, interpretation in [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11.
- **Gotchas 15, 16: GPU probe + KEEP evaluation** (another tree: 512 threads, four roles, pair UMMA, 1 CTA/SM). Gotcha 15's region table:
  a GPU probe, after three rounds `PROBE PASS (max_diff=0.00036487803405893526) — 6/6 PASS` (each of four candidates
  compiled once, regions cut at `USETMAXREG` + `CUTE_DSL_LINEINFO=1` attribution, plus whole-machine numerics at one level; job id not recorded).
  Gotcha 15's two KEEPs (NCU dossier census + gate window; eval job id not recorded in the status record): axis `rings` `[]` →
  `["stride=0x1"]`, gate 1.3169 → 1.3203 ms; axis `rings` `["stride=0x1,slots=9"]` → `["stride=0x1,wrap=sel"]`,
  gate 1.0564 → 0.9888 ms (the confirmation run of that reading read 0.9859 ms). Gotcha 16's KEEP: axis `rings`
  `[…, "stride=0x30a6f,wrap=mask"]` → `[…, "stride=0x3126f,wrap=mask"]`, gate 0.9844 → 1.0548 ms; the static readings of the two levels after it
  come from another probe (`PROBE PASS (max_diff=0.00036487803405893526) — 7/7 PASS`:
  176/128/80/128 reads 4/4, 168/136/72/136 reads 8/9, 184/120/72/136 and 184/128/72/128 read 0/0).
- **Gotcha 17: GPU probe**,
  `PROBE PASS max_diff=0.0`, round 1 of this certification, all six contracts passed (the three variants numerically bit-correct; the `min_blocks_per_mp=1` variant's PTX has
  `.reqntid` + `.minnctapersm 1`, REG 167, `LDL = STL = 0`; the `max_number_threads=[1024,1,1]` variant's PTX has only `.maxntid 1024, 1, 1`,
  REG 32, `LDL / STL` 133 / 133); cutlass-dsl 4.4.2, sm_100a. The generation rules for `nvvm.reqntid` / `nvvm.maxntid` / `nvvm.minctasm` are in
  `cutlass_dsl`'s `_generate_kernel_attrs` and `LaunchConfig` (`max_number_threads`). There is also a `--ptxas-options=-maxrregcount=96`
  variant whose dumped cubin is still REG 167 (INFO, outside the certification coverage, no conclusion drawn).
- Gotcha 12's reading: a KEEP evaluation on an sm90 tree, transcribed; the factual sources of this entry's other gotchas are official docs / official source / NVIDIA answers + this repo's GPU probes.
