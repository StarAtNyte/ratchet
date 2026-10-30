# ldg_issue_pin_structural_fence (a batch of register-fed global loads: LDGs written first in the source get sunk by ptxas in SASS past the first consumer; three structural fences that pin them and several spellings that do not work)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified, three independent certifications by GPU probes (the probes are not included in this release):
> - Spellings A / B and the list of "spellings that do not work": GPU probe, after 3 rounds
>   `PROBE PASS max_diff=0.000133` (numerics: 10/10 variants × tiers within atol 2e-2 of fp64, padding rows 0;
>   the pinning contract "LDG before the first HMMA ≥ the source prologue's lookahead count and no local memory" selected `membar` and `brk`).
> - The "multi-path census artifact" and the fence comparison table: GPU probe,
>   `PROBE PASS max_diff=0.0`, round 1 (all three contracts numeric / single-path pinned / multi-path per-path census pass).
> - Spelling C (independent cut points) and its comparison table: GPU probe, after 6 rounds
>   `PROBE PASS max_diff=0.0` (numerics bit-exact at both tiers; the pinning contract at both tiers "LDG before the first HMMA ≥ the source prologue count,
>   the number of HMMA between each LDG outside the prologue and its first reader ≥ the floor, no `LDL/STL`" all pass).
>
> **What is certified**: only the SASS shape (number of LDG before the first HMMA, distance from LDG to its first reader, REG, `STL/LDL`) and the numerics; **no gain is certified**; the whole-kernel readings are under "When needed".

## Purpose

In supply of the "global → register → `mma.sync`" kind that bypasses smem, the results of a batch of `ld.global{.nc}.v8.b32` (`LDG.*.256`) are used directly as
MMA operand registers. The source issues the whole batch of LDG first and writes the consumers afterwards, hoping to hide the first-byte latency of the loads under the preceding work.
But what really decides the issue position is ptxas's scheduler: under register pressure it **sinks** part of the LDG to just before their first use
(shortening the live ranges), and the lookahead / early issue disappears in SASS. This entry answers: which spellings cannot stop the sinking, which three structures can,
and how to accept it in SASS.

Why source order and LLVM-level ordering points do not work:
- `llvm.inline_asm(..., has_side_effects=True)` and a `~{memory}` clobber only constrain LLVM / NVVM; ptxas gets flat PTX
  and reorders by register dependences, scoreboards and its own pressure heuristics. volatile only guarantees the relative order among these loads; it does not stop the whole group from moving back.
- `ld.global.nc` declares the data read-only, so ptxas need not order it against ordinary stores / fences.
- `bar.warp.sync`, `bar.sync` (CTA-level or one named barrier per warp) only constrain shared / synchronization semantics and are transparent to pure-register MMA.
- A run-time `if` at the DSL level (even if the block contains a side-effecting store) does not work either: in the comparison table of spelling C it is almost the same as no boundary.
- A register cap (`-maxrregcount` / `.maxnreg`, or raising it to 255) is only a ceiling; it does not make the scheduler use more registers.

What can stop it are **structures** ptxas does not cross:
- **A: `membar.cta` after the group** (loads use weak `ld.global`, without `.nc`) -- a real memory fence; loads in the group cannot move past it.
- **B: the whole group in one asm, wrapped inside the asm by a branch on a run-time predicate** -- a basic-block boundary; ptxas does not move loads out of the block.
- **C: loads written one by one as usual, with independent cut points inserted between "finished issuing a group of loads" and "the stretch of HMMA before consuming it"** -- an asm block:
  when a run-time value differs from some constant, `bra` jumps over a `trap`. Also a basic-block boundary; **the constant compared at each cut point must be distinct** (gotcha 10).
- `fence.acq_rel.cta` also pins, but it is a memory-ordering round trip and expensive (see the table below).
- Another family of spellings (a data-dependence token, putting the loads themselves into a single-trip backward-branching loop) is certified on another shape (load results go through FP arithmetic first),
  with its own list of "does not pin"; see [`ldg_issue_order_pin`](ldg_issue_order_pin.md).

## When needed (trigger)

- In SASS the number of `LDG` before the first `HMMA` is less than the number the source issues before consuming; or on the same code path some `LDG.*256` are scheduled after that path's first
  `HMMA` although the source writes them before the consumers; or only one or two HMMA remain between each LDG and its first reader, far fewer than the lead the source arranged.
- `REG` is clearly below the cap (the cap given by `.reqntid` / `setmaxnreg`) while `stall_long_scoreboard` is high.
- Changing the load asm to `has_side_effects=True`, merging into one asm, swapping source order, adding an empty `~{memory}` asm, raising `maxrregcount`:
  the SASS is unchanged or almost unchanged.
- **Rule out the census artifact first (gotcha 1)**: when a kernel has several compile-time paths selected by `blockIdx`, each issuing its own batch, the whole-kernel "number of LDG after the first HMMA"
  counts the second path's entire batch, even if no path sinks anything internally.
- **Whole-kernel readings** (the whole kernels where this entry was found):
  - The kernel of spelling B (register cap 128): pinning all lookahead pushed REG to 128 and spilled (`STL` 20–80), yet the geometric mean was still about 4% faster (0.959 / 0.956);
    another pinning was 9.5% slower; what was finally adopted was an issue table that "pins only some groups", with a same-job patch ratio 0.9646 → 1.0129 and gate 0.0084 → 0.0080 ms
    (in the same package as an issue-table reshuffle; the pin's own share was not separated out).
  - The kernel of spelling C (another register-fed `mma.sync` kernel, two size tiers): a same-job A/B with only the cut points on and everything else unchanged, the smaller tier 7.46 → 6.91 µs,
    7.55 → 6.98 µs (two jobs), REG clearly up (not at the cap), LDG before the first HMMA clearly more, the minimum lead outside the prologue from 0 to positive;
    for the larger tier, with only the cut points on, REG hit 128 with `LDL/STL` 46, not timed (the situation of gotcha 4). It then landed in the same package as an adjustment of the source lookahead depth,
    gate 0.0106 → 0.0083 ms (same package; the cut points' own share is only the two smaller-tier runs above).
  The gain varies by kernel; before pinning, confirm per gotcha 1 that there really is sinking, and after pinning price it same-job.

## Minimal runnable example

### Spelling A: `membar.cta` after the group (what is certified is the combination "weak `ld.global` + `membar.cta`")

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

@dsl_user_op
def fence_asm(text, *, loc=None, ip=None):
    llvm.inline_asm(None, [], text, "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

# After each issue group (all LDG of one or more chunks) has been issued:
#   regs = [ld_v8(addr) for addr in group]     # each "ld.global.v8.b32 {8 x =r}, [$8];"
#   fence_asm("membar.cta;")
```

### Spelling B: the whole group in one asm + a basic-block boundary inside the asm (loads may keep `.nc`)

Write all `ld.global.nc.v8.b32` of one issue group into **one** inline asm, and skip the whole group inside the asm with a predicate that is true at **run time**:

```python
# body (n loads, outputs $0..$(8n-1), addresses $(8n)..$(9n-1), guard $(9n)):
#   { .reg .pred q;
#     setp.ne.b32 q, $G, 0;
#     @!q bra SKIP${:uid};
#     ld.global.nc.v8.b32 {$0,...,$7}, [$A0];
#     ... (the rest of this group's loads)
#   SKIP${:uid}: }
# constraints: "=r"*8n + "l"*n + "r"; result type llvm.StructType.get_literal([T.i32()] * 8n)
# has_side_effects=False is enough (the pinning comes from the branch, not from volatile)
```

The guard value must be a run-time quantity ptxas cannot prove always true (the certification probe uses `M > 0` computed from a kernel parameter); `${:uid}` makes each
expanded instance's label unique. Predicated loads (zero-filled tail block) can go into the same asm: first `mov.b32 $i, 0;` then `@p ld…`.

### Spelling C: independent cut points (`bra` over `trap`, a different constant per cut point; loads may keep `.nc`, one asm per load as usual)

```python
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T as TT
import cutlass
I32, I64 = cutlass.Int32, cutlass.Int64
ATT = llvm.AsmDialect.AD_ATT
MAGIC = 1515870810        # any constant; the run-time flag is always passed as 0, so never equal
_cut_id = [0]             # trace-time counter: one unique constant + unique label per cut point

@dsl_user_op
def ld256(addr, *, loc=None, ip=None):          # a 32 B load fetched straight into registers (address 32 B aligned)
    s = llvm.inline_asm(
        llvm.StructType.get_literal([TT.i32()] * 8), [I64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
    return [I32(llvm.extractvalue(TT.i32(), s, [i], loc=loc, ip=ip)) for i in range(8)]

@dsl_user_op
def issue_cut(flag, *, loc=None, ip=None):
    # One real basic-block boundary: setp + bra over trap. The constant MAGIC+n differs per cut point ⇒ no cut point can be inferred from an earlier one.
    _cut_id[0] += 1
    L = "ISSUECUT_%d" % _cut_id[0]
    llvm.inline_asm(
        None, [I32(flag).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $0, %d; @p bra.uni %s; trap; %s: }"
        % (MAGIC + _cut_id[0], L, L),
        "r", has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# Add a flag: cutlass.Int32 to the kernel signature; the host always passes I32(0).
# In straight-line (range_constexpr-unrolled) code, follow every "finished issuing a group of loads to be issued early" with a cut point:
#
#   regs_g = [ld256(a) for a in group_g]      # a group to be issued early
#   issue_cut(flag)                           # this group's loads are not moved past the cut point
#   ...  HMMA consuming earlier groups          # this stretch of HMMA covers group_g's latency
#   issue_cut(flag)                           # stretch boundaries need a cut too (even if this stretch issued no load), or later loads can slide back and forth
```

### Certified readings

**Spellings A / B**: a toy whose source issues several groups of lookahead loads before consuming; the criterion is **the number of LDG before the first HMMA** in SASS
as a fraction of "the source prologue's lookahead count" (the part above 100% is lookahead loads the source writes separately before the first consuming stretch).
One tier for each of two group sizes, 8 warps/CTA (REG cap 255), plus a counterexample at 16 warps/CTA (cap 128):

| Variant | LDG before first HMMA / source prologue lookahead (larger group tier) | REG | STL/LDL | smaller group tier | cold-L2 median |
|---|---|---|---|---|---|
| base (`.nc`, one asm per load, no fence) | 43% | 78 | 0 / 0 | 50% (REG 72) | 20.26 µs |
| wbar (weak `ld.global` + `bar.sync warp, 32` after each group) | 43% | 78 | 0 / 0 | 50% | 22.66 µs |
| **membar** (weak `ld.global` + `membar.cta` after each group) | **150% (= all of the source's lookahead)** | 206 | 0 / 0 | 150% (REG 170) | 16.70 µs |
| **brk** (`.nc`, the whole group in one asm, wrapped inside the asm by a run-time predicate branch) | **150% (= all of the source's lookahead)** | 202 | 0 / 0 | 150% (REG 170) | 18.50 µs |
| brk16 (same as brk, 16 warps/CTA → REG cap 128) | 150% | 128 | 44 / 78 | 150%, STL 26 / LDL 46 | 20.42 µs |

Earlier rounds on the same toy also tested: `has_side_effects=True`, merging the whole group into one asm (no branch), `bar.warp.sync`, `bar.sync 0`,
one named barrier per warp, REG cap raised to 255 -- all still 43%. 150% corresponds to all the lookahead written in the source:
what gets pinned is **all of the source's lookahead**, not "exactly those prologue groups". The timings are toy readings, not promises.

**Fence comparison**: a single-path toy, one batch of `LDG.*.256` per thread, with one mbarrier spin before consumption:

| Variant | Pin | LDG after the first HMMA (fraction of the batch) | first_hmma@ / last_ldg@ | REG | LDL/STL | time / L_base |
|---|---|---|---|---|---|---|
| L_base | none | 40% | 159 / 202 | 146 | 0 | 1.0000 |
| L_clob | empty `~{memory}` asm | 40% | 159 / 202 | 146 | 0 | 0.9981 |
| L_wsync | `bar.warp.sync -1` | 40% | 160 / 203 | 146 | 0 | 0.9955 |
| L_fence | `fence.acq_rel.cta` | **0** | 169 / 148 | 158 | 0 | **1.2094** |

(`@N` is the SASS instruction index.) Two-path toy (two compile-time paths selected by `blockIdx`): without a pin the whole-kernel reading "after the first HMMA" is nonzero (exactly the second path's whole batch),
and after splitting at `EXIT` both paths are 0 on their own.

**Spelling C and its comparison**: a straight-line toy whose source issues a batch of loads in the prologue, then before each stretch of HMMA first issues the loads needed later;
two tiers (different numbers of HMMA per stretch), 8 warps/CTA (REG cap 255). "Before first HMMA" = the number of LDG before the first HMMA compared with the source prologue count (fewer / equal / more);
"lead" = the number of HMMA between each LDG and the first instruction reading its destination registers (the minimum counts only LDG outside the prologue, the median counts all); spelling C passed the contract's lead floor at both tiers:

| Spelling | smaller tier REG | before first HMMA | lead min / median | larger tier REG | before first HMMA | lead min / median |
|---|---|---|---|---|---|---|
| pure asm loads, no boundary | 64 | fewer than prologue | 1 / 2 | 80 | fewer than prologue | 3 / 4 |
| load asm `has_side_effects=True` | 64 | fewer than prologue | 1 / 2 (SASS as above) | 80 | fewer than prologue | 3 / 4 (as above) |
| boundary = `bar.warp.sync -1` | 76 | fewer than prologue | 2 / 3 | 80 | fewer than prologue | 2 / 4 |
| boundary = DSL run-time `if flag == MAGIC:` wrapping a side-effecting store | 72 | fewer than prologue | 2 / 3 | 86 | fewer than prologue | 2 / 5 |
| boundary = `bra` over `trap`, **the same constant at all cut points** | 78 | = prologue | 4 / 5 | 106 | = prologue | 7 / 10 |
| **boundary = `bra` over `trap`, a different constant per cut point (spelling C)** | **102** | **slightly more than prologue** | **8 / 12** | **136** | **= prologue** | **12 / 17** |
| no boundary, `launch(min_blocks_per_mp=1)` | 154 | more than prologue | 26 / 21 | 156 | more than prologue | 26 / 26 |

With spelling C the smaller tier has slightly more before the first HMMA than the source prologue (ptxas hoisted a little more), and the larger tier exactly equals the prologue.
The same spelling C, larger tier, 16 warps/CTA (REG cap 128): REG 128, `LDL/STL` 22, the minimum lead drops to 2 (median 16) -- fails the contract (gotcha 4).
With the same constant at all cut points, only the first cut point's branch + `trap` remain in SASS (gotcha 10).

## Gotchas

1. **The whole-kernel census of a multi-path kernel is a source of artifacts.** In SASS address order, the second path's whole batch of LDG comes after the first path's HMMA.
   Judging sinking must be done **per path**: split the SASS at unpredicated `EXIT`, each segment (containing HMMA) is one path, and count only
   "LDG after the first HMMA" within the segment. A whole-kernel reading `after_first_hmma > 0` is not evidence of sinking.
2. **volatile is not a fence.** `has_side_effects=True`, merging into one asm, `~{memory}`, `bar.warp.sync`, `bar.sync`
   (CTA-level or named per warp), and a DSL run-time `if` do not change the sinking. Count the LDG before the first HMMA in SASS first, then talk about spellings.
3. **What is pinned is all of the source's lookahead, not the depth you want.** Spellings A / B / C pin all loads written in the source before the fence / block / cut point. If you want exactly N groups,
   you must place the extra group at the right position at the source level and then pin that position with the fence / branch.
4. **The REG cap decides whether you can pin.** After pinning, the live set ≈ the load results in flight at once + the accumulators; exceeding the cap given by `.reqntid` / `setmaxnreg` spills
   (brk16 row: STL 44 / LDL 78 under cap 128; spelling C larger tier at 16 warps/CTA: `LDL/STL` 22, lead back down to 2). Before pinning, do the accounting
   `per-thread cap = 65536 / number of threads` (and ≤ 255); a "pin" with nonzero `STL/LDL` does not count as a win; use the pin only on the path that really sinks.
5. **`membar.cta` has a semantic cost.** It is a real memory fence and also applies to earlier stores / reds of the same thread. Within the certified scope only loads precede it;
   if there are `red` / `st.global` before the issue group, the cost is untested. Only the combination weak `ld.global` + `membar.cta` is certified.
   `fence.acq_rel.cta` pins too, but the single-kernel time is ×1.21; do not use it.
6. **The guard / trip count / cut-point flag must be run-time values.** Spelling B's predicate or spelling C's flag written as a compile-time constant (a Python constant, an `Int32` literal)
   gets folded, and the basic-block boundary disappears with it. Pass it in from a kernel parameter.
   Spelling B de-duplicates the label of each expanded instance with `${:uid}`; spelling C gives each cut point a unique label with a trace-time counter.
7. **After pinning, the MMA of later stretches may be pushed back as a whole** (the tail of the membar sequence is a long run of HMMA): ptxas still schedules MMA freely between fences.
   The criterion looks only at the number of LDG before the first HMMA and at spills; the interleaving of later stretches is not guaranteed by this entry.
8. **Pinned ≠ the first consumer waits less.** With all LDG hoisted, the first consumer may wait at once on more loads sharing one write scoreboard (outside this entry's certified scope).
9. **Different tiers on the same source can diverge too.** Under the same spelling, read the SASS separately for each group size; whether lookahead "survives naturally" depends on the specific kernel's register pressure and cannot be extrapolated.
10. **The cut-point constants of spelling C must each be different.** When every cut point compares the same `flag != MAGIC`, ptxas knows the comparison result after the first cut point
    and deletes the later cut points (the "same constant" row of the comparison table: only one branch + `trap` remain in SASS; the count before the first HMMA is reached, the lead outside the prologue is not).
    Give each cut point `MAGIC + n` with a trace-time counter. Each surviving cut point adds one branch and one `BPT` (trap) in SASS; small overhead, but not 0.
    It is not the same spelling as the non-pinning "a separate asm conditional branch after the loads" in [`ldg_issue_order_pin`](ldg_issue_order_pin.md) gotcha 1:
    there it is one cut point that skips a store, tested on another shape; which point makes the difference (number of cut points / distinct constants or not / skipping a `trap` or a store) has not been A/B-tested separately.
11. **`min_blocks_per_mp=1` is not a pin.** With the register budget released, ptxas hoists loads further on its own (the count before the first HMMA exceeds the source prologue),
    the position is decided by the scheduler and does not follow the source, and REG in the 150s depresses occupancy; it can be tried as a "release the budget" side path, but it cannot implement a specified issue order.
12. **Accept on SASS, not PTX; when measuring the lead, look at both destination register groups of a 256-bit load.** The load order in PTX matches the source and says nothing about the SASS order.
    On sm100 `ld.global.nc.v8.b32` lands as `LDG.E.ENL2.256.CONSTANT Ra, Rb, [addr]`: the destination is **two groups of 4** registers (`Ra..Ra+3`, `Rb..Rb+3`),
    not 8 consecutive registers from `Ra`; when computing the "first reader" both groups must be matched, or the lead is mismeasured. When fetching the cubin, set `CUTE_DSL_KEEP_CUBIN=1`,
    `CUTE_DSL_DUMP_DIR`, a separate `CUTE_DSL_CACHE_DIR` + `CUTE_DSL_DISABLE_FILE_CACHING=1` in the subprocess (otherwise the JIT file cache is hit and no dump appears).

## Sources

- PTX ISA: the vector form of `ld.global{.nc}` (`.v8.b32`), `membar` / `fence`, `bar.warp.sync`, `bar.sync`, `bra` / predicates inside inline asm, `trap`,
  `mma.sync.aligned.m16n8k16`, `mbarrier.try_wait.parity`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Binary Utilities (`cuobjdump -sass`, `cuobjdump -res-usage`, `nvdisasm`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- API symbols of public infra: `cutlass._mlir.dialects.llvm.inline_asm` (`has_side_effects`), `cutlass.cutlass_dsl.dsl_user_op`,
  `cute.jit`, `cutlass.const_expr`, `cutlass.range_constexpr`, `kernel(...).launch(min_blocks_per_mp=...)`.
- Certification probe (spellings A / B): `PROBE PASS max_diff=0.000133`.
- Certification probe (fence comparison, multi-path artifact): `PROBE PASS max_diff=0.0`.
- Certification probe (spelling C and comparison): `PROBE PASS max_diff=0.0`, round 6.
