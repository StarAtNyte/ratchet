# umma_raw_desc_issue (issue `tcgen05.mma` without the DSL MMA atom: hand-built 64-bit smem descriptor + 32-bit instruction descriptor, issued via inline PTX; a K step = descriptor low word +2, the accumulate bit is a predicate operand, and per step one `elect_one` wraps all of that step's MMAs and the commit)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Prerequisites: [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) (how the whole UMMA path is assembled, DSL atom path),
> [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md) (unrolling by NSTAGE to delete stage bookkeeping; orthogonal to this entry),
> [`elect_one`](elect_one.md), [`ptx_escape`](ptx_escape.md) (how to write inline PTX).
>
> **Status: ✅ Verified (numerics only; no time gain)** — certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (round 3 of this certification; three contracts all passed in the same round: `C0-lowers`, `C1-raw-exact` 0.0, `C2-dsl-exact` 0.0).
> **In the same probe the hand-built path and the DSL atom path have the same per-step issue / completion cycles;
> swapped in alone on a whole-kernel tree, the reading was flat** (see "When needed").

## Purpose

The DSL path `cute.gemm(tiled_mma, acc, tCrA[(None, None, kk, s)], tCrB[...], acc)` + `tiled_mma.set(Field.ACCUMULATE, True)`
lets the MMA atom compute the A/B descriptors from the fragment layout and write the accumulate bit into the instruction descriptor. This entry bypasses the atom, builds the descriptors directly from the PTX ISA bit fields,
and issues `tcgen05.mma.cta_group::1.kind::f16` with inline PTX:

- **the base descriptor of each operand face is computed once outside the loop**; inside the loop only integer additions "stage offset + constant" remain;
- **one step along K inside the same 128 B swizzle atom (one row of 128 B = 64 f16; K16 = 32 B) = descriptor low word +2** (the start-address field is in 16 B units);
  the swizzle is applied by hardware from the absolute address bits, no other field needs changing;
- **the accumulate bit is the instruction's predicate operand** (`enable-input-d`), not state you `set` repeatedly: pass 0 for the first instruction of the whole accumulation, 1 for the rest;
- per step: **the whole warp waits on the full barrier**, then **one** `with cute.arch.elect_one():` wraps all of the step's MMAs together with `tcgen05.commit`
  (commit tracks only tcgen05 operations issued earlier by the thread that issues it, so the MMAs and the commit must come from the same thread).

## When needed (trigger)

- You need to express something the DSL MMA atom cannot give: operand faces not coming from `make_fragment_A/B`, control over descriptor fields yourself, the accumulate bit given directly at single-instruction granularity
  (predicate operand; whether DSL's `set(ACCUMULATE, <runtime bool>)` can do it is untested), or the whole MMA warp is already inline PTX.
- **It is not a speedup technique.** The probe compared the hand-built path with the DSL atom path at two MMA N widths in two modes (streaming: the producer refills the ring as usual; resident: the ring is filled once,
  and the MMA loop repeatedly reads stages that are already ready, zero waiting, measuring pure issue / tensor-pipe cost): in resident mode both paths have identical per-step issue and completion cycles run after run
  (difference < 0.2%); in streaming mode the difference (−0.4% … +2.6%, mixed directions) falls within the run-to-run variation of a single path (4–13%, first launch slowest); in resident mode completion cycles are only
  about 1% above issue cycles, i.e. issue is already throttled by the tensor pipe, and hand-building cannot save anything.
- On a whole-kernel tree (single-CTA issue, several UMMAs per step, the MMA warp's per-step issue segment longer than its wait segment, the ablation removing UMMAs above noise) this entry was swapped in alone:
  gate 0.0152 → 0.0152 ms, same-job ratio 0.8844 → 0.8886 (the tree's most recent reading before the swap vs after; within noise). **Flat**.
  Later landings on that tree continued on the hand-built path; no further DSL-path control was done.

**Not the right remedy**: if you want to shorten the MMA warp's per-step chain — first see the SASS criterion and the negative results in [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md)
(moving instructions in source does not change SASS order; one variant of that probe issued the MMAs back-to-back via inline PTX and the chain got longer).

## Minimal runnable example

Conditions: single CTA (`cta_group::1`), A/B both K-major f16, 128 B swizzle (`cute.make_swizzle(3, 4, 3)`),
each TMA box is `rows × 64` f16 (each row exactly one 128 B atom), box base 1024 B aligned; one stage holds `ATOMS_PER_STAGE` boxes along K (any constant ≥ 1);
fp32 accumulator in TMEM. Producer, barrier init, TMEM allocation / readback are the same as in [`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md);
only the descriptors and the MMA warp are shown here. `lds_u32` is a single `ld.shared.u32` in inline PTX (spelling in [`ptx_escape`](ptx_escape.md)).

```python
import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# ---- smem descriptor (PTX ISA "Shared memory descriptor"), K-major SWIZZLE_128B
#   lo: bits 0-13 start address>>4; bits 16-29 LBO>>4 (unused for swizzled K-major, set to 1)
#   hi: bits 32-45 SBO>>4 (8 rows × 128 B = 1024 B -> 64); bits 46-48 fixed 0b001;
#       bits 49-51 base offset (0 when the atom is 1024 B aligned); bits 61-63 swizzle mode (2 = 128B)
DESC_LO_LBO = 1 << 16
DESC_HI = 64 | (1 << 14) | (2 << 29)
K16_PER_ATOM = 64 // 16                      # a 128 B atom has 4 K16 steps along K (f16)

# ---- instruction descriptor kind::f16: D=f32, A/B=f16, both K-major
def idesc_f16_f32(m, n):
    return (1 << 4) | ((n >> 3) << 17) | ((m >> 4) << 24)   # bits 4-5 D format; 17-22 N>>3; 24-28 M>>4

@dsl_user_op
def umma_f16(taddr, alo, ahi, blo, bhi, idesc, acc, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (taddr, alo, ahi, blo, bhi, idesc, acc)],
        "{\n.reg .pred p;\n.reg .b64 da, db;\nsetp.ne.b32 p, $6, 0;\n"
        "mov.b64 da, {$1, $2};\nmov.b64 db, {$3, $4};\n"
        "tcgen05.mma.cta_group::1.kind::f16 [$0], da, db, $5, p;\n}",
        "r,r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

# ---- inside the kernel (kernel top level, outside any dynamic if)
pA8 = smem.allocate(NSTAGE * ATOMS_PER_STAGE * A_BOX_BYTES, 1024)   # A ring: box pitch A_BOX_BYTES
pB8 = smem.allocate(NSTAGE * ATOMS_PER_STAGE * B_BOX_BYTES, 1024)
a_base, b_base = pA8.toint(), pB8.toint()             # 32-bit addresses within the shared window
hold_addr = tmem_holder.toint()                       # the smem word the TMEM base is written to at allocation

if warp == MMA_WARP:
    # ... after the rendezvous with the allocating warp's TMEM base publication
    taddr = lds_u32(hold_addr)                        # TMEM base
    a_lo0 = cutlass.Int32(((a_base >> 4) & 0x3FFF) | DESC_LO_LBO)   # outside the loop: base descriptor low word
    b_lo0 = cutlass.Int32(((b_base >> 4) & 0x3FFF) | DESC_LO_LBO)
    IDESC = idesc_f16_f32(UMMA_M, UMMA_N)             # Python int: compile-time constant
    A_STAGE16 = ATOMS_PER_STAGE * A_BOX_BYTES // 16   # one stage's span in descriptor units (16 B)
    B_STAGE16 = ATOMS_PER_STAGE * B_BOX_BYTES // 16
    for it in cutlass.range(K_TILES, unroll=1):
        s = it % NSTAGE
        cute.arch.mbarrier_wait(bar_full + s, (it // NSTAGE) % 2)   # the whole warp waits
        with cute.arch.elect_one():                   # one elected region per step: MMA and commit in the same thread
            a_s = a_lo0 + s * A_STAGE16
            b_s = b_lo0 + s * B_STAGE16
            first = (it + (K_TILES - 1)) // K_TILES   # 0 on step 0, 1 otherwise, branch-free
            for u in range(ATOMS_PER_STAGE * K16_PER_ATOM):     # unrolled at trace time (Python range)
                j, kk = u // K16_PER_ATOM, u % K16_PER_ATOM
                acc = cutlass.Int32(1)
                if u == 0:
                    acc = first                       # only instruction 0 of each step carries a runtime accumulate bit
                umma_f16(taddr,
                         a_s + (j * A_BOX_BYTES // 16 + 2 * kk), DESC_HI,   # K step inside an atom = low word +2; across atoms move by the box pitch
                         b_s + (j * B_BOX_BYTES // 16 + 2 * kk), DESC_HI,
                         IDESC, acc)
            tcgen05.commit(bar_empty + s)             # already inside the elected region; do not nest another elect_one
    with cute.arch.elect_one():
        tcgen05.commit(bar_acc)                       # accumulator full -> reader
```

Note: in the probe the unrolled body inside the elected region is written in a plain Python helper function (called at trace time, with no dynamic control flow inside and no DSL value reassigned);
`range` / `if u == 0` are both evaluated at trace time.

## Gotchas

- **The start-address field is only 14 bits, in 16 B units**: `(addr >> 4) & 0x3FFF` covers a 256 KB shared window; inside the loop only additions are done,
  and as long as the address does not leave that range there is no carry into the LBO field. The address must come from the shared window (`ptr.toint()`), not a generic address.
- **The low-word +2 K step only holds inside one 128 B atom**. Across atoms you move by the box pitch (`A_BOX_BYTES // 16`), not by continuing +2;
  the box base must be 1024 B aligned, otherwise the base offset field (bits 49-51) is no longer 0.
- **SBO = 1024 B is the 8 rows × 128 B swizzle period**; under 128 B swizzle K-major the LBO is unused (set to 1). For 64B / 32B swizzle
  the SBO, the swizzle-mode field and the K step all have to change; this entry has not certified those modes.
- **idesc's M field is M>>4 and its N field is N>>3**; D format 1 = f32; A/B format under `kind::f16`: 0 = f16, 1 = bf16; bits 15/16 = 0 mean K-major.
  Legal N values follow the PTX ISA `kind::f16` shape table (for M = 128, N is a multiple of 16); the probe tested only two N.
- **The accumulate bit is a per-instruction predicate**: if the first MMA is passed 1, the stale value in TMEM is added into the result — the probe catches this class of error by prefilling the output with NaN and launching the same compiled entry three times in a row.
  The DSL path's `tiled_mma.set(ACCUMULATE, False)` is equivalent to passing 0 here.
- **When this probe wrapped the whole MMA issue loop (including `mbarrier_wait`) in one `elect_one`, all three rounds segfaulted in `cute.compile`** (subprocess rc = -11;
  in the last two rounds the stack stopped in the DSL's `cute.compile`), with both descriptor spellings and both N. That elected region contained the in-loop inline-PTX MMA + `tcgen05.commit`,
  the out-of-loop timing asm, and the acc commit / wait; **the cause of the crash was not isolated**, so do not read it as "a dynamic loop cannot go inside `elect_one`". The spelling "warp-level wait + one elected region per step"
  is numerically exact. Inside the elected region write `tcgen05.commit` directly (the probe's spelling; nesting another `elect_one` is untested, `[INFERRED]` it would hang); values computed inside the elected region cannot leave it.
- **Build the descriptor from two 32-bit registers (`mov.b64 da, {lo, hi}`)**: assembling `lo | (DESC_HI << 32)` into an `Int64` and passing it directly as the descriptor operand with an `"l"` constraint
  is also numerically identical, but in the probe the per-step issue cycles were longer: about 4.5% more in resident mode (consistent across two certification rounds), 1–3% more in streaming mode.
- **Read the TMEM base from the holder word with `ld.shared.u32`**, after the allocating warp's publication rendezvous.
- **The inline asm must be `has_side_effects=True`**, otherwise an MMA with no outputs is deleted as dead code.

## Sources

- PTX ISA: `tcgen05.mma` (`enable-input-d` predicate operand), Shared memory descriptor, Instruction descriptor (`kind::f16`), `tcgen05.commit` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma>
- Public infra API: `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`, `cute.arch.elect_one`, `cute.arch.mbarrier_wait`,
  `cutlass.cute.nvgpu.tcgen05.commit`, `cutlass.range` / `cutlass.range_constexpr`, `cute.make_swizzle`, `SmemAllocator.allocate`, `Pointer.toint`
- GPU certification probe: single CTA, one TMA warp, one MMA warp, several readback warps;
  the hand-built path (main form: warp-level wait + one elected region per step, descriptor from two 32-bit halves) and the DSL atom control path, each run at two MMA N widths × streaming / resident modes,
  A/B drawn from {-1, 0, 1} so the fp32 result is exact, each configuration launching the same compiled entry three times. `PROBE PASS max_diff=0.0` (round 3):
  `C0-lowers`, `C1-raw-exact` 0.0, `C2-dsl-exact` 0.0. Diagnostic ladder (not contracts): the 64-bit `"l"` spelling 0.0 in both modes; the spelling with the whole loop inside one elected region
  segfaulted in `cute.compile`. Per-step issue / completion cycles are the probe's `[[T]]` information rows (`%clock64`), not contracts.
