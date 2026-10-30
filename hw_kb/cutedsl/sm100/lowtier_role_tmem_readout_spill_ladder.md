# lowtier_role_tmem_readout_spill_ladder (TMEM readback in a low-tier role: splitting inline-PTX `tcgen05.ld` from `.x32` into `.x16`; the "two-pass read" spelling; spill census by role and by source line; spills are decided by the tier, and narrowing the readback is not a monotone gain)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**.
>
> **Status: ✅ Verified** -- certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=0.0` (round 3 of this certification): `twopass_bits` 0.0, `spill_in_dec_role_only` (`dec(56)` LDL 8 / STL 10, other regions 0), `tier_relief_footprint` (the same source under `dec(104)`: LDL = STL = LOCAL = STACK = 0), `ctrl_bits` 0.0, all four pass in the same round; the fact lines agree with the gotcha 10 table (`split_hold` NL = 22: 15 / 16; `.x32` NL = 8: 0 / 0).
>
> Division of labour with [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md): that one covers the general arithmetic of four levers for readback segments written with CuTe's `Ld32x32bOp`
> in **high-tier** (112 / 128 / 176) receiving roles. This entry covers only a corner it does not:
> in a **`.dec` role with a tier of only 48–64**, the readback is written in **inline PTX** (one asm with
> `tcgen05.ld … .x32` immediately followed by `tcgen05.wait::ld`), and the segment has the shape **a whole-row reduction must precede any element-wise operation**
> (row max / row abs-max / row sum, which must be complete before element-wise normalisation). In this shape "split into two `.x16`" does not by itself lower the peak,
> and "re-read from TMEM in a second pass" lowers the peak **in theory** -- but measured on the GPU it is **not guaranteed** to reduce spills (see the measured table in gotcha 10).
> What this entry certifies: the two-pass spelling is bit-correct, a census by `setmaxnreg` region + source line pins the spills to the low-tier role,
> and changing only the tier of the same source zeroes that region's spills. A narrowed readback is a candidate to be **verified one by one with the census**, not a fixed recipe.

## Purpose

`tcgen05.ld.sync.aligned.32x32b.xN.b32` writes **N 32-bit registers** in one instruction; written in the same asm as
`tcgen05.wait::ld.sync.aligned`, these N outputs are **all live at the same time** from the moment the asm ends
until the last consumer -- ptxas cannot split an asm's output set. In a role with tier 56:

```
peak(segment) = N (readback fragment)
              + values resident across rounds (loop-invariant per-thread scalars / remote addresses / barrier addresses / row constants)
              + addresses and loop counters (around ten)                 ≤ 56
```

With `N = 32` only about 24 are left for the last two items; with a few more resident values, ptxas first moves **the resident values** to local
(one `STL` before the loop, an `LDL` at each use every round), and if that is still not enough it moves the fragment itself. These `LDL` execute in every loop round,
showing on NCU as a large `spill_ld_sectors` and that role's `stall_long_scoreboard` dominating; the downstream MMA chain waiting for this role
to publish its results idles accordingly (`tensor_pct` drops).

A segment of the **reduce-first, element-wise-after** kind has an additional constraint: the element-wise step uses the whole row's reduction value, so the general lever
"read a chunk, use a chunk, drop a chunk" does not work -- all 32 columns of the row must live through the reduction. The way out is that TMEM itself is re-readable storage:

- **First pass**: read a chunk with `.x16` → fold it into the reduction accumulator → the chunk dies; read the next chunk → fold → dies. Peak 16 + 1.
- **Second pass**: **re-read** the same `.x16` chunk from TMEM → element-wise → pairwise pack → write out → dies; likewise for the next chunk. Peak 16 + the reduction value.

The cost is one more `tcgen05.ld.x16` + `wait::ld` per chunk (on chip, not leaving the SM); what it hopes to replace is several local-memory
round trips per round (through L1, falling to L2 on a miss). **But the peak arithmetic above is only a live-value count; ptxas's actual allocation at tiers 48–64 does not follow it**:
measured, the two-pass read spilled with 14 resident values while `.x32` did not (gotcha 10). So every spelling is judged by the census reading.

## When needed (trigger)

**Structural trigger**: in a role whose `setmaxregister_decrease` tier is ≤ 64, there is an asm that reads back ≥ 32 columns at once,
and (a) the readback values must first be reduced over the whole row before element-wise use, or (b) the role must also keep ≥ 16 values resident across rounds (remote `mapa` addresses,
barrier addresses, row constants, prefetched table entries).

**Symptom triggers**:

| What you see | Hit |
|---|---|
| NCU `spill_ld_sectors` far larger than `spill_st_sectors` (stored once, read many times per round), `stall_long_scoreboard` dominating in the role that carries the readback | resident values squeezed out by the readback fragment -- this entry |
| Region census (below): `LDL/STL` all in **one** low-tier region like `dec(56)`, other regions 0 | this entry; when the spills are in an `.inc` region go to [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) |
| Line attribution: `STL` on the lines that load resident values before the loop, `LDL` on the loop-body lines that use them; or `STL/LDL` in runs on the readback asm and reduction lines | the former = resident values squeezed out; the latter = the fragment itself moved -- both driven by the readback width |
| +8 tier for this role gives a small positive gain, adding registers to the neighbouring high-tier role ≈ 0 | the spills really are in this low-tier role; but +8 cannot catch up with a 16-wide deficit, so take ladder step r1 first |
| You have already split one `.x32` into two `.x16`, and the spill count barely changed (or even rose) | the segment is of the "reduce first, element-wise after" shape and both chunks live until the element-wise step -- you can try the r1′ two-pass read, but it may equally fail to lower it (gotcha 10); if the census does not drop, roll back and go straight to r3 |

## Minimal runnable example

128 threads (one warpgroup, tier 56) each hold one TMEM row of 32 f32 columns; each round: take the row max `m`, then element-wise
`y = (x − m) · s + c[j]`, pairwise packed into bf16x2 and written out. `c[]` is NL loop-invariant per-thread values (creating resident pressure).
The certification probe's `twopass_x16` variant is exactly this segment.

```python
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ATT = llvm.AsmDialect.AD_ATT


def _v(x, t, loc, ip):
    return t(x).ir_value(loc=loc, ip=ip)


def make_tmem_ld(n):
    """One asm: tcgen05.ld 32x32b.xN + wait::ld. The N outputs are all live from here on."""
    @dsl_user_op
    def ld(taddr, *, loc=None, ip=None):
        outs = ",".join("$%d" % i for i in range(n))
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * n), [_v(taddr, Int32, loc, ip)],
            "tcgen05.ld.sync.aligned.32x32b.x%d.b32 {%s}, [$%d];\n\t"
            "tcgen05.wait::ld.sync.aligned;" % (n, outs, n),
            ",".join(["=f"] * n) + ",r,~{memory}",
            has_side_effects=True,          # the two passes must not be merged into one (gotcha 4)
            is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
        return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(n)]
    return ld


tmem_ld16 = make_tmem_ld(16)
# fmax2 / pack_bf2 / stg_u32 / ldg_f32: one inline asm each of max.f32 / cvt.rn.bf16x2.f32 / st.global / ld.global.nc
# (has_side_effects=False for pure computation), written the same way as above.


def tree_max(v):
    v = list(v)
    while len(v) > 1:
        v = [fmax2(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)] + ([v[-1]] if len(v) % 2 else [])
    return v[0]


# ---- written inside the @cute.kernel body, in the low-tier role's branch ----
if wg == SEG_WG:
    cute.arch.setmaxregister_decrease(56)          # the first statement of the branch
    tbase = ld_shared_u32(hold)                    # read the TMEM base after the dec (it does not cross the tier boundary)
    lane_t = tbase + ((32 * (warp % 4)) << 16)     # TMEM address: lane in the high 16 bits, column in the low 16 bits
    c = []
    for i in cutlass.range_constexpr(NL):          # loop-invariant resident values
        c.append(ldg_f32(c_ptr + 4 * (tidx * NL + i)))
    for r in cutlass.range(n_rounds, unroll=1):
        s = scale_of(r)
        ob = out_ptr + 4 * ((r * 128 + tidx) * 16)
        # first pass: reduce chunk by chunk; a chunk dies once folded into m
        m = tree_max(tmem_ld16(lane_t))
        m = fmax2(m, tree_max(tmem_ld16(lane_t + 16)))
        # second pass: re-read each chunk from TMEM; it dies once computed, packed and written out
        for ch in cutlass.range_constexpr(2):
            y = tmem_ld16(lane_t + 16 * ch)
            for k in cutlass.range_constexpr(8):
                j = 16 * ch + 2 * k
                a = (y[2 * k] - m) * s + c[j % NL]
                b = (y[2 * k + 1] - m) * s + c[(j + 1) % NL]
                stg_u32(ob + 4 * (8 * ch + k), pack_bf2(a, b))
    ...                                            # teardown (dealloc etc.) stays inside this branch
else:
    cute.arch.setmaxregister_increase(REG_HI)      # Σ tiers unchanged (e.g. 56 + 3 × 152 = 512)
```

Control group (same tier, same amount of resident values; compiled in the certification probe as a negative control):

```python
x = tmem_ld32(lane_t)          # one .x32 + wait: 32 registers live from here to the last pack
m = tree_max(x)
for k in cutlass.range_constexpr(16):
    stg_u32(ob + 4 * k, pack_bf2((x[2 * k] - m) * s + c[(2 * k) % NL],
                                 (x[2 * k + 1] - m) * s + c[(2 * k + 1) % NL]))
```

### Spill census by role and by source line

```python
import os, re, subprocess
os.environ.update(CUTE_DSL_KEEP_CUBIN="1", CUTE_DSL_DUMP_DIR=d, CUTE_DSL_LINEINFO="1")
compiled = cute.compile(launch, ...)               # a .cubin appears under d/

sass = subprocess.run(["nvdisasm", "-c", "-g", cubin], capture_output=True, text=True).stdout
# fallback: nvdisasm --print-line-info -c; without line info the region method still works, just without source lines
OPC  = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
SRC  = re.compile(r'//##\s*File\s*"([^"]+)",\s*line\s*(\d+)')
region, src, tally = "prelude", "?", {}
for ln in sass.splitlines():
    if (m := SRC.search(ln)):
        src = "%s:%s" % (os.path.basename(m.group(1)), m.group(2)); continue
    if not (m := OPC.search(ln)):
        continue
    op = m.group(1)
    if "MAXREG" in op:                             # region cut point
        imm = re.search(r"0x[0-9a-fA-F]+", ln.split(op, 1)[1])
        region = "%s(%d)" % ("dec" if "DEALLOC" in op else "inc", int(imm.group(0), 16))
    fam = op.split(".")[0]
    if fam in ("LDL", "STL"):
        t = tally.setdefault((region, src), [0, 0]); t[fam == "STL"] += 1
for (reg, s), (nl, ns) in sorted(tally.items(), key=lambda kv: -sum(kv[1])):
    print(reg, s, "LDL", nl, "STL", ns)
```

How to read it:

- **Regions**: `USETMAXREG.DEALLOC` opens a `dec(tier)` region, `USETMAXREG.TRY_ALLOC` opens an `inc(tier)` region,
  and each region is compiled at that instruction's tier. Each role does its dec/inc as the first statement of its own branch, so "region ↔ role".
  When several roles share a tier the immediates are the same and the regions cannot be told apart -- then give each role a **different** tier for the census (still keeping Σ),
  or separate them by source line.
- **Source lines**: within a region, aggregate further by line. `STL` on load lines outside the loop and `LDL` at uses in the loop body = resident values squeezed out;
  `STL`/`LDL` right at the readback asm line = the fragment itself moved.
- **Static counts ≠ dynamic cost**: one `LDL` in the loop body executes once per round; one `STL` outside the loop executes only once.
  When comparing two versions look at "the number of `LDL` in the loop body", not only the total; NCU's `spill_ld_sectors` is the dynamic total --
  use it to confirm the direction, and use this census to locate the line.

### Yielding ladder (this role; rerun the census after every step)

| Step | Action | When to use |
|---|---|---|
| r1 | `.x32` → 2 × `.x16`, each asm with its own `wait::ld`, each chunk "read, used, written out" before issuing the next | the segment's consumer is per-chunk (does not need the whole-row reduction value). **Candidate**: roll back if the census does not drop |
| r1′ | two-pass read: in the first pass each `.x16` chunk dies once folded into the reduction value; in the second pass each chunk is re-read from TMEM, computed element-wise, packed and written out | the segment is "whole-row reduction first, element-wise after" -- r1 alone does not lower the peak. **Candidate**: measured, it can be worse than `.x32`; roll back if the census does not drop |
| r2 | do not hoist resident addresses: derive remote / barrier / slot addresses from a seed recomputed every round (the `opaque` seed spelling of `tmem_readout_segment_live_set` lever (3)), or recompute them in place per chunk | after r1/r1′ the census still shows "`STL` outside the loop + `LDL` inside the loop" landing on address-computation lines |
| r3 | raise this role's tier (the value of `setmaxregister_decrease`, in steps of 8), taken from a role whose census is 0 and whose tier has room, keeping Σ unchanged ([`warp_specialization`](warp_specialization.md) "choose your own split"). Roles may have different tiers, e.g. 64 + 152 + 152 + 144 = 512 | the only step **certified to always work**: the region budget is the tier, and once the tier is enough the region's spills go to zero. Use a tier sweep (the probe's 64/72/80 fact lines) to find the lowest sufficient tier |

## Gotchas

1. **The number of an asm's outputs is the live set.** An asm writing `.x32` + `wait::ld` has all 32 outputs live when the asm returns;
   ptxas will not split it into two halves for scheduling. If you want 16, write an asm with 16 outputs.
2. **Split into two `.x16` but with both chunks kept until the element-wise step, the peak does not drop** (measured NL = 22: `split_hold` 15 / 16, even more than `.x32`'s 11 / 12). In a "reduce first, element-wise after" segment, the first chunk can only be used after the second chunk
   is read back and reduced, so the two chunks' 32 registers are live at the same time, the same as `.x32`. The certification probe's `split_hold`
   variant prints this footprint as a fact line.
3. **The two-pass read requires the TMEM source to stay valid until the second pass has finished reading.** If this TMEM region is written by another role (the MMA accumulator,
   an upstream write), the signal telling it "you may overwrite" (mbarrier arrive / commit) must move to **after the `wait::ld` of the last
   `tcgen05.ld` of the second pass**, with `tcgen05.fence::before_thread_sync` before the arrive.
   Moving it later makes the writer wait this stretch longer; that is r1′'s cost -- measure it on the timeline after shipping; do not look only at spills going to zero.
4. **Keep the readback asm `has_side_effects=True`.** The two passes are the same address and the same asm text; the certification covers the
   `True` spelling. Pure-computation asm (`max.f32`, `cvt`) stays `False` as usual, letting the compiler schedule freely.
5. **Do not fetch the TMEM base before the role branch.** A base read in the prologue lives across the `setmaxnreg` boundary into the low-tier region, becoming
   the longest-lived value in the segment and the first to spill; read it once from smem after the dec (one `ld.shared`). See
   [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) "the confluence tail".
6. **Loop-invariant `ld.global` results are squeezed out first.** ptxas cannot recompute such values (it will not reissue the load), and when the budget is short
   the footprint is "one `STL` before the loop, one `LDL` at each use in the loop body". If they are in smem anyway, you can switch to an `ld.shared`
   right at the use (not resident; whether that is faster is decided by census + timing); otherwise make the readback width give way to them.
7. **Address format**: `taddr = base + (lane_base << 16) + column`. Warp `w` can only access lanes
   `32 · (w % 4) … 32 · (w % 4) + 31`; the second `.x16` chunk is column `+16`, in units of 32-bit columns.
8. **Identify regions by mnemonic, not by text order.** The order in which role branches appear in SASS is not guaranteed to match the source; when counting regions go by
   `DEALLOC`/`TRY_ALLOC` and the immediate. With `USETMAXREG == 0` the whole kernel has only one region -- first fix
   `min_blocks_per_mp` ([`warp_specialization`](warp_specialization.md) gotcha 1).
9. **With `setmaxnreg`, the `REG` reading is the entry tier** (`cuobjdump -res-usage`), not the peak of the low-tier role;
   whether the low-tier role has headroom can only be read from "`LDL == STL == 0` in that region".

10. **Narrowing the readback is not a monotone gain -- measurement overturned the "peak arithmetic".** Tier 56, static `LDL / STL` (all in the `dec(56)` region):

    | Resident values NL | `.x32` control | two-pass 2 × `.x16` | `split_hold` |
    |---|---|---|---|
    | 8 | 0 / 0 | — | — |
    | 14 | **0 / 0** | **8 / 10** | — |
    | 22 | 11 / 12 | 4 / 6 | 15 / 16 |

    The two-pass read is worse than `.x32` at NL = 14 and better at NL = 22; the two-pass read itself spills less at NL = 22 than at NL = 14.
    Conclusion: at tiers 48–64 ptxas's allocation results are sensitive to the spelling and do not change monotonically with "the number of live values". **Do not choose a spelling by arithmetic**:
    compile each candidate once and run the census once, keep the one with the fewest `LDL` in the loop body; if none is 0, raise the tier (r3).
    The tier-sweep readings are in the certification probe's fact lines.
11. **Raising the tier works, and the same source can verify it.** The two-pass read with NL = 14 spills 8 / 10 in `dec(56)`; the same source with only that role
    changed to `dec(104)` (the rest 3 × 136) gives `LDL == STL == 0`. The region budget is that region's `setmaxnreg` tier,
    unrelated to the entry `REG` (gotcha 9).

## Sources

- **PTX ISA**, the "TensorCore 5th Generation Instructions" chapter: `tcgen05.ld` (the `.32x32b` shape, the `.x16` / `.x32`
  repetition, the number of destination registers, the lane range a warp can access), `tcgen05.wait::ld`, `tcgen05.alloc` /
  `tcgen05.dealloc` / `tcgen05.relinquish_alloc_permit`, `tcgen05.fence::before_thread_sync` /
  `after_thread_sync`; `setmaxnreg` in the "Miscellaneous Instructions" section
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- **CUDA Binary Utilities** (`nvdisasm -c -g` / `--print-line-info`, `cuobjdump -res-usage`)
  <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html>
- **CuTe DSL / CUTLASS public API symbols**: `cutlass._mlir.dialects.llvm.inline_asm` / `StructType.get_literal` /
  `extractvalue`, `cutlass.cutlass_dsl.dsl_user_op`, `cute.arch.setmaxregister_decrease` /
  `setmaxregister_increase`, `cute.arch.make_warp_uniform`, `cute.arch.warp_idx`, `cute.arch.barrier`,
  `cutlass.utils.SmemAllocator`, `cutlass.range(n, unroll=1)`, `cutlass.range_constexpr`,
  `cutlass.const_expr`, `launch(min_blocks_per_mp=1)`; the compile-artifact channel `CUTE_DSL_KEEP_CUBIN` /
  `CUTE_DSL_DUMP_DIR` / `CUTE_DSL_LINEINFO`.
- **GPU certification probe**: a 512-thread CTA, one warpgroup
  `dec(tier)` carrying the readback segment, the other three `inc` (Σ = 512); 32 columns per row, 4 rounds. Contracts:
  `twopass_bits` (tier 56, NL = 14: the two-pass read's packed words and row maxima bit-equal to the numpy RNE reference),
  `spill_in_dec_role_only` (region census of the same build: spills > 0 inside `dec(56)`, = 0 in other regions),
  `tier_relief_footprint` (the same source with the tier changed to 104: `LDL == STL == 0`, `LOCAL == STACK == 0`, `USETMAXREG ≥ 2`),
  `ctrl_bits` (the `.x32` control group is numerically correct as well). Fact lines: tier sweeps (56/64/72/80) of the two-pass read at NL = 14 and of `.x32` at NL = 22,
  the footprints of `.x32` / two-pass / `split_hold` at NL = 22 and 8, and each variant's spill histogram by (region, source line).
