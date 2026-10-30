# tmem_chunk_e4m3_quant_chain (TMEM chunked read-back → per-chunk amax → UE8M0 → e4m3 packing → 256-bit store: make this per-chunk chain one branch-free straight-line block, and hoist the next chunk's `tcgen05.ld` explicitly)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**. For the register peak of the read-back segment see [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md);
> for the ordering edges between read-back and face writes / fences / rendezvous see [`tmem_ld_epilogue_issue_order`](tmem_ld_epilogue_issue_order.md).
> This entry covers one thing only: what the **per-chunk MX quantisation chain** (one UE8M0 scale per 32 f32, e4m3 payload, one `st.global.v8.b32` per chunk) looks like in SASS, which step chops it up, and how to write it as straight-line code that can be scheduled across chunks.
>
> **Status: ✅ Verified** (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=0.0`, certification round 2; five contracts: the division / integer / integer + look-ahead variants have bit-exact payload and SF (including boundary chunks), the division-free variant has `FCHK` 0 and `CALL` 6 → 2, the look-ahead variant has `LDTM` = 4 before the first `STG`. Only values and static SASS are certified, no whole-kernel reading; toy cycle counts are INFO: fused form 999 / look-ahead 1031 cyc/round, and in the fused form ptxas already hoisted 2 `LDTM`s, see gotcha 5).

## Purpose

One chunk (32 f32, the 32 columns of one row per thread) has to do:

```
x[0..31]  ← tcgen05.ld 32x32b (one .x32, or two .x16)
amax      ← max |x|                         (abs + 5-level max tree)
e         ← UE8M0 byte: the smallest s = 2^(e−127) such that amax / s ≤ 448
q[i]      ← x[i] · 2^(127−e)                 (multiply by a power of 2, exact)
w[0..7]   ← cvt.rn.satfinite.e4m3x2.f32 ×16 + word assembly
st.global.v8.b32  [row + 32·k], w            (one 256-bit store)
```

Two hardware-level facts determine how short this chain can be:

1. **`div.rn.f32` is not one instruction.** IEEE correctly rounded f32 division in SASS is "`MUFU.RCP` + a few `FFMA` Newton iterations + an `FCHK` test + a conditional branch to a slow-path subroutine (`CALL`)".
   One division per chunk = one **basic-block boundary** per chunk — ptxas only schedules within a basic block, so the next chunk's `LDTM` is never moved ahead of this chunk's division,
   and the whole epilogue is chopped into CH segments, each a serial chain (even when the chunks are already unrolled with `range_constexpr` in source).
2. **"÷448 then round up to a power of 2" has an exact division-free equivalent.** 448 = 1.75 · 2⁸. Let the bit pattern of `amax` be `b`, with exponent field `Ea` and mantissa field `ma`:
   - `ma ≥ 0x600000` (mantissa ≥ 1.75): `amax/448 = 2^(Ea−135) · r`, `r ∈ [1, 8/7)`, rounded-up exponent = `Ea − 8 + [ma > 0x600000]`;
   - `ma < 0x600000`: `amax/448 = 2^(Ea−136) · 2r`, `2r ∈ (1, 2)`, the mantissa is necessarily non-zero, exponent = `Ea − 8`.

   Combined: **`e = ((b + 0x1FFFFF) >> 23) − 8`** (`ma + 0x1FFFFF` carries into the exponent field ⇔ `ma > 0x600000`), then clamp to `[0, 254]`.
   Three integer instructions, no branch; bit-identical to "take the exponent after `div.rn`, +1 if the mantissa is non-zero" over the whole domain where `amax/448` is an f32 normal
   (the rounding of `div.rn` never rounds the next representable number above `448·2^k` back to `2^k`: the nearest one is at `2^k·(1 + 2⁻²³/1.75)`, more than half an ulp away).
   The inverse scale is assembled from bits directly: the bit pattern of `2^(127−e)` is `(254 − e) << 23`.

With these two steps done, the whole CH-chunk epilogue is one basic block, and only then can ptxas place chunk k+1's `LDTM` among chunk k's arithmetic.
The "split ld / wait + values threaded through the wait" spelling below **pins** the hoist in the source (values and SASS order are certified), but it is **not a proven speed-up lever**:
in the certification probe, once the division was removed ptxas already hoisted 2 `LDTM`s on its own in the fused form, and the toy cycle counts were look-ahead 1031 vs fused 999 cyc/round (single INFO reading, look-ahead slower);
there is no whole-kernel reading either. Remove the division first and look at the SASS order; only when SASS shows the next chunk's read-back really was not hoisted is it a candidate, and same-job timing decides (gotcha 5).

## When needed (trigger)

- The per-chunk epilogue segment (read-back → scale → amax → scale byte → narrow-type packing → store) is on the serial handshake chain, and an ablation proves that "how long this segment takes" determines whole-kernel time while "who does it" does not change it (moving it to another warpgroup gave no gain).
- In the stall ledger this segment is dominated by `long_scoreboard` (read-back / write-back waits) and `wait` / `short_scoreboard` (dependency chain), with few dispatch / selection stalls — the shape of a single serial chain.
- SASS has a `MUFU.RCP` … `FCHK` … `BRA` group per chunk, or a `CALL.REL.NOINC` into a division subroutine; or SASS / the dossier shows several branch regions / loops without async operations in the epilogue.
- The read-back uses the fused spelling "`tcgen05.ld` + `tcgen05.wait::ld` in one asm", and you want the next chunk's read-back to land under this chunk's arithmetic.

## Minimal runnable example

UE8M0 byte and inverse scale (division-free; replaces the version with `div.rn.f32 y, amax, 448.0`):

```python
from cutlass import Int32, Float32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

@dsl_user_op
def ue8m0_scale(amax, *, loc=None, ip=None):
    """amax >= 0 -> (e, 2^(127-e)),  e = clamp(127 + ceil(log2(amax / 448)), 0, 254)"""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]), [Float32(amax).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 b;\n\t.reg .s32 e;\n\t"
        "mov.b32 b, $2;\n\t"
        "add.u32 b, b, 0x1FFFFF;\n\t"      # mantissa > 0x600000 (1.75) carries into the exponent
        "shr.u32 b, b, 23;\n\t"
        "sub.s32 e, b, 8;\n\t"              # 448 = 1.75 * 2^8
        "max.s32 e, e, 0;\n\tmin.s32 e, e, 254;\n\t"
        "mov.b32 $0, e;\n\t"
        "sub.s32 e, 254, e;\n\tshl.b32 e, e, 23;\n\t"   # bits of 2^(127-e)
        "mov.b32 $1, e;\n\t}",
        "=r,=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return (Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))
```

The read-back split into two asms, "issue the ld only" and "wait and thread the values out" (`n` values pass through the wait via the tied constraints `"0".."n-1"`):

```python
@dsl_user_op
def tmem_ld16_nowait(taddr, *, loc=None, ip=None):
    outs = ",".join("$%d" % i for i in range(16))
    res = llvm.inline_asm(llvm.StructType.get_literal([T.f32()] * 16), [Int32(taddr).ir_value(loc=loc, ip=ip)],
                          "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%s}, [$16];" % outs,
                          ",".join(["=f"] * 16) + ",r,~{memory}", has_side_effects=True,
                          is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(16)]

@dsl_user_op
def wait_ld_thru(vals, *, loc=None, ip=None):
    n = len(vals)
    res = llvm.inline_asm(llvm.StructType.get_literal([T.f32()] * n),
                          [Float32(v).ir_value(loc=loc, ip=ip) for v in vals],
                          "tcgen05.wait::ld.sync.aligned;",
                          ",".join(["=f"] * n) + "," + ",".join(str(i) for i in range(n)) + ",~{memory}",
                          has_side_effects=True, is_align_stack=False,
                          asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(n)]
```

Processing one chunk (a plain Python helper; loops use Python `range`) and the body of the one-chunk look-ahead (written inside the `@cute.kernel` body):

```python
def process(x, addr):                    # x: 32 x Float32, already waited on
    amax = tree_max([fabs_f(t) for t in x])          # abs.f32 + balanced max.f32 tree
    e8, isc = ue8m0_scale(amax)
    w = [pack_e4m3x4(x[4*k] * isc, x[4*k+1] * isc, x[4*k+2] * isc, x[4*k+3] * isc) for k in range(8)]
    stg256(addr, w)                                  # st.global.v8.b32, addr 32 B aligned
    return e8

sfw = Int32(0)
cur = wait_ld_thru(tmem_ld16_nowait(t0) + tmem_ld16_nowait(t0 + 16))
for ch in cutlass.range_constexpr(CH):
    nxt = []
    if cutlass.const_expr(ch + 1 < CH):              # chunk k+1 in flight ...
        nxt = tmem_ld16_nowait(t0 + 32 * (ch + 1)) + tmem_ld16_nowait(t0 + 32 * (ch + 1) + 16)
    sfw = sfw | (process(cur, row + Int64(32 * ch)) << (8 * ch))   # ... chunk k being computed
    if cutlass.const_expr(ch + 1 < CH):
        cur = wait_ld_thru(nxt)                      # only chunk k+1 is in flight: the wait waits for it alone
```

`pack_e4m3x4` (4 f32 → one word, `a` in the lowest byte):

```
cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;   // first source → high byte: pass (b, a) to get a in the low byte
cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
mov.b32 $0, {lo, hi};
```

Acceptance:
- Values: compare byte by byte against a numpy reference (RNE, satfinite, −0 preserved); compare each UE8M0 byte against the exact integer solution "smallest k such that 448·2^k ≥ amax"; boundary chunks must cover `amax = 448·2^k`, one ulp above and below it, all-zero chunks, and exact powers of 2.
- SASS (`CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>` → `nvdisasm -c`): in the division-free version `FCHK` and `MUFU.RCP` are 0 and the `CALL` count is lower than in the division version (the division slow-path `CALL` disappears; what remains is kernel scaffolding `CALL` identical in every variant — do not accept by "`CALL` = 0"); in the look-ahead version the `LDTM`s before the first `STG` are ≥ 2 chunks' worth.

## Gotchas

1. **`div.rn.f32` by a constant is not folded into a multiplication** (1/448 is not a power of 2; multiplying by the reciprocal does not guarantee correct rounding); it keeps the full division sequence and the slow-path branch. Replacing it with `div.approx` / `div.full` changes the rounding and can be one scale step off the exact definition; the bit-arithmetic form above is a **bit-equivalent** replacement.
2. **Domain of the integer form**: `amax/448` is an f32 normal (`amax ≥ 448 · 2⁻¹²⁶`) or `amax = 0`. `amax = 0` lands on the clamp's lower bound `e = 0`, inverse scale `2¹²⁷`, and an all-zero chunk's payload is still 0. `amax` must be an absolute value first (a carry from the sign bit would pollute the exponent); NaN inputs are outside the domain.
3. **`tcgen05.wait::ld` waits for all `tcgen05.ld` this thread issued before it**; there is no "wait up to the k-th" form. So the look-ahead must be "this chunk already waited → issue the next chunk → compute this chunk → wait": at the wait only the next chunk is in flight. Issuing the next chunk's ld **before** this chunk's wait means waiting for both together, and the look-ahead is lost.
4. **When ld and wait are in two asms, the values must be threaded through the wait** (outputs `=f`, inputs tied to `"0".."n−1"`). `has_side_effects` / `~{memory}` alone are not enough: LLVM only orders side-effecting asms among themselves, so pure arithmetic consumers (`abs` / `max` / `mul`) can be hoisted above the wait and read unfinished registers — silently wrong, and constant payloads will not show it.
5. **The fused spelling (ld + wait in one asm) can also compile to a cross-chunk `LDTM` hoist** — provided there is no basic-block boundary between the two chunks. So remove the division branch first, then look at the SASS order; where ptxas already hoisted, a source-level look-ahead is a no-op change (readings in the fact line of the source probe).
6. **Register budget**: the look-ahead raises the peak from "one chunk of 32 + this chunk's intermediates" to "two chunks of 64 + this chunk's intermediates". When the budget is tight (low `setmaxnreg` tier) see [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) first: you can hoist only the first `.x16` half of the next chunk and issue the second half after this chunk's amax is done.
7. **`st.global.v8.b32` (`STG.E.256`) requires a 32 B aligned address** and exists only on sm100; row stride and chunk offset must both be multiples of 32 B. Misalignment is a runtime misaligned address, not a compile error.
8. **Assemble the SF bytes into a word** with `sfw | (e << 8·k)`; `e` is already clamped to `[0, 254]` and does not cross the byte; compute `e` as `.s32` (`− 8` may be negative), and only treat it as unsigned after the clamp.

## Sources

- PTX ISA: `tcgen05.ld` / `tcgen05.wait::ld` (Tensor Memory section), `cvt` (`.e4m3x2` destination, `.satfinite`, source order), `div` (rounding semantics of `.rn` / `.approx` / `.full`), `st.global.v8.b32` — https://docs.nvidia.com/cuda/parallel-thread-execution/
- OCP Microscaling Formats (MX) v1.0 specification: E8M0 scale and E4M3 element formats — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Public API symbols: `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.range_constexpr`, `cutlass.const_expr`.
- Certification probe: `PROBE PASS max_diff=0.0` (round 2).
