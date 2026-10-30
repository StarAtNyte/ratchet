# A "one-batch" gather after a dependency wait: N independent gmem loads in one L2 round trip

> One file per hardware interface. Operator-agnostic; mechanism only.

> **Status**: ✅ Verified (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=9.46e-06`, certification round 3; all three contracts passed in the same round: the naive / batched / db variants, through the PDL producer→consumer path, match numpy with a max relative error of 9.46e-06; in the batched consumer's SASS all 33 `LDG` are before the first `SHFL.BFLY`/`FMNMX`, 0 after, 0 `LDL/STL`; naive/batched timing ratio 1.998).
>
> **Certified surface**: one shape — one row per warp, per chunk 1 per-lane 8 B metadata load + 32 4 B data loads with lane-broadcast addresses,
> 128-thread CTAs, 148 CTAs. The naive variant in the timing comparison is **two trips by construction** (the load's predicate comes from the reduction result,
> a true data dependency), so 1.998 is the price of "really two trips vs one trip", not the price of "wrong source order" (for the latter see gotcha 1).

## Purpose
After `griddepcontrol.wait` (PDL) or any dependency wait, a warp has to read a per-lane piece of metadata
(e.g. a pair of f32) **and** data gathered from 32 rows with lane-broadcast addresses, then do a warp reduction
(`SHFL.BFLY` / `FMNMX`) + weighted accumulation (`FFMA`). This entry covers how to get these 33 loads **issued as one batch**,
paying a single L2 round trip, instead of two serial trips "metadata one trip → reduction → data another trip". With more than 32 rows,
each chunk pays one trip. Note: both spellings of cross-chunk double buffering were slower than the single batch in the probe (see gotcha 10).

## When needed (trigger)
- Timestamps from the same job show **two serial round trips** after the wait: `wait.end → metadata arrives` is about one L2
  latency, then `metadata arrives → accumulation ends` is again about one L2 latency plus compute.
- In SASS there is still a run of `LDG` **after** the first `SHFL.BFLY` / `FMNMX` (the data-row loads appear after
  the reduction chain). This is the criterion; the source shape is only a suspicion (gotcha 1: ptxas does not schedule in source order).
- With more than 32 rows there is a `unroll=1` chunk loop, and every chunk repeats the two trips above.

## Minimal runnable example
```python
# Contains no operator algorithm: the "max / coefficient / accumulate" in consume() is only a placeholder consumer;
# replace it with your own. The mechanism only concerns the dependency between the loads and the consumer.
from cutlass import Float32, Int32, Int64
import cutlass, cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
ATT = llvm.AsmDialect.AD_ATT

@dsl_user_op
def ld_f32_if_nonneg(addr, key, *, loc=None, ip=None):
    """key >= 0 ? ld.global.cg.f32 [addr] : 0. The predicate is the sign bit of the broadcast value; no memory access is issued otherwise."""
    return Float32(llvm.inline_asm(
        T.f32(), [Int64(addr).ir_value(loc=loc, ip=ip), Int32(key).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ge.s32 p, $2, 0;\n\tmov.b32 $0, 0;\n"
        "\t@p ld.global.cg.f32 $0, [$1];\n}",
        "=f,l,r", has_side_effects=True, is_align_stack=False, asm_dialect=ATT,
        loc=loc, ip=ip))
# ld_v2f_if(addr, pred): same spelling, returns struct<(f32,f32)>, @p ld.global.cg.v2.f32 (see the probe).
# lt_i / sel_i / sel_f / fmaxf: inline asm of setp + selp / max.f32 (see the probe).

def dep_tok(tok, z, vals):
    """Dependency token: tok | ((xor of the bit patterns of vals) & z). z is a kernel parameter that is 0 at run time,
    so the value is always tok; but ptxas cannot prove that, so its users can only execute once all of vals have arrived."""
    n = len(vals)
    txt = "{\n\t.reg .b32 a, t;\n\tmov.b32 a, $3;\n"
    for k in range(1, n):
        txt += "\tmov.b32 t, $%d;\n\txor.b32 a, a, t;\n" % (3 + k)
    txt += "\tand.b32 a, a, $2;\n\tor.b32 $0, $1, a;\n}"
    return Int32(llvm.inline_asm(
        T.i32(), [Int32(tok).ir_value(), Int32(z).ir_value()] + [Float32(v).ir_value() for v in vals],
        txt, "=r,r,r," + ",".join(["f"] * n),
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT))

def or_tok(x, tok):                               # bit pattern of x OR tok (tok==0 -> x unchanged)
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(x).ir_value(), Int32(tok).ir_value()],
        "{\n\t.reg .b32 m;\n\tmov.b32 m, $1;\n\tor.b32 m, m, $2;\n\tmov.b32 $0, m;\n}",
        "=f,f,r", has_side_effects=False, is_align_stack=False, asm_dialect=ATT))

def fence(x, z, buf):                             # groups of 8 values, keeping the operand count small
    tok = Int32(0)
    for g in range(0, 34, 8):
        tok = dep_tok(tok, z, [buf[i] for i in range(g, min(g + 8, 34))])
    return or_tok(x, tok)

def prep(row, ch, lane, n):
    """Registers only, reads nothing the producer wrote -> may be placed before the wait."""
    p = ch * 32 + lane
    valid = lt_i(p, n)
    slot = slot_of(row, p)                       # your address function (arithmetic, or read before the wait)
    key = sel_i(valid, slot, -1)                 # predicate folded into the sign: only 1 shfl per row
    keys = [cute.arch.shuffle_sync(key, ii) for ii in range(32)]
    return valid, slot, keys

def issue(meta_a, x_a, lane, valid, slot, keys, buf):
    """33 loads into mutually independent registers; no instruction that reads a result sits between them."""
    ab = ld_v2f_if(meta_a + Int64(slot) * 8, valid)
    buf[0] = ab[0]; buf[1] = ab[1]
    for ii in range(32):
        buf[2 + ii] = ld_f32_if_nonneg(x_a + Int64(keys[ii] * 32 + lane) * 4, keys[ii])

def consume(buf, valid, acc, z):                 # placeholder consumer
    mx = fence(sel_f(valid, buf[0], -3.0e38), z, buf)   # the first consumer waits for all 34 values to arrive
    for o in range(5):
        mx = fmaxf(mx, cute.arch.shuffle_sync_bfly(mx, 1 << o))
    coef = sel_f(valid, (buf[0] - mx) * buf[1] + Float32(1.0), Float32(0.0))
    s = acc[0]
    for ii in range(32):
        s = s + cute.arch.shuffle_sync(coef, ii) * buf[2 + ii]
    acc[0] = s

@cute.kernel
def k(meta_a: Int64, x_a: Int64, out_a: Int64, n: Int32, Z: Int32):   # host passes Z=0
    tidx, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
    lane = tidx & 31; row = bid * 4 + (tidx >> 5)
    nch = (n + 31) >> 5
    acc = cute.make_rmem_tensor((1,), Float32); acc[0] = Float32(0.0)
    buf = cute.make_rmem_tensor((34,), Float32)          # one batch of 34 values, constant indices -> registers
    cute.arch.griddepcontrol_wait()                      # (1) dependency wait, once only
    for ch in cutlass.range(nch, unroll=1):
        valid, slot, keys = prep(row, ch, lane, n)       # (2) compute the 32 broadcasts into registers first
        issue(meta_a, x_a, lane, valid, slot, keys, buf) # (3) 33 loads go out as one batch
        consume(buf, valid, acc, Z)                      # (4) BFLY/FMNMX/FFMA only after the fence
    # store acc ...
# The producer kernel calls cute.arch.griddepcontrol_launch_dependents() at its start;
# the consumer .launch(..., use_pdl=True).
```
Each chunk is one round trip plus compute instead of two plus compute. With only one chunk (≤ 32 rows),
`prep` can be moved before the wait, and the 33 loads follow right after the wait (this shape was not tested separately by the probe).

## Gotchas
1. **Source order ≠ issue order, and things move in both directions.** Inline asm with `has_side_effects=True` has no boundaries once it reaches PTX;
   ptxas schedules by its own rules. Measured on GPU (certification round 1, a round without the token yet):
   (a) in the double-buffered variant, the next chunk's 33 `LDG` placed in source **before** this chunk's consumer were all scheduled **after** the first
   `SHFL.BFLY` (before 33 / after 33); (b) the single-batch form without a token failed the contract "0 LDG after the first BFLY";
   (c) conversely, a group of loads written in source **after** the reduction but not depending on its result did not all stay after the reduction either
   (the contract "≥ 32 LDG after the first BFLY" likewise FAILED). For (b)(c) only the verdicts exist; the exact counts were not kept in the returned log tail.
   What reliably constrains ptxas is a **data dependency**.
2. **Pin the first consumer after all loads with a dependency token** (`fence` above): xor the bit patterns of the 34 results,
   AND with a **kernel parameter that is 0 at run time**, then OR into the first consumer's input. A constant 0 does not work,
   nor does `x*0`: a constant is folded and the dependency vanishes with it; a floating-point `x*0` also pollutes the result on inf/NaN.
   The instruction cost of the token itself was not priced separately by the probe. Since the 33 loads are issued together, "the first consumer waits for all to arrive"
   only waits extra for the latest load in this batch, not for another round trip.
3. **Compute the address / predicate broadcasts into registers before issuing the loads.** The probe's batch form first computes the 32 broadcast values into registers
   (`keys`, unrolled at trace time), then issues the 33 loads; how ptxas schedules when the broadcasts are interleaved with the bfly chain was not tested separately.
4. **Do not let a load's predicate or address depend on the reduction result** (e.g. skipping rows by "whether the weight is 0"): that makes the load
   depend on the bfly chain through data, and it is always two trips. The probe's naive variant is written this way: 1 `LDG` before the first BFLY in SASS,
   32 after, and its timing is 1.998 times the batch form.
5. **Fold the predicate into the sign bit of the broadcast value** (`key = valid ? slot : -1`, and `setp.ge.s32` in the load),
   saving one `shfl(valid, ii)` per row, i.e. 32 SHFL for 32 rows. The address computed for an invalid row is out of bounds, but with the predicate off,
   `@p ld` does not access memory (the probe's last chunk has such rows, and the numerical contract passes).
6. **Use predicated `@p ld` for loads, not `if`.** In the asm, first `mov.b32 $0, 0`, then `@p ld`; rows turned off read as 0.
   What happens when every load is wrapped in a dynamic `if` was not tested separately by the probe — only this predicated spelling is certified.
7. **The loads' asm has no `"memory"` clobber, and no `fence` / `bar` is inserted between loads** — that is how the probe's batch form is written;
   what happens if they are added was not tested separately.
8. **Values carried across dynamic `cutlass.range` iterations must be `cute.make_rmem_tensor`**, accessed only with constant
   indices (they stay in registers). A Python list cannot be carried across dynamic iterations. Also, if a name assigned in a dynamic loop body
   appeared **anywhere earlier** in the same function (including in another `const_expr` branch), the DSL
   treats it as a loop-carried variable and then reports `` `v1` is None prior to this `for`, and update to Int32 inside of this `for` is not supported ``
   (one variant in certification round 2 failed to compile exactly this way). Give local names in the loop body unique names.
9. **Out-of-range lanes in the last chunk switch off naturally**: when `p ≥ n` the predicate is 0 and the key is -1; no dynamic `if` is needed.
   This assumes the slot/address function can also be computed for out-of-range p (without reading a table); if the slot comes from a gmem table, reads of that table
   must be predicated too.
10. **Both spellings of cross-chunk double buffering were slower than the single batch in the probe (measured on GPU, same job).**
    Copy style (db): two 34-element rmem tensors cur/nxt, the next chunk's 32 keys carried one beat ahead in an rmem tensor,
    loop body `issue(nxt) → consume(cur) → prep(c+2) → cur[i]=nxt[i]`. On a 128-thread CTA it measured 41 `LDL/STL`,
    14 `LDG` after the first `SHFL.BFLY`, and 2.52 times the single batch's time (420.4 vs 166.5 µs).
    Ping-pong style (db2, two buffers taking turns, two chunks per trip, no copy): `LDL/STL` is 0, but there are 66 `LDG` after the first `SHFL.BFLY`
    — 33 of them are after it in source order anyway, the other 33 are next-chunk loads placed before the consumer in source but outside the token dependency,
    which ptxas moved later — and 1.13 times the single batch's time (188.6 vs 166.5 µs).
    So use the single batch by default; to double-buffer, check the SASS first: 0 `LDL/STL`, this chunk's LDG not after the BFLY,
    and same-job timing really faster than the single batch — use it only if all three hold.
11. **Place the wait before the first load that reads producer data, and only once.** Any work that does not read the producer's output
    (row-index scans, address broadcasts) can be placed before the wait.
12. **Accept on SASS, not on source**: in the consumer kernel's linear SASS, **the number of `LDG` after the first `SHFL.BFLY` /
    `FMNMX` should be 0**. With a single batch there are at least 33 before it, and `LDL/STL` is 0. How to get the cubin:
    `CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=<dir>` (the probe gives each variant its own `CUTE_DSL_CACHE_DIR`),
    then `cuobjdump -sass <cubin>`, split by `Function :` sections, and count only the consumer's section.
13. **Timing must be device-bound**: in certification round 1 each row had only 16 chunks, and back-to-back timing of all three variants landed at 23–25 µs per launch
    (naive 23.2 / batched 24.1 / db 25.0) — that measured host-side launch overhead, and the timing contract measured nothing.
    Round 3 raised the timed launch to 160 chunks per row, and only then did the readings separate (332.6 / 166.5 µs).
    When comparing variants, make the work per launch large (CUDA graphs are another route; the probe did not use them).

## Sources
- PTX ISA: `griddepcontrol`, the `.cg` modifier of `ld` and predicated execution (`@p`), `shfl.sync`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Programming Guide, Programmatic Dependent Launch:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
- CuTe DSL API symbols: `cute.arch.griddepcontrol_wait`, `cute.arch.griddepcontrol_launch_dependents`,
  `.launch(..., use_pdl=True)`, `cute.arch.shuffle_sync`, `cute.arch.shuffle_sync_bfly`,
  `cute.make_rmem_tensor`, `cutlass.range(..., unroll=1)`, `cutlass.range_constexpr`,
  `llvm.inline_asm` (`dsl_user_op`).
- Certification probe:
  variants naive / batched / db / db2 compute the same quantity. Contracts: numerics, batched SASS (all LDG before the first BFLY/FMNMX, 0 LDL/STL),
  naive/batched timing ratio ≥ 1.15. Layout and timing of db / db2 are INFO only. Verdict: `PROBE PASS max_diff=9.46e-06`;
  same-job readings naive 332.6 µs / batched 166.5 µs / db 420.4 µs (41 LDL/STL) / db2 188.6 µs (0 LDL/STL, 66 LDG after BFLY).
  The counterexample readings of gotcha 1 come from round 1 of the same certification (only the numerical one of 6 contracts passed).
