# branchfree_masked_elementwise_pass (a branch-free masked element-wise segment over a TMEM accumulator fragment: select mask + one vector `LDS` for row/column scalars + the accumulator-independent factor chain hoisted before the wait)

> **Certification source: the sm100 tree** -- **Status: ✅ GPU-certified** -- `PROBE PASS max_diff=3.8657465348084646e-07`
> (certified by a GPU probe that is not included in this release, round 3 of this certification; the six contracts
> `frag_roundtrip` / `numeric` / `sel_census` / `lds_batched` / `sel_time` / `hoist_time` all
> `PROBE PASS` in the same round, readings in Sources).
>
> **Variant section "[row-lane × col] fragment"** (the `Ld32x32b` / `St32x32b` shape: one whole row per thread, column scalars only,
> field-major face + one `LDS.128` per 4 columns + one `.f32x2` per adjacent column pair + in-place write-back) -- status: **✅ GPU-certified** --
> `PROBE PASS max_diff=1.020912233514069e-06` (a second probe,
> round 1 of this certification; the six contracts `frag_roundtrip` / `rowlane_struct` / `numeric` / `lds_vec` / `packed_census` /
> `colvec_time` all `PROBE PASS` in the same round, readings in Sources).
>
> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry follows [`branch_free_select_mask`](branch_free_select_mask.md) (a mask is a predicate + `FSEL`, not arithmetic;
> sm90 readings) and [`threshold_bitmap_predicate_select`](threshold_bitmap_predicate_select.md) (the bitmap form of a threshold mask),
> and deals with **another kind of segment** on sm100: not a row reduction (the softmax kind), but a **purely element-wise transformation** on the accumulator
> fragment read back into registers by `tcgen05.ld` -- each element is multiplied by a factor determined by "the scalar of the element's row" and "the scalar of its column",
> then kept or zeroed by a (row, col) mask, and handed downstream (conversion to 16-bit, `stmatrix`, write-back).
> Such a segment has no reduction chain to interleave; its time is decided by three things: the per-element control flow, the number of `LDS` per element,
> and **which work is scheduled after the blocking wait**. This entry gives one deterministic spelling for each of the three.

## Purpose

An element-wise pass of the form

```
y[e] = mask(row(e), col(e)) ? φ(rowscalar[row(e)], colscalar[col(e)]) · x[e] : 0      e = 0 .. E−1
```

(`x` is the accumulator fragment just read back from TMEM, E elements per thread, with each element's (row, col)
given by the copy atom's identity partition) should, in hardware, be **one stretch of straight-line code**: one `FSEL` per element,
one vector `LDS` per row group / per 4 columns, and the scalar-only chain `φ` (`FADD`/`MUFU.EX2`/`FMUL`)
already issued **before the `mbarrier` wait**. The three things are independent, can be accepted separately, and together form one segment.

## When needed (trigger)

In the segment ledger / SASS census all of the following appear:

1. **A run-time `if` per element**: in SASS the count of `BSSY`/`BRA`/`BSYNC` ≈ E (or `ISETP` ≈ E with predicated jumps),
   and the segment's main stall items are `no_instructions` (the signature of branch / fetch starvation) + `short_scoreboard`;
2. **`LDS` ≈ 3·E**: per element one read each of the row scalar and the column scalar (the same row's scalar read again and again by the same thread);
3. the segment comes right after an `mbarrier` wait, while **more than half the instructions** in the segment (scalar reads, `FADD`, `MUFU.EX2`, `FMUL`)
   do not depend on the waited data -- they could have executed during the wait;
4. ablation: merely changing the `if` to a select with the instruction count unchanged drops the segment time by double-digit percent -- the defining shape of "the structure is right, the shape inside the segment is wrong".

All four together = this entry. When only item 1 appears and the valid set is a contiguous threshold stretch, also see
[`threshold_bitmap_predicate_select`](threshold_bitmap_predicate_select.md) to eliminate the `ISETP` as well.

## Mechanism: the instruction accounting of the three levers (per trip, per thread; E = 32, columns in groups of 4, two row scalars stored as a pair)

| Lever | Before | After | Saved |
|---|---|---|---|
| (1) mask | per element `ISETP` + `BSSY` + `BRA` + `BSYNC`; the `LDS`/`MUFU` inside the branch body are separated by basic blocks and cannot interleave with neighbouring elements | per element `ISETP` + `FSEL` (predicated straight-line code) | 2–3 instructions per element + one pipeline drain per element |
| (2) scalars | 3 scalar `LDS` per element → 96 | one `LDS.64` per row group (two row scalars as a pair) + one `LDS.128` per 4 columns → ≈ 4 + 8 | ≈ 84 MIO instructions |
| (3) order | wait → `LDTM` → scalars → factor chain → multiply → mask | scalars → factor chain → **wait** → `LDTM` → multiply → mask | the whole latency of the factor chain hidden inside the wait |

Three key points:

- **The mask is a select on the product, not a park on the exponent.** `y = (col < row) ? f·x : 0` is one `selp.f32`;
  a masked element is always exactly 0, regardless of whether `x` is finite. Folding the mask into the exponent argument (parking at −1e30 so that `exp2` underflows to 0,
  then multiplying by `x`) is also branch-free, but `0 · x` is NaN when `x` is not finite -- the product select has no such precondition.
  Neither is "convert 0/1 to float and multiply" (that is the `I2F` form, `branch_free_select_mask` gotcha 1).
- **Read the scalars into registers once, following the fragment's structure.** The copy atom's identity partition
  `thr.partition_D(make_identity_tensor(tile))` tells you which elements share a row (row groups) and which elements have contiguous
  columns (column runs). This table is a static property of atom + tile + thread, not of the data -- read it once (dump each
  element's (row, col) on the GPU, or print it at trace time), write it as a trace-time constant table, then one `ld.shared.v2` per row group and
  one `ld.shared.v4` per run. The smem layout must cooperate: column scalars contiguous by column (run start 16 B aligned);
  the two row scalars used together interleaved as a pair `(s, a)`, one 64-bit read fetching both.
- **Do everything that can be done before the wait.** ptxas does not move instructions across an `mbarrier` wait loop (that is a loop of side-effecting asm
  and a basic-block boundary too), so "which work executes in the shadow of the wait" is **decided entirely by source order** --
  one of the few places in this tree where source order really lands (contrast the conclusion in `ew_segment_latency_schedule` that "batch-hoisting within a segment
  does not work": that is reordering within one basic block, which belongs to ptxas; ordering across a wait belongs to you). The factor chain that depends only on already-published data
  (smem scalars protected by an **earlier** barrier) is hoisted as a whole; after the wait only `LDTM`, one `FMUL` per element,
  one `FSEL` and the downstream conversion remain. The hoisted values live across the wait; the register account counts E factors + the scalars.

## Minimal runnable example

### (a) Three primitives (PTX escape hatch; select is a pure function, vector reads **must** carry the side-effect mark)

```python
import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

ATT = llvm.AsmDialect.AD_ATT


@dsl_user_op
def sel_lt_s32(xi, yi, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """xi < yi ? a : b   (Int32 comparison, fp32 select) -> ISETP + FSEL, no region."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(xi).ir_value(loc=loc, ip=ip), cutlass.Int32(yi).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))


def lds_v4(addr):
    """One 128-bit shared read (addr: Int32 shared-window address, 16 B aligned) -> 4 Float32.
    No decorator, inlined at trace time; the struct type is parsed at trace time (needs a live MLIR context).
    has_side_effects=True + ~{memory}: otherwise LLVM may hoist it above the barrier that publishes these bytes."""
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [cutlass.Int32(addr).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT)
    return [cutlass.Float32(llvm.extractvalue(T.f32(), r, [k])) for k in range(4)]


def lds_v2(addr):
    """One 64-bit shared read (8 B aligned) -> 2 Float32."""
    r = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [cutlass.Int32(addr).ir_value()],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT)
    return [cutlass.Float32(llvm.extractvalue(T.f32(), r, [k])) for k in range(2)]
```

For the operand numbering of struct multi-result asm (results take `$0..` first, the first input follows) see
[`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md); smem addresses use
`tensor.iterator.toint()` (an Int32 shared-window address, same as [`cp_async_bulk_1d_mbarrier_gather`](cp_async_bulk_1d_mbarrier_gather.md)).

### (b) Read the fragment's structure from the identity partition (once; the structure is a property of the atom, not of the data)

```python
# the same tiled copy and the same thread slice as the real accumulator readback:
tiled_t2r = tcgen05.make_tmem_copy(
    cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32), plane)
thr = tiled_t2r.get_slice(tidx)
cAcc = thr.partition_D(idAcc)        # idAcc: the identity tensor of that plane (through the MMA's C partition, then flat_divide to the tile)
rAcc = cute.make_rmem_tensor(cAcc.shape, cutlass.Float32)
E = cute.size(rAcc)
# dump it once (or print at trace time) and compute on the host:
#   row_groups: list of sets of elements sharing the same row          -> one lds_v2 per group
#   runs:       quadruples of elements with columns c0..c0+3 (c0 % 4 == 0) -> one lds_v4 per group, sorted by ascending column
for e in cutlass.range_constexpr(E):
    gCoord[(tidx, e, 0)] = cAcc[e][0]
    gCoord[(tidx, e, 1)] = cAcc[e][1]
```

On the host, judge runs by "grouped by (row, col // 4), exactly 4 per group, column numbers exactly 4q..4q+3, the same element set on every thread";
if not satisfied, fall back to `v2` (by col // 2) or scalars -- **do not assume "adjacent e means adjacent columns"**; the V-mode order differs between atoms.
The certification probe hands this table to the pass as a trace-time constant; the numeric contract also validates the table (a wrong table makes the result miss the reference).

### (c) Segment shape: scalars and factor chain before the wait; only multiply, select and conversion after it

```python
row_base = cutlass.Int32(sRow.iterator.toint())     # sRow: (rows, 2) interleaved (s, a), 16 B aligned
col_base = cutlass.Int32(sCol.iterator.toint())     # sCol: (cols,) contiguous by column, 16 B aligned


def preload(cAcc):                                  # ordinary Python helper, unrolled at trace time
    s_row, a_row, s_col = [None] * E, [None] * E, [None] * E
    for grp in ROW_GROUPS:                          # the constant table read out in (b)
        v = lds_v2(row_base + cAcc[grp[0]][0] * 8)
        for e in grp:
            s_row[e], a_row[e] = v[0], v[1]
    for run in RUNS:                                # quadruples, ascending by column
        v = lds_v4(col_base + cAcc[run[0]][1] * 4)
        for k, e in enumerate(run):
            s_col[e] = v[k]
    return s_row, a_row, s_col


def chain(b, s_row, a_row, s_col):                 # depends only on published scalars: may go before the wait
    return [a_row[e] * cute.math.exp2((s_row[e] + b) - s_col[e], fastmath=True)
            for e in range(E)]


for it in cutlass.range(n):
    b = gScalar[it]
    s_row, a_row, s_col = preload(cAcc)             # smem protected by an earlier barrier
    f = chain(b, s_row, a_row, s_col)               # E of FADD/FADD/MUFU/FMUL -- before the wait
    cute.arch.mbarrier_wait(mbar + BAR_ACC_FULL, it % 2)
    cute.copy(tiled_t2r, tSrc, rAcc)                # LDTM
    cute.arch.fence_view_async_tmem_load()
    for e in cutlass.range_constexpr(E):
        y = sel_lt_s32(cAcc[e][1], cAcc[e][0], f[e] * rAcc[e], cutlass.Float32(0.0))  # FMUL + FSEL
        ...                                         # downstream: convert to 16-bit / stmatrix / accumulate
    cute.arch.mbarrier_arrive(mbar + BAR_ACC_EMPTY)
```

The form being replaced (control, do not write it): per element `rY[e] = 0.0; if col < row: rY[e] = sRow[(row, 1)] * exp2(sRow[(row, 0)] + b - sCol[col]) * rAcc[e]`,
with the whole segment after `mbarrier_wait` -- this is where the four items of "When needed" come from.

### (d) Acceptance: SASS census + two time ratios (u = trip-loop unroll factor, taken as `MUFU.EX2 / E`)

1. `FSEL ≥ E·u`; `MUFU.EX2 = E·u` (the exponent work itself is unchanged; what moves is the shape around it);
2. `LDS ≤ E·u` (≈ 3E·u before the change), and `LDS.128` / `LDS.64` appear (the vector reads really landed);
3. `LDL == STL == 0`; the counts of `BRA` and `BSSY` are **independent of E** (coming only from role branches, the trip loop and wait loops), with the criterion `BRA < E`: one `if` per element adds at least E more `BRA` (or E pairs of `BSSY`/`BSYNC`), while handshake/wait loops add only a dozen or so (the same shape with a handshake: `BRA` 13 → 25); do not use one round's reading plus a fixed margin as the cap;
4. Numerics: masked elements **exactly 0**; unmasked elements within ~1e-6 relative error of the float64 reference (`ex2.approx`),
   not bit-identical to the `if` form (`FFMA` contraction differs by one rounding);
5. Time: with no handshake, 1 CTA/SM and **3 warps per scheduler** (3 consumer warpgroups reading the same TMEM plane),
   `t(if form) / t(select form) ≥ 1.03`;
   with a handshake (the producer arriving late) and 1 warp per scheduler, `t(chain after the wait) / t(chain before the wait) ≥ 1.03` --
   the second ratio measures "how much can hide in the shadow of the wait" and depends on the length of the producer's delay; certified readings in Sources.
   The two ratios are deliberately measured at **different occupancy scales**: levers (1)(2) save issue / MIO slots, lever (3) saves latency,
   see gotcha 12.

## Gotchas

1. **A run-time `if` is not a spelling of select.** The DSL's `if` is an scf region, and whether it lowers to `FSEL` is up to the optimizer;
   even when it does, the `LDS`/`MUFU` inside the branch body are separated by basic blocks and cannot interleave with neighbouring elements. For determinism use §(a).
2. **Converting 0/1 to float and multiplying = the `I2F` form** (`branch_free_select_mask` gotcha 1). The valid bit can only be used as a predicate.
3. **Select the product; do not park the exponent.** The park form relies on `exp2` underflow + `0 · x`; it is NaN when `x` is not finite.
   The masked result of the product select is 0 unconditionally. Both have the same `FSEL` count, so there is no reason to choose the former.
4. **Vector-read asm must have `has_side_effects=True` + `~{memory}`.** For asm marked as a pure function, its operands
   (smem base + constant offset) are available right at kernel start, and LLVM may hoist it above `sync_threads` /
   `mbarrier_wait` -- reading the bytes from before publication, wrong only on the first trip, silently.
5. **Alignment.** `ld.shared.v4` needs 16 B: run start `col % 4 == 0` and the smem base 16 B aligned; `v2` needs 8 B.
   Misalignment is an `illegal address` or wrong data, not slowness.
6. **Read the run table from the identity partition; do not guess.** `Ld32x32b` and `Ld16x256b` have different V-mode orders,
   and x8 / x16 repetitions differ too; the same atom with a different tile may also change. The dump + host judgement of §(b) is one-off,
   and the numeric contract is the safety net (wrong table → wrong column scalars used → misses the reference).
7. **Store row scalars interleaved.** Fields used together are interleaved as a pair `(s, a)`, one `LDS.64` fetching both; storing them as separate planes means
   two reads per row group -- still acceptable; what must be avoided is scalar reads **per element**.
8. **Hoist only work "protected by an earlier barrier".** The smem a hoisted chain reads must already have been published by an earlier wait;
   reading data that is only published by this wait is a race. Source order does land here (ptxas does not reorder across a wait loop),
   so putting it in the wrong place **will not be corrected**.
9. **Hoisted values live across the wait.** E factors + scalars + coordinates ≈ 1.5E registers span the `mbarrier_wait`;
   budget it by the role's `setmaxnreg` tier ([`warp_specialization`](warp_specialization.md)); at the budget ceiling fall back to
   the half-hoist "scalars before the wait, factor chain after the wait".
10. **Put temporary rmem tensors outside the staged loop.** A `make_rmem_tensor` inside the loop body is an alloca inside the loop body
    and is not guaranteed to be promoted out of local memory (`LDL`/`STL` show up, see `warp_specialization` gotcha 7).
11. **When two fragments share the same coordinate table and the same set of scalars, compute the factors once.** Two successive accumulator readbacks (the same
    tiled copy, the same slice) have identical element coordinates one by one, so `f[e]` holds for both; the second segment is left with only multiply, select and conversion.
12. **Levers (1)(2) and lever (3) pay off at different scales; do not measure them with one ruler.** When each scheduler has **only one warp**,
    this segment is latency-bound (the `LDTM` → `FMUL` → `FSEL` chain), and ptxas predicates a small, speculable
    `if` body by itself -- then the time difference between the `if` form and the select form is **< 2%** (certification probe round 1 reading
    1.017, `if` form SASS `BSSY = 0`): the `ISETP`/`LDS` saved are free when nobody competes for issue slots.
    Levers (1)(2) save **issue slots and MIO slots** and pay off only when the scheduler is shared -- in a warp-specialized kernel
    the warp holding the element segment shares schedulers with other roles' warps on the same SM, exactly this scale; the first ratio of §(d)-5
    is therefore measured at 3 warps per scheduler. Lever (3) (hoisting across the wait) saves **latency**, most directly exposed at 1 warp per scheduler,
    and covered by other warps at high occupancy. Also: **whether** the `if` form gets predicated is decided by ptxas's size threshold
    (a few more address computations in the body, a 16-bit conversion, or a store not promoted into registers can flip it into E real branches,
    `BSSY`/`BRA` ≈ E, main stall item `no_instructions`) -- the select form does not depend on that judgement, and census §(d)-3 is
    the way to check whether it flipped.

## Variant: [row-lane × col] fragment (`Ld32x32b` / `St32x32b`: one whole row per thread, column scalars only)

The three levers above were certified on the `Ld16x256b` shape: each thread's E elements split into several **row groups** (sharing row scalars)
and several **column runs**. TMEM has another readback shape -- the `.32x32b` family of `tcgen05.ld`: **each lane corresponds to one row**,
one instruction brings back 32 consecutive columns of that row, the `x32` repetition is 32 columns of one whole row per thread, and a `(128, 64)` plane is
64 elements per thread, all in the same row. An element-wise segment on such a fragment has only one kind of scalar: **column scalars** (every element's column
is different). Of the three levers, (1) is unchanged, (3) is unchanged, and (2) degenerates into a simpler and cheaper shape, plus two things specific to this shape.

### Instruction accounting (per trip, per thread; E = 64 columns, F column-scalar fields)

| | Before | After |
|---|---|---|
| column-scalar reads | one scalar `LDS` per element per field: `F·E` (128 when F = 2) | one `LDS.128` per field per 4 columns: `F·E/4` (32) |
| column scalars that must first be multiplied by an in-trip constant (`scale·g[c]` and the like) | one `FMUL` per element (E) | **premultiplied at publication**: the publisher does K `FMUL` per trip, written into a separate face; 0 on the consumer side |
| element-wise multiply / fma | one `FMUL` / `FFMA` per element (E) | one `FMUL2` / `FFMA2` per adjacent column pair (E/2) |
| order | wait → `LDTM` → read scalars → multiply → write back | read scalars → factor chain → **wait** → `LDTM` → multiply → write back |

### (a) Column scalars in smem must be a **field-major face**, not token-major records

With F scalar fields per column, there are two ways to store them:

```
token-major records  s[(col, field)]   stride (F, 1)  -- adjacent columns of the same field are 4F B apart
field-major face     s[(field, col)]   stride (K, 1)  -- one row of K fp32 per field, contiguous, 16 B aligned
```

A vector read needs **the 4 adjacent columns of the same field to fall in one 16 B word**, and only the field-major face satisfies this. Token-major records
**structurally** cannot when F is odd (5, 3, …): adjacent columns are 4F B apart, neither contiguous nor more than 4 B aligned --
LLVM's load merging (`LoadStoreVectorizer`) cannot save it either; it merges only when it can prove alignment. When F is even it can assemble
`LDS.64`, but still one per 2 columns, and what it gets is two fields of the same column rather than two columns of the same field.

Token-major records are often chosen for the **write side** (the publishing warp writes the F fields of one token per lane, avoiding bank conflicts).
The two sides have different requirements, so the answer is **to publish the same data twice**: the write side keeps its records and additionally publishes F field-major faces
(each face K × 4 B, 16 B aligned), within the same publish / reclaim phase (the same pair of `full` / `empty` mbarriers);
the consumer side reads only the faces. With K = 64, F = 3 this is 768 B of smem, traded for `F·E − F·E/4` MIO instructions per thread per trip.

**Premultiply at publication too**: if a field appears on the consumer side only in the form `constant × field` (the constant invariant within a trip),
let the publisher write the product into the face as a field -- K `FMUL` done once by the publishing warp instead of E by every consumer thread.

### (b) Column runs and adjacent column pairs are still read from the identity partition; do not assume "e is the column number"

```python
tiled_t2r = tcgen05.make_tmem_copy(
    cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), cutlass.Float32), plane)
tiled_r2t = tcgen05.make_tmem_copy(
    cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32), tcgen05.Unpack.NONE),
                        cutlass.Float32), plane)
thr_t2r, thr_r2t = tiled_t2r.get_slice(tidx), tiled_r2t.get_slice(tidx)
cAcc = thr_t2r.partition_D(idAcc)      # (row, col) of each element of the readback fragment
cSt = thr_r2t.partition_S(idAcc)       # (row, col) of each slot of the write-back fragment
```

Dump the two coordinate tables once (the §(b) approach), and judge three things on the host: **exactly one row** per thread; column runs = quadruples within a thread
with column numbers `4q..4q+3` (ascending by column); adjacent column pairs = `(0,1)`, `(2,3)` within a run. Plus one more table:
**the write-back permutation** `perm[es] = e`, such that `cSt[es]` and `cAcc[e]` have the same coordinates -- `Ld32x32b` and `St32x32b` are mirror
atoms and it is usually the identity permutation, but **read it out before using it** (the certification probe's `rowlane_struct` contract judges these three things together).

### (c) One packed instruction per adjacent column pair: both operands are register pairs, and the scalar pair comes straight from the vector read

`packed_f32x2_chain_single_asm` teaches "a chain + broadcast constants built on the fly inside the block". Here each step has only one packed
instruction, and **both operands are pairs**: `(f[c], f[c+1])` and `(x[e], x[e+1])` -- the scalar pair is just fields 0/1 (or 2/3) brought back by `ld.shared.v4`,
no constants to build. The in-trip scalar (`b`) is broadcast with `mov.b64 pb, {$4, $4}`.

```python
_FMA2 = ("{\n\t.reg .b64 pa, pb, pc, t;\n\t"
         "mov.b64 pa, {$2, $3};\n\t"          # (a[c], a[c+1])   two fields from the v4
         "mov.b64 pb, {$4, $4};\n\t"          # in-trip scalar broadcast
         "mov.b64 pc, {$5, $6};\n\t"          # (c[c], c[c+1])
         "fma.rn.ftz.f32x2 t, pa, pb, pc;\n\t"
         "mov.b64 {$0, $1}, t;\n}")
_MUL2 = ("{\n\t.reg .b64 pa, pb, t;\n\t"
         "mov.b64 pa, {$2, $3};\n\tmov.b64 pb, {$4, $5};\n\t"
         "mul.rn.ftz.f32x2 t, pa, pb;\n\t"
         "mov.b64 {$0, $1}, t;\n}")


def fma2_pair_bcast(a0, a1, b, c0, c1):        # -> (a0*b + c0, a1*b + c1)
    r = llvm.inline_asm(ir.Type.parse("!llvm.struct<(f32, f32)>"),
                        [F32(v).ir_value() for v in (a0, a1, b, c0, c1)],
                        _FMA2, "=f,=f,f,f,f,f,f",
                        has_side_effects=False, is_align_stack=False, asm_dialect=ATT)
    return F32(llvm.extractvalue(T.f32(), r, [0])), F32(llvm.extractvalue(T.f32(), r, [1]))


def mul2_pair(a0, a1, b0, b1):                 # -> (a0*b0, a1*b1)
    r = llvm.inline_asm(ir.Type.parse("!llvm.struct<(f32, f32)>"),
                        [F32(v).ir_value() for v in (a0, a1, b0, b1)],
                        _MUL2, "=f,=f,f,f,f,f",
                        has_side_effects=False, is_align_stack=False, asm_dialect=ATT)
    return F32(llvm.extractvalue(T.f32(), r, [0])), F32(llvm.extractvalue(T.f32(), r, [1]))
```

`$0` is the low half (column `c`), `$1` the high half (column `c+1`); the order of `mov.b64 {lo, hi}` matches the v4 field order,
so after "runs ascending by column + pairs taken as `(0,1)`, `(2,3)`" no swapping is needed.

### (d) Segment shape: face reads + factor chain before the wait; after the wait only `LDTM`, one `FMUL2` per pair, write-back

```python
face_a = I32(sFace.iterator.toint()) + slot * (F * K * 4)      # the face of field 0 (this slot)
face_c = face_a + K * 4                                         # the face of field 1


def preload(cAcc):                       # one LDS.128 per run per field
    s_a, s_c = [None] * E, [None] * E
    for run in RUNS:
        off = cAcc[run[0]][1] * 4
        va, vc = lds_v4(face_a + off), lds_v4(face_c + off)
        for k, e in enumerate(run):
            s_a[e], s_c[e] = va[k], vc[k]
    return s_a, s_c


def chain(b, s_a, s_c):                  # one FFMA2 per pair; depends only on published faces
    f = [None] * E
    for (e0, e1) in PAIRS:
        f[e0], f[e1] = fma2_pair_bcast(s_a[e0], s_a[e1], b, s_c[e0], s_c[e1])
    return f


for it in cutlass.range(n):
    s_a, s_c = preload(cAcc)                                    # protected by an earlier barrier
    f = chain(gScalar[it], s_a, s_c)                            # -- before the wait
    cute.arch.mbarrier_wait(mbar + BAR_ACC_FULL, it % 2)
    cute.copy(tiled_t2r, tSrc, rAcc)                            # LDTM
    cute.arch.fence_view_async_tmem_load()
    for (e0, e1) in PAIRS:
        rAcc[e0], rAcc[e1] = mul2_pair(f[e0], f[e1], rAcc[e0], rAcc[e1])   # FMUL2
    for es, e in enumerate(PERM):
        rSt[es] = rAcc[e]                                       # register renaming, zero instructions
    cute.copy(tiled_r2t, rSt, tDst)                             # STTM: in-place write-back
    cute.arch.fence_view_async_tmem_store()
    cute.arch.mbarrier_arrive(mbar + BAR_ACC_EMPTY)
```

The form being replaced (control, do not write it): `rAcc[e] = rAcc[e] * (sTok[(slot, col, 0)] * b + sTok[(slot, col, 1)])`
-- 2 scalar `LDS` + `FFMA` + `FMUL` per element, all after the wait.

**Register account**: what gets hoisted is `F·E` scalars (or E factors once the chain is computed) + E accumulator elements. With E = 64,
"compute the whole chain, then wait" is 64 factors + 64 elements ≈ 150 registers; when the budget cannot hold it, split into strips of 32 columns (one
`.32x32b` instruction each): strip s+1's face reads / chain are issued after strip s's `LDTM` and before its multiply -- only two strips'
factors are live at any time, still all in the shadow of the wait.

**A cheaper level: a rolling lead per run** (when the budget cannot even hold two strips, e.g. when the segment must also keep E
accumulator f32 + E/2 bf16 pairs of another operand live, leaving only about 110 registers for the whole segment). Only the face reads of the first S runs
go before the wait; in the run loop after the wait, first issue the vector read of run q + S, then multiply run q -- each vector read
leads its consumer by S runs, at most S + 1 runs' scalars are live at any moment, and the first S runs lead the whole wait.
S is tunable (the certified reading uses 4). The guard `q + S < len(RUNS)` is a Python boolean, but **must be written as
`if cutlass.const_expr(...)`**: a bare `if` inside a trace body is lowered to a run-time region regardless of its condition, and the values of vector reads issued inside the region
are consumed outside the region (the next round's multiplication) -- exactly the shape of
[`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) gotcha 1.

```python
s_a = [None] * E
for q in cutlass.range_constexpr(S):                       # the first S runs: before the wait
    run = RUNS[q]; v = lds_v4(face_a + cAcc[run[0]][1] * 4)
    for k, e in enumerate(run): s_a[e] = v[k]
cute.arch.mbarrier_wait(mbar + BAR_ACC_FULL, it % 2)
cute.copy(tiled_t2r, tSrc, rAcc); cute.arch.fence_view_async_tmem_load()
for q in cutlass.range_constexpr(len(RUNS)):
    if cutlass.const_expr(q + S < len(RUNS)):              # guard folded away at trace time
        nrun = RUNS[q + S]; v = lds_v4(face_a + cAcc[nrun[0]][1] * 4)
        for k, e in enumerate(nrun): s_a[e] = v[k]        # run q+S's read issued first
    e0, e1 = RUNS[q][0], RUNS[q][1]
    rAcc[e0], rAcc[e1] = mul2_pair(s_a[e0], s_a[e1], rAcc[e0], rAcc[e1])   # then multiply run q
```

When a fragment of the `Ld16x256b` shape lands on the same (128, 64) plane, its runs are only two columns long (first column even),
the vector-read ceiling is `ld.shared.v2`, and `v4` does not hold -- the run table is still read from the identity partition per §(b);
do not extrapolate from the four-column runs of `.32x32b`.

### (e) Acceptance: SASS census + one time ratio (u = trip-loop unroll factor, taken as `LDTM / LDTM per trip`)

1. `LDS.128 ≥ F·E/4·u`, total `LDS` ≤ `F·E/4·u + a few` (`F·E·u` before the change);
2. `FFMA2 + FMUL2 ≥ E·u` (one each per pair), scalar `FFMA + FMUL` independent of E (≈ 0);
3. `LDL == STL == 0`;
4. Numerics: ~1e-6 relative error against the float64 reference (`.ftz` is bit-equivalent to scalar `fma`/`mul` when both sides are in the normal range);
5. Time: with no handshake, 2 warps per scheduler and 1 CTA/SM, `t(scalar records + scalar multiply) / t(face + v4 + packed) ≥ 1.03`.

### Gotchas of this shape

- **Token-major records with odd F cannot be read as vectors; this is a property of the layout, not a missed compiler optimization.** Switch to faces; do not tune the compiler.
- **Every row of a face must be 16 B aligned** (`K·4` a multiple of 16 suffices): a misaligned `ld.shared.v4` is an `illegal address`.
- **Read `perm` out before using it.** Even if the mirror atom's write-back slot order and readback element order are the identity today, that is a property of the atom,
  and may change with a different repetition / plane shape; a wrong write-back permutation = silently wrong columns, with the numeric contract as the safety net.
- **Both fences of the in-place write-back are required**: after `LDTM`, `fence_view_async_tmem_load` (the same address may be overwritten only after the registers have the data);
  after `STTM`, `fence_view_async_tmem_store` (only then does the next trip's `LDTM` see it).
- **The packed helpers carry no decorator** (they return a pair, which `dsl_user_op` cannot describe), and `ir.Type.parse` is called at trace time;
  vector reads `has_side_effects=True + ~{memory}`, packed arithmetic `has_side_effects=False` (a pure function; if its result is unused it gets eliminated --
  exactly the property it should have).
- **`.ftz` consistent on both sides** (`packed_f32x2_chain_single_asm` gotcha); if the scalar control uses the DSL's `*`/`+`,
  it lowers to `fma` without `.ftz` -- diverging only on denormals.

## Sources

- PTX ISA: `selp` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>;
  `setp` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>;
  `ld` (the `.shared` state space, the `.v2`/`.v4` vector forms and alignment requirements)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>;
  `mbarrier.try_wait` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait>;
  the `.16x256b` shape and datapath layout of `tcgen05.ld` / `tcgen05.st`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-ld>;
  `%globaltimer` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer>.
- CUDA Binary Utilities, Instruction Set Reference (`FSEL`, `BSSY`/`BSYNC`, `LDS`, `MUFU` as SASS mnemonics)
  <https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference>.
- NVVM/LLVM inline asm constraints (`r` / `f` / `=f`, the `~{memory}` clobber) <https://llvm.org/docs/NVPTXUsage.html>,
  <https://llvm.org/docs/LangRef.html#inline-assembler-expressions>.
- Public infra API symbols: `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.cutlass_dsl.T`,
  `cutlass._mlir.dialects.llvm.inline_asm` / `llvm.extractvalue`, `cutlass._mlir.ir.Type.parse`,
  `cutlass.Int32(...).ir_value` / `cutlass.Float32(...).ir_value`, `Pointer.toint`,
  `cute.make_identity_tensor`, `cute.flat_divide`, `cute.make_rmem_tensor`, `cute.size`,
  `cutlass.range` / `cutlass.range_constexpr` / `cutlass.const_expr`, `cute.math.exp2(..., fastmath=True)`,
  `cutlass.cute.nvgpu.tcgen05.make_tmem_copy` / `Ld16x256bOp` / `St16x256bOp` / `Repetition.x8` / `Unpack.NONE` /
  `MmaF16BF16Op` / `CtaGroup.ONE` / `OperandSource.SMEM` / `OperandMajorMode.K`, `TiledMma.make_fragment_C` /
  `partition_shape_C` / `get_slice(...).partition_C`, `ThrCopy.partition_S` / `partition_D`,
  `cute.arch.alloc_tmem` / `relinquish_tmem_alloc_permit` / `retrieve_tmem_ptr` / `dealloc_tmem`,
  `cute.arch.fence_view_async_tmem_load` / `fence_view_async_tmem_store`,
  `cute.arch.mbarrier_init` / `mbarrier_init_fence` / `mbarrier_wait` / `mbarrier_arrive`,
  `cute.arch.barrier(barrier_id=, number_of_threads=)`, `cute.arch.elect_one`, `cute.arch.make_warp_uniform`,
  `cute.arch.globaltimer`, `cutlass.utils.SmemAllocator.allocate_array` / `allocate_tensor`.
- GPU certification probe -- a 64 × 64 fp32 plane,
  `St16x256b` x8 writing known values into TMEM by identity coordinates, `Ld16x256b` x8 reading them back (32 elements per thread),
  row scalars `(s, a)` interleaved, column scalars contiguous by column, mask `col < row`, factor `a · exp2(s_row + b_it − s_col)`,
  one scalar `b_it` read from gmem per trip so the chain cannot be folded as a loop invariant, row/column scalars kept in a 2-slot ring and fetched by a dynamic slot number each trip
  (the address is not loop-invariant, matching the scalars republished stage by stage in a real pipeline); three shapes, one process each: `if_scalar`
  (per-element `if` + 3 scalar reads, chain after the wait), `sel_vec` (the select + vector reads of §(a)(b), chain after the wait),
  `sel_vec_hoist` (§(c)); the handshake shape additionally has a producer warp: wait for the consumer's `empty`, spin 400 ns on `%globaltimer`,
  then arrive on `full`; the no-handshake shape has 3 consumer warpgroups per CTA reading the same TMEM plane, each writing its own
  output plane (3 warps per scheduler). Six contracts: **frag_roundtrip** (TMEM round trip bit-equal, all 4096 coordinates covered,
  and the run table dumped) / **numeric** (all three shapes, the output planes of all warpgroups within relative error ≤ 1e-4 of the float64 reference,
  masked elements exactly 0) /
  **sel_census** (`sel_vec_hoist`: `FSEL ≥ 32`, `MUFU.EX2 ≥ 32`, `LDL = STL = 0`, `BRA < E = 32`, `BSSY ≤ 6`) /
  **lds_batched** (`LDS ≤ 32u` and containing `LDS.128`/`LDS.64`) / **sel_time** (no handshake, 1 CTA/SM, 3 warps per scheduler:
  `t(if_scalar)/t(sel_vec) ≥ 1.03`) / **hoist_time** (handshake + 400 ns delay, 1 warp per scheduler:
  `t(sel_vec)/t(sel_vec_hoist) ≥ 1.03`).
  Round 1 readings (1 warp per scheduler): frag_roundtrip 0.0, numeric 3.87e-07, `sel_vec_hoist` census
  `FSEL=32 MUFU.EX2=32 LDS=13 (64:10 32:3) LDL=STL=0 BRA=23 BSSY=0 REG=158` vs `if_scalar` `LDS=46 BRA=11 BSSY=0`,
  hoist_time 1.124 (1.408 / 1.253 ms) -- all pass; sel_time 1.017 failed (the source of gotcha 12), and round 2 switched to 3 warps per scheduler.
  Round 2 readings (sel_time at 3 warps per scheduler, scalar ring with dynamic slots): frag_roundtrip 0.0, numeric 3.87e-07,
  sel_time 1.450 (1.609 / 1.109 ms; `if_scalar` `LDS=47 BRA=13 instrs=560 REG=87` vs `sel_vec` `LDS=13 BRA=13 instrs=504 REG=95`),
  hoist_time 1.429 (1.745 / 1.221 ms), lds_batched `LDS=13 (64:10 32:3)` -- all pass; `sel_vec_hoist` census
  `FSEL=32 MUFU.EX2=32 BRA=25 BSSY=0 LDL=STL=0 REG=158`, `BRA` 25 hit the fixed cap of 24 set in round 1 (the handshake ring added 2 role/slot branches),
  so round 3 changed the criterion to the E-tied `BRA < E` (see §(d)-3).
  Round 3 readings (certification round): frag_roundtrip 0.0 (all 4096 coordinates covered, unset 0), numeric 3.87e-07
  (all four shapes `if_scalar` / `sel_vec` / `sel_vec+hs` / `sel_vec_hoist+hs` ok, masked elements exactly 0),
  sel_census `FSEL=32 ISETP=84 MUFU.EX2=32 FMUL=64 FADD=66 LDS=13 (64:10 32:3) LDL=STL=0 BRA=25 BSSY=0 SYNCS=10 LDTM=1 STTM=1 REG=158 instrs=616 u=1`
  (`BRA < E` holds: 25 < 32; the handshake ring adds 12 `BRA` relative to the no-handshake shape), lds_batched `LDS=13` vs `if_scalar` `LDS=47`,
  sel_time 1.457 (1.615 / 1.108 ms, 3 warps per scheduler), hoist_time 1.429 (1.746 / 1.222 ms, 400 ns delay;
  `sel_vec+hs` 1.746 ms / `REG=96` vs `sel_vec_hoist+hs` 1.222 ms / `REG=158` -- the cost of hoisting the chain is 62 registers, see §(c)).
  Status: **✅ GPU-certified** -- `PROBE PASS max_diff=3.8657465348084646e-07`.
- GPU certification probe of the variant section -- a 128 × 64 fp32 plane, written by `St32x32b` x32 at identity coordinates and read back by `Ld32x32b` x32 (64 elements per thread, one row); the two column-scalar fields each stored both as a token-major record (F = 3, odd) and as a field-major face (2-slot ring, dynamic slot number per trip); one scalar `b_it` read from gmem per trip; the pass multiplies the plane in place by `a[c]·b_it + c[c]`; three shapes, one process each: `scalar_lds` (records + 2 scalar reads per element + scalar `FFMA`/`FMUL`), `vec_packed` (faces + v4 + packed, chain after the wait), `vec_packed_hoist` (§(d), handshake shape). Six contracts: **frag_roundtrip** (round trip bit-equal, all 8192 elements covered) / **rowlane_struct** (one row per thread, run length 4, E/2 pairs, a write-back permutation exists and is consistent across threads) / **numeric** (three shapes within relative error ≤ 1e-4 of float64) / **lds_vec** (`LDS.128 ≥ 32u`, `LDS ≤ 40u`, `LDL = STL = 0`) / **packed_census** (`FFMA2 ≥ 32u`, `FMUL2 ≥ 32u`, scalar `FFMA + FMUL ≤ 8u`) / **colvec_time** (no handshake, 2 warps per scheduler: `t(scalar_lds)/t(vec_packed) ≥ 1.03`).
  Round 1 readings (certification round): frag_roundtrip 0.0 (all 8192 elements covered, unset 0), rowlane_struct
  (exactly one row per thread, run length 4, 32 pairs, write-back permutation the identity and consistent across threads), numeric 1.02e-06 (all three shapes `scalar_lds` / `vec_packed` /
  `vec_packed_hoist+hs` ok, tol 1e-4), lds_vec `LDS=35 (128:32 64:0 32:3) LDL=STL=0`
  vs the control `scalar_lds` `LDS=131 (128:64 64:0 32:67)` -- 2 scalar reads per element collapsed into 1 `LDS.128` per 4 columns,
  packed_census `FFMA2=32 FMUL2=32 FADD2=0 F2_OTHER=0`, scalar `FFMA=FMUL=0` (control `FFMA=128 FMUL=128`),
  colvec_time 1.379 (0.980 / 0.711 ms, 2 warps per scheduler; `scalar_lds` `REG=100 instrs=704` vs `vec_packed`
  `REG=92 instrs=512 LDS=67 (128:64 32:3) FFMA2=64 FMUL2=64`; with u = 2 the vector and packed counts scale linearly with u);
  handshake shape `vec_packed_hoist+hs` `REG=140 instrs=376 SYNCS=10 LDTM=4 STTM=4 BRA=25 BSSY=0`, median 1.211 ms
  (148 CTAs × 160 threads, 2048 trips, 400 ns delay) -- the cost of hoisting the face reads + factor chain before the wait is of the same type as the parent entry's §(c) (registers 92 → 140).
  Status: **✅ GPU-certified** -- `PROBE PASS max_diff=1.020912233514069e-06`.
- **KEEP evaluation of the parent recipe (three levers) on a real tree** (NCU dossier census + gate window, the index's second class of evidence; the eval job number was not recorded):
  on a tree with a single CTA, three `setmaxnreg` roles and 1 CTA/SM, two `Ld16x256b` x8 masked element-wise segments of one consumer warpgroup (E = 32, mask `col < row`)
  first landed per §(a)(b)(c) in one package -- per-element `if` replaced by a `selp` product select, row/column scalars replaced via the table dumped from the identity partition (2 row groups × 16, 16 two-column runs)
  by `ld.shared.v2` (scalar `LDS` per thread per chunk 96 → 18, `MUFU.EX2` 32 unchanged), and the factor chain hoisted before the accumulator `full` wait; two new scalar faces
  +1.5 KiB smem, zero changes to barriers / tiles / data flow. gate **0.20600 → 0.16109 ms (−21.8%)**, NCU kernel clock 339.5 → 264.0 µs (−22.3%),
  `stall_long_sb` 5.17 → 4.47, `tensor_pct` 3.23 → 4.17; this step was the largest single step on that tree up to meeting the bar. A later KEEP only tightened lever (3) (`G·log2e`
  premultiplied on the publishing side, `scale` folded into the hoisted chain, 32 fewer `FMUL` per thread per chunk; one more TMEM read wait moved into the store/fence shadow): 0.16109 → 0.15536 ms.
  **Certification boundary [UNVERIFIED-BOUNDARY]**: the tree's subsequent margin from 0.80x to 0.99x came from **another segment** of the same role (a blocked restructuring and
  in-segment ILP of a 64 × 64 unit lower-triangular inverse), unrelated to this entry's three levers, and this entry does not take credit for that segment; the three levers landed in one package, so the production price of a single lever exists only as the probe's `sel_time` / `hoist_time` ratios,
  not priced separately on a real tree.
- **KEEP evaluation of the variant section** (NCU dossier census + gate window, the index's second class of evidence; the eval job number was not recorded): on a tree with a single CTA,
  three `setmaxnreg` roles and 1 CTA/SM, three element-wise segments of one consumer warpgroup (one of the `Ld16x256b` x8 shape + two of the
  `Ld32x32b` x32 shape, E = 64) rewritten per this variant: token-major records (F = 5, odd) replaced by a field-major face array
  `[slot][field][token]` (the same 2560 B, 256 B per row, 16 B aligned), two fields premultiplied / pre-negated on the publishing side (the publishing warp does
  2 more `FMUL`/`FNEG` per lane per chunk, while each of the 128 consumer lanes saves one), `v2` (two-column runs) / `v4` (four-column runs) vector reads + `FFMA2` /
  `FMUL2`, rolling lead per run with S = 4 (the register-saving level of §(d)). Zero changes to the mbarrier source, register tiers unchanged. gate 0.12595 →
  0.11789 ms (**−6.40%, under this structure**: across the three segments 256 → 96 fewer static `LDS` and 64 fewer scalar `FMUL` per lane per chunk,
  8 more `STS` with 2-way conflicts on the publishing side; a same-job ablation on that tree had priced the ceiling of these three segments' scalar reads at +7.8%); contract axis `rings`
  `[stride=0x1, stride=0x1,slots=536870911]` → `[stride=0x1]` (codegen moving along, unrelated to this rewrite; for its interpretation see
  [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11), `merge` moving along. A compile-contract probe
  (no TMEM / mbarrier / roles; it runs only the staging mechanism inside a dynamic-trip loop) pinned down the two causes of the
  two `cute.compile` failures before landing: a nested `def` capturing traced values ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)), and
  subscripting a Python list with the induction variable of `cutlass.range` ([`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) gotcha 8).
