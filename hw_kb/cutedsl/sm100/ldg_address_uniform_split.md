# ldg_address_uniform_split (global load address = a warp-invariant uniform base + one shared per-thread offset: pushing down the per-thread integer instructions between the entry and the last `LDG`; the compile-time constant displacements of the thread's other loads / stores folded into `[R.64+imm]`)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Related: [`ldg_issue_order_pin`](ldg_issue_order_pin.md) (the load **issue position**; this entry is about **how the load address is computed**; the two are orthogonal).
>
> **Status: ✅ Verified** -- certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=5.96e-08`,
> round 1 of this certification, both contracts pass in the same round: three variants numerically against numpy; SASS shape (`LDG` / `STG` counts matching the probe source) + 0 `LDL/STL`, and for the two split variants
> the per-thread integer instructions "entry → last `LDG`" strictly fewer than the naive spelling.
>
> **What is certified**: only **the SASS footprint and the numerics**, no timing. The same-job paired timing of the same mechanism on a whole kernel is **flat** (see gotcha 7).
>
> The section "Folding in-batch constant displacements into the immediate" has three further independent certifications, likewise certifying only the SASS footprint and the numerics:
> - load side: `PROBE PASS max_diff=0.0`, round 1 of this certification, the numeric + SASS shape contracts both pass in the same round;
> - store side: `PROBE PASS max_diff=0.0`, round 1 of this certification, the numeric + SASS shape contracts both pass in the same round;
> - per-thread offset computed in 32 bits and widened once (gotcha 13): `PROBE PASS max_diff=0.0`, round 2 of this certification (in round 1 the SASS window contract failed),
>   two variants bit-identical numerically + one SASS window contract, all three pass in the same round.
> Whole-kernel readings: load side flat, store side clearly faster, gotcha 13's spelling flat within its package (see "Whole-kernel readings" in that section).

## Purpose

Each SM100 warp has two integer datapaths:

- **the vector (per-thread) path**: `R*` registers, `IMAD` / `IMAD.WIDE` / `IADD3` / `LEA` / `LOP3` / `SHF` / `ISETP` / `SEL`.
  Each takes one issue slot of the warp;
- **the uniform path**: `UR*` registers, `UIMAD` / `UIADD3` / `ULEA` / `ULOP3` / `USHF` …, computed once per warp,
  sourced from `S2UR SR_CTAID.*` (blockIdx), kernel parameters in the constant bank (`LDCU` / `c[0x0][...]`), and immediates.

ptxas does its own uniform analysis: **as long as a value is derived only from blockIdx / kernel parameters / compile-time constants**, it can live in `UR*`,
and the address operand of `LDG` / `STG` can take a `UR` directly (`IADD3 R, P, PT, R_off, UR_base, RZ` + `IADD3.X`) as well as an immediate offset (`[R.64+0x10]`).
**Once `threadIdx` is mixed in before the multiplication by the row pitch / the modulo / the multiplication by the element width**, the whole chain after it becomes per-thread instructions, and each load stream computes it once more.

This entry's spelling: **split the address into "the part depending only on blockIdx / parameters / constants (fully computed on the uniform side first)" + "one per-thread offset (computed only once)"**;
all load streams and stores share the same per-thread offset; the second 128-bit load of the same contiguous run uses an immediate `+16`.
For the thread's other loads / stores whose displacements differ only by compile-time constants (across rows and across blocks too), how to write them so that they really fold into immediates is in the section
"Folding in-batch constant displacements into the immediate".

## When needed (trigger)

- A latency-bound small shape (only a few loads per thread, only one load round trip in the whole kernel), while in SASS the stretch **entry → last `LDG`** has
  a dozen or more per-thread integer instructions (`IMAD*` / `IADD3` / `LEA*` / `LOP3` / `SHF` / `ISETP` / `SEL` / `VIADD`), growing linearly with the number of load streams;
- the source writes the address as `base + Int64(row) * row_bytes + Int64(idx) * VEC_B`, where `idx` is `blockIdx * C + threadIdx` (possibly also under a modulo),
  once per stream;
- you have already confirmed that the load **issue position** is fine (see `ldg_issue_order_pin`), and what remains is the integer tail before the `LDG`.
- A batch of loads / stores of the same thread (in-batch displacements differing only by compile-time constants) where in SASS each one is preceded by its own `IADD3` + `IADD3.X` (or `LEA` + `LEA.HI.X`, `IMAD.WIDE*`),
  the addresses are `[R<n>.64]` with a different `R<n>` each; or NCU hangs `wait` / `short_scoreboard` on these address instructions -- see "Folding in-batch constant displacements into the immediate".
  This shape measured clearly faster on a whole kernel for a tail write-back (each thread writing a run of scalars along the row pitch), and flat for a prologue load batch.
- The entry window has one (signed) `IMAD.WIDE`, and in the source every term of the per-thread offset is first `Int64(...)`, then multiplied, then added -- see gotcha 13 (it differs by only this one instruction).

**Not the right remedy**: bandwidth-bound large shapes (the entry instructions are covered by the load round trip); a busy FP pipe (this only touches the integer pipe).
**Price it first**: on the whole kernel where this entry was found the spelling was flat (gotcha 7); a few fewer integer instructions at the entry is not the same as faster end to end.

## Minimal runnable example

```python
import cutlass, cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NTHR = 128                    # threads per CTA (arbitrary)
NV = 2                        # how many consecutive 128-bit loads per thread per stream (arbitrary)
VEC_B = 16 * NV               # contiguous bytes per thread per stream (NV 128-bit loads)

@dsl_user_op
def ld4(addr, *, loc=None, ip=None):          # one 128-bit non-coherent read
    s = llvm.inline_asm(llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), s, [i], loc=loc, ip=ip)) for i in range(4)]

@cute.kernel
def k(x_a: Int64, v_a: Int64, y_a: Int64, rowb: Int32):
    tid, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    # ---- uniform part: only blockIdx / parameters / constants -- ptxas puts it in UR, one copy per warp
    urow = Int64(by) * Int64(rowb)                 # still uniform with a run-time row pitch (gotcha 4)
    ucol = Int64(bx * NTHR) * Int64(VEC_B)         # start of this CTA's column slice
    # ---- per-thread part: one offset, shared by all streams
    off = Int64(tid * VEC_B)
    ax = (x_a + urow + ucol) + off                 # one row of a 2D tensor
    av = (v_a + ucol) + off                        # a 1D vector (by column)
    ay = (y_a + urow + ucol) + off                 # output, same shape as x
    xv = [w for i in range(NV) for w in ld4(ax + 16 * i)]   # the later ones of the same run: immediate +16*i, no separate address computation
    vv = [w for i in range(NV) for w in ld4(av + 16 * i)]
    # ... consumers; stores use ay + 16*i (the same off; the store address is no longer an independent integer chain)
```

Control (**counterexample**, a whole per-thread chain per stream):

```python
    c = bx * NTHR + tid                            # threadIdx mixed in before the multiplication by the element width
    ax = x_a + Int64(by) * Int64(rowb) + Int64(c) * Int64(VEC_B)
    av = v_a + Int64(c) * Int64(VEC_B)
    ay = y_a + Int64(by) * Int64(rowb) + Int64(c) * Int64(VEC_B)
```

Acceptance (`cuobjdump -sass`): count the per-thread integer instructions between **the entry and the last `LDG`**, and the same kind between the last `LDG` and the first consumer;
count `U*` instructions (not `ULDC`) separately. Readings are under "Measured".

## Measured

Certification probe, 128 threads / CTA, three variants numerically identical (max_diff 5.96e-08). One of the probe's read streams has a modulo in its address
(the sign correction of gotcha 3 comes from it), so its shape differs from the example above -- the absolute counts in the table hold only for the probe's shape; look at the relative changes.
W0 = entry → first `LDG`, W1 = entry → last `LDG`, W2 = last `LDG` → first `STG`; integer = per-thread `IMAD*/IADD3/LEA*/LOP3/SHF/ISETP/SEL/VIADD`,
U = uniform ALU (not counting `ULDC`).

| Spelling | W0 integer | W1 integer | W1 U | W2 integer | first `LDG` slot (from 0) | instructions from entry to last `LDG` (inclusive) |
|---|---|---|---|---|---|---|
| naive (a whole per-thread chain per stream, run-time row pitch) | 9 | 22 | 1 | 3 | 19 | 38 |
| split (uniform base + one offset, row pitch a run-time parameter) | 12 | **13** | 9 | 3 | 30 | 37 |
| split (same, row pitch a compile-time constant) | 17 | 18 | 3 | **0** | 27 | 34 |

How to read it:
- Per-thread integer W1 22 → 13 (run-time row pitch): the uniform part really landed in `UR*` (W1 U 1 → 9), and the address operand is `IADD3 R, P, PT, R_off, UR_base, RZ` + `IADD3.X`.
- **A compile-time constant row pitch is actually worse** (W1 18, U 3): ptxas folds the constant multiplication back into per-thread forms like `IMAD.WIDE` / `IMAD.SHL`, and the uniform side shrinks. See gotcha 4.
- **The first `LDG` comes later** (slot 19 → 30 / 27, W0 9 → 12 / 17): after the split, ptxas computes all addresses before issuing all the `LDG` back to back; the naive spelling issues as it computes.
  What drops is the total "before the last `LDG`", not the amount "before the first `LDG`". See gotcha 6.

## Folding in-batch constant displacements into the immediate (loads and stores alike)

The address operand of a SASS global memory access is `[R.64+imm]`: a 64-bit base register pair plus a signed immediate encoded in the instruction.
ptxas can fold "**a pointer that is already 64-bit** + a compile-time constant" directly into this immediate. So for a batch of accesses of the same thread whose displacements differ only by compile-time constants,
only one pair of base registers is needed, and every other access is a `[R.64+0x…]` with zero extra integer instructions. Two spellings block the folding, leaving each access its own 64-bit address chain
(`IADD3` + `IADD3.X`, `LEA` + `LEA.HI.X` or `IMAD.WIDE*`):

- **The constant is wrapped inside a 32-bit sum and then widened**: spellings like `base + Int64(row_t + R_i) * ROW_B` (`R_i` being the compile-time row displacement of the i-th access), where the constant `R_i` first goes into a 32-bit addition and is then `sext`-ed to 64 bits.
  A 32-bit addition can wrap, so `sext(a + C)` cannot be rewritten as `sext(a) + C`, and the compiler can only compute it literally;
- **The displacement has a run-time factor**: most commonly the tensor's row pitch is a run-time value (a layout from `mark_layout_dynamic`, or any dynamic stride),
  so the offset of `mY[(row0 + k, col)]` is `(row0 + k) * stride_rt + col`, one 64-bit multiply-add per store.

Spelling:

```python
from cutlass import Int64
# load: compute only one per-thread 64-bit pointer p for the whole batch (the uniform part added first, see above); the in-batch displacement C is a Python int (trace-time constant)
p = ubase + Int64(lane_off)                # ld_v8: the `ld.global.nc.v8.b32` version of ld4 above (8 "=r" + one "l")
regs = [ld_v8(p + C) for C in DISP]        # spelling 1: the constant added to the Int64 pointer, folded by ptxas
# spelling 2: write C directly into the inline asm's address template, not relying on ptxas folding:
#   "ld.global.nc.v8.b32 {$0, ..., $7}, [$8+%d];" % C    (write [$8] when C == 0)

# store: compute the pointer to (row0, col) only once per thread, wrapped as a 1-D view with a compile-time constant row pitch
yt = cute.make_tensor(mY.iterator + (row0 * N + col),          # N: row length (elements), a Python int
                      cute.make_layout((T,), stride=(N,)))     # stride is a Python int → static
for i in cutlass.range_constexpr(T):
    if row0 + i < m_tot:                   # the bounds guard stays a predicate, not part of the address
        yt[i] = v[i]                       # → STG [R.64 + i*N*elem_bytes]
```

Counterexample:

```python
off = (Int64(row_t + R_i) * ROW + Int64(col_t + C_i)) * 2   # constants R_i, C_i inside the 32-bit sum
regs.append(ld_v8(base + off))
mY[(row0 + i, col)] = v[i]                 # mY has a run-time row pitch → one IMAD.WIDE per STG
```

Acceptance (`nvdisasm -c` / `cuobjdump -sass`): the addresses of all accesses in the batch are **the same** `[R<n>.64+0x…]`, differing only in the immediate (the first one may have none);
the per-thread integer instructions and `.X` carry instructions in the entry → last-access window are fewer than in the counterexample.

### Readings

Load side (a batch of `N_LD` 256-bit `ld.global.nc.v8.b32`, all displacements compile-time constants; W = entry → last `LDG`):

| Spelling | distinct `LDG` bases | `LDG` with immediate | W per-thread integer | of which `.X` carries | last `LDG` slot |
|---|---|---|---|---|---|
| counterexample (each its own `Int64(row + constant) * ROW`) | 7 | 0 | 61 | 11 | 92 |
| uniform base already hoisted, constant still inside a 32-bit sum then widened | 1 | `N_LD − 1` | 19 | 2 | 49 |
| one `Int64` pointer + Python int displacement (spelling 1) | 1 | `N_LD − 1` | **18** | 2 | 46 |
| same, displacement written into the asm address template (spelling 2) | 1 | `N_LD − 1` | **18** | 2 | 46 |

The second row is a borderline case: with only one copy of the per-thread part left, ptxas folded it into the immediate this time too, but that relies on it proving no wraparound by itself and is not guaranteed (gotcha 8).
The largest immediate was about 0.46 MB and was still folded into the `LDG`.

Store side (each thread writes `T` fp16 scalars along the row pitch (`T` in the tens), with a `row < M` predicate; all 6 run-time M bit-exact):

| Spelling | `STG` with immediate | base registers | `IMAD.WIDE*` | total SASS instructions of the probe kernel |
|---|---|---|---|---|
| counterexample: `mY[(row, col)]`, run-time row pitch | 0 | 2 | `T` | 616 |
| base computed only once, but the 1-D view's stride taken from the run-time `mY.layout.stride[0]` | 0 | 3 | about `0.85·T` | 480 |
| base computed only once + Python int row pitch (the spelling above) | `T − 1` | 1 | 1 | 296 |

Hoisting only the base while the row pitch is still a run-time value removes only about 15% of the wide multiplications: to fold into the immediate, the displacement must be a compile-time constant in its entirety (gotcha 9).

The bit width of the per-thread part (the run-time offset) itself is covered by gotcha 13: when it can be proven non-negative and < 2³¹, compute it fully in `Int32` and widen once with `cvt.u64.u32`;
on another probe the per-thread integers of the entry window went 14 → 13, removing only that `IMAD.WIDE`; immediate folding happened with both spellings.
Gotcha 8 (compile-time constants added to an already widened pointer) and gotcha 13 (the run-time per-thread offset computed in 32 bits and then widened) govern the two parts of the address; together they are
`p = base + widen_u32(off32)`, then `p + C`.

### Whole-kernel readings

- **Load side flat**: on the whole kernel where it was found (a latency-bound small-shape kernel, a batch of 256-bit loads in the prologue), switching to spelling 2 gave
  gate 0.0055 ms (previous best 0.0054 ms, device window at two significant digits); afterwards a same-job pairing (single base + immediate displacement vs computing each address separately, event timing after an L2 flush,
  geometric mean over 8 M) gave 12.543 vs 12.537 µs, flat. A few dozen fewer integer instructions at the entry is not the same as faster end to end.
- **Store side clearly faster**: on another whole kernel (each write-back thread writes a few dozen fp16 scalars along the row pitch, the write-back on the tail's critical path):
  switching only the write-back to the spelling above, the ablation pricing gave NCU kernel clock +14.16%; after landing, gate 0.0132 → 0.0113 ms (the confirmation run reads 0.0113).
- **Gotcha 13's spelling flat within its package**: on a third whole kernel it landed in the same package as an issue-order adjustment of the entry segment, gate 0.0084 → 0.0084 ms (geometric mean over 8 workloads), not a separate A/B.

## Gotchas

1. **Where `threadIdx` gets mixed in decides which register file the whole chain lives in.** `Int64(bx*C + tid) * VEC_B` drags both the multiplication and the base addition into per-thread;
   `Int64(bx*C) * VEC_B + Int64(tid*VEC_B)` costs only one extra per-thread addition. Rule: add all uniform terms first, and add the per-thread offset **last**.
2. **Source order is not a lever.** Writing the store address / predicate after the loads, ptxas can still schedule them before the `LDG` (or the other way round).
   What changes the SASS is the **count**: let the store reuse the load's per-thread offset and it no longer has its own chain.
3. **Signed modulo.** An `Int32 % 2^k` in the address lowers to `srem`; ptxas does not know the dividend is non-negative and adds sign-correction instructions
   (the naive variant's W1 has `SHF.R.S32.HI` ×2, `LEA.HI` ×2); on the uniform side it is paid once per warp, on the per-thread side per thread.
4. **A run-time row pitch does not drive the base out of the uniform file**: kernel parameters are warp-invariant by nature (the constant bank), and `Int64(by) * Int64(rowb)` is still a `UIMAD`.
   There is no need to make the row pitch a compile-time constant for uniformity -- **measured, a constant row pitch is worse**: W1 per-thread integers 13 → 18, uniform instructions 9 → 3.
5. **Accept by counting the window, not the whole kernel.** The whole kernel's integer instruction count is diluted by consumers, stores and teardown; what to look at is the stretch entry → last `LDG`.
6. **"Fewer instructions before the last `LDG`" is not "the first `LDG` comes earlier".** After the split ptxas tends to compute all addresses before issuing all the `LDG` back to back
   (measured first `LDG` slot 19 → 27 / 30). If your critical path is the landing of the **first-consumed** group of loads, count W0 and W1 separately,
   and use [`ldg_issue_order_pin`](ldg_issue_order_pin.md) to manage the issue position; count and position are two things, each priced on its own.
7. **On a whole kernel it is flat.** On the whole kernel where this entry was found (a latency-bound straight-line row-reduction kernel, 6 shapes × 9 rounds, same-job paired geometric mean, ratio = control time / variant time):
   switching to this entry's spelling on a kernel whose issue position was already pinned gave 0.996 with a run-time row pitch and 1.004 with a compile-time constant row pitch.
   The spelling's SASS footprint is real, but on this kind of "one load round trip dominates" kernel it does not bear load; price it same-job before acting.
8. **The constant must be added to an already widened pointer.** `Int64(a + C)` and `Int64(a) + C` are not mathematically equivalent (the 32-bit `a + C` can wrap);
   to fold into the immediate write `p + C`: `p` is already `Int64`, `C` a Python int. With the uniform part hoisted and only one per-thread term left, ptxas can sometimes
   fold it by itself (second row of the reading table); that is its decision, not a guarantee; count the SASS after the change. This is about **compile-time constant** displacements; a run-time per-thread offset can be computed in 32 bits
   and widened once, see gotcha 13.
9. **The displacement must be a trace-time constant in its entirety.** The loop variable of `cutlass.range_constexpr` times a Python int is; the run-time loop variable of `cutlass.range`,
   a kernel-parameter stride, and the `layout.stride` of a tensor from `mark_layout_dynamic` are not -- building a view with a run-time stride hoists the base,
   but each access's `i * stride_rt` still needs one wide multiplication (second row of the store reading table). When the row pitch is actually fixed in the problem but the tensor object carries it in as a run-time value,
   build the view with a Python int.
   Not a contradiction with gotcha 4: in gotcha 4 the constant row pitch multiplies a **run-time** index (blockIdx), and ptxas folds the constant multiplication back into per-thread; here the constant multiplies a **compile-time** unrolled variable,
   the product itself is a constant and goes straight into the immediate. Criterion: whether the displacement has any run-time factor besides constants.
10. **The bounds guard stays in the predicate.** `if row0 + i < M:` is `ISETP` + `@P STG` and does not touch the address; the spelling of the store reading table carries this guard and still folds into the immediate.
    When some load in the batch must be skipped on a run-time condition, put the predicate into the asm too (`@p ld…`), and still write the address as `p + C`.
11. **Immediate range and alignment.** The probes cover displacements up to about 0.46 MB (loads) and about 0.3 MB (stores); larger displacements were never tested, and beyond the encoding range
    ptxas falls back to a register addition -- count the SASS before relying on it. Alignment is a requirement on the final address `p + C` (32 B for `v8.b32`, 16 B for `v4`): `p` aligned and `C` an integer multiple of the alignment granule
    suffices; the immediate does not relax alignment.
12. **A shared base binds the whole batch to one pair of registers**: that pair cannot be reused before the last access; watch `LDL/STL` when registers are near the cap.
    A lower count only shortens the integer chain before the accesses; it does not manage the issue position of the `LDG` (ptxas may still sink it near the consumers, see [`ldg_issue_order_pin`](ldg_issue_order_pin.md)).
13. **Bit width of the per-thread offset: if it can be proven non-negative and < 2³¹, compute the whole stretch in `Int32` and widen only once at the end.** When every term is first `Int64(...)`, then multiplied, then added,
    ptxas treats it as a 64-bit integer chain and leaves an `IMAD.WIDE` in the entry window; instead compute the per-thread byte offset in `Int32`, widen it with one `cvt.u64.u32` (zero extension),
    and add it to the 64-bit base only once. Not a contradiction with gotcha 8: gotcha 8's compile-time constant displacements are still added to the pointer **after** widening (the `a0 + …` in the example below).

    ```python
    @dsl_user_op
    def widen_u32(x, *, loc=None, ip=None):          # zero-extend a 32-bit offset to 64 bits: one cvt.u64.u32
        return Int64(llvm.inline_asm(
            T.i64(), [Int32(x).ir_value(loc=loc, ip=ip)], "cvt.u64.u32 $0, $1;", "=l,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    off = row * ROW_B + col_b                        # all Int32; write down the upper bound first and confirm < 2^31
    a0 = x_a + widen_u32(off)                        # one widening + one 64-bit addition
    v0 = ld(a0)
    v1 = ld(a0 + K_ROWS * ROW_B)                     # another row of the same tensor: a Python integer constant
    v2 = ld(a0 + CHUNK_B)                            # the next run of the same row: a Python integer constant
    ```

    - **Measured** (the two variants differ only in the address spelling: every term promoted to `Int64` first vs this spelling; 6 prologue 256-bit `LDG`):
      W1 per-thread integers 14 → 13, of which `IMAD.WIDE` 1 → 0, first `LDG` slot 20 → 19, both variants with 6 prologue `LDG`.
      With this spelling each tensor base is exactly one pair `IADD3 Rlo, P, PT, Roff, URbase, RZ` + `IMAD.X Rhi, RZ, RZ, URbase_hi, P` (the high half of the zero extension is directly `RZ`).
    - **Immediate folding is not brought by this spelling**: both variants folded "another row of the same tensor / the next run of the same row" into `[R.64+imm]` (6 `LDG` using only 2 address register pairs).
      The difference of this spelling from all-`Int64` is only that one `IMAD.WIDE` -- the gap is 1 instruction. Constant offsets must still be written as Python integers and confirmed in SASS to have entered the immediate field (gotcha 8).
    - **Overflow is silent**: a signed `Int32` product ≥ 2³¹ wraps, and before zero extension the offset must also be non-negative; a wrong address is not necessarily out of bounds, it may just read the wrong numbers.
      When the offset may be ≥ 2³¹ (large tensors, 64-bit batch strides), fall back to 64 bits. An ordinary `Int32` → `Int64` conversion is signed (measured: the instruction left in the all-`Int64` variant is the signed `IMAD.WIDE`, not `.U32`);
      for zero extension use the `cvt.u64.u32` above.
    - **Flat on the whole kernel**: on the whole kernel where it was found, this spelling landed in the same package as an issue-order adjustment of the entry segment, gate 0.0084 → 0.0084 ms (geometric mean over 8 workloads), not a separate A/B;
      no gain is claimed; price it same-job before acting.

## Sources
- PTX ISA: the vector forms of `ld.global.nc` and `st.global`, `cvt` (`cvt.u64.u32` zero extension), the `%ctaid` / `%tid` special registers:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`cuobjdump -sass`; the SASS instruction set list includes the uniform-datapath instructions):
  https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `cute.arch.thread_idx`, `cute.arch.block_idx`, `cutlass._mlir.dialects.llvm.inline_asm`,
  `cutlass.cutlass_dsl.dsl_user_op`, `cute.compile`.
- Certification probe (variants naive / split_rt / split;
  `PROBE PASS max_diff=5.96e-08`).
- Certification probes of the section "Folding in-batch constant displacements into the immediate": load side
  (variants naive / widen_late / onebase / asmimm; `PROBE PASS max_diff=0.0`, round 1);
  store side (variants naive / hoist_rt / hoist; `PROBE PASS max_diff=0.0`, round 1);
  gotcha 13 (variants i64 / off32; `PROBE PASS max_diff=0.0`, round 2).
- CuTe DSL API symbols (that section): `cute.make_tensor`, `cute.make_layout`, `Tensor.iterator` (pointer + element offset), `cutlass.range_constexpr`,
  `cute.runtime.from_dlpack(...).mark_layout_dynamic`.
