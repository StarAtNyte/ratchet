# rmem_materialize_packed_reduce (sm100)

> One file per hardware interface. Operator-agnostic: mechanism only.
> Architecture scope: **sm100 (Blackwell) only.**
> `softmax_segment_ilp` §(d) null (4-chain slower than serial) was
> certified on sm90; the public reduction API separates sm90 and sm100 with an
> `arch < 100` branch — that null does not transfer.

## Purpose

On sm100, `cute.arch.add_packed_f32x2` (paired fp32 add, 2 elements per
instruction) and the 3-input form of `nvvm.fmax` can reduce a 128-element row
in 4 independent chains with no extra pack cost — but **only after the operand
TensorSSA has been materialized into a fresh `cute.make_rmem_tensor` via
`.store()`**. Without that step the elements sit in registers already placed by
a prior chain (t2r destination, exp2 output, etc.) and ptxas must emit one
`mov.b64` per pair to satisfy the aligned-pair constraint, costing back what the
packed instruction saved.

The sm90 §(d) null arose because the fragment was pre-placed before measurement.
On sm100 with a fresh allocation, ptxas assigns an adjacent block and packed ops
carry no extra move.

## When needed (trigger)

- You hold a TensorSSA of 128 (or any multiple of 8) fp32 elements from a t2r
  partition and need to reduce to a scalar row sum or row max.
- `softmax_segment_ilp` §(d) is cited as authority against 4-chain
  widening, but the probe that established it predates sm100.
- The addends for `add_packed_f32x2` are the t2r destination registers (or exp2
  output), not a freshly allocated block: ptxas will insert `mov.b64` per pair.

## Minimal runnable example

### Step 1 — materialize before reducing

```python
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass._mlir.dialects import nvvm

# --- 3-input nvvm.fmax helper (2-input when c is None) ---
# cute.arch.fmax is TWO-input only; the 3-input form requires nvvm directly.
# The NVVM API changed between CUDA 12.9 and later releases.
@dsl_user_op
def fmax3(a, b, c=None, *, loc=None, ip=None) -> Float32:
    from cutlass import CUDA_VERSION
    kw = dict(c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
              loc=loc, ip=ip)
    av = Float32(a).ir_value(loc=loc, ip=ip)
    bv = Float32(b).ir_value(loc=loc, ip=ip)
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        return Float32(nvvm.fmax(T.f32(), av, bv, **kw))  # old API: explicit result type
    else:
        return Float32(nvvm.fmax(av, bv, **kw))            # new API: type inferred
```

### Step 2 — 4-chain `add_packed_f32x2` row sum (requires N % 8 == 0)

```python
@cute.jit
def fadd_reduce_sm100(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    # Materialize x into a FRESH rmem block so ptxas assigns it an aligned pair.
    # Without this step the addends inherit their placement from the upstream
    # chain and ptxas inserts a mov.b64 per pair — erasing the savings.
    res = cute.make_rmem_tensor(x.shape, Float32)
    res.store(x)                                       # <- the critical step

    # Chain 0 folds init_val in through the packed add; chains 1-3 seed directly.
    if const_expr(init_val is not None):
        local_sum = [
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1])),
            (res[2], res[3]),
            (res[4], res[5]),
            (res[6], res[7]),
        ]
    else:
        local_sum = [
            (res[0], res[1]),
            (res[2], res[3]),
            (res[4], res[5]),
            (res[6], res[7]),
        ]
    for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], (res[i + 0], res[i + 1]))
        local_sum[1] = cute.arch.add_packed_f32x2(local_sum[1], (res[i + 2], res[i + 3]))
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], (res[i + 4], res[i + 5]))
        local_sum[3] = cute.arch.add_packed_f32x2(local_sum[3], (res[i + 6], res[i + 7]))
    local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
    local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
    local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
    return local_sum[0][0] + local_sum[0][1]
```

### Step 3 — 4-chain 3-input `fmax` row max (requires N % 8 == 0)

```python
@cute.jit
def fmax_reduce_sm100(x: cute.TensorSSA, init_val: float | Float32 | None = None) -> Float32:
    res = cute.make_rmem_tensor(x.shape, Float32)
    res.store(x)                                       # <- same materialization step

    # Chain 0 may fold in init_val via the 3-input fmax.
    local_max_0 = (fmax3(init_val, res[0], res[1])
                   if const_expr(init_val is not None)
                   else fmax3(res[0], res[1]))          # 2-input when no init
    local_max = [
        local_max_0,
        fmax3(res[2], res[3]),
        fmax3(res[4], res[5]),
        fmax3(res[6], res[7]),
    ]
    for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
        local_max[0] = fmax3(local_max[0], res[i],     res[i + 1])
        local_max[1] = fmax3(local_max[1], res[i + 2], res[i + 3])
        local_max[2] = fmax3(local_max[2], res[i + 4], res[i + 5])
        local_max[3] = fmax3(local_max[3], res[i + 6], res[i + 7])
    local_max[0] = fmax3(local_max[0], local_max[1])
    return fmax3(local_max[0], local_max[2], local_max[3])
```

### Step 4 — fallback for N % 8 != 0 or sm90

For fragment sizes that are not a multiple of 8, or on sm90, the materialization
step is the same but the reduction uses 4-chain 2-input `fmax` / scalar add:

```python
@cute.jit
def fmax_reduce_fallback(x: cute.TensorSSA) -> Float32:
    res = cute.make_rmem_tensor(x.shape, Float32)
    res.store(x)
    local_max = [res[0], res[1], res[2], res[3]]
    for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        local_max[0] = cute.arch.fmax(local_max[0], res[i + 0])
        local_max[1] = cute.arch.fmax(local_max[1], res[i + 1])
        local_max[2] = cute.arch.fmax(local_max[2], res[i + 2])
        local_max[3] = cute.arch.fmax(local_max[3], res[i + 3])
    local_max[0] = cute.arch.fmax(local_max[0], local_max[1])
    local_max[2] = cute.arch.fmax(local_max[2], local_max[3])
    return cute.arch.fmax(local_max[0], local_max[2])
```

The `const_expr(arch < 100 or cute.size(x.shape) % 8 != 0)` guard selects
between the two paths at trace time.

## Gotchas

- **Skipping `res.store(x)`.** `make_rmem_tensor` allocates without initializing,
  so without the store you reduce over whatever those registers held — undefined
  values, not `x`. This is a silent correctness bug: the kernel compiles and runs,
  and the wrong result is not reliably a recognizable sentinel, so do not expect a
  zero / `-inf` output to flag it.

- **`-inf` as the `init_val` operand.** Folding a `-inf` row-max seed through the
  3-input form is certified: `fmax3(-inf, a, b) == max(a, b)`, including when two
  of the three operands are `-inf`. NaN operands are *not* covered here — PTX
  `max.f32` returns the non-NaN operand while `torch.maximum` propagates NaN, so
  do not port a NaN-carrying reduction to this path without re-checking.

- **Using `cute.arch.fmax` for the 3-input form.** `cute.arch.fmax` is two-input
  only. The 3-input form requires `nvvm.fmax(..., c=...)`. Passing a third
  positional argument to `cute.arch.fmax` raises at trace time.

- **CUDA version branching for `nvvm.fmax`.** CUDA 12.9 requires the result type
  as the first positional argument (`nvvm.fmax(T.f32(), a, b, c=c)`). Later
  releases infer it automatically (`nvvm.fmax(a, b, c=c)`). Omitting the branch
  causes an NVVM backend compilation failure on the older release.

- **Fragment size not a multiple of 8.** The sm100 fast path — both
  `add_packed_f32x2` and the 3-input fmax loop body — is structured for 8 elements
  per iteration. For N % 8 != 0, fall back to the 4-chain 2-input form (Step 4).

- **Applying this to addends that are already t2r destination registers.** If the
  fragment was loaded from TMEM as the t2r destination, the registers are already
  placed; calling `make_rmem_tensor` + `store` creates a copy and the fresh block
  gets the aligned allocation. The copy has a cost (one load per element), but
  it is what enables the aligned-pair constraint. Measure before assuming the copy
  is free; it is worthwhile when the downstream reduction is long.

- **`add_packed_f32x2` result type.** The return value is a 2-tuple `(Float32,
  Float32)`, not a single `Float32`. The final scalar result is
  `local_sum[0][0] + local_sum[0][1]`.

- **`softmax_segment_ilp` §(d) null.** That null was measured on sm90 with
  the fragment already pre-placed by the t2r/exp2 chain — the packed form had no
  aligned pair to exploit. On sm100 with a fresh `make_rmem_tensor` allocation the
  constraint is satisfied differently; re-measure before concluding the null
  transfers.

## Sources

- PTX ISA, "Floating-Point Instructions: `max`"
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>
- PTX ISA, "Alternate Floating Point Instructions: `add`" (covers `.f32x2` packed form)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#alternate-floating-point-instructions-add>
- CuTe DSL public API symbols: `cute.make_rmem_tensor(shape, dtype)`,
  `cute.TensorSSA.store(src)`, `cute.arch.add_packed_f32x2(a_pair, b_pair)`,
  `cute.arch.fmax(a, b)` (2-input), `cutlass._mlir.dialects.nvvm.fmax(a, b, c=c)`
  (3-input, CUDA version branching required), `cutlass.CUDA_VERSION`,
  `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.range_constexpr`,
  `cutlass.const_expr`.
- GPU certification probe (not included in this release)
  — **status: ✅ GPU-certified.** Final verdict
  `PROBE PASS max_diff=4.053116e-06` (round 2 of this
  certification), all three contracts PASS in the same run:
  `fadd_4chain` 4.053116e-06 — 4-chain `add_packed_f32x2` row sum over a
  materialized 128-element fragment vs `torch.sum` (fp32 reassociation
  tolerance 1e-3); `fmax_4chain` 0.0 — 4-chain 3-input `nvvm.fmax` row max vs
  `torch.max`, exact, which exercises both the 2-input and the 3-input form on
  every chain; `fmax3_3input` 0.0 — `fmax3(a, b, c=c)` vs `max(a, b, c)` across
  256 lanes, one lane per element, 8 deterministic edge triples (all-equal,
  `-inf` as a row-max seed, two `-inf` operands, signed zero, `-inf` in the c
  slot, wide exponent, all-negative, all-zero) plus random triples. NaN is
  deliberately out of scope: `max.f32` returns the non-NaN operand while
  `torch.maximum` propagates NaN, so that case is a semantic difference, not a
  certified result.
