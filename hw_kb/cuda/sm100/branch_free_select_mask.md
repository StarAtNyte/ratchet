# branch_free_select_mask (per-column valid bit: write a select, do not turn the bit into a float for multiply-add)

> **Certification source: B200 GPU-certified in CUDA mode (all 6 contracts PROBE PASS; the probe is not included in this release).**
> The SASS shape is certified for all three downstreams; the timing is a gain only on the FMA-pipe-bound downstream, and when the ALU pipe is the bound the float form is not slower (see gotcha 3).
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

One valid bit per column (mask / padding flag) decides whether the column takes the data value or a parked value. Written as a select, the SASS is
`ISETP` + `FSEL` per column; written as "bit → float → `x * f + (1 - f) * PARK`", each column becomes `I2FP` + `FADD` + `FMUL` + `FFMA`,
and everything extra is FMA-pipe instructions.

## When needed (trigger)

- SASS has one `I2F*` (`I2FP.F32.U32`) per column, followed by `FMUL` / `FFMA`; the source has `(float)bit` or an equivalent conversion.
- The segment is FMA-pipe / issue bound (FFMA / FMUL / FADD dominate).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// b: this column's flag (0 / 1), park: the parked value. Taken from this entry's probe.
y = b ? x : park;                       // C++ ternary -> ISETP + FSEL, no branch

// Equivalent inline PTX (the probe measured the same SASS histogram as the ternary; no need to hand-write it):
asm("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %1, 0;\n\tselp.f32 %0, %2, %3, p;\n\t}"
    : "=f"(y) : "r"(b), "f"(x), "f"(park));

// Anti-pattern:
const float f = (float)b;               // I2FP.F32.U32
y = fmaf(x, f, (1.f - f) * park);       // FADD + FMUL + FFMA
```

## Gotchas

1. **A C++ ternary is already a branch-free select**: in the probe the ternary variant has `FSEL=16`, `I2F*=0`, `BSSY=0` per 16 columns, a histogram identical to the inline
   `setp` + `selp` variant — no asm spelling is needed under CUDA.
2. **The float form costs one extra `I2FP` per column plus several FMA-pipe instructions** (probe: for 16 columns, 16 extra `I2FP` and 48
   FMUL+FFMA+FADD; 182 instructions in total vs 151).
3. **Whether it is slower depends on which pipe is the bottleneck.** The select's `ISETP` / `FSEL` go to the ALU pipe; the float form moves the work to the FMA pipe.
   Probe readings: with an FFMA-chain downstream (FMA-pipe bound) t(float)/t(select) = 1.203; with a row max + `FMNMX`
   clamp downstream (ALU-pipe bound) = 0.965 (the float form is not slower, even slightly faster); with an `ex2` downstream (MUFU bound) = 1.011 (hidden by MUFU).
   The FMA-pipe-bound downstream is still 1.182 with only 1 warp per SMSP. The pipe attribution is an interpretation of these readings; this entry did not certify it separately with NCU per-pipe counters.
   Write the select by default; only when SASS shows the ALU pipe is the bottleneck is there a reason to consider another form, and it must be measured in the same job.
4. Use a finite large negative number as the parked value (e.g. `-1e30f`), not `-inf`: if something like `x - max` follows, `(-inf) - (-inf) = NaN`.
5. Judge the form by the `FSEL` count and the number of FMA-pipe instructions, not just `I2F`: if the compiler can see that the flag only takes 0 / 1, the shape of the conversion instruction
   may change, but the extra multiply-adds do not go away. The probe extracts the flag byte from a packed word with `__byte_perm`, so the compiler only sees "some byte".

## Sources

- PTX ISA: *Comparison and Selection Instructions: `setp` / `selp`*;
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certified by the probe (2 CTAs × 256 threads per SM,
  two interleaved timing rounds, the two rounds differ by <3%) —
  `PROBE PASS max_diff=2.749e-07 branch_free_select_mask/numeric`;
  `PROBE PASS max_diff=n/a branch_free_select_mask/sel_cpp_fsel fsel=[16, 16, 16] i2f=[0, 0, 0] bssy=[0, 0, 0]`;
  `PROBE PASS max_diff=n/a branch_free_select_mask/sel_asm_fsel fsel=[16, 16, 16] i2f=[0, 0, 0]`;
  `PROBE PASS max_diff=n/a branch_free_select_mask/i2f_extra_fma_pipe extra_fma_pipe=[48, 48, 48] i2f_any=[16, 16, 16]`;
  `PROBE PASS max_diff=n/a branch_free_select_mask/form_time_fma_bound ratio=1.2032`;
  `PROBE PASS max_diff=n/a branch_free_select_mask/form_time_alu_bound ratio=0.9648`.
