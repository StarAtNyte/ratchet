# rmem_array_runtime_index_local_frame (a thread-private array with runtime indices goes into the local stack frame: how to read it and two fixes)

> **Certification source: B200 GPU-certified in CUDA mode (all 3 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: an array written and read with runtime indices lands in local memory (STACK, `STL` / `LDL`) while registers are far from full; both fixes, a scalar cursor and "fully unroll both sides + guard", bring STACK to zero;
> a 16-element array read at only one runtime index is kept in registers by ptxas (a fact reading). **Uncertified**: the impact of the local traffic on whole-kernel time.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

A thread-private array (`T a[K]`) can live entirely in registers only if **every index is a compile-time constant** — registers cannot be addressed by a runtime value.
If even one write or read uses a runtime index (that the compiler cannot reduce to a constant), the whole array goes into the **stack frame** in local memory:
`cuobjdump -res-usage` shows `STACK > 0`, SASS shows `STL` / `LDL`, and NCU's local memory sectors are non-zero.
This is **not a spill caused by register pressure**: more registers will not bring it back, and tuning `__launch_bounds__` / `-maxrregcount` does not help.

## When needed (trigger)

- In `cuobjdump -res-usage` some kernel has `STACK > 0` while `REG` is far below the cap (probe: REG 18, STACK 40).
- SASS has `STL` / `LDL` near a small table that is "appended to conditionally, then read back by count" (written by cursor `n++`, read by `for (it < n)`).
- After relaxing `__launch_bounds__` / raising the register cap, `STACK` does not change at all.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
#define L 16
struct Rec { int a, b, c; };

// Lesion (T): write at runtime slot n, read at runtime slot it -> the whole table goes into the stack frame
//   probe: STACK 40, REG 18, 47 STL, 3 LDL
Rec tab[3]; int n = 0;
for (int i = 0; i < L && n < 3; ++i) {
  int v = in[g * L + i];
  if (v > thr) { tab[n].a = i; tab[n].b = v; tab[n].c = v ^ i; ++n; }
}
int acc = 0;
for (int it = 0; it < n; ++it) acc += tab[it].a * 3 + tab[it].b * 5 + tab[it].c;

// Fix K: a scalar cursor — each record is consumed the moment it is produced, so no table is needed at all (STACK 0, REG 14)
int n = 0, acc = 0;
for (int i = 0; i < L && n < 3; ++i) {
  int v = in[g * L + i];
  if (v > thr) { acc += i * 3 + v * 5 + (v ^ i); ++n; }
}

// Fix U: keep the table, but fully unroll both the write and the read side to a static upper bound and pick the slot with a guard -> every index is a constant (STACK 0, REG 54)
Rec tab[3] = {};
int n = 0;
#pragma unroll
for (int i = 0; i < L; ++i) {
  int v = in[g * L + i];
  bool take = (v > thr) && (n < 3);
#pragma unroll
  for (int s = 0; s < 3; ++s)
    if (take && s == n) { tab[s].a = i; tab[s].b = v; tab[s].c = v ^ i; }
  n += take ? 1 : 0;
}
int acc = 0;
#pragma unroll
for (int s = 0; s < 3; ++s)
  if (s < n) acc += tab[s].a * 3 + tab[s].b * 5 + tab[s].c;
```

Commands to read it: `cuobjdump -res-usage <module.so>` (look at `REG` / `STACK`) and `cuobjdump -sass <module.so> | grep -cE 'LDL|STL'`.

## Gotchas

1. **`STACK > 0` does not mean too few registers.** In the probe the lesion version uses only 18 registers yet has a 40-byte stack frame — it is decided by the array's **placement**.
   Criterion: `REG` far below the cap with a non-zero `STACK` — look for an array with runtime indices first; do not tune the register cap.
2. **Fix K is the cheapest** (REG 14, no table): if you can consume as you produce, do not build a table.
3. **Fix U must unroll both sides.** Both the write side `tab[s]` and the read side `tab[s]` must be inside constant `#pragma unroll` loops, selecting the slot with a guard (`s == n` / `s < n`);
   the cost is registers (probe: 54, every field of the table resident) plus the guards' predicate instructions. Unrolling only one side is outside this entry's certification.
4. **A small array read at only one runtime index does not necessarily go into the stack frame.** In the probe, 16 floats all written at constant indices and read at only one place by `(tid + u) & 15`
   give STACK 0, 0 `LDL` / `STL`, REG 31 — ptxas turned that single read into a selection over registers. Do not conclude from the source shape; **go by `-res-usage` and SASS**.
5. **Whether PTX has `ld.local` is an unreliable criterion**: the final placement is decided by ptxas (gotcha 4 is an example); read SASS and `STACK`.

## Sources

- CUDA C++ Best Practices Guide: *Local Memory* (the compiler places arrays whose indices it cannot determine into local memory) —
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUDA Binary Utilities: `cuobjdump -res-usage` / `-sass` — https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 local_frame/numeric` (the three spellings agree with the host value by value, the peek values are right);
  `PROBE PASS max_diff=n/a local_frame/table_local STACK=40 REG=18 LDL=3 STL=47`;
  `PROBE PASS max_diff=n/a local_frame/fixes_clean STACK={cursor 0, unroll 0} REG={cursor 14, unroll 54}`.
  Fact line: `peek_16 STACK=0 REG=31 LDL=0 STL=0`.
