# ldg_prologue_mov_cctl (prologue of a register-fed MMA: the MOV-zero string that predicated zero-fill loads leave before the first HMMA, and removing it entirely with address clamping)

> **Certification source: CUDA-mode GPU probe (toy B; not included in this release),
> `PROBE PASS` (`ldg-prologue-zero-fill`, `ldg-pin-toyB-numeric` max_diff=0).** Certifies only the SASS shape and numerics, **not the gain**
> (in the probe both spellings time at 9.38 µs per single wave, dominated by launch overhead, which says nothing). The prefetch (`prefetch.global.L2` → `CCTL`) placement half is
> uncertified on the CUDA side and not covered by this entry.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

In the prologue of "global → registers → `mma.sync`" (kernel entry up to the first `HMMA`, called segment P below), the idiomatic predicated-load spelling "first `mov.b32 r, 0`, then
`@p ld`" costs 8 zero writes per 256-bit load (`MOV R, RZ` / `CS2R R, SRZ` and the like), and since they write the same registers as that load (WAW) they can only be placed before
the LDG, all crowded into segment P taking issue slots. When an out-of-bounds row affects only its own part of the output, and that part is discarded by the store guard anyway, switch to **address clamping**:
the out-of-bounds lane reads a legal row (clamped to M−1), the load carries no predicate, and the zero writes disappear entirely.

## When needed (trigger)

- Segment P's stalls are mostly `long_scoreboard` / `lg_throttle`, and segment P has a long string of `MOV R, RZ` / `CS2R R, SRZ`, numbering predicated loads × 8.
- Out-of-bounds rows / columns are discarded by the store guard on the output side (the precondition for clamping to be correct, see gotcha 1).

## Minimal runnable example

```cpp
// predicate + zero fill (8 zero writes each in segment P)
asm("{\n\t.reg .pred p;\n\tsetp.lt.s32 p, %8, %9;\n\t"
    "mov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\tmov.b32 %2, 0;\n\tmov.b32 %3, 0;\n\t"
    "mov.b32 %4, 0;\n\tmov.b32 %5, 0;\n\tmov.b32 %6, 0;\n\tmov.b32 %7, 0;\n\t"
    "@p ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%10];\n\t}"
    : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
    : "r"(row), "r"(M), "l"(p));

// address clamping (zero writes: 0): out-of-bounds lanes read row M-1, the store side keeps the if (m < M) guard
int rc = row - (row >= M) * (row - (M - 1));             // pure arithmetic, no branch
asm("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
    : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
    : "l"(buf + (size_t)rc * ROW_BYTES));
```

Acceptance (SASS; numerics cannot show it): `cuobjdump -sass`, take everything before the first `HMMA` as segment P, count `MOV R, RZ` / `MOV R, 0x0` / `IMAD.MOV.U32 R, RZ, RZ, RZ` (1 each)
and `CS2R R, SRZ` (2 each). The accumulator's initial value must come from a runtime value (e.g. a `float bias` parameter), otherwise its zeroing also gets mixed into this count.

## Gotchas

1. **Clamping is correct only when out-of-bounds results are discarded**: the out-of-bounds lane reads real data and the product is non-zero garbage; once the store side's `m < M` guard is removed, wrong values are written.
   When out-of-bounds elements must take part in the computation **as real zeros** (e.g. out of bounds along the K reduction direction), clamping cannot be used.
2. **The zero writes cannot move after the load**: `mov 0` and `@p ld` write the same registers (WAW).
3. **Probe readings** (12 `ld.global.nc.v8.b32` per lane feeding 24 `mma.sync.m16n8k16` directly, `__launch_bounds__(256, 1)`): the zero-fill version has
   **96** zero writes in segment P (= 12 × 8), the first HMMA is SASS instruction #104; the clamped version has **0** zero writes in segment P, the first HMMA is instruction #99; both have `LDL/STL` 0,
   REG 114 / 110, and numerics bitwise identical to the host's simulation of the mma fragments.

## Sources

- PTX ISA: vector forms of `ld.global.nc` (`.v8.b32`), predicated execution, the operand fragment layout of `mma.sync.aligned.m16n8k16`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Binary Utilities (`cuobjdump -sass` / `-res-usage`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probe (`k_b_zf` / `k_b_clamp`) — `PROBE PASS max_diff=n/a`
  (`ldg-prologue-zero-fill`) and `PROBE PASS max_diff=0` (`ldg-pin-toyB-numeric`).
