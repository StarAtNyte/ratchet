# ldg_issue_order_pin (pin the whole first batch of `LDG` before the first FP consumer: two pinning methods, dependency token / single-trip loop)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS` (`ldg-pin-toyA-numeric` max_diff=0, `ldg-pin-toyA-sass`).** Certifies only the numerics and the static SASS order,
> **not the gain**: whether to pin must be timed in the same job on your own kernel.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

A thread issues several mutually independent groups of global loads at the start and only then begins floating-point work. To use fewer registers, ptxas may move the later-used groups' `LDG`s down, past the
first FP consumer of an earlier group (even past a `BAR`); the moved loads are then issued only after the earlier group's data has returned and its consumers have run — one more serialized memory round trip.
This entry gives two spellings that pin every `LDG` before the first FP at the SASS level.

| Method | How it constrains ptxas | Cost (probe readings) |
|---|---|---|
| A dependency token | xor all result words of the later-used groups → AND with a kernel parameter that is 0 at runtime → OR into **every** early-consumed word | the early-consumed group must wait for all loads to land (a landing gate); REG 48 → 56 |
| C single-trip loop | write all loads into **one** inline PTX block, wrapped in a backward-jump loop that runs only once | ptxas duplicates the loop, the static `LDG` count triples (only one copy runs at runtime); REG 48 → 56 |

## When needed (trigger)

- **Judge by SASS, not by the source**: in the linear SASS (`cuobjdump -sass`), some `LDG` has an index greater than that of "the first `FADD/FMUL/FFMA/MUFU/HFMA2` after the first `LDG`"
  (FP instructions placed before all `LDG`s are address / initialization arithmetic and do not count).
- The loads are already all written at the very top of the source, yet in SASS some of them land in the middle of the FP chain or around a `BAR`.
- **Conversely, if no `LDG` sinks in SASS, do not pin**: any pinning method is then pure cost. Recount every time the arithmetic after the loads changes.

## Minimal runnable example

```cpp
__device__ __forceinline__ void ld4(const char* p, uint32_t* w) {      // not volatile: lets ptxas schedule freely
  asm("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "l"(p));
}

// ---- A dependency token: Z is a kernel parameter that is 0 at runtime (the host passes 0); the compiler cannot prove it is 0
uint32_t w[12 * 4];
for (int g = 0; g < 12; ++g) ld4(base + g * GSTR, &w[4 * g]);
uint32_t t = 0;
for (int i = 16; i < 48; ++i) t ^= w[i];       // all result words of the later-used groups (4..11)
t &= (uint32_t)Z;                              // always 0, but ptxas can only wait for all of them to land
for (int i = 0; i < 16; ++i) w[i] |= t;        // OR into every early-consumed word (integer domain: bit pattern unchanged)
/* ... FP work of the early-consumed groups, block reduction (__syncthreads), then use the later-used groups ... */

// ---- C single-trip loop: all loads in one asm block; c starts at Z, after subtracting 1 setp.ge is false, the body runs only once
asm volatile("{\n\t.reg .pred q;\n\t.reg .b32 c;\n\tmov.b32 c, %49;\n"
             "PIN_LOOP:\n\t"
             "ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%48+0];\n\t"
             "ld.global.nc.v4.b32 {%4, %5, %6, %7}, [%48+16777216];\n\t"
             /* ... the other groups: outputs %8 .. %47, address = %48 + immediate offset ... */
             "sub.s32 c, c, 1;\n\tsetp.ge.s32 q, c, 0;\n\t@q bra PIN_LOOP;\n\t}"
             : /* 48 x "=r"(w[i]) */ : "l"(base), "r"(Z) : "memory");
```

## Gotchas

1. **Z must be a runtime kernel parameter.** Written as a constant 0, `x * 0` or any expression the compiler can prove is 0, the token is folded, the single-trip loop is straightened, and the SASS reverts to the original.
2. **The token must go into every early-consumed word**, not just the first: a dependency constrains only the instructions that use it.
3. **The token is a landing gate**: every consumer of the early-consumed groups waits for all later-used groups to land; if you want "compute whatever arrives first", use C.
4. **Keep the token in the integer domain** (bit-pattern OR): a floating-point `x + 0*t` contaminates the result on inf / NaN and may also be folded.
5. **In C all loads must be in the same asm block**; the label is written inside `{ }`. Whether inlining the same labelled asm block several times in one kernel causes a name clash has not been tested.
6. **C multiplies the static `LDG` count** (probe: 12 → 36, plus some extra `BRA`); accept by "the count is an integer multiple of the original, each carrying `.CONSTANT`".
7. **The register cost is real**: once pinned, the later-used groups' result words live from entry until their consumers; after the change, first confirm `LDL/STL` is still 0.
8. **Probe readings** (12 groups of `ld.global.nc.v4.b32` per thread, the first 4 consumed before the block reduction; identical readings for `__launch_bounds__(256, 1)` and `(256, 4)`):
   unpinned, **4/12** `LDG`s sink below the first FP (REG 48); token version 0/12 (REG 56); single-trip-loop version 0/36 (REG 56); all three have `LDL/STL` 0,
   numerics bitwise identical to the host reference.

## Sources

- PTX ISA: `ld.global.nc`, the `.b32` forms of `xor` / `and` / `or`, `setp` / `bra` / predicated execution, `.reg` and labels inside `{ }` blocks:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/>; inline PTX constraints: <https://docs.nvidia.com/cuda/inline-ptx-assembly/>
- CUDA Binary Utilities (`cuobjdump -sass` / `-res-usage`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probe (toy A) — `PROBE PASS max_diff=0` (numerics) and
  `PROBE PASS max_diff=n/a` (`ldg-pin-toyA-sass`: under both launch bounds, "unpinned sinks, token and single-trip loop do not").
