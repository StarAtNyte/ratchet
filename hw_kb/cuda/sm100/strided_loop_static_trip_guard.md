# strided_loop_static_trip_guard ("go round a few more times if there are not enough threads" loops: a runtime upper bound leaves a backward `BRA` and `#pragma unroll` has no effect; a compile-time trip count + runtime guard gives straight-line SASS)

> **Certification source: B200 GPU-certified in CUDA mode (certified by a GPU probe (the probe is not included in this release); all 7 contracts PROBE PASS).**
> Certification covers: four spellings of the same computation (runtime-bounded loop, plus `#pragma unroll`, compile-time trip count + guard, plus PTX predicated loads) bytewise identical over two template instances
> and a sweep of runtime n (out-of-range positions prefilled with NaN to check they are not written); SASS census of backward jumps / total instructions / the position of the first `LDG`.
> **Uncertified**: any timing gain (the probe does no timing; in a whole kernel this change was bundled with other prologue changes, see "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. A loop with a compile-time constant trip count gets fully unrolled, and to keep it rolled you write `#pragma unroll 1`, see
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md); this entry is the opposite direction: you want it unrolled but the upper bound is a runtime value.

## Purpose

A per-thread strided loop `for (o = threadIdx.x; o < n; o += kThreads)` (cross-warp reduction writeback, epilogue writeback, "go round once more when there are more outputs than threads") has
a runtime upper bound; `#pragma unroll` cannot unroll it, and SASS keeps a backward-jump loop. If n has a compile-time upper bound (a template parameter / `constexpr` tile count), take the trip count as
the compile-time `ceil(MAX / kThreads)` and guard the loop body with the runtime n: a guard produces only forward branches or predicates, never a backward jump.

## When needed (trigger)

- SASS has a backward `BRA` (target address < its own address; not counting the self-jump after `EXIT` at the end of the kernel), while you thought this part of the source was already unrolled.
- The source has a loop whose stride is the thread count and whose upper bound contains a runtime value, and the number of outputs has a compile-time upper bound.
- Only the SASS shape is certified: whether it is worth anything must be timed on your own kernel (the cost of code size and the backward jump in a short kernel was not measured on the probe). A change of the same kind was committed in a whole kernel together with other prologue changes,
  gate 0.009 → 0.0088 ms, not attributable separately.

**Scope / anti-trigger**: this entry targets short per-thread strided loops for memory access and reduction writeback (the body is load / store / reduction, not a long string of compute).
A register-fed `mma.sync` HMMA run goes the other way: when the operands are loop-invariant, a straight-line unrolled HMMA run has register moves before every HMMA,
and putting it into a rolled loop with a runtime trip count actually zeroes the moves ([`mma_sync_hmma_run_register_feed`](mma_sync_hmma_run_register_feed.md) fact 3) —
there, do not unroll the HMMA segment into straight-line code just to "remove the back jump". Both certify only the SASS shape, and neither whole-kernel reading has an attributable gain.

## Minimal runnable example

```cpp
// kMaxOut: compile-time upper bound on the number of outputs (from a template parameter / constexpr tile count); n_out: runtime value, <= kMaxOut
template <int kMaxOut, int kThreads>
__device__ __forceinline__ void strided_outputs(int n_out) {
    constexpr int kTrips = (kMaxOut + kThreads - 1) / kThreads;     // compile time
#pragma unroll
    for (int it = 0; it < kTrips; ++it) {
        const int o = threadIdx.x + it * kThreads;
        if (o < n_out) {
            // ... process output o (inner fixed-length loops likewise #pragma unroll + compile-time length) ...
        }
    }
}

// Counter-example (numerics right, SASS has a backward jump; the same with #pragma unroll, since the upper bound is a runtime value):
//   for (int o = threadIdx.x; o < n_out; o += kThreads) { ... }
```

Acceptance (SASS): `cuobjdump -sass <module>.so`, count backward `BRA`s in the target function (parsed target address < this instruction's address, excluding the self-jump at the end); it should be 0; record the total instruction count as well.

## Measured readings (probe, two template instances: small / large tile count; the large instance has more outputs than threads and needs two rounds)

| Spelling | Small instance: total / backward jumps / `BRA` | Large instance: total / backward jumps / `BRA` |
|---|---|---|
| runtime-bounded loop | 216 / 1 / 3 | 248 / 1 / 3 |
| runtime-bounded loop + `#pragma unroll` | 216 / 1 / 3 (identical to the row above) | 248 / 1 / 3 (identical to the row above) |
| compile-time trip count + guard | **120 / 0 / 1** | **192 / 0 / 2** |
| compile-time trip count + guard, with the guarded load changed to a PTX `@p ld` predicate | 120 / 0 / 1 | 192 / 0 / 2 |

- Adding `#pragma unroll` to a runtime upper bound is **completely ineffective** here; the SASS does not differ by a single instruction.
- With a compile-time trip count the backward jump disappears and the total drops by 44% / 23%.
- The first `LDG` did not move earlier because of it: small instance 29 → 30, large instance 35 → 41 (instruction index).
- Changing the C++ `if (valid) v = ld(...); else v = 0;` (`ld` being `asm volatile`) into "zero first, then `@p ld`": instruction and `BRA` counts unchanged, and the first `LDG` actually moves later
  (30 → 37, 41 → 51). This step gains nothing in this probe.
- An empty kernel has 24 instructions in total, the first three `LDC R1, c[0x0][0x37c]`, `LDCU.64`, `ISETP`: the first instructions at entry are the ABI prologue, which the source cannot move.

## Gotchas

1. **`#pragma unroll` is only a request.** When the upper bound contains a runtime value it may be ignored or only partially applied; whether unrolling is complete can only be judged by whether SASS has a backward jump.
2. **The guard must not be removed.** The compile-time trip count is an upper bound; a round with `o >= n_out` must read and write nothing. The probe prefills the output with NaN to check that out-of-range positions really are not written.
3. **Take a tight trip bound.** The looser the bound, the more unrolled copies and the larger the code; let the template parameter (tile count) decide the bound directly, so each instance unrolls only the rounds it can use.
4. **Straight-lining does not move the first memory accesses earlier.** To make `LDG` appear early, separately adjust the order of the address arithmetic at entry, and verify in SASS.
5. **It does not change the numerics.** The four spellings are bytewise identical, so `max_diff` cannot detect shape problems; accept by SASS census, and the gain by timing.

## Sources

- CUDA C++ Programming Guide: `#pragma unroll` — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#pragma-unroll
- PTX ISA: predicated execution (`@p`), `ld.global.nc` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Binary Utilities: `cuobjdump -sass` — https://docs.nvidia.com/cuda/cuda-binary-utilities/
- GPU certification probe: `PROBE PASS max_diff=0` (num_m0 / num_m1 / num_m2 / num_m3),
  `PROBE PASS max_diff=n/a` (straight_m2 / straight_m3 / detector_m0); the SASS counts are same-job fact lines.
