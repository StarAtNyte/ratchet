# lowtier_role_tmem_readout_spill_ladder (TMEM readback in a low-tier `setmaxnreg.dec` role: the 32 outputs of `.x32` + `wait::ld` in one asm are all live at once; a "two-pass read" re-reads TMEM to halve the peak; spilling is decided by the tier, count `LDL` / `STL` per role)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the probe is not included in this release; verdicts and SASS / ptxas readings under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; **the readings differ from the DSL entry**: on this probe's CUDA spelling the "two-pass read" removed the spill at tier 56
> (the DSL entry read the two-pass read spilling more on its kernel) — narrowing the readback remains a candidate to be verified by census each time, not a fixed rule.
> The CuTe DSL tree's `tmem_readout_segment_live_set` (the same mechanism at a high tier with the DSL copy-atom spelling) is merged into this entry in the CUDA tree.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Architecture scope: sm100.

## Purpose

One `tcgen05.ld.sync.aligned.32x32b.xN.b32` instruction writes **N 32-bit registers**; written in the same asm as `tcgen05.wait::ld`, these N outputs
are **all live at once** from the end of the asm until the last consumer — ptxas cannot split an asm's output set. In a role that has `setmaxnreg.dec`'d to a very low tier:

```
peak(segment) = N (fragment in flight / just read back)
              + values resident across rounds (loop-invariant per-thread scalars, addresses, barrier bases)
              + addresses and loop counters                    <= tier
```

Beyond that ptxas reports no error and moves things into local memory: `STL` + `LDL` at the segment head / in the loop, and ptxas `-v` reports `bytes stack frame` / `spill stores`.

In a **reduce first, then element-wise** segment (the row amax / row sum must be done before element-wise normalization), "read a block, use a block" does not work — the whole row must live across the reduction. The fix uses the fact that TMEM can be re-read:

- **First pass**: `.x16` read one half → fold into the reduction → dead; read the other half → fold in → dead. Peak 16 + 1.
- **Second pass**: **re-read** the same half `.x16` from TMEM → element-wise → pack → write out → dead. Peak 16 + reduction values.

The cost is one extra `tcgen05.ld.x16` + `wait::ld` per half (on chip), traded for the per-round local-memory round trips.

## When needed (trigger)

- A `setmaxnreg.dec` role has a "TMEM readback → row reduction → element-wise → narrow-type write-out" segment, ptxas `-v` reports non-zero `spill stores`, and SASS has `STL` / `LDL`
  in this role's range (after `setmaxnreg`);
- NCU `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st` is non-zero, `stall_long_scoreboard` dominates in this role, and the downstream MMA chain waiting on it idles.

## Minimal runnable example

384 threads = 3 warpgroups; warpgroup 0 drops to `TIER`, the others raise their tier; each round warpgroup 0 reads its own TMEM row (32 f32), computes the row amax,
`y = x * (1/amax) * res[j % 14]`, packs bf16x2 and writes out; 14 per-thread resident scalars live across rounds. The spelling matches the certification probe:

```cpp
__device__ inline void ld_x32_wait(uint32_t taddr, uint32_t* v) {     // one asm: 32 outputs live at once
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, ..., %31}, [%32];\n\t"
               "tcgen05.wait::ld.sync.aligned;" : /* 32 x "=r" */ : "r"(taddr) : "memory");
}
__device__ inline void ld_x16_wait(uint32_t taddr, uint32_t* v);      // same as above, .x16

__global__ void __launch_bounds__(384, 1) k(...) {
  // ... TMEM allocation, warpgroup 0 fills rows with tcgen05.st
  if (threadIdx.x / 128 == 0) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(TIER));   // executed by the whole warpgroup together
    float res[14]; /* resident */
    for (int it = 0; it < ROUNDS; ++it) {
      // two-pass read:
      float amax = 0.f;
      for (int h = 0; h < 2; ++h) { uint32_t x[16]; ld_x16_wait(trow + 16 * h, x);
        for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(__uint_as_float(x[j]))); }
      const float inv = __fdiv_rn(1.f, amax);
      for (int h = 0; h < 2; ++h) { uint32_t x[16]; ld_x16_wait(trow + 16 * h, x);       // re-read the same half
        for (int j = 0; j < 8; ++j) out[8 * h + j] = pack_bf16x2(x0 * inv * res[..], x1 * inv * res[..]); }
    }
  } else {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(INC));    // INC: (384*168 - 128*TIER) / 256 rounded down to a multiple of 8 (pool = threads x entry cap, not 65536)
  }
}
```

Acceptance: count this kernel's `STL` / `LDL` with `cuobjdump -sass` (in this probe only the low-tier role has work that can spill, so the whole-kernel count is that role's count),
ptxas `-v`'s `bytes stack frame`; numerics bitwise against the host reference.

## Gotchas

1. **`setmaxnreg` is a warpgroup-level `.sync.aligned`**: all 4 warps must execute the same value; after `.dec` the role's register cap is the tier, ptxas allocates this code against it,
   and anything beyond spills — no compile error. The kernel's `Used N registers` is the number at launch (168 in this probe), not after the tier drop.
2. **Splitting into two `.x16` but keeping both halves until normalization leaves the peak unchanged**: this probe's "split but keep both" variant spills just like `.x32` (see Sources). What saves registers is letting the first half die first.
3. **The two-pass read needs the second pass to really re-read TMEM**: the re-reading `tcgen05.ld` reads the same columns as the first pass, and nobody may overwrite those columns in between (readback segments usually come after the accumulator is complete).
4. **Raising the tier is a separate lever**: the same source with only the tier changed from 56 to 104 brings the spills of both the `.x32` and the split-read spellings to zero. Tiers come from the warp-specialization register budget;
   raising one role means lowering another, so do the total accounting: `128 × sum of the warpgroup tiers ≤ threads × entry cap` (this probe 384 × 168 = 64512, **not 65536**;
   both settings 56 / 224 / 224 and 104 / 200 / 200 equal 64512 exactly). Beyond it `.inc` never obtains its registers and the kernel hangs without an error
   ([`warp_specialization`](warp_specialization.md) gotcha 2 has a measured hang with sum = 65536).
5. **Narrowing / re-reading does not guarantee a monotone gain**: ptxas's actual allocation does not follow the live-value count above exactly, and the same mechanism has read in the opposite direction on other spellings; recount `STL` / `LDL` after every spelling change.

## Sources

- PTX ISA: `setmaxnreg` (`.inc` / `.dec`, `.sync.aligned`, warpgroup level), `tcgen05.ld` / `tcgen05.wait::ld` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide: `__launch_bounds__`; the `bytes stack frame` / `spill stores` / `spill loads` output of ptxas `-v` (`-Xptxas=-v`).
- GPU-certification probe (tiers 56 / 104 × `.x32` in one asm / split read keeping both / two-pass read), verdict
  `PROBE PASS`: `G1-tier56-var{0,1,2}-exact`, `G1-tier104-var{0,1,2}-exact` (8192 packed words, 0 mismatches). SASS / ptxas:
  at tier 56 the `.x32`-in-one-asm and "split but keep both" spellings each have `LDL` 7 / `STL` 7 (ptxas: 24 bytes stack frame, 12 bytes spill stores / loads),
  the two-pass read `LDL` = `STL` = 0; at tier 104 all three spellings are 0.
