# ldg_issue_pin_structural_fence (a batch of global loads feeding `mma.sync` straight from registers: ptxas sinks part of the LDGs below the first HMMA; two structural fences that can pin them, and the spill when the pinned active set exceeds the cap; for the next batch in the k-loop steady state, measure the lead first instead of hard-pinning it into a burst)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS` (`ldg-pin-b16-numeric` max_diff=0, `ldg-pin-b16-sass`); the 12-group control is
> toy B of a separate probe.** Certifies only the SASS shape and numerics, **not the gain** —
> once the probe's shape is pinned, its active set necessarily exceeds the cap, so it spills and is slower (see gotcha 3); for the gain of pinning when the results fit, there is no CUDA-side reading.
> §"k-loop steady state" and gotchas 6–9: a second CUDA probe,
> `PROBE PASS max_diff=0.0`; likewise certifies only the SASS shape, the lead and numerics.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the shape where load results first go through FP arithmetic see
> [`ldg_issue_order_pin`](ldg_issue_order_pin.md) (dependency token / single-trip loop). For the shape of the HMMA run itself in this class of kernel (same-accumulator distance, `MOV` relays for A quads,
> straight-line vs rolled) see [`mma_sync_hmma_run_register_feed`](mma_sync_hmma_run_register_feed.md).

## Purpose

In a "global → registers → `mma.sync`" feed that bypasses smem, the results of a batch of `ld.global{.nc}.v8.b32` (`LDG.*.256`) are used directly as MMA operand registers.
The source issues the whole batch before consuming it, hoping to hide the load latency up front; what really decides the issue position is ptxas: by its own pressure heuristics it sinks part of the LDGs down to just before their first use
(shortening live ranges); whether it sinks cannot be judged just from how far REG is from the cap (see "When needed"). What can block the sinking is **structure** that ptxas does not cross:

- **A: `membar.cta` after the group** (loads use weak `ld.global`, without `.nc`) — a real memory fence; loads in the group cannot move past it;
- **B: the whole group in one asm, wrapped in a runtime-predicated branch inside the asm** — a basic-block boundary; ptxas does not move loads out of the block (the loads can keep `.nc`).

## When needed (trigger)

- In SASS (after splitting by path) some `LDG.*.256` sit after that path's first `HMMA`, while in the source they are written before consumption.
- **Do not judge whether it sinks by how far REG is from the cap; count the SASS first.** The two shapes of this entry's probe: 12 load groups, `__launch_bounds__(256, 1)`,
  REG 114, the bare spelling sinks 0; 16 groups, `(256, 2)` (cap 128), REG 126, 2/16 sink — the two changed group count, minBlocks and grid at once,
  so the difference cannot be attributed to the cap. Another shape (load results first go through FP arithmetic, see [`ldg_issue_order_pin`](ldg_issue_order_pin.md)) already sinks 4/12 at REG 48
  with cap 255.

## Minimal runnable example

```cpp
// A: weak load + membar.cta after the group
__device__ __forceinline__ void ld8weak(const char* p, uint32_t* w) {
  asm("ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7]) : "l"(p));
}
for (int k = 0; k < NCH; ++k) ld8weak(base + k * CHUNK_B, &w[8 * k]);
asm volatile("membar.cta;" ::: "memory");
// ... mma.sync over NCH chunks ...

// B: the whole group in one asm; guard is a kernel parameter that is true at runtime (the compiler cannot prove it always true)
asm("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %129, 0;\n\t@!q bra SKIP_GRP;\n\t"
    "ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%128+0];\n\t"
    "ld.global.nc.v8.b32 {%8, %9, %10, %11, %12, %13, %14, %15}, [%128+1024];\n\t"
    /* ... 16 in total, outputs %0 .. %127, address = %128 + immediate ... */
    "SKIP_GRP:\n\t}"
    : /* 128 x "=r"(w[i]) */ : "l"(base), "r"(guard));
```

## k-loop steady state: the batch for the next k-block (measure the lead first; do not hard-pin it into a burst)

The two fences above deal with the batch "before the first HMMA". In a k-loop each k-block also issues a batch of `LDG.*.256` for the next k-block;
in SASS they are often scattered one by one among the current k-block's HMMAs ("trickle": one LDG every few HMMAs), which looks like "the next batch leaves too late",
so one wants to merge them into "one or two bursts right after the current k-block's first HMMA". **Measure the lead first, then decide whether to touch it**:

1. Take each `LDG`'s destination registers: in a shape like `LDG.E.….256 Ra, Rb, desc[..][..]`, the operands before `[` are all destinations; `.256` is 8 32-bit registers,
   and when spread over two operands each has 4 (both groups must be matched, not 8 consecutive from `Ra`);
2. walk forward through the SASS to the first instruction whose **source operands** contain one of them: an HMMA's A is 4 consecutive, B 2 consecutive, C 4 consecutive; a
   `MOV` / `IMAD.MOV` assembling load results into an A quad also counts as a reader;
3. the number of HMMAs between the two is that load's lead. Count only LDGs issued after the first HMMA (the steady-state batch), and look at the minimum and the mean.

**Criterion**: when the minimum steady-state lead is ≥ one k-block's worth of HMMAs, the issue position is no longer the bottleneck; the `long_scoreboard` on the HMMAs then means the memory latency
itself exceeds the duration of that stretch of HMMAs, and changing the issue order cannot move it.

Probe readings (register-fed `mma.sync.m16n8k16`, k-loop fully unrolled, cap 128 registers per thread; two settings of loads per k-block,
where in the larger one "current batch + the full next batch + accumulators" exceeds the cap; the lead is in units of "one k-block's HMMA count"):

| Spelling (all write the next batch after the current k-block's first HMMA) | Smaller LDL / STL | Smaller min lead | Smaller single launch | Larger LDL / STL | Larger min lead | Larger single launch |
|---|---|---|---|---|---|---|
| plain spelling (`ld.global.nc`, not volatile, left to ptxas) | 0 / 0 | **1.4** | 6.16 µs | 0 / 0 | **1.05** | 8.21 µs |
| A whole batch: weak `ld.global` + `membar.cta` | 1 / 1 | 0.75 | 6.16 µs | **29 / 17** | 0 | **10.26 µs** |
| A in segments: issue part + `membar.cta`, issue the rest once the first groups of the current batch have no readers left + `membar.cta` | 0 / 0 | 1.08 | 6.17 µs | 7 / 5 | 0 | 8.21 µs |
| B whole batch in one asm + runtime-predicated branch | **51 / 27** | 0 | **10.25 µs** | **101 / 45** | 0 | **14.36 µs** |

- The plain spelling is REG 128 with 0 spills in both settings, and the minimum lead is ≥ one k-block in both: ptxas has already filled the registers and is using them to issue loads early.
- Neither A pinning method merged the trickle into a burst (number of LDG runs: smaller 9 → 9 / 10, larger 17 → 13 / 15); they only disturbed register reuse: either spilling
  or a shorter minimum lead. B pinning the whole batch mid-loop spills heavily in both settings, single launch 66% / 75% slower.
- Three controls in the same job — "one set of names / two sets of names (by kb parity)", "`asm volatile` / not volatile", "HMMA k-step outer / column-group outer": in the smaller setting the LDG / HMMA
  run strings are identical character for character; in the larger one the LDG count and REG are the same, `MOV` and runs differ in one or two places — **naming, volatile and HMMA order are not pins**
  (the order of the HMMA run itself is likewise not decided by the source scan order or `volatile`, see fact 4 of [`mma_sync_hmma_run_register_feed`](mma_sync_hmma_run_register_feed.md)).

**Register accounting** (why hard-pinning a burst near the cap loses): the active set at the burst point "after the current k-block's first HMMA" ≈ the result registers of the current batch that still have readers
(in reader order, each is live until its last reader) + the part of the next batch already issued + accumulators + addresses / indices. In the larger setting two whole batches plus accumulators exceed 128, so a full burst does not fit under this cap;
even when it fits on paper, the fence disturbs ptxas's register-reuse order (the smaller setting's whole-batch A has 1 / 1 spills).

**Whole-kernel reading in a real kernel (flat to slightly positive)**: in a register-fed `mma.sync` kernel with a 128-register cap, changing the k-loop to two fixed sets of register names by kb parity,
with the next batch written in the source after the current k-block's first HMMA (left to ptxas, no fence), was 0.32% / 0.42% faster in same-job geometric mean (two variants).
In that whole kernel the renaming **did change** the register allocation: one instance went from LDL / STL 3 / 3 → 0, minimum lead 14 → 17 HMMAs (in the probe, renaming did not change SASS in the smaller setting and changed only the `MOV` count and one or two runs in the larger one, see above) — go by your own SASS.

## Gotchas

1. **volatile is not a fence**: `asm volatile` and the `"memory"` clobber constrain only the compiler front end; ptxas gets flat PTX and reorders by register pressure. `.nc` loads
   need not be ordered against stores / fences, which is why A must use weak loads.
2. **The guard must be a runtime value** (computed from kernel parameters); a constant gets folded and the block boundary disappears with it; inline a labelled asm block only once per kernel.
3. **Once pinned, the active set = all load results before the fence / block + accumulators; beyond the per-thread cap it spills.** This entry's probe shape has 128 result registers against
   a cap of ≤ 128, so pinning necessarily spills — the "pinned is slower" in the table below is decided by this toy's construction, not a fixed cost of the fence itself; for the gain of pinning when the results fit,
   there is no CUDA-side reading. Probe readings (16 groups of `ld.global.nc.v8.b32` per lane = 128 result registers, feeding 32 `mma.sync.m16n8k16` directly,
   `__launch_bounds__(256, 2)` i.e. ≤ 128 registers):

   | Spelling | LDGs after the first HMMA | REG | LDL / STL | Single-wave timing |
   |---|---|---|---|---|
   | bare spelling (one non-volatile asm per load) | **2/16** | 126 | 0 / 0 | 15.01 µs |
   | A weak load + `membar.cta` | 0/16 | 128 | **16 / 11** | 16.29 µs |
   | B whole group in one asm + runtime-predicated branch | 0/16 | 128 | **19 / 13** | 16.38 µs |

   What gets pinned is **every load written in the source before the fence / block**; if the active set exceeds the cap it spills; a "pin" with non-zero `LDL/STL` is not a win. Do the accounting before pinning
   (per-thread cap ≈ 65536 / (threads × minBlocks), rounded down to a multiple of 8, ≤ 255; this probe 256 × 2 → 128), and time it in the same job after pinning.
4. **`membar.cta` has a semantic cost**: it also applies to the same thread's earlier stores / reds; in the certified setting only loads precede it.
5. **Count per path in a multi-path kernel**: split the SASS at unpredicated `EXIT`s and count only LDGs after the first HMMA within the same path, otherwise the second path's whole batch is counted as sunk.
6. **A steady-state "trickle" does not mean "late"**: when comparing two versions, compare the lead (minimum / mean) and same-job timing, not how nice the LDG / HMMA run strings look.
   (Inference, not certified separately: with registers fully loaded, a load's destination register becomes free only after the previous occupant's last reader has issued, so ptxas places it right after that HMMA;
   the loads are therefore scattered but each is issued as early as possible.)
7. **`membar.cta` pins "no later than"**: it only forbids earlier loads from moving past it downward; ptxas still hoists loads upward and inserts them among earlier HMMAs,
   so mid-loop A neither forms a burst nor leaves register reuse alone (table in §"k-loop steady state"). Weak loads also require this data to be read-only for the duration of the kernel.
8. **Spelling B is only for the batch "before the first HMMA"**: pinning a whole batch mid-loop with "one asm block + runtime-predicated branch" spills heavily in both settings (LDL 51 / 101).
9. **When registers are already at the cap and the steady-state lead is insufficient** (this section's probe did not hit this case; inference): look at the register budget first — accumulators, address registers, `MOV` temporaries assembling A quads,
   or the per-thread cap (threads / minBlocks); inserting a fence at the cap spills (table above). For the gain of pinning when REG is clearly below the cap, there is no CUDA-side reading.

## Sources

- PTX ISA: vector forms of `ld.global{.nc}` (`.v8.b32`), `membar` / `fence`, `bra` / predicates inside inline asm, `mma.sync.aligned.m16n8k16`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA Binary Utilities (`cuobjdump -sass` / `-res-usage`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- CUDA C++ Programming Guide, `__launch_bounds__`: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- GPU-certification probe — `PROBE PASS max_diff=0` (the three spellings' numerics are bitwise identical to the host's
  simulation of the mma fragments) and `PROBE PASS max_diff=n/a` (`ldg-pin-b16-sass`: the bare spelling sinks, neither A nor B sinks);
  the 12-group control is toy B of that separate probe (12 groups, REG 114, the bare spelling sinks 0).
- GPU-certification probe (§"k-loop steady state" and gotchas 6–9) —
  `PROBE PASS max_diff=0.0` (7 variants per setting, numerics bitwise identical; `trickle` minimum lead ≥ one k-block in both settings; `fence` A whole batch / in segments does not win in either setting;
  `blockpin` larger setting LDL 101 / STL 45; `names` / `vol` SASS identical in both settings); the timings and runs in the table are fact lines of the same run.
- Whole-kernel reading in a real kernel: numerics of the three variants in the same run `PROBE PASS max_diff=0.1250`, same as the baseline.
