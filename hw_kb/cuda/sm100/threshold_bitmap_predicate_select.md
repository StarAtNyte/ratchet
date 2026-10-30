# threshold_bitmap_predicate_select (contiguous column-range mask: a two-word threshold bitmap + per-column bit test → `R2P` + predicated `FSEL`)

> **Certification source: B200 GPU-certified in CUDA mode (all 6 contracts PROBE PASS; the probe is not included in this release).**
> Timing was taken only in this entry's probe under an issue-bound configuration (64 columns / thread, each column's select followed by FFMA + FMNMX, no MUFU,
> 296 CTAs × 256 threads); the gain shrinks when MUFU-bound (see gotcha 6).
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Each thread holds a fixed set of columns (column numbers are compile-time constants `c = 0..K-1`), and the valid columns are a contiguous range `base + c >= thr` (causal /
length masks and the like). Writing an integer compare per column costs one `ISETP` per column (plus address arithmetic); instead, **compute a threshold bitmap once per beat**
`ones_from(t)` (`t = thr - base`, one word per 32 columns) and do only a bit test per column: ptxas feeds the predicated `FSEL` with `R2P` (moves several bits of one register
into 7 predicate registers at once), and the per-column compare instructions disappear.

## When needed (trigger)

- SASS has one `ISETP` + one `FSEL` per column (`ISETP ≈ K` for K columns), plus one `VIADD` / `IADD3` per column
  (`base + c`), and the segment is issue-bound (main stalls `math_pipe_throttle` / `not_selected` / `dispatch`,
  not long scoreboard).
- The mask shape is **one contiguous range** (lower or upper bound), not an arbitrary bit pattern.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// Taken from this entry's probe. A PTX shl.b32 shift amount >= 32 clamps to 32 (result 0); a negative t must first be clamped to 0.
__device__ __forceinline__ uint32_t ones_from(int t) {   // bits >= t set
  uint32_t w;
  asm("{\n\t.reg .s32 tl;\n\t.reg .b32 ones;\n\t"
      "max.s32 tl, %1, 0;\n\t"
      "min.s32 tl, tl, 32;\n\t"
      "mov.b32 ones, -1;\n\t"
      "shl.b32 %0, ones, tl;\n\t}"
      : "=r"(w) : "r"(t));
  return w;
}

// once per beat: a two-word bitmap covers 64 columns
const int t = thr - base;                  // column c valid <=> c >= t
const uint32_t w0 = ones_from(t), w1 = ones_from(t - 32);
#pragma unroll
for (int c = 0; c < 64; ++c) {
  const uint32_t w = (c < 32) ? w0 : w1;   // c is a compile-time constant: the word is selected at compile time
  y[c] = (w & (1u << (c & 31))) ? s[c] : PARK;   // a plain C++ bit test is enough: gives R2P + predicated FSEL
}
```

Probe readings (64 columns / thread, the version without MUFU):

| Spelling | `R2P` | `ISETP` | `VIADD` | `FSEL` | Total | Time (relative to the C++ compare) |
|---|---|---|---|---|---|---|
| C++ compare `(base + c >= thr) ? s : PARK` | 0 | 65 | 59 | 64 | 499 | 1.000 |
| bitmap + C++ bit-test ternary | 8 | 3 | 1 | 64 | 396 | 1 / 1.461 |
| bitmap + per-column inline `and` / `setp` / `selp` | 8 | 1 | 1 | 64 | 396 | 1 / 1.461 |
| bitmap built by explicit C++ clamping + ternary | 8 | 7 | 2 | 64 | 403 | 1 / 1.388 |

**Once you have the bitmap, a plain C++ bit test is enough**: nvcc synthesizes `R2P` by itself, and inline asm brings no extra gain (per-column asm and asm in blocks of 8 columns
compile to the same instruction counts). The only step really worth hand-writing is building the bitmap (gotcha 1). All spellings use 77–80 registers, no `LDL` / `STL`.

## Gotchas

1. **Do not build the bitmap directly with C++ `~0u << t`**: it is undefined behavior when `t >= 32` or `t < 0`. Use the PTX `shl.b32` above
   (the shift amount clamps to 32, giving 0), or explicit clamping `t >= 32 ? 0u : (t <= 0 ? ~0u : ~0u << t)` (the probe measured that it also gives R2P, but with
   4 more `SEL` and a few more `ISETP` / `P2R`, time 1.388 vs 1.461).
2. Get the second word with the same function on `t - 32`; no branch.
3. Write the bit test as `w & (1u << c)` (a constant mask), not `w >> c` then `& 1` — that costs an extra shift.
4. Column numbers must be relative to the thread's own base (compile-time constants); written as `base + c` taking part in the arithmetic, the per-column integer adds come back.
5. If the threshold is a compile-time constant, the whole mask is constant-folded and there is no R2P — that is good, not a regression.
6. The gain appears only when the segment is issue-bound: in the probe, when MUFU-bound (one `ex2` per column), t(compare)/t(bitmap) drops from 1.461 to 1.067.
7. When counting SASS, exclude the self-jump `BRA` after `EXIT` (alignment padding), otherwise the branch count is inflated.

## Sources

- PTX ISA: *Integer Arithmetic Instructions: `min` / `max`*; *Logic and Shift Instructions: `shl`* (clamping when the shift amount exceeds the register
  width); *Comparison and Selection Instructions: `setp` / `selp`*;
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certified by the probe —
  `PROBE PASS max_diff=0.0 threshold_bitmap_predicate_select/ones_from_table n=128`;
  `PROBE PASS max_diff=6.943e-07 threshold_bitmap_predicate_select/numeric`;
  `PROBE PASS max_diff=n/a threshold_bitmap_predicate_select/cmp_baseline isetp=[65, 65] r2p=[0, 0]`;
  `PROBE PASS max_diff=n/a threshold_bitmap_predicate_select/bitmap_r2p arms_ok=['bm_asm', 'bm_asm8'] r2p_asm=8 r2p_asm8=8 isetp_asm=1 isetp_asm8=1`;
  `PROBE PASS max_diff=n/a threshold_bitmap_predicate_select/bitmap_r2p_cpp r2p=[8, 8] isetp=[3, 3] lop3=[7, 7] ldl=[0, 0] reg=[80, 78]`;
  `PROBE PASS max_diff=n/a threshold_bitmap_predicate_select/bitmap_time ratio=1.4611`.
