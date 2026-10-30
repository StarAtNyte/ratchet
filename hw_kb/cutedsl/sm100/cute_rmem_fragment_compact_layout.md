# cute_rmem_fragment_compact_layout (sm100)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Interface: `cute.make_rmem_tensor(layout_or_shape, dtype)` -- allocation of a register fragment.
>
> Status: ✅ Verified (GPU probe, not included in this release; `PROBE PASS max_diff=0.015625`, round 2 of this certification; all four contracts pass).
>
> **What is certified**: the probe tested only one ownership layout `(8,4):(1,1024)` (128 threads, 4 chunks × 8 bf16 per thread),
> two fragment element types bf16 / fp32, sm_100a, and the ptxas on the stack at the time. "Allocation by cosize" is the layout algebra itself
> and independent of geometry; "whether the hole array ends up in local memory" depends on ptxas and on the kernel's structure, and was measured only on the five variants of gotcha 7 below.

## Purpose

`cute.make_rmem_tensor(L, T)` allocates elements by **`cosize(L)`** (the span of the layout's codomain), not by
`size(L)` (the number of elements). Using an **ownership layout with a large stride** (a layout meant to mark out this thread's few
byte chunks in gmem / smem) directly to open a register fragment gives, in PTX, a
`.local` array "as long as the span, mostly holes in between". Whether it can then live in registers depends entirely on whether ptxas can
break it up: in a simple kernel it can (measured STACK 0), but when the kernel has an access shape ptxas cannot break up (measured:
**a cp.async smem ring -- cp.async into smem first, then read back from smem into that fragment**), the whole block stays in
**local memory**, and every read and write becomes `LDL` / `STL`. A compact layout has no `.local` array at the PTX level and does not depend on ptxas's luck. The correct approach: **use the ownership layout for gmem / smem views, and a compact layout of the same shape for the register
fragment**. With the same shape on both sides, the element-wise logical coordinates correspond one to one,
and `cute.copy` works as usual.

## When needed (trigger)

- A thread's ownership is "a few contiguous chunks, far apart from each other", e.g. `make_layout((8, 4), stride=(1, 1024))`
  meaning 4 chunks per thread, 8 elements per chunk, chunk distance 1024 elements; and the code writes
  `cute.make_rmem_tensor(own, dtype)`.
- NCU shows nonzero `spill_ld / spill_st` sectors while `regs_per_thread` is not high, or only a few above
  expectation. SASS has `LDL` / `STL`, and `cuobjdump -res-usage` reports `STACK` > 0.
- Side evidence (an NCU observation on one affected kernel, not a probe contract): `stall_long_sb` and `stall_mio` both clearly higher than
  a comparable kernel without spills, and `dram_bytes_write` also noticeably higher than the spill-free version (local-memory write-back). Note
  that NCU's `dram_bytes_write` can be far below the output-tensor bytes (measured: about 40-50% of the output bytes for the spill-free version, about 60% for the spilling one);
  do not use the output bytes as a baseline; compare against the spill-free version.
- Pricing fingerprint: `spill_st sectors ÷ number of rows ≈ threads per row × fp32 fragment bytes per thread ÷ 32 B`
  (example: 128 threads per row, 32 fp32 per thread = 128 B → 512 sectors / row), i.e. each row writes a whole fragment
  out to local memory once.
- The PTX has `.local` declarations (0 under a compact layout).
- You can check directly at trace time: `print(cute.cosize(frag.layout), cute.size(frag.layout))`;
  if the two numbers differ you have hit this entry.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32

VEC, NCHUNK, THREADS = 8, 4, 128

@cute.kernel
def k(px: cute.Pointer, po: cute.Pointer, R: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    L = VEC * NCHUNK * THREADS                      # elements per row

    # (1) ownership layout: used only to build gmem / smem views
    own  = cute.make_layout((VEC, NCHUNK), stride=(1, THREADS * VEC))   # cosize 3080
    # (2) register fragment: same shape, compact stride (default column-major → stride (1, 8))
    frag = cute.make_layout((VEC, NCHUNK))                              # cosize 32

    atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16,
                               num_bits_per_copy=128)
    tofs = tidx * VEC

    xb = cute.make_rmem_tensor(frag, BFloat16)      # not make_rmem_tensor(own, ...)
    yf = cute.make_rmem_tensor(frag, Float32)
    yb = cute.make_rmem_tensor(frag, BFloat16)

    nrows = (R - bidx + gdim - 1) // gdim
    for it in cutlass.range(nrows, unroll=1):
        row = bidx + it * gdim
        gx = cute.make_tensor((px + row * L + tofs).align(16), own)   # view: ownership layout
        cute.copy(atom, gx, xb)                     # 128-bit: mode 0 is 8 contiguous elements on both sides
        xf = xb.load().to(Float32)
        yf.store(xf * xf)                           # any element-wise computation
        yb.store(yf.load().to(BFloat16))
        go = cute.make_tensor((po + row * L + tofs).align(16), own)
        cute.copy(atom, yb, go)
```

(This toy kernel is a simplified form of the probe's round-1 minimal kernel; in round 1 it was mistakenly written as `make_rmem_tensor(own, …)`,
ptxas broke up the hole array anyway, and nothing showed in SASS, see gotcha 7. It only demonstrates the correct spelling and cannot prove that "the wrong spelling is fine".
The full-structure compact variant of the certification round is in the probe file.)

Both sides have shape `(8, 4)`, so the `i`-th element of the fragment (coordinate `(i % 8, i // 8)`) is
the `i`-th element of the gmem view. Element-wise subscripts like `frag[i]` and `.load()` / `.store()` need no change;
the only thing that changes is the layout passed to `make_rmem_tensor`.

Acceptance (machine-read, all done on the dumped cubin):

```bash
# set before compiling (must precede import cutlass); one process and one directory per variant
export CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_DUMP_DIR=$TMPDIR/dump CUTE_DSL_CACHE_DIR=$TMPDIR/cache
cuobjdump -res-usage $TMPDIR/dump/*.cubin        # STACK:0 (the LOCAL field is not a criterion: with the affected variant at STACK 12336, LOCAL is still 0)
cuobjdump -sass      $TMPDIR/dump/*.cubin | grep -cE '[^A-Za-z0-9_](LDL|STL)[. ]'   # 0
# an earlier criterion independent of ptxas: the PTX should have no .local array (needs CUTE_DSL_KEEP_PTX=1 dumped as well)
grep -c '\.local' $TMPDIR/dump/*.ptx                                              # 0
```

## Gotchas

1. **`make_rmem_tensor` allocates by cosize, not by size.** For `(8,4):(1,1024)`,
   `size = 32`, `cosize = 3·1024 + 7 + 1 = 3080`. One fragment is 3080 "register elements",
   far beyond the per-thread limit of 255 registers, so when ptxas cannot break it up it can only land in local memory. Check `print(cute.cosize(t.layout))` at trace time first.
2. **The numerics are completely correct, so tests do not catch it.** Demotion to local memory affects only performance, not results:
   in the probe the strided and compact full-structure variants have the same `max_diff` against numpy (both 0.015625,
   a one-ulp-level rounding difference in the bf16 output); the variant without the reduction + rsqrt gives 0.0, showing that this small difference comes from the reduction / rsqrt path
   and is unrelated to the layout. Only readings like `LDL/STL`, `STACK`, `spill_*` reveal it,
   and it looks a lot like "high register pressure", so it is easily misjudged as a need to shrink the live set. **Check cosize first, then do the live-set accounting.**
3. **Do not copy the stride too just to "match the shape".** What is needed is **the same shape** (element-wise coordinates correspond one to one,
   copies can pair up), not the same stride. A copy atom only requires that the stretch each atom covers be contiguous on both sides:
   mode 0 of `(8,4):(1,8)` is 8 contiguous bf16 = 16 B, pairing with mode 0 of `(8,4):(1,1024)`,
   so the 128-bit copy holds as usual.
4. **When the same ownership layout is also used for fp32 copies, the problem doubles.** E.g. `xf`, `wf`, `yf` all using
   `make_rmem_tensor(own, Float32)`, each 3080 × 4 B. When fixing, switch **all** register
   fragments to the compact layout together (the probe tested only "switch all" and "switch none"; what happens when only some are switched is untested;
   treat it as switch-all).
5. **Do not use `regs_per_thread` to judge whether it is fixed.** An array demoted to local memory occupies no registers, so when affected
   the register count may be only a few above expectation; once fixed, the register count may actually rise -- that is the fragment really entering registers.
   The criterion is only `LDL + STL == 0` and `STACK == 0`. Measured (same probe kernel): strided 83 regs + STACK 12336
   + LDL 7 / STL 8; compact 85 regs + STACK 0 + LDL/STL 0.
6. **Readings must be taken in separate processes.** The name of the dumped cubin file is built only from the host function signature (measured, of the form
   `cutlass_host_Ptrgmem_Ptrgmem_Ptrgmem___CUstream0x0.sm_100a.cubin`, carrying no variant information), so compiling two
   variants in the same process overwrites in place, and may also hit the same compile-cache entry (give every variant a fresh `CUTE_DSL_CACHE_DIR`).
7. **A toy kernel does not reproduce it; do not conclude from that that "cosize is not a problem".** In the minimal gmem→rmem→gmem kernel, the strided
   fragment was broken up completely by ptxas (STACK 0, LDL/STL 0, identical to compact; no PTX was taken in that round). In the full structure with a
   cp.async smem ring + invariant weight loads + a cross-warp barrier reduction inside the CTA, item-by-item
   ablation (one process per item; all strided fragments except the last row):

   | Variant | REG | STACK | LDL / STL | PTX `.local` |
   |---|---|---|---|---|
   | full structure | 83 | 12336 | 7 / 8 | 17 |
   | without the invariant (`invariant=True`) weight load | 83 | 12336 | 7 / 8 | 17 |
   | without the barrier reduction | 72 | 12336 | 7 / 8 | 17 |
   | **without the smem ring (no more cp.async into smem; x goes from gmem straight into the fragment)** | 82 | **0** | **0 / 0** | 17 |
   | full structure + compact layout | 85 | 0 | 0 / 0 | 0 |

   That is: among these five variants, **what decides whether the hole array stays in local memory is the smem ring** (cp.async into smem +
   reading back from smem into the fragment; the probe removed the two together as one item and did not split them further), not the barrier or the invariant load.
   But this is only the current ptxas version's behaviour on this structure; another structure may change it again. The PTX `.local` count is 17 in every strided
   variant (including the non-spilling "without the smem ring") and 0 in compact: it is a stable criterion for "the layout is written wrong";
   whether it actually spills is shown by `LDL/STL` and `STACK` (gotcha 5). **Always use the compact layout; do not bet on ptxas.**

## Sources

- CuTe DSL public API: `cute.make_rmem_tensor`, `cute.make_layout`, `cute.cosize`, `cute.size`,
  `cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), ..., num_bits_per_copy=128)`, `cute.copy`;
  CUTLASS CuTe layout-algebra documentation (definitions of size and cosize):
  https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html
- Tools: `cuobjdump -res-usage` / `cuobjdump -sass` (CUDA Binary Utilities documentation:
  https://docs.nvidia.com/cuda/cuda-binary-utilities/).
- GPU certification probe: `PROBE PASS max_diff=0.015625`
  (round 2, factorial form; contracts: (1) cosize strided=3080 / compact=32; (2) all 5 variants numerically against numpy,
  worst 0.015625; (3) full-structure compact: LDL=STL=0, STACK 0; (4) full-structure strided: STACK 12336, LDL 7 / STL 8, the gotcha reproduced).
