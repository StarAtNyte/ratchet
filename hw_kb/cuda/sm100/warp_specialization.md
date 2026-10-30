# warp_specialization (`setmaxnreg`: redistributing registers among warpgroups by role)

> **Certification source: B200 GPU-certified in CUDA mode (certified by a GPU probe, not included in this release; all 6 contracts PROBE PASS).**
> Certification covers: where the static cap comes from, when ptxas silently drops `setmaxnreg`, how to lay out role branches so they do not spill, and the hang shape when the pool is overcommitted.
> **Uncertified**: any end-to-end performance gain (the probe measures only spills and correctness, no timing); the interaction of `setmaxnreg` with clusters / 2-CTA.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

An SM has 64K 32-bit registers. ptxas sets one **uniform entry cap** for the whole kernel from `__launch_bounds__`
(384 threads, 1 CTA per SM → floor(65536 / 384) rounded down to a multiple of 8 = **168**). In a warp-specialized kernel the roles' needs
differ widely (TMA / MMA issue warps need only a few dozen, epilogue / element-wise warpgroups 200+), so a uniform cap makes heavy roles spill while
light roles sit idle. `setmaxnreg.{dec,inc}.sync.aligned.u32 N` redistributes the CTA's register pool among warpgroups at runtime:
light roles `dec` to give registers up, heavy roles `inc` to take them; ptxas allocates the code section after each `setmaxnreg` against the new cap.

SASS footprint: `USETMAXREG.DEALLOC.CTAPOOL 0x18` (dec 24), `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0x100` (inc 256) —
`TRY_ALLOC` is an acquisition that **spins and retries**, waiting forever if the pool is short.

## When needed (trigger)

- In `cuobjdump -res-usage` the kernel's REG equals exactly the entry cap (e.g. 168), with `STACK > 0` and `LDL` / `STL` in SASS,
  and the spills are concentrated in one role's branch; other roles (warps issuing TMA / `tcgen05.mma` from a single thread) barely use registers.
- You want to give an element-wise / epilogue warpgroup a register budget larger than "65536 / thread count".
- You have already written `setmaxnreg`, but **cannot find `USETMAXREG`** in SASS (dropped by ptxas, see gotcha 1).

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// 384 threads = 3 warpgroups: WG0 light role (24), WG1 / WG2 heavy roles (224 / 256).
// 24*128 + 224*128 + 256*128 = 64512 = 384 * 168 (pool = thread count x entry cap, not 65536).
#define SETMAXNREG_DEC(R) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(R))
#define SETMAXNREG_INC(R) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(R))

__global__ void __launch_bounds__(384, 1)        // minBlocks is required: without it setmaxnreg is silently dropped
k_ws(const uint32_t* in, uint32_t* out, int iters) {
  const int wg = threadIdx.x / 128;               // a warpgroup-uniform predicate
  // ... shared prologue (smem / mbarrier init, TMEM allocation, etc.) ...
  __syncthreads();                                // CTA-level sync goes [before] the role branch
  if (wg == 0) {
    SETMAXNREG_DEC(24);                           // first statement of the branch; executed by all 4 warps
    /* the light role's whole persistent loop lives inside this branch */
  } else {
    if (wg == 1) { SETMAXNREG_INC(224); } else { SETMAXNREG_INC(256); }
    /* the heavy roles' whole persistent loops live here */
  }
  // nothing is done after reconverging below the branch: which cap the reconverged region is compiled against is out of your control (see gotcha 4)
}
```

Acceptance: in `cuobjdump -sass` each role has one `USETMAXREG`, and `LDL` / `STL` are 0 in the heavy-role branches; REG in `cuobjdump -res-usage`
still shows the entry cap (168), which is normal — it reports the entry value, not the value inside the branches.

## Gotchas

1. **With only `__launch_bounds__(384)` (no minBlocks), ptxas silently drops `setmaxnreg`.** Measured: the same code with
   `__launch_bounds__(384, 1)` → 3 `USETMAXREG`, 0 spills; `__launch_bounds__(384)` → **0 `USETMAXREG`**,
   REG 168, `LDL`/`STL` 101 / 100; no launch bounds at all → 0 `USETMAXREG`, REG 254, and a 384-thread launch fails outright
   (`cudaErrorLaunchOutOfResources`, 701). The drop produces only a ptxas warning, which a torch extension build does not show by default —
   **always accept by whether SASS has `USETMAXREG`**; do not trust the source.
2. **The cap on the `.inc` total is "thread count × entry cap", not 65536.** With 384 threads the pool = 384 × 168 = 64512.
   Measured: 32 / 224 / 256 (sum = 65536) compiles to SASS with the same structure as 24 / 224 / 256 (3 `USETMAXREG`, 0 spills),
   but **hangs** at runtime (`TRY_ALLOC` never gets its registers, 25 s timeout). No error, no illegal instruction, just a kernel that never returns.
3. **The predicate must be warpgroup-uniform, and all 4 warps must reach it** (`.sync.aligned`). Use quantities like `threadIdx.x / 128`;
   do not put it inside an `elect` / single-thread block. Put dec / inc as the first statement of their respective branches, with no barrier between them.
4. **"Move the budget first, reconverge, then split into roles" is the wrong layout.** Measured, the split shape (a few ifs at the start that only do `setmaxnreg`, then the role branches after reconverging):
   the same 3 `USETMAXREG`, but `LDL`/`STL` **381 / 380**, STACK 784, and **all** 761 spill instructions
   land in the reconverged region after the last `USETMAXREG` (the fused shape is 0 / 0). Rule: shared prologue → one role branch → each role's
   own persistent loop stays inside the branch → the branches never reconverge (teardown is also written into each branch).
5. **The control group must prove "it would spill".** In a probe validating that `setmaxnreg` works, the control (same code without `setmaxnreg`) must
   really have `LDL`/`STL` in SASS, otherwise "no spill" proves nothing. The live set must use values the compiler can neither recompute nor shrink
   (read from gmem, live across iterations, updated through an `asm volatile` `xor.b32` chain); constants or recomputable values get rematerialized by ptxas.
6. **Plan code shared by two warpgroups against the lower quota.** In the example WG1 / WG2 take 224 / 256 respectively, and the common body shared by both in the `else` branch
   can only count on 224 (in the probe 192 live words spill 0 in this section, which is consistent with that; "compiled against the lower quota" itself
   **has no separate probe reading**). For a role to really use all 256, give it its own code section.
7. **An architecture-specific target is required** (`sm_100a`, which the operator tree's default gencode already is).
8. **A register array with dynamic indices still goes into local memory**; no amount of registers from `setmaxnreg` can save it (see also the local-frame class of problems).

## Sources

- PTX ISA: *Miscellaneous Instructions: `setmaxnreg`* (`.dec` / `.inc`, `.sync.aligned`, values a multiple of 8 in
  [24, 256], uniform within a warpgroup) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA C++ Programming Guide: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` —
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- The same spelling in the public CUTLASS headers: `cutlass::arch::warpgroup_reg_alloc<N>()` / `cutlass::arch::warpgroup_reg_dealloc<N>()`
  (`asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount))`).
- GPU-certified by the certification probe:
  `PROBE PASS max_diff=0.0 ws/numeric-uniform`, `ws/numeric-fused`, `ws/numeric-split` (numerics of all three shapes bitwise exact);
  `PROBE PASS ws/uniform-control-spills ldl=96 stl=95 reg=168`;
  `PROBE PASS ws/fused-no-spill usetmaxreg=3 ldl=0 stl=0 reg=168`;
  `PROBE PASS ws/split-worse-than-fused usetmaxreg=3 ldl=381 stl=380`.
  Fact lines: the `__launch_bounds__(384)` shape usetmaxreg=0, ldl/stl 101/100; the shape without launch bounds usetmaxreg=0, reg=254,
  launch error 701; the sum = 65536 shape's child process timed out at 25 s (hang).
