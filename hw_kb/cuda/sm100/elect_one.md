# elect_one (single-thread election: the scope of PTX `elect.sync`, the outer predicate, and "which instructions need an election and which must not have one")

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: `elect.sync` is warp-scoped; a warp-uniform outer predicate + elect = exactly one thread in the whole CTA;
> the publication order `elect → mbarrier.init → __syncthreads → arrive/wait`; `__activemask()` as the membermask in a lane-divergent region.
> **Uncertified (written from the PTX ISA documentation)**: the consequences of wrapping a whole-warp `.sync.aligned` instruction in elect (deadlock / undefined; the probe did not trigger it);
> the initialization order of cluster-scoped mbarriers.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

`elect.sync d|p, membermask` selects **exactly one** of the threads named in membermask; the selected thread gets predicate `p = true`,
and every participating thread gets the selected thread's lane number `d`. It is a warp-level primitive: in a warpgroup each warp elects one of its own.

In CUDA the rule for "who should be selected" is simple: **inline PTX / CuTe C++ copy and mma issues are all per-thread instructions**, with no
internal election — if a whole warp reaches a TMA issue, it is issued 32 times.

## When needed (trigger)

- Issuing an instruction that may be issued "only once per CTA / per warp" on a code path that the whole warp (or warpgroup) reaches:
  `cp.async.bulk.tensor` (TMA) / `cp.async.bulk` / `cp.reduce.async.bulk`, `tcgen05.mma`, `tcgen05.commit`,
  `mbarrier.init`, `mbarrier.arrive.expect_tx` (in a single-thread-arrive protocol).
- Symptom: a count / reduction result is exactly 32× the right value (or 4× = one per warp), an mbarrier's tx count is expected repeatedly,
  the phase flips early or is never reached.

## Minimal runnable example

```cpp
// The same PTX as CUTLASS cute::elect_one_sync() (include/cute/arch/cluster_sm90.hpp): membermask = whole warp
__device__ __forceinline__ uint32_t elect_one_sync_full() {
  uint32_t pred = 0, laneid = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "     elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "     mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFFu));
  return pred;
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(count) : "memory");
}

__global__ void k(int* out, int value) {
  __shared__ __align__(8) uint64_t bar;
  __shared__ int payload;
  const int warp = threadIdx.x / 32;
  // only one thread in the whole CTA: warp-uniform outer predicate + elect. && short-circuits, only warp 0 executes elect.sync
  if (warp == 0 && elect_one_sync_full()) mbar_init(&bar, blockDim.x);
  __syncthreads();                 // a CTA sync between init and first use (CTA scope)
  if (threadIdx.x == 0) payload = value;
  /* everyone mbarrier.arrive (release) → mbarrier.try_wait.parity(…, 0) (acquire) → read payload */
}

// electing one inside a lane-divergent region: membermask is the current active set, not a hard-coded 0xFFFFFFFF
if (lane < n_active) {
  uint32_t m = __activemask();
  uint32_t leader, pred;
  asm volatile("{\n.reg .b32 %%rx;\n.reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n@%%px mov.s32 %1, 1;\n     mov.s32 %0, %%rx;\n}\n"
               : "+r"(leader), "+r"(pred) : "r"(m));       // set leader / pred to 0 first
}
```

## Gotchas

1. **It is warp-scoped, not CTA-scoped.** Measured: with all of 128 threads calling `elect_one_sync()`, the number selected per warp =
   **[1, 1, 1, 1]** (control group without election = [32, 32, 32, 32]). For one per CTA, wrap it in a warp-uniform predicate:
   `warp == 0 && elect_one_sync()` measured = [1, 0, 0, 0]. The number-one cause of "the mbarrier count is 4×" is a missing outer shell.
2. **The outer predicate must be warp-uniform; an elect with a whole-warp mask must not be put in a lane-divergent region.** `cute::elect_one_sync()` hard-codes
   `0xFFFFFFFF`, and the PTX ISA requires every thread in membermask to execute this `elect.sync`. To elect one inside a divergent region such as `if (lane < k)`,
   use `__activemask()` as the membermask: measured with 13 / 14 / 15 / 16 active lanes in the four warps, each warp selected exactly 1,
   and every selected lane was in the active set.
3. **Which thread is selected is not guaranteed.** On this machine the selected one was always the lowest active lane (lane 0 for a full warp), but the PTX ISA only promises "exactly one".
   If you need a specific lane, write `if (lane == 0)` (a plain predicate, not an election); to use the selected thread's private value, broadcast it explicitly via `d` (the leader lane) or
   `__shfl_sync`.
4. **Which instructions must be issued by one selected thread**: TMA (`cp.async.bulk.tensor.*`), `cp.async.bulk` / `cp.reduce.async.bulk`,
   `tcgen05.mma`, `tcgen05.commit`, `mbarrier.init`, `mbarrier.arrive.expect_tx` in a single-thread protocol. These are all per-thread instructions;
   the issue functions in CUDA / CuTe C++ have no internal election.
5. **Which instructions must never be wrapped in elect**: whole-warp (or whole-warpgroup) instructions carrying `.sync.aligned` — `tcgen05.alloc` /
   `tcgen05.dealloc` / `tcgen05.relinquish_alloc_permit`, `tcgen05.ld` / `tcgen05.st` / `tcgen05.wait::*`,
   `setmaxnreg`. The PTX ISA requires every thread of the warp to execute the same `.aligned` instruction; letting only one thread in is
   undefined behavior (typically a hang). This entry's probe did not trigger this shape; the basis is the documentation.
6. **There must be a sync between `mbarrier.init` and its first use.** The election only guarantees that one thread did the init, not that others can see it. CTA scope:
   `[elect] init → __syncthreads() → use`; measured: all 128 threads read the published value via arrive / wait, no hang. A barrier used by other CTAs in the cluster
   additionally needs `fence.mbarrier_init.release.cluster` + a cluster sync (uncertified in this entry).
7. **Without the sm_90+ feature macros, `cute::elect_one_sync()` degrades to `threadIdx.x % 32 == 0`** (the `#elif` branch in the CUTLASS header):
   still one per warp, but no longer `elect.sync` and without `.sync` semantics. To confirm which path you are on, look for `elect.sync` in the `cuobjdump -ptx` output.

## Sources

- PTX ISA: *Parallel Synchronization and Communication Instructions: `elect.sync`*; `mbarrier` in the same chapter (release / acquire semantics of init / arrive /
  try_wait, `fence.mbarrier_init`); *Tensor Core 5th Generation Family Instructions*
  (the `.sync.aligned` requirement of `tcgen05.alloc` / `ld` / `st` etc.) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA C++ Programming Guide: Warp Vote Functions (`__activemask()`) — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- Public CUTLASS headers: `cute::elect_one_sync()` (`include/cute/arch/cluster_sm90.hpp`).
- GPU-certified by the probe:
  `PROBE PASS max_diff=0.0 elect_one/R0-no-elect-control per_warp=[32, 32, 32, 32]`,
  `PROBE PASS max_diff=0.0 elect_one/R1-elect-is-warp-scope per_warp=[1, 1, 1, 1]`,
  `PROBE PASS max_diff=0.0 elect_one/R2-warp0-guard-plus-elect per_warp=[1, 0, 0, 0]`,
  `PROBE PASS max_diff=0.0 elect_one/R3-elect-init-sync-arrive-wait-publish`,
  `PROBE PASS max_diff=0.0 elect_one/R4-divergent-activemask-elect elected=[1, 1, 1, 1] leader_lane=[0, 0, 0, 0]`.
