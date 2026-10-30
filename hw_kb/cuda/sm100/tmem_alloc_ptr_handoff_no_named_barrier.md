# tmem_alloc_ptr_handoff_no_named_barrier (how to hand the TMEM base address to other roles without a cross-role named barrier: `cta_group::1` uses a count-1 mbarrier + a pair of `tcgen05.fence`; a CTA pair uses a second, release cluster barrier)

> **Certification source: re-certified on a GPU probe in CUDA mode** (two probe kernels, shape A and shape B, not included in this release;
> verdicts under Sources). The mechanism is the same as the same-named entry in the CuTe DSL tree; the trace-time errors about allocator objects there are DSL-specific and are not carried over.
>
> One file per hardware interface. Operator-agnostic: mechanism only. Architecture scope: sm100.

## Purpose

`tcgen05.alloc` is issued by one warp, and the hardware **writes the allocated TMEM base address into a word in smem** (the holder). Other warps (MMA issue, readback)
must first "see" this word. A common spelling opens a named barrier between the allocating warp and all consuming warps (something like `bar.sync id, 32+128`);
this entry gives two ways of publishing it without one:

| Shape | Applies to | How the base address is published | Who has to wait for the allocation |
|---|---|---|---|
| **B** | `cta_group::1` | allocating warp: `tcgen05.fence::before_thread_sync` → one lane arrives on a **count = 1** mbarrier; warps that use the base: `mbarrier.try_wait` (phase 0) → `tcgen05.fence::after_thread_sync` → read the holder | only the warps that use TMEM |
| **A** | CTA pair (`cta_group::2`) | allocate between two cluster barriers; the second uses the **release** form of `barrier.cluster.arrive` + `wait` (acquire), after which every thread does `tcgen05.fence::after_thread_sync` and reads the holder | everyone (including the TMA producer) |

Basis (PTX ISA): an mbarrier arrive defaults to `.release.cta`, try_wait to `.acquire.cta`; `barrier.cluster.arrive` defaults to release,
and `.relaxed` drops the release; the holder write by `tcgen05.alloc` is a tcgen05 operation, so per the tcgen05 memory consistency model the two fences
`before_thread_sync` / `after_thread_sync` must sit between it and the thread synchronization.

## When needed (trigger)

- The synchronization contract requires "named barriers only for rendezvous within a role; cross-role handoff via mbarrier or cluster barrier", while the kernel has an extra
  `bar.sync` of "allocating warp + consuming warps" width;
- in the prologue the TMA producer is also dragged behind the allocation by a whole-CTA rendezvous (shape B lets it not wait);
- some warp that **did not take part** in the synchronization also reads the base address — it may read the stale value from before the allocation (probe fact line: the producer warp reading the holder without synchronizing;
  the number of sentinel reads is under Sources).

**Anti-trigger**: do not apply shape B to a CTA pair — the peer CTA's MMA also writes this TMEM and allocation and release are paired; use shape A.

## Minimal runnable example

### Shape B: count-1 mbarrier (`cta_group::1`, same spelling as the certification probe)

```cpp
__shared__ __align__(8) uint64_t tready;     // one-shot: only phase 0 completes per launch
__shared__ uint32_t holder;                  // the base-address word written by tcgen05.alloc
if (threadIdx.x == 0) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&tready)));   // count 1
  // ... other mbarrier inits
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__syncthreads();                             // only so the init is visible (the prologue's only whole-CTA rendezvous)

if (warp == ALLOC_WARP) {                    // the whole warp executes alloc (.sync.aligned)
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(smem_u32(&holder)), "r"(NCOL) : "memory");                    // NCOL: a power of 2, 32..512
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncwarp();
  if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(&tready)) : "memory");
}
// every role that uses TMEM, before its first use of the base address (the allocating warp itself takes this step too):
mbar_wait(smem_u32(&tready), 0);             // mbarrier.try_wait.parity ... phase 0
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
const uint32_t tbase = *(volatile uint32_t*)&holder;
// The TMA producer does not use the base address: it does not wait on tready and issues its first load right after the init rendezvous.
```

**Release goes through the same pair of fences**: warps that read TMEM finish with `tcgen05.wait::ld` → `tcgen05.fence::before_thread_sync` → synchronize (mbarrier arrive or
`__syncthreads()`); the allocating warp, after the synchronization, `tcgen05.fence::after_thread_sync` → `tcgen05.dealloc.cta_group::1.sync.aligned.b32 tbase, NCOL;`
(it must be **the warp that allocated**).

### Shape A: the second cluster barrier of a CTA pair

```cpp
// thread 0: mbarrier init + fence.mbarrier_init.release.cluster
asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // rendezvous 1: only so the init is visible to the cluster
asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
if (warp == ALLOC_WARP) {                                               // the same-numbered warp in both CTAs
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem_u32(&holder)), "r"(NCOL) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");           // rendezvous 2: release (not .relaxed)
asm volatile("barrier.cluster.wait.aligned;" ::: "memory");             //               acquire
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
const uint32_t tbase = *(volatile uint32_t*)&holder;                    // any warp may read it, no further sync
// ... role dispatch; finish: __syncthreads(); one more cluster arrive/wait; the allocating warp fence::after + dealloc.cta_group::2
```

## Gotchas

1. **`barrier.cluster` waits for all non-exited threads of the cluster to arrive** (`.aligned` requires the branch condition to be warp-uniform). Placing it at the top level, before role dispatch, is safest
   (the top-level spelling is what the probe certified); placing it inside branches also works, but the branches together must cover all non-exited threads. If only some warps arrive and the others neither arrive
   nor exit, it completes only when they exit — and if they are still waiting on a result from inside the branches, it hangs (the probe did not run this negative case).
2. **In shape A the second one must be the release form** (`barrier.cluster.arrive.aligned`, without `.relaxed`). The first is only for mbarrier init visibility,
   already preceded by `fence.mbarrier_init.release.cluster`, so it may be relaxed. The probe certified only the release form and did not run a control with "the second also relaxed".
3. **Shape B's count = the number of threads that actually arrive**: if only lane 0 arrives, init to 1; with 32 arrivals from the whole warp but init 1, phase 0 flips early
   and the extra arrivals hit the next phase. This mbarrier completes only once per launch, and waiters always use phase 0; a persistent kernel that reallocates per tile must flip the phase per round (uncertified).
4. **Every warp that reads the base address must wait on its own first.** Do not read the holder before the wait and pass the value into the branches — that is the same thing as "reading without synchronizing".
   Probe fact line: a producer warp not taking part in the synchronization reads the holder at the start of the launch, and how often it sees the pre-allocation sentinel varies with scheduling (see Sources),
   while every thread that waited on tready reads the same non-sentinel base address every time.
5. **Write the two `tcgen05.fence`s as a pair**: the allocating side does `before_thread_sync` before the arrive (or cluster arrive), the waiting side does `after_thread_sync` after the wait and before reading the holder /
   the first tcgen05 instruction. The probe has both fences and did not test a control without them.
6. **alloc / relinquish / dealloc are all `.sync.aligned`, executed by one whole warp**; dealloc uses the same column count as alloc; the column count is a power of 2 in 32..512.
7. **The cost of shape A**: the second one is a whole-cluster rendezvous, so the TMA producer's first load is also delayed until after the allocation; shape B has no such cost but is certified only for `cta_group::1`.

## Sources

- PTX ISA: `tcgen05.alloc` / `dealloc` / `relinquish_alloc_permit` (address written to smem, `.cta_group::2`), the tcgen05 memory consistency model
  (`tcgen05.fence::before_thread_sync` / `after_thread_sync`), `mbarrier.arrive` (default `.release`) / `mbarrier.try_wait` (default `.acquire`),
  `barrier.cluster` (arrive defaults to `.release`, `.relaxed`, `.aligned`) —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-dealloc-relinquish-alloc-permit>
- Public infra API: CUTLASS `cute::TMEM::Allocator1Sm` / `Allocator2Sm` (`include/cute/arch/tmem_allocator_sm100.hpp`, same PTX spelling).
- GPU-certification probe: shape B (contracts `A2-publish-*`),
  shape A (contract `B2-pair-publish`), verdict
  `PROBE PASS max_diff=0.0`: `A2-publish-v64` and the other five settings (shape B: per CTA, 160 threads that waited on the count-1 mbarrier read the same non-sentinel base address),
  `B2-pair-publish` (shape A: 8 CTAs × 192 threads read the same base address after the release cluster barrier), `A1-exact-*` / `B1-pair-exact` (the GEMM under the same handoff is bitwise exact).
  Fact line: the unsynchronized producer warp reading the holder at the start of the launch read the pre-allocation sentinel in all 384 reads.
