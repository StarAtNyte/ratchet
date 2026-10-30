# teardown_subset_rendezvous_early_exit (before releasing TMEM, rendezvous only the warps that touched TMEM; the other roles exit early on their own)

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS; the probe is not included in this release).**
> Certification covers: the single-CTA `cta_group::1` shape — readers `tcgen05.wait::ld` + `bar.arrive`, owner `bar.sync` + `tcgen05.dealloc`,
> warps that do not touch TMEM exit directly without taking part in the rendezvous; exact readback, no hang over repeated launches, SASS footprint, exit order (judged by data, not timestamps).
> **Uncertified**: the dealloc handshake of `cta_group::2` (2-CTA pair), the conditions for removing the trailing cluster barrier, any performance gain;
> the consequences of a wrong rendezvous count (too many / too few) were not triggered on the GPU.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Before `tcgen05.dealloc`, every warp that read this TMEM must have finished reading. The simplest spelling is a whole-CTA `__syncthreads()` then release,
but that also holds the **roles that never touched TMEM** (TMA producer, MMA issue warp, idle warps) at the kernel tail, waiting along.
Use a rendezvous on a **named barrier whose count covers only the TMEM warps**: readers do a non-blocking `bar.arrive`, the owner a blocking `bar.sync` and then releases;
the other warps exit once they have fulfilled their own obligations.

## When needed (trigger)

- The kernel tail has a whole-CTA `__syncthreads()` / `bar.sync 0` whose only reason to exist is "everyone has finished reading before the dealloc".
- The role split is clear: only the epilogue warpgroup reads TMEM, and the producer / MMA warps have long had nothing to do.

## Minimal runnable example

```cpp
// 8 warps; warps 4..7 = the TMEM warpgroup (warp 4 does alloc / dealloc), warps 0..3 never touch TMEM.
__device__ __forceinline__ void tm_ld4_wait(uint32_t taddr, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(taddr) : "memory");
  // thread the loaded registers through the wait, so no use can be scheduled before the wait
  asm volatile("tcgen05.wait::ld.sync.aligned;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) :: "memory");
}

__global__ void __launch_bounds__(256, 1) k(/* ... */) {
  __shared__ uint32_t tmem_slot;
  const int warp = threadIdx.x >> 5;
  if (warp < 4) {                                  // non-TMEM roles: leave once their own work is done, take part in no rendezvous
    /* ... */
    return;
  }
  if (warp == 4) {                                 // executed by the whole warp (.sync.aligned); do not wrap in elect
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(&tmem_slot)), "r"(32) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  asm volatile("bar.sync 1, 128;" ::: "memory");   // publish the address only among the 4 TMEM warps
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t base = tmem_slot;
  const uint32_t taddr = base + ((uint32_t)(32 * (warp & 3)) << 16);   // the lane quadrant this warp may access
  /* ... several rounds of tcgen05.st / tcgen05.ld, every ld followed by wait::ld ... */
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  if (warp != 4) {
    asm volatile("bar.arrive 2, 128;" ::: "memory");   // the 3 reader warps arrive without blocking
    return;
  }
  asm volatile("bar.sync 2, 128;" ::: "memory");       // the owner waits for 3 readers + itself = 128 threads
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(base), "r"(32) : "memory");
}
```

## Gotchas

1. **Count N = 32 × (warps that arrive + warps that sync)**; `bar.arrive` and `bar.sync` count toward the same counter.
   N too large → the owner waits forever (hang); N too small → early release, and the dealloc races reads still in progress. **The numeric readback cannot show the latter**;
   only checking the count against the list of participants can. A warp not counted in N must never arrive, not even once.
2. **Readers do `tcgen05.wait::ld` first, then `tcgen05.fence::before_thread_sync`, then arrive**; the owner, after `bar.sync`, does
   `tcgen05.fence::after_thread_sync` and then dealloc. `tcgen05.ld` is asynchronous; "done reading" without the wait does not count as done reading.
3. **One barrier id per purpose.** Id 0 is used by `__syncthreads()`; use 1 for address publication and 2 for teardown, each with its own count and pairing;
   do not reuse one id for two things.
4. **"Not taking part in the rendezvous" does not mean "free to exit at any time".** Non-TMEM roles must still fulfil their own obligations before exiting: the TMA producer must first wait out the empty barriers of in-flight
   stages (otherwise late async arrivals land in the smem of a CTA that has already exited); a role that issued bulk stores must
   `cp.async.bulk.wait_group.read 0`.
5. **Accept by footprint, not by numerics.** Measured SASS of the subset shape: `@P0 BAR.ARV 0x2, 0x80` + `BAR.SYNC.DEFER_BLOCKING 0x2, 0x80`,
   no `BAR.SYNC ... 0x0`; the whole-CTA control shape: no `BAR.ARV`, `BAR.SYNC.DEFER_BLOCKING 0x0` at two call sites.
   Both shapes read back bitwise exact; the numerics cannot tell them apart.
6. **Judge exit order by data, not by `%globaltimer` stamps.** Have each TMEM warp, after it finishes reading, write a "completion token" unique to this launch into smem,
   and have the other warps count how many they can see before exiting — however many they see, that many have finished reading. Measured: in the subset shape, when the non-TMEM warps exit,
   592 / 592 warps see 0 tokens (early exit holds); in the whole-CTA control 592 / 592 see all 4 (the barrier held them back).
   Yet in that same control, the timestamp the non-TMEM warps take **after** `bar.sync 0` (in SASS `CS2R ..., SR_GLOBALTIMERLO` really is scheduled after
   `BAR.SYNC.DEFER_BLOCKING 0x0`) is about 11 µs earlier than the stamp the TMEM warps take **before** the barrier, the same as the subset shape's reading.
   That is, the order obtained by comparing `%globaltimer` stamps across warps can be the opposite of the real happens-before (cause not found);
   to judge "who left first", trust only data.
7. **`tcgen05.alloc` / `relinquish_alloc_permit` / `dealloc` are whole-warp instructions** (`.sync.aligned`), executed together by all
   32 threads of the owner warp; do not wrap them in `elect`.
8. **The pair shape (`cta_group::2`) has an additional dealloc handshake between the two CTAs**, uncertified in this entry; do not drop it by copying the single-CTA spelling.

## Sources

- PTX ISA: *Tensor Core 5th Generation Family Instructions* (`tcgen05.alloc` / `dealloc` / `relinquish_alloc_permit`,
  `tcgen05.ld` / `st` / `wait::ld` / `wait::st`, `tcgen05.fence::before_thread_sync` / `after_thread_sync`, the lane quadrants of TMEM addresses);
  *Parallel Synchronization and Communication Instructions: `bar` / `barrier`* (id and thread count of `bar.arrive` / `bar.sync`) — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- GPU-certified by the probe (148 CTAs × 8 warps, 64 rounds of `tcgen05.ld` + `wait::ld` per TMEM warp, 3 back-to-back launches per shape):
  `PROBE PASS max_diff=0.0 td/readback-subset launches=3`, `PROBE PASS max_diff=0.0 td/readback-control launches=3`,
  `PROBE PASS td/footprint-bar-arv subset_arv=1 control_arv=0`,
  `PROBE PASS td/early-exit-subset min_early_frac=1.000`, `PROBE PASS td/no-early-exit-control max_early_frac=0.000`
  (all 5 PASS). Fact line td/sass-order-control: the non-TMEM path is `BAR.SYNC.DEFER_BLOCKING 0x0` →
  `CS2R ..., SR_GLOBALTIMERLO` → `EXIT`; token counts all 0 in subset, all 4 in the control, timestamp lead about 11 µs in both shapes.
