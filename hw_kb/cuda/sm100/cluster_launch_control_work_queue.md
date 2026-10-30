# cluster_launch_control_work_queue (cluster launch control work queue: the grid is declared per work item, resident CTAs / clusters claim the next item with `clusterlaunchcontrol.try_cancel`)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0.0` (`clc-clc1`, `clc-clc2`).** Certifies only the queue's correctness (every item exactly once, the pair reads the same string, the cancel path is actually taken), not the gain.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

A persistent kernel's "work item → CTA" dispatch can be written two ways. **Static stride**: grid = number of resident CTAs, the c-th one does
`c, c + N, …`. **Cluster launch control (CLC)**: the grid is declared as the total number of work items, the hardware launches only the batch that fits; each launched
CTA (or cluster) sends a `try_cancel` to the scheduler each time — cancelling a CTA / cluster that **has not launched yet** and taking its `blockIdx` as its own
next item; a response saying "nothing cancelled" means it is done. The trip count becomes a runtime quantity, and who does which item is decided by the hardware's launch order.

## When needed (trigger)

- Work items have uneven work, so under a static stride some CTAs finish early while others straggle (this is the applicability the mechanism gives; this entry has no priced reading).
- The contract requires `grid` to equal the total number of work items (the declared shape itself is part of the contract).
- **Anti-trigger**: work items have equal work and under a static stride the CTAs' item counts differ by only 1 — a queue cannot even out an imbalance that does not exist.

## Minimal runnable example

```cpp
__device__ __forceinline__ void clc_try_cancel(uint32_t resp /*16 B aligned*/, uint32_t bar) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
               :: "r"(resp), "r"(bar) : "memory");
}
// cluster version: the response and complete_tx land at the same smem offset in every CTA of the cluster
__device__ __forceinline__ void clc_try_cancel_mc(uint32_t resp, uint32_t bar) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"
               ".multicast::cluster::all.b128 [%0], [%1];" :: "r"(resp), "r"(bar) : "memory");
}
__device__ __forceinline__ uint32_t clc_query(uint32_t resp, uint32_t& x) {   // 1 = cancelled one, x = first CTA id of the cancelled one
  uint32_t ok, cx;
  asm volatile("{\n\t.reg .b128 h;\n\t.reg .pred p;\n\t"
               "ld.shared.b128 h, [%2];\n\t"
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, h;\n\t"
               "selp.u32 %0, 1, 0, p;\n\tmov.u32 %1, 0xffffffff;\n\t"
               "@p clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %1, h;\n\t}"
               : "=r"(ok), "=r"(cx) : "r"(resp) : "memory");
  x = cx; return ok;
}

// no-cluster version: only one query in flight at a time, issued before this item's work
__shared__ __align__(16) uint4 resp;  __shared__ __align__(8) uint64_t bar;
if (threadIdx.x == 0) { mbar_init(smem_u32(&bar), 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__syncthreads();
uint32_t item = blockIdx.x, phase = 0;                 // first item = own blockIdx (the queue only hands out later items)
while (true) {
  if (threadIdx.x == 0) { mbar_arrive_expect_tx(smem_u32(&bar), 16); clc_try_cancel(smem_u32(&resp), smem_u32(&bar)); }
  do_work(item);
  while (!mbar_try_wait(smem_u32(&bar), phase)) {}
  phase ^= 1;
  uint32_t x; const uint32_t ok = clc_query(smem_u32(&resp), x);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic read of the response -> before the next async write
  __syncthreads();                                                 // reissue only after everyone has read it
  if (!ok) break;                                                  // done at the first "nothing cancelled"; no more queries
  item = x;
}
// cluster (2,1,1) version: one leader thread arms its own barrier (local arrive.expect_tx 16) and the peer's
// (remote mbarrier.arrive.expect_tx.shared::cluster 16, address mapa(bar, 1)), then issues clc_try_cancel_mc;
// both CTAs wait on their own barrier and read the same response; the leader reissues only after one cluster rendezvous following the read; item = x / 2;
// the prologue needs one cluster rendezvous after init (the peer's barrier must exist before it can be armed remotely).
```

## Gotchas

1. **A launched CTA's first item is its own `blockIdx`.** `try_cancel` only cancels CTAs that have not launched; taking the first item from the queue makes every resident
   CTA drop one item — the coverage contract must count holes over all work items.
2. **The response slot is 16 B aligned** (`.b128`); wait for that barrier phase before reading; after reading and before reissuing, `fence.proxy.async.shared::cta` +
   a whole-CTA rendezvous (the response is written by the async proxy and read by generic reads).
3. **Do not query again after the first "nothing cancelled"**; no query may still be in flight at exit (in this form only one is in flight at a time, and it has been waited on before exit).
4. **The cluster version must arm 16 B on the barrier of every CTA in the cluster**, which assumes every CTA has the same smem layout (response slot and barrier at the same offsets).
5. **It only counts as tested if the cancel path is actually taken**: the grid must be much larger than the resident count (the probe uses about 200 KB of dynamic smem to force 1 CTA/SM), otherwise all
   CTAs launch and the queue never cancels anything. Probe readings: 2048 CTAs declared (no-cluster version) / 1024 clusters (pair version), 148 CTAs actually launched, all the rest claimed via cancellation; every item exactly once, no holes, no duplicates, at most 14 items claimed by a single CTA, identical results over three repeated launches.
6. **The profiler's grid size is the declared value**; host-side resources allocated per CTA must be sized by the declared grid; the resident count is still decided by occupancy.

## Sources

- PTX ISA, `clusterlaunchcontrol.try_cancel` (`.multicast::cluster::all`), `clusterlaunchcontrol.query_cancel`
  (`.is_canceled`, `.get_first_ctaid`), `ld.shared.b128`, `mbarrier.arrive.expect_tx`, `fence.proxy.async`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public libcu++ wrappers: `cuda::ptx::clusterlaunchcontrol_try_cancel` / `clusterlaunchcontrol_query_cancel_*` (`<cuda/ptx>`; this wrapper path is uncertified).
- GPU-certification probe — `PROBE PASS max_diff=0.0`: `clc1` (no cluster) and `clc2` (cluster (2,1,1) + multicast response), 3 launches each; the contract is in gotcha 5.
