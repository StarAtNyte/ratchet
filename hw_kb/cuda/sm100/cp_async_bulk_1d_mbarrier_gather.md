# cp_async_bulk_1d_mbarrier_gather (non-tensor-map 1D bulk async copy: row-by-row indirect gather into an mbarrier-accounted smem ring)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0.0` (`bulk-g2s-ring`, `bulk-g2s-ring-l2hint`, `bulk-sass-opcodes`).** Certifies only byte correctness and the SASS footprint, **not any performance gain**.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Turn "move rows into smem by an index table" from per-thread LDG→STS (or per-thread 16 B `cp.async`) into **one byte-counted bulk async copy per row**:
`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], nbytes, [mbar]`. The source is a plain gmem
pointer + a byte count (no tensor map needed, so it does not matter that the row numbers come from another table); completion is the hardware booking the arrived bytes into the mbarrier; the consumer side only waits for
one phase flip. No swizzle (the destination is linear; to land in a swizzled operand face see
[`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md)).

## When needed (trigger)

- The feed is an indirect row-by-row gather, each row ≥ 16 B contiguous, the destination linear, and a tensor map's box cannot frame it.
- You want the `UBLKCP.S.G` footprint (one instruction per whole segment, completion via tx-count), and today it is `LDG`+`STS` or `LDGSTS`.
- **Price it first**: swapping a long-scoreboard wait for a barrier wait does not mean faster; whether a shape where a single thread issues dozens serially with only one stage in flight
  can hide the latency must be timed in the same job on your own kernel (this entry has no CUDA-side timing readings).

## Minimal runnable example

```cpp
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t nb, uint32_t mbar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(nb), "r"(mbar) : "memory");
}
// with an L2 policy: .L2::cache_hint + a 5th operand (64-bit policy, produced by createpolicy, built once per CTA)
__device__ __forceinline__ void bulk_g2s_hint(uint32_t dst, const void* src, uint32_t nb, uint32_t mbar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1], %2, [%3], %4;" :: "r"(dst), "l"(src), "r"(nb), "r"(mbar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}

// 2-slot ring, R rows per stage, two segments per row (A table 1024 B + B table 128 B), row pitch PITCH (multiple of 16 B)
constexpr uint32_t TX = R * (1024 + 128);                   // compile-time constant: out-of-range rows are issued and counted too
auto issue = [&](int slot, int tile) {                      // called by thread 0 only
  mbar_arrive_expect_tx(bar0 + 8 * slot, TX);               // arm first, then issue row by row
  for (int p = 0; p < R; ++p) {
    int rid = sIdx[tile * R + p];
    rid = rid - (rid & (rid >> 31));                        // negative index -> row 0 (branch-free)
    const uint32_t d = face + (slot * R + p) * PITCH;
    bulk_g2s(d,        A + (size_t)rid * 1024, 1024, bar0 + 8 * slot);
    bulk_g2s(d + 1024, B + (size_t)rid * 128,  128,  bar0 + 8 * slot);
  }
};
if (threadIdx.x == 0) issue(0, 0);
for (int c = 0; c < ntile; ++c) {
  const int cur = c & 1;
  while (!mbar_try_wait(bar0 + 8 * cur, (c >> 1) & 1)) {}   // phase = parity of the lap
  if (c + 1 < ntile && threadIdx.x == 0) issue((c + 1) & 1, c + 1);   // into the slot drained on the previous lap
  /* ... whole CTA reads face[cur] ... */
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");      // generic reads before the next lap's async writes
  __syncthreads();                                                   // the slot is reused only after it is drained
}
```

## Gotchas

1. **`expect_tx` must equal exactly the bytes this stage issues; invalid rows must be issued and booked too.** Issuing one fewer is a **hang**, not an exception. Sentinel indices
   (−1) are clamped branch-free to a legal row and issued anyway, so tx stays a constant. The probe puts −1 in the middle and at the end of every stage.
2. **All copies of a stage are issued by one thread, the barrier is `init(count = 1)`, and only that thread does `arrive.expect_tx`.**
3. **Byte counts and alignment**: PTX requires the byte count to be a multiple of 16 and source / destination 16 B aligned (this entry certified 1024 B and 128 B segments with a 1168 B row pitch).
4. **The cache policy does not show in the opcode**: the SASS of the two kernels with and without `.L2::cache_hint` both read only `UBLKCP.S.G` (32 static instructions each), the opcode unchanged. It is numerically invisible; its value is not priced in this entry.
5. **Drain before reusing a slot**: only after the consumers have read (`fence.proxy.async.shared::cta` + `__syncthreads`) may the next lap be issued into the same slot.
6. **Bulk copies shared → shared (to a peer in the cluster) have a separate certified recipe**:
   [`cp_async_bulk_s2s_cluster_exchange`](cp_async_bulk_s2s_cluster_exchange.md).

## Sources

- PTX ISA, `cp.async.bulk` (non-tensor form, `.mbarrier::complete_tx::bytes`, `.L2::cache_hint`, size and alignment),
  `createpolicy`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`, `fence.proxy.async`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public libcu++ wrapper for the same instruction family: `cuda::ptx::cp_async_bulk` (`<cuda/ptx>`; this entry did not certify the wrapper path).
- GPU-certification probe — `PROBE PASS max_diff=0.0`: 4 CTAs, a 2-slot ring running 3 tiles (phase flips and slot reuse both exercised), 64 rows x (1024 + 128) B per stage, one stretch of −1 indices in the middle and at the end of every stage; the staging face equals `A[idx] ++ B[idx]` segment by segment (negative indices as row 0); the L2-hint cell is exact as well.
