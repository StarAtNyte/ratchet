# cp_async_bulk_s2s_cluster_exchange (1D bulk async copy from smem into another CTA's smem in the cluster: peer mbarrier accounting, source drained by a remote arrive, the sender-armed variant)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS` (`bulk-s2s-ack-drain` max_diff=0, `bulk-s2s-sender-armed-b1` / `-b2` max_diff=0.0, `bulk-s2s-receiver-must-not-arm`, `bulk-sass-opcodes`), passed in two separate runs.** Certifies only the accounting mechanism and byte correctness, not performance.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

Turn "move a contiguous stretch of one CTA's smem as a whole into another CTA's smem in the cluster" from per-thread `st.shared::cluster` (one per 8 B,
and after issuing you still need a cluster rendezvous to know the peer is written) into one bulk async copy:
`cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dp], [src], nb, [mp]`. `src` is this CTA's
smem address; `dp` and `mp` are both **peer** addresses (converted from local offsets with `mapa.shared::cluster`; both sides have the same smem layout).
Completion is booked on the peer's barrier, the peer waits only on its own barrier, and the sender moves on once it has issued.

Two accounting forms; **do not mix them**:

| | Form A (receiver arms) | Form B (sender arms) |
|---|---|---|
| Receive barrier | `init(1)`, the receiver does `arrive.expect_tx(Σ bytes it will receive)` **before** the cluster rendezvous | `init(N)` (N = number of senders), the receiver **arms nothing** |
| Sender | issues the copy right after the rendezvous | each sender sends the peer one remote `mbarrier.arrive.expect_tx.shared::cluster.b64 _, [mp], tx`, tx = the bytes it actually sends to that peer |
| Use when | the receiver knows the total bytes in advance | senders send different byte counts (whole stretch + small header / small header only), and the receiver should not make a DRAM round trip before the rendezvous for this |

## When needed (trigger)

- Exchanging whole KB-sized smem stretches between CTAs of a cluster without a second `barrier.cluster` to confirm "the peer is written".
- Already writing this PTX and stuck on: the peer barrier never fills / guarding the source with `commit_group + wait_group.read` still sends dirty data /
  not knowing when it is safe to exit.

## Minimal runnable example

```cpp
__device__ __forceinline__ void bulk_s2s(uint32_t dst_local, uint32_t src, uint32_t nb, uint32_t mbar_local, uint32_t peer) {
  const uint32_t dp = mapa_u32(dst_local, peer), mp = mapa_u32(mbar_local, peer);   // dst and mbar both mapped to the peer, src not
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dp), "r"(src), "r"(nb), "r"(mp) : "memory");
}
__device__ __forceinline__ void mbar_arrive_remote(uint32_t cbar) {           // remote arrive: the destination operand must be the sink _
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(cbar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx_remote(uint32_t cbar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;" :: "r"(cbar), "r"(tx) : "memory");
}

// ---- Form A: pairwise exchange, the same source face reused for two rounds (cluster 8, peer = rank ^ 4)
if (tid == 0) {
  mbar_init(data1, 1); mbar_init(data2, 1); mbar_init(ack, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  mbar_arrive_expect_tx(data1, NB); mbar_arrive_expect_tx(data2, NB);   // arm before the rendezvous
}
/* ... write source face sSrc ... */
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");             // generic writes -> visible to the copy engine
__syncthreads();
cluster_sync();                                                          // the kernel's only cluster rendezvous
if (tid == 0) bulk_s2s(sR1, sSrc, NB, data1, peer);
while (!mbar_try_wait(data1, 0)) {}             // the peer's round 1 has fully arrived => the peer's source face has been read
if (tid == 0) mbar_arrive_remote(mapa_u32(ack, peer));                   // tell the peer
while (!mbar_try_wait(ack, 0)) {}               // wait for the peer to tell me the same: only then may the source face be rewritten
/* ... rewrite sSrc; fence.proxy.async.shared::cta; __syncthreads ... */
if (tid == 0) bulk_s2s(sR2, sSrc, NB, data2, peer);
while (!mbar_try_wait(data2, 0)) {}
// exit: every data barrier I wait on has flipped (nobody is still writing me) + the ack arrived (nobody is still reading me) => no second cluster rendezvous needed

// ---- Form B: all-to-all, receive barrier init(CSZ), the receiver does not arm
if (tid == 0) { mbar_init(rbar, CSZ); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
/* receive face ready before the rendezvous (sentinel prefill) */ __syncthreads(); cluster_sync();
/* only after the rendezvous: read gmem, write the source face, fence.proxy.async, __syncthreads */
const uint32_t tx = 8 + (has_payload ? SLOT_B : 0);                      // the bytes I actually send to this peer
if (lane == 0) for (int h = warp; h < CSZ; h += NWARP) {
  mbar_arrive_expect_tx_remote(mapa_u32(rbar, h), tx);                   // arrive + expect_tx in one instruction
  // 8 B small header: st.async also books complete_tx on the same peer barrier (dst and mbar both via mapa)
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
               :: "r"(mapa_u32(tag + rank * 8, h)), "r"(a), "r"(b), "r"(mapa_u32(rbar, h)) : "memory");
  if (has_payload) bulk_s2s(recv + rank * SLOT_B, src + h * SLOT_B, SLOT_B, rbar, h);
}
while (!mbar_try_wait(rbar, 0)) {}              // flip => all bytes of the CSZ senders have landed
```

## Gotchas

1. **`dst` and `mbar` must both be `mapa`'d to the peer; `src` is not.** Completion is booked on the peer's barrier; the sender has nothing to wait on for this instruction.
2. **`cp.async.bulk.commit_group` + `wait_group.read 0` does not guard copies that use the mbarrier completion mechanism.** Probe cell `s2s_waitgroup`
   (commit + wait_group.read 0 after issue, then immediately rewrite the source face): the peer data read this time was clean. But the PTX ISA states that a bulk async-group only collects copies with the `.bulk_group` completion mechanism — an mbarrier-completed s2s is not among them, so this clean result can only count as a timing coincidence and **must not be used as a guard**; the `s2s_noack` cell with no guard at all: the peer's round-1 data is dirty (max_diff 63997), which shows the reuse window really exists and the ack guard discriminates.
   The certified drain is a pairwise ack: the receiver waits until its own data barrier flips, then sends one remote arrive to the sender's ack barrier.
3. **If the source face was just written by ordinary `st.shared`, issue `fence.proxy.async.shared::cta` + `__syncthreads()` before the copy.**
4. **In Form A the arm must come before the cluster rendezvous** (after the rendezvous the peer may send at any time); **the receive face and barrier must be ready before the rendezvous**.
5. **The destination operand of a remote `arrive` / `arrive.expect_tx` must be the sink `_`** (no state can be returned when the barrier is not in this CTA).
   The spelling without `.sem.scope` passes.
6. **In Form B the receiver must never arm as well.** Probe cell `bd` (the receiver does an extra local `expect_tx(total bytes)` before the rendezvous): under all five payload patterns all 64 CTAs had `flipped = 0` —
   the symptom is a hang (FAIL under a bounded wait), not wrong numbers.
7. **In Form B the order of the sender's `arrive.expect_tx` and the copy is not load-bearing** (`b1` arrive first and `b2` arrive later both pass): a complete_tx
   arriving first only makes the tx-count temporarily negative; the phase cannot complete while pending arrivals are non-zero.
8. **In Form B put only init + fence + receive-face readiness before the rendezvous**: `cluster_wait` waits for the slowest peer, so every DRAM round trip before the rendezvous is
   paid by the whole cluster at the slowest value.
9. **The destination address and barrier of `st.async` must be in the same (destination) CTA**, both addresses via `mapa`; `.v2.b32` is one 8 B instruction.

## Sources

- PTX ISA, `cp.async.bulk` (`.shared::cluster.shared::cta` form + `.mbarrier::complete_tx::bytes`),
  `cp.async.bulk.commit_group / wait_group` (tracks only the bulk_group completion mechanism), `mapa`, `mbarrier.arrive` (`.shared::cluster`
  remote form, sink `_`), `mbarrier.arrive.expect_tx`, the mbarrier phase-completion condition (pending arrivals and tx-count both reach zero),
  `st.async`, `fence.proxy.async`, `fence.mbarrier_init`, `barrier.cluster`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide, Distributed Shared Memory: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- GPU-certification probe — `PROBE PASS max_diff=0` (an earlier run passed as well). Form A: cluster 8, 64 CTAs, two copies of 16 KB + 128 B per round, the same source face reused for two rounds, ack guard, no second cluster rendezvous; both rounds bitwise identical to the peer's source data, every CTA received the ack. Form B: cluster 8 all-to-all, each sender sends every peer (itself included) an 8 B `st.async` small header + an optional 2048 B payload; all five "who has a payload" patterns (all / none / even / one per cluster / pseudo-random) exact word for word with 64/64 flips (`b1` arrive first, `b2` arrive later); the `bd` negative control is in gotcha 6. SASS: s2s reads `UBLKCP.S.S`, the remote arrive.expect_tx reads `SYNCS.ARRIVE.TRANS64.RED`, `st.async` reads `STAS.64`.
