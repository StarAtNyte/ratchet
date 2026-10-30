# tma_refill_post_wait_hoist (TMA producer refill segment: compute the next beat's addresses / coordinates / tx bytes before the empty wait; moving only the source gets sunk back by the compiler, so pin with an empty `asm volatile("" : "+r"(v))`)

> **Certification source: B200 GPU-certified in CUDA mode, ✅ Verified** (certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=0.0`).
> What is certified is the SASS window shape and numerical equivalence; **the whole-kernel gain is not certified** — a same-job whole-kernel A/B in a real kernel was flat (see "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. Neighboring entries: [`umma_issue_form`](umma_issue_form.md) (how to choose the thread for a single-thread issue role; that probe does not cover `UTMALDG`),
> [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) (the wait itself: probing empty one beat ahead).

## Purpose

In a warp-specialized TMA → consumer ring, the producer's per-beat refill segment is "empty wait returns → this beat's last `UTMALDG` issued". When the moment the consumer releases the slot
is on the critical path (the producer really waits on empty), every instruction in this segment adds directly to the "slot released → data starts in flight" latency.

This segment really only needs: one `mbarrier.arrive.expect_tx` + one `cp.async.bulk.tensor` per box. Everything else — the full barrier address,
the stage tx byte count (including runtime row counts), each box's smem destination address, the k coordinate — **does not depend on the wait's result** and can be computed by the whole warp before the wait.

This entry gives the spelling: compute these values before the wait, and **pin** each one before the wait with an empty `asm volatile("" : "+r"(v))`; after the wait returns,
the elect body contains only expect_tx + bare TMA. The numerics are bytewise unchanged; only the post-wait window in SASS changes.

## When needed (trigger)

- On the timeline, the producer segment from the end of `wait_empty` → all of this beat's TMAs issued is longer than "1 expect_tx + N TMAs" should cost;
- a critical-path probe shows the producer refill segment is on the critical path (artificially slowing the producer refill slows the whole kernel in step);
- in SASS, after the empty wait's `SYNCS.PHASECHK.TRANS64.TRYWAIT` and before the first `UTMALDG`, there is a string of `IMAD` / `LEA` / `UIADD3` / `R2UR`;
- some box's row count / stride / tx bytes are runtime values, recomputed after the wait every beat.

**Anti-trigger (issuer spelling)**: when the producer is written as `if (lane == 0) { the whole ring loop }`, each `UTMALDG` in SASS may be wrapped in its own ring of `ELECT` / `R2UR` / back jump;
that is a question of issuer choice, not this entry; and do not change the issuer spelling just because you see that ring of `ELECT`. Whole-kernel readings: on the same single-CTA tcgen05 GEMM, changing the `lane == 0`
producer to the elect spelling twice gave gate ratios −1.3% / −0.7% (cross-job, at the edge of noise, both unfavorable) — read your own SASS and timeline first.

**Whole-kernel reading (in a real kernel, flat)**: on a single-CTA tcgen05 GEMM whose refill segment really is on the critical path, same-job interleaved A/B (geometric mean over several problem sizes, cold L2):
with the refill loop fully unrolled, pinned / unpinned 9.350 / 9.341 µs; rolled, 9.354 / 9.371 µs — all four combinations within 0.3%.
**What this entry saves is a few integer instructions per beat; do not expect it in whole-kernel readings; accept the change with a same-job A/B.**

## Minimal runnable example

```cpp
// fk-cuda-cflags: (no extra flags)
// Empty asm + read-write operand: this value must be in a register at this point of the volatile asm sequence,
// so it cannot be sunk below the later asm volatile (the wait) and computed there. Emits no SASS instruction.
__device__ __forceinline__ void pin(uint32_t& v) { asm volatile("" : "+r"(v)); }
__device__ __forceinline__ void pin(int& v)      { asm volatile("" : "+r"(v)); }

// producer: rolled ring loop (S slots, KB boxes per operand per beat); how the issuer is chosen is outside this entry (see the whole-kernel reading in "When needed")
#pragma unroll 1
for (int j = 0; j < T; ++j) {
  const int s = j % S;
  // ---- before the wait: all operands needed by this beat's issue
  uint32_t fb = full0 + 8u * s;
  uint32_t tx = stage_tx;                        // may be a runtime value
  uint32_t dX[KB], dY[KB];
  int kc[KB];
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) {
    dX[kb] = sX + (s * KB + kb) * XBOX;
    dY[kb] = sY + s * Y_STAGE + kb * y_kb_bytes; // runtime stride
    kc[kb] = (j * KB + kb) * BK;
  }
  pin(fb); pin(tx);
#pragma unroll
  for (int kb = 0; kb < KB; ++kb) { pin(dX[kb]); pin(dY[kb]); pin(kc[kb]); }

  if (j >= S) mbar_wait(empty0 + 8u * s, ((j - S) / S) & 1);   // hint-free try_wait loop

  // ---- after the wait: only expect_tx + bare TMA remain
  if (is_issuer) {                                // how the issuer is chosen is outside this entry
    mbar_expect_tx(fb, tx);                       // mbarrier.arrive.expect_tx.shared::cta.b64 _, [fb], tx;
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) tma_load_2d(dX[kb], &tmX, kc[kb], x0, fb);
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) tma_load_2d(dY[kb], &tmY, kc[kb], y0, fb);
  }
}
```

`tma_load_2d` is `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [dst], [tmap, {c0, c1}], [bar];`
(the tensor map is the address of a `const __grid_constant__ CUtensorMap` parameter); `mbar_wait` is a hint-free `mbarrier.try_wait.parity` loop
(phase rules in [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md)).

## Gotchas

Probe readings (four spellings, two runtime box row counts, all outputs bytewise identical max_diff=0.0). "Window" = from the empty wait's `TRYWAIT` to this beat's last `UTMALDG`;
"post-wait cycles" = the difference between two `%clock64` reads by the issuing thread, after the wait returns and after the last TMA (median over refill beats, identical for both row counts):

| Spelling | Window instructions | Address integer arithmetic (`UIADD3` / `UIMAD` / `ULEA`) | `R2UR` | `UMOV` | Post-wait cycles | Probe single launch (two row counts) |
|---|---|---|---|---|---|---|
| computed after the wait inside the elect body (inline) | 52 | 13 | 2 | 18 | 105 | 21.06 / 27.50 µs |
| computed before the wait + `pin()` | 51 | **4** | **10** | 18 | **98** | 20.54 / 27.49 µs |
| computed before the wait, no `pin()` | 53 | 12 | 4 | 18 | 111 | 20.86 / 27.28 µs |
| computed before the wait + `pin()` + elect hoisted out of the loop (`leader` register) | 46 | 4 | 10 | 18 | **79** | 21.10 / 27.50 µs |

0. **Know the magnitude**: pinning cuts the address integer arithmetic in the window from 13 to 4, but `R2UR` goes from 2 to 10 — `pin()` pins the values in **vector** registers,
   which still have to be moved into uniform registers before issue; the window total shrinks by only 1 instruction (52 → 51), post-wait cycles 105 → 98. The bulk of what remains in the window is the `UMOV`s that move coordinates /
   smem addresses into grouped uniform registers before each `UTMALDG` (18 in all four spellings) — moves required by the `UTMALDG` operand format, which precomputation cannot remove.
1. **Just moving the computation before the wait in the source is not enough**: the wait is `asm volatile(... : "memory")`, and the `memory` clobber constrains only memory accesses;
   pure integer arithmetic can be sunk by the compiler below the wait, right next to its use. The unpinned "hoisted" spelling still has 12 address operations in the window, longer than computing in place (53 vs 52 instructions, 111 vs 105 cycles).
   An "empty volatile asm with read-write register operands" such as `pin()` keeps these operations before the wait (for the other face of the same phenomenon see
   [`branchfree_masked_elementwise_pass`](branchfree_masked_elementwise_pass.md) gotcha 3: independent ALU work written before a wait gets partly moved after it by ptxas).
2. **Hoisting the elect out of the loop** (`leader = elect_one()` kept in a register, `if (leader)` in the loop): the shortest window (46 instructions, 79 cycles; no per-beat
   `ELECT` / `BSSY` / `WARPSYNC` left in the window); but the probe's whole launch did not get faster for it (21.10 / 27.50 µs vs 20.54 / 27.49 µs). A shorter window does not mean faster.
   On the same kernel, changing the producer to this shape (elect once outside the loop, the whole warp runs the loop, `if (leader)` issues) made the whole kernel 0.7% slower (cross-job ratio);
   a similar rewrite on another kernel (elect once, whole warp runs the loop, predicated issue) was 1.09% slower in the same job.
3. **Even when the token is true / the slot is already empty, `arrive.expect_tx` is required**; see [`pipeline_tma_producer_try_acquire_lookahead`](pipeline_tma_producer_try_acquire_lookahead.md) gotcha 2.
4. **Meaningful only for rolled loops with a runtime stage**: when the producer is already hand-unrolled by S (stage a compile-time constant), barrier / slot addresses already fold into "base + immediate",
   and pinning with an empty asm before the wait saves no cycles (wrap segment 210.7 vs 210.9); see [`mbarrier_ring_unrolled_const_addr`](mbarrier_ring_unrolled_const_addr.md) gotcha 7.
5. This entry certifies only the SASS window shape and numerical equivalence: what it saves is a few integer instructions after each beat's wait; verify the magnitude on your own kernel's timeline
   (look at the `wait_empty.end → refill.issued` segment and a same-job whole-kernel A/B).

## Sources

- PTX ISA: `cp.async.bulk.tensor`, `mbarrier.try_wait` / `mbarrier.arrive.expect_tx`, `elect.sync` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ documentation, Inline PTX Assembly (`asm volatile`, the operand constraint `"+r"`, the `"memory"` clobber) —
  <https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html>
- CUDA Driver API: `cuTensorMapEncodeTiled`.
- GPU-certification probe: `PROBE PASS max_diff=0.0`:
  `H1-bytes-identical-all-variants`, `H2-hoist-pin-shrinks-post-wait-window arith_inline=36 arith_hoist_pin=35 elect_between_tma_hoist_pin=0`
  (the contract's "arith" sums the integer arithmetic and the `UMOV` / `R2UR` moves); the per-class counts and cycles / timings in the table are fact lines of the same job.
- Whole-kernel reading: same-job A/B (numerics of the four combinations `PROBE PASS max_diff=0.0630493`, same as the baseline).
