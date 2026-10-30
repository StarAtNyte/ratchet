# tmem_ld_epilogue_issue_order (the order of `tcgen05.ld` / `tcgen05.wait::ld` / conversion / `st.shared` / `fence.proxy.async` / rendezvous / TMA store in a chunked epilogue: which edges are hard, whether in-flight readbacks are stretched by them, what source reordering leaves in SASS)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cells `chain` / `order` plus a SASS order census; the probe is not included in this release;
> verdicts and readings under Sources). The mechanism is the same as the same-named entry in the CuTe DSL tree; the readings were re-taken on the CUDA spelling, and the numbers differ from the DSL entry's.
> **The gain is not certified**: there is no whole-kernel reading.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For how the four legs are assembled see [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md).

## Purpose

### §1 The hard ordering edges of one epilogue chunk (PTX ISA)

| Step | Instruction | Must come after | What it blocks |
|---|---|---|---|
| ① readback | `tcgen05.ld` | the wait on the accumulator full + `tcgen05.fence::after_thread_sync` | the destination registers cannot be read before ② |
| ② wait for readback | `tcgen05.wait::ld` | ① | **all** `tcgen05.ld` previously issued by this thread, regardless of which, with no counted "wait until the k-th" form |
| ③ conversion | `cvt.rn.bf16x2.f32` etc. | ② | — |
| ④ write slot | `st.shared` / `stmatrix` | ③; and the slot's TMA store of the previous round has been read (⑧) | — |
| ⑤ cross proxy | `fence.proxy.async.shared::cta` | this thread's **last** ④ of this chunk | ⑥ |
| ⑥ rendezvous | `bar.sync id, N` (or mbarrier) | ⑤ | ⑦ |
| ⑦ issue store | `cp.async.bulk.tensor` S2G + `cp.async.bulk.commit_group` (one thread) | ⑥ | — |
| ⑧ return slot | `cp.async.bulk.wait_group.read N` (only on the thread that issued the store) | ⑦ | the next ④ writing that slot (passed to other threads through one ⑥) |

### §2 Are in-flight readbacks stretched by X (probe readings)

Dependency-chain method: the address of the i-th `tcgen05.ld.32x32b.x32` depends on the data read back by the (i−1)-th (adding a quantity the compiler cannot prove to be 0); loop time of 256 iterations / 256
= the "readback → data available" latency plus whatever part of X is on the critical path. X is inserted between `ld` and `wait::ld` (128 threads, 1 CTA, best of 3):

| X | cycles / dependent readback |
|---|---|
| none | 168 |
| `st.shared.v4` (one) | 166 |
| `bar.sync 1, 128` | 184 |
| `fence.proxy.async.shared::cta` | 192 |
| all three | 199 |

How to read it: in-flight `tcgen05.ld` **largely overlaps** with these X — `st.shared` hides completely under the readback latency, `bar.sync` / `fence.proxy.async` add only 16 / 24 cycles,
all three together 31; none of the X puts the whole readback latency onto the critical path. So "issue the next chunk's `ld` before this chunk's face writes / fence / rendezvous" is a real overlap in hardware,
but how much it is worth must be measured on your own kernel.

### §3 What ptxas has already moved for you (SASS readings)

- `fence.proxy.async.shared::cta` is two instructions in SASS: `MEMBAR` + `FENCE.VIEW.ASYNC.S`.
- `tcgen05.wait::ld` is not a standalone instruction: it lands on the first instruction that reads the destination registers (a scoreboard wait). So as long as there is no rendezvous in between, ptxas moves the next chunk's `LDTM` forward by itself.
  In the probe's chunked quantization ([`tmem_chunk_e4m3_quant_chain`](tmem_chunk_e4m3_quant_chain.md)), the two sources "explicitly issue the next chunk's `ld` early" and "each chunk's `ld` immediately followed by `wait`"
  compile to key instruction sequences that are **identical item by item** (both `LDTM ×2` at the start, then one `LDTM` per chunk) — the source rotation is a no-op; whereas in the spelling with one `div.rn.f32` per chunk, `FCHK` / `CALL`
  cut each chunk apart and `LDTM` is no longer hoisted.
- Conclusion: source reordering between two rendezvous often does not change SASS; what reliably changes SASS is "moving an instruction across a rendezvous", "changing the number of rendezvous" or "removing what cuts basic blocks" (division slow paths, branches).

## When needed (trigger)

- You plan to do any of "cross-chunk rotation / interleaving face writes with conversion / replacing the rendezvous with an mbarrier", or on the timeline the wrap in each epilogue chunk period is longer than the rendezvous wait;
- in static SASS each `LDTM` is immediately followed by an instruction reading its destination registers, and you suspect the readback latency is on the critical path.

## Minimal runnable example

Dependency-chain method (same spelling as the certification probe):

```cpp
uint32_t z = 0, acc = 0;
const unsigned long long t0 = clk();
#pragma unroll 1
for (int it = 0; it < 256; ++it) {
  uint32_t v[32];
  tm_ld32x32b_x32(tbase + ((uint32_t)(32 * warp) << 16) + z, v);   // the address depends on the previous round's data
  /* X: st.shared / fence.proxy.async.shared::cta / bar.sync 1, 128 */
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  acc += v[it & 31];
  z = (v[0] == 0xDEADBEEFu) ? 1u : 0u;                              // the payload is small integers: always 0, unknown to the compiler
}
const unsigned long long per_readout = (clk() - t0) / 256;
```

## Gotchas

1. **Do not drop each chunk's `wait::ld`**. The SASS scoreboard may happen to cover it, but PTX defines `tcgen05.ld` as asynchronous, with completion observed by `wait::ld`; the acceptance payload must encode coordinates — a constant payload cannot reveal an early read.
2. **The safety condition for slot reuse depends only on which two rendezvous `wait_group.read N` falls between**: the thread that issues the store executes `wait_group.read N` after chunk c and before chunk c+1's rendezvous;
   the slot `(c+2) % S` that other threads write after chunk c+1's rendezvous needs group `c+2−S` to have been read, i.e. **`N ≤ S − 2`** (derived from bulk-group semantics).
3. **`wait_group` executes only on the thread that issued the store** — bulk-group accounting is per thread; the other threads inherit slot safety from the next rendezvous.
4. **`fence.proxy.async` once per thread per chunk, after the chunk's last write and before the rendezvous**; inserted between pieces it covers only the writes before it (TMA reads stale bytes, silently wrong).
5. **The `%clock64` stamping trap**: clock reads have no data dependency, and ptxas may schedule them before the scoreboard wait — "read the clock once after the ld and once after the wait" cannot measure the readback latency
   (the probe's `order` fact line is exactly this failure mode: the difference is only a few cycles). To measure latency use the dependency chain above.

## Sources

- PTX ISA: `tcgen05.ld` / `tcgen05.wait::ld`, `fence.proxy.async`, `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group.read`, `bar.sync`, `%clock64` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe: cells `chain` (the dependency chains for the five X with per-thread checksums) and `order` (checksums + the stamping-trap fact line),
  plus a SASS order census, verdict
  `PROBE PASS`: `C7-chain-data` (all 640 per-thread checksums right under the five X), `C6-order-data`; dependency-chain readings in §2, SASS readings in §3.
