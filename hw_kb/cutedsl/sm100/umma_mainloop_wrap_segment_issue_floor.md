# umma_mainloop_wrap_segment_issue_floor (the "wrap segment" of a TMA → UMMA CTA-pair mainloop: the issue path between a wait returning and the next wait starting — how to measure it on the GPU, and what its floor consists of)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Neighbouring entries: [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md) (counting SASS chain length: unrolling removes stage bookkeeping),
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) (the rolled ring).
>
> **Status: ✅ Verified** (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=0.0`,
> second round of this certification, six contracts passed in the same round).
> This entry is a **measurement method + floor readings**, not a speed-up recipe.

## Purpose

In a warp-specialized TMA → UMMA mainloop, the period of each role per k-block = **wait** (`try_wait` spinning or suspended on `full` / `empty`)
+ **wrap segment** (the role's own instruction stream between the wait returning and the next wait starting).
The content of the wrap segment is fixed:

| Role | What is in the wrap segment |
|---|---|
| MMA issuer (leader) | `n_umma` `tcgen05.mma` (each = computing the two A/B smem descriptors from the runtime stage + issue) + setting `ACCUMULATE` + one `tcgen05.commit` (the handle's `release()`, with a multicast mask in the pair form) + `PipelineState` advance + back edge |
| TMA producer | this beat's `cp.async.bulk.tensor` issues for each operand + advance + back edge (the leader's `arrive.expect_tx` is hidden inside `acquire_and_advance()` and counts on the "wait" side) |

This entry provides two things: (1) a `%clock64` stamping method that **wraps only the wait**, cleanly separating the wrap segment from the wait;
(2) the GPU-measured **floor** of the wrap segment and its decomposition (cycles per UMMA, fixed cycles for commit + advance + back edge),
and the effect of "rolled / not rolled" on it. With these readings you can tell whether the wrap segment on your own timeline is already at the issue floor;
if it is, "reordering the wrap segment / changing the ring spelling" will not make it shorter.

## When needed (trigger)

- On the timeline both roles' waits are non-zero, the busy segment reads 0, all remaining cycles are in the **unmarked wrap segment**,
  and you are about to shrink the wrap segment by "moving the commit earlier / moving the stage advance later / precomputing descriptors before the wait".
- You have already rolled the mainloop (`cutlass.range(n, unroll=1)`) or hand-written the ring,
  the NCU duration does not move out of the noise, and you want to know how much of the wrap segment is left to shrink.
- You want to estimate how "how many UMMAs per stage (BK)" amortizes the fixed per-k-block overhead.

**Whole-kernel readings (one pair tree)**: on the timeline the MMA takes about 145 cycles per beat, about 90 of them waiting; changing the mainloop to rolled `unroll=1` landed twice,
both with gate 0.0097–0.0098 ms (0.0097 ms before landing) — the "rolled is a dozen or so cycles longer than not rolled" of the table below did not show up at whole-kernel level. Rolling the ring has whole-kernel readings in both directions (see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md):
rolled on both sides was 3.45% slower on one tree; on another tree the unrolled MMA side was 16% slower on one workload); the wrap segment itself is not where its gain comes from.

## Minimal runnable example

### (A) Stamping: wrap only the wait; wrap segment = total duration − sum of waits

```python
import cutlass, cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
Int64 = cutlass.Int64

@dsl_user_op
def clk(*, loc=None, ip=None):                       # %clock64: SM cycles; volatile, the compiler will not move it across the wait
    return Int64(llvm.inline_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l",
                                 has_side_effects=True, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

# MMA issuer (leader CTA), PipelineTmaUmma ring
if warp_idx == MMA_WARP:
    if is_leader_cta:
        wsum = Int64(0)                              # loop-carried: declared as Int64 outside the loop
        t_start = clk()
        for k in cutlass.range(num_k_tiles, unroll=1):
            tb = clk()
            hf = ab_consumer.wait_and_advance()      # only the wait is wrapped by tb / te
            te = clk()
            wsum = wsum + (te - tb)
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                kc = (None, None, kb, hf.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[kc], tCrB[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            hf.release()
        t_end = clk()
        # mean wrap per k-block = (t_end - t_start - wsum) / num_k_tiles
        if tidx % 32 == 0:
            gStat[cta, 0] = wsum
            gStat[cta, 1] = t_end - t_start

# TMA producer, same method: tb / te wrap ab_producer.acquire_and_advance(); wrap = this beat's cute.copy calls + advance + back edge
```

Each trip adds only two `CS2R` (reading `%clock64`) and one 64-bit add, with no store inside the loop; the statistics are written once outside the loop.
**Do not** write a gmem / smem timestamp every trip inside the loop — that store itself lands in the wrap segment.

### (B) Measured readings (GPU certification probe: fp16, CTA pair UMMA (`CtaGroup.TWO`, M = 256), A multicast within the cluster, B pair-private, shallow ring, static shape)

Median wrap segment per k-block (cycles, `%clock64`; each cell is the median across CTAs, brackets are min..max; cold = the first
launch after compiling, warm = the second and later). Two K values, the number of beats differing by 2×. Variants: `r64` = BK 64 (4 UMMAs per stage), rolled with `unroll=1`;
`u64` = BK 64 **without** `unroll=1` (compiler-default partial unrolling: in SASS the MMA loop body is one copy per 2 beats, the TMA loop body one copy per 4 beats);
`r128` = BK 128 (8 UMMAs per stage), rolled.

| Variant | K | MMA wrap cold | MMA wrap warm | TMA wrap cold | TMA wrap warm |
|---|---|---|---|---|---|
| r64 rolled | small | 72.1 [71.9..72.4] | 72.2 [72.2..72.2] | 52.0 [51.9..52.9] | 52.2 [51.9..52.6] |
| r64 rolled | large | 71.1 [71.0..71.2] | 71.1 [71.1..71.2] | 50.5 [50.4..51.0] | 50.5 [50.4..50.8] |
| u64 not rolled | small | 60.2 [57.5..60.8] | 57.9 [57.4..57.9] | 42.5 [42.4..45.4] | 42.7 [42.2..43.0] |
| u64 not rolled | large | 57.9 [56.5..58.1] | 56.5 [56.5..56.7] | 37.6 [37.3..39.2] | 37.8 [37.3..38.2] |
| r128 rolled | small | 146.6 [141.6..148.0] | 117.5 [117.5..130.4] | 87.5 [86.4..90.0] | 86.9 [86.6..87.0] |
| r128 rolled | large | 129.1 [125.6..131.5] | 117.8 [116.2..120.6] | 84.8 [84.2..86.0] | 84.8 [84.2..85.4] |

MMA wrap decomposition (the two rolled variants, per-UMMA = (r128 − r64) / 4, fixed = r64 − 4 × per-UMMA):

| K | per-UMMA | fixed (commit + advance + back edge) |
|---|---|---|
| small, warm | 11.3 | 26.9 |
| large, warm | 11.7 | 24.5 |
| small, cold | 18.6 | −2.4 (the cold first launch is noisy; the fit is unusable) |
| large, cold | 14.5 | 13.1 |

How to read it (warm is authoritative):

- **Each UMMA costs about 11–12 cycles in the wrap segment** (runtime addresses of the two smem descriptors + issue), **commit + advance + back edge
  about 25 cycles**. The MMA wrap floor with 4 UMMAs per stage ≈ 4 × 11.5 + 25 ≈ 71 cycles; with 8 the measurement is ≈ 118 cycles —
  doubling the UMMAs per k-block adds only ≈ 46 cycles of wrap, and the fixed 25 cycles are amortized.
- **Not rolled is shorter than rolled**: at the same BK 64, the MMA wrap of `u64` is ≈ 57 vs ≈ 71 rolled (−14 cycles), TMA wrap ≈ 38–43 vs ≈ 51 (−8..−14 cycles).
  Once rolled, the barrier / descriptor addresses become runtime `base + stage` arithmetic, and that lands in the wrap segment.
  **So "rolling up an unrolled ring" does not shorten the wrap segment; it lengthens it by about 10–14 cycles**; if rolling the ring gains anything,
  the gain does not come from the wrap segment (whole-kernel readings both ways: see `mbarrier_ring_rolled_loop`).
- The wrap segment is essentially unchanged between warm and the two K values (`u64`'s TMA is about 5 cycles shorter at large K; `r128` is longer when cold); it is the fixed cost of the issue path;
  what changes with cold/warm is mainly the wait (same-job median wait per k-block: BK 64 variants cold ≈ 265–314, warm ≈ 205–252 cycles).
- Verdict: if the MMA wrap on your own timeline is already around ≈ UMMAs per stage × 11.5 + 25 (rolled), or ≈ 14 less (not rolled), the wrap segment is at the issue floor;
  reordering commit / advance / descriptors cannot move it; the remaining cycles are in the wait, and the supply side is what must get shorter.

## Gotchas

1. **Stamp around the wait, not around the issue.** If written as "stamp when the wait returns, stamp again when the next wait returns", what you measure is the whole period,
   with wait and wrap mixed together. Only "wrap the wait + subtract from the total duration" isolates the wrap segment.
2. **`%clock64` is this SM's cycle counter; it is not comparable across CTAs / SMs.** Only compare differences within the same thread; to convert to ns use
   the same job's SM frequency; do not use `%globaltimer` (ns granularity, coarse resolution, and a wrap segment is only a few dozen cycles).
3. **The leader's `arrive.expect_tx` is inside the producer's `acquire_and_advance()`** — with the stamping of (A) it counts on the producer's
   "wait" side; to look at expect_tx separately you need a hand-written ring that moves it out of the wrapped stretch.
4. **Rolled / not rolled does not change the number of issues in the wrap segment, but it changes its length — not rolled is shorter.** Not one of the `n_umma` MMAs or the one commit
   can be saved; what rolling adds is the runtime stage-address arithmetic (MMA side ≈ +14 cycles, TMA side ≈ +8..14 cycles, see (B)).
   Do not expect "rolling the ring" to shrink the wrap segment. **Omitting `unroll=1` does not mean fully unrolled either**: the compiler only replicates the loop body into a copy per few beats; for acceptance count
   `UTC*MMA` / `UTMALDG` in SASS.
5. **To amortize the fixed overhead over fewer commits / waits, put more UMMAs per stage (larger BK)** — that is a structural change of ring depth / stage bytes,
   and the smem budget doubles with it; first use the decomposition of (B) to work out how many cycles it saves before deciding.
6. **ACCUMULATE is loop-carried state in a rolled loop**: `tiled_mma.set(..., True)` is in the same region as `cute.gemm` (the loop body),
   so the first MMA of the first trip reads the False set at construction; a persistent spelling must explicitly `set(..., False)` before each new tile.
7. **Pass the scalar field of a `@cute.struct` directly to `utils.TmemAllocator(...)` / `two_cta_tmem_dealloc_mbar_ptr=`**; the error from writing `.ptr`: see
   gotcha 10 of [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) (the first round of this certification hit it too).
   Create both pipelines with `create(..., defer_sync=True)`, finish constructing the `TmemAllocator` too (it inits the pair-dealloc mbarrier),
   then do one `mbarrier_init_fence()` + `cluster_arrive()` / `cluster_wait()` for all of them.

## Sources

- **PTX ISA**: special register `%clock64`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64>;
  `tcgen05.mma` / `tcgen05.commit`: <https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-instructions>;
  mbarrier: <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
- **CuTe DSL public API symbols**: `cutlass.range(..., unroll=)`, `cutlass.range_constexpr`, `cutlass.pipeline.PipelineTmaUmma`
  (`make_participants`, `acquire_and_advance`, `wait_and_advance`, the handle's `release` / `index` / `count` / `barrier`, `tail`),
  `PipelineUmmaAsync`, `cute.gemm`, `tcgen05.Field.ACCUMULATE`, `cutlass.cutlass_dsl.dsl_user_op`,
  `cutlass._mlir.dialects.llvm.inline_asm`.
- **GPU certification probe**: fp16 GEMM, CTA pair UMMA, A multicast within the cluster, shallow ring, static shape; three variants
  `r64` (BK 64, `unroll=1`) / `u64` (BK 64, no unroll) / `r128` (BK 128, `unroll=1`), two K values.
  Contracts: V×3 C bit-equal to an fp32 einsum; S×2 the rolled variants' PTX `try_wait` / SASS `PHASECHK` counts identical across the two K values and non-zero;
  W all stamps of both roles readable. **Status: ✅ Verified (`PROBE PASS max_diff=0.0`, second round of this certification, six contracts passed in the same round: V r64 / V u64 / V r128 0.0; S r64, S r128 PTX `try_wait` 7/7, SASS `PHASECHK` 29/29 unchanged across the two K values; W stamps readable).**
