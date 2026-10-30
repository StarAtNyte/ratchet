# umma_mainloop_wrap_segment_issue_floor (the MMA warp's per-k-block "wrap segment" in a TMA → UMMA main loop: a `%clock64` stamping method that brackets only the wait; resident readings and breakdown on the CUDA spelling for three tiers: rolled ring, unrolled by stage, single-thread back-to-back issue; three shapes on a pair: "per-beat elect region / single-thread loop / single thread + unrolled by S")

> **Certification source: re-certified on a GPU probe in CUDA mode** (the `time` / `res` modes of a UMMA GEMM probe that is not included in this release; verdicts and readings under Sources).
> The measurement method is the same as the same-named entry in the CuTe DSL tree; **all readings were re-taken on the CUDA spelling, and both the numbers and the breakdown differ from the DSL entry's** (see "Purpose").
> This entry is a **measurement method + resident readings**, not a speed-up recipe.
> Incremental certification: three issue shapes on a pair (`cta_group::2`), a second GPU probe
> (`PROBE PASS max_diff=0`), certifying numerics and cycle readings; the whole-kernel reading was flat, no gain (see that section).
>
> One file per hardware interface. Operator-agnostic: mechanism only. Neighboring entries: [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md) (unrolled vs runtime stage),
> [`umma_raw_desc_issue`](umma_raw_desc_issue.md) (issue spelling).

## Purpose

In a warp-specialized TMA → UMMA main loop, the MMA warp's cycles per k-block = **wait** (`try_wait` on the full mbarrier) + **wrap segment** (from the wait's return to
the start of the next wait, this warp's own instruction stream: fence, `elect.sync`, NU `tcgen05.mma`, commit, `__syncwarp`, back edge). This entry gives:

1. a stamping method that **brackets only the wait**: wrap segment = total loop time − sum of waits (only two extra `%clock64` reads and one 64-bit add per beat, with the statistics written once outside the loop);
2. a **resident ring** to measure the issue cost: the producer fills the S stages only once, and the MMA warp reads the ready stages over and over (no waiting for k ≥ S), so what is measured is the pure issue cost;
3. the resident readings and breakdown the probe measured on the CUDA spelling (M = 128, N = 64, bf16, one K16 per instruction, single CTA, `%clock64` cycles, median of launches 1–3):

| Resident ring, per k-block | NU = 1 | NU = 2 | NU = 4 |
|---|---|---|---|
| one `elect.sync` region + `__syncwarp` per k-block (the common spelling) | 281 | 288 | 295 |
| as above, with `tcgen05.fence::after_thread_sync` removed for k ≥ S | 281 | 288 | 295 |
| as above, with one extra `tcgen05.commit` per k-block (hitting a dummy mbarrier that never completes) | 281 | 288 | 295 |
| unrolled by S = 4 / 8 (the stage number becomes a compile-time constant; **still** one `elect.sync` region + `__syncwarp` per k-block) | — | — | **229 / 214** |
| **one selected thread runs the whole resident loop** (no per-k-block elect / reconvergence) | **53** | **101** | **197** |

How to read it:
- With one thread issuing back to back, each N = 64 UMMA takes about **48 cycles** (53 / 101 / 197, linear in NU) — that is tensor-pipe throughput, not issue overhead.
- The runtime-stage rolled spelling (one `elect.sync` region + `__syncwarp` per k-block) costs ≈ 281–295 cycles per k-block, nearly independent of NU (from NU 1 to 4
  only 14 cycles more), 5.3× (NU = 1) / 1.5× (NU = 4) slower than single-thread back-to-back issue; removing the per-k-block fence or adding an extra commit makes no measurable difference.
- **≈ 280 is not the floor**: in the same job, unrolling by S (still one elect / reconvergence per k-block) brings the resident reading down to 229 / 214 (S = 4 / 8, 22–28% less),
  and single-thread back-to-back issue to 197. The source of the difference between rolled and unrolled (67–82 cycles) was not isolated (inference: runtime stage number, `k % S`, computing the phase, and the back edge;
  see [`umma_issue_chain_precomputed_desc`](umma_issue_chain_precomputed_desc.md)); the difference between unrolled and single-thread (16–31 cycles) is an upper bound on the overhead of one
  elect / reconvergence (fence included) per k-block. Putting more UMMAs per stage (a larger BK) likewise amortizes the fixed per-k-block overhead.
- At N = 256 (4× the work per MMA) the resident reading is 508 cycles / k-block, already dominated by tensor throughput.

**Streaming ring** (TMA refilling the ring as usual, KT = 8): the common spelling totals 459 per k-block (wait 136 + wrap 324, NU = 4); putting the whole loop (including `try_wait`, fence, MMA, commit)
into **one** selected thread drops the wrap segment to 288 / 201 / 195 (NU = 4 / 2 / 1), but the wait grows correspondingly, and **the total per k-block does not get shorter** (NU = 4: 526 vs 459, NU = 1: 434 vs 429) —
when streaming the bottleneck is the feed. Judgment: in this probe, reordering commit / fence or changing how descriptors are passed makes no measurable difference; with the wrap segment near the rolled reading, unrolling by S or single-thread issue
can squeeze it further (resident readings in the table above), but in a streaming ring a shorter wrap segment does not necessarily buy a shorter total (gotcha 4); before changing the issue spelling, first confirm the bottleneck is not the feed.

### Three issue shapes on a pair (`cta_group::2`)

Another probe (pair geometry: 64 rows of A per CTA, the leader issues `tcgen05.mma.cta_group::2`, commit multicast `0b11` to `empty[s]` on both sides;
two instance sets with different N and S, NU MMAs per k-block) compares several spellings of the MMA warp loop; the numerics of the three below are bitwise identical across both instance sets and both resident / streaming modes:

- **A**: the whole warp enters the loop, each beat `if (elect_one()) { NU MMAs + commit }` + `__syncwarp()`, stage computed as `kb % S` (the "common spelling" of the table above);
- **B**: `elect_one()` once outside the loop, **one thread runs the whole loop** (doing its own `try_wait`), stage / phase / descriptor base maintained incrementally;
- **C**: B's loop unrolled by S (stage, phase and descriptor offsets all compile-time constants) + a runtime tail.

(B is (c) of [`umma_issue_form`](umma_issue_form.md), where its SASS and cadence are compared with a `lane == 0` branch on a single CTA; for the spelling with a separate elect per instruction see (e) there; not repeated in this section.)

Cycles per k-block (median; resident = the ring is filled once and the same set of stages read over and over, streaming = TMA refills the ring as usual; wrap = per beat "wait returns → next wait starts"):

| Shape | Instance a resident wrap / wait / total | Instance a streaming wrap / wait / total | Instance a streaming kernel µs | Instance b resident wrap / wait / total | Instance b streaming wrap / wait / total | Instance b streaming kernel µs |
|---|---|---|---|---|---|---|
| A | 307.5 / 158.8 / 466.6 | 310.2 / 202.0 / 512.2 | 22.45 | 308.1 / 173.4 / 480.9 | 312.8 / 216.7 / 529.2 | 24.83 |
| B | 262.7 / 238.2 / 500.9 | 265.1 / 275.1 / 539.2 | 22.70 | 262.1 / 240.9 / 502.9 | 263.8 / 289.2 / 553.3 | 25.95 |
| C | **139.9** / 229.3 / **369.2** | **166.7** / 268.7 / **435.4** | 22.48 | **118.5** / 233.3 / **351.5** | **174.0** / 287.8 / **461.3** | 24.51 |

How to read it:
- **Switching only to a single-thread loop (B) does nothing**: the wrap segment is 45–49 cycles shorter, the wait 67–79 cycles longer, and the total per k-block actually grows; consistent with the single-CTA streaming ring above,
  where "a single thread running the whole loop did not shorten the total". In this probe's resident mode the wait is not zero either; its source was not broken down.
- **What saves cycles in the probe is unrolling by S (C)**: the total per k-block is 21–27% (resident) / 13–15% (streaming) less than A.
- **Kernel time shows no difference**: the streaming kernels of the three spellings are 22.45–22.70 µs (instance a) and 24.51–25.95 µs (instance b, B slowest) — the issue cycles saved per beat
  turned into waiting. **Whole-kernel readings (the same class of pair GEMM kernel)**: changing the MMA warp to "pick one lane outside the loop, it runs all k-blocks, fully unrolled",
  gate 0.0120 → 0.0121 ms (+0.83%); then changing it to unrolled by S, 0.0121 ms; both within noise. **The per-beat cycles C saves in the probe are not a whole-kernel gain**;
  do not treat B / C as speed-up spellings; switching away from A requires a same-job whole-kernel A/B.

## When needed (trigger)

- On the timeline the MMA warp's wait is non-zero, the remaining cycles sit in an unmarked wrap segment, and you are about to squeeze the wrap segment;
- you want to estimate how "how many UMMAs per stage (BK)" amortizes the fixed per-k-block overhead;
- you want to know which of this probe's three tiers of readings (rolled ring / unrolled by S / single-thread back-to-back issue) your issue loop falls into.
- The pair's MMA warp is written as "the whole warp enters the loop, one elect region + `__syncwarp` per beat", and you want to know how much switching to a single-thread loop /
  single thread + unrolled by S would save (readings in "Three issue shapes on a pair") — both whole-kernel readings were within noise; the answer is nothing.

**Not the right remedy**: the wait takes most of each beat (feed-bound; look at the producer / ring depth first); the wrap segment is already close to the resident reading;
wanting to switch the pair's shape A to B / C for speed — the whole-kernel readings were flat (see "Three issue shapes on a pair").

## Minimal runnable example

### (A) Stamping: bracket only the wait

```cpp
unsigned long long t0 = clk(), tw = 0;                 // clk() = mov.u64 %0, %clock64; (asm volatile)
for (int k = 0; k < KT; ++k) {
  const int s = k % S;
  const unsigned long long w0 = clk();
  mbar_wait(full0 + 8 * s, (k / S) & 1);               // only the wait is bracketed by w0 / clk()
  tw += clk() - w0;
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (elect_one()) { /* NU tcgen05.mma + tcgen05.commit(empty[s]) */ }
  __syncwarp();
}
const unsigned long long t1 = clk();
// wrap segment per k-block = (t1 - t0 - tw) / KT; write the statistics once outside the loop, never write gmem inside the loop
```

### (B) Resident ring + single-thread back-to-back issue (the lowest tier of readings)

```cpp
for (int s = 0; s < S; ++s) mbar_wait(full0 + 8 * s, 0);   // the whole warp waits once for the ring to fill
asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
if (elect_one()) {
  const unsigned long long t0 = clk();
  for (int k = 0; k < KT; ++k) issue_mmas(k, k % S);        // issue only MMAs, reading the ready stages over and over
  // per k-block = (clk() - t0) / KT
}
__syncwarp();
```

## Gotchas

1. **Stamping brackets the wait, not the issue**: written as "stamp when the wait returns, stamp again when the next wait returns", it measures the whole period.
2. **`%clock64` is this SM's cycle counter**, not comparable across CTAs / SMs; compare only differences within one thread. Cold-start readings (the first launch) are inflated; take the later ones.
3. **A single thread running the whole loop must do its own `try_wait`**: an mbarrier wait can be done by a single thread (`mbarrier.try_wait` is not `.aligned`); `tcgen05.fence`,
   `tcgen05.mma` and `tcgen05.commit` are all thread-level instructions. The other 31 lanes go straight to the `__syncwarp()` after the loop. The probe's version of this spelling is numerically bitwise exact (contract `A1-exact-v64-f8`).
4. **A shorter wrap segment ≠ a faster whole kernel**: in a streaming ring the single-thread spelling shortened the wrap segment but not the total (the wait ate it). Before changing the issue spelling, first confirm the bottleneck is not the feed.
5. **How descriptors are passed makes no measurable difference**: two 32-bit registers assembled into a pair vs one 64-bit operand, resident 295.3 / 295.3, streaming 324.0 / 324.0.
6. **In a single-thread loop, the commit must be issued by the same thread that issued the MMAs, right after the stage's last MMA**: `tcgen05.commit` tracks the async tcgen05 operations previously issued by **the issuing thread**;
   a commit from another thread cannot track these MMAs. If `.sync.aligned` instructions (`tcgen05.alloc` /
   `dealloc` / `tcgen05.ld`) follow the single-thread loop, they must be executed by the whole warp after the `__syncwarp()`.
7. **When the descriptor base is incremented per stage, the wraparound must reset it to stage 0's base** (reset on the beat where the stage count wraps, increment by the stage bytes on the others); the start-address field of the descriptor low word is 14 bits of (smem address >> 4),
   and base + offset must not carry into the adjacent field.
8. **Unrolling by S must handle the `KT % S != 0` tail** (stage from 0, phase carrying on); change only the MMA warp's loop shape — producer, barrier counts,
   `expect_tx` bytes and TMEM layout stay the same — and run the numeric comparison first after the change.

## Sources

- PTX ISA: `%clock64`, `mbarrier.try_wait`, `tcgen05.mma`, `tcgen05.commit`, `tcgen05.fence`, `elect.sync` —
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- GPU-certification probe (`time` / `res` modes; `res` flags: 1 removes the per-k-block fence,
  2 single thread runs the resident loop, 4 one dummy commit per k-block; `time` flag 8 = single thread runs the whole streaming loop). Every timing setting also checked numerics (`exact=1`), verdict
  `PROBE PASS max_diff=0.0`: `A1-exact-v64`, `A1-exact-v64-f8` (the spelling with a single thread running the whole streaming loop is bitwise exact); the readings in the tables are timing fact lines of the same job.
- Three pair issue shapes: GPU-certification probe —
  `PROBE PASS max_diff=0 umma_pair_issue_loop_single_thread/U1-four-issue-forms-exact`; the readings in that section's table are timing fact lines of the same job (a fourth spelling the probe also measured is not included in this entry).
