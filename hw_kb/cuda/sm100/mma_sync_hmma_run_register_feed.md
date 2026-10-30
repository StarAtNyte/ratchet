# mma_sync_hmma_run_register_feed (register-fed `mma.sync` HMMA runs: same-accumulator dependency latency and saturation distance, how A quads land and `MOV` relays (two sources: smem and 256-bit global loads), straight-line unrolled vs rolled, source scan order and `volatile` do not control SASS)

> **Certification source: B200 GPU-certified in CUDA mode, four GPU probes (the probes are not included in this release):**
> the chain-distance / quad-feed probe (PROBE PASS max_diff=0.0, round 1 18/18),
> the operand-slot probe (PROBE PASS max_diff=0.0, round 2 16/16),
> the HMMA issue-order probe (PROBE PASS max_diff=0.0, round 2 8/8),
> the global-load quad-feed probe (PROBE PASS max_diff=0.000275, round 1 15/15; fact 5).
> What is certified is the **numerics** (each variant bitwise identical to the torch reference, and the variants bitwise identical to each other) and the **static SASS shape** (HMMA count; the minimum same-destination-register distance in a rolled body == D;
> ≥ 16 window moves when quads are assembled from row-wise LDS, ≤ 8 for native quads / `ldmatrix`; ≥ 2 moves per HMMA when unrolled straight-line, ≤ 8 in a rolled body;
> moving the smem operand to the B slot does not reduce moves; the rotated spelling with and without `volatile` gives an identical HMMA stream instruction by instruction; when A is fed by row-wise `ld.global.v8`, the first reader of every A load is a move,
> with a quad-native global layout the A slot has zero moves and every first reader is an HMMA; `LDL/STL` = 0). The cycle tables below are fact readings of the same jobs, **not certified items, and they do not certify any whole-kernel gain**.
> Same-job whole-kernel timings on the same class of kernel are in gotcha 9: none of the changes tested showed a whole-kernel gain (quad-native layout and `ldmatrix` not measured on a whole kernel).
>
> One file per hardware interface. Operator-agnostic: mechanism only. Neighboring entries: [`ldg_issue_pin_structural_fence`](ldg_issue_pin_structural_fence.md)
> (issue position of global load batches in the same class of kernel: the fence before the first batch, the lead in the k-loop steady state), [`strided_loop_static_trip_guard`](strided_loop_static_trip_guard.md)
> (removing back jumps from memory-access / writeback loops: the opposite direction to fact 3 here, different scope, see gotcha 3).

## Purpose

The A operand of `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (SASS `HMMA.16816.F32`) is 4 consecutive aligned registers `{a0, a1, a2, a3}`,
B is 2, C/D is 4. When all operands come from registers, the shape of the HMMA run in SASS — whether each HMMA is preceded by register moves
(`MOV` / `IMAD.MOV*` / `PRMT` / `UMOV`), whether adjacent HMMAs share operands (`.reuse`), how far apart two HMMAs writing the same accumulator are — is laid out by ptxas from
the dependency graph and register allocation. This entry gives five measured facts and the matching spellings:

1. **Same-accumulator distance D and pipe cycles**: with D independent accumulators each appearing once in a rolled loop, the same-accumulator distance in SASS equals D exactly;
   the curve of cycles / HMMA against D gives the latency of a dependent HMMA and the saturation distance.
2. **How A quads must land to need no `MOV`**: a quad interleaves row g and row g+8; the destination register group of a row-wise `ld.shared.v4` is never an HMMA A group, so ptxas has to move.
   Two spellings that let the LDS land directly as quads: a **quad-native smem layout** (the producer interleaves, one `ld.shared.v4` = one quad), and **`ldmatrix.x4`**.
3. **Straight-line unrolled vs rolled**: when the operands are loop-invariant, a straight-line unrolled HMMA run has moves before every HMMA; put the same segment into a rolled loop with a runtime trip count and the moves disappear.
4. **Source scan order and `asm volatile` do not control the HMMA order in SASS**: i-outer and n-outer land the same in SASS; rotation produces a different stream, but not the order written in the source; `volatile` changes nothing.
5. **When the operands come directly from `ld.global.v8` (256-bit LDG)**: assembling A quads from row-wise loads likewise always needs moves, and the first reader of every A load is a move —
   the scoreboard wait is booked on the move; a global layout pre-interleaved by quad makes each of an LDG's two 4-register destination groups one quad, and the A slot has zero moves.

## When needed (trigger)

- A register-fed HMMA segment whose stall ledger is dominated by `wait` / `math_pipe_throttle`, and you want to "rotate accumulators to spread dependencies" or "add a few more independent accumulators":
  first check the current residency (warps per SMSP) and the minimum same-accumulator distance against fact 1 — with 2 warps per SMSP it is already saturated at D ≥ 3.
- In SASS each HMMA is preceded by grouped `MOV` / `IMAD.MOV.U32` whose source registers are the destinations of `LDS.128`: quads assembled from row-wise LDS (fact 2).
- The same moves, with source registers that are the destinations of `LDG.*.256` (256-bit global loads); or in the NCU source view long_scoreboard piles up on these `MOV` / `IMAD.MOV` rows
  rather than on the HMMAs: quads assembled from row-wise LDG (fact 5) — that is where the wait is booked; it does not mean the moves themselves are expensive.
- The HMMA run is unrolled straight-line, the number of moves across the HMMA span is of the same order as the number of HMMAs, and the operands are already in registers before the segment (fact 3).
- You want to change the HMMA order / `.reuse` in SASS by changing the (i, n) scan order in the source or by making `mma.sync` `volatile` (fact 4: it does not control them).
- **Read gotcha 9's whole-kernel readings first**: on the same class of kernel, zeroing the moves in the HMMA window, changing the scan order and rolling were all within noise or slower in same-job whole-kernel timing.
  These are explanations of "why the SASS shape is the way it is", not ready-made speed-up levers.

## Minimal runnable example

```cpp
// fk-cuda-cflags: (no extra flags)
__device__ __forceinline__ void hmma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                     uint32_t b0, uint32_t b1) {          // not volatile: ptxas may schedule freely
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void lds128(uint32_t a, uint32_t (&w)[4]) {
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t (&w)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]) : "r"(a) : "memory");
}
// g = lane >> 2, t = lane & 3

// (1) same-accumulator distance = D: rolled loop, runtime trip count (nrep is a kernel parameter), fully unrolled inside the body
#pragma unroll 1
for (int r = 0; r < nrep; ++r)
#pragma unroll
  for (int d = 0; d < D; ++d) hmma(acc[d], a[d % QA][0], a[d % QA][1], a[d % QA][2], a[d % QA][3], b[d % QB][0], b[d % QB][1]);

// (2a) row-wise LDS: lo = 16 B of row g, hi = 16 B of row g+8; the quad of k16 step j interleaves the two rows -> 4 MOVs per quad
lds128(row_g, lo); lds128(row_g8, hi);
hmma(acc, lo[2*j], hi[2*j], lo[2*j+1], hi[2*j+1], bw[2*j], bw[2*j+1]);

// (2b) quad-native layout: in smem, 16 B per lane per k16 step = {a0, a1, a2, a3} (interleaved by whoever writes smem, with st.shared)
lds128(sm + ((blk * KSTEPS + j) * 32 + lane) * 16, q);
hmma(acc, q[0], q[1], q[2], q[3], bw[2*j], bw[2*j+1]);

// (2c) ldmatrix.x4: lane L supplies the 16 B row address of row (L & 7) + 8*((L >> 3) & 1), column 16*j + 8*(L >> 4) (elements);
//      result q[0..3] = a0..a3 of the standard fragment mapping. B must also use the standard mapping: b0 = B[n = g][16j + 2t ..], b1 = B[g][16j + 8 + 2t ..]
const int row = 16 * blk + (lane & 7) + ((lane >> 3) & 1) * 8, col = 16 * j + (lane >> 4) * 8;
ldsm_x4(sm + row * PITCH + col * 2, q);
hmma(acc, q[0], q[1], q[2], q[3], b0_std, b1_std);

// (3) when the operands are loop-invariant, put the whole straight-line HMMA segment into a rolled loop with a runtime trip count (here nrep is passed 1); still fully unrolled inside
#pragma unroll 1
for (int r = 0; r < nrep; ++r)
#pragma unroll
  for (int s = 0; s < KSTEPS; ++s) run_step(s, acc, xa, yb);    // run_step fully unrolls both (i, n) levels

// (4) (i, n) scan order within one k16 step: k stays outermost (the fp32 accumulation order of each accumulator is unchanged -> any scan order is bitwise identical)
template <int ORD> struct Order {          // p = 0 .. MI*NG-1
  // ORD 0: i outer, n inner; ORD 1: n outer, i inner; ORD 2: CRT rotation i = p % MI, n = p % NG (requires gcd(MI, NG) = 1)
  static constexpr int i(int p) { return ORD == 0 ? p / NG : p % MI; }
  static constexpr int n(int p) { return ORD == 0 ? p % NG : (ORD == 1 ? p / MI : p % NG); }
};
```

Acceptance (`cuobjdump -sass`):

| Count | How to count |
|---|---|
| same-accumulator distance | in the HMMA subsequence, how many HMMAs lie between two whose first operand (destination register) is the same (a rolled body counted with its end joined to its start) |
| moves in the window / span | number of `MOV*` / `IMAD.MOV*` / `PRMT` / `UMOV` within the timed window (or between the first and last HMMA) |
| `.reuse` / adjacent sharing | number of `.reuse` suffixes on HMMA operands; number of adjacent pairs in the HMMA subsequence whose 2nd / 3rd operands (with `.reuse` stripped) are the same (operand order `D, A, B, C`) |
| `LDS` / `LDSM` count | still one per quad after changing how they land |
| `LDL` / `STL` | must be 0 |

## Probe readings (same-job fact readings, not certified items)

**Fact 1** (chain-distance / quad-feed probe, cycles / HMMA per warp; `W4` = 1 warp per SMSP, `W8` = 2 warps per SMSP, the latter given per SMSP):

| D | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 warp / SMSP | 24.21 | 14.11 | 12.41 | 11.06 | 10.45 | 10.04 | 9.53 | 9.23 | 9.03 | 8.84 |
| 2 warps / SMSP (per SMSP) | 12.10 | 9.05 | 8.04 | 8.03 | 8.02 | 8.02 | 8.02 | 8.01 | 8.03 | 8.09 |

- The back-to-back interval of dependent HMMAs ≈ 24 cycles (D = 1). With only 1 warp per SMSP, cycles / HMMA are still falling at D = 16 (8.84);
  with 2 warps per SMSP it saturates at ≈ 8 cycles / HMMA / SMSP from D ≥ 3, and more independent accumulators do not help.
- The operand-sharing shape does not change pipe cycles: at D = 4 "all shared" and "none shared" are both 11.06; at D = 10 all shared / block shared / none shared are 9.23 / 9.33 / 9.23
  (`.reuse` 16 / 7 / 0 respectively).

**Fact 2** (same job, rolled body with two k16 steps per round, A re-read from smem every round, ping-pong register groups):

| How A quads land | Moves in window | Moves in span | Cycles / HMMA (1 warp / SMSP) | Cycles / HMMA (2 warps / SMSP, per warp) |
|---|---|---|---|---|
| row-wise `LDS.128` (2a) | 49 | 28 | 9.10 | 16.55 |
| quad-native layout (2b) | 7 | 4 | 8.33 | 16.07 |
| `ldmatrix.x4` (2c) | 7 | 4 | 8.33 | 16.05 |

**Fact 3** (operand-slot probe, a straight-line segment of several dozen HMMAs per warp; the "window" starts after all operands have landed in registers):

| Spelling | Moves in HMMA span | `.reuse` | Window cycles (1 / 2 warps per SMSP) |
|---|---|---|---|
| straight-line unrolled, serial LDS within the window | 193 | 8 | 1627 / 2408 |
| straight-line unrolled, half-block `LDS.128` lookahead within the window | 172 | 14 | 1279 / 2005 |
| straight-line unrolled, `LDS.64` lookahead per k step within the window | 168 | 16 | 1826 / 3230 |
| straight-line unrolled, all operands in registers beforehand (no LDS in the window) | 232 | 1 | 1562 / 2454 |
| the same segment in a rolled loop with runtime trip count = 1 (operands in registers beforehand) | **0** | 47 | 1381 / 2144 |

(Multi-trip rolled steady state: 9.94 cycles / HMMA, 1 warp / SMSP.) Moving the smem operand to the B slot and assembling the register operand into the A quad (computing C^T): under serial scheduling, moves within the window
240 vs 208 in the A slot, window cycles 1748 / 3032 vs 1627 / 2408 — more and slower.

**Fact 4** (HMMA issue-order probe, a segment of several dozen HMMAs; `.reuse` / adjacent shared A / shared B / minimum same-accumulator distance):

| Source spelling | HMMA stream in SASS |
|---|---|
| i outer, n inner | `.reuse` 15, adjacent shared A / shared B 37 / 9, minimum distance 2 |
| n outer, i inner | **same as i outer**: every count identical, the D/A/B registers of the first 12 HMMAs identical one by one |
| CRT rotation | changed: `.reuse` 23, 38 / 11, minimum distance 4 (in the source no two adjacent ones share A or B, yet in SASS there are more sharing pairs) |
| CRT rotation + `asm volatile` | the whole HMMA stream (every D/A/B, the order) identical one by one to the rotation without `volatile` (certified contract) |
| i outer + `asm volatile` | the whole HMMA stream identical one by one to i outer without `volatile` |

Segment cycles (median of per-launch p50): i outer / n outer 1895, CRT rotation 1814 cycles — a difference on the isolated segment, **whole kernel flat in the same job** (gotcha 9).

**Fact 5** (global-load quad-feed probe; A and B both fed directly by one `ld.global.v8` per lane (32 contiguous B per lane), the A 16-row fragment = row g and row g+8;
"first reader" = the first SASS instruction reading this load's destination registers):

| A's global layout / spelling | Moves in HMMA span (of which into the A slot) | First reader of A loads | Registers / thread |
|---|---|---|---|
| row-wise, B not zero-filled | 102 (71) | all moves | 110 |
| row-wise, out-of-bounds B rows zero-filled (`mov 0` + predicated load inside asm, or a C++ branch) | 114–118 (88–90) | all moves | 114 |
| quad-native (each lane's `{a0, a1, a2, a3}` per k16 step pre-interleaved in global memory into a contiguous 16 B) | **0** (0) | all HMMAs | 102 |

- Within each row, with or without `volatile` on the load / mma asm (two to four combinations each), every count is identical: the only deciding factor is whether the word order in global memory makes an LDG's 4-register destination group exactly one quad.
- Row-wise, ptxas pulls the moves ahead of the HMMAs in stretches and the HMMAs get chopped up by moves; quad-native, the SASS alternates "a batch of LDGs, a string of HMMAs".
- B pairs (two adjacent even-aligned words of one row) are taken directly from the load's destination registers; none of the three zero-fill spellings moves them, and the first reader of every B load is an HMMA.
- Reading the NCU source view (inference; this probe did not run NCU: a scoreboard wait is attached to the first reader): long_scoreboard is booked on a load's first reader. With the row-wise layout it piles up on the `MOV` / `IMAD.MOV` rows; that is where this shape books it, not evidence that the moves themselves are expensive.
- **Timing**: in the same probe the per-launch median of all 10 variants is 8.29 µs (7 timings, range 8.25–8.34) — 0 moves and over a hundred moves make no difference on the isolated kernel.
- Cost: quad-native needs an A buffer interleaved by lane (written by a preprocessing pass; when the operands change every call, this pass must be counted separately); B's K mapping is unchanged, and the numerics are right only if K agrees on both sides.
- Whole kernel: the quad-native global layout is **not measured**; on the same class of kernel, the whole-kernel reading of "leave the layout alone, only reorder within the block to move the wait onto the HMMAs" was negative, see gotcha 9.

## Gotchas

1. **A register group alignment**: when the 4 `"r"` passed into the inline asm do not come from the destination group of a single `ld.shared.v4` / `ldmatrix` (or come in a different order),
   ptxas assembles them into one contiguous group with `MOV`. A B pair that is an adjacent even-aligned pair of words in a load vector needs no move.
2. **Row-wise LDS / LDG can never assemble a quad**: a quad interleaves row g and row g+8, and one row-wise `ld.shared.v4` / `ld.global.v8` gets only one row; changing the load order, adding `volatile` or changing the zero-fill spelling cannot remove this group of moves (facts 2, 5).
3. **Zero moves in a rolled body hold only when the operands are loop-invariant**: the rolled variant of fact 3 loads its operands outside the loop, so the register groups are allocated only once;
   the rolled body of fact 2 re-reads A from smem every round, and row-wise LDS still moves every round (49 vs 7). Rolling is no substitute for landing quads correctly.
   Short per-thread strided loops such as memory access / reduction writeback go the opposite way: there you want a compile-time trip + guard to remove the back jump ([`strided_loop_static_trip_guard`](strided_loop_static_trip_guard.md));
   "rolled is better" applies only to the register allocation of HMMA runs; do not generalize it to other loops.
4. **The rolled loop must be a real loop**: with a compile-time constant trip count of 1 the compiler deletes the loop and it becomes straight-line code again; use a runtime parameter (`#pragma unroll 1` + a runtime bound).
   The inside of the loop body (k steps, i, n) stays fully unrolled, otherwise non-constant register-array indices fall into local memory — check `LDL/STL` first after the change.
5. **TMA cannot produce a quad-native layout**: `cuTensorMapEncodeTiled` requires `boxDim[0] * elementSize` to be a multiple of 16 B, so it cannot interleave two rows at 4 B granularity.
   A quad-native layout can only be written by threads with `st.shared`, or use `ldmatrix` instead.
6. **`ldmatrix` dictates the K mapping**: lane (g, t) gets elements 2t and 2t+1 of an 8-element row segment, so within one k16 step lane t's A elements are `seg0 + 2t`, `seg1 + 2t`,
   and the B side must hold the same physical K. If B is register-fed with each lane reading 32 contiguous B of K (`ld.global.v8`), its K mapping is incompatible with `ldmatrix`, and B must switch to the same K permutation
   (`ldmatrix` / `.trans` or 4 B-granular reads); the numerics are right only if both sides agree. Each `ldmatrix` row address is 16 B aligned; the probe uses an 80 B row pitch (8 rows land in 8 different 16 B units);
   bank conflicts under other row pitches / swizzles were not measured by the probe.
7. **Source order is not SASS order, and `asm volatile` is not a ptxas scheduling barrier**: `volatile` only constrains the NVVM level not to delete, merge or reorder across volatile;
   ptxas reorders mutually independent HMMAs by the dependency graph (fact 4). The source order by itself cannot change the SASS order; forcing an order by changing the dependency graph with `__syncwarp()` and the like made the whole kernel slower on the same class of kernel (gotcha 9).
   CRT rotation requires gcd(MI, NG) = 1, otherwise `i = p % MI, n = p % NG` is not a bijection (missed / duplicated accumulators, numerics wrong at once);
   changing only (i, n) with k kept outermost leaves the result bitwise unchanged; moving k inward changes the fp32 accumulation order.
8. **Swapping slots or an `LDS.64` lookahead per k step is not the cure** (fact 3): swapping slots gives more moves and is slower, and also transposes the accumulator coordinates (`c0, c1` become two adjacent rows on the smem side), so the writeback mapping must follow;
   `LDS.64` doubles the instruction count and is slower than an `LDS.128` covering two k steps at once.
9. **Whole-kernel readings (the same class of register-fed `mma.sync` kernel, one straight-line HMMA run per warp, same-job interleaved timing, geometric mean new / old)**:
   - 6 scan orders within one k step (including CRT rotation): 0.9999–1.0000; switching to CRT rotation alone: 1.0000.
   - Pinning the order with a `__syncwarp()` after every HMMA (changing the dependency graph): 1.0041–1.0066, 3–6% slower on the larger instances (one after each k step: 0.9999).
   - The whole HMMA segment in a rolled loop: 0.9951 (other ways of cutting the roll 0.9966 / 1.0007 / 0.9994), at the edge of noise; no improvement seen in cross-job evaluation.
   - Zeroing the moves in the HMMA window (another spelling: word-by-word `ld.volatile.shared.b32` landing into quad slots per k step, window moves 186 → 0, `.reuse` 21 → 61):
     smaller instance 1.0000, larger instance 1.296× (slower); the version adopted only on the smaller instance was actually slower in cross-job evaluation (0.0055 → 0.0058 ms, inconsistent with the same-job flat reading, not re-measured).
   - On the same class of kernel fed directly by 256-bit global loads with a row-wise layout, trying per fact 5 to move the loads' scoreboard wait from the moves onto the HMMAs without changing the global layout, only reordering within the block
     (assembling quads in place per k16 step, swapping the HMMA order, three steps in sequence; a fourth step also split each row's one 256-bit load into two 128-bit ones). The same-job ratios of four evaluations were
     0.9987 / 0.9933 / 0.9927 / 0.9856 (before the change 0.9998; a re-measure of the reverted tree, byte-for-byte identical to before the change, 0.9978) — slower at every step, the last step about 1.4% slower for the whole group, and the whole group was reverted.
     The ratio of the workload containing this kind of kernel went 0.956 → 0.958 / 0.939 / 0.922, and 0.864 after splitting the loads — the more it changed, the slower.
   - The two landings, quad-native layout (smem or global) and `ldmatrix.x4`, are **not measured** on a whole kernel.
   Conclusion: the whole-kernel time of such segments is not decided by the number of moves in the HMMA window, the `.reuse` count or the scan order; use the facts of this entry to read SASS and rule out false suspects, not as speed-up levers.

## Sources

- PTX ISA: the A / B / C fragment layouts of `mma.sync.aligned.m16n8k16`, `ldmatrix`, `ld.shared.v4`: <https://docs.nvidia.com/cuda/parallel-thread-execution/>;
  inline PTX constraints and `volatile`: <https://docs.nvidia.com/cuda/inline-ptx-assembly/>; `#pragma unroll`: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA Driver API `cuTensorMapEncodeTiled` (`boxDim[0] * elementSize` a multiple of 16 B): <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- CUDA Binary Utilities (`cuobjdump -sass`, the `.reuse` operand suffix): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probes: the chain-distance / quad-feed probe (D ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16} × three operand-sharing shapes; three A-quad landings; 1 / 2 warps per SMSP) —
  PROBE PASS max_diff=0.0; the operand-slot probe (2 slots × 6 schedules × 2 residencies) — PROBE PASS max_diff=0.0;
  the HMMA issue-order probe (i outer / n outer / CRT rotation / rotation + volatile / i outer + volatile) — PROBE PASS max_diff=0.0;
  the global-load quad-feed probe (row-wise / quad-native A × load / mma `volatile` × three B zero-fill spellings, 10 variants) — PROBE PASS max_diff=0.000275.
