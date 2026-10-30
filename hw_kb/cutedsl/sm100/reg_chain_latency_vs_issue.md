# reg_chain_latency_vs_issue (splitting the accumulator of a register reduction chain: when it is latency-bound and when issue-bound; ptxas does not reassociate but does contract; `fma.rn.f32.bf16` saves the unpacking)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Sibling entries: [`packed_f32x2_chain_single_asm`](packed_f32x2_chain_single_asm.md) (how to spell a packed chain),
> [`rmem_materialize_packed_reduce`](rmem_materialize_packed_reduce.md) (the register-placement precondition of a single packed reduction step).
> This entry answers only: **how many cycles splitting an "unpack → square → reduce to one scalar within the thread" register chain into k independent accumulators / a balanced tree buys back on sm100**,
> and how that answer changes with the number of warps resident per SMSP.

> **Status**: ✅ Verified (GPU probe `PROBE PASS max_diff=1.58e-07`,
> certification round 2, both contracts passed in the same round: the scalars of all eight variants against a numpy fp64 sum of squares, max relative error 1.58e-07;
> SASS contract — the `FFMA2/FMUL2/FADD2` counts of the k-way packed variants equal the PTX one for one, the full-tree variant shows mul+add → `FFMA2` contraction (PTX 2N−1 → SASS 3N/2−1), all with 0 `LDL/STL`).
>
> **Certified surface**: what is certified is **numerics and SASS counts**. The cycle table below is INFO readings from the same job (median of 5 per variant per level), without a significance test.

## Purpose
A thread holds N bf16x2 words that have already landed; they must be unpacked to fp32, squared and reduced to one scalar before going into `SHFL`. The cycles of this stretch are set by the **larger** of two things:

- **dependency depth × per-dependency latency** (latency-bound): a serial accumulation `A = fma(x, x, A)` has depth N; split into k accumulators it is `N/k + log2(k)`;
- **total instruction count of this stretch over all warps on this SMSP ÷ issue rate** (issue-bound): each SMSP issues at most one warp instruction per cycle;
  unpacking (`SHF`/`LOP3`/`IMAD.U32`), packed FP, the tail tree and the feedback all take issue slots.

Splitting the accumulator only lowers the first term, and usually **adds** instructions (each extra accumulator adds one `FADD2` to the tail tree). So it can only pay off on shapes with "few warps, few instructions, deep chain";
once the warps resident on this SMSP can fill each other's dependency waits, the chain depth is hidden by issue, and further splitting gains nothing or even loses.

## When needed (trigger)
- You want to shorten a "land → square → in-thread reduce → `SHFL`" register chain with "multiple independent accumulators / a balanced tree".
  **Estimate before building**: compare "depth × dependency latency" with "warps resident on this SMSP × SASS count of this stretch"; if the latter is larger, it is issue-bound and splitting the accumulator will not be faster.
  Count the stretch's instructions from `cuobjdump -sass` (including unpacking and `MOV`).
- The chain is already k accumulators + a tree and you want to push to more ways or a full tree — first check in the readings below whether that step already gains nothing at the same warp count.
- Unpacking instructions (2 integer instructions per bf16x2 word) outnumber FP instructions: first see whether `fma.rn.f32.bf16` can consume bf16 directly (see (c)).

## Mechanism

### (a) ptxas does not reassociate, but contracts `mul → add` into `FFMA2`
fp32 additions / FMAs are not reassociated, and a k-way packed `.f32x2` accumulation chain is neither split nor merged: however many `mul/fma/add.rn.f32x2` you write in PTX, that is how many
`FMUL2/FFMA2/FADD2` you get in SASS (1 / 2 / 4 / 8 ways measured N / N+1 / N+3 / N+7, equal to the PTX one for one). There is no such thing as "the compiler will turn it into a tree for you".
**Exception: the full tree.** Written as "N `mul.rn.f32x2` + an `add.rn.f32x2` balanced tree", even with `.rn`, ptxas contracts each first-level add with one of the muls
into one `FFMA2`: 2N−1 instructions in PTX, only 3N/2−1 left in SASS, shaped as an N/2-way accumulation (N/2 `FMUL2` + N/2 `FFMA2` + N/2−1 `FADD2`).
So writing a full tree in PTX gets you an N/2-way accumulation; the rounding order also becomes the single rounding of FMA (numerics still match at the 1e-7 level).

### (b) The timing loop must be written entirely in one asm block
To measure the latency of "land → scalar", the timing loop itself (`%clock64` → loop body → `%clock64`) goes into **one** inline-PTX block, with `.pragma "nounroll";` at the loop head,
and the next iteration's input depends on the previous result (`t = bits(s) & Z`, Z a kernel parameter that is 0 at run time, `w_i ^= t`). Otherwise LLVM / ptxas unroll the loop and
hoist the loop-invariant unpacking and partial products out of it, and you measure throughput, not latency.

### (c) Readings (B200, sm_100a, measured in the same job as the certification probe)
Probe instance: N bf16x2 words per thread (the count and depth columns of the table are written in terms of this N). Cycles / iteration (lane 0 of each warp, `REPS=1024`); columns are warps resident per SMSP;
SASS counts are for the whole kernel:

| Variant | packed count (SASS) | Depth | SASS total | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| packed, 1 accumulator | N | N | 104 | 143 | 188 | 342 | 649 |
| packed, 2 accumulators | N+1 | N/2+1 | 104 | 152 | 200 | 370 | 697 |
| packed, 4 accumulators + 2-level tree | N+3 | N/4+2 | 112 | 182 | 231 | 384 | 743 |
| packed, 8 accumulators + 3-level tree | N+7 | N/8+3 | 112 | 142 | 219 | 393 | 729 |
| packed, full tree | 3N/2−1 (PTX 2N−1) | log2 N + 1 | 112 | 143 | 214 | 395 | 745 |
| scalar fp32, 1 accumulator | 0 | 2N | 120 | 150 | 197 | 376 | 737 |
| scalar fp32, 8 accumulators + tree | 0 | 2N/8+3 | 128 | 188 | 250 | 419 | 813 |
| `fma.rn.f32.bf16`, 8 accumulators + tree | 0 | 2N/8+3 | 96 | **106** | **150** | **266** | **516** |

The fixed per-iteration overhead (N xor feedbacks + counter + branch) is the same in all variants; compare the differences between variants.

How to read this table:
- **At this size it is already not latency-bound at 1 warp/SMSP.** The serial chain of depth N (scalar 2N) is not slower than 4-way / 8-way; the serial version is actually fastest at 2, 4 and 8 warps (8 warps: 649 vs 697–745).
  Splitting the accumulator only adds instructions.
- **Relation to the end of [`softmax_segment_ilp`](softmax_segment_ilp.md) §(d) / [`rmem_materialize_packed_reduce`](rmem_materialize_packed_reduce.md)**:
  those two cover the row reduction of a 128-element fragment (chain depth 128), saying that "splitting the chain is no longer necessarily null on sm100"; this entry measures a short chain of only a few dozen values per thread,
  and concludes that at this depth splitting does not pay on sm100 either. The two are distinguished by chain depth and do not overturn each other.
- The 1-warp readings are not monotonic (4-way 182 vs 8-way 142); that is an individual scheduling / register-bank effect, not a depth effect; do not use a single 1-warp point to argue that "splitting k ways pays".
- **The only clear gain comes from cutting instructions.** `fma.rn.f32.bf16` (SASS `FHFMA.BF16`) consumes a bf16 half directly and accumulates in fp32, eliminating the `SHF`/`LOP3` unpacking:
  about 26% (1 warp) / 20% (8 warps) faster than packed serial, SASS total 104 → 96, numerics consistent with the fp32 unpacking path (max rel diff ≈ 1e-7).

### (d) Whole-kernel readings of the same mechanism
On the full kernel where this was found (a latency-bound straight-line row-reduction kernel, 6 shapes × 9 rounds, paired geometric mean in the same job, ratio = control time / variant time): rewriting the in-thread sum of squares as
the bundle "bf16 halves fed directly into `fma.rn.f32.bf16`, then split into a few independent accumulators" read 1.020,
1.019, 1.015 in three readings. That is the combined price of "drop the unpacking + split the chain", not priced separately;
earlier rewrites on the same kernel that only did "multi-way accumulation / earlier start" without dropping the unpacking did not read better than the original chain at the gate.

## Minimal runnable example
```python
# k-way split packed accumulators + a balanced tree, all in one inline-PTX block. X0..X(N-1) are the unpacked .b64 pairs.
def acc_packed(n_word, k):
    t = ["mul.rn.f32x2 A%d, X%d, X%d;" % (j, j, j) for j in range(k)]
    for i in range(k, n_word):
        j = i % k                                  # round-robin over k independent chains
        t.append("fma.rn.f32x2 A%d, X%d, X%d, A%d;" % (j, i, i, j))
    m = k
    while m > 1:                                   # log2(k)-level FADD2 balanced tree -> A0
        for j in range(m // 2):
            t.append("add.rn.f32x2 A%d, A%d, A%d;" % (j, 2 * j, 2 * j + 1))
        m //= 2
    t += ["mov.b64 {sl, sh}, A0;", "add.rn.f32 s, sl, sh;"]   # combine the two halves into one f32
    return t

# Unpacking (two integer instructions per word; bf16 -> fp32 is just shift / mask):
#   shl.b32 lo, w, 16;  and.b32 hi, w, 0xffff0000;  mov.b64 X, {lo, hi};
# The spelling without unpacking (one FHFMA.BF16 per bf16):
#   mov.b32 {h0, h1}, w;                           // .reg .b16 h0, h1
#   fma.rn.f32.bf16 S, h0, h0, S;  fma.rn.f32.bf16 S', h1, h1, S';
# Wrap the whole block in llvm.inline_asm(..., "=f,...", has_side_effects=False);
# spelling rules: see packed_f32x2_chain_single_asm. The timing version (%clock64 + nounroll loop) is in the certification probe.
```

## Gotchas
1. **Depth is not the price; issue may be.** Splitting the accumulator only shrinks dependency depth; when the warps on this SMSP can fill each other's dependency waits, depth is completely hidden and the only price left is the instruction count.
   First use the table in (c) to determine which column you are in.
2. **A full tree in PTX is not a full tree in SASS.** A full tree of N `mul` + N−1 `add`: ptxas contracts the first level into `FFMA2`, and SASS ends up as an N/2-way accumulation (see (a)).
   If you really want "k ways", write k ways directly.
3. **Unpacking instructions usually outnumber FP.** Each bf16x2 word needs 2 integer instructions to unpack, matched with only 1 packed FP; in the issue-bound regime, what can be saved is the unpacking and `MOV`, not the shape of the FP chain.
4. **Do not use C-style timing (two clock reads around a piece of code) to measure a chain of a few dozen cycles**: ptxas may reorder clock reads with independent ALU work. Use the circular dependency of (b) + averaging over many iterations.
5. **Put `.pragma "nounroll";` after the loop-head label.** In the probe, the labelled timing block has only one call site per kernel; whether inlining it several times in one kernel causes duplicate names was not tested.

## Sources
- PTX ISA: the `.f32x2` forms of `add/sub/mul/fma`, `fma.rn.f32.bf16` (mixed-precision FMA), `mov.b64 {lo, hi}`, `mov.b32 {h0, h1}`, `%clock64`,
  `.pragma "nounroll"`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`cuobjdump -sass`): https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `cutlass._mlir.dialects.llvm.inline_asm`, `llvm.extractvalue`, `cutlass.cutlass_dsl.dsl_user_op`, `cute.compile`.
- Certification probe (not included in this release; variants p_serial / p_acc2 / p_acc4 / p_acc8 / p_tree / s_serial / s_acc8 / m_acc8;
  `PROBE PASS max_diff=1.58e-07`).
