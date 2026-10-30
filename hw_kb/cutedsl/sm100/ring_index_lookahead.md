# ring_index_lookahead (in a rolled loop, the "next beat's" barrier address / wait parity / data-slot base of an mbarrier ring are **computed a full beat ahead and loop-carried**: the address operand of each `try_wait` / `arrive` / `ld.shared` comes straight from a carried register, and in the loop body they are no longer immediately preceded by the integer / uniform instructions that produce them)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> **Status: ✅ Verified (semantics + SASS distance; no gain certified)** — certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (certification round 2): the `site` / `carry` / `ahead` forms are bit-correct on cross-warp data,
> the bounded spin has zero timeouts, re-runs have zero drift; SASS contract `ahead-addr`: in the `ahead` form the `[address]` registers of the three `try_wait`s are ≥ 18 instructions away from their nearest writer
> (three readings: no writer within the 60-instruction window / 18 / 35). Controls (info rows): `site` minimum 2, `carry` minimum 4. In all three forms the parity operand is produced by a
> `SHF.L.U32 …, 0x1f` right next to the `SYNCS` (gotcha 6); the `IMAD.HI` / `IMAD.WIDE.U32` count is 4 in all three forms (gotcha 5).
> Per-beat cycles in the probe toy (info rows, not a criterion): `site` 403 / `carry` 382 / `ahead` 380. Whole-kernel readings: see the end of "When needed".
>
> Neighbouring entries: the spelling that only replaces `k % S` / `k // S` with a loop-carried (stage, phase) while addresses are still computed at the point of use from the carried index is
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) (B) and its gotcha 16 — this entry goes one step further: what is carried is **the already computed addresses and parity**,
> and they are computed inside the previous beat's body. The encoding of the wait itself is in [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md).
> The fully carried spelling on the issue side (smem addresses of TMA / UMMA, descriptor low words, coordinates) and its dependency-chain readings are in [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) gotcha 17.

## Purpose

Every beat, a rolled loop does `try_wait.parity` / `arrive` on several mbarrier rings and then reads / writes the data slot selected by the slot number. Slot number, parity, barrier address
and data-slot base are all functions of the beat number g:

```
slot   = g mod N            parity = (g div N) & 1          (^1 again when waiting for "previous pass released")
mbar   = mbar_base + 8*slot  data  = data_base + slot*SLOT_B + parity*BUF_B
```

When computed at the point of use (or with addresses computed on the spot from a carried slot), the chain of `LOP3/ULOP3`, `ISETP`, `IMAD/LEA/ULEA` sits right next to its consumers in SASS — the address operand of `SYNCS`,
the address of `LDS`. These are all **fixed-latency** producers; the consumer reads their result directly, and the warp carries a stretch of producer latency at the consumer (on the profiler this shows up as a stall before the consumer,
with an integer instruction such as `LOP3/ISETP` as the producer).

**One beat ahead**: the values for g+1 are computed inside beat g's body, after g's wait has been issued, and carried into the next beat as loop-carried variables. The next beat's consumer reads a register that has long been ready,
with a whole beat of independent work between producer and consumer. The cost is a few more live registers (1–3 per ring, estimated, not measured).

## When needed (trigger)

- Profiler attribution shows: stalls before `SYNCS` (try_wait / arrive) or `LDS` / `STAS` in the loop body whose producer is **integer / uniform address and parity arithmetic**
  (`LOP3/ULOP3`, `ISETP/UISETP`, `IMAD`, `LEA`), not a variable-latency instruction such as `LDS` / `LDTM`.
- One beat touches several rings (with different slot counts, some not powers of 2), and every ring redoes a chain of slot / parity / address arithmetic every beat.
- **Not applicable**: the wait really is waiting for the peer (large spin exposure) — that is a producer-cadence problem, and moving the address arithmetic will not make it faster; the register budget is already at the edge (1–3 more live registers per ring, estimated, not measured).
- **Whole-kernel readings (no gain claimed)**: on one full-kernel tree whose critical chain sits on an elementwise role (rolled loop, 0 spills), landing the basic form of this entry in one go
  moved the gate 1.3291 → 1.3248 ms (−0.32%, two cross-job evals), while the same-job control ratio went 0.9176 → 0.9157 (−0.2%); there was no same-job paired A/B, so the direction is undetermined. Later, following the same idea, the remaining integer producers
  (another parity, a statistics barrier address, the first-beat selection, a guard flag) were also made carried or hoisted out of the loop, in three steps, each bundled with other small changes in the same package;
  cumulatively 1.3248 → 1.3317 ms (+0.52%, slower). In the probe toy, per-beat cycles went `site` 403 → `carry` 382 → `ahead` 380: going from `carry`
  on to "one beat ahead" saves only about 0.5% more. So before landing it, do a same-job A/B on your own kernel; a producer sample count in the stall sampling alone does not prove the time can be recovered.

## Minimal runnable example

```python
# One rolled loop consuming two mbarrier rings at once: A (N_A = 2**K_A slots) and B (N_B slots, not a power of 2); more rings follow the same pattern.
# Each barrier's count = number of warps that arrive per beat. Each data slot of A / B has two buffers, read / written alternately by pass parity:
# SLOT_B = 2 * BUF_B, BUF_B a power of 2, slot bases aligned to SLOT_B, so "write buffer = read buffer ^ BUF_B".
# mbar_wait / lds / sts / mbar_arrive are thin inline-PTX wrappers (mbarrier.try_wait.parity / ld.shared / st.shared / mbarrier.arrive).
K_B = (N_B - 1).bit_length()            # 2**K_B >= N_B
WB = (1 << K_B) - N_B + 1               # branch-free wrap constant for N_B (gotcha 5)

# Before the loop: beat 0's state (indices + already computed addresses / wait parities), declared as cutlass.Int32 (gotcha 2)
sa, pa = cutlass.Int32(0), cutlass.Int32(0)
sbb, pb = cutlass.Int32(0), cutlass.Int32(0)
mba, mbb = mb + 8 * MB_A, mb + 8 * MB_B
wpa, wpb = cutlass.Int32(1), cutlass.Int32(1)  # pass 0 waits on the preceding phase
ra, rb = dA, dB                                # slot base + read-buffer offset

for g in cutlass.range(G, unroll=1):           # must be rolled for the carried variables to be real registers (gotcha 1)
    mbar_wait(mba, wpa)                        # consumers: every operand is a carried register
    mbar_wait(mbb, wpb)
    va = lds(ra + rofs)
    vb = lds(rb + rofs)
    # ---- state for g+1: computed at the start of this beat's body, a whole body away from the next beat's consumers ----
    wa   = (sa + 1) >> K_A                     # power of 2: wrap bit
    sa_n = (sa + 1) & (N_A - 1)
    pa_n = pa ^ wa
    wb   = (sbb + WB) >> K_B                   # not a power of 2: slot N_B-1 -> wrap (branch-free, no division)
    sb_n = sbb + 1 - N_B * wb
    pb_n = pb ^ wb
    mba_n = mb + 8 * MB_A + 8 * sa_n
    mbb_n = mb + 8 * MB_B + 8 * sb_n
    ra_n = dA + sa_n * SLOT_B + pa_n * BUF_B
    rb_n = dB + sb_n * SLOT_B + pb_n * BUF_B
    # ---- this beat's independent work ----
    ...
    sts((ra ^ BUF_B) + wofs, out_a)            # write buffer = read buffer ^ BUF_B
    sts((rb ^ BUF_B) + wofs, out_b)
    cute.arch.sync_warp()
    if lane == 0:
        mbar_arrive(mba); mbar_arrive(mbb)     # wait and arrive use the same carried value (gotcha 3)
    # ---- the end of the beat only moves registers (usually absorbed by register allocation) ----
    sa, pa, sbb, pb = sa_n, pa_n, sb_n, pb_n
    mba, mbb = mba_n, mbb_n
    wpa, wpb = pa_n ^ 1, pb_n ^ 1
    ra, rb = ra_n, rb_n
```

Control forms (compiled and timed in the same job in the probe): `site` — every point of use computes `g & (N_A-1)`, `g % N_B`, `(g // N_B) & 1` and the addresses from g on the spot;
`carry` — (slot, parity) carried and advanced at the end of the beat, addresses computed at the point of use from the carried slot.

## Gotchas

1. **The loop must be rolled (`cutlass.range(n, unroll=1)`).** Once unrolled, slot numbers become compile-time constants and addresses become immediates, and this entry does not apply; `cutlass.range` without `unroll=1`
   may be partially unrolled (see [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md)).
2. **Declare carried variables outside the loop as `cutlass.Int32(...)` and assign them inside the body.** A variable that first appears inside the loop body is not carried; a Python int initial value is turned into a constant at trace time.
3. **Two consumers of the same value must use the same carried register.** E.g. both wait and arrive use `mba`; if one uses the carried value and the other recomputes it from g, then as soon as either of the two formulas
   has a mistake (wrap, ^1), wait and arrive land on different barriers — the symptom is a hang or an intermittent data race, not a compile error.
4. **Distinguish "pass parity" from "wait parity".** The side that waits for "the previous pass has fully arrived" waits on `pass_parity ^ 1` (pass 0 goes through directly by the
   preceding-phase rule); data double buffering uses the pass parity itself. Carry both; do not `^1` again at the consumer (that would be yet another producer right next to the consumer).
5. **When N is not a power of 2, write the wrap as branch-free arithmetic.** Take `K = ceil(log2 N)`, `WB = 2**K − N + 1` (Python constants); `w = (s + WB) >> K`
   is 1 exactly when `s == N − 1`, then `s = s + 1 − N*w`, `p ^= w`. Do not write `s % N` / `s // N`, and do not write a dynamic `if s == N` either: the latter adds a basic block,
   and the carried value needs extra `SEL`s at the join. (This general formula is derived; the probe only ran one non-power-of-2 N. Another branch-free advance and the readings of three advance spellings are in
   [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md) gotcha 16.) Also note: in a rolled loop ptxas may already have strength-reduced
   `g % N` — the probe measured the same `IMAD.HI` / `IMAD.WIDE.U32` count in all three forms. So the gain of this entry **is not removing the division**,
   but moving the address arithmetic away from its consumers (gotcha 6).
6. **ptxas reorders.** "Compute g+1 at the start of the body" in source only gives the scheduler the opportunity; what must actually be checked is how far the address register of `SYNCS` is from its nearest writer in SASS
   (the probe's `try_wait SYNCS nearest-producer ADDRESS` reading). **The parity operand is an exception**: in SASS the parity of `try_wait.parity` is always
   re-encoded to bit 31 by a `SHF.L.U32 Rp, Rp, 0x1f` right next to `SYNCS` (in all three forms; carrying cannot remove it; it is part of the encoding itself, a single ALU latency). What this entry can move away are the producers of
   **the barrier address** and the data-slot addresses; accept on the distance from the `[address]` register of `SYNCS` to its nearest writer. The `x = x_n` moves at the end of the beat are usually coalesced by register allocation;
   a run of `MOV`s in SASS means the coalescing failed.
7. **Register cost.** Each ring carries 1–3 more registers (estimated, not measured); on a warpgroup at the edge of its register budget, first check whether `LDL/STL` shows up.
8. **A numeric probe cannot see whether a parity error releases early**, unless the data really depends across warps. The certification probe makes each thread read the row that **another warp** wrote in the previous pass,
   and uses a bounded spin instead of an unbounded try_wait, so a wrong parity surfaces as a data mismatch or a timeout count instead of hanging the job.

## Sources

- PTX ISA: `mbarrier.init` / `mbarrier.arrive` / `mbarrier.try_wait.parity` (phase and parity semantics, the preceding-phase rule), `ld.shared` / `st.shared`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`nvdisasm` / `cuobjdump -sass`): https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `cutlass.range(..., unroll=1)`, `cutlass.range_constexpr`, `cutlass.Int32`, `cute.arch.sync_warp`,
  `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`, `cutlass.utils.SmemAllocator`.
- Certification probe: `PROBE PASS max_diff=0.0`, round 2.
