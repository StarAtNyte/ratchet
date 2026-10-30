# consumer_warpgroup_pingpong_parity (two consumer warpgroups alternating by round parity: one role body + one warp-uniform parity scalar, cross-WG handoff through constant phases of single-slot mbarriers, the shared exchange face handed over by WAR)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry rests on the index's second class of evidence (the KEEP census of a contract-axis change): no numeric probe can see the shape itself --
> the bytes computed by two WGs are exactly the same as by one WG. What it records are the **knob** (how to spell it in the DSL), the **footprint** (which dossier axis moves how)
> and the **load-bearing conditions** (under which structure the ms readings were taken). This entry's ms readings are the side where it is **not** worth anything, recorded as is:
> a later run that sees the same shape should not treat it as the prime suspect again.

## Purpose

A consumer segment (TMEM readback → element-wise → row reduction → write the operand face) handshakes serially with the MMA thread every round. Widen this segment from
one warpgroup (4 warps) to two (8 warps): rounds with an even global round number g go to WG A, odd ones to WG B, and
the MMA thread's issue order, wait targets and arrive counts do not change by a single character -- it still sees "the full / empty of round g",
only the set of threads flipping them has changed.

Three things that must hold at the same time:

1. **One role body**: both WGs take the same branch and the same code; the only run-time difference is one warp-uniform
   scalar `p` (the parity it owns).
2. **The phases of the handoff barriers are constants**: every cross-WG signal is one-directional (one side only arrives, the other side only waits for the completion of the previous global
   round), and every single-slot barrier completes exactly once per global round, so the waiter's phase is the constant `p` or `1 - p`,
   with no phase arithmetic in the round body.
3. **The shared exchange face is handed over by WAR**: when smem is at its ceiling the two WGs can only share one exchange face, and an mbarrier with count = 4
   hands "the other side has finished reading in the previous round" to the first `STS` of this round.

## When needed (trigger)

- In the timeline / dossier, the exposed length of the consumer segment is of the same order as the round period, **and the MMA thread is waiting on this segment's full / empty**
  (waits of the `s_empty` / `p_full` kind are the main item of the MMA thread's period).
- The register pool can hold two tiers: the two consumer WGs on one tier + the producer on another, Σ(threads × tier) ≤ threads × entry allotment
  ([`warp_specialization`](warp_specialization.md) gotcha 13). Certified tree: 512 threads,
  256 × 168 + 256 × 80 = 63 488 ≤ 65 536.
- A two-slot ring (NS = NP = 2) is already in place: slot number = parity, with no slot arithmetic in the whole segment.

**Do not do it because of the following**: the segment itself is not on the period's binding chain (the MMA thread's waits do not fall on this segment's full / empty).
This entry's two landing readings are exactly that side -- see "Load-bearing conditions".

## Minimal runnable example

Skeleton extracted from the certified tree (the two consumer WGs are warps 0–7, `WG_HI` is the higher-numbered consumer WG; the names are generic):

```python
# ---- prologue (one elected thread): three local count-4 handoff barriers; each of the releasing WG's 4 warps arrives once
cute.arch.mbarrier_init(b_xchg_free, 4)       # WAR handoff of the exchange face: finished reading this round -> the other side may write next round
cute.arch.mbarrier_init(b_stat_ready, 4)      # handoff of the row statistics in round order: published this round -> the other side reads next round
cute.arch.mbarrier_init(b_tail_x, 4)          # once per item: the non-finishing side hands its partial statistics to the finishing side
# "Round -1 has already released the exchange face": pre-arrive one completion, so the first round's wait phase lands on a constant.
# b_stat_ready is deliberately not pre-arrived: an item's first round does not read the previous round's statistics, so completions stay exactly one per round.
cute.arch.mbarrier_arrive(b_xchg_free, arrive_count=4)
# The consumer release rows the MMA thread waits on (s_empty / o_empty) are pre-arrived here with a full count for generation 0:
# round 0 runs on slots nobody has read; pre-arriving makes the MMA thread's wait uniformly "the g-th release of this row", with no first-lap special case.
for i in cutlass.range_constexpr(NS):
    cute.arch.mbarrier_arrive(b_s_empty + i, arrive_count=8)

# ---- role entry: the two consumer WGs in one branch
if wg <= WG_HI:
    cute.arch.setmaxregister_increase(REG_CONSUMER)            # both consumer WGs on the same tier
    p = cute.arch.make_warp_uniform(WG_HI - wg)                # 0 / 1: the parity it owns
    tid128 = tidx % 128                                        # both WGs folded to 0..127
    bar_pair = BAR_PAIR_BASE + BAR_PAIR_STRIDE * p + ((tid128 // 32) % 2)  # this WG's own id pair
    slot = p                                                   # S slot = P slot = parity, no slot arithmetic in the whole segment
    ph_xchg = p                                                # pre-arrived once: round g waits for completion g+1, phase g % 2 = p
    ph_stat = p ^ 1                                            # not pre-arrived: round g waits for completion g, phase (g-1) % 2 = 1-p
    for it in cutlass.range(n_items):
        for k in cutlass.range(ROUNDS // 2):                   # only the rounds of its own parity
            r = 2 * k + p                                      # ROUNDS even => parity of r == parity of the global round number
            cute.arch.mbarrier_wait(b_s_full + slot, k % 2)    # handshake with the MMA thread: generation k of slot p
                                                               # (when ROUNDS % (2*NS) == 0 a count within the item suffices)
            cute.copy(ldt, s_other_half, rX)                   # first read back the half to send to the partner (does not touch the exchange face)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_wait(b_xchg_free, ph_xchg)      # the other side has finished reading the exchange face in the previous round
            # <rX -> exchange face>                             # the first STS comes after this
            cute.copy(ldt, s_own_half, rS)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(b_s_empty + slot, peer_cta_rank_in_cluster=leader_rank)
            cute.arch.barrier(barrier_id=bar_pair, number_of_threads=64)   # paired rendezvous within the WG
            # <exchange face -> accumulated in place into rS>
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(b_xchg_free)         # release right after reading: once per warp, count 4
            # an item's first round does not read the previous round's statistics (see gotcha 1)
            # <wait b_stat_ready(ph_stat) -> statistics -> publish -> arrive b_stat_ready
            #  -> write operand face slot -> arrive p_full[slot]>
```

Trace-time contract (written as an assert, not as a run-time branch): `ROUNDS % 2 == 0` -- otherwise the round parity within an item decouples from the
global parity, and every ring counter that runs on a flat count (slots and phases of NKV / NS / NP) goes wrong with it.

## Machine-read footprint (what to look at after switching to this shape)

The same package landed once on each of two baselines of the same tree (NCU dossier census + gate window):

| | Landing 1 | Landing 2 |
|---|---|---|
| contract axis `rings` | `[]` → `["stride=0x10200,wrap=mask", "stride=0x10200,wrap=mask"]` | `[]` → `["stride=0x1"]` |
| mbarrier table | 27 → 30 rows (three count-4) | same |
| `smem_dynamic` | 231 712 → 232 248 B (limit 232 448) | 231 712 → 231 992 B |
| `regs_per_thread` / spill | 128 / 0 → 128 / 0 | same |
| gate (device window) | 1.1692 → 1.1647 ms (−0.38%) | 1.1062 → 1.1183 ms (+1.09%) |
| `tensor_pct` | 46.20 → 46.61 | 50.94 → 50.59 |
| `stall_barrier` | 0.29 → 0.29 | 0.30 → 0.29 |

The two readings of `rings` are the codegen fingerprints of the same source change on the two baselines
([`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) gotcha 11, the fifth tree): they prove that the shape made it into the cubin,
and prove nothing else. The count axes (loops / mbar_objs / mbar_tiers / mbar_per_role / edges / merge) move along and are not accounted.

## Load-bearing conditions (under which structure this shape is worth something -- this entry's readings are the side where it is **not**)

Structural conditions: sm100 pair UMMA (`cta_group::2`), 512 threads / CTA, 1 CTA/SM, one resident wave; the MMA thread's chain per round
is `wait kv_full → wait s_empty → QK chain → wait p_full → PV chain → release`, and its issue order / wait targets /
arrive counts are byte-for-byte identical before and after the change; each WG blocks on exactly two named barriers per round; the exchange face is a single shared copy.

Under these conditions the two landings gave −0.38% / +1.09%, both within the noise floor (1.62%) of that tree's ablation: **doubling the consumer segment's thread count
did not change the period**. How to read it: this lever can only shorten "the length the consumer segment itself exposes"; if the period is decided by the MMA thread's own chain
or by supply, it is worth nothing; if a later run wants to open it, first confirm on the timeline that the MMA thread is waiting on this segment's full / empty.

The third part of the same package -- splitting each item's TMEM readback by column halves between the two WGs (one warp-uniform scalar moves both the
TMEM window pointer and the gmem address, the arrive count of `o_empty` 8 → 16, no new barrier): `rings` two
`0x10200` → `["stride=0x38ad0"]`, gate 1.1647 → 1.1778 ms (+1.12%), `tensor_pct` 46.61 → 46.21.
Likewise not load-bearing.

## Gotchas

1. **The pre-arrive decides whether the phase constant is `p` or `1 - p`.** A handoff barrier completes exactly once per global round: for the one pre-arrived
   once in the prologue (exchange-face WAR), round g waits for completion g+1, phase g % 2 = p; for the one not pre-arrived (statistics in round order),
   round g waits for completion g, phase (g−1) % 2 = 1 − p, and an item's first round must skip this wait, or the completions are no longer
   one per round and the constant phase breaks. The two cannot use the same pre-arrive.
2. **ROUNDS being even is a trace-time contract.** With an odd number of rounds, the parity of an item's last round equals that of the next item's first round, and the two WGs
   collide on the same parity; "WG B always owns the item's last round" (the finishing side statically determined) also breaks. Write
   `assert ROUNDS % 2 == 0`.
3. **The two WGs cannot share one pair of named barrier ids.** The paired rendezvous is a disjoint 64-thread rendezvous, each WG with its own pair
   (e.g. {2, 3} and {6, 7}), the id selected by `p`.
4. **One body, not two copies of the body.** Two hand-copied segment bodies drift; one body plus one `make_warp_uniform` scalar does not.
   Put the truly asymmetric things like "who is the finishing side / who takes the inbound load path" into a small `wg == WG_FIN`
   region; do not duplicate the segment body.
5. **WAR handoff of the exchange face: arrive after the reads, once per warp.** count = the number of warps of the releasing WG (4),
   arrive inside `elect_one`; this release is exactly the ordering that puts this WG's `LDS` before the other side's next-round `STS`.
   The wait goes before the first `STS` and after the TMEM readback (the readback does not touch the exchange face, so it is issued first).
6. **The consumer release rows the MMA thread waits on must be pre-arrived for generation 0 with a full count in the prologue.** Round 0 runs on S slots nobody has read and on an O
   nobody has drained; the pre-arrive leaves the MMA thread's wait without a first-lap special case.
7. **Fold the thread ids of both WGs to 0..127.** `tidx % 128` (not a subtraction) -- the second WG's threads are
   128..255, and the pair grouping, exchange-face slots and TMEM lanes all use the folded id.
8. **Run an offline randomly-interleaved model of a multi-WG protocol change before going to the GPU.** Before the certified tree landed, a random-scheduler model walked
   the protocol (several seeds × several input families), with a negative control (remove one wait → the model must report an error); a deadlock on the GPU only
   burns the whole job's cap ([`gpu_probe_ladder`](gpu_probe_ladder.md)).
9. **Do not take the `rings` reading as a gain.** Two `wrap=mask` or one bare `stride=0x1` are both fingerprints of this shape and unrelated to
   ms (table above).
10. **The two-slot write side: `slot = p` cannot be spelled as "a hoisted slot-0 partition iterator + a run-time whole-slot offset".** The operand ring's
    swizzle lives in the iterator (`allocate_tensor(..., swizzle=)`), and `cute.copy` with a 128-bit `CopyUniversalOp`
    is rejected at IR verification, so no contract gets to run:

        'cute.copy' op D ptr alignment (16 bits) does not meet requirement (128 bits)
        of atom universal_copy<bf16, 128 b>

    This is the same family of rejection as [`warp_mma_ldsm_operand_displaced_view`](warp_mma_ldsm_operand_displaced_view.md) gotcha 1 (the declared alignment drops to the element width with a dynamic offset),
    moved to **the D side, on a ring slot with a swizzled iterator**. On the same tiled copy
    and the same ring layout, four write-side spellings were priced against a fixed read side (read back by the layout's own stage coordinates and a compile-time slot number) --
    a 128-thread micro-kernel, no MMA / TMEM, values from gmem, element-wise readback comparison, control row = both slots written by compile-time
    coordinates, each spelling `cute.compile`d once on its own, each in its own try/except, so that one ICE row does not take the whole table with it:

    | Write-side spelling | Result |
    |---|---|
    | control row: `range_constexpr` over both slots, compile-time coordinates | max_diff 0 |
    | form 0: `[thr.partition_D(sPflat[(None, None, j)]) for j in range(NP)]` cut into two outside the round body by compile-time stage coordinates, one of the two chosen inside the round by a warp-uniform scalar branch | max_diff 0, both slots `zero_cells = [0, 0]`, **selected** |
    | form 1: `dst0.iterator + cute.assume(SLOT * j, divby=SLOT)` (whole-slot element stride `SLOT`, restating divisibility) | max_diff 0, compiles |
    | form 2: `dst0.iterator + SLOT * j` bare offset | does not lower (the error above) |
    | form 3: form 2's address with a 16-bit atom (diagnostic only) | compiles, reads back 0 |

    How to read it: what the verifier rejects is **the alignment fact of a 128-bit store landing on a dynamically offset swizzled pointer**, not the offset itself
    (form 3 lowers); `cute.assume(..., divby=)` restating divisibility fixes it, and the selected form does not need that fact at all -- the two slots
    are cut where the coordinates are still Python ints, the swizzle is folded in by the DSL itself, and inside the round only one uniform branch remains, wrapping a few
    `STS.128`, with no address arithmetic in the loop; the two partition names are only read-from subscripts in the round body and enter no region's operand bundle.
    The read side (MMA operand) takes the same slot through the partition's stage mode, the same spelling as the other operand taking the KV stage; "the write side and the read side are the same
    bijection" cannot be checked at trace time, and if it is wrong it is silently wrong numbers, not an error, so every row in the table is a readback comparison, not just a compile check.

## Sources

- **PTX ISA**: *Parallel Synchronization and Communication Instructions* -- the phase semantics of `mbarrier.init` /
  `mbarrier.arrive` / `mbarrier.try_wait.parity`, the id
  and thread-count operands of `bar.sync` / `barrier.sync` <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- **CuTe DSL public API symbols**: `cutlass.cute.arch.make_warp_uniform`, `cute.arch.mbarrier_init` /
  `mbarrier_arrive(..., arrive_count=)` / `mbarrier_arrive(..., peer_cta_rank_in_cluster=)` /
  `mbarrier_wait`, `cute.arch.barrier(barrier_id=, number_of_threads=)`, `cute.arch.elect_one`,
  `cute.arch.setmaxregister_increase`, `cute.arch.fence_view_async_tmem_load`, `cutlass.range` /
  `cutlass.range_constexpr`.
- **KEEP certification (contract-axis change, this index's second class of evidence; the eval job numbers were not recorded in the status record)**: two landings of the same package,
  axis `rings` `[]` → `["stride=0x10200,wrap=mask", "stride=0x10200,wrap=mask"]` (gate
  1.1692 → 1.1647 ms) and `[]` → `["stride=0x1"]` (gate 1.1062 → 1.1183 ms); the third part (readback split by column
  halves) `rings` → `["stride=0x38ad0"]`, gate 1.1647 → 1.1778 ms. All three correctness checks PASS.
- **Certification boundary [UNVERIFIED-BOUNDARY]**: the build chain carrying this shape was later abandoned as a whole, and the code of the same tree that finally met the bar
  has the single-consumer-WG shape -- this entry certifies the spelling, the contracts and the footprint, plus the reading "not load-bearing under this structure"; it does not certify a gain
  in any other tree.
- **Probe certification of gotcha 10**: certified by a GPU probe (the probe is not included in this release), stage `NP2-pslot` (the form table). Round 1: the spelling with a bare run-time slot offset
  died at IR verification on every contract of that probe (the error text above); round 2: the table holds -- control row 0, forms 0 / 1 / 3 lower with
  max_diff 0, form 2 does not lower; the selected form 0 is handed to the same probe's `NP2-lse` 0.009 / `NP2-out` 0.122 tolerance units,
  `NP2-rescale` 0.008 / 0.047, `NP2-relaunch3` / `NP2-shapes` (non-numeric), passing in the same round. The verdict line the driver recorded:
  `PROBE PASS (max_diff=0.121656) — 6/6 PASS` (two runs of the same probe agree; the job number was not recorded); the per-contract readings are transcribed from the KEEP's
  change record and source comments. Public API symbols used by gotcha 10: `cute.nvgpu.CopyUniversalOp`,
  `cute.make_copy_atom(..., num_bits_per_copy=)`, `cute.make_tiled_copy_tv`, `ThrCopy.partition_D`,
  `cute.assume(..., divby=)`, `cutlass.utils.SmemAllocator.allocate_tensor(..., swizzle=)`.
- Related entries: [`warp_specialization`](warp_specialization.md) (the upper bound on the sum of tiers),
  [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) (phase conventions, the `rings` interpretation of gotcha 11),
  [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) (the readback segment batched by column sub-views;
  this entry's segment body reuses it).
