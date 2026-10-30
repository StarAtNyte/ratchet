# midchain_stage_wait (placing the consumer's wait for half-depth stages: a wait can only hide inside **MMAs already issued and in flight**, not inside the producer's lag)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is about **where the consumer-side stage-full wait goes**: when one smem stage holds only
> **part of the K depth** of an MMA chain (e.g. 64 rows × half of d), a QK chain consumes two stages,
> and the wait for the second stage can be written either "at the loop top" or "mid-chain". The two
> have identical bytes, instruction counts and registers; what differs is whether the tensor pipe has
> work during the wait. This entry gives: **what the placement is worth (measured 6.7%)**, **what its
> ceiling is (the duration of the in-flight MMAs, not the producer's lag)**, and three machine-read
> footprints. Prerequisites: the mbarrier ring is in
> [`cp_async_mbarrier_ring`](cp_async_mbarrier_ring.md), WGMMA async groups in
> [`wgmma_deferred_wait`](wgmma_deferred_wait.md).

## Purpose

In dense attention / GEMM one stage is a complete K block and one MMA chain consumes one stage, so the
wait for the next stage naturally lands after the chain ends (at the loop top), and the TMA producer is
always a few segments ahead, so the top wait hardly ever really waits. When both premises break at once
(the stage is half-depth, and the producer is a per-row gather that is slower than the consumer), where
the wait goes decides how long the tensor pipe idles:

- **Mid-chain**: `wait(stage0); 16×MMA; wait(stage1); 16×MMA; commit; wait_group(0)` — while waiting
  for stage1 the first 16 MMAs are in flight, and their execution time covers the wait.
- **Loop top**: `wait(stage0); wait(stage1); 32×MMA; …` — the first MMA can only be issued once both
  stages are complete; the wait is fully exposed.

## When needed (trigger)

Check in order:

1. The K depth of a stage is **smaller** than the K depth of an MMA chain (the chain spans stages).
   This does not hold in dense; the entry does not apply.
2. The producer is the consumer's **slow side**: in the role exposure ledger the `spin` stall share is
   far above the reference (on this entry's certification tree: 52.9% vs reference 0.2%,
   `long_scoreboard` 98%). When the producer leads, the two forms cost the same and the entry does not
   apply.
3. In the dossier the `SYNCS.PHASECHK` spins appear **as adjacent pairs** inside the HGMMA sequence
   (`…SS 32H…`), while the reference's two spins are separated by 16 HGMMAs (`…16H SS 16H…`).

## Minimal runnable example (CuTe DSL)

```python
# stage = 64 key x 256 d (half of d=512); one QK over 64 keys spans two stages
QK_SEG = 256 // 16                 # 16 x HGMMA.64x64x16 per d-slab
for t in cutlass.range(n_trip):
    cute.arch.mbarrier_wait(mbar + (FULL + slab0), t & 1)      # only the FIRST wait here
    atom.set(warpgroup.Field.ACCUMULATE, False)
    warpgroup.fence()
    for kb in cutlass.range_constexpr(QK_SEG):                  # segment 1: slab0 in flight
        cute.gemm(atom, acc_s, dA[(None, None, kb)], dB[(None, None, kb, slab0)], acc_s)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    # the mid-chain wait: 16 MMAs are already issued when this spins
    cute.arch.mbarrier_wait(mbar + (FULL + slab1), t & 1)
    for kb in cutlass.range_constexpr(QK_SEG):                  # segment 2
        cute.gemm(atom, acc_s, dA[(None, None, QK_SEG + kb)], dB[(None, None, kb, slab1)], acc_s)
    warpgroup.commit_group()
    warpgroup.wait_group(0)
```

`acc_s` stays live across the wait (both segments accumulate into the same set of registers); it must
not be spilled / reset at the wait. No `commit_group()` is needed before the wait: the mbarrier spin
does not look at the wgmma group state, and adding it generates **the same cubin** (see footprint 3).

## Machine-read footprint (acceptance criteria)

1. **Placement A/B timing** (the criterion of this entry's certification probe): moving only one
   `mbarrier_wait` on the same tree, `top` must be significantly slower than `mid`. Measured (this
   entry's certification tree, H100, three rounds in the same job): mid 3.4103 / 3.4103 / 3.4105 ms,
   top 3.6542 / 3.6542 / 3.6547 ms → **+7.2% time / −6.7% ratio**, three-round jitter < 0.02%, HGMMA
   count (44) and LDL/STL (20/20) identical in both versions.
2. **SASS sequence**: in program order of `re.findall(r"\b(HGMMA|SYNCS\.PHASECHK)")`, the mid-chain form
   reads `…16H SS 16H…` (the two spins separated by 16 HGMMAs), the top form reads `…SS 32H…`
   (`adjacent_spin_pairs` one higher). The number of spin regions in the dossier's `orch_tools` is
   unchanged; their position changes.
3. **cubin identity**: adding `warpgroup.commit_group()` before the wait does not change the cubin sha
   (measured: both versions have the same cubin sha), showing that the gain of the "mid-chain wait"
   comes entirely from program order and has nothing to do with wgmma groups.

## Ceiling (the most important reading of this entry)

**The wait that placement hides = the execution time of the in-flight MMAs, not one MMA more.** On this
entry's probe 16 HGMMA.64x64x16 are worth about 0.24 ms/kernel, and the placement gain is exactly
3.654 − 3.410 = 0.244 ms. Meanwhile the exposed ns of `spin` in the role exposure ledger **barely moves**
between the two versions: mid 1.129 ms (share 52.9%), top 1.233 ms (51.7%); tensor duty cycle 37.8% vs
35.2%. In other words the producer lags by ~1.1 ms, the mid-chain wait swallows only 0.24 ms of it, and
the rest is exposed as before.

So when you see a high `spin` share, first compute "in-flight MMA duration": wait placement can recover
at most that much; the remaining gap is on the producer side (supply depth, index prefetch, address
chain), not in the consumer's wait placement. Moving the consumer's waits around further will not make
it faster — this is an observed degeneration (one round −17.9%).

## Gotchas

1. **Reading "two adjacent spins" as normal**: in dense experience the top wait is the norm, and the
   dossier will not raise an alarm; it is only a problem when stage depth < chain depth — look at the
   geometry first, then at the placement.
2. **Read the geometry wrong and there is no mid-chain**: if the stage is read as "32 rows × full
   depth", one chain consumes one stage, the wait naturally sits at the top, and "mid-chain" has nowhere
   to go — reading the axis as "32 rows × full depth" and the top wait are two sides of the same error.
   For judging the axis see the recon rule "a discriminating experiment that ran empty is not a ruling".
3. **Using `commit_group` to get "stronger in-flight"**: no effect, the cubin is unchanged. What is in
   flight is the HGMMAs already issued, not the group.
4. **Using placement to rescue producer lag**: see the ceiling section. When spin exposed ns is far
   larger than the in-flight MMA duration, placement has hit its ceiling; work on the producer.

## Sources

- ✅ Verified: certified by a GPU probe (the probe is not included in this release; based on a
  gather-attention tree that had passed first-green)
  (H100, cutlass-dsl 4.6.1 image): three variants × three rounds,
  `PROBE PASS top delta_vs_base=-6.69% gates=ok`; `mid_commit` reports
  `gates=cubin_same_as_base` because its cubin equals base (which is exactly footprint 3's conclusion,
  not a probe failure). Both versions have role exposure ledger and PM timeline
  `ROLE-OK … tensor=` / `PMTL-OK`.
- Price tags: `[PRICED: -6.7% (mid vs top)]`; ceiling `[PRICED: placement gain 0.244 ms
  = in-flight duration of 16×HGMMA; spin exposure 1.13→1.23 ms unmoved]`.
- Public interfaces: `cutlass.cute.arch.mbarrier_wait`, `cutlass.cute.nvgpu.warpgroup.{fence,commit_group,wait_group}`,
  PTX `mbarrier.try_wait` / `wgmma.commit_group` / `wgmma.wait_group`.
