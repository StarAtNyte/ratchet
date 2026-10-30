"""attn_fwd kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):
  attn_fwd(q, k, v) -> o
    q: torch.Tensor (B, H_Q, S, D) cuda bfloat16
    k: torch.Tensor (B, H_KV, S, D) cuda bfloat16
    v: torch.Tensor (B, H_KV, S, D) cuda bfloat16
    o: torch.Tensor (B, H_Q, S, D) cuda bfloat16
  Causal masking, softmax scale = 1/sqrt(D), GQA mapping
  h_kv = h_q // (H_Q // H_KV), fp32 softmax/accumulation inside.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc. Delegating the attention math to torch / SDPA / library calls fails the
harness. JIT-compile once and cache the compiled kernel across calls (the
bench times repeated calls).

----------------------------------------------------------------------------
M0 delivered architecture (milestone D0.15 build order complete; gate row)
----------------------------------------------------------------------------
The build order ran its four mounting steps, each one closed by a whole-machine
smoke (real shape, lower + launch to completion) so that a launch fault always
bisects to the mechanism mounted last.  No decision is in a transitional form
and the [PLACEHOLDER:] ledger is empty.  Step 4 was re-mounted for the D0.14
reopen (meta step 6) in the form this file now carries -- per-warpgroup
epilogue: STSM into this WG's own 64 rows of the current Q slot, a WG-internal
width-128 barrier for the STSM -> TMA ordering, this WG's own UTMASTG.4D x2
through a (64, 16, 4, 1) box at c2 = s0 + 4w, and a Q-empty of count 2 -- with
the two width-256 tile-end rendezvous deleted and the tile head moved inside
the ping-pong window, i.e. four mechanisms that fail three different ways
(silent aliasing, silent misordering, hang) and none of which a trace judges.
Its certificate is `notes/probe_d014_reopen_smoke.py`, one process per rung
under its own timeout so that a hang is reported rather than waited on: B=2
H_Q=16 D=128 at S = 8 / 256 / 1024 / 4096 (the last over three launches), every
rung lowered, launched and PASSed the campaign's own criterion
|o - ref32|_max <= 2e-2 |ref32|_max, all finite (worst rung max_diff
0.00475633).

Step 1 ("skeleton"):
  * D0.4  persistent grid = SM count, block 384, static zigzag tile schedule
  * D0.5  3 warpgroups: WG0 producer / WG1+WG2 consumer, setmaxnreg split
  * D0.7  smem buffer list: 12 mbarriers + Q x2 slots + K ring x2 + V ring x2
          = 197632 B, all four data faces K_SW128 (128B swizzle) with the
          two 64-column TMA half-blocks 0x4000 apart
  * the four TMA descriptors (Q 4D pack-GQA box, K/V 3D box, O 4D store box)

Step 2 ("producer supply ring"):
  * D0.9  the whole supply position sequence in producer warp 0
  * D0.8  the mbarrier pipeline: K/V 2-stage ring + Q double slot, slot/phase
          off a per-CTA running block count that never resets at a tile edge

Step 3 ("consumer chains + ping-pong"):
  * D0.6  MMA atoms 64x128x16: QK as 8 x SS (A = Q descriptor, B = K
          descriptor, both k-major), PV as 8 x RS (B = V descriptor read
          MN-major through a transposed view of the very same bytes)
  * D0.10 the P intermediate stays in registers ([SOURCE: RS]): the QK fp32
          accumulator is packed pairwise to bf16 straight into the PV A
          fragment, no smem round trip and no extra handshake
  * D0.11 the consumer issue order and its wait targets: K(j) full -> QK(j)
          (this trip's block), V(j-1) full -> PV(j-1) (last trip's, the
          one-block lag), wait_group(1) retires QK(j) and releases K(j)-empty
          while PV(j-1) still flies, the mask segment runs inside that flight,
          wait_group(0) retires PV(j-1) and releases V(j-1)-empty, and only
          then the two pieces that touch the retired chain's registers -- the
          O rescale and the P(j) packing.  Segment ledger: S(j) 64 + O 64 +
          P(j-1) 32 fp32/bf16 registers coexist across the issue segment, the
          row scalars are 3 resident fp32 per row (m, l, alpha), and exp2
          writes back in place so no fp32 copy of S(j) outlives the packing
  * D0.12 the two consumer warpgroups ping-pong: WG w blocks on NamedBarrier
          8+w before its issue window and is arrive-only on 9-w after it,
          both width 256; WG1 seeds barrier 8 once per CTA so WG0 leads; the
          tile head runs the same window protocol (D0.11 prologue row) and the
          tile end carries no cross-WG signal at all, so the phase offset is
          continuous across the tile boundary (D0.12 cross-tile row: WG1's
          ARV(8) on tile t's last block pairs with WG0's SYNC(8) on tile t+1's
          head); the row statistics carry NO cross-WG signal
          and, by the merge-position ruling, no per-block cross-LANE signal
          for the row sum either -- only the row max merges inside the loop
          (SHFL.BFLY 1/2), the row sum merges once at the tile end
  * D0.13 the softmax numerical path: predicate+select mask on the diagonal
          block only (row position = s0 + (row >> 4), park value -1e30),
          row max in the raw domain with FMNMX + 4-lane SHFL.BFLY 1/2, the
          scale folded into the single FFMA that feeds exp2, the O rescale in
          order after the retire, the row sum kept as a lane-private partial
          (D0.12 ruling, (d)) and 1/l deferred to the tile end ((e)).  Two
          call sites, because at the tile head the online correction is the
          identity: `_stats_seg` for a steady block, `_stats_seg_head` for
          block 0, which opens the running row scalars outright and emits
          neither the alpha path nor `_o_rescale` (m_old is the park value,
          alpha = exp2(-1.3e29) = 0, and acc_o was just zeroed).  The head
          form is bit-identical and is 64 FMUL + 2 EX2 + 2 FMNMX + 2 FMUL
          per thread per tile shorter, on the one softmax segment of the tile
          that has no PV of its own to hide behind (D0.11 prologue row)

Step 4 ("epilogue"), which cleared the placeholder ledger:
  * D0.14 (reopened form, meta step 6) the O staging reuses the *current* Q
          slot and every consumer warpgroup walks the tile end ALONE: the row
          sum's one 4-lane merge, MUFU.RCP, O*(1/l) and the bf16 packing in
          registers -> STSM.16.M88.4 x8 per warp into this WG's own 64 rows of
          that slot -> fence_view_async_shared + a WG-internal width-128
          NamedBarrier on EPI_BAR + w (form (i)) -> this WG's store warp issues
          UTMASTG.4D x2 with a (64, 16, 4, 1) box at c2 = s0 + 4w -> commit ->
          (at the next round's head, D2.3 (c)) cp_async_bulk_wait_group(0,
          read=True) -> Q-empty arrive on the count-2 barrier.  The two
          width-256 symmetric rendezvous the reference has here are gone; the
          two WGs never meet at a tile boundary.  smem stays at 197632 B:
          there is no separate O region, and each WG's 64-row half is 4096
          elements, a whole multiple of the Sw<3,4,3> period, so the halves do
          not share a byte.

The two foundation decisions are carried by the structure itself rather than by
any one step: D0.1 is the single kernel with Q resident for a whole tile, K/V
streamed through the ring, online softmax over the block loop and exactly one
write of O; D0.3 is the precision contract -- fp32 S / O / m / l, P packed to
bf16 in registers, 1/sqrt(D) folded with log2(e) into C_SCALE so exp2 is fed by
one FFMA, and a finite park value NEG_PARK for the mask.

Where the M0 gate checklist reads this file (its ncu / sass counterparts are
the driver's to measure):
  kernel_count / dominant_time_pct   one cute kernel per call, nothing else on
                                     the stream: attn_fwd -> _build -> _jit
  grid_size == 132                   _build passes the device SM count as the
                                     persistent grid (D0.4)
  block_size == 384                  NUM_THREADS = 3 warpgroups (D0.5)
  regs_per_thread / spill            PRODUCER_REGS 24 / CONSUMER_REGS 240; the
                                     consumer live set is 64 + 64 fp32 (acc_s,
                                     acc_o) + 32 + 32 packed bf16 registers
                                     (P, O staging) (D0.5 / D0.11)
  smem_dynamic in [196608, 198656]   SharedStorage = 1 KiB mbarrier arena + 3 x
                                     64 KiB faces = 197632 B (D0.7)
  tensor_pct                         the ping-pong window (PP_BAR / PP_ARV) and
                                     the wait_group(1) ladder (D0.11 / D0.12)
  smem_wavefronts / bankconf         every smem face is one K_SW128 atom and is
                                     touched only by TMA and STSM; P never
                                     round-trips through smem (D0.7 / D0.10);
                                     the two STSM bursts are no longer aligned
                                     by a rendezvous (D0.14 reopen), so this is
                                     the axis that reading answers
  mma_a_reg >= 8 / mma_a_desc >= 8   _pv_chain issues 8 RS wgmma off the packed
                                     bf16 fragment, _qk_chain 8 SS (D0.10)
  commit axis LE 0x1                 the steady block's wait_group(1) between
                                     QK(j) and the K(j)-empty release (D0.11)
  nbar publish-handoff width=256     PP_BAR / PP_ARV are the consumer-width
                                     pair; the producer never participates
  no symmetric rendezvous width=256  EPI_BAR + w is width 128 and WG-internal;
                                     the per-tile blocking rendezvous count is
                                     n (the head window plus n - 1 steady
                                     ones), with 0 at the tile end (D0.14)

----------------------------------------------------------------------------
M1 delivered state (masking-depth levers on the frozen topology; gate row)
----------------------------------------------------------------------------
  D1.1 row-statistics wait depth: gate 1, landed as one commit with the D0.12
       merge-position flip because neither pays on its own.  The steady block
       reads ARV(9-w) -> wait_group(1) -> release K(j)-empty -> [mask, last
       block only] -> _stats_seg -> wait_group(0) -> release V(j-1)-empty ->
       _o_rescale -> _pack_p: row max, the rescale factor, exp2 and the
       lane-private row sum all run with PV(j-1) still in flight, and what is
       left behind the retire is only the 64 FMUL that touch the PV
       accumulator and the 32 F2FP that overwrite its A fragment -- the two
       register sets a segment may not touch while the chain flies.  The
       segment is hideable only because D0.12 took the row sum's 4-lane SHFL
       out of the block loop: that was its one cross-lane dependency chain.
  D1.2 register-yield ladder: already in the M0 delivery and unchanged.  Every
       stage index, phase, row coordinate and tile counter is derived from
       bidx + loop counters + compile-time constants with no lane component
       (_tile_of / _nblk_of / _s0_of / the run counter), warp_idx, cta and wg
       are marked make_warp_uniform, the row scalars m and l are resident
       fragments updated in place (_stats_seg_head / _stats_seg), no fp32 copy
       of S(j) is kept -- exp2 writes back into the accumulator that fed it.
       D1.1's depth-1 arm adds one resident fp32 per row (`arun`, the alpha
       parked across wait_group(0)) and removes two SHFL per row per block.
       Measured on the gate tree: regs_per_thread 168, spill_ld / spill_st 0.
  D1.3 producer wait encoding and lead: the blocking mbarrier_wait arm, in
       _q_issue and _kv_issue.  Supply order is Q(0), then per block K(j) then
       V(j) into ring stage j & 1, the single supply warp running ahead until
       the 2-stage ring's empty wait blocks it; the next tile's Q is issued
       after the current tile's last K/V trip.  The trigger metric on the gate
       tree is stall_long_sb 0.68 (advisory reference 0.86).
  D1.4 ping-pong handoff point: arm (a) = ARV(9-w) after the PV(j-1) issue,
       arm (b) = the K(j) / V(j-1) full waits inside the window, i.e. after
       barrier(PP_BAR + w).  One blocking rendezvous per steady block, as in
       M0; the tile head and the tile tail are unchanged.

Where the M1 gate checklist reads this file:
  ms_device <= 0.2336                the {D0.12 flip, D1.1 gate 1} pair above
  tensor_pct >= 70                   PP_BAR / PP_ARV plus the wait_group(1)
                                     ladder (D0.11 / D0.12) plus the second
                                     masking layer D1.1 now adds inside
                                     PV(j-1)'s own flight
  spill == 0, regs <= 168            the D1.2 ladder above
  commit: LE 0x1 present, LE 0x0     the steady block still has exactly one
  count not increased                wait_group(1) and one wait_group(0); only
                                     the code between them grew
  merge: shfl_in_loop <= 13          the block loop's only cross-lane merge is
                                     the row max's two SHFL.BFLY per row; the
                                     row sum's merge moved to the tile end
  steady blocking rendezvous == 1    barrier(PP_BAR + w) is the only blocking
                                     NamedBarrier in the block loop; PP_ARV is
                                     arrive-only and EPI_BAR + w is per tile,
                                     width 128 and inside one warpgroup

----------------------------------------------------------------------------
M1 tail + M2 delivered state (the last of the reopened decisions)
----------------------------------------------------------------------------
  D1.2 register-yield ladder, unchanged and re-stated against the current
       segment: rung r1 -- every stage index, ring phase, row coordinate and
       tile counter comes from bidx + the loop counters + compile-time
       constants (`_tile_of` / `_nblk_of` / `_s0_of` / the per-CTA run
       counter), with no lane component, and warp_idx / cta / wg carry
       make_warp_uniform; rung r2 -- the row scalars are three resident fp32
       per row in 2-entry fragments (mrun / lrun / arun), updated in place,
       and exp2 writes back into the accumulator that fed it so no fp32 copy
       of S(j) outlives the packing.  What the ladder has to pay for is
       D1.1's depth-1 arm; the reading it is judged by is
       regs_per_thread <= 168 with spill_ld / spill_st == 0.
  D1.4 ping-pong handoff point.  (a) ARV(9-w) stays after the PV(j-1) issue,
       (b) the K(j) and V(j-1) full waits stay inside the window (after
       barrier(PP_BAR + w)) -- both at the arm the tree measured best.  (c),
       new with the D0.14 reopen: the tile head's Q-full wait is in front of
       SYNC(8+w) (the starting arm, so a late Q never holds the window shut),
       and the tail PV(n-1) is NOT wrapped in a window -- that arm was measured
       noop and leaving it out is what makes the per-tile blocking rendezvous
       count exactly n (head + n - 1 steady), the account D0.12 / the Worked
       Example give for the de-lockstepped tree.  One blocking rendezvous per
       steady block, unchanged.
  D2.1 softmax segment instruction form, in real form: the exp2 argument is
       an explicit `fma.rn.f32` PTX instruction (`_ffma(s, C_SCALE, -mc)`)
       rather than an LLVM contraction, so C_SCALE occurs only inside the one
       fused multiply-add feeding each MUFU.EX2 and no independent FADD
       precedes any EX2; the row max is `max.f32` -> FMNMX rather than an
       FSETP+FSEL pair; the mask is one predicate + one select per element
       (`_sel_ge_s32`, no effective-bit-to-float arithmetic, so no I2F
       anywhere in the segment); bf16 packing is pairwise cvt.  The
       rescale-factor path keeps `cute.math.exp2((m_old - m) * C_SCALE,
       fastmath=True)`: a fused argument would carry a ~1e22 residue at the
       park magnitude and exp2 would return inf.
  D2.2 elementwise in-flight width: the lever's domain is empty on this tree.
       Its trigger asks for >= 16 registers of headroom under 168 with
       spill == 0; regs_per_thread is 168, which for 384 threads is the
       hardware ceiling itself, and the segment ledger is already at
       S 64 + O 64 + P 32 + non-accumulator ~113 = ~241 against the
       consumer's 240.  There is no headroom to spend on width, so the
       segment is left at its element-major form.
  D2.3 epilogue refinement, all three items now at an arm.  (a) is delivered:
       `_epi_norm_pack` folds O*(1/l) into the same pass as the bf16 packing
       -- one loop over the compact pair view, each pair's two FMUL feeding
       its own F2FP, instead of one 64-wide scaling pass followed by a
       separate 32-wide packing pass.  Which row a pair belongs to is fixed
       by the C-fragment order (compact index = c0 + 2r + 4cV, so pair
       j = r + 2cV and row = j & 1), so both elements of a pair take the same
       reciprocal.  (b) is absorbed by the D0.14 reopen: the cross-WG
       rendezvous count is not 2 -> 1 but 2 -> 0, and the Q-empty count 2 that
       pays for it is in the D0.8 table, so (b) is no longer an M2 lever.
       (c) stays at the moved arm -- the O store's
       `cp_async_bulk_wait_group(0, read=True)` is in `_epi_drain` at the next
       round's QK(0) shadow, per warpgroup, with the CTA's last store drained
       after the round loop.  (d) is at form (i): one width-128 WG-internal
       barrier plus two stores per WG (four per tile per CTA), not the
       zero-barrier form (ii) with sixteen.
  D2.4 tile-loop tail, unchanged in substance: the CTA's last tile does not
       prefetch Q (`if i + 1 < n_rounds`), the supply warp leaves without a
       further empty wait, and the consumer's only tile-end barrier is the
       width-128 WG-internal one (D0.14), which contains neither the producer
       nor the other consumer WG.

----------------------------------------------------------------------------
REOPEN closing set: the reopened decisions the tile-boundary package did not
carry, re-stated on the de-lockstepped tree
----------------------------------------------------------------------------
The de-lockstep package (D0.7 / D0.8 / D0.11 / D0.12 / D0.14 / D0.15 / D1.4 /
D2.3 / D2.4) rebuilt the tile boundary.  The rest of the reopened closure is
re-stated here against that boundary rather than against the tree it was first
written on:

  D0.9  producer supply position sequence, unchanged and still conforming: one
        elected thread in producer warp 0 issues every TMA, the four
        descriptors are prefetched in the shared prologue, the prologue loads
        Q(0), and the main loop walks `K(j) empty -> K full expect_tx -> two
        64-column half blocks -> V(j) empty -> V full expect_tx -> two half
        blocks`, stage = j & 1 off the per-CTA run counter, with the next
        tile's Q issued after the tile's last K/V trip.  Every coordinate is
        `cta` + loop counters + compile-time constants, so nothing in the
        chain carries a lane component and no address-source load appears in
        the loop.  What the reopen changed on the consumer side does not reach
        it: the Q-empty it waits on now takes two arrivals instead of one,
        which is a barrier initial value (D0.8 table), not a supply order.
  D0.13 softmax numerical path, re-stated with the head specialisation above:
        (a) mask on the diagonal block only, and on the head when the tile is
        a single block; (b) row max in the raw domain, FMNMX + two
        SHFL.BFLY, the O rescale in order after the retire -- now emitted for
        steady blocks only, since at the head it is the identity; (c) one
        `fma.rn.f32` feeds each EX2; (d) the row sum stays a lane-private
        partial through the block loop; (e) 1/l and O*(1/l) at the tile end.
  D1.1  row-statistics wait depth, still gate 1 and now measured on a tree
        whose tile head runs the same window protocol as a steady block: the
        steady segment between `wait_group(1)` and `wait_group(0)` is the
        mask (last block only), the row max, the alpha and the exp2 + private
        row sum, and the only things left behind the retire are the 64 FMUL
        that touch the PV accumulator and the 32 F2FP that overwrite its A
        fragment.  The head's copy of that segment is the shortened one and
        rides the other WG's QK(0) window (D0.11 prologue row) rather than
        its own flight, which is why its alpha path was worth dropping.
  D1.2  register-yield ladder, unchanged: rungs r1 (uniform bookkeeping, no
        lane component in any index, make_warp_uniform on warp_idx / cta /
        wg) and r2 (row scalars resident in 2-entry fragments, exp2 writing
        back in place).  The head specialisation gives one register per row
        back inside the head segment -- `arun` is dead there now -- which the
        ladder does not need but does not refuse either.
  D1.3  producer wait encoding, recorded as an N/A-lever: its own trigger is
        `stall_long_sb > 0.86` or a timeline showing the consumer waiting on
        K/V full, and the reading on the de-lockstepped tree is 0.68 with the
        consumer's exposed waits sitting on the ping-pong window, not on
        supply.  Left at the blocking `mbarrier_wait` arm in `_q_issue` /
        `_kv_issue`, lead unchanged (ring depth 2 = two blocks ahead).
  D2.1  softmax segment instruction form, unchanged: explicit `fma.rn.f32`
        into each EX2 so C_SCALE occurs nowhere else and no independent FADD
        precedes an EX2, `max.f32` -> FMNMX for the row max, one predicate +
        one select per masked element and therefore no I2F, pairwise bf16
        cvt.  The head segment is the same instruction vocabulary with the
        identity terms removed, so it cannot introduce a new class.
  D2.2  elementwise in-flight width, domain still empty.  The trigger asks
        for >= 16 registers of headroom under 168; 168 is the ceiling for 384
        threads and the segment ledger is already at S 64 + O 64 + P 32 +
        non-accumulator against the consumer's 240.  The one form that would
        not need headroom -- splitting the row sum's 32-long FADD chain into
        W partial sums -- is ruled out by the decision's own interface note
        (hand interleaving / trees / 4-way partial sums measured null to
        negative), so the segment stays element-major.

----------------------------------------------------------------------------
M0 gate row (reopened tree; submitted alone, no functional delta)
----------------------------------------------------------------------------
The reopened closure [D0.7, D0.8, D0.9, D0.11, D0.12, D0.13, D0.14, D0.15,
D1.1, D1.2, D1.3, D1.4, D2.1, D2.2, D2.3, D2.4] was redone on the best tree in
two commits -- the tile-boundary de-lockstep package (bb52f541, the whole
{D0.14, D0.11 prologue row, D0.12 cross-tile} group with D0.8's Q-empty count
2 and D0.7's cost row) and the closing set (0162267, the tile-head statistics
specialisation).  [PLACEHOLDER:] ledger empty; D0.15's smoke certificate is
`notes/probe_d014_reopen_smoke.py`, four rungs S = 8 / 256 / 1024 / 4096 (the
last over three launches), every rung lowered, launched and PASSed, worst
max_diff 0.00475633, and both commits then took a whole-machine eval with
correctness PASS.

Line by line, with the reading each one is judged on taken from the last eval
of this tree (sha 0162267 `[MEASURED, status.jsonl iter 26]`; the tree this
row carries is that sha plus this docstring):

  correctness                     PASS on the last two evals of this lineage
  kernel_count == 1,              1 / 100.0
    dominant_time_pct >= 99
  grid_size == 132                132
  block_size == 384               384
  regs <= 168, spill == 0         168 / 0 / 0
  smem in [196608, 198656]        197632 dynamic + 0 static
  tensor_pct >= 58                75.18
  wavefronts_st + ld <= 393318    352439.5 + 0
  bankconf_st <= 2048 [ESTIMATE]  52320.  The one knob this lineage has ever
                                  seen move this axis is the width-256
                                  rendezvous that used to sit between
                                  `_epi_norm_pack` and `_epi_stsm`: present
                                  1334 / 2165 / 1688 / 1565, absent 135295 /
                                  70589 / 52320, over four add/remove pairs
                                  `[MEASURED, status.jsonl iters 4, 6, 9, 19,
                                  20, 25, 26]`, with `smem_wavefronts_st`
                                  tracking it one for one off a conflict-free
                                  floor of 262144 (= 1024 tiles x 128 rows x
                                  128 cols x 2 B / 128 B).  That rendezvous is
                                  the first of the two D0.14 names under
                                  "deleted" and is forbidden by the line two
                                  rows below, so the STSM bursts of the two
                                  warpgroups slide freely against each other
                                  and against the other WG's wgmma descriptor
                                  traffic on this tree.  Layout side of the
                                  §5.3 conflict branch unchanged and identical
                                  to the tree that read 1565: same K_SW128
                                  face, same `STMatrix8x8x16bOp(.., 4)` member
                                  off the PV MMA's C-frag layout, same 64-row
                                  region per WG at +0x2000*w, no scalar STS.
  mma_a_reg >= 8, mma_a_desc >= 8 `_pv_chain` 8 RS wgmma off the packed bf16
                                  fragment, `_qk_chain` 8 SS
  commit: LE 0x1 in steady loop   DEPBAR {LE 0x0 x1, LE 0x1 x1}
  nbar: publish-handoff width 256 'publish-handoff (ARV one side, SYNC other)
                                  width=128/256' (PP_BAR / PP_ARV at 256,
                                  EPI_BAR + w at 128)
  merge: shfl_in_loop <= 13       12
  nbar: no symmetric rendezvous   the width=256 symmetric entry present on
    width=256                     every tree up to da6a46f671 is gone; the
                                  remaining unwidthed symmetric entry is the
                                  WG-internal EPI_BAR + w / the prologue
                                  sync_threads
  {D0.14, D0.11 head, D0.12       device 0.2262 (da6a46f671) -> 0.2198
    cross-tile} group line        (bb52f541) / 0.2212 (0162267) = 1.0291 /
    >= 1.02, tensor >= 74,        1.0226; tensor 73.09 -> 74.92 / 75.18;
    HGMMA 32, depbar == base      wgmma_total 32 on all three; depbar
                                  histogram {LE 0x0 x1, LE 0x1 x1} + {LE 0x0
                                  x1} x2 on all three
  advisory (no verdict)           stall_barrier 1.09 / stall_long_sb 0.68 /
                                  dram_pct 7.06 against 1.21 / 0.69 / 6.76
"""

import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum


# ---------------------------------------------------------------------------
# architecture constants (M0; frozen by the milestone, not tunable)
# ---------------------------------------------------------------------------
HEAD_DIM = 128              # D
BM = 128                    # M-tile rows = GQA heads x positions (D0.2)
BN = 128                    # keys per KV block (D0.6)
GQA = 16                    # query heads per kv head
POS = BM // GQA             # 8 query positions per M-tile
HALF_D = 64                 # TMA half block: 128B swizzle width cap (D0.7)

Q_SLOTS = 2                 # Q double buffer (D0.8)
KV_STAGE = 2                # K/V ring depth (D0.8; depth 3 does not fit smem)

WG_THREADS = 128
PRODUCER_WGS = 1
CONSUMER_WGS = 2
NUM_THREADS = (PRODUCER_WGS + CONSUMER_WGS) * WG_THREADS        # 384
PRODUCER_WARPS = PRODUCER_WGS * 4                               # warps 0..3
CONSUMER_WARPS = CONSUMER_WGS * 4                               # 8
CONSUMER_THREADS = CONSUMER_WGS * WG_THREADS                    # 256

# setmaxnreg split (D0.5): 128*24 + 256*240 = 64512 <= 65536.
PRODUCER_REGS = 24
CONSUMER_REGS = 240

# NamedBarrier ids (D0.12 / D0.14).  The ping-pong pair is consumer-width
# (256): the producer never participates.  The epilogue barrier is WG-INTERNAL
# (width 128) after the D0.14 reopen, so the two warpgroups take two different
# ids -- one id with count 128 arrived at by 256 threads would pair them up
# arbitrarily, which is the lockstep this decision removes.
PP_BAR = 8                  # WG w blocks on PP_BAR + w ...
PP_ARV = 9                  # ... and is arrive-only on PP_ARV - w
EPI_BAR = 2                 # WG w orders STSM -> TMA store on EPI_BAR + w

# mbarrier table (D0.8): 12 objects, 8B each, in the 1 KB barrier arena.
MB_Q_FULL = 0                               # + slot, count 1 (+ expect_tx)
MB_Q_EMPTY = MB_Q_FULL + Q_SLOTS            # + slot, count 2 (D0.14 reopen)
MB_K_FULL = MB_Q_EMPTY + Q_SLOTS            # + stage, count 1 (+ expect_tx)
MB_K_EMPTY = MB_K_FULL + KV_STAGE           # + stage, count 8
MB_V_FULL = MB_K_EMPTY + KV_STAGE           # + stage, count 1 (+ expect_tx)
MB_V_EMPTY = MB_V_FULL + KV_STAGE           # + stage, count 8
NUM_MBAR = MB_V_EMPTY + KV_STAGE            # 12
KV_EMPTY_ARRIVE = CONSUMER_WARPS            # every consumer warp arrives once
# D0.8 table, D0.14 reopen: each consumer WG's store warp arrives once on the
# Q-empty of the slot it has just drained its own O store out of.
Q_EMPTY_ARRIVE = CONSUMER_WGS               # 2

# smem element counts (D0.7): 3 buffers x 2 slots x 128x128 bf16 = 192 KiB.
Q_ELEMS = BM * HEAD_DIM * Q_SLOTS           # 32768 elems = 64 KiB
KV_ELEMS = BN * HEAD_DIM * KV_STAGE         # 32768 elems = 64 KiB
HALF_OFF = HALF_D * BM                      # 8192 elems = 0x4000 B
SLOT_OFF = BM * HEAD_DIM                    # 16384 elems = 0x8000 B
TMA_BYTES = BN * HEAD_DIM * 2               # 32768 B per Q tile / KV block

# TMA box shapes = the cta_tiler of each descriptor (D0.2 / D0.7).
Q_TILER = (HALF_D, GQA, POS, 1)             # (d, head, pos, batch)
KV_TILER = (HALF_D, BN, 1)                  # (d, key, batch)
# D0.14 reopen: the O store box is per warpgroup, so it covers this WG's 4
# query positions (= its 64 rows of the M tile) instead of the tile's 8.
O_POS = POS // CONSUMER_WGS                 # 4
O_TILER = (HALF_D, GQA, O_POS, 1)           # (64, 16, 4, 1)
# WG w's 64-row region inside a Q-slot half block: rows 64w.. -> +64w*64
# elements = 0x2000 B, a whole multiple of the Sw<3,4,3> period (1024 elems).
WG_ROW_OFF = HALF_D * GQA * O_POS           # 4096 elements

# precision contract (D0.3): one constant folds 1/sqrt(D) and log2(e) so that
# exp2's argument is a single FFMA; the mask parks to a finite big negative.
C_SCALE = math.log2(math.e) / math.sqrt(HEAD_DIM)       # 0.12751743
NEG_PARK = -1.0e30


@cute.struct
class SharedStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, NUM_MBAR]
    sQ: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, Q_ELEMS], 1024]
    sK: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]
    sV: cute.struct.Align[
        cute.struct.MemRange[cutlass.BFloat16, KV_ELEMS], 1024]


# ---------------------------------------------------------------------------
# timeline probe hook (driver-owned; a no-op in the shipped tree)
# ---------------------------------------------------------------------------
def _seg(site, row=0):
    return None


# ---------------------------------------------------------------------------
# PTX escapes for the three scalar shapes the DSL has no primitive for
# (hw_kb/cutedsl/sm90/branch_free_select_mask.md (a), hw_kb/cutedsl/sm90/ptx_escape.md form 3).
# All three are pure: has_side_effects=False so ptxas may CSE and hoist them.
# ---------------------------------------------------------------------------
@dsl_user_op
def _sel_ge_s32(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """x >= y ? a : b  (x, y int32; a, b fp32)  -> one ISETP + one FSEL."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip),
         cutlass.Int32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ge.s32 p, $1, $2;\n"
        "\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _rcp_f32(a, *, loc=None, ip=None) -> cutlass.Float32:
    """1/a on the SFU -> one MUFU.RCP (D0.13 (e): once per row per tile)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
        "rcp.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _ffma(a, b, c, *, loc=None, ip=None) -> cutlass.Float32:
    """a*b + c in one FFMA — no preceding independent FADD (D2.1 fold guarantee).

    Used to ensure the scale constant C_SCALE appears only inside the single
    fma.rn.f32 that feeds each MUFU.EX2: `_ffma(s, C_SCALE, -mc)` = s*C - m*C.
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip),
         cutlass.Float32(c).ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _max_f32(a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """fallback spelling of fmax -> FMNMX (one instruction, not FSETP+FSEL)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _bfly_f32(a, lane_mask, *, loc=None, ip=None) -> cutlass.Float32:
    """fallback spelling of the butterfly shuffle -> one SHFL.BFLY."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(lane_mask).ir_value(loc=loc, ip=ip)],
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# prefer the DSL's own symbols when this release exports them
_fmax = getattr(cute.arch, "fmax", None) or _max_f32
_bfly = getattr(cute.arch, "shuffle_sync_bfly", None) or _bfly_f32


# ---------------------------------------------------------------------------
# gmem views handed to the TMA descriptors
# ---------------------------------------------------------------------------
def _qo_view(t):
    """(B, H, S, D) -> (D, H, S, B); strides (1, S*D, D, H*S*D)  [D0.2]."""
    return t.permute(3, 1, 2, 0)


def _kv_view(t):
    """(B, 1, S, D) -> (D, S, B); strides (1, D, S*D)  [D0.7]."""
    return t[:, 0].permute(2, 1, 0)


def _composed(swz, off, lay):
    """`make_composed_layout` has moved namespace between DSL releases."""
    f = getattr(cute, "make_composed_layout", None)
    if f is None:
        import cutlass.cute.core as cute_core
        f = cute_core.make_composed_layout
    return f(swz, off, lay)


def _stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        from cuda.bindings.driver import CUstream
        return CUstream(raw)
    except Exception:
        from cuda.cuda import CUstream          # older cuda-python spelling
        return CUstream(raw)


def _acc_mn(lay):
    """((2,2,V), MMA_M, MMA_N) -> (row, col).

    The zero-cost row/column view of a WGMMA fp32 accumulator (public infra
    `quack/layout_utils.py:168-205`, quoted by hw_kb/cutedsl/sm90/
    cfrag_elementwise_fastpath.md (a)).  It is a pure layout composition: the
    result aliases the very same registers.  GPU-verified on this tiling by
    `notes/probe_m0_step2_consumer.py` rung `coord` (row is constant along
    mode-1, the two marginals cover the same 64 elements, and warpgroup w owns
    rows 64w..64w+63), which is what makes the M-axis hoist of the row
    statistics legal here.
    """
    cm = cute.make_layout(lay.shape)
    shape = ((cm.shape[0][1], cm.shape[1]),
             (cm.shape[0][0], *cm.shape[0][2:], cm.shape[2]))
    stride = ((cm.stride[0][1], cm.stride[1]),
              (cm.stride[0][0], *cm.stride[0][2:], cm.stride[2]))
    return cute.composition(lay, cute.make_layout(shape, stride=stride))


def _r2s(mma):
    """The r2s tiled copy the epilogue's STSM rides on (D0.14 step 3).

    Three layers, order fixed (hw_kb/cutedsl/sm90/tma_store_epilogue.md gotcha 3):
    `get_smem_store_op(c_layout, d_ty, acc_ty)` picks the stmatrix member from
    the *output* face's layout -> `make_copy_atom(StMatrix8x8x16bOp(is_m_major,
    4))` + `make_tiled_copy_C_atom(., mma)` hangs the value grouping on this
    MMA's C-frag layout -> `make_tiled_copy_S` joins the two.

    One LayoutEnum drives both call sites (that gotcha's 9 /
    wgmma_cfrag_publish_stsm.md gotcha 9): the O face *is* the Q slot, whose
    contiguous mode is D = the PV MMA's N mode, i.e. ROW_MAJOR, i.e. the
    non-transposed member `STSM.16.M88.4`.  `mma` must be the PV (RS) tiled MMA
    that produced the accumulator, not an SS look-alike.

    Spelling sweep for the same reason as `_mma_rs`: plain Python at module
    level so the try/except is a real one.
    """
    bf, f32 = cutlass.BFloat16, cutlass.Float32
    e = LayoutEnum.ROW_MAJOR
    store_op = None
    for fn in (getattr(sm90_utils, "get_smem_store_op", None),
               getattr(sm90_utils, "sm90_get_smem_store_op", None)):
        if fn is None:
            continue
        try:
            store_op = fn(e, elem_ty_d=bf, elem_ty_acc=f32)
            break
        except Exception:
            continue
    if store_op is None:
        raise RuntimeError("no smem store op spelling was accepted")

    atom_c = None
    for make in (lambda: cute.make_copy_atom(
                     cute.nvgpu.warp.StMatrix8x8x16bOp(e.is_m_major_c(), 4),
                     bf),
                 lambda: cute.make_copy_atom(
                     sm90_utils.StMatrix8x8x16bOp(num_matrices=4), bf)):
        try:
            atom_c = make()
            break
        except Exception:
            continue
    if atom_c is None:
        raise RuntimeError("no StMatrix8x8x16bOp spelling was accepted")
    return cute.make_tiled_copy_S(store_op,
                                  cute.make_tiled_copy_C_atom(atom_c, mma))


def _mma_rs(tiler_n):
    """PV's tiled MMA: A from registers (D0.10 row 3), B (= V) MN-major.

    The keyword that carries `OperandSource.RMEM` has been spelled three
    different ways across DSL releases; sweep them at trace time rather than
    betting on one.  Plain Python on purpose (module level, unpreprocessed) so
    that the try/except really is a Python try/except.
    """
    src = warpgroup.OperandSource
    base = dict(atom_layout_mnk=(CONSUMER_WGS, 1, 1), tiler_mn=(64, tiler_n))
    for kw in (dict(base, a_source=src.RMEM),
               dict(base, a_src=src.RMEM),
               dict(base, a_operand_source=src.RMEM)):
        try:
            return sm90_utils.make_trivial_tiled_mma(
                cutlass.BFloat16, cutlass.BFloat16,
                warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
                cutlass.Float32, **kw)
        except Exception:
            continue
    raise RuntimeError("no OperandSource.RMEM kwarg spelling was accepted")


# ---------------------------------------------------------------------------
# schedule + supply + consumer helpers
#
# These live at MODULE level on purpose.  Everything the kernel calls from
# inside dynamic control flow (`cutlass.range`, a runtime `if`) is closure-
# checked by the DSL preprocessor, and a helper that captures a *value* from an
# enclosing Python scope is rejected outright ("is a closure that captures
# variables and is not supported in dynamic control flow").  Module-level
# functions have no nonlocals at all: every build-time constant (`grid`, `s`,
# ...) and every runtime object (`mbar`, the descriptors, the fragments) is
# passed in explicitly.
#
# The flip side of living outside the `@cute.kernel` body: the AST preprocessor
# only rewrites the decorated function's own source, so these bodies run as
# plain Python at trace time.  `cutlass.range_constexpr` / `cutlass.range` are
# preprocessor markers and raise ("should be preprocessed by preprocessor") if
# they are reached as real calls -- a compile-time unrolled loop in here must be
# spelled as a plain Python `range`, which is what it already means at trace
# time.  Dynamic control flow (a runtime `if`, a dynamic `range`) cannot appear
# in these helpers at all, which is why every runtime branch (the diagonal-block
# guard, the role fork, the ping-pong seed) sits in the kernel body itself;
# only `with cute.arch.elect_one():`, an ordinary context manager, works here.
# ---------------------------------------------------------------------------
def _rounds_of(cta, grid, n_full, rem, par):
    """number of zigzag rounds this CTA walks (D0.4)."""
    # the last (short) round starts from whichever end of the snake it is on;
    # written as a 0/1 indicator division, never a branch (KB: persistent
    # _tile_loop_ring_continuity gotcha 5)
    off_last = cta if par == 0 else (grid - 1) - cta
    return n_full + (rem + (grid - 1) - off_last) // grid


def _tile_of(cta, i, grid):
    """round i of CTA `cta` -> tile id (D0.4).

    Even round forward, odd round reversed, so two adjacent rounds carry a
    CTA-independent constant weight under the causal ramp.
    """
    return i * grid + cta + (i % 2) * ((grid - 1) - 2 * cta)


def _nblk_of(t, s):
    """tile id -> causal KV block count of that tile's q segment (D0.4)."""
    qseg = t // 2
    return ((s - 1 - POS * qseg) // BN) + 1


def _s0_of(t, s):
    """tile id -> first query position of that tile (D0.4: s0 = S-8*(qseg+1))."""
    return s - POS - POS * (t // 2)


def _issue(tma_atom, gmem, land, half_slice, gcoord, bar):
    """one 64-column half block, whole-warp TMA, into `land`'s slot."""
    g = cute.local_tile(gmem, *gcoord)
    d = land[half_slice]
    rank = len(gcoord[0])
    tAs, tAg = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(d, 0, rank), cute.group_modes(g, 0, rank))
    # the copy atom elects inside itself over the whole warp; wrapping it in
    # elect_one strands the other 31 lanes (KB elect_one gotcha 6)
    cute.copy(tma_atom, tAg, tAs, tma_bar_ptr=bar)


def _q_issue(rnd, mbar, tma_q, tQ, sQl, cta, grid, s):
    """load the Q tile of round `rnd` into slot rnd & 1 (D0.8 / D0.9)."""
    slot = rnd % Q_SLOTS
    # producer leads by one phase: the first use of a slot falls through the
    # empty-wait (parity 1 on a fresh barrier)
    ph = (rnd // Q_SLOTS + 1) % 2
    t = _tile_of(cta, rnd, grid)
    sblk = (s // POS) - 1 - (t // 2)          # s0 / POS, s0 = S - 8*(qseg+1)
    batch = t % 2
    _seg("copy-producer.qempty.wait.begin", rnd)
    cute.arch.mbarrier_wait(mbar + (MB_Q_EMPTY + slot), ph)
    _seg("copy-producer.qempty.wait.end", rnd)
    bar = mbar + (MB_Q_FULL + slot)
    with cute.arch.elect_one():               # expect_tx is single-thread
        cute.arch.mbarrier_arrive_and_expect_tx(bar, cutlass.Int32(TMA_BYTES))
    for hb in range(2):                       # plain Python: see helper note
        _issue(tma_q, tQ, sQl, (None, None, None, None, hb, slot),
               (Q_TILER, (hb, 0, sblk, batch)), bar)


def _kv_issue(tma_atom, gmem, land, mb_full, mb_empty, idx, ph, jblk, batch,
              stem, m, mbar):
    """one KV block (both half blocks) into ring stage `idx` (D0.8 / D0.9)."""
    _seg("copy-producer." + stem + ".wait.begin", m)
    cute.arch.mbarrier_wait(mbar + (mb_empty + idx), ph)
    _seg("copy-producer." + stem + ".wait.end", m)
    bar = mbar + (mb_full + idx)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(bar, cutlass.Int32(TMA_BYTES))
    for hb in range(2):                       # plain Python: see helper note
        _issue(tma_atom, gmem, land, (None, None, None, hb, idx),
               (KV_TILER, (hb, jblk, batch)), bar)


def _atom(mma, accumulate):
    """a fresh MMA atom for THIS call-site, carrying the accumulate field.

    The accumulate field must never be set on the shared tiled MMA: `set`
    produces a new IR value, so a `set` reached under control flow (the round
    loop, the KV block loop) leaves the shared object pointing at a value
    defined inside a child region and the enclosing loop's `scf.yield` then
    fails MLIR verification with `operand #0 does not dominate this use`
    (hw_kb/cutedsl/sm90/wgmma_ss_min_gemm.md gotcha 2, wgmma_deferred_wait.md gotcha 4).
    A per-call-site atom is the certified spelling: its value is born and dies
    inside the region that uses it.  The tiled MMA stays untouched and is still
    what the A/B/C fragments were partitioned from.
    """
    at = cute.make_mma_atom(mma.op)
    at.set(warpgroup.Field.ACCUMULATE, accumulate)
    return at


def _qk_chain(mma, acc, tQ, tK, qslot, kstage, nk):
    """QK(j): 8 x HGMMA.64x128x16, SS, zero-initialised (D0.6 / D0.11 step 1).

    Zero init is the accumulate field of the first wgmma, not 64 register
    moves: everything after it accumulates into the same fragment.
    """
    at = _atom(mma, False)
    warpgroup.fence()
    for kb in range(nk):                      # plain Python: see helper note
        cute.gemm(at, acc, tQ[(None, None, kb, qslot)],
                  tK[(None, None, kb, kstage)], acc)
        at.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()


def _pv_chain(mma, acc, tP, tV, vstage, nk):
    """PV(j-1): 8 x HGMMA.64x128x16, RS (A = the packed P registers), B = V
    read MN-major through the transposed view of the same bytes (D0.6/D0.10)."""
    at = _atom(mma, True)
    warpgroup.fence()                         # RMEM A operand needs the fence
    for kb in range(nk):                      # plain Python: see helper note
        cute.gemm(at, acc, tP[(None, None, kb)],
                  tV[(None, None, kb, vstage)], acc)
    warpgroup.commit_group()


def _stats_seg_head(acc_mn, mrun, lrun, nr, nc):
    """D0.13 (b)-(d) at the tile head: the same segment with the online
    correction folded out, because at block 0 the correction is the identity.

    The steady-state segment (`_stats_seg`) opens the running state with
    m_old = -1e30 / l_old = 0 and then computes, per row,
    `m = max(rowmax, m_old)`, `alpha = exp2((m_old - m) * C)`,
    `l = l_old * alpha + sum(p)` and, after the retire, `O *= alpha`.  At the
    tile head every one of those terms is known at trace time: m_old is the
    park value so `m == rowmax` (the 4-lane merged row max is a real score --
    pack-GQA row m sits at position s0 + (m >> 4) >= 0, so key 0 is never
    masked and the row is never fully parked), alpha is exp2(-1.3e29) = 0,
    `l = 0 * 0 + sum(p) = sum(p)`, and `acc_o` was just zeroed so `O *= 0`
    leaves it zero.  Emitting them costs, per thread per tile, 64 FMUL in
    `_o_rescale` plus the two FMNMX / two FMUL / two EX2 of the alpha path,
    and none of it can change a bit of the result.

    Dropping them shortens exactly the segment the D0.11 prologue row is
    trying to get under the other warpgroup's QK(0) window: the tile head has
    no PV in flight of its own (PV lags one block), so everything between
    wait_group(0) and _pack_p is exposed unless the other WG's chains cover
    it.  `arun` is not written here -- it is dead at the head, and the steady
    block writes it before `_o_rescale` reads it.

    Bit-identical to the old head sequence, not an approximation of it.
    """
    for r in range(nr):                       # plain Python: see helper note
        m = acc_mn[r, 0]
        for c in range(nc - 1):
            m = _fmax(m, acc_mn[r, c + 1])
        m = _fmax(m, _bfly(m, 1))
        m = _fmax(m, _bfly(m, 2))
        mrun[r] = m
        mc = -(m * C_SCALE)
        lsum = cutlass.Float32(0.0)
        for c in range(nc):
            p = cute.math.exp2(_ffma(acc_mn[r, c], C_SCALE, mc),
                               fastmath=True)
            acc_mn[r, c] = p
            lsum = lsum + p
        lrun[r] = lsum


def _mask_seg(acc_mn, cS_mn, s0, kbase, nr, nc):
    """diagonal block only (D0.13 a): keep key <= query position.

    pack-GQA row m carries head = m & 15 and position = s0 + (m >> 4), so the
    comparison is per (row, column) but the row term hoists out of the column
    loop.  One ISETP + one FSEL per element; the park value is finite (-1e30),
    never -inf, so a fully masked column pair cannot make a NaN.
    """
    for r in range(nr):                       # plain Python: see helper note
        lim = s0 + (cS_mn[r, 0][0] >> 4) - kbase
        for c in range(nc):
            acc_mn[r, c] = _sel_ge_s32(lim, cS_mn[r, c][1],
                                       acc_mn[r, c], NEG_PARK)


def _stats_seg(acc_mn, mrun, lrun, arun, nr, nc):
    """D0.13 (b)-(d): row max, the rescale factor, exp2, lane-private row sum.

    The row max is taken in the raw score domain (FMNMX is insensitive to a
    positive scale) and merged across the 4 lanes that share a row with two
    butterfly shuffles; the scale then folds into the single FFMA that feeds
    exp2 (`s*C - m*C`), so pass 3 has no separate FADD.  The rescale factor is
    `exp2((m_old - m) * C)` and NOT a fused `fma(m_old, C, -m*C)`: at the park
    magnitude the fused form's residue is ~1e22 and exp2 would return inf.
    exp2 writes back in place, which ends the raw score's live range.

    D0.12 merge-position ruling: the row SUM is NOT merged across lanes here.
    Each lane carries its own fp32 partial `lrun[r]` over the whole tile and
    rescales it in place with the block's alpha; alpha is lane-uniform for a
    given row precisely because the row max above *is* merged, so the four
    partials stay on a common scale and one 4-lane merge at the tile end
    (D0.14 step 1) reconstructs the exact same sum.  That removes the only
    cross-lane dependency chain in the segment, which is what shortens it
    enough for D1.1 to hide it inside PV(j-1)'s flight.

    D1.1 gate 1: nothing in here reads or writes the PV accumulator or the
    P(j-1) fragment, so the whole segment is legal with PV(j-1) still in
    flight (wgmma_backedge_pending_group.md: a segment instruction that
    touches an in-flight chain's accumulator or A registers collapses the
    tensor pipe).  The alpha of each row is parked in `arun` and consumed by
    `_o_rescale` after wait_group(0) retires that chain.

    D2.1: the fold is written as an explicit `fma.rn.f32` rather than left to
    the contraction heuristic, so the scale constant provably occurs only in
    the one instruction feeding each EX2 -- 64 FFMA + 64 MUFU per block, with
    no independent FADD and no I2F anywhere in the segment.
    """
    for r in range(nr):                       # plain Python: see helper note
        m = acc_mn[r, 0]
        for c in range(nc - 1):
            m = _fmax(m, acc_mn[r, c + 1])
        m = _fmax(m, _bfly(m, 1))
        m = _fmax(m, _bfly(m, 2))
        m_old = mrun[r]
        m = _fmax(m, m_old)
        mrun[r] = m
        alpha = cute.math.exp2((m_old - m) * C_SCALE, fastmath=True)
        arun[r] = alpha
        mc = -(m * C_SCALE)
        lsum = cutlass.Float32(0.0)
        for c in range(nc):
            p = cute.math.exp2(_ffma(acc_mn[r, c], C_SCALE, mc),
                               fastmath=True)
            acc_mn[r, c] = p
            lsum = lsum + p
        lrun[r] = lrun[r] * alpha + lsum


def _o_rescale(acco_mn, arun, nr, nco):
    """D0.11 step 8 / D0.13 (b): the O half of the online correction.

    Split out of `_stats_seg` by D1.1: this is the only part of the row
    statistics that touches the PV accumulator, so it is the only part that
    has to wait for wait_group(0).  64 FMUL per block off the alpha the
    segment already computed inside the flight window.
    """
    for r in range(nr):                       # plain Python: see helper note
        alpha = arun[r]
        for co in range(nco):
            acco_mn[r, co] = acco_mn[r, co] * alpha


def _pack_p(p_src, p_dst, npair):
    """P(j) -> bf16 pairs straight into the PV A fragment (D0.3 / D0.10 row 3).

    `p_src` and `p_dst` are compact 2 x npair views of the fp32 accumulator and
    of the RS A fragment; the two compact colex orders agree, so this is the
    whole of the "[SOURCE: RS]" aliasing -- no cross-lane motion, no smem.
    Legal only after wait_group(0): P(j-1) is PV(j-1)'s A operand and must not
    be overwritten while that chain is in flight.
    """
    for j in range(npair):                    # plain Python: see helper note
        pv2 = cute.arch.cvt_f32x2_bf16x2(p_src[(None, j)].load())
        p_dst[(None, j)].store(cute.TensorSSA(pv2, (2,), cutlass.BFloat16))


def _epi_norm_pack(lrun, p_src, p_dst, nr, npair):
    """D0.14 step 1: the row-sum merge, one MUFU.RCP per row, O * (1/l), then
    the fp32 -> bf16 pair packing.  Registers only -- no smem is touched and,
    after the D0.14 reopen, no barrier of any width stands in front of it: this
    warpgroup runs it the moment its own PV(n-1) retires, whatever the other
    one is doing.

    `lrun[r]` carries this lane's own partial of the tile's rescaled row sum
    (D0.12 merge-position ruling: the block loop never merges it), so the one
    and only 4-lane merge of the row sum happens right here -- two
    SHFL.BFLY per row, once per tile instead of once per KV block.

    D2.3 (a): the normalisation and the packing are ONE pass over the pair
    view, not two passes over 64 elements.  The pair view is the same compact
    2 x npair spelling `_pack_p` uses, and which row a pair belongs to is
    fixed by the wgmma C-fragment order: the accumulator's inner mode is
    (col2, row2, colV) with compact strides (1, 2, 4) -- the reading `_acc_mn`
    is built on and which `notes/probe_m0_step2_consumer.py` rung `coord`
    certified on this tiling -- so compact element index = c0 + 2r + 4cV and
    compact pair j = index >> 1 = r + 2cV, i.e. row = j & 1.  Both elements of
    a pair therefore share a row and take the same reciprocal, which is what
    lets the two FMUL feed their F2FP immediately instead of the whole 64-wide
    scaled accumulator having to be live before the packing starts.
    """
    assert nr == 2, "pair -> row identity below assumes MMA_M == 1 (nr == 2)"
    rcp = []
    for r in range(nr):                       # plain Python: see helper note
        lsum = lrun[r]
        lsum = lsum + _bfly(lsum, 1)
        lsum = lsum + _bfly(lsum, 2)
        rcp.append(_rcp_f32(lsum))
    for j in range(npair):
        s = rcp[j % nr]
        p_src[(0, j)] = p_src[(0, j)] * s
        p_src[(1, j)] = p_src[(1, j)] * s
        pv2 = cute.arch.cvt_f32x2_bf16x2(p_src[(None, j)].load())
        p_dst[(None, j)].store(cute.TensorSSA(pv2, (2,), cutlass.BFloat16))


def _epi_stsm(r2s, thr_r2s, sQ, rO, slot):
    """D0.14 step 2 (reopen): STSM.16.M88.4 x8 per warp into the current Q slot.

    O reuses the Q buffer, so the destination is the slot this tile read its Q
    from.  There is NO cross-WG rendezvous in front of it any more: the tiled
    copy is hung off the PV MMA's (2,1,1) atom layout, so warpgroup w's 128
    threads address exactly rows 64w..64w+63 -- the same rows w's own QK
    descriptor read, retired by this WG's own wait_group, and rows the other
    WG never touches.  Each 64-row half is 4096 elements, a whole multiple of
    the Sw<3,4,3> period, so the two halves do not share a byte either.

    The source is re-viewed rather than `retile`d: `partition_D`
    keeps the smem tile's rest-mode structure while the already-packed fragment
    keeps the C-frag's, and `cute.copy` compares modes one by one
    (tma_store_epilogue.md gotcha 14).  Re-viewing the iterator with the
    destination's shape is the zero-register fix; it bypasses the mode check,
    so the bijection was proved separately on the GPU with a dual marginal ramp
    (`notes/probe_m0_step4_epilogue.py` rung `epi`, both ramps max_diff = 0).
    """
    dst = thr_r2s.partition_D(sQ)[(None, None, None, slot)]
    src = cute.make_tensor(rO.iterator, cute.make_layout(dst.shape))
    cute.copy(r2s, src, dst)


def _epi_store(tma_o, tO, sOl, slot, wg, sblk4, batch):
    """D0.14 step 3 (reopen): this WG's own UTMASTG.4D x2 -> commit.

    The box is (64, 16, 4, 1) -- this warpgroup's four query positions times
    the 16 GQA heads, i.e. exactly the 64 rows it just STSM'd -- so its third
    gmem coordinate is s0 + 4w, in units of 4 positions: 2*(s0/8) + w.  The
    smem side is the same Q-slot face read through `o_land`, whose extra mode
    carries the +0x2000*w row offset.  Two stores per WG, four per tile per
    CTA, and no thread outside this warpgroup is involved.

    The pack-GQA row identity m = head + 16*(pos - s0) is inverted by the same
    box arithmetic that built it (D0.2).  Whole-warp issue: the atom elects
    internally and a single-lane issue is UB (tma_store_epilogue.md gotcha 6).
    The drain is NOT here (D2.3 c).
    """
    for hb in range(2):                       # plain Python: see helper note
        g = cute.local_tile(tO, O_TILER, (hb, 0, sblk4, batch))
        d = sOl[(None, None, None, None, hb, slot, wg)]
        tAs, tAg = cpasync.tma_partition(
            tma_o, 0, cute.make_layout(1),
            cute.group_modes(d, 0, 4), cute.group_modes(g, 0, 4))
        cute.copy(tma_o, tAs, tAg)
    cute.arch.cp_async_bulk_commit_group()


def _epi_drain():
    """D2.3 (c): the throttle point of the O store, moved off the store itself.

    Per warpgroup after the D0.14 reopen: each WG's store warp drains only its
    own two stores and then arrives once on the Q-empty of count 2, so the
    producer refills the slot when BOTH halves have been read out of it.

    `read=True` waits for the smem source to be drained, not for the global
    write to land.  What it protects is the *next* overwrite of that Q slot --
    the producer's next Q fill (gated by the Q-empty arrive that follows it)
    and, two tiles later, this consumer's own STSM.  So it belongs next to the
    overwrite, not next to the store: parked right after the store it leaves
    the whole tile boundary waiting behind it, parked here it runs in the
    shadow of the next tile's QK(0) chain (tma_store_epilogue.md gotcha 13).
    Issued unconditionally: a warp with no bulk group in flight returns at
    once, so the CTA's first round needs no special case.
    """
    cute.arch.cp_async_bulk_wait_group(0, read=True)


# ---------------------------------------------------------------------------
# build + launch
# ---------------------------------------------------------------------------
def _build(q, k, v, o, grid):
    b, h_q, s, d = q.shape
    n_tiles = (s // POS) * b

    # --- static zigzag schedule (D0.4), uniform-register arithmetic only ---
    # the schedule itself is `_rounds_of` / `_tile_of` / `_nblk_of` at module
    # level; only their build-time parameters are computed here
    n_full = n_tiles // grid
    rem = n_tiles - n_full * grid
    par = n_full % 2

    # --- smem faces (D0.7): one K_SW128 atom drives all four exchange faces --
    @cute.jit
    def _jit(mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
             mO: cute.Tensor, stream):
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, cutlass.BFloat16, HEAD_DIM),
            cutlass.BFloat16)
        # (rows, D, slots): tile_to_shape puts the two 64-column half blocks
        # 8192 elements (0x4000 B) apart, the slot stride is 16384 (0x8000 B)
        q_face = cute.tile_to_shape(atom, (BM, HEAD_DIM, Q_SLOTS),
                                    order=(0, 1, 2))
        kv_face = cute.tile_to_shape(atom, (BN, HEAD_DIM, KV_STAGE),
                                     order=(0, 1, 2))
        # PV wants V with the key axis as the contraction mode, i.e. MN-major.
        # sVt[(k, n, s)] == sV[(n, k, s)]: sV's colex index at (n, k, s) is
        # n + k*BN + s*BN*D, hence the transposer strides (BN, 1, BN*D).  This
        # is a second *reading* of the same bytes -- nothing is transposed in
        # smem (D0.10 row 2, hw_kb/cutedsl/sm90/wgmma_mn_major_operand.md step 3).
        vt_face = cute.composition(
            kv_face, cute.make_layout((HEAD_DIM, BN, KV_STAGE),
                                      stride=(BN, 1, BN * HEAD_DIM)))

        # TMA box layouts: same swizzle as the atom, box-shaped view of one
        # half block.  Q/O box (d, head, pos, batch) = (64, 16, 8, 1) over the
        # (D, H, S, B) view; K/V box (d, key, batch) = (64, 128, 1).
        #
        # The box strides are the *_face arithmetic rewritten in box
        # coordinates.  tile_to_shape lays the (8, 64) K_SW128 atom out with a
        # uniform row stride of 64 elements ((m%8)*64 + (m//8)*512 == m*64),
        # the second 64-column half block at +8192 (0x4000 B) and the next slot
        # at +16384 (0x8000 B); a pack-GQA row is m = head + 16*pos, hence the
        # head stride 64 and the pos stride 16*64.  Both +8192 and +16384 are
        # multiples of the Sw<3,4,3> period (1024 elements), so the half-block
        # and slot bases do not disturb the swizzle: the box view and the face
        # view address the same bytes.
        swz = cute.make_swizzle(3, 4, 3)
        q_box = _composed(
            swz, 0, cute.make_layout((HALF_D, GQA, POS, 1),
                                     stride=(1, HALF_D, HALF_D * GQA,
                                             HALF_D * GQA * POS)))
        kv_box = _composed(
            swz, 0, cute.make_layout((HALF_D, BN, 1),
                                     stride=(1, HALF_D, HALF_D * BN)))
        # D0.14 reopen: the O box is half the Q box along the position mode,
        # so that each consumer WG stores its own 64 rows without ever having
        # to wait for the other one.  4096 elements is a whole multiple of the
        # Sw<3,4,3> period (1024), so the half-tile base leaves the swizzle
        # pattern of the box identical to the face's.
        o_box = _composed(
            swz, 0, cute.make_layout((HALF_D, GQA, O_POS, 1),
                                     stride=(1, HALF_D, HALF_D * GQA,
                                             WG_ROW_OFF)))

        # ... and the same boxes with the half-block and slot modes kept, so
        # that one landing tensor per buffer serves every (half, slot): the
        # half is a compile-time slice, the slot a runtime one.  `o_land`
        # carries one more runtime mode, the warpgroup's +0x2000*w row offset.
        q_land = _composed(
            swz, 0, cute.make_layout((HALF_D, GQA, POS, 1, 2, Q_SLOTS),
                                     stride=(1, HALF_D, HALF_D * GQA, 0,
                                             HALF_OFF, SLOT_OFF)))
        o_land = _composed(
            swz, 0, cute.make_layout((HALF_D, GQA, O_POS, 1, 2, Q_SLOTS,
                                      CONSUMER_WGS),
                                     stride=(1, HALF_D, HALF_D * GQA, 0,
                                             HALF_OFF, SLOT_OFF, WG_ROW_OFF)))
        kv_land = _composed(
            swz, 0, cute.make_layout((HALF_D, BN, 1, 2, KV_STAGE),
                                     stride=(1, HALF_D, 0,
                                             HALF_OFF, SLOT_OFF)))

        tma_q, tQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, q_box,
            (HALF_D, GQA, POS, 1))
        tma_k, tK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mK, kv_box, (HALF_D, BN, 1))
        tma_v, tV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mV, kv_box, (HALF_D, BN, 1))
        tma_o, tO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, o_box, O_TILER)

        # MMA atoms (D0.6): one 64x128x16 bf16->fp32 warpgroup atom per
        # consumer WG, so the (2,1,1) atom layout covers the 128-row M tile.
        mma_qk = sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            cutlass.Float32, atom_layout_mnk=(CONSUMER_WGS, 1, 1),
            tiler_mn=(64, BN))
        mma_pv = _mma_rs(HEAD_DIM)
        # the epilogue's register->smem copy hangs off the PV MMA's C-frag
        # layout, so it is built from `mma_pv` and from nothing else (D0.14)
        r2s = _r2s(mma_pv)

        _kernel(tma_q, tQ, tma_k, tK, tma_v, tV, tma_o, tO,
                mma_qk, mma_pv, r2s,
                q_face, kv_face, vt_face, q_land, o_land, kv_land).launch(
            grid=[grid, 1, 1],
            block=[NUM_THREADS, 1, 1],
            min_blocks_per_mp=1,        # without it ptxas drops setmaxnreg
            stream=stream)

    @cute.kernel
    def _kernel(tma_q: cute.CopyAtom, tQ: cute.Tensor,
                tma_k: cute.CopyAtom, tK: cute.Tensor,
                tma_v: cute.CopyAtom, tV: cute.Tensor,
                tma_o: cute.CopyAtom, tO: cute.Tensor,
                mma_qk: cute.TiledMma, mma_pv: cute.TiledMma,
                r2s: cute.TiledCopy,
                q_face, kv_face, vt_face, q_land, o_land, kv_land):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta = cute.arch.make_warp_uniform(bidx)

        # ---- shared prologue: everything that both roles need is built here,
        # BEFORE the single role fork, so that it compiles under the entry
        # budget rather than under the producer's reduced one (KB
        # warp_specialization gotcha 11).
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar = storage.mbar.data_ptr()
        sQ = storage.sQ.get_tensor(q_face.outer, swizzle=q_face.inner)
        sK = storage.sK.get_tensor(kv_face.outer, swizzle=kv_face.inner)
        # the MN-major reading of the very same V bytes (D0.10 row 2)
        sVt = storage.sV.get_tensor(vt_face.outer, swizzle=vt_face.inner)
        # the same three faces seen through the descriptors' box modes
        # (d, ..., half, slot) -- TMA landing side of the exchange faces
        sQl = storage.sQ.get_tensor(q_land.outer, swizzle=q_land.inner)
        # the same Q slot again, but cut into the two per-WG 64-row regions the
        # D0.14 reopen stores out separately
        sOl = storage.sQ.get_tensor(o_land.outer, swizzle=o_land.inner)
        sKl = storage.sK.get_tensor(kv_land.outer, swizzle=kv_land.inner)
        sVl = storage.sV.get_tensor(kv_land.outer, swizzle=kv_land.inner)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_o)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(Q_SLOTS):
                    cute.arch.mbarrier_init(mbar + (MB_Q_FULL + i),
                                            cutlass.Int32(1))
                    cute.arch.mbarrier_init(mbar + (MB_Q_EMPTY + i),
                                            cutlass.Int32(Q_EMPTY_ARRIVE))
                for i in cutlass.range_constexpr(KV_STAGE):
                    cute.arch.mbarrier_init(mbar + (MB_K_FULL + i),
                                            cutlass.Int32(1))
                    cute.arch.mbarrier_init(
                        mbar + (MB_K_EMPTY + i),
                        cutlass.Int32(KV_EMPTY_ARRIVE))
                    cute.arch.mbarrier_init(mbar + (MB_V_FULL + i),
                                            cutlass.Int32(1))
                    cute.arch.mbarrier_init(
                        mbar + (MB_V_EMPTY + i),
                        cutlass.Int32(KV_EMPTY_ARRIVE))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        n_rounds = _rounds_of(cta, grid, n_full, rem, par)

        # ---- the one and only role fork; dec/inc is its first statement ----
        # The supply position sequence (D0.9) below is `_q_issue` / `_kv_issue`
        # at module level.  Every coordinate they compute comes from `cta`, the
        # loop counters and compile-time constants: no lane component, so the
        # whole bookkeeping chain stays in uniform registers and fits the
        # producer's 24.
        if warp_idx < PRODUCER_WARPS:
            cute.arch.setmaxregister_decrease(PRODUCER_REGS)
            _seg("copy-producer.entry")
            if warp_idx == 0:                    # single supply warp (D0.9)
                # prologue: this CTA's Q(0)
                _q_issue(0, mbar, tma_q, tQ, sQl, cta, grid, s)
                run = cutlass.Int32(0)           # never resets at a tile edge
                for i in cutlass.range(n_rounds, unroll=1):
                    t = _tile_of(cta, i, grid)
                    n_blk = _nblk_of(t, s)
                    batch = t % 2
                    for j in cutlass.range(n_blk, unroll=1):
                        m = run + j
                        idx = m % KV_STAGE
                        ph = (m // KV_STAGE + 1) % 2
                        _kv_issue(tma_k, tK, sKl, MB_K_FULL, MB_K_EMPTY,
                                  idx, ph, j, batch, "kempty", m, mbar)
                        _kv_issue(tma_v, tV, sVl, MB_V_FULL, MB_V_EMPTY,
                                  idx, ph, j, batch, "vempty", m, mbar)
                    # the next tile's Q goes out at the end of this tile's last
                    # K/V trip; the CTA's last tile does not prefetch (D0.8)
                    if i + 1 < n_rounds:
                        _q_issue(i + 1, mbar, tma_q, tQ, sQl, cta, grid, s)
                    run = run + n_blk
        else:
            cute.arch.setmaxregister_increase(CONSUMER_REGS)
            _seg("mma-consumer.entry")

            # ---- per-warpgroup and per-thread partitioning (D0.6 / D0.11) --
            # A/B/accumulator *shapes* are per warpgroup (which 64-row half of
            # the M tile this WG owns); the C *element* partitioning is per
            # thread.  Consumer thread ids start at WG_THREADS, so the tiled
            # MMA is sliced with the consumer-local index.
            ctid = tidx - WG_THREADS
            wg = cute.arch.make_warp_uniform(warp_idx // 4 - PRODUCER_WGS)
            wg_layout = cute.make_layout(CONSUMER_WGS, stride=WG_THREADS)
            wgs_qk = mma_qk.get_slice(wg_layout(wg))
            wgs_pv = mma_pv.get_slice(wg_layout(wg))
            thr_qk = mma_qk.get_slice(ctid)

            tSrQ = wgs_qk.make_fragment_A(wgs_qk.partition_A(sQ))
            tSrK = wgs_qk.make_fragment_B(wgs_qk.partition_B(sK))
            tOrP = wgs_pv.make_fragment_A(wgs_pv.partition_shape_A((BM, BN)))
            tOrV = wgs_pv.make_fragment_B(wgs_pv.partition_B(sVt))
            acc_s = cute.make_fragment(wgs_qk.partition_shape_C((BM, BN)),
                                       cutlass.Float32)
            acc_o = cute.make_fragment(wgs_pv.partition_shape_C((BM, HEAD_DIM)),
                                       cutlass.Float32)
            nk_qk = cute.size(tSrQ.shape[2])            # 8 = D / 16
            nk_pv = cute.size(tOrP.shape[2])            # 8 = BN / 16

            # zero-cost row/col views of the two accumulators, and the static
            # coordinate tensor the mask segment reads its row position from.
            # O needs no such tensor any more: with D0.14 mounted the output
            # coordinates are the TMA box arithmetic, not per-element indices.
            acc_s_mn = cute.make_tensor(acc_s.iterator, _acc_mn(acc_s.layout))
            acc_o_mn = cute.make_tensor(acc_o.iterator, _acc_mn(acc_o.layout))
            cS = thr_qk.partition_C(cute.make_identity_tensor((BM, BN)))
            cS_mn = cute.make_tensor(cS.iterator, _acc_mn(cS.layout))
            NR = cute.size(acc_s_mn, mode=[0])          # 2 rows per thread
            NC = cute.size(acc_s_mn, mode=[1])          # 32 score columns
            NCO = cute.size(acc_o_mn, mode=[1])         # 32 output columns

            # the P pair views: same compact colex order on both sides, which
            # is exactly what makes the RS aliasing free (D0.10 row 3)
            NPAIR = cute.size(tOrP) // 2
            p_src = cute.make_tensor(acc_s.iterator,
                                     cute.make_layout((2, NPAIR)))
            p_dst = cute.make_tensor(tOrP.iterator,
                                     cute.make_layout((2, NPAIR)))

            # row scalars live in 2-entry fragments (D0.11 ladder r2).  `lrun`
            # is this lane's private partial of the row sum -- the 4-lane merge
            # is deferred to the tile end (D0.12) -- and `arun` carries each
            # row's rescale factor across wait_group(0), from the segment that
            # now runs inside PV(j-1)'s flight to the O rescale that cannot
            # (D1.1).  Three fp32 per row, 6 registers for NR = 2.
            mrun = cute.make_fragment(NR, cutlass.Float32)
            lrun = cute.make_fragment(NR, cutlass.Float32)
            arun = cute.make_fragment(NR, cutlass.Float32)

            # ---- D0.14 epilogue state ---------------------------------------
            # the bf16 staging of O, plus the same pair-view aliasing the P
            # packing uses.  One fragment for the whole kernel (32 registers):
            # it is written and drained inside one tile end, never carried.
            rO = cute.make_fragment(cute.size(acc_o), cutlass.BFloat16)
            NPAIR_O = cute.size(acc_o) // 2
            o_src = cute.make_tensor(acc_o.iterator,
                                     cute.make_layout((2, NPAIR_O)))
            o_dst = cute.make_tensor(rO.iterator,
                                     cute.make_layout((2, NPAIR_O)))
            # the r2s thread slice, like the MMA slices, is consumer-local
            # (tma_store_epilogue.md gotcha 4)
            thr_r2s = r2s.get_slice(ctid)

            # ---- D0.12 seed: WG1 arrives on PP_BAR once per CTA so that WG0
            # falls through its first blocking window and leads the ping-pong.
            if wg == 1:
                cute.arch.barrier_arrive(barrier_id=PP_BAR,
                                         number_of_threads=CONSUMER_THREADS)

            run = cutlass.Int32(0)
            for i in cutlass.range(n_rounds, unroll=1):
                t = _tile_of(cta, i, grid)
                n_blk = _nblk_of(t, s)
                s0 = _s0_of(t, s)
                batch = t % 2
                qslot = i % Q_SLOTS
                _seg("mma-consumer.qfull.wait.begin", i)
                cute.arch.mbarrier_wait(mbar + (MB_Q_FULL + qslot),
                                        (i // Q_SLOTS) % 2)
                _seg("mma-consumer.qfull.wait.end", i)

                # the running row scalars are opened by the head segment
                # itself (`_stats_seg_head` writes mrun / lrun outright), so
                # there is no separate initialisation pass any more
                acc_o.fill(0.0)

                # ---- tile head, block 0 (D0.11 prologue row, D0.14 reopen):
                # the head now runs the SAME window protocol as a steady block
                # -- SYNC(8+w) -> K(0) full -> QK(0) x8 -> ARV(9-w) -- and
                # differs from one only by having no PV chain to issue (PV lags
                # a block).  softmax(0) therefore falls under the other WG's
                # QK(0) window instead of being done by both WGs at once, and
                # because the tile end no longer rendezvous (D0.14) the phase
                # offset the ping-pong built carries straight across the tile
                # boundary: WG1's ARV(8) on tile t's last block pairs with
                # WG0's SYNC(8) on tile t+1's head (D0.12 cross-tile row).
                # The Q-full wait stays in front of the window (D1.4 (c)
                # starting arm) so a late Q never holds the window shut.
                idx0 = run % KV_STAGE
                _seg("mma-consumer.window.wait.begin", run)
                cute.arch.barrier(barrier_id=PP_BAR + wg,
                                  number_of_threads=CONSUMER_THREADS)
                _seg("mma-consumer.window.wait.end", run)
                _seg("mma-consumer.kfull.wait.begin", run)
                cute.arch.mbarrier_wait(mbar + (MB_K_FULL + idx0),
                                        (run // KV_STAGE) % 2)
                _seg("mma-consumer.kfull.wait.end", run)
                _qk_chain(mma_qk, acc_s, tSrQ, tSrK, qslot, idx0, nk_qk)
                cute.arch.barrier_arrive(barrier_id=PP_ARV - wg,
                                         number_of_threads=CONSUMER_THREADS)

                # ---- D2.3 (c): THIS WG's own store of the previous tile is
                # drained here, with this tile's QK(0) already in flight, and
                # only then is that Q slot handed back to the producer.  After
                # the D0.14 reopen every consumer WG has a store of its own, so
                # the drain and the arrive are per WG and the Q-empty count is
                # 2: the producer refills the slot when both 64-row halves have
                # been read out of it.  The drain itself is unconditional (an
                # empty bulk group returns at once); only the arrive is guarded,
                # because a spurious arrive would flip the barrier's parity and
                # desynchronise the producer.  The producer needs this slot
                # again at the end of THIS tile's K/V trips, which is after this
                # point in every case.
                if warp_idx % 4 == 0:           # each WG's store warp
                    _seg("mma-consumer.ostore.wait.begin", i)
                    _epi_drain()
                    _seg("mma-consumer.ostore.wait.end", i)
                    if i > 0:
                        with cute.arch.elect_one():   # Q-empty count is 2
                            cute.arch.mbarrier_arrive(
                                mbar + (MB_Q_EMPTY + (i + 1) % Q_SLOTS))

                _seg("mma-consumer.qk.wait.begin", run)
                warpgroup.wait_group(0)
                _seg("mma-consumer.qk.wait.end", run)
                with cute.arch.elect_one():     # one per warp: 8 = the count
                    cute.arch.mbarrier_arrive(mbar + (MB_K_EMPTY + idx0))
                # a single-block tile's head IS its diagonal block
                if n_blk == 1:
                    _mask_seg(acc_s_mn, cS_mn, s0, cutlass.Int32(0), NR, NC)
                # D0.13 at the head: the online correction is the identity
                # here (running max = park, running sum = 0, acc_o just
                # zeroed), so the head segment opens the row scalars directly
                # and neither the alpha path nor the 64-FMUL `_o_rescale` is
                # emitted.  Bit-identical, and it shortens the one softmax
                # segment of the tile that has no PV of its own to hide under
                # -- the segment the D0.11 prologue row puts inside the other
                # WG's QK(0) window.
                _stats_seg_head(acc_s_mn, mrun, lrun, NR, NC)
                _pack_p(p_src, p_dst, NPAIR)

                # ---- steady state, blocks 1 .. n_blk-1 (D0.11 steps 0-8) ----
                for j in cutlass.range(n_blk - 1, unroll=1):
                    blk = j + 1
                    m = run + blk
                    idx = m % KV_STAGE
                    pidx = (m - 1) % KV_STAGE
                    _seg("mma-consumer.window.wait.begin", m)
                    cute.arch.barrier(barrier_id=PP_BAR + wg,
                                      number_of_threads=CONSUMER_THREADS)
                    _seg("mma-consumer.window.wait.end", m)
                    _seg("mma-consumer.kfull.wait.begin", m)
                    cute.arch.mbarrier_wait(mbar + (MB_K_FULL + idx),
                                            (m // KV_STAGE) % 2)
                    _seg("mma-consumer.kfull.wait.end", m)
                    _qk_chain(mma_qk, acc_s, tSrQ, tSrK, qslot, idx, nk_qk)
                    _seg("mma-consumer.vfull.wait.begin", m)
                    cute.arch.mbarrier_wait(mbar + (MB_V_FULL + pidx),
                                            ((m - 1) // KV_STAGE) % 2)
                    _seg("mma-consumer.vfull.wait.end", m)
                    _pv_chain(mma_pv, acc_o, tOrP, tOrV, pidx, nk_pv)
                    # hand the issue window to the other WG before waiting on
                    # anything of our own (D0.12: arrive-only, never blocking)
                    cute.arch.barrier_arrive(
                        barrier_id=PP_ARV - wg,
                        number_of_threads=CONSUMER_THREADS)
                    _seg("mma-consumer.qk.wait.begin", m)
                    warpgroup.wait_group(1)     # QK(j) retired, PV(j-1) flies
                    _seg("mma-consumer.qk.wait.end", m)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar + (MB_K_EMPTY + idx))
                    if blk == n_blk - 1:        # mask rides PV(j-1)'s flight
                        _mask_seg(acc_s_mn, cS_mn, s0, BN * blk, NR, NC)
                    # D1.1 gate 1: the row statistics of S(j) ride PV(j-1)'s
                    # flight too.  Legal because QK(j) is already retired (the
                    # wait_group(1) above) and because the segment touches
                    # neither acc_o nor tOrP, the two register sets PV(j-1) is
                    # still reading/writing.  D0.12's deferred sum merge is
                    # what makes the segment short enough to be hidden.
                    _stats_seg(acc_s_mn, mrun, lrun, arun, NR, NC)
                    _seg("mma-consumer.pv.wait.begin", m)
                    warpgroup.wait_group(0)     # PV(j-1) retired -> P reusable
                    _seg("mma-consumer.pv.wait.end", m)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar + (MB_V_EMPTY + pidx))
                    _o_rescale(acc_o_mn, arun, NR, NCO)
                    _pack_p(p_src, p_dst, NPAIR)

                # ---- tile tail: the last PV, NOT in an issue window ---------
                # D0.11 tail row / D1.4 (c): with the head now inside the
                # protocol, one window per KV block is the whole per-tile
                # account (n windows, the reference's n - 1 steady ones plus
                # the head).  Wrapping the tail PV as well would make it n + 1
                # and was measured noop; leaving it out is what lets a WG walk
                # straight from its last PV into its own epilogue and on into
                # the next tile without ever meeting the other one.
                last = run + n_blk - 1
                lidx = last % KV_STAGE
                _seg("mma-consumer.vfull.wait.begin", last)
                cute.arch.mbarrier_wait(mbar + (MB_V_FULL + lidx),
                                        (last // KV_STAGE) % 2)
                _seg("mma-consumer.vfull.wait.end", last)
                _pv_chain(mma_pv, acc_o, tOrP, tOrV, lidx, nk_pv)
                _seg("mma-consumer.pv.wait.begin", last)
                warpgroup.wait_group(0)
                _seg("mma-consumer.pv.wait.end", last)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mbar + (MB_V_EMPTY + lidx))

                # ---- epilogue (D0.14 reopen): per warpgroup, no cross-WG
                # signal of any kind.  Each WG enters as soon as its OWN
                # PV(n-1) has retired and walks out into the next tile's head
                # window without ever meeting the other one; the two width-256
                # symmetric rendezvous the reference has here are gone, which
                # is what stops every tile boundary from re-aligning the two
                # WGs and flattening the ping-pong's phase offset.
                # (1) row-sum merge + MUFU.RCP + O*(1/l) + bf16 packing, all in
                # registers (D2.3 (a)).
                _epi_norm_pack(lrun, o_src, o_dst, NR, NPAIR_O)
                # (2) STSM into this WG's own 64 rows of the current Q slot.
                # Legal with no rendezvous in front of it: the rows are the
                # ones this WG's own QK read, and its last read of them retired
                # at its own wait_group; the other WG neither reads nor writes
                # a byte of them.
                _epi_stsm(r2s, thr_r2s, sQ, rO, qslot)
                # (3) generic proxy (STSM) -> async proxy (TMA) ordering, now
                # settled INSIDE the warpgroup: the fence makes this thread's
                # stores visible to the async proxy, and a width-128 barrier on
                # this WG's own id makes all 128 of them visible to the warp
                # that issues the store (form (i) of D0.14; dropping either one
                # is a silent numerical error, not a crash).
                cute.arch.fence_view_async_shared()
                _seg("mma-consumer.epi.wait.begin", i)
                cute.arch.barrier(barrier_id=EPI_BAR + wg,
                                  number_of_threads=WG_THREADS)
                _seg("mma-consumer.epi.wait.end", i)
                # (4) this WG's store warp issues its own two half blocks and
                # moves on: the drain and the Q-slot handback happen at the
                # head of the next round, in the shadow of its QK(0) chain
                # (D2.3 (c)), and the Q-empty is the count-2 barrier.
                if warp_idx % 4 == 0:
                    _epi_store(tma_o, tO, sOl, qslot, wg,
                               2 * ((s // POS) - 1 - (t // 2)) + wg, batch)
                run = run + n_blk

            # the CTA's last O store has no next round to be drained in, and
            # its smem is reclaimed at exit: drain it here (D2.3 (c) tail).
            if warp_idx % 4 == 0:
                _epi_drain()

    return cute.compile(
        _jit,
        from_dlpack(_qo_view(q), assumed_align=16),
        from_dlpack(_kv_view(k), assumed_align=16),
        from_dlpack(_kv_view(v), assumed_align=16),
        from_dlpack(_qo_view(o), assumed_align=16),
        _stream())


_CACHE = {}


def attn_fwd(q, k, v):
    b, h_q, s, d = q.shape
    assert d == HEAD_DIM and h_q % GQA == 0 and s % POS == 0
    o = torch.empty_like(q)

    key = (tuple(q.shape), tuple(k.shape), q.dtype, q.device.index)
    compiled = _CACHE.get(key)
    if compiled is None:
        n_tiles = (s // POS) * b
        grid = min(
            torch.cuda.get_device_properties(q.device).multi_processor_count,
            n_tiles)
        compiled = _build(q, k, v, o, grid)
        _CACHE[key] = compiled

    # the stream is read per call, not cached: the device-window bench replays
    # this entry under CUDA graph capture, which runs on its own stream
    compiled(from_dlpack(_qo_view(q), assumed_align=16),
             from_dlpack(_kv_view(k), assumed_align=16),
             from_dlpack(_kv_view(v), assumed_align=16),
             from_dlpack(_qo_view(o), assumed_align=16),
             _stream())
    return o
