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

REOPEN -- D0.1 is persistent now, and G0 comes with it.
  The milestone patch reopened the whole D0.x closure.  The one decision whose
  CONTENT changed is D0.1: the grid is no longer "one CTA per Q tile" (1024
  CTAs, one tile each, exit) but "one CTA per SM" (132), each walking several Q
  tiles of a static boustrophedon schedule.  D0.3 / D0.9 / D0.10 / D0.13 are its
  (x) partners (the G0 package): the Q slot lifecycle, the smem map, the supply
  protocol and the epilogue all hang off the same question of what happens at a
  tile boundary.  Everything else in the closure -- D0.2, D0.4-D0.7, D0.11,
  D0.12, D1.1-D1.4, D2.1-D2.3 -- is unchanged text and is carried through this
  restructure in the form the sections below already describe; those sections
  stay, and where they say "grid 1024" or "the CTA's tile" they now mean "the
  CTA's CURRENT tile".
  G0 ships through the milestone's declared stepwise channel, three commits:
    part 1 of 3  -- the tile loop itself, the rings made continuous across the
                 boundary, one Q slot reloaded per tile behind a real
                 full/empty handshake, and the O store moved to the consumer
                 side and waited on the spot.
    part 2 of 3  (THIS revision) -- the second Q slot (+32 KiB -> 197632 B, 12
                 mbarriers) and the Q(t+1) prefetch issued after tile t's R-th
                 KV stage instead of at the top of tile t+1.
    part 3 of 3  -- the store wait deferred to just before the slot is reused
                 (Q-empty arrives after the bulk wait, one tile later).
  The package's reading is settled at part 3; parts 1 and 2 are expected to be
  about flat, because part 1 pays the per-tile store wait on the critical path
  that part 3 exists to remove, and part 2 buys the Q load a whole tile of
  cover but still pays that same store wait.

part 2 of 3 -- what is in this revision.
  D0.9  the smem map grows the second Q slot: sQ is `q_staged`, a (row, d,
        slot) 128x128x2 K-major SW128 tensor with the slot as the slow mode
        (stride BM*BK), exactly the shape the KV rings already had.  The map is
        12 mbarrier (96 B, padded to 1024) + 2x32 KiB Q + 2x32 KiB K + 2x32 KiB
        V = 197632 B, which is where the M0 gate's [196608, 199680] window
        wants it.  Every view partitioned off sQ -- the WGMMA A fragment, the
        TMA load box, the TMA store box, the epilogue STSM destination --
        carries the slot through as its LAST mode and is sliced per tile with
        `_slice_stage`, so the slot is address arithmetic and no container is
        rebuilt inside the tile loop.
  D0.10 the mbarrier table goes to 12: Q-full and Q-empty are now per slot
        (`slot = tile & 1`, consumer parity `(tile/Q_STAGE) % 2`, producer
        empty-wait parity `(tile/Q_STAGE + 1) % 2` -- the same counter-derived
        convention the KV rings use, so each slot's first use falls through
        against the fresh barrier).  The issue point moves (rule ii): the CTA's
        first Q goes out in the producer prologue, and Q(t+1) is issued from
        INSIDE tile t's supply loop, right after tile t's R-th KV stage, so it
        has the remaining n_kv - R KV iterations to land instead of standing in
        front of tile t+1's first K.  `min(R, n_kv)` is spelled as two mutually
        exclusive issue sites (in-loop at stage R-1 for n_kv >= R, post-loop
        for the short tail tiles) rather than a value computed in a runtime
        branch, which a runtime if-region would not let escape.
  D0.3 / D0.13  O is still staged in the alias of its OWN tile's Q slot and the
        store is still waited on the spot (deferring it is part 3), but the
        wait now gates a slot the producer does not need again until two tiles
        later, rather than the one it is about to load.
  Why this cannot deadlock: the Q-empty the producer blocks on inside tile t is
        slot(t+1) = slot(t-1), released by the consumer's epilogue for tile
        t-1 -- an epilogue that needs nothing the blocked supply warp still
        owes it (all of tile t-1's K/V is already in flight or landed).  The
        supply warp can stall there; it cannot close a cycle.

part 1 of 3 -- what it established (still in force).
  D0.1  grid = min(SM count, 1024) = 132, read from device properties on the
        host.  Round i hands CTA c the tile `t = i*GRID + (c if i even else
        GRID-1-c)` and stops at t >= 1024; tile t is the same work the
        non-persistent tree gave CTA t (`b = t & 1`, `s0 = (S-8) - 8*(t>>1)`),
        so D2.2's shipped s0-decreasing batch-interleaved order is preserved as
        the ORDER OF t.  The snake rather than a plain stride: tile weight
        ((s0+7)/128 + 1 KV tiles) falls linearly in t, so a plain stride gives
        CTA 0 ~144 KV tiles against CTA 131's ~112 (spread 1.125x, host
        arithmetic in the probe) while reversing every other round makes two
        consecutive rounds' weight constant in c (spread 1.000x at this shape).
        The whole map is fed by `bidx` and compile-time constants only -- no
        lane component anywhere in the chain -- so the bookkeeping stays warp
        invariant and lands in the uniform register file rather than the
        240-register consumer budget (uniform_reg_budget mechanism 1).
  D0.10 the two K/V rings are ONE STREAM PER CTA.  The stage index and the
        mbarrier parity come off a running count `kv_run` that is never reset at
        a tile boundary and is kept independently but identically by the
        producer and the consumer (`idx = run % STAGE`, consumer parity
        `(run/STAGE) % 2`, producer parity `(run/STAGE + 1) % 2` -- the same
        "producer starts a phase ahead" convention the PipelineState object
        encoded, written out of a counter so it survives the boundary).  A reset
        on one side only is a hang, not a wrong answer, which is why the probe
        has to run several tiles per CTA.  Q gets a real handshake: Q-full as
        before, Q-empty released by the consumer thread that issued O(t)'s
        store.  Q-empty's arrive count is therefore 1, not 8.
  D0.3 / D0.9  ONE Q slot in part 1 (164864 B); part 2 above replaces this with
        the two-slot map at 197632 B.
  D0.13 the O store is issued by ONE consumer warp (warp_idx 4 -- the predicate
        has to be warp aligned because the copy atom elects inside itself over
        the whole warp) and waited on the spot with
        cp_async_bulk_wait_group(0, read=True); Q-empty is arrived after that
        wait, which is what keeps Q(t+1) from overwriting bytes the store is
        still reading.  The producer is no longer a party to the epilogue, so
        the 288-wide CTA rendezvous is gone and the epilogue is two 256-wide
        consumer rendezvous (alias-may-be-overwritten, then
        all-STSM-visible-to-the-async-proxy).
  Containers: every fragment and view is allocated once outside the tile loop
        and overwritten per tile (fixed_container_tile_loop); the only per-tile
        state reset is the O accumulator and the two row statistics.
  D0.12 under persistence WG1's ping-pong pre-arrive happens ONCE PER CTA, at
        the tail of the first tile's peel -- the arrive/sync counts run on
        across the tile boundary (the last body arrive of tile t pairs with the
        first body sync of tile t+1), so repeating it per tile would be a count
        error rather than a no-op.
  Probe certificate for part 1 (notes/probe_g0_part1_persistent.py, one
  child process per rung so a hang is reported as a hang, all six contracts
  PASS):
    C1  S=512 with the CTA count pinned to 40 -- 3 or 4 tiles per CTA and the
        2-stage rings wrapping several times per CTA -- matches an in-probe
        fp32 reference inside the 2e-2 tolerance.  A ring index or parity that
        is reset at a tile boundary, a Q-slot handshake that does not
        alternate, or a pre-arrive repeated per tile all hang rather than
        mis-compute, and none of them shows up on a CTA's FIRST tile, so the
        rung has to run several (pipeline_runtime_subring gotcha 5).
    C2  the same shape pinned to 25 CTAs -- 5 or 6 tiles per CTA and the other
        parity of the short final snake round -- matches the same reference.
    C3  the campaign shape (B=2 H_Q=16 H_KV=1 S=4096 D=128) at the real grid of
        132, 7 or 8 tiles per CTA: matches at |o-ref32|_max / |ref32|_max =
        0.00311646 against a 2e-2 tolerance.  This is also the whole-machine
        smoke certificate for this tree -- lowered, launched and retired at the
        shipped shape.
    C4  determinism at the campaign shape: 12 launches, output NaN-poisoned
        before each, every one inside tolerance and bitwise identical to the
        first.  A cross-tile handshake that is merely racy rather than wrong
        survives one launch, not twelve.
    C5  the campaign cubin keeps what part 1 is not allowed to move: LDL+STL ==
        0 (the tile loop pushed no container to local memory -- the gates read
        spill as a hard line), HGMMA still 32, and `LE 0x1` still in the DEPBAR
        histogram (the D0.11 masking relation survived the restructure).
    C6  the work map is a bijection: over every (N_WORK, GRID) pair the probe
        runs, the union of the per-CTA tile lists is exactly {0 .. N_WORK-1}
        with no repeats, every CTA runs at least the wanted number of tiles,
        and the snake's spread of KV-tile weight over CTAs is reported next to
        the plain stride it replaces.

M0 construction, steps 1-6 (M0 gate PASS'd on the pre-REOPEN topology at step
6; the sections below describe the decisions as built, with D0.1 / D0.3 / D0.9 /
D0.10 / D0.13 superseded by the REOPEN block above).
  step 1-2 (already recorded: D0.1+D0.8+D0.9+D0.10)
    D0.1  grid = B * (S/8) = 1024, one Q tile per CTA, non-persistent, no cluster
    D0.8  384 threads = 1 producer WG + 2 consumer WG, setmaxnreg 24 / 240
    D0.9  smem map = 10 mbarrier + Q 32KiB + K ring 2x32KiB + V ring 2x32KiB
    D0.10 producer warp loads Q once (TMA 4D box) and streams K/V through two
          independent 2-stage TMA rings (expect-tx 0x8000 per stage)
  step 3 (already recorded: D0.2+D0.3+D0.4+D0.5+D0.6+D0.7+D0.13 — the compute
          path with straight-line waits)
    D0.2  M tile = 8 query positions x 16 Q heads, head the fast row axis;
          mask limit uses pos = s0 + (row >> 4)
    D0.3  Q resident in smem, K/V streamed, S/O/P register-resident, O staged
          through the Q alias region
    D0.4  bf16 MMA inputs / f32 accumulation, exp2 softmax with the host-folded
          log2(e)/sqrt(D) constant, row max kept in the scaled domain, row
          max/sum merged inside the quad every iteration
    D0.5  P never touches smem: the S-chain C-fragment is packed pairwise into
          the RS A-fragment of the PV chain
    D0.6  every KV tile does the full causal mask (predicate + select), tile
          count = (s0 + 7)/128 + 1
    D0.7  bM = 128 (2 x 64), bN = 128 keys, bK = 128 = D; S chain SS (Q/K
          k-major descriptors), PV chain RS (P registers, V MN-major)
    D0.13 prologue peeled, epilogue = 1/l scaling -> bf16 -> STSM into the Q
          alias -> CTA rendezvous -> producer warp TMA store
  TMA issue form: expect-tx is elected (single-thread instruction), the
  copies are not -- the DSL copy atom elects internally over the whole warp,
  so an elect_one around it deadlocks the warp.  See the producer branch.

  step 4-5 (recorded as D0.11+D0.12 — the G2 pair, D0.11 (x) D0.12)
    D0.11 the wait ladder and the segment split: the body commits the S chain
          of tile j and the PV chain of tile j-1, then wait_group(1) retires
          only the S chain, segment B (mask, row max, scaled-domain running
          max, p = exp2) runs while PV_{j-1} is still in flight, K-empty is
          released there, wait_group(0) retires PV_{j-1}, V-empty is released,
          and segment C (O rescale, row sum, l update, P_j pack) runs with no
          chain in flight.  Segment B touches only the S accumulator; segment
          C is the only thing that touches O or the RS A-fragment.  No group
          crosses the loop back edge.
    D0.12 ping-pong: two NamedBarrier ids (8 = WG0's own, 9 = WG1's own, width
          256).  Each consumer warpgroup blocks on its own id at the top of the
          body and arrives (non-blocking) on the other's after its PV chain is
          committed; WG1 pre-arrives once on id 8 at the tail of the peeled
          first tile so WG0 issues the first body's MMA.  Never an arrive on
          the id one blocks on.

  Whole-machine smoke certificate for THIS tree (notes/probe_m0_step6_ladder.py,
  all three contracts PASS):
    C1  every mount rung retires, including the shipped level at the campaign
        shape (B=2 H_Q=16 H_KV=1 S=4096 D=128, grid 1024), and that run's
        output matches an in-probe fp32 reference at
        |o-ref32|_max / |ref32|_max = 0.00311646 (gate tolerance 2e-2).
    C2  the shipped cubin's wgmma wait axis carries the D0.11 ladder (`LE 0x1`
        in the DEPBAR immediate histogram) and shows no serialisation
        fingerprint (DEPBAR/HGMMA <= 0.25, no run of >= 8 HGMMA-then-DEPBAR):
        ptxas kept the pipeline, so segment B really does run under PV_{j-1}.
    C3  the ping-pong is asymmetric: level 7 has zero BAR.ARV, level 8 has the
        body handoff plus WG1's prologue pre-arrive.

  M0 gate (step 6).  Placeholder ledger: empty -- every decision of D0.1-D0.13
  is mounted in its real form, nothing was landed conservatively, so there is
  no [PLACEHOLDER:] to clear.  The structural lines the gate reads are all
  static properties of the code above: one launch per call (the torch.empty_like
  for o is not a launch), grid 1024 / block 384 from _attn_jit, smem_dynamic
  = 1024 + 32768 + 2 * 65536 = 164864 B with no static smem, 8 HGMMA per chain
  with the S chain's A a descriptor and the PV chain's A registers, DEPBAR
  LE 0x1 / LE 0x0 in the body from the D0.11 ladder, and the only generic-proxy
  smem traffic is the epilogue STSM.

M1 — levers on the frozen topology.
  D1.1 wait-depth rungs and release points (relations unchanged, values swept)
    (a) K-empty release point: moved from "after segment B" to "immediately
        after wait_group(1)".  LE 1 retires the S chain, which is K_j's only
        reader, so the stage is dead from that instruction onward; releasing it
        there hands the producer its credit one whole segment B earlier.
    (b) forward-move of row sum / the remaining EX2 out of segment C and into
        the flight window: NOT taken this round.  Segment C is the only thing
        standing between wait_group(0) and V-empty, so pulling work across LE 0
        moves the drain later in time and delays that credit -- the opposite
        sign of (a) -- while the drain wait it would shorten is already short
        (segment B outlasts PV_{j-1}'s flight by a wide margin).
    (c) wait_group encoding: no group crosses the back edge (both rungs stay
        wait_group(1) / wait_group(0) inside the body).  Both of this body's
        wait points also return a ring stage, which is exactly the shape
        wgmma_backedge_pending_group gotcha 8 says not to carry: a carried wait
        defers its stage release, and that costs more than the sliver of
        loop-top work it covers.

  D1.2 ring depth: the 2-stage rung is kept, the 3-stage rung is not taken.
    The decision's own trigger is "consumer's share of waiting on K/V-full >
    the reference 3.5%", and the two machine readings this tree has both say
    the binding edge is elsewhere: NCU stall_long_sb 0.75 and stall_barrier
    0.40 are BELOW the reference's 0.82 / 0.44 (the two supply-side stall
    signals), and in the phase timeline the producer spends 2846 of its 3099
    cycle period blocked on empty credits -- it is already as far ahead as a
    2-deep ring lets it be, while the consumer's three largest segments are
    all busy, not wait.  A third stage buys the producer lead it cannot use
    and costs +65536 B of smem (ring_depth_lever gotcha 6 records the same
    structure paying +0.67% for it).  Not taken; KV_STAGE stays 2.

  D1.3 softmax segment instruction form (D13_TIGHT below, a trace-time A/B
    knob).  The budget this decision is judged on is non-MMA instructions per
    iteration per thread (target <= 603).  Two forms in the step-3 tree spend
    instructions the segment does not need:
    (a) the causal mask asked the question in the key domain: `key = kv_base +
        col` (one IADD3 per element, 64/iteration) and then `setp.le key,
        pos`.  Moved into the column domain: `lim = pos - kv_base - col0` once
        per row, and each element's static column offset is folded into the
        PTX as an immediate (`setp.ge.s32 p, lim, <off>`), so the per-element
        cost is one ISETP + one FSEL and nothing else.  The offsets are read
        off the (static) C-fragment partition layout at trace time and
        validated there (32 distinct values, first == 0, all < BN); if this
        stack does not hand them over as python ints the form falls back to
        hoisting `key` per column instead of per element (32 IADD3 rather than
        64), which needs no layout introspection.  MASK_KIND records which one
        was traced.
    (b) every tile opened its S chain with `acc_s.fill(0.0)` = 64 MOV per
        thread, sitting between the K-full wait and the first HGMMA, i.e. in
        the handoff path the ping-pong uses to keep the tensor pipe fed.  The
        WGMMA scale-D flag does the same job inside the instruction: the
        chain's first HGMMA is issued with ACCUMULATE False (it overwrites the
        accumulator) and the remaining seven with True.  The O accumulator is
        unchanged -- it accumulates across tiles and is still zeroed once per
        CTA.
    The rest of the decision's list was already in the M0 form and is
    unchanged here: the mask is predicate + select and never "flag -> float ->
    arithmetic" (branch_free_select_mask), the row scalars are hoisted out of
    the column loop onto the M axis (cfrag_elementwise_fastpath (b)), exp2 is
    `fastmath=True` (one MUFU.EX2 per element), and the P pack is one
    `cvt.rn.bf16x2.f32` per pair (32 F2FP, not 64 scalar converts).  The max
    reduction is already `max.f32` -> FMNMX rather than a compare/select pair,
    which is softmax_segment_ilp (c), the -1.2% rung of that recipe.
    Not taken, because the same recipe measured them null or negative on this
    structure: reshaping the serial max / sum chains into trees or 4-way
    partial sums (+3.8%, registers 227 -> 245), hand-interleaving the two
    rows (SASS instruction-for-instruction identical), and moving the
    ping-pong `arrive` earlier in the source (time ratio 1.002 -- the
    scheduler already sinks the dependent work past it).  The recipe's other
    real lever, wide-word flag reads, has no site here: this segment reads no
    smem at all (the mask limit is index arithmetic, not a flag array).

  Probe certificate for the D1.3 form (notes/probe_d13_segform.py, one child
  process per rung, the knob read once per process at trace time, all four
  contracts PASS):
    C1  tight form at the campaign shape (B=2 H_Q=16 H_KV=1 S=4096 D=128,
        grid 1024) matches an in-probe fp32 reference at |o-ref32|_max /
        |ref32|_max = 0.00311646 (tolerance 2e-2).  Both edits are
        silent-wrong-answer shaped -- offsets read in the wrong mode order
        would mask the wrong columns, and a scale-D flag that did not
        overwrite would accumulate tile j on top of tile j-1 -- and only a
        shape that exercises every diagonal position sees either.
    C2  the baseline form at the same shape matches the same reference: the
        control that puts a C1 failure on the edit rather than on the probe.
    C3  the tight cubin keeps what the edit is not allowed to move: LDL+STL
        == 0 (the 64 MOV that disappeared were also slack the allocator had),
        the same HGMMA count as the baseline, and `LE 0x1` still in the
        DEPBAR histogram (the D0.11 masking relation survived).
    C4  the tight cubin carries strictly fewer MOV+IADD3 than the baseline:
        the positive form of the edit, since a form that compiled back into
        the same code would be a null result rather than a pass.

  D1.4 wait encoding and supply bit-order (_wait_hot below).  Three rungs,
    one of them taken.
    (a) mbarrier wait encoding.  Every wait in the step-6 tree was
        `cute.arch.mbarrier_wait`, the suspending encoding: it hard-codes a
        10 ms suspendTimeHint, so an unsatisfied wait sleeps the warp and pays
        a wake-up latency when the barrier flips.  The non-blocking family
        keeps the warp resident and retries, at one issue slot plus one smem
        barrier read per retry -- so the encoding is a per-site question, and
        this tree's two sides answer it in opposite directions:
          - the two IN-LOOP consumer waits (K_j full at the top of the body,
            V_{j-1} full before the PV chain) are SHORT and ON the critical
            path.  The consumer's whole period is 2960 cycles of which 430 is
            wait spread over five points, and neither full-wait is among the
            three largest segments, so each is at most a couple of hundred
            cycles -- the regime where a wake-up latency is a visible fraction
            of the wait.  These two are now the retry loop (_wait_hot).
          - the producer's two empty-waits are the opposite regime: 2748 of
            its 2965-cycle period is spent inside them (2627 on V-empty
            alone).  A resident retry loop there would hold an issue slot for
            93% of the kernel against the consumer warps sharing its
            scheduler, and buy nothing: the producer is not on the critical
            path, it is waiting for credits the consumer has not released yet.
            They keep the suspending form.  So do the one-shot waits (Q-full,
            the peeled tile's K-full, the collapsed tile's V-full): one
            wake-up per CTA, and Q-full is a long cold wait, which is the
            other regime the anti-trigger of mbarrier_wait_encoding names.
        Arrive / release / expect-tx counts are untouched (gotcha 8), and no
        wait target moves: this changes what the warp does while it is not yet
        satisfied, nothing else.
    (b) producer K/V issue order (K_s0,V_s0,K_s1,V_s1 as shipped, vs
        K_s0,K_s1,V_s0,V_s1): NOT taken, no site.  The alternative exists to
        stop the V-empty wait from standing in front of the next tile's K
        load.  That wait does stand there -- the producer's iteration jk needs
        K-empty from consumer iteration jk-2 but V-empty from consumer
        iteration jk-1, so the K load of jk+1 is issued one segment B later
        than its credit allows.  It costs nothing measurable because the lead
        that remains is still ~5500 cycles (the producer issues K_{jk+1} 438
        cycles before the end of consumer iteration jk-1 and it is not read
        until the top of iteration jk+1, two 2960-cycle periods later) against
        a 32 KiB TMA out of an L2 that hits 93.5%.  The machine agrees: the
        consumer's K-full and V-full waits are both outside its three largest
        segments.
    (c) tensormap prefetch position: NOT taken, no site.  The four
        `prefetch_descriptor` calls are already the first thing the producer
        warp executes, ahead of the Q TMA; there is no earlier point inside
        the role branch, and moving them before the branch would only make the
        two consumer warpgroups execute them too.

M1 gate.  Submitted alone, no decision bundled: D1.1-D1.4 are all landed and
  kept on this tree and this commit changes only this ledger -- no executable
  line differs from the D1.4 tree, so the gate reads the same machine the
  D1.4 row measured.  Placeholder ledger: empty (nothing in D1.1-D1.4 was
  landed in a conservative or transitional form; the rungs that were not taken
  were declined outright, with the reading that declined them recorded above,
  and none of them left a placeholder behind).
  Whole-machine smoke certificate for this tree: notes/probe_d14_waitenc.py,
  all five contracts PASS on the shipped form at the campaign shape
  (B=2 H_Q=16 H_KV=1 S=4096 D=128, grid 1024) -- lowered, launched 12 times,
  every launch inside tolerance at |o-ref32|_max / |ref32|_max = 0.00311646
  and bitwise identical to the first.
  The gate lines, each against this tree's own measurement (the eval of the
  D1.4 commit, the tree this row ships unchanged):
    correctness PASS                                         -- PASS
    device <= 0.301 ms                                       -- 0.2492 ms
    tensor_pct >= 60                                         -- 68.21
    spill_ld_sectors == 0 and spill_st_sectors == 0          -- 0 / 0
    smem_wavefronts_ld + smem_wavefronts_st <= 393216        -- 0 + 262144
  The last line is also the topology-drift check: D1.2 declined the third ring
  stage, so smem_dynamic is still 164864 B and the generic-proxy smem traffic
  is still only the epilogue STSM, exactly as at the M0 gate.  The other
  frozen structural readings are unmoved as well (kernel_count 1 at
  dominant_time_pct 100, grid 1024, block 384, regs_per_thread 168,
  bankconf_ld + bankconf_st 0).
  DEFERRED lines from M0: none.  The M0 gate closed clean -- every one of its
  lines passed on the step-6 tree -- so no line was handed down to be
  adjudicated here and none is handed further down.
  Standing against target after M1: target_ms is 0.2493 ms device and the
  stopping rule is the same-job paired ratio >= 1.05; this tree's paired
  device ratio is 0.9765 (0.2492 student vs 0.2433 reference in the same job),
  so the M2 decisions (D2.1 (x) D1.3, D2.2, D2.3) are still owed.

M2 — the tail decisions, on the same frozen topology.
  D2.1 mask only the tiles that have a masked element (the guard at the top of
    _softmax_b below, now the single shipped form; G3 with D1.3, which already
    tightened the segment).
    Every KV tile ran the full mask in the M0 form -- one ISETP + one FSEL per
    element, ~130 of the ~603 non-MMA instructions per iteration per thread
    (D0.6 accepted that as M0's fixed tax and named this decision as the place
    to drop it).  A tile only has a masked element if its largest key is NOT
    visible to this CTA's smallest query position: the tile spans keys
    `kv_base .. kv_base+127` and the rows span positions `s0 .. s0+7`, so
    `kv_base + 127 <= s0` means every element of the tile is causally visible
    and the mask is the identity on all 64 of the thread's elements.  The whole
    block now sits behind that forward guard in `_softmax_b`, which is also
    where the peeled tile goes through (kv_base = 0).
    Per CTA that leaves exactly one or two masked tiles -- the last tile
    always, plus the one before it when the 8 positions straddle a 128 boundary
    (at this shape s0 is a multiple of 8, so `(s0+7) % 128 >= 7` and only the
    last tile ever straddles).  Over the grid: 16896 tiles of which 1024 + 0
    keep the mask, i.e. the guard skips it on ~94% of the iterations.
    Form: a real forward branch, not predication.  The guarded block is ~130
    instructions, far past the length ptxas if-converts (guard_predicate_issue
    _distance gotcha 5), and the predicate is CTA-uniform with both inputs
    already in registers, so it is not the shape gotcha 6 (divergence) or the
    §(a)-5 `d_src <= 8` reading (a long-latency input feeding the guard) warn
    about -- there is no load anywhere in the closure of this predicate.  The
    alternative form the decision allows (two copies of the loop body, an
    unmasked steady one and a masked tail one) buys the branch back at the
    price of a second copy of the whole body in the instruction stream; it is
    not taken.
    What does NOT change: the tile count is still `(s0+7)/128 + 1` (D0.6 --
    the guard removes work inside a tile, never a tile), the segment split and
    both wait targets are untouched (the guard lives strictly inside segment
    B), and no barrier, ring stage or arrive moves.

  Probe certificate for the D2.1 form (notes/probe_d21_nomask.py, one child
  process per rung, all four contracts PASS):
    C1  the guarded form at the campaign shape (B=2 H_Q=16 H_KV=1 S=4096
        D=128, grid 1024) matches an in-probe fp32 reference at
        |o-ref32|_max / |ref32|_max = 0.00311646 (tolerance 2e-2).  This edit
        is silent-wrong-answer shaped: an off-by-one in the guard skips the
        mask on a tile that does have masked elements, which folds keys from
        the future into that row's softmax without crashing, and only a shape
        that exercises every value of s0 mod 128 sees it -- hence the campaign
        shape, plus a small shape as the fast failure.  The worst row of the
        difference is reported per rung for the same reason.
    C2  the baseline form (mask on every tile) at the same shape matches the
        same reference: the control that puts a C1 failure on the edit.
    C3  the guarded cubin keeps what the edit is not allowed to move: LDL+STL
        == 0 (the conditional writes to the S accumulator did not push the
        fragment to local memory, which the M2 gate reads as a hard line), the
        same HGMMA count as the baseline, and `LE 0x1` still in the DEPBAR
        histogram.
    C4  the positive form of the edit, and the one that matters here: unlike
        D1.3 this decision removes ISSUES, not instructions -- the mask block
        is still in the binary, it is jumped over.  So the guarded cubin must
        carry strictly more BRA than the baseline, NO predicated FSEL (an
        if-converted block issues its work anyway), and an FSEL count within
        5% of the baseline's (skipped, not deleted).  A form that compiled
        back into straight-line predicated code would pass C1-C3 and be worth
        nothing.

  D2.2 dispatch order: the A/B is measured and the M0 map is what ships.  The
    decision asked for three ctaid -> work maps compared in the device window,
    and all three were built as separate kernels and timed against each other
    (the two alternative maps lived behind a trace-time knob for that run and
    are gone from this file again -- the shipped map is the only one left):
      DISPATCH_DESC  (SHIPPED) s0 decreasing, the two batches interleaved (the
                     M0 / reference map): `k = ctaid >> 1`, `s0 = 4088 - 8k`,
                     `b = ctaid & 1`.  Work per CTA is `(s0+7)/128 + 1` tiles,
                     so this issues the heavy CTAs first and the tail wave
                     (1024 = 7 x 132 + 100, i.e. 100 of 132 slots) is made of
                     the 1-tile CTAs.
      s0 increasing  the same interleave the other way round -- the heavy CTAs
                     land in the tail wave instead.  Not shipped.
      batch-major    all 512 tiles of batch 0 (s0 decreasing), then all 512 of
                     batch 1: the same multiset of CTA weights in a sawtooth
                     rather than a monotone sequence.  Not shipped.
    All three are bijections onto the same 1024 (b, s_tile) pairs, so the map
    changes which CTA does which tile and nothing else: no tile count, no smem
    byte, no barrier, no wait target moves.  D0.1 already ruled residency out
    of the topology contract (K/V working set 4.2 MB against a 50 MB L2, DRAM
    read amplification 1.00x), so the only thing the map can price is the tail
    wave, and this decision costs one measurement and no topology change by
    construction.  Keeping DESC leaves the ctaid map byte-identical to every
    tree since M0: no executable line of the shipped kernel changed for D2.2.
  D2.3 row-sum merge deferred to the epilogue -- TAKEN, and now the single
    shipped form (the D23_LATE_SUM A/B knob the probe used is removed).
    The M0 form merged this tile's row sum inside the quad every iteration
    (two SHFL.BFLY per row, xor 1 then xor 2, 4 per iteration for the thread's
    two rows) and added the merged value into the running `l`.  Now `l`
    (`stats[2r+1]`) carries the lane's own partial sum over the columns that
    lane holds, and the single quad merge happens in the epilogue, immediately
    before 1/l.  It is the same number: the rescale factor alpha is
    quad-uniform, because it is built from m2 and the row MAX keeps its
    per-iteration merge (segment B computes p from it, so it must), and
    rescaling four partial sums by one common factor and then adding them is
    rescaling their sum.  That trades 4 SHFL per iteration for 4 per CTA (16896
    tile-iterations -> 1024 CTAs over the grid) and takes the one cross-lane
    dependency out of segment C's chain that did not have to be there; the
    epilogue merge sits on the same lanes the loop would have used, since the O
    fragment holds the same two rows as the S fragment.
    What does NOT change: no barrier, arrive, wait target, ring stage or smem
    byte moves (KV_STAGE 2, 164864 B), the D0.11 segment split is untouched
    (the shuffles that left segment C were the trailing part of the `l` update,
    which touches neither the O accumulator's data flow nor the RS
    A-fragment), and the register account is unchanged -- `stats` already held
    a per-row `l` slot.

  Probe certificate for the D2.2 / D2.3 A/B (notes/probe_d22_d23.py, five arms
  compiled in ONE process and their device windows interleaved round by round
  with the arm order rotated per round, all ten contracts PASS):
    C1  base (the shipped tree) at the campaign shape (B=2 H_Q=16 H_KV=1
        S=4096 D=128, grid 1024) matches an in-probe fp32 reference at
        |o-ref32|_max / |ref32|_max = 0.00311646 (tolerance 2e-2) -- the
        control for C2-C4.
    C2  the late-sum arm matches the same reference.  This edit is
        silent-wrong-answer shaped twice over: an alpha that were not
        quad-uniform would rescale the four partials inconsistently, and an O
        fragment whose lanes did not hold the S fragment's rows would make the
        epilogue butterfly sum four unrelated partials -- both give a finite,
        plausible, wrong o.
    C3  the s0-increasing arm matches it, C4 the batch-major arm does.  Each
        arm's output buffer is NaN-poisoned before its launch, so a map that is
        not a bijection onto the 1024 (b, s_tile) pairs leaves NaN behind and
        fails the finite check instead of averaging out.
    C5  the four A/B arms are four different kernels (one cubin dumped per arm,
        four distinct sha256, every compile took real time against a floor
        calibrated an order of magnitude under a real compile and an order
        over a cache hit) and the A/A null control is byte-identical to base --
        without this, four equal timings would read as "the map does not
        matter" when they mean "one kernel ran four times".
    C6  the late-sum cubin carries strictly fewer SHFL than base's -- the
        positive form of D2.3 -- with LDL+STL == 0, base's HGMMA count, and
        `LE 0x1` still in the DEPBAR histogram: the butterflies left the loop
        and neither the M2 gate's hard lines nor the D0.11 relation moved with
        them.
    C7  the measurement is calibrated before anything is read off it: no arm is
        host-bound (worst host-enqueue share of its own device window under
        0.5), and the A/A null -- base compiled a second time and timed as a
        fifth arm through the same rotation -- is reported as a TIE by the very
        rule C8-C10 are judged by, its whole paired-median CI inside +-1.5%.
        A harness that manufactured a difference between two copies of one
        binary at the scale C8-C10 report would fail here and nothing below it
        could be read.
    C8-C10  the three comparisons RESOLVE: for each pair the distribution-free
        CI for the median of the per-round paired ratios (round statistic =
        that round's minimum window, since interference on a shared device only
        ever adds time; first window after each arm switch discarded) either
        excludes zero, establishing a direction, or lies entirely inside
        +-1.5%, bounding the effect under what is worth acting on.  C8 is the
        shipped s0-decreasing map against s0 increasing, C9 against
        batch-major, C10 the deferred row-sum merge against the per-iteration
        one.  FAIL would have meant an interval that both straddles zero and
        reaches past what these rounds can read, i.e. no direction may be read
        off at all.

  With D2.1 (the guard), D2.2 (map measured, DESC kept) and D2.3 (deferred
  merge, landed) the M2 decision list is complete, so the next row is the M2
  gate and it goes alone.  This commit carries no gate line, and the
  placeholder ledger is empty: both M2 decisions here are in their real form --
  D2.3 is the deferred merge itself and D2.2's real form is a shipped map,
  which is what the A/B chose, so neither is a conservative stand-in.

M2 gate.  Submitted alone, no decision bundled: D2.1, D2.2 and D2.3 are all
  landed and kept on this tree and this commit changes only this ledger -- no
  executable line differs from the D2.2+D2.3 tree, so the gate reads the same
  machine that row measured.  Placeholder ledger: empty (D2.1 ships the guard
  itself, D2.3 the deferred merge itself, and D2.2's real form is a shipped
  ctaid map; nothing in M0, M1 or M2 was ever landed in a conservative or
  transitional form, so there is no [PLACEHOLDER:] anywhere in the lineage to
  clear before this row).
  Whole-machine smoke certificate for this tree: notes/probe_d22_d23.py, whose
  `late_sum` arm is this tree's shipped semantics (DISPATCH = s0-decreasing
  interleave, deferred row-sum merge) -- lowered and launched at the campaign
  shape (B=2 H_Q=16 H_KV=1 S=4096 D=128, grid 1024) with the output buffer
  NaN-poisoned first, matching an in-probe fp32 reference at |o-ref32|_max /
  |ref32|_max = 0.00311646 against a 2e-2 tolerance (contract C2), alongside
  C5's evidence that the arm is its own cubin and not a cache hit serving
  another arm's code.  The D2.2+D2.3 commit that removed the probe's knobs then
  took a full eval of this exact tree, which lowered, launched and passed
  correctness at the campaign shape.
  The gate lines, each against this tree's own measurement (the eval of the
  D2.2+D2.3 commit, the tree this row ships unchanged):
    correctness PASS                                         -- PASS
    device <= 0.2493 ms                                      -- 0.2423 ms
    tensor_pct >= 63                                         -- 68.96
    spill_ld_sectors == 0 and spill_st_sectors == 0          -- 0 / 0
  The frozen structural readings are unmoved from the M0 and M1 gates, so the
  M2 levers stayed inside the topology they were allowed: kernel_count 1 at
  dominant_time_pct 100, grid 1024, block 384, smem_dynamic 164864 B with
  smem_static 0, regs_per_thread 168, smem_wavefronts_ld + smem_wavefronts_st
  = 0 + 262144, bankconf_ld + bankconf_st 0.
  Final adjudication of the deferred lines: there are none.  M0 closed clean
  and handed nothing down, M1 closed clean and handed nothing down, and every
  line of this gate passes on this tree, so no line is carried out of M2
  unadjudicated and none is declared INFEASIBLE.
  Standing against target.  target_ms is 0.2493 ms device and the stopping rule
  is the same-job paired ratio >= 1.05.  This tree's absolute device window is
  0.2423 ms, inside the gate's line, but its paired ratio is 1.0071 (0.2423
  student against 0.2441 reference in the same job), so the gate's parenthetical
  stopping clause is not met even though every checkbox above is.  The decision
  graph reaches M2 and stops there: D0.1-D0.13 are frozen topology, D1.1-D1.4
  and D2.1-D2.3 are all landed or measured-and-declined with the reading that
  declined them recorded above, and there is no further decision in the phase
  order to take.
  Advisory readings on this tree, none of them a gate line (§5.2): tensor_pct
  68.96 against the reference's 66.44, stall_long_sb 0.93 (reference 0.82),
  stall_mio 0.10 (0.07), dram_pct 5.99 with dram_bytes_read 37.79 MB (5.76 /
  37.79 MB), warp_active_pct 13.48 (13.52), and stall_barrier 1.12 against the
  reference's 0.44.  The last one is above its x1.25 attribution threshold and
  rose across the two M2 rows (0.36 through M1, 1.03 at D2.1, 1.12 at
  D2.2+D2.3), i.e. it tracks the decisions that took work OUT of segments B and
  C rather than any change of wait target: §5.2's causal branch for a high
  stall_barrier is read together with tensor_pct, and tensor_pct went UP with
  it and sits above the reference, so this is the consumer reaching its
  unchanged waits earlier, not the phase pathology or a wrong wait target.  No arrive,
  release, expect-tx count or wait target has moved since the M0 gate.

SMOKE_LEVEL below is the construction-order mount switch: a trace-time
constant that removes a suffix of the mechanism list, so each of them can be
smoked on its own instead of all eight going on the machine in one step. The
shipped value mounts everything and the guards fold away at trace time.
"""

import math

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from typing import Callable

# --- fixed campaign point (shape A) ------------------------------------------
B, H_Q, H_KV, S, D = 2, 16, 1, 4096, 128
G = H_Q // H_KV                 # 16 q heads per kv head
BM_POS = 8                      # query positions per CTA tile
BM = BM_POS * G                 # 128 rows per CTA tile (head is the fast axis)
BN = 128                        # keys per KV tile
BK = D                          # 128
KV_STAGE = 2                    # D0.10 ring depth (M1 lever D1.2)
Q_STAGE = 2                     # D0.9 Q slots (G0 part 2: slot = tile & 1)
NUM_WG = 3
NUM_THREADS = NUM_WG * 128      # 384
PRODUCER_THREADS = 128
CONSUMER_THREADS = NUM_THREADS - PRODUCER_THREADS       # 256
PRODUCER_REGS = 24
CONSUMER_REGS = 240
N_WORK = B * (S // BM_POS)      # 1024 Q tiles of work
SM_COUNT = 132                  # refreshed from device properties in build()
SM_FORCE = None                 # probe hook: pin the CTA count instead
GRID = min(SM_COUNT, N_WORK)    # D0.1 persistent: grid = SM count
SCALE2 = math.log2(math.e) / math.sqrt(D)   # D0.4 host-folded exp2 constant
PARK = -1.0e30                  # masked-out score: finite, not -inf
M_INIT = -1.0e30                # running row max (scaled domain) before tile 0
BAR_CONSUMER = 2                # NamedBarrier (0 belongs to sync_threads)
BAR_PP = 8                      # D0.12 ping-pong: 8 = WG0's own id, 9 = WG1's
PP_THREADS = CONSUMER_THREADS   # 256 = the 128 blocking + the 128 arriving

# --- construction-time mount ladder ------------------------------------------
# A hang carries no source line, so the high-risk mechanisms of the M0
# construction order are switchable one at a time: each level adds exactly one
# of them and nothing else.  SMOKE_LEVEL is read while the kernel is traced
# (cute_dsl_const_expr_if gotcha 4: one level per process), and the shipped
# value mounts the whole machine, so the default costs nothing.
SMOKE_ROLES = 1                 # role branch + setmaxnreg + smem map + init
SMOKE_Q = 2                     # + Q TMA and the Q-full wait
SMOKE_RING = 3                  # + K/V TMA rings and every full/empty handshake
SMOKE_S = 4                     # + S chain, softmax, P pack
SMOKE_PV = 5                    # + PV chain
SMOKE_EPI = 6                   # + epilogue STSM, rendezvous, TMA store
SMOKE_MASK = 7                  # + D0.11 wait ladder (segment B under PV_{j-1})
SMOKE_PHASE = 8                 # + D0.12 ping-pong between the consumer WGs
SMOKE_LEVEL = SMOKE_PHASE

# D1.3 lever: the tightened segment form (mask in the column domain, S chain
# opened with the WGMMA scale-D flag instead of 64 MOV).  Trace-time constant,
# so the two forms are one A/B knob and the shipped value folds away.
D13_TIGHT = True
MASK_KIND = "unset"             # filled in at trace time by _col_offsets

# D2.2: the ctaid -> (batch, query tile) map.  The A/B over the three maps is
# measured (notes/probe_d22_d23.py) and the s0-decreasing interleave ships, so
# the two alternative maps are gone and the map below is the only one traced.
# All three were bijections onto the same 1024 pairs -- only the ORDER in which
# the CTAs are issued differed (D0.1 put the map outside the topology contract
# for exactly this reason), so this decision left no executable line behind.

# --- D0.10 mbarrier table ----------------------------------------------------
# 12 objects: the reference's 10 plus the pair the second Q slot brings with it
# (G0 part 2).  Q and K/V are indexed the same way -- a base plus the slot /
# stage -- so the slot arithmetic is one formula for all three buffers.
MB_Q_FULL = 0                   # + slot
MB_Q_EMPTY = MB_Q_FULL + Q_STAGE
MB_K_FULL = MB_Q_EMPTY + Q_STAGE    # + stage
MB_K_EMPTY = MB_K_FULL + KV_STAGE
MB_V_FULL = MB_K_EMPTY + KV_STAGE
MB_V_EMPTY = MB_V_FULL + KV_STAGE
NUM_MBAR = MB_V_EMPTY + KV_STAGE    # 2*Q_STAGE + 4*KV_STAGE = 12
CONSUMER_ARRIVE = 8             # 2 consumer WG x 4 warps x 1 elected thread
Q_EMPTY_ARRIVE = 1              # the one consumer thread that issued O(t)'s store
TILE_BYTES = BN * BK * 2        # 0x8000 expect-tx per K/V stage


# --- D0.1 persistent work decomposition --------------------------------------
# grid = SM count, so the 1024 Q tiles are dealt out over several ROUNDS.  The
# map is the static boustrophedon of D0.1: round i hands CTA c the tile
#     t = i*GRID + (c if i even else GRID-1-c)
# and tile t is the work (b = t & 1, s0 = (S-8) - 8*(t >> 1)).  Tile weight
# ((s0+7)/128 + 1 KV tiles) falls linearly with t, so a plain stride would give
# CTA 0 ~140 KV tiles against CTA 131's ~109 and stretch the makespan to ~1.10x;
# reversing every other round makes the weight of two consecutive rounds
# constant in c, leaving only the last short round unbalanced.
#
# Everything here is fed by `bidx` and compile-time constants only -- no lane
# component anywhere -- so the whole bookkeeping chain stays warp invariant and
# lands in the uniform register file rather than the 240-register consumer
# budget (uniform_reg_budget mechanism 1).

def set_shape(s_len, sm_count=None):
    """Host-side reconfiguration of the two things the work map depends on."""
    global S, N_WORK, SM_COUNT, GRID
    S = int(s_len)
    N_WORK = B * (S // BM_POS)
    if sm_count is not None:
        SM_COUNT = int(sm_count)
    GRID = min(SM_COUNT, N_WORK)


def _n_rounds(cta):
    """How many rounds this CTA has work in.  `n_full` complete rounds plus the
    trailing partial one when this CTA's offset in it lands inside the
    remainder.  Written as a division rather than a branch so it stays one
    uniform expression: 0 <= off_last <= GRID-1 and 0 <= rem <= GRID-1 make
    `(rem + GRID-1 - off_last) // GRID` exactly the 0/1 indicator."""
    n_full = N_WORK // GRID
    rem = N_WORK - n_full * GRID
    if n_full % 2 == 0:                 # host-side constant: which end of the
        off_last = cta                  # snake the last round runs from
    else:
        off_last = (GRID - 1) - cta
    return n_full + (rem + (GRID - 1) - off_last) // GRID


def _tile_work(i, cta):
    """round i -> (s_tile, b, s0, n_kv) for this CTA (the snake above)."""
    off = cta + (i & 1) * ((GRID - 1) - 2 * cta)
    t = i * GRID + off
    s_tile = (S // BM_POS - 1) - (t >> 1)
    s0 = BM_POS * s_tile
    # D0.6: keys 0 .. s0+7 are causally visible to this tile's 8 positions.
    return s_tile, t & 1, s0, (s0 + BM_POS - 1) // BN + 1


# --- scalar primitives -------------------------------------------------------
# The masked select is the D0.6 / branch_free_select_mask form: a predicate and
# one FSEL, never "flag -> float -> arithmetic".  The PTX escape is the
# deterministic spelling; the 0/1 blend is the fallback if this stack does not
# export the inline-asm machinery (same values, more instructions).

try:
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op, T as _T

    @_dsl_user_op
    def _keep_le(i, j, a, b, *, loc=None, ip=None) -> cutlass.Float32:
        """(i <= j) ? a : b  for Int32 i/j, Float32 a/b -> SETP + FSEL."""
        return cutlass.Float32(_llvm.inline_asm(
            _T.f32(),
            [cutlass.Int32(i).ir_value(loc=loc, ip=ip),
             cutlass.Int32(j).ir_value(loc=loc, ip=ip),
             cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.le.s32 p, $1, $2;"
            "\n\tselp.f32 $0, $3, $4, p;\n}",
            "=f,r,r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @_dsl_user_op
    def _keep_ge_imm(lim, off, a, b, *, loc=None, ip=None) -> cutlass.Float32:
        """(off <= lim) ? a : b, `off` a trace-time int folded into the PTX.

        D1.3 (a): the same predicate as _keep_le asked in the column domain,
        so the element's column offset is an ISETP immediate instead of an
        IADD3 that has to run first.  One ISETP + one FSEL per element.
        """
        return cutlass.Float32(_llvm.inline_asm(
            _T.f32(),
            [cutlass.Int32(lim).ir_value(loc=loc, ip=ip),
             cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tsetp.ge.s32 p, $1, %d;"
            "\n\tselp.f32 $0, $2, $3, p;\n}" % int(off),
            "=f,r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @_dsl_user_op
    def _ptx_fmax(a, b, *, loc=None, ip=None) -> cutlass.Float32:
        return cutlass.Float32(_llvm.inline_asm(
            _T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
             cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;", "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    @_dsl_user_op
    def _mbar_try_wait_i32(addr, phase, *, loc=None, ip=None) -> cutlass.Int32:
        """One parity attempt -> 1 satisfied / 0 not (D1.4, the (e') form of
        mbarrier_wait_encoding: the stand-in used when this build does not
        export cute.arch.mbarrier_try_wait).

        The operand is the Int32 shared-window address (Pointer.toint()), not a
        generic 64-bit pointer.  has_side_effects AND the ~{memory} clobber are
        both required: without them a pure-function asm can be hoisted out of
        the retry loop, which spins forever on a stale register (a hang, not a
        wrong answer).
        """
        return cutlass.Int32(_llvm.inline_asm(
            _T.i32(),
            [cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
             cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
            "selp.b32 $0, 1, 0, p;\n}",
            "=r,r,r,~{memory}",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    SELECT_KIND = "ptx-selp"
except Exception:                                   # pragma: no cover
    _ptx_fmax = None
    _keep_ge_imm = None
    _mbar_try_wait_i32 = None

    def _keep_le(i, j, a, b):
        # blend form: v is exactly 1.0 / 0.0 so neither term cancels
        v = cutlass.Float32(((j - i) >> 31) + 1)
        return a * v + b * (cutlass.Float32(1.0) - v)

    SELECT_KIND = "blend-shift"


_FMAX = getattr(cute.arch, "fmax", None)
if _FMAX is None:
    _FMAX = getattr(cute.math, "fmax", None)
if _FMAX is None:
    _FMAX = _ptx_fmax
FMAX_KIND = getattr(_FMAX, "__name__", str(_FMAX))


def _fmax(a, b):
    """max.f32 -> one FMNMX (softmax_segment_ilp (c))."""
    return _FMAX(a, b)


def _shfl_needs_mask():
    """shuffle_sync_bfly(value, offset, mask, mask_and_clamp): whether the two
    trailing arguments carry defaults on this stack."""
    import inspect
    try:
        ps = list(inspect.signature(cute.arch.shuffle_sync_bfly).parameters.values())
    except Exception:
        return False
    need = [p for p in ps
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(need) > 2


SHFL_WITH_MASK = _shfl_needs_mask()


def _shfl_xor(v, lane_mask):
    """butterfly shuffle inside the quad (row max / row sum merge, D0.4)."""
    if SHFL_WITH_MASK:
        return cute.arch.shuffle_sync_bfly(v, lane_mask, 0xFFFFFFFF, 0x1F)
    return cute.arch.shuffle_sync_bfly(v, lane_mask)


def _exp2(x):
    return cute.math.exp2(x, fastmath=True)


# --- D1.4 (a) mbarrier wait encoding -----------------------------------------
# Two encodings of the same acquire on the same barrier and the same phase.
# `cute.arch.mbarrier_wait` is the suspending one: it hard-codes a 10 ms
# suspendTimeHint, so a wait that is not yet satisfied puts the warp to sleep
# and pays a wake-up latency when the barrier flips (SASS NANOSLEEP).  The
# non-blocking family returns a token and the retry loop is written here, so
# the warp stays resident and no wake-up latency is paid -- at the price of an
# issue slot and one smem barrier read per retry.
#
# Which one is better is a property of the wait, not of the code: it is worth
# spinning only where waits are short and on the critical path.  The site
# selection is in _wait_hot's call sites; this is just the encoding.

_ARCH_TRY_WAIT = getattr(cute.arch, "mbarrier_try_wait", None)

if _ARCH_TRY_WAIT is not None:
    WAIT_SPIN_KIND = "arch.mbarrier_try_wait"

    @cute.jit
    def _bar_wait_spin(bar, phase):
        ok = _ARCH_TRY_WAIT(bar, phase)
        while ok == 0:
            ok = _ARCH_TRY_WAIT(bar, phase)

elif _mbar_try_wait_i32 is not None:
    WAIT_SPIN_KIND = "inline-ptx mbarrier.try_wait.parity"

    @cute.jit
    def _bar_wait_spin(bar, phase):
        addr = bar.toint()
        ok = _mbar_try_wait_i32(addr, phase)
        while ok == 0:
            ok = _mbar_try_wait_i32(addr, phase)

else:                                               # pragma: no cover
    WAIT_SPIN_KIND = "unavailable"
    _bar_wait_spin = None

# The A/B is settled (probe_d14_waitenc, C1-C5): the retry loop is what the
# two in-loop waits ship.  The suspending form survives only as the fallback
# for a build that exports neither try-wait path -- there the site selection
# below degenerates to the step-6 tree.
WAIT_KIND = ("spin: " + WAIT_SPIN_KIND if _bar_wait_spin is not None
             else "block: arch.mbarrier_wait (no try-wait in this build)")


def _wait_hot(bar, phase):
    """The encoding at the body's two full-barrier waits (K_j at the top of
    the iteration, V_{j-1} before the PV chain) -- the only waits this kernel
    takes more than once per CTA, both short and both on the critical path.
    Everything else (the one-shot Q-full / peeled-K-full / collapsed-V-full
    waits and the producer's two empty-waits) keeps the suspending form."""
    if _bar_wait_spin is not None:
        _bar_wait_spin(bar, phase)
    else:                                           # pragma: no cover
        cute.arch.mbarrier_wait(bar, phase)


# --- D0.12 ping-pong handoff -------------------------------------------------
# Take the baton by blocking on one's OWN id, pass it by arriving on the other
# warpgroup's id; never an arrive on the id one blocks on.  The width is
# explicit and unchecked: a wrong number_of_threads never fires and never warns
# (warp_specialization), so it is written once here.  The id is a warpgroup
# uniform value, not a trace constant -- both consumer warpgroups run the same
# code path with the two ids swapped.

_BAR_ARRIVE = getattr(cute.arch, "barrier_arrive", None)


def _dyn_barrier_id():
    """Whether `barrier_id` takes a runtime value on this stack.  A register
    id is the compact form (one code path, the two ids swapped per warpgroup);
    if the parameter is annotated as a trace constant instead, the same thing
    is spelled as a uniform branch over two literal ids."""
    import inspect
    try:
        ann = inspect.signature(cute.arch.barrier).parameters[
            "barrier_id"].annotation
    except Exception:
        return True
    return "Constexpr" not in str(ann)


_PP_DYN_ID = _dyn_barrier_id()


def _bar_arrive(bid):
    if _BAR_ARRIVE is not None:
        _BAR_ARRIVE(barrier_id=bid, number_of_threads=PP_THREADS)
    else:
        pipeline.NamedBarrier(barrier_id=bid, num_threads=PP_THREADS).arrive()


def _bar_sync(bid):
    cute.arch.barrier(barrier_id=bid, number_of_threads=PP_THREADS)


def _pp_bar(op, bid, lit0, lit1, wg_c):
    """`bid` is the runtime id; lit0 / lit1 are the same id spelled out for
    WG0 / WG1, used only when the id has to be a trace constant."""
    if _PP_DYN_ID:
        op(bid)
    elif wg_c == 0:
        op(lit0)
    else:
        op(lit1)


def _pp_sync(bid, wg_c):
    """Block on one's own id: WG0 -> 8, WG1 -> 9."""
    _pp_bar(_bar_sync, bid, BAR_PP, BAR_PP + 1, wg_c)


def _pp_arrive(bid, wg_c):
    """Arrive (never block) on the other's id: WG0 -> 9, WG1 -> 8."""
    _pp_bar(_bar_arrive, bid, BAR_PP + 1, BAR_PP, wg_c)


# --- host-side layout construction (must run inside a @cute.jit context) ------

def _smem_atom():
    return warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.BFloat16, BK),
        cutlass.BFloat16)


def _layouts():
    """All smem layouts of the D0.9 map.

    q_staged : (row, d, slot) 128x128x2 K-major SW128 -- the WGMMA / epilogue
               view.  Two slots (G0 part 2), stage stride BM*BK, so the slot is
               reached exactly the way a KV ring stage is.
    q_box5   : (h, s, d, b, slot) -- the TMA view of the same bytes; row =
               h + 16*s so the descriptor dim order comes out (D, H, S, B) (D0.2)
    q_box4   : same minus the slot mode (TMA atom builder wants one box)
    kv_staged: (key, d, stage) 128x128x2 K-major SW128
    kv_box3  : (key, d, 1) -- one stage, the TMA box (descriptor (D, S, B))
    kv_mn    : (d, key, stage) -- same bytes, MN-major, the PV chain's B view
    """
    atom = _smem_atom()
    q_staged = cute.tile_to_shape(atom, (BM, BK, Q_STAGE), order=(0, 1, 2))
    # colex index of q_staged at (row, d, slot) is row + BM*d + slot*BM*BK,
    # and row = h + G*s.
    q_box5 = cute.composition(
        q_staged,
        cute.make_layout((G, BM_POS, BK, 1, Q_STAGE),
                         stride=(1, G, BM, 0, BM * BK)))
    q_box4 = cute.slice_(q_box5, (None, None, None, None, 0))
    kv_staged = cute.tile_to_shape(atom, (BN, BK, KV_STAGE), order=(0, 1, 2))
    kv_box3 = cute.tile_to_shape(atom, (BN, BK, 1), order=(0, 1, 2))
    # colex index of kv_staged at (key, d, stage) is key + d*BN + stage*BN*BK
    kv_mn = cute.composition(
        kv_staged,
        cute.make_layout((BK, BN, KV_STAGE), stride=(BN, 1, BN * BK)))
    return q_staged, q_box5, q_box4, kv_staged, kv_box3, kv_mn


def _make_storage(q_staged, kv_staged):
    """D0.9 smem map. 12 mbarrier (96 B) then three 1024 B-aligned buffers:
    1024 + 65536 + 65536 + 65536 = 197632 B -- the reference's 164864 plus the
    second Q slot (G0 part 2), inside the gate's [196608, 199680] window."""

    @cute.struct
    class SharedStorage:
        mbar: cute.struct.MemRange[cutlass.Int64, NUM_MBAR]
        sQ: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(q_staged)], 1024]
        sK: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(kv_staged)], 1024]
        sV: cute.struct.Align[
            cute.struct.MemRange[cutlass.BFloat16, cute.cosize(kv_staged)], 1024]

    return SharedStorage


def _mmas():
    """D0.7 atoms: S chain SS (A = Q desc, B = K desc, both K-major), PV chain
    RS (A = P registers, B = V descriptor MN-major).  Both stack two 64-row
    atoms along M so that each consumer warpgroup owns 64 of the 128 rows."""
    mma_s = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        cutlass.Float32, atom_layout_mnk=(2, 1, 1), tiler_mn=(64, BN))

    def _helper(kw):
        return sm90_utils.make_trivial_tiled_mma(
            cutlass.BFloat16, cutlass.BFloat16,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            cutlass.Float32, atom_layout_mnk=(2, 1, 1), tiler_mn=(64, BK),
            **{kw: warpgroup.OperandSource.RMEM})

    def _explicit(wrap_atom):
        op = warpgroup.MmaF16BF16Op(
            ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32,
            instruction_shape=(64, BK, 16),
            a_src=warpgroup.OperandSource.RMEM,
            a_major_mode=warpgroup.OperandMajorMode.K,
            b_major_mode=warpgroup.OperandMajorMode.MN)
        return cute.make_tiled_mma(cute.make_mma_atom(op) if wrap_atom else op,
                                   (2, 1, 1))

    last = None
    for build in (lambda: _helper("a_source"), lambda: _helper("a_src"),
                  lambda: _explicit(True), lambda: _explicit(False)):
        try:
            return mma_s, build()
        except Exception as e:                      # spelling sweep, not logic
            last = e
    raise last


def _r2s_copy(mma_pv):
    """D0.13 register -> swizzled smem store path (STSM.16.M88.4)."""
    enum_c = LayoutEnum.ROW_MAJOR                # O is (row, d) with d contiguous
    atom_c = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
        cutlass.BFloat16)
    tiled_c = cute.make_tiled_copy_C_atom(atom_c, mma_pv)
    get_op = getattr(sm90_utils, "sm90_get_smem_store_op", None)
    if get_op is None:
        get_op = getattr(sm90_utils, "get_smem_store_op", None)
    cands = []
    if get_op is not None:
        cands.append(lambda: get_op(enum_c, elem_ty_d=cutlass.BFloat16,
                                    elem_ty_acc=cutlass.Float32))
        cands.append(lambda: get_op(enum_c, cutlass.BFloat16, cutlass.Float32))
    cands.append(lambda: atom_c)
    last = None
    for make_op in cands:
        try:
            return cute.make_tiled_copy_S(make_op(), tiled_c)
        except Exception as e:                      # spelling sweep, not logic
            last = e
    raise last


def _gmem_views(q, k, v, o):
    """torch views whose dlpack strides give the descriptor dim orders of D0.2:
    Q/O -> (H, S, D, B) with d contiguous; K/V -> (S, D, B) (H_KV folded out)."""
    return (q.permute(1, 2, 3, 0), k[:, 0].permute(1, 2, 0),
            v[:, 0].permute(1, 2, 0), o.permute(1, 2, 3, 0))


def _make_tma_atoms(mQ, mK, mV, mO, q_box4, kv_box3):
    tma_q, tQ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mQ, q_box4,
        (G, BM_POS, BK, 1), num_multicast=1)
    tma_k, tK = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mK, kv_box3,
        (BN, BK, 1), num_multicast=1)
    tma_v, tV = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mV, kv_box3,
        (BN, BK, 1), num_multicast=1)
    tma_o, tO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mO, q_box4, (G, BM_POS, BK, 1))
    return (tma_q, tQ), (tma_k, tK), (tma_v, tV), (tma_o, tO)


def _seg(site, row=0):
    return None


def _slice_stage(t, s):
    """Last-mode slice of a partition taken over a staged smem tensor.

    G0 part 2 gives sQ a slot mode, so every view partitioned from it (the
    WGMMA A fragment, the epilogue STSM destination, the TMA boxes) carries
    that mode through as its LAST mode -- exactly the way sK / sV already carry
    the ring stage.  `s` is a runtime slot index, which is legal here for the
    same reason `tSrK[(None, None, kb, idx)]` is: the mode is a pure address
    stride, so the slice is address arithmetic, not a new container."""
    try:
        r = cute.rank(t)
    except Exception:                       # spelling sweep, not logic
        r = len(t.shape)
    return t[tuple([None] * (r - 1) + [s])]


def _q_tma_issue(tma_q, tQgQ, tQsQ, mbar, cta, rnd, site):
    """G0 part 2: put round `rnd`'s Q tile into slot `rnd & 1`.

    Issue form (elect_one gotcha 6 / tma_store_epilogue gotcha 6): expect-tx is
    a single-thread instruction and needs the election, the TMA copy is NOT --
    the DSL copy atom elects inside itself with the whole warp as the member
    mask, so an elect_one around it would leave the rest of the warp outside a
    block it is waiting on (official docstring: "can cause GPU deadlock").  The
    copy is therefore issued by the whole warp, one statement after the elected
    expect-tx.

    The empty-wait parity is `(rnd/Q_STAGE + 1) % 2` -- the same "producer
    starts a phase ahead" convention the KV rings use, written out of the round
    counter, so each slot's FIRST use asks for a parity the fresh barrier is
    not at and falls through without a branch, while round r >= Q_STAGE waits
    for the release of the tile that last held the slot.

    Plain trace-time helper, not a jit region: it is inlined at each of its
    call sites, which is what keeps the elected / warp-wide split above intact.
    """
    q_slot = rnd % Q_STAGE
    q_ph = (rnd // Q_STAGE + 1) % 2
    st_n, b_n, _, _ = _tile_work(rnd, cta)
    _seg(site + ".begin", rnd)
    cute.arch.mbarrier_wait(mbar[(None, MB_Q_EMPTY + q_slot)].iterator, q_ph)
    _seg(site + ".end", rnd)
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive_and_expect_tx(
            mbar[(None, MB_Q_FULL + q_slot)].iterator, TILE_BYTES)
    cute.copy(tma_q, tQgQ[(None, (0, st_n, 0, b_n))], tQsQ[(None, q_slot)],
              tma_bar_ptr=mbar[(None, MB_Q_FULL + q_slot)].iterator)


# --- register-side views and segments ----------------------------------------

def _acc_mn(acc_layout):
    """((2,2,V), MMA_M, MMA_N) -> (row, col): mode 0 = the thread's rows,
    mode 1 = its columns.  Pure layout composition, no data movement.
    (quack/layout_utils convert_layout_acc_mn.)"""
    ref = cute.make_layout(acc_layout.shape)
    shape = ((ref.shape[0][1], ref.shape[1]),
             (ref.shape[0][0], *ref.shape[0][2:], ref.shape[2]))
    stride = ((ref.stride[0][1], ref.stride[1]),
              (ref.stride[0][0], *ref.stride[0][2:], ref.stride[2]))
    return cute.composition(acc_layout, cute.make_layout(shape, stride=stride))


@cute.jit
def _pack_pairs(src_pairs, dst_pairs):
    """f32 -> bf16 pairwise conversion through two linear (2, n_pairs) views
    (D0.5: the C-fragment element order is the RS A-fragment order).

    Decorated so the preprocessor rewrites the constexpr loop; n_pairs is read
    off the view instead of being passed, so no python int crosses the jit
    boundary (an unannotated int param would bind dynamic).
    """
    n_pairs = cute.size(src_pairs, mode=[1])
    for j in cutlass.range_constexpr(n_pairs):
        vv = cute.arch.cvt_f32x2_bf16x2(src_pairs[(None, j)].load())
        dst_pairs[(None, j)].store(cute.TensorSSA(vv, (2,), cutlass.BFloat16))


def _col_offsets(c_mn, n_col):
    """D1.3 (a): the static column offset of each element of one row of the
    C-fragment view, relative to element 0.

    The partition's layout is static (only the iterator carries the thread's
    dynamic base), so these are trace-time integers -- but whether this stack
    hands them over as python ints, and in which mode order, is not something
    to assume: the result is validated here (n_col distinct offsets, the first
    one 0, all of them inside the tile) and rejected otherwise, which is also
    what catches a row/column mode mix-up (the row component is constant along
    c, so it fails the distinctness check).  None = use the fallback form.
    """
    global MASK_KIND
    if not D13_TIGHT:
        MASK_KIND = "key-domain-per-element (D13_TIGHT off)"
        return None
    if _keep_ge_imm is None:
        MASK_KIND = "hoist-key-per-col (no ptx escape)"
        return None
    try:
        lay = c_mn.layout
        base = int(lay((0, 0))[1])
        offs = [int(lay((0, c))[1]) - base for c in range(n_col)]
    except Exception:
        MASK_KIND = "hoist-key-per-col (layout not static)"
        return None
    if (len(set(offs)) != n_col or offs[0] != 0
            or min(offs) < 0 or max(offs) >= BN):
        MASK_KIND = "hoist-key-per-col (offsets rejected: %s)" % offs[:4]
        return None
    MASK_KIND = "col-domain-imm %d..%d" % (offs[0], offs[-1])
    return offs


@cute.jit
def _mask_tile(acc_mn, c_mn, kv_base, s0):
    """D0.6 causal mask over one KV tile: predicate + select, every element of
    the S accumulator, park value PARK (finite, not -inf).

    D1.3 (a) asks the predicate in the column domain instead of the key domain:
    `key <= pos` becomes `off <= pos - kv_base - col0`, the right hand side
    hoisted once per row and the left hand side a PTX immediate.  The fallback
    form keeps the key domain but computes each key once per column instead of
    once per element (the two rows of a thread hold the same columns), so it
    costs 32 IADD3 per iteration rather than 64.
    """
    n_row = cute.size(acc_mn, mode=[0])
    n_col = cute.size(acc_mn, mode=[1])
    offs = _col_offsets(c_mn, n_col)     # None unless D13_TIGHT and validated
    if cutlass.const_expr(offs is not None):
        for r0 in cutlass.range_constexpr(n_row):
            # pos = s0 + (row >> 4) (D0.2: 16 heads per query position)
            lim = (s0 + (c_mn[r0, 0][0] >> 4)) - kv_base - c_mn[r0, 0][1]
            for c0 in cutlass.range_constexpr(n_col):
                acc_mn[r0, c0] = _keep_ge_imm(lim, offs[c0],
                                              acc_mn[r0, c0], PARK)
    else:
        if cutlass.const_expr(D13_TIGHT):
            for c1 in cutlass.range_constexpr(n_col):
                key = kv_base + c_mn[0, c1][1]
                for r1 in cutlass.range_constexpr(n_row):
                    acc_mn[r1, c1] = _keep_le(
                        key, s0 + (c_mn[r1, 0][0] >> 4), acc_mn[r1, c1], PARK)
        else:
            for r2 in cutlass.range_constexpr(n_row):
                pos = s0 + (c_mn[r2, 0][0] >> 4)
                for c2 in cutlass.range_constexpr(n_col):
                    key = kv_base + c_mn[r2, c2][1]
                    acc_mn[r2, c2] = _keep_le(key, pos, acc_mn[r2, c2], PARK)


@cute.jit
def _softmax_b(acc_mn, c_mn, stats, mprev, kv_base, s0):
    """D0.11 segment B: everything that touches ONLY the S accumulator, so it
    can run while the PV chain of tile j-1 is still in flight.

    mask (predicate + select, raw domain) -> row max (FMNMX chain + quad
    butterfly) -> scaled-domain running max -> p = exp2(fma(s, SCALE2, -m2))
    written back over the scores.  The previous running max is parked in
    `mprev` for segment C, which is where the O rescale is allowed to happen.

    D2.1: the mask block runs only on the tiles that need it.  A KV tile whose
    largest key is causally visible to this CTA's SMALLEST query position --
    `kv_base + BN-1 <= s0`, the decision's condition with kv_base = 128*jk --
    has no masked element at all, so the whole block (one ISETP + one FSEL per
    element, ~130 instructions per iteration per thread) is skipped behind a
    forward guard.  The guard's predicate is CTA-uniform (kv_base is the loop's
    induction value, s0 comes from ctaid) and its inputs are both already in
    registers -- no long-latency value feeds it, which is the one thing
    guard_predicate_issue_distance says makes a guard expensive.  The peeled
    tile takes the same path with kv_base = 0.  Correctness rests on the same
    reading either way: with no mask, every column of the tile is a real key
    that every row of the tile may see, so the row max, the exp2 and the row
    sum all range over exactly the causally visible set.
    """
    n_row = cute.size(acc_mn, mode=[0])
    n_col = cute.size(acc_mn, mode=[1])
    if kv_base + (BN - 1) > s0:
        _mask_tile(acc_mn, c_mn, kv_base, s0)

    for r in cutlass.range_constexpr(n_row):
        m = acc_mn[r, 0]
        for cm in cutlass.range_constexpr(n_col - 1):
            m = _fmax(m, acc_mn[r, cm + 1])
        m = _fmax(m, _shfl_xor(m, 1))
        m = _fmax(m, _shfl_xor(m, 2))
        m2_old = stats[2 * r]
        m2 = _fmax(m * SCALE2, m2_old)
        mprev[r] = m2_old
        stats[2 * r] = m2
        for cp in cutlass.range_constexpr(n_col):
            acc_mn[r, cp] = _exp2(acc_mn[r, cp] * SCALE2 - m2)


@cute.jit
def _softmax_c(acc_mn, o_mn, stats, mprev):
    """D0.11 segment C: the part that touches the O accumulator, hence runs
    only after wait_group(0) has retired PV_{j-1}.

    O *= exp2(m2_old - m2_new) (issue-then-correct is not used: the factor is
    built from the final max) -> row sum of this tile merged inside the quad
    -> running l rescaled by the same factor.

    D2.3 (shipped): the quad merge of the row sum was the one cross-lane
    dependency in this segment that does not have to be here, and it is gone.
    alpha is built from m2, which IS merged every iteration, so it is the same
    value in all four lanes of the quad -- rescaling each lane's partial sum by
    it and merging the partials once at the end gives the same l.  So
    `stats[2r+1]` now carries the lane's partial sum over its own columns
    through the loop and the two butterflies happen once, in the epilogue (4
    SHFL per CTA instead of 4 per iteration).  The row MAX keeps its
    per-iteration merge: p is computed from it inside segment B.
    """
    n_row = cute.size(acc_mn, mode=[0])
    n_col = cute.size(acc_mn, mode=[1])
    n_d = cute.size(o_mn, mode=[1])
    for r in cutlass.range_constexpr(n_row):
        alpha = _exp2(mprev[r] - stats[2 * r])
        lt = cutlass.Float32(0.0)
        for cp in cutlass.range_constexpr(n_col):
            lt = lt + acc_mn[r, cp]
        for dd in cutlass.range_constexpr(n_d):
            o_mn[r, dd] = o_mn[r, dd] * alpha
        stats[2 * r + 1] = stats[2 * r + 1] * alpha + lt


# --- kernel ------------------------------------------------------------------

@cute.kernel
def _attn_kernel(
    tma_q: cute.CopyAtom, mQ: cute.Tensor,
    tma_k: cute.CopyAtom, mK: cute.Tensor,
    tma_v: cute.CopyAtom, mV: cute.Tensor,
    tma_o: cute.CopyAtom, mO: cute.Tensor,
    mma_s: cute.TiledMma,
    mma_pv: cute.TiledMma,
    q_staged: cute.ComposedLayout,
    q_box5: cute.ComposedLayout,
    kv_staged: cute.ComposedLayout,
    kv_mn: cute.ComposedLayout,
    SharedStorage: cutlass.Constexpr[Callable],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- D0.1 persistent: this CTA's round count ---------------------------
    # One CTA per SM, each walking several Q tiles of the static snake.  Only
    # the round count is needed here; the per-round work comes out of
    # _tile_work inside the tile loop of each role.
    cta = cute.arch.make_warp_uniform(bidx)
    n_rounds = _n_rounds(cta)

    # ---- shared prologue: everything before the single role branch ----------
    # (warp_specialization gotcha 11: a prologue after the .dec is materialised
    #  by the producer under the reduced budget.)
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    sQ_tma = storage.sQ.get_tensor(q_box5.outer, swizzle=q_box5.inner)
    sQ = storage.sQ.get_tensor(q_staged.outer, swizzle=q_staged.inner)
    sK = storage.sK.get_tensor(kv_staged.outer, swizzle=kv_staged.inner)
    sV = storage.sV.get_tensor(kv_staged.outer, swizzle=kv_staged.inner)
    sVt = storage.sV.get_tensor(kv_mn.outer, swizzle=kv_mn.inner)
    mbar = storage.mbar.get_tensor(cute.make_layout((1, NUM_MBAR)))

    if warp_idx == 0:
        with cute.arch.elect_one():
            for qs in cutlass.range_constexpr(Q_STAGE):
                cute.arch.mbarrier_init(mbar[(None, MB_Q_FULL + qs)].iterator,
                                        1)
                cute.arch.mbarrier_init(mbar[(None, MB_Q_EMPTY + qs)].iterator,
                                        Q_EMPTY_ARRIVE)
            for s in cutlass.range_constexpr(KV_STAGE):
                cute.arch.mbarrier_init(mbar[(None, MB_K_FULL + s)].iterator, 1)
                cute.arch.mbarrier_init(mbar[(None, MB_K_EMPTY + s)].iterator,
                                        CONSUMER_ARRIVE)
                cute.arch.mbarrier_init(mbar[(None, MB_V_FULL + s)].iterator, 1)
                cute.arch.mbarrier_init(mbar[(None, MB_V_EMPTY + s)].iterator,
                                        CONSUMER_ARRIVE)
    cute.arch.mbarrier_init_fence()

    # gmem tiling + TMA partitions (address arithmetic only, no traffic yet)
    gQ = cute.zipped_divide(mQ, (G, BM_POS, BK, 1))
    gK = cute.zipped_divide(mK, (BN, BK, 1))
    gV = cute.zipped_divide(mV, (BN, BK, 1))
    gO = cute.zipped_divide(mO, (G, BM_POS, BK, 1))

    tQsQ, tQgQ = cpasync.tma_partition(
        tma_q, 0, cute.make_layout(1), cute.group_modes(sQ_tma, 0, 4), gQ)
    tKsK, tKgK = cpasync.tma_partition(
        tma_k, 0, cute.make_layout(1), cute.group_modes(sK, 0, 2), gK)
    tVsV, tVgV = cpasync.tma_partition(
        tma_v, 0, cute.make_layout(1), cute.group_modes(sV, 0, 2), gV)
    tOsO, tOgO = cpasync.tma_partition(
        tma_o, 0, cute.make_layout(1), cute.group_modes(sQ_tma, 0, 4), gO)

    cute.arch.sync_threads()

    # ---- the one and only role branch (D0.8) -------------------------------
    if warp_idx < 4:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        # warps 1..3 of the producer WG have nothing to do and fall through.
        if warp_idx == 0:
            _seg("producer.entry")
            cpasync.prefetch_descriptor(tma_q)
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            cpasync.prefetch_descriptor(tma_o)

            # The CTA's FIRST Q goes out here (D0.10 rule ii); every later
            # round's Q is issued from inside the PREVIOUS round's supply loop,
            # so it is in flight for a whole tile's worth of KV work.  The slot
            # is `round & 1` and the handshake is _q_tma_issue's.
            if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_Q):
                _q_tma_issue(tma_q, tQgQ, tQsQ, mbar, cta,
                             cutlass.Int32(0),
                             "producer.prologue.q_empty")

            # ---- D0.1 persistent tile loop (producer side) -----------------
            # `kv_run` is the CTA-lifetime running count of KV stages supplied.
            # It is NOT reset at a tile boundary: the two rings are one stream
            # per CTA (D0.10 rule i), and the consumer derives its stage index
            # and parity from the same running count, so a reset on one side
            # only would be a hang, not a wrong answer.
            kv_run = cutlass.Int32(0)
            for i in cutlass.range(n_rounds):
                s_tile, b_idx, s0, n_kv = _tile_work(i, cta)

                # G0 part 2, D0.10 rule (ii): Q(t+1) goes out once this tile's
                # first min(R, n_kv) KV stages have been issued -- late enough
                # that the ring is primed and the Q TMA is not queued in front
                # of the data the consumer is blocked on, early enough that
                # Q(t+1) has the rest of tile t (n_kv - R KV iterations, ~14.5 x
                # 2658 cyc at this shape against a ~2297 cyc Q latency) to land.
                # `min` is spelled as two mutually exclusive sites rather than a
                # value computed inside a runtime branch: the in-loop site fires
                # only for n_kv >= R and the post-loop site only for n_kv < R
                # (the short tail tiles), so nothing has to escape an if-region
                # (cute_dsl_const_expr_if: names bound in a runtime region do
                # not leave it).
                q_next = i + 1

                # K / V: two independent rings, one stage = one whole
                # 128key x 128d tile.  The empty-wait parity is
                # (run/STAGE + 1) % 2, which is the pipeline-state convention
                # (producer starts a phase ahead) written out of a running
                # counter instead of a carried state object, so it survives the
                # tile boundary unchanged.
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_RING):
                    for jk in cutlass.range(n_kv):
                        m = kv_run + jk
                        idx = m % KV_STAGE
                        ph = (m // KV_STAGE + 1) % 2
                        _seg("producer.supply.k_empty.begin", m)
                        cute.arch.mbarrier_wait(
                            mbar[(None, MB_K_EMPTY + idx)].iterator, ph)
                        _seg("producer.supply.k_empty.end", m)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar[(None, MB_K_FULL + idx)].iterator,
                                TILE_BYTES)
                        cute.copy(
                            tma_k, tKgK[(None, (jk, 0, b_idx))],
                            tKsK[(None, idx)],
                            tma_bar_ptr=mbar[(None, MB_K_FULL + idx)].iterator)

                        _seg("producer.supply.v_empty.begin", m)
                        cute.arch.mbarrier_wait(
                            mbar[(None, MB_V_EMPTY + idx)].iterator, ph)
                        _seg("producer.supply.v_empty.end", m)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar[(None, MB_V_FULL + idx)].iterator,
                                TILE_BYTES)
                        cute.copy(
                            tma_v, tVgV[(None, (jk, 0, b_idx))],
                            tVsV[(None, idx)],
                            tma_bar_ptr=mbar[(None, MB_V_FULL + idx)].iterator)

                        # the Q of the NEXT round, issued here (D0.10 ii).  The
                        # slot it goes into is the one the tile before this one
                        # held, so this wait is answered by an epilogue that has
                        # already happened by the time the consumer reaches this
                        # tile -- it can stall the supply warp, it cannot close a
                        # cycle.
                        if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_Q):
                            if q_next < n_rounds:
                                if jk == KV_STAGE - 1:
                                    _q_tma_issue(tma_q, tQgQ, tQsQ, mbar, cta,
                                                 q_next,
                                                 "producer.tile.q_empty")
                # tail tiles (n_kv < R): the loop above never reached stage
                # R-1, so the same issue happens here, after the whole (1-stage)
                # supply of this tile.  Exactly one of the two sites runs.
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_Q):
                    if q_next < n_rounds:
                        if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_RING):
                            if n_kv < KV_STAGE:
                                _q_tma_issue(tma_q, tQgQ, tQsQ, mbar, cta,
                                             q_next,
                                             "producer.tile.q_empty")
                        else:
                            # mount ladder only: below SMOKE_RING there is no
                            # supply loop to hang the issue off.
                            _q_tma_issue(tma_q, tQgQ, tQsQ, mbar, cta,
                                         q_next, "producer.tile.q_empty")
                kv_run = kv_run + n_kv
    else:
        cute.arch.setmaxregister_increase(CONSUMER_REGS)
        _seg("consumer.entry")
        tidx_c = tidx - PRODUCER_THREADS
        wg_c = cute.arch.make_warp_uniform(tidx_c // 128)
        wg_slot = cute.make_layout(2, stride=128)(wg_c)
        own_id = BAR_PP + wg_c              # D0.12: block on one's own id,
        other_id = BAR_PP + 1 - wg_c        # arrive on the other's

        # ---- setup: every fragment / view allocated exactly once -----------
        wg_s = mma_s.get_slice(wg_slot)
        wg_pv = mma_pv.get_slice(wg_slot)
        thr_s = mma_s.get_slice(tidx_c)

        tSrQ = wg_s.make_fragment_A(wg_s.partition_A(sQ))
        tSrK = wg_s.make_fragment_B(wg_s.partition_B(sK))
        tOrP = wg_pv.make_fragment_A(wg_pv.partition_shape_A((BM, BN)))
        tOrV = wg_pv.make_fragment_B(wg_pv.partition_B(sVt))
        acc_s = cute.make_fragment(wg_s.partition_shape_C((BM, BN)),
                                   cutlass.Float32)
        acc_o = cute.make_fragment(wg_pv.partition_shape_C((BM, BK)),
                                   cutlass.Float32)
        mma_s.set(warpgroup.Field.ACCUMULATE, True)
        mma_pv.set(warpgroup.Field.ACCUMULATE, True)
        n_kb_s = cute.size(tSrQ.shape[2])
        n_kb_pv = cute.size(tOrP.shape[2])
        n_pairs = cute.size(tOrP) // 2

        p_src = cute.make_tensor(acc_s.iterator, cute.make_layout((2, n_pairs)))
        p_dst = cute.make_tensor(tOrP.iterator, cute.make_layout((2, n_pairs)))
        acc_s_mn = cute.make_tensor(acc_s.iterator, _acc_mn(acc_s.layout))
        acc_o_mn = cute.make_tensor(acc_o.iterator, _acc_mn(acc_o.layout))
        cS = thr_s.partition_C(cute.make_identity_tensor((BM, BN)))
        cS_mn = cute.make_tensor(cS.iterator, _acc_mn(cS.layout))
        stats = cute.make_fragment(4, cutlass.Float32)   # (m2, l) x 2 rows
        mprev = cute.make_fragment(2, cutlass.Float32)   # m2_old, B -> C
        n_pairs_o = cute.size(acc_o) // 2
        rO = cute.make_fragment(cute.size(acc_o), cutlass.BFloat16)
        o_src = cute.make_tensor(acc_o.iterator,
                                 cute.make_layout((2, n_pairs_o)))
        o_dst = cute.make_tensor(rO.iterator, cute.make_layout((2, n_pairs_o)))
        tiled_r2s = _r2s_copy(mma_pv)
        thr_r2s = tiled_r2s.get_slice(tidx_c)
        # partitioned over the STAGED sQ, so this view carries the Q slot as its
        # last mode and the epilogue picks the slot this tile's Q occupied --
        # O is staged in the alias of its OWN tile's Q (D0.9), not of whichever
        # tile the producer is currently loading.
        tRS_sO = thr_r2s.partition_D(sQ)
        rO_src = cute.make_tensor(rO.iterator,
                                  cute.make_layout(_slice_stage(tRS_sO, 0).shape))

        # ---- D0.1 persistent tile loop (consumer side) ---------------------
        # Everything above is allocated once for the life of the CTA and
        # overwritten per tile (fixed_container_tile_loop: the 0-spill form is
        # "one container, rewritten", not "released and rebuilt").  `kv_run` is
        # the same CTA-lifetime running KV count the producer keeps; the stage
        # index and the parity are derived from it on both sides, so the rings
        # are one stream per CTA and nothing is reset at a tile boundary.
        kv_run = cutlass.Int32(0)
        for i in cutlass.range(n_rounds):
            s_tile, b_idx, s0, n_kv = _tile_work(i, cta)
            # G0 part 2: this tile's Q slot, and therefore also the alias its O
            # is staged in and the Q-empty it releases at the end.
            q_slot = i % Q_STAGE
            q_phase = (i // Q_STAGE) % 2

            # per-tile state: O accumulator and the row statistics only.
            for r0 in cutlass.range_constexpr(2):
                stats[2 * r0] = cutlass.Float32(M_INIT)
                stats[2 * r0 + 1] = cutlass.Float32(0.0)
            acc_o.fill(0.0)

            if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_Q):
                _seg("consumer.tile.q_full.begin", i)
                cute.arch.mbarrier_wait(
                    mbar[(None, MB_Q_FULL + q_slot)].iterator, q_phase)
                _seg("consumer.tile.q_full.end", i)

            if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_RING):
                # ---- D0.13 per-tile prologue: peel KV tile 0 ---------------
                k0_idx = kv_run % KV_STAGE
                k0_phase = (kv_run // KV_STAGE) % 2
                _seg("consumer.prologue.k_full.begin", kv_run)
                cute.arch.mbarrier_wait(
                    mbar[(None, MB_K_FULL + k0_idx)].iterator, k0_phase)
                _seg("consumer.prologue.k_full.end", kv_run)
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_S):
                    # D1.3 (b): the chain's first HGMMA overwrites the
                    # accumulator (scale-D 0) instead of 64 MOV clearing it.
                    if cutlass.const_expr(D13_TIGHT):
                        mma_s.set(warpgroup.Field.ACCUMULATE, False)
                    else:
                        acc_s.fill(0.0)
                    warpgroup.fence()
                    for kb0 in cutlass.range_constexpr(n_kb_s):
                        cute.gemm(mma_s, acc_s, tSrQ[(None, None, kb0, q_slot)],
                                  tSrK[(None, None, kb0, k0_idx)], acc_s)
                        if cutlass.const_expr(D13_TIGHT):
                            mma_s.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()
                    _seg("consumer.prologue.drain.begin", kv_run)
                    warpgroup.wait_group(0)
                    _seg("consumer.prologue.drain.end", kv_run)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        mbar[(None, MB_K_EMPTY + k0_idx)].iterator)
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_S):
                    _softmax_b(acc_s_mn, cS_mn, stats, mprev,
                               cutlass.Int32(0), s0)
                    _softmax_c(acc_s_mn, acc_o_mn, stats, mprev)
                    _pack_pairs(p_src, p_dst)

                # D0.12: WG1 pre-arrives ONCE PER CTA so WG0 takes the first
                # baton.  Under persistence the arrive/sync counts run on
                # across the tile boundary (the last body arrive of tile t
                # pairs with the first body sync of tile t+1), so repeating the
                # pre-arrive per tile would be a count error, not a no-op.
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_PHASE):
                    if wg_c == 1:
                        if i == 0:
                            _seg("consumer.prologue.pp_prearrive")
                            _pp_arrive(BAR_PP, wg_c)

                # ---- steady state: S_j issued, then PV of tile j-1 (D0.11) -
                for it in cutlass.range(n_kv - 1):
                    jk = it + 1
                    mk = kv_run + jk            # running KV index of K_j
                    mv = kv_run + it            # ... and of V_{j-1}
                    k_idx = mk % KV_STAGE
                    k_phase = (mk // KV_STAGE) % 2
                    v_idx = mv % KV_STAGE
                    v_phase = (mv // KV_STAGE) % 2

                    if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_PHASE):
                        _seg("consumer.body.phase.begin", mk)
                        _pp_sync(own_id, wg_c)
                        _seg("consumer.body.phase.end", mk)

                    _seg("consumer.body.k_full.begin", mk)
                    # D1.4 (a): the body's two full-waits are the only waits
                    # this kernel takes more than once per tile, and both are
                    # short (the ring is ahead) and on the critical path -- so
                    # they carry the retry loop, not the suspend.
                    _wait_hot(mbar[(None, MB_K_FULL + k_idx)].iterator,
                              k_phase)
                    _seg("consumer.body.k_full.end", mk)
                    if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_S):
                        # D1.3 (b): scale-D 0 on the first HGMMA of the chain.
                        if cutlass.const_expr(D13_TIGHT):
                            mma_s.set(warpgroup.Field.ACCUMULATE, False)
                        else:
                            acc_s.fill(0.0)
                        warpgroup.fence()
                        for kb in cutlass.range_constexpr(n_kb_s):
                            cute.gemm(mma_s, acc_s,
                                      tSrQ[(None, None, kb, q_slot)],
                                      tSrK[(None, None, kb, k_idx)], acc_s)
                            if cutlass.const_expr(D13_TIGHT):
                                mma_s.set(warpgroup.Field.ACCUMULATE, True)
                        warpgroup.commit_group()
                        _seg("consumer.body.s_commit", mk)

                    _seg("consumer.body.v_full.begin", mk)
                    _wait_hot(mbar[(None, MB_V_FULL + v_idx)].iterator,
                              v_phase)
                    _seg("consumer.body.v_full.end", mk)
                    if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_PV):
                        warpgroup.fence()
                        for kv in cutlass.range_constexpr(n_kb_pv):
                            cute.gemm(mma_pv, acc_o, tOrP[(None, None, kv)],
                                      tOrV[(None, None, kv, v_idx)], acc_o)
                        warpgroup.commit_group()
                        _seg("consumer.body.pv_commit", mk)

                    # D0.12: the baton goes to the other warpgroup once this
                    # one's two chains are committed -- arrive-only, nothing is
                    # waited on here, so its MMA segment starts under ours.
                    if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_PHASE):
                        _pp_arrive(other_id, wg_c)

                    if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_MASK):
                        # D0.11 ladder.  wait_group(1) retires the S chain
                        # only; segment B runs inside the PV chain's flight
                        # window and touches nothing but the S accumulator.
                        _seg("consumer.body.le1.begin", mk)
                        warpgroup.wait_group(1)
                        _seg("consumer.body.le1.end", mk)
                        # D1.1 (a): K_j's only reader is the S chain and LE 1
                        # just retired it, so the stage goes back HERE, at the
                        # top of the flight window, not after segment B.
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                mbar[(None, MB_K_EMPTY + k_idx)].iterator)
                        _softmax_b(acc_s_mn, cS_mn, stats, mprev,
                                   cutlass.Int32(BN) * jk, s0)
                        _seg("consumer.body.drain.begin", mk)
                        warpgroup.wait_group(0)
                        _seg("consumer.body.drain.end", mk)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                mbar[(None, MB_V_EMPTY + v_idx)].iterator)
                        _softmax_c(acc_s_mn, acc_o_mn, stats, mprev)
                        _pack_pairs(p_src, p_dst)
                    else:
                        if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_S):
                            _seg("consumer.body.drain.begin", mk)
                            warpgroup.wait_group(0)
                            _seg("consumer.body.drain.end", mk)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive(
                                mbar[(None, MB_K_EMPTY + k_idx)].iterator)
                            cute.arch.mbarrier_arrive(
                                mbar[(None, MB_V_EMPTY + v_idx)].iterator)
                        if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_S):
                            _softmax_b(acc_s_mn, cS_mn, stats, mprev,
                                       cutlass.Int32(BN) * jk, s0)
                            _softmax_c(acc_s_mn, acc_o_mn, stats, mprev)
                            _pack_pairs(p_src, p_dst)

                # ---- D0.13 collapse: this tile's last PV chain -------------
                ml = kv_run + (n_kv - 1)
                l_idx = ml % KV_STAGE
                l_phase = (ml // KV_STAGE) % 2
                _seg("consumer.epilogue.v_full.begin", ml)
                cute.arch.mbarrier_wait(
                    mbar[(None, MB_V_FULL + l_idx)].iterator, l_phase)
                _seg("consumer.epilogue.v_full.end", ml)
                if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_PV):
                    warpgroup.fence()
                    for kb9 in cutlass.range_constexpr(n_kb_pv):
                        cute.gemm(mma_pv, acc_o, tOrP[(None, None, kb9)],
                                  tOrV[(None, None, kb9, l_idx)], acc_o)
                    warpgroup.commit_group()
                    _seg("consumer.epilogue.drain.begin", ml)
                    warpgroup.wait_group(0)
                    _seg("consumer.epilogue.drain.end", ml)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        mbar[(None, MB_V_EMPTY + l_idx)].iterator)

            # the running KV count advances by this tile's KV tile count, at
            # the top level of the tile-loop body so it is unambiguously the
            # loop's carried value.
            kv_run = kv_run + n_kv

            # ---- D0.13 epilogue: O / l -> bf16 -> STSM -> TMA store --------
            if cutlass.const_expr(SMOKE_LEVEL >= SMOKE_EPI):
                n_row_o = cute.size(acc_o_mn, mode=[0])
                n_col_o = cute.size(acc_o_mn, mode=[1])
                for r1 in cutlass.range_constexpr(n_row_o):
                    # D2.3: stats[2r+1] is this lane's partial row sum -- the
                    # quad merge the loop no longer does happens here, once, on
                    # the same lanes the loop would have used (the O fragment
                    # holds the same two rows as the S fragment did).
                    l_row = stats[2 * r1 + 1]
                    l_row = l_row + _shfl_xor(l_row, 1)
                    l_row = l_row + _shfl_xor(l_row, 2)
                    inv = cutlass.Float32(1.0) / l_row
                    for d1 in cutlass.range_constexpr(n_col_o):
                        acc_o_mn[r1, d1] = acc_o_mn[r1, d1] * inv

                _pack_pairs(o_src, o_dst)

                # This tile's Q dies only when both consumer warpgroups have
                # retired their last S chain -- the rendezvous that lets O
                # alias the Q region.
                _seg("consumer.epilogue.cons_sync.begin", i)
                cute.arch.barrier(barrier_id=BAR_CONSUMER,
                                  number_of_threads=CONSUMER_THREADS)
                _seg("consumer.epilogue.cons_sync.end", i)
                cute.copy(tiled_r2s, rO_src, _slice_stage(tRS_sO, q_slot))
                cute.arch.fence_view_async_shared()
                # Second 256-wide rendezvous: every STSM of both warpgroups is
                # visible to the async proxy before the store is issued.  The
                # producer is no longer a party to this (D0.13 under
                # persistence: the store is issued on the consumer side), so
                # the old 288-wide meeting is gone.
                _seg("consumer.epilogue.vis_sync.begin", i)
                cute.arch.barrier(barrier_id=BAR_CONSUMER,
                                  number_of_threads=CONSUMER_THREADS)
                _seg("consumer.epilogue.vis_sync.end", i)

                # The store is issued by ONE consumer warp (warp-uniform
                # predicate -- the copy atom elects inside itself over the whole
                # warp, so the branch has to be warp aligned) and, in part 2,
                # still waited on the spot; the slot is handed back to the
                # producer only once that wait has retired.  With two slots the
                # producer's next use of THIS slot is two tiles out, so the wait
                # no longer gates the immediately following Q -- moving it off
                # the epilogue's critical path altogether is `part 3`.
                if warp_idx == 4:
                    cute.copy(tma_o, tOsO[(None, q_slot)],
                              tOgO[(None, (0, s_tile, 0, b_idx))])
                    cute.arch.cp_async_bulk_commit_group()
                    _seg("consumer.epilogue.store_wait.begin", i)
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    _seg("consumer.epilogue.store_wait.end", i)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            mbar[(None, MB_Q_EMPTY + q_slot)].iterator)


@cute.jit
def _attn_jit(mq: cute.Tensor, mk: cute.Tensor, mv: cute.Tensor,
              mo: cute.Tensor, stream):
    q_staged, q_box5, q_box4, kv_staged, kv_box3, kv_mn = _layouts()
    (tma_q, tQ), (tma_k, tK), (tma_v, tV), (tma_o, tO) = _make_tma_atoms(
        mq, mk, mv, mo, q_box4, kv_box3)
    mma_s, mma_pv = _mmas()
    SharedStorage = _make_storage(q_staged, kv_staged)
    _attn_kernel(tma_q, tQ, tma_k, tK, tma_v, tV, tma_o, tO,
                 mma_s, mma_pv,
                 q_staged, q_box5, kv_staged, kv_mn, SharedStorage).launch(
        grid=[GRID, 1, 1], block=[NUM_THREADS, 1, 1],
        min_blocks_per_mp=1, stream=stream)


# --- python entry ------------------------------------------------------------

_COMPILED = {}


def _current_stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        import cuda.bindings.driver as _drv
        return _drv.CUstream(raw)
    except Exception:
        from cuda import cuda as _drv
        return _drv.CUstream(raw)


def _dl(t):
    return from_dlpack(t, assumed_align=16)


def build(q, k, v, o, options=None):
    """Compile the shipped entry for these buffers. Exposed so a probe can
    drive the same constructors the shipped path uses."""
    # D0.1: the CTA count IS the SM count, read from device properties on the
    # host (SM_FORCE is the probe hook that pins it instead).
    n_sm = (SM_FORCE if SM_FORCE is not None
            else torch.cuda.get_device_properties(q.device).multi_processor_count)
    set_shape(int(q.shape[2]), n_sm)
    qv, kv_, vv, ov = _gmem_views(q, k, v, o)
    args = (_dl(qv), _dl(kv_), _dl(vv), _dl(ov), _current_stream())
    if options is None:
        return cute.compile(_attn_jit, *args), args
    return cute.compile(_attn_jit, *args, options=options), args


def attn_fwd(q, k, v):
    o = torch.empty_like(q)
    key = (tuple(q.shape), tuple(k.shape), q.dtype, q.device.index)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled, _ = build(q, k, v, o)
        _COMPILED[key] = compiled
    qv, kv_, vv, ov = _gmem_views(q, k, v, o)
    compiled(_dl(qv), _dl(kv_), _dl(vv), _dl(ov), _current_stream())
    return o
