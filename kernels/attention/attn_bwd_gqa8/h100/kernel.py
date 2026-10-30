"""flash_attn_bwd kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py, do not change):

  flash_attn_bwd(q, k, v, o, do, lse) -> (dq, dk, dv)

    q   : torch.Tensor (B, H_Q,  N, D) cuda float16
    k   : torch.Tensor (B, H_KV, N, D) cuda float16
    v   : torch.Tensor (B, H_KV, N, D) cuda float16
    o   : torch.Tensor (B, H_Q,  N, D) cuda float16   forward output (given)
    do  : torch.Tensor (B, H_Q,  N, D) cuda float16   grad wrt o
    lse : torch.Tensor (B, H_Q,  N)    cuda float32   forward log-sum-exp (given)

    dq  : torch.Tensor (B, H_Q,  N, D) cuda float16
    dk  : torch.Tensor (B, H_KV, N, D) cuda float16
    dv  : torch.Tensor (B, H_KV, N, D) cuda float16

  Causal masking, softmax scale = 1/sqrt(D), GQA mapping h_kv = h_q // (H_Q // H_KV),
  fp32 accumulation inside. dk/dv accumulate over the q-heads sharing a kv-head.

The implementation is composed CuTe DSL (cutlass.cute) code.  JIT-compiled once
and cached across calls (the bench times repeated calls).

--------------------------------------------------------------------------
M0 ledger (delivered: D0.1 D0.2 D0.3 D0.4 D0.5 D0.6 D0.7 D0.8 D0.9 D0.10
           D0.11 D0.12 -- the whole M0 decision set.  The M0 Gate closing row
           PASSED, so the kernel signature and the whole resource-topology
           contract are FROZEN; the gate certificate is transcribed further
           down this docstring.)
M1 ledger (delivered: D1.1 D1.2 D1.3 -- D1.1 + D1.2 as one co-committed
           pair, then D1.3, the scale-fold position, dialled from a to b.
           The M1 Gate closing row was submitted alone with no executable
           change: gate lines (1) spill and (3) end-to-end PASS, line (2)
           tensor_pct read 49.60% against a >= 50% bar and was carried
           forward to M2, where it clears at 50.29.)
M2 ledger (delivered: D2.1 D2.2 D2.3 -- the wait-depth dial (segment F1,
           1 -> 2, the milestone's only executable change), then the Q/dO
           stage count and the exchange-surface swizzle mode co-committed on
           one budget, both closing at their starting points.  The M2
           decision list is complete and the M2 Gate closing row is this
           commit, submitted alone with no executable change: gate line (3)
           end-to-end PASSES at 2.4124 ms against 2.55 ms and the advisory
           line is clean, while the two smem-traffic mechanism lines read
           over on their READ half and are carried into M3 with their
           attribution.  Transcribed at the END of this docstring, after the
           M1 gate section.)
M3 ledger (delivered: D3.1 -- PRE's access width dialled to the 128-bit
           ceiling on both of its streams (8 x fp16 per read instruction,
           4 x f32 per zero-init store), rows per thread closed at 1 with
           grid and block unchanged.  D3.2 -- POST's write side re-routed
           through a padded smem tile so the fp16 store is 16 B per lane and
           lane-contiguous instead of 2 B scattered over 8 rows, and dK/dV
           merged into ONE launch (three POST launches become two).  D3.3 --
           the workspace's dk/dv half backed by ONE contiguous buffer so the
           per-call reset is one fill launch instead of two, and the four
           workspace dlpack descriptors plus the CUstream wrapper hoisted out
           of the per-call path.  D3.4 and the M3 Gate are outstanding.  Transcribed at the END of
           this docstring, after the M2 gate section.)
--------------------------------------------------------------------------
Construction step 2 of M0 adds D0.6 on top of step 1.

Step 1 ("data-flow through"): PRE -> MAIN -> POST is built
end to end with the simplest per-kernel form, so the whole production/consumption
chain is real (real reads, real writes) before the high-risk runtime mechanisms
are mounted one at a time.

Certified by notes/probe_m0_dataflow.py (GPU, B=1 H_Q=2 H_KV=1 N=256 D=64):
the composed backward matches an analytic fp32 reference to rel max_diff
4.43e-4, and a second back-to-back call reproduces call 1 bit-for-bit (at the
time that second call also caught a missing wq_counter re-zero; the counter
itself is gone as of D0.2's re-delivery below, so what it now catches is a
missing dk/dv workspace re-zero).  Small shape only -- the real shape is the
gate row's job, not this construction sub-step's.

One high-risk runtime mechanism is mounted here and only one:
cp.reduce.async.bulk (S2G ADD f32), used by both D0.11 and D0.12.  TMA fills,
the mbarrier pipeline and setmaxnreg stay off until D0.5/D0.9/D0.4, each with
its own smoke, per the "mount high-risk mechanisms one at a time" rule.

D0.1  three-stage split PRE(1) -> MAIN(1) -> POST(3: dQ/dK/dV), intermediates in
      global f32 workspace.  Cross-kernel ledger as delivered:
        delta   (B,H_Q,N)   f32 : produced by PRE (full write, no zeroing needed),
                                  consumed by MAIN.  As of D1.3 dial b it holds
                                  `scale * rowsum(dO . O)`, not the bare row sum:
                                  the softmax scale rides in on this row scalar
                                  and on sV, not on a per-element FMUL.
        dq_accum(B,H_Q,N,D) f32 : host `empty`, ZEROED BY PRE, produced by MAIN
                                  (per-iteration bulk reduce-add), consumed by POST(dQ).
        dk_accum/dv_accum   f32 : host `zeros` (two fill kernels before MAIN),
                                  produced by MAIN's per-CTA epilogue,
                                  consumed by POST(dK)/POST(dV).
      There is no work-queue counter: D0.2's grid IS the work assignment, so the
      ledger has four intermediates, not five.
      Ordering between the five kernels is the stream launch order.
D0.2  work item = (b, h_q, n_blk), KV-stationary, causal walk over the q blocks
      m = n_blk .. N/bM-1.  The work item IS the grid coordinate:
      grid = (N/bN, H_Q, B) = (32, 16, 10) = 5120 CTAs, block = 384, one work item
      per CTA, and x -- the KV block -- is the fastest varying axis.  `ctaid.x * BN`
      is the KV start row and `ctaid.x` is also the causal walk's first q block;
      the map is linear, with no swizzle, no lookup and no load balancing (the
      triangular work distribution is averaged out by 38.79 waves, not by a
      scheduler).  No persistent scheduler, no work queue, no counter and no
      global atomic: the axis order alone is what keeps the concurrent footprint
      (~132/32 = 4.1 `(b, h_q)` pairs of Q + dO + dq_accum) inside L2, on both the
      read side and the dq_accum reduce-add write side.
D0.3  bM = bN = 128, bK = D = 64 (no K loop).  WGMMA atom M-mode 64, two consumer
      warpgroups stack along M (atom_layout_mnk=(2,1,1)).  Chain K depths as built:
      chain0/chain1 K=64 (=D), chain2/chain3/chain4 K=128 (=tile edge).
D0.6  per-consumer operand-source ruling for the two MMA-produced intermediates.
      chain2 (dV += P^T.dO) and chain4 (dK += dS^T.Q) take their A operand from
      REGISTERS: chain0's accumulator is S^T (M axis = KV rows) and chain1's is
      dP^T (M axis = KV rows), so P^T / dS^T are the orientation the consumer
      already wants -- `[SOURCE: RS]` on both rows.  They are fed by one RS tiled
      MMA (`mma_rs`, A operand source RMEM, A k-major, B MN-major, tiler (64,HD),
      atom_layout (2,1,1)); the A fragments `rP` / `rDS` are shaped by
      `partition_shape_A((BN, BM))` and filled from the C fragment by flat index
      (that C-fragment -> A-fragment bijection is what notes/probe_d06_rs_operand.py
      certified, exact-integer inputs, max_diff=0).  Each warpgroup owns 64 KV
      rows x the full 128-wide q (=K) extent, so neither RS chain needs data from
      the partner warpgroup.
      chain3 (dQ = dS.K) is the one `[SOURCE: smem]` row and it is FORCED: the
      register-resident orientation is dS^T (M = KV rows) while chain3 wants
      dS (M = q rows), and its K extent spans both warpgroups' 64 KV rows.  It
      keeps reading sDS, which the dS segment writes through the transposed view
      sDSt, followed by the cross-warpgroup barrier.
      Buffer endorsement: sDS survives on that third row; sP had no endorsement
      and is DELETED (32 KiB of smem and a full P round trip removed).  P is now
      read back out of `rP` when the dS segment needs it, not out of smem.
D0.7  fp32 accumulators, fp16 operands, fp32 workspace; dS is converted to fp16
      before feeding the downstream MMA.  softmax uses exp2 with scale*log2e folded
      into a host-side constexpr; lse stays in the natural-log domain given by the
      harness and is multiplied by log2e per row inside MAIN.  scale is carried by
      dS, so dQ = dS'K and dK += dS'^T Q are both correct and neither epilogue
      rescales; dV = P^T dO uses the unscaled P.  WHERE the scale is folded was
      left to D1.3, which has since dialled it from a (one FMUL per dS element)
      to b (sV and delta pre-scaled) -- the frozen part, "the epilogues do not
      rescale", is untouched by that move.
      delta is never recomputed in MAIN.
D0.11 dQ leaves per iteration: each consumer warpgroup stages its 64x64 f32 dQ tile
      into smem, and one thread issues cp.reduce.async.bulk (S2G ADD f32) into
      dq_accum.  No per-thread atomicAdd / red.global.add anywhere.
D0.12 dK/dV accumulate in registers across the whole work item (accumulate-semantics
      MMA chains) and are flushed once per work item through the same smem staging
      region (alias reuse) with the same bulk reduce-add primitive.

D0.5  part 1 of 4 -- ONE of the buffer's two access atoms, and only that one:
      the sdS WRITE side.  The dS segment no longer stores `sDSt[(kv, qr)]` per
      element (64 scalar STS per thread); it issues one transposed-`stmatrix`
      tiled copy (`r2s_ds`, 8x STSM.16.MT88.4 per thread) whose register source
      is rDS -- the fp16 A fragment D0.6 already fills -- reinterpreted with the
      shape of `partition_D(sDSt)`.  Certified by notes/probe_d05_smem_atoms.py
      on GPU: rung `reinterp2wg`, C2 (face dumped back through the chain3 read
      view) and C3 (the same bytes consumed as a K-major WGMMA A operand) both
      max_diff=0, and the `[diag] today's K-major face` line re-runs the winning
      rung against the face THIS file allocates -- also 0.  So the buffer's
      layout, its size and the chain3 read side are all untouched here; the
      swap is write-side only.  `.retile` is NOT usable for this source (it
      disagrees in rank with the C-fragment nesting: "same size in mode-1, got
      1 and 8", the trace-time error that killed both `direct` rungs).

D0.5  part 2 of 4 -- the dQ/dK/dV STAGING write side (this commit).  The three
      flush sites no longer store `sStage[(r, c)]` per element into a
      row-major (BM, HD) f32 tile; all three now write the flat-TV face
      through ONE shared partition `tSt_sD`, and the IN-LOOP dQ site issues it
      as a single 128-bit tiled copy `cute.autovec_copy(dq_st, tSt_sD)` =
      8x STS.128 per thread per warpgroup slot.  The two epilogue sites (dV,
      dK) write the SAME partition in the SAME byte order with a scalar loop.

      Why the two spellings differ -- the D0.5 part-2 HANG bisect.  The
      revision of this part that vectorised ALL THREE sites hung on the device
      (event incomplete after 20 s at (1, 2, 1, 256), nvidia-smi 100%, no
      fault, no launch failure).  notes/probe_d05_staging_atom.py bracketed it
      by source-patch ablation, one variant per child process:
        ship       (unpatched)                                   device-HANG
        ship1cta   (same source, grid=1)                         device-HANG
        neither    (copies AND bulk reduce-adds removed)         complete
        noflush    (only the three autovec copies removed)       complete
        nobulk     (only the three bulk reduce-adds removed)     complete
        nodq       (in-loop dQ flush block removed)              device-HANG
        noepi      (dV/dK epilogue flushes removed)              complete
        scalarcopy (all three copies as scalar loops)            complete
      Reading: one CTA is enough to spin, so it is intra-CTA; the same byte
      order spelled scalar does not spin, so it is the VECTORISED spelling,
      not the face; and `nodq` hangs while `noepi` completes, so the carrier
      is the vectorised copy at the EPILOGUE sites specifically -- the in-loop
      dQ site's autovec + bulk pair completes on its own (`noepi`).  This
      commit therefore ships exactly the two device-certified rungs: the dQ
      site vectorised (certified by `noepi`) and the epilogue scalar
      (certified by `scalarcopy`).  Vectorising the epilogue is deferred to a
      later single-mechanism swap-back, priced on its own round.  It is the
      cheap half anyway: the dQ flush runs once per trip-loop iteration
      (n_mblk times per work item), the epilogue flushes once per work item.

      The staging face becomes a
      flat `(SLOT, NUM_WG)` layout (no swizzle) driven by `tv_st` =
      `make_tiled_copy_tv(CopyUniversalOp f32 @128 bits, thread layout
      (128, NUM_WG), value layout 4 f32)`, so a warp's 32 stores land in 32
      distinct banks.  Why the flat face and not the WGMMA-native
      `partition_C`: SM90's `stmatrix` has no 32-bit form, so an fp32
      C-fragment written through a swizzled 2-D staging tile degrades to
      scalar stores -- that is the documented origin of this file's
      `bankconf_st` reading, and the flat TV is the only conflict-free route.
      The non-tensor bulk reduce-add reproduces smem byte order in gmem, so
      the f32 workspace is now in TV ORDER, not logical (m, d) order.  Two
      consequences, both load-bearing:
        - the three reduce-add destinations are addressed by TILE id
          (`(b*hq + h)*n_mblk + m` etc. x STAGE_BYTES), not by (row, d);
        - `_post_kernel` inverts through the SAME `tv_st` bijection and the
          same C-fragment view, then scatters to (m, d); its block size goes
          128 -> 256 to match the TV's thread layout.
      One `_make_flat_staging()` feeds both call-sites so the two ends cannot
      drift apart (a second spelling scrambles silently, with no error).
      The partition and the three register views are hoisted out of the trip
      loop as loop invariants.  The intra-tile bijection C fragment -> smem ->
      bulk reduce-add -> `_post_kernel` is GPU-certified at max_diff=0 by
      notes/probe_d05_staging_atom.py C1, and the tile-id/offset agreement
      plus ADD-not-overwrite by its C2.  smem bytes, the bulk reduce-add's
      fence/commit/wait order, the staging region's alias reuse by the dK/dV
      epilogue, sdS and all five MMA chains are unchanged.

D0.5  part 3 of 4 -- the OPERAND-FACE write side (this commit): sK, sV, sQ and
      sdO stop being filled by a cooperative per-element generic-proxy loop
      (32 scalar loads + 32 scalar stores per thread per face) and are filled
      by TMA-bulk G->S, descriptor driven, exactly as the buffer list pins.
      Nothing else moves: the four smem layouts, every WGMMA descriptor, the
      row scalars, sdS, the staging face and all five chains are byte-identical
      to part 2.

      Mechanism, one high-risk item mounted alone (the smoke discipline):
        - Four `cpasync.make_tiled_tma_atom(CopyBulkTensorTileG2SOp(), ...)`
          descriptors are built in `_launch` -- not in `_build_objects` -- so
          each is bound to its global tensor.  Each is handed the SAME smem
          layout object MAIN reads through, so the TMA descriptor's swizzle and
          the WGMMA descriptor's swizzle cannot drift apart.
        - The (B, H, N, D) tensors are re-viewed rank-4 as (N, D, H, B) with
          stride (D, 1, N*D, H*N*D).  With the tile's two modes leading, ONE
          rank-2 tiler ((BN, HD) for K/V, (BM, HD) for Q/dO) plus a
          `local_tile` coordinate (n_blk, 0, h, b) addresses any tile, and
          `cpasync.tma_partition(atom, 0, make_layout(1), ...)` (no multicast,
          single-CTA cluster) gives the src/dst pair.
        - Two mbarriers come out of the new `mbarrier + padding` cell of the
          buffer list, placed FIRST in SharedStorage so the operand faces'
          1024-B alignment absorbs the residual.  One barrier covers the K+V
          group, one the Q+dO group; each group declares its exact transaction
          byte count once via `arrive_and_expect_tx`, then issues its two
          `cute.copy` calls against that barrier, and every thread blocks on
          `mbarrier_wait`.
        - Barrier re-initialisation per fill, NOT a running phase parity: both
          fill sites sat under dynamic loops at the time (`while` over work
          items, `range` from a dynamic `n_blk`), so no parity was derivable
          from the trip index here.
        - Predicate discipline: `mbarrier_init` and `arrive_and_expect_tx` are
          single-thread instructions and sit in `elect_one`; the `cute.copy`
          atom elects internally and is issued from the warp-uniform predicate
          `warp_idx == 0` WITHOUT a second single-thread wrapper.  Wrapping it
          again is what deadlocks the warp.
        - The row-scalar gather stays a generic-proxy store and keeps its own
          `fence_view_async_shared`; it is now issued while the Q/dO TMA group
          is in flight.  The async-proxy visibility of the four operand faces
          comes from the mbarrier wait, so their old fence is gone.

      GPU certification, notes/probe_d05_tma_fill.py, all 11 rungs completed
      and every bounded wait was satisfied: C1 PipelineTmaAsync control face,
      C2/C3 TMA face == cooperative face at descriptor coord (0,0,0) and at a
      non-zero (b, h, n_blk), C4 the same fill with the BLOCKING
      `mbarrier_wait` this file ships, C5 under a cluster launch, C6 through a
      rank-2 descriptor, C7 the re-init-per-trip discipline this file ships
      over 5 trips, C8 the phase-parity alternative, C9 a TMA-filled sK/sQ pair
      consumed by the chain-0 WGMMA against an exact reference -- every one at
      max_diff=0.

D0.5  part 4 of 4 -- the ROW-SCALAR access atoms and the STAGE COUNTS that
      close the 199680 B buffer list (this commit).  With it the D0.5 table is
      fully built: every buffer in the list exists at its pinned size and both
      of its access atoms are the pinned ones.

      (a) sLSE / sdPsum WRITE atom.  The two row-scalar vectors are no longer
          gathered by `if tidx < BM: sLSE[tidx] = mLSE[...]` (128 scalar
          global loads + 128 scalar smem stores per trip).  Each is filled by
          ONE non-tensor 1-D bulk copy of ROW_BYTES = 512 B:
              cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
          (`_bulk_g2s`, inline PTX -- the DSL wraps only the bulk-TENSOR
          direction, the same gap `hw_kb/cutedsl/sm90/s2g_bulk_reduce.md` documents for
          the S2G reduce).  Unlike the DSL's TMA copy atom this bare PTX does
          NOT elect internally, so it is issued from `elect_one` inside the
          warp-uniform `warp_idx == 0`.
      (b) They ride the Q/dO barrier as a MIXED transaction: `barQ`'s single
          `arrive_and_expect_tx` now declares
          `tx_q = 2*sQ_bytes + 2*ROW_BYTES = 33792`, covering two bulk-TENSOR
          TMA faces and two non-tensor bulk vectors on one barrier.  No third
          wait, and the row scalars' old `fence_view_async_shared` is gone --
          the mbarrier carries that ordering now.
      (c) D0.7's `lse * log2e` moves off the per-element path and onto the row:
          the bulk copy lands the natural-log domain lse verbatim, and one
          thread per row scales it in place (`sL[tidx] *= LOG2E`) between the
          mbarrier wait and the barrier that already followed it.  One
          instruction per row, no new barrier.
      (d) sLSE / sdPsum READ atom: `LDS.64`.  Both elementwise segments
          (the exp2/P segment and the dS segment) stepped the C fragment by 1
          and indexed `sLSE[qr]` / `sDel[qr]` once per element.  They now step
          by 2 and issue ONE 64-bit load per adjacent-q pair through a
          `(2, BM/2):(1, 2)` pair view, `cute.autovec_copy(sL2[None, qr0 // 2],
          rl)`.  The pairing is a property of the C fragment's FLAT INDEX
          ORDER, not of host-visible coordinates: `cS[i]` yields Int32 (a WGMMA
          C fragment's q coordinate carries lane), so no trace-time pair table
          exists; what holds is that for even i, qr[i] is even, qr[i+1] ==
          qr[i]+1 and kv[i+1] == kv[i].
      (e) STAGE COUNTS.  Q/dO 2 stage, sdS 2 stage, row scalars 2 stage; K/V
          stay single-buffered (CTA lifetime invariant).  Each staged face is
          ONE MemRange holding a 3-mode `tile_to_shape(atom, (.., .., NSTAGE),
          order=(0,1,2))` face, sliced at the dynamic stage `m % NSTAGE`; the
          transposed views carry the stage mode with stride = single-stage
          cosize.  Offsetting a MemRange iterator and rebuilding the view with
          `make_tensor(ptr, <ComposedLayout>)` does NOT survive the tracer --
          the sliced route is the only one that does, and it is also the route
          `get_tensor(l.outer, swizzle=l.inner)` already uses everywhere else
          in this file.  The TMA descriptors keep addressing the SINGLE-stage
          layout (a descriptor describes one tile); only the destination is a
          stage slice.
          The buffer list closes exactly:
            mbarrier + padding         1024
            sK + sV                   32768   (single)
            sQ + sdO                  65536   (2 stage x 16 KiB each)
            sdS                       65536   (2 stage x 32 KiB)
            sdQ staging               32768   (2 WG x 16 KiB, dK/dV alias it)
            sLSE + sdPsum              2048   (2 stage x 512 B each)
                                    = 199680 B

      GPU certification, notes/probe_d05_rowscalar_stage.py, all 8 rungs
      completed: C1/C2 the 1-D bulk fill == the cooperative gather under both
      barrier-address spellings, C3 the mixed transaction (2 TMA faces + 2 bulk
      vectors on ONE expect_tx) == control on all four faces, C4 the LDS.64
      pair read over the real C-fragment partition (plus qr[i+1]-qr[i] == 1 on
      even columns only), C5/C6 both stages of the sliced staged Q face and the
      sliced staged row-scalar face == control, C7 a staged face consumed by
      the chain-0 WGMMA and read back through the staged transposed view, C8
      the whole composition (staged fill + mixed tx + pair read + WGMMA) over a
      dynamic trip loop -- every one at max_diff=0.

  D0.4 ⊗ D0.8, part 1 of 2 — warp roles, thread count and the register split.
      MAIN now launches 384 threads = 3 warpgroups with `min_blocks_per_mp=1`,
      and forks ONCE on the warpgroup-uniform predicate `warp_idx < 4`:

        warpgroup 0 (warps 0-3)   producer.  `setmaxnreg.dec 24`.  Warp 0 alone
                                  issues every TMA, every 1-D bulk G->S and
                                  every bulk reduce-add; warps 1-3 carry only
                                  the barrier skeleton and the BM-wide row
                                  pass that applies log2e to lse.
        warpgroups 1-2 (warps 4-11)  consumer.  `setmaxnreg.inc 240`.  Both
                                  tiled MMAs, the two elementwise segments and
                                  the staging writes are built over the
                                  consumer-local id `ctid = tidx - 128`, so
                                  every TV partition still spans 256 lanes.

      128*24 + 256*240 = 64512 <= 65536.  The shared preamble (smem base, the
      mbarrier addresses and every smem face) stays IN FRONT of the fork and
      every accumulator fragment is created INSIDE the consumer branch, so no
      heavy live range crosses the producer's `.dec` point
      (hw_kb/cutedsl/sm90/warp_specialization.md gotchas 5, 6, 11) -- which is also why
      the D0.2 grid decode is repeated inside each branch instead of being
      hoisted.  The two bodies run identical barrier sequences, once per CTA:
      B1 B2 [trip loop, no whole-CTA barrier]* B8 B9 B10 B11 B12.

      Certified on GPU by notes/probe_d04_ws.py, 7/7 contracts PASS.  C1 the
      whole machine (PRE+MAIN+POST) at a small shape against a float32
      reference computed in the probe, max_diff=8.157e-4 (harness bar 2e-2 *
      |ref|_max).  C2 the gate's prerequisite smoke certificate: all three
      kernels lower and all launches complete at the REAL benchmark shape
      (10,16,2,4096,64), outputs finite and non-degenerate.  C3 the PTX
      carries `.reqntid 384`, `.minnctapersm 1`, `setmaxnreg.dec ... 24` and
      `setmaxnreg.inc ... 240`.  C4 MAIN's SASS LDL/STL == 0.  C5 the SHIPPED
      SASS carries the setmaxnreg-family instructions, i.e. ptxas did not
      silently discard the split (the discriminator has to be the SASS: NCU's
      `regs_per_thread` reads the CTA mean 65536/384 = 168 whether the split
      survived or not, and the PTX carries the instruction either way).  C6 the
      control with the two `setmaxregister_*` calls deleted carries ZERO of
      them, so C5's count discriminates rather than matching noise.  C7 every
      rung completed on the device -- no barrier-count drift between the two
      halves.  The informational `min_blocks_per_mp`-less rung came back at
      LDL/STL = 0 too: this kernel's live set still fits under the 168-register
      entry budget, which is why the spill count could not serve as C5's
      discriminator here.

  D0.4 ⊗ D0.8, part 2 of 2 — the 7-segment issue order and its per-segment
      live-set budget.  Part 1 froze WHO runs and on what register split; this
      step freezes WHEN each chain is issued and which elementwise segment runs
      in which chain's shadow.  MAIN's trip body used to issue every chain and
      drain it immediately (`commit_group` + `wait_group(0)`), so no elementwise
      work was ever covered.  It now reads:

        A  wait Q (+ the two row scalars) -> issue chain 0 (S^T = K.Q^T)
           -> ONLY THEN wait dO -> issue chain 1 (dP^T = V.dO^T)
        B  wait_group(1): chain 0 retires, CHAIN 1 STAYS IN FLIGHT
           -> the softmax segment (mask, scale fold, exp2, f16 pack into rP)
        C  issue chain 2 (dV += P^T.dO, A out of rP) FIRST
           -> wait_group(1): chain 1 retires, CHAIN 2 STAYS IN FLIGHT
           -> the dS segment (dP - delta, times P; the scale is already in
              both terms since D1.3 dial b)
        D  the f32->f16 pack and the transposed stmatrix store to sdS,
           INTERLEAVED one chunk at a time -- also in chain 2's shadow
        E  no MMA issue (chain 2 moved up to C)
        F  fence -> the kernel's ONE symmetric rendezvous (256 threads, the
           cross-warpgroup dS exchange) -> issue chain 3 (dQ = dS.K)
           -> wait_group(1) (chain 2 retires: `rP` dies, dO is free)
           -> issue chain 4 (dK += dS^T.Q) -> wait_group(1) (chain 3 retires)
           [the two depths on this line were later dialled by D2.1: the first
            is now wait_group(2) and retires nothing, so chains 2 and 3 both
            fly across chain 4's issue and the second retires them together;
            the SEGMENT ORDER above is unchanged.  See the M2/D2.1 entry.]
        G  staging-empty handshake -> vectorised sdQ write -> fence ->
           staging-full handshake -> wait_group(0) (chain 4 retires: `rDS` and
           the Q face die) -> back edge

      Two mechanisms carry it.  (1) The Q+dO fill barrier is SPLIT: `barQ` now
      carries the Q TMA plus the two bulk row-scalar vectors (`tx_q`) and the
      new `mbardO` carries dO alone (`tx_do`), because a consumer that cannot
      wait for Q without also waiting for dO has no window to hide dO's TMA in.
      The mbarrier cell keeps its declared 8-barrier footprint (`mbarRsv` drops
      6 -> 5), so smem stays at 199680 B and the barrier skeleton is unchanged:
      both roles still run B1 B2 [B3..B8]* B9..B12.  (2) Segment D's store is
      driven per chunk through `_copy_chunk` over the grouped trailing modes of
      the same `r2s_ds` face, so a packed f16 chunk is stored before the next
      chunk's f32 values are computed and no high-precision copy of a packed
      element survives the segment.

      Register-wise this is close to neutral: every accumulator and both RMEM A
      fragments were already allocated once outside the trip loop, so the peak
      the graded waits pin down (chain 1's 64 accumulator registers alive
      through the softmax segment, `rP`'s 32 alive from the softmax segment to
      chain 2's retirement in segment F) is a live-range statement, not a new
      allocation.  The one live-range EXTENSION this step makes is `rP`'s, and
      it lands in segments C-E where chain 1's accumulator has already died.

      The issue order itself has no machine-readable check on the student side
      (every architectural line passes with the windows empty); its fallback
      line is the M0 Gate's `tensor_pct` floor, and `spill_{ld,st}_sectors == 0`
      remains the only live-set line that gates.

      The two pieces of it that CPU-side reasoning cannot settle were run on
      GPU first by notes/probe_d08_order.py, 4/4 contracts PASS: C1a the
      monolithic sdS stmatrix write as a rig control (max_diff=0), C1b the
      chunked interleaved write against the same face and payload (max_diff=0),
      C2a `barQ` = Q + row scalars consumed before dO's wait, C2b `bardO` = dO
      alone waited after the Q segment -- C2 on the real 384-thread fork, since
      a wrong arrive count on a split barrier hangs rather than faults.

      Not part of this step, by contract: the row-scalar READ atom stays fused
      into the elementwise segments (D0.5 froze it as one LDS.64 per adjacent-q
      pair); hoisting those reads up into segment A is D1.5, an M1 lever.  The
      wait DEPTHS (0/1/2) are D2.1's candidate domain -- what is frozen here is
      the shadowing relation, not the numbers.

  D0.9 part 1 of 2 — the two 2-stage {full, empty} operand pipelines.
      D0.9 rules three handoff classes.  This step lands class 1 (the
      producer -> consumer operand flow) and NOTHING else; part 2 of 2 lands
      class 2 (the dQ staging credit ring, 4 asymmetric named barriers at 160
      participants) and class 3 (the dS exchange as the kernel's only
      symmetric 256-thread rendezvous), and only then can the producer
      actually run a tile ahead.  The split is forced by the mechanism, not by
      taste: the trip loop is still fenced by the whole-CTA barriers B5..B8, so
      after this step producer and consumer are STILL in lockstep trip for
      trip.  That is the point — the ring's arrive counts, its slot map and its
      phase arithmetic go up first with those barriers as a net, and part 2
      pulls the net.  Anything the ring gets wrong is a deadlock either way,
      but with the net it is a deadlock in ONE new mechanism.

      What the operand flow was until this step: one barrier per fill group,
      RE-INITIALISED EVERY TRIP, with a whole-CTA rendezvous (B3, B4) on each
      side of it.  Every fill was therefore a convoy — the producer could not
      start tile m+1's TMA until the consumer had finished tile m.

      What it is now.  Two independent 2-stage pipelines, A = Q + LSE and
      B = dO + delta, each stage carrying a {full, empty} pair, in the pinned
      slot layout `base + stage*8` with `{+0x00 full_A, +0x10 empty_A,
      +0x20 full_B, +0x30 empty_B}` (`B_FA`/`B_EA`/`B_FB`/`B_EB`).  Three
      barriers sit outside that 8-slot footprint: `B_KV` (K/V is single
      buffered and filled once per work item, so it is not a pipeline) and
      `B_RL[stage]`, the private landing barrier the LSE vector needs.

      Why LSE needs a private barrier — this is the one place the pipeline
      shape is decided by a D0.7 ruling rather than by D0.9.  D0.7 item 3
      keeps `lse` in the natural-log domain across the kernel boundary and
      applies `log2e` ONCE PER ROW inside MAIN (it explicitly vetoes a
      log2-domain copy out of PRE).  So stage A is not publishable the moment
      its bytes land: the producer must see the vector, scale it, and only
      then publish.  A barrier cannot be both the thing the producer waits on
      and the thing the producer must arrive on, so the vector lands on
      `B_RL[stage]` (512 B) and the publication is the producer's own arrive
      on `full_A[stage]`.  Q's 16 KiB TMA is still in flight across that wait.
      delta needs no such detour and rides `full_B` directly, which is why
      `tx_b` is the mixed transaction now (one bulk-TENSOR dO face + one
      non-tensor bulk vector on one expect_tx) and `tx_a` is Q alone.

      Arrive counts (wrong small = silent early release, wrong big = deadlock;
      neither has a line number):
        full_A  1 + BM = 129   Q's `arrive_and_expect_tx`, plus one arrive from
                               each of the BM producer threads after it has
                               scaled its own lse row
        empty_A 256            every consumer thread releases
        full_B  1              `arrive_and_expect_tx` alone (tx publishes)
        empty_B 256
        B_RL    1              the lse vector's tx
        B_KV    1              the K+V tx

      Phase arithmetic.  The barriers are initialised ONCE PER CTA (not per
      trip), ahead of B1, and that is what makes a parity derivable at all:
      the trip index `t = m - n_blk` starts at 0, so `lap = t // NSTAGE` and
      the parity a wait names is `lap % 2` for the full/landing barriers and
      `(lap + 1) % 2` for the empty barriers (the previous lap's).  The producer
      SKIPS its empty wait for the first NSTAGE trips (fresh slots, nothing
      has ever released them).  Both roles compute `t`, `stage`, `lap` and the
      parities from the same two runtime values; the buffer slice index moves
      from `m % NSTAGE` to `t % NSTAGE` on both sides so slot and lap line up.

      Release points are per pipeline, not per trip end: pipeline B is released
      one whole segment earlier than A, right after the graded wait that
      retires chain 2 (dO's last reader; delta's last reader, the dS segment,
      is already behind it), while A is released after the segment-G drain
      that retires chain 4 (Q's last reader).

      Addressing.  `stage` is a runtime value, so the barrier a trip uses has
      to be selected by a runtime index.  The barrier cell becomes ONE
      `MemRange[Int64, 11]` viewed as `(1, NMBAR)`, and `_bar(bar_t, i)` =
      `bar_t[(None, i)].iterator` -- a runtime-sliced sub-range's iterator IS a
      `cute.Pointer`, the route hw_kb/cutedsl/sm90/pipeline_runtime_subring.md
      certifies for this exact problem.  88 B of barriers still fit the 1024-B
      `mbarrier + padding` cell, so smem stays 199680 B.
      `_mb_arrive` is inline PTX for the same reason `_bulk_g2s` is (the DSL
      exports the expect_tx form; the bare arrive's spelling moves).

      Retired by this step: the B3/B4 whole-CTA fill rendezvous, and the
      per-trip `mbarrier_init`.  The remaining barrier skeleton is
      outer B1 B2 [inner]* B9..B12, inner B5 B6 B7 B8, identical in both roles
      (the labels B3/B4 are left retired rather than renumbered so the D0.8
      segment map keeps naming the same barriers).

  D0.9 part 2 of 2 - handoff classes 2 and 3, and the producer run-ahead.
      Part 1 mounted the operand pipelines with the whole-CTA barriers
      B5..B8 still around them as a net, so producer and consumer were still
      in lockstep trip for trip.  This step pulls the net.  Nothing about
      class 1 changes; what changes is everything the trip loop used
      `cute.arch.barrier()` for.

      Class 3 - the dS exchange.  B5 becomes `bar.sync NB_DS, 256`: the
      kernel's ONE symmetric rendezvous, and CONSUMER THREADS ONLY.  The
      producer used to be dragged into it 384-wide once per trip for a
      handoff it has no part in.  Two consumer warpgroups exchanging one
      2-stage sdS face need exactly one meeting point per trip: WG w's write
      of stage s (segment D) precedes it, both WGs' chain-3 read of stage s
      follows it, and chain 3 is retired by the `wait_group(1)` two segments
      later -- inside the same trip -- so the WAR edge against trip t+2's
      write of the same stage is already covered without a second barrier.

      Class 2 - the dQ staging credit ring, 4 asymmetric named barriers.
      Each consumer WG owns a pair, `NB_FULL + wg` and `NB_EMPTY + wg`, both
      with STAGE_CREDIT = 160 = 128 + 32 participants (one consumer WG plus
      the producer's warp 0).  The asymmetry is the point and it runs one way
      per barrier:
        report full   consumer's 128 threads `bar.arrive` (non-blocking),
                      producer's warp 0 `bar.sync` (blocks)
        hand back     producer's warp 0 `bar.arrive` (non-blocking),
                      consumer's 128 threads `bar.sync` (blocks)
      A named barrier is legal here where it would be a protocol disease on
      the operand flow (hw_kb/cutedsl/sm90/handoff_sync_choice.md gotcha 1): this is a
      constant-frequency structural handoff of a buffer the consumer owns,
      not a streaming fill.  Two barriers become four because the staging face
      already IS two disjoint 16 KiB halves -- `(SLOT, NUM_WG)`, stride
      `(1, SLOT)`, thread layout `(lane, wg)` -- so WG0 and WG1 never touch
      each other's bytes and never need a common point.  The producer's single
      32 KiB `cp.reduce.async.bulk` therefore splits into two 16 KiB ones at
      `stage_addr` and `stage_addr + SLOT_BYTES`, byte-for-byte identical in
      the workspace to the one they replace, each gated by its own WG's
      `full`.  It gives buffer 0 back with buffer 1's write-back still in
      flight (`cp.async.bulk.wait_group.read 1`).

      The producer run-ahead.  With the net gone the loop would still be
      lockstep for a reason that has nothing to do with barriers: the producer
      issues trip t's TMAs and then waits for trip t's dQ, which the consumer
      only produces at the END of trip t -- so tile t+1's fill would be issued
      after the consumer had already asked for it.  So the staging service is
      LAGGED ONE TRIP: at trip t the producer retires trip t-1's tile.  Trip 0
      services nothing (a pre-loop `bar.arrive` on each `empty` is the ring's
      first credit) and the last trip is serviced after the loop.  Arrivals
      balance over a work item by construction: `empty` gets 1 + (n-1) grants
      against n waits, `full` gets n arrives against n syncs, so both
      barriers are quiescent when the CTA reaches its epilogue.

      B8 survives, but once per CTA instead of once per trip, and it
      now carries a single meaning: the producer's last dQ write-back has
      finished reading the staging face, so the D0.12 dK/dV flush may alias
      it.  The trip loop contains no whole-CTA barrier at all.

      Why the producer warpgroup does not tear itself apart: warps 1-3 are
      not in any of the new barriers, so they run past the service block into
      the next trip -- and park there on that trip's `B_RL` wait, which only
      warp 0's bulk copy can satisfy.  The producer's four warps therefore
      cannot drift more than one trip, and warps 1-3 can never arrive on a
      `full_A` the warp 0 has not yet filled.

      `_nb_sync` / `_nb_arrive` are inline PTX (`bar.sync` / `bar.arrive`,
      both operands in registers) for the same reason the mbarrier arrive is:
      the id has to carry a runtime `wg_idx`, and a wrong participant count is
      a deadlock or a silent early release with no line number either way.

  D0.2 re-delivered to its declared value: the persistent 132-CTA work queue is
      gone and MAIN launches grid = (N/bN, H_Q, B) = (32, 16, 10) = 5120 CTAs,
      one work item per CTA, x = the KV block and the fastest varying axis.
      What the queue cost and what replaces it:

        wq_counter (1,) i32     deleted from the D0.1 ledger; PRE no longer
                                zeroes it, `_workspace` no longer allocates it,
                                and the `atom.global.add.u32` wrapper is gone
                                (the reference's SASS carries ATOM*/RED = 0).
        sIdx (16 B smem)        deleted from the D0.5 buffer list.  It lived in
                                the mbarrier cell's alignment residual, so the
                                total is still exactly 199680 B.
        `while sIdx[0] < n_work` deleted from BOTH role bodies; each is now
                                straight-line and runs its barrier skeleton
                                once.  B13 (the work-item loop's closing
                                rendezvous) and the "claim the next id before
                                the epilogue" atomic go with it.
        (b, h_q, n_blk)         decoded from `cute.arch.block_idx()` INSIDE
                                each role branch -- the producer takes all
                                three (plus `hk = h // gqa`), the consumer only
                                `n_blk`.  Keeping the decode behind the fork is
                                deliberate: a preamble value would live across
                                `setmaxnreg.dec` into a 24-register budget.

      The mapping is linear -- no swizzle, no lookup, no load balancing for the
      triangular causal work distribution; 5120 CTAs over 132 SMs is 38.79
      waves and the imbalance averages out across them.

  D0.10 consumption phase `[PHASE: convoy]` and supply form `[GATHER: none]`.
      Two rulings, both transcribed against the code as it now stands.

      1. `[PHASE: convoy]`.  The two consumer warpgroups step in lockstep and
         meet EXACTLY ONCE per trip, at the dS exchange (`_nb_sync(NB_DS, 256)`
         in segment F).  There is no phase token, no ping-pong, no `bar.arrive`
         / `bar.sync` pair carrying a warpgroup-asymmetric turn: the only
         consumer<->consumer signal in the kernel is that one symmetric
         rendezvous, and the per-WG credit barriers (NB_FULL/NB_EMPTY + wg) are
         each private to one warpgroup, so neither WG can observe the other
         there.  The elementwise segments are NOT masked by the partner
         warpgroup -- they are masked by this warpgroup's own in-flight MMA
         chains, which is what D0.8's segment order buys: the softmax segment
         runs under chain 1 (wait depth 1) and the dS + pack/store segments run
         under chain 2, issued ahead of them in segment C.

      2. `[GATHER: none]`.  No global access in MAIN is address-dependent.  The
         four operand faces come from `.tile` TMA descriptors addressed by
         affine coordinates `(m, 0, h, b)` / `(n_blk, 0, hk, b)`; the two row
         scalars come from 1-D `cp.async.bulk` runs at an affine element offset
         `((b*H_Q + h)*N + m*BM)`; the three write-backs are
         `cp.reduce.async.bulk` at affine tile offsets.  The GQA map
         `hk = h // gqa` is integer arithmetic on a grid coordinate with a
         compile-time divisor, not an `x[idx[i]]` indirection.  With the D0.2
         work queue gone, MAIN issues no global load and no global atomic
         outside these descriptors and bulk runs.

      Blocking-point census per trip, against D0.10's advisory accounting line
      (reference: consumer 7, producer 8):
        consumer 9 = 2 named-barrier blocks (NB_DS, NB_EMPTY+wg)
                   + 2 mbarrier `full` waits (full_A, full_B)
                   + 5 WGMMA wait points (four graded waits after chains
                     1/2/3/4 and one `wait_group(0)` drain; the count is what
                     this census fixes, the depths are D2.1's dial and the
                     third of them now reads `wait_group(2)`).
                     The reference spends 3 here; the two extra are wait-depth
                     positions, which is D2.1's dial, not this decision's.
        producer 7 = 2 named-barrier blocks (NB_FULL+0, NB_FULL+1)
                   + 3 mbarrier waits (empty_A, empty_B, the private B_RL
                     landing barrier)
                   + 2 bulk write-back waits (`wait_group(1)`, `wait_group(0)`).

  ==========================================================================
  M0 Gate -- closing row (submitted alone; no decision is bundled with it)
  ==========================================================================

  Prerequisite (checklist first line): a passing WHOLE-MACHINE SMOKE probe on
  this lineage.  notes/probe_m0_operands.py rung `smoke`, taken on the parent
  commit b5eca0d that this row ships: PRE + MAIN + 3x POST all lower and all
  five launches COMPLETE at the real shape (10, 16, 2, 4096, 64) = the declared
  32x16x10 grid, dq/dk/dv finite and non-degenerate (|dv|_max = 6.80859).  No
  numeric compare there by construction -- the numbers are this row's own eval.

  correctness
    dq/dk/dv vs ref32 .......... owned by this row's eval (last measured
                                 correctness=true, iter 41, sha b5eca0d).
    data-flow-through line ..... notes/probe_m0_dataflow.py: the composed
                                 backward tracks an analytic fp32 reference
                                 (rel max_diff 4.43e-4) and a back-to-back
                                 second call reproduces call 1 bit-for-bit, so
                                 the output is a function of the input and not
                                 an initialisation constant.  No `constant_output`
                                 flag has been raised on this lineage.

  Architectural-conformance lines   [VERIFY: ncu-direct, digest of sha b5eca0d]
    block_size ................. 384          == 384                    (D0.4)
    grid_size_dims / grid_size . 32x16x10 / 5120  == declared           (D0.2)
    smem_dynamic / smem_static . 199680 / 0   in [180224, 204800] / == 0 (D0.5)
    regs_per_thread ............ 168          in [150, 185]             (D0.4)
    spill_ld / spill_st sectors  0 / 0        == 0 / 0              (D0.4/D0.8)
    kernel_count ............... 6            in [3, 6]                 (D0.1)
    dominant_time_pct .......... 86.59%       >= 80%                    (D0.1)
      -- the anti-delegation reading of that last line: the whole attention
      backward math sits in ONE kernel (MAIN), not scattered over a pile of
      torch elementwise kernels.  The probe's `rank` rung ranks the composed
      call independently and agrees: MAIN 86.24%, PRE 7.0%, the three POSTs
      4.7 / 0.8 / 0.7%, and the only non-ours entry is 0.6% of fill kernels.

  Mechanism-verification lines
    tensor_pct ................. 47.84%   >= 45%   (D0.8) [VERIFY: ncu-derived]
      IN THE [45%, 50%) ATTRIBUTION BAND -- PASS, and the band obliges this row
      to name the masking-window evidence rather than just the number.  The
      evidence is the segment order as built, all of it visible above in the
      consumer body:
        (a) the pipeline-A wait (Q + lse) is taken, then chain 0 (S^T = K.Q^T)
            is ISSUED, and only THEN is the pipeline-B wait (dO + delta) taken.
            dO's TMA latency therefore runs underneath chain 0 instead of in
            front of it.  Hoisting the B wait up next to the A wait leaves
            every segment below unchanged, spills nothing and passes every
            architectural line above -- and empties this window; that is why
            the split is load-bearing and not cosmetic.
        (b) the wait in front of the softmax segment is a GRADED
            `wait_group(1)`, not `wait_group(0)`: chain 0 retires because the
            softmax reads its accumulator, and chain 1 (dP^T = V.dO^T) stays
            in flight ACROSS the whole softmax segment.  One chain airborne at
            wait depth 1 is exactly the "leave one in flight" the band asks to
            see.
        (c) the same shape one segment down: chain 2 (dV += P^T.dO, K depth
            128 = 8 HGMMA beats, the trip's longest) is issued in segment C
            AHEAD of the dS segment, then another graded `wait_group(1)`
            retires chain 1 and leaves chain 2 airborne over the dS segment and
            the pack/store segment -- the trip's only priced elementwise cost.
      Wait-depth positions themselves are D2.1's dial, not this gate's.
    residency, READ side ....... dram_bytes_read 4.039e8 B  <= 6.0e8 B
                                 dram_pct 8.24%             <= 25%      (D0.2)
    residency, WRITE side ...... dram_bytes_write 1.928e8 B <= 2.4e8 B   (D0.2)
      -- an independent branch from the read side.  dq_accum's per-iteration
      reduce-add moves 2.768e9 B of raw traffic (84480 trips x 32 KiB, pure
      shape arithmetic); landing at 1.928e8 B means L2 absorbed it ~14x, which
      is what the D0.2 axis order (x = the KV block, fastest varying) buys.
    sync mechanism ............. stall_barrier 1.96%  <= 4.0%           (D0.9)
    operand source ............. mma_a_reg = 16 >= 8                    (D0.6)
      [VERIFY: sass-derived].  This arm's ncu task does not run `sass_census`,
      so no digest on this lineage carries mma_* fields; the census is taken
      instead by notes/probe_m0_operands.py per
      hw_kb/cutedsl/sm90/mma_operand_source_verify.md, and computed by the script:
        SASS-OPERANDS kernel=kernel_cutlass__main_kernel_tensor0000o40966...
                      mma_total=32 a_reg=16 a_desc=16 a_other=0
      Adoption prerequisite (that recipe's discipline 3) met: the censused
      section IS the dominant kernel (86.24% of CUDA time, bar 80%), checked in
      the probe's own `rank` rung rather than assumed.
      Reading: D0.6's two `[SOURCE: RS]` rows -- chain 2 (dV += P^T.dO) and
      chain 4 (dK += dS^T.Q), both tiler (64,64) over K=128 = 8 beats -- put
      their A operand in the register file, 16 beats total; SASS shows them as
      `HGMMA.64x64x16.F32 | A=R88..R112`.  The SS chains 0/1/3 supply the other
      population, `A=gdesc[UR8]`, a_desc=16, with a_other=0 unclassified.  Two
      populations out of ONE classifier is what makes the count non-vacuous: a
      classifier that scored everything REG would have reported a_desc=0.  A
      host-only control rung reproduces a hand-built SASS table exactly
      (mma_total=4 a_reg=2 a_desc=2 a_other=0), including the `gdesc[UR4]`
      A-position descriptor that must NOT score as a register and the
      HMMA/LDSM lines that must not enter a wgmma census.
      Paired line: MAIN's SASS carries ZERO LDL/STL in the same dump, so the
      register pressure RS buys is not being paid back as spill.
    absolute performance ....... NOT SET at M0 by construction; the performance
                                 ladder starts at M1.

  Closing action: with this gate PASSED, the kernel signature and the whole
  resource-topology contract (kernel split, grid, thread count and register
  split, the smem buffer list and its access atoms, operand sources, precision,
  segment order, sync topology, the two reduction protocols) are FROZEN.  M1+
  is incremental only and must not modify the topology.

  ==========================================================================
  M1 -- D1.1 (x) D1.2, delivered as ONE commit
  ==========================================================================

  The two decisions are co-committed because they share one budget (the
  consumer's 240 registers / segment B's non-accumulator working set) and one
  observation (`spill_{ld,st}` + `tensor_pct`): D1.2 changes how many
  instructions and how many live values the two elementwise segments need,
  D1.1 decides where the integer part of that working set lives.  Neither can
  be priced with the other's occupancy on top of it.

  Shared budget and knobs, stated once for the pair:
    budget ..... consumer `setmaxnreg.inc 240`, producer `setmaxnreg.dec 24`
                 (D0.4, frozen); segment B's non-accumulator working set
                 target W <= 97 (D0.8); accumulator residency unchanged --
                 no chain's issue point, wait depth or live range moves here.
    knobs ...... D1.1 ladder rung = r1 ONLY (the candidate domain's smallest
                 member; r2 = warp-broadcast row scalars and r3 = hoisted
                 containers are NOT applied).  D1.2 = all three of its items.
    unchanged .. smem 199680 B, grid 32x16x10, 384 threads, tile (BN,BM) =
                 (128,128), the D0.8 seven-segment order, every wait depth,
                 the D0.6 operand sources, the D0.7 precision contract.  Both
                 elementwise segments still read the same accumulators and
                 write the same two RS fragments and the same sdS face.

  D1.2  instruction count of the elementwise segments, three items.

    item 1  `exp2` on the fast path.  Both arms of the softmax segment call
            `cute.math.exp2(..., fastmath=True)`; the argument is
            `acc_s[i] * scale_log2e - lse_row`, i.e. the D0.7 constant fold is
            already done (log2e is applied once per row by the producer, and
            the softmax scale rides in `scale_log2e`), so the fast path sees
            one FFMA and one EX2 per element and nothing else.

    item 2  packed convert straight into the RS operand.  `_cvt_f16x2_f32`
            emits ONE `cvt.rn.f16x2.f32` (one SASS F2FP) per adjacent element
            pair in place of two scalar `F16(x)` converts, and its result is
            written through `_as_i32`, a 32-bit view of the SAME fragment the
            WGMMA A operand reads -- element i of the view IS the f16 pair
            (2i, 2i+1), so the fragment/accumulator bijection every consumer
            already relies on is untouched and no extra staging buffer
            appears.  Operand order is the trap and is spelled `$0, $2, $1`:
            `cvt.rn.f16x2.f32 d, a, b` puts the FIRST source in the HIGH half,
            while the register pair is little-endian, so the LOWER flat index
            must be handed to the SECOND operand slot.
            The dS segment keeps D0.8's interleave: pack one chunk, write that
            chunk with the transposed `stmatrix` atom, then pack the next --
            the f32 dS value never outlives its chunk.  Under the causal mask
            the two masked values are produced through the 2-element `rv`
            scratch so both mask arms stay stores and the convert still sees a
            value pair.
            `_chunk_pairs` (pair-aligned chunk check) is replaced by
            `_chunk_groups` (group-aligned), so a chunk boundary can split
            neither the packed pair nor the row-scalar group.

    item 3  row-scalar hoist.  `_ew_group` derives from the accumulator's own
            mode 0 how many elements share one row-scalar pair.  Mode 0 of a
            WGMMA C fragment is `(2, 2, V)` = (adjacent-N pair, M pair, N
            groups), so flat indices [4g, 4g+4) are a 2x2 block over ONE q
            pair: one LDS.64 of the row scalar now serves FOUR elements
            instead of two, halving the row-scalar loads in both elementwise
            segments.  The width is derived and checked (a leading extent
            other than 2 raises at trace time), not assumed.

    GPU certificate, notes/probe_d12_pack.py, 6/6 PROBE PASS on the SHIPPING
    objects (the same `_build_objects()` tiled MMAs, the same sdS stmatrix
    face, the real tile (128,128), 256 consumer threads), inputs chosen so
    every intermediate is exact in fp16 and the bar is max_diff == 0.0 with no
    tolerance band:
      C1 max_diff=0  segment B off-diagonal: hoist + packed convert reproduces
                     exp2(S*scale_log2e - lse) elementwise.
      C2 max_diff=0  segment B diagonal: same, causal mask through `rv`.
      C3 max_diff=0  CONTROL, the pre-D1.2 scalar per-pair form on the same
                     partition and data -- it shares the reference and the
                     read-back path, so a C1 failure with C3 passing would
                     indict the rewrite and nothing else.
      C4 max_diff=0  segments C+D read back OUT OF sdS equal
                     P * (dP - delta) * scale at the transposed address.
                     (Certified against the dial-a segment; D1.3 has since
                     moved the trailing `* scale` off it onto sV + delta,
                     leaving the address map and the pack unchanged.)
      C5 n/a         the 2x2 group contract on the real `partition_C`
                     coordinate map: within every group of 4, q[+1]=q[+0]+1,
                     q[+2]=q[+0], q[+3]=q[+0]+1, kv[+0]=kv[+1], q[+0] even,
                     and the 256x64 map is a bijection onto the tile.
      C6 n/a         trace-time widths: `_ew_group` = 4 on the shipping
                     accumulator shape and the segment-D chunk width is a
                     multiple of it.

  D1.1  non-accumulator working set, ladder rung r1 only.

    r1 = "the integer bookkeeping chain runs on the uniform register file".
    SM90 has a second, integer-only register file (`UR*`, per warp) that does
    not draw on the per-thread allocation `setmaxnreg` hands each role; a
    value lands there only while ptxas can prove it warp-invariant, and one
    `+ tidx` anywhere in the chain knocks the whole chain back into the
    per-thread file (hw_kb/cutedsl/sm90/uniform_reg_budget.md mechanism 1).
    `cute.arch.make_warp_uniform` is the marker, aliased here as `_MWU`.
    Marked, in both role bodies:
      - the D0.2 work-item decode (`n_blk`, and on the producer side `h`, `b`,
        `hk`), re-taken INSIDE each role branch because the marking does not
        survive the role `scf.if`;
      - the whole D0.9 trip state `t`, `stage`, `lap`, `ph_f` (and the
        producer's `ph_e`), which feeds the runtime barrier selector and every
        smem stage slice;
      - the producer's row-scalar element offset `rs_row`.
    On the consumer side the marking of `stage` is also what pins the D0.8
    WGMMA operand descriptors: `stage` slices sQ/sdO/sDS/sQt/sdOt, and those
    slices are the coordinates fed to `partition_A/B` -- the only markable
    point on a descriptor chain, since the descriptor itself is a return value
    (hw_kb/cutedsl/sm90/wgmma_desc_uniform.md r1).
    Nothing lane-dependent is marked: the marker is an assertion, not a
    conversion.  `ctid`, the per-thread partitions and every fp temporary are
    untouched -- UR is an integer path.
    Rungs r2 (row scalars by warp broadcast) and r3 (containers allocated once
    outside the trip loop) are NOT taken: the ladder is applied in order and
    r1 is the candidate domain's smallest member.  The accumulators, `rl`,
    `rd`, `rv` and both RS fragments are already allocated ahead of the trip
    loop and reused, and the row-scalar reads are still the D0.5 LDS.64 pair
    atom.

  D1.3  scale fold position: dial a -> dial b.

    D0.7 froze WHAT carries the softmax scale (dS, so that dQ and dK are both
    correct with no epilogue rescale) and left WHERE it is folded to this
    decision.  Dial a spent one FMUL per dS element.  Dial b spends none:

      - sV is multiplied by `scale` in place, once per CTA, right after the
        K/V fill barrier and before B2.  Chain 1 (`dP^T = V . dO^T`) is V's
        only reader -- dV = P^T.dO takes dO, not V -- so a scaled face is
        arithmetically a scaled dP.  The face is walked through `sVf`, a FLAT
        view of the same MemRange (no swizzle): a uniform multiply commutes
        with the swizzle permutation, so no thread<->element bijection is
        being claimed.  Scalar LDS/STS, not `cute.autovec_copy`, which is the
        known reg->smem device-hang carrier at the epilogue sites; this site
        runs once per CTA where the width buys nothing.  The generic-proxy
        stores are made visible to the WGMMA (async) proxy by
        `fence_proxy(async_shared)` and to the other 255 lanes by B2.
      - delta arrives pre-scaled from PRE (`delta = scale * rowsum(dO . O)`),
        which is the cheapest possible site for it: once per (b, h_q, row) for
        the whole launch, not once per row per trip.  Both terms of
        `dP - delta` are then already scaled, and their difference is too.
      - the dS segment is therefore
        `dS^T = P^T * (dP^T - delta)` with no trailing `* scale`.

    Accounting: -64 elementwise FMUL per lane per TRIP (the segment-C/D
    fragment is 64 elements per lane) against +32 f16 load-multiply-stores per
    lane ONCE per CTA, plus one FFMA per delta element in a PRE that is two
    streams and a store.  The real shape averages 16.5 trips per CTA.
    scale = 2^-3 is a power of two and exact in fp16, so the only
    representation risk dial b carries is a v small enough that `scale * v`
    falls subnormal; correctness is the arbiter of that, per the decision.

    GPU certificate, notes/probe_d13_scale.py, 5/5 PROBE PASS on the SHIPPING
    kernel (each rung a separate child process under a device-event budget, so
    a cross-proxy visibility bug that manifests as a stall lands as a legible
    hang rather than a silent pass):
      C1 max_diff=0.000815749
                     ship: the WHOLE machine (PRE + MAIN + POST) at
                     B,H_Q,H_KV,N,D = 2,4,2,640,64 -> grid (5,4,2) with trip
                     counts 5,4,3,2,1 (a one-trip CTA and a five-trip CTA both
                     scale their face exactly once, which is the property the
                     once-per-CTA placement has to have), against a float32
                     reference computed in the probe, under the harness bar
                     |g - ref32|_max <= 2e-2 * |ref32|_max.
      C2 n/a         real: PRE, MAIN and POST all lower and every launch
                     completes at the real shape (10,16,2,4096,64); dq/dk/dv
                     finite and not all-zero.
      C3 n/a         CONTROL `noscale` -- the multiply in the sV in-place loop
                     becomes a copy, so the loop still loads and stores every
                     element of the face but scales nothing -- does NOT
                     reproduce the reference.  This is what separates "the
                     scaled face reached the WGMMA through the proxy fence"
                     from "the answer happened to be right anyway".
      C4 n/a         CONTROL `nodelta` -- PRE writes the bare row sum again --
                     does NOT reproduce the reference, so a C1 pass cannot be
                     a kernel that dropped the scale twice.
      C5 n/a         every non-control rung completed on the device (no hang,
                     no fault) inside its event budget.

  ==========================================================================
  M1 GATE -- the M1 closing row, submitted ALONE (no decision bundled)
  ==========================================================================

  Submitted with NO executable change: the diff of this commit is confined to
  this docstring (the module AST outside it is byte-identical to 8d2e8cf), so
  the reading this row's eval produces prices the delivered M1 set and nothing
  that arrived with the gate.

  Delivered M1 set, complete: D1.1 (ladder rung r1 only), D1.2 (all three
  items), D1.3 (dial b).  Every one of them is transcribed above with its GPU
  certificate.  No M1 decision is deferred and none is outstanding.

  The three whole-machine evals this row reads, all on this lineage, all at
  the real shape (10,16,2,4096,64), device window:

    sha      iter  what                       device_ms  tensor_pct  spill
    7625649   42   M0 Gate (the pre-M1 anchor)  2.5149      47.84     0/0
    6e02d11   43   D1.1 (x) D1.2                2.4291      49.84     0/0
    8d2e8cf   44   D1.3 dial b  <- SHIPPING     2.4385      49.60     0/0

  Gate line (1) -- mechanism, spill    [VERIFY: ncu-direct]        PASS
    spill_ld_sectors == 0 and spill_st_sectors == 0.  Measured 0 / 0 on the
    shipping sha, and 0 / 0 on every eval of this lineage since iter 37 (the
    setmaxnreg split).  Bar: == 0 / == 0; reference 0 / 0 [MEASURED, job
    1787925084].  This line is not deferred and needs no role attribution:
    with both counters at zero there is nothing to attribute to the
    producer's 24-register region or the consumer's 240.  Paired with the M0
    gate's SASS line (ZERO LDL/STL in MAIN's dump), the register economy is
    clean on two independent instruments.

  Gate line (2) -- mechanism, tensor_pct    [VERIFY: ncu-derived]  SHORT
    Bar >= 50%; reference 55.67% [MEASURED, job 1787925084].  Measured 49.60%
    at the shipping sha and 49.84% at the same code with D1.3 on dial a.
    The bar is missed by 0.40 points -- 0.8% of MAIN's 2.0832 ms.
    This reading is NOT noise-limited the way the device window is: two evals
    of near-identical code (iter 41 / 42) both returned 47.84% while their
    device medians moved 2.5135 -> 2.5149, so the +1.76 point step to 49.60
    and the -0.24 point step from dial a are both reproducible orderings.
    The gate obliges a re-attribution rather than a re-run, and the evidence
    for it is inside M1 itself:
      (a) D1.2 DID take effect.  tensor_pct rose 47.84 -> 49.84 and MAIN fell
          2.1615 -> 2.0877 ms across exactly the commit that packed the
          converts, hoisted the row scalars 2->4 elements per load and put
          exp2 on the fast path.  The "not risen => D1.2 未生效" branch of
          this gate line does not apply.
      (b) Elementwise instruction count is no longer what binds.  D1.3 dial b
          deletes 64 FFMA per lane per trip (16.5 trips per CTA) from segment
          C/D and pays for it with 32 f16 load-multiply-stores once per CTA;
          it bought -0.24 tensor points and +0.0094 ms.  A pure deletion of
          ~1000 elementwise instructions per lane per CTA that moves the
          tensor pipe backwards says the remaining MAIN time is not sitting
          in the elementwise segments' issue slots.
      (c) Where it does sit, from the shipping digest's own signature (each
          number is this lineage's, against the reference band in milestone
          section 5):
            bankconf_ld ........ 3.922e6   ref 5.0e3    bar <= 1.0e5  OVER 39x
            smem_wavefronts st+ld 1.530e8  ref 7.295e7  bar <= 1.1e8  OVER
            bankconf_st ........ 3.399e6   ref 4.502e6  bar <= 6.8e6  in band
            stall_long_sb ...... 2.92%     ref 1.34%    bar <= 3.4%   in band
            stall_mio .......... 0.68%     ref 0.45%    advisory
            stall_barrier ...... 2.18%     ref 1.47%    bar <= 4.0%   in band
            dram read / write .. 4.040e8 / 1.932e8 B    ref 4.037e8 / 1.931e8
            dram_pct ........... 8.55%     ref 9.58%    bar <= 25%    in band
          The read side of the exchange surface is the only signature on this
          list that diverges by more than 2x, and it diverges on BOTH branches
          of the section-5 dual-branch table at once (conflicts over the line
          AND volume over the line).  Those two branches name D2.3 (swizzle
          mode) and the D0.5/D0.6 contracts respectively; D2.3 is an M2 lever
          and D0.5/D0.6 are frozen M0 topology.  Recorded as an attribution
          and not as a promise: a diverging signature is where the residual is
          VISIBLE, and only a measurement inside M2 can say how much of it is
          priced.  Neither is reachable from any
          M1 decision: M1's whole candidate domain is the elementwise segments'
          instruction count and non-accumulator working set, and both of those
          are now measured out (spill 0, all three D1.2 items in, the D1.1
          ladder's r1 applied and r2/r3 untriggered with spill at zero, and
          D1.3 priced at both dials).
      Consequence recorded here so the next phase reads it: this row leaves
      gate line (2) SHORT by 0.40 points with M1's levers exhausted, and hands
      the residual to M2, whose own gate carries the smem-traffic dual-branch
      line that this signature already fails.  M1 is not being closed out as
      "achieved by other means" -- the shortfall is named, attributed and
      carried forward.

  Gate line (3) -- performance ladder, end to end                   PASS
    Bar: device median <= 2.90 ms [ESTIMATE, milestone M1 Gate].  Measured
    2.4385 ms at the shipping sha (2.4291 ms at dial a), 15.9% under the bar.
    e2e window reported alongside per the milestone's two-window rule and NOT
    used as the judge: 2.5194 ms.  For orientation only, in the same device
    window: target_ms 2.2174, so this row stands 9.97% above target with M2
    and M3 still ahead.  MFU 35.6% (reference 39.15%).
    The bar's purpose per the milestone is to catch "structure did not take
    effect" (wrong axis order / empty masking window / spill blowout, all of
    which land above 3.5 ms).  Clearing it by 16% with the M0 architectural
    lines all still green is that check passing, not a claim of parity.

  Gate line (4) -- NCU analysis (advisory, diagnose only)
    Architectural lines re-read on the shipping sha and unchanged from the M0
    gate, i.e. M1 modified no topology: block_size 384, grid 32x16x10 / 5120,
    smem_dynamic 199680 B, smem_static 0, regs_per_thread 168, kernel_count 6,
    dominant_time_pct 86.25%, dominant_launches 2.  warp_active_pct 18.20%
    (reference 14.71%, advisory: occupancy is register-locked at 1 CTA/SM by
    D0.4, so this number is not a target).
    MAIN itself is 2.0832 ms against the reference's 1.8553 ms [MEASURED,
    run0 trace split] = +12.3%; the auxiliary kernels are 0.332 ms per call
    against the reference's 0.2317 ms = +43%, which is M3's D3.1/D3.2/D3.3
    territory and is recorded here only so the next phase does not have to
    re-derive the split.

  Closing action: the M1 decision list is complete and this row prices it.
  M2 is next in the phase order (D2.1 wait depth, D2.2 Q/dO stage count,
  D2.3 exchange-surface swizzle), and per the milestone it may only move
  those dials -- the M0 topology stays frozen.

  ==========================================================================
  M2 -- D2.1  wait-depth dials inside D0.8's frozen issue order
  ==========================================================================

  D2.1  segment-F1 wait depth: 1 -> 2.  ZERO topology movement -- smem
        199680 B, grid 32x16x10, 384 threads, tile (128,128), the D0.5 buffer
        list and access atoms, the D0.6 operand sources, the D0.7/D1.3
        precision contract and the D0.8 SEGMENT ORDER (which chain is issued
        where, which elementwise segment runs in which chain's shadow) are all
        untouched.  Only two wait numbers and the position of one mbarrier
        arrive move; no value in the kernel changes, which is why this dial
        had to be priced on device rather than reasoned about.

    The trip body has four graded WGMMA wait points plus the segment-G drain.
    D2.1's candidate domain maps onto them as follows, and every member was
    run round-robin in ONE process at the real shape 10,16,2,4096,64
    (9 timed rounds after 3 warm calls) by notes/probe_d21_depth.py:

      seg B {1, 2}    NOT REALIZABLE in the frozen order and not carried as a
                      rung: only two groups have ever been committed when
                      segment B's wait executes, so depth 2 retires nothing
                      and the softmax would read an unfinished `acc_s`.  A
                      third airborne group would have to come from a chain
                      issued earlier and none can be (chain 2's A operand is
                      what segment B produces, chain 3 needs the rendezvous,
                      chain 4 needs `rDS`); carrying one across the back edge
                      is blocked by segment G's pipeline-A release.
      seg C {1, 0}    the milestone's FALSIFICATION dial.  Depth 0 (`cdrain`,
                      = the pre-D0.8 bare form) measured 2.6062 ms against
                      base 2.5274 ms = +3.12% [MEASURED, probe rung `pricea`].
                      Chain 2's window is therefore NOT empty and the
                      report-back-to-meta branch does not apply; the dial
                      stays at 1.
      seg F1 {1, 0}   depth 0 (`f1drain`) = 2.5531 ms = +1.02% [MEASURED,
                      same round-robin].  Not taken.
      seg F2 {1, 0}   depth 0 (`f2drain`) = 2.6276 ms = +3.97% [MEASURED,
                      same round-robin].  Not taken.
      seg F1 deeper   depth 2 (`f1deep`) = 2.4990 ms against base 2.5263 ms
                      in its own process = -1.08% [MEASURED, probe rung
                      `priceb`].  TAKEN -- it is the one realizable "leave one
                      more in flight" position in the whole body.

    What shipping `f1deep` does, in two edits:

      (a) the F1 wait goes `warpgroup.wait_group(1)` -> `wait_group(2)`.  At
          depth 1 chain 2 retired there and chain 4 (dK += dS^T.Q) was issued
          into a one-chain shadow; at depth 2 nothing retires there, chains 2
          AND 3 are both airborne across chain 4's issue, and chain 2 retires
          one segment later at the F2 point.  F2's own depth is unchanged at
          1: three groups are outstanding when it executes, so it retires
          chains 2 and 3 together and still leaves chain 4 in flight across
          the segment-G staging handshake.
      (b) the pipeline-B consumer release `_mb_arrive(bEB)` moves down from
          the F1 site to immediately after the F2 wait.  It is only sound
          once chain 2 -- the last reader of this trip's dO face -- has
          actually retired, and under (a) it has not retired at F1.  It still
          runs one segment ahead of pipeline A's release at the drain.

    D2.1's cost line is "look at spill before you talk about gain": the dial
    lengthens `rP`'s live range by one segment, since a WGMMA may read its
    RMEM A operand any time before its group retires.  A SASS census of both
    cubins reads MAIN LDL+STL = 0 for base AND for f1deep [MEASURED, probe
    rung `dump`, via nvdisasm], so the longer live range bought no spill; the
    base zero is the known-true anchor that makes the f1deep zero mean
    something.  Correctness is unchanged by construction and was checked
    anyway: all five variants reproduce a float32 reference over the whole
    machine (PRE+MAIN+POST) at B,HQ,HKV,N,D = 2,4,2,640,64, grid (5,4,2) with
    trip counts 5,4,3,2,1, worst of the fifteen tensors 0.000815749 against
    the harness bar 0.047542 [MEASURED, probe rung `num`].

    D2.2 and D2.3 are closed by the entry below; the M2 gate is not written
    here.

  ==========================================================================
  M2 -- D2.2 Q/dO stage count, D2.3 exchange-surface swizzle mode
  ==========================================================================

  Co-committed because the milestone bundles them on ONE budget: both dials
  move bytes in the same D0.5 buffer list (stage count moves the byte total,
  swizzle mode moves the layout of those same bytes) and both are judged by
  the same M2 gate observation pair (the smem-traffic dual-branch line).
  Both ship at their starting points, so this row is ZERO code movement:
  smem 199680 B, grid 32x16x10, 384 threads, tile (128,128), NSTAGE = 2,
  both smem atoms K_SW128.  What the row delivers is the closure of the two
  dials, i.e. the evidence that neither alternative is takeable.

  D2.2  Q/dO stage count: candidate domain {2, 3}, ships at 2.

    The milestone's own cost line reads "+1 stage = +16 KiB sQ +16 KiB sdO
    = 32 KiB, 195 KiB -> 227 KiB, 顶到 228 KiB 上限".  That arithmetic is
    one field short of the buffer list this kernel actually allocates.  The
    frozen field list is measured at 199680 B [MEASURED, probe rung `fit`:
    the stage-2 control struct reads exactly 199680 B and a kernel
    allocating that many bytes launches on device], and a third Q/dO stage
    costs, on that list:

      sQ    (BM, HD, NSTAGE) f16   2 -> 3 stages          +16384 B
      sdO   (BM, HD, NSTAGE) f16   2 -> 3 stages          +16384 B
      sLSE  NSTAGE * BM f32        2 -> 3 slots           +  512 B
      sDel  NSTAGE * BM f32        2 -> 3 slots           +  512 B
      mbar  5*NSTAGE + 1 Int64     11 -> 16 (88 -> 128 B)      0 B
            (absorbed: the cell is padded to the 1024-B alignment of sK)

    The two row-scalar fields are the ones the milestone's number omits, and
    they are not optional: LSE rides pipeline A's publication and delta
    rides pipeline B's transaction (D0.9 barrier cell above), so a Q/dO
    stage without its own row-scalar slot would be overwritten by the
    producer while a consumer still owns the operand face.  Total at stage
    3 = 199680 + 32768 + 1024 = 233472 B [DERIVED from the frozen field
    list] against the cap the DSL enforces for sm_90, 227*1024 = 232448 B.
    Stage 3 is over the cap by 1024 B -- it is not a slow dial, it is an
    inadmissible one, and the DSL only reports it at launch
    (CUDA_ERROR_INVALID_VALUE).  Note how close the miss is: dropping the
    row scalars back to 2 slots would land on 232448 B, exactly the cap,
    which is why the milestone's estimate read as "几乎没有空间" rather than
    "over".  The dial stays at 2 and D2.2 is closed as excluded, which is
    what the milestone asked of it ("扫它是为了排除，不是为了指望").

  D2.3  exchange-surface swizzle mode: candidate domain {128B, 64B} x
        {sdS, sQ/sdO/sK/sV}, ships at 128B on both faces.

    Two atoms drive the entire exchange surface, so the domain is exactly
    four points and each is a single-token change (the milestone's cost line
    -- "一个 LayoutEnum 同时驱动四处 ... 改一半不改另一半会静默算错" -- is
    satisfied by construction here, because every downstream object is built
    from these two objects and nothing re-derives a swizzle independently):

      atom_hd -> sK, sV, sQ, sdO.  Reaches the TMA descriptors (built from
                 the same layout objects in `_launch`), the WGMMA
                 descriptors, and the transposed read views (which are
                 `composition`s of the swizzled layout with a pure domain
                 permutation, so they follow the swizzle for free).
      atom_bn -> sdS.  Reaches the transposed stmatrix write partition and
                 the chain-3/4 descriptor reads.

    Both currently resolve through `get_smem_layout_atom(ROW_MAJOR, F16, n)`,
    which returns K_SW128 for both extents (HD = 64 f16 = 1024 b, BN = 128
    f16 = 2048 b, both multiples of 1024 b).  The shipping value therefore
    got there by DEFAULT, which is why the milestone left it open: this is
    the one item reconnaissance did not measure (the reference's descriptor
    swizzle field was never decoded).

    What notes/probe_d22_d23.py establishes is ADMISSIBILITY of the whole
    domain, on device: the variants `ds64` (atom_bn -> K_SW64), `hd64`
    (atom_hd -> K_SW64) and `both64` all build, launch and reproduce a
    float32 reference over the whole machine (PRE+MAIN+POST) at
    B,HQ,HKV,N,D = 2,4,2,640,64, grid (5,4,2), trip counts 5,4,3,2,1, worst
    of the fifteen tensors 0.000815749 against the harness bar 0.047542
    [MEASURED, probe rungs `numa`/`numb`] -- the same figure base reports,
    i.e. a swizzle mode is a permutation of the same bytes and no variant is
    silently half-applied.  `hd64` was the one that could have been refused
    outright rather than merely slower (it changes the smem layout a
    bulk-tensor TMA descriptor is built from, and a 64 B swizzle under a
    128 B contiguous tile extent is where `make_tiled_tma_atom` can reject),
    and it was not refused.  Every variant also produced a real-shape median
    round-robin in one process [MEASURED, probe rungs `pricea`/`priceb`].

    None of the three is taken.  The 128B mode is the mode that matches the
    surface: sdS rows are BN = 128 f16 = 256 B and the operand faces are
    HD = 64 f16 = 128 B, so a 64 B swizzle permutes within half of the unit
    the descriptor and the stmatrix partition actually stride by, and there
    is no measured improvement in hand that would justify moving off it.
    D2.3's re-open trigger is written into the M2 gate, not into prose: gate
    line (2) is the conflict branch (`bankconf_st <= 6.8e6` and
    `bankconf_ld <= 1.0e5`).  If the gate lands over that band while the
    volume branch (gate line (1), `smem_wavefronts_st + _ld <= 1.1e8`) is
    clean, the diagnosis is conflict-type and the lever is this dial -- and
    it is then one token on `atom_bn` (sdS first, per the milestone's
    "逐块试" order) or on `atom_hd`, with the admissibility of both already
    certified above so that re-open costs a price run and not a bring-up.

    The M2 decision list (D2.1, D2.2, D2.3) is complete after this row; the
    M2 gate is next in the phase order and is submitted alone.

  ==========================================================================
  M2 GATE -- the M2 closing row, submitted ALONE (no decision bundled)
  ==========================================================================

  Submitted with NO executable change: the diff of this commit is confined to
  this docstring (the module AST outside it is byte-identical to e807651), so
  the reading this row's eval produces prices the delivered M2 set and nothing
  that arrived with the gate.

  Delivered M2 set, complete: D2.1 (wait depth, segment F1 dialled 1 -> 2),
  D2.2 (Q/dO stage count, closed at 2 -- stage 3 is inadmissible, not slow),
  D2.3 (exchange-surface swizzle, closed at 128B on both faces, whole domain
  certified admissible on device).  All three are transcribed above with their
  GPU certificates.  No M2 decision is deferred and none is outstanding.

  The three whole-machine evals this row reads, all on this lineage, all at
  the real shape (10,16,2,4096,64), device window; last line is the archived
  reference for the same counters [MEASURED, job 1787925084]:

    sha      iter  what                   device_ms  wf_st    wf_ld    bc_st    bc_ld
    b1718d0   45   M1 Gate (pre-M2 anchor)  2.4386  6.239e7  9.066e7  3.418e6  3.921e6
    28c2e59   46   D2.1 `f1deep`            2.4096  6.507e7  9.370e7  6.115e6  3.996e6
    e807651   47   D2.2+D2.3  <- SHIPPING   2.4124  6.508e7  9.368e7  6.122e6  3.994e6
    --        --   reference                2.2174  5.960e7  1.335e7  4.502e6  5.0e3

  Rows 46 and 47 are the SAME executable (D2.2+D2.3 was zero code movement),
  so that pair is a same-binary repeat and doubles as this lineage's
  repeatability estimate: device 0.12%, wf_st 0.01%, wf_ld 0.02%, bc_st 0.11%,
  bc_ld 0.05%.  Nothing below is inside that band.

  Gate line (1) -- mechanism, VOLUME branch   [VERIFY: ncu-derived]     OVER
    wf_st + wf_ld = 6.508e7 + 9.368e7 = 1.5876e8 against a <= 1.1e8 bar
    (reference 7.295e7, bar = x1.5).  1.443x the bar, 2.18x the reference.
    Side split, which the aggregate hides:
      write side  6.508e7 vs 5.960e7 = 1.09x the reference  -- in family
      read side   9.368e7 vs 1.335e7 = 7.02x the reference  -- the overshoot
    Read-side excess alone is 8.03e7 = 94% of the 8.58e7 by which the pair
    exceeds the reference.  The write side is not a contributor to line (1).

  Gate line (2) -- mechanism, CONFLICT branch [VERIFY: ncu-derived]    SPLIT
    bc_st 6.122e6 <= 6.8e6                                             PASS
      (1.36x the reference's 4.502e6, sitting at 90% of the bar).  Recorded
      movement with no layout movement: D2.1's wait-depth dial took bc_st
      3.418e6 -> 6.115e6 (x1.79) across iter 45 -> 46 while the swizzle atoms,
      the stmatrix partition and the smem field list all stayed put; the only
      thing that changed is when the two consumer warpgroups' sdS stores are
      in flight relative to each other.  Reported as an observation; no
      mechanism is asserted here.
    bc_ld 3.994e6 vs <= 1.0e5                                          OVER
      39.9x the bar, 799x the reference's 5.0e3.

  Branch determination (the dual-branch table, §5)
    The table's two rows are "volume over WITH bankconf clean" (lever: the
    D0.5 atoms + the D0.6 ledger) and "bankconf over WITH volume clean"
    (lever: D2.3).  This machine matches NEITHER row cleanly -- volume is over
    and the read half of the conflict pair is over -- so the determination is
    made by asking which OBJECTS carry the two counters.  Static enumeration
    of this file (not inference): MAIN issues LSU smem LOADS at exactly three
    sites, all of them the D0.5 part-4 row-scalar read atom --
    `cute.autovec_copy(sL2[None, qg // 2], rl)` in segment B's two mask arms
    and `cute.autovec_copy(sD2[None, qg // 2], rd)` in segment D.  Every other
    smem consumer in MAIN is a WGMMA descriptor read (sQ/sdO/sK/sV/sdS, which
    issue no LDS), and every other smem touch is a store (STS/stmatrix/bulk).
    The only other `autovec_copy` load in the file is POST's, and its source
    is GLOBAL.  The two objects those three sites read, `sLSE` and `sDel`, are
    declared as plain `cute.struct.MemRange[F32, NSTAGE * BM]` -- they carry
    NO swizzle atom.  D2.3's dial is exactly two atoms (`atom_hd` -> sK/sV/
    sQ/sdO, `atom_bn` -> sdS); it does not reach `sLSE`/`sDel` at all.
    Therefore neither the read-side volume nor the read-side conflicts are
    movable by the conflict-branch lever, and D2.3's own written re-open
    trigger ("line (2) over WHILE line (1) clean") is not fired by its terms.
    Both over-lines land on the same frozen objects: D0.5(a)/(d)'s row-scalar
    access atoms and D0.6's consumption ledger, frozen by the M0 gate.

  What sets the read-side volume, as a count [DERIVED, from this file plus
  the launch geometry -- no new measurement]
    Trips per MAIN launch: `for m in cutlass.range(n_blk, n_mblk)` with
    n_mblk = 4096/128 = 32 gives 32 - n_blk trip bodies per CTA, so
    sum(32-j, j=0..31) = 528 per (h_q, b) and 528 * 16 * 10 = 84480 per
    launch.  Per trip body each consumer thread executes two of the three
    sites (one segment-B mask arm + segment D), each NG_EW = size(acc_s) /
    GW_EW = 64 / 4 = 16 loads, so 32 LDS.64 per thread = 256 warp-level
    LDS.64 per trip body over the 8 consumer warps.  Total 2.163e7 LDS.64 per
    launch, whose floor at perfect coalescing (64 bit x 32 lanes = 256 B) is
    2 wavefronts each = 4.325e7.  D1.2 item 3 already halved this count (one
    load per 2x2 C block rather than per pair) and D0.5(d) halved it again
    before that (LDS.64 pair load rather than a scalar index per element);
    what remains is the count itself, i.e. that the row scalars are re-read
    out of smem inside the elementwise segments at all.  That is D0.6's
    `[SOURCE:]` ledger for lse/delta, frozen at M0 and not an M2 dial.
    Caveat on units: the digest does not document whether its counters are
    summed over `dominant_launches = 2`, so the measured 9.368e7 is either
    4.33x or 2.17x this derived floor; the side-split and reference ratios
    above are convention-free because both sides come from the same digest.
    An off-lineage prior on this same operator (a previous arm's best kernel,
    row-scalar smem reads deleted outright: 2.3597 -> 2.3418 ms) put the wall
    price of the entire read side at ~0.018 ms; it is recorded here as a
    prior, NOT as a certificate for this lineage, which has never measured it.

  Gate line (3) -- performance, end to end                             PASS
    device median 2.4124 ms against a <= 2.55 ms bar, 5.4% under it.  M2's
    ladder: 2.4386 (M1 gate anchor) -> 2.4096 (D2.1) -> 2.4124 (D2.2+D2.3,
    same executable).  M2's whole movement is -0.026 ms = -1.1%, inside the
    +-5% noise floor: both stage count and swizzle mode shipped at their
    starting points by measurement, so the wait-depth dial was the only
    executable change in the milestone.  Distance to `target_ms` 2.2174 is
    +8.8%, which is M3's bar, not this one.

  Gate line (4) -- NCU advisory signature, shipping sha
    tensor_pct 50.29 (>= 50)          spill_ld/st 0 / 0 (== 0)
    regs_per_thread 168 ([150,185])   smem_dynamic 199680 ([180224,204800])
    dram_bytes_read 4.039e8 (<=6.0e8) dram_bytes_write 1.928e8 (<=2.4e8)
    dram_pct 8.66 (<=25)              stall_barrier 2.24 (<=4.0)
    stall_long_sb 2.85 (<=3.4)        stall_mio 0.55 (advisory)
    warp_active_pct 18.21 (advisory)
    Every band in §5 except the two smem-traffic rows is met.  This also
    CLEARS the line the M1 gate carried forward: tensor_pct was 49.60 there
    against a >= 50 bar and reads 50.29 here, the move having come from
    D2.1's wait-depth dial (49.62 -> 50.27 across iter 45 -> 46) rather than
    from any M1 lever.

  Carried into M3: gate line (1) and the read half of gate line (2), both
  attributed above to the M0-frozen D0.5/D0.6 row-scalar contracts and
  therefore not reachable from an M2 dial.  M3's decision list (D3.1, D3.2,
  D3.3, D3.4) is next in the phase order.

  ==========================================================================
  M3 -- D3.1 PRE kernel access width
  ==========================================================================

  Shipped alone.  D3.1's candidate domain is {vectorisation width, rows per
  thread}; this row moves the first to the 128-bit ceiling on BOTH of PRE's
  streams and closes the second at 1.  The executable edit itself landed in
  the probe commit 82be561 (which carries no eval), so against the last sha
  this lineage evaluated -- e807651, the M2 gate -- this row's executable
  delta is exactly D3.1 and nothing else.

  What PRE was, per thread, before this row: `for d in range(HD)` indexing
  `mO[(bz, by, row, d)]` and `mDO[...]` one fp16 at a time, plus a zero-init
  loop storing one f32 at a time.  Thread `tidx` owned row `bx*BM + tidx`,
  so a warp's 32 lanes sat on 32 DIFFERENT rows 128 B apart and each scalar
  load asked the LSU for 32 distinct sectors to use 2 B of each.

  What it is now: the row is walked as `HD / PRE_VEC_H` = 8 groups of 8 fp16
  (16 B per instruction per lane) through `cute.autovec_copy` into a register
  fragment, and the zero-init as `PRE_VEC_F` = 4 f32 groups (16 B), also via
  `autovec_copy`.  Both sides address flat views built on the tensors' own
  iterators -- (B, H_Q, N, D) is contiguous, so the three leading modes
  collapse into one row index (read side) or one BM-tile index (write side).
  Grid `(N/BM, H_Q, B)` and block 128 are byte-identical to before, and the
  delta store is unchanged.

  Why rows per thread closes at 1 [DERIVED, from the launch geometry -- no
  new measurement].  The dial's purpose is bandwidth, and at width 128 b the
  DRAM side is already exact: lanes hold consecutive rows, so one warp's 32
  rows are 32 x 128 B = one contiguous 4 KiB run per stream, every byte of
  which is consumed.  What rows-per-thread would still buy is L1 wavefronts,
  not DRAM bytes: each 16 B lane request lands in its own 128 B line, so a
  load instruction issues 32 wavefronts where a lane-contiguous arrangement
  would issue 4.  Counting that cost for the real shape (10,16,2,4096,64):
  655360 rows / 32 = 20480 warps x 8 groups x 2 streams = 3.277e5 load
  instructions x 32 wavefronts = 1.049e7 wavefronts, which at ~4 per cycle
  per SM over 132 SMs at 1.755 GHz is ~11 us of LSU occupancy -- against a
  DRAM floor for the same launch of 167.8 MB (O) + 167.8 MB (dO) read plus
  167.8 MB (dq_accum zero) + 2.6 MB (delta) written = 506 MB, ~113 us at
  3 TB/s (the reference PRE measures 120.7 us [MEASURED, run0 trace], i.e.
  the reference itself sits at that roof).  The L1 term is ~10% of a figure
  it overlaps with, and buying it back costs a cross-lane reduction for
  `delta` (8 lanes per 128 B row) plus a rows-per-thread that no longer
  divides BM.  Closed at 1; re-open only if `aux_ms` stays over budget with
  the POST sub-budget met.

  GPU certificate [MEASURED, probe notes/probe_d31_pre_width.py, four PROBE
  lines, all PASS]:
    C1  PRE alone at the real shape (10,16,2,4096,64): `delta` against a
        chunked fp32 torch reference for `scale * rowsum(dO . O)`.  This is
        the read-side flat view and the gmem->reg autovec width.
    C2  same launch, dq_accum pre-filled with a 12345.0 sentinel: all
        4.194e7 f32 come back exactly 0.0, max_diff 0.0.  This is the
        write-side bijection, whose failure mode is silent (MAIN accumulates
        into whatever survived, so the symptom would be a wrong dq).
    C3  whole machine PRE+MAIN+POST at (2,4,2,640,64) against an fp32
        reference the probe computes itself: dq/dk/dv all inside the harness
        bar.
    C4  whole machine at the real shape: every kernel lowers, every launch
        completes, dq/dk/dv finite and non-zero (non-numeric, max_diff=n/a).
  Worst numeric residual over the three numeric contracts: 1.084e-3.

  What prices this row: D3.4's `aux_ms` = total_time_ns x (1 - dominant_time
  _pct/100) / dominant_launches / 1e6, budget 0.271 ms with sub-budgets fill
  0.016 / PRE 0.133 / POST 0.122.  The pre-D3.1 reading was 0.332 ms
  [MEASURED, iter 46 digest: 4779392 ns x (1 - 0.861) / 2], 1.35x budget and
  0.061 ms over; PRE's share of that overshoot is what this row moves.
  D3.2, D3.3 and D3.4 are not written here and are next in the phase order.

  ==========================================================================
  M3 -- D3.2 POST kernel bandwidth
  ==========================================================================

  Shipped alone.  D3.2's declared candidate domain is {3 launches separate,
  dK/dV merged into 1 launch}; this row takes the merge AND, on the same
  budget, the access form of the POST body itself, because the merge alone
  reaches only the 27.3 us the two small POSTs cost in the reference
  [MEASURED, run0 trace] while POST(dQ) is 75% of the POST sub-budget.

  What POST was: read the f32 workspace through `tv_st` (already a 128-bit
  universal-copy atom, i.e. the read side was at the ceiling), then write the
  fp16 output straight out of the C fragment, `mOut[(bz, by, base+r, c)] =
  F16(acc_back[i])`.  In the WGMMA C bijection one instruction's 32 lanes sit
  on 8 different output rows at 2 B per lane per row, so the write side of a
  pure streaming kernel was issued at 2 B granularity: a warp-wide store
  touched 8 rows x 8 B rather than one contiguous run.

  What it is now, in `_post_body` (factored out of the kernel entry so the
  merged entry can call it twice):
    - the C fragment is scattered into an smem tile `sT` of shape (BM, HD)
      f16 with row stride HD + POSTV_PAD = 72 f16 = 144 B.  The pad is what
      keeps that scatter conflict-free: the 32 lanes of one C instruction sit
      on 8 rows x 4 columns, and with a 72-f16 row stride their banks come
      out 4*(l/4) + l%4 + 4k, all 32 distinct.
    - one `cute.arch.barrier()`, then the tile is drained as POSTV_VEC = 8
      f16 (16 B) per thread over POST_NCH / NUM_THREADS = 4 passes, chunk
      index `i * NUM_THREADS + tidx`.  Consecutive lanes hold consecutive
      16 B chunks, so a warp writes 512 contiguous bytes and the write side
      issues 8x fewer wavefronts than the scatter did.  The read side and the
      workspace bijection are byte-identical to before.
  Cost: one extra smem round trip per tile, and BM * (HD + POSTV_PAD) * 2 =
  18432 B of smem per POST block (POST is its own launch and shares nothing
  with MAIN's 199680 B, so the M0 smem contract is untouched).

  The launch dial: `_post_kernel2` does dK and dV back to back in one block.
  They are the only two of the three POSTs that share a grid (H_KV, not H_Q)
  and a tile count.  The two bodies share the single staging tile, hence the
  barrier between them -- the second scatter must not overtake the first
  drain.  Three POST launches become two; grids and block (NUM_THREADS = 256,
  which the TV thread layout 128 x NUM_WG fixes) are otherwise unchanged.

  GPU certificate [MEASURED, probe notes/probe_d32_post_aux.py, five PROBE
  lines, all PASS]:
    C1  whole machine PRE+MAIN+POST at (2,4,2,640,64) against an fp32
        reference the probe computes itself, for SIX variants of the POST
        access form / merge / PRE row-tiling cross product -- including the
        `postboth` variant this row ships (vectorised POST + merged dK/dV).
        dq/dk/dv all inside the harness bar; worst residual over the 18
        tensors 8.157e-4.  This is the contract that matters here: both the
        smem re-route of the OUTPUT bijection and the merged entry are
        exactly the class of change that is silently wrong.
    C2  launch ablation of the SHIPPED kernel at the real shape
        (10,16,2,4096,64) -- s0 = the host-side dk/dv fills only, s1 = + PRE,
        s2 = + MAIN, base = + the POSTs -- all four produced a device-event
        median and the account is monotone (non-numeric, max_diff=n/a).
    C3  the same ablation on the candidate (non-numeric, max_diff=n/a).
    C4  all four POST variants lower, launch and produce a real-shape median
        (non-numeric, max_diff=n/a).
    C5  base and the two PRE-side variants likewise (non-numeric,
        max_diff=n/a).

  What prices this row: D3.4's `aux_ms`, budget 0.271 ms, POST sub-budget
  0.122 ms.  The pre-D3.2 reading was 0.346 ms [MEASURED, iter 49 digest:
  4802656 ns x (1 - 0.8559) / 2], 1.28x budget.  D3.1's rows-per-thread dial
  stays closed at 1 this round (its re-open trigger is live, but pricing it
  in the same commit as this row would leave `aux_ms` -- a TOTAL, not a split
  -- unable to attribute either).  D3.3 and D3.4 are next in the phase order.

  ==========================================================================
  M3 -- D3.4 per-launch account, and the D3.1 re-open it authorises
  ==========================================================================

  Shipped alone.  D3.4 is M3's PRICING channel, not a code decision: its
  deliverable is the account that tells D3.1 and D3.2 which of them is fat.
  Running it turned out to close a decision, because the account fired the
  re-open condition D3.1 wrote for itself.

  The instrument.  `aux_ms` is a TOTAL -- it says the auxiliary segment is
  over budget, never which launch is over.  D3.4's text authorises exactly
  one way to split it ("同 job launch 消融，成本一整轮"): a cumulative launch
  ablation of the SHIPPED kernel at the real shape, differences of adjacent
  rungs being the per-launch cost.  Rungs s0 = the host-side dk/dv fill only,
  s1 = + PRE, s2 = + MAIN, base = + the two POSTs.

  The account [MEASURED, probe notes/probe_d34_split.py rung `split`, device-
  event median, real shape (10,16,2,4096,64), 9 rounds after 3 warm calls]:

    rung   cumulative      delta = that launch      sub-budget    verdict
    s0        0.0929 ms       fill  0.0929 ms          0.016        OVER
    s1        0.2725 ms       PRE   0.1796 ms          0.133        OVER
    s2        2.3239 ms       MAIN  2.0513 ms          --           --
    base      2.4308 ms       POST  0.1069 ms          0.122        OK
                              aux = 0.3795 ms          0.271        OVER

  This is a per-call e2e window (it carries host overhead the digest's device
  window does not, which is why aux reads 0.3795 here against 0.2943 from the
  iter 51 digest); the RANKING is what it is for, and it is unambiguous.
  POST, after D3.2, is inside its sub-budget.  PRE is 0.047 ms over its own.
  That is exactly the condition D3.1 wrote down for itself -- "re-open only
  if `aux_ms` stays over budget with the POST sub-budget met" -- so the
  rows-per-thread dial D3.1 closed at 1 re-opens here on its own terms.
  (The fill row is over its 0.016 sub-budget too, but it is 2 x 20 MiB of
  `torch.zeros` already merged into one launch by D3.3; the remaining 0.077 ms
  is a launch this file does not write.  PRE is the actionable one.)

  What the re-open changes.  D3.1's close was [DERIVED], and it named its own
  weak point: at 1 row per thread a warp's 32 lanes sit on 32 rows 128 B
  apart, so each 16 B lane request lands in its own 128 B line and one load
  instruction issues 32 L1 wavefronts where a lane-contiguous arrangement
  issues 4.  D3.1 estimated that term at ~11 us against a ~113 us DRAM floor
  and judged it not worth a cross-lane reduction.  Measured, it is worth far
  more than the estimate: the mapping alone takes PRE from 0.1985 to 0.1266 ms
  [MEASURED, rung `pre`], -36%, i.e. -0.072 ms where the estimate said -0.011.
  The estimate's error was to treat the L1 term as overlapping the DRAM term;
  at this arithmetic intensity (PRE does one FFMA per 4 B read) it does not.

  What ships: a thread now owns 16 B chunk `i * PRE_THREADS + tidx` of the
  flattened BM x HD tile on BOTH streams, so consecutive lanes hold
  consecutive chunks and one warp covers 512 contiguous bytes per
  instruction.  The zero-init side was already written that way, so its only
  change is the new PRE_THREADS.  The read side moves off one-row-per-thread,
  which spreads a row over PRE_CH_ROW = 8 lanes and so needs a cross-lane sum
  for `delta`: three butterfly shuffles, then lane `c % 8 == 0` stores the row
  scalar.  The shuffles run OUTSIDE the store predicate -- a shuffle's member
  mask is the whole warp, so every lane must reach every step; predicating
  them is the way this construction goes silently wrong.  Block goes 128 ->
  256 threads (PRE_THREADS = 2 * WG_THREADS); grid `(N/BM, H_Q, B)`, the
  128-bit widths, the delta store's value and the D1.3 scale placement are
  unchanged.  MAIN's topology is untouched -- PRE is its own launch and shares
  no smem, register or barrier budget with it.

  Why 256 and not 512.  The dial was measured at all three [MEASURED, rung
  `pre`]: 128 -> 0.1301, 256 -> 0.1266, 512 -> 0.1240 ms.  512 is 2% faster
  than 256, inside the +-5% noise floor, and it is the one width whose whole-
  machine numeric contract (C3) and real-shape smoke (C6) were NOT run -- both
  ran at 128 and 256.  Taking a 2% in-noise gain on an uncertified width
  against a certified one is not a trade this file makes; 512 stays available
  as a one-constant move if a later row wants to certify it.

  Divisibility, which is what makes the mapping legal: PRE_NCH = 1024 chunks
  per tile / PRE_THREADS = 256 gives 4 read passes; (BM*HD) / (PRE_VEC_F *
  PRE_THREADS) gives 8 zero-init passes; PRE_THREADS % PRE_CH_ROW == 0 and
  PRE_CH_ROW = 8 divides the warp, so a row's 8 chunks always sit in 8
  consecutive lanes of ONE warp and never straddle a warp boundary.  Per
  pass the block covers PRE_THREADS / PRE_CH_ROW = 32 rows, x 4 passes = 128
  = BM rows exactly, so every row of the tile is stored exactly once.

  GPU certificate [MEASURED, probe notes/probe_d34_split.py, six PROBE lines,
  all PASS]:
    C1  PRE alone at the real shape, `delta` from every full variant against
        a chunked fp32 torch reference for `scale * rowsum(dO . O)`; worst
        4.768e-7, and the shipped mapping's own reading 1.788e-7.  This is
        the contract that catches a mis-grouped shuffle or a wrong row index
        -- the two ways the butterfly can be wrong.
    C2  same launches, dq_accum pre-filled with a 12345.0 sentinel: back at
        exactly 0.0, max_diff 0.0.  The zero-init's failure mode is silent
        (MAIN accumulates into whatever survived; only dq is wrong).
    C3  whole machine PRE+MAIN+POST with the candidate PRE at (2,4,2,640,64)
        against an fp32 reference the probe computes itself, at 128 and 256
        threads: dq/dk/dv inside the harness bar, worst 1.084e-3 vs 6.102e-2.
    C4  the launch ablation above: four cumulative rungs, all timed, account
        monotone (non-numeric, max_diff=n/a).
    C5  all eight PRE variants lowered, launched and timed (non-numeric).
    C6  real-shape smoke of the candidate whole machine: every kernel lowers,
        every launch completes, dq/dk/dv finite and non-zero (non-numeric).

  Expected effect on the M3 gate line, stated so it can be checked against
  the eval rather than assumed: PRE 0.1796 -> ~0.127 ms moves the aux total
  by ~0.053 ms.  That clears PRE's own sub-budget (0.133) but does not by
  itself bring `aux_ms` under 0.271 -- the fill row's 0.077 ms overshoot is
  outside this file's PRE/POST dials.  D3.1 and D3.2 are now both inside
  their sub-budgets, which is the disposition D3.4 exists to establish.

  ------------------------------------------------------------------------
  M3 -- D3.3 workspace allocation, JIT cache and host overhead
  ------------------------------------------------------------------------
  Shipped alone.  D3.3's declared candidate domain is {allocate per call,
  reuse across calls} and its judging metric is the device median.  Three of
  its four subjects were already in the lineage's host side and are only
  restated here; the fourth is the one executable move left, and it is the
  reason this row is not a no-op:

  (1) dq_accum stays `torch.empty`, zeroed on device by PRE.  The host
      `zeros` retreat this item exists to forbid was never taken, so there
      is nothing to change back.
  (2) dk/dv_accum stay `torch.zeros`, but they are now two leading slices of
      ONE contiguous (2, b, hkv, n, d) f32 buffer (`ws["pair"]`), and the
      per-call reset that their cross-CTA accumulation requires is that
      buffer's single `zero_()`.  Bytes are identical (2 x 20 MiB at the
      real shape); the launch count for the fill drops from 2 to 1, which is
      the shape of the account D3.4 keeps -- its reference budgets ONE fill
      at 0.016 ms.  This is the only device-visible edit in the row.
  (3) the compiled artifact was already cached in `_COMPILED` and the
      buffers in `_WORKSPACE`, both keyed by shape; bench timing repeats the
      call, so neither the JIT nor the 160 MiB of workspace may be rebuilt
      per call.  Added on the host side: the four workspace `from_dlpack`
      descriptors are built once alongside the buffers they address (that
      storage never moves) and the CUstream wrapper is memoised on the raw
      stream handle, so a call now constructs 9 descriptors instead of 13
      and no driver object at all.  Keying the stream on the raw pointer
      rather than caching it outright keeps a caller that switches streams
      correct.

  What this cannot do: (3) is host work by construction and cannot move the
  device median -- it is claimed against the host clock only.  (2) is the
  only part with a device-side claim.

  Verified before landing by notes/probe_d33_ws.py, which ran the arms of
  the dial as drivers over ONE loaded module and ONE compiled artifact, so
  the device code under them is byte-identical:

    C1  numeric: the shipped form and this candidate both reproduce a f32
        reference at (2,4,2,640,64) on call 1 AND on call 2 (the call that
        catches a reset which misses one of the two dk/dv views -- silently
        right once, doubled after), worst of 12 tensors, max_diff=0.000815749
        against the harness bar 2e-2 * |ref|_max.
    C2  non-numeric: the cumulative launch ablation of the shipped `_launch`
        at the real shape (10,16,2,4096,64) -- fills only, + PRE, + MAIN,
        + the POSTs -- produced four device-event medians and a monotone
        account (max_diff=n/a).
    C3  non-numeric: all six drivers (shipped, workspace-per-call,
        host-zeros-on-dq, one-fill, cached-descriptors, and the two
        combined) produced a real-shape median on both clocks
        (max_diff=n/a).

  What prices this row: the device median (D3.3's own judging metric) and
  D3.4's `aux_ms`, budget 0.271 ms with sub-budgets fill 0.016 / PRE 0.133 /
  POST 0.122.  The pre-D3.3 reading was 0.2949 ms [MEASURED, iter 50 digest:
  4699680 ns x (1 - 0.8745) / 2], 1.09x budget.  D3.4 and the M3 Gate are
  next in the phase order.
"""

import inspect
import math
from types import SimpleNamespace
from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

try:  # MLIR type helpers (used only by the inline-PTX wrappers below)
    from cutlass._mlir.extras import types as _T
except Exception:  # pragma: no cover - layout of the private module varies
    from cutlass._mlir import extras as _extras
    _T = _extras.types


# --- D0.6: the kwarg that selects the WGMMA A-operand source on this toolchain.
#     Resolved by introspection (spelling varies across CUTLASS DSL releases);
#     notes/probe_d06_rs_operand.py certified the resulting RS mma on GPU.
_A_SRC_KW = None
for _cand in ("a_src", "a_source", "a_operand_source", "operand_source", "a_kind"):
    if _cand in inspect.signature(sm90_utils.make_trivial_tiled_mma).parameters:
        _A_SRC_KW = _cand
        break


# --- D0.5 TMA fill: the mbarrier entry points, resolved by introspection for
#     the same reason as `_A_SRC_KW` (the spelling moves across DSL releases).
#     Every one of them is certified on GPU by notes/probe_d05_tma_fill.py.
def _arch_op(*cands):
    for _n in cands:
        _f = getattr(cute.arch, _n, None)
        if _f is not None:
            return _f
    return None


# D0.4 `setmaxnreg`: DSL 4.4.2 spells it `setmaxregister_{decrease,increase}`;
# the older `warpgroup_reg_{dealloc,alloc}` forward to the same nvvm op.
_REG_DEC = _arch_op("setmaxregister_decrease", "warpgroup_reg_dealloc")
_REG_INC = _arch_op("setmaxregister_increase", "warpgroup_reg_alloc")

# D1.1 r1: the marker that keeps an integer chain on the SM90 uniform data
# path (`UR*`), where it does not draw on the per-thread budget `setmaxnreg`
# hands each role.  It is an ASSERTION, not a conversion (hw_kb/cutedsl/sm90/
# wgmma_desc_uniform.md gotcha 1): only values that really are equal across
# the 32 lanes of a warp may be marked.  Everything marked below is derived
# from `block_idx`, the trip counter, or compile-time constants -- never from
# `tidx` or a lane index.
_MWU = cute.arch.make_warp_uniform

_MB_INIT = _arch_op("mbarrier_init")
_MB_INIT_FENCE = _arch_op("mbarrier_init_fence", "fence_barrier_init")
_MB_TX = _arch_op("mbarrier_arrive_and_expect_tx")
_MB_WAIT = _arch_op("mbarrier_wait")


def _mb_fence():
    if _MB_INIT_FENCE is not None:
        _MB_INIT_FENCE()


# --- D0.8 segment D: one chunk of the interleaved sdS write. ------------------
def _rank_of(t):
    """Static rank of a partitioned tensor.  Plain Python helper (never a
    traced body), so the try/except cannot upset the DSL preprocessor."""
    try:
        return cute.rank(t)
    except Exception:
        return len(t.shape)


def _copy_chunk(tiled, atom, src, dst):
    """Emit ONE chunk of segment D's pack-a-batch / write-a-batch sdS store.

    Same reason as `_arch_op` for being a plain Python helper: the choice is
    made once, at trace time, and a try/except inside a traced body would upset
    the DSL preprocessor.  notes/probe_d08_order.py ran BOTH spellings of the
    chunked write against this exact face and payload (rungs `tiled` and
    `atom`); driving the TiledCopy is the preferred one, and the raw StMatrix
    atom is the fallback for the DSL builds whose `cute.copy` refuses a
    TiledCopy on a rank-1 chunk slice.  `cute.copy` validates shapes before it
    emits anything, so a rejected spelling leaves no partial IR behind.
    """
    try:
        cute.copy(tiled, src, dst)
    except Exception:
        cute.copy(atom, src, dst)


def _tma_atom_4d(mT, smem_l, tiler, n, d, h, b):
    """G->S bulk-tensor atom over a rank-4 (N, D, H, B) re-view of a (B,H,N,D)
    tensor: the descriptor's leading two modes are the tile's, so one rank-2
    tiler and a (n_blk, 0, h, b) coordinate address any tile."""
    t4 = cute.make_tensor(
        mT.iterator,
        cute.make_layout((n, d, h, b), stride=(d, 1, n * d, h * n * d)))
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), t4, smem_l, tiler)


# ---------------------------------------------------------------- D0.3 tiles --
BM = 128            # Q rows per tile
BN = 128            # KV rows per tile (one work item owns one of these)
HD = 64             # head dim = bK, contracted in one shot
NUM_WG = 2          # consumer warpgroups (WGMMA atom M-mode is 64)
NUM_THREADS = NUM_WG * 128
WG_THREADS = 128

# --- D0.4 warp roles / thread count / register split ---------------------------
# MAIN runs 3 warpgroups = 384 threads: warpgroup 0 is the producer WG (its warp
# 0 issues EVERY TMA / bulk copy / bulk reduce-add; its warps 1-3 carry only the
# barrier skeleton), warpgroups 1-2 are the two consumer WGs the tiled MMAs are
# built over.  `setmaxnreg` moves the CTA register pool across that fork:
#   128 * 24 + 256 * 240 = 64512 <= 65536.
# The profiler still reports the CTA mean 65536/384 = 168 -- per
# hw_kb/cutedsl/sm90/warp_specialization.md that number cannot distinguish the split
# from "no split at all"; the machine-readable proof is spill == 0.
PRODUCER_WARPS = 4                          # warps of the producer warpgroup
MAIN_THREADS = (NUM_WG + 1) * WG_THREADS    # 384
PRODUCER_REGS = 24
CONSUMER_REGS = 240

# D0.5 buffer list: Q/dO, sdS and the row scalars are 2 stage; K/V single.
# D2.2 closed here: candidate domain {2, 3}, ships at 2.  Stage 3 is not a
# slower dial but an inadmissible one -- the field list below totals 233472 B
# at NSTAGE = 3 against the 232448 B sm_90 cap the DSL enforces (the two
# row-scalar fields, which ride the two pipelines' publications and so must
# grow with the stage count, are what push it over).  See the ledger.
NSTAGE = 2
ROW_BYTES = BM * 4  # one row-scalar vector = BM f32 = 512 B

# --- D3.1 PRE access width and lane mapping -----------------------------------
# PRE is a pure stream (read O + dO, write delta + the dq_accum zero-init), so
# its dials are how many bytes one thread touches per instruction and how those
# threads are laid over the tile.  Both sides run at the 128-bit ceiling: 8 fp16
# on the read side, 4 f32 on the write side.
#
# The lane mapping is LANE-CONTIGUOUS on both sides: a thread owns 16 B chunk
# `i * PRE_THREADS + tidx` of the flattened BM x HD tile, so consecutive lanes
# hold consecutive chunks and one warp covers 512 contiguous bytes per
# instruction.  The earlier one-row-per-thread mapping put a warp's 32 lanes on
# 32 rows 128 B apart, which is DRAM-exact but issues 32 L1 wavefronts per
# instruction where this issues 4.  A row is now spread over PRE_CH_ROW = 8
# lanes, so `delta` takes a cross-lane sum (see `_pre_kernel`).
PRE_THREADS = 2 * WG_THREADS        # 256 threads per BM tile
PRE_VEC_H = 8                       # fp16 per read  vector = 16 B
PRE_VEC_F = 4                       # f32  per write vector = 16 B
PRE_CH_ROW = HD // PRE_VEC_H        # 8 read chunks per output row
PRE_NCH = (BM * HD) // PRE_VEC_H    # 1024 read chunks per BM tile

# `shuffle_sync_bfly` is the XOR-mode shuffle; the fallback spells the same
# butterfly with the general index form.  Resolved at trace time, not run time.
_HAS_BFLY = hasattr(cute.arch, "shuffle_sync_bfly")

# --- D3.2 POST access form ---------------------------------------------------
# POST's READ side was already at the 128-bit ceiling (it inverts through
# `tv_st`, whose atom is a 128-bit universal copy).  Its WRITE side was not:
# the C-fragment bijection puts one instruction's 32 lanes on 8 different
# output rows at 2 B per lane per row, so a warp-wide fp16 store touched 8
# rows x 8 B.  The tile now leaves through smem instead: the fragment is
# scattered into a (BM, HD) f16 tile whose row stride is padded to
# `HD + POSTV_PAD` f16, then drained as `POSTV_VEC` f16 (16 B) per thread with
# a lane-contiguous chunk index, so one warp writes 512 contiguous bytes.
POSTV_PAD = 8                        # f16 of row padding in the POST smem tile
POSTV_VEC = 8                        # f16 per gmem store = 16 B
POST_ROWCH = HD // POSTV_VEC         # 16 B chunks per output row
POST_NCH = (BM * HD) // POSTV_VEC    # 16 B chunks in one tile

# --- D0.9 barrier cell: index = slot, one Int64 (8 B) each -------------------
# The first EIGHT reproduce the pinned slot layout of the two 2-stage operand
# pipelines, `base + stage*8` with `{+0x00 full_A, +0x10 empty_A, +0x20 full_B,
# +0x30 empty_B}`:
#     A = Q + LSE      B = dO + delta
# The three after them are not part of that footprint and are named separately:
# the K/V fill barrier (per work item, not per trip -- K/V is single buffered
# and CTA-lifetime per work item), and the two PRIVATE landing barriers the
# LSE vector needs.  LSE cannot ride `full_A`'s transaction directly because
# D0.7 item 3 puts the `log2e` row multiply inside MAIN: the producer has to
# see the vector land, scale it, and only THEN publish stage A -- so the bulk
# copy completes on `B_RL` and the publication is the producer's own arrive on
# `B_FA` (see the arrive-count table in the ledger).
B_FA = 0            # full_A[stage]   Q face landed AND lse scaled
B_EA = 2            # empty_A[stage]  consumers done with Q + lse
B_FB = 4            # full_B[stage]   dO face + delta vector landed
B_EB = 6            # empty_B[stage]  consumers done with dO + delta
B_KV = 8            # K/V fill, one per work item
B_RL = 9            # lse vector landed (producer private), one per stage
NMBAR = 11

# D0.5 staging face: flat TV slots, one per warpgroup (see _make_flat_staging).
SLOT = BM * HD // NUM_WG        # f32 per warpgroup slot
STAGE_ELEMS = BM * HD           # f32 in the whole staging region
STAGE_BYTES = STAGE_ELEMS * 4
SLOT_BYTES = SLOT * 4           # one warpgroup's staging half = 16 KiB

# --- D0.9 part 2 named barriers ----------------------------------------------
# Hardware named barriers 0..15.  Id 0 belongs to `cute.arch.barrier()` (the
# whole-CTA rendezvous), so these start at 1.
#   NB_DS            class 3: the kernel's ONE symmetric rendezvous, the dS
#                    exchange.  Consumer threads only, so 256 participants --
#                    the producer warpgroup does NOT take part any more.
#   NB_FULL + wg     class 2, "report full": the consumer WG arrives (128),
#                    the producer warp blocks (32).
#   NB_EMPTY + wg    class 2, "hand the buffer back": the producer warp
#                    arrives (32), the consumer WG blocks (128).
# Both class-2 directions therefore have STAGE_CREDIT = 160 = 128 + 32
# participants, and each consumer WG owns its own pair -- the two WGs write
# disjoint 16 KiB halves of the staging face, so they need no common point.
NB_DS = 1
NB_FULL = 2                     # -> 2, 3
NB_EMPTY = 4                    # -> 4, 5
STAGE_CREDIT = WG_THREADS + 32  # 160 = one consumer WG + the producer's warp 0

LOG2E = 1.4426950408889634

F16 = cutlass.Float16
F32 = cutlass.Float32
I32 = cutlass.Int32


# ------------------------------------------------------- inline PTX wrappers --
@dsl_user_op
def _bulk_reduce_add_f32(dst_addr, src_addr, nbytes, *, loc=None, ip=None):
    """smem -> gmem block ADD, one instruction per CTA.  Issued by ONE thread."""
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(dst_addr, src_addr, nbytes, bar_addr, *, loc=None, ip=None):
    """D0.5 row-scalar WRITE atom: non-tensor 1-D `cp.async.bulk` G->S
    completing on an mbarrier (`UBLKCP.S.G`).

    The DSL wraps only the bulk-TENSOR direction, so this is inline PTX -- and
    unlike the DSL's TMA copy atom it does NOT elect internally, so the caller
    must wrap it in `elect_one` (single-thread instruction, like the S2G reduce
    above).  Both addresses must be 16 B aligned and `nbytes` a multiple of 16;
    a row-scalar vector is BM f32 = 512 B.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
         cutlass.Int32(bar_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_commit_group(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.commit_group;", "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _bulk_wait_group_read(n, *, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.wait_group.read {};".format(int(n)), "",
                    has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _nb_sync(bar_id, count, *, loc=None, ip=None):
    """D0.9 class 2/3: `bar.sync id, count` -- a BLOCKING arrive on a hardware
    named barrier.

    `bar.sync` is `barrier.sync.aligned`; the barrier id and the participant
    count are both register operands, which is what lets the id carry a runtime
    `wg_idx`.  The count must be a multiple of the warp size and must equal the
    number of threads that will ever arrive on that id between two completions;
    too small releases early (silent wrong data), too big deadlocks -- neither
    has a line number, which is why this mechanism goes up behind a probe.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(bar_id).ir_value(loc=loc, ip=ip),
         cutlass.Int32(count).ir_value(loc=loc, ip=ip)],
        "bar.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nb_arrive(bar_id, count, *, loc=None, ip=None):
    """D0.9 class 2: `bar.arrive id, count` -- a NON-blocking arrive on a named
    barrier.

    This is the asymmetric half.  The arriving side contributes its counts and
    walks on; the barrier completes only when the blocking side's `bar.sync`
    brings the total to `count`.  That is what makes one named barrier a
    one-way credit instead of a rendezvous.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(bar_id).ir_value(loc=loc, ip=ip),
         cutlass.Int32(count).ir_value(loc=loc, ip=ip)],
        "bar.arrive $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mb_arrive(bar_addr, *, loc=None, ip=None):
    """D0.9: a plain `mbarrier.arrive` on the barrier at this shared address.

    The DSL exports `mbarrier_arrive_and_expect_tx` (the producer's transaction
    form, already wired as `_MB_TX`) but the spelling of the bare arrive moves
    across releases, so it is inline PTX here for the same reason `_bulk_g2s`
    is.  Every thread that calls it contributes ONE count; `mbarrier.arrive`
    carries `.release.cta` semantics on sm_90, which is what publishes this
    thread's prior shared-memory stores to whoever waits on the barrier.
    The returned state token is unused.
    """
    return cutlass.Int64(llvm.inline_asm(
        _T.i64(),
        [cutlass.Int32(bar_addr).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 $0, [$1];",
        "=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    ))


@dsl_user_op
def _cvt_f16x2_f32(a, b, *, loc=None, ip=None):
    """D1.2 item 2: ONE `cvt.rn.f16x2.f32` (one SASS `F2FP`) per adjacent
    element pair, replacing the two scalar `F16(x)` conversions.

    Gotcha (silent and total when got wrong): `cvt.rn.f16x2.f32 d, a, b` puts
    the FIRST source in the HIGH half and the second in the LOW half.  The
    register pair this feeds is little-endian, so the element with the LOWER
    flat index must land in the LOW half -- hence the operands are spelled
    `$0, $2, $1`, i.e. argument `a` (the lower index) goes to `$1` = the low
    half.  Written `$0, $1, $2` every packed pair is swapped and every
    downstream GEMM is wrong with no diagnostic anywhere.
    """
    return cutlass.Int32(llvm.inline_asm(
        _T.i32(),
        [F32(a).ir_value(loc=loc, ip=ip), F32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    ))


def _flat_size(s):
    """Static extent of a (possibly nested) shape entry.  Plain Python helper
    (never a traced body), so the try/except cannot upset the DSL
    preprocessor."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return int(cute.size(s))


def _as_i32(flat):
    """A 32-bit view of a compact rank-1 f16 fragment view: element i of the
    result covers f16 elements 2i and 2i+1, which is exactly the pairing
    `_cvt_f16x2_f32` produces.  Plain Python helper: the two spellings of the
    recast are chosen once, at trace time."""
    try:
        return cute.recast_tensor(flat, I32)
    except Exception:
        return cute.make_tensor(cute.recast_ptr(flat.iterator, dtype=I32),
                                cute.make_layout(cute.size(flat) // 2))


def _ew_group(acc_shape):
    """D1.2 item 3: how many accumulator elements share ONE row-scalar pair.

    Mode 0 of a WGMMA C fragment is `(2, 2, V)`: the leading 2 is the pair of
    ADJACENT N columns (the two q rows one LDS.64 brings in), the second 2 is
    the M pair (two kv rows 8 apart, SAME q), V the N groups.  The fragment is
    compact with mode 0 innermost, so elements `[4g, 4g+4)` are one 2x2 block
    over one column pair -- one row-scalar load serves four elements, not two.
    Returning a derived (not assumed) width, with the adjacency the packing
    relies on checked here so a layout that broke it is a trace-time error
    rather than silently wrong numbers.
    """
    m0 = acc_shape[0]
    n_inner = _flat_size(m0[0])
    if n_inner != 2:
        raise ValueError(
            "D1.2: C-fragment mode 0 leading extent is {}, not the adjacent-N "
            "pair the packed convert and the row-scalar hoist assume"
            .format(n_inner))
    return n_inner * _flat_size(m0[1])


def _chunk_groups(nv, gw):
    """Row-scalar groups per segment-D chunk.  Plain Python helper, so a chunk
    width that split a group raises at trace time instead of turning into a
    predicate (a split group would re-read the row scalar for half the chunk --
    correct, but back at the un-hoisted count)."""
    if nv % gw:
        raise ValueError(
            "D0.8 segment D: chunk width {} is not a multiple of the "
            "row-scalar group {}".format(nv, gw))
    return nv // gw


def _bar(bar_t, i):
    """The `cute.Pointer` of barrier `i` in the D0.9 barrier cell.

    `i` may be a runtime value (`stage` is dynamic), and a runtime-sliced
    barrier sub-range's `.iterator` IS a `cute.Pointer` -- the route
    hw_kb/cutedsl/sm90/pipeline_runtime_subring.md certifies for exactly this problem.
    Plain Python helper (never a traced body) and everything is passed
    explicitly, so nothing closes over a runtime value (gotcha 4 there).
    """
    return bar_t[(None, i)].iterator


# --------------------------------------------------- D0.5 staging access atom --
def _make_flat_staging():
    """The ONE object pair that both write side and read side must share.

    Flat `(SLOT, NUM_WG)` f32 slots (linear stride, no swizzle, no quad
    interleave) + a flat TV copy giving每 thread 4 contiguous f32 = one
    128-bit STS, warp-wide across 32 banks.  Returned as a pair so MAIN's
    flush sites and POST's read side cannot drift apart: byte order in the
    f32 workspace IS this TV's order, so any second spelling silently
    scrambles.
    """
    flat_l = cute.make_layout((SLOT, NUM_WG), stride=(1, SLOT))
    tv_st = cute.make_tiled_copy_tv(
        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), F32,
                            num_bits_per_copy=128),
        cute.make_layout((WG_THREADS, NUM_WG)),   # thread layout: (lane, wg)
        cute.make_layout(128 // F32.width),       # value layout: 4 contiguous f32
    )
    return flat_l, tv_st


# ------------------------------------------------------------- PRE / POST -----
def _pre_xor_sum(v, off, lane):
    """One butterfly step of PRE's cross-lane row sum.

    Plain Python, so the spelling is chosen while the kernel is being traced
    rather than at run time: `_HAS_BFLY` is a module-level bool.
    """
    if _HAS_BFLY:
        return cute.arch.shuffle_sync_bfly(v, off)
    return cute.arch.shuffle_sync(v, lane ^ off)


@cute.kernel
def _pre_kernel(mO: cute.Tensor, mDO: cute.Tensor, mDelta: cute.Tensor,
                mDQacc: cute.Tensor, scale: cutlass.Constexpr[float],
                hq: cutlass.Constexpr[int], n_mblk: cutlass.Constexpr[int],
                nrow: cutlass.Constexpr[int]):
    """delta = scale * rowsum(dO . O), plus the dq_accum zero-init (D0.1).

    D1.3 dial b: the workspace delta carries the softmax scale.  `delta` is a
    ROW SCALAR, so this is the cheapest place the scale can be applied at all
    -- once per (b, h_q, row) for the whole launch, against once per row per
    TRIP had MAIN done it, against once per ELEMENT in the dial-a FMUL.  It
    costs one FFMA in a kernel whose whole body is two streams and a store.

    D3.1 sets the access width of both streams to 128 bits AND lays the
    threads out lane-contiguously.  A thread owns 16 B chunk
    `c = i * PRE_THREADS + tidx` of the flattened BM x HD tile, so one warp's
    32 lanes cover 512 contiguous bytes per instruction.  Both sides go
    through flat views built on the tensors' own iterators: (B, H_Q, N, D) is
    contiguous, so the three leading modes collapse to one BM-tile index.
    `nrow = B * H_Q * N` is the flattened row count and `nrow // BM` the tile
    count of both streams.

    Because a row is spread over PRE_CH_ROW = 8 lanes, the read side needs a
    cross-lane sum for `delta`.  PRE_THREADS % PRE_CH_ROW == 0 and 8 divides
    32, so the 8 chunks of one row are always held by 8 consecutive lanes of
    ONE warp and `c // PRE_CH_ROW` is that row inside the tile.  The three
    butterfly steps run OUTSIDE any predicate -- a shuffle's member mask is
    the whole warp, so every lane must reach it -- and only afterwards does
    lane `c % 8 == 0` store the row scalar.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    tile = (bz * hq + by) * n_mblk + bx
    lane = tidx % 32

    # --- read side: two 128-bit lane-contiguous streams over the tile -------
    ch_l = cute.make_layout((PRE_VEC_H, PRE_NCH, nrow // BM),
                            stride=(1, PRE_VEC_H, BM * HD))
    gO = cute.make_tensor(mO.iterator, ch_l)[None, None, tile]
    gDO = cute.make_tensor(mDO.iterator, ch_l)[None, None, tile]

    rO = cute.make_fragment(PRE_VEC_H, F16)
    rDO = cute.make_fragment(PRE_VEC_H, F16)
    for i in cutlass.range_constexpr(PRE_NCH // PRE_THREADS):
        c = i * PRE_THREADS + tidx
        cute.autovec_copy(gO[None, c], rO)
        cute.autovec_copy(gDO[None, c], rDO)
        s = F32(0.0)
        for e in cutlass.range_constexpr(PRE_VEC_H):
            s += F32(rO[e]) * F32(rDO[e])
        for st in cutlass.range_constexpr(3):
            s = s + _pre_xor_sum(s, 1 << st, lane)
        if tidx % PRE_CH_ROW == 0:
            mDelta[(bz, by, bx * BM + c // PRE_CH_ROW)] = s * F32(scale)

    # --- write side: the dq_accum zero-init, same 128-bit width -------------
    zero_l = cute.make_layout(
        (PRE_VEC_F, (BM * HD) // PRE_VEC_F, nrow // BM),
        stride=(1, PRE_VEC_F, BM * HD))
    gZ = cute.make_tensor(mDQacc.iterator, zero_l)[None, None, tile]

    rZ = cute.make_fragment(PRE_VEC_F, F32)
    for e in cutlass.range_constexpr(PRE_VEC_F):
        rZ[e] = F32(0.0)
    for i in cutlass.range_constexpr(
            (BM * HD) // (PRE_VEC_F * PRE_THREADS)):
        cute.autovec_copy(rZ, gZ[None, i * PRE_THREADS + tidx])


@cute.jit
def _post_body(mAcc, mOut, mma_kmn, tv_st,
               n_mblk: cutlass.Constexpr[int],
               nh: cutlass.Constexpr[int],
               n_tiles: cutlass.Constexpr[int],
               tidx, bx, by, bz, sT):
    """One gradient tile: f32 workspace -> fp16 output, through smem.

    D0.5: the workspace is written by MAIN's bulk reduce-add, which reproduces
    smem byte order in gmem -- so it is in `tv_st` order, NOT logical (m, d)
    order.  This read side therefore inverts through the SAME TV bijection and
    the SAME C-fragment view MAIN wrote from.

    D3.2 changes only where the C fragment lands.  It used to be scattered
    straight to gmem at 2 B per lane per row; it is now scattered into the
    padded smem tile `sT` and drained from there at `POSTV_VEC` f16 per thread,
    chunk index `i * NUM_THREADS + tidx`, so consecutive lanes hold consecutive
    16 B chunks of the tile and the gmem write side is fully coalesced.  The
    read side is untouched (it was already 128-bit).

    `@cute.jit` is load-bearing: the preprocessor that rewrites
    `cutlass.range_constexpr` only walks the source of a DECORATED function, so
    an undecorated helper called from a traced body dies at compile time with
    `range_constexpr should be preprocessed by preprocessor`.  The constexpr
    ints keep their annotations across the boundary for the same reason the
    kernel entries do -- re-spelling them unannotated would bind them dynamic.
    """
    # The tile this block owns, addressed exactly as MAIN's flush site did:
    # tile id = ((bz*nh + by)*n_mblk + bx), each tile STAGE_ELEMS f32 wide.
    # Viewed as a 3-mode stack (slot, wg, tile) so the tile index is a plain
    # runtime slice rather than pointer arithmetic.
    stack = cute.make_tensor(mAcc.iterator, cute.make_layout(
        (SLOT, NUM_WG, n_tiles), stride=(1, SLOT, STAGE_ELEMS)))
    tile = (bz * nh + by) * n_mblk + bx
    gflat = stack[None, None, tile]

    tG = tv_st.get_slice(tidx).partition_S(gflat)
    rF = cute.make_fragment_like(tG, F32)
    cute.autovec_copy(tG, rF)

    # Reinterpret back to MAIN's C-fragment layout and scatter by its coords,
    # into smem rather than into the output tensor.
    thr_o = mma_kmn.get_slice(tidx)
    cO = thr_o.partition_C(cute.make_identity_tensor((BN, HD)))
    acc_back = cute.make_tensor(rF.iterator, cute.make_layout(cute.shape(cO)))
    for i in cutlass.range_constexpr(cute.size(cO)):
        r, c = cO[i]
        sT[(r, c)] = F16(acc_back[i])
    cute.arch.barrier()

    # Drain: (vec, chunk, tile) views of the padded smem tile and of the
    # contiguous (B, NH, N, D) output, one 16 B chunk per thread per pass.
    sflat = cute.make_tensor(sT.iterator, cute.make_layout(
        (POSTV_VEC, POST_ROWCH, BM),
        stride=(1, POSTV_VEC, HD + POSTV_PAD)))
    gout = cute.make_tensor(mOut.iterator, cute.make_layout(
        (POSTV_VEC, POST_NCH, n_tiles),
        stride=(1, POSTV_VEC, BM * HD)))[None, None, tile]
    rS = cute.make_fragment(POSTV_VEC, F16)
    for i in cutlass.range_constexpr(POST_NCH // NUM_THREADS):
        j = i * NUM_THREADS + tidx
        cute.autovec_copy(sflat[None, j % POST_ROWCH, j // POST_ROWCH], rS)
        cute.autovec_copy(rS, gout[None, j])


def _post_smem():
    """The one staging tile a POST block owns: BM x (HD + POSTV_PAD) f16."""
    smem = cutlass.utils.SmemAllocator()
    return smem.allocate_tensor(
        F16, cute.make_layout((BM, HD), stride=(HD + POSTV_PAD, 1)),
        byte_alignment=16)


@cute.kernel
def _post_kernel(mAcc: cute.Tensor, mOut: cute.Tensor,
                 mma_kmn: cute.TiledMma, tv_st: cute.TiledCopy,
                 n_mblk: cutlass.Constexpr[int], nh: cutlass.Constexpr[int],
                 n_tiles: cutlass.Constexpr[int]):
    """One gradient tensor per launch -- dQ, whose grid is the H_Q one."""
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    sT = _post_smem()
    _post_body(mAcc, mOut, mma_kmn, tv_st, n_mblk, nh, n_tiles,
               tidx, bx, by, bz, sT)


@cute.kernel
def _post_kernel2(mAccA: cute.Tensor, mOutA: cute.Tensor,
                  mAccB: cute.Tensor, mOutB: cute.Tensor,
                  mma_kmn: cute.TiledMma, tv_st: cute.TiledCopy,
                  n_mblk: cutlass.Constexpr[int], nh: cutlass.Constexpr[int],
                  n_tiles: cutlass.Constexpr[int]):
    """D3.2's launch dial: dK and dV in ONE launch instead of two.

    They are the only two of the three POSTs that share a grid (`H_KV`, not
    `H_Q`) and a tile count, so a block can do both tiles back to back and the
    pair costs one launch's fixed overhead instead of two.  The two bodies
    share the single staging tile, hence the barrier between them: the second
    body's scatter must not overtake the first body's drain.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, bz = cute.arch.block_idx()
    sT = _post_smem()
    _post_body(mAccA, mOutA, mma_kmn, tv_st, n_mblk, nh, n_tiles,
               tidx, bx, by, bz, sT)
    cute.arch.barrier()
    _post_body(mAccB, mOutB, mma_kmn, tv_st, n_mblk, nh, n_tiles,
               tidx, bx, by, bz, sT)


# ----------------------------------------------------------------- MAIN -------
@cute.kernel
def _main_kernel(
    tQ: cute.Tensor, tK: cute.Tensor, tV: cute.Tensor, tdO: cute.Tensor,
    aQ, aK, aV, adO,
    mLSE: cute.Tensor, mDelta: cute.Tensor,
    mDQacc: cute.Tensor, mDKacc: cute.Tensor, mDVacc: cute.Tensor,
    mma_s: cute.TiledMma, mma_kmn: cute.TiledMma, mma_rs: cute.TiledMma,
    r2s_ds: cute.TiledCopy, atom_ds, tv_st: cute.TiledCopy, flat_l: cute.Layout,
    sK_l: cute.ComposedLayout, sV_l: cute.ComposedLayout,
    sQs_l: cute.ComposedLayout, sdOs_l: cute.ComposedLayout,
    sDSs_l: cute.ComposedLayout,
    sKt_l: cute.ComposedLayout, sQts_l: cute.ComposedLayout,
    sdOts_l: cute.ComposedLayout, sDSts_l: cute.ComposedLayout,
    Storage: cutlass.Constexpr[Callable],
    n_mblk: cutlass.Constexpr[int],
    hq: cutlass.Constexpr[int], gqa: cutlass.Constexpr[int],
    scale_log2e: cutlass.Constexpr[float], scale: cutlass.Constexpr[float],
    tx_kv: cutlass.Constexpr[int], tx_a: cutlass.Constexpr[int],
    tx_b: cutlass.Constexpr[int],
):
    # ---------------------------------------------------------------------
    # D0.4 shared preamble.  Everything here is compiled under the ENTRY
    # budget (floor(65536/384) = 168) and must stay in front of the single
    # role branch below: `setmaxnreg` is a runtime instruction and every
    # statement AFTER a `.dec` has to fit the reduced budget
    # (hw_kb/cutedsl/sm90/warp_specialization.md gotcha 11).  The smem base, the two
    # mbarrier addresses, the work-item cell and EVERY smem face live here;
    # each role's branch body then names faces only, never `storage`.
    # ---------------------------------------------------------------------
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(Storage)

    # D0.9 barrier cell.  ONE MemRange, viewed as `(1, NMBAR)` so that
    # `bar_t[(None, i)].iterator` is the `cute.Pointer` of barrier `i` for a
    # RUNTIME `i` -- which is what the two 2-stage pipelines need, since the
    # stage a trip uses is a runtime value.  Slot assignment: see B_FA/B_EA/
    # B_FB/B_EB/B_KV/B_RL above.
    bar_t = storage.mbar.get_tensor(cute.make_layout((1, NMBAR), stride=(1, 1)))
    barKV = _bar(bar_t, B_KV)

    # ---- D0.2: this CTA's work item is its grid coordinate ----------------
    # grid = (N/bN, H_Q, B) = (32, 16, 10), x = the KV block and the fastest
    # varying axis, so the 132 CTAs resident at any moment cover ~4.1 adjacent
    # `(b, h_q)` pairs and their Q/dO/dq_accum footprint stays inside L2.  One
    # CTA owns exactly one `(b, h_q, n_blk)`; there is no work queue, no
    # counter and no atomic claim -- `ctaid.x * BN` IS the KV start row.
    # The decode itself is done INSIDE each role branch, not here: `block_idx`
    # is a free SR read, while a value computed in the preamble would have to
    # live across the producer's `setmaxnreg.dec` into a 24-register budget
    # (hw_kb/cutedsl/sm90/warp_specialization.md gotcha 6).

    # ---------------------------------------------------------------------
    # Every smem FACE is built here, in the shared preamble, and never inside
    # a role branch.  Two reasons, one hard and one soft:
    #   hard: the DSL rewrites each kernel-body `if` into a traced region and
    #         packs the outer locals the body references into it as dynamic
    #         values.  `storage` is a `@cute.struct` instance -- a plain
    #         Python object with no dynamic-expression form -- so a single
    #         `storage.<field>` inside a branch body aborts the trace with
    #         `DSLRuntimeError: The 'if' statement encountered a user-defined
    #         Python object` naming the branch line, not the field line.
    #         `cute.Tensor` packs fine, so the faces may cross the boundary;
    #         the struct they come out of may not.
    #   soft: hw_kb/cutedsl/sm90/warp_specialization.md's placement rule (gotcha 11)
    #         puts the smem tensors in the preamble anyway, where they compile
    #         under the ENTRY budget floor(65536/384) = 168.
    # Each face is a shared address plus a compile-time layout, and a face
    # used by only one role stays dead on the other role's path.
    # ---------------------------------------------------------------------
    # single-buffered K/V, in both roles' hands
    sK = storage.sK.get_tensor(sK_l.outer, swizzle=sK_l.inner)
    sV = storage.sV.get_tensor(sV_l.outer, swizzle=sV_l.inner)
    # D1.3 dial b: a second, FLAT view of sV's MemRange -- the same bytes in
    # physical order, no swizzle.  The consumer multiplies the whole face by
    # `scale` through it once per CTA.  A uniform scalar multiply is invariant
    # under the swizzle permutation, so this view asserts no thread<->element
    # bijection at all; it only says "these are the 8192 f16 of the face".
    VSC_N = cute.cosize(sV_l)
    VSC_REP = VSC_N // NUM_THREADS
    assert VSC_REP * NUM_THREADS == VSC_N
    sVf = storage.sV.get_tensor(cute.make_layout(VSC_N))
    # D0.5 stage counts: Q/dO and sdS are 2-stage faces.  One MemRange each,
    # viewed through a 3-mode layout whose outermost mode is the stage; the
    # trip loop slices it at `m % NSTAGE`.
    sQall = storage.sQ.get_tensor(sQs_l.outer, swizzle=sQs_l.inner)
    sdOall = storage.sdO.get_tensor(sdOs_l.outer, swizzle=sdOs_l.inner)
    sDSall = storage.sDS.get_tensor(sDSs_l.outer, swizzle=sDSs_l.inner)
    # consumer only: second view of the same bytes, MN-major / transposed
    # operand faces
    sKt = storage.sK.get_tensor(sKt_l.outer, swizzle=sKt_l.inner)
    sQtall = storage.sQ.get_tensor(sQts_l.outer, swizzle=sQts_l.inner)
    sdOtall = storage.sdO.get_tensor(sdOts_l.outer, swizzle=sdOts_l.inner)
    sDStall = storage.sDS.get_tensor(sDSts_l.outer, swizzle=sDSts_l.inner)
    # D0.5 row scalars.  Producer fills them as a flat (BM, NSTAGE) face;
    # the consumer reads them through the `(2, BM/2)` PAIR view the LDS.64
    # read side loads through.  Same bytes, two views.
    sLall = storage.sLSE.get_tensor(
        cute.make_layout((BM, NSTAGE), stride=(1, BM)))
    sDall = storage.sDel.get_tensor(
        cute.make_layout((BM, NSTAGE), stride=(1, BM)))
    sL2all = storage.sLSE.get_tensor(
        cute.make_layout((2, BM // 2, NSTAGE), stride=(1, 2, BM)))
    sD2all = storage.sDel.get_tensor(
        cute.make_layout((2, BM // 2, NSTAGE), stride=(1, 2, BM)))
    # D0.5 staging face: flat TV slots, no swizzle.  Written by the consumer,
    # read (as a raw address) by the producer's bulk reduce-add.
    sStage = storage.sStage.get_tensor(flat_l)

    # ---------------------------------------------------------------------
    # D0.4: the ONE role branch.  `setmaxnreg` is `.sync.aligned`, so the
    # predicate is warpgroup-uniform (`warp_idx < 4`), and the dec/inc is the
    # first statement of each body with no barrier between them.  Inside the
    # producer warpgroup a second, warp-level predicate (`warp_idx == 0`)
    # keeps every TMA / bulk copy / bulk reduce-add on warp 0 alone; warps 1-3
    # walk the same barrier skeleton and issue nothing.
    #
    # Both bodies run the SAME whole-CTA barrier sequence, ONCE per CTA (the
    # CTA owns exactly one work item, D0.2):
    #   B1 B2 [trip loop, no whole-CTA barrier at all]* B8 B9 B10 B11 B12
    # D0.9 emptied the trip loop of whole-CTA rendezvous in two steps: part 1
    # retired B3/B4 (the Q/dO fill convoy) into the mbarrier operand pipelines,
    # part 2 retired B5 into a consumer-only 256-thread named barrier and
    # B6/B7 into the per-WG 160-participant dQ credit ring, and moved B8 out of
    # the loop.  B13 closed the work-item loop and retired with it.  Retired
    # labels are left retired rather than renumbered so the D0.8 segment map
    # keeps naming the same points.
    # Both bodies derive their loop bounds from the same grid coordinate, so
    # the two halves cannot drift in barrier count.
    # ---------------------------------------------------------------------
    if warp_idx < PRODUCER_WARPS:
        _REG_DEC(PRODUCER_REGS)

        # D0.4: the producer holds only addresses.  Its faces are the WRITE
        # ends of the D0.5 buffer list; the views themselves are built in the
        # shared preamble (see there), so nothing in this body touches the
        # `storage` object.
        stage_addr = sStage.iterator.toint()
        dq_base = mDQacc.iterator.toint()
        dk_base = mDKacc.iterator.toint()
        dv_base = mDVacc.iterator.toint()

        # ---- D0.2 work item, decoded from the grid ----
        # D1.1 r1: this decode is the SEED of every integer chain in the
        # producer body -- the TMA tile coordinates, the row-scalar element
        # offset, and the three write-back addresses all descend from it.
        # Marked here, inside the role branch, because the marking does not
        # survive the role `scf.if` (wgmma_desc_uniform.md gotcha 2) and this
        # branch runs on a 24-register budget, where a bookkeeping chain that
        # fell back to the regular file is the most expensive thing there is.
        bidx, bidy, bidz = cute.arch.block_idx()
        n_blk = _MWU(bidx)
        h = _MWU(bidy)
        b = _MWU(bidz)
        hk = _MWU(h // gqa)

        # ---- D0.5 write atom for sK/sV: TMA-bulk G->S, descriptor driven -
        # D0.9: the WHOLE barrier cell is initialised here, once per CTA,
        # and never again inside the trip loop.  That is what buys a
        # derivable phase parity: the trip index `t = m - n_blk` starts at 0,
        # so lap `t // NSTAGE` and its parity are ordinary arithmetic.  The
        # init sits ahead of B1, the first point both roles are rendezvoused.
        # `expect_tx` and `init` are single-thread instructions and sit
        # inside `elect_one`; the `cute.copy` atom elects internally and
        # must NOT be wrapped again -- both predicates are warp-uniform.
        #
        # Arrive counts, the number a mistake in which is a deadlock (too
        # big) or a silent early release (too small):
        #   full_A  1 + BM = 129  one `arrive_and_expect_tx` (Q's tx) plus
        #                         one arrive from each of the BM producer
        #                         threads AFTER it has scaled its lse row
        #   empty_A NUM_THREADS = 256   every consumer thread releases
        #   full_B  1            `arrive_and_expect_tx` alone: dO + delta
        #                         are pure transactions, nothing to post-
        #                         process, so the tx count publishes them
        #   empty_B NUM_THREADS = 256
        #   B_RL    1            the lse vector's own tx
        #   B_KV    1            the K+V tx
        gK = cute.local_tile(tK, (BN, HD), (n_blk, 0, hk, b))
        gV = cute.local_tile(tV, (BN, HD), (n_blk, 0, hk, b))
        tKs, tKg = cpasync.tma_partition(
            aK, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVs, tVg = cpasync.tma_partition(
            aV, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))
        if warp_idx == 0:
            with cute.arch.elect_one():
                _MB_INIT(barKV, 1)
                for _st in cutlass.range_constexpr(NSTAGE):
                    _MB_INIT(_bar(bar_t, B_FA + _st), 1 + BM)
                    _MB_INIT(_bar(bar_t, B_EA + _st), NUM_THREADS)
                    _MB_INIT(_bar(bar_t, B_FB + _st), 1)
                    _MB_INIT(_bar(bar_t, B_EB + _st), NUM_THREADS)
                    _MB_INIT(_bar(bar_t, B_RL + _st), 1)
        _mb_fence()
        cute.arch.barrier()                                        # B1
        if warp_idx == 0:
            with cute.arch.elect_one():
                _MB_TX(barKV, I32(tx_kv))
            cute.copy(aK, tKg, tKs, tma_bar_ptr=barKV)
            cute.copy(aV, tVg, tVs, tma_bar_ptr=barKV)
        _MB_WAIT(barKV, I32(0))
        cute.arch.barrier()                                        # B2

        # ---- D0.9 class 2: prime the dQ staging credit ring ----------
        # Both 16 KiB halves are untouched at this point (the CTA has run
        # no trip yet), so the producer hands out the first credit here.
        # Exactly one grant
        # per half: the in-loop service below grants one more per trip it
        # retires, and the post-loop service grants none, which is what
        # makes the arrival counts balance over a whole work item
        # (grants 1 + (n-1) = n, consumer waits n).
        if warp_idx == 0:
            _nb_arrive(I32(NB_EMPTY + 0), I32(STAGE_CREDIT))
            _nb_arrive(I32(NB_EMPTY + 1), I32(STAGE_CREDIT))

        for m in cutlass.range(n_blk, n_mblk):
            # ---- D0.9 ring state.  `t` is the trip index INSIDE this work
            # item, so it starts at 0 no matter which q block the causal
            # walk starts from; `stage` is the buffer slot and `lap` the
            # number of times this slot has been round-tripped.  A barrier
            # completes once per lap, so the parity a wait names is the
            # lap's low bit -- `ph_f` for the full/landing barriers this
            # lap fills, `ph_e` for the empty barriers the PREVIOUS lap
            # left behind.  Both roles compute these from the same two
            # runtime values (`m`, `n_blk`); a disagreement is a deadlock,
            # not a wrong number.
            # D1.1 r1: the whole trip state stays on the uniform path.  It
            # is fed by `m` (the loop counter) and `n_blk` (a grid
            # coordinate), so it is warp-invariant by construction; the
            # marking matters because `stage` then indexes a barrier at
            # runtime and slices the smem faces below, and ptxas's own
            # invariance proof does not survive either of those.
            t = _MWU(m - n_blk)
            stage = _MWU(t % NSTAGE)
            lap = _MWU(t // NSTAGE)
            ph_f = _MWU(lap % 2)
            ph_e = _MWU((lap + 1) % 2)
            bFA = _bar(bar_t, B_FA + stage)
            bEA = _bar(bar_t, B_EA + stage)
            bFB = _bar(bar_t, B_FB + stage)
            bEB = _bar(bar_t, B_EB + stage)
            bRL = _bar(bar_t, B_RL + stage)
            sQ = sQall[None, None, stage]
            sdO = sdOall[None, None, stage]
            sL = sLall[None, stage]
            sD = sDall[None, stage]

            # ---- D0.9 producer acquire: this slot's previous tenants must
            # be gone before either face is overwritten.  Skipped on the
            # first lap, where the slots are fresh out of the re-init above
            # and no consumer has ever released them (the standard "first
            # NSTAGE trips do not wait" of a stage ring; a wait here would
            # block on a phase nothing will ever complete).
            if t >= NSTAGE:
                _MB_WAIT(bEA, I32(ph_e))
                _MB_WAIT(bEB, I32(ph_e))

            # ---- D0.5 write atoms for sQ/sdO AND for the row scalars, on
            # the D0.9 pipeline barriers.  A = Q on `full_A` + lse on the
            # private `B_RL`; B = dO + delta on `full_B` as one MIXED
            # transaction (a bulk-TENSOR face and a non-tensor 1-D bulk
            # vector on one expect_tx).  Splitting A from B is what lets
            # the consumer issue chain 0 between the two waits (D0.8
            # segment A).
            # `cute.copy` elects internally and must not be wrapped again;
            # the bare-PTX `_bulk_g2s` does not, so it must be.
            gQ = cute.local_tile(tQ, (BM, HD), (m, 0, h, b))
            gdO = cute.local_tile(tdO, (BM, HD), (m, 0, h, b))
            tQs, tQg = cpasync.tma_partition(
                aQ, 0, cute.make_layout(1),
                cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
            tdOs, tdOg = cpasync.tma_partition(
                adO, 0, cute.make_layout(1),
                cute.group_modes(sdO, 0, 2), cute.group_modes(gdO, 0, 2))
            # lse and delta are both (B, H_Q, N) f32 and contiguous, so a
            # row block is one 512 B run at this linear element offset.
            rs_row = _MWU((b * hq + h) * (n_mblk * BM) + m * BM)
            srcL = mLSE.iterator.toint() + cutlass.Int64(rs_row) * 4
            srcD = mDelta.iterator.toint() + cutlass.Int64(rs_row) * 4
            if warp_idx == 0:
                with cute.arch.elect_one():
                    _MB_TX(bFA, I32(tx_a))
                    _MB_TX(bFB, I32(tx_b))
                    _MB_TX(bRL, I32(ROW_BYTES))
                cute.copy(aQ, tQg, tQs, tma_bar_ptr=bFA)
                cute.copy(adO, tdOg, tdOs, tma_bar_ptr=bFB)
                with cute.arch.elect_one():
                    _bulk_g2s(sL.iterator.toint(), srcL, I32(ROW_BYTES),
                              bRL.toint())
                    _bulk_g2s(sD.iterator.toint(), srcD, I32(ROW_BYTES),
                              bFB.toint())
            # D0.7 item 3: lse stays in the natural-log domain across the
            # kernel boundary and log2e is applied ONCE PER ROW inside
            # MAIN.  That ruling is what forces the private landing barrier
            # `B_RL`: the producer has to SEE the vector before it can
            # scale it, and it must publish stage A only afterwards.  So
            # the wait here is on 512 B (not on Q's 16 KiB, still in
            # flight), each of the BM producer threads scales its own row,
            # and each then arrives on `full_A` -- the 128 arrives of the
            # 129 count.  `mbarrier.arrive` is a release at CTA scope, so
            # the scaled row is visible to whoever completes the wait.
            _MB_WAIT(bRL, I32(ph_f))
            if tidx < BM:
                sL[tidx] = sL[tidx] * F32(LOG2E)
            _mb_arrive(bFA.toint())

            # ---- D0.9 class 2 + the producer run-ahead it unlocks ----
            # The whole-CTA B5/B6/B7/B8 that fenced this loop are gone.
            # What replaces them is a per-WG credit ring serviced ONE TRIP
            # BEHIND: at trip t the producer has just issued trip t's TMAs
            # and now retires trip t-1's dQ tile.  That lag is the whole
            # point -- with the service on trip t itself the producer would
            # block until the consumer finished trip t, publish trip t+1
            # only afterwards, and the consumer would wait on a fill it had
            # just triggered.  Lagged, the producer's fill for tile t+1 is
            # already in flight while the consumer computes tile t.
            # `stage_addr + SLOT_BYTES` is warpgroup 1's half; the two
            # reduce-adds together are byte-for-byte the single 32 KiB one
            # they replace, because the staging layout is
            # `(SLOT, NUM_WG)` with stride `(1, SLOT)`.
            # The producer gives buffer 0 back with buffer 1's write-back
            # STILL IN FLIGHT (`wait_group(1)`, one bulk group outstanding)
            # -- the D0.11 [MEASURED] shape of this handoff.
            # Only warp 0 takes part (the 32 of the 160), so warps 1-3 walk
            # straight through to the next trip and park on that trip's
            # `B_RL` wait, which warp 0 alone can satisfy: the producer
            # warpgroup cannot drift more than one trip apart.
            dq_prev = dq_base + cutlass.Int64(
                (b * hq + h) * n_mblk + m - 1) * STAGE_BYTES
            if t > 0:
                if warp_idx == 0:
                    _nb_sync(I32(NB_FULL + 0), I32(STAGE_CREDIT))
                    if tidx == 0:
                        _bulk_reduce_add_f32(dq_prev, stage_addr,
                                             SLOT_BYTES)
                        _bulk_commit_group()
                    _nb_sync(I32(NB_FULL + 1), I32(STAGE_CREDIT))
                    if tidx == 0:
                        _bulk_reduce_add_f32(dq_prev + SLOT_BYTES,
                                             stage_addr + SLOT_BYTES,
                                             SLOT_BYTES)
                        _bulk_commit_group()
                        _bulk_wait_group_read(1)
                    _nb_arrive(I32(NB_EMPTY + 0), I32(STAGE_CREDIT))
                    if tidx == 0:
                        _bulk_wait_group_read(0)
                    _nb_arrive(I32(NB_EMPTY + 1), I32(STAGE_CREDIT))

        # ---- D0.9 class 2: the last trip's dQ tile ----------------------
        # The in-loop service runs one trip behind, so trip n_mblk-1 is
        # still unserviced here.  No `empty` credit is granted for it: the
        # next consumer of the staging face is the dK/dV epilogue below,
        # and B8 (relocated out of the trip loop, once per work item now)
        # is what tells it the face is drained.
        if warp_idx == 0:
            dq_last = dq_base + cutlass.Int64(
                (b * hq + h) * n_mblk + n_mblk - 1) * STAGE_BYTES
            _nb_sync(I32(NB_FULL + 0), I32(STAGE_CREDIT))
            if tidx == 0:
                _bulk_reduce_add_f32(dq_last, stage_addr, SLOT_BYTES)
                _bulk_commit_group()
            _nb_sync(I32(NB_FULL + 1), I32(STAGE_CREDIT))
            if tidx == 0:
                _bulk_reduce_add_f32(dq_last + SLOT_BYTES,
                                     stage_addr + SLOT_BYTES, SLOT_BYTES)
                _bulk_commit_group()
                _bulk_wait_group_read(0)
        cute.arch.barrier()                                        # B8

        # ---- D0.12: dV then dK flushed once per work item ----
        cute.arch.barrier()                                        # B9
        if tidx == 0:
            _bulk_reduce_add_f32(
                dv_base + cutlass.Int64(
                    (b * (hq // gqa) + hk) * n_mblk + n_blk) * STAGE_BYTES,
                stage_addr, STAGE_BYTES)
            _bulk_commit_group()
            _bulk_wait_group_read(0)
        cute.arch.barrier()                                        # B10

        cute.arch.barrier()                                        # B11
        if tidx == 0:
            _bulk_reduce_add_f32(
                dk_base + cutlass.Int64(
                    (b * (hq // gqa) + hk) * n_mblk + n_blk) * STAGE_BYTES,
                stage_addr, STAGE_BYTES)
            _bulk_commit_group()
            _bulk_wait_group_read(0)
        cute.arch.barrier()                                        # B12

    else:
        _REG_INC(CONSUMER_REGS)

        # D0.4: every heavy register state is created INSIDE this branch, so
        # no accumulator's live range crosses the producer's `.dec` point
        # (gotcha 6).  `ctid` is the consumer-local thread id: the tiled MMAs,
        # the r2s copy and the staging TV are all built over NUM_THREADS = 256
        # lanes, and this branch owns lanes 128..383.
        ctid = tidx - WG_THREADS

        # Every smem face this body reads or writes is built in the shared
        # preamble (see there); no reference to `storage` survives inside a
        # role branch.
        wg_idx = cute.arch.make_warp_uniform(ctid // WG_THREADS)
        wg_layout = cute.make_layout(NUM_WG, stride=WG_THREADS)
        # D0.9 class 2: this WG's own pair of credit barriers.  The id is a
        # register operand of `bar.sync`/`bar.arrive`, so a runtime `wg_idx`
        # is a legal barrier selector; both are loop invariants.
        nb_full = I32(NB_FULL) + wg_idx
        nb_empty = I32(NB_EMPTY) + wg_idx

        thr_s = mma_s.get_slice(ctid)
        thr_o = mma_kmn.get_slice(ctid)
        wg_s = mma_s.get_slice(wg_layout(wg_idx))
        wg_kmn = mma_kmn.get_slice(wg_layout(wg_idx))
        wg_rs = mma_rs.get_slice(wg_layout(wg_idx))

        cS_id = cute.make_identity_tensor((BN, BM))      # (kv, q) coords
        cS = thr_s.partition_C(cS_id)
        acc_s = cute.make_fragment(cS.shape, F32)
        acc_dp = cute.make_fragment(cS.shape, F32)

        cO_id = cute.make_identity_tensor((BN, HD))
        acc_dv = cute.make_fragment(thr_o.partition_C(cO_id).shape, F32)
        acc_dk = cute.make_fragment(thr_o.partition_C(cO_id).shape, F32)
        acc_dq = cute.make_fragment(thr_o.partition_C(cO_id).shape, F32)

        # D0.6 [SOURCE: RS]: the A operands of chain2 / chain4 live in
        # registers for their whole lifetime.
        rP = wg_rs.make_fragment_A(wg_rs.partition_shape_A((BN, BM)))
        rDS = wg_rs.make_fragment_A(wg_rs.partition_shape_A((BN, BM)))
        rP_flat = cute.make_tensor(rP.iterator, cute.make_layout(cute.size(rP)))
        rDS_flat = cute.make_tensor(rDS.iterator,
                                    cute.make_layout(cute.size(rDS)))

        # D0.5 sdS write atom: transposed `stmatrix` instead of 64 scalar
        # stores.  The source is the SAME register file as rDS, reinterpreted
        # with the shape of the copy's destination partition -- not `.retile`,
        # which disagrees in rank with the C-fragment nesting here.
        rDS_st = cute.make_tensor(rDS.iterator, cute.make_layout(cute.shape(
            r2s_ds.get_slice(ctid).partition_D(sDStall[None, None, 0]))))

        # D0.8 segment D: the same face, driven one chunk at a time.  Grouping
        # the trailing modes into one gives a single integer that indexes both
        # the copy chunk and -- because the compact relabel above is colex with
        # mode 0 innermost -- the matching register range of `rDS_flat`, which
        # is [c*NV_DS, (c+1)*NV_DS).  NV_DS is a multiple of the D1.2 row-
        # scalar group width, so a chunk boundary never splits either the
        # adjacent-q pair the LDS.64 read atom brings in or the 2x2 block that
        # shares it (`_chunk_groups` raises at trace time if it ever does).
        gsrc_ds = cute.group_modes(rDS_st, 1, _rank_of(rDS_st))
        NCH_DS = cute.size(gsrc_ds.shape[1])
        NV_DS = cute.size(rDS_flat) // NCH_DS

        # D1.2 item 2: the two RS fragments seen 32 bits at a time.  Element
        # i of these views IS the f16 pair (2i, 2i+1) of the fragment above,
        # so the bijection every downstream consumer already relies on
        # (rP_flat[i] <-> acc_s[i], rDS_flat[i] <-> the sdS store partition)
        # is untouched -- only the number of instructions that fill it moves.
        rP_i32 = _as_i32(rP_flat)
        rDS_i32 = _as_i32(rDS_flat)

        # D1.2 item 3: one row-scalar load per 2x2 C-fragment block instead of
        # one per pair.  GW_EW is derived from the accumulator's own mode 0
        # (and checked there), NG_EW / NG_DS are the resulting trip counts of
        # the two elementwise segments.
        GW_EW = _ew_group(acc_s.shape)
        NG_EW = cute.size(acc_s) // GW_EW
        NG_DS = _chunk_groups(NV_DS, GW_EW)

        # D0.5 row-scalar READ atom: the destination of one LDS.64.
        rl = cute.make_fragment(2, F32)
        rd = cute.make_fragment(2, F32)
        # D1.2 item 2 under the causal mask: the masked branch assigns a
        # PREDICATED STORE into this 2-element scratch and the pack reads it
        # back, so the two mask arms stay stores (what the DSL lowers cleanly)
        # while the convert still sees a value pair.
        rv = cute.make_fragment(2, F32)

        # D0.5 staging write side.  Loop invariants: one destination
        # partition, and three linear reinterpretations of the accumulator
        # register files at its shape.
        tSt_sD = tv_st.get_slice(ctid).partition_D(sStage)
        _st_shape = cute.make_layout(cute.shape(tSt_sD))
        dq_st = cute.make_tensor(acc_dq.iterator, _st_shape)
        dv_st = cute.make_tensor(acc_dv.iterator, _st_shape)
        dk_st = cute.make_tensor(acc_dk.iterator, _st_shape)

        # ---- D0.2 work item: the consumer needs only the KV block, which is
        # the causal walk's first q block.
        # D1.1 r1: same marking as the producer's decode, re-taken on this
        # side of the role branch.
        _bx, _, _ = cute.arch.block_idx()
        n_blk = _MWU(_bx)

        cute.arch.barrier()                                        # B1
        _MB_WAIT(barKV, I32(0))

        # ---- D1.3 dial b: fold the softmax scale into sV ------------------
        # `dP^T = V . dO^T` (chain 1) is the ONLY reader of V -- dV = P^T.dO
        # takes dO, not V -- so scaling the face is arithmetically identical
        # to scaling every dP element, and with delta arriving pre-scaled from
        # PRE the dS segment's `* scale` FMUL is gone: 64 fewer elementwise
        # instructions per lane per TRIP, paid for by 32 stores per lane ONCE
        # per CTA.  scale = 2^-3 is a power of two and exact in fp16, so the
        # only representation risk is a v small enough to fall subnormal.
        # Scalar spelling: the vectorised `cute.autovec_copy` reg->smem form
        # is a known device-hang carrier at the epilogue sites (see B9/B10),
        # and this site runs once per CTA, where the width buys nothing.
        for _vr in cutlass.range_constexpr(VSC_REP):
            _vi = _vr * NUM_THREADS + ctid
            sVf[_vi] = F16(F32(sVf[_vi]) * F32(scale))
        # Generic-proxy stores into a face the ASYNC proxy (WGMMA) reads: the
        # fence is what makes them visible across proxies, B2 what makes them
        # visible across the 256 lanes.  Same pairing as the staging flush.
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                              space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier()                                        # B2

        acc_dv.fill(0.0)
        acc_dk.fill(0.0)

        for m in cutlass.range(n_blk, n_mblk):
            # ---- D0.9 ring state, computed exactly as the producer's ----
            # D1.1 r1: and marked exactly as the producer's.  Here the
            # marking also carries the D0.8 operand descriptors: `stage`
            # slices sQ/sdO/sDS/sQt/sdOt below, and those slices are the
            # coordinates fed to `partition_A/B`, which is the one place a
            # WGMMA descriptor can be pinned to the uniform file
            # (hw_kb/cutedsl/sm90/wgmma_desc_uniform.md r1 -- the descriptor itself
            # is not markable, its slice coordinate is).
            t = _MWU(m - n_blk)
            stage = _MWU(t % NSTAGE)
            lap = _MWU(t // NSTAGE)
            ph_f = _MWU(lap % 2)
            bFA = _bar(bar_t, B_FA + stage)
            bEA = _bar(bar_t, B_EA + stage)
            bFB = _bar(bar_t, B_FB + stage)
            bEB = _bar(bar_t, B_EB + stage)
            # ---- D0.5 stage slice ----
            sQ = sQall[None, None, stage]
            sdO = sdOall[None, None, stage]
            sQt = sQtall[None, None, stage]
            sdOt = sdOtall[None, None, stage]
            sDS = sDSall[None, None, stage]
            sDSt = sDStall[None, None, stage]
            sL2 = sL2all[None, None, stage]
            sD2 = sD2all[None, None, stage]

            # =========== D0.8 segment A ===========================
            # D0.9 consumer wait on pipeline A: Q's face plus the lse row
            # the producer has already scaled.  dO is NOT waited for yet:
            # chain 0 only needs K and Q, so it is issued first and dO's
            # TMA latency runs underneath it.  Hoisting the pipeline-B wait
            # up here next to this one leaves every segment below in place,
            # spills nothing, passes every architectural line -- and
            # empties this window.
            _MB_WAIT(bFA, I32(ph_f))
            # ---- chain 0: S^T = K . Q^T  (both k-major) ----
            tSrK = wg_s.make_fragment_A(wg_s.partition_A(sK))
            tSrQ = wg_s.make_fragment_B(wg_s.partition_B(sQ))
            atom0 = cute.make_mma_atom(mma_s.op)
            atom0.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tSrK.shape[2])):
                cute.gemm(atom0, acc_s, tSrK[None, None, kb],
                          tSrQ[None, None, kb], acc_s)
                atom0.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            # ---- chain 1: dP^T = V . dO^T  (issued in segment A, behind
            #      chain 0, so that only ITS operand wait is on dO) ----
            # D0.9 pipeline B carries dO AND delta, so this one wait also
            # covers the row scalar the dS segment reads further down.
            _MB_WAIT(bFB, I32(ph_f))
            tPrV = wg_s.make_fragment_A(wg_s.partition_A(sV))
            tPrdO = wg_s.make_fragment_B(wg_s.partition_B(sdO))
            atom1 = cute.make_mma_atom(mma_s.op)
            atom1.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tPrV.shape[2])):
                cute.gemm(atom1, acc_dp, tPrV[None, None, kb],
                          tPrdO[None, None, kb], acc_dp)
                atom1.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            # =========== D0.8 segment B (the peak segment) ========
            # Graded wait: chain 0 retires (its accumulator is the softmax
            # input), chain 1 STAYS IN FLIGHT.  `wait_group(0)` here would
            # be correct, cost no register and hide nothing.
            warpgroup.wait_group(1)

            # ---- P^T = exp2(S^T * (scale*log2e) - lse*log2e), masked ----
            #      lands straight in the chain2 A fragment (D0.6 [SOURCE: RS])
            # D0.5 row-scalar READ atom: ONE LDS.64 per adjacent-q pair.
            if m == n_blk:
                for g in cutlass.range_constexpr(NG_EW):
                    i0 = GW_EW * g
                    kvg, qg = cS[i0]
                    cute.autovec_copy(sL2[None, qg // 2], rl)
                    for u in cutlass.range_constexpr(GW_EW // 2):
                        i = i0 + 2 * u
                        kv0, qr0 = cS[i]
                        kv1, qr1 = cS[i + 1]
                        if qr0 >= kv0:
                            rv[0] = cute.math.exp2(
                                acc_s[i] * F32(scale_log2e) - rl[0],
                                fastmath=True)
                        else:
                            rv[0] = F32(0.0)
                        if qr1 >= kv1:
                            rv[1] = cute.math.exp2(
                                acc_s[i + 1] * F32(scale_log2e) - rl[1],
                                fastmath=True)
                        else:
                            rv[1] = F32(0.0)
                        rP_i32[i // 2] = _cvt_f16x2_f32(rv[0], rv[1])
            else:
                for g in cutlass.range_constexpr(NG_EW):
                    i0 = GW_EW * g
                    kvg, qg = cS[i0]
                    cute.autovec_copy(sL2[None, qg // 2], rl)
                    for u in cutlass.range_constexpr(GW_EW // 2):
                        i = i0 + 2 * u
                        rP_i32[i // 2] = _cvt_f16x2_f32(
                            cute.math.exp2(acc_s[i] * F32(scale_log2e) - rl[0],
                                           fastmath=True),
                            cute.math.exp2(acc_s[i + 1] * F32(scale_log2e)
                                           - rl[1], fastmath=True))

            # =========== D0.8 segment C (chain 2 goes first) ======
            # Chain 2 is issued HERE, ahead of the dS segment, not down in
            # segment E: its A operand `rP` is complete at the end of
            # segment B and its B operand is this warpgroup's own dO face,
            # so it depends on nothing that the cross-warpgroup rendezvous
            # supplies.  It is the longest chain of the trip (K depth 128 =
            # 8 HGMMA beats), and the dS segment plus the pack/store
            # segment below are the trip's only priced elementwise cost --
            # they run in its shadow.  Pushing chain 2 back behind the
            # rendezvous re-exposes both of them.
            # ---- chain 2: dV += P^T . dO   (A from registers, D0.6) ----
            tVrdO = wg_rs.make_fragment_B(wg_rs.partition_B(sdOt))
            atom2 = cute.make_mma_atom(mma_rs.op)
            atom2.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(rP.shape[2])):
                cute.gemm(atom2, acc_dv, rP[None, None, kb],
                          tVrdO[None, None, kb], acc_dv)
            warpgroup.commit_group()

            # Graded wait: chain 1 retires (the dS segment reads its
            # accumulator), chain 2 stays in flight.  Chain 1's 64 f32
            # accumulator registers die here -- that is the segment B peak
            # falling back, and it is what pays for chain 2 being airborne.
            warpgroup.wait_group(1)

            # =========== D0.8 segments C + D (in chain 2's shadow) =
            # dS^T = P^T . (dP^T - delta), packed to f16 and stored
            # to sdS one CHUNK at a time (pack a batch, write that batch),
            # D1.3 dial b: the scale is not here any more -- it is already in
            # BOTH terms of the difference (sV was scaled once per CTA, delta
            # once per row in PRE), so this segment carries no `* scale`.
            # not one whole-fragment pack followed by one whole-fragment
            # store.  The f32 dS value never outlives the chunk it is in:
            # nothing here holds a high-precision copy of a packed element.
            # The sdS write partition is stage dependent, so it is rebuilt
            # here from the sliced face; only its shape is invariant.
            tDS_sD = r2s_ds.get_slice(ctid).partition_D(sDSt)
            gdst_ds = cute.group_modes(tDS_sD, 1, _rank_of(tDS_sD))
            for c in cutlass.range_constexpr(NCH_DS):
                for g in cutlass.range_constexpr(NG_DS):
                    i0 = c * NV_DS + GW_EW * g
                    kvg, qg = cS[i0]
                    cute.autovec_copy(sD2[None, qg // 2], rd)
                    for u in cutlass.range_constexpr(GW_EW // 2):
                        i = i0 + 2 * u
                        rDS_i32[i // 2] = _cvt_f16x2_f32(
                            F32(rP_flat[i]) * (acc_dp[i] - rd[0]),
                            F32(rP_flat[i + 1]) * (acc_dp[i + 1] - rd[1]))
                _copy_chunk(r2s_ds, atom_ds,
                            gsrc_ds[None, c], gdst_ds[None, c])

            # =========== D0.8 segments E + F ======================
            # Segment E issues no MMA (chain 2 has moved up to segment C);
            # segment F is the fence plus the kernel's ONE symmetric
            # rendezvous, where the two consumer warpgroups exchange dS.
            cute.arch.fence_view_async_shared()
            # D0.9 class 3: the kernel's ONE symmetric rendezvous.  It is
            # the only consumer<->consumer signal there is, and the
            # producer no longer takes part -- 256 participants, not 384,
            # so the producer walks past this point instead of being
            # dragged into every trip's dS exchange.
            _nb_sync(I32(NB_DS), I32(NUM_THREADS))                 # B5

            # ---- chain 3: dQ = dS . K ----
            tQrDS = wg_kmn.make_fragment_A(wg_kmn.partition_A(sDS))
            tQrK = wg_kmn.make_fragment_B(wg_kmn.partition_B(sKt))
            atom3 = cute.make_mma_atom(mma_kmn.op)
            atom3.set(warpgroup.Field.ACCUMULATE, False)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(tQrDS.shape[2])):
                cute.gemm(atom3, acc_dq, tQrDS[None, None, kb],
                          tQrK[None, None, kb], acc_dq)
                atom3.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            # D2.1, dial `f1deep`: this wait point goes from depth 1 to
            # depth 2 -- it retires NOTHING and leaves chains 2 AND 3 both
            # in flight across chain 4's issue.  At depth 1 chain 2 retired
            # here and chain 4 was issued into a one-chain shadow; at depth
            # 2 chain 4 is issued with two chains still airborne and chain 2
            # retires one segment later, at the F2 point below.  The cost is
            # `rP`'s live range: it now spans chain 4's issue as well,
            # because a WGMMA may read its RMEM A operand any time before
            # the group retires.
            warpgroup.wait_group(2)
            # D0.9 consumer release, pipeline B, MOVED DOWN with the dial:
            # the dO slot may only be handed back once chain 2 -- the last
            # reader of this trip's dO face -- has actually retired, and
            # under depth 2 it has not retired yet at this point.  The
            # arrive now sits immediately after the F2 wait below.

            # ---- chain 4: dK += dS^T . Q   (A from registers, D0.6) ----
            tKrQ = wg_rs.make_fragment_B(wg_rs.partition_B(sQt))
            atom4 = cute.make_mma_atom(mma_rs.op)
            atom4.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(rDS.shape[2])):
                cute.gemm(atom4, acc_dk, rDS[None, None, kb],
                          tKrQ[None, None, kb], acc_dk)
            warpgroup.commit_group()
            # Graded wait: chains 2 AND 3 retire here (chain 2 because the
            # F1 point above no longer drains it under D2.1's `f1deep`
            # dial, chain 3 because segment G stages its accumulator), and
            # chain 4 stays in flight across the staging handshake below.
            # This is where `rP` dies.
            warpgroup.wait_group(1)
            # D0.9 consumer release, pipeline B -- the F1-site arrive moved
            # here by D2.1's `f1deep` dial.  Chain 2 was the last reader of
            # this trip's dO face and the dS segment was the last reader of
            # its delta row; both are dead now that chain 2 has retired, so
            # the slot is free -- still one segment ahead of pipeline A's
            # release below.
            _mb_arrive(bEB.toint())

            # =========== D0.8 segment G ===========================
            # ---- D0.11: dQ tile leaves through smem staging ----
            # D0.9 class 2, the asymmetric half.  B6 and B7 were whole-CTA
            # rendezvous; they are now this WG's own 160-participant credit
            # pair, so WG0 and WG1 hand their disjoint 16 KiB halves over
            # independently and neither waits for the other here.
            # `empty` BLOCKS (the producer's previous bulk reduce-add must
            # have finished READING this half before it is overwritten);
            # `full` does NOT block (report and walk on) -- that
            # asymmetry is what lets the consumer start the next trip while
            # the producer is still draining this one.
            _nb_sync(nb_empty, I32(STAGE_CREDIT))                  # B6
            cute.autovec_copy(dq_st, tSt_sD)      # 8x STS.128 per thread
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                                  space=cute.arch.SharedSpace.shared_cta)
            _nb_arrive(nb_full, I32(STAGE_CREDIT))                 # B7
            # Drain: chain 4 retires, so `rDS` and this trip's Q face are
            # both dead before the pipeline-A release below.
            warpgroup.wait_group(0)
            # D0.9 consumer release, pipeline A.  Chain 4 was the last
            # reader of this trip's Q face and the softmax segment the last
            # reader of its lse row; both are dead once chain 4 retires.
            _mb_arrive(bEA.toint())

        # ---- D0.9 class 2 handover to the epilogue -------------------
        # B8 used to close every trip; it now runs ONCE per work item and
        # means one thing only: the producer's last dQ write-back has
        # finished reading the staging face, so the dK/dV flush below may
        # alias it.
        cute.arch.barrier()                                        # B8

        # ---- D0.12: dV then dK flushed once per work item ----
        # Scalar spelling through the SAME `tSt_sD` partition as the dQ
        # site above (same byte order, no vectorisation).  The vectorised
        # `cute.autovec_copy` spelling at THESE two sites is the device
        # HANG carrier bisected by notes/probe_d05_staging_atom.py.
        for _i in cutlass.range_constexpr(cute.size(tSt_sD)):
            tSt_sD[_i] = dv_st[_i]
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                              space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier()                                        # B9
        cute.arch.barrier()                                        # B10

        for _i in cutlass.range_constexpr(cute.size(tSt_sD)):
            tSt_sD[_i] = dk_st[_i]
        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared,
                              space=cute.arch.SharedSpace.shared_cta)
        cute.arch.barrier()                                        # B11
        cute.arch.barrier()                                        # B12


# ------------------------------------------------------------- host side ------
def _make_storage(sK_l, sV_l, sQs_l, sdOs_l, sDSs_l):
    """The D0.5 buffer list, closed.  Field order IS the smem layout, and the
    total lands exactly on 199680 B:

        0     mbar (D0.9: 11 mbarriers, 88 B) + pad
              -> the `mbarrier + padding` cell, 1024 B, because the operand
                 faces' 1024-B alignment absorbs everything in it.
        1024  sK, sV                 single buffered  2 x 16384 B
        33792 sQ, sdO                2 stage          2 x 32768 B
        99328 sDS                    2 stage              65536 B
        164864 sStage (sdQ + the dK/dV epilogue alias) 32768 B
        197632 sLSE, sDel            2 stage          2 x  1024 B
        199680 end
    """
    @cute.struct
    class SharedStorage:
        # D0.5 `mbarrier + padding` cell, placed first so the 1024-B alignment
        # of the operand faces absorbs the residual.  D0.9 makes it ONE
        # MemRange of NMBAR Int64 rather than a field per barrier: `stage` is a
        # runtime value, so the barrier a trip uses has to be selected by a
        # runtime index, and a runtime-sliced sub-range's `.iterator` is the
        # certified way to get that address (see `_bar`).  Slots 0..7 are the
        # two pipelines' declared 8-barrier footprint; 8..10 are the K/V fill
        # barrier and the two private LSE landing barriers.
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, NMBAR], 8]
        sK: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(sK_l)], 1024]
        sV: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(sV_l)], 1024]
        sQ: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(sQs_l)], 1024]
        sdO: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(sdOs_l)], 1024]
        sDS: cute.struct.Align[cute.struct.MemRange[F16, cute.cosize(sDSs_l)], 1024]
        sStage: cute.struct.Align[cute.struct.MemRange[F32, STAGE_ELEMS], 1024]
        sLSE: cute.struct.Align[cute.struct.MemRange[F32, NSTAGE * BM], 128]
        sDel: cute.struct.Align[cute.struct.MemRange[F32, NSTAGE * BM], 128]

    return SharedStorage


def _build_objects():
    """Every host-side CuTe object MAIN and POST are parameterised by.

    Shape-independent (all of it is driven by the BM/BN/HD/NUM_WG constants),
    but it MUST be called from inside a `cute.jit` trace: `get_smem_store_op`
    and the `make_tiled_copy_*` builders emit IR and need an MLIR context.
    `_launch` and the stall-bisect probe both go through this one function, so
    a per-kernel launch cannot drift from the composed one.
    """
    # --- MMA flavours (D0.3): C(M,N) += A(M,K) . B(N,K) ---
    mma_s = sm90_utils.make_trivial_tiled_mma(          # chains 0,1
        F16, F16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        F32, atom_layout_mnk=(NUM_WG, 1, 1), tiler_mn=(64, BM))
    mma_kmn = sm90_utils.make_trivial_tiled_mma(        # chain 3 (A from smem)
        F16, F16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
        F32, atom_layout_mnk=(NUM_WG, 1, 1), tiler_mn=(64, HD))
    # D0.6 [SOURCE: RS]: chains 2 and 4 -- same shape, A operand out of RMEM.
    _rs_kw = dict(atom_layout_mnk=(NUM_WG, 1, 1), tiler_mn=(64, HD))
    _rs_kw[_A_SRC_KW] = warpgroup.OperandSource.RMEM
    mma_rs = sm90_utils.make_trivial_tiled_mma(         # chains 2,4
        F16, F16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
        F32, **_rs_kw)

    # --- D0.5 sdS write atom: register -> smem transposed `stmatrix`.  One
    #     LayoutEnum drives both call-sites: the store op that emits the
    #     instruction (STSM.16.MT88.4) and the C atom that groups the register
    #     values.  COL_MAJOR because the C fragment feeding it is m-major with
    #     M = KV rows.
    _E_DS = LayoutEnum.COL_MAJOR
    atom_ds = sm90_utils.get_smem_store_op(_E_DS, elem_ty_d=F16, elem_ty_acc=F32)
    r2s_ds = cute.make_tiled_copy_S(
        atom_ds,
        cute.make_tiled_copy_C_atom(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(_E_DS.is_m_major_c(), 4), F16),
            mma_s))

    # --- smem layouts.  Operand faces are stored the way the producer writes
    #     them (contiguous mode last) and read through transposed views. ---
    #     D2.3 closed on these two objects: they are the WHOLE swizzle dial
    #     (atom_hd -> sK/sV/sQ/sdO and thereby the TMA + WGMMA descriptors and
    #     the transposed read views; atom_bn -> sdS and thereby the transposed
    #     stmatrix partition and the chain-3/4 descriptor reads), so a mode
    #     change is one token here and cannot be half-applied downstream.
    #     Both resolve to K_SW128 (HD = 1024 b, BN = 2048 b contiguous) and
    #     ship there; the K_SW64 alternatives are certified admissible on
    #     device but not taken.  Re-open trigger = M2 gate line (2).
    atom_hd = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, F16, HD), F16)
    atom_bn = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, F16, BN), F16)

    # Single-stage faces.  K/V ARE this (CTA lifetime invariant); for Q/dO and
    # sdS these describe ONE stage and stay alive for two reasons: the TMA
    # descriptor addresses a single tile, and the transaction byte count below
    # is a per-tile quantity.
    sK_l = cute.tile_to_shape(atom_hd, (BN, HD), order=(0, 1))
    sV_l = cute.tile_to_shape(atom_hd, (BN, HD), order=(0, 1))
    sQ_l = cute.tile_to_shape(atom_hd, (BM, HD), order=(0, 1))
    sdO_l = cute.tile_to_shape(atom_hd, (BM, HD), order=(0, 1))
    sDS_l = cute.tile_to_shape(atom_bn, (BM, BN), order=(0, 1))   # (q, kv), kv contiguous

    # D0.5 stage counts: Q/dO 2 stage, sdS 2 stage.  A staged face is ONE
    # 3-mode `tile_to_shape` with the stage mode outermost, sliced at a dynamic
    # stage at the use site -- NOT an iterator offset plus a rebuilt view,
    # which does not survive the tracer.
    cq = cute.cosize(sQ_l)
    cds = cute.cosize(sDS_l)
    sQs_l = cute.tile_to_shape(atom_hd, (BM, HD, NSTAGE), order=(0, 1, 2))
    sdOs_l = cute.tile_to_shape(atom_hd, (BM, HD, NSTAGE), order=(0, 1, 2))
    sDSs_l = cute.tile_to_shape(atom_bn, (BM, BN, NSTAGE), order=(0, 1, 2))

    # transposed views (same bytes): colex coefficients of the write domain,
    # reordered to the read view's coordinate order, with the stage mode
    # carried at stride = single-stage cosize.
    sKt_l = cute.composition(sK_l, cute.make_layout((HD, BN), stride=(BN, 1)))
    sQts_l = cute.composition(
        sQs_l, cute.make_layout((HD, BM, NSTAGE), stride=(BM, 1, cq)))
    sdOts_l = cute.composition(
        sdOs_l, cute.make_layout((HD, BM, NSTAGE), stride=(BM, 1, cq)))
    sDSts_l = cute.composition(
        sDSs_l, cute.make_layout((BN, BM, NSTAGE), stride=(BM, 1, cds)))

    # D0.9 transaction byte counts, one per barrier.  The two operand pipelines
    # are A = Q + LSE and B = dO + delta, and D0.8's shadowing needs them
    # SEPARATE (a consumer that cannot wait for Q without also waiting for dO
    # has no window to hide dO's TMA in).  Only ONE of the four transfers is
    # not on its pipeline's own full barrier: the LSE vector lands on the
    # private `B_RL` barrier (ROW_BYTES) because the producer has to scale it
    # by log2e before stage A may be published.  delta needs no such detour and
    # rides `full_B` directly, so `tx_b` is a MIXED transaction (one
    # bulk-TENSOR dO face + one non-tensor bulk vector on one expect_tx).
    tx_kv = 2 * (cute.cosize(sK_l) + cute.cosize(sV_l))
    tx_a = 2 * cq
    tx_b = 2 * cute.cosize(sdO_l) + ROW_BYTES

    Storage = _make_storage(sK_l, sV_l, sQs_l, sdOs_l, sDSs_l)

    # --- D0.5 staging atom: ONE object pair feeds both the write side (MAIN's
    #     three flush sites) and the read side (POST).  The workspace byte
    #     order IS this TV's order.
    flat_l, tv_st = _make_flat_staging()

    return SimpleNamespace(
        mma_s=mma_s, mma_kmn=mma_kmn, mma_rs=mma_rs, r2s_ds=r2s_ds,
        atom_ds=atom_ds,
        sK_l=sK_l, sV_l=sV_l, sQ_l=sQ_l, sdO_l=sdO_l, sDS_l=sDS_l,
        sQs_l=sQs_l, sdOs_l=sdOs_l, sDSs_l=sDSs_l,
        sKt_l=sKt_l, sQts_l=sQts_l, sdOts_l=sdOts_l, sDSts_l=sDSts_l,
        tx_kv=tx_kv, tx_a=tx_a, tx_b=tx_b,
        Storage=Storage, flat_l=flat_l, tv_st=tv_st)


@cute.jit
def _launch(mQ, mK, mV, mO, mDO, mLSE, mDelta, mDQacc, mDKacc, mDVacc,
            mDQ, mDK, mDV,
            b: cutlass.Constexpr[int], hq: cutlass.Constexpr[int],
            hkv: cutlass.Constexpr[int], n: cutlass.Constexpr[int],
            scale_log2e: cutlass.Constexpr[float], scale: cutlass.Constexpr[float],
            stream):
    n_mblk = n // BM
    gqa = hq // hkv

    ob = _build_objects()

    _pre_kernel(mO, mDO, mDelta, mDQacc, scale, hq, n_mblk,
                b * hq * n).launch(
        grid=[n_mblk, hq, b], block=[PRE_THREADS, 1, 1], stream=stream)

    # D0.5 TMA descriptors.  Built here rather than in `_build_objects` because
    # a bulk-tensor atom is bound to the global tensor it addresses.  The smem
    # layout handed to each atom is the SAME object MAIN reads through, so the
    # descriptor's swizzle and the WGMMA descriptor's swizzle cannot drift.
    aQ, tQ = _tma_atom_4d(mQ, ob.sQ_l, (BM, HD), n, HD, hq, b)
    aK, tK = _tma_atom_4d(mK, ob.sK_l, (BN, HD), n, HD, hkv, b)
    aV, tV = _tma_atom_4d(mV, ob.sV_l, (BN, HD), n, HD, hkv, b)
    adO, tdO = _tma_atom_4d(mDO, ob.sdO_l, (BM, HD), n, HD, hq, b)

    _main_kernel(tQ, tK, tV, tdO, aQ, aK, aV, adO,
                 mLSE, mDelta, mDQacc, mDKacc, mDVacc,
                 ob.mma_s, ob.mma_kmn, ob.mma_rs, ob.r2s_ds, ob.atom_ds,
                 ob.tv_st, ob.flat_l,
                 ob.sK_l, ob.sV_l, ob.sQs_l, ob.sdOs_l, ob.sDSs_l,
                 ob.sKt_l, ob.sQts_l, ob.sdOts_l, ob.sDSts_l,
                 ob.Storage, n_mblk, hq, gqa, scale_log2e, scale,
                 ob.tx_kv, ob.tx_a, ob.tx_b).launch(
        grid=[n_mblk, hq, b], block=[MAIN_THREADS, 1, 1],
        min_blocks_per_mp=1, stream=stream)

    # POST reads the TV-ordered workspace through the same bijection, so its
    # block size must match the TV's thread layout (128 lanes x NUM_WG).
    # D3.2: two launches, not three -- dK and dV share the H_KV grid and are
    # done back to back by one block of `_post_kernel2`.
    _post_kernel(mDQacc, mDQ, ob.mma_kmn, ob.tv_st, n_mblk, hq, b * hq * n_mblk).launch(
        grid=[n_mblk, hq, b], block=[NUM_THREADS, 1, 1], stream=stream)
    _post_kernel2(mDKacc, mDK, mDVacc, mDV, ob.mma_kmn, ob.tv_st, n_mblk, hkv, b * hkv * n_mblk).launch(
        grid=[n_mblk, hkv, b], block=[NUM_THREADS, 1, 1], stream=stream)


_COMPILED = {}
_WORKSPACE = {}
_STREAMS = {}


def _cu_stream(raw):
    try:
        import cuda.bindings.driver as _drv
        return _drv.CUstream(raw)
    except Exception:
        pass
    try:
        from cuda import cuda as _drv2
        return _drv2.CUstream(raw)
    except Exception:
        return raw


def _stream():
    """D3.3 item (3), host half: the CUstream wrapper is built once per raw
    handle instead of once per call.  Keyed on the raw pointer rather than
    cached outright, so a caller that switches streams still gets its own.
    """
    raw = torch.cuda.current_stream().cuda_stream
    st = _STREAMS.get(raw)
    if st is None:
        st = _cu_stream(raw)
        _STREAMS[raw] = st
    return st


def _workspace(b, hq, hkv, n, d, device):
    """D3.3 items (1)(2)(3), device half.

    (1) dq_accum is `empty` -- PRE zeroes it on device (D3.1's 4 x f32 store
        stream); there is no host `zeros`/`zero_()` on it, ever.
    (2) dk/dv_accum are `zeros`, but backed by ONE contiguous (2, b, hkv, n, d)
        buffer whose two leading slices ARE dk_acc and dv_acc.  They accumulate
        across CTAs, so they must start at zero every call; through the pair
        that reset is ONE `zero_()` over 2 x 20 MiB rather than two over
        20 MiB each.  D3.4's reference account budgets ONE fill (0.016 ms).
    (3) the buffers are allocated once per shape and reused across calls, and
        their four `from_dlpack` descriptors are built once with them -- the
        storage they address never moves, so rebuilding them per call is pure
        host work.  Only the caller's own tensors get fresh descriptors.
    """
    key = (b, hq, hkv, n, d, device)
    ws = _WORKSPACE.get(key)
    if ws is None:
        pair = torch.zeros((2, b, hkv, n, d), dtype=torch.float32, device=device)
        tensors = (
            torch.empty((b, hq, n), dtype=torch.float32, device=device),
            torch.empty((b, hq, n, d), dtype=torch.float32, device=device),
            pair[0],
            pair[1],
        )
        ws = {
            "pair": pair,
            "tensors": tensors,
            "args": [from_dlpack(t, assumed_align=16) for t in tensors],
        }
        _WORKSPACE[key] = ws
    else:
        # dk/dv accumulate across CTAs -> must start at zero every call, and
        # the pair covers BOTH slices in one launch.
        ws["pair"].zero_()
    return ws


def flash_attn_bwd(q, k, v, o, do, lse):
    b, hq, n, d = q.shape
    hkv = k.shape[1]

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    ws = _workspace(b, hq, hkv, n, d, q.device)

    scale = 1.0 / math.sqrt(d)
    # 13 tensor arguments: 6 caller inputs, then the 4 cached workspace
    # descriptors (D3.3 item 3), then the 3 caller outputs.
    args = [
        from_dlpack(q, assumed_align=16), from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16), from_dlpack(o, assumed_align=16),
        from_dlpack(do, assumed_align=16), from_dlpack(lse, assumed_align=16),
    ] + ws["args"] + [
        from_dlpack(dq, assumed_align=16), from_dlpack(dk, assumed_align=16),
        from_dlpack(dv, assumed_align=16),
    ]
    consts = [b, hq, hkv, n, scale * LOG2E, scale]
    stream = _stream()

    key = (b, hq, hkv, n, d)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(_launch, *args, *consts, stream)
        _COMPILED[key] = fn
    # Constexpr params (`consts`) are baked into the compiled artifact by
    # cute.compile and must NOT be passed again to the compiled callable --
    # doing so is an arity error before any kernel runs.
    fn(*args, stream)
    return dq, dk, dv
