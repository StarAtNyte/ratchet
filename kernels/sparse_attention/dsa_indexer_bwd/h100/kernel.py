"""dsa_indexer_bwd kernel — the ONLY file the student edits.

Frozen contract (see eval/bench.py and README.md, do not change):

  dsa_indexer_bwd(index_q, weights, index_k, attn_score, index_score,
                  topk_indices, grad_loss, sm_scale=1.0, loss_coeff=1.0)
      -> (d_index_q, d_weights, d_index_k)

    index_q      : torch.Tensor (1, S_Q, H_IDX, D_IDX)  cuda bfloat16
    weights      : torch.Tensor (1, S_Q, H_IDX)         cuda bfloat16
    index_k      : torch.Tensor (1, S_KV, D_IDX)        cuda bfloat16
    attn_score   : torch.Tensor (1, S_Q, TOPK)          cuda float32  seam 'target'
    index_score  : torch.Tensor (1, S_Q, TOPK)          cuda float32  seam 'predict'
    topk_indices : torch.Tensor (1, S_Q, TOPK)          cuda int32    invalid family: -1
    grad_loss    : torch.Tensor ()                      cuda float32  incoming dLoss
    sm_scale     : float   1.0 (chain canonical: indexer scale folded into weights)
    loss_coeff   : float   grad_scale = loss_coeff / (B*S_Q)

    d_index_q : torch.Tensor (1, S_Q, H_IDX, D_IDX) cuda bfloat16
    d_weights : torch.Tensor (1, S_Q, H_IDX)        cuda bfloat16
    d_index_k : torch.Tensor (1, S_KV, D_IDX)       cuda bfloat16

  Semantics: gradient of -grad_loss·grad_scale·sum(attn_score·log(predict))
  w.r.t. (index_q, weights, index_k), where predict is recomputed internally
  from (index_q, weights, index_k, topk_indices) — the incoming index_score
  may be used as the recompute (they agree by the seam contract). The kernel
  MAY overwrite attn_score / index_score in place (the bench hands fresh
  copies per call). fp32 accumulation; d_index_k accumulates across rows in
  fp32 and casts bf16 on the way out; invalid slots contribute nothing.

The implementation must be composed CuTe DSL (cutlass.cute) code — WGMMA/TMA
etc. Delegating the math to torch, to gradient-tape machinery, or to any
reference library reachable in the image fails the harness (eval/bench.py
scans this file; the banned-token list is in its fence). JIT-compile once and
cache the compiled kernel across calls (the bench times repeated calls with
rotating buffers).

Structure (four launches per call, one stream, no events):
  K1 ScoreGrad (CuTe DSL, one CTA per query row) -> dI overwrites index_score
  zero the fp32 dK workspace (torch)
  K2 main (CuTe DSL, one CTA per query row, 384 threads = 1 copy-producer
     warpgroup at 56 registers + 2 mma-consumer warpgroups at 224; the
     producer gathers the runtime indexed K rows into a 3-slot mbarrier ring
     with cp.async.cg, the two consumers run three SS WGMMA chains per kv
     chunk: S / dK-transpose / dQ, exchange dS through an STSM face and
     ping-pong their chain issue over two publish-handoff named barriers;
     Q arrives by TMA box and dQ leaves by TMA box over the same smem face,
     dK leaves by cp.reduce.async.bulk into an fp32 workspace)
  cast the fp32 dK workspace to bf16 (torch)

Construction ledger. The D0.19 ladder is finished (4 of 4 steps, nothing
stubbed) and the whole-machine smoke certificate is in hand —
notes/probe_m0_iter4_smoke_cert.py, four rungs, all PASS: L (both entries
cute.compile at the campaign shape 32768/8192/1024), U (one CTA, per-output
verdicts + untouched-workspace-row check on the bulk-reduce addressing), X
(court geometry 4096x2048 topk 512, mixed tl + -1, one verdict per output),
S (campaign shape on replay-consistent operands: launch + complete + finite +
non-constant + sampled dq/dw), worst reported max_diff 0.002793.

The first M0 gate submission came back `correctness FAIL: [replay] outside
criteria`. The layer was K1's row sum, not the data flow or any layout:
against the harness oracle the three outputs read cos ~= 0.72-0.80 with
|student|/|oracle| ~= 8.3, while dI matched the T = 1 spelling of the D0.20
closed form exactly. The oracle differentiates
-sum(t * log(p.clamp_min(tiny))), and clamp_min passes no gradient below the
clamp; the campaign logits are peaked enough that ~90% of the VALID slots
underflow to p < FLT_MIN, so those slots drop out of both their own -t_k and
the row sum. K1 now carries that live mask (see P_LIVE_GT): T is the live
partial mass, not 1. Verified by notes/probe_m0_iter6_live_slot_T.py against
the frozen harness's own oracle and the gate's own criteria (cosine >= 0.98,
rms_rel <= 0.35) at BOTH judged points — court 4096x2048/512 and the campaign
replay point — every verdict line PASS, worst max_diff 0.0191577.

D0.20, second submission. No functional change since the amendment above; the
gate's prerequisite (both kernels lower and launch to completion at the
campaign shape) is carried by probe_m0_iter4_smoke_cert.py and re-covered, at
both judged geometries and with numerical verdicts, by
probe_m0_iter6_live_slot_T.py. What the first submission read on the NCU side,
for the record of what this one re-reads: kernel_count 4 /
dominant_time_pct 99.32, grid 32768x1x1, block 384x1x1, smem_dynamic 204800
with smem_static 0, regs_per_thread 168, cache_ops covering
LDGSTS.E.BYPASS.128 / UBLKRED.G.S.ADD.F32.RN / UTMALDG.2D / UTMASTG.2D,
2 publish-handoff named barriers at width 256, dram_bytes_read 8.166e8,
smem_wavefronts st+ld 4.61e8, bankconf st+ld 6.51e6 — against
spill_ld_sectors 8.28e7 (B6 line 3.7e7), tensor_pct 7.49 (C1 line 8) and
stall_long_sb 5.00 (C5 line 3.83) on a 25.94 ms dominant kernel.

M1, D1.1 (x) D1.4. Both edits sit on the consumer register account at the
unchanged 56/224 setmaxnreg split and target the two lines the M0 gate read
red. D1.1: the dW partial sums drop from one fp32 per C-fragment element (32)
to two, because the 64xN WGMMA C fragment covers exactly two M rows per thread
(value sub-mode (i//2)%2) — the epilogue does 2 red.shared.add.f32 per thread
at rows cS[0][0] / cS[2][0] instead of 32, and segment A gets 30 words back.
D1.4: the per-chunk dK reduce issue goes from one 128-deep chain on lane 0 to
four concurrent 32-deep chains, one per warp of the consumer WG (warp w owns
kv rows [w*32, w*32+32), its lane 0 drives them, each warp closes and drains
its own bulk group before the width-128 WG barrier that guards the staging
buffer). The four ranges partition [0, BN), so every kv row is still reduced
by exactly one thread and the arithmetic is unchanged.

M1, D1.2 + D1.3. Both edits are producer-side only at the unchanged 56/224
setmaxnreg split; no MMA chain, smem face, barrier, or consumer segment is
touched. (Both of the two paragraphs below are SUPERSEDED by the D0.8
supply-cadence paragraph at the end of this docstring, which reopened the
starting points they had moved off: the completion signal is back at
commit_group -> wait_group(0) -> fence -> arrive, and the row index is back on
B6 — read one group ahead, from between two gather slices, instead of eight
role-entry registers. They are kept here as the record of what D1.2's and
D1.3's levers do when they are next re-priced on top of the cadence contract.)

D1.2: cp.async completion signal changed from commit_group -> wait_group(0) ->
fence -> mbarrier_arrive to arrive-on-completion
(cp.async.mbarrier.arrive.noinc). Each producer thread now issues its 16 x
16 B cp.async.cg K-row gather (unchanged) plus a 4-byte cp.async.ca for the
per-row dI scalar (sDI[s][tidx], replacing the LDG->STS pair), then
fence_view_async_shared() + cp_async_mbarrier_arrive_noinc(full[s]). The
mbarrier init count stays at WG = 128 (one .noinc arrival per producer
thread). No commit_group / wait_group(0) in the loop body — the mbarrier
ring's empty guard still ensures the slot is never overwritten while consumers
are reading it, so up to 3 slots' gathers are in flight concurrently.

D1.3: lane-owned index registers. At role-entry (after setmaxregister_decrease
but before the chunk loop) each producer thread loads all n_chunk = 8 row
indices from topk_indices into local registers (ix_regs[j] = imax0(mIdx[row,
j*BN+tidx])). The per-chunk krow LDS from sIX is eliminated from the gather's
issue path; sIX (B6) is still written by the shared prelude so the consumers'
bulk-reduce destination reads from sIX are unaffected. The 8 additional index
registers fit inside the producer's 56-word budget.

M1, D1.5 + D1.6. Two encoding A/Bs on the two remaining places where a thread
waits on somebody else; neither moves a buffer, a barrier count, a chain or a
register quota (the 56/224 setmaxnreg split is unchanged, and both edits cost
scratch registers only).

D1.5: the K ring's 13 dynamic waits per CTA (8 consumer full[ch%3], 5 producer
empty[ch%3]) go from the suspending face (cute.arch.mbarrier_wait -> try_wait
with a pinned 10 ms suspendTimeHint and a compiler-generated sleep/wake retry
loop) to the non-blocking face: cute.arch.mbarrier_try_wait (one attempt, no
hint) driven by a caller-written while inside the @cute.jit helper
mbar_wait_spin, with the inline-PTX mbar_try_wait_i32 (mbarrier.try_wait.parity
+ selp on the shared-window address) standing in on builds that do not export
the arch face. The one-shot Q-box TMA wait in the shared prelude stays on the
blocking face. Arrive counts, init counts and phase parities are untouched —
this is the waiting side only.

D1.6: the epilogue dW merge goes from 512 smem reduce-adds per CTA (every
consumer thread publishing both of its C-fragment rows, 8 writers per address)
to a warp butterfly plus 128 (2 writers per address): two shfl.sync.bfly.b32
steps with lane masks 1 and 2 close the four lanes of each {4g..4g+3} group,
which by the 64xN C fragment layout hold partial sums of the same two M rows,
and lane 4g alone touches sDW. Same fp32 accumulation, same bf16 store.

Certification for both (notes/probe_m1_iter13_wait_and_dwmerge.py, rev 3, all
14 contracts PASS, worst metric 0.0192): F4 read the C-fragment coordinates off
partition_C with no MMA issued and confirmed the D1.6 lane map exactly (each
thread's 32 elements cover 2 M rows split by (i//2)%2 at 16*(lt//32)+(lt%32)//4
and +8; the 4 lanes of every group carry the same two rows; their N columns
partition [0,64) once). T ran the tiny geometry against the live-slot closed
form plus a per-h reading of d_weights (min per-h cosine and the least-squares
scale, which a flattened norm can survive); X and S ran court and campaign
geometry against the frozen oracle under the gate criteria. R launched the same
court inputs 12x on the real ring: every run met the criteria and d_index_q /
d_weights were bitwise identical run to run, which is the lost-acquire failure
mode a single launch can miss (d_index_k is excluded — its cross-row reduce-add
order is nondeterministic by D0.6).

D1.7, M1 Gate. Standalone row, no functional change: the code is byte-identical
to the D1.5+D1.6 tree, and this block is the ledger of what the gate reads. All
six M1 levers are in the tree, each landed and measured on its own commit off
the M0-gate tree (25.144 ms device):

  D1.1  56/224 setmaxnreg split, dW partial sums 32 -> 2 registers
  D1.2  cp.async completion signal -> cp.async.mbarrier.arrive.noinc
  D1.3  lane-owned producer index registers (8 per thread), sIX off the
        gather issue path
  D1.4  per-chunk dK bulk-reduce issue: one 128-deep chain -> four concurrent
        32-deep chains, one per consumer warp
  D1.5  the 13 dynamic K-ring waits per CTA -> non-blocking try_wait + a
        caller-written spin (waiting side only)
  D1.6  epilogue dW merge -> shfl.sync.bfly x2 + 128 smem reduce-adds

Last whole-machine readings on this tree (iteration 13 eval, sha 23bf30a0):
device median 12.6119 ms, correctness PASS, spill_ld_sectors 0 /
spill_st_sectors 0, tensor_pct 16.06, stall_long_sb 3.27, stall_barrier 2.01,
dram_bytes_read 8.161e8, smem_dynamic 204800 with smem_static 0, grid
32768x1x1, block 384x1x1, kernel_count 4. Against the M1 Gate checklist those
put the four mechanism lines inside their bands (spill == 0; tensor_pct >= 16;
stall_long_sb <= 3.83; dram_bytes_read <= 1.0e9) and the performance line
(<= 11.17 ms) outside by 1.13x. The consumer-side segments the M1 levers do
not reach are still bare-run: D0.16 step (7) (the S-chain drain plus the
per-element/STSM segment) and step (16) (the reduce issue) both run with the
tensor pipe idle, which is what D2.1 / D2.2 move into MMA flight and what D2.3
removes the top-of-chunk rendezvous for.

D1.8: the shape of the serial path inside each of D1.4's four issue chains.
The chain's 32 destination row indices are read out of B6 (sIX) in one
clustered pass at the head of the chain — 32 mutually independent LDS that
pipeline against each other — instead of one LDS immediately in front of each
issue, and the invalid-row family (idx < 0) is held off by the issue
instruction's own predicate (`setp.ge.s32 p, idx, 0; @p cp.reduce.async.bulk
...` in a single asm block, new op bulk_reduce_add_f32_pred) instead of the
`if ridx >= 0` forward branch. Between two adjacent UBLKRED sites there is now
only address arithmetic: no LDS, no ISETP, no BRA. The four chains, their
32-row partition of [0, BN), the per-lane commit_group and both drain sites are
D1.4's and unchanged, as are the numerics — predicate off == not issued, so an
invalid row still contributes exactly nothing and its destination address (an
Int64 computed but never dereferenced) is irrelevant. Cost: the elected lane
holds 32 index registers across the issue segment (D0.17 segment C, whose
non-accumulator line is 40 against the 224 consumer quota; the D1.1 spill == 0
line is re-read at the next gate).

M2, D2.1 (x) D2.2 — the two halves of one change to the dK write-back, which
share the staging buffer B4 and the meeting that guards it. Neither the buffer
list, the access atoms, the warp roles, the exchange face nor the three chain
shapes move: this is D0.16's step-16 POSITION and D0.12's payload GRANULARITY,
the two things D0.16 / D0.12 marked as M2's to move.

D2.2: the reduce payload goes from a 256 B d-half to a whole 512 B kv row.
Consumer c reduces rows [c*64, c*64+64) of the chunk end to end and its warp w
owns [c*64 + w*16, +16), so the chunk's 128 rows are still issued by exactly one
lane each — 128 instructions per chunk over eight 16-deep chains, where
D1.4/D1.8 left 256 over eight 32-deep ones. Same bytes, same one-reduce-per-row,
same predicate guard, same clustered index load at the head of each chain. The
consequence is that B4 stops being per-warpgroup scratch: the half of every row
I reduce was written by my partner. So the two per-WG 128-wide staging barriers
(BAR_ST0 / BAR_ST1) collapse into one 256-wide BAR_STG, which publishes "B4 has
been read out by BOTH warpgroups" before either may overwrite it.

D2.1: the issue site leaves the chunk tail. Chunk ch's reduces are issued in
chunk ch+1, right after chain0 is committed and before the wait that drains it,
so their acceptance latency runs under chain0's flight and the whole of segment
A instead of bare between the staging write and the chunk's tail. The source is
published by the top-of-chunk BAR_DS, which also absorbs the "both halves of B4
are written" duty that the deleted post-staging barrier carried; the drain moved
with the issue, so the BAR_STG meeting now sits immediately before segment B, a
full segment A plus the chain1/chain2 issue after the site. The last chunk has
no following chain0, so its issue lands in the epilogue — but its drain does
not, and the dQ pack plus the dW merge run in its flight. Blocking meetings per
chunk per consumer go 7 -> 6.

Certification (notes/probe_m2_iter16_reduce_move_and_payload.py): rung U scores
d_index_k's two d-halves SEPARATELY against the closed form on a geometry whose
indices never name a row >= 300 (a lost partner half reads scale 0 on one side,
a doubling reads 2, a destination stride slip scribbles into the untouched
rows); rungs T / X / S run the shipped kernel against the closed form and the
frozen oracle at tiny, court and campaign geometry; rung R launches the same
court inputs 12x, which is the only way to see D2.1's actual failure mode — the
partner rewriting B4 while a reduce issued a chunk earlier is still reading it —
since a race passes a single launch by luck. R3 holds dK's run-to-run drift to
reordering scale (1e-2 relative), the axis a wrong summand moves and a
reassociated fp32 sum does not.

D0.8, the supply-cadence contract (reopened). A group of gather is 16
LDGSTS.128 per thread by VOLUME; this fixes its ISSUE SHAPE, which the tree
had left at "one burst, three groups in flight". Two of the contract's three
axes land here, both producer-side only — no buffer, atom, barrier count,
chain, consumer segment or register quota moves.

P1, in-flight depth 1: the completion signal goes back from
cp.async.mbarrier.arrive.noinc (up to three slots in flight) to
commit_group -> wait_group(0) -> fence_view_async_shared -> mbarrier_arrive.
Same 128 arrivals on full[s], same parities, same ring; what changes is that
the producer thread drains its own group before it signals, so the SM is
offered one group's worth of LDGSTS at a time.

P2, four slices per group: the row is issued by four sites instead of one
burst, with the address sources between them. (The exact shape of a slice is
restated by the sector-efficiency block at the end of this docstring, which
re-cut the group along rows rather than along columns of one row.)

P3, the empty-arrive phase, is NOT in the contract's landed value. It was
built (release of slot ch moved out of chunk ch's tail and behind chunk ch+1's
reduce issue block, so a refill burst could not open inside the window the
consumer feeds UBLKRED into) and measured: 10.4091 ms device against the
10.0912 ms of P1 + P2, a 3.15% regression against a 0.50% noise band. The
contract's rule is "the value is the largest positively priced subset", so the
lever set closes at {one_in_flight, gather_sliced_4x} and the release goes back
where D0.16 step 17 (in D2.1's form) puts it: at the tail of chunk ch, right
after chain2 has been drained, which is the earliest point at which the slot is
provably read out. No fourth phase form is tried on this decision.

D0.9 rides P2. The whole row of 1024 indices still lands in B6 once in the
prelude (gmem LDG clustered there, before any empty-wait), and inside a group
the only address-source traffic is the B6 LDS for the next chunk's row plus the
single dI cp.async.ca — which is exactly the interleave P2 asks for, so the two
decisions share one construct rather than each having their own.

D0.12 / D0.16 / D0.17 / D0.19 are unchanged by the contract once P3 is out:
the reduce path still stages f32 into B4 and issues predicated whole-row
bulk reduce-adds (D2.2 payload) from four lanes per WG (D1.4) with the
destination indices clustered at the head of the chain (D1.8); the issue order
still runs segment B inside chain2's flight behind wait_group(1) and the reduce
block inside the next chunk's chain0 flight (D2.1); the segment budget still
holds S acc + dQ acc + working set with the pack terminating the S live range;
and the mount order's smoke certificate stands from the lineage's M0 gate.

D0.20, M0 Gate (reopened; standalone row, no functional change). The code is
byte-identical to the D0.8+D0.9+D0.12+D0.16+D0.17+D0.19 tree that closed the
reopened M0 body; this block is the ledger of what the gate re-reads on it.

Prerequisite, the whole-machine smoke certificate: both entries lower and
launch to completion at the campaign shape 32768/8192/1024 —
notes/probe_m0_iter4_smoke_cert.py rung L (compile) and rung S (launch +
complete + finite + non-constant on replay-consistent operands), re-covered
with numerical verdicts at both judged geometries by
notes/probe_m0_iter6_live_slot_T.py, and standing on this tree through the
D0.8 axis probe notes/probe_m2_iter21_gather_cadence.py.

All nineteen M0 decisions are in the tree, the thirteen the REOPEN kept as
built (D0.1-D0.7 less D0.8, D0.10, D0.11, D0.13, D0.14, D0.15, D0.18) and the
six redone against the patched milestone on this tree (D0.8 at the lever subset
{one_in_flight, gather_sliced_4x}, D0.9, D0.12, D0.16, D0.17, D0.19). Last
whole-machine readings, iteration 23 eval, sha eed719a4 (device median
10.088 ms, correctness PASS):

  B1 kernel_count 4, dominant_time_pct 98.21   B2 grid 32768x1x1
  B3 block 384x1x1     B4 smem_dynamic 204800 with smem_static 0
  B5 regs_per_thread 168                       B6 spill_ld_sectors 2.10e6
  B8 cache_ops covering LDGSTS.E.BYPASS.128 (with UBLKRED.G.S.ADD.F32.RN,
     UTMALDG.2D, UTMASTG.2D)
  C1 tensor_pct 19.72  C2 dram_bytes_read 8.160e8
  C3 smem_wavefronts st+ld 3.36e8              C5 stall_long_sb 5.32
  C4 bankconf st+ld 1.41e7 (6.03e6 st + 8.08e6 ld)
  D  stall_barrier 3.08, stall_mio 2.01, dram_pct 4.13, warp_active_pct 18.48;
     copy-producer commit axis DEPBAR {} LDGDEPBAR x1, supply axis
     interleaved + prefetched on four of five producer loops; two
     publish-handoff named barriers at width 256

Two of those sit outside their band and neither has moved since it was last
read on this lineage. C4's line is 1.0e7: the reading is 1.41e7, having gone
1.01e7 -> 1.41e7 across the D0.8 P1+P2 commit, against the ~4e8 that the
line's stated failure mode (an unpadded 4-way-conflicting B4 transpose write)
produces — B4's row stride is the contracted 528 B (STRIDE_F32 = 132 f32).
C5's line is 3.83 against 5.32, which has been above it on every eval this
lineage has run, M0 gate included (5.11 at 25.144 ms). The M1 lines the gate
does not judge read spill_ld_sectors 2.10e6 (D1.1's line is 0; it was 0 from
iteration 7 through 15 and returned at the D2.1+D2.2 commit) and tensor_pct
19.72 against M1's 16 and M2's 20.

D0.7 + D0.8 + D0.9, the sector-efficiency contract (second reopen). The
milestone patch adds a lane->row mapping requirement to D0.8, transcribes it
into D0.7's B1 write-side atom row, and puts the old mapping in the forbidden
list; D0.9 rides the same construct, so the three are one edit. Producer-side
only: no buffer, atom, barrier count, chain, consumer segment, register quota,
tile shape or completion protocol moves.

Shared budget and knobs across the three packaged decisions, since they are
one construct: the producer register quota (PROD_REGS = 56, D1.1's split,
unchanged — the mapping trades the one-group-ahead index register for the
per-slice clamps, so the live address-source set is <= 4 per slice rather than
8 at role entry); the K-ring smem budget (B1, 3 slots x 32 KiB, addresses and
SW128 swizzle untouched, so the D0.7 total stays 205,824 B); the ring's
synchronization knobs (NSTAGE = 3, full/empty arrival counts, P1 depth 1, P3
release point) all held at their pre-patch values; and the P2 slicing knob
NSLICE = 4, which the three now share as `4 slices x ROWS_PER_SLICE = 4 rows`
instead of `4 slices x SLICE_ELEMS = 32 elements of one row`. The per-thread
LDGSTS volume (16) is the invariant that lets all of this move at once.

The mapping. A K row is D_IDX * 2 = 256 B and a 128-bit cp.async moves 16 B,
so covering a row takes ROW_LANES = 16 lanes. The tree had been running the
other assignment since iteration 1: one thread owned one whole row and walked
it with 16 x 16 B, which puts lane L and lane L+1 on addresses 256 B apart —
each lane covers half of a 32 B sector and no lane covers the other half.
Under `.cg` (which bypasses L1) the missing half is a second L2 request, so
the L2 read-sector count doubles for the same bytes. build_row_copy's tiler is
now the whole row, make_tiled_copy_tv(atom, layout(ROW_LANES = 16),
layout(ROW_ELEMS = 8)): lane%16 selects the 16 B column, a warp issues two
whole rows, the producer WG issues eight, and lane `col16` of team
`tidx//16` owns the ROWS_PER_TEAM = 16 rows {i*8 + team}. Per-thread LDGSTS
volume, byte count, `.cg` qualifier, swizzled destination and the numbers
gathered are all unchanged — this is a re-partition of the same 16,384 copies
per CTA per chunk-set.

P2 re-cut along the new mapping: a group is NSLICE = 4 slices of
ROWS_PER_SLICE = 4 rows (16 rows, 16 copies per thread). A slice opens with
its own four B6 index LDS — the 16 lanes of a team read one address, so this
is an smem broadcast, 4 warp-level LDS per slice and 64 per group per WG —
followed by the imax0 clamps and then that slice's four whole-row copies. The
dI cp.async.ca and its 64-bit address arithmetic sit between slices 0 and 1.
The one-group-ahead index register that D1.3 had left in place is gone: with
16 rows per lane per group the index has to be read inside the group, which is
what the patched D0.8 asks for ("逐片 LDS 装载、与 LDGSTS 交织"), and it keeps
the address-source loads interleaved with the copies rather than clustered at
the top. P1 (depth 1: commit_group -> wait_group(0) -> fence -> arrive) and
P3's disposition (the empty-arrive stays at the tail of chunk ch) are
unchanged from the previous reopen.

D0.9 is unchanged in substance: the whole row of 1024 indices still lands in
B6 once in the prelude, and inside a group the only address-source traffic is
B6 LDS plus the single dI cp.async.ca. What changed is the count (16 LDS per
thread per group instead of 1) and the fact that they are broadcasts.

Certification (notes/probe_m2_iter26_gather_row16.py): rung G runs the shipped
producer body verbatim against a reader warpgroup that reads the slot back
through the same swizzled logical view the WGMMA descriptors use, over four
chunks on three slots so the ring turns behind an empty wait — G1 holds every
gathered row to index_k[max(idx,0)] EXACTLY (a wrong TV bijection, a swapped
team/column role, or a swizzle that does not survive the row slicing all move
it off zero, and none of them fault), G2 holds the dI scalar that rides the
same group. Rungs T / X / S then run the whole machine at tiny, court and
campaign geometry.
"""

from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.nvgpu import warp as nvwarp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

H_IDX = 64                 # one WGMMA M
D_IDX = 128
BN = 128                   # kv rows per chunk
NSLICE = 4                 # D0.8 P2: gather slices per group (>= 4 required)
HALF = 64                  # d-half owned by one consumer in the dK staging write
RPW = BN // 8              # D2.2: kv rows one issuing lane owns (2 WG x 4 warps)
WG = 128                   # threads per warpgroup
# D0.8 sector-efficiency contract.  A K row is 256 B; a 128-bit cp.async moves
# 16 B; so ROW_LANES = 16 CONSECUTIVE lanes cover one whole row and every 32 B
# sector the instruction touches is fully used.  A warp is therefore two rows
# per issue, the producer WG is eight, and one lane owns BN/ROW_TEAMS = 16 rows
# — the same 16 LDGSTS per thread per group as before, re-shaped.
ROW_LANES = 16                    # lanes co-gathering one 256 B K row
ROW_ELEMS = D_IDX // ROW_LANES    # 8 bf16 = 16 B = one cp.async per lane per row
ROW_TEAMS = WG // ROW_LANES       # 8 rows issued per producer-WG instruction
ROWS_PER_TEAM = BN // ROW_TEAMS   # 16 rows owned by one lane per group
ROWS_PER_SLICE = ROWS_PER_TEAM // NSLICE   # 4 rows (and 4 index LDS) per slice
NUM_THREADS = 384          # 1 producer WG + 2 consumer WG
K1_THREADS = 128
# A topk slot carries a gradient path only where the loss's log(p) term is
# live, i.e. p >= FLT_MIN (smallest normal fp32, 1.1754943508222875e-38); the
# -inf-masked and underflowed slots (p subnormal or 0) contribute neither
# their own -t_k nor their t_k to the row sum T.  See K1.  The constant below
# is the largest fp32 strictly under FLT_MIN (the biggest subnormal), so the
# single `p > P_LIVE_GT` compare is exactly `p >= FLT_MIN` — no fp32 lies
# between the two.
P_LIVE_GT = 1.1754942106924411e-38
NSTAGE = 3                 # K ring slots (producer leads the consumers by 2)
STRIDE_F32 = 132           # dK staging row stride: 512 B payload + 16 B pad
PROD_REGS = 56             # 56*128 + 224*256 = 64512 <= 65536
CONS_REGS = 224
BAR_DS = 3                 # symmetric rendezvous of the two consumer WGs
BAR_DS_W = 256
BAR_C1T0 = 4               # publish-handoff c1 -> c0 (c1 arrives, c0 syncs)
BAR_C0T1 = 5               # publish-handoff c0 -> c1 (c0 arrives, c1 syncs)
BAR_STG = 6                # dK staging guard, 256 wide (D2.2 makes B4 shared)

F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32

_mk_rmem = getattr(cute, "make_rmem_tensor", None) or getattr(cute, "make_fragment")


def _retile(thr_copy, frag):
    """Value-order view of a C fragment for a tiled copy (no instruction).

    The DSL member is `retile`; the fallback keeps the call site alive on
    builds that spell it `retile_S`.
    """
    fn = (getattr(thr_copy, "retile", None)
          or getattr(thr_copy, "retile_S", None))
    return fn(frag)


def _seg(site, row=0):        # timeline marker hook (no-op in the shipped tree)
    return None


# --------------------------------------------------------------------------
# PTX escapes (DSL has no select / scalar-atomic / non-bulk cp.async group /
# named-barrier spelling)
# --------------------------------------------------------------------------

@dsl_user_op
def select_f32(cond, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """cond != 0 ? a : b   (cond int32) -> one FSEL, no region."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(cond).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.f32 $0, $2, $3, p;\n}",
        "=f,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def select_f32_gt(x, y, a, b, *, loc=None, ip=None) -> cutlass.Float32:
    """x > y ? a : b   (all fp32) -> FSETP + FSEL, no region."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip),
         cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.gt.f32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,f,f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def imax0(x, *, loc=None, ip=None) -> cutlass.Int32:
    """max(x, 0) — clamps the -1 invalid index family to a legal row."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(x).ir_value(loc=loc, ip=ip)],
        "max.s32 $0, $1, 0;", "=r,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def bulk_reduce_add_f32(dst_addr, src_addr, nbytes, *, loc=None, ip=None):
    """smem -> gmem fp32 bulk reduce-add: one 256 B half row per instruction.

    dst: Int64 global address, src: Int32 shared::cta window address, both
    16 B aligned. Exactly ONE thread may issue a given copy — a second issuing
    thread is a silent doubling, not a fault. The group is closed with the
    A-tier `cp_async_bulk_commit_group` and drained with
    `cp_async_bulk_wait_group(0, read=True)` before the staging buffer is
    rewritten (hw_kb/cutedsl/sm90/s2g_bulk_reduce.md gotcha 5).
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def bulk_reduce_add_f32_pred(dst_addr, src_addr, nbytes, idx,
                             *, loc=None, ip=None):
    """`bulk_reduce_add_f32` with the invalid-row family held off by an
    instruction predicate instead of a branch (D1.8).

    `setp.ge.s32 p, idx, 0` and `@p cp.reduce...` live in the same asm block,
    so the guard costs no basic block: the issue path between two adjacent
    UBLKRED sites carries only address arithmetic, and the predicate does not
    have to be resolved before the *next* site can be reached. A branch here
    would instead stall the warp on its own `LDS -> ISETP -> BRA` chain, and
    ptxas will not hoist an issue across a basic-block boundary.

    Numerics are unchanged from the branched form: predicate off == not
    issued, so an invalid row (`idx < 0`) contributes exactly nothing and its
    (never dereferenced) destination address is irrelevant.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int64(dst_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
         cutlass.Int32(idx).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "setp.ge.s32 p, $3, 0;\n\t"
        "@p cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
        "[$0], [$1], $2;\n}",
        "l,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def shfl_bfly_f32(v, lane_mask, *, loc=None, ip=None) -> cutlass.Float32:
    """shfl.sync.bfly.b32: trade `v` with the lane whose id differs from mine
    in the bits of `lane_mask` (full membermask, clamp 31 -> whole warp).

    Marked side-effecting so the exchange is not hoisted across the predicate
    that follows it; the whole warp must reach it (`shfl.sync`).
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(v).ir_value(loc=loc, ip=ip),
         cutlass.Int32(lane_mask).ir_value(loc=loc, ip=ip)],
        "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;", "=f,f,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def red_shared_add_f32(addr, val, *, loc=None, ip=None):
    """smem fp32 reduce-add (per-thread). addr: Int32 shared-window address."""
    llvm.inline_asm(
        None,
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip),
         cutlass.Float32(val).ir_value(loc=loc, ip=ip)],
        "red.shared.add.f32 [$0], $1;", "r,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None):
    """Close the calling thread's outstanding cp.async into one group."""
    llvm.inline_asm(
        None, [], "cp.async.commit_group;", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait_group(n, *, loc=None, ip=None):
    """Wait until at most n of the calling thread's cp.async groups are in flight."""
    llvm.inline_asm(
        None, [], "cp.async.wait_group %d;" % int(n), "~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def named_bar_sync(bar_id, count, *, loc=None, ip=None):
    """Blocking rendezvous of `count` threads on named barrier `bar_id`."""
    llvm.inline_asm(
        None, [], "bar.sync %d, %d;" % (int(bar_id), int(count)), "~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def named_bar_arrive(bar_id, count, *, loc=None, ip=None):
    """Non-blocking arrival on `bar_id` — the publish half of a handoff pair.

    `bar.arrive` counts toward the same width as `bar.sync`, so a publisher
    that runs two arrivals ahead of its partner's sync would satisfy the
    barrier by itself; the D0.15 ping-pong is built so that each arrival is
    consumed by one partner sync before the next arrival can be reached.
    """
    llvm.inline_asm(
        None, [], "bar.arrive %d, %d;" % (int(bar_id), int(count)), "~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def cp_async_ca_4b(dst_smem, src_global, *, loc=None, ip=None):
    """4-byte cp.async.ca: one f32 element from gmem into smem (.ca = cache-all).

    dst_smem  : Int32 shared::cta window address (byte-addressed)
    src_global: Int64 global byte address
    Joins the calling thread's open cp.async group.  Close with
    cp_async_mbar_arrive_noinc_ptx (D1.2 arrive-on-completion path) or with
    cp_async_commit_group + cp_async_wait_group.
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst_smem).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src_global).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared::cta.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def lds_i32(addr, *, loc=None, ip=None) -> cutlass.Int32:
    """One `ld.shared.b32`, pinned where it is written.

    This is the address-source load D0.8 P2 places BETWEEN two gather slices:
    the next chunk's row index, read out of B6. It is marked side-effecting so
    it keeps its position among the surrounding cp.async issues instead of
    being sunk to the first use — the point of the load is not its value (it
    is needed a whole group later) but the gap it opens in the LSU's
    acceptance queue.

    `addr`: Int32 shared::cta window address, as for the mbarrier escapes.
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0, [$1];", "=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def cp_async_mbar_arrive_noinc_ptx(mbar_addr, *, loc=None, ip=None):
    """cp.async.mbarrier.arrive.noinc: all prior async copies are committed to the
    mbarrier without incrementing the expect-tx counter.  With this form the
    mbarrier's init count == the number of calling producer threads (WG = 128).

    `mbar_addr`: Int32 shared::cta window address — an mbarrier lives in smem,
    so the operand is a 32-bit window address (`Pointer.toint()`), not a
    generic 64-bit pointer, which the DSL refuses to convert at all
    (`ptr<i64, smem> to integer conversion is not supported`).
    """
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def mbar_try_wait_i32(mbar_addr, phase, *, loc=None, ip=None) -> cutlass.Int32:
    """ONE non-blocking attempt at a phase-parity wait -> 1 satisfied, 0 not.

    This is the non-suspending face of the same barrier `mbarrier_wait` uses:
    same acquire semantics, same parity, but it returns instead of putting the
    warp to sleep, and it carries no suspendTimeHint. The retry loop belongs to
    the caller (`mbar_wait_spin`); `has_side_effects=True` plus the memory
    clobber keep the attempt inside its own loop body, since a hoisted attempt
    would spin forever on a stale register.

    `mbar_addr`: Int32 shared::cta window address, as for the arrive escape
    above.
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Int32(mbar_addr).ir_value(loc=loc, ip=ip),
         cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\t"
        "selp.b32 $0, 1, 0, p;\n}",
        "=r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


# Prefer the first-class cute.arch spellings when the build exports them; both
# fall back to the PTX escape above (same semantics, same instruction), which
# takes the shared-window address rather than the pointer.
_arch_arrive_noinc = getattr(cute.arch, "cp_async_mbarrier_arrive_noinc", None)
_arch_try_wait = getattr(cute.arch, "mbarrier_try_wait", None)

if _arch_arrive_noinc is not None:
    _mbar_arrive_noinc = _arch_arrive_noinc
else:
    @cute.jit
    def _mbar_arrive_noinc(mbar_ptr):
        cp_async_mbar_arrive_noinc_ptx(mbar_ptr.toint())


# D1.5 arm (e): drop-in for `cute.arch.mbarrier_wait(bar, phase)`.
#
# The blocking face lowers to try_wait with a pinned 10 ms suspendTimeHint and
# a compiler-generated retry loop that sleeps and wakes the warp on every
# unsatisfied attempt (SASS `NANOSLEEP`); that wake-up latency is paid once per
# wait and shows up in no throughput metric. Here the warp stays resident and
# retries itself. Arrive/init counts and parities are untouched — this edit is
# on the waiting side only (hw_kb/cutedsl/sm90/mbarrier_wait_encoding.md gotcha 8), and
# the `while` lives inside a @cute.jit scope so the dynamic Boolean is traced as
# a loop rather than folded (gotcha 5).
if _arch_try_wait is not None:
    @cute.jit
    def mbar_wait_spin(mbar_ptr, phase):
        ok = _arch_try_wait(mbar_ptr, phase)
        while ok == 0:
            ok = _arch_try_wait(mbar_ptr, phase)
else:
    @cute.jit
    def mbar_wait_spin(mbar_ptr, phase):
        addr = mbar_ptr.toint()
        ok = mbar_try_wait_i32(addr, phase)
        while ok == 0:
            ok = mbar_try_wait_i32(addr, phase)


# --------------------------------------------------------------------------
# host-side builders (IR builders: must run inside a @cute.jit)
# --------------------------------------------------------------------------

def build_smem_layouts():
    """(write view, transposed read view) for the three bf16 smem faces.

    Every face is allocated k-major (its last logical mode contiguous); the
    transposed view is a host-side composition on the layout, so the same
    bytes are read MN-major by the consumer that needs it (no second copy).
    """
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, D_IDX), BF16)
    atom_kv = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BF16, BN), BF16)
    q_l = cute.tile_to_shape(atom, (H_IDX, D_IDX), order=(0, 1))
    k_l = cute.tile_to_shape(atom, (BN, D_IDX), order=(0, 1))
    ds_l = cute.tile_to_shape(atom_kv, (H_IDX, BN), order=(0, 1))
    qt_l = cute.composition(q_l, cute.make_layout((D_IDX, H_IDX), stride=(H_IDX, 1)))
    kt_l = cute.composition(k_l, cute.make_layout((D_IDX, BN), stride=(BN, 1)))
    dst_l = cute.composition(ds_l, cute.make_layout((BN, H_IDX), stride=(H_IDX, 1)))
    return q_l, k_l, ds_l, qt_l, kt_l, dst_l


def build_mmas():
    """The three per-chunk SS WGMMA chains (A and B both smem descriptors)."""
    km = warpgroup.OperandMajorMode.K
    mn = warpgroup.OperandMajorMode.MN
    mma_s = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, km, km, F32, (1, 1, 1), tiler_mn=(H_IDX, 64))
    mma_k = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, mn, mn, F32, (1, 1, 1), tiler_mn=(64, BN))
    mma_q = sm90_utils.make_trivial_tiled_mma(
        BF16, BF16, km, mn, F32, (1, 1, 1), tiler_mn=(H_IDX, 64))
    return mma_s, mma_k, mma_q


def build_row_copy():
    """ROW_LANES lanes co-gather one whole 256 B K row (D0.8 sector contract).

    The tiler is one K row: ROW_LANES = 16 threads x ROW_ELEMS = 8 bf16, so
    lane%16 selects the 16 B column of the row and a warp issues two whole rows
    per instruction. Every 32 B L2 sector the instruction touches is covered by
    two adjacent lanes and used whole; under the previous mapping (one thread
    owns one row, 16 x 16 B down that row) a lane took only half of each sector,
    and since `.cg` bypasses L1 the other half was fetched again by the next
    instruction — the L2 read-sector count doubled.

    A lane owns ROWS_PER_TEAM = 16 rows of the chunk, issued as NSLICE slices of
    ROWS_PER_SLICE = 4 rows, so the group still has four issue sites with
    address-source loads between them (D0.8 P2) and the per-thread LDGSTS volume
    is unchanged at 16.

    `.cg` (LoadCacheMode.GLOBAL) bypasses L1 — with ~200 KB of smem there is
    no L1 left to allocate lines in, and every `.ca` LDGSTS queues on that
    allocation (hw_kb/cutedsl/sm90/cp_async.md, cache qualifier section).
    """
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF16,
        num_bits_per_copy=128)
    return cute.make_tiled_copy_tv(atom,
                                   cute.make_layout(ROW_LANES),
                                   cute.make_layout(ROW_ELEMS))


def build_ds_store(mma_s):
    """acc -> dS exchange face: STSM x4, non-transposing, K-major (kv) face.

    The face is bf16 with kv contiguous, so the NON-transposing stmatrix member
    is the one that matches; x4 writes a thread's four 8x8 matrices in one
    instruction. The tiled copy is built off the MMA's own C partitioning, so
    `thr.retile(accS)` hands the C fragment straight to the store with no
    intermediate permutation.
    """
    stsm = cute.make_copy_atom(nvwarp.StMatrix8x8x16bOp(False, 4), BF16)
    return cute.make_tiled_copy_S(stsm,
                                  cute.make_tiled_copy_C_atom(stsm, mma_s))


def build_tma(mQc, mDQc, q_l):
    """The two 64h x 128d bf16 box atoms: Q load (G2S) and dQ store (S2G).

    Both gmem sides are the (s_q*H_IDX, D_IDX) flattening of the row-block
    tensors, so one (H_IDX, D_IDX) tile == one query row's block, and both
    ride the same swizzled smem layout `q_l` (the dQ staging reuses sQ, which
    is dead once the last chunk's WGMMA have retired).
    """
    tma_q, mQt = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mQc, q_l, (H_IDX, D_IDX),
        num_multicast=1)
    tma_dq, mDQt = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mDQc, q_l, (H_IDX, D_IDX))
    return tma_q, mQt, tma_dq, mDQt


def k1_storage():
    @cute.struct
    class K1Storage:
        sred: cute.struct.Align[cute.struct.MemRange[F32, K1_THREADS], 16]
    return K1Storage


def k2_storage(q_l, k_l, ds_l, topk):
    """B1 K ring (3 x 32 KiB) / B2 Q / B3 dS / B4 dK staging / B5 dI slot ring
    / B6 index row.

    B9 = one contiguous mbarrier array: [0..2] full, [3..5] empty, [6] Q-TMA.
    Budget: 16 + 96 + 16 + 66 + 1.5 + 4 + 0.4 KiB = 204,728 B dynamic smem,
    i.e. one CTA per SM (D0.7).
    """
    @cute.struct
    class K2Storage:
        sQ: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(q_l)], 1024]
        sK0: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(k_l)], 1024]
        sK1: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(k_l)], 1024]
        sK2: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(k_l)], 1024]
        sDS: cute.struct.Align[cute.struct.MemRange[BF16, cute.cosize(ds_l)], 1024]
        sST: cute.struct.Align[cute.struct.MemRange[F32, BN * STRIDE_F32], 1024]
        sDI0: cute.struct.Align[cute.struct.MemRange[F32, BN], 16]
        sDI1: cute.struct.Align[cute.struct.MemRange[F32, BN], 16]
        sDI2: cute.struct.Align[cute.struct.MemRange[F32, BN], 16]
        sIX: cute.struct.Align[cute.struct.MemRange[I32, topk], 16]
        sWT: cute.struct.Align[cute.struct.MemRange[BF16, H_IDX], 16]
        sDW: cute.struct.Align[cute.struct.MemRange[F32, H_IDX], 16]
        mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64,
                                                     2 * NSTAGE + 1], 16]
    return K2Storage


# --------------------------------------------------------------------------
# K1 — ScoreGrad: dI = c * (p*T - t) on live slots, 0 elsewhere, in place over
# index_score.  Live = (idx >= 0) & (p >= FLT_MIN); T = sum of t over the live
# slots of the row.
# --------------------------------------------------------------------------

@cute.kernel
def _k1_score_grad(mT: cute.Tensor, mP: cute.Tensor, mIdx: cute.Tensor,
                   mG: cute.Tensor, scale: cutlass.Float32,
                   slots: cutlass.Constexpr[int],
                   Storage: cutlass.Constexpr[Callable]):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sred = st.sred.get_tensor(cute.make_layout(K1_THREADS))

    tvals = []
    pvals = []
    zero = cutlass.Float32(0.0)
    plive = cutlass.Float32(P_LIVE_GT)
    part = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(slots):
        k = j * K1_THREADS + tidx
        iv = mIdx[(0, row, k)]
        p0 = mP[(0, row, k)]
        # A slot is live only where the loss term t_k*log(p_k) still has a
        # gradient path: idx_k >= 0 AND p_k >= FLT_MIN.  Dead slots drop out
        # of BOTH the row sum T and their own -t_k, so T is the live-slot
        # partial mass, not 1.
        live = select_f32_gt(p0, plive, cutlass.Float32(1.0), zero)
        live = select_f32(iv + 1, live, zero)
        tv = live * mT[(0, row, k)]
        pv = live * p0
        tvals.append(tv)
        pvals.append(pv)
        part = part + tv

    sred[tidx] = part
    cute.arch.sync_threads()
    step = K1_THREADS // 2
    for _ in cutlass.range_constexpr(7):
        if tidx < step:
            sred[tidx] = sred[tidx] + sred[tidx + step]
        cute.arch.sync_threads()
        step = step // 2

    tsum = sred[0]
    c = mG[(0,)] * scale
    for j in cutlass.range_constexpr(slots):
        k = j * K1_THREADS + tidx
        mP[(0, row, k)] = c * (pvals[j] * tsum - tvals[j])


# --------------------------------------------------------------------------
# K2 — main kernel: warp-specialized, 3-slot K ring, three SS WGMMA chains
# --------------------------------------------------------------------------

@cute.kernel
def _k2_main(tma_q: cute.CopyAtom, mQt: cute.Tensor,
             mW: cute.Tensor, mK: cute.Tensor, mDI: cute.Tensor,
             mIdx: cute.Tensor,
             tma_dq: cute.CopyAtom, mDQt: cute.Tensor, mDW: cute.Tensor,
             mWs: cute.Tensor,
             mma_s: cute.TiledMma, mma_k: cute.TiledMma, mma_q: cute.TiledMma,
             tiled_ld: cute.TiledCopy, r2s: cute.TiledCopy,
             q_l: cute.ComposedLayout, k_l: cute.ComposedLayout,
             ds_l: cute.ComposedLayout, qt_l: cute.ComposedLayout,
             kt_l: cute.ComposedLayout, dst_l: cute.ComposedLayout,
             st_l: cute.Layout,
             n_chunk: cutlass.Constexpr[int],
             Storage: cutlass.Constexpr[Callable]):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(Storage)
    sQ = st.sQ.get_tensor(q_l.outer, swizzle=q_l.inner)
    sQt = st.sQ.get_tensor(qt_l.outer, swizzle=qt_l.inner)
    sK = [st.sK0.get_tensor(k_l.outer, swizzle=k_l.inner),
          st.sK1.get_tensor(k_l.outer, swizzle=k_l.inner),
          st.sK2.get_tensor(k_l.outer, swizzle=k_l.inner)]
    sKt = [st.sK0.get_tensor(kt_l.outer, swizzle=kt_l.inner),
           st.sK1.get_tensor(kt_l.outer, swizzle=kt_l.inner),
           st.sK2.get_tensor(kt_l.outer, swizzle=kt_l.inner)]
    sDS = st.sDS.get_tensor(ds_l.outer, swizzle=ds_l.inner)
    sDSt = st.sDS.get_tensor(dst_l.outer, swizzle=dst_l.inner)
    sST = st.sST.get_tensor(st_l)
    sDI = [st.sDI0.get_tensor(cute.make_layout(BN)),
           st.sDI1.get_tensor(cute.make_layout(BN)),
           st.sDI2.get_tensor(cute.make_layout(BN))]
    sIX = st.sIX.get_tensor(cute.make_layout(n_chunk * BN))
    sWT = st.sWT.get_tensor(cute.make_layout(H_IDX))
    sDW = st.sDW.get_tensor(cute.make_layout(H_IDX))

    zero = cutlass.Float32(0.0)
    _seg("all.prologue.entry.begin")

    # --- shared prelude: every barrier is initialized before the role split --
    mbar = st.mbar.data_ptr()
    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbar + 0, WG)      # full[0]: 128 producer arrivals
            cute.arch.mbarrier_init(mbar + 1, WG)      # full[1]
            cute.arch.mbarrier_init(mbar + 2, WG)      # full[2]
            cute.arch.mbarrier_init(mbar + 3, 2)       # empty[0]: 1 per consumer WG
            cute.arch.mbarrier_init(mbar + 4, 2)       # empty[1]
            cute.arch.mbarrier_init(mbar + 5, 2)       # empty[2]
            cute.arch.mbarrier_init(mbar + 6, 1)       # Q box (expect-tx)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # --- Q box in / dQ box out: same smem face, same swizzled layout --------
    qbar = mbar + 2 * NSTAGE
    gQ = cute.zipped_divide(mQt, (H_IDX, D_IDX))
    tQsQ, tQgQ = cpasync.tma_partition(tma_q, 0, cute.make_layout(1),
                                       cute.group_modes(sQ, 0, 2), gQ)
    q_src = tQgQ[(None, (row, 0))]
    gDQ = cute.zipped_divide(mDQt, (H_IDX, D_IDX))
    tDsD, tDgD = cpasync.tma_partition(tma_dq, 0, cute.make_layout(1),
                                       cute.group_modes(sQ, 0, 2), gDQ)
    dq_dst = tDgD[(None, (row, 0))]

    _seg("producer.qtma.issue.begin")
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_q)
        cpasync.prefetch_descriptor(tma_dq)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                qbar, cute.size_in_bytes(BF16, q_l))
        cute.copy(tma_q, q_src, tQsQ, tma_bar_ptr=qbar)
    _seg("producer.qtma.issue.end")

    if tidx < H_IDX:
        sWT[tidx] = mW[(0, row, tidx)]
        sDW[tidx] = zero

    # --- B6: the whole row of indices lands in smem once, in the prelude ----
    # after this the producer's per-group row addresses come from LDS, never
    # from a gmem round trip on the gather's critical path (D0.9).
    _seg("all.prologue.idx.begin")
    if tidx < WG:
        for j in cutlass.range_constexpr(n_chunk):
            sIX[j * WG + tidx] = mIdx[(0, row, j * WG + tidx)]
    _seg("all.prologue.idx.end")

    _seg("all.prologue.qtma.wait.begin")
    cute.arch.mbarrier_wait(qbar, 0)
    _seg("all.prologue.qtma.wait.end")

    cute.arch.fence_view_async_shared()
    _seg("all.prologue.bar.begin")
    cute.arch.sync_threads()
    _seg("all.prologue.bar.end")

    # ---------------------------------------------------------------- WG0 --
    # copy-producer: 16 lanes per gathered K row, 3-slot ring, 3-beat warm-up
    if tidx < WG:
        # first statement of the role branch: the shared prelude above ran
        # under the entry budget, everything below runs on 56 registers
        cute.arch.setmaxregister_decrease(PROD_REGS)
        _seg("producer.entry.begin")
        # D0.8 sector contract: lane `col16` owns the 16 B column [col16*8,
        # +8) of every row this team gathers, and the ROW_TEAMS = 8 teams
        # interleave over the chunk's rows (team t takes rows t, 8+t, ...,
        # 120+t), so a warp-level LDGSTS covers exactly two whole 256 B rows =
        # sixteen fully used 32 B sectors.
        ix_base = sIX.iterator.toint()
        col16 = tidx % ROW_LANES
        team = tidx // ROW_LANES
        thr_ld = tiled_ld.get_slice(col16)

        def gather_row(s, r, ix):
            """One whole K row: ROW_LANES lanes x one cp.async.cg 16 B each.

            Source is the gmem row named by the runtime index (contiguous, so
            the TV partition lands lane `col16` on bytes [col16*16, +16));
            destination is the same logical row of the swizzled K slot, whose
            swizzle permutes whole 16 B granules — so each lane's granule is
            still one contiguous 16 B store.
            """
            cute.copy(tiled_ld,
                      thr_ld.partition_S(mK[(0, ix, None)]),
                      thr_ld.partition_D(sK[s][(r, None)]))

        for ch in cutlass.range_constexpr(n_chunk):
            s = ch % NSTAGE
            if cutlass.const_expr(ch >= NSTAGE):
                # slot s is still being read by chunk ch-3's consumers
                _seg("producer.empty.wait.begin", ch)
                mbar_wait_spin(mbar + NSTAGE + s, (ch // NSTAGE - 1) % 2)
                _seg("producer.empty.wait.end", ch)
            _seg("producer.gather.issue.begin", ch)
            # D0.8 P2: the group's 16 LDGSTS are NOT one burst. Four slices of
            # four rows, and between the slices sit the address sources: each
            # slice opens with its own four B6 index LDS (the 16 lanes of a
            # team read one address, an smem broadcast) plus the imax0 clamps
            # that depend on them, and the dI copy's 64-bit address arithmetic
            # + issue sits between slices 0 and 1. A producer warp and the
            # consumer warps share the four subpartitions' MIO issue ports and
            # the SM's L2 ports, so a whole group offered at once parks the
            # consumers' STSM / LDS / UBLKRED behind it.
            base = ix_base + cutlass.Int32(ch * BN) * 4
            for sl in cutlass.range_constexpr(NSLICE):
                if cutlass.const_expr(sl == 0):
                    _seg("producer.idx.load.begin", ch)
                ixs = []
                rows = []
                for j in cutlass.range_constexpr(ROWS_PER_SLICE):
                    r = (sl * ROWS_PER_SLICE + j) * ROW_TEAMS + team
                    rows.append(r)
                    ixs.append(imax0(lds_i32(base + r * 4)))
                if cutlass.const_expr(sl == 0):
                    _seg("producer.idx.load.end", ch)
                for j in cutlass.range_constexpr(ROWS_PER_SLICE):
                    gather_row(s, rows[j], ixs[j])
                if cutlass.const_expr(sl == 0):
                    # dI arrives via cp.async.ca (a 4-byte copy into sDI[s])
                    # instead of LDG->STS, so it joins this group and the
                    # completion signal below covers both the K rows and the
                    # dI scalar. One scalar per kv row, addressed by tidx.
                    di_src = (mDI.iterator.toint()
                              + (cutlass.Int64(row)
                                 * cutlass.Int64(cute.size(mDI, mode=[2]))
                                 + cutlass.Int64(ch * BN + tidx)) * 4)
                    di_dst = (sDI[s].iterator.toint()
                              + cutlass.Int32(tidx) * 4)
                    cp_async_ca_4b(di_dst, di_src)
            _seg("producer.gather.issue.end", ch)
            # D0.8 P1: in-flight depth 1. The group is closed and drained by
            # its own producer thread before full[s] is signalled, so at any
            # instant the SM is offered ONE group's worth of LDGSTS and the
            # producer — not the consumers — pays the round trip. The fence
            # publishes the slot into the async-shared proxy; the arrive is the
            # plain one (128 arrivals, one per producer thread), because with
            # the wait behind us there is nothing left to arrive ON completion.
            cp_async_commit_group()
            _seg("producer.gather.drain.begin", ch)
            cp_async_wait_group(0)
            _seg("producer.gather.drain.end", ch)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(mbar + s)
        _seg("producer.entry.end")

    # ---------------------------------------------------------- WG1 / WG2 --
    if tidx >= WG:
        cute.arch.setmaxregister_increase(CONS_REGS)
        _seg("consumer.entry.begin")
        lt = tidx % WG
        wid = lt // 32                     # warp within this consumer WG
        cid = cutlass.Int32(tidx // WG) - 1

        # --- D0.15 phase contract -----------------------------------------
        # 0x3 is the symmetric pair (dS overwrite / dS written); 0x4 and 0x5
        # are one-way publish-handoffs that keep c1 one MMA chain behind c0.
        # 0x6 is the guard around the dK staging buffer: D2.2 makes B4 a
        # cross-WG resource (each consumer reduces WHOLE rows, half of which
        # the partner wrote), so it is 256 wide.
        def wait_partner():
            if cid == 0:
                named_bar_sync(BAR_C1T0, BAR_DS_W)
            else:
                named_bar_sync(BAR_C0T1, BAR_DS_W)

        def publish():
            if cid == 0:
                named_bar_arrive(BAR_C0T1, BAR_DS_W)
            else:
                named_bar_arrive(BAR_C1T0, BAR_DS_W)

        def stage_drain():
            # bulk groups are per-thread: each of the four issuing lanes drains
            # its own (D1.4), then the 256-wide meeting publishes "B4 has been
            # read out by BOTH WGs" — with D2.2 payloads my partner's reduces
            # cover the half of every row that I am about to overwrite.
            if lt % 32 == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            named_bar_sync(BAR_STG, BAR_DS_W)

        thr_s = mma_s.get_slice(lt)
        cS = thr_s.partition_C(cute.make_identity_tensor((H_IDX, 64)))
        accS = _mk_rmem(cS.shape, F32)
        tSrA = mma_s.make_fragment_A(thr_s.partition_A(sQ))

        thr_k = mma_k.get_slice(lt)
        cK = thr_k.partition_C(cute.make_identity_tensor((64, BN)))
        accK = _mk_rmem(cK.shape, F32)
        sQt_half = cute.local_tile(sQt, (64, H_IDX), (cid, 0))
        tKrA = mma_k.make_fragment_A(thr_k.partition_A(sQt_half))
        tKrB = mma_k.make_fragment_B(thr_k.partition_B(sDSt))

        thr_q = mma_q.get_slice(lt)
        cQ = thr_q.partition_C(cute.make_identity_tensor((H_IDX, 64)))
        accQ = _mk_rmem(cQ.shape, F32)
        tQrA = mma_q.make_fragment_A(thr_q.partition_A(sDS))

        # dS exchange face: this consumer owns columns [cid*64, cid*64+64) of
        # the K-major face; `retile` re-reads the S accumulator registers in
        # the order STSM x4 wants them, so the pack loop is a plain elementwise
        # bf16 cast with no permutation of its own. The view is rebuilt inside
        # the chunk loop (a trace-time reindex, no instruction) so it cannot
        # outlive the accumulator writes it reads.
        thr_ds = r2s.get_slice(lt)
        tRS_sD = thr_ds.partition_D(cute.local_tile(sDS, (H_IDX, HALF),
                                                    (0, cid)))
        stgDS = _mk_rmem(tRS_sD.shape, BF16)

        # D1.1: the 64xN WGMMA C fragment gives each thread exactly TWO M rows
        # (value sub-mode (i // 2) % 2 selects between them), so the dW partial
        # sums are two fp32 registers, not one per fragment element — 30 words
        # handed back to segment A's non-accumulator working set (D0.17).
        dw0 = cutlass.Float32(0.0)
        dw1 = cutlass.Float32(0.0)
        accQ.fill(0.0)
        ws_base = mWs.iterator.toint()
        st_base = sST.iterator.toint()

        def emit_reduce(cch):
            """Issue chunk `cch`'s share of the dK bulk reduce-adds.

            D2.2: the payload is a WHOLE 512 B kv row, not a 256 B d-half.
            Consumer c owns kv rows [c*64, c*64+64) of the chunk and its warp w
            owns [c*64 + w*16, +16), so the chunk's 128 rows are still issued by
            exactly one lane each and the arithmetic is untouched — but the TMA
            unit is asked to accept 128 instructions per chunk instead of 256,
            over eight 16-deep chains instead of eight 32-deep ones. The price
            is that B4 becomes cross-WG (the half of each row I reduce was
            written by my partner), which is what widened BAR_STG to 256.

            D1.8's chain shape is preserved: the RPW destination row indices are
            read out of B6 in one clustered pass at the head of the chain, and
            the invalid family is held off by the issue instruction's own
            predicate, so between two adjacent UBLKRED sites there is only
            address arithmetic.
            """
            if lt % 32 == 0:
                rbase = cid * (BN // 2) + wid * RPW
                ridx = []
                for rr in cutlass.range_constexpr(RPW):
                    ridx.append(sIX[cch * BN + rbase + rr])
                for rr in cutlass.range_constexpr(RPW):
                    bulk_reduce_add_f32_pred(
                        ws_base + cutlass.Int64(ridx[rr]) * (D_IDX * 4),
                        st_base + (rbase + rr) * (STRIDE_F32 * 4),
                        D_IDX * 4, ridx[rr])
                cute.arch.cp_async_bulk_commit_group()

        for ch in cutlass.range_constexpr(n_chunk):
            s = ch % NSTAGE
            _seg("all.chunk.top.begin", ch)
            # the partner has retired chunk ch-1's chain1/chain2 -> the single
            # buffered dS face may be overwritten
            _seg("consumer.top.bar.begin", ch)
            named_bar_sync(BAR_DS, BAR_DS_W)
            _seg("consumer.top.bar.end", ch)

            _seg("consumer.kfull.wait.begin", ch)
            mbar_wait_spin(mbar + s, (ch // NSTAGE) % 2)
            _seg("consumer.kfull.wait.end", ch)

            # the partner has issued its previous MMA chain. c0 skips this on
            # the first chunk (it opens the ping-pong), which is what makes the
            # two warpgroups run one chain out of step for the whole loop.
            _seg("consumer.partner.wait.begin", ch)
            if cutlass.const_expr(ch == 0):
                if cid != 0:
                    named_bar_sync(BAR_C0T1, BAR_DS_W)
            else:
                wait_partner()
            _seg("consumer.partner.wait.end", ch)

            sK_own = cute.local_tile(sK[s], (64, D_IDX), (cid, 0))
            tSrB = mma_s.make_fragment_B(thr_s.partition_B(sK_own))
            _seg("consumer.chain0.issue.begin", ch)
            warpgroup.fence()
            at_s = cute.make_mma_atom(mma_s.op)
            at_s.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(D_IDX // 16):
                cute.gemm(at_s, accS, tSrA[(None, None, kb)],
                          tSrB[(None, None, kb)], accS)
                at_s.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            _seg("consumer.chain0.issue.end", ch)
            publish()

            # D2.1: the previous chunk's reduce issue rides THIS chunk's
            # chain0 flight. Its source (B4) was published by the top-of-chunk
            # BAR_DS above, and nothing rewrites B4 until the stage_drain
            # below, which is a whole segment A + chain1/chain2 issue away —
            # the acceptance latency of the 8x16 chains lands inside chain0's
            # flight and the segment A that follows it instead of running bare
            # between the staging write and the chunk's tail (D0.16 step 16).
            if cutlass.const_expr(ch > 0):
                _seg("spin.reduce.emit.begin", ch)
                emit_reduce(ch - 1)
                _seg("spin.reduce.emit.end", ch)

            _seg("consumer.chain0.drain.begin", ch)
            warpgroup.wait_group(0)
            _seg("consumer.chain0.drain.end", ch)

            # segment A: relu mask, dW partial sums, dS = dI*w*mask. The dS
            # values overwrite the S accumulator in place, so the F2FP pack
            # below terminates the S live range instead of doubling it (D0.17).
            _seg("consumer.ew_a.pack.begin", ch)
            for i in cutlass.range_constexpr(cute.size(cS)):
                a = accS[i]
                h = cS[i][0]
                kvc = cid * 64 + cS[i][1]
                di = sDI[s][kvc]
                rel = di * select_f32_gt(a, zero, a, zero)
                if cutlass.const_expr((i // 2) % 2 == 0):
                    dw0 = dw0 + rel
                else:
                    dw1 = dw1 + rel
                wv = sWT[h].to(cutlass.Float32)
                accS[i] = select_f32_gt(a, zero, di * wv, zero)
            srcDS = _retile(thr_ds, accS)
            for v in cutlass.range_constexpr(cute.size(stgDS)):
                stgDS[v] = srcDS[v].to(BF16)
            cute.copy(r2s, stgDS, tRS_sD)
            cute.arch.fence_view_async_shared()   # generic STSM -> async proxy
            _seg("consumer.ew_a.pack.end", ch)
            _seg("consumer.ds.bar.begin", ch)
            named_bar_sync(BAR_DS, BAR_DS_W)
            _seg("consumer.ds.bar.end", ch)

            sKt_half = cute.local_tile(sKt[s], (64, BN), (cid, 0))
            tQrB = mma_q.make_fragment_B(thr_q.partition_B(sKt_half))
            _seg("consumer.partner.wait2.begin", ch)
            wait_partner()
            _seg("consumer.partner.wait2.end", ch)
            # two commit groups, not one: `wait_group(1)` below retires dK only
            # and leaves dQ in flight over segment B (D0.16, the single
            # overlap segment).
            _seg("consumer.chain1.issue.begin", ch)
            warpgroup.fence()
            at_k = cute.make_mma_atom(mma_k.op)
            at_k.set(warpgroup.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(H_IDX // 16):
                cute.gemm(at_k, accK, tKrA[(None, None, kb)],
                          tKrB[(None, None, kb)], accK)
                at_k.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            _seg("consumer.chain1.issue.end", ch)
            _seg("consumer.chain2.issue.begin", ch)
            at_q = cute.make_mma_atom(mma_q.op)
            at_q.set(warpgroup.Field.ACCUMULATE, True)
            for kb in cutlass.range_constexpr(BN // 16):
                cute.gemm(at_q, accQ, tQrA[(None, None, kb)],
                          tQrB[(None, None, kb)], accQ)
            warpgroup.commit_group()
            _seg("consumer.chain2.issue.end", ch)
            publish()
            _seg("consumer.chain1.drain.begin", ch)
            warpgroup.wait_group(1)
            _seg("consumer.chain1.drain.end", ch)

            # segment B (in chain2's flight): dK^T acc is (M = d, N = kv) and
            # the reduce source wants contiguous [kv][d] half rows, so the 64
            # scalar stores per thread are the transpose.
            _seg("consumer.stage.wait.begin", ch)
            if cutlass.const_expr(ch > 0):
                stage_drain()
            _seg("consumer.stage.wait.end", ch)
            _seg("consumer.ew_b.stage.begin", ch)
            for i in cutlass.range_constexpr(cute.size(cK)):
                sST[(cK[i][1], cid * HALF + cK[i][0])] = accK[i]
            _seg("consumer.ew_b.stage.end", ch)
            cute.arch.fence_view_async_shared()
            cute.arch.fence_proxy("async.shared", space="cta")
            # No meeting here any more: "both halves of B4 are written" is
            # carried by the top-of-chunk BAR_DS of chunk ch+1 (and by the
            # epilogue's BAR_DS for the last chunk), which every consumer
            # thread crosses before the D2.1 issue site that reads them.

            _seg("consumer.chain2.drain.begin", ch)
            warpgroup.wait_group(0)
            _seg("consumer.chain2.drain.end", ch)

            # slot s is read out (chain0/chain2 retired, dI consumed before the
            # dS rendezvous) -> the producer may refill it for chunk ch+3.
            # D0.8 closed without P3, so this is the release point again: the
            # earliest instant the slot is provably free, per D0.16 step 17 in
            # the form D2.1 gives it (the reduce block reads B4, not the K slot,
            # so the release does not wait behind it).
            _seg("consumer.empty.arrive.begin", ch)
            if lt == 0:
                cute.arch.mbarrier_arrive(mbar + NSTAGE + s)
            _seg("consumer.empty.arrive.end", ch)

        _seg("consumer.epilogue.entry.begin")
        # One meeting, three duties: the partner has retired the last chunk's
        # chain1 (its A operand is the Q face) before this WG overwrites that
        # face with dQ; both halves of B4 hold chunk n-1 and are visible in the
        # async proxy; and no reduce is in flight against B4.
        _seg("consumer.epilogue.bar0.begin")
        named_bar_sync(BAR_DS, BAR_DS_W)
        _seg("consumer.epilogue.bar0.end")
        # D2.1 leaves the last chunk with no following chain0 to hide behind,
        # so its issue lands here — but the drain does not: the dQ pack and the
        # dW merge below run in its flight, and only then is B4 read out.
        _seg("spin.reduce.emit.begin", n_chunk)
        emit_reduce(n_chunk - 1)
        _seg("spin.reduce.emit.end", n_chunk)
        # dQ acc -> bf16 into the (now dead) Q face, each consumer its d-half
        for i in cutlass.range_constexpr(cute.size(cQ)):
            sQ[(cQ[i][0], cid * 64 + cQ[i][1])] = accQ[i].to(BF16)
        # D1.6: the dW merge closes the warp-internal contention with two XOR
        # butterflies before it touches smem. The 64xN C fragment gives thread
        # lt the M rows cS[0][0] / cS[2][0] = 16*(lt//32) + (lt%32)//4 (+8), so
        # the four lanes of a group {4g..4g+3} carry partial sums of the SAME
        # two rows and differ only in the N columns they covered; lane masks 1
        # and 2 sum that group with zero contention and lane 4g publishes for
        # it. Per CTA that is 128 smem reduce-adds onto 64 addresses (2 writers
        # each) instead of 512 (8 writers each) — the reference's CAS-spin
        # (ATOMS.CAST.SPIN) form is what the contended arm looks like.
        dw0 = dw0 + shfl_bfly_f32(dw0, 1)
        dw0 = dw0 + shfl_bfly_f32(dw0, 2)
        dw1 = dw1 + shfl_bfly_f32(dw1, 1)
        dw1 = dw1 + shfl_bfly_f32(dw1, 2)
        dw_base = sDW.iterator.toint()
        if lt % 4 == 0:
            red_shared_add_f32(dw_base + cutlass.Int32(cS[0][0]) * 4, dw0)
            red_shared_add_f32(dw_base + cutlass.Int32(cS[2][0]) * 4, dw1)
        # the last chunk's reduces must have read B4 out before the CTA retires
        # (smem is reclaimed on exit); nothing else waits on them, so they have
        # had the dQ pack and the dW merge to fly under
        _seg("spin.reduce.drain.begin")
        if lt % 32 == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        _seg("spin.reduce.drain.end")
        cute.arch.fence_proxy("async.shared", space="cta")   # generic -> async
        _seg("consumer.epilogue.entry.end")

    _seg("consumer.epilogue.bar.begin")
    cute.arch.sync_threads()
    _seg("consumer.epilogue.bar.end")
    if tidx < H_IDX:
        mDW[(0, row, tidx)] = sDW[tidx].to(BF16)
    if warp_idx == 0:
        cute.copy(tma_dq, tDsD, dq_dst)
        cute.arch.cp_async_bulk_commit_group()
        _seg("spin.dqstore.drain.begin")
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        _seg("spin.dqstore.drain.end")


# --------------------------------------------------------------------------
# launch entries
# --------------------------------------------------------------------------

@cute.jit
def _jit_k1(mT: cute.Tensor, mP: cute.Tensor, mIdx: cute.Tensor, mG: cute.Tensor,
            scale: cutlass.Float32, stream):
    s_q = cute.size(mT, mode=[1])
    topk = cute.size(mT, mode=[2])
    Storage = k1_storage()
    _k1_score_grad(mT, mP, mIdx, mG, scale, topk // K1_THREADS, Storage).launch(
        grid=[s_q, 1, 1], block=[K1_THREADS, 1, 1], stream=stream)


@cute.jit
def _jit_k2(mQc: cute.Tensor, mW: cute.Tensor, mK: cute.Tensor, mDI: cute.Tensor,
            mIdx: cute.Tensor, mDQc: cute.Tensor, mDW: cute.Tensor,
            mWs: cute.Tensor, stream):
    s_q = cute.size(mW, mode=[1])
    topk = cute.size(mDI, mode=[2])
    q_l, k_l, ds_l, qt_l, kt_l, dst_l = build_smem_layouts()
    mma_s, mma_k, mma_q = build_mmas()
    tiled_ld = build_row_copy()
    r2s = build_ds_store(mma_s)
    tma_q, mQt, tma_dq, mDQt = build_tma(mQc, mDQc, q_l)
    st_l = cute.make_layout((BN, STRIDE_F32), stride=(STRIDE_F32, 1))
    Storage = k2_storage(q_l, k_l, ds_l, topk)
    # min_blocks_per_mp pins the occupancy the setmaxnreg split is budgeted
    # against; without it ptxas drops the instruction on the floor
    _k2_main(tma_q, mQt, mW, mK, mDI, mIdx, tma_dq, mDQt, mDW, mWs,
             mma_s, mma_k, mma_q, tiled_ld, r2s,
             q_l, k_l, ds_l, qt_l, kt_l, dst_l, st_l,
             topk // BN, Storage).launch(
        grid=[s_q, 1, 1], block=[NUM_THREADS, 1, 1],
        min_blocks_per_mp=1, stream=stream)


_COMPILED = {}


def _stream():
    raw = torch.cuda.current_stream().cuda_stream
    try:
        import cuda.bindings.driver as _drv
    except Exception:
        from cuda import cuda as _drv
    return _drv.CUstream(raw)


def _dl(t):
    return from_dlpack(t, assumed_align=16)


def dsa_indexer_bwd(index_q, weights, index_k, attn_score, index_score,
                    topk_indices, grad_loss, sm_scale=1.0, loss_coeff=1.0):
    b, s_q, h_idx, d_idx = index_q.shape
    s_kv = index_k.shape[1]
    topk = index_score.shape[2]
    assert h_idx == H_IDX and d_idx == D_IDX, "h_idx/d_idx outside this kernel's regime"
    assert topk % BN == 0, "topk must be a multiple of the kv chunk"

    dev = index_q.device
    scale = float(loss_coeff) / float(b * s_q) * float(sm_scale)
    key = (int(s_q), int(s_kv), int(topk), round(scale, 12))

    # the TMA boxes address the row-block flattening: one (H_IDX, D_IDX) tile
    # per query row, so both gmem sides are handed over as (s_q*H_IDX, D_IDX)
    dq_flat = torch.empty((s_q * H_IDX, D_IDX), dtype=torch.bfloat16, device=dev)
    d_weights = torch.empty((1, s_q, H_IDX), dtype=torch.bfloat16, device=dev)
    ws = torch.zeros((s_kv, D_IDX), dtype=torch.float32, device=dev)

    stream = _stream()
    mT = _dl(attn_score)
    mP = _dl(index_score)
    mIdx = _dl(topk_indices)
    mG = _dl(grad_loss.reshape(1))
    mQc = _dl(index_q.reshape(s_q * H_IDX, D_IDX))
    mW = _dl(weights)
    mK = _dl(index_k)
    mDQc = _dl(dq_flat)
    mDW = _dl(d_weights)
    mWs = _dl(ws)
    sc = cutlass.Float32(scale)

    entry = _COMPILED.get(key)
    if entry is None:
        entry = (cute.compile(_jit_k1, mT, mP, mIdx, mG, sc, stream),
                 cute.compile(_jit_k2, mQc, mW, mK, mP, mIdx, mDQc, mDW, mWs,
                              stream))
        _COMPILED[key] = entry
    k1, k2 = entry

    k1(mT, mP, mIdx, mG, sc, stream)
    k2(mQc, mW, mK, mP, mIdx, mDQc, mDW, mWs, stream)

    d_index_q = dq_flat.view(1, s_q, H_IDX, D_IDX)
    d_index_k = ws.to(torch.bfloat16).reshape(1, s_kv, D_IDX)
    return d_index_q, d_weights, d_index_k
