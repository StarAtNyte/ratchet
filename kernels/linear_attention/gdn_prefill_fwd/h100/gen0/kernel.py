"""gdn_prefill_fwd — the student's kernel file (the only file the student edits).

Gated Delta Rule (GDN) chunked prefill forward, fixed campaign point
qk16_v32_d128_n8192 (Qwen3.5-35B TP1 head geometry, single 8192-token
sequence). The signature below is the card's frozen contract; the student
implements the body in CuTe DSL; the signature itself may not change.

Semantics (per token t, per state head h, fp32 state S[h] of [D, D], zeros
init; q/k are GVA-shared: state head h reads qk head h // (H_V // H_QK)):

    S = alpha[t, h] * S                       # per-token scalar forget gate
    v_old = k[t] @ S                          # read the key's current value
    S = S + beta[t, h] * outer(k[t], v[t] - v_old)   # delta-rule write
    o[t, h] = scale * (q[t] @ S)              # inclusive readout

Returns (o, final_state). The state layout is K-major: S[h, i, j] contracts
the key index i, so o = q @ S. k arrives L2-normalized along the head dim
(the harness constructs it that way); the kernel must not re-normalize.

-------------------------------------------------------------------------------
D2.4 — part 1 of 3: (i) the h = 8 level spreads over four warps.

The decision's default channel is one commit carrying all three members. That
was taken twice and failed on the machine both times, so Rules 2③ closes it: a
third as-is re-delivery is not allowed and the decision's own delivery channel
names the fallback — a declared split package, n = 3, in the order (i) -> (iii)
-> (ii). This commit is part 1, member (i) alone, on the tree the M2 gate row
left (d49a11f, 0.4173 ms) and at that tree's register tiers, 24 / 104 / 192.

  landing 1 (c04ec64, all three members at 24 / 104 / 192): correctness PASS on
    all three cases, 0.4873 ms against 0.4173 ms, spill 950,272 ld / 623,104 st
    sectors, tensor_pct 12.41 -> 10.68.
  landing 2 (4b7d169, the same three members at 24 / 120 / 184, the remedy the
    decision's cost line names for `spill != 0`): 0.5059 ms, spill 520,192 each
    way, tensor_pct 10.27.

Correctness held both times, so the members are arithmetically right and both
failures are the same resource failure. The member that spends the registers is
(ii): its 32 decay factors are live from the A_kk mask pass across chain 1's
eight wgmma, their drain and two mbarrier spins to the A_qk mask pass, whereas
what (iii) hands back is a fragment live only inside one narrowing — the count
balances, the peak does not, which is why moving 16 registers onto the solver
tier did not close it either (it bought the solver its peak out of the consumer
pair's slack and the pair started spilling instead: spill fell by only 45% and
the device time rose again). Splitting is therefore also the diagnosis: part 1
carries the member that is priced positive on its own, and (iii) then (ii) land
on top of it in the decision's order, each with its own reading.

MEMBER (i). There are exactly four block pairs at the h = 8 level and four warps
in the solver, so `b3 = swid` gives each warp one pair. D1.4 had given warps 0/1
two pairs each precisely so that the pair set a warp wrote at h = 8 was the pair
set it consumed at h = 16, which made the edge between the levels intra-warp and
let the two merge into one segment; warps 2/3 idled through it. That merge is
unwound here: h = 16's warp b4 consumes pairs 2*b4 and 2*b4+1, which two
different warps now write, so the levels are separated by a width-128 rendezvous
again and the solver's rendezvous count per chunk goes 4 -> 5 (entry face,
diagonal blocks, h = 8, h = 16, h = 32's load guard) — still below M0's seven.
Item 1 of the D1.4 note below is superseded by this; item 2 (the h = 32 level's
split-K merge) stands untouched.

Not in this commit, and not in the tree: member (iii)'s `_narrow_f16` /
`_narrow_bf16` and the deleted f32 staging fragment, member (ii)'s resident
decay factors, and the 120 / 184 tiers. Both mask passes compute their own
`exp2(cum_i - cum_j)` again and both faces are narrowed through the f32 staging
fragment again, exactly as at d49a11f. No buffer, ring depth, arrival count,
copy atom, tile shape, register tier or consumer-side line changes; the only
difference from d49a11f's executable is seg 3 / seg 4 and the barrier that now
sits between them.

-------------------------------------------------------------------------------
M2 GATE — closing row, submitted alone, no decision delivered.

The tree is exactly as the D1.1 candidate-(b) revert left it: the only edit in
this commit is this docstring, so no executable line changes and the acceptance
readings are not entangled with any new mechanism. The decision graph is
complete — every decision in the milestone's phase order has been delivered or
exercised on a real reading — so this is the closing round, and per the M2 Gate
it also adjudicates the M0 and M1 lines (M1's gate row was never submitted: the
meta patch's REOPEN row moved the lineage from the M1 tree into M2 before it was
taken, and M1's four levers were all measured before that move).

DELIVERY LEDGER (disposition of every decision):
  M0  D0.1-D0.13 delivered; M0 gate row taken alone at aea49828, correctness
      PASS, 6.1325 ms device, all architecture-conformance and mechanism lines
      read there.
  M1  D1.1 candidate (a) (chains 1 ‖ 2) ⊗ D1.3 delivered and standing.
      D1.1 candidate (b) (chains 4 ‖ 5) delivered at 7c7be312 and measured
      0.4269 ms against 0.4173 ms (+2.30%, noise floor 0.50%); backed out.
      D1.2 (T/A_qk ring 2 -> 3, g/beta 5 -> 3) delivered at 0a68b66c and
      measured 0.4379 ms against 0.4350 ms; not carried forward — the decision
      lists its other two candidates (k 3 -> 4, O 2 -> 3) as expected-zero and
      says not to scan them.  D1.4 (solver rendezvous 7 -> 5) standing.
  M2  D2.2 ⊗ D2.3 delivered as one commit at fb5254e6 and standing — 0.4173 ms
      against 0.4350 ms, the only measured gain since the fp16 solve.
      D2.1 (Q K^T + the A_qk pass moved onto consumer WG1) delivered three
      times: 6965fdcb 0.6569 ms with 7,022,592 spill sectors each way; d01a0d42
      0.4463 ms with spill 0 after binding segment Z onto D0.11's dead
      fragments (the register relief its trigger clause names); and a third
      form with segment Z placed inside segment C, which did not compile
      (`Expected source and destination tensors to have the same size in
      mode-1, but got 1 and 4`).  Backed out at 1f2cea8c.

GATE LINES, read from this executable's last full eval (d99b6d90, the same
bytes this commit ships):
  correctness            three cases PASS                                   OK
  device median          0.4173 ms vs target 0.4107 ms                     FAIL
                         (+1.61%; NCU kernel clock 415,576 ns)
  tensor_pct             12.40 vs the M2 line 11.0 and M1's 10.0             OK
  spill_ld/st_sectors    0 / 0; regs_per_thread 128                         OK
  kernel_count           1, dominant_time_pct 100                           OK
  grid / block           32x1x1 / 512x1x1                                   OK
  smem_dynamic           186,368 B, inside [176,000, 200,704]               OK
  smem_wavefronts        ld 5,934,132 + st 1,634,097 = 7.57M <= 10.2M       OK
  bankconf               ld 12,445 <= 240,000; st 94,619 <= 110,000         OK
  dram_bytes             read 136,381,984 <= 177 MB; write 57,263,936 <= 75 MB OK
  advisory               stall_barrier 1.09 (ref 1.73), stall_long_sb 3.20
                         (3.44), stall_mio 0.11 (0.13), dram_pct 13.87
                         (14.09), warp_active_pct 5.95 (5.97)
  mechanism line 1       the profile this arm returns carries wgmma_total 56,
                         wgmma_depbar0 9, wgmma_tight 5, serial_ratio 0.161;
                         it does not carry mma_a_reg / mma_a_desc, so the RS
                         ledger (X1-X3) is recorded by construction — chains
                         1-5 take their A operand from registers, the solver's
                         chain 0 is SS.

The only failing line is the performance line, by 1.61%. Where the machine
stands on the last driver timeline of this executable (@fb5254e664, cycles per
chunk): solver period 5,743 = busy 2,585 + wait 2,970, its largest single
segment being solve.begin -> solve.end 2,229; consumer WG1 period 5,758 = busy
3,036 + wait 1,928, WG2 busy 3,360 + wait 1,403; consumer.C.t_full 767 cycles,
against the 1,308 the M2 decisions were written against; the four service warps
are idle (supply busy 187, store 88, gate 0, beta 0).

-------------------------------------------------------------------------------
D1.1 candidate (b) backed out: the chain 4 ‖ chain 5 merge is unmounted again
and the lever returns to 0, which is the tier this file has carried since D0.11.

D1.1 gives two candidate pairs and calls each pair its own A/B over the tier
range {0, 1}. Candidate (a), chain 1 ‖ chain 2, was measured and stands.
Candidate (b), chain 4 ‖ chain 5 in one commit group with the D1.3 yield
coarsened to 3 BAR.SYNC per steady chunk, was mounted on top of this tree and
measured 0.4269 ms device / 425,256 ns NCU kernel clock against this tree's
0.4173 ms / 415,480 ns -- +2.30% and +2.35%, against a 0.50% NCU noise floor.
`spill` stayed 0 and the register peak stayed in segment C as predicted, so the
cost is not the +16 registers the decision budgeted for the second live w^T
packing; what the merge also forced was two segment moves -- segment G's
elementwise ahead of the fence (it produces chain 5's A operand) and segment F's
o publication after the merged drain (accO is final only once chain 4 drains).
Other readings across the pair: tensor_pct 12.13 against 12.42, wgmma_depbar0 7
against 9, smem_wavefronts_st 2,686,398 against 1,633,389, bankconf_ld 33,765
against 12,316.

This commit restores the executable content of the pre-merge tree exactly: the
diff against it is empty outside this docstring, so chains 4 and 5 are two
groups again, each with its own fence / commit / `wait_group(0)` and its own
baton pair; segment G returns after segment F and carries the S^T decay, the
w^T column scaling and `rPg`'s packing between the two groups; segment F
publishes o between chain 4's drain and chain 5's issue; the A_qk / k / g
releases return to their D0.11 positions. Blocking rendezvous per consumer
warpgroup per steady chunk: 4 BAR.SYNC + 8 mbarrier waits = 12 (ceiling 15).
Timeline sites 97, 70 distinct names. Of M1's four levers, D1.1 (candidate a)
⊗ D1.3 and D1.4 stand; candidate (b) is measured and rejected; D1.2 (ring
depths) is unmounted.

-------------------------------------------------------------------------------
D2.2 ⊗ D2.3: the solver's exit read-modify-write is deleted, and chain 3 eats
the fp16 inverse face directly.

One package, because it is a correctness atom: the T slot's deliverable changes
meaning and chain 3's operands change dtype, and either half alone is a wrong
number, not a slow one.

D2.2 -- THE EXIT SEGMENT IS GONE. The solve has always computed in fp16, in
place on the T slot; what stood after it was a full-face read-modify-write:
`ldmatrix` the face back into C-fragment order, scale column j by beta_j, narrow
fp16 -> bf16, `stmatrix` it onto the same bytes, and -- because that read spans
the whole face -- a width-128 rendezvous ahead of it so every warp's h = 32
store was visible first. The only reason it existed was that D0.4 pinned chain
3's operands to bf16. Now the last doubling level's store IS the delivery: each
thread fences its own store into the async proxy and arrives on T-full. Nothing
replaces the rendezvous, because T-full is strictly stronger -- it takes all 128
solver arrivals, so a consumer that sees it has seen every warp's store. Solver
rendezvous per chunk 5 -> 4.

D2.3 -- CHAIN 3 IS fp16, AND diag(beta) MOVES TO THE OTHER SIDE. The identity
is w = Tinv . diag(beta) . A' = Tinv . (diag(beta) . A'), so beta can be applied
to A' instead of to T, and it is still applied exactly once. Segment C therefore
waits beta[i]-full as well (new; the beta ring's empty count goes 128 -> 384
because both consumer warpgroups now arrive on it), and forms
A'^T[:, i] = beta_i v^T[:, i] - (kS)^T[:, i] (e_i beta_i) in f32 before packing
it to **fp16** -- `cvt.rn.f16x2.f32` per pair, since this build exports
`cvt_f32x2_bf16x2` but no fp16 counterpart. Chain 3 itself becomes
`HGMMA.64x64x16.F32.F16` on a second warpgroup mma that differs from `mma_rs`
only in operand dtype, with its B descriptor over the fp16 reading of the T slot
(same bytes, same swizzle, same trailing stage mode). Chains 1/2/4/5 are
untouched and still bf16. Precision moves up, not down: fp16 carries 10 mantissa
bits against bf16's 7, on both operands.

Where the work lands: the added elementwise (16 column FMULs + 32 FMUL/FFMA and
8 `LDS.64`) sits in segment C, which runs inside the sibling warpgroup's chain
flight, and the removed work sits on the solver, which the M1 timeline shows as
the zero-slack critical path with the consumer pair burning 1.1-1.3k cycles per
chunk waiting on T-full. Registers: `rPf` (chain 3's fp16 A) and `rP` (chain 4's
bf16 A) are the one packing slot split by dtype and keep the disjoint live ranges
`rP` alone had, so the consumer tier stays 192; the solver gives back the exit's
read-back fragment, and its tier stays 104.

`notes/probe_d23_fp16_chain3.py` certified the operand path before it was
mounted -- both of its failure modes are silent. Y1: the `cvt.rn.f16x2.f32`
packing (whose `$0, $2, $1` operand order is the trap -- with `$0, $1, $2` the
pair swaps and every product is wrong but finite) into an fp16 RS wgmma against
the shipped T slot, on integers exact in fp16, `max_diff == 0.0`. Y2: the same
path carrying THIS package's algebra -- A'^T columns pre-scaled by beta, the
face holding Tinv with no beta on it -- against an f64 reference, which is the
beta-folding equivalence and the fp16 range claim (D2.3's one new risk) in one
measurement. Y3: the same issue repeated at ring stage 1, because chain 3 issues
against `ts = i % T_STAGE` and a stage-1 descriptor reading the wrong bytes
would corrupt every odd chunk while leaving a single-chunk probe clean.

-------------------------------------------------------------------------------
D1.1 (candidate a) ⊗ D1.3: chains 1 and 2 merged into one commit group, and the
ping-pong yield coarsened to match.

D1.1's two candidate pairs are each their own A/B; this commit delivers pair (a),
chain 1 ‖ chain 2, and the D1.3 yield granularity that is atomic with it. Pair
(b), chain 4 ‖ chain 5, is not mounted here -- it needs the +16 registers for a
second live w^T packing, which the decision itself gates behind D0.11's liveness
ladder r1.

The two chains are data independent in all three operand roles: A is the SAME
`rS` (the S^T packing, written once before either issues and read-only through
both), B is a different ring each (q for chain 1, k for chain 2), and the
accumulators are different (accO for o_inter^T, accW for (kS)^T). So they go
into ONE commit group behind ONE `warpgroup.fence()` and are drained by ONE
`wait_group(0)`, where D0.11's starting tier gave each chain its own fence,
commit and full drain. Per chunk that removes one pipeline drain and one
`ARV -> SYNC` hand-off, and it doubles the elementwise window the sibling
warpgroup gets to hide inside.

D1.3 is the ⊗ half and is not optional. A yield left BETWEEN the two chains lets
the sibling's `SYNC` pass while this warpgroup still has chain 2 unissued, so
both halves of the pair issue into the tensor pipe at once and the merge's gain
is handed straight back as contention. The baton is therefore taken once before
the fence and given once after the commit: 4 BAR.SYNC per steady chunk against
D0.11's 5 (the contract's blocking-rendezvous ceiling drops with it, and the
first-chunk peel is unaffected -- it never issues chains 1/2, so both
warpgroups still run 3 take/give pairs there and the baton stays balanced).

The wait-target relation of D0.11 is unchanged in kind: q[i]-full, g[i]-full and
k[i]-full all name chunk i and all sit at the top of the merged segment, ahead of
the baton region, so no yield is held across an mbarrier wait. k[i]-full has
moved up by half a chain (it used to head segment B), which means the k ring
stage is held from slightly earlier -- the decision's stated cost, and the
release point in segment G is unmoved. q[i] is released right after the merged
drain, the first point at which chain 1 has actually read it.

Registers: candidate (a) adds nothing. Both chains already shared `rS`, and accO
and accW are both in the 128-register accumulator base; nothing new is live
across the merge, so the consumer tier stays 192 with D0.11's W = 54 working set
untouched. (It is candidate (b) that would need +16 for the second w^T packing.)

-------------------------------------------------------------------------------
D1.4: the solver's triangular inverse at FIVE rendezvous instead of seven.

M1's third lever on the critical path. D0.9's solve spent seven width-128
rendezvous on `BAR_SOLVER`; two of them carried no dependency that the work
could not be re-assigned around, and this commit removes both:

  1. THE h = 8 AND h = 16 LEVELS BECOME ONE SEGMENT. The barrier between them
     was a real cross-warp edge: the h = 16 warp b consumed the h = 8 corners of
     warps 2b and 2b+1. Warps 0/1 now take TWO block pairs each at h = 8 (warp 0
     -> pairs 0, 1; warp 1 -> pairs 2, 3), which is exactly the pair set that
     same warp consumes at h = 16, so the edge is intra-warp -- a warp-wide
     `stmatrix` and the later warp-wide `ldmatrix` over those bytes, both
     `.aligned` on a converged warp, hold in program order with no barrier.
     Warps 2/3 idle through the merged segment and meet the others at its end.

  2. THE h = 32 LEVEL LOSES ITS SPLIT-K SMEM MERGE. The 2x2 warp grid used to
     split GEMM1's N (== GEMM2's K) in half, so the two warps holding the same
     output strip had to ADD their partial products -- and a cross-warp add
     cannot live in registers, which cost one `stmatrix`, one `ldmatrix` and two
     rendezvous (partial visible / all stores done). The grid is now (yh =
     output M half, nh = output N half): GEMM1 runs at the FULL K = 32 and
     N = 32, so the sum lives in its own accumulator and never enters smem, and
     the level stores exactly once. Each M half recomputes GEMM1 once per N
     partner -- 8 extra `m16n8k16` per warp traded for the two rendezvous and
     the smem round trip.

Everything else in the solve is D0.9 verbatim: the fp16 entry face with the
mask fused into the narrowing, the 8 diagonal 8x8 back-substitutions on two
warps, and the accumulator -> A register chaining inside each level. (The exit
that read the face back, applied the diag(beta) column scale and stored bf16 was
D0.9's until D2.2 deleted it; the rendezvous count quoted below is that build's.)
Rendezvous per chunk: 5 BAR.SYNC (entry face,
diagonal blocks, the merged 8/16 level, the h = 32 load guard, the h = 32
stores) + 6 mbarrier waits = 11, down from 13.

`notes/probe_d14_solver_rendezvous.py` certified both restructures ON THIS T
SLOT before they were mounted -- both fail SILENTLY (wrong numbers, no crash)
if a reading or an ordering assumption is wrong. New and unverified before the
probe: the (n = 32, k = 32) transposed reading of the face that supplies GEMM1's
B; the accumulator -> A chain at MN1 = (16, 32), a ratio the shipped tree never
exercised; GEMM2's B taken from the (16, 32) transposed reading at (n half, row
block 0); and the barrier-free `stmatrix` -> `ldmatrix` order inside one warp.
PROBE PASS on all five contracts: R1 (the solve through h = 32 vs the f64
inverse, relative to |T|max, diagonal exactly 1 and upper triangle exactly 0),
R1b/R2b (the untouched stage bit-exact), R2 (the whole solve plus the exit vs
`inv(I + A) . diag(beta)` in bf16), C1 (named `bar.sync` == 7 EXACTLY -- one
prologue fill + five solve rendezvous + one exit, against nine in the D0.9
build; plus both `ldmatrix` members, `stmatrix` >= 10, `m16n8k8` >= 4,
`m16n8k16` >= 14, `shfl.idx` >= 16, zero narrow `st.shared`, zero local memory).

Register tiers are unchanged at 24 / 104 / 192: the h = 32 level's live set
grows (GEMM1's B is the whole 32x32 and its accumulator the whole 16x32) but
the split-K merge's two fragments -- the partial and the read-back strip --
leave with it, and the solver's peak segment is the entry pass, not this one.

-------------------------------------------------------------------------------
M0 construction state (everything else standing, unchanged by this commit).

Standing since the first iterations: D0.1 one fused kernel, grid = h_v x nseq,
one CTA per v-head walking the whole sequence in 64-token chunks, no cluster;
D0.3 block 512 = 4 warpgroups split on `tid >> 7`; D0.6 three-tier setmaxnreg off
the 512x128 entry allocation (back at 24 / 104 / 192 now that the solver no
longer needs the 16 the scalar solve borrowed -- see the constant's comment).

Already standing (the iteration before):

  * D0.2  the chunk algebra. C = 64 tokens per chunk (== the wgmma M mode, no
          further M blocking). Per chunk, with g_i the log2-domain running gate
          sum inside the chunk and gamma_i = 2^g_i:

            A_kk[i,j] = beta_i * gamma_i/gamma_j * (k_i.k_j)   i > j      [64x64]
            A_qk[i,j] = scale  * gamma_i/gamma_j * (q_i.k_j)   i >= j     [64x64]
            T         = (I + A_kk)^-1 . diag(beta)                        [64x64]
            A'        = v - gamma (x) (k S_0)                             [64xd_v]
            w         = T A'                                             [64xd_v]
            o         = scale*gamma (x) (q S_0) + A_qk w                  [64xd_v]
            S_C       = gamma_C S_0 + sum_i (gamma_C/gamma_i) k_i^T w_i   [d_kxd_v]

          Each consumer warpgroup keeps S^T[64 v x 128 k] f32 in registers
          (M = v, N = k), the pair splitting d_v in half (WG1 v 0..63, WG2
          v 64..127); S never touches smem. The five chains per consumer WG,
          oriented by the M/N the registers actually carry:

            1  o_inter^T[v,tok] = S^T . q^T      64x64x16 x8   A=REG  B=q ring
            2  (kS)^T [v,tok]   = S^T . k^T      64x64x16 x8   A=REG  B=k ring
            3  w^T    [v,tok]   = A'^T . T^T     64x64x16 x4   A=REG  B=T slot
                                                              (fp16 since D2.3)
            4  o^T    [v,tok]  += w^T . A_qk^T   64x64x16 x4   A=REG  B=A_qk slot
            5  S^T    [v,k]    += (w^T (x) g).k  64x128x16 x4  A=REG  B=k ring,
                                                                      MN-major
          plus the solver's two SS chains (chain 0 K K^T, chain 1 Q K^T, both
          operands smem descriptors, K = 128). All seven operand paths are the
          ones certified by `notes/probe_m0_step2_chains.py` (X1-X5, every
          contract `max_diff == 0.0`), reusing the very same construction.

  * D0.4  precision contract. Every accumulator (S^T, o^T, w^T, (kS)^T, and the
          solver's two) is f32; every wgmma A/B operand is 16-bit -- bf16
          everywhere except chain 3, which D2.2 ⊗ D2.3 moved to fp16 (the
          decision's own precision row); A' = v - (kS)(x)e
          is formed in f32 and converted once (the variant D0.4 explicitly
          allows: f32 FADD + a single narrowing, more instructions, better
          numerics); the gates are f32 throughout; final_state is written
          straight out of the f32 accumulator.

  * D0.5  gate representation. The gate warp turns 64 alphas into three 64-f32
          chunk tables in the g ring slot -- `cum[i] = sum_{s<=i} log2(alpha_s +
          1e-10)`, `e[i] = exp2(cum[i])`, `es[i] = exp2(cum[i]) * scale` -- and
          the beta warp writes the 64-f32 beta table. The consumer scales
          o_inter^T by `es[]` after chain 1 (readout scale folded into the inter
          term), forms A' with `e[]`, decays S^T by `exp2(cum[63])` and scales
          w^T's columns by `exp2(cum[63] - cum[i])` before chain 5. The solver
          builds both masked matrices out of `cum[]` / the beta table, folding
          `scale` into A_qk's elements (the intra half of the same readout
          scale). log2 goes through `lg2.approx.f32`, exp2 through
          `cute.math.exp2(fastmath=True)` -- one MUFU each.

  * D0.7  the buffer list is consumed at its milestone sizes and ring depths
          (q 2 / k 3 / v 2 / A_qk 2 / T 2 / O 2 / g 5 / beta 5, 185,728 B), and
          the ledger's SOURCE decisions are the ones built: X1-X3 register-source
          (S^T, (kS)^T, w^T all reach their consumers as REG A operands, nothing
          about S or A' lands in smem), X4-X6 smem (A_kk's in-place solve, and
          the two cross-warpgroup faces A_qk and T), X8 smem (the gate tables),
          I1-I3 smem via TMA. Read-side atoms are the certified ones: wgmma
          descriptors over the rings for q/k/T/A_qk, the MN-major composition
          view of the k ring for chain 5, and `LdMatrix8x8x16bOp(True, 4)` for
          v^T. The two write-side exchange faces are now at their tabled atom
          too: A_kk enters the T slot as fp16 and stays fp16 (since D2.2 the
          inverse face itself is what chain 3 reads), A_qk
          leaves through its own slot as bf16, both `STSM.16.M88.4` x4 per
          warp off the C-fragment map of `mma_ss` (the accumulator they narrow),
          so the solver's write side has no scalar `st.shared` left.

  * D0.8  the eight mbarrier ring pairs are all live now, at the milestone's
          arrive counts: q/k/v full 1 (TMA expect-tx) empty 12/12/8 (one lane per
          consuming warp), A_qk and T full 128 (solver) empty 256 (consumer
          pair), g full 32 (gate warp) empty 384 (WG1+2+3), beta full 32 empty
          128 (solver). Every cross-role hand-off is an mbarrier full/empty pair;
          the only named barriers are inside a role (0x6 for the gate warp's
          scan, 0xd for the solver's segments). D2.3 moved beta's empty count to
          384: segment C reads the table now, so both consumer warpgroups arrive
          on it as well.

  * D0.9  solver warpgroup. Chain 0 (K K^T) and chain 1 (Q K^T) as SS wgmma over
          the k and q rings, then the branch-free mask/decay/beta pass
          (`setp`+`selp` predicate select, one FFMA into each `MUFU.EX2`), then
          the inversion of (I + A_kk), which is handed off as the fp16 face
          itself (D2.2; diag(beta) rides out on the consumer's A' instead).
          The solve is this decision's fp16 block-doubling form,
          `HMMA.1688` / `HMMA.16816` in place of the 2016-step FFMA chain, the
          whole thing in place on the T slot -- restructured by D1.4 (see the
          header) from seven segments to five, and by D2.2 to four.

Standing since the previous iteration:

  * D0.13 first/last chunk peel and the prologue warm-up.

          The consumer body exists in two traced copies instead of one. The
          first-chunk peel runs against S_0 = 0, which collapses the algebra:
          o_inter^T = S^T q^T and (kS)^T = S^T k^T are both identically zero,
          so chains 1 and 2 are not issued at all (16 HGMMA and the S^T packing
          that feeds them), A' = v^T is formed by the same segment C as always
          with the (kS) term simply dropped (the `LDSM.MT` fragment, the f32
          staging, one `_pack_bf16` into chain 3's A operand -- no subtraction
          and no e[] lookup), chain
          4 OPENS o^T (`ACCUMULATE = False`) because there is no inter term to
          accumulate onto -- which also removes segment D's es[] column scaling
          -- and chain 5 OPENS S^T for the same reason, so the accumulator is
          never explicitly zeroed: the reference's `RZ, !UPT`. The peel keeps
          every wait pointed at chunk 0 and at the top of its segment (D0.11),
          keeps every release (q[0] is released without being read -- safe,
          because q-empty needs all 12 arrivals and the solver's four land
          after its own chain 1 drains), and is run by BOTH warpgroups, so the
          ping-pong baton stays balanced at 3 take/give pairs against the
          steady body's 5. The steady loop then runs chunks 1..n_chunks-1.

          The last-chunk peel is the post-loop epilogue itself: its content is
          "the steady body plus the final_state write-out, and no further S^T
          packing", and in this issue order the S^T packing is at the TOP of
          segment A (it feeds chains 1/2 of the same chunk), not at the tail,
          so the last steady trip needs no specialization -- only the D0.12
          write-out that already follows the loop.

          The prologue warm-up is the ring's initial phase, as the decision's
          own parenthetical says: the producer's first Q/K/V_STAGE empty waits
          test a parity the barriers have not left yet, so they pass without
          blocking and the three rings fill back to back. The producer is
          therefore one 128-trip loop with no peel (the tabled trip count), and
          the solver's one-chunk lead over the consumer is likewise emergent:
          its T/A_qk empty waits for chunk 0 pass on the same initial phase, so
          it is already producing T[0] while the consumer is still waiting for
          it.

Already standing (the iteration before):

  * D0.12 the epilogue, both halves of it.

          o: the o^T accumulator is M = v / N = tok but the output wants rows of
          tok with v contiguous, so the orientation is inconsistent and the
          ledger (row X7) forces the smem exchange face. Segment F now waits
          O-empty[i mod 2], packs o^T to bf16 (F2FP), and writes its own 64
          v-columns into the O staging slot with the TRANSPOSED stmatrix member
          (`STSM.16.MT88.4`, 4 per warp) over the same C-fragment thread/value
          map the v^T read already uses -- the write face (v, tok) has v
          contiguous, i.e. M-major C, which is exactly what selects the
          transposing member. Then a generic->async proxy fence and arrive
          O-full (256, so the slot only moves once both halves of the consumer
          pair are in it). WG0.w1, parked until now, is the store warp: wait
          O-full -> fence -> one bulk `UTMASTG.3D` of the 64 tok x 128 v slot ->
          `cp_async_bulk_commit_group` -> `wait_group(0)` -> width-32 rendezvous
          -> 32 O-empty arrivals. The rendezvous is load-bearing: the TMA atom
          elects one lane, so only that lane's bulk-group is non-empty and the
          other 31 lanes' waits return immediately.

          final_state: S^T is held (v, k) and the contract is [h, k, v] with v
          contiguous, so a straight write-out is a 4-byte scatter across k rows.
          Decision (b), through the O staging slots, which are idle once the
          last chunk has been stored: an f32 reading of the same 32 KB
          allocation (no buffer is opened) gives each consumer warpgroup 4,096
          f32 = one 64k x 64v batch, two batches covering S^T's 128 k. Scattered
          STS in, fully coalesced out (consecutive threads read consecutive f32
          and store to consecutive v), a private width-128 rendezvous per
          warpgroup between the two -- private because the pair is one chain out
          of phase and must not meet here.

          Blocking rendezvous per consumer warpgroup per chunk: 4 BAR.SYNC (5
          before D1.3) + 7 mbarrier waits (contract ceiling 15). The new O-empty
          wait sits outside the ping-pong baton region, so no yield is held
          across it.

Already standing (two iterations back):

  * D0.10 the consumer pair's phase contract, `[PHASE: ping-pong]`. WG1 and WG2
          run one body and diverge only on the ping-pong predicate `half`
          (WG1 = 0, WG2 = 1). Every one of the five chains is now wrapped
          `SYNC(own) -> wgmma fence -> HGMMA x n -> commit -> ARV(other)`, so
          exactly one warpgroup issues into the tensor pipe at a time and the
          other's elementwise runs inside that shadow. Named barriers 0x4 and
          0x5, both width 256, both **publish-handoff** and never symmetric --
          one side only ever ARV, the other only ever SYNC: 0x4 carries
          WG2 -> WG1 (WG2 arrives, WG1 blocks), 0x5 carries WG1 -> WG2. The
          prologue stagger is WG2's single `BAR.ARV 0x4` ahead of the loop, the
          initial token for WG1's first `SYNC 0x4`; from there the baton
          alternates strictly, so the pair is exactly one chain out of phase and
          cannot drift further. No statistic is merged between the two (o and S
          stay independent along d_v -- nothing f32 crosses between them), and
          no ping-pong signal is ever held across an mbarrier wait: every
          full/empty wait sits outside the baton region, so a yield can never
          close a cycle with the producer, the gate warp or the solver.
          Blocking rendezvous per warpgroup per chunk: 4 BAR.SYNC (5 until
          D1.3 coarsened the yield) + 6 mbarrier waits (contract ceiling 15).

  * D0.11 issue order and segment liveness. The body is the position table's A-G
          order with the tabled wait targets and release points: A waits
          q[i]/g[i], packs S^T in the sibling's shadow, issues chain 1; B
          releases q[i], waits k[i], issues chain 2; C waits v[i], loads v^T,
          forms A' and packs it, waits T[i]; D issues chain 3 and runs the
          o_inter column scaling *in its own chain's flight* -- the one
          elementwise segment that does, and it is legal because those 32 FMULs
          touch accO, chain 1's already-retired accumulator, never chain 3's
          accW -- then drains; E releases v[i] and T[i], packs w^T, waits
          A_qk[i], issues chain 4, releases A_qk[i]; F publishes o; G decays
          S^T, column-scales and packs w^T, issues chain 5, releases k[i] and
          g[i]. Every wait points at chunk i and sits at the top of its segment,
          never after the warpgroup's own chain, so the consumer carries zero
          lookahead of its own -- all of it comes from the other roles.
          Liveness: accumulator base 128 (S^T 64 + o^T 32 + the one shared 32
          that carries (kS)^T and then w^T, the f32 (kS)^T dying exactly where
          it is packed into A'), and the three A-operand packings keep disjoint
          live ranges. The wait tier is still `wait_group(0)`, but D1.1 has
          moved WHAT a group is: chains 1 and 2 now share one, so the steady
          body drains four times per chunk instead of five.

D2.1 (moving Q K^T and the A_qk mask off the solver onto consumer WG1) was
mounted on top of this tree twice and backed out again; this file is the tree it
was mounted on. Form 1 (segment Z declaring its own 64x64 accumulator, f32 exit
staging and bf16 face beside the body's fragments) measured device median
0.6569 ms with 7,022,592 spill sectors each way; form 2 (the same segment with
those three temporaries aliased onto accO / accW / rO, spill back to 0, tensor
11.62%) measured 0.4463 ms. Both are against 0.4173 ms / tensor 12.42% for this
tree, i.e. +57.4% and +7.0% on a +-1% noise floor, so neither can carry the M2
gate row. The decision's own trigger clause -- "consumer waiting on T-full is
still the largest exposure, the solver is the zero-slack critical path" -- is
what D2.2 spent: on this tree the driver timeline reads solver busy 2585 of a
5743-cycle period with 2970 cycles of wait, while the consumer pair reads busy
3036 (WG1) and 3360 (WG2), and consumer.C.t_full is down to 767 cycles from the
1308 the decision was written against. The solver is the role with slack now and
the consumer pair is the one without, so D2.1 moves an eight-step SS chain and a
64x64 elementwise pass from the slack side onto the busier of the two ping-pong
halves, where it also serializes into the baton instead of overlapping.

Of M1's four levers, D1.1 (candidate a) ⊗ D1.3 and D1.4 are standing; D1.1
candidate (b) was measured on top of this tree at +2.30% and is unmounted again
(top of this docstring); D1.2 (ring depths) is unmounted.

NOTE: this file must never carry `from __future__ import annotations` -- PEP 563
stringifies the `@cute.struct` / `@cute.kernel` annotations and CuTeDSL builds
its types from the annotation objects themselves.
"""

from typing import Callable, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.utils import LayoutEnum


# --- timeline probe hook -------------------------------------------------------
# Kept per the arm's timeline-probe convention: the driver rewrites this body to
# stamp a marker in the probe arm; in the shipped tree it is a no-op evaluated at
# trace time and costs nothing.
def _seg(site, row=0):
    return None


# --- D0.2 chunking / D0.3 block shape -----------------------------------------
CHUNK = 64          # C: tokens per chunk == wgmma M-mode
DHEAD = 128         # d == d_k == d_v
NUM_THREADS = 512   # D0.3: 4 warpgroups
HALF = DHEAD // 2   # d_v columns owned by one consumer warpgroup

# --- D0.7 ring depths (starting points; D1.2 is the lever that moves them) ----
Q_STAGE, K_STAGE, V_STAGE = 2, 3, 2
AQK_STAGE, T_STAGE, O_STAGE = 2, 2, 2
G_STAGE, B_STAGE = 5, 5

# --- D0.8 mbarrier arrive counts (full / empty) -------------------------------
Q_FULL_CNT, Q_EMPTY_CNT = 1, 12       # expect-tx ; 4 solver + 8 consumer warps
K_FULL_CNT, K_EMPTY_CNT = 1, 12
V_FULL_CNT, V_EMPTY_CNT = 1, 8        # 8 consumer warps
AQK_FULL_CNT, AQK_EMPTY_CNT = 128, 256
T_FULL_CNT, T_EMPTY_CNT = 128, 256
O_FULL_CNT, O_EMPTY_CNT = 256, 32
G_FULL_CNT, G_EMPTY_CNT = 32, 384
# D2.3: the beta table gains two consumers. It used to be read only by the
# solver (128 arrivals); now segment C folds diag(beta) into A'^T's column
# scaling, so both consumer warpgroups read it too -- 128 + 256 = 384. A missing
# arrival here is a hang, not a wrong number.
B_FULL_CNT, B_EMPTY_CNT = 32, 384

# one q/k/v tile = [64 tok x 128 d] bf16 = 2 TMA boxes of 64x64
TILE_BYTES = CHUNK * DHEAD * 2

# --- D0.9 / D0.5 named barriers (intra-role lockstep only; id 0 is reserved) ---
BAR_GATE = 6        # width 32: the gate warp's serial prefix scan
BAR_SOLVER = 13     # width 128: the solver's segment boundaries
BAR_STORE = 7       # width 32: the store warp, after the bulk-group drain
# width 128 each, one per consumer warpgroup: the final_state smem transpose.
# Two ids, not one, because the two consumer warpgroups reach the epilogue at
# different times (they are one chain out of phase) and must not rendezvous.
BAR_STATE_1, BAR_STATE_2 = 8, 9

# --- D0.10 ping-pong: two width-256 named barriers, both publish-handoff ------
# 0x4 carries WG2 -> WG1 (WG2 is the arrive-only side, WG1 the blocking side),
# 0x5 carries WG1 -> WG2. Neither is ever SYNCed from both sides -- a symmetric
# rendezvous would lock the pair into lockstep instead of staggering it.
BAR_PP_1 = 4        # WG1 blocks on it; WG2 arrives on it
BAR_PP_2 = 5        # WG2 blocks on it; WG1 arrives on it
PP_WIDTH = 256      # the consumer pair

GATE_EPS = 1.0e-10  # alpha + eps before log2 (D0.4)

# --- D0.6 register tiers (starting points; the split itself is the student's) -
# 24 * 128 (WG0) + 104 * 128 (WG3) + 192 * 256 (WG1+WG2) == 65,536 == the whole
# RF, so the pool is exactly balanced: what the two `.dec` release is exactly
# what the two `.inc` claim, and an `.inc` can never sit blocked on an
# over-subscribed pool.
# The split itself is the student's (the gate only asks for `regs_per_thread <=
# 128 ∧ spill == 0`, which is the 512x128 launch allocation, not the tiers).
# The first-green solver had to borrow 16 from the consumer pair -- its scalar
# forward substitution kept a whole 64-deep f32 column of T resident across a
# 2016-step serial chain, which does not fit under 104 once the chain
# accumulator and addressing are counted. The block-doubling solve gives that
# borrowing back: its working set is warp-level fragments (per level, an A
# fragment, two B fragments and two <= 16x32 accumulators), so the tier returns
# to the reference's 104 and the 16 it releases go where they came from.
#
#   solver 104: two 64x64 chain accumulators are never live at once (chain 0's
#     is consumed by the entry narrowing before chain 1 opens); the doubling's
#     peak is one level's fragments plus the C-fragment staging.
#   consumer 192: accumulator base 128 (S^T 64 + o^T 32 + the one slot that
#     carries (kS)^T and then w^T, 32) plus the peak packing overlap in segment C
#     (v^T 16 + the A' packing 16) = 160, leaving 32 for descriptors and
#     addressing. The A-operand packings have disjoint live ranges (rS dies at
#     chain 2, rP at chain 4, rPg at chain 5), so they share registers rather
#     than summing.
#
# Sum is exactly the whole RF: 24*128 + 104*128 + 192*256 == 65,536, so what the
# two `.dec` release is exactly what the one `.inc` claims.
WG0_REGS, SOLVER_REGS, CONSUMER_REGS = 24, 104, 192

# DSL 4.4.2 renamed these to `setmaxregister_{decrease,increase}` and left the
# `warpgroup_reg_{dealloc,alloc}` spellings as deprecated forwarders; both lower
# to the same `nvvm.setmaxregister`. Resolve at trace time (not import time, so
# module import order can never turn a spelling question into an ImportError)
# and take whichever this build exports.
def _reg_dec(n):
    fn = getattr(cute.arch, "setmaxregister_decrease", None)
    (fn or cute.arch.warpgroup_reg_dealloc)(n)


def _reg_inc(n):
    fn = getattr(cute.arch, "setmaxregister_increase", None)
    (fn or cute.arch.warpgroup_reg_alloc)(n)


def _rmem(shape, dtype):
    """Allocate a per-thread rmem tensor of `shape`.

    This DSL build exports the constructor as `make_rmem_tensor`; others spell
    it `make_fragment`. Both take (shape, dtype); resolve at trace time.
    """
    fn = getattr(cute, "make_rmem_tensor", None) or getattr(cute, "make_fragment")
    return fn(shape, dtype)


_exec_cache = {}


# ------------------------------------------------------------------------------
# scalar primitives that have no DSL spelling (ptx_escape form 3: pure, no side
# effects, the `.reg .pred` lives in its own block scope so call sites cannot
# collide)
# ------------------------------------------------------------------------------
@dsl_user_op
def _lg2(x, *, loc=None, ip=None) -> cutlass.Float32:
    """log2(x) in one MUFU.LG2 (x > 0; the caller adds GATE_EPS)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
        "lg2.approx.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _sel_gt(a, b, x, y, *, loc=None, ip=None) -> cutlass.Float32:
    """a > b ? x : y  (a, b int)  -> ISETP + FSEL, no region."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.gt.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _sel_ge(a, b, x, y, *, loc=None, ip=None) -> cutlass.Float32:
    """a >= b ? x : y  (a, b int)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ge.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _sel_eq(a, b, x, y, *, loc=None, ip=None) -> cutlass.Float32:
    """a == b ? x : y  (a, b int)  -- the unit diagonal of the solve."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Float32(x).ir_value(loc=loc, ip=ip),
         cutlass.Float32(y).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.eq.s32 p, $1, $2;\n\tselp.f32 $0, $3, $4, p;\n}",
        "=f,r,r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f16_lo(w, *, loc=None, ip=None) -> cutlass.Float32:
    """Low fp16 half of a packed 32-bit word -> f32.

    D0.9's diagonal segment keeps each row of an 8x8 block resident as four
    fp16 PAIRS and broadcasts the packed word, so the unpack has to be off the
    chain. bf16's `shl`/`and` trick does not carry over -- bf16 is the high
    half of an f32, fp16 is not -- so this is `mov.b32` + `cvt.f32.f16`
    (KB tri_inverse_block_doubling gotcha 6).
    """
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tcvt.f32.f16 $0, lo;\n}",
        "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _f16_hi(w, *, loc=None, ip=None) -> cutlass.Float32:
    """High fp16 half of a packed 32-bit word -> f32 (see `_f16_lo`)."""
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tcvt.f32.f16 $0, hi;\n}",
        "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def _cvt_f16x2(a, b, *, loc=None, ip=None) -> cutlass.Int32:
    """Pack (a, b) -> one fp16x2 word, `a` in the LOW half.

    D2.3's chain 3 takes fp16 A operands and this build has no DSL export for
    the narrowing: `cute.arch` carries `cvt_f32x2_bf16x2` but neither
    `cvt_f32x2_f16x2` nor `cvt_f16x2_f32` (measured in
    `notes/probe_d23_fp16_chain3.py`, stage `sig`). The PTX takes the HIGH half
    first, so the asm has to read `$0, $2, $1`: with `$0, $1, $2` the pair swaps,
    every product picks up the wrong K element, and the result is wrong but
    finite (hw_kb `cfrag_elementwise_fastpath.md`). Contract Y1 of that probe
    pins this exact spelling at `max_diff == 0.0`.
    """
    return cutlass.Int32(llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _exp2(x):
    return cute.math.exp2(x, fastmath=True)


# ------------------------------------------------------------------------------
# host-side plan: swizzled smem layouts + the SharedStorage type
# ------------------------------------------------------------------------------
def _plan():
    """Build the D0.7 smem layouts and the SharedStorage struct type.

    Called from inside `_gdn_host` (a `@cute.jit`): `cute.tile_to_shape` and
    `cute.cosize` are IR builders and need a live context. The results are IR
    objects owned by *that* trace, so nothing here may be cached across
    `cute.compile` calls.
    """
    bf16 = cutlass.BFloat16
    f32 = cutlass.Float32

    # 128B swizzle for the d-major (K-major) 64x128 tiles; the 64x64 exchange
    # slots take the atom their own major length selects.
    atom_d = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, DHEAD), bf16)
    atom_c = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, bf16, CHUNK), bf16)

    q_l = cute.tile_to_shape(atom_d, (CHUNK, DHEAD, Q_STAGE), order=(0, 1, 2))
    k_l = cute.tile_to_shape(atom_d, (CHUNK, DHEAD, K_STAGE), order=(0, 1, 2))
    v_l = cute.tile_to_shape(atom_d, (CHUNK, DHEAD, V_STAGE), order=(0, 1, 2))
    o_l = cute.tile_to_shape(atom_d, (CHUNK, DHEAD, O_STAGE), order=(0, 1, 2))
    a_l = cute.tile_to_shape(atom_c, (CHUNK, CHUNK, AQK_STAGE), order=(0, 1, 2))
    t_l = cute.tile_to_shape(atom_c, (CHUNK, CHUNK, T_STAGE), order=(0, 1, 2))

    # --- second views over the SAME allocations (one buffer, two readings) ----
    # sK's (tok, d, stage) domain has colex linear index tok + d*CHUNK +
    # s*CHUNK*DHEAD, so the MN-major (d, tok, stage) read view chain 5 needs has
    # transposer strides (CHUNK, 1, CHUNK*DHEAD).  [probe X4]
    kt_l = cute.composition(
        k_l, cute.make_layout((DHEAD, CHUNK, K_STAGE),
                              stride=(CHUNK, 1, CHUNK * DHEAD)))
    # sV read as (d_half, tok, half, stage) with d = half*64 + d_half, so each
    # consumer warpgroup's own 64 v-columns come out as a (d, tok) face in
    # C-fragment order under the transposed ldmatrix member.  [probe X5]
    vt_l = cute.composition(
        v_l, cute.make_layout((CHUNK, CHUNK, 2, V_STAGE),
                              stride=(CHUNK, 1, HALF * CHUNK, CHUNK * DHEAD)))
    # The O staging slot is written through the mirror image of that face: the
    # C fragment o^T has M = v and N = tok, the slot holds o[tok, v] with v
    # contiguous (the layout TMA wants), so the write face is (v, tok) with the
    # M mode contiguous -- M-major C, i.e. the TRANSPOSED stmatrix member. Same
    # composition as vt_l because o_l and v_l are the same shape and atom.
    ot_l = cute.composition(
        o_l, cute.make_layout((CHUNK, CHUNK, 2, O_STAGE),
                              stride=(CHUNK, 1, HALF * CHUNK, CHUNK * DHEAD)))

    # --- D0.9 operand-role readings of the T slot -----------------------------
    # The in-place triangular inverse takes its A / B / D operands as nine
    # different READINGS of the one T allocation, never as pointer surgery: a
    # tiled layout per role, with `MemRange.get_tensor(layout, swizzle=t_l.inner,
    # dtype=Float16)` leaving the swizzle on the pointer and the block
    # coordinates as trailing modes (KB tri_inverse_block_doubling gotcha 1 --
    # `flat_divide` + `select` + `iterator.align(16)` drops the swizzle and reads
    # the wrong bytes, silently).
    #
    # Element coordinate (r, c, s) -> r*CHUNK + c + s*STG, so every role layout
    # carries the SAME trailing stage mode of stride STG; that stride is the
    # assumption `notes/probe_d09_tri_inverse.py` pins from both sides (solve in
    # one stage, read the other back bit-unchanged, the two numerical rungs on
    # opposite stages).
    STG = CHUNK * CHUNK
    # ((m, rep), k, rb, cb, s): the h = 8 A / D reading -- an 8-row block copied
    # by a stride-0 mode into the 16 rows `m16n8k8` wants.
    tb8_l = cute.make_layout(((8, 2), 8, 8, 8, T_STAGE),
                             stride=((CHUNK, 0), 1, 512, 8, STG))
    # (n, k, cb, rb, s): the transposed reading, i.e. some 8x8 block's B operand.
    tt8_l = cute.make_layout((8, 8, 8, 8, T_STAGE),
                             stride=(1, CHUNK, 8, 512, STG))
    t16_l = cute.make_layout((16, 16, 4, 4, T_STAGE),        # (m,k,rb,cb,s)
                             stride=(CHUNK, 1, 1024, 16, STG))
    tt16_l = cute.make_layout((16, 16, 4, 4, T_STAGE),       # (n,k,cb,rb,s)
                              stride=(1, CHUNK, 16, 1024, STG))
    t16x32_l = cute.make_layout((16, 32, 4, 2, T_STAGE),     # (m,k,rb16,cb32,s)
                                stride=(CHUNK, 1, 1024, 32, STG))
    tt16x32_l = cute.make_layout((16, 32, 4, 2, T_STAGE),    # h=32 GEMM2 B
                                 stride=(1, CHUNK, 16, 2048, STG))
    # (n, k, cb32, rb32, s): the 32x32 transposed reading.  D1.4's h = 32 level
    # runs GEMM1 at the FULL K = 32 and N = 32, so its B is the whole 32x32 A21^T
    # out of this reading; the split-K form it replaces took a half-K GEMM2 B
    # from a (32, 16) reading, which no longer exists.
    tt32x32_l = cute.make_layout((32, 32, 2, 2, T_STAGE),    # h=32 GEMM1 B
                                 stride=(1, CHUNK, 32, 2048, STG))

    @cute.struct
    class Storage:
        # --- D0.8 mbarrier arena: 46 barriers, 8 pipelines, full/empty pairs ---
        b_qf: cute.struct.MemRange[cutlass.Int64, Q_STAGE]
        b_qe: cute.struct.MemRange[cutlass.Int64, Q_STAGE]
        b_kf: cute.struct.MemRange[cutlass.Int64, K_STAGE]
        b_ke: cute.struct.MemRange[cutlass.Int64, K_STAGE]
        b_vf: cute.struct.MemRange[cutlass.Int64, V_STAGE]
        b_ve: cute.struct.MemRange[cutlass.Int64, V_STAGE]
        b_af: cute.struct.MemRange[cutlass.Int64, AQK_STAGE]
        b_ae: cute.struct.MemRange[cutlass.Int64, AQK_STAGE]
        b_tf: cute.struct.MemRange[cutlass.Int64, T_STAGE]
        b_te: cute.struct.MemRange[cutlass.Int64, T_STAGE]
        b_of: cute.struct.MemRange[cutlass.Int64, O_STAGE]
        b_oe: cute.struct.MemRange[cutlass.Int64, O_STAGE]
        b_gf: cute.struct.MemRange[cutlass.Int64, G_STAGE]
        b_ge: cute.struct.MemRange[cutlass.Int64, G_STAGE]
        b_bf: cute.struct.MemRange[cutlass.Int64, B_STAGE]
        b_be: cute.struct.MemRange[cutlass.Int64, B_STAGE]
        # --- D0.7 buffer list --------------------------------------------------
        sQ: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(q_l)], 1024]
        sK: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(k_l)], 1024]
        sV: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(v_l)], 1024]
        sA: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(a_l)], 1024]
        sT: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(t_l)], 1024]
        sO: cute.struct.Align[
            cute.struct.MemRange[bf16, cute.cosize(o_l)], 1024]
        sG: cute.struct.Align[
            cute.struct.MemRange[f32, G_STAGE * 3 * CHUNK], 128]
        sB: cute.struct.Align[
            cute.struct.MemRange[f32, B_STAGE * CHUNK], 128]

    return dict(q_l=q_l, k_l=k_l, v_l=v_l, o_l=o_l, a_l=a_l, t_l=t_l,
                kt_l=kt_l, vt_l=vt_l, ot_l=ot_l,
                tb8_l=tb8_l, tt8_l=tt8_l, t16_l=t16_l, tt16_l=tt16_l,
                t16x32_l=t16x32_l, tt16x32_l=tt16x32_l, tt32x32_l=tt32x32_l,
                Storage=Storage)


# ------------------------------------------------------------------------------
# kernel-side helpers
# ------------------------------------------------------------------------------
def _bar(bt, idx):
    """Pointer to barrier `idx` of a (1, N) barrier tensor."""
    return bt[(None, idx)].iterator


@cute.jit
def _bar_wait(bar, phase):
    """Non-blocking try-wait + explicit tight spin (hand-allocated barriers)."""
    ok = cute.arch.mbarrier_try_wait(bar, phase)
    while ok == 0:
        ok = cute.arch.mbarrier_try_wait(bar, phase)


@cute.jit
def _pack_bf16(acc, frg):
    """f32 accumulator -> bf16 wgmma A operand, one F2FP per pair.

    `@cute.jit` is load-bearing, not decoration: the preprocessor that rewrites
    `range_constexpr` only walks the source of decorated functions, so an
    undecorated helper carrying one dies at trace time with
    `range_constexpr should be preprocessed by preprocessor`.  Writes still land
    in the caller's fragment across the jit boundary.

    The two tensors alias the same element order (compact col-major), so the
    linear (2, p) reinterpretation is the bijection both call sites agree on --
    this is verbatim the mapping certified by probe X2/X3/X4.
    """
    p = cute.size(acc) // 2
    src = cute.make_tensor(acc.iterator, cute.make_layout((2, p)))
    dst = cute.make_tensor(frg.iterator, cute.make_layout((2, p)))
    for j in cutlass.range_constexpr(p):
        v2 = cute.arch.cvt_f32x2_bf16x2(src[(None, j)].load())
        dst[(None, j)].store(cute.TensorSSA(v2, (2,), cutlass.BFloat16))


@cute.jit
def _pack_f16(acc, frg):
    """f32 accumulator -> fp16 wgmma A operand, one `cvt.rn.f16x2.f32` per PAIR.

    The fp16 mirror of `_pack_bf16` (D2.3). Same bijection -- both tensors alias
    the same compact element order, so the linear reinterpretation is the mapping
    the two call sites agree on -- with only the narrowing instruction changed;
    the destination is recast to `Int32` because the packed word is what the asm
    produces. Certified together with the descriptor side by
    `notes/probe_d23_fp16_chain3.py` (Y1 exact, Y2/Y3 on the package's own
    algebra at both ring stages).
    """
    n = cute.size(acc)
    src = cute.make_tensor(acc.iterator, cute.make_layout(n))
    dst = cute.recast_tensor(
        cute.make_tensor(frg.iterator, cute.make_layout(n)), cutlass.Int32)
    for j in cutlass.range_constexpr(n // 2):
        dst[j] = _cvt_f16x2(src[2 * j], src[2 * j + 1])


@cute.jit
def _chain_ss(mma: cutlass.Constexpr, acc, frg_a, sa, frg_b, sb,
              accumulate: cutlass.Constexpr):
    """Issue one SS wgmma chain (both operands smem descriptors), then drain.

    `sa` / `sb` are the ring stages of the A and B fragments; the drain is
    `wait_group(0)` -- D0.11's starting wait tier, which D1.1 is the lever on.
    `mma` / `accumulate` stay `Constexpr` so the atom construction and the
    ACCUMULATE field keep the trace-time Python values the inlined form had.
    """
    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, accumulate)
    warpgroup.fence()
    for kb in cutlass.range_constexpr(cute.size(frg_b.shape[2])):
        cute.gemm(atom, acc, frg_a[(None, None, kb, sa)],
                  frg_b[(None, None, kb, sb)], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)


@cute.jit
def _gemm_rs(mma: cutlass.Constexpr, acc, frg_a, frg_b, sb,
             accumulate: cutlass.Constexpr):
    """Put one RS wgmma chain into the currently OPEN commit group.

    No fence and no commit of its own: D1.1's merged pair issues two chains
    between one `warpgroup.fence()` and one `commit_group()`, so the fence (which
    orders the register writes that feed the A operand against the async proxy)
    belongs to the caller and there is exactly one of it. A second fence BETWEEN
    two chains of the same group would be wrong, not merely redundant -- nothing
    writes the A registers between them.
    """
    atom = cute.make_mma_atom(mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, accumulate)
    for kb in cutlass.range_constexpr(cute.size(frg_b.shape[2])):
        cute.gemm(atom, acc, frg_a[(None, None, kb)],
                  frg_b[(None, None, kb, sb)], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)


@cute.jit
def _issue_rs(mma: cutlass.Constexpr, acc, frg_a, frg_b, sb,
              accumulate: cutlass.Constexpr):
    """Issue one RS wgmma chain (A from registers, B a ring descriptor) alone.

    Fence, chain, commit -- the drain is the caller's, so D0.11's segment D can
    put its elementwise between the commit and the `wait_group(0)`.
    """
    warpgroup.fence()
    _gemm_rs(mma, acc, frg_a, frg_b, sb, accumulate)
    warpgroup.commit_group()


@cute.jit
def _tri_z(mma: cutlass.Constexpr, tcA: cutlass.Constexpr,
           tcB: cutlass.Constexpr, A, B1, B2, c16, lane,
           MN1: cutlass.Constexpr, MN2: cutlass.Constexpr):
    """One doubling level's corner block: `Z = -(T22 . A21) . T11`, into `c16`.

    Two warp-level GEMMs with the first's accumulator chained straight into the
    second's A operand -- no smem round trip between them, which is legal here
    because every level has GEMM2's M == GEMM1's M and GEMM2's K == GEMM1's N
    (KB tri_inverse_block_doubling gotcha 8; D1.4's full-K h = 32 level keeps it
    true at MN1 = (16, 32) / MN2 = (16, 16)). `MN1` / `MN2` are the two
    accumulator shapes.

    The chaining view is built from the fragments' own sizes rather than a
    written-out ratio: `m16n8k8` chains at ratio 1 and `m16n8k16` at ratio 2, so
    a hard-coded ratio silently breaks when the level changes (gotcha 7).

    `c16` is the caller's fragment (the same pattern `_pack_bf16` uses) so the
    output shape stays owned by the segment that stores it.
    """
    thr = mma.get_slice(lane)
    thrA, thrB = tcA.get_slice(lane), tcB.get_slice(lane)
    frA = thr.make_fragment_A(thr.partition_A(A))
    cute.copy(tcA, thrA.partition_S(A), thrA.retile(frA))
    frB1 = thr.make_fragment_B(thr.partition_B(B1))
    cute.copy(tcB, thrB.partition_S(B1), thrB.retile(frB1))
    frB2 = thr.make_fragment_B(thr.partition_B(B2))
    cute.copy(tcB, thrB.partition_S(B2), thrB.retile(frB2))

    acc1 = _rmem(thr.partition_shape_C(MN1), cutlass.Float32)
    acc1.fill(0.0)
    cute.gemm(mma, acc1, frA, frB1, acc1)                   # P = T22 . A21
    for e in cutlass.range_constexpr(cute.size(acc1)):
        acc1[e] = -acc1[e]
    opA = thr.make_fragment_A(thr.partition_shape_A(MN1))
    cv = cute.size(acc1, mode=[0])
    av = cute.size(opA, mode=[0])
    ratio = av // cv
    view = cute.make_tensor(
        opA.iterator,
        cute.make_layout((cv, cute.size(acc1, mode=[1]), ratio,
                          cute.size(acc1, mode=[2]) // ratio),
                         stride=(1, av, cv, av * cute.size(acc1, mode=[1]))))
    for e in cutlass.range_constexpr(cute.size(acc1)):
        view[e] = cutlass.Float16(acc1[e])
    acc2 = _rmem(thr.partition_shape_C(MN2), cutlass.Float32)
    acc2.fill(0.0)
    cute.gemm(mma, acc2, opA, frB2, acc2)                   # Z = P . T11
    for e in cutlass.range_constexpr(cute.size(acc2)):
        c16[e] = cutlass.Float16(acc2[e])


@cute.jit
def _pp_take(half):
    """D0.10 SYNC(own): block until the sibling warpgroup yields the pipe.

    The predicate is the ping-pong parity, exactly the reference's `@P0` /
    `@!P0` split: WG1 (half == 0) only ever blocks on 0x4, WG2 only ever blocks
    on 0x5. `half` is warpgroup-uniform (it comes off a `make_warp_uniform`ed
    warp index), so every thread of a warpgroup takes the same arm.
    """
    if half == 0:
        cute.arch.barrier(barrier_id=BAR_PP_1, number_of_threads=PP_WIDTH)
    else:
        cute.arch.barrier(barrier_id=BAR_PP_2, number_of_threads=PP_WIDTH)


@cute.jit
def _pp_give(half):
    """D0.10 ARV(other): arrive-only, never blocks -- the other half of the pair.

    WG1 arrives on 0x5, WG2 on 0x4, and neither ever arrives on the barrier it
    blocks on: that asymmetry is what makes the hand-off a publish rather than a
    rendezvous.
    """
    if half == 0:
        cute.arch.barrier_arrive(barrier_id=BAR_PP_2,
                                 number_of_threads=PP_WIDTH)
    else:
        cute.arch.barrier_arrive(barrier_id=BAR_PP_1,
                                 number_of_threads=PP_WIDTH)


@cute.jit
def _state_bar(half):
    """D0.12 step 2: the transpose rendezvous, private to one consumer WG.

    `half` is a runtime (warpgroup-uniform) value, so the id has to be picked by
    a predicate rather than by indexing a Python tuple -- same shape as
    `_pp_take` / `_pp_give`.
    """
    if half == 0:
        cute.arch.barrier(barrier_id=BAR_STATE_1, number_of_threads=128)
    else:
        cute.arch.barrier(barrier_id=BAR_STATE_2, number_of_threads=128)


# ------------------------------------------------------------------------------
# the kernel
# ------------------------------------------------------------------------------
@cute.kernel
def _gdn_kernel(
    tma_atom_q, tma_q,
    tma_atom_k, tma_k,
    tma_atom_v, tma_v,
    tma_atom_o, tma_o,
    mAl: cute.Tensor, mBe: cute.Tensor, mSt: cute.Tensor,
    scale: cutlass.Float32,
    mma_ss: cute.TiledMma,          # solver: desc A, desc B, (64, 64)
    mma_rs: cute.TiledMma,          # consumer chains 1/2/4: REG A, desc B, bf16
    mma_rf: cute.TiledMma,          # consumer chain 3: REG A, desc B, fp16 (D2.3)
    mma_mn: cute.TiledMma,          # consumer chain 5: REG A, MN-major B, (64,128)
    tiled_ldsm: cute.TiledCopy,     # v^T out of the v ring, transposed ldmatrix
    tiled_stsm: cute.TiledCopy,     # o^T into the O slot, transposed stmatrix
    # --- D0.9 triangular inverse: the exchange-face copies and the three
    # --- warp-level mma / copy sets the doubling levels use -------------------
    r2s_f16: cute.TiledCopy,        # C fragment -> fp16 face  (seg 1 entry)
    r2s_bf16: cute.TiledCopy,       # C fragment -> bf16 face  (A_qk out)
    mma8: cute.TiledMma,            # m16n8k8   (h = 8)
    tc_a8: cute.TiledCopy,
    tc_b8: cute.TiledCopy,
    tc_c8: cute.TiledCopy,
    mma16: cute.TiledMma,           # m16n8k16  (h = 16, h = 32)
    tc_a16: cute.TiledCopy,
    tc_b16: cute.TiledCopy,
    tc_cst: cute.TiledCopy,
    q_l: cute.ComposedLayout,
    k_l: cute.ComposedLayout,
    v_l: cute.ComposedLayout,
    o_l: cute.ComposedLayout,
    a_l: cute.ComposedLayout,
    t_l: cute.ComposedLayout,
    kt_l: cute.ComposedLayout,
    vt_l: cute.ComposedLayout,
    ot_l: cute.ComposedLayout,
    tb8_l: cute.Layout,
    tt8_l: cute.Layout,
    t16_l: cute.Layout,
    tt16_l: cute.Layout,
    t16x32_l: cute.Layout,
    tt16x32_l: cute.Layout,
    tt32x32_l: cute.Layout,
    Storage: cutlass.Constexpr[Callable],
    n_chunks: cutlass.Constexpr,
    rep: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    hv = bidx                       # D0.1: one CTA per v-head (nseq == 1)
    lane = tidx % 32
    wg_idx = warp_idx // 4          # tid >> 7
    lt = tidx % 128                 # lane within the warpgroup

    # ---------------- shared prologue (stays ahead of the single role branch,
    # ---------------- so the later setmaxnreg dec/inc cannot swallow it) -------
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(Storage)

    b_qf = storage.b_qf.get_tensor(cute.make_layout((1, Q_STAGE)))
    b_qe = storage.b_qe.get_tensor(cute.make_layout((1, Q_STAGE)))
    b_kf = storage.b_kf.get_tensor(cute.make_layout((1, K_STAGE)))
    b_ke = storage.b_ke.get_tensor(cute.make_layout((1, K_STAGE)))
    b_vf = storage.b_vf.get_tensor(cute.make_layout((1, V_STAGE)))
    b_ve = storage.b_ve.get_tensor(cute.make_layout((1, V_STAGE)))
    b_af = storage.b_af.get_tensor(cute.make_layout((1, AQK_STAGE)))
    b_ae = storage.b_ae.get_tensor(cute.make_layout((1, AQK_STAGE)))
    b_tf = storage.b_tf.get_tensor(cute.make_layout((1, T_STAGE)))
    b_te = storage.b_te.get_tensor(cute.make_layout((1, T_STAGE)))
    b_of = storage.b_of.get_tensor(cute.make_layout((1, O_STAGE)))
    b_oe = storage.b_oe.get_tensor(cute.make_layout((1, O_STAGE)))
    b_gf = storage.b_gf.get_tensor(cute.make_layout((1, G_STAGE)))
    b_ge = storage.b_ge.get_tensor(cute.make_layout((1, G_STAGE)))
    b_bf = storage.b_bf.get_tensor(cute.make_layout((1, B_STAGE)))
    b_be = storage.b_be.get_tensor(cute.make_layout((1, B_STAGE)))

    sQ = storage.sQ.get_tensor(q_l.outer, swizzle=q_l.inner)
    sK = storage.sK.get_tensor(k_l.outer, swizzle=k_l.inner)
    sV = storage.sV.get_tensor(v_l.outer, swizzle=v_l.inner)
    sA = storage.sA.get_tensor(a_l.outer, swizzle=a_l.inner)
    sO = storage.sO.get_tensor(o_l.outer, swizzle=o_l.inner)
    # D0.9 / D2.2: the T slot is read as fp16 by everyone. The inverse computes
    # in fp16 in place, and since D2.2 the fp16 face IS the deliverable -- the
    # solver no longer reads it back to narrow it to bf16, and the consumer's
    # chain 3 puts its wgmma B descriptor on these very bytes with the fp16
    # reading (same MemRange, same swizzle, same trailing stage mode; only the
    # element interpretation differs, KB tri_inverse_block_doubling fact 3). The
    # bf16 reading of this allocation no longer exists.
    f16 = cutlass.Float16
    sTh = storage.sT.get_tensor(t_l.outer, swizzle=t_l.inner, dtype=f16)
    sTb8 = storage.sT.get_tensor(tb8_l, swizzle=t_l.inner, dtype=f16)
    sTt8 = storage.sT.get_tensor(tt8_l, swizzle=t_l.inner, dtype=f16)
    sT16 = storage.sT.get_tensor(t16_l, swizzle=t_l.inner, dtype=f16)
    sTt16 = storage.sT.get_tensor(tt16_l, swizzle=t_l.inner, dtype=f16)
    sT16x32 = storage.sT.get_tensor(t16x32_l, swizzle=t_l.inner, dtype=f16)
    sTt16x32 = storage.sT.get_tensor(tt16x32_l, swizzle=t_l.inner, dtype=f16)
    sTt32x32 = storage.sT.get_tensor(tt32x32_l, swizzle=t_l.inner, dtype=f16)
    # second readings of the k / v / O allocations (no extra smem)
    sKt = storage.sK.get_tensor(kt_l.outer, swizzle=kt_l.inner)
    sVt = storage.sV.get_tensor(vt_l.outer, swizzle=vt_l.inner)
    sOt = storage.sO.get_tensor(ot_l.outer, swizzle=ot_l.inner)
    # D0.12 step 2: after the last chunk the O staging ring is idle, so the two
    # slots (2 x 16 KB = 8,192 f32) are the transpose scratch for final_state --
    # 4,096 f32 per consumer warpgroup, i.e. one 64k x 64v batch each, two
    # batches to cover S^T's 128 k. Same allocation, an f32 reading of it; no
    # buffer is opened (D0.7: an unbacked buffer may not be opened).
    sOf = cute.recast_tensor(
        storage.sO.get_tensor(cute.make_layout(CHUNK * DHEAD * O_STAGE)),
        cutlass.Float32)
    sSt = cute.make_tensor(
        sOf.iterator,
        cute.make_layout((CHUNK, CHUNK, 2), stride=(CHUNK, 1, CHUNK * CHUNK)))
    # gate tables: row 0 cum, row 1 exp2(cum), row 2 exp2(cum)*scale
    sG = storage.sG.get_tensor(
        cute.make_layout((3, CHUNK, G_STAGE), stride=(CHUNK, 1, 3 * CHUNK)))
    sB = storage.sB.get_tensor(
        cute.make_layout((CHUNK, B_STAGE), stride=(1, CHUNK)))

    if warp_idx == 0:
        # tensormap prefetch belongs to the shared prologue (ahead of the role
        # branch and therefore ahead of the register tiering).
        cpasync.prefetch_descriptor(tma_atom_q)
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_v)
        cpasync.prefetch_descriptor(tma_atom_o)
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(Q_STAGE):
                cute.arch.mbarrier_init(_bar(b_qf, s), Q_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_qe, s), Q_EMPTY_CNT)
            for s in cutlass.range_constexpr(K_STAGE):
                cute.arch.mbarrier_init(_bar(b_kf, s), K_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_ke, s), K_EMPTY_CNT)
            for s in cutlass.range_constexpr(V_STAGE):
                cute.arch.mbarrier_init(_bar(b_vf, s), V_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_ve, s), V_EMPTY_CNT)
            for s in cutlass.range_constexpr(AQK_STAGE):
                cute.arch.mbarrier_init(_bar(b_af, s), AQK_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_ae, s), AQK_EMPTY_CNT)
            for s in cutlass.range_constexpr(T_STAGE):
                cute.arch.mbarrier_init(_bar(b_tf, s), T_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_te, s), T_EMPTY_CNT)
            for s in cutlass.range_constexpr(O_STAGE):
                cute.arch.mbarrier_init(_bar(b_of, s), O_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_oe, s), O_EMPTY_CNT)
            for s in cutlass.range_constexpr(G_STAGE):
                cute.arch.mbarrier_init(_bar(b_gf, s), G_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_ge, s), G_EMPTY_CNT)
            for s in cutlass.range_constexpr(B_STAGE):
                cute.arch.mbarrier_init(_bar(b_bf, s), B_FULL_CNT)
                cute.arch.mbarrier_init(_bar(b_be, s), B_EMPTY_CNT)

    # gmem tiles: (CHUNK, DHEAD, n_chunks) views of the CTA's own head
    hqk = hv // rep
    gQ = cute.local_tile(tma_q, (CHUNK, DHEAD), (None, 0, hqk))
    gK = cute.local_tile(tma_k, (CHUNK, DHEAD), (None, 0, hqk))
    gV = cute.local_tile(tma_v, (CHUNK, DHEAD), (None, 0, hv))
    gO = cute.local_tile(tma_o, (CHUNK, DHEAD), (None, 0, hv))

    tAsQ, tAgQ = cpasync.tma_partition(
        tma_atom_q, 0, cute.make_layout(1),
        cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2))
    tAsK, tAgK = cpasync.tma_partition(
        tma_atom_k, 0, cute.make_layout(1),
        cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
    tAsV, tAgV = cpasync.tma_partition(
        tma_atom_v, 0, cute.make_layout(1),
        cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))
    # D0.12 step 1: the S2G side of the same partitioning, bound here in the
    # shared prologue (not inside the store warp's arm) so no bulky DSL object
    # is first bound inside a role branch.
    tAsO, tAgO = cpasync.tma_partition(
        tma_atom_o, 0, cute.make_layout(1),
        cute.group_modes(sO, 0, 2), cute.group_modes(gO, 0, 2))

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # ---------------- the single role branch (D0.3) ---------------------------
    # D0.6: the register hand-over is the first statement of each branch body,
    # and every warpgroup executes exactly one of them (`setmaxnreg` is
    # `.sync.aligned`, so the predicate has to be warpgroup-uniform -- `wg_idx`
    # is derived from a `make_warp_uniform`ed warp index).
    if wg_idx == 0:
        _reg_dec(WG0_REGS)
        # WG0: the four service warps.
        if warp_idx == 0:
            # --- TMA load producer: q/k/v rings, one expect-tx per ring -------
            _seg("supply.enter")
            for i in cutlass.range(n_chunks):
                qs = i % Q_STAGE
                qp = (i // Q_STAGE) % 2
                ks = i % K_STAGE
                kp = (i // K_STAGE) % 2
                vs = i % V_STAGE
                vp = (i // V_STAGE) % 2

                _seg("supply.q.empty.begin", i)
                _bar_wait(_bar(b_qe, qs), 1 - qp)
                _seg("supply.q.empty.end", i)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        _bar(b_qf, qs), TILE_BYTES)
                cute.copy(tma_atom_q, tAgQ[(None, i)], tAsQ[(None, qs)],
                          tma_bar_ptr=_bar(b_qf, qs))

                _seg("supply.k.empty.begin", i)
                _bar_wait(_bar(b_ke, ks), 1 - kp)
                _seg("supply.k.empty.end", i)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        _bar(b_kf, ks), TILE_BYTES)
                cute.copy(tma_atom_k, tAgK[(None, i)], tAsK[(None, ks)],
                          tma_bar_ptr=_bar(b_kf, ks))

                _seg("supply.v.empty.begin", i)
                _bar_wait(_bar(b_ve, vs), 1 - vp)
                _seg("supply.v.empty.end", i)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        _bar(b_vf, vs), TILE_BYTES)
                cute.copy(tma_atom_v, tAgV[(None, i)], tAsV[(None, vs)],
                          tma_bar_ptr=_bar(b_vf, vs))

        elif warp_idx == 1:
            # --- D0.12 TMA store warp: one bulk S2G per chunk ------------------
            # The whole warp issues the copy: the CuTeDSL TMA atom elects a lane
            # internally with a full-warp member mask, so a single-lane `if`
            # would be UB (KB tma_store_epilogue gotcha 6). The non-electing
            # lanes hold empty bulk-groups, so their `wait_group(0)` returns at
            # once -- hence the width-32 rendezvous before the 32 O-empty
            # arrivals, which is what actually makes the release mean "the store
            # has drained".
            _seg("store.enter")
            for i in cutlass.range(n_chunks):
                os = i % O_STAGE
                op = (i // O_STAGE) % 2
                _seg("store.o.full.begin", i)
                _bar_wait(_bar(b_of, os), op)
                _seg("store.o.full.end", i)
                cute.arch.fence_view_async_shared()
                cute.copy(tma_atom_o, tAsO[(None, os)], tAgO[(None, i)])
                cute.arch.cp_async_bulk_commit_group()
                _seg("store.o.drain.begin", i)
                cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.barrier(barrier_id=BAR_STORE, number_of_threads=32)
                _seg("store.o.drain.end", i)
                cute.arch.mbarrier_arrive(_bar(b_oe, os))

        elif warp_idx == 2:
            # --- D0.5 gate warp: three 64-f32 chunk tables per chunk ----------
            # 32 lanes take the two log2s each; the prefix scan itself is
            # serial, so lane 0 walks the cum row in place between two width-32
            # rendezvous; then the 32 lanes take the two exp2 pairs each.
            _seg("gate.enter")
            for i in cutlass.range(n_chunks):
                gs = i % G_STAGE
                gp = (i // G_STAGE) % 2
                _seg("gate.g.empty.begin", i)
                _bar_wait(_bar(b_ge, gs), 1 - gp)
                _seg("gate.g.empty.end", i)

                base = i * CHUNK
                for j in cutlass.range_constexpr(2):
                    idx = j * 32 + lane
                    a = mAl[(base + idx, hv)] + cutlass.Float32(GATE_EPS)
                    sG[(0, idx, gs)] = _lg2(a)
                cute.arch.barrier(barrier_id=BAR_GATE, number_of_threads=32)
                if lane == 0:
                    run = cutlass.Float32(0.0)
                    for t in cutlass.range_constexpr(CHUNK):
                        run = run + sG[(0, t, gs)]
                        sG[(0, t, gs)] = run
                cute.arch.barrier(barrier_id=BAR_GATE, number_of_threads=32)
                for j in cutlass.range_constexpr(2):
                    idx = j * 32 + lane
                    e = _exp2(sG[(0, idx, gs)])
                    sG[(1, idx, gs)] = e
                    sG[(2, idx, gs)] = e * scale
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(_bar(b_gf, gs))

        elif warp_idx == 3:
            # --- D0.5 beta warp: one 64-f32 table per chunk -------------------
            _seg("beta.enter")
            for i in cutlass.range(n_chunks):
                bs = i % B_STAGE
                bp = (i // B_STAGE) % 2
                _seg("beta.b.empty.begin", i)
                _bar_wait(_bar(b_be, bs), 1 - bp)
                _seg("beta.b.empty.end", i)
                base = i * CHUNK
                for j in cutlass.range_constexpr(2):
                    idx = j * 32 + lane
                    sB[(idx, bs)] = mBe[(base + idx, hv)]
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(_bar(b_bf, bs))

    else:
        if wg_idx == 3:
            # 104 is BELOW the 128 entry allocation -> `.dec` (this tier is a
            # release, like WG0's; the consumer pair's 192 is the only `.inc`).
            _reg_dec(SOLVER_REGS)
            # ================= D0.9 solver warpgroup ==========================
            wgs = mma_ss.get_slice(0)
            thr_s = mma_ss.get_slice(lt)
            frgQ_s = wgs.make_fragment_A(wgs.partition_A(sQ))
            frgK_a = wgs.make_fragment_A(wgs.partition_A(sK))
            frgK_b = wgs.make_fragment_B(wgs.partition_B(sK))
            cS = thr_s.partition_C(cute.make_identity_tensor((CHUNK, CHUNK)))
            acc = _rmem(thr_s.partition_shape_C((CHUNK, CHUNK)),
                        cutlass.Float32)

            # --- D0.9 triangular inverse: the pieces that do not vary by chunk
            # The warpgroup's own warp index (0..3): the solver is WG3, so
            # `warp_idx` runs 12..15 and this stays warp-uniform because it is
            # derived from the already-uniform `warp_idx`.
            swid = warp_idx - 12
            thr_e16 = r2s_f16.get_slice(lt)         # C fragment -> fp16 face
            thr_ebf = r2s_bf16.get_slice(lt)        # C fragment -> bf16 face
            rs_acc16 = r2s_f16.retile(acc)
            rs_accbf = r2s_bf16.retile(acc)
            # The store op's own staging shape (stage-independent, so it is read
            # off stage 0 at trace time). Every fragment below is allocated once
            # here, outside the chunk loop -- the same shape the body would have
            # asked for each trip, without an alloca on the back edge.
            stg_shape = cute.make_layout(
                cute.shape(thr_e16.partition_S(sTh[(None, None, 0)]))[:3]).shape
            stg32 = _rmem(stg_shape, cutlass.Float32)
            stg16 = _rmem(stg_shape, cutlass.Float16)
            stgbf = _rmem(stg_shape, cutlass.BFloat16)
            thr8 = mma8.get_slice(lane)
            thr16 = mma16.get_slice(lane)
            z8 = _rmem(thr8.partition_shape_C((16, 8)), cutlass.Float16)
            z16 = _rmem(thr16.partition_shape_C((16, 16)), cutlass.Float16)
            # seg 2 keeps one row of an 8x8 block resident as four fp16 pairs.
            rRow = _rmem((8, 1), cutlass.Float16)
            rOut = _rmem((8, 1), cutlass.Float16)
            cdi = lane % 8                  # column inside the diagonal block
            blk = swid * 4 + lane // 8      # which of the 8 diagonal blocks
            dbase = lane - cdi              # the block's first lane
            drow = blk * 8 + cdi            # the row this lane back-substitutes
            nh = swid // 2                  # h=32: output N half
            yh = swid % 2                   # h=32: output M half

            _seg("solver.enter")
            for i in cutlass.range(n_chunks):
                qs = i % Q_STAGE
                qp = (i // Q_STAGE) % 2
                ks = i % K_STAGE
                kp = (i // K_STAGE) % 2
                ts = i % T_STAGE
                tp = (i // T_STAGE) % 2
                asl = i % AQK_STAGE
                ap = (i // AQK_STAGE) % 2
                gs = i % G_STAGE
                gp = (i // G_STAGE) % 2
                bs = i % B_STAGE
                bp = (i // B_STAGE) % 2

                _seg("solver.chain0.k_full.begin", i)
                _bar_wait(_bar(b_kf, ks), kp)
                _seg("solver.chain0.k_full.end", i)
                _seg("solver.gate.g_full.begin", i)
                _bar_wait(_bar(b_gf, gs), gp)
                _seg("solver.gate.g_full.end", i)
                _seg("solver.gate.b_full.begin", i)
                _bar_wait(_bar(b_bf, bs), bp)
                _seg("solver.gate.b_full.end", i)

                # chain 0: A_kk_raw = K K^T (both operands smem descriptors)
                _chain_ss(mma_ss, acc, frgK_a, ks, frgK_b, ks, False)

                # ---- seg 1: mask / decay / beta, then A_kk into the T slot ---
                # The elementwise pass ends in the narrowing, so the face is
                # written exactly once: the strictly-lower mask, the chunk decay
                # and beta_i all land on the accumulator IN PLACE, and the same
                # values go straight out through `STSM.16.M88.4` x4 as fp16 --
                # the precision D0.4 tables for A_kk on the way in, because the
                # inverse's conditioning is sensitive to mantissa bits.
                _seg("solver.akk.t_empty.begin", i)
                _bar_wait(_bar(b_te, ts), 1 - tp)
                _seg("solver.akk.t_empty.end", i)
                for e in cutlass.range_constexpr(cute.size(acc)):
                    ii, jj = cS[e]
                    d = _exp2(sG[(0, ii, gs)] - sG[(0, jj, gs)])
                    val = acc[e] * d * sB[(ii, bs)]
                    acc[e] = _sel_gt(ii, jj, val, cutlass.Float32(0.0))
                for e in cutlass.range_constexpr(cute.size(stg32)):
                    stg32[e] = rs_acc16[e]
                stg16.store(stg32.load().to(cutlass.Float16))
                cute.copy(r2s_f16, stg16,
                          thr_e16.partition_D(sTh[(None, None, ts)]))
                cute.arch.barrier(barrier_id=BAR_SOLVER,
                                  number_of_threads=128)             # bar 1

                # chain 1: A_qk_raw = Q K^T
                _seg("solver.chain1.q_full.begin", i)
                _bar_wait(_bar(b_qf, qs), qp)
                _seg("solver.chain1.q_full.end", i)
                _chain_ss(mma_ss, acc, frgQ_s, qs, frgK_b, ks, False)
                if lane == 0:
                    cute.arch.mbarrier_arrive(_bar(b_qe, qs))

                _seg("solver.aqk.a_empty.begin", i)
                _bar_wait(_bar(b_ae, asl), 1 - ap)
                _seg("solver.aqk.a_empty.end", i)
                # Same shape as the A_kk face: mask (inclusive of the diagonal
                # here) + decay + the intra half of the readout scale in place on
                # the accumulator, then one `STSM.16.M88.4` x4 out -- bf16,
                # because the consumer's chain 4 reads this slot as a wgmma
                # descriptor.
                for e in cutlass.range_constexpr(cute.size(acc)):
                    ii, jj = cS[e]
                    d = _exp2(sG[(0, ii, gs)] - sG[(0, jj, gs)])
                    val = acc[e] * d * scale
                    acc[e] = _sel_ge(ii, jj, val, cutlass.Float32(0.0))
                for e in cutlass.range_constexpr(cute.size(stg32)):
                    stg32[e] = rs_accbf[e]
                stgbf.store(stg32.load().to(cutlass.BFloat16))
                cute.copy(r2s_bf16, stgbf,
                          thr_ebf.partition_D(sA[(None, None, asl)]))
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(_bar(b_af, asl))
                if lane == 0:
                    cute.arch.mbarrier_arrive(_bar(b_ke, ks))

                # ---- triangular inverse (I + A_kk)^-1, in place on the T slot
                # D0.9's block-doubling form. The face already holds the masked
                # A_kk in fp16 (seg 1 above, and bar 1 with it), so what is left
                # is the diagonal back-substitution and the three doubling
                # levels; the diag(beta) column scale rides out on the exit.
                # `[[T11, 0], [A21, T22]]^-1` has corner `-T22 . A21 . T11`, so
                # every level computes only that corner and drops it onto A21's
                # own bytes -- which is why each level opens with a rendezvous
                # (the block it is about to overwrite is the previous level's
                # output and this level's operand).
                _seg("solver.solve.begin", i)

                # ---- seg 2: the 8 diagonal 8x8 blocks, warps 0/1 ------------
                # 8 blocks x 8 lanes = 64 lanes exactly, so two warps carry it
                # and the other two wait at bar 2; spreading it over four warps
                # would not shorten any lane's chain, only double the shuffles.
                # Each lane holds one ROW of its block (one `LDS.128`, four fp16
                # pairs) and broadcasts the packed word, so every unpack is off
                # the dependent chain.
                f8 = cute.flat_divide(sTh[(None, None, ts)], (8, 8))
                if swid < 2:
                    cute.autovec_copy(
                        f8[(drow % 8, None, drow // 8, blk)],
                        rRow[(None, 0)])
                    rW = cute.recast_tensor(rRow, cutlass.Int32)
                    yr = [cutlass.Float32(0.0)] * 8
                    for qq in cutlass.range_constexpr(4):
                        qw = 3 - qq
                        sodd = cutlass.Float32(0.0)
                        sevn = cutlass.Float32(0.0)
                        for tt in cutlass.range_constexpr(6 - 2 * qw):
                            jj = 7 - tt
                            wd = cute.arch.shuffle_sync(rW[(qw, 0)], dbase + jj)
                            sodd = sodd + _f16_hi(wd) * yr[jj]
                            sevn = sevn + _f16_lo(wd) * yr[jj]
                        yr[2 * qw + 1] = _sel_eq(2 * qw + 1, cdi,
                                                 cutlass.Float32(1.0), -sodd)
                        wd = cute.arch.shuffle_sync(rW[(qw, 0)],
                                                    dbase + 2 * qw + 1)
                        sevn = sevn + _f16_lo(wd) * yr[2 * qw + 1]
                        yr[2 * qw] = _sel_eq(2 * qw, cdi,
                                             cutlass.Float32(1.0), -sevn)
                    for e in cutlass.range_constexpr(8):
                        rOut[(e, 0)] = cutlass.Float16(yr[e])
                    cute.autovec_copy(
                        rOut[(None, 0)],
                        f8[(drow % 8, None, drow // 8, blk)])
                cute.arch.barrier(barrier_id=BAR_SOLVER,
                                  number_of_threads=128)             # bar 2

                # ---- seg 3: h = 8, one block pair per warp (D2.4 i) ---------
                # T22 is an 8-row block and `m16n8k8` wants 16, so the A reading
                # carries a stride-0 mode that aliases rows 8..15 onto 0..7 --
                # the whole partition is copied as usual, the second half is just
                # a second view of the same bytes.
                # There are exactly four block pairs at this level and four warps
                # in the solver, so `b3 = swid` gives each warp one and the
                # level's serial length halves against D1.4's form, where warps
                # 0/1 took two pairs each and warps 2/3 idled. What that form
                # bought was the merge with h = 16: a warp consumed only corners
                # it had written itself, so the two levels needed no rendezvous
                # between them. Spreading gives that back -- h = 16's warp b4
                # consumes pairs 2*b4 and 2*b4+1, which are now two DIFFERENT
                # warps' output -- so bar 3 separates the levels again and the
                # solver's rendezvous count per chunk goes 4 -> 5, still under
                # M0's seven. The trade is one barrier arrival against half a
                # level's dependent chain.
                b3 = swid
                _tri_z(mma8, tc_a8, tc_b8,
                       sTb8[(None, None, 2 * b3 + 1, 2 * b3 + 1, ts)],
                       sTt8[(None, None, 2 * b3, 2 * b3 + 1, ts)],
                       sTt8[(None, None, 2 * b3, 2 * b3, ts)],
                       z8, lane, (16, 8), (16, 8))
                cute.copy(tc_c8, tc_c8.get_slice(lane).retile(z8),
                          tc_c8.get_slice(lane).partition_D(
                              sTb8[(None, None, 2 * b3 + 1, 2 * b3, ts)]))
                cute.arch.barrier(barrier_id=BAR_SOLVER,
                                  number_of_threads=128)             # bar 3

                # ---- seg 4: h = 16, warps 0/1 -------------------------------
                # Warp b4 == swid takes the 16-wide corner over the two h = 8
                # pairs 2*b4 and 2*b4+1; bar 3 above is what makes the other
                # warp's half of that input visible.
                if swid < 2:
                    b4 = swid
                    _tri_z(mma16, tc_a16, tc_b16,
                           sT16[(None, None, 2 * b4 + 1, 2 * b4 + 1, ts)],
                           sTt16[(None, None, 2 * b4, 2 * b4 + 1, ts)],
                           sTt16[(None, None, 2 * b4, 2 * b4, ts)],
                           z16, lane, (16, 16), (16, 16))
                    cute.copy(tc_cst, tc_cst.get_slice(lane).retile(z16),
                              tc_cst.get_slice(lane).partition_D(
                                  sT16[(None, None, 2 * b4 + 1, 2 * b4, ts)]))
                cute.arch.barrier(barrier_id=BAR_SOLVER,
                                  number_of_threads=128)             # bar 4

                # ---- seg 5: h = 32, a 2x2 warp grid (D1.4 merge 2) ----------
                # The last level is one 32x32 corner, too big for one warp. The
                # grid is (yh = output M half, nh = output N half): GEMM1 runs at
                # the FULL K = 32 and N = 32, so the sum that the split-K form
                # pushed through smem (a cross-warp add cannot happen in
                # registers) lives in the accumulator itself and the level stores
                # exactly once. Each M half computes GEMM1 twice, once per N
                # partner -- 8 extra `m16n8k16` per warp bought against two
                # rendezvous, one `stmatrix` and one `ldmatrix`.
                _tri_z(mma16, tc_a16, tc_b16,
                       sT16x32[(None, None, 2 + yh, 1, ts)],
                       sTt32x32[(None, None, 0, 1, ts)],
                       sTt16x32[(None, None, nh, 0, ts)],
                       z16, lane, (16, 32), (16, 16))
                # bar 5: the strip about to be written IS this level's B operand
                # (A21) -- every warp must have loaded before any warp stores.
                cute.arch.barrier(barrier_id=BAR_SOLVER,
                                  number_of_threads=128)             # bar 5
                cute.copy(tc_cst, tc_cst.get_slice(lane).retile(z16),
                          tc_cst.get_slice(lane).partition_D(
                              sT16[(None, None, 2 + yh, nh, ts)]))
                _seg("solver.solve.end", i)

                # ---- exit: none. The fp16 face IS the deliverable (D2.2) -----
                # What used to stand here was a read-modify-write of the whole
                # face -- `ldmatrix` it back in C order, scale column j by
                # beta_j, narrow fp16 -> bf16, `stmatrix` it onto the same bytes
                # -- plus the rendezvous that had to precede it (the read spans
                # the whole face, so every warp's h = 32 store had to be visible
                # first). Both are gone: `(I + A_kk)^-1` in fp16, exactly as the
                # last doubling level left it, is what chain 3 consumes, and
                # diag(beta) is folded into the consumer's A'^T column scaling
                # instead (D2.3; w = Tinv . diag(beta) . A' = Tinv . (diag(beta)
                # . A'), so beta is still multiplied exactly once).
                #
                # The rendezvous is not replaced by anything, because T-full
                # already carries the same order for a stronger reason: it takes
                # all 128 solver arrivals, so a consumer that sees it has seen
                # every warp's store. Each thread fences its own store into the
                # async proxy and then arrives -- the solve keeps no exit
                # rendezvous, and the five above are all of them (D2.4 (i) put
                # back the one D1.4's h = 8 / h = 16 merge had removed).
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(_bar(b_tf, ts))

                cute.arch.mbarrier_arrive(_bar(b_ge, gs))
                cute.arch.mbarrier_arrive(_bar(b_be, bs))

        else:
            _reg_inc(CONSUMER_REGS)
            # ================= D0.2 consumer pair =============================
            # WG1 owns v-columns 0..63, WG2 owns 64..127; each keeps its own
            # S^T[64 v x 128 k] f32 accumulator resident and never spills it to
            # smem (ledger row X1).
            half = wg_idx - 1
            cbase = half * HALF

            wgr = mma_rs.get_slice(0)
            wgf = mma_rf.get_slice(0)
            wgm = mma_mn.get_slice(0)
            thr_r = mma_rs.get_slice(lt)
            thr_m = mma_mn.get_slice(lt)
            thr_l = tiled_ldsm.get_slice(lt)
            thr_s = tiled_stsm.get_slice(lt)

            frgQ = wgr.make_fragment_B(wgr.partition_B(sQ))
            frgK = wgr.make_fragment_B(wgr.partition_B(sK))
            # D2.3: chain 3's B is the T slot read as fp16 -- the same bytes,
            # swizzle and trailing stage mode the bf16 descriptor used to name,
            # off the fp16 warpgroup mma. Both ring stages carry it (chain 3
            # issues against `ts = i % T_STAGE`), which is what probe contract Y3
            # pins.
            frgT = wgf.make_fragment_B(wgf.partition_B(sTh))
            frgA = wgr.make_fragment_B(wgr.partition_B(sA))
            frgKt = wgm.make_fragment_B(wgm.partition_B(sKt))

            # A operands, by the (M, K) each chain contracts. `rS` is the S^T
            # packing shared by chains 1 and 2 (64 v, 128 k); `rPf` is chain 3's
            # A'^T, now an fp16 fragment (D2.3), and `rP` chain 4's w^T, still
            # bf16 -- they are the one 64x64 packing slot split in two by dtype,
            # with the same disjoint live ranges as before (A'^T dies at chain
            # 3's drain, w^T is packed after it), so they share physical
            # registers instead of summing; `rPg` is chain 5's separately-scaled
            # w^T packing, which has to be an mma_mn fragment.
            rS = wgr.make_fragment_A(wgr.partition_shape_A((CHUNK, DHEAD)))
            rPf = wgf.make_fragment_A(wgf.partition_shape_A((CHUNK, CHUNK)))
            rP = wgr.make_fragment_A(wgr.partition_shape_A((CHUNK, CHUNK)))
            rPg = wgm.make_fragment_A(wgm.partition_shape_A((CHUNK, CHUNK)))

            # Accumulator base = 128 f32: S^T 64 + o^T 32 + ONE shared 32-reg
            # slot that carries (kS)^T from chain 2 and then w^T from chain 3
            # ((kS)^T dies where it is packed into rP, so the two never overlap).
            accS = _rmem(thr_m.partition_shape_C((CHUNK, DHEAD)),
                         cutlass.Float32)
            accO = _rmem(thr_r.partition_shape_C((CHUNK, CHUNK)),
                         cutlass.Float32)
            accW = _rmem(thr_r.partition_shape_C((CHUNK, CHUNK)),
                         cutlass.Float32)
            rV = _rmem(thr_r.partition_shape_C((CHUNK, CHUNK)),
                       cutlass.BFloat16)
            # D0.12's o packing: same C-fragment shape as rV and a live range
            # that starts where rV's ends (rV dies in segment C, rO is born in
            # segment F), so the two share physical registers and segment F adds
            # nothing to D0.11's working-set budget.
            rO = _rmem(thr_r.partition_shape_C((CHUNK, CHUNK)),
                       cutlass.BFloat16)

            cC = thr_r.partition_C(cute.make_identity_tensor((CHUNK, CHUNK)))
            cS = thr_m.partition_C(cute.make_identity_tensor((CHUNK, DHEAD)))

            # accS is deliberately NOT zeroed: D0.13's first-chunk peel opens it
            # with chain 5's `ACCUMULATE = False` (the reference's `RZ, !UPT`),
            # which writes the whole 64x128 accumulator.

            # D0.10 prologue stagger: WG2 hands WG1 the initial token, so the
            # very first `SYNC 0x4` has something to consume. From here the
            # baton alternates and the pair stays exactly one chain apart.
            if half == 1:
                cute.arch.barrier_arrive(barrier_id=BAR_PP_1,
                                         number_of_threads=PP_WIDTH)

            _seg("consumer.enter")

            # =============== D0.13 first-chunk peel (chunk 0, S_0 = 0) ========
            # S_0 = 0 collapses the chunk algebra: o_inter^T = S^T q^T and
            # (kS)^T = S^T k^T are both identically zero, so chains 1 and 2 are
            # dead (16 HGMMA saved), A' = v^T needs no subtraction and no e[]
            # lookup, o carries no inter term (so chain 4 OPENS o^T instead of
            # accumulating onto it, and the es[] column scaling of segment D
            # disappears with it), and S_C has no gamma_C S_0 term (so chain 5
            # OPENS S^T -- this is the peel's zero-initialized accumulator).
            # Every wait still points at chunk 0 and still sits at the top of
            # its segment (D0.11), and both warpgroups run this same peel, so
            # the ping-pong baton stays balanced: 3 take/give pairs here where
            # the steady body has 5.
            _seg("consumer.A.g_full.begin", 0)
            _bar_wait(_bar(b_gf, 0), 0)
            _seg("consumer.A.g_full.end", 0)
            # q[0] is never read here, so the peel only hands the slot back.
            # Releasing a slot it never read is safe: q-empty[0] takes all 12
            # arrivals to flip and the solver's four land after its own chain 1
            # has drained, so the producer cannot overwrite the tile early.
            if lane == 0:
                cute.arch.mbarrier_arrive(_bar(b_qe, 0))
            # k[0] is still needed -- it is chain 5's B operand.
            _seg("consumer.B.k_full.begin", 0)
            _bar_wait(_bar(b_kf, 0), 0)
            _seg("consumer.B.k_full.end", 0)

            # ---- segment C: A'^T = beta (x) v^T (no (kS) term to subtract) ---
            _seg("consumer.C.v_full.begin", 0)
            _bar_wait(_bar(b_vf, 0), 0)
            _seg("consumer.C.v_full.end", 0)
            # D2.3: the peel needs the beta table too -- diag(beta) now rides on
            # A'^T's columns instead of on T, and that is true of the peel's
            # A' = v^T as much as of the steady body's.
            _seg("consumer.C.b_full.begin", 0)
            _bar_wait(_bar(b_bf, 0), 0)
            _seg("consumer.C.b_full.end", 0)
            cute.copy(tiled_ldsm,
                      thr_l.partition_S(sVt[(None, None, half, 0)]),
                      thr_l.retile(rV))
            # A' = beta (x) v^T, i.e. the steady body's segment C with the (kS)^T
            # term dropped rather than a different path: same fragment, same f32
            # staging, same single fp16 packing into chain 3's A operand.
            for e in cutlass.range_constexpr(cute.size(accW)):
                _, tok = cC[e]
                accW[e] = cutlass.Float32(rV[e]) * sB[(tok, 0)]
            _pack_f16(accW, rPf)
            cute.arch.mbarrier_arrive(_bar(b_be, 0))
            _seg("consumer.C.t_full.begin", 0)
            _bar_wait(_bar(b_tf, 0), 0)
            _seg("consumer.C.t_full.end", 0)

            # ---- segment D: chain 3, w^T = A'^T . T^T (fp16, D2.3) -----------
            _seg("consumer.D.yield.begin", 0)
            _pp_take(half)
            _seg("consumer.D.yield.end", 0)
            _issue_rs(mma_rf, accW, rPf, frgT, 0, False)
            _seg("consumer.D.chain3.commit", 0)
            _pp_give(half)
            warpgroup.wait_group(0)
            _seg("consumer.D.chain3.drain", 0)

            # ---- segment E: chain 4, o^T = w^T . A_qk^T (opens, not +=) ------
            if lane == 0:
                cute.arch.mbarrier_arrive(_bar(b_ve, 0))
            cute.arch.mbarrier_arrive(_bar(b_te, 0))
            _pack_bf16(accW, rP)
            _seg("consumer.E.a_full.begin", 0)
            _bar_wait(_bar(b_af, 0), 0)
            _seg("consumer.E.a_full.end", 0)
            _seg("consumer.E.yield.begin", 0)
            _pp_take(half)
            _seg("consumer.E.yield.end", 0)
            _issue_rs(mma_rs, accO, rP, frgA, 0, False)
            _seg("consumer.E.chain4.commit", 0)
            _pp_give(half)
            warpgroup.wait_group(0)
            _seg("consumer.E.chain4.drain", 0)
            cute.arch.mbarrier_arrive(_bar(b_ae, 0))

            # ---- segment F: publish o (unchanged from the steady body) -------
            _seg("consumer.F.o_empty.begin", 0)
            _bar_wait(_bar(b_oe, 0), 1)
            _seg("consumer.F.o_empty.end", 0)
            _pack_bf16(accO, rO)
            cute.copy(tiled_stsm, thr_s.retile(rO),
                      thr_s.partition_D(sOt[(None, None, half, 0)]))
            cute.arch.fence_view_async_shared()
            _seg("consumer.F.o_full.arrive", 0)
            cute.arch.mbarrier_arrive(_bar(b_of, 0))

            # ---- segment G: chain 5 opens S^T (nothing to decay) -------------
            glast0 = sG[(0, CHUNK - 1, 0)]
            for e in cutlass.range_constexpr(cute.size(accW)):
                _, tok = cC[e]
                accW[e] = accW[e] * _exp2(glast0 - sG[(0, tok, 0)])
            _pack_bf16(accW, rPg)
            _seg("consumer.G.yield.begin", 0)
            _pp_take(half)
            _seg("consumer.G.yield.end", 0)
            _issue_rs(mma_mn, accS, rPg, frgKt, 0, False)
            _seg("consumer.G.chain5.commit", 0)
            _pp_give(half)
            warpgroup.wait_group(0)
            _seg("consumer.G.chain5.drain", 0)
            if lane == 0:
                cute.arch.mbarrier_arrive(_bar(b_ke, 0))
            cute.arch.mbarrier_arrive(_bar(b_ge, 0))

            # =============== steady body, chunks 1 .. n_chunks-1 ==============
            # The trip index is offset by the peeled chunk (a one-argument
            # `cutlass.range` is the spelling this DSL build is exercised on).
            for tstep in cutlass.range(n_chunks - 1):
                i = tstep + 1
                qs = i % Q_STAGE
                qp = (i // Q_STAGE) % 2
                ks = i % K_STAGE
                kp = (i // K_STAGE) % 2
                vs = i % V_STAGE
                vp = (i // V_STAGE) % 2
                ts = i % T_STAGE
                tp = (i // T_STAGE) % 2
                asl = i % AQK_STAGE
                ap = (i // AQK_STAGE) % 2
                gs = i % G_STAGE
                gp = (i // G_STAGE) % 2
                bs = i % B_STAGE
                bp = (i // B_STAGE) % 2
                os = i % O_STAGE
                op = (i // O_STAGE) % 2

                # ---- segments A+B: chains 1 and 2, ONE group (D1.1 (a) ⊗ D1.3)
                # The two chains are data independent -- both take the SAME S^T
                # packing as their A operand (`rS`, written once before either
                # issues), they read different rings (q and k), and they land in
                # different accumulators (o_inter^T in accO, (kS)^T in accW) --
                # so nothing between them has to be ordered and they go into one
                # commit group behind one `warpgroup.fence()`, drained by one
                # `wait_group(0)`. That saves one hand-off pair and one pipeline
                # drain per chunk, and doubles the sibling's elementwise window.
                #
                # D1.3 is the ⊗ half of the same change and is not optional: the
                # yield point has to coarsen with the merge. A yield left BETWEEN
                # the two chains would let the sibling's `SYNC` pass while this
                # warpgroup still had chain 2 unissued, so both halves of the pair
                # would be issuing into the tensor pipe at once -- the merge's
                # gain handed straight back as contention. One take before the
                # fence, one give after the commit: 4 BAR.SYNC per steady chunk
                # where D0.11 had 5.
                #
                # Wait targets are D0.11's, unchanged in kind: all three still
                # name chunk i and all three still sit at the top of the segment
                # they gate. k[i]-full simply joins them, having moved up from
                # what used to be segment B's head; the k ring is therefore held
                # from here rather than from half a chain later, which is the
                # cost side of this merge (the release point at segment G is
                # unmoved). q[i] is released after the merged drain, the first
                # point at which chain 1 has actually read it.
                _seg("consumer.A.q_full.begin", i)
                _bar_wait(_bar(b_qf, qs), qp)
                _seg("consumer.A.q_full.end", i)
                _seg("consumer.A.g_full.begin", i)
                _bar_wait(_bar(b_gf, gs), gp)
                _seg("consumer.A.g_full.end", i)
                _seg("consumer.B.k_full.begin", i)
                _bar_wait(_bar(b_kf, ks), kp)
                _seg("consumer.B.k_full.end", i)
                _pack_bf16(accS, rS)
                _seg("consumer.A.yield.begin", i)
                _pp_take(half)
                _seg("consumer.A.yield.end", i)
                warpgroup.fence()
                _gemm_rs(mma_rs, accO, rS, frgQ, qs, False)
                _gemm_rs(mma_rs, accW, rS, frgK, ks, False)
                warpgroup.commit_group()
                _seg("consumer.A.chain1.commit", i)
                _pp_give(half)
                warpgroup.wait_group(0)
                _seg("consumer.B.chain2.drain", i)
                if lane == 0:
                    cute.arch.mbarrier_arrive(_bar(b_qe, qs))

                # ---- segment C: A'^T = beta (x) (v^T - (kS)^T (x) e) --------
                # No chain here: the whole segment is elementwise and rides the
                # sibling's flight. The f32 (kS)^T dies where it is packed.
                # D2.3 folds diag(beta) in here -- the column factor is
                # e_i * beta_i on the (kS)^T term and beta_i on v^T, which is
                # beta_i (v_i - e_i (kS)_i), i.e. exactly the row scaling the
                # solver's exit used to apply to T's columns. The packing is
                # fp16, not bf16: chain 3's B is now the fp16 inverse face
                # itself, and the two operands have to agree.
                _seg("consumer.C.v_full.begin", i)
                _bar_wait(_bar(b_vf, vs), vp)
                _seg("consumer.C.v_full.end", i)
                _seg("consumer.C.b_full.begin", i)
                _bar_wait(_bar(b_bf, bs), bp)
                _seg("consumer.C.b_full.end", i)
                cute.copy(tiled_ldsm,
                          thr_l.partition_S(sVt[(None, None, half, vs)]),
                          thr_l.retile(rV))
                for e in cutlass.range_constexpr(cute.size(accW)):
                    _, tok = cC[e]
                    bt = sB[(tok, bs)]
                    accW[e] = (cutlass.Float32(rV[e]) * bt
                               - accW[e] * (sG[(1, tok, gs)] * bt))
                _pack_f16(accW, rPf)
                cute.arch.mbarrier_arrive(_bar(b_be, bs))
                _seg("consumer.C.t_full.begin", i)
                _bar_wait(_bar(b_tf, ts), tp)
                _seg("consumer.C.t_full.end", i)

                # ---- segment D: chain 3, w^T = A'^T . T^T (fp16, D2.3) ------
                _seg("consumer.D.yield.begin", i)
                _pp_take(half)
                _seg("consumer.D.yield.end", i)
                _issue_rs(mma_rf, accW, rPf, frgT, ts, False)
                _seg("consumer.D.chain3.commit", i)
                _pp_give(half)
                # D0.11's one elementwise that runs in its OWN chain's flight:
                # the o_inter column scaling. It touches accO -- chain 1's
                # retired accumulator -- and never chain 3's accW, so the
                # pipelined section stays legal.
                for e in cutlass.range_constexpr(cute.size(accO)):
                    _, tok = cC[e]
                    accO[e] = accO[e] * sG[(2, tok, gs)]
                warpgroup.wait_group(0)
                _seg("consumer.D.chain3.drain", i)

                # ---- segment E: chain 4, o^T += w^T . A_qk^T -----------------
                if lane == 0:
                    cute.arch.mbarrier_arrive(_bar(b_ve, vs))
                cute.arch.mbarrier_arrive(_bar(b_te, ts))
                _pack_bf16(accW, rP)
                _seg("consumer.E.a_full.begin", i)
                _bar_wait(_bar(b_af, asl), ap)
                _seg("consumer.E.a_full.end", i)
                _seg("consumer.E.yield.begin", i)
                _pp_take(half)
                _seg("consumer.E.yield.end", i)
                _issue_rs(mma_rs, accO, rP, frgA, asl, True)
                _seg("consumer.E.chain4.commit", i)
                _pp_give(half)
                warpgroup.wait_group(0)
                _seg("consumer.E.chain4.drain", i)
                cute.arch.mbarrier_arrive(_bar(b_ae, asl))

                # ---- segment F: publish o (D0.12 step 1) ---------------------
                # o^T lives as M = v, N = tok; the output wants tok-major rows
                # of v. The only true transpose path is the smem exchange face
                # (ledger row X7): pack to bf16, `STSM.MT` the 64x64 block into
                # this warpgroup's own 64 v-columns of the O slot, proxy-fence,
                # then arrive O-full (256 = the whole consumer pair, so the slot
                # is only handed over once both halves are in it). The wait sits
                # outside the ping-pong baton region, so no yield is ever held
                # across it.
                _seg("consumer.F.o_empty.begin", i)
                _bar_wait(_bar(b_oe, os), 1 - op)
                _seg("consumer.F.o_empty.end", i)
                _pack_bf16(accO, rO)
                cute.copy(tiled_stsm, thr_s.retile(rO),
                          thr_s.partition_D(sOt[(None, None, half, os)]))
                cute.arch.fence_view_async_shared()
                _seg("consumer.F.o_full.arrive", i)
                cute.arch.mbarrier_arrive(_bar(b_of, os))

                # ---- segment G: decay + chain 5, S^T += (w^T (x) g) . k ------
                glast = sG[(0, CHUNK - 1, gs)]
                gmul = _exp2(glast)
                for e in cutlass.range_constexpr(cute.size(accS)):
                    accS[e] = accS[e] * gmul
                for e in cutlass.range_constexpr(cute.size(accW)):
                    _, tok = cC[e]
                    accW[e] = accW[e] * _exp2(glast - sG[(0, tok, gs)])
                _pack_bf16(accW, rPg)
                _seg("consumer.G.yield.begin", i)
                _pp_take(half)
                _seg("consumer.G.yield.end", i)
                _issue_rs(mma_mn, accS, rPg, frgKt, ks, True)
                _seg("consumer.G.chain5.commit", i)
                _pp_give(half)
                warpgroup.wait_group(0)
                _seg("consumer.G.chain5.drain", i)
                if lane == 0:
                    cute.arch.mbarrier_arrive(_bar(b_ke, ks))
                cute.arch.mbarrier_arrive(_bar(b_ge, gs))

            # ---- final_state, K-major [h, k, v] (D0.12 step 2) --------------
            # S^T is held as (v, k) but the contract is [h, k, v] with v
            # contiguous, so a straight write-out is a 4-byte scatter across k
            # rows. Decision (b): transpose once through the O staging slots,
            # which are idle now that the last chunk has been stored. 8,192 f32
            # of O ring / 2 warpgroups = 4,096 f32 each = one 64k x 64v batch,
            # two batches to cover S^T's 128 k. The write side is scattered STS
            # (once per kernel), the read side is fully coalesced: consecutive
            # `lt` reads consecutive f32 and stores to consecutive v.
            _seg("consumer.state.o_empty.begin")
            for s in cutlass.range_constexpr(O_STAGE):
                j = n_chunks + s
                _bar_wait(_bar(b_oe, j % O_STAGE), 1 - (j // O_STAGE) % 2)
            _seg("consumer.state.o_empty.end")
            for b in cutlass.range_constexpr(DHEAD // CHUNK):
                _state_bar(half)
                for e in cutlass.range_constexpr(cute.size(accS)):
                    vv, kk = cS[e]
                    if kk // CHUNK == b:
                        sSt[(kk % CHUNK, vv, half)] = accS[e]
                _state_bar(half)
                for u in cutlass.range_constexpr(CHUNK * CHUNK // 128):
                    idx = u * 128 + lt
                    mSt[(hv, b * CHUNK + idx // CHUNK, cbase + idx % CHUNK)] = \
                        sSt[(idx // CHUNK, idx % CHUNK, half)]
            _seg("consumer.state.store")


# ------------------------------------------------------------------------------
# host entry
# ------------------------------------------------------------------------------
@cute.jit
def _gdn_host(
    tq, tk, tv, ta, tb, to, ts,
    scale,
    N: cutlass.Constexpr,
    H_QK: cutlass.Constexpr,
    H_V: cutlass.Constexpr,
    stream,
):
    plan = _plan()
    q_l, k_l, v_l = plan["q_l"], plan["k_l"], plan["v_l"]
    o_l, a_l, t_l = plan["o_l"], plan["a_l"], plan["t_l"]
    kt_l, vt_l, ot_l = plan["kt_l"], plan["vt_l"], plan["ot_l"]
    tb8_l, tt8_l = plan["tb8_l"], plan["tt8_l"]
    t16_l, tt16_l = plan["t16_l"], plan["tt16_l"]
    t16x32_l, tt16x32_l = plan["t16x32_l"], plan["tt16x32_l"]
    tt32x32_l = plan["tt32x32_l"]
    Storage = plan["Storage"]

    n_chunks = N // CHUNK
    rep = H_V // H_QK

    bf16, f32 = cutlass.BFloat16, cutlass.Float32
    MJK = warpgroup.OperandMajorMode.K
    MJMN = warpgroup.OperandMajorMode.MN

    # (N, d, head) views: d is the contiguous mode, head is the batch mode, so a
    # 64x128 tile of one head is a box in the 3-D TMA coordinate space.
    mQ = cute.make_tensor(tq.iterator, cute.make_layout(
        (N, DHEAD, H_QK), stride=(H_QK * DHEAD, 1, DHEAD)))
    mK = cute.make_tensor(tk.iterator, cute.make_layout(
        (N, DHEAD, H_QK), stride=(H_QK * DHEAD, 1, DHEAD)))
    mV = cute.make_tensor(tv.iterator, cute.make_layout(
        (N, DHEAD, H_V), stride=(H_V * DHEAD, 1, DHEAD)))
    mAl = cute.make_tensor(ta.iterator, cute.make_layout(
        (N, H_V), stride=(H_V, 1)))
    mBe = cute.make_tensor(tb.iterator, cute.make_layout(
        (N, H_V), stride=(H_V, 1)))
    mO = cute.make_tensor(to.iterator, cute.make_layout(
        (N, DHEAD, H_V), stride=(H_V * DHEAD, 1, DHEAD)))
    mSt = cute.make_tensor(ts.iterator, cute.make_layout(
        (H_V, DHEAD, DHEAD), stride=(DHEAD * DHEAD, DHEAD, 1)))

    tma_atom_q, tma_q = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mQ,
        cute.slice_(q_l, (None, None, 0)), (CHUNK, DHEAD), num_multicast=1)
    tma_atom_k, tma_k = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mK,
        cute.slice_(k_l, (None, None, 0)), (CHUNK, DHEAD), num_multicast=1)
    tma_atom_v, tma_v = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mV,
        cute.slice_(v_l, (None, None, 0)), (CHUNK, DHEAD), num_multicast=1)
    # D0.12: the S2G side. Same 3-D (tok, d, head) coordinate space as the load
    # atoms, so the 64x128 O staging slot is one box; S2G takes no multicast.
    tma_atom_o, tma_o = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), mO,
        cute.slice_(o_l, (None, None, 0)), (CHUNK, DHEAD))

    # D0.2's chain shapes, one TiledMma per operand-source class.
    mma_ss = sm90_utils.make_trivial_tiled_mma(
        bf16, bf16, MJK, MJK, f32,
        atom_layout_mnk=(1, 1, 1), tiler_mn=(CHUNK, CHUNK))
    mma_rs = sm90_utils.make_trivial_tiled_mma(
        bf16, bf16, MJK, MJK, f32,
        atom_layout_mnk=(1, 1, 1), tiler_mn=(CHUNK, CHUNK),
        a_source=warpgroup.OperandSource.RMEM)
    # D2.3: chain 3 alone is fp16 in both operand roles -- same shape, same
    # operand sources, same major modes as `mma_rs`, only the A/B element type
    # changes (`HGMMA.64x64x16.F32.F16`). It exists because D2.2 makes the fp16
    # inverse face the T slot's deliverable, so narrowing it to bf16 for the
    # consumer would mean putting the read-modify-write back.
    mma_rf = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16, MJK, MJK, f32,
        atom_layout_mnk=(1, 1, 1), tiler_mn=(CHUNK, CHUNK),
        a_source=warpgroup.OperandSource.RMEM)
    mma_mn = sm90_utils.make_trivial_tiled_mma(
        bf16, bf16, MJK, MJMN, f32,
        atom_layout_mnk=(1, 1, 1), tiler_mn=(CHUNK, DHEAD),
        a_source=warpgroup.OperandSource.RMEM)
    # v^T out of the v ring: the (d, tok) face has d contiguous, i.e. d is the C
    # fragment's M mode -> the TRANSPOSED ldmatrix member (ledger row X2's read
    # side, probe X5).
    tiled_ldsm = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 4), bf16), mma_rs)
    # o^T into the O staging slot: the mirror image of the v^T read. The write
    # face (v, tok) has the M mode (v) contiguous -- M-major C -- so it is the
    # TRANSPOSED stmatrix member over the same C-fragment thread/value map
    # (ledger row X7; `STSM.16.MT88.4` x4 per warp).
    tiled_stsm = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(True, 4), bf16), mma_rs)

    # --- D0.9's triangular inverse -------------------------------------------
    # The two exchange-face copies both hang off `mma_ss`, the solver's own
    # accumulator map, so the values leave the accumulator in C-fragment order
    # and land in the face without a detour. The face is c-contiguous, i.e.
    # ROW_MAJOR C, which is what selects the NON-transposed `stmatrix` member;
    # the f16 and bf16 spellings differ only in the element type, and therefore
    # write the same bytes at the same addresses. (The read-back copy that used
    # to pair with them left with D2.2's exit segment: nothing reads the face
    # back into registers any more.)
    f16 = cutlass.Float16
    ENC = LayoutEnum.ROW_MAJOR
    r2s_f16 = cute.make_tiled_copy_S(
        sm90_utils.get_smem_store_op(ENC, elem_ty_d=f16, elem_ty_acc=f32),
        cute.make_tiled_copy_C_atom(
            cute.make_copy_atom(
                warp.StMatrix8x8x16bOp(ENC.is_m_major_c(), 4), f16), mma_ss))
    r2s_bf16 = cute.make_tiled_copy_S(
        sm90_utils.get_smem_store_op(ENC, elem_ty_d=bf16, elem_ty_acc=f32),
        cute.make_tiled_copy_C_atom(
            cute.make_copy_atom(
                warp.StMatrix8x8x16bOp(ENC.is_m_major_c(), 4), bf16), mma_ss))
    # The doubling levels are WARP-level mma, not warpgroup: `m16n8k8` for the
    # 8 -> 16 step (K = 8, so accumulator -> A chains at ratio 1) and
    # `m16n8k16` for 16 -> 32 and 32 -> 64 (ratio 2). fp16 in, f32 accumulate.
    mma8 = cute.make_tiled_mma(warp.MmaF16BF16Op(f16, f32, (16, 8, 8)),
                               (1, 1, 1))
    tc_a8 = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 1), f16), mma8)
    tc_b8 = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 1), f16), mma8)
    tc_c8 = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 1), f16), mma8)
    mma16 = cute.make_tiled_mma(warp.MmaF16BF16Op(f16, f32, (16, 8, 16)),
                                (1, 1, 1))
    tc_a16 = cute.make_tiled_copy_A(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), f16), mma16)
    tc_b16 = cute.make_tiled_copy_B(
        cute.make_copy_atom(warp.LdMatrix8x8x16bOp(True, 2), f16), mma16)
    tc_cst = cute.make_tiled_copy_C(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(False, 2), f16), mma16)

    _gdn_kernel(
        tma_atom_q, tma_q,
        tma_atom_k, tma_k,
        tma_atom_v, tma_v,
        tma_atom_o, tma_o,
        mAl, mBe, mSt,
        cutlass.Float32(scale),
        mma_ss, mma_rs, mma_rf, mma_mn, tiled_ldsm, tiled_stsm,
        r2s_f16, r2s_bf16,
        mma8, tc_a8, tc_b8, tc_c8,
        mma16, tc_a16, tc_b16, tc_cst,
        q_l, k_l, v_l, o_l, a_l, t_l, kt_l, vt_l, ot_l,
        tb8_l, tt8_l, t16_l, tt16_l, t16x32_l, tt16x32_l, tt32x32_l,
        Storage,
        n_chunks, rep,
    ).launch(
        grid=[H_V, 1, 1],
        block=[NUM_THREADS, 1, 1],
        min_blocks_per_mp=1,
        stream=stream,
    )


def _cute_args(q, k, v, alpha, beta, o, state):
    """The exact (dynamic) argument tuple `_gdn_host` is compiled/called with.

    Probes must build their arguments through this so the certified spelling and
    the shipped spelling can never drift apart.
    """
    return (
        from_dlpack(q, assumed_align=16),
        from_dlpack(k, assumed_align=16),
        from_dlpack(v, assumed_align=16),
        from_dlpack(alpha, assumed_align=16),
        from_dlpack(beta, assumed_align=16),
        from_dlpack(o, assumed_align=16),
        from_dlpack(state, assumed_align=16),
    )


def _current_stream():
    try:
        import cuda.bindings.driver as _drv
    except ImportError:  # older cuda-python
        from cuda import cuda as _drv
    return _drv.CUstream(torch.cuda.current_stream().cuda_stream)


def gdn_prefill_fwd(
    q: torch.Tensor,        # [N, H_QK, D] bf16   N=8192, H_QK=16, D=128
    k: torch.Tensor,        # [N, H_QK, D] bf16   L2-normalized along D
    v: torch.Tensor,        # [N, H_V,  D] bf16   H_V=32 (GVA: 2 v-heads per qk-head)
    alpha: torch.Tensor,    # [N, H_V] fp32       forget gate, values in (0, 1)
    beta: torch.Tensor,     # [N, H_V] fp32       write strength, values in (0, 1)
    scale: float,           # D ** -0.5, applied to the readout
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (o [N, H_V, D] bf16, final_state [H_V, D, D] fp32 K-major)."""
    N, H_QK, Dh = q.shape
    H_V = v.shape[1]
    assert Dh == DHEAD and N % CHUNK == 0

    o = torch.empty(N, H_V, Dh, device=q.device, dtype=torch.bfloat16)
    state = torch.empty(H_V, Dh, Dh, device=q.device, dtype=torch.float32)

    key = (N, H_QK, H_V, Dh)
    fn = _exec_cache.get(key)
    if fn is None:
        args = _cute_args(q, k, v, alpha, beta, o, state)
        fn = cute.compile(_gdn_host, *args, scale, N, H_QK, H_V,
                          _current_stream())
        _exec_cache[key] = fn

    fn(*_cute_args(q, k, v, alpha, beta, o, state), scale, _current_stream())
    return o, state
