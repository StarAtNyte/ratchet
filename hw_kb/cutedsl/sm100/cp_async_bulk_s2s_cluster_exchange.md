# cp_async_bulk_s2s_cluster_exchange (1D bulk async copy from smem → the smem of another CTA in the cluster: accounted on the peer's mbarrier, source side drained by a remote arrive)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; footprint readings taken from an sm100 SASS census (`UBLKCP.S.S`).
>
> What this entry records is **one knob + its SASS footprint + the conditions under which it bears load**, not a percentage:
>
> | | |
> |---|---|
> | **Knob** | one `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dp], [src], nb, [mp]`: `src` is this CTA's smem address, `dp` and `mp` are both smem addresses of **the peer CTA**, converted from local offsets by `mapa.shared::cluster.u32 <out>, <local offset>, <peer rank>` (every CTA has the same smem layout, so the peer's receive face and the peer's barrier can both be addressed by your own offsets). Completion = the tx bytes of **the peer's** barrier all arrived; the sender does not arrive and waits on nothing of its own. Between the source-side writes and the issue: `fence.proxy.async.shared::cta` + one CTA barrier. **Draining the source face is not `commit_group / wait_group.read`** (measured not to cover it, see gotcha 3); it is the receiver, after its own data barrier flips, firing a remote `mbarrier.arrive.shared::cluster.b64` at the sender's ack barrier |
> | **Footprint** | dossier census axis `cache_ops` (verbatim dotted opcode): `["UBLKCP.S.G"]` → `["UBLKCP.S.G", "UBLKCP.S.S"]`. g2s and s2s are two suffixes of the same family; when both coexist in one kernel the census reads one each |
> | **Form B (sender arms)** | the same copy instruction, but **with the accounting direction reversed**: the receive barrier is `init(count = number of senders)`, and the receiver **does not arm and does not know the byte count in advance**; each sender fires one remote `mbarrier.arrive.expect_tx.shared::cluster.b64 _, [mp], tx` at each peer (`mp` via `mapa`; arrive and tx in the same instruction), `tx` = the bytes that sender actually delivers to this peer (it may differ between senders: a whole payload run + a small header, or only the small header). Prologue order = `init → fence → receive face ready → one pair of cluster rendezvous → only then gmem reads / issue`; **no** DRAM round trip before the rendezvous. Certification in the "Form B" section and gotchas 8–12 |
| **Load-bearing conditions** | under the structure "cluster of 16 CTAs, 256 threads per CTA, grid 128, 4 rounds of pairwise exchange per CTA, each round one s2s of 16/8/4/2 KB + one of 128 B issued by a single thread, the whole CTA waiting on the barrier of the round the peer sent, one fence + CTA barrier + local smem merge between rounds", replacing "per-thread `st.shared::cluster.v2.b32` writing directly to the peer (4096 per CTA) + a second cluster rendezvous" with it moved the gate window **0.01522 → 0.0168 ms (+10.38%, slower)**. The same delivery also changed the smem access shape of the local merge, so the ms is not the price of the copy instruction alone, see the "Load-bearing conditions" section |
>
> **This entry is certified at two levels**: the **byte correctness** of the copy + peer accounting + source-side drain is certified by a GPU probe
> (the same source face reused over two rounds: no ack and the `wait_group.read` guard are both dirty, the ack guard 0.0);
> the footprint on the `cache_ops` axis and the ms readings are certified by the dossier census + gate window of one delivered KEEP (that KEEP
> uses this mechanism for the whole cluster's merge exchange, numerically PASS, with **no** second cluster rendezvous before exit).

## Purpose

Change "move a contiguous run of one CTA's smem bytes as a whole into the smem of another CTA in the cluster" from **per-thread remote
st.shared::cluster** (one instruction per 8 B, and after issuing, one more cluster-level rendezvous is needed to know the peer has finished writing) into
**one byte-counted bulk async copy**: both the address and the barrier point at the peer via `mapa`, the arrival of the bytes is accounted by **the peer's**
mbarrier `complete_tx`, the peer waits only on one barrier of its own, and the sender leaves as soon as it has issued. It is
the branch of the `cp.async.bulk` family with both source and destination in shared ([`cp_async_bulk_1d_mbarrier_gather`](cp_async_bulk_1d_mbarrier_gather.md)
is the global → shared branch; the PTX spelling, tx accounting and single-thread issue guard share their origin with it).

## When needed (trigger)

- CTAs within a cluster must exchange **whole runs** (KB-scale) of smem data, and you do not want a second `barrier.cluster`
  rendezvous to confirm "the peer has finished writing"; or the number of per-thread `st.shared::cluster` (4096 per CTA before this entry's change)
  shows up in the dossier as the main item of `merge.scalar_sts_in_loop`.
- The dossier census's `cache_ops` axis has `UBLKCP.S.G` but not `UBLKCP.S.S`, and the latter is what you want.
- **N senders each send an unequal number of bytes to the same receive barrier** (some send a whole run, some only a small header /
  flag), and the receiver does not want to (and should not) do a DRAM round trip before the rendezvous to "census" who will send how much -- this is the trigger for form B:
  `arrive.expect_tx` is fired remotely by **the sender** at the peer's barrier, and the receiver only does `init(count = N)`.
  Symptom: in the prologue, before `cluster_wait`, an extra warp-wide LDG + `redux` + local `expect_tx`,
  plus one more whole-CTA `barrier()` to broadcast that number -- in the landing measurement what moved was `stall_barrier` (2.81 → 1.57),
  while `stall_long_sb` barely moved (5.39 → 5.33), see "Production price of form B"; do not take long_sb as its fingerprint.
- You are already writing this PTX and are stuck on: s2s issue gives `an illegal instruction was encountered` /
  the peer barrier never fills / guarding the source with `commit_group + wait_group.read` still sends dirty data to the peer /
  not knowing when a CTA can safely exit.

## Minimal runnable example

Skeleton extracted from the certification probe: the `CSZ` CTAs of the cluster are paired (peer = `rank ^ (CSZ/2)`),
and each CTA uses the same `NB`-byte staging face **for two rounds** (round 1 sends X; after it is rewritten as Y, round 2 sends again), with the rewrite between the two rounds
allowed only after the peer's ack. The copy, `mapa`, remote arrive and fence go through inline PTX
(form (2) of [`ptx_escape`](ptx_escape.md)); barrier init / arm / wait use the DSL's own
`cute.arch.mbarrier_*`; the cluster barrier and rank use `cute.arch.cluster_arrive / cluster_wait /
block_idx_in_cluster` (fall back to `barrier.cluster.arrive / wait` and
`mov.u32 %cluster_ctarank` when the DSL version does not expose them).

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as _MT
from cutlass.cutlass_dsl import dsl_user_op

ATT = llvm.AsmDialect.AD_ATT

# ---- one s2s bulk copy: dst / mbar are passed as local offsets, both converted to the peer via mapa.
@dsl_user_op
def bulk_s2s(dst, src, nb, mbar, peer, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
         cutlass.Int32(src).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nb).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 dp, mp; "
        "mapa.shared::cluster.u32 dp, $0, $4; "
        "mapa.shared::cluster.u32 mp, $3, $4; "
        "cp.async.bulk.shared::cluster.shared::cta"
        ".mbarrier::complete_tx::bytes [dp], [$1], $2, [mp]; }",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# ---- remote arrive: after its own data barrier flips, the receiver fires once at the peer's ack barrier.
@dsl_user_op
def mbar_arrive_remote(mbar, peer, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 mp; mapa.shared::cluster.u32 mp, $0, $1; "
        "mbarrier.arrive.shared::cluster.b64 _, [mp]; }",
        "r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# ---- non-blocking query: 1 = the phase has flipped. A bounded spin makes "the ack never came" a FAIL rather than a hang.
@dsl_user_op
def mbar_try_wait(mbar, phase, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(llvm.inline_asm(
        _MT.i32(),
        [cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int32(phase).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; "
        "selp.b32 $0, 1, 0, p; }",
        "=r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))

@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):           # step of the bounded spin (form (1))
    llvm.inline_asm(None, [], "nanosleep.u32 %d;" % int(ns), "",
                    has_side_effects=True, is_align_stack=False, asm_dialect=ATT,
                    loc=loc, ip=ip)

@dsl_user_op
def fence_async_shared(*, loc=None, ip=None):      # makes ordinary st.shared writes visible to the copy engine
    llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "~{memory}",
                    has_side_effects=True, is_align_stack=False, asm_dialect=ATT,
                    loc=loc, ip=ip)

CSZ, THREADS = 16, 256
NF, MLF = 4096, 32                 # 16 KB data block + 128 B small block (two copies, the same barrier)
NB = NF * 4
ACK_SPIN = 64

@cute.kernel
def exchange_twice(mS: cute.Tensor, mO1: cute.Tensor, mO2: cute.Tensor, mA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass_utils.SmemAllocator()
    lay = cute.make_layout((NF + MLF,), stride=(1,))
    sSrc = smem.allocate_tensor(element_type=cutlass.Float32, layout=lay, byte_alignment=16)
    sR1 = smem.allocate_tensor(element_type=cutlass.Float32, layout=lay, byte_alignment=16)
    sR2 = smem.allocate_tensor(element_type=cutlass.Float32, layout=lay, byte_alignment=16)
    mb = smem.allocate_tensor(element_type=cutlass.Int64,
                              layout=cute.make_layout((3,), stride=(1,)), byte_alignment=8)

    if tidx == 0:
        cute.arch.mbarrier_init(mb.iterator + 0, 1)      # round-1 data (count 1 = that local arm)
        cute.arch.mbarrier_init(mb.iterator + 1, 1)      # round-2 data
        cute.arch.mbarrier_init(mb.iterator + 2, 1)      # ack (count = how many remote arrives to receive)
    cute.arch.mbarrier_init_fence()
    if tidx == 0:                                        # arm first, then enter the cluster rendezvous (gotcha 2)
        cute.arch.mbarrier_arrive_and_expect_tx(mb.iterator + 0, NB + 128)
        cute.arch.mbarrier_arrive_and_expect_tx(mb.iterator + 1, NB + 128)

    rank = cute.arch.block_idx_in_cluster()
    for i in cutlass.range_constexpr((NF + MLF) // THREADS + 1):
        j = tidx + i * THREADS
        if j < (NF + MLF):
            sSrc[j] = mS[(bidx, j)]
    cute.arch.barrier()
    cute.arch.cluster_arrive()                           # the only pair of cluster barriers in the whole kernel:
    cute.arch.cluster_wait()                             # after this the peer's barriers are initialised + armed

    peer = rank ^ (CSZ // 2)
    sbase = sSrc.iterator.toint()
    r1, r2 = sR1.iterator.toint(), sR2.iterator.toint()
    ackp = (mb.iterator + 2).toint()
    fence_async_shared()                                 # the source face was just written by st.shared (gotcha 4)
    cute.arch.barrier()
    if tidx == 0:                                        # ---- round 1
        bulk_s2s(r1, sbase, NB, (mb.iterator + 0).toint(), peer)
        bulk_s2s(r1 + NF * 4, sbase + NF * 4, 128, (mb.iterator + 0).toint(), peer)

    cute.arch.mbarrier_wait(mb.iterator + 0, 0)          # round 1 from the peer has all arrived
    if tidx == 0:
        mbar_arrive_remote(ackp, peer)                   # ⇒ the peer's source face has been fully read; tell it
    nack = cutlass.Int32(0)
    for _ in cutlass.range_constexpr(ACK_SPIN):          # wait for the peer to tell me the same thing
        nack = nack + mbar_try_wait(ackp, cutlass.Int32(0))
        nanosleep(50000)
    cute.arch.barrier()

    for i in cutlass.range_constexpr((NF + MLF) // THREADS + 1):   # only now may the source face be rewritten
        j = tidx + i * THREADS
        if j < (NF + MLF):
            sSrc[j] = mS[(bidx, j)] + cutlass.Float32(10000.0)
    fence_async_shared()
    cute.arch.barrier()
    if tidx == 0:                                        # ---- round 2, the same source face
        bulk_s2s(r2, sbase, NB, (mb.iterator + 1).toint(), peer)
        bulk_s2s(r2 + NF * 4, sbase + NF * 4, 128, (mb.iterator + 1).toint(), peer)

    cute.arch.mbarrier_wait(mb.iterator + 1, 0)
    for i in cutlass.range_constexpr((NF + MLF) // THREADS + 1):
        j = tidx + i * THREADS
        if j < (NF + MLF):
            mO1[(bidx, j)] = sR1[j]
            mO2[(bidx, j)] = sR2[j]
    if tidx == 0:
        mA[bidx] = nack                                  # 0 = some CTA never saw the ack (judged FAIL)
    # Exit contract (gotcha 5): both rounds' data barriers flipped ⇒ no peer will still write my smem;
    # the ack arrived ⇒ no peer is still reading my smem. No second cluster rendezvous is needed.

# launch: grid=[128,1,1], block=[THREADS,1,1], cluster=[CSZ,1,1]
```

Numeric contract (the probe's criterion): `mO1[b] == mS[peer(b)]`, `mO2[b] == mS[peer(b)] + 10000`
byte-for-byte equal (`max_diff = 0`), and every CTA has `nack > 0`. The same kernel with the whole ack section removed
serves as the control: the round-1 data reaching the peer **must** be dirty (reading 12732), otherwise the reuse window is too narrow and the ack verdict has no power.

### Form B: the sender arms -- remote `arrive.expect_tx`, receive barrier count = number of senders, zero foreknowledge on the receiver

Skeleton extracted from the certification probe: the `CSZ` CTAs of the cluster go **all-to-all** -- each CTA sends to **every** peer
(including itself) a payload run of `SLOT_B` bytes + an 8 B small header (two different async writes, accounted on **the same** barrier of the peer),
or only the 8 B small header (that CTA has no payload). The peer only knows "there will be `CSZ` arrives",
and each arrive carries its own byte count. The whole kernel has only the prologue's pair of cluster barriers, and there are **no gmem reads at all** before the rendezvous.

```python
# reuse form A's bulk_s2s / mbar_try_wait / nanosleep / fence_async_shared; three additions:

@dsl_user_op
def st_async_v2(dst, a, b, mbar, peer, *, loc=None, ip=None):
    # 8 B small header written directly into the peer's smem, complete_tx 8 accounted on the peer's barrier. PTX requires dst and mbar
    # to be in the same (destination) CTA -- so both addresses go through mapa.
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
         cutlass.Int32(a).ir_value(loc=loc, ip=ip),
         cutlass.Int32(b).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 dp, mp; "
        "mapa.shared::cluster.u32 dp, $0, $4; "
        "mapa.shared::cluster.u32 mp, $3, $4; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [dp], {$1, $2}, [mp]; }",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

@dsl_user_op
def mbar_arrive_expect_tx_remote(mbar, tx, peer, *, loc=None, ip=None):
    # one instruction = expect-tx(tx) + arrive-on(1), fired at the peer's barrier.
    # The remote form cannot return a state: the destination operand must be the sink `_` (PTX ISA text).
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int32(tx).ir_value(loc=loc, ip=ip),
         cutlass.Int32(peer).ir_value(loc=loc, ip=ip)],
        "{ .reg .u32 mp; mapa.shared::cluster.u32 mp, $0, $2; "
        "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [mp], $1; }",
        "r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# DSL-native equivalents (the same mechanism, another cell of the same probe):
#   cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tx, peer_cta_rank_in_cluster=peer)
#   cute.arch.store_async_dsmem(smem_ptr, value, mbar_ptr, peer)      # 4 B each, two of them = 8 B

CSZ, THREADS, NWARP = 16, 256, 8
SLOT_W = 512                       # one payload slot = 512 × i32 = 2048 B
SLOT_B, TAG_W = SLOT_W * 4, 2      # small header 8 B
SPIN = 40                          # bounded wait: 40 × 50 µs

@cute.kernel
def all_to_all(mS: cute.Tensor, mAct: cute.Tensor, mO: cute.Tensor, mT: cute.Tensor, mH: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp, lane = tidx // 32, tidx % 32
    rank = cute.arch.block_idx_in_cluster()
    smem = cutlass_utils.SmemAllocator()
    lay = cute.make_layout((CSZ * SLOT_W,), stride=(1,))
    sSrc = smem.allocate_tensor(element_type=cutlass.Int32, layout=lay, byte_alignment=16)   # the 16 runs I send
    sRecv = smem.allocate_tensor(element_type=cutlass.Int32, layout=lay, byte_alignment=16)  # one slot for each of the 16 senders
    sTag = smem.allocate_tensor(element_type=cutlass.Int32,
                                layout=cute.make_layout((CSZ * TAG_W,), stride=(1,)), byte_alignment=16)
    mb = smem.allocate_tensor(element_type=cutlass.Int64,
                              layout=cute.make_layout((1,), stride=(1,)), byte_alignment=8)

    # ---- prologue: init(count = number of senders) → fence → receive face ready → the only pair of cluster barriers. That is all.
    if tidx == 0:
        cute.arch.mbarrier_init(mb.iterator, CSZ)      # the receiver does not arm and does no expect_tx at all
    cute.arch.mbarrier_init_fence()
    for i in cutlass.range_constexpr(CSZ * SLOT_W // THREADS):
        sRecv[tidx + i * THREADS] = cutlass.Int32(-1)  # receive face ready before the rendezvous (so unwritten slots can be told apart)
    if tidx < CSZ * TAG_W:
        sTag[tidx] = cutlass.Int32(-1)
    cute.arch.barrier()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    # from this moment on, peers may drop bytes into sRecv / sTag / mb at any time

    # ---- all of the sender's own work comes after the rendezvous: gmem reads, staging, fence, issue
    act = mAct[bidx]                                   # whether this CTA has a payload (0/1)
    for i in cutlass.range_constexpr(CSZ * SLOT_W // THREADS):
        j = tidx + i * THREADS
        sSrc[j] = mS[(bidx, j)]
    fence_async_shared()
    cute.arch.barrier()
    mba, recv_a, src_a, tag_a = (mb.iterator.toint(), sRecv.iterator.toint(),
                                 sSrc.iterator.toint(), sTag.iterator.toint())
    tx = cutlass.Int32(TAG_W * 4) + act * cutlass.Int32(SLOT_B)   # 2056 or 8: the bytes I actually send to this peer
    if lane == 0:                                      # warp w serves peers w and w+8
        for hh in cutlass.range_constexpr(CSZ // NWARP):
            h = warp + hh * NWARP
            mbar_arrive_expect_tx_remote(mba, tx, h)   # arrive.expect_tx first, then issue (the order bears no load, gotcha 10)
            st_async_v2(tag_a + rank * TAG_W * 4, rank * 100 + h, act, mba, h)
            if act != 0:
                bulk_s2s(recv_a + rank * SLOT_B, src_a + h * SLOT_B, SLOT_B, mba, h)

    # ---- receiver: waits only on its own barrier (bounded); flipped ⇒ the bytes of all 16 senders have landed
    hits = cutlass.Int32(0)
    for _ in cutlass.range_constexpr(SPIN):
        hits = hits + mbar_try_wait(mba, cutlass.Int32(0))
        nanosleep(50000)
    cute.arch.barrier()
    for i in cutlass.range_constexpr(CSZ * SLOT_W // THREADS):
        j = tidx + i * THREADS
        mO[(bidx, j)] = sRecv[j]
    if tidx < CSZ * TAG_W:
        mT[(bidx, tidx)] = sTag[tidx]
    if tidx == 0:
        mH[bidx] = hits                                # 0 = never flipped (judged FAIL)

# launch: grid=[128,1,1], block=[THREADS,1,1], cluster=[CSZ,1,1]
```

Numeric contract (the probe's criterion): peer h's slot s = sender s's h-th run (when the sender has a payload), otherwise the slot stays −1;
small header = `(s*100 + h, act[s])`; byte-for-byte equal (`max_diff = 0`) and every CTA `hits > 0`. Five "who has a
payload" patterns (all / none / even ranks / one per cluster / pseudo-random), one launch each, the same compiled entry.
Negative control of the same kernel: the receiver **also** does one local `expect_tx(total bytes)` before the rendezvous -- not a single CTA's barrier
flips within 2 ms (gotcha 9).

### Production price of form B (certification boundary [UNVERIFIED-BOUNDARY])

The five contracts above certify the **accounting mechanism** (byte-exact, every CTA flips, the sender's internal order bears no load, the DSL spelling is equivalent,
a receiver that arms again never flips). "How many ms saving that DRAM round trip in the prologue is worth" is not judged by the probe; it comes from two readings of the same production tree
(cluster 16 all-to-all, 256 threads per CTA, grid 128), with these boundaries:

- **Single-variable price (on the gate tree base / +one / all / all−one, three rounds each, consistent across the three)**:
  base 0.01206 ms → adding only form B 0.01146 ms (**+5.44%**); removing it alone from the full set −5.68%; in the same package as "batching the index
  fetch into one round trip after the rendezvous" +7.85%. Only the ms is cited: the lever-class verdict of that job is held with reservations.
- **Landing (KEEP, the eval job number was not recorded in the status record)**: device 0.01166 → 0.01106 ms (−5.1%). The same delivery also
  deleted the pre-rendezvous census (warp-wide LDG + `redux` + local `expect_tx` + whole-CTA `barrier()`), moved the index prefetch to
  after the rendezvous and merged it with the fast-path word into one round trip, and moved the fast-path `exit` later -- the ms is not form B's own price. NCU:

  | Reading | Before landing | After landing |
  |---|---|---|
  | `stall_barrier` | 2.81 | 1.57 |
  | `stall_long_sb` | 5.39 | 5.33 |
  | `stall_mio` | 0.12 | 0.11 |
  | `nbar_verdicts` (two rows symmetric rendezvous) | unchanged | unchanged |

  The landing moved only the barrier axis; **`stall_long_sb` did not come down with form B**, and the attribution of the prologue's long_sb is outside this entry's certified scope.
  Accept a landing of form B by `stall_barrier` and by the LDG / `barrier()` gone from the prologue, not by long_sb.

## Gotchas

1. **Both the `dst` and the `mbar` operands must be converted to the peer via `mapa`; `src` is not.** The certified spelling passes the local offsets
   in, does `mapa.shared::cluster.u32` twice in the same asm block (once for the data face, once for the barrier),
   then issues `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`.
   Completion is recorded on **the peer's** barrier; the sender has nothing to wait on for this instruction (gotcha 3).
2. **Who arms the receive barrier is a choice; each of the two forms has its own certified scope (both form A and form B are GPU-certified, job numbers in Sources); do not mix them.**
   **Form A (the receiver arms)**: each receive barrier `init(…, 1)`, the receiver does `arrive.expect_tx(Σ bytes the peers will send to
   this barrier)`, **and only then** `barrier.cluster.arrive / wait`; the peer may issue once past the rendezvous. The precondition is that
   the receiver **knows the total bytes in advance** (too few is a hang, not an exception).
   **Form B (the sender arms)**: receive barrier `init(…, N)` (N = number of senders), the receiver **arms nothing**; each
   sender fires one remote `mbarrier.arrive.expect_tx.shared::cluster.b64 _, [mp], tx` at the peer, `tx` being the bytes it itself
   actually sends to that peer. The receiver needs to know no byte count in advance, and `tx` may differ between senders (whole run + small header /
   small header only); it flips only when the arrival count N and the tx count reach zero together (the PTX phase-completion condition: pending arrivals and tx-count
   zero at the same time), so "the bytes carried by the first arrive all arriving first" does not flip it early.
   Both forms share one invariant: **the receive face (and the barrier) must be ready before the cluster rendezvous**; after the rendezvous
   a peer running ahead may drop bytes into it at any time, so the receive face must be dedicated and must not overlap any buffer still in use.
3. **`cp.async.bulk.commit_group` + `cp.async.bulk.wait_group.read 0` do not cover s2s copies using the mbarrier completion
   mechanism.** Measured: overwriting the source face with −1 right after issue gives max_diff 12732 without a guard, and with the
   `commit_group; wait_group.read 0` guard **also** 12732 -- async groups only track copies using the bulk_group completion
   mechanism. The certified source-side drain is a **paired ack**: the receiver waits until its own data barrier flips (which proves the peer's
   copy has landed in full and therefore the peer's source face has been fully read), then fires a remote
   `mbarrier.arrive.shared::cluster.b64 _, [mp]` at the peer's ack barrier (`mp` via `mapa`); before rewriting / releasing the source face the sender
   waits on its own ack barrier (the `init` count = the number of acks it must receive; this entry certified 1 and 4).
   With this step added the same test gives 0.0. In the spelling ladder of the remote arrive, the `mbarrier.arrive.shared::cluster.b64` **without** a semantics modifier
   compiled and ran first; the two forms `.release.cluster` / `.relaxed.cluster` were never reached (uncertified).
4. **The source face was just written by an ordinary `st.shared`; before issue, `fence.proxy.async.shared::cta` + one CTA barrier are needed.**
   The certified form has both steps before every issue. Earlier, in another probe, an attempt of "issuing s2s right after the source face was written by st.shared,
   without a fence" gave `an illegal instruction was encountered` at launch -- whether the missing
   fence caused it **was not determined**; this entry vouches only for the form with the fence.
5. **The exit contract does not need a second cluster rendezvous.** A CTA's smem must not be read or written by peers after it exits;
   the certified form guarantees this by two things: (a) the data barrier of every round it receives has flipped ⇒ no peer will still write to it;
   (b) its ack barrier has received everything ⇒ no peer is still reading its source face. The whole kernel has only the prologue's one pair of
   `barrier.cluster.arrive / wait` (the former can be issued as early as right after init + arm, the latter pushed down to just before the first remote
   issue), and the KEEP delivery was numerically PASS in this form.
6. **Make "the ack never came" a FAIL rather than a hang.** In the probe the ack wait is a bounded spin of `mbarrier.try_wait.parity` + `nanosleep`,
   writing each CTA's hit count back to gmem, with a minimum of 0 judged FAIL; one spelling per subprocess
   ([`gpu_probe_ladder`](gpu_probe_ladder.md)). In a production kernel use `mbarrier_wait` instead.
7. **The footprint only says the instruction changed family, not that it got faster.** See the "Load-bearing conditions" section: this entry's KEEP reading is the slower side.
8. **The destination operand of a remote `arrive` / `arrive.expect_tx` must be the sink `_`.** PTX ISA: when the barrier is in
   `.shared::cluster` and not in this CTA's `.shared::cta`, it "cannot return a value. Sink symbol `_` is
   mandatory". The `tx` operand is a 32-bit register (the certified form passes a dynamic value with an `r` constraint: 2056 or 8, multiplied out of a 0/1 predicate,
   no branch needed). The spelling without `.sem.scope` (default `.release.cta`) is the first rung of the ladder;
   `.release.cluster` is the second rung (the probe tries rung by rung; the first that compiles wins).
9. **In form B the receiver must never arm as well.** Negative control (probe contract D): after the receiver does one extra local `mbarrier.expect_tx(total bytes)` before the rendezvous,
   the tx-count never gets back to zero and not a single CTA's barrier can flip -- the symptom is a **hang** (a FAIL under a bounded wait),
   not wrong numbers. "Sender expect_tx + receiver expect_tx too" is a shape stitched together from half of each form -- exactly this kind of mixing.
10. **The order of `arrive.expect_tx` and the copy within the sender bears no load.** Arrive first then copy, and copy first then arrive,
    are two cells of the probe (contracts B1 / B2): `complete_tx` arriving before `expect_tx` only makes the tx-count temporarily negative (the PTX-allowed range is
    −(2²⁰−1) … 2²⁰−1), and the phase cannot complete before the pending arrivals reach zero. By convention, arrive.expect_tx first, then issue.
11. **Prologue order: `init → fence → receive face ready → one pair of cluster rendezvous → only then this CTA's gmem reads / issue`.**
    Each CTA's `cluster_wait` waits for the `cluster_arrive` of the **slowest peer**, so every DRAM
    round trip placed before the rendezvous (reading indices, census, computing total bytes, broadcasting it again) is paid by the whole cluster 16 times at the slowest value; form B moves "the numbers the receiver needs to
    know" entirely into the sender's one `arrive.expect_tx`, leaving only init + fence + receive face ready before the rendezvous.
    The gmem reads, fences, whole-CTA barriers and issue after the rendezvous are this CTA's own business and no longer hold others back.
12. **Use `st.async` for the 8 B small header, accounted on the same peer barrier as the bulk copy.** PTX requires the destination address of `st.async` and the
    barrier to be in **the same (destination) CTA**, with a cluster larger than 1 -- so both addresses go through `mapa` (like the copy of gotcha 1);
    `.v2.b32` is one 8 B instruction, `complete_tx` 8. The DSL equivalent `cute.arch.store_async_dsmem(smem_ptr, value,
    mbar_ptr, peer)` is 4 B per instruction, two make 8 B (tx counted by the bytes actually sent).

### Load-bearing conditions (under which structure this footprint is worth something -- this entry's readings are the side where it is **not**)

Structural conditions delivered together with `cache_ops ["UBLKCP.S.G"] → ["UBLKCP.S.G", "UBLKCP.S.S"]`:
a grid of 128 CTAs, cluster 16, 256 threads per CTA, `warp_active_pct` ≈ 4.7 (one resident wave);
the exchange is 4 rounds of pairwise recursive halving, each round one s2s of 16/8/4/2 KB + one of 128 B issued by one thread, the whole CTA
waiting on the barrier of the peer's round, then one local smem merge (per round fence + CTA barrier + merge + CTA barrier),
8 copies / 30.5 KB sent per CTA in total; before the change, per-thread `st.shared::cluster.v2.b32` wrote directly to 16 peers
(4096 per CTA) + a second cluster rendezvous + one local 16-way merge. Under these conditions:

| Reading | Before | After |
|---|---|---|
| gate window | 0.01522 ms | 0.0168 ms (**+10.38%, slower**) |
| `stall_barrier` | 8.18 | 1.77 |
| `stall_long_sb` | 1.74 | 8.09 |
| `stall_mio` | 0.24 | 0.92 |
| `bankconf_st` | 7560.5 | 98453.5 |
| `bankconf_ld` | 90.5 | 33873.75 |
| `regs_per_thread` | 64 | 55 |
| `smem_dynamic` | 208968 | 205424 |
| `tensor_pct` | 1.63 | 1.55 |

That is: the barrier wait of the second cluster rendezvous really did disappear, but it turned into 4 **serial** rounds of "single thread issues → whole CTA
waits for the peer → local merge", and the waiting moved to long-scoreboard; the same delivery changed the local merge into a per-round FMA
smem round trip, and the jump in `bankconf_*` belongs to that side, not to the copy instruction -- so this ms is the combined price of "s2s exchange + per-round merge",
not the price of `UBLKCP.S.S` alone. What carries over is the way of reading: s2s bulk replaces "confirm the peer has finished writing" from one
cluster-level rendezvous with each CTA waiting on its own barrier; whether it bears load depends on **whether the rounds are serial and whether each round's
wait is covered by other work**; the shape of one wait per exchange round, with the whole CTA idling between rounds, did not hide the
latency on a tree with one resident wave. On another tree (several rounds in flight concurrently, issue spread out, merge overlapping the wait) this reading will change, and this entry does not vouch for them.

## Sources

- PTX ISA, "Data Movement and Conversion Instructions": `cp.async.bulk` (the `.shared::cluster.shared::cta`
  form + `.mbarrier::complete_tx::bytes`), `mapa`, `mbarrier.arrive` (the `.shared::cluster` remote
  form), `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`, `fence.proxy.async`,
  `barrier.cluster`, `cp.async.bulk.commit_group / wait_group` (tracking only the bulk_group completion mechanism)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- PTX ISA, the items used by form B: the `mbarrier.arrive` syntax table (`mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}{.multicast}.b64 _, [addr], txCount`)
  and "`.shared::cluster` but not in `.shared::cta` cannot return a value. Sink symbol `_` is mandatory";
  the count-range table in "Contents of the mbarrier object" (tx-count minimum −(2²⁰−1)); "Phase Completion of the
  mbarrier object" (pending arrivals at zero **and** tx-count at zero); `st.async{.weak}{.shared::cluster}.mbarrier::complete_tx::bytes{.vec}.type [a], b, [mbar]`
  (destination address and mbar must be in the same CTA, `%cluster_nctarank > 1`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols: `cutlass.cute.arch.mbarrier_init`,
  `cutlass.cute.arch.mbarrier_init_fence`, `cutlass.cute.arch.mbarrier_arrive_and_expect_tx`,
  `cutlass.cute.arch.mbarrier_wait`, `cutlass.cute.arch.cluster_arrive`,
  `cutlass.cute.arch.cluster_wait`, `cutlass.cute.arch.block_idx_in_cluster`,
  `cutlass.cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, bytes, peer_cta_rank_in_cluster=…)`
  (form B's remote arm, DSL-native), `cutlass.cute.arch.mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=…)`,
  `cutlass.cute.arch.store_async_dsmem(smem_ptr, value, mbar_ptr, peer_cta_rank)`,
  `cutlass.cute.arch.mbarrier_try_wait`,
  `cutlass.utils.SmemAllocator.allocate_tensor`, `Pointer.toint`,
  `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`;
  the launch's `cluster=[CSZ, 1, 1]`. For the general rules of the inline-asm channel see [`ptx_escape`](ptx_escape.md).
- Certified by a GPU probe (the probe is not included in this release) --
  **Status: ✅ GPU-certified**, round 2 of this certification. Four contracts: `A-dsl-s2s-api` (n/a, signature printed verbatim),
  `B-s2s-1round` (one round of 16 KB + 128 B pairwise s2s within cluster 16, accounted on the peer barrier, byte-exact),
  `C-s2s-drain-ack` (the same source face reused over two rounds: the no-ack control dirty, the `wait_group.read` guard dirty 12732,
  the ack guard 0.0), `D-merge4-exact` (4 exchange rounds composed into a whole-machine merge, `max_diff = 0`). Round 1's contract C
  was "guard the source with `commit_group + wait_group.read 0`", judged dirty (12732) -- that is not a failed probe; it
  turned gotcha 3 from a hypothesis into a measured fact; round 2 changed the contract to the ack accordingly. Verdict lines: round 1 `PROBE FAIL max_diff=12732  # C-s2s-readwait` (the other three
  PASS); round 2, four `PROBE PASS` (`A-dsl-s2s-api` n/a, `B-s2s-1round` 0,
  `C-s2s-drain-ack` 0, `D-merge4-exact` 0).
- Certification probe (form B, a second probe) -- **Status: ✅ GPU-certified**,
  `PROBE PASS max_diff=0.0`, round 1 of this certification, all five contracts pass in the same round.
  The five contracts: `A-api` (n/a, the DSL symbols and the `peer_cta_rank_in_cluster` parameter exist), `B1-sender-armed-ptx`
  (cluster 16 all-to-all, PTX remote `arrive.expect_tx` before the copy, five payload patterns byte-exact, every
  CTA flips), `B2-sender-order-free` (arrive after the copy), `C-sender-armed-dsl` (DSL-native
  `mbarrier_arrive_and_expect_tx(…, peer_cta_rank_in_cluster=)` + `store_async_dsmem`),
  `D-receiver-must-not-arm` (the control exact; when the receiver also arms, none flips, n/a).
  Measured readings: the three cells B1 / B2 / C under all five payload patterns (active 128 / 0 / 64 / 8 / 85 of 128) are
  `max_diff=0`, `flipped=128/128`, `min_hits=40` (every CTA's receive barrier flips, slots and small headers byte-exact);
  the first rung of the remote `arrive.expect_tx` spelling ladder, plain (no sem/scope modifier), compiled, with no need for `.release.cluster`;
  the negative control of cell D (the receiver doing one extra local `expect_tx` before the rendezvous) is `flipped=0/128`, `min_hits=0` under all five patterns,
  matching gotcha 9 "a receiver that arms again = never flips" word for word.
- KEEP certification (contract-axis change): KEEP in an eval job (the job number was not recorded in the status record), axis `cache_ops`
  `["UBLKCP.S.G"]` → `["UBLKCP.S.G", "UBLKCP.S.S"]`, gate 0.01522 → 0.0168 ms; the
  NCU readings of the same KEEP are in the "Load-bearing conditions" table; that KEEP is numerically PASS (the whole cluster's 4 exchange rounds + the exit contract with no second rendezvous).
