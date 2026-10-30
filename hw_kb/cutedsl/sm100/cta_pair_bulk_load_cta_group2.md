# cta_pair_bulk_load_cta_group2 (bulk-tensor loads with `.cta_group::2`: one instruction, one barrier, two CTAs)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> This entry answers one question: **on a CTA pair, how to make the bytes moved into both CTAs accounted by the same mbarrier.**
>
> The answer is not "redirect the barrier pointer to the leader" -- that is the **result**, not the **mechanism**. The mechanism is
> the `.cta_group` modifier on `cp.async.bulk.tensor`: it decides whether this load **is allowed** to deliver its completion signal
> to another CTA's barrier. The default `.cta_group::1` **does not allow** it; doing so is UB;
> only `.cta_group::2` allows it.
>
> | | Modifier | Where the mbar may be | Consequence of redirecting to the peer |
> |---|---|---|---|
> | default | `.cta_group::1` | **must** be in the same CTA as `dstMem` | does not signal, **UB** |
> | pair | `.cta_group::2` | in the same CTA as `dstMem` **or** its peer-CTA | this is exactly what it is for |
>
> In CuTe DSL this modifier is **one constructor argument** of the op:
> `CopyBulkTensorTileG2SOp()` and `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` differ only in it.
> For both, the `cute.copy` call shape, descriptor, box and byte count are **all identical** -- so **omitting it raises no error,
> fails no compile, gives no wrong numbers; that barrier just never fills**, showing up as a hang. This is the number-one shape this entry guards against.
>
> **But the hang is only its first day.** A hanging pipeline does not survive an afternoon: it gets patched with an extra cross-CTA rendezvous
> (one more per-slot barrier layer, letting the two CTAs each announce "my half has arrived"), and then **it no longer hangs and no longer gives wrong numbers**,
> and looks finished. What is left is a pipeline whose delivery side runs at the default `cta_group` and whose consumer side runs at `2`, with the difference paid every slot by that
> extra rendezvous. **This end state is the shape actually encountered, and the "hang" fingerprint does not cover it** --
> for the detection rule see the third class of triggers under "When needed"; for the construction order that takes it back see "Retirement order".
>
> "Who is entitled to wait on the full barrier / whether the last round should still peek" are two **downstream** guards, see
> [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md): that one covers **waiting**, this one covers **delivery**.
> The order is: build this entry's delivery first, then guard those two -- while delivery is not built, however right the consumer-side guards are, they wait on a
> barrier that never fills.

## Purpose

Blackwell's `tcgen05` MMA has a `cta_group=2` variant: a **CTA pair** completes one MMA together.
The pair is defined by the hardware, not by convention: **two CTAs whose `%cluster_ctarank` differ only in the lowest bit form a pair**;
the one whose last bit is 0 is the even CTA, the one with 1 the odd CTA, and they are each other's peer (PTX ISA §9.7.18.5.1 / §9.7.18.5.2).

So the supply side has to answer a question a 1-CTA pipeline never meets: **who accounts the bytes of the pair's two halves.**
There are only two choices, and this entry covers the second:

| Accounting shape | one barrier per CTA | **one barrier shared by the pair** |
|---|---|---|
| who arms | each arms its own share of bytes | only the leader arms, for the bytes of **the whole pair** |
| `.cta_group` | `::1` is enough | **must be `::2`** |
| consumer side | each of the two CTAs waits on its own | only the leader is entitled to wait (see the downstream entry) |
| extra rendezvous | **needed**: the MMA is issued by the leader alone, which must know the peer's half has also arrived | not needed: the barrier being full means both halves have arrived |

The first shape runs perfectly well; its cost is one extra cross-CTA rendezvous per slot, and that rendezvous sits on the critical path of MMA issue.
The second **folds that rendezvous into the TMA completion signal** -- that is why `.cta_group::2` exists.

## When needed (trigger)

**Structural trigger** (deterministic; look at this first): the code contains both

1. `tcgen05.CtaGroup.TWO` (or a `cluster` laid out along M with `mma_tiler[0]` twice the single-CTA row count);
2. an operand face **spanning two CTAs** that you want accounted by **one** barrier.

Then check against the table below item by item -- **all four must hold at the same time**; missing any one gives the same symptom (a hang):

| # | Requirement | What missing it looks like |
|---|---|---|
| 1 | the load op is built as `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` | the default `CtaGroup.ONE`, **redirecting to the peer is UB** |
| 2 | the byte count of `arrive_and_expect_tx` = the bytes actually delivered for **the whole pair** | arming only your own half ⇒ the barrier flips early and reads half a face |
| 3 | arming is done only once, by the **leader** | both sides arm ⇒ the expected value doubles and never fills |
| 4 | the barrier pointer every issuer passes points at **the leader's** barrier | each delivers to its own ⇒ the leader receives only its own half |

**Symptom triggers** (it has already run). This entry's omissions **do not slow things down; they hang**, and the fingerprint is very specific:

- the job is killed for timing out (rc=137 or similar), not an illegal instruction, not wrong numbers;
- the hang is at the **first** consumption point, not after wrapping around -- the first slot never filled;
- the same code runs fine on a single CTA / `cluster=1` -- this is what best separates it from "the descriptor is built wrong",
  because a wrong descriptor is wrong at every cluster size.

**Criterion for telling it apart from "the byte count is wrong"**: temporarily change `expect_tx` to the byte count of **your own half** and run again.
If it goes from hanging to "finishes but only half the data is right", what is missing is requirement 1/4 (the delivery did not cross over);
if it still hangs, what is missing is requirement 3 (both sides armed).

**Compensation trigger (the non-hanging kind, and the most common end state)**. The two classes above assume the symptom is still there. Once someone patches it with
a cross-CTA rendezvous, the symptom is gone, and only three checkable traces remain:

1. **One more barrier layer than the contract.** The contract wants two per slot (full / empty); the compensated shape has three per slot, and the extra
   layer's arrival count **equals the number of CTAs in the pair** (each CTA announcing once "my half has arrived").
   A per-slot barrier layer with count=2 is the signature of "the two halves accounted separately, needing an extra rendezvous".
2. **A hard-coded exclusion assertion in the source**, of the form "the peer's barrier cannot be completed remotely / a transaction
   can only complete its own CTA's barrier", and **usually backed by an experiment someone ran themselves**. The precondition for this assertion to be true
   is that the constructor argument was left at its default -- see gotcha 8.
3. **A machine-checkable predicate (the only one that does not rely on reading comments): in the disassembly of one kernel, the `.2CTA` suffixes of the two families diverge.**

```bash
# collect every UT* dotted opcode used by this one kernel
nvdisasm -c k.cubin | grep -oE '\bUT[A-Z]+[A-Z0-9_.]*' | sort -u
#   delivery family  A = UTMALDG* / UTMASTG*
#   consumer family  B = UTCCP* / UTCOMMA* / UTCBAR* / UTCATOMSWS* / UTC*MMA*
# predicate: B non-empty with .2CTA appearing in B, while A non-empty with **no** .2CTA anywhere in A
#   ⇒ the delivery side is built at cta_group=1 and the consumer side runs at cta_group=2: the two sides disagree.
```

On a consistent tree the two families either both carry `.2CTA` or both do not; **if only one family carries it, the two sides do not match**. This predicate is
mechanical: it reads no comments, no variable names, does not depend on a symptom showing up, and **it holds just the same when the numerics are all right and the profiler is all green**.

**Mind the coverage when using the existing census**: the `cache_ops` axis of `orch_tools` collects `UTMALDG*` / `UTMASTG*` verbatim
(the delivery family is on the axis), but the tcgen05 family is **not** in the cache family table, so **`cache_ops` alone shows only half of the
predicate** -- the other half must be taken from the dump yourself with the one-line grep above. Seeing bare
`UTMALDG.2D` / `UTMALDG.3D` in `cache_ops` while this tree runs a `cta_group=2` MMA is already enough for suspicion, but **the verdict
needs both families**: a load family without the suffix is legal in itself (a genuine 1-CTA pipeline should look like that).

**Anti-trigger**: `cta_group=1` pipelines, single CTA, and shapes where the two CTAs of a pair each hold independent operands and each
issue MMA -- those use the first accounting shape, and this entry does not apply (in that case **do not** add `CtaGroup.TWO`;
it turns a self-consistent 1-CTA pipeline into one that needs leader accounting).

## Mechanism: three layers; missing any one leaves the hang unexplained

**Layer 1: `.cta_group` decides where the barrier may live.** PTX ISA §9.7.10.28.5.3 fixes this:
under `.cta_group::1`, the mbar **must** be in the shared memory of the same CTA as `dstMem`; under `.cta_group::2`,
the mbar may be in the same CTA **or its peer-CTA**. The same section has a separate constraint: when the mbar is in the shared memory of another CTA in the
cluster (a remote barrier), `.dst` must be `.shared::cluster`;
writing `.shared::cta` **does not signal that remote barrier** and is UB.

Together these two sentences are this entry's foundation: **"redirecting the pointer to the leader" under the default modifier is not a more aggressive spelling;
it is an undefined spelling.** It raises no error -- nothing anywhere reports it -- it simply does not deliver. So code that "wrote
`map_dsmem_ptr` as instructed yet still hangs" is usually missing not the pointer line but that argument on the op.

**Layer 2: one instruction delivers one box, not one pair.** What `.cta_group::2` opens up is **where the accounting goes**
(the completion signal may hit the peer's barrier), **not** the amount moved. The descriptor's box is still that box, and
one `cute.copy` call still moves one share. So the pair's two halves of bytes are completed by **each CTA issuing one load**,
both delivering their completion to the leader's same barrier, and what the leader arms is the sum of the two.

This layer has a direct construction corollary, and it is the easiest place to get backwards: **both CTAs on the producer side must issue;
no leader guard may be added**. The consequence of adding a leader guard to the producer is not slowness but a **deadlock** -- the follower no longer
issues its half, and the leader's barrier is forever half the bytes short. The consumer side is exactly the opposite (see the downstream entry).
**The guards on the two sides point in opposite directions**; this is the most counter-intuitive point of the pair supply path.

**Layer 3: `expect_tx` is a self-consistency invariant, not a constant you can copy.** The condition for the barrier to flip is
"accumulated arrived bytes == the bytes declared at arm time". So the only correct usage is:

```
expect_tx (armed once by the leader) == Σ (bytes of every load actually issued in this slot by all CTAs of the pair)
```

Compute both sides from the `cosize` of **the same** smem layout; do not count the layout on one side and copy a number on the other --
as long as either number is hand-written, they diverge when the tile size changes, and the divergence shows up as a **hang** (undercounted) or as
**reading a slot that has not finished being written** (overcounted: the barrier flips early, and **nothing anywhere reports it**).

## Minimal runnable example (CuTe DSL)

Only the delivery path is shown. The roles, the barrier accounting, the two loads, and the **single** `cta_group` argument:

```python
from cutlass.cute.nvgpu import cpasync, tcgen05

# --- host side: this argument on the op is the whole mechanism -------------------------------
# omitted = CtaGroup.ONE = the completion signal can only hit its own barrier.
op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
tma_atom, gX = cpasync.make_tiled_tma_atom(op, mX, x_smem_layout, cta_tiler, 1)
gX = cute.zipped_divide(gX, cta_tiler)

# expect_tx computed from the layout, the same source for both sides: per_cta is the bytes one CTA issues per slot,
# pair_bytes is the number the leader arms. Hand-writing either one makes them diverge when the tile changes.
per_cta = cute.cosize(cute.select(x_smem_layout, mode=[0, 1])) * X_DTYPE.width // 8
pair_bytes = 2 * per_cta

# --- device side: roles within the pair -------------------------------------------
# pair members differ only in the lowest bit of %cluster_ctarank; laid out along M, the leader is the one whose last bit is 0.
cta_rank = cute.arch.block_idx_in_cluster()
coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
is_leader = coord_vmnk[0] == 0
leader_rank = cta_rank - coord_vmnk[0]       # the rank of this pair's leader in the cluster

tXsX, tXgX = cpasync.tma_partition(tma_atom, 0, cute.make_layout(1),
                                   cute.group_modes(sX, 0, 2), gX)

# --- every slot: arm once (leader), issue two (one from each CTA) -------------------
if is_leader:                                  # requirement 3: arm only once
    with cute.arch.elect_one():                # the arm needs elect; the copy does not (see gotcha 3)
        cute.arch.mbarrier_arrive_and_expect_tx(full_bar + s, pair_bytes)
# the arm must be visible to all CTAs of the pair before the two loads, or the follower may deliver to a
# barrier not yet armed. In the steady-state pipeline this is guaranteed by the empty-barrier handshake;
# in the prologue / first slot there is no such handshake, so place one cluster rendezvous yourself.
cute.arch.cluster_arrive()
cute.arch.cluster_wait()

# no guard: both CTAs must issue their own half. A leader guard = deadlock.
bar = cute.arch.map_dsmem_ptr(full_bar + s, leader_rank)   # requirement 4: deliver to the leader's
cute.copy(tma_atom, tXgX[None, (my_tile, k)], tXsX[None, s], tma_bar_ptr=bar)

# afterwards only the leader is entitled to wait on this barrier -- for those two guards see cta_pair_consumer_peek.
```

**With a ready-made pipeline** the same mechanism is handled by the infrastructure, but it takes effect **only if you hand it the pair's shape**:
`pipeline.PipelineTmaUmma.create(..., tx_count=<the pair's bytes>, cta_layout_vmnk=<vmnk layout>)`
-- it infers `cta_group` from the V mode of `cta_layout_vmnk` and puts `arrive_and_expect_tx` under the
leader predicate. **When `cta_layout_vmnk` is omitted or its V mode is 1, it degrades to `CtaGroup.ONE`**,
yet `tx_count` is armed with the value you gave -- which combines into exactly the hang "armed the pair's bytes, but only half will arrive".
It also needs a **matching `cta_group=2` load atom**: `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A`
/ `_B` / `_SFB` choose which op to use by `(cluster_shape, tiled_mma.thr_id)`,
and `cute.nvgpu.make_tiled_tma_atom_A` / `_B` then project the box by the MMA tiler and `tiled_mma`.
**This set of helpers does not exist in some builds** (the class names of the `pipeline` submodule have changed across versions) --
`dir()` it first to confirm; if absent, build the hand-written form above; the mechanism is the same.

## Gotchas

1. **Changing only the pointer, not the op.** `map_dsmem_ptr(bar, leader_rank)` written exactly right, while
   `CopyBulkTensorTileG2SOp()` is left at its default -- this is this entry's most common failure, because **it looks right**:
   it compiles, it runs, it reports no error at all; the delivery is just UB per the spec, and the barrier never fills.
   Criterion: **check the op's constructor argument first, then the pointer line.**
2. **Adding a leader guard on the producer side.** Getting the direction backwards gives a deadlock, not slowness: the follower does not issue its half,
   and the leader's barrier is half the bytes short. Mnemonic: **delivery is the business of both CTAs; waiting is the leader's alone.**
3. **Wrapping `cute.copy` in any lane-level predicate.** The TMA atom elects by itself (membermask = the whole warp);
   one more layer outside -- whether `with elect_one():` or `if tidx == 0:` -- locks the other 31 lanes
   out of its own `elect.sync`: a **deadlock**. The harm is not in the name `elect_one` but in "a lane predicate pressed on top of a
   warp-sync instruction": the copy's predicate may only use quantities of warp granularity or coarser
   (`rank` / `warp_idx`, the whole warp entering and leaving together). Conversely, `mbarrier_init` and
   `arrive_and_expect_tx` **must** be single-threaded, and the outer predicate of the `elect_one` the latter relies on must likewise be
   warp-uniform -- for both rules see [`elect_one`](elect_one.md) gotchas 4 and 6.
   In one stretch of code these two kinds of calls sit right next to each other, so it is easy to wrap both or neither.
   **Symptom fingerprint**: the very first launch does not return, the job grinds to the wall clock and is KILLed (rc=137), and the log has
   not a single verdict line. This is a deadlock, not wrong numbers -- look at the predicate granularity first; do not go hunting for a numeric bug.
4. **Copying a constant into `expect_tx`.** See mechanism layer 3. Undercounting hangs; overcounting reads a slot not yet fully written, and **the latter reports no error**.
   Compute both sides from the `cosize` of the same layout.
5. **The first slot's arm not ordered before the delivery.** In steady state the empty barrier covers it; in the prologue nothing does. Without this rendezvous
   the failure is **intermittent** (depending on the relative progress of the two CTAs), harder to track down than a deterministic deadlock.
6. **Adding `CtaGroup.TWO` to a load that does not need pair accounting.** It does not make the load faster -- it only opens up
   where the accounting goes. If after adding it each still delivers to its own barrier, the shape is unchanged and it was added for nothing; if you also change the arm to
   pair bytes along the way without changing the consumer side, a self-consistent pipeline turns into a hanging one.
7. **The silence of the default.** The `cta_group` field of `CopyBulkTensorTileG2SOp` **has a default value**; construction succeeds whether you write it or not,
   and the `cute.copy` call shape of the two atoms is **exactly the same**. So this argument never shows up in any
   review because it was left out -- **it shows up only in the SASS opcode suffix** (whether `UTMALDG.*` carries `.2CTA`).
   To confirm which one you built, read that axis of the SASS census, not the source.
8. **Asking "can the peer really deliver" with an experiment that pins `cta_group` to its default.** This is the cognitive version of gotcha 1,
   and the most expensive failure in this entry: the experiment itself is very clean -- the three axes **pointer** (direct / `map_dsmem_ptr`
   self-mapped / redirected to the leader), **arm position** (leader arms alone / each CTA arms once with the peer rank) and
   **byte count** (own half / whole pair) are split one by one, and the control group (each delivering to its own) passes nicely.
   All three axes changed, but the one constructor argument that decides "allowed or not" stayed at its default from start to finish, so this table measures
   "under `.cta_group::1` no change works" -- which is exactly the UB the spec describes. Swap the op of the same spelling for
   `CtaGroup.TWO`, and it comes back to life on the spot, byte-exact.
   **And the most dangerous thing about this table is not that it is consistent, but that it is not** (measured: of five cells, four dead and one alive). Every spelling where "flipping
   must rely on the remote half's accounting" -- arming the whole pair while delivering only half yourself, or delivering not a single byte yourself
   -- **flips in no case**; the one cell that flips is the one that **arms only its own half**, and what flips it is **its own**
   local delivery (a legal `.cta_group::1`); whether the remote half signed in never enters this reading (see gotcha 9).
   So the table does not read like "the hardware does not allow it" but like **"oh, then only arming your own half works"** --
   and that sentence is the design spec of the compensated architecture: each arms its own half, flips locally, and the cross-CTA step is patched with another barrier layer.
   **An experiment that did not put `cta_group` on an axis does not just fail to find the answer; it hands you a wrong architecture.**
   **Criterion**: in any experiment about "where the completion signal can land", `cta_group` must be **one axis** of the experiment,
   not scenery; if it is not on an axis, the experiment is uninterpretable, however completely the table is filled.
   The cost is not only that this experiment was wasted: such conclusions get written into comments as measured facts, a per-slot rendezvous barrier gets added on that basis,
   and every round afterwards pays rent on it -- see "Retirement order".
9. **Taking "the barrier flipped" as evidence that "the pair was accounted".** The tx-count records **bytes**, not **where they came from**:
   as long as `expect_tx` does not exceed the bytes the waiter delivered **itself**, that barrier fills on local delivery alone,
   and whether the remote half was ever credited is **unobservable in this reading**. So --
   **a cell of the experiment is discriminating only when `armed_tx` is strictly greater than the waiter's own delivered bytes** (the extreme and cleanest cell:
   let the waiter deliver not a single byte, leaving only the remote delivery; if the barrier still flips, it really crossed over).
   The same holds at run time: in the compensated shape each CTA arms its own half, so **every slot's full barrier flips on time**,
   watchdog, court and numeric checks are all green -- no hang, no wrong numbers; the pair's rendezvous has just been moved to another barrier layer that charges for it.
   **No run-time reading can tell this shape from the correct one**, which is exactly why the verdict must rest on the SASS suffix
   predicate (see item 3 of "Compensation trigger") rather than on symptoms.
10. **Forgetting the fence / visibility when waiting on a remote barrier on a pair.** The completion signal carries `.release`
   semantics, scope `.cluster` (same section of the PTX ISA). The waiting side just uses the mbarrier phase protocol as usual;
   but cross-CTA smem reads and writes **you write yourself** do not enjoy this, and you must order them yourself.

## Retirement order: taking back the compensating barrier layer

When the compensated shape is already built, **do not change it all in one step**. That barrier layer is a real cross-CTA rendezvous, and besides "both halves have arrived"
it very likely also carries some other ordering on the side; deleting it in one go may lose something other than what you think. Follow the
order below; at the end of each step the tree should be runnable with unchanged numerics (the first three steps swap the mechanism; only step 4 touches that layer):

| Step | Action | What you should observe afterwards |
|---|---|---|
| 1 | **Only** change the load op of every operand face feeding the pair to `CopyBulkTensorTileG2SOp(CtaGroup.TWO)`, touching nothing else | numerics unchanged, topology unchanged; `UTMALDG` starts carrying `.2CTA`, and the "suffix divergence" predicate goes out |
| 2 | redirect every issuer's barrier pointer to the leader (`map_dsmem_ptr(bar, leader_rank)`); change the arm to **once by the leader**, `expect_tx` = pair bytes (**from the same source** as the per-CTA number: one layout's `cosize`), and delete each CTA's own arm | the full barrier now has one party accounting both halves; **on the delivery side both CTAs still issue**, and adding a leader guard here = deadlock |
| 3 | change the consumer side to wait on the leader's one full barrier, and add the `is_leader` entitlement guard to **every** consumer call (including non-blocking peeks) | the guards on the two sides point in opposite directions: delivery is the business of both CTAs, waiting is the leader's alone (see [`cta_pair_consumer_peek`](cta_pair_consumer_peek.md)) |
| 4 | **enumerate first, then delete**: list **all** arrive/wait call sites on that barrier layer, and for each ask "is what it waits for 'both halves of bytes have arrived'". If yes, delete it; if not, that ordering must move to the empty barrier or elsewhere, **and must not be deleted along with it** | one fewer per-slot barrier layer; numerics unchanged |
| 5 | machine re-check | both families' suffixes consistent (both carrying `.2CTA`), per-slot barrier layers back to 2, numerics byte-for-byte unchanged |

Step 1 can stand on its own because what `.cta_group::2` opens up is **where the accounting goes**, not the amount moved: while the pointer still
points at its own barrier, it is a legal and equivalent spelling. This gives a **zero-risk foothold** -- first get that
argument down to the ISA and see it in SASS, then touch the topology.

**Step 4 is the only step in this order that can go wrong silently.** A per-slot barrier layer with count=2 is by nature a whole-pair rendezvous;
deleting it deletes an edge **everyone can wait on**; if some CTA used to rely on it to know "the peer has finished reading this slot",
that edge belongs to the **empty/release** direction, unrelated to "the bytes have arrived", and must stay on the empty barrier. Enumerating call sites is a mechanical action;
skipping it from memory costs an intermittent, progress-dependent wrong number.

**This section is a construction order, not a certified mechanism**: what is GPU-certified is the mechanism each of its steps rests on (see Sources).

## Sources

- Certification probe (the probes are not included in this release) -- **Status: ✅ GPU-certified**,
  final verdict `PROBE PASS max_diff=0.0` (round 3 of this certification; the probe's own criterion is
  "PASS only if all four PROBE lines pass", so PASS means all four pass in the same round). The same data, one cluster of 2, seven spellings, **one launch per spelling**: all differences between spellings
  are made `Constexpr`, so every launch gets fresh smem and fresh barriers, and a spelling
  that does not complete can neither pollute its neighbours nor hang the job (every wait is a bounded `test_wait` spin). Each time the
  smem is first prefilled with −1, and after the run it is read back element by element and compared with numpy -- because what `.cta_group::2` opens up is where the accounting goes
  and not the amount moved, "the barrier flipped" and "the bytes really arrived" must be tested separately. Coverage: the control group (1-CTA local
  barrier), four `.cta_group::2` spellings (both CTAs issue × local / leader barrier; only the
  leader issues × single box / pair bytes), a sixth spelling with the descriptor box spanning the whole pair, and the UB shape of redirecting to the peer under the default
  modifier (placed last).
- Certification probe of the detection rule -- **Status: ✅ GPU-certified**,
  final verdict `PROBE PASS max_diff=0.0` (round 2 of this certification; the probe's own criterion is
  "PASS only if all five PROBE lines pass", so PASS means all five pass in the same round).
  It exercises only two things, orthogonal to the previous probe's four cells: (1) **the criterion of the confounded experiment** -- on the same data, the three axes pointer /
  arm position / byte count are split into five remote spellings with the op left at the default `cta_group`, and grouped by "whether flipping **must**
  rely on the remote half's accounting": the three cells with `armed_tx` strictly greater than the waiter's own delivered bytes (arming the whole pair
  while delivering half yourself ×2, delivering 0 bytes yourself ×1) **flip in no case**, and the two cells with `armed_tx` equal to the own delivered bytes **both flip** --
  in the latter "only the leader issues" and "both CTAs issue" give the same reading, so that flip is **attributed** to local
  delivery rather than remote (the control group "each delivering to its own" passes in the same round, showing the table is not empty). And the same whole-pair spelling with **only**
  the constructor argument changed to `CtaGroup.TWO` completes on the spot with both halves byte-exact ⇒ what stands accused is the argument that was not treated as a
  variable, and the single "alive" reading is exactly where the compensated architecture comes from; (2) **the suffix divergence predicate** -- a kernel containing both the
  load family and the tcgen05 family, differing only in the load op's constructor argument: in the disassembly of the two compilations the predicate is a true positive and a true negative respectively
  ⇒ the predicate is machine-checkable and does not depend on a symptom showing up.
  The method follows the previous one: one launch per spelling (all differences made `Constexpr`), every wait a bounded
  `test_wait` spin, smem prefilled with a sentinel outside the data domain to test "the barrier flipped" and "the bytes arrived" separately,
  spellings that do not complete placed last.
- Public interfaces: `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp` (field `cta_group`),
  `CopyBulkTensorTileG2SMulticastOp`, `cpasync.make_tiled_tma_atom`, `cpasync.tma_partition`,
  `cpasync.create_tma_multicast_mask`, `cutlass.cute.nvgpu.make_tiled_tma_atom_A` / `_B`,
  `cutlass.utils.blackwell_helpers.cluster_shape_to_tma_atom_A` / `_B` / `_SFB`,
  `cutlass.cute.nvgpu.tcgen05.CtaGroup`, `cutlass.cute.arch.map_dsmem_ptr`,
  `cutlass.cute.arch.block_idx_in_cluster`, `cutlass.cute.arch.mbarrier_arrive_and_expect_tx`,
  `cutlass.pipeline.PipelineTmaUmma` (the `tx_count` / `cta_layout_vmnk` parameters of `create`).
- PTX ISA, the `.cta_group` clauses of `cp.async.bulk.tensor` (the mbar may live in the peer-CTA,
  a remote barrier requires `.shared::cluster`, the default is `.cta_group::1`):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor>
- PTX ISA, the definitions of CTA Pair and Peer CTA (`%cluster_ctarank` differing only in the lowest bit):
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#cta-pair>
- Disassembly and census conventions (`UT*` dotted opcodes collected verbatim, the coverage of the cache family table): this repo's `orch_tools`
  `cache_ops` axis and the `CUTE_DSL_KEEP_CUBIN` / `CUTE_DSL_DUMP_DIR` +
  `nvdisasm -c` / `cuobjdump -sass` path of `orch_census_run`.
