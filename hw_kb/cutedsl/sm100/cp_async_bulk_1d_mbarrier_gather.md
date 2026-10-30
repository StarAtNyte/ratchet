# cp_async_bulk_1d_mbarrier_gather (non-tensor-map 1D bulk async copy: row-by-row indirect gather into an mbarrier-accounted smem ring)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; footprint readings taken from an sm100 SASS census (`UBLKCP.S.G`).
>
> What this entry records is **one knob + its SASS footprint + the conditions under which it bears load**, not a percentage:
>
> | | |
> |---|---|
> | **Knob** | for each indexed row issue **one** (or a few) `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes` (non-tensor-map, by byte count, destination address any 16 B aligned), with all rows of a stage issued serially by **one thread**; one `mbarrier.arrive.expect_tx(Σ bytes actually issued in this stage)` per stage before the batch, and `mbarrier_wait(parity)` on the consumer side; the optional `.L2::cache_hint` operand is produced by `createpolicy.fractional.L2::evict_first.b64` |
> | **Footprint** | dossier census axis `cache_ops` (verbatim dotted opcode): `[]` → `["UBLKCP.S.G"]`; `rings`: `[]` → `["stride=0x1,slots=9"]` (a 2-slot ring in the source -- this axis is a codegen fingerprint, see gotcha 8). The copies with and without `.L2::cache_hint` are in the same kernel, and the census reads only **one** dotted opcode ⇒ the policy operand **does not show in the opcode** |
> | **Load-bearing conditions** | under the structure "one independent gather stream per CTA, one 2-slot ring per CTA, 128 bulk copies per stage issued serially by one thread, the whole CTA waiting on one mbarrier", the gate window went **0.15613 → 0.19545 ms (+25.18%, slower)**; on the same tree the form "a batch of 16 B `cp.async` per thread (LDGSTS), 36 in flight at once, one wait" reads **0.156 → 0.0734 ms**. The footprint appearing ≠ it being worth something, see the "Load-bearing conditions" section |
>
> **This entry is certified at two levels**: the **byte correctness** of the copy + accounting + ring phase is certified by a GPU probe
> (`PROBE PASS max_diff=0`, all four contracts pass in the same round); the footprint on the `cache_ops` axis and the ms readings are certified by the dossier census + gate window of one
> delivered KEEP. The cache-policy bits themselves are numerically invisible; their **value** was not priced separately in this entry
> (they came in the same package as the copy form); only "they can be issued and the data is still byte-exact" is certified.

## Purpose

Change "move rows into smem one by one according to an index table" from **a per-thread LDG→STS round trip** (or a per-thread 16 B `cp.async`)
into **one byte-counted bulk async copy per row**: one instruction moves a whole contiguous run of bytes (this entry has certified the two lengths 1024 B and
128 B), and completion goes neither through registers nor through `commit_group / wait_group`; instead **the hardware records the number of arrived
bytes in an mbarrier** (`complete_tx`); the consumer side only waits for one barrier's phase flip.
It is the branch of the TMA family that **needs no tensor map**: the source address is a plain gmem pointer + a byte count, so
indirect addressing (row numbers coming from another table) needs the descriptor to express nothing.

## When needed (trigger)

- The supply is an **indirect row-by-row gather** (row numbers from an index table, rows not contiguous in gmem), and each row is a contiguous run of
  ≥ 16 B; a tensor-map TMA box cannot frame it.
- The dossier census's `cache_ops` axis is `[]` or has only the `LDGSTS` family, and you want the `UBLKCP.S.G` footprint
  (one async copy per whole run, completion through the mbarrier tx-count).
- The profiler symptom is the producer segment dominated by `stall_long_sb` (the latency of each LDG paid serially) -- **but read
  the "Load-bearing conditions" section first**: trading long_scoreboard for barrier waits is not the same as getting faster; on one tree this entry measured a slowdown.
- You are already writing this PTX and are stuck on: the DSL copy-atom spelling does not compile / what to put in `expect_tx` / how to account invalid
  rows (index −1) / how to guard the issuing thread.

**Look at the other path first**: when the destination is a **swizzled operand face** read directly by MMA and each row is 128 B, tensor-map TMA's
`.tile::gather4` lands 4 rows from 4 arbitrary row coordinates in one instruction, with the descriptor applying the swizzle, so the issue count is 1/4 of "one per row" --
see [`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md). This entry's 1D bulk carries no swizzle and suits linear destinations.

## Minimal runnable example

Skeleton extracted from the certification probe: a 2-slot ring, R rows per stage, each row in two pieces (piece A of `LAT_B` bytes
from table A, piece B of `ROT_B` bytes from table B) landing in a row-major staging face (row pitch `PITCH_B`, a multiple of 16 B).
The copy and `expect_tx` go through inline PTX (form (2) of [`ptx_escape`](ptx_escape.md)),
`createpolicy` through form (3); barrier init / wait use the DSL's own `cute.arch.mbarrier_*`
(in the probe **both** the PTX and the DSL spellings of init / arrive are certified; pick either).

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as _MT
from cutlass.cutlass_dsl import dsl_user_op

ATT = llvm.AsmDialect.AD_ATT

# ---- form (2): address operands, side effects only. The smem operand is the Int32 shared-window
#      address given by Pointer.toint(), gmem is Int64; nb is the byte count; mbar is also an Int32 smem address.
@dsl_user_op
def bulk_g2s(dst, src, nb, mbar, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nb).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global"
        ".mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# The same instruction with an L2 cache-policy operand (the fifth operand, Int64).
@dsl_user_op
def bulk_g2s_hint(dst, src, nb, mbar, pol, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [cutlass.Int32(dst).ir_value(loc=loc, ip=ip),
         cutlass.Int64(src).ir_value(loc=loc, ip=ip),
         cutlass.Int32(nb).ir_value(loc=loc, ip=ip),
         cutlass.Int32(mbar).ir_value(loc=loc, ip=ip),
         cutlass.Int64(pol).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)

# ---- form (3): return value only. Built once per CTA; every copy afterwards carries the same value.
@dsl_user_op
def l2_evict_first(*, loc=None, ip=None) -> cutlass.Int64:
    return cutlass.Int64(llvm.inline_asm(
        _MT.i64(), [],
        "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l",
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))

# ---- geometry (the certified readings come from this set): R rows/stage, two pieces per row, row pitch a multiple of 16 B.
NSTG, R = 2, 64
LAT_B, ROT_B = 1024, 128          # byte counts of the two pieces (both multiples of 16)
PITCH_B = 1168                    # row pitch: 1024 + 128 + 16 B pad, a multiple of 16 B
TX = R * (LAT_B + ROT_B)          # expect_tx of one stage = Σ bytes actually issued = 73728

@cute.kernel
def gather_ring(mA: cute.Tensor, mB: cute.Tensor, mIdx: cute.Tensor,
                mOut: cute.Tensor, ntile: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass_utils.SmemAllocator()
    sFace = smem.allocate_tensor(            # (stage, row, 16B column, 4 words), row-major
        element_type=cutlass.Int32,
        layout=cute.make_layout((NSTG, R, PITCH_B // 16, 4),
                                stride=(R * PITCH_B // 4, PITCH_B // 4, 4, 1)),
        byte_alignment=16)
    sIdx = smem.allocate_tensor(element_type=cutlass.Int32,
                                layout=cute.make_layout((ntile * R,), stride=(1,)),
                                byte_alignment=16)
    mb = smem.allocate_tensor(element_type=cutlass.Int64,
                              layout=cute.make_layout((NSTG,), stride=(1,)),
                              byte_alignment=8)

    face_a = sFace.iterator.toint()          # Int32 shared-window address
    a_a = mA.iterator.toint()                # Int64 global address
    b_a = mB.iterator.toint()
    pol = l2_evict_first()

    if tidx == 0:                            # each barrier has only 1 arrive (the issuing thread itself)
        for s in cutlass.range_constexpr(NSTG):
            cute.arch.mbarrier_init(mb.iterator + s, 1)
    cute.arch.mbarrier_init_fence()
    # ... read the index table into sIdx, cute.arch.barrier() ...

    def issue_stage(slot, base):
        """One thread: first arm this stage's tx, then issue two bulk copies per row."""
        cute.arch.mbarrier_arrive_and_expect_tx(mb.iterator + slot, TX)
        for p in cutlass.range_constexpr(R):
            rid = sIdx[base + p]
            rid = rid - (rid & (rid >> 31))  # negative index -> row 0: issued and accounted as usual (gotcha 1)
            da = face_a + (slot * R + p) * PITCH_B
            bulk_g2s_hint(da, a_a + cutlass.Int64(rid) * LAT_B, LAT_B,
                          mb.iterator.toint() + slot * 8, pol)
            bulk_g2s_hint(da + LAT_B, b_a + cutlass.Int64(rid) * ROT_B, ROT_B,
                          mb.iterator.toint() + slot * 8, pol)

    if tidx == 0:                            # prologue: warm up S-1 = 1 stage
        issue_stage(0, 0)

    for c in cutlass.range(ntile):
        cur = c % NSTG
        cute.arch.mbarrier_wait(mb.iterator + cur, (c // NSTG) % 2)   # phase = parity of the lap
        if c + 1 < ntile:                    # the next stage goes into the slot drained on the previous lap
            if tidx == 0:
                issue_stage((c + 1) % NSTG, (c + 1) * R)
        # ... the whole CTA consumes sFace[cur] (in the probe it is copied back to gmem verbatim and compared byte by byte) ...
        cute.arch.barrier()                  # this slot can be reused by the next lap only after it is drained
```

Numeric contract (the probe's criterion): 3 tiles run the 2-slot ring around one full lap (both the phase flip and slot reuse are exercised),
the index table has a run of −1 placed in the middle and at the end of a stage, and the staging face is byte-for-byte equal to the numpy gather
`cat([A[idx], B[idx]])` (`max_diff = 0`).

## Gotchas

1. **`expect_tx` must equal exactly the bytes actually issued in this stage, and invalid rows must also be issued and accounted.** Copies arrive at the barrier by
   bytes; issuing one too few is **a hang, not an exception** (the waited phase never flips). The sentinel value in the index table
   (−1 in this entry's certification) cannot be handled by "skip this row" -- that would make tx vary with each stage's number of valid rows;
   the certified form clamps negative indices **branch-free** to a legal row (`r - (r & (r >> 31))`, negative → 0) and
   issues as usual, so `TX` is a compile-time constant. The probe put a run of −1 in the middle and at the end of a stage, byte-exact.
2. **The issuing thread's guard is a warp-uniform `if tidx == 0`, and the barrier's arrive count is 1.**
   Certified form: all copies of a stage (128 in this entry) are issued serially by one thread, `mbarrier_init(bar, 1)`,
   and `arrive.expect_tx` is also done only by this one thread. Wrapping in `elect_one` never returned a verdict in this entry; do not treat it as known.
3. **The DSL copy-atom spelling has not been certified here.** `cutlass.cute.nvgpu.cpasync.CopyBulkG2SOp`
   exists, and its trait's `unpack` signature takes the two kwargs `mbar_ptr` and `cache_policy` (printed verbatim by the probe),
   but one attempt at `cute.make_copy_atom(CopyBulkG2SOp(), Int32)` + `cute.copy(atom, gmem_slice,
   smem_slice, mbar_ptr=…, cache_policy=…)` threw `DSLRuntimeError` in `cute.compile`
   (the error body did not survive the log tail). **The certified path is inline PTX**; the correct DSL atom spelling is `[to be checked]`.
4. **Both barrier spellings passed.** DSL: `cute.arch.mbarrier_init(ptr, 1)` /
   `cute.arch.mbarrier_arrive_and_expect_tx(ptr, tx)` / `cute.arch.mbarrier_wait(ptr, parity)`;
   PTX: `mbarrier.init.shared::cta.b64 [$0], $1;` /
   `mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;`. The barrier operand in the copy instruction is
   `mb.iterator.toint() + slot * 8` (an Int32 smem address); `mbarrier_init_fence()` goes after the init.
5. **The cache-policy operand does not show in the opcode.** With copies both with and without `.L2::cache_hint` in
   one kernel, the census's `cache_ops` reads only one dotted opcode, `UBLKCP.S.G` -- it is the footprint of "this instruction
   was issued", not of "the policy bits took effect". `createpolicy.fractional.L2::evict_first.b64`
   is built once per CTA through an `=l` result-type asm (`has_side_effects=False`), and the data is byte-for-byte unchanged (probe
   `S3-l2-hint`); `evict_last` was not run (`evict_first` passed at the first try). The **value** of the policy is not priced in this entry.
6. **A wrong tx / a predicate of the wrong granularity = a hang, which takes the whole job with it.** The certification probe runs one spelling per subprocess,
   with the parent process timing each spelling, so a hanging spelling only forfeits its own verdict ([`gpu_probe_ladder`](gpu_probe_ladder.md)).
   The previous round ran three spellings in one process, and a single `DSLRuntimeError` took all three contracts with it, with the error body
   not surviving the log tail.
7. **The shared → shared bulk form has a separate certified recipe.** In this probe, one attempt at
   `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes`
   (issued **directly** after the source face was written with a plain st.shared, without `fence.proxy.async`) gave at launch
   `an illegal instruction was encountered`; the cause was not determined. It was later certified separately in the form "source-side fence + CTA barrier,
   destination address and barrier both pointed at the peer via `mapa`, source-face drain via the peer's remote arrive"
   (footprint `UBLKCP.S.S`); for the spelling, tx accounting and exit contract see
   [`cp_async_bulk_s2s_cluster_exchange`](cp_async_bulk_s2s_cluster_exchange.md).
   When a CTA needs a resident smem face of **its own**, it is still another g2s bulk from gmem (attached to its own
   barrier); s2s is for moving data between CTAs within a cluster.
8. **The `rings` axis reading is a codegen fingerprint.** The source has a 2-slot ring and the census reads `stride=0x1,slots=9`;
   consistent with the conclusion of [`pipeline_tma_umma_stage_count`](pipeline_tma_umma_stage_count.md):
   judge the ring depth by `smem_dynamic` and the number of barrier objects, not by the number of `rings` entries.
9. **Byte counts and alignment**: the piece lengths certified in this entry are 1024 B and 128 B, with a destination row pitch of 1168 B (a multiple of 16 B).
   The PTX ISA requires the byte count of `cp.async.bulk` to be a multiple of 16 and the addresses 16 B aligned; the failure modes of other lengths / misalignment
   were not tested here.

### Load-bearing conditions (under which structure this footprint is worth something -- this entry's readings are the side where it is **not**)

Structural conditions delivered together with `cache_ops [] → ["UBLKCP.S.G"]`: a grid of 128 CTAs, 256
threads per CTA, one **independent** gather stream per CTA (one 2-slot ring, 64 rows × 1152 B = 73728 B per stage,
128 bulk copies per stage issued serially by one thread, S−1 = 1 stage in flight), the whole CTA waiting on the same mbarrier,
`warp_active_pct` ≈ 3.3 (one resident wave; the SM is barely filled by concurrent CTAs). Under these conditions:

| Reading | Before | After |
|---|---|---|
| gate window | 0.15613 ms | 0.19545 ms (**+25.18%, slower**) |
| `stall_long_sb` | 16.39 | 1.27 |
| `stall_barrier` | 0.45 | 9.77 |
| `spin_loops` (mbarrier spinning) | 0 | 389 |
| `regs_per_thread` | 60 | 40 |
| `tensor_pct` | 0.54 | 0.67 |
| `smem_dynamic` | 166128 | 165904 |

That is: the long-scoreboard waits of the individual LDGs really did disappear, but **they turned into barrier waits**, and the total duration is longer.
Another same-job pricing on the same tree (device window, noise floor 1.84%) gives the control reading:
changing the same gather into **a per-thread 16 B `cp.async.cg` (LDGSTS), 36 in flight at once per thread,
one `commit_group + wait_all`**, 0.156 → 0.0734 ms. The two readings are not from the same job (the node has a
±15% band), but they point in opposite directions and both far exceed the noise.

What carries over is the way of reading, not the percentage: `UBLKCP.S.G` appearing in `cache_ops` only says that the copy switched instruction family;
whether it bears load depends on **how many bytes are in flight per SM at once and who is waiting** -- the form with one thread issuing 128 copies serially, only one extra
stage in flight, and the whole consumer side waiting on one barrier did not hide the latency on a tree with one resident wave. On another tree
(several CTAs/SM, a deeper ring, issue spread over several warps) this reading will change, and this entry does not vouch for them.

## Sources

- PTX ISA, "Data Movement and Conversion Instructions": `cp.async.bulk` (the non-tensor form,
  `.shared::cluster.global` + `.mbarrier::complete_tx::bytes`, the optional `.L2::cache_hint`
  operand, size and alignment requirements), `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`,
  `createpolicy`  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols: `cutlass.cute.arch.mbarrier_init`,
  `cutlass.cute.arch.mbarrier_init_fence`, `cutlass.cute.arch.mbarrier_arrive_and_expect_tx`,
  `cutlass.cute.arch.mbarrier_wait`, `cutlass.utils.SmemAllocator.allocate_tensor`,
  `Pointer.toint`, `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`;
  `cutlass.cute.nvgpu.cpasync.CopyBulkG2SOp` (exists, spelling uncertified, see gotcha 3).
  For the general rules of the inline-asm channel see [`ptx_escape`](ptx_escape.md).
- Certification probe (the probe is not included in this release) --
  **Status: ✅ GPU-certified**, round 2 of this certification, all four contracts pass in the same round: `A1-bulk-api` (n/a, signature printed
  verbatim), `S1-bulk-issue` 0.0, `S2-ring-3tile` 0.0 (the 2-slot ring runs 3 tiles, one full lap, exercising phase and
  slot reuse; the PTX barrier spelling and the DSL barrier spelling are each exact), `S3-l2-hint` 0.0
  (every copy carrying `.L2::cache_hint`). One spelling per subprocess; fact line: the s2s spelling without a fence gives
  illegal instruction (gotcha 7; for the certified s2s form see [`cp_async_bulk_s2s_cluster_exchange`](cp_async_bulk_s2s_cluster_exchange.md)). In round 1, the three contracts of the DSL copy-atom spelling in a companion supply probe all died together with
  `DSLRuntimeError` (gotcha 3); its read-side contract is unrelated to this entry.
- KEEP certification (contract-axis change): KEEP in an eval job (the job number was not recorded in the status record), axis `cache_ops`
  `[]` → `["UBLKCP.S.G"]`, `rings` `[]` → `["stride=0x1,slots=9"]`, gate 0.15613 → 0.19545 ms;
  the NCU readings of the same KEEP are in the "Load-bearing conditions" table. Control reading: the same tree, the per-thread
  `cp.async` form, 0.156 → 0.0734 ms.
