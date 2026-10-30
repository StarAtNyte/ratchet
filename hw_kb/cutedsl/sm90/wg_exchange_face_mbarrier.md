# wg_exchange_face_mbarrier (exchange face between warpgroups of the same CTA: one full/empty mbarrier pair per stage, write side STSM + fence + arrive, read side direct WGMMA descriptor read, zero CTA-level barriers in the main loop)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is a **composite recipe**:
> it joins the hand-built ring of [`ring_depth_lever`](ring_depth_lever.md), the STSM delivery of
> [`wgmma_cfrag_publish_stsm`](wgmma_cfrag_publish_stsm.md) and the descriptor read of
> [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) into one **functionally split multi-warpgroup chain**
> (WG0 → WG1 → WG2), with **not a single** 384-thread `bar.sync` in the steady-state loop.

## Purpose

Several math warpgroups in one CTA split by **function** (not homogeneous warpgroups split by M), where the
WGMMA accumulator of an upstream WG must become a WGMMA operand of a downstream WG. The handover face is a
16-bit swizzled face in smem; this entry gives that face's **synchronisation protocol**: one full / empty
mbarrier pair per stage, on the write side each thread does `fence + arrive(full)`, on the read side
`wait(full) → direct descriptor read → wgmma wait → arrive(empty)`. Upstream waits only on the empty of the
stage it is about to write, downstream only on the full of the stage it is about to read; the three WGs each
advance at their own pace and their phases stagger naturally.

## When needed (trigger)

- Structural symptom: the per-chunk handover between several math WGs is written as
  `cute.arch.barrier(..., number_of_threads=384)` (or `NamedBarrier(384).arrive_and_wait`) — several
  all-hands rendezvous per chunk lock the three WGs into one step, and nobody can get ahead. In the dossier:
  `mbar_objs` only has the few of the DMA ring, 384 appears among the named-barrier widths, `rings` has no
  inter-WG edge.
- On NCU the triad of [`handoff_sync_choice`](handoff_sync_choice.md) (`tensor_pct` low + `dram_pct` low +
  `stall_barrier` high), and by that entry's SASS-form criterion the rendezvous primitive on the steady-state
  path sits between consumer and consumer rather than between DMA and consumer. **Classification belongs to
  that entry** (triad + SASS criterion, its gotcha 2); this entry only gives the spelling of the protocol and
  does not promise that replacing the rendezvous cashes in the triad's gain — see the next section.
- You already have a DMA→consumer ring like [`ring_depth_lever`](ring_depth_lever.md) /
  [`pipeline_runtime_subring`](pipeline_runtime_subring.md), but do not know how to join a consumer→consumer
  face with the same primitive: **this entry is that instance**.

## Certification boundary [UNVERIFIED-BOUNDARY]

This entry certifies **interface correctness** (the four contracts of the probe in "Sources"), **with no
production price**. The only time it was dropped into a production tree with "several math WGs doing several
CTA-wide rendezvous per chunk", the gate window read **+7.3%** and the main kernel time +15%, while the tree
that later reached the performance bar still kept its CTA-wide rendezvous and never took this protocol back —
so the production value of "removing the CTA-level rendezvous" has never been isolated: it can be read
neither as a negative gain nor as cashing in the triad's gain. Before adopting it, run a **same-job A/B**
(rendezvous version vs this protocol) to decide whether the rendezvous is what drags; after landing it, use
the machine-readable footprint of gotcha 9 to check that the protocol made it into the cubin, attribute the
gate reading separately, and do not take "BAR.SYNC disappeared" as evidence of a speed-up.

## Minimal runnable example

One CTA, 384 threads = three warpgroups, no DMA warp. WG0 builds a 64×64 accumulator and publishes it to
face0, WG1 reads face0 through a descriptor for its WGMMA and publishes the result to face1, WG2 reads face1
for its WGMMA and accumulates across chunks. The full certification probe is not included in this release; only the mechanism skeleton is listed
below.

```python
STAGE = 2                                    # ring depth of each exchange face
TILE = 64

@cute.struct
class SharedStorage:
    # one full / empty pair per face, STAGE Int64s per pair; placed before the 1024 B-aligned faces,
    # they land in the alignment padding and do not enter the smem bill (ring_depth_lever gotcha 2)
    b_f0: cute.struct.MemRange[cutlass.Int64, STAGE]
    b_e0: cute.struct.MemRange[cutlass.Int64, STAGE]
    b_f1: cute.struct.MemRange[cutlass.Int64, STAGE]
    b_e1: cute.struct.MemRange[cutlass.Int64, STAGE]
    face0: cute.struct.Align[cute.struct.MemRange[BF16, TILE * TILE * STAGE], 1024]
    face1: cute.struct.Align[cute.struct.MemRange[BF16, TILE * TILE * STAGE], 1024]
    sB:    cute.struct.Align[cute.struct.MemRange[BF16, TILE * TILE], 1024]

# host: K-major swizzled face + trailing stage mode; r2s = tiled copy C fragment -> STSM x4
E = LayoutEnum.ROW_MAJOR
atom = warpgroup.make_smem_layout_atom(sm90_utils.get_smem_layout_atom(E, BF16, TILE), BF16)
face_layout = cute.tile_to_shape(atom, (TILE, TILE, STAGE), order=(0, 1, 2))
mma = sm90_utils.make_trivial_tiled_mma(BF16, BF16, MM.K, MM.K, F32,
                                        atom_layout_mnk=(1, 1, 1), tiler_mn=(TILE, TILE))
r2s = cute.make_tiled_copy_S(
    sm90_utils.get_smem_store_op(E, elem_ty_d=BF16, elem_ty_acc=F32),
    cute.make_tiled_copy_C_atom(
        cute.make_copy_atom(warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), BF16), mma))

# kernel
def _bar(bt, idx):                            # the idx-th pointer of a (1, STAGE) barrier tensor
    return bt[(None, idx)].iterator

def publish(r2s_, thr_r, sF, stg, acc, slot):  # f32 C fragment -> bf16 -> STSM x4 -> stage slot of the face
    s32 = ALLOC(stg.shape, F32); s16 = ALLOC(stg.shape, BF16)
    tacc = r2s_.retile(acc)
    for v in range(cute.size(s32)):
        s32[v] = tacc[v]
    s16.store(s32.load().to(BF16))
    cute.copy(r2s_, s16, thr_r.partition_D(sF[(None, None, slot)]))

def chain(mma, thr, acc, sA_t, sB_t, nk, accumulate):   # WGMMA SS: fence -> gemm xnk -> commit -> wait_group(0)
    at = cute.make_mma_atom(mma.op); at.set(warpgroup.Field.ACCUMULATE, accumulate)
    fa = thr.make_fragment_A(thr.partition_A(sA_t))     # descriptor built on the face's stage view
    fb = thr.make_fragment_B(thr.partition_B(sB_t))
    warpgroup.fence()
    for kb in range(nk):
        cute.gemm(at, acc, fa[None, None, kb], fb[None, None, kb], acc)
        at.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)

wid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
tw = tidx % 128
sF0 = storage.face0.get_tensor(face_layout.outer, swizzle=face_layout.inner)
sF1 = storage.face1.get_tensor(face_layout.outer, swizzle=face_layout.inner)
bl = cute.make_layout((1, STAGE))
bF0, bE0 = storage.b_f0.get_tensor(bl), storage.b_e0.get_tensor(bl)
bF1, bE1 = storage.b_f1.get_tensor(bl), storage.b_e1.get_tensor(bl)

# prologue: single-thread init (full/empty counts both = 128 = all threads of that side's warpgroup)
if wid == 0:
    with cute.arch.elect_one():
        for s in cutlass.range_constexpr(STAGE):
            cute.arch.mbarrier_init(_bar(bF0, s), 128); cute.arch.mbarrier_init(_bar(bE0, s), 128)
            cute.arch.mbarrier_init(_bar(bF1, s), 128); cute.arch.mbarrier_init(_bar(bE1, s), 128)
cute.arch.mbarrier_init_fence()
...                                            # one-time operand fill
cute.arch.fence_view_async_shared()
cute.arch.sync_threads()                      # the only CTA-level rendezvous in the whole kernel

thr, thr_r = mma.get_slice(tw), r2s.get_slice(tw)
stg = cute.make_layout(cute.shape(thr_r.partition_S(sF0[(None, None, 0)]))[:3])
acc = ALLOC(thr.partition_C(cute.make_identity_tensor((TILE, TILE))).shape, F32)

if wid < 4:                                   # WG0: write side of face0
    for i in cutlass.range(n_chunk):
        s = i % STAGE
        p = (i // STAGE) % 2
        ...                                    # compute this chunk's acc
        cute.arch.mbarrier_wait(_bar(bE0, s), 1 - p)   # the write side waits on empty with the opposite phase; a new barrier passes phase 1 immediately
        publish(r2s, thr_r, sF0, stg, acc, s)
        cute.arch.fence_view_async_shared()             # generic-proxy STSM -> async-proxy wgmma
        cute.arch.mbarrier_arrive(_bar(bF0, s))         # every thread arrives on its own (128)
elif wid < 8:                                 # WG1: read side of face0 + write side of face1
    for i in cutlass.range(n_chunk):
        s = i % STAGE
        p = (i // STAGE) % 2
        cute.arch.mbarrier_wait(_bar(bF0, s), p)
        chain(mma, thr, acc, sF0[(None, None, s)], sB, 4, False)   # direct descriptor read of face0[s]
        cute.arch.mbarrier_arrive(_bar(bE0, s))         # chain already did wait_group(0): release the slot only after reading
        cute.arch.mbarrier_wait(_bar(bE1, s), 1 - p)
        publish(r2s, thr_r, sF1, stg, acc, s)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(_bar(bF1, s))
else:                                         # WG2: read side of face1, accumulating across chunks
    acc.fill(0.0)                             # ACCUMULATE is a trace-time attribute of the atom; no runtime flag can be passed
    for i in cutlass.range(n_chunk):
        s = i % STAGE
        p = (i // STAGE) % 2
        cute.arch.mbarrier_wait(_bar(bF1, s), p)
        chain(mma, thr, acc, sF1[(None, None, s)], sB, 4, True)
        cute.arch.mbarrier_arrive(_bar(bE1, s))
```

Each exchange face is just these five steps: `wait(empty, 1-p) → write → fence → arrive(full)` /
`wait(full, p) → read → wgmma wait → arrive(empty)`. When one WG is both the read side of one face and the
write side of the next, order the two groups of five by data dependency (release the upstream slot first,
then wait on the downstream slot). If a WG still needs inter-segment sync internally, use a 128-thread named
barrier; **its width must not be the CTA thread count**.

## Gotchas

1. **The arrive count = all threads of that side's warpgroup (128), counted separately for each side.**
   The full count is the write-side thread count, the empty count is the read-side thread count; each
   thread arrives after **its own** `fence_view_async_shared()`, and once the read side passes `wait(full)`
   it sees every warp's STSM — stronger and cheaper than a 384-thread rendezvous. Writing 1 (intending a
   single-thread arrive) while all threads arrive → the phase flips early, wrong data; writing 384 (keeping
   the rendezvous width) → never fills → deadlock.
2. **The write side waits on empty with `1 - p`, the read side waits on full with `p`**
   (`p = (i // STAGE) % 2`). A freshly initialised mbarrier passes `try_wait.parity` on phase 1 immediately,
   so the write side does not wait for its first STAGE chunks; getting it backwards deadlocks on the first
   chunk, it does not compute wrong. Do not flip the phase with bit operations; `1 - p` is enough.
3. **The read side's arrive(empty) must come after `wgmma.wait_group`.** WGMMA is an asynchronous proxy;
   when `cute.gemm` is issued the descriptor has not been read yet; the slot can be released only after
   `commit_group` + `wait_group(0)`. When the wait is deferred to the next chunk
   ([`wgmma_deferred_wait`](wgmma_deferred_wait.md)), the empty arrive must be deferred together with that wait.
4. **The write side's `fence_view_async_shared()` must be issued by every thread, before the arrive.** STSM
   goes through the generic proxy, the wgmma descriptor through the async proxy; without the fence you
   silently read stale bytes (wrong numbers, no crash, and not reliably reproducible).
5. **When one WG is the read side of one face and the write side of another, release upstream first, then
   wait downstream.** The order `arrive(empty_up) → wait(empty_down)` gives the upstream WG its slot back as
   early as possible; the reverse (waiting downstream first) passes downstream back-pressure to upstream and
   shortens the pipeline depth.
6. **Certification shape: the chunk count is not a multiple of STAGE, and the ring wraps more than once**
   (the probe uses 9 chunks / 2 stages). With one lap or a whole number of laps the phases stay in lockstep
   forever and a phase error cannot be detected (gotcha 5 of
   [`pipeline_runtime_subring`](pipeline_runtime_subring.md), gotcha 4 of [`ring_depth_lever`](ring_depth_lever.md)).
7. **The face's stage index is a runtime value and must be passed in as the trailing mode of the face
   layout**: `sF[(None, None, s)]`, with the descriptor built on this slice; do not `flat_divide` out a
   single-stage face and then `align` it (gotcha 1 of [`tri_inverse_block_doubling`](tri_inverse_block_doubling.md)).
8. **Every mismatch shows up as a deadlock or silently wrong numbers, not a compile error.** Count / phase /
   order errors are not checked by the DSL — every change needs a GPU probe; the probe's parent process must
   have a timeout and report "hang" as a contract (D in this entry's probe).
9. **Machine readings**: the main-loop SASS should be `SYNCS.PHASECHK` (wait) + `SYNCS.ARRIVE` (arrive) +
   `HGMMA` + `STSM`, with `BAR.SYNC` only in the prologue; in the dossier `mbar_objs` = the sum of 2×STAGE
   per face + the DMA ring, and every named-barrier width ≤ 128. That is the formal difference between this
   entry and "three WGs doing `bar.sync 384` together".
10. **Only one data direction on the face avoids a transpose**: this entry's write side
    `StMatrix8x8x16bOp(False, 4)` lands a K-major face that the read side uses as a K-major A operand; to
    read it as an MN-major operand (a transposed multiplication) change the write-side member with
    [`stsm_transpose_exchange`](stsm_transpose_exchange.md); the protocol itself is unchanged.

## Sources

- **PTX ISA**: phase semantics of `mbarrier.init` / `mbarrier.arrive` / `mbarrier.try_wait.parity`, the
  generic→async proxy ordering of `fence.proxy.async`, `wgmma.commit_group` / `wgmma.wait_group`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier>
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions>
- **CuTe DSL public API symbols**: `cute.arch.mbarrier_init` / `mbarrier_init_fence` / `mbarrier_wait` /
  `mbarrier_arrive` / `fence_view_async_shared` / `elect_one` / `make_warp_uniform`,
  `cute.struct.MemRange` / `cute.struct.Align`, `MemRange.get_tensor(layout, swizzle=)`,
  `cutlass.utils.hopper_helpers.make_trivial_tiled_mma` / `get_smem_layout_atom` / `get_smem_store_op`,
  `warpgroup.make_smem_layout_atom` / `fence` / `commit_group` / `wait_group` / `Field.ACCUMULATE`,
  `warp.StMatrix8x8x16bOp`, `cute.make_tiled_copy_S` / `make_tiled_copy_C_atom` / `make_mma_atom` / `gemm`.
- **This repo's GPU-certified probe** — four contracts: A (WG2 output vs the
  f64 reference, per element `max_diff=0.0`, two hops, 9 chunks / 2 stages), B (SASS `BAR.*` ≤ 4, prologue
  only), C (`SYNCS.PHASECHK` / `SYNCS.ARRIVE` / `HGMMA` / `STSM` each ≥ 1), D (no hang in the subprocess).
  **Status: ✅ Verified** (`PROBE PASS max_diff=0.0`, round 1 of this certification: all four
  contracts A/B/C/D passed); **production price [UNVERIFIED-BOUNDARY]**, see "Certification boundary".
- Related entries: [`handoff_sync_choice`](handoff_sync_choice.md) (choosing), [`ring_depth_lever`](ring_depth_lever.md)
  (ring depth), [`wgmma_cfrag_publish_stsm`](wgmma_cfrag_publish_stsm.md) (write-side path),
  [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) (read-side descriptor), [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md)
  (how the wait is encoded), [`warp_specialization`](warp_specialization.md).
