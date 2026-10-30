# wgmma_cfrag_publish_stsm (WGMMA C-frag 64×N → elementwise segment → bf16 pair packing → STSM ×4 published to a K-major handover face)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry is the complete publish
> path of a **transposed handover face**: the WGMMA fp32 accumulator (64×N, one warpgroup), after an
> elementwise segment (scaling, subtracting/multiplying broadcast-axis scalars, exp2), is packed as bf16 pairs
> and stored with `make_tiled_copy_S(get_smem_store_op(ROW_MAJOR), make_tiled_copy_C_atom(StMatrix8x8x16bOp(False, 4)))`
> ×4 into a **K_SW128** face (N contiguous); downstream consumes it as an **MN-major** operand through a
> `cute.composition` transposed read view (K = the upstream M = 64), or reads it with a native K-major
> descriptor.
>
> Division of labour: [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md) teaches the instruction
> form of the elementwise segment, [`stsm_transpose_exchange`](stsm_transpose_exchange.md) teaches how to
> choose the transposing member, mechanism 2b of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)
> teaches register residency of broadcast-axis scalars. This entry **strings the three into one path** and
> gives the machine-readable self-check before delivery: **packing + STSM landing + scalar register residency
> + self-check**.

## Purpose

Publish a WGMMA's fp32 C-frag (64×N, one warpgroup), after elementwise processing, to an smem handover face
for a downstream MMA to read as an operand, while meeting three constraints:

- **Mechanism A (STSM ×4)**: exactly 4 `STSM.16.M88.4` per warpgroup per 64×64 epi-tile (32 bf16 per thread
  / 8 per x4 store), not degrading into per-element scalar STS (NCU/census symptom:
  `scalar_sts_in_loop ≈ element count`, STSM missing from `cache_ops`, `smem_wavefronts_st` and
  `bankconf_st` over the line together).
- **Mechanism B (register residency of broadcast-axis scalars)**: the scalar vectors broadcast along the N
  axis that the elementwise segment consumes (lse / delta and the like, `NSTAT` of them) stay in registers
  throughout: one staging per warp (`NSTAT × JMAX` 64-bit LDS) followed by `shfl.sync.idx` broadcasts, **no
  per-element LDS re-reads inside the loop body** (NCU symptom: `smem_wavefronts_ld ≥ NSTAT × NELEM/4`,
  pinned at the smem-gather lower bound).
- **Mechanism C (native K_SW128 allocation)**: the face is allocated with **the upstream N mode contiguous**
  (`LayoutEnum.ROW_MAJOR` drives four sites: store-op member / C-atom / smem atom family / `order=(0,1)`), and
  downstream swaps the logical mode order with a transposed read view — do not do the reverse, writing with
  the transposing member and having downstream read K-major (that pays a wavefront volume tax, see gotcha 9
  of [`stsm_transpose_exchange`](stsm_transpose_exchange.md)).

## When needed (trigger)

- The handover-face ledger has ruled "a real smem transpose", and downstream reads through an MMA descriptor
  (a 64×N accumulator such as softmax probabilities / their gradient must become the A or B operand of a
  downstream chain).
- Symptoms (the three can appear independently, and often appear together):
  - census `scalar_sts_in_loop ≈ element count`, no STSM in `cache_ops` (mechanism A missing)
  - volume-type `smem_wavefronts_ld` over the line, `wpwi ≥ NSTAT × NELEM/4` (mechanism B missing)
  - `smem_wavefronts_st` over the line while `bankconf_st` is low instead (mechanism C missing, running
    against the swizzle grain)
- Build-time symptom: `AttributeError: module 'cutlass.cute' has no attribute 'make_fragment'`
  — see gotcha 0; do not change it to `make_fragment_like` as the error message suggests.

## Minimal runnable example

A single warpgroup (128 threads), one 64×64 epi-tile, two scalar vectors broadcast along N (`NSTAT = 2`).
The full runnable version (including the SASS self-check) is the certification probe (the probe is not
included in this release).

### 0) Allocating a register tensor by shape: the name drifts with the DSL version, so probe for it first

```python
def _rmem_alloc():
    """cute.make_fragment(shape, dtype) has been removed in some DSL versions; the new name for
    allocating by shape is cute.make_rmem_tensor. The make_fragment_like suggested by the error message
    "mimics an existing tensor" and is not a replacement."""
    import inspect
    fn = (getattr(cute, "make_rmem_tensor", None)
          or getattr(cute, "make_fragment", None))
    first = list(inspect.signature(fn).parameters)[0].lower()
    dtype_first = first in ("dtype", "element_type", "elem_type", "ty", "t")
    return (lambda shape, dtype: fn(dtype, shape)) if dtype_first \
        else (lambda shape, dtype: fn(shape, dtype))

ALLOC = _rmem_alloc()
```

### 1) Host side: native K_SW128 face allocation + three-layer r2s construction + scalar slab

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute import nvgpu
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum

bf, f32 = cutlass.BFloat16, cutlass.Float32
M, N = 64, 64          # C-frag (M, N); M = 64 fixed (WGMMA atom)
NSTAT = 2              # number of scalar vectors broadcast along N
NPAIR = N // 2         # number of bf16 pairs
JMAX = NPAIR // 32     # pair slots resident per lane (must divide evenly)

@cute.jit
def launch(mOut, mStat, st):
    mma = sm90_utils.make_trivial_tiled_mma(
        bf, bf, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        f32, atom_layout_mnk=(1, 1, 1), tiler_mn=(M, N))

    # one LayoutEnum drives four sites: ROW_MAJOR => non-transposing member / K-family atom / N is the contiguous mode / order=(0,1)
    E = LayoutEnum.ROW_MAJOR
    contig = M if E.is_m_major_c() else N
    face_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(E, bf, contig), bf)
    face_layout = cute.tile_to_shape(
        face_atom, (M, N), order=(1, 0) if E.is_m_major_c() else (0, 1))

    store_op = sm90_utils.get_smem_store_op(E, elem_ty_d=bf, elem_ty_acc=f32)
    atom_c = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), bf)
    r2s = cute.make_tiled_copy_S(store_op, cute.make_tiled_copy_C_atom(atom_c, mma))

    # scalar slab: (NSTAT, N) flat row-major, **no swizzle** (mechanism B's pair view
    # is raw pointer arithmetic over these bytes, see gotcha 5)
    stat_layout = cute.make_layout((NSTAT, N), stride=(N, 1))

    # 128-bit universal copy (fills the scalar slab gmem->smem; the same atom can also do STS.128 zero-fill/prefill)
    tiled_sf = cute.make_tiled_copy_tv(
        cute.make_copy_atom(nvgpu.CopyUniversalOp(), f32, num_bits_per_copy=128),
        cute.make_layout(16), cute.make_layout(N // 16))

    SS = _storage(cute.cosize(face_layout), NSTAT * N)   # @cute.struct built inside jit
    kern(mOut, mStat, mma, face_layout, stat_layout, r2s, tiled_sf, SS).launch(
        grid=[1, 1, 1], block=[128, 1, 1], stream=st)
```

### 2) Kernel side: scalar staging → elementwise segment (shfl broadcast) → packing → STSM → fence

```python
@cute.kernel
def kern(mOut, mStat, mma, face_layout, stat_layout, r2s, tiled_sf,
         SharedStorage: cutlass.Constexpr[Callable]):
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    face = storage.face.get_tensor(face_layout.outer, swizzle=face_layout.inner)
    sStat = storage.stat.get_tensor(stat_layout)

    # scalar slab gmem->smem: one LDG.128 + STS.128 per thread (not per-element STS)
    thr_sf = tiled_sf.get_slice(tidx % 16)
    for ns in cutlass.range_constexpr(NSTAT):
        cute.copy(tiled_sf, thr_sf.partition_S(mStat[(ns, None)]),
                  thr_sf.partition_D(sStat[(ns, None)]))
    cute.arch.sync_threads()

    # --- mechanism B: one staging per warp, NSTAT x JMAX LDS.64, resident in registers -------
    sPair = cute.make_tensor(sStat.iterator,
                             cute.make_layout((NSTAT, 2, NPAIR), stride=(N, 1, 2)))
    rStat = ALLOC((NSTAT, 2, JMAX), f32)          # [stat, parity (even/odd N), slot]
    for ns in cutlass.range_constexpr(NSTAT):
        for j in cutlass.range_constexpr(JMAX):
            cute.autovec_copy(sPair[(ns, None, 32 * j + lane)], rStat[(ns, None, j)])

    # --- C-frag (faked here with static identity coordinates; in a real kernel it is the WGMMA accumulator) ------
    cS = mma.get_slice(tidx).partition_C(cute.make_identity_tensor((M, N)))
    acc = ALLOC(cS.shape, f32)
    for i in cutlass.range_constexpr(cute.size(cS)):
        m, n = cS[i]
        acc[i] = f32(2 * m)

    # --- elementwise segment: each 4-element group = one N pair x two rows; scalars via shfl, not LDS -------
    for v in cutlass.range_constexpr(cute.size(acc) // 4):
        q = cS[4 * v][1] // 2          # runtime pair index (the four elements of a group share one pair)
        j = (4 * v) // 32              # compile-time constant: register slot (independent of lane)
        src = q % 32                   # runtime, differs per lane: feeds the shfl source lane, legal
        s00 = cute.arch.shuffle_sync(rStat[(0, 0, j)], src)   # stat0 @ even n
        s01 = cute.arch.shuffle_sync(rStat[(0, 1, j)], src)   # stat0 @ odd n
        s10 = cute.arch.shuffle_sync(rStat[(1, 0, j)], src)   # stat1 @ even n
        s11 = cute.arch.shuffle_sync(rStat[(1, 1, j)], src)   # stat1 @ odd n
        for sub in cutlass.range_constexpr(4):
            i = 4 * v + sub
            s0 = s00 if sub % 2 == 0 else s01   # trace-time ternary choice, no if statement (gotcha 6)
            s1 = s10 if sub % 2 == 0 else s11
            x = acc[i] * f32(0.5) - s0           # e.g.: scale*qk - lse
            acc[i] = cute.math.exp2(x, fastmath=True) * s1

    # --- mechanism A: f32->bf16 packing + STSM x4 ----------------------------------------
    thr = r2s.get_slice(tidx)
    tRS_sD = thr.partition_D(face)
    tRS_acc = r2s.retile(acc)
    stg_layout = cute.make_layout(cute.shape(thr.partition_S(face))[:3])
    stg32 = ALLOC(stg_layout.shape, f32)
    stg16 = ALLOC(stg_layout.shape, bf)
    for v in cutlass.range_constexpr(cute.size(stg32)):
        stg32[v] = tRS_acc[v]
    stg16.store(stg32.load().to(bf))              # cvt packing
    cute.copy(r2s, stg16, tRS_sD)                 # <- 4 STSM.16.M88.4

    # generic proxy (STSM) -> async proxy (MMA descriptor): both fence and rendezvous are required
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()        # for a cross-warpgroup handover, replace with a barrier_arrive / barrier pair
```

For the downstream read side (a second `get_tensor` on the same `MemRange` giving a transposed view,
`make_fragment_A/B` descriptors) see §3-§4 of [`stsm_transpose_exchange`](stsm_transpose_exchange.md) and
[`wgmma_mn_major_operand`](wgmma_mn_major_operand.md); not repeated here.

### 3) Build self-check (machine-readable, mandatory before delivery; if it fails, nothing has been delivered)

Dump path: in the **subprocess** environment set `CUTE_DSL_DUMP_DIR` / `CUTE_DSL_KEEP_CUBIN=1` /
`CUTE_DSL_KEEP_PTX=1` / a separate `CUTE_DSL_CACHE_DIR` (so a JIT cache hit does not skip the dump), with
`cute.compile(..., options="--dump-dir=… --keep-cubin --keep-ptx")` as a second safeguard, then
`cuobjdump -sass` that cubin. **Classify by opcode family, not by literal regex** — a bare `STS` (32-bit)
has no `.U32` suffix, and a regex like `STS\.[UF]` misses it.

```python
ops = [opcode(line) for line in sass]           # first token after stripping /*addr*/ and the @P predicate
fam = lambda op: op.split(".")[0]
n_stsm   = sum(1 for op in ops if op == "STSM.16.M88.4")
n_scalar = sum(1 for op in ops if fam(op) == "STS" and op != "STS.128")
n_shfl   = sum(1 for op in ops if fam(op) == "SHFL")
assert n_stsm == 4 * (tiles_per_wg)                  # 1. STSM count = ledger count
assert n_scalar == 0                                  # 2. no scalar/narrow STS in the loop body
assert n_shfl >= 2 * NSTAT * (NELEM_T // 4)           # 3. broadcast footprint of mechanism B (NELEM_T = elements per thread)
# 4. orch census: cache_ops contains STSM (and UTMALDG/UTMASTG on the other side of the handover face), mbar_objs agrees with the ledger
```

Read all four together: **correct numbers ≠ correct mechanism** (the transposing and non-transposing members
give cell-for-cell identical numbers; only the footprint and wavefronts tell them apart).

## Gotchas

0. **`cute.make_fragment` does not exist in this DSL version, and the `make_fragment_like` in the error
   message is misleading.** Allocate by shape with `cute.make_rmem_tensor` (the probing spelling of §0 covers
   both names at once). Same as gotcha 4 of [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md).
1. **Broadcast-axis scalars of the elementwise segment must be register-resident.** Writing `sStat[n]` as a
   per-element LDS pins you at the smem-gather lower bound (`wpwi ≥ NSTAT × NELEM/4`), which ptxas cannot
   merge away. Only mechanism B's staging + `shfl.sync.idx` can break through that bound.
2. **The broadcast is two shfls, "per stat × per parity".** A 4-element group covers two rows of the same N
   pair: even `sub` takes the even-N scalar, odd takes the odd-N scalar. Shuffling only parity 0 computes the
   odd-N elements wrong, which the numbers catch immediately; the `2 × NSTAT` SHFLs per group are also where
   self-check item 3 comes from.
3. **`j = (4v) // 32` is a compile-time constant, `src = q % 32` is a runtime value.** The former indexes the
   register array (must be a compile-time constant, otherwise it lands in local memory), the latter feeds the
   shfl source lane (legal at run time) — the invariant of §2b.3 of
   [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md), which depends on the linear order of the
   C-frag; re-verify when changing the tiler/atom layout.
4. **`order=(0,1)` and `ROW_MAJOR` must come from the same source.** The face is allocated N-contiguous and
   downstream swaps the mode order with a transposed read view; getting it backwards gives a layout that is
   neither MN nor K, shape-legal, caught only by the numbers or the volume
   (gotcha 3 of [`stsm_transpose_exchange`](stsm_transpose_exchange.md)).
5. **The scalar slab must not be swizzled.** The pair view `make_tensor(sStat.iterator, ((NSTAT,2,NPAIR),(N,1,2)))`
   is raw pointer arithmetic; if the slab is built as an SW128 layout with `get_smem_layout_atom`, the view
   reads scrambled bytes, and the 64-bit `autovec_copy` requires the source to be truly contiguous (mode 0
   stride must be 1).
6. **Compile-time choices inside a `range_constexpr` body use ternary expressions, not `if` statements.** In
   DSL ≥ 4.1 an `if` is translated as a runtime branch by default;
   `s0 = s00 if sub % 2 == 0 else s01` is evaluated at trace time and produces no instruction
   ([`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)).
7. **Both `fence_view_async_shared()` and a rendezvous are required.** A generic store (STSM) → async-proxy
   read (MMA descriptor) needs both; missing either gives 0 or garbage
   (gotcha 1 of [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)). For a cross-warpgroup handover the rendezvous
   is a `barrier_arrive` / `barrier` pair, and both the STSM and the fence must precede the arrive
   ([`softmax_segment_ilp`](softmax_segment_ilp.md), "STS before arrive").
8. **The `SharedStorage` annotation must be `cutlass.Constexpr[Callable]`, and the `@cute.struct` is built in
   a helper inside jit** (gotchas 0/1 of [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md)).
   The file must not contain `from __future__ import annotations`.
9. **Two call sites, one enum.** `get_smem_store_op(E, …)` issues the instruction,
   `StMatrix8x8x16bOp(E.is_m_major_c(), 4)` governs the register grouping; changing only one leaves the SASS
   unchanged or makes `cute.copy` report a value-count mismatch (gotcha 2 of
   [`stsm_transpose_exchange`](stsm_transpose_exchange.md)).
10. **The §3 self-check is mandatory before delivery.** If the ledger says STSM ×4 and the census counts
    `scalar_sts_in_loop > 0`, that is "build not delivered", not a contract defect; do not price that gap
    against wait depth or the commit axis.

## Out of scope for this entry

- The instruction form of the elementwise segment itself (fastmath exp2, cvt packing count, MN view) —
  [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md).
- The criterion for choosing the transposing member (when to use `MT88`) —
  [`stsm_transpose_exchange`](stsm_transpose_exchange.md). This entry pins the non-transposing member + a
  transposed read view.
- Register budget and live ranges — mechanism 3 of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md).
- TMA loading/write-back on the other side of the handover face (`UTMALDG` / `UTMASTG` + mbarrier) —
  [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md), [`tma_store_epilogue`](tma_store_epilogue.md).

## Sources

- Public infra (nvidia-cutlass-dsl 4.4.2, `python/CuTeDSL/cutlass/`):
  - `utils/hopper_helpers.py` `get_smem_store_op` L44-85, `get_smem_layout_atom` L173-212,
    `make_smem_layout_epi` L329-397, `make_trivial_tiled_mma`
  - `cute/nvgpu/warp/copy.py` `StMatrix8x8x16bOp` L228-256
  - `cute/atom.py` `make_tiled_copy_S` L996-1014, `make_tiled_copy_C_atom` L1040-1074
  - `cute/arch/nvvm_wrappers.py` `shuffle_sync_op` (`cute.arch.shuffle_sync`)
  - `cute/core.py` `make_rmem_tensor` / `autovec_copy` / `make_tiled_copy_tv`
- GPU-certified KB entries of the same tree (skeleton and sub-mechanisms):
  §2/§4 of [`stsm_transpose_exchange`](stsm_transpose_exchange.md),
  mechanism 2b of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md),
  §(c) of [`cfrag_elementwise_fastpath`](cfrag_elementwise_fastpath.md),
  [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md),
  gotcha 4 of [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md),
  [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md).
- GPU-certified by the certification probe — **✅ Verified: `PROBE PASS max_diff=0.0`** (certification round 2 of this entry)
  (all four contracts passed: A numbers correct cell for cell (exp2/broadcast/packing/STSM end to end,
  `max_diff` = maximum relative error); B `STSM.16.M88.4 == 4`; C the STS family other than `STS.128` is 0;
  D `SHFL ≥ 2×NSTAT×NELEM_T/4`; B/C/D are `max_diff=n/a`).
