# stsm_transpose_exchange (transposing stmatrix on the write side → K-major descriptor on the read side: who decides the contiguous mode of a C-frag handover face)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry covers how to choose between the
> **two members** (transposing / non-transposing) of `cute.nvgpu.warp.StMatrix8x8x16bOp`, and how that choice
> decides **which mode is contiguous** once the C tile lands in smem — and therefore whether the downstream MMA gets
> a **native K-major descriptor** or an MN-major transposed read view.
>
> Division of labour: §3 of [`tma_store_epilogue`](tma_store_epilogue.md) teaches the **skeleton** of the three-layer
> r2s construction (that entry's destination is gmem, the orientation is given by the output tensor, and it never needs
> to choose a member); [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md) teaches how to **swap the logical mode
> order for free** (`cute.composition`). This entry fills the gap between the two: **the contiguous mode is a physical
> property; composition cannot change it, it can only be decided at allocation + member choice**. Only the three
> together form a complete handover-face path.

## Purpose

An smem handover face is produced by one MMA in C-frag `(M, N)` form and must be consumed as an operand by a
downstream MMA, where **the downstream K mode = the upstream M mode** (a "transposed handover" in math notation). This
entry gives the exact spelling for putting the transpose on the **write side**: use the transposing member of
`stmatrix` to land a face with **the M mode contiguous**, so that downstream reads an ordinary **K-major** operand —
there is no transposed read view anywhere on the path.

The core fact (counter-intuitive, and the reason this entry exists):

> **The transposing member does not transpose your logical matrix.** `face[m, n]` reads back **exactly the same
> values** under both members. All it changes is **which mode is contiguous in smem** (and with it the byte order and
> the major mode of the downstream descriptor).

So "I already transposed once in the read view; using MT88 on the write side as well transposes twice and cancels
out" is wrong — the two are not the same degree of freedom: the read view changes the **logical mode order** (free,
reversible), the member changes the **physical contiguous mode** (decided at allocation, composition cannot move it).

## When needed (trigger)

- The handover-face ledger ruled "a real smem transpose", and that buffer's **read side is a direct MMA descriptor
  read** (no LDS).
- **Symptom (this entry's fingerprint)**: the transpose is stuffed into an MN-major read view via
  `cute.composition`; the numbers are all right, `max_diff == 0`, the probe all green, but on NCU
  `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{st,ld}` is clearly above the reference while
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` **goes down instead**.
  The combination **zero conflicts + high volume** = few useful bytes per wavefront = the access runs against the
  swizzle grain. This is not a fence / accumulate / bijection bug; do not look there.
- Reverse symptom: out of fear of "transposing twice", the pinned transposing member is switched back to the
  non-transposing member and the transpose is patched in through the read view. That is exactly the step this entry
  is meant to stop (see the end of "Purpose").
- Build-time symptom: you changed only the flag of `StMatrix8x8x16bOp` and find the instruction in SASS **did not
  change** (see gotcha 2), or after changing the flag `cute.copy` reports a per-thread value-count mismatch.

## Mechanism: one `LayoutEnum` drives four sites; missing one makes it inconsistent

CuTe DSL gathers all orientation decisions of this path into **one** `LayoutEnum`. In the official epilogue path
(`hopper_helpers.py`) these four sites must come from the same enum value:

| # | Decides what | Official call | Relation to the enum |
|---|---|---|---|
| 1 | **which instruction is issued** (transposing member or not) | `get_smem_store_op(E, elem_ty_d, elem_ty_acc)` | for 16-bit output always returns `StMatrix8x8x16bOp(E.is_m_major_c(), 4)` (CuTeDSL `hopper_helpers.py` L78-83) |
| 2 | **register-side value grouping** | `StMatrix8x8x16bOp(E.is_m_major_c(), 4)` → `make_tiled_copy_C_atom(·, mma)` | the same `E.is_m_major_c()` |
| 3 | **atom family of the destination layout** + **element count of the contiguous mode** | `get_smem_layout_atom(E, dtype, <contiguous-mode extent>)` | `E.sm90_mma_major_mode()` → COL_MAJOR=MN family / ROW_MAJOR=K family (CuTeDSL `hopper_helpers.py` L198-212); extent = `epi_tile[0] if E.is_m_major_c() else epi_tile[1]` (CuTeDSL L367) |
| 4 | **the `order` of `tile_to_shape`** | `tile_to_shape(atom, shape, order)` | `(1,0,…) if E.is_m_major_c() else (0,1,…)` (CuTeDSL `hopper_helpers.py` L384-390) |

`LayoutEnum.COL_MAJOR.is_m_major_c() == True`, `ROW_MAJOR == False`
(CuTeDSL `utils/layout.py` L49-53). Hence:

```
E = COL_MAJOR  ->  StMatrix8x8x16bOp(True,  4)  ->  PTX  stmatrix…x4.trans…  ->  SASS  STSM.16.MT88.4
                  + MN_SW* atom + order=(1,0)  ->  the face's M mode is contiguous
E = ROW_MAJOR  ->  StMatrix8x8x16bOp(False, 4)  ->  PTX  stmatrix…x4 (no .trans) ->  SASS  STSM.16.M88.4
                  + K_SW*  atom + order=(0,1)  ->  the face's N mode is contiguous
```

**MN_SW\* and K_SW\* are two different physical atoms, not two views of the same bytes.**
CuTeDSL `warpgroup/helpers.py` L43-85: the two families use **the same swizzle**
(SW128 = `make_swizzle(3,4,3)`, SW64 = `(2,4,3)`, SW32 = `(1,4,3)`, INTER = `(0,4,3)`),
but their base layouts are transposes of each other —

```
MN family:  make_layout((C, 8), stride=(1, C))     # mode 0 (MN) contiguous
K  family:  make_layout((8, C), stride=(C, 1))     # mode 1 (K) contiguous
C = num_contiguous_bits // dtype.width         # SW128 and f16 -> C = 1024/16 = 64
```

So "transposing write + native read" and "non-transposing write + composed transposed read" land on **different
physical layouts**: the logical values are the same, the bytes differ, and the major mode of the read-side descriptor
differs too. That is why `max_diff` cannot choose between them, while the wavefront counters can.

## Minimal runnable example

Single CTA, one warpgroup (128 threads). Producer C-frag `(M, N) = (64, 64)`; the face is allocated with **the
upstream M mode contiguous**; downstream consumes it as a **K-major A operand** (`K = the upstream M`).
`M` must be 64 (the WGMMA atom's M mode is always 64, see gotcha 0 of
[`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md)).

### 1) Host side: allocating the face — the four orientation sites agree

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum

F16, F32 = cutlass.Float16, cutlass.Float32
M, N = 64, 64                       # (M, N) of the upstream C-frag

@cute.jit                           # all of these are IR builders and must be built inside jit (see gotcha 8)
def launch(mB, mC):
    mma_p = sm90_utils.make_trivial_tiled_mma(
        F16, F16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        F32, atom_layout_mnk=(1, 1, 1), tiler_mn=(M, N))

    # ---- orientation switch: this one value decides member, atom family, extent and order together ----
    E = LayoutEnum.COL_MAJOR        # m-major C => the face's M mode is contiguous => transposing member

    # the third argument is the extent of the **contiguous mode**: M for m-major, N for n-major (CuTeDSL hopper_helpers.py L367)
    contig = M if E.is_m_major_c() else N
    face_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(E, F16, contig), F16)
    #  m-major -> order=(1,0); n-major -> order=(0,1) (CuTeDSL hopper_helpers.py L384-390)
    face_order = (1, 0) if E.is_m_major_c() else (0, 1)
    face_layout = cute.tile_to_shape(face_atom, (M, N), order=face_order)
```

> The four lines above are the expansion of the official `sm90_utils.make_smem_layout_epi(epi_dtype, epi_layout, epi_tile,
> epi_stage, …)` (CuTeDSL `hopper_helpers.py` L329-397) — it `append`s a stage mode and uses
> `order=(1,0,2)/(0,1,2)`. **Use it directly when you can**; this section expands it by hand only to lay out item by
> item "which argument flips with the orientation". With the staged version, on the destination side follow gotcha 2
> of [`tma_store_epilogue`](tma_store_epilogue.md): `partition_D` first, then slice by the stage index.

### 2) Host side: three-layer r2s construction — two call sites, one enum

```python
    # (1) the atom that actually issues the instruction
    store_op = sm90_utils.get_smem_store_op(E, elem_ty_d=F16, elem_ty_acc=F32)
    # (2) register-side value grouping (only grouping, issues no instruction -- see gotcha 2)
    atom_c = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(E.is_m_major_c(), 4), F16)
    # (3) store op as the atom, expanded by the C-atom's tiling
    r2s = cute.make_tiled_copy_S(
        store_op, cute.make_tiled_copy_C_atom(atom_c, mma_p))
```

For 16-bit output `get_smem_store_op` **always** returns the stmatrix family and never degrades into a generic vector
store (the degraded branch only hits when `elem_ty_d.width != 16`, CuTeDSL `hopper_helpers.py` L80-85) — so "is it STSM
at all" is not the worry; "which member" is.

### 3) Host side: the read side — logical mode relabel (free), orientation already fixed by §1

```python
    # the colex linear index of the face's (m, n) domain = m + M*n,
    # so the transposer of the read view (n, m) has stride = (M, 1)
    # (derivation in wgmma_mn_major_operand §3; the shape is the read view's, the strides are the write view's
    #  domain colex coefficients reordered in the read view's coordinate order)
    faceA_layout = cute.composition(
        face_layout, cute.make_layout((N, M), stride=(M, 1)))

    # downstream: A = (N, M), whose K mode = M = the contiguous mode => K-major. B is k-major too.
    # OperandMajorMode.MN appears nowhere.
    mma_c = sm90_utils.make_trivial_tiled_mma(
        F16, F16,
        warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
        F32, atom_layout_mnk=(1, 1, 1), tiler_mn=(N, N2))
```

The `composition` in this step **only renames the logical modes**; it does not change the contiguous mode — that was
fixed in §1. For why it is a byte identity (associativity with the swizzle) see §3 of
[`wgmma_mn_major_operand`](wgmma_mn_major_operand.md); not repeated here.

### 4) Kernel side: write the face → fence → second `get_tensor` on the same `MemRange` → descriptor read

```python
@cute.kernel
def kernel(mB, mC, mma_p, mma_c, face_layout, faceA_layout, sB_layout, r2s,
           SharedStorage: cutlass.Constexpr[Callable]):
    tidx, _, _ = cute.arch.thread_idx()
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    # a ComposedLayout must be split into .outer + swizzle=.inner (wgmma_mn_major_operand gotcha 5)
    face = storage.sX.get_tensor(face_layout.outer, swizzle=face_layout.inner)

    # --- write side: the staging skeleton of tma_store_epilogue §3, unchanged word for word ---
    thr = r2s.get_slice(tidx)
    tRS_sD  = thr.partition_D(face)
    tRS_acc = r2s.retile(acc)                       # acc = upstream C-frag (fp32)
    stg_layout = cute.make_layout(cute.shape(thr.partition_S(face))[:3])
    stg32 = cute.make_fragment(stg_layout.shape, F32)
    stg16 = cute.make_fragment(stg_layout.shape, F16)
    for v in cutlass.range_constexpr(cute.size(stg32)):
        stg32[v] = tRS_acc[v]
    stg16.store(stg32.load().to(F16))
    cute.copy(r2s, stg16, tRS_sD)                   # <- this issues STSM.16.MT88.4

    # --- stmatrix is the generic proxy, the MMA descriptor read is the async proxy: fence + barrier required ---
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    # --- read side: same MemRange, second get_tensor, K-major descriptor ---
    faceA = storage.sX.get_tensor(faceA_layout.outer, swizzle=faceA_layout.inner)
    wg = mma_c.get_slice(0)                         # SS operands are sliced by the warpgroup's first thread
    tSrA = wg.make_fragment_A(wg.partition_A(faceA))
    tSrB = wg.make_fragment_B(wg.partition_B(sB))
    # ... the WGMMA trio (fence / commit_group / wait_group(0)) and the per-k-block loop: see
    #   wgmma_ss_min_gemm §3; not repeated here
```

For faking the upstream C-frag (static identity coordinates, so no dynamic mode is carried into the fragment) see
gotcha (d) of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md).

## Gotchas

1. **The transposing member does not change the logical values.** `face[m, n]` reads back **cell for cell the same**
   under both members (both numerical lines A/B of this entry's certification probe are `0.0`). Therefore:
   - switching members does **not** need a "transpose back" anywhere else;
   - conversely, **switching members cannot fix any numerical bug** — it only moves the byte order and the downstream
     major mode.
   Reasoning about it as "one logical transpose" leads to the wrong conclusion "MT88 + transposed read view =
   transposed twice", and from there to falling back to the non-transposing member and stuffing the transpose into
   the read view, paying the volume tax.
2. **Two call sites, one bijection.** `make_tiled_copy_S(atom, tiled_copy)` uses
   `tiled_copy.layout_src_tv_tiled` (the **source**, i.e. the register-side TV), while **the instruction is issued by
   the `atom` you pass to it** (`cute/atom.py` L996-1014). Consequences:
   - changing only the flag of the `make_tiled_copy_C_atom` atom and not the store op ⇒ **the issued instruction does
     not change at all**; the code looks changed, SASS still has the old member (silently ineffective, the hardest
     kind to track down);
   - changing only the store op and not the C-atom ⇒ the value grouping does not match the register count the
     instruction needs, an error at build time or at `cute.copy`.
   The official form (`examples/python/CuTeDSL/hopper/dense_gemm.py` L931-950) feeds **the same
   `self.c_layout` to both sites** — copy this structure; do not hand-write two independent flags.
3. **`order` must flip with the orientation**: m-major `order=(1,0,…)`, n-major `order=(0,1,…)`
   (CuTeDSL `make_smem_layout_epi` L384-390; likewise on the operand side, `make_smem_layout_b` L317-323).
   Forgetting to flip ⇒ atom family and order fight, giving a layout that is neither MN nor K: **the shape is still
   legal**, only the numbers or the volume can catch it.
4. **The third argument of `get_smem_layout_atom` is the element count of the "contiguous mode", and it changes too
   when the orientation flips** (M for m-major, N for n-major, CuTeDSL `hopper_helpers.py` L367). A wrong value silently selects another swizzle level
   (1024 bit → SW128 / 512 → SW64 / 256 → SW32 / otherwise → INTER), **with no error**,
   hence "zero conflicts but high volume". Same as gotcha 8 of
   [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md): re-check this whenever the tile size changes.
5. **`cute.composition` cannot change the contiguous mode.** It is a pure coordinate permutation (logical mode
   relabel), a byte identity; the contiguous mode is a physical property decided **at allocation** by the atom family
   + `order`. "Making a K_SW\* face M-contiguous through a read view" is impossible — what you get is an **MN-major**
   view describing the same bytes, so the downstream descriptor can only be MN-major. This is the dividing line
   between this entry and [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md): **that one teaches how to swap the
   logical mode order for free (when the orientation is already right); this one teaches who decides the orientation
   itself.**
6. **`num_matrices` can only be 1 / 2 / 4** (CuTeDSL `warp/copy.py` L238-242), and
   `unpack_bits` must be `None` (CuTeDSL `warp/copy.py` L243-244). `get_smem_store_op` always uses 4.
   The `num_matrices` of the two call sites must also agree (as in gotcha 2).
7. **Write side generic proxy → read side async proxy requires `fence_view_async_shared()` + a barrier.**
   The symptom of leaving it out is no error, a result of 0 or garbage (same as gotcha 1 of
   [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)). If the read side is TMA rather than an MMA descriptor, use
   `fence_proxy(async_shared, shared_cta)` + a NamedBarrier instead
   (gotcha 5 of [`tma_store_epilogue`](tma_store_epilogue.md)).
8. **`tile_to_shape` / `composition` / `make_trivial_tiled_mma` / `make_copy_atom` are all IR builders and must be
   called inside `@cute.jit`**; the `SharedStorage` annotation must be `cutlass.Constexpr[Callable]` (same as
   gotchas 0 / 1 of [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md)).
   The file must not contain `from __future__ import annotations`.
9. **Acceptance cannot look at `max_diff` alone (the most important item of this entry).** The two members are
   numerically **cell for cell identical**, so any probe that only checks numbers is **structurally blind** to this
   choice — the same kind of blindness as gotcha 12 of [`tma_store_epilogue`](tma_store_epilogue.md). There are two
   machine-readable criteria, and both must be used:
   - **Which member was issued** (static, cheap, run this first):
     ```python
     compiled = cute.compile(_jit, *args,
                             options=f"--dump-dir={d} --keep-ptx --keep-cubin")
     # count `stmatrix` in d/*.ptx: every one of the transposing member carries `.trans`, the non-transposing member has none;
     # the **counts of the two members should be equal** (the transposing member is not "more stores").
     # The corresponding SASS names: STSM.16.MT88.4 (transposing) / STSM.16.M88.4 (non-transposing).
     nvdisasm -c d/*.cubin | grep -cE 'STSM\.16\.MT88'
     ```
   - **Volume** (dynamic, end to end): `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum`
     `+ …_op_ld.sum`. **Mind the direction**: when this line gets worse, `bank_conflicts` often gets **better** at the
     same time (see the fingerprint in "When needed") — so watching only bankconf reads this degradation as an
     improvement. The absolute threshold is given by the caller's reference; this entry sets no number.
10. **Diagnosis order**: first run a control with "non-transposing member + n-major face" through **the same kernel
    body**, changing only that `LayoutEnum`. Only when the control passes is the scaffolding known to be fine and the
    failure really in the member/orientation; if the control fails, do not go looking at the transposing member (this
    entry's certification probe is arranged exactly this way, with two extra **cross controls** — transposing op with
    a K face, non-transposing op with an MN face — that only print `[diag]` and do not enter the ruling, used to prove
    that the enum really is load-bearing).

## Out of scope for this entry

- **A quantitative model of wavefront volume** (the microarchitectural derivation of "why against-the-grain access has
  fewer useful bytes per wavefront"). This entry only gives the fingerprint and the machine-readable criteria, no
  formula; thresholds are anchored on the caller's measured reference.
- **Members with `num_matrices != 4`**, `StMatrix16x8x8bOp`, and `LdMatrix8x8x16bOp` on the `ldmatrix` side (needed
  only when the read side goes through LDSM rather than an MMA descriptor). Uncertified by this entry's probe; do not
  use them as established fact.
- **The MN-major read side itself** (when the downstream K mode **is** the upstream N mode and the orientation already
  agrees, the read side can of course be MN-major). That path is in
  [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md). This entry only handles handovers where "the orientation
  disagrees".

## Sources

- **Public infra (nvidia-cutlass-dsl 4.4.2, `python/CuTeDSL/cutlass/`)** — the mechanism and exact APIs of this entry
  are all taken line by line from here; no operator-side implementation was read:
  - `utils/hopper_helpers.py`
    - `get_smem_store_op` L44-85 (16-bit ⇒ `StMatrix8x8x16bOp(layout_d.is_m_major_c(), 4)`;
      only non-16-bit degrades to `CopyUniversalOp`); `sm90_get_smem_store_op` is its alias L89
    - `get_smem_layout_atom` L173-212 (splits MN/K families by `sm90_mma_major_mode()`,
      then picks the SW128/64/32/INTER level by `major_mode_size × dtype.width`)
    - `make_smem_layout_epi` L329-397 (**the official prototype of §1 of this entry**: choice of the contiguous-mode
      extent L367, `order` flipping with `is_m_major_c()` L384-390)
    - `make_smem_layout_b` L273-325 (the same `order` rule on the operand side L317-323)
  - `utils/layout.py` `LayoutEnum` L19-53 (`is_m_major_c()` ⇔ COL_MAJOR;
    `sm90_mma_major_mode()` ⇔ ROW_MAJOR→K / COL_MAJOR→MN)
  - `cute/nvgpu/warp/copy.py` `BaseOp` L24-46 (field order `transpose, num_matrices,
    unpack_bits`), `StMatrix8x8x16bOp` L228-256 (`num_matrices ∈ {1,2,4}`;
    `transpose` as a UnitAttr of `CopyAtomStsmType`)
  - `cute/nvgpu/warpgroup/helpers.py` `make_smem_layout_atom` L23-85
    (**the MN family `((C,8),(1,C))` and the K family `((8,C),(C,1))` share the same swizzle** —
    the origin of this entry's "two different physical atoms")
  - `cute/atom.py` `make_tiled_copy_S` L996-1014 (uses `layout_src_tv_tiled`,
    i.e. the origin of "the instruction is issued by the atom passed in" in gotcha 2),
    `make_tiled_copy_C_atom` L1040-1074 (truncates the MMA's C TV layout by the value count of the atom's source
    layout ⇒ the `num_matrices` of the two sites must agree)
  - `base_dsl/compiler.py` L505-537 + L626-630 (the parsing path of `options="--dump-dir=… --keep-ptx
    --keep-cubin"`, the machine-readable path of gotcha 9)
- **Official examples (same API surface; the origin of "copy this structure" in gotcha 2)**:
  `examples/python/CuTeDSL/hopper/dense_gemm.py` L931-950,
  CuTeDSL `dense_gemm_persistent.py` L799-810, `hopper/fmha.py` L1161-1170 — all three feed
  **the same `c_layout` enum to both `get_smem_store_op` and `StMatrix8x8x16bOp(…)`**,
  and that enum always comes from `LayoutEnum.from_tensor(c)` (CuTeDSL `dense_gemm.py` L404), never hand-written.
  The transposing branch is **reachable but not the default**: `--c_major` of CuTeDSL `dense_gemm.py` takes `{n, m}` (CuTeDSL L162),
  and `m` hits the transposing member, but the default and every example command in the docs use `n` (CuTeDSL `dense_gemm.py` L74/L89/L162).
  That is: **the default path of the public examples never reaches the transposing member**; the branch this entry
  teaches is not actually executed anywhere in the examples — which is why this entry needs GPU certification rather
  than citing the examples.
  (Note also that `--c_major m` brings in an alignment precheck
  `check_contigous_16B_alignment(c_dtype, c_major == "m", …)`, CuTeDSL `fmha.py` L1377.)
- **GPU-certified KB entries of the same tree** (skeleton reused directly, not repeated here):
  §3 of [`tma_store_epilogue`](tma_store_epilogue.md) (three-layer r2s construction + staging skeleton),
  §3-§4 of [`wgmma_mn_major_operand`](wgmma_mn_major_operand.md) (deriving the transposer stride,
  a second `get_tensor` on the same `MemRange`, associativity with the swizzle),
  §3 of [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) (the WGMMA trio, warpgroup-slot slicing,
  generic-store → async-proxy fence),
  gotchas 0 / (d) of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md) (atom M mode always 64,
  faking the C-frag with static identity coordinates).
- **GPU-certified** by a GPU probe (the probe is not included in this release) — ✅ Verified
  (`PROBE PASS max_diff=0.0`, all four contracts passed).
  The four contracts: A transposing member + MN_SW128 face lands correct cell for cell (numerical);
  B non-transposing member + K_SW128 face equally correct (numerical, **proving the two members are logically
  equivalent** — the origin of gotcha 1); C the bytes written by the transposing member are consumed by a real WGMMA
  as a **K-major** A operand, and multiplying by an identity B gives the transpose of the payload (numerical); D PTX
  census: every `stmatrix` of A carries `.trans`, none of B's does, **the two counts are equal** (non-numerical,
  `max_diff=n/a`). Two extra cross controls (member and face orientation deliberately mismatched) only print
  `[diag]`, used to prove that the `LayoutEnum` really is load-bearing.
