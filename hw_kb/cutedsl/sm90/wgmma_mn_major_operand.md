# wgmma_mn_major_operand (MN-major WGMMA smem operand: one allocation, written k-major, read MN-major)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry fills the gap in §2 of
> [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md): that entry only teaches how to build a **k-major**
> operand; the minimal example of [`wgmma_interleave`](wgmma_interleave.md) only **consumes** an MN-major
> transposed view (`wg_2.partition_B(sVt)`) and never **constructs** it.
> This entry gives the exact construction of views like `sVt`.

## Purpose

CuTe's SS convention is `C(M, N) += A(M, K) . B(N, K)`: an operand whose **K mode is contiguous** is
called k-major, one whose **MN mode is contiguous** is called MN-major. When a GEMM must **reduce along
the non-contiguous mode in smem**, that operand has to be presented to WGMMA in MN-major form —
otherwise the descriptor does not match the actual byte arrangement.

The mechanism of this entry is **one allocation, two views, zero extra smem**: the buffer is allocated and
written in the producer's natural orientation (contiguous mode last, i.e. the form a TMA / row-major
producer leaves behind), and WGMMA reads it MN-major through a **transposed view of the same bytes**. TMA
can only land smem in the gmem orientation, so a TMA-fed buffer has no other route.

## When needed (trigger)

- One buffer is consumed by two GEMMs with **different reduction modes** (one reduces along the contiguous
  mode, the other along the non-contiguous mode), and the smem budget does not allow storing a second,
  transposed copy.
- Symptom: you feed `partition_B` / `make_fragment_B` a layout that "looks the right shape", and it
  **raises at build time** (incompatible shape / illegal descriptor) rather than giving wrong numbers — this
  is the signature of a misspelled major mode, not a numerical bug; do not go looking at fences or
  accumulate.
- You are tempted to permute the modes of a k-major layout by hand, or to hand-write **flat** strides with
  `make_layout` to fake a transpose (explicitly forbidden by gotcha 5 of `wgmma_ss_min_gemm`: the SS
  descriptor requires core-matrix structure).
  Note: the `make_layout` used in this entry is a **transposer** (the right operand of a composition), not
  a layout fed directly to WGMMA — do not mix the two up (see gotcha 6).

## Minimal runnable example

`C(M, N) += A(M, K) . B(N, K)`, f16 inputs / f32 accumulation, `M=128` (2 warpgroups stacking atoms along
M), `N=32`, `K=64`. A is always k-major.
(`N != K` is deliberate: if B were square, a transposer written in the wrong direction would still build a
shape-legal view, and only the numbers could catch it.)

### 1) The tiled_mma declares B as MN — and the whole set of IR objects is built inside `@cute.jit`

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils
import cutlass.utils.hopper_helpers as sm90_utils
from typing import Callable                      # <-- needed for gotcha 0
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum

M, N, K = 128, 32, 64
NUM_WG = M // 64                                 # the WGMMA atom M-mode is always 64

@cute.jit                                        # <-- required. See gotcha 1
def launch(mA, mX, mC):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.Float16, cutlass.Float16,
        warpgroup.OperandMajorMode.K,            # A k-major
        warpgroup.OperandMajorMode.MN,           # B MN-major  <-- the switch of this entry
        cutlass.Float32,
        atom_layout_mnk=(NUM_WG, 1, 1),
        tiler_mn=(64, N),
    )
```

`OperandMajorMode` only tells the atom how to encode the descriptor; **the layout you feed in must really
have that orientation** — if the two disagree, that is gotcha 3.

### 2) Write view: allocate in the producer's orientation (this is just the ordinary k-major recipe)

```python
    # --- still inside @cute.jit ---
    # logical (K, N), N contiguous. The third argument of get_smem_layout_atom is the element count of the **contiguous mode** = N.
    x_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, cutlass.Float16, N),
        cutlass.Float16)
    sX_layout = cute.tile_to_shape(x_atom, (K, N), order=(0, 1))
```

### 3) Read view: compose a **flat** transposer on the **layout**

```python
    # sXt[(n, k)] == sX[(k, n)], the bytes do not move.
    # The colex linear index of sX's (K, N) domain at (k, n) is k + n*K,
    # so the transposer maps (n, k) to k*1 + n*K, i.e. stride=(K, 1).
    sXt_layout = cute.composition(
        sX_layout, cute.make_layout((N, K), stride=(K, 1)))
```

**The transposer is a flat `make_layout` + explicit colex strides**, not `make_ordered_layout`, and you do
not try to match the nested sub-shapes left by `tile_to_shape`. `cute.composition` linearises the left
layout in colex order before feeding the right one, and **does not require the two to be congruent**.

How to derive the strides (the only step that needs thought): write down the colex linear index of the
**write view's** domain, then copy the coefficients in the coordinate order of the read view. With a stage
mode it is the same — `sV (N, K, S)` at `(n, k, s)` is `n + k*N + s*N*K`, so
`cute.make_layout((K, N, S), stride=(N, 1, N*K))`
(this is exactly the certified spelling of `sVt` in [`wgmma_interleave`](wgmma_interleave.md)).

### 4) Kernel side: `get_tensor` twice on the same `MemRange`

```python
@cute.kernel
def my_kernel(mA: cute.Tensor, mX: cute.Tensor, mC: cute.Tensor,
              tiled_mma: cute.TiledMma,
              sX_layout: cute.ComposedLayout,          # write view
              sXt_layout: cute.ComposedLayout,         # read view
              SharedStorage: cutlass.Constexpr[Callable]):
    ...
    sX = storage.sX.get_tensor(sX_layout.outer, swizzle=sX_layout.inner)
    # ... cooperatively fill sX[(k, n)] ...
    cute.arch.fence_view_async_shared()
    cute.arch.sync_threads()

    # same MemRange, second get_tensor -- logical (N, K), MN-major
    sXt = storage.sX.get_tensor(sXt_layout.outer, swizzle=sXt_layout.inner)

    tSrB = wg_mma.make_fragment_B(wg_mma.partition_B(sXt))
```

**The transpose happens on the host-side layout, not on the kernel-side tensor**; the kernel just takes the
same `MemRange` with a different layout and calls `get_tensor` once more.

**Why it is a byte identity (not "offsets that happen to line up")**: composition is associative, and the
swizzle acts last, on the linear index, so

```
swizzle ∘ (layout ∘ T)  ==  (swizzle ∘ layout) ∘ T
```

The swizzle is carried over untouched, and **there is no need to re-derive offsets per swizzle level** — which
is also why `sXt_layout` is still a `ComposedLayout` and `.outer` / `.inner` split as usual.

The WGMMA trio (`warpgroup.fence()` / `commit_group()` / `wait_group(0)`), changing `ACCUMULATE` on a fresh
atom, warpgroup-slot slicing, and the async-proxy fence after generic stores — all the same as in
[`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md); not repeated here.

## Gotchas

0. **The annotation of `SharedStorage` must be `cutlass.Constexpr[Callable]`** (with
   `from typing import Callable`), **not a bare `cutlass.Constexpr`**. The bare spelling blows up right at
   the kernel signature — and it blows up **every** path through that kernel, so the symptom looks like
   "even the control group fails", and is very easily misdiagnosed as a layout problem. Likewise
   `tiled_mma: cute.TiledMma`, `layout: cute.ComposedLayout`, scalar `n: cutlass.Constexpr[int]`.
1. **Layout / tiled_mma / `@cute.struct` must be built inside `@cute.jit`**.
   `cute.tile_to_shape`, `cute.composition`, `cute.make_layout` and `make_trivial_tiled_mma` are all IR
   builders that need a live `ir.Context`; calling them at module level / in an ordinary function raises
   (**a build-time exception, not a numerical error**). Canonical form: build them in the `@cute.jit` host
   function and pass them as **arguments** to the `@cute.kernel`.
   (The `@cute.struct` class itself may be defined in an ordinary helper function, but that helper must be
   **called from inside `@cute.jit`** — because `cute.cosize(layout)` has to be evaluated in a context.)
2. **The transposer is a flat `make_layout` + explicit colex strides, not `make_ordered_layout`**.
   After `tile_to_shape` the layout's shape is nested (e.g. `((8,8),(32,1))`), but **composition does not
   require the right operand to be congruent with it** — it linearises in colex order. Building a
   `make_ordered_layout` from the nested sub-shapes is actually wrong.
3. **`OperandMajorMode` and the actual layout must change together**. Switching only the tiled_mma's
   `b_leading_mode` to `MN` while still feeding a k-major layout (or the reverse) is not a numerical error but
   **a build-time raise or a silent wrong result**. The two call sites `partition_B` and `make_fragment_B`
   must see the same view object.
4. **Compose on the layout and call `get_tensor` a second time in the kernel; do not compose on the
   tensor**. The canonical path is `cute.composition(sX_layout, T)` on the host side giving a
   `ComposedLayout`, and on the kernel side another `get_tensor` on **the same `MemRange`**.
5. **`MemRange.get_tensor(layout, swizzle=...)`: if the layout is a `ComposedLayout` and `swizzle` is also
   passed, it fails immediately with `TypeError("incompatible layout with swizzle")`**.
   Always split a swizzled layout into `.outer` + `swizzle=.inner`. The same `MemRange` may be
   `get_tensor`'d several times for different views — the mechanism of this entry relies on exactly that.
6. **Do not confuse the transposer with "the layout fed to WGMMA"**. Here `make_layout` serves only as the
   **right operand** of the composition (a pure coordinate permutation); hand-writing a flat `make_layout`
   and feeding it to `partition_B` is still forbidden (same as gotcha 5 of `wgmma_ss_min_gemm`: the SS
   descriptor requires core-matrix structure). Let the layout algebra compute the result.
7. **The stride direction is easy to get backwards**. Both `(N, K)` and `(K, N)` build shape-legal views;
   the wrong one is caught only by the **numbers** — so validation must compare values; guarding
   construction with try/except is useless. Self-check: the transposer's shape is the **read view's**
   shape, and its strides are the colex coefficients of the **write view's** domain reordered in the read
   view's coordinate order.
8. **Not every contiguous-mode width can be transposed**. The bit width of the contiguous mode determines
   the swizzle level (1024 → SW128 / 512 → SW64 / 256 → SW32, see `get_smem_layout_atom`).
   In the example the write view's contiguous mode = 32 f16 = 512 bits. Re-check this whenever the tile
   size changes.
9. **The file must not contain `from __future__ import annotations`** (same as gotcha 8 of
   [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)).
10. **Diagnosis order**: when debugging this, first run a control GEMM with both operands k-major (the
    `wgmma_ss_min_gemm` recipe) at **the same dims and with the same kernel body**. Only when the control
    passes is the scaffolding known to be fine and the failure really in the MN-majorness; if the control
    fails, do not go looking at the transposed view (this entry's certification probe is arranged exactly
    like this; its two verdict lines localise the failure by themselves).

## Out of scope for this entry

- **MN-native allocation** (`get_smem_layout_atom(LayoutEnum.COL_MAJOR, ...)` +
  `tile_to_shape(..., order=(1, 0))`, i.e. the producer writes directly in `(N, K)` orientation).
  For the official form see the non-k-major branch of `make_smem_layout_b` in
  `cutlass/utils/hopper_helpers.py`. **This entry's probe did not certify that path**; probe it separately
  when needed, do not use it as an established fact.

## Sources

- GPU-certified KB entries of the same tree (the exact spelling of the transposed view is taken directly
  from their certification probes):
  - [`wgmma_interleave`](wgmma_interleave.md) — the construction of the staged `sVt` in its certification
    probe, `cute.composition(sV_layout, cute.make_layout((TILE_K, TILE_N, NSTAGE),
    stride=(TILE_N, 1, TILE_N*TILE_K)))`, and the kernel-side form of a second `get_tensor` on the same
    `MemRange`.
  - [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md) — the k-major write view, the WGMMA trio,
    warpgroup-slot slicing, the signature form with `cutlass.Constexpr[Callable]`.
- Public infra modules (`third_party/cutlass_dsl`, nvidia-cutlass-dsl 4.4.2):
  - `cutlass/utils/hopper_helpers.py` `get_smem_layout_atom` (picks the swizzle level by
    `major_mode_size × dtype.width` bits), `make_trivial_tiled_mma` (f16/bf16 → `MmaF16BF16Op(...,
    a_leading_mode, b_leading_mode)`)
  - `cutlass/utils/layout.py` `LayoutEnum.sm90_mma_major_mode` / `is_k_major_b`
  - `cutlass/cute/core.py` `composition` (`R(c) = lhs(rhs(c))` semantics),
    `MemRange.get_tensor` (the `TypeError` for ComposedLayout + swizzle)
- GPU-certified by a GPU probe (the probe is not included in this release) — **`PROBE PASS max_diff=0.0`**
  (rung 0 = the control with both operands k-major; rung A = one allocation + a host-side composition
  transposed view. Both rungs go through **the same kernel body**, changing only the layout and the B major
  mode; small integer inputs vs an exact fp32 reference, requiring `max_diff == 0.0`; a non-integer
  sentinel 0.5 prefill catches unwritten coordinates.)
