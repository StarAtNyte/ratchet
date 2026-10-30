# tma_im2col (the im2col mode of SM90 TMA: hardware sliding-window gather + out-of-bounds zero-fill)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry covers the **im2col** variant of `cp.async.bulk.tensor`: it uses the same TMA unit and the same
> mbarrier handshake as the certified [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) (the `.tile` variant);
> **only the descriptor encoding differs**. The mbarrier / WGMMA / smem swizzle parts are not repeated; see
> that entry.

## Version threshold: requires nvidia-cutlass-dsl ≥ 4.6

**✅ 4.6.0.dev0 has a final GPU verdict: usable** (certified by an H100 GPU probe, not included in this release; all three rungs PASS):

| rung | Result |
|---|---|
| 0 descriptor construction + `atom_tma_partition` | **PASS** (certified form = both rest modes free, see §3) |
| 1 identity-descriptor end-to-end copy | **PASS max_diff=0.0** |
| 2 padded 3×3 gather, no consumer masking | **PASS max_diff=0.0**, 21.0% of the reference points are hardware-zero-filled halo |

This image's **default 4.4.2 cannot express this path** (`atom_tma_partition` rejects the im2col atom, and
there is no way around it; 10 different constructions were tried and all rejected — **do not rerun those
experiments**). To get "sliding window + out-of-bounds zero-fill" on 4.4.2, use
[`tma_ndim_oob_zerofill`](tma_ndim_oob_zerofill.md).

Recipe for switching to 4.6 inside a job: `scripts/dsl_pin_install.sh` (sourceable, `DSL_PIN` env), or by hand:

```bash
export PIP_CONFIG_FILE=/dev/null   # the image's pip.conf points at pypi.ngc.nvidia.com, unreachable from the cluster
pip install --no-cache-dir --no-deps --timeout 60 \
    --index-url <a reachable PyPI mirror> \
    "nvidia-cutlass-dsl==4.6.0.dev0" "nvidia-cutlass-dsl-libs-base==4.6.0.dev0"
# ⚠ the shell package is only 10kB, the real thing is libs-base; with --no-deps both must be installed together, else import silently falls back to 4.4.2
# ⚠ do not install with dependencies: it drags cuda-python to 13.x and breaks cuda.bindings.driver
python3 -c "import cutlass,sys; sys.exit(0 if cutlass.__version__.startswith('4.6') else 1)"
export CUTE_DSL_CACHE_DIR=/tmp/dsl46_jit_cache   # do not share 4.4.2's warm JIT cache
```

4.6 also completes the official front end: `cpasync` adds the whole
`CopyBulkTensorIm2ColG2S{,Multicast}{Op,Trait,NonExecTrait}` family, and
`cute/algorithm.py` / `nvgpu/helpers.py` carry im2col support. **The official front end has not yet been
certified on GPU in this repo**; what follows is the **certified** construction path (the rev5 probe, verbatim,
passed all three rungs under 4.6). A new effort should try the official front end first (shorter, maintained),
and backfill this entry once it works.

## Purpose

`.tile`-mode TMA can only move "an axis-aligned box of the tensor". **im2col mode** makes the same TMA unit
move "the rectangle obtained by unrolling a sliding window" instead: the descriptor additionally carries the
*corner* (window origin offset, may be negative), *traversal stride* and *filter offset/dilation*, and the
hardware performs the gather

```
dst[m, (c, s, r)]  =  src[ n, h(m)*stride_h + r*dil_h + lower_corner_h,
                              w(m)*stride_w + s*dil_w + lower_corner_w, c ]
```

during the move, and **fills window positions that fall outside the tensor directly with 0** — zero-fill is
TMA-unit behaviour, costs no SM instruction, writes no smem and needs no extra barrier.

In one sentence: move "fetch tiles by sliding window + zero the borders" entirely off the consumer's critical
path and into the DMA hardware.

## When needed (trigger)

- The dataflow is "take tiles from an N-D tensor by a **sliding window** and feed the tensor cores; the window
  goes out of bounds, and out-of-bounds positions must be 0".
- You already use `.tile` TMA, but to express the sliding window you had to degrade into one of the following
  forms — that is the signal to switch to im2col:
  - split the window into taps, shift the coordinate origin with `cute.domain_offset` per tap and issue an
    ordinary TMA each time, then **mask the out-of-bounds rows to zero on the consumer side**;
  - materialise the source tensor as an unrolled matrix in gmem beforehand (an extra read/write pass +
    several times the memory).
- Profiler symptoms (the machine-readable footprint of the degraded forms):
  - `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` clearly above the performance baseline, and
    **drifting with main-loop levers (stage count / wait depth / issue order)** — conflicts of a pure epilogue
    do not move with main-loop levers; the drift says the conflict source is the masking write in the main loop;
  - `sm__pipe_tensor_cycles_active` share far below the performance baseline (the masking write + rendezvous
    occupy the consumer, the tensor cores wait);
  - `smsp__average_warps_issue_stalled_barrier_per_issue_active` elevated (one extra rendezvous per k-tile).

## Minimal runnable example (path GPU-certified on 4.6)

### 1) Atom construction (the im2col twin of `make_tiled_tma_atom`)

Apart from the op name, the seven geometry tuples and how `cta_v_map` is built, every line is copied from `cpasync/helpers.py:51-190`.

```python
import cutlass.cute as cute
from cutlass.cute import atom as cute_atom
from cutlass.cute import core as cute_core
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync.copy import CopyBulkTensorTileG2SNonExecTrait
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir


def make_im2col_tma_atom(op, gmem_tensor, smem_layout, cta_tiler,
                         lower_corner_whd, upper_corner_whd,
                         lower_padding_whd, upper_padding_whd,
                         stride_whd, lower_srt, stride_srt,
                         num_multicast=1):
    # Note: the identity layout is built on product_each(shape) (unlike the .tile version, which uses shape directly),
    # matching C++ make_identity_layout(product_each(shape(tensor_cwhdn))).compose(cta_tiler)
    cta_v_map = cute_core.composition(
        cute_core.make_identity_layout(cute.product_each(gmem_tensor.shape)),
        cta_tiler)

    if isinstance(smem_layout, cute_core._ComposedLayout):   # swizzled -> take .value
        smem_layout = smem_layout.value
    op.smem_layout = smem_layout

    kind = (_cute_nvgpu_ir.Im2ColTmaLoadEnum.sm_90 if num_multicast == 1
            else _cute_nvgpu_ir.Im2ColTmaLoadEnum.sm_90_multicast)

    # pack the seven geometry tuples with _pack_int_tuple (the only one measured on GPU to build, see gotcha 2;
    # going by the C++ type name LowerCornerStride and using _pack_stride gives SIGSEGV)
    pack = cute_core._pack_int_tuple
    res = _cute_nvgpu_ir.atom_make_non_exec_im2col_tma_load(
        gmem_tensor.value, smem_layout, cta_v_map, kind,
        pack(lower_corner_whd), pack(upper_corner_whd),
        pack(lower_padding_whd), pack(upper_padding_whd),
        pack(stride_whd), pack(lower_srt), pack(stride_srt),
        num_multicast=num_multicast)

    # the field enum of the exec atom (AtomCopyFieldTmaLoad: tma_bar=1) is shared by .tile and im2col,
    # so the .tile Trait is reused directly and cute.copy(..., tma_bar_ptr=) works as is (gotcha 9, demonstrated).
    return cute_atom.CopyAtom(op, CopyBulkTensorTileG2SNonExecTrait(res[0])), res[1]
```

### 2) Host side: tensor arrangement + geometry encoding

```python
@cute.jit
def host(mA_whnc: cute.Tensor, ..., stream):
    # smem layout: exactly as for .tile (rank-2 (M,K), may be swizzled), see wgmma_tma_mbarrier §1
    smem_layout_staged = cute.tile_to_shape(smem_atom, (BM, BK, STAGE), order=(0, 1, 2))
    smem_layout_1stage = cute.slice_(smem_layout_staged, (None, None, 0))

    # The source tensor must be rank-2 ((W,H,D,N), C): spatial in reverse order, batch outermost in mode 0,
    # channel a mode of its own with stride==1. If you hold a 4-D (W,H,N,C) view, group the first three modes:
    mA = cute.group_modes(mA_whnc, 0, 3)              # ((W,H,N), C)

    tma_atom, tma_tensor = make_im2col_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), mA, smem_layout_1stage, (BM, BK),
        lower_corner_whd  = (-pad_w, -pad_h),
        upper_corner_whd  = (pad_w - (S-1)*dil_w, pad_h - (R-1)*dil_h),
        lower_padding_whd = (pad_w, pad_h),
        upper_padding_whd = (pad_w, pad_h),
        stride_whd        = (stride_w, stride_h),
        lower_srt         = (0, 0),
        stride_srt        = (dil_w, dil_h),
    )
    # the logical shape of tma_tensor is already the **unrolled** ((q,p,n), (c,s,r)), not the source tensor's shape.
```

The two formulas of the geometry encoding (the hardware derives the descriptor's box extent from them):

```
along each spatial mode:
  output length = (input length + upper_corner - lower_corner - 1) / stride_whd + 1
  window taps   = 1 + (upper_padding - upper_corner) / stride_srt      # when stride_srt > 0
                  1 + (lower_corner  - lower_padding) / stride_srt    # when stride_srt < 0
base offset = (0, lower_corner_whd, 0, lower_srt)
```

Self-check by substituting the values above: taps `= 1 + (pad - (pad - (S-1)*dil))/dil = S` ✅;
output length `= (Win + (pad-(S-1)*dil) - (-pad) - 1)/stride + 1`
`= (Win + 2*pad - dil*(S-1) - 1)/stride + 1` ✅.

### 3) Kernel side: partition + copy (certified form)

**Leave both rest modes free and slice after the partition** (the isomorphic spelling of the CUTLASS conv collective,
`sm100_implicit_gemm_umma_warpspecialized.hpp:781-800`; the winning form of rung 0 in the final verdict run):

```python
gA = cute.local_tile(tma_tensor, (BM, BK), (None, None))       # (BM, BK, m, k)
tAsA, tAgA = cpasync.tma_partition(
    tma_atom, 0, cute.make_layout(1),
    cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
cute.copy(tma_atom, tAgA[(None, mt, kt)], tAsA[(None, pstate.index)],
          tma_bar_ptr=pipe.producer_get_barrier(pstate))
```

After grouping, the gmem operand is **rank-3** `((BM,BK), m, k)`, matched against the **rank-2** smem
`((BM,BK), PIPE)` — unequal ranks on the two sides are **normal** (C++ `tma_partition` does
`append<rank>(layout_V, _)` on each operand separately and only requires equal `size<0>`). The form that pins
M or K first and passes a rank-2 operand in is **uncertified**; do not change it on your own initiative.
`PipelineTmaAsync`, `expect_tx`, tx_count by single-stage smem bytes — all the same as `.tile` (see
[`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)).

## Gotchas

1. **The source tensor must be arranged as rank-2 `((W,H,D,N), C)`**: spatial modes in **reverse order** (W
   innermost, D outermost, the opposite of the D,H,W array customary in problem statements), batch the
   outermost dimension of mode 0, channel alone as mode 1 with **stride==1 required**. Given an `(N,H,W,C)`
   4-D tensor, or with C mixed into mode 0, the descriptor still builds but the addressing is wrong. Starting
   from a `(W,H,N,C)` view, `cute.group_modes(t, 0, 3)` is the easiest.
2. **The seven geometry tuples are MLIR values, not Python tuples**, and must be packed first; their rank must
   equal the number of spatial modes (2 for 2-D, excluding N and C), again in reverse order (W,H,D) / (S,R,T).

   ✅ **The packing must be `cute_core._pack_int_tuple`** (measured on H100).
   On the C++ side these parameters have type names `LowerCornerStride` / `TraversalStride` / `DilationStride`;
   going by the names and using `cute_core._pack_stride` is **wrong** — measured to SIGSEGV outright.

   ⚠️ **Wrong packing does not raise; it SIGSEGVs.** This op is an ODS-generated generic builder with very
   loose type constraints on the geometry operands; a wrong packing **passes** ODS verification and is then
   converted unchecked by `cast<>` in the C++ builder — a release build has no assertions, so it is straight UB
   and a segfault, **not a Python exception**, and `try/except` cannot catch it. Any probe that touches this
   layer of raw MLIR ops **for the first time** should **spawn one subprocess per candidate** (`subprocess`
   re-entering the same script, the parent never touches CUDA, segfault returncode is `-11`), print a
   `flush=True` marker line before each crash-capable call, and locate a crash by the last marker printed.
3. **Corner and padding are two different things; do not mix them up.** `*_corner_whd` goes into the
   descriptor for **addressing** and may be negative (fprop's `lower_corner = -pad` is always negative);
   `*_padding_whd` **only** takes part in the formula for the tap count, **must be ≥ 0**, and never enters the
   descriptor. Filling `lower_padding_whd` with 0 by assumption is a common mistake — it should hold the real
   pad value (it only affects the tap count when `stride_srt < 0`, but a wrong value silently changes the K
   length when you switch to that branch).
4. **Value ranges have hard ceilings** (exceeding them is a silent error, not an error message):
   - padding: with 1/2/3 spatial modes the ceilings are 65536 / 256 / 16 respectively, and it must be ≥ 0;
   - corner: signed, range `[-2^(16/S_dims-1), 2^(16/S_dims-1) - 1]`;
   - filter offset (`lower_srt`, `(tap-1)*dilation`): unsigned, `[0, 2^(16/S_dims) - 1)`.
   When you need negative taps (e.g. a dgrad form), do not give a negative `lower_srt`; instead raise
   `lower_srt` to positive and make `stride_srt` negative — the tap-count formula then switches to the
   `lower_*` branch automatically.
5. **`cta_v_map` must be built on `product_each(shape)`**, not on `gmem_tensor.shape` as in the `.tile`
   version. Mode 0 of the source tensor is the nested `(W,H,N)`; building the identity layout on the nested
   shape and composing a rank-2 tiler gives a profile mismatch. The op retrieves the nested structure from
   `gmem_tensor` itself internally, so no information is lost.
6. **smem constraints**: `rank(slayout) >= 2` (im2col uses at most 2 smem modes to align vectorisation with
   gmem), and `product_each(shape(slayout)) == product_each(shape(cta_v_map))`, i.e. `cta_tiler` must equal the
   shape of the single-stage smem layout mode by mode. What is passed to the builder is still the single-stage
   layout **with the stage mode sliced off** (same as `.tile`, see `wgmma_tma_mbarrier` gotcha 2). For a
   swizzled layout take `.value` of the `_ComposedLayout` before passing it.
7. **The returned coordinate tensor has the "unrolled" shape `((q,p,n), (c,s,r))`, not the source tensor's
   shape.** On it, `local_tile((BM,BK), ...)`: the M direction walks the tiling of **output** pixels `(n,p,q)`
   (q fastest), the K direction walks the composite mode `(c, s, r)` (c fastest). So when `BK == C`, one k-tile
   is exactly one tap, `k = c + C*s + C*S*r`. Do not tile with the source tensor's shape.
8. **Do not write a single line of masking on the consumer side.** Out-of-bounds zero-fill is done by the TMA
   unit; adding an "insurance" mask hands back the whole gain of this recipe (one fewer smem write + one fewer
   rendezvous) and introduces bank conflicts too.
9. **The field enum of the exec atom is shared with `.tile`** (`AtomCopyFieldTmaLoad.tma_bar = 1`), so
   `CopyBulkTensorTileG2SNonExecTrait` can be reused as is, and `cute.copy(..., tma_bar_ptr=)`, the
   `expect_tx` byte count and the usage of `PipelineTmaAsync` are exactly the same as `.tile` —
   **tx_count is still computed from the single-stage smem bytes** (im2col does not change how many bytes one
   TMA moves). Demonstrated end to end by rungs 1/2 of the final verdict run.
10. **Multicast**: change `kind` to `Im2ColTmaLoadEnum.sm_90_multicast`, pass the number of participating CTAs
    in the cluster as `num_multicast`, and give `cute.copy` an extra `mcast_mask=` (same as `.tile`). The
    constraint is `size(slayout) % num_multicast == 0`. (The multicast branch is uncertified on GPU.)
11. **All pack / layout / atom construction must be inside `@cute.jit` or `@cute.kernel`**:
    `_pack_*`, `tile_to_shape` and `get_smem_layout_atom` all need a live MLIR context; at module scope they
    raise on the very first call, with the symptom "not a single PROBE line was printed".
12. **The file must not contain `from __future__ import annotations`** (same as gotcha 13 of
    [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)).
13. **`ValueError: Operation creation failed` carries no information because nobody is listening for MLIR
    diagnostics at trace time — not because MLIR has nothing to say.** This holds for **any** place that calls
    raw MLIR ops directly, and is the most reusable item of this recipe.

    CuTeDSL does install a diagnostic handler, but at the wrong time: `diagnostic()` at CuTeDSL `base_dsl/dsl.py:898-927`
    is only called in `compile_and_jit` (CuTeDSL `dsl.py:967`), i.e. **after the module has been built**.
    A failed raw-op construction happens **during tracing**, when there is no handler at all in the context,
    so MLIR's `emitError` falls on the floor and the Python side only gets `Operation creation failed`.

    ✅ Fix: **attach your own handler at trace time**, and turn on `emit_error_diagnostics`:

    ```python
    from cutlass._mlir import ir

    _HANDLERS = []          # a handler that gets GC'd detaches automatically; keep a reference

    def enable_mlir_diagnostics():
        ctx = ir.Context.current            # only exists inside @cute.jit / @cute.kernel
        def cb(d):
            print(f"  [MLIR {d.severity}] {d.message}", flush=True)
            for n in d.notes:
                print(f"  [MLIR note] {n.message}", flush=True)
            return False                    # False = do not swallow, keep propagating
        _HANDLERS.append(ctx.attach_diagnostic_handler(cb))
        ctx.emit_error_diagnostics = True   # without this, errors do not go through the handler
    ```

    Call it on the first line of the `@cute.jit` function body (the MLIR context is alive by then).
    Wrap every step of the function in try/except — the diagnostic facility itself must never crash the run it
    is supposed to explain.

    Other localisation means still apply:
    - **Collect stderr for every non-ok status**, not only in the `signal`/`timeout` branches — the
      informative MLIR diagnostic line and the traceback are both on stderr; printing half of it loses all of it.
    - `@cute.kernel` traces **by executing Python**, so inserting `print(..., flush=True)` in the kernel body is
      legal and effective — put a marker line between two trace-time calls, and when an exception is raised the
      last marker is the statement that failed.

## Sources

- Mechanism and parameter semantics extracted from the **public CUTLASS sources** (4.4.2 byte-identical to GitHub `main`):
  - `include/cute/atom/copy_traits_sm90_im2col.hpp`: `make_tma_atom_im2col` L577
    (operand order, the `((w,h,d,n),c)` arrangement, smem rank assertion L632-634,
    `product_each` constraint L594-595); `make_im2col_tma_copy_desc` L393,
    the two extent formulas L497-513, base offset L551
  - `include/cutlass/conv/collective/detail.hpp` L186-261 (values of corner / `lower_srt`)
  - `include/cutlass/conv/collective/sm90_implicit_gemm_gmma_ss_warpspecialized.hpp`
    L361-399 (legal ranges of padding / corner / filter offset)
  - `include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp`
    L781-800 (the CUTLASS origin of the §3 certified form: both rest modes of `local_tile` free,
    `group_modes` then passed to `tma_partition`)
  - `include/cutlass/conv/convnd_problem_shape.hpp` L398-420 (`get_shape_A()` → `((W,H,D,N),(C))`)
- GPU-certified (H100, this repo's standalone GPU jobs):
  - **final verdict run** (nvidia-cutlass-dsl 4.6.0.dev0): rev5 probe, all three rungs PASS —
    descriptor + partition (mk_free form), identity end to end `max_diff=0.0`,
    padded 3×3 gather `max_diff=0.0` (21.0% of the reference points hardware-zero-filled).
  - an earlier run: packing = `_pack_int_tuple` (`_pack_stride` → SIGSEGV),
    `cta_v_map` = `product_each`; the returned coordinate tensor has bit for bit the same shape as C++
    `make_im2col_tma_copy_desc`, i.e. the extent formulas and corner encoding of §2 are confirmed by the
    hardware-side descriptor construction.
  - Failure localisation under 4.4.2 and the 10 rejected constructions: the conclusion has been superseded by
    4.6; of historical interest only.
