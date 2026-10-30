# tma_ndim_oob_zerofill (multi-dimensional TMA `.tile` descriptor: per-axis out-of-bounds zero-fill)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry and the certified [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) use the same TMA unit and
> the same mbarrier handshake; **only the dimension split of the descriptor differs**. Atom construction /
> pipelining / swizzle are not repeated; see that entry.
> If you want the hardware im2col mode, read [`tma_im2col`](tma_im2col.md) first — that one **needs
> DSL ≥ 4.6** (GPU-certified; the image default 4.4.2 does not work). Where you are stuck on 4.4.2, this
> entry is its substitute.

## Purpose

A TMA descriptor **bounds-checks every dimension independently**: the part of the box that falls outside
the tensor is written as 0 directly by the TMA unit. Zero-fill costs no SM instruction, writes no smem and
needs no extra barrier.

The key is not "does TMA fill zeros" (it does), but **whether you left the axis that needs checking to
the descriptor**.

## When needed (trigger)

The dataflow is "take tiles from an N-D tensor by a **sliding window** and feed the tensor cores; window
positions out of bounds must be 0", and the current implementation **flattens the tensor first** and then
shifts the coordinate origin. This spelling has a silent correctness gap:

after flattening `(n,h,w)` into one axis `M` and shifting the origin by the scalar pixel offset
`(r-pad)*W + (s-pad)`,

| Out-of-bounds category | What happens after flattening |
|---|---|
| the whole pixel leaves the tensor (`M < 0` or `M >= N*H*W`) | **still** zero-filled by hardware ✅ |
| only stepping out of the current row (`w < 0` or `w >= W`) | lands on a **valid pixel of the neighbouring row** and is read in as real data ❌ |

The second category is invisible to the descriptor — after flattening there is no `w` axis left to
bounds-check. So it can only be **masked to zero on the consumer side**, and that one smem write + one
rendezvous sits on the critical path of the tensor pipeline.

Machine-readable footprint in the profiler:

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` higher than the performance baseline, and
  **drifting with main-loop levers (stage count / wait depth / issue order)** — conflicts of a pure
  epilogue do not move with main-loop levers; the drift says the conflict source is the masking write in
  the main loop;
- `sm__pipe_tensor_cycles_active` share far below the performance baseline (the masking write + rendezvous
  occupy the consumer, the tensor cores wait);
- `smsp__average_warps_issue_stalled_barrier_per_issue_active` elevated (one extra rendezvous per k-tile).

**Fix**: do not flatten. Give every spatial axis its own descriptor dimension (`(N,H,W,C)` rank-4 instead
of `(N*H*W, C)` rank-2), then shift the origin **per axis**. Then `w = q + s - pad` out of bounds is really
out of bounds, the hardware fills 0, and not a single line of masking is needed on the consumer side.

## Minimal runnable example

Certified on H100 (see "Sources"). Apart from the points marked below, everything is word for word the
same as the `.tile` usage.

### 1) Host side: smem layout of the rank-4 box + atom

```python
# smem: rank-4 box + stage mode. order is **the stride order of each mode, 0 = fastest**,
# so C is 0 and stage is 4 (slowest). Get it backwards and a single stage becomes strided;
# make_tiled_tma_atom then aborts the whole process (not an exception).
staged = cute.make_ordered_layout((1, BH, BW, BK, STAGE), order=(3, 2, 1, 0, 4))
one    = cute.slice_(staged, (None, None, None, None, 0))   # one coordinate per mode

tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), mA,      # mA is the rank-4 view of (N,H,W,C)
    one, (1, BH, BW, BK))                       # cta_tiler has the same shape as the box
```

### 2) Self-check: are the axes really separate (**this step cannot be skipped**)

Print the coordinate tensor and look at the basis of each mode:

```python
print(cute_core.pretty_str(tma_tensor.layout))
# BH=4, BW=4 (< H,W):  (4,8,8,64):(1@2,1@1,64@0,1@0)   <- N@2  H@1  W@0  C@0 ✅
# BW == W       :      (4,8,8,64):(1@2,?@1,?@1,1@0)    <- H and W are both @1  ❌
```

The `b` in `x@b` is the descriptor axis number; **two modes sharing one basis means they were merged into
one axis**. The criterion only looks at **whether H and W share a basis**:

- **Merging W with C is harmless** (in the ✅ line above W and C are both `@0`): the box always covers C
  completely, so an out-of-bounds `w` still falls outside the range of the merged axis and is zero-filled
  as before.
- **Merging W with H is fatal**: the box does not cover all of H, an overflowing `w` aliases onto the
  pixels of the next row, and the wrap bug comes back unchanged — exactly what this entry is meant to
  eliminate.

### 3) Kernel side: shift the origin per axis

```python
for r in range(R):
  for s in range(S):
    # The whole mechanism is this one line: H and W are shifted **separately**, each still a bounds-checked axis.
    gA = cute.local_tile(
        cute.domain_offset((0, r - pad, s - pad, 0), tma_tensor),
        (1, BH, BW, BK), (n, hb, wb, 0))        # all coordinates pinned -> no rest mode

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 4),             # group **all four** modes of the box
        cute.group_modes(gA, 0, 4))

    # with all coordinates pinned tAgA is rank-1; pass it directly, do not slice it again
    cute.copy(tma_atom, tAgA, tAsA[(None, pstate.index)],
              tma_bar_ptr=pipe.producer_get_barrier(pstate))
    ...
    pipe.consumer_wait(cstate)
    # <- no masking here, no extra sync_threads: out-of-bounds positions already hold hardware-written 0s; issue WGMMA directly.
```

## Gotchas

1. **A box covering a whole axis → that axis gets merged and its bounds check disappears.** CuTe merges
   contiguous gmem modes when building the descriptor. With `BW == W` the box covers the entire width, so W
   and H legally merge into one axis and `w` is no longer checked on its own. **Keep `BW < W` (and
   `BH < H` when needed)**, and print the basis per §2 as a self-check. This is a silent error: after the
   merge everything compiles and runs as usual, only the halo quietly reads real data from the
   neighbouring row.
2. **`order` is "the stride order of each mode, 0 fastest", not a permutation of modes.** Stage must be
   the slowest (for a rank-5 box that is `order=(3,2,1,0,4)`). Writing `(4,3,2,1,0)` makes stage the
   fastest mode and the single-stage layout becomes `(1,8,8,64):(0,1024,128,2)` (contiguous axis
   stride=2); `make_tiled_tma_atom` **aborts the process outright (signal 6)** — not a Python exception,
   try/except cannot catch it.
3. **The coordinate tuple of `cute.slice_` must have one entry per mode.** A rank-5 staged layout needs
   `(None,None,None,None,0)`; one entry short reports `expects coord and shape of view are weakly
   congruent`. Copying from a rank-2 example is the easiest way to hit this.
4. **`group_modes` must group all modes of the box into the first mode, on both the smem and the gmem
   side.** For a rank-4 box that is `group_modes(t, 0, 4)`; grouping only the first 3 leaves BK dangling,
   and `tma_partition` reports `expects smem and gmem have the same size in the first rank,
   but got smemShape: ((1,8,8),64,2) and gmemShape: ((1,8,8,64))`.
5. **When `local_tile` pins all coordinates, the `tAgA` returned by `tma_partition` is rank-1.**
   Pass it directly to `cute.copy`; copying the multi-k-tile spelling `tAgA[(None, kt)]` reports a
   non-congruent coord. To keep an iteration dimension, give `None` in `local_tile`; only then is there a
   rest mode to slice.
6. **`domain_offset` offsets may be negative and must be given per axis.** `(0, dh, dw, 0)` corresponds
   one-to-one to the tensor's modes; combining the two spatial offsets into one scalar goes back to the
   flattened spelling, and the wrap bug comes right back.
7. **Do not write a single line of masking on the consumer side.** Out-of-bounds zero-fill is done by the
   TMA unit; adding an "insurance" mask hands back the whole gain of this recipe (one fewer smem write +
   one fewer rendezvous) and introduces bank conflicts too.
8. **Cost: the M-direction tile changes from "a run of contiguous pixels" to "a (BH,BW) rectangle".** This
   is a structural premise of this entry, not an option — the GEMM's M dimension must be cut by spatial
   rectangles, `BM = BH*BW`. The choice of `BH`/`BW` is squeezed between gotcha 1 (cannot cover a whole
   axis) and the WGMMA M-mode constraint.
9. **All layout / atom construction must be inside `@cute.jit` or `@cute.kernel`**, and the file must not
   contain `from __future__ import annotations` (same as gotchas 12/13 of
   [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md)).
10. **When debugging at this level, attach the MLIR diagnostic handler at trace time first.** CuTeDSL
    installs its own handler only in `compile_and_jit` (CuTeDSL `base_dsl/dsl.py:967`), i.e. after the module is built; but a failed
    op construction happens during tracing, when nobody is listening, and the Python side only gets an
    uninformative `ValueError: Operation creation failed`. With the handler attached, the same failure
    states `expects smem and gmem have the same size in the first rank, ...` directly.
    This holds for **any** place that calls raw MLIR ops directly.

    ```python
    ctx = ir.Context.current
    ctx.attach_diagnostic_handler(lambda d: (print(f"[MLIR {d.severity}] {d.message}"), False)[1])
    ctx.emit_error_diagnostics = True
    ```

## Sources

- **GPU-certified** by an H100 GPU probe (the probe is not included in this release),
  all three contracts passed:

  ```
  PROBE PASS max_diff=n/a  # rung 0 (H and W are separate descriptor axes): H@1 and W@0
  PROBE PASS max_diff=0.0  # rung 1 (centre tap, in-bounds rank-4 box copy)
  PROBE PASS max_diff=0.0  # rung 2 (all 9 taps, per-axis TMA zero-fill,
                             no consumer masking); 21.0% of the reference is
                             hardware-zero-filled halo
  ```

  Shape `N=4 C=64 H=W=8 R=S=3 pad=1`, box `(1,4,4,64)`, 16 CTAs; the reference is torch's zero-padded
  gather, with no masking at all on the kernel side. Rung 2 additionally asserts that the halo share is
  > 5%, otherwise the test is judged to be running empty.
- **Verbatim error messages of gotchas 1/2/4/5**: from earlier GPU runs (coordinate tensor with H/W merged,
  signal 6, first-rank size mismatch, coord not congruent respectively).
- **Basis for gotcha 10**: in `nvidia-cutlass-dsl 4.4.2`, `diagnostic()` at `base_dsl/dsl.py:898-927`
  is called only by `compile_and_jit` at `dsl.py:967`.
- **Why not the hardware im2col mode**: when this entry was written, `atom_tma_partition` of DSL 4.4.2 did
  not accept an im2col atom (see the "version threshold" section of [`tma_im2col`](tma_im2col.md)). Fixed
  in DSL ≥ 4.6; where the version can be switched inside the job (`scripts/dsl_pin_install.sh`), prefer
  the hardware im2col.
