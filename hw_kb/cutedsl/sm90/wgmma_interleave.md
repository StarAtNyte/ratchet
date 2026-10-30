# wgmma_interleave (software pipelining of two WGMMA groups in the same warpgroup: split wait_group + a loop-carried register fragment)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry adds one layer on top
> of [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md): **two mutually dependent WGMMA commit groups are in
> flight interleaved within the same warpgroup**, with `wait_group(1)` as a graded wait, so that a
> register segment (elementwise transform, format conversion) overlaps the second WGMMA group; one RMEM
> A-fragment is carried across iterations.

## Purpose

When a main loop contains two data-dependent GEMMs (GEMM1 produces a tile → register transform → GEMM2
consumes that tile and accumulates across iterations), a sequential "issue-wait-issue-wait" leaves the
tensor cores idle during the register segment. This mechanism issues the commits of GEMM2(i-1) and
GEMM1(i) back to back, `wait_group(1)` retires only the oldest GEMM1 group, the register transform runs
while GEMM2 is still in flight, and `wait_group(0)` then drains — the tensor-core gap is filled by the
register segment.

## When needed (trigger)

- The main loop looks like: `S_i = GEMM1(i)` → `P_i = f(S_i)` (register elementwise + precision
  conversion) → `O += GEMM2(P_{i-1})`, and GEMM2's A operand is in registers (RS form).
- Symptom: the sequential version is correct, but the profiler shows low tensor-core duty cycle and the
  tensor pipe idle during the register segment (the f transform); after interleaving the two GEMMs the DSL
  raises at compile time (this entry gives the exact spelling that traces; for the root causes of the
  compile raise see gotchas 4/5/12).

## Minimal runnable example

Skeleton = **the peeled three-part form**: prologue (synchronously produce the first carried fragment) →
dynamic steady-state loop (each body: one GEMM1 commit + one GEMM2 commit + graded waits) → epilogue
(the last GEMM2). tile (128,128), 2 warpgroups (256 threads), NSTAGE smem stages each for B/V, stage
index and loop trip count both runtime Int32. tiled_mma_1 = SS (as in `wgmma_ss_min_gemm`),
tiled_mma_2 = RS (`a_source=warpgroup.OperandSource.RMEM`, B is an MN-major transposed view).

**Dynamic-loop discipline (the core; any violation raises at compile time)**: allocate every
fragment/view once outside the loop; the body only does in-place memref reads/writes; zero accumulators
with `fill(0.0)`; set ACCUMULATE once with `set(True)` in setup; slice staged operands in place with a
4-mode index inside the `cute.gemm` call; the body builds no fragment, builds no atom and rebinds no
outer name (see gotchas 4/5/7/12 for details).

```python
# ===== setup: everything allocated/bound only once =====
tSrA = wg_1.make_fragment_A(wg_1.partition_A(sA))
tSrB = wg_1.make_fragment_B(wg_1.partition_B(sB))       # staged, 4-mode
tOrP = wg_2.make_fragment_A(wg_2.partition_shape_A((TILE_M, TILE_N)))  # carried
tOrVt = wg_2.make_fragment_B(wg_2.partition_B(sVt))     # staged, 4-mode
acc_s = cute.make_fragment(wg_1.partition_shape_C((TILE_M, TILE_N)),
                           cutlass.Float32)
acc_o = cute.make_fragment(wg_2.partition_shape_C((TILE_M, TILE_D)),
                           cutlass.Float32)
tiled_mma_1.set(warpgroup.Field.ACCUMULATE, True)       # preset once (gotcha 5)
tiled_mma_2.set(warpgroup.Field.ACCUMULATE, True)
g1_kblocks = cute.size(tSrA.shape[2])
g2_kblocks = cute.size(tOrP.shape[2])
P_PAIRS = cute.size(tOrP) // 2
p_src_pairs = cute.make_tensor(acc_s.iterator, cute.make_layout((2, P_PAIRS)))
p_dst_pairs = cute.make_tensor(tOrP.iterator, cute.make_layout((2, P_PAIRS)))
acc_o.fill(0.0)
# n_blocks is a runtime Int32

# ===== prologue: GEMM1(n-1) synchronous, produces the first carried fragment =====
acc_s.fill(0.0)                                         # zero with fill (gotcha 4)
warpgroup.fence()
for kb0 in cutlass.range_constexpr(g1_kblocks):
    cute.gemm(tiled_mma_1, acc_s, tSrA[(None, None, kb0)],
              tSrB[(None, None, kb0, n_blocks - 1)],    # 4-mode in-place slice (gotcha 7)
              acc_s)
warpgroup.commit_group()
warpgroup.wait_group(0)
# f(acc_s) register transform ... then f32->bf16 written into tOrP through pair views (gotcha 10)
for j0 in cutlass.range_constexpr(P_PAIRS):
    qv2 = cute.arch.cvt_f32x2_bf16x2(p_src_pairs[(None, j0)].load())
    p_dst_pairs[(None, j0)].store(cute.TensorSSA(qv2, (2,), cutlass.BFloat16))

# ===== steady-state loop: two groups in flight per body =====
for it in cutlass.range(n_blocks - 1):                  # runtime trip count
    blk = n_blocks - 2 - it                             # body-local Int32 (gotcha 12)
    vblk = blk + 1                                      # GEMM2 consumes the previous iteration's stage
    # (1) commit GEMM1(blk), no wait
    acc_s.fill(0.0)
    warpgroup.fence()
    for kb in cutlass.range_constexpr(g1_kblocks):
        cute.gemm(tiled_mma_1, acc_s, tSrA[(None, None, kb)],
                  tSrB[(None, None, kb, blk)], acc_s)
    warpgroup.commit_group()
    # (2) commit GEMM2(previous iteration), no wait -- reads the carried tOrP, writes acc_o
    warpgroup.fence()                                   # fence even when a group is in flight (gotcha 6)
    for kv in cutlass.range_constexpr(g2_kblocks):
        cute.gemm(tiled_mma_2, acc_o, tOrP[(None, None, kv)],
                  tOrVt[(None, None, kv, vblk)], acc_o)
    warpgroup.commit_group()
    # (3) graded wait: GEMM1 retires (the oldest group, committed first), GEMM2 stays in flight (gotcha 1)
    warpgroup.wait_group(1)
    # register transform f(acc_s) -- overlaps the in-flight GEMM2; must not touch acc_o / tOrP here (gotchas 2/3)
    # (4) drain GEMM2; only after this may acc_o be changed and tOrP overwritten
    warpgroup.wait_group(0)
    # (optional) a register transform of acc_o goes here (gotcha 3)
    for j1 in cutlass.range_constexpr(P_PAIRS):         # f32->bf16 overwrites the carried fragment (gotcha 10)
        pv2 = cute.arch.cvt_f32x2_bf16x2(p_src_pairs[(None, j1)].load())
        p_dst_pairs[(None, j1)].store(cute.TensorSSA(pv2, (2,), cutlass.BFloat16))

# ===== epilogue: the last GEMM2 (stage 0), synchronous =====
warpgroup.fence()
for kb9 in cutlass.range_constexpr(g2_kblocks):
    cute.gemm(tiled_mma_2, acc_o, tOrP[(None, None, kb9)],
              tOrVt[(None, None, kb9, 0)], acc_o)
warpgroup.commit_group()
warpgroup.wait_group(0)
# for writing back acc_o see the epilogue-type entries
```

## Gotchas

1. **`wait_group(N)` retires the oldest groups in commit order**: GEMM1 commits first, GEMM2 second, so
   `wait_group(1)` (= leave at most 1 group in flight) retires GEMM1 and leaves GEMM2 in flight. The count
   only looks at this warpgroup's commit-group sequence, regardless of which tiled_mma a group uses.
2. **Lifetime of the carried RMEM A-fragment**: allocated once outside the loop
   (`make_fragment_A(partition_shape_A(...))`), read **asynchronously** by GEMM2 at the top of the body
   (a WGMMA register operand may be read at any time before its group retires), so it may only be
   overwritten after the `wait_group(0)` drain. Overwriting between commit and drain = data race (silent
   wrong values).
3. **Likewise for GEMM2's accumulator**: do not read or write `acc_o` after the commit and before
   `wait_group(0)`. For register transforms such as a rescale there are two verified positions: after the
   `wait_group(0)` drain (the default position, i.e. the comment at step (4) of this skeleton); or store
   the coefficients and apply them in the next iteration **before** committing GEMM2. At any moment at
   most one group per accumulator is in flight (the wait discipline of this skeleton guarantees it).
4. **Do not `cute.make_fragment` inside a dynamic-loop body**: allocate every accumulator once in setup
   and zero it at the top of the body with `acc.fill(0.0)` (the dynamic loops that passed on GPU in this
   environment — `probe_kv_2stage_staged_gemm` and the sequential-mainloop control — all have this form).
   "New fragment per iteration + ACCUMULATE False→True zero-init" is the verified form for **straight-line
   code** (`probe_p_rs_reinterpret_gemm`); put inside a `cutlass.range` body it raises at compile time.
5. **Set ACCUMULATE once in setup with `tiled_mma.set(ACCUMULATE, True)` and pass tiled_mma directly to
   `cute.gemm`** (the dynamic-loop form verified by `probe_kv_2stage_staged_gemm`). Do not `set(...)` on
   the shared tiled_mma inside the loop (`operand #0 does not dominate this use`, see gotcha 2 of
   [`wgmma_ss_min_gemm`](wgmma_ss_min_gemm.md)), and do not `cute.make_mma_atom` inside the body (an atom
   is a type that cannot be yielded; it triggers gotcha 12). The fresh-atom False→True idiom is only for
   straight-line regions.
6. **Every gemm region starts with `warpgroup.fence()`**, even when another group is in flight at that
   moment — the issue helper of public infra fences unconditionally on every call (quack `sm90_utils.gemm` L163).
7. **Slice staged operands in place with a 4-mode index inside the `cute.gemm` call**:
   `frag[(None, None, kb, stage)]` (stage is a runtime Int32; verified by
   `probe_kv_2stage_staged_gemm`). Do not first bind `rB = frag[(None, None, None, stage)]` in the body and
   then slice — if the name `rB` is also bound outside the loop (typically: the peeled prologue), you hit
   gotcha 12.
8. **The structure must be peeled into prologue / steady state / epilogue**: the prologue synchronously
   (`wait_group(0)`) produces the first carried fragment; the epilogue does the last GEMM2 outside the loop.
   The unpeeled "single loop + dummy GEMM2 adding zero" form is unverified; do not use it.
9. **A literal is fine as the argument of `wait_group`**; passing N as a `cutlass.Constexpr` parameter
   compiles to the same result as the literal and is not a source of failure.
10. **f32→bf16 conversion of the carried fragment**: GEMM1's C-fragment and GEMM2's A-fragment have the
    same TV ownership over (M,N), so they can be linearly reinterpreted through the same iterator and
    converted in pairs — the public-infra form is reinterpreting with `cute.make_tensor(acc.iterator, frgA_layout)`
    (quack `layout_utils.reshape_acc_to_frgA` L250-251) then a paired PTX cvt;
    the equivalent certified spelling builds a `(2, size//2)` linear pair view on both ends +
    `cute.arch.cvt_f32x2_bf16x2`. Build the pair views once in setup and reuse them inside the loop
    (verified by the sequential-mainloop control). With a 128×128 tile `.load().to(dtype)` is also correct
    (other than 128×128 it degrades to per-scalar cvt — a performance problem, not a correctness problem).
11. **The file must not contain `from __future__ import annotations`** (same as
    `wgmma_tma_mbarrier` gotcha 13).
12. **The name-rebinding rule inside a `cutlass.range` body (the number-one source of compile raises)**:
    the DSL traces the body into an scf.for region; a Python name assigned in the body that lives across the
    loop boundary becomes a loop-carried iter_arg — and only scalar kinds (Int32/bool/pipeline state) can be
    yielded. **If a name of tensor / fragment / view / atom type was bound outside the loop and is assigned
    again inside the body, trace time raises**. Discipline: such objects are either bound once in setup and
    only read/written in place in the body (`fill` / index assignment / `.store`), or use brand-new names
    that appear only in the body; in the peeled structure it is safest to keep the local names of the
    prologue/body/epilogue (including constexpr loop variables and TensorSSA temporaries) distinct.

## Sources

- Carried-fragment reinterpretation: quack `layout_utils.py` `reshape_acc_to_frgA` L250-251
- Verified forms of the dynamic-loop body (zero with fill, ACCUMULATE preset, 4-mode in-place slicing of
  staged indices, pair views reused from setup): the GPU probe `probe_kv_2stage_staged_gemm` (GPU PASS; the probe is
  not included in this release) and the sequential-mainloop control rung (s31, GPU compile + run passed)
- GPU-certified by the certification probe, rung `a` (single kernel,
  3 stages, runtime dynamic trip count/stage index, small integer inputs vs an exact fp32 reference, tol=0)
