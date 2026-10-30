# warp_mma_ldsm_operand_displaced_view (the ldmatrix operand path of warp-level `mma.sync m16n8k16`: a displaced smem view has to **restate** its 16 B alignment)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry and [`smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`](smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128.md)
> are different sites of the same family of 8×8×16b matrix atoms: that one covers **C fragment → STSM → face** (exit), this one covers
> **face → LDSM → A/B fragment → `cute.gemm`** (entry), plus the one rejection on this entry path in CuTe DSL that is
> **not at the spelling level but at the pointer-declaration level**: `cute.copy` checks the source pointer's
> **declared** alignment against the atom's requirement, and `iterator + dynamic offset` drops the declared alignment to the element width.

## Purpose

A warp uses `mma.sync.m16n8k16` (bf16 in, f32 accumulate) to compute its own 16-row × N-column tile;
the A / B operands are taken per k-step via `ldmatrix.x4` from a **padded row-major smem face** (row pitch not a power of 2, no swizzle);
each warp's sub-block start = **dynamic row index × row pitch + static column offset**.
This entry gives: how to spell the atom and the tiled copy, two compilable spellings of the displaced view, a zero-cost trace-time alignment
audit, the certified A-fragment / C-fragment thread-value maps, and the lane arithmetic for reducing along M on the C fragment.

## When needed (trigger)

- `cute.compile` dies in `cute.copy` with

      'cute.copy' op src ptr alignment (16 bits) does not meet requirement (128 bits)
      of atom ldsm<bf16, (8,8), u16, num_matrices=4>

  even though the face was clearly allocated with `byte_alignment=128` and the address is by construction a multiple of 16 B — the view was built as
  `cute.make_tensor(face.iterator + r0 * pitch + c0, ...)`.
- The same atoms passed on a toy geometry with offset 0, and die as soon as "one dynamic row block per warp" is introduced.
- You need to do a row reduction **along M** (across the 8 lane groups) on the m16n8k16 C fragment per lane, and need the fragment map.

## Minimal runnable example

Below is the skeleton extracted from the certification probe: one (16 × 16) accumulator tile, K-major A face (M, K) and
B face (N, K) (k contiguous, so both sides use **non-transposed** `ldmatrix`, one `x4` covering a 16×16 sub-block).

```python
import cutlass, cutlass.cute as cute
from cutlass.cute.nvgpu import warp as cwarp
BF16, F32 = cutlass.BFloat16, cutlass.Float32
MT, NT, KSTEP = 16, 16, 16

# ---- atom level: built inside the kernel body (trace-time objects) ------------------------------
mma  = cute.make_tiled_mma(cwarp.MmaF16BF16Op(BF16, F32, (16, 8, 16)), (1, 1, 1), (MT, NT, KSTEP))
ldsm = cute.make_copy_atom(cwarp.LdMatrix8x8x16bOp(False, 4), BF16)   # non-transposed: k contiguous on the face
tcA  = cute.make_tiled_copy_A(ldsm, mma)
tcB  = cute.make_tiled_copy_B(ldsm, mma)
idC  = mma.get_slice(lane).partition_C(cute.make_identity_tensor((MT, NT)))  # C-fragment coordinates
NV   = cute.size(idC)                                                          # (16,16) tile -> 8
acc  = cute.make_rmem_tensor(mma.partition_shape_C((MT, NT)), F32)

# ---- displaced view: an undecorated plain Python helper, called from the kernel body ------------
def view(face, pitch, r0, c0, nrow, ncol):
    """(nrow x ncol) sub-block starting at (r0, c0). r0 may be a runtime value; c0 / nrow / ncol static.
    `face.iterator + off` lowers to add_offset, and the result's **declared** alignment = the element's 2 B;
    the ldsm atom wants 16 B. When the address itself is by construction a multiple of 16 B (pitch bytes and c0 bytes both multiples of 16),
    `.align(16)` only restates the fact the compiler dropped."""
    p = (face.iterator + r0 * pitch + c0).align(16)
    return cute.make_tensor(p, cute.make_layout((nrow, ncol), stride=(pitch, 1)))

def view_local_tile(face, pitch, nrow_all, ncol_all, r0, c0, nrow, ncol):
    """Coordinate-slicing spelling of the same sub-block (the CuTe way), also certified; pick either."""
    full = cute.make_tensor(face.iterator, cute.make_layout((nrow_all, ncol_all), stride=(pitch, 1)))
    return cute.local_tile(full, (nrow, ncol), (r0 // nrow, c0 // ncol))

# ---- one K segment: acc[16 rows][16 cols] += A[16][nk*16] . B[16][nk*16]^T ----------------
def seg(mma, tcA, tcB, acc, sA, sB, nk, lane):
    thr = mma.get_slice(lane)
    pA, pB = thr.partition_A(sA), thr.partition_B(sB)
    rA, rB = mma.make_fragment_A(pA), mma.make_fragment_B(pB)
    sAv = tcA.get_slice(lane).partition_S(sA)
    sBv = tcB.get_slice(lane).partition_S(sB)
    rAv, rBv = tcA.retile(rA), tcB.retile(rB)
    for k in range(nk):                      # plain range: this function is not rewritten by the preprocessor (gotcha 4)
        cute.copy(tcA, sAv[(None, None, k)], rAv[(None, None, k)])
        cute.copy(tcB, sBv[(None, None, k)], rBv[(None, None, k)])
        cute.gemm(mma, acc, rA[(None, None, k)], rB[(None, None, k)], acc)

# inside the kernel body: row0 is a runtime value (derived from the warp), col0 static
seg(mma, tcA, tcB, acc,
    view(sA_face, PITCH_A, row0, col0, MT, nk * KSTEP),
    view(sB_face, PITCH_B, 0,    col0, NT, nk * KSTEP), nk, lane)

# ---- trace-time alignment audit: three Python ints, zero cost, decided before any rung -------------
base = sA_face.iterator
disp = base + row0 * PITCH_A + col0
print("align base=%d displaced=%d align16=%d" % (base.alignment, disp.alignment, disp.align(16).alignment))
# reading at certification: base=128 displaced=2 align16=16 (ldsm wants 16)

# ---- reduction along M on the C fragment (max over the 16 rows of each column, result placed by column at idC coordinates) ----
for n in cutlass.range_constexpr(2):         # two n8 sub-tiles
    for c in cutlass.range_constexpr(2):     # two adjacent columns
        m = fmax(acc[c + 4 * n], acc[c + 2 + 4 * n])     # r and r+8 of the same lane
        for off in cutlass.range_constexpr(3):
            m = fmax(m, bfly(m, 4 << off))               # the other 7 lane groups: xor 4 / 8 / 16
        out[(idC[c + 4 * n][1])] = m                     # column index from the identity partition
```

`fmax` / `bfly` are each a single-instruction PTX escape in one `@dsl_user_op` (`max.f32` /
`shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff`, constraints `"=f,f,f"` / `"=f,f,r"`,
[`ptx_escape`](ptx_escape.md) form (3)); the DSL names `cute.arch.fmax` / `cute.arch.shuffle_sync_bfly`
have certified usage in [`softmax_segment_ilp`](softmax_segment_ilp.md); pick either.

**Certified thread-value maps** ((16 × 16) tile, `(1,1,1)` thread layout, `(16,16,16)` permutation):

- C fragment: `NV = 8`, layout `((2,2),1,2):((1@1,8@0),0,8@1)`, value index `v = c + 2*j + 4*n`,
  `row = lane // 4 + 8*j`, `col = 8*n + 2*(lane % 4) + c`. So the 16 rows of one column are spread over
  "(v, v+2) of the same lane" × "bits 2/3/4 of the lane" — reducing along M = one in-lane max +
  butterfly 4 / 8 / 16; for reducing along N ((v, v+1) of the same lane and the n sub-tiles + butterfly 1 / 2) see
  [`softmax_segment_ilp`](softmax_segment_ilp.md).
- A fragment (one k-step loaded by a non-transposed `x4` ldsm): value index `i` →
  `row = lane // 4 + 8*((i // 2) % 2)`, `k = 2*(lane % 4) + i % 2 + 8*(i // 4)`,
  i.e. the standard PTX m16n8k16 A map; the fragment loaded by `make_tiled_copy_A(ldsm, mma)` + `partition_S` + `retile` +
  `cute.copy` is bit-equal to it.

**When two warps each compute half of K and exchange partial sums through an smem face**, writer and reader address with the **same** `idC` coordinates
(`face[(half, tile, idC[i][0], idC[i][1])] = acc[i]`, barrier, then `acc[i] += face[(1-half, ...)]`):
both sides have the same lane layout, there is no second bijection to get wrong, and both halves get the complete result.

## Gotchas

1. **`iterator + offset` drops the declared alignment to the element width, and `cute.copy` rejects it per the atom's requirement.**
   `Pointer.__add__` lowers to `add_offset`, and the result's `.alignment` is the most conservative value computed for "a dynamic offset"
   (2 B for bf16), regardless of the `byte_alignment=128` at allocation; `LdMatrix8x8x16bOp` wants 16 B, so
   `cute.copy` reports at trace time "src ptr alignment (16 bits) does not meet requirement (128 bits)".
   **The address itself is fine** — the declaration was lost, the arithmetic is not wrong. Two fixes, both certified:
   `(it + off).align(16)` (restate), or `cute.local_tile(whole-face tensor, (nrow, ncol), (r0 // nrow, c0 // ncol))`
   (coordinate slicing, no pointer arithmetic). The zero-displacement control passed in the same round, proving the atom / partition / retile themselves
   are fine and only the displacement lost the declaration.
   The **D-side** rejection of the same family (a 128-bit `CopyUniversalOp` storing into a ring slot of a swizzled iterator, slot index a runtime offset, reporting
   `D ptr alignment (16 bits) … of atom universal_copy<bf16, 128 b>`) and the pricing of its four spellings are in
   [`consumer_warpgroup_pingpong_parity`](consumer_warpgroup_pingpong_parity.md) gotcha 10.
2. **Use `.align(16)` only when the address really is aligned.** What is certified is the form "row-pitch bytes and column-offset bytes are both multiples of 16"
   (the restatement is a fact); the consequence of restating `.align(16)` on a truly misaligned address is **untested** in this entry;
   do not treat it as a switch to bypass the check. For a face whose row pitch is not a multiple of 16 B, change the pad first.
3. **The alignment audit is a trace-time Python int, zero cost, before any rung.** `ptr.alignment` is directly readable;
   print base / displaced / `.align(16)` as a fact row (certified reading `128 / 2 / 16`), and the hypothesis is decided before any
   GPU round; when writing a probe it belongs in the first rung.
4. **In an undecorated helper, the k loop uses plain `range`, not `range_constexpr`.** Plain Python functions like `seg` / `view`
   called from the kernel body are not rewritten by the preprocessor; `cutlass.range_constexpr` inside them is
   a trace-time `DSLRuntimeError`, while `cutlass.const_expr` on Python values works as usual — with a static upper bound
   `range` unrolls the same as `range_constexpr`. Details and error text in
   [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 5.
5. **The same spelling passes on a toy with offset 0 and dies on the real geometry; the only difference is the operand view.** The failure in gotcha 1 is exactly
   this form: the atom spelling (`make_tiled_mma` / `make_tiled_copy_A` / `make_fragment_A` /
   `retile` / `gemm`) and the accumulator spelling were all 0.0 on a single-warp toy, and diffing toy and real left only "is the view start a
   dynamic offset". When "toy passes, real dies", diff how the operand views are built first; do not re-guess the atom.
6. **Both sides K-major means both sides non-transposed.** When the A face (M, K) and the B face (N, K) are both k-contiguous, one `LdMatrix8x8x16bOp(False, 4)`
   atom feeds both `make_tiled_copy_A` and `make_tiled_copy_B`; the criterion for the transposed member (which mode is contiguous on the face)
   is in [`smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`](smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128.md) gotcha 2.
7. **Use a strict `0.0` judge: pick inputs that are exact small bf16 values so every partial sum is exact in f32 regardless of accumulation order.**
   Certification used quarter-integers in [-0.5, 0.5] (products are multiples of 2^-4, sums of a few hundred terms far below 2^24),
   so any nonzero `max_diff` is a bijection error, not rounding; the numpy reference uses elementwise multiply-add, not matmul.

## Sources

- GPU certification probe (not included in this release) — **Status: ✅ Verified**;
  after 6 rounds all three verdicts passed in the same round: `PROBE PASS max_diff=0.0  # Q1-s-tile0` (the complete
  (64 × 16) result after the two K halves are exchanged through the face; each half's copy is judged), `PROBE PASS max_diff=0.0  # Q2-s-tile1` (another row base),
  `PROBE PASS max_diff=0.0  # Q3-rowmax` (lane reduction along M, pinning the C-fragment map). Same-round fact rows:
  `align <face> base=128 displaced=2 align16=16 (ldsm wants 16)` (the reading for gotchas 1 / 3),
  `s8 … A-frag vs standard m16n8k16 map=0.0` (A-fragment map), `c1 … compiled+launched` (the form that splits the K halves
  into runtime `if` branches, with a different nk in each branch, also compiles and launches; recorded as a fact only).
  The first five rounds each died at one point, all recorded in the probe's per-version docstring: storing to smem via a Python object attribute subscript
  inside a staged region ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 6) → an `[Internal Error]` without a line number
  (switched to a ladder, [`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 12) → the cause of death truncated at 150 characters
  ([`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 13) → `range_constexpr` inside an undecorated helper
  ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 5) → lowest failing rung pinned to `cute.copy`,
  three controls (`.align(16)` / zero displacement / `local_tile`) all passed in the same round, each with A-fragment map 0.0
  ([`gpu_probe_ladder`](gpu_probe_ladder.md) gotcha 15) → the next round passed with `.align(16)` as the shipped spelling.
- Public interfaces: `cutlass.cute.nvgpu.warp.MmaF16BF16Op` / `LdMatrix8x8x16bOp`; `cute.make_tiled_mma`
  (`permutation_mnk`) / `make_copy_atom` / `make_tiled_copy_A` / `make_tiled_copy_B` /
  `TiledMma.get_slice` / `ThrMma.partition_A` / `partition_B` / `partition_C` /
  `TiledMma.make_fragment_A` / `make_fragment_B` / `partition_shape_C` / `TiledCopy.get_slice` /
  `ThrCopy.partition_S` / `TiledCopy.retile` / `cute.copy` / `cute.gemm` / `cute.local_tile` /
  `cute.make_identity_tensor` / `cute.make_rmem_tensor` / `cute.make_tensor` / `cute.make_layout`;
  `Pointer.__add__` / `Pointer.align` / `Pointer.alignment`; `cutlass.utils.SmemAllocator`
  (`byte_alignment=`); `cutlass.cutlass_dsl.dsl_user_op` + `cutlass._mlir.dialects.llvm.inline_asm`.
- Official docs: PTX ISA `ldmatrix`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix>;
  per-lane distribution of the `mma.m16n8k16` A / C fragments
  <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma>;
  `shfl.sync` <https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync>.
