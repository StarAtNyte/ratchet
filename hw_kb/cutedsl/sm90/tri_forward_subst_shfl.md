# tri_forward_subst_shfl (serial forward substitution of a small triangular block: `LDS.128` resident rows → `SHFL.IDX` broadcast of packed pairs → FFMA/FSEL register chain → `STS.128` in-place write-back)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Division of labour: [`warp_mma_f16`](warp_mma_f16.md) / [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md) handle the part
> that **can be handed to matrix instructions** (the off-block Schur update, face ↔ fragment movement);
> §2b of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) handles scalar vectors with "one staging per warp
> + `shfl` broadcast" (`LDS.64` fetched by pair). This entry only fills in **the small block that matrix instructions
> cannot take**: the in-place solve / inverse of an 8×8 / 16×16 **unit lower-triangular block** on a 16-bit smem face —
> a row-by-row dependent serial recurrence, whose only lever is moving the **operands of every recurrence step from
> smem to registers + cross-lane broadcast**, so that only FFMAs remain on the dependency chain.

## Purpose

Solve `(I + A) X = I` (or `(I + A) x = b`) where `A` is an NB×NB strictly lower-triangular block stored on a 16-bit
smem face. The recurrence `x[i] = b[i] − Σ_{j<i} A[i][j]·x[j]` is truly serial: step i waits for step i−1. What this
entry changes is not the arithmetic count (both spellings have the same number of FFMAs), but **how many instructions
are issued outside the recurrence, and how**:

| Spelling | Static count (NB=16, per lane) | Cost outside the recurrence (mechanism description, **not timed**) |
|---|---|---|
| per-element `face[(i, j)]` embedded in the recurrence | NB(NB−1)/2 = 120 `LDS.U16` + 120 bf16→f32 conversions + NB `STS.U16` (sub-word; counted directly from the source form) | 120 sub-word issues + sub-word bank conflicts; the load of `A[i][j]` **does not depend on x**, how far ptxas can hoist it is limited only by the register budget, and every load not hoisted far enough exposes one smem latency before its FFMA (`long_scoreboard`) |
| this entry: resident rows + `SHFL.IDX` broadcast | 2 `LDS.128` + 2 `STS.128` + 64 `SHFL.IDX` (column-held / transposed-system back substitution; 105 row-held) + bit unpacking (certified by the probe's differential counts) | the broadcasts depend only on the loads and are all off the chain; the critical path per recurrence step = the last FFMA + one FSEL (the partial sum of the first i−1 terms can be computed ahead) |

> The **cycle difference between the two rows has no timing contract**: the certification probe only judges the
> numbers and the differential counts ("Acceptance" below); estimates of the "N k cycles" kind are not a basis for a
> ruling. What this segment is worth in your kernel must be measured with an ablation (stub out this segment) or
> segment timing; see the gotcha "Shortening the chain does not make the whole kernel faster".

Three things together produce the effect:

1. **Resident rows**: a lane loads only **its own row** (8 bf16 = 16 B = one `LDS.128`; 16 = two), and after loading
   holds it in registers as **32-bit packed pairs** (two adjacent bf16), without unpacking.
2. **Cross-lane broadcast**: the `A[i][0..i-1]` needed by recurrence step i is held by lane i, and `shfl.sync.idx`
   sends its packed words to the whole warp (ceil(i/2) per step i). These words **depend only on the loads, not on
   x**, so all `SHFL`s are **off** the dependency chain, and the compiler places them ahead of the chain or
   interleaves them into it.
3. **Only FFMA + one FSEL on the chain**: unpacking is bit manipulation (bf16 is the high half-word of f32:
   `lo = w << 16`, `hi = w & 0xffff0000`, two integer instructions, not the conversion pipe), the unit diagonal uses a
   predicated select (`x[i] = (i == c) ? 1 : −s`, [`branch_free_select_mask`](branch_free_select_mask.md)), and the
   result column is accumulated in f32 and packed to bf16 **once** at the very end, written back with one `STS.128`.

## When needed (trigger)

- The ledger / contract says "shuffle broadcast + FFMA, `LDS.128`/`STS.128` in place", but the implementation has
  **per-element smem reads embedded in the recurrence** like `s = s + Float32(face[(r0 + i, r0 + j)]) * x[j]`,
  then writes back per element by column with `face[(r0 + i, r0 + c)] = BFloat16(x[i])`;
- SASS of that segment: `LDS.U16` count ≈ NB(NB−1)/2, `STS.U16` ≈ NB, `SHFL.IDX` absent or in single digits;
- NCU: `bank_conflicts_shared_op_ld/st` several times that of an implementation of the same shape (the sub-word
  access fingerprint), `smem_wavefronts_ld` up, `long_scoreboard` concentrated in this segment (sub-word LDS not
  hoisted far enough by ptxas, each exposed before its FFMA);
- Ablation: stubbing out this small block makes the whole kernel much faster while the matrix-instruction segments
  stay the same — the latency tax is all on the chain.

## Mechanism: two ways for a lane to hold data; they differ only in "what is broadcast" and "landing orientation"

A lane always loads **one row of A** (contiguous, `LDS.128`); it always writes back to **the position of its own
row** (contiguous, `STS.128`). So which vector of X a lane holds decides the orientation in which the block lands:

| Variant | lane c holds | What is broadcast at step i / j (from which lane) | Total `SHFL.IDX` NB=8 / 16 | SHFL on the chain? | Lands as |
|---|---|---|---|---|---|
| **column-held** (default) | row c of A (packed) + **column c of X** (f32) | the packed words of row i of A, ceil(i/2) of them (lane i) | **16 / 64** | **no** (depends only on loads) | **X^T** within the block: column c written into row c |
| row-held | row i of A (packed) + **row i of X** (f32) | the j f32 values of the finished row j of X (lane j) | 21 / 105 | yes (step j waits for lane j to finish) | **X** (orientation unchanged) |
| **transposed-system back substitution** (row-held + `X(I+A)=I`) | row c of A (packed) + **row c of X** (f32) | the packed word of row j of A containing `A[j][i]` (lane j, i decreasing, one per (j, word) pair) | **16 / 64** (Σ_{q<NB/2} (NB−1−2q); the probe only certified NB=16) | **no** (depends only on loads) | **X** (orientation unchanged) |

Column-held has the shortest chain and a lossless broadcast (it sends the original bf16 words), at the cost of
**landing the block transposed**; row-held keeps the orientation, at the cost of broadcasts on the chain and twice
as many of them. **The third point combines the strengths of both**: the same equation `(I+A) X = I` also reads
`X (I+A) = I`, i.e. `X[c][i] = δ(c,i) − Σ_{j>i} X[c][j]·A[j][i]` — lane c holds **row c of X** (orientation
unchanged), and the coefficient `A[j][i]` (j > i) needed at step i comes from the row loaded by lane j, **depending
only on the loads, not on the recurrence**, so all SHFLs are off the chain; with i decreasing, lane j's packed word
q = i//2 serves both steps 2q+1 and 2q, so each (j, q) pair is broadcast only once:
for NB=16 that is Σ_q (15 − 2q) = 64 SHFLs, 120 FFMAs, 16 FSELs, 2 `LDS.128` in and 2 `STS.128` out — the same price
as column-held, landing as X. How to choose: if downstream reads this block with matrix instructions (fragments), the
transpose is **free** — just switch to another member in the member table of
[`ldmatrix_stmatrix`](ldmatrix_stmatrix.md) (table below); if downstream is a descriptor / requires the orientation
unchanged and you do not want another move, use row-held.

| Stored X^T, want | Contiguous mode on the face | Member |
|---|---|---|
| A operand `(M, K)` = X | M | `LdMatrix8x8x16bOp(True, n)` (`.trans`) |
| B operand `(N, K)`, `B = X` i.e. `(K, N)` | K | `LdMatrix8x8x16bOp(False, n)` |
| `(M, N)` in C-fragment order = X | M | `LdMatrix8x8x16bOp(True, 4)` |

**In-place safety**: in both variants every lane only overwrites **the row it loaded itself**; whatever other lanes
need from that row has long since been `shfl`ed into their registers and is never fetched from smem again. So no
`sync_warp` is needed within the warp; `shfl.sync` itself is the rendezvous point. Reading this face across warps
still needs a normal barrier.

## Minimal runnable example

A 64×64 bf16 face (may be a 64-wide atom with `K_SW128` swizzle; the 16 B chunk of an `LDS.128` stays contiguous
under the swizzle), warp w handles diagonal blocks `wid*BPW .. wid*BPW+BPW-1`, one column per lane.

```python
# assumed inside the kernel: sF = 64x64 bf16 smem face (swizzled or plain), NB in {8, 16}
lane = cute.arch.lane_idx()
wid  = cute.arch.warp_idx()
NW   = NB // 8                       # 16 B chunks per row = number of LDS.128 / STS.128
BPW  = 32 // NB                      # how many diagonal blocks one warp solves at once
f8   = cute.flat_divide(sF, (8, 8))  # (8, 8, 8, 8): [(r%8, None, r//8, chunk)] = one 16 B chunk of row r
if wid < NBLK // BPW:                # warp-uniform guard; shfl needs the whole warp present
    blk  = wid * BPW + lane // NB
    c    = lane % NB                 # my column
    r0   = blk * NB
    base = lane - c                  # the lane holding column 0 of this block
    rr   = r0 + c                    # the row I load / write

    # 1. resident row: NW LDS.128 -> (8, NW) bf16 -> viewed as (4, NW) 32-bit packed words
    rRow = cute.make_rmem_tensor((8, NW), cutlass.BFloat16)
    for m in cutlass.range_constexpr(NW):
        cute.autovec_copy(f8[(rr % 8, None, rr // 8, r0 // 8 + m)], rRow[(None, m)])
    rW = cute.recast_tensor(rRow, cutlass.Int32)

    # 2. column-held forward substitution: x[i] = (i == c) ? 1 : -Σ_{j<i} A[i][j] x[j]
    x = [cutlass.Float32(0.0)] * NB                      # scalar table with compile-time indices = registers
    for i in cutlass.range_constexpr(NB):
        s = cutlass.Float32(0.0)
        for q in cutlass.range_constexpr((i + 1) // 2):  # the words holding the first i elements of row i
            w = cute.arch.shuffle_sync(rW[(q % 4, q // 4)], base + i)   # source lane differs per lane, legal
            s = s + _bf16_lo(w) * x[2 * q]               # lo = w << 16  (bit op, not F2F)
            if cutlass.const_expr(2 * q + 1 < i):
                s = s + _bf16_hi(w) * x[2 * q + 1]       # hi = w & 0xffff0000
        x[i] = _sel_eq(i, c, cutlass.Float32(1.0), -s)  # setp.eq + selp: unit diagonal, branch-free

    # 3. pack once, NW STS.128 write back my own row -> the block lands as X^T
    rOut = cute.make_rmem_tensor((8, NW), cutlass.BFloat16)
    for i in cutlass.range_constexpr(NB):
        rOut[(i % 8, i // 8)] = cutlass.BFloat16(x[i])
    for m in cutlass.range_constexpr(NW):
        cute.autovec_copy(rOut[(None, m)], f8[(rr % 8, None, rr // 8, r0 // 8 + m)])
```

The row-held variant only replaces step 2 (steps 1 and 3 are word for word the same; lands as X):

```python
    xr = [cutlass.Float32(0.0)] * NB
    for k in cutlass.range_constexpr(NB):
        xr[k] = _sel_eq(k, c, cutlass.Float32(1.0), cutlass.Float32(0.0))   # e_i
    for j in cutlass.range_constexpr(NB - 1):
        w = rW[((j // 2) % 4, (j // 2) // 4)]
        if cutlass.const_expr(j % 2 == 0):               # my A[i][j]: low / high half of word j//2
            a = _bf16_lo(w)
        else:
            a = _bf16_hi(w)
        mlt = _sel_gt(c, j, a, cutlass.Float32(0.0))     # lanes with i <= j multiply by 0, no branch
        for k in cutlass.range_constexpr(j):
            xr[k] = xr[k] - mlt * cute.arch.shuffle_sync(xr[k], base + j)   # take lane j's finished row
        xr[j] = xr[j] - mlt                              # X[j][j] == 1
```

Bit unpacking and select are three-line PTX via `dsl_user_op` + `llvm.inline_asm` (form 3 of [`ptx_escape`](ptx_escape.md)):

```python
@dsl_user_op
def _bf16_lo(w, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(), [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $1, 16;\n\tmov.b32 $0, t;\n}", "=f,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
# _bf16_hi: replace the shl with "and.b32 t, $1, 0xffff0000;"
# _sel_eq / _sel_gt: "setp.eq.s32 p, $1, $2; selp.f32 $0, $3, $4, p;" (constraints "=f,r,r,f,f")
```

The transposed-system back-substitution variant likewise only replaces step 2 (steps 1 and 3 are word for word the
same; lands as X). Lane j's word q holds `A[j][2q]` (low half) and `A[j][2q+1]` (high half); i walks from high to
low, first settling `X[c][2q+1]` with the high half of word q, then multiplying it back into the low half of word q
to settle `X[c][2q]`:

```python
    y = [cutlass.Float32(0.0)] * NB                      # row c of X, held by me
    for qq in cutlass.range_constexpr(NB // 2):
        q = NB // 2 - 1 - qq                              # word index, decreasing
        s_odd = cutlass.Float32(0.0)
        s_even = cutlass.Float32(0.0)
        for t in cutlass.range_constexpr(NB - 2 - 2 * q): # j = NB-1 ... 2q+2
            j = NB - 1 - t
            w = cute.arch.shuffle_sync(rW[(q % 4, q // 4)], base + j)   # A[j][2q], A[j][2q+1]
            s_odd = s_odd + _bf16_hi(w) * y[j]            # depends only on loads: off the chain
            s_even = s_even + _bf16_lo(w) * y[j]
        y[2 * q + 1] = _sel_eq(2 * q + 1, c, cutlass.Float32(1.0), -s_odd)
        w = cute.arch.shuffle_sync(rW[(q % 4, q // 4)], base + 2 * q + 1) # A[2q+1][2q]
        s_even = s_even + _bf16_lo(w) * y[2 * q + 1]
        y[2 * q] = _sel_eq(2 * q, c, cutlass.Float32(1.0), -s_even)
    # step 3 as before: pack y[k] once, STS.128 write back my own row -> the block lands as X
```

**Acceptance (machine-readable)**: compile two kernels with the solve segment compiled out / in via `Constexpr`,
count with `cuobjdump -sass` and take the **difference**:
`ΔLDS.128 ≥ NW`, `ΔSTS.128 ≥ NW`, `Δ(sub-word/scalar LDS, STS) = 0`, `ΔSHFL.IDX = Σ_{i<NB} ceil(i/2)`
(column-held 16 / 64), `Σ_{j<NB-1} j` (row-held 21 / 105) or `Σ_{q<NB/2} (NB−1−2q)` (transposed-system back
substitution, 64 for NB=16), `STL = LDL = 0`. Differential counting keeps the fill / read-out loops from polluting
the criterion; the certification probe computes it exactly this way.

## Gotchas

- **Embedding smem reads in the recurrence**: `s = s + Float32(face[(i, j)]) * x[j]` gives all-correct numbers and
  is invisible to the correctness gate, but the cost is all outside the recurrence: NB(NB−1)/2 sub-word `LDS` + as
  many conversion instructions + sub-word bank conflicts, plus the smem latency exposed before the FFMA by whatever
  loads ptxas could not hoist. These loads **do not depend on x**, so "strung on the chain" is not structural — it
  depends on how far the register budget allows hoisting, and the reading must come from segment timing / ablation,
  not be inferred from counts. Resident rows + broadcast remove all three at once; 4 warps splitting the blocks in
  parallel **does not shorten** the per-lane chain depth.
- **Unpacking with a `Float32(BFloat16)` conversion**: that is `F2F`/`PRMT` through the conversion pipe; bf16 is just
  the high half-word of f32, so two integer instructions `shl` / `and` suffice, and they can dual-issue with FFMA.
- **The broadcast source lane is runtime and differs per lane** (`base + i`) — the source operand of
  `shfl.sync.idx` allows that by design; do not split 32 lanes into two rounds of 16 to get a "constant source".
  But **the whole segment must be inside a warp-uniform guard** (`if wid < K`); a lane-divergent predicate around
  `shfl.sync` hangs or reads undefined values.
- **Switching to per-element column writes on write-back to "keep the orientation"**: that is NB `STS.U16` +
  sub-word bank conflicts, giving back the whole gain of step 3. Swap the orientation back with a member at the next
  matrix-instruction read (table above), or use the row-held variant.
- **The row-held variant packing the broadcast X rows to bf16 first**: saves half the `SHFL`s, but puts a bf16
  rounding into the recurrence, with error accumulating with NB; keep the on-chain broadcast in f32 and pack only
  once at the end.
- **Scalar tables with runtime indices** (`x[c]`, `x[i + off]`): a Python list is registers only under
  **compile-time** indices; runtime indices make the compiler open a local frame (`STL`/`LDL`).
- **Seeing `STL` but no `LDL`**: this is not register pressure (pressure-type spills come as `STL` and `LDL` in
  pairs, often with more `LDL` than `STL`). It is a write-only local frame — the runtime index of the previous
  gotcha, or a temporary copy the compiler puts on the stack when a fragment is written per element and then re-read
  whole by a vector copy. First find in SASS which `setmaxnreg` segment the `STL` lands in and what it writes, and
  change the spelling; the "shrink the live set" ladder does not help with it.
- **Both ends of `autovec_copy` must have static shapes and 16 B alignment**: cut 8-element chunks with
  `flat_divide(sF, (8, 8))` and then slice; feeding the whole 64-element row `sF[(rr, None)]` directly moves the
  entire row. On a swizzled face (`K_SW128` and other 16 B-granularity swizzles) 8 adjacent bf16 are still
  contiguous; when slicing, the swizzle is carried by the iterator.
- **NB and vector width**: NB=8 is exactly one `LDS.128` / one `STS.128`; NB=16 two. NB=32 means 4 per lane, 256
  `SHFL`s and a 496-FFMA chain — at that point you should block it (16×16 diagonal + off-block matrix instructions),
  not push NB up.
- **Shortening the chain does not make the whole kernel faster — first measure whether this segment is on the
  critical path.** A KEEP reading: level 0 switched from "2 warps × 2 blocks per warp, row-held (105 SHFLs on the
  chain)" to "4 warps × 1 block per warp, transposed-system back substitution (64 SHFLs all off the chain, only the
  low 16 lanes of each warp working)"; the probe certified the numbers and the differential counts, but the
  whole-kernel gate window went 0.5482 → 0.5632 ms (**+2.74%, slower**), while in the same eval the spill sectors
  actually dropped, ld 2,621,440 → 1,703,936 / st 2,165,248 → 1,572,864. Two independent variables (recurrence form,
  warp allocation) were mixed in one change, so the reading cannot be split; the only transferable discipline is:
  **the value of a recurrence form depends on whether it is on that role's critical path** — locate it first with an
  ablation (stub out this segment) or a PM timeline, then decide whether to spend this 64-SHFL rewrite. In the same
  eval the dossier's `rings` axis gained an extra bare `stride=0x1` beyond `[stride=0x1,slots=9, stride=0x1,wrap=sel]`
  (the mbarrier source was untouched; it is the address-chain reading that codegen moves along with, see gotcha 12 of
  [`wgmma_desc_uniform`](wgmma_desc_uniform.md)); do not take it as this entry's footprint.

## Sources

- PTX ISA: `shfl.sync` (https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync),
  vector forms of `ld`/`st` (`.v4.b32`, https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld),
  `selp` / `setp` (https://docs.nvidia.com/cuda/parallel-thread-execution/#comparison-and-selection-instructions-selp).
- CUDA C++ Programming Guide, warp shuffle functions (source lane differing per thread, full-warp membermask):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
- CuTe DSL public API: `cute.arch.shuffle_sync(value, offset, mask, mask_and_clamp)`,
  `cute.arch.lane_idx` / `warp_idx`, `cute.autovec_copy`, `cute.recast_tensor`, `cute.flat_divide`,
  `cute.make_rmem_tensor` (older `cute.make_fragment`), `warpgroup.make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, dtype)`
  + `cute.tile_to_shape`, `cutlass.cutlass_dsl.dsl_user_op` + `llvm.inline_asm`.
- This repo's GPU-certified probe (not included in this release) — three numerical contracts R1 (NB=8,
  plain face, column-held), R2 (NB=16, `K_SW128` face, column-held, two blocks per warp), R3 (NB=16, `K_SW128` face,
  row-held) + C1–C3 differential SASS counts + C4 `STL = LDL = 0`. Status: **✅ Verified** (`PROBE PASS max_diff=0.00012207`, certification round 1, R1–R3 + C1–C4 all passed).
- This repo's GPU-certified probe (transposed-system back-substitution variant) —
  B1 (NB=16, `K_SW128` 64×64 face, four diagonal blocks inverted in place, off-block bit for bit unchanged, the
  diagonal blocks equal to `inv(I+A_blk)` rounded once to bf16) + C1 (differential SASS counts: `ΔSHFL.IDX ∈ [64, 66]`,
  `ΔLDS.128 ≥ 2`, `ΔSTS.128 ≥ 2`, no new narrow LDS/STS, `STL = LDL = 0`). Status: **✅ Verified**
  (`PROBE PASS max_diff=0.00012207`, B1 / C1 both passed in the same round, C1 read
  `ΔSHFL.IDX = 64`, `ΔLDS.128 = ΔSTS.128 = 2`, narrow LDS/STS 0, `STL = LDL = 0`; for the KEEP reading when this
  variant later landed in the main loop see the gotcha "Shortening the chain does not make the whole kernel faster").
- Related entries: [`branch_free_select_mask`](branch_free_select_mask.md) (predicated select),
  §2b of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md) (scalar-vector `LDS.64` staging + `shfl` broadcast),
  [`ldmatrix_stmatrix`](ldmatrix_stmatrix.md) (member choice after a transposed landing), [`ptx_escape`](ptx_escape.md).
