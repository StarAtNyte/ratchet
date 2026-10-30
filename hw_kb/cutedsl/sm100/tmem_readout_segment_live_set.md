# tmem_readout_segment_live_set (live-set discipline for a TMEM `Ld32x32b` readout segment inside a register budget: batch by columns, accumulate in place, recompute addresses every round, a pair conversion ends the f32)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**. TMEM readout (`tcgen05.ld`) is a path new in this generation;
> this entry is about **how a segment of t2r readout + element-wise work + narrow-type write-out stays spill-free inside a fixed per-thread
> register budget** — the budget comes from `.reqntid` (`floor(65536 / blockDim)`) or from
> the `setmaxnreg` of [`warp_specialization`](warp_specialization.md); the mechanism is the same for both.
>
> **Status: ✅ Verified** — certified by a GPU probe (the probe is not included in this release):
> `PROBE PASS max_diff=0.0` (third round of this certification; five contracts passed in the same round, see "Sources").
>
> This entry is the **positive spelling** for gotcha 14 of [`warp_specialization`](warp_specialization.md) ("a few `STL`s + then one `LDL`
> at the head of every segment" = long-lived values squeezed out by the peak segment), and it merges, on the "readout segment" side,
> spelling ③ of [`cvt_f32x2_bf16x2_pack_into_r2t_fragment`](cvt_f32x2_bf16x2_pack_into_r2t_fragment.md)
> and constraint ③ of [`tmem_r2t_recast_fragment`](tmem_r2t_recast_fragment.md) (the peak is set by the last consumer of the source
> fragment). Each of those three covers one thing; this entry covers **how the live set adds up when they are combined in one segment of code,
> which term each of the four levers shrinks, and what the machine-readable criterion is**.

## Purpose

One `tcgen05.ld` instruction writes `32 lanes × N columns` of f32 into **N registers** at once (`Repetition.xN`);
until it completes (`tcgen05.wait::ld`, the DSL's `fence_view_async_tmem_load()`) all N registers are live,
and ptxas cannot split it. The register peak of a "readout → element-wise → convert to narrow type → write out" segment is therefore

```
peak(segment) = Σ|readout fragments in flight at once|      ← set by how many LDTMs you issue before waiting
              + |second / third fragments live alongside|   ← the half that came back from the exchange, a "final value" copy, a bf16 copy
              + |branch-level long-lived values|            ← hoisted addresses / row scalars / loop state / mbarrier base
              + addresses and temporaries                   ≤ budget − headroom
```

The budget is a constant; of the four terms only the first three are written by you. Once over, ptxas does **not** error; it moves values to local memory
in order of lowest cost: first **the third term** (store once, reload at each use — footprint: one `LDL` at each segment head),
then whole fragments (footprint: a run of `STL` followed by the same number of `LDL`). Non-zero `spill_ld / spill_st`,
and distorted `stall_long_scoreboard` and `tensor_pct`, are all downstream of this one peak.

## When needed (trigger)

**Structural trigger**: a role has a "TMEM readout → element-wise → narrow-type write-out" segment, and this role's budget
(the `.inc` value of `setmaxnreg`, or `floor(65536 / blockDim)`) minus the long-lived values it must keep resident
is not enough to hold all the fragments you currently write at the same time.

**Symptom trigger** (after the SASS census is cut by the region method of [`warp_specialization`](warp_specialization.md)):

| Symptom | Hit |
|---|---|
| All spills land in **one** `.inc` region; `CUTE_DSL_LINEINFO=1` attribution: `STL` concentrated at the hoist statements before the segment (exchange-face addresses, row scalars), `LDL` at the segment head **every round** | The third term is squeezed out by the first / second (gotcha 14's footprint) — shrink the peak segment, do not touch the budget |
| Spilled `STL` and `LDL` counts equal, all on the narrow-type conversion lines, count ≈ elements per thread | A whole fragment is moved to local at the narrow-type copy: the conversion spelling has degraded (`F2FP == N`) or the bf16 copy is live together with the f32 source |
| `F2FP` count == elements per thread (not / 2) | Element-wise `.to(bf16)` (spelling ① of the cvt entry) — switch to pair conversion |
| In one round, `LDTM` count == number of chunks, all ordered before the first `FMUL/FFMA` | All chunks read before computing: first term = all columns |
| The segment contains writes to ≥ 3 equal-length fragments at the same time (the readout, the exchanged, the summed) | Second term: the sum goes into a new fragment instead of in place |
| The budget has already been tried ±8, Σ at the cap, and the spill is still in the same region | Not a split problem; this entry |
| An exchange segment in one role: the two halves' `LDTM` back to back sharing one wait, a runtime `if` choosing which half to send, the sum into a new list/fragment, **and** the role's budget not raised with `setmaxnreg`; batching alone or raising the budget alone, each measured once, gives ≈ 0 gain | The "combined form" subsection: batching × budget raise is a ⊗ bundle, land them together |

**Anti-trigger**: spills in a `.dec` region or in the merge region after the branches → `warp_specialization` gotcha 11;
`USETMAXREG == 0` → gotcha 1 there. This entry covers only spills **within one claiming region, caused by the readout segment itself**.

## Minimal runnable example

A self-contained readout segment (the `a_lever` / `b_lever` kernels of the certification probe are exactly this segment),
128 threads each owning one TMEM row. Two forms:

- **Form A (half-row exchange)**: one row of 64 f32 columns; thread `t` is responsible only for the final values of 32 of them, the other 32
  belong to partner thread `t ^ 64` (another warp); each side holds a partial sum, and they are added through an smem exchange face.
- **Form B (chunked write-out)**: one row of 256 f32 columns, read back in 64-column chunks, multiplied by a row scalar, converted to bf16, written out.

Each of the four levers is marked in the code: **(1) batch by columns**, **(2) accumulate in place**, **(3) recompute addresses every round**,
**(4) a pair conversion ends the f32**.

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutils
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

F32, BF16, I32 = cutlass.Float32, cutlass.BFloat16, cutlass.Int32
M, NS, HALF = 128, 64, 32        # form A: 128 rows × 64 columns, the thread keeps 32 columns
NO, CH = 256, 64                 # form B: 128 rows × 256 columns, 64-column chunks
LANE = 65536                     # TMEM lane stride, in 32-bit columns
BAR = 1                          # named barrier of the working warpgroup


@dsl_user_op
def cvt_bf16x2_f32(lo, hi, *, loc=None, ip=None) -> cutlass.Int32:
    """(lo, hi) -> one bf16x2 word, lo in the low half (spelling (a) of the cvt entry)."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [F32(lo).ir_value(loc=loc, ip=ip), F32(hi).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def opaque_i32(x, *, loc=None, ip=None) -> cutlass.Int32:
    """An identity the optimizer cannot see through: values derived from its result inside a loop are recomputed every round, not hoisted into resident registers."""
    return cutlass.Int32(llvm.inline_asm(
        T.i32(), [I32(x).ir_value(loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def flat(t, n):
    return cute.make_tensor(t.iterator, cute.make_layout(n))


def v4(t, ncol):                 # (4, ncol) view: column c = elements 4c..4c+3 = one 16 B vector
    return cute.make_tensor(t.iterator, cute.make_layout((4, ncol), stride=(1, 4)))


# ---- form A: written inside the @cute.kernel body, in the working warpgroup's branch (tmem_ptr / sX already built) ----
#      static loops always cutlass.range_constexpr, compile-time switches always cutlass.const_expr (gotcha 13)
def segment_a(tmem_ptr, sX, gInv, gOut, n_rounds, tidx, warp):
    h = cute.arch.make_warp_uniform(warp // 2)     # which 32 columns this warp keeps: a warp-uniform value
    key = tidx % 8                                 # XOR key of the exchange face's 16 B slots
    xb = (tidx // 32) * (32 * HALF) + (tidx % 32) * HALF          # own 128 B segment
    pb = ((tidx ^ 64) // 32) * (32 * HALF) + (tidx % 32) * HALF   # the partner's
    xw = v4(cute.make_tensor(sX.iterator + xb, cute.make_layout(HALF)), 8)
    xr = v4(cute.make_tensor(sX.iterator + pb, cute.make_layout(HALF)), 8)
    rInv = cute.make_rmem_tensor((8,), F32)        # long-lived row scalars (third term)
    for j in cutlass.range_constexpr(8):
        rInv[j] = gInv[(tidx, j)]

    # (1) batch by columns: two (M, 32) column sub-views, the offset a warp-uniform runtime value;
    #     the tiled copy is built from a static view, then used to partition the dynamic views. Offset unit is 32-bit columns.
    ld32 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), F32)
    half_layout = cute.make_layout((M, HALF), stride=(LANE, 1))
    tLo = cute.make_tensor(tmem_ptr, half_layout)
    tOwn = cute.make_tensor(tmem_ptr + h * HALF, half_layout)         # the half that is kept
    tOth = cute.make_tensor(tmem_ptr + (h ^ 1) * HALF, half_layout)   # the half that is handed over
    t2r = tcgen05.make_tmem_copy(ld32, tLo)
    thr = t2r.get_slice(tidx)
    src_own, src_oth = thr.partition_S(tOwn), thr.partition_S(tOth)
    rOwn = cute.make_rmem_tensor(thr.partition_D(tLo).shape, F32)     # 32 f32
    rOth = cute.make_rmem_tensor(thr.partition_D(tLo).shape, F32)     # 32 f32, short-lived
    rOwnF, rOth4 = flat(rOwn, HALF), v4(rOth, 8)
    rT = cute.make_rmem_tensor((4,), F32)                             # takes only 4 at a time
    rW = cute.make_rmem_tensor((HALF // 2,), I32)                     # 16 packed words
    rSc = cute.make_rmem_tensor((1,), F32)
    rSc[0] = F32(1.0)
    for r in cutlass.range(n_rounds, unroll=1):
        kr = opaque_i32(key)                       # (3) this round's address seed
        # (1a) the half handed over: read, send, dead — its 32 registers end before the kept half is consumed
        cute.copy(t2r, src_oth, rOth)
        cute.arch.fence_view_async_tmem_load()
        for v in cutlass.range_constexpr(8):
            cute.autovec_copy(rOth4[(None, v)], xw[(None, kr ^ v)])   # addresses derived from kr
        # (1b) the kept half is in flight while the partner's half lands
        cute.copy(t2r, src_own, rOwn)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.barrier(barrier_id=BAR, number_of_threads=M)
        # (2) accumulate in place: the partner's half is added into own registers 4 at a time, no third fragment
        for v in cutlass.range_constexpr(8):
            cute.autovec_copy(xr[(None, kr ^ v)], rT)
            for i in cutlass.range_constexpr(4):
                rOwnF[4 * v + i] = rOwnF[4 * v + i] + rT[i]
        cute.arch.barrier(barrier_id=BAR, number_of_threads=M)    # the next round may write the face only after this round's reads
        sc = rSc[0]
        for j in cutlass.range_constexpr(HALF):                      # element-wise also in place
            rOwnF[j] = rOwnF[j] * sc + rInv[j % 8]
        # (4) pair conversion: one f32 pair → one 32-bit word; the pair of f32 ends here
        for k in cutlass.range_constexpr(HALF // 2):
            rW[k] = cvt_bf16x2_f32(rOwnF[2 * k], rOwnF[2 * k + 1])
        for k in cutlass.range_constexpr(HALF // 2):
            gOut[(r, tidx, k)] = rW[k]
        rSc[0] = sc * F32(0.5)


# ---- form B: chunked write-out (tO[k] is the TMEM view of the k-th (M, 64) column chunk) ----
def segment_b(tO, gInvB, gOut, n_iter, tidx):
    ld64 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), F32)
    t2r = tcgen05.make_tmem_copy(ld64, tO[0])
    thr = t2r.get_slice(tidx)
    src = [thr.partition_S(tO[k]) for k in range(NO // CH)]   # partitions hoisted into a Python list
    rO = cute.make_rmem_tensor(thr.partition_D(tO[0]).shape, F32)   # only one chunk in hand
    rOF = flat(rO, CH)
    rW = cute.make_rmem_tensor((CH // 2,), I32)
    inv = gInvB[tidx]
    for it in cutlass.range(n_iter, unroll=1):
        for k in cutlass.range_constexpr(NO // CH):
            cute.copy(t2r, src[k], rO)             # (1) read one chunk
            cute.arch.fence_view_async_tmem_load()
            for e in cutlass.range_constexpr(CH):
                rOF[e] = rOF[e] * inv              # in place
            for w in cutlass.range_constexpr(CH // 2):
                rW[w] = cvt_bf16x2_f32(rOF[2 * w], rOF[2 * w + 1])   # (4)
            for w in cutlass.range_constexpr(CH // 2):
                gOut[(it, tidx, (CH // 2) * k + w)] = rW[w]         # issue the next LDTM only after the write-out
```

Discipline at the call level (all certified by other entries; this only says why this entry depends on them):

- Static loops inside a `@cute.kernel` body are **always** `cutlass.range_constexpr`: a bare `range(n)` lowers to an IR loop even when `n` is
  a Python constant, and the induction variable becomes a staged value — indexing a Python list with it (`src[k]`) raises
  `PHASE_DYNAMIC_INDEX` at trace time, and indexing a fragment with it (`rF[i]`) becomes a runtime register index that sinks the whole array into
  local (`warp_specialization` gotcha 7). Compile-time switches (form / spelling selection) are always
  `if cutlass.const_expr(...)`: a bare `if` is a runtime region, and names first bound inside one of its branches (sub-views, tiled copies)
  cannot leave the region ([`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md)).
- TMEM allocation goes through `cutlass.utils.TmemAllocator` (`allocate` / `wait_for_alloc` / `retrieve_ptr` /
  `relinquish_alloc_permit` / `free`, retrieved with a whole-CTA `pipeline.NamedBarrier`), single CTA
  passes `is_two_cta=False`; with a single budget (no `setmaxnreg`) `free` can be called by all threads after an all-thread `cute.arch.barrier()` at the
  end of the segment. **Do not finish like this when there are `setmaxnreg` role branches**: TMEM is freed by the role holding it at the end of its own
  branch body (after this warpgroup's named barrier, one warp calls
  `cute.arch.dealloc_tmem(tmem_ptr, ncols, is_two_cta=False)`), and there is no code at all after the branches merge
  (see the "merge tail" in the combined-form subsection below).
- Partitions (`partition_S/D`) are built into a Python list outside the loop; inside the loop there is only `cute.copy` — calling methods inside a region on a
  TiledCopy / ThrCopy bound outside threads it into the region's operand bundle.
- The offset unit of readout column sub-views is **32-bit columns** (the pointer-arithmetic rule of `tmem_r2t_recast_fragment`).

### Machine-readable criterion (used for both certification and regression)

```
LDL == STL == 0, LOCAL == 0, STACK == 0   in the .inc region that carries this segment (region method: see the warp_specialization acceptance)
F2FP.BF16.F32.PACK_AB count == elements written out per thread / 2     (== element count ⇒ degraded spelling ①)
LDTM count per round == the number of batches you intend to have in flight at once (form A: 2; form B: number of chunks, each LDTM immediately followed by its own consumer)
REG from cuobjdump -res-usage is the true live-set reading of the segment: it should be clearly below the budget; the headroom is where the third term can live
```

### Yield ladder (sorted by expected gain; recount after each rung before taking the next)

| Rung | Action | Term it shrinks | Expected gain |
|---|---|---|---|
| r1 | Batch the readout by columns (`x64` → 2 × `x32`, or number-of-chunks × `x64` one chunk at a time); read, use and write out each batch before issuing the next | first | Peak drops from "all columns" to "one batch"; the only lever that can bring the peak below `|fragment|` |
| r2 | Write the exchange / reduction result **in place** into the source fragment; no third copy like `rF = rS + rR` | second | Saves a whole fragment |
| r3 | Pair conversion (spelling ③ of the cvt entry), an f32 pair dies once converted; no whole-row bf16 copy moved as a block | second | Saves the `N/2`-register bf16 copy + lets the f32 source end pair by pair |
| r4 | Slot addresses of the exchange face / write-out face recomputed every round from the lane key (`opaque_i32` seed), not hoisted into 16 resident registers | third | Costs 2 integer instructions / slot |
| r5 | Budget ±8 (`warp_specialization` self-chosen split) | budget | Meaningful only if after r1–r4 you are still short by ≤ 8 |

**Actions not on the ladder**: making the write-out / store finer (`tmem_r2t_recast_fragment` gotcha "store granularity is neutral";
the source fragment's live range is not set by store granularity); changing the fragment to bf16 element type and writing element-wise
(spelling ① of the cvt entry, `F2FP == N`); "read all chunks first to hide LDTM latency" (see gotcha 8).

### Combined form: half-row exchange × `setmaxnreg` re-split is one ⊗ bundle (r1 + budget raise landed together)

> Status: ✅ Verified — a second probe:
> `PROBE PASS max_diff=0.0` (third round of this certification; six contracts passed in the same round, see "Sources").

**When the two must go in together.** When the segment's resident load (the third term: row scalars, loop state and per-round values that stay live across rounds)
is itself close to the budget, **neither lever alone reaches spill == 0**; only both together do:

```
Let  H  = columns kept per thread (a row has 2H columns, the other H go to partner t ^ 64)
     NL = number of values resident across rounds, s = scalars such as addresses / loop counters (a dozen or so)
     R0 = uniform budget (.reqntid 512 → 128), R↑ = this role's setmaxregister_increase value

no batching (two halves' LDTM back to back, one wait): peak ≥ 2H + NL + s
batching only (r1)                                   : peak ≈  H + NL + s

r1 only        : H + NL + s  > R0   → still spills (resident values reloaded by LDL at the segment head)
budget R↑ only : 2H + NL + s > R↑   → still spills (both halves in flight squeeze out the resident values)
r1 + R↑        : H + NL + s  ≤ R↑   → spill == 0
```

This step is not the ±8 fine-tuning of rung r5: r5 covers "short by ≤ 8 after r1–r4"; here the budget gap is of the same order as a whole H-wide
fragment, and r1 cuts exactly one H off it (no contradiction with gotcha 11: chasing with the budget alone still cannot catch up).

So tuning one of them alone and measuring "no gain, even slightly negative" **does not mean that lever is useless** — it is one factor in
a product. Criterion: first compute the peak of the two single-lever forms with the inequalities above; if both are > their budgets, submit the two as one patch and
measure them together. The certification probe compiled all four corners with `H = 48, NL = 80, R0 = 128, R↑ = 176` (see "Sources").

**How to re-split the budget (Σ unchanged).** The pool of a 512-thread CTA is `128 × 512 = 65536` (entry budget 128 divides it exactly);
giving one warpgroup `+Δ` means taking a total of `Δ` from the other warpgroups' `setmaxregister_decrease`:
`Σ_wg budget_wg ≤ 4 × 128 = 512`, each budget a multiple of 8 in [24, 256]. The yielding side's budget must fit
its own peak (if the yielding side itself spills, the problem has only moved house — count LDL/STL separately per `USETMAXREG` region using the region method of
[`warp_specialization`](warp_specialization.md)). The four-piece kit (in-kernel dec/inc,
`launch(min_blocks_per_mp=1)`, dec/inc as the first statement of each role's branch body, shared prologue before the branches)
is all in the `warp_specialization` entry and not repeated here.

**The batching side's spelling (same idea as form A, with an `x16` atom covering an H = 48 wide column view):**

```python
# in the role branch, inc is the first statement; the other warpgroups' branches start with dec
if wg == SEG_WG:
    cute.arch.setmaxregister_increase(REG_HI)          # e.g. 176
    tmem_ptr = tmem.retrieve_ptr(F32)                  # the prologue only does allocate / wait_for_alloc; the base is read after inc
    ...
    h = cute.arch.make_warp_uniform(warp_in_wg // 2)   # which half this warp keeps: warp-uniform
    half = cute.make_layout((128, H), stride=(65536, 1))
    tLo = cute.make_tensor(tmem_ptr + S_COL, half)                     # static view: builds the tiled copy
    t2r = tcgen05.make_tmem_copy(
        cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), F32), tLo)
    thr = t2r.get_slice(t)
    src_keep = thr.partition_S(cute.make_tensor(tmem_ptr + S_COL + h * H, half))
    src_send = thr.partition_S(cute.make_tensor(tmem_ptr + S_COL + (h ^ 1) * H, half))
    rSend = cute.make_rmem_tensor(thr.partition_D(tLo).shape, F32)
    rKeep = cute.make_rmem_tensor(thr.partition_D(tLo).shape, F32)
    # exchange face float4 at [slot q][thread t]: slot address = one base + immediate, no address set to hoist
    # sX = smem.allocate_tensor(F32, cute.make_layout((4, 128, H // 4), stride=(1, 4, 4 * 128)))
    for r in cutlass.range(n_rounds, unroll=1):
        cute.copy(t2r, src_send, rSend)                # ① the half handed over: read
        cute.arch.fence_view_async_tmem_load()         #    wait
        for q in cutlass.range_constexpr(H // 4):      #    send — rSend dies here
            cute.autovec_copy(v4(rSend)[(None, q)], sX[(None, t, q)])
        cute.arch.barrier(barrier_id=BAR, number_of_threads=128)   # partner has finished writing; also a scheduling fence
        cute.copy(t2r, src_keep, rKeep)                # ② the kept half: issued only after the barrier
        cute.arch.fence_view_async_tmem_load()
        for q in cutlass.range_constexpr(H // 4):      # ③ partner's partial sums added into rKeep in place, 4 at a time
            cute.autovec_copy(sX[(None, t ^ 64, q)], rT)
            for e in cutlass.range_constexpr(4):
                rKeepF[4 * q + e] = rKeepF[4 * q + e] + rT[e]
        cute.arch.barrier(barrier_id=BAR, number_of_threads=128)   # gotcha 9: the next round may write only after the reads
        for k in cutlass.range_constexpr(H // 2):      # ④ end in place: each pair, once computed, converted to bf16x2 and written out
            a = f(rKeepF[2 * k]);  b = f(rKeepF[2 * k + 1])
            out[k] = cvt_bf16x2_f32(a, b)
else:
    cute.arch.setmaxregister_decrease(REG_LO)          # e.g. 112 (176 + 3 × 112 = 512)
```

Point by point against the "no batching" spelling:

| No-batching spelling (both need changing) | Spelling in the bundle |
|---|---|
| `if h == 0: copy(S1→rSend); copy(S0→rKeep) else: …`, then **one** `fence` | Column offsets `h * H` / `(h ^ 1) * H` folded into the TMEM views, no runtime branch (gotcha 2); the send `fence` comes **before** the keep LDTM (gotcha 1) |
| `x = [rKeep[i] + rv[i] …]` (a new third copy as a Python list) | `rKeepF[i] = rKeepF[i] + rT[e]` in place |
| compute the whole row first, then pack / write out together | per-pair `cvt.rn.bf16x2.f32` ends it (r3) |
| budget 128 (`.reqntid` even split) | this role `inc(R↑)`, the others `dec`, Σ unchanged |

**A merge tail leaves a cross-budget spill.** If there is code after the role branches merge (e.g. an all-thread `barrier()` + `tmem.free(tmem_ptr)`),
prologue values like `tmem_ptr` must stay live through both the "inc region" and "dec region" paths up to the merge point; ptxas handles it by
`STL`-ing once in the **shared prologue** and `LDL`-ing once in each branch (region-method reading: STL=1 before TRY_ALLOC, LDL=1 in the inc region,
LDL=1 after dec, `STACK=8`) — the bundle itself has no pressure any more, yet these 2+1 instructions break `LDL == STL == 0`.
Fix: leave no tail after the branches merge, free TMEM at the end of the holding role's branch body; and **do not take `tmem_ptr` in the prologue** —
the prologue only does `allocate` / `wait_for_alloc` / `relinquish_alloc_permit`, and `retrieve_ptr` goes after inc
(it is just one smem read). With only the tail removed and `retrieve_ptr` still in the prologue, the reading is "STL=1 before TRY_ALLOC,
LDL=2 in the inc region, `STACK=8`": the base address lives across the `setmaxnreg` boundary into the high-pressure segment and is spilled as the longest-lived value in the segment.

**Put that named barrier between the send STS and the keep LDTM.** `fence_view_async_tmem_load` only waits on LDTMs in flight;
it does not stop ptxas from scheduling the later `tcgen05.ld` above the send STS (the two have no data dependence), and once
it is moved up, `rSend` and `rKeep` are live together, the peak returns to `2H + NL + s`, and the bundle is wasted. Move the "partner finished writing"
barrier between the two: semantics unchanged (after the keep half is read back it can still read the partner's slots), and `bar.sync` is a scheduling fence that ptxas will not let
`tcgen05.ld` cross.

```python
if wg == SEG_WG:
    cute.arch.setmaxregister_increase(REG_HI)
    ...                                             # the segment
    cute.arch.barrier(barrier_id=BAR, number_of_threads=128)   # all LDTMs of this WG have been waited on
    if warp == SEG_WG * 4:
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLS, is_two_cta=False)
else:
    cute.arch.setmaxregister_decrease(REG_LO)
    ...
# nothing here
```

**Acceptance (did the bundle land)**: in the same cubin `LDL == STL == 0`, `LOCAL == 0`, `STACK == 0`, `USETMAXREG ≥ 2`
(at least one inc and one dec), `F2FP` all `PACK_AB` and **per round body** `F2FP == H / 2`. Note that ptxas may emit
the round body of `cutlass.range(n, unroll=1)` twice in SASS (measured by the certification probe: the bundle form has `LDTM=6`, `F2FP=24`,
exactly one round body; the raise-only form emits `LDTM=12` from the same source, with one round being `2H/16 = 6` LDTMs and 24 F2FPs) —
so first count the number of round-body copies as `LDTM ÷ LDTM per round`, then divide;
comparing the whole-kernel `F2FP` directly against `H/2` misjudges; then confirm with the `warp_specialization` region method that the yielding
region is also 0. Each single-lever form should have `LDL + STL > 0` at the same sizes — without that pair of "negative controls"
you cannot tell "the bundle worked" from "there was never any pressure".

## Gotchas

1. **`fence_view_async_tmem_load()` waits for all in-flight `tcgen05.ld`, not the most recent one.** Issue both readout batches
   and then wait, and at the moment the wait returns both batches are live — batching by columns presupposes "read one batch, consume it, then issue the next";
   in form A the half handed over must be sent off before the kept half's LDTM. Two LDTMs back to back + one fence,
   and the first term is still 64.
2. **Choose "which half is mine" with a column offset, not with a runtime branch slicing a 64-wide fragment.** In the form `if h == 0: use rS[0:32]
   else: use rS[32:64]`, all 64 registers are live before the branch, and after the branch ptxas cannot merge the "final values"
   back into one half either (the two paths have different source registers) — the result is three copies coexisting, 64 + 32 + 32. Fold `h` into the
   TMEM column offset (`tmem_ptr + h * 32`) and the fragment is only 32 wide, with no branch.
3. **The column sub-view offset must be warp-uniform and is in 32-bit columns.** `tcgen05.ld` is a warp-collective instruction;
   the address register is per thread but must be uniform across the warp; values derived from `warp_idx` go through `make_warp_uniform`
   before entering pointer arithmetic. Offsetting by element count (number of bf16) steps into the neighbouring column chunk, and TMEM reports no out-of-bounds.
4. **Build the tiled copy from a static view, then `partition_S` the dynamic view.** The two views have the same layout and differ only in
   the runtime offset of the iterator; the constructor looks only at the layout, the partition only carries the iterator. Handing the dynamic view directly to the constructor
   also works on some DSL builds, but there is no need to gamble.
5. **LICM hoists loop-invariant slot addresses into resident registers.** `xw[(None, key ^ v)]` with eight static `v` and
   `key` loop-invariant — the compiler precomputes 8 write addresses + 8 read addresses into 16 registers that live across the whole loop;
   under quota pressure they are exactly what ptxas spills first (gotcha 14 footprint: one `LDL` at each segment head). Derive the addresses from `kr`
   after `kr = opaque_i32(key)` every round, and they become temporaries recomputed every round.
6. **Accumulate in place 4 at a time.** Read all 32 of the partner's values into `rR` first and then add one by one, and `rR` is fully live (32) before the first addition;
   read 4 (one `LDS.128`) and add 4, and the second term is 4.
7. **`rP[j] = rF[j].to(BF16)` into a bf16 rmem tensor does not "end the f32".** Element-wise half-register
   writes under quota pressure are `F2FP == N` + `PRMT` (the certified boundary of the cvt entry), and `rF` stays live as a whole row until after the last
   `.to`; `rP` itself takes another `N/2` registers. Pair conversion produces 32-bit words directly, and both problems disappear together.
8. **"Issue all chunks' LDTMs first, then compute" is not latency hiding; it is the first term × number of chunks.** Number of chunks × 64 registers
   all live on one `wait::ld`; at budget 128, 4 chunks = 256 registers must spill (the certification probe's
   `allchunks_control_spills` contract pins exactly this). LDTM latency is filled by the multiply / pack / write-out of **the previous chunk**:
   after a chunk's `LDTM` is issued, what follows is its own consumer, not the next `LDTM`.
9. **The exchange face needs two barriers per round.** First: I may read only after the partner has written; second: the partner may overwrite the same slot in the next round
   only after I have read. Missing the second raises no error and gives intermittent wrong numbers (the next round's write overwrites a slot this round has not finished reading).
   Saving this barrier needs a double-buffered exchange face, i.e. trading smem for a barrier, which is outside this entry.
10. **REG below the budget does not mean there is enough headroom.** The `REG` of `-res-usage` is the cap ptxas reports at **entry**, not the segment's
    peak; the segment peak can only be read from the two facts "`LDL == STL == 0` and `REG < budget`" together. With `LDL` present,
    `REG` always sits at the budget — that is not information.
11. **Do not chase a whole-fragment gap with the budget.** The budget moves in steps of 8 registers; one `x64` chunk is 64.
    When the spill (`LDL + STL` count / round) is ≥ 16, a self-chosen split cannot bring it back to 0; do r1–r4 first.
12. **`STACK`/`LOCAL` non-zero with `LDL == 0`**: not a spill; some rmem array has been demoted to local as a whole by a runtime index
    (`warp_specialization` gotcha 7); make the index static.
13. **When this entry's code is written inside a kernel body, two DSL syntax rules stop you before any register problem.**
    (a) A bare `for … in range(constant)` inside `@cute.kernel` is an IR loop, not unrolled — Python-list indexing like `src[k]`, `tO[k]`
    raises `PHASE_DYNAMIC_INDEX` at trace time, and fragment indices silently become runtime
    indices (gotcha 12's footprint); write `cutlass.range_constexpr`. (b) A form / spelling switch with a bare `if` opens a
    runtime region, and `tOwn` / `t2r` first bound inside it raise `NameError` outside the region; write
    `if cutlass.const_expr(...)`. Only a genuine runtime condition (like `if h == 0:`) stays a bare `if` — and that is exactly the form
    gotcha 2 tells you to avoid.

14. **A single lever's "zero gain" is not a veto.** When the resident load NL is close to the budget, batching only (peak = H + NL + s > 128)
    and raising the budget only (peak = 2H + NL + s > R↑) both still spill — each measuring ±0.x% is inevitable and does not show the lever is useless.
    First compute the four corners with the inequalities of the "combined form" subsection; when both single-lever corners are > budget, submit the two as one patch.
    Conversely: if after the bundle lands `spill == 0` but LDL/STL appear in the yielding region, `dec` was too aggressive; give the yielding side back
    some budget (Σ still ≤ 512 / number of warpgroups × 128).
15. **An exchange face laid out as `[slot][thread]` needs no address seed.** Slot address = one base register + immediate
    (`16 × (q × 128 + t)` bytes), no 8 / 16 XOR addresses to hoist; lever (3) is needed only for the layout "one segment per thread,
    slot index XORed with the lane key". Consecutive lanes access consecutive 16 B, with no bank conflicts by construction.

## Sources

- **PTX ISA**, chapter "TensorCore 5th Generation Instructions", `tcgen05.ld` / `tcgen05.st`
  (the `.32x32b` shape, `.x32` / `.x64` repetition, number of destination registers = repetition, `tcgen05.wait::ld` waits for
  **all** outstanding `tcgen05.ld`) and the `cvt` section (`cvt.rn.bf16x2.f32 d, a, b`)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- **CuTe DSL / CUTLASS public API symbols**: `cutlass.cute.nvgpu.tcgen05.Ld32x32bOp / St32x32bOp /
  Repetition.x32 / Repetition.x64 / Unpack.NONE / make_tmem_copy`, `cute.make_copy_atom`,
  `cute.make_tensor(iterator, layout)`, `cute.make_rmem_tensor`, `cute.autovec_copy`,
  `cutlass.utils.TmemAllocator` (`allocate / wait_for_alloc / retrieve_ptr / relinquish_alloc_permit /
  free`), `cutlass.pipeline.NamedBarrier`, `cutlass.utils.SmemAllocator`, `cute.struct`,
  `cute.arch.fence_view_async_tmem_load / fence_view_async_tmem_store`,
  `cute.arch.make_warp_uniform`, `cute.arch.barrier(barrier_id, number_of_threads)`,
  `cutlass.range(n, unroll=1)`, `cutlass.range_constexpr`, `cutlass.const_expr`,
  `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`.
- **Build-artifact observation channel**: `CUTE_DSL_KEEP_CUBIN=1` + `CUTE_DSL_KEEP_PTX=1` + `CUTE_DSL_DUMP_DIR`,
  `nvdisasm -c` / `cuobjdump -sass` to count `LDL` / `STL` / `F2FP` / `PRMT` / `LDTM`,
  `cuobjdump -res-usage` to read `REG` / `LOCAL` / `STACK`; the region method and `CUTE_DSL_LINEINFO=1` line attribution: see the
  [`warp_specialization`](warp_specialization.md) acceptance.
- **GPU certification probe** — **Status: ✅ Verified**,
  `PROBE PASS max_diff=0.0` (third round of this certification). Measured footprints: form A lever form
  `LDL=0 STL=0 LOCAL=0 STACK=0 REG=118 F2FP=16 PACK_AB=16 LDTM=2`; form B batched form
  `LDL=0 STL=0 LOCAL=0 STACK=0 REG=112 F2FP=128 PACK_AB=128 LDTM=4`; negative contract "read all chunks first, then compute"
  `LDL=85 STL=86 STACK=384 REG=128` (same budget 128, same LDTM=4, only the issue order differs).
  512-thread CTA (`.reqntid 512` → budget 128), the working warpgroup's 128 threads each owning one row; each kernel in
  its own subprocess + its own dump directory. PASS only if all five contracts pass:
  `exchange_readout_bits` (form A lever form: every round, each thread's 16 packed words are bit-equal to a host RNE reference,
  proving the dynamic column offset reads the right half row and in-place accumulation and pair conversion are correct);
  `exchange_readout_footprint` (same kernel: `LDL == STL == 0`, `LOCAL == STACK == 0`,
  `F2FP == 16`);
  `chunked_readout_bits` / `chunked_readout_footprint` (form B lever form: bit-equal;
  `LDL == STL == 0`, `F2FP == 128`);
  `allchunks_control_spills` (form B control: four `LDTM`s issued first, then waited, at the same budget
  `LDL + STL > 0` or `LOCAL > 0` — first proving the budget really cannot hold 256 registers).
  Additional report-only fact lines: the footprint and `REG` of the form A control (`x64` + runtime branch to choose the half + third fragment + hoisted addresses +
  element-wise `.to(bf16)`), the footprint of the form B element-wise `.to(bf16)` form, and which of the two column sub-view spellings
  (pointer arithmetic / three-mode slicing) compiles.
- **GPU certification probe (combined-form subsection)** —
  **Status: ✅ Verified**, `PROBE PASS max_diff=0.0` (third round of this certification). Measured footprints
  (four corners at the same sizes, `REG=128` being the entry budget): bundle form `LDL=0 STL=0 LOCAL=0 STACK=0 USETMAXREG=4
  F2FP=24 PACK_AB=24 LDTM=6`, both inc / dec regions `LDL0/STL0`; batching only `LDL=44 STL=26 STACK=184
  USETMAXREG=0`; raise only `LDL=24 STL=12 STACK=96 USETMAXREG=4`, all spills in the segment after the yielding side's `dec` region
  (inc region 0); neither `LDL=64 STL=36 STACK=256`. Values of all four corners bit-equal (12288 words,
  0 mismatches). 512-thread CTA, 4 warpgroups, one warpgroup carrying the exchange segment (H = 48, per thread
  NL = 80 values resident across rounds), four corners compiled as 2 × 2: {batched, not batched} × {`inc(176)` + three `dec(112)`,
  uniform 128}, launch always `min_blocks_per_mp=1`. Contracts: `bundle_bits` (bundle form bit-equal to the host RNE
  reference), `bundle_footprint` (`LDL == STL == 0`, `LOCAL == 0`, `USETMAXREG ≥ 2`, `F2FP` all `PACK_AB` and 24 per round body;
  TMEM freed by `dealloc_tmem` inside the segment role's branch, no tail after the merge),
  `batch_only_bits` / `batch_only_spills` (batching only: values right, `USETMAXREG == 0` and spills),
  `regs_only_bits` / `regs_only_spills` (raise only: values right, `USETMAXREG ≥ 2` and spills).
