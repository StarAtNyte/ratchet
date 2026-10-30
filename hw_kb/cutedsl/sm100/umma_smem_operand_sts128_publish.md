# umma_smem_operand_sts128_publish (a bf16 UMMA operand computed in registers: hand-written STS.128 into a K-major swizzle face → async-proxy publish → read by `tcgen05.mma`)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> Applies to this form: the UMMA A (or B) operand is not brought in by TMA but **computed in registers** by a warpgroup
> (f32 → bf16x2 pairwise conversion); each thread holds several 16 B chunks of one row (or half a row), writes them with hand-written `st.shared.v4.b32`
> into a K-major swizzle face built by `make_smem_layout_a/b`, then hands them to the MMA-issuing warp via an mbarrier.
> The face is single-buffered (each round the face-writing role first waits for `tcgen05.commit` to release the face).
>
> **Status**: ✅ Verified (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=0.0`, round 2 of this certification; six contracts all passed in the same round: for `face_sw64` / `face_sw128` the granule-register addresses equal the inline-formula addresses exactly; for three UMMA variants D and the row sums are bit-equal to numpy; `footprint_stream_granule_sw64` STS128=12, LDL=0, STL=0)

## Purpose

Answers three things: (1) how to derive the hand-written addresses from the face layout, which part can be hoisted out of the loop, and into how many registers;
(2) in one round's write tail (compute values → pack → STS → fence → arrive), which orderings are **hard constraints** and which are left to ptxas;
(3) how to accept, by SASS readings, that the "write tail" really got shorter rather than only the source order changing.

## When needed (trigger)

- The face-writing role's segment between "face-release wait returns → publish arrive" is long each round, and on the timeline it is its own instruction stream (not waiting on another role),
  and the ablation removing STS + address arithmetic is above noise;
- in SASS every 16 B chunk is preceded by a string of `LOP3`/`SHF`/`IADD3` (the swizzle address is recomputed in the loop body);
- or you want to change "compute all values, then STS in a batch" into "STS each chunk as soon as it is ready", and want to know whether the source order reaches SASS.

## Mechanism

### (a) Address algebra: the UMMA swizzle XORs only the 16 B granule within a row, and depends only on the row

The swizzle of a UMMA K-major face (`.outer` from `make_smem_layout_a/b` + `.inner` on the pointer, face aligned to 1024 B) is
`bits[4, 4+B) ^= bits[7, 7+B)` on the byte address, B = 1/2/3 for SW32/SW64/SW128,
atom row width W = 16·2^B bytes (`.inner` prints as `S<B,4,3>`). Split the **unswizzled** byte offset of element (m, k) into

```
off(m, k) = 2*crd2idx(crd(m, k), outer) = bm(m) + ko(k)   # crd: see the example: coordinates split by the outer hierarchy
bm(m) = 2*crd2idx(crd(m, 0), outer)      # row part: multiple of the in-atom row pitch W + atom spacing
ko(k) = kin(k) + katom(k)                # kin < W: byte within the atom row; katom: atom spacing along K (a multiple of >= 1024)
```

Then the XOR source bits [7, 7+B) **come only from bm** (kin < W does not reach bit 7+B or above, katom does not touch [7, 7+B)), and the target bits
[4, 4+B) are always 0 in bm, so

```
swz(bm + ko) = (face + bm) ^ x(m) ^ kin + katom,   x(m) = ((bm >> 7) & (2^B - 1)) << 4
```

For a role where each thread has a fixed row and writes the same set of k chunks every round (single buffer = face base unchanged): **all addresses are per-thread constants**.
The cheapest hoist builds registers per "16 B granule within the row", not one per chunk:

- `xb = (face + bm) ^ x` (computed once outside the loop);
- there are only W/16 distinct `kin` within a row (SW64: 4, SW128: 8), one register each, `R_kin = xb ^ kin`;
- chunk q is written to `[R_{kin_q} + katom_q]` — `katom_q` is a trace-time constant, written directly as the PTX immediate offset.

Address ALU in the loop body = 0; resident registers = number of distinct kin (≤ W/16), not the number of chunks.
(One register per chunk keeps as many addresses as chunks resident for the whole loop; on a register-tight role this is a spill source, see
[`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) lever (3).)

### (b) Publish: the face-writing thread's only hard constraint is "STS → fence.proxy.async → same-warp convergence → arrive"

`tcgen05.mma` reads the face through the smem descriptor, via the **async proxy**; a plain `st.shared` goes through the generic proxy. So each face-writing thread
issues `fence.proxy.async.shared::cta` after its own **last** STS and before the arrive
(`cute.arch.fence_proxy("async.shared", space="cta")`); when a warp arrives only once,
first `sync_warp()`, then `mbarrier_arrive` inside `elect_one` (barrier count = number of warps).
The MMA-issuing warp: `mbarrier_wait(full)` → `tcgen05.fence::after_thread_sync` → `cute.gemm` ×(K/16)
→ `tcgen05.commit(empty)` inside `elect_one`. Next round the face-writing role first does `mbarrier_wait(empty, parity ^ 1)`.

**Pure register work** after the fence and after the arrive (FADD for row sums, preparation for the next round) is not subject to these constraints:
it does not touch smem, and is legal before or after the arrive — the order belongs to ptxas, see (c).

### (c) "STS each chunk when ready" and "move row sums after the arrive": source order vs SASS order

When `st.shared` is written as inline asm with `has_side_effects=True` + `~{memory}`, LLVM preserves the relative order among STS and
between STS and fence / arrive; **MUFU / F2FP / FADD are not side-effecting instructions**, and ptxas interleaves them freely by dependency and
latency. Source written as chunked streaming is not necessarily streaming in SASS; source written as a batch is not necessarily batched in SASS. Acceptance can only read SASS
(see "Acceptance"); readings are in the table below.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100
from cutlass.cute.nvgpu import tcgen05
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

BF16, F32, I32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32

def _asm(ret, args, text, cons, se, loc, ip):
    return llvm.inline_asm(ret, args, text, cons, has_side_effects=se, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def pack(lo, hi, *, loc=None, ip=None):          # lo -> low half-word (cvt's second source goes to the low half)
    return I32(_asm(T.i32(), [F32(hi).ir_value(loc=loc, ip=ip), F32(lo).ir_value(loc=loc, ip=ip)],
                    "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", False, loc, ip))

_sts = {}
def sts4(imm):                                     # st.shared.v4.b32 [addr + imm], imm is a trace-time int
    if imm not in _sts:
        where = "[$0]" if imm == 0 else "[$0+%d]" % imm
        @dsl_user_op
        def f(addr, a, b, c, d, *, loc=None, ip=None):
            _asm(None, [I32(v).ir_value(loc=loc, ip=ip) for v in (addr, a, b, c, d)],
                 "st.shared.v4.b32 %s, {$1, $2, $3, $4};" % where, "r,r,r,r,r,~{memory}", True, loc, ip)
        _sts[imm] = f
    return _sts[imm]

def crd(outer, m, k):                              # outer = ((MMA_M, MMA_K), rest_M, rest_K, stage)
    shp = outer.shape
    ma, ka = int(shp[0][0]), int(shp[0][1])        # split coordinates by hierarchy; a flat (m, k) has a different rank and crd2idx errors
    return ((m % ma, k % ka), m // ma, k // ka) + (0,) * (len(shp) - 3)

def addr_plan(outer, K, B):                        # undecorated helper: trace-time Python arithmetic only
    W = 16 << B
    plan = []
    for q in range(K // 8):                        # chunk q = k 8q..8q+7 = 16 B
        ko = 2 * int(cute.crd2idx(crd(outer, 0, 8 * q), outer))
        kin, katom = ko % W, ko - ko % W
        assert (katom >> 7) & ((1 << B) - 1) == 0  # atom spacing must not touch the swizzle source bits
        plan.append((kin, katom))
    return plan

# ---- host: the face is built by the helper; the swizzle bit count is read from .inner
#   mma   = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(BF16, F32, (128, N, 16), tcgen05.CtaGroup.ONE,
#               tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
#   a_lay = sm100.make_smem_layout_a(mma, (128, N, K), BF16, 1)      # .inner prints S<B,4,3>
#   B     = int(re.search(r"S<(\d+),", str(a_lay.inner)).group(1))

# ---- kernel, face-writing role (thread m holds the K columns of row m; single-buffered face)
sA = smem.allocate_tensor(element_type=BF16, layout=a_lay.outer, byte_alignment=1024,
                          swizzle=a_lay.inner)
face = cute.recast_ptr(sA.iterator, None, dtype=BF16).toint()   # strip the pointer's swizzle to get the raw address
plan = addr_plan(a_lay.outer, K, B)
bm = 2 * cute.crd2idx(crd(a_lay.outer, m, 0), a_lay.outer)
xb = (face + bm) ^ (((bm >> 7) & ((1 << B) - 1)) << 4)          # once, outside the loop
regs = {}
for q in cutlass.range_constexpr(K // 8):                         # one register per distinct kin
    kin = plan[q][0]
    if cutlass.const_expr(kin not in regs):
        regs[kin] = xb ^ kin if kin else xb

for r in cutlass.range(n_rounds, unroll=1):
    # ...this round's f32 values v[0..K) are already in registers...
    cute.arch.mbarrier_wait(empty, (r & 1) ^ 1)                   # the face has been fully read by the previous round's MMA
    for q in cutlass.range_constexpr(K // 8):
        w = []
        for e in cutlass.range_constexpr(4):
            w.append(pack(v[8 * q + 2 * e], v[8 * q + 2 * e + 1]))
        kin, katom = plan[q]
        sts4(katom)(regs[kin], *w)                                # STS.128 [R_kin + katom]
    cute.arch.fence_proxy("async.shared", space="cta")            # generic-proxy writes → visible to the async proxy
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(full)                           # full's count = number of face-writing warps
    # wrap-up that does not touch smem (row sums etc.) is legal here or above

# ---- kernel, the MMA-issuing warp
frA, frB = mma.make_fragment_A(sA), mma.make_fragment_B(sB)
for r in cutlass.range(n_rounds, unroll=1):
    cute.arch.mbarrier_wait(full, r & 1)
    tc_after()                                                    # tcgen05.fence::after_thread_sync
    for kb in cutlass.range_constexpr(K // 16):
        if cutlass.const_expr(kb == 0):
            mma.set(tcgen05.Field.ACCUMULATE, r != 0)
        else:
            mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.gemm(mma, acc, frA[(None, None, kb, 0)], frB[(None, None, kb, 0)], acc)
    with cute.arch.elect_one():
        tcgen05.commit(empty)                                     # release the face back to the face-writing role
```

When each thread holds only half a row (two threads split the K of one row), only `plan` changes: it is computed from that thread's start column `c0 + 8q`, and the row index becomes
that thread's row — if `c0` is a runtime quantity, `kin` / `katom` are no longer trace-time constants; then cut one `plan` for each of the two values of `c0`
and select with a warp-uniform branch (or fold `ko(c0)` into `bm` and hoist it out of the loop together).

## Acceptance (SASS, not the source)

Read over the write-tail segment (this round's first `MUFU` / value computation → the publishing `SYNCS.ARRIVE`):

| Reading | Meaning |
|---|---|
| `STS.128` count == K/8 (per round body) | one per 16 B, not split into narrow STS |
| integer ALU count between the first and last `STS.128` in the segment | whether addresses were hoisted out of the loop (the granule hoist should give 0, or only address-unrelated ones) |
| how many `MUFU` remain after the first `STS.128` | whether SASS really "stores while computing" |
| `FADD` count from `SYNCS.ARRIVE` to the back edge | whether the row sums really landed after the arrive |
| `LDL` / `STL` | whether hoisting addresses / extending f32 live ranges caused spills |

Measured readings (certification probe, M = 128 rows, one full row per thread, 8 rounds single-buffered; cutlass-dsl 4.6.2, sm_100a):

| Variant | Face | Address | Write-tail order | Integer ALU in segment | MUFU after first STS | FADD after arrive | Median cycles (face release returns → arrive) |
|---|---|---|---|---|---|---|---|
| batch_inline | SW64, K=96 | inline formula per chunk | batched | 0 | 86 | 6 | 8108 |
| stream_granule | SW64, K=96 | granule register + immediate | streaming, row sums after arrive | 10 | 47 | 98 | 6888 |
| stream_granule | SW128, K=64 | granule register + immediate | streaming, row sums after arrive | 14 | 54 | 66 | 5158 |

Reading this table (the two rows with the same job and the same face geometry SW64 K=96 are the comparison):
- Cycles (thread 0, face-release wait returns → arrive, median of 8 rounds): streaming 6888 vs batched 8108, about 15% shorter; both bit-exact numerically, both without LDL/STL, streaming STS128 = 12 = K/8.
- **Source order is not SASS order**: in the batched variant's SASS 86 MUFU still follow the first STS (96 in the whole segment), i.e. ptxas had already interleaved most of the batched spelling by itself; integer ALU in the segment is actually 0 (the single-buffered face address does not depend on the round, and ptxas hoisted the inline formula out of the loop entirely). So the gain of reordering source into "STS each chunk when ready" cannot be estimated from the source; read SASS and timestamps by the columns of this table.
- The column that really separates the two variants is "FADD after arrive": 6 → 98 (66 for SW128) — moving register work the MMA does not depend on, such as row sums, to after the arrive is what shortens the write tail visibly in SASS; the streaming variant's `int_between_first_last_sts` = 0 (SW64) / 4 (SW128), so address arithmetic is not between the STS.

## Gotchas

1. **The swizzle is byte-level, at 16 B granularity**: `bits[4,4+B) ^= bits[7,7+B)`, independent of element type; applying it to element offsets misplaces data without any error.
   Read B from `str(layout.inner)`, do not pick it by hand from K — the helper picks SW32/64/128 by the byte width of K (K=96 bf16 → SW64, K=64 → SW128).
2. **The face base must be 1024 B aligned** (`allocate_tensor(..., byte_alignment=1024)`), otherwise the low bits of `(face + bm)` are no longer 0 and the XOR decomposition breaks.
3. **`recast_ptr(it, None, dtype=...)` strips the pointer's swizzle before `.toint()`**: doing raw address arithmetic on a swizzled iterator applies the swizzle twice.
4. **`katom` saves registers only as an immediate**: write it as a PTX immediate `[$0+%d]`; writing `addr + katom` and leaving it to the compiler usually also folds into the addressing mode, but SASS is the authority.
5. **The fence belongs to each face-writing thread**: `fence.proxy.async` only covers the thread's own earlier generic-proxy writes; when a warp arrives only once, `fence` → `sync_warp` → `elect_one` arrive, and missing any of the three breaks it: no fence = the MMA reads stale data (not reproducible every time), no `sync_warp` = released before the other lanes' STS have completed.
6. **The face-release wait on a single-buffered face is a precondition for writing**: the first STS of round r must come after `mbarrier_wait(empty, (r&1)^1)`; moving an STS before this wait = overwriting the face the MMA is reading. Streaming reorders can only happen between "wait returns → arrive".
7. **Moving pure register work such as row sums to after the arrive extends the live ranges of the f32 values** (they must live past all STS); when registers are tight, look at `LDL/STL`; and ptxas may already schedule it after the arrive — read the SASS before touching the source.
8. **Clearing `ACCUMULATE` in the first round** uses the runtime bool `r != 0` on the `set` at kb == 0, with `set` and `gemm` in the same region ([`tcgen05_umma_issue_cta_pair`](tcgen05_umma_issue_cta_pair.md) gotcha 4).
9. Static loops in the kernel body always use `cutlass.range_constexpr`, and branches on Python values use `cutlass.const_expr` ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)); a helper like `addr_plan` that only does trace-time arithmetic is undecorated and uses plain `range`.
10. **The `.outer` of `make_smem_layout_a/b` has the MMA partition shape `((MMA_M, MMA_K), rest_M, rest_K, stage)` (rank 4, e.g. `((128,16),1,4,1):((64,1),0,16,0)`), not `(M, K, stage)`**: passing a flat `(m, k)`/`(m, k, 0)` to `crd2idx` or as a tensor index fails at trace time with `unable to compute crd2idx`. Split the coordinate hierarchically into `((m % MMA_M, k % MMA_K), m // MMA_M, k // MMA_K, 0)`, where MMA_K is the K of one UMMA (16 for bf16), not the atom row width.

## Sources

- PTX ISA: `st.shared` / `fence.proxy.async` / `mbarrier.arrive` / `tcgen05.mma`, `tcgen05.commit`, `tcgen05.fence`; the swizzle modes of the shared-memory matrix descriptor (SWIZZLE_32B/64B/128B) — https://docs.nvidia.com/cuda/parallel-thread-execution/
- Public infra API: `cutlass.utils.blackwell_helpers.make_smem_layout_a/b`, `cute.crd2idx`, `cute.recast_ptr`, `cute.arch.fence_proxy`, `cute.arch.mbarrier_wait/arrive`, `tcgen05.commit`, `tcgen05.MmaF16BF16Op`, `TiledMma.make_fragment_A/B`
- Certification probe: `PROBE PASS max_diff=0.0`, round 2
