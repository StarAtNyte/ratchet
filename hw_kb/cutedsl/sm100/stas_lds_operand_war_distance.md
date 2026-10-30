# stas_lds_operand_war_distance (SASS facts about the source-operand registers of STAS / LDS: each `STAS` needs a complete 64-bit address pair and has no immediate offset (how many pairs are used is decided by ptxas reuse); `LDS.128` can fold `[R+imm]` into the instruction; in rolled loops the address becomes an in-place increment right next to the `STAS` — used for reading WAR inside `short_scoreboard`)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; `st.async.shared::cluster` (SASS `STAS`), `ld.shared` (SASS `LDS`).
>
> **Status: ✅ Verified (SASS facts + numerics; no gain of any rewrite is certified)** — certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (certification round 2): the bytes landed by all four spellings are bit-exact; three SASS contracts —
> `stas-no-imm` (every `STAS` of all four spellings is `[Rn.64]`, none with an immediate offset), `stas-pairs-asm` (4 `st.async` in one asm block
> use 4 different address register pairs), `lds-imm` (3 of 4 `ld.shared.v4` become `LDS.128 [R+0x..]`; the k = 0 one has no offset).
> Whole-kernel readings are at the end of "When needed": on that tree, hoisting addresses out of the loop never read as a gain.

## Purpose

`STAS` / `STS` / `LDS` are **variable-latency** instructions: only after issue does the memory unit read the source registers (address, store data),
and the hardware guards these source registers with a scoreboard slot. Any later instruction that **writes** one of these registers (even with no data relation to the access)
has to wait for that read to complete — this is a WAR (write-after-read) wait. NCU records it under the same stall reason as the RAW wait (waiting for an LDS result):
`short_scoreboard`.

This entry gives three sm100 SASS facts needed to read this kind of WAR and to estimate the register account:

- **`STAS` has no immediate-offset form.** The address operand is always a complete 64-bit address pair `[Rn.64]`; even if PTX writes `[ra+16]`,
  ptxas has to compute the complete address in registers for each `st.async` (plus the data registers, 4 per `.v4`). For 4 separately written ones the probe used 3 pairs
  (after the previous STAS issues, ptxas reuses the same pair and rewrites it in place — exactly the WAR shape); 4 written in one asm block used 4 pairs; 32 fully unrolled used 6 pairs. The mbarrier operand written separately in PTX does not take a separate operand slot in SASS
  ([inferred] it is in the high half `Rn+1` of the pair; for the address-assembly step see `PRMT …, 0x654` in [`cluster_xchg_transport_ladder`](cluster_xchg_transport_ladder.md)).
- **`LDS` has an immediate-offset form.** `ld.shared.v4 [ra+16k]` is folded into `LDS.128 R, [R0+0x10]`; a batch of loads needs only one address register.
- **In rolled loops ptxas strength-reduces the address into an in-place increment right next to the `STAS`.** Writing
  `ra = base + r*ROW_B` inside `for r in cutlass.range(n, unroll=1)` gives SASS `STAS.128 [R2.64], R8 … IADD3 R2, PT, PT, R2, <ROW_B>, RZ`: the increment lands on the same address register pair,
  only 2 instructions after that `STAS` — exactly the WAR shape.

## When needed (trigger)

- A segment's stalls are dominated by `short_scoreboard`, but RAW does not explain it: the waiting instruction **does not read** an LDS result,
  it **writes** a source register of an in-flight `STAS` / `LDS` (in PC-level sampling, "the written register = the address or data of an earlier STAS/LDS").
- This shape is visible in SASS: within a few instructions after `STAS [R2.64], R4` there is a `MOV R2, …` / `IMAD.U32 R2, …` / `IADD3 R2, …`;
  or every `STAS` is preceded by a stretch of `mapa` / `PRMT … 0x654` address recomputation.
- You need to estimate the register account of a batch of `st.async`: each needs a ready complete address pair, and "same base plus a constant" does not save the address arithmetic; the number of pairs is decided by ptxas reuse — go by the SASS.
- **Anti-trigger**: the `short_scoreboard` is mainly RAW (waiting for an LDS result to be consumed) — then what needs widening is the distance between load and use, see
  [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md), [`merge_face_lds128_read_map`](merge_face_lds128_read_map.md).
- **Whole-kernel readings (the same whole-kernel tree; no gain is claimed)**:
  - Hoisting the loop-invariant peer base, the peer mbarrier address and the peer addresses of each block out of the outer loop, computed once per thread: same-process paired timing, the three workloads
    −0.71% / +0.14% / +0.30% respectively (positive = faster), geometric mean −0.09%. Before and after, the static WAR on that segment's `STAS` source registers was 0 (window of 6 instructions after each `STAS`),
    address arithmetic 10 → 6 instructions. The following eval gate 1.332 → 1.3357 ms. **Flat**.
  - Also hoisting another group of local barrier addresses and smem slot addresses out of the outer loop (two barriers, two slots; the other two addresses recomputed at the start of each beat from the hoisted base):
    same-process pairing, "not hoisted / hoisted" − 1 for the three workloads −0.02% / −1.15% / −0.93%, geometric mean −0.70%, i.e. the hoisted variant was slower.
    In the static census, the hoisted variant also had **one extra WAR**: the 5th instruction after a `STAS` rewrote its address register.
  - The opposite direction (an earlier tree, at the edge of a `setmaxnreg` level): in two roles, deleting loop invariants such as the hoisted peer addresses and recomputing them at the point of use instead
    (peer address written as `mapa(local base, peer) + constant`), gate 1.3913 → 1.3821 ms (−0.66%, cross-job eval, spill unchanged).
  - How to read this: hoisting or not hoisting out of the loop never read as a gain on this tree; what it changes is register liveness and allocation, and it may even create WAR. If you change it, do a
    same-job A/B on your own kernel, and use the SASS reading below to check that WAR really went down.

## Minimal runnable example

```python
import cutlass
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
ATT = llvm.AsmDialect.AD_ATT

def _v(x, loc, ip):
    return Int32(x).ir_value(loc=loc, ip=ip)

@dsl_user_op
def st_async_v4(ra, a, b, c, d, rm, *, loc=None, ip=None):
    """16 B -> [ra] (shared::cluster), complete_tx recorded on [rm]. SASS: STAS.128 [Rn.64], Rd -- one complete address pair each, no immediate."""
    llvm.inline_asm(None, [_v(x, loc, ip) for x in (ra, a, b, c, d, rm)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=ATT,
        loc=loc, ip=ip)

@dsl_user_op
def lds_v4xN_imm(a, n_chunks, *, loc=None, ip=None):
    """n_chunks ld.shared.v4 from [a+16k]: one address register, offsets go into the immediate field of LDS.128 (the k = 0 one has no offset)."""
    n = 4 * n_chunks
    body = "".join("ld.shared.v4.b32 {$%d, $%d, $%d, $%d}, [$%d+%d];\n"
                   % (4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3, n, 16 * k) for k in range(n_chunks))
    res = llvm.inline_asm(llvm.StructType.get_literal([T.i32()] * n), [_v(a, loc, ip)], body,
                          ",".join(["=r"] * n) + ",r,~{memory}", has_side_effects=True,
                          is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]

# st_async_v4(ra + 16 * k, ...): even with k a compile-time constant, each one in SASS is still a complete address pair without an immediate (stas-no-imm contract).
# lds_v4xN_imm(a, n): LDS.128 [R], [R+0x10], [R+0x20] ..., one address register (lds-imm contract).
```

Acceptance (look at SASS, not source): in `cuobjdump -sass`, for each `STAS`, search forward for the first instruction that writes any of its source registers (both halves of the address pair + data),
and record the distance; if the distance is small and that writer carries `short_scoreboard` in PC sampling, it is WAR. The toy readings from the same method in the probe (INFO rows):
the minimum distances of the four spellings are 5 (each store computes its own address) / 3 (shared base, one asm per store) / 7 (shared base, 4 in one asm block) /
1 (round loop fully unrolled). Full unrolling did not widen the distance.

## Gotchas

1. **Register names in PTX do not control physical allocation.** ptxas converts to SSA first and then allocates; neither "reusing the same `.reg`" nor
   "declaring a new `.reg`" in PTX decides which physical register SASS uses. All the source can influence is: how many different values are live at once
   (LDS's immediate offset turns N addresses into 1; STAS cannot), the dependency order, and whether there is a loop-carried address — the final shape is whatever the SASS shows.
2. **(LDS) The immediate offset must be a compile-time constant.** The `%d` in `[$0+%d]` is a Python integer spliced into the PTX string at trace time;
   a run-time offset (e.g. an XOR interleave `(16*k) ^ swz`) cannot go into the immediate field, and each load still needs its own address register.
   It folds only when the run-time part can be merged into a base computed **once per batch** and the remaining offset really is a constant **addition** (`a ^ c ≠ a + c`; do not write XOR as addition).
3. **Do not expect `st.async [ra+16]` to save address arithmetic.** PTX allows it and the numerics are correct, but in SASS each one is still a complete address pair
   `[Rn.64]` and the offset does not go into the instruction; how many pairs are used is decided by ptxas reuse (probe: 3 pairs for 4 written separately, 4 pairs for 4 in one asm block (`stas-pairs-asm`), 6 pairs for 32 fully unrolled).
4. **`mapa` is pure arithmetic (`has_side_effects=False`)**; the compiler may move and merge it freely. Whether it ends up inside or outside the loop is decided by ptxas
   and the register headroom; "hoist it out of the loop" is not a reliable direction of gain (the whole-kernel readings at the end of "When needed").
5. **Keep `~{memory}` + `has_side_effects=True` on stores / loads**, otherwise the compiler may hoist a load above an mbarrier
   wait, or move an `st.async` across a barrier.
6. **Accept on the WAR share, not instruction count.** A rewrite may leave the instruction count almost unchanged; what matters is whether, in that segment's `short_scoreboard` samples,
   the share of "written register is a source of an in-flight STAS/LDS" drops, and the segment time in the same job. Fewer address recomputations are only a by-product,
   and a segment whose static WAR is 0 will not get faster from the rewrite (the first reading at the end of "When needed").
7. **The mbarrier address is also a source operand of STAS.** [inferred] It lives in the high half `Rn+1` of the address pair and is likewise read asynchronously;
   rewriting either half of that pair is a WAR.

## Sources

- PTX ISA: `st.async` (`.shared::cluster`, `.mbarrier::complete_tx::bytes`, `.v4.b32`), `mapa.shared::cluster`,
  `ld.shared.v4`, the `[reg+imm]` address-operand form (legal at the PTX level; the SASS of STAS does not keep it):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- NCU stall reason `short_scoreboard` (scoreboard dependency of MIO-class operations):
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Public infra API: `cutlass._mlir.dialects.llvm.inline_asm`, `cutlass.cutlass_dsl.dsl_user_op`,
  `cutlass.range` / `cutlass.range_constexpr`, `cute.arch.cluster_arrive_relaxed` / `cute.arch.cluster_wait`.
- Certification probe: `PROBE PASS max_diff=0.0`, round 2.
