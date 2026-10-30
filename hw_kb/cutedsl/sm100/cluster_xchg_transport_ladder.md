# cluster_xchg_transport_ladder (all-to-all exchange of one f32 per CTA within a cluster: the SASS footprints of several DSMEM transport primitives)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**, cluster launch; readings come from sm100 SASS (`nvdisasm -c`).
>
> **Status: ✅ Verified (numerics + SASS footprint)** --
> certified by GPU probes (the probes are not included in this release): the transport probe `PROBE PASS max_diff=5.961e-08` (round 1: the plain and the clocked versions of the four transports give
> cluster sums bit-identical to numpy, `rsqrt.approx` error ≤ 2e-6);
> the flag-word probe `PROBE PASS max_diff=5.961e-08` (round 2; round 1 was NO VERDICT because the counterexample reusing the epoch crashed the process, see gotcha 8);
> the point "the `BAR.SYNC` after the cluster wait is skipped by an unconditional `BRA`" additionally rests on the cluster-tail probe `PROBE PASS max_diff=0.0` (round 2).
> **This entry is not a recommendation of "which transport to choose"** and gives no speed ranking among the transports; in a whole kernel only one related knob was measured, with no gain (see "When needed").

## Purpose

When a cluster of CL CTAs (CL ≤ 8) must exchange "one f32 per CTA" all-to-all into "every thread has the cluster sum", there are
several ways to go between CTAs. This entry records only **what each PTX spelling lands as in SASS** (so that when reading SASS you can recognise every instruction of the exchange segment) and the correctness constraints of each.

| Spelling | "Peer has started" guarantee | How the data travels | Completion signal | Before exit |
|---|---|---|---|---|
| push_mbar | `barrier.cluster.arrive.relaxed` at kernel start, `barrier.cluster.wait` before sending | CL threads each issue 1 `st.async … mbarrier::complete_tx::bytes` to the peer's slot | local mbarrier (`expect_tx` = the bytes this CTA will receive) | nothing |
| push_bar | same as above | CL threads each issue 1 plain `st.shared::cluster.f32` | a second `barrier.cluster.arrive.release` + `wait.acquire` | nothing |
| pull_all | merged with the completion signal into one barrier | one thread writes only the slot in **its own** smem; after the barrier each thread issues CL `ld.shared::cluster.f32` | that one `arrive.release` + `wait.acquire` | `arrive` after reading, `wait` before exit (the peer's smem must live until I have finished reading) |
| pull_lane | same as pull_all | 1 `ld.shared::cluster` per thread (groups of 8 lanes, each reading one peer), merged with 3 levels of `SHFL.BFLY` | same as pull_all | same as pull_all |
| flag64 | same as push_mbar | CL threads each issue 1 8 B aligned `st.relaxed.cluster.shared::cluster.b64 {value bits, epoch}` | no mbarrier: the receiver polls its own CL 8 B slots in one asm block and leaves the loop only when all high words equal this launch's epoch | nothing (holds only when the epoch is unique, see gotcha 8) |

## SASS footprint (read instruction by instruction from the SASS output of the certification jobs)

| PTX | SASS |
|---|---|
| `barrier.cluster.arrive.relaxed.aligned` | a single `UCGABAR_ARV` |
| `barrier.cluster.arrive.release.aligned` (the default arrive is release) | `MEMBAR.ALL.GPU` → `ERRBAR` → `CGAERRBAR` → `UCGABAR_ARV` |
| `barrier.cluster.wait.aligned` (acquire by default) / `wait.acquire.aligned` | `UCGABAR_WAIT` → `CCTL.IVALL` → `BRA` → `BAR.SYNC.DEFER_BLOCKING 0x0`; that `BAR.SYNC` is skipped by the preceding unconditional `BRA` and is not on the execution path of a cluster launch; do not chase it |
| `mapa.shared::cluster` + `st.async.shared::cluster.mbarrier::complete_tx::bytes` | `PRMT …, 0x654, …` assembles two remote addresses → `STAS [R.64]` |
| `st.shared::cluster.f32` | `S2R SR_SWINHI` assembles a generic address → generic `ST.E` |
| `ld.shared::cluster.f32` | `S2R SR_SWINHI` assembles a generic address → generic `LD.E` (not `LDS`; latency is that of a generic memory access) |
| `st.relaxed.cluster.shared::cluster.b64` | generic `ST.E.64.STRONG.GPU` |
| `ld.relaxed.cluster.shared::cta.v2.b64` (local polling read) | two adjacent `v2.b64` can be merged into one `LDS.128`; an unpaired read is `LDS.64` |
| `mbarrier.try_wait.parity` retry loop | `SYNCS.PHASECHK.TRANS64.TRYWAIT` + `@!P0 BRA` (layout see [`mbarrier_try_wait_fastpath_layout`](mbarrier_try_wait_fastpath_layout.md)) |

## When needed (trigger)

- You are reading the SASS / stall ledger of a cluster exchange segment: `MEMBAR.ALL.GPU` before `UCGABAR_ARV`, generic `LD.E` / `ST.E` with addresses assembled from `SR_SWINHI`
  in the exchange segment, a seemingly redundant `BAR.SYNC` after `UCGABAR_WAIT` -- use the table above to recognise which PTX each comes from.
- The exchange segment ("the CTA's value is ready → the sum is in registers") has a large share and you want to switch the transport primitive: this entry gives only footprints and constraints; speed must be measured by a same-job A/B on your own kernel.
- **Whole-kernel readings (no gain; no gain is claimed)**:
  - one whole-kernel lever pricing: switching the default (release) cluster arrive in the prologue to `.relaxed`, add-one −1.35%
    (negative = slower), removed from the full set −0.63%, noise band of that pricing 7.61%, judged null;
  - the other transports of this table (push_bar / pull / flag64) have no whole-kernel readings.
- **Anti-trigger**: when the exchange segment does not show up in the whole kernel (a long kernel, the exchange only a tiny stretch), the difference from switching transport primitives cannot come out;
  under the probe's launch-loop timing the differences among the five spellings all fall within the noise of launch overhead.

## Minimal runnable example

The PTX spelling of each method (`llvm.inline_asm`, form (2) of [`ptx_escape`](ptx_escape.md)). For setting up push_mbar (init / fence /
`expect_tx` / early arrive) see [`cp_async_bulk_s2s_cluster_exchange`](cp_async_bulk_s2s_cluster_exchange.md).

```python
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
ATT = llvm.AsmDialect.AD_ATT

def _void(asm, args, cons, loc, ip):
    llvm.inline_asm(None, args, asm, cons, has_side_effects=True, is_align_stack=False,
                    asm_dialect=ATT, loc=loc, ip=ip)

@dsl_user_op
def cluster_arrive_release(*, loc=None, ip=None):    # SASS: MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; UCGABAR_ARV
    _void("barrier.cluster.arrive.release.aligned;", [], "~{memory}", loc, ip)

@dsl_user_op
def cluster_wait_acquire(*, loc=None, ip=None):      # SASS: UCGABAR_WAIT; CCTL.IVALL
    _void("barrier.cluster.wait.acquire.aligned;", [], "~{memory}", loc, ip)

@dsl_user_op
def mapa(addr, peer, *, loc=None, ip=None) -> Int32:  # local smem offset -> the peer's shared::cluster address (pure arithmetic)
    return Int32(llvm.inline_asm(T.i32(), [Int32(addr).ir_value(loc=loc, ip=ip), Int32(peer).ir_value(loc=loc, ip=ip)],
        "mapa.shared::cluster.u32 $0, $1, $2;", "=r,r,r",
        has_side_effects=False, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))

@dsl_user_op
def ld_cluster_f32(saddr, *, loc=None, ip=None) -> Float32:   # SASS: generic LD.E; ~{memory} keeps it from being hoisted above the barrier
    return Float32(llvm.inline_asm(Float32.mlir_type, [Int32(saddr).ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.f32 $0, [$1];", "=f,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip))

@dsl_user_op
def st_cluster_f32(saddr, v, *, loc=None, ip=None):          # SASS: generic ST.E; updates no mbarrier
    _void("st.shared::cluster.f32 [$0], $1;",
          [Int32(saddr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)], "r,f,~{memory}", loc, ip)

@dsl_user_op
def st_flagword(dp, val, epoch, *, loc=None, ip=None):       # SASS: ST.E.64.STRONG.GPU; value and flag in the same 8 B word
    _void("{\n\t.reg .b32 vb;\n\t.reg .b64 d;\n\tmov.b32 vb, $1;\n\tmov.b64 d, {vb, $2};\n\t"
          "st.relaxed.cluster.shared::cluster.b64 [$0], d;\n\t}",
          [Int32(dp).ir_value(loc=loc, ip=ip), Float32(val).ir_value(loc=loc, ip=ip),
           Int32(epoch).ir_value(loc=loc, ip=ip)], "r,f,r,~{memory}", loc, ip)

# flag64 receive: the whole polling loop is written in one asm block (labels inside { }), the result is a struct of CL f32:
#   FW_POLL:
#     ld.relaxed.cluster.shared::cta.v2.b64 {a0,a1}, [base];       # slots 0,1 (two adjacent reads are merged into LDS.128)
#     ...                                                         # the other slots likewise read in pairs
#     mov.b64 {l_i, h_i}, a_i;  x = OR_i (h_i ^ epoch);  it += 1
#     @(x != 0 && it < CAP) bra FW_POLL;                           # with an iteration cap (gotcha 9)
#     out_i = l_i
# epoch is an Int32 parameter of the kernel; the host increments it by 1 on every launch (gotcha 8).
```

Order of the pull spelling: compute all `mapa` first (pure arithmetic, before any barrier) → write your own slot → `cluster_arrive_release()` +
`cluster_wait_acquire()` (one barrier: the peer has started + the peer's slot is written) → issue the CL `ld_cluster_f32` together → fixed-tree sum →
`cluster_arrive_release()` (my remote reads are done) → …use the sum… → `barrier.cluster.wait` before exit.

## Gotchas

1. **Remote smem is accessible only when the peer has started and has not yet exited.** The push spellings rely on the early-arrive cluster barrier at the start to guarantee
   "has started", and must `barrier.cluster.wait` before sending; in the pull spellings the data barrier carries this guarantee too, but **there must be one more before exit**
   (`arrive` after reading, `wait` before exit), or the peer may exit while I am reading its smem.
2. **The pull data barrier must use `arrive.release` + `wait.acquire`** (which is the semantics when no qualifier is written); after switching to `arrive.relaxed`,
   the `st.shared` with which the peer wrote its own slot is not guaranteed to be visible to me.
3. **A plain `st.shared::cluster` updates no mbarrier**; its visibility relies only on the release/acquire of the cluster barrier that follows,
   and you cannot wait for it with an mbarrier.
4. **`ld.shared::cluster` needs `~{memory}` and `has_side_effects=True`**, or the compiler may hoist the read above the barrier.
   `mapa` is pure arithmetic and can (and should) be placed before all barriers.
5. **Empty lanes in pull_lane** (the extra lanes of a group that have no peer): let them read a legal peer and then zero the value with `selp` (branch-free);
   do not wrap the remote read in an `if` -- that generates a branch region.
6. **A `.release` cluster arrive is not cheap**: its SASS is `MEMBAR.ALL.GPU` + `ERRBAR` + `CGAERRBAR` + `UCGABAR_ARV`.
   Use `.relaxed` (no membar) for an early arrive that only "guarantees the peer has started"; use release only for the one that must publish plain writes.
   Note the whole-kernel reading under "When needed" above: switching one release arrive to relaxed measured no gain in the whole kernel.
7. **All CTAs in the cluster use the same addition tree / the same set of shuffles**; only then are the sums of all CTAs bit-identical.
8. **flag64: smem is not zeroed between launches, and the epoch must differ on every launch; reusing an epoch crashes the whole context.**
   With two consecutive launches of the same kernel, the CTAs very likely land on the same smem, and the `{value, epoch}` left by the previous launch is still there. Measured on the GPU:
   two consecutive launches with the same epoch give `unspecified launch failure` on trial 0; inside the same process the whole process died before printing a verdict,
   so the probe runs this counterexample in a subprocess. A possible explanation: the receiver is satisfied early by the stale word, finishes and exits, and the peer's remote store of this launch
   hits the smem of a CTA that has already exited (not separately verified). Increment the epoch with a host counter on every launch; CUDA graph replay freezes
   launch parameters, so this spelling cannot be used directly there. Garbage left by another kernel may also happen to equal the epoch (32 bits; the probability is tiny but not 0).
9. **For flag64 the value and the flag must be in the same aligned 64-bit scalar access**: an aligned 64-bit scalar access is single-copy atomic;
   splitting it into two 32-bit stores gives no ordering guarantee. When the receiver reads with `.v2.b64`, **each b64 element** is atomic but nothing is guaranteed between the two elements,
   so each slot checks its own high word. The polling read must be `ld.relaxed.cluster` (or `ld.volatile`) and the whole loop must be in one asm block,
   or the compiler may hoist the read out of the loop; the loop has an iteration cap, so a wrongly written transport makes the probe FAIL instead of hanging until the job times out.
10. **The `%clock` end reading must be predicated on the result** (e.g. `setp.nan.f32 p, x, x; @!p mov.u32 t, %clock;`), or ptxas may schedule it
   before the reduction.

## Sources

- PTX ISA: `barrier.cluster.arrive{.release|.relaxed}.aligned`, `barrier.cluster.wait{.acquire}.aligned`,
  `mapa.shared::cluster`, `ld.shared::cluster`, `st.shared::cluster`, `st.async`, `st.relaxed.cluster` / `ld.relaxed.cluster`,
  the single-copy atomicity of aligned scalar accesses and the per-element treatment of vector accesses in the memory consistency model, `mbarrier.try_wait.parity`,
  `mbarrier.arrive.expect_tx`, `shfl.sync.bfly`, `selp`, `rsqrt.approx.ftz.f32`, `%clock`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide, Distributed Shared Memory (before accessing remote smem, make sure the peer has started; before exiting, make sure remote accesses have completed):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Binary Utilities (`nvdisasm`, SASS `UCGABAR_ARV/WAIT`, `MEMBAR`, `CCTL`, `STAS`, `LD/ST`):
  https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL API symbols: `cute.arch.cluster_arrive_relaxed`, `cute.arch.cluster_arrive`, `cute.arch.cluster_wait`,
  `cute.arch.mbarrier_try_wait`, `cute.arch.mbarrier_init`, `cute.arch.mbarrier_init_fence`,
  `cute.arch.mbarrier_arrive_and_expect_tx`, `cute.arch.shuffle_sync_bfly`, `cute.arch.block_idx_in_cluster`,
  `cutlass.utils.SmemAllocator.allocate_tensor`, `llvm.inline_asm` (`dsl_user_op`).
- GPU certification probes: the transport probe (✅ `PROBE PASS max_diff=5.961e-08`, round 1);
  the flag-word probe (✅ `PROBE PASS max_diff=5.961e-08`, round 2);
  the cluster-tail probe (✅ `PROBE PASS max_diff=0.0`, round 2).
