# softmax_segment_ilp (the row-reduction elementwise segment on a serial handshake chain: which rewrites reach SASS and the clock, and which do not)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry complements [`ew_segment_latency_schedule`](ew_segment_latency_schedule.md). That entry's
> conclusion is "instruction order within one basic block belongs to ptxas; reordering the source gains
> nothing". This entry **extends that conclusion to the `arrive` of a named barrier** (it is equally
> transparent to register work — measured on GPU, the two publish positions compile to instruction-for-
> instruction isomorphic SASS), and gives the two levers that **really change the instruction class / MIO
> count** inside the segment, plus three rewrites that "look like levers" but measured null. All readings
> come from this entry's certification probe (see "Sources").

## Purpose

Two warpgroups hand over a **row statistic** (typically the row max) one way through a named barrier on
every pass, and the chain is serial: A publishes → B waits and computes → B publishes → A waits and
corrects. The period of each pass is priced by "who waits on whom at the `sync`" along this chain. This
entry answers: which source rewrites inside the publishing segment shorten the peer's wait, and which only
change the spelling while the SASS stays the same instruction for instruction.

## When needed (trigger)

In the same-position segment ledger (one collection, bucketed by SASS position) you see:

1. **The peer's blocking bucket dominated by `barrier`**: on the side that waits at the `sync`, the main
   stall term of that bucket is `barrier`, while the MMA rendezvous bucket itself is on par with the
   baseline — the wait is not for the MMA nor for the supply, but for **the peer's elementwise segment**.
2. **Stall inside the publishing segment dominated by `short_scoreboard` / `mio_throttle`**, with ≥ 16
   `LDS` in the loop body (reading flags / masks byte by byte) — the MIO count is the lever here (§(b)).
3. **The row max uses a runtime `if v > m: m = v`**: each hop in SASS is an `FSETP` + `FSEL` pair (not a
   branch, and not `FMNMX`) — the instruction class is the lever here (§(c)).

**Do not change the source because of the following** (§(d), measured null): moving `arrive` earlier /
later in the source; interleaving the two rows' reductions in the source; turning the serial max / sum chain
into a tree or 4-way partial sums.

## Mechanism

### (a) The `arrive` of a named barrier is transparent to register work

`bar.arrive` only orders **shared memory accesses** (the publishing side's `STS` must be written before it,
the peer's `LDS` must read after the `sync`). For pure register work (`FMUL` / `MUFU` / `FFMA`) it is **not**
a scheduling fence: the compiler (LLVM + ptxas) treats it as an ordinary instruction and freely reorders the
ALU work around it by dependency and latency.

Measured on GPU (certification probe; the two variants differ only in whether "the 128 `FMUL` + 2 `MUFU`
that scale the accumulator by alpha" are written before or after the `arrive`):

| | Loop-body instruction count | `BAR.ARV` position | `FMUL` between the first `SHFL` → `BAR.ARV` | `MUFU` before `BAR.ARV` | Time ratio | NCU `barrier` |
|---|---|---|---|---|---|---|
| scaling written before `arrive` | 724 | 264 | 9 | 12 | 1.000 | 0.82 |
| scaling written after `arrive` | 724 | 264 | 9 | 12 | 1.002 | 0.82 |

The two SASS listings are isomorphic instruction for instruction: all 128 scaling `FMUL` are sunk below
`BAR.ARV` (they depend on the long-latency `MUFU.EX2`, become ready late, and the scheduler naturally places
them later); conversely, 12 `MUFU` (2 for alpha + 10 of the exp2 pass) are hoisted above `BAR.ARV`.
**"Publish first" is a zero lever at the source level**: the scheduler already does it for you, and more
finely than hand-written code.

Corollary: in the chain period `T = A + max(C, D)`, `A` (the work before the publishing side's `arrive`) is
decided by the scheduler, not by the source. To shorten the peer's wait you can only **reduce the segment's
total work / MIO count / dependency hops** (§(b)(c)), not move the `arrive`.

### (b) Real lever one: read flags / masks as wide words, cutting the MIO count

16 byte-wise `LDS.U8` → 4 32-bit words (the compiler fuses them into 1 `LDS.128`) and then bit extraction.
Measured: per-loop-body `LDS` 17 → 2, kernel-wide 51 → 6; the integer bit extraction adds 16 `LOP3` + 12
`SHF` + 18 `IADD3`, the `I2F` count is unchanged; **time −4.7%** (rest of the source identical). When the
segment's main stall term is `short_scoreboard` / `mio_throttle`, the MIO count is what prices the segment,
and a few dozen extra ALU instructions do not matter.

### (c) Real lever two: `fmax`, not `if v > m`

| Spelling | SASS (per hop) | Loop-body reading |
|---|---|---|
| `if v > m: m = v` (runtime `if`) | `FSETP` + `FSEL`, two instructions (**not a branch**: 0 `BRA`/`BSSY` in the loop body besides the back edge) | `FSETP` 36 / `FSEL` 36 / `FMNMX` 0 |
| `fmax(m, v)` | one `FMNMX` | `FSETP` 0 / `FSEL` 0 / `FMNMX` 36 |

The max reduction's instruction count halves and the dependency chain loses one instruction per hop;
measured time −1.2% (small, but real; the main value is making the segment shorter, making room for MIO
beyond §(b)).

### (d) Three rewrites measured null (don't)

| Rewrite | What it changes | Measured |
|---|---|---|
| move `arrive` earlier in the source (publish first) | only the source position | SASS isomorphic instruction for instruction, time ratio 1.002, `barrier` share unchanged (§(a)) |
| interleave the two rows' reductions in the source (4 `SHFL` back to back, 2 `MUFU` back to back, `FMUL` alternating rows) | only the source order | instruction count unchanged; see the probe readings in "Sources" |
| serial chain → tree / 4-way partial sums (max chain depth 16 → 4, sum depth 16 → 6) | the dependency graph, count unchanged | interleave + tree stacked: registers 227 → 245 (no spill), **time +3.8% (slower)** |

The reason is the same as in [`ew_segment_latency_schedule`](ew_segment_latency_schedule.md): the two rows
are already two mutually independent chains, and ptxas interleaves them by itself; a hand-written
interleave / tree only makes more partial results live at once, so registers grow and the scheduler's
freedom actually shrinks. In-flight width is not what is lacking here; the MIO count is (§(b)).

## Minimal runnable example

### (a) Handshake skeleton: before `arrive` only the `STS` must be guaranteed; everything else is free

```python
import cutlass
import cutlass.cute as cute

# one thread holds 2 rows x 16 columns of scores (the 4 lanes of a quad share these two rows),
# plus a 2 x 64 fp32 accumulator acc (scaled by alpha on every pass).
fmax = cute.arch.fmax                      # -> max.f32 -> FMNMX (see §(b))

m0 = s[0][0]
m1 = s[1][0]
for c in range(1, 16):                     # plain Python, unrolled at trace time: a serial chain is enough
    m0 = fmax(m0, s[0][c])
    m1 = fmax(m1, s[1][c])
m0 = fmax(m0, cute.arch.shuffle_sync_bfly(m0, 1))
m0 = fmax(m0, cute.arch.shuffle_sync_bfly(m0, 2))
m1 = fmax(m1, cute.arch.shuffle_sync_bfly(m1, 1))
m1 = fmax(m1, cute.arch.shuffle_sync_bfly(m1, 2))
m0 = fmax(m0, m_old0)                      # merge with the running value
m1 = fmax(m1, m_old1)

# publish: the peer only needs these two words; the STS must precede arrive (the only thing arrive constrains)
sStat[row0] = m0                           # the 4 lanes of a quad write the same word with the same value: branch-free
sStat[row1] = m1
cute.arch.barrier_arrive(barrier_id=BAR_A, number_of_threads=256)   # publish and move on, no wait

# whether scaling / exp2 is written before or after arrive makes no difference -- the scheduler orders by dependency, not by source
a0 = cute.math.exp2(m_old0 - m0, fastmath=True)
a1 = cute.math.exp2(m_old1 - m1, fastmath=True)
l0 = l0 * a0
l1 = l1 * a1
for co in cutlass.range_constexpr(NCOL_O):
    acc[(0, co)] = acc[(0, co)] * a0
    acc[(1, co)] = acc[(1, co)] * a1
ls0 = row_sum(s[0], m0)                    # 16 serial adds are fine (§(d))
ls1 = row_sum(s[1], m1)
```

The peer side is symmetric: block on `cute.arch.barrier(barrier_id=BAR_A, number_of_threads=256)` →
`LDS` the statistic → its own row reduction (the value read in is one more `fmax` input, folded in before
the `SHFL`) → `STS` + `barrier_arrive(BAR_B)` → its own scaling and exp2.

> The width is an explicit parameter: a wrong `number_of_threads` raises no error and no warning, it just
> never releases ([`warp_specialization`](warp_specialization.md)).

### (b) Two spellings of `fmax`

```python
# 1) built into the DSL (prefer it when available; the certification probe uses it)
fmax = cute.arch.fmax                      # -> max.f32 -> FMNMX

# 2) PTX escape hatch when it is missing (see ptx_escape)
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

@dsl_user_op
def fmax(a, b, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip),
         cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
        "max.f32 $0, $1, $2;", "=f,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))
```

The form not to write (two instructions per hop, `FSETP` + `FSEL`):

```python
for c in cutlass.range_constexpr(NCOL):
    v = s[c]
    if v > m:            # runtime if: lowered to compare + select, doubling the instruction count
        m = v
```

### (c) Batch-read flags: wide words + bit extraction instead of byte-wise `LDS`

```python
sF32 = cute.recast_tensor(sFlags, cutlass.Int32)          # 32-bit view of the same smem range
w = [sF32[fbase // 4 + k] for k in range(4)]              # 16 bytes = 4 words (the compiler fuses them into LDS.128)
fl = [((w[c // 4] >> (8 * (c % 4))) & 1).to(cutlass.Float32) for c in range(16)]
for c in range(16):
    cv = fl[c] * SCALE
    bv = (fl[c] - 1.0) * NEG_MAG                          # invalid column -> -NEG_MAG
    for r in range(2):
        s[r][c] = x[r][c] * cv + bv                       # one FFMA does mask + scale
```

`fbase` must be 16 B aligned (one lane holds 16 consecutive flags), otherwise the bit extraction is
misaligned **without any error**.

> **This example only demonstrates the wide-word read (the MIO lever of §(b))**. The three lines after
> `.to(cutlass.Float32)` are the "I2F form" of the mask — one `I2F` per column (same throughput tier as
> `MUFU.EX2`) + `FMUL` + `FFMA`, and pass 3 needs a separate `FADD` before it can enter `EX2`. The hardware
> form of the mask itself is a predicated select (`FSEL`), with the scaling folded into the `FFMA` that feeds
> `EX2`: follow [`branch_free_select_mask`](branch_free_select_mask.md) and use the extracted `bit` directly
> as the predicate source, without converting to float.

### (d) Acceptance: SASS counts + the segment ledger, not end-to-end

1. **Loop-body `LDS` count**: after going from byte-wise to wide-word reads, `LDS` per loop body ≤ 6 (this
   entry's probe: 17 → 2).
2. **Instruction class of the max**: the `FSETP`/`FSEL` pairs disappear, `FMNMX` appears, and the loop body
   has no `BRA`/`BSSY` besides the back edge.
3. **The segment itself**: `LDL == STL == 0`; the peer's `sync` bucket `t_ns` drops **and** its `barrier`
   share drops.
4. **Do not use "the number of `FMUL` between `SHFL` → `BAR.ARV`" as acceptance**: it is decided by the
   scheduler and has nothing to do with the position of `arrive` in the source (table in §(a)).

## Gotchas

- **Treating moving `arrive` earlier as a lever**. Not a single SASS instruction changes, time ratio 1.002.
  What must change is the segment's MIO count and instruction class, not the publish point. Likewise, do
  not infer the source spelling from "how many `FMUL` precede `BAR.ARV`" — that is the scheduler's output.
- **Using `if v > m: m = v` as a max**. Two instructions per hop (`FSETP` + `FSEL`), neither a branch nor
  `FMNMX`; use `fmax`.
- **Hand-written two-row interleave / tree / 4-way partial sums**. Registers 227 → 245, time +3.8%. ptxas
  interleaves two independent chains by itself; changing the dependency graph could only be worth anything
  when both register headroom and in-flight width are lacking; here what is lacking is MIO.
- **Alignment of wide-word reads**. After `recast_tensor` to 32-bit, indices are word indices; a lane's 16
  flags must start at a 16 B boundary, otherwise each lane reads its neighbour's bits — wrong numbers while
  compiling fine.
- **`.to(Float32)` after the bit extraction and then multiplying**. This is the "I2F form": one `I2F` per
  column (on the same pipe as `EX2`) plus two arithmetic instructions, and another `FADD` in pass 3; the mask
  should go through a predicated select, see [`branch_free_select_mask`](branch_free_select_mask.md).
- **Publishing by writing smem under a leader-only runtime `if`**. That is a divergent region; the 4 lanes of
  a quad writing the same word with the same value is the legal branch-free spelling (the form of this
  entry's probe).
- **Leaving an `LDS` of the statistic before the `arrive` / writing the `STS` after the `arrive`**. `arrive`
  only orders shared accesses: the publishing side's writes must precede the `arrive`, the peer's reads must
  follow the `sync`; the reverse is a race, and raises no error.

## Sources

- PTX ISA, "Parallel Synchronization and Communication Instructions: bar, barrier"
  (`bar{.cta}.arrive` only marks arrival without waiting; `bar{.cta}.sync` waits until everyone has arrived;
  both give the participating threads an ordering guarantee for the preceding shared accesses — what is
  guaranteed is **memory access**, not the scheduling position of register arithmetic)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar>.
- PTX ISA, the floating-point `max` instruction (`max.f32`, a single instruction, no branch)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>.
- Nsight Compute Kernel Profiling Guide, "Warp Scheduler States" (`barrier` = waiting for a CTA rendezvous;
  `short_scoreboard` = waiting for an MIO-class result; `mio_throttle` = MIO queue full; `wait` = waiting on a
  fixed-latency dependency)
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>.
- Public-infra API symbols: `cute.arch.barrier_arrive(barrier_id=, number_of_threads=)`,
  `cute.arch.barrier(barrier_id=, number_of_threads=)` (fallback spelling `cutlass.pipeline.NamedBarrier`),
  `cute.arch.fmax`, `cute.arch.shuffle_sync_bfly`, `cute.math.exp2(..., fastmath=True)`,
  `cute.recast_tensor`, `cute.make_rmem_tensor`, `cutlass.range_constexpr`, `cutlass.const_expr`
  (without `fmax`: `cutlass._mlir.dialects.llvm.inline_asm` + `cutlass.cutlass_dsl.dsl_user_op`,
  see [`ptx_escape`](ptx_escape.md)).
- GPU-certified probe (not included in this release) — eight variants of the same segment, changing
  only five independent source switches (`fmax` / tree + partial sums / two-row interleave / wide-word flag
  read / publish first), ten contracts judged in the same round:
  **numeric** each of the eight variants reproduces the fp32 numpy reference; **no_spill** `LDL == STL == 0`;
  **arrive_transparent** between the publish-first and publish-late variants, the loop-body instruction count,
  the `FMUL` between `SHFL` → `BAR.ARV` and the `MUFU` before `BAR.ARV` each differ by ≤ 4, and in the
  publish-late variant that `FMUL` count ≤ 48 (the 128 scaling instructions sunk below `arrive` by the
  scheduler); **publish_position_time** `t(publish late)/t(publish first) ∈ [0.98, 1.02]`;
  **barrier_stall_ncu** the NCU `barrier` shares of the two differ by ≤ 10%; **fmax_class_sass** the `if`
  chain's loop body `FSETP ≥ 32 / FSEL ≥ 32 / FMNMX = 0`, the `fmax` loop body `FMNMX ≥ 32 / FSETP = FSEL = 0`,
  both branch-free apart from the back edge; **lds_batch** wide-word read `LDS ≤ 6` per loop body and a
  total ratio ≥ 2.5; **flags_wide_time** `t(byte-wise)/t(wide-word) ≥ 1.025`; **reshape_null_time**
  `t(serial)/t(interleave + tree) ≤ 1.01`; **segment_time** `t(baseline)/t(fmax + wide-word) ≥ 1.03`.
  First-round readings (same GPU, same DSL 4.6.1): the two publish-position variants have SASS isomorphic
  instruction for instruction, time ratio 1.0019, `barrier` 0.82 = 0.82; `LDS` 17 → 2 / 51 → 6, wide-word
  read −4.7%; `fmax` −1.2%; interleave + tree +3.8%, registers 227 → 245. Status: **✅ GPU-certified** — all
  ten contracts of the certification job `PROBE PASS`, numeric contract max_diff=5.94e-06, the other contracts
  each passed at the thresholds above.
