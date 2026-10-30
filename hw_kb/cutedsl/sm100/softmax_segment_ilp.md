# softmax_segment_ilp (the row-reduction elementwise segment on a serial handshake chain: which rewrites land in SASS and on the clock, and which do not)

> **Certification source: sm90 campaign · not re-checked on sm100.**
>
> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry complements [`ew_segment_latency_schedule`](../sm90/ew_segment_latency_schedule.md). That entry concludes
> "instruction order within one basic block belongs to ptxas; reordering source gains nothing". This entry **extends that conclusion to the
> `arrive` of a named barrier** (it is equally transparent to register work — measured on GPU, the two publish positions compile to SASS that is isomorphic instruction for instruction),
> and gives the two levers that **really change instruction class / MIO count** within the segment, plus three rewrites that "look like levers" but measured
> null. All readings come from this entry's certification probe (see "Sources").
> Names of the form `sm90/xxx` are entries of the Hopper tree and **are not in the read set of an sm100 run**: the loop only hands
> the one tree `hw_kb/cutedsl/<kb_arch>/` to a run (`loop.toml [loop] kb_arch`). They are named so that you
> know they exist; every reading in them was taken on Hopper and **is not a conclusion for this architecture** — if you need it, re-earn it here.

## Purpose

Two warpgroups hand over a **row statistic** (typically the row max) one way each trip through a named barrier;
the chain is serial: A publishes → B waits, then computes → B publishes → A waits, then corrects. Each trip's cycle count is priced by
"who waits on whom at the `sync`" along this chain. This entry answers: which source rewrites within the publishing side's segment can shorten the partner's wait,
and which only change the spelling while not a single SASS instruction changes.

## When needed (trigger)

The same-position segment ledger (one collection, bucketed by SASS position) shows:

1. **The partner's blocking bucket is dominated by `barrier`**: on the side waiting at `sync`, the dominant stall of that bucket is `barrier`,
   while the MMA rendezvous bucket itself is at parity with the baseline — the wait is not for the MMA, nor for supply, but for **the partner's elementwise segment**.
2. **The publishing side's segment stalls are dominated by `short_scoreboard` / `mio_throttle`**, and the loop body has
   ≥ 16 `LDS` (byte-by-byte reads of flags / masks) — the MIO count is the lever here (§(b)).
3. **The row max uses a run-time `if v > m: m = v`**: each hop in SASS is an `FSETP` + `FSEL` pair (not a branch,
   and not `FMNMX`) — instruction class is the lever here (§(c)).

**Do not change the source for the following** (§(d), measured null): moving `arrive` earlier / later in the source;
interleaving the reductions of the two rows in the source; turning the serial max / sum chain into a tree or 4-way partial sums — **the third one is null only on the
16-deep chain on sm90 and has been overturned on sm100**, see the architecture qualification at the end of §(d).

## Mechanism

### (a) The `arrive` of a named barrier is transparent to register work

`bar.arrive` only orders **shared memory accesses** (the publishing side's `STS` must be written before it, the partner's `LDS`
must read after the `sync`). For pure register work (`FMUL` / `MUFU` / `FFMA`) it is **not** a scheduling fence:
the compiler (LLVM + ptxas) treats it as an ordinary instruction and freely reorders the ALU work around it by dependency and latency.

Measured on GPU (certification probe; the two variants differ only in whether "the 128 `FMUL` + 2 `MUFU` that scale the accumulator by alpha"
are written before or after `arrive`):

| | Loop-body count | `BAR.ARV` position | `FMUL` between first `SHFL` → `BAR.ARV` | `MUFU` before `BAR.ARV` | Time ratio | NCU `barrier` |
|---|---|---|---|---|---|---|
| scaling written before `arrive` | 724 | 264 | 9 | 12 | 1.000 | 0.82 |
| scaling written after `arrive` | 724 | 264 | 9 | 12 | 1.002 | 0.82 |

The two SASS listings are isomorphic instruction for instruction: all 128 scaling `FMUL`s sink below `BAR.ARV` (they depend on the long-latency
`MUFU.EX2`, become ready late, and the scheduler naturally puts them later); conversely, 12 `MUFU`s (2 for alpha
+ 10 for the exp2 pass) are hoisted above `BAR.ARV`. **"Publish first" is a zero lever at the source level**: the scheduler
already does it for you, and more finely than by hand.

Corollary: in the chain period `T = A + max(C, D)`, `A` (the publishing side's work before `arrive`) is decided by the scheduler,
not by the source. To shorten the partner's wait, you can only **reduce the segment's total work / MIO count / dependency hops** (§(b)(c)),
not move `arrive`.

### (b) Real lever one: read flags / masks as wide words, cut the MIO count

16 byte-wise `LDS.U8` → 4 32-bit words (the compiler synthesises 1 `LDS.128`), then bit-unpack. Measured: per loop body
`LDS` 17 → 2, kernel-wide 51 → 6; an extra 16 `LOP3` + 12 `SHF` + 18 `IADD3` of integer bit-unpacking,
the `I2F` count unchanged; **time −4.7%** (rest of the source identical). When the segment's dominant stall is `short_scoreboard` /
`mio_throttle`, the MIO count is the segment's pricing term; a few dozen more ALU instructions do not matter.

### (c) Real lever two: `fmax`, not `if v > m`

| Spelling | SASS (per hop) | Loop-body reading |
|---|---|---|
| `if v > m: m = v` (run-time `if`) | two instructions, `FSETP` + `FSEL` (**not a branch**: 0 `BRA`/`BSSY` in the loop body apart from the back edge) | `FSETP` 36 / `FSEL` 36 / `FMNMX` 0 |
| `fmax(m, v)` | one `FMNMX` | `FSETP` 0 / `FSEL` 0 / `FMNMX` 36 |

The max reduction's instruction count halves and the dependency chain loses one instruction per hop; measured time −1.2% (small, but real; the main value is
making the segment shorter, leaving room for MIO beyond §(b)).

### (d) Three rewrites measured null (do not do them)

| Rewrite | What it changes | Measured |
|---|---|---|
| move `arrive` earlier in the source (publish first) | source position only | SASS isomorphic instruction for instruction, time ratio 1.002, `barrier` share unchanged (§(a)) |
| interleave the two rows' reductions in the source (4 `SHFL` back to back, 2 `MUFU` back to back, `FMUL` alternating between rows) | source order only | instruction count unchanged; see the probe readings under "Sources" |
| serial chain → tree / 4-way partial sums (max chain 16 deep → 4 deep, sum 16 deep → 6 deep) | dependency graph, same count | interleave + tree combined: registers 227 → 245 (no spill), **time +3.8% (slower)** |

The reason is the same as in [`ew_segment_latency_schedule`](../sm90/ew_segment_latency_schedule.md): the two rows are already
two mutually independent chains, and ptxas interleaves them itself; hand-written interleaving / trees only force more partial results to be live at the same time,
registers grow and the scheduler's freedom actually shrinks. What is missing here is not in-flight width but MIO count (§(b)).

**The null of the third rewrite (chain → tree / 4-way partial sums) is qualified twice, by architecture and by shape; do not extrapolate it wholesale.** The
+3.8% in the table above was measured on sm90, 2 rows × 16 columns per thread, on a **16-deep** chain: at this depth in-flight width is not lacking,
and widening only buys more simultaneously live partial results. On sm100 both premises change — the per-thread fragment is 128 elements
(8× the 16-deep chain; the latency of a serial chain is no longer hidden by other work), and there are two instructions sm90 lacks, `add_packed_f32x2` and the 3-input
`nvvm.fmax`, so the public reduction API forks the two forms on `arch < 100`.
On sm100, read `rmem_materialize_packed_reduce` (in this tree) instead: widening itself is not free;
**the precondition is to first materialise the TensorSSA into a fresh `cute.make_rmem_tensor`**, otherwise the addends inherit the register placement already laid out by the upstream chain, ptxas has to add one `mov.b64` per pair,
and what the packed instruction saved is paid back — that entry is GPU-certified
(`PROBE PASS max_diff=4.053116e-06`). sm90 runs have neither that entry nor those instructions; the table above stands for them.

## Minimal runnable example

### (a) Handshake skeleton: before `arrive` only the `STS` needs to be guaranteed, everything else is free

```python
import cutlass
import cutlass.cute as cute

# One thread holds 2 rows x 16 columns of scores (the quad's 4 lanes share these two rows),
# and a 2 x 64 fp32 accumulator acc (scaled by alpha every trip).
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

# Publish: the partner only needs these two words; the STS must come before arrive (the only thing arrive constrains)
sStat[row0] = m0                           # the 4 lanes of a quad write the same word with the same value: branch-free
sStat[row1] = m1
cute.arch.barrier_arrive(barrier_id=BAR_A, number_of_threads=256)   # publish and move on, no wait

# Writing the scaling / exp2 before or after arrive makes no difference -- the scheduler orders by dependency, not by source
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

The partner side is symmetric: `cute.arch.barrier(barrier_id=BAR_A, number_of_threads=256)` blocks →
`LDS` the statistic → its own row reduction (the value read in is treated as one more `fmax` input, folded in before the `SHFL`) →
`STS` + `barrier_arrive(BAR_B)` → its own scaling and exp2.

> The width is an explicit parameter: a wrong `number_of_threads` raises no error and no warning, it just never releases
> (`sm90/warp_specialization`).

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
    if v > m:            # run-time if: lowered to compare + select, doubling the count
        m = v
```

### (c) Read flags in bulk: wide words + bit unpacking, not byte-wise `LDS`

```python
sF32 = cute.recast_tensor(sFlags, cutlass.Int32)          # 32-bit view of the same smem
w = [sF32[fbase // 4 + k] for k in range(4)]              # 16 bytes = 4 words (the compiler synthesises LDS.128)
fl = [((w[c // 4] >> (8 * (c % 4))) & 1).to(cutlass.Float32) for c in range(16)]
for c in range(16):
    cv = fl[c] * SCALE
    bv = (fl[c] - 1.0) * NEG_MAG                          # invalid column -> -NEG_MAG
    for r in range(2):
        s[r][c] = x[r][c] * cv + bv                       # one FFMA does mask + scale
```

`fbase` must be 16 B aligned (one lane holds 16 consecutive flags), otherwise the bit unpacking is misaligned and **no error is raised**.

> **This example only demonstrates the wide-word read (the MIO lever of §(b))**. The three lines after `.to(cutlass.Float32)` are the
> "I2F form" of the mask — one `I2F` per column (same throughput class as `MUFU.EX2`) + `FMUL` + `FFMA`, and pass 3 needs
> one more independent `FADD` before it can go into `EX2`. The hardware form of the mask itself is a predicate select (`FSEL`), with the scaling folded into the `FFMA` that feeds
> `EX2`: follow [`branch_free_select_mask`](branch_free_select_mask.md),
> using the unpacked `bit` directly as the predicate source, without converting to float.

### (d) Acceptance: SASS counts + segment ledger, not end-to-end

1. **Loop-body `LDS` count**: after switching from byte-wise to wide-word reads, ≤ 6 `LDS` per loop body (this entry's probe: 17 → 2).
2. **Instruction class of the max**: `FSETP`/`FSEL` pairs disappear, `FMNMX` appears, and the loop body has no `BRA`/`BSSY` apart from the back edge.
3. **The segment itself**: `LDL == STL == 0`; the partner's `sync` bucket `t_ns` drops **and** its `barrier` share drops.
4. **Do not use "the number of `FMUL` between `SHFL` → `BAR.ARV`" for acceptance**: it is decided by the scheduler and has nothing to do with where
   `arrive` sits in the source (table in §(a)).

## Gotchas

- **Treating an earlier `arrive` as a lever**. Not a single SASS instruction changes, time ratio 1.002. What has to change is the segment's MIO count and
  instruction class, not the publish point. Likewise, do not infer the source spelling from "how many `FMUL` precede `BAR.ARV`" —
  that is the scheduler's output.
- **Using `if v > m: m = v` as max**. Two instructions per hop (`FSETP` + `FSEL`), neither a branch nor `FMNMX`;
  use `fmax`.
- **Hand-written two-row interleaving / tree / 4-way partial sums**. Registers 227 → 245, time +3.8%. ptxas interleaves two independent chains
  itself; changing the dependency graph can only be worth something when both register headroom and in-flight width are lacking; what is lacking here is MIO.
- **Alignment of the wide-word read**. After `recast_tensor` to 32-bit, indices are word indices; a lane's 16 flags must start at a
  16 B boundary, otherwise each lane reads its neighbour's bits — wrong values, yet it compiles.
- **`.to(Float32)` after unpacking, then multiplying**. That is the "I2F form": one `I2F` per column (same pipe as `EX2`) plus two arithmetic instructions,
  and one more `FADD` in pass 3; the mask should go through a predicate select, see
  [`branch_free_select_mask`](branch_free_select_mask.md).
- **Publishing with a leader-only run-time `if` writing smem**. That is a divergent region; the 4 lanes of a quad
  writing the same word with the same value is the legal branch-free spelling (the form used in this entry's probe).
- **Leaving an `LDS` of the statistic before `arrive` / putting the `STS` after `arrive`**. `arrive` only
  orders shared accesses: the publishing side's write must come before `arrive`, the partner's read after `sync`;
  the other way round is a race, with no error.

## Sources

- PTX ISA, "Parallel Synchronization and Communication Instructions: bar, barrier"
  (`bar{.cta}.arrive` only marks arrival, without waiting; `bar{.cta}.sync` waits until all have arrived; both guarantee ordering of preceding
  shared accesses across participating threads — the guarantee is about **memory accesses**, not the scheduling position of register operations)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar>.
- PTX ISA, the floating-point `max` instruction (`max.f32`, one instruction, branch-free)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>.
- Nsight Compute Kernel Profiling Guide, "Warp Scheduler States" (`barrier` = waiting for a CTA rendezvous;
  `short_scoreboard` = waiting for an MIO-class result; `mio_throttle` = MIO queue full; `wait` = waiting on a fixed-latency dependency)
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>.
- Public infra API symbols: `cute.arch.barrier_arrive(barrier_id=, number_of_threads=)`,
  `cute.arch.barrier(barrier_id=, number_of_threads=)` (fallback spelling `cutlass.pipeline.NamedBarrier`),
  `cute.arch.fmax`, `cute.arch.shuffle_sync_bfly`, `cute.math.exp2(..., fastmath=True)`,
  `cute.recast_tensor`, `cute.make_rmem_tensor`, `cutlass.range_constexpr`, `cutlass.const_expr`
  (if `fmax` is missing, `cutlass._mlir.dialects.llvm.inline_asm` + `cutlass.cutlass_dsl.dsl_user_op`,
  see [`ptx_escape`](ptx_escape.md)).
- GPU certification probe (not included in this release) — eight variants of the same computation, changing only five
  independent source switches (`fmax` / tree + partial sums / two-row interleave / wide-word flag read / publish first), ten contracts judged in the same round:
  **numeric** each of the eight variants reproduces the fp32 numpy reference; **no_spill** `LDL == STL == 0`;
  **arrive_transparent** for the publish-first and publish-late variants, the loop-body count, the `FMUL` between `SHFL` → `BAR.ARV`
  and the `MUFU` before `BAR.ARV` each differ by ≤ 4, and the publish-late variant has ≤ 48 such `FMUL` (the 128 scaling instructions are sunk by the scheduler
  below `arrive`); **publish_position_time** `t(publish late)/t(publish first) ∈ [0.98, 1.02]`;
  **barrier_stall_ncu** the two NCU `barrier` shares differ by ≤ 10%; **fmax_class_sass** the `if`-chain loop body has
  `FSETP ≥ 32 / FSEL ≥ 32 / FMNMX = 0`, the `fmax` loop body has `FMNMX ≥ 32 / FSETP = FSEL = 0`, and neither has
  a branch apart from the back edge; **lds_batch** the wide-word read has `LDS ≤ 6` per loop body and a total ratio ≥ 2.5;
  **flags_wide_time** `t(byte-wise)/t(wide word) ≥ 1.025`; **reshape_null_time**
  `t(serial)/t(interleave + tree) ≤ 1.01`; **segment_time** `t(baseline)/t(fmax + wide word) ≥ 1.03`.
  First-round readings (same GPU, same DSL 4.6.1): the two publish-position variants are isomorphic instruction for instruction in SASS, time ratio 1.0019,
  `barrier` 0.82 = 0.82; `LDS` 17 → 2 / 51 → 6, wide-word read −4.7%; `fmax` −1.2%; interleave + tree +3.8%,
  registers 227 → 245. Status: **✅ GPU-certified** — all ten contracts of the certification job are `PROBE PASS`, the numeric contract max_diff=5.94e-06, the other contracts each pass at the thresholds above.
