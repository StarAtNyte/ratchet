# warp_specialization (producer/consumer warpgroup split + `setmaxnreg` register budget)

> One file per hardware interface. Operator-agnostic; mechanism only. This entry only covers **how the register budget
> inside a CTA is moved between warpgroups**; for the path that moves **data** between producer and consumer see
> [`wgmma_tma_mbarrier`](wgmma_tma_mbarrier.md) (TMA + mbarrier pipeline).

## Purpose

An SM90 CTA holds at most **65536 registers**. The larger the CTA, the smaller the static per-thread quota the compiler
can give: under `.reqntid N` the per-thread ceiling = `floor(65536 / N)` rounded down to a multiple of 8. A 384-thread
CTA is therefore stuck at **168** — far from enough for a consumer that has to hold several groups of WGMMA
accumulators at once, while a producer that only issues TMA cannot use 168 at all.

`setmaxnreg` redistributes this budget at run time: the producer warpgroup hands registers back (`.dec`), the consumer
warpgroups take them (`.inc`), and the total is conserved. A typical 384-thread split is
`128×24 + 256×240 = 64512 ≤ 65536` — **the `registers_per_thread` reported by the profiler is still 168**
(= 64512/384), but the consumer actually gets 240.

## When needed (trigger)

- CTA ≥ 256 threads with asymmetric warpgroup roles (one moves data, the others compute), and the consumer's register
  working set > `floor(65536 / blockDim)`.
- **Decisive symptom: `spill_ld / spill_st` (ncu `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}.sum`) non-zero**,
  while `launch__registers_per_thread` looks "normal". Halved tensor utilisation and elevated `stall_barrier` are
  downstream effects of the spill, not independent causes.
- Conversely: `registers_per_thread` **cannot** be used to judge whether the split took effect. It is a CTA average,
  and "asymmetric 128×24 + 256×240" gives **the same number** as "no split at all, 168 for everyone". The only
  machine-readable quantity that distinguishes the two is the spill count (which must be exactly 0 under a correct
  split).
- **`spill_* > 0` while `min_blocks_per_mp` is already given and dec/inc are written** — do not stop at gotcha 1, go
  on to gotcha 11 (budget move placed too early, the shared prologue falling into the reduced quota). The profiler
  symptoms of these two causes are identical; only the **statement order** in the code tells them apart.


## Minimal runnable example

All three are required: **dec/inc inside the kernel**, **`min_blocks_per_mp` on the launch side**, and **where dec/inc
are placed**. Writing only the first two still spills massively (see gotcha 11).

**Placement rule (one sentence)**: open **only one** role branch in the whole kernel; dec/inc are the first statement
of their respective branch bodies; **the shared prologue (smem addressing / barrier init / construction of
descriptors and pipeline objects / whole-CTA syncs) all goes before this branch**.

```python
WG_THREADS    = 128
NUM_THREADS   = 640          # 1 producer WG + 4 consumer WGs
PRODUCER_REGS = 32           # a multiple of 8
CONSUMER_REGS = 112          # 128*32 + 512*112 = 61440 <= 65536

@cute.kernel
def _kernel(...):
    tidx, _, _ = cute.arch.thread_idx()
    # the predicate must be warp-uniform: setmaxnreg is .sync.aligned (gotcha 3)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ---- shared prologue: compiled under the [entry quota], before dec ----
    # smem tensors / mbarrier init / TMA partition / pipeline objects are all built here.
    # put after dec = the producer has to materialise all of them with 32 registers (gotcha 11).
    storage = smem.allocate(SharedStorage)
    ...
    cute.arch.sync_threads()     # the whole-CTA sync also goes before the branch

    # ---- the one and only role branch: dec/inc are the first statement of each branch body ----
    if warp_idx < 4:
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
        ...                      # light work such as issuing TMA, right in this branch
    else:
        cute.arch.setmaxregister_increase(CONSUMER_REGS)
        # heavy register state is built [inside] the branch; do not let it straddle the dec point (gotcha 5)
        acc = cute.make_rmem_tensor(..., cutlass.Float32)
        ...

@cute.jit
def _jit(..., stream):
    _kernel(...).launch(
        grid=grid, block=[NUM_THREADS, 1, 1],
        min_blocks_per_mp=1,     # <<< without this, the two lines above might as well not exist (gotcha 1)
        stream=stream)
```

**Counter-example (gotcha 11, two `if`s splitting the budget move from the role body)** — it satisfies the letter of
gotcha 5 (dec/inc are the first statement of the branch, no barrier in between, `sync_threads` before "the branch"),
yet it puts the shared prologue into the producer's 32-register region:

```python
    if warp_idx < 4:                       # branch #1: only moves the budget
        cute.arch.setmaxregister_decrease(PRODUCER_REGS)
    else:
        cute.arch.setmaxregister_increase(CONSUMER_REGS)

    storage = smem.allocate(SharedStorage) # the shared prologue is now [after] dec
    ...                                    # <<< the producer materialises this stretch with 32 registers
    cute.arch.sync_threads()

    if warp_idx < 4:                       # branch #2: the role body
        ...
```


In DSL 4.4.2 `cute.arch.warpgroup_reg_dealloc / warpgroup_reg_alloc` still exist but are `@deprecated`, forwarding
equivalently to `setmaxregister_decrease / _increase`; both lower to `nvvm.setmaxregister`, ultimately
`setmaxnreg.{dec,inc}.sync.aligned.u32 <imm>;`. New code uses `setmaxregister_*`.

**Acceptance (not doing it is the same as not doing the change)**: dump the cubin at compile time and count the local
memory instructions in SASS; it must be 0:

```python
compiled = cute.compile(_jit, *args,
                        options=f"--dump-dir={d} --keep-cubin --keep-ptx")
# d/*.ptx   should contain .reqntid / .minnctapersm / setmaxnreg.dec / setmaxnreg.inc once each
# nvdisasm -c d/*.cubin | grep -cE '\bLDL\b|\bSTL\b'   ->  must be 0
```

A more direct reading is `cuobjdump -res-usage <dump>.cubin` — `REG` / `STACK` / `SPILL` come from the cubin built by
the DSL's own ptxas, with no version issue.

`ptxas -arch=sm_90a -O3 -v <dump>.ptx -o /dev/null` prints `Used N registers, M bytes spill stores` and the warning of
gotcha 1 straight to stderr, but **the ptxas in the image is often older than the PTX the DSL emits** (this repo hit
`Unsupported .version 8.8; current version is '8.7'`, with ptxas aborting outright). In that case rewrite the
`.version` header to the current version it reports and run again. In any case **remember to check the return code**:
when ptxas fails to compile, stderr has not a single `register/spill` line, which looks exactly like "clean, no
spill" — it is easy to read a muted instrument as a pass.

**When doing an A/B, first prove that your control really spills** (otherwise the whole experiment is empty, see
gotcha 10).

## Gotchas

1. **No `min_blocks_per_mp` → `setmaxnreg` is silently dropped (the number-one gotcha of this entry).**
   ptxas only executes `setmaxnreg` when it can **statically determine the per-thread register count at entry**;
   otherwise it drops the instruction, leaving only an info-level warning
   `(C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to determine
   register count at entry` — the DSL's JIT does not surface it to the Python level, so **the code looks right, the
   compiler compiles for a uniform 168, and the consumer spills massively**. In CuTe DSL the entry register count is
   pinned by two nvvm attributes: `nvvm.reqntid` (generated **automatically** from `launch(block=...)`) and
   `nvvm.minctasm` (generated **only when** `LaunchConfig.min_blocks_per_mp > 0`; default 0 = not generated). The
   CUDA C++ counterpart is "giving `__launch_bounds__` only one argument is not enough; give the complete
   `__launch_bounds__(maxThreads, minBlocksPerSM)`". **`min_blocks_per_mp=1` is enough**, and does not change the
   occupancy semantics.
2. **Do not "help" it with `-maxrregcount` / a lower launch bound.** That is a different route (pinning the entry
   count directly), and stacking it with `setmaxnreg` makes them fight: known phenomena include "with the bound it
   launches but hangs at the first WGMMA" and "change the threshold and it jumps between spilling and deadlock". Pick
   one: either `reqntid + minctasm + setmaxnreg`, or a global maxrregcount; do not mix.
3. **The predicate must be warpgroup-uniform.** `setmaxnreg` is `.sync.aligned`: all 128 threads of the warpgroup must
   reach the same instruction. Compare a compiler-visible warp-uniform quantity obtained with
   `cute.arch.make_warp_uniform(cute.arch.warp_idx())` (`warp_idx < 4`); **do not** use a per-thread `tidx`
   predicate, and do not put dec/inc inside a single-thread branch such as `if lane == 0` / `elect_one`.
   (The spelling and scope of single-thread election itself — the DSL has no `elect_one_sync`, `elect_one()` is a
   `with` block rather than a predicate, and its scope is the **warp** — has its own entry:
   [`elect_one`](elect_one.md).)
4. **`.inc` blocks; oversubscription is a deadlock, not an error and not a spill.** The increase must wait until the
   CTA register pool really has free registers; the pool is filled by other warpgroups of the same CTA doing `.dec`.
   The constraint is `Σ(threads per warpgroup × that warpgroup's quota) ≤ 65536`, with every quota a **multiple of 8
   within [24, 256]**. Overshoot (e.g. 3 warpgroups wanting `(80 + 224 + 224)×128 = 67584`) and all consumers wait on
   each other at `.inc` — showing up as the kernel being killed on timeout, without a single error line in the log.
5. **dec/inc go first in their respective branches, with no barrier in between.** The consumer's `.inc` must wait for
   the producer's `.dec` to happen first; if a barrier that both sides must take part in sits between the two (or the
   consumer, before `.inc`, waits on an mbarrier that only the producer will arrive at later), that is a circular
   wait. Whole-CTA `sync_threads()` (e.g. the visibility sync after mbarrier init) must go **before the branch**.
6. **Build heavy register state inside the branch.** If an accumulator / fragment is created before `if warp_idx` and
   still read after the branch, its live range straddles the producer's `.dec` point; whether ptxas can prove it dead
   is the optimiser's luck, and if it cannot, it spills in the 24-register region. Writing the consumer's heavy state
   inside the consumer branch is the safest.
7. **Getting 240 does not mean anything goes: a register array with dynamic indices is demoted wholesale to local.**
   Registers have no indirect addressing; index a fragment with a runtime `Int32` and ptxas can only put the whole
   array in local memory, and the spill count explodes regardless of the budget. Always traverse fragment elements
   with `cutlass.range_constexpr` (compile-time constant indices); hoist quantities that change across iterations out
   of the loop as scalars, rather than building "a register array indexed by iteration number".
8. **Quota rounding rule**: `floor(65536 / blockDim)` rounded down to a multiple of 8 — 384→168, 256→256 (capped at
   255/256, see the ISA), 512→128. To know "the ceiling without a split", compute this number; if the consumer's
   working set exceeds it, this entry's mechanism is required.
   **Moving** budget between roles is balanced at the same granularity: quotas are multiples of 8 per thread and
   `Σ(threads × quota)` conserves 65536, so a 256-thread role giving up 8 per thread exactly equals a 128-thread role
   gaining 16 per thread; giving a 128-thread role only "+8" would require "−4" per thread on the 256-thread side,
   which **cannot be encoded**. Measured, changing only the tiers from 104/192 to 120/184 (rest of the AST identical):
   the receiver's spill 950,272 ld / 623,104 st → 520,192 / 520,192, halved, yet the gate went 0.4873 → 0.5059 ms and
   `stall_barrier` 1.32 → 3.06 — the pair of roles that lost 8 was the pole (gotcha 12), and taking registers from the
   pole to give to whoever waits on it costs more than the receiver saves.
9. **Requires `sm_90a` (not `sm_90`)**. The CUTLASS C++ wrapper is guarded by `__CUDA_ARCH_FEAT_SM90_ALL`, and on a
   non-`a` target the whole instruction is `#if`'d out; the DSL counterpart is `CUTE_DSL_ARCH=sm_90a`.
10. **To verify this mechanism, first prove that your control really spills — building a working set with a lower
    bound on liveness is harder than it looks; this repo's probes fell into it twice.** Three things must be blocked
    at once:
    - (i) **Rematerialisation**: when every value can be traced back to a global load, ptxas would rather **re-issue
      the LDG** than write local. So the whole chain may be fed by only **one** seed value.
    - (ii) **Reordering**: a fully unrolled "stir" is a DAG with huge scheduling freedom, and ptxas reorders it into a
      topological order whose peak liveness is far below the element count. So production must be a serial chain
      (`v[i]` depends on `v[i-1]`) and consumption a **reverse-order** serial chain — the two orders are mutual
      reverses, so all N values are live exactly when `v[N-1]` is produced.
    - (iii) **Reassociation / distributivity** (where the second attempt fell): integer `*` **distributes** over `+`.
      A reverse Horner consumption `acc = acc*(v[i]+1) + (i+1)` looks serial and looks like it pins N live values, but
      it is identical to the forward streaming `acc = Σ_i (i+1)·Π_{j<i}(v[j]+1)` — needing only 2 live registers.
      That is exactly what ptxas did: all three variants reported `LDL/STL = 0`, a false negative for the control.
      **Fix: insert an XOR between the accumulator and the multiplication** — `acc = (acc ^ v[i]) * M2 + (i+1)`; XOR
      and multiplication do not distribute in either direction, so no forward streaming form exists.
    Also, **do not carry heavy values in a `cute.make_fragment` array**: an alloca that SROA refuses to promote goes
    straight to local memory, so every variant "spills" and the verdict is equally void. Hold the SSA values in a
    Python list (compile-time constant indices, fully unrolled), so local memory can only come from register-allocator
    spills. Pick N in the middle of `(168, 240]` (this repo uses 192), so the verdict is insensitive to how many
    registers the scaffolding takes.
11. **dec/inc split from the role body into two `if`s → the shared prologue falls into the reduced budget and still
    spills massively.** This is the number-one reason for "gotcha 1 done right and still spilling", with **symptoms
    identical to gotcha 1** (`spill_* > 0` with `registers_per_thread` normal), but a different cause: this time
    `setmaxnreg` **did take effect — too early**.
    `setmaxnreg` is a **run-time** instruction, and ptxas uses it to cut the kernel into several "budget regions":
    all code **after** `.dec` must fit into the reduced quota, and what does not fit spills to local memory (this is
    also the mechanism of gotcha 6). Pulling the budget move out into an early `if` amounts to declaring "from here
    on, the producer has only 32 registers" — and the shared prologue right after it (smem tensors, mbarrier init, TMA
    partition, pipeline objects) **is straight-line code, before the role branch, executed by every thread**, with
    most of its values still live for the consumer at the branch, so ptxas cannot prove them dead early. The producer
    is thus forced to materialise the whole prologue in 32 registers.
    (Measured: prologue live set 56, entry quota 96, `.dec` to 32: the fused form gives `LDL/STL == 0`, the split
    form `LDL/STL > 0`; see the certification below.)
    **Fix**: always put the shared prologue **before** dec/inc (it is then compiled under the entry quota
    `floor(65536/blockDim)`, usually enough), and write dec/inc and the role body in **the same** branch. Note this
    does not contradict gotcha 5: gotcha 5 requires "the whole-CTA `sync_threads()` before the branch", this gotcha
    requires "the shared prologue before the branch too" — **on the premise that the whole kernel has only one role
    branch**. The original wording of gotcha 5 assumed this without saying it; the two-`if` spelling can satisfy the
    letter of gotcha 5 while violating this gotcha.
12. **Moving work between roles: look at both sides' busy time, not at who waits; every beat moved into the pole role
    counts at full price in the period.**
    The period of a warp-specialized kernel is set by the **busiest** role. A role with a high wait share only shows
    that it waits on others, not that "it has spare time to do others' work": move work onto it and it gets busier
    while the period does not shrink; move work onto a role **busier than it** and the period lengthens directly by the
    amount moved. The criterion is the per-role busy period (the busy / wait breakdown of the driver timeline), not the
    wait share, and not "how many fewer HGMMAs the giving side has per iteration".
    Measured (one SS wgmma chain ×8 + the following elementwise mask + bf16 `STSM` out of the slot, moved from role A
    to role B of a three-role kernel: A loses 8 HGMMAs and two mbarrier wait points per iteration, B gains one):
    - timeline before the move: A period 5,743 cyc = busy 2,585 + wait 2,970 (looks like half its time is waiting);
      B's pair of warpgroups busy 3,036 / 3,360 — **B is the pole**, and A's wait is waiting on B.
    - the first landing declared the new accumulator + f32 staging + bf16 output face **alongside** B's existing
      fragments: spill 7,022,592 ld / 7,022,592 st, gate 0.4173 → 0.6569 ms, `tensor_pct` 12.42 → 7.94.
    - the second landing let the three temporaries borrow the registers of fragments of B already dead at that point
      (the "borrow dead fragments" gotcha of [`cfrag_fused_pack_liveness`](cfrag_fused_pack_liveness.md)):
      spill 0, but the gate was still 0.4463 (+6.9%), `tensor_pct` 11.62, `stall_barrier` 1.09 → 1.50
      — the register bill balanced, the period bill did not: B's busy time grew by the amount moved in, and A just
      waited in a different way.
    - moved back: 0.4173.
    Under the structural condition "A's busy time is shorter than B's, and A spends about half of each iteration
    waiting", this move (after removing the spill) is worth +6.9%; it could only be negative if A itself were the pole.
    Take a timeline before moving, and only move towards the role with **shorter busy time**.

## Sources

- **PTX / CUTLASS official**: `setmaxnreg.{inc,dec}.sync.aligned.u32 <imm>;`
  — cutlass 4.4.2 `include/cutlass/arch/reg_reconfig.h` L70-88
  (`warpgroup_reg_alloc<N>` / `warpgroup_reg_dealloc<N>`, `CUDA_CTA_RECONFIG_ACTIVATED`
  requires `__CUDA_ARCH_FEAT_SM90_ALL`); PTX ISA §9.7.20.5 *Miscellaneous
  Instructions: setmaxnreg*.
- **CuTe DSL 4.4.2 spelling**: `cutlass/cute/arch/nvvm_wrappers.py` L889-931
  (`setmaxregister_increase` / `setmaxregister_decrease`, and the `@deprecated`
  `warpgroup_reg_alloc` / `warpgroup_reg_dealloc`, all lowering to `nvvm.setmaxregister`);
  exports in `cutlass/cute/arch/__init__.py` L91-94.
- **The exact condition for `min_blocks_per_mp` → `nvvm.minctasm`**:
  `cutlass/cutlass_dsl/cutlass.py` `_generate_kernel_attrs` L299-328
  (`nvvm.reqntid` generated automatically from the block size; `nvvm.minctasm` only when
  `min_blocks_per_mp > 0`); `LaunchConfig` fields and defaults in
  `cutlass/base_dsl/dsl.py` L839-850; `launch(**kwargs)` constructs `LaunchConfig` directly,
  see `cutlass_dsl/cutlass.py` `KernelLauncher.launch` L1155-1160.
- **C7508 / launch-bounds conflicts (gotchas 1, 2)**: NVIDIA developer forums, answers by an NVIDIA employee
  (Robert Crovella) —
  ["'setmaxnreg' ignored; unable to determine register count at entry"](https://forums.developer.nvidia.com/t/setmaxnreg-ignored-unable-to-determine-register-count-at-entry/361145)
  ("a complete specification such as `__launch_bounds__(512,4)`") and
  ["Hopper __launch_bounds__ and setmaxnreg conflicts?"](https://forums.developer.nvidia.com/t/hopper-launch-bounds-and-setmaxnreg-conflicts/317408)
  (hang at the first wgmma with the bound added / heavy spills with it removed).
- **`.inc` blocking and oversubscription deadlock (gotcha 4)**: same forums,
  ["About setmaxnreg"](https://forums.developer.nvidia.com/t/about-setmaxnreg/356190)
  — the three-warpgroup split `(80 + 224×2)×128 = 67584 > 65536` waiting on each other; the NVIDIA answer confirms
  that while waiting it does **not** degrade into a spill.
- **GPU-certified** by a GPU probe (the probe is not included in this release)
  — 384-thread CTA (1 producer + 2 consumer WGs), each consumer thread with 192 simultaneously live int32
  (> the 168 ceiling, < the 240 budget), the working set built per gotcha 10 as "forward serial chain production +
  reverse-order XOR-multiply chain consumption" to guarantee a lower bound on liveness. Two compiled variants
  differing only in two knobs serve as each other's control:
  A (`setmaxnreg` + `min_blocks_per_mp=1`) and C (only `min_blocks_per_mp`).
  Criterion = A numerically exact (all 384 threads reach the trailing `sync_threads`, i.e. `.inc` did not deadlock)
  + `.minnctapersm` appears if and only if `min_blocks_per_mp > 0`
  + **C's `LDL/STL` > 0** (first prove the control really spills) + **A's `LDL/STL` == 0**.
  **Conclusion: `PROBE PASS max_diff=0.0` on GPU, all four contracts passed → this entry is certified.**
  Gotcha 10 was paid for by two failed versions of this probe: the first version "read a pile of values from gmem and
  permuted them in place", the second used integer Horner reverse-order consumption; all three variants reported
  `LDL/STL = 0`, the control was a false negative and the experiment empty.
  Variant B (only `setmaxnreg`, no `min_blocks_per_mp`) is still compiled in the probe and prints `LDL/STL`, but **only
  as INFO, not as a verdict** — gotcha 1 rests on NVIDIA's official answer below; this probe does not rule on it.
- **Origin of the form of the placement rule (gotcha 11)**: the rule itself: prologue first, only one role branch in
  the whole kernel, dec/inc as the first statement of the branch body; the second argument of `__launch_bounds__` on
  the launch side is exactly the `minBlocksPerSM` gotcha 1 needs; the prologue uses **lazily evaluated addressers**
  rather than materialised address arrays — addresses are recomputed on demand and take no live registers across the
  branch (same origin as gotcha 7's "do not build a register array indexed by iteration number").
- **GPU-certified (gotcha 11 / placement rule)**: **Status = ✅ certified** (on GPU
  `PROBE PASS max_diff=0.0`, all four contracts passed).
  A separate placement probe — 640-thread CTA (128 producer + 512 consumer), `PRODUCER_REGS=32` /
  `MATH_REGS=112` / `min_blocks_per_mp=1`; the budget is honoured
  (`128*32 + 512*112 = 61440 ≤ 65536`, so `.inc` does not block and the probe does not hang).
  The same computation compiled as two variants differing only in **statement order**: `fused` (prologue before dec,
  single branch) and `split` (an early `if` only moving the budget, prologue after dec, the second `if` being the role
  body). The prologue is built per gotcha 10 (single gmem seed + serial production chain + reverse-order consumption +
  an XOR between accumulator and multiplication), depth `P=56` chosen within `(32, 96)`: fits when compiled before dec,
  does not fit after dec.
  Four contracts: `fused` has `LDL/STL == 0`, `split` has `LDL/STL > 0` (first prove the control really spills), both
  variants numerically equal to an exact int32 reference per lane (a spill is a performance defect, not a numerical
  one; both sides must compute the same function, otherwise `place` would be comparing two different programs), and
  `fused`'s PTX contains `.reqntid 640` / `.minnctapersm 1` / `setmaxnreg.dec...32` / `setmaxnreg.inc...112` together.
- The facts of gotchas 1-10 come from official sources / official documentation / NVIDIA answers + this repo's GPU probes.
- **Readings of gotcha 12 and of gotcha 8's "move granularity"** (KEEP evaluations, NCU dossier census + gate window,
  same tree): the role migration was KEEP in
  an eval job (receiver spill in seven digits, gate +57%; the flip of the `rings` axis is codegen moving along, see
  gotcha 11 of [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md)); the re-landing borrowing dead fragments was KEEP
  (spill 0, still 7% slower than before the move); moving back was KEEP (restored). The per-role
  busy / wait before the move comes from sampling with the driver timeline probe (`notes/timeline_probe.py`) on the
  same tree. The tier move was KEEP (104/192 → 120/184: receiver spill halved, gate 3.8% slower).
