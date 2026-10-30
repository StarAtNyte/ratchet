# parked_warp_barrier_wait_stamp (stamp each warp's blocking named-barrier wait with `%globaltimer` to find the warps that "sit on a rendezvous for the whole launch"; the clock read after `BAR.SYNC` must be pinned to a read-back that the barrier orders, otherwise the measured wait is 0)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)** warp-specialized kernels (the mechanism is PTX `%globaltimer` / `%clock64` + `bar.sync` / `bar.arrive`, equally applicable on sm90).
>
> **Status: ✅ Verified** — certified by a GPU probe (the probe is not included in this release), `PROBE PASS max_diff=0.0` (certification round 4).
> All four contracts passed in the same round: `parked_roundtrip` / `exit_roundtrip` 0.0 (the `bar.arrive` / `bar.sync` hand-off is bit-exact, 592 CTAs × 3 launches),
> `stamp_coverage` (every warp, including wait-only and role-less ones, writes 4 monotonic stamps), `park_attribution` (in the parked form the waiter's wait ≈ the worker's busy time,
> median park ratio 0.999; in the exit form the role-less warps live 352 ns; the whole-CTA parked share is clearly higher in the parked form than in the exit form). The first three rounds failed `park_attribution` because of gotcha 3.
> **Only the stamping mechanism and the attribution readings are certified, not any whole-kernel gain** (see the end of "When needed").

## Purpose

NCU's `stall_barrier` is PC sampling of **resident warps**: as long as a warp is alive and sitting on a `bar.sync`, every sample point casts a vote for `barrier`,
whether or not the warp has any work. So warps that "sit on some rendezvous for the whole launch and only do a little at the end" (idle warps, groups that spend most of their time parked on one rendezvous)
push this reading up; and in a timeline that only stamps segments of working code, these warps have not a single segment row, so the long parked stretch is never accounted.

This entry's mechanism: **every warp (including wait-only / role-less warps) takes four stamps** — entry t0, before the blocking wait t1, after the wait t2, exit t3. From these

- per warp: `lifetime = t3 − t0`, `wait = t2 − t1`, park ratio `wait / lifetime`;
- whole CTA: parked warp-time share `Σwait / Σlifetime`, corresponding to "what fraction of resident-warp samples would land on barrier".

A warp with a park ratio ≈ 1 that is itself not on the critical path is a source of `stall_barrier`.

## When needed (trigger)

- `stall_barrier` is high, yet no corresponding segment can be found on the per-segment stamped timeline (the stall falls where a role row "has no segment at all").
- In the dossier's `nbar_widths`, some rendezvous is wider than the sum of the roles that really do work, and you want to know how long the extra warps sit there.
- You want to compare the parked share of two trees A/B in the same job **without running NCU**.

**Anti-trigger**: the waiters' park ratio is low and `stall_barrier` is still high — that is a question of where the rendezvous sits / who waits for whom, not of parked warps.

**Measured parking ≠ removing it makes things faster.** On one pair-GEMM tree, after this entry measured a group of warps with park ratio 0.97 and that group was removed
(width back to only the working roles), the same-job geometric mean removed / kept = 1.0017 (the removed side was even slightly slower, inside the noise). This entry only provides attribution.

## Minimal runnable example

Structure of the certification probe: a single CTA, no cluster; `N_WORK` working warps are busy for a while, then `bar.arrive`; one `OWNER` warp collects with `bar.sync`;
in the parked form another `N_HELP` helper warps also `bar.sync` (wait the whole time, only do a little at the end); in the exit form the helpers do not join the rendezvous and only stamp.

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import dsl_user_op

I32, I64 = cutlass.Int32, cutlass.Int64
N_WORK, N_HELP = ..., ...         # number of working warps, number of helper warps; OWNER = N_WORK, helpers right after it
N_WARPS = N_WORK + 1 + N_HELP
NB_ID = ...                       # one of 1..15 not used elsewhere
PARKED = True
W = 32 * (N_WORK + 1 + (N_HELP if PARKED else 0))   # width = number of threads that actually show up (arrive counts too)

@dsl_user_op
def gtimer(*, loc=None, ip=None):
    # pure register read of %globaltimer -- used only for the illustrative busy-spin, not for the post-wait stamp (gotcha 3)
    v = _llvm.inline_asm(_ir.IntegerType.get_signless(64), [], "mov.u64 $0, %globaltimer;", "=l",
                         has_side_effects=True, is_align_stack=False,
                         asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return I64(v)

@dsl_user_op
def stamp(gate, addr, *, loc=None, ip=None):
    # lanes with gate == 0: st.global.u64 [addr] = %globaltimer; the read and the store are in the same asm
    _llvm.inline_asm(None, [I32(gate).ir_value(loc=loc, ip=ip), I64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b64 t;\n\tsetp.eq.b32 p, $0, 0;\n"
        "\tmov.u64 t, %globaltimer;\n\t@p st.global.u64 [$1], t;\n}",
        "r,l,~{memory}", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def stamped_bar(gate, a1, a2, op, *, loc=None, ip=None):
    # stamp 1 -> bar.<op> NB_ID, W -> volatile read-back of stamp 1's slot -> clock read predicated on the read-back value -> store stamp 2; all in one asm
    # op = "sync" (blocking) | "arrive" (non-blocking). The read-back is required, see gotcha 3.
    _llvm.inline_asm(None, [I32(gate).ir_value(loc=loc, ip=ip), I64(a1).ir_value(loc=loc, ip=ip),
                            I64(a2).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p, q;\n\t.reg .b64 t, r;\n\tsetp.eq.b32 p, $0, 0;\n"
        "\tmov.u64 t, %globaltimer;\n\t@p st.global.u64 [$1], t;\n"
        "\tbar." + op + " %d, %d;\n" % (NB_ID, W) +
        "\tld.volatile.global.u64 r, [$1];\n\tsetp.eq.u64 q, r, 0xFFFFFFFFFFFFFFFF;\n"
        "\tmov.u64 t, 0;\n\t@!q mov.u64 t, %globaltimer;\n"
        "\t@p st.global.u64 [$2], t;\n}",
        "r,l,l,~{memory}", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

TBASE = gT_torch.data_ptr()    # stamp table [cta][warp][4] int64; the certification probe bakes the base address in as a constant at trace time (a new buffer needs a recompile)

@cute.kernel
def kern(gOut: cute.Tensor, gHelp: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    row = I64(TBASE) + ((bidx * N_WARPS + warp) * 32).to(I64)   # this warp's 4 int64 slots
    stamp(lane, row)                                    # stamp 0: entry -- every warp, including role-less ones
    smem = utils.SmemAllocator()
    sX = smem.allocate_tensor(I32, cute.make_layout(32 * N_WORK), byte_alignment=16)

    if warp < N_WORK:                                   # workers
        ...                                             # own work (in the probe, a gtimer busy-spin)
        sX[tidx] = ...
        stamped_bar(lane, row + 8, row + 16, "arrive")  # non-blocking, wait ≈ 0
    if warp == N_WORK:                                  # OWNER: the side that really has to wait
        stamped_bar(lane, row + 8, row + 16, "sync")    # stamp 1 / block / stamp 2
        ...                                             # read sX
    if cutlass.const_expr(PARKED):
        if warp > N_WORK:                               # parked form: wait the whole time, only do a little at the end
            stamped_bar(lane, row + 8, row + 16, "sync")
            ...
    else:
        if warp > N_WORK:                               # role-less: no wait; fill in stamp 1 / stamp 2
            stamp(lane, row + 8)
            stamp(lane, row + 16)
    stamp(lane, row + 24)                               # stamp 3: exit
```

Host side (numpy):

```python
T = gT.cpu().numpy()                                  # [cta, warp, 4] int64 ns
life = T[:, :, 3] - T[:, :, 0]
wait = T[:, :, 2] - T[:, :, 1]
park_frac = wait / np.maximum(life, 1)               # per-warp park ratio
share = wait.sum() / life.sum()                       # parked warp-time share
assert (T > 0).all() and (np.diff(T, axis=2) >= 0).all()   # every warp left 4 monotonic stamps
```

## Gotchas

1. **Stamping only the working segments = parked warps are invisible.** Stamps go at the entry and exit of **every warp**, and before and after every blocking `bar.sync` /
   `cute.arch.barrier` / `pipeline.NamedBarrier.arrive_and_wait()`; role-less warps also need entry / exit stamps.
2. **`bar.arrive` does not block; do not count it as a wait.** In the probe, the difference around the workers' arrive is just the latency of that read-back (median 928 ns in the certified form,
   32 ns in the spelling without read-back); parking only happens on the sync side.
3. **A clock read right after `BAR.SYNC` is not held back by it; the post-wait stamp must be pinned to a memory access ordered by the barrier.** In SASS, `bar.sync` is
   `BAR.SYNC.DEFER_BLOCKING`; the barrier only orders the **memory accesses** after it, and a clock read right behind it is issued early anyway. Three spellings measured on GPU that fail to capture the wait:
   (a) `t1 = gtimer(); cute.arch.barrier(...); t2 = gtimer()`, then stored by `lane == 0` — `wait` = 0 for every warp, `lifetime` normal (about 200 µs);
   (b) clock read + `st.global` + `bar.sync` + clock read + `st.global` all in one inline-PTX block, with a SASS order `GT STG BAR.SYNC.DEFER GT STG` that looks perfectly right —
   `wait` is only 32–64 ns;
   (c) same as (b) but reading `%clock64` — likewise ≈ 0 (a few dozen cycles).
   The spelling that works: after `bar.sync`, first read back stamp 1's slot with `ld.volatile.global`, then read the clock. The certified form is `stamped_bar` above (the clock read executes under a predicate computed from the read-back value),
   park ratio 0.999; the `%clock64` version likewise 0.999; a read-back without the predicate dependency also measured it in the same job (0.991), but was not certified as a contract.
   The read-back adds one gmem read-back latency to the wait segment (within about 1 µs, see gotcha 2), negligible for parks of tens of µs or more; subtract it when measuring short waits. At acceptance, first check whether `wait` is almost all 0 — if so, you hit this gotcha.
4. **Use `%globaltimer` for cross-warp / cross-CTA comparison, `%clock64` for short segments.** `%globaltimer` is in ns and has a single source across the GPU, but its resolution may be anywhere from tens of ns to
   µs depending on the implementation; enough for µs-scale parking. `%clock64` is only comparable within one SM.
5. **Stamps do not change the rendezvous itself.** The `bar.sync` / `bar.arrive` inside `stamped_bar` is the original rendezvous (same id, same width); use it to **replace** the original
   `cute.arch.barrier(...)` / `barrier_arrive(...)`, do not call it a second time; do not insert other gmem writes or syncs between stamp 1 and the block.
6. **Parked share ≠ the value of NCU `stall_barrier` itself.** The latter is a PC-sampling proportion; the share gives **direction and ownership** (who parks, for how long, how the share moves when a switch is flipped);
   for the absolute value, whole-kernel NCU is authoritative.
7. **The width must match the number of threads that actually show up.** After parked warps are removed, the rendezvous width must be reduced to match (arrive counts too), otherwise the sync side never gets enough arrivals and hangs;
   counting one warp too many (the same warp arriving and then syncing) releases early, see [`teardown_subset_rendezvous_early_exit`](teardown_subset_rendezvous_early_exit.md) gotcha 1 / gotcha 10.

## Sources

- PTX ISA: the `%globaltimer` / `%clock64` special registers https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer ;
  `bar.sync` / `bar.arrive` https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier
- Nsight Compute warp stall sampling (`smsp__pcsamp_warps_issue_stalled_barrier`): https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler
- Public infra API symbols: `cute.arch.barrier(barrier_id=, number_of_threads=)`, `cute.arch.barrier_arrive(barrier_id=, number_of_threads=)`,
  `cute.arch.make_warp_uniform`, `cute.arch.warp_idx`, `cutlass.cutlass_dsl.dsl_user_op`, `cutlass._mlir.dialects.llvm.inline_asm`,
  `cutlass.utils.SmemAllocator`, `cutlass.const_expr`.
- GPU certification probe: ✅ `PROBE PASS max_diff=0.0` (round 4). Single CTA, 592 CTAs × 3 launches,
  workers busy for 200 µs. Readings (median):
  - certified form (read-back + predicate, `%globaltimer`): parked form OWNER wait 201344 ns (park ratio 0.999), helper wait 201312 ns (0.999), helper lifetime 201536 ns;
    exit form OWNER as above, helper wait 32 ns, lifetime 352 ns; the whole-CTA parked share is clearly higher in the parked form than in the exit form;
  - same-job controls (fact rows only): no read-back, `wait` 32–64 ns (`attribution_ok=False`); read-back without predicate 0.991 (True); `%clock64` in the certified form 0.999 (True);
  - the first three rounds (rounds 1–2: spelling (a), `wait` = 0; round 3: spellings (b)(c), `wait` 32–64 ns / a few dozen cycles) all failed `park_attribution`.
