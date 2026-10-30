# cluster_barrier_tmem_alloc_sass_footprint (SASS footprint of the CTA-pair prologue / teardown: every cluster barrier brings along one unreachable widthless `BAR.SYNC 0x0`, and the TMEM allocation path contributes a few scalar `STS` / `CALL`)

> One entry per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**, kernels with cluster (2,1,1) + `cta_group::2`.
>
> **Status: ✅ Verified (certifies only the SASS footprint and correctness, not speed)** -- certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0` (round 1 of this certification, all ten contracts pass in the same round): all 6 forms compile and run, the DSMEM exchange is bit-exact,
> and the footprint table and instruction predecessors below are measured values.

## Purpose

Tools that count statically over **SASS** (stall attribution, the named-barrier census, the "scalar `STS` in a loop" census) will count, in the prologue and teardown of a CTA-pair kernel,
two kinds of instructions for which no corresponding spelling is visible in the source:

1. **The cluster barrier.** Every `barrier.cluster.arrive` / `barrier.cluster.wait` pair is, in SASS, one `UCGABAR_ARV`, one
   `UCGABAR_WAIT`, one `CCTL.IVALL`, plus one **widthless** `BAR.SYNC.DEFER_BLOCKING 0x0`. There is no `bar.sync 0;` in the PTX; this one is added by ptxas.
   **It is unreachable on the execution path**: in the probe its preceding instruction is either an unconditional `BRA` (a cluster barrier in the prologue) or `EXIT`
   (the kernel's last cluster barrier). It is the same thing as "`UCGABAR_WAIT` → `CCTL.IVALL` → `BRA` → `BAR.SYNC`, that `BAR.SYNC` is skipped" in the footprint table of
   [`cluster_xchg_transport_ladder`](cluster_xchg_transport_ladder.md).
   A census that classifies named barriers statically will record it as one "widthless symmetric rendezvous (all sides SYNC)".
2. **TMEM allocation.** The `TmemAllocator` allocate / relinquish / free path of a CTA pair adds 4 scalar `STS` in SASS (1 of them
   an `STS.U8`) and 3 `CALL`s. There is no corresponding smem write in the source.

What this entry gives: which instructions each spelling brings along (the footprint table), and how to **attribute** census counts to the source, so that these instructions are not fixed as construction defects.

## When needed (trigger)

- The PTX no longer has `bar.sync 0;` (no `cute.arch.barrier()` / `sync_threads()` in the source), yet the named-barrier census **still** reports one
  "widthless symmetric rendezvous"; or the "scalar `STS` in a loop" census reads > 0 and no corresponding smem write can be found in the source.
- SASS localisation (`nvdisasm -c -g` + `CUTE_DSL_LINEINFO=1`) shows the widthless `BAR.SYNC 0x0` right after `UCGABAR_WAIT; CCTL.IVALL`,
  or the scalar `STS` lands on `TmemAllocator`'s allocate / relinquish / free.

**Anti-trigger:** the widthless `BAR.SYNC` is **not** preceded by `BRA` / `EXIT` and sits in the tile loop (the source really has a `cute.arch.barrier()`), or the `STS`
lands on a staging write. Those two are real construction problems and not this entry's business (for the all-hands rendezvous at teardown see [`teardown_subset_rendezvous_early_exit`](teardown_subset_rendezvous_early_exit.md)).

**This is not a performance lever.** That `BAR.SYNC` is never executed; the other instructions execute once per kernel (one group per cluster barrier, one alloc / free),
not in the tile loop. Merging or deleting cluster barriers to clean up the census reading changes synchronization semantics (gotcha 2), not this `BAR.SYNC`.
In one kernel, merging the prologue cluster barriers and deleting the cluster barrier before exit along these lines gave single-evaluation readings in inconsistent directions, with no same-job A/B;
**no gain whatsoever is claimed**.

## Minimal runnable example

### Footprint table (measured, sm_100a, cluster (2,1,1), 128 threads per CTA)

Probe skeleton: every thread writes its own CTA's smem face → cluster barrier → reads the peer CTA's face with `mapa.shared::cluster` + `ld.shared::cluster`
→ writes global → one more cluster barrier before exit. The `*_tmem` forms insert a CTA-pair TMEM allocation between the two barriers and do one extra cluster
barrier to publish the base address, so there are 3 barriers in total. Counts are taken from `nvdisasm -c`, with `CUTE_DSL_KEEP_CUBIN=1` keeping the cubin.

| Form | Cluster barrier spelling | Count | Widthless `BAR.SYNC` | `BAR` with width | `UCGABAR_ARV`/`WAIT` | `CCTL` | `STS` (of which `.U8`) | `CALL` | PTX `bar.sync 0;` |
|---|---|---|---|---|---|---|---|---|---|
| dsl | `cute.arch.cluster_arrive()` + `cute.arch.cluster_wait()` | 2 | 2 | 0 | 2 / 2 | 2 | 1 (0) | 0 | 0 |
| dsl_al | `cute.arch.cluster_arrive(aligned=True)` + `cluster_wait()` | 2 | 2 | 0 | 2 / 2 | 2 | 1 (0) | 0 | 0 |
| asm_al | inline PTX `barrier.cluster.arrive.release.aligned;` + `barrier.cluster.wait.acquire.aligned;` | 2 | 2 | 0 | 2 / 2 | 2 | 1 (0) | 0 | 0 |
| asm_na | inline PTX `barrier.cluster.arrive.release;` + `barrier.cluster.wait.acquire;` (not aligned) | 2 | 2 | 0 | 2 / 2 | 2 | 1 (0) | 0 | 0 |
| dsl_tmem | dsl + the CTA-pair TMEM path | 3 | 3 | 0 | 3 / 3 | 3 | 5 (1) | 3 | 0 |
| asm_al_tmem | asm_al + the CTA-pair TMEM path | 3 | 3 | 0 | 3 / 3 | 3 | 5 (1) | 3 | 0 |

The 1 `STS` in the non-tmem forms comes from the skeleton's own smem-face write. The three instructions preceding each widthless `BAR.SYNC` (printed one by one by the probe):
in the prologue `UCGABAR_WAIT | CCTL.IVALL | BRA`; for the last cluster barrier `UCGABAR_WAIT | CCTL.IVALL | EXIT`.

**How to read it:**

1. **The number of widthless `BAR.SYNC 0x0` = the number of cluster barriers, regardless of spelling.** DSL default, DSL aligned, inline PTX aligned and inline PTX
   non-aligned are identical in every column above (the run of `MEMBAR` before a release arrive is not counted in this table, see
   [`cluster_xchg_transport_ladder`](cluster_xchg_transport_ladder.md)). The PTX always has 0 `bar.sync 0;`.
2. **It is unreachable.** Its predecessor is an unconditional `BRA` or `EXIT`. It only affects static census readings and never enters the run time.
3. **The TMEM path always adds 4 `STS` (1 of them `.U8`) and 3 `CALL`s, regardless of the cluster-barrier spelling.** These items are the same under both barrier spellings.
   In the probe the 3 plain `STS` have address `[R+0x200]` (the probe's holding word sits right after the face of 128 `Int32`, at offset 0x200), and the 1 `STS.U8`
   comes right after `UVIRTCOUNT.DEALLOC.SMPOOL`. As long as `TmemAllocator` is still used for the CTA-pair allocation / release, these instructions are there.
4. **`barrier_for_retrieve` takes a `NamedBarrier` but, without a call to `wait_for_alloc()`, generates no `BAR` with width** (the `*_tmem` forms have 0 `BAR` with width in the table),
   consistent with the readings of [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md).

### Attribution steps (census reading → source)

```bash
CUTE_DSL_LINEINFO=1 CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=$TMPDIR/d python run_kernel.py
nvdisasm -c -g $(ls -t $TMPDIR/d/**/*.cubin | head -1) > k.sass
grep -c 'barrier.cluster.wait' $TMPDIR/d/**/*.ptx           # = the number of cluster barriers you wrote
grep -cE 'BAR\.SYNC(\.DEFER_BLOCKING)? 0x0 *;' k.sass       # widthless BAR: if == the previous line, all come from cluster barriers
grep -c  'UCGABAR_WAIT' k.sass                               # should equal the previous line
grep -B3 -E 'BAR\.SYNC(\.DEFER_BLOCKING)? 0x0 *;' k.sass    # the predecessors should be UCGABAR_WAIT / CCTL.IVALL / BRA or EXIT
```

- Number of widthless `BAR.SYNC` = number of `barrier.cluster.wait` in the PTX, no `bar.sync 0;` in the PTX, predecessor is `BRA` / `EXIT`: these are all
  static instructions brought along by the cluster barrier, not construction defects, and changing the spelling does not remove them. Only the difference (more widthless `BAR` in SASS than cluster waits) is a real
  `cute.arch.barrier()`.
- Scalar `STS` exceeding the smem writes in the source by 4 (1 of them `.U8`), with `-g` line info landing on `TmemAllocator`'s allocate / relinquish / free:
  inherent to the TMEM path.

## Gotchas

1. **Do not expect a different spelling to help.** `aligned=True`, inline PTX `.release` / `.acquire`, dropping `.aligned`: the four spellings are identical in every column of the table above.
2. **Do not delete cluster barriers for the sake of the census reading.** That `BAR.SYNC` is never executed anyway; the cluster barrier itself carries synchronization: the first one in the prologue
   makes the mbarrier init and the smem face visible to the cluster; the one that publishes the TMEM base address must use the release-semantics `cluster_arrive()` (see
   [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md) gotcha 2); the one before exit guarantees that this CTA exits only after the peer has finished reading / the remote
   arrives have landed. This probe proves only the one-to-one correspondence "count ↔ footprint", not which one can be deleted. Whether the one before exit can be deleted, and what must be checked item by item,
   see [`teardown_subset_rendezvous_early_exit`](teardown_subset_rendezvous_early_exit.md) gotcha 9.
3. **The probe's loop classifier does not count the TMEM path's `STS` as in a loop.** It delimits "the [target, branch] interval of a backward `BRA` = a loop",
   and `sts_in_loop` reads 0. Another classifier that also counts the retry region of a `CALL` as a loop may count these 4 as "scalar `STS` in a loop". This entry certifies only
   "the TMEM path contributes 4 `STS` (1 of them `.U8`)", not how any particular classifier will classify them.
4. **Only a single-role 128-thread kernel was measured.** In a warp-specialized kernel, if the cluster barrier is placed after role dispatch or inside a branch, the instructions ptxas inserts
   may differ. In that case count them yourself with the attribution steps above (especially whether the predecessor is `BRA` / `EXIT`); do not reuse the numbers in the table.
5. **The TMEM base-address consistency contract here is weak.** Both CTAs read base address 0, so equality is inevitable. It only shows that the CTA-pair allocation went through
   and cannot serve as evidence that base-address publication is reliable; that evidence is in [`tmem_alloc_ptr_handoff_no_named_barrier`](tmem_alloc_ptr_handoff_no_named_barrier.md).

## Sources

- PTX ISA: `barrier.cluster.arrive{.release|.relaxed}{.aligned}` / `barrier.cluster.wait{.acquire}{.aligned}`,
  `mapa.shared::cluster`, `ld.shared::cluster`, `tcgen05.alloc` / `tcgen05.relinquish_alloc_permit` / `tcgen05.dealloc` --
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`nvdisasm -c -g`, SASS `UCGABAR_ARV` / `UCGABAR_WAIT` / `CCTL` / `BAR`):
  https://docs.nvidia.com/cuda/cuda-binary-utilities/
- Public infra API: `cute.arch.cluster_arrive(aligned=...)`, `cute.arch.cluster_wait`,
  `cutlass.utils.TmemAllocator(allocate / relinquish_alloc_permit / retrieve_ptr / free, is_two_cta=True)`,
  `cutlass._mlir.dialects.llvm.inline_asm`.
- GPU certification probe: `PROBE PASS max_diff=0.0` (round 1).
  Contracts: C0 all six forms compile and launch; C1 the DSMEM exchange of all six forms is bit-equal to numpy (3 launches each); C2 the CTA-pair base addresses of the two tmem forms are equal;
  C3 the footprint is recorded. Cluster (2,1,1), 64 clusters, 128 threads per CTA; the tmem forms allocate 256 columns.
