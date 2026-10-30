# role_loop_state_spill_free (in a warp-specialized CTA, **another role's scalars** live into this role's loop: write role dispatch as a mutually exclusive chain and read the scalars in place at role entry; **do not** split run-time ring geometry into several constexpr loop bodies)

> One file per hardware interface. Operator-agnostic; mechanism only (ptxas register allocation × CuTe DSL control-flow lowering).
>
> **Status: ✅ Verified (only the compiled-artifact footprint and numerics are verified; no whole-kernel gain is certified)** — certified by a GPU probe (the probe is not included in this release):
> `PROBE PASS max_diff=0.0`, certification round 4.
> A 2 × 2 × 3 factorial (dispatch seq / elif × scalar read point prologue / role entry × geometry carried / two constexpr bodies / recomputed), each cell bit-exact under two run-time geometries:
> elif + role entry (carried or recomputed) `REG 128 / STACK 0 / LDL 0 / STL 0`; elif alone (scalars still read in the prologue) or role-entry read alone (dispatch still seq) are also 0;
> the positive control seq + prologue has lots of `LDL / STL`; two constexpr bodies on elif + role entry give `STACK 32`, `LDL 18 / STL 15`.
> **Whole kernel**: landed twice on a whole kernel with non-zero spill (once the dispatch rewrite alone, once S in a package with other changes); neither cleared the spill, and the gate was flat both times (readings at the end of "When needed").
>
> Neighbouring entries: budgeted regions, a single role branch, no rejoin after the branch, "long-lived values squeezed out by the peak segment" are in
> [`warp_specialization`](warp_specialization.md) (gotchas 6, 11, 14); how the ring itself is built (rolled main loop, stage / phase carried) is in
> [`mbarrier_ring_rolled_loop`](mbarrier_ring_rolled_loop.md); rmem arrays with run-time indices going to local memory are in
> [`rmem_array_runtime_index_local_frame`](rmem_array_runtime_index_local_frame.md); the peak of the **read-back segment** itself is in
> [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) /
> [`lowtier_role_tmem_readout_spill_ladder`](lowtier_role_tmem_readout_spill_ladder.md). This entry handles only one thing:
> scalars **belonging to another role** are still alive in this role's loop.

## Purpose

A warp-specialized kernel writes all roles in the **same** function, and ptxas does liveness analysis over the **whole control-flow graph**:
a value lives from its definition to **the last use on any path**. So there is a kind of register pressure that has nothing to do with the operator's data and is decided purely by the spelling —

```
peak(loop of role R) = R's own working set (fragments, addresses, counters)
                     + scalars belonging to **other roles** that are "not yet dead" inside R's loop
```

The second term is what this entry removes. Two levers (either one alone is enough, see the probe readings) plus one counter-example:

| Source | Mechanism | Lever |
|---|---|---|
| Role dispatch written as a string of independent `if …:`, and the compiler cannot prove they are mutually exclusive (see gotcha 1) | after finishing its own branch, each warp **falls back onto the main path** and goes on to check the next `if`; values needed by later roles are live inside every earlier role's branch | **D: mutually exclusive dispatch** `if … elif … else`, branches rejoin only at the end, with no other role body after them |
| Scalars read in a common prologue (especially smem read via inline asm with `has_side_effects=True`), including ring geometry (e.g. ring depth) | a side-effecting asm is not sunk / rematerialised by LLVM, so the value read is pinned in the prologue; ptxas will not replay an `ld.shared` to lower pressure either | **S: read in place at role entry**: each role reads / computes only the few scalars it needs at the start of its own branch |
| Run-time ring depth forked into several constexpr loop bodies (`if …: body(NST_X) else: body(NST_Y)`) | the loop body is duplicated and register allocation gets worse | **counter-example**, see gotcha 3: keep the geometry at run time, one loop body |

Neither D nor S changes the algorithm or the data flow; they only change the shape of **the same work** on the CFG.

## When needed (trigger)

- Dossier `spill_ld / spill_st` non-zero, `cuobjdump -res-usage` reports `STACK` > 0, and `REG` sits exactly at `65536 / blockDim` rounded down to a multiple of 8.
- Compile the cubin with `CUTE_DSL_LINEINFO=1` and aggregate `LDL` / `STL` by source line with `nvdisasm -c -g` (method below):
  - `STL` lands on the lines in the **prologue** that read the table / compute geometry, `LDL` on the lines in **another role's** loop that use it → this entry (D, S);
  - in SASS the geometry scalar is computed in the prologue and immediately `STL`'d → the geometry is computed in the prologue (a special case of S);
  - `STL` lands on this role's own ring bookkeeping lines (`ph ^ 1`, `slot = …`), and `LDL` right after `mbarrier.try_wait` fetches the barrier address or phase →
    **this only says the loop is at the ceiling and these lines are the batch that got squeezed out, not the root cause**. The root cause may be another role's scalars (this entry),
    or another peak segment in the same role ([`warp_specialization`](warp_specialization.md) gotcha 14, the read-back segment entries). Check both by line; do not assume it is this entry first.
- **Whole-kernel readings (two, reported as they are)**: on one warp-specialized whole kernel with `REG` at the ceiling and non-zero spill —
  - **S** (read geometry in place at role entry, one run-time loop body) landed once in a package with another change: `spill_ld / spill_st_sectors` 818295 / 421312 → 603367.5 / 286400
    (−26% / −32%), gate 0.1008 → 0.1010 ms (flat), NCU duration 90760 → 94468 ns (+4%, in the slower direction; same package, cannot be separated); after landing the census still showed `STACK 120`, `LDL 206 / STL 48`.
  - afterwards **D only** (a string of independent `if warp == …` turned into `if / elif / else`, nothing else changed): `spill_ld / spill_st_sectors` 603250 / 286400 → 600918 / 279296 (almost unchanged),
    gate 0.0996 → 0.1000 ms (+0.4%, slower direction, inside the noise), NCU duration 93088 → 93336 ns.

  That kernel's spill was later cleared by changing an epilogue read-back segment to read block by block (`REG` 128 → 112, spill → 0 / 0, gate 0.1000 → 0.1002 ms flat, NCU 93336 → 87056 ns) —
  i.e. the spill remaining on that tree was not caused by what this entry addresses. **No gain is claimed**: first confirm by a per-line census that "another role's scalars occupy this role's loop" before using this entry.

## Minimal runnable example

Below is the skeleton of the form recommended by the certification probe (D = elif, S = geometry read in place at role entry, run-time `nst`, `slot` / `ph` carried).
`ld_shared_u32` / `mbar_wait` / `mbar_arrive` are small inline-PTX helpers (a `mbarrier.try_wait.parity.shared::cta.b64` spin,
`mbarrier.arrive.shared::cta.b64`; spelling in [`ptx_escape`](ptx_escape.md)). `W_*`, `MB_FULL` / `MB_EMPTY`, `SLOT_BYTES`,
`SM_*` are Python constants. The ring depth at `W_NST` is a run-time value fixed in smem only after the prologue; `role` is this warp's role id,
also read from smem (when the role key comes from run-time state, the compiler cannot prove mutual exclusion).

```python
@cute.kernel
def k(p_tab: cute.Pointer, p_out: cute.Pointer, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    warp = tidx // 32
    lane = tidx % 32
    smem = cutlass.utils.SmemAllocator()
    sbase = smem.allocate_tensor(element_type=cutlass.Uint8,
                                 layout=cute.make_layout(SMEM_TOTAL), byte_alignment=1024).iterator.toint()
    s_ring, s_geo, s_mbar = sbase + SM_RING, sbase + SM_TAB, sbase + SM_MBAR
    # ---- prologue: only what all roles need (fill the table, init barriers, one bar.sync); read no role's scalars, compute no geometry ----
    ...
    bar_sync_all()
    role = ld_shared_u32(s_geo + W_ROLE + warp * 4)

    if role == 0:                                  # ---- consumer role ----
        nst = ld_shared_u32(s_geo + W_NST)       # S: geometry (a run-time value) read in place at role entry, one loop body
        c0 = ld_shared_u32(s_geo + W_C0)         # read only what it needs
        slot = cutlass.Int32(0)
        ph = cutlass.Int32(0)
        it = cutlass.Int32(0)
        while it < n:
            mbar_wait(s_mbar + 8 * (MB_FULL + slot), ph)
            ...                                    # read s_ring + slot * SLOT_BYTES, compute, write out
            mbar_arrive(s_mbar + 8 * (MB_EMPTY + slot))
            it = it + 1
            slot = slot + 1
            if slot == nst:
                slot = cutlass.Int32(0)
                ph = ph ^ 1
    elif role == 1:                                # D: elif, not a new if
        nst = ld_shared_u32(s_geo + W_NST)       # the producer role reads it again itself (gotcha 5)
        ...                                        # wait on empty with ph ^ 1, otherwise as above
    elif role == 2:
        a0 = ld_shared_u32(s_geo + W_A0)         # scalars unique to this role are born only here
        ...
```

**Either one** of D and S is enough to make "another role's scalars" die in this role's loop (in the probe, the D-only and S-only cells each have 0 spill); using both is the recommended form.

### Acceptance (census)

```python
import os, re, subprocess, glob
os.environ.update(CUTE_DSL_KEEP_CUBIN="1", CUTE_DSL_DUMP_DIR=d, CUTE_DSL_LINEINFO="1",
                  CUTE_DSL_CACHE_DIR=fresh_dir, CUTE_DSL_DISABLE_FILE_CACHING="1")
compiled = cute.compile(launch, ...)                           # a .cubin appears under d/
cub = max(glob.glob(d + "/**/*.cubin", recursive=True), key=os.path.getsize)
print(subprocess.run(["cuobjdump", "-res-usage", cub], capture_output=True, text=True).stdout)  # REG / STACK
sass = subprocess.run(["nvdisasm", "-c", "-g", cub], capture_output=True, text=True).stdout
OPC = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")
SRC = re.compile(r'//##\s*File\s*"([^"]+)",\s*line\s*(\d+)')
line, tally = "?", {}
for ln in sass.splitlines():
    if (m := SRC.search(ln)): line = "%s:%s" % (os.path.basename(m.group(1)), m.group(2)); continue
    if (m := OPC.search(ln)) and m.group(1).split(".")[0] in ("LDL", "STL"):
        t = tally.setdefault(line, [0, 0]); t[m.group(1).startswith("STL")] += 1
```

Criterion: `STACK:0`, and `LDL` = `STL` = 0 in SASS. Use a **separate** dump directory and cache directory for each variant (a JIT file-cache hit produces no dump).

## Gotchas

1. **A serial `if` chain equals elif only when the compiler can prove mutual exclusion.** When every `if` compares **the same, unmodified** SSA key against a constant (`warp`, or a `role` read only once in the prologue),
   the serial chain measured no spill, the same as the exclusive chain (presumably LLVM jump threading: entering the `role == 0` branch tells it that the later `role == 2` is false; the IR was not checked).
   When the conditions are **not on the same SSA value**, that no longer works: each `if` re-reads the key itself (a side-effecting `ld.shared`), the condition is a combination of different predicates (warp range plus a run-time flag), or the key is modified between two tests.
   In that case, after finishing its own branch, each warp still passes through all the later `if`s, and the inputs of later roles keep occupying registers throughout the earlier roles' entire loops (the probe's positive control measured lots of `LDL/STL`).
   Do not rely on the compiler to prove mutual exclusion: write `if / elif / else`.
   This cost only holds for values that **differ per lane**: with the same seq + prologue spelling, if the other roles' scalars are warp-uniform (every lane reads the same smem word),
   it measured 0 spill (presumably they go into uniform registers and do not use the per-thread budget; not verified separately).
2. **A value read by side-effecting asm is pinned where it is.** A `has_side_effects=True` `ld.shared` written in the prologue is not sunk by LLVM into the only branch that uses it,
   and ptxas will not replay it under high pressure — either read it again at the entry of the role that uses it, or switch to a non-side-effecting read so the compiler can sink it
   (for the latter you must confirm yourself that the barrier / visibility semantics are not broken; this entry's probe did not test that path).
3. **Do not fork run-time geometry into several constexpr loop bodies.** Intuitively `% NST` becomes an immediate and barrier addresses become "base + immediate", which should save registers;
   measured the opposite: the two-body version in the certification round gave elif + role entry `STACK 32`, `LDL 18 / STL 15`, elif + prologue `STACK 32`, `LDL 12 / STL 14`,
   seq + role entry `STACK 32`, `LDL 17 / STL 15`, while the same configurations with one body (run-time `nst`) were all 0; the SASS instruction count was about half again as large.
   Across the four probe rounds, 15 of 16 two-body cells spilled (an earlier round read `STACK 24`, about 10 each; the numbers float with ptxas allocation); the only cell that did not spill was the first round's
   seq + prologue (`REG 127`, 0 spill), so the direction is not absolute and must be confirmed by census. What needs saving is **the other roles'** registers, not the single `nst`.
4. **Carrying or recomputing the phase both work.** Carrying `slot` / `ph` (`if slot == nst: slot = 0; ph ^= 1`) and recomputing `slot = it % nst`, `ph = (it // nst) & 1`
   give the same census in the same configuration (both 0 in the recommended form, and almost the same spill in the positive control); the producer role waits on empty with `ph ^ 1`.
5. **The geometry each role computes for itself must agree.** If the producer role's `nst` differs from the consumer's, it deadlocks or reads the wrong slot;
   so every role reads the geometry from **the same** smem word (read only after the prologue `bar.sync`) (or recomputes it from that word).
6. **Read the census by source line, not just totals.** An `STL` outside the loop executes once, an `LDL` in the loop body executes every iteration; when comparing two versions, look at the `LDL` inside the loop body.
7. **The positive control's allocation shape can be outlandish.** In the certification round the three seq + prologue cells (including the two-body one) were reported by `cuobjdump` as `REG 32`, `STACK` about 1.3 KB, 431–773 `LDL` — ptxas took an allocation there different from
   "squeeze out a few values at the ceiling"; the cause was not isolated. This entry only uses it to prove "there is spill"; do not take its spill volume as the typical scale of this disease.

## Sources

- PTX ISA: `mbarrier.try_wait.parity`, `mbarrier.arrive`, `mbarrier.init` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
- CUDA C++ Programming Guide, register limits and `maxntid` / per-SM register file — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximum-number-of-registers-per-thread
- CUDA Binary Utilities (`cuobjdump -res-usage`, `nvdisasm -c -g`) — https://docs.nvidia.com/cuda/cuda-binary-utilities/
- CuTe DSL public API: `cute.kernel` / `cute.jit`, `cutlass.range_constexpr`, `cutlass.utils.SmemAllocator`, `cute.compile`;
  compiled-artifact channel `CUTE_DSL_KEEP_CUBIN` / `CUTE_DSL_DUMP_DIR` / `CUTE_DSL_CACHE_DIR` / `CUTE_DSL_DISABLE_FILE_CACHING` / `CUTE_DSL_LINEINFO`.
- Certification probe: `PROBE PASS max_diff=0.0`.
