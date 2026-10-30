# rmem_array_runtime_index_local_frame (a small per-thread array accessed with a run-time index → the whole array goes to local memory: stack frame > 0 while spill = 0)

> One file per hardware interface. Operator-agnostic; mechanism only. The language surface is CUDA C++ (`nvcc`); the mechanism lives in ptxas,
> and the same array pattern in CuTe DSL (an rmem tensor accessed with a run-time index) ends up in the same local memory.
>
> **Status**: ✅ Verified (certified by a GPU probe, not included in this release:
> `PROBE PASS max_diff=0.0`, certification round 1; this entry uses its contracts C1 (the record-and-consume sums of all four spellings equal numpy item by item)
> and C2 (the run-time-indexed array: `stack=40`, 6 PTX `.local` accesses, NCU local sectors 342; the two rewrites K / U collected here: stack 0, `.local` 0, sectors 0)).
> The "CuTe DSL surface: iterator views" subsection: ✅ Verified (a second GPU probe:
> `PROBE PASS max_diff=0.0`, certification round 7, all four PROBE lines pass).
>
> **Certified surface**: an array of ≤ 3 records of 3 ints each per thread, computed independently per thread, no smem; only register residency and numerics were measured, not speed.

## Purpose

Each thread holds a small table (≤ K records), filled by a data-dependent loop and then read with a loop counter. The register file has no indirect addressing:
**when the write or read index is a run-time value**, ptxas usually puts the whole array into per-thread **local memory** (measured so for CUDA C++ counterexample T;
exception: for a very small array with a run-time index on only one side, ptxas may lift it back into registers, see the boundary reading under "CuTe DSL surface", so always judge by STACK and SASS `LDL/STL`) —
`ptxas -v` reports `N bytes stack frame`, SASS has `STL`/`LDL`, NCU's `l1tex__t_sectors_pipe_lsu_mem_local_op_{ld,st}` is non-zero
(the loop's dossier records these two readings as `spill_ld` / `spill_st`), **while ptxas's own `spill stores/loads` is still 0 bytes**.
This entry gives two spellings that keep the table in registers.

## When needed (trigger)

- `spill_ld / spill_st` is non-zero in the dossier or NCU, but `ptxas -v` reports `0 bytes spill stores`, only the `stack frame` is above 0, and the register count is not high.
  This is not register pressure: changing the register level, tuning `__launch_bounds__`, shrinking the live set — none of it helps.
- The code has `struct { int a[K]; int b[K]; … }` or `int x[K]`, filled by a `while` / data-dependent loop and then read as `x[it]` (`it` a run-time
  loop variable). Several roles each holding a copy and each reading it by `it` is a common shape.
- Corroborating sign: `stall_long_sb` is high (`LDL` is also recorded on the long scoreboard).

## Minimal runnable example

The record contents are written abstractly here as `make_rec(state, …)` / `advance(state)`; counterexample T and spellings K / U correspond to the kernels of the same names in the certification probe.

```cpp
struct Rec { int a, b, c; };

// ---- Counterexample T: write index n and read index it are both run-time -> the whole table goes to local (stack frame > 0, LDL/STL) ----
Rec tab[3]; int n = 0; State st = init(...);
while (has_next(st)) { tab[n] = make_rec(st); ++n; st = advance(st); }
for (int it = 0; it < n; ++it) use(tab[it]);

// ---- Spelling K: a scalar cursor. The next record is derived from the previous one; the state is only a few scalars ----
State st = init(...); Rec r;
while (has_next(st)) { r = make_rec(st); st = advance(st); use(r); }

// ---- Spelling U: keep the array, fully unroll both filling and reading to the bound K + guards, so every index is a compile-time constant ----
Rec tab[3]; int n = 0; State st = init(...);
#pragma unroll
for (int r = 0; r < 3; ++r) {
    tab[r] = Rec{0, 0, 0};
    if (has_next(st)) { tab[r] = make_rec(st); n = r + 1; st = advance(st); }
}
#pragma unroll
for (int r = 0; r < 3; ++r) if (r < n) use(tab[r]);                 // the loop body is replicated K times
```

Acceptance (the first and third must pass; the second is for reference):

- `nvcc -Xptxas=-v` reports **`0 bytes stack frame`** for the kernel;
- (reference) in the PTX from `nvcc -ptx`, that `.entry` has 0 `ld.local` / `st.local` — when there is `.local` but stack frame 0 and no `LDL/STL` in SASS, the latter two are authoritative;
- NCU `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` and `..._st.sum` are both 0.

### CuTe DSL surface: rebuilding a view from a fragment's iterator is **not** a trigger (✅ Verified, probe `PROBE PASS max_diff=0.0`)

> What is certified is a negative result: "iterator views make a fragment land in local" does not hold; plus a positive control (a run-time index really is caught by the same detection)
> and a boundary case (PTX has `.local`, but ptxas lifts it back into registers).

In CuTe DSL, `cute.make_rmem_tensor` is a per-thread alloca. In a "TMEM read-back → fp32 / fp16 fragment → write out" stretch of code
(the fragment allocated once outside a run-time loop `cutlass.range(n, unroll=1)`, with run-time branches in the loop body), each of the four patterns below was tried in a "view" and a "compact" form:

| Factor | View form | Compact form |
|---|---|---|
| d | `cute.make_tensor(h32.iterator + j*16, cute.make_layout(16))` takes half of a 32-element fragment | copy element by element with `range_constexpr` into a `make_rmem_tensor((16,), T)` |
| h | `cute.make_tensor(h16.iterator, cute.make_layout(16))` flattens the whole fragment | as above, copy element by element into a compact `(16,)` |
| f | `r16f = cute.make_tensor(r16.iterator, cute.make_layout(16))` as an alias, used to write gmem / accumulate | no alias; use `r16`'s shape directly |
| g | a compile-time-unrolled `j` selects an independent rmem tensor `buf[j]`, copied into under a run-time guard | no guard; copy unconditionally for each `j` |

Conclusions (GPU-verified):

- All 10 factorial variants (all views, all compact, single factors, three factors) have **`STACK 0`, no `LDL`/`STL` in SASS, no `.local` in PTX**, and numerics equal to numpy (max_diff 0.0);
  registers 56–94, varying with the factor combination, none high.
  When `j` is `range_constexpr`, `buf[j]` are distinct Python objects (each its own alloca); the run-time guard only decides whether to write, it does not create a run-time index.
- Positive control: a compact `(128,)` rmem tensor kept across iterations inside a `cutlass.range` loop, **written with a run-time index and read with a run-time index**
  (`t[wi] = t[ri] + 1`) — this does land in local (measured `STACK 528`, SASS `LDL` 2 / `STL` 33, PTX `ld.local` 2 / `st.local` 33).
- Boundary: when a compact `(16,)` fp32 tensor is **only read** once with a run-time index (`t[(tidx + u) % 16]`), `ld.local` / `st.local` appear in PTX (measured 1 / 4),
  but ptxas lifts it back into registers: `STACK 0`, no `LDL`/`STL` in SASS. So **`.local` in PTX does not mean local traffic at run time**;
  judge by `STACK` from `cuobjdump -res-usage` and by SASS `LDL`/`STL` (or NCU local sectors). Conversely,
  no `.local` in PTX only rules out the "run-time-indexed array" source; register spills still produce local traffic (gotcha 1).
- So: when the dossier shows local traffic and the register count is not high, **do not spend time rewriting iterator views**. Following the rule above, find the real run-time index
  (a `cutlass.range` loop variable, a run-time-computed `Int32` used to index an rmem tensor, or a run-time value used to select "which fragment");
  then check the `cosize` of the fragment layout ([`cute_rmem_fragment_compact_layout`](cute_rmem_fragment_compact_layout.md)).
  Re-check every change with the acceptance below; compile PTX with `CUTE_DSL_LINEINFO=1` and see which source line the nearest `.loc` before each `ld.local` / `st.local` points to.

Certified surface: single CTA, 128 threads, Float32/Float16 fragments of 16/32 elements; only register residency and numerics measured, not speed; the combination of a run-time index **on top of** an iterator view was not tested.

Whole-kernel reading (from the full kernel where this subsection was found; not certified by an independent probe): the local traffic was eventually traced to a different pattern — declaring the alignment of a global pointer with `Pointer.align(16)`.
After switching to stating divisibility with `cute.assume(offset, divby=…)` and no longer calling `.align()`, that 16 B stack frame disappeared,
gate 0.0588 → 0.0584 ms (−0.7%). The explanation offered at the time is that the round-up in `.align()` materialises the pointer into a 64-bit register, which is hoisted out of the loop and then spills;
this explanation is not certified by this entry; when you meet the same symptom, confirm it yourself with the `.loc` localisation method above.

## Gotchas

1. **ptxas's `spill stores` being 0 does not mean there is no local traffic.** A run-time-indexed array is counted in the `stack frame`, not in spill; NCU's local
   sectors count both sources together. Accept on `stack frame == 0` or no `.local` in PTX, not on `spill`. The probe's counterexample: `(stack, spill_st, spill_ld)
   = (40, 0, 0)`, NCU local ld / st 171 sectors each.
2. **Both filling and reading must use compile-time indices.** The probe's spelling U unrolls both sides; a spelling that unrolls only the read while filling is still a `while` was not tested by the probe — by
   the rule (the write index is still run-time) it should still land in local; confirm it yourself with the acceptance above. Either unroll both sides (U), or do not build the table at all (K).
3. **Spelling K can only be used forward and monotonically.** When you need to skip around (e.g. peek at the next record), K must hold one more cursor pushed ahead, or switch to spelling U.
4. **When several roles each reproduce the same sequence, share one `make_rec` / `advance`**; do not hand-copy it at every call site — the same `it` must yield the same record.
5. **Spelling U replicates the read loop body K times.** If the loop body is a large chunk of code, the code size multiplies (the probe's loop body is tiny; the I-cache effect was not measured).
   Then prefer K: turn the records into scalars and keep the loop itself rolled (`#pragma unroll 1`).
6. **After rewriting, look at the spill axis on the whole kernel first, then judge speed.** On the full kernel where this was found, rewriting away the run-time-indexed table (together with an smem store-width change in the same package)
   brought `spill_ld/st` from about 20k / 50k sectors to 0 and registers 101 → 93, but device time moved by only −1.2%, inside the loop's 1.5% noise band —
   when the local traffic is not on the critical path, eliminating it does not necessarily show a speedup.

## Sources

- CUDA C++ Best Practices Guide, "Local Memory": arrays whose indices the compiler cannot determine to be constant are placed in local memory —
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#local-memory
- PTX ISA: the `.local` state space, `ld.local` / `st.local` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- NCU metrics: `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st.sum`.
- Certification probe: `PROBE PASS max_diff=0.0` (round 1, C1 / C2).
- Certification probe (CuTe DSL surface, iterator-view subsection): `PROBE PASS max_diff=0.0` (round 7; numerics of the 10 variants / no local in the 10 variants / local in the positive control / boundary: PTX `.local` but STACK 0).
- C3 / C4 of the first probe certify a different mechanism, see [`smem_sw64_row_face_store_width`](smem_sw64_row_face_store_width.md).
