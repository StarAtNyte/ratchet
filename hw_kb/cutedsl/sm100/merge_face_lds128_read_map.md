# merge_face_lds128_read_map (lane-major fp32 exchange face: the bank accounting when the write side uses `STS.64` and the read side uses `LDS.128` under a different thread mapping)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> Applies to this shape: several warps each store their register-held fp32 fragments into an smem face **in lane-major order**
> (each row = 32 lanes × 2 fp32, 8 B per lane, one `STS.64`); after a sync, a **different thread mapping**
> reads 16 B at a time (`LDS.128`) to stitch the warps' fragments together (cross-warp reduction / transpose / change of ownership).
> The write side is naturally conflict-free; **the conflicts are almost always in the read-side mapping**.
>
> **Status**: ✅ Verified (GPU probe, not included in this release; `PROBE PASS max_diff=0.0`, certification round 1; all five contracts passed in the same round: the values read by `data-V0` (P=64 row mapping) / `data-V1` (P=68 row mapping) / `data-V2` (P=64 blocked mapping) are bit-equal to numpy; `lds-timing`: `clock64` timing of 512×8 `ld.shared.v4.f32` per warp gives V0/V1 = 3.842, V2/V1 = 0.999 (the bank model gives 32 / 4 / 4 wavefronts per instruction); `sts-timing`: write side of 512×16 `st.shared.v2.f32`, P=68/P=64 = 0.988, padding does not make the write side more expensive)
>
> **Certified surface**: one geometry — 4 warps, 16 rows per warp face, row pitch P = 64 / 68 fp32, 128 threads on the read side;
> the timing uses bursts of inline-PTX `ld.volatile.shared.v4.f32` / `st.volatile.shared.v2.f32`, the data contracts use
> `cute.autovec_copy` (data bit-exact, but the width of the SASS accesses it lowers to was not separately censused). Only padding +0 and +16 B were measured;
> the conflict degrees for other paddings below are derived from the model in (c).

## Purpose

Given the lane-major face of the write side and the read side's "thread → (row, 16 B chunk)" mapping, compute how many
wavefronts each `LDS.128` takes; when it is not 1 per phase, bring it back down with one of two remedies: **pad the row pitch by 16 B**, or **change the read-side mapping**
so that the 8 threads of one phase spread along contiguous addresses within a row.

## When needed (trigger)

- NCU `bankconf_ld` (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`) is several times over budget
  while `bankconf_st` is clean; in SASS the read side is `LDS.128` and the write side `STS.64`;
- the read-side address has the form `row(t) * pitch + col(t)`, the **pitch is a multiple of 256 B** (64 fp32 = 32 lanes × 8 B,
  exactly the natural length of a lane-major row), and `t % 8` lands on `row(t)`;
- this only applies to the conflict kind. Classify first with [`smem_traffic_triage`](smem_traffic_triage.md): if wavefronts are over the line but bankconf is clean,
  it is the volume kind and not this entry's business.

## Mechanism

### (a) An smem access proceeds by phases; each phase consumes 128 B

smem has 32 banks of 4 B each. A warp-wide access is split into several phases, each serving at most 128 B:

| Instruction | Bytes per thread | Phase split (contiguous lanes) | Condition for a conflict-free phase |
|---|---|---|---|
| `STS.64` / `LDS.64` | 8 B | 2 phases × 16 threads | the 16 8 B chunks fall on 16 distinct bank pairs (or same-address broadcast) |
| `LDS.128` / `STS.128` | 16 B | 4 phases × 8 threads | the `(addr/16) % 8` of the 8 16 B chunks are pairwise distinct (or same-address broadcast) |

The number of wavefronts a phase actually needs equals "the maximum number of distinct 4 B words accessed on one bank". The
wavefront count of the whole instruction is the sum over its phases: conflict-free `LDS.128` = 4, worst case = 32 (8-way). The probe's `model_wavefronts()`
is exactly this model; the timing ratio is 3.84× (not 8×) because the clock reading also contains issue and latency, not just the wavefront count.

### (b) Write side: lane-major rows, `STS.64` is always clean

Each lane writes 8 B to `face + row*pitch + lane*8`. A half-warp writes one **contiguous** 128 B segment,
so whatever `pitch` is (as long as it is 8 B aligned), it occupies each of the 32 banks exactly once. Hence **padding the row pitch does not make the write side more expensive**;
that is what the probe's `sts-timing` contract checks (P=68 / P=64 = 0.988).

### (c) Read side: the 8 threads of a phase must not differ only in "row"

A common read-side mapping is `t → (row = t % R, col = f(t // R))`, each thread reading 16 B from `(row, col)`:

```
addr(t) = row(t) * pitch_B + col(t) * 16
bank_quad(t) = (addr(t) / 16) % 8
```

If only `row` varies within a phase (`t = 8q .. 8q+7`), then with `pitch_B = 256 B`, `256 / 16 = 16 ≡ 0 (mod 8)`,
so the 8 threads fall on the **same** bank quad, an 8-way conflict, and the `LDS.128` grows from 4 wavefronts to 32.

**Remedy 1: pad the pitch by 16 B (pitch 64 → 68 fp32, 256 B → 272 B).** `272/16 = 17 ≡ 1 (mod 8)`,
so adjacent rows are offset by one quad and 8 rows cover all 8 quads. General form of the model: when the 8 threads of a phase vary only over 8 adjacent rows,
the conflict degree = `gcd(pitch_B / 16, 8)`, so it is conflict-free when `(pitch_B / 16)` is odd.

**Remedy 2: change the read-side mapping so a phase spreads along a row.** For example `chunk_id = t + 128*j`, `row = chunk_id // 16`,
`col = chunk_id % 16`. Then a phase reads 128 contiguous bytes of one row, conflict-free without padding.
The cost is that each thread now gets a different set of data, and the downstream code unrolled by thread ownership must change with it.

Both remedies only touch addresses; the data bijection is unchanged. The probe's `data-*` contracts compare the values read by the three mappings bit by bit.

### (d) Same address is a broadcast

If several threads of one phase read the **same** address (e.g. one scalar per row shared by several threads), that is a broadcast
and does not count as a conflict (the bank model of the CUDA Programming Guide). The probe did not test this shape separately.

## Minimal runnable example

```python
import cutlass, cutlass.cute as cute, cutlass.utils as cutlass_utils
F32 = cutlass.Float32
NWARP, ROWS = 4, 16
P = 68                       # row pitch (number of fp32): 64 = natural lane-major length, +4 = 16 B padding
WF = ROWS * P                # length of each warp's face (fp32); 16 B aligned: WF*4 % 16 == 0

def _view(base_ptr, off, shape, stride, align):
    return cute.make_tensor((base_ptr + off).align(align), cute.make_layout(shape, stride=stride))

# inside the kernel body:
smem = cutlass_utils.SmemAllocator()
face = smem.allocate_tensor(F32, cute.make_layout((NWARP * WF,)), byte_alignment=128)
face_p = face.iterator

# write side: per warp, per row, lane-major 8 B (STS.64)
t2 = cute.make_rmem_tensor((2,), F32)
for kk in cutlass.range_constexpr(ROWS):
    t2[0] = frag[2 * kk]; t2[1] = frag[2 * kk + 1]
    cute.autovec_copy(t2, _view(face_p, warp * WF + kk * P + lane * 2, (2,), (1,), 8))
cute.arch.barrier()

# read side: row = tidx % 16 (a phase varies only over rows), col = two 16 B chunks starting at 8 * (tidx // 16) (LDS.128)
row, col = tidx % 16, 8 * (tidx // 16)
t4 = cute.make_rmem_tensor((4,), F32)
for w in cutlass.range_constexpr(NWARP):
    for j in cutlass.range_constexpr(2):
        cute.autovec_copy(_view(face_p, w * WF + row * P + col + 4 * j, (4,), (1,), 16), t4)
        ...   # consume t4
```

The write side and the read side must use the **same** `P` constant, so that the two call sites are the same bijection. Switching to 68 touches only this one constant;
the rest of the code is unchanged.

The full self-contained version (including the bit-by-bit comparison of the three mappings and the `clock64`-timed bank-conflict comparison) is in the certification probe.

## Gotchas

1. **A pitch that is a multiple of 256 B, with a read-side phase varying only over rows, is an 8-way conflict.** The numerics are entirely correct; it shows up only in
   `bankconf_ld` or in timing. Before shipping, write out the quad set of every phase with the formula in (c) and check it;
   the probe's `model_wavefronts()` is this model, and it gives 32 / 4 / 4 for the three mappings.
2. **The padding must make `(pitch_B/16)` odd.** Model derivation (the probe only measured +0 and +16 B): starting from a 256 B row,
   +16 B → 1-way; +32 B → 2-way; +64 B → 4-way; +128 B → still 8-way.
3. **The pitch constant of the write side and the read side must be the same name.** If only the read side (or the write side) is changed, the values are silently misplaced: you read
   the data of an adjacent row, with no error.
4. **The face gets bigger.** Each warp's face grows from `ROWS*256 B` to `ROWS*272 B`. If the face is overlaid on another buffer, re-check
   that all the enlarged warp faces still fall inside that buffer.
5. **`LDS.128` addresses must be 16 B aligned.** `P*4`, `WF*4` and the column offset `col*4` must all be multiples of 16. P=68 → 272 B, satisfied.
   `_view(..., align=16)` is an alignment promise to the compiler; what happens when the promise is false was not tested by the probe.
6. **Remedy 2 (changing the mapping) has downstream consequences.** Which elements a thread gets changes, so all later code written by thread ownership must change with it
   (coalescing of the gmem write-back, destination addresses of later exchanges, etc.). If you only want to bring `bankconf_ld` down without touching ownership, choose remedy 1.
7. **Accept on NCU counts, not just wall clock.** If the exchange face is traversed only once in the kernel, the conflicts may be a small fraction of total time;
   confirm the remedy took effect by comparing `bankconf_ld` readings before and after.

## Sources

- CUDA C++ Programming Guide, Shared Memory / bank conflicts section (compute capability 5.x and later: 32 banks of 4 B each;
  64/128-bit accesses are split into half-warp / quarter-warp phases): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x
- Nsight Compute metrics `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` / `..._op_st.sum`:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- Public infra API: `cutlass.utils.SmemAllocator.allocate_tensor`, `cute.autovec_copy`, `cute.make_tensor`,
  `cute.make_layout`, `cute.arch.barrier`.
- Certification probe: `PROBE PASS max_diff=0.0`.
