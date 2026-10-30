# merge_face_lds128_read_map (a lane-major fp32 exchange face: the bank accounting after writing 8 B / lane (`STS.64`) and reading 16 B at once (`LDS.128`) under a different thread mapping)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cells `merge V0/V1/V2`; the probe is not included in this release; verdicts and
> timing / NCU readings under Sources). The mechanism is the same as the same-named entry in the CuTe DSL tree; the DSL's view-alignment spelling is not carried over, and the readings were re-taken.
>
> One file per hardware interface. Operator-agnostic: mechanism only. First establish with [`smem_traffic_triage`](smem_traffic_triage.md) that it is conflict-type, then read this entry.

## Purpose

Several warps each land their register fp32 fragments **in lane-major order** into an smem face (each row = 32 lanes × 2 fp32, 8 B per lane, one `st.shared.v2.f32`);
after a sync, **a different thread mapping** reads 16 B at once (`ld.shared.v4.f32`) to stitch the warps' fragments together (cross-warp reduction / transpose / change of ownership).
The write side is naturally conflict-free; **conflicts almost all come from the read-side mapping**: for a given mapping, compute the wavefronts per `LDS.128`; when it is not 4 per instruction, push it back down in one of two ways:
**pad the row pitch by 16 B**, or **change the read-side mapping** so that the 8 threads of one phase spread along contiguous addresses within a row.

## Mechanism

- **An smem access proceeds by phases, at most 128 B per phase**: `LDS.128` / `STS.128` is 4 phases × 8 threads, and a phase is conflict-free when the 8 16 B blocks'
  `(addr/16) % 8` are pairwise distinct (or the same address, broadcast); `LDS.64` / `STS.64` is 2 phases × 16 threads. A conflict-free `LDS.128` = 4 wavefronts, the worst 32.
- **Write side**: each lane writes 8 B to `face + row*pitch + lane*8`; a half-warp writes a **contiguous** 128 B, covering all 32 banks whatever the row pitch —
  padding the row pitch does not make the write side more expensive.
- **Read side**: a common mapping `t → (row = t % R, chunk = t / R)`, `addr = row*pitch_B + chunk*16`. When only row changes within a phase (8 adjacent t),
  the conflict degree = `gcd(pitch_B / 16, 8)`: pitch 256 B (64 fp32) → 8-way; 272 B (68 fp32) → conflict-free.
  Another fix: `row = t / 8, chunk = t % 8`, so the phase walks a contiguous 128 B within a row, no padding needed.

## When needed (trigger)

- NCU's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is several times above budget while the write side is clean; in SASS the read side is `LDS.128` and the write side `STS.64`;
- the read-side address is written as `row(t) * pitch + col(t)`, the pitch is an integer multiple of 256 B, and `t % 8` lands on `row(t)`.

## Minimal runnable example

4 warps, a 16-row face per warp, a row pitch of `P` fp32, 128 reading threads. The spelling matches the certification probe:

```cpp
constexpr int P = 68;                                   // 64 = the natural lane-major length; +4 = 16 B padding
__shared__ __align__(16) float face[4 * 16 * P];
float* fw = face + warp * 16 * P;
for (int r = 0; r < 16; ++r)                            // write side: lane-major, 8 B / lane
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" :: "r"(smem_u32(fw + r * P + 2 * lane)), "f"(a[r]), "f"(b[r]) : "memory");
__syncthreads();
const int row = t % 16, ch = t / 16;                    // read side: only the row changes within a phase (8-way conflict at P = 64, conflict-free at P = 68)
// const int row = t / 8, ch = t % 8;                   // or: the phase spreads along a row, conflict-free even at P = 64
for (int ww = 0; ww < 4; ++ww)
  for (int h = 0; h < 2; ++h) {
    float4 x;
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(x.x), "=f"(x.y), "=f"(x.z), "=f"(x.w)
                 : "r"(smem_u32(face + ww * 16 * P + row * P + (ch + 8 * h) * 4)));
    // consume x
  }
```

The write side and the read side use **the same** `P` constant, so the two access points are the same bijection; switching to 68 touches only this one constant.

## Gotchas

1. **A row pitch that is a multiple of 256 B, with the read side varying only by row within a phase, is an 8-way conflict.** The numerics are entirely right; it shows only in `bankconf_ld` or timing.
2. **The padding must make `pitch_B/16` odd** (model derivation; the probe measured only +0 and +16 B): +32 B → 2-way, +64 B → 4-way, +128 B → still 8-way.
3. **The row-pitch constant of the write side and the read side must be the same name**: change only one side and you read the adjacent row's data, with no error.
4. **The face gets bigger**: per warp from 16×256 B to 16×272 B; recheck boundaries when it is overlaid on another buffer.
5. **The address of `ld.shared.v4` must be 16 B aligned**: `P*4`, the length of each warp's face and the column offsets must all be multiples of 16 (P = 68 → 272 B satisfies it).
6. **Changing the mapping ripples downstream**: the elements a thread gets change, so all later code that writes by thread ownership must follow; padding touches only addresses.

## Sources

- CUDA C++ Programming Guide, Shared Memory / bank conflicts — https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x
- PTX ISA: vector forms of `ld.shared` / `st.shared`, `%clock64` — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- NCU metrics: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`.
- GPU-certification probe (`merge V0` = row mapping P 64, `V1` = row mapping P 68, `V2` = chunk mapping P 64), verdict
  `PROBE PASS max_diff=0.0`: `E2-merge-V0-rowmap-P64` / `V1-rowmap-P68` / `V2-chunkmap-P64` (4096 values read back bitwise equal).
  NCU: V0 read wavefronts 524288 / read instructions 16384 = 32 each, `bankconf_ld` 458752; V1, V2 4 each, `bankconf_ld` 0; write side `bankconf_st` 0 for all three.
  `%clock64` (512 repetitions × 8 `ld.shared.v4`): V0 / V1 = 6.31, V2 / V1 = 1.00.
