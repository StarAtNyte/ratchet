# softmax_segment_ilp (three language-independent SASS facts in a named-barrier handoff segment: `bar.arrive` is not a register scheduling fence / a ternary max is not `FMNMX` / byte flags must be readable in bulk)

> **Certification source: B200 GPU-certified in CUDA mode (all 5 contracts PROBE PASS; the probe is not included in this release).**
> Certifies only SASS shape and numerics, **with no timing conclusion at all**: all three are facts of "how the source is written → what instructions it compiles to"; which one is worth anything in your
> kernel must be read off your own segment ledger (see "When needed").
>
> One file per hardware interface. Operator-agnostic: mechanism only. This entry covers no row reduction / online softmax algorithm; it covers only what three common spellings within the segment become in SASS when two warpgroups
> hand off one value one way through a named barrier.

## Purpose

Two warpgroups hand off a statistic computed in registers (e.g. a row max) through `bar.arrive` / `bar.sync` (same barrier id, same thread count):
the publishing side does `STS` → `bar.arrive`, the consuming side `bar.sync` → `LDS`. This entry answers three questions about the publishing side's segment:

1. does SASS change if the publishing side's register work (scaling accumulators etc.) is written before / after `bar.arrive` in the source;
2. is the instruction class the same whether the running max is written as `fmaxf` or `v > m ? v : m`;
3. how many `LDS` it takes to read 16 byte flags per thread from smem.

## When needed (trigger)

- The segment ledger shows the publishing side stalling a lot before `BAR.ARV`, and you plan to "publish earlier / move the scaling later" — read §1 first: the source position does not change SASS.
- The running max in SASS is a string of `FSETP` + `FSEL` pairs rather than `FMNMX` / `FMNMX3` — §2.
- A thread reading its own 16 byte flags in the loop body gives 16 `LDS.U8`, and the segment's main stall is
  `short_scoreboard` / `mio_throttle` — §3.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// Two warpgroups (256 threads) hand off through named barrier 1; taken from this entry's probe.
__device__ __forceinline__ float ex2_approx(float x) {
  float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}

__global__ void __launch_bounds__(256, 1) handoff(const float* acc_in, const float* mold, const float* mnew,
                                                  float* acc_out, float* m_out) {
  __shared__ float mface[128];
  const int tid = threadIdx.x;
  if (tid < 128) {                                   // publishing warpgroup
    const int g = blockIdx.x * 128 + tid;
    float acc[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = acc_in[g * 64 + i];
    const float mn = mnew[g];
    const float scale = ex2_approx(mold[g] - mn);
    mface[tid] = mn;                                 // the STS must precede the arrive (arrive orders only shared accesses)
    asm volatile("bar.arrive 1, 256;" ::: "memory"); // non-blocking: counts toward barrier 1's 256 arrivals
    #pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] *= scale;    // written before or after the arrive, the SASS is identical instruction by instruction (§1)
    #pragma unroll
    for (int i = 0; i < 64; ++i) acc_out[g * 64 + i] = acc[i];
  } else {                                           // consuming warpgroup
    asm volatile("bar.sync 1, 256;" ::: "memory");   // same id, same thread count
    m_out[blockIdx.x * 128 + tid - 128] = mface[tid - 128];  // the LDS must follow the sync
  }
}

// §2: write the running max with fmaxf (-> FMNMX3 / FMNMX), not a ternary (-> FSETP + FSEL)
float m = v[0];
#pragma unroll
for (int i = 1; i < 16; ++i) m = fmaxf(m, v[i]);

// §3: 16 byte flags in one 128-bit read, then split in registers
__shared__ __align__(16) uint8_t fs[256 * 16];
const uint4 q = reinterpret_cast<const uint4*>(fs)[threadIdx.x];   // one LDS.128
const uint32_t w[4] = {q.x, q.y, q.z, q.w};
// flag c: (w[c >> 2] >> (8 * (c & 3))) & 0xff
```

## Gotchas

1. **`bar.arrive` is transparent to register work: the source position is a zero lever.** In the probe the two variants with the 64 scaling `FMUL`s written before and after `bar.arrive`
   have SASS of identical length (234 instructions), `BAR.ARV` at the same position (instruction #91), and before `BAR.ARV` only
   1 `FMUL` and 1 `MUFU` remain in both — the other 63 are scheduled after `BAR.ARV` wherever the source puts them. `asm volatile` +
   `"memory"` only constrains the order of memory accesses and other volatile asm, not pure register arithmetic. To shorten the other side's wait, the only way is to reduce the segment's
   total work / number of memory accesses / dependency hops, not to move the `arrive`.
2. **`STS` before `bar.arrive` and `LDS` after `bar.sync` is the order the barrier actually guarantees**; the id and thread count of `bar.arrive` and
   `bar.sync` must match (the thread count is the total of participating threads on both sides, a multiple of 32); a mismatch is a silent hang or an early release.
3. **`v > m ? v : m` compiles to `FSETP` + `FSEL`, not `FMNMX`** — even with `--use_fast_math` on by default. Probe: a 16-element
   running max, the ternary spelling `FSETP=15 FSEL=15 FMNMX=0`; the `fmaxf` spelling compiles to 7 three-input `FMNMX3` + 1 `FMNMX`
   (on sm_100a nvcc merges a chain of pairwise maxes into three-input maxes). The reason is that the two differ in NaN / signed-zero semantics, so the compiler cannot
   swap one for the other; to get `FMNMX`, write `fmaxf` (NaN behavior: `fmaxf` returns the non-NaN operand).
4. **When alignment is provable, ordinary byte-wise indexing is automatically merged into `LDS.128`; when it is not, it is not.** Probe: an `__align__(16)`
   array + `fs[t * 16 + c]` (c a compile-time constant) → 1 `LDS.128`; an explicit `uint4` read → 1 `LDS.128`; the same
   index with a runtime offset `fs[t * 16 + off + c]` → 16 `LDS.U8`. As soon as the base-address arithmetic mixes in a quantity for which the compiler cannot prove
   16-byte alignment, write the `uint4` read yourself (and guarantee the alignment yourself, otherwise it is an illegal address or a wrong read).

## Sources

- PTX ISA: *Parallel Synchronization and Communication Instructions: `bar` / `barrier`* (id, thread count and shared-access ordering semantics of `bar.arrive` /
  `bar.sync`); *Floating Point Instructions: `max`*;
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide: *Mathematical Functions* (NaN semantics of `fmaxf`).
- GPU-certified by the probe —
  `PROBE PASS max_diff=n/a softmax_segment_ilp/fmax_is_fmnmx fmnmx=1 fmnmx3=7`;
  `PROBE PASS max_diff=n/a softmax_segment_ilp/ternary_is_fsetp fsetp=15 fsel=15 fmnmx=0`;
  `PROBE PASS max_diff=n/a softmax_segment_ilp/arrive_transparent fmul_before=[1, 1] n=[234, 234]`;
  `PROBE PASS max_diff=n/a softmax_segment_ilp/lds_batch lds_vec=1 lds_rt_u8=16 lds_plain_aligned=1`;
  `PROBE PASS max_diff=1.178e-07 softmax_segment_ilp/numeric`.
