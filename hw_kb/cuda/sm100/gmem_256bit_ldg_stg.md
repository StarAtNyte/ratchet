# gmem_256bit_ldg_stg (256-bit global memory access on sm100: how `LDG.*.256` / `STG.*.256` actually land in SASS)

> **Certification source: CUDA-mode GPU probe (not included in this release),
> `PROBE PASS max_diff=0.0` (`ldg-width-numeric`, `ldg-sass-256`); the 32 B aligned vector type cell is
> a fact line of the same run.** What is certified is "whether it lowers to `.256`" and the numerics;
> the timing is only a reading on a streaming toy, not a promise of gain.
>
> One file per hardware interface. Operator-agnostic: mechanism only.

## Purpose

The widest single-thread global memory access on sm100 is **32 B** (PTX `ld.global{.nc}.v8.b32` / `st.global.v8.b32`, or `.v4.b64`; per the PTX ISA
from `sm_100` on, `.global` only). For the same number of bytes, half the instruction count of 128-bit: one warp instruction covers 32 × 32 B = 1024 B.

## When needed (trigger)

- A streaming kernel dominated by `LDG/STG.E.128`, where each lane already holds 32 contiguous B (two adjacent 128-bit accesses).
- You want to confirm whether your 32 B access is one `.256` or two `.128` in SASS.
- **Do the accounting first**: when using it requires changing the lanes' element ownership to "32 contiguous B per lane", the ownership change itself may cost more than the width saves — decide by same-job whole-kernel timing.

## Preconditions

1. Each lane owns **contiguous** 32 B per instruction (two non-adjacent 16 B pieces can only give `.128`).
2. Address 32 B aligned: the real runtime alignment is yours to guarantee (allocator base, row pitch a multiple of 32 B), assert on the host; on the inline PTX path the compiler checks nothing.
3. On the register side, 8 32-bit (or 4 64-bit) values.

## Minimal runnable example

Probe readings (one warp per 1024 B row, 32 contiguous B per lane, one read and one write, SASS read per kernel):

| Spelling | Read-side SASS | Write-side SASS |
|---|---|---|
| two `uint4` accesses (128-bit control) | `LDG.E.128` ×2 | `STG.E.128` ×2 |
| inline PTX `ld.global.nc.v8.b32` / `st.global.v8.b32` | `LDG.E.ENL2.256.CONSTANT` | `STG.E.ENL2.256` |
| inline PTX `ld.global.nc.L1::no_allocate.v8.b32` / `st.global.L1::evict_first.v8.b32` | `LDG.E.NA.ENL2.256.CONSTANT` | `STG.E.EF.ENL2.256` |
| plain C++ dereference of `struct alignas(32) { uint4 a, b; }` | `LDG.E.ENL2.256` | `STG.E.ENL2.256` |
| plain C++ dereference of `ulonglong4_32a` (the 32 B aligned vector type of CUDA 13) | `LDG.E.ENL2.256` | `STG.E.ENL2.256` |

```cpp
// read: .nc (read-only data path, SASS carries .CONSTANT); write: plain st.global
__device__ __forceinline__ void ld256(const void* p, uint32_t (&w)[8]) {
  asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]), "=r"(w[6]), "=r"(w[7])
               : "l"(p));
}
__device__ __forceinline__ void st256(void* p, const uint32_t (&w)[8]) {
  asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
               :: "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7])
               : "memory");
}
// lane l holds bytes [32 l, 32 l + 32):
const size_t off = (size_t)row * 1024 + (size_t)(threadIdx.x % 32) * 32;
uint32_t w[8]; ld256(x + off, w); /* ... element-wise ... */ st256(y + off, w);

// asm-free equivalent (when the compiler can prove 32 B alignment):
struct alignas(32) V32 { uint4 a, b; };
V32 v = *reinterpret_cast<const V32*>(x + off);        // LDG.E.ENL2.256 (no .CONSTANT)
*reinterpret_cast<V32*>(y + off) = v;                    // STG.E.ENL2.256
```

## Gotchas

1. **Numerics do not reveal the width**: in the probe all five spellings are numerically right (including out-of-range sentinel rows left unwritten); the width can only be seen in SASS: 1 LDG / STG with `.256` per issue point,
   0 `.128`.
2. **For the read-only path (`.CONSTANT` / `.nc`) write inline PTX**: in the probe the `.256` of a plain C++ dereference does not carry `.CONSTANT`; `__ldg` has no 32 B overload.
3. **Cache hints go straight into the mnemonic**: `.L1::no_allocate` (read), `.L1::evict_first` (write) lower to `.256` with `.NA` / `.EF`.
4. **`.v8` goes with `.b32`, `.v4` with `.b64`**; bf16 / fp16 data is packed into 32-bit words.
5. **Alignment is a promise, not a check**: `alignas(32)` / `_32a` types tell the compiler the address is 32 B aligned; sub-pointers produced by slicing are not necessarily aligned, and this probe did not test the runtime behavior of misalignment.
6. **Readings on a streaming toy** (1 GiB read + 1 GiB write, single reading): 128-bit control 0.3343 ms, PTX `.256` 0.3150 ms, `alignas(32)` struct 0.3163 ms
   — with the same lane ownership `.256` is about 6% faster. For a kernel whose lane ownership has to change for this, the reading does not apply; time your own kernel in the same job.

## Sources

- PTX ISA: vector forms of `ld` / `st` (`.v8.b32` / `.v4.b64` 256-bit, `.global` only, from `sm_100`), `ld.global.nc`, `.L1::evict_first` /
  `.L1::no_allocate`: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide / the 32 B aligned vector types of the CUDA 13 runtime headers (`ulonglong4_32a` etc.) and `alignas`:
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA Binary Utilities (`cuobjdump -sass`): <https://docs.nvidia.com/cuda/cuda-binary-utilities/>
- GPU-certification probe — `PROBE PASS max_diff=0.0` (numeric contract
  `ldg-width-numeric`, SASS contract `ldg-sass-256`); the 32 B aligned vector type cell (same run, fact line: numerically exact,
  `LDG.E.ENL2.256` / `STG.E.ENL2.256`).
