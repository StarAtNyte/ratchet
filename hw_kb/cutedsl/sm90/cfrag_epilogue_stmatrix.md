# cfrag_epilogue_stmatrix (WGMMA C-frag→SMEM epilogue staging, CUDA/PTX idiom)

> One file per hardware interface. Operator-agnostic; mechanism only.
> This entry is the CUDA/inline-PTX counterpart of [`flat_tv_r2s_cfrag`](flat_tv_r2s_cfrag.md) (the
> CuTe DSL idiom): it solves the same "WGMMA fp32 C accumulator → SMEM" staging, but for hand-written
> CUDA kernels, and the exit is a TMA store (the SMEM layout must agree with the TMA descriptor's
> swizzle at both ends).

## Purpose

Convert WGMMA's fp32 C accumulator (one register array per thread) down to fp16, write it into SMEM,
then write it back to GMEM via TMA store, while meeting two hardware constraints:

- **Mechanism A (register residency)**: the accumulator array stays in registers throughout; the
  epilogue produces no local-memory round trips (LDL/STL ≈ 0).
- **Mechanism B (conflict-free store)**: C-frag→SMEM uses `stmatrix.x4` (one 16B store per lane)
  into a 128B XOR-swizzle layout, with zero bank conflicts within the warp; the TMA store descriptor
  is configured with `SWIZZLE_128B` so the same layout is written back to GMEM as canonical row-major.

## When needed (trigger)

NCU symptoms (the two can appear independently and often appear together):

- **local round trips ≈ C traffic**: `l1tex__t_sectors_pipe_lsu_mem_local_op_ld/st` × 32B ≈ the bytes
  of one read+write of the output matrix, completely insensitive to mainloop changes; with
  `regs_per_thread` already maxed and the register-budget levers (`setmaxnreg` etc.) already cashed
  in, this metric **does not move at all** — meaning the cause is not an insufficient register
  budget but missing mechanism A: the epilogue has a **dynamic index** into the accumulator array
  (see gotcha 1), and ptxas demotes the whole array to local memory.
- **one-sided bank conflicts on the store side**:
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` high while `..._op_ld` ≈ 0 (the load-side
  swizzle is already right) — meaning C-frag→SMEM uses scalar 2B/4B per-cell stores or a row-major
  unswizzled layout: mechanism B is missing.

Dataflow shape: after the mainloop the accumulator has to land in SMEM and then go out via TMA store
(any "acc → SMEM → bulk copy" epilogue counts, independent of the operator).

## Minimal runnable example

Representative shape: 1 warpgroup (128 threads) handles one 64×64 fp16 epi-tile (row width exactly
128B = one swizzle atom; for a wider N, loop over several 64-column epi-tiles — the row width must stay
128B). The complete directly runnable version is the certification probe
(the probe is not included in this release).

```cuda
// ---- (register→coordinate) bijection of the WGMMA fp32 C-frag (wgmma.m64nNk16.f32 output layout) ----
//   reg 4j+q of lane l, warp w  ->  row = w*16 + (l>>2) + 8*(q>>1)
//                                   col = 8*j + 2*(l&3) + (q&1)
// With N=64 each thread holds 32 fp32 (j=0..7).

__device__ __forceinline__ void stmatrix_x4(
    uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                 :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

// ---- Mechanism A: acc only uses compile-time constant indices (#pragma unroll + constexpr bounds) ----
float acc[32];                       // from WGMMA; every index here must be static

// ---- Mechanism B: half2 pack + stmatrix.x4 -> 128B-XOR-swizzle SMEM ----
// Column pair j2 covers cols [16*j2, 16*j2+16); one stmatrix.x4 writes 4 8x8 matrices:
//   reg k holds the 8x8 whose addresses are "provided by lanes 8k..8k+7" (grouping fixed by the PTX ISA):
//   reg0=rows-lo/chunk-lo  reg1=rows-hi/chunk-lo  reg2=rows-lo/chunk-hi  reg3=rows-hi/chunk-hi
// The data mapping is the same as ldmatrix: within a matrix row = l>>2, col = 2*(l&3) (two adjacent b16
// per reg, low half-word = even column) — matching the WGMMA C-frag bijection above bit for bit, so no
// cross-thread shuffle is needed.
const uint32_t base = smem32(tile);           // tile: 1024B-aligned SMEM fp16[64][64]
const int rr = l & 7, hi = (l >> 3) & 1;
const int row_a = w * 16 + hi * 8 + rr;       // the SMEM row this lane addresses
#pragma unroll
for (int j2 = 0; j2 < 4; j2++) {
    uint32_t r[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {             // down-conversion done in registers (gotcha 5)
        __half2 h2 = __floats2half2_rn(acc[8*j2 + 2*k], acc[8*j2 + 2*k + 1]);
        r[k] = *reinterpret_cast<uint32_t*>(&h2);
    }
    int chunk = 2 * j2 + (l >> 4);            // 16B chunk column index
    // 128B XOR swizzle: chunk ^= row%8. Same bijection as TMA SWIZZLE_128B (gotcha 2).
    uint32_t addr = base + row_a * 128u + ((uint32_t)(chunk ^ (row_a & 7)) << 4);
    stmatrix_x4(addr, r[0], r[1], r[2], r[3]);
}
__syncthreads();

// ---- Exit: generic-proxy write -> async-proxy read needs a proxy fence first (gotcha 3) ----
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
if (tid == 0) {
    // tC descriptor: FLOAT16, 2D, box=(64,64), CU_TENSOR_MAP_SWIZZLE_128B
    tma_store_2d(&tC, tile, gmem_col, gmem_row);
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");   // must wait before reusing the buffer (gotcha 6)
```

Why it is conflict-free: each 8-lane group of `stmatrix.x4` addresses 8 different rows of the same
chunk column, and the XOR swizzle scatters them over 8 **different 16B bank groups** (`chunk ^ (row%8)`
is a bijection over row=0..7); 16B per lane, 512B per warp, completed in 4 conflict-free wavefronts —
after landing it, verify `..._op_st` ≈ 0 with NCU. This is exactly the idiom of the cutlass sm90 fp16
epilogue (STSM + swizzle atom).

## Gotchas

- **(1) One dynamic index vetoes everything (mechanism A)**: a **single** `acc[runtime value]`
  (typical: the outer epi-tile loop is not `#pragma unroll`ed and `epi_idx` in
  `acc[epi_idx*8 + ...]` is the loop variable; or a runtime tile number is passed as an argument into
  a helper and then used as an index) makes ptxas demote **the whole array** to local memory — the
  compiler does no partial residency. The symptom is the trigger section's "local traffic ≈ C size and
  insensitive to the register budget". Fix: `#pragma unroll` the epi loop with a constexpr trip count,
  make helpers `__forceinline__` with the tile number a compile-time constant (or template parameter).
  Check: `cuobjdump -sass out.so | grep -c 'LDL\|STL'` = 0; NCU local sectors ≈ 0.
- **(2) The swizzle must be the same bijection at both ends**: the stmatrix write side's
  `chunk ^ (row%8)` and the TMA store descriptor's `CU_TENSOR_MAP_SWIZZLE_128B` are the same formula
  (half-word index `r*64 + ((c/8)^(r%8))*8 + (c%8)`, row pitch 128B). Swizzle at one end and
  `SWIZZLE_NONE` at the other → GMEM data silently misplaced (no error). Conversely, row-major SMEM +
  descriptor `SWIZZLE_NONE` is functionally correct too, but stmatrix's 8-lane groups then conflict on
  the same bank — which is exactly the trigger symptom.
- **(3) proxy fence**: stmatrix is a generic-proxy store, TMA store is an async-proxy read of SMEM;
  `fence.proxy.async.shared::cta` is required in between; when several warps write the same buffer,
  a `__syncthreads()` is also needed before the fence. Missing it → occasional stale reads
  (nondeterministic error).
- **(4) Alignment and atom width**: the SMEM tile base is 1024B-aligned (swizzle period = 8 rows ×
  128B); the epi-tile row width must be exactly 128B (fp16 × 64 columns) — this is the atom width of
  SWIZZLE_128B and also the range in which the certified formula applies. Wider outputs loop over
  64-column slices.
- **(5) Down-convert in registers**: pack with `__floats2half2_rn`, then store. Landing fp32 in SMEM
  first and converting a second time doubles the SMEM traffic and adds a read-modify-write round.
- **(6) bulk_group lifetime**: after `cp.async.bulk.commit_group`, the SMEM buffer must not be
  overwritten until `cp.async.bulk.wait_group N` releases it; a single buffer uses `wait_group 0`, a
  K-buffer pipeline with reuse uses `wait_group K-1`.
- **(7) Don't trust intuition for stmatrix's reg↔matrix grouping**: reg k corresponds to the matrix
  whose addresses are "provided by lanes 8k..8k+7", and the data mapping (within a matrix
  `row=l>>2, col=2*(l&3)`, low half-word of b32 = even column) is the same as ldmatrix. Getting either
  one backwards silently misplaces values — verify with a GPU round-trip probe (known ramp →
  stmatrix → read back/TMA store → compare against numpy; it holds only if `max_diff==0`).

## Sources

- Official docs: PTX ISA `stmatrix.sync.aligned.m8n8.x4.shared.b16` (sm_90; data mapping the same as
  `ldmatrix`, address grouping lanes 8k..8k+7 ↔ reg k); CUDA C++ Programming Guide "Local Memory"
  (arrays indexed with non-compile-time-constant indices are placed in local memory).
- 128B swizzle half-word formula: GPU-certified by a separate probe
  (stage 1, PASS); this entry reuses the same formula on the store side.
- Certification probe (ramp C-frag → stmatrix →
  raw SMEM dump compared against the swizzle formula; TMA store → GMEM compared against canonical
  row-major).
