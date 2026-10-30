# tmem_wide_readback_stsm_mn_box (accumulator readback → bf16 → a warp-private C staging box **contiguous along TMEM lanes**: switch to `tcgen05.ld.16x256b` + `stmatrix.trans` instead of element-wise `st.shared.u16`)

> **Certification source: re-certified on a GPU probe in CUDA mode** (the two `mmajor` settings of the probe, which is not included in this release; verdicts under Sources).
> The mechanism is the same as the same-named entry in the CuTe DSL tree; the readings were re-taken on the CUDA spelling.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the shape with a shared staging face + one warp issuing the store see
> [`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md); this entry is **a small box per warp**, written, fenced and
> TMA-stored by the warp itself, with no cross-warp rendezvous.

## Purpose

The accumulator in TMEM is `lane × column` (lane = the MMA's M). When the output C has **M as its contiguous dimension** in gmem (swap-AB, or the MMA's M landing on the output's
contiguous dimension), the staging box must be contiguous along lanes: one box row = one TMEM column, holding the 16-bit values of 32 adjacent lanes (64 B). Then the **shape** of the register fragment
decides the width of the smem write instruction:

| Readback | What each thread gets | Widest spelling usable on a lane-contiguous box | Per warp per 1 KiB box (32 lanes × 16 columns) |
|---|---|---|---|
| `tcgen05.ld.32x32b.x16` | thread = one lane, 16 columns (64 B apart from each other in the box) | only element-wise `st.shared.u16`: a 16 B segment is the values of 8 adjacent lanes, spread over 8 threads | 16 `st.shared.u16` / thread |
| `tcgen05.ld.16x256b.x2` (twice: lanes 0–15, 16–31) | m16n8 C fragment shape: two lanes 8 apart × two adjacent columns × two column groups | `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`: one instruction writes 4 8×8 matrices, each matrix row = 8 adjacent lanes = 16 B | 2 `stmatrix.x4.trans` / thread |

So "replacing the epilogue's element-wise writes with matrix atoms" **requires changing the TMEM readback instruction first**, not the store instruction.

## When needed (trigger)

- You have already decided to remove the element-wise writes (your own same-job A/B shows the epilogue write side is on the critical path), the readback is `.32x32b`, and the box's contiguous dimension is the TMEM lane;
- you want 16 B smem writes but find no 8 lane-adjacent values in registers to assemble them from — **change the readback instruction**, do not do a shuffle transpose.

**This entry certifies only the spelling, numerics and instruction family, not performance.** Do a same-job A/B on your own kernel before switching.

Does not apply: when the box's contiguous dimension is the TMEM **column** (N-major C) — each thread's contiguous columns from `.32x32b` already assemble into 16 B plain writes; see
[`tcgen05_epilogue_t2r_stsm_tma_store`](tcgen05_epilogue_t2r_stsm_tma_store.md).

## Minimal runnable example

4 warps, warp `w` reads lane segment `32*w..+31`; each warp has two private 1 KiB boxes (double buffering), box = 16 rows (columns) × 64 B (32 lanes),
SW64 image; C^T is stored as `[N][M]` (M contiguous), tensor map box `{32, 16}`, `CU_TENSOR_MAP_SWIZZLE_64B`. Same spelling as the certification probe:

```cpp
__device__ inline uint32_t sw64(uint32_t off) { return off ^ (((off >> 7) & 3u) << 4); }   // 16 B block number ^= address bits [7,9)

const uint32_t box = box0 + (blk & 1) * 1024;                 // box0: this warp's 1024 B-aligned base
for (int h = 0; h < 2; ++h) {                                 // the two halves, lanes 0-15 and 16-31
  uint32_t v[8];
  tm_ld16x256b_x2(tbase + ((uint32_t)(32 * w + 16 * h) << 16) + 16 * blk, v);   // 16 lanes x 16 columns
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  // matrix i = lb + 2*cb: rows = lanes 16h+8lb..+7, columns = 16blk+8cb..+7; thread t holds (row t/4, columns 2(t%4), 2(t%4)+1)
  uint32_t r[4];
  for (int i = 0; i < 4; ++i) {
    const int lb = i & 1, cb = i >> 1;
    r[i] = pack_bf16x2(__uint_as_float(v[2 * lb + 4 * cb]), __uint_as_float(v[2 * lb + 4 * cb + 1]));
  }
  // .trans: storage row j of matrix i = TMEM column 8cb+j = box row 8cb+j; within the row, 8 adjacent lanes starting at 16h+8lb
  const int i = lane / 8, j = lane % 8, lb = i & 1, cb = i >> 1;
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(box + sw64((8 * cb + j) * 64 + 32 * h + 16 * lb)), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
__syncwarp();
if (lane == 0) {
  tma_store_2d(&mct, box, m0 + 32 * w, 16 * blk);            // coords = {m (innermost dimension), n}
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");   // the box from two blocks ago has been read
}
__syncwarp();
// after the loop, lane 0: cp.async.bulk.wait_group 0
```

Element-wise baseline (same box, same swizzle): `tm_ld32x32b_x16(tbase + ((32*w) << 16) + 16*blk, v)`, then 16
`st.shared.u16 [box + sw64(j*64 + lane*2)]`.

## Gotchas

1. **A `.32x32b` fragment has no wide spelling on a lane-contiguous box**: thread = one lane, and a 16 B segment of the box is 8 adjacent lanes — in 8 different threads.
   To use the wide spelling, first switch the readback to `.16x256b`.
2. **Registers and addresses of `stmatrix.trans`**: register `r_i` holds the two adjacent elements of this thread's row in matrix i (`pack_bf16x2(even column, odd column)`, even column in the low half);
   the 8 **storage row** addresses of matrix i are supplied by threads `8i..8i+7`. `.trans` stores the matrix transposed: storage row j = original column j = box row `8cb+j`.
3. **Register order of `.16x256b`**: register i lands at lane `+t/4 + 8*((i>>1)&1)`, column `+2(t%4) + (i&1) + 8*(i>>2)` (x2) — so `v[2lb+4cb]`,
   `v[2lb+4cb+1]` are exactly this thread's pair in matrix (lb, cb). This map is pinned separately by the probe contract `C3-ld16x256b-map`.
4. **With swizzle, the box and the tensor map use the same mode**: 64 B per box row ⇒ `CU_TENSOR_MAP_SWIZZLE_64B`, smem image = 16 B block number XOR address bits [7,9);
   the box base is 1024 B aligned (so the swizzle phase at the start of each 1 KiB block matches what TMA sees).
5. **Why swizzle**: in the transposed form the 8 storage rows of one 8×8 matrix are 64 B apart, and without swizzle they land, mod 128 B, in only 2 16 B slots;
   SW64 spreads them out. Both boxes are numerically right; check conflicts with NCU's `bankconf_st` (readings under Sources).
6. **Bulk-group accounting is per warp**: each warp commits its own, does `wait_group.read 1` before writing a box (double buffering), and `wait_group 0` before exit.
7. **Accept by instruction family**: `STSM` appears in SASS and `STS.U16` goes to zero (the probe's SASS census readings are under Sources); other setup writes leave a few plain `STS`,
   so do not write the contract as "all STS = 0".

## Sources

- PTX ISA: `tcgen05.ld` (`.16x256b` / `.32x32b`), `stmatrix` (`.trans`, `.x4`), `cvt.rn.bf16x2.f32`, `fence.proxy.async`,
  `cp.async.bulk.tensor` (`SWIZZLE_64B`), `cp.async.bulk.wait_group.read` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUDA driver `cuTensorMapEncodeTiled`; CUTLASS `cute::SM90_U16x8_STSM_T`, `cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b2x` (same PTX spelling).
- GPU-certification probe: settings `mmajor stsm` / `mmajor sts16` / `map16x256b`, verdict
  `PROBE PASS max_diff=0.0`: `C2-mmajor-ld16x256b-stsm-trans`, `C2-mmajor-ld32x32b-sts16` (two launches, C^T bitwise equal to the payload),
  `C3-ld16x256b-map`. SASS: in the stmatrix version `STSM` corresponds one to one with `LDTM`, `STS.U16` = 0, plain `STS` 3 (setup); the element-wise version has 16 `STS.U16` per block per thread.
  NCU: stmatrix version `bankconf_st` 0, smem write instructions 20 / wavefronts 532; element-wise version `bankconf_st` 250, write instructions 1044 / wavefronts 1294.
