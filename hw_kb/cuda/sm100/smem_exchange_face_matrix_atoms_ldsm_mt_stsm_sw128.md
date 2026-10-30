# smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128 (matrix-atom assembly for 16-bit exchange faces: an SW128 face → read with `ldmatrix.x4.trans` into a fragment isomorphic to `tcgen05.ld.16x256b`; C-fragment-shaped registers → `stmatrix.x4` → UMMA B operand face)

> **Certification source: re-certified on a GPU probe in CUDA mode** (cell `ldsmT` of an smem-face probe (site A),
> cell `stsm` of a UMMA-operand probe (site B); the probes are not included in this release; verdicts under Sources). The mechanism is the same as the same-named entry in the CuTe DSL tree; the spelling there that derives the bijection from a tiled copy is not carried over;
> here the addresses and register order are given directly.
>
> One file per hardware interface. Operator-agnostic: mechanism only. For the output side (accumulator → `stmatrix` → staging face → TMA store) see
> [`tmem_wide_readback_stsm_mn_box`](tmem_wide_readback_stsm_mn_box.md).

## Purpose

The same family of 8×8×16b matrix atoms has two more sites that often degrade into element-wise `ld.shared.u16` / `st.shared.u16`:

- **Site A**: a K-major SW128 face (R rows × 64 bf16 columns, columns contiguous, the box shape TMA lands) must **meet a TMEM accumulator element by element**, where the accumulator's
  lanes are the face's columns and the TMEM columns are the face's rows — the face must be read **transposed** into registers **isomorphic** to the `tcgen05.ld.16x256b` fragment shape.
- **Site B**: registers in m16n8 C-fragment shape (the C fragment of a warp-level `mma.sync`, or a fragment read back with `.16x256b`) are narrowed to bf16 and written into a
  K-major SW128 face **that UMMA reads through a descriptor as its B operand**.

The criterion common to both sites: **the fragment shape decides the atom**. A `.32x32b` fragment (thread = lane, one thread several contiguous columns) has no matrix-atom partner; a `.16x256b` fragment is exactly
the m16n8 C fragment shape (each thread two rows 8 apart, two adjacent columns, groups of 8 columns), bitwise isomorphic to the fragment shape scattered by `ldmatrix.x4(.trans)` / `stmatrix.x4`.

## When needed (trigger)

- A face meets an accumulator / C fragment element by element, but the implementation reads/writes the face by element index (in SASS the loop body's `LDS.U16` / `STS.U16` count ≈ elements per thread);
- you wrote every readback as `.32x32b` and then found that "transposed read / transposed write can only be element-wise" — the fragment shape was chosen wrong, not the face.

**This entry certifies the bijection and the instruction family; it does not guarantee that NCU's bank-conflict axis drops with it**: the main conflicts may not be at these two sites; attribute `bankconf_*` separately.

## Minimal runnable example

### Site A: SW128 face (64 rows r × 64 columns c, c contiguous) → `ldmatrix.x4.trans` = the `.16x256b` fragment of the transposed matrix

```cpp
__device__ inline uint32_t sw128(uint32_t off) { return off ^ (((off >> 7) & 7u) << 4); }
// warp w takes lane segment (= the face's columns) c0 = 16w..+15 and TMEM column segment (= the face's rows) 0..15; matrix i = lb + 2*cb
const int c0 = 16 * w, i = lane / 8, j = lane % 8, lb = i & 1, cb = i >> 1;
uint32_t r[4];
asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
             : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
             : "r"(face + sw128((8 * cb + j) * 128 + (c0 + 8 * lb) * 2)));   // thread 8i+j supplies the j-th storage row of matrix i
// result: half word hh of r[q] (low half = 0) = face[r = 8*cb + 2*(t%4) + hh][c = c0 + 8*lb + t/4], q = lb + 2*cb
//      = the pair read, after transposing the face (lane = c, column = r), by the thread-value map of tcgen05.ld.16x256b, packed into one word
```

So the accumulator register pair `(v[2lb+4cb], v[2lb+4cb+1])` read back with `.16x256b` and the two half words of `r[lb+2cb]` refer to the same (lane, column), and meet element by element directly.

### Site B: C-fragment-shaped registers → `stmatrix.x4` (not transposed) → K-major SW128 B face → UMMA

B face of 64 rows (n) × 64 columns (k, contiguous); warp w holds C-fragment-shaped registers for n = 16w..16w+15 (thread t: rows `t/4` and `t/4+8`, a pair starting at column `8kt + 2(t%4)`):

```cpp
for (int jp = 0; jp < 4; ++jp) {                      // two k8 blocks at a time
  uint32_t r[4];
  for (int i = 0; i < 4; ++i) {                       // matrix i: nb = i&1 (row +8), kt = 2jp + (i>>1)
    const int nb = i & 1, kt = 2 * jp + (i >> 1);
    const int n = 16 * w + 8 * nb + lane / 4, k = 8 * kt + 2 * (lane % 4);
    r[i] = pack_bf16x2(B(n, k), B(n, k + 1));         // even k goes into the low half
  }
  const int i = lane / 8, j = lane % 8, nb = i & 1, kt = 2 * jp + (i >> 1);
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
               :: "r"(sB + sw128((16 * w + 8 * nb + j) * 128 + kt * 16)), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // then sync, then issue the MMA (B descriptor = plain SW128 K-major)
```

## Gotchas

1. **A `.32x32b` fragment has no matrix-atom partner**: each (thread, column) has only one 16-bit element, so landing it transposed in a face leaves only the element-wise path. To meet a face through matrix atoms, read back with `.16x256b`.
2. **`.trans` goes on the side where "what is contiguous in the face is the TMEM lane"**: site A's face has contiguous columns with column = lane, so the read uses the transposing member; site B's face has contiguous k with register row = n,
   so the write uses the non-transposing member. Picking the wrong member gives entirely wrong numbers without an error.
3. **Thread `8i + j` supplies the address of the j-th storage row of matrix i**, 16 B aligned; the swizzle image is computed by the **storage row** number (the face's row), not the register's row.
4. **After writing the face, `fence.proxy.async.shared::cta`, then sync**, before UMMA can read it through the descriptor (same as [`umma_smem_operand_sts128_publish`](umma_smem_operand_sts128_publish.md)).
5. **Accept with an asymmetric payload**: the probe's face values are a function of the coordinates (different for each (r, c)), so wrong transposes, swapped blocks and reversed half words all show up; a symmetric or constant payload cannot reveal them.

## Sources

- PTX ISA: `ldmatrix` / `stmatrix` (`.x4`, `.trans`, the rule for threads supplying row addresses), `tcgen05.ld` (lane–column assignment of `.16x256b`),
  the m16n8 C fragment layout of `mma.sync`, `fence.proxy.async` — <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public infra API: CUTLASS `cute::SM75_U16x8_LDSM_T`, `cute::SM90_U32x4_STSM_N` (same PTX spelling).
- GPU-certification probes (`ldsmT`, `stsm`), verdict
  `PROBE PASS max_diff=0.0`: site A `E3-ldsm-trans-sw128-is-16x256b-map` (4 warps × 32 threads × 4 registers × 2 half words, 0 mismatches against the
  `.16x256b` map; one `LDSM` in SASS), site B `D2-operand-stsm-sw128` (4 `STSM` write the B face, UMMA reads it through the descriptor, bitwise exact over two launches).
