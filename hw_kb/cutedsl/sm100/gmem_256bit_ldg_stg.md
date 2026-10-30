# gmem_256bit_ldg_stg (256-bit global memory access on sm100: how `LDG.*.256` / `STG.*.256` really land in SASS)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe, not included in this release; `PROBE PASS max_diff=0.0`, round 2 of this certification; nine spellings under cutlass 4.4.2, SASS inspected one by one:
> **only the two inline-PTX spellings land `.256` on both the read and the write side**; all five DSL `CopyUniversalOp` 256-bit spellings widen only the write side, and the read side is split into 16 `LDG.E.U16`;
> the 128-bit control lands `LDG/STG.E.128`; the 256-bit atom with `assumed_align=16` is rejected at compile time. Every spelling that runs is numerically bit-exact, and the out-of-bounds sentinel rows are not written)
>
> **What is certified**: only "whether it can land as `.256`" and the numerics; **no performance gain whatsoever is certified**. A straight-line toy (`y = 2x`), bf16, with one warp per 1024 B row, 32 contiguous B per lane,
> one read and one write per lane.
>
> **Contrary reading (read this before deciding whether to switch)**: in a whole-kernel same-job timing on a bf16 row-reduction straight-line kernel, the true `.256` version
> was **3.4% slower than the shipped 128-bit version** (switching only the read side to `.256`: 3.0% slower); against the version "same lane ownership, 128-bit PTX" it is 0.957× --
> that is, the lane ownership that has to change for `.256` is itself 8% more expensive, and what the width saves does not make it up. See gotcha 9.

## Purpose

The widest single-thread global memory access on sm100 is **32 B** (PTX `ld.global{.nc}.v8.b32` / `st.global.v8.b32`, or `.v4.b64`;
per the PTX ISA documentation this form exists from `sm_100` on and only for `.global`). For the same number of bytes, **the instruction count is half that of 128-bit**:
one instruction of one warp covers a contiguous range of 32 × 32 B = 1024 B.

## When needed (trigger)

- You want to change `num_bits_per_copy` to 256, but at compile time you get
  `S ptr alignment (128 bits) does not meet requirement (256 bits)` (or the same text for the D side).
- After changing to 256 **only one side** widens: `STG.*.256` appears, but `LDG` does not widen, and even turns into a string of `LDG.E.U16`
  (on NCU: `st_inst` halves while `ld_inst` does not drop but rises).
- You want to try the width lever on a straight-line kernel dominated by `LDG/STG.E.128`. **Read the contrary reading above first**: width is not free; decide only after same-job whole-kernel timing.

## Preconditions (all three are required)

1. **Each lane owns 32 contiguous B per instruction**. If a lane's 32 B consist of two non-adjacent 16 B pieces, only 128-bit can be issued.
   Switch to lane l holding 32 contiguous B (for bf16, `[16l, 16l+16)`).
2. **Addresses are 32 B aligned**. You guarantee the real run-time alignment (allocator base address, row pitch a multiple of 32 B), asserted on the host side.
   On the DSL write-side atom path the compiler must also know it: declare `assumed_align=32` on the **pointer type**, and preserve it on sub-view offsets with
   `cute.assume(off, divby=<number of elements in 32 B>)`.
3. **The register side** is a 32 B contiguous rmem fragment (for bf16, `(16,)`).

## Minimal runnable example

**Certified conclusion** (cutlass 4.4.2, nine spellings of the same probe with SASS inspected one by one, counted with `nvdisasm`):

| Spelling | Read-side SASS | Write-side SASS |
|---|---|---|
| `CopyUniversalOp` bf16 atom, `num_bits_per_copy=128`, pointer `assumed_align=16` (control) | `LDG.E.128` | `STG.E.128` |
| Same with `=256`, pointer `assumed_align=16` | rejected at compile time | rejected at compile time |
| Same with `=256`, pointer `assumed_align=32` | **16 × `LDG.E.U16`** | `STG.E.ENL2.256` |
| Same, read atom with `invariant=True`, write atom with `l1c_evict_priority=EVICT_FIRST` | **16 × `LDG.E.U16.CONSTANT`** | `STG.E.EF.ENL2.256` |
| Int32 atom (recast view, with or without `invariant=True`; or an Int32 pointer + an `(M, H/2)` tensor), `=256` | **16 × `LDG.E.U16`** (with `.CONSTANT` under `invariant`) | `STG.E.ENL2.256` |
| inline PTX `ld.global.nc.v8.b32` / `st.global.v8.b32` | `LDG.E.ENL2.256.CONSTANT` ✓ | `STG.E.ENL2.256` ✓ |
| inline PTX `ld.global.nc.L1::no_allocate.v8.b32` / `st.global.L1::evict_first.v8.b32` | `LDG.E.NA.ENL2.256.CONSTANT` ✓ | `STG.E.EF.ENL2.256` ✓ |

So: **use inline PTX on the read side**; on the write side both the DSL atom and inline PTX work (below, both sides use PTX).
Every spelling is numerically correct -- **only the width is wrong, the numerics cannot show it, and you must look at the SASS**.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
from cuda.bindings import driver as cuda_drv

H, VEC = 512, 16        # a row of 512 bf16 = 1024 B = one 256-bit instruction of one warp; 16 per lane = 32 B
WARPS = 4               # warps per CTA, arbitrary; one row per warp
BF16, F32, I32, I64 = cutlass.BFloat16, cutlass.Float32, cutlass.Int32, cutlass.Int64

@dsl_user_op
def ld256(addr, *, loc=None, ip=None):            # addr: Int64 byte address, must be 32 B aligned
    s = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8), [I64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(I32(llvm.extractvalue(T.i32(), s, [i], loc=loc, ip=ip)) for i in range(8))

@dsl_user_op
def st256(addr, w, *, loc=None, ip=None):         # w: 8 Int32
    llvm.inline_asm(
        None, [I64(addr).ir_value(loc=loc, ip=ip)] + [I32(v).ir_value(loc=loc, ip=ip) for v in w],
        "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};", "l,r,r,r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@cute.kernel
def k(mX: cute.Tensor, mY: cute.Tensor, M: I32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane, row = tidx % 32, bidx * WARPS + tidx // 32
    if row < M:
        rX = cute.make_rmem_tensor((VEC,), BF16)
        rY = cute.make_rmem_tensor((VEC,), BF16)
        off = I64(row * H + lane * VEC) * I64(2)          # byte offset; lane l holds [16l, 16l+16)
        w = ld256(mX.iterator.toint() + off)              # 1 LDG.E.ENL2.256.CONSTANT
        rX32 = cute.recast_tensor(rX, I32)                # 8 words <-> 16 bf16
        for i in cutlass.range_constexpr(8):              # a Python range cannot be used (see gotcha 7)
            rX32[i] = w[i]
        rY.store((rX.load().to(F32) * F32(2.0)).to(BF16)) # placeholder for element-wise computation
        rY32 = cute.recast_tensor(rY, I32)
        ys = []
        for i in cutlass.range_constexpr(8):
            ys.append(rY32[i])
        st256(mY.iterator.toint() + off, ys)              # 1 STG.E.ENL2.256

@cute.jit
def launch(pX: cute.Pointer, pY: cute.Pointer, M: I32, stream: cuda_drv.CUstream):
    mX = cute.make_tensor(pX, cute.make_layout((M, H), stride=(H, 1)))
    mY = cute.make_tensor(pY, cute.make_layout((M, H), stride=(H, 1)))
    k(mX, mY, M).launch(grid=[(M + WARPS - 1) // WARPS, 1, 1], block=[32 * WARPS, 1, 1],
                        stream=stream)

# host: the real addresses must be 32 B aligned (the PTX path bypasses the compiler's alignment check)
assert x.data_ptr() % 32 == 0 and y.data_ptr() % 32 == 0
pX = make_ptr(BF16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
pY = make_ptr(BF16, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
```

Cache hints go straight into the mnemonic: `ld.global.nc.L1::no_allocate.v8.b32`, `st.global.L1::evict_first.v8.b32`
(certified to land as `.256` instructions with the `.NA` / `.EF` modifiers).

### If you only want the write side wider: a DSL atom is enough

```python
def lane_view(mT, row, lane):
    off = cute.assume(row * H + lane * VEC, divby=VEC)    # preserve the 32 B alignment
    return cute.make_tensor(mT.iterator + off, cute.make_layout((VEC,), stride=(1,)))

st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=256)
cute.copy(st, rY, lane_view(mY, row, lane))                # STG.E.ENL2.256 (the pointer needs assumed_align=32)
```
The same atom used on the read side raises no error and is numerically correct, but with this probe's spelling the SASS is 16 `LDG.E.U16`.

### Acceptance (SASS; the numerics cannot see it, so you must look at the footprint)

`CUTE_DSL_KEEP_CUBIN=1 CUTE_DSL_KEEP_PTX=1 CUTE_DSL_DUMP_DIR=<dir>` (a separate
`CUTE_DSL_CACHE_DIR` per spelling) → `nvdisasm -c <cubin>`, count the full dotted mnemonics of `LDG` / `STG`:
**1 LDG / STG with `.256` per issue site, 0 of `.128` and `.U16`**. On the PTX side this corresponds to `ld.global….v8.` / `st.global….v8.`
(or `.v4.b64`); in this probe the spellings whose read side degrades are already `ld.global{.nc}.b16` in the PTX, showing that the split happens before ptxas.

## Gotchas

1. **`assumed_align` not written as 32**: under `make_ptr(..., assumed_align=16)` the 256-bit atom is rejected at compile time:
   `S ptr alignment (128 bits) does not meet requirement (256 bits)`. The fix is to declare 32 on the **pointer**, not to change the atom.
2. **Sub-view offsets need `divby`**: the alignment of `iterator + off` depends on the base alignment and on the known divisibility of `off`. When the offset contains run-time quantities
   (row index, lane), the probe always uses `cute.assume(off, divby=VEC)`; what happens without it was not tested separately in this probe.
3. **Alignment is a promise, not a check**: `assumed_align=32` only tells the compiler; on the inline-PTX path the compiler checks nothing at all.
   Assert the real address's 32 B alignment on the host first; the base address of a sub-tensor produced by slicing (`x[k:]`) is not necessarily aligned. What misalignment does at run time was not tested in this probe.
4. **The lanes' element ownership decides whether you can go wide**: an ownership where a lane's 32 B are not contiguous cannot become 256; it must change to 32 contiguous B per lane.
   Changing ownership changes the data distribution within the warp: quantities tied to the lane position (e.g. a vector broadcast by column) must also be fetched according to the new ownership. **Changing ownership itself has a cost**, see gotcha 9.
5. **The read side of a DSL 256-bit atom degrades to element-wise `LDG.E.U16`**: in all five DSL spellings of this probe (bf16 / Int32 atom, recast view,
   Int32 pointer, with or without `invariant=True`) the read side does this, and the write side is always `.256`. This is the cause of "`st_inst` halves, `ld_inst` unchanged or even higher".
   Use inline PTX on the read side; verify the SASS of the read and write sides separately.
6. **`.v8` pairs with `.b32`, `.v4` with `.b64`**: the element width of the 256-bit PTX vector forms is 32 / 64 bit; bf16 data is packed via an Int32 recast
   (in the probe, the DSL write side with `invariant` + `EVICT_FIRST` generates exactly `st.global.L1::evict_first.v4.b64`, which also lands as `.256`).
7. **In the kernel body, the loop that subscripts the tuple returned by `ld256` must be `cutlass.range_constexpr(8)`**: a Python `range(8)` is treated by the tracer
   as a dynamic loop, the subscript becomes a run-time value, and compilation reports `'<class '…ArithValue'>' object cannot be interpreted as an integer`
   (both PTX spellings died here in round 1 of the certification).
8. **`cute.nvgpu.CopyG2ROp` / `CopyR2GOp` do not exist in cutlass 4.4.2**: `AttributeError: module 'cutlass.cute.nvgpu' has no attribute 'CopyG2ROp'`;
   do not look for them according to old documentation.
9. **Width is not free: `.256` lost in whole-kernel same-job timing** (a performance reading, not a certification probe). On a bf16 row-reduction straight-line kernel,
   the shipped version is 128-bit (a lane's 32 B not contiguous); same job, cold L2, cupti timing, median of 14 rounds:
   - true `.256` reads and writes (0 `.128` in SASS, 0 LDL/STL): **1.034×** (3.4% slower);
   - only the read side switched to `.256`: **1.030×**;
   - the "32 contiguous B per lane" ownership but still 128-bit PTX: **1.080×**; `.256` against it is 0.957×.
   The cost of changing ownership (8%) exceeds the gain of the width (about 4%). Before switching on your own kernel, do a same-job whole-kernel A/B over these three levels.

## Sources

- PTX ISA: the vector forms of `ld` / `st` (`.v8.b32` / `.v4.b64` 256-bit access, `.global` only, from `sm_100`), `ld.global.nc`,
  `.L1::evict_first` / `.L1::no_allocate`:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>
- API symbols of public infra: `cutlass.cute.runtime.make_ptr(..., assumed_align=)`, `cute.assume(..., divby=)`,
  `cute.nvgpu.CopyUniversalOp` (`num_bits_per_copy`, `invariant`, `l1c_evict_priority`), `cute.make_copy_atom`,
  `cute.recast_tensor`, `cutlass._mlir.dialects.llvm.inline_asm` / `llvm.extractvalue` / `llvm.StructType.get_literal`.
  For the calling convention of the inline asm channel see [`ptx_escape`](ptx_escape.md).
- Certification probe: `PROBE PASS max_diff=0.0` (round 2).
