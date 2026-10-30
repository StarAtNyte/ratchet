# tma_descriptor_l2_promotion (the L2 promotion field of a TMA tensor map: when the CuTe DSL TMA atom offers no knob for it, make the atom read a separately host-encoded descriptor — and what that costs)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; tensor-map TMA loads built by `cpasync.make_tiled_tma_atom*` in CuTe DSL 4.6.x;
> `cuda.bindings.driver`.
>
> | | |
> |---|---|
> | **Knob** | The L2 promotion field in the `CUtensorMap` (128 B) (`CU_TENSOR_MAP_L2_PROMOTION_{NONE, L2_64B, L2_128B, L2_256B}`), written only when encoding with `cuTensorMapEncodeTiled`. `make_tiled_tma_atom*` has **no** parameter for it; two independent probes read back the descriptor the DSL atom carries, and in both the promotion is **`L2_128B`** |
> | **Method** | Let the DSL atom handle partitioning and coordinates as usual and only swap the descriptor it reads: `cute.copy(atom, src, dst, tma_bar_ptr=…, tma_desc_ptr=<a descriptor in a gmem slot>)`. The swapped-in descriptor is **encoded on the host once and uploaded once** (called path (1) below) |
> | **Footprint** | Invisible in values: the bytes delivered into smem are bit-identical under both promotion settings. In path (1) the default setting and the target setting are the same compiled artifact (they differ only in one field in gmem). This entry's probes did no PTX / SASS census |
> | **Load-bearing condition** | **Not a known lever.** On a GEMM tree that "streams a single pass over an fp16 operand larger than L2, 1 CTA/SM", whole-kernel timing of three variants in the same job (µs, 10 rounds × 40 runs): descriptor carried in the param **45.636**, path (1) default setting **45.692**, path (1) target setting **45.637** — target vs the same path's default 0.9988, the path itself +0.12%, all within noise. |

## Purpose

The contiguous gmem segment of one TMA box is often only 128 B (fp16, K-contiguous, SW128, box 64 elements wide). The promotion field tells L2
at what granularity to fetch from DRAM into L2 when this TMA read misses. It is **a property of the descriptor**: zero extra instructions, zero smem, zero registers.
CuTe DSL encodes the descriptor for you on the host and does not expose this field, so to A/B it you must make the atom read a descriptor you supply.
This "swap the descriptor" path also applies to other fields that the host API does not expose or that must be rewritten on the device — only the field position changes.

## When needed (trigger)

- You want to A/B promotion as a variable within the same job and find `make_tiled_tma_atom*` has no parameter for it.
- **Read the load-bearing condition first.** This entry certifies **the correctness of the path** and the field position; in both whole-kernel A/Bs with readings the target setting gave no gain, and switching paths costs
  the path's own price (path (1) +1.7% on an isolated pure-load kernel). It only counts if your same-job
  three-variant timing (gotcha 5) reads a difference above the noise floor.

## Minimal runnable example

### Common step: locate the field with the driver encoder (do not hard-code bit positions)

```python
from cuda.bindings import driver as drv

U64 = (1 << 64) - 1
P = drv.CUtensorMapL2promotion
PROMOS = [P.CU_TENSOR_MAP_L2_PROMOTION_NONE, P.CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
          P.CU_TENSOR_MAP_L2_PROMOTION_L2_128B, P.CU_TENSOR_MAP_L2_PROMOTION_L2_256B]

def encode_fp16_sw128(ptr, dims, strides_bytes, box, promo):
    # dims[0] = contiguous dimension (elements); strides_bytes = byte strides of dims[1:] (multiples of 16)
    res, tm = drv.cuTensorMapEncodeTiled(
        drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16, len(dims), ptr,
        [drv.cuuint64_t(d) for d in dims], [drv.cuuint64_t(s) for s in strides_bytes],
        [drv.cuuint32_t(b) for b in box], [drv.cuuint32_t(1) for _ in dims],
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B, promo,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(res) == 0, res
    return [int(w) & U64 for w in tm.opaque]                  # 16 u64 = 128 B, no ctypes needed

def words32(u64s):
    return [x for w in u64s for x in (w & 0xFFFFFFFF, w >> 32)]

# X: (ROWS, K) fp16, K-contiguous; box identical to your atom's (here BOX_ROWS rows × 64 elements)
args = (X.data_ptr(), [K, ROWS], [2 * K], [64, BOX_ROWS])
w = [words32(encode_fp16_sw128(*args, p)) for p in PROMOS]
pw = next(i for i in range(32) if len({x[i] for x in w}) > 1)   # the 32-bit word holding the field (there should be only one)
bits = 0
for x in w[1:]:
    bits |= x[pw] ^ w[0][pw]
shift = (bits & -bits).bit_length() - 1
code = lambda p: (w[PROMOS.index(p)][pw] & bits) >> shift          # the code of each setting
```

### Path (1): host-encode a map that differs only in promotion, upload once

```python
import cutlass, cutlass.cute as cute, cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
import torch

to_i64 = lambda ws: [x - (1 << 64) if x >= (1 << 63) else x for x in ws]
# one slot = 16 int64 (128 B); slot 0 = control setting, slot 1 = target setting; one compiled artifact selects via runtime sel
desc = torch.tensor(to_i64(encode_fp16_sw128(*args, P.CU_TENSOR_MAP_L2_PROMOTION_L2_128B))
                    + to_i64(encode_fp16_sw128(*args, PROMO)), dtype=torch.int64, device="cuda")
# cache desc by X.data_ptr(): the descriptor has the global address baked in (gotcha 6)

@dsl_user_op
def prefetch_tmap(addr, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
                    "prefetch.tensormap [$0];", "l", has_side_effects=True,
                    is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

# inside @cute.kernel (mDesc: the int64 tensor above; sel: Int32 runtime parameter)
mgr = utils.TensorMapManager(utils.TensorMapUpdateMode.GMEM, 128)
dptr = mgr.get_tensormap_ptr(mDesc.iterator + sel * 16, cute.AddressSpace.generic)   # must be generic (gotcha 2)
if warp_idx == TMA_WARP:
    with cute.arch.elect_one():
        prefetch_tmap(dptr.toint())                    # once at role entry
    ...
    cute.copy(tma_atom, tAgA[(None, ...)], tAsA[(None, stage)],
              tma_bar_ptr=bar, tma_desc_ptr=dptr)      # other arguments unchanged
```

A descriptor written by an H2D copy before launch **does not need** an in-kernel tensormap fence (gotcha 4).

## Gotchas

1. **Locate the field with the driver encoder; do not hard-code it.** `CUtensorMap` is an opaque 128 B. Encode the same map four times changing only the promotion;
   the bits that change are the field. Two independent probes read the same: 2 bits wide, bits 17–18 of the 2nd 32-bit word (i.e. the low half of the 1st u64),
   codes `NONE / 64B / 128B / 256B = 0 / 1 / 2 / 3`; rank-2 and rank-3 maps give the same position.
2. **The pointer handed to `cute.copy(..., tma_desc_ptr=)` must be in the generic address space.** Build two pointers to the same slot: `copy_tensormap`
   uses the `AddressSpace.gmem` one, `tma_desc_ptr` uses the `AddressSpace.generic` one. In the certification sweep,
   switching the acquire and `tma_desc_ptr` to the gmem pointer stopped compilation at `DSLRuntimeError: Failed creating the llvm::Module`
   (in that sweep the two shared one pointer; which of them triggers it was not separated). In the form that only does the acquire and issues TMA via raw PTX, a gmem pointer is fine
   (the example in [`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md)).
3. **The host-encoded map must be geometrically equivalent to the atom's descriptor, but it will not be byte-identical.** Measured: the host-encoded descriptor and the one read back from the atom
   differ in three u64s (one more bit differs in the word holding promotion; two other u64s are non-zero on the host and 0 in the atom read-back),
   yet the two byte sets loaded through the same TMA path agree element-wise; in an encode-knob sweep (OOB fill, dtype, swizzle setting, box rows) the bits each knob changes
   never fall in the promotion field. Self-check: a small kernel copies the atom's own descriptor to gmem with `TensorMapManager.init_tensormap_from_atom(atom, gmem_ptr, 0)`
   and reads it back to the host; require (a) the same value in the promotion field; (b) no differing bit inside the field;
   (c) after setting the atom's bytes to the target setting, the XOR against the host target-setting encoding is exactly the same as for the control setting; (d) the tile loaded through that descriptor equals the source tensor
   element-wise. Geometric non-equivalence raises no error; it just misplaces data silently.
4. **The fence exists only for descriptors "just written on the device".** Path (1)'s descriptor is written by H2D before launch and needs no
   in-kernel fence (none of the certification probe's five load paths has a fence, and all are element-wise exact).
5. **The A/B must have three variants, in the same job.** "param descriptor + default setting" vs "gmem descriptor + target setting" changes two things at once. To decide a setting use three variants:
   param (status quo), gmem + default setting (path changed only), gmem + target setting (field changed only); target / gmem default prices the field,
   gmem default / param prices the path; interleave them in the same job and measure with your timing convention (including the dirty-L2 window). On an isolated pure-load kernel
   (an fp16 tensor larger than L2, 12 rounds × 10 runs interleaved, µs): param 39.77, atom bytes placed in gmem 40.35, host default setting 40.45,
   host target setting 40.38 — target / default = 0.9983, host default / param = 1.0171.
6. **The descriptor has the global address baked in.** In path (1), any change of the tensor address requires re-encoding and re-uploading: cache by `data_ptr()`; when the address is unchanged each
   launch costs nothing extra; if every call uses a new pointer, encode + H2D land inside the timed window.
7. **How to prove the hardware reads your slot.** The field is invisible in values; `max_diff` cannot prove that `tma_desc_ptr` took effect. The certification probe's method:
   put the descriptor of **a different tensor** in the slot and issue with the original tensor's atom for partitioning; only if the output equals the other tensor is the hardware reading this slot.
   Then read the slot back to the host and check word by word: it should equal the atom's original descriptor, differing only in the field bits.
8. **Slot alignment**: each descriptor takes a 128 B slot, allocated 128-B-aligned (32 words per slot for an int32 buffer, 16 words per slot for an int64 buffer).
9. **Do not confuse it with `.L2::cache_hint`.** Promotion is a descriptor property (L2 fill granularity); `.L2::cache_hint` is a 64-bit policy operand on each TMA instruction
    (eviction priority), spelled as in gotcha 11 of [`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md). The two are independent.

## Sources

- CUDA Driver API: `cuTensorMapEncodeTiled`, `CUtensorMapL2promotion`
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html>
- PTX ISA: `cp.async.bulk.tensor`, `prefetch.tensormap`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- Public API symbols: `cuda.bindings.driver.cuTensorMapEncodeTiled`, `CUtensorMap.opaque`,
  `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom`, `cute.copy(..., tma_desc_ptr=)`, `cutlass.utils.TensorMapManager`
  (`TensorMapUpdateMode.GMEM`, `get_tensormap_ptr`, `init_tensormap_from_atom`), `cute.AddressSpace`,
  `cutlass._mlir.dialects.llvm.inline_asm`.
- GPU certification probes (two, independent of each other; not included in this release):
  - Path (1): ✅ `PROBE PASS max_diff=0` (round 4; field location,
    promotion isolation, five load paths element-wise exact).
  - Field location / discrimination of which slot the hardware reads (second independent certification): ✅ `PROBE PASS max_diff=0.0`
    (round 4; field location, discrimination of the slot the hardware reads, word-by-word check of the slot contents).
- Whole-kernel readings: three-variant same-job A/B.
