# tma_param_desc_inline_ptx (when issuing tensor-map TMA via inline PTX, take `[tmap]` directly as the address of the DSL atom's descriptor in the kernel parameters: no gmem copy, no tensormap fence pair)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Status: ✅ Verified (certified by a GPU probe, not included in this release: `PROBE PASS max_diff=0.0`, certification round 1,
> all three contracts pass in the same round: `C0-lowers` n/a, `C1-exact` 0.0, `C2-ptx` n/a). A bit-exact reading of the plain issue string without cache-hint using the same addressing is in the second item of "Sources".

| | |
|---|---|
| **Knob** | Inline PTX `cp.async.bulk.tensor.*` needs the **address** of the tensor map. If that map is exactly the descriptor of some `make_tiled_tma_atom*` atom and the atom is passed in as a `@cute.kernel` parameter, take the generic address of its copy in the kernel parameters directly (`param_desc_addr` in the example), instead of having every launch and every CTA first copy the descriptor into a gmem slot and then do `fence_tensormap_initialization` + `fence_tma_desc_acquire` |
| **Why** | PTX ISA: the tensor-map operand of `cp.async.bulk.tensor` is the generic address of the object residing in the `.param` / `.const` / `.global` state space. The descriptor in the kernel parameters is written before launch, and DSL's own `cute.copy` uses exactly it (the PTX definition chain has the same shape, see gotcha 8). The gmem-copy path adds per launch: one 128 B copy, one pair of tensormap proxy fences; if the copy and the use are on different threads, also one rendezvous (measured: on two trees this rendezvous already existed, so it is not an extra cost) |
| **Footprint** | PTX: **0** lines of `fence.proxy.tensormap` / `tensormap.cp_fenceproxy` in the kernel; 1 line of `prefetch.tensormap` (emitted by `cpasync.prefetch_descriptor(atom)`); the definition chain of the `[tmap]` register is `mov.b64 %rdA, <kernel parameter symbol>` → `cvta.param.u64 %rdB, %rdA`, the same as DSL's own `cute.copy`; the gmem-copy path is instead `ld.param` for the tensor pointer followed by `add.s64` for the offset |
| **Values** | Bytes delivered into smem are bit-identical to DSL `cute.copy` (correctness contract = bit-identical) |

## Purpose

TMA modifiers or operands that DSL's `cute.copy(tma_atom, ...)` does not expose have to go through inline PTX ([`ptx_escape`](ptx_escape.md)).
Inline PTX needs the **address** of the descriptor. This entry gives a way to obtain that address without a gmem copy — take, from the atom itself, the address of its copy in the kernel parameters; of the five addressing spellings tried side by side in the same probe,
this is the only one that ran on the GPU.

## When needed (trigger)

- You already copy the atom's descriptor into one gmem slot per CTA with `TensorMapManager(GMEM)` and acquire it with a fence pair, solely to give inline PTX
  a `[tmap]` address (the criterion is tensormap fence lines in the PTX and a `[tmap]` definition chain of `ld.param` + `add.s64`; a `membar` stall in the prologue
  is not a criterion — it may still rank first after switching to this entry).
- The descriptor is constant for the whole launch. Scenarios that swap the descriptor are not covered here: for swapping encode-time fields see
  [`tma_descriptor_l2_promotion`](tma_descriptor_l2_promotion.md) (host-encode once, hand it to `cute.copy(..., tma_desc_ptr=)`).
  The example in [`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md) also takes the atom descriptor's address via a gmem copy; switching that one to this entry's addressing is **untested**.

> **Whole-kernel readings** (same tree, only the descriptor addressing changed; that submission only removed the gmem slot + fence pair, switched to this entry's addressing, and added
> `prefetch_descriptor` for this atom): gmem-copy path → this entry, gate 0.0144 → 0.01375 ms (two readings 0.0138 / 0.0137, the latter from the confirmation job).
> In the timeline readings of the same period the steady-state cycles per beat did not change (producer 384 → 386, MMA 386 → 388 cycles) — what is saved is a fixed per-launch segment, not the main loop.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
from cutlass.cutlass_dsl import dsl_user_op, T

I32, I64 = cutlass.Int32, cutlass.Int64
ATT = llvm.AsmDialect.AD_ATT

@dsl_user_op
def param_desc_addr(tma_atom, *, loc=None, ip=None) -> I64:
    """Int64 generic address of the atom's descriptor copy in the kernel parameters (the only spelling that works, see gotcha 7)."""
    ptr_ty = _cute_ir.PtrType.get(_cute_nvgpu_ir.TmaDescriptorTiledType.get(),
                                  _cute_ir.AddressSpace.generic, 64)        # generic, not gmem
    ex = _cute_nvgpu_ir.atom_make_exec_tma(tma_atom._trait.value, loc=loc, ip=ip)
    fld = ir.Attribute.parse("#cute_nvgpu.atom_copy_field_tmaload<tma_descriptor_ptr>")
    p = _cute_nvgpu_ir.atom_get_value(ptr_ty, ex, fld, loc=loc, ip=ip)
    pv = p.value if hasattr(p, "value") and not isinstance(p, ir.Value) else p   # gotcha 9: cannot be omitted
    return I64(_cute_ir.ptrtoint(T.i64(), pv, loc=loc, ip=ip))

@dsl_user_op
def tma_tile(dst, tmap, c0, c1, mbar, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [I32(dst).ir_value(loc=loc, ip=ip), I64(tmap).ir_value(loc=loc, ip=ip),
         I32(c0).ir_value(loc=loc, ip=ip), I32(c1).ir_value(loc=loc, ip=ip),
         I32(mbar).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
        ".mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r,~{memory}",
        has_side_effects=True, is_align_stack=False, asm_dialect=ATT, loc=loc, ip=ip)
# Form with an L2 policy operand (the one certified by the probe): append `.L2::cache_hint` to the mnemonic,
# append `, $5` to the operand string and `,l` to the constraints, pass one more I64 policy value. Which policy value and which operand path it goes on are outside this entry.

# ---- kernel: tma_a is the atom built on the host by make_tiled_tma_atom* and passed in as a @cute.kernel parameter ----
if warp == PRODUCER_WARP:
    cpasync.prefetch_descriptor(tma_a)                 # prefetch in the prologue, as before
# mbarrier init / fence / CTA rendezvous as before; no descriptor slot tensor, TensorMapManager or any tensormap fence needed
if warp == PRODUCER_WARP:
    tmap = param_desc_addr(tma_a)                      # take it once, outside the loop
    # ... each stage:
    with cute.arch.elect_one():                        # inline PTX has no built-in election
        cute.arch.mbarrier_arrive_and_expect_tx(full_bar + s, NB * BOX_BYTES)
        for j in cutlass.range_constexpr(NB):
            tma_tile(slot_base + (s * NB + j) * BOX_BYTES,   # destination: linear (unswizzled) offset from the slot base
                     tmap,
                     k0 + j * BOX_K, row0,                    # {c0 = element coordinate of the innermost (contiguous) dim, c1 = row}
                     (full_bar + s).toint())
```

In the certification probe every slot base is 1024 B aligned (SW128) and the inner box dimension = one SW128 atom; coordinates and destination addresses as written above: 16 issues (static PTX line count), and the 8 boxes of the last two rounds read back bit-identical to numpy.

## Gotchas

1. **What is certified is an atom "built on the host and passed in as a kernel parameter".** Whether an atom built inside the kernel has this parameter copy is untested.
2. **Do not mix the two paths.** After switching to this entry, delete the per-CTA descriptor slot, `TensorMapManager`, `init_tensormap_from_atom` / `copy_tensormap`,
   `fence_tensormap_initialization` and `fence_tma_desc_acquire` together; accept by checking that the number of tensormap fence lines in the PTX is 0
   (certified readings: this entry 0, gmem-copy path 1).
3. **No acquire fence needed.** The certified form has no tensormap fence at all, and lands bit-correct. The acquire fence is for descriptors just written by the device into gmem / smem.
4. **Prefetch still uses `cpasync.prefetch_descriptor(atom)`**; in the certified form it emits 1 line of `prefetch.tensormap`.
5. **Values cannot tell which descriptor the hardware read.** The two paths are bit-identical; look at the PTX: 0 fence lines, the `[tmap]` definition chain traced to the kernel parameter symbol
   (`cvta.param`). Counting `.L2::cache_hint` modifiers cannot tell them apart either: in the certification job the DSL `cute.copy` control, the gmem-copy path and this entry all have
   16 / 16 instructions with `.L2::cache_hint` — the TMA emitted by DSL `cute.copy` carries this modifier itself.
6. **What is saved is a fixed per-launch segment.** Main-loop cycles per beat are unchanged (whole-kernel readings under "When needed"); in the probe "kernel entry → first barrier flip"
   is 5929 cycles for this entry and 5957 cycles for DSL `cute.copy` — both read the same descriptor; flat.
7. **Only one addressing spelling compiles.** The same probe ran five spellings, each in its own subprocess; only the one in the example (`atom_make_exec_tma` →
   `atom_get_value(…<tma_descriptor_ptr>)`, result pointer type in the **generic** address space) passed compile, run and values. The four that stopped at compile:
   the same path with the pointer type written in the **gmem** address space (LLVM translation fails, `Failed creating the llvm::Module`);
   `cute_nvgpu.get_grid_constant_pointer(atom)` and `cute_nvgpu.get_tma_desc_addr(atom)` (IR verification: neither accepts a non-executable atom);
   reading the `tma_desc` field of the non-executable atom first and then `get_tma_desc_addr` (IR verification reports
   `'cute_nvgpu.get_tma_desc_addr' op operand #0 must be CopyAtomTypeInterface instance, but got '!cute_nvgpu.tma_descriptor_tiled'`).
   When you hit this kind of `ICE IR Verification Failed`, do not tweak arguments; go back to the example's spelling.
8. **Definition-chain readings (an instance of the footprint criterion).** This entry and DSL `cute.copy`: `mov.b64 %rdA, <kernel parameter symbol>` → `cvta.param.u64 %rdB, %rdA`;
   gmem-copy path: `ld.param.u64` for the tensor pointer → `add.s64` for the per-CTA offset.
9. **Unwrap the result of `atom_get_value` before handing it to `ptrtoint`.** The `pv = p.value if …` line in the example: the result may be wrapped by the DSL into a pointer object, and handing it directly to
   `ptrtoint` raises `AssertionError` at trace time (`assert isinstance(arg, _cext.ir.Value)` in `get_op_result_or_value`) — on another tree
   the same addressing without this line failed exactly that way.
10. **It uses the bindings under `cutlass._mlir.dialects`** (`atom_make_exec_tma` / `atom_get_value` / `PtrType` / `ptrtoint`), not the public `cute.*`
    layer. This entry is certified on one DSL version only; before switching versions, rerun a per-spelling probe like the one in gotcha 7.

## Sources

- PTX ISA, "Data Movement and Conversion Instructions": `cp.async.bulk.tensor` (state space and generic addressing of the tensor-map operand,
  `.L2::cache_hint`), `prefetch.tensormap`, `fence.proxy.tensormap`
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA C++ Programming Guide, "Using TMA to transfer multi-dimensional arrays" (passing `CUtensorMap` as `__grid_constant__`)
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CuTe DSL public API symbols: `cpasync.make_tiled_tma_atom`, `cpasync.CopyBulkTensorTileG2SOp`, `cpasync.prefetch_descriptor`,
  `cutlass.utils.TensorMapManager`, `cutlass._mlir.dialects.cute_nvgpu.atom_make_exec_tma` / `atom_get_value` /
  `TmaDescriptorTiledType`, `cutlass._mlir.dialects.cute.PtrType` / `ptrtoint` (the field name `tma_descriptor_ptr` is
  `cutlass.cute.nvgpu.cpasync.copy.TMA_DESC_PTR_FIELD_NAME`).
- Related entries in this KB: [`ptx_escape`](ptx_escape.md), [`tma_descriptor_l2_promotion`](tma_descriptor_l2_promotion.md),
  [`tma_gather4_indexed_rows`](tma_gather4_indexed_rows.md).
- GPU certification probe — **✅ `PROBE PASS max_diff=0.0`**
  (certification round 1). Seven variants, one subprocess each: DSL `cute.copy` control, gmem-copy path control, five addressing spellings; what ran were the two controls and the example's spelling,
  all three bit-identical to numpy; `C2-ptx` (per variant: tensor lines / lines with hint / tensormap fence lines / `prefetch.tensormap` lines): this entry 16 / 16 / 0 / 1,
  gmem-copy path 16 / 16 / 1 / 1.
- Plain issue string (no `.L2::cache_hint`) with the same addressing: another G3 probe (its own entry does not include it; only this reading is borrowed) `PROBE PASS max_diff=0.0`
  has two variants in which one operand path uses the example's `tma_tile` form + this entry's addressing, landing bit-identical to numpy (`bad_words` 0).
