# launch_identity_stamp (the kernel reports "which compiled instance am I and how was I launched": trace-time constants + launch special registers → a small gmem stamp tensor)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status: ✅ Verified** (GPU probe, not included in this release; `PROBE PASS max_diff=0.0`, round 1 of this certification): two compiled instances of the same parameterised stub kernel
> (differing in variant id, stage constant, grid shape, cluster direction and dynamic smem bytes), launched in the order V1 → V2 → V1 again from the cache;
> contracts stamp_V1 / stamp_V2 / stamp_V1_again / dyn_smem all `max_diff=0.0`, gate_catches `PASS (n/a)`. How to read it: every CTA's stamp row equals the expected row
> field by field; the V1 fetched again from the cache is field-by-field identical to the first one; `%dynamic_smem_size` is the same on every CTA and exactly equals the bytes allocated for that instance;
> checking V2's launch against V1's expectation table is judged a mismatch.
>
> This entry is an **instrument**: it does not change what the kernel does; it only makes the kernel write out its own identity. It makes no performance claim whatsoever.

## Purpose

A parameterised kernel often has several compiled instances (different tile / stage count / grid shape), picked by host-side dispatch according to the problem size.
What has to be judged is **the instance that is actually launched**, not the one the host code "should" pick. This entry has the kernel write its own identity into a small gmem tensor:

- **trace-time constants** (variant id, stage count, etc., i.e. the values compiled into this executable);
- **launch special registers**: `%nctaid` (grid), `%ntid` (block), `%cluster_nctaid` / `%cluster_nctarank` (cluster shape / size),
  `%dynamic_smem_size` (the dynamic smem bytes this launch carried).

The host compares against an expectation table CTA by CTA. The readings come from the hardware itself and do not depend on host dispatch code, on the profiler's kernel-name parsing, or on printing launch parameters.

## When needed (trigger)

- The grid / smem / shape in the kernel name read by the profiler does not match the instance you think is running; or you want the correctness / timing script to **assert automatically**
  "which instance runs", instead of checking profiler output afterwards.
- A dispatch cache (a `dict` storing `cute.compile` results by key) holds several instances, and you suspect an incomplete key fetches the wrong executable.
- A change also touched the dispatch routing or the stage formula along the way, and you need to see on the spot that "the actually launched instance changed".
- You want to know how much dynamic smem this launch actually carried, instead of computing it by hand.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

I32 = cutlass.Int32
# stamp fields: 0 variant, 1 stages, 2 bid, 3-5 grid xyz, 6 block x, 7-8 cluster xy, 9 cluster size,
#        10 dynamic smem bytes, 11 magic
F_VARIANT, F_STAGES, F_BID, F_GX, F_GY, F_GZ, F_BX, F_CX, F_CY, F_CSZ, F_SMEM, F_MAGIC = range(12)

@dsl_user_op
def dyn_smem_size(*, loc=None, ip=None):          # CuTe DSL has no ready-made wrapper; drop down to one PTX instruction
    return I32(llvm.inline_asm(T.i32(), [], "mov.u32 $0, %dynamic_smem_size;", "=r",
                               has_side_effects=True, is_align_stack=False,
                               asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

class Stamped:
    def __init__(self, variant_id, stages, grid, cluster, pad_bytes):
        self.variant_id, self.stages = variant_id, stages
        self.grid, self.cluster, self.pad_bytes = grid, cluster, pad_bytes

    @cute.jit
    def __call__(self, gStamp: cute.Tensor, stream):
        self.kernel(gStamp).launch(grid=list(self.grid), block=[128, 1, 1],
                                   cluster=list(self.cluster), stream=stream)

    @cute.kernel
    def kernel(self, gStamp: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        gx, gy, gz = cute.arch.grid_dim()              # %nctaid
        nx, _, _ = cute.arch.block_dim()               # %ntid
        cx, cy, _ = cute.arch.block_in_cluster_dim()   # %cluster_nctaid (cluster shape, in CTAs)
        csz = cute.arch.cluster_size()                 # %cluster_nctarank
        smem = utils.SmemAllocator()
        pad = smem.allocate_tensor(element_type=cutlass.Int8,
                                   layout=cute.make_layout(self.pad_bytes), byte_alignment=1024)
        if tidx == 0:
            pad[0] = cutlass.Int8(1)
            bid = bx + by * gx
            gStamp[bid, F_VARIANT] = I32(self.variant_id)   # trace-time constant = the value compiled into this executable
            gStamp[bid, F_STAGES] = I32(self.stages)
            gStamp[bid, F_BID] = bid
            gStamp[bid, F_GX] = gx; gStamp[bid, F_GY] = gy; gStamp[bid, F_GZ] = gz
            gStamp[bid, F_BX] = nx
            gStamp[bid, F_CX] = cx; gStamp[bid, F_CY] = cy; gStamp[bid, F_CSZ] = csz
            gStamp[bid, F_SMEM] = dyn_smem_size()
            gStamp[bid, F_MAGIC] = I32(0x5EED)

# host: the cache is keyed by the "full instance tuple"; the stamp tensor is prefilled with -1, rows of CTAs that were not launched stay -1
#   buf = torch.full((MAXCTA, 12), -1, dtype=torch.int32, device="cuda")
#   fn = cache.get(key) or cache.setdefault(key, cute.compile(Stamped(*key), t, stream))
#   fn(t, stream); torch.cuda.synchronize()
#   assert (buf[:n] except F_SMEM == expected rows).all() and (buf[n:] == -1).all()
#   assert pad <= buf[0, F_SMEM] <= pad + alignment slack, and the same on every CTA
```

Using it in a real kernel: add one optional small `Int32` stamp tensor to the existing kernel parameters, and wrap the write section in a
`cutlass.const_expr(self.stamp)` switch (when off, the whole section folds away at trace time and the launched body is unchanged);
the correctness script runs once with it on and compares the stamp rows field by field against the expectation table (grid, cluster, stage, smem); if any row differs it fails and prints the stamp row.

## Gotchas

1. **The stamp is of the executable, not of the dispatch intent.** Trace-time constants (`I32(self.variant_id)`) are burned into the executable at `cute.compile`;
   if the host-side routing function changed while the cache key did not, the stamp faithfully reports "the old instance" -- which is exactly what it is for. Conversely, do not pass the variant id
   in from the host as a run-time argument; then the stamp records the caller's intent, and the whole point of the mechanism is lost.
2. **The cache key must cover every parameter that changes the executable.** If the key covers only some parameters while the stage count is computed by a formula elsewhere, once the formula changes,
   what is compiled under the same key is already another instance; a `stages` field in the stamp shows it on the spot. The probe uses "V1 → V2 → V1 again from the cache" to verify that what the cache returns
   really is V1's executable.
3. **For the cluster shape use `block_in_cluster_dim()` (`%cluster_nctaid`), not `cluster_dim()`.** `[read from the public wrapper source; the probe used only the former]`
   The latter reads `%nclusterid` = **how many clusters the grid has**; the name looks alike but the semantics differ; mixing them up reads the number of clusters instead of the cluster width.
4. **`%dynamic_smem_size` is the total dynamic smem bytes this launch carried.** In the probe, both instances read exactly the allocated bytes, the same on every CTA.
   `[INFERRED]` when the allocator inserts padding for alignment, the reading exceeds the hand-computed value, so write the assertion as "`pad ≤ reading ≤ pad + alignment slack` and the same on every CTA";
   do not assert exact equality with the hand-computed value. CuTe DSL has no wrapper for this special register; use `llvm.inline_asm` with `mov.u32 $0, %dynamic_smem_size;`.
5. **Let only one thread write, one row per linear CTA id** (`bid = bx + by*gx (+ bz*gx*gy)`), with the stamp tensor prefilled with the sentinel −1. Then "the number of CTAs" needs no separate count:
   the first n rows all non-sentinel and every row from n on a sentinel is itself the grid assertion.
6. **The stamp section must be switchable off with `const_expr`.** An always-on stamp adds a stretch of STG and a tensor parameter on the hot path; turn it on for correctness runs and off for timing and profiler runs
   (or show that the two times are within noise).
7. **Write the expectation table in the acceptance script; do not import it from the kernel module.** If the expectation is imported from the code under test, it changes when the code under test changes, and the assertion is always true.

## Sources

- PTX ISA, Special Registers: `%nctaid`, `%ntid`, `%cluster_nctaid`, `%cluster_nctarank`, `%nclusterid`, `%dynamic_smem_size` --
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers>
- CuTe DSL public API: `cute.arch.grid_dim`, `cute.arch.block_dim`, `cute.arch.block_in_cluster_dim`, `cute.arch.cluster_dim`,
  `cute.arch.cluster_size`, `cute.arch.block_idx`, `utils.SmemAllocator`, `cute.compile`, `cutlass.cutlass_dsl.dsl_user_op`,
  `cutlass._mlir.dialects.llvm.inline_asm`.
- Certification probe: contracts stamp_V1 / stamp_V2 /
  stamp_V1_again / dyn_smem / gate_catches (`PROBE PASS max_diff=0.0`, round 1 of this certification).
