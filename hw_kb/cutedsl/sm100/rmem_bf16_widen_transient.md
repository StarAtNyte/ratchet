# rmem_bf16_widen_transient (bit-widening register-packed bf16 → f32: how to keep it at the point of use and stop it being merged into a resident f32 copy)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Interface: bit-widening register-packed bf16 (one pair per 32-bit register) → f32, and how to keep it "per use point, transient" under the CuTe DSL / NVVM / ptxas
> compile chain.
>
> **Status**: ✅ Verified (certified by a GPU probe (the probe is not included in this release),
> `PROBE PASS max_diff=0.0`, certification round 3; all seven contracts passed in the same round: the numerics of the four variants are bit-identical to numpy,
> and the three `cuobjdump -res-usage` register contracts are in the measured table below).
>
> **Certified surface**: only **register count** and numerics are certified, **not speed**. Same-job timing of the same mechanism on a whole kernel gave mixed readings (see gotcha 9):
> lever ② alone got slower, lever ① alone and both together got faster. Before changing the spelling, do a same-job A/B on your own kernel.

## Purpose

bf16 → f32 is itself just one shift / and (bf16 is the high 16 bits of f32): **recomputing it is almost free, storing it is expensive**:
a packed bf16 data set takes `N/2` registers, the widened f32 copy takes `N`. Two source spellings get quietly turned by the compiler into "the whole f32 copy stays resident":

- the same packed data is written as `t.load().to(Float32)` once at each of two use points (with other computation in between): after mem2reg the two loads are the same
  SSA value, the `fpext` is CSE'd, and one f32 copy lives from use point 1 to use point 2;
- a loop invariant is written as `.to(Float32)` inside the loop: the widening does not depend on the loop variable, so LICM hoists it into the prologue, and the f32 copy stays resident for the whole loop.

The source looks "per block, transient", yet the compiled live set gains two extra copies of `N` out of nowhere. The numerics are entirely correct; only the register count shows it.

**The merging happens at two levels, and both must be blocked**: the LLVM level (NVVM's CSE / LICM) and **the ptxas level**. Blocking only at the LLVM level
(side-effecting inline asm with identical asm text at both sites) is **not enough**: ptxas sees two identical `shl`/`and` sequences and merges and hoists them anyway;
measured on GPU, the register count is **exactly the same** as the naive spelling. The two levers that actually work:
① **load the invariant late, per segment, at the point of use** (do not let it stay resident); ② **widen at the two use points with different PTX forms** (ptxas sees two different values, nothing to merge).

## When needed (trigger)

- By design "the packed bf16 is the only resident copy and the widening is per block and transient", yet `regs_per_thread` is about one or two
  f32 copies above the live-set account you computed from that design; spill is 0 (with no register cap, ptxas simply uses more registers), and the number of resident warps drops accordingly.
- The prologue (before the loop) has an unusually high instruction count: the invariant's `PRMT` / `SHL` / `IMAD.U32 …, 0x10000` / `LOP3 …0xffff0000` widening was hoisted out of the loop.
- In the source, `.load().to(Float32)` of the same rmem tensor appears in ≥ 2 segments, with other computation in between (a warp shuffle reduction etc.).
- You already wrapped the widening in a `has_side_effects=True` inline asm, and the register count did not budge.

## Minimal runnable example

The three helpers match the spelling in the certification probe (the probe's `ld_v4` / `widen_pair` / `widen_pair_prmt`); the kernel body below only sketches the structure of the two
use points; take the geometry (words per segment, number of segments) from your own kernel.

```python
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector

def ld_v4(addr):
    """16 B global read -> vector<4xi32> (4 packed bf16x2 words). has_side_effects=True: LLVM does not hoist it out of the loop."""
    i32 = Int32.mlir_type
    r = llvm.inline_asm(ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
                        [Int64(addr).ir_value()],
                        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l",
                        has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)
    return vector.from_elements(ir.VectorType.get([4], i32),
                                [llvm.extractvalue(i32, r, [k]) for k in range(4)])

def widen_pair(u):
    """One packed bf16x2 word -> (low element f32, high element f32). Form A: shl / and.
    .b32 instructions may write .f32 registers directly (PTX same-width type relaxation)."""
    f32 = Float32.mlir_type
    r = llvm.inline_asm(ir.Type.parse("!llvm.struct<(f32, f32)>"), [Int32(u).ir_value()],
                        "{\n\tshl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xffff0000;\n\t}",
                        "=f,=f,r", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(llvm.extractvalue(f32, r, [0])), Float32(llvm.extractvalue(f32, r, [1]))

def widen_pair_prmt(u):
    """Bit-identical result to widen_pair, form B: byte permute with zeros."""
    f32 = Float32.mlir_type
    r = llvm.inline_asm(ir.Type.parse("!llvm.struct<(f32, f32)>"), [Int32(u).ir_value()],
                        "{\n\tprmt.b32 $0, $2, 0, 0x1044;\n\tprmt.b32 $1, $2, 0, 0x3244;\n\t}",
                        "=f,=f,r", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT)
    return Float32(llvm.extractvalue(f32, r, [0])), Float32(llvm.extractvalue(f32, r, [1]))

# Sketch of the kernel body (P: packed data, (4, NSEG) Int32; Q: loop invariant, same shape; Z: Int32 kernel parameter passed by the host, 0 at run time)
#   P = cute.make_rmem_tensor(cute.make_layout((4, NSEG)), Int32)       # the only resident copy: packed words
#   Q = cute.make_rmem_tensor(cute.make_layout((4, NSEG)), Int32)       # invariant: only per-segment scratch, not resident
#   for k in cutlass.range_constexpr(NSEG):
#       P[None, k].store(cute.TensorSSA(ld_v4(p_addr(k)), (4,), Int32))
#
#   # use point 1: widen with form B, dead right after use
#   for k in cutlass.range_constexpr(NSEG):
#       for i in cutlass.range_constexpr(4):
#           lo, hi = widen_pair_prmt(P[i, k]); ...
#   ...  # middle segment: only the packed words are alive now
#
#   # use point 2: widen again (form A), without reusing use point 1's f32; the invariant is loaded late, per segment, here
#   for k in cutlass.range_constexpr(NSEG):
#       Q[None, k].store(cute.TensorSSA(ld_v4(q_addr(k)), (4,), Int32))   # lever ①
#       for i in cutlass.range_constexpr(4):
#           xl, xh = widen_pair(P[i, k])          # lever ② (equivalent alternative: widen_pair at both sites, feed P[i, k] ^ Z here)
#           wl, wh = widen_pair(Q[i, k])
#           ...
```

Two levers, each handling one transformation:

| Transformation | Symptom | Lever |
|---|---|---|
| LICM: a loop invariant (and its widening) is hoisted into the prologue and stays resident for the whole loop | live set +N, prologue bloats | load the invariant **late, per segment** at the point of use (side-effecting `ld_v4`), widen only at the point of use. Wrapping only the widening in asm while the invariant is still loaded in the prologue is **ineffective** |
| CSE (at the LLVM level **and the ptxas level**): the same widening at two use points is merged into one, and the f32 copy lives from use point 1 to use point 2 | live set +N | widen at the two use points with **different PTX forms**: `widen_pair_prmt` at one, `widen_pair` at the other; or feed `word ^ Z` at the second (Z = a kernel parameter that is 0 at run time). Relying only on `has_side_effects=True` with identical asm text at both sites is **ineffective** |

Measured on GPU (the certification probe's toy kernel: per lane, 32 packed data words + a packed invariant of the same size, with one warp butterfly
reduction between the two use points; REG read with `cuobjdump -res-usage`, STACK 0 everywhere, no `LDL`/`STL` in SASS; the four counted variants are all bit-identical to the numpy reference):

| Spelling | REG | Notes |
|---|---|---|
| naive: `.load().to(Float32)` at both sites, invariant `cute.copy` in the prologue + `.to(Float32)` in the loop | 168 | both the data copy and the invariant copy stay resident as f32 |
| only switched to side-effecting `widen_pair` (same asm text at both sites), invariant still loaded in the prologue | 168 | **zero gain**: ptxas merges and hoists the identical `shl`/`and` anyway |
| + invariant loaded late, per segment | 94 | lever ① |
| + use point 1 switched to `widen_pair_prmt` (two forms) | **58** | lever ② |
| + use point 2 fed `word ^ Z` instead (same form) | **58** | equivalent alternative for lever ② (64 extra `LOP3`) |

The two levers must be applied **together** to reach the bottom: ① alone stops at 94, ① + ② reaches 58.

Element order: when `vector<4xi32>` and `vector<8xbf16>` are bitcast into each other, bf16 element `2i` is in the **low** 16 bits of word `i`,
element `2i+1` in the high 16 bits; `widen_pair` returns `(element 2i, element 2i+1)`.

Acceptance (independent of numerics; must be checked on the cubin):

```python
# ask the compile options to keep the cubin / PTX
compiled = cute.compile(launch, *args, options="--dump-dir=%s --keep-ptx --keep-cubin" % d)
# then: cuobjdump -res-usage <d>/**/*.cubin  -> REG should be back to the account "packed copy + transient block + addresses/scalars"; STACK:0
# if there is no cubin: ptxas -arch=sm_100a -v <d>/**/*.ptx -> "Used N registers", spill 0
```

## Gotchas

1. **"Per block" in the source is not "per block" when compiled.** Writing `for k: xk = t[None,k].load().to(Float32)` twice gives, after mem2reg, the same
   SSA value; the `fpext` is CSE'd and one f32 copy stays resident across the middle segment. The numerics are entirely correct; only the register count shows it.
2. **The invariant's widening gets LICM'd.** `cute.copy` into bf16 rmem in the prologue, `.to(Float32)` in the loop: the widening is hoisted out of the loop, and the f32 invariant stays resident for the whole
   loop. The packed bf16 invariant itself also takes `N/2` — to save more, load it late (lever ①).
3. **Load late with a `has_side_effects=True` inline load.** Whether a plain load can stay in the loop depends on whether the compiler can prove it does not alias the writes in the loop;
   the probe did not test a late-load spelling with plain loads, so do not rely on it. The L1 hit behaviour of the late load was not tested by the probe either.
4. **`has_side_effects=True` only governs the LLVM level, not ptxas.** Once the inline asm goes into PTX verbatim, ptxas treats it as ordinary instructions; two identical `shl`/`and` sequences
   on the same input register are merged and hoisted (measured 168 → 168). To make ptxas see "two different values": two PTX forms (`prmt` vs
   `shl`/`and`), or XOR one site's input with a **run-time** 0 (a compile-time constant 0 gets folded away).
   This does not contradict the `opaque_i32` seed of [`tmem_readout_segment_live_set`](tmem_readout_segment_live_set.md) gotcha 5: what is certified there is
   "recompute the address from the seed every round", while what fails here is "the same input with the same asm text appearing twice" — under that condition do not expect a side effect to block the merge.
5. **Constraint string of the inline asm**: two outputs `=f,=f`, one input `r`, returns `!llvm.struct<(f32, f32)>`, split with `llvm.extractvalue`;
   `$0/$1` are outputs, `$2` is the input. Wrap the block in `{ }`.
6. **`prmt` selector**: the 4 nibbles of `prmt.b32 d, a, 0, sel`, from low to high, select result bytes 0..3; values 0–3 take bytes of `a`, 4–7 take bytes of the second operand
   (here 0). `0x1044` puts bytes 0, 1 of `a` into result bytes 2, 3 and fills the low two bytes with 0 (low element); `0x3244` puts bytes 2, 3
   (high element). The probe's split variant is bit-identical to numpy.
7. **Setting only environment variables like `CUTE_DSL_KEEP_CUBIN` may leave no cubin in the dump directory.** That is what happened in the probe's first round (`cubin=0`, register contracts
   could not be judged); only after switching to `cute.compile(..., options="--dump-dir=… --keep-ptx --keep-cubin")` was the cubin obtained.
8. **Without a register cap, a higher register count does not produce spill.** So spill 0 does not mean the live set is fine; compute the live-set account from the design and compare it with REG.
   A cap is not the solution: one probe round put a `min_blocks_per_mp` cap on the naive spelling (an INFO reading, not a contract), and the price of pressing registers below the cap was
   STACK > 0 and hundreds of `LDL`/`STL` in SASS — ptxas chose to spill, not to recompute the widening for you. The problem must be solved at the source level with the two levers.
9. **Fewer registers is not the same as faster — the whole-kernel readings are mixed.** The same mechanism was timed with same-job pairing on the full kernel where this was found (locked clocks, three rounds): lever ② alone (two forms) **2.67% slower**; lever ① alone **5.26% faster**;
   both together 8.14% faster. After that kernel switched to this spelling no valid full eval was obtained again, so there is no KEEP reading. Before copying it, do a same-job A/B on your own kernel.

## Sources

- PTX ISA: `shl`, `and`, `prmt`, `ld.global.v4`, and the type-relaxation rule "operands of `.bN` instructions may be registers of any type of the same width":
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- LLVM Language Reference: the `sideeffect` marker of inline asm: https://llvm.org/docs/LangRef.html#inline-assembler-expressions
- CuTe DSL / cutlass public API: `cutlass._mlir.dialects.llvm.inline_asm`, `llvm.extractvalue`, `llvm.bitcast`,
  `vector.from_elements`, `cute.TensorSSA`, `cute.make_rmem_tensor`, `cutlass.range_constexpr`, `cute.compile(options=…)`.
- Certification probe (the toy kernel's geometry matches the student kernel where this was found):
  `PROBE PASS max_diff=0.0` (round 3).
