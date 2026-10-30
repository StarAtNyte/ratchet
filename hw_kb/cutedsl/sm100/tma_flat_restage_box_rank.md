# tma_flat_restage_box_rank (re-target the destination face onto a flat gmem staging buffer: the 256-element box limit and the descriptor-rank footprint)

> One file per hardware interface. Operator-agnostic; mechanism only.
> Architecture scope: **sm100 (Blackwell)**; footprint readings taken from sm100 SASS (`UTMALDG.*` / `UTMASTG.*`).
> The box-dimension limit itself is a fact of the descriptor encoder, not specific to this generation; what carries over directly is the way to read it,
> not the readings.
>
> This entry records **one knob + its SASS footprint + the condition under which it is load-bearing**, not a percentage:
>
> | | |
> |---|---|
> | **Knob** | When the contiguous run handed to TMA exceeds **256 elements**, it cannot be a single box mode. Split it into two modes as `(run/256, 256)`; **the address function is unchanged**, the cost is one more descriptor dimension |
> | **Footprint** | The suffix of `UTMALDG.<rank>D` counts **the rank of the tensor map**. Verbatim dotted-opcode census before and after delivery: `["UTMALDG.2D", "UTMASTG.2D"]` → `["UTMALDG.2D", "UTMALDG.3D", "UTMASTG.2D"]` |
> | **Load-bearing condition** | The mode split itself is worth nothing; it is the **ticket of admission**: it lets "a flat gmem staging buffer rearranged once up front" be delivered into the face the consumer wants by **one** TMA. The gain belongs to that rearrangement; readings in the last section of "Gotchas" |
>
> **This entry's certification coverage is the dossier census + gate window of one KEEP, not a numeric probe.** That is not a downgrade; it is the nature of this
> axis: before and after the split the address function is byte-identical, so the computed numbers are byte-identical too; any `max_diff`
> probe is **inherently blind** to it, and only the SASS census sees the rank.

## Purpose

TMA only delivers byte orders that **a box can frame in gmem**. When the face a downstream consuming instruction wants (e.g. the UTCCP face of `tcgen05`,
or any instruction with a fixed requirement on the byte order of its smem face) is not the image of any
box over the original tensor, the delivered form **takes this out of the descriptor**: rearrange the bytes once, by the destination address function,
into a gmem staging buffer; from then on delivery degenerates into "a trivial copy of one contiguous run", and the descriptor no longer
needs to express any interleaving.

The cost is the only descriptor-level constraint that appears at this point: **one box dimension spans at most 256 elements**. A longer run
is split into two modes as `(run/256, 256)` — the split is stride-exact and the address function is unchanged, but the descriptor gets one more dimension,
and that dimension is visible in SASS.

## When needed (trigger)

- In the smem byte order the downstream instruction wants, the 16 B contiguous units come from rows far apart in the original gmem tensor ⇒ no
  box can produce it, and TMA swizzle cannot fix it either (it only permutes **whole** 16 B units within a 128 B line, never splits
  a unit into pieces). **This point is a derivation from the descriptor constraints and is outside this entry's certification coverage.**
- Your current form **rearranges in smem once per stage**, so the rearrangement sits on the producer's critical path:
  symptoms are high smem store-side wavefronts and low tensor-unit active (magnitudes in the last section of "Gotchas").
- You wrote a flat descriptor and the run exceeds 256 elements.
- You thought you built a 2D descriptor but SASS shows `.3D` (or the other way round) — the rank can only be read back from the census;
  the construction side will not tell you.

## Minimal runnable example

Three views sit on **the same bytes** and interlock: the consumer's library layout, the TMA view of the same smem, and the descriptor view of the gmem
staging buffer. Below is the delivered form; placeholders are marked with `<>`.

```python
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

# Units are elements, not bytes. One stage's destination face = RUN contiguous elements; STAGES slots in smem form a ring.
# The gmem staging buffer is blocked as (C0, C1): block (c0, c1) is the RUN contiguous elements at c0*RUN + c1*RUN*C0.
RUN, STAGES, C0, C1 = 2048, 5, 8, 32

# (1) The consumer's smem face: allocate it with the library layout the downstream instruction requires; do not invent one.
#    The only condition for the second view to be legal is "each stage is exactly one contiguous run". Assert it on the host:
#    if the library layout ever changes, that is a compile-time error rather than a silently wrong face.
sFace = smem.allocate_tensor(<DTYPE>, <consumer_layout>, byte_alignment=1024)
assert cute.cosize(<consumer_layout>) == RUN * STAGES

# (2) The TMA view of the same bytes. The descriptor cannot express the library layout, and need not -- copy only needs
#    the address function, and 256*i + j of (RUN/256, 256) is byte-identical to "single mode RUN, stride 1".
assert RUN % 256 == 0
flat = cute.make_layout((RUN // 256, 256, STAGES), stride=(256, 1, RUN))
sFace_tma = cute.make_tensor(sFace.iterator, flat)

# (3) The descriptor view of the gmem staging buffer: the innermost box mode spans 256 contiguous elements, both non-innermost strides
#    (256 and RUN*C0) are multiples of 16 B.
mG = cute.make_tensor(<scratch_ptr>, cute.make_layout(
    (C0 * (RUN // 256), 256, C1), stride=(256, 1, RUN * C0)))

box = (RUN // 256, 256)        # box has two modes, tensor has three -> tensor map rank 3 -> SASS .3D
op = cpasync.CopyBulkTensorTileG2SOp()
tma, gT = cpasync.make_tiled_tma_atom(op, mG, flat, box, 1)   # last argument = multicast count
gT = cute.zipped_divide(gT, box)       # from here on one block coordinate names one block directly

# On both sides first fuse the two box modes into one vector mode; tma_partition then handles both sides at once.
# Second and third arguments (0, make_layout(1)) = no multicast. Return order is (smem side, gmem side).
triv = cute.make_layout(1)
tGs, tGg = cpasync.tma_partition(tma, 0, triv,
                                 cute.group_modes(sFace_tma, 0, 2), gT)

# Issue: the third mode needs a coordinate too, and the middle dimension is always 0 -- the box already covers the whole mode.
cute.copy(tma, tGg[None, (<c0>, 0, <c1>)], tGs[None, <s>], tma_bar_ptr=<bar> + <s>)
```

## Gotchas

- **256 is elements, not bytes.** The limit counts elements: for a 1-byte type 256 elements = 256 B; for a 2-byte type
  the same 256 elements = 512 B. The split factor only has to **divide** the run and be at most 256 (the invariance of the address function
  does not depend on the factor); the delivered form uses 256 and asserts divisibility on the host.
- **The split must be stride-exact, otherwise you get silently wrong numbers.** `(run/256, 256)` with stride `(256, 1)`
  has the same address function as "single mode run, stride 1", byte for byte — that is the **whole** reason the consumer side (UTCCP / MMA) is entirely
  unaffected. Any other stride quietly changes the address function: the numbers are wrong, and neither the construction side,
  the descriptor constructor nor ptxas reports anything.
- **Only the SASS census sees the rank, and it lands on the cache-policy axis.** The census collects verbatim
  dotted opcodes, so the rank change shows up in the readings of the "cache-policy bits" axis, although it has nothing to do with cache policy.
  `.<rank>D` counts **the rank of the tensor map, not the number of box modes**: after delivery both forms coexist —
  the operand descriptor built on a three-mode tensor whose last mode has stride 0 (shaped like `(m, k, 1)` / stride `(k, 1, 0)`)
  still reads `.2D`, the one built on the staging layout whose three modes are all non-degenerate reads `.3D`, and both are built by the same
  `make_tiled_tma_atom` + `CopyBulkTensorTileG2SOp`. So: the cost of the split is exactly
  one dimension, and "is the descriptor I built the one I think it is" can only be read back on this axis.
- **Each call site gets one more coordinate that is always 0.** The box covers the whole middle mode, so the block coordinate is written
  `(c0, 0, c1)`. It is forced by the box split, not some optional optimisation headroom.
- **Assert the precondition of the second view; do not assume it.** The two views are byte-identical under the condition "each stage of the library face
  is one contiguous run". Written as the host-side assert `cute.cosize(...) == RUN * STAGES`, a change in the library layout
  hits a compile error; if you assume it, you get a face that compiles, runs to completion, and is numerically wrong.
- **Fuse the box modes on both sides into one vector mode before handing them to `tma_partition`** (`group_modes(..., 0, 2)`,
  which fuses both sides at once). This is the delivered form; "what happens if you do not fuse" has never returned a verdict in this KB, so do not treat it as a certified rule.

### Load-bearing condition (who this gain belongs to, and the boundary of this entry's certification coverage)

The structural condition delivered together with this rank change: before, the producer's critical path **rearranged the destination face in
smem once per stage**; after, the prologue writes all bytes once, by the destination address function, into the gmem staging buffer
(with a grid-level rendezvous holding the mainloop behind it), and the mainloop's TMA reads the staging buffer directly. Under this condition
the gate window went **0.22346 → 0.19305 ms (−13.61%)**, and the NCU readings of the same delivery are smem store-side
wavefronts **6493394 → 1106686**, tensor-unit active **58.83% → 72.36%**.

This percentage **belongs to that rearrangement, not to the split**: the split itself has no measurable cost; it is only the ticket that lets the flat staging buffer be
delivered by one TMA. On another tree the price of the rearrangement changes; the way to write the ticket does not.

**Two things this entry does not cover** (no verdict ever returned; must not be used as rules): that "no box over the original tensor can produce
the destination face" is a derivation from the descriptor constraints; how exactly it fails when the innermost box mode is less than 16 B contiguous, or a non-innermost stride is not a
multiple of 16 B, was never tested in this KB — the delivered descriptor **satisfies** both conditions; that is its
property, not a certified necessary condition.

## Sources

- CUDA Driver API, "Tensor Map Object Management" (the per-dimension boxDim limit of `cuTensorMapEncodeTiled`; the 256 limit in this entry is the one
  the delivered code asserts on the host)
  <https://docs.nvidia.com/cuda/cuda-driver-api/index.html>
- PTX ISA, "Data Movement and Conversion Instructions", `cp.async.bulk.tensor`
  (box / block coordinate semantics of the tensor-tile form)
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CuTe DSL / CUTLASS public API symbols:
  `cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`,
  `cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom`,
  `cutlass.cute.nvgpu.cpasync.tma_partition`;
  `cute.make_layout`, `cute.make_tensor`, `cute.zipped_divide`, `cute.group_modes`,
  `cute.cosize`, `cute.copy`.
- Certification evidence (**census + gate window, not a numeric probe** — see the note at the top of this entry on "invisible in values"):
  the dossier census of one KEEP delivery, axis `cache_ops` (verbatim dotted opcodes)
  `["UTMALDG.2D", "UTMASTG.2D"]` → `["UTMALDG.2D", "UTMALDG.3D", "UTMASTG.2D"]`,
  gate window 0.22346 → 0.19305 ms (−13.61%). Three pieces of corroborating evidence from the same delivery,
  all GPU-certified readings:
  NCU (smem store-side wavefronts 6493394 → 1106686, tensor active 58.83% → 72.36%),
  the device window plus a byte-by-byte check of the staging buffer against an independent numpy blocking (0 bytes mismatched),
  and ptxas (CLEAN, spill 0/0).
