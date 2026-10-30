# guard_predicate_issue_distance (forward guard branches inside the mainloop: the issue distance of the predicate)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> This entry is in the same family as [`ew_segment_latency_schedule`](ew_segment_latency_schedule.md)
> but on a **different axis**: that one governs the producer→consumer spacing inside one stretch of
> **straight-line code** (ptxas buys it with register headroom; source order has no effect); this one
> governs one thing **across basic blocks** — when the guard branch's predicate is ready. A branch cuts
> the basic block, and ptxas's list scheduler does not hoist work from after the branch to before it
> (that would be speculation, not reordering), so this spacing has **no compiler safety net** — it is
> purely a property of the source.

---

## Purpose

An SM90 forward guard branch — `@!P1 BRA <skip>` skipping a stretch of optional work — is not
zero-cost control flow on the ledger. A warp **cannot issue** past a `BRA` until the guard predicate
register `P*`/`UP*` is ready; and the predicate being ready waits for its production chain. Hence:

**If the guard predicate's input is a long-latency value that only arrives this iteration (a gmem/L2
fetch, an smem fetch, an SFU result), the whole latency is charged to this `BRA`** — in NCU's per-PC
ledger it shows up as "one or two `BRA`s monopolise the top of the ledger, with `long_scoreboard` /
`short_scoreboard` as their dominant stall".

This cost does **not** show up on any of the usual health lines: registers, spill, smem bytes, DRAM
ratio, `tensor_pct`, barrier share can all be perfectly on target, while every mainloop iteration still
stands for hundreds of cycles at each of the two branches. This entry answers three things: how to
measure it, whom it belongs to, and the only source lever that moves it.

---

## When needed (trigger)

In the NCU per-PC ledger (per-PC pricing such as `build_ledger` / `price_report` in
`notes/stall_tools.py`) **all** of these hold:

1. One or two or three `BRA`s on their own become the bulk of the top of the ledger (a single family
   share ≥ 5%, or a few together ≥ 10%);
2. The dominant stall of those `BRA`s is **`long_scoreboard`** (waiting on gmem/L2) or
   `short_scoreboard` (waiting on smem/SFU), **not** `not_selected` / `branch_resolving` / `barrier` /
   `wait`;
3. The branch is **forward** (its target is after itself) — forward means it is not a spin wait and
   not loop control;
4. It is **inside the mainloop body** (located by the back-edge interval, see §(a)-2).

**First rule out four neighbouring forms; they do not belong to this entry:**

| Form | Signature | Go to |
|---|---|---|
| dominant stall on the BRA is `barrier` / `wait` | waiting on sync or on an async unit | [`mbarrier_wait_encoding`](mbarrier_wait_encoding.md) / [`wgmma_deferred_wait`](wgmma_deferred_wait.md) |
| the branch is a **back edge** (target before itself) | loop control itself | unrolling/blocking, not this entry |
| the branch **diverges** (predicate not uniform within the warp) | paired with `BSSY` / `BSYNC` in SASS | both paths execute, a completely different cost model; **this entry's distance readings mean nothing for divergent branches** — remove the divergence first |
| no guard branch, `short_scoreboard` + `mio_throttle` dominate inside the segment | latency/throughput of a straight-line segment | [`ew_segment_latency_schedule`](ew_segment_latency_schedule.md) |

**Occupancy prerequisite**: when there are enough warps on the same SMSP, other warps cover this
latency and the whole entry does not apply (check `sm__warps_active` first). Low occupancy (1–4 warps
per SMSP) — the norm for large-budget warp-specialized kernels — is the precondition for this entry.

---

## Minimal runnable example

### (a) Measurement: two readings, `d_src` and `d_prod`

**1) Get the SASS.** Whichever of three routes works first, exactly as in
[`ew_segment_latency_schedule`](ew_segment_latency_schedule.md) §(a):
`cute.compile(..., options="--dump-dir=<dir> --keep-cubin --keep-ptx")` writes the cubin → `nvdisasm -c`
(if `-c` is unavailable fall back to `nvdisasm`, at the cost of having no labels — the next step then has
to use addresses); if what lands is PTX, run `ptxas -arch=sm_90a -O3` yourself; if neither, use
`ncu --page source --print-source sass`. **When comparing variants, always use text obtained by the
same route**; the two disassembly formats have different line shapes.

**2) Locate the mainloop body = the back-edge interval.** A back edge = a `BRA`/`BRX`/`JMP` whose target
label is before itself; take the mainloop body as "the back-edge interval densest in floating-point/tensor
instructions" (or the smallest interval containing all MMAs — `find_loop()` in
`notes/sched_live_tools.py` uses exactly this method: the largest back-edge interval, falling back to
the HGMMA envelope).

**3) Find the guard branches.** `BRA`s inside the interval that carry a predicate field and whose target
label is **after** themselves:

```python
_INSTR  = re.compile(r"^(@!?U?P(?:T|\d+)\s+)?([A-Z][A-Za-z0-9_.]*)\b\s*(.*)$")
_PREDTOK = re.compile(r"@(!?)(U?P)(\d+)")
# instrs[i] = (addr, pred, opcode, operands); lab2idx is built from the `.L_x:` lines
def guarded_branches(instrs, lab2idx, lo, hi):
    out = []
    for i in range(lo, hi + 1):
        _, pred, op, ops = instrs[i]
        if op.split(".", 1)[0] != "BRA":
            continue
        pm = _PREDTOK.match(pred)                 # no predicate = unconditional jump, not a guard
        if not pm:
            continue
        tgt = None
        for lab in re.findall(r"\(?(\.[A-Za-z_][A-Za-z0-9_.]*)\)?", ops):
            tgt = lab2idx.get(lab, tgt)
        if tgt is not None and tgt > i:           # only forward ones are guards
            out.append((i, (pm.group(2), int(pm.group(3)))))
    return out
```

**4) Walk the predicate's dependency closure backwards from the BRA** and take two distances. **Key
implementation detail: duplicate the loop body before walking** (`dbody = body + body`, starting from
`j + len(body)`) — otherwise, when the predicate is produced by the **previous** iteration, the walk
falls off the head of the slice and reads "no producer" instead of its real cross-back-edge distance.

```python
def chain_distances(body, j, pred_reg):
    n, want = len(body), {pred_reg}
    dbody, jj = body + body, j + len(body)
    d_prod = d_src = None
    for k in range(jj - 1, -1, -1):
        defs, uses = du(dbody[k])          # (defs, uses) = {(kind, idx)}, including R/UR/P/UP
        if not (defs & want):
            continue
        if d_prod is None:
            d_prod = jj - k                                  # nearest predicate producer
        want = (want - defs) | uses
        if d_src is None and dbody[k][2].split(".", 1)[0] in LONG_LAT:
            d_src = jj - k                                   # nearest long-latency instruction in the closure
        if not want:
            break
    return (d_prod or 2 * n), (d_src or 2 * n)

LONG_LAT = {"LDG", "LD", "LDL", "LDS", "LDSM", "LDGSTS", "MUFU",
            "ATOM", "ATOMG", "ATOMS", "I2F", "F2I"}
```

A minimal usable `du()` (a guard chain contains only the `LDG`/`IMAD`/`ISETP`/`MOV`/`LOP3` families, so
no general decoder is needed): split operands on top-level commas; for instructions with **no
destination operand** such as `BRA`/`BSSY`/`ST*`/`BAR*`, everything is a use; for the other instructions
the **first 1** operand is a def, and for opcodes containing `SETP`/`PLOP3`/`VOTEU`/`R2P` the **first 2**
are defs (`P0` and `PT` of `ISETP.NE.AND P0, PT, R5, RZ, PT`); any operand containing `[` is a use;
catch register tokens with `\b(UR|UP|R|P)(\d+)(\.(?:64|128))?` and expand them by width (`R2.64` →
R2,R3); `RZ`/`PT`/`URZ`/`UPT` naturally don't match.

**5) Reading table.** Read the two together; **`d_src` carries the weight, `d_prod` is secondary**:

| `d_src` (long-latency source → BRA) | Reading | What to change |
|---|---|---|
| **≤ 8 slots** | the guard predicate's long-latency input is fetched **this iteration**, and the whole latency is charged to this BRA | **move the fetch to the previous iteration** (§(b)) |
| **≥ one loop-body length** (closure crosses the back edge) | the input was issued in the previous iteration; what the BRA stands for is only the residue | this entry is exhausted; if the BRA is still high, check divergence / occupancy |
| 8–24 slots | intermediate: the fetch is in this iteration but partly covered by earlier work | see how much further §(b) can move it; also confirm that this preceding work is not fake work excluded by gate 3 of §(c) |

`d_prod` (the **nearest predicate producer** → BRA) is another cut of the same scene, and it is
**equivalent to `d_src` only when `d_prod` itself is a long-latency consumer**:

- small `d_prod` and large `d_src` = the `ISETP` hugs the branch, but its inputs have long been in
  registers — **healthy**, no change needed;
- small `d_prod` and small `d_src` = the problem this entry fixes;
- large `d_prod` = the predicate was computed early with independent work still queued behind it —
  healthy, and this is exactly what SASS commonly looks like after §(b) lands (the wait moves from the
  BRA to the `ISETP`/fetch before the branch; on the ledger, "pre-branch families rise, the BRA family
  falls").

**Don't report only `d_prod`.** The two readings diverge: ptxas has every reason to place an `ISETP`
whose "inputs are already ready" right next to the BRA (it owes no latency; issuing early or late is the
same), in which case `d_prod` reads 1–2 and looks like a problem when there is none at all.

---

### (b) The only lever: move the guard predicate's **long-latency input** to the previous iteration

The compiler cannot do this, for two independent reasons; remember both:

1. **It will not hoist a fetch across the back edge.** Issuing iteration i+1's load in iteration i is
   rewriting the dependency graph (software pipelining), not reordering; ptxas's list scheduler only
   does the latter.
2. **It will not hoist work from after a branch to before it.** That is speculation across control
   flow. So how much "independent work in front of the BRA" there is is decided entirely by the source.

Hence three spellings, from least to most invasive. All three have **exactly the same control flow**
(the same number of guard BRAs, the same amount of work) and differ only in **when the input is
fetched**:

**(b1) The current form (the one to fix)**: the input is fetched this iteration, compared on the spot,
branched on the spot.

```python
for i in cutlass.range(n_iter, unroll=1):
    f = mFlag[(bidx, idx_of(i))]      # long latency: only issued this iteration
    if f != 0:                        # predicate computed right at the branch
        ...                           # optional section
    ...                               # rest of this iteration's work (depends on the results above, cannot be hoisted)
```

**(b2) Loop-carried input (lookahead)**: this iteration fetches the input the **next** iteration needs;
the predicate is still computed in this iteration. The AST preprocessing of `cutlass.range`
automatically turns scalars re-bound in the loop body into loop-carried values (`write_args`), so this is
plain Python assignment, no special API needed:

```python
f = mFlag[(bidx, idx_of(0))]              # prologue: fetch one slot first
for i in cutlass.range(n_iter, unroll=1):
    f_next = mFlag[(bidx, idx_of(i + 1))]  # this iteration issues the next iteration's fetch
    if f != 0:                             # consumes the value fetched in the previous iteration
        ...
    ...
    f = f_next                             # loop-carried
```

**(b3) Loop-carried predicate itself**: even the comparison moves one iteration earlier, and the branch
consumes a loop-carried `Boolean`. `Boolean` is a public numeric type of `cutlass` and can be used
directly as a loop-carried scalar:

```python
p = mFlag[(bidx, idx_of(0))] != 0
for i in cutlass.range(n_iter, unroll=1):
    p_next = mFlag[(bidx, idx_of(i + 1))] != 0
    if p:
        ...
    ...
    p = p_next
```

b2 and b3 have the same `d_src` (one loop body each); they differ only in `d_prod`. **Do b2 first**:
it is the smallest change and already removes the term that carries the weight; b3 only adds value when
you also need to move the `ISETP` away from the branch (e.g. when the predicate's comparison chain is
itself long).

**(b4) No branch at all (if-conversion)**: when the optional section is only a few instructions, write it
as predicated arithmetic with a ternary expression / `cutlass.select_`; the `BRA` disappears entirely and
this entry no longer applies. The cost is that the work of both paths is issued — only worth it when the
optional section is **very short**, and ptxas already does this on its own for sufficiently short guarded
sections (see gotcha 5).

**Out-of-bounds budget of the prologue (needed by both b2/b3)**: fetching one extra slot means the last
iteration reads `idx_of(n_iter)`. Either leave one extra slot in the guard-input array, or clamp the index
(`min(i + 1, n_iter - 1)`) — the value fetched for the clamped slot is **not consumed** in that iteration,
so where you clamp does not affect the result, but you **must clamp**, otherwise the last iteration is a
real out-of-bounds access.

---

### (c) Acceptance gates (use all four together)

1. **Pure distance axis**: before and after the change, the mainloop body has **the same number of guard
   BRAs**, and the two ratios `FADD+FFMA / BRA` and `LDG / BRA` differ by ≤ 2%. If not, you changed more
   than the distance, and the comparison that follows is invalid.
2. **The distance really changed**: `d_src` goes from ≤8 to ≥ one loop-body length. (This step must be
   machine-read, not "I wrote it into the previous iteration" — see gotchas 2 and 3.)
3. **Ledger transfer**: in the **same job's** NCU, the family share of those BRAs drops while the family
   share of the `ISETP`/fetch/`BSSY` before the branch rises, by comparable magnitudes. The wait moving
   from the branch to before the branch is exactly the signature of this entry working; if only the BRA
   drops and nothing else rises, you deleted work along the way rather than changing the distance.
4. **Pay-off**: the kernel `t_ns` of the same job drops. **Do not accept on the on-board device
   median** — this entry's gain is usually single-digit percent, while the on-board round-to-round
   spread is often over ten percent, which cannot resolve it and can even read the opposite sign.

---

## Gotchas

1. **Reporting only `d_prod` produces false lesions.** A predicate producer hugging the branch is not the
   problem; its **input** arriving only this iteration is. Report both readings and hang the criterion
   on `d_src` (§(a)-5).

2. **Moving only the `ISETP` away while the fetch stays in this iteration = zero gain.** The latency is
   on the fetch, not on the comparison. After the change you must machine-read `d_src`, because in the
   source diff "the `ISETP` was moved far away" and "the fetch was moved far away" look the same, while
   in time they differ by a whole memory-access latency.

3. **Pin the unrolling before taking readings.** ptxas unrolling multiplies the instruction counts of the
   loop body by a factor, and after unrolling it may also hoist the second copy's fetch before the first
   copy's branch — when the two variants have different unroll factors, the ratios of §(c) gate 1 and the
   median of `d_src` are not comparable. Request no unrolling with `cutlass.range(..., unroll=1)` and
   **check** that the guard BRA counts of both sides are really equal (if not, normalise by count, or use
   the per-BRA distance distribution instead of the median).

4. **`a or b` has no runtime short-circuit in the DSL.** The AST preprocessing expands `and`/`or` into
   `cutlass.and_` / `cutlass.or_`, which short-circuits only when **the left operand is a trace-time
   Python `bool`** (that branch eliminates the whole right side at trace time); when both sides are
   runtime values, **both operands are evaluated**. So the C intuition "put the cheap loop-invariant term
   before the `or` to avoid the expensive runtime term" **only holds here if the cheap term is written as
   a trace-time constant**; written as a runtime `Boolean`, the expensive term is issued anyway and the
   guard's critical path is not shortened at all.

5. **When the guarded section is too short there is no BRA at all.** ptxas if-converts sufficiently
   short guarded sections directly into predicated instructions (`@P0 FADD ...`), and no guard BRA can be
   found in the ledger. This entry then does not apply — and **do not "make the section longer" in order
   to use this entry**: predication is already the better form.

6. **Divergent guards cannot use this entry's readings.** `BSSY`/`BSYNC` near the guard BRA in SASS
   means divergent reconvergence; both paths are taken and the time composition is completely different.
   First confirm the guard is uniform within the warp (the predicate lands in `UP*` or comes from a warp
   invariant), then talk about distance.

7. **Loop-carried values may be rematerialised.** Across `scf.if` / across a `@cute.jit` function
   boundary, the compiler may fail to prove some quantity invariant and recompute it — moving the
   producer you worked hard to move away back next to the branch. After the change, confirm by reading
   SASS, not by reading the source (similar forms in [`wgmma_desc_uniform`](wgmma_desc_uniform.md),
   [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md)).

8. **Don't drop the "fetch one extra slot" prologue.** The prologue fetch of b2/b3 is outside the loop;
   without it the first iteration consumes an uninitialised value — which is often **invisible**
   numerically (the guard takes the optional section one time more/less, and the optional section is
   often idempotent or approximately so), and only occasionally blows up on boundary shapes.

9. **Don't accept on on-board absolute values.** See §(c) gate 4.

---

## Sources

1. Official docs: PTX ISA — Predicated execution / `setp` / `bra` (operand conventions for predicate
   registers and branches); CUDA Binary Utilities (`nvdisasm -c` control-flow + label output,
   `cuobjdump -res-usage`); the Warp Stall Reasons definitions of the Nsight Compute Kernel Profiling
   Guide (the meaning of `long_scoreboard` / `short_scoreboard` / `branch_resolving` / `not_selected`,
   and the accounting convention that a stall is charged to **the instruction being blocked**).
2. Public infra API symbols: `cutlass.range` (the `unroll=` parameter and the loop-carried `write_args`
   of the AST preprocessing), `cutlass.range_constexpr`, `cutlass.Boolean`,
   `cutlass.and_` / `cutlass.or_` / `cutlass.select_`, `cute.compile(..., options=
   "--dump-dir=... --keep-cubin --keep-ptx")`.
3. GPU-certified by this repo's probe (not included in this release)
   (six contracts C1–C6: each of the three variants reproduces the same numpy reference, pure distance
   axis, discriminating power of `d_src`, priced). The verdict is in this entry's status column of
   [`index.md`](index.md).
4. This repo's reading tools: `notes/stall_tools.py` (per-PC ledger / segment ledger),
   `notes/sched_live_tools.py` (back-edge location, register-range decoding), `notes/sass_tools.py`
   (parsing of both disassembly formats).
