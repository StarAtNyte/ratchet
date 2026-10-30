# Static SASS verification of the MMA operand source (RS vs SS; how to write `[VERIFY: sass-derived]`)

**Reading recipe**: turn a `[SOURCE: RS]` ruling row of the ledger into a machine-checkable gate line.
No dynamic pricing (per-SASS counters) is needed; one static disassembly census is enough.

## Why this exists

An observed failure mode: the milestone rules the A operand of some GEMM chain RS (the accumulator
fragment stays in registers and feeds the next chain directly), while the M0 gate only checks the
`regs_per_thread` range. The student's register count happens to land on the producer/consumer blended
mean — **the budget is fully met** — yet the operand still goes through smem (wavefronts on both the
write and the read side each ~1.9× higher, see [[smem_traffic_triage]]). "Budget reserved" ≠ "path
built", and the latter is recognisable at a glance in SASS:

## How to recognise it in SASS

The operand order of the SM90 wgmma instruction family (`HGMMA`/`QGMMA`/`IGMMA`/`BGMMA`) is
`acc, A, B, ...`:

- **SS form (A via smem)**: the A slot is a matrix descriptor — `HGMMA.64x64x16.F32 R80, gdesc[UR4], gdesc[UR6], ...`
- **RS form (A from registers)**: the A slot is a register — `HGMMA.64x64x16.F32 R80, R8, gdesc[UR6], ...`

(The B slot is always a descriptor; in both dump styles, cuobjdump and the NCU source page, the
descriptor is spelled `desc[...]`/`gdesc[...]`.)

## Tool and output line

`src/forgekernel/inner/sass_census.py` (the driver vendors it into recon `notes/` together with
sass_tools): counts the A-slot form over the dominant kernel's SASS dump and prints one machine-read
verdict line:

```
SASS-OPERANDS kernel=<name> mma_total=N a_reg=N a_desc=N a_other=N
```

After the operator's ncu task runs the census, the driver merges it into the profiler digest:
`mma_total` / `mma_a_reg` / `mma_a_desc` / `mma_a_other` / `mma_census_kernel`.

## How to write the gate line (template A.5 "operand-source verification line")

```
- [ ] <chain name> A operand RS built: `mma_a_reg ≥ <chain beat count>` (the chain's HGMMA beats per tile,
      all of which must be REG-A form) [VERIFY: sass-derived], against the matching [SOURCE: RS] row of the ledger
- [ ] main kernel `spill_ld_sectors + spill_st_sectors == 0` (RS's register cost must not spill)
      [VERIFY: ncu-direct], paired with the line above
```

Three rules:

1. **Threshold = the beat count of that RS chain** (derived from the ledger's chain table); do not
   write ">0" — one lucky REG-A cannot vouch for the whole chain.
2. **Must be paired with the spill line**: RS trades register pressure for smem traffic, and "RS built
   + spill" is worse than not building it (an observed failure mode: only the feed path was built,
   not the register economy; spill went from 0 to tens of millions of sectors and the whole kernel
   got slower). Census green + spill red = the register account does not close → re-rule the row
   `[SOURCE: smem]` and record the reason in the ledger (the template takes no position on RS).
3. **Before trusting it, check that `mma_census_kernel` matches the digest's dominant kernel** — the
   census line shares the pipe with other stdout, and a census whose kernel name does not match
   proves nothing (same family of failure modes: a whole NCU round sampled the wrong kernel without
   anyone noticing, an anti-cheat regex matched itself and produced a false positive — verify the
   source of every machine-read line first).
