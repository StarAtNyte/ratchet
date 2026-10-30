# stall

Where the time actually leaks: the measured PC-sampling stall counters of a `.ncu-rep`
joined per SASS instruction into a ledger.

- `stall_tools.py` -- the ledger. Rows are printed until they cover at least 90% of all
  stall samples, between `=== STALL LEDGER BEGIN ===` / `=== STALL LEDGER END ===`, and
  the run ends in exactly one `STALL-OK ...` or `STALL-FAIL reason=...` line.
- `stall_price_run.py` -- the frozen collection recipe the driver runs on the best tree at
  a plateau diagnosis (`python3 notes/stall_price_run.py`, with `--pm` to collect the
  PM-sampling timeline in the same job). Not a student tool: a student asks for a probe
  with `PROBE_REQUEST` instead.

Read the ledger before naming a bottleneck: a locus the ledger prices at 1% is not the
bottleneck however plausible it reads.
