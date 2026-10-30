## Verdict experiments (optional, budget {MAX_VERDICT_PROBE_ROUNDS} GPU-job rounds)

When you are unsure between (a)/(b)/(c), do not force a ruling from the shape of the signature — a matching shape signature (regs/smem/grid) does not mean a matching mechanism:
a kernel whose stall composition or wavefront volume is several times the reference's can be perfectly clean in shape. Before ruling you may run
bounded experiments on real hardware: write a self-contained script to `notes/<name>.py` and make that turn end with a single line on its own,
`PROBE_REQUEST: notes/<name>.py` (without writing cause_class); the driver runs it on the GPU and feeds the log readings
back. The rules are the same as for diagnose: pair within one job and read only ratios, one variable per variant, the script itself prints the
`PROBE PASS/FAIL max_diff=<reading>` verdict line, verify contract landing down to SASS, measure only and do not build (do not deliver
a new implementation or an ORACLE implementation body). After the diagnosis the workspace is reset to its pre-probe state, and probe files are not left to the student.
When the hypothesis is "some existing mechanism is currently costing something", you need not write a script: `PROBE_REQUEST: notes/<name>.json`, with the file containing
`{"ablations": [...]}` (same format as the pricing round's `ablations`); the driver uses a frozen recipe to read on the NCU kernel clock
how much faster it is without that mechanism (wrong results allowed; ncu runs first, a hang is marked hung and not touched again, the segment has a total budget); a positive Δ above the noise floor
is an upper bound on that mechanism's cost and can be cited as `reopen_price: <Δ> via ablation:<name>`; a negative Δ = it is helping.
