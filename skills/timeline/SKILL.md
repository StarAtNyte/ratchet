# timeline

When the time leaks, which the per-instruction ledger cannot say (it sums every phase that
touches the same instructions).

- `pm_timeline_tools.py` + `pm_timeline_run.py` -- the PM-sampling timeseries of a
  `.ncu-rep` as downsampled per-series timelines and a phase summary (ramp, plateau,
  drain), between `=== PM TIMELINE BEGIN ===` / `=== PM TIMELINE END ===`, ending in
  `PMTL-OK` or `PMTL-FAIL`. The run script freezes the collection recipe (sampling interval,
  launch count, profiling window): `python3 notes/pm_timeline_run.py`, `--interval 2048`
  for kernels under 0.5 ms.
- `timeline_probe.py` -- a per-segment clock account of the student's kernel on the GPU.
  The student places markers, no-ops in the shipped kernel:

  ```python
  def _seg(site, row=0):          # keep this exact no-op definition
      return None
  ```

  and the driver rewrites them into `%clock64` stamps in the probe only, then prints cycles
  per segment and per role and who waited on whom. The reference is never instrumented.
  The tree's copy must equal the harness copy; the driver says so when it does not.
