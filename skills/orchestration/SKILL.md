# orchestration

Sync topology and supply organisation read off the SASS page, and who leaves the tensor pipe
idle.

- `orch_tools.py` -- the orchestration dossier: regions and loops labelled by role
  (copy producer, MMA consumer, scalar, spin), every mbarrier and named-barrier site with
  its address, in-flight depth, supply order, cache-policy bits. It is what a hand
  transcription of thousands of SASS lines keeps dropping (one barrier per stage versus two
  per half block reads the same by eye).
- `orch_census_run.py` -- the same dossier summary (`ORCH-SUMMARY-JSON`) on every kept
  iteration, riding the NCU job that already runs for it, so a kept change can be tied to
  the contract axis it moved.
- `role_exposure_tools.py` -- the three reads joined: stall samples attributed to the
  dossier's regions, aggregated per role and scaled by the non-tensor fraction of the
  kernel into an exposure estimate per role, in ns -- the same table for the student and
  the reference, side by side.

The driver runs these with frozen recipes; a probe imports them bare from `notes/`.
