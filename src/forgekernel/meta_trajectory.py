"""Structured meta-loop trajectory: meta_status.jsonl — the meta agent's memory.

Where the inner trajectory (``forgekernel.inner.trajectory``) records what the
student did to the kernel, this records what the meta agent did to the *milestone*
(and to the KB): the diagnosis, the patch class + changed entries, and — crucially —
the OUTCOME of the patch measured against the post-patch inner segment. A milestone
patch that produced no attributable improvement is recorded ``REVERT_PATCH`` and
rolled back; the next diagnosis reads this memory so the meta agent does not
re-propose a direction that already regressed (the tile-shrink dead-end problem).

Mirrors the inner trajectory's jsonl + unknown-key-tolerant load. Unlike the inner
trajectory, the last record is UPDATED in place (``update_last_outcome``) because a
patch's outcome is only known after its inner segment runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# outcome lifecycle: PENDING -> {KEEP_PATCH, REVERT_PATCH, HOLD_PATCH} for milestone
# patches — HOLD_PATCH means the post-patch segment produced no metric reading at all
# (pre-green construction only), so the patch is unattributable: nothing is rolled
# back, the transaction stays open, and a later patch joins it for one joint judgment;
# terminal KB_CERTIFIED / KB_UNCERTIFIED for the G3 KB-recipe branch;
# AUTHORED for the initial distillation step (meta writes milestone v1 from the oracle);
# AUDIT_CLEAN for a forced M0 audit that found the milestone consistent with the
# hardware truth (no patch, terminal);
# UNPRICED_REOPEN for an R1 / structural patch refused because it cited no price
# the driver measured (no reopen without a price — nothing patched, terminal);
# REOPEN_CAPPED for an R1 / structural / split patch refused because the run's
# reopen budget is spent ([loop] reopen_max_fruitless reopen transactions in a row
# closed REVERT_PATCH) — nothing patched, terminal;
# INFEASIBLE_STOP for a plateau diagnosis that asked to stop the run (nothing
# patched, terminal — the run ended there; the stop is the run's verdict);
# INFEASIBLE_REFUSED for a stop request the driver's reachability check did not
# accept (not BLOCKED, or the floor does not clear target x 1.20, or too early):
# nothing patched, terminal, the run goes on with a fresh diagnosis.
OUTCOMES = {"PENDING", "KEEP_PATCH", "REVERT_PATCH", "HOLD_PATCH", "KB_CERTIFIED",
            "KB_UNCERTIFIED", "AUTHORED", "AUDIT_CLEAN", "UNPRICED_REOPEN",
            "REOPEN_CAPPED", "INFEASIBLE_STOP", "INFEASIBLE_REFUSED"}


@dataclass
class MetaRecord:
    meta_step: int
    cause_class: str            # G1|G2|G3|G4|R1|R2 from diagnose
    deficient_locus: str        # milestone decision id, or KB:<interface> for G3
    patch_class: str            # additive|structural|split|kb-example
    changed_entries: list[str]  # milestone entry ids the patch touched (feeds prefix reuse)
    patch_direction: str        # one-line direction from the diagnosis (for reflection)
    milestone_before: str = ""  # full milestone text snapshot before the patch (rollback source)
    resume_after: str | None = None
    best_before: float | None = None
    best_after: float | None = None
    outcome: str = "PENDING"
    certified: bool | None = None
    milestone_violations: list[str] = field(default_factory=list)
    # Did the patch agent actually write the milestone? The driver never applies the
    # `patch:` text from the reply, so a patch exists only if the file changed. Storing
    # the answer (and the size of the change) makes a no-op patch visible in the ledger
    # instead of only in the agent transcript.
    patch_landed: bool | None = None
    milestone_delta: str = ""
    # GPU verdict probes the diagnose step ran before committing to this cause_class.
    # 0 also covers records from before the mechanism existed.
    probe_rounds: int = 0
    # What those probes MEASURED, one entry per pre-registered contract line:
    # {"path", "round", "contract", "verdict", "reading", "note"}. The readings
    # used to live only in that step's prompt tail; the next step's memory
    # carried "R1 @ D0.14 -> REVERT" and nothing else, so the teacher re-spent
    # a probe re-measuring the same quantity (on grouped-gemm: the per-K-block
    # exposure count, measured in five consecutive meta steps). Empty for
    # records from before the field existed.
    probe_readings: list[dict] = field(default_factory=list)
    # Per-gate allowed regression the patch declared (`gate_contract:` in the
    # patch output), e.g. {"M0-gate": 0.15}. Empty = same-stage parity at every
    # gate. Judged by the driver at each gate row of the post-patch segment.
    gate_contract: dict = field(default_factory=dict)
    # The lever pricing table this step's diagnosis read (plateau diagnosis and
    # M0 audit alike): one dict per lever {"kind":"lever","name","add_pct",
    # "remove_pct","cls",...} plus one {"kind":"stack",...}. The next step's
    # memory shows it and its citations verify against it, so a lever priced
    # here is not re-priced there. Empty for pre-mechanism records.
    lever_prices: list[dict] = field(default_factory=list)
    # The `reopen_price:` citation an R1 / structural patch carried (`<delta> via
    # probe:<path>#<n> | lever:<name> | KB:<entry>`), verbatim. Empty when the
    # step reopened nothing, or for records from before the rule existed.
    reopen_price: str = ""
    # The `phase_verdict:` the diagnosis gave when the price table read the
    # current phase's levers as exhausted (`close` / `continue:<D id>`), or the
    # driver's default when the teacher left the question unanswered
    # (`close (default: unanswered)`). Empty when no phase was read exhausted.
    phase_verdict: str = ""
    # INFEASIBLE stop requests the driver refused during this step (the reason
    # put back to the teacher, one per refusal), whatever class the step ended
    # on. Empty when none was made, or for records from before the gate.
    infeasible_refusals: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError(f"invalid outcome: {self.outcome!r}")


_FIELDS = {f.name for f in fields(MetaRecord)}


def append_meta_record(path: str | Path, rec: MetaRecord) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def load_meta_records(path: str | Path) -> list[MetaRecord]:
    """Load records, dropping any unknown keys (tolerates schema evolution)."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            out.append(MetaRecord(**{k: v for k, v in d.items() if k in _FIELDS}))
    return out


def update_last_outcome(path: str | Path, *, outcome: str,
                        best_after: float | None = None) -> None:
    """Rewrite the last PENDING record's outcome (a patch is judged after its inner
    segment runs). When none is PENDING the last HOLD_PATCH row is the open
    transaction (a held construction judged at its later stop); only when there is
    neither does it fall back to the last record.

    Targeting the open row rather than the last row matters once M0 audits
    interleave with patches: an AUDIT_CLEAN or UNPRICED_REOPEN row appended after
    an open patch must not absorb that patch's judgment.

    The whole file is small (one line per meta step), so a load->mutate->rewrite is
    cheaper than any append-a-terminal-event scheme and keeps one row per meta step.
    """
    recs = load_meta_records(path)
    if not recs:
        raise ValueError(f"no meta records to update at {path}")
    idx = next((i for i in range(len(recs) - 1, -1, -1)
                if recs[i].outcome == "PENDING"), None)
    if idx is None:
        idx = next((i for i in range(len(recs) - 1, -1, -1)
                    if recs[i].outcome == "HOLD_PATCH"), len(recs) - 1)
    recs[idx].outcome = outcome
    if best_after is not None:
        recs[idx].best_after = best_after
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def reflect_summary(records: list[MetaRecord]) -> str:
    """A compact reflection block injected into the diagnose prompt.

    Marks REVERT_PATCH directions as forbidden (do not re-propose) and lists
    certified KB interfaces (do not re-certify). Objective memory, one line each.
    """
    if not records:
        return "(No earlier meta steps: this is the first diagnosis.)"
    lines = ["Below are your past meta steps (objective memory: reflect on them, and do not "
             "repeat a direction that already failed). A step with \"measured\" sub-rows carries "
             "the per-contract readings of that step's verdict probe — **quote quantities already "
             "measured directly; do not spend probes re-measuring them**; keep the probe slots for "
             "hypotheses not yet verified:"]
    for r in records:
        if r.outcome == "REVERT_PATCH":
            lines.append(f"- [dead end] step{r.meta_step} {r.cause_class} @ {r.deficient_locus}: "
                         f"\"{r.patch_direction}\" → REVERT_PATCH (no improvement, rolled back; "
                         f"do not repeat this direction)")
        elif r.outcome == "KEEP_PATCH":
            lines.append(f"- [effective] step{r.meta_step} {r.cause_class} @ {r.deficient_locus}: "
                         f"\"{r.patch_direction}\" → KEEP_PATCH")
        elif r.outcome == "KB_CERTIFIED":
            lines.append(f"- [KB certified] step{r.meta_step} {r.deficient_locus}: "
                         f"interface certified into the KB; do not certify it again")
        elif r.outcome == "KB_UNCERTIFIED":
            lines.append(f"- [KB not certified] step{r.meta_step} {r.deficient_locus}: "
                         f"\"{r.patch_direction}\" → certification failed")
        elif r.outcome == "AUDIT_CLEAN":
            lines.append(f"- [M0 audit clean] step{r.meta_step}: reconciling the M0 signature "
                         f"against the reference found no milestone defect (this segment's M0 has "
                         f"been audited; do not audit the same segment again)")
        elif r.outcome == "UNPRICED_REOPEN":
            lines.append(f"- [unpriced reopen refused] step{r.meta_step} {r.cause_class} @ "
                         f"{r.deficient_locus}: \"{r.patch_direction}\" → unpriced; the driver "
                         f"refused the reopen and left the milestone unchanged. To propose this "
                         f"reopen again, first build the minimal variant on the best tree with a "
                         f"verdict probe and measure it in the same job, then cite the reading in "
                         f"the diagnosis with `reopen_price:` (or cite the pricing table / a "
                         f"certified KB entry)")
        elif r.outcome == "REOPEN_CAPPED":
            lines.append(f"- [reopen budget exhausted, refused] step{r.meta_step} {r.cause_class} @ "
                         f"{r.deficient_locus}: \"{r.patch_direction}\" → consecutive reopens "
                         f"brought no new best; the driver refused to reopen a delivered decision "
                         f"again and left the milestone unchanged (the budget is restored after a "
                         f"KEEP_PATCH)")
        elif r.outcome == "INFEASIBLE_STOP":
            # A fact about the trajectory, nothing more: no word about what a
            # human thought of the stop and no pointer to human-written
            # direction — either would feed a human's meta decision back into
            # the run under test.
            lines.append(f"- [stop request] step{r.meta_step} INFEASIBLE @ {r.deficient_locus}: "
                         f"\"{r.patch_direction}\" → the loop stopped at this step; milestone "
                         f"unchanged")
        elif r.outcome == "INFEASIBLE_REFUSED":
            lines.append(f"- [stop request refused] step{r.meta_step} INFEASIBLE @ "
                         f"{r.deficient_locus}: \"{r.patch_direction}\" → failed the reachability "
                         f"check; the loop continued, milestone unchanged")
        else:
            lines.append(f"- [in progress] step{r.meta_step} {r.cause_class} @ {r.deficient_locus}: "
                         f"\"{r.patch_direction}\"")
        lines.extend(_reading_lines(r))
        for why in (getattr(r, "infeasible_refusals", None) or []):
            lines.append(f"  · INFEASIBLE refused: {why}")
        if getattr(r, "phase_verdict", ""):
            lines.append(f"  · phase ruling {r.phase_verdict} (the pricing table reads every lever "
                         f"of this phase as null)")
        for lv in (getattr(r, "lever_prices", None) or [])[:_MAX_READING_LINES]:
            if lv.get("kind") == "stack":
                lines.append(f"  · pricing table all-on={_pct(lv.get('all_pct'))} "
                             f"sum-of-add-ones={_pct(lv.get('sum_singles_pct'))} "
                             f"coupling={_pct(lv.get('coupling_pp'))}")
            else:
                lines.append(f"  · priced {lv.get('name', '?')} add={_pct(lv.get('add_pct'))} "
                             f"remove={_pct(lv.get('remove_pct'))} [{lv.get('cls', '?')}]")
    return "\n".join(lines)


_MAX_READING_LINES = 8
_MAX_NOTE = 110


def _pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def _reading_lines(r: "MetaRecord") -> list[str]:
    """The probe readings of one step, as sub-rows of its memory line."""
    out = []
    for d in (r.probe_readings or [])[:_MAX_READING_LINES]:
        note = str(d.get("note") or "")
        note = note if len(note) <= _MAX_NOTE else note[:_MAX_NOTE - 1] + "…"
        out.append(f"  · measured {d.get('path', '?')} #{d.get('contract', '?')} "
                   f"{d.get('verdict', '?')} max_diff={d.get('reading', 'unreported')}"
                   + (f" — {note}" if note else ""))
    extra = len(r.probe_readings or []) - _MAX_READING_LINES
    if extra > 0:
        out.append(f"  · ({extra} more contract readings in meta_status)")
    return out
