"""Deterministic parsers for the two-layer driver.

The driver shells out to an agent for each student/meta turn and to the GPU
harness for each eval/probe. These pure functions turn that free-form text into
typed decisions the orchestration core can branch on — so the loop logic is
unit-testable without an agent or a GPU.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple
from dataclasses import dataclass, field

from forgekernel.inner.trajectory import GATE_WINDOW_DEFAULT, gate_ms


@dataclass
class Record:
    """A student's completed decision — OBJECTIVE fields only (the technique + file).

    The student's subjective analysis (hypothesis, bottleneck theory) is not
    parsed / not recorded; the meta agent reasons from measured NCU truth instead.
    """
    decision_id: str
    change: str
    files_touched: str = ""


@dataclass
class ProbeRequest:
    """The student wants a GPU layout-probe run before committing the decision."""
    path: str


@dataclass
class ReconRequest:
    """The teacher wants a reconnaissance script run against the reference.

    Distinct from ProbeRequest because it runs in a different tree and answers a
    different question. A probe validates the student's own code; a recon script
    measures the *reference* — profiling it, disassembling it, or computing against
    hardware limits — and it runs in the teacher-only submission tree so that neither
    the script nor what it learns can reach a student job.
    """
    path: str


@dataclass
class Exhausted:
    """No further phase-order decision is available in the milestone."""
    reason: str


StudentTurn = Record | ProbeRequest | ReconRequest | Exhausted

_FIELD = re.compile(r"^\s*([a-z_]+)\s*:\s*(.*\S)\s*$")


_PATH_LWRAP = "`'\"*([<"
_PATH_RWRAP = "`'\"*)]>,;:."


def _clean_path(tok: str) -> str:
    """A marker's path with the markdown an agent wraps it in stripped: the
    ``\\S+`` capture keeps a closing backtick / quote / paren, and that one
    character reaches the GPU machine's ``bash -c`` as an unmatched quote — the job
    dies before its begin marker, which the ssh runner reads as a transport
    failure, for a path that never existed."""
    return tok.lstrip(_PATH_LWRAP).rstrip(_PATH_RWRAP)


def parse_student_turn(text: str, *, allow_recon: bool = False) -> StudentTurn:
    """Classify a student agent's final message into the three terminal shapes.

    PROBE_REQUEST / EXHAUSTED take precedence (they are explicit single-line
    markers); otherwise the record block fields are extracted from anywhere in
    the text (agents often emit prose before the block).

    RECON_REQUEST is only a valid shape for the reconnaissance caller
    (``allow_recon=True``). Recognizing it everywhere let a student turn that
    merely echoed the marker — likely in a blackbox run, where the vocabulary
    saturates the prompts — reach the probe branch and burn a GPU job on a
    path that exists only in the teacher's tree.
    """
    probe = re.search(r"PROBE_REQUEST:\s*(\S+)", text)
    if probe:
        return ProbeRequest(path=_clean_path(probe.group(1)))
    if allow_recon:
        recon = re.search(r"RECON_REQUEST:\s*(\S+)", text)
        if recon:
            return ReconRequest(path=_clean_path(recon.group(1)))
    exhausted = re.search(r"EXHAUSTED:\s*(.+)", text)
    if exhausted:
        return Exhausted(reason=exhausted.group(1).strip())

    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = _FIELD.match(line)
        if m and m.group(1) in ("decision_id", "change", "files_touched"):
            fields.setdefault(m.group(1), m.group(2))
    if "decision_id" not in fields:
        raise ValueError("student turn is neither a record, a probe request, nor exhausted")
    return Record(
        decision_id=fields["decision_id"],
        change=fields.get("change", ""),
        files_touched=fields.get("files_touched", ""),
    )


@dataclass
class EvalResult:
    correctness: bool
    bwd_median_ms: float | None = None
    mfu: float | None = None
    # In-job reading of THE declared reference, when the run's ref can ride the eval
    # job (single-ref contract — see IterationRecord in inner/trajectory.py). Absolute
    # latency for an unchanged commit moves ~6% between jobs — wider than most
    # single-round gains — so the absolute is partly a reading of which node the round
    # landed on; the same-job ratio holds across that swing. None when the ref cannot
    # ride the job (blackbox runs time only the student — no stand-in library).
    ref_median_ms: float | None = None
    vs_ref: float | None = None
    # Second timing window, when the harness measures one. bwd_median_ms is a per-call
    # window (record/call/record/sync each rep): the stream starts idle, so host work —
    # launch latency, python wrappers — lands inside it. device_median_ms is a pipelined
    # batch window (one record pair around N back-to-back calls, divided by N): the
    # stream never drains, host cost is hidden, and what remains is ≈ device time.
    # Observed live: the same cuDNN kernel reads 0.28 ms in the first window and 0.25 ms
    # in the second, and the two sides of a ratio can pay *different* host taxes (a
    # python ctx manager on the library side vs a thin pointer path on the student's) —
    # so an e2e ratio is not a kernel-quality ratio. The gate stays on the e2e window;
    # this pair exists so kernel-vs-kernel claims can be read from matching windows.
    device_median_ms: float | None = None
    ref_device_ms: float | None = None
    vs_ref_device: float | None = None
    # The device window's PRECONDITION: it equals device time only while the host can
    # enqueue faster than the device drains. host_enqueue_ms times the host alone (N
    # calls, no synchronize in between); host_bound_ratio is that over the device
    # reading. Near 1 means the pipeline never filled and the "device" number is
    # python dispatch cost. Parsed and recorded rather than left in the job log —
    # a criterion that never reaches the trajectory cannot gate anything and cannot
    # be fed back to the student, which is the same as not having written it.
    host_enqueue_ms: float | None = None
    host_bound_ratio: float | None = None
    # Same check for the in-job reference, when the bench times one. A host-bound
    # REFERENCE is a measurement defect rather than a kernel fact, and it invalidates
    # vs_ref_device in the reference's favour-losing direction.
    ref_host_bound_ratio: float | None = None
    # The basis the device reading was taken on (timing.py's device_window_basis:
    # plain_batch / cuda_graph / cupti_kernel_span). Decides whether the enqueue
    # clock describes the reading at all — under a host-free basis it does not, and
    # the host-bound audit must not divide the two (on dsa_sparse: three rounds of
    # "your reading is inflated" against a CUPTI span that cannot contain host time).
    device_window_basis: str | None = None
    # What one call of the in-job reference costs to enqueue, when the bench timed
    # it: the comparison a thick student wrapper is reported against (as a
    # deployment cost, never as a gate) under a host-free basis.
    ref_host_enqueue_ms: float | None = None
    # The bench's own words for a correctness fail (forbidden-op verdict, bad-shape
    # detail, import traceback). A bare "correctness FAIL" sent four agents in a row
    # chasing phantom numeric bugs while the real verdict — a scanner match on the
    # entry symbol — sat unread in the job log; the student and the meta diagnoser
    # can only be as right as the failure text they are shown.
    error: str = ""
    # The bench's window-outside prep step, when the timed call receives
    # symmetrized inputs (timing.measure_windows(fn, prep=...)): ``outputs`` =
    # {name: {shape, dtype}} of what it produced, plus its own median_ms /
    # device_median_ms. The driver holds ``outputs`` to window.toml
    # (forgekernel.window_contract.check_prep_report) and banks the block on the
    # iteration record. None when the bench ran no prep step.
    prep: dict | None = None
    # Multi-workload cards (timing.measure_windows_workloads): the top-level
    # readings above are the geometric mean over ``n_workloads`` workloads of one
    # fixed shape, and ``workloads`` keeps the per-workload readings compact —
    # [{name, device_median_ms, bwd_median_ms, ref_device_ms?, vs_ref_device?,
    # vs_ref?}] — for the log line, the teacher's digest and the confirmation
    # record. Empty and 1 for a single-input bench.
    n_workloads: int = 1
    workloads: list = field(default_factory=list)
    # The eval JSON's top-level ``precision`` block (src/forgekernel/card_template/eval/precision.py:
    # kernel vs reference error against a float64 truth). The bench never folds it into
    # ``correctness``; the driver applies the card's [loop] precision_contract
    # (loop/precision_gate.py). None = the bench reported none.
    precision: dict | None = None


_WORKLOAD_KEYS = ("device_median_ms", "bwd_median_ms", "ref_device_ms", "vs_ref_device", "vs_ref")


def _workloads(perf: dict) -> list:
    """Compact per-workload readings from a multi-workload performance block."""
    out = []
    for w in perf.get("workloads") or ():
        if not isinstance(w, dict):
            continue
        item = {"name": str(w.get("name", len(out)))}
        for key in _WORKLOAD_KEYS:
            v = w.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                item[key] = float(v)
        out.append(item)
    return out


def describe_workload_readings(workloads: list) -> str:
    """``name device-ms (xratio)`` per workload, for one log / digest line."""
    parts = []
    for w in workloads or ():
        ms = w.get("device_median_ms", w.get("bwd_median_ms"))
        text = f"{w.get('name')} {ms:.4f} ms" if isinstance(ms, (int, float)) else f"{w.get('name')} n/a"
        ratio = w.get("vs_ref_device", w.get("vs_ref"))
        if isinstance(ratio, (int, float)):
            text += f" (x{ratio:.4f})"
        parts.append(text)
    return "; ".join(parts)


def _last_json_object(text: str) -> dict | None:
    """The last JSON object carrying a 'correctness' or 'performance' key."""
    dec = json.JSONDecoder()
    found = None
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            try:
                obj, j = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict) and ("correctness" in obj or "performance" in obj):
                found = obj
            i = j
        else:
            i += 1
    return found


def _correctness_error(corr_obj: dict) -> str:
    """The bench's failure verdict, verbatim when it wrote one.

    Harness-level rejections (forbidden-op scan, import failure, bad return shape)
    arrive as a single ``error`` string; numeric fails arrive as per-tensor detail
    dicts instead, so summarise those into one line.
    """
    if err := str(corr_obj.get("error") or "").strip():
        return err
    parts = []
    for name in ("dq", "dk", "dv"):
        det = corr_obj.get(name)
        if not isinstance(det, dict):
            continue
        if det_err := det.get("error"):
            parts.append(f"{name}: {det_err}")
        elif "max_diff" in det:
            parts.append(f"{name}: max_diff={det['max_diff']} tol={det.get('tolerance')}")
    return "; ".join(parts) if parts else "no failure detail in the eval JSON"


def parse_eval(text: str) -> EvalResult:
    """Parse the gpu_entry ``=== JSON ===`` eval block (correctness + perf)."""
    obj = _last_json_object(text)
    if not obj:
        from forgekernel.submit.jobs import clock_lock_refusal
        if (why := clock_lock_refusal(text)) is not None:
            # RealBackend.submit_gpu never returns such a log; another submitter
            # might — say what happened instead of "the job died"
            return EvalResult(correctness=False,
                              error=f"the job timed nothing: the board's clock lock did not "
                                    f"hold on its node (a node fault, not the kernel): "
                                    f"{why[:300]}")
        return EvalResult(correctness=False,
                          error="no eval JSON block in the job log (job died before the verdict?)")
    if scope := obj.get("workload_scope"):
        # a group card timed on part of its group (FK_LB_WORKLOAD_SCOPE, the seeded
        # meta tools' setting) — a pricing reading, never a gate: the gate is the
        # geomean over every workload of the group
        return EvalResult(correctness=False,
                          error=f"eval JSON carries workload_scope={json.dumps(scope)}: a group card "
                                f"timed on part of its group is a meta reading, not a gate")
    corr_obj = obj.get("correctness", {}) or {}
    correct = bool(corr_obj.get("pass", False))
    perf = obj.get("performance") or {}
    ms = perf.get("bwd_median_ms")
    mfu = perf.get("bwd_mfu_pct", perf.get("mfu"))

    def _opt(key: str, legacy: str | None = None) -> float | None:
        # legacy: pre-rename bench key (cublas_* — the in-job reference was recorded
        # under that name whatever library it was). Read so older cards' bench.py
        # keeps parsing; new benches emit the ref_* names only.
        v = perf.get(key, perf.get(legacy) if legacy else None)
        return float(v) if v is not None else None

    def _opt_any(*keys: str) -> float | None:
        for key in keys:
            v = _opt(key)
            if v is not None:
                return v
        return None

    # The prep block rides inside performance (timing.py's contract); a bench
    # that placed it at the top level is read too.
    prep = perf.get("prep") if isinstance(perf.get("prep"), dict) else obj.get("prep")
    if not isinstance(prep, dict):
        prep = None

    workloads = _workloads(perf)
    n_wl = perf.get("n_workloads")
    n_wl = int(n_wl) if isinstance(n_wl, int) and not isinstance(n_wl, bool) and n_wl > 0 \
        else max(1, len(workloads))

    return EvalResult(
        prep=prep,
        precision=obj.get("precision") if isinstance(obj.get("precision"), dict) else None,
        n_workloads=n_wl,
        workloads=workloads,
        correctness=correct,
        error="" if correct else _correctness_error(corr_obj),
        bwd_median_ms=float(ms) if ms is not None else None,
        mfu=float(mfu) if mfu is not None else None,
        ref_median_ms=_opt("ref_median_ms", "cublas_median_ms"),
        vs_ref=_opt("vs_ref", "vs_cublas"),
        device_median_ms=_opt("device_median_ms"),
        ref_device_ms=_opt("ref_device_ms", "cublas_device_ms"),
        vs_ref_device=_opt("vs_ref_device", "vs_cublas_device"),
        host_enqueue_ms=_opt_any("host_enqueue_ms", "enqueue_ms"),
        host_bound_ratio=_opt("host_bound_ratio"),
        ref_host_bound_ratio=_opt_any("ref_host_bound_ratio", "cublas_host_bound_ratio"),
        device_window_basis=(str(perf["device_window_basis"])
                             if perf.get("device_window_basis") else None),
        ref_host_enqueue_ms=_opt_any("ref_host_enqueue_ms", "cublas_host_enqueue_ms"),
    )


PROBE_MAX_DIFF_CEILING = 1e3
"""Largest ``max_diff`` the driver will accept behind a PASS.

Not a correctness tolerance — the driver cannot know a probe's tensor scale, so
the real bound stays where it can be computed (``eval/bench.py``, relative to
``|ref32|_max``). This is a plausibility floor under that: for fp16 gradients an
error norm above 1e3 means the kernel is catastrophically wrong, so a PASS above
it is a FAIL either way. What it actually catches is the number not being an
error norm at all — probes have reported values like 8e10 and 8.1e15, smuggling
sub-check bitmasks through the one field the driver echoes back.
"""

_PROBE_VERDICT_RE = re.compile(r"PROBE\s+(PASS|FAIL)(?:\s+max_diff=(\S+))?([^\n]*)")
_PROBE_NON_NUMERIC = {"n/a", "na", "none", "-"}
_PROBE_NOTE_LEAD = " \t#-/:;|"
_PROBE_READING_MAX = 96

# The operator entry script's closing line (gpu_entry.sh prints it after the
# task, whatever the task was). Both spellings of the tag are in the field.
ENTRY_DONE_RE = re.compile(r"^\[gpu_entry(?:\.sh)?\] task=(?P<task>\S+) finished", re.M)

# Any line the entry script itself printed: proof the log is a job log (as opposed
# to bare verdict text), so its closing line can be demanded.
_ENTRY_LINE_RE = re.compile(r"^\[gpu_entry(?:\.sh)?\] ", re.M)

# The entry script's own bookends, walked as a stack: `timeout=` opens a run,
# `task=... finished` closes it cleanly, `KILLED:` closes it dead. A cmd job may
# nest entry runs (orch_census streams `bash gpu_entry.sh ncu`; a KB/teacher probe
# shells out the same way, and a ladder rung may run an inner entry under its own
# cap), so a kill line mid-log can be an inner rung the probe handled, and a
# `finished` line mid-log can be an inner run while the outer one is still open.
_ENTRY_EVENT_RE = re.compile(
    r"^\[gpu_entry(?:\.sh)?\] (?:(?P<open>timeout=)|(?P<done>task=\S+ finished)|(?P<killed>KILLED:))",
    re.M)

# Kill evidence written by the layers around the entry script, anchored to the
# spelling those layers use (a probe quoting a child's wrapper line on the FACT
# channel is not the wrapper): bash's report of the SIGKILL from
# `timeout --signal=KILL` (carries no code), the runner's closing line (written
# only for a non-zero rc) and its run-timeout line (forgekernel.submit.direct).
_JOB_ABEND_RES = (
    re.compile(r"gpu_entry\.sh: line \d+:\s+\d+ Killed\b"),
    re.compile(r"^\[direct\] gpu_entry .+? exited rc=(?P<rc>[1-9]\d*)", re.M),  # mode may carry spaces
    re.compile(r"^\[direct\] gpu_entry .+? exceeded \S+ -- killed", re.M),
)

JOB_END_WHOLE, JOB_END_KILLED, JOB_END_CUT = "", "killed", "cut"


class JobEnd(NamedTuple):
    """How a job log ends: ``kind`` is ``JOB_END_WHOLE`` (``""``: the job ran to
    completion, or the text is not a job log at all), ``JOB_END_KILLED`` (the log
    carries kill evidence) or ``JOB_END_CUT`` (the outer entry run never closed
    and nothing says why: the job died silently, or the archive is cut).
    ``reason`` is the sentence for a log line / feedback; ``""`` when whole."""
    kind: str
    reason: str = ""


def job_abend(log: str) -> JobEnd:
    """Whether ``log`` says the job ran to completion, and if not, why.

    A PROBE line is a claim about the rung that printed it. The rungs after it
    (a launch that hangs, a relaunch, a teardown) print nothing when the job is
    killed under them, and a log that ends in a kill is not a log of a job that
    ran to completion — whatever it printed before. Bare verdict text with no
    entry-script line at all (tests, hand-run probes) is left as read.

    Entry runs are walked as a stack (see ``_ENTRY_EVENT_RE``): the job ran to
    completion when every run that opened has closed and the last one closed
    with its ``finished`` line — an inner run killed under its own cap and
    handled by the probe does not veto an outer run that finished, and an inner
    ``finished`` line does not stand in for an outer run that is still open. Only
    an open outer run consults the wrapper layers' kill lines.
    """
    if not _ENTRY_LINE_RE.search(log):
        return JobEnd(JOB_END_WHOLE)
    depth, last = 0, None
    for m in _ENTRY_EVENT_RE.finditer(log):
        if m["open"]:
            depth += 1
        else:
            depth = max(depth - 1, 0)
            last = "done" if m["done"] else "killed"
    if depth == 0 and last == "done":
        return JobEnd(JOB_END_WHOLE)
    hits = [m for m in (rx.search(log) for rx in _JOB_ABEND_RES) if m]
    if last == "killed" or hits:
        # bash's "Killed" line carries no code; the wrapper lines after it do
        rc = next((m.group("rc") for m in hits if m.groupdict().get("rc")), None)
        return JobEnd(JOB_END_KILLED,
                      f"the job was killed (exit {rc})" if rc else "the job was killed")
    return JobEnd(JOB_END_CUT, (
        "the entry script never printed its closing line and the log shows no kill — "
        "the archive is cut (the log archiver lags the job) or the job died silently"))


def _as_finite_float(raw: str | None) -> float | None:
    """The token as an error norm / reading, or None when it is not a finite number."""
    if raw is None:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _clip(text: str, limit: int = _PROBE_READING_MAX) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(frozen=True)
class ProbeContract:
    """One ``PROBE PASS/FAIL`` line of a probe job log, as the probe printed it.

    A probe is allowed (and told) to print one line per contract it checks. Those
    lines used to be collapsed into a single bit plus the first float-parseable
    ``max_diff`` — so a five-contract experiment whose decisive contract PASSed
    with a reading table and whose control contract FAILed on a tooling hiccup
    surfaced as ``PROBE FAIL (max_diff=unreported)``. Each line is now kept.
    """
    index: int             # 1-based, log order
    verdict: str           # "PASS" | "FAIL"
    raw: str | None        # the max_diff token exactly as printed; None when absent
    note: str = ""         # trailing text on the line (contract name / reason)

    @property
    def diff(self) -> float | None:
        return _as_finite_float(self.raw)

    @property
    def reading(self) -> str:
        """What the line reported, for logs: a number, ``n/a``, a reading string,
        or ``unreported`` when the line carried no max_diff at all."""
        return "unreported" if self.raw is None else _clip(self.raw)

    def describe(self) -> str:
        head = f"#{self.index} {self.verdict} max_diff={self.reading}"
        return f"{head} — {_clip(self.note, 120)}" if self.note else head


def _contracts(text: str) -> tuple[ProbeContract, ...]:
    out = []
    for i, m in enumerate(_PROBE_VERDICT_RE.finditer(text), start=1):
        note = m.group(3).strip().lstrip(_PROBE_NOTE_LEAD).strip()
        out.append(ProbeContract(i, m.group(1), m.group(2) or None, note))
    return tuple(out)


@dataclass(frozen=True)
class ProbeVerdict:
    """A probe's GPU verdict, after the driver has audited the probe's own claim.

    ``passed`` is the driver's judgement, not the probe's: a self-reported PASS is
    downgraded when the accompanying ``max_diff`` cannot be an error norm. ``found``
    separates "the probe ran and said FAIL" from "no verdict line at all" — those
    reached the student as the same message before, so a probe that never built
    was fed back as a numerical failure.

    ``contracts`` keeps every verdict line the probe printed; ``reading`` keeps the
    representative ``max_diff`` token when it was not a number (``n/a``, or a
    ``k=v,...`` reading string a verdict experiment packed into the field), so a
    line that reported something is never logged as ``unreported``.
    """
    passed: bool
    max_diff: float | None = None
    numeric: bool = True
    found: bool = True
    reason: str = ""
    contracts: tuple[ProbeContract, ...] = ()
    reading: str | None = None

    @property
    def diff_text(self) -> str:
        """How the diff should read in a log line / in feedback to the agent."""
        if self.max_diff is not None:
            return f"{self.max_diff}"
        if self.reading:
            return _clip(self.reading)
        return "n/a (non-numeric probe)" if not self.numeric else "unreported"

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.contracts if c.verdict == "PASS")

    @property
    def summary(self) -> str:
        """Per-contract tally for multi-line probes: ``2/4 PASS; FAIL #1 max_diff=n/a
        — <note>, #3 ...``. Empty for a single-line probe (the diff_text says it all)."""
        if len(self.contracts) < 2:
            return ""
        fails = [c for c in self.contracts if c.verdict == "FAIL"]
        text = f"{self.n_pass}/{len(self.contracts)} PASS"
        if fails:
            text += "; FAIL " + ", ".join(c.describe() for c in fails)
        return text

    def contract_table(self) -> str:
        """One line per contract, for feedback blocks."""
        return "\n".join(c.describe() for c in self.contracts)


def _audit_pass(raw: str | None) -> ProbeVerdict:
    """Decide whether one self-reported PASS survives the driver's audit."""
    if raw is None:
        return ProbeVerdict(False, reason=(
            "reported PASS with no max_diff=... — a PASS must declare what it "
            "verified: a numeric diff, or max_diff=n/a for a non-numeric contract"))
    if raw.strip().lower() in _PROBE_NON_NUMERIC:
        return ProbeVerdict(True, max_diff=None, numeric=False)
    try:
        diff = float(raw)
    except ValueError:
        return ProbeVerdict(False, reason=(
            f"reported PASS with an unparseable max_diff={raw!r} — report a number, "
            f"or max_diff=n/a for a non-numeric contract"))
    if diff != diff or diff in (float("inf"), float("-inf")):  # NaN / inf
        return ProbeVerdict(False, reason=(
            f"reported PASS with max_diff={raw} — a non-finite diff is a failure"))
    if abs(diff) > PROBE_MAX_DIFF_CEILING:
        return ProbeVerdict(False, reason=(
            f"reported PASS with max_diff={diff}, above the {PROBE_MAX_DIFF_CEILING:g} "
            f"ceiling — that is not an error norm. Do not encode sub-check codes in "
            f"max_diff; report the real diff, or max_diff=n/a plus one PROBE line per "
            f"contract"))
    return ProbeVerdict(True, max_diff=diff)


def _representative(contracts: tuple[ProbeContract, ...]) -> tuple[float | None, str | None]:
    """(first finite numeric diff, first non-numeric token) among the given lines."""
    diff = next((c.diff for c in contracts if c.diff is not None), None)
    reading = next((c.raw for c in contracts if c.raw is not None and c.diff is None), None)
    return diff, reading


def parse_probe_verdict(text: str, *, audit: bool = True) -> ProbeVerdict:
    """Audit the ``PROBE PASS/FAIL max_diff=<x>`` verdict(s) in a probe job's log.

    The verdict is written by the probe the student authored, so it is a claim, not
    evidence. Three ways that claim used to be taken at face value, all of which
    let a probe certify a kernel the eval then rejected:

    * only the *first* verdict was read, so a probe printing PASS for check 1 and
      FAIL for check 2 certified as PASS. Every verdict now has to agree — any FAIL
      loses.
    * a log with no verdict at all (build error, killed job) was indistinguishable
      from a numerical FAIL. It is now ``found=False``.
    * ``max_diff`` was decorative. It is now audited (see ``_audit_pass``).

    ``audit=True`` is the certification reading (student probes, KB recipe probes):
    a PASS is a claim that a kernel is numerically right, so its ``max_diff`` must
    be an error norm. ``audit=False`` is the verdict-experiment reading (diagnose /
    m0-audit): the lines are answers to pre-registered hypotheses and ``max_diff``
    carries the reading — a number, ``n/a``, or a ``k=v,...`` string — and is
    recorded as printed. A PASS is never downgraded for the shape of its reading
    there, because a downgraded PASS was reaching the teacher as "FAIL", i.e. as
    the hypothesis being falsified, which is the opposite of what the experiment
    said (e.g. ``PROBE PASS max_diff=ref_exposed=106.3,...`` on grouped-gemm).
    """
    contracts = _contracts(text)
    if not contracts:
        return ProbeVerdict(False, found=False, reason=(
            "no 'PROBE PASS/FAIL' line in the job log — the probe did not build, "
            "or the job died before reaching the verdict"))

    fails = tuple(c for c in contracts if c.verdict == "FAIL")
    if fails:
        diff, reading = _representative(fails)
        reason = ""
        if len(fails) < len(contracts):
            reason = (f"{len(fails)} of {len(contracts)} PROBE lines reported FAIL — "
                      f"a probe passes only when every check passes")
        return ProbeVerdict(False, max_diff=diff, reason=reason,
                            contracts=contracts, reading=reading)

    if not audit:
        diff, reading = _representative(contracts)
        diffs = [c.diff for c in contracts if c.diff is not None]
        if reading is not None and reading.strip().lower() in _PROBE_NON_NUMERIC:
            reading = None  # `n/a` on every line: a non-numeric experiment, as before
        return ProbeVerdict(True, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs) or reading is not None,
                            contracts=contracts, reading=reading)

    # all PASS: every one must survive the audit, and the worst diff is the one
    # that represents the probe.
    audited = [_audit_pass(c.raw) for c in contracts]
    rejected = next((v for v in audited if not v.passed), None)
    if rejected is not None:
        return ProbeVerdict(False, reason=rejected.reason, contracts=contracts)
    diffs = [v.max_diff for v in audited if v.max_diff is not None]
    # Every line said PASS — but a PASS certifies only a job that ran to completion.
    # A multi-rung probe that printed PASS for its first rung and then hung in the
    # next was scored PASS on that one line (on attn-bwd: `T-trace` PASS, `S-smoke`
    # hung 1800 s, rc=137, and the hanging tree was delivered as certified).
    end = job_abend(text)
    if end.kind == JOB_END_KILLED:
        return ProbeVerdict(False, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs), contracts=contracts, reason=(
                                f"every PROBE line said PASS but {end.reason} — the rungs "
                                f"after the last PROBE line (a launch that hung, a "
                                f"relaunch, a teardown) have no verdict. Bound every "
                                f"rung so a hang costs its own cap, not the job, and "
                                f"print one PROBE line per rung"))
    if end.kind == JOB_END_CUT:
        # Not a rejection of the probe: nothing in the log blames it. NO VERDICT
        # sends the student back for a run that ends, not hunting for a defect.
        return ProbeVerdict(False, found=False, max_diff=max(diffs) if diffs else None,
                            numeric=bool(diffs), contracts=contracts, reason=(
                                f"every PROBE line said PASS but {end.reason}. If the "
                                f"probe printed every line it should have, re-emit the "
                                f"same PROBE_REQUEST unchanged"))
    return ProbeVerdict(True, max_diff=max(diffs) if diffs else None,
                        numeric=bool(diffs), contracts=contracts)


def detect_plateau(records, window: int = 3, noise_floor: float = 0.015,
                   gate_window: str = GATE_WINDOW_DEFAULT) -> bool:
    """True when the last ``window`` evaluated iters set no new best beyond noise.

    Evaluated = has a reading in the gate window (REVERTs with None are skipped).
    The plateau is the meta-loop trigger: phase-internal levers have stopped paying —
    so it must watch the same window the loop gates on, or host-side progress on a
    device-gated run (and vice versa) reads as a plateau and hands the run to meta.

    "New best" must clear the measurement noise, not an absolute epsilon. The
    running best is the luckiest draw so far (max-of-noise ratchet), and the
    same kernel re-measured spans ~1.37% on this harness (4-sample span)
    — so the margin is relative: ``noise_floor`` (default 1.5%) of the
    prior best. The old absolute eps (0.005 ms ≈ 0.2% of a 2.5 ms kernel) sat
    an order of magnitude below that noise, which handed the trigger to the
    dice in both directions: real progress that failed to out-roll a lucky
    early sample read as plateau (meta invoked while levers still paid), and a
    lucky sub-noise draw read as progress (a dead phase kept grinding).
    """
    evaluated = [r for r in records if gate_ms(r, gate_window) is not None]
    if len(evaluated) <= window:
        return False
    prior_best = min(gate_ms(r, gate_window) for r in evaluated[:-window])
    recent_best = min(gate_ms(r, gate_window) for r in evaluated[-window:])
    return recent_best >= prior_best * (1.0 - noise_floor)
