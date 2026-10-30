"""The ref baseline: what the reference implementation actually does on this hardware.

The teacher used to write milestone v1 from the oracle source alone, so every gate
number in it was an [ESTIMATE] — including ``target_ms``, which is the driver's stop
threshold. The template requires the opposite ("every later gate number must be anchored
to the [MEASURED, job] readings of these two steps"), and that is impossible when distillation runs before anything
has been measured.

So the measurement moves in front of distillation and lands here: one archived record
per operator, holding the ref's end-to-end timing and (when a collection path exists)
its per-kernel NCU signature. Distillation reads it and writes gates anchored to real
numbers; the student never sees it as a milestone to execute.

The archive is a file rather than a step of the loop because a baseline is a property
of (operator, shape, hardware) — not of one run. Re-running a task should not have
to re-measure the same cuDNN number, and two runs of the same operator should be
comparing themselves against the identical baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forgekernel.inner.trajectory import GATE_WINDOW_DEFAULT

BASELINE_FILENAME = "ref_baseline.json"
ARCH_BASELINE_FILENAME = "arch_baseline.json"

# Two references answer two different questions, and collapsing them into one is how
# a run ends up optimising toward the wrong number:
#
#   library      — "is this competitive with what you would otherwise call?"  (cuDNN)
#   architecture — "did you reproduce the machine the milestone teaches?"     (the oracle)
#
# For flash_attn_bwd the library baseline is ~20% slower than the architecture it is
# meant to reproduce, so a target derived from the library stops the loop while it is
# still far short of the oracle. target_ms belongs to the architecture reference; the
# library one stays as the competitiveness check it always was.
ROLE_LIBRARY = "library"
ROLE_ARCHITECTURE = "architecture"

# How a device-window reading was obtained. Archived alongside the number because the
# two are not interchangeable and cannot be told apart afterwards from the value.
BASIS_PLAIN = "plain_batch"   # one event pair around N back-to-back calls / N
BASIS_GRAPH = "cuda_graph"    # replay of a captured graph: the host is out of the loop
# CUPTI device-event spans (torch.profiler CUDA activity), per call: the host is out of
# the loop too. The reading for a launch-bound call whose launcher cannot be captured
# into a graph (a tvm-ffi / driver-API launcher that torch's capture stream never sees
# comes back as an EMPTY graph), and the same currency the FlashInfer-Bench harness
# reports. Host-free for every check that exempts the graph basis.
BASIS_CUPTI = "cupti_kernel_span"
HOST_FREE_BASES = (BASIS_GRAPH, BASIS_CUPTI)
_ROLE_FILENAMES = {ROLE_LIBRARY: BASELINE_FILENAME, ROLE_ARCHITECTURE: ARCH_BASELINE_FILENAME}

# Bumped when a field changes meaning. A stale archive is refused rather than
# silently half-read: a gate anchored to a misread baseline is worse than no gate.
SCHEMA_VERSION = 1


@dataclass
class RefBaseline:
    """One measurement of the reference implementation.

    ``ms`` is the end-to-end median the target is derived from: a per-call timing
    window (record/call/record/sync each rep), so host work — launch latency, python
    wrappers — lands inside it. ``ms_device`` is the same reference through a
    pipelined batch window (one event pair around N back-to-back calls, divided by
    N): the stream never drains, host cost is hidden, and what remains is ≈ device
    time. The same cuDNN kernel has read 0.28 ms in the first window and 0.25 ms in
    the second, and the two sides of a ratio can pay different host taxes — so
    target_ms anchors to whichever window the operator gates on (loop.toml
    ``gate_window``, default "device": the series' benchmark question is kernel
    quality), and
    kernel-vs-kernel claims are read against ``ms_device``. None when the ref
    script only times one window — which blocks a device-gated run from starting,
    by design. ``ncu`` is the per-kernel signature when a
    collection path was available — empty otherwise, which is a legitimate state
    (the template keys NCU duties on availability, and a workload with no NCU path
    simply carries no NCU-anchored gate lines).
    """

    op: str
    ms: float
    role: str = ROLE_LIBRARY           # which question this reference answers
    job: str = ""                      # provenance: the GPU job the number came from
    shape: str = ""                    # free-form; whatever the bench script reports
    hardware: str = ""
    # Which GPU machine the number was measured on. Absolute latency for the same
    # kernel moves more between machines than most single-round gains, so a
    # baseline is only a valid anchor for readings from its own machine; the
    # loop refuses to reuse one measured on another machine (loop.cli's
    # baseline guards). "" = archived before the field existed / unknown.
    cluster: str = ""
    ref_name: str = ""                 # e.g. "cuDNN", "flash-attn cute"
    tflops: float | None = None
    mfu: float | None = None
    ms_device: float | None = None     # the device-window reading; see docstring
    # HOW ms_device was measured. Without this an archive cannot be audited after the
    # fact: a device number inflated 86% by a thick python wrapper looks exactly like a
    # clean one, and that indistinguishability is what let a 1.86x-loose target through.
    # "" = archived before the field existed (unknown, treated as plain_batch).
    device_window_basis: str = ""      # "" | BASIS_PLAIN | BASIS_GRAPH
    ms_device_plain: float | None = None   # pipelined batch reading, when also reported
    ms_device_graph: float | None = None   # CUDA-graph replay reading (host removed)
    # The device window's precondition check (see forgekernel.timing_audit): how long
    # the host takes to enqueue one call, and that over the device reading. A ratio
    # near 1 means the pipelined window measured python, not the kernel.
    host_enqueue_ms: float | None = None
    host_bound_ratio: float | None = None
    ncu: dict = field(default_factory=dict)
    # Every kernel the reference's timed window launched (inner.ncu.kernel_table:
    # name / launches / time_ns / pct), from the same CSV ``ncu`` was read off.
    # ``ncu`` keeps the dominant kernel's signature; this keeps the decomposition,
    # which is what the timed-window contract (forgekernel.window_contract) is
    # read against — an auxiliary launch inside the reference window is a fact
    # the student's contract must either mirror or declare. [] = archived
    # before the field existed, or no NCU path.
    kernels: list[dict] = field(default_factory=list)
    # Multi-workload references (the ref script reports timing.workloads_performance's
    # geomean block): every reading above is the geometric mean over ``n_workloads``
    # workloads of the card's shape, named in ``workloads``. 1 / [] = one input set,
    # and every archive written before the field existed. A gate reading is
    # comparable to this archive only over the same workloads (workloads_mismatch).
    n_workloads: int = 1
    workloads: list[str] = field(default_factory=list)
    samples_ms: list[float] = field(default_factory=list)  # every run; ms is their min
    raw_stdout: str = ""               # kept for drill-down; not parsed again
    schema_version: int = SCHEMA_VERSION

    @property
    def has_ncu(self) -> bool:
        return bool(self.ncu)

    @property
    def device_anchor_ms(self) -> float | None:
        """The device reading a kernel-vs-kernel claim may be anchored to.

        A validated CUDA-graph reading wins whenever one exists: it is the same
        reference with the host taken out, so it is the closer estimate of kernel
        time and it is immune to the wrapper thickness the plain window cannot see.
        """
        if self.ms_device_graph is not None:
            return self.ms_device_graph
        return self.ms_device

    @property
    def device_anchor_basis(self) -> str:
        """Which window ``device_anchor_ms`` came from."""
        if self.ms_device_graph is not None:
            return BASIS_GRAPH
        return self.device_window_basis or BASIS_PLAIN

    @property
    def host_bound(self):
        """The device window's precondition verdict for THIS reference.

        The enqueue ratio only means anything against a reading the host could have
        paced — i.e. a PLAIN-window one. A graph replay has the host out of the loop
        by construction, so dividing the enqueue clock by it compares two unrelated
        things and lands near 1.0 for a reference that is in fact host-free; an
        archive that already followed the prescribed fix (host-free reading in
        ms_device, basis declared cuda_graph) would then be refused as host-bound.
        With no plain reading to divide by, there is nothing to check and the verdict
        is unmeasured — which is an absent verdict, not a passing one (see
        forgekernel.timing_audit).
        """
        from forgekernel.timing_audit import classify
        plain = self.ms_device_plain
        if plain is None and self.device_anchor_basis not in HOST_FREE_BASES:
            plain = self.ms_device
        return classify(host_enqueue_ms=self.host_enqueue_ms,
                        device_median_ms=plain,
                        ratio=self.host_bound_ratio if plain is not None else None)

    @property
    def spread_pct(self) -> float | None:
        """Max-to-min spread across samples, or None if there is nothing to compare.

        Surfaced rather than averaged away: a wide spread means the node was busy,
        which is a fact about the measurement worth seeing next to the number.
        """
        if len(self.samples_ms) < 2:
            return None
        return (max(self.samples_ms) - min(self.samples_ms)) / min(self.samples_ms) * 100

    def target_ms(self, ratio: float, gate_window: str = GATE_WINDOW_DEFAULT,
                  allow_host_bound: bool = False) -> float:
        """The equivalence threshold: reproduce the ref to within ``ratio`` throughput.

        ratio=1.0 (the repo default) means the student must match the ref outright;
        a looser ratio like 0.95 would allow ~5.3% slower, since a throughput
        ratio inverts into a latency one. A ratio ABOVE 1 asks for more throughput
        than the reference (1.05 = beat it by 5%: the threshold is anchor / 1.05)
        — the self-evolution runs, whose reference is an earlier run's best.

        ``gate_window`` names the window the loop gates on, and the threshold must
        be derived from the ref's reading in that same window — a device-gated run
        whose target came from the e2e number would be chasing the ref's host tax
        as if it were kernel time. Refuses loudly when the archive never measured
        the requested window: deriving from the other one instead is exactly the
        silent-degradation this parameter exists to prevent.

        Same refusal when the device window itself is invalid for this reference
        (host-bound, no host-free reading archived) — the number exists but is not
        device time. ``allow_host_bound`` is the deliberate override for someone who
        has read the warning and wants the loose target anyway.
        """
        if not 0 < ratio:
            raise ValueError(f"ratio must be positive, got {ratio}")
        if gate_window == "device":
            if self.device_anchor_ms is None:
                # The prescribed fix differs by role: the library baseline is
                # re-measured by a command the operator can just run, while the
                # architecture reference may exist only as declared loop.toml values
                # (declared BECAUSE it cannot be run) — telling that operator to
                # re-measure is a dead end.
                if self.role == ROLE_ARCHITECTURE:
                    raise ValueError(
                        f"gate_window='device' but the archived {self.role} baseline "
                        f"for {self.op} has no device-window reading (ms_device) — "
                        f"re-run the architecture reference measurement, or declare "
                        f"arch_ref_device_ms in loop.toml (a declared-sourced archive "
                        f"refreshes from the toml on the next start)")
                raise ValueError(
                    f"gate_window='device' but the archived {self.role} baseline for "
                    f"{self.op} has no device-window reading (ms_device) — re-measure "
                    f"with a ref script that reports both windows "
                    f"(`forge {self.op} baseline`)")
            # The device window is only device time while the host keeps the pipeline
            # fed. A reference with a thick python wrapper breaks that precondition,
            # and the target derived from it is loose by whatever the wrapper costs —
            # measured once at 1.86x, which bought an 11.5-hour run whose result could
            # not be cited. Refuse instead, unless the anchor is host-free. The
            # exemption keys on the anchor's BASIS, not on which field carried it: an
            # archive may put the graph reading straight into ms_device and declare
            # the basis (what `arch_ref_device_basis` in loop.toml produces), and
            # refusing that would gate the very fix this check prescribes.
            from forgekernel.timing_audit import reference_blocker
            host_free = (self.ms_device_graph
                         if self.ms_device_graph is not None
                         else (self.device_anchor_ms
                               if self.device_anchor_basis in HOST_FREE_BASES else None))
            blocker = reference_blocker(self.host_bound, graph_ms=host_free,
                                        ref_name=self.ref_name, op=self.op)
            if blocker and not allow_host_bound:
                raise ValueError(blocker)
            return self.device_anchor_ms / ratio
        return self.ms / ratio

    def render(self, gate_window: str = GATE_WINDOW_DEFAULT) -> str:
        """The block injected into the distill prompt. Prose, not JSON — the teacher
        reads this as evidence, and a table is easier to anchor gate lines to.

        ``gate_window`` moves the "target anchors here" marker onto the reading
        the run actually gates on. The marker is instruction, not decoration: the
        teacher copies whichever line carries it into §0's target derivation, so
        leaving it on the e2e line for a device-gated run plants an e2e-anchored
        target that gates ~one host tax loose.
        """
        head = (f"- ref: {self.ref_name or '(unnamed)'}"
                f"{f' @ {self.shape}' if self.shape else ''}"
                f"{f' on {self.hardware}' if self.hardware else ''}")
        e2e_anchor = gate_window != "device"
        lines = [head, f"- end-to-end median: **{self.ms} ms** [MEASURED, job {self.job or '?'}]"
                 + (" (per-call window, includes host overhead — the target anchors on this number)"
                    if e2e_anchor else
                    " (per-call window, includes host overhead; this run gates on the device window, "
                    "the target does not anchor on this number)")]
        if self.device_anchor_ms is not None:
            basis = {BASIS_GRAPH: "CUDA graph replay, host excluded",
                     BASIS_CUPTI: "CUPTI kernel span (per-launch device duration from the profiler), "
                                  "host excluded"
                     }.get(self.device_anchor_basis, "pipelined amortized window")
            lines.append(f"- device median: **{self.device_anchor_ms} ms** [MEASURED, job "
                         f"{self.job or '?'}] ({basis} ≈ pure device — "
                         + ("the target anchors on this number; " if not e2e_anchor else "")
                         + "kernel-to-kernel comparisons read this number; do not mix it with "
                         "end-to-end)")
            if self.host_bound.host_bound:
                plain = self.ms_device_plain or self.ms_device
                lines.append(
                    f"- ⚠ About {self.host_bound.pct} of this reference's plain pipelined amortized "
                    f"reading ({plain} ms) is host-side enqueue overhead (host_bound_ratio="
                    f"{self.host_bound.ratio:.2f}): that window is invalid for it, so the device "
                    f"number above already uses a host-free basis. **Do not write \"the reference "
                    f"has large host overhead\" into the milestone, and do not set host-orchestration "
                    f"gates on that basis** — it is a fact about the measurement layer; putting it in "
                    f"the blueprint only teaches the score-gaming solution \"make the python path "
                    f"thinner and you win\". What this run has to reproduce is the kernel.")
        elif not e2e_anchor:
            lines.append("- device median: not measured — this run gates on the device window; any "
                         "number anchored to this baseline may only come from a device reading taken "
                         "afterwards, and must not be substituted with the end-to-end reading")
        if self.n_workloads > 1:
            lines.append(f"- the readings above are the geomean over {self.n_workloads} workloads"
                         + (f" ({', '.join(self.workloads)})" if self.workloads else ""))
        if (spread := self.spread_pct) is not None:
            lines.append(f"- fastest of {len(self.samples_ms)} measurements"
                         f" ({', '.join(str(s) for s in self.samples_ms)}; spread {spread:.1f}%)")
        if self.tflops is not None:
            lines.append(f"- throughput: {self.tflops} TFLOP/s"
                         + (f" ({self.mfu}% MFU)" if self.mfu is not None else ""))
        if self.ncu:
            lines.append("- per-kernel NCU signature [MEASURED, job "
                         f"{self.job or '?'}] (the gates' mechanism-verification lines anchor on "
                         "these numbers):")
            for k, v in sorted(self.ncu.items()):
                lines.append(f"    {k}: {v}")
        else:
            lines.append("- NCU: no collection path; this operator writes no NCU-anchored gate line")
        if self.kernels:
            from forgekernel.window_contract import render_kernel_rows
            lines.append(f"- every launch inside the reference's timed window ({len(self.kernels)} "
                         f"kernels, by time share; the signature above is only the main kernel) — the "
                         f"student contract either mirrors them or declares them in window.toml:")
            lines.extend(render_kernel_rows(self.kernels, self.device_anchor_ms))
        return "\n".join(lines)


def _describe_workloads(n: int) -> str:
    return "one input set" if n <= 1 else f"the geomean over {n} workloads"


def workloads_mismatch(ref_n, ref_names, gate_n, gate_names) -> str | None:
    """Why a gate reading over ``gate_n`` workloads cannot be held against a
    reference reading over ``ref_n`` — None when it can.

    A multi-workload card's gate reading is the geometric mean over its
    workloads (timing.workloads_performance). An absolute target, a cluster
    scale or a median-vs-median ratio built from a reference reading of ANOTHER
    input set divides two different quantities: the workloads' cost mix, not the
    kernel, then sets the gap, in either direction. Same count required; when
    both sides name their workloads, the same names.
    """
    ref_n = max(1, int(ref_n or 1))
    gate_n = max(1, int(gate_n or 1))
    if ref_n != gate_n:
        return (f"the reference reading is {_describe_workloads(ref_n)}, the gate "
                f"reading {_describe_workloads(gate_n)}")
    ref_set = sorted(str(n) for n in (ref_names or []))
    gate_set = sorted(str(n) for n in (gate_names or []))
    if ref_n > 1 and ref_set and gate_set and ref_set != gate_set:
        return (f"the reference reading covers workloads {ref_set}, the gate reading "
                f"{gate_set}")
    return None


def baseline_path(op_dir: Path, role: str = ROLE_LIBRARY) -> Path:
    return Path(op_dir) / _ROLE_FILENAMES.get(role, BASELINE_FILENAME)


def save_baseline(op_dir: Path, baseline: RefBaseline) -> Path:
    p = baseline_path(op_dir, baseline.role)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(baseline), indent=2, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    return p


def load_baseline(op_dir: Path, role: str = ROLE_LIBRARY) -> RefBaseline | None:
    """The archived baseline, or None if absent. Raises on a corrupt/stale archive.

    Absence is normal (nothing has measured yet); a file that cannot be read as the
    current schema is not — proceeding would anchor gates to misread numbers.
    """
    p = baseline_path(op_dir, role)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"ref baseline at {p} is unreadable: {e}") from e
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"ref baseline at {p} has schema_version={version}, expected "
            f"{SCHEMA_VERSION} — re-measure with `forge {data.get('op', '<op>')} baseline`")
    known = {f for f in RefBaseline.__dataclass_fields__}
    return RefBaseline(**{k: v for k, v in data.items() if k in known})
