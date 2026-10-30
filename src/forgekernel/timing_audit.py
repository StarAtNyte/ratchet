"""Is a "device time" reading actually device time?

The device window (``src/forgekernel/card_template/eval/timing.py``) amortizes N back-to-back
calls under one event pair. That equals device time only while the host can enqueue
faster than the device drains — a precondition, not a result. When the callee's
per-call host cost approaches its per-call GPU cost the stream drains between
launches, the GPU idles, and the window reports host dispatch cost under a device
label. Nothing downstream can see it: correctness passes, the number is stable
across jobs, and node-speed noise is already controlled for.

Measured (a library reference with a 593-line python wrapper in front of a light
kernel):

    plain loop, device window        0.4889 ms
    same, outputs pre-allocated      0.5003 ms
    CUDA-graph replay (host removed) 0.2598 ms

46% of that "device time" was per-call python. The student kernel measured in the
same window agreed with its own NCU kernel total, so the window was valid on one
side of the comparison and invalid on the other, and the resulting ratio was 1.86x
in the student's favour.

The two sides need OPPOSITE handling, which is why this module splits them:

* **reference host-bound** — a defect of the measurement, not of anything the loop
  can optimize. It must never reach the student: "the reference's python wrapper is
  thick" reads to a kernel writer as "a thin python path wins", which is a way of
  teaching metric-gaming. It also must not become a target: re-measure the reference
  through a host-free window (graph replay, or NCU kernel time) and anchor there, or
  refuse to start. The alternative has been priced — one run spent 11.5 hours against a
  target that was 1.86x too loose and produced a result that cannot be cited.
* **student host-bound** — a real performance defect (per-call recompiles, python-side
  allocation, redundant launches, a synchronize inside the call), and it inflates the
  student's OWN reading, so it belongs in the diagnosis the student gets next round.

**Host-free bases are exempt from all of the above.** A device reading taken under
CUDA-graph replay or as a CUPTI per-launch kernel span (``baseline.HOST_FREE_BASES``)
has the host out of the loop BY CONSTRUCTION: the enqueue clock still exists, but it
no longer describes the gate reading, and dividing the two says nothing about the
window. A 9.5 µs DSA sparse-attention kernel on B200 was measured behind a 19–65 µs
python path on BOTH sides — that is exactly why its card gates on the span — and the
card reported ``host_bound_ratio: null`` for it; the driver then recomputed
host_enqueue / device_median = 3–6 and told the student, three rounds running, that
its gate reading was "inflated by the host path — a real defect". It was not: the
span cannot contain host time. Under a host-free basis the host path is reported as
what it is — a DEPLOYMENT cost, in its own note, never as a gate verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgekernel.baseline import HOST_FREE_BASES

# host_enqueue_ms / device_median_ms at or above this = the device window is
# measuring the host. Conservative on purpose: a GPU-bound callable sits one to two
# orders of magnitude below it. Kept in sync with HOST_BOUND_RATIO_WARN in
# src/forgekernel/card_template/eval/timing.py, which is copied verbatim into job tarballs and
# so cannot import this module.
HOST_BOUND_RATIO_WARN = 0.5

# Below this share of the timed window accounted for by profiled kernel time, the
# profile and the clock disagree about what was measured. Same 20% the mis-aim alarm
# has always used; see ncu_coverage.
NCU_COVERAGE_FLOOR = 0.20


@dataclass(frozen=True)
class HostBoundVerdict:
    """What the enqueue clock says about one device-window reading.

    ``measured`` separates "the bench checked and the window is fine" from "the
    bench never reported the enqueue clock". The second is not a pass: benches
    predating the check simply carry no verdict, and a caller must not read the
    absence as evidence.
    """

    ratio: float | None = None
    host_bound: bool = False
    measured: bool = False
    threshold: float = HOST_BOUND_RATIO_WARN
    # The basis the device reading was taken on, when the bench declared one. A
    # host-free basis (graph replay / CUPTI kernel span) makes the verdict "the
    # window cannot be host-bound": ratio stays None because the enqueue clock does
    # not describe that reading, and host_bound is False by construction.
    basis: str | None = None

    @property
    def host_free(self) -> bool:
        return is_host_free_basis(self.basis)

    @property
    def pct(self) -> str:
        return "unmeasured" if self.ratio is None else f"{self.ratio * 100:.0f}%"


def is_host_free_basis(basis) -> bool:
    """Is ``basis`` (a bench's ``device_window_basis``) one where the host is out of
    the timed window by construction? See baseline.HOST_FREE_BASES."""
    return bool(basis) and basis in HOST_FREE_BASES


def host_bound_ratio(host_enqueue_ms, device_median_ms):
    """host enqueue / device reading, or None when either side is missing/zero."""
    if not host_enqueue_ms or not device_median_ms:
        return None
    return host_enqueue_ms / device_median_ms


def classify(host_enqueue_ms=None, device_median_ms=None, ratio=None,
             threshold: float = HOST_BOUND_RATIO_WARN, basis=None) -> HostBoundVerdict:
    """Verdict from whatever the bench reported.

    ``ratio`` is used when the bench already computed it (timing.py does); otherwise
    it is derived from the two timings. Neither present -> an unmeasured verdict.

    ``basis`` is the bench's ``device_window_basis``. On a host-free basis the two
    timings are NOT divided: the reading is a kernel span / graph replay and the
    enqueue clock is a fact about the python path around it, not about the window.
    The verdict then carries the basis, is measured (the window's precondition holds
    by construction) and is not host-bound — whatever a bench that still printed a
    ratio for the plain window would have said.
    """
    if is_host_free_basis(basis):
        return HostBoundVerdict(measured=True, threshold=threshold, basis=basis)
    value = ratio if ratio is not None else host_bound_ratio(host_enqueue_ms,
                                                             device_median_ms)
    if value is None:
        return HostBoundVerdict(threshold=threshold, basis=basis)
    return HostBoundVerdict(ratio=value, host_bound=bool(value - threshold >= 0),
                            measured=True, threshold=threshold, basis=basis)


def host_path_note(host_enqueue_ms, device_ms, *, basis, ref_host_enqueue_ms=None,
                   threshold: float = HOST_BOUND_RATIO_WARN) -> str:
    """The operator-facing line for a thick python path under a HOST-FREE basis, or "".

    Not a verdict on the gate reading — the span / graph number cannot contain
    host time, so nothing here says "inflated". It is the deployment fact the gate
    deliberately excludes: what one call costs to enqueue, next to what the in-job
    reference costs when the bench timed it, so a wrapper N× thicker than the
    reference's is visible without becoming a target. Silent below the same
    threshold the plain-window verdict uses, and on a plain basis (where the verdict
    itself speaks).
    """
    if not is_host_free_basis(basis):
        return ""
    share = host_bound_ratio(host_enqueue_ms, device_ms)
    if share is None or share - threshold < 0:
        return ""
    head = (f"HOST PATH: one call takes {host_enqueue_ms:.4g} ms to enqueue against a "
            f"{device_ms:.4g} ms kernel (basis={basis}: the gate reads the kernel alone, "
            f"so the gate reading is NOT affected)")
    if ref_host_enqueue_ms:
        head += (f"; the in-job reference enqueues in {ref_host_enqueue_ms:.4g} ms "
                 f"({host_enqueue_ms / ref_host_enqueue_ms:.1f}x thinner)")
    return head + ". Deployment pays the python path; it is a cost to record, not a gate."


def host_path_feedback(host_enqueue_ms, device_ms, *, basis, ref_host_enqueue_ms=None,
                       threshold: float = HOST_BOUND_RATIO_WARN) -> str:
    """The student-facing counterpart of host_path_note, or "".

    Same four host-side causes as student_feedback, but framed as what it is under
    a host-free basis: a deployment cost outside the gate. Says so in the first
    clause, because the alternative — a diagnosis that reads "your reading is
    inflated" — sent a student after its kernel body for a cost that lives in the
    wrapper, and sent a teacher discounting a span reading that was already clean.
    """
    if not host_path_note(host_enqueue_ms, device_ms, basis=basis,
                          ref_host_enqueue_ms=ref_host_enqueue_ms, threshold=threshold):
        return ""
    ref = (f"; the reference's enqueue takes only {ref_host_enqueue_ms:.4g} ms"
           f" (yours is {host_enqueue_ms / ref_host_enqueue_ms:.1f}x that)"
           if ref_host_enqueue_ms else "")
    return (f"- Deployment note (**not part of the gate**): this round's gate reading is pure "
            f"kernel time on the {basis} basis ({device_ms:.4g} ms), **not inflated by the "
            f"host**; but the python enqueue path of each call takes "
            f"{host_enqueue_ms:.4g} ms{ref}. A deployment pays this. It is a separate matter: "
            f"do not change the kernel body for it, and do not set a gate on it; if it is cheap "
            f"to fix along the way, check the four host-side causes: recompiling / JIT on every "
            f"call, python-side tensor allocation inside the call (including repeated "
            f"dlpack/view wrapping), extra kernel launches, and synchronization inside the call "
            f"(.item()/.cpu()/synchronize).\n")


def ncu_coverage(profiler: dict, timed_ms):
    """Share of a timed window that PROFILED KERNEL TIME accounts for, or None.

    Both sides must be per CALL. ``total_time_ns`` is a window total — all kernels
    over all their launches — while ``timed_ms`` is one call's reading, so the raw
    quotient scales with whatever ``--launch-count`` the ncu task was configured
    with (24 in the template's ``ref_ncu_cmd``, which alone would turn a known
    mis-aimed profile from 0.029 into 0.70 and silence the alarm that
    caught it). ``dominant_launches`` is the per-call normaliser: the profiled
    window holds that many calls' worth under the same "one dominant launch per
    call" convention the mis-aim alarm has always assumed.

    Uses the all-kernel total rather than the dominant kernel alone: a legitimately
    decomposed operator spreads its work over several kernels, and judging the clock
    against the dominant one reads a decomposition as a measurement error.

    A coverage well under 1 means the GPU was idle for the rest of the window, which
    is the same physical fact host_bound_ratio measures — from the device side. It
    stays corroboration rather than the primary judgement: the per-call convention
    holds for a kernel launched once per call and undercounts one launched several
    times, so every use of it below WARNS and none of them stops a run.
    """
    total_ns = profiler.get("total_time_ns")
    launches = profiler.get("dominant_launches") or 1
    if total_ns:
        per_call_ns = total_ns / max(1, launches)
    else:
        # Legacy digest (no all-kernel total): the dominant kernel's own per-launch
        # time is already per call under the same convention.
        per_call_ns = profiler.get("time_ns")
    if not per_call_ns or not timed_ms:
        return None
    return per_call_ns / (timed_ms * 1e6)


_REF_FIX = ("Re-measure the reference through a host-free window and anchor target "
            "there: measure_windows_with_graph() in eval/timing.py (CUDA-graph replay), "
            "or the reference's NCU kernel total. Do NOT keep the pipelined reading and "
            "do NOT hand this to the student — a thick wrapper on the reference is a "
            "fact about the measurement, and telling a kernel writer about it teaches "
            "metric-gaming, not kernel quality.")


def reference_warning(verdict: HostBoundVerdict, *, ref_name: str = "",
                      device_ms=None, graph_ms=None) -> str:
    """Loud text for a host-bound REFERENCE reading, or "" when there is nothing to say."""
    if not verdict.host_bound:
        return ""
    who = ref_name or "the reference"
    head = (f"HOST-BOUND REFERENCE: {who} spends {verdict.pct} of its device-window "
            f"reading{f' ({device_ms} ms)' if device_ms else ''} on host-side enqueue "
            f"(host_bound_ratio={verdict.ratio:.2f}, threshold {verdict.threshold}). "
            f"The pipelined window never fills, so that number is dispatch cost, not "
            f"kernel time.")
    if graph_ms:
        return (f"{head} A host-free reading IS archived ({graph_ms} ms via graph replay) "
                f"and target anchors on it instead.")
    return f"{head} {_REF_FIX}"


def reference_blocker(verdict: HostBoundVerdict, *, graph_ms=None, ref_name: str = "",
                      op: str = "") -> str:
    """Why target must not be derived from this reference, or "" when it may be.

    Fires only on a MEASURED host-bound verdict with no host-free reading archived.
    An unmeasured verdict cannot block: benches predating the enqueue clock are the
    normal state of every archive written before it existed, and refusing those would
    stop every run rather than the broken ones.
    """
    if not verdict.host_bound or graph_ms:
        return ""
    who = ref_name or "the reference"
    return (f"gate_window='device' but {who}"
            f"{f' for {op}' if op else ''} is HOST-BOUND "
            f"(host_bound_ratio={verdict.ratio:.2f} at threshold {verdict.threshold}): "
            f"its device reading is host dispatch cost, so a target derived from it is "
            f"loose by whatever the wrapper costs. {_REF_FIX}")


def student_warning(verdict: HostBoundVerdict, *, device_ms=None) -> str:
    """Loud text for a host-bound STUDENT reading, or "" when there is nothing to say."""
    if not verdict.host_bound:
        return ""
    return (f"HOST-BOUND STUDENT KERNEL: {verdict.pct} of the device-window reading"
            f"{f' ({device_ms} ms)' if device_ms else ''} is host-side enqueue "
            f"(host_bound_ratio={verdict.ratio:.2f}, threshold {verdict.threshold}). "
            f"The gate reading is inflated by the host path, and the deployment pays "
            f"it too — this is a real defect, not a measurement artifact.")


def student_feedback(verdict: HostBoundVerdict, *, device_ms=None) -> str:
    """The diagnosis handed to the student for its next decision, or "".

    Deliberately names the four host-side causes rather than the symptom: the
    student cannot act on "host_bound_ratio is high", and a vague note sends it
    editing the kernel body, which is the one place the cost is NOT.
    """
    if not verdict.host_bound:
        return ""
    return (f"- Timing diagnosis: about {verdict.pct} of this round's device-window reading is "
            f"**host-side enqueue overhead** (host_bound_ratio={verdict.ratio:.2f}"
            f"{f', device reading {device_ms} ms' if device_ms else ''}). "
            f"The pipeline is not kept full and the GPU is waiting on your python path, so this "
            f"reading is not pure kernel time yet, and a deployment pays it too. Next, check the "
            f"four host-side causes first: recompiling / JIT on every call, python-side tensor "
            f"allocation inside the call, extra kernel launches, and synchronization inside the "
            f"call (.item()/.cpu()/synchronize). Return to the kernel body once these are gone.\n")


def ncu_reconciliation_warning(profiler: dict, timed_ms, verdict: HostBoundVerdict,
                               *, floor: float = NCU_COVERAGE_FLOOR) -> str:
    """Warn when profiled kernel time cannot account for the timed window.

    Two very different causes produce the same shortfall, and the enqueue clock is
    what tells them apart, so both readings are named in the message:

    * the kernel under test never reached the CSV (mis-aimed profile — the recorded
      metrics belong to some setup kernel, and every NCU-anchored gate line is being
      compared against the wrong thing);
    * the GPU was genuinely idle for the rest of the window (host-bound).

    ALWAYS a warning, never a stop. See ncu_coverage for why the per-call
    normalisation is not trustworthy enough to gate on.
    """
    share = ncu_coverage(profiler, timed_ms)
    if share is None or share - floor >= 0:
        return ""
    # Report the same per-call quantity the share was computed from, not the window
    # total — a message quoting a 24x-inflated kernel time next to a per-call clock
    # reads as a contradiction and sends the reader after the wrong thing.
    launches = profiler.get("dominant_launches") or 1
    total_ns = profiler.get("total_time_ns")
    per_call_ns = (total_ns / max(1, launches)) if total_ns else profiler.get("time_ns")
    detail = (f"profiled kernel time {per_call_ns / 1e6:.3f} ms/call "
              f"covers only {share * 100:.1f}% of the timed {timed_ms:.3f} ms "
              f"(kernel_count={profiler.get('kernel_count', '?')}, "
              f"dominant_launches={profiler.get('dominant_launches', '?')}, "
              f"total_launches={profiler.get('total_launches', '?')})")
    if verdict.host_free:
        # A span / graph reading has no host time in it to be idle behind: the
        # only way profiled kernel time falls short of it is the profile looking
        # at the wrong launches.
        return (f"{detail} — the timed window is a host-free reading "
                f"(basis={verdict.basis}), so the gap cannot be host time: this is a "
                f"mis-aimed profile. Check the profiled window in eval/ncu_run.py, "
                f"not the kernel.")
    if verdict.host_bound:
        return (f"{detail} — and the enqueue clock says host-bound "
                f"(host_bound_ratio={verdict.ratio:.2f}), so the gap is host time, not a "
                f"mis-aimed profile. Fix the host path, not eval/ncu_run.py.")
    if verdict.measured:
        return (f"{detail} — the enqueue clock says the host is NOT the bottleneck "
                f"(host_bound_ratio={verdict.ratio:.2f}), so this is a mis-aimed "
                f"profile: the kernel under test is probably absent from the CSV. "
                f"Check the profiled window in eval/ncu_run.py, not the kernel.")
    return (f"{detail} — cause unresolved. Either the profile is mis-aimed (the kernel "
            f"under test never reached the CSV — check the profiled window in "
            f"eval/ncu_run.py) or the GPU sat idle behind a host bottleneck. The bench "
            f"reports no host_enqueue_ms, so the two cannot be told apart; have "
            f"eval/bench.py report the enqueue clock (timing.py measure_windows).")
