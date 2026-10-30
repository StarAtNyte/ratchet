"""Canonical dual-window timing for operator benches. Copy into <op>/eval/, do not edit.

Every bench must measure BOTH windows and report both in the eval JSON, because
they answer different questions and the same kernel reads visibly different in
them (the same cuDNN attention: 0.28 ms e2e vs 0.25 ms device):

  e2e     — per-call window: record / call / record / synchronize each rep. The
            stream starts idle every rep, so host work (launch latency, python
            wrappers) lands inside the number. This is single-call latency, what
            an inference caller pays.
  device  — pipelined batch window: one event pair around N back-to-back calls,
            divided by N. PROVIDED the host can enqueue faster than the device
            drains, the stream never drains, host cost hides under the pipeline,
            and what remains is ≈ pure device time — kernel quality, what a
            saturated training step pays.

THAT PROVISO IS A PRECONDITION, NOT A RESULT. It holds only while the callee's
per-call GPU time exceeds its per-call host time. Point a thick-python-wrapper
library (validation + dispatch + logging decorators in front of a light kernel)
at this window and the pipeline never fills: the GPU idles between launches and
the "device" reading is the host's dispatch cost wearing a device number's
clothes. Measured live: one library reference read 0.4889 ms in this window and
0.2598 ms under CUDA-graph replay (host removed) — 46% of its "device time" was
per-call python. A thin-path student kernel measured in the same window agreed
with its own NCU kernel total, so the window was valid for it and invalid for
the reference, and the ratio between them was off by 1.86x in the student's
favour. Nothing downstream could tell: correctness passed, the reading was
reproducible across jobs, and node-speed noise was already ruled out.

So the window ships with its own precondition check. ``measure_windows`` also
times how fast the host can ENQUEUE the call (``time_host_enqueue``: N calls,
no synchronize in between, so the loop measures python + launch and nothing on
the device) and reports:

  host_enqueue_ms   — per-call host cost of getting the work submitted
  host_bound_ratio  — host_enqueue_ms / device_median_ms

GPU-bound, the host runs ahead and the ratio is small. Host-bound, the host is
the pipeline and the ratio approaches 1 — at which point the device number is
not a kernel-quality reading and must not be used as one. See
``time_device_graph`` for the replacement reading, and the driver's
``forgekernel.timing_audit`` for what each side of a comparison must do about it.

Which window the loop gates on is the run's loop.toml [loop].gate_window
(default "device"); the bench's job is to always report both so the choice stays in
config instead of in the measurement. Map the results into the eval JSON
performance block as:

    e2e:    "bwd_median_ms"     (+ ref: "ref_median_ms",  ratio "vs_ref")
    device: "device_median_ms"  (+ ref: "ref_device_ms",  ratio "vs_ref_device")

SINGLE-REF CONTRACT: the ref_* keys may carry exactly one thing — THE reference
this run declares in loop.toml, timed back-to-back in the same job. If the ref
cannot ride the student job (blackbox runs: the ref package is readable source),
the bench times ONLY the student and emits no ref_* keys at all. Never time a
stand-in library "to calibrate node speed" — that retired instrument concept made
every vs_* number ambiguous (a same-job 1.0418x once read as "beat FA" when the
denominator was actually cuDNN), and the driver now drops such readings loudly
(loop/driver.py _enforce_single_ref). Cross-job noise for ref-less runs is handled
by repetition instead: the post-stop confirmation stage re-measures best and ref
through their own channels and compares medians.
(parse_eval still reads the pre-rename cublas_* keys as aliases for old benches.)

Ratios must divide readings from the SAME window: the two sides of an e2e ratio
can pay different host taxes, so an e2e ratio is a call-latency claim and only
the device ratio is a kernel-vs-kernel claim — and only when the device window is
valid for BOTH sides (host_bound_ratio low on both).

WHEN THE REFERENCE RIDES THE JOB, take both sides through ``measure_paired`` and
map the result with ``paired_performance`` — never two bare ``measure_windows``
calls in a row. The same-job ratio is only fair when both sides sit on the same
warm plateau: the paired measurement burns the part in with the kernel under
test and then runs ref, kernel, kernel, ref (mirrored), each side keeping the
mean of its two device readings. Measured back to back without that, the
first-measured side reads 2-5% faster for no kernel reason — the whole margin
of a tight equivalence target. Section "paired in-job reference" below.

PREP — WORK THE TIMED CALL RECEIVES FROM OUTSIDE THE WINDOW. Both windows say how
a reading is taken; the card's ``window.toml`` says what work sits inside it. A
backward's reference is called the way its autograd Function calls it, with the
tensors its forward saved; a student timed from the raw inputs rebuilds them
inside its window and pays a fixed tax no kernel lever removes. The contract may
SYMMETRIZE such a tensor: the student then receives it too, produced by a prep
step that runs here, outside every window, the way the reference's forward does::

    def prep_saved():                      # a module-level def in bench.py, torch only
        return {"A": solve_A(k, beta), "g_cumsum": torch.cumsum(g, dim=1)}

    win = measure_windows(lambda saved: bwd(q, k, v, saved["A"], saved["g_cumsum"], ...),
                          prep=prep_saved)
    performance["prep"] = win["prep"]      # ride it in the eval JSON, verbatim

The prep step is the one place work legitimately leaves the window, so it is
fenced: what it returns is DESCRIBED (name / shape / dtype), TIMED, and held
READ-ONLY — the same tensors are handed to every timed call, so a kernel that
wrote its result or any state into them would make every call after the first
a copy; a snapshot is compared after the windows and the names that changed are
reported as ``mutated``. All of it rides the eval JSON; the driver holds it to
the declaration (an undeclared extra, a missing one, a wrong shape or dtype, a
mutated tensor fails the iteration the way a correctness FAIL does; a bench
that reports no prep block at all stops the run) and shows the teacher what the
outside step costs. The prep function is the CARD AUTHOR's, written in bench.py
with torch: it must not reference anything imported from ``kernel`` — the
student owns the timed call, never the step outside it — and ``eval/ncu_run.py``
must call the same function before ``torch.cuda.profiler.start()`` so prep
launches stay out of the NCU table (``forgekernel.window_contract`` checks all
of this statically at ``loop start``).
"""

from __future__ import annotations

import inspect
import math
import statistics
import time
import warnings

import torch

WARMUP_ITERS = 20
E2E_ITERS = 50        # per-call samples; median reported
DEVICE_BATCH = 50     # back-to-back calls inside one amortized window
DEVICE_REPS = 5       # windows measured; median of the per-call amortizations
HOST_ENQUEUE_ITERS = 50   # enqueue-only samples for the precondition check
PREP_ITERS = 10           # samples of the window-outside prep step (reported, never gated)

# host_bound_ratio at or above this means the device window is measuring the host
# rather than the kernel. Deliberately conservative: a healthy GPU-bound kernel
# sits one to two orders of magnitude below it, and the live failure that motivated
# the check sat at ~0.5-1.0. Mirrored in forgekernel.timing_audit (this file is
# copied verbatim into job tarballs and must not import the driver package).
HOST_BOUND_RATIO_WARN = 0.5

# A graph replay faster than this share of the ordinary device reading is not a
# measurement, it is an empty capture. Real host removal has been worth ~2x; 20x
# is not physical. (The empty-graph case reads ~0.0013 ms against a 0.4 ms kernel.)
GRAPH_MIN_PLAUSIBLE_SHARE = 0.05


def time_e2e(fn, *, warmup: int = WARMUP_ITERS, iters: int = E2E_ITERS) -> list[float]:
    """Per-call latency samples in ms (host tax included). Median is the reading."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return samples


def time_device(fn, *, warmup: int = WARMUP_ITERS, batch: int = DEVICE_BATCH,
                reps: int = DEVICE_REPS) -> list[float]:
    """Amortized per-call times in ms from pipelined batch windows (≈ device time).

    One event pair brackets ``batch`` back-to-back launches; the batch keeps the
    stream from draining, so host cost hides under the pipeline. Returns one
    amortized reading per window; median is the reading.

    Valid only while the host can enqueue faster than the device drains — check
    that with ``time_host_enqueue`` / ``host_bound_ratio`` before reading the
    result as kernel time.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        for _ in range(batch):
            fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) / batch)
    return samples


def time_host_enqueue(fn, *, warmup: int = WARMUP_ITERS,
                      iters: int = HOST_ENQUEUE_ITERS) -> float:
    """Per-call wall time for the HOST to submit the work, in ms.

    No synchronize inside the loop: the host only enqueues, so what the clock sees
    is python + argument validation + launch, with the device running behind.
    Cheap — needs no profiler and no graph capture — and it is the whole
    precondition test for the device window; compare it against the device reading
    with ``host_bound_ratio``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed_ms = (time.perf_counter() - t0) * 1e3 / max(1, iters)
    torch.cuda.synchronize()   # drain what the loop queued; outside the clock
    return elapsed_ms


def host_bound_ratio(host_enqueue_ms, device_median_ms):
    """host / device. Near 0 = the device window is valid; near 1 = it is the host.

    None when either side is missing or zero — an absent verdict, never a passing
    one.
    """
    if not host_enqueue_ms or not device_median_ms:
        return None
    return host_enqueue_ms / device_median_ms


def run_prep(prep) -> dict:
    """Run the window-outside prep step once and hand back what it produced.

    ``prep()`` must return ``{name: tensor}`` — the names are what the card's
    ``window.toml`` declares as symmetrized inputs. Synchronized before
    returning, so nothing of it can spill into the first timed window.
    """
    saved = prep()
    if not isinstance(saved, dict) or not all(isinstance(k, str) for k in saved):
        raise TypeError("prep() must return a dict {name: tensor}; the names are the "
                        "symmetrized inputs declared in window.toml")
    torch.cuda.synchronize()
    return saved


def describe_saved(saved: dict) -> dict:
    """``{name: {"shape": [...], "dtype": "bfloat16"}}`` — what the driver holds
    against the declaration. Shape as plain ints, dtype without the ``torch.``
    prefix; the driver normalises aliases (bf16 / bfloat16) on its side."""
    out = {}
    for name, t in saved.items():
        shape = [int(s) for s in getattr(t, "shape", ())]
        dtype = str(getattr(t, "dtype", ""))
        if dtype.startswith("torch."):
            dtype = dtype[len("torch."):]
        out[name] = {"shape": shape, "dtype": dtype}
    return out


def time_prep(prep, *, iters: int = PREP_ITERS) -> dict:
    """Both windows of the prep step itself, briefly: what the outside step
    costs, so it is a reading on the ledger rather than a black hole work can be
    moved into. Reported, never gated."""
    e2e = time_e2e(prep, warmup=2, iters=iters)
    dev = time_device(prep, warmup=2, batch=iters, reps=3)
    return {"median_ms": statistics.median(e2e), "device_median_ms": statistics.median(dev)}


def _snapshot(saved: dict) -> dict:
    """A private copy of every prep tensor, taken before the first timed call."""
    return {k: (v.clone() if hasattr(v, "clone") else v) for k, v in saved.items()}


def _same(after, before) -> bool:
    try:
        if hasattr(after, "shape") and hasattr(before, "shape"):
            return bool(torch.equal(after, before))
        return after == before
    except Exception:
        return False


def mutated_saved(saved: dict, snapshot: dict) -> list[str]:
    """Names of prep tensors the timed calls changed — by value, by replacing the
    dict entry, by adding a key, or by hanging an attribute on the tensor. Any
    of them means the timed call kept state across calls in memory that was
    meant to be read-only input; the driver fails the iteration on it."""
    out = []
    for name, before in snapshot.items():
        after = saved.get(name)
        if after is None or not _same(after, before):
            out.append(name)
            continue
        attrs_after = set(vars(after)) if hasattr(after, "__dict__") else set()
        attrs_before = set(vars(before)) if hasattr(before, "__dict__") else set()
        if attrs_after - attrs_before:
            out.append(name)
    out.extend(k for k in saved if k not in snapshot)
    return sorted(set(out))


class _Prep:
    """The prep step's tensors, their snapshot, and the block to report."""

    def __init__(self, prep):
        self.prep = prep
        self.saved = run_prep(prep)
        self.snapshot = _snapshot(self.saved)

    def call(self, fn):
        return lambda: fn(self.saved)

    def block(self) -> dict:
        torch.cuda.synchronize()
        return {"outputs": describe_saved(self.snapshot),
                "mutated": mutated_saved(self.saved, self.snapshot),
                **time_prep(self.prep)}


def _bind_prep(fn, prep):
    """The callable the windows time, and the prep state (or None)."""
    if prep is None:
        return fn, None
    state = _Prep(prep)
    return state.call(fn), state


def _measure(call) -> dict:
    e2e = time_e2e(call)
    dev = time_device(call)
    device_median = statistics.median(dev)
    host_ms = time_host_enqueue(call)
    out = dict(median_ms=statistics.median(e2e), min_ms=min(e2e))
    out["device_median_ms"] = device_median
    out["host_enqueue_ms"] = host_ms
    ratio = host_bound_ratio(host_ms, device_median)
    if ratio is not None:
        out["host_bound_ratio"] = ratio
    return out


def measure_windows(fn, *, prep=None) -> dict:
    """Both windows plus the device window's precondition check, for one callable.

    Keys: "median_ms", "min_ms" (e2e), "device_median_ms", and the check —
    "host_enqueue_ms" and "host_bound_ratio". The e2e and device numbers are
    computed exactly as they always were; the new keys are additive, so historical
    readings stay comparable.

    The e2e keys keep their historical un-prefixed names because every consumer
    (parse_eval, ref-baseline archiving, the records repo) already reads them.

    ``prep``: the window-outside step (module doc). When given it runs ONCE
    before any window, ``fn`` is called as ``fn(saved)`` with what it returned,
    and the result carries a "prep" block — ``outputs`` (name / shape / dtype of
    every tensor it produced), ``mutated`` (names the timed calls changed; must
    be empty) plus the step's own ``median_ms`` / ``device_median_ms`` — for the
    bench to place in the eval JSON verbatim.
    """
    call, state = _bind_prep(fn, prep)
    out = _measure(call)
    if state is not None:
        out["prep"] = state.block()
    return out


def measure_windows_rotating(fns, *, prep=None) -> dict:
    """measure_windows, but successive calls rotate through ``fns`` — one
    callable per pre-built input buffer (e.g. 4 clones of the input tensor).

    Why this exists: a real pipeline hands the kernel a FRESH upstream tensor
    every call, so its data pointers differ call to call. A bench that reuses
    one fixed tensor can never see host-side machinery that (wrongly) keys on
    pointers — a pointer-keyed dispatch cache turned a 0.4 ms kernel into a
    435 ms/call JIT recompile the first time it ran inside a larger pipeline
    while its own bench read 0.4 ms throughout. Taking the perf reading through
    this rotation (2-4 buffers suffice) makes a pointer-dependent host path show
    up as a timing explosion instead of hiding until the kernel is composed.
    """
    import itertools

    it = itertools.cycle(list(fns))
    return measure_windows(lambda *saved: next(it)(*saved), prep=prep)


# --- paired in-job reference: same job, same warm plateau ---------------------
#
# When the reference rides the student job (loop.toml ref_in_job), the verdict
# currency is the SAME-JOB ratio ref / student in the gate window. Measuring the
# two sides back to back is not enough to make that ratio fair: the device window
# drifts with the GPU's clock and thermal state, and the drift is systematic in
# whichever side is measured first.
#
#   - The FIRST window of a job is the fastest, whoever owns it: the part is cold
#     and at full boost. The same kernel read ~3-5% faster in a job's first window
#     than in the next three.
#   - Later windows drift with temperature. One kernel timed twice in the same
#     process, once as "the kernel" and once as "the reference", read ~2.7% apart —
#     more than half of a 5% equivalence margin, in the first-measured side's favour.
#
# Two mechanisms, both here:
#
#   burn-in   heat the part with the kernel under test for a few seconds BEFORE any
#             measured window, so every window sits on the same warm plateau;
#   mirroring measure ref, kernel, kernel, ref (A B B A) and give each side the
#             mean of its two device readings — both sides then occupy the same
#             mean position in whatever drift remains.
#
# Mirroring alone only moves the cold-start bias to the other side; burn-in alone
# leaves the slow thermal drift. Together, the same kernel measured on both sides
# reads a self-ratio of ~1.00.
#
# A reference failure (absent, raising) leaves the kernel readings intact and the
# result carries no reference block — the bench then emits no ref_* keys, per the
# single-ref contract above.

BURN_IN_S = 4.0            # seconds of the kernel under test before the first window
BURN_IN_CALLS_PER_ROUND = 20


def burn_in(fn, *, seconds: float = BURN_IN_S,
            calls_per_round: int = BURN_IN_CALLS_PER_ROUND) -> int:
    """Run ``fn`` back to back for ``seconds`` of wall time; returns the rounds run.

    Use the kernel under test, not the reference: the kernel is what the run is
    measuring, and heating with the reference would hand the reference a warm
    JIT / cache state the kernel does not get.
    """
    if seconds <= 0:
        return 0
    t0 = time.perf_counter()
    rounds = 0
    while time.perf_counter() - t0 < seconds:
        for _ in range(calls_per_round):
            fn()
        torch.cuda.synchronize()
        rounds += 1
    return rounds


def _burn_in_with(fn, prep, seconds: float) -> int:
    """``burn_in`` on the kernel under test, fed the prep step's tensors when the
    card has one: a prep-taking callable is ``fn(saved)``, and calling it bare
    raised before the first window. The burn-in's own prep run stays outside
    every window and is never reported."""
    if seconds <= 0:
        return 0
    call, _state = _bind_prep(fn, prep)
    return burn_in(call, seconds=seconds)


def _as_fns(fns) -> list:
    if fns is None:
        return []
    if callable(fns):
        return [fns]
    return list(fns)


def _measure_side(fns: list, prep) -> dict:
    if len(fns) == 1:
        return measure_windows(fns[0], prep=prep)
    return measure_windows_rotating(fns, prep=prep)


def _mirrored(first: dict, second: dict) -> dict:
    """One side's reading from its mirrored pair.

    The e2e keys and the host-enqueue reading keep the FIRST measurement (their
    historical shape: one window, host tax included); the device reading — the
    gate window — is the mean of the two, and ``host_bound_ratio`` is recomputed
    against that mean so the precondition check reads the same number the gate
    does. ``device_readings_ms`` keeps both raw readings for the job log.
    """
    out = dict(first)
    readings = [first["device_median_ms"], second["device_median_ms"]]
    out["device_median_ms"] = sum(readings) / 2.0
    out["device_readings_ms"] = readings
    ratio = host_bound_ratio(out.get("host_enqueue_ms"), out["device_median_ms"])
    if ratio is not None:
        out["host_bound_ratio"] = ratio
    return out


def measure_paired(kernel_fns, ref_fns=None, *, prep=None,
                   burn_in_s: float = BURN_IN_S) -> dict:
    """Kernel and in-job reference on one warm plateau: burn-in, then A B B A.

    ``kernel_fns`` / ``ref_fns``: a callable, or a list of callables to rotate
    through (one per pre-built input buffer, as ``measure_windows_rotating``).
    ``ref_fns`` may be None (no reference rides this job). ``prep`` as in
    ``measure_windows``, applied to every window of both sides.

    Returns::

        {"kernel": <measure_windows dict, device = mean of its two readings>,
         "ref":    <same for the reference> | None,
         "ref_error": None | "<why the reference produced no reading>",
         "order": ["ref", "kernel", "kernel", "ref"],   # the windows actually run
         "burn_in_rounds": <int>}

    A reference that raises in either of its windows yields ``ref = None`` with
    the exception in ``ref_error`` — never a half pair, never a stand-in.
    """
    kernels = _as_fns(kernel_fns)
    if not kernels:
        raise ValueError("measure_paired needs at least one kernel callable")
    refs = _as_fns(ref_fns)
    order: list[str] = []
    rounds = _burn_in_with(kernels[0], prep, burn_in_s)

    ref_first = ref_error = None
    if refs:
        try:
            ref_first = _measure_side(refs, prep)
            order.append("ref")
        except Exception as exc:  # noqa: BLE001 — the reference is a foreign callable
            ref_error = f"{type(exc).__name__}: {exc}"
    kernel_first = _measure_side(kernels, prep)
    order.append("kernel")
    kernel_second = _measure_side(kernels, prep)
    order.append("kernel")
    ref_second = None
    if ref_first is not None:
        try:
            ref_second = _measure_side(refs, prep)
            order.append("ref")
        except Exception as exc:  # noqa: BLE001
            ref_error = f"{type(exc).__name__}: {exc}"

    ref = None
    if ref_first is not None and ref_second is not None:
        ref = _mirrored(ref_first, ref_second)
    elif refs and ref_error is None:
        ref_error = "reference produced no reading"
    return {
        "kernel": _mirrored(kernel_first, kernel_second),
        "ref": ref,
        "ref_error": ref_error,
        "order": order,
        "burn_in_rounds": rounds,
    }


def describe_paired(paired: dict) -> str:
    """One log line with the four raw device readings in the order they ran."""
    k = paired["kernel"].get("device_readings_ms", [paired["kernel"]["device_median_ms"]] * 2)
    r = (paired["ref"] or {}).get("device_readings_ms", [None, None])

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    line = (f"mirrored device readings (ms): ref {fmt(r[0])} / kernel {fmt(k[0])} / "
            f"kernel {fmt(k[1])} / ref {fmt(r[1])}")
    if paired.get("ref_error"):
        line += f"  [reference: {paired['ref_error']}]"
    return line


def paired_performance(paired: dict, *, e2e_key: str = "bwd_median_ms",
                       ndigits: int = 4) -> dict:
    """The eval-JSON ``performance`` keys from a ``measure_paired`` result.

    Emits the student's two windows plus the precondition check; adds the
    ``ref_*`` / ``vs_*`` keys only when the reference produced a mirrored pair
    (single-ref contract: no reference, no ref keys). Ratios are ref / student,
    so above 1.0 means the student is faster, same window on both sides. The
    bench adds its own derived keys (MFU, TFLOPS) on top.
    """
    k = paired["kernel"]
    out = {
        e2e_key: round(k["median_ms"], ndigits),
        "min_ms": round(k["min_ms"], ndigits),
        "device_median_ms": round(k["device_median_ms"], ndigits),
        "device_readings_ms": [round(v, ndigits) for v in k.get("device_readings_ms", [])],
        "host_enqueue_ms": round(k["host_enqueue_ms"], ndigits),
        "host_bound_ratio": (round(k["host_bound_ratio"], 3)
                             if k.get("host_bound_ratio") is not None else None),
        "in_job_ref": "paired" if paired.get("ref") else (paired.get("ref_error") or "absent"),
    }
    r = paired.get("ref")
    if r:
        out.update({
            "ref_median_ms": round(r["median_ms"], ndigits),
            "vs_ref": round(r["median_ms"] / k["median_ms"], ndigits),
            "ref_device_ms": round(r["device_median_ms"], ndigits),
            "vs_ref_device": round(r["device_median_ms"] / k["device_median_ms"], ndigits),
            "ref_device_readings_ms": [round(v, ndigits) for v in r.get("device_readings_ms", [])],
            "ref_host_enqueue_ms": round(r["host_enqueue_ms"], ndigits),
            "ref_host_bound_ratio": (round(r["host_bound_ratio"], 3)
                                     if r.get("host_bound_ratio") is not None else None),
        })
    return out


# --- several workloads of one shape: the perf verdict is their geometric mean --
#
# A kernel tuned against ONE input set learns that input set: the tiling, the
# split points, the per-CTA work assignment and every constant that divides the
# work end up fitted to the one campaign point. Re-measured on other workloads
# of the SAME fixed shape (same tensor shapes, different contents — for a varlen
# op, other sequence splits of the same token count; for sparse attention, other
# index tables; for dense ops, other value distributions), kernels that read
# above 1.0x at their tuning point have read well below it, and a near-peak
# attention kernel lost its whole margin on a key-scale ramp its speculative
# fast path had never been timed on.
#
# So a card times PERF_WORKLOADS (three) representative workloads of its fixed
# shape, each paired with the in-job reference exactly as a single-input card
# would be, and reports the aggregate in the SAME top-level keys the driver
# already gates on: the geometric mean of each side's per-workload readings.
# The ratio of the two geometric means IS the geometric mean of the per-workload
# ratios, so every ms- and ratio-based consumer downstream (best, plateau,
# target, confirmation, re-anchor, patch judging, lever pricing) reads the
# aggregate without knowing there are three. The per-workload readings ride
# along under ``workloads`` for the log, the teacher and the confirmation record.
#
# Pick the three to span the shape's real variation, not three seeds of one
# draw: e.g. the campaign workload, the heaviest, and the most skewed split of
# the same token count; for dense ops, the campaign draw plus two value
# distributions (sharp logits, a dominant sink key, a rising key scale, real
# activations). The FIRST entry is the campaign workload: ``eval/ncu_run.py``
# profiles it, and correctness replays it first.

PERF_WORKLOADS = 3


def _geomean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals or any(v <= 0 for v in vals):
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def _workload_prep(prep, name: str):
    """``prep`` bound to one workload: a zero-argument step the windows run like
    any single-input card's, producing THIS workload's tensors."""
    def bound():
        return prep(name)
    return bound


def _check_workload_prep(prep) -> None:
    """A multi-workload prep step takes the workload's name. A zero-argument one
    can only close over one workload's inputs, and every other workload's timed
    call would then receive tensors derived from the wrong input set."""
    try:
        inspect.signature(prep).bind("workload")
    except TypeError:
        raise TypeError(
            "measure_windows_workloads calls prep(name) once per workload, so each "
            "workload's timed call receives tensors derived from ITS inputs; this prep "
            "takes no argument and would hand every workload the same tensors — give "
            "it a workload-name parameter (eval/ncu_run.py calls it with the campaign "
            "workload's name)") from None
    except ValueError:   # no introspectable signature: trust the call itself
        pass


def measure_windows_workloads(workloads, *, prep=None,
                              burn_in_s: float = BURN_IN_S) -> dict:
    """``measure_paired`` over several workloads of one fixed shape.

    ``workloads``: a sequence of ``(name, kernel_fns, ref_fns)`` — each entry is
    exactly what ``measure_paired`` takes for one input set (``ref_fns`` may be
    None when no reference rides the job). Names must be unique; the first entry
    is the campaign workload.

    ``prep``: the window-outside step (module doc), a module-level def in
    bench.py that takes the WORKLOAD NAME — ``prep(name) -> {tensor name:
    tensor}`` — because each workload's symmetrized tensors derive from its own
    inputs. It runs outside every window of every workload, each workload's
    kernel and reference receive that workload's tensors, and
    ``workloads_performance`` reports one ``prep`` block holding every
    workload's (``eval/ncu_run.py`` calls it with the campaign workload's name
    before the profiler starts).

    The part is burned in ONCE, with the first workload's kernel (fed its prep
    tensors); each workload then runs its own mirrored ref / kernel / kernel /
    ref. Returns::

        {"workloads": [<measure_paired result + "name">, ...],
         "burn_in_rounds": <int>}

    Map it with ``workloads_performance``.
    """
    items = [(str(n), k, r) for n, k, r in workloads]
    if not items:
        raise ValueError("measure_windows_workloads needs at least one workload")
    names = [n for n, _, _ in items]
    if len(set(names)) != len(names):
        raise ValueError(f"workload names must be unique: {names}")
    first = _as_fns(items[0][1])
    if not first:
        raise ValueError(f"workload {names[0]!r} has no kernel callable")
    if prep is not None:
        _check_workload_prep(prep)
    preps = {n: (_workload_prep(prep, n) if prep is not None else None) for n in names}
    rounds = _burn_in_with(first[0], preps[names[0]], burn_in_s)
    results = []
    for name, kernel_fns, ref_fns in items:
        paired = measure_paired(kernel_fns, ref_fns, prep=preps[name], burn_in_s=0.0)
        paired["name"] = name
        results.append(paired)
    return {"workloads": results, "burn_in_rounds": rounds}


def _workloads_prep_block(runs: list) -> dict | None:
    """One eval-JSON ``prep`` block from every workload's kernel side, or None.

    Top level: the campaign (first) workload's block — its outputs and the prep
    step's cost — with ``mutated`` the union over all workloads. ``workloads``:
    each workload's own block with its ``name``; the driver holds EVERY one of
    them to window.toml (``forgekernel.window_contract.check_prep_report``), so
    a workload whose prep produced another shape, dtype or tensor set fails the
    iteration the way the first one would.
    """
    blocks = [(p.get("name"), p["kernel"].get("prep")) for p in runs]
    blocks = [(n, b) for n, b in blocks if isinstance(b, dict)]
    if not blocks:
        return None
    top = dict(blocks[0][1])
    top["mutated"] = sorted({str(m) for _, b in blocks for m in (b.get("mutated") or [])})
    top["workloads"] = [{"name": n, **b} for n, b in blocks]
    return top


def workloads_performance(result: dict, *, e2e_key: str = "bwd_median_ms",
                          ndigits: int = 4) -> dict:
    """The eval-JSON ``performance`` keys from ``measure_windows_workloads``.

    Top level: the SAME keys ``paired_performance`` emits, each the geometric
    mean over the workloads (``host_bound_ratio`` / ``ref_host_bound_ratio``: the
    worst workload, so the precondition audit is never softened by averaging).
    The ``ref_*`` / ``vs_*`` keys appear only when EVERY workload produced a
    mirrored reference pair — a ratio over a subset would silently re-weight the
    verdict. Plus ``aggregate`` = "geomean", ``n_workloads``, and ``workloads``:
    one ``paired_performance`` block per workload, carrying its ``name``. With a
    prep step, ``prep``: every workload's block in one (``_workloads_prep_block``).
    """
    runs = result["workloads"]
    kernels = [p["kernel"] for p in runs]
    k_e2e = _geomean(k["median_ms"] for k in kernels)
    k_dev = _geomean(k["device_median_ms"] for k in kernels)
    hb = [k["host_bound_ratio"] for k in kernels if k.get("host_bound_ratio") is not None]
    out = {
        e2e_key: round(k_e2e, ndigits),
        "min_ms": round(_geomean(k["min_ms"] for k in kernels), ndigits),
        "device_median_ms": round(k_dev, ndigits),
        "host_enqueue_ms": round(_geomean(k["host_enqueue_ms"] for k in kernels), ndigits),
        "host_bound_ratio": round(max(hb), 3) if hb else None,
        "aggregate": "geomean",
        "n_workloads": len(runs),
    }
    refs = [p.get("ref") for p in runs]
    if all(refs):
        ref_e2e = _geomean(r["median_ms"] for r in refs)
        ref_dev = _geomean(r["device_median_ms"] for r in refs)
        rhb = [r["host_bound_ratio"] for r in refs if r.get("host_bound_ratio") is not None]
        out.update({
            "in_job_ref": "paired",
            "ref_median_ms": round(ref_e2e, ndigits),
            "vs_ref": round(ref_e2e / k_e2e, ndigits),
            "ref_device_ms": round(ref_dev, ndigits),
            "vs_ref_device": round(ref_dev / k_dev, ndigits),
            "ref_host_enqueue_ms": round(_geomean(r["host_enqueue_ms"] for r in refs), ndigits),
            "ref_host_bound_ratio": round(max(rhb), 3) if rhb else None,
        })
    else:
        missing = [str(p.get("name")) for p in runs if not p.get("ref")]
        errors = [p["ref_error"] for p in runs if p.get("ref_error")]
        if len(missing) == len(runs):
            out["in_job_ref"] = errors[0] if errors else "absent"
        else:
            out["in_job_ref"] = f"partial: no reference pair on {', '.join(missing)}"
    if (prep := _workloads_prep_block(runs)) is not None:
        out["prep"] = prep
    per = []
    for paired in runs:
        block = paired_performance(paired, e2e_key=e2e_key, ndigits=ndigits)
        block["name"] = paired.get("name")
        per.append(block)
    out["workloads"] = per
    return out


def describe_workloads(result: dict) -> str:
    """One log line per workload: its name and mirrored device readings."""
    return "\n".join(f"[{p.get('name')}] {describe_paired(p)}" for p in result["workloads"])


# --- graph window: the reading to use when the host is the pipeline -----------
#
# ADDITIVE, never a replacement for time_device. The two windows above keep their
# algorithms so every historical number stays comparable; this one exists for the
# case they cannot answer — a reference whose python wrapper is thicker than its
# kernel. Capturing the call into a CUDA graph and replaying it removes the host
# from the loop entirely, so what the replay costs is the device work and nothing
# else.
#
# Capture fails in ordinary, legitimate ways (a callee that syncs, allocates from
# the caller's stream, or runs on a non-capturable path), and it fails in one
# dangerous way: it "succeeds" and produces an EMPTY graph. PyTorch only warns
# ("The CUDA Graph is empty..."), and replaying nothing is very fast — a live
# CuTe DSL kernel captured empty and replayed in 0.0013 ms against its true
# 0.4 ms. An unchecked graph reading is therefore worse than no graph reading, so
# every failure below returns available=False with a reason, and a plausibility
# floor rejects readings that are too fast to be measurements.


def graph_unavailable(reason: str) -> dict:
    """The negative verdict, in the shape callers branch on."""
    return dict(available=False, device_median_ms=None, samples_ms=[], reason=reason)


def empty_capture_reason(caught) -> str:
    """The warning text if PyTorch said the capture produced an empty graph."""
    for record in caught or ():
        text = str(getattr(record, "message", record))
        if "graph is empty" in text.lower():
            return text.strip()
    return ""


def implausible_graph_reason(graph_ms, reference_ms,
                             share: float = GRAPH_MIN_PLAUSIBLE_SHARE) -> str:
    """Why a replay reading cannot be believed, or "" when it can.

    ``reference_ms`` is the ordinary device reading for the same callable. Removing
    the host can only take a measurement DOWN to the device time it was already
    hiding, so a replay far below that is not a faster measurement — it is a graph
    that captured little or nothing. Without a reference nothing can be checked and
    the reading is accepted (the caller then owns the plausibility).
    """
    if not reference_ms or not graph_ms:
        return ""
    floor = reference_ms * share
    if graph_ms - floor >= 0:
        return ""
    return (f"replay {graph_ms:.6f} ms is under {share:.0%} of the {reference_ms:.6f} ms "
            f"device reading — that is an empty or near-empty capture, not a measurement")


def time_device_graph(fn, *, warmup: int = WARMUP_ITERS, batch: int = DEVICE_BATCH,
                      reps: int = DEVICE_REPS, reference_ms=None) -> dict:
    """Per-call ms from CUDA-graph replay, with the host removed. Optional reading.

    Returns a dict: ``available`` (bool), ``device_median_ms`` (None unless
    available), ``samples_ms``, ``reason`` (why it is unavailable, or how it was
    validated). Never raises and never returns a number it could not defend —
    a caller that gets available=False must say so rather than fall back silently.

    ``reference_ms`` should be the callable's ordinary ``time_device`` median; it
    is what the plausibility floor is measured against.

    Use it for a REFERENCE whose host_bound_ratio says the pipelined window is
    measuring python. Do not use it as the student's gate reading: a kernel that
    only performs inside a graph has a host problem the deployment still pays.
    """
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        return graph_unavailable("this torch build exposes no CUDA graph capture API")
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
    except Exception as exc:
        return graph_unavailable(f"pre-capture warmup raised {type(exc).__name__}: {exc}")

    graph = torch.cuda.CUDAGraph()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.cuda.graph(graph):
                fn()
    except Exception as exc:
        return graph_unavailable(f"capture raised {type(exc).__name__}: {exc}")

    if empty := empty_capture_reason(caught):
        return graph_unavailable(f"capture produced an EMPTY graph ({empty})")

    try:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(reps):
            start.record()
            for _ in range(batch):
                graph.replay()
            stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) / batch)
    except Exception as exc:
        return graph_unavailable(f"replay raised {type(exc).__name__}: {exc}")

    median = statistics.median(samples)
    if bad := implausible_graph_reason(median, reference_ms):
        return graph_unavailable(f"capture looks empty: {bad}")
    note = "CUDA-graph replay (host removed)"
    if reference_ms:
        note += f"; {median / reference_ms:.3f}x the pipelined device reading"
    return dict(available=True, device_median_ms=median, samples_ms=samples, reason=note)


def measure_windows_with_graph(fn, *, prep=None) -> dict:
    """``measure_windows`` plus a graph-replay reading when one can be defended.

    Adds "graph_device_median_ms" (only when the capture was validated) and always
    "graph_status" — the reason string, so a bench that could not get a graph
    reading reports WHY instead of quietly reporting one window less. Intended for
    ref-measurement scripts (``ref_cmd``); the driver archives the graph reading and
    anchors target on it when the ordinary device window is host-bound. ``prep``
    as in ``measure_windows``: the graph captures the timed call only.
    """
    call, state = _bind_prep(fn, prep)
    out = _measure(call)
    graph = time_device_graph(call, reference_ms=out.get("device_median_ms"))
    out["graph_status"] = graph["reason"]
    if graph["available"]:
        out["graph_device_median_ms"] = graph["device_median_ms"]
    if state is not None:
        out["prep"] = state.block()      # after the replays: they read the tensors too
    return out
