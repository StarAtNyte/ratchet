"""PM-sampling phase timeline — the time dimension the stall ledger can't give.

`stall_tools` prices WHERE the time leaks (per instruction, summed over the
whole run); this tool reads WHEN — the PM-sampling timeseries out of a
`.ncu-rep` collected with the PmSampling sections, rendered as downsampled
per-series timelines plus a phase summary (ramp / plateau / drain spans).
Phase-shaped hypotheses — ping-pong anti-phasing, a serial section stretching
the loop, a tail effect — are exactly the class the per-PC ledger disperses:
each phase touches the same PCs, so the ledger sums them flat.

Ledger contract (same trust model as STALL-OK):
  * the tables sit between `=== PM TIMELINE BEGIN ===` / `=== PM TIMELINE END ===`
    marker lines so a driver can lift them into a prompt verbatim;
  * exactly one machine verdict line ends the run:
        PMTL-OK kernel=<name> series=<N> stall_series=<M> stall_samples=<K>
        PMTL-FAIL reason=<enum> ...
    printed only by this tool after its self-checks (PM series present at all,
    stall series populated, per-series sample floor against a too-coarse
    interval). Prose cannot open the gate.

Collection recipe (measured on H100 / ncu 2025.4; frozen in
`pm_timeline_run.py`):

    ncu --profile-from-start off --launch-count 1 --kill yes -f -o /tmp/pmtl \\
        --section PmSampling --section PmSampling_WarpStates \\
        --pm-sampling-interval 8192 --pm-sampling-buffer-size 268435456 \\
        python3 eval/ncu_run.py

`--pm-sampling-interval` is in GPU cycles and the WARP-STALL series follow it
directly: 65536 (the coarse default) gives ~45 stall samples across a ~1.7 ms
kernel — too flat to see phase structure — while 8192 gives ~370. The
`*_realtime` throughput series sample on their own finer cadence either way.

Reading path: ncu_report's Python API drifts between releases (2025.4 removed
`IMetric.per_instance_values()`, present in 2025.1) and a reader that swallows
accessor errors misreports fully populated reports as empty — that failure
mode hid a month of good PM data once. This reader distinguishes absent /
empty / populated and FAILS LOUDLY (reason=reader-api) on accessor breakage.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

# --- ncu_report locator (inlined: a vendored file travels alone) --------------

def _locate_ncu_report():
    candidates = ["/usr/local/cuda/nsight-compute/extras/python"]
    for root in ("/usr/local", "/opt/nvidia", "/opt/cuda"):
        p = Path(root)
        if not p.is_dir():
            continue
        for pat in ("cuda-*/nsight-compute-*/extras/python",
                    "nsight-compute-*/extras/python",
                    "NVIDIA-Nsight-Compute-*/extras/python",
                    "nsight-compute/*/extras/python"):
            candidates.extend(str(q) for q in p.glob(pat))
    for c in candidates:
        if (Path(c) / "ncu_report.py").exists():
            return c
    return None


def _import_ncu_report():
    try:
        import ncu_report  # noqa: F401
        return ncu_report
    except ImportError:
        found = _locate_ncu_report()
        if found is None:
            return None
        sys.path.insert(0, found)
        import ncu_report  # noqa: F401
        return ncu_report


# --- constants ----------------------------------------------------------------

PM_PREFIX = "pmsampling:"
STALL_MARK = "warps_issue_stalled"

# Full timeline rows are printed for stall series plus these phase-relevant
# throughput anchors; every other populated series gets a one-line stat row.
CORE_SUBSTRINGS = ("pipe_tensor", "inst_executed_realtime",
                   "dram__throughput", "cycles_active")

MIN_STALL_SAMPLES = 64     # fewer = the interval was too coarse for phases
MIN_DURATION_NS = 20_000   # a "dominant" kernel shorter than this is a mis-aim -- unless
                           # the window vouches for it (== stall_tools.window_vouches:
                           # an audit found every one of 1497 PMTL mis-aims to be
                           # the kernel under pricing at 7-20 us; see stall_tools)
WINDOW_MIN_REPEATS = 2     # == stall_tools.WINDOW_MIN_REPEATS
WINDOW_MIN_SHARE = 0.5     # == stall_tools.WINDOW_MIN_SHARE
BUCKETS = 100
MAX_TIMELINE_ROWS = 14

BEGIN = "=== PM TIMELINE BEGIN ==="
END = "=== PM TIMELINE END ==="


class MetricReadError(RuntimeError):
    """An accessor failed on a metric that exists — ncu_report API breakage
    (or a corrupt report), NOT missing data. Never render it as empty."""


def short_series(name: str) -> str:
    s = name[len(PM_PREFIX):] if name.startswith(PM_PREFIX) else name
    return s.replace("smsp__warps_issue_stalled_", "stall:")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]", "_", name)[:48] or "unknown"


# --- reading (three-state: absent / empty / populated; loud on API drift) -----

def _series_values(m) -> list:
    try:
        n = m.num_instances()
    except Exception as e:
        raise MetricReadError(f"num_instances() failed: {e!r}") from e
    if n == 0:
        return []
    # kind()/ValueKind are loop invariants AND part of the drift-prone API
    # surface — resolve them once, inside the loud path: swallowed here they
    # would masquerade as "no PM series", the exact failure mode this module
    # exists to prevent.
    try:
        as_int = m.kind() == m.ValueKind_UINT64
    except Exception as e:
        raise MetricReadError(f"kind()/ValueKind lookup failed: {e!r}") from e
    out = []
    for i in range(n):
        try:
            out.append(float(m.as_uint64(i)) if as_int else float(m.as_double(i)))
        except Exception as e:
            raise MetricReadError(f"instance accessor failed at {i}: {e!r}") from e
    return out


def collect_pm_series(action) -> "OrderedDict[str, list]":
    """name -> values for every populated pmsampling series, stalls first.

    Every failure on an ENUMERATED metric raises MetricReadError — a metric
    that metric_names() lists but the accessors cannot read is API drift, and
    the one contract this module must keep is that drift is loud, never an
    empty result.
    """
    populated = []
    for name in action.metric_names():
        if not name.lower().startswith(PM_PREFIX):
            continue
        try:
            m = action[name]
        except Exception as e:
            raise MetricReadError(
                f"{name}: __getitem__ failed on an enumerated metric: {e!r}") from e
        vals = _series_values(m)
        if vals:
            populated.append((name, vals))
    populated.sort(key=lambda kv: (STALL_MARK not in kv[0], kv[0]))
    return OrderedDict(populated)


def _duration(action):
    try:
        return action["gpu__time_duration.sum"].value() or 0
    except Exception:
        return 0


def pick_dominant_action(report, kernel_substr: str = ""):
    """Longest-duration captured launch (optionally name-filtered) — selection,
    not launch counting (which mis-aims onto setup launches)."""
    rng = report.range_by_idx(0)
    best, best_d = None, -1
    for i in range(rng.num_actions()):
        a = rng.action_by_idx(i)
        if kernel_substr and kernel_substr.lower() not in a.name().lower():
            continue
        d = _duration(a)
        if d > best_d:
            best, best_d = a, d
    return best


def window_vouches(report, action) -> "tuple[bool, str]":
    """Does the profiler window vouch for a short dominant kernel (stall_tools'
    rule, inlined: this file travels alone)? A repeated launch holding most of the
    captured GPU time is the kernel the window was opened around."""
    rng = report.range_by_idx(0)
    acts = [rng.action_by_idx(i) for i in range(rng.num_actions())]
    total = sum(_duration(a) for a in acts)
    same = [a for a in acts if a.name() == action.name()]
    share = sum(_duration(a) for a in same) / total if total else 0.0
    ok = len(same) >= WINDOW_MIN_REPEATS and share >= WINDOW_MIN_SHARE
    return ok, (f"{len(same)} launch(es) of it, {100 * share:.0f}% of the captured GPU "
                f"time (vouches at >= {WINDOW_MIN_REPEATS} and "
                f">= {100 * WINDOW_MIN_SHARE:.0f}%)")


# --- pure rendering core (testable without a report) --------------------------

def phase_stats(vals: list) -> dict:
    """Active span + coarse phase shape of one timeseries.

    active = indices where the value exceeds 1% of the series max (leading and
    trailing dead air excluded); plateau = mean over the middle half of the
    active span; ramp/drain = the fraction of the active span spent before the
    value first reaches / after it last holds 90% of the plateau — i.e. how
    much of the kernel's active time is startup and tail. A sawtooth or serial
    section shows up as a mid-span dip in the printed value row, not in these
    numbers.
    """
    mx = max(vals) if vals else 0.0
    if mx <= 0:
        return {"n": len(vals), "active": (0, -1), "max": 0.0,
                "plateau": 0.0, "ramp": 0.0, "drain": 0.0}
    thresh = mx * 0.01
    idx = [i for i, v in enumerate(vals) if v > thresh]
    lo, hi = idx[0], idx[-1]
    act = vals[lo:hi + 1]
    q = max(1, len(act) // 4)
    mid = act[q:len(act) - q] or act
    plateau = sum(mid) / len(mid)
    level = plateau * 0.9
    first = next((i for i, v in enumerate(act) if v >= level), len(act) - 1)
    last = next((i for i in range(len(act) - 1, -1, -1) if act[i] >= level), 0)
    ramp = first / len(act)
    drain = (len(act) - 1 - last) / len(act)
    return {"n": len(vals), "active": (lo, hi), "max": mx,
            "plateau": plateau, "ramp": ramp, "drain": drain}


def _bucketize(vals: list, buckets: int) -> list:
    step = max(1, len(vals) // buckets)
    out = []
    for i in range(0, len(vals), step):
        chunk = vals[i:i + step]
        out.append(sum(chunk) / len(chunk))
    return out


def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")


def build_timeline(series: "OrderedDict[str, list]", out=print,
                   buckets: int = BUCKETS) -> "tuple[int, int, int]":
    """Print the timeline tables; return (series, stall_series, stall_samples).

    Full downsampled rows for stall + core-anchor series (capped at
    MAX_TIMELINE_ROWS); a one-line stat row for everything else, so nothing
    populated is silently absent from the block.
    """
    stall_names = [n for n in series if STALL_MARK in n]
    stall_samples = max((len(series[n]) for n in stall_names), default=0)

    def wants_row(name):
        return STALL_MARK in name or any(s in name for s in CORE_SUBSTRINGS)

    rows = 0
    out(f"[timeline] {len(series)} populated pmsampling series, "
        f"{len(stall_names)} stall series (max {stall_samples} samples)")
    for name, vals in series.items():
        st = phase_stats(vals)
        head = (f"{short_series(name)} n={st['n']} "
                f"active=[{st['active'][0]},{st['active'][1]}] "
                f"max={_fmt(st['max'])} plateau={_fmt(st['plateau'])} "
                f"ramp={st['ramp']:.2f} drain={st['drain']:.2f}")
        if wants_row(name) and rows < MAX_TIMELINE_ROWS and st["max"] > 0:
            rows += 1
            out(f"PM {head}")
            out("   " + " ".join(_fmt(v) for v in _bucketize(vals, buckets)))
        else:
            out(f"PM- {head}")
    return len(series), len(stall_names), stall_samples


def tensor_active_plateau(series: "OrderedDict[str, list]"):
    """Steady-state tensor-pipe activity (pct) off the PM series, or None.

    The role exposure ledger's scale factor: the plateau of the
    ``pipe_tensor_cycles_active`` series is the mainloop's tensor duty cycle,
    which the whole-kernel aggregate dilutes with prologue and drain."""
    for name, vals in series.items():
        if "pipe_tensor_cycles_active" in name and vals:
            st = phase_stats(vals)
            if st["max"] > 0:
                return float(st["plateau"])
    return None


# --- the full run -------------------------------------------------------------

def timeline_report(report_path, kernel_substr: str = "",
                    buckets: int = BUCKETS, out=print) -> bool:
    """The full timeline run off one .ncu-rep; True iff PMTL-OK was printed."""
    ncu_report = _import_ncu_report()
    if ncu_report is None:
        out("PMTL-FAIL reason=no-ncu-report-module (Nsight extras/python not found)")
        return False
    try:
        report = ncu_report.load_report(str(report_path))
    except Exception as e:
        out(f"PMTL-FAIL reason=report-unreadable ({e})")
        return False
    action = pick_dominant_action(report, kernel_substr)
    if action is None:
        out("PMTL-FAIL reason=no-kernel-matched")
        return False
    dur = _duration(action)
    out(f"[sanity] kernel={action.name()[:80]}")
    out(f"[sanity] duration_ns={dur} — dominant across every captured launch")
    if dur < MIN_DURATION_NS:
        ok, ev = window_vouches(report, action)
        if not ok:
            out(f"PMTL-FAIL reason=mis-aim (dominant kernel {dur}ns < "
                f"{MIN_DURATION_NS}ns and the window does not vouch for it: {ev} "
                f"— the main kernel was not captured)")
            return False
        out(f"[sanity] short kernel ({dur}ns < {MIN_DURATION_NS}ns) accepted: {ev}")

    try:
        series = collect_pm_series(action)
    except MetricReadError as e:
        out(f"PMTL-FAIL reason=reader-api ({e} — ncu_report accessor drift, "
            f"NOT missing data; fix the reader, do not waive)")
        return False
    if not series:
        out("PMTL-FAIL reason=no-pm-series (no populated pmsampling: metrics — "
            "were --section PmSampling --section PmSampling_WarpStates passed?)")
        return False

    out(BEGIN)
    n_series, n_stall, stall_samples = build_timeline(series, out=out,
                                                      buckets=buckets)
    out(END)
    if n_stall == 0:
        out("PMTL-FAIL reason=no-stall-series (PmSampling_WarpStates section "
            "missing or dead on this image)")
        return False
    if stall_samples < MIN_STALL_SAMPLES:
        out(f"PMTL-FAIL reason=interval-too-coarse (max {stall_samples} stall "
            f"samples < {MIN_STALL_SAMPLES} — stall series sample AT the "
            f"requested interval, so a fast kernel starves at the default; "
            f"recollect FINER, e.g. quarter the interval (2048 resolves a "
            f"~0.3 ms kernel), and/or widen the window — PM samples accumulate "
            f"across every launch profiled, so --launch-count 8 multiplies "
            f"the sample count. Both are knobs: pm_timeline_run.py --interval "
            f"N / stall_price_run.py --pm --pm-interval N)")
        return False
    out(f"PMTL-OK kernel={_sanitize(action.name())} series={n_series} "
        f"stall_series={n_stall} stall_samples={stall_samples}")
    return True


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True)
    ap.add_argument("--kernel-substr", default="")
    ap.add_argument("--buckets", type=int, default=BUCKETS)
    args = ap.parse_args(argv)
    ok = timeline_report(args.report, kernel_substr=args.kernel_substr,
                         buckets=args.buckets)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
