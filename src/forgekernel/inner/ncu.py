"""NCU profiler parsing + per-iteration metric-evolution trace.

The inner student loop must record NCU hardware metrics EVERY iteration (not the
empty ``profiler={}`` default), so the meta agent classifies the binding
constraint from measured truth instead of the student's prose self-report. The
two failure modes a milestone diagnosis must tell apart are visible directly in
these metrics:

  * ``bankconf_st`` high  -> a degenerate / conflicting smem layout (G3: the
    milestone names the decision but the student shipped a conflict-prone
    fallback because it lacks a verified KB recipe);
  * ``smem_wavefronts_*`` far above the arch ref -> a scalar / per-element smem
    access path (the buffer's copy atom, not its swizzle: a conflict-free
    per-element path still multiplies wavefront volume several-fold);
  * ``spill_*_sectors`` high + ``regs_per_thread`` high -> register pressure
    (R1: a pinned foundation caps the per-thread footprint).

``parse_ncu_csv`` -> ``key_metrics`` produces the compact dict stored in each
``IterationRecord.profiler``; ``build_ncu_trace`` / ``render_ncu_trace`` turn the
whole trajectory into the metric-evolution view the meta agent reads.

NCU reports one row set per LAUNCH. Everything here therefore aggregates per
launch before anything else reads it: additive counters are summed over the
launches of a kernel, intensities come from that kernel's dominant launch, and
the launch count travels with the row. Keeping only the last launch (which is
what a plain overwrite does) makes a multi-launch kernel look several times
faster than it is, and the resulting gap between profiled kernel time and the
timed wall clock is indistinguishable from a host-side bottleneck -- so the
under-count would silently corrupt every reconciliation built on these numbers.
"""

from __future__ import annotations

import csv
import io
import re

from forgekernel.paths import operators_dir

# NCU metric name -> (short key, what a high value implies for diagnosis).
# Only these decision-relevant metrics survive into the per-iter profiler dict.
KEY_METRICS: dict[str, str] = {
    "gpu__time_active.sum": "time_ns",
    "launch__registers_per_thread": "regs_per_thread",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": "spill_ld_sectors",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum": "spill_st_sectors",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "bankconf_st",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "bankconf_ld",
    # Shared-memory pipe VOLUME, not collisions. Conflicts count only the replay
    # wavefronts from same-bank collisions; a scalar / per-element access path
    # multiplies TOTAL wavefronts several-fold while staying conflict-free, and
    # no conflict threshold ever flags it (observed on a bwd kernel: the load side ran 6x
    # the ref's wavefronts with near-zero conflicts and stayed invisible for 26
    # iterations). Volume has no absolute "too high" — it is only meaningful
    # against the arch ref, so its flag is the ratio branch in
    # ``summarize_metrics`` (``baseline`` parameter), not ``_THRESHOLDS``.
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": "smem_wavefronts_st",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": "smem_wavefronts_ld",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed": "warp_active_pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "stall_long_sb",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "stall_mio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "stall_barrier",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct",
    # DRAM *volume*, not rate. The percentage above answers "is the bus busy"; these
    # answer "how many bytes crossed it". For a fixed problem shape, total traffic is
    # a function of how many times each operand tile is re-read, so it constrains the
    # tile independently of the launch geometry — which on its own only yields the
    # product (a 132-CTA persistent grid looks identical for 128x256 and 256x128).
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    # Launch attributes — compile/launch facts, not performance. These are what
    # make an M0 architecture gate verifiable: shared memory pins the stage
    # budget the way registers already pin setmaxregister. The agent authors the
    # kernel but cannot author these, so a gate line bound to them is evidence
    # rather than a claim.
    "launch__shared_mem_per_block_static": "smem_static",
    "launch__shared_mem_per_block_dynamic": "smem_dynamic",
}

# The `--metrics` argument for `ncu`. Single source of truth: an operator's
# gpu_entry.sh should request exactly what the parser knows how to read.
NCU_METRICS_ARG = ",".join(KEY_METRICS)

# Launch geometry comes back as standard NCU CSV *columns*, not metrics — no
# `--metrics` request needed, and it is therefore readable from NCU output that
# predates this mapping. Values are dim tuples ("(256, 1, 1)"); stored as the
# product, which is what a gate compares against (threads per CTA pins the
# warpgroup count; total CTAs plus a known problem shape pins the tile).
LAUNCH_COLUMNS: dict[str, str] = {
    "Block Size": "block_size",
    "Grid Size": "grid_size",
}

# Trace columns in display order (subset that exists; missing keys are skipped).
_TRACE_COLUMNS = [
    "time_ns", "regs_per_thread", "bankconf_st", "bankconf_ld",
    "smem_wavefronts_st", "smem_wavefronts_ld",
    "spill_ld_sectors", "spill_st_sectors", "tensor_pct", "warp_active_pct",
    "wgmma_serial_ratio",
    "stall_long_sb", "stall_barrier", "stall_mio", "dram_pct",
    "dram_bytes_read", "dram_bytes_write",
    "block_size", "block_size_dims", "grid_size", "grid_size_dims",
    "smem_static", "smem_dynamic",
    "dominant_launches", "kernel_count", "total_launches", "total_time_ns",
    "dominant_time_pct",
]

TIME_METRIC = "gpu__time_active.sum"

# Pseudo-metric key ``parse_ncu_csv`` writes next to the real ones: how many
# launches of that kernel the profiled window contained. It cannot collide with
# an NCU metric name (those all carry a ``__`` or a ``.``).
LAUNCH_COUNT_KEY = "launches"


def _is_additive(metric: str) -> bool:
    """Does this metric ADD UP across the launches of one kernel?

    NCU emits one row set per launch. Counters (``.sum`` -- device time, sectors,
    bank conflicts, wavefronts, DRAM bytes) are per-launch totals, so the window's
    total is their sum. Everything else in ``KEY_METRICS`` is an intensity or a
    launch attribute (``.pct_of_peak_sustained_elapsed``, ``.ratio``, ``launch__*``):
    summing those would be meaningless, so they are taken from the launch that
    dominates the kernel's own runtime instead.
    """
    return metric.endswith(".sum")


def _to_number(value: str) -> float | None:
    """NCU CSV cell -> number, tolerating thousands separators and ``n/a``."""
    text = (value or "").strip().replace(",", "")
    if text in ("", "n/a", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt_number(total: float) -> str:
    """Render an accumulated total the way NCU would have -- integers stay integers,
    so a single-launch kernel parses byte-identically to before this aggregation."""
    return str(int(total)) if float(total).is_integer() else repr(total)


def parse_ncu_csv(text: str) -> dict[str, dict[str, str]]:
    """Group NCU ``--csv`` output into ``{kernel_name: {metric_name: value}}``.

    Column positions are read from the header row, so the parser does not break
    if NCU reorders columns. Lines before the header are ignored (NCU prepends
    setup logging when run via gpu_entry.sh).

    **Per-launch aggregation.** NCU emits one row set per LAUNCH, so a kernel
    launched N times inside the profiled window appears N times. This used to be a
    plain overwrite (``entry[metric] = value``), which kept only the LAST launch:
    the kernel's recorded device time then came out ~N x too small. That
    under-count is indistinguishable from a host tax -- in both cases the profiled
    kernel accounts for a small share of the timed wall clock -- so it false-fires
    the "profile is mis-aimed" alarm and would false-fire any host-bound detector
    built on the same reconciliation. Now:

    * additive counters (``_is_additive``) are SUMMED over the launches;
    * intensities and launch attributes are taken from the launch with the
      largest ``gpu__time_active.sum`` (the launch that decides the kernel's
      character) rather than from whichever happened to be last;
    * ``LAUNCH_COUNT_KEY`` records how many launches went into the row, so a
      reader of an absolute count knows it is a window total over N launches.

    Launches are separated by the CSV's ``ID`` column when present; without it a
    repeat of a metric name is taken as the start of the next launch.
    """
    reader = csv.reader(io.StringIO(text))
    header = None
    # kernel -> list of per-launch {metric: value}, in first-seen order
    launches: dict[str, list[dict[str, str]]] = {}
    by_id: dict[tuple[str, str], dict[str, str]] = {}
    geometry: dict[str, dict[str, str]] = {}
    for row in reader:
        if not row:
            continue
        if header is None:
            if "Kernel Name" in row and "Metric Name" in row:
                header = {name: i for i, name in enumerate(row)}
            continue
        try:
            kernel = row[header["Kernel Name"]]
            metric = row[header["Metric Name"]]
            value = row[header["Metric Value"]]
        except (IndexError, KeyError):
            continue
        buckets = launches.setdefault(kernel, [])
        launch_id = None
        if "ID" in header:
            try:
                launch_id = row[header["ID"]]
            except IndexError:
                launch_id = None
        if launch_id is not None:
            bucket = by_id.get((kernel, launch_id))
            if bucket is None:
                bucket = {}
                by_id[(kernel, launch_id)] = bucket
                buckets.append(bucket)
        else:
            if not buckets or metric in buckets[-1]:
                buckets.append({})
            bucket = buckets[-1]
        bucket[metric] = value
        # Launch geometry rides along on every row as a column; record it once.
        geo = geometry.setdefault(kernel, {})
        for column, short in LAUNCH_COLUMNS.items():
            if short not in geo and column in header:
                try:
                    geo[short] = row[header[column]]
                except IndexError:
                    pass
    return {kernel: _aggregate_launches(buckets, geometry.get(kernel, {}))
            for kernel, buckets in launches.items()}


def _aggregate_launches(buckets: list[dict[str, str]],
                        geometry: dict[str, str]) -> dict[str, str]:
    """Collapse one kernel's per-launch rows into the single row the rest reads."""
    entry: dict[str, str] = {}
    dominant = max(buckets, key=lambda b: _to_number(b.get(TIME_METRIC, "")) or 0.0,
                   default={})
    for metric, value in dominant.items():
        if not _is_additive(metric):
            entry[metric] = value
    totals: dict[str, float] = {}
    present: set[str] = set()
    for bucket in buckets:
        for metric, value in bucket.items():
            if not _is_additive(metric):
                continue
            present.add(metric)
            number = _to_number(value)
            if number is not None:
                totals[metric] = totals.get(metric, 0.0) + number
    for metric in present:
        # A metric present but never numeric anywhere ('n/a' in every launch)
        # keeps its raw spelling, so _coerce still resolves it to None downstream.
        entry[metric] = (_fmt_number(totals[metric]) if metric in totals
                         else dominant.get(metric, "n/a"))
    entry[LAUNCH_COUNT_KEY] = str(len(buckets))
    entry.update(geometry)
    return entry


def launch_count(entry: dict[str, str]) -> int:
    """How many launches one parsed kernel row aggregates (1 for legacy rows)."""
    return int(_to_number(entry.get(LAUNCH_COUNT_KEY, "1")) or 1)


def _total_launches(parsed: dict[str, dict[str, str]]) -> int:
    """Every profiled launch, across every kernel."""
    return sum(launch_count(entry) for entry in parsed.values())


def _coerce(value: str) -> float | int | None:
    value = str(value).strip().replace(",", "")
    if value in ("", "n/a", "N/A"):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None


def select_dominant_kernel(parsed: dict[str, dict[str, str]]) -> str:
    """The kernel that dominates runtime (max ``gpu__time_active.sum``).

    The bottleneck lives in the slowest kernel; profiling the pre/post-process
    kernels would mis-route the diagnosis.
    """
    if not parsed:
        raise ValueError("no kernels in NCU output")
    # Time is now the kernel's TOTAL over its launches (see parse_ncu_csv), so a
    # kernel launched many times per call can win dominance over one slower launch
    # -- which is the honest reading of "where the runtime went".
    return max(parsed, key=lambda k: kernel_time_ns(parsed[k]) or 0.0)


def kernel_time_ns(entry: dict[str, str]) -> float | None:
    """One parsed kernel row's total device time in ns, or None if unmeasured."""
    return _to_number(entry.get(TIME_METRIC, ""))


def total_time_ns(parsed: dict[str, dict[str, str]]) -> float | None:
    """Device time of ALL profiled kernels in ns -- the window's kernel-side total.

    This, not the dominant kernel alone, is what a timed wall clock has to be
    reconciled against: a legitimately decomposed operator spreads its work over
    several kernels, and comparing only the dominant one against the wall clock
    reads a decomposition as if it were a measurement error.
    """
    times = [t for e in parsed.values() if (t := kernel_time_ns(e)) is not None]
    return sum(times) if times else None


def _coerce_dims(value: str) -> int | None:
    """``"(32, 16, 10)"`` -> 5120. NCU reports launch geometry as a dim tuple; the
    product is the gate-comparable number (threads per CTA / total CTAs)."""
    nums = re.findall(r"-?\d+", value or "")
    if not nums:
        return None
    product = 1
    for n in nums:
        product *= int(n)
    return product


def _dims_tuple(value: str) -> str | None:
    """``"(32, 16, 10)"`` -> ``"32x16x10"``. The product alone is lossy in the one
    direction reconnaissance needs: a 132-CTA persistent grid is ``(132, 1, 1)``
    whatever the tile, and a tiled grid ``(64, 128, 1)`` and ``(8192, 1, 1)`` share a
    product while implying different rasterisation. Both are kept — the product for
    gate lines that already compare against it, the shape for reading the launch
    geometry back out."""
    nums = re.findall(r"-?\d+", value or "")
    return "x".join(nums) if nums else None


def key_metrics(
    parsed: dict[str, dict[str, str]], *, kernel: str | None = None
) -> dict[str, float | int | str]:
    """Decision-relevant metric subset for one kernel (default: the dominant one).

    **Unit: PER LAUNCH.** ``parse_ncu_csv`` sums additive counters over the launches
    of a kernel — that total is what the runtime split and the reconciliation need —
    but every consumer of THIS digest reads one launch's worth:

    * ``_THRESHOLDS`` bounds absolute counts (bank conflicts, spill sectors) against
      numbers calibrated per launch; a window total crosses them by the launch count
      alone and hands the student a "severe" verdict for a clean kernel.
    * ``_wavefront_ratio`` divides a student digest by an archived reference
      signature. The two are profiled through independently configured launch counts
      (``ref_ncu_cmd`` vs the operator's own ncu task), so only a per-launch ratio is
      launch-count independent — and every archive written before this aggregation
      holds one launch's worth, so per-launch is also what keeps them comparable.
    * milestone gate lines quote these numbers verbatim from an archived signature.

    Dividing here rather than at each consumer keeps one unit for the whole digest.
    A single-launch profile is unchanged by the division, so archives and gate lines
    written before per-launch aggregation still parse to the same values.
    """
    if not parsed:
        return {}
    kernel = kernel or select_dominant_kernel(parsed)
    metrics = parsed.get(kernel, {})
    launches = launch_count(metrics)
    out: dict[str, float | int | str] = {}
    for ncu_name, short in KEY_METRICS.items():
        if ncu_name in metrics:
            v = _coerce(metrics[ncu_name])
            if v is not None:
                # Additive counters arrive as a window total; report one launch's
                # worth (see the docstring). Intensities and launch attributes were
                # never summed and pass through untouched.
                if _is_additive(ncu_name) and launches > 1:
                    v = v / launches
                    if float(v).is_integer():
                        v = int(v)
                out[short] = v
    for short in LAUNCH_COLUMNS.values():
        if short in metrics:
            v = _coerce_dims(metrics[short])
            if v is not None:
                out[short] = v
            dims = _dims_tuple(metrics[short])
            if dims is not None:
                out[f"{short}_dims"] = dims
    # How many launches the dominant kernel had in the profiled window. The metrics
    # above are per launch; this is what turns them back into window totals, and what
    # ``timing_audit.ncu_coverage`` divides ``total_time_ns`` by to reach a per-call
    # basis.
    if LAUNCH_COUNT_KEY in metrics:
        out["dominant_launches"] = launches
    return out


def kernel_split(parsed: dict[str, dict[str, str]]) -> dict[str, float | int]:
    """The launch-level shape of the decomposition: how many kernels ran, and how
    the runtime divides between the dominant one and the rest.

    ``key_metrics`` deliberately reports only the dominant kernel — the bottleneck
    lives there. But an M0 that freezes "split into preprocess / main / postprocess"
    needs the split itself checked, and NCU already has it: the CSV is grouped by
    kernel name and ``gpu_entry.sh`` profiles every launch (no ``--kernel-name``
    filter). Both numbers are launch facts the agent cannot self-report away.
    """
    if not parsed:
        return {}

    def _time(k: str) -> float:
        return kernel_time_ns(parsed[k]) or 0.0

    total = sum(_time(k) for k in parsed)
    out: dict[str, float | int] = {"kernel_count": len(parsed),
                                     "total_launches": _total_launches(parsed)}
    if total > 0:
        out["total_time_ns"] = total
        out["dominant_time_pct"] = round(100.0 * _time(select_dominant_kernel(parsed)) / total, 2)
    return out


def profiler_from_ncu(csv_text: str, *, kernel: str | None = None) -> dict[str, float | int]:
    """Parse raw NCU CSV straight into the per-iteration ``profiler`` dict."""
    parsed = parse_ncu_csv(csv_text)
    return {**key_metrics(parsed, kernel=kernel), **kernel_split(parsed)}


def kernel_table(parsed: dict[str, dict[str, str]]) -> list[dict]:
    """Every profiled kernel as a row: ``name`` / ``launches`` / ``time_ns`` / ``pct``.

    ``kernel_split`` keeps only the *shape* of the decomposition (how many
    kernels, the dominant one's share); this keeps the decomposition itself.
    It is what the timed-window contract (``forgekernel.window_contract``) is
    priced against: the student launches that exist only to rebuild an input
    the reference's window receives for free are named by regex over these
    rows, and their share of ``time_ns`` is the contract tax. Window totals
    (over every profiled launch), sorted by time, so ``pct`` is launch-count
    independent and the two sides — profiled through independently configured
    launch counts — compare by share.
    """
    rows = []
    total = 0.0
    for name, entry in parsed.items():
        t = kernel_time_ns(entry)
        rows.append({"name": name, "launches": launch_count(entry),
                     "time_ns": t if t is not None else 0.0})
        total += t or 0.0
    for r in rows:
        r["pct"] = round(100.0 * r["time_ns"] / total, 2) if total > 0 else 0.0
    return sorted(rows, key=lambda r: r["time_ns"], reverse=True)


def kernels_from_ncu(csv_text: str) -> list[dict]:
    """Raw NCU CSV -> per-kernel table (``kernel_table``); [] on empty input."""
    return kernel_table(parse_ncu_csv(csv_text))


def build_ncu_trace(records) -> list[dict]:
    """Per-iteration metric series (only iterations that recorded NCU)."""
    trace = []
    for r in records:
        prof = getattr(r, "profiler", None) or {}
        if not prof:
            continue
        trace.append({"iter": r.iter, "decision_id": r.decision_id, **prof})
    return trace


def render_ncu_trace(records) -> str:
    """Markdown table of the metric evolution — the meta agent's NCU view."""
    trace = build_ncu_trace(records)
    if not trace:
        return "(no NCU profiler data recorded in this trajectory)"

    cols = [c for c in _TRACE_COLUMNS if any(c in row for row in trace)]
    head = "| iter | decision | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [head, sep]
    for row in trace:
        cells = [str(row.get(c, "")) for c in cols]
        lines.append(f"| {row['iter']} | {row['decision_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Thresholds for flagging a metric as a pathology (SM90 heuristics; tunable).
# Each: (key, kind, severe, elevated) — kind "hi" = high-is-bad, "lo" = low-is-bad.
_THRESHOLDS = [
    ("bankconf_st", "hi", 10_000_000, 1_000_000),
    ("bankconf_ld", "hi", 10_000_000, 1_000_000),
    ("spill_st_sectors", "hi", 10_000_000, 1_000_000),
    ("spill_ld_sectors", "hi", 10_000_000, 1_000_000),
    ("dram_pct", "hi", 80, 70),
    ("stall_long_sb", "hi", 20, 5),
    ("stall_barrier", "hi", 20, 5),
    ("stall_mio", "hi", 20, 5),
    ("tensor_pct", "lo", 15, 30),      # <15 severe, <30 elevated
    ("warp_active_pct", "lo", 10, 20),
    # wgmma waits per wgmma (sass_census.wgmma_serialization_from_text, rides the
    # per-iteration census): a pipelined kernel waits once per chain (the gdn
    # reference reads 0.15), ptxas's whole-function fallback waits once per
    # instruction (~1.0). Static, so it names the compiler — not the kernel's
    # algorithm — as the cause of a regression the perf number alone cannot
    # attribute (a gdn run once forbade a direction over exactly this).
    ("wgmma_serial_ratio", "hi", 0.9, 0.5),
]

# Wavefront volume is judged as a RATIO against the arch ref's signature: the
# same kernel is "too many wavefronts" only relative to how the reference moves
# the same data. Thresholds: student/ref >= 2.0 severe, >= 1.3 elevated (the
# bwd kernel above measured 3.2x; the job-to-job noise floor on count metrics is ~0).
_WAVEFRONT_KEYS = ("smem_wavefronts_st", "smem_wavefronts_ld")
_WAVEFRONT_RATIO_SEVERE = 2.0
_WAVEFRONT_RATIO_ELEVATED = 1.3


def _wavefront_ratio(m: dict, baseline: dict | None) -> float | None:
    """student/ref ratio of total shared-memory wavefronts, or None.

    None when either side lacks the metrics — legitimate for baselines archived
    before the wavefront columns existed; the flag simply does not fire (the
    absolute conflict thresholds still do)."""
    if not baseline:
        return None
    stu = sum(m.get(k) or 0 for k in _WAVEFRONT_KEYS)
    ref = sum(baseline.get(k) or 0 for k in _WAVEFRONT_KEYS)
    if not stu or not ref:
        return None
    return stu / ref


_NOTES = {
    "bankconf_st": "SMEM store bank conflicts (degenerate/conflicting smem layout)",
    "bankconf_ld": "SMEM load bank conflicts",
    "smem_wavefronts_st": "SMEM store wavefront volume",
    "smem_wavefronts_ld": "SMEM load wavefront volume",
    "smem_wavefronts_vs_ref": (
        "total SMEM wavefronts vs arch ref — a scalar/per-element access path "
        "(the copy atom, NOT the swizzle: switch to STSM/LDSM, vectorized, or "
        "WGMMA-descriptor feeds; swizzle tuning cannot reduce volume)"),
    "spill_st_sectors": "register spill (local store) — register pressure",
    "spill_ld_sectors": "register spill (local load) — register pressure",
    "dram_pct": "DRAM bandwidth utilization",
    "stall_long_sb": (
        "long-scoreboard stall — in mbarrier-pipelined kernels this bucket is "
        "DOMINATED by mbarrier/TMA spin-waits (NANOSLEEP), not DRAM latency; "
        "measured 100% of NANOSLEEP samples land here"),
    "stall_barrier": (
        "barrier stall — in WGMMA kernels this is mostly wgmma wait_group "
        "(WARPGROUP.DEPBAR, measured 98%) plus named barriers: read it as "
        "wait-depth/overlap quality, not just CTA-barrier imbalance"),
    "stall_mio": "MIO throttle (smem/SFU pipe)",
    "tensor_pct": "tensor-pipe active (low = tensor core starved)",
    "warp_active_pct": "warp activity (low = latency-bound)",
    "wgmma_serial_ratio": (
        "WGMMA SERIALIZED by ptxas — DEPBAR 0x0 waits per wgmma; ~1.0 means the "
        "compiler inserted a wait after EVERY wgmma for the whole function "
        "(accumulator spill / non-wgmma def of an accumulator / wgmma on a divergent "
        "path / register pressure). A compiler fallback, not the algorithm: remove "
        "the trigger before judging the change"),
}


def summarize_metrics(m: dict, baseline: dict | None = None) -> dict:
    """Digest a kernel's key metrics into flagged pathologies + a one-line verdict.

    Returns {"flags": [{key, value, severity, note}], "verdict": <str>}. This is
    the 'show the key parts directly' layer: the agent reads a bottleneck verdict
    instead of eyeballing dozens of raw numbers.

    ``baseline`` is the arch ref's NCU signature (``RefBaseline.ncu``). It powers
    the ratio-based wavefront-volume flag — a volume tax has no absolute "too
    high", only "N x what the reference pays for the same data".
    """
    flags = []
    for key, kind, severe, elevated in _THRESHOLDS:
        if key not in m or m[key] is None:
            continue
        val = m[key]
        sev = None
        if kind == "hi":
            sev = "severe" if val >= severe else ("elevated" if val >= elevated else None)
        else:  # lo
            sev = "severe" if val <= severe else ("elevated" if val <= elevated else None)
        if sev:
            flags.append({"key": key, "value": val, "severity": sev, "note": _NOTES.get(key, "")})

    ratio = _wavefront_ratio(m, baseline)
    if ratio is not None and ratio >= _WAVEFRONT_RATIO_ELEVATED:
        sev = "severe" if ratio >= _WAVEFRONT_RATIO_SEVERE else "elevated"
        flags.append({"key": "smem_wavefronts_vs_ref", "value": round(ratio, 2),
                      "severity": sev, "note": _NOTES["smem_wavefronts_vs_ref"]})

    def sv(key):  # severity of a flagged key, else None
        return next((f["severity"] for f in flags if f["key"] == key), None)

    # one-line binding-constraint read, severity-ordered. Wavefront volume is
    # checked BEFORE bank conflicts: when both fire, the access atom is the
    # cause and the conflicts are its side effect — a bwd run patched
    # the swizzle first (conflicts 73M -> 14M) and the wall time did not move,
    # because the per-element atoms still paid the volume tax.
    if sv("wgmma_serial_ratio") == "severe":
        # Checked FIRST: every other pathology below is measured on a kernel whose
        # MMA chains the compiler already un-pipelined; fix the trigger, re-read.
        verdict = ("WGMMA SERIALIZED by ptxas (issue-one-wait-one for the whole "
                   "function — a compiler fallback, not the change's algorithm)")
    elif sv("smem_wavefronts_vs_ref") == "severe":
        verdict = ("SMEM access-path bound (per-element copy atoms — wavefront "
                   "volume vs ref; swizzle tuning cannot fix this)")
    elif sv("bankconf_st") == "severe" or sv("bankconf_ld") == "severe":
        verdict = "SMEM bank-conflict bound (degenerate/conflicting smem layout)"
    elif sv("dram_pct") in ("severe", "elevated"):
        verdict = "memory-bandwidth bound"
    elif (sv("spill_st_sectors") or sv("spill_ld_sectors")) and sv("warp_active_pct"):
        verdict = "register-spill / latency bound"
    elif sv("tensor_pct") and sv("warp_active_pct"):
        verdict = "latency bound (tensor core starved)"
    elif sv("smem_wavefronts_vs_ref"):
        verdict = "elevated SMEM wavefront volume vs ref (check the copy atoms)"
    elif sv("bankconf_st") or sv("bankconf_ld"):
        verdict = "elevated SMEM bank conflicts"
    elif not m:
        verdict = "no NCU data"
    else:
        verdict = "no dominant pathology flagged"
    return {"flags": flags, "verdict": verdict}


def render_ncu_report(records, baseline: dict | None = None) -> str:
    """Per-iteration NCU SUMMARY — verdict + flagged key metrics (not the raw dump).

    ``baseline`` (the arch ref's NCU signature) enables the ratio-based
    wavefront-volume flag; without it only the absolute thresholds fire."""
    lines = []
    for r in records:
        prof = getattr(r, "profiler", None) or {}
        if not prof:
            continue
        s = summarize_metrics(prof, baseline=baseline)
        head = f"iter {r.iter} [{r.verdict}] {r.decision_id}"
        if getattr(r, "bwd_median_ms", None) is not None:
            mfu = getattr(r, "mfu", None)
            head += f"  ({r.bwd_median_ms} ms" + (f", MFU {mfu}%)" if mfu is not None else ")")
        lines.append(head)
        lines.append(f"  → {s['verdict']}")
        for f in s["flags"]:
            lines.append(f"    [{f['severity']:>8}] {f['key']} = {f['value']}  — {f['note']}")
    return "\n".join(lines) if lines else "(no NCU profiler data recorded in this trajectory)"


# --- two-sided reconciliation vs the ref's archived NCU signature -------------
#
# The report and trace above difference the student against ITSELF; the ref's
# same-schema signature (``RefBaseline.ncu``) only powered the wavefront ratio,
# so a plateau whose whole gap is "tensor pipe active 75% vs the ref's 82%"
# never surfaced as a line anyone could read — structural knobs kept being
# turned at a kernel whose tensor work already matched the ref. These are the
# ref-side keys the block quotes next to the derived ratios.
_RECON_REF_KEYS = ("tensor_pct", "time_ns", "dram_bytes_read", "dram_bytes_write",
                   "bankconf_st", "regs_per_thread", "smem_dynamic")
# Below this the ref is not a tensor-pipe kernel and a tensor-work
# reconciliation reads nothing — the whole block is omitted, not rendered.
_RECON_MIN_REF_TENSOR_PCT = 5.0
# "Same tensor work" band for tensor_cycles_ratio. Doubles as the minimum
# ncu_time_ratio for the verdict line: a residual smaller than the band the
# isomorphism claim itself tolerates is noise, and attributing noise would
# fire the verdict on every near-parity kernel.
_TENSOR_ISO_BAND = (0.97, 1.03)
# The verdict fires only when the occupancy bubble (tensor_pct gap x student
# time) explains at least this share of the time residual.
_BUBBLE_SHARE_MIN_PCT = 60.0
# ``baseline.ROLE_ARCHITECTURE``, kept literal so this module stays free of the
# archive layer. Only a signature of this role may pronounce the isomorphism
# verdict: a library baseline is a DIFFERENT implementation (see baseline.py on
# why the two roles must not collapse), so "same tensor work" against it is a
# coincidence, not evidence that the student's structure is finished.
_RECON_ROLE_ARCHITECTURE = "architecture"


def _recon_num(d: dict, key: str) -> float | None:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def render_ncu_reconciliation(student: dict | None, ref: dict | None, *,
                              student_iter: int | None = None,
                              ref_role: str = "", ref_name: str = "") -> str | None:
    """Student-best vs ref NCU reconciliation block, or None when it cannot speak.

    ``student`` is one iteration's ``profiler`` dict (the sitting best KEEP, its
    iter passed as ``student_iter`` so the block names which kernel it read);
    ``ref`` is the archived signature (``RefBaseline.ncu``) — same schema, and
    both hold one launch's worth (see ``key_metrics``), so the ratios are
    launch-count independent. ``ref_role`` / ``ref_name`` identify WHICH
    reference is speaking (``RefBaseline.role`` / ``.ref_name``); the header
    prints them, and the verdict below keys on the role. Derived fields:

    * ``tensor_pct_vs_ref`` — ref minus student, in pp (positive = student's
      tensor occupancy is behind);
    * ``tensor_cycles_ratio`` — student.time_ns x tensor_pct over the ref's same
      product: the ABSOLUTE tensor-work reconciliation. ~1 means the two kernels
      issue the same tensor work and the time gap is purely how densely it is
      packed — the one reading per-iteration differencing can never produce;
    * ``ncu_time_ratio`` / ``dram_bytes_*_ratio`` — plain side-by-side ratios.

    The OVERLAP-BOUND verdict line fires when the work is isomorphic (cycles
    ratio inside ``_TENSOR_ISO_BAND``), the time gap exceeds that same band, and
    the occupancy bubble covers >= ``_BUBBLE_SHARE_MIN_PCT`` of the residual —
    the "stop turning structure knobs" signal. Only an ARCHITECTURE-role
    signature may fire it; when a library baseline meets the same criteria the
    ratios still render but the verdict is replaced by an explicit suppression
    note — a different implementation cannot certify the student's structure.

    Silent None (never a raise) when the ref is not a tensor kernel
    (< ``_RECON_MIN_REF_TENSOR_PCT``) or either side lacks tensor_pct/time_ns:
    a missing block degrades to the status quo, a crash takes down diagnosis.
    """
    if not student or not ref:
        return None
    ref_tensor, ref_time = _recon_num(ref, "tensor_pct"), _recon_num(ref, "time_ns")
    stu_tensor, stu_time = _recon_num(student, "tensor_pct"), _recon_num(student, "time_ns")
    if None in (ref_tensor, ref_time, stu_tensor, stu_time):
        return None
    if ref_tensor < _RECON_MIN_REF_TENSOR_PCT or ref_time <= 0 or stu_time <= 0:
        return None
    gap_pp = ref_tensor - stu_tensor
    cycles_ratio = (stu_time * stu_tensor) / (ref_time * ref_tensor)
    time_ratio = stu_time / ref_time
    ident = ref_role or "unknown-role"
    if ref_name:
        ident += f": {ref_name}"
    student_label = ("student(best KEEP)" if student_iter is None
                     else f"student(best KEEP, iter {student_iter})")
    lines = [
        f"ref_ncu [{ident}]: " + "  ".join(f"{k}={ref[k]}" for k in _RECON_REF_KEYS
                                           if _recon_num(ref, k) is not None),
        f"{student_label}: tensor_pct={student['tensor_pct']}  time_ns={student['time_ns']}",
        f"tensor_pct_vs_ref = {gap_pp:+.1f} pp (ref − student; positive = student occupancy behind)",
        f"tensor_cycles_ratio = {cycles_ratio:.3f} (student time×tensor_pct ÷ ref same product; "
        f"~1 = same absolute tensor work)",
        f"ncu_time_ratio = {time_ratio:.3f} (student.time_ns ÷ ref.time_ns)",
    ]
    for key in ("dram_bytes_read", "dram_bytes_write"):
        s_v, r_v = _recon_num(student, key), _recon_num(ref, key)
        if s_v is not None and r_v:
            lines.append(f"{key}_ratio = {s_v / r_v:.3f}")
    lo, hi = _TENSOR_ISO_BAND
    if lo <= cycles_ratio <= hi and gap_pp > 0 and time_ratio >= hi:
        # time_ratio >= hi guarantees a positive residual; the bubble is the time
        # the student's tensor pipe sat idle beyond the ref's own idle share.
        bubble_share = (gap_pp / 100.0 * stu_time) / (stu_time - ref_time) * 100.0
        if bubble_share >= _BUBBLE_SHARE_MIN_PCT:
            if ref_role == _RECON_ROLE_ARCHITECTURE:
                lines.append(
                    f"OVERLAP-BOUND: tensor work is isomorphic to the ref "
                    f"(tensor_cycles_ratio {cycles_ratio:.3f}) and the occupancy bubble "
                    f"accounts for ~{bubble_share:.0f}% of the residual — the gap is "
                    f"overlap/issue quality, NOT structure; stop structure knobs. "
                    f"Read the role exposure ledger (the role with the "
                    f"largest Δexposed owns the residual) and the aligned segment "
                    f"table FIRST; spend verdict probes only on falsifying that "
                    f"role's lever, not on re-measuring where the exposure sits.")
            else:
                lines.append(
                    f"({ref_role or 'unknown-role'} baseline — different implementation; "
                    f"isomorphism verdict suppressed)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``forge <op> ncu-trace`` — render the per-iteration NCU metric evolution."""
    return _render_cli(argv, render_ncu_trace, "forge <op> ncu-trace")


def report_main(argv: list[str] | None = None) -> int:
    """``forge <op> ncu-report`` — per-iteration NCU summary (verdict + key parts)."""
    return _render_cli(argv, render_ncu_report, "forge <op> ncu-report")


def _render_cli(argv, renderer, prog) -> int:
    import argparse
    from pathlib import Path

    from forgekernel.inner.trajectory import load_records

    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("op", help="operator name under operators/")
    p.add_argument("--base-dir", default=None, help="repo root override (testing)")
    args = p.parse_args(argv)

    ops = Path(args.base_dir) / "operators" if args.base_dir else operators_dir()
    status = ops / args.op / "student" / "status.jsonl"
    if not status.exists():
        print(f"(no trajectory at {status})")
        return 0
    print(renderer(load_records(status)))
    return 0
