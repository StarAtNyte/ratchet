"""Per-SASS-instruction stall pricing — the dynamic ledger the static tools can't give.

`sass_tools` reads structure and `sched_live_tools` reads schedule + register
liveness, all from SASS *text*. This tool reads the *measured* PC-sampling stall
counters out of a `.ncu-rep` collected with `--set full` or `--set source`, joins
them per SASS instruction, and prints a ledger that answers the one question the
static channels and the aggregate digest both dodge: **which instructions is the
time actually leaking from, and how much at each**. Provenance: a one-off manual
forensic that located a third of all stalls in five wait_group points — an
analysis that was never productized, so a later run's meta spent 10h on two
wrong loci that a ledger would have priced at ~1pp each.

Ledger contract (the machine gate downstream drivers grep for):
  * rows are printed until they cover >= 90% of all true-stall samples — a ledger
    that only prices the top of the distribution reads as "covered everything"
    when it didn't (slice pricing that doesn't sum lies);
  * the tables sit between `=== STALL LEDGER BEGIN ===` / `=== STALL LEDGER END ===`
    marker lines so a driver can lift them into a prompt verbatim;
  * exactly one machine verdict line ends the run:
        STALL-OK kernel=<name> samples=<N> rows=<R> coverage=<pct>
        STALL-FAIL reason=<enum> ...
    printed only by this tool after its self-checks (samples floor, duration
    floor against mis-aim, coverage floor). Prose cannot open the gate.

Collection recipe (proven on the job image):
    export CUTE_DSL_LINEINFO=1          # student side: per-line mapping
    export CUTE_DSL_KEEP_IR=1
    export CUTE_DSL_DUMP_DIR=/tmp/dsl_ir
    export CUTE_DSL_CACHE_DIR=/tmp/dslcache_li   # JIT cache ignores the knob —
                                                 # a stale dir serves the old cubin
    ncu --set full --target-processes all --launch-count 8 -f -o /tmp/p \
        python3 eval/ncu_run.py
    python3 notes/stall_tools.py --report /tmp/p.ncu-rep --ir /tmp/dsl_ir

The dominant kernel is picked by duration across every captured launch, so
`--launch-count` can be generous instead of aimed — the known mis-aim mode (a
whole run profiling the preprocess kernel because setup ate the launch budget)
is handled by selection, not by counting.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# --- ncu_report locator (inlined: a vendored file travels alone) --------------

def _locate_ncu_report():
    candidates = [
        "/usr/local/cuda/nsight-compute/extras/python",
    ]
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


# --- stall metric set ---------------------------------------------------------
# `selected` counts issued cycles, not stalls — excluded from every total.
STALL_METRICS = [
    "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_short_scoreboard",
    "smsp__pcsamp_warps_issue_stalled_wait",
    "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle",
    "smsp__pcsamp_warps_issue_stalled_mio_throttle",
    "smsp__pcsamp_warps_issue_stalled_lg_throttle",
    "smsp__pcsamp_warps_issue_stalled_not_selected",
    "smsp__pcsamp_warps_issue_stalled_dispatch_stall",
    "smsp__pcsamp_warps_issue_stalled_drain",
    "smsp__pcsamp_warps_issue_stalled_no_instructions",
    "smsp__pcsamp_warps_issue_stalled_branch_resolving",
    "smsp__pcsamp_warps_issue_stalled_barrier",
    "smsp__pcsamp_warps_issue_stalled_tex_throttle",
    "smsp__pcsamp_warps_issue_stalled_sleeping",
    "smsp__pcsamp_warps_issue_stalled_misc",
    "smsp__pcsamp_warps_issue_stalled_membar",
]

MIN_SAMPLES = 500          # below this the sampler was effectively off / mis-aimed
MIN_DURATION_NS = 20_000   # a "dominant" kernel shorter than this is a mis-aim -- unless
                           # the profiler window vouches for it (window_vouches below)
COVERAGE_FLOOR = 90.0      # the ledger's whole point: rows must sum

# Short kernels (an audit of every earlier SOL-ExecBench job log): the two floors above were set
# for attention-sized kernels and fail closed on the SOL-ExecBench ones, which mostly run
# 5-20 us. PC sampling lands ~24 samples/us on one launch (1083 STALL-OK runs: p10-p90
# 22-29), so a single launch reaches MIN_SAMPLES only at ~20 us -- the duration floor
# is the samples floor in disguise. Over every job: 1642 logs STALL-FAIL mis-aim and 335
# too-few-samples against 1234 STALL-OK; the mis-aimed "dominant" kernel was the kernel
# under pricing in all 1642 (the student's kernel_cutlass_* / C++ kernel, the
# reference's nvjet_*, 7.0-19.6 us), never a setup kernel. The class the floor was built
# for (setup launches eating the --launch-count budget) cannot reach a window opened
# with --profile-from-start off around the timed calls (ncu_window: 8 calls of the
# kernel), so a short dominant kernel is accepted when the window vouches for it -- it is
# a repeated launch (>= WINDOW_MIN_REPEATS) holding most of the captured GPU time
# (>= WINDOW_MIN_SHARE) -- and its samples are pooled over its launches.
WINDOW_MIN_REPEATS = 2
WINDOW_MIN_SHARE = 0.5
# Pooling joins launches of one kernel in one process: one module, one load address, so
# their sampled PCs are the same code. A launch whose sampled PCs barely meet the
# dominant one's (a same-named kernel from another module) is left out; sparse sampling
# alone keeps the overlap well above this.
POOL_MIN_OVERLAP = 0.2
MAX_ROWS = 80

BEGIN = "=== STALL LEDGER BEGIN ==="
END = "=== STALL LEDGER END ==="


def short_stall(full: str) -> str:
    return full.replace("smsp__pcsamp_warps_issue_stalled_", "")


def _metric_value(m, i):
    k = m.kind()
    if k == m.ValueKind_UINT64:
        return m.as_uint64(i)
    if k in (m.ValueKind_DOUBLE, m.ValueKind_FLOAT):
        return m.as_double(i)
    try:
        return m.as_uint64(i)
    except Exception:
        try:
            return m.as_double(i)
        except Exception:
            return None


def collect_per_pc(action) -> dict:
    """dict[pc] -> dict[stall_metric -> count], true stalls only."""
    per_pc: dict = defaultdict(lambda: defaultdict(int))
    for sn in STALL_METRICS:
        try:
            m = action[sn]
            n = m.num_instances()
        except Exception:
            continue
        if n == 0 or not m.has_correlation_ids():
            continue
        cor = m.correlation_ids()
        for i in range(n):
            try:
                pc = cor.as_uint64(i)
            except Exception:
                try:
                    pc = int(cor.as_double(i))
                except Exception:
                    continue
            v = _metric_value(m, i)
            if v:
                per_pc[pc][sn] += int(v)
    return per_pc


def _duration(action):
    try:
        return action["gpu__time_duration.sum"].value() or 0
    except Exception:
        return 0


def pick_dominant_action(report, kernel_substr: str = ""):
    """The captured launch with the longest duration (optionally name-filtered).

    Selection instead of launch counting: `--launch-count` arithmetic is what
    once mis-aimed a whole run's NCU rounds at a preprocess kernel.
    """
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


def _captured(report) -> list:
    rng = report.range_by_idx(0)
    return [rng.action_by_idx(i) for i in range(rng.num_actions())]


def window_evidence(report, action) -> "tuple[int, float]":
    """(captured launches of ``action``'s kernel, their share of all captured GPU time)."""
    name = action.name()
    acts = _captured(report)
    total = sum(_duration(a) for a in acts)
    same = [a for a in acts if a.name() == name]
    return len(same), (sum(_duration(a) for a in same) / total if total else 0.0)


def window_vouches(report, action) -> "tuple[bool, str]":
    """Does the profiler window vouch for a short dominant kernel as the one under
    pricing? (ok, evidence text) -- a repeated launch holding most of the captured time."""
    n, share = window_evidence(report, action)
    ok = n >= WINDOW_MIN_REPEATS and share >= WINDOW_MIN_SHARE
    return ok, (f"{n} launch(es) of it, {100 * share:.0f}% of the captured GPU time "
                f"(vouches at >= {WINDOW_MIN_REPEATS} and >= {100 * WINDOW_MIN_SHARE:.0f}%)")


def misaim_reason(report, action, floor_ns: int = MIN_DURATION_NS, out=None):
    """None when ``action`` passes the mis-aim check, else the reason text: at or above
    the duration floor, or below it with the window vouching for it (window_vouches)."""
    dur = _duration(action)
    if dur >= floor_ns:
        return None
    ok, ev = window_vouches(report, action)
    if ok:
        if out is not None:
            out(f"[sanity] short kernel ({dur}ns < {floor_ns}ns) accepted: {ev}")
        return None
    return (f"dominant kernel {dur}ns < {floor_ns}ns and the window does not vouch for "
            f"it: {ev}")


def _sample_total(per_pc: dict) -> int:
    return sum(sum(s.values()) for s in per_pc.values())


def collect_per_pc_pooled(report, action, min_samples: int = MIN_SAMPLES):
    """(per_pc, launches pooled): ``action``'s own samples when they reach
    ``min_samples``, else summed over every captured launch of the same kernel whose
    sampled PCs overlap its own (POOL_MIN_OVERLAP) -- a short kernel's 8 timed calls
    carry 8x one launch's samples. No samples at all stays as is (the sampler was off;
    pooling has nothing to align on)."""
    own = collect_per_pc(action)
    if report is None or _sample_total(own) >= min_samples or not own:
        return own, 1
    base = set(own)
    pooled: dict = defaultdict(lambda: defaultdict(int))
    n = 0
    for a in _captured(report):
        if a.name() != action.name():
            continue
        other = collect_per_pc(a)
        if not other:
            continue
        overlap = len(base & set(other)) / min(len(base), len(other))
        if overlap < POOL_MIN_OVERLAP:
            continue
        n += 1
        for pc, s in other.items():
            for k, v in s.items():
                pooled[pc][k] += v
    if n <= 1:
        return own, 1
    return pooled, n


def dominant_launch(report_path, kernel_substr: str = ""):
    """(index, name) of the dominant action in the report's first range —
    the position the source page prints it at — or (None, None)."""
    ncu_report = _import_ncu_report()
    if ncu_report is None:
        return None, None
    try:
        report = ncu_report.load_report(str(report_path))
        rng = report.range_by_idx(0)
    except Exception:                                        # noqa: BLE001
        return None, None
    best, best_d, best_i = None, -1, None
    for i in range(rng.num_actions()):
        a = rng.action_by_idx(i)
        if kernel_substr and kernel_substr.lower() not in a.name().lower():
            continue
        d = _duration(a)
        if d > best_d:
            best, best_d, best_i = a, d, i
    return best_i, (best.name() if best is not None else None)


def load_sass_map(report_path, sass_csv=None):
    """dict[pc] -> SASS text, from the source-page CSV (spawns ncu if needed)."""
    import csv as _csv
    import shutil
    if sass_csv is None:
        if shutil.which("ncu") is None:
            return {}, "`ncu` not in PATH and no --sass-csv given"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        tmp.close()
        sass_csv = tmp.name
        cmd = ["ncu", "--import", str(report_path), "--page", "source",
               "--print-source", "sass", "--csv"]
        with open(sass_csv, "w") as out:
            rc = subprocess.run(cmd, stdout=out, stderr=subprocess.DEVNULL).returncode
        if rc != 0:
            return {}, f"`ncu --import` exited {rc}"
    raw = list(_csv.reader(open(sass_csv, errors="ignore")))
    header_idx = next(
        (i for i, row in enumerate(raw[:15]) if any(c.strip() == "Address" for c in row)),
        None)
    if header_idx is None:
        return {}, "no 'Address' header row in the source CSV"
    hdr = [c.strip() for c in raw[header_idx]]
    pc2sass = {}
    for row in raw[header_idx + 1:]:
        if len(row) != len(hdr):
            continue
        d = dict(zip(hdr, row))
        addr = (d.get("Address") or "").strip()
        if addr.startswith("0x"):
            pc2sass[int(addr, 16)] = (d.get("Source") or "").strip()
    return pc2sass, None


def load_ir_line_table(ir_path):
    """MLIR loc("file":line:col) -> dict[line] -> set(file basenames).

    Works around the CuTe DSL defect where the PTX .file table collapses to one
    entry: line numbers stay correct, file attribution comes from the IR dump.
    """
    line2files = defaultdict(set)
    if not ir_path:
        return line2files
    p = Path(ir_path)
    paths = []
    if p.is_dir():
        paths = list(p.glob("**/*.mlir")) + list(p.glob("**/*.ir"))
        if not paths:
            paths = [q for q in p.iterdir() if q.is_file()]
    elif p.is_file():
        paths = [p]
    for path in paths:
        try:
            text = open(path, errors="ignore").read()
        except OSError:
            continue
        for fname, line in re.findall(r'loc\("([^"]+)":(\d+):\d+\)', text):
            line2files[int(line)].add(os.path.basename(fname))
    return line2files


def opcode_of(sass_text) -> str:
    tokens = (sass_text or "?").split()
    op = next((t for t in tokens if not t.startswith("@")), "?")
    return op.split(".")[0]


# --- pure ledger core (testable without a report) -----------------------------

def build_ledger(per_pc: dict, pc2sass: dict, loc_of, out=print):
    """Aggregate, print the ledger tables, and return (total, rows, coverage_pct).

    `loc_of(pc)` returns a "file:line" label or "" — when it resolves for enough
    of the samples the ledger keys on source lines (the student side, LINEINFO
    on); otherwise on (opcode, pc-bucket) segments (the ref side, no lineinfo).
    Rows are emitted most-expensive first until cumulative coverage crosses the
    90% floor — the contract is that the ledger SUMS, not that it samples.
    """
    total = sum(sum(s.values()) for s in per_pc.values())
    if total == 0:
        return 0, 0, 0.0

    resolved_samples = sum(sum(s.values()) for pc, s in per_pc.items() if loc_of(pc))
    by_line = resolved_samples / total >= 0.3

    groups: dict = defaultdict(lambda: defaultdict(int))
    rep_sass: dict = {}
    for pc, stalls in per_pc.items():
        if by_line:
            key = loc_of(pc) or f"<unresolved>@{opcode_of(pc2sass.get(pc))}"
        else:
            key = f"{opcode_of(pc2sass.get(pc))}@{pc // 0x200:#x}"
        for sn, v in stalls.items():
            groups[key][sn] += v
        cur = sum(stalls.values())
        if cur > rep_sass.get(key, (0, ""))[0]:
            rep_sass[key] = (cur, pc2sass.get(pc, hex(pc)))

    out(f"[ledger] keying={'source-line' if by_line else 'opcode/pc-bucket'} "
        f"({100 * resolved_samples / total:.0f}% of samples line-resolved)")
    out(f"\n--- ledger rows (cumulative to >= {COVERAGE_FLOOR:.0f}% of "
        f"{total} true-stall samples) ---")
    ranked = sorted(groups.items(), key=lambda kv: -sum(kv[1].values()))
    cum = 0
    rows = 0
    for key, stalls in ranked[:MAX_ROWS]:
        seg = sum(stalls.values())
        cum += seg
        rows += 1
        brk = ", ".join(f"{short_stall(k)}:{100 * v / seg:.0f}%" for k, v in
                        sorted(stalls.items(), key=lambda x: -x[1])[:3])
        out(f"{seg:8d} {100 * seg / total:5.2f}% (cum {100 * cum / total:5.1f}%)  "
            f"{key[:44]:44s} {rep_sass[key][1][:40]:40s} | {brk}")
        if 100 * cum / total >= COVERAGE_FLOOR and rows >= 10:
            break
    coverage = 100 * cum / total

    fam: dict = defaultdict(lambda: defaultdict(int))
    for pc, stalls in per_pc.items():
        op = opcode_of(pc2sass.get(pc))
        for k, v in stalls.items():
            fam[op][k] += v
    out("\n--- opcode family x dominant stall ---")
    for op, stalls in sorted(fam.items(), key=lambda x: -sum(x[1].values()))[:15]:
        tot = sum(stalls.values())
        top = ", ".join(f"{short_stall(k)}={100 * v / tot:.0f}%" for k, v in
                        sorted(stalls.items(), key=lambda x: -x[1])[:3])
        out(f"{tot:8d} {100 * tot / total:5.2f}%  {op:22s} | {top}")
    return total, rows, coverage


def _sanitize(name: str) -> str:
    """A kernel name safe for a single-token verdict field."""
    return re.sub(r"[^A-Za-z0-9_.]", "_", name)[:48] or "unknown"


def price_report(report_path, ir=None, kernel_substr: str = "",
                 sass_csv=None, out=print) -> bool:
    """The full pricing run off one .ncu-rep; True iff STALL-OK was printed."""
    ncu_report = _import_ncu_report()
    if ncu_report is None:
        out("STALL-FAIL reason=no-ncu-report-module (Nsight extras/python not found)")
        return False
    try:
        report = ncu_report.load_report(str(report_path))
    except Exception as e:
        out(f"STALL-FAIL reason=report-unreadable ({e})")
        return False
    action = pick_dominant_action(report, kernel_substr)
    if action is None:
        out("STALL-FAIL reason=no-kernel-matched")
        return False
    dur = _duration(action)
    out(f"[sanity] kernel={action.name()[:80]}")
    out(f"[sanity] duration_ns={dur} — dominant across every captured launch")
    why = misaim_reason(report, action, out=out)
    if why is not None:
        out(f"STALL-FAIL reason=mis-aim ({why} — the main kernel was not captured; "
            f"widen --launch-count or fix --kernel-substr)")
        return False

    per_pc, pooled = collect_per_pc_pooled(report, action)
    total = _sample_total(per_pc)
    if pooled > 1:
        out(f"[sanity] samples pooled over {pooled} launches of the dominant kernel "
            f"({total} total; one launch fell short of {MIN_SAMPLES})")
    if total < MIN_SAMPLES:
        out(f"STALL-FAIL reason=too-few-samples ({total} < {MIN_SAMPLES} — was PC "
            f"sampling collected? needs --set full or --set source)")
        return False

    pc2sass, err = load_sass_map(report_path, sass_csv)
    if err:
        out(f"[warn] no SASS text ({err}) — rows fall back to raw PCs")

    line2files = load_ir_line_table(ir)

    def loc_of(pc):
        try:
            si = action.source_info(pc)
        except Exception:
            si = None
        if si is None:
            return ""
        line = si.line()
        cand = line2files.get(line)
        if cand:
            name = next(iter(cand)) if len(cand) == 1 else "|".join(sorted(cand))
            return f"{name}:{line}"
        return f"{os.path.basename(si.file_name())}:{line}"

    out(BEGIN)
    total, rows, coverage = build_ledger(per_pc, pc2sass, loc_of, out=out)
    out(END)
    if coverage < COVERAGE_FLOOR:
        out(f"STALL-FAIL reason=coverage ({coverage:.1f}% < {COVERAGE_FLOOR:.0f}% "
            f"— the ledger does not sum; distribution too flat for {MAX_ROWS} rows)")
        return False
    out(f"STALL-OK kernel={_sanitize(action.name())} samples={total} rows={rows} "
        f"coverage={coverage:.1f}")
    return True


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report", required=True)
    ap.add_argument("--ir", default=None, help="CUTE_DSL_DUMP_DIR for line->file recovery")
    ap.add_argument("--kernel-substr", default="")
    ap.add_argument("--sass-csv", default=None)
    args = ap.parse_args(argv)
    ok = price_report(args.report, ir=args.ir, kernel_substr=args.kernel_substr,
                      sass_csv=args.sass_csv)
    segment_ledger(args.report, kernel_substr=args.kernel_substr)
    return 0 if ok else 1


# --- segment ledger: stall cost joined to the schedule, in one address space --
#
# The row ledger above keys on source lines when LINEINFO resolves — but on the
# student side ~30% of samples (every HGMMA/DEPBAR/barrier: the @cute.jit
# decorator has no line) pile into one unresolved bucket, so exactly the rows a
# scheduling diagnosis needs (the DEPBAR handshake, the depth-0x1 maths) are
# invisible. This ledger keys on PC instead: each sampled PC resolves to a body
# instruction, contiguous same-kind instructions form a segment, each segment
# inherits the DEPBAR depth in force at its start. One --set full collection
# feeds the SASS text, the schedule and the per-PC samples, so the join happens
# inside a single address space (joining PCs across jobs is unsound — ASLR).

SEG_BEGIN = "=== SEGMENT LEDGER BEGIN ==="
SEG_END = "=== SEGMENT LEDGER END ==="
JOIN_FLOOR = 0.50   # below this the join is address-space mismatch, not a table
MAX_SEG_ROWS = 18


def _sched_modules():
    """sass_tools + sched_live_tools — package import or vendored-flat (ws notes/)."""
    try:
        from forgekernel.skills import sass_tools, sched_live_tools
        return sass_tools, sched_live_tools
    except ImportError:
        try:
            import sass_tools          # noqa: F401  (vendored next to this file)
            import sched_live_tools    # noqa: F401
            return sass_tools, sched_live_tools
        except ImportError:
            return None, None


def render_sass_text(report_path):
    """The report's SASS page as text (not CSV) — what parse_sass consumes."""
    import shutil
    if shutil.which("ncu") is None:
        return None, "`ncu` not in PATH"
    proc = subprocess.run(
        ["ncu", "--import", str(report_path), "--page", "source",
         "--print-source", "sass"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return None, f"`ncu --import` exited {proc.returncode}"
    return proc.stdout, None


def _kind_of(ins, owner, idx, slt):
    """One instruction -> segment kind. Mirrors sched_live_tools.transcribe's
    families, except BRA/EXIT are kept — they carry real measured cost (branch
    has been the single most expensive family) and dropping them would delete
    the largest cost in the kernel. Segments therefore cover every instruction."""
    f = slt.fam_of(ins.opcode)
    if f == "HGMMA":
        ch = owner.get(idx)
        return ("HG", ch.label() if ch else "HGMMA")
    if f == "WARPGROUP":
        if "DEPBAR" in ins.opcode:
            im = slt._HEX.findall(ins.operands)
            return ("DEPBAR", f"DEPBAR.LE {im[-1] if im else '?'}")
        return ("WG", ins.opcode)
    if f in ("BAR", "BSSY", "BSYNC"):
        return ("BAR", ins.opcode)
    if f == "SYNCS":
        kind = ("arrive" if "ARRIVE" in ins.opcode else
                "trywait" if ("TRYWAIT" in ins.opcode or "PHASECHK" in ins.opcode)
                else ins.opcode)
        return ("SYNCS", f"mbarrier.{kind}")
    if f == "NANOSLEEP":
        return ("SLEEP", "NANOSLEEP(spin)")
    if f in ("STSM", "LDSM"):
        return (f, ins.opcode)
    if f.startswith("UTMA") or f.startswith("UBLK"):
        return ("TMA", ins.opcode)
    if f in ("MEMBAR", "FENCE", "ELECT"):
        return ("FENCE", ins.opcode)
    if f in ("BRA", "BRX", "JMP"):
        return ("BRA", "branch")
    if f == "EXIT":
        return ("EXIT", "EXIT")
    return ("EW", "elementwise")


def _build_segments(instrs, owner, lo, hi, slt):
    """Contiguous same-kind runs over [lo,hi] -> [[start,end,kind,text,depth]].

    depth = DEPBAR immediate in force at the segment's start: set by a DEPBAR,
    cleared by an HGMMA run — the same state machine sched_verdict applies to
    its collapsed events, applied here to full-coverage segments."""
    segs = []
    depth = "none"
    i = lo
    while i <= hi:
        k, t = _kind_of(instrs[i], owner, i, slt)
        j = i
        while j + 1 <= hi and _kind_of(instrs[j + 1], owner, j + 1, slt) == (k, t):
            j += 1
        segs.append([i, j, k, t, depth])
        if k == "DEPBAR":
            m = slt._HEX.findall(instrs[i].operands)
            depth = m[-1] if m else "?"
        elif k == "HG":
            depth = "none"
        i = j + 1
    return segs


def _depbar_imm(text: str) -> str:
    m = re.search(r"0x[0-9a-fA-F]+", text or "")
    return m.group(0) if m else "?"


def segment_ledger(report_path, kernel_substr: str = "", out=print,
                   sass_text=None) -> bool:
    """Print the per-segment stall account between SEG markers; True iff JOIN-OK.

    Emits, in order: machine `SEG` lines (one per costed segment, program order:
    kind / depth / share of body / top-2 stalls — what the driver's aligned
    comparison table parses), the human tables (top segments, elementwise cost
    by DEPBAR depth, cost by kind), and a JOIN verdict line. Every failure mode
    prints its reason inside the markers rather than raising."""
    out(SEG_BEGIN)
    ok = False
    try:
        ok = _segment_ledger_body(report_path, kernel_substr, out, sass_text)
    except Exception as e:                                   # noqa: BLE001
        out(f"JOIN-FAIL reason=exception {type(e).__name__}: {e}")
    out(SEG_END)
    return ok


def _segment_ledger_body(report_path, kernel_substr, out, sass_text=None) -> bool:
    sass_tools, slt = _sched_modules()
    if slt is None:
        out("JOIN-FAIL reason=no-sched-tools (sass_tools/sched_live_tools not importable)")
        return False
    ncu_report = _import_ncu_report()
    if ncu_report is None:
        out("JOIN-FAIL reason=no-ncu-report-module")
        return False
    report = ncu_report.load_report(str(report_path))
    action = pick_dominant_action(report, kernel_substr)
    if action is None:
        out("JOIN-FAIL reason=no-kernel-matched")
        return False
    per_pc, pooled = collect_per_pc_pooled(report, action)
    total = _sample_total(per_pc)
    if total < MIN_SAMPLES:
        out(f"JOIN-FAIL reason=too-few-samples ({total} < {MIN_SAMPLES})")
        return False
    if pooled > 1:
        out(f"[join] samples pooled over {pooled} launches of the dominant kernel")
    if sass_text is None:
        # the caller may hand over a page it already rendered (stall_price_run
        # renders once and shares it with the dossier: `ncu --import` of a
        # full-set report is minutes, not seconds)
        sass_text, err = render_sass_text(report_path)
        if err:
            out(f"JOIN-FAIL reason=no-sass ({err})")
            return False
    parsed = sass_tools.parse_sass(sass_text)
    instrs = parsed.instructions
    if not instrs:
        out("JOIN-FAIL reason=sass-parsed-empty")
        return False
    lo, hi, tag = slt.find_loop(instrs)
    _, owner = slt.build_chains(instrs[lo:hi + 1])
    owner = {lo + k: v for k, v in owner.items()}

    addr2idx = {}
    for n, ins in enumerate(instrs):
        try:
            addr2idx[int(ins.addr or "0", 16)] = n
        except ValueError:
            continue
    matched = sum(sum(s.values()) for p, s in per_pc.items() if p in addr2idx)
    frac = matched / total if total else 0.0
    out(f"[join] {len(per_pc)} sampled PCs vs {len(addr2idx)} parsed addresses; "
        f"{100 * frac:.1f}% of {total} stall samples resolved")
    if frac < JOIN_FLOOR:
        out(f"[join] UNSAFE ({100 * frac:.1f}% < {100 * JOIN_FLOOR:.0f}%) — "
            "address spaces did not line up; unmatched top PCs:")
        for p, s in sorted(per_pc.items(), key=lambda kv: -sum(kv[1].values()))[:8]:
            if p not in addr2idx:
                out(f"   | {p:#x}  {sum(s.values())} samples")
        out("JOIN-FAIL reason=address-space-mismatch")
        return False

    segs = _build_segments(instrs, owner, lo, hi, slt)
    cost = defaultdict(int)
    stalls_of = defaultdict(lambda: defaultdict(int))
    outside = 0
    for p, s in per_pc.items():
        n = addr2idx.get(p)
        if n is None:
            continue
        tot = sum(s.values())
        if n < lo or n > hi:
            outside += tot
            continue
        for si, seg in enumerate(segs):
            if seg[0] <= n <= seg[1]:
                cost[si] += tot
                for k, v in s.items():
                    stalls_of[si][k] += v
                break
    body_tot = sum(cost.values())
    n_hg = sum(1 for s in segs if s[2] == "HG")
    n_db = sum(1 for s in segs if s[2] == "DEPBAR")
    out(f"[join] body {tag}: {len(segs)} segments ({n_hg} HGMMA runs, {n_db} DEPBARs); "
        f"{body_tot} samples inside, {outside} outside (prologue/epilogue)")
    if body_tot == 0:
        out("JOIN-FAIL reason=no-samples-in-body")
        return False

    def _brk(si, n=2):
        c = cost[si] or 1
        return ",".join(f"{short_stall(k)}:{100 * v / c:.0f}" for k, v in
                        sorted(stalls_of[si].items(), key=lambda x: -x[1])[:n])

    # machine lines, program order — the driver's aligned table parses these
    for si, seg in enumerate(segs):
        if not cost.get(si):
            continue
        imm = _depbar_imm(seg[3]) if seg[2] == "DEPBAR" else ""
        out(f"SEG idx={si} kind={seg[2]} depth={seg[4]}"
            + (f" imm={imm}" if imm else "")
            + f" pct={100 * cost[si] / body_tot:.1f} stalls={_brk(si)}")

    out(f"\n--- top segments by stall cost (program order; top {MAX_SEG_ROWS}) ---")
    ranked = sorted(cost.items(), key=lambda kv: -kv[1])[:MAX_SEG_ROWS]
    for si in sorted(r[0] for r in ranked):
        seg = segs[si]
        out(f"{si:>4} {seg[0] - lo:>5}-{seg[1] - lo:<6} {seg[2]:<8} {str(seg[4]):>5} "
            f"{cost[si]:>8} {100 * cost[si] / body_tot:>5.1f}%  "
            f"{seg[3][:26]:<26} | {_brk(si)}")

    out("\n--- elementwise cost by DEPBAR depth in force ---")
    by_depth = defaultdict(int)
    n_seg = defaultdict(int)
    for si, seg in enumerate(segs):
        if seg[2] == "EW":
            by_depth[str(seg[4])] += cost.get(si, 0)
            n_seg[str(seg[4])] += 1
    ew_tot = sum(by_depth.values()) or 1
    for d, c in sorted(by_depth.items(), key=lambda x: -x[1]):
        out(f"   depth {d:<5} {n_seg[d]:>3} segments  {c:>7} samples  "
            f"{100 * c / ew_tot:>5.1f}% of elementwise  {100 * c / body_tot:>5.1f}% of body")

    out("\n--- cost by segment kind ---")
    by_kind = defaultdict(int)
    for si, seg in enumerate(segs):
        by_kind[seg[2]] += cost.get(si, 0)
    for k, c in sorted(by_kind.items(), key=lambda x: -x[1]):
        out(f"   {k:<8} {c:>8} {100 * c / body_tot:>6.1f}% of body")

    out(f"JOIN-OK resolved={100 * frac:.1f}% body_samples={body_tot} "
        f"segments={len(segs)} hgmma_runs={n_hg} depbars={n_db}")
    return True


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
