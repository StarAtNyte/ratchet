"""Driver-seeded stall-pricing run — collects and prices the CURRENT kernel.

Submitted by the loop driver at plateau diagnosis time (`cmd:python3
notes/stall_price_run.py`, cwd = repo root) against the best-so-far commit, so
the diagnose step reads a measured per-instruction stall ledger instead of
guessing a locus from the aggregate digest. Not a student tool: students keep
using PROBE_REQUEST; this file exists so every commit after seeding carries the
frozen recipe (git archive ships committed files only).

Recipe frozen from earlier forensic and feasibility runs: LINEINFO costs zero
perf, the JIT cache must get a fresh dir because it ignores the LINEINFO knob,
and the dominant kernel is picked by duration in the analysis instead of aimed
with launch arithmetic (aiming by launch index mis-aims).

`--pm`: ride the PM-sampling sections on the SAME collection and render the
phase timeline (`pm_timeline_tools`) off the same report — one job answers
both WHERE (ledger) and WHEN (timeline). The driver's pricing job passes it
(the role exposure ledger scales by the timeline's tensor plateau; without
the sections it falls back to the whole-kernel aggregate); probes may pass it
for phase-shaped hypotheses (add `--pm-interval 2048` for sub-0.5 ms kernels —
the stall series sample AT the interval and starve at the 8192 default).
Guard: `--set full` has crashed at the last collection pass on some
kernels (cluster-of-2 fp8) and extra sections can only widen that class,
so a failed combined collection falls back to the plain recipe — the ledger
never regresses because the timeline was requested; the timeline then reports
PMTL-FAIL and the standalone `notes/pm_timeline_run.py` (sections-only, much
lighter) is the retry path.

`--app <argv...>` (must be last): the launch target, when the tree under
pricing is not a student workspace. Default `python eval/ncu_run.py`. A
reference tree that exposes the profiler window some other way (e.g. a
dsa_sparse recon tree: `python3 dsa_ref.py ncu`) is priced with the same
frozen recipe by naming it here — the driver reads it from loop.toml
`[loop] ref_price_app`. The window contract is unchanged: the target must
open the profiler only around the kernel calls (--profile-from-start off).

`--workload <name>` (before `--app`): price one workload of a group card. The
launch target's ncu_window follows FK_LB_WORKLOAD_SCOPE (set here, inherited
by everything ncu starts); without the flag a group card is priced on its
representative. The driver prices the card's other meta workloads (the
group's smallest and largest) this way, one job each, beside the
representative's reading.
"""
import base64
import lzma
import os
import re
import subprocess
import sys

REPORT = "/tmp/stallprice"
IR_DIR = "/tmp/dsl_ir"
WORKLOAD_SCOPE_ENV = "FK_LB_WORKLOAD_SCOPE"   # == leaderboard.WORKLOAD_SCOPE_ENV

# The analysis output, replayed compactly at the very END of the job log.
# A log read back with a length cap keeps only its tail, and the pricing output
# is long — a 500-segment SEGMENT LEDGER, a 165-line ORCH DOSSIER, the ROLE and
# PM TIMELINE blocks — so the STALL LEDGER at the head of the output is the
# first thing to go, silently. The bundle is the whole analysis text, xz +
# base64, chunked into a dozen long lines: the driver expands it before it
# parses any block (expand_bundle), so the head can go missing without the
# diagnosis losing evidence. Human readers still get the plain blocks above.
BUNDLE_BEGIN = "=== PRICE BUNDLE BEGIN ==="
BUNDLE_END = "=== PRICE BUNDLE END ==="
BUNDLE_LINE = 1000


class _Tee:
    """Records everything printed through it while still printing it."""

    def __init__(self, stream):
        self.stream = stream
        self.parts: list = []

    def write(self, s):
        self.stream.write(s)
        self.parts.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()

    def text(self) -> str:
        return "".join(self.parts)


def emit_bundle(text: str, out=print) -> None:
    data = base64.b64encode(lzma.compress(text.encode("utf-8"), preset=9)).decode("ascii")
    out(BUNDLE_BEGIN)
    n = 0
    for i in range(0, len(data), BUNDLE_LINE):
        out(data[i:i + BUNDLE_LINE])
        n += 1
    out(BUNDLE_END)
    out(f"BUNDLE-OK bytes={len(text)} lines={n}")


def expand_bundle(log: str) -> str:
    """The log with its bundle decoded and appended (unchanged when there is
    none, or it is unreadable). The plain blocks above the bundle stay where
    they are — a consumer's first regex match lands on them when the head
    survived and on the decoded copy when it did not."""
    b = log.find(BUNDLE_BEGIN)
    if b < 0:
        return log
    e = log.find(BUNDLE_END, b)
    if e < 0:
        return log
    payload = "".join(log[b + len(BUNDLE_BEGIN):e].split())
    try:
        text = lzma.decompress(base64.b64decode(payload)).decode("utf-8")
    except Exception:  # noqa: BLE001 — a clipped bundle is no worse than none
        return log
    if text in log:
        return log
    return log + "\n=== PRICE BUNDLE EXPANDED ===\n" + text

# The phase-resolving PM recipe (measured on H100 / ncu 2025.4):
# the warp-stall series sample AT --pm-sampling-interval, and 8192 cycles is
# what resolves phase structure (65536 gives ~45 samples on a ~1.7 ms kernel).
# Fast kernels starve at 8192 — pass `--pm-interval 2048` for sub-0.5 ms ones.
PM_INTERVAL = 8192


def pm_sections(interval: int) -> list:
    return ["--section", "PmSampling", "--section", "PmSampling_WarpStates",
            "--pm-sampling-interval", str(interval),
            "--pm-sampling-buffer-size", "268435456"]


DEFAULT_APP = [sys.executable, "eval/ncu_run.py"]


def build_cmd(pm: bool, pm_interval: int = PM_INTERVAL, app: list | None = None) -> list:
    # --profile-from-start off is load-bearing, not optional: eval/ncu_run.py's
    # contract is to open the profiler window only around the kernel calls, and
    # without the flag --launch-count is eaten by the input-construction launch
    # storm (an argsort alone is a dozen-plus launches), so every channel
    # prices the setup kernel — which can be long enough (~2 ms radix sort) to
    # sail past the mis-aim duration floor and fail SILENTLY-WRONG (measured:
    # a job priced DeviceSegmentedRadixSortKernel; a dsa-bwd run shipped four
    # such mis-aimed ledgers). If an operator's ncu_run.py lacks
    # the window, the run now fails LOUD (no-kernel / too-few-samples) instead.
    return (["ncu", "--set", "full", "--target-processes", "all",
             "--profile-from-start", "off",
             "--launch-count", "8", "-f", "-o", REPORT]
            + (pm_sections(pm_interval) if pm else [])
            + list(app or DEFAULT_APP))


def clear_report(base: str) -> None:
    """Remove the report a previous job left at ``base``.ncu-rep before collecting.

    The report's existence is what gates the analysis (_collect, the finer PM
    re-collection), and a GPU machine keeps /tmp across jobs: a collection that
    produced nothing would otherwise be analyzed off the PREVIOUS job's report --
    another tree, another workload -- and printed as this one's verdict."""
    try:
        os.remove(base + ".ncu-rep")
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"[stall_price] could not remove the previous report {base}.ncu-rep ({e}); "
              f"a failed collection below may read it", flush=True)


def _collect(pm: bool, pm_interval: int = PM_INTERVAL, app: list | None = None) -> bool:
    """One ncu collection; True iff the report file exists afterwards.

    The report's existence is the ONLY analysis gate: the cluster-of-2 fp8 crash class
    (app/teardown dies after the passes ran) exits nonzero with a complete,
    priceable report, and requiring rc==0 would throw that report away and
    misreport it as no-report. A nonzero rc is logged; whatever data the crash
    clipped is caught downstream by each tool's own self-checks.
    """
    cmd = build_cmd(pm, pm_interval, app)
    clear_report(REPORT)
    print(f"[stall_price] collecting{' (+PM sections)' if pm else ''}: "
          f"{' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr)[-1500:]
    print(f"[stall_price] ncu rc={proc.returncode}; output tail:\n{tail}", flush=True)
    exists = os.path.exists(REPORT + ".ncu-rep")
    if exists and proc.returncode != 0:
        print("[stall_price] ncu exited nonzero but the report exists "
              "(teardown-crash class) — analyzing it; per-tool self-checks "
              "cover anything the crash clipped", flush=True)
    return exists


def _render_page(stall_tools):
    """(SASS page text, dominant launch index, dominant launch name) — rendered
    ONCE and shared by the segment ledger and the dossier (a second `ncu
    --import` of a full-set report costs minutes). Each part degrades to None
    on its own. The name lets the page readers pick the kernel's own launch
    even when a bystander launch fused with it on the page."""
    sass_text = launch = name = None
    render = getattr(stall_tools, "render_sass_text", None)
    if render is not None:
        try:
            sass_text, err = render(REPORT + ".ncu-rep")
            if err:
                print(f"[stall_price] SASS page unavailable ({err})", flush=True)
                sass_text = None
        except Exception as e:  # noqa: BLE001
            print(f"[stall_price] SASS page render failed ({type(e).__name__}: "
                  f"{e})", flush=True)
            sass_text = None
    pick = getattr(stall_tools, "dominant_launch", None)
    if sass_text is not None and pick is not None:
        try:
            launch, name = pick(REPORT + ".ncu-rep")
            print(f"[stall_price] dominant launch index={launch} name={name}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[stall_price] dominant launch lookup failed "
                  f"({type(e).__name__}: {e}) — the dossier takes the largest "
                  "launch on the page", flush=True)
    return sass_text, launch, name


# A short kernel's PM stall series starve at the driver's interval: on B200 2048
# lands ~4 stall samples per us of kernel over the 8 launches (1346 PMTL-OK runs,
# p25-p75 3-10/us), so under ~16 us the series stays below MIN_STALL_SAMPLES (64).
# Once the timeline reports exactly that, one PM-only collection (sections only,
# far lighter than --set full) at the interval that doubles the floor replaces it.
REPORT_PM = "/tmp/stallprice_pm"
PM_MIN_INTERVAL = 128
_COARSE_RE = re.compile(r"^PMTL-FAIL reason=interval-too-coarse \(max (\d+) stall samples < (\d+)")


def finer_interval(interval: int, samples: int, floor: int) -> int | None:
    """The interval that brings ``samples`` (read at ``interval``) to twice ``floor``,
    rounded down to a power of two; None when it would not be finer or nothing
    was sampled (dead series: no interval revives them)."""
    if samples <= 0 or floor <= 0:
        return None
    want = interval * samples / (2.0 * floor)
    p = PM_MIN_INTERVAL
    while p * 2 <= want:
        p *= 2
    return p if p < interval else None


def build_pm_cmd(pm_interval: int, app: list | None = None) -> list:
    """The PM sections alone, same window and launch set as the pricing run."""
    return (["ncu", "--target-processes", "all", "--profile-from-start", "off",
             "--launch-count", "8", "-f", "-o", REPORT_PM]
            + pm_sections(pm_interval) + list(app or DEFAULT_APP))


def _timeline(pm_timeline_tools, pm_interval: int, app: list | None) -> None:
    """The timeline off the pricing collection, re-collected once finer when it
    reports interval-too-coarse. The first attempt's block is held back when the
    retry replaces it: the driver banks the FIRST PM TIMELINE block in a log."""
    lines: list = []
    pm_timeline_tools.timeline_report(REPORT + ".ncu-rep", out=lines.append)
    m = next((m for m in map(_COARSE_RE.match, lines) if m), None)
    finer = finer_interval(pm_interval, int(m.group(1)), int(m.group(2))) if m else None
    if finer is None:
        for ln in lines:
            print(ln)
        return
    print(f"[stall_price] timeline at interval {pm_interval} starved ({m.group(0)}) "
          f"— re-collecting the PM sections alone at {finer}", flush=True)
    cmd = build_pm_cmd(finer, app)
    clear_report(REPORT_PM)
    print(f"[stall_price] collecting: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(REPORT_PM + ".ncu-rep"):
        tail = (proc.stdout + proc.stderr)[-800:]
        print(f"[stall_price] finer PM collection produced no report (rc="
              f"{proc.returncode}); keeping the first verdict. tail:\n{tail}", flush=True)
        for ln in lines:
            print(ln)
        return
    pm_timeline_tools.timeline_report(REPORT_PM + ".ncu-rep")


def _kw_if_accepted(fn, name: str, value) -> dict:
    """{name: value} when ``fn`` takes that keyword — the page readers are
    seeded per tree and an older copy without the kernel-name argument must
    keep working (it falls back to the launch index / largest chunk)."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {name: value}
    return {}


def _orch_dossier(sass_text=None, launch=None, kernel=None) -> None:
    """The student's orchestration dossier rides the same collection.

    The stall ledger prices WHERE time leaks; the dossier transcribes WHAT
    sync structure was actually built — barrier objects, cross-role edges,
    in-flight depths. The driver banks it from this log and diffs it against
    the reference's recon-time dossier, giving the diagnosis a structural
    face the stall taxonomy cannot provide. Strictly additive: any failure
    here degrades to the pre-dossier behavior and must not cost the ledger.

    ``sass_text`` is the page already rendered for the segment ledger; the
    page carries every captured launch (--launch-count 8 -> eight copies),
    and ``launch`` (the dominant action's index) tells orch_tools which one
    is the kernel — the ref's recon dossier came off a --launch-count 1 page,
    so the student's must be one launch too or every count is 8x.
    """
    try:
        import orch_tools
        if sass_text is None:
            proc = subprocess.run(
                ["ncu", "--import", REPORT + ".ncu-rep", "--page", "source",
                 "--print-source", "sass"],
                capture_output=True, text=True, timeout=900)
            sass_text = proc.stdout
            print(f"[stall_price] SASS page rendered for the dossier alone "
                  f"(rc={proc.returncode})", flush=True)
        n = len(sass_text.splitlines())
        print(f"[stall_price] SASS page for the dossier: {n} lines", flush=True)
        if n < 200:
            print("[stall_price] page too small for a dossier — skipping "
                  "(ledger unaffected)", flush=True)
            return
        orch_tools.report(sass_text, launch=launch,
                          **_kw_if_accepted(orch_tools.report, "kernel", kernel or ""))
    except Exception as e:  # noqa: BLE001 — additive channel, never fatal
        print(f"[stall_price] dossier skipped ({type(e).__name__}: {e}) — "
              "ledger unaffected", flush=True)


def _role_exposure(sass_text=None, launch=None, kernel=None) -> None:
    """The role exposure ledger rides the same collection: the dossier's
    regions x the PC-sampled stalls x the PM tensor plateau (or the aggregate
    tensor metric when the run had no PM sections). Additive like the dossier:
    any failure degrades to the pre-join behavior and never costs the ledger."""
    try:
        import role_exposure_tools
        role_exposure_tools.report(REPORT + ".ncu-rep", sass_text=sass_text,
                                   launch=launch,
                                   **_kw_if_accepted(role_exposure_tools.report,
                                                     "kernel_substr", kernel or ""))
    except Exception as e:  # noqa: BLE001 — additive channel, never fatal
        print(f"[stall_price] role exposure skipped ({type(e).__name__}: {e}) — "
              "ledger unaffected", flush=True)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # everything after --app is the launch target's own argv: a recipe flag
    # there belongs to the app, not to this script (and is never stripped)
    recipe = args[:args.index("--app")] if "--app" in args else args
    pm = "--pm" in recipe
    pm_interval = PM_INTERVAL
    if "--pm-interval" in recipe:
        pm_interval = int(recipe[recipe.index("--pm-interval") + 1])
    if "--workload" in recipe:
        # one workload of a group card (the driver prices the card's meta_workloads one
        # job each): the launch target's ncu_window follows the variable, which every
        # process ncu starts inherits
        want = (recipe[recipe.index("--workload") + 1:] or [""])[0]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", want):
            print(f"STALL-FAIL reason=bad-args (--workload {want!r})")
            return 0
        os.environ[WORKLOAD_SCOPE_ENV] = want
        print(f"[stall_price] workload scope: {want} ({WORKLOAD_SCOPE_ENV})", flush=True)
    app = None
    if "--app" in args:
        app = args[args.index("--app") + 1:]
        if not app:
            print("STALL-FAIL reason=bad-args (--app given without a launch target)")
            return 0

    os.environ["CUTE_DSL_LINEINFO"] = "1"
    # a CUDA C++ kernel: the card's loader (load_cuda_entry) gives this run its own
    # -lineinfo build, so the ledger keys on CUDA source lines instead of PC buckets; a caller
    # that only wants role exposure (the lever pricing's role pass) passes "0"
    os.environ.setdefault("FK_CUDA_LINEINFO", "1")
    os.environ["CUTE_DSL_KEEP_IR"] = "1"
    os.environ["CUTE_DSL_DUMP_DIR"] = IR_DIR
    os.environ["CUTE_DSL_CACHE_DIR"] = "/tmp/dslcache_lineinfo"
    os.makedirs(IR_DIR, exist_ok=True)

    pm_collected = pm
    ok = _collect(pm, pm_interval, **({"app": app} if app else {}))
    if not ok and pm:
        # Ledger integrity first: the pricing channel predates the timeline and
        # a widened section set must never cost the diagnosis its ledger.
        print("[stall_price] combined collection produced no report — retrying "
              "WITHOUT PM sections (the ledger must not regress)", flush=True)
        pm_collected = False
        ok = _collect(False, **({"app": app} if app else {}))
    if not ok:
        print("STALL-FAIL reason=no-report (ncu produced no .ncu-rep — see tail above)")
        if pm:
            print("PMTL-FAIL reason=no-report (same collection)")
        return 0  # verdict travels in the line, not the exit code

    sys.path.insert(0, "notes")
    import stall_tools
    tee = _Tee(sys.stdout)
    sys.stdout = tee
    try:
        _analyze(stall_tools, pm, pm_collected, pm_interval, app)
    finally:
        sys.stdout = tee.stream
    emit_bundle(tee.text())
    return 0  # ditto: a FAIL verdict is a result, not a job failure


def _analyze(stall_tools, pm: bool, pm_collected: bool, pm_interval: int = PM_INTERVAL,
             app: list | None = None) -> None:
    stall_tools.price_report(REPORT + ".ncu-rep", ir=IR_DIR)
    # Per-segment account on top of the row ledger: the row ledger keys on source
    # lines and ~30% of samples (every HGMMA/DEPBAR/barrier — the @cute.jit
    # decorator has no line) pile into one unresolved bucket, hiding exactly the
    # scheduling rows a diagnosis needs. The segment ledger keys on PC.
    sass_text, launch, kernel = _render_page(stall_tools)
    stall_tools.segment_ledger(REPORT + ".ncu-rep", sass_text=sass_text)
    _orch_dossier(sass_text, launch, kernel)
    _role_exposure(sass_text, launch, kernel)

    if pm:
        if pm_collected:
            import pm_timeline_tools
            _timeline(pm_timeline_tools, pm_interval, app)
        else:
            print("PMTL-FAIL reason=combined-collection-failed (ncu crashed "
                  "with the PM sections; ledger was recollected without them — "
                  "retry the timeline standalone: python3 notes/pm_timeline_run.py)")


if __name__ == "__main__":
    sys.exit(main())
