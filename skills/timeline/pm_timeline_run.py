"""Driver-seeded PM-timeline run — collects and renders the CURRENT kernel.

Frozen collection recipe for the phase-timeline channel: a probe runs
`cmd:python3 notes/pm_timeline_run.py` (cwd = repo root) and gets the
downsampled PM-sampling timelines + `PMTL-OK`/`PMTL-FAIL` verdict in the job
log. Exists for the same reason as `stall_price_run.py`: the recipe rides
every commit after seeding (git archive ships committed files only), so no
probe has to reinvent — and mis-tune — the interval arithmetic.

Recipe notes (measured on H100 / ncu 2025.4): the
warp-stall series sample AT `--pm-sampling-interval` (cycles) — 65536 is too
coarse for phase structure (~45 samples on a ~1.7 ms kernel), 8192 resolves
it (~370/launch). For sub-0.5 ms kernels pass a finer `--interval` (2048
resolves ~0.3 ms). `--profile-from-start off` because eval/ncu_run.py opens
the profiler window at the kernel (otherwise setup launches get profiled), and
`--launch-count 8` for two load-bearing reasons: dominant-by-duration kernel
selection is only a real check when there is more than one captured launch to
choose from (with 1 it degenerates to the identity and a mis-aim gets a green
PMTL-OK), and PM samples accumulate across every profiled launch, which is
what keeps fast kernels above the sample floor.
"""
import os
import subprocess
import sys

REPORT = "/tmp/pm_timeline"
INTERVAL = 8192            # cycles; phase-resolving default — see --interval
BUFFER = 268435456


def build_cmd(interval: int) -> list:
    return ["ncu", "--profile-from-start", "off", "--launch-count", "8",
            "--kill", "yes", "-f", "-o", REPORT,
            "--section", "PmSampling", "--section", "PmSampling_WarpStates",
            "--pm-sampling-interval", str(interval),
            "--pm-sampling-buffer-size", str(BUFFER),
            sys.executable, "eval/ncu_run.py"]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    interval = INTERVAL
    if "--interval" in args:
        interval = int(args[args.index("--interval") + 1])

    cmd = build_cmd(interval)
    # a GPU machine keeps /tmp across jobs: a collection that produces nothing must
    # not be read off the previous job's report (the existence check below is the gate)
    try:
        os.remove(REPORT + ".ncu-rep")
    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"[pm_timeline] could not remove the previous report ({e})", flush=True)
    print(f"[pm_timeline] collecting: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr)[-1500:]
    print(f"[pm_timeline] ncu rc={proc.returncode}; output tail:\n{tail}", flush=True)
    if not os.path.exists(REPORT + ".ncu-rep"):
        print("PMTL-FAIL reason=no-report (ncu produced no .ncu-rep — see tail above)")
        return 0  # verdict travels in the line, not the exit code

    sys.path.insert(0, "notes")
    import pm_timeline_tools
    pm_timeline_tools.timeline_report(REPORT + ".ncu-rep")
    return 0  # ditto: a FAIL verdict is a result, not a job failure


if __name__ == "__main__":
    sys.exit(main())
