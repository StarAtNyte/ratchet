"""``forge <op> monitor``: run the correctness monitor on one tree, on demand.

    forge <op> monitor [--workspace WS] [--sha SHA] [--profiles base,seed1,...]
                       [--budget-s N] [--dry-run]

Defaults: the workspace is ``operators/<op>-ws`` when it exists; the sha is the one
in ``operators/<op>/best_confirmation.json``, else the workspace HEAD. One GPU job
(the card's own submit config), advisory result appended to
``operators/<op>/correctness_monitor.jsonl``. Exit 0 = ok, 1 = warn, 2 = alarm,
3 = could not run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from forgekernel import monitor
from forgekernel.paths import operators_dir


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_sha(op_dir: Path, ws: Path) -> str:
    conf = op_dir / "best_confirmation.json"
    if conf.exists():
        try:
            sha = json.loads(conf.read_text(encoding="utf-8")).get("sha")
            if sha:
                return str(sha)
        except (OSError, ValueError):
            pass
    out = subprocess.run(["git", "-C", str(ws), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forge <op> monitor", description=__doc__.split("\n\n")[0])
    p.add_argument("op")
    p.add_argument("--workspace", default=None,
                   help="the student workspace (default operators/<op>-ws)")
    p.add_argument("--sha", default=None, help="tree to monitor (default: confirmed best, "
                                               "else the workspace HEAD)")
    p.add_argument("--profiles", default=",".join(monitor.DEFAULT_PROFILES))
    p.add_argument("--kernel-file", default="kernel.py")
    p.add_argument("--budget-s", type=float, default=None,
                   help="wall-clock for all profiles (default: 85%% of the job timeout)")
    p.add_argument("--branch", default=None, help="submit branch (default <op>-monitor)")
    p.add_argument("--base-dir", default=None, help="repo root override (testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the tree, branch and job mode; submit nothing")
    args = p.parse_args(argv)

    ops = Path(args.base_dir) / "operators" if args.base_dir else operators_dir(_repo_root())
    op_dir = ops / args.op
    ws = Path(args.workspace) if args.workspace else op_dir.parent / f"{args.op}-ws"
    if not (ws / ".git").exists():
        sys.stderr.write(f"[monitor] error: no workspace repo at {ws} (pass --workspace)\n")
        return 3
    profiles = tuple(x.strip() for x in args.profiles.split(",") if x.strip())
    try:
        mode = monitor.job_mode(profiles, args.kernel_file, args.budget_s)
    except ValueError as e:
        sys.stderr.write(f"[monitor] error: {e}\n")
        return 3
    sha = args.sha or _default_sha(op_dir, ws)
    if not sha:
        sys.stderr.write("[monitor] error: no sha to monitor\n")
        return 3
    branch = args.branch or f"{args.op}-monitor"

    from forgekernel.loop.cli import RealBackend, _load_loop_config, resolve_job_env_files
    be = RealBackend(args.op, ws, None, "forge-monitor", "forge-monitor@local")
    try:
        be.job_env_files = resolve_job_env_files(_load_loop_config(op_dir), op_dir,
                                                 kernel_file=args.kernel_file)
    except ValueError as e:
        sys.stderr.write(f"[monitor] error: {e}\n")
        return 3
    if args.dry_run:
        tree = monitor.overlay(be.with_file, sha)
        print(f"[monitor] sha {sha}\n[monitor] tree {tree} (runtime overlaid)\n"
              f"[monitor] branch {branch}\n[monitor] mode {mode}")
        return 0
    try:
        rec = monitor.run(be, sha, branch=branch, op_dir=op_dir, profiles=profiles,
                          kernel_file=args.kernel_file, budget_s=args.budget_s,
                          trigger="manual", log=lambda s: print(f"[monitor] {s}", flush=True))
    except Exception as e:  # noqa: BLE001 — a CLI reports, it does not trace
        sys.stderr.write(f"[monitor] error: {e.__class__.__name__}: {e}\n")
        return 3
    return {"ok": 0, "warn": 1, "alarm": 2}[rec["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
