"""CLI: run and inspect GPU jobs for an operator tree.

    python -m forgekernel.submit <op> submit --mode all [--branch <b>]
    python -m forgekernel.submit <op> submit --mode "cmd:nvidia-smi"   # smoke
    python -m forgekernel.submit <op> status <job_id>
    python -m forgekernel.submit <op> logs   <job_id>
    python -m forgekernel.submit <op> stop   <job_id>

``submit`` detaches a runner and prints ``job=<id>`` (forgekernel.submit.jobs); the job
runs on this machine or the tree's ssh host ([repo] delivery, forgekernel.submit.direct).
``--dry-run`` says what would run without running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forgekernel.submit import jobs
from forgekernel.submit.config import load_config


def _die(msg: str):
    sys.stderr.write(f"[submit] error: {msg}\n")
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forgekernel.submit")
    p.add_argument("op", help="operator name under operators/")
    p.add_argument("action", choices=["submit", "status", "logs", "stop", "_run"])
    p.add_argument("job_id", nargs="?", help="the job id (status / logs / stop)")
    p.add_argument("--mode", default="all", help="gpu_entry mode, or 'cmd:<command>'")
    p.add_argument("--branch", default=None, help="the branch to run (default: the tree's)")
    p.add_argument("--dry-run", action="store_true", help="say what would run, run nothing")
    p.add_argument("--base-dir", default=None, help="the directory holding operators/")
    args = p.parse_args(argv)

    cfg = load_config(args.op, base_dir=args.base_dir)
    if args.branch:
        cfg.branch = args.branch
    op_dir = Path(cfg.op_dir)

    if args.action == "submit":
        if args.dry_run:
            print(f"[submit] {cfg.delivery} delivery: would run gpu_entry {args.mode} of "
                  f"branch {cfg.branch} on {cfg.exec_host or 'this machine'} "
                  f"(GPU {cfg.exec_gpu})")
            return 0
        jid = jobs.spawn_runner(cfg, args.mode, cfg.branch)
        print(f"[submit] job={jid}")
        print(f"[submit] status: python -m forgekernel.submit {args.op} status {jid}")
        return 0
    job = (args.job_id or _die("this action requires a <job_id>")).removeprefix("tasks/")
    if args.action == "_run":
        return jobs.runner_main(cfg, job, args.mode, cfg.branch)
    if not jobs.is_job(op_dir, job):
        _die(f"no job {job} under {jobs.jobs_dir(op_dir)}")
    if args.action == "status":
        print(json.dumps(jobs.job_status(op_dir, job)))
        return 0
    if args.action == "logs":
        sys.stdout.write(jobs.job_log(op_dir, job))
        return 0
    stopped = jobs.stop_job(op_dir, job)
    print(json.dumps({"stopped": job} if stopped else {"not_running": job}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
