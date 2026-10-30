#!/usr/bin/env python3
"""Self-evolution: each generation's best kernel becomes the next generation's reference.

    python scripts/self_evolve.py tasks/<op> [--step 1.05] [--max-gen N]

A generation is an ordinary run (`forge new`, then `forge run`). When it ends with a confirmed
best kernel faster than its reference by more than the noise floor, the next generation is
written as a new task, ``tasks/<op>-g<k>/``: the same task.toml, description and shapes, the
best kernel copied into its reference/ (so the new task carries everything it needs and can be
committed and rerun elsewhere), and a target ``--step`` times the new reference's speed.

The chain stops only when the hardware gives no more: at a generation whose gain is within the
noise floor, or one that ended without a confirmed best. The noise floor is the spread of the
generation's own same-job confirmation ratios, never below NOISE_FLOOR_MIN (the loop's plateau
margin). ``--max-gen`` caps the number of generations; there is no cap by default.

The reference has source, so every generation after the first runs in source mode: the
teacher reads the previous best, the student never does (the run seals it).

Only the public commands are used; nothing here is special to the harness.
"""

from __future__ import annotations

import argparse
import itertools
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgekernel import launch  # noqa: E402
from forgekernel.runs import list_runs  # noqa: E402
from forgekernel.task import (TaskError, dump_task_toml, load_task,  # noqa: E402
                              read_task_toml)

REF_DIR = "reference"
# == forgekernel.loop.parse.detect_plateau's default: below this, two readings of one
# kernel are not told apart, however tight a handful of confirmation jobs happened to be
NOISE_FLOOR_MIN = 0.015


def latest_run(task_name: str):
    runs = [r for r in list_runs() if r.task_name == task_name]
    return max(runs, key=lambda r: r.record().get("created", "")) if runs else None


def noise_floor(best: dict) -> float:
    """The relative gain a generation must clear: the spread of its same-job ratios
    ((max - min) / min), never below NOISE_FLOOR_MIN."""
    ratios = best.get("ratios") or []
    spread = (max(ratios) - min(ratios)) / min(ratios) if len(ratios) > 1 else 0.0
    return max(NOISE_FLOOR_MIN, spread)


def next_generation(task, best: dict, dest: Path, step: float) -> Path:
    """tasks/<op>-g<k>/: the task with ``best`` as its reference and target ``step``x."""
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it or start the chain elsewhere")
    if task.language != "cutedsl":
        raise ValueError("self-evolution needs a Python kernel to seal as the next reference "
                         f"(this task's language is {task.language!r})")
    dest.mkdir(parents=True)
    for name in ("description.md", "shapes.json"):
        shutil.copy2(task.dir / name, dest / name)
    if (task.dir / "inputs").is_dir():
        shutil.copytree(task.dir / "inputs", dest / "inputs")
    (dest / REF_DIR).mkdir()
    (dest / REF_DIR / "kernel.py").write_text(best["kernel"], encoding="utf-8")
    cfg = read_task_toml(task.dir)
    cfg["reference"] = {"run": f"{REF_DIR}/kernel.py:run", "source": REF_DIR}
    cfg["target"] = {**cfg.get("target", {}), "ratio": float(step)}
    cfg["target"].pop("ms", None)
    (dest / "task.toml").write_text(dump_task_toml(cfg), encoding="utf-8")
    load_task(dest)                       # the new generation must be a valid task
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="self_evolve.py", description=__doc__.split("\n\n")[0])
    p.add_argument("task", help="the first generation's task directory")
    p.add_argument("--step", type=float, default=1.05,
                   help="each new generation's target, x its reference's speed")
    p.add_argument("--max-gen", type=int, default=None,
                   help="stop after this many generations (default: no cap)")
    args = p.parse_args(argv)
    task_dir = Path(args.task).resolve()
    base = re.sub(r"-g\d+$", "", task_dir.name)
    for gen in itertools.count(1):
        try:
            task = load_task(task_dir)
        except TaskError as e:
            sys.stderr.write(f"[self-evolve] {e}\n")
            return 2
        print(f"[self-evolve] generation {gen}: {task_dir}", flush=True)
        rc = launch.cmd_new([str(task_dir)])
        run = latest_run(task.name) if rc == 0 else None
        if run is not None:
            rc = launch.cmd_run([str(run.dir)])
        best = launch.best_of(run, task) if run is not None else None
        if best is None or not best.get("kernel"):
            print(f"[self-evolve] generation {gen} ended without a confirmed best kernel "
                  f"(loop exit {rc}): the chain stops here", flush=True)
            return 0 if gen > 1 else 1
        gain = best.get("speedup")
        floor = noise_floor(best)
        print(f"[self-evolve] generation {gen}: best {best['sha'][:12]}, "
              + (f"{gain:.4f}x its reference" if gain else "speedup unknown")
              + f" (noise floor {floor:.1%})", flush=True)
        if gain is None or gain <= 1.0 + floor:
            print(f"[self-evolve] gain within the {floor:.1%} noise floor: the chain stops here",
                  flush=True)
            return 0
        if args.max_gen is not None and gen >= args.max_gen:
            print(f"[self-evolve] reached --max-gen {args.max_gen}", flush=True)
            return 0
        task_dir = next_generation(task, best, task_dir.parent / f"{base}-g{gen + 1}", args.step)
        print(f"[self-evolve] next generation: {task_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
