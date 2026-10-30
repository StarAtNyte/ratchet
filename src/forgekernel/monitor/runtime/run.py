#!/usr/bin/env python3
"""Correctness monitor, job side: run the card's own correctness task once per
profile with the monitor hook loaded, and report what each run saw.

    bash gpu_entry.sh cmd python3 .fk_monitor/run.py [--profiles base,seed1,...]

Standard library only — it runs under whatever ``python3`` the job's PATH offers,
which on some images has no torch; the bench it drives runs under the card's own
interpreter (gpu_entry decides), and that is where the hook lives.

Per profile: ``bash gpu_entry.sh correctness`` with FK_MON_PROFILE set and the hook
directory first on PYTHONPATH. The card's eval JSON gives its own verdict; the hook's
files give the kernel's output summaries and how much of the input the profile
actually reshaped. When a reshaped input fails the card's judge and the card has a
``cardref`` task (the reference run through the bench as the implementation), the
reference is run under the same profile, so a failure the reference shares is told
apart from one only the kernel has.

After every profile it prints ``=== FK MONITOR PROFILE ===`` and that profile's record
on one line; at the end, ``=== FK MONITOR JSON ===`` and the whole result. The harness
reads the whole result, or rebuilds a partial one from the profile blocks when the job
was cut short, and classifies it (forgekernel.monitor).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

MARKER = "=== FK MONITOR JSON ==="
PROFILE_MARKER = "=== FK MONITOR PROFILE ==="
VERSION = 1
DEFAULT_PROFILES = ("base", "seed1", "zero_block", "int_edges", "amp", "spikes", "repeat",
                    "tiny", "const_rows", "seed2")
STRESS = ("amp", "tiny", "spikes", "zero_block", "const_rows", "int_edges")
TAIL_LINES = 25

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HOOK_DIR = os.path.join(HERE, "hook")


def _eval_json(text: str) -> dict | None:
    """The card's eval block: the last JSON object after ``=== JSON ===`` that
    carries a ``correctness`` key (a one-line or an indented dump)."""
    at = text.rfind("=== JSON ===")
    region = text[at:] if at != -1 else text
    dec = json.JSONDecoder()
    found = None
    for m in re.finditer(r"\{", region):
        try:
            obj, _ = dec.raw_decode(region, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and "correctness" in obj:
            found = obj
    return found


def _case_brief(corr: dict) -> list[dict]:
    cases = corr.get("cases")
    out = []
    if isinstance(cases, list):
        for c in cases[:24]:
            if not isinstance(c, dict):
                continue
            name = c.get("case") or c.get("name") or c.get("label")
            ok = c.get("pass", c.get("pass_", c.get("ok")))
            nums = {k: v for k, v in c.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)}
            out.append({"case": name, "pass": ok,
                        **dict(list(nums.items())[:6])})
    return out


def _hook_files(d: str) -> list[dict]:
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if fn.startswith("hook-") and fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as fh:
                    out.append(json.load(fh))
            except (OSError, ValueError):
                continue
    return out


def _merge_hooks(hooks: list[dict]) -> dict:
    """One summary per profile run: the process(es) that imported the kernel carry
    its calls; every process contributes what it reshaped and any hook error."""
    calls, errors, kmods = [], [], []
    perturbed = reseeded = unrecorded = 0
    for h in hooks:
        calls.extend(h.get("calls") or [])
        errors.extend(h.get("errors") or [])
        perturbed += int(h.get("perturbed") or 0)
        reseeded += int(h.get("reseeded") or 0)
        unrecorded += int(h.get("unrecorded") or 0)
        if h.get("kernel_module"):
            kmods.append(h["kernel_module"])
    return {"processes": len(hooks), "kernel_module": kmods[0] if kmods else None,
            "calls": calls, "unrecorded_calls": unrecorded, "perturbed": perturbed,
            "reseeded": reseeded, "errors": errors[:10]}


def _run_task(entry: str, task: str, profile: str, out_dir: str, timeout_s: float,
              kernel: str | None) -> dict:
    env = dict(os.environ)
    env["FK_MON_PROFILE"] = profile
    env["FK_MON_OUT"] = out_dir
    env["FK_MON_ROOT"] = ROOT
    if kernel:
        env["FK_MON_KERNEL"] = kernel
    env["PYTHONPATH"] = HOOK_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TASK_TIMEOUT"] = str(max(60, int(timeout_s)))
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(["bash", entry, task], cwd=ROOT, env=env, capture_output=True,
                           text=True, timeout=timeout_s + 60)
        rc, text = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = 137
        text = ""
        for part in (e.stdout, e.stderr):
            if part:
                text += part.decode(errors="replace") if isinstance(part, bytes) else part
    ev = _eval_json(text)
    corr = (ev or {}).get("correctness") or {}
    res = {"task": task, "rc": rc, "seconds": round(time.time() - t0, 1),
           "timed_out": timed_out, "card_pass": corr.get("pass") if ev else None,
           "cases": _case_brief(corr), "hook": _merge_hooks(_hook_files(out_dir))}
    if ev is None or corr.get("pass") is not True:
        res["tail"] = text.splitlines()[-TAIL_LINES:]
    return res


def _has_task(entry: str, task: str) -> bool:
    try:
        with open(entry, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
    except OSError:
        return False
    return re.search(rf"^\s*(?:[\w|]+\|)?{re.escape(task)}(?:\|[\w|]+)?\)", src, re.M) is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    ap.add_argument("--entry", default="gpu_entry.sh")
    ap.add_argument("--kernel", default="kernel.py",
                    help="the student's kernel file, relative to the tree root")
    ap.add_argument("--budget-s", type=float, default=None,
                    help="wall-clock for every run together (default 85%% of TASK_TIMEOUT, "
                         "else 3600)")
    ap.add_argument("--no-cardref", action="store_true")
    args = ap.parse_args(argv)

    entry = os.path.join(ROOT, args.entry)
    kernel = os.path.join(ROOT, args.kernel)
    kernel = kernel if os.path.isfile(kernel) else None
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    outer = float(os.environ.get("TASK_TIMEOUT") or 0)
    budget = args.budget_s or (0.85 * outer if outer > 0 else 3600.0)
    cardref = not args.no_cardref and _has_task(entry, "cardref")
    work = tempfile.mkdtemp(prefix="fk_monitor_")
    t0 = time.time()
    result = {"version": VERSION, "profiles": [], "skipped": [], "cardref_task": cardref,
              "kernel_file": args.kernel if kernel else None, "budget_s": round(budget)}
    print(f"[fk-monitor] {len(profiles)} profile(s), budget {budget:.0f}s, "
          f"cardref {'available' if cardref else 'absent'}", flush=True)
    slowest = 0.0
    try:
        for prof in profiles:
            left = budget - (time.time() - t0)
            if left < max(60.0, 1.2 * slowest):
                result["skipped"].append({"profile": prof, "why": f"budget ({left:.0f}s left)"})
                print(f"[fk-monitor] {prof}: skipped, {left:.0f}s of budget left", flush=True)
                continue
            rec = {"profile": prof,
                   **_run_task(entry, "correctness", prof, os.path.join(work, prof), left, kernel)}
            slowest = max(slowest, rec["seconds"])
            if (cardref and prof in STRESS + ("seed1", "seed2") and rec["card_pass"] is False
                    and budget - (time.time() - t0) > max(60.0, 1.2 * slowest)):
                ref = _run_task(entry, "cardref", prof, os.path.join(work, prof + "-ref"),
                                budget - (time.time() - t0), None)
                rec["ref_pass"] = ref["card_pass"]
                rec["ref_seconds"] = ref["seconds"]
                rec["ref_perturbed"] = ref["hook"]["perturbed"] + ref["hook"]["reseeded"]
            result["profiles"].append(rec)
            # each profile as it finishes: a job killed later still leaves what ran
            print(PROFILE_MARKER, flush=True)
            print(json.dumps(rec, separators=(",", ":")), flush=True)
            h = rec["hook"]
            print(f"[fk-monitor] {prof}: card {rec['card_pass']} rc={rec['rc']} "
                  f"{rec['seconds']}s; kernel calls seen {len(h['calls'])}, "
                  f"inputs reshaped {h['perturbed']}, reseeds {h['reseeded']}"
                  + (f"; reference under the same profile: {rec['ref_pass']}"
                     if "ref_pass" in rec else ""), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        result["seconds"] = round(time.time() - t0, 1)
        print(MARKER, flush=True)
        print(json.dumps({**result, "complete": True}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
