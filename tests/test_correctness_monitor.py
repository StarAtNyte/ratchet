"""Correctness monitor (forgekernel.monitor): the job-side runner and hook driven
end to end on a toy card (CPU torch; skipped without it), the harness-side
classification, the job mode and the overlay/submit path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from forgekernel import monitor

GPU_ENTRY = """#!/bin/bash
set -euo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$W"
export PYTHONPATH="$W:$W/eval:${PYTHONPATH:-}"
PY="${FK_PYTHON:-python3}"
TASK="${1:-help}"
shift || true
case "$TASK" in
    correctness)
        "$PY" eval/bench.py --mode correctness
        ;;
    cardref)
        "$PY" eval/bench.py --mode correctness --impl ref
        ;;
    cmd)
        bash -c "$*"
        ;;
esac
"""

# A packed-output op in the shape of fused_attn_rope_cast: attention, then a
# per-32-channel scale. The reference guards an all-zero group; the buggy kernel
# divides by zero there, which randn inputs never reach.
BENCH = '''
import argparse, json, sys, torch
D, G = 256, 32
SEED = 20260921

def make(seed):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(64, 64, generator=g) / 10
    k = torch.randn(96, 64, generator=g) / 10
    v = torch.randn(96, D, generator=g) / 10
    return q, k, v

def reference(q, k, v):
    o = torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v
    og = o.view(o.shape[0], D // G, G)
    amax = og.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    return (og / s).round().mul(s).view(o.shape[0], D)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="correctness")
ap.add_argument("--impl", default="kernel")
a = ap.parse_args()
if a.impl == "ref":
    fn = reference
else:
    import kernel
    fn = kernel.call
cases, ok = [], True
for i, seed in enumerate((SEED, SEED + 1)):
    q, k, v = make(seed)
    got, want = fn(q, k, v), reference(q, k, v)
    cos = torch.nn.functional.cosine_similarity(got.flatten().double(),
                                                want.flatten().double(), dim=0).item()
    good = bool(torch.isfinite(got).all()) and (1 - cos) <= 1e-3
    ok &= good
    cases.append({"case": f"seed{i}", "pass": good, "cos_diff": 1 - cos})
print("=== JSON ===")
print(json.dumps({"correctness": {"pass": ok, "cases": cases}}))
'''

KERNEL_OK = '''
import torch
D, G = 256, 32

def _pack(o):
    og = o.view(o.shape[0], D // G, G)
    amax = og.abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    return (og / s).round().mul(s).view(o.shape[0], D)

def call(q, k, v):
    return _pack(torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v)
'''

KERNEL_ZERO_GROUP_NAN = KERNEL_OK.replace(
    "s = torch.where(amax > 0, amax / 448, torch.ones_like(amax))", "s = amax / 448")

# inputs built without the RNG: nothing for the stress profiles to reshape
BENCH_NO_RNG = BENCH.replace(
    "    g = torch.Generator().manual_seed(seed)\n"
    "    q = torch.randn(64, 64, generator=g) / 10\n"
    "    k = torch.randn(96, 64, generator=g) / 10\n"
    "    v = torch.randn(96, D, generator=g) / 10\n",
    "    q = torch.linspace(-1, 1, 64 * 64).view(64, 64).sin() / 10\n"
    "    k = torch.linspace(-1, 1, 96 * 64).view(96, 64).cos() / 10\n"
    "    v = torch.linspace(-2, 2, 96 * D).view(96, D).sin() / 10\n")


# Per-row key counts, the shape of topk_length: the bench fills them to the maximum,
# so an empty row never reaches the kernel on the card's own inputs.
BENCH_LEN = BENCH.replace(
    "def reference(q, k, v):\n    o = torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v\n",
    "def reference(q, k, v, n):\n"
    "    s = (q @ k.T * 64 ** -0.5).masked_fill(\n"
    "        torch.arange(k.shape[0])[None, :] >= n[:, None], float('-inf'))\n"
    "    o = torch.nan_to_num(torch.softmax(s, -1)) @ v\n").replace(
    "    return q, k, v\n", "    return q, k, v, torch.full((64,), 96, dtype=torch.int32)\n").replace(
    "    q, k, v = make(seed)\n    got, want = fn(q, k, v), reference(q, k, v)\n",
    "    q, k, v, n = make(seed)\n    got, want = fn(q, k, v, n), reference(q, k, v, n)\n")

KERNEL_LEN_OK = KERNEL_OK.replace(
    "def call(q, k, v):\n    return _pack(torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v)\n",
    "def call(q, k, v, n):\n"
    "    live = torch.full((k.shape[0],), 1, dtype=torch.int32)   # the kernel's own buffer\n"
    "    assert int(live.sum()) == k.shape[0], 'kernel buffer was reshaped'\n"
    "    s = (q @ k.T * 64 ** -0.5).masked_fill(\n"
    "        torch.arange(k.shape[0])[None, :] >= n[:, None], float('-inf'))\n"
    "    return _pack(torch.nan_to_num(torch.softmax(s, -1)) @ v)\n")

KERNEL_EMPTY_ROW_NAN = KERNEL_LEN_OK.replace("torch.nan_to_num(torch.softmax(s, -1))",
                                             "torch.softmax(s, -1)")

# Destination passing: the bench allocates the output and the kernel fills it.
BENCH_DPS = BENCH.replace(
    "    got, want = fn(q, k, v), reference(q, k, v)\n",
    "    out = torch.empty(q.shape[0], D)\n"
    "    if a.impl == 'ref':\n"
    "        out.copy_(fn(q, k, v))\n"
    "    else:\n"
    "        fn(q, k, v, out)\n"
    "    got, want = out, reference(q, k, v)\n")

# The task suite's way: the inputs come from the judge's make_inputs, compiled by the
# bench from source (filename "<op judge>"), and the kernel is loaded by file path
# under a name of the bench's own (eval/leaderboard.py load_entry)
BENCH_TASK = BENCH.replace(
    "def make(seed):\n"
    "    g = torch.Generator().manual_seed(seed)\n"
    "    q = torch.randn(64, 64, generator=g) / 10\n"
    "    k = torch.randn(96, 64, generator=g) / 10\n"
    "    v = torch.randn(96, D, generator=g) / 10\n"
    "    return q, k, v\n",
    "JUDGE = '''\n"
    "import torch\n"
    "def make_inputs(seed):\n"
    "    g = torch.Generator().manual_seed(seed)\n"
    "    return (torch.randn(64, 64, generator=g) / 10, torch.randn(96, 64, generator=g) / 10,\n"
    "            torch.randn(96, 256, generator=g) / 10)\n"
    "'''\n"
    "ns = {}\n"
    "exec(compile(JUDGE, '<toy judge>', 'exec'), ns)\n"
    "make = ns['make_inputs']\n").replace(
    "    import kernel\n    fn = kernel.call\n",
    "    import importlib.util, os\n"
    "    spec = importlib.util.spec_from_file_location('lb_card_kernel', os.path.abspath('kernel.py'))\n"
    "    mod = importlib.util.module_from_spec(spec)\n"
    "    spec.loader.exec_module(mod)\n"
    "    hook = sys.modules.get('fk_monitor_hook')\n"
    "    if hook is not None and hasattr(hook, 'watch_loaded'):\n"
    "        hook.watch_loaded(mod, os.path.abspath('kernel.py'))\n"
    "    fn = mod.call\n")

KERNEL_DPS_OK = KERNEL_OK.replace(
    "def call(q, k, v):\n    return _pack(torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v)\n",
    "def call(q, k, v, out):\n    out.copy_(_pack(torch.softmax(q @ k.T * 64 ** -0.5, -1) @ v))\n")

KERNEL_DPS_ZERO_GROUP_NAN = KERNEL_DPS_OK.replace(
    "s = torch.where(amax > 0, amax / 448, torch.ones_like(amax))", "s = amax / 448")

# two identical runs that disagree (a stand-in for a race): the repeat run's result
# is 1% larger — invisible to the card's cosine judge, not to the checksum
KERNEL_DPS_RACE = KERNEL_DPS_OK.replace(
    "    out.copy_(_pack(",
    "    import os\n"
    "    f = 1.01 if os.environ.get('FK_MON_PROFILE') == 'repeat' else 1.0\n"
    "    out.copy_(f * _pack(")


def _card(tmp_path: Path, kernel: str, bench: str = BENCH, cardref: bool = True) -> Path:
    root = tmp_path / "card"
    (root / "eval").mkdir(parents=True)
    entry = GPU_ENTRY if cardref else GPU_ENTRY.replace(
        '    cardref)\n        "$PY" eval/bench.py --mode correctness --impl ref\n        ;;\n', "")
    (root / "gpu_entry.sh").write_text(entry)
    (root / "eval" / "bench.py").write_text(bench)
    (root / "kernel.py").write_text(kernel)
    for rel, text in monitor.runtime_files().items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _run(root: Path, profiles: str) -> dict:
    env = dict(os.environ, FK_PYTHON=sys.executable)
    env.pop("PYTHONPATH", None)
    p = subprocess.run([sys.executable, str(root / ".fk_monitor" / "run.py"),
                        "--profiles", profiles, "--budget-s", "600"],
                       cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    res = monitor.parse_log(p.stdout)
    assert res is not None, p.stdout[-2000:]
    return res


# --- end to end on the toy card ------------------------------------------------
needs_torch = pytest.mark.skipif(
    subprocess.run([sys.executable, "-c", "import torch"], capture_output=True).returncode != 0,
    reason="needs torch")


@needs_torch
def test_zero_group_nan_is_an_alarm_the_card_misses_on_its_own_inputs(tmp_path):
    res = _run(_card(tmp_path, KERNEL_ZERO_GROUP_NAN), "base,zero_block,seed1,repeat,amp")
    recs = {r["profile"]: r for r in res["profiles"]}
    assert recs["base"]["card_pass"] is True           # the card's own inputs never hit it
    assert recs["zero_block"]["card_pass"] is False
    assert recs["zero_block"]["ref_pass"] is True        # the reference survives the same inputs
    calls = recs["zero_block"]["hook"]["calls"]
    assert calls and sum(o["nan"] for c in calls for o in c["outputs"]) > 0
    v = monitor.classify(res)
    assert v["status"] == "alarm"
    alarms = [f for f in v["findings"] if f["severity"] == "alarm"]
    assert any(f["profile"] == "zero_block" and "NaN" in f["what"] for f in alarms)
    assert any(f["profile"] == "zero_block" and "reference" in f["what"] for f in alarms)
    assert not any(f["profile"] in ("base", "seed1", "repeat", "amp") and f["severity"] == "alarm"
                   for f in v["findings"])


@needs_torch
def test_a_correct_kernel_reads_ok_and_every_profile_reshapes_the_inputs(tmp_path):
    res = _run(_card(tmp_path, KERNEL_OK), "base,seed1,zero_block,spikes,repeat,const_rows")
    v = monitor.classify(res)
    assert v["status"] == "ok", v
    assert set(v["ran"]) == {"base", "seed1", "zero_block", "spikes", "repeat", "const_rows"}
    recs = {r["profile"]: r for r in res["profiles"]}
    base_in = recs["base"]["hook"]["calls"][0]["in_sums"]
    assert recs["seed1"]["hook"]["calls"][0]["in_sums"] != base_in
    assert recs["repeat"]["hook"]["calls"][0]["in_sums"] == base_in
    assert recs["seed1"]["hook"]["reseeded"] > 0
    assert recs["spikes"]["hook"]["perturbed"] > 0


@needs_torch
def test_a_task_judges_draws_and_a_by_path_kernel_are_both_seen(tmp_path):
    assert "import kernel" not in BENCH_TASK and "<toy judge>" in BENCH_TASK
    res = _run(_card(tmp_path, KERNEL_ZERO_GROUP_NAN, bench=BENCH_TASK), "base,zero_block,seed1")
    recs = {r["profile"]: r for r in res["profiles"]}
    assert recs["base"]["hook"]["calls"], "the by-path kernel was not watched"
    assert recs["zero_block"]["hook"]["perturbed"] > 0, "the judge's draws were not reshaped"
    v = monitor.classify(res)
    assert not v["not_applicable"], v
    assert v["status"] == "alarm" and any(f["profile"] == "zero_block" for f in v["findings"])


def test_load_entry_hands_the_kernel_to_the_monitor_hook(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(monitor.__file__).resolve().parents[1] / "card_template" / "eval"))
    try:
        import leaderboard as lb_eval
    finally:
        sys.path.pop(0)
    seen = []
    fake = type(sys)("fk_monitor_hook")
    fake.watch_loaded = lambda mod, origin: seen.append((mod.__name__, origin))
    monkeypatch.setitem(sys.modules, "fk_monitor_hook", fake)
    (tmp_path / "kernel.py").write_text("def run(x):\n    return x\n")
    fn = lb_eval.load_entry(tmp_path / "kernel.py", "run")
    assert fn(3) == 3
    assert seen == [(f"lb_{tmp_path.name}_kernel", str((tmp_path / "kernel.py").resolve()))]


@needs_torch
def test_inputs_not_drawn_from_the_rng_make_stress_profiles_not_applicable(tmp_path):
    res = _run(_card(tmp_path, KERNEL_ZERO_GROUP_NAN, bench=BENCH_NO_RNG), "base,zero_block,seed1")
    v = monitor.classify(res)
    assert set(v["not_applicable"]) == {"zero_block", "seed1"}
    assert v["status"] == "ok"


@needs_torch
def test_without_cardref_a_stress_failure_is_a_warning(tmp_path):
    res = _run(_card(tmp_path, KERNEL_ZERO_GROUP_NAN, cardref=False), "base,zero_block")
    assert res["cardref_task"] is False
    v = monitor.classify(res)
    # the NaN itself is still an alarm; the verdict line without a reference is a warning
    assert any(f["severity"] == "warn" and "not run under it" in f["what"] for f in v["findings"])
    assert any(f["severity"] == "alarm" and "NaN" in f["what"] for f in v["findings"])


@needs_torch
def test_int_edges_reaches_an_empty_row_the_card_never_draws(tmp_path):
    res = _run(_card(tmp_path, KERNEL_EMPTY_ROW_NAN, bench=BENCH_LEN), "base,int_edges,zero_block")
    recs = {r["profile"]: r for r in res["profiles"]}
    assert recs["base"]["card_pass"] is True
    assert recs["zero_block"]["card_pass"] is True       # float reshaping cannot empty a row
    assert recs["int_edges"]["card_pass"] is False and recs["int_edges"]["ref_pass"] is True
    v = monitor.classify(res)
    assert v["status"] == "alarm"
    assert all(f["profile"] == "int_edges" for f in v["findings"] if f["severity"] == "alarm")


@needs_torch
def test_int_edges_leaves_the_kernels_own_buffers_alone(tmp_path):
    # KERNEL_LEN_OK asserts its own torch.full buffer is intact; empty rows are handled
    res = _run(_card(tmp_path, KERNEL_LEN_OK, bench=BENCH_LEN), "base,int_edges")
    v = monitor.classify(res)
    assert v["status"] == "ok", v
    assert "int_edges" in v["ran"]


@needs_torch
def test_a_kernel_writing_its_out_argument_is_watched_like_one_returning(tmp_path):
    # the same zero-group NaN, delivered through the caller's `out` buffer: the NaN is
    # the kernel's output, not a non-finite input that would void the profile
    res = _run(_card(tmp_path, KERNEL_DPS_ZERO_GROUP_NAN, bench=BENCH_DPS),
               "base,zero_block,repeat")
    recs = {r["profile"]: r for r in res["profiles"]}
    assert recs["zero_block"]["card_pass"] is False and recs["zero_block"]["ref_pass"] is True
    [c0, *_] = recs["zero_block"]["hook"]["calls"]
    assert c0["written"] == [3] and c0["inputs_nonfinite"] == 0
    assert c0["in_sums"][3] is None
    [out] = [o for o in c0["outputs"] if o["slot"] == "arg3"]
    assert out["nan"] > 0
    v = monitor.classify(res)
    assert v["status"] == "alarm", v
    assert "zero_block" in v["ran"] and "zero_block" not in v["not_applicable"]
    alarms = [f for f in v["findings"] if f["severity"] == "alarm"]
    assert any(f["profile"] == "zero_block" and "NaN" in f["what"] for f in alarms)
    assert any(f["profile"] == "zero_block" and "reference" in f["what"] for f in alarms)


@needs_torch
def test_a_correct_out_argument_kernel_reads_ok_and_its_output_is_checked_for_races(tmp_path):
    v = monitor.classify(_run(_card(tmp_path / "ok", KERNEL_DPS_OK, bench=BENCH_DPS),
                              "base,zero_block,repeat"))
    assert v["status"] == "ok", v
    assert set(v["ran"]) == {"base", "zero_block", "repeat"}
    v = monitor.classify(_run(_card(tmp_path / "race", KERNEL_DPS_RACE, bench=BENCH_DPS),
                              "base,repeat"))
    assert v["status"] == "warn", v
    assert any(f["profile"] == "repeat" and "race" in f["what"] for f in v["findings"])


@needs_torch
def test_hook_reads_arguments_before_the_call_and_reports_the_written_ones(tmp_path):
    root = tmp_path / "k"
    root.mkdir()
    (root / "kernel.py").write_text(textwrap.dedent('''
        def fill(x, out):
            out.copy_(x * 2)
        def fill_ret(x, out):
            out.copy_(x * 2)
            return out
        def bump_(x):
            x.add_(1)
    '''))
    out = tmp_path / "out"
    code = textwrap.dedent('''
        import torch, kernel
        x = torch.ones(4)
        kernel.fill(x, torch.full((4,), float("nan")))    # an out buffer holding NaN
        kernel.fill_ret(x, torch.zeros(4))
        kernel.bump_(torch.ones(4))
    ''')
    env = dict(os.environ, FK_MON_PROFILE="base", FK_MON_OUT=str(out),
               PYTHONPATH=f"{monitor.RUNTIME_DIR / 'hook'}{os.pathsep}{root}")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    [hook] = [json.loads(f.read_text()) for f in out.glob("hook-*.json")]
    fill, fill_ret, bump = hook["calls"]
    assert fill["written"] == [1] and fill["in_sums"] == [4.0, None]
    assert fill["in_nonfinite"] == [0, 4] and fill["inputs_nonfinite"] == 0
    assert [(o["slot"], o["sum"], o["nan"]) for o in fill["outputs"]] == [("arg1", 8.0, 0)]
    # the returned tensor IS the written argument: reported once
    assert [o["slot"] for o in fill_ret["outputs"]] == ["ret0"]
    assert bump["written"] == [0] and bump["outputs"][0]["sum"] == 8.0


@needs_torch
def test_hook_leaves_the_kernels_internal_calls_alone(tmp_path):
    root = tmp_path / "k"
    root.mkdir()
    (root / "kernel.py").write_text(textwrap.dedent('''
        import torch
        CALLS = []
        def helper(x):
            CALLS.append("helper")
            return x * 2
        def call(x):
            return helper(x) + 1
        class Thing:
            pass
    '''))
    out = tmp_path / "out"
    code = textwrap.dedent('''
        import torch, kernel
        from kernel import call
        x = torch.ones(4)
        call(x); kernel.call(x)
        assert kernel.Thing.__name__ == "Thing"
        assert kernel.CALLS == ["helper", "helper"]
        assert getattr(kernel.call, "__fk_monitor__", False)
        assert kernel.call.__wrapped__.__name__ == "call"
    ''')
    env = dict(os.environ, FK_MON_PROFILE="base", FK_MON_OUT=str(out),
               PYTHONPATH=f"{monitor.RUNTIME_DIR / 'hook'}{os.pathsep}{root}")
    p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    [hook] = [json.loads(f.read_text()) for f in out.glob("hook-*.json")]
    assert [c["fn"] for c in hook["calls"]] == ["call", "call"]   # helper never surfaces
    assert hook["calls"][0]["outputs"][0]["sum"] == 12.0


def test_sitecustomize_hands_over_to_the_images_own(tmp_path):
    image = tmp_path / "image"
    image.mkdir()
    marker = tmp_path / "marker"
    (image / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    env = dict(os.environ, FK_MON_PROFILE="base", FK_MON_OUT=str(tmp_path / "out"),
               PYTHONPATH=f"{monitor.RUNTIME_DIR / 'hook'}{os.pathsep}{image}")
    p = subprocess.run([sys.executable, "-c", "import sys; print('fk_monitor_hook' in sys.modules)"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "True"
    assert marker.read_text() == "ran"


def test_without_a_profile_the_hook_does_nothing(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(monitor.RUNTIME_DIR / "hook"))
    env.pop("FK_MON_PROFILE", None)
    p = subprocess.run([sys.executable, "-c",
                        "import sys, fk_monitor_hook as h; print(h._state.get('installed'), "
                        "any(type(f).__name__ == '_KernelFinder' for f in sys.meta_path))"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert p.stdout.strip() == "None False", p.stderr


# --- classification ------------------------------------------------------------------
def _call(fn="call", i=0, *, nan=0, abs_sum=10.0, sq_sum=5.0, in_sums=(1.0,), bad_in=0):
    return {"fn": fn, "i": i, "inputs_nonfinite": bad_in, "in_sums": list(in_sums),
            "outputs": [{"nan": nan, "posinf": 0, "neginf": 0, "abs_sum": abs_sum,
                         "sq_sum": sq_sum}]}


def _rec(profile, card_pass=True, calls=(), **kw):
    return {"profile": profile, "card_pass": card_pass, "rc": 0, "seconds": 1.0,
            "cases": kw.pop("cases", []),
            "hook": {"calls": list(calls), "perturbed": kw.pop("perturbed", 1),
                     "reseeded": kw.pop("reseeded", 0)}, **kw}


def test_classify_nondeterminism_and_verdict_flip():
    res = {"profiles": [_rec("base", calls=[_call()]),
                        _rec("repeat", calls=[_call(abs_sum=10.1)])]}
    v = monitor.classify(res)
    assert v["status"] == "warn"
    assert "race" in v["findings"][0]["what"]
    res["profiles"][1] = _rec("repeat", card_pass=False, calls=[_call()])
    v = monitor.classify(res)
    assert v["status"] == "alarm" and "flips" in v["findings"][0]["what"]
    res["profiles"][1] = _rec("repeat", calls=[_call(abs_sum=10.0000001)])
    v = monitor.classify(res)
    assert v["status"] == "ok" and "bitwise" in v["findings"][0]["what"]


def test_classify_seed_failure_and_crash_and_nonfinite_inputs():
    base = _rec("base", calls=[_call()])
    v = monitor.classify({"profiles": [
        base, _rec("seed1", card_pass=False, calls=[_call(in_sums=(2.0,))],
                   cases=[{"case": "mid", "pass": False}])]})
    assert v["status"] == "alarm"
    assert "does not generalise" in v["findings"][0]["what"] and "mid" in v["findings"][0]["what"]
    v = monitor.classify({"profiles": [base, _rec("amp", card_pass=None, rc=134)]})
    assert v["status"] == "alarm" and "died" in v["findings"][0]["what"]
    v = monitor.classify({"profiles": [base, _rec("amp", card_pass=False,
                                                  calls=[_call(in_sums=(3.0,), bad_in=4)])]})
    assert v["status"] == "ok" and v["not_applicable"] == ["amp"]


def _dps_call(i=0, *, x_sum=1.0, x_bad=0, out_left=0, out_nan=0, written=True):
    """A destination-passing call: argument 0 is read, argument 1 is the out buffer
    (``out_left`` non-finite values left in it from before the call)."""
    return {"fn": "call", "i": i, "n_inputs": 2,
            "in_sums": [x_sum, None if written else 3.0],
            "in_nonfinite": [x_bad, out_left], "written": [1] if written else [],
            "inputs_nonfinite": x_bad + (0 if written else out_left),
            "outputs": [{"slot": "arg1", "nan": out_nan, "posinf": 0, "neginf": 0,
                         "abs_sum": 10.0, "sq_sum": 5.0}] if written else []}


def test_classify_an_out_buffers_leftovers_are_not_non_finite_inputs():
    base = _rec("base", calls=[_dps_call(0), _dps_call(1)])
    # call 1's buffer is the block call 0's NaN result was freed from, and the kernel
    # refilled it with the same NaN — it reads as unwritten, but position 1 is an
    # output of `call` (call 0 wrote it), so the leftovers are no input
    amp = _rec("amp", card_pass=False, ref_pass=True,
               calls=[_dps_call(0, x_sum=4.0, out_nan=8),
                      _dps_call(1, x_sum=4.0, out_left=8, written=False)])
    v = monitor.classify({"profiles": [base, amp]})
    assert v["status"] == "alarm", v
    assert "amp" in v["ran"] and not v["not_applicable"]
    assert any("NaN" in f["what"] for f in v["findings"] if f["severity"] == "alarm")
    # a real input made non-finite by the profile still voids it
    amp = _rec("amp", card_pass=False, calls=[_dps_call(0, x_sum=4.0, x_bad=2, out_nan=8)])
    v = monitor.classify({"profiles": [_rec("base", calls=[_dps_call(0)]), amp]})
    assert v["status"] == "ok" and v["not_applicable"] == ["amp"]


def test_classify_an_in_place_input_change_falls_back_to_the_draw_counters():
    # the kernel updates argument 1 in place and the profile reshaped only that one:
    # the signature (written positions left out) cannot see it, the counters can
    base = _rec("base", calls=[_dps_call(0)], perturbed=0)
    v = monitor.classify({"profiles": [base, _rec("spikes", calls=[_dps_call(0)], perturbed=2)]})
    assert v["ran"] == ["base", "spikes"] and not v["not_applicable"]
    v = monitor.classify({"profiles": [base, _rec("spikes", calls=[_dps_call(0)], perturbed=0)]})
    assert v["not_applicable"] == ["spikes"]


def test_classify_reference_failing_too_is_information_not_an_alarm():
    v = monitor.classify({"profiles": [
        _rec("base", calls=[_call()]),
        _rec("tiny", card_pass=False, ref_pass=False, calls=[_call(in_sums=(0.1,))])]})
    assert v["status"] == "ok"
    assert "so does the reference" in v["findings"][0]["what"]


def test_classify_base_failure_and_missing_result():
    v = monitor.classify({"profiles": [_rec("base", card_pass=False)]})
    assert v["status"] == "alarm"
    assert monitor.classify(None)["status"] == "warn"
    v = monitor.classify({"profiles": [_rec("base", calls=[_call(nan=3)])]})
    assert v["status"] == "warn" and "passes it" in v["findings"][0]["what"]


# --- job mode, overlay, run ----------------------------------------------------------
def test_job_mode_is_quote_free_and_rejects_unknown_profiles():
    mode = monitor.job_mode(("base", "seed1"), "kernel.py", 1200)
    assert mode == ("cmd:python3 .fk_monitor/run.py --profiles base,seed1 "
                    "--kernel kernel.py --budget-s 1200")
    assert not any(c in mode for c in "'\"")
    with pytest.raises(ValueError):
        monitor.job_mode(("base", "nope"))
    with pytest.raises(ValueError):
        monitor.job_mode(("base",), "my kernel.py")


class _Backend:
    def __init__(self, log_text):
        self.log_text = log_text
        self.files, self.pushed, self.modes = {}, [], []
        self.last_job_id = "4242"

    def with_file(self, ref, path, text, message):
        self.files[path] = text
        return f"{ref}+{Path(path).name}"

    def push_ref(self, ref, branch):
        self.pushed.append((ref, branch))

    def submit_gpu(self, branch, mode):
        self.modes.append(mode)
        return self.log_text


def test_run_overlays_submits_and_appends_the_ledger(tmp_path):
    result = {"version": 1, "profiles": [_rec("base", calls=[_call()])], "skipped": []}
    be = _Backend("noise\n" + monitor.MARKER + "\n" + json.dumps(result) + "\n")
    lines = []
    rec = monitor.run(be, "abc123", branch="op-monitor", op_dir=tmp_path,
                      profiles=("base",), log=lines.append)
    assert set(be.files) == set(monitor.runtime_files())
    assert be.pushed == [(rec["tree"], "op-monitor")] and rec["tree"].startswith("abc123+")
    assert be.modes == [monitor.job_mode(("base",))]
    assert rec["status"] == "ok" and rec["job"] == "4242"
    ledger = [json.loads(x) for x in (tmp_path / monitor.LEDGER).read_text().splitlines()]
    assert ledger[0]["sha"] == "abc123" and ledger[0]["status"] == "ok"
    assert lines and lines[-1].startswith("correctness monitor: OK")


def test_run_without_a_result_is_a_warning(tmp_path):
    rec = monitor.run(_Backend("job died\n"), "abc", branch="b", op_dir=tmp_path,
                      profiles=("base",), log=lambda s: None)
    assert rec["status"] == "warn" and rec["result"] is None


def test_a_cut_short_job_is_read_from_its_profile_blocks():
    blocks = [_rec("base", calls=[_call()]),
              _rec("seed1", card_pass=False, calls=[_call(in_sums=(2.0,))])]
    text = "".join(f"{monitor.PROFILE_MARKER}\n{json.dumps(b)}\nnoise\n" for b in blocks)
    v = monitor.classify(monitor.parse_log(text))
    assert v["status"] == "alarm"
    assert any("cut short after 2" in f["what"] for f in v["findings"])
    assert monitor.parse_log("nothing here") is None


# --- the loop's finish -----------------------------------------------------------------
def test_finish_monitor_records_next_to_the_confirmation(tmp_path):
    from types import SimpleNamespace

    from forgekernel.loop.cli import _run_finish_monitor
    result = {"profiles": [_rec("base", card_pass=False)], "complete": True}
    be = _Backend(monitor.MARKER + "\n" + json.dumps(result) + "\n")
    ctx = SimpleNamespace(branch_prefix="op-run", status_path=tmp_path / "status.jsonl")
    confirm = {"sha": "feedbeef", "median_ms": 1.0, "jobs": []}
    summary: dict = {}
    rec = _run_finish_monitor(ctx, be, tmp_path, confirm, summary, ("base",), "kernel.py")
    assert rec["status"] == "alarm" and summary["correctness_monitor"] == "alarm"
    assert be.pushed[0][1] == "op-run-monitor"
    written = json.loads((tmp_path / "best_confirmation.json").read_text())
    assert written["correctness_monitor"]["sha"] == "feedbeef"
    assert written["correctness_monitor"]["alarms"]


def test_finish_monitor_never_raises(tmp_path):
    from types import SimpleNamespace

    from forgekernel.loop.cli import _run_finish_monitor

    class Broken(_Backend):
        def submit_gpu(self, branch, mode):
            raise RuntimeError("cluster down")
    ctx = SimpleNamespace(branch_prefix="p", status_path=tmp_path / "status.jsonl")
    summary: dict = {}
    assert _run_finish_monitor(ctx, Broken(""), tmp_path, {"sha": "x", "jobs": []},
                               summary, ("base",), "kernel.py") is None
    assert "correctness_monitor" not in summary
