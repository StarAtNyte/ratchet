"""Job-environment blindness (observed on a fused-norm-rope B200 run).

That run's student card hard-imported ``cuda.cudart`` in eval/ncu_run.py (the image
ships cuda-python 13: ``cuda.bindings`` only) and its gpu_entry.sh left the image's
venv off PATH on an ssh host, so every harness channel that re-spawns a bare
``python3`` died on ``import torch``. For 2.5 days: 97/97 NCU collections, 33/33
timelines, 7/7 stall ledgers dead, logged as "ncu empty -> retry",
"base-bench-failed-or-incorrect", "job returned no ledger". The teacher found and
fixed it three times; each fix was discarded by a reset or never reached a job.

These tests pin the four answers: failures on the job's environment are named as
such and not retried (loop/job_env.py + the driver channels); a launch preflight
refuses to run blind; the card's job-environment files reach every job through
[loop] job_env_files without entering the student lineage; and a reset that throws
away edits outside notes/ says so.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import (  # noqa: E402
    _DIAG, _STALL_LOG_OK, FakeBackend, _ctx, _lever_ctx, _lever_design, _patching,
    _stall_ctx,
)

from forgekernel.loop import cli  # noqa: E402
from forgekernel.loop.driver import (  # noqa: E402
    _JOB_ENV_CHECK_CMD, _LEVER_PRICING_CMD, _ncu_profiler, _run_timeline,
    _timeline_feedback, job_env_preflight, run_meta_step,
)
from forgekernel.loop.job_env import env_failure  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The job logs, as that run's jobs left them.
CUDART_LOG = (
    "[task] NCU key-metrics profile\n"
    "Traceback (most recent call last):\n"
    '  File "/workspace/repo/eval/ncu_run.py", line 7, in <module>\n'
    "    import cuda.cudart as cudart  # type: ignore\n"
    "ModuleNotFoundError: No module named 'cuda.cudart'\n"
    "==ERROR== The application returned an error code (1).\n"
    "[gpu_entry.sh] NCU exited rc=1 — continuing without a profile\n"
    "ORCH-CENSUS-SKIP reason=no-cubin-dumped\n")
TORCH_TRACE = (
    "$ python3 eval/bench.py --mode all\n"
    "[bench:base] rc=1 no JSON; stderr tail:\n"
    "Traceback (most recent call last):\n"
    '  File "/workspace/repo/eval/bench.py", line 25, in <module>\n'
    "    import torch\n"
    "ModuleNotFoundError: No module named 'torch'\n")
TIMELINE_ENV_LOG = ("TIMELINE-SITES 2 0:a.entry 1:a.end\n" + TORCH_TRACE
                    + "\nTIMELINE-FAIL reason=base-bench-failed-or-incorrect\n")
PRICE_ENV_LOG = (TORCH_TRACE + "PRICE ROUND 0 base pass=False\n"
                 "PRICE-FAIL reason=base-failed-correctness\n")


# --- the detector -----------------------------------------------------------------

def test_env_failure_names_the_missing_binding_and_the_missing_torch():
    assert env_failure(CUDART_LOG) == "ModuleNotFoundError: No module named 'cuda.cudart'"
    assert env_failure(TORCH_TRACE) == "ModuleNotFoundError: No module named 'torch'"


def test_env_failure_leaves_the_trees_own_failures_alone():
    assert env_failure(None) is None and env_failure("") is None
    # the student's kernel module missing is the tree, not the image
    assert env_failure("ModuleNotFoundError: No module named 'kernel'\n") is None
    # a scaffold kernel that raises, a correctness miss: not the environment
    assert env_failure("NotImplementedError: M0 not built yet\nPRICE-FAIL reason=base-failed-correctness\n") is None
    assert env_failure("TIMELINE-FAIL reason=base-bench-failed-or-incorrect\n") is None


def test_env_failure_reads_a_missing_shared_object_and_a_missing_tool():
    assert "libcuda.so.1" in env_failure(
        "ImportError: libcuda.so.1: cannot open shared object file: No such file or directory\n")
    assert env_failure("bash: line 1: ncu: command not found\n").endswith("ncu: command not found")


# --- the seeded check script ------------------------------------------------------

def _fake_python3(tmp_path, body: str) -> dict:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "python3"
    exe.write_text("#!/bin/sh\n" + body)
    exe.chmod(0o755)
    return dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")


def _run_check(env) -> str:
    script = REPO / "skills" / "job-env-check" / "job_env_check.py"
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          env=env, timeout=60).stdout


def test_job_env_check_reports_a_python3_without_torch(tmp_path):
    env = _fake_python3(tmp_path, "echo \"ModuleNotFoundError: No module named 'torch'\" >&2\nexit 1\n")
    out = _run_check(env)
    assert "JOB-ENV-FAIL python3=" in out and "No module named 'torch'" in out
    assert env_failure(out) == "ModuleNotFoundError: No module named 'torch'"


def test_job_env_check_passes_a_python3_with_torch(tmp_path):
    env = _fake_python3(tmp_path, "echo 2.13.0+cu130\n")
    out = _run_check(env)
    assert out.strip().startswith("JOB-ENV-OK python3=") and "torch=2.13.0+cu130" in out
    assert env_failure(out) is None


# --- the NCU channel --------------------------------------------------------------

def test_ncu_env_failure_is_named_and_not_retried(capsys):
    """A launch target that cannot import its profiler binding replays the same
    error on every retry: one job, an ENV-BROKEN line, the count kept."""
    be = FakeBackend([], [CUDART_LOG, CUDART_LOG, CUDART_LOG])
    ctx = SimpleNamespace(backend=be, ncu_retries=2, orch_census=False, workspace="/nonexistent")
    assert _ncu_profiler(ctx, "b", correctness=True) == {}
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", "ncu")]
    out = capsys.readouterr().out
    assert "ENV-BROKEN ncu (x1 in a row): ModuleNotFoundError: No module named 'cuda.cudart'" in out
    assert "ncu empty on a passing kernel -> retry" not in out
    _ncu_profiler(ctx, "b", correctness=True)
    assert ctx.env_broken == {"ncu": 2}
    assert "ENV-BROKEN ncu (x2 in a row)" in capsys.readouterr().out


def test_an_empty_ncu_without_an_env_signature_is_still_retried(capsys):
    """The GPU-side best-effort failure the retry exists for is untouched."""
    be = FakeBackend([], ["(no csv)", "(no csv)", "(no csv)"])
    ctx = SimpleNamespace(backend=be, ncu_retries=2, orch_census=False, workspace="/nonexistent")
    _ncu_profiler(ctx, "b", correctness=True)
    assert len([c for c in be.calls if c[0] == "gpu"]) == 3
    assert "ENV-BROKEN" not in capsys.readouterr().out


def test_a_failing_kernel_is_not_blamed_on_the_environment(capsys):
    be = FakeBackend([], [CUDART_LOG])
    ctx = SimpleNamespace(backend=be, ncu_retries=2, orch_census=False, workspace="/nonexistent")
    _ncu_profiler(ctx, "b", correctness=False)
    assert "ENV-BROKEN" not in capsys.readouterr().out


# --- the timeline channel ---------------------------------------------------------

def test_timeline_env_failure_is_tagged_and_the_feedback_says_not_the_markers(tmp_path, capsys):
    be = FakeBackend([], [TIMELINE_ENV_LOG])
    ctx = _ctx(tmp_path, be)
    tl = _run_timeline(ctx, "sha0", "iter13:gate row")
    assert tl is not None and tl.get("env") == "ModuleNotFoundError: No module named 'torch'"
    out = capsys.readouterr().out
    assert ("FAIL — base-bench-failed-or-incorrect [job environment: ModuleNotFoundError: "
            "No module named 'torch']") in out
    assert "ENV-BROKEN timeline (x1 in a row)" in out
    fb = _timeline_feedback(tl)
    assert "job environment" in fb and "Do not send" in fb and "Add the missing markers" not in fb
    # a failure that is the tree's keeps its old feedback
    plain = dict(tl)
    plain.pop("env")
    assert "Add the missing markers" in _timeline_feedback(plain)


# --- the stall-price channel ------------------------------------------------------

def test_stall_price_without_a_ledger_names_the_environment(tmp_path, capsys):
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")],
                     [TORCH_TRACE])
    ctx = _stall_ctx(tmp_path, be)
    run_meta_step(ctx)
    out = capsys.readouterr().out
    assert "job returned no ledger (job environment: ModuleNotFoundError: No module named 'torch')" in out
    assert "ENV-BROKEN stall-price (x1 in a row)" in out


# --- the lever-price channel ------------------------------------------------------

def test_lever_price_env_failure_is_not_redesigned_and_the_teacher_is_told(tmp_path, capsys):
    """The tool says base-failed-correctness; the base is the best tree, which passed
    its own eval. No design retry (levers.json cannot fix an interpreter), and the
    diagnosis is told the job environment is the reason there is no table."""
    be = FakeBackend([_lever_design(tmp_path, 1), _DIAG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [_STALL_LOG_OK, PRICE_ENV_LOG])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert be.calls.count(("gpu", _LEVER_PRICING_CMD)) == 1
    diag = be.prompts[1]
    assert "The pricing job rejected the lever table" not in diag
    assert "job environment" in diag and "No module named 'torch'" in diag and "None of the levers designed this time was priced" in diag
    out = capsys.readouterr().out
    assert "ENV-BROKEN lever-price (x1 in a row)" in out and "no design retry" in out


# --- discarded edits --------------------------------------------------------------

class _TracksEdits(FakeBackend):
    """A backend that can say what a reset would throw away: the design turn edited
    two job-environment files besides its levers.json."""
    def __init__(self, *a, edits=(), **k):
        super().__init__(*a, **k)
        self.edits = list(edits)
        self.asked = []

    def changed_paths(self, base):
        self.asked.append(base)
        # nothing uncommitted before the round; the pricing commits carry the edits
        return list(self.edits) if len(self.asked) > 1 else []


def test_a_pricing_reset_that_drops_edits_outside_notes_says_so(tmp_path, capsys):
    be = _TracksEdits([_lever_design(tmp_path, 1), _DIAG,
                       _patching(tmp_path, "\n### D6 added\nx\n",
                                 "patch_class: additive\nchanged_entries: []\n")],
                      [_STALL_LOG_OK, PRICE_ENV_LOG],
                      edits=["notes/levers.json", "gpu_entry.sh", "eval/ncu_run.py"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    out = capsys.readouterr().out
    assert ("meta/lever-price: discarding edits outside notes/ with the reset: "
            "eval/ncu_run.py, gpu_entry.sh") in out
    assert "notes/levers.json" not in out.split("discarding edits outside notes/")[1].splitlines()[0]
    diag = be.prompts[1]
    assert "## Edits discarded by a reset" in diag and "gpu_entry.sh" in diag and "job_env_files" in diag
    assert ctx.discarded_edits == ""          # said once


def test_a_pricing_round_with_nothing_outside_notes_is_silent(tmp_path, capsys):
    be = _TracksEdits([_lever_design(tmp_path, 1), _DIAG,
                       _patching(tmp_path, "\n### D6 added\nx\n",
                                 "patch_class: additive\nchanged_entries: []\n")],
                      [_STALL_LOG_OK, PRICE_ENV_LOG], edits=["notes/levers.json"])
    ctx = _lever_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "discarding edits" not in capsys.readouterr().out
    assert "## Edits discarded by a reset" not in be.prompts[1]


# --- the launch preflight ---------------------------------------------------------

def _preflight_ctx(tmp_path, logs, *, seeded=True):
    be = FakeBackend([], list(logs))
    ctx = _ctx(tmp_path, be)
    if seeded:
        (tmp_path / "notes").mkdir(exist_ok=True)
        (tmp_path / "notes" / "job_env_check.py").write_text("# seeded\n")
    return be, ctx


def test_preflight_refuses_a_blind_run(tmp_path, capsys):
    """That run's first two jobs, had they been asked: both channels dead."""
    be, ctx = _preflight_ctx(tmp_path, [
        "Traceback...\nModuleNotFoundError: No module named 'torch'\n"
        "JOB-ENV-FAIL python3=/usr/bin/python3 ModuleNotFoundError: No module named 'torch'\n",
        CUDART_LOG])
    why = job_env_preflight(ctx, "refuse")
    assert why == ("tools: ModuleNotFoundError: No module named 'torch'; "
                   "ncu: ModuleNotFoundError: No module named 'cuda.cudart'")
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", _JOB_ENV_CHECK_CMD), ("gpu", "ncu")]
    assert ("push", "fs-demo-jobenv") in be.calls
    assert "job-env preflight: FAIL" in capsys.readouterr().out


def test_preflight_passes_a_scaffold_kernel_that_raises(tmp_path, capsys):
    """At a fresh start the kernel is a stub: its NCU job fails on the tree, which is
    the tree's business — only an environment failure refuses the launch."""
    be, ctx = _preflight_ctx(tmp_path, [
        "JOB-ENV-OK python3=/opt/sglang/bin/python3 torch=2.13.0+cu130\n",
        "Traceback...\nNotImplementedError: M0 not built yet\n==ERROR== The application returned an error code (1).\n"])
    assert job_env_preflight(ctx, "refuse") is None
    out = capsys.readouterr().out
    assert "job-env preflight: OK (JOB-ENV-OK python3=/opt/sglang/bin/python3 torch=2.13.0+cu130)" in out


def test_preflight_off_costs_nothing_and_unseeded_trees_check_ncu_only(tmp_path):
    be, ctx = _preflight_ctx(tmp_path, [])
    assert job_env_preflight(ctx, "off") is None and not [c for c in be.calls if c[0] == "gpu"]
    (tmp_path / "x").mkdir()
    be2, ctx2 = _preflight_ctx(tmp_path / "x", ["(no csv)"], seeded=False)
    assert job_env_preflight(ctx2, "refuse") is None
    assert [c for c in be2.calls if c[0] == "gpu"] == [("gpu", "ncu")]


def test_preflight_that_cannot_submit_does_not_judge(tmp_path):
    from forgekernel.loop.driver import BackendError

    class _Down(FakeBackend):
        def submit_gpu(self, branch, mode):
            raise BackendError("queue died")
    be = _Down([], [])
    ctx = _ctx(tmp_path, be)
    assert job_env_preflight(ctx, "refuse") is None

    # nor does one whose toolkit seeding cannot commit: the CLI calls this outside
    # the meta layer's BackendError net, so it must not raise
    class _NoCommit(FakeBackend):
        def commit(self, message):
            raise BackendError("index.lock")
    be2 = _NoCommit([], [])
    (tmp_path / "y").mkdir()
    ctx2 = _ctx(tmp_path / "y", be2)
    ctx2.stall_pricing = True
    assert job_env_preflight(ctx2, "refuse") is None
    assert not [c for c in be2.calls if c[0] == "gpu"]


# --- [loop] job_env_files: the card's copies reach every job ----------------------

def _git(ws, *args):
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _card_and_ws(tmp_path):
    """The card (operators/<op>, a git repo) and the workspace cloned from it, as a
    launch sets them up; the card's working copy of the job environment is then
    fixed the way that run's teacher fixed it — and never committed."""
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    _git(card, "init", "-q", "-b", "main")
    _git(card, "config", "user.name", "t")
    _git(card, "config", "user.email", "t@x")
    (card / "gpu_entry.sh").write_text("PY=/opt/sglang/bin/python\n")
    (card / "eval" / "ncu_run.py").write_text("import cuda.cudart as cudart\n")
    (card / "kernel.py").write_text("stub\n")
    _git(card, "add", "-A")
    _git(card, "commit", "-q", "-m", "card")
    ws = tmp_path / "ws"
    subprocess.run(["git", "clone", "-q", str(card), str(ws)], check=True, capture_output=True)
    _git(ws, "config", "user.name", "t")
    _git(ws, "config", "user.email", "t@x")
    (ws / "kernel.py").write_text("student work\n")
    _git(ws, "commit", "-q", "-am", "D0.1")
    return card, ws


def test_job_env_files_overlay_the_card_on_every_job_without_touching_the_lineage(tmp_path, capsys):
    card, ws = _card_and_ws(tmp_path)
    (card / "eval" / "ncu_run.py").write_text("import torch\ntorch.cuda.profiler.start()\n")
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0,
                         git_retries=1, git_retry_sleep=0.0)
    be.job_env_files = cli.resolve_job_env_files(
        {"loop": {"job_env_files": ["gpu_entry.sh", "eval/ncu_run.py"]}}, card,
        kernel_file="kernel.py")
    head = _git(ws, "rev-parse", "HEAD")
    be.push_branch("fs-demo-i7")
    pushed = _git(card, "rev-parse", "refs/heads/fs-demo-i7")
    # the job runs the card's fixed launch target on the student's kernel
    assert _git(card, "show", f"{pushed}:eval/ncu_run.py").startswith("import torch")
    assert _git(card, "show", f"{pushed}:kernel.py") == "student work"
    assert _git(card, "rev-parse", f"{pushed}^") == head
    assert _git(card, "diff", "--name-only", head, pushed) == "eval/ncu_run.py"
    # the lineage, the index and the working tree are untouched
    assert _git(ws, "rev-parse", "HEAD") == head and _git(ws, "status", "--porcelain") == ""
    out = capsys.readouterr().out
    assert "jobs now run the card's eval/ncu_run.py" in out
    # push_ref (stall pricing on the best sha, the timeline overlay) gets it too
    be.push_ref(head, "fs-demo-stallprice")
    assert _git(card, "show", "refs/heads/fs-demo-stallprice:eval/ncu_run.py").startswith("import torch")


def test_job_env_files_are_a_no_op_while_the_tree_matches_the_card(tmp_path, capsys):
    card, ws = _card_and_ws(tmp_path)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0,
                         git_retries=1, git_retry_sleep=0.0)
    be.job_env_files = cli.resolve_job_env_files(
        {"loop": {"job_env_files": ["gpu_entry.sh"]}}, card, kernel_file="kernel.py")
    be.push_branch("fs-demo-i1")
    assert _git(card, "rev-parse", "refs/heads/fs-demo-i1") == _git(ws, "rev-parse", "HEAD")
    assert "no overlay" in capsys.readouterr().out
    # and off entirely by default: push_branch pushes HEAD exactly as before
    be.job_env_files = {}
    be.push_branch("fs-demo-i2")
    assert _git(card, "rev-parse", "refs/heads/fs-demo-i2") == _git(ws, "rev-parse", "HEAD")


def test_changed_paths_is_what_a_hard_reset_throws_away(tmp_path):
    card, ws = _card_and_ws(tmp_path)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    base = _git(ws, "rev-parse", "HEAD")
    (ws / "notes").mkdir()
    (ws / "notes" / "levers.json").write_text("{}")
    (ws / "gpu_entry.sh").write_text("fixed\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "meta-price")
    (ws / "eval" / "ncu_run.py").write_text("uncommitted fix\n")
    (ws / "scratch.txt").write_text("untracked survives a reset\n")
    assert sorted(be.changed_paths(base)) == ["eval/ncu_run.py", "gpu_entry.sh", "notes/levers.json"]
    assert be.changed_paths("not-a-ref") == []


def test_job_env_files_config_is_validated(tmp_path):
    card = tmp_path / "card"
    (card / "eval").mkdir(parents=True)
    (card / "gpu_entry.sh").write_text("x\n")
    (card / "eval" / "ncu_run.py").write_text("x\n")
    ok = cli.resolve_job_env_files({"loop": {"job_env_files": ["gpu_entry.sh", "eval/ncu_run.py"]}},
                                   card, kernel_file="kernel.py")
    assert ok == {"gpu_entry.sh": card / "gpu_entry.sh", "eval/ncu_run.py": card / "eval" / "ncu_run.py"}
    assert cli.resolve_job_env_files({}, card) == {}
    for bad in (["missing.sh"], ["../escape.sh"], ["/abs/gpu_entry.sh"], ["kernel.py"],
                ["notes/levers.json"], "notes/x.py", [3]):
        if bad == ["kernel.py"]:
            (card / "kernel.py").write_text("stub\n")
        with pytest.raises(ValueError):
            cli.resolve_job_env_files({"loop": {"job_env_files": bad}}, card, kernel_file="kernel.py")


def test_the_check_script_rides_the_pricing_toolkit():
    from forgekernel.loop.driver import PRICING_SEEDED_MODULES
    assert "forgekernel.skills.job_env_check" in PRICING_SEEDED_MODULES
    assert _JOB_ENV_CHECK_CMD == "cmd:python3 notes/job_env_check.py"
    assert json.dumps(_JOB_ENV_CHECK_CMD).count(" ") == 1        # quote-free, as cmd: requires
