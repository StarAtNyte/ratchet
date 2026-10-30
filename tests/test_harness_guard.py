"""The evaluation harness is the card's (loop/harness_guard.py): a job never runs a
student-edited gpu_entry.sh / eval/ / ref_bin/, and a delivery that edits them is
reverted before any job is bought."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx  # noqa: E402

from forgekernel.inner.trajectory import load_records  # noqa: E402
from forgekernel.loop import cli, harness_guard  # noqa: E402
from forgekernel.loop.driver import run_inner_iteration  # noqa: E402


# --- what counts as harness ----------------------------------------------------------

@pytest.mark.parametrize("path, want", [
    ("eval/bench.py", True), ("eval/leaderboard/problem.json", True), ("gpu_entry.sh", True),
    ("ref_bin/binref_x.so", True), (".gitattributes", True), ("./eval/timing.py", True),
    ("kernel.py", False), ("notes/probe.py", False), ("evaluation.py", False),
    ("gpu_entry.sh.bak", False), ("window.toml", False), ("README.md", False),
])
def test_harness_paths(path, want):
    assert harness_guard.is_harness_path(path, "kernel.py") is want


def test_the_kernel_file_is_never_harness_even_under_eval():
    assert not harness_guard.is_harness_path("eval/kernel.py", "eval/kernel.py")
    assert harness_guard.harness_only(["eval/kernel.py", "eval/bench.py", "notes/a.py"],
                                      "eval/kernel.py") == ["eval/bench.py"]


@pytest.mark.parametrize("raw, mode", [(None, "revert"), ("warn", "warn"), ("OFF", "off"),
                                       (True, "revert"), (False, "off")])
def test_guard_mode(raw, mode):
    cfg = {"loop": {} if raw is None else {"harness_guard": raw}}
    assert harness_guard.resolve_guard_mode(cfg) == mode


def test_a_misspelt_guard_mode_is_an_error_not_off():
    with pytest.raises(ValueError, match="harness_guard"):
        harness_guard.resolve_guard_mode({"loop": {"harness_guard": "revret"}})


# --- a real card and its workspace ---------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _card(tmp_path):
    card = tmp_path / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (card / "ref_bin").mkdir()
    (card / "kernel.py").write_text("def run(): pass\n")
    (card / "gpu_entry.sh").write_text("#!/bin/bash\npython3 eval/bench.py\n")
    (card / "eval" / "bench.py").write_text("print('judge')\n")
    (card / "eval" / "leaderboard" / "blob.bin").write_bytes(bytes(range(256)) * 4)
    (card / "ref_bin" / "ref.so").write_bytes(b"\x7fELF\x00\x01binary")
    (card / ".gitignore").write_text("ref_bin/*.key\n")
    (card / "ref_bin" / "ref_payload.key").write_text("SECRET")      # gitignored
    _git(tmp_path, "init", "-q", str(card))
    _commit(card, "card")
    ws = tmp_path / "ws"
    subprocess.run(["git", "clone", "-q", str(card), str(ws)], check=True)
    return card, ws


def _backend(ws, card=None):
    be = cli.RealBackend("demo", ws, None, "t", "t@t")
    be.kernel_file = "kernel.py"
    be.harness_card = card
    return be


def _tree(ws, ref):
    return set(_git(ws, "ls-tree", "-r", "--name-only", ref).splitlines())


def _show(ws, ref, path):
    return subprocess.run(["git", "-C", str(ws), "show", f"{ref}:{path}"],
                          check=True, capture_output=True).stdout


def test_an_untouched_tree_is_pushed_as_is(tmp_path):
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    head = be.head()
    assert be._with_job_env(head) == head


def test_the_job_tree_carries_the_cards_harness_whatever_the_lineage_holds(tmp_path):
    card, ws = _card(tmp_path)
    (ws / "eval" / "bench.py").write_text("print('PASS always')\n")
    (ws / "eval" / "sitecustomize.py").write_text("import os\n")       # added by the student
    (ws / "eval" / "leaderboard" / "blob.bin").write_bytes(b"zero")
    (ws / "gpu_entry.sh").write_text("echo hacked\n")
    (ws / "kernel.py").write_text("def run(): return 1\n")
    head = _commit(ws, "student")
    be = _backend(ws, card)
    job = be._with_job_env(head)
    assert job != head and _git(ws, "rev-parse", "HEAD") == head      # lineage untouched
    assert _show(ws, job, "eval/bench.py") == b"print('judge')\n"
    assert _show(ws, job, "eval/leaderboard/blob.bin") == bytes(range(256)) * 4
    assert _show(ws, job, "gpu_entry.sh").startswith(b"#!/bin/bash")
    assert _show(ws, job, "kernel.py") == b"def run(): return 1\n"    # the student's work
    tree = _tree(ws, job)
    assert "eval/sitecustomize.py" not in tree
    assert "ref_bin/ref_payload.key" not in tree                       # never the key
    assert _git(ws, "ls-tree", job, "gpu_entry.sh").split()[0] == "100755"


def test_without_the_guard_only_job_env_files_are_overlaid(tmp_path):
    card, ws = _card(tmp_path)
    (ws / "eval" / "bench.py").write_text("print('PASS always')\n")
    (ws / "gpu_entry.sh").write_text("echo old\n")
    head = _commit(ws, "student")
    be = _backend(ws, None)
    be.job_env_files = cli.resolve_job_env_files({"loop": {"job_env_files": ["gpu_entry.sh"]}},
                                                 card, kernel_file="kernel.py")
    job = be._with_job_env(head)
    assert _show(ws, job, "gpu_entry.sh").startswith(b"#!/bin/bash")
    assert _show(ws, job, "eval/bench.py") == b"print('PASS always')\n"


def test_harness_changes_drift_and_restore(tmp_path):
    card, ws = _card(tmp_path)
    be = _backend(ws, card)
    anchor = be.head()
    (ws / "eval" / "bench.py").write_text("tampered\n")
    (ws / "kernel.py").write_text("def run(): return 2\n")
    probe = _commit(ws, "probe")
    assert be.harness_changes(anchor, probe) == ["eval/bench.py"]
    (ws / "eval" / "extra.py").write_text("x\n")                         # untracked
    (ws / "notes").mkdir()
    (ws / "notes" / "kept.py").write_text("kept probe\n")                # untracked, not harness
    assert be.harness_drift(anchor) == ["eval/bench.py", "eval/extra.py"]
    new = be.restore_harness(anchor, ["eval/bench.py", "eval/extra.py"], "restore")
    assert new != probe and be.harness_drift(anchor) == []
    assert (ws / "eval" / "bench.py").read_text() == "print('judge')\n"
    assert not (ws / "eval" / "extra.py").exists()
    assert (ws / "kernel.py").read_text() == "def run(): return 2\n"     # kernel kept
    assert (ws / "notes" / "kept.py").exists()                           # not swept in
    assert "notes/kept.py" not in _tree(ws, new)


def test_preflight_fresh_start_over_a_drifted_workspace_is_refused(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "eval" / "bench.py").write_text("tampered\n")
    _commit(ws, "drift")
    be = _backend(ws)
    rc = cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False, dry_run=False)
    assert rc == cli.EXIT_HARNESS_DRIFT == 19
    assert "~eval/bench.py" in capsys.readouterr().err


def test_preflight_resume_over_a_drifted_workspace_warns_and_overlays(tmp_path, capsys):
    card, ws = _card(tmp_path)
    (ws / "eval" / "bench.py").write_text("tampered\n")
    _commit(ws, "drift")
    be = _backend(ws)
    assert cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=True,
                                        dry_run=False) == 0
    assert be.harness_card == card
    assert "WARNING: resuming" in capsys.readouterr().err


def test_preflight_matching_workspace_goes_on(tmp_path, capsys):
    card, ws = _card(tmp_path)
    be = _backend(ws)
    assert cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=False,
                                        dry_run=False) == 0
    assert be.harness_card == card
    assert "identical to the workspace" in capsys.readouterr().out


def test_preflight_card_that_is_not_a_git_repo_warns(tmp_path, capsys):
    card = tmp_path / "plain"
    card.mkdir()
    be = _backend(tmp_path)
    assert cli._harness_guard_preflight(card, tmp_path, "kernel.py", be, resuming=False,
                                        dry_run=False) == 0
    assert be.harness_card is None and "not a git repository" in capsys.readouterr().err


def test_a_card_inside_another_repository_is_not_its_own_git_repo(tmp_path, capsys):
    """operators/<op> under ForgeKernel without its own `git init`: `rev-parse` succeeds
    from the parent, but the card tracks nothing there -- read as an empty harness, every
    job tree would lose gpu_entry.sh and eval/."""
    parent = tmp_path / "fk"
    card = parent / "operators" / "op"
    (card / "eval").mkdir(parents=True)
    (card / "gpu_entry.sh").write_text("#!/bin/bash\n")
    (card / "eval" / "bench.py").write_text("print('judge')\n")
    (parent / ".gitignore").write_text("/operators/*\n")
    _git(tmp_path, "init", "-q", str(parent))
    _commit(parent, "fk")
    assert not harness_guard.is_git_repo(card) and harness_guard.is_git_repo(parent)
    assert harness_guard.card_harness_files(card) is None
    _, ws = _card(tmp_path)
    be = _backend(ws)
    assert cli._harness_guard_preflight(card, ws, "kernel.py", be, resuming=True,
                                        dry_run=False) == 0
    assert be.harness_card is None and "not a git repository" in capsys.readouterr().err


def test_a_card_that_tracks_no_harness_overlays_nothing(tmp_path):
    card, ws = _card(tmp_path)
    _git(card, "rm", "-q", "-r", "--cached", "gpu_entry.sh", "eval", "ref_bin")
    _git(card, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "untrack")
    assert harness_guard.card_harness_files(card, "kernel.py") is None
    be = _backend(ws, card)
    head = be.head()
    sha = be._with_job_env(head)
    assert {"gpu_entry.sh", "eval/bench.py"} <= _tree(ws, sha)


def test_exit_19_names_nothing_else():
    taken = {v for k, v in vars(cli).items() if k.startswith("EXIT_") and k != "EXIT_HARNESS_DRIFT"}
    assert cli.EXIT_HARNESS_DRIFT not in taken


# --- the driver ----------------------------------------------------------------------

class GuardBackend(FakeBackend):
    def __init__(self, agent_outputs, gpu_logs, touched=(), drift=()):
        super().__init__(agent_outputs, gpu_logs)
        self.touched = list(touched)
        self.drift = list(drift)

    def harness_changes(self, base, head):
        self.calls.append(("harness-changes", base, head))
        return list(self.touched)

    def harness_drift(self, anchor):
        return list(self.drift)

    def restore_harness(self, anchor, paths, message):
        self.calls.append(("restore", anchor, tuple(paths)))
        self.drift = []
        self._sha += 1
        return f"sha{self._sha}"


def test_a_delivery_that_edits_the_harness_is_reverted_without_a_job(tmp_path):
    be = GuardBackend([REC("D1.1", 0)], [EVAL(1.0), NCU], touched=["eval/bench.py"])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "REVERT" and rec.correctness is None
    assert "harness guard" in rec.error and "eval/bench.py" in rec.error
    assert not [c for c in be.calls if c[0] in ("gpu", "push")]
    assert ("reset", "sha0") in be.calls                     # back to the iteration's base
    assert load_records(ctx.status_path)[-1].error == rec.error


def test_warn_mode_keeps_the_delivery(tmp_path, capsys):
    be = GuardBackend([REC("D1.1", 0)], [EVAL(2.4), NCU], touched=["gpu_entry.sh"])
    ctx = _ctx(tmp_path, be)
    ctx.harness_guard = "warn"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and ("gpu", "all") in be.calls
    assert "harness_guard = warn" in capsys.readouterr().out


def test_a_clean_delivery_is_evaluated(tmp_path):
    be = GuardBackend([REC("D1.1", 0)], [EVAL(2.4), NCU], touched=[])
    ctx = _ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and ("harness-changes", "sha0", "sha1") in be.calls


def test_a_drifted_lineage_is_restored_before_the_student_turn(tmp_path, capsys):
    be = GuardBackend([REC("D1.1", 0)], [EVAL(2.4), NCU], drift=["eval/extra.py"])
    ctx = _ctx(tmp_path, be)
    ctx.harness_anchor = "sha0"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert be.calls[0] == ("restore", "sha0", ("eval/extra.py",))
    assert rec.base_sha == "sha1" and rec.verdict == "KEEP"   # built on the restored tree
    assert "restored eval/extra.py to the launch tree" in capsys.readouterr().out


def test_guard_off_does_nothing(tmp_path):
    be = GuardBackend([REC("D1.1", 0)], [EVAL(2.4), NCU], touched=["eval/bench.py"],
                      drift=["eval/bench.py"])
    ctx = _ctx(tmp_path, be)
    ctx.harness_guard = "off"
    ctx.harness_anchor = "sha0"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP"
    assert not [c for c in be.calls if c[0] in ("restore", "harness-changes")]
