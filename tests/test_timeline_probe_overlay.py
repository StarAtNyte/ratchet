"""RealBackend.with_file: one file overlaid on a ref as a child commit, built
from a temporary index — the timeline probe runs the harness copy of
notes/timeline_probe.py against an older tree (the best sha) without the
tree under judgment, the working tree, the index or HEAD changing."""
import subprocess

import pytest

from forgekernel.loop import cli
from forgekernel.loop.driver import BackendError


def _git(ws, *args, inp=None):
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True,
                          text=True, input=inp).stdout.strip()


def _repo(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes").mkdir()
    _git(ws, "init", "-q", "-b", "main")
    _git(ws, "config", "user.name", "t")
    _git(ws, "config", "user.email", "t@x")
    (ws / "notes" / "timeline_probe.py").write_text("old probe\n", encoding="utf-8")
    (ws / "kernel.py").write_text("k1\n", encoding="utf-8")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "best")
    best = _git(ws, "rev-parse", "HEAD")
    (ws / "kernel.py").write_text("k2\n", encoding="utf-8")
    _git(ws, "commit", "-q", "-am", "later")
    head = _git(ws, "rev-parse", "HEAD")
    return ws, best, head


def test_with_file_overlays_one_file_on_a_ref_without_touching_the_tree(tmp_path):
    ws, best, head = _repo(tmp_path)
    (ws / "scratch.txt").write_text("uncommitted\n", encoding="utf-8")   # a dirty tree stays dirty, untouched
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    new = be.with_file(best, "notes/timeline_probe.py", "new probe\n", "overlay")
    assert new != best and len(new) == 40
    # the child of ``best``, differing in that one file only
    assert _git(ws, "rev-parse", f"{new}^") == best
    assert _git(ws, "show", f"{new}:notes/timeline_probe.py") == "new probe"
    assert _git(ws, "show", f"{new}:kernel.py") == "k1"
    assert _git(ws, "diff", "--name-only", best, new) == "notes/timeline_probe.py"
    assert _git(ws, "log", "-1", "--format=%s", new) == "overlay"
    # HEAD, the index and the working tree are exactly as they were
    assert _git(ws, "rev-parse", "HEAD") == head
    assert _git(ws, "status", "--porcelain") == "?? scratch.txt"
    assert (ws / "notes" / "timeline_probe.py").read_text(encoding="utf-8") == "old probe\n"
    # already that text: the ref itself, no commit made
    assert be.with_file(new, "notes/timeline_probe.py", "new probe\n", "overlay") == new
    # a path the tree does not have is added
    added = be.with_file(best, "notes/extra.py", "x\n", "add")
    assert _git(ws, "show", f"{added}:notes/extra.py") == "x" and _git(ws, "rev-parse", f"{added}^") == best
    # the same overlay on the same ref builds the same tree
    again = be.with_file(best, "notes/timeline_probe.py", "new probe\n", "overlay")
    assert _git(ws, "diff", "--stat", new, again) == "" and _git(ws, "rev-parse", f"{again}^") == best


def test_with_file_on_an_unknown_ref_is_a_backend_error(tmp_path):
    ws, best, head = _repo(tmp_path)
    be = cli.RealBackend("demo", ws, None, "n", "e", poll_s=0.0)
    with pytest.raises(BackendError, match="git read-tree failed on deadbeef"):
        be.with_file("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "notes/timeline_probe.py", "new\n", "m")
    # nothing leaked into the tree
    assert _git(ws, "rev-parse", "HEAD") == head and _git(ws, "status", "--porcelain") == ""
