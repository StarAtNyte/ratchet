"""Tests for main-repo harvest accounting (the artifacts nothing commits)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from forgekernel.loop.harvest import (
    Change,
    format_report,
    git_status,
    kb_blockers,
    parse_porcelain,
)

PORCELAIN = """\
?? hw_kb/cutedsl/sm90/tma_store_epilogue.md
 M hw_kb/cutedsl/sm90/index.md
A  hw_kb/cutedsl/sm90/wgmma.md
R  hw_kb/cutedsl/sm90/old.md -> hw_kb/cutedsl/sm90/new.md
"""


def test_parse_splits_status_and_path():
    changes = parse_porcelain(PORCELAIN)
    assert [c.path for c in changes] == [
        "hw_kb/cutedsl/sm90/tma_store_epilogue.md",
        "hw_kb/cutedsl/sm90/index.md",
        "hw_kb/cutedsl/sm90/wgmma.md",
        "hw_kb/cutedsl/sm90/new.md",  # rename recorded under its destination
    ]
    assert [c.status for c in changes] == ["??", " M", "A ", "R "]


def test_untracked_and_added_are_new_modified_is_not():
    changes = parse_porcelain(PORCELAIN)
    assert [c.is_new for c in changes] == [True, False, True, False]


def test_parse_ignores_blank_and_truncated_lines():
    assert parse_porcelain("\n\n?? \nx\n") == []


def test_parse_unquotes_paths_with_specials():
    assert parse_porcelain('?? "hw_kb/cutedsl/sm90/a b.md"') == [Change("??", "hw_kb/cutedsl/sm90/a b.md")]


def test_report_names_the_root_so_the_wrong_checkout_is_visible(tmp_path):
    out = format_report(parse_porcelain(PORCELAIN), tmp_path)
    assert str(tmp_path) in out
    assert "new      hw_kb/cutedsl/sm90/tma_store_epilogue.md" in out
    assert "modified hw_kb/cutedsl/sm90/index.md" in out


def test_report_on_clean_tree_says_nothing_left():
    assert format_report([], Path("/repo")) == "[harvest] nothing left behind in the main repo"


# --- git-backed helpers ----------------------------------------------------

def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "hw_kb" / "cutedsl" / "sm90").mkdir(parents=True)
    (tmp_path / "hw_kb" / "cutedsl" / "sm90" / "index.md").write_text("# index\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


def test_kb_blockers_empty_on_clean_tree(tmp_path):
    root = _repo(tmp_path)
    assert kb_blockers(root, root / "hw_kb" / "cutedsl") == []


def test_kb_blockers_sees_an_uncommitted_recipe(tmp_path):
    root = _repo(tmp_path)
    (root / "hw_kb" / "cutedsl" / "sm90" / "wgmma.md").write_text("# recipe\n")
    blockers = kb_blockers(root, root / "hw_kb" / "cutedsl")
    assert [c.path for c in blockers] == ["hw_kb/cutedsl/sm90/wgmma.md"]
    assert blockers[0].is_new


def test_kb_blockers_ignores_changes_outside_the_kb(tmp_path):
    root = _repo(tmp_path)
    (root / "other.md").write_text("noise\n")
    assert kb_blockers(root, root / "hw_kb" / "cutedsl") == []


def test_kb_outside_the_repo_is_not_guarded(tmp_path):
    root = _repo(tmp_path)
    assert kb_blockers(root, tmp_path.parent / "elsewhere" / "hw_kb" / "cutedsl") == []


def test_git_status_returns_none_outside_a_repo(tmp_path):
    assert git_status(tmp_path / "not-a-repo", ["."]) is None


def test_harvest_failure_never_raises(tmp_path, monkeypatch):
    """Bookkeeping must not be able to take the loop down."""
    def _boom(*a, **k):
        raise OSError("git missing")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert git_status(tmp_path, ["."]) is None
