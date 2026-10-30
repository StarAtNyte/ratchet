"""Tests for L0-isolated student workspace setup (single-branch clone)."""

import subprocess

from forgekernel.inner.workspace import can_resolve, setup_student_workspace


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _make_source_with_secret(src):
    src.mkdir()
    _git("init", "-q", cwd=src)
    _git("config", "user.email", "t@t", cwd=src)
    _git("config", "user.name", "t", cwd=src)
    (src / "flash_bwd.py").write_text("# baseline\n")
    _git("add", "-A", cwd=src)
    _git("commit", "-qm", "baseline", cwd=src)
    main = subprocess.run(["git", "branch", "--show-current"], cwd=src,
                          capture_output=True, text=True).stdout.strip()
    # a "secret" sibling branch the student must NOT be able to read (L0)
    _git("checkout", "-qb", "secret-optimized", cwd=src)
    (src / "flash_bwd.py").write_text("# the optimized answer\n")
    _git("commit", "-aqm", "answer", cwd=src)
    _git("checkout", "-q", main, cwd=src)
    return main


def test_single_branch_clone_isolates_siblings(tmp_path):
    src = tmp_path / "src"
    main = _make_source_with_secret(src)
    wd = tmp_path / "wd"

    setup_student_workspace(src, main, wd)

    assert (wd / "flash_bwd.py").read_text() == "# baseline\n"
    assert can_resolve(wd, "HEAD")
    # L0: the sibling answer branch is physically absent from the workspace
    assert not can_resolve(wd, "secret-optimized")
    assert not can_resolve(wd, "origin/secret-optimized")
