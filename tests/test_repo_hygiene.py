"""The operator boundary as an executable invariant.

The main repo is the operator-agnostic factory: it tracks no operator tree and no
run. Everything a run produces -- milestones, recon reports, measured baselines,
student trajectories -- lives in the run's directory under ``.artifacts/`` (or a
hand-made tree under ``operators/``), because a tracked answer sheet contaminates
every later run of the same operator: its teacher can read the previous run's
conclusions, and the run's "distilled from zero" claim is void.

``.gitignore`` already blocks casual adds, but it neither untracks history nor
stops ``git add -f`` -- this test turns the boundary into a suite failure
instead of a code-review catch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _tracked(prefix: str) -> list[str]:
    proc = subprocess.run(["git", "-C", str(REPO), "ls-files", prefix],
                          capture_output=True, text=True)
    if proc.returncode != 0:  # not a git checkout (tarball install) — nothing to guard
        pytest.skip("not a git checkout")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def test_no_operator_tree_or_run_is_tracked():
    offenders = _tracked("operators/") + _tracked(".artifacts/")
    assert not offenders, (
        "answer-sheet contamination: the main repo tracks op-specific files -- "
        "these belong in a run's directory, never here (a later run of "
        f"the same operator would distill from them): {offenders}")
