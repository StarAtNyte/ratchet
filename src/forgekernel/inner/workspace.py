"""L0-isolated student workspace.

The student must reproduce performance from the milestone WITHOUT reading the
operator's optimized implementation or the oracle. The mechanical guard is a
single-branch clone: sibling branches' refs (and thus any flash_bwd.py answer
on them) are physically absent from the workspace, so cross-branch git reads
simply fail. The oracle (flash-attn-4) is never installed locally either.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def setup_student_workspace(source_repo: str | Path, branch: str,
                            workdir: str | Path) -> Path:
    """Single-branch clone of `branch` from `source_repo` into `workdir`."""
    workdir = Path(workdir)
    subprocess.run(
        ["git", "clone", "--single-branch", "--branch", branch,
         str(source_repo), str(workdir)],
        check=True, capture_output=True,
    )
    return workdir


def can_resolve(workdir: str | Path, ref: str) -> bool:
    """Whether `ref` resolves inside `workdir` (used to assert L0 isolation)."""
    return subprocess.run(
        ["git", "-C", str(workdir), "rev-parse", "--verify", "--quiet", ref],
        capture_output=True,
    ).returncode == 0
