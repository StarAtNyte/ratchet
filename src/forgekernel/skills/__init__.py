"""The repository's skills/ as one importable package.

A skill is a directory under the repository's ``skills/``: a ``SKILL.md`` saying what it is
and how an agent uses it, and the scripts it ships (as in ForgeTrain). The harness uses the
same scripts as a library -- it parses their verdict lines, and the driver seeds them into a
tree's ``notes/`` so a GPU job runs them as ``python3 notes/<tool>.py`` -- so every skill
directory is on this package's path: ``from forgekernel.skills import sass_tools``.

Inside a tree the scripts sit side by side in ``notes/`` and import each other bare
(``import sass_tools``); here the bare import fails and each falls back to this package.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from forgekernel.paths import skills_dir

SKILL_FILE = "SKILL.md"
# Where Claude Code looks for a project's skills, relative to the agent's working directory.
MIRROR_RELPATH = Path(".claude") / "skills"


def skill_dirs() -> list[Path]:
    root = skills_dir()
    return sorted(d for d in root.iterdir() if (d / SKILL_FILE).is_file()) if root.is_dir() else []


def mirror(workspace: Path) -> list[str]:
    """Copy every skill into ``<workspace>/.claude/skills/<name>/`` (the caller keeps
    ``.claude/`` out of git), replacing whatever a previous turn left there."""
    dst_root = Path(workspace) / MIRROR_RELPATH
    names = []
    for d in skill_dirs():
        dst = dst_root / d.name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(d, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        names.append(d.name)
    return names


__path__ = [str(d) for d in skill_dirs()]
