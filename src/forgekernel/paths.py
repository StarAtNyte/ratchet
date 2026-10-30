"""Where the harness finds its files.

The harness code, the hardware knowledge bases and the prompts live in the repository
(``repo_root``). The operator trees a run works on -- the student's tree, the reference
tree, the recon tree -- live under ``operators_dir``: the repository's ``operators/`` by
default, or a run directory's ``operators/`` when ``FORGEKERNEL_OPERATORS_DIR`` names one.
``forge run`` sets it for itself, and every subprocess it starts (the submit CLI, a GPU
runner) inherits it, so one run never sees another run's trees.

A caller that is handed an explicit base directory (``--base-dir``, ``base_dir=``) keeps
``<base>/operators``: an explicit base is a test or a tool pointed at one tree on purpose.
"""

from __future__ import annotations

import os
from pathlib import Path

OPERATORS_ENV = "FORGEKERNEL_OPERATORS_DIR"


def repo_root() -> Path:
    # src/forgekernel/paths.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def operators_dir(default_root: str | Path | None = None) -> Path:
    """The operator trees of the current run (``FORGEKERNEL_OPERATORS_DIR``), else
    ``<default_root>/operators`` (``default_root`` defaults to the repository)."""
    env = os.environ.get(OPERATORS_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(default_root if default_root is not None else repo_root()) / "operators"


def template_dir() -> Path:
    """The operator template the harness generates trees from (always the repository's)."""
    return Path(__file__).resolve().parent / "card_template"


def global_config_dir() -> Path:
    """The bottom configuration layer every task is merged onto (models.toml, loop.toml,
    backends.toml)."""
    return repo_root() / "global_config"


def skills_dir() -> Path:
    """The tools the agents use, one directory per skill (SKILL.md + its scripts)."""
    return repo_root() / "skills"
