"""Job-environment failures: a tool that cannot start on the image is not a reading
about the tree.

A harness job can fail two ways that look alike in its log: the tree under test is
wrong (a kernel that does not build, a correctness miss), or the job cannot run the
tool on this image at all — a launch target imports a binding the image does not
ship (``import cuda.cudart`` against cuda-python 13, which only has
``cuda.bindings``), a driver tool re-spawns a bare ``python3`` that has no torch on
an ssh host (a non-login ``ssh`` shell does not put the image's venv on PATH). The
second kind replays identically on every retry and says nothing about the tree.

A fused-norm-rope B200 run read it as the first kind from start to end: 97 of
97 NCU collections, 33 of 33 timelines and 7 of 7 stall ledgers died on the two
imports above while the loop logged "ncu empty on a passing kernel -> retry",
"base-bench-failed-or-incorrect" and "job returned no ledger", and no line anywhere
said the run was blind.

The detector is applied only where the same tree has already passed its own eval
(the NCU job of a passing kernel, the base bench of a green tree), so an import
failure there is the job's environment by construction.
"""
from __future__ import annotations

import re

_MISSING_MODULE_RE = re.compile(r"ModuleNotFoundError: No module named '([^']+)'")
# The student's own tree modules: a missing kernel module is the tree, not the image.
_TREE_MODULES = frozenset({"kernel"})
_OTHER_ENV_RES = (
    re.compile(r"ImportError: [^\n]*cannot open shared object file[^\n]*"),
    re.compile(r"(?m)^[^\n]*\b(?:python3?|ncu|nvcc|nvidia-smi): command not found"),
)


def env_failure(log: str | None) -> str | None:
    """The first job-environment failure in ``log`` as one short line, else None."""
    if not log:
        return None
    for m in _MISSING_MODULE_RE.finditer(log):
        if m.group(1).split(".")[0] not in _TREE_MODULES:
            return f"ModuleNotFoundError: No module named {m.group(1)!r}"
    for rx in _OTHER_ENV_RES:
        if (m := rx.search(log)) is not None:
            return m.group(0).strip()[:200]
    return None
