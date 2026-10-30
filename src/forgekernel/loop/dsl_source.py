"""Where an offline student may read the DSL's own Python source from.

The student runs without network (loop/agent_policy.py), so the CuTe-DSL API it writes
against has to be readable from disk. ``[loop] dsl_source_dirs`` names that checkout
explicitly; when a run leaves it empty, the student is left to discover ``cutlass.cute``
by trial and error — in one GDN run a large share of the probe failures were spellings
of the 4.6.0 API, not ideas being tested.

This module derives a default for a ``cutedsl`` run from the ``nvidia-cutlass-dsl``
distribution installed in a Python interpreter on this machine (the one running the
loop, or ``[loop] dsl_python``): the wheel family puts the whole Python DSL under one
directory (``nvidia_cutlass_dsl/dsl_packages/``, holding ``cutlass`` and its siblings;
a ``.pth`` file adds it to ``sys.path``), and that directory is what the student needs.

Three things keep the default honest:

* **Version.** The job pins the DSL with ``[env] DSL_PIN`` (submit.toml); a local
  install of another version is not the API the kernel is compiled against and must
  not be handed over silently. Mismatch → no default, a warning that names both.
* **Reference source.** A default that lies inside (or contains) a declared
  ``ref_source_paths`` entry would hand the student the reference through the DSL
  door. Overlap → no default, a note that says so.
* **Explicit wins.** A declared ``dsl_source_dirs`` is used as written, whatever is
  installed here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LANGUAGE_CUTEDSL = "cutedsl"
KEY_DSL_PYTHON = "dsl_python"
ENV_DSL_PIN = "DSL_PIN"
DIST_SHELL = "nvidia-cutlass-dsl"
DIST_CORE = "nvidia-cutlass-dsl-libs-core"   # the pure-Python DSL body
PROBE_TIMEOUT_S = 60

ORIGIN_EXPLICIT = "explicit"
ORIGIN_DEFAULT = "default"
ORIGIN_NONE = "none"

# Runs in the target interpreter: report the distribution's version and the package
# roots the DSL imports from. Metadata first (no import side effects, works for an
# interpreter that cannot import the DSL, e.g. a CPU-only venv holding the wheel);
# ``import cutlass`` as the fallback for an install without dist-info (a source
# checkout on sys.path).
_PROBE = r"""
import importlib.metadata as m, json, os, sys
out = {"version": None, "roots": [], "how": None}
def _dist(name):
    try:
        return m.distribution(name)
    except m.PackageNotFoundError:
        return None
for name in (%(shell)r, %(core)r):
    d = _dist(name)
    if d is not None and out["version"] is None:
        out["version"] = d.version
core = _dist(%(core)r)
roots = []
if core is not None and core.files:
    for f in core.files:
        parts = f.parts
        if len(parts) >= 3 and parts[-1].endswith(".py") and "dsl_packages" in parts:
            i = parts.index("dsl_packages")
            roots.append(str((core.locate_file(f).resolve().parents[len(parts) - i - 2])))
    out["how"] = "metadata"
if not roots:
    try:
        import cutlass
        roots = [str(os.path.dirname(os.path.abspath(p))) for p in cutlass.__path__]
        out["how"] = "import"
    except Exception:
        pass
seen = []
for r in roots:
    if r not in seen:
        seen.append(r)
out["roots"] = seen
print(json.dumps(out))
"""


@dataclass(frozen=True)
class DslSource:
    """The resolved ``dsl_source_dirs`` and where they came from.

    ``origin``: ``explicit`` (declared in loop.toml), ``default`` (derived from an
    installed distribution), ``none`` (nothing usable). ``version`` is the installed
    distribution's version when one was found, whether or not it was accepted.
    ``notes`` are what the loop prints — one line each, ``[loop]``-prefixed by the
    caller."""

    dirs: tuple[Path, ...] = ()
    origin: str = ORIGIN_NONE
    version: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def defaulted(self) -> bool:
        return self.origin == ORIGIN_DEFAULT

    def describe(self) -> str:
        """The dry-run line's value."""
        if self.origin == ORIGIN_EXPLICIT:
            return f"explicit {[str(p) for p in self.dirs]}"
        if self.origin == ORIGIN_DEFAULT:
            return (f"default {[str(p) for p in self.dirs]}  "
                    f"({DIST_SHELL} {self.version}, from the installed distribution)")
        why = self.notes[-1] if self.notes else "not declared"
        return f"(none — {why})"


def _abs(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def probe_interpreter(python: str, *, env: dict | None = None,
                      timeout_s: float = PROBE_TIMEOUT_S) -> dict | None:
    """``{"version", "roots", "how"}`` from ``python``, or None when the probe could
    not run (missing interpreter, timeout, unparsable output)."""
    code = _PROBE % {"shell": DIST_SHELL, "core": DIST_CORE}
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True,
                              timeout=timeout_s, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("roots"), list):
        return None
    return data


def _overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def resolve(cfg: dict, *, language: str, submit_env: dict | None = None,
            python: str | None = None, probe_env: dict | None = None,
            pin_readable: bool = True) -> DslSource:
    """The run's ``dsl_source_dirs``: as declared, else derived, else none.

    ``cfg`` is the parsed loop.toml; ``language`` the resolved kernel language;
    ``submit_env`` the card's ``[env]`` (for ``DSL_PIN``); ``python`` overrides the
    interpreter (``[loop] dsl_python`` then ``sys.executable``).

    ``pin_readable`` False means the caller could not read the card's submit config
    at all, so ``submit_env`` is not evidence of "this card pins nothing": the
    version gate would be off without saying so, and a default derived here would
    hand the student whatever this machine happens to have installed. Refuse to
    default instead — an unreadable card is a config problem the operator can see,
    a silent version mismatch is not."""
    lc = cfg.get("loop", {}) if isinstance(cfg, dict) else {}
    declared = lc.get("dsl_source_dirs") or []
    if isinstance(declared, list) and declared:
        return DslSource(dirs=tuple(_abs(p) for p in declared), origin=ORIGIN_EXPLICIT)
    if language != LANGUAGE_CUTEDSL:
        return DslSource(notes=(f"language={language!r}: no DSL source to default",))
    if not pin_readable:
        return DslSource(notes=(
            f"dsl_source_dirs is empty and the card's submit config could not be read "
            f"(submit.toml / submit.local.toml under operators/) — its [env] {ENV_DSL_PIN} "
            f"cannot be checked against the installed {DIST_SHELL}, so nothing is "
            f"defaulted. Fix the card, or declare [loop] dsl_source_dirs explicitly",))

    interp = python or lc.get(KEY_DSL_PYTHON) or sys.executable
    interp_note = f"[loop] {KEY_DSL_PYTHON}" if lc.get(KEY_DSL_PYTHON) else "the loop's interpreter"
    data = probe_interpreter(str(interp), env=probe_env)
    if data is None:
        return DslSource(notes=(
            f"dsl_source_dirs is empty and {DIST_SHELL} could not be located in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))
    version = data.get("version")
    roots = [Path(r).resolve() for r in data.get("roots", []) if r]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        return DslSource(version=version, notes=(
            f"dsl_source_dirs is empty and {DIST_SHELL} is not installed in "
            f"{interp_note} ({interp}) — the student is offline, so it can read the "
            f"DSL's API only from a local checkout named in [loop] dsl_source_dirs",))

    pin = (submit_env or {}).get(ENV_DSL_PIN)
    pin = str(pin).strip() if pin is not None else ""
    if pin and version and pin != version:
        return DslSource(version=version, notes=(
            f"dsl_source_dirs is empty; {DIST_SHELL} {version} is installed in "
            f"{interp_note} but the job pins [env] {ENV_DSL_PIN} = {pin} — not defaulting "
            f"to another version's source. Install {pin} there, point [loop] "
            f"{KEY_DSL_PYTHON} at an interpreter that has it, or declare [loop] "
            f"dsl_source_dirs explicitly",))
    if pin and not version:
        return DslSource(notes=(
            f"dsl_source_dirs is empty; a DSL package was found in {interp_note} but "
            f"without a {DIST_SHELL} version to check against [env] {ENV_DSL_PIN} = {pin} "
            f"— declare [loop] dsl_source_dirs explicitly",))

    refs = [_abs(p) for p in (lc.get("ref_source_paths") or [])
            if isinstance(p, str)]
    clash = [(r, ref) for r in roots for ref in refs if _overlaps(r, ref)]
    if clash:
        r, ref = clash[0]
        return DslSource(version=version, notes=(
            f"dsl_source_dirs is empty; the installed {DIST_SHELL} at {r} overlaps the "
            f"declared reference source {ref} — not defaulting (the DSL door must not "
            f"open onto the reference). Declare [loop] dsl_source_dirs explicitly",))

    return DslSource(
        dirs=tuple(roots), origin=ORIGIN_DEFAULT, version=version,
        notes=(f"dsl_source_dirs: defaulted to {[str(r) for r in roots]} "
               f"({DIST_SHELL} {version or 'unversioned'}, {interp_note}"
               + (f", matches [env] {ENV_DSL_PIN}" if pin else "") + ")",))
