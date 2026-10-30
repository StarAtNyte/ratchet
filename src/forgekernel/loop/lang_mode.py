"""Language modes: the CuTe DSL mode and the CUDA mode, kept apart.

A loop ``language`` (loop.toml ``[loop].language`` / ``forge loop --language``)
used to steer one thing only -- the ``{LANGUAGE}`` token in the student and
distill prompts. Everything else the loop hands a run (the prompts, the hw_kb
tree, the timeline instrumentation, the census and pricing harnesses) was
written for a Python CuTe DSL ``kernel.py``. A CUDA C++ student (``kernel.cu``,
language ``cuda-sol``) then read DSL recipes, was told to keep a Python
``def _seg`` marker, and bought timeline / pricing jobs that could not read or
outran its kernel.

This module is the one place that says which MODE a language belongs to and
what each mode reads, so the two modes stay independent:

- the **cutedsl mode** (every language that is not a CUDA mode language) reads
  ``meta/<prompt>``, ``hw_kb/cutedsl/<arch>/``,
  ``docs/kb-provenance/``. Nothing here changes a byte of its path.
- the **cuda mode** (``CUDA_LANGUAGES``) reads its own prompt set
  (``meta/cuda/<prompt>`` wherever one exists, the shared language-neutral
  prompt otherwise), its own KB root (``hw_kb/cuda/<arch>/``) and files its
  provenance under ``docs/kb-provenance/cuda/``. Its tools key on the kernel
  file's suffix (``.cu``), which ``language_problem`` keeps in step with the
  language.
"""
from __future__ import annotations

from pathlib import Path

MODE_CUTEDSL = "cutedsl"
MODE_CUDA = "cuda"

# The languages whose student writes a CUDA C++ kernel file the card builds with
# nvcc. cuda-sol is the leaderboard cuda_cpp solution (one .cu with its own
# PYBIND11_MODULE binding, built the way SOL-ExecBench builds it); it is the only
# CUDA language with a card harness behind it.
CUDA_LANGUAGES = frozenset({"cuda-sol"})
CUDA_KERNEL_SUFFIXES = (".cu",)

KB_ROOT = {MODE_CUTEDSL: "hw_kb/cutedsl", MODE_CUDA: "hw_kb/cuda"}
# Under the shared docs/kb-provenance/ root, so every "startswith docs/kb-provenance/"
# listing (harvest leftovers, publish) still covers it, while an entry name the two
# KBs share (gpu_probe_ladder, ...) keeps one provenance file per KB.
PROVENANCE_SUBDIR = {MODE_CUTEDSL: "", MODE_CUDA: "cuda"}
PROMPT_OVERLAY_DIR = {MODE_CUTEDSL: None, MODE_CUDA: "cuda"}

DEFAULT_CUDA_LANGUAGE = "cuda-sol"


def mode_of(language: str | None) -> str:
    """The mode a loop language belongs to (anything not CUDA is the DSL mode)."""
    return MODE_CUDA if language in CUDA_LANGUAGES else MODE_CUTEDSL


def is_cuda(language: str | None) -> bool:
    return mode_of(language) == MODE_CUDA


def is_cuda_kernel_file(kernel_file: str | Path | None) -> bool:
    return str(kernel_file or "").endswith(CUDA_KERNEL_SUFFIXES)


def kb_root_name(language: str | None) -> str:
    """The KB root directory (under the repo) the language's mode reads."""
    return KB_ROOT[mode_of(language)]


def kb_repo_root(kb_dir: Path) -> Path:
    """The repository a mode's KB root (``hw_kb/<mode>``) sits in."""
    return Path(kb_dir).parents[1]


def provenance_dir(repo_root: Path, language: str | None) -> Path:
    """Where the mode's KB provenance lives (human review only, no agent read set)."""
    base = Path(repo_root) / "docs" / "kb-provenance"
    sub = PROVENANCE_SUBDIR[mode_of(language)]
    return base / sub if sub else base


def owns_provenance(rel_path: str, language: str | None) -> bool:
    """Whether a repo-relative ``docs/kb-provenance/...`` path belongs to the
    language's mode: the CUDA mode owns ``docs/kb-provenance/cuda/``, the DSL mode
    everything else under the root (the shared per-op ``gaps/`` archive included)."""
    in_cuda = str(rel_path).startswith(f"docs/kb-provenance/{PROVENANCE_SUBDIR[MODE_CUDA]}/")
    return in_cuda if is_cuda(language) else not in_cuda


def prompt_path(meta_dir: Path, name: str, language: str | None) -> Path:
    """``meta/cuda/<name>`` for the CUDA mode when that overlay exists, else
    ``meta/<name>``. A prompt with no overlay is language-neutral and shared."""
    overlay = PROMPT_OVERLAY_DIR[mode_of(language)]
    if overlay:
        p = Path(meta_dir) / overlay / name
        if p.is_file():
            return p
    return Path(meta_dir) / name


def language_for_kernel_file(kernel_file: str | Path | None) -> str | None:
    """The language a kernel file implies when the run names none: a ``.cu`` is
    the CUDA mode's student; anything else leaves the old default alone."""
    return DEFAULT_CUDA_LANGUAGE if is_cuda_kernel_file(kernel_file) else None


def language_problem(language: str, kernel_file: str | Path) -> str | None:
    """Why ``language`` cannot drive a student that edits ``kernel_file``, or None.

    The language picks the prompts and the KB, the file picks what the card
    builds; out of step, the student is told to write CuTe DSL into a ``.cu`` (or
    C++ into a ``.py``) and every tool reads the wrong thing."""
    cu = is_cuda_kernel_file(kernel_file)
    if is_cuda(language) and not cu:
        return (f"language {language!r} is the CUDA mode but the kernel file is "
                f"{kernel_file!r}: a CUDA student edits a .cu (scaffold the card with "
                f"--kernel-file kernel.cu)")
    if cu and not is_cuda(language):
        return (f"the kernel file {kernel_file!r} is CUDA C++ but language is "
                f"{language!r}: set loop.toml [loop].language = "
                f"{DEFAULT_CUDA_LANGUAGE!r} (or leave it unset to follow the file)")
    return None
