"""The evaluation harness is the card's, not the student's.

A job judges whatever ``gpu_entry.sh`` and ``eval/`` its tree carries, and the tree a
job runs is the student's lineage: every student turn ends in ``git add -A``, so an edit
to ``eval/bench.py`` (a looser tolerance, a correctness block that always passes, a
leaderboard.py whose timer skips the kernel) used to reach the next eval and be judged
by itself. Nothing the launch preflights checked could see it -- they read the CARD's
copies, before the run.

Two layers, neither relying on the agent sandbox:

* **driver path check** (``[loop] harness_guard``, default ``"revert"``): a delivered
  commit whose diff from the tree the iteration started on touches a harness path is
  REVERTed before any job is bought, with the paths named in the record; and at the
  start of every iteration the lineage is put back to the launch tree's harness files
  (a probe commit left behind by an iteration that ended without a record, a file a
  teacher turn left in the shared workspace) in a commit of its own.
  ``"warn"`` logs instead of reverting; ``"off"`` disables both.
* **job overlay** (``RealBackend._with_job_env``): every pushed job tree carries the
  card's tracked harness files byte for byte (binary blobs and the sealed reference
  included), and harness files the card does not have are removed from it. The
  lineage and the recorded shas are unchanged -- the overlay is a child commit, the
  same channel ``[loop] job_env_files`` always was.

What counts as harness: ``HARNESS_PREFIXES`` below, minus the kernel file and
``notes/`` (the student's tree). ``window.toml`` and README are export-ignored and the
driver reads the card's copies, so they are not listed; ``.gitattributes`` is, because
``git archive`` reads its export rules from the archived tree.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# A prefix ending in "/" names a directory; the others name a file.
HARNESS_PREFIXES = ("eval/", "gpu_entry.sh", "ref_bin/", ".gitattributes")
# The same set as git pathspecs (directories without the slash).
HARNESS_PATHSPECS = tuple(p.rstrip("/") for p in HARNESS_PREFIXES)

GUARD_REVERT = "revert"
GUARD_WARN = "warn"
GUARD_OFF = "off"
GUARD_MODES = (GUARD_REVERT, GUARD_WARN, GUARD_OFF)


def resolve_guard_mode(cfg: dict) -> str:
    """[loop] harness_guard: "revert" (default) | "warn" | "off" (true / false read as
    "revert" / "off"). Anything else raises ValueError -- a typo must not read as off."""
    raw = (cfg.get("loop", {}) or {}).get("harness_guard", GUARD_REVERT)
    if isinstance(raw, bool):
        return GUARD_REVERT if raw else GUARD_OFF
    mode = str(raw).strip().lower()
    if mode not in GUARD_MODES:
        raise ValueError(f"[loop] harness_guard must be one of {', '.join(GUARD_MODES)}, "
                         f"got {raw!r}")
    return mode


def is_harness_path(path: str, kernel_file: str = "") -> bool:
    """Whether ``path`` (tree-relative, posix) is card-owned evaluation harness."""
    p = str(path).strip()
    if p.startswith("./"):
        p = p[2:]
    if not p or p == str(kernel_file) or p.split("/")[0] == "notes":
        return False
    return any(p == pre.rstrip("/") or (pre.endswith("/") and p.startswith(pre))
               for pre in HARNESS_PREFIXES)


def harness_only(paths, kernel_file: str = "") -> list[str]:
    """The harness paths among ``paths``, sorted and de-duplicated."""
    return sorted({p for p in (paths or ()) if is_harness_path(p, kernel_file)})


def describe(paths, limit: int = 6) -> str:
    paths = list(paths)
    return ", ".join(paths[:limit]) + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def _git(repo: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=text)


def is_git_repo(path: Path) -> bool:
    """``path`` is the top of its own git repository. A card that was never ``git init``-ed
    but sits inside another repository (operators/<op> under ForgeKernel, gitignored
    there) is not: its tracked set would read as empty, and an empty card harness would
    strip gpu_entry.sh and eval/ from every job tree."""
    try:
        out = _git(Path(path), "rev-parse", "--show-toplevel")
    except OSError:
        return False
    if out.returncode != 0:
        return False
    try:
        return Path(out.stdout.strip()).resolve() == Path(path).resolve()
    except OSError:
        return False


def card_harness_files(card_dir: Path, kernel_file: str = "") -> dict[str, Path] | None:
    """The card's TRACKED harness files: {tree path: the card's working copy}. Tracked
    only -- a gitignored file next to them (a sealed reference's key) must never ride a
    job tree. None when the card is not a git repository (nothing to read the tracked
    set from) or when it tracks none -- a card without a judge is a listing gone wrong
    (nothing committed yet), and reading it as empty would strip gpu_entry.sh and eval/
    from every job tree; files tracked but missing on disk are left out."""
    card_dir = Path(card_dir)
    if not is_git_repo(card_dir):
        return None
    out = _git(card_dir, "ls-files", "-z", "--", *HARNESS_PATHSPECS)
    if out.returncode != 0:
        return None
    files = {}
    for rel in out.stdout.split("\0"):
        if rel and is_harness_path(rel, kernel_file) and (card_dir / rel).is_file():
            files[rel] = card_dir / rel
    return files or None


def tree_harness_files(repo: Path, ref: str, kernel_file: str = "") -> dict[str, tuple[str, str]] | None:
    """{path: (mode, blob sha)} of the harness files in commit ``ref`` of ``repo``."""
    out = _git(Path(repo), "ls-tree", "-r", "-z", ref, "--", *HARNESS_PATHSPECS)
    if out.returncode != 0:
        return None
    files = {}
    for entry in out.stdout.split("\0"):
        if not entry or "\t" not in entry:
            continue
        meta, rel = entry.split("\t", 1)
        mode, _kind, sha = meta.split()
        if is_harness_path(rel, kernel_file):
            files[rel] = (mode, sha)
    return files


def blob_sha(repo: Path, data: bytes, write: bool = False) -> str:
    args = ["hash-object", "--stdin"] + (["-w"] if write else [])
    out = subprocess.run(["git", "-C", str(repo), *args], input=data, capture_output=True)
    if out.returncode != 0:
        raise OSError(f"git hash-object failed: {out.stderr.decode(errors='replace')[:200]}")
    return out.stdout.decode().strip()


def harness_drift(card_dir: Path, workspace: Path, ref: str = "HEAD",
                  kernel_file: str = "") -> list[str] | None:
    """How the harness in ``workspace``'s ``ref`` differs from the card's: ``~path``
    (content), ``+path`` (in the card, missing from the tree), ``-path`` (in the tree, not
    in the card). [] = identical; None = cannot say (the card or the tree is not a git
    repository)."""
    card = card_harness_files(card_dir, kernel_file)
    tree = tree_harness_files(workspace, ref, kernel_file)
    if card is None or tree is None:
        return None
    out = []
    for rel, src in sorted(card.items()):
        if rel not in tree:
            out.append(f"+{rel}")
            continue
        try:
            if blob_sha(workspace, src.read_bytes()) != tree[rel][1]:
                out.append(f"~{rel}")
        except OSError:
            out.append(f"~{rel}")
    out += [f"-{rel}" for rel in sorted(tree) if rel not in card]
    return out
