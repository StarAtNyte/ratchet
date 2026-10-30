"""Sealed binary reference for a leaderboard card.

A blackbox run needs a reference the agents can run, time and profile (SASS included)
but never read. For a compiled solution that is its binary: a CUDA C++ kernel (e.g. a
gqa_paged_decode solution) is sealed with its sources in a recon tree and the .so in the
card's ref_bin/. This module does it for any SOL-ExecBench / FlashInfer-Bench cuda_cpp
solution — typically an existing best solution for the card's problem:

    forge init <op> --leaderboard sol-execbench ... --binary-ref solution.json \\
        [--ref-source-path PATH ...]
        -> card: problem.json reference_impl kind "binary" (ref_bin/<module>.so, timed
           in-job beside the kernel), a fence on kernel.py, a guarded `cmd` task, and a
           blackbox loop.toml.example;
        -> recon tree operators/_<op>ref/: src/ (the sources), binref.json (module,
           entry, compile options), binref.py (build / check / crosscheck / bench / ncu),
           a copy of the card's eval/ and a gpu_entry.sh whose `cmd` task is guarded.
    (recon tree, on a GPU)  bash gpu_entry.sh build      -> ref_bin/<module>.so
    python -m forgekernel.binref install <card> <so> --ref-tree <recon tree> [--note K=V ...]
        -> the .so and PROVENANCE.txt committed into the card and the recon tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

CPP_LANGUAGES = {"cuda_cpp", "cpp", "cuda", "c++"}
CPP_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
TOOL = Path(__file__).resolve().parent / "binref_tool.py"
REF_TASK = "binref"

# Command-line words that would read the sealed sources (or their locations) from a
# `cmd` job; SASS of the built .so stays open to the recon tree (cuobjdump -sass,
# ncu --print-source sass), line-level CUDA source and PTX do not.
REF_TREE_FORBID = [r"src/", r"binref\.json", r"binref_build", r"\.cuh?\b", r"\.cpp\b", r"\.cc\b",
                   r"\.cxx\b", r"\.hp?p?\b", r"getsource",
                   r"inspect\.getfile", r"find_spec", r"pip +(show|download)",
                   r"--print-source[= ]*(cuda|all|src|ptx)", r"--import-source",
                   r"--source-folders", r"-ptx\b", r"--dump-ptx", r"/Users/", r"find +/"]
# The C/C++ suffixes of that list are a tripwire for reaching the sealed sources without
# naming src/ (a glob, an endswith): they also refused every C/C++ PROBE the recon prompt asks
# the teacher for (the CUDA mode's expressibility ladder: a minimal .cu compiled in the job, a
# CUTLASS header included -- CUDA-mode runs lost whole recon runs to it). In the recon
# tree they are matched after the probe's own spellings are taken out (PROBE_ALLOW_SED): a
# plain file name, bare or under notes/, /tmp/, /usr/ or /opt/ (the toolkit's and CUTLASS's
# headers are evidence the prompt asks for), and an `#include <lib/path.h>`; a glob, a bare
# suffix, a path through any other directory (the tree, /workspace) and every word of
# REF_TREE_HARD (src/, binref_build, the sealed sources' own names, /Users/ ...) still refuse.
REF_TREE_SUFFIXES = [r"\.cuh?\b", r"\.cpp\b", r"\.cc\b", r"\.cxx\b", r"\.hp?p?\b"]
REF_TREE_HARD = [p for p in REF_TREE_FORBID if p not in REF_TREE_SUFFIXES]
# sed -E scripts (GNU and BSD): a library include, PTX's `.cc` carry modifier (add.cc.u32),
# then a probe's file name (twice: a match eats the separator the next name starts after).
_PROBE_FILE = (r"s,(^|[^A-Za-z0-9_./*-])((notes|/(tmp|usr|opt)(/[A-Za-z0-9_][A-Za-z0-9_.+-]*)*)/)?"
               r"[A-Za-z0-9_][A-Za-z0-9_-]*\.(cuh|cu|cpp|cc|cxx|hpp|h)([^A-Za-z0-9_]|$),\1\7,g")
PROBE_ALLOW_SED = (r"s,#[[:space:]]*include[[:space:]]*<[A-Za-z0-9_][A-Za-z0-9_/+-]*\.(cuh|hpp|hxx|hh|h|inl)>,,g",
                   r"s,\.cc(\.[A-Za-z0-9]),_cc\1,g", _PROBE_FILE, _PROBE_FILE)
# The card's `cmd` task (the student's probes) may not open the reference binary: its command
# line may not name the binary or a tool that reads one, its notes/ scripts may not name the
# binary. The tool names are not a reach inside notes/: the loop seeds the student's notes/
# with its SASS / pricing toolkit (sass_tools.py runs cuobjdump, orch_census_run.py nvdisasm,
# on the student's OWN cubins), and scanning for them there refused every student `cmd` from
# the first pricing round on.
# A C/C++ probe (or a CUDA kernel file compiled by a `cmd`) reaches through the dynamic loader.
CARD_REACH = [r"ref_bin", r"ExtensionFileLoader", r"\bdlopen\b", r"dl_iterate_phdr"]
# The C/C++ files of the card's notes/ are scanned too (a CUDA student compiles its probes).
CARD_CPP_NOTES = ("notes/*.cu", "notes/*.cuh", "notes/*.cpp", "notes/*.cc", "notes/*.h", "notes/*.hpp")
CARD_FORBID = CARD_REACH + [r"cuobjdump", r"nvdisasm", r"objdump", r"readelf", r"\bstrings\b",
                            r"\bnm\b"]
# The shared tripwire (scripts/blackbox-guard.sh) the loop seeds into a blackbox run's recon
# notes/ (loop.driver.BLACKBOX_GUARD_NAME): its forbid list is made of forbidden words, so the
# recon tree's scan skips it — the exemption that guard gives itself, whose own gpu_entry.sh
# sources it from this very path.
SEEDED_GUARD = "blackbox-guard.sh"


def load_solution(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def module_name(sol: dict) -> str:
    h = hashlib.sha256(json.dumps(sol.get("sources"), sort_keys=True).encode()).hexdigest()
    return f"binref_{h[:10]}"


def solution_problems(sol: dict, definition: str) -> list[str]:
    """Why this solution cannot be a card's binary reference (empty = it can)."""
    errs = []
    spec = sol.get("spec") or {}
    langs = {str(x).lower() for x in spec.get("languages") or []}
    if not langs & CPP_LANGUAGES:
        errs.append(f"languages {sorted(langs)}: a binary reference is a compiled (cuda_cpp) "
                    f"solution; seal a Python one with eval/closed_ref.py instead")
    if sol.get("definition") != definition:
        errs.append(f"solution is for {sol.get('definition')!r}, the card for {definition!r}")
    ep = spec.get("entry_point") or ""
    if "::" not in ep:
        errs.append(f"entry_point {ep!r} is not '<file>::<function>'")
    if not any("/" not in str(s.get("path", "")) and Path(str(s.get("path", ""))).suffix in CPP_SUFFIXES
               for s in sol.get("sources") or []):
        errs.append("no C/C++/CUDA source at the top of the solution: SOL-ExecBench compiles only "
                    "those (binref_tool.compiled_sources); files in subdirectories are #included")
    if not any("PYBIND11_MODULE(TORCH_EXTENSION_NAME" in (s.get("content") or "").replace(" ", "")
               for s in sol.get("sources") or []):
        errs.append("no PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...) in the sources: the build names "
                    "the module, so the binding must take its name from the build")
    return errs


def reference_impl(sol: dict, module: str, name: str | None = None) -> dict:
    spec = sol["spec"]
    return {"kind": "binary", "name": name or f"sealed binary reference {module}",
            "path": f"../../ref_bin/{module}.so", "module": module,
            "entry": spec["entry_point"].split("::", 1)[1],
            "dps": bool(spec.get("destination_passing_style", True))}


def _guard_block(patterns: list[str], notes_patterns: list[str], what: str,
                 skip_notes: tuple[str, ...] = (), extra_files: tuple[str, ...] = (),
                 soft: list[str] | None = None, allow_sed: tuple[str, ...] = (),
                 hard_files: tuple[str, ...] = ()) -> str:
    """``extra_files``: tree files scanned like notes/ (a sealed source reference's card
    scans its kernel.py: a `cmd` that imports it runs it outside the bench's fence).
    ``soft``: patterns matched (on the words and the scanned scripts) only after the
    ``allow_sed`` scripts took the allowed spellings out; ``hard_files``: files scanned
    for ``notes_patterns`` alone (a C/C++ probe: its `#include`s are its code)."""
    def alt(ps: list[str]) -> str:
        return "(" + "|".join(ps).replace("'", "'\\''") + ")"
    skip = " ".join(f"notes/{n}" for n in skip_notes)
    scanned = " ".join(["notes/*.py", "notes/*.sh", *extra_files])
    files = "notes/*.py|*.sh" + "".join(f", {f}" for f in extra_files)
    if not soft:
        return f'''
# Blackbox guard ({what}): a `cmd` whose words (BLACKBOX_FORBID), or whose {files}
# (BLACKBOX_FORBID_NOTES), reach for the sealed reference is refused.
BLACKBOX_FORBID='{alt(patterns)}'
BLACKBOX_FORBID_NOTES='{alt(notes_patterns)}'
blackbox_guard() {{
    if printf '%s' "$*" | grep -Eq -- "$BLACKBOX_FORBID"; then
        echo "[gpu_entry.sh] REFUSED: the command reaches for the sealed reference (blackbox guard)"
        exit 3
    fi
    local f
    for f in {scanned}; do
        [ -f "$f" ] || continue
        case " {skip} " in *" $f "*) continue ;; esac
        if grep -Eq -- "$BLACKBOX_FORBID_NOTES" "$f"; then
            echo "[gpu_entry.sh] REFUSED: $f reaches for the sealed reference (blackbox guard)"
            exit 3
        fi
    done
}}
'''
    seds = " ".join(f"-e '{s}'" for s in allow_sed)
    hard = " ".join(hard_files)
    hint = ("C/C++ probes: plain file names, bare or under notes/, /tmp/, /usr/ or /opt/, and "
            "#include <lib/path.h>; no globs or bare suffixes")
    return f'''
# Blackbox guard ({what}): a `cmd` whose words, or whose {files}, reach for the sealed
# reference is refused: BLACKBOX_FORBID as written, BLACKBOX_SOFT once a probe's own file
# names and library includes are taken out (blackbox_soft); the C/C++ probes in notes/
# ({hard or "none"}) against BLACKBOX_FORBID_NOTES.
BLACKBOX_FORBID='{alt(patterns)}'
BLACKBOX_FORBID_NOTES='{alt(notes_patterns)}'
BLACKBOX_SOFT='{alt(soft)}'
blackbox_soft() {{
    sed -E {seds} | grep -Eq -- "$BLACKBOX_SOFT"
}}
blackbox_guard() {{
    if printf '%s' "$*" | grep -Eq -- "$BLACKBOX_FORBID" || printf '%s\\n' "$*" | blackbox_soft; then
        echo "[gpu_entry.sh] REFUSED: the command reaches for the sealed reference (blackbox guard; {hint})"
        exit 3
    fi
    local f
    for f in {scanned}; do
        [ -f "$f" ] || continue
        case " {skip} " in *" $f "*) continue ;; esac
        if grep -Eq -- "$BLACKBOX_FORBID_NOTES" "$f" || blackbox_soft < "$f"; then
            echo "[gpu_entry.sh] REFUSED: $f reaches for the sealed reference (blackbox guard; {hint})"
            exit 3
        fi
    done
    for f in {hard}; do
        [ -f "$f" ] || continue
        if grep -Eq -- "$BLACKBOX_FORBID_NOTES" "$f"; then
            echo "[gpu_entry.sh] REFUSED: $f reaches for the sealed reference (blackbox guard)"
            exit 3
        fi
    done
}}
'''


def _split_dispatch(text: str) -> tuple[str, str, str]:
    """(before, the final `case "$TASK" in ... esac` dispatch, after) of a card gpu_entry.sh."""
    i = text.rindex('case "$TASK" in\n    correctness)')
    j = text.index("\nesac\n", i) + len("\nesac\n")
    return text[:i], text[i:j], text[j:]


def guard_card_gpu_entry(text: str, module: str, kernel_file: str = "kernel.py") -> str:
    """The card's gpu_entry.sh with its `cmd` task guarded against opening the binary. A CUDA
    kernel file is scanned like notes/: a `cmd` may compile it outside the bench's fence."""
    if "blackbox_guard" in text:
        return text
    before, dispatch, after = _split_dispatch(text)
    dispatch = dispatch.replace('    cmd)         run_with_timeout bash -c "$*" ;;',
                                '    cmd)         blackbox_guard "$*"; run_with_timeout bash -c "$*" ;;')
    if "blackbox_guard" not in dispatch:
        raise ValueError("card gpu_entry.sh: no `cmd` task to guard")
    mod = [re.escape(module)]
    extra = CARD_CPP_NOTES + ((kernel_file,) if Path(kernel_file).suffix in CPP_SUFFIXES else ())
    return before + _guard_block(CARD_FORBID + mod, CARD_REACH + mod, "card",
                                 extra_files=extra) + "\n" + dispatch + after


def ref_tree_gpu_entry(card_text: str, sources: list[str]) -> str:
    """The recon tree's gpu_entry.sh: the card's environment and clock lock, the
    binref tasks, and a guarded `cmd`."""
    card_text = re.sub(r"\n# Blackbox guard \(.*?\n}\n", "\n", card_text, flags=re.S)  # the card's own
    before, _, after = _split_dispatch(card_text)
    names = [re.escape(Path(s).name) for s in sources]
    head = ("#!/bin/bash\n# Recon tree of a sealed binary reference (forgekernel.binref) — tasks:\n"
            "#   build | check | crosscheck | bench (= ref) | ncu | binref (check + bench + ncu)"
            " | cmd <CMD> (guarded)\n")
    before = head + before.split("\n", 1)[1]
    tasks = '''case "$TASK" in
    build)       run_with_timeout python3 binref.py build ;;
    check)       run_with_timeout python3 binref.py check ;;
    crosscheck)  run_with_timeout python3 binref.py crosscheck ;;
    bench|ref)   run_with_timeout python3 binref.py bench ;;
    ncu)
        ncu_bounded ncu --csv --target-processes all --profile-from-start off \\
            --launch-count 24 --kill yes --metrics "$NCU_METRICS" python3 eval/ncu_run.py
        ;;
    binref)
        run_with_timeout python3 binref.py check
        run_with_timeout python3 binref.py bench
        ncu_bounded ncu --csv --target-processes all --profile-from-start off \\
            --launch-count 24 --kill yes --metrics "$NCU_METRICS" python3 eval/ncu_run.py
        ;;
    cmd)         blackbox_guard "$*"; run_with_timeout bash -c "$*" ;;
    help|*)
        sed -n '2,4p' "$0"
        exit 1
        ;;
esac
'''
    forbid = REF_TREE_HARD + names
    return before + _guard_block(forbid, forbid, "recon tree", skip_notes=(SEEDED_GUARD,),
                                 soft=REF_TREE_SUFFIXES, allow_sed=PROBE_ALLOW_SED,
                                 hard_files=CARD_CPP_NOTES) + "\n" + tasks + after


REF_NCU_RUN = '''"""NCU target of the recon tree: the sealed reference's calls on one input set inside
the profiler window (input construction and warm-up stay outside it)."""

from leaderboard import ncu_window

if __name__ == "__main__":
    ncu_window(which="ref")
'''

REF_GITIGNORE = "ref_bin/\nbuild/\n*.so\n__pycache__/\n*.pyc\nnotes/*.log\n"


def write_ref_tree(ref_dir: Path, card_dir: Path, sol: dict, module: str) -> list[Path]:
    """The recon tree next to the card: sources, meta, tool, the card's eval copy."""
    ref_dir, card_dir = Path(ref_dir), Path(card_dir)
    if ref_dir.exists():
        raise FileExistsError(f"{ref_dir} exists")
    created: list[Path] = []

    def put(rel: str, text: str, mode: int | None = None) -> None:
        p = ref_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        if mode:
            p.chmod(mode)
        created.append(p)

    sources = []
    for s in sol["sources"]:
        rel = s["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"source path {rel!r} leaves the tree")
        put(f"src/{rel}", s["content"])
        sources.append(rel)
    spec = sol.get("spec") or {}
    put("binref.json", json.dumps({
        "module": module, "definition": sol.get("definition"), "solution": sol.get("name"),
        "entry": spec["entry_point"].split("::", 1)[1], "languages": spec.get("languages"),
        "dps": bool(spec.get("destination_passing_style", True)),
        # null vs {} matters: SOL-ExecBench gives no flags without compile_options but the
        # defaults (-O3 --use_fast_math, -lcuda) for every key a present one leaves out.
        "compile_options": spec.get("compile_options"), "sources": sources}, indent=2) + "\n")
    shutil.copy2(TOOL, ref_dir / "binref.py")
    created.append(ref_dir / "binref.py")
    shutil.copytree(card_dir / "eval", ref_dir / "eval",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "ncu_run.py"))
    created.append(ref_dir / "eval")
    put("eval/ncu_run.py", REF_NCU_RUN)
    put("gpu_entry.sh", ref_tree_gpu_entry((card_dir / "gpu_entry.sh").read_text(), sources), 0o755)
    if (card_dir / "submit.toml").exists():
        text = (card_dir / "submit.toml").read_text()
        put("submit.toml", re.sub(r"(?m)^timeout\s*=\s*\d+", "timeout = 2700", text))
    put(".gitignore", REF_GITIGNORE)
    # No notes/ here: the loop seeds it at reconnaissance, and a non-empty notes/ on a fresh
    # start reads as a previous run's probes (loop.cli.fresh_start_contamination refuses).
    put("README.md", f"""# _{card_dir.name}ref — recon tree of a sealed binary reference

The card `{card_dir.name}` times its kernel against `ref_bin/{module}.so`, built here by
`bash gpu_entry.sh build` from sources that stay in this tree (`src/`, off limits to the
agents: `cmd` jobs that reach for them are refused). SASS of the built binary is open
(`cuobjdump -sass ref_bin/{module}.so`, `ncu --print-source sass`).

    bash gpu_entry.sh binref      check + bench + NCU in one job ([loop] ref_mode)
    bash gpu_entry.sh crosscheck  the card's judge on the binary: must PASS
""")
    return created


def install(card_dir: Path, so_path: Path, notes: dict | None = None, commit: bool = True,
            ref_tree: Path | None = None) -> Path:
    """Copy the built .so into the card's ref_bin/ with PROVENANCE.txt (and commit it).
    With ``ref_tree`` the recon tree gets the same binary committed too: its jobs start
    from a fresh extract of the tree, so without it every recon job rebuilds the .so —
    minutes each, and a rebuild is not byte-identical to what the card times."""
    card_dir = Path(card_dir).resolve()  # `git -C` resolves pathspecs against it
    problem = json.loads((card_dir / "eval" / "leaderboard" / "problem.json").read_text())
    ref = problem.get("reference_impl") or {}
    if ref.get("kind") != "binary":
        raise ValueError(f"{card_dir}: the card's reference is not a binary one")
    dst = card_dir / "ref_bin" / f"{ref['module']}.so"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so_path, dst)
    data = dst.read_bytes()
    lines = [f"module   {ref['module']}", f"entry    {ref.get('entry')} (dps={ref.get('dps')})",
             f"sha256   {hashlib.sha256(data).hexdigest()}", f"size     {len(data)}",
             f"installed {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "note     ABI-bound to the torch/CUDA of the image that built it: a job in another "
             "image cannot load it (no ref_* keys)"]
    lines += [f"{k:<8} {v}" for k, v in (notes or {}).items()]
    (dst.parent / "PROVENANCE.txt").write_text("\n".join(lines) + "\n")
    if commit:
        subprocess.run(["git", "-C", str(card_dir), "add", "-f", str(dst), str(dst.parent / "PROVENANCE.txt")],
                       check=True)
        subprocess.run(["git", "-C", str(card_dir), "commit", "-q", "-m",
                        f"ref_bin: sealed binary reference {ref['module']}"], check=True)
    if ref_tree is not None:
        ref_tree = Path(ref_tree).resolve()
        rdst = ref_tree / "ref_bin" / dst.name
        rdst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, rdst)
        shutil.copy2(dst.parent / "PROVENANCE.txt", rdst.parent / "PROVENANCE.txt")
        if commit:
            subprocess.run(["git", "-C", str(ref_tree), "add", "-f", str(rdst),
                            str(rdst.parent / "PROVENANCE.txt")], check=True)
            subprocess.run(["git", "-C", str(ref_tree), "commit", "-q", "-m",
                            f"ref_bin: the binary the card times ({ref['module']})"], check=True)
    return dst


def fill_ref_source_paths(card_dir: Path, paths: list[str], urls: list[str] = ()) -> None:
    """Write the declared plaintext locations into the card's loop.toml.example."""
    p = Path(card_dir) / "loop.toml.example"
    text = p.read_text()
    text = re.sub(r"(?m)^ref_source_paths\s*=.*$", f"ref_source_paths   = {json.dumps(list(paths))}", text)
    text = re.sub(r"(?m)^ref_source_urls\s*=.*$", f"ref_source_urls    = {json.dumps(list(urls))}", text)
    p.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m forgekernel.binref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("install", help="commit a built .so into the card's ref_bin/")
    ins.add_argument("card")
    ins.add_argument("so")
    ins.add_argument("--note", action="append", default=[], metavar="KEY=VALUE")
    ins.add_argument("--no-commit", action="store_true")
    ins.add_argument("--ref-tree", default=None,
                     help="the recon tree: commit the same binary there (no rebuild per job)")
    args = ap.parse_args(argv)
    if args.cmd == "install":
        notes = dict(n.split("=", 1) for n in args.note)
        dst = install(Path(args.card), Path(args.so), notes, commit=not args.no_commit,
                      ref_tree=Path(args.ref_tree) if args.ref_tree else None)
        print(f"[binref] installed {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
