"""Sealed SOURCE reference for a leaderboard card: a Python solution (PyTorch, Triton,
CuTe DSL; one file or many) the agents can run, time and profile but never read.

The binary route (forgekernel.binref) needs a compiled solution. A Python solution has
no binary to ship, so it ships as source the agents cannot decode: every file of the
solution sealed into one keyed payload (src/forgekernel/card_template/eval/closed_ref.py, bundle
format), committed into the card and its recon tree; the key sits in the gitignored
``submit.local.toml`` ``[env]`` of both trees (denied to the student and a blackbox
teacher by the agent policy) and reaches every job as ``FK_REF_KEY``, which the tree's
gpu_entry.sh moves at once into ``ref_bin/ref_payload.key`` and out of the environment
every later process would inherit (key_preamble); inside the job the card's judge
decodes the payload IN MEMORY into a private package and times it beside the kernel:

    forge init <op> --leaderboard sol-execbench ... --sealed-ref solution.json \\
        [--ref-source-path PATH ...]
        -> card: problem.json reference_impl kind "sealed" (ref_bin/ref_payload.bin, timed
           in-job beside the kernel), eval/closed_ref.py, the key in submit.local.toml, a
           fence on kernel.py (loader, payload, key, environment, reflection hooks, JIT
           caches), a guarded `cmd` task, and a blackbox loop.toml.example;
        -> recon tree operators/_<op>ref/: the same payload and key, sealref.json,
           sealref.py (check / crosscheck / bench / ncu), a copy of the card's eval/, and
           a gpu_entry.sh whose `cmd` task is guarded.

There is nothing to build: both trees run as soon as they are committed. No plaintext of
the reference is written into either tree (``forge new`` checks both with
``closed_ref.check_tree`` before it returns); where the solution itself lives goes into
loop.toml ``ref_source_paths``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

from forgekernel import binref

TOOL = Path(__file__).resolve().parent / "sealref_tool.py"
KEY_ENV = "FK_REF_KEY"
PAYLOAD = "ref_bin/ref_payload.bin"
REF_TASK = "sealref"

# What a `cmd` job (or a notes/ script, or the card's kernel.py) may not name: the payload,
# its loader and key, the in-memory package, source introspection, and the places a JIT
# leaves the reference's IR (Triton's cache, the DSLs' dump switches). Scanned in notes/
# and kernel.py too, so none of it may be a word the loop's own seeded toolkits use.
REACH = [r"ref_bin", r"ref_payload", r"closed_ref", r"_REF_KEY", r"\blinecache\b", r"getsource",
         r"/nonexistent", r"unseal", r"keystream", r"__closed_ref", r"meta_path",
         r"\.triton\b", r"triton/cache", r"TRITON_CACHE", r"TRITON_DUMP", r"TRITON_KERNEL_DUMP",
         r"MLIR_ENABLE_DUMP", r"\.ttg?ir\b", r"\.llir\b",
         r"/proc/[^ ]*environ", r"printenv"]
# The card's `cmd` words, on top: binary tooling (a JIT'd reference's cubins sit in the JIT
# cache), the CuTe DSL's IR dump switches (the loop's seeded pricing tools set them for the
# student's own kernel from inside notes/, which is why they are words here and not REACH),
# the environment the key arrives in (gpu_entry.sh moves it out at once: KEY_PREAMBLE) and
# other processes' (/proc), and the interpreter's reflection hooks.
CARD_WORDS = [r"cuobjdump", r"nvdisasm", r"objdump", r"readelf", r"\bstrings\b", r"\bnm\b",
              r"CUTE_DSL_KEEP", r"CUTE_DSL_DUMP", r"/proc/", r"getenv",
              r"\benviron\b", r"ENVIRON", r"\bENV\b", r"\benv\b", r"compgen", r"declare +-[a-zA-Z]*p",
              r"export +-p", r'(^|[^-_[:alnum:]])set *($|[;&|)"])', r"\.ptx\b", r"\.cubin\b",
              r"sys\.modules", r"__globals__", r"f_globals", r"settrace", r"setprofile"]
# The recon tree's `cmd` words (the teacher's probes): the card's minus the SASS tools —
# SASS is black-box reconnaissance there (`ncu --print-source sass`, cuobjdump -sass on a
# cubin the probe produced) — plus line-resolved source and PTX.
REF_TREE_WORDS = [w for w in CARD_WORDS if w not in (r"cuobjdump", r"nvdisasm", r"\.cubin\b")] + [
    r"inspect\.getfile", r"find_spec", r"pip +(show|download)",
    r"--print-source[= ]*(cuda|all|src|ptx|python)", r"--import-source", r"--source-folders",
    r"-ptx\b", r"--dump-ptx", r"/Users/", r"find +/"]

KEY_FILE = PAYLOAD.rsplit(".", 1)[0] + ".key"   # = closed_ref.key_file(PAYLOAD)


def module_words(names) -> list[str]:
    """The guard patterns for a task's ``deny_modules``: each name as a whole word, so
    ``reflib`` refuses ``import reflib.ops`` and ``.../site-packages/reflib/...`` but not
    ``reflib_bwd``. Extended-regex syntax, read alike by ``grep -E`` and Python's ``re``
    (the names are restricted to letters, digits, ``_``, ``.`` and ``-``)."""
    return [r"(^|[^A-Za-z0-9_])" + n.replace(".", r"\.") + r"([^A-Za-z0-9_]|$)" for n in names]


def key_preamble(key_env: str = KEY_ENV) -> str:
    """gpu_entry.sh's first act in a tree that loads the payload. The key reaches the job in
    its environment (submit.local.toml [env]), and every process of the job would inherit it
    from there — a `cmd` probe, the student's kernel beside the reference, a profiler's
    report of the launch environment, /proc/<pid>/environ of this very script for as long
    as the job runs. So it goes into the key file beside the payload (mode 600; closed_ref
    reads it there) and the script re-executes itself without it: from then on no
    environment on the node carries the key."""
    key_dir = KEY_FILE.rsplit("/", 1)[0]
    return (
        "\n# The sealed reference's key (forgekernel.sealref): out of the job environment before\n"
        f"# anything runs — into {KEY_FILE} (mode 600, gitignored; eval/closed_ref.py reads it\n"
        f"# there), then this script re-executes itself without {key_env}.\n"
        f'if [ -n "${{{key_env}:-}}" ]; then\n'
        '    _fk_tree="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'    mkdir -p "$_fk_tree/{key_dir}"\n'
        f"    (umask 077 && printf '%s\\n' \"${key_env}\" > \"$_fk_tree/{KEY_FILE}\")\n"
        f'    exec env -u {key_env} bash "${{BASH_SOURCE[0]}}" "$@"\n'
        "fi\n")


def _with_key_preamble(text: str, key_env: str) -> str:
    if "# The sealed reference's key" in text:
        return text
    anchor = "set -euo pipefail\n"
    if anchor not in text:
        raise ValueError("gpu_entry.sh: no `set -euo pipefail` line to put the key preamble after")
    return text.replace(anchor, anchor + key_preamble(key_env), 1)


def load_solution(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def module_name(sol: dict) -> str:
    h = hashlib.sha256(json.dumps(sol.get("sources"), sort_keys=True).encode()).hexdigest()
    return f"sealref_{h[:10]}"


def closed_ref(template_eval: Path):
    """The template's closed_ref.py (the same file the card and the recon tree get)."""
    spec = importlib.util.spec_from_file_location("_fk_sealref_closed_ref",
                                                  Path(template_eval) / "closed_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solution_problems(sol: dict, definition: str, template_eval: Path) -> list[str]:
    """Why this solution cannot be a card's sealed source reference (empty = it can)."""
    errs = []
    spec = sol.get("spec") or {}
    langs = {str(x).lower() for x in spec.get("languages") or []}
    if langs & binref.CPP_LANGUAGES:
        errs.append(f"languages {sorted(langs)}: a compiled solution is sealed as a binary "
                    f"(--binary-ref), not as source")
    if sol.get("definition") != definition:
        errs.append(f"solution is for {sol.get('definition')!r}, the card for {definition!r}")
    try:
        closed_ref(template_eval).bundle_from_solution(sol)
    except (ValueError, KeyError) as exc:
        errs.append(str(exc))
    return errs


def non_python_sources(sol: dict) -> list[str]:
    """Files that travel in the payload but are not materialised on the node."""
    return [s["path"] for s in sol.get("sources") or [] if not str(s["path"]).endswith(".py")]


def reference_impl(sol: dict, module: str, name: str | None = None) -> dict:
    spec = sol["spec"]
    return {"kind": "sealed", "name": name or f"sealed source reference {module}",
            "path": f"../../{PAYLOAD}", "module": module,
            "entry": spec["entry_point"].split("::", 1)[1],
            "dps": bool(spec.get("destination_passing_style", True)), "key_env": KEY_ENV,
            "languages": list(spec.get("languages") or [])}


def local_submit_toml(key: str) -> str:
    return f"""# Gitignored (*.local.toml). The key of the sealed reference in {PAYLOAD}
# (forgekernel.sealref): every job of this tree gets it as {KEY_ENV}, and the card's judge
# decodes the reference with it in memory. The agent policy denies this file to the student
# and a blackbox teacher; never commit it, never copy it outside this run. Other local
# settings (a clone delivery's git_url) go in their own tables beside this one.
[env]
{KEY_ENV} = "{key}"
"""


def write_key(tree: Path, key: str) -> Path:
    p = Path(tree) / "submit.local.toml"
    if p.exists():
        raise FileExistsError(f"{p} exists: add [env] {KEY_ENV} to it by hand")
    p.write_text(local_submit_toml(key))
    return p


def guard_card_gpu_entry(text: str, module: str, key_env: str = KEY_ENV,
                         kernel_file: str = "kernel.py", deny_modules=()) -> str:
    """The card's gpu_entry.sh with its `cmd` task guarded against the sealed reference. The
    key file sits in the card for the whole job, and a C/C++ probe (or a CUDA kernel file a
    `cmd` compiles) opens a file as readily as a Python one: notes/ C/C++ files and the
    card's kernel file are scanned like notes/*.py."""
    if "blackbox_guard" in text:
        return text
    before, dispatch, after = binref._split_dispatch(text)
    dispatch = dispatch.replace('    cmd)         run_with_timeout bash -c "$*" ;;',
                                '    cmd)         blackbox_guard "$*"; run_with_timeout bash -c "$*" ;;')
    if "blackbox_guard" not in dispatch:
        raise ValueError("card gpu_entry.sh: no `cmd` task to guard")
    reach = REACH + [re.escape(module), re.escape(key_env)] + module_words(deny_modules)
    return (_with_key_preamble(before, key_env)
            + binref._guard_block(reach + CARD_WORDS, reach, "card, sealed source reference",
                                  extra_files=tuple(dict.fromkeys(
                                      ("kernel.py", *binref.CARD_CPP_NOTES, kernel_file))))
            + "\n" + dispatch + after)


def ref_tree_gpu_entry(card_text: str, module: str, key_env: str = KEY_ENV,
                      deny_modules=()) -> str:
    """The recon tree's gpu_entry.sh: the card's environment and clock lock, the sealref
    tasks, and a guarded `cmd`."""
    card_text = re.sub(r"\n# Blackbox guard \(.*?\n}\n", "\n", card_text, flags=re.S)  # the card's own
    before, _, after = binref._split_dispatch(card_text)
    head = ("#!/bin/bash\n# Recon tree of a sealed source reference (forgekernel.sealref) — tasks:\n"
            "#   check | crosscheck | bench (= ref) | ncu | sealref (check + bench + ncu)"
            " | cmd <CMD> (guarded)\n")
    before = _with_key_preamble(head + before.split("\n", 1)[1], key_env)
    ncu = ('ncu_bounded ncu --csv --target-processes all --profile-from-start off \\\n'
           '            --launch-count 24 --kill yes --metrics "$NCU_METRICS" python3 eval/ncu_run.py')
    tasks = f'''case "$TASK" in
    check)       run_with_timeout python3 sealref.py check ;;
    crosscheck)  run_with_timeout python3 sealref.py crosscheck ;;
    bench|ref)   run_with_timeout python3 sealref.py bench ;;
    ncu)
        {ncu}
        ;;
    sealref)
        run_with_timeout python3 sealref.py check
        run_with_timeout python3 sealref.py bench
        {ncu}
        ;;
    cmd)         blackbox_guard "$*"; run_with_timeout bash -c "$*" ;;
    help|*)
        sed -n '2,4p' "$0"
        exit 1
        ;;
esac
'''
    reach = REACH + [re.escape(module), re.escape(key_env)] + module_words(deny_modules)
    return (before + binref._guard_block(reach + REF_TREE_WORDS, reach, "recon tree, sealed source reference",
                                         skip_notes=(binref.SEEDED_GUARD,),
                                         extra_files=binref.CARD_CPP_NOTES)
            + "\n" + tasks + after)


REF_NCU_RUN = binref.REF_NCU_RUN
REF_GITIGNORE = f"*.local.toml\n{KEY_FILE}\n__pycache__/\n*.pyc\nnotes/*.log\n.jobs/\n.exec/\n"


def write_trees(card_dir: Path, ref_dir: Path, sol: dict, module: str, template_eval: Path,
                key: str | None = None, kernel_file: str = "kernel.py",
                deny_modules=()) -> tuple[list[Path], str]:
    """Seal the solution into the card (payload, loader, key) and write the recon tree next
    to it. Returns (created paths, the key). ``deny_modules``: the reference's package as
    installed on the GPU machine, refused to both trees' `cmd` like the payload is."""
    card_dir, ref_dir = Path(card_dir), Path(ref_dir)
    if ref_dir.exists():
        raise FileExistsError(f"{ref_dir} exists")
    cr = closed_ref(template_eval)
    key = key or cr.generate_key()
    manifest = cr.bundle_from_solution(sol)
    payload = cr.seal_bundle(manifest, key)
    created: list[Path] = []

    def put(tree: Path, rel: str, data: str | bytes, mode: int | None = None) -> None:
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data, encoding="utf-8")
        if mode:
            p.chmod(mode)
        created.append(p)

    # the card
    put(card_dir, PAYLOAD, payload)
    shutil.copy2(Path(template_eval) / "closed_ref.py", card_dir / "eval" / "closed_ref.py")
    created.append(card_dir / "eval" / "closed_ref.py")
    created.append(write_key(card_dir, key))
    entry = card_dir / "gpu_entry.sh"
    entry.write_text(guard_card_gpu_entry(entry.read_text(), module, kernel_file=kernel_file,
                                          deny_modules=deny_modules))
    gi = card_dir / ".gitignore"
    text = gi.read_text() if gi.exists() else ""
    if KEY_FILE not in text.split():
        gi.write_text(text + ("\n" if text and not text.endswith("\n") else "") + KEY_FILE + "\n")

    # the recon tree
    ref_dir.mkdir(parents=True)
    spec = sol.get("spec") or {}
    put(ref_dir, PAYLOAD, payload)
    put(ref_dir, "sealref.json", json.dumps({
        "kind": "sealed", "module": module, "definition": sol.get("definition"),
        "solution": sol.get("name"), "entry": spec["entry_point"].split("::", 1)[1],
        "languages": spec.get("languages"), "dps": bool(spec.get("destination_passing_style", True)),
        "key_env": KEY_ENV, "files": len(manifest["files"]),
        "source_sha256": cr.source_digest(payload), "payload_sha256": cr.payload_digest(payload)},
        indent=2) + "\n")
    shutil.copy2(TOOL, ref_dir / "sealref.py")
    created.append(ref_dir / "sealref.py")
    shutil.copytree(card_dir / "eval", ref_dir / "eval",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "ncu_run.py"))
    created.append(ref_dir / "eval")
    put(ref_dir, "eval/ncu_run.py", REF_NCU_RUN)
    put(ref_dir, "gpu_entry.sh", ref_tree_gpu_entry(entry.read_text(), module,
                                                    deny_modules=deny_modules), 0o755)
    if (card_dir / "submit.toml").exists():
        text = (card_dir / "submit.toml").read_text()
        put(ref_dir, "submit.toml", re.sub(r"(?m)^timeout\s*=\s*\d+", "timeout = 2700", text))
    created.append(write_key(ref_dir, key))
    put(ref_dir, ".gitignore", REF_GITIGNORE)
    # No notes/ here: the loop seeds it at reconnaissance (see binref.write_ref_tree).
    put(ref_dir, "README.md", f"""# _{card_dir.name}ref — recon tree of a sealed source reference

The card `{card_dir.name}` times its kernel against `{PAYLOAD}`: the reference's sources,
sealed with a key that lives only in the gitignored `submit.local.toml` of the card and of
this tree, decoded in memory inside the job (`eval/closed_ref.py`). No plaintext is here.

    bash gpu_entry.sh sealref     check + bench + NCU in one job ([loop] ref_mode)
    bash gpu_entry.sh crosscheck  the card's judge on the reference: must PASS
    bash gpu_entry.sh ncu         NCU of the reference (`--print-source sass` is open)
""")
    for tree in (card_dir, ref_dir):
        if hits := cr.check_tree(str(tree), payload, key):
            raise RuntimeError(f"{tree} holds the sealed reference in plaintext: {hits[:3]}")
    return created, key
