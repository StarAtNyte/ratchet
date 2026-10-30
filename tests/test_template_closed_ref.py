"""``src/forgekernel/card_template/eval/closed_ref.py``: a single-file reference sealed into
an opaque payload and imported IN MEMORY on the node.

Copied verbatim into a card's eval/ and into the reference tree, so it is loaded
here from its path the way ``test_template_timing.py`` loads ``timing.py``.
"""

from __future__ import annotations

import importlib.util
import importlib
import inspect
import os
import re
import sys
from pathlib import Path

import pytest

CLOSED_REF_PY = (Path(__file__).resolve().parents[1]
                 / "src" / "forgekernel" / "card_template" / "eval" / "closed_ref.py")

REFERENCE_SOURCE = '''"""A pretend reference kernel: the thing an earlier run produced."""

SECRET_TILE_M = 128
SECRET_TILE_N = 64


def _ring_depth_for_the_exchange_face(stage_count):
    return stage_count * 2 + 1


def attn_fwd(q, k, v):
    """The reference entry: what the bench times next to the student."""
    return ("attn_fwd", q, k, v, _ring_depth_for_the_exchange_face(SECRET_TILE_N))
'''


@pytest.fixture
def closed_ref():
    spec = importlib.util.spec_from_file_location("_tmpl_closed_ref", CLOSED_REF_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    # modules the tests registered
    for name in list(sys.modules):
        if name.startswith("_closed_ref_test_"):
            sys.modules.pop(name, None)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not str(getattr(f, "pkg", "")).startswith("_closed_ref_test_")]


def _payload(closed_ref, tmp_path: Path, key: str | None) -> Path:
    path = tmp_path / "ref_bin" / "ref_payload.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(closed_ref.seal(REFERENCE_SOURCE, key))
    return path


# --- sealing ------------------------------------------------------------------

def test_plain_roundtrip(closed_ref):
    payload = closed_ref.seal(REFERENCE_SOURCE)
    assert payload.isascii() and b"\n" not in payload
    assert b"SECRET_TILE" not in payload
    assert closed_ref.unseal(payload) == REFERENCE_SOURCE


def test_keyed_roundtrip_needs_the_key(closed_ref):
    key = closed_ref.generate_key()
    payload = closed_ref.seal(REFERENCE_SOURCE, key)
    assert closed_ref.unseal(payload, key) == REFERENCE_SOURCE
    with pytest.raises(PermissionError):
        closed_ref.unseal(payload)                       # no key
    with pytest.raises(PermissionError):
        closed_ref.unseal(payload, closed_ref.generate_key())   # wrong key


def test_the_header_carries_the_plaintext_digest_without_the_key(closed_ref):
    import hashlib
    key = closed_ref.generate_key()
    payload = closed_ref.seal(REFERENCE_SOURCE, key)
    assert closed_ref.source_digest(payload) == hashlib.sha256(
        REFERENCE_SOURCE.encode()).hexdigest()
    # and two payloads of the same source under different keys differ as bytes
    assert closed_ref.payload_digest(payload) != closed_ref.payload_digest(
        closed_ref.seal(REFERENCE_SOURCE, closed_ref.generate_key()))


def test_garbage_is_refused_as_not_a_payload(closed_ref):
    with pytest.raises(ValueError):
        closed_ref.unseal(b"not a payload at all")
    with pytest.raises(ValueError):
        closed_ref.unseal(closed_ref.seal("x")[:10])


# --- loading in memory ------------------------------------------------------------

def test_load_materialises_the_module_with_no_file_on_disk(closed_ref, tmp_path):
    key = closed_ref.generate_key()
    path = _payload(closed_ref, tmp_path, key)
    module = closed_ref.load(str(path), name="_closed_ref_test_a", key_hex=key)
    assert module.attn_fwd(1, 2, 3)[0] == "attn_fwd"
    assert module.__file__ == closed_ref.FAKE_FILE
    assert not os.path.exists(closed_ref.FAKE_FILE)
    assert "_closed_ref_test_a" in sys.modules
    # nothing of the source landed under the payload's directory
    assert [p.name for p in path.parent.iterdir()] == ["ref_payload.bin"]


def test_the_dsl_route_to_source_works_through_linecache_inside_a_window_only(closed_ref, tmp_path):
    # CuTe DSL decorators call inspect.getsource on the reference's functions;
    # with no file on disk that resolves through the loader's get_source — while the
    # reference is loading or being called, and not after (a traceback or a warning
    # printed then would quote the source into the job log).
    import linecache
    path = _payload(closed_ref, tmp_path, None)
    module = closed_ref.load(str(path), name="_closed_ref_test_b", key_env=None)
    with closed_ref.window():
        src = inspect.getsource(module.attn_fwd)
    assert src.startswith("def attn_fwd(q, k, v):")
    assert closed_ref.FAKE_FILE not in linecache.cache
    with pytest.raises(OSError):
        inspect.getsource(module.attn_fwd)


def test_load_is_idempotent_per_name(closed_ref, tmp_path):
    path = _payload(closed_ref, tmp_path, None)
    a = closed_ref.load(str(path), name="_closed_ref_test_c", key_env=None)
    b = closed_ref.load(str(path), name="_closed_ref_test_c", key_env=None)
    assert a is b


def test_a_keyed_payload_without_its_env_var_says_which_variable(closed_ref, tmp_path,
                                                                monkeypatch):
    key = closed_ref.generate_key()
    path = _payload(closed_ref, tmp_path, key)
    monkeypatch.delenv("FK_REF_KEY", raising=False)
    with pytest.raises(PermissionError) as exc:
        closed_ref.load(str(path), name="_closed_ref_test_d")
    assert "FK_REF_KEY" in str(exc.value)
    monkeypatch.setenv("FK_REF_KEY", key)
    module = closed_ref.load(str(path), name="_closed_ref_test_d")
    assert module.SECRET_TILE_M == 128


def test_without_the_env_var_the_key_is_read_from_the_key_file_beside_the_payload(
        closed_ref, tmp_path, monkeypatch):
    """Where gpu_entry.sh moves the key (forgekernel.sealref.key_preamble)."""
    key = closed_ref.generate_key()
    path = _payload(closed_ref, tmp_path, key)
    monkeypatch.delenv("FK_REF_KEY", raising=False)
    assert closed_ref.key_file(str(path)) == str(tmp_path / "ref_bin" / "ref_payload.key")
    (tmp_path / "ref_bin" / "ref_payload.key").write_text(key + "\n")
    module = closed_ref.load(str(path), name="_closed_ref_test_kf")
    assert module.SECRET_TILE_M == 128


def test_entry_names_a_missing_entry(closed_ref, tmp_path):
    path = _payload(closed_ref, tmp_path, None)
    fn = closed_ref.entry("attn_fwd", str(path), name="_closed_ref_test_e", key_env=None)
    assert fn(0, 0, 0)[0] == "attn_fwd"
    with pytest.raises(AttributeError):
        closed_ref.entry("attn_bwd", str(path), name="_closed_ref_test_e", key_env=None)


def test_a_reference_that_fails_at_import_is_not_left_registered(closed_ref, tmp_path):
    path = tmp_path / "ref_payload.bin"
    path.write_bytes(closed_ref.seal("raise RuntimeError('no cuda here')\n"))
    with pytest.raises(RuntimeError):
        closed_ref.load(str(path), name="_closed_ref_test_f", key_env=None)
    assert "_closed_ref_test_f" not in sys.modules


def test_default_payload_location_prefers_what_exists(closed_ref, tmp_path):
    here = tmp_path / "eval"
    (here / "ref_bin").mkdir(parents=True)
    assert closed_ref.default_payload_path(str(here)) == str(here / "ref_bin" / "ref_payload.bin")
    (here / "ref_payload.bin").write_bytes(b"x")
    assert closed_ref.default_payload_path(str(here)) == str(here / "ref_payload.bin")


# --- the plaintext leak check ------------------------------------------------------

def test_check_tree_finds_an_exact_copy_without_the_key(closed_ref, tmp_path):
    key = closed_ref.generate_key()
    path = _payload(closed_ref, tmp_path, key)
    tree = tmp_path / "card"
    (tree / "eval").mkdir(parents=True)
    (tree / "eval" / "ref_kernel.py").write_text(REFERENCE_SOURCE)
    (tree / "kernel.py").write_text("def attn_fwd(q, k, v):\n    return q\n")
    hits = closed_ref.check_tree(str(tree), path.read_bytes())
    assert len(hits) == 1 and "ref_kernel.py" in hits[0] and "exact copy" in hits[0]


def test_check_tree_finds_an_edited_copy_only_with_the_key(closed_ref, tmp_path):
    key = closed_ref.generate_key()
    path = _payload(closed_ref, tmp_path, key)
    tree = tmp_path / "card"
    tree.mkdir()
    edited = REFERENCE_SOURCE.replace("SECRET_TILE_M = 128", "SECRET_TILE_M = 256") + "\n# mine\n"
    (tree / "kernel.py").write_text(edited)
    assert closed_ref.check_tree(str(tree), path.read_bytes()) == []
    hits = closed_ref.check_tree(str(tree), path.read_bytes(), key)
    assert len(hits) == 1 and "kernel.py" in hits[0] and "distinctive lines" in hits[0]


def test_check_tree_passes_a_clean_tree_and_skips_venv_and_git(closed_ref, tmp_path):
    path = _payload(closed_ref, tmp_path, None)
    tree = tmp_path / "card"
    (tree / ".venv" / "lib").mkdir(parents=True)
    (tree / ".venv" / "lib" / "copy.py").write_text(REFERENCE_SOURCE)
    (tree / "eval").mkdir()
    (tree / "eval" / "bench.py").write_text(
        "from closed_ref import entry\nref = entry('attn_fwd')\n")
    (tree / "kernel.py").write_text("def attn_fwd(q, k, v):\n    return q + k + v\n")
    assert closed_ref.check_tree(str(tree), path.read_bytes()) == []


# --- command line ------------------------------------------------------------------

def test_cli_seal_gen_key_digest_and_check_tree(closed_ref, tmp_path, capsys):
    src = tmp_path / "kernel.py"
    src.write_text(REFERENCE_SOURCE)
    out = tmp_path / "card" / "eval" / "ref_bin" / "ref_payload.bin"
    assert closed_ref.main(["seal", str(src), "-o", str(out), "--gen-key"]) == 0
    printed = capsys.readouterr().out
    key = printed.rsplit(": ", 1)[1].strip()
    assert len(key) == 64 and "keyed" in printed
    assert closed_ref.unseal(out.read_bytes(), key) == REFERENCE_SOURCE

    assert closed_ref.main(["digest", str(out)]) == 0
    assert "keyed" in capsys.readouterr().out

    card = tmp_path / "card"
    (card / "kernel.py").write_text("def attn_fwd(q, k, v):\n    return q\n")
    assert closed_ref.main(["check-tree", str(card), str(out), "--key", key]) == 0
    assert "clean" in capsys.readouterr().out

    (card / "notes.py").write_text(REFERENCE_SOURCE)
    assert closed_ref.main(["check-tree", str(card), str(out), "--key", key]) == 3
    assert "LEAK" in capsys.readouterr().out


def test_cli_seal_refuses_to_guess_the_key_mode(closed_ref, tmp_path):
    src = tmp_path / "kernel.py"
    src.write_text(REFERENCE_SOURCE)
    with pytest.raises(SystemExit):
        closed_ref.main(["seal", str(src), "-o", str(tmp_path / "p.bin")])
    assert closed_ref.main(["seal", str(src), "-o", str(tmp_path / "p.bin"), "--plain"]) == 0
    assert closed_ref.unseal((tmp_path / "p.bin").read_bytes()) == REFERENCE_SOURCE


# --- the source window and sanitized errors ------------------------------------------

RAISING_SOURCE = '''
import warnings


def run(x):
    if x < 0:
        raise ValueError("SECRET_TILE_M = 128 would be quoted here")
    warnings.warn("SECRET warning text")
    return x + 1
'''


def test_an_exception_from_the_reference_is_sanitized(closed_ref, tmp_path, capsys):
    import traceback
    path = tmp_path / "ref_payload.bin"
    path.write_bytes(closed_ref.seal(RAISING_SOURCE))
    fn = closed_ref.entry("run", str(path), name="_closed_ref_test_g", key_env=None)
    assert fn(1) == 2                                   # the warning is swallowed, not printed
    assert "SECRET" not in capsys.readouterr().err
    with pytest.raises(closed_ref.SealedReferenceError) as exc:
        fn(-1)
    text = "".join(traceback.format_exception(exc.value))
    assert "ValueError" in text and "reference:7" in text
    assert "SECRET" not in text and "raise ValueError" not in text
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_a_reference_failing_at_import_says_where_not_what(closed_ref, tmp_path):
    path = tmp_path / "ref_payload.bin"
    path.write_bytes(closed_ref.seal("SECRET = 1\nraise KeyError('SECRET text')\n"))
    with pytest.raises(closed_ref.SealedReferenceError) as exc:
        closed_ref.load(str(path), name="_closed_ref_test_h", key_env=None)
    assert "KeyError" in str(exc.value) and "reference:2" in str(exc.value)
    assert "SECRET" not in str(exc.value)


# --- bundles (a multi-file solution) ------------------------------------------------

BUNDLE_FILES = {
    "kernel.py": (
        "import os\nimport sys\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendor'))\n"
        "from helpers import scale_by_the_secret_factor\n"
        "from .sub.deep import DEEP_CONSTANT_OF_THE_REFERENCE\n"
        "import vlib.inner as vendored_inner\n\n\n"
        "def run(x):\n"
        "    from late_import_target import LATE\n"
        "    return scale_by_the_secret_factor(x) + DEEP_CONSTANT_OF_THE_REFERENCE + vendored_inner.V + LATE\n"),
    "helpers.py": "def scale_by_the_secret_factor(x):\n    return 2 * x\n",
    "late_import_target.py": "LATE = 1000\n",
    "sub/__init__.py": "",
    "sub/deep.py": "from ..helpers import scale_by_the_secret_factor\n"
                   "DEEP_CONSTANT_OF_THE_REFERENCE = scale_by_the_secret_factor(50)\n",
    "vendor/vlib/__init__.py": "from .inner import V\n",
    "vendor/vlib/inner.py": "V = 7\n",
    "README.md": "not a module\n",
}
BUNDLE_SOLUTION = {"name": "s", "definition": "d", "spec": {"entry_point": "kernel.py::run"},
                   "sources": [{"path": p, "content": t} for p, t in BUNDLE_FILES.items()]}


def _bundle(closed_ref, tmp_path: Path, key: str | None) -> Path:
    path = tmp_path / "ref_bin" / "ref_payload.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(closed_ref.seal_bundle(closed_ref.bundle_from_solution(BUNDLE_SOLUTION), key))
    return path


def test_a_bundle_roundtrips_and_carries_its_files_digests(closed_ref):
    import hashlib
    import json
    key = closed_ref.generate_key()
    manifest = closed_ref.bundle_from_solution(BUNDLE_SOLUTION)
    payload = closed_ref.seal_bundle(manifest, key)
    assert closed_ref.is_bundle(payload) and b"scale_by" not in payload
    assert closed_ref.unseal_bundle(payload, key) == manifest
    with pytest.raises(PermissionError):
        closed_ref.unseal_bundle(payload)
    assert closed_ref._parse(payload)[3] == [
        hashlib.sha256(t.encode()).digest() for _, t in sorted(BUNDLE_FILES.items())
        if len(t.encode()) >= closed_ref.MIN_DIGEST_BYTES]
    assert closed_ref.source_digest(payload) == hashlib.sha256(closed_ref._canonical(manifest)).hexdigest()
    assert json.loads(closed_ref.unseal(payload, key))["entry"] == "kernel.py::run"


@pytest.mark.parametrize("files, entry, why", [
    ({"../x.py": "x"}, "x.py::run", "inside the bundle"),
    ({"/abs.py": "x"}, "abs.py::run", "inside the bundle"),
    ({"my-kernel.py": "x"}, "my-kernel.py::run", "importable"),
    ({"k.py": "x"}, "k.py", "<file>.py::<function>"),
    ({"k.py": "x"}, "other.py::run", "not among"),
])
def test_bad_bundles_are_refused(closed_ref, files, entry, why):
    with pytest.raises(ValueError, match=re.escape(why)):
        closed_ref.bundle_manifest(files, entry)


def test_a_bundle_loads_as_a_private_in_memory_package(closed_ref, tmp_path):
    key = closed_ref.generate_key()
    path = _bundle(closed_ref, tmp_path, key)
    fn = closed_ref.entry(None, str(path), name="_closed_ref_test_pkg", key_hex=key)
    # relative, absolute-own-file, vendored-on-sys.path and call-time imports all resolve
    assert fn(1) == 2 + 100 + 7 + 1000
    mod = closed_ref.load(str(path), name="_closed_ref_test_pkg", key_hex=key)
    assert mod.__name__ == "_closed_ref_test_pkg.kernel"
    assert mod.__file__ == f"{closed_ref.FAKE_ROOT}/_closed_ref_test_pkg/kernel.py"
    assert "_closed_ref_test_pkg.vendor.vlib.inner" in sys.modules
    # nothing under the solution's own top-level names, nothing on disk
    for top in ("helpers", "kernel", "vlib", "late_import_target", "sub"):
        assert not str(getattr(sys.modules.get(top), "__file__", "")).startswith(closed_ref.FAKE_ROOT)
    with pytest.raises(ImportError):
        importlib.import_module("late_import_target")
    assert not os.path.exists(closed_ref.FAKE_ROOT)
    assert [p.name for p in path.parent.iterdir()] == ["ref_payload.bin"]
    # the source is there for a JIT inside a window, and gone after it
    with closed_ref.window():
        assert "def run(x):" in inspect.getsource(mod.run)
        helpers = sys.modules["_closed_ref_test_pkg.helpers"]
        assert inspect.getsource(helpers.scale_by_the_secret_factor).startswith("def scale_by")
    with pytest.raises(OSError):
        inspect.getsource(mod.run)
    # a second load by name (another copy of the loader, same process) finds the entry file
    spec = importlib.util.spec_from_file_location("_tmpl_closed_ref_2", CLOSED_REF_PY)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    assert other.load(str(path), name="_closed_ref_test_pkg") is mod


def test_a_reference_call_can_import_its_flat_siblings_through_importlib(closed_ref, tmp_path):
    """CuTe DSL's preprocessor replays a kernel module's imports with plain importlib while
    it compiles; a bundle whose files import each other by flat name (KDA's gdn_prefill
    package) must resolve them then -- and only then."""
    files = {"entry.py": ("import importlib\n"
                          "from sibling_of_the_ref import FACTOR\n\n\n"
                          "def run(x):\n"
                          "    replayed = importlib.import_module('sibling_of_the_ref')\n"
                          "    return x * FACTOR + replayed.FACTOR\n"),
             "sibling_of_the_ref.py": "FACTOR = 3\n"}
    manifest = closed_ref.bundle_manifest(files, "entry.py::run")
    path = tmp_path / "ref_payload.bin"
    path.write_bytes(closed_ref.seal_bundle(manifest))
    fn = closed_ref.entry(None, str(path), name="_closed_ref_test_flat", key_env=None)
    assert fn(2) == 2 * 3 + 3
    # outside the call the flat name is gone again: the student's kernel cannot import it
    assert "sibling_of_the_ref" not in sys.modules
    with pytest.raises(ImportError):
        importlib.import_module("sibling_of_the_ref")


def test_a_bundle_that_fails_to_import_leaves_nothing_registered(closed_ref, tmp_path):
    manifest = closed_ref.bundle_manifest({"k.py": "from missing_mod_xyz import thing\n"}, "k.py::run")
    path = tmp_path / "ref_payload.bin"
    path.write_bytes(closed_ref.seal_bundle(manifest))
    before = list(sys.meta_path)
    with pytest.raises(closed_ref.SealedReferenceError, match="missing_mod_xyz"):
        closed_ref.load(str(path), name="_closed_ref_test_bad", key_env=None)
    assert not [m for m in sys.modules if m.startswith("_closed_ref_test_bad")]
    assert sys.meta_path == before


def test_check_tree_finds_a_bundle_file_in_plaintext(closed_ref, tmp_path):
    key = closed_ref.generate_key()
    path = _bundle(closed_ref, tmp_path, key)
    tree = tmp_path / "ws"
    tree.mkdir()
    (tree / "copied.py").write_text(BUNDLE_FILES["kernel.py"])
    hits = closed_ref.check_tree(str(tree), path.read_bytes())
    assert len(hits) == 1 and "exact copy" in hits[0]
    (tree / "copied.py").write_text(BUNDLE_FILES["kernel.py"].replace("1000", "1") + "# mine\n")
    assert closed_ref.check_tree(str(tree), path.read_bytes()) == []
    hits = closed_ref.check_tree(str(tree), path.read_bytes(), key)
    assert len(hits) == 1 and "kernel.py's distinctive lines" in hits[0]


def test_cli_seals_a_solution_json_and_a_directory(closed_ref, tmp_path, capsys):
    import json
    sol = tmp_path / "solution.json"
    sol.write_text(json.dumps(BUNDLE_SOLUTION))
    out = tmp_path / "p.bin"
    assert closed_ref.main(["seal", str(sol), "-o", str(out), "--plain"]) == 0
    assert "bundle of 8 file(s), entry kernel.py::run" in capsys.readouterr().out
    assert closed_ref.unseal_bundle(out.read_bytes())["files"] == dict(sorted(BUNDLE_FILES.items()))
    d = tmp_path / "soldir"
    for p, t in BUNDLE_FILES.items():
        (d / p).parent.mkdir(parents=True, exist_ok=True)
        (d / p).write_text(t)
    with pytest.raises(SystemExit):
        closed_ref.main(["seal", str(d), "-o", str(out), "--plain"])        # no --entry
    assert closed_ref.main(["seal", str(d), "-o", str(out), "--plain", "--entry", "kernel.py::run"]) == 0
    assert closed_ref.unseal_bundle(out.read_bytes()) == closed_ref.bundle_from_solution(BUNDLE_SOLUTION)
    assert closed_ref.main(["digest", str(out)]) == 0
    assert "bundle" in capsys.readouterr().out
