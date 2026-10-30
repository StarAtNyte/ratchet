"""A CLOSED reference (one file, or a whole multi-file solution) for in-job pairing.
Copy into <op>/eval/ (and the reference tree), do not edit. Standard library only: it
runs inside the job.

WHAT IT IS FOR. Most closed-source runs pair against a LIBRARY the job image
already ships (the bench imports the library's entry and times it next to the
kernel). A self-evolution run pairs against an earlier run's own ``kernel.py`` —
a source file. Committing that file into the operator card makes it readable to
the student, which is the one thing a closed-source run forbids; timing it in a separate
tree instead loses the same-job ratio and puts a tight target back inside the
node-to-node noise. This module is the third way: the source travels as an
opaque payload and becomes an importable module IN MEMORY only, on the node.

    seal   (operator, on the Mac)   kernel.py  ->  eval/ref_bin/ref_payload.bin
    load   (bench, in the job)      payload    ->  module object; call its entry

Nothing is ever written to disk on the node: the module is registered with a
loader object that serves the decoded source to ``linecache`` — which is how a
CuTe DSL decorator's (or Triton's ``@jit``) ``inspect.getsource`` finds it — and to
nothing else.

PAYLOAD FORMAT (one ASCII line, base85). A single file::

    FKCR1 + mode + sha256(source) + body
    mode "p"  plain:  body = zlib(source)
    mode "k"  keyed:  body = zlib(source) XOR keystream(key)

A BUNDLE — every source file of a solution (a FlashInfer-Bench / SOL-ExecBench
solution JSON's ``sources``, or a directory) and its ``<file>.py::<function>`` entry::

    FKCR2 + mode + sha256(manifest) + u16 n + n x sha256(file) + body
    manifest = canonical JSON {"entry": "kernel.py::run", "files": {path: text}}
    body     = zlib(manifest) [XOR keystream(key)], as above

The per-file digests cover the files long enough to be recognised (``check-tree``
without the key); short ones (an empty ``__init__.py``) would match every tree.

A bundle loads as a PACKAGE that lives in memory only: file ``a/b.py`` becomes module
``<name>.a.b`` with ``__file__`` ``/nonexistent/closed_reference/<name>/a/b.py`` (never
created). Relative imports resolve inside the package; so do the absolute imports a
solution makes of its own files (``from helpers import x``, or a subdirectory it puts
on ``sys.path`` from its ``__file__``, the way SOL-ExecBench runs a solution from its
directory): each bundle module imports through its own ``__import__``, which maps a
top-level name the bundle provides to the in-memory module and leaves every other
name (torch, triton, cutlass, ...) to the real import system. So nothing of the
reference is importable under its own top-level names — a student kernel named
``kernel`` and a reference file ``kernel.py`` do not collide, and ``import helpers``
from the student finds nothing. ``importlib.import_module`` by a bundle-relative name
is not mapped (the solution's ``import`` statements are), and files other than ``.py``
travel in the manifest but are not materialised (a solution that reads a data file
next to its ``__file__`` cannot be sealed this way).

THE SOURCE IS SERVED IN WINDOWS ONLY. ``linecache`` is also how a traceback or a
warning prints the offending line — so a reference that raises inside a job would
print its own source into the job log the student reads. The loader therefore serves
the source only while the reference is being imported or called through ``entry()``
(or inside an explicit ``window()``), purges what ``linecache`` holds when the window
closes, silences warnings inside it, and turns any exception that escapes the
reference into ``SealedReferenceError`` naming only the exception type and the
reference-relative ``file:line`` (an exception message can quote the source — a
Triton ``CompilationError`` does). Call the reference through ``entry()``; a function
taken off ``load()``'s module and called outside a window runs, but a JIT that needs
the source at its first call (CuTe DSL) will not find it there.

The keyed mode is the one to use on a run: the key lives in the gitignored
``submit.local.toml`` ``[env]`` block of every tree whose jobs load the payload
(the operator card, the reference tree) and reaches the job as an environment
variable (``FK_REF_KEY`` unless the bench names another). That file sits INSIDE
the run's tree, so the cipher protects nothing by itself: the harness's agent policy
denies it (Read rules + Bash sandbox) to the student and to a closed-source run's
teacher, the same way it denies the reference checkouts — that denial is what
keeps the source from the Mac-side agents, and a card that loads the key from
anywhere else has no such rule. On the node the key must not STAY in the job
environment, where every process of the job inherits it (a `cmd` probe, the
student's kernel, a profiler's report of the launch environment, ``/proc/*/environ``):
the tree's gpu_entry.sh moves it at once into ``<payload stem>.key`` next to the
payload (``ref_bin/ref_payload.key``, mode 600, never committed) and re-executes
itself without it (``forgekernel.sealref.KEY_PREAMBLE``), and ``load()`` reads it
there when the variable is absent. On the node the payload is decoded inside the
bench process; the boundary there is the process plus the tripwire below. The
plaintext digest in the header lets ``check-tree`` recognise a plaintext copy
of the reference in a tree WITHOUT the key.

WHAT THE STUDENT MAY DO WITH IT: call it, time it, profile it (NCU without
line-resolved source), disassemble its SASS. Decoding the payload for any other
purpose, printing the module's source, or reaching into the loader object is
off limits; the shared ``blackbox-guard.sh`` tripwire refuses commands that name
this module, the payload, or the decoding primitives, and the bench's own fence
must list ``closed_ref`` among the names the kernel may not import.

COMMAND LINE::

    python eval/closed_ref.py seal path/to/kernel.py -o eval/ref_bin/ref_payload.bin --gen-key
    python eval/closed_ref.py seal path/to/kernel.py -o ... --key <hex>        # same key, other tree
    python eval/closed_ref.py seal solution.json -o ... --gen-key              # a bundle (its sources)
    python eval/closed_ref.py seal path/to/dir -o ... --entry k.py::run --gen-key   # a bundle
    python eval/closed_ref.py digest eval/ref_bin/ref_payload.bin
    python eval/closed_ref.py check-tree <operator card or workspace> eval/ref_bin/ref_payload.bin
"""

from __future__ import annotations

import base64
import builtins
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
import os
import sys
import threading
import warnings
import zlib

MAGIC = b"FKCR1"
MAGIC_BUNDLE = b"FKCR2"
MODE_PLAIN = b"p"
MODE_KEYED = b"k"
DEFAULT_KEY_ENV = "FK_REF_KEY"
DEFAULT_MODULE_NAME = "_closed_ref_module"
FAKE_FILE = "/nonexistent/closed_reference.py"   # never created
FAKE_ROOT = "/nonexistent/closed_reference"      # a bundle's files "live" under it; never created
MIN_DIGEST_BYTES = 64                            # bundle files shorter than this get no header digest

_HERE = os.path.dirname(os.path.abspath(__file__))
_lock = threading.Lock()
_modules: dict[str, object] = {}
_fake_files: dict[str, str] = {}   # fake path -> reference-relative name (error locations)
_window_lock = threading.Lock()
_window_depth = 0
_flat_aliases: set[str] = set()     # flat names a window served (_BundleFinder._flat_spec)


class SealedReferenceError(RuntimeError):
    """What escapes the sealed reference instead of its own exception (whose message
    and traceback lines can quote the source)."""


# --- sealing --------------------------------------------------------------------

def keystream(key_hex: str, n: int) -> bytes:
    """``n`` bytes derived from the key: sha256(key || counter) blocks."""
    key = bytes.fromhex(key_hex)
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:n])


def generate_key() -> str:
    return os.urandom(32).hex()


def _xor(body: bytes, key_hex: str) -> bytes:
    ks = keystream(key_hex, len(body))
    return (int.from_bytes(body, "big") ^ int.from_bytes(ks, "big")).to_bytes(len(body), "big")


def _wrap(magic: bytes, raw: bytes, key_hex: str | None, extra: bytes = b"") -> bytes:
    body = zlib.compress(raw, 9)
    if key_hex:
        body, mode = _xor(body, key_hex), MODE_KEYED
    else:
        mode = MODE_PLAIN
    return base64.b85encode(magic + mode + hashlib.sha256(raw).digest() + extra + body)


def seal(source: str, key_hex: str | None = None) -> bytes:
    """Source text -> payload bytes (one base85 line)."""
    return _wrap(MAGIC, source.encode("utf-8"), key_hex)


def bundle_manifest(files: dict, entry: str) -> dict:
    """A validated bundle manifest: relative POSIX paths that stay inside the bundle,
    importable ``.py`` paths, and an ``<file>.py::<function>`` entry among them."""
    out: dict[str, str] = {}
    for path, text in files.items():
        p = str(path).replace("\\", "/")
        parts = p.split("/")
        if not p or p.startswith("/") or any(x in ("", ".", "..") for x in parts):
            raise ValueError(f"bundle path {path!r} is not a relative path inside the bundle")
        if p.endswith(".py") and not all(x.isidentifier() for x in parts[:-1] + [parts[-1][:-3]]):
            raise ValueError(f"bundle path {path!r} is not importable (every component must be "
                             f"a Python identifier)")
        if not isinstance(text, str):
            raise ValueError(f"bundle file {path!r} is not text")
        out[p] = text
    file, sep, fn = str(entry or "").partition("::")
    if not sep or not fn.isidentifier() or not file.endswith(".py"):
        raise ValueError(f"entry {entry!r} is not '<file>.py::<function>'")
    if file not in out:
        raise ValueError(f"entry file {file!r} is not among the bundle's files {sorted(out)}")
    return {"entry": f"{file}::{fn}", "files": dict(sorted(out.items()))}


def bundle_from_solution(sol: dict) -> dict:
    """The manifest of a FlashInfer-Bench / SOL-ExecBench solution JSON (its ``sources``
    and ``spec.entry_point``)."""
    files = {s["path"]: s.get("content") or "" for s in sol.get("sources") or []}
    return bundle_manifest(files, (sol.get("spec") or {}).get("entry_point") or "")


def bundle_from_dir(root: str, entry: str) -> dict:
    files = {}
    for path in _walk(root, suffixes=None):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        if rel.endswith(".pyc"):
            continue
        with open(path, encoding="utf-8") as f:
            files[rel] = f.read()
    return bundle_manifest(files, entry)


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def seal_bundle(manifest: dict, key_hex: str | None = None) -> bytes:
    """Bundle manifest -> payload bytes (one base85 line)."""
    manifest = bundle_manifest(manifest["files"], manifest["entry"])
    digests = [hashlib.sha256(t.encode("utf-8")).digest() for t in manifest["files"].values()
               if len(t.encode("utf-8")) >= MIN_DIGEST_BYTES]
    extra = len(digests).to_bytes(2, "big") + b"".join(digests)
    return _wrap(MAGIC_BUNDLE, _canonical(manifest), key_hex, extra)


def _parse(payload: bytes) -> tuple[bytes, bytes, bytes, list[bytes], bytes]:
    """(magic, mode, digest of the plaintext, per-file digests, body)."""
    try:
        blob = base64.b85decode(payload.strip())
    except ValueError as exc:
        raise ValueError(f"not a closed-reference payload (base85: {exc})") from None
    magic = blob[:len(MAGIC)]
    if magic not in (MAGIC, MAGIC_BUNDLE) or len(blob) < len(MAGIC) + 1 + 32:
        raise ValueError("not a closed-reference payload (bad header)")
    mode = blob[len(MAGIC):len(MAGIC) + 1]
    digest = blob[len(MAGIC) + 1:len(MAGIC) + 33]
    rest = blob[len(MAGIC) + 33:]
    if magic == MAGIC:
        return magic, mode, digest, [digest], rest
    if len(rest) < 2:
        raise ValueError("not a closed-reference payload (bad bundle header)")
    n = int.from_bytes(rest[:2], "big")
    if len(rest) < 2 + 32 * n:
        raise ValueError("not a closed-reference payload (bad bundle header)")
    files = [rest[2 + 32 * i:2 + 32 * (i + 1)] for i in range(n)]
    return magic, mode, digest, files, rest[2 + 32 * n:]


def _split(payload: bytes) -> tuple[bytes, bytes, bytes]:
    magic, mode, digest, _, body = _parse(payload)
    return mode, digest, body


def is_bundle(payload: bytes) -> bool:
    return _parse(payload)[0] == MAGIC_BUNDLE


def source_digest(payload: bytes) -> str:
    """sha256 (hex) of the PLAINTEXT (source, or a bundle's canonical manifest), read
    from the header — no key needed."""
    return _parse(payload)[2].hex()


def payload_digest(payload: bytes) -> str:
    """sha256 (hex) of the payload bytes as shipped (provenance line in job logs)."""
    return hashlib.sha256(payload).hexdigest()


def unseal(payload: bytes, key_hex: str | None = None) -> str:
    """Payload bytes -> plaintext (the source; a bundle's canonical manifest JSON).
    Raises when the key is missing or wrong."""
    _, mode, digest, _, body = _parse(payload)
    if mode == MODE_KEYED:
        if not key_hex:
            raise PermissionError("the reference payload is keyed and no key was given")
        body = _xor(body, key_hex)
    elif mode != MODE_PLAIN:
        raise ValueError(f"unknown payload mode {mode!r}")
    try:
        raw = zlib.decompress(body)
    except zlib.error:
        raise PermissionError("reference unseal failed (wrong key?)") from None
    if hashlib.sha256(raw).digest() != digest:
        raise PermissionError("reference unseal failed (digest mismatch)")
    return raw.decode("utf-8")


def unseal_bundle(payload: bytes, key_hex: str | None = None) -> dict:
    if not is_bundle(payload):
        raise ValueError("the payload is a single file, not a bundle")
    return json.loads(unseal(payload, key_hex))


# --- the source window --------------------------------------------------------------

@contextlib.contextmanager
def window():
    """While open, the reference's source is served to ``inspect`` / ``linecache``
    (its JIT decorators need it); warnings are silenced (their display prints the
    source line). On close ``linecache`` forgets every line it read."""
    global _window_depth
    with _window_lock:
        _window_depth += 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        with _window_lock:
            _window_depth -= 1
            if _window_depth == 0:
                for f in _fake_files:
                    linecache.cache.pop(f, None)
                for alias in _flat_aliases:
                    sys.modules.pop(alias, None)
                _flat_aliases.clear()


def _sanitized(exc: BaseException) -> SealedReferenceError:
    where = None
    tb = exc.__traceback__
    while tb is not None:
        rel = _fake_files.get(tb.tb_frame.f_code.co_filename)
        if rel is not None:
            where = f"{rel}:{tb.tb_lineno}"
        tb = tb.tb_next
    detail = ""
    if isinstance(exc, ImportError) and getattr(exc, "name", None):
        detail = f" (module {exc.name!r})"
    return SealedReferenceError(
        f"the sealed reference raised {type(exc).__name__}{detail}"
        + (f" at {where}" if where else "")
        + " — message and traceback withheld (they can quote its source)")


def _guarded(fn, label: str):
    """``fn`` called inside a window, its exceptions sanitized."""
    def call(*args, **kwargs):
        with window():
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — the reference is foreign code
                err = _sanitized(exc)
        raise err from None
    call.__name__ = call.__qualname__ = label
    return call


# --- loading, in memory ---------------------------------------------------------

class _SourceLoader(importlib.abc.Loader):
    """Serves the decoded source to ``linecache`` (the route ``inspect.getsource``
    takes for a module with no file on disk), inside a window only, and to nothing
    else."""

    def __init__(self, name: str, src: str, filename: str = FAKE_FILE,
                 is_package: bool = False, builtins_ns: dict | None = None) -> None:
        self._name = name
        self._src = src
        self._filename = filename
        self._is_package = is_package
        self._builtins = builtins_ns

    def get_source(self, fullname: str) -> str:   # linecache.lazycache calls this
        if fullname != self._name or not _window_depth:
            raise ImportError(fullname)
        return self._src

    def get_filename(self, fullname: str) -> str:
        return self._filename

    def is_package(self, fullname: str) -> bool:
        return self._is_package

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self._builtins is not None:
            module.__dict__["__builtins__"] = self._builtins
        # dont_inherit=True: compile() otherwise inherits THIS module's __future__
        # flags, and this module has `from __future__ import annotations`. A sealed
        # CuTe DSL kernel then gets PEP 563 annotations it was never written for, and
        # `@cute.struct` — which builds its layout from the annotation OBJECTS — sees
        # strings and refuses the class ("Struct element only support struct/array/
        # base_dsl scalar, but got cute.struct.Align[...]"). The reference must be
        # compiled under its own semantics, not the loader's.
        code = compile(self._src, self._filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 — the reference module itself


def _import_roots(files: dict[str, str]) -> list[str]:
    """Directories a solution may import from by top-level name: the bundle root, and
    every directory that is neither a package nor inside one (a vendored tree the
    solution puts on ``sys.path``), shallowest first."""
    py = [p for p in files if p.endswith(".py")]
    dirs = {""}
    for p in py:
        parts = p.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:i]))
    pkgs = {d for d in dirs if d and f"{d}/__init__.py" in files}

    def free(d: str) -> bool:
        parts = d.split("/") if d else []
        return not any("/".join(parts[:i]) in pkgs for i in range(1, len(parts) + 1))

    return sorted((d for d in dirs if free(d)), key=lambda d: (d.count("/") + bool(d), d))


class _BundleFinder(importlib.abc.MetaPathFinder):
    """Serves ``<pkg>`` and ``<pkg>.<dotted path>`` from the bundle, nothing else."""

    def __init__(self, pkg: str, files: dict[str, str]) -> None:
        self.pkg = pkg
        self.mods: dict[str, tuple[str, str, bool]] = {}
        py = {p: t for p, t in files.items() if p.endswith(".py")}
        for p in py:
            parts = p[:-3].split("/")
            for i in range(len(parts)):   # every enclosing directory is a package
                d = "/".join(parts[:i])
                name = ".".join([pkg] + parts[:i])
                if name not in self.mods:
                    init = f"{d}/__init__.py" if d else "__init__.py"
                    self.mods[name] = (py.get(init, ""), f"{FAKE_ROOT}/{pkg}/{init}", True)
            if parts[-1] != "__init__":
                self.mods[".".join([pkg] + parts)] = (py[p], f"{FAKE_ROOT}/{pkg}/{p}", False)
        tops: dict[str, str] = {}
        for root in _import_roots(files):
            prefix = f"{root}/" if root else ""
            base = [pkg] + (root.split("/") if root else [])
            for p in py:
                if not p.startswith(prefix):
                    continue
                first = p[len(prefix):].split("/")[0]
                top = first[:-3] if first.endswith(".py") else first
                if top != "__init__" and top.isidentifier():
                    tops.setdefault(top, ".".join(base + [top]))
        self.tops = tops
        real = builtins.__import__

        def bundle_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and name:
                top, dot, rest = name.partition(".")
                target = tops.get(top)
                if target is not None:
                    mod = real(target + dot + rest, globals, locals, fromlist, 0)
                    return mod if fromlist else sys.modules[target]
            return real(name, globals, locals, fromlist, level)

        self.builtins = dict(builtins.__dict__)
        self.builtins["__import__"] = bundle_import
        for src, fake, _ in self.mods.values():
            _fake_files[fake] = fake[len(f"{FAKE_ROOT}/{pkg}/"):]

    def find_spec(self, fullname, path=None, target=None):
        m = self.mods.get(fullname)
        if m is None:
            return self._flat_spec(fullname)
        src, fake, is_pkg = m
        loader = _SourceLoader(fullname, src, fake, is_pkg, self.builtins)
        spec = importlib.machinery.ModuleSpec(fullname, loader, origin=fake, is_package=is_pkg)
        spec.has_location = True
        return spec

    def _flat_spec(self, fullname):
        """A top-level name the bundle provides, asked for through plain ``importlib``
        rather than the bundle's own ``__import__``: CuTe DSL's preprocessor replays a
        kernel module's imports that way while it compiles, so a solution importing its
        sibling files by flat name (``from gdn_core import ...``) needs it. Served only
        inside a window -- that is, during a reference call -- as an alias of the bundle's
        module, and dropped from ``sys.modules`` when the window closes, so outside the
        reference the name still resolves to nothing."""
        if not _window_depth:
            return None
        top, dot, rest = fullname.partition(".")
        base = self.tops.get(top)
        if base is None or (base + dot + rest) not in self.mods:
            return None
        module = importlib.import_module(base + dot + rest)
        _flat_aliases.add(fullname)
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(module))


class _AliasLoader(importlib.abc.Loader):
    """Hands back a module the bundle has already loaded, under a second name."""

    def __init__(self, module) -> None:
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        pass


def default_payload_path(here: str = _HERE) -> str:
    """``ref_payload.bin`` next to this file (reference tree) or under ``ref_bin/``
    (operator card) — whichever exists, else the card location."""
    for cand in (os.path.join(here, "ref_payload.bin"),
                 os.path.join(here, "ref_bin", "ref_payload.bin")):
        if os.path.exists(cand):
            return cand
    return os.path.join(here, "ref_bin", "ref_payload.bin")


def key_from_env(key_env: str | None = DEFAULT_KEY_ENV) -> str | None:
    if not key_env:
        return None
    return os.environ.get(key_env, "").strip() or None


def key_file(payload_path: str) -> str:
    """Where the job keeps the key once gpu_entry.sh has moved it out of the environment:
    ``<payload stem>.key`` beside the payload (``ref_bin/ref_payload.key``)."""
    return os.path.splitext(payload_path)[0] + ".key"


def key_from_file(payload_path: str) -> str | None:
    try:
        with open(key_file(payload_path), encoding="ascii") as f:
            return f.read().strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def _plaintext(payload: bytes, key: str | None, key_env: str | None) -> str:
    try:
        return unseal(payload, key)
    except PermissionError as exc:
        if key is None and _parse(payload)[1] == MODE_KEYED:
            raise PermissionError(
                f"{exc}: the closed reference expects its key in the job environment "
                f"variable {key_env!r} (submit.local.toml [env]), or in the key file "
                f"gpu_entry.sh moves it to beside the payload") from None
        raise


def _load_single(name: str, src: str):
    loader = _SourceLoader(name, src)
    spec = importlib.util.spec_from_loader(name, loader, origin=FAKE_FILE)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = FAKE_FILE
    _fake_files[FAKE_FILE] = "reference"
    sys.modules[name] = module
    try:
        with window():
            loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(name, None)
        err = _sanitized(exc)
        raise err from None
    return module


def _load_bundle(name: str, manifest: dict):
    finder = _BundleFinder(name, manifest["files"])
    sys.meta_path.insert(0, finder)
    mod_name = ".".join([name] + manifest["entry"].split("::", 1)[0][:-3].split("/"))
    try:
        with window():
            module = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)
        for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(m, None)
        err = _sanitized(exc)
        raise err from None
    module.__closed_ref_entry__ = manifest["entry"].split("::", 1)[1]
    sys.modules[name].__closed_ref_module__ = module   # load() by name finds the entry file
    return module


def load(payload_path: str | None = None, *, name: str = DEFAULT_MODULE_NAME,
         key_env: str | None = DEFAULT_KEY_ENV, key_hex: str | None = None):
    """Import the closed reference once as module ``name`` (a bundle: package ``name``,
    returning the module of its entry file); returns the module.

    ``key_hex`` overrides ``key_env`` (tests); without the variable the key is read from
    ``key_file(payload_path)`` (where gpu_entry.sh moves it); a keyed payload with none raises
    ``PermissionError`` naming the variable, so a job without the key says so
    instead of failing somewhere inside the reference. An exception the reference
    raises while importing comes out as ``SealedReferenceError``.
    """
    path = payload_path or default_payload_path()
    with _lock:
        if name in _modules:
            return _modules[name]
        if name in sys.modules:
            m = sys.modules[name]
            _modules[name] = getattr(m, "__closed_ref_module__", m)
            return _modules[name]
        with open(path, "rb") as f:
            payload = f.read()
        key = key_hex or key_from_env(key_env) or (key_from_file(path) if key_env else None)
        text = _plaintext(payload, key, key_env)
        if is_bundle(payload):
            module = _load_bundle(name, json.loads(text))
        else:
            module = _load_single(name, text)
        _modules[name] = module
        return module


def entry(entry_name: str | None = None, payload_path: str | None = None, **load_kw):
    """The reference's entry point, guarded: every call runs inside a source window and
    an exception comes out sanitized (``SealedReferenceError``). ``entry_name`` defaults
    to a bundle's own entry function."""
    module = load(payload_path, **load_kw)
    entry_name = entry_name or getattr(module, "__closed_ref_entry__", None)
    if not entry_name:
        raise ValueError("a single-file closed reference needs the entry's name")
    try:
        fn = getattr(module, entry_name)
    except AttributeError:
        raise AttributeError(f"closed reference has no entry {entry_name!r}") from None
    return _guarded(fn, entry_name)


# --- plaintext leak check -----------------------------------------------------------

_SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
MIN_DISTINCTIVE_LINE = 24     # characters, stripped; shorter lines are too common
LINE_OVERLAP_REFUSE = 0.4     # share of one reference file's distinctive lines found in one file


def _distinctive_lines(text: str) -> set[str]:
    return {ln.strip() for ln in text.splitlines() if len(ln.strip()) >= MIN_DISTINCTIVE_LINE}


def _walk(root: str, suffixes=(".py",)):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if suffixes is None or fn.endswith(suffixes):
                yield os.path.join(dirpath, fn)


def _walk_py(root: str):
    yield from _walk(root)


def check_tree(root: str, payload: bytes, key_hex: str | None = None) -> list[str]:
    """Files under ``root`` that are (a file of) the reference in plaintext.

    Always: a file whose sha256 equals a header digest (exact copy). With the key: a
    file carrying at least ``LINE_OVERLAP_REFUSE`` of one reference file's distinctive
    lines (a lightly edited copy). Returns ``"<path>: <why>"`` lines; an empty list is a
    clean tree.
    """
    magic, mode, _, digests, _ = _parse(payload)
    ref_texts: dict[str, set[str]] = {}
    if key_hex or mode == MODE_PLAIN:
        text = unseal(payload, key_hex)
        if magic == MAGIC_BUNDLE:
            ref_texts = {p: _distinctive_lines(t) for p, t in json.loads(text)["files"].items()
                         if p.endswith(".py")}
        else:
            ref_texts = {"reference": _distinctive_lines(text)}
        ref_texts = {p: ls for p, ls in ref_texts.items() if ls}
    digests_set = set(digests)
    hits: list[str] = []
    for path in _walk_py(root):
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        if hashlib.sha256(raw).digest() in digests_set:
            hits.append(f"{path}: exact copy of the reference source")
            continue
        if ref_texts:
            try:
                lines = _distinctive_lines(raw.decode("utf-8"))
            except UnicodeDecodeError:
                continue
            for rel, ref_lines in ref_texts.items():
                shared = len(ref_lines & lines)
                share = shared / len(ref_lines)
                if share >= LINE_OVERLAP_REFUSE:
                    of = "" if rel == "reference" else f" {rel}"
                    hits.append(f"{path}: {share:.0%} of the reference{of}'s distinctive lines "
                                f"({shared}/{len(ref_lines)})")
                    break
    return hits


# --- command line -------------------------------------------------------------------

def _read_bundle_or_source(source: str, entry_arg: str | None):
    """(single-file source text, None) or (None, bundle manifest) for the seal command."""
    if os.path.isdir(source):
        if not entry_arg:
            raise SystemExit("seal <dir>: --entry <file>.py::<function> is required")
        return None, bundle_from_dir(source, entry_arg)
    with open(source, encoding="utf-8") as f:
        text = f.read()
    if source.endswith(".json"):
        sol = json.loads(text)
        manifest = bundle_from_solution(sol)
        if entry_arg:
            manifest = bundle_manifest(manifest["files"], entry_arg)
        return None, manifest
    return text, None


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="closed_ref.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal", help="kernel source (a .py, a solution .json, a directory) "
                                    "-> opaque payload")
    s.add_argument("source")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--entry", default=None,
                   help="a bundle's entry, <file>.py::<function> (required for a directory; "
                        "overrides a solution's spec.entry_point)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--key", help="hex key (the other tree's payload must use the same one)")
    g.add_argument("--gen-key", action="store_true", help="generate a key and print it")
    g.add_argument("--plain", action="store_true", help="no key (payload decodes anywhere)")

    d = sub.add_parser("digest", help="print the payload's digests")
    d.add_argument("payload")

    c = sub.add_parser("check-tree", help="refuse a tree that holds the reference in plaintext")
    c.add_argument("tree")
    c.add_argument("payload")
    c.add_argument("--key", help="hex key; enables the edited-copy check")
    c.add_argument("--key-env", default=DEFAULT_KEY_ENV,
                   help=f"environment variable holding the key (default {DEFAULT_KEY_ENV})")

    a = p.parse_args(argv)
    if a.cmd == "seal":
        src, manifest = _read_bundle_or_source(a.source, a.entry)
        key = None
        if a.gen_key:
            key = generate_key()
        elif a.key:
            key = a.key
        elif not a.plain:
            p.error("choose --gen-key, --key <hex> or --plain")
        payload = seal(src, key) if manifest is None else seal_bundle(manifest, key)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "wb") as f:
            f.write(payload)
        what = (f"bundle of {len(manifest['files'])} file(s), entry {manifest['entry']}"
                if manifest else "single file")
        print(f"sealed {a.source} -> {a.out} ({len(payload)} bytes, {what}, "
              f"{'keyed' if key else 'plain'})")
        print(f"source sha256  {source_digest(payload)}")
        print(f"payload sha256 {payload_digest(payload)}")
        if a.gen_key:
            print(f"key (put in submit.local.toml [env] {DEFAULT_KEY_ENV} of every tree "
                  f"that loads it; never commit it): {key}")
        return 0
    if a.cmd == "digest":
        with open(a.payload, "rb") as f:
            payload = f.read()
        magic, mode, _, digests, _ = _parse(payload)
        print(f"mode           {'keyed' if mode == MODE_KEYED else 'plain'}")
        if magic == MAGIC_BUNDLE:
            print(f"kind           bundle ({len(digests)} file digest(s) in the header)")
        print(f"source sha256  {source_digest(payload)}")
        print(f"payload sha256 {payload_digest(payload)}")
        return 0
    if a.cmd == "check-tree":
        with open(a.payload, "rb") as f:
            payload = f.read()
        key = a.key or key_from_env(a.key_env)
        hits = check_tree(a.tree, payload, key)
        if hits:
            print("CLOSED-REFERENCE LEAK: the reference source is in this tree in plaintext")
            for h in hits:
                print(f"  {h}")
            return 3
        scope = "exact and edited copies" if (key or _parse(payload)[1] == MODE_PLAIN) \
            else "exact copies only (no key: edited copies not checked)"
        print(f"clean: {a.tree} holds no plaintext copy of the reference ({scope})")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
