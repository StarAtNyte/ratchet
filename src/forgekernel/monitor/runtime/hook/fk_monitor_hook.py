"""Correctness monitor, in-process half: rides into a card's bench through
``sitecustomize`` and does two things, neither of which needs the card's help.

1. **Watch the kernel's outputs.** The first ``FK_MON_MAX_CALLS`` calls into the
   card's ``kernel`` module (any public function, reached from outside the module)
   are summarised: shape, dtype, NaN / +inf / -inf counts and finite-only checksums of
   every tensor it produced, plus whether any floating input was non-finite. What it
   produced is what it returned AND every tensor argument it wrote: the arguments are
   summarised before the call and again after it, so a kernel that fills a caller's
   ``out`` (destination passing) or updates an argument in place is watched like one
   that returns, and what an output buffer held going in is never read as an input.
   Calls inside ``kernel.py`` go through its globals, never through the module
   attribute, so the student's own call graph (and anything handed to a JIT
   compiler) is untouched.

2. **Shift the inputs the bench draws.** The bench builds its inputs from torch's RNG
   and hands the same tensors to the kernel and to its own reference, then judges them
   with its own tolerance. Perturbing the draw therefore keeps the comparison honest:
   both sides see the same new inputs, and the card's judge decides. Profiles:

   - ``base`` / ``repeat``: no change (``repeat`` exists to compare checksums).
   - ``seed1`` / ``seed2``: every seed the bench sets (torch global, torch.Generator,
     numpy, random) is offset, so the same distribution is drawn afresh.
   - ``amp`` (x4) / ``tiny`` (x1/16): scale every normal draw.
   - ``spikes``: 0.1 % of every normal draw set to +/-12.
   - ``zero_block``: in every normal draw of rank >= 2, the leading 1/8 of the last
     dimension and the leading 1/16 of the rows are zeroed.
   - ``const_rows``: the leading 1/16 of the rows of every rank >= 2 normal draw repeat
     row 0.
   - ``int_edges``: integer tensors move to the edges of the range the bench asked for —
     ``torch.randint`` draws put 1/16 of their entries at ``low`` and 1/16 at
     ``high - 1``; ``torch.full`` integer tensors with a non-zero fill put their leading
     1/16 at 0 (a length / count of zero: empty rows). Values stay inside what the bench
     itself could have produced, so the input stays valid; float draws are untouched.

   The float profiles reshape only normal draws (``torch.randn``, ``randn_like``,
   ``normal``, ``Tensor.normal_``); uniform draws are left alone so the input's
   structure stays what the bench meant.

Nothing here may break the bench: every hook swallows its own errors and records them.
The summary lands in ``$FK_MON_OUT/hook-<pid>.json``, rewritten after each record so a
killed process still leaves what it saw.
"""

from __future__ import annotations

import functools
import importlib.abc
import json
import os
import sys
import time
import types

ENV_PROFILE = "FK_MON_PROFILE"
ENV_OUT = "FK_MON_OUT"
ENV_KERNEL = "FK_MON_KERNEL"
ENV_MAX_CALLS = "FK_MON_MAX_CALLS"
ENV_ROOT = "FK_MON_ROOT"

SEED_OFFSETS = {"seed1": 1_000_003, "seed2": 2_000_029}
SCALE = {"amp": 4.0, "tiny": 1.0 / 16.0}
SPIKE_VALUE = 12.0
SPIKE_FRACTION = 1e-3
MIN_NUMEL = 64          # smaller normal draws (a per-head sink vector) keep their shape
MAX_TENSORS_PER_CALL = 32

_state: dict = {}
_torch = None


# --- bookkeeping ----------------------------------------------------------------
def _err(where: str, e: BaseException) -> None:
    errs = _state.setdefault("errors", [])
    if len(errs) < 20:
        errs.append(f"{where}: {e.__class__.__name__}: {e}"[:300])


def _dump() -> None:
    out = _state.get("out")
    if not out:
        return
    try:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"hook-{os.getpid()}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in _state.items() if k != "out"}, fh)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — the monitor never costs the bench
        pass


# --- tensor summaries -------------------------------------------------------------
def _tensors(obj, depth: int = 0, acc=None):
    acc = [] if acc is None else acc
    t = _torch
    if len(acc) >= MAX_TENSORS_PER_CALL or depth > 3:
        return acc
    if t is not None and isinstance(obj, t.Tensor):
        acc.append(obj)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _tensors(x, depth + 1, acc)
    elif isinstance(obj, dict):
        for x in obj.values():
            _tensors(x, depth + 1, acc)
    return acc


def _summary(x) -> dict:
    t = _torch
    s = {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""),
         "numel": int(x.numel())}
    if x.numel() == 0:
        return s
    if x.is_floating_point() or x.is_complex():
        f = x.detach()
        if f.is_complex():
            f = t.view_as_real(f)
        f = f.float() if f.element_size() < 4 else f
        nan = t.isnan(f)
        pinf = t.isposinf(f)
        ninf = t.isneginf(f)
        s["nan"] = int(nan.sum().item())
        s["posinf"] = int(pinf.sum().item())
        s["neginf"] = int(ninf.sum().item())
        fin = t.where(nan | pinf | ninf, t.zeros((), dtype=f.dtype, device=f.device), f).double()
        s["sum"] = float(fin.sum().item())
        s["abs_sum"] = float(fin.abs().sum().item())
        s["sq_sum"] = float((fin * fin).sum().item())
        s["zeros"] = int((f == 0).sum().item())
    elif x.dtype == t.bool:
        s["sum"] = float(x.sum().item())
    else:
        v = x.detach().to(t.int64)
        s["sum"] = float(v.sum().item())
        s["abs_sum"] = float(v.abs().sum().item())
    return s


def _capturing() -> bool:
    t = _torch
    try:
        return bool(t.cuda.is_available() and t.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _nonfinite_of(s: dict) -> int:
    return int(s.get("nan") or 0) + int(s.get("posinf") or 0) + int(s.get("neginf") or 0)


def _same_storage(a, b) -> bool:
    try:
        return a is b or (a.data_ptr() == b.data_ptr() and a.shape == b.shape
                          and a.dtype == b.dtype)
    except Exception:  # noqa: BLE001
        return False


def _before(args, kwargs):
    """The call's tensor arguments and what they hold going in, or None when this
    call is not recorded (past FK_MON_MAX_CALLS, or inside a CUDA graph capture,
    where a host read would break the capture)."""
    if len(_state["calls"]) >= _state["max_calls"] or _capturing():
        return None
    try:
        ins = _tensors((args, kwargs))
        return ins, [_summary(x) for x in ins]
    except Exception as e:  # noqa: BLE001
        _err("snapshot", e)
        return None


def _record(name: str, pre, out) -> None:
    """One call: its tensor arguments as they went in (``pre``) and came out, and
    what it returned. An argument whose summary changed across the call was WRITTEN
    — a destination-passing ``out``, or an in-place update — so it is reported with
    the outputs (slot ``arg<j>``) and its pre-call content (an output buffer's
    leftovers, for all anyone knows) counts neither as an input checksum nor as a
    non-finite input."""
    ins, before = pre
    try:
        per = _state.setdefault("per_fn", {})
        idx = per.get(name, 0)
        per[name] = idx + 1
        after = [_summary(x) for x in ins]
        written = [j for j, (b, a) in enumerate(zip(before, after)) if b != a]
        sums, nonfinite = [], []
        for j, s in enumerate(before):
            nonfinite.append(_nonfinite_of(s))
            sums.append(None if j in written else float(s.get("abs_sum", s.get("sum", 0.0))))
        rets = _tensors(out)
        outs = [{**_summary(x), "slot": f"ret{k}"} for k, x in enumerate(rets)]
        outs += [{**after[j], "slot": f"arg{j}"} for j in written
                 if not any(_same_storage(ins[j], r) for r in rets)]
        _state["calls"].append({
            "fn": name, "i": idx, "n_inputs": len(ins), "in_sums": sums,
            "in_nonfinite": nonfinite, "written": written,
            "inputs_nonfinite": sum(n for j, n in enumerate(nonfinite) if j not in written),
            "outputs": outs})
    except Exception as e:  # noqa: BLE001
        _err(f"record {name}", e)
    _dump()


# --- the kernel module -------------------------------------------------------------
def _wrap(fn, name: str):
    @functools.wraps(fn)
    def watched(*args, **kwargs):
        if _torch is None:
            _load_torch()
        # read the arguments BEFORE the call: after it, an output buffer the kernel
        # filled would read as an input
        pre = _before(args, kwargs) if _torch is not None else None
        out = fn(*args, **kwargs)
        if pre is not None:
            _record(name, pre, out)
        elif _torch is not None:
            _state["unrecorded"] = _state.get("unrecorded", 0) + 1
        return out
    watched.__fk_monitor__ = True
    return watched


class _WatchedModule(types.ModuleType):
    """The kernel module, seen from outside: public functions come back wrapped.
    Code inside the module resolves names through its globals dict, which this
    never touches."""

    def __getattribute__(self, name):
        obj = super().__getattribute__(name)
        if name.startswith("_") or not isinstance(obj, types.FunctionType):
            return obj
        d = super().__getattribute__("__dict__")
        if obj.__module__ != d.get("__name__"):
            return obj
        cache = d.setdefault("__fk_monitor_wrappers__", {})
        hit = cache.get(name)
        if hit is None or hit[0] is not obj:
            hit = (obj, _wrap(obj, name))
            cache[name] = hit
        return hit[1]


def _instrument(module) -> None:
    try:
        module.__class__ = _WatchedModule
        _state["kernel_module"] = getattr(module, "__file__", None) or module.__name__
    except Exception as e:  # noqa: BLE001
        _err("instrument", e)
    _dump()


def watch_loaded(module, origin: str | None) -> None:
    """For a bench that loads the kernel by file path under a name of its own
    (``importlib.util.spec_from_file_location``): that never reaches a meta-path
    finder, so the bench hands the module over here once it has run it."""
    if not _state.get("installed"):
        return
    if _state.get("kernel_path"):
        if not _is_kernel(origin):
            return
    elif not origin or os.path.splitext(os.path.basename(origin))[0] != _kernel_name():
        return
    _instrument(module)


def _kernel_name() -> str:
    want = _state.get("kernel_path")
    return os.path.splitext(os.path.basename(want))[0] if want else "kernel"


def _is_kernel(origin: str | None) -> bool:
    """With FK_MON_KERNEL set, only that file is the kernel (a same-named module
    elsewhere on the path is not); without it, the top-level ``kernel`` module."""
    want = _state.get("kernel_path")
    if not want:
        return True
    try:
        return bool(origin) and os.path.realpath(origin) == os.path.realpath(want)
    except OSError:
        return False


class _KernelFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _kernel_name():
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            try:
                spec = finder.find_spec(name, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module") \
                or not _is_kernel(spec.origin):
            return None
        real = spec.loader.exec_module

        def exec_module(module, _real=real):
            _real(module)
            _instrument(module)

        try:
            spec.loader.exec_module = exec_module
        except Exception as e:  # noqa: BLE001
            _err("loader", e)
            return None
        return spec


# --- the RNG ------------------------------------------------------------------------
def _perturb(x):
    prof = _state["profile"]
    t = _torch
    try:
        if not isinstance(x, t.Tensor) or not x.is_floating_point() or x.numel() < MIN_NUMEL:
            return x
        with t.no_grad():
            if prof in SCALE:
                x.mul_(SCALE[prof])
            elif prof == "spikes":
                flat = x.view(-1)
                k = max(1, int(flat.numel() * SPIKE_FRACTION))
                g = t.Generator(device=x.device).manual_seed(20260929 + _state["perturbed"])
                idx = t.randint(0, flat.numel(), (k,), generator=g, device=x.device)
                sign = t.randint(0, 2, (k,), generator=g, device=x.device).to(x.dtype) * 2 - 1
                flat[idx] = sign * SPIKE_VALUE
            elif prof == "zero_block":
                if x.dim() < 2:
                    return x
                x[..., : max(1, x.shape[-1] // 8)] = 0
                x[: max(1, x.shape[0] // 16)] = 0
            elif prof == "const_rows":
                if x.dim() < 2 or x.shape[0] < 2:
                    return x
                n = max(2, x.shape[0] // 16)
                x[1:n] = x[0:1]
            else:
                return x
        _state["perturbed"] += 1
    except Exception as e:  # noqa: BLE001
        _err(f"perturb {prof}", e)
    return x


def _from_bench(depth: int = 2) -> bool:
    """Is the draw being made by the bench's own code? Only files inside the tree
    count (FK_MON_ROOT), never the kernel file or the monitor itself: a buffer the
    student's kernel allocates with torch.full, or a draw inside a sealed reference,
    must not be reshaped. Without FK_MON_ROOT every caller counts."""
    root = _state.get("root")
    if not root:
        return True
    try:
        fn = sys._getframe(depth).f_code.co_filename
    except ValueError:
        return False
    if not fn:
        return False
    if fn.startswith("<"):
        # source the bench compiles itself: only a task's judge (its make_inputs draws
        # the inputs both sides get); a reference decoded in memory is not the bench
        return fn.endswith(" judge>")
    try:
        path = os.path.realpath(fn)
    except OSError:
        return False
    if not path.startswith(root + os.sep):
        return False
    if path.startswith(os.path.join(root, ".fk_monitor") + os.sep):
        return False
    kp = _state.get("kernel_real")
    return not (kp and path == kp)


def _edge_ints(x, lo: int | None, hi: int | None):
    """``int_edges`` on one integer tensor: randint draws (``lo``/``hi`` known) get
    1/16 of entries at ``lo`` and 1/16 at ``hi - 1``; a constant fill gets its leading
    1/16 at 0."""
    t = _torch
    try:
        if not isinstance(x, t.Tensor) or x.is_floating_point() or x.is_complex() \
                or x.dtype == t.bool or x.numel() < 16:
            return x
        flat = x.view(-1)
        k = max(1, flat.numel() // 16)
        with t.no_grad():
            if lo is not None and hi is not None:
                if hi - 1 <= lo:
                    return x
                g = t.Generator(device=x.device).manual_seed(20260929 + _state["perturbed"])
                idx = t.randperm(flat.numel(), generator=g, device=x.device)[: 2 * k]
                flat[idx[:k]] = lo
                flat[idx[k:]] = hi - 1
            else:
                if not bool((flat[:k] != 0).any()):
                    return x
                flat[:k] = 0
        _state["perturbed"] += 1
    except Exception as e:  # noqa: BLE001
        _err("int_edges", e)
    return x


def _install_int_edges() -> None:
    t = _torch

    def randint(orig):
        def f(*a, **k):
            out = orig(*a, **k)
            if not _from_bench():
                return out
            try:
                ints = [v for v in a if isinstance(v, int) and not isinstance(v, bool)]
                if "low" in k or "high" in k:
                    lo, hi = int(k.get("low", 0)), int(k["high"])
                elif len(ints) >= 2:
                    lo, hi = ints[0], ints[1]
                elif ints:
                    lo, hi = 0, ints[0]
                else:
                    return out
            except Exception as e:  # noqa: BLE001
                _err("randint args", e)
                return out
            return _edge_ints(out, lo, hi)
        return f
    _patch(t, "randint", randint)

    def full(orig):
        def f(*a, **k):
            out = orig(*a, **k)
            return _edge_ints(out, None, None) if _from_bench() else out
        return f
    _patch(t, "full", full)


def _reseed_generator(g) -> None:
    """Offset a torch.Generator's seed the first time a draw uses it (and again
    whenever the bench reseeds it)."""
    try:
        seen = _state.setdefault("_gens", {})
        cur = g.initial_seed()
        if seen.get(id(g)) == cur:
            return
        new = (int(cur) + _state["seed_offset"]) % (2 ** 63)
        g.manual_seed(new)
        seen[id(g)] = new
        _state["reseeded"] += 1
    except Exception as e:  # noqa: BLE001
        _err("reseed generator", e)


def _patch(owner, attr: str, make) -> None:
    orig = getattr(owner, attr, None)
    if orig is None or getattr(orig, "__fk_monitor__", False):
        return
    try:
        new = make(orig)
        new.__fk_monitor__ = True
        setattr(owner, attr, new)
    except Exception as e:  # noqa: BLE001
        _err(f"patch {attr}", e)


_NORMAL_FUNCS = ("randn", "randn_like", "normal")
_RNG_FUNCS = ("randn", "randn_like", "normal", "rand", "rand_like", "randint", "randint_like",
              "randperm", "bernoulli", "multinomial", "poisson")
_RNG_METHODS = ("normal_", "uniform_", "random_", "exponential_", "bernoulli_",
                "geometric_", "cauchy_", "log_normal_")


def _install_rng() -> None:
    t = _torch
    prof = _state["profile"]
    if prof in SEED_OFFSETS:
        off = _state["seed_offset"] = SEED_OFFSETS[prof]
        # a bench that never seeds still draws something new
        try:
            t.manual_seed(off)
        except Exception as e:  # noqa: BLE001
            _err("default seed", e)

        def seeded(orig):
            def f(seed, *a, **k):
                _state["reseeded"] += 1
                return orig((int(seed) + off) % (2 ** 63), *a, **k)
            return f
        for owner in (t, t.random):
            _patch(owner, "manual_seed", seeded)
        for attr in ("manual_seed", "manual_seed_all"):
            _patch(t.cuda, attr, seeded)

        def gen_aware(orig):
            def f(*a, **k):
                g = k.get("generator")
                if g is not None:
                    _reseed_generator(g)
                return orig(*a, **k)
            return f
        for name in _RNG_FUNCS:
            _patch(t, name, gen_aware)
        for name in _RNG_METHODS:
            _patch(t.Tensor, name, gen_aware)
        try:
            import random
            _patch(random, "seed", lambda orig: (lambda a=None, *r, **k: orig(
                None if a is None else (hash(a) + off), *r, **k)))
        except Exception as e:  # noqa: BLE001
            _err("random", e)
        try:
            import numpy as np
            _patch(np.random, "seed", lambda orig: (lambda s=None: orig(
                None if s is None else (int(s) + off) % (2 ** 32))))
            _patch(np.random, "default_rng", lambda orig: (lambda s=None, *a, **k: orig(
                None if s is None else (int(s) + off) % (2 ** 63), *a, **k)))
        except Exception:  # noqa: BLE001 — no numpy, nothing to offset
            pass
        return

    if prof == "int_edges":
        _install_int_edges()
        return
    if prof in ("base", "repeat") or prof not in (*SCALE, "spikes", "zero_block", "const_rows"):
        return

    def perturbing(orig):
        def f(*a, **k):
            out = orig(*a, **k)
            return _perturb(out) if _from_bench() else out
        return f
    for name in _NORMAL_FUNCS:
        _patch(t, name, perturbing)

    def perturbing_method(orig):
        def f(self, *a, **k):
            out = orig(self, *a, **k)
            return _perturb(out) if _from_bench() else out
        return f
    _patch(t.Tensor, "normal_", perturbing_method)


# --- entry ---------------------------------------------------------------------------
def _load_torch():
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except Exception as e:  # noqa: BLE001 — a process without torch has nothing to watch
            _err("import torch", e)
    return _torch


def install() -> None:
    prof = os.environ.get(ENV_PROFILE, "").strip()
    if not prof or _state.get("installed"):
        return
    _state.update({
        "installed": True, "profile": prof, "pid": os.getpid(), "argv": sys.argv[:4],
        "started": time.time(), "calls": [], "errors": [], "perturbed": 0,
        "reseeded": 0, "seed_offset": 0, "kernel_module": None,
        "max_calls": int(os.environ.get(ENV_MAX_CALLS, "64") or 64),
        "kernel_path": os.environ.get(ENV_KERNEL) or None,
        "out": os.environ.get(ENV_OUT) or None,
    })
    root = os.environ.get(ENV_ROOT)
    _state["root"] = os.path.realpath(root) if root else None
    if _state["kernel_path"]:
        _state["kernel_real"] = os.path.realpath(_state["kernel_path"])
    sys.meta_path.insert(0, _KernelFinder())
    # RNG patches need torch now: the bench draws its inputs right after import.
    # A process that is not the bench (a shell helper, nvidia-smi wrapper) skips
    # them when torch is absent — importing it there costs a second or two at most.
    if prof not in ("base", "repeat") and _load_torch() is not None:
        _install_rng()
    import atexit
    atexit.register(_dump)
