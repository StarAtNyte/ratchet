"""Precision gate: how far the kernel's outputs are from the math, next to how far the
reference's own outputs are. Copy into <op>/eval/ (``forge new`` does), do not edit.

A tolerance check (|d| <= atol + rtol*|ref|) cannot tell a kernel that rounds where the
reference rounds from one that accumulates in bf16, multiplies in tf32 or quantizes to
fp8: at 1e-2 both pass. Tightening the tolerance would fail honest kernels whose
reduction order merely differs. What separates the two is an ERROR RATIO against a
higher-precision truth:

    truth   the reference run in float64: its floating inputs and every float cast it
            makes (``.float()``, ``.to(torch.bfloat16)``, ``dtype=torch.float32``) are
            lifted to float64 (``upcast_call``). A reference that cannot run there
            (an op without a float64 kernel) runs in float32 instead; one that cannot run
            either way is recorded ``unavailable`` with the exception -- never a silent
            pass. fp8 / fp4 inputs are exact in float64 and stay as they are.
    e_ref   relative L2 error of the reference's own output (native dtypes) vs truth
    e_cand  the same for the kernel's output, on the same inputs
    fail    e_cand > K_RATIO * e_ref + max(FLOOR_ULPS * u(output dtype), FLOOR_REL)

The floor keeps an exact reference (e_ref ~ 0: a copy, a gather) from failing every
kernel that rounds once: one unit of roundoff of the OUTPUT dtype, and never less than
FLOOR_REL = 2^-17 relative -- an fp32 output computed with fast-math exp2 / log2 and
another reduction order sits ~1e-6 from the truth while its reference sits ~1e-7, a
ratio of 20 that is no precision drop at all (tf32's own roundoff is 2^-11). Relative L2
(||y - t|| / ||t|| over one output tensor) is used because reduced-precision arithmetic
moves every element, which a norm sees with little noise; a NaN / inf where the truth is
finite makes e_cand infinite (an overflowed fp16 intermediate is a precision failure).

K_RATIO, FLOOR_ULPS and FLOOR_REL were calibrated on B200 over 14 SOL-ExecBench problems
(GEMM, rmsnorm, GQA / MLA attention, fp8 MoE, fp32 L1/L2 blocks; 1956 readings):
honest implementations -- the reference run in fp32 or with another reduction order,
the repository's best leaderboard solutions -- reach at most a quarter of the limit
(worst e_cand / limit 0.25, a split-K GEMM at 1.33x its reference's error), while
the same references with bf16 / fp16 accumulation, tf32 matmuls, fp8-quantized inputs
or fp32 intermediates rounded to bf16 / fp16 are caught on 643 of the 745 readings
they moved (194 of 279 that the official tolerance passes). What passes is a drop
within ~5x of the output's own rounding.

Stress profiles (``stress_tensor``) feed the same measurement with inputs that reach the
RANGE shortcuts a natural draw does not: ``offset`` (+1 on every normal draw: a large
common component -- an exp without the max subtracted, a variance as E[x^2] - E[x]^2),
``spikes`` (0.1% of elements at +-12: outliers that coarse scaling or a narrow
accumulator flattens) and ``amp`` (x4: fp16 intermediates overflow). On B200 they moved
no honest reading past the limit and never made a reference's own output non-finite;
an fp16 softmax without max subtraction that every natural input set passed came back
inf under all three. A profile under which the reference's own output is not finite is
recorded ``unavailable`` (the stress broke the dtype, not the kernel).

Who acts on the verdict is the loop, not this file: the eval JSON carries a top-level
``precision`` block (``PrecisionRecorder.block()``) and the driver applies the card's
``[loop] precision_contract`` -- ``strict`` (a failed check REVERTs like a correctness
failure), ``official`` (recorded only) or ``off``. The bench's own ``correctness.pass``
is never changed here. A leaderboard card measures only when its problem.json sets
``precision_contract`` to strict / official; by default (off) eval/leaderboard.py judges
exactly as the board does and never calls into this file.

A hand-written bench uses it like this::

    from precision import PrecisionRecorder, truth_of, snapshot, restore, stress_tensor
    prec = PrecisionRecorder()
    for i, inputs in enumerate(cases):                 # the inputs correctness judges
        pristine = snapshot(inputs)                    # before anything writes into them
        out = kernel(*inputs); ref = reference(*inputs)
        prec.measure(f"case{i}", out, ref, lambda: truth_of(reference, restore(pristine)))
    # optional: one stressed set per profile (the bench decides which inputs may move),
    # prec.measure("stress", ..., stress=profile)
    result["precision"] = prec.block()
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch.overrides import TorchFunctionMode

SCHEMA = 1
METRIC = "rel_l2"
# Calibrated on B200 (see the module docstring); the loop reads these from the block.
K_RATIO = 3.0
FLOOR_ULPS = 1.0
FLOOR_REL = 2.0 ** -17
# A check whose reference is this far from its own upcast run is not a precision reading
# (randomness inside the reference, a branch on the dtype): recorded unavailable.
REF_SANITY = 0.25
STRESS_PROFILES = ("offset", "spikes", "amp")
SPIKE_FRACTION = 1e-3
SPIKE_VALUE = 12.0
_LOW_FLOATS = (torch.float32, torch.float16, torch.bfloat16)
_MAX_CHECKS = 48


def unit_roundoff(dtype: torch.dtype) -> float | None:
    """u = 2^-p (p = significand bits incl. the implicit one); None for non-floats."""
    bits = {torch.float64: 53, torch.float32: 24, torch.float16: 11, torch.bfloat16: 8,
            torch.float8_e4m3fn: 4, torch.float8_e5m2: 3}.get(dtype)
    return None if bits is None else 2.0 ** -bits


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


# --- the truth -------------------------------------------------------------------------

class _Upcast(TorchFunctionMode):
    """Every float32 / float16 / bfloat16 dtype a function asks for becomes ``to``."""

    def __init__(self, to: torch.dtype):
        super().__init__()
        self.to = to

    def _lift(self, d):
        return self.to if isinstance(d, torch.dtype) and d in _LOW_FLOATS else d

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        name = getattr(func, "__name__", "")
        if name in ("float", "half", "bfloat16") and args and isinstance(args[0], torch.Tensor) \
                and args[0].is_floating_point():
            return args[0].to(self.to)
        if "dtype" in kwargs:
            kwargs["dtype"] = self._lift(kwargs["dtype"])
        if name in ("to", "type") and len(args) > 1:
            args = (args[0], *[self._lift(a) for a in args[1:]])
        return func(*args, **kwargs)


def _lift_input(x, to):
    # Always a copy: an in-place reference must not write into the caller's inputs (the
    # float32 retry after a float64 failure reads them again).
    if isinstance(x, torch.Tensor):
        return x.to(to, copy=True) if x.dtype in _LOW_FLOATS else x.clone()
    if isinstance(x, (list, tuple)):
        return type(x)(_lift_input(v, to) for v in x)
    return x


class _Saved:
    __slots__ = ("host", "device", "requires_grad")

    def __init__(self, t: torch.Tensor):
        self.host = t.detach().to("cpu", copy=True)
        self.device = t.device
        self.requires_grad = t.requires_grad


def snapshot(inputs) -> list:
    """A host copy of ``inputs`` (tensors, possibly inside lists / tuples), to be taken
    BEFORE the reference or the kernel runs on them: a definition that writes into an
    input (a residual, a KV cache) leaves it changed, and a truth computed from the
    changed inputs is the truth of another problem. ``restore`` gives it back."""
    def take(x):
        if isinstance(x, torch.Tensor):
            return _Saved(x)
        if isinstance(x, (list, tuple)):
            return type(x)(take(v) for v in x)
        return x
    return [take(x) for x in inputs]


def restore(saved) -> list:
    """Fresh tensors on their original devices from a ``snapshot``."""
    def give(x):
        if isinstance(x, _Saved):
            return x.host.to(x.device, copy=True).requires_grad_(x.requires_grad)
        if isinstance(x, (list, tuple)):
            return type(x)(give(v) for v in x)
        return x
    return [give(x) for x in saved]


def upcast_call(fn, inputs, to: torch.dtype = torch.float64):
    """``fn(*inputs)`` with its float inputs and float casts lifted to ``to``; the RNG
    state is restored afterwards (a seeded protocol's next draw must not move)."""
    lifted = [_lift_input(x, to) for x in inputs]
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.no_grad(), _Upcast(to):
        out = fn(*lifted)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out


def truth_of(fn, inputs, normalize=None):
    """(outputs, basis, why) -- the reference's outputs in float64, else float32, else
    (None, "unavailable", the two exceptions). ``normalize`` maps a raw return value to
    the output list (a leaderboard card's definition order)."""
    errors = []
    for to in (torch.float64, torch.float32):
        try:
            out = upcast_call(fn, inputs, to)
            outs = normalize(out) if normalize else _as_list(out)
            return outs, _dtype_name(to), ""
        except Exception as e:  # noqa: BLE001 — a reference op without a kernel for `to`
            errors.append(f"{_dtype_name(to)}: {type(e).__name__}: {str(e)[:160]}")
    return None, "unavailable", "; ".join(errors)


def _as_list(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, dict):
        return list(out.values())
    return list(out)


# --- the measure -----------------------------------------------------------------------

def rel_l2(out: torch.Tensor, truth: torch.Tensor) -> float:
    """||out - truth|| / ||truth|| in float64 over the positions where the truth is
    finite; inf when ``out`` is non-finite there, or differs where the truth is not."""
    y = out.detach().to(torch.float64)
    t = truth.detach().to(torch.float64)
    if y.shape != t.shape:
        return math.inf
    fin = torch.isfinite(t)
    if not bool(fin.all()):
        if bool((y[~fin] != t[~fin]).any()):
            return math.inf
        y, t = y[fin], t[fin]
    if not bool(torch.isfinite(y).all()):
        return math.inf
    num = float(torch.linalg.vector_norm(y - t))
    den = float(torch.linalg.vector_norm(t))
    if den == 0.0:
        return 0.0 if num == 0.0 else math.inf
    return num / den


def limit(e_ref: float, dtype: torch.dtype, k: float = K_RATIO, floor_ulps: float = FLOOR_ULPS,
          floor_rel: float = FLOOR_REL) -> float:
    return k * e_ref + max(floor_ulps * (unit_roundoff(dtype) or 0.0), floor_rel)


def _num(x: float):
    return None if x is None else (x if math.isfinite(x) else "inf")


def stress_tensor(x: torch.Tensor, profile: str, gen: torch.Generator | None = None) -> torch.Tensor:
    """``x`` (a normal draw) under a stress profile; a new tensor of the same dtype."""
    if profile == "offset":
        return (x.float() + 1.0).to(x.dtype)
    if profile == "amp":
        return (x.float() * 4.0).to(x.dtype)
    if profile == "spikes":
        flat = x.detach().float().reshape(-1).clone()
        n = max(1, int(flat.numel() * SPIKE_FRACTION))
        idx = torch.randperm(flat.numel(), device=flat.device, generator=gen)[:n]
        sign = torch.randint(0, 2, (n,), device=flat.device, generator=gen) * 2 - 1
        flat[idx] = sign.float() * SPIKE_VALUE
        return flat.reshape(x.shape).to(x.dtype)
    raise ValueError(f"unknown stress profile {profile!r} (one of {STRESS_PROFILES})")


class PrecisionRecorder:
    """Collects one row per (case, output) and renders the eval JSON's ``precision``
    block. Never raises out of ``measure``: a failure to measure is a row that says so."""

    def __init__(self, k: float = K_RATIO, floor_ulps: float = FLOOR_ULPS,
                 floor_rel: float = FLOOR_REL):
        self.k, self.floor_ulps, self.floor_rel = float(k), float(floor_ulps), float(floor_rel)
        self.rows: list[dict] = []
        self.notes: list[str] = []
        self.bases: set = set()

    def unavailable(self, case: str, why: str, *, stress: str | None = None) -> None:
        self.rows.append({"case": case, "status": "unavailable", "why": why[:400],
                          **({"stress": stress} if stress else {})})

    def measure(self, case: str, outputs, ref_outputs, truth, *, names=None,
                stress: str | None = None) -> None:
        """Compare ``outputs`` (the kernel's) and ``ref_outputs`` (the reference's, native
        dtypes) with ``truth``: a list of tensors, or a zero-argument callable returning
        (list, basis, why) like ``truth_of`` / a list (``upcast_call``)."""
        try:
            basis, why = "float64", ""
            if callable(truth):
                truth = truth()
            if isinstance(truth, tuple) and len(truth) == 3 and isinstance(truth[1], str):
                truth, basis, why = truth
            if truth is None:
                self.unavailable(case, f"the reference could not run upcast: {why}", stress=stress)
                return
            self.bases.add(basis)
            outs, refs, truths = _as_list(outputs), _as_list(ref_outputs), _as_list(truth)
            names = list(names) if names else [f"output[{i}]" for i in range(len(outs))]
            for i, (y, r, t) in enumerate(zip(outs, refs, truths)):
                self._row(case, names[i] if i < len(names) else f"output[{i}]", y, r, t,
                          basis, stress)
        except Exception as e:  # noqa: BLE001 — measuring must never break the judge
            self.unavailable(case, f"{type(e).__name__}: {str(e)[:300]}", stress=stress)

    def _row(self, case, name, y, r, t, basis, stress):
        row = {"case": case, "output": name, **({"stress": stress} if stress else {})}
        if not (isinstance(y, torch.Tensor) and isinstance(r, torch.Tensor)
                and isinstance(t, torch.Tensor)):
            row.update(status="unavailable", why="non-tensor output")
        elif not r.is_floating_point() or unit_roundoff(r.dtype) is None:
            row.update(status="n/a", dtype=_dtype_name(r.dtype),
                       why="not a float output (exact dtype)")
        elif basis == "float32" and r.dtype == torch.float32:
            row.update(status="unavailable", dtype="float32",
                       why="the truth (float32: the reference has no float64 run) is no more "
                           "precise than the output")
        else:
            e_ref, e_cand = rel_l2(r, t), rel_l2(y, t)
            lim = limit(e_ref if math.isfinite(e_ref) else math.inf, r.dtype, self.k,
                        self.floor_ulps, self.floor_rel)
            row.update(dtype=_dtype_name(r.dtype), truth=basis, e_ref=_num(e_ref),
                       e_cand=_num(e_cand), limit=_num(lim),
                       ratio=_num(e_cand / e_ref) if e_ref and math.isfinite(e_ref)
                       and math.isfinite(e_cand) else None)
            if not math.isfinite(e_ref):
                row.update(status="unavailable",
                           why="the reference's own output is not finite against the truth"
                               + (" (the stress overflows the dtype)" if stress else ""))
            elif e_ref > REF_SANITY:
                row.update(status="unavailable",
                           why=f"the reference is {e_ref:.3g} from its own upcast run "
                               f"(randomness or a dtype branch inside it)")
            else:
                row["status"] = "pass" if e_cand <= lim else "fail"
        self.rows.append(row)

    def block(self) -> dict:
        """The JSON-ready ``precision`` block: overall status (fail > pass > unavailable),
        the worst row, and the rows (capped)."""
        judged = [r for r in self.rows if r.get("status") in ("pass", "fail")]
        failed = [r for r in judged if r["status"] == "fail"]
        status = "fail" if failed else ("pass" if judged else "unavailable")

        def margin(r):
            e, lim = r.get("e_cand"), r.get("limit")
            if e == "inf":
                return math.inf
            return (e / lim) if isinstance(e, (int, float)) and isinstance(lim, (int, float)) \
                and lim > 0 else 0.0

        worst = max(judged, key=margin) if judged else None
        out = {"schema": SCHEMA, "metric": METRIC, "k": self.k, "floor_ulps": self.floor_ulps,
               "floor_rel": self.floor_rel,
               "rule": "fail iff e_cand > k*e_ref + max(floor_ulps*u(output dtype), floor_rel); "
                       "e = relative L2 error against the reference run in float64",
               "status": status, "truth": "/".join(sorted(self.bases)) or None,
               "n_checks": len(judged), "n_failed": len(failed),
               "worst": worst, "rows": self.rows[:_MAX_CHECKS]}
        if len(self.rows) > _MAX_CHECKS:
            out["rows_dropped"] = len(self.rows) - _MAX_CHECKS
        if status == "unavailable":
            whys = sorted({r.get("why", "") for r in self.rows if r.get("why")})
            out["why"] = "; ".join(whys)[:600] or "no float output was measured"
        if failed:
            f = failed[0] if worst is None or worst.get("status") != "fail" else worst
            out["why"] = (f"{f['case']}{' [' + f['stress'] + ']' if f.get('stress') else ''} "
                          f"{f['output']}: e_cand={f['e_cand']} > {self.k:g}*e_ref({f['e_ref']}) + "
                          f"floor = {f['limit']}")
        return out


def merge_blocks(named: list) -> dict | None:
    """One block over several workloads' blocks [(name, block), ...] (a group card);
    rows are prefixed with their workload. None when there is no block at all."""
    named = [(n, b) for n, b in named if isinstance(b, dict)]
    if not named:
        return None
    first = named[0][1]
    rec = PrecisionRecorder(first.get("k", K_RATIO), first.get("floor_ulps", FLOOR_ULPS),
                            first.get("floor_rel", FLOOR_REL))
    for name, b in named:
        for r in b.get("rows") or []:
            rec.rows.append({**r, "case": f"{name}/{r.get('case')}"})
        if b.get("truth"):
            rec.bases.update(str(b["truth"]).split("/"))
    return rec.block()


@contextlib.contextmanager
def no_tf32():
    """Run a block with tf32 off (the reference's own native run must not pick it up
    from a global flag the kernel set)."""
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m, c
