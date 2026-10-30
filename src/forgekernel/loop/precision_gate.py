"""The loop's side of the precision gate (the measure is src/forgekernel/card_template/eval/precision.py).

Every eval JSON may carry a top-level ``precision`` block: per (input set, output) the
relative L2 error of the kernel (e_cand) and of the reference itself (e_ref) against the
reference re-run in float64, and whether e_cand stays within k*e_ref + floor. The bench
never changes ``correctness.pass`` for it; what a failed check costs is the card's
``[loop] precision_contract``:

    strict    a failed check is a correctness failure: REVERT, the record's error names
              the precision gate and the worst ratio. The default for a card that judges
              with its own eval/bench.py (its tolerance is the author's, and nothing else
              would see an fp32 accumulation dropped to bf16 inside it).
    official  recorded only (job JSON, run log, iteration record).
    off       not read, and on a leaderboard card not measured either: the DEFAULT for a
              leaderboard card, whose verdict must be the board's alone -- its bench runs
              no float64 re-run and no stressed inputs. A leaderboard card turns the
              measure on with ``precision_contract`` in eval/leaderboard/problem.json
              (``strict`` / ``official``), which is also its default here.

A bench that reports no block under ``strict`` cannot be held to it: a fresh start over
such a card is refused (``bench_findings``; a resumed run only warns), and at run time the
row records ``unreported``. Once the bench is wired to report it, an eval without the
block under ``strict`` is a correctness failure (``MISSING_BLOCK``): the kernel runs in
the bench's process and could have suppressed it.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

STRICT = "strict"
OFFICIAL = "official"
OFF = "off"
CONTRACTS = (STRICT, OFFICIAL, OFF)
LEADERBOARD_PROBLEM = Path("eval") / "leaderboard" / "problem.json"
HELPER = Path("eval") / "precision.py"
MISSING_BLOCK = ("precision gate (strict): the eval JSON carries no `precision` block although "
                 "the card's bench reports one — an eval that skipped the measurement (e.g. "
                 "FK_PRECISION switched off from inside the kernel's process) cannot pass a "
                 "strict card; leave the bench's environment alone")


def is_leaderboard_card(op_dir: Path) -> bool:
    return (Path(op_dir) / LEADERBOARD_PROBLEM).is_file()


def card_contract(op_dir: Path) -> str | None:
    """A leaderboard card's own ``precision_contract`` (eval/leaderboard/problem.json), the
    switch its bench measures by; None when the card does not set one."""
    try:
        raw = json.loads((Path(op_dir) / LEADERBOARD_PROBLEM).read_text(encoding="utf-8")) \
            .get("precision_contract")
    except (OSError, ValueError, AttributeError):
        return None
    return str(raw).strip().lower() if raw is not None else None


def resolve_contract(cfg: dict, op_dir: Path) -> tuple[str, bool]:
    """([loop] precision_contract, else the card's default, and whether it was defaulted).
    A leaderboard card defaults to its problem.json ``precision_contract``, else off (its
    verdict is the board's; the bench then measures nothing). ValueError on anything
    outside CONTRACTS -- a typo must not read as off."""
    raw = (cfg.get("loop", {}) or {}).get("precision_contract")
    if raw is None:
        if is_leaderboard_card(op_dir):
            raw = card_contract(op_dir) or OFF
            if raw not in CONTRACTS:
                raise ValueError(f"{LEADERBOARD_PROBLEM} precision_contract must be one of "
                                 f"{', '.join(CONTRACTS)}, got {raw!r}")
            return raw, True
        return STRICT, True
    value = str(raw).strip().lower()
    if value not in CONTRACTS:
        raise ValueError(f"[loop] precision_contract must be one of {', '.join(CONTRACTS)}, "
                         f"got {raw!r}")
    return value, False


def _calls_recorder(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name == "PrecisionRecorder":
                return True
    return False


def bench_findings(op_dir: Path) -> list[str]:
    """Why the card's bench cannot report the precision block ([] = it can). A
    leaderboard card's bench is generated (eval/leaderboard.py measures); only the helper
    file has to be there. A hand-written bench must call PrecisionRecorder itself."""
    op_dir = Path(op_dir)
    out = []
    if is_leaderboard_card(op_dir) and card_contract(op_dir) not in (STRICT, OFFICIAL):
        out.append(f"{LEADERBOARD_PROBLEM} does not set precision_contract to \"strict\" / "
                   f"\"official\" — its bench measures nothing (the leaderboard default is off)")
    if not (op_dir / HELPER).is_file():
        out.append(f"{HELPER} is missing — copy it from src/forgekernel/card_template/eval/"
                   + (" (with the current eval/leaderboard.py)" if is_leaderboard_card(op_dir)
                      else ""))
    if not is_leaderboard_card(op_dir):
        bench = op_dir / "eval" / "bench.py"
        try:
            src = bench.read_text(encoding="utf-8")
        except OSError:
            src = ""
        if not _calls_recorder(src):
            out.append("eval/bench.py never calls precision.PrecisionRecorder — the eval JSON "
                       "will carry no `precision` block (see eval/precision.py's docstring)")
    return out


def _brief_row(r: dict | None) -> dict | None:
    if not r:
        return None
    return {k: r[k] for k in ("case", "stress", "output", "dtype", "e_ref", "e_cand", "limit",
                              "ratio") if k in r}


def summarize(contract: str, block) -> dict:
    """The iteration record's compact ``precision`` field."""
    if contract == OFF:
        return {}
    if not isinstance(block, dict):
        return {"contract": contract, "status": "unreported"}
    rows = [r for r in block.get("rows") or [] if isinstance(r, dict)]
    seen: dict = {}
    for r in rows:
        if r.get("stress"):
            seen.setdefault(r["stress"], []).append(r.get("status"))
    stress = {p: ("fail" if "fail" in st else "pass" if "pass" in st else st[0])
              for p, st in seen.items()}
    out = {"contract": contract, "status": block.get("status"), "truth": block.get("truth"),
           "k": block.get("k"), "floor_ulps": block.get("floor_ulps"),
           "floor_rel": block.get("floor_rel"),
           "n_checks": block.get("n_checks"), "n_failed": block.get("n_failed"),
           "worst": _brief_row(block.get("worst"))}
    if stress:
        out["stress"] = stress
    if block.get("why"):
        out["why"] = str(block["why"])[:500]
    return {k: v for k, v in out.items() if v is not None}


def gate_failure(contract: str, summary: dict) -> str | None:
    """The REVERT text when the contract makes this record a correctness failure."""
    if contract != STRICT or summary.get("status") != "fail":
        return None
    w = summary.get("worst") or {}
    ratio = w.get("ratio")
    where = f"{w.get('case')}{' [stress ' + w['stress'] + ']' if w.get('stress') else ''} {w.get('output')}"
    return (f"precision gate (strict): {where}: the kernel's output is "
            f"{'x' + format(ratio, '.3g') + ' ' if isinstance(ratio, (int, float)) else ''}"
            f"farther from the float64 truth than the reference's own (e_cand {w.get('e_cand')} > "
            f"{summary.get('k')}*e_ref({w.get('e_ref')}) + floor = {w.get('limit')}). Correctness within the tolerance is not enough on this card: "
            f"keep the reference's precision — fp32 accumulation, no tf32 / fp8 / reduced-"
            f"precision intermediates or approximate math the reference does not use"
            + (f"; failed {summary.get('n_failed')} of {summary.get('n_checks')} checks"
               if summary.get("n_failed") else ""))


def describe(summary: dict) -> str:
    """One log fragment: ``precision=pass (worst x1.2, truth float64)``."""
    if not summary:
        return ""
    st = summary.get("status")
    w = summary.get("worst") or {}
    bits = [f"{summary.get('contract')}"]
    if isinstance(w.get("ratio"), (int, float)):
        bits.append(f"worst e_cand/e_ref x{w['ratio']:.3g}")
    if summary.get("truth"):
        bits.append(f"truth {summary['truth']}")
    if summary.get("stress"):
        bits.append("stress " + ",".join(f"{k}:{v}" for k, v in summary["stress"].items()))
    text = f"precision={st} ({'; '.join(bits)})"
    if st in ("fail", "unavailable") and summary.get("why"):
        text += f" — {summary['why'][:200]}"
    return text


# --- the SASS cross-check (warning only) ------------------------------------------------
#
# The error ratio reads what the kernel computed; the census reads what it was compiled
# to compute with. The per-iteration orchestration census (recon/orch_census_run.py)
# carries the kernel's MMA operand / accumulator types (orch_tools.mma_precision); a
# reference dossier built since the field exists carries the reference's. A drop below
# the reference's -- or below the card's own input dtypes -- is logged and recorded
# beside the ratio, never gated: tcgen05 keeps its kind in a register, and a kernel may
# legitimately split work across MMA kinds the census cannot weigh.
_INPUT_RANK = {"f64": 7, "f32": 6, "tf32": 5, "f16": 4, "bf16": 4, "f16|bf16|tf32": 4,
               "e4m3": 2, "e5m2": 2, "f8": 2, "f8f6f4": 2, "s8": 2, "u8": 2, "i8": 2,
               "e3m2": 1, "e2m3": 1, "e2m1": 1, "mxf4": 1, "s4": 1, "u4": 1, "b1": 0}
_ACC_RANK = {"f64": 3, "f32": 2, "s32": 2, "f16": 1}
_DTYPE_RANK = {"float64": 7, "float32": 6, "float16": 4, "bfloat16": 4, "float8_e4m3fn": 2,
               "float8_e5m2": 2, "float4_e2m1": 1, "float4_e2m1fn_x2": 1}


def _lowest(names, rank) -> "tuple[str, int] | None":
    known = [(n, rank[n]) for n in names or () if n in rank]
    return min(known, key=lambda x: x[1]) if known else None


def sass_findings(student: dict | None, reference: dict | None = None,
                  input_dtypes=()) -> dict:
    """{"student", "reference", "warnings"} for the record; {} when the census carried no
    MMA reading for the student."""
    if not isinstance(student, dict) or not student.get("ops"):
        return {}
    warnings = []
    s_in = _lowest(student.get("inputs"), _INPUT_RANK)
    s_acc = _lowest(student.get("accums"), _ACC_RANK)
    ref_ok = isinstance(reference, dict) and reference.get("ops")
    if ref_ok:
        r_in = _lowest(reference.get("inputs"), _INPUT_RANK)
        r_acc = _lowest(reference.get("accums"), _ACC_RANK)
        if s_in and r_in and s_in[1] < r_in[1]:
            warnings.append(f"MMA inputs {s_in[0]} below the reference's {r_in[0]}")
        if s_acc and r_acc and s_acc[1] < r_acc[1]:
            warnings.append(f"MMA accumulates in {s_acc[0]}, the reference in {r_acc[0]}")
    d_in = _lowest([str(d) for d in input_dtypes], _DTYPE_RANK)
    if s_in and d_in and s_in[1] < d_in[1]:
        warnings.append(f"MMA inputs {s_in[0]} below the card's {d_in[0]} inputs")
    if s_acc and s_acc[0] == "f16" and not any("accumulates" in w for w in warnings):
        warnings.append("MMA accumulates in f16")
    out = {"student": {k: student.get(k) for k in ("inputs", "accums", "ops")},
           "reference": ({k: reference.get(k) for k in ("inputs", "accums")} if ref_ok
                         else None),
           "warnings": warnings}
    return out
