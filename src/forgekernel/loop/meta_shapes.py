"""A group card's meta readings, one workload at a time.

A group card (problem.json ``group``) is one kernel tuned on a size class of workloads;
its gate is the geomean over all of them. The loop's meta tools time many variants per
job (lever pricing) or profile one launch target (stall / SASS pricing, the timeline
probe), so they read the card one workload at a time (FK_LB_WORKLOAD_SCOPE). They used to
read the representative alone, and an audit of 59 earlier group-card runs (3024 evals) found
the representative moved the same way as the rest of the group only 61% of the times the
rest moved by >= 2% (28% it stayed flat, 11% it moved the other way) -- a price on one
shape is not a price on the group.

So they read problem.json ``meta_workloads`` (``forge new`` writes them: the
representative, the group's smallest and its largest -- the middle one in their place
when the representative is an end): lever pricing runs one job per
workload and this module merges the per-workload tables into the one table every
consumer already reads -- each delta the geomean over the workloads, the noise floor
of that mean, the class recomputed on the merged readings, and ``shape_split`` when a lever
ADDED ALONE moves one workload up and another down beyond each one's own floor (a size-dependent
lever: averaged, it would read null and close a phase it should have opened). Only the add
leg splits: a remove leg that disagrees across workloads says how the lever couples with the
rest of the stack at each size, not that the lever pays on some of them (4 of the 8
splits in that audit were remove-only, one of them a lever that slowed all three
workloads when added -- harmful, read as size-dependent). The
per-workload rows ride along (``*-SHAPE`` lines) for the teacher. Stall / SASS pricing
and the timeline probe price the representative as before (every machine reading stays
on it) and the other meta workloads in their own jobs, rendered beside it.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from urllib.parse import unquote

PROBLEM_FILE = Path("eval") / "leaderboard" / "problem.json"
# per-workload rounds when a table is priced on several workloads: the merged reading
# pools them (2 rounds x 3 workloads = 6 readings per variant, against 3 on one)
META_ROUNDS = 2
NOISE_FLOOR_PCT = 0.5            # == lever_pricing_run.NOISE_FLOOR_PCT
ENTRY_CLOSE = "[gpu_entry.sh] task=cmd finished"

_CUBIN_SAME_RE = re.compile(r"^LEVER (\S+) .*\bcubin_same_as_base=(\S+)", re.M)
_HEADER_RE = re.compile(r"^PRICE kernel_file=.*$", re.M)
_ABL_NOISE_RE = re.compile(r"^PRICE ablation_noise_floor=([\d.]+)%", re.M)
_ROUNDS_RE = re.compile(r"^PRICE-(?:OK|TIMED) levers=\d+ rounds=(\d+)", re.M)
_SHAPE_LINE_RE = re.compile(r"^(LEVER|VARIANT|ABLATE|STACK)-SHAPE (.*)$", re.M)
_SHAPES_RE = re.compile(r"^PRICE-SHAPES (.*)$", re.M)
# the workloads whose ablation pass did not close (merged without them): wl:why, why =
# "cut" (no ABLATE verdict: the job's clock or a kill ended it first) or "fail:<reason>"
_ABL_SHAPES_RE = re.compile(r"^ABLATE-SHAPES missing=(\S+)", re.M)

# Text the driver writes into its timeline tables and stall-ledger blocks that this
# module parses back (_timeline_lean, _ledger_head). The driver builds those lines from
# these constants, so producer and parser cannot drift apart.
# a stall-ledger warning line ("(WARNING: ledger self-check failed — ...)")
LEDGER_WARNING_PREFIX = "(WARNING"
# line prefixes of the per-run caveats under a timeline table ("**Coverage note (partial)**: ...")
TL_COVERAGE_NOTE = "**Coverage note"
TL_SELFCHECK_NOTE = "**Probe self-check note"
# the tail of a role's segment-table heading ("**<role>** " + this)
TL_SEGMENT_LEDGER_HEADING = ("segment ledger (execution order; wait = `.begin`→`.end` pair; "
                             "wrap = this block's last site → next block's first site)")


def card_meta_workloads(tree) -> list[str]:
    """The meta workloads of the card checked out at ``tree`` (problem.json
    ``meta_workloads``, the representative first) when there are at least two, else [] --
    a one-workload card, or a group card built before the field (its meta tools read the
    representative, and its card runtime would not understand a named scope)."""
    try:
        problem = json.loads((Path(tree) / PROBLEM_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    # a meta workload may be one of the card's correctness-only workloads (a scoped reading
    # times it alone)
    names = [str(g.get("name")) for g in (problem.get("group") or []) + (problem.get("check_only") or [])]
    meta = [str(m) for m in problem.get("meta_workloads") or [] if str(m) in names]
    return meta if len(meta) >= 2 else []


def card_workload_axes(tree) -> dict:
    """{group entry name: its axes} of the card checked out at ``tree`` ({} when none)."""
    try:
        problem = json.loads((Path(tree) / PROBLEM_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(g.get("name")): (g.get("workload") or {}).get("axes") or {}
            for g in (problem.get("group") or []) + (problem.get("check_only") or [])}


def rounds_for(rounds: int, n_workloads: int) -> int:
    """Rounds per job when a table is priced on ``n_workloads`` workloads."""
    return rounds if n_workloads < 2 else min(rounds, META_ROUNDS)


# --- merging per-workload price tables ---------------------------------------------

def _geo_pct(vals):
    """The geomean of per-workload deltas (%), None when any is None."""
    if not vals or any(v is None for v in vals):
        return None
    try:
        return 100.0 * (math.exp(sum(math.log1p(v / 100.0) for v in vals) / len(vals)) - 1.0)
    except ValueError:                 # a -100% reading: no geomean
        return None


def _geo(vals):
    if not vals or any(v is None or v <= 0 for v in vals):
        return None
    return round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 4)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def merged_floor(floors) -> float:
    """The noise floor of a mean over workloads read independently, each at its own
    floor (3 sd of its delta): 3 sd of the mean = sqrt(sum f^2) / k, never below the
    tool's own floor."""
    fs = [float(f) for f in floors if f is not None]
    if not fs:
        return NOISE_FLOOR_PCT
    return max(NOISE_FLOOR_PCT, math.sqrt(sum(f * f for f in fs)) / len(fs))


def is_split(readings, floors) -> bool:
    """A leg that moves one workload up and another down, each beyond its own floor (the
    merge asks it of the add leg only)."""
    pairs = [(r, f) for r, f in zip(readings, floors) if r is not None]
    return any(r > f for r, f in pairs) and any(r < -f for r, f in pairs)


def _pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


def _tok(x) -> str:
    return re.sub(r"\s+", "_", str(x)).strip("_") or "none"


def _enc_reasons(reasons: dict | None) -> str:
    """{workload: why} as one whitespace-free token, ``wl:why;wl:why`` (``none`` when
    empty). A reason is free text -- a BackendError's message, a tool's PRICE-FAIL
    reason -- and may hold ``;`` itself, which used to cut the list where it stood
    (and the rest of that reason read as a workload): ``%`` and ``;`` are escaped as
    ``%25`` / ``%3B`` and _dec_reasons undoes it."""
    return ";".join(f"{k}:{_tok(v).replace('%', '%25').replace(';', '%3B')}"
                    for k, v in (reasons or {}).items()) or "none"


def _dec_reasons(tok: str | None) -> dict:
    """_enc_reasons' token back into {workload: why}."""
    out = {}
    for part in (tok or "none").split(";"):
        k, sep, v = part.partition(":")
        if sep and k:
            out[k] = unquote(v)
    return out


def merge_lever_logs(shapes: list, *, missing: dict | None = None) -> str:
    """One lever_pricing_run log from per-workload jobs: ``shapes`` = [(workload name, log,
    job id)], the representative first, every log a table with a verdict. The merged log
    carries the lines the driver parses (LEVER / VARIANT / STACK / ABLATE / PRICE / the
    verdicts, ending with the entry's closing line so a timing-only copy still counts),
    each reading merged over the workloads, plus the per-workload rows and PRICE-SHAPES
    (the workloads, their jobs, the ones ``missing`` maps to why). The ablations merge
    over the workloads whose ablation pass closed (ABLATE-OK); the others are named on
    ABLATE-SHAPES with why ("cut": no verdict, the job's clock or a kill ended it first;
    "fail:<reason>": ABLATE-FAIL) -- a missing reading, never a flat one."""
    from forgekernel.loop.driver import (_ABLATE_FAIL_RE, _ABLATE_OK_RE, _PRICE_OK_RE,
                                         _PRICE_SCOPE_RE, parse_lever_prices)
    from forgekernel.skills.lever_pricing_run import lever_class

    per = []
    for wl, log, job in shapes:
        rows, stack, _ = parse_lever_prices(log)
        abl_noise = _ABL_NOISE_RE.findall(log)
        rounds = _ROUNDS_RE.findall(log)
        scope = list(_PRICE_SCOPE_RE.finditer(log))
        afail = _ABLATE_FAIL_RE.search(log)
        per.append({
            "wl": wl, "job": str(job or ""), "stack": stack,
            "levers": {r["name"]: r for r in rows if r.get("kind", "lever") == "lever"},
            "variants": {r["name"]: r for r in rows if r.get("kind") == "variant"},
            "ablations": {r["name"]: r for r in rows if r.get("kind") == "ablation"},
            "same": dict(_CUBIN_SAME_RE.findall(log)),
            "price_ok": bool(_PRICE_OK_RE.search(log)),
            "ablate_ok": bool(_ABLATE_OK_RE.search(log)),
            "ablate_fail": afail.group(1).strip() if afail else None,
            "abl_noise": float(abl_noise[-1]) if abl_noise else NOISE_FLOOR_PCT,
            "rounds": rounds[-1] if rounds else "?",
            "of": scope[-1].group(3) if scope else "?",
            "noise": (stack or {}).get("noise_pct") or NOISE_FLOOR_PCT,
            "metric": (stack or {}).get("metric"),
            "header": (m.group(0) if (m := _HEADER_RE.search(log)) else None)})
    rep = per[0]
    names = [p["wl"] for p in per]
    out: list[str] = []
    if rep["header"]:
        out.append(rep["header"])
    floors = [p["noise"] for p in per]
    noise = merged_floor(floors)

    lever_rows = {}
    for name, r0 in rep["levers"].items():
        rows = [p["levers"].get(name) for p in per]
        if any(r is None for r in rows):
            rows = [r for r in rows if r is not None]
        adds = [r["add_pct"] for r in rows]
        rems = [r["remove_pct"] for r in rows]
        add, rem = _geo_pct(adds), _geo_pct(rems)
        fl = [p["noise"] for p in per if name in p["levers"]]
        classes = [r["cls"] for r in rows]
        same = all(p["same"].get(name) == "True" for p in per if name in p["levers"])
        if "broken" in classes:
            cls = "broken"
        elif is_split(adds, fl):
            cls = "shape_split"
        else:
            cls = lever_class(add, rem, noise=noise, ok_add=True, same=same,
                              all_in_spec=any(r is not None for r in rems) or "unstacked" in classes,
                              all_ok="unstacked" not in classes)
        lever_rows[name] = add
        out.append(f"LEVER {name} add={_pct(add)} remove={_pct(rem)} "
                   f"add_ms={_geo([r['add_ms'] for r in rows])} remove_ms={_geo([r['remove_ms'] for r in rows])} "
                   f"class={cls} spin_add={_mean([r['spin_add_ns'] for r in rows])} "
                   f"spin_all_minus={_mean([r['spin_all_minus_ns'] for r in rows])} "
                   f"cubin_same_as_base={same}")
    for name, v0 in rep["variants"].items():
        rows = [p["variants"].get(name) for p in per]
        if any(r is None for r in rows) or any(r["levers"] != v0["levers"] for r in rows):
            continue      # all-pos on a different harmful set per workload: per-workload rows only
        out.append(f"VARIANT {name} levers={','.join(v0['levers'])} "
                   f"delta={_pct(_geo_pct([r['delta_pct'] for r in rows]))} "
                   f"ms={_geo([r['ms'] for r in rows])} spin={_mean([r['spin_ns'] for r in rows])} "
                   f"q={v0['question']}")
    stacks = [p["stack"] for p in per]
    if all(stacks):
        s0 = stacks[0]
        all_d = _geo_pct([s["all_pct"] for s in stacks])
        singles = [a for a in lever_rows.values() if a is not None]
        total = sum(singles)
        pos_same = all(s["all_pos_levers"] == s0["all_pos_levers"] and s["dropped"] == s0["dropped"]
                       for s in stacks)
        pos = _geo_pct([s["all_pos_pct"] for s in stacks]) if pos_same else None
        out.append(f"STACK all={_pct(all_d)} sum_singles={_pct(total)} "
                   f"coupling={_pct(None if all_d is None else all_d - total)} all_pos={_pct(pos)} "
                   f"all_pos_levers={','.join(s0['all_pos_levers']) if pos_same and s0['all_pos_levers'] else 'none'} "
                   f"dropped={','.join(s0['dropped']) if pos_same and s0['dropped'] else 'none'}")
        out.append(f"PRICE-SCOPE workloads={','.join(names)} rep={rep['wl']} of={rep['of']}")
        metric = "vs_ref_device" if all(p["metric"] == "vs_ref_device" for p in per) else "device_ms"
        out.append(f"PRICE noise_floor={noise:.2f}% metric={metric} sigma=n/a rounds={rep['rounds']} "
                   f"outliers=none")
        n_levers = len(rep["levers"])
        if all(p["price_ok"] for p in per):
            out.append(f"PRICE-OK levers={n_levers} rounds={rep['rounds']} "
                       f"base_ms={_geo([s.get('base_ms') for s in stacks])} "
                       f"all_ms={_geo([s.get('all_ms') for s in stacks])} "
                       f"spin_base={_mean([s.get('spin_base_ns') for s in stacks])} "
                       f"spin_all={_mean([s.get('spin_all_ns') for s in stacks])} "
                       f"tensor_base={_mean([s.get('tensor_base') for s in stacks])} "
                       f"tensor_all={_mean([s.get('tensor_all') for s in stacks])}")
        else:             # a role pass cut by the job clock on some workload: the timing copy
            out.append(f"PRICE-TIMED levers={n_levers} rounds={rep['rounds']}")
    abl_shapes = [p for p in per if p["ablate_ok"]]
    # every job ran the same levers.json: a workload whose ablation pass did not close
    # is a reading that is missing, not one that is flat -- named with why
    abl_missing = {p["wl"]: (f"fail:{p['ablate_fail']}" if p["ablate_fail"] else "cut")
                   for p in per if not p["ablate_ok"]}
    if rep["ablations"] and rep["ablate_ok"]:
        for name, a0 in rep["ablations"].items():
            rows = [p["ablations"].get(name) for p in abl_shapes]
            rows = [r for r in rows if r is not None]
            status = next((r["status"] for r in rows if r["status"] != "ok"), "ok")
            corr = next((r["correctness"] for r in rows if r["correctness"] != "pass"), "pass")
            out.append(f"ABLATE {name} delta={_pct(_geo_pct([r['delta_pct'] for r in rows]))} "
                       f"ncu_ns={_geo([r['ncu_ns'] for r in rows])} "
                       f"base_ncu_ns={_geo([r['base_ncu_ns'] for r in rows])} correctness={corr} "
                       f"device_delta={_pct(_geo_pct([r['device_delta_pct'] for r in rows]))} "
                       f"kernel={_tok(a0['kernel'])} status={status} h={a0['hypothesis']}")
        out.append(f"PRICE ablation_noise_floor={merged_floor([p['abl_noise'] for p in abl_shapes]):.2f}% "
                   f"metric=ncu_ns")
        base_ns = next((r["base_ncu_ns"] for r in rep["ablations"].values() if r["base_ncu_ns"]), None)
        out.append(f"ABLATE-OK ablations={len(rep['ablations'])} rounds=? base_ncu_ns={base_ns}")
        if abl_missing:
            out.append(f"ABLATE-SHAPES missing={_enc_reasons(abl_missing)}")
    elif rep["ablate_fail"]:
        out.append(f"ABLATE-FAIL reason={rep['ablate_fail']}")
    # the per-workload rows (the teacher reads them beside the merged ones)
    for p in per:
        for name, r in p["levers"].items():
            out.append(f"LEVER-SHAPE {name} wl={p['wl']} add={_pct(r['add_pct'])} "
                       f"remove={_pct(r['remove_pct'])} class={r['cls']} noise={p['noise']:.2f}% "
                       f"spin_add={r['spin_add_ns']}")
        for name, r in p["variants"].items():
            out.append(f"VARIANT-SHAPE {name} wl={p['wl']} delta={_pct(r['delta_pct'])} "
                       f"levers={','.join(r['levers']) or 'none'}")
        if p in abl_shapes:
            for name, r in p["ablations"].items():
                out.append(f"ABLATE-SHAPE {name} wl={p['wl']} delta={_pct(r['delta_pct'])} "
                           f"status={r['status']} correctness={r['correctness']} noise={p['abl_noise']:.2f}%")
        if (s := p["stack"]):
            out.append(f"STACK-SHAPE wl={p['wl']} all={_pct(s['all_pct'])} all_pos={_pct(s['all_pos_pct'])} "
                       f"dropped={','.join(s['dropped']) or 'none'} base_ms={s.get('base_ms')} "
                       f"noise={p['noise']:.2f}% metric={p['metric']}")
    out.append(f"PRICE-SHAPES workloads={','.join(names)} rep={rep['wl']} "
               f"jobs={','.join(p['job'] or '?' for p in per)} rounds={rep['rounds']} "
               f"missing={_enc_reasons(missing)}")
    out.append(ENTRY_CLOSE)
    return "\n".join(out) + "\n"


def _kv(text: str) -> dict:
    return dict(m.groups() for m in re.finditer(r"(\w+)=(\S+)", text))


def _num(tok):
    if tok in (None, "n/a", "None", "?"):
        return None
    try:
        return float(str(tok).rstrip("%"))
    except ValueError:
        return None


def attach_shape_rows(log: str, rows: list, stack: dict | None) -> None:
    """Hang a merged table's per-workload rows (``*-SHAPE`` / PRICE-SHAPES lines) on the
    parsed rows: ``row["shapes"]`` = {workload: {...}}, ``stack["shapes"]`` and
    ``stack["meta_shapes"]`` = {"workloads", "rep", "jobs", "missing"}, and on every
    ablation row ``row["shapes_missing"]`` = {workload: why} (ABLATE-SHAPES). A
    one-workload table has none of these lines and is left as it is."""
    if not log or "-SHAPE " not in log:
        return
    by_kind = {"LEVER": {}, "VARIANT": {}, "ABLATE": {}}
    for r in rows:
        kind = {"lever": "LEVER", "variant": "VARIANT", "ablation": "ABLATE"}.get(r.get("kind", "lever"))
        if kind:
            by_kind[kind][r["name"]] = r
    for m in _SHAPE_LINE_RE.finditer(log):
        kind, rest = m.group(1), m.group(2)
        if kind == "STACK":
            kv = _kv(rest)
            if stack is not None:
                stack.setdefault("shapes", {})[kv.get("wl", "?")] = {
                    "all_pct": _num(kv.get("all")), "all_pos_pct": _num(kv.get("all_pos")),
                    "dropped": [x for x in kv.get("dropped", "none").split(",") if x and x != "none"],
                    "base_ms": _num(kv.get("base_ms")), "noise_pct": _num(kv.get("noise")),
                    "metric": kv.get("metric")}
            continue
        name, _, rest = rest.partition(" ")
        kv = _kv(rest)
        row = by_kind[kind].get(name)
        if row is None:
            continue
        rec = {k: (_num(v) if k in ("add", "remove", "delta", "noise", "spin_add") else v)
               for k, v in kv.items() if k != "wl"}
        row.setdefault("shapes", {})[kv.get("wl", "?")] = rec
    if (am := list(_ABL_SHAPES_RE.finditer(log))):
        abl_missing = _dec_reasons(am[-1].group(1))
        for row in by_kind["ABLATE"].values():
            row["shapes_missing"] = dict(abl_missing)
    if stack is not None and (sm := list(_SHAPES_RE.finditer(log))):
        kv = _kv(sm[-1].group(1))
        stack["meta_shapes"] = {
            "workloads": [x for x in kv.get("workloads", "").split(",") if x],
            "rep": kv.get("rep"),
            "jobs": [x for x in kv.get("jobs", "").split(",") if x],
            "missing": _dec_reasons(kv.get("missing"))}


# --- rendering ----------------------------------------------------------------------

def scope_note(stack: dict | None) -> str:
    """The price table's workload-scope paragraph for the teacher, or ''."""
    sc = (stack or {}).get("workload_scope")
    if not sc:
        return ""
    wls = str(sc.get("workloads") or "")
    if wls == "representative":
        return (f"\n**Multi-workload tree: this table is timed only on the representative workload "
                f"`{sc.get('rep')}`** (of {sc.get('of')} in the group; the whole group × rounds × "
                f"variants would overrun the job clock). The price tags are readings on the "
                f"representative shape, not the group geomean. The KEEP gate is still judged on the "
                f"group geomean, and a lever that pays on the representative shape may not pay on the "
                f"other shapes.\n")
    names = [x for x in wls.split(",") if x]
    meta = (stack or {}).get("meta_shapes") or {}
    miss = meta.get("missing") or {}
    gone = (f"Workloads left unpriced: {'; '.join(f'`{k}` ({v})' for k, v in miss.items())}. "
            if miss else "")
    if len(names) < 2:
        # every other meta workload's job failed: what is left is the representative's
        # own table, and it must not read as a merge
        return (f"\n**Multi-workload tree: this table was planned as one pricing job on each of the "
                f"operator tree's {len(names) + len(miss)} meta workloads, then a merge, but only the "
                f"representative `{sc.get('rep')}` got priced** (of {sc.get('of')} in the group): "
                f"the price tags are readings on the representative shape, not a geomean — a lever "
                f"that pays on the representative shape may not pay on the other shapes. {gone}"
                f"The KEEP gate is still judged on the group geomean.\n")
    return (f"\n**Multi-workload tree: this table was priced with one job on each of {len(names)} "
            f"workloads ({', '.join(f'`{n}`' for n in names)}; the representative `{sc.get('rep')}` "
            f"first, of {sc.get('of')} in the group) and merged**: "
            f"every number is the geomean of these {len(names)} workloads' readings, the noise floor "
            f"is the floor of that mean, and the class is re-derived from the merged readings; the "
            f"per-workload readings are under the table. {gone}"
            "The KEEP gate is still judged on the group geomean.\n")


def scope_log(stack: dict | None) -> str:
    """The pricing log line's scope clause, or ''."""
    sc = (stack or {}).get("workload_scope")
    if not sc:
        return ""
    if sc.get("workloads") == "representative":
        return f"; priced on the representative workload {sc.get('rep')} only (of {sc.get('of')})"
    if "," not in str(sc.get("workloads") or ""):
        return (f"; priced on the representative workload {sc.get('rep')} only -- the other "
                f"meta workloads' jobs left no table (of {sc.get('of')})")
    return f"; merged over workloads {sc.get('workloads')} (one job each; of {sc.get('of')})"


def shape_block(table: list) -> str:
    """The per-workload rows of a merged price table, with the ``shape_split`` rule, or ''."""
    stack = next((t for t in table if t.get("kind") == "stack"), None) or {}
    levers = [t for t in table if t.get("kind") == "lever" and t.get("shapes")]
    others = [t for t in table if t.get("kind") in ("variant", "ablation") and t.get("shapes")]
    names = (stack.get("meta_shapes") or {}).get("workloads") or sorted(
        {w for t in levers + others for w in [*t["shapes"], *(t.get("shapes_missing") or {})]})
    if not (levers or others or stack.get("shapes")) or len(names) < 2:
        return ""          # one workload left: its rows are the merged rows
    head = "| lever / subset / ablation | merged | " + " | ".join(f"`{n}`" for n in names) + " |"
    rows = [head, "|---|---|" + "---|" * len(names)]
    for lv in levers:
        cells = []
        for n in names:
            s = lv["shapes"].get(n)
            cells.append("—" if s is None else
                         f"{_pct(s.get('add'))} / {_pct(s.get('remove'))} {s.get('class')}")
        rows.append(f"| `{lv['name']}` | {_pct(lv.get('add_pct'))} / {_pct(lv.get('remove_pct'))} "
                    f"**{lv.get('cls')}** | " + " | ".join(cells) + " |")
    abl_gaps: dict = {}
    for t in others:
        cells = []
        for n in names:
            s = t["shapes"].get(n)
            why = (t.get("shapes_missing") or {}).get(n) if t["kind"] == "ablation" else None
            if s is None and why:
                abl_gaps[n] = why
                cells.append(f"**incomplete** ({_ablation_gap(why)})")
            elif s is None:
                cells.append("—")
            elif t["kind"] == "ablation" and s.get("status") == "budget":
                abl_gaps.setdefault(n, "budget")
                cells.append(f"**incomplete** ({_ablation_gap('budget')})")
            else:
                cells.append(_pct(s.get("delta")) + (f" {s.get('status')}" if t["kind"] == "ablation" else ""))
        tag = "subset" if t["kind"] == "variant" else "ablation"
        rows.append(f"| {tag} `{t['name']}` | {_pct(t.get('delta_pct'))} | " + " | ".join(cells) + " |")
    if stack.get("shapes"):
        cells = [("—" if (s := stack["shapes"].get(n)) is None else
                  f"{_pct(s.get('all_pct'))} (floor {s.get('noise_pct')}%)") for n in names]
        rows.append(f"| all-on | {_pct(stack.get('all_pct'))} (floor {stack.get('noise_pct')}%) | "
                    + " | ".join(cells) + " |")
    split = [lv["name"] for lv in levers if lv.get("cls") == "shape_split"]
    return ("\n**Per-workload readings (lever cells = add-one / remove-one + that workload's own "
            "class; in parentheses, that workload's noise floor)**:\n" + "\n".join(rows) + "\n"
            "`shape_split` = this lever's **add-one** gets faster beyond its own noise floor on one "
            "workload and slower beyond the floor on another — the merged reading averages it to ≈ 0, "
            "but it is **not** null: its price changes with size, and the candidate spelling is a "
            "size dispatch (enabled only on the shape range where it gets faster); a citation must say "
            "on which workloads it holds. Only the add-one counts: a remove-one whose direction differs "
            "across workloads says that its coupling with the other levers in all-on changes with "
            "size, not that it pays at some sizes — read it by the merged class. "
            + (f"This table's shape_split: {', '.join(f'`{n}`' for n in split)}. " if split else "")
            + f"The merged class is the verdict over these {len(names)} workloads together; when one "
            "workload's reading contradicts the merge, the diagnosis must say whose it is. "
            + (f"An ablation cell \"**incomplete**\" = that workload's ablation pass has no reading ("
               f"{'; '.join(f'`{n}`: {_ablation_gap(w)}' for n, w in abl_gaps.items())}), **not** "
               f"\"no change without it\": a workload whose whole pass did not finish stays out of the "
               f"merged reading (the merged value comes from the other workloads only), and a single "
               f"entry that was never reached has a merged reading of n/a; this must not be taken to "
               f"mean the mechanism has the same price on the unfinished workloads."
               if abl_gaps else "")
            + "\n")


def _ablation_gap(why: str) -> str:
    """A workload's missing ablation reading, for the teacher (ABLATE-SHAPES reasons
    and a row's own status=budget)."""
    if why == "cut":
        return ("the job ended before its ablation pass closed (the job clock ran out or it was "
                "killed); no ABLATE verdict")
    if why == "budget":
        return "the ablation budget ran out before this entry was reached"
    if why.startswith("fail:"):
        return f"ablation pass failed: {why[5:]}"
    return why


def _head(text: str | None, lines: int, chars: int) -> str:
    body = "\n".join((text or "").splitlines()[:lines])
    return body[:chars]


# The other-workloads section rides in every teacher entry of a group card (the design
# turn and the diagnosis each carry one): on B200 e2e logs it was 20.9k chars /
# 296 lines for two workloads, most of it what the representative's own blocks already
# say -- the dossier's equal axes, every REGION row and the ROLE-JSON of both role
# tables, the opcode-family half of the ledger, the timeline's per-run caveats. Each
# reading is cut to what can differ by size, and the section is capped.
READINGS_MAX_CHARS = 9000
LEDGER_HEAD_ROWS = 6
_LINE_MAX = 220
_TIMELINE_MAX = 1800
_ROLE_LINE_RE = re.compile(r"^ROLE label=(\S+) (.*)$", re.M)
_TENSOR_LINE_RE = re.compile(r"^\[tensor\] active=(\S+)", re.M)


def _clip(line: str, n: int = _LINE_MAX) -> str:
    return line if len(line) <= n else line[:n - 1] + "…"


def _diff_mismatches(text: str | None) -> "tuple[list, int]":
    """(the ``!`` axes of an orch_tools dossier diff with their per-side values, the
    number of ``=`` axes left out)."""
    keep, same, inside = [], 0, False
    for line in (text or "").splitlines():
        st = line.strip()
        if st.startswith("! "):
            inside = True
            keep.append(_clip(line.rstrip()))
        elif st.startswith("= "):
            inside = False
            same += 1
        elif inside and line.startswith("      ") and st:
            keep.append(_clip(line.rstrip()))
        else:
            inside = False
    return keep, same


def _roles(text: str | None) -> "tuple[dict, str | None]":
    """({role: {share, exposed_ns, stall}}, tensor active) of a role-exposure block."""
    out = {}
    for m in _ROLE_LINE_RE.finditer(text or ""):
        kv = _kv(m.group(2))
        out[m.group(1)] = {"share": kv.get("share"), "exposed_ns": kv.get("exposed_ns"),
                           "stall": (kv.get("stalls") or "").split(",")[0] or None}
    t = _TENSOR_LINE_RE.search(text or "")
    return out, (t.group(1) if t else None)


def _role_cell(r: dict | None) -> str:
    if not r:
        return "—"
    ns = f" / {r['exposed_ns']} ns" if r.get("exposed_ns") not in (None, "None") else ""
    return f"{r.get('share')}%{ns}" + (f" · {r['stall']}" if r.get("stall") else "")


def _role_table(rep: str, wl: str, roles, ref_roles, rep_roles) -> list:
    """The role accounts side by side (student here / reference here / student on the
    representative): share, exposed ns, top stall -- the REGION rows and ROLE-JSON the
    full blocks carry are left out."""
    stu, t_stu = _roles(roles)
    ref, t_ref = _roles(ref_roles)
    rp, t_rp = _roles(rep_roles)
    if not (stu or ref):
        return []
    cols = [("student", stu, t_stu, roles, "(missing: the pricing job produced no role account)"),
            ("reference (same workload)", ref, t_ref, ref_roles,
             "(missing: this run has no reference pricing tree, or the reference was not priced on "
             "this workload)")]
    if rp:
        cols.append((f"student on the representative `{rep}`", rp, t_rp, rep_roles, "(missing)"))
    labels = sorted(set(stu) | set(ref) | set(rp),
                    key=lambda r: -float((stu.get(r) or ref.get(r) or {}).get("share") or 0))
    out = ["| role (share / exposed / top stall) | " + " | ".join(c[0] for c in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for lb in labels:
        out.append(f"| {lb} | " + " | ".join(_role_cell(c[1].get(lb)) for c in cols) + " |")
    out.append("| tensor active | " + " | ".join((c[2] or "—") if c[3] else c[4] for c in cols) + " |")
    return out


def _ledger_head(text: str | None, rows: int = LEDGER_HEAD_ROWS) -> list:
    """The top ``rows`` ledger rows (most expensive first), runs of spaces folded; the
    opcode-family table under them is left out."""
    lines = (text or "").splitlines()
    try:
        at = next(i for i, ln in enumerate(lines) if ln.startswith("--- ledger rows"))
    except StopIteration:
        return [_clip(ln) for ln in lines[:rows] if ln.strip()]
    out = [lines[at].strip()]
    for ln in lines[at + 1:]:
        if not ln.strip() or ln.startswith("---"):
            break
        out.append(_clip(re.sub(r" {2,}", "  ", ln.strip())))
        if len(out) > rows:
            break
    return [ln for ln in lines[:at] if ln.startswith(LEDGER_WARNING_PREFIX)] + out


def _timeline_lean(text: str | None) -> "tuple[str, int]":
    """A timeline table without its per-run caveat paragraphs (the representative's block
    states them) and empty segment tables, capped; and how many caveats were dropped."""
    out, dropped = [], 0
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith((TL_COVERAGE_NOTE, TL_SELFCHECK_NOTE)):
            dropped += 1
            i += 1
            continue
        if ln.endswith(TL_SEGMENT_LEDGER_HEADING) \
                and (i + 3 >= len(lines) or not lines[i + 3].startswith("| `")):
            i += 3                    # heading + two header lines, no segment rows
            if out and not out[-1]:
                out.pop()             # and the blank line that led into it
            continue
        out.append(ln)
        i += 1
    body = "\n".join(out).strip("\n")
    if len(body) > _TIMELINE_MAX:
        body = body[:body.rfind("\n", 0, _TIMELINE_MAX)] + "\n… (timeline table truncated)"
    return body, dropped


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit - 200)
    cut = cut if cut > 0 else limit - 200
    # never leave a code fence open across the cut
    tail = "\n```" if text[:cut].count("```") % 2 else ""
    return (text[:cut] + tail + f"\n\n… (section truncated: over the {limit}-character cap; the "
            f"remaining {len(text) - cut} characters are not listed — readings that are not listed "
            "must not be treated as read.)\n")


def workload_readings_block(rep: str, rows: list, axes: dict, sha: str, *,
                            rep_roles: str | None = None,
                            limit: int = READINGS_MAX_CHARS) -> str:
    """The teacher's section for a group card's other meta workloads, per workload what
    can differ from the representative's readings above it: the SASS dossier's axes that
    differ from the representative's (or "the same structure"), the reference-vs-student
    axes that differ on that workload, the role accounts side by side (student, the
    reference on the same workload, the student on the representative: ``rep_roles``),
    the head of the stall ledger and the timeline tables -- each cut to that, the whole
    capped at ``limit`` chars with the cut said. ``rows``: {"wl", and any of "skipped",
    "missing", "same_as_rep", "diff_rep", "diff_ref", "roles", "ref_roles", "ledger",
    "timeline"}."""
    if not rows:
        return ""
    out = [f"\n## Readings of the multi-workload tree's other workloads (one job per workload)\n"
           f"The NCU, stall ledger, dossier diff, role accounts and timeline above are all readings of "
           f"the representative workload `{rep}` (the driver's machine verdicts read only it, too). "
           f"Below is what **may differ** from the representative for the other {len(rows)} "
           f"workload(s) on the same tree `{sha}`: the dossier axes that differ from the "
           f"representative, the reference-vs-student axes that differ, the side-by-side role "
           f"accounts, the first {LEDGER_HEAD_ROWS} stall-ledger rows, and the timeline (axes equal to "
           f"the representative's, REGION details, opcode tables and the probe's generic notes are "
           f"left out). A multi-workload tree's KEEP gate is the group geomean: when a workload's "
           f"bottleneck or structure differs from the representative's, the diagnosis must say so; a "
           f"locus that holds only on the representative cannot be taken as the conclusion for the "
           f"group, and a size-dispatch spelling must say on which shape range it is enabled.\n"]
    for r in rows:
        wl = r["wl"]
        out.append(f"\n### `{wl}` (axes `{json.dumps(axes.get(wl) or {}, ensure_ascii=False)}`)\n")
        if r.get("skipped"):
            out.append(f"- Not measured: {r['skipped']}.\n")
            continue
        if r.get("missing"):
            out.append(f"- {r['missing']} — this workload's student readings are absent and must not "
                       f"be treated as read.\n")
        if r.get("same_as_rep"):
            out.append(f"- SASS orchestration: the dossier summary is **identical axis by axis** to the "
                       f"representative `{rep}`'s (the same structure); differences can only be in the "
                       f"timing readings (role accounts / ledger / timeline below).\n")
        elif r.get("diff_rep"):
            bang, same = _diff_mismatches(r["diff_rep"])
            out.append(f"- SASS orchestration: axes that differ from the representative `{rep}` → `{wl}` "
                       f"(this size compiles to a different structure; the other {same} axes are the "
                       f"same):\n```\n" + "\n".join(bang or ["(no `!` axes)"]) + "\n```\n")
        if r.get("diff_ref"):
            bang, same = _diff_mismatches(r["diff_ref"])
            out.append(f"- Reference vs student (both on `{wl}`): axes that differ (the other {same} "
                       f"axes are the same):\n```\n"
                       + "\n".join(bang or ["(no `!` axes)"]) + "\n```\n")
        if r.get("roles") or r.get("ref_roles"):
            table = _role_table(rep, wl, r.get("roles"), r.get("ref_roles"), rep_roles)
            if table:
                out.append("- Role accounts:\n" + "\n".join(table) + "\n")
            else:
                out.append("- Role accounts:\n```\n" + _head(r.get("roles") or r.get("ref_roles"), 12, 1200)
                           + "\n```\n")
        if r.get("ledger"):
            out.append(f"- Stall ledger (student, first {LEDGER_HEAD_ROWS} rows):\n```\n"
                       + "\n".join(_ledger_head(r["ledger"])) + "\n```\n")
        if r.get("timeline"):
            body, dropped = _timeline_lean(r["timeline"])
            out.append("- Timeline (student):\n" + body + "\n"
                       + (f"({dropped} more coverage / probe self-check notes left out; read them as in "
                          f"the representative's timeline block.)\n"
                          if dropped else ""))
    return _cap("".join(out), limit)
