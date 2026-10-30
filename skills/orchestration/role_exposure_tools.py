"""Role exposure ledger — the three orchestration reads joined into one table.

The 0.95x-class residual (seen on attn bwd GQA / grouped gemm / dsa ixbwd) is
non-tensor time the audit already calls "same tensor work as the ref": the
question left is WHICH WARP ROLE leaves the tensor pipe idle, and by how much.
Three tools each hold one axis of that answer and none of them joins it:

* ``orch_tools`` partitions the SASS page into regions and labels the loops by
  role (copy-producer / mma-consumer / scalar / spin) — WHO, statically;
* the PC-sampled stall ledger says on which instruction warps wait — WHERE,
  but summed over every role that touches the PC range;
* the PM-sampling timeline says how busy the tensor pipe is over the run —
  the only reading that turns "warps waited" into "the pipe idled".

The teacher has been re-deriving the join by hand in a verdict probe on every
meta step (on grouped-gemm: "math region 159.5 exposed beats per K-block vs
ref 106.3", read off a bespoke script, lost by the next step). This module is
that script, frozen: stall samples are attributed to the dossier's regions by
PC, aggregated per role, and scaled by the non-tensor fraction of the kernel
into an exposure estimate per role, in ns — the same table for student and
ref, so the driver can put them side by side.

Estimate, not measurement: exposure is attributed to roles in proportion to
their stall share (a role waiting while the tensor pipe is busy is hidden,
and the per-PC sampler cannot tell hidden from exposed). The components are
printed beside the estimate so a diagnosis can re-derive or reject it.

Verdict line, after the ``=== ROLE EXPOSURE BEGIN/END ===`` block:
``ROLE-OK roles=<n> samples=<N> resolved=<pct> tensor=<pct>`` when the tensor
duty cycle was measured (the exposure column is real); ``ROLE-PARTIAL roles=<n>
samples=<N> resolved=<pct> tensor=none`` when it was not — shares only, every
exposed_ns null. The split is a lesson from dsa_sparse: a ref table whose
tensor came back "unavailable" printed ROLE-OK, the driver banked it as the
ref side of the overlap account, and the teacher diagnosed against a column of
nulls for a whole run. A partial table is still worth banking (the stall
shares are real), but it must not pass for the account. ``ROLE-FAIL
reason=...`` when the join itself is unusable. The machine core rides one
``ROLE-JSON:`` line, for driver-side diffs without either page in hand (same
contract as ORCH-SUMMARY-JSON).

The tensor term comes from the PM-sampling sections (``stall_price_run.py
--pm``) or, failing that, the whole-kernel ``sm__pipe_tensor_cycles_active``
aggregate when the report carries it. A bare ``--set source`` collection
carries neither — the recipe that produced that null column.

stdlib-only at import; the NCU / SASS / orch toolkits are imported lazily,
package-or-vendored-flat like every other recon tool.
"""
import json
from bisect import bisect_right
from collections import Counter, defaultdict

BEGIN = "=== ROLE EXPOSURE BEGIN ==="
END = "=== ROLE EXPOSURE END ==="

MIN_SAMPLES = 500        # same floor as the stall ledger: below it the sampler was off
JOIN_FLOOR = 0.50        # below this the sampled PCs and the page are different code
MAX_REGION_ROWS = 10
ROLE_ORDER = ("copy-producer", "mma-consumer", "scalar", "spin", "straight")
STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
TENSOR_PM_SUBSTR = "pipe_tensor_cycles_active"
# Whole-kernel fallback when the PM series is absent (plain --set full run):
# the same metric the digest reports as tensor_pct.
TENSOR_AGG_METRICS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
)


def short_stall(name: str) -> str:
    return name[len(STALL_PREFIX):] if name.startswith(STALL_PREFIX) else name


def _addr_int(ins):
    try:
        return int(ins.addr, 16) if ins.addr else None
    except ValueError:
        return None


def role_of(region) -> str:
    """A region's role label: the loop label, or ``straight`` for the code
    between loops (prologue / per-tile scheduling / epilogue)."""
    return region.label if region.kind == "loop" else "straight"


# --- the join -------------------------------------------------------------------

def role_account(instrs, regions, per_pc: dict, addr_int=None) -> dict:
    """Attribute PC-sampled stall counts to the dossier's regions and roles.

    ``per_pc``: {pc(int) -> {stall_metric -> count}} (stall_tools.collect_per_pc).
    ``regions``: orch_tools.find_regions over ``instrs`` (one launch's page).
    Returns {"total", "resolved", "roles": {label: {...}}, "regions": {rid: {...}}}
    where every entry carries ``samples`` and a ``stalls`` Counter.
    """
    if addr_int is None:
        addr_int = _addr_int
    spans = []
    for r in regions:
        lo, hi = addr_int(instrs[r.lo]), addr_int(instrs[r.hi])
        if lo is None or hi is None:
            continue
        spans.append((lo, hi, r))
    spans.sort(key=lambda s: s[0])
    los = [s[0] for s in spans]

    roles: dict = defaultdict(lambda: {"samples": 0, "stalls": Counter(),
                                       "regions": set()})
    per_region: dict = defaultdict(lambda: {"samples": 0, "stalls": Counter()})
    total = resolved = 0
    for pc, stalls in per_pc.items():
        n = sum(stalls.values())
        total += n
        i = bisect_right(los, pc) - 1
        if i < 0:
            continue
        lo, hi, r = spans[i]
        if pc > hi:
            continue
        resolved += n
        label = role_of(r)
        roles[label]["samples"] += n
        roles[label]["regions"].add(r.rid)
        per_region[r.rid]["samples"] += n
        per_region[r.rid]["label"] = label
        for k, v in stalls.items():
            roles[label]["stalls"][k] += v
            per_region[r.rid]["stalls"][k] += v
    return {"total": total, "resolved": resolved,
            "roles": dict(roles), "regions": dict(per_region)}


def exposure_rows(acc: dict, tensor_active_pct, duration_ns) -> list:
    """Per-role rows: share of resolved samples, and the exposure estimate.

    exposed_frac = 1 - tensor_active; a role's exposed share = its stall
    share x exposed_frac; in ns = x duration. None when tensor_active is
    unknown (the shares still print — they are the measured half)."""
    resolved = acc["resolved"] or 1
    exposed_frac = None
    if tensor_active_pct is not None:
        exposed_frac = max(0.0, 1.0 - float(tensor_active_pct) / 100.0)
    rows = []
    labels = [l for l in ROLE_ORDER if l in acc["roles"]]
    labels += sorted(l for l in acc["roles"] if l not in labels)
    for label in labels:
        e = acc["roles"][label]
        share = e["samples"] / resolved
        row = {"role": label, "samples": e["samples"], "share_pct": 100.0 * share,
               "regions": sorted(e["regions"]),
               "stalls": _top_stalls(e["stalls"], e["samples"]),
               "exposed_pct": None, "exposed_ns": None}
        if exposed_frac is not None:
            row["exposed_pct"] = 100.0 * share * exposed_frac
            if duration_ns:
                row["exposed_ns"] = float(duration_ns) * share * exposed_frac
        rows.append(row)
    return rows


def _top_stalls(stalls: Counter, samples: int, n: int = 3) -> dict:
    tot = samples or 1
    return {short_stall(k): round(100.0 * v / tot, 1)
            for k, v in stalls.most_common(n)}


def _fmt_stalls(d: dict) -> str:
    return ",".join(f"{k}:{v:.0f}" for k, v in d.items()) or "-"


# --- tensor activity: PM plateau first, aggregate metric second -------------------

def tensor_active(action):
    """(pct, source): mainloop plateau of the PM tensor-active series when the
    report carries PM sampling, else the whole-kernel aggregate, else None."""
    pmt = _pm_tools()
    if pmt is not None:
        try:
            v = pmt.tensor_active_plateau(pmt.collect_pm_series(action))
            if v is not None:
                return v, "pm-plateau"
        except Exception:      # noqa: BLE001 — a reader failure is the timeline's
            pass               # problem to report; here it only means "no series"
    for m in TENSOR_AGG_METRICS:
        try:
            v = action[m].value()
        except Exception:      # noqa: BLE001
            continue
        if v is not None:
            return float(v), "ncu-aggregate"
    return None, "unavailable"


# --- the full run -------------------------------------------------------------------

def report(report_path, sass_text=None, launch=None, kernel_substr: str = "",
           out=print) -> bool:
    """The role exposure ledger off one .ncu-rep (+ its SASS page); True iff
    ROLE-OK printed. ``sass_text``/``launch`` are the page and launch index the
    pricing job already rendered and picked (shared with the segment ledger and
    the dossier); rendered here only when absent."""
    st = _stall_tools()
    orch = _orch()
    if st is None or orch is None:
        out("ROLE-FAIL reason=no-tools (stall_tools/orch_tools not importable)")
        return False
    ncu_report = st._import_ncu_report()
    if ncu_report is None:
        out("ROLE-FAIL reason=no-ncu-report-module")
        return False
    try:
        rep = ncu_report.load_report(str(report_path))
    except Exception as e:     # noqa: BLE001
        out(f"ROLE-FAIL reason=report-unreadable ({e})")
        return False
    action = st.pick_dominant_action(rep, kernel_substr)
    if action is None:
        out("ROLE-FAIL reason=no-kernel-matched")
        return False
    # a short kernel's one launch falls short of the floor; its other timed launches
    # carry the rest (stall_tools.collect_per_pc_pooled -- an older seeded copy without
    # it reads the one launch, as before)
    pool = getattr(st, "collect_per_pc_pooled", None)
    per_pc, pooled = pool(rep, action, MIN_SAMPLES) if pool else (st.collect_per_pc(action), 1)
    total = sum(sum(s.values()) for s in per_pc.values())
    if total < MIN_SAMPLES:
        out(f"ROLE-FAIL reason=too-few-samples ({total} < {MIN_SAMPLES})")
        return False
    if pooled > 1:
        out(f"[sanity] role samples pooled over {pooled} launches of the dominant kernel")
    if sass_text is None:
        sass_text, err = st.render_sass_text(report_path)
        if err:
            out(f"ROLE-FAIL reason=no-sass ({err})")
            return False
    sass = _sass()
    if sass is None:
        out("ROLE-FAIL reason=no-tools (sass_tools not importable)")
        return False
    try:
        chunk, n_launches, used = orch.select_launch(sass_text, launch, kernel_substr)
    except TypeError:
        # an older orch_tools copy in the tree: no kernel-name pick, launch index only
        chunk, n_launches, used = orch.select_launch(sass_text, launch)
    instrs = list(sass.parse_sass(chunk).instructions)
    if not instrs:
        out("ROLE-FAIL reason=sass-parsed-empty")
        return False
    regions = orch.find_regions(instrs)
    duration_ns = st._duration(action)
    t_active, t_source = tensor_active(action)
    return emit(instrs, regions, per_pc, t_active, t_source, duration_ns,
                out=out, launches=(n_launches, used))


def emit(instrs, regions, per_pc, tensor_active_pct, tensor_source,
         duration_ns, out=print, launches=(1, 0)) -> bool:
    """Print the block from already-joined inputs (what ``report`` computes;
    split out so the join is testable without an .ncu-rep)."""
    acc = role_account(instrs, regions, per_pc)
    out(BEGIN)
    n_launches, used = launches
    if n_launches > 1:
        out(f"[parse] page carried {n_launches} launches; using launch {used}")
    frac = acc["resolved"] / acc["total"] if acc["total"] else 0.0
    out(f"[join] {len(per_pc)} sampled PCs onto {len(regions)} regions; "
        f"{100 * frac:.1f}% of {acc['total']} stall samples resolved")
    if frac < JOIN_FLOOR:
        out(END)
        out(f"ROLE-FAIL reason=address-space-mismatch ({100 * frac:.1f}% < "
            f"{100 * JOIN_FLOOR:.0f}%)")
        return False
    t_txt = ("unknown" if tensor_active_pct is None
             else f"{tensor_active_pct:.1f}%")
    exposed = (None if tensor_active_pct is None
               else max(0.0, 1.0 - tensor_active_pct / 100.0))
    out(f"[tensor] active={t_txt} source={tensor_source} duration_ns={duration_ns}"
        + (f" exposed_frac={exposed:.3f}" if exposed is not None else ""))
    out("[read] share = the role's slice of every stall sample; exposed = share x "
        "(1 - tensor_active), an attribution estimate — a role that waits while "
        "the tensor pipe is busy is hidden, not exposed")
    rows = exposure_rows(acc, tensor_active_pct, duration_ns)
    for r in rows:
        ex = ("" if r["exposed_pct"] is None else
              f" exposed_pct={r['exposed_pct']:.1f}"
              + (f" exposed_ns={r['exposed_ns']:.0f}" if r["exposed_ns"] is not None
                 else ""))
        out(f"ROLE label={r['role']} regions={len(r['regions'])} "
            f"samples={r['samples']} share={r['share_pct']:.1f}{ex} "
            f"stalls={_fmt_stalls(r['stalls'])}")
    ranked = sorted(acc["regions"].items(), key=lambda kv: -kv[1]["samples"])
    resolved = acc["resolved"] or 1
    for rid, e in ranked[:MAX_REGION_ROWS]:
        r = regions[rid]
        out(f"REGION r{rid} {e['label']:13s} [{instrs[r.lo].addr}..{instrs[r.hi].addr}] "
            f"{r.hi - r.lo + 1:5d} instrs share={100 * e['samples'] / resolved:.1f} "
            f"stalls={_fmt_stalls(_top_stalls(e['stalls'], e['samples']))}")
    core = {
        "tensor_active": (None if tensor_active_pct is None
                          else round(float(tensor_active_pct), 2)),
        "tensor_source": tensor_source,
        "duration_ns": duration_ns,
        "resolved_pct": round(100 * frac, 1),
        "roles": {r["role"]: {
            "share_pct": round(r["share_pct"], 2),
            "exposed_pct": (None if r["exposed_pct"] is None
                            else round(r["exposed_pct"], 2)),
            "exposed_ns": (None if r["exposed_ns"] is None
                           else round(r["exposed_ns"])),
            "regions": len(r["regions"]),
            "stalls": r["stalls"],
        } for r in rows},
    }
    out("ROLE-JSON: " + json.dumps(core, sort_keys=True))
    out(END)
    if tensor_active_pct is None:
        out(f"ROLE-PARTIAL roles={len(rows)} samples={acc['total']} "
            f"resolved={100 * frac:.1f} tensor=none")
        out("[role] the tensor duty cycle is missing, so no role has an exposed_ns "
            "— shares only. Collect with the PM sections (python3 "
            "notes/stall_price_run.py --pm) or add "
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed to "
            "the --metrics list; a bare --set source collection carries neither.")
        return True
    out(f"ROLE-OK roles={len(rows)} samples={acc['total']} "
        f"resolved={100 * frac:.1f} tensor={tensor_active_pct:.1f}")
    return True


# --- driver-side diff (no page needed) --------------------------------------------

def parse_core(block_text: str):
    """The ROLE-JSON core out of a banked block, or None."""
    for line in (block_text or "").splitlines():
        if line.startswith("ROLE-JSON: "):
            try:
                return json.loads(line[len("ROLE-JSON: "):])
            except json.JSONDecodeError:
                return None
    return None


def align_roles(ref: dict, student: dict, names=("ref", "student")) -> str:
    """Same-role comparison of two cores, one row per role, plus the tensor
    line — the table a diagnosis reads instead of two raw blocks."""
    na, ns = names
    lines = []

    def t(c):
        v = c.get("tensor_active")
        return ("unknown" if v is None else f"{v:.1f}%") + f" ({c.get('tensor_source')})"

    lines.append(f"tensor_active: {na} {t(ref)} | {ns} {t(student)}; "
                 f"duration_ns: {na} {ref.get('duration_ns')} | {ns} {student.get('duration_ns')}")
    lines.append("role            | ref share% | student share% | ref exposed% | "
                 "student exposed% | Δexposed ns | ref dominant stall | student dominant stall")
    roles = [r for r in ROLE_ORDER
             if r in ref.get("roles", {}) or r in student.get("roles", {})]
    # Parenthesised on purpose: `a | b - c` binds as `a | (b - c)`, which put
    # every ref role back into the list and printed each row twice (seen on an
    # offline grouped-gemm table).
    roles += sorted((set(ref.get("roles", {})) | set(student.get("roles", {})))
                    - set(roles))
    for role in roles:
        a, b = ref.get("roles", {}).get(role), student.get("roles", {}).get(role)

        def f(e, k, fmt="{:.1f}"):
            if e is None or e.get(k) is None:
                return "—"
            return fmt.format(e[k])

        delta = "—"
        if a and b and a.get("exposed_ns") is not None and b.get("exposed_ns") is not None:
            delta = f"{b['exposed_ns'] - a['exposed_ns']:+.0f}"
        lines.append(
            f"{role:<15} | {f(a, 'share_pct'):>10} | {f(b, 'share_pct'):>13} | "
            f"{f(a, 'exposed_pct'):>12} | {f(b, 'exposed_pct'):>15} | {delta:>11} | "
            f"{_fmt_stalls((a or {}).get('stalls', {})):<16} | "
            f"{_fmt_stalls((b or {}).get('stalls', {}))}")
    return "\n".join(lines)


# --- lazy toolkit imports (package or vendored-flat) --------------------------------

def _stall_tools():
    try:
        from forgekernel.skills import stall_tools
        return stall_tools
    except ImportError:
        try:
            import stall_tools     # noqa: F401
            return stall_tools
        except ImportError:
            return None


def _orch():
    try:
        from forgekernel.skills import orch_tools
        return orch_tools
    except ImportError:
        try:
            import orch_tools      # noqa: F401
            return orch_tools
        except ImportError:
            return None


def _sass():
    try:
        from forgekernel.skills import sass_tools
        return sass_tools
    except ImportError:
        try:
            import sass_tools      # noqa: F401
            return sass_tools
        except ImportError:
            return None


def _pm_tools():
    try:
        from forgekernel.skills import pm_timeline_tools
        return pm_timeline_tools
    except ImportError:
        try:
            import pm_timeline_tools   # noqa: F401
            return pm_timeline_tools
        except ImportError:
            return None
