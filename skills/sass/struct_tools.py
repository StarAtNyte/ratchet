"""Per-segment construction table of a kernel read off its SASS — the STRUCT channel.

Why this exists (learned on a gdn_prefill_fwd blackbox run): every other reconnaissance
channel is a *count* or a *price* — opcode histogram, schedule transcription,
stall ledger, role exposure. None of them says how a phase is BUILT: which warps
do it, on what tile granule, in what internal precision, how the partial results
are handed to the next phase. That run's whole gap sat in one such phase (the
solver's triangular inverse: skipping it at runtime met the target exactly), the
recon report inferred its construction from the instruction mix ("four 16x16
diagonal blocks"), the milestone and a certified KB entry inherited that
inference, and the student built the wrong shape for 17 hours. Decoded offline
afterwards from the SAME archived data (SASS page + per-PC execution counts) the
reference turned out to use 8x8 blocks, three doubling levels, fp16 internally
and a split-K reduction — every one of those facts is mechanical to read once
the kernel is cut into phases.

What it reads, all from one ``ncu --set source`` collection (no source, no GPU
beyond the collection the SASS channel already makes):

1. **Segments** — a named barrier (``BAR.SYNC <id>, <threads>``) that recurs
   inside one loop body fences the phases of a warp-cooperative construction.
   Consecutive hits of the same barrier form a *region*; the code between two
   hits is a *segment*. An unrolled body shows up as several copies of the same
   region and is reported once.
2. **Who** — the barrier's thread-count operand names the cooperating warps; an
   instruction's per-PC "Instructions Executed" count divided by the barrier's
   own count, times those warps, is the number of warps that execute it. No
   grid size or trip count is needed for that ratio. Non-integral results mean
   predication the ratio cannot see — the table is then not believed (FAIL).
   Two downgrades, both rendered as ``x<n>?`` instead of a warp count: a ratio
   above the barrier's own warps (a nested loop's trips, or warps outside the
   barrier running the same code), and a barrier id that also takes
   ``BAR.ARV`` (the operand's thread count then includes arrive-only warps, so
   the sync side's count is unknown for the whole region).
3. **What shape** — the smem immediates of the segment's LDSM/STSM/LDS/STS
   instructions: their smallest recurring stride is the tile granule the phase
   addresses (128 B = an 8x8 fp16 tile), the MMA opcode carries its own shape.
4. **What precision** — the MMA opcode's input-type suffix (none = fp16,
   ``.BF16``, ``.TF32``, ``.E4M3``…) and the F2FP conversions' target type.
5. **Hand-off** — offsets one segment stores (STSM/STS) and a later one loads
   (LDSM/LDS) name the smem hand-off between phases; a reduction phase shows
   as HADD2/FADD/SHFL after MMA.

Design rules, inherited from ``sass_tools``: parsing is delegated (no hand-rolled
parser), silence is failure — ``report`` ends in exactly one ``STRUCT-OK ...`` or
``STRUCT-FAIL reason=...`` line and the driver's finish gate looks for the
former. What the tool does NOT do: name the algorithm. It prints the table;
reading "8 -> 16 -> 32 -> 64 doubling" off it is the teacher's job, and the
table is the citation that reading must carry.

Stdlib only; vendored flat into ``notes/`` (``import struct_tools``).
"""

from __future__ import annotations

import csv
import io
import re
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

try:  # vendored alongside this file in a job tree ("import sass_tools" via sys.path)
    import sass_tools  # type: ignore
except ImportError:  # harness-side import path
    from forgekernel.skills import sass_tools

MIN_INSTRUCTIONS = sass_tools.MIN_INSTRUCTIONS
MIN_SEGMENTS = 2            # a "region" with one barrier has no inside to describe
REGION_GAP = 400            # instructions between two hits of a barrier before a new region starts
WARP_INTEGRAL_TOL = 0.06    # |warps - round(warps)| accepted per costed instruction
WARP_INTEGRAL_SHARE = 0.90  # share of costed instructions that must read integral
MAX_STRIDES = 6
MAX_CTA_WARPS = 32          # 1024 threads: the hard ceiling on any warp count
TILE_FAMILIES = ("LDSM", "STSM")

MMA_FAMILIES = ("HMMA", "HGMMA", "QGMMA", "IGMMA", "BGMMA", "IMMA", "DMMA")
SMEM_FAMILIES = ("LDSM", "STSM", "LDS", "STS")
STORE_FAMILIES = ("STSM", "STS")
LOAD_FAMILIES = ("LDSM", "LDS")
CONVERT_FAMILIES = ("F2FP", "F2F", "I2F", "F2I")
REDUCE_FAMILIES = ("HADD2", "HFMA2", "HMUL2", "FADD", "FFMA", "SHFL", "REDUX", "VOTE")
XCEND_FAMILIES = ("MUFU",)
REPORTED = MMA_FAMILIES + SMEM_FAMILIES + CONVERT_FAMILIES + REDUCE_FAMILIES + XCEND_FAMILIES
# Opcodes whose full dotted form (shape / type) is the reading; others report by family.
SHAPED = MMA_FAMILIES + ("LDSM", "STSM", "F2FP")
# A synchronisation point that ends the segment BEFORE the region's first barrier.
SYNC_FAMILIES = ("BAR", "WARPGROUP", "MEMBAR", "DEPBAR", "SYNCS", "ERRBAR")
DECISIVE_FAMILIES = MMA_FAMILIES + ("LDSM", "STSM")

_BAR_OPERANDS = re.compile(r"^(0x[0-9a-fA-F]+|\d+)\s*(?:,\s*(0x[0-9a-fA-F]+|\d+))?\s*$")
_IMM_OFFSET = re.compile(r"\+\s*0x([0-9a-fA-F]+)\s*\]")
# MMA input-type suffixes; an HMMA/HGMMA carrying none of them takes fp16 inputs.
_MMA_TYPES = ("BF16", "TF32", "E4M3", "E5M2", "F8", "S8", "U8", "S4", "U4", "F64")


def fam(opcode: str) -> str:
    return opcode.split(".", 1)[0]


def _int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


# ------------------------------------------------------------------ inputs
def source_csv_to_sass_and_counts(csv_text: str) -> Tuple[str, Dict[str, int]]:
    """``ncu --import rep --page source --print-source sass --csv`` -> (SASS text
    in the bare-address style ``parse_sass`` reads, {address: warp-level
    "Instructions Executed"}).

    The exec column is the warp-level one ("Instructions Executed", not
    "Thread Instructions Executed" / "Predicated-On ..."): the who-does-what
    ratio counts warps. Rows without a ``0x`` address are headers/banners.
    """
    rows = list(csv.reader(io.StringIO(csv_text)))
    h_i = next((i for i, r in enumerate(rows[:60])
                if any(c.strip() == "Address" for c in r)), None)
    if h_i is None:
        return "", {}
    hdr = [c.strip() for c in rows[h_i]]
    ai = hdr.index("Address")
    si = hdr.index("Source") if "Source" in hdr else None
    ei = next((j for j, c in enumerate(hdr)
               if c == "Instructions Executed"), None)
    if ei is None:
        ei = next((j for j, c in enumerate(hdr)
                   if "Instructions Executed" in c and "Thread" not in c
                   and "Predicated" not in c), None)
    lines: List[str] = []
    counts: Dict[str, int] = {}
    for r in rows[h_i + 1:]:
        if len(r) != len(hdr):
            continue
        addr = r[ai].strip()
        if not addr.lower().startswith("0x"):
            continue
        ins = r[si].strip() if si is not None else ""
        lines.append(f"{addr}  {ins}")
        if ei is not None:
            v = r[ei].strip().replace(",", "")
            try:
                counts[_norm_addr(addr)] = int(float(v))
            except ValueError:
                pass
    return "\n".join(lines) + ("\n" if lines else ""), counts


def _norm_addr(addr: Optional[str]) -> str:
    if addr is None:
        return ""
    try:
        return hex(int(addr, 16))
    except ValueError:
        return addr.lower()


# ------------------------------------------------------------------ segmentation
def named_barriers(instrs: Sequence[sass_tools.Instr]) -> List[Tuple[int, str, Optional[int]]]:
    """[(index, key, warps)] for every ``BAR.SYNC``-family instruction with an
    explicit id operand. key = ``<id>,<threads>`` (``<id>`` alone when the count
    is implicit — a CTA-wide barrier); warps = threads / 32 when given."""
    out = []
    for i, ins in enumerate(instrs):
        if fam(ins.opcode) != "BAR" or "SYNC" not in ins.opcode:
            continue
        m = _BAR_OPERANDS.match(ins.operands.strip())
        if not m:
            continue
        bid = _int(m.group(1))
        thr = _int(m.group(2)) if m.group(2) else None
        key = f"{hex(bid)},{hex(thr)}" if thr is not None else hex(bid)
        out.append((i, key, (thr // 32) if thr else None))
    return out


def arrive_only_ids(instrs: Sequence[sass_tools.Instr]) -> set:
    """Barrier ids that also take ``BAR.ARV`` hits. On such an id the thread count
    of ``BAR.SYNC <id>, <threads>`` covers arrive-only warps too, so the number
    of warps that execute the fenced segments (the sync side) is not readable
    off the operand: the who-column of that region is downgraded."""
    ids = set()
    for ins in instrs:
        if fam(ins.opcode) != "BAR" or "ARV" not in ins.opcode:
            continue
        m = _BAR_OPERANDS.match(ins.operands.strip())
        if m:
            ids.add(_int(m.group(1)))
    return ids


def regions(instrs: Sequence[sass_tools.Instr]) -> List[dict]:
    """Recurring named barriers -> regions: {key, warps, bar_warps, arv_split,
    bars, start, segs}.

    Hits of one barrier key closer than ``REGION_GAP`` instructions belong to
    the same region; a region needs >= MIN_SEGMENTS barriers. The first
    segment runs from the last synchronisation point before the first hit.
    ``warps`` is None (``arv_split`` True) when the id also takes ``BAR.ARV``:
    only the sync-side warps execute the segments and their count is unknown."""
    by_key: Dict[str, List[Tuple[int, Optional[int]]]] = {}
    for i, key, warps in named_barriers(instrs):
        by_key.setdefault(key, []).append((i, warps))
    arv = arrive_only_ids(instrs)
    out = []
    for key, hits in by_key.items():
        split = _int(key.split(",", 1)[0]) in arv
        clusters: List[List[int]] = []
        for i, _ in hits:
            if clusters and i - clusters[-1][-1] <= REGION_GAP:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        for bars in clusters:
            if len(bars) < MIN_SEGMENTS:
                continue
            first = bars[0]
            prev_sync = next((j for j in range(first - 1, -1, -1)
                              if fam(instrs[j].opcode) in SYNC_FAMILIES), None)
            if prev_sync is None or first - prev_sync > REGION_GAP:
                # no sync point in reach: open the first segment one median
                # barrier gap before the first hit
                gaps = [b - a for a, b in zip(bars, bars[1:])]
                start = max(0, first - int(statistics.median(gaps)))
            else:
                start = prev_sync + 1
            bounds = [start] + [b + 1 for b in bars]
            segs = [(bounds[k], bounds[k + 1] - 1) for k in range(len(bounds) - 1)]
            out.append({"key": key, "warps": None if split else hits[0][1],
                        "bar_warps": hits[0][1], "arv_split": split, "bars": bars,
                        "start": start, "segs": segs})
    out.sort(key=lambda r: r["start"])
    return out


def _fingerprint(instrs, seg) -> tuple:
    a, b = seg
    return tuple(sorted(Counter(_label(instrs[i].opcode) for i in range(a, b + 1)
                                if fam(instrs[i].opcode) in REPORTED).items()))


def dedupe_regions(instrs, regs: List[dict]) -> List[dict]:
    """Unrolled copies of one body (same barrier key, same per-segment class
    counts) collapse into the first, which records ``copies``."""
    seen: Dict[tuple, dict] = {}
    out = []
    for r in regs:
        fp = (r["key"], tuple(_fingerprint(instrs, s) for s in r["segs"]))
        if fp in seen:
            seen[fp]["copies"] += 1
            continue
        r = dict(r, copies=1)
        seen[fp] = r
        out.append(r)
    return out


# ------------------------------------------------------------------ readings
def _label(opcode: str) -> str:
    f = fam(opcode)
    return opcode if f in SHAPED else f


def mma_input_type(opcode: str) -> Optional[str]:
    """'fp16' | 'bf16' | 'tf32' | 'e4m3' | ... for an MMA opcode, None otherwise."""
    if fam(opcode) not in MMA_FAMILIES:
        return None
    parts = opcode.split(".")[1:]
    for t in _MMA_TYPES:
        if t in parts:
            return t.lower()
    return "fp16"


def smem_offsets(instrs, seg, families=SMEM_FAMILIES) -> List[int]:
    a, b = seg
    offs = set()
    for i in range(a, b + 1):
        ins = instrs[i]
        if fam(ins.opcode) in families:
            for m in _IMM_OFFSET.finditer(ins.operands):
                offs.add(int(m.group(1), 16))
    return sorted(offs)


def strides(offsets: Sequence[int]) -> List[int]:
    return sorted({b - a for a, b in zip(offsets, offsets[1:]) if b > a})


def decode(instrs: Sequence[sass_tools.Instr],
           exec_counts: Optional[Dict[str, int]] = None) -> List[dict]:
    """The construction table: one dict per (deduplicated) region.

    ``exec_counts`` maps normalised addresses to warp-level execution counts;
    without it the who-column is absent (the table is static only)."""
    regs = dedupe_regions(instrs, regions(instrs))
    table = []
    for r in regs:
        bar_exec = None
        if exec_counts:
            vals = [exec_counts.get(_norm_addr(instrs[b].addr)) for b in r["bars"]]
            vals = [v for v in vals if v]
            bar_exec = statistics.median(vals) if vals else None
        seg_rows = []
        integral = [0, 0]
        types: Counter = Counter()
        for si, (a, b) in enumerate(r["segs"], 1):
            classes: Dict[str, dict] = {}
            for i in range(a, b + 1):
                ins = instrs[i]
                f = fam(ins.opcode)
                if f not in REPORTED:
                    continue
                lab = _label(ins.opcode)
                row = classes.setdefault(lab, {"count": 0, "warps": [], "loops": []})
                row["count"] += 1
                t = mma_input_type(ins.opcode)
                if t:
                    types[t] += 1
                if exec_counts and bar_exec and r["warps"]:
                    e = exec_counts.get(_norm_addr(ins.addr))
                    if e:
                        w = e / bar_exec * r["warps"]
                        if w > r["warps"] + WARP_INTEGRAL_TOL:
                            # executed more often than the barrier's own warps: a
                            # loop nested inside the segment (a trip multiple) or
                            # warps outside the barrier running the same code —
                            # ambiguous either way, never a warp count
                            row["loops"].append(e / bar_exec)
                        else:
                            row["warps"].append(w)
                            integral[1] += 1
                            if abs(w - round(w)) <= WARP_INTEGRAL_TOL:
                                integral[0] += 1
                elif exec_counts and bar_exec and r.get("arv_split"):
                    # sync-side warp count unknown: keep the raw ratio to the
                    # barrier so the reader still sees who runs more/less often
                    e = exec_counts.get(_norm_addr(ins.addr))
                    if e:
                        row["loops"].append(e / bar_exec)
            for row in classes.values():
                ws, ls = row.pop("warps"), row.pop("loops")
                row["warps"] = round(statistics.median(ws), 2) if ws else None
                row["loop_x"] = round(statistics.median(ls), 1) if ls else None
                row["ambiguous"] = bool(ls) and row["warps"] is None
            offs = smem_offsets(instrs, (a, b))
            st = strides(offs)
            tile_st = strides(smem_offsets(instrs, (a, b), TILE_FAMILIES))
            seg_rows.append({
                "seg": si, "lines": (a, b), "classes": classes,
                "smem_offsets": offs, "strides": st[:MAX_STRIDES],
                # the granule is read off the MATRIX ops (LDSM/STSM move whole
                # 8x8 tiles); scalar LDS/STS immediates step by 4 B and say nothing
                # about the tile
                "granule": tile_st[0] if tile_st else None,
                "stores": smem_offsets(instrs, (a, b), STORE_FAMILIES),
                "loads": smem_offsets(instrs, (a, b), LOAD_FAMILIES),
            })
        handoffs = []
        for i, s in enumerate(seg_rows):
            for t in seg_rows[i + 1:]:
                common = sorted(set(s["stores"]) & set(t["loads"]))
                if common:
                    handoffs.append((s["seg"], t["seg"], common))
        table.append({
            "key": r["key"], "warps": r["warps"], "bar_warps": r.get("bar_warps"),
            "arv_split": bool(r.get("arv_split")), "copies": r["copies"],
            "bars": r["bars"], "bar_exec": bar_exec, "segments": seg_rows,
            "mma_types": dict(types), "handoffs": handoffs,
            "integral": tuple(integral),
        })
    return table


def precision_verdict(region: dict) -> str:
    types = region.get("mma_types") or {}
    if not types:
        return "none"
    return "mixed" if len(types) > 1 else next(iter(types))


def granule_verdict(region: dict) -> Optional[int]:
    gs = [s["granule"] for s in region["segments"] if s["granule"]]
    return min(gs) if gs else None


# ------------------------------------------------------------------ report
STRUCT_BEGIN = "=== STRUCT TABLE BEGIN ==="
STRUCT_END = "=== STRUCT TABLE END ==="


def _fmt_classes(classes: dict) -> str:
    parts = []
    for lab, row in sorted(classes.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
        w, lx = row.get("warps"), row.get("loop_x")
        # "x<ratio>?" = executed <ratio> times per barrier hit by warps whose count
        # is not readable: a nested loop's trips, warps outside the barrier, or
        # (arv_split) an unknown sync-side count — never a warp count
        tag = f"@{w:g}w" if w is not None else (f"@x{lx:g}?" if lx is not None else "")
        parts.append(f"{lab} x{row['count']}{tag}")
    return "  ".join(parts) if parts else "-"


def render_table(table: List[dict], out=print, ncta: Optional[int] = None) -> None:
    for ri, r in enumerate(table, 1):
        if r["warps"]:
            warps = f"{r['warps']} warps"
        elif r.get("arv_split"):
            warps = (f"warps=? ({r.get('bar_warps')} threads/32 on the operand, but BAR.ARV "
                     f"hits the same id: sync-side warp count unknown, who-column downgraded)")
        else:
            warps = "warps=? (CTA-wide barrier)"
        trips = ""
        if r["bar_exec"] and ncta and r["warps"]:
            trips = f"; trips={r['bar_exec'] / ncta / r['warps']:.0f} (ncta={ncta})"
        out(f"region {ri}: BAR.SYNC {r['key']} -> {warps}; {len(r['bars'])} barriers -> "
            f"{len(r['segments'])} segments; copies={r['copies']}{trips}")
        out("  seg | lines        | classes (count @ warps executing; x<n>? = per-barrier "
            "exec ratio with no warp reading: nested trips / warps outside the barrier / ARV split)")
        for s in r["segments"]:
            a, b = s["lines"]
            out(f"  {s['seg']:>3} | {a:>5}-{b:<6} | {_fmt_classes(s['classes'])}")
        out("  smem addressing per segment (immediate offsets of LDSM/STSM/LDS/STS):")
        for s in r["segments"]:
            if s["smem_offsets"]:
                out(f"    seg {s['seg']}: {len(s['smem_offsets'])} offsets, granule "
                    f"{hex(s['granule']) if s['granule'] else '-'}, strides "
                    f"{[hex(x) for x in s['strides']]}")
        for a, b, common in r["handoffs"]:
            out(f"  hand-off: seg {a} stores -> seg {b} loads at offsets "
                f"{[hex(x) for x in common[:6]]}{'…' if len(common) > 6 else ''}")
        out(f"  mma input types: {r['mma_types'] or 'none'} -> precision={precision_verdict(r)}")
        if r["integral"][1]:
            ok, tot = r["integral"]
            out(f"  warp attribution: {ok}/{tot} costed instructions read integral warps")


def self_check(table: List[dict], parsed: sass_tools.ParseResult,
               have_counts: bool) -> Tuple[List[str], Optional[dict]]:
    """(problems, the region the verdict line describes). The verdict region is
    the decisive region whose barrier executed most — the hot loop body, not a
    prologue/epilogue fence hit once per CTA; ties go to the most segments."""
    problems: List[str] = []
    if parsed.errors:
        problems.append(f"{len(parsed.errors)} profiler error/warning line(s) in the stream")
    if len(parsed.instructions) < MIN_INSTRUCTIONS:
        problems.append(f"only {len(parsed.instructions)} instructions (< {MIN_INSTRUCTIONS})")
    if not have_counts:
        problems.append("no per-PC execution counts — collect the source page as CSV "
                        "(ncu --import rep --page source --print-source sass --csv) and pass "
                        "source_csv_to_sass_and_counts()")
    decisive = [r for r in table if any(
        fam(lab) in DECISIVE_FAMILIES or lab in DECISIVE_FAMILIES
        for s in r["segments"] for lab in s["classes"])]
    if not table:
        problems.append("no recurring named barrier (BAR.SYNC <id>, <threads>) — the kernel "
                        "has no barrier-fenced construction to segment; write STRUCT-WAIVED "
                        "with this reason if that is what the SASS really shows")
    elif not decisive:
        problems.append("no region carries an MMA / LDSM / STSM family — the segmented "
                        "barriers fence something other than a tile construction")
    pick = (max(decisive, key=lambda r: (r["bar_exec"] or 0, len(r["segments"])))
            if decisive else None)
    if pick is not None and have_counts and not pick.get("arv_split"):
        ok, tot = pick["integral"]
        if tot == 0:
            problems.append("execution counts present but none joined the region's "
                            "instructions (address mismatch between SASS text and counts?)")
        elif ok / tot < WARP_INTEGRAL_SHARE:
            problems.append(f"warp attribution non-integral on {tot - ok}/{tot} instructions "
                            f"(predicated-off lanes the exec ratio cannot see) — the who-column "
                            f"is not believed")
    return problems, pick


def report(sass_text: str, exec_counts: Optional[Dict[str, int]] = None,
           kernel: str = "unnamed", ncta: Optional[int] = None, out=print) -> bool:
    """Parse, segment, render, self-check, verdict. True iff STRUCT-OK was printed."""
    parsed = sass_tools.parse_sass(sass_text)
    out(f"[struct] parsed {len(parsed.instructions)} instructions, "
        f"{len(exec_counts or {})} addresses with execution counts")
    table = decode(parsed.instructions, exec_counts) if parsed.instructions else []
    out(STRUCT_BEGIN)
    if table:
        render_table(table, out=out, ncta=ncta)
    else:
        out("(no barrier-fenced region found)")
    out(STRUCT_END)
    problems, pick = self_check(table, parsed, bool(exec_counts))
    if problems:
        for p in problems:
            out(f"self-check FAILED: {p}")
        out(f"STRUCT-FAIL reason={'; '.join(problems)}")
        return False
    assert pick is not None
    g = granule_verdict(pick)
    out(f"STRUCT-OK kernel={'_'.join(kernel.strip()[:80].split())} regions={len(table)} "
        f"segments={len(pick['segments'])} warps={pick['warps'] or '?'} "
        f"precision={precision_verdict(pick)} granule={hex(g) if g else 'none'}")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 struct_tools.py <source.csv | sass.txt> [--kernel NAME] [--ncta N]``.

    A ``.csv`` argument is the ncu source page (SASS + counts); any other file is
    SASS text alone (static table, no who-column — FAILs the gate by design)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: struct_tools.py <source.csv|sass.txt> [--kernel NAME] [--ncta N]")
        return 2
    path, kernel, ncta = args[0], "unnamed", None
    i = 1
    while i < len(args):
        if args[i] == "--kernel" and i + 1 < len(args):
            kernel = args[i + 1]
            i += 2
        elif args[i] == "--ncta" and i + 1 < len(args):
            ncta = int(args[i + 1])
            i += 2
        else:
            i += 1
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if path.lower().endswith(".csv"):
        sass, counts = source_csv_to_sass_and_counts(text)
    else:
        sass, counts = text, None
    return 0 if report(sass, counts, kernel=kernel, ncta=ncta) else 1


if __name__ == "__main__":
    sys.exit(main())
