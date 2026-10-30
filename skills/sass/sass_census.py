"""MMA operand-source census over a SASS dump — the ``[VERIFY: sass-derived]`` channel.

Why this exists: a milestone ledger can adjudicate an MMA A operand as RS
(register-sourced) while the gate checks only ``regs_per_thread`` — and a kernel
that reserves the register budget but still feeds the operand through smem then
passes the gate and gets sealed into the architecture (observed live).
"budget reserved" is not "path built". The observable that separates the two is
static and cheap: in SM90 SASS, an MMA's A operand is either a register
(``HGMMA... R24, R8, gdesc[UR6]`` — RS) or a shared-memory descriptor
(``HGMMA... R24, gdesc[UR4], gdesc[UR6]`` — SS). This module counts those forms
and renders one machine line the driver folds into the profiler digest, so a
milestone gate line can anchor to ``mma_a_reg`` / ``mma_a_desc`` the same way it
anchors to ``bankconf_st``.

Two halves, one file:

* **in-job half** — ``census_from_text`` + ``render_census_line`` (and ``main``):
  runs where the SASS dump is, prints exactly one ``SASS-OPERANDS ...`` or
  ``SASS-OPERANDS-FAIL ...`` line. Parsing is delegated to the vendored
  ``sass_tools`` (format-agnostic, self-checking) — no hand-rolled parser, per its
  history.
* **harness half** — ``parse_census_line``: scans a job log for the verdict line
  and returns digest-ready fields. Silence returns ``{}`` (operators that never
  run the census keep their digests unchanged — the channel is opt-in per op).

Trust note: the census line rides the same job stdout as everything else, so a
student kernel could in principle print a forged one. The line therefore carries
the kernel name and the MMA total, and ``parse_census_line`` keeps them for the
audit to cross-check against the NCU digest's dominant kernel; a census whose
kernel does not match the profiled one is evidence of nothing. The durable fence
(harness-owned eval scripts emitting the dump) is out of scope here.
"""

from __future__ import annotations

import re
import sys
from typing import Dict, Iterable, Optional, Union

try:  # vendored alongside this file in a job tree ("import sass_tools" via sys.path)
    import sass_tools  # type: ignore
except ImportError:  # harness-side import path
    from forgekernel.skills import sass_tools

# Families whose instructions carry MMA operand lists worth classifying. Mirrors
# sass_tools.SM90_TENSOR_FAMILIES's MMA subset — TMA/…GMMA-adjacent movement ops
# have no A/B operand semantics.
MMA_FAMILIES = ("HGMMA", "QGMMA", "IGMMA", "BGMMA")

# Operand forms. A register operand is "R<n>" (WGMMA register fragments are named
# by their first register); a shared-memory operand is a matrix descriptor,
# spelled desc[...] / gdesc[...] across dump styles.
_REG = re.compile(r"^-?R\d+$")
_DESC = re.compile(r"\bg?desc\s*\[")

_CENSUS_LINE = re.compile(
    r"SASS-OPERANDS\s+kernel=(\S+)\s+mma_total=(\d+)\s+a_reg=(\d+)\s+a_desc=(\d+)\s+a_other=(\d+)"
)


def _classify_a_operand(operands: str) -> str:
    """'reg' | 'desc' | 'other' for the A operand (second in the list).

    The operand string of an MMA is "acc, A, B, ..." — descriptors contain no
    top-level commas, so a plain split is faithful for both dump styles.
    """
    parts = [p.strip() for p in operands.split(",")]
    if len(parts) < 3:
        return "other"
    a = parts[1]
    if _REG.match(a):
        return "reg"
    if _DESC.search(a):
        return "desc"
    return "other"


def census_from_text(sass_text: str,
                     families: Iterable[str] = MMA_FAMILIES) -> Dict[str, int]:
    """SASS dump text -> {"mma_total", "mma_a_reg", "mma_a_desc", "mma_a_other"}."""
    fams = tuple(families)
    parsed = sass_tools.parse_sass(sass_text)
    counts = {"mma_total": 0, "mma_a_reg": 0, "mma_a_desc": 0, "mma_a_other": 0}
    for instr in parsed.instructions:
        if not instr.opcode.startswith(fams):
            continue
        counts["mma_total"] += 1
        counts["mma_a_" + _classify_a_operand(instr.operands)] += 1
    return counts


def render_census_line(census: Dict[str, int], kernel: str) -> str:
    """Exactly one machine-readable verdict line (see module docstring for trust).

    A dump with zero MMA instructions renders the FAIL form: for the kernels this
    channel exists for, "no MMA seen" means the dump is of the wrong kernel or the
    parse silently failed — both must be visible, not read as a_reg=0.
    """
    if census["mma_total"] == 0:
        return f"SASS-OPERANDS-FAIL kernel={kernel} reason=no-mma-instructions-parsed"
    return (f"SASS-OPERANDS kernel={kernel} mma_total={census['mma_total']} "
            f"a_reg={census['mma_a_reg']} a_desc={census['mma_a_desc']} "
            f"a_other={census['mma_a_other']}")


def parse_census_line(log_text: str) -> Dict[str, Union[int, str]]:
    """Job log -> digest fields ({} when the census never ran or FAILed).

    Takes the LAST census line: retried dumps append, and the final one describes
    the kernel that was actually shipped.
    """
    last: Optional[re.Match[str]] = None
    for m in _CENSUS_LINE.finditer(log_text):
        last = m
    if last is None:
        return {}
    return {
        "mma_census_kernel": last.group(1),
        "mma_total": int(last.group(2)),
        "mma_a_reg": int(last.group(3)),
        "mma_a_desc": int(last.group(4)),
        "mma_a_other": int(last.group(5)),
    }


# --- wgmma serialization census -------------------------------------------------
#
# ptxas decides per FUNCTION whether wgmma.mma_async may stay asynchronous. When
# it cannot prove the accumulator registers untouched between issue and wait
# (spilled accumulators, a non-wgmma instruction defining them, a wgmma on a
# divergent path, register pressure), it falls back to inserting a wait after
# EVERY wgmma — the whole kernel's MMA chains turn from pipelined into
# issue-one-wait-one, and it says so once in an advisory nobody reads inside a
# JIT log ("Potential Performance Loss: wgmma.mma_async instructions are
# serialized due to ..."). Observed live (on a gdn_prefill_fwd blackbox run,
# over three iterations): the student changed the SOLVER's structure, ptxas serialized the
# CONSUMER's 28 HGMMAs (D/H went 12/28 -> 28/28), the tree read -12 %, and the
# teacher — seeing only the perf number — declared the solver direction a
# forbidden zone. The compiler's verdict is static and countable: HGMMA
# instructions vs ``WARPGROUP.DEPBAR.LE gsb0, 0x0`` waits. A pipelined kernel
# waits once per chain (the reference: 100 HGMMA, 15 DEPBAR 0x0, ratio 0.15);
# a serialized one waits once per instruction (ratio ~1.0).
WGMMA_FAMILIES = ("HGMMA", "QGMMA", "IGMMA", "BGMMA")
WGMMA_SERIAL_RATIO_SEVERE = 0.9
WGMMA_SERIAL_RATIO_ELEVATED = 0.5
# how many instructions after an HGMMA a DEPBAR 0x0 may sit and still count as
# "tight" (issue-then-wait with nothing in between but the arrive/commit glue)
_TIGHT_WINDOW = 4
_DEPBAR0 = re.compile(r"\b0x0\b")
_PTXAS_SERIAL = re.compile(
    r"Potential Performance Loss:.*?serialized due to[^\n]*|wgmma\.mma_async instructions "
    r"are serialized[^\n]*", re.I)

_SERIAL_LINE = re.compile(
    r"WGMMA-SERIAL\s+kernel=(\S+)\s+wgmma=(\d+)\s+depbar0=(\d+)\s+tight=(\d+)\s+ratio=([\d.]+)"
)


def _is_depbar0(instr) -> bool:
    return (instr.opcode.startswith("WARPGROUP.DEPBAR")
            and _DEPBAR0.search(instr.operands) is not None)


def wgmma_serialization_from_text(sass_text: str) -> Dict[str, Union[int, float]]:
    """SASS dump text -> {"wgmma_total", "wgmma_depbar0", "wgmma_tight",
    "wgmma_serial_ratio"}.

    ``wgmma_depbar0`` counts ``WARPGROUP.DEPBAR.LE gsb0, 0x0`` (wait for ALL
    outstanding wgmma); ``wgmma_tight`` counts wgmma instructions followed by
    such a wait within ``_TIGHT_WINDOW`` instructions with no wgmma in between —
    the literal issue-one-wait-one shape. ``wgmma_serial_ratio`` = depbar0 /
    wgmma (0.0 when the dump has no wgmma).
    """
    instrs = sass_tools.parse_sass(sass_text).instructions
    total = depbar0 = tight = 0
    for i, ins in enumerate(instrs):
        if _is_depbar0(ins):
            depbar0 += 1
        if not ins.opcode.startswith(WGMMA_FAMILIES):
            continue
        total += 1
        for nxt in instrs[i + 1:i + 1 + _TIGHT_WINDOW]:
            if nxt.opcode.startswith(WGMMA_FAMILIES):
                break
            if _is_depbar0(nxt):
                tight += 1
                break
    ratio = round(depbar0 / total, 3) if total else 0.0
    return {"wgmma_total": total, "wgmma_depbar0": depbar0, "wgmma_tight": tight,
            "wgmma_serial_ratio": ratio}


def ptxas_serialization_hits(log_text: str) -> list:
    """The ptxas advisory lines (if the JIT let them through to the log)."""
    return [m.group(0).strip()[:200] for m in _PTXAS_SERIAL.finditer(log_text or "")]


def render_serial_line(d: Dict[str, Union[int, float]], kernel: str) -> str:
    """Exactly one machine line; a dump without wgmma renders the SKIP form (a
    non-wgmma kernel has nothing to serialize — not a zero ratio to celebrate)."""
    if not d.get("wgmma_total"):
        return f"WGMMA-SERIAL-SKIP kernel={kernel} reason=no-wgmma-instructions"
    return (f"WGMMA-SERIAL kernel={kernel} wgmma={d['wgmma_total']} "
            f"depbar0={d['wgmma_depbar0']} tight={d['wgmma_tight']} "
            f"ratio={d['wgmma_serial_ratio']}")


def parse_serial_line(log_text: str) -> Dict[str, Union[int, float, str]]:
    """Job log -> digest fields ({} when no WGMMA-SERIAL line). Last line wins.
    ``wgmma_ptxas_serialized`` is 1 when the log also carries the ptxas advisory
    (the compiler's own confession), else 0 — the ratio is the evidence either way."""
    last: Optional[re.Match[str]] = None
    for m in _SERIAL_LINE.finditer(log_text or ""):
        last = m
    if last is None:
        return {}
    return {
        "wgmma_census_kernel": last.group(1),
        "wgmma_total": int(last.group(2)),
        "wgmma_depbar0": int(last.group(3)),
        "wgmma_tight": int(last.group(4)),
        "wgmma_serial_ratio": float(last.group(5)),
        "wgmma_ptxas_serialized": 1 if ptxas_serialization_hits(log_text) else 0,
    }


def serialization_verdict(prev: Optional[dict], cur: dict) -> Optional[str]:
    """One sentence for the student when the ratio says the compiler fell back —
    a regression against the previous KEEP when there is one, a standing state
    otherwise; None when the chains are pipelined (or no census ran)."""
    r = cur.get("wgmma_serial_ratio")
    if r is None or not cur.get("wgmma_total"):
        return None
    if r < WGMMA_SERIAL_RATIO_ELEVATED:
        return None
    pr = (prev or {}).get("wgmma_serial_ratio")
    where = (f" (the previous KEEP was {pr})" if pr is not None else "")
    grade = "serialization" if r >= WGMMA_SERIAL_RATIO_SEVERE else "partial serialization"
    hint = ("This is a verdict ptxas passes on the **whole function**, not a sign that the "
            "algorithm in the part you changed is slow: as soon as anything anywhere in the "
            "function keeps it from proving that the accumulator is untouched between issue and "
            "wait (accumulator spill, a non-wgmma instruction writing the accumulator, a wgmma "
            "inside a branch, register pressure), it inserts a wait after every wgmma. Remove the "
            "trigger first (clear the spill / let only wgmma define the accumulator / move the "
            "wgmma out of the branch), then evaluate the algorithm itself.")
    regressed = pr is not None and pr < WGMMA_SERIAL_RATIO_ELEVATED
    head = ("**WGMMA serialization regression**" if regressed
            else f"**WGMMA {grade} (persistent)**")
    return (f"\n## Compiler collateral check\n- {head}: {cur.get('wgmma_total')} wgmma, "
            f"{cur.get('wgmma_depbar0')} DEPBAR 0x0, tight {cur.get('wgmma_tight')}, "
            f"ratio {r}{where}"
            + ("; the job log has ptxas's serialized notice" if cur.get("wgmma_ptxas_serialized")
               else "")
            + f". {hint}\n")


def main(argv: Optional[list] = None) -> int:
    """``python3 sass_census.py <sass-dump-file> [<kernel-name>]`` inside a job."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("SASS-OPERANDS-FAIL kernel=? reason=no-dump-file-argument")
        return 1
    path, kernel = args[0], (args[1] if len(args) > 1 else "unnamed")
    try:
        text = open(path).read()
    except OSError as e:
        print(f"SASS-OPERANDS-FAIL kernel={kernel} reason=unreadable-dump:{e.__class__.__name__}")
        return 1
    print(render_census_line(census_from_text(text), kernel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
