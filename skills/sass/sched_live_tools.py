"""Schedule transcription + register-liveness ledger over parsed SASS.

The two reconnaissance deliverables the counting probes never produced, and a
pair that must travel together (shown on an attention bwd kernel):

* **Schedule (§2d)** — the loop body as an ordered event stream: HGMMA runs
  labelled by accumulator chain, ``WARPGROUP.DEPBAR.LE`` immediates (the wait
  depths), commit/handshake/TMA/fence markers, elementwise runs collapsed to
  counts. The decisive read is the DEPBAR immediate in force when each
  elementwise run starts: ``LE 1`` means a chain is deliberately left in flight
  over it, ``LE 0`` means drain-everything.
* **Liveness (§2e)** — backward def-use over the doubled loop body (steady
  state for loop-carried values) with register-RANGE decode (HGMMA accumulator
  width from the shape token, RS A-operand M*K/256, LDSM/STSM ``.xN``, LDS/STS
  width suffixes). Output: peak live registers, where the peak lands, and a
  breakdown by producer class.

The schedule half has one mode per MMA ISA, chosen by a census of the page
(``mma_isa_census``) before anything is transcribed — B200 runs showed
that a Hopper-only grammar reads a Blackwell mainloop as "no MMA at all":

* ``wgmma`` (Hopper HGMMA): chains by accumulator base register, the decisive
  read is the ``WARPGROUP.DEPBAR.LE`` immediate in force at each elementwise
  run. Verdict ``SCHED-OK loop=… hgmma_runs=N depbars=N``.
* ``tcgen05`` (Blackwell UTC*MMA): chains by TMEM accumulator operand, the
  handoffs are ``UTCBAR`` (tcgen05.commit → mbarrier) and ``SYNCS`` waits, the
  accumulator traffic is ``LDTM``/``STTM``. Verdict
  ``SCHED-OK isa=tcgen05 loop=… utcmma_runs=N commits=N mbar_waits=N …``. An
  HMMA side path (the gdn triangular inverse) is transcribed inside it.
* ``mma_sync`` (per-warp HMMA/IMMA/DMMA/BMMA): chains by accumulator base
  register, the handoffs are ``DEPBAR.LE SB`` copy-group waits, named/cluster
  barriers and mbarrier. Verdict
  ``SCHED-OK isa=mma_sync loop=… mma_runs=N barriers=N ldsm=N …``.

Why both are mandatory together: a schedule without its liveness price invites
exactly the failed transplant once measured (ref schedule alone into
the student kernel: correct, 6% slower, spill 0 -> 10.8M sectors);
the liveness ledger is what prices the schedule's coexistence. Reference
numbers from FA4 bwd: peak 232/240 inside softmax pass A —
dP accumulator 64 fully live, S residue only 7 (the pack terminates its live
range), non-accumulator working set 93 words. That last figure is the student's
budget target.

Like ``sass_tools`` this file is seeded verbatim into the recon tree's
``notes/`` by the driver, so probes reach it with a bare import and never
hand-roll the analysis (probe1/2/9/13-rev1 each independently re-learned why).
Collection stays probe-side: run the oracle, profile with NCU, render the SASS
page — see sass_tools' module docstring for the proven path.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, NamedTuple, Tuple

try:  # seeded copy sits next to sass_tools in notes/
    import sass_tools as _st
except ImportError:  # in-package import (tests, tooling)
    from forgekernel.skills import sass_tools as _st

_REG = re.compile(r"\bR(\d+)\b")
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_SHAPE = re.compile(r"HGMMA\.(\d+)x(\d+)x(\d+)")
_PURE_REG = re.compile(r"R\d+(\.\w+)?$")

_SKIP = {"BRA", "BRX", "JMP", "NOP", "CALL", "RET", "EXIT", "YIELD"}
STORE_FAMS = {"STS", "STG", "ST", "STL", "STSM", "RED", "ATOM", "ATOMG"}
LOAD_FAMS = {"LDS", "LDG", "LD", "LDL"}
KNOWN_ROLE = {"IMAD", "IADD3", "LEA", "MOV", "SHF", "LOP3", "S2R", "CS2R",
              "MUFU", "FFMA", "FADD", "FMUL", "FSEL", "HADD2", "HMUL2",
              "HFMA2", "F2F", "F2FP", "PRMT", "I2F", "F2I", "SHFL", "SEL",
              "ISETP", "FSETP", "P2R", "R2P", "POPC", "FLO", "FMNMX", "IMNMX",
              "BREV", "SGXT", "PLOP3", "VOTEU", "LDC",
              # Blackwell (B200) pages added these to the known set
              "FMNMX3", "FADD2", "FMUL2", "VIMNMX", "VIADD", "R2UR", "S2UR",
              "WARPSYNC", "ENDCOLLECTIVE", "LDCU"}
_NO_GUESS = {"SYNCS", "NANOSLEEP", "BAR", "BSSY", "BSYNC", "ELECT", "MEMBAR",
             "FENCE", "VOTEU", "WARPGROUP", "DEPBAR", "LDGDEPBAR", "LDGSTS",
             "CCTL", "STAS", "LDTM", "STTM"}


def addr_int(ins) -> int:
    return int(ins.addr or "0", 16)


def fam_of(op: str) -> str:
    return op.split(".", 1)[0]


# The schedule this module models is Hopper's: warpgroup MMA accumulating in the
# REGISTER FILE, clustered into chains by accumulator base register, waited on by
# ``WARPGROUP.DEPBAR.LE`` immediates whose depth is the decisive read. Blackwell's
# tcgen05 has neither half -- it issues through ``UTC*MMA``, keeps the accumulator
# in TMEM (so there is no accumulator register to cluster by) and synchronises on
# mbarrier (so there is no DEPBAR immediate to read).
#
# Left unsaid, that mismatch does not read as "wrong instrument", it reads as
# "nothing to report": ``fam_of("UTCHMMA")`` matches no known family, so transcribe
# files every tcgen05 MMA as ELEMENTWISE work and the run ends in
# ``SCHED-FAIL ... hgmma_runs=0 depbars=0``. Measured on a gdn_prefill B200
# reference whose mainloop is UTCHMMA: the transcription printed
# ``elementwise x41 [UIADD3x11,UMOVx11,UTCHMMAx8,USELx4]`` -- the MMA counted as
# elementwise -- and the teacher spent the rest of its round budget trying to make
# a Hopper tool answer a Blackwell question. So the census below runs first and the
# report says which ISA it is looking at.
WGMMA_FAMILIES = ("HGMMA", "QGMMA", "IGMMA", "BGMMA")
# mma.sync (per-warp, synchronous: HMMA / IMMA / DMMA / BMMA). Its accumulator IS
# in registers, but there is no warpgroup chain and no DEPBAR.LE immediate to read
# either -- the wait structure is the scoreboard plus cp.async / ldmatrix groups.
# The dsa_sparse B200 reference is an HMMA mainloop: it came back
# ``SCHED-FAIL ... hgmma_runs=0 depbars=0`` six times in one day, the same
# zeroed verdict as the tcgen05 case, for the same reason (wrong instrument).
MMA_SYNC_FAMILIES = ("HMMA", "IMMA", "DMMA", "BMMA")
_TCGEN05_MMA = re.compile(r"^UTC[A-Z0-9]*MMA$")


class MmaCensus(NamedTuple):
    """How many MMA instructions of each ISA the page carries, plus the family
    histogram. ``mode`` picks the schedule grammar ``report`` transcribes with."""
    wgmma: int
    tcgen05: int
    mma_sync: int
    fams: Counter

    @property
    def mode(self) -> "str | None":
        """``wgmma`` / ``tcgen05`` / ``mma_sync`` — the grammar that reads this
        page — or None for a page with NO MMA at all (a helper kernel, a
        prologue-only dump): that is a wrong page, not a wrong instrument, and
        it keeps failing loudly rather than opening the gate on a false premise.
        Precedence wgmma > tcgen05 > mma_sync: a tcgen05 mainloop with an HMMA
        side path (the gdn triangular inverse: 88 UTCHMMA + 22 HMMA on one
        page) is a tcgen05 kernel whose transcription includes the HMMA runs."""
        if self.wgmma:
            return "wgmma"
        if self.tcgen05:
            return "tcgen05"
        if self.mma_sync:
            return "mma_sync"
        return None

    @property
    def not_applicable(self) -> "str | None":
        """The ``isa=`` / ``reason=`` pair the Hopper grammar would refuse with.
        Kept for the seeded copies still in flight and for callers that only
        know the wgmma model; ``report`` itself now transcribes both other ISAs
        (see the module docstring) and never prints SCHED-NOTAPPLICABLE."""
        if self.wgmma:
            return None
        if self.tcgen05:
            return "isa=tcgen05 reason=accumulator-in-tmem-no-depbar"
        if self.mma_sync:
            return "isa=mma-sync reason=per-warp-mma-no-warpgroup-chain"
        return None


def mma_isa_census(instrs) -> MmaCensus:
    fams: Counter = Counter()
    wgmma = tcgen05 = mma_sync = 0
    for ins in instrs:
        f = fam_of(ins.opcode)
        if f in WGMMA_FAMILIES:
            wgmma += 1
        elif _TCGEN05_MMA.match(f):
            tcgen05 += 1
        elif f in MMA_SYNC_FAMILIES:
            mma_sync += 1
        else:
            continue
        fams[f] += 1
    return MmaCensus(wgmma, tcgen05, mma_sync, fams)


class Chain:
    """One WGMMA accumulator chain: identity = (accumulator base, MMA shape)."""

    def __init__(self, cid, acc_base, shape, a_is_reg, zero_init):
        self.cid, self.acc_base, self.shape = cid, acc_base, shape
        self.a_is_reg, self.zero_init = a_is_reg, zero_init
        self.count = 0

    @property
    def acc_width(self) -> int:
        return (int(self.shape[0]) * int(self.shape[1])) // 128

    def name(self) -> str:
        return f"chain{self.cid}"

    def label(self) -> str:
        kind = "A=REG" if self.a_is_reg else "A=DESC"
        init = "zero-init" if self.zero_init else "accum"
        return (f"chain{self.cid}[R{self.acc_base}..+{self.acc_width}w "
                f"{self.shape[0]}x{self.shape[1]} {kind} {init}]")


def is_zero_init(ins) -> bool:
    flat = ins.operands.replace(" ", "")
    return ",RZ," in flat + "," or flat.endswith(",RZ") or "!UPT" in flat


def build_chains(body) -> Tuple[List[Chain], Dict[int, Chain]]:
    """Cluster HGMMAs by accumulator base register; returns (chains, index->chain)."""
    chains, order, owner = {}, [], {}
    for i, ins in enumerate(body):
        if fam_of(ins.opcode) != "HGMMA":
            continue
        m = _SHAPE.search(ins.opcode)
        regs = _REG.findall(ins.operands)
        if not (m and regs):
            continue
        key = (int(regs[0]), m.group(1), m.group(2))
        if key not in chains:
            ops = [o.strip() for o in ins.operands.split(",")]
            a_is_reg = len(ops) > 1 and bool(_PURE_REG.fullmatch(ops[1]))
            chains[key] = Chain(len(order), key[0], m.groups(), a_is_reg,
                                is_zero_init(ins))
            order.append(key)
        chains[key].count += 1
        owner[i] = chains[key]
    return [chains[k] for k in order], owner


def find_loop(instrs, mma_fams=("HGMMA",), envelope="hgmma-envelope") -> Tuple[int, int, str]:
    """Largest backward-branch region; falls back to the MMA envelope.

    ``mma_fams`` names the families whose first..last span is the fallback
    body when no backward branch exists (the wgmma default keeps the JOIN
    callers in stall_tools byte-identical); a callable takes the family."""
    addr_ix = {addr_int(ins): i for i, ins in enumerate(instrs)}
    best = None
    for i, ins in enumerate(instrs):
        if fam_of(ins.opcode) not in ("BRA", "BRX"):
            continue
        hexes = _HEX.findall(ins.operands)
        if not hexes:
            continue
        j = addr_ix.get(int(hexes[-1], 16))
        if j is not None and j < i and (best is None or (i - j) > (best[1] - best[0])):
            best = (j, i)
    if best:
        return best[0], best[1], "backedge"
    is_mma = mma_fams if callable(mma_fams) else (lambda f: f in mma_fams)
    hg = [i for i, ins in enumerate(instrs) if is_mma(fam_of(ins.opcode))]
    if hg:
        return hg[0], hg[-1], envelope
    return 0, len(instrs) - 1, "whole-kernel"


def transcribe(body, owner) -> List[Tuple[str, str]]:
    """Loop body -> ordered (kind, text) events; adjacent duplicates collapse."""
    events, ew, n_shfl = [], Counter(), 0

    def flush_ew():
        nonlocal ew, n_shfl
        total = sum(ew.values())
        if total >= 8:
            top = ",".join(f"{k}x{v}" for k, v in ew.most_common(4))
            tag = f" SHFLx{n_shfl}" if n_shfl else ""
            events.append(("EW", f"elementwise x{total} [{top}]{tag}"))
        ew, n_shfl = Counter(), 0

    run_chain, run_n = None, 0

    def flush_hg():
        nonlocal run_chain, run_n
        if run_chain is not None:
            events.append(("HG", f"{run_chain.label()} x{run_n}"))
        run_chain, run_n = None, 0

    for i, ins in enumerate(body):
        f = fam_of(ins.opcode)
        if f == "HGMMA":
            flush_ew()
            ch = owner.get(i)
            if ch is run_chain:
                run_n += 1
            else:
                flush_hg()
                run_chain, run_n = ch, 1
            continue
        if f in _SKIP:
            continue
        emit = None
        if f == "WARPGROUP":
            if "DEPBAR" in ins.opcode:
                im = _HEX.findall(ins.operands)
                emit = ("DEPBAR", f"** DEPBAR.LE {im[-1] if im else '?'} **")
            else:
                emit = ("WG", ins.opcode)
        elif f in ("BAR", "BSSY", "BSYNC"):
            emit = ("BAR", f"{ins.opcode} {ins.operands[:24]}")
        elif f == "SYNCS":
            kind = ("arrive" if "ARRIVE" in ins.opcode else
                    "trywait" if ("TRYWAIT" in ins.opcode or "PHASECHK" in ins.opcode)
                    else ins.opcode)
            emit = ("SYNCS", f"mbarrier.{kind}")
        elif f == "NANOSLEEP":
            emit = ("SLEEP", "NANOSLEEP(spin)")
        elif f in ("STSM", "LDSM"):
            emit = (f, ins.opcode)
        elif f.startswith("UTMA") or f.startswith("UBLK"):
            emit = ("TMA", ins.opcode)
        elif f in ("MEMBAR", "FENCE", "ELECT"):
            emit = ("FENCE", ins.opcode)
        if emit:
            flush_ew()
            flush_hg()
            events.append(emit)
        else:
            ew[f] += 1
            if f == "SHFL":
                n_shfl += 1
    flush_ew()
    flush_hg()

    out, i = [], 0
    while i < len(events):
        j = i
        while j + 1 < len(events) and events[j + 1] == events[i]:
            j += 1
        k, txt = events[i]
        out.append((k, txt + (f"  x{j - i + 1}" if j > i else "")))
        i = j + 1
    return out


def sched_verdict(events) -> List[str]:
    """Per elementwise run: the DEPBAR immediate in force when it starts.

    The immediate is extracted by pattern, not position: transcribe()'s
    duplicate-collapse appends "  x<N>" to the event text (adjacent identical
    DEPBARs are common when a short EW run between them is dropped by the
    >=8-op floor), and taking the last whitespace token published "x2" as the
    wait depth — corrupting the single most decisive schedule fact.
    """
    pre, depth = [], None
    for k, txt in events:
        if k == "DEPBAR":
            m = re.search(r"DEPBAR\.LE\s+(\S+?)\s*(?:\*|$)", txt)
            depth = m.group(1) if m else "?"
        elif k == "HG":
            depth = None
        elif k == "EW":
            pre.append(depth if depth is not None else "none")
    return pre


# --------------------------------------------------------------------------
# mma.sync (HMMA / IMMA / DMMA / BMMA) and tcgen05 (UTC*MMA) schedule grammars.
#
# Both were first written by teachers under round pressure — a dsa_sparse B200
# run's p5_sched_live_hmma.py (SCHED-OK mma_runs=296 barriers=22 ldsm=87,
# rejected by the driver's HGMMA-only regex until the report double-wrote a
# DELIVERED-MISMATCH line) and a gdn_prefill B200 run's recon_04 STEP 8
# event stream (1497 events; the report's §6 substitute for a waived channel).
# Sedimented here so the channel is delivered by the seeded tool, on its own
# verdict line, and the driver's acceptance regex knows all three forms.

# (accumulator, A, B) register widths per thread for one mma.sync shape token.
# Only the liveness ledger reads these; the schedule needs the chain identity.
_MMA_SYNC_WIDTHS = {"16816": (4, 4, 2), "1688": (4, 2, 1), "1684": (4, 2, 1),
                    "16832": (4, 4, 2), "16864": (4, 4, 2), "884": (8, 2, 2)}
_MMA_SYNC_SHAPE = re.compile(r"\.(\d{3,5})(?:\.|$)")
# a bracketed operand with its prefix word: ``tmem[UR40]`` / ``[UR40+0x80]`` /
# ``gdesc[UR18]`` — the accumulator is the first one that is NOT a descriptor
_BRACKET = re.compile(r"([A-Za-z]*)\[([^\]]+)\]")
_MBAR_ADDR = re.compile(r"\[([^\]]+)\]")
_TMEM_X = re.compile(r"\.x(\d+)\b", re.I)


def _mma_sync_widths(opcode: str) -> Tuple[int, int, int]:
    m = _MMA_SYNC_SHAPE.search(opcode)
    acc, a, b = _MMA_SYNC_WIDTHS.get(m.group(1) if m else "", (4, 2, 2))
    up = opcode.upper()
    if fam_of(opcode) == "HMMA" and ".F32" not in up and ".F16" in up:
        acc = max(1, acc // 2)          # f16 accumulator packs two per register
    return acc, a, b


class SyncChain:
    """One mma.sync accumulator chain: identity = (accumulator base, mnemonic)."""

    def __init__(self, cid, acc_base, opcode, zero_init):
        self.cid, self.acc_base, self.opcode = cid, acc_base, opcode
        self.zero_init = zero_init
        self.count = 0
        self.acc_width = _mma_sync_widths(opcode)[0]

    def name(self) -> str:
        return f"chain{self.cid}"

    def label(self) -> str:
        init = "zero-init" if self.zero_init else "accum"
        return f"chain{self.cid}[R{self.acc_base}..+{self.acc_width}w {self.opcode} {init}]"


class TmemChain:
    """One tcgen05 accumulator chain: identity = (TMEM accumulator operand,
    mnemonic). The accumulator lives in TMEM, so the chain owns no registers;
    ``!UPT`` on the issue is the zero-init (accumulate-disable) flag."""

    acc_width = 0

    def __init__(self, cid, acc, opcode, zero_init):
        self.cid, self.acc, self.opcode = cid, acc, opcode
        self.zero_init = zero_init
        self.count = 0

    def name(self) -> str:
        return f"chain{self.cid}"

    def label(self) -> str:
        init = "zero-init" if self.zero_init else "accum"
        return f"chain{self.cid}[tmem {self.acc} {self.opcode} {init}]"


def build_mma_sync_chains(body, chains=None, order=None, owner=None):
    """Cluster mma.sync issues by (accumulator base register, mnemonic).

    The optional dicts let the tcgen05 transcription add the page's HMMA side
    path to its own chain table (one numbering, one owner map)."""
    chains = {} if chains is None else chains
    order = [] if order is None else order
    owner = {} if owner is None else owner
    for i, ins in enumerate(body):
        if fam_of(ins.opcode) not in MMA_SYNC_FAMILIES:
            continue
        regs = _REG.findall(ins.operands)
        if not regs:
            continue
        key = ("sync", int(regs[0]), ins.opcode)
        if key not in chains:
            chains[key] = SyncChain(len(order), key[1], ins.opcode, is_zero_init(ins))
            order.append(key)
        chains[key].count += 1
        owner[i] = chains[key]
    return [chains[k] for k in order], owner


def build_tcgen05_chains(body):
    """Cluster UTC*MMA issues by TMEM accumulator operand; the HMMA side path
    (if any) joins the same table. Returns (chains, index->chain)."""
    chains, order, owner = {}, [], {}
    for i, ins in enumerate(body):
        if not _TCGEN05_MMA.match(fam_of(ins.opcode)):
            continue
        acc = next((m.group(2).replace(" ", "") for m in _BRACKET.finditer(ins.operands)
                    if m.group(1).lower() in ("", "tmem")), "?")
        key = ("tmem", acc, ins.opcode)
        if key not in chains:
            chains[key] = TmemChain(len(order), acc, ins.opcode, "!UPT" in ins.operands)
            order.append(key)
        chains[key].count += 1
        owner[i] = chains[key]
    build_mma_sync_chains(body, chains, order, owner)
    return [chains[k] for k in order], owner


def _mbar_text(ins) -> str:
    m = _MBAR_ADDR.search(ins.operands)
    return m.group(1).replace(" ", "") if m else "?"


def _sync_kind(op: str) -> str:
    return ("arrive" if "ARRIVE" in op else
            "trywait" if ("TRYWAIT" in op or "PHASECHK" in op) else
            "exch" if "EXCH" in op else op)


def _transcribe_isa(body, owner, mma_kind_of) -> List[Tuple[str, str]]:
    """Shared event-stream builder for the non-wgmma grammars.

    ``mma_kind_of(fam)`` returns the event kind for an MMA family ("UMMA" /
    "MMA") or None. Handoff markers are transcribed with their object (the
    mbarrier address, the commit target, the copy-group immediate) because
    on these ISAs the object IS the schedule: there is no DEPBAR depth to
    read, only which barrier each run waits on."""
    events, ew, n_shfl = [], Counter(), 0

    def flush_ew():
        nonlocal ew, n_shfl
        total = sum(ew.values())
        if total >= 8:
            top = ",".join(f"{k}x{v}" for k, v in ew.most_common(4))
            tag = f" SHFLx{n_shfl}" if n_shfl else ""
            events.append(("EW", f"elementwise x{total} [{top}]{tag}"))
        ew, n_shfl = Counter(), 0

    run_chain, run_kind, run_n = None, None, 0

    def flush_mma():
        nonlocal run_chain, run_kind, run_n
        if run_chain is not None:
            events.append((run_kind, f"{run_chain.label()} x{run_n}"))
        run_chain, run_kind, run_n = None, None, 0

    for i, ins in enumerate(body):
        f = fam_of(ins.opcode)
        kind = mma_kind_of(f)
        if kind is not None:
            flush_ew()
            ch = owner.get(i)
            if ch is run_chain and ch is not None:
                run_n += 1
            else:
                flush_mma()
                run_chain, run_kind, run_n = ch, kind, 1
            continue
        if f in _SKIP or f in ("BSSY", "BSYNC"):
            continue
        emit = None
        if f == "UTCBAR":
            emit = ("COMMIT", f"tcgen05.commit -> [{_mbar_text(ins)}]")
        elif f in ("UTCATOMSWS", "UTCATOM", "UTCALLOC", "UTCDEALLOC"):
            emit = ("TMEM", f"{ins.opcode} {ins.operands[:32]}")
        elif f in ("LDTM", "STTM"):
            emit = (f, ins.opcode)
        elif f == "USETMAXREG":
            im = _HEX.findall(ins.operands)
            emit = ("SETMAXREG", f"{ins.opcode} {im[-1] if im else '?'}")
        elif f == "SYNCS":
            emit = ("SYNCS", f"mbarrier.{_sync_kind(ins.opcode)} [{_mbar_text(ins)}]")
        elif f == "DEPBAR":
            im = _HEX.findall(ins.operands)
            sb = re.search(r"\bSB\d\b", ins.operands)
            emit = ("DEPBAR", f"** DEPBAR.LE {sb.group(0) if sb else 'SB?'} "
                              f"{im[-1] if im else '?'} **")
        elif f == "LDGSTS":
            emit = ("CPASYNC", ins.opcode)
        elif f == "LDGDEPBAR":
            emit = ("CPCOMMIT", "LDGDEPBAR")
        elif f == "WARPGROUP":
            emit = ("WG", ins.opcode)
        elif f == "BAR":
            emit = ("BAR", f"{ins.opcode} {ins.operands[:24]}")
        elif f in ("UCGABAR_ARV", "UCGABAR_WAIT"):
            emit = ("CLUSTERBAR", ins.opcode)
        elif f == "STAS":
            emit = ("DSMEM", ins.opcode)
        elif f == "NANOSLEEP":
            emit = ("SLEEP", "NANOSLEEP(spin)")
        elif f in ("STSM", "LDSM"):
            emit = (f, ins.opcode)
        elif f.startswith("UTMA") or f.startswith("UBLK"):
            emit = ("TMA", ins.opcode)
        elif f in ("MEMBAR", "FENCE", "ELECT"):
            emit = ("FENCE", ins.opcode)
        if emit:
            flush_ew()
            flush_mma()
            events.append(emit)
        else:
            # an MMA run ends at the first non-MMA instruction: on these ISAs
            # nothing like WARPGROUP.ARRIVE follows every issue, so the run must
            # be closed here or it would be emitted AFTER the elementwise run
            # that follows it (order is the whole point of the transcription)
            flush_mma()
            ew[f] += 1
            if f == "SHFL":
                n_shfl += 1
    flush_ew()
    flush_mma()

    out, i = [], 0
    while i < len(events):
        j = i
        while j + 1 < len(events) and events[j + 1] == events[i]:
            j += 1
        k, txt = events[i]
        out.append((k, txt + (f"  x{j - i + 1}" if j > i else "")))
        i = j + 1
    return out


def transcribe_mma_sync(body, owner) -> List[Tuple[str, str]]:
    """Loop body of an mma.sync kernel -> ordered (kind, text) events."""
    return _transcribe_isa(body, owner,
                           lambda f: "MMA" if f in MMA_SYNC_FAMILIES else None)


def transcribe_tcgen05(body, owner) -> List[Tuple[str, str]]:
    """Loop body of a tcgen05 kernel -> ordered (kind, text) events; the HMMA
    side path (if any) shows up as MMA runs in the same stream."""
    return _transcribe_isa(body, owner,
                           lambda f: ("UMMA" if _TCGEN05_MMA.match(f) else
                                      "MMA" if f in MMA_SYNC_FAMILIES else None))


_HANDOFF_KINDS = ("COMMIT", "SYNCS", "DEPBAR", "CPCOMMIT", "BAR", "CLUSTERBAR")


def handoff_verdict(events) -> List[str]:
    """Per elementwise run: the handoff in force when it starts — the last
    commit / mbarrier wait / DEPBAR / barrier event before it, or the MMA run
    it follows directly ("<kind>:<chain>"). The mma.sync/tcgen05 analogue of
    ``sched_verdict``'s DEPBAR depth: with no depth immediate to read, the
    schedule fact is WHICH object each elementwise segment runs behind."""
    pre, last = [], "none"
    for k, txt in events:
        if k in _HANDOFF_KINDS:
            last = txt.strip("* ").split("  x")[0]
        elif k in ("UMMA", "MMA"):
            last = f"{k}:{txt.split('[', 1)[0]}"
        elif k == "EW":
            pre.append(last)
    return pre


def _mma_sync_report(instrs, census, out) -> Tuple[bool, bool]:
    lo, hi, how = find_loop(instrs, lambda f: f in MMA_SYNC_FAMILIES, "mma-envelope")
    body = instrs[lo:hi + 1]
    out(f"[loop] {how}: {len(body)} instrs, "
        f"addr {hex(addr_int(instrs[lo]))}..{hex(addr_int(instrs[hi]))}")
    chains, owner = build_mma_sync_chains(body)
    out(f"[chains] {len(chains)} mma.sync accumulator chains in body:")
    for c in chains:
        out(f"    {c.label()} count={c.count}")
    events = transcribe_mma_sync(body, owner)
    out(f"\n=== schedule transcription (isa=mma_sync, {len(events)} events) ===")
    for n, (k, txt) in enumerate(events):
        out(f"  {n:3d} {txt}")
    pre = handoff_verdict(events)
    kinds = Counter(k for k, _ in events)
    mma_n, bar_n = kinds["MMA"], kinds["BAR"] + kinds["CLUSTERBAR"]
    ldsm_n, dep_n = kinds["LDSM"], kinds["DEPBAR"]
    wait_n = sum(1 for k, t in events if k == "SYNCS" and "trywait" in t)
    # A body with MMA runs and at least one handoff object to hang them on.
    # The floor is deliberately low: the dsa_sparse B200 page (296 runs, 22
    # barriers, 87 LDSM) is the large end; a small decode mainloop may carry
    # two runs and a single named barrier and still be a schedule.
    ok_a = mma_n >= 2 and (bar_n >= 1 or ldsm_n >= 4 or dep_n >= 1 or wait_n >= 1)
    out(f"\nSCHED-{'OK' if ok_a else 'FAIL'} isa=mma_sync loop={how} mma_runs={mma_n} "
        f"barriers={bar_n} ldsm={ldsm_n} depbars={dep_n} "
        f"cp_async_groups={kinds['CPCOMMIT']} mbar_waits={wait_n} ew_runs={len(pre)} "
        f"handoff_before_each_ew={pre}")
    if not ok_a:
        out("[sched] raw samples for debugging:")
        for ins in body[:12]:
            out(f"    {ins.addr}  {ins.opcode} {ins.operands[:60]}")
    ok_b = _liveness_report(body, owner, out)
    return ok_a, ok_b


def _tcgen05_report(instrs, census, out) -> Tuple[bool, bool]:
    is_mma = lambda f: bool(_TCGEN05_MMA.match(f)) or f in MMA_SYNC_FAMILIES  # noqa: E731
    lo, hi, how = find_loop(instrs, is_mma, "umma-envelope")
    body = instrs[lo:hi + 1]
    out(f"[loop] {how}: {len(body)} instrs, "
        f"addr {hex(addr_int(instrs[lo]))}..{hex(addr_int(instrs[hi]))}")
    chains, owner = build_tcgen05_chains(body)
    n_tmem = sum(1 for c in chains if isinstance(c, TmemChain))
    out(f"[chains] {len(chains)} accumulator chains in body "
        f"({n_tmem} tcgen05/TMEM, {len(chains) - n_tmem} mma.sync side path):")
    for c in chains:
        out(f"    {c.label()} count={c.count}")
    events = transcribe_tcgen05(body, owner)
    out(f"\n=== schedule transcription (isa=tcgen05, {len(events)} events) ===")
    for n, (k, txt) in enumerate(events):
        out(f"  {n:3d} {txt}")
    pre = handoff_verdict(events)
    kinds = Counter(k for k, _ in events)
    umma_n, commit_n = kinds["UMMA"], kinds["COMMIT"]
    wait_n = sum(1 for k, t in events if k == "SYNCS" and "trywait" in t)
    arrive_n = sum(1 for k, t in events if k == "SYNCS" and "arrive" in t)
    # UMMA issue runs with the commit that publishes them and a wait that
    # consumes them: the three reads a tcgen05 schedule is made of. Missing any
    # one means the page is not the mainloop (a prologue, a helper kernel).
    ok_a = umma_n >= 2 and commit_n >= 1 and wait_n >= 1
    out(f"\nSCHED-{'OK' if ok_a else 'FAIL'} isa=tcgen05 loop={how} utcmma_runs={umma_n} "
        f"commits={commit_n} mbar_waits={wait_n} mbar_arrives={arrive_n} "
        f"tmem_ld={kinds['LDTM']} tmem_st={kinds['STTM']} mma_sync_runs={kinds['MMA']} "
        f"barriers={kinds['BAR'] + kinds['CLUSTERBAR']} ew_runs={len(pre)} "
        f"handoff_before_each_ew={pre}")
    if not ok_a:
        out("[sched] raw samples for debugging:")
        for ins in body[:12]:
            out(f"    {ins.addr}  {ins.opcode} {ins.operands[:60]}")
    ok_b = _liveness_report(body, owner, out)
    return ok_a, ok_b


def _width_from(op: str) -> int:
    return 4 if ".128" in op else (2 if ".64" in op else 1)


def _stsm_width(op: str) -> int:
    u = op.upper()
    return 4 if ".X4" in u else (2 if ".X2" in u else 1)


def reg_accesses(ins) -> Tuple[set, set, bool]:
    """(defs, uses, guessed-role?) as vector-register index sets, range-decoded."""
    f = fam_of(ins.opcode)
    ops = [o.strip() for o in ins.operands.split(",") if o.strip()]
    rin = lambda s: [int(x) for x in _REG.findall(s)]  # noqa: E731
    defs, uses = set(), set()
    guessed = False
    if f == "HGMMA":
        m = _SHAPE.search(ins.opcode)
        M, N, K = (int(x) for x in m.groups()) if m else (64, 64, 16)
        r = rin(ops[0]) if ops else []
        if r:
            d0 = r[0]
            defs |= set(range(d0, d0 + (M * N) // 128))
            if not is_zero_init(ins):
                uses |= set(range(d0, d0 + (M * N) // 128))
        if len(ops) > 1 and _PURE_REG.fullmatch(ops[1]):
            a0 = rin(ops[1])[0]
            uses |= set(range(a0, a0 + max(1, (M * K) // 256)))
    elif f in MMA_SYNC_FAMILIES:
        # D, A, B, C — per-thread fragments; C is the accumulator input (RZ or
        # the same base as D when accumulating in place).
        acc_w, a_w, b_w = _mma_sync_widths(ins.opcode)
        r = rin(ops[0]) if ops else []
        if r:
            defs |= set(range(r[0], r[0] + acc_w))
        for o, w in zip(ops[1:4], (a_w, b_w, acc_w)):
            if _PURE_REG.fullmatch(o):
                b0 = rin(o)[0]
                uses |= set(range(b0, b0 + w))
    elif _TCGEN05_MMA.match(f) or f in ("UTCBAR", "UTCATOMSWS", "UTCATOM",
                                        "UTCALLOC", "UTCDEALLOC", "USETMAXREG"):
        # descriptors and TMEM addresses ride uniform registers; no vector
        # register is defined or consumed by the issue itself
        pass
    elif f in ("LDTM", "STTM"):
        # tcgen05.ld / tcgen05.st: .xN = N 32-bit registers per thread
        m = _TMEM_X.search(ins.opcode)
        n = int(m.group(1)) if m else 1
        r = rin(ops[0]) if ops else []
        if f == "LDTM":
            if r:
                defs |= set(range(r[0], r[0] + n))
            for o in ops[1:]:
                uses |= set(rin(o))
        else:
            for o in ops:
                if "[" not in o and rin(o):
                    d0 = rin(o)[0]
                    uses |= set(range(d0, d0 + n))
                    break
            for o in ops:
                if "[" in o:
                    uses |= set(rin(o))
    elif f == "LDSM":
        n = _stsm_width(ins.opcode)
        r = rin(ops[0]) if ops else []
        if r:
            defs |= set(range(r[0], r[0] + n))
        for o in ops[1:]:
            uses |= set(rin(o))
    elif f in STORE_FAMS:
        n = _stsm_width(ins.opcode) if f == "STSM" else _width_from(ins.opcode)
        data = None
        for o in reversed(ops):
            if "[" not in o and rin(o):
                data = rin(o)[0]
                break
        if data is not None:
            uses |= set(range(data, data + n))
        for o in ops:
            if "[" in o:
                uses |= set(rin(o))
    elif f in LOAD_FAMS:
        n = _width_from(ins.opcode)
        r = rin(ops[0]) if ops else []
        if r:
            defs |= set(range(r[0], r[0] + n))
        for o in ops[1:]:
            uses |= set(rin(o))
    else:
        if ops:
            d = rin(ops[0])
            if d:
                defs.add(d[0])
            for o in ops[1:]:
                uses |= set(rin(o))
        if (f not in KNOWN_ROLE and f not in _SKIP and f not in _NO_GUESS
                and not f.startswith(("UTMA", "UBLK", "U"))):
            guessed = True
    return defs, uses, guessed


def liveness(body):
    """(per-instr live counts, peak index, live set at peak, guessed count).

    The body is doubled so loop-carried values reach steady state; counts are
    read off the first copy.
    """
    dbl = body + body
    live, guessed_n = set(), 0
    counts = [0] * len(dbl)
    for i in range(len(dbl) - 1, -1, -1):
        defs, uses, g = reg_accesses(dbl[i])
        guessed_n += bool(g)
        live -= defs
        live |= uses
        counts[i] = len(live)
    first = counts[:len(body)]
    peak_i = max(range(len(first)), key=lambda i: first[i])
    live = set()
    peak_set = frozenset()
    for i in range(len(dbl) - 1, -1, -1):
        defs, uses, _ = reg_accesses(dbl[i])
        live -= defs
        live |= uses
        if i == peak_i:
            peak_set = frozenset(live)
            break
    return first, peak_i, peak_set, guessed_n // 2


def provenance(body, peak_i, peak_set, owner) -> Counter:
    """Classify each live register at the peak by its (cyclically) nearest def."""
    n = len(body)
    last_def = {}
    for k in range(n):
        i = (peak_i - 1 - k) % n
        defs, _, _ = reg_accesses(body[i])
        for r in defs:
            if r not in last_def:
                last_def[r] = i
    cls = Counter()
    for r in peak_set:
        i = last_def.get(r)
        if i is None:
            cls["undef/param"] += 1
            continue
        f = fam_of(body[i].opcode)
        if f == "HGMMA" or f in MMA_SYNC_FAMILIES:
            ch = owner.get(i)
            cls[f"acc:{ch.name() if ch else '?'}"] += 1
        elif f == "LDTM":
            cls["tmem-load"] += 1
        elif f in ("F2F", "F2FP", "PRMT", "HADD2", "HMUL2", "HFMA2"):
            cls["pack/f16"] += 1
        elif f == "SHFL":
            cls["shfl-temp"] += 1
        elif f in ("LDS", "LDSM", "LDC"):
            cls["smem/const-load"] += 1
        elif f in ("IMAD", "IADD3", "LEA", "MOV", "SHF", "LOP3", "S2R", "CS2R",
                   "SGXT"):
            cls["addr/temp"] += 1
        elif f in ("MUFU", "FFMA", "FADD", "FMUL", "FSEL", "FMNMX"):
            cls["fp-temp"] += 1
        else:
            cls[f] += 1
    return cls


def report(sass_text: str, out=print) -> Tuple[bool, bool]:
    """The full §2d + §2e report off one SASS page; returns (sched_ok, live_ok).

    Ends in exactly one SCHED-OK/SCHED-FAIL and one LIVE-OK/LIVE-FAIL machine
    line each (sass_tools' silence-is-failure rule); on failure raw samples are
    dumped so the next turn debugs evidence, not a summary.
    """
    res = _st.parse_sass(sass_text)
    instrs = list(res.instructions)
    out(f"[parse] {len(instrs)} instructions, unparsed={res.unparsed}, "
        f"noise={len(res.errors)}")
    if len(instrs) < _st.MIN_INSTRUCTIONS:
        _st.dump_raw(sass_text)
        out("SCHED-FAIL too-few-instructions")
        out("LIVE-FAIL too-few-instructions")
        return False, False

    # Which MMA ISA is this? Say it before transcribing, so a tcgen05 kernel is
    # never silently filed as "a Hopper kernel with no MMA in it".
    census = mma_isa_census(instrs)
    out(f"[isa] MMA census: wgmma={census.wgmma} tcgen05={census.tcgen05} "
        f"mma_sync={census.mma_sync}"
        + (f" ({', '.join(f'{k}x{v}' for k, v in census.fams.most_common())})"
           if census.fams else ""))
    mode = census.mode
    if mode == "tcgen05":
        # Accumulator in TMEM, handoffs on mbarrier: chains by TMEM operand,
        # UTCBAR commits and SYNCS waits are the schedule objects (no DEPBAR
        # depth exists to read). The HMMA side path rides in the same stream.
        out("[isa] tcgen05 grammar: chains by TMEM accumulator operand; the "
            "decisive read is which commit / mbarrier wait each elementwise run "
            "sits behind (handoff_before_each_ew).")
        return _tcgen05_report(instrs, census, out)
    if mode == "mma_sync":
        # Per-warp synchronous MMA: chains by accumulator base register, the
        # handoffs are DEPBAR.LE SB copy-group waits, named / cluster barriers
        # and mbarrier — no warpgroup chain, no depth immediate.
        out("[isa] mma.sync grammar: chains by accumulator base register; the "
            "decisive read is which DEPBAR.LE SB / barrier / mbarrier wait each "
            "elementwise run sits behind (handoff_before_each_ew).")
        return _mma_sync_report(instrs, census, out)

    lo, hi, how = find_loop(instrs)
    body = instrs[lo:hi + 1]
    out(f"[loop] {how}: {len(body)} instrs, "
        f"addr {hex(addr_int(instrs[lo]))}..{hex(addr_int(instrs[hi]))}")
    chains, owner = build_chains(body)
    out(f"[chains] {len(chains)} accumulator chains in body:")
    for c in chains:
        out(f"    {c.label()} count={c.count}")

    events = transcribe(body, owner)
    out(f"\n=== schedule transcription ({len(events)} events) ===")
    for n, (k, txt) in enumerate(events):
        out(f"  {n:3d} {txt}")
    pre = sched_verdict(events)
    hg_n = sum(1 for k, _ in events if k == "HG")
    dep_n = sum(1 for k, _ in events if k == "DEPBAR")
    ok_a = hg_n >= 4 and dep_n >= 1 and len(pre) >= 1
    out(f"\nSCHED-{'OK' if ok_a else 'FAIL'} loop={how} hgmma_runs={hg_n} "
        f"depbars={dep_n} ew_runs={len(pre)} depbar_before_each_ew={pre}")
    if not ok_a:
        out("[sched] raw samples for debugging:")
        for ins in body[:12]:
            out(f"    {ins.addr}  {ins.opcode} {ins.operands[:60]}")

    ok_b = _liveness_report(body, owner, out)
    return ok_a, ok_b


def _liveness_report(body, owner, out) -> bool:
    """The §2e half: register-liveness over the doubled body. ISA-independent —
    tcgen05 keeps its accumulator in TMEM rather than registers, which changes
    what the peak is made of, not how it is measured."""
    out("\n=== liveness over doubled body ===")
    prof, peak_i, peak_set, guessed = liveness(body)
    out(f"[live] peak={max(prof)} regs at body instr {peak_i} "
        f"({body[peak_i].opcode[:44]}), guessed-role instrs={guessed}/{len(body)}")
    cls = provenance(body, peak_i, peak_set, owner)
    if cls:
        out("[live] peak breakdown: " + ", ".join(
            f"{k}={v}" for k, v in sorted(cls.items(), key=lambda x: -x[1])))
    q = max(1, len(prof) // 8)
    for s in range(0, len(prof), q):
        seg = prof[s:s + q]
        out(f"[live] instrs {s:4d}-{min(s + q, len(prof)) - 1:4d}: "
            f"peak {max(seg):3d}  mean {sum(seg) // len(seg):3d}")
    frac = guessed / max(1, len(body))
    ok_b = 96 <= max(prof) <= 256 and frac < 0.25
    out(f"LIVE-{'OK' if ok_b else 'FAIL'} peak={max(prof)} guessed_frac={frac:.2f}")
    if not ok_b:
        out("[live] unknown-role samples:")
        shown = 0
        for ins in body:
            _, _, g = reg_accesses(ins)
            if g and shown < 10:
                out(f"    {ins.opcode} {ins.operands[:60]}")
                shown += 1
    return ok_b
