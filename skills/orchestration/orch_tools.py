"""Orchestration dossier over parsed SASS — the machine form of recon §2g.

§2g (sync topology & supply orchestration) is a mandatory recon sub-item with no
tool behind it: the teacher eyeballs thousands of SASS lines and transcribes the
barrier topology by hand. Two documented failure shapes motivated mechanising it:

* Hand transcription keeps the coarse fact and drops the load-bearing detail.
  A recon correctly counted "4 ARRIVES.LDGSTSBAR slots" yet the blueprint still
  carried one-mbarrier-per-stage; the reference's per-half-block barriers (the
  fine-grained release that pays for its seam overlap) never reached the
  milestone, and the run re-derived them through GPU verdict probes at ~6 min a
  hypothesis.
* ``sched_live_tools`` transcribes ONE instruction stream and anonymises sync
  events: every mbarrier op renders as ``mbarrier.arrive``/``mbarrier.trywait``
  with the address operand discarded, so one-barrier-per-stage and
  two-barriers-per-half-block are indistinguishable in its output — and the
  producer's loop (where supply organisation lives) is not the largest backedge
  region, so it never appears at all.

What this module adds, all static, all from the same SASS page §2 already
collects (zero extra GPU cost):

1. **Region/role partition** — every maximal backedge loop, not just the
   largest, classified by instruction mix (copy-producer / mma-consumer /
   scalar). Warp-specialised kernels lay their roles out as disjoint loops.
2. **Barrier identity** — named barriers by immediate id; mbarrier objects by
   resolving the address operand's def-chain (const base / ring induction with
   stride+wrap / unresolved — tiered, never guessed: an UNRESOLVED label is a
   deliverable, a fabricated address is not).
3. **Edges** — arrive sites and wait sites joined per barrier object across
   roles: the who-signals-whom table §2g asks the teacher to derive by hand.
4. The remaining §2g bullets as machine sections: merge placement, the
   issue-then-correct fingerprint (nominated, never convicted — account-level
   evidence only earns a nomination; conviction takes an intervention probe),
   supply ordering in producer loops (clustered prefetch vs interleaved, plus
   the tight-chain count that exposed a serial gather), and the cache
   policy bits verbatim.

Fingerprints are canonical IDENTITIES, not verified physical addresses: the
def-chain tracer follows LEA/SHF without evaluating their shifted-add
arithmetic, so absolute smem values on chains through them are approximate
(consistent per chain — grouping holds), and one physical barrier array
reached through two roles' distinct derivation chains can appear as two
object families (observed on FlashMLA's ping-pong pages:
the second consumer WG re-derives the array through its own register with an
extra in-loop +stride, landing a parallel ``smem+0x68exx`` family beside the
INIT-enumerated ``smem+0x349xx`` one). Audit absolutes against the INIT
enumeration — INIT sites sit in the prologue on constant chains and are the
trustworthy roster.

``report`` ends in exactly one ``ORCH-OK``/``ORCH-FAIL`` machine line
(sass_tools' silence-is-failure rule). ``diff_report`` compares two dossiers —
reference vs student — and is the construction-acceptance half: the blueprint
says two barriers per half-block, the student's compiled SASS answers whether
they exist, independent of end-to-end perf.

Stdlib only: this file is seeded verbatim into the recon tree's ``notes/`` and
imported by probes with a bare ``import orch_tools``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, NamedTuple, Optional, Tuple

try:  # seeded copy sits next to sass_tools in notes/
    import sass_tools as _st
except ImportError:  # in-package import (tests, tooling)
    from forgekernel.skills import sass_tools as _st

_HEX = re.compile(r"-?0x[0-9a-fA-F]+")
# Observed operand forms (verbatim from _dsaref/_fp8ref recon pages):
#   [R62]  [UR5+0x80]  [R12.64+0x100]  [R3+URZ+0x345b8]  [R15+URZ+0x34c80]
_MEM = re.compile(r"\[(U?R[0-9Z]+)(?:\.\w+)?(?:\+(U?R[0-9Z]+))?"
                  r"(?:\+(-?0x[0-9a-fA-F]+))?\]")
_REG = re.compile(r"\b(U?R)(\d+)\b")


def _mem_parts(match) -> Tuple[Optional[Tuple[str, int]], int]:
    """A _MEM match -> (base register or None, immediate offset).

    RZ/URZ contribute nothing; two live registers keep the first as base (the
    second is an index — the resolution honestly lands UNRESOLVED unless the
    base alone settles it)."""
    base = None
    for g in (match.group(1), match.group(2)):
        if not g or g.endswith("Z"):
            continue
        m = _REG.fullmatch(g)
        if m and base is None:
            base = (m.group(1), int(m.group(2)))
    off = int(match.group(3), 16) if match.group(3) else 0
    return base, off

MAX_CHASE = 12          # def-chain steps before honesty beats depth
_ITC_WINDOW = 24        # instructions after a wait scanned for a fix-up cluster
_ITC_FLOOR = 6          # fp-op count that qualifies as a correction cluster

_BRANCH = {"BRA", "BRX"}
# Families that define no general/uniform register worth chasing through.
_NO_DEF = {"BRA", "BRX", "JMP", "EXIT", "RET", "NOP", "CALL", "YIELD",
           "BAR", "BSSY", "BSYNC", "MEMBAR", "FENCE", "ERRBAR",
           "NANOSLEEP", "DEPBAR", "LDGDEPBAR", "WARPGROUP",
           "STS", "STG", "ST", "STL", "STSM", "RED", "ATOM", "ATOMG",
           "LDGSTS", "ARRIVES", "UTMALDG", "UTMASTG", "UTMACCTL",
           "SYNCS", "USYNCS", "CCTL", "EXIT"}
_FP_FAMS = {"FFMA", "FMUL", "FADD", "FSEL", "FMNMX", "HFMA2", "HMUL2", "HADD2",
            "MUFU"}
_COPY_FAMS = {"LDGSTS", "UTMALDG"}
_CACHE_FAMS = {"LDGSTS", "UTMALDG", "UTMASTG", "CREATEPOLICY", "UBLKCP",
               "UBLKRED"}
# Families that mark a loop as doing real work; a tiny loop with none of them
# is a spin-wait (PHASECHK/BRA retry, NANOSLEEP backoff, drain loops).
_WORK_FAMS = {"HGMMA", "QGMMA", "IGMMA", "BGMMA", "LDGSTS", "UTMALDG",
              "UTMASTG", "STSM", "LDSM", "STS", "STG", "LDG", "FFMA",
              "HFMA2", "MUFU"}
_SPIN_MAX = 8

# Line shapes, duplicated from sass_tools (vendored-standalone rule) for the
# predication pre-scan: parse_sass drops the @!P guard, but a guarded
# pointer-advance is CONDITIONAL — reading it as a per-iteration stride is
# how the first real-page run minted ring(stride=0x34480) artifacts.
_LINE_CUOBJDUMP = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*)$")
_LINE_NCU = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+(\S.*)$")


def pred_addrs(sass_text: str) -> frozenset:
    """Addresses of guard-predicated instructions (@P / @!P / @UP)."""
    out = set()
    for line in sass_text.splitlines():
        m = _LINE_CUOBJDUMP.match(line) or _LINE_NCU.match(line)
        if not m:
            continue
        body = re.sub(r"/\*.*?\*/", "", m.group(2)).strip()
        if body.startswith("@"):
            out.add(int(m.group(1), 16))
    return frozenset(out)


def fam_of(op: str) -> str:
    return op.split(".", 1)[0]


def addr_int(ins) -> int:
    return int(ins.addr or "0", 16)


# --------------------------------------------------------------------------- #
# 1. Region / role partition
# --------------------------------------------------------------------------- #

class Region(NamedTuple):
    rid: int
    lo: int             # instruction index, inclusive
    hi: int             # inclusive
    kind: str           # "loop" | "straight"
    label: str          # role for loops (after _attach_scalar_loops), "-" for straight
    mix: Counter        # family -> count
    base: str = ""      # the mix-only label when a relabel changed it, else ""
    why: str = ""       # the relabel signal: store | section | neighbours


_STORE_FAMS = {"UTMASTG", "STSM"}
_ROLE_LOOPS = ("mma-consumer", "copy-producer")


def _role_label(mix: Counter) -> str:
    gmma = sum(v for k, v in mix.items() if k.endswith("GMMA"))
    copy = sum(mix.get(f, 0) for f in _COPY_FAMS)
    if gmma:
        return "mma-consumer"
    if copy:
        return "copy-producer"
    return "scalar"


def find_regions(instrs) -> List[Region]:
    """Innermost-loop partition: every instruction goes to the SMALLEST
    backedge span containing it; consecutive same-owner indices form a region.

    Merging overlapping spans into one maximal region — the first cut of this
    function — read a whole warp-specialised kernel as ONE 3378-instruction
    loop on the real FlashMLA page: the persistent outer loop
    contains the producer's and the consumers' inner loops as disjoint
    branches, and the merge swallowed the role structure the dossier exists to
    expose (no copy-producer loop, no cross-role edges, empty supply section).
    Innermost attribution gives each inner loop its own region; the outer
    loop's own code (scheduling, epilogue-per-tile) appears as the chunks
    between them, labelled from their own mix.
    """
    addr_ix = {addr_int(ins): i for i, ins in enumerate(instrs)}
    spans: List[Tuple[int, int]] = []
    for i, ins in enumerate(instrs):
        if fam_of(ins.opcode) not in _BRANCH:
            continue
        hexes = _HEX.findall(ins.operands)
        if not hexes:
            continue
        j = addr_ix.get(int(hexes[-1], 16))
        if j is not None and j < i:
            spans.append((j, i))
    spans = sorted(set(spans), key=lambda s: (s[0], -(s[1])))

    def _owner(i: int) -> Optional[Tuple[int, int]]:
        best = None
        for lo, hi in spans:
            if lo > i:
                break
            if lo <= i <= hi and (best is None
                                  or hi - lo < best[1] - best[0]):
                best = (lo, hi)
        return best

    regions: List[Region] = []

    def _push(lo: int, hi: int, kind: str) -> None:
        if hi < lo:
            return
        mix = Counter(fam_of(ins.opcode) for ins in instrs[lo:hi + 1])
        if kind != "loop":
            label = "-"
        elif (hi - lo + 1) <= _SPIN_MAX and not (set(mix) & _WORK_FAMS):
            label = "spin"       # PHASECHK/BRA retry, NANOSLEEP backoff, drain
        else:
            label = _role_label(mix)
        regions.append(Region(len(regions), lo, hi, kind, label, mix))

    cur_owner = _owner(0) if instrs else None
    start = 0
    for i in range(1, len(instrs)):
        o = _owner(i)
        if o != cur_owner:
            _push(start, i - 1, "loop" if cur_owner else "straight")
            cur_owner, start = o, i
    if instrs:
        _push(start, len(instrs) - 1, "loop" if cur_owner else "straight")
    return _attach_scalar_loops(instrs, regions)


def _attach_scalar_loops(instrs, regions: List[Region]) -> List[Region]:
    """Warp-role attribution for the loops the instruction mix cannot name.

    ``_role_label`` names a loop by what it issues: GMMA -> mma-consumer, a
    global->smem copy -> copy-producer, anything else -> scalar. In a
    warp-specialised kernel most of that "anything else" is NOT a third role:
    it is the math warps' epilogue (the STSM/UTMASTG store path), their
    per-tile bookkeeping between two mainloops, or the producer's scheduling
    — code the same warps run before or after their labelled loop. Measured
    on an offline pricing of grouped gemm (deep_gemm ref vs student
    best): the ref's epilogue loops (STSM=24, UTMASTG=1) came out `scalar`
    while the student's identical epilogue sat inside a GMMA-carrying loop and
    came out `mma-consumer`, so the ref read "9 scalar loops / 11 mbarriers /
    10.8% stall share" against the student's 4 / 0 / 3.9% — a labelling
    artefact the dossier diff raised as structural `!` rows and the diagnose
    step then routed on (a third worker role that does not exist).

    Three signals, in order, each a property of the code rather than a guess
    about intent. The mix-only label survives in ``base`` and the report
    prints every relabel with its reason, so a reader can undo it:
      1. store — the loop issues UTMASTG/STSM: the epilogue, math-warp code
         in every warp-specialised GEMM/attention design on this platform;
      2. section — EXIT splits the page into the role branches' code sections
         (producer warps exit early, consumers exit at the end); a section
         whose labelled loops are all ONE role owns its scalar loops;
      3. neighbours — with no section structure to go on, a scalar loop whose
         nearest labelled loop on BOTH sides is the same role belongs to it.
    A scalar loop none of these reach stays scalar: that is what a real third
    role looks like, and the tables then compare it as one.
    """
    exits = [i for i, ins in enumerate(instrs) if fam_of(ins.opcode) == "EXIT"]
    from bisect import bisect_left

    def section_of(r: Region) -> int:
        return bisect_left(exits, r.lo)

    labelled = [r for r in regions if r.kind == "loop" and r.label in _ROLE_LOOPS]
    by_section: dict = {}
    for r in labelled:
        by_section.setdefault(section_of(r), set()).add(r.label)
    out: List[Region] = []
    for r in regions:
        if r.kind != "loop" or r.label != "scalar":
            out.append(r)
            continue
        new, why = None, ""
        if set(r.mix) & _STORE_FAMS:
            new, why = "mma-consumer", "store"
        else:
            owners = by_section.get(section_of(r), set())
            if len(owners) == 1:
                new, why = next(iter(owners)), "section"
            else:
                prev = next((x.label for x in reversed(labelled) if x.hi < r.lo), None)
                nxt = next((x.label for x in labelled if x.lo > r.hi), None)
                if prev is not None and prev == nxt:
                    new, why = prev, "neighbours"
        if new is None:
            out.append(r)
        else:
            out.append(r._replace(label=new, base="scalar", why=why))
    return out


def region_of(regions: List[Region], idx: int) -> Region:
    for r in regions:
        if r.lo <= idx <= r.hi:
            return r
    return regions[-1]


# --------------------------------------------------------------------------- #
# 2. Barrier sites and address resolution
# --------------------------------------------------------------------------- #

class Site(NamedTuple):
    idx: int            # instruction index
    kind: str           # ARRIVE | WAIT | CPASYNC_ARRIVE
    trans: bool         # transaction-count arrive (TMA/cp.async byte counting)
    rid: int            # region id


class Resolution(NamedTuple):
    tier: str           # CONST | RING | THREAD | UNRESOLVED
    fingerprint: str    # grouping key — canonical, never fabricated
    detail: str         # human line: the def-chain evidence


def _defs_of(ins) -> set:
    """Registers written by this instruction, as (kind, num) pairs.

    A ``.64``/``.WIDE`` form defines a register PAIR (``ULDC.64 UR16, c[..]``
    writes UR16 AND UR17). Recording only the named half is how the student
    side of a FlashMLA comparison came back 0/20 resolved: every barrier base was UR17,
    whose def literally never appeared."""
    f = fam_of(ins.opcode)
    if f in _NO_DEF or f.endswith("GMMA"):
        return set()
    ops = [o.strip() for o in ins.operands.split(",") if o.strip()]
    if not ops:
        return set()
    m = _REG.search(ops[0])
    if not m:
        return set()
    out = {(m.group(1), int(m.group(2)))}
    if ".64" in ins.opcode or ".WIDE" in ins.opcode:
        out.add((m.group(1), int(m.group(2)) + 1))
    return out


def _srcs_of(ins) -> set:
    ops = [o.strip() for o in ins.operands.split(",") if o.strip()]
    out = set()
    for o in ops[1:]:
        for m in _REG.finditer(o):
            out.add((m.group(1), int(m.group(2))))
    return out


def _imms_of(ins) -> List[int]:
    return [int(h, 16) for h in _HEX.findall(ins.operands)]


def _scan_order(regions: List[Region],
                site_idx: int) -> List[Tuple[int, bool]]:
    """Backward walk from a site: cyclic within its loop region (steady state,
    same doubling idea as sched_live's liveness), then linear into the code
    before the region (the init half of an induction).

    Each entry carries a ``before_site`` flag: True while the walk has not
    wrapped past the loop head (the instruction executes BEFORE the site in
    one iteration's program order). A self-increment found after the wrap has
    not applied yet when the site runs — adding its stride into the site's
    address is how one page shifted a whole barrier family by +0x34480."""
    reg = region_of(regions, site_idx)
    size = reg.hi - reg.lo + 1
    order: List[Tuple[int, bool]] = []
    if reg.kind == "loop":
        for k in range(1, size):
            j = reg.lo + ((site_idx - reg.lo - k) % size)
            order.append((j, j < site_idx))
        order.extend((j, True) for j in range(reg.lo - 1, -1, -1))
    else:
        order.extend((j, True) for j in range(site_idx - 1, -1, -1))
    return order


def resolve_addr(instrs, regions: List[Region], site_idx: int,
                 base: Tuple[str, int], offset: int,
                 pred: frozenset = frozenset()) -> Resolution:
    """Chase the barrier-address register's def-chain to a canonical identity.

    Tiers, most to least resolved — and the tier is part of the deliverable:
    * CONST      — the chain bottoms out in immediates: absolute smem offset.
    * RING       — a self-increment inside a loop (rotating slot): stride, plus
                   wrap evidence when an AND-mask or SEL sits on the cycle.
    * THREAD     — the chain passes through S2R (tid/ctaid): per-thread address.
    * UNRESOLVED — chain too deep / unknown producer. The fingerprint degrades
                   to base-reg+offset within the region: syntactically identical
                   sites still group (single-reaching-def approximation), and
                   the label says so instead of pretending.

    The scan cursor is monotone: after a def is consumed the walk resumes
    BEHIND it, never restarts at the site — a restart re-finds the same
    self-increment forever and the ring's wrap mask and init constant (the
    slot-count evidence) are never reached.
    """
    reg = region_of(regions, site_idx)
    order = _scan_order(regions, site_idx)
    cursor = 0
    cur = base
    acc = offset
    trail: List[str] = []
    stride = mask = None
    wrap_sel = hi_half = False
    adv = 0                    # sum of guard-predicated self-adds: CONDITIONAL
    applied = 0                # unpredicated self-adds executed BEFORE the site
    before_site = True         # position flag of the def _find_def returned

    def _find_def() -> Optional[int]:
        nonlocal cursor, hi_half, before_site
        while cursor < len(order):
            j, pre = order[cursor]
            cursor += 1
            defs = _defs_of(instrs[j])
            if cur in defs:
                before_site = pre
                # was cur the pair's high half? (.64 def adds primary+1)
                ops0 = instrs[j].operands.split(",", 1)[0]
                m = _REG.search(ops0)
                if m and int(m.group(2)) + 1 == cur[1] \
                        and m.group(1) == cur[0]:
                    hi_half = True
                return j
        return None

    def _follow(srcs: set) -> bool:
        """Move `cur` to the next register to chase, deterministically:
        prefer cur itself (address chains accumulate in place), else the
        lowest-numbered source — set iteration order must not decide what a
        dossier says."""
        nonlocal cur
        cands = sorted(s for s in srcs if s[0] in ("R", "UR"))
        if not cands:
            return False
        cur = cur if cur in srcs else cands[0]
        return True

    def _ring(base_val: Optional[int]) -> Resolution:
        slots = ((mask + 1) // stride
                 if mask and stride and (mask + 1) % stride == 0 else None)
        parts = [f"stride=0x{stride:x}"]
        if base_val is not None:
            parts.insert(0, f"base=0x{base_val + acc:x}")
        if slots:
            parts.append(f"slots={slots}")
        elif mask:
            parts.append(f"mask=0x{mask:x}")
        elif wrap_sel:
            parts.append("wrap=sel")
        return Resolution("RING", f"ring({','.join(parts)})",
                          " <- ".join(trail) or "self-increment")

    for _ in range(MAX_CHASE):
        j = _find_def()
        if j is None:
            break
        ins = instrs[j]
        f = fam_of(ins.opcode)
        srcs = _srcs_of(ins)
        imms = _imms_of(ins)
        trail.append(f"{ins.opcode} {ins.operands[:40]}")
        if f in ("LDC", "ULDC") or (
                f in ("MOV", "UMOV", "IMAD", "UIMAD") and not srcs
                and "c[" in ins.operands):
            # LDC R1, c[0x0][0x28] / IMAD.MOV.U32 Rx, RZ, RZ, c[..][..]:
            # a kernel-parameter load — canonical as the cbank slot. The
            # #hi marker keeps the pair's two halves from merging into one
            # fabricated object (both halves fingerprint the same slot).
            cb = (ins.operands[ins.operands.index("c["):][:24]
                  if "c[" in ins.operands else ins.operands[:24]).strip()
            return Resolution("CONST",
                              f"cbank:{cb}{'#hi' if hi_half else ''}"
                              f"+0x{acc:x}",
                              " <- ".join(trail))
        if f in ("MOV", "UMOV", "IMAD", "UIMAD") and not srcs:
            # MOV Rx, 0x..  /  IMAD.MOV.U32 Rx, RZ, RZ, 0x..  (RZ carries no
            # digits, so the register regex reads it as no source at all)
            if imms:
                note = f" [+cond-advance 0x{adv:x}]" if adv else ""
                if stride is not None:
                    init_reg = region_of(regions, j)
                    if init_reg.kind == "loop":
                        # Init executes inside a loop: the self-add is a
                        # per-iteration re-derivation chain reusing one
                        # register, not a rotating ring. The address is a
                        # constant; classify it as one so the site groups
                        # with the INIT-enumerated object. (The
                        # ring(stride=0x34480) artifacts on the FlashMLA
                        # page were exactly this.) A per-tile ring whose
                        # init sits in an OUTER loop is demoted too —
                        # identity grouping outranks the slot count, and
                        # the trail keeps the evidence for a human call.
                        # Only the pre-site portion of the adds has applied
                        # at the site (the full-stride formula once
                        # shifted a whole barrier family by +0x34480).
                        return Resolution(
                            "CONST",
                            f"smem+0x{imms[-1] + applied + acc:x}",
                            " <- ".join(trail) + " [rederivation]" + note)
                    return _ring(imms[-1] + applied)
                if adv:
                    # Only conditional advances on the chain: identity
                    # anchors to the init constant; the advance is real but
                    # not an every-iteration stride, so it stays out of the
                    # fingerprint and in the evidence trail.
                    return Resolution("CONST",
                                      f"smem+0x{imms[-1] + acc:x}",
                                      " <- ".join(trail) + note)
                return Resolution("CONST", f"smem+0x{imms[-1] + acc:x}",
                                  " <- ".join(trail))
            break
        if f in ("S2R", "S2UR", "CS2R"):
            if "SR_TID" in ins.operands or "SR_CTAID" in ins.operands:
                return Resolution("THREAD",
                                  f"thread({ins.operands[:24]})+0x{acc:x}",
                                  " <- ".join(trail))
            break
        if f in ("IADD3", "UIADD3"):
            if cur in srcs and imms:
                if addr_int(ins) in pred:
                    adv += imms[0]               # guarded: conditional advance
                    continue
                j_reg = region_of(regions, j)
                if j_reg.kind == "loop":
                    # self-add in a loop: ring rotation OR re-derivation —
                    # the terminal init's region decides which (above).
                    # Chained adds sum (a re-derivation often splits its
                    # constant across several IADD3s). Only an add that
                    # executes BEFORE the site has applied to this
                    # iteration's address; a post-site add is next
                    # iteration's business.
                    stride = (stride or 0) + imms[0]
                    if before_site:
                        applied += imms[0]
                else:
                    acc += imms[0]               # one-time add: plain offset
                continue
            nxt = [s for s in srcs if s[0] in ("R", "UR")]
            if imms:
                acc += imms[0]
            if nxt:
                cur = sorted(nxt)[0]
                continue
            break
        if f in ("LOP3", "ULOP3"):
            if cur in srcs and imms:             # AND-mask on the cycle: wrap
                mask = imms[0]
                others = sorted(s for s in srcs if s != cur)
                if others:
                    cur = others[0]
                continue
            break
        if f in ("SEL", "USEL"):
            wrap_sel = True                      # conditional wrap (ring reset)
            others = sorted(s for s in srcs if s != cur)
            if others:
                cur = others[0]
            continue
        if f == "PRMT":
            # Byte-permute is NOT value-preserving: on cluster kernels it is
            # how a remote CTA's dsmem address is composed (the peer's rank
            # spliced into the high bits — on deep_gemm, following
            # it minted a fabricated CONST smem+0x400 for what is really the
            # cross-CTA empty-arrive target; its fingerprint — arrives only,
            # zero waits/inits this CTA — is exactly the remote pattern).
            # The chain ends here, honestly.
            return Resolution(
                "UNRESOLVED",
                f"{base[0]}{base[1]}+0x{offset:x}#r{reg.rid}",
                " <- ".join(trail) + " [PRMT address splice — likely "
                "remote-CTA/cluster target]")
        if f in ("LEA", "ULEA", "SHF", "USHF", "MOV", "UMOV", "R2UR",
                 "UR2R"):
            if _follow(srcs):
                continue
            break
        if f in ("IMAD", "UIMAD") and len(srcs) == 1:
            # IMAD.U32 R8, RZ, RZ, UR10 — a register move in multiply clothing
            cur = next(iter(srcs))
            continue
        break
    if stride is not None:
        return _ring(None)
    fp = f"{base[0]}{base[1]}+0x{offset:x}#r{reg.rid}"
    return Resolution("UNRESOLVED", fp,
                      " <- ".join(trail) or "no def found in region")


def mbarrier_sites(instrs, regions: List[Region],
                   pred: frozenset = frozenset()):
    """(site, mem-operand, resolution) for every mbarrier-addressed sync op.

    SYNCS.* carries the barrier in its bracketed operand; ARRIVES.LDGSTSBAR is
    the cp.async completion arrival (the producer side of a supply ring);
    SYNCS.EXCH is the barrier INIT — init sites usually sit together in the
    prologue with constant addresses, which makes them the enumeration anchor
    the loop-body sites group against."""
    out = []
    for i, ins in enumerate(instrs):
        f = fam_of(ins.opcode)
        kind = None
        if f in ("SYNCS", "USYNCS"):
            if "ARRIVE" in ins.opcode:
                kind = "ARRIVE"
            elif ("TRYWAIT" in ins.opcode or "PHASECHK" in ins.opcode
                  or "WAIT" in ins.opcode):
                kind = "WAIT"
            elif "EXCH" in ins.opcode:
                kind = "INIT"
        elif f == "ARRIVES":
            # Standalone cp.async-group arrival (ARRIVES.LDGSTSBAR.64 [addr]).
            # An ARRIVES suffix on LDGSTS itself is NOT handled here: its first
            # bracket is the smem destination, and resolving it as a barrier
            # would fabricate an object.
            kind = "CPASYNC_ARRIVE"
        if kind is None:
            continue
        m = _MEM.search(ins.operands)
        rid = region_of(regions, i).rid
        site = Site(i, kind, "TRANS" in ins.opcode, rid)
        base, offset = _mem_parts(m) if m else (None, 0)
        if m is None or base is None and offset == 0:
            out.append((site, m.group(0) if m else None,
                        Resolution("UNRESOLVED", f"no-mem-operand#r{rid}",
                                   ins.operands[:40])))
            continue
        if base is None:                         # [URZ+0x...]: literal offset
            out.append((site, m.group(0),
                        Resolution("CONST", f"smem+0x{offset:x}", "URZ base")))
            continue
        out.append((site, m.group(0),
                    resolve_addr(instrs, regions, i, base, offset, pred)))
    return out


def named_barrier_sites(instrs, regions: List[Region]):
    """(site-idx, id-or-None, count-or-None, ARV|SYNC, rid) per BAR.* site.

    Operand grammar (observed): first token is the barrier id — an immediate
    (``BAR.SYNC.DEFER_BLOCKING 0xc, 0x100``) or a register (``... R6, 0xa0``,
    id honestly None); the second token, when present, is the rendezvous
    thread count. Reading imms[0] as the id would swallow the register form's
    count as its id."""
    out = []
    for i, ins in enumerate(instrs):
        if fam_of(ins.opcode) != "BAR":
            continue
        ops = [o.strip() for o in ins.operands.split(",") if o.strip()]
        bid = count = None
        if ops:
            m0 = _HEX.match(ops[0])
            if m0:
                bid = int(m0.group(0), 16)
            if len(ops) > 1:
                m1 = _HEX.match(ops[1])   # match, not int(): probe-rendered
                if m1:                    # pages append markers ("0x100 <<<")
                    count = int(m1.group(0), 16)
        kind = "ARV" if ".ARV" in ins.opcode else "SYNC"
        out.append((i, bid, count, kind, region_of(regions, i).rid))
    return out


# --------------------------------------------------------------------------- #
# 3-4. The remaining §2g sections
# --------------------------------------------------------------------------- #

def merge_placement(instrs, regions: List[Region]) -> Dict[str, int]:
    """§2g merge-position: reduction fingerprints in-loop vs out-of-loop."""
    out = Counter()
    for i, ins in enumerate(instrs):
        f = fam_of(ins.opcode)
        in_loop = region_of(regions, i).kind == "loop"
        if f == "SHFL":
            out["shfl_in_loop" if in_loop else "shfl_out_of_loop"] += 1
        elif f == "STS" and "STSM" not in ins.opcode and in_loop:
            out["scalar_sts_in_loop"] += 1
    return dict(out)


def issue_then_correct(instrs, regions: List[Region], mb_sites) -> List[str]:
    """§2g issue-then-correct fingerprint — NOMINATIONS only, by design.

    Account-level evidence (an MMA issued before a wait, fp ops clustered
    after it) earns a candidate line; whether the cluster is a correction or
    ordinary epilogue math is a conviction, and convictions take an
    intervention probe. The line says which chain and which barrier so the
    probe knows what to flip."""
    waits = [(s.idx, res) for (s, _, res) in mb_sites if s.kind == "WAIT"]
    cands = []
    for wi, res in waits:
        reg = region_of(regions, wi)
        if reg.kind != "loop" or not any(k.endswith("GMMA")
                                         for k in reg.mix):
            continue
        gmma_before = next((j for j in range(wi - 1, reg.lo - 1, -1)
                            if fam_of(instrs[j].opcode).endswith("GMMA")), None)
        if gmma_before is None:
            continue
        window = instrs[wi + 1: min(wi + 1 + _ITC_WINDOW, reg.hi + 1)]
        fp_n = sum(1 for x in window if fam_of(x.opcode) in _FP_FAMS)
        if fp_n < _ITC_FLOOR:
            continue
        acc = _REG.search(instrs[gmma_before].operands)
        cands.append(
            f"candidate @{instrs[wi].addr}: chain "
            f"{acc.group(0) if acc else '?'} issued before WAIT "
            f"{res.fingerprint}; {fp_n} fp ops in the {_ITC_WINDOW}-instr "
            f"window after it")
    return cands


def supply_order(instrs, regions: List[Region]) -> List[str]:
    """§2g supply ordering, per copy-producer loop.

    Two spellings of the same disease this section exists to catch:
    * address-source loads scattered through the fill loop instead of hoisted
      to its top (latency exposed on the critical path), and
    * the tight chain — an LDGSTS whose address register was produced by a
      load at most 4 instructions earlier. A serial gather (index load ->
      address -> LDGSTS, line by line, long_sb 12.7x) is all tight chains; the
      reference's batch prefetch has none.
    """
    lines = []
    for reg in regions:
        if reg.kind != "loop" or reg.label != "copy-producer":
            continue
        body = instrs[reg.lo:reg.hi + 1]
        copies = [i for i, x in enumerate(body)
                  if fam_of(x.opcode) in _COPY_FAMS]
        loads = [i for i, x in enumerate(body)
                 if fam_of(x.opcode) in ("LDG", "LDS", "LDL", "LD")]
        if not copies:
            continue
        tight = 0
        for ci in copies:
            # The gather-latency address is the LAST memory operand: LDGSTS
            # spells "desc[UR4][R16.64+0x..], P2, [R23+0x..]" with the global
            # source trailing; the first bracket is the smem destination.
            ms = list(_MEM.finditer(body[ci].operands))
            if not ms:
                continue
            base, _ = _mem_parts(ms[-1])
            if base is None:
                continue
            lo_w = max(0, ci - 4)
            if any(fam_of(body[j].opcode) in ("LDG", "LDS", "LDL", "LD")
                   and base in _defs_of(body[j]) for j in range(lo_w, ci)):
                tight += 1
        if loads:
            q = max(1, len(body) // 4)
            top = sum(1 for i in loads if i < q)
            shape = ("clustered-at-top" if top >= 0.7 * len(loads)
                     else "interleaved")
            lines.append(
                f"r{reg.rid}: addr-source loads {len(loads)} "
                f"({top}/{len(loads)} in top quarter -> {shape}); "
                f"copies {len(copies)}; tight-chained copies {tight}")
        else:
            lines.append(f"r{reg.rid}: copies {len(copies)}, "
                         f"no in-loop address-source loads (hoisted) ; "
                         f"tight-chained copies {tight}")
    return lines


def cache_policy(instrs) -> Counter:
    """§2g cache-policy bits: the full dotted opcodes, verbatim, counted."""
    return Counter(i.opcode for i in instrs
                   if fam_of(i.opcode) in _CACHE_FAMS)


def commit_groups(instrs, regions: List[Region]) -> List[str]:
    """In-flight depth per loop: DEPBAR immediates and LDGDEPBAR counts."""
    lines = []
    for reg in regions:
        if reg.kind != "loop":
            continue
        depths = Counter()
        ldgdep = 0
        for ins in instrs[reg.lo:reg.hi + 1]:
            if fam_of(ins.opcode) == "WARPGROUP" and "DEPBAR" in ins.opcode:
                imms = _HEX.findall(ins.operands)
                depths[imms[-1] if imms else "?"] += 1
            elif fam_of(ins.opcode) == "LDGDEPBAR":
                ldgdep += 1
        if depths or ldgdep:
            d = ", ".join(f"LE {k} x{v}" for k, v in sorted(depths.items()))
            lines.append(f"r{reg.rid}({reg.label}): DEPBAR {{{d}}} "
                         f"LDGDEPBAR x{ldgdep}")
    return lines


# --------------------------------------------------------------------------- #
# 5. Dossier assembly, report, diff
# --------------------------------------------------------------------------- #

# The result header ncu prints before every launch on every page:
#   <kernel name> (grid)x(block), <date>, Context N, Stream N, Device N, CC X.Y
_LAUNCH_HEADER = re.compile(
    r"^\s*(?P<name>\S.*?)(?:\s+\(\d+,\s*\d+,\s*\d+\)x\(\d+,\s*\d+,\s*\d+\))?"
    r",\s*(?:[^,]*,\s*)?Context \d+, Stream \d+")


def split_launches_named(sass_text: str) -> List[Tuple[Optional[str], str]]:
    """The page's launches as (kernel name or None, text chunk).

    ``ncu --import --page source`` prints EVERY captured launch's SASS back to
    back (``--launch-count 8`` -> eight copies of the kernel); parse_sass is a
    flat stream, so an unsplit page reads as one kernel eight times as long —
    every count 8x, the seven earlier copies fused into a giant straight
    region. A launch boundary is ncu's result header (it names the kernel —
    the page is NOT kernel-filtered, so a bystander launch inside the profiler
    window is on it too) or, on a page without headers, where the instruction
    address stops increasing (each kernel's code starts over at its base; a
    bystander sitting BELOW the kernel would fuse with the kernel's first
    launch on that reading alone, which is why the header wins). A single-
    launch page comes back as one chunk."""
    chunks: List[List[str]] = [[]]
    names: List[Optional[str]] = [None]
    prev = None
    for line in sass_text.splitlines():
        h = _LAUNCH_HEADER.match(line)
        if h is not None and not (_LINE_CUOBJDUMP.match(line) or _LINE_NCU.match(line)):
            if chunks[-1]:
                chunks.append([])
                names.append(None)
            names[-1] = h.group("name").strip()
            prev = None
            chunks[-1].append(line)
            continue
        m = _LINE_CUOBJDUMP.match(line) or _LINE_NCU.match(line)
        if m:
            addr = int(m.group(1), 16)
            if prev is not None and addr <= prev and chunks[-1]:
                chunks.append([])
                names.append(None)
            prev = addr
        chunks[-1].append(line)
    return [(n, "\n".join(c)) for n, c in zip(names, chunks) if c]


def split_launches(sass_text: str) -> List[str]:
    """The page's launches, one text chunk each (see split_launches_named)."""
    return [t for _, t in split_launches_named(sass_text)]


def _instr_lines(chunk: str) -> int:
    return sum(1 for line in chunk.splitlines()
               if _LINE_CUOBJDUMP.match(line) or _LINE_NCU.match(line))


def select_launch(sass_text: str, launch: Optional[int] = None, kernel: str = ""
                  ) -> Tuple[str, int, int]:
    """(chunk text, launches on the page, index used).

    ``launch`` names the caller's pick (the report's dominant action index —
    the page prints launches in report order); ``kernel`` (a substring of the
    dominant action's name) restricts the pick to launches whose page header
    names that kernel, so a bystander launch never stands in for it. Out of
    range, None or not a candidate: the largest candidate wins (by instruction
    lines — a bystander kernel (RNG, preprocess) is never the longest one on a
    tensor-kernel page)."""
    named = split_launches_named(sass_text)
    chunks = [t for _, t in named]
    if len(chunks) <= 1:
        return sass_text, len(chunks), 0
    candidates = list(range(len(chunks)))
    if kernel:
        by_name = [i for i, (n, _) in enumerate(named)
                   if n is not None and kernel.lower() in n.lower()]
        if by_name:
            candidates = by_name
    if launch is None or launch not in candidates:
        launch = max(candidates, key=lambda i: _instr_lines(chunks[i]))
    return chunks[launch], len(chunks), launch


def build_dossier(sass_text: str, launch: Optional[int] = None, kernel: str = ""
                  ) -> Optional[dict]:
    sass_text, n_launches, used = select_launch(sass_text, launch, kernel)
    res = _st.parse_sass(sass_text)
    instrs = list(res.instructions)
    if len(instrs) < _st.MIN_INSTRUCTIONS:
        return None
    regions = find_regions(instrs)
    mb = mbarrier_sites(instrs, regions, pred_addrs(sass_text))
    nb = named_barrier_sites(instrs, regions)

    objects: Dict[str, dict] = {}
    for site, mem, r in mb:
        o = objects.setdefault(r.fingerprint, {
            "tier": r.tier, "detail": r.detail, "sites": []})
        o["sites"].append(site)

    nbar: Dict[object, dict] = {}
    for i, bid, count, kind, rid in nb:
        b = nbar.setdefault(bid, {"ARV": [], "SYNC": [], "counts": set()})
        b[kind].append((i, rid))
        if count is not None:
            b["counts"].add(count)

    edges = []
    for fp, o in objects.items():
        arr = [s for s in o["sites"] if s.kind in ("ARRIVE", "CPASYNC_ARRIVE")]
        wai = [s for s in o["sites"] if s.kind == "WAIT"]
        if arr and wai:
            for a in {s.rid for s in arr}:
                for w in {s.rid for s in wai}:
                    edges.append((fp, a, w))
    return {
        "instrs": instrs, "parse": res, "regions": regions,
        "launches": (n_launches, used),
        "mb_sites": mb, "objects": objects, "nbar": nbar, "edges": edges,
        "merge": merge_placement(instrs, regions),
        "itc": issue_then_correct(instrs, regions, mb),
        "supply": supply_order(instrs, regions),
        "cache": cache_policy(instrs),
        "commit": commit_groups(instrs, regions),
    }


def _nbar_verdict(b: dict) -> str:
    if b["ARV"] and b["SYNC"]:
        return "publish-handoff (ARV one side, SYNC other -> serial chain)"
    if b["SYNC"]:
        return "symmetric rendezvous (all sides SYNC -> lockstep)"
    return "arrive-only"


def _region_line(instrs, r: Region) -> str:
    keys = ("HGMMA", "QGMMA", "IGMMA", "LDGSTS", "UTMALDG", "UTMASTG",
            "LDS", "STS", "LDSM", "STSM", "SHFL", "LDG", "STG")
    mixs = " ".join(f"{k}={r.mix[k]}" for k in keys if r.mix.get(k))
    tag = f"  (was {r.base}: {r.why})" if r.base else ""
    return (f"r{r.rid} {r.kind:8s} {r.label:13s} "
            f"[{instrs[r.lo].addr}..{instrs[r.hi].addr}] "
            f"{r.hi - r.lo + 1:5d} instrs  {mixs}{tag}")


def report(sass_text: str, out=print, launch: Optional[int] = None,
           kernel: str = "") -> bool:
    """The full §2g dossier off one SASS page; True iff ORCH-OK printed.

    Ends in exactly one ORCH-OK/ORCH-FAIL machine line; on failure raw samples
    are dumped so the next turn debugs evidence, not a summary. The verdict
    ALSO rides inside the ORCH-SUMMARY-JSON line (``verdict``): a dossier that
    fails for no-sync-sites is exactly the "blueprint declared, nothing built"
    reading the tool exists to deliver, and a consumer that keys on the
    trailing ORCH-OK line alone would drop it. ``launch`` / ``kernel`` pick
    the page's launch when it carries several (see select_launch)."""
    d = build_dossier(sass_text, launch, kernel)
    if d is None:
        chunk, n_launches, used = select_launch(sass_text, launch, kernel)
        res = _st.parse_sass(chunk)
        if n_launches > 1:
            out(f"[parse] page carried {n_launches} launches; using launch "
                f"{used}")
        out(f"[parse] {len(res.instructions)} instructions, "
            f"unparsed={res.unparsed}")
        _st.dump_raw(chunk)
        out("ORCH-FAIL too-few-instructions")
        return False
    instrs, regions = d["instrs"], d["regions"]
    out("=== ORCH DOSSIER BEGIN ===")
    n_launches, used = d["launches"]
    if n_launches > 1:
        out(f"[parse] page carried {n_launches} launches; using launch {used} "
            f"(one kernel, not {n_launches} copies fused into one stream)")
    out(f"[parse] {len(instrs)} instructions, unparsed={d['parse'].unparsed}")
    loops = [r for r in regions if r.kind == "loop"]
    spins = [r for r in loops if r.label == "spin"]
    out(f"\n=== regions ({len(regions)}, {len(loops)} loops, "
        f"{len(spins)} spin) ===")
    for r in regions:
        if r.label == "spin":
            continue         # aggregated below: 15 PHASECHK/BRA micro-loops
        out("  " + _region_line(instrs, r))     # would drown the role table
    if spins:
        rids = ",".join(f"r{r.rid}" for r in spins[:10])
        out(f"  spin-wait micro-loops x{len(spins)}: {rids}"
            + (",…" if len(spins) > 10 else ""))

    out(f"\n=== named barriers ({len(d['nbar'])} ids) ===")
    for bid, b in sorted(d["nbar"].items(),
                         key=lambda kv: (kv[0] is None, kv[0] or 0)):
        rid_a = sorted({r for _, r in b["ARV"]})
        rid_s = sorted({r for _, r in b["SYNC"]})
        width = ("" if not b["counts"] else
                 " width=" + "/".join(str(c) for c in sorted(b["counts"])))
        out(f"  id {hex(bid) if bid is not None else '<reg>'}: "
            f"ARV x{len(b['ARV'])} @r{rid_a} SYNC x{len(b['SYNC'])} @r{rid_s}"
            f"{width} -> {_nbar_verdict(b)}")
    sync_per_loop = Counter(
        rid for _, _, _, kind, rid in named_barrier_sites(instrs, regions)
        if kind == "SYNC")
    for r in loops:
        if sync_per_loop.get(r.rid):
            out(f"  blocking rendezvous in loop r{r.rid}: "
                f"{sync_per_loop[r.rid]} per body")

    objs = d["objects"]
    resolved = sum(1 for o in objs.values() if o["tier"] != "UNRESOLVED")
    out(f"\n=== mbarrier objects ({len(objs)}, {resolved} resolved) ===")
    for fp, o in sorted(objs.items()):
        by = Counter((s.kind, s.rid) for s in o["sites"])
        sites = " ".join(f"{k}@r{r} x{n}" for (k, r), n in sorted(by.items()))
        out(f"  [{o['tier']}] {fp}: {sites}")
        if o["tier"] != "UNRESOLVED":
            out(f"      via {o['detail'][:120]}")
    if d["edges"]:
        out("\n=== edges (arrive-role -> wait-role per object) ===")
        for fp, a, w in d["edges"]:
            la = regions[a].label if regions[a].kind == "loop" else "straight"
            lw = regions[w].label if regions[w].kind == "loop" else "straight"
            out(f"  {fp}: r{a}({la}) -> r{w}({lw})")

    if d["commit"]:
        out("\n=== commit groups (in-flight depth) ===")
        for line in d["commit"]:
            out("  " + line)
    out("\n=== merge placement ===")
    out("  " + (", ".join(f"{k}={v}" for k, v in sorted(d["merge"].items()))
                or "no reduction fingerprints"))
    out("\n=== issue-then-correct (nominations, not convictions) ===")
    for line in d["itc"] or ["  none nominated"]:
        out("  " + line if not line.startswith("  ") else line)
    out("\n=== supply order (copy-producer loops) ===")
    for line in d["supply"] or ["  no copy-producer loop found"]:
        out("  " + line if not line.startswith("  ") else line)
    out("\n=== cache policy bits (verbatim opcodes) ===")
    for op, n in d["cache"].most_common(12) or [("none", 0)]:
        out(f"  {op} x{n}")

    n_sync = len(d["mb_sites"]) + sum(
        len(b["ARV"]) + len(b["SYNC"]) for b in d["nbar"].values())
    verdict = ("ORCH-FAIL no-loop-region" if not loops else
               "ORCH-FAIL no-sync-sites" if n_sync == 0 else "ORCH-OK")
    # One machine-parseable line carrying the comparable core: the driver
    # banks the dossier from a job log and later diffs ref vs student WITHOUT
    # either SASS page in hand — this line is what makes that possible. The
    # verdict rides in it so a FAIL dossier is still a comparable (and
    # damning) reading rather than a dropped one.
    summary = _summary_from_dossier(d)
    summary["verdict"] = verdict
    out("ORCH-SUMMARY-JSON: " + json.dumps(summary, sort_keys=True))
    out("=== ORCH DOSSIER END ===")
    if verdict != "ORCH-OK":
        out("[orch] raw samples for debugging:")
        for ins in instrs[:12]:
            out(f"    {ins.addr}  {ins.opcode} {ins.operands[:60]}")
        out(verdict)
        return False
    out(f"\nORCH-OK regions={len(regions)} loops={len(loops)} "
        f"nbar_ids={len(d['nbar'])} mbar_objs={len(objs)} "
        f"resolved={resolved} edges={len(d['edges'])} "
        f"itc_candidates={len(d['itc'])}")
    return True


_RING_FP = re.compile(r"ring\((.*)\)")
_REGION_PREFIX = re.compile(r"^r\d+(?:\(([^)]*)\))?:\s*")


def _ring_key(fingerprint: str) -> str:
    """ring(base=0x2000,stride=0x900,slots=4) -> stride=0x900,slots=4.

    The base is where the ring sits, not what it is: the same ring assembled
    at another smem offset must compare equal. A raw AND-mask value is
    base-relative too, so an underivable slot count keeps only the wrap
    kind."""
    m = _RING_FP.search(fingerprint)
    parts = dict(p.split("=", 1) for p in m.group(1).split(",")) if m else {}
    key = [f"stride={parts['stride']}"] if "stride" in parts else []
    if "slots" in parts:
        key.append(f"slots={parts['slots']}")
    elif "mask" in parts:
        key.append("wrap=mask")
    elif "wrap" in parts:
        key.append(f"wrap={parts['wrap']}")
    return ",".join(key) or fingerprint


def _drop_region_id(line: str, keep_label: bool) -> str:
    """'r6(mma-consumer): DEPBAR ...' -> 'mma-consumer: DEPBAR ...';
    'r1: addr-source loads ...' -> 'addr-source loads ...'."""
    m = _REGION_PREFIX.match(line)
    if not m:
        return line
    label = m.group(1)
    rest = line[m.end():]
    return f"{label}: {rest}" if keep_label and label else rest


def _summary_from_dossier(d: dict) -> dict:
    """The comparable core of a built dossier. JSON-safe by construction —
    the summary travels through job logs as one ORCH-SUMMARY-JSON line and
    must survive the round trip unchanged.

    Every axis is keyed by STRUCTURE, never by placement: role labels and
    depth multisets, stride/slots, verdict/width multisets. Region ids, smem
    offsets and named-barrier ids are where a page happened to put things —
    an isomorphic kernel assembled elsewhere (regions renumbered, smem base
    moved, barrier ids reassigned) must diff clean, or the `!` rows a
    diagnosis routes on are phantoms."""
    regions = d["regions"]
    objs = d["objects"]
    # Counting axes are keyed on what the static analysis RESOLVED, never on
    # how many things codegen happened to emit: UNRESOLVED mbarrier objects
    # (a base the resolver could not derive — 49 vs 14 on one real pair with
    # identical contracts), spin micro-loops (132 vs 42, one PHASECHK/BRA
    # pair each) and per-loop load/copy counts all moved with the compiler,
    # not the structure, and stayed `!` after every real fix. They are kept
    # as reported fields (not in _DIFF_KEYS) so a reader still sees them.
    resolved = {fp: o for fp, o in objs.items() if o["tier"] != "UNRESOLVED"}
    per_role: Counter = Counter()
    for o in resolved.values():
        for rid in {s.rid for s in o["sites"]}:
            r = regions[rid]
            per_role[r.label if r.kind == "loop" else "straight"] += 1
    nbar_entries = []
    for b in d["nbar"].values():
        width = "/".join(str(c) for c in sorted(b["counts"]))
        nbar_entries.append(_nbar_verdict(b) + (f" width={width}" if width else ""))
    loops = [r for r in regions if r.kind == "loop"]
    return {
        "loops": dict(Counter(r.label for r in loops if r.label != "spin")),
        "spin_loops": sum(1 for r in loops if r.label == "spin"),
        "mbar_objs": len(resolved),
        "mbar_unresolved": len(objs) - len(resolved),
        "mbar_tiers": dict(Counter(o["tier"] for o in resolved.values())),
        "mbar_per_role": dict(per_role),
        "rings": sorted(_ring_key(fp) for fp, o in objs.items()
                        if o["tier"] == "RING"),
        "nbar_verdicts": sorted(nbar_entries),
        "nbar_widths": sorted(sorted(v["counts"]) for v in d["nbar"].values()
                              if v["counts"]),
        "edges": [list(e) for e in
                  sorted({(regions[a].label, regions[w].label)
                          for _, a, w in d["edges"]})],
        "commit": sorted(_drop_region_id(line, keep_label=True)
                         for line in d["commit"]),
        "merge": d["merge"],
        "itc_candidates": len(d["itc"]),
        "supply": sorted({_supply_class(line) for line in d["supply"]}),
        "supply_counts": sorted(_drop_region_id(line, keep_label=False)
                                for line in d["supply"]),
        "cache_ops": sorted(d["cache"]),
        # reported, not a contract axis (not in _DIFF_KEYS): the precision gate's
        # SASS cross-check reads it (loop/precision_gate.sass_precision_findings)
        "mma_precision": mma_precision(d["instrs"]),
    }


# --- MMA precision (the precision gate's SASS cross-check) ---------------------------
#
# What an MMA opcode says about its operand and accumulator types. HMMA / HGMMA spell
# ``<shape>.<ACC>[.<IN>]`` (a lone F16 / F32 = that accumulator over f16 inputs), the
# narrow-type families ``<shape>.<ACC>.<A>.<B>``. tcgen05 (UTC*MMA) carries its kind in
# the instruction descriptor, a register: only the family's class is static (UTCHMMA =
# kind::f16 / tf32, UTCQMMA = f8f6f4, UTCIMMA = i8, UTCOMMA = mxf4), the accumulator not.
_MMA_FAMILY = re.compile(r"^(HMMA|HGMMA|QGMMA|IGMMA|BGMMA|IMMA|DMMA|UTC[A-Z0-9]*MMA)$")
_MMA_TYPES = ("F64", "F32", "TF32", "BF16", "F16", "E4M3", "E5M2", "E2M1", "E3M2", "E2M3",
              "S32", "S8", "U8", "S4", "U4")
_MMA_FAMILY_CLASS = {"DMMA": "f64", "IMMA": "i8", "IGMMA": "i8", "QGMMA": "f8", "BGMMA": "b1",
                     "UTCHMMA": "f16|bf16|tf32", "UTCQMMA": "f8f6f4", "UTCIMMA": "i8",
                     "UTCOMMA": "mxf4"}


def mma_operand_types(opcode: str) -> "Optional[Tuple[str, str]]":
    """(input type, accumulator type) of an MMA opcode ('?' where the opcode does not
    say); None for a non-MMA opcode."""
    parts = opcode.split(".")
    fam = parts[0]
    if not _MMA_FAMILY.match(fam):
        return None
    types = [p for p in parts[1:] if p in _MMA_TYPES]
    if fam in ("HMMA", "HGMMA"):
        if not types:
            return "f16", "?"
        return (types[1] if len(types) > 1 else "F16").lower(), types[0].lower()
    if len(types) >= 2:
        return types[1].lower(), types[0].lower()
    return _MMA_FAMILY_CLASS.get(fam, fam.lower()), (types[0].lower() if types else "?")


def mma_precision(instrs) -> dict:
    """{"ops": {opcode-class: count}, "inputs": [...], "accums": [...]} over a kernel's
    MMA instructions; empty lists when it issues none."""
    ops: Counter = Counter()
    inputs, accums = set(), set()
    for ins in instrs:
        t = mma_operand_types(getattr(ins, "opcode", ""))
        if t is None:
            continue
        fam = ins.opcode.split(".")[0]
        ops[f"{fam}:{t[0]}/{t[1]}"] += 1
        inputs.add(t[0])
        accums.add(t[1])
    return {"ops": dict(ops), "inputs": sorted(inputs), "accums": sorted(accums)}


_SUPPLY_LOADS_RE = re.compile(r"addr-source loads (\d+) \(\d+/\d+ in top quarter -> ([a-z-]+)\)")
_SUPPLY_TIGHT_RE = re.compile(r"tight-chained copies (\d+)")


def _supply_class(line: str) -> str:
    """The structural class of one copy-producer loop's supply line, with
    every count dropped: WHERE the address-source loads sit (hoisted /
    clustered-at-top / interleaved) and WHETHER the copies chain on them
    (serial-gather vs prefetched). 'addr-source loads 20' vs '4' is codegen;
    'interleaved' vs 'hoisted' is the §2g contract."""
    m = _SUPPLY_LOADS_RE.search(line)
    loads = m.group(2) if m else ("hoisted" if "hoisted" in line else "?")
    t = _SUPPLY_TIGHT_RE.search(line)
    tight = int(t.group(1)) if t else 0
    return f"addr-source loads {loads}; {'serial-gather' if tight > 0 else 'prefetched'}"


def summarize(sass_text: str, launch: Optional[int] = None) -> Optional[dict]:
    """The comparable core of a dossier, for diffing ref vs student."""
    d = build_dossier(sass_text, launch)
    return None if d is None else _summary_from_dossier(d)


_DIFF_KEYS = ("loops", "mbar_objs", "mbar_tiers", "mbar_per_role", "rings",
              "nbar_verdicts", "nbar_widths", "edges", "commit", "merge",
              "itc_candidates", "supply", "cache_ops")


def diff_summaries(a: dict, b: dict, out=print,
                   names: "tuple[str, str]" = ("ref", "student")) -> int:
    """Two dossier summaries (``names``: ref vs student by default), axis by
    axis.

    Both sides are normalised through a JSON round trip so a summary that
    travelled through a job log as an ORCH-SUMMARY-JSON line compares equal
    to one built in memory (tuples become lists either way)."""
    a = json.loads(json.dumps(a))
    b = json.loads(json.dumps(b))
    mismatches = 0
    na, nb = names
    width = max(len(na), len(nb)) + 1
    out(f"=== dossier diff ({na} vs {nb}) ===")
    for k in _DIFF_KEYS:
        va, vb = a.get(k), b.get(k)
        if va == vb:
            out(f"  = {k}: {va}")
        else:
            mismatches += 1
            out(f"  ! {k}:")
            out(f"      {na + ':':<{width}} {va}")
            out(f"      {nb + ':':<{width}} {vb}")
    out(f"ORCH-DIFF mismatches={mismatches}")
    return mismatches


def diff_report(ref_sass: str, student_sass: str, out=print) -> Optional[int]:
    """Reference dossier vs student dossier, axis by axis, from SASS pages.

    This is the construction-acceptance read: the blueprint's [SYNC:]/[GATHER:]
    adjudications came from the reference's dossier; whether the student's
    compiled SASS actually carries them is answerable here, statically, before
    a single perf number. Returns the mismatch count (None if either side is
    unparsable), and prints an ORCH-DIFF machine line."""
    a, b = summarize(ref_sass), summarize(student_sass)
    if a is None or b is None:
        out("ORCH-DIFF-FAIL "
            + ("ref" if a is None else "student") + "-unparsable")
        return None
    return diff_summaries(a, b, out)
