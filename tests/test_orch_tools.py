"""orch_tools on a synthetic two-role kernel with a known orchestration.

Every instruction form is verbatim from archived recon pages (_dsaref /
_fp8ref / student job logs), so the regex layer is tested against the
spellings the real pages use: SYNCS.EXCH init, ARRIVES.LDGSTSBAR ring arrive,
double-register mbarrier operands, BAR.SYNC's id-vs-count operand grammar,
LDGSTS's trailing global-source bracket.

The synthetic topology mirrors the FlashMLA shape the tool exists to catch:
a copy-producer loop feeding a ring (stride 0x900, 4 slots) plus a constant
mbarrier, an mma-consumer loop waiting on both, and a named-barrier
publish-handoff pair — cross-role object identity is the load-bearing claim
(the eyeball transcription this mechanises kept losing exactly that).
"""
import io
import re

from forgekernel.skills import orch_tools as OT
from forgekernel.skills.sass_tools import parse_sass

PROLOGUE = """
        /*0010*/  LDC R1, c[0x0][0x28] ;
        /*0020*/  S2UR UR7, SR_CTAID.X ;
        /*0030*/  UMOV UR5, 0x1900 ;
        /*0040*/  SYNCS.EXCH.64 URZ, [UR5+0x80], UR8 ;
        /*0050*/  IMAD.MOV.U32 R62, RZ, RZ, 0x1980 ;
        /*0060*/  MOV R40, 0x2000 ;
        /*0070*/  MOV R41, 0x2000 ;
        /*0080*/  SHFL.IDX PT, R15, R16, R17, 0x1f ;
"""

PRODUCER = """
        /*0100*/  LDG.E.CONSTANT R18, desc[UR6][R18.64] ;
        /*0110*/  LDG.E R23, desc[UR6][R30.64] ;
        /*0120*/  LDGSTS.E.BYPASS.LTC256B.128 desc[UR4][R16.64+0x80], P2, [R23+0x2000] ;
        /*0130*/  LDGSTS.E.BYPASS.LTC256B.128 desc[UR4][R16.64+0x100], P2, [R25+0x4000] ;
        /*0140*/  ARRIVES.LDGSTSBAR.64.ARVCNT [R40] ;
        /*0150*/  SYNCS.ARRIVE.TRANS64 RZ, [UR5+0x80], R8 ;
        /*0160*/  IADD3 R40, R40, 0x900, RZ ;
        /*0170*/  LOP3.LUT R40, R40, 0x23ff, RZ, 0xc0, !PT ;
        /*0180*/  BRA 0x100 ;
"""

CONSUMER = """
        /*0200*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], RZ, !UPT ;
        /*0210*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0220*/  WARPGROUP.ARRIVE ;
        /*0230*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0240*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R62], R58 ;
        /*0250*/  FFMA R12, R10, R13, R14 ;
        /*0260*/  FFMA R12, R10, R13, R14 ;
        /*0270*/  FFMA R12, R10, R13, R14 ;
        /*0280*/  FFMA R12, R10, R13, R14 ;
        /*0290*/  FFMA R12, R10, R13, R14 ;
        /*02a0*/  FFMA R12, R10, R13, R14 ;
        /*02b0*/  FFMA R12, R10, R13, R14 ;
        /*02c0*/  SHFL.BFLY PT, R15, R16, R17, 0x1f ;
        /*02d0*/  BAR.SYNC.DEFER_BLOCKING 0xc, 0x100 ;
        /*02e0*/  BAR.ARV 0xd, 0x80 ;
        /*02f0*/  BAR.SYNC.DEFER_BLOCKING R6, 0xa0 ;
        /*0300*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R41], R58 ;
        /*0310*/  IADD3 R41, R41, 0x900, RZ ;
        /*0320*/  LOP3.LUT R41, R41, 0x23ff, RZ, 0xc0, !PT ;
        /*0330*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0340*/  BRA 0x200 ;
"""

SASS_SMALL = PROLOGUE + PRODUCER + CONSUMER + """
        /*0350*/  EXIT ;
"""


def _big_page() -> str:
    pad = "".join(f"        /*{0x1000 + 16 * i:04x}*/  "
                  "FFMA R12, R10, R13, R14 ;\n" for i in range(520))
    return PROLOGUE + PRODUCER + CONSUMER + pad + "        /*9000*/  EXIT ;\n"


def _instrs(text=SASS_SMALL):
    return list(parse_sass(text).instructions)


def _regions(instrs):
    return OT.find_regions(instrs)


def test_two_role_partition():
    instrs = _instrs()
    loops = [r for r in _regions(instrs) if r.kind == "loop"]
    assert [r.label for r in loops] == ["copy-producer", "mma-consumer"]
    # the producer loop is NOT the largest backedge region — the single-loop
    # view (sched_live's find_loop) is exactly what this widens
    assert loops[0].hi - loops[0].lo < loops[1].hi - loops[1].lo


def test_const_object_identity_across_roles():
    """[UR5+0x80] (producer arrive), [R62] (consumer wait) and the EXCH init
    all resolve to smem+0x1980 — one object, three roles' sites."""
    instrs = _instrs()
    regions = _regions(instrs)
    sites = OT.mbarrier_sites(instrs, regions)
    by_fp = {}
    for site, mem, res in sites:
        by_fp.setdefault(res.fingerprint, []).append(site.kind)
    assert sorted(by_fp["smem+0x1980"]) == ["ARRIVE", "INIT", "WAIT"]


def test_ring_object_identity_across_roles():
    """Producer's ARRIVES on R40 and consumer's PHASECHK on R41 share
    (init base, stride, slots) — the per-half-block-vs-per-stage distinction
    the anonymised sched_live transcription cannot make."""
    instrs = _instrs()
    regions = _regions(instrs)
    sites = OT.mbarrier_sites(instrs, regions)
    ring = [(s, r) for s, _, r in sites if r.tier == "RING"]
    assert len(ring) == 2
    fps = {r.fingerprint for _, r in ring}
    assert fps == {"ring(base=0x2000,stride=0x900,slots=4)"}
    assert {s.kind for s, _ in ring} == {"CPASYNC_ARRIVE", "WAIT"}


def test_edges_join_producer_to_consumer():
    d = OT.build_dossier(_big_page())
    regions = d["regions"]
    labeled = {(regions[a].label, regions[w].label) for _, a, w in d["edges"]}
    assert ("copy-producer", "mma-consumer") in labeled


def test_named_barrier_grammar_and_verdicts():
    instrs = _instrs()
    d_sites = OT.named_barrier_sites(instrs, _regions(instrs))
    by_id = {bid: (count, kind) for _, bid, count, kind, _ in d_sites}
    assert by_id[0xc] == (0x100, "SYNC")
    assert by_id[0xd] == (0x80, "ARV")
    # register-id barrier: id honestly None, count still read (not mistaken
    # for the id — imms[0] there is the COUNT)
    assert by_id[None] == (0xa0, "SYNC")


def test_supply_order_tight_chain():
    instrs = _instrs()
    lines = OT.supply_order(instrs, _regions(instrs))
    assert len(lines) == 1
    # R23 fed by the LDG one instruction earlier: 1 of 2 copies tight-chained
    assert "tight-chained copies 1" in lines[0]
    assert "copies 2" in lines[0]


def test_issue_then_correct_nomination():
    instrs = _instrs()
    regions = _regions(instrs)
    cands = OT.issue_then_correct(instrs, regions,
                                  OT.mbarrier_sites(instrs, regions))
    assert len(cands) >= 1
    assert "smem+0x1980" in cands[0]


def test_cache_policy_verbatim():
    cache = OT.cache_policy(_instrs())
    assert cache["LDGSTS.E.BYPASS.LTC256B.128"] == 2


def test_commit_groups_depths():
    instrs = _instrs()
    lines = OT.commit_groups(instrs, _regions(instrs))
    assert any("LE 0x0 x1" in ln and "LE 0x1 x1" in ln for ln in lines)


def test_report_refuses_small_page():
    buf = io.StringIO()
    ok = OT.report(SASS_SMALL, out=lambda s: buf.write(s + "\n"))
    assert not ok
    assert "ORCH-FAIL too-few-instructions" in buf.getvalue()


ORCH_OK = re.compile(r"^ORCH-OK regions=\d+ loops=\d+ nbar_ids=\d+ "
                     r"mbar_objs=\d+ resolved=\d+ edges=\d+ "
                     r"itc_candidates=\d+", re.M)


def test_report_full_page_verdict_line():
    buf = io.StringIO()
    ok = OT.report(_big_page(), out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok, text
    m = ORCH_OK.search(text)
    assert m, text
    assert len(ORCH_OK.findall(text)) == 1   # exactly one verdict line
    assert '"verdict": "ORCH-OK"' in text     # and the summary carries it too
    # the summary line rides INSIDE the brackets so a banked dossier carries it
    assert text.index("ORCH-SUMMARY-JSON:") < text.index("ORCH DOSSIER END")
    assert "ring(base=0x2000,stride=0x900,slots=4)" in text
    assert "publish-handoff" not in text.split("ORCH-OK")[1]


NESTED = PROLOGUE + """
        /*00a0*/  IADD3 R50, R50, 0x1, RZ ;
""" + PRODUCER + CONSUMER + """
        /*0350*/  ISETP.GE.AND P3, PT, R50, 0x10, PT ;
        /*0360*/  BRA 0x0a0 ;
        /*0370*/  EXIT ;
"""


def test_innermost_partition_survives_outer_loop():
    """A persistent outer loop must not swallow the role structure — the real
    FlashMLA page came back as ONE 3378-instr loop under the
    old maximal-merge and the supply section starved."""
    instrs = _instrs(NESTED)
    regions = OT.find_regions(instrs)
    leaf_labels = [r.label for r in regions if r.kind == "loop"]
    assert "copy-producer" in leaf_labels
    assert "mma-consumer" in leaf_labels
    # producer and consumer are separate regions, not one merged span
    prod = next(r for r in regions if r.label == "copy-producer")
    cons = next(r for r in regions if r.label == "mma-consumer")
    assert prod.hi < cons.lo
    # ring identity still resolves across roles under the outer loop
    sites = OT.mbarrier_sites(instrs, regions)
    fps = {r.fingerprint for _, _, r in sites if r.tier == "RING"}
    assert "ring(base=0x2000,stride=0x900,slots=4)" in fps


REDERIVE = """
        /*0400*/  UMOV UR8, 0x400 ;
        /*0410*/  UIADD3 UR8, UR8, 0x34480, URZ ;
        /*0420*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8+0x108], R2 ;
        /*0430*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0440*/  BRA 0x400 ;
        /*0450*/  EXIT ;
"""


def test_rederivation_demotes_to_const():
    """A self-add whose init sits in the SAME loop is a per-iteration
    re-derivation, not a rotating ring — the ring(stride=0x34480) artifacts
    on the real FlashMLA page were exactly this, and demoting them to CONST
    is what lets the site group with its INIT-enumerated object."""
    instrs = _instrs(REDERIVE)
    regions = OT.find_regions(instrs)
    sites = OT.mbarrier_sites(instrs, regions)
    (_, _, res), = sites
    assert res.tier == "CONST"
    assert res.fingerprint == "smem+0x34988"        # 0x400+0x34480+0x108
    assert "[rederivation]" in res.detail


PAIRDEF = """
        /*0500*/  ULDC.64 UR16, c[0x0][0x118] ;
        /*0510*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0520*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR17+0x8], R3 ;
        /*0530*/  BRA 0x510 ;
        /*0540*/  EXIT ;
"""


def test_pair_def_resolves_high_half():
    """ULDC.64 UR16 defines the UR16/UR17 pair; chasing UR17 must find it
    (a real FlashMLA page's student side was 0/20 resolved for want of this) and the
    #hi marker keeps the two halves from merging into one object."""
    instrs = _instrs(PAIRDEF)
    regions = OT.find_regions(instrs)
    sites = OT.mbarrier_sites(instrs, regions)
    (_, _, res), = sites
    assert res.tier == "CONST"
    assert res.fingerprint == "cbank:c[0x0][0x118]#hi+0x8"


CONDADV = """
        /*0600*/  UMOV UR8, 0x400 ;
        /*0610*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0620*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8+0x118], R2 ;
        /*0630*/  @!P3 UIADD3 UR8, UR8, 0x34480, URZ ;
        /*0640*/  BRA 0x610 ;
        /*0650*/  EXIT ;
"""


def test_conditional_advance_stays_out_of_identity():
    """A guard-predicated self-add is a CONDITIONAL advance (phase flip,
    window move) — parse_sass drops the @!P guard, and reading the add as an
    every-iteration stride minted the ring(stride=0x34480) artifacts on the
    real FlashMLA page. Identity anchors to the init constant; the advance
    survives in the evidence trail."""
    instrs = _instrs(CONDADV)
    regions = OT.find_regions(instrs)
    sites = OT.mbarrier_sites(instrs, regions, OT.pred_addrs(CONDADV))
    (_, _, res), = sites
    assert res.tier == "CONST"
    assert res.fingerprint == "smem+0x518"          # 0x400 + 0x118
    assert "+cond-advance 0x34480" in res.detail


def test_post_site_add_does_not_shift_the_address():
    """A self-add AFTER the site hasn't applied when the site runs; with the
    in-loop re-init at the top, the address is init+offset every iteration.
    (Observed: including post-site adds shifted a barrier family by
    +0x34480.)"""
    sass = """
        /*0800*/  UMOV UR8, 0x400 ;
        /*0810*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8+0x108], R2 ;
        /*0820*/  UIADD3 UR8, UR8, 0x34480, URZ ;
        /*0830*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0840*/  BRA 0x800 ;
        /*0850*/  EXIT ;
"""
    instrs = _instrs(sass)
    regions = OT.find_regions(instrs)
    (_, _, res), = OT.mbarrier_sites(instrs, regions)
    assert res.tier == "CONST"
    assert res.fingerprint == "smem+0x508"          # 0x400 + 0x108, no shift


def test_prmt_spliced_address_is_never_a_const():
    """PRMT composes a remote CTA's dsmem address (peer rank spliced into the
    high bits) — following it as a move minted a fabricated CONST for the
    cross-CTA empty-arrive target on the real deep_gemm cluster page (job
    844076). The chain must end there, honestly, with the splice named."""
    sass = """
        /*0900*/  MOV R25, UR38 ;
        /*0910*/  IADD3 R25, R25, 0x34c98, RZ ;
        /*0920*/  PRMT R25, R0, 0x654, R25 ;
        /*0930*/  HGMMA.64x64x16.F32.BF16 R24, gdesc[UR8], R24, UP0 ;
        /*0940*/  SYNCS.ARRIVE.TRANS64 RZ, [R25+0x400], R8 ;
        /*0950*/  BRA 0x900 ;
        /*0960*/  EXIT ;
"""
    instrs = _instrs(sass)
    regions = OT.find_regions(instrs)
    (_, _, res), = OT.mbarrier_sites(instrs, regions)
    assert res.tier == "UNRESOLVED"
    assert "PRMT address splice" in res.detail


def test_spin_wait_micro_loop_label():
    spin = """
        /*0700*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R62], R58 ;
        /*0710*/  NANOSLEEP.SYNCS 0x989680 ;
        /*0720*/  BRA 0x700 ;
        /*0730*/  EXIT ;
"""
    regions = OT.find_regions(_instrs(spin))
    labels = [r.label for r in regions if r.kind == "loop"]
    assert labels == ["spin"]


def test_diff_identical_and_perturbed():
    big = _big_page()
    buf = io.StringIO()
    assert OT.diff_report(big, big, out=lambda s: buf.write(s + "\n")) == 0
    assert "ORCH-DIFF mismatches=0" in buf.getvalue()
    # student built one barrier per stage where the ref rings 4 slots, and
    # dropped the ARV half of the handoff pair: both must surface
    student = big.replace("LOP3.LUT R41, R41, 0x23ff, RZ, 0xc0, !PT ;",
                          "NOP ;").replace(
        "BAR.ARV 0xd, 0x80 ;", "NOP ;")
    buf2 = io.StringIO()
    n = OT.diff_report(big, student, out=lambda s: buf2.write(s + "\n"))
    assert n and n >= 2
    assert "ORCH-DIFF mismatches=" in buf2.getvalue()


# --------------------------------------------------------------------------- #
# Summary invariance: placement is not structure                              #
# --------------------------------------------------------------------------- #

def _relayout(blocks, start=0x100):
    """Re-address the given instruction blocks sequentially from ``start``,
    retargeting each block's back-edge BRA to its new top: the same loops,
    in another order, at other addresses."""
    out, addr = [], start
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        top = addr
        for ln in lines:
            body = ln.split("*/", 1)[1]
            if body.strip().startswith("BRA "):
                body = f"  BRA {hex(top)} ;"
            out.append(f"        /*{addr:04x}*/{body}")
            addr += 16
    return "\n".join(out) + "\n"


def _isomorphic_page() -> str:
    """_big_page() with every PLACEMENT changed and no STRUCTURE changed:
    consumer loop before the producer loop (region ids swap), the smem bases
    moved (0x1900/0x1980/0x2000 -> 0x7900/0x7980/0x8000), the named-barrier
    ids reassigned (0xc/0xd -> 0x3/0x4)."""
    prologue = (PROLOGUE.replace("UMOV UR5, 0x1900", "UMOV UR5, 0x7900")
                .replace("R62, RZ, RZ, 0x1980", "R62, RZ, RZ, 0x7980")
                .replace("MOV R40, 0x2000", "MOV R40, 0x8000")
                .replace("MOV R41, 0x2000", "MOV R41, 0x8000"))
    consumer = (CONSUMER.replace("BAR.SYNC.DEFER_BLOCKING 0xc, 0x100",
                                 "BAR.SYNC.DEFER_BLOCKING 0x3, 0x100")
                .replace("BAR.ARV 0xd, 0x80", "BAR.ARV 0x4, 0x80"))
    pad = "".join(f"        /*{0x1000 + 16 * i:04x}*/  "
                  "FFMA R12, R10, R13, R14 ;\n" for i in range(520))
    return (prologue + _relayout([consumer, PRODUCER]) + pad
            + "        /*9000*/  EXIT ;\n")


def test_summary_is_invariant_under_relocation_renumbering_and_region_order():
    """Regions renumbered, smem base moved, barrier ids reassigned: the human
    report changes on every one of those axes, the comparable summary on
    none — a `!` row must mean a structural difference, or the diagnosis
    routes on phantoms."""
    big, iso = _big_page(), _isomorphic_page()
    buf_big, buf_iso = io.StringIO(), io.StringIO()
    assert OT.report(big, out=lambda s: buf_big.write(s + "\n"))
    assert OT.report(iso, out=lambda s: buf_iso.write(s + "\n"))
    # the perturbation is real: placement differs everywhere on the page
    assert "ring(base=0x2000," in buf_big.getvalue()
    assert "ring(base=0x8000," in buf_iso.getvalue()
    assert "id 0xc:" in buf_big.getvalue() and "id 0x3:" in buf_iso.getvalue()
    assert "r2(mma-consumer): DEPBAR" in buf_big.getvalue()
    assert "r1(mma-consumer): DEPBAR" in buf_iso.getvalue()
    # and the structure diffs clean
    buf = io.StringIO()
    assert OT.diff_report(big, iso, out=lambda s: buf.write(s + "\n")) == 0, \
        buf.getvalue()
    s = OT.summarize(big)
    assert s["rings"] == ["stride=0x900,slots=4"]
    assert s["commit"] == ["mma-consumer: DEPBAR {LE 0x0 x1, LE 0x1 x1} LDGDEPBAR x0"]
    assert all(not line.startswith("r") for line in s["supply"])
    assert isinstance(s["nbar_verdicts"], list) and len(s["nbar_verdicts"]) == 3
    assert "verdict" not in OT._DIFF_KEYS


def test_structural_change_still_diffs_after_normalisation():
    """Normalisation must not flatten real differences: one barrier per
    stage instead of a 4-slot ring, and a dropped ARV half, both surface."""
    big = _big_page()
    student = big.replace("LOP3.LUT R41, R41, 0x23ff, RZ, 0xc0, !PT ;",
                          "NOP ;").replace("BAR.ARV 0xd, 0x80 ;", "NOP ;")
    buf = io.StringIO()
    n = OT.diff_report(big, student, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert n and n >= 2
    assert "! rings:" in text and "! nbar_verdicts:" in text


# --------------------------------------------------------------------------- #
# One launch per dossier                                                      #
# --------------------------------------------------------------------------- #

def test_multi_launch_page_reduces_to_one_launch():
    """`ncu --import --page source` prints every captured launch; a page with
    N copies of the kernel must read as ONE kernel, not one N times as long."""
    big = _big_page()
    doubled = big + big
    assert len(OT.split_launches(doubled)) == 2
    assert len(OT.split_launches(big)) == 1
    assert OT.summarize(doubled) == OT.summarize(big)
    buf = io.StringIO()
    assert OT.report(doubled, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert "page carried 2 launches; using launch 0" in text
    assert "[parse] 559 instructions" in text       # not 1118
    # an explicit launch index is honoured; a bad one falls back to the largest
    buf2 = io.StringIO()
    OT.report(doubled, out=lambda s: buf2.write(s + "\n"), launch=1)
    assert "using launch 1" in buf2.getvalue()
    bystander_first = SASS_SMALL + big          # RNG-sized kernel, then ours
    chunk, n, used = OT.select_launch(bystander_first, launch=7)
    assert n == 2 and used == 1 and chunk == OT.split_launches(bystander_first)[1]


# --------------------------------------------------------------------------- #
# The FAIL verdict travels with the summary                                   #
# --------------------------------------------------------------------------- #

def test_fail_verdict_rides_in_the_summary():
    """A page with loops but no sync sites is the 'blueprint declared,
    nothing built' reading: report() returns False and prints ORCH-FAIL, but
    the summary line is still there and names the verdict, so a consumer
    keyed on the summary keeps the (damning) comparison."""
    lines = []
    for ln in _big_page().splitlines():
        body = ln.split("*/", 1)[1] if "*/" in ln else ""
        if any(tok in body for tok in ("SYNCS.", "ARRIVES.", "BAR.")):
            ln = ln.split("*/", 1)[0] + "*/  NOP ;"
        lines.append(ln)
    page = "\n".join(lines) + "\n"
    buf = io.StringIO()
    ok = OT.report(page, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok is False
    assert text.rstrip().endswith("ORCH-FAIL no-sync-sites")
    assert not ORCH_OK.search(text)
    m = re.search(r"^ORCH-SUMMARY-JSON: (\{.*\})$", text, re.M)
    assert m, text
    import json
    summ = json.loads(m.group(1))
    assert summ["verdict"] == "ORCH-FAIL no-sync-sites"
    assert summ["mbar_objs"] == 0 and summ["nbar_verdicts"] == []
    assert text.index("ORCH-SUMMARY-JSON:") < text.index("ORCH DOSSIER END")


# --------------------------------------------------------------------------- #
# Warp-role attribution of scalar loops (grouped-gemm artefact)
# --------------------------------------------------------------------------- #

def _loop(start: int, body: list) -> str:
    """A back-edge loop at ``start`` whose body is ``body`` (opcode strings)."""
    lines = []
    addr = start
    for op in body:
        lines.append(f"        /*{addr:04x}*/  {op} ;")
        addr += 0x10
    lines.append(f"        /*{addr:04x}*/  BRA 0x{start:x} ;")
    return "\n".join(lines) + "\n"


_FFMA = "FFMA R12, R10, R13, R14"
_SCALAR_BODY = [_FFMA] * 9                        # > _SPIN_MAX, no GMMA / copy
_EPILOGUE_BODY = [_FFMA, "STSM.16.M88.4 [R40], R24", _FFMA,
                  "UTMASTG.2D [UR4, R8], [R40]", _FFMA, _FFMA, _FFMA, _FFMA, _FFMA]


def _labels(text):
    return {r.rid: (r.label, r.base, r.why) for r in _regions(_instrs(text))
            if r.kind == "loop"}


def test_epilogue_store_loop_joins_the_consumer():
    page = (PROLOGUE + PRODUCER + CONSUMER + _loop(0x400, _EPILOGUE_BODY)
            + "        /*0500*/  EXIT ;\n")
    loops = _labels(page)
    store = [v for v in loops.values() if v[1] == "scalar"]
    assert store == [("mma-consumer", "scalar", "store")]


def test_exit_sections_own_their_scalar_loops():
    # producer section: scheduling loop + producer loop, then EXIT;
    # consumer section: consumer loop + bookkeeping loop, then EXIT.
    page = (PROLOGUE + _loop(0x0a0, _SCALAR_BODY[:5]) + PRODUCER
            + "        /*0190*/  EXIT ;\n" + CONSUMER
            + _loop(0x400, _SCALAR_BODY) + "        /*0500*/  EXIT ;\n")
    loops = _labels(page)
    relabelled = sorted(v for v in loops.values() if v[1] == "scalar")
    assert relabelled == [("copy-producer", "scalar", "section"),
                          ("mma-consumer", "scalar", "section")]


def test_neighbours_rule_and_a_real_third_role():
    # One section (no EXIT structure): producer, scalar A, consumer, scalar B,
    # consumer. A sits between two roles -> stays scalar (a real third role
    # looks exactly like this); B sits between two consumers -> consumer.
    page = (PROLOGUE + PRODUCER + _loop(0x1a0, _SCALAR_BODY) + CONSUMER
            + _loop(0x400, _SCALAR_BODY)
            + CONSUMER.replace("/*02", "/*06").replace("/*03", "/*07")
                      .replace("BRA 0x200", "BRA 0x600")
            + "        /*0800*/  EXIT ;\n")
    loops = _labels(page)
    scalars = [(rid, v) for rid, v in loops.items() if v[0] == "scalar" or v[1]]
    kinds = sorted(v for _, v in scalars)
    assert kinds == [("mma-consumer", "scalar", "neighbours"), ("scalar", "", "")]


def test_relabel_is_printed_in_the_region_listing():
    page = (PROLOGUE + PRODUCER + CONSUMER + _loop(0x400, _EPILOGUE_BODY)
            + "        /*0500*/  EXIT ;\n")
    buf = io.StringIO()
    OT.report(_big_page().replace("        /*9000*/  EXIT ;\n",
                                  _loop(0x9100, _EPILOGUE_BODY)
                                  + "        /*9300*/  EXIT ;\n"),
              out=lambda s="": buf.write(s + "\n"))
    assert "(was scalar: store)" in buf.getvalue()


def _hdr(name: str) -> str:
    return (f"{name} (132, 1, 1)x(384, 1, 1), 2026-Sep-04 10:00:00, "
            f"Context 1, Stream 7, Device 0, CC 9.0\n")


def test_launch_headers_split_what_the_address_rule_fuses():
    """The page is not kernel-filtered: a bystander launch whose code sits at
    a LOWER address than the kernel fuses with the kernel's first launch on
    the address rule alone; ncu's per-launch header names the boundary."""
    big = _big_page()
    shifted = re.sub(r"/\*([0-9a-fA-F]+)\*/",
                     lambda m: "/*%04x*/" % (int(m.group(1), 16) + 0x10000), big)
    assert len(OT.split_launches(big + shifted)) == 1              # the rule's limit
    named = OT.split_launches_named(_hdr("rng_kernel") + SASS_SMALL
                                    + _hdr("main_kernel") + big + _hdr("main_kernel") + shifted)
    assert [n for n, _ in named] == ["rng_kernel", "main_kernel", "main_kernel"]
    page = _hdr("rng_kernel") + SASS_SMALL + _hdr("main_kernel") + big + _hdr("main_kernel") + shifted
    # the caller's index is honoured only among launches of the named kernel
    chunk, n, used = OT.select_launch(page, launch=0, kernel="main_kernel")
    assert n == 3 and used in (1, 2) and "rng" not in chunk.splitlines()[0]
    chunk, n, used = OT.select_launch(page, launch=None, kernel="rng")
    assert used == 0
    # no name known: the largest chunk, as before
    assert OT.select_launch(page)[2] in (1, 2)
    buf = io.StringIO()
    assert OT.report(page, out=lambda s: buf.write(s + "\n"), launch=0, kernel="main_kernel")
    assert "page carried 3 launches" in buf.getvalue()
