"""sched_live_tools on a synthetic loop body with a known schedule and liveness.

The synthetic SASS is the miniature of the FA4-bwd shape the tool was proven on:
two zero-init SS chains, an elementwise run under a DEPBAR.LE 0x1,
a drain, an RS accumulate chain, the exchange-surface store and the handshakes.
Assertions pin the load-bearing behaviours: backedge detection, chain identity
(width / A-operand / init class), the event order, the per-elementwise DEPBAR
verdict, and the liveness peak's composition.
"""
import io

from forgekernel.skills import sched_live_tools as SL
from forgekernel.skills.sass_tools import parse_sass

SASS = """
        /*0100*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
        /*0110*/  HGMMA.64x128x16.F32 R88, gdesc[UR16], RZ, !UPT ;
        /*0120*/  HGMMA.64x128x16.F32 R88, gdesc[UR16], R88, UP0 ;
        /*0130*/  WARPGROUP.ARRIVE ;
        /*0140*/  HGMMA.64x128x16.F32 R24, gdesc[UR18], RZ, !UPT ;
        /*0150*/  WARPGROUP.ARRIVE ;
        /*0160*/  WARPGROUP.DEPBAR.LE gsb0, 0x1 ;
        /*0170*/  MUFU.EX2 R10, R11 ;
        /*0180*/  FFMA R12, R10, R13, R14 ;
        /*0190*/  FFMA R12, R10, R13, R14 ;
        /*01a0*/  FFMA R12, R10, R13, R14 ;
        /*01b0*/  SHFL.IDX PT, R15, R16, R17, 0x1f ;
        /*01c0*/  F2FP.PACK_AB R18, R12, R12 ;
        /*01d0*/  FFMA R12, R10, R13, R14 ;
        /*01e0*/  MUFU.EX2 R10, R11 ;
        /*01f0*/  WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
        /*0200*/  HGMMA.64x64x16.F32 R184, R88, gdesc[UR16].tnspB, R184, UP0 ;
        /*0210*/  WARPGROUP.ARRIVE ;
        /*0220*/  STSM.16.MT88.4 [R20], R18 ;
        /*0230*/  SYNCS.ARRIVE.TRANS64.ART0 RZ, [UR5+0x80], R58 ;
        /*0240*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R62], R58 ;
        /*0250*/  NANOSLEEP.SYNCS 0x989680 ;
        /*0260*/  BAR.SYNC.DEFER_BLOCKING 0x1 ;
        /*0270*/  UTMALDG.2D.TILED [UR4], [R30] ;
        /*0280*/  BRA 0x110 ;
        /*0290*/  EXIT ;
"""


def _body():
    instrs = list(parse_sass(SASS).instructions)
    lo, hi, how = SL.find_loop(instrs)
    assert how == "backedge"
    return instrs[lo:hi + 1]


def test_chain_identity_and_widths():
    chains, owner = SL.build_chains(_body())
    assert [c.acc_base for c in chains] == [88, 24, 184]
    assert [c.acc_width for c in chains] == [64, 64, 32]
    assert [c.zero_init for c in chains] == [True, True, False]
    assert [c.a_is_reg for c in chains] == [False, False, True]


def test_event_order_and_depbar_verdict():
    body = _body()
    chains, owner = SL.build_chains(body)
    events = SL.transcribe(body, owner)
    kinds = [k for k, _ in events]
    # HG(chain0) WG HG(chain1) WG DEPBAR EW DEPBAR HG(chain2) WG STSM SYNCS
    # SYNCS SLEEP BAR TMA
    assert kinds == ["HG", "WG", "HG", "WG", "DEPBAR", "EW", "DEPBAR", "HG",
                     "WG", "STSM", "SYNCS", "SYNCS", "SLEEP", "BAR", "TMA"]
    assert "DEPBAR.LE 0x1" in events[4][1]
    assert "DEPBAR.LE 0x0" in events[6][1]
    # the decisive read: the elementwise run starts under a depth-1 wait
    assert SL.sched_verdict(events) == ["0x1"]


def test_sched_verdict_reads_the_immediate_through_the_collapse_suffix():
    """transcribe() collapses adjacent identical events with an '  x<N>' suffix
    (adjacent DEPBARs are common when a short EW run between them falls under
    the >=8-op floor). The verdict must still read the immediate, not 'x2'."""
    events = [("DEPBAR", "** DEPBAR.LE 0x2 **  x2"),
              ("EW", "elementwise x12 [FFMA x8]"),
              ("DEPBAR", "** DEPBAR.LE ? **"),
              ("EW", "elementwise x9 [FFMA x9]")]
    assert SL.sched_verdict(events) == ["0x2", "?"]


def test_liveness_peak_is_the_coexistence_account():
    body = _body()
    chains, owner = SL.build_chains(body)
    prof, peak_i, peak_set, guessed = SL.liveness(body)
    # chain0's 64 accumulator words + chain2's 32 dominate the peak
    assert max(prof) >= 96
    cls = SL.provenance(body, peak_i, peak_set, owner)
    assert cls["acc:chain0"] == 64
    assert cls["acc:chain2"] == 32


def test_report_verdict_lines():
    buf = io.StringIO()
    # the synthetic body is far below MIN_INSTRUCTIONS, so report must refuse
    ok_a, ok_b = SL.report(SASS, out=lambda s: buf.write(s + "\n"))
    assert (ok_a, ok_b) == (False, False)
    text = buf.getvalue()
    assert "SCHED-FAIL too-few-instructions" in text
    assert "LIVE-FAIL too-few-instructions" in text


def test_hgmma_accumulate_reads_its_own_range():
    body = _body()
    accum = next(i for i in body if "R184" in i.operands)
    defs, uses, guessed = SL.reg_accesses(accum)
    assert set(range(184, 216)) <= defs
    assert set(range(184, 216)) <= uses          # accumulate reads C
    assert set(range(88, 92)) <= uses            # RS A-operand, M*K/256 = 4 regs
    assert not guessed


# --- Blackwell tcgen05: the instrument must say it cannot see this ISA ----------
#
# This module models Hopper's warpgroup MMA (chains by accumulator base register,
# WARPGROUP.DEPBAR.LE wait depths). tcgen05 keeps the accumulator in TMEM and waits
# on mbarrier, so neither read exists. Before the SCHED-NOTAPPLICABLE verdict, a
# UTCHMMA mainloop came back as `SCHED-FAIL ... hgmma_runs=0 depbars=0` with every
# MMA filed as elementwise work — indistinguishable from "this kernel has a bad
# schedule", which can burn a run's whole recon budget.

TCGEN05_SASS = "\n".join(
    ["        /*0100*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"]
    + [f"        /*{0x110 + 16 * i:04x}*/  UTCHMMA.16832.F32.BF16 [UR8], [UR10], "
       f"gdesc[UR12], 0x1 ;" if i % 64 == 0 else
       f"        /*{0x110 + 16 * i:04x}*/  FFMA R{10 + i % 8}, R11, R12, R13 ;"
       for i in range(600)]
    + ["        /*2800*/  BRA 0x110 ;", "        /*2810*/  EXIT ;"])


HMMA_SASS = "\n".join(
    ["        /*0100*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"]
    + [f"        /*{0x110 + 16 * i:04x}*/  HMMA.16816.F32.BF16 R{8 + (i % 4) * 4}, R24, "
       f"R28, R{8 + (i % 4) * 4} ;" if i % 8 == 0 else
       f"        /*{0x110 + 16 * i:04x}*/  FFMA R{10 + i % 8}, R11, R12, R13 ;"
       for i in range(600)]
    + ["        /*2800*/  BRA 0x110 ;", "        /*2810*/  EXIT ;"])

NO_MMA_SASS = "\n".join(
    ["        /*0100*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;"]
    + [f"        /*{0x110 + 16 * i:04x}*/  FFMA R{10 + i % 8}, R11, R12, R13 ;"
       for i in range(600)]
    + ["        /*2800*/  BRA 0x110 ;", "        /*2810*/  EXIT ;"])


def test_mma_isa_census_separates_the_three_families():
    hopper = list(parse_sass(SASS).instructions)
    c = SL.mma_isa_census(hopper)
    assert (c.wgmma, c.tcgen05, c.mma_sync) == (4, 0, 0)
    assert c.fams["HGMMA"] == 4 and c.not_applicable is None

    black = list(parse_sass(TCGEN05_SASS).instructions)
    c = SL.mma_isa_census(black)
    assert c.wgmma == 0 and c.tcgen05 > 0 and c.mma_sync == 0
    assert c.fams["UTCHMMA"] == c.tcgen05
    assert c.not_applicable.startswith("isa=tcgen05 ")

    sync = list(parse_sass(HMMA_SASS).instructions)
    c = SL.mma_isa_census(sync)
    assert c.wgmma == 0 and c.tcgen05 == 0 and c.mma_sync > 0
    assert c.fams["HMMA"] == c.mma_sync
    assert c.not_applicable.startswith("isa=mma-sync ")

    # no MMA at all is a wrong PAGE, not a wrong instrument: the model still applies
    none = list(parse_sass(NO_MMA_SASS).instructions)
    assert SL.mma_isa_census(none).not_applicable is None


def test_tcgen05_page_is_transcribed_with_its_own_grammar_not_a_zeroed_fail():
    """The synthetic page has UTCHMMA issues but no commit / wait pairing, so the
    tcgen05 grammar must FAIL on its own verdict line — and never print the
    Hopper zeros (hgmma_runs=0) or the retired NOTAPPLICABLE refusal."""
    buf = io.StringIO()
    ok_a, ok_b = SL.report(TCGEN05_SASS, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok_a is False
    # ten issues each closed by the elementwise run behind it -> ten runs
    assert "SCHED-FAIL isa=tcgen05 loop=backedge utcmma_runs=10 commits=0 mbar_waits=0" in text
    assert "SCHED-NOTAPPLICABLE" not in text
    assert "hgmma_runs=0" not in text
    assert "[isa] tcgen05 grammar" in text


def test_tcgen05_still_reports_liveness():
    """Register pressure is measured the same way on both ISAs — tcgen05 just
    keeps its accumulator out of the register file. The LIVE half must not be
    collateral damage of the SCHED refusal."""
    buf = io.StringIO()
    SL.report(TCGEN05_SASS, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert "=== liveness over doubled body ===" in text
    assert ("LIVE-OK" in text) or ("LIVE-FAIL" in text)


def test_hopper_path_is_unchanged_by_the_census():
    """A wgmma kernel must take exactly the old path."""
    buf = io.StringIO()
    SL.report(SASS, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert "SCHED-NOTAPPLICABLE" not in text
    assert "SCHED-FAIL too-few-instructions" in text


def test_mma_sync_page_is_transcribed_with_its_own_grammar():
    """A dsa_sparse B200 reference is an HMMA mainloop: it came
    back SCHED-FAIL hgmma_runs=0 six times in a day. The mma.sync grammar reads
    it: chains by accumulator base register, MMA runs, handoff objects."""
    buf = io.StringIO()
    ok_a, ok_b = SL.report(HMMA_SASS, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    # 75 HMMAs on 4 accumulator bases, no barrier / ldsm / depbar / mbarrier at
    # all: MMA runs exist but nothing to hang a schedule on -> FAIL on ITS line
    assert ok_a is False
    assert "SCHED-FAIL isa=mma_sync loop=backedge mma_runs=" in text
    assert "SCHED-NOTAPPLICABLE" not in text and "hgmma_runs=0" not in text
    assert "[chains] 1 mma.sync accumulator chains" in text     # one base, R8
    assert "=== liveness over doubled body ===" in text


def test_a_page_with_no_mma_at_all_still_fails_loudly():
    """A helper kernel or a prologue-only dump has no MMA of ANY ISA. That is not
    the instrument's fault and must not open the gate: SCHED-FAIL, as before."""
    buf = io.StringIO()
    ok_a, _ = SL.report(NO_MMA_SASS, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok_a is False
    assert "SCHED-NOTAPPLICABLE" not in text
    assert "SCHED-FAIL loop=backedge hgmma_runs=0" in text
    assert "[isa] MMA census: wgmma=0 tcgen05=0 mma_sync=0" in text


# --- the two Blackwell-era grammars on richer synthetic bodies -----------------
#
# Spellings are the ones real B200 pages carried (dsa_sparse: HMMA.16816 +
# BAR.SYNC + LDSM.16.MT88.4 + UBLKCP + SYNCS + UCGABAR_*; gdn_prefill: UTCHMMA +
# UTCBAR + SYNCS.PHASECHK.TRANS64.TRYWAIT + LDTM.16dp256bit.x8 + STTM + USETMAXREG
# + UTMALDG.3D + FENCE.VIEW.ASYNC.T, with an HMMA side path for the triangular
# inverse). The bodies are synthetic: the sealed references' listings stay in
# their runs.

def _pad(n, base):
    return [f"        /*{base + 16 * i:04x}*/  FFMA R{10 + i % 8}, R11, R12, R13 ;"
            for i in range(n)]


MMA_SYNC_BODY = "\n".join(
    ["        /*0100*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;",
     "        /*0110*/  LDGSTS.E.BYPASS.128 [R2], [R4.64] ;",
     "        /*0120*/  LDGDEPBAR ;",
     "        /*0130*/  DEPBAR.LE SB0, 0x1 ;",
     "        /*0140*/  LDSM.16.M88.4 R8, [R20] ;",
     "        /*0150*/  LDSM.16.MT88.4 R24, [R22] ;",
     "        /*0160*/  HMMA.16816.F32.BF16 R32, R8, R24, RZ ;",
     "        /*0170*/  HMMA.16816.F32.BF16 R32, R8, R26, R32 ;",
     "        /*0180*/  HMMA.16816.F32.BF16 R36, R12, R24, RZ ;",
     "        /*0190*/  HMMA.16816.F32.BF16 R36, R12, R26, R36 ;"]
    + _pad(12, 0x1a0)
    + ["        /*0260*/  BAR.SYNC.DEFER_BLOCKING 0x0 ;",
       "        /*0270*/  UBLKCP.S.G [UR4], [UR6], [UR8] ;",
       "        /*0280*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR67+0x2d850], R58 ;",
       "        /*0290*/  NANOSLEEP.SYNCS 0x989680 ;",
       "        /*02a0*/  HMMA.16816.F32.BF16 R32, R8, R24, R32 ;",
       "        /*02b0*/  HMMA.16816.F32.BF16 R36, R12, R24, R36 ;"]
    + _pad(10, 0x2c0)
    + ["        /*0360*/  UCGABAR_ARV ;",
       "        /*0370*/  UCGABAR_WAIT ;",
       "        /*0380*/  STAS.E.64 [R40], R32 ;",
       "        /*0390*/  SYNCS.ARRIVE.TRANS64 RZ, [UR67+0x2d860], R2 ;"]
    + _pad(500, 0x3a0)
    + ["        /*22e0*/  BRA 0x110 ;", "        /*22f0*/  EXIT ;"])


TCGEN05_BODY = "\n".join(
    ["        /*0100*/  USETMAXREG.TRY_ALLOC.CTAPOOL UP6, 0xe0 ;",
     "        /*0110*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR33+0x20], R5 ;",
     "        /*0120*/  UTCHMMA tmem[UR40], gdesc[UR18], gdesc[UR18], !UPT ;",
     "        /*0130*/  UTCHMMA tmem[UR40], gdesc[UR22], gdesc[UR22], UPT ;",
     "        /*0140*/  UTCBAR [UR41] ;",
     "        /*0150*/  UTCHMMA tmem[UR63], gdesc[UR18], gdesc[UR42], !UPT ;",
     "        /*0160*/  UTCHMMA tmem[UR63], gdesc[UR22], gdesc[UR44], UPT ;",
     "        /*0170*/  UTCBAR [UR64] ;",
     "        /*0180*/  SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [UR33+0x80], R5 ;",
     "        /*0190*/  LDTM.16dp256bit.x8 R8, [UR16+0x100] ;",
     "        /*01a0*/  LDSM.16.MT88.4 R40, [R20] ;",
     "        /*01b0*/  HMMA.16816.F32.BF16 R48, R40, R44, RZ ;",
     "        /*01c0*/  HMMA.16816.F32.BF16 R48, R40, R46, R48 ;",
     "        /*01d0*/  BAR.SYNC.DEFER_BLOCKING 0x2, 0x80 ;",
     "        /*01e0*/  STSM.16.MT88.4 [R30], R48 ;",
     "        /*01f0*/  MEMBAR.ALL.CTA ;",
     "        /*0200*/  SYNCS.ARRIVE.TRANS64 RZ, [UR33+0xa0], R3 ;"]
    + _pad(20, 0x210)
    + ["        /*0350*/  F2FP.BF16.F32.PACK_AB R60, R8, R9 ;",
       "        /*0360*/  STTM.16dp128bit.x8 [UR16+0x180], R60 ;",
       "        /*0370*/  FENCE.VIEW.ASYNC.T ;",
       "        /*0380*/  SYNCS.ARRIVE.TRANS64 RZ, [UR33+0x68], R3 ;",
       "        /*0390*/  UTMALDG.3D [UR4], [R30] ;"]
    + _pad(500, 0x3a0)
    + ["        /*22e0*/  BRA 0x110 ;", "        /*22f0*/  EXIT ;"])


def test_mma_sync_grammar_chains_events_and_handoffs():
    instrs = list(parse_sass(MMA_SYNC_BODY).instructions)
    lo, hi, how = SL.find_loop(instrs, lambda f: f in SL.MMA_SYNC_FAMILIES, "mma-envelope")
    assert how == "backedge"
    body = instrs[lo:hi + 1]
    chains, owner = SL.build_mma_sync_chains(body)
    # two accumulator bases (R32, R36), same mnemonic -> two chains, each first
    # issued with C=RZ (zero-init) and revisited across the barrier
    assert [c.acc_base for c in chains] == [32, 36]
    assert [c.zero_init for c in chains] == [True, True]
    assert [c.count for c in chains] == [3, 3]
    assert chains[0].acc_width == 4                      # f32 m16n8k16 fragment
    events = SL.transcribe_mma_sync(body, owner)
    kinds = [k for k, _ in events]
    assert kinds[:9] == ["CPASYNC", "CPCOMMIT", "DEPBAR", "LDSM", "LDSM",
                         "MMA", "MMA", "EW", "BAR"]
    assert "DEPBAR.LE SB0 0x1" in events[2][1]
    assert kinds[9:13] == ["TMA", "SYNCS", "SLEEP", "MMA"]
    assert "mbarrier.trywait [UR67+0x2d850]" == events[10][1]
    assert "CLUSTERBAR" in kinds and "DSMEM" in kinds
    # the decisive read: each elementwise run's handoff object
    pre = SL.handoff_verdict(events)
    assert pre[0] == "MMA:chain1"                          # first EW follows the MMA run
    assert pre[1] == "MMA:chain1"
    assert pre[2].startswith("mbarrier.arrive [UR67+0x2d860]")


def test_mma_sync_report_prints_the_isa_verdict_the_driver_reads():
    buf = io.StringIO()
    ok_a, ok_b = SL.report(MMA_SYNC_BODY, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok_a is True
    line = [ln for ln in text.splitlines() if ln.startswith("SCHED-")][0]
    assert line.startswith("SCHED-OK isa=mma_sync loop=backedge mma_runs=4 barriers=3 "
                           "ldsm=2 depbars=1 cp_async_groups=1 mbar_waits=1 ew_runs=3")
    assert "handoff_before_each_ew=[" in line
    from forgekernel.loop.driver import _SCHED_OK
    assert _SCHED_OK.search(text) is not None


def test_tcgen05_grammar_chains_by_tmem_operand_and_pairs_commits():
    instrs = list(parse_sass(TCGEN05_BODY).instructions)
    lo, hi, how = SL.find_loop(instrs, lambda f: bool(SL._TCGEN05_MMA.match(f)), "umma-envelope")
    assert how == "backedge"
    body = instrs[lo:hi + 1]
    chains, owner = SL.build_tcgen05_chains(body)
    tmem = [c for c in chains if isinstance(c, SL.TmemChain)]
    sync = [c for c in chains if isinstance(c, SL.SyncChain)]
    # accumulator identity is the TMEM operand, NOT the descriptor register:
    # UR40 and UR63 are two chains, gdesc[UR18] shared between them is not one
    assert [c.acc for c in tmem] == ["UR40", "UR63"]
    assert [c.zero_init for c in tmem] == [True, True]    # first issue !UPT
    assert [c.count for c in tmem] == [2, 2]
    assert [c.acc_base for c in sync] == [48]              # the HMMA side path
    events = SL.transcribe_tcgen05(body, owner)
    kinds = [k for k, _ in events]
    # the body starts at the backedge target (0x110): the SETMAXREG at 0x100
    # is prologue
    assert kinds[:7] == ["SYNCS", "UMMA", "COMMIT", "UMMA", "COMMIT", "SYNCS", "LDTM"]
    assert events[2][1] == "tcgen05.commit -> [UR41]"
    assert events[4][1] == "tcgen05.commit -> [UR64]"
    assert events[5][1] == "mbarrier.trywait [UR33+0x80]"
    assert "MMA" in kinds and "STTM" in kinds and "TMA" in kinds and "FENCE" in kinds
    pre = SL.handoff_verdict(events)
    # the first elementwise run sits behind the HMMA side path's arrive on 0xa0,
    # the second behind the STTM/fence/arrive on 0x68
    assert pre[0] == "mbarrier.arrive [UR33+0xa0]"
    assert pre[1] == "mbarrier.arrive [UR33+0x68]"


def test_tcgen05_report_prints_the_isa_verdict_the_driver_reads():
    buf = io.StringIO()
    ok_a, ok_b = SL.report(TCGEN05_BODY, out=lambda s: buf.write(s + "\n"))
    text = buf.getvalue()
    assert ok_a is True
    line = [ln for ln in text.splitlines() if ln.startswith("SCHED-")][0]
    assert line.startswith("SCHED-OK isa=tcgen05 loop=backedge utcmma_runs=2 commits=2 "
                           "mbar_waits=2 mbar_arrives=2 tmem_ld=1 tmem_st=1 "
                           "mma_sync_runs=1 barriers=1 ew_runs=2")
    assert "[chains] 3 accumulator chains in body (2 tcgen05/TMEM, 1 mma.sync side path)" in text
    from forgekernel.loop.driver import _SCHED_OK
    assert _SCHED_OK.search(text) is not None
    # liveness still measured; the TMEM load is its own provenance class
    assert "LIVE-" in text


def test_liveness_decodes_mma_sync_fragments_and_tmem_traffic():
    ins = list(parse_sass("        /*0100*/  HMMA.16816.F32.BF16 R32, R8, R24, R32 ;\n"
                          "        /*0110*/  LDTM.16dp256bit.x8 R8, [UR16+0x100] ;\n"
                          "        /*0120*/  STTM.16dp128bit.x8 [UR16+0x180], R60 ;\n"
                          "        /*0130*/  HMMA.1688.F16.F16 R4, R8, R24, RZ ;\n").instructions)
    d, u, g = SL.reg_accesses(ins[0])
    assert d == set(range(32, 36)) and u == set(range(8, 12)) | {24, 25} | set(range(32, 36))
    assert not g
    d, u, g = SL.reg_accesses(ins[1])
    assert d == set(range(8, 16)) and not g               # .x8 = 8 registers
    d, u, g = SL.reg_accesses(ins[2])
    assert u == set(range(60, 68)) and not d and not g
    d, u, g = SL.reg_accesses(ins[3])
    assert d == {4, 5}                                     # f16 accumulator, 1688 shape


def test_driver_sched_regex_accepts_all_three_forms_and_nothing_looser():
    from forgekernel.loop.driver import _SCHED_OK
    assert _SCHED_OK.search("SCHED-OK loop=backedge hgmma_runs=5 depbars=4 ew_runs=3\n")
    assert _SCHED_OK.search("SCHED-OK isa=mma_sync loop=backedge mma_runs=296 barriers=12 "
                            "ldsm=87 depbars=0 cp_async_groups=0 mbar_waits=8 ew_runs=27\n")
    assert _SCHED_OK.search("SCHED-OK isa=tcgen05 loop=backedge utcmma_runs=44 commits=15 "
                            "mbar_waits=16 mbar_arrives=9 tmem_ld=6 tmem_st=8\n")
    # a teacher's hand-written line (no isa=) now passes too
    assert _SCHED_OK.search("SCHED-OK loop=backedge mma_runs=296 barriers=22 ldsm=87\n")
    assert _SCHED_OK.search("SCHED-OK loop=backedge\n") is None
    assert _SCHED_OK.search("prose: SCHED-OK isa=tcgen05 loop=x utcmma_runs=1 commits=1 "
                            "mbar_waits=1\n") is None
