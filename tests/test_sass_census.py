"""Tests for the MMA operand-source census (the [VERIFY: sass-derived] channel)."""

from forgekernel.skills.sass_census import (
    census_from_text,
    parse_census_line,
    render_census_line,
)

# Both dump styles carry both operand forms: RS (A = R8) and SS (A = gdesc[...]).
CUOBJDUMP_STYLE = """\
        Function : student_kernel
/*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;   /* 0x00000a00ff017624 */
/*0010*/                   HGMMA.64x64x16.F32 R80, R8, gdesc[UR6], !UPT ;  /* 0x... */
/*0020*/                   HGMMA.64x64x16.F32 R80, gdesc[UR4], gdesc[UR6] ;  /* 0x... */
/*0030*/                   HGMMA.64x128x16.F32 R24, R8, gdesc[UR8] ;  /* 0x... */
/*0040*/                   BAR.SYNC 0x0 ;  /* 0x... */
"""

NCU_STYLE = """\
0x7fc07f7a1800           LDC R1, c[0x0][0x28]
0x7fc07f7a1810           QGMMA.64x64x32.F32.E4M3 R40, gdesc[UR4], gdesc[UR6]
0x7fc07f7a1820           HGMMA.64x64x16.F32 R80, R12, gdesc[UR6]
"""


def test_census_counts_both_forms_and_styles():
    c = census_from_text(CUOBJDUMP_STYLE)
    assert c == {"mma_total": 3, "mma_a_reg": 2, "mma_a_desc": 1, "mma_a_other": 0}
    c2 = census_from_text(NCU_STYLE)
    assert c2 == {"mma_total": 2, "mma_a_reg": 1, "mma_a_desc": 1, "mma_a_other": 0}


def test_non_mma_instructions_do_not_count():
    c = census_from_text("/*0000*/  UTMALDG.2D [UR4], [R2] ;\n/*0010*/  BAR.SYNC 0x0 ;\n")
    assert c["mma_total"] == 0


def test_render_and_parse_roundtrip():
    line = render_census_line(census_from_text(CUOBJDUMP_STYLE), "student_kernel")
    fields = parse_census_line(f"noise before\n{line}\nnoise after\n")
    assert fields == {"mma_census_kernel": "student_kernel", "mma_total": 3,
                      "mma_a_reg": 2, "mma_a_desc": 1, "mma_a_other": 0}


def test_zero_mma_renders_fail_line_and_parses_to_empty():
    line = render_census_line(census_from_text("no instructions here"), "k")
    assert line.startswith("SASS-OPERANDS-FAIL")
    assert parse_census_line(line) == {}


def test_parse_takes_the_last_line():
    first = render_census_line(census_from_text(NCU_STYLE), "old")
    second = render_census_line(census_from_text(CUOBJDUMP_STYLE), "new")
    fields = parse_census_line(f"{first}\n{second}\n")
    assert fields["mma_census_kernel"] == "new"
    assert fields["mma_total"] == 3


def test_absent_line_returns_empty_so_digests_stay_unchanged():
    assert parse_census_line("plain ncu csv output, no census ran") == {}
