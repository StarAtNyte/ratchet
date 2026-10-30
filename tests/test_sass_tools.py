"""The SASS toolkit's contract: both observed dump formats parse, and every §5
failure mode from blackbox-fa-fwd-0810 turns into a loud SASS-FAIL instead of a
plausible wrong answer."""

import re

from forgekernel.skills import sass_tools as st

FA_KERNEL = ("kernel_cutlass_kernel_flash_attncuteflash_fwd_sm90"
             "FlashAttentionForwardSm90_16x384x128")
RNG_KERNEL = ("void at::(anonymous namespace)::distribution_elementwise_grid_"
              "stride_kernel<float, 4, ...>")
LAUNCH_PAIRS = [(RNG_KERNEL, 7584.0), (FA_KERNEL, 280512.0)]  # RNG launches first

# The style probe1's parser assumed: address in a /* */ comment, trailing ` ;` and
# an encoding comment. Verbatim cuobjdump shapes.
CUOBJDUMP_SAMPLE = """\
        Function : _ZN7cutlass13device_kernelE
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;      /* 0x00000a0000017a02 */
        /*0010*/                   S2R R0, SR_TID.X ;                           /* 0x0000000000007919 */
        /*0020*/                   UTMACCTL.PF.IVALL [UR4+0x70] ;               /* 0x0000000000007ab9 */
        /*0030*/                   HGMMA.64x128x16.F32.BF16.BF16 R0, gdesc[UR8], gdesc[UR10], R0 ; /* 0x0 */
        /*0040*/              @!P1 BRA 0x20 ;                                   /* 0x0000000000007947 */
"""

# The style the node's NCU actually emits (recorded verbatim from a real
# probe): a bare runtime-address column, no comments, no trailing semicolon.
NCU_SAMPLE = """\
    ADDRESS            SOURCE
    0x7fc07f7a1800           LDC R1, c[0x0][0x28]
    0x7fc07f7a1810           S2R R0, SR_TID.X
    0x7fc07f7a1820           UTMACCTL.PF.IVALL [UR4+0xf0]
    0x7fc07f7a1830           USETMAXREG.DEALLOC.CTAPOOL 0x28
    0x7fc07f7a1840           SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R11+URZ+0x30], R5
    0x7fc07f7a1850     @!P1  NANOSLEEP.SYNCS 0x989680
    0x7fc07f7a1860     @!P1  BRA 0x7fc07f7a1840
    0x7fc07f7a1870           HGMMA.64x128x16.F32.BF16.BF16 R24, gdesc[UR8], gdesc[UR10], R24
    0x7fc07f7a1880           EXIT
"""

# What a probe stream looks like when ncu could not do its job: counting these two
# lines as instructions once produced the infamous `opcodes={ERROR: 2}`.
ERROR_SAMPLE = """\
==ERROR== Could not open report file /tmp/rep.ncu-rep
==ERROR== An error occurred while trying to load the report
"""


def _fa_like_sass(n_filler: int) -> str:
    """A plausibly-sized tensor-kernel dump in NCU style: decisive opcodes up top,
    n_filler FFMA lines behind them."""
    lines = [NCU_SAMPLE]
    for i in range(n_filler):
        lines.append(f"    0x7fc07f7a{2000 + i:04x}           "
                     f"FFMA R{i % 32}, R2, R3, R{i % 32}")
    return "\n".join(lines)


def test_cuobjdump_format_parses_with_full_dotted_opcodes():
    parsed = st.parse_sass(CUOBJDUMP_SAMPLE)
    ops = [i.opcode for i in parsed.instructions]
    assert "HGMMA.64x128x16.F32.BF16.BF16" in ops   # not truncated at the first dot
    assert "UTMACCTL.PF.IVALL" in ops
    assert "BRA" in ops                              # predicated line still parses
    assert not parsed.errors
    # the Function banner is unparsed, never misread as an instruction
    assert parsed.unparsed == 1


def test_ncu_bare_address_format_parses_the_same_stream():
    """The format probe1's parser could not see: 2615 correct lines, 0 opcodes."""
    parsed = st.parse_sass(NCU_SAMPLE)
    ops = [i.opcode for i in parsed.instructions]
    assert "USETMAXREG.DEALLOC.CTAPOOL" in ops
    assert "SYNCS.PHASECHK.TRANS64.TRYWAIT" in ops
    assert "NANOSLEEP.SYNCS" in ops                  # predicate stripped, opcode kept
    assert "EXIT" in ops                             # operand-less instruction
    assert len(parsed.instructions) == 9
    assert parsed.unparsed == 1                      # the ADDRESS/SOURCE header


def test_profiler_error_lines_are_evidence_not_instructions():
    parsed = st.parse_sass(ERROR_SAMPLE)
    assert parsed.instructions == []
    assert len(parsed.errors) == 2


def test_pick_kernel_ranks_by_duration_not_launch_order(capsys):
    """probe2 took kernels[0] — the first launch is torch.randn's RNG kernel."""
    name, dur = st.pick_kernel(LAUNCH_PAIRS)
    assert name == FA_KERNEL and dur == 280512.0
    assert "ranking" in capsys.readouterr().out


def test_self_check_catches_the_wrong_kernel():
    parsed = st.parse_sass(_fa_like_sass(600))
    problems = st.self_check(RNG_KERNEL, FA_KERNEL, parsed)
    assert any("bystander" in p for p in problems)


def test_self_check_catches_a_helper_kernel_sized_dump():
    """400 instructions against 2615 lines was the unread alarm of probe2."""
    parsed = st.parse_sass(_fa_like_sass(100))      # well under the 500 floor
    problems = st.self_check(FA_KERNEL, FA_KERNEL, parsed)
    assert any("< 500" in p for p in problems)


def test_self_check_calls_a_zero_parse_a_parser_story():
    parsed = st.parse_sass("garbage that matches nothing\nsecond line")
    problems = st.self_check(FA_KERNEL, FA_KERNEL, parsed)
    assert any("the parser, not the kernel" in p for p in problems)


def test_self_check_requires_a_tensor_family():
    """An RNG kernel parses cleanly — MUFU.SIN, BAR.SYNC — and must still fail:
    its all-ABSENT decisive table was self-consistent on the wrong kernel."""
    rng = "\n".join(f"    0x7fc07f7a{1000 + i:04x}           MUFU.SIN R4, R{i % 8}"
                    for i in range(600)) + \
          "\n    0x7fc07f7b0000           BAR.SYNC 0x0"
    parsed = st.parse_sass(rng)
    problems = st.self_check(FA_KERNEL, FA_KERNEL, parsed)
    assert any("not the tensor kernel" in p for p in problems)


def test_analyze_success_speaks_the_verdict_line_the_driver_gates_on(capsys):
    ok = st.analyze(_fa_like_sass(600), FA_KERNEL, LAUNCH_PAIRS)
    out = capsys.readouterr().out
    assert ok
    m = re.search(r"^SASS-OK kernel=\S.* instructions=(\d+) opcodes=\d+$",
                  out, re.M)
    assert m and int(m.group(1)) >= 500
    assert "opcode histogram" in out


def test_analyze_failure_dumps_raw_evidence_and_never_says_ok(capsys):
    """Raw lines beat a wrong summary: §1.2 of a recon report was rebuilt from the
    50 raw lines a failing probe printed."""
    ok = st.analyze(ERROR_SAMPLE, FA_KERNEL, LAUNCH_PAIRS)
    out = capsys.readouterr().out
    assert not ok
    assert "SASS-FAIL" in out and "SASS-OK" not in out
    assert "raw dump" in out and "Could not open report file" in out
