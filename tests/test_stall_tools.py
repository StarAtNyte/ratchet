"""The per-SASS stall pricing ledger: keying, coverage contract, verdict gate."""
from forgekernel.loop.driver import _STALL_LEDGER_RE, _STALL_OK, _STALL_WAIVED
from forgekernel.skills import stall_tools

LS = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
BAR = "smsp__pcsamp_warps_issue_stalled_barrier"


def _per_pc(spec):
    """{pc: total} -> per_pc dict with a fixed stall split."""
    return {pc: {LS: (2 * v) // 3, BAR: v - (2 * v) // 3} for pc, v in spec.items()}


def test_build_ledger_keys_on_source_lines_when_mapping_resolves():
    per_pc = _per_pc({0x100: 600, 0x110: 300, 0x900: 100})
    pc2sass = {0x100: "HMMA.16816.F32 R0, R2, R4", 0x110: "MUFU.EX2 R8, R9",
               0x900: "LDS.128 R12, [R20]"}
    loc = {0x100: "kernel.py:101", 0x110: "kernel.py:101", 0x900: "kernel.py:230"}
    out = []
    total, rows, cov = stall_tools.build_ledger(
        per_pc, pc2sass, lambda pc: loc.get(pc, ""), out=out.append)
    text = "\n".join(out)
    assert total == 1000 and cov >= 90.0
    assert "keying=source-line" in text
    assert "kernel.py:101" in text and "kernel.py:230" in text
    # the opcode-family table rides along regardless of keying
    assert "HMMA" in text and "MUFU" in text


def test_build_ledger_falls_back_to_opcode_buckets_without_lineinfo():
    # the ref side: a blackbox binary compiled without lineinfo
    per_pc = _per_pc({0x100: 500, 0x300: 400, 0x2100: 100})
    pc2sass = {0x100: "WARPGROUP.DEPBAR.LE 1", 0x300: "WARPGROUP.DEPBAR.LE 1",
               0x2100: "STS.128 [R4], R8"}
    out = []
    total, rows, cov = stall_tools.build_ledger(
        per_pc, pc2sass, lambda pc: "", out=out.append)
    text = "\n".join(out)
    assert "keying=opcode/pc-bucket" in text
    assert "WARPGROUP@" in text and cov >= 90.0


def test_ledger_rows_sum_to_the_coverage_floor_not_just_the_top():
    # a flat distribution: rows must keep printing past the top handful
    per_pc = _per_pc({0x100 + 0x400 * i: 50 for i in range(40)})
    out = []
    total, rows, cov = stall_tools.build_ledger(
        per_pc, {}, lambda pc: "", out=out.append)
    assert cov >= stall_tools.COVERAGE_FLOOR
    assert rows >= 36  # 90% of 40 equal buckets


def test_verdict_and_marker_spellings_match_the_driver_gate():
    ok_line = "STALL-OK kernel=main_kernel samples=242351 rows=18 coverage=93.4"
    assert _STALL_OK.search(ok_line)
    # prose cannot open the gate (the ^ anchor, same trust model as SASS-OK)
    assert not _STALL_OK.search("the probe should print STALL-OK kernel=x " )
    assert _STALL_WAIVED.search("STALL-WAIVED: pcsamp metrics empty on this image")
    body = f"{stall_tools.BEGIN}\nrow1\nrow2\n{stall_tools.END}"
    m = _STALL_LEDGER_RE.search(f"noise\n{body}\nSTALL-OK kernel=k samples=9 "
                                f"rows=1 coverage=99.0")
    assert m and m.group(1) == "row1\nrow2"


def test_price_report_fails_closed_without_the_ncu_report_module(monkeypatch, tmp_path):
    monkeypatch.setattr(stall_tools, "_import_ncu_report", lambda: None)
    out = []
    ok = stall_tools.price_report(tmp_path / "x.ncu-rep", out=out.append)
    assert ok is False
    assert any(line.startswith("STALL-FAIL") for line in out)
    assert not any(_STALL_OK.search(line) for line in out)
