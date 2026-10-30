"""The compiler-fallback census: wgmma waits per wgmma off the SASS dump
(sass_census), the ratio in the NCU digest's thresholds, the census runner
printing it, the driver's standing student note, and the timeline probe's
site-cap fall-through — the three mechanisms a gdn run once lacked."""
import json
import os
import sys
from pathlib import Path

import pytest

from forgekernel.skills import sass_census as C
from forgekernel.inner.ncu import summarize_metrics
from forgekernel.loop.driver import _parse_serial_census, _sass_serial_status
from forgekernel.skills import orch_census_run as R
from forgekernel.skills import timeline_probe as T

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx  # noqa: E402
from test_orch_tools import _big_page  # noqa: E402


def _chain(n_mma: int, wait_each: bool) -> str:
    out = ["0x7f0000000000  WARPGROUP.ARRIVE"]
    a = 0x7f0000000010
    for i in range(n_mma):
        out.append(f"{hex(a)}  HGMMA.64x64x16.F32.BF16 R56, gdesc[UR8], R56")
        a += 16
        if wait_each:
            out.append(f"{hex(a)}  WARPGROUP.DEPBAR.LE gsb0, 0x0")
            a += 16
    if not wait_each:
        out.append(f"{hex(a)}  WARPGROUP.DEPBAR.LE gsb0, 0x0")
    return "\n".join(out) + "\n"


def test_pipelined_vs_serialized_ratio():
    piped = C.wgmma_serialization_from_text(_chain(8, wait_each=False))
    assert piped == {"wgmma_total": 8, "wgmma_depbar0": 1, "wgmma_tight": 1,
                     "wgmma_serial_ratio": 0.125}
    serial = C.wgmma_serialization_from_text(_chain(8, wait_each=True))
    assert serial["wgmma_serial_ratio"] == 1.0 and serial["wgmma_tight"] == 8
    # a DEPBAR at another depth is not a full drain
    assert C.wgmma_serialization_from_text(
        "0x10  HGMMA.64x64x16.F32.BF16 R56, gdesc[UR8], R56\n"
        "0x20  WARPGROUP.DEPBAR.LE gsb0, 0x1\n")["wgmma_depbar0"] == 0


def test_render_parse_roundtrip_and_ptxas_advisory():
    line = C.render_serial_line(C.wgmma_serialization_from_text(_chain(4, True)), "k")
    assert line == "WGMMA-SERIAL kernel=k wgmma=4 depbar0=4 tight=4 ratio=1.0"
    log = ("ptxas info: Potential Performance Loss: wgmma.mma_async instructions are "
           "serialized due to insufficient register resources for the wgmma pipeline\n"
           + line + "\n")
    f = C.parse_serial_line(log)
    assert f["wgmma_serial_ratio"] == 1.0 and f["wgmma_ptxas_serialized"] == 1
    assert C.parse_serial_line(line)["wgmma_ptxas_serialized"] == 0
    assert C.render_serial_line({"wgmma_total": 0}, "k").startswith("WGMMA-SERIAL-SKIP")
    assert C.parse_serial_line("WGMMA-SERIAL-SKIP kernel=k reason=x") == {}


def test_digest_flags_serialization_first():
    s = summarize_metrics({"wgmma_serial_ratio": 1.0, "bankconf_st": 20_000_000,
                           "tensor_pct": 40.0})
    assert s["verdict"].startswith("WGMMA SERIALIZED by ptxas")
    sev = {f["key"]: f["severity"] for f in s["flags"]}
    assert sev["wgmma_serial_ratio"] == "severe" and sev["bankconf_st"] == "severe"
    assert summarize_metrics({"wgmma_serial_ratio": 0.15})["flags"] == []
    assert summarize_metrics({"wgmma_serial_ratio": 0.6})["flags"][0]["severity"] == "elevated"


def test_verdict_names_a_regression_against_the_previous_keep():
    cur = {"wgmma_total": 28, "wgmma_depbar0": 28, "wgmma_tight": 28, "wgmma_serial_ratio": 1.0}
    assert C.serialization_verdict({"wgmma_serial_ratio": 0.43}, cur).startswith(
        "\n## Compiler collateral check\n- **WGMMA serialization regression**")
    assert "whole function" in C.serialization_verdict(None, cur)
    assert C.serialization_verdict({"wgmma_serial_ratio": 1.0}, cur).startswith(
        "\n## Compiler collateral check\n- **WGMMA serialization (persistent)**")
    assert C.serialization_verdict(None, dict(cur, wgmma_serial_ratio=0.2)) is None
    assert C.serialization_verdict(None, {"wgmma_total": 0}) is None


def test_census_runner_prints_the_serial_line_after_the_summary(tmp_path, monkeypatch):
    d = tmp_path / "dump"
    (d / "a.cubin").parent.mkdir(parents=True)
    (d / "a.cubin").write_bytes(b"\x7fELF")
    page = "Function : main_kernel\n" + _big_page()
    monkeypatch.setattr(R, "disassemble", lambda cubin: (0, page))
    lines = []
    R.census(str(d), out=lines.append)
    assert [l for l in lines if l.startswith("ORCH-SUMMARY-JSON: ")]
    serial = [l for l in lines if l.startswith("WGMMA-SERIAL")]
    assert len(serial) == 1 and "kernel=main_kernel" in serial[0]
    # after the NCU stdout, before the two census lines that close the log
    assert lines.index(serial[0]) < lines.index(
        next(l for l in lines if l.startswith("ORCH-CENSUS ")))


def test_driver_folds_the_ratio_into_the_digest_and_the_student_note(tmp_path):
    serial_log = ("WGMMA-SERIAL kernel=k wgmma=28 depbar0=28 tight=28 ratio=1.0\n")
    be = FakeBackend([REC("D2.3", 2.6)], [EVAL(2.6), NCU + "\n" + serial_log])
    ctx = _ctx(tmp_path, be)
    from forgekernel.loop.driver import run_inner_iteration
    rec = run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "KEEP"
    # NCU parse yields {} for the "(no csv)" fixture, so the digest stays empty
    # and the serial fields are not folded (no profile, no verdict)
    assert ctx.sass_serial_note == ""
    # with a digest present the fields join it and the note stands
    prof = {"time_ns": 1.0, "wgmma_serial_ratio": 1.0, "wgmma_total": 28,
            "wgmma_depbar0": 28, "wgmma_tight": 28, "wgmma_ptxas_serialized": 0}
    raw = _parse_serial_census(prof)
    assert raw["wgmma_serial_ratio"] == 1.0 and "time_ns" in prof
    note = _sass_serial_status(ctx, raw)
    assert "Compiler collateral check" in note and "whole function" in note
    # the census absent on a later round leaves the standing note in place
    ctx.sass_serial_note = note
    assert _sass_serial_status(ctx, {}) == note


def test_timeline_site_cap_falls_through_to_wait_sites():
    body = "\n".join(f'    _seg("r.s{i}.x", ci)' for i in range(T.NSITE + 3))
    src = ('def _seg(site, row=0):\n    return None\n\n'
           'def kern():\n    for ci in range(4):\n' + body +
           '\n        _seg("r.w.begin", ci)\n        _seg("r.w.end", ci)\n')
    assert len(T.markers(src)) == T.NSITE + 5
    with pytest.raises(T.SiteCapExceeded):
        T.stamp_source(src, "all")
    stamped, kept = T.stamp_source(src, "waits")
    assert kept == ["r.w.begin", "r.w.end"] and "_tl_store(gate" in stamped
    assert T.NSITE == 128
