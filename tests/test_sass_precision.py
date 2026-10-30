"""The precision gate's SASS cross-check (warning only): MMA operand / accumulator types
off the orchestration census, against the reference dossier and the card's dtypes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import NCU, REC, FakeBackend, _ctx  # noqa: E402

from forgekernel.loop import precision_gate as pg  # noqa: E402
from forgekernel.loop import driver  # noqa: E402
from forgekernel.loop.driver import run_inner_iteration  # noqa: E402
from forgekernel.skills import orch_tools  # noqa: E402


@pytest.mark.parametrize("opcode, want", [
    ("HMMA.16816.F32.BF16", ("bf16", "f32")),
    ("HMMA.16816.F32", ("f16", "f32")),
    ("HMMA.16816.F16", ("f16", "f16")),
    ("HMMA.1688.F32.TF32", ("tf32", "f32")),
    ("HGMMA.64x128x16.F32.BF16", ("bf16", "f32")),
    ("QGMMA.64x128x32.F32.E4M3.E4M3", ("e4m3", "f32")),
    ("UTCHMMA", ("f16|bf16|tf32", "?")),
    ("UTCQMMA.2CTA", ("f8f6f4", "?")),
    ("FFMA", None),
])
def test_mma_operand_types(opcode, want):
    assert orch_tools.mma_operand_types(opcode) == want


def test_mma_precision_counts_classes():
    instrs = [SimpleNamespace(opcode=o) for o in
              ("HGMMA.64x128x16.F32.BF16", "HGMMA.64x128x16.F32.BF16", "FFMA", "HMMA.16816.F16")]
    got = orch_tools.mma_precision(instrs)
    assert got == {"ops": {"HGMMA:bf16/f32": 2, "HMMA:f16/f16": 1},
                   "inputs": ["bf16", "f16"], "accums": ["f16", "f32"]}


BF16 = {"ops": {"HGMMA:bf16/f32": 8}, "inputs": ["bf16"], "accums": ["f32"]}
FP8 = {"ops": {"QGMMA:e4m3/f32": 8}, "inputs": ["e4m3"], "accums": ["f32"]}


def test_sass_findings():
    assert pg.sass_findings(BF16, BF16, ["bfloat16"])["warnings"] == []
    w = pg.sass_findings(FP8, BF16, ["bfloat16", "bfloat16"])["warnings"]
    assert "MMA inputs e4m3 below the reference's bf16" in w
    assert "MMA inputs e4m3 below the card's bfloat16 inputs" in w
    tf32 = {"ops": {"HGMMA:tf32/f32": 4}, "inputs": ["tf32"], "accums": ["f32"]}
    assert pg.sass_findings(tf32, None, ["float32"])["warnings"] == [
        "MMA inputs tf32 below the card's float32 inputs"]
    f16acc = {"ops": {"HMMA:f16/f16": 4}, "inputs": ["f16"], "accums": ["f16"]}
    assert pg.sass_findings(f16acc, BF16, ["float16"])["warnings"] == [
        "MMA accumulates in f16, the reference in f32"]
    assert pg.sass_findings({"ops": {}}, BF16) == {}
    # an fp8 card: fp8 MMA is the card's own precision
    assert pg.sass_findings(FP8, None, ["float8_e4m3fn", "float32"])["warnings"] == []


def test_the_driver_records_and_logs_a_drop(tmp_path, monkeypatch, capsys):
    ev = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
          '{"bwd_median_ms": 2.4, "device_median_ms": 2.4}}')
    monkeypatch.setattr(driver, "_ncu_profiler",
                        lambda ctx, branch, correctness=True: {
                            driver._ORCH_PROF_KEY: {"loops": {}, "mma_precision": FP8}})
    monkeypatch.setattr(driver, "_read_ref_orch_dossier",
                        lambda ctx: "ORCH-SUMMARY-JSON: " + json.dumps({"mma_precision": BF16}))
    be = FakeBackend([REC("D1.1", 0)], [ev])
    ctx = _ctx(tmp_path, be)
    ctx.precision_contract = "official"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP"                                   # never gated
    assert rec.precision["sass"]["warnings"] == ["MMA inputs e4m3 below the reference's bf16"]
    assert "WARNING SASS precision: MMA inputs e4m3 below the reference's bf16" in capsys.readouterr().out


def test_contract_off_reports_no_sass_check(tmp_path, monkeypatch, capsys):
    ev = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
          '{"bwd_median_ms": 2.4, "device_median_ms": 2.4}}')
    monkeypatch.setattr(driver, "_ncu_profiler",
                        lambda ctx, branch, correctness=True: {
                            driver._ORCH_PROF_KEY: {"loops": {}, "mma_precision": FP8}})
    be = FakeBackend([REC("D1.1", 0)], [ev])
    ctx = _ctx(tmp_path, be)
    assert ctx.precision_contract == "off"
    rec = run_inner_iteration(ctx, 1, metric_before=2.5, last_good_ref="sha0", green=True)
    assert rec.verdict == "KEEP" and rec.precision == {}
    assert "SASS precision" not in capsys.readouterr().out


def test_the_census_prints_mma_precision_even_off_a_tiny_kernel(tmp_path, monkeypatch):
    from forgekernel.skills import orch_census_run
    sass = "\n".join(f"        /*{16 * i:04x}*/  {op} ;"
                     for i, op in enumerate(["HMMA.16816.F16 R0, R4, R8, R0", "FFMA R1, R2, R3, R4",
                                             "EXIT"]))
    monkeypatch.setattr(orch_census_run, "pick_kernel",
                        lambda d, out=print: ("k.cubin", "_Z4gemm", sass, 3))
    lines = []
    orch_census_run.census(str(tmp_path), out=lines.append)
    line = next(x for x in lines if x.startswith("SASS-PRECISION-JSON: "))
    got = json.loads(line.split(": ", 1)[1])
    assert got["kernel"] == "_Z4gemm" and got["accums"] == ["f16"]
    assert any(x.startswith("ORCH-CENSUS-SKIP") for x in lines)         # no dossier, still read
    m = driver._SASS_PRECISION_RE.search("\n".join(lines))
    assert m and json.loads(m.group(1))["inputs"] == ["f16"]
    assert "SASS-PRECISION-JSON" not in driver._without_census_lines("\n".join(lines))
