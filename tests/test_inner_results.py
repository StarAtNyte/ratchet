"""Tests for parsing GPU job output (correctness + perf) from job logs."""

from forgekernel.inner import (
    parse_correctness,
    parse_perf,
)

_CORRECT_PASS = """\
  PASS  forward (causal=False)
  PASS  forward (causal=True)
  PASS  backward dQ/dK/dV (causal=False)
  PASS  backward dQ/dK/dV (causal=True)
all 4 cases PASS
"""

_CORRECT_FAIL = """\
  PASS  forward (causal=False)
  FAIL  backward dQ/dK/dV (causal=True)
        max_diff=4.28e+00 exceeds 1e-2
1/4 FAILED
"""

_PERF = """\
=== Correctness (production shape) ===
{
  "fwd_max_diff": 0.0,
  "dq": 0.00195,
  "dk": 0.0039,
  "dv": 0.0078,
  "pass": true
}
=== Performance (BWD) ===
{
  "shape": "B=10 H_q=16 H_kv=2 N=4096 D=64 causal GQA-8:1",
  "bwd_median_ms": 2.381,
  "bwd_min_ms": 2.38,
  "bwd_max_ms": 2.40,
  "bwd_mfu_pct": 36.46,
  "device": "H100",
  "repeat": 10
}
=== JSON ===
{"correctness": {"fwd_max_diff": 0.0, "dq": 0.00195, "dk": 0.0039, "dv": 0.0078, "pass": true}, "performance": {"shape": "B=10 ...", "bwd_median_ms": 2.381, "bwd_min_ms": 2.38, "bwd_max_ms": 2.40, "bwd_mfu_pct": 36.46, "device": "H100", "repeat": 10}}
"""


def test_parse_correctness_pass():
    r = parse_correctness(_CORRECT_PASS)
    assert r["passed"] is True
    assert r["fails"] == 0
    assert len(r["cases"]) == 4


def test_parse_correctness_fail():
    r = parse_correctness(_CORRECT_FAIL)
    assert r["passed"] is False
    assert r["fails"] == 1
    assert any(c["status"] == "FAIL" for c in r["cases"])


def test_parse_perf_extracts_json_block():
    r = parse_perf(_PERF)
    assert r["passed"] is True
    assert r["bwd_median_ms"] == 2.381
    assert r["bwd_mfu_pct"] == 36.46
    assert r["dq"] == 0.00195


def test_parse_perf_missing_returns_none_fields():
    r = parse_perf("no json here")
    assert r["bwd_median_ms"] is None
    assert r["passed"] is None
