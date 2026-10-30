"""The precision gate's measure (src/forgekernel/card_template/eval/precision.py) and a leaderboard
card's record of it (eval/leaderboard.py). Needs torch (skipped without it)."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_EVAL = REPO / "src" / "forgekernel" / "card_template" / "eval"

torch = pytest.importorskip("torch")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("fk_template_precision", TEMPLATE_EVAL / "precision.py")
lb = _load("fk_template_leaderboard_prec", TEMPLATE_EVAL / "leaderboard.py")


def _rms_ref(x, w):
    xf = x.to(torch.float32)
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
    return ((xf * inv) * w.float()).to(x.dtype)


def test_upcast_lifts_every_float_cast_and_keeps_the_rng():
    seen = []

    def ref(x):
        seen.append((x.dtype, x.float().dtype, x.to(torch.bfloat16).dtype,
                     torch.zeros(2, dtype=torch.float32).dtype, x.half().dtype))
        return torch.randn(3)

    x = torch.randn(4, dtype=torch.bfloat16)
    state = torch.random.get_rng_state()
    P.upcast_call(ref, [x])
    assert set(seen[0]) == {torch.float64}
    assert torch.equal(torch.random.get_rng_state(), state)


def test_truth_falls_back_to_float32_then_unavailable():
    def no_double(x):
        if x.dtype == torch.float64:
            raise RuntimeError("no float64 kernel")
        return x * 2

    outs, basis, why = P.truth_of(no_double, [torch.ones(2, dtype=torch.bfloat16)])
    assert basis == "float32" and outs[0].dtype == torch.float32

    def never(x):
        raise RuntimeError("boom")

    outs, basis, why = P.truth_of(never, [torch.ones(2)])
    assert outs is None and basis == "unavailable" and "float64" in why and "float32" in why


def test_rel_l2_edges():
    t = torch.tensor([1.0, -math.inf, 2.0])
    assert P.rel_l2(torch.tensor([1.0, -math.inf, 2.0]), t) == 0.0
    assert P.rel_l2(torch.tensor([1.0, 0.0, 2.0]), t) == math.inf
    assert P.rel_l2(torch.tensor([math.nan, -math.inf, 2.0]), t) == math.inf
    assert P.rel_l2(torch.zeros(2), torch.zeros(2)) == 0.0


def _chunked(a, b, dt, c=16):
    acc = None
    for i in range(0, a.shape[1], c):
        part = a[:, i:i + c].float() @ b[:, i:i + c].float().T
        acc = part.to(dt) if acc is None else (acc.float() + part).to(dt)
    return acc.to(a.dtype)


def test_an_honest_kernel_passes_and_reduced_precision_accumulation_fails():
    torch.manual_seed(0)
    a, b = torch.randn(32, 2048, dtype=torch.float16), torch.randn(64, 2048, dtype=torch.float16)
    gemm = lambda a, b: a @ b.T  # noqa: E731
    ref = gemm(a, b)
    rec = P.PrecisionRecorder()
    rec.measure("honest", _chunked(a, b, torch.float32, 256), ref, lambda: P.truth_of(gemm, [a, b]))
    rec.measure("f16acc", _chunked(a, b, torch.float16, 16), ref, lambda: P.truth_of(gemm, [a, b]))
    rec.measure("bf16acc", _chunked(a, b, torch.bfloat16, 128), ref, lambda: P.truth_of(gemm, [a, b]))
    status = {r["case"]: r["status"] for r in rec.rows}
    assert status == {"honest": "pass", "f16acc": "fail", "bf16acc": "fail"}
    block = rec.block()
    assert block["status"] == "fail" and block["n_failed"] == 2 and block["truth"] == "float64"
    assert "e_cand" in block["why"]


def test_rows_that_are_not_readings_say_why():
    x = torch.randn(8, 16)
    rec = P.PrecisionRecorder()
    rec.measure("fp32", [x], [x], ([x.clone()], "float32", ""))            # truth no finer
    rec.measure("int", [torch.arange(3)], [torch.arange(3)], [torch.arange(3)])
    rec.measure("random", [x], [x], [torch.randn(8, 16)])                 # e_ref ~ 1.4
    rec.measure("gone", [x], [x], (None, "unavailable", "float64: boom; float32: boom"))
    rec.measure("raises", [x], [x], lambda: 1 / 0)
    st = [r["status"] for r in rec.rows]
    assert st == ["unavailable", "n/a", "unavailable", "unavailable", "unavailable"]
    block = rec.block()
    assert block["status"] == "unavailable" and "boom" in block["why"]


def test_stress_profiles():
    x = torch.randn(4096, dtype=torch.bfloat16)
    assert torch.equal(P.stress_tensor(x, "amp"), (x.float() * 4).to(torch.bfloat16))
    assert torch.allclose(P.stress_tensor(x, "offset").float(), x.float() + 1, atol=0.05)
    s = P.stress_tensor(x, "spikes", torch.Generator().manual_seed(1))
    assert int((s.float().abs() == 12).sum()) == 4 and s.dtype == torch.bfloat16
    with pytest.raises(ValueError):
        P.stress_tensor(x, "nope")


def test_merge_blocks_names_the_workloads():
    rec = P.PrecisionRecorder()
    x = torch.randn(4, 8)
    rec.measure("round0", [x], [x], [x.double()])
    merged = P.merge_blocks([("wl1", rec.block()), ("wl2", rec.block())])
    assert [r["case"] for r in merged["rows"]] == ["wl1/round0", "wl2/round0"]
    assert P.merge_blocks([]) is None


# --- a leaderboard card records it, and never moves the verdict ----------------------------

RMS_DEF = {
    "name": "rmsnorm_h512", "op_type": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 512}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
               "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
    "reference": ("import torch\n\ndef run(hidden_states, weight):\n"
                  "    x = hidden_states.to(torch.float32)\n"
                  "    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)"
                  " * weight.to(torch.float32)\n"
                  "    return y.to(hidden_states.dtype)\n"),
}


def _problem(contract="official"):
    p = {"suite": lb.SUITE_SOL, "definition": json.loads(json.dumps(RMS_DEF)),
         "workload": {"uuid": "w0", "axes": {"batch_size": 16},
                      "inputs": {"hidden_states": {"type": "random"},
                                 "weight": {"type": "random"}}},
         "dps": True, "_dir": "."}
    if contract is not None:
        p["precision_contract"] = contract
    return p


def _rms_fp32(hidden_states, weight, output):
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())


def _rms_bf16(hidden_states, weight, output):
    # inside the official 1e-2 tolerance, ~1e2 x the fp32 reference's own error
    x = hidden_states.to(torch.bfloat16)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.to(torch.bfloat16)
    output.copy_(y.float())


def test_the_card_records_precision_and_keeps_the_boards_verdict(monkeypatch):
    honest, _ = lb._correctness(_problem(), _rms_fp32, True, "cpu")
    lowp, _ = lb._correctness(_problem(), _rms_bf16, True, "cpu")
    assert honest["pass"] and lowp["pass"]                      # the board's verdict
    assert honest["precision"]["status"] == "pass"
    assert lowp["precision"]["status"] == "fail"
    stress = {r.get("stress") for r in honest["precision"]["rows"]}
    assert {"offset", "spikes", "amp"} <= stress
    assert len([r for r in honest["precision"]["rows"] if r["case"].startswith("round")]) \
        == lb.PRECISION_SETS
    # the suite's own draws are untouched by the record (seed-200 sequence, every round)
    monkeypatch.setattr(lb, "PRECISION_ENV_OFF", True)
    plain, _ = lb._correctness(_problem(), _rms_bf16, True, "cpu")
    assert "precision" not in plain
    assert [c["max_abs"] for c in plain["checks"]] == [c["max_abs"] for c in lowp["checks"]]


def test_the_json_carries_it_at_the_top_level():
    verdict, _ = lb._correctness(_problem(), _rms_bf16, True, "cpu")
    assert "precision" not in lb._correctness_json(verdict)
    view = dict(_problem(), _wl_name="wl0")
    assert lb.precision_result([(view, verdict)])["status"] == "fail"
    merged = lb.precision_result([(dict(view, _wl_name="a"), verdict),
                                  (dict(view, _wl_name="b"), verdict)])
    assert merged["rows"][0]["case"].startswith("a/")


def test_a_card_without_the_helper_says_so(tmp_path):
    shutil.copy2(TEMPLATE_EVAL / "leaderboard.py", tmp_path / "leaderboard.py")
    old = _load("fk_lb_no_helper", tmp_path / "leaderboard.py")
    verdict, _ = old._correctness(_problem(), _rms_fp32, True, "cpu")
    assert verdict["pass"] and verdict["precision"]["status"] == "unavailable"
    assert "precision.py is missing" in verdict["precision"]["why"]


def test_the_reference_timed_alone_and_correctness_only_workloads_are_not_measured():
    v, _ = lb._correctness(dict(_problem(), _no_precision=True), _rms_fp32, True, "cpu")
    c, _ = lb._correctness(dict(_problem(), _check_only=True), _rms_fp32, True, "cpu")
    assert "precision" not in v and "precision" not in c


def test_an_fp32_output_a_few_ulps_off_is_not_a_precision_drop():
    """fast-math exp2 / another reduction order: ~1e-6 from the truth against a
    reference at ~1e-7 is a ratio of 10 that the relative floor (2^-17) absorbs; a
    tf32-class error (~1e-4) is not absorbed."""
    t = torch.randn(4096, dtype=torch.float64)
    ref = (t * (1 + 1e-7 * torch.randn(4096, dtype=torch.float64))).float()
    close = (t * (1 + 1e-6 * torch.randn(4096, dtype=torch.float64))).float()
    tf32ish = (t * (1 + 2e-4 * torch.randn(4096, dtype=torch.float64))).float()
    rec = P.PrecisionRecorder()
    rec.measure("close", [close], [ref], [t])
    rec.measure("tf32ish", [tf32ish], [ref], [t])
    assert [r["status"] for r in rec.rows] == ["pass", "fail"]
    assert rec.rows[0]["ratio"] > 5


def test_a_leaderboard_card_by_default_judges_exactly_as_its_board(monkeypatch):
    """No precision_contract in problem.json (the default, off): no float64 re-run, no
    stressed inputs, no block -- the same verdict, from the same draws, as the board."""
    calls = []
    monkeypatch.setattr(lb, "_precision", lambda: calls.append(1) or P)
    off, _ = lb._correctness(_problem(None), _rms_bf16, True, "cpu")
    on, _ = lb._correctness(_problem("official"), _rms_bf16, True, "cpu")
    assert "precision" not in off and on["precision"]["status"] == "fail"
    assert [c["max_abs"] for c in off["checks"]] == [c["max_abs"] for c in on["checks"]]
    assert off["pass"] is on["pass"] is True
    assert lb.precision_on(_problem(None)) is False and lb.precision_on(_problem("off")) is False
    assert lb.precision_on(_problem("strict")) is True
    n_on = len(calls)
    lb._correctness(_problem(None), _rms_bf16, True, "cpu")
    assert len(calls) == n_on                            # off never reaches precision.py


def test_the_kernel_cannot_switch_the_measure_off_from_its_own_process(monkeypatch):
    """FK_PRECISION counts as read when the bench was imported. The kernel module is
    imported later, in the same process: setting it then (here, before _correctness, as
    an import-time side effect would) changes nothing."""
    monkeypatch.setattr(lb, "PRECISION_ENV_OFF", False)
    monkeypatch.setenv(lb.PRECISION_ENV, "0")
    verdict, _ = lb._correctness(_problem("strict"), _rms_bf16, True, "cpu")
    assert verdict["precision"]["status"] == "fail"


def _rms_add_inplace(hidden_states, residual, weight, output):
    # fused add + rmsnorm that writes the sum back into ``residual`` (in place)
    residual.add_(hidden_states)
    x = residual.float()
    output.copy_((x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
                  * weight.float()).to(output.dtype))


def test_the_truth_runs_on_the_inputs_as_drawn_not_as_an_in_place_reference_left_them():
    """The reference writes into ``residual``; the float64 truth must see the drawn
    residual, or e_ref is large (row unavailable, gate silently off) and an honest
    kernel is compared against the truth of another problem."""
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    res = torch.randn(8, 64, dtype=torch.bfloat16)
    w = torch.randn(64, dtype=torch.bfloat16)
    pristine = P.snapshot([x, res, w])
    ref_out = torch.empty(8, 64, dtype=torch.bfloat16)
    _rms_add_inplace(x, res, w, ref_out)                # the reference writes into res

    def ref(hidden_states, residual, weight):
        out = torch.empty(hidden_states.shape, dtype=torch.bfloat16)
        _rms_add_inplace(hidden_states, residual, weight, out)
        return out

    stale = P.PrecisionRecorder()
    stale.measure("stale", [ref_out.clone()], [ref_out], lambda: P.truth_of(ref, [x, res, w]))
    fresh = P.PrecisionRecorder()
    fresh.measure("fresh", [ref_out.clone()], [ref_out],
                  lambda: P.truth_of(ref, P.restore(pristine)))
    assert stale.rows[0]["status"] != "pass"
    assert fresh.rows[0]["status"] == "pass"
    # truth_of never writes into what it is given (the float32 retry reads it again)
    before = P.restore(pristine)
    keep = [t.clone() for t in before]
    P.truth_of(ref, before)
    assert all(torch.equal(a, b) for a, b in zip(before, keep))


def test_snapshot_restore_round_trip():
    a = torch.randn(3, dtype=torch.bfloat16, requires_grad=False)
    b = torch.arange(4)
    g = torch.randn(2, requires_grad=True)
    snap = P.snapshot([a, (b, 7), g, "x"])
    a.zero_(); b.zero_()
    ra, (rb, seven), rg, s = P.restore(snap)
    assert not torch.equal(ra, a) and torch.equal(rb, torch.arange(4))
    assert seven == 7 and s == "x" and rg.requires_grad and rg.dtype == g.dtype
