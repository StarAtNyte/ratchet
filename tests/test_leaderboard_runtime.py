"""Tests for the job-side leaderboard runtime (src/forgekernel/card_template/eval/leaderboard.py):
the two suites' judges, input draws, correctness protocols, the pointer-shifting pool,
the kernel-span extraction and the eval-JSON mapping. CPU only — the clocks' GPU paths
are exercised through a synthetic profiler trace.

Needs torch (skipped without it). Parity with the pinned upstream code itself was
checked offline against the upstream sources (see docs/leaderboards.md, "Fidelity");
these tests pin the rules so a later edit cannot drift from them silently.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fk_template_leaderboard", REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval" / "leaderboard.py")
lb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lb)

RMS_REF = """import torch

def run(hidden_states, weight):
    x = hidden_states.to(torch.float32)
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
"""
RMS_DEF = {
    "name": "rmsnorm_h64", "op_type": "rmsnorm",
    "axes": {"batch_size": {"type": "var"}, "hidden_size": {"type": "const", "value": 64}},
    "inputs": {"hidden_states": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"},
               "weight": {"shape": ["hidden_size"], "dtype": "float32"}},
    "outputs": {"output": {"shape": ["batch_size", "hidden_size"], "dtype": "float32"}},
    "reference": RMS_REF,
}


def _problem(suite: str, **wl) -> dict:
    workload = {"uuid": "w0", "axes": {"batch_size": 5},
                "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}}
    workload.update(wl)
    return {"suite": suite, "definition": json.loads(json.dumps(RMS_DEF)),
            "workload": workload, "dps": True, "_dir": "."}


def _rms(hidden_states, weight, output):
    x = hidden_states.float()
    output.copy_(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float())


# --- FlashInfer-Bench judge ------------------------------------------------------------

def test_fib_element_fails_only_when_both_bounds_fail() -> None:
    ref = torch.tensor([1.0, 0.0, 100.0, 0.001])
    out = torch.tensor([1.02, 0.005, 100.5, 0.02])
    # e0: abs .02 > .01 and rel .02 > .01 -> fail; e1: abs .005 ok; e2: rel .005 ok;
    # e3: abs .019 > .01 and rel 19 > .01 -> fail
    max_abs, max_rel, exceeds, matched = lb.fib_error_stats(out, ref, 1e-2, 1e-2)
    assert matched == pytest.approx(0.5) and exceeds
    assert max_abs == pytest.approx(0.5)
    assert lb.fib_error_stats(out, ref, 1e-2, 1e-2, 0.5)[2] is False


def test_fib_judge_fatal_and_numerical() -> None:
    cfg = dict(lb.FIB_DEFAULTS)
    ref = [torch.ones(4)]
    assert lb.fib_judge([torch.ones(5)], ref, cfg)["status"] == "INCORRECT_SHAPE"
    assert lb.fib_judge([torch.ones(4, dtype=torch.float64)], ref, cfg)["status"] == \
        "INCORRECT_DTYPE"
    bad = torch.ones(4)
    bad[1] = float("nan")
    r = lb.fib_judge([bad], ref, cfg)
    assert r["fatal"] and r["status"] == "INCORRECT_NUMERICAL" and "nan" in r["error"]
    off = torch.ones(4)
    off[0] = 1.5
    r = lb.fib_judge([off], ref, cfg)
    assert not r["fatal"] and r["status"] == "INCORRECT_NUMERICAL"
    assert "matched 0.750000 < 1.0" in r["error"]


def test_fib_config_matched_ratio_overrides() -> None:
    p = _problem(lb.SUITE_FIB)
    assert lb.fib_config(p)["required_matched_ratio"] is None
    p["definition"]["op_type"] = "moe"
    assert lb.fib_config(p)["required_matched_ratio"] == 0.95
    p["definition"]["op_type"] = "gemm"
    p["definition"]["name"] = "moe_fp8_block_scale_x"
    assert lb.fib_config(p)["required_matched_ratio"] == 0.95


# --- SOL-ExecBench judge -----------------------------------------------------------------

TOL = dict(lb.SOL_TOLERANCE_DEFAULTS)


def test_sol_allclose_bound_and_matched_ratio() -> None:
    ref = torch.linspace(-1, 1, 1000)
    out = ref.clone()
    out[:5] += 1.0                           # 0.5% of elements far off
    stats, exceeds = lb.sol_error_stats(out, ref, TOL)
    assert stats["matched_ratio"] == pytest.approx(0.995) and not exceeds
    stats, exceeds = lb.sol_error_stats(out, ref, dict(TOL, max_error_cap=0.5))
    assert exceeds                           # the cap catches the rare outliers
    out[:20] += 1.0
    assert lb.sol_error_stats(out, ref, TOL)[1]


def test_sol_non_finite_in_either_tensor_fails() -> None:
    ref = torch.zeros(8) + 1
    out = ref.clone()
    ref_inf = ref.clone()
    ref_inf[0] = float("inf")
    assert lb.sol_error_stats(out, ref_inf, TOL)[1]            # reference inf fails too
    both = ref.clone()
    both[0] = float("-inf")
    assert lb.sol_error_stats(both.clone(), both, TOL)[1]     # matching -inf, no flag
    assert not lb.sol_error_stats(both.clone(), both, dict(TOL, allow_negative_inf=True))[1]


def test_sol_all_zero_solution_against_nonzero_reference_fails() -> None:
    stats, exceeds = lb.sol_error_stats(torch.zeros(4), torch.full((4,), 1e-4), TOL)
    assert exceeds and "all-zero" in stats["why"]
    assert not lb.sol_error_stats(torch.zeros(4), torch.zeros(4), TOL)[1]


# --- definitions and inputs -----------------------------------------------------------------

def test_expr_axes_and_digit_dims() -> None:
    defn = {"axes": {"b": {"type": "var"}, "h": {"type": "const", "value": 8},
                     "t": {"type": "expr", "expression": "b * h // 2"}}}
    axes = lb.resolved_axes(defn, {"b": 3})
    assert axes == {"h": 8, "b": 3, "t": 12}
    assert lb.tensor_shapes({"x": {"shape": ["t", "4"]}, "s": {"shape": None}}, axes) == \
        {"x": (12, 4), "s": None}
    with pytest.raises(ValueError, match="not an integer"):
        lb.resolve_shape_expression("b / 2", {"b": 3})
    with pytest.raises(TypeError):
        lb.resolve_shape_expression("abs(b)", {"b": 3})


def test_fib_inputs_types_and_sampling_softmax() -> None:
    p = _problem(lb.SUITE_FIB)
    p["definition"]["inputs"]["idx"] = {"shape": ["batch_size"], "dtype": "int32"}
    p["definition"]["inputs"]["scale"] = {"shape": None, "dtype": "float32"}
    p["workload"]["inputs"]["scale"] = {"type": "scalar", "value": 0.5}
    x, w, idx, scale = lb.gen_inputs(p, "cpu")
    assert x.shape == (5, 64) and w.shape == (64,) and idx.dtype == torch.int32
    assert int(idx.min()) >= -1024 and int(idx.max()) < 1024 and scale == 0.5
    s = _problem(lb.SUITE_FIB)
    s["definition"]["op_type"] = "sampling"
    s["definition"]["inputs"] = {"probs": {"shape": ["batch_size", "hidden_size"],
                                           "dtype": "float32"}}
    s["workload"]["inputs"] = {}
    (probs,) = lb.gen_inputs(s, "cpu")
    assert torch.allclose(probs.sum(-1), torch.ones(5))


def test_sol_name_heuristics() -> None:
    p = _problem(lb.SUITE_SOL)
    p["definition"]["inputs"] = {
        "input_layernorm_weight": {"shape": ["hidden_size"], "dtype": "float32"},
        "norm_bias": {"shape": ["hidden_size"], "dtype": "float32"},
        "attention_mask": {"shape": ["batch_size", "batch_size"], "dtype": "float32",
                           "description": "causal attention mask"},
        "rstd": {"shape": ["batch_size"], "dtype": "float32"},
        "up_proj": {"shape": ["hidden_size", "hidden_size"], "dtype": "float32"},
        "A_log": {"shape": ["hidden_size"], "dtype": "float32"},
    }
    p["workload"]["inputs"] = {}
    torch.manual_seed(0)
    nw, nb, mask, rstd, proj, a_log = lb.gen_inputs(p, "cpu")
    assert torch.equal(nw, torch.ones(64)) and torch.equal(nb, torch.zeros(64))
    assert mask[0, 1] == torch.finfo(torch.float32).min and mask[1, 0] == 0
    assert float(rstd.min()) >= 0.1
    assert float(proj.std()) < 0.3           # randn / sqrt(fan_in=64) ~ 0.125
    assert float(a_log.max()) <= -1.0 and float(a_log.min()) >= -5.0


def test_sol_custom_inputs_entrypoint_gets_axes_and_scalars() -> None:
    p = _problem(lb.SUITE_SOL)
    p["definition"]["inputs"]["topk"] = {"shape": None, "dtype": "int32"}
    p["definition"]["inputs"]["router"] = {"shape": ["batch_size"], "dtype": "int32"}
    p["workload"]["inputs"]["topk"] = {"type": "scalar", "value": 3}
    p["workload"]["inputs"]["router"] = {"type": "custom"}
    seen = {}

    def custom(axes_and_scalars, device):
        seen.update(axes_and_scalars)
        return {"router": torch.arange(axes_and_scalars["batch_size"], dtype=torch.int32)}

    out = lb.gen_inputs(p, "cpu", custom_inputs_fn=custom)
    assert seen == {"hidden_size": 64, "batch_size": 5, "topk": 3}
    assert torch.equal(out[3], torch.arange(5, dtype=torch.int32))


def test_safetensors_input_is_shape_and_dtype_checked(tmp_path: Path) -> None:
    st = pytest.importorskip("safetensors.torch")
    st.save_file({"w": torch.ones(64)}, str(tmp_path / "b.safetensors"))
    p = _problem(lb.SUITE_FIB, inputs={
        "hidden_states": {"type": "random"},
        "weight": {"type": "safetensors", "path": "b.safetensors", "tensor_key": "w"}})
    p["_dir"] = str(tmp_path)
    assert torch.equal(lb.gen_inputs(p, "cpu")[1], torch.ones(64))
    p["definition"]["inputs"]["weight"]["dtype"] = "bfloat16"
    lb._SAFETENSORS_CACHE.clear()
    with pytest.raises(ValueError, match="expected torch.bfloat16"):
        lb.gen_inputs(p, "cpu")


# --- correctness protocols ------------------------------------------------------------------

def test_fib_protocol_passes_a_correct_kernel_on_every_set() -> None:
    verdict, sets = lb.fib_correctness(_problem(lb.SUITE_FIB), _rms, True, "cpu")
    assert verdict["pass"] and verdict["status"] == "PASSED"
    assert len(sets) == 3 and len(verdict["checks"]) == 3


def test_fib_protocol_fails_a_wrong_kernel_with_the_boards_reason() -> None:
    def off(hidden_states, weight, output):
        _rms(hidden_states, weight, output)
        output[0, 0] += 1.0

    verdict, _ = lb.fib_correctness(_problem(lb.SUITE_FIB), off, True, "cpu")
    assert not verdict["pass"] and verdict["status"] == "INCORRECT_NUMERICAL"
    assert verdict["error"].startswith("set 0:")


def test_fib_protocol_runtime_error_is_reported() -> None:
    def boom(*a):
        raise RuntimeError("bad launch")

    verdict, _ = lb.fib_correctness(_problem(lb.SUITE_FIB), boom, True, "cpu")
    assert verdict["status"] == "RUNTIME_ERROR" and "bad launch" in verdict["error"]
    assert not verdict["pass"]


def test_sol_protocol_ten_seeded_rounds_and_timed_set() -> None:
    verdict, timed = lb.sol_correctness(_problem(lb.SUITE_SOL), _rms, True, "cpu")
    assert verdict["pass"] and len(verdict["checks"]) == lb.SOL_ROUNDS
    assert verdict["threads_after_first_call"] is not None and len(timed) == 1
    # same seed -> the timed set is reproducible run to run
    _, again = lb.sol_correctness(_problem(lb.SUITE_SOL), _rms, True, "cpu")
    assert torch.equal(timed[0][0], again[0][0])


def test_sol_protocol_uses_the_workload_tolerance() -> None:
    def slightly_off(hidden_states, weight, output):
        _rms(hidden_states, weight, output)
        output.mul_(1.001)

    loose = _problem(lb.SUITE_SOL)
    assert lb.sol_correctness(loose, slightly_off, True, "cpu")[0]["pass"]
    tight = _problem(lb.SUITE_SOL, tolerance={"max_atol": 1e-6, "max_rtol": 1e-6})
    verdict, _ = lb.sol_correctness(tight, slightly_off, True, "cpu")
    assert verdict["status"] == "INCORRECT_NUMERICAL" and "round 0" in verdict["error"]


def test_sol_protocol_rejects_lazy_outputs() -> None:
    class Lazy(torch.Tensor):
        pass

    def lazy(hidden_states, weight):
        x = hidden_states.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float()
        return y.as_subclass(Lazy)

    p = _problem(lb.SUITE_SOL)
    verdict, _ = lb.sol_correctness(p, lazy, False, "cpu")
    assert verdict["status"] == "REWARD_HACK" and "Lazy" in verdict["error"]


# --- clock pieces ----------------------------------------------------------------------------

def test_shifted_pool_gives_every_call_a_fresh_pointer() -> None:
    x = torch.arange(10, dtype=torch.float32)
    pool = lb._Shifted([x, 3], total_calls=4, seed=200)
    calls = [pool.next() for _ in range(4)]
    ptrs = {c[0].data_ptr() for c in calls}
    assert len(ptrs) == 4 and all(torch.equal(c[0], x) and c[1] == 3 for c in calls)
    assert all((p - calls[0][0].data_ptr()) % lb.SOL_POOL_ALIGNMENT == 0 for p in ptrs)


def test_profiler_spans_take_the_calls_own_kernel_sequence(monkeypatch) -> None:
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a, **k: None)
    traces = [
        [("gemm", 0, 5), ("epi", 5, 7)],                     # discovery: the call alone
    ]
    timed = []
    t = 100
    for i in range(3):                                      # flush, gemm, epi per call
        timed += [("fill", t, t + 50), ("gemm", t + 60, t + 70 + i), ("epi", t + 71 + i, t + 80)]
        t += 1000
    traces.append(timed)
    monkeypatch.setattr(lb, "_profile", lambda fn, *a: (fn(), traces.pop(0))[1])
    spans = lb.time_profiler_spans(lambda a: None, lambda: [], lambda: None,
                                   warmup=0, iters=3)
    assert spans == [pytest.approx(20 / 1e6)] * 3


def test_profiler_spans_survive_an_extra_launch(monkeypatch) -> None:
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a, **k: None)
    timed = [("fill", 0, 1), ("gemm", 2, 12), ("fill", 20, 21), ("stray", 22, 23),
             ("gemm", 24, 30)]
    traces = [[("gemm", 0, 10)], timed]
    monkeypatch.setattr(lb, "_profile", lambda fn, *a: (fn(), traces.pop(0))[1])
    spans = lb.time_profiler_spans(lambda a: None, lambda: [], lambda: None,
                                   warmup=0, iters=2)
    assert spans == [pytest.approx(10 / 1e6), pytest.approx(6 / 1e6)]


def test_profiler_spans_count_memory_activities_only_for_sol(monkeypatch) -> None:
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a, **k: None)
    seen = []

    def fake(fn, include_memory=False):
        seen.append(include_memory)
        fn()
        return [("gemm", 0, 10)]

    monkeypatch.setattr(lb, "_profile", fake)
    monkeypatch.setattr(lb, "_flush_buffer", lambda device: torch.empty(1))
    monkeypatch.setattr(lb, "_reset_persisting_l2", lambda: None)
    for suite, want in ((lb.SUITE_FIB, False), (lb.SUITE_SOL, True)):
        seen.clear()
        lb._time_once(suite, "profiler", lambda *a: None, [torch.ones(2)], [], "cpu",
                      warmup=0, iters=1, seed=0)
        assert seen == [want, want]


def test_choose_timer_respects_the_override(monkeypatch) -> None:
    monkeypatch.setenv("FK_LB_TIMER", "events")
    assert lb.choose_timer(lb.SUITE_FIB) == ("events", lb.BASIS_COLD_EVENTS)
    monkeypatch.setenv("FK_LB_TIMER", "profiler")
    assert lb.choose_timer(lb.SUITE_SOL) == ("profiler", lb.BASIS_CUPTI)
    monkeypatch.setenv("FK_LB_TIMER", "warm")
    with pytest.raises(ValueError, match="FK_LB_TIMER"):
        lb.choose_timer(lb.SUITE_FIB)


def test_choose_timer_prefers_cupti_python_over_the_profiler(monkeypatch) -> None:
    monkeypatch.delenv("FK_LB_TIMER", raising=False)
    monkeypatch.setattr(lb, "_upstream_timer", lambda suite: None)
    monkeypatch.setattr(lb, "_cupti_python", lambda: True)
    assert lb.choose_timer(lb.SUITE_SOL) == ("cupti", lb.BASIS_CUPTI)
    monkeypatch.setattr(lb, "_cupti_python", lambda: False)
    assert lb.choose_timer(lb.SUITE_SOL) == ("profiler", lb.BASIS_CUPTI)
    monkeypatch.setenv("FK_LB_TIMER", "cupti")
    with pytest.raises(RuntimeError, match="cupti-python"):
        lb.choose_timer(lb.SUITE_FIB)


class _FakeCupti:
    """cupti-python's surface as time_cupti_spans uses it: activities are delivered
    at flush time, timestamps come from a shared clock the fake 'GPU' advances."""

    class ActivityKind:
        CONCURRENT_KERNEL, MEMCPY, MEMSET, RUNTIME, DRIVER = "K", "C", "S", "R", "D"

    class _Act:
        def __init__(self, kind, name, start, end):
            self.kind, self.name, self.start, self.end = kind, name, start, end
            self.bytes, self.copy_kind, self.value = 64, 8, 0

    def __init__(self):
        self.now = 1000
        self.pending: list = []
        self.enabled: set = set()
        self.complete = None

    def activity_enable(self, kind):
        self.enabled.add(kind)

    def activity_disable(self, kind):
        self.enabled.discard(kind)

    def activity_register_callbacks(self, request, complete):
        self.complete = complete

    def activity_flush_all(self, flag):
        self.complete([a for a in self.pending if a.kind in self.enabled])
        self.pending = []

    def finalize(self):
        pass

    def get_timestamp(self):
        return self.now

    def launch(self, kind, name, dur):
        if kind in self.enabled:               # CUPTI records only while enabled
            self.pending.append(self._Act(kind, name, self.now + 5, self.now + 5 + dur))
        self.now += 5 + dur


def test_cupti_spans_select_the_calls_sequence_inside_each_window(monkeypatch) -> None:
    fake = _FakeCupti()
    mod = type(sys)("cupti")
    mod.cupti = fake
    monkeypatch.setitem(sys.modules, "cupti", mod)
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a, **k: None)
    K, C = fake.ActivityKind.CONCURRENT_KERNEL, fake.ActivityKind.MEMCPY
    durs = iter([100, 200, 300, 400, 500, 600, 700, 800])

    def call(_args):
        fake.launch(K, "gemm", 10)
        fake.launch(C, "", 7)                  # a memcpy inside the call
        fake.launch(K, "epilogue", next(durs))

    def flush():
        fake.launch(K, "fill", 50)             # lands in the window: skipped by identity

    spans = lb.time_cupti_spans(call, lambda: [], flush, warmup=1, iters=3,
                                include_memory=True)
    # warm-up consumed 100, discovery 200; each span = gemm start -> epilogue end
    assert spans == [pytest.approx((10 + 5 + 7 + 5 + d) / 1e6) for d in (300, 400, 500)]
    fib = lb.time_cupti_spans(call, lambda: [], flush, warmup=0, iters=1)
    assert fib == [pytest.approx((10 + 5 + 7 + 5 + 700) / 1e6)]  # memcpy not collected
    # flashinfer's way: API tracing switched on while timing, still kernels-only spans
    seen = []
    orig = fake.activity_enable
    fake.activity_enable = lambda k: (seen.append(k), orig(k))
    durs = iter([10, 20])
    fi = lb.time_cupti_spans(call, lambda: [], flush, warmup=0, iters=1,
                             sync_after_flush=True, trace_api=True)
    assert fi == [pytest.approx((10 + 5 + 7 + 5 + 20) / 1e6)]
    assert {"R", "D"} <= set(seen)


# --- eval JSON ---------------------------------------------------------------------------------

def _m(basis: str, ref_ms=0.02) -> dict:
    return {"suite": lb.SUITE_FIB, "timer": "profiler", "basis": basis,
            "protocol": {"warmup": 10, "iters": 50, "aggregate": "mean", "seed": 0},
            "kernel_ms": 0.01, "kernel_per_set_ms": [0.01, 0.01, 0.01],
            "ref_ms": ref_ms, "ref_per_set_ms": [ref_ms] * 3 if ref_ms else [],
            "ref_error": None if ref_ms else "ImportError: flashinfer"}


def test_performance_block_maps_the_board_reading_to_the_gate_keys() -> None:
    p = _problem(lb.SUITE_FIB)
    kwin = {"median_ms": 0.03, "min_ms": 0.028, "host_enqueue_ms": 0.004}
    rwin = {"median_ms": 0.05, "min_ms": 0.049, "host_enqueue_ms": 0.02}
    perf = lb.performance_block(p, _m(lb.BASIS_CUPTI), kwin, rwin)
    assert perf["device_median_ms"] == 0.01 and perf["ref_device_ms"] == 0.02
    assert perf["vs_ref_device"] == 2.0 and perf["vs_ref"] == pytest.approx(1.6667, abs=1e-4)
    assert perf["device_window_basis"] == lb.BASIS_CUPTI
    assert perf["host_bound_ratio"] is None             # host-free basis: no ratio
    assert perf["leaderboard"]["kernel_per_set_ms"] == [0.01, 0.01, 0.01]
    ev = lb.performance_block(p, _m(lb.BASIS_COLD_EVENTS), kwin, rwin)
    assert ev["host_bound_ratio"] == pytest.approx(0.4)


def test_unstable_mirrored_pairs_are_flagged() -> None:
    assert lb.unstable_sets([[0.044, 0.045], [0.044, 0.111], [0.0035, 0.0036]]) == [1]
    m = _m(lb.BASIS_CUPTI)
    m["unstable"] = {"kernel": [1], "ref": []}
    perf = lb.performance_block(_problem(lb.SUITE_SOL), m,
                                {"median_ms": 0.03, "min_ms": 0.03, "host_enqueue_ms": 0.004},
                                {"median_ms": 0.05, "min_ms": 0.05, "host_enqueue_ms": 0.004})
    assert perf["leaderboard"]["unstable_sets"] == {"kernel": [1]}
    assert "not a kernel fact" in perf["leaderboard"]["unstable_note"]
    clean = lb.performance_block(_problem(lb.SUITE_SOL), _m(lb.BASIS_CUPTI),
                                 {"median_ms": 0.03, "min_ms": 0.03, "host_enqueue_ms": 0.004},
                                 None)
    assert "unstable_sets" not in clean["leaderboard"]


def test_performance_block_without_reference_emits_no_ref_keys() -> None:
    perf = lb.performance_block(_problem(lb.SUITE_FIB), _m(lb.BASIS_CUPTI, ref_ms=None),
                                {"median_ms": 0.03, "min_ms": 0.03, "host_enqueue_ms": 0.004},
                                None)
    assert not any(k.startswith(("ref_", "vs_")) for k in perf)
    assert perf["in_job_ref"].startswith("ImportError")
    assert math.isfinite(perf["device_median_ms"])


# --- SOL board score, clock state, untimed reference ---------------------------------------

def _sol_board_problem() -> dict:
    p = _problem(lb.SUITE_SOL)
    p["time_reference"] = False
    p["sol_board"] = {"kernel_id": 7, "workload_index": 3, "t_b_ms": 0.04, "t_sol_ms": 0.004,
                      "clock_mhz": {"sm_mhz": 1500, "mem_mhz": 3996}}
    return p


def test_sol_score_is_the_boards_formula() -> None:
    assert lb.sol_score(0.04, 0.04, 0.004) == 0.5
    assert lb.sol_score(0.004, 0.04, 0.004) == 1.0
    assert lb.sol_score(0.013, 0.04, 0.004) == pytest.approx(0.8)
    assert lb.sol_score(0.01, 0.004, 0.04) == 1.0                  # T_b not above T_SOL


def test_sol_board_block_scores_the_reading_and_flags_an_unlocked_clock(monkeypatch) -> None:
    p = _sol_board_problem()
    m = {**_m(lb.BASIS_CUPTI, ref_ms=None), "ref_error": None, "suite": lb.SUITE_SOL}
    kwin = {"median_ms": 0.03, "min_ms": 0.03, "host_enqueue_ms": 0.004}
    monkeypatch.setenv(lb.CLOCK_ENV, "locked sm=1500 mem=3996 (read 1500,3996)")
    perf = lb.performance_block(p, m, kwin, None)
    board = perf["leaderboard"]["sol_board"]
    assert board["sol_score"] == pytest.approx(1 / (1 + 0.006 / 0.036), abs=1e-4)
    assert board["vs_t_b"] == 4.0 and "clock_warning" not in board
    assert perf["leaderboard"]["clock"].startswith("locked")
    assert perf["in_job_ref"].startswith("not timed")
    monkeypatch.setenv(lb.CLOCK_ENV, "lock refused (nvidia-smi -lgc / -lmc failed)")
    warned = lb.performance_block(p, m, kwin, None)["leaderboard"]["sol_board"]
    assert "two clocks" in warned["clock_warning"]
    monkeypatch.delenv(lb.CLOCK_ENV)
    assert lb.performance_block(p, m, kwin, None)["leaderboard"]["clock"].startswith("unknown")
    fib = lb.performance_block(_problem(lb.SUITE_FIB), _m(lb.BASIS_CUPTI), kwin, kwin)
    assert "sol_board" not in fib["leaderboard"]


def test_bench_times_the_reference_only_when_the_card_says_so(tmp_path: Path, monkeypatch,
                                                               capsys) -> None:
    card = tmp_path / "card"
    (card / "eval" / "leaderboard").mkdir(parents=True)
    (card / "kernel.py").write_text("def run(hidden_states, weight, output):\n"
                                    "    output.copy_(hidden_states)\n")
    path = card / "eval" / "leaderboard" / "problem.json"
    seen = []

    def fake_measure(problem, sets, kernel, kernel_dps, ref=None, ref_dps=False,
                     device="cuda", **kw):
        seen.append(ref)
        m = _m(lb.BASIS_CUPTI, ref_ms=0.02 if ref is not None else None)
        return {**m, "ref_error": None, "suite": problem["suite"]}

    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: (
        {"pass": True, "status": "PASSED", "threads_after_first_call": None}, [[]]))
    monkeypatch.setattr(lb, "measure", fake_measure)
    monkeypatch.setattr(lb, "_side_windows", lambda *a, **k: {
        "median_ms": 0.03, "min_ms": 0.03, "host_enqueue_ms": 0.004})
    for time_ref in (False, True):
        p = _sol_board_problem()
        p.pop("_dir")
        p["time_reference"] = time_ref
        path.write_text(json.dumps(p))
        assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
        out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
        assert (seen[-1] is not None) is time_ref
        assert ("ref_device_ms" in out["performance"]) is time_ref
        assert out["performance"]["leaderboard"]["sol_board"]["t_b_ms"] == 0.04


# --- workload groups ----------------------------------------------------------------------

def _group_problem(suite: str = lb.SUITE_SOL, sizes=(3, 17, 64), boards: bool = True) -> dict:
    p = _problem(suite)
    p["group"] = []
    for i, b in enumerate(sizes):
        wl = {"uuid": f"w{i}", "axes": {"batch_size": b},
              "inputs": {"hidden_states": {"type": "random"}, "weight": {"type": "random"}}}
        g = {"index": 10 + i, "name": f"wl{10 + i}", "workload": wl}
        if boards:
            g["sol_board"] = {"kernel_id": 7, "workload_index": 10 + i, "t_b_ms": 0.04 * (i + 1),
                              "t_sol_ms": 0.004 * (i + 1),
                              "clock_mhz": {"sm_mhz": 1500, "mem_mhz": 3996}}
        p["group"].append(g)
    p["workload"] = p["group"][0]["workload"]
    if boards:
        p["sol_board"] = p["group"][0]["sol_board"]
    return p


def test_workload_views_give_each_workload_its_own_shape_and_board() -> None:
    p = _group_problem()
    views = lb.workload_views(p)
    assert [v["_wl_name"] for v in views] == ["wl10", "wl11", "wl12"]
    assert [v["workload"]["axes"]["batch_size"] for v in views] == [3, 17, 64]
    assert [v["sol_board"]["t_b_ms"] for v in views] == pytest.approx([0.04, 0.08, 0.12])
    assert [tuple(lb.allocate_outputs(v, "cpu")[0].shape) for v in views] == \
        [(3, 64), (17, 64), (64, 64)]
    assert lb.workload_views(_problem(lb.SUITE_SOL)) == [_problem(lb.SUITE_SOL)]
    assert "sol_board" not in lb.workload_views(_group_problem(boards=False))[1]


def test_group_judge_passes_only_when_every_workload_passes() -> None:
    p = _group_problem()
    verdict, runs = lb.judge_workloads(p, _rms, True, "cpu")
    assert verdict["pass"] and verdict["n_workloads"] == 3
    assert [w["name"] for w in verdict["workloads"]] == ["wl10", "wl11", "wl12"]
    assert [tuple(sets[0][0].shape) for _, sets in runs] == [(3, 64), (17, 64), (64, 64)]

    def wrong_on_big(hidden_states, weight, output):
        _rms(hidden_states, weight, output)
        if hidden_states.shape[0] > 10:
            output.mul_(2.0)
    bad, runs = lb.judge_workloads(p, wrong_on_big, True, "cpu")
    assert not bad["pass"] and bad["status"] == "INCORRECT_NUMERICAL"
    assert bad["error"].startswith('workload wl11 {"batch_size": 17}:')
    assert "1 later workload(s) not run" in bad["error"] and len(runs) == 2


def _fake_measure(seen):
    def measure(problem, sets, kernel, kernel_dps, ref=None, ref_dps=False, device="cuda", **kw):
        b = problem["workload"]["axes"]["batch_size"]
        seen.append(b)
        k = 0.001 * b
        r = 2.0 * k if ref is not None else None
        return {"suite": problem["suite"], "timer": "cupti", "basis": lb.BASIS_CUPTI,
                "protocol": {"warmup": 10, "iters": 50, "aggregate": "single", "seed": 200},
                "kernel_ms": k, "kernel_per_set_ms": [k], "kernel_readings_ms": [[k, k]],
                "ref_ms": r, "ref_per_set_ms": [r] if r else [],
                "ref_readings_ms": [[r, r]] if r else [],
                "ref_error": None, "unstable": {"kernel": [], "ref": []}}
    return measure


def _windows(fn, dps, problem, inp, device):
    return {"median_ms": 0.003 * problem["workload"]["axes"]["batch_size"], "min_ms": 0.001,
            "host_enqueue_ms": 0.004}


def _group_card(tmp_path: Path, p: dict) -> Path:
    path = tmp_path / "card" / "eval" / "leaderboard" / "problem.json"
    path.parent.mkdir(parents=True)
    (tmp_path / "card" / "kernel.py").write_text("def run(hidden_states, weight, output):\n"
                                                 "    output.copy_(hidden_states)\n")
    p.pop("_dir", None)
    path.write_text(json.dumps(p))
    return path


def test_group_bench_gates_on_the_geomean_and_keeps_every_workload(tmp_path: Path, monkeypatch,
                                                                   capsys) -> None:
    from forgekernel.loop.parse import parse_eval
    seen: list = []
    monkeypatch.setattr(lb, "measure", _fake_measure(seen))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (_rms, True) if which == "kernel"
                        else (lambda hidden_states, weight: hidden_states, False))
    monkeypatch.setattr(lb, "warm_reference", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: lb.sol_correctness(
        problem, impl, dps, "cpu"))
    p = _group_problem()
    p["time_reference"] = True
    p["reference_impl"] = {"kind": "binary", "name": "sealed"}
    path = _group_card(tmp_path, p)
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    text = capsys.readouterr().out
    out = json.loads(text.split("=== JSON ===")[1])
    perf = out["performance"]
    assert out["correctness"]["pass"] and seen == [3, 17, 64]
    geo = math.exp(sum(math.log(0.001 * b) for b in (3, 17, 64)) / 3)
    assert perf["device_median_ms"] == pytest.approx(geo, abs=1e-4)
    assert perf["vs_ref_device"] == pytest.approx(2.0) and perf["in_job_ref"] == "paired"
    assert perf["aggregate"] == "geomean" and perf["n_workloads"] == 3
    assert [w["name"] for w in perf["workloads"]] == ["wl10", "wl11", "wl12"]
    board = perf["leaderboard"]["sol_board"]
    assert set(board["sol_score"]) == {"wl10", "wl11", "wl12"}
    assert board["sol_score_mean"] == pytest.approx(sum(board["sol_score"].values()) / 3, abs=1e-3)
    # the driver reads the geomean as the gate and every workload for its log
    ev = parse_eval(text)
    assert ev.correctness and ev.n_workloads == 3
    assert ev.vs_ref_device == pytest.approx(2.0)
    assert [w["name"] for w in ev.workloads] == ["wl10", "wl11", "wl12"]
    # the gate ratio is the first one in the log (a card-reference check reads the first match)
    import re
    first = re.search(r'"vs_ref_device":\s*([0-9.]+)', text)
    assert float(first.group(1)) == perf["vs_ref_device"]


def test_group_ref_mode_prints_one_geomean_block_the_archive_reads(tmp_path: Path, monkeypatch,
                                                                   capsys) -> None:
    from forgekernel.baseline_cli import parse_ref_stdout
    seen: list = []
    monkeypatch.setattr(lb, "measure", _fake_measure(seen))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: (
        {"pass": True, "status": "PASSED"}, [[]]))
    path = _group_card(tmp_path, _group_problem())
    assert lb.run_leaderboard_bench(["--mode", "ref"], problem_path=path) == 0
    got = parse_ref_stdout(capsys.readouterr().out)
    assert got["n_workloads"] == 3 and got["workloads"] == ["wl10", "wl11", "wl12"]
    geo = math.exp(sum(math.log(0.001 * b) for b in (3, 17, 64)) / 3)
    assert got["ms_device"] == pytest.approx(geo, abs=1e-4)      # rounded to 4 digits
    assert seen == [3, 17, 64]


SOURCE_AT_CALL_REF = """import inspect


def _body(hidden_states, weight, output):
    output.copy_(hidden_states)


def run(hidden_states, weight, output):
    inspect.getsource(_body)   # a CuTe DSL JIT reads its source at the first call
    _body(hidden_states, weight, output)
"""


def test_group_ref_mode_times_a_sealed_reference_through_the_callable_it_loaded(
        tmp_path: Path, monkeypatch, capsys) -> None:
    # A sealed source reference serves its source only inside the windows of the
    # closed_ref instance that imported it; loading it again per workload gave a
    # callable whose window does not open that loader, so a reference that reads its
    # source at call time (CuTe DSL) failed on every workload of `--mode ref`.
    spec = importlib.util.spec_from_file_location(
        "_tmpl_closed_ref_group", REPO_ROOT / "src" / "forgekernel" / "card_template" / "eval" / "closed_ref.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    module = "_fk_test_sealed_group_ref"
    monkeypatch.delitem(sys.modules, module, raising=False)
    p = _group_problem()
    p["reference_impl"] = {"kind": "sealed", "name": "sealed", "path": "ref_bin/ref_payload.bin",
                           "module": module, "entry": "run", "dps": True}
    path = _group_card(tmp_path, p)
    (path.parent / "ref_bin").mkdir()
    (path.parent / "ref_bin" / "ref_payload.bin").write_bytes(cr.seal(SOURCE_AT_CALL_REF))
    called: list = []

    def measure(problem, sets, kernel, kernel_dps, ref=None, ref_dps=False, device="cuda", **kw):
        b = problem["workload"]["axes"]["batch_size"]
        x = torch.randn(b, 64)
        out = torch.empty(b, 64)
        kernel(x, torch.ones(64), out)
        assert torch.equal(out, x)
        called.append(b)
        return _fake_measure([])(problem, sets, kernel, kernel_dps, ref, ref_dps, device)
    monkeypatch.setattr(lb, "measure", measure)
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: (
        {"pass": True, "status": "PASSED"}, [[]]))
    try:
        assert lb.run_leaderboard_bench(["--mode", "ref"], problem_path=path) == 0
    finally:
        sys.modules.pop(module, None)
    assert called == [3, 17, 64]
    assert '"aggregate": "geomean"' in capsys.readouterr().out


def test_group_thread_injection_is_checked_per_workload(tmp_path: Path, monkeypatch,
                                                        capsys) -> None:
    import threading
    import time as _time
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (_rms, True))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    judged = []

    def corr(problem, impl, dps, device):
        judged.append(problem["_wl_name"])
        return {"pass": True, "status": "PASSED",
                "threads_after_first_call": threading.active_count()}, [[]]

    def measure(problem, sets, *a, **k):
        if problem["_wl_name"] == "wl11":  # starts a thread while it is timed
            threading.Thread(target=_time.sleep, args=(0.5,), daemon=True).start()
        return _fake_measure([])(problem, sets, *a, **k)
    monkeypatch.setattr(lb, "_correctness", corr)
    monkeypatch.setattr(lb, "measure", measure)
    p = _group_problem()
    p["time_reference"] = False
    path = _group_card(tmp_path, p)
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    out = json.loads(capsys.readouterr().out.split("=== JSON ===")[1])
    assert out["correctness"]["status"] == "REWARD_HACK" and out["performance"] is None
    assert "workload wl11" in out["correctness"]["error"] and judged == ["wl10", "wl11"]


def test_group_card_scoped_to_its_representative_times_one_workload_and_is_no_gate(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """FK_LB_WORKLOAD_SCOPE=representative (the seeded meta tools' setting; lever
    pricing used to time every workload x rounds x variants and outlived the job clock): only the
    first group entry is judged and timed, the JSON says so, and the loop's gate parser
    refuses it. A single-workload card ignores the variable."""
    from forgekernel.loop.parse import parse_eval
    seen: list = []
    monkeypatch.setattr(lb, "measure", _fake_measure(seen))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (_rms, True) if which == "kernel"
                        else (lambda hidden_states, weight: hidden_states, False))
    monkeypatch.setattr(lb, "warm_reference", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: lb.sol_correctness(
        problem, impl, dps, "cpu"))
    p = _group_problem()
    p["time_reference"] = True
    p["reference_impl"] = {"kind": "binary", "name": "sealed"}
    path = _group_card(tmp_path, p)
    monkeypatch.setenv(lb.WORKLOAD_SCOPE_ENV, "representative")
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    text = capsys.readouterr().out
    out = json.loads(text.split("=== JSON ===")[1])
    assert seen == [3]                                     # the representative only
    assert out["correctness"]["pass"]
    assert out["workload_scope"] == {"workloads": "representative", "rep": "wl10", "of": 3}
    assert out["performance"]["device_median_ms"] == pytest.approx(0.003, abs=1e-6)
    assert out["performance"]["leaderboard"]["sol_board"]["t_b_ms"] == pytest.approx(0.04)
    assert "WORKLOAD SCOPE: representative workload wl10 only (1 of 3" in text
    ev = parse_eval(text)
    assert not ev.correctness and "workload_scope" in ev.error and "not a gate" in ev.error

    # the same card without the variable is timed whole, no scope tag, a gate as before
    seen.clear()
    monkeypatch.delenv(lb.WORKLOAD_SCOPE_ENV)
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    text = capsys.readouterr().out
    out = json.loads(text.split("=== JSON ===")[1])
    assert seen == [3, 17, 64] and "workload_scope" not in out and parse_eval(text).correctness


def test_workload_scope_leaves_a_single_workload_card_alone() -> None:
    p = _problem(lb.SUITE_SOL)
    before = json.loads(json.dumps(p))
    assert lb.apply_workload_scope(p, env={lb.WORKLOAD_SCOPE_ENV: "representative"}) is None
    assert p == before
    g = _group_problem()
    assert lb.apply_workload_scope(g, env={}) is None and len(g["group"]) == 3
    assert lb.apply_workload_scope(g, env={lb.WORKLOAD_SCOPE_ENV: "all"}) is None


def test_workload_scope_names_one_workload_of_the_group() -> None:
    """The meta tools read each of meta_workloads in turn: a group entry's name narrows
    the card to it, board row included; a name the group lacks raises, never times the group."""
    g = _group_problem()
    rec = lb.apply_workload_scope(g, env={lb.WORKLOAD_SCOPE_ENV: "wl12"})
    assert rec == {"workloads": "wl12", "rep": "wl12", "of": 3}
    assert [x["name"] for x in g["group"]] == ["wl12"] and g["workload"]["axes"]["batch_size"] == 64
    assert g["sol_board"]["t_b_ms"] == pytest.approx(0.12)
    with pytest.raises(ValueError, match="names no workload"):
        lb.apply_workload_scope(_group_problem(), env={lb.WORKLOAD_SCOPE_ENV: "wl99"})


def test_named_scope_times_that_workload_and_is_no_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    from forgekernel.loop.parse import parse_eval
    seen: list = []
    monkeypatch.setattr(lb, "measure", _fake_measure(seen))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (_rms, True) if which == "kernel"
                        else (lambda hidden_states, weight: hidden_states, False))
    monkeypatch.setattr(lb, "warm_reference", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: lb.sol_correctness(
        problem, impl, dps, "cpu"))
    p = _group_problem()
    p["time_reference"] = True
    p["reference_impl"] = {"kind": "binary", "name": "sealed"}
    path = _group_card(tmp_path, p)
    monkeypatch.setenv(lb.WORKLOAD_SCOPE_ENV, "wl11")
    assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
    text = capsys.readouterr().out
    out = json.loads(text.split("=== JSON ===")[1])
    assert seen == [17] and out["workload_scope"] == {"workloads": "wl11", "rep": "wl11", "of": 3}
    assert "WORKLOAD SCOPE: workload wl11 only (1 of 3" in text
    assert not parse_eval(text).correctness


def test_ncu_window_profiles_the_scoped_workload(tmp_path: Path, monkeypatch, capsys) -> None:
    """eval/ncu_run.py follows FK_LB_WORKLOAD_SCOPE, so a meta tool's NCU (stall pricing, the
    role pass, ablation timing, the timeline launcher) reads the workload its bench timed;
    unset, a group card profiles its representative as before."""
    built: list = []
    monkeypatch.setattr(lb, "kernel_fence", lambda problem: None)
    monkeypatch.setattr(lb, "_impl", lambda problem, which: (lambda *a: None, False))
    monkeypatch.setattr(lb, "reference_callables", lambda problem: (None, None))
    monkeypatch.setattr(lb, "gen_inputs", lambda problem, device, custom_inputs_fn=None: (
        built.append(problem["workload"]["axes"]["batch_size"]) or []))
    monkeypatch.setattr(lb.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(lb.torch.cuda.profiler, "start", lambda: None)
    monkeypatch.setattr(lb.torch.cuda.profiler, "stop", lambda: None)
    path = _group_card(tmp_path, _group_problem())
    monkeypatch.delenv(lb.WORKLOAD_SCOPE_ENV, raising=False)
    lb.ncu_window(problem_path=path)
    monkeypatch.setenv(lb.WORKLOAD_SCOPE_ENV, "wl12")
    lb.ncu_window(problem_path=path)
    assert built == [3, 64]
    assert "ncu: workload wl12 (1 of 3" in capsys.readouterr().out


# --- correctness-only workloads (problem.json check_only) ---------------------------------

def _with_checks(p: dict, sizes=(128, 200)) -> dict:
    p["check_only"] = [{"index": 20 + i, "name": f"wl{20 + i}",
                        "workload": {"uuid": f"c{i}", "axes": {"batch_size": b},
                                     "inputs": {"hidden_states": {"type": "random"},
                                                "weight": {"type": "random"}}}}
                       for i, b in enumerate(sizes)]
    return p


def _only_upto(limit):
    """A kernel specialised to the timed sizes: it refuses a bigger batch outright."""
    def run(hidden_states, weight, output):
        if hidden_states.shape[0] > limit:
            raise RuntimeError(f"unsupported M = {hidden_states.shape[0]}")
        _rms(hidden_states, weight, output)
    return run


def test_check_views_are_judged_never_timed_and_named() -> None:
    p = _with_checks(_group_problem())
    views = lb.check_views(p)
    assert [v["_wl_name"] for v in views] == ["wl20", "wl21"]
    assert all(v["_check_only"] and "group" not in v and "sol_board" not in v for v in views)
    assert [v["_wl_name"] for v in lb.workload_views(p)] == ["wl10", "wl11", "wl12"]
    assert "+2 correctness-only: wl20,wl21" in lb.describe_card_workloads(p)
    assert lb.check_views(_group_problem()) == []
    verdict, runs = lb.judge_workloads(p, _rms, True, "cpu")
    assert verdict["pass"] and verdict["n_workloads"] == 5 and len(runs) == 3
    rows = {w["name"]: w for w in verdict["workloads"]}
    assert rows["wl20"]["check_only"] and "check_only" not in rows["wl10"]
    # a kernel written for the timed sizes alone fails on the first correctness-only one
    bad, runs = lb.judge_workloads(p, _only_upto(64), True, "cpu")
    assert not bad["pass"] and runs == []
    assert bad["error"].startswith('workload wl20 {"batch_size": 128} (correctness-only')
    assert "unsupported M = 128" in bad["error"]


def test_one_workload_card_judges_its_correctness_only_workloads_too() -> None:
    p = _with_checks(_problem(lb.SUITE_SOL), sizes=(40,))
    verdict, runs = lb.judge_workloads(p, _rms, True, "cpu")
    assert verdict["pass"] and verdict["n_workloads"] == 2 and len(runs) == 1
    bad, _ = lb.judge_workloads(p, _only_upto(10), True, "cpu")
    assert not bad["pass"] and "wl20" in bad["error"]


def test_group_bench_fails_on_a_correctness_only_workload_before_timing_any(
        tmp_path: Path, monkeypatch, capsys) -> None:
    from forgekernel.loop.parse import parse_eval
    seen: list = []
    monkeypatch.setattr(lb, "measure", _fake_measure(seen))
    monkeypatch.setattr(lb, "_side_windows", _windows)
    monkeypatch.setattr(lb, "warm_reference", lambda *a, **k: None)
    monkeypatch.setattr(lb, "_correctness", lambda problem, impl, dps, device: lb.sol_correctness(
        problem, impl, dps, "cpu"))
    for kernel, ok in ((_rms, True), (_only_upto(64), False)):
        monkeypatch.setattr(lb, "_impl", lambda problem, which, k=kernel: (k, True) if which == "kernel"
                            else (lambda hidden_states, weight: hidden_states, False))
        seen.clear()
        path = _group_card(tmp_path / str(ok), _with_checks(_group_problem()))
        assert lb.run_leaderboard_bench(["--mode", "all"], problem_path=path) == 0
        text = capsys.readouterr().out
        out = json.loads(text.split("=== JSON ===")[1])
        assert out["correctness"]["pass"] is ok
        if ok:     # timed exactly as before: the correctness-only workloads never are
            assert seen == [3, 17, 64] and out["performance"]["n_workloads"] == 3
            assert "(5/5 workloads judged, 2 of them correctness-only)" in text
            assert parse_eval(text).n_workloads == 3
        else:      # out before any timed workload was paid for
            assert seen == [] and out["performance"] is None
            assert "correctness-only" in out["correctness"]["error"]
            assert not parse_eval(text).correctness


def test_workload_scope_reads_a_correctness_only_workload_alone() -> None:
    p = _with_checks(_group_problem())
    scope = lb.apply_workload_scope(p, env={lb.WORKLOAD_SCOPE_ENV: "wl21"})
    assert scope == {"workloads": "wl21", "rep": "wl21", "of": 3}
    assert [g["name"] for g in p["group"]] == ["wl21"] and "check_only" not in p
    assert p["workload"]["axes"]["batch_size"] == 200
    q = _with_checks(_group_problem())
    lb.apply_workload_scope(q, env={lb.WORKLOAD_SCOPE_ENV: "wl11"})
    assert "check_only" not in q          # a meta reading never judges them
    with pytest.raises(ValueError, match="wl20, wl21"):
        lb.apply_workload_scope(_with_checks(_group_problem()), env={lb.WORKLOAD_SCOPE_ENV: "wl9"})
