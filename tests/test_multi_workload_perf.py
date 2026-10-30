"""Multi-workload perf: one fixed shape, several representative workloads, the
verdict is the geometric mean.

Covers the template helpers (``measure_windows_workloads`` /
``workloads_performance`` in ``src/forgekernel/card_template/eval/timing.py``, stub-torch
harness as ``test_template_timing.py``), the eval-JSON parse, the confirmation's
per-workload medians, the reference-baseline parse, and the ``loop start``
static preflight.
"""

from __future__ import annotations

import json
import math

import pytest

from test_template_timing import FakeGpu, load_timing

from forgekernel.baseline_cli import parse_ref_stdout
from forgekernel.inner.trajectory import IterationRecord, build_digest
from forgekernel.loop.driver import _confirm_workload_medians
from forgekernel.loop.parse import describe_workload_readings, parse_eval
from forgekernel.window_contract import PERF_WORKLOADS, bench_workloads_findings


def _cost(gpu: FakeGpu, ms: float):
    """A callable that costs ``ms`` of device time on ``gpu``'s timeline."""
    def call():
        gpu.calls += 1
        gpu.clock_ms += ms
    return call


def _gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# --- template helpers ----------------------------------------------------------

def test_perf_workloads_constant_is_mirrored():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    assert timing.PERF_WORKLOADS == PERF_WORKLOADS == 3


def test_aggregate_is_the_geomean_and_its_ratio_the_geomean_of_ratios():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    kernel_ms = [1.0, 2.0, 4.0]
    ref_ms = [1.2, 1.8, 6.0]
    wl = [(f"w{i}", _cost(gpu, k), _cost(gpu, r))
          for i, (k, r) in enumerate(zip(kernel_ms, ref_ms))]
    res = timing.measure_windows_workloads(wl, burn_in_s=0)
    perf = timing.workloads_performance(res)
    assert perf["aggregate"] == "geomean"
    assert perf["n_workloads"] == 3
    assert perf["device_median_ms"] == pytest.approx(_gm(kernel_ms), rel=1e-3)
    assert perf["ref_device_ms"] == pytest.approx(_gm(ref_ms), rel=1e-3)
    ratios = [r / k for k, r in zip(kernel_ms, ref_ms)]
    assert perf["vs_ref_device"] == pytest.approx(_gm(ratios), rel=1e-3)
    assert [w["name"] for w in perf["workloads"]] == ["w0", "w1", "w2"]
    assert [w["vs_ref_device"] for w in perf["workloads"]] == pytest.approx(ratios, rel=1e-3)
    assert perf["in_job_ref"] == "paired"


def test_one_workload_losing_drags_the_aggregate_below_the_tuning_point():
    # the overfit case this exists for: fast at the campaign point, slow elsewhere
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    wl = [("campaign", _cost(gpu, 1.0), _cost(gpu, 1.1)),
          ("heavy", _cost(gpu, 1.0), _cost(gpu, 0.8)),
          ("skewed", _cost(gpu, 1.0), _cost(gpu, 0.9))]
    perf = timing.workloads_performance(timing.measure_windows_workloads(wl, burn_in_s=0))
    assert perf["workloads"][0]["vs_ref_device"] == pytest.approx(1.1, rel=1e-3)
    assert perf["vs_ref_device"] < 1.0


def test_no_ref_keys_when_any_workload_lacks_a_reference_pair():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)

    def broken():
        raise RuntimeError("ref not loadable")

    wl = [("a", _cost(gpu, 1.0), _cost(gpu, 1.0)),
          ("b", _cost(gpu, 1.0), broken),
          ("c", _cost(gpu, 1.0), _cost(gpu, 1.0))]
    perf = timing.workloads_performance(timing.measure_windows_workloads(wl, burn_in_s=0))
    assert not any(k.startswith(("ref_", "vs_")) for k in perf)
    assert perf["in_job_ref"].startswith("partial") and "b" in perf["in_job_ref"]


def test_ref_absent_everywhere_reports_like_a_single_input_card():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    wl = [(n, _cost(gpu, 1.0), None) for n in ("a", "b", "c")]
    perf = timing.workloads_performance(timing.measure_windows_workloads(wl, burn_in_s=0))
    assert perf["in_job_ref"] == "absent"
    assert "vs_ref_device" not in perf and perf["device_median_ms"] == pytest.approx(1.0)


def test_host_bound_ratio_is_the_worst_workload():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    res = {"workloads": [
        {"name": n, "ref": None, "ref_error": None,
         "kernel": {"median_ms": 1.0, "min_ms": 1.0, "device_median_ms": 1.0,
                    "host_enqueue_ms": h, "host_bound_ratio": h}}
        for n, h in (("a", 0.1), ("b", 0.7), ("c", 0.2))]}
    assert timing.workloads_performance(res)["host_bound_ratio"] == 0.7


def test_burn_in_runs_once_with_the_first_workload():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    seen = {"first": 0, "other": 0}

    def first():
        seen["first"] += 1
        gpu.clock_ms += 1.0

    def other():
        seen["other"] += 1
        gpu.clock_ms += 1.0

    res = timing.measure_windows_workloads(
        [("a", first, None), ("b", other, None), ("c", other, None)], burn_in_s=0.01)
    assert res["burn_in_rounds"] >= 1
    assert all(p["burn_in_rounds"] == 0 for p in res["workloads"])


def test_duplicate_or_empty_workloads_are_rejected():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    with pytest.raises(ValueError):
        timing.measure_windows_workloads([], burn_in_s=0)
    with pytest.raises(ValueError):
        timing.measure_windows_workloads([("a", gpu.call, None), ("a", gpu.call, None)],
                                         burn_in_s=0)


# --- eval JSON -> driver --------------------------------------------------------

def _eval_log(perf: dict) -> str:
    return "=== JSON ===\n" + json.dumps({"correctness": {"pass": True}, "performance": perf})


def test_parse_eval_reads_the_aggregate_and_compact_workloads():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    wl = [(f"w{i}", _cost(gpu, k), _cost(gpu, r))
          for i, (k, r) in enumerate([(1.0, 1.1), (2.0, 2.0), (4.0, 3.6)])]
    perf = timing.workloads_performance(timing.measure_windows_workloads(wl, burn_in_s=0))
    ev = parse_eval(_eval_log(perf))
    assert ev.device_median_ms == pytest.approx(perf["device_median_ms"])
    assert ev.vs_ref_device == pytest.approx(perf["vs_ref_device"])
    assert ev.n_workloads == 3
    assert [w["name"] for w in ev.workloads] == ["w0", "w1", "w2"]
    assert set(ev.workloads[0]) <= {"name", "device_median_ms", "bwd_median_ms",
                                    "ref_device_ms", "vs_ref_device", "vs_ref"}
    line = describe_workload_readings(ev.workloads)
    assert "w0" in line and "(x1.1000)" in line


def test_single_input_bench_parses_as_one_workload():
    ev = parse_eval(_eval_log({"bwd_median_ms": 1.0, "device_median_ms": 0.9}))
    assert ev.n_workloads == 1 and ev.workloads == []


def test_digest_shows_which_workload_carries_the_best():
    rec = IterationRecord(iter=1, decision_id="d", change="c", correctness=True,
                          verdict="KEEP", bwd_median_ms=1.0, device_median_ms=1.0,
                          n_workloads=3, workloads=[{"name": "a", "vs_ref_device": 1.2}])
    achieved = build_digest([rec], {"value": 1.0})["achieved"]
    assert achieved["aggregate"] == "geomean over 3 workloads"
    assert achieved["workloads"][0]["name"] == "a"


def test_confirmation_keeps_per_workload_medians():
    samples = [
        {"workloads": [{"name": "a", "device_median_ms": 1.0, "vs_ref_device": 1.10},
                       {"name": "b", "device_median_ms": 2.0, "vs_ref_device": 0.90}]},
        {"workloads": [{"name": "a", "device_median_ms": 1.2, "vs_ref_device": 1.06},
                       {"name": "b", "device_median_ms": 2.2, "vs_ref_device": 0.94}]},
        {"workloads": [{"name": "a", "device_median_ms": 1.1, "vs_ref_device": 1.08}]},
    ]
    out = _confirm_workload_medians(samples, "device")
    assert [w["name"] for w in out] == ["a", "b"]
    assert out[0]["n"] == 3 and out[0]["vs_ref_device"] == pytest.approx(1.08)
    assert out[1]["n"] == 2 and out[1]["device_median_ms"] == pytest.approx(2.1)
    assert _confirm_workload_medians([{"ms": 1.0}], "device") == []


# --- reference baseline --------------------------------------------------------

def test_ref_baseline_archives_the_geomean_not_the_lightest_workload():
    report = {"aggregate": "geomean", "n_workloads": 3,
              "bwd_median_ms": 2.0, "device_median_ms": 1.9,
              "workloads": [{"name": "light", "bwd_median_ms": 0.5, "device_median_ms": 0.4},
                            {"name": "mid", "bwd_median_ms": 2.0, "device_median_ms": 1.9},
                            {"name": "heavy", "bwd_median_ms": 8.0, "device_median_ms": 7.5}]}
    got = parse_ref_stdout(json.dumps(report))
    assert got["ms"] == 2.0 and got["ms_device"] == 1.9


def test_ref_baseline_without_aggregate_still_takes_the_fastest_backend():
    report = {"flash": {"median_ms": 1.0}, "cudnn": {"median_ms": 0.8}}
    assert parse_ref_stdout(json.dumps(report))["ms"] == 0.8


# --- loop start preflight --------------------------------------------------------

def test_preflight_flags_a_single_input_bench():
    src = "from timing import measure_paired\nres = measure_paired(k, r)\n"
    assert bench_workloads_findings(src)


def test_preflight_accepts_three_workloads():
    src = ("import timing\n"
           "res = timing.measure_windows_workloads([('a', k, r), ('b', k, r), ('c', k, r)])\n")
    assert bench_workloads_findings(src) == []


def test_preflight_counts_a_literal_list_that_is_too_short():
    src = "res = measure_windows_workloads([('a', k, r), ('b', k, r)])\n"
    (finding,) = bench_workloads_findings(src)
    assert "2 workload(s)" in finding


def test_preflight_cannot_count_a_runtime_list_and_does_not_guess():
    src = "wl = build()\nres = measure_windows_workloads(wl, prep=None)\n"
    assert bench_workloads_findings(src) == []


# --- prep step per workload ------------------------------------------------------

def test_prep_takes_the_workload_name_and_each_workload_gets_its_own_tensors():
    from test_template_timing import FakeTensor
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    prepped, seen = [], {"a": set(), "b": set(), "c": set()}

    def prep(workload):
        prepped.append(workload)
        return {"A": FakeTensor((4, 32, 64), "torch.bfloat16", data=workload)}

    def kernel(name):
        def call(saved):
            seen[name].add(saved["A"].data)
            gpu.call()
        return call

    # burn-in on: the prep-taking kernel must be fed its tensors there too
    res = timing.measure_windows_workloads(
        [(n, kernel(n), None) for n in ("a", "b", "c")], prep=prep, burn_in_s=0.01)
    assert res["burn_in_rounds"] >= 1
    assert seen == {"a": {"a"}, "b": {"b"}, "c": {"c"}}   # never another workload's tensors
    assert set(prepped) == {"a", "b", "c"}
    perf = timing.workloads_performance(res)
    prep_block = perf["prep"]
    assert prep_block["outputs"] == {"A": {"shape": [4, 32, 64], "dtype": "bfloat16"}}
    assert prep_block["mutated"] == []
    assert [w["name"] for w in prep_block["workloads"]] == ["a", "b", "c"]


def test_a_mutation_on_any_workload_reaches_the_top_level_prep_block():
    from test_template_timing import FakeTensor
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)

    def prep(workload):
        return {"A": FakeTensor((2, 2), "torch.float16", data=0)}

    def tamper(saved):
        saved["A"].data = 7
        gpu.call()

    res = timing.measure_windows_workloads(
        [("a", lambda saved: gpu.call(), None), ("b", tamper, None)], prep=prep, burn_in_s=0)
    perf = timing.workloads_performance(res)
    assert perf["prep"]["mutated"] == ["A"]
    assert {w["name"]: w["mutated"] for w in perf["prep"]["workloads"]} == {"a": [], "b": ["A"]}


def test_a_zero_argument_prep_is_refused_for_several_workloads():
    timing = load_timing(FakeGpu(device_ms=1.0))
    with pytest.raises(TypeError, match="prep\\(name\\)"):
        timing.measure_windows_workloads(
            [("a", lambda saved: None, None), ("b", lambda saved: None, None)],
            prep=lambda: {}, burn_in_s=0)


def test_measure_paired_burn_in_feeds_the_prep_tensors():
    from test_template_timing import FakeTensor
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    got = []
    out = timing.measure_paired(lambda saved: (got.append(sorted(saved)), gpu.call()),
                                prep=lambda: {"A": FakeTensor((1,), "torch.float32")},
                                burn_in_s=0.01)
    assert out["burn_in_rounds"] >= 1 and got and all(g == ["A"] for g in got)


def test_the_driver_holds_every_workloads_prep_block_to_the_declaration():
    from test_window_contract import PREP_OK, SYMMETRIZED

    from forgekernel.window_contract import check_prep_report, parse_window_contract
    contract = parse_window_contract(SYMMETRIZED)
    bad = dict(PREP_OK["outputs"], A={"shape": [4, 32, 32], "dtype": "bf16"})
    report = dict(PREP_OK, workloads=[dict(PREP_OK, name="a"),
                                      {"name": "b", "outputs": bad, "mutated": []},
                                      {"name": "c", "outputs": bad, "mutated": []}])
    (problem,) = check_prep_report(contract, report)
    assert "shape [4, 32, 32], declared [4, 32, 64]" in problem
    assert problem.endswith("(workloads b, c)")
    ok = dict(PREP_OK, workloads=[dict(PREP_OK, name=n) for n in "abc"])
    assert check_prep_report(contract, ok) == []


# --- the absolute target: the reference must cover the same workloads -------------

def test_workloads_mismatch_compares_count_then_names():
    from forgekernel.baseline import workloads_mismatch
    assert workloads_mismatch(1, [], 1, []) is None
    assert workloads_mismatch(3, ["a", "b", "c"], 3, ["c", "b", "a"]) is None
    assert workloads_mismatch(3, [], 3, ["a", "b", "c"]) is None   # unnamed: count only
    assert "one input set" in workloads_mismatch(1, [], 3, ["a", "b", "c"])
    assert "geomean over 3" in workloads_mismatch(3, ["a", "b", "c"], 1, [])
    assert "covers workloads" in workloads_mismatch(3, ["a", "b", "x"], 3, ["a", "b", "c"])


def test_ref_baseline_records_which_workloads_its_geomean_covers():
    report = {"aggregate": "geomean", "n_workloads": 3,
              "bwd_median_ms": 2.0, "device_median_ms": 1.9,
              "workloads": [{"name": n, "bwd_median_ms": 1.0, "device_median_ms": 0.9,
                             "graph_device_median_ms": 0.8, "tflops": 99.0}
                            for n in ("campaign", "heavy", "skewed")]}
    got = parse_ref_stdout(json.dumps(report))
    assert got["n_workloads"] == 3 and got["workloads"] == ["campaign", "heavy", "skewed"]
    # a side reading only one workload carries is not archived beside the geomean
    assert "ms_device_graph" not in got and "tflops" not in got
    single = parse_ref_stdout(json.dumps({"median_ms": 1.0, "device_median_ms": 0.9}))
    assert "n_workloads" not in single


def test_ref_baseline_workloads_survive_the_archive_round_trip(tmp_path):
    from forgekernel.baseline import RefBaseline, load_baseline, save_baseline
    save_baseline(tmp_path, RefBaseline(op="demo", ms=2.0, ms_device=1.9, n_workloads=3,
                                        workloads=["a", "b", "c"]))
    back = load_baseline(tmp_path)
    assert back.n_workloads == 3 and back.workloads == ["a", "b", "c"]
    assert "the geomean over 3 workloads (a, b, c)" in back.render("device")


def test_declared_arch_ref_is_one_input_set_unless_its_workloads_are_named():
    from forgekernel.loop.cli import _declared_arch_baseline
    cfg = {"loop": {"arch_ref_ms": 2.0, "arch_ref_device_ms": 1.9}}
    assert _declared_arch_baseline("demo", cfg).n_workloads == 1
    cfg["loop"]["arch_ref_workloads"] = ["a", "b", "c"]
    arch = _declared_arch_baseline("demo", cfg)
    assert arch.n_workloads == 3 and arch.workloads == ["a", "b", "c"]


def _card(tmp_path, bench_src):
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "bench.py").write_text(bench_src, encoding="utf-8")
    return tmp_path


_MULTI_BENCH = "res = measure_windows_workloads(build())\n"
_SINGLE_BENCH = "res = measure_paired(k, r)\n"


def test_loop_start_flags_a_single_input_archive_under_a_multi_workload_bench(tmp_path):
    from forgekernel.baseline import ROLE_ARCHITECTURE, RefBaseline
    from forgekernel.loop.cli import _target_workloads_problems
    op_dir = _card(tmp_path, _MULTI_BENCH)
    single = RefBaseline(op="demo", ms=2.0)
    multi = RefBaseline(op="demo", ms=2.0, n_workloads=3, workloads=["a", "b", "c"],
                        role=ROLE_ARCHITECTURE)
    (finding,) = _target_workloads_problems("demo", op_dir, (multi, single))
    assert "ONE input set" in finding
    assert _target_workloads_problems("demo", op_dir, (multi, None)) == []
    # the reverse: a geomean archive under a bench that times one input set
    op_dir = _card(tmp_path, _SINGLE_BENCH)
    (finding,) = _target_workloads_problems("demo", op_dir, (multi, None))
    assert "geomean over 3 workloads" in finding
    # no bench to read: nothing to hold the archives against
    assert _target_workloads_problems("demo", tmp_path / "nowhere", (single,)) == []


def test_the_finish_ratio_refuses_ref_runs_over_other_workloads():
    from forgekernel.loop.cli import _fresh_ref_workloads_mismatch
    confirm = {"workloads": [{"name": n} for n in ("a", "b", "c")]}
    single_runs = [{"ms": 1.0, "ms_device": 0.9}]
    assert "one input set" in _fresh_ref_workloads_mismatch(single_runs, confirm)
    same = [{"ms": 1.0, "n_workloads": 3, "workloads": ["a", "b", "c"]}]
    assert _fresh_ref_workloads_mismatch(same, confirm) is None
    assert _fresh_ref_workloads_mismatch(single_runs, {}) is None   # single-input card


def EVAL_WL(ms):
    """A multi-workload bench's eval JSON."""
    return "=== JSON ===\n" + json.dumps({
        "correctness": {"pass": True},
        "performance": {"bwd_median_ms": ms, "device_median_ms": ms, "aggregate": "geomean",
                        "n_workloads": 3,
                        "workloads": [{"name": n, "bwd_median_ms": ms, "device_median_ms": ms}
                                      for n in ("a", "b", "c")]}})


def test_an_absolute_hit_against_a_single_input_archive_does_not_stop_the_run(tmp_path):
    """The gate reading is the geomean over 3 workloads, the archive one input
    set: 1.85 under a 1.90 target is the workloads' cost mix, not a win — the
    run neither stops nor buys a re-anchor on it."""
    from test_false_win_guard import PushingBackend, _ctx_with_anchor
    from test_loop_driver import NCU, REC

    from forgekernel.loop.driver import run_inner_loop
    calls = []
    be = PushingBackend([REC("D1", 1.85), REC("D2", 1.85)],
                        [EVAL_WL(1.85), NCU, EVAL_WL(1.85), NCU])
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00,
                           reanchor=lambda: calls.append(1) or ([2.0, 2.0], ""))
    ctx.max_iters = 2
    ctx.plateau_window = 10
    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert calls == []


def test_target_mismatch_reads_the_best_keeps_workloads(tmp_path):
    from dataclasses import replace

    from test_loop_driver import FakeBackend, _ctx

    from forgekernel.inner.trajectory import IterationRecord, append_record
    from forgekernel.loop.driver import _target_workloads_mismatch
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert _target_workloads_mismatch(ctx) is None          # nothing kept yet
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=1.0, device_median_ms=1.0, n_workloads=3,
        workloads=[{"name": n, "device_median_ms": 1.0} for n in ("a", "b", "c")]))
    assert "one input set" in _target_workloads_mismatch(ctx)
    ctx.ref_baseline = replace(ctx.ref_baseline, n_workloads=3, workloads=["a", "b", "c"])
    assert _target_workloads_mismatch(ctx) is None
    ctx.ref_baseline = replace(ctx.ref_baseline, workloads=["a", "b", "x"])
    assert "covers workloads" in _target_workloads_mismatch(ctx)


def test_phantom_gap_skips_the_absolute_residual_under_a_mismatch(tmp_path, monkeypatch):
    from test_loop_driver import FakeBackend, _ctx

    from forgekernel.inner.trajectory import IterationRecord, append_record
    from forgekernel.loop import driver
    calls, paired = [], []
    ctx = _ctx(tmp_path, FakeBackend([], []), target_ms=1.0)
    ctx.gate_window = "device"
    ctx.home_cluster = ""
    ctx.reanchor_band = 0.1
    ctx.target_anchor_ms = 1.0
    ctx.reanchor_ref = lambda: calls.append(1) or ([1.0, 1.0], "")
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=1.05, device_median_ms=1.05, n_workloads=3,
        workloads=[{"name": n} for n in ("a", "b", "c")]))
    assert driver._phantom_gap_reanchor(ctx, 1.05) is None
    assert calls == []                                       # no ref jobs bought on it
    # a ref_in_job run still gets its same-card ratio, with no absolute residual quoted
    monkeypatch.setattr(driver, "_paired_reanchor",
                        lambda c, best, gap: paired.append(gap) or None)
    ctx.ref_in_job = True
    driver._phantom_gap_reanchor(ctx, 1.05)
    assert paired == [None]


def test_the_workload_count_warning_is_a_plain_bench_contract(tmp_path):
    from types import SimpleNamespace
    from forgekernel.loop.driver import _too_few_workloads
    ctx = SimpleNamespace(workspace=tmp_path)
    assert _too_few_workloads(ctx, SimpleNamespace(n_workloads=2))
    assert not _too_few_workloads(ctx, SimpleNamespace(n_workloads=PERF_WORKLOADS))
    # a task card: two shapes, each a geomean over its own input groups
    (tmp_path / "eval" / "leaderboard").mkdir(parents=True)
    (tmp_path / "eval" / "leaderboard" / "problem.json").write_text('{"suite": "forgekernel"}')
    assert not _too_few_workloads(ctx, SimpleNamespace(n_workloads=2))
