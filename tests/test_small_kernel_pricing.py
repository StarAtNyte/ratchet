"""Short kernels (5-20 us, the SOL-ExecBench sizes): the profiler window vouches for a
dominant kernel below the 20 us mis-aim floor, its samples pool over its timed
launches, and a starved PM timeline is re-collected once at a finer interval."""
import sys
import types

from forgekernel.loop.driver import _PM_TIMELINE_RE, _PMTL_OK, _STALL_OK
from forgekernel.skills import pm_timeline_tools as pmt
from forgekernel.skills import role_exposure_tools as RE
from forgekernel.skills import stall_price_run as spr
from forgekernel.skills import stall_tools as st

LS = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"


class _Val:
    def __init__(self, v):
        self.v = v

    def value(self):
        return self.v


class _Cor:
    def __init__(self, pcs):
        self.pcs = pcs

    def as_uint64(self, i):
        return self.pcs[i]


class _Metric:
    ValueKind_UINT64, ValueKind_DOUBLE, ValueKind_FLOAT = 1, 2, 3

    def __init__(self, per_pc):
        self.pcs = sorted(per_pc)
        self.vals = [per_pc[p] for p in self.pcs]

    def num_instances(self):
        return len(self.pcs)

    def has_correlation_ids(self):
        return True

    def correlation_ids(self):
        return _Cor(self.pcs)

    def kind(self):
        return self.ValueKind_UINT64

    def as_uint64(self, i):
        return self.vals[i]


class _Action:
    def __init__(self, name, dur_ns, per_pc=None):
        self._name, self.dur, self.per_pc = name, dur_ns, per_pc or {}

    def name(self):
        return self._name

    def __getitem__(self, key):
        if key == "gpu__time_duration.sum":
            return _Val(self.dur)
        if key == LS and self.per_pc:
            return _Metric(self.per_pc)
        raise KeyError(key)

    def source_info(self, pc):
        return None


class _Range:
    def __init__(self, actions):
        self.actions = actions

    def num_actions(self):
        return len(self.actions)

    def action_by_idx(self, i):
        return self.actions[i]


class _Report:
    def __init__(self, actions):
        self.rng = _Range(actions)

    def range_by_idx(self, i):
        return self.rng


def _spread(n_pcs, per, base=0x1000):
    """{pc: samples}: one launch's samples spread over n_pcs PCs."""
    return {base + 16 * i: per for i in range(n_pcs)}


def _install(monkeypatch, mod, actions):
    fake = types.SimpleNamespace(load_report=lambda p: _Report(actions))
    monkeypatch.setattr(mod, "_import_ncu_report", lambda: fake)


# observed in an audit of earlier runs: a 13 us CuTe DSL GEMM, 8 timed launches,
# 232 PC samples on the dominant one -> STALL-FAIL mis-aim, ROLE-FAIL too-few-samples.
def test_short_kernel_the_window_vouches_for_is_priced_on_pooled_samples(monkeypatch, tmp_path):
    acts = [_Action("kernel_cutlass_gemm", 13056.0 - i, _spread(80, 3)) for i in range(8)]
    _install(monkeypatch, st, acts)
    out = []
    ok = st.price_report(tmp_path / "r.ncu-rep", out=out.append)
    text = "\n".join(out)
    assert ok, text
    assert "short kernel (13056.0ns < 20000ns) accepted: 8 launch(es)" in text
    assert "pooled over 8 launches" in text
    assert _STALL_OK.search(text) and "samples=1920 " in text


def test_baseline_floor_would_have_failed_the_same_window():
    # the pre-fix rule: duration alone, one launch's samples alone
    acts = [_Action("kernel_cutlass_gemm", 13056.0, _spread(80, 3)) for _ in range(8)]
    assert st._duration(acts[0]) < st.MIN_DURATION_NS
    assert st._sample_total(st.collect_per_pc(acts[0])) < st.MIN_SAMPLES


def test_a_one_off_short_kernel_is_still_a_mis_aim(monkeypatch, tmp_path):
    # the setup-kernel class: a setup kernel that happens to be the longest captured launch,
    # with no repeat of it in the window -> rejected as before
    acts = [_Action("at_native_fill", 9000.0, _spread(80, 30)),
            _Action("elementwise_copy", 4000.0), _Action("cub_scan", 3000.0)]
    _install(monkeypatch, st, acts)
    out = []
    assert st.price_report(tmp_path / "r.ncu-rep", out=out.append) is False
    assert any(l.startswith("STALL-FAIL reason=mis-aim") and "does not vouch" in l for l in out)


def test_a_repeated_short_kernel_with_a_minor_share_is_still_a_mis_aim(monkeypatch, tmp_path):
    acts = ([_Action("small_aux", 15000.0, _spread(80, 3)) for _ in range(2)]
            + [_Action("other", 14000.0) for _ in range(6)])
    _install(monkeypatch, st, acts)
    out = []
    assert st.price_report(tmp_path / "r.ncu-rep", out=out.append) is False
    assert any("STALL-FAIL reason=mis-aim" in l for l in out)


def test_pooling_skips_a_same_named_launch_from_another_address_space():
    dom = _Action("k", 13000.0, _spread(40, 5, base=0x1000))
    twin = _Action("k", 12000.0, _spread(40, 5, base=0x1000))
    alien = _Action("k", 12500.0, _spread(40, 5, base=0x900000))
    per_pc, n = st.collect_per_pc_pooled(_Report([dom, twin, alien]), dom)
    assert n == 2
    assert st._sample_total(per_pc) == 400
    assert not any(pc >= 0x900000 for pc in per_pc)


def test_pooling_only_when_one_launch_falls_short():
    dom = _Action("k", 30000.0, _spread(100, 6))
    other = _Action("k", 29000.0, _spread(100, 6))
    per_pc, n = st.collect_per_pc_pooled(_Report([dom, other]), dom)
    assert n == 1 and st._sample_total(per_pc) == 600     # a long kernel reads as before


def test_no_samples_at_all_is_not_pooled_into_a_pass(monkeypatch, tmp_path):
    # sampler off (probe without --set full): every launch is empty -> still too few
    acts = [_Action("k", 13000.0) for _ in range(8)]
    _install(monkeypatch, st, acts)
    out = []
    assert st.price_report(tmp_path / "r.ncu-rep", out=out.append) is False
    assert any(l.startswith("STALL-FAIL reason=too-few-samples") for l in out)


def test_role_exposure_pools_the_same_way(monkeypatch, tmp_path):
    acts = [_Action("kernel_cutlass_gemm", 13056.0 - i, _spread(80, 3)) for i in range(8)]
    _install(monkeypatch, st, acts)
    monkeypatch.setattr(RE, "_stall_tools", lambda: st)
    monkeypatch.setattr(RE, "_orch", lambda: types.SimpleNamespace())
    seen = {}

    def stop_after_samples(*a, **k):        # the samples gate passed: stop at the page
        seen["rendered"] = True
        return None, "stop"
    monkeypatch.setattr(st, "render_sass_text", stop_after_samples)
    out = []
    RE.report(tmp_path / "r.ncu-rep", out=out.append)
    assert seen.get("rendered"), out
    assert any("role samples pooled over 8 launches" in l for l in out)
    assert not any("too-few-samples" in l for l in out)


def test_pm_timeline_accepts_a_vouched_short_kernel(monkeypatch, tmp_path):
    acts = [_Action("kernel_cutlass_gemm", 13056.0, {}) for _ in range(8)]
    _install(monkeypatch, pmt, acts)
    monkeypatch.setattr(pmt, "collect_pm_series", lambda a: {})
    out = []
    pmt.timeline_report(tmp_path / "r.ncu-rep", out=out.append)
    assert not any("mis-aim" in l for l in out), out
    assert any(l.startswith("PMTL-FAIL reason=no-pm-series") for l in out)


def test_pm_timeline_still_rejects_an_unvouched_short_kernel(monkeypatch, tmp_path):
    acts = [_Action("setup", 13056.0, {}), _Action("other", 1000.0, {})]
    _install(monkeypatch, pmt, acts)
    out = []
    assert pmt.timeline_report(tmp_path / "r.ncu-rep", out=out.append) is False
    assert any(l.startswith("PMTL-FAIL reason=mis-aim") for l in out)


def test_window_rule_constants_agree_across_the_two_tools():
    assert pmt.WINDOW_MIN_REPEATS == st.WINDOW_MIN_REPEATS
    assert pmt.WINDOW_MIN_SHARE == st.WINDOW_MIN_SHARE
    assert pmt.MIN_DURATION_NS == st.MIN_DURATION_NS


# --- the starved PM timeline: one finer re-collection -------------------------------

def test_finer_interval_doubles_the_floor_in_powers_of_two():
    assert spr.finer_interval(2048, 52, 64) == 512        # 2048*52/128 = 832 -> 512
    assert spr.finer_interval(2048, 10, 64) == 128        # clamps at the minimum
    assert spr.finer_interval(2048, 0, 64) is None        # dead series: nothing to scale
    assert spr.finer_interval(128, 40, 64) is None        # already at the minimum


def _coarse_then(monkeypatch, retry_report: bool, calls):
    def timeline_report(path, out=print, **k):
        calls.append(str(path))
        if "stallprice_pm" in str(path):
            out("=== PM TIMELINE BEGIN ===\nfine\n=== PM TIMELINE END ===")
            out("PMTL-OK kernel=k series=9 stall_series=4 stall_samples=200")
        else:
            out("=== PM TIMELINE BEGIN ===\ncoarse\n=== PM TIMELINE END ===")
            out("PMTL-FAIL reason=interval-too-coarse (max 52 stall samples < 64 — x)")
        return True
    runs = []
    monkeypatch.setattr(spr.subprocess, "run", lambda cmd, **k: (runs.append(cmd), types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))[1])
    monkeypatch.setattr(spr.os.path, "exists", lambda p: retry_report)
    return types.SimpleNamespace(timeline_report=timeline_report), runs


def test_starved_timeline_is_replaced_by_the_finer_one(monkeypatch, capsys):
    calls = []
    pt, runs = _coarse_then(monkeypatch, True, calls)
    spr._timeline(pt, 2048, ["python3", "binref.py", "ncu"])
    log = capsys.readouterr().out
    assert len(runs) == 1 and "--pm-sampling-interval 512" in " ".join(runs[0])
    assert runs[0][-3:] == ["python3", "binref.py", "ncu"]       # same launch target
    assert "--set" not in runs[0]                                  # sections only
    # the driver banks the FIRST block: it must be the fine one
    assert _PM_TIMELINE_RE.search(log).group(1) == "fine"
    assert _PMTL_OK.search(log)


def test_failed_finer_collection_keeps_the_first_verdict(monkeypatch, capsys):
    calls = []
    pt, runs = _coarse_then(monkeypatch, False, calls)
    spr._timeline(pt, 2048, None)
    log = capsys.readouterr().out
    assert "keeping the first verdict" in log
    assert "PMTL-FAIL reason=interval-too-coarse" in log
    assert calls == ["/tmp/stallprice.ncu-rep"]


def test_a_timeline_that_is_not_starved_is_printed_as_is(monkeypatch, capsys):
    def timeline_report(path, out=print, **k):
        out("PMTL-OK kernel=k series=9 stall_series=4 stall_samples=90")
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    spr._timeline(types.SimpleNamespace(timeline_report=timeline_report), 2048, None)
    assert "PMTL-OK" in capsys.readouterr().out


def test_main_threads_interval_and_app_into_the_timeline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(spr, "_collect", lambda pm, pm_interval=spr.PM_INTERVAL, app=None: True)
    seen = {}
    st_fake = types.SimpleNamespace(price_report=lambda *a, **k: None,
                                    segment_ledger=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "stall_tools", st_fake)
    monkeypatch.setitem(sys.modules, "pm_timeline_tools", types.SimpleNamespace())
    monkeypatch.setattr(spr, "_timeline", lambda mod, iv, app: seen.update(iv=iv, app=app))
    spr.main(["--pm", "--pm-interval", "2048", "--app", "python3", "ref.py", "ncu"])
    assert seen == {"iv": 2048, "app": ["python3", "ref.py", "ncu"]}


# ------------------------------------------- a failed collection never reads the last job's


def test_a_failed_collection_is_not_read_off_the_previous_jobs_report(monkeypatch, tmp_path):
    """Regression: a GPU machine keeps /tmp across jobs, and the report's existence
    is the only gate -- a collection that produced nothing was analyzed off the previous
    job's report (another tree, another workload) and printed as this one's verdict."""
    monkeypatch.setattr(spr, "REPORT", str(tmp_path / "stallprice"))
    (tmp_path / "stallprice.ncu-rep").write_text("the previous job's report")
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="==ERROR== no kernels were profiled"))
    assert spr._collect(pm=True) is False
    assert not (tmp_path / "stallprice.ncu-rep").exists()


def test_a_failed_finer_pm_collection_is_not_read_off_the_previous_jobs_report(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(spr, "REPORT_PM", str(tmp_path / "stallprice_pm"))
    (tmp_path / "stallprice_pm.ncu-rep").write_text("the previous job's report")
    calls = []

    def timeline_report(path, out=print, **k):
        calls.append(str(path))
        out("PMTL-FAIL reason=interval-too-coarse (max 52 stall samples < 64 — x)")
    monkeypatch.setattr(spr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr=""))
    spr._timeline(types.SimpleNamespace(timeline_report=timeline_report), 2048, None)
    assert "keeping the first verdict" in capsys.readouterr().out
    assert calls == [spr.REPORT + ".ncu-rep"]           # the stale fine report was never read
    assert not (tmp_path / "stallprice_pm.ncu-rep").exists()


def test_pm_timeline_run_does_not_read_the_previous_jobs_report(monkeypatch, tmp_path, capsys):
    from forgekernel.skills import pm_timeline_run as ptr
    monkeypatch.setattr(ptr, "REPORT", str(tmp_path / "pm_timeline"))
    (tmp_path / "pm_timeline.ncu-rep").write_text("the previous job's report")
    monkeypatch.setattr(ptr.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr=""))
    assert ptr.main([]) == 0
    assert "PMTL-FAIL reason=no-report" in capsys.readouterr().out
