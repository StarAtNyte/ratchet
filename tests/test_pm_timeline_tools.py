"""The PM-sampling phase timeline: rendering, phase stats, verdict gate."""
from collections import OrderedDict

from forgekernel.loop.driver import _PM_TIMELINE_RE, _PMTL_OK, _PMTL_WAIVED
from forgekernel.skills import pm_timeline_tools as pmt

STALL = "pmsampling:smsp__warps_issue_stalled_barrier.avg"
TENSOR = "pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed"
MISC = "pmsampling:l1tex__t_sector_hit_rate.pct"


def _ramp_plateau_drain(n=400, plateau=100.0):
    """Dead air, linear ramp, flat plateau, linear drain, dead air."""
    vals = [0.0] * 20
    vals += [plateau * i / 40 for i in range(1, 41)]
    vals += [plateau] * (n - 120)
    vals += [plateau * (40 - i) / 40 for i in range(40)]
    vals += [0.0] * 20
    return vals


def test_phase_stats_reads_the_shape():
    st = pmt.phase_stats(_ramp_plateau_drain())
    lo, hi = st["active"]
    assert lo >= 20 and hi < 381        # dead air excluded on both sides
    assert abs(st["plateau"] - 100.0) < 1.0
    # 40-sample ramp/drain over a ~360-sample active span ≈ 0.1 each
    assert 0.0 < st["ramp"] < 0.2 and 0.0 < st["drain"] < 0.2
    assert pmt.phase_stats([0.0] * 8)["max"] == 0.0  # all-zero series is inert


def test_build_timeline_full_rows_for_stall_and_core_only():
    series = OrderedDict([
        (STALL, _ramp_plateau_drain()),
        (TENSOR, _ramp_plateau_drain(plateau=48.0)),
        (MISC, [1.0] * 300),
    ])
    out = []
    n, n_stall, samples = pmt.build_timeline(series, out=out.append)
    assert (n, n_stall, samples) == (3, 1, 400)
    text = "\n".join(out)
    # stall + tensor get value rows ("PM <head>" then a data line); misc a stat row
    assert f"PM {pmt.short_series(STALL)}" in text
    assert f"PM {pmt.short_series(TENSOR)}" in text
    assert f"PM- {pmt.short_series(MISC)}" in text
    # nothing populated goes silently missing from the block
    for name in series:
        assert pmt.short_series(name) in text


def test_short_series_compresses_the_stall_prefix():
    assert pmt.short_series(STALL) == "stall:barrier.avg"


def test_verdict_and_marker_spellings_match_the_driver_gate():
    ok = "PMTL-OK kernel=main_kernel series=27 stall_series=13 stall_samples=368"
    assert _PMTL_OK.search(ok)
    # prose cannot open the gate (^ anchor, same trust model as STALL-OK)
    assert not _PMTL_OK.search("the probe should print PMTL-OK kernel=x ")
    assert _PMTL_WAIVED.search("PMTL-WAIVED: vGPU image, PM sampling unsupported")
    body = f"{pmt.BEGIN}\nrow1\nrow2\n{pmt.END}"
    m = _PM_TIMELINE_RE.search(f"noise\n{body}\n{ok}")
    assert m and m.group(1) == "row1\nrow2"


def test_timeline_report_fails_closed_without_the_ncu_report_module(
        monkeypatch, tmp_path):
    monkeypatch.setattr(pmt, "_import_ncu_report", lambda: None)
    out = []
    ok = pmt.timeline_report(tmp_path / "x.ncu-rep", out=out.append)
    assert ok is False
    assert any(line.startswith("PMTL-FAIL") for line in out)
    assert not any(_PMTL_OK.search(line) for line in out)


def test_reader_api_breakage_is_loud_not_empty():
    class BrokenNum:
        def num_instances(self):
            raise AttributeError("accessor moved in a new ncu release")

    class BrokenKind:
        # num_instances survives the drift but kind()/ValueKind moved — this
        # exact shape once leaked past a blanket except as "no PM series"
        def num_instances(self):
            return 3

        def kind(self):
            raise AttributeError("kind moved in a new ncu release")

    def action_with(metric):
        class FakeAction:
            def metric_names(self):
                return [STALL]

            def __getitem__(self, k):
                return metric
        return FakeAction()

    for metric in (BrokenNum(), BrokenKind()):
        try:
            pmt.collect_pm_series(action_with(metric))
        except pmt.MetricReadError:
            pass
        else:
            raise AssertionError(
                f"{type(metric).__name__}: breakage must raise, never read as empty")


def test_pm_timeline_run_cmd_captures_a_choosable_launch_set():
    from forgekernel.skills import pm_timeline_run as ptr
    cmd = " ".join(ptr.build_cmd(ptr.INTERVAL))
    # dominant-by-duration selection is only a real check with >1 captured
    # launch (with 1 it is the identity and a mis-aim gets a green verdict),
    # and PM samples accumulate per launch — fast kernels need the extras
    assert "--launch-count 8" in cmd
    assert "--profile-from-start off" in cmd
    assert "--pm-sampling-interval 2048" in " ".join(ptr.build_cmd(2048))
