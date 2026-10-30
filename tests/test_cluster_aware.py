"""Machine-aware loop verdicts: after a run moves to another machine, absolute-ms
comparisons read same-machine rows only and the target is restated from a
reference anchored on the machine the run is actually on."""

from __future__ import annotations

import json

import pytest

from forgekernel.inner.trajectory import IterationRecord as IR, append_record, load_records
from forgekernel.loop.driver import (
    _PendingPatch,
    _cluster_scale,
    _judge_patch,
    _on_cluster,
    run_inner_iteration,
    run_inner_loop,
)
from forgekernel.loop.parse import detect_plateau

from test_loop_driver import EVAL, NCU, REC, FakeBackend, _ctx


class ClusterBackend(FakeBackend):
    """A backend that reports where each submit went, and can move mid-run."""

    def __init__(self, agent_outputs, gpu_logs, cluster="home", moves=None):
        super().__init__(agent_outputs, gpu_logs)
        self.last_cluster = cluster
        # {n: cluster}: the n-th GPU submit (0-based) lands on that cluster
        self.moves = dict(moves or {})
        self.gpu_calls = 0

    def submit_gpu(self, branch: str, mode: str) -> str:
        if self.gpu_calls in self.moves:
            self.last_cluster = self.moves[self.gpu_calls]
        self.gpu_calls += 1
        return super().submit_gpu(branch, mode)


# An eval whose bench also timed the in-job reference (ref rides the job).
EVAL_REF = lambda dev, ref: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "device_median_ms": %r, "bwd_mfu_pct": 32.0, '
    '"ref_device_ms": %r, "vs_ref_device": %r}}' % (dev, dev, ref, ref / dev)
)


def _row(i, ms, cluster="", verdict="KEEP", sha=None):
    return IR(iter=i, decision_id=f"D{i}", change="x", correctness=verdict == "KEEP",
              verdict=verdict, bwd_median_ms=ms, device_median_ms=ms,
              sha=sha or f"s{i}", cluster=cluster)


def _home_ctx(tmp_path, be, **kw):
    ctx = _ctx(tmp_path, be, **kw)
    ctx.home_cluster = "home"
    ctx.gate_window = "device"
    return ctx


# ------------------------------------------------------------- stamping

def test_records_are_stamped_with_the_cluster_the_eval_ran_on(tmp_path):
    be = ClusterBackend([REC("D1", 2.6)], [EVAL(2.6), NCU], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=0.1)
    rec = run_inner_iteration(ctx, 0, metric_before=None)
    assert rec.cluster == "away"
    assert load_records(ctx.status_path)[0].cluster == "away"


def test_unstamped_rows_and_a_backend_that_cannot_say_are_one_cluster(tmp_path):
    """One machine is the identity: a plain FakeBackend (no
    last_cluster) and legacy rows without a stamp all compare together."""
    be = FakeBackend([REC("D1", 1.8)], [EVAL(1.8), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.9)
    append_record(ctx.status_path, _row(0, 2.5))
    assert run_inner_loop(ctx, start_iter=1) == "target"
    assert load_records(ctx.status_path)[1].cluster == ""


def test_a_machine_change_mid_iteration_rereads_the_best_from_the_new_one(tmp_path):
    # The student's eval fails over during the iteration: the best it entered
    # with (2.0 at home) must not be the number a 2.5 reading on the new
    # cluster is judged against — there is no prior reading there.
    be = ClusterBackend([REC("D1", 2.5)], [EVAL(2.5), NCU], cluster="home",
                        moves={0: "away"})
    ctx = _home_ctx(tmp_path, be, target_ms=0.1)
    append_record(ctx.status_path, _row(0, 2.0, cluster="home"))
    rec = run_inner_iteration(ctx, 1, metric_before=2.0)
    assert rec.cluster == "away"
    assert rec.metric_before is None


# ------------------------------------------------------------- partitioning

def test_best_and_plateau_read_same_cluster_rows_only(tmp_path):
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=0.1)
    rows = [_row(0, 1.9, "home"), _row(1, 1.9, "home"), _row(2, 1.9, "home"),
            _row(3, 1.9, "home"), _row(4, 2.5, "away"), _row(5, 2.4, "away")]
    assert [r.iter for r in _on_cluster(rows, ctx)] == [4, 5]
    assert [r.iter for r in _on_cluster(rows, ctx, "home")] == [0, 1, 2, 3]
    # Home has plateaued; the away stay has not — a mixed ledger would have
    # read the away rows as four non-improving iterations behind home's 1.9.
    assert detect_plateau(_on_cluster(rows, ctx, "home"), window=3, gate_window="device")
    assert not detect_plateau(_on_cluster(rows, ctx), window=3, gate_window="device")


# ------------------------------------------------------------- target scaling

def test_target_is_scaled_on_a_foreign_cluster_from_the_in_job_ref(tmp_path):
    # Home ref 2.0 ms, target 1.9. On "away" the same reference reads 2.6 in
    # the job, so 2.4 there is 1.083x FASTER than the ref — past target — while
    # the unscaled home number would have called it 26% short.
    be = ClusterBackend([REC("D1", 2.4)], [EVAL_REF(2.4, 2.6), NCU], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    assert run_inner_loop(ctx, start_iter=0) == "target"


def test_target_stop_is_suspended_where_the_ref_has_no_reading(tmp_path):
    # No in-job ref, no re-anchor: 1.8 on "away" would beat the home target,
    # but the home number cannot be restated there, so the loop does not stop.
    be = ClusterBackend([REC("D1", 1.8)], [EVAL(1.8), NCU], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.max_iters = 1
    assert run_inner_loop(ctx, start_iter=0) == "max_iters"


def test_fresh_reanchor_runs_once_per_cluster_and_is_persisted(tmp_path):
    calls = []

    def reanchor():
        calls.append(1)
        return ([2.6, 2.65], "away")

    be = ClusterBackend([REC("D1", 2.5), REC("D2", 2.5)],
                        [EVAL(2.5), NCU, EVAL(2.5), NCU], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.reanchor_ref = reanchor
    ctx.max_iters = 2
    # 2.5 on away vs scaled target 1.9 * 2.6/2.0 = 2.47: short, keep going
    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert len(calls) == 1, "the anchor is measured once, then read from disk"
    anchors = json.loads((tmp_path / "cluster_anchors.json").read_text())
    assert anchors["away"]["ref_ms"] == 2.6 and anchors["away"]["home_ref_ms"] == 2.0
    assert _cluster_scale(ctx) == pytest.approx(1.3)
    assert len(calls) == 1


def test_failed_reanchor_is_retried_a_bounded_number_of_times(tmp_path):
    calls = []

    def reanchor():
        calls.append(1)
        return ([], "away")

    be = ClusterBackend([REC(f"D{i}", 1.0) for i in range(4)],
                        [x for _ in range(4) for x in (EVAL(1.0), NCU)], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.reanchor_ref = reanchor
    ctx.max_iters = 4
    ctx.plateau_window = 10   # four flat readings would otherwise stop as plateau
    # 1.0 would be far past any target, but with no anchor the stop stays suspended
    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert len(calls) == 2


def test_home_cluster_scale_is_identity(tmp_path):
    be = ClusterBackend([], [], cluster="home")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.reanchor_ref = lambda: (_ for _ in ()).throw(AssertionError("must not run at home"))
    assert _cluster_scale(ctx) == 1.0


# ------------------------------------------------------------- patch judgment

def test_patch_judged_on_another_cluster_is_held_without_a_local_prior(tmp_path):
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, _row(0, 2.5, "away"))
    out = _judge_patch(ctx, _PendingPatch(best_before=2.0, milestone_before="m",
                                          good_ref="good", records_before=0,
                                          cluster="home"))
    assert out == "HOLD_PATCH"
    assert not any(c[0] == "reset" for c in be.calls)


def test_patch_is_priced_against_the_same_cluster_prior_best(tmp_path):
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=0.1)
    ctx.meta_status_path = None
    append_record(ctx.status_path, _row(0, 2.6, "away"))   # pre-patch, on away
    append_record(ctx.status_path, _row(1, 2.5, "away"))   # post-patch: -3.8%
    out = _judge_patch(ctx, _PendingPatch(best_before=2.0, milestone_before="m",
                                          good_ref="good", records_before=1,
                                          cluster="home"))
    assert out == "KEEP_PATCH"


# ------------------------------------------------------------- ref follows the student

_CARD = """
[job]
timeout = 600
{fallback}
[repo]
delivery = "local"
branch = "main"
"""
_FB = """
[[job.fallback]]
cluster = "machine_a"
resource_pool = "speech-wx"
image = "machine_a-img"
"""


def _card(tmp_path, op, fallback=""):
    d = tmp_path / "operators" / op
    d.mkdir(parents=True, exist_ok=True)
    (d / "submit.toml").write_text(_CARD.format(fallback=fallback))
    return d

def test_reanchor_samples_from_another_cluster_do_not_anchor_this_one(tmp_path):
    """The channel is supposed to follow the stay; if it did not (samples
    tagged with a different cluster), they must not become this cluster's
    anchor — a home reading would make the scale 1.0 by accident."""
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.reanchor_ref = lambda: ([2.0, 2.0], "home")
    assert _cluster_scale(ctx) is None
    ctx.reanchor_ref = lambda: ([2.6, 2.65], "away")
    assert _cluster_scale(ctx) == pytest.approx(1.3)


def test_phantom_gap_reanchor_is_skipped_on_a_foreign_cluster(tmp_path, capsys):
    from forgekernel.loop.driver import _phantom_gap_reanchor
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9)
    ctx.reanchor_ref = lambda: (_ for _ in ()).throw(AssertionError("must not measure"))
    ctx.target_anchor_ms = 2.0
    assert _phantom_gap_reanchor(ctx, 1.95) is None
    assert "home-cluster instrument" in capsys.readouterr().out


# ----------------------------------------------------------------- review fixes

def test_scale_divides_min_by_the_plain_device_reading(tmp_path):
    """Medium 2: numerator and denominator must be the same estimator on the
    same window — a median-over-graph mix loosens the scaled target inside
    the noise band."""
    from forgekernel.baseline import RefBaseline
    be = ClusterBackend([], [], cluster="away")
    ctx = _home_ctx(tmp_path, be, target_ms=1.9,
                    ref_baseline=RefBaseline(op="demo", ms=2.2, ms_device=2.0,
                                             ms_device_graph=1.9))
    for i, ref in enumerate((2.7, 2.6, 2.8)):
        append_record(ctx.status_path, IR(iter=i, decision_id="D", change="x",
                                          correctness=True, verdict="KEEP",
                                          device_median_ms=2.5, ref_device_ms=ref,
                                          cluster="away"))
    assert _cluster_scale(ctx) == pytest.approx(2.6 / 2.0)

def test_an_adopted_job_is_stamped_with_the_cluster_it_ran_on(tmp_path, monkeypatch):
    """Medium 6."""
    from forgekernel.loop import cli
    (tmp_path / ".git").mkdir()
    be = cli.RealBackend("demo", tmp_path, None, "n", "e")
    monkeypatch.setattr(cli.RealBackend, "_branch_sha", lambda self, b: "abc")
    be.last_cluster = "machine_a"
    be._record_inflight("tasks/9", "br", "all")
    assert json.loads((tmp_path / ".git" / "fk-inflight.json").read_text())["cluster"] == "machine_a"
    be.last_cluster = "machine_b"     # what a submit made NOW would resolve to
    assert be._adopt_inflight("br", "all") == "tasks/9"
    assert be.last_cluster == "machine_a"
