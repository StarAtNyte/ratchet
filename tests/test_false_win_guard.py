"""The false-win guard: an absolute target hit is verified same-period before it
may end a run.

The phantom-gap instrument only ever guarded the FALSE-GAP direction (thinking
you are behind when the anchor drifted). A self-ref run hit the other
one: it stopped as `target` at iteration 0 having changed no executable code,
because its launch anchor had been drawn on a slower node than the student's job
landed on — 5.1% apart on byte-identical code, against a 5% goal.
"""

from __future__ import annotations

import json

from forgekernel.loop.driver import _confirm_apparent_target, run_inner_loop

from test_loop_driver import _EVAL_CONFIRM, EVAL, NCU, REC, FakeBackend, _ctx


class PushingBackend(FakeBackend):
    """A backend the confirmation stage can actually drive: it can push a named
    ref, which is what run_best_confirmation requires before it will re-measure
    a specific sha."""

    def push_ref(self, ref, branch):
        self.calls.append(("push_ref", ref, branch))


def _ctx_with_anchor(tmp_path, be, *, target_ms, anchor_ms, reanchor):
    ctx = _ctx(tmp_path, be, target_ms=target_ms)
    ctx.gate_window = "device"
    ctx.home_cluster = ""
    ctx.target_anchor_ms = anchor_ms
    ctx.target_anchor_cluster = ""
    ctx.reanchor_ref = reanchor
    return ctx


def test_apparent_target_from_a_lucky_node_draw_does_not_stop_the_run(tmp_path):
    """Student reads 1.85 against a 1.90 target — but the ref, re-measured this
    period, comes back 10% slower than the launch anchor, so the target restates
    to 2.09... and the student's own same-period re-measure (2.20) is short of it.
    The hit was the node, not the kernel: keep climbing."""
    reanchor_calls = []

    def reanchor():
        reanchor_calls.append(1)
        return ([2.20, 2.21], "")

    # iter 0 reads 1.85 (target met on paper); the same-period confirmation of
    # the student comes back 2.20 -> short of the re-anchored 2.09.
    be = PushingBackend(
        [REC("D1", 1.85), REC("D2", 1.85)],
        [EVAL(1.85), NCU,            # iter 0
         EVAL(2.20), EVAL(2.20),     # same-period student confirmation (2 jobs)
         EVAL(1.85), NCU],           # iter 1 keeps going
    )
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00,
                           reanchor=reanchor)
    ctx.max_iters = 2
    ctx.plateau_window = 10  # flat readings must not end the run as plateau

    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert reanchor_calls, "the apparent win must buy a fresh ref measurement"
    assert ctx.reanchor_note and "False win" in ctx.reanchor_note, \
        "the false win must be threaded into the next diagnosis"
    record = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert record["verdict"] == "false_win"
    assert record["check"] == "apparent-target"


def test_apparent_target_confirmed_same_period_still_stops(tmp_path):
    """Same shape, but this period's ref matches the launch anchor and the
    student's re-measure holds: a real win must still end the run."""
    def reanchor():
        return ([2.00, 2.01], "")

    be = PushingBackend(
        [REC("D1", 1.85)],
        [EVAL(1.85), NCU,           # iter 0
         EVAL(1.85), EVAL(1.86)],   # same-period student confirmation
    )
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00,
                           reanchor=reanchor)
    ctx.max_iters = 2

    assert run_inner_loop(ctx, start_iter=0) == "target"
    record = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert record["verdict"] == "target"


def test_target_still_stops_when_the_check_cannot_run(tmp_path):
    """A run with no re-anchor channel cannot verify anything, and one that can
    never verify must still be able to finish: stop, loudly, on the anchor."""
    be = FakeBackend([REC("D1", 1.85)], [EVAL(1.85), NCU])
    ctx = _ctx(tmp_path, be, target_ms=1.90)
    ctx.gate_window = "device"
    ctx.home_cluster = ""
    ctx.reanchor_ref = None          # no channel
    ctx.max_iters = 2

    assert run_inner_loop(ctx, start_iter=0) == "target"


# --- which currency the check is paid in ------------------------------------------
# The two readings an absolute hit compares were drawn on different nodes. On a
# run whose jobs carry the declared reference (ref_rides_job / ref_in_job), the
# same-job ratio is node- and period-matched and the cross-node absolute is never
# consulted; a ref_in_job run does not even buy the check — its absolute hit
# defers to the paired stop.


def test_a_ref_in_job_run_buys_no_cross_node_check_for_an_absolute_hit(tmp_path):
    reanchor_calls = []

    def reanchor():
        reanchor_calls.append(1)
        return ([2.00, 2.01], "")

    be = PushingBackend([REC("D1", 1.85), REC("D2", 1.85)],
                        [EVAL(1.85), NCU, EVAL(1.85), NCU])  # no in-job ref reading: paired stop inert
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00, reanchor=reanchor)
    ctx.ref_rides_job = False
    ctx.ref_in_job = True
    ctx.max_iters = 2
    ctx.plateau_window = 10

    assert run_inner_loop(ctx, start_iter=0) == "max_iters"   # 1.85 <= 1.90 on paper: not a stop
    assert reanchor_calls == [], "a ref_in_job run must not buy fresh-ref jobs for an absolute hit"
    assert not any(c[0] == "push_ref" for c in be.calls), "nor student re-measures"
    assert not (tmp_path / "reanchor.jsonl").exists()


def test_with_the_ref_riding_the_job_the_check_is_the_same_job_ratio(tmp_path):
    """The confirmation jobs carry the declared reference: the verdict is the
    median SAME-JOB ratio, no fresh-ref jobs are bought, and the record names the
    currency. Here both re-measures read the student 2-3% slower than the ref on
    the same card — a node draw, not a win."""
    reanchor_calls = []

    def reanchor():
        reanchor_calls.append(1)
        return ([2.00, 2.01], "")

    be = PushingBackend(
        [REC("D1", 1.85), REC("D2", 1.85)],
        [_EVAL_CONFIRM(1.85, 2.00), NCU,                        # iter 0: 1.85 <= 1.90 on paper
         _EVAL_CONFIRM(1.85, 1.80), _EVAL_CONFIRM(1.85, 1.81),  # same-card re-measure: ratio < 1
         _EVAL_CONFIRM(1.85, 2.00), NCU])                       # iter 1 keeps going
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00, reanchor=reanchor)
    ctx.ref_rides_job = True
    ctx.max_iters = 2
    ctx.plateau_window = 10

    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert reanchor_calls == [], "the ref rode the job: no fresh-ref jobs"
    record = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert record["verdict"] == "false_win" and record["currency"] == "same-job"
    assert record["median_ratio"] < 1.0
    assert "same-card re-measure" in ctx.reanchor_note


def test_with_the_ref_riding_the_job_a_confirmed_ratio_still_stops(tmp_path):
    be = PushingBackend(
        [REC("D1", 1.85)],
        [_EVAL_CONFIRM(1.85, 2.00), NCU,
         _EVAL_CONFIRM(1.85, 1.90), _EVAL_CONFIRM(1.85, 1.92)])  # ratios 1.027 / 1.038 >= 1.0
    # a fresh ref that would FAIL an absolute check — proving it is not consulted
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00,
                           reanchor=lambda: ([9.0, 9.0], ""))
    ctx.ref_rides_job = True
    ctx.max_iters = 2

    assert run_inner_loop(ctx, start_iter=0) == "target"
    record = json.loads((tmp_path / "reanchor.jsonl").read_text().splitlines()[-1])
    assert record["verdict"] == "target" and record["currency"] == "same-job"


# --- what a false win remembers ---------------------------------------------------


def test_a_new_best_still_short_of_the_fresh_target_is_not_re_bought(tmp_path):
    """After a false win the TRANSPORTED target is memoised (1.52 here: this
    period's nodes are fast). A later KEEP at 1.80 is under the stale 1.90 but
    still over 1.52, so the answer is already known — no ref jobs, no student
    jobs, the residual threaded into the next diagnosis."""
    reanchor_calls = []

    def reanchor():
        reanchor_calls.append(1)
        return ([1.60, 1.61], "")          # drift x0.8 -> fresh target 1.52

    be = PushingBackend(
        [REC("D1", 1.85), REC("D2", 1.80)],
        [EVAL(1.85), NCU,            # iter 0
         EVAL(1.85), EVAL(1.86),     # confirmation: 1.855 > 1.52 -> false win
         EVAL(1.80), NCU])           # iter 1: a new best, still > 1.52
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00, reanchor=reanchor)
    ctx.ref_rides_job = False
    ctx.max_iters = 2
    ctx.plateau_window = 10

    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    assert reanchor_calls == [1], "the second hit must re-use the first check"
    assert be.gpu_logs == [], "nothing bought for the second hit"
    assert "carried over" in ctx.reanchor_note and ctx.reanchor_memo[0] == 1.80
    verdicts = [json.loads(l)["verdict"]
                for l in (tmp_path / "reanchor.jsonl").read_text().splitlines()]
    assert verdicts == ["false_win"]


def test_the_false_win_note_is_replayed_after_a_meta_entry_consumed_it(tmp_path):
    """run_two_layer clears ctx.reanchor_note once it has threaded it into a meta
    entry; _phantom_gap_reanchor returns before its memo replay on gap <= 0. With
    the best unchanged, the next check re-threads the same note — buying nothing."""
    reanchor_calls = []

    def reanchor():
        reanchor_calls.append(1)
        return ([2.20, 2.21], "")

    be = PushingBackend([REC("D1", 1.85)],
                        [EVAL(1.85), NCU, EVAL(2.20), EVAL(2.20)])
    ctx = _ctx_with_anchor(tmp_path, be, target_ms=1.90, anchor_ms=2.00, reanchor=reanchor)
    ctx.ref_rides_job = False
    ctx.max_iters = 1
    ctx.plateau_window = 10

    assert run_inner_loop(ctx, start_iter=0) == "max_iters"
    note = ctx.reanchor_note
    assert note and "False win" in note
    ctx.reanchor_note = None                                # the meta entry consumed it
    assert _confirm_apparent_target(ctx, 1.85, 1) is False
    assert ctx.reanchor_note == note                        # re-threaded from the memo
    assert reanchor_calls == [1] and be.gpu_logs == []      # nothing re-bought
