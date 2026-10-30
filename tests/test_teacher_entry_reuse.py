"""A teacher entry does not pay again for a reading it already has.

In one large batch of runs the loop was strictly serial and the
teacher side was over half of it. Three of its costs were repeats: the stall
pricing job re-run on a best KEEP that had not moved (553 of 1026 jobs), the
entry timeline re-probing a sha whose table was banked minutes earlier (1672 of
1887), and the INFEASIBLE-mention bounce firing on diagnoses whose own
reachability line read OPEN, where the gate refuses a stop anyway (1157 of 1163;
695 teacher turns).
"""
import json

from forgekernel.inner.trajectory import IterationRecord, append_record
from forgekernel.loop import driver
from forgekernel.loop.driver import (
    _STALL_PRICE_LOGS,
    _TIMELINE_CMD,
    _run_stall_pricing,
    _run_timeline,
    run_meta_step,
)
from test_loop_driver import (
    _ORCH_PRICING_BLOCK,
    _STALL_LOG_OK,
    FakeBackend,
    _ctx,
    _patching,
    _stall_ctx,
)
from test_timeline_driver import TL_LOG, _banked, _tl_ctx


class _RefPushing(FakeBackend):
    def push_ref(self, ref, branch):
        self.calls.append(("push_ref", ref, branch))


def _gpu_calls(be, mode=None):
    return [c for c in be.calls if c[0] == "gpu" and (mode is None or c[1] == mode)]


def _with_best(ctx, sha="shabest"):
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D1", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=2.0, device_median_ms=2.0, sha=sha))


# --- stall pricing ---------------------------------------------------------------

def test_a_second_entry_on_the_same_best_reuses_the_whole_reading(tmp_path, capsys):
    be = _RefPushing([], [_STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    ctx.evidence_dir = tmp_path / "evidence"
    _with_best(ctx)
    first = _run_stall_pricing(ctx)
    dossier = ctx.evidence_dir / "student_orch_dossier.txt"
    dossier.unlink()
    second = _run_stall_pricing(ctx)
    assert first == second and first[0] is not None and first[2] is not None
    assert len(_gpu_calls(be)) == 1
    assert dossier.is_file(), "a reused reading is still banked"
    assert "reusing the pricing of shabest" in capsys.readouterr().out


def test_a_partial_reading_is_priced_afresh(tmp_path):
    # the first job came back without its dossier: nothing to reuse
    be = _RefPushing([], [_STALL_LOG_OK, _STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    _with_best(ctx)
    assert _run_stall_pricing(ctx)[2] is None
    assert _run_stall_pricing(ctx)[2] is not None
    assert len(_gpu_calls(be)) == 2


def test_a_moved_best_or_another_run_is_priced_afresh(tmp_path):
    be = _RefPushing([], [_STALL_LOG_OK + _ORCH_PRICING_BLOCK] * 3)
    ctx = _stall_ctx(tmp_path, be)
    _with_best(ctx)
    _run_stall_pricing(ctx)
    _run_stall_pricing(ctx, ref="shanew")
    other = tmp_path / "other"
    other.mkdir()
    ctx2 = _stall_ctx(other, be)
    _with_best(ctx2)
    _run_stall_pricing(ctx2)
    assert [c[1] for c in be.calls if c[0] == "push_ref"] == ["shabest", "shanew", "shabest"]
    assert len(_gpu_calls(be)) == 3


def test_the_cache_holds_only_whole_readings_keyed_by_run(tmp_path):
    be = _RefPushing([], [_STALL_LOG_OK + _ORCH_PRICING_BLOCK])
    ctx = _stall_ctx(tmp_path, be)
    _with_best(ctx)
    _run_stall_pricing(ctx)
    keys = [k for k in _STALL_PRICE_LOGS if k[0] == str(ctx.status_path)]
    assert len(keys) == 1 and keys[0][1] == "shabest"


# --- timeline ----------------------------------------------------------------------

def test_an_entry_on_a_probed_sha_reuses_the_banked_table(tmp_path, capsys):
    be = _RefPushing([], [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    first = _run_timeline(ctx, "shabest", "iter3:gate row")
    second = _run_timeline(ctx, "shabest", "plateau")
    assert first["roles"]["state"]["wait"] == second["roles"]["state"]["wait"] == 1790
    assert len(_gpu_calls(be, _TIMELINE_CMD)) == 1
    assert len(_banked(ctx)) == 1 and _banked(ctx)[0]["probe"]
    assert "reusing the banked table (iter3:gate row" in capsys.readouterr().out


def test_a_table_banked_by_another_probe_version_is_not_reused(tmp_path):
    be = _RefPushing([], [TL_LOG, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    _run_timeline(ctx, "shabest", "plateau")
    p = ctx.evidence_dir / "timelines.jsonl"
    recs = _banked(ctx)
    recs[0].pop("probe")                         # banked before the field existed
    p.write_text(json.dumps(recs[0]) + "\n", encoding="utf-8")
    _run_timeline(ctx, "shabest", "plateau")
    assert len(_gpu_calls(be, _TIMELINE_CMD)) == 2


def test_an_unusable_table_or_a_student_request_runs_the_probe(tmp_path):
    fail = "TIMELINE-FAIL reason=markers (kernel carries no `def _seg(` marker definition)\n"
    be = _RefPushing([], [fail, TL_LOG, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    _run_timeline(ctx, "shabest", "plateau")     # failed closed: banked, not usable
    _run_timeline(ctx, "shabest", "plateau")     # so this one probes
    _run_timeline(ctx, "shabest", "iter4:student", branch="fs-demo-probe")
    assert len(_gpu_calls(be, _TIMELINE_CMD)) == 3


def test_a_second_meta_entry_on_the_same_best_runs_no_timeline_job(tmp_path):
    diag = ("cause_class: R2\ndeficient_locus: D6\npatch_direction: add decision\n"
            "locus_segment: state.C.wait_kk\n")
    be = _RefPushing([diag, _patching(tmp_path, "\n### D6 added\nx\n",
                                      "patch_class: additive\nchanged_entries: []\n"),
                      diag, _patching(tmp_path, "\n### D7 added\nx\n",
                                      "patch_class: additive\nchanged_entries: []\n")],
                     [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    _with_best(ctx)
    run_meta_step(ctx)
    run_meta_step(ctx)
    assert len(_gpu_calls(be, _TIMELINE_CMD)) == 1
    assert "## Timeline" in be.prompts[2]          # the second entry still renders the table


# --- the INFEASIBLE-mention bounce ---------------------------------------------------

def _mention(reach):
    return (f"reachability: {reach}\n"
            "cause_class: R2\ndeficient_locus: D6\n"
            "evidence: floor sits inside the target, so the INFEASIBLE stop is not available\n"
            "patch_direction: add decision\n")


def test_a_mention_on_an_open_line_is_not_bounced(tmp_path):
    be = FakeBackend([_mention("floor_opt=1.20 ms vs target=1.9 ms -> OPEN"),
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    res = run_meta_step(_ctx(tmp_path, be))
    assert res.patch_class == "additive"
    assert len(be.prompts) == 2
    assert not any("## INFEASIBLE has only one channel" in p for p in be.prompts)


def test_a_mention_on_a_blocked_line_is_still_bounced(tmp_path):
    # BLOCKED under a patchable class is the spin shape the bounce exists for
    blocked = _mention("floor_opt=2.40 ms vs target=1.9 ms -> BLOCKED")
    be = FakeBackend([blocked, blocked,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")], [])
    run_meta_step(_ctx(tmp_path, be))
    assert sum("## INFEASIBLE has only one channel" in p for p in be.prompts) == 1


def test_the_open_skip_reads_the_verdict_the_gate_reads():
    # the skip and the gate must never disagree on what OPEN is
    for line, verdict in (("floor_opt=1 ms vs target=2 ms -> OPEN", "OPEN"),
                          ("floor_opt=3 ms vs target=2 ms → BLOCKED", "BLOCKED"),
                          ("OPEN or BLOCKED, see evidence", "")):
        assert driver._parse_reach_verdict(line) == verdict
