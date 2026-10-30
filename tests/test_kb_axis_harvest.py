"""KEEPs that move a SASS contract axis owe a KB write-up (kb_axis_debt).

The kb-tripwire bills probe fights; nothing billed a one-round KEEP that
changed the built structure — the channel the cp.async cache-mode line fell
through twice. These tests pin the whole path: the census summary parked by
the NCU job lands on the IterationRecord (not in the flat profiler digest),
consecutive censused KEEPs are diffed axis by axis, the harvest turn fires on
axis debt alone, and the profile job mode switches to the vendored runner
only when it is actually in the workspace."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import EVAL, REC, FakeBackend, _ctx  # noqa: E402

from forgekernel.inner.trajectory import IterationRecord, append_record, load_records  # noqa: E402
from forgekernel.loop.driver import (  # noqa: E402
    _ORCH_CENSUS_CMD, _ncu_mode, _ncu_profiler, _pop_orch_summary, kb_axis_debt,
    run_inner_iteration, run_kb_harvest,
)

_CSV = (Path(__file__).parent / "fixtures" / "ncu_sample.csv").read_text()
_BASE = {"loops": {"copy-producer": 1, "mma-consumer": 1}, "mbar_objs": 2,
         "cache_ops": ["LDGSTS.E.128"], "commit": [], "verdict": "ORCH-OK"}
_BYPASS = dict(_BASE, cache_ops=["LDGSTS.E.BYPASS.128"])


def _census_log(summary: dict) -> str:
    import json
    return (_CSV + "\nORCH-CENSUS cubin=k.cubin kernel=k instrs=900\n"
            "ORCH-SUMMARY-JSON: " + json.dumps(summary, sort_keys=True) + "\n")


def _keep(ctx, did, ms, summary=None, sha="s", change="c"):
    append_record(ctx.status_path, IterationRecord(
        iter=len(load_records(ctx.status_path)), decision_id=did, change=change,
        correctness=True, verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
        sha=sha, profiler={}, orch_summary=summary or {}))


def test_ncu_profiler_parks_census_summary_off_the_digest():
    ctx = SimpleNamespace(backend=FakeBackend([], [_census_log(_BASE), _CSV]),
                          ncu_retries=0, orch_census=False, workspace="/nonexistent")
    prof = _ncu_profiler(ctx, "b1", correctness=True)
    summ = _pop_orch_summary(prof)
    assert summ == _BASE and "_orch_summary" not in prof
    assert prof["regs_per_thread"] == 168          # the CSV parse is untouched
    prof2 = _ncu_profiler(ctx, "b2", correctness=True)
    assert _pop_orch_summary(prof2) == {} and "regs_per_thread" in prof2


def test_ncu_mode_switches_only_when_the_runner_is_seeded(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert ctx.orch_census is True
    assert _ncu_mode(ctx) == "ncu"                 # nothing vendored in this workspace
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "orch_census_run.py").write_text("# vendored")
    assert _ncu_mode(ctx) == _ORCH_CENSUS_CMD
    ctx.orch_census = False
    assert _ncu_mode(ctx) == "ncu"


def test_iteration_records_summary_and_logs_the_moved_axis(tmp_path, capsys):
    be = FakeBackend(
        [REC("D1.2", 3.07), REC("D1.3", 2.95)],
        [EVAL(3.07), _census_log(_BASE), EVAL(2.95), _census_log(_BYPASS)],
    )
    ctx = _ctx(tmp_path, be)
    run_inner_iteration(ctx, 0, metric_before=3.2)
    run_inner_iteration(ctx, 1, metric_before=3.07)
    recs = load_records(ctx.status_path)
    assert recs[0].orch_summary == _BASE and recs[1].orch_summary == _BYPASS
    assert "_orch_summary" not in recs[1].profiler
    debt = kb_axis_debt(recs, ctx.gate_window)
    assert len(debt) == 1
    b = debt[0]
    assert b.iter == 1 and b.decision_id == "D1.3" and b.prev_iter == 0
    assert b.axes == {"cache_ops": {"before": ["LDGSTS.E.128"], "after": ["LDGSTS.E.BYPASS.128"]}}
    assert b.ms_before == 3.07 and b.ms_after == 2.95 and b.delta_pct == -3.91
    out = capsys.readouterr().out
    assert "kb-axis: D1.3 moved SASS contract axes ['cache_ops']" in out
    assert "-3.91%" in out


def test_axis_debt_is_between_nearest_censused_keeps_only(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _keep(ctx, "D0.1", 3.5, _BASE)               # baseline: nothing to compare
    _keep(ctx, "D0.2", 3.4)                      # census skipped: transparent
    append_record(ctx.status_path, IterationRecord(  # a REVERT never bills
        iter=2, decision_id="D0.3", change="c", correctness=False, verdict="REVERT",
        orch_summary=_BYPASS))
    _keep(ctx, "D0.4", 3.3, _BASE)               # same axes as D0.1: no move
    _keep(ctx, "D0.5", 3.0, _BYPASS, sha="abc")  # the move
    _keep(ctx, "D0.6", 3.0, _BYPASS)             # no further move
    debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    assert [(b.iter, b.decision_id, b.prev_iter, sorted(b.axes)) for b in debt] == [
        (4, "D0.5", 3, ["cache_ops"])]
    assert debt[0].ms_before == 3.3 and debt[0].sha == "abc"


def test_harvest_fires_on_axis_debt_alone_with_the_evidence_block(tmp_path):
    be = FakeBackend(["RECIPE_RECORDED: hw_kb/cutedsl/sm90/cp_async.md"], [])
    ctx = _ctx(tmp_path, be)
    ctx.kb_tripwire_rounds = 6                   # no probe fights at all
    _keep(ctx, "D1.2", 3.072, _BASE, sha="aaaaaaaaaa1")
    _keep(ctx, "D1.3", 2.951, _BYPASS, sha="61f555b46b3",
          change="flipped CopyG2SOp to cache_mode=LoadCacheMode.GLOBAL")
    assert run_kb_harvest(ctx) == ["hw_kb/cutedsl/sm90/cp_async.md"]
    assert [c for c in be.calls if c[0] == "gpu"] == []
    prompt = be.prompts[0]
    assert "## Contract-axis change list" in prompt and "## Probe battle list" not in prompt
    assert "iter 1 D1.3 (sha 61f555b46b" in prompt
    assert "`cache_ops`" in prompt and '"LDGSTS.E.BYPASS.128"' in prompt
    assert "3.072 -> 2.951 ms (-3.94%)" in prompt
    assert "LoadCacheMode.GLOBAL" in prompt
    # the prompt template itself tells the teacher how to write this category
    assert "How to write contract-axis change items" in prompt


def test_harvest_stays_silent_without_any_debt(tmp_path):
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    _keep(ctx, "D1.2", 3.0, _BASE)
    _keep(ctx, "D1.3", 2.9, _BASE)
    assert run_kb_harvest(ctx) == [] and be.calls == []


def test_axis_debt_ignores_counting_axes_and_l2_hints_but_names_them(tmp_path):
    """The bill triggers on the count-free contract signature (what the
    axis harvest bills on), so codegen noise — mbarrier object counts,
    per-role counts, a `.LTC256B` prefetch hint a probe priced at ~0 —
    never becomes a recipe request; the raw keys that moved ride as a note."""
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _keep(ctx, "D0.1", 3.5, _BASE)
    noisy = dict(_BASE, mbar_objs=5, mbar_per_role={"mma-consumer": 3},
                 cache_ops=["LDGSTS.E.LTC256B.128"], itc_candidates=7)
    _keep(ctx, "D0.2", 3.5, noisy)                # codegen moved, contract did not
    real = dict(noisy, cache_ops=["LDGSTS.E.BYPASS.LTC256B.128"], mbar_objs=6)
    _keep(ctx, "D0.3", 3.0, real)                 # the .BYPASS bit: a contract move
    debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    assert [(b.iter, sorted(b.axes)) for b in debt] == [(2, ["cache_ops"])]
    b = debt[0]
    assert b.axes["cache_ops"] == {"before": ["LDGSTS.E.LTC256B.128"],
                                   "after": ["LDGSTS.E.BYPASS.LTC256B.128"]}
    assert b.noise_axes == ["mbar_objs"]
    from forgekernel.loop.driver import _axis_battle_block
    block = _axis_battle_block(b)
    assert "Counting axes that moved at the same time" in block and "mbar_objs" in block


def test_axis_debt_pairs_a_keep_with_its_lineage_not_its_neighbour(tmp_path):
    """After a meta reopen the next KEEP is built on an earlier tree; diffing
    it against the rolled-back KEEP would bill the rollback to its change."""
    ctx = _ctx(tmp_path, FakeBackend([], []))

    def keep(did, ms, summ, sha, base):
        append_record(ctx.status_path, IterationRecord(
            iter=len(load_records(ctx.status_path)), decision_id=did, change="c",
            correctness=True, verdict="KEEP", bwd_median_ms=ms, device_median_ms=ms,
            sha=sha, base_sha=base, profiler={}, orch_summary=summ))
    keep("D0.1", 3.5, _BASE, "a1", "root")
    keep("D0.2", 3.4, _BYPASS, "a2", "a1")         # the real move, on a1
    # meta reopen: workspace reset to a1, the next KEEP is built on a1 again
    keep("D0.3", 3.3, _BASE, "a3", "a1")           # same contract as its base: no move
    keep("D0.4", 3.0, _BYPASS, "a4", "a3")         # the move again, on a3
    debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    assert [(b.iter, b.prev_iter, sorted(b.axes)) for b in debt] == [
        (1, 0, ["cache_ops"]), (3, 2, ["cache_ops"])]
    assert debt[1].ms_before == 3.3               # priced against its own base, not D0.2
    # a base outside the records (baseline / meta-patched tree): nothing to compare
    keep("D0.5", 2.9, _BYPASS, "a5", "unknown")
    assert len(kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)) == 2
    # an uncensused KEEP in the chain is transparent
    append_record(ctx.status_path, IterationRecord(
        iter=5, decision_id="D0.6", change="c", correctness=True, verdict="KEEP",
        bwd_median_ms=2.9, device_median_ms=2.9, sha="a6", base_sha="a4"))
    keep("D0.7", 2.8, _BASE, "a7", "a6")
    debt = kb_axis_debt(load_records(ctx.status_path), ctx.gate_window)
    assert (debt[-1].iter, debt[-1].prev_iter) == (6, 3)


def test_iteration_records_the_base_it_was_built_on(tmp_path):
    be = FakeBackend([REC("D1.2", 3.07)], [EVAL(3.07), _census_log(_BASE)])
    ctx = _ctx(tmp_path, be)
    base = be.head()
    run_inner_iteration(ctx, 0, metric_before=3.2)
    rec = load_records(ctx.status_path)[0]
    assert rec.base_sha == base and rec.sha and rec.sha != base


def test_census_lines_after_the_csv_are_stripped_before_the_parse():
    """The runner prints the census AFTER the NCU CSV (a job's archived log may
    keep only the tail); the JSON line has enough commas to read as a CSV row."""
    import json
    from forgekernel.loop.driver import _without_census_lines
    log = (_CSV + "\n[orch-census] 3 cubin(s) dumped\n"
           "ORCH-CENSUS cubin=k.cubin kernel=k instrs=900\n"
           "ORCH-SUMMARY-JSON: " + json.dumps(dict(_BASE, edges=[["a", "b"], ["c", "d"]]),
                                              sort_keys=True) + "\n")
    stripped = _without_census_lines(log)
    assert "ORCH-" not in stripped and "[orch-census]" not in stripped
    assert stripped.strip() == _CSV.strip()
    ctx = SimpleNamespace(backend=FakeBackend([], [log]), ncu_retries=0,
                          orch_census=False, workspace="/nonexistent")
    prof = _ncu_profiler(ctx, "b1", correctness=True)
    assert _pop_orch_summary(prof)["edges"] == [["a", "b"], ["c", "d"]]
    assert prof["regs_per_thread"] == 168
    assert not any("ORCH" in str(v) for v in prof.values())
