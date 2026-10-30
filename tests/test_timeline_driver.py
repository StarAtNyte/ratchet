"""The timeline probe in the driver: when it runs (teacher entry, gate rows,
new bests beyond the margin, on request), what it hands the teacher and the
student, and the locus_segment line it makes the diagnosis carry."""
import json

from forgekernel.loop.driver import (
    _TIMELINE_CMD,
    _student_prompt,
    _timeline_on,
    run_inner_iteration,
    run_meta_step,
)
from test_loop_driver import _DIAG, _MILESTONE, EVAL, NCU, REC, FakeBackend, _ctx, _patching

TL_LOG = (
    "TIMELINE-SITES 5 0:aux.entry 1:state.A.top 2:state.C.wait_kk.begin 3:state.C.wait_kk.end 4:aux.c.kk.commit\n"
    "TL-ROLE role=state warps=4 rows=5 period=6614 iqr=0.077 busy=3093 wait=1790 wrap=1731\n"
    "TL-SEG role=state from=state.A.top to=state.C.wait_kk.begin kind=busy median=1200 p25=1180 p75=1230 n=40\n"
    "TL-SEG role=state from=state.C.wait_kk.begin to=state.C.wait_kk.end kind=wait median=1334 p25=1300 p75=1360 n=40\n"
    "TL-SEG role=state from=state.C.wait_kk.end to=state.A.top_(next) kind=wrap median=4080 p25=4000 p75=4100 n=40\n"
    "TL-ROLE role=aux warps=2 rows=5 period=6614 iqr=0.079 busy=4998 wait=949\n"
    "TL-SEG role=aux from=aux.a.top to=aux.c.kk.commit kind=busy median=2300 p25=2200 p75=2400 n=20\n"
    "TL-PHASE role=aux site=aux.c.kk.commit offset=1283\n"
    "TL-PHASE role=state site=state.C.wait_kk.end offset=1334\n"
    "TIMELINE mode=all sites=5 rows=5 slots=2 numeric=identical perturbation=1.057 span_ratio=0.98 "
    "mono_bad=0/80 sm_mhz=1590 base_ms=0.5255 stamped_ms=0.5552 sm_src=sampled\n"
    "TIMELINE-WARN perturbation 1.0570 > 1.03 even in mode=all — absolute segment lengths carry that overhead\n"
    "TIMELINE-OK mode=all\n")
TL_FAIL = "TIMELINE-FAIL reason=markers (kernel carries no `def _seg(` marker definition)\n"
DIAG_SEG = _DIAG + "\nlocus_segment: state.C.wait_kk\n"
KF = "flash_attn_dsl/flash_bwd.py"


def _tl_ctx(tmp_path, be, mode="on"):
    ctx = _ctx(tmp_path, be)
    ctx.timeline_probe = mode
    ctx.evidence_dir = tmp_path / "evidence"
    return ctx


def _banked(ctx):
    p = ctx.evidence_dir / "timelines.jsonl"
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if p.is_file() else []


def test_timeline_mode_off_on_auto(tmp_path):
    ctx = _tl_ctx(tmp_path, FakeBackend([], []), mode="off")
    assert not _timeline_on(ctx)
    ctx.timeline_probe = "on"
    assert _timeline_on(ctx)
    ctx.timeline_probe = "auto"
    assert not _timeline_on(ctx)
    ctx.milestone.write_text(_MILESTONE + "\nbottleneck: latency\n", encoding="utf-8")
    assert _timeline_on(ctx)


def test_meta_entry_runs_the_timeline_renders_it_first_and_demands_a_locus_segment(tmp_path):
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
    (ctx.evidence_dir / "ref_role_exposure.txt").write_text(
        "[parse] page carried 8 launches; using launch 0\n"
        "[tensor] active=12.8% source=pm-plateau duration_ns=410784.0 exposed_frac=0.872\n"
        "REGION r26 mma-consumer  [0x1..0x2]   630 instrs share=47.8 stalls=barrier:33\n", encoding="utf-8")
    run_meta_step(ctx)
    assert ("gpu", _TIMELINE_CMD) in be.calls
    first = be.prompts[0]
    assert "## Timeline (measured by the driver with %clock64 stamps on the student tree" in first
    assert "| `state.C.wait_kk.begin` → `state.C.wait_kk.end` | wait | 1334 |" in first
    assert "| state | 4 | 6614 | 3093 | 1790 | 1731 |" in first and "| aux | 2 | 6614 | 4998 | 949 | n/a |" in first
    assert "| aux | `aux.c.kk.commit` | 1283 |" in first and "busy + wait + wrap ≈ period" in first
    assert "REGION r26 mma-consumer" in first and "locus_segment: <role>.<segment>" in first
    assert "instrumentation stretched device time by 5.7%" in first and "Probe self-check note**: perturbation 1.0570" in first
    assert "SM 1590.0 MHz (sampled)" in first
    # placed after the NCU truth + reflection the prompt always opens with, ahead of the stall ledgers
    assert first.index("## NCU hardware ground truth") < first.index("## Timeline")
    # no locus_segment -> one bounce, then the diagnosis with it goes through to the patch
    assert "lacks a `locus_segment:` line" in be.prompts[1] and len(be.prompts) == 3
    recs = _banked(ctx)
    assert len(recs) == 1 and recs[0]["ok"] and recs[0]["label"] == "plateau" and recs[0]["ms"] == 0.5255
    assert recs[0]["table"]["roles"]["state"]["wait"] == 1790


def test_meta_entry_without_the_probe_or_with_a_failed_one_asks_for_no_segment(tmp_path):
    # off: no job, no bounce
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")], [])
    ctx = _tl_ctx(tmp_path, be, mode="off")
    run_meta_step(ctx)
    assert ("gpu", _TIMELINE_CMD) not in be.calls and "## Timeline" not in be.prompts[0] and len(be.prompts) == 2
    # on but the probe failed closed: banked as not ok, not rendered, no bounce
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")], [TL_FAIL])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert ("gpu", _TIMELINE_CMD) in be.calls and "## Timeline" not in be.prompts[0] and len(be.prompts) == 2
    recs = _banked(ctx)
    assert len(recs) == 1 and recs[0]["ok"] is False and recs[0]["usable"] is False \
        and "markers" in recs[0]["table"]["fail"]


# A probe that read the dominant kernel of a multi-kernel tree and not the
# small launches around it: span below the band, so the probe's own verdict was
# FAIL — but the table is real (four warpgroups, per-segment cycles,
# who-waits-on-whom). A GDN bwd run read 0.60-0.65 on 16 of its 17
# probes against a tree whose dominant kernel is 69% of device time, every one
# banking a complete K3 table, and every one rendered nowhere: its three meta
# entries diagnosed from aggregate NCU ratios alone — the failure mode the
# probe exists to prevent. Below the band is a partial table and it is shown.
TL_PARTIAL = (
    "TL-ROLE role=state warps=4 rows=5 period=6614 iqr=0.077 busy=3093 wait=1790 wrap=1731\n"
    "TL-SEG role=state from=state.A.top to=state.C.wait_kk.begin kind=busy "
    "median=1200 p25=1180 p75=1230 n=40\n"
    "TL-SEG role=state from=state.C.wait_kk.begin to=state.C.wait_kk.end kind=wait "
    "median=1334 p25=1300 p75=1360 n=40\n"
    "TIMELINE mode=waits sites=72 rows=26 slots=8 numeric=identical perturbation=1.01 "
    "span_ratio=0.621 mono_bad=0/200 sm_mhz=1590 base_ms=0.8255 stamped_ms=0.84 sm_src=sampled\n"
    "TIMELINE-WARN span_ratio 0.621 below [0.75, 1.25] — short\n"
    "TIMELINE-COVERAGE partial\n"
    "TIMELINE-OK mode=waits\n")


def test_a_short_coverage_table_is_rendered_not_dropped(tmp_path):
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_PARTIAL])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    first = be.prompts[0]
    # the table is in the entry, with its numbers
    assert "## Timeline" in first
    assert "| state | 4 | 6614 | 3093 | 1790 | 1731 |" in first
    # and it says, in the rendered table, what the span can say: the stamps
    # bracket 0.621 of the window in this mode, the segments read as they are,
    # the rows do not add up to the call (NOT "the numbers are short": that
    # run's mode=all / mode=waits tables read 0.947 vs 0.60 with the same
    # per-row periods)
    assert "Coverage note (partial)" in first and "0.621 of device time (mode=waits)" in first
    assert "cycles are in-row stamp differences; read the absolute values as they are" in first and "do not inflate them by 1/coverage" in first
    assert "smaller by the coverage ratio" not in first and "main kernel" not in first
    # it drives the same locus demand a full table does
    assert "lacks a `locus_segment:` line" in be.prompts[1] and len(be.prompts) == 3
    recs = _banked(ctx)
    assert len(recs) == 1
    assert recs[0]["ok"] is True and recs[0]["usable"] is True and recs[0]["coverage"] == "partial"
    assert recs[0]["table"]["roles"]["state"]["wait"] == 1790


def test_a_short_coverage_table_is_logged_ok_not_fail(tmp_path, capsys):
    """The log line is the operator's only view of a probe that reads short.

    It used to read ``tl["ok"]``, which a partial table never has — the probe
    stops at TIMELINE-COVERAGE instead of TIMELINE-OK — so every such run logged
    ``FAIL — None`` (the reason is None: nothing failed) and the "read short"
    annotation beside it was unreachable. This is the assertion whose absence let
    that through: the render test above checks the prompt, not the log."""
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_PARTIAL])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    lines = [l for l in capsys.readouterr().out.splitlines() if "timeline[plateau]" in l]
    assert len(lines) == 1
    assert "FAIL" not in lines[0] and "OK" in lines[0] and "rows do not add up to the call" in lines[0]
    assert "span=0.621" in lines[0]


def test_everything_that_reads_a_table_asks_the_same_question():
    """One rule, one place. ``table_usable`` decides; ``parse_timeline`` banks its
    answer and the driver reads that field back. Re-deriving it in the driver is
    how the retrieval, the log line and the banked record came to disagree."""
    from forgekernel.loop.driver import _timeline_renderable, _timeline_short_coverage
    from forgekernel.skills.timeline_probe import parse_timeline, table_usable

    def tl_from(log):
        return parse_timeline(log)

    partial = tl_from(TL_PARTIAL)
    over = tl_from(TL_PARTIAL.replace("span_ratio=0.621", "span_ratio=1.9")
                             .replace("TIMELINE-COVERAGE partial\n", "")
                             .replace("TIMELINE-OK mode=waits\n",
                                      "TIMELINE-FAIL reason=span outside [0.75, 1.25]\n"))
    full = tl_from(TL_LOG)
    failed = tl_from(TL_FAIL)
    # the parse-time answer and the driver's re-read agree on all four
    for tl in (partial, over, full, failed):
        assert tl["usable"] == _timeline_renderable(tl), tl.get("coverage")
    # and the rule itself is the parser's, not the driver's
    assert table_usable(roles=partial["roles"], ok=partial["ok"], coverage="partial",
                        span_ratio=0.621, sm_src="sampled") is True
    assert partial["usable"] is True and full["usable"] is True
    assert over["usable"] is False and failed["usable"] is False
    # short-coverage is named for the caveat, on the partial table and on a
    # pre-TIMELINE-COVERAGE log read through on its span (ok false, unknown class)
    assert _timeline_short_coverage(partial) is True
    legacy = dict(partial, ok=False, coverage="unknown")
    assert _timeline_renderable(legacy) is True and _timeline_short_coverage(legacy) is True
    assert _timeline_short_coverage(full) is False and _timeline_short_coverage(failed) is False


def test_the_student_note_carries_the_shortage_caveat_too(tmp_path):
    """The student sizes its own segments off these numbers. The teacher's copy
    says the absolutes read short; the student's copy is the same table and must
    say it too."""
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_PARTIAL])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    from forgekernel.loop.driver import _timeline_student_note
    note = _timeline_student_note(ctx)
    assert "Latest table" in note and "Coverage note" in note and "read the absolute values as they are" in note and "mode=waits" in note
    # a full-coverage table does not carry it
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    be2 = FakeBackend([_DIAG, DIAG_SEG,
                       _patching(full_dir, "\n### D6 added\nx\n",
                                 "patch_class: additive\nchanged_entries: []\n")], [TL_LOG])
    ctx2 = _tl_ctx(full_dir, be2)
    run_meta_step(ctx2)
    assert "Latest table" in _timeline_student_note(ctx2)
    assert "Coverage note" not in _timeline_student_note(ctx2)


def test_an_over_spanning_table_is_still_dropped(tmp_path):
    """Above the band the slab holds more than one launch, so the segments mix
    trees: a wrong table, which is worse than none. That stays a failure."""
    over = TL_PARTIAL.replace("span_ratio=0.621", "span_ratio=1.9") \
                     .replace("0.621 below [0.75, 1.25] — short", "1.900 above [0.75, 1.25]") \
                     .replace("TIMELINE-COVERAGE partial\n", "") \
                     .replace("TIMELINE-OK mode=waits\n",
                              "TIMELINE-FAIL reason=span_ratio 1.900 outside [0.75, 1.25]\n")
    be = FakeBackend([_DIAG, _patching(tmp_path, "\n### D6 added\nx\n",
                                       "patch_class: additive\nchanged_entries: []\n")], [over])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert "## Timeline" not in be.prompts[0]
    recs = _banked(ctx)
    assert len(recs) == 1 and recs[0]["usable"] is False


def test_teacher_can_request_a_timeline_as_a_verdict_probe(tmp_path):
    from forgekernel.loop.driver import _is_timeline_request
    assert _is_timeline_request("`timeline`") and _is_timeline_request("notes/timeline.") \
        and _is_timeline_request("TIMELINE") and not _is_timeline_request("notes/timeline_x.py")
    # the entry already read the tree: the request re-uses that table, no second job
    be = FakeBackend(["PROBE_REQUEST: `timeline`", DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    ctx.max_verdict_probe_rounds = 2
    run_meta_step(ctx)
    assert [c for c in be.calls if c == ("gpu", _TIMELINE_CMD)] == [("gpu", _TIMELINE_CMD)]
    assert "not re-run" in be.prompts[1] and [r["label"] for r in _banked(ctx)] == ["plateau"]
    # the entry's probe failed: the request runs it (a verdict-probe round), the block lands
    w2 = tmp_path / "w2"
    w2.mkdir()
    be = FakeBackend(["PROBE_REQUEST: timeline", DIAG_SEG,
                      _patching(w2, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_FAIL, TL_LOG])
    ctx = _tl_ctx(w2, be)
    ctx.max_verdict_probe_rounds = 2
    run_meta_step(ctx)
    assert [c for c in be.calls if c == ("gpu", _TIMELINE_CMD)] == [("gpu", _TIMELINE_CMD)] * 2
    assert "## Timeline (measured by the driver" in be.prompts[1]
    assert [r["label"] for r in _banked(ctx)] == ["plateau", "meta0:teacher"]
    # ...but not past the verdict-probe budget
    w3 = tmp_path / "w3"
    w3.mkdir()
    be = FakeBackend(["PROBE_REQUEST: timeline", DIAG_SEG,
                      _patching(w3, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_FAIL])
    ctx = _tl_ctx(w3, be)
    ctx.max_verdict_probe_rounds = 0
    run_meta_step(ctx)
    assert [c for c in be.calls if c == ("gpu", _TIMELINE_CMD)] == [("gpu", _TIMELINE_CMD)]
    assert "Verdict-experiment budget exhausted" in be.prompts[1]


def test_keep_path_reads_gate_rows_and_new_bests_beyond_the_margin(tmp_path):
    be = FakeBackend([REC("M0-gate", 0.6), REC("D2.3", 0.5), REC("D2.4", 0.49), REC("D2.5", 0.47)],
                     [EVAL(0.6), NCU, TL_LOG,          # gate row: read whatever the reading
                      EVAL(0.5), NCU, TL_LOG,          # new best, 17% under the gate row's tree
                      EVAL(0.49), NCU,                 # new best within 3% of the last read tree: skipped
                      EVAL(0.47), NCU, TL_LOG])        # 6% under it: read
    ctx = _tl_ctx(tmp_path, be)
    run_inner_iteration(ctx, 5, metric_before=0.9)
    run_inner_iteration(ctx, 6, metric_before=0.6)
    run_inner_iteration(ctx, 7, metric_before=0.5)
    run_inner_iteration(ctx, 8, metric_before=0.49)
    assert [c for c in be.calls if c == ("gpu", _TIMELINE_CMD)] == [("gpu", _TIMELINE_CMD)] * 3
    assert [r["label"] for r in _banked(ctx)] == ["iter5:gate row", "iter6:new best", "iter8:new best"]
    assert be.gpu_logs == []
    # a KEEP that is not a new best reads nothing
    be = FakeBackend([REC("D2.6", 0.8)], [EVAL(0.8), NCU])
    ctx = _tl_ctx(tmp_path, be)
    run_inner_iteration(ctx, 9, metric_before=0.5)
    assert ("gpu", _TIMELINE_CMD) not in be.calls


def test_student_can_request_a_timeline_of_its_own_tree(tmp_path):
    be = FakeBackend(["PROBE_REQUEST: timeline", REC("D2.3", 0.5)], [TL_LOG, EVAL(0.5), NCU, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 3, metric_before=0.9)
    assert rec.verdict == "KEEP"
    assert ("commit", "probe: timel") in be.calls and ("push", "fs-demo-i3probe") in be.calls
    assert "## Timeline (measured on the current tree)" in be.prompts[1] and "| aux | 2 | 6614 | 4998 | 949 |" in be.prompts[1]
    assert [r["label"] for r in _banked(ctx)] == ["iter3:student", "iter3:new best"]
    # off: told so, no job
    be = FakeBackend(["PROBE_REQUEST: timeline", REC("D2.3", 0.5)], [EVAL(0.5), NCU])
    ctx = _tl_ctx(tmp_path, be, mode="off")
    run_inner_iteration(ctx, 4, metric_before=0.9)
    assert ("gpu", _TIMELINE_CMD) not in be.calls and "unavailable" in be.prompts[1]


def test_student_note_carries_the_convention_the_markers_and_the_latest_table(tmp_path):
    be = FakeBackend([], [])
    ctx = _tl_ctx(tmp_path, be)
    kf = tmp_path / KF
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text("x = 1\n")
    note = _student_prompt(ctx, 2)
    assert "## Timeline probe (enabled for this run)" in note and "The current tree has no markers" in note
    assert "PROBE_REQUEST: timeline" in note
    kf.write_text('def _seg(site, row=0):\n    return None\n\n_seg("state.A.top", ci)\n_seg("aux.entry")\n')
    note = _student_prompt(ctx, 2)
    assert "Markers in the current tree (2): `state.A.top`, `aux.entry`" in note
    ctx.evidence_dir.mkdir(parents=True, exist_ok=True)
    from forgekernel.loop.driver import _bank_timeline
    from forgekernel.skills.timeline_probe import parse_timeline
    _bank_timeline(ctx, "sha9", "777", "iter9:new best", parse_timeline(TL_LOG))
    note = _student_prompt(ctx, 3)
    assert "Latest table (@`sha9`, iter9:new best)" in note
    assert "state: period 6614, busy 3093, wait 1790; largest three segments" in note
    assert "`state.C.wait_kk.end`→`state.A.top (next)` 4080 (wrap)" in note
    ctx.timeline_probe = "off"
    assert "## Timeline probe" not in _student_prompt(ctx, 4)


def test_timeline_probe_reruns_once_on_a_dirty_node(tmp_path):
    """The probe's own stamped bench timed out and its launch target stayed on
    the GPU (NODE-DIRTY in the log), so
    the probe is re-run once on a fresh node — same contract as the ablation
    and lever-price probes, no design turn spent."""
    dirty = ("NODE-HYGIENE after the stamped bench timed out (2400s): GPU still held by pid(s) [77] — killing\n"
             "NODE-DIRTY after the stamped bench timed out (2400s): pid(s) [77] still hold the GPU 30s after SIGKILL\n"
             "TIMELINE-FAIL reason=bench (the stamped bench produced no JSON)\n")
    be = FakeBackend([DIAG_SEG, _patching(tmp_path, "\n### D6 added\nx\n",
                                          "patch_class: additive\nchanged_entries: []\n")],
                     [dirty, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    assert [c for c in be.calls if c == ("gpu", _TIMELINE_CMD)] == [("gpu", _TIMELINE_CMD)] * 2
    assert "## Timeline" in be.prompts[0]          # the rerun's table is what the teacher sees
    recs = _banked(ctx)
    assert len(recs) == 1 and recs[0]["ok"] and recs[0]["ms"] == 0.5255


# What an operator tree's OLD probe prints for the same run: the span reason is
# a FAIL line and there is no TIMELINE-COVERAGE. Nothing in the harness refreshes
# notes/timeline_probe.py (the job runs the tree's copy), so this is what every
# run in flight — and every tree scaffolded before the line — hands the driver.
TL_LEGACY = TL_PARTIAL.replace("TIMELINE-WARN span_ratio 0.621 below [0.75, 1.25] — short\n", "") \
                      .replace("TIMELINE-COVERAGE partial\n", "") \
                      .replace("TIMELINE-OK mode=waits\n",
                               "TIMELINE-FAIL reason=span_ratio 0.621 outside [0.75, 1.25] (the longest "
                               "sampled CTA covers a fraction of the device time)\n")


def test_a_table_from_a_tree_with_the_old_probe_is_read_through_with_the_caveat(tmp_path, capsys):
    """The GDN bwd run's 17 banked records are exactly this shape (ok false, span
    0.57-0.65, sampled clock, four roles). The legacy rule renders them, and the
    render must carry the same coverage caveat a probe-classified partial table
    does — the tree's probe being behind is not a reason to show its table as
    a whole-call account."""
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_LEGACY])
    ctx = _tl_ctx(tmp_path, be)
    run_meta_step(ctx)
    first = be.prompts[0]
    assert "## Timeline" in first and "| state | 4 | 6614 | 3093 | 1790 | 1731 |" in first
    assert "Coverage note (partial)" in first and "0.621 of device time (mode=waits)" in first and "read the absolute values as they are" in first
    assert "lacks a `locus_segment:` line" in be.prompts[1] and len(be.prompts) == 3
    recs = _banked(ctx)
    assert len(recs) == 1
    assert recs[0]["ok"] is False and recs[0]["usable"] is True and recs[0]["coverage"] == "unknown"
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if "timeline[plateau]" in l][0]
    assert "OK" in line and "FAIL" not in line and "rows do not add up to the call" in line
    # above the band an old-probe log stays out, same as a new one
    over = TL_LEGACY.replace("span_ratio=0.621", "span_ratio=1.9")
    over_dir = tmp_path / "over"
    over_dir.mkdir()
    be2 = FakeBackend([_DIAG, _patching(over_dir, "\n### D6 added\nx\n",
                                        "patch_class: additive\nchanged_entries: []\n")], [over])
    ctx2 = _tl_ctx(over_dir, be2)
    run_meta_step(ctx2)
    assert "## Timeline" not in be2.prompts[0] and _banked(ctx2)[0]["usable"] is False


def test_a_stale_tree_probe_is_named_once_in_the_log_and_in_the_student_note(tmp_path, capsys):
    """The job runs notes/timeline_probe.py from the TREE, and the harness never
    refreshes it. When the two differ (or the tree has none) the operator is told
    once per loop and the student is told where the current file is; when they
    are identical, neither is said."""
    from forgekernel.loop.driver import _timeline_probe_stale, _timeline_student_note
    from forgekernel.skills import timeline_probe as probe_mod
    be = FakeBackend([_DIAG, DIAG_SEG,
                      _patching(tmp_path, "\n### D6 added\nx\n",
                                "patch_class: additive\nchanged_entries: []\n")],
                     [TL_PARTIAL])
    ctx = _tl_ctx(tmp_path, be)
    assert _timeline_probe_stale(ctx) == probe_mod.__file__          # no tree copy at all
    run_meta_step(ctx)
    stale_lines = [l for l in capsys.readouterr().out.splitlines()
                   if "notes/timeline_probe.py in the tree differs" in l]
    assert len(stale_lines) == 1 and probe_mod.__file__ in stale_lines[0]
    note = _timeline_student_note(ctx)
    assert "does not match the harness version" in note and probe_mod.__file__ in note
    # identical copy: silent
    tree_copy = tmp_path / "notes" / "timeline_probe.py"
    tree_copy.parent.mkdir(parents=True, exist_ok=True)
    tree_copy.write_bytes(open(probe_mod.__file__, "rb").read())
    assert _timeline_probe_stale(ctx) is None
    assert "does not match the harness version" not in _timeline_student_note(ctx)
    # an older copy: named again
    tree_copy.write_text(tree_copy.read_text(encoding="utf-8").replace("TIMELINE-COVERAGE", "TIMELINE-COVERAGX"),
                         encoding="utf-8")
    assert _timeline_probe_stale(ctx) == probe_mod.__file__


def test_a_preamble_the_card_fence_would_refuse_skips_the_probe_before_any_job(tmp_path, capsys):
    """A GDN card fenced `os\\s*\\.\\s*environ` without a word boundary
    and the pre-0318c147 preamble read `_tl_os.environ`: three stamped jobs, three
    no-slab-dumped. The self-check reads the workspace's eval/bench.py, names the
    token, and spends no job; a fence the preamble clears runs the probe as before."""
    from forgekernel.loop.driver import _run_timeline
    from test_timeline_probe import GDN_FENCE_BENCH
    be = FakeBackend([], [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    (tmp_path / "eval").mkdir()
    bench = tmp_path / "eval" / "bench.py"
    # a fence the current preamble trips (it registers its slab dump with atexit)
    bench.write_text(GDN_FENCE_BENCH.replace("|cublas|cudnn", "|cublas|cudnn|\\batexit\\b"),
                     encoding="utf-8")
    assert _run_timeline(ctx, "sha1", "audit") is None
    assert not [c for c in be.calls if c[0] == "gpu"]
    out = capsys.readouterr().out
    assert "timeline[audit]: preamble hits the card fence: 'atexit'" in out
    assert "word boundary" in out
    # the GDN fence as written, against the getenv preamble: clears, and the job runs
    bench.write_text(GDN_FENCE_BENCH, encoding="utf-8")
    tl = _run_timeline(ctx, "sha1", "audit")
    assert tl is not None and tl["ok"]
    assert [c for c in be.calls if c[0] == "gpu"] == [("gpu", _TIMELINE_CMD)]


def test_a_stamped_bench_refusal_reaches_the_log_with_the_benchs_reason(tmp_path, capsys):
    from forgekernel.loop.driver import _run_timeline
    refused = ("TIMELINE-FAIL reason=stamped-bench-failed mode=all "
               "detail=forbidden op in kernel: os.environ\n")
    be = FakeBackend([], [refused])
    ctx = _tl_ctx(tmp_path, be)
    tl = _run_timeline(ctx, "sha1", "iter3:gate row")
    assert tl is not None and tl["ok"] is False and not tl["usable"]
    assert "FAIL — stamped-bench-failed mode=all detail=forbidden op in kernel: os.environ" \
        in capsys.readouterr().out
    assert _banked(ctx)[0]["table"]["fail"].startswith("stamped-bench-failed")


def test_student_note_names_marker_calls_whose_site_is_not_a_literal(tmp_path):
    """Regression: six `_seg("…%d…" % k, i)` / `_seg(NAMES[k], i)` calls
    the probe's site table cannot see. The note states the rule in the
    convention line and names the calls every turn — the student's only other
    signal was a failure line that blamed the numerics."""
    be = FakeBackend([], [])
    ctx = _tl_ctx(tmp_path, be)
    kf = tmp_path / KF
    kf.parent.mkdir(parents=True, exist_ok=True)
    kf.write_text('def _seg(site, row=0):\n    return None\n\n'
                  '_seg("wg0.segA.begin", gt)\n'
                  'for seg in range(4):\n'
                  '    _seg("wg0.segA.ld%d.begin" % seg, gt)\n'
                  '    _seg(_SEGA_LD_E[seg], gt)\n')
    note = _student_prompt(ctx, 2)
    assert "must be string literals" in note
    assert "Markers in the current tree (1): `wg0.segA.begin`" in note
    assert "2 `_seg(` calls have site names that are not string literals; the probe cannot read them" in note
    assert '`_seg("wg0.segA.ld%d.begin" % seg, gt)`, `_seg(_SEGA_LD_E[seg], gt)`' in note
    # literal-only tree: the rule stays, the list does not appear
    kf.write_text('def _seg(site, row=0):\n    return None\n\n_seg("wg0.segA.begin", gt)\n')
    note = _student_prompt(ctx, 3)
    assert "must be string literals" in note and "site names that are not string literals" not in note


# --- the harness probe overlaid on the probed ref ------------------------------
# The job runs the TREE's copy of notes/timeline_probe.py and nothing refreshed
# it, so a probe fix never reached a tree committed before it: a run's
# plateau timeline would have run on a best sha carrying the pre-fix
# probe. With a backend that can overlay one file on a ref (with_file), the
# job archives ref + harness probe; what is banked and logged stays ref.
class OverlayBackend(FakeBackend):
    def __init__(self, agent_outputs, gpu_logs, *, tree_is_harness=False, fail=False):
        super().__init__(agent_outputs, gpu_logs)
        self.tree_is_harness, self.fail, self.overlays = tree_is_harness, fail, []

    def push_ref(self, ref, branch):
        self.calls.append(("push_ref", ref, branch))

    def with_file(self, ref, path, text, message):
        from forgekernel.loop.driver import BackendError
        assert path == "notes/timeline_probe.py" and "overlaid" in message and "TIMELINE-FAIL" in text
        if self.fail:
            raise BackendError("git read-tree failed")
        if self.tree_is_harness:
            return ref
        self.overlays.append(ref)
        return f"{ref}+probe"


def test_the_probe_job_archives_the_ref_with_the_harness_probe_overlaid(tmp_path, capsys):
    from forgekernel.loop.driver import _run_timeline
    be = OverlayBackend([], [TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    tl = _run_timeline(ctx, "shabest", "plateau")
    assert tl is not None and tl["usable"]
    assert be.overlays == ["shabest"]
    assert ("push_ref", "shabest+probe", "fs-demo-timeline") in be.calls
    assert ("push_ref", "shabest", "fs-demo-timeline") not in be.calls
    # banked and logged as the tree under judgment, not the overlay
    assert [(r["sha"], r["label"]) for r in _banked(ctx)] == [("shabest", "plateau")]
    out = capsys.readouterr().out
    assert ("timeline[plateau]: notes/timeline_probe.py at shabest is not the harness copy — "
            "probing shabest with the harness probe overlaid (shabest+pr; the tree under judgment "
            "is unchanged)") in out
    assert "timeline[plateau] @ shabest: OK" in out


def test_a_tree_already_carrying_the_harness_probe_is_pushed_as_is(tmp_path, capsys):
    from forgekernel.loop.driver import _run_timeline
    be = OverlayBackend([], [TL_LOG], tree_is_harness=True)
    ctx = _tl_ctx(tmp_path, be)
    _run_timeline(ctx, "shabest", "plateau")
    assert ("push_ref", "shabest", "fs-demo-timeline") in be.calls and be.overlays == []
    assert "with the harness probe overlaid (" not in capsys.readouterr().out


def test_an_overlay_the_backend_cannot_build_falls_back_to_the_trees_copy(tmp_path, capsys):
    from forgekernel.loop.driver import _run_timeline
    be = OverlayBackend([], [TL_LOG], fail=True)
    ctx = _tl_ctx(tmp_path, be)
    tl = _run_timeline(ctx, "shabest", "plateau")
    assert tl is not None and ("push_ref", "shabest", "fs-demo-timeline") in be.calls
    assert ("could not overlay the harness probe on shabest (git read-tree failed) — probing the "
            "tree's own copy") in capsys.readouterr().out


def test_a_student_requested_timeline_repoints_its_pushed_branch_at_the_overlay(tmp_path):
    be = OverlayBackend(["PROBE_REQUEST: timeline", REC("D2.3", 0.5)], [TL_LOG, EVAL(0.5), NCU, TL_LOG])
    ctx = _tl_ctx(tmp_path, be)
    rec = run_inner_iteration(ctx, 3, metric_before=0.9)
    assert rec.verdict == "KEEP"
    # the request pushed its branch at HEAD, then the overlay re-pointed it
    calls = be.calls
    assert calls.index(("push", "fs-demo-i3probe")) < calls.index(("push_ref", "sha1+probe", "fs-demo-i3probe"))
    # the new-best read of the same iteration goes through the overlay too
    assert ("push_ref", "sha2+probe", "fs-demo-timeline") in calls
    assert [(r["sha"], r["label"]) for r in _banked(ctx)] == [("sha1", "iter3:student"), ("sha2", "iter3:new best")]
