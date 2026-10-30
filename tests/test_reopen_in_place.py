"""In-place reopening of structural / split patches.

compute_resume cuts the trajectory at the earliest affected decision and
invalidates everything after it — a one-decision foundation patch became a
whole-segment rebuild (sparse: six of ten meta steps). In place, the
best green tree stays, the student redoes only the dependency closure of the
changed entries (announced as a REOPEN status row + driver note), a reverted
patch voids the REOPEN row, and pre-green runs keep the old reset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_loop_driver import FakeBackend, _ctx  # noqa: E402

from forgekernel.inner.trajectory import IterationRecord as IR, load_records  # noqa: E402
from forgekernel.loop.driver import (  # noqa: E402
    MetaResult, _judge_patch, _note_reopen, _PendingPatch, _resolve_resume_in_place,
    _student_prompt,
)
from forgekernel.resume import Decision, ReopenPlan, compute_reopen, compute_resume  # noqa: E402


def _traj():
    return [
        Decision("D0.1"),                                  # tile (foundation)
        Decision("D0.2", depends_on={"D0.1"}),             # warp roles
        Decision("D0.3", depends_on={"D0.1"}),             # smem layout
        Decision("D0.4", depends_on={"D0.2"}, companions={"D0.5"}),  # KV ring write atom
        Decision("D0.5", depends_on={"D0.3"}, companions={"D0.4"}),  # KV ring read atom
        Decision("D1.1", depends_on={"D0.4"}),             # ring depth
        Decision("D1.2", depends_on={"D0.3"}),             # index prefetch
        Decision("D2.1", depends_on={"D1.1", "D1.2"}),     # epilogue
    ]


# --- the closure -------------------------------------------------------------

def test_reopen_is_the_dependency_closure_not_the_suffix():
    plan = compute_reopen(_traj(), {"D0.4"})
    # D0.4 itself, its ⊗ companion D0.5, D1.1 (depends on D0.4), D2.1 (depends on D1.1)
    assert plan == ReopenPlan(reopen=["D0.4", "D0.5", "D1.1", "D2.1"],
                              kept=["D0.1", "D0.2", "D0.3", "D1.2"])
    # the prefix-reuse plan for the same patch invalidates the whole suffix
    assert compute_resume(_traj(), {"D0.4"}).invalidated == ["D0.4", "D0.5", "D1.1", "D1.2", "D2.1"]


def test_reopen_closure_propagates_through_companions_and_late_dependencies():
    # changing D0.3 reaches D0.5 (depends), then D0.4 (companion of D0.5, declared
    # EARLIER in the list), then D1.1 (depends on D0.4) — one ordered pass misses it
    plan = compute_reopen(_traj(), {"D0.3"})
    assert plan.reopen == ["D0.3", "D0.4", "D0.5", "D1.1", "D1.2", "D2.1"]
    assert plan.kept == ["D0.1", "D0.2"]
    assert compute_reopen(_traj(), {"D9.9"}) == ReopenPlan(reopen=[], kept=[d.id for d in _traj()])
    assert compute_reopen([], {"D0.1"}) == ReopenPlan(reopen=[], kept=[])
    # a foundation change reopens everything but still ON the tree (no reset)
    assert compute_reopen(_traj(), {"D0.1"}).kept == []


# --- the driver ---------------------------------------------------------------

def _keeps():
    return [IR(iter=0, decision_id="D0.1+D0.2+D0.3", change="m0", correctness=True,
               verdict="KEEP", bwd_median_ms=3.5, device_median_ms=3.5, sha="shaM0"),
            IR(iter=1, decision_id="D0.4+D0.5", change="ring", correctness=True,
               verdict="KEEP", bwd_median_ms=3.2, device_median_ms=3.2, sha="shaRING"),
            IR(iter=2, decision_id="D1.1", change="depth", correctness=True,
               verdict="KEEP", bwd_median_ms=3.0, device_median_ms=3.0, sha="shaBEST"),
            IR(iter=3, decision_id="D1.2", change="prefetch", correctness=True,
               verdict="KEEP", bwd_median_ms=3.1, device_median_ms=3.1, sha="shaLATER")]


def test_structural_patch_reopens_on_the_best_tree_instead_of_resetting(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    meta = MetaResult("d", "structural", ["D0.4"], resume_after="D0.3",
                      invalidated=["D0.4", "D0.5", "D1.1", "D1.2"],
                      reopened=["D0.4", "D0.5", "D1.1"], kept=["D0.1", "D0.2", "D0.3", "D1.2"])
    ref, reopened = _resolve_resume_in_place(ctx, meta, _keeps(), "shaBASE")
    assert ref == "shaBEST" and reopened == ["D0.4", "D0.5", "D1.1"]
    # opted out -> the prefix-reuse reset (to the checkpoint before D0.4)
    ctx.reopen_in_place = False
    assert _resolve_resume_in_place(ctx, meta, _keeps(), "shaBASE") == ("shaM0", [])


def test_reopen_falls_back_to_reset_when_nothing_is_green_or_nothing_reopens(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    meta = MetaResult("d", "structural", ["D0.1"], resume_after=None, invalidated=["D0.1"],
                      reopened=["D0.1", "D0.2"], kept=[])
    scaffold = [IR(iter=0, decision_id="D0.1", change="s", correctness=None,
                   verdict="NEUTRAL", sha="shaS")]
    assert _resolve_resume_in_place(ctx, meta, scaffold, "shaBASE") == ("shaBASE", [])
    # a foundation patch WITH a green best: reopen everything on that tree, no restart
    assert _resolve_resume_in_place(ctx, meta, _keeps(), "shaBASE") == ("shaBEST", ["D0.1", "D0.2"])
    # additive / kb-example never reopen
    add = MetaResult("d", "additive", ["D3.1"], resume_after="D1.2", invalidated=[])
    be = ctx.backend
    be._sha = 9
    assert _resolve_resume_in_place(ctx, add, _keeps(), "shaBASE") == ("sha9", [])
    empty = MetaResult("d", "structural", ["D9.9"], resume_after="D1.2", invalidated=[])
    assert _resolve_resume_in_place(ctx, empty, _keeps(), "shaBASE") == ("shaLATER", [])


def test_reopen_row_and_student_note(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    for r in _keeps():
        from forgekernel.inner.trajectory import append_record
        append_record(ctx.status_path, r)
    meta = MetaResult("d", "structural", ["D0.4"], resume_after="D0.3",
                      invalidated=[], reopened=["D0.4", "D0.5", "D1.1"],
                      kept=["D0.1", "D0.2", "D0.3", "D1.2"])
    _note_reopen(ctx, meta, "shaBEST0000")
    recs = load_records(ctx.status_path)
    row = recs[-1]
    assert row.decision_id == "REOPEN" and row.verdict == "NEUTRAL" and row.sha == ""
    assert row.correctness is None and row.iter == 4
    assert "[D0.4, D0.5, D1.1]" in row.change and "[D0.1, D0.2, D0.3, D1.2]" in row.change
    assert "shaBEST000" in row.change
    prompt = _student_prompt(ctx, 5)
    assert "Reopened in place (REOPEN)" in prompt and "D0.4, D0.5, D1.1" in prompt
    assert "remain valid; do not rebuild them" in prompt
    # the template itself explains REOPEN / REOPEN-REVERT rows to the student
    assert "decision_id: REOPEN" in prompt and "REOPEN-REVERT" in prompt


def test_reverted_reopen_patch_voids_the_reopen_row(tmp_path):
    from forgekernel.inner.trajectory import append_record
    be = FakeBackend([], [])
    ctx = _ctx(tmp_path, be)
    ctx.meta_status_path = None          # no meta ledger row to update in this unit test
    ctx.milestone.write_text("OLD")
    for r in _keeps():
        append_record(ctx.status_path, r)
    meta = MetaResult("d", "structural", ["D0.4"], resume_after=None, invalidated=[],
                      reopened=["D0.4", "D0.5", "D1.1"], kept=["D0.1"])
    _note_reopen(ctx, meta, "shaBEST")
    n = len(load_records(ctx.status_path))
    pending = _PendingPatch(best_before=3.0, milestone_before="OLD", good_ref="shaBEST",
                            records_before=n, reopened=list(meta.reopened))
    # the reopened segment measured a KEEP but no gain -> REVERT_PATCH
    append_record(ctx.status_path, IR(iter=n, decision_id="D0.4+D0.5", change="redo",
                                      correctness=True, verdict="KEEP", bwd_median_ms=3.05,
                                      device_median_ms=3.05, sha="shaREDO"))
    assert _judge_patch(ctx, pending) == "REVERT_PATCH"
    assert ("reset", "shaBEST") in be.calls
    last = load_records(ctx.status_path)[-1]
    assert last.decision_id == "REOPEN-REVERT" and last.verdict == "NEUTRAL"
    assert "[D0.4, D0.5, D1.1]" in last.change and "stand again" in last.change
    assert ctx.reopen_note == ""
    assert ctx.milestone.read_text() == "OLD"
