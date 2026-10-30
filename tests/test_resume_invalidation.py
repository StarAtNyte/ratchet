"""Tests for prefix-reuse checkpoint invalidation (design doc §6).

When the outer meta loop patches a milestone, the inner student loop must not
restart from scratch: only the checkpoints whose decisions depend on a changed
milestone entry are invalidated, and the inner loop resumes from the latest
still-valid checkpoint.

A trajectory is the ordered list of inner decisions. Each decision has:
  - id          : the milestone entry it implements (e.g. "D2.3")
  - depends_on  : other milestone entries it relies on (e.g. {"D1.1", "D1.2"})

A decision is *affected* by a patch when the patch's changed entries include
its own id OR intersect its depends_on. The earliest affected decision and
everything after it are invalidated; resume happens just before it.
"""

from forgekernel.resume import Decision, ResumePlan, compute_resume


def _fa_like_trajectory():
    # Foundation-first ordering, mirroring an FA-style milestone DAG.
    return [
        Decision("D1.1", depends_on={"D0.1"}),          # tile (foundation)
        Decision("D2.3", depends_on={"D1.1"}),          # causal-skip
        Decision("D2.1", depends_on={"D1.1", "D2.3"}),  # staging
        Decision("D3.1", depends_on={"D1.3", "D2.1"}),  # RS-mode dKV
    ]


def test_patch_touching_unused_entry_reuses_whole_prefix():
    traj = _fa_like_trajectory()
    plan = compute_resume(traj, changed_entries={"D9.9"})
    assert plan == ResumePlan(resume_after="D3.1", invalidated=[])


def test_empty_patch_reuses_whole_prefix():
    traj = _fa_like_trajectory()
    plan = compute_resume(traj, changed_entries=set())
    assert plan == ResumePlan(resume_after="D3.1", invalidated=[])


def test_foundation_patch_restarts_from_baseline():
    traj = _fa_like_trajectory()
    plan = compute_resume(traj, changed_entries={"D0.1"})
    assert plan == ResumePlan(
        resume_after=None,
        invalidated=["D1.1", "D2.3", "D2.1", "D3.1"],
    )


def test_mid_patch_reuses_prefix_before_first_dependent():
    traj = _fa_like_trajectory()
    # Changing milestone decision D2.3: D2.3 itself (id match) and D2.1 (depends).
    plan = compute_resume(traj, changed_entries={"D2.3"})
    assert plan == ResumePlan(
        resume_after="D1.1",
        invalidated=["D2.3", "D2.1", "D3.1"],
    )


def test_additive_patch_is_free():
    traj = _fa_like_trajectory()
    # A brand-new decision appended to the DAG end: nothing implemented yet
    # depends on it, so the whole existing prefix is reused.
    plan = compute_resume(traj, changed_entries={"D4.1"})
    assert plan == ResumePlan(resume_after="D3.1", invalidated=[])


def test_empty_trajectory():
    plan = compute_resume([], changed_entries={"D0.1"})
    assert plan == ResumePlan(resume_after=None, invalidated=[])


def test_earliest_affected_wins_when_multiple_match():
    traj = _fa_like_trajectory()
    # Both D1.1 (depends D0.1) and D3.1 region could match; earliest must win.
    plan = compute_resume(traj, changed_entries={"D0.1", "D2.1"})
    assert plan.resume_after is None
    assert plan.invalidated == ["D1.1", "D2.3", "D2.1", "D3.1"]
