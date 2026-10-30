"""Tests for the meta-loop trajectory (meta_status.jsonl) — the meta agent's memory."""

from forgekernel.meta_trajectory import (
    MetaRecord,
    append_meta_record,
    load_meta_records,
    reflect_summary,
    update_last_outcome,
)


def _mrec(**over):
    base = dict(
        meta_step=0,
        cause_class="R1",
        deficient_locus="D1.4",
        patch_class="structural",
        changed_entries=["D1.4", "D0.9"],
        patch_direction="shrink tile_m 128->64",
    )
    base.update(over)
    return MetaRecord(**base)


def test_meta_record_roundtrip(tmp_path):
    p = tmp_path / "meta_status.jsonl"
    append_meta_record(p, _mrec(meta_step=0))
    append_meta_record(p, _mrec(meta_step=1, cause_class="G3", patch_class="kb-example",
                                outcome="KB_CERTIFIED", certified=True))
    recs = load_meta_records(p)
    assert [r.meta_step for r in recs] == [0, 1]
    assert recs[1].outcome == "KB_CERTIFIED" and recs[1].certified is True


def test_meta_record_defaults_pending():
    assert _mrec().outcome == "PENDING"
    assert _mrec().best_before is None and _mrec().best_after is None


def test_load_meta_records_drops_unknown_keys(tmp_path):
    import json
    p = tmp_path / "meta_status.jsonl"
    p.write_text(json.dumps({
        "meta_step": 0, "cause_class": "R1", "deficient_locus": "D1.4",
        "patch_class": "structural", "changed_entries": [], "patch_direction": "x",
        "some_future_field": "ignore me",
    }) + "\n")
    recs = load_meta_records(p)
    assert recs[0].cause_class == "R1"


def test_update_last_outcome_rewrites_pending(tmp_path):
    # the patch outcome is judged only AFTER the post-patch inner segment runs,
    # so the PENDING record is updated in place (not appended).
    p = tmp_path / "meta_status.jsonl"
    append_meta_record(p, _mrec(meta_step=0, best_before=2.66))
    update_last_outcome(p, outcome="REVERT_PATCH", best_after=3.88)
    recs = load_meta_records(p)
    assert len(recs) == 1
    assert recs[0].outcome == "REVERT_PATCH" and recs[0].best_after == 3.88


def test_reflect_summary_flags_reverted_directions(tmp_path):
    # the reflection summary the diagnose prompt consumes marks REVERT_PATCH
    # directions as forbidden and lists certified KB interfaces.
    p = tmp_path / "meta_status.jsonl"
    append_meta_record(p, _mrec(meta_step=0, patch_direction="shrink tile_m 128->64",
                                outcome="REVERT_PATCH"))
    append_meta_record(p, _mrec(meta_step=1, cause_class="G3", deficient_locus="KB:swizzle",
                                patch_class="kb-example", outcome="KB_CERTIFIED"))
    summary = reflect_summary(load_meta_records(p))
    assert "REVERT_PATCH" in summary and "shrink tile_m" in summary
    assert "swizzle" in summary


def test_reflect_summary_names_a_refused_unpriced_reopen():
    from forgekernel.meta_trajectory import MetaRecord, reflect_summary
    r = MetaRecord(meta_step=1, cause_class="R1", deficient_locus="D0.2",
                   patch_class="unpriced-reopen", changed_entries=[],
                   patch_direction="flip the axis", outcome="UNPRICED_REOPEN")
    text = reflect_summary([r])
    assert "[unpriced reopen refused] step1 R1 @ D0.2" in text and "reopen_price:" in text


def test_reflect_summary_records_an_infeasible_stop_as_a_bare_fact():
    """The memory row of an INFEASIBLE_STOP states what happened and nothing
    else: no word on what a human made of the stop and no pointer to
    human-written direction — either would feed a human meta decision back
    into the run under test."""
    from forgekernel.meta_trajectory import MetaRecord, reflect_summary
    r = MetaRecord(meta_step=4, cause_class="INFEASIBLE", deficient_locus="none",
                   patch_class="infeasible-stop", changed_entries=[],
                   patch_direction="re-distil a new candidate", outcome="INFEASIBLE_STOP")
    row = next(line for line in reflect_summary([r]).splitlines() if "[stop request]" in line)
    assert "step4 INFEASIBLE" in row and "re-distil a new candidate" in row
    for word in ("user", "ruling", "adopt", "new direction", "as-is"):
        assert word not in row, word


def test_reflect_summary_empty_is_none_marker():
    assert reflect_summary([]).strip() != ""  # a non-empty "no prior meta steps" marker



def test_probe_readings_roundtrip_and_tolerate_old_records(tmp_path):
    from forgekernel.meta_trajectory import (MetaRecord, append_meta_record,
                                            load_meta_records)
    p = tmp_path / "meta_status.jsonl"
    rec = MetaRecord(meta_step=3, cause_class="R1", deficient_locus="D0.14",
                     patch_class="structural", changed_entries=["D0.14"],
                     patch_direction="x", probe_rounds=1,
                     probe_readings=[{"path": "notes/p.py", "round": 1, "contract": 3,
                                      "verdict": "PASS",
                                      "reading": "ref_exposed=106.3,student_exposed=159.6",
                                      "note": "c3 (decisive)"}])
    append_meta_record(p, rec)
    # a record written before the field existed loads with an empty list
    p.write_text(p.read_text() + '{"meta_step": 4, "cause_class": "G1", '
                 '"deficient_locus": "D1", "patch_class": "additive", '
                 '"changed_entries": [], "patch_direction": "y"}\n')
    recs = load_meta_records(p)
    assert recs[0].probe_readings[0]["reading"] == "ref_exposed=106.3,student_exposed=159.6"
    assert recs[1].probe_readings == []


def test_reflect_summary_carries_the_measured_readings():
    from forgekernel.meta_trajectory import MetaRecord, reflect_summary
    readings = [{"path": "notes/exposure.py", "round": 1, "contract": i,
                 "verdict": "PASS" if i == 3 else "FAIL",
                 "reading": "ref_exposed=106.3,student_exposed=159.6" if i == 3 else "n/a",
                 "note": f"c{i}"} for i in range(1, 12)]
    rec = MetaRecord(meta_step=9, cause_class="R1", deficient_locus="D0.14",
                     patch_class="structural", changed_entries=["D0.14"],
                     patch_direction="re-cut the shadow relation", outcome="REVERT_PATCH",
                     probe_rounds=1, probe_readings=readings)
    text = reflect_summary([rec])
    assert "quote quantities already measured directly" in text
    assert ("  · measured notes/exposure.py #3 PASS "
            "max_diff=ref_exposed=106.3,student_exposed=159.6 — c3") in text
    # bounded: eight sub-rows, then a pointer to the ledger for the rest
    assert text.count("  · measured") == 8 and "3 more contract readings" in text
    # the direction line itself is unchanged (still the forbidden-direction marker)
    assert "[dead end] step9 R1 @ D0.14" in text



def test_update_last_outcome_targets_the_open_hold_over_a_trailing_terminal_row(tmp_path):
    """A held construction is judged at a later stop; an audit's terminal row
    (UNPRICED_REOPEN / AUDIT_CLEAN) appended after it must not absorb that
    judgment."""
    p = tmp_path / "meta_status.jsonl"
    append_meta_record(p, _mrec(meta_step=0, outcome="HOLD_PATCH"))
    append_meta_record(p, _mrec(meta_step=1, patch_class="unpriced-reopen",
                                outcome="UNPRICED_REOPEN"))
    update_last_outcome(p, outcome="KEEP_PATCH", best_after=2.1)
    recs = load_meta_records(p)
    assert [r.outcome for r in recs] == ["KEEP_PATCH", "UNPRICED_REOPEN"]
    assert recs[0].best_after == 2.1
