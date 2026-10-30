"""Host-bound classification: the device window's precondition, and who acts on it.

The failure this guards against: a reference with a thick
python wrapper read 0.4889 ms in the pipelined device window and 0.2598 ms under
CUDA-graph replay -- 46% of its "device time" was per-call python. The student,
measured in the same window, was genuinely GPU-bound. Every existing defence passed:
correctness, cross-job repeatability, node-speed noise. The resulting target was
1.86x too loose.
"""

from __future__ import annotations

import pytest

from forgekernel.baseline import BASIS_CUPTI, BASIS_GRAPH, BASIS_PLAIN
from forgekernel.timing_audit import (
    HOST_BOUND_RATIO_WARN,
    classify,
    host_bound_ratio,
    host_path_feedback,
    host_path_note,
    is_host_free_basis,
    ncu_coverage,
    ncu_reconciliation_warning,
    reference_blocker,
    reference_warning,
    student_feedback,
    student_warning,
)

# The live readings, so the thresholds are exercised against the case they exist for.
REF_DEVICE_MS = 0.4889     # MEASURED: pipelined window
REF_GRAPH_MS = 0.2598      # MEASURED: same reference, host removed by graph replay
# MODELLED, not measured -- the enqueue clock is what this change adds, so it did not
# exist during the incident. In a host-paced pipeline the host IS the pacer, so the
# enqueue clock reads essentially the whole window and the ratio saturates near 1;
# the 46% figure quoted elsewhere is the window's IDLE share (1 - graph/device), a
# different quantity computed from the two measured numbers above.
REF_HOST_MS = 0.4712
STUDENT_DEVICE_MS = 0.4085  # MEASURED: agreed with this kernel's own NCU total
STUDENT_HOST_MS = 0.004     # MODELLED: a thin python path, GPU-bound


def test_ratio_is_absent_rather_than_passing_when_unmeasured():
    # A bench that never reported the enqueue clock has NO verdict. Reading the
    # absence as "fine" would make every pre-check archive silently certified.
    v = classify()
    assert v.measured is False and v.host_bound is False and v.ratio is None
    assert v.pct == "unmeasured"


def test_gpu_bound_student_is_not_flagged():
    v = classify(host_enqueue_ms=STUDENT_HOST_MS, device_median_ms=STUDENT_DEVICE_MS)
    assert v.measured is True and v.host_bound is False
    assert student_warning(v) == ""
    assert student_feedback(v) == ""


def test_host_bound_reference_is_flagged():
    v = classify(host_enqueue_ms=REF_HOST_MS, device_median_ms=REF_DEVICE_MS)
    assert v.host_bound is True
    assert v.ratio - HOST_BOUND_RATIO_WARN >= 0


def test_a_precomputed_ratio_is_used_as_reported():
    v = classify(ratio=0.62)
    assert v.host_bound is True and v.ratio == 0.62


def test_host_bound_ratio_helper_needs_both_sides():
    assert host_bound_ratio(0.2, None) is None
    assert host_bound_ratio(None, 0.2) is None
    assert host_bound_ratio(0.0, 0.2) is None
    assert host_bound_ratio(0.1, 0.2) == 0.5


# --- the two sides need OPPOSITE handling ------------------------------------

def test_a_host_bound_reference_blocks_target_derivation():
    v = classify(host_enqueue_ms=REF_HOST_MS, device_median_ms=REF_DEVICE_MS)
    blocker = reference_blocker(v, ref_name="flashinfer", op="gdn_prefill_fwd")
    assert blocker
    assert "HOST-BOUND" in blocker
    assert "graph" in blocker.lower()          # names the replacement measurement


def test_an_archived_host_free_reading_unblocks_it():
    # The prescribed fix, applied: there IS a defensible device number, so the run
    # may start -- anchored on that one.
    v = classify(host_enqueue_ms=REF_HOST_MS, device_median_ms=REF_DEVICE_MS)
    assert reference_blocker(v, graph_ms=REF_GRAPH_MS) == ""
    warn = reference_warning(v, graph_ms=REF_GRAPH_MS)
    assert "target anchors on it" in warn


def test_an_unmeasured_reference_cannot_block():
    # Every archive written before the enqueue clock existed is unmeasured.
    # Blocking those would stop every run rather than the broken ones.
    assert reference_blocker(classify()) == ""
    assert reference_warning(classify()) == ""


def test_the_reference_side_never_produces_student_feedback():
    # "the reference's python wrapper is thick" reads to a kernel writer as "a thin
    # python path wins". student_feedback is the only channel to the student, and a
    # reference verdict must not travel through it.
    v = classify(host_enqueue_ms=REF_HOST_MS, device_median_ms=REF_DEVICE_MS)
    text = reference_warning(v, ref_name="flashinfer")
    assert "metric-gaming" in text or "teaches" in text
    assert "Do NOT" in text


def test_the_student_side_is_a_diagnosis_naming_host_causes():
    # It inflates the student's OWN reading and the deployment pays it, so it is a
    # real defect -- and the note has to point at the host path, because a vague
    # note sends the agent editing the kernel body where the cost is not.
    v = classify(host_enqueue_ms=0.30, device_median_ms=0.40)
    note = student_feedback(v, device_ms=0.40)
    assert note.startswith("- ")
    assert "host" in note
    for cause in ("recompiling", "allocation", "launch", "synchronization"):
        assert cause in note
    assert student_warning(v, device_ms=0.40).startswith("HOST-BOUND STUDENT")


# --- NCU reconciliation: same shortfall, two causes, opposite fixes -----------

def test_coverage_uses_the_all_kernel_total_not_the_dominant_one():
    # A decomposed operator (4 kernels, 0.26 ms together) against a 0.48 ms window:
    # judging by the dominant kernel alone would read the decomposition as an error.
    prof = dict(total_time_ns=260_000, time_ns=120_000, kernel_count=4)
    assert ncu_coverage(prof, 0.48) == pytest.approx(0.5417, rel=1e-3)
    assert ncu_coverage(dict(), 0.48) is None
    assert ncu_coverage(prof, None) is None


def test_reconciliation_blames_the_host_when_the_enqueue_clock_agrees():
    prof = dict(total_time_ns=117_216, kernel_count=1, total_launches=1)
    v = classify(ratio=0.9)
    warn = ncu_reconciliation_warning(prof, 4.0097, v)
    assert "host-bound" in warn
    assert "not a mis-aimed profile" in warn


def test_reconciliation_blames_the_profile_when_the_host_is_fine():
    prof = dict(total_time_ns=117_216, kernel_count=1, total_launches=1)
    v = classify(ratio=0.01)
    warn = ncu_reconciliation_warning(prof, 4.0097, v)
    assert "mis-aimed" in warn and "eval/ncu_run.py" in warn


def test_reconciliation_says_so_when_it_cannot_tell():
    prof = dict(total_time_ns=117_216, kernel_count=1)
    warn = ncu_reconciliation_warning(prof, 4.0097, classify())
    assert "cause unresolved" in warn
    assert "host_enqueue_ms" in warn


def test_reconciliation_is_silent_when_the_kernels_account_for_the_window():
    prof = dict(total_time_ns=2_077_216, kernel_count=3)
    assert ncu_reconciliation_warning(prof, 2.1074, classify()) == ""


def test_reconciliation_is_silent_without_numbers():
    assert ncu_reconciliation_warning(dict(), 4.0, classify()) == ""
    assert ncu_reconciliation_warning(dict(total_time_ns=1), None, classify()) == ""


# --- a host-free basis exempts the reading from the enqueue-clock verdict ---------
#
# dsa_sparse on B200 (MEASURED): gate = CUPTI kernel span 0.01522 ms;
# the enqueue clock read 0.06463 ms/call; the in-job reference's read 0.01911. The
# card reported host_bound_ratio: null for the span basis; the driver divided the
# two raw timings anyway (4.25) and told the student its gate reading was inflated.

SPAN_DEVICE_MS = 0.01522
SPAN_HOST_MS = 0.06463
SPAN_REF_HOST_MS = 0.01911


def test_the_two_host_free_bases_are_recognised():
    assert is_host_free_basis(BASIS_CUPTI) and is_host_free_basis(BASIS_GRAPH)
    assert not is_host_free_basis(BASIS_PLAIN)
    assert not is_host_free_basis(None) and not is_host_free_basis("")


def test_a_span_reading_is_never_host_bound_whatever_the_raw_timings_say():
    v = classify(host_enqueue_ms=SPAN_HOST_MS, device_median_ms=SPAN_DEVICE_MS,
                 basis=BASIS_CUPTI)
    assert v.host_free and v.basis == BASIS_CUPTI
    assert v.host_bound is False
    assert v.ratio is None          # the clock does not describe this reading
    assert v.measured is True       # the precondition holds by construction
    assert student_warning(v, device_ms=SPAN_DEVICE_MS) == ""
    assert student_feedback(v, device_ms=SPAN_DEVICE_MS) == ""
    assert reference_blocker(v) == ""


def test_a_ratio_the_bench_printed_for_the_plain_window_does_not_leak_into_a_span_verdict():
    v = classify(ratio=4.25, basis=BASIS_CUPTI)
    assert v.host_bound is False and v.ratio is None


def test_the_same_timings_on_the_plain_basis_keep_the_old_verdict():
    v = classify(host_enqueue_ms=SPAN_HOST_MS, device_median_ms=SPAN_DEVICE_MS,
                 basis=BASIS_PLAIN)
    assert v.host_bound is True and v.ratio == pytest.approx(4.246, rel=1e-3)
    assert not v.host_free
    assert "HOST-BOUND STUDENT" in student_warning(v)
    # and a bench that declares no basis at all behaves as before
    assert classify(host_enqueue_ms=SPAN_HOST_MS, device_median_ms=SPAN_DEVICE_MS).host_bound


def test_the_host_path_note_records_the_deployment_cost_without_touching_the_gate():
    note = host_path_note(SPAN_HOST_MS, SPAN_DEVICE_MS, basis=BASIS_CUPTI,
                          ref_host_enqueue_ms=SPAN_REF_HOST_MS)
    assert note.startswith("HOST PATH:")
    assert "NOT affected" in note and "not a gate" in note
    assert "3.4x thinner" in note                   # 0.06463 / 0.01911
    assert "inflated" not in note and "HOST-BOUND" not in note


def test_the_host_path_note_is_silent_on_a_plain_basis_and_under_the_threshold():
    assert host_path_note(SPAN_HOST_MS, SPAN_DEVICE_MS, basis=BASIS_PLAIN) == ""
    assert host_path_note(0.004, 0.4085, basis=BASIS_CUPTI) == ""     # thin wrapper
    assert host_path_note(None, SPAN_DEVICE_MS, basis=BASIS_CUPTI) == ""


def test_the_host_path_feedback_tells_the_student_the_gate_is_clean():
    fb = host_path_feedback(SPAN_HOST_MS, SPAN_DEVICE_MS, basis=BASIS_CUPTI,
                            ref_host_enqueue_ms=SPAN_REF_HOST_MS)
    assert "not part of the gate" in fb and "not inflated by the host" in fb
    assert "3.4x" in fb
    assert "dlpack" in fb                           # the dsa_sparse wrapper's actual cost
    assert host_path_feedback(0.004, 0.4085, basis=BASIS_CUPTI) == ""
    assert host_path_feedback(SPAN_HOST_MS, SPAN_DEVICE_MS, basis=BASIS_PLAIN) == ""


def test_reconciliation_under_a_host_free_basis_blames_the_profile_not_the_host():
    v = classify(host_enqueue_ms=SPAN_HOST_MS, device_median_ms=SPAN_DEVICE_MS,
                 basis=BASIS_CUPTI)
    text = ncu_reconciliation_warning({"total_time_ns": 1000, "dominant_launches": 1},
                                      SPAN_DEVICE_MS, v)
    assert "host-free reading" in text and "mis-aimed profile" in text
    assert "host_bound_ratio" not in text
