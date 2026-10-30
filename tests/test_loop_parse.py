"""Tests for the two-layer driver's deterministic parsers."""

from __future__ import annotations

import pytest

from forgekernel.loop.parse import (
    Exhausted,
    ProbeRequest,
    Record,
    ReconRequest,
    detect_plateau,
    parse_eval,
    parse_probe_verdict,
    parse_student_turn,
    JOB_END_CUT, JOB_END_KILLED, JOB_END_WHOLE, JobEnd, job_abend,
)
from forgekernel.inner.trajectory import IterationRecord

RECORD_TEXT = """\
some reasoning prose the agent emitted first...
decision_id:        D2.4
change:             degenerate 32-bit -> conflict-free 128-bit flat-TV dQ R2S
hypothesis:         bank conflicts collapse -> MFU up
binding_constraint: register/latency after this
files_touched:      flash_attn_dsl/flash_bwd.py
"""


def test_parse_record_extracts_fields_amid_prose():
    turn = parse_student_turn(RECORD_TEXT)
    assert isinstance(turn, Record)
    assert turn.decision_id == "D2.4"
    assert "128-bit flat-TV" in turn.change
    assert turn.files_touched == "flash_attn_dsl/flash_bwd.py"


def test_parse_probe_request():
    turn = parse_student_turn("I need to verify a layout.\nPROBE_REQUEST: notes/dq_probe.py")
    assert isinstance(turn, ProbeRequest)
    assert turn.path == "notes/dq_probe.py"


def test_recon_request_is_not_a_student_shape():
    """RECON_REQUEST is the reconnaissance caller's vocabulary only. Parsed
    everywhere, a student turn merely echoing the marker reached the probe
    branch and burned a GPU job on a teacher-tree-only path."""
    text = "let me measure the reference first\nRECON_REQUEST: notes/probe1.py"
    with pytest.raises(ValueError):
        parse_student_turn(text)                      # student caller: protocol error
    turn = parse_student_turn(text, allow_recon=True)  # recon caller: valid shape
    assert isinstance(turn, ReconRequest)
    assert turn.path == "notes/probe1.py"


def test_parse_exhausted():
    turn = parse_student_turn("EXHAUSTED: reached M5, no further phase-order decision available")
    assert isinstance(turn, Exhausted)
    assert "M5" in turn.reason


def test_probe_request_takes_precedence_over_stray_record_words():
    # a turn that requests a probe must not be misread as a record
    turn = parse_student_turn("decision_id mentioned in passing\nPROBE_REQUEST: notes/x.py")
    assert isinstance(turn, ProbeRequest)


def test_malformed_turn_raises():
    with pytest.raises(ValueError):
        parse_student_turn("just some prose with no record, no probe, no exhausted marker")


def test_parse_eval_from_gpu_entry_json():
    log = """
[task] Correctness + Performance
=== JSON ===
{"correctness": {"fwd_max_diff": 0.0, "dq": 0.001953, "pass": true}, "performance": {"bwd_median_ms": 2.603, "bwd_mfu_pct": 33.35}}
[gpu_entry.sh] task=all finished (exit 0)
"""
    ev = parse_eval(log)
    assert ev.correctness is True
    assert ev.bwd_median_ms == 2.603
    assert ev.mfu == 33.35


def test_parse_eval_correctness_fail():
    log = '=== JSON ===\n{"correctness": {"pass": false}, "performance": {}}'
    ev = parse_eval(log)
    assert ev.correctness is False
    assert ev.bwd_median_ms is None


def test_parse_eval_carries_the_bench_failure_verdict():
    # A harness-level rejection arrives as a single error string; it must survive
    # into EvalResult verbatim — "correctness FAIL" alone once hid a forbidden-op
    # scanner match behind 14h of phantom numeric-bug chasing.
    log = ('=== JSON ===\n{"correctness": {"pass": false, "max_diff": null, '
           '"error": "forbidden op in kernel.py: \'flash_attn\'"}, "performance": null}')
    ev = parse_eval(log)
    assert ev.correctness is False
    assert "forbidden op" in ev.error


def test_parse_eval_summarises_a_numeric_fail_per_tensor():
    log = ('=== JSON ===\n{"correctness": {"pass": false, "max_diff": 3.1, '
           '"dq": {"max_diff": 3.1, "tolerance": 0.9}, '
           '"dk": {"error": "bad dk: shape=() dtype=None (want cuda bfloat16 ...)"}}, '
           '"performance": null}')
    ev = parse_eval(log)
    assert "dq: max_diff=3.1" in ev.error
    assert "dk: bad dk" in ev.error


def test_parse_eval_error_empty_on_pass():
    log = '=== JSON ===\n{"correctness": {"pass": true}, "performance": {"bwd_median_ms": 2.0}}'
    assert parse_eval(log).error == ""


def test_parse_eval_missing_json_block_says_so():
    ev = parse_eval("[gpu_entry.sh] job killed mid-run")
    assert ev.correctness is False
    assert "no eval JSON block" in ev.error


def test_parse_eval_carries_the_in_job_library_calibration():
    """The ratio is the comparable number, so it has to survive the parser.

    Absolute latency for one unchanged commit moves ~6% between jobs, which is wider
    than most single-round gains; a ratio against a library measured in the *same* job
    does not. Drop these two fields and every cross-round and cross-run comparison
    silently falls back to reading node-to-node variance.
    """
    log = """
=== JSON ===
{"correctness": {"pass": true},
 "performance": {"bwd_median_ms": 1.4008, "bwd_mfu_pct": 79.33,
                 "cublas_median_ms": 1.5441, "vs_cublas": 1.1023}}
"""
    ev = parse_eval(log)
    assert ev.ref_median_ms == 1.5441
    assert ev.vs_ref == 1.1023


def test_parse_eval_uncalibrated_harness_reads_as_none():
    """An operator whose bench does not measure a library must not fabricate a ratio."""
    log = '=== JSON ===\n{"correctness": {"pass": true}, "performance": {"bwd_median_ms": 2.6}}'
    ev = parse_eval(log)
    assert ev.ref_median_ms is None and ev.vs_ref is None
    # single-window harness: the device-window trio is absent, not invented
    assert (ev.device_median_ms is None and ev.ref_device_ms is None
            and ev.vs_ref_device is None)


def test_parse_eval_carries_the_device_window_when_measured():
    """Two timing windows, two claims. The per-call window (bwd_median_ms) includes
    host work and is the gate metric; the pipelined batch window (device_median_ms)
    hides host cost and is what kernel-vs-kernel comparisons read. The same cuDNN
    kernel measured 0.28 ms in one and 0.25 ms in the other, and the two sides of a
    ratio pay different host taxes — drop these fields and the e2e ratio silently
    gets quoted as a kernel-quality number."""
    log = """
=== JSON ===
{"correctness": {"pass": true},
 "performance": {"bwd_median_ms": 0.2835, "bwd_mfu_pct": 49.0,
                 "cublas_median_ms": 0.2744, "vs_cublas": 0.9679,
                 "device_median_ms": 0.2731, "cublas_device_ms": 0.2491,
                 "vs_cublas_device": 0.9121}}
"""
    ev = parse_eval(log)
    assert ev.device_median_ms == 0.2731
    assert ev.ref_device_ms == 0.2491
    assert ev.vs_ref_device == 0.9121


def test_parse_eval_no_json_is_failure():
    ev = parse_eval("Traceback (most recent call last): ...\nRuntimeError")
    assert ev.correctness is False
    assert ev.bwd_median_ms is None


def test_parse_probe_verdict():
    assert parse_probe_verdict("PROBE PASS max_diff=0.0").passed is True
    assert parse_probe_verdict("PROBE FAIL max_diff=1.13").passed is False
    assert parse_probe_verdict("compile error, no verdict").passed is False


def test_parse_probe_verdict_malformed_max_diff_does_not_certify():
    # agent-generated max_diff can be garbage. On a FAIL it is only informational,
    # so it degrades to None; on a PASS it is the whole claim, so it is rejected.
    v = parse_probe_verdict("PROBE FAIL max_diff=E")
    assert v.passed is False and v.max_diff is None
    v = parse_probe_verdict("PROBE PASS max_diff=nan-ish")
    assert v.passed is False and "unparseable" in v.reason


def test_no_verdict_is_distinct_from_a_numerical_fail():
    # both used to surface as "FAIL max_diff=None", so a probe that never built was
    # fed back to the agent as a numerical failure.
    absent = parse_probe_verdict("ptxas error: unresolved symbol\nexit 1")
    assert absent.passed is False and absent.found is False
    real = parse_probe_verdict("PROBE FAIL max_diff=1.13")
    assert real.passed is False and real.found is True


def test_any_fail_among_several_verdicts_loses():
    # first-match-wins let a probe certify by printing PASS before its FAIL
    v = parse_probe_verdict("PROBE PASS max_diff=0.0\nPROBE FAIL max_diff=2.5")
    assert v.passed is False and v.max_diff == 2.5
    assert "2 PROBE lines" in v.reason
    assert parse_probe_verdict("PROBE PASS max_diff=0.0\n"
                               "PROBE PASS max_diff=1e-3").passed is True


def test_pass_above_the_ceiling_is_not_an_error_norm():
    # observed in a bwd run: sub-check bitmasks smuggled through max_diff,
    # the one field the driver echoes back. Every one of these self-certified.
    for junk in ("80000000000.0", "8100051205110506.0", "8064406440644.0"):
        v = parse_probe_verdict(f"PROBE PASS max_diff={junk}")
        assert v.passed is False, junk
        assert "ceiling" in v.reason
    assert parse_probe_verdict("PROBE PASS max_diff=0.003853").passed is True


def test_pass_must_declare_what_it_verified():
    bare = parse_probe_verdict("PROBE PASS")
    assert bare.passed is False and "no max_diff" in bare.reason
    # ...but a non-numeric contract can say so explicitly, and is recorded as such
    na = parse_probe_verdict("PROBE PASS max_diff=n/a")
    assert na.passed is True and na.numeric is False and na.max_diff is None
    assert na.diff_text == "n/a (non-numeric probe)"


def test_non_finite_pass_is_rejected():
    for junk in ("nan", "inf", "-inf"):
        assert parse_probe_verdict(f"PROBE PASS max_diff={junk}").passed is False


def _rec(it, ms, verdict="KEEP"):
    # dual-window record, same reading in both (the plateau math is window-agnostic)
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict=verdict, bwd_median_ms=ms, device_median_ms=ms)


def test_detect_plateau_true_when_no_improvement_over_window():
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.68), _rec(4, 2.69)]
    assert detect_plateau(recs, window=3) is True


def test_detect_plateau_false_when_recent_new_best():
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.50)]
    assert detect_plateau(recs, window=3) is False


def test_detect_plateau_a_sub_noise_new_best_is_not_progress():
    # 2.66 -> 2.65 is a 0.38% "improvement" — well inside the ~1.37% measurement
    # noise. Under the old absolute eps (0.005 ms) this lucky draw counted as a
    # new best, reset the window, and kept a dead phase grinding; it must read
    # as plateau.
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.65), _rec(4, 2.68)]
    assert detect_plateau(recs, window=3) is True


def test_detect_plateau_a_supra_noise_improvement_is_progress():
    # 2.66 -> 2.60 is 2.26% — distinguishable from noise, so not a plateau
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.60), _rec(4, 2.68)]
    assert detect_plateau(recs, window=3) is False


def test_detect_plateau_needs_enough_points():
    recs = [_rec(0, 3.0), _rec(1, 2.66)]
    assert detect_plateau(recs, window=3) is False


def _rec2(it, e2e, dev):
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict="KEEP", bwd_median_ms=e2e, device_median_ms=dev)


def test_detect_plateau_watches_the_gate_window():
    # device window keeps improving while e2e is flat: a device-gated run is
    # making progress (kernel getting better under a constant host tax), an
    # e2e-gated one has plateaued. Same records, opposite verdicts by window.
    recs = [_rec2(0, 0.290, 0.280), _rec2(1, 0.280, 0.270), _rec2(2, 0.280, 0.262),
            _rec2(3, 0.281, 0.254), _rec2(4, 0.280, 0.246)]
    assert detect_plateau(recs, window=3, gate_window="device") is False
    assert detect_plateau(recs, window=3, gate_window="e2e") is True


def test_detect_plateau_device_gate_skips_records_without_the_window():
    # records with no device reading (e.g. pre-migration history) are not
    # evaluated points for a device gate — they must not count toward the window
    recs = [_rec(0, 3.0), _rec(1, 2.9), _rec(2, 2.8), _rec(3, 2.7)]
    assert detect_plateau(recs, window=3, gate_window="device") is False


# --- the device window's precondition reaches the trajectory (not just the log) --
#
# A bench can print host_enqueue_ms / host_bound_ratio all it likes; if the driver's
# perf parsing does not map them they arrive as None in status.jsonl, the criterion
# never gates anything and never reaches the student, and "written but not executed"
# is the same as not written.

_PERF_WITH_HOST = (
    '{"correctness": {"pass": true}, "performance": {"bwd_median_ms": 0.52, '
    '"device_median_ms": 0.4085, "host_enqueue_ms": 0.004, '
    '"host_bound_ratio": 0.0098, "cublas_device_ms": 0.4889, '
    '"cublas_host_bound_ratio": 0.964, "vs_cublas_device": 1.196}}')


def test_parse_eval_reads_the_enqueue_clock():
    ev = parse_eval(_PERF_WITH_HOST)
    assert ev.host_enqueue_ms == 0.004
    assert ev.host_bound_ratio == 0.0098


def test_parse_eval_reads_the_in_job_reference_verdict_too():
    ev = parse_eval(_PERF_WITH_HOST)
    assert ev.ref_host_bound_ratio == 0.964


def test_the_keys_are_optional_for_benches_that_predate_them():
    ev = parse_eval('{"correctness": {"pass": true}, '
                    '"performance": {"bwd_median_ms": 2.0, "device_median_ms": 1.8}}')
    assert ev.host_enqueue_ms is None
    assert ev.host_bound_ratio is None
    assert ev.ref_host_bound_ratio is None
    assert ev.device_window_basis is None
    assert ev.ref_host_enqueue_ms is None


def test_parse_eval_reads_the_window_basis_and_the_reference_enqueue_clock():
    # a dsa_sparse card's eval: span basis, host_bound_ratio deliberately null,
    # the enqueue clocks of both sides reported.
    ev = parse_eval('{"correctness": {"pass": true}, "performance": {'
                    '"bwd_median_ms": 0.08398, "device_median_ms": 0.01522, '
                    '"device_window_basis": "cupti_kernel_span", '
                    '"host_enqueue_ms": 0.06463, "host_bound_ratio": null, '
                    '"host_bound_ratio_plain": 4.246, '
                    '"ref_device_ms": 0.00957, "ref_host_enqueue_ms": 0.01911, '
                    '"ref_host_bound_ratio": null}}')
    assert ev.device_window_basis == "cupti_kernel_span"
    assert ev.host_bound_ratio is None
    assert ev.host_enqueue_ms == 0.06463
    assert ev.ref_host_enqueue_ms == 0.01911


# --- per-contract verdicts: what a probe printed is what gets recorded ----

def test_fail_with_na_is_reported_as_na_not_unreported():
    # observed: 30 of 62 verdict-probe FAILs logged as "max_diff=unreported" —
    # the lines had said max_diff=n/a (a structural contract) and the reading
    # was dropped because it was not a float.
    v = parse_probe_verdict("PROBE FAIL max_diff=n/a  # contract 2: D0.13 retracted form")
    assert v.passed is False and v.found is True
    assert v.diff_text == "n/a"
    assert v.contracts[0].note == "contract 2: D0.13 retracted form"


def test_fail_keeps_a_reading_string_verbatim():
    v = parse_probe_verdict("PROBE FAIL max_diff=d_wait=-6.7pp,student_wait=52.6,ref_wait=59.3  # c2")
    assert v.max_diff is None
    assert v.diff_text == "d_wait=-6.7pp,student_wait=52.6,ref_wait=59.3"


def test_fail_without_any_max_diff_is_the_only_unreported_case():
    v = parse_probe_verdict("PROBE FAIL")
    assert v.diff_text == "unreported" and v.contracts[0].raw is None


def test_multi_contract_probe_keeps_every_line_and_names_the_failures():
    log = ("PROBE FAIL max_diff=n/a  # c1 (control): a segment variant returned no reading\n"
           "PROBE PASS max_diff=ref_exposed=106.3,student_exposed=159.6  # c3 (decisive)\n"
           "PROBE FAIL max_diff=-0.0308  # c4 (control): watermark not outlier-poisoned\n")
    v = parse_probe_verdict(log, audit=False)
    assert v.passed is False
    assert [c.verdict for c in v.contracts] == ["FAIL", "PASS", "FAIL"]
    assert v.max_diff == -0.0308            # the first numeric FAIL reading
    assert v.summary.startswith("1/3 PASS; FAIL #1 FAIL max_diff=n/a — c1 (control)")
    assert "#3 FAIL max_diff=-0.0308" in v.summary
    assert "c3 (decisive)" in v.contract_table()
    assert "2 of 3 PROBE lines reported FAIL" in v.reason


def test_single_line_probe_has_no_summary():
    assert parse_probe_verdict("PROBE PASS max_diff=0.0").summary == ""


def test_verdict_mode_accepts_a_pass_whose_reading_is_not_an_error_norm():
    # certification audit rejects these (they are not error norms); a verdict
    # experiment's PASS is the hypothesis being confirmed and the field is its
    # reading — downgrading it to FAIL told the teacher the opposite.
    for raw in ("ref_exposed=106.3,student_exposed=159.6", "1047.7", "878496.0"):
        line = f"PROBE PASS max_diff={raw}"
        assert parse_probe_verdict(line).passed is False, raw
        v = parse_probe_verdict(line, audit=False)
        assert v.passed is True, raw
        assert v.diff_text == raw


def test_verdict_mode_pass_without_max_diff_is_still_a_pass():
    v = parse_probe_verdict("PROBE PASS  [q1] two_kernels=True", audit=False)
    assert v.passed is True and v.contracts[0].note == "[q1] two_kernels=True"


def test_certification_audit_is_unchanged_by_the_contract_records():
    v = parse_probe_verdict("PROBE PASS max_diff=0.0\nPROBE PASS max_diff=80000000000.0")
    assert v.passed is False and "ceiling" in v.reason and len(v.contracts) == 2


def test_probe_request_path_sheds_markdown_wrapping():
    """`PROBE_REQUEST: notes/x.py`` (a closing backtick, a quote, a paren) is the
    path notes/x.py — the stray character reached the node's shell as an
    unmatched quote and was judged a transport failure four times over."""
    for wrapped in ("`notes/x.py`", "notes/x.py`", "'notes/x.py'", "\"notes/x.py\"",
                    "(notes/x.py)", "notes/x.py.", "**notes/x.py**", "notes/x.py,"):
        turn = parse_student_turn(f"prose\nPROBE_REQUEST: {wrapped}")
        assert isinstance(turn, ProbeRequest) and turn.path == "notes/x.py", wrapped
    turn = parse_student_turn("RECON_REQUEST: `notes/recon9.py`", allow_recon=True)
    assert turn.path == "notes/recon9.py"
    # a leading ./ or _ is part of a path, never wrapping
    assert parse_student_turn("PROBE_REQUEST: ./notes/_p.py").path == "./notes/_p.py"


def test_pass_lines_from_a_job_killed_afterwards_do_not_certify():
    # observed in an attn-bwd run: `T-trace` printed PASS, `S-smoke` hung for the whole
    # 1800 s task timeout, the job died rc=137 — and the tree was certified on
    # that one line. A PASS is a claim about the rungs that printed it; a job
    # killed under the rungs after them ran to no verdict at all.
    log = ("[gpu_entry.sh] timeout=1800s\n"
           "T-trace: lowered in 1.5 s\n"
           "PROBE PASS max_diff=n/a\n"
           "S-smoke: launching the three kernels ...\n"
           "gpu_entry.sh: line 61:  3072 Killed                  timeout --signal=KILL\n"
           "[direct] gpu_entry cmd:python3 notes/p.py exited rc=137\n")
    v = parse_probe_verdict(log)
    assert v.passed is False and v.found is True
    assert "exit 137" in v.reason and "after the last PROBE line" in v.reason
    assert v.n_pass == 1  # the line it did print is kept, as printed
    # the same lines, with the entry's closing line in place, certify as before
    whole = ("[gpu_entry.sh] timeout=1800s\nPROBE PASS max_diff=n/a\n\n"
             "[gpu_entry.sh] task=cmd finished\n")
    assert parse_probe_verdict(whole).passed is True
    # bare verdict text (no entry-script line at all) is still read as printed
    assert parse_probe_verdict("PROBE PASS max_diff=0.0").passed is True


def test_pass_needs_the_entry_closing_line_or_a_clean_exit():
    # the entry script's own KILLED report (when set -e lets it print one) ...
    killed = ("[gpu_entry.sh] timeout=600s\nPROBE PASS max_diff=0.0\n"
              "[gpu_entry.sh] KILLED: task exceeded 600s timeout (likely kernel deadlock)\n")
    v = parse_probe_verdict(killed)
    assert v.passed is False and "killed" in v.reason
    # ... and a job log cut before its closing line, with no kill report at all
    # ... is NO VERDICT, not a rejection: the archiver lags the job (cli._fetch_log
    # hands back a settled-but-unclosed archive on purpose) and nothing in the log
    # blames the probe, so the student is sent back for a run that ends, not to
    # "fix" a probe that passed
    cut = "[gpu_entry.sh] timeout=600s\nPROBE PASS max_diff=0.0\n"
    v = parse_probe_verdict(cut)
    assert v.passed is False and v.found is False and "closing line" in v.reason
    assert "archive is cut" in v.reason and "re-emit the same PROBE_REQUEST" in v.reason
    # a FAIL stays a FAIL (found, with its diff), whatever killed the job after it
    v = parse_probe_verdict("[gpu_entry.sh] timeout=600s\nPROBE FAIL max_diff=2.0\n"
                            "[gpu_entry.sh] KILLED: task exceeded 600s timeout\n")
    assert v.passed is False and v.found is True and v.max_diff == 2.0
    # the verdict-experiment reading (audit=False) records the lines as printed
    assert parse_probe_verdict(cut, audit=False).passed is True


def test_entry_runs_nest_and_only_the_outer_one_decides():
    # A cmd job may run inner entry runs (orch_census streams `bash gpu_entry.sh
    # ncu`; a ladder rung may run one under its own cap). The bookends are walked
    # as a stack: an inner run killed under its own cap and handled by the probe
    # does not veto an outer run that finished ...
    outer, inner = "[gpu_entry.sh] timeout=1800s\n", "[gpu_entry.sh] timeout=600s\n"
    done_ncu, done_cmd = "[gpu_entry.sh] task=ncu finished\n", "[gpu_entry.sh] task=cmd finished\n"
    handled = (outer + inner + "[gpu_entry.sh] KILLED: task exceeded 600s timeout\n"
               "gpu_entry.sh: line 42:  99 Killed                  timeout --signal=KILL\n"
               "[[F]] shipped form hangs (expected); fixed form below\n"
               "PROBE PASS max_diff=0.0\n\n" + done_cmd)
    assert job_abend(handled).kind == JOB_END_WHOLE
    assert parse_probe_verdict(handled).passed is True
    assert parse_probe_verdict(outer + inner + done_ncu + "PROBE PASS max_diff=0.0\n\n" + done_cmd).passed
    # ... and an inner `finished` line does not stand in for an outer run still open
    cut = outer + inner + done_ncu + "PROBE PASS max_diff=0.0\n"
    assert job_abend(cut).kind == JOB_END_CUT
    v = parse_probe_verdict(cut)
    assert v.passed is False and v.found is False
    # the runner's run-timeout line names the kill for an open outer run
    e = job_abend(cut + "[direct] gpu_entry cmd:python3 p.py exceeded 1900s -- killed\n")
    assert e.kind == JOB_END_KILLED
    # the entry script's own KILLED report on the outer run is a kill, whole stack or not
    assert job_abend(outer + "PROBE PASS max_diff=0.0\n[gpu_entry.sh] KILLED: task exceeded "
                     "1800s timeout\n").kind == JOB_END_KILLED


def test_wrapper_kill_lines_are_read_only_around_an_open_outer_run():
    # a probe quoting a child's wrapper line on the FACT channel is not the wrapper:
    # with the outer run closed, the quote is payload
    whole = ("[gpu_entry.sh] timeout=1800s\n[[F]] rA: child Process exited with status 137\n"
             "[[F]] rB: gpu_entry cmd exited rc=1 (sanitizer, expected)\n"
             "PROBE PASS max_diff=n/a\n\n[gpu_entry.sh] task=cmd finished\n")
    assert job_abend(whole).kind == JOB_END_WHOLE
    assert parse_probe_verdict(whole).passed is True
    # the runner's rc line is read at its own line start only
    assert job_abend("[gpu_entry.sh] timeout=1s\nPROBE PASS max_diff=0\n"
                     "[direct] gpu_entry cmd:python3 p.py exited rc=1\n").reason.endswith("(exit 1)")
