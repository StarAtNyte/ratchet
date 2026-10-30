"""Tests for the ref baseline: measure once, archive, anchor every gate to it."""

from __future__ import annotations

import json

import pytest

from forgekernel.baseline import (
    BASIS_CUPTI,
    HOST_FREE_BASES,
    BASIS_GRAPH,
    BASIS_PLAIN,
    SCHEMA_VERSION,
    RefBaseline,
    baseline_path,
    load_baseline,
    save_baseline,
)
from forgekernel.baseline_cli import _looks_like_ncu_csv, _measure, parse_ref_stdout


def _b(**kw) -> RefBaseline:
    return RefBaseline(**{"op": "demo", "ms": 2.3641, **kw})


# --- archive round-trip ----------------------------------------------------

def test_save_then_load_round_trips(tmp_path):
    saved = _b(job="42", ref_name="cuDNN", tflops=363.4, mfu=36.7,
               ncu={"tensor_pct": 61.0})
    save_baseline(tmp_path, saved)
    loaded = load_baseline(tmp_path)
    assert loaded == saved


def test_absent_baseline_is_none_not_an_error(tmp_path):
    """Nothing has measured yet — the normal state before `forge <op> baseline`."""
    assert load_baseline(tmp_path) is None


def test_corrupt_archive_raises_rather_than_reading_half(tmp_path):
    baseline_path(tmp_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        load_baseline(tmp_path)


def test_stale_schema_is_refused(tmp_path):
    """A gate anchored to a misread baseline is worse than no gate."""
    baseline_path(tmp_path).write_text(
        json.dumps({"op": "demo", "ms": 1.0, "schema_version": SCHEMA_VERSION + 1}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_baseline(tmp_path)


def test_unknown_fields_are_ignored_not_fatal(tmp_path):
    """Forward-compat within a schema version: an extra key must not break the load."""
    baseline_path(tmp_path).write_text(
        json.dumps({"op": "demo", "ms": 1.5, "schema_version": SCHEMA_VERSION,
                    "some_future_field": 7}),
        encoding="utf-8")
    assert load_baseline(tmp_path).ms == 1.5


# --- target derivation -----------------------------------------------------

def test_target_inverts_the_throughput_ratio():
    """95% of ref throughput is ~5.3% more latency, not 5%."""
    assert _b(ms=2.0, ms_device=2.0).target_ms(0.95) == pytest.approx(2.10526, rel=1e-4)


def test_target_of_one_is_the_ref_itself():
    assert _b(ms=2.0, ms_device=2.0).target_ms(1.0) == 2.0


def test_a_ratio_above_one_asks_to_beat_the_reference():
    # self-evolution runs: 1.05 = 5% more throughput than the reference, so the
    # stop threshold is anchor / 1.05 (a bwd run, refused this, passed
    # 0.952381 instead and got a target 5% SLOWER than its reference)
    assert _b(ms=2.0, ms_device=2.0).target_ms(1.05) == pytest.approx(2.0 / 1.05)
    with pytest.raises(ValueError):
        _b(ms=2.0, ms_device=2.0).target_ms(0)


@pytest.mark.parametrize("ratio", [0, -0.5, 1.5])
def test_nonsense_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        _b().target_ms(ratio)


def test_target_derives_from_the_gate_window():
    """A device-gated run's threshold comes from the ref's device reading —
    deriving it from e2e would set the bar at the ref's host tax too."""
    b = _b(ms=0.2796, ms_device=0.2491)
    assert b.target_ms(0.95, gate_window="device") == pytest.approx(0.2491 / 0.95)
    assert b.target_ms(0.95, gate_window="e2e") == pytest.approx(0.2796 / 0.95)
    assert b.target_ms(0.95) == pytest.approx(0.2491 / 0.95)  # default is device


def test_device_gate_over_a_single_window_archive_is_refused():
    """Silently falling back to the e2e number is the exact degradation the
    window parameter exists to prevent."""
    with pytest.raises(ValueError, match="ms_device"):
        _b(ms=0.2796).target_ms(0.95, gate_window="device")


def test_the_architecture_roles_refusal_prescribes_the_declared_fix():
    """An architecture reference may exist only as declared loop.toml values —
    declared BECAUSE it cannot be run — so telling that operator to re-measure
    is a dead end. Its refusal must name arch_ref_device_ms instead."""
    from forgekernel.baseline import ROLE_ARCHITECTURE
    with pytest.raises(ValueError, match="arch_ref_device_ms"):
        _b(ms=1.978, role=ROLE_ARCHITECTURE).target_ms(1.0, gate_window="device")


def test_render_moves_the_target_anchor_marker_with_the_gate_window():
    """The marker is instruction, not decoration: the teacher copies whichever
    line says "the target anchors on this number" into §0, so it must sit on the gate window's line."""
    b = _b(ms=0.2796, ms_device=0.2491)
    e2e = b.render("e2e")
    dev = b.render()  # device is the default window
    assert "0.2796" in e2e.split("\n")[1] and "the target anchors on this number" in e2e.split("\n")[1]
    assert "the target anchors on this number" not in dev.split("\n")[1]      # e2e line loses the marker
    assert "the target anchors on this number" in [ln for ln in dev.split("\n") if "device median" in ln][0]
    # a device-gated run over an archive with no device reading gets an explicit
    # "do not substitute the e2e number" line instead of a silently-wrong anchor
    warn = _b(ms=0.2796).render("device")
    assert "not measured" in warn and "must not be substituted with the end-to-end reading" in warn


# --- prompt rendering ------------------------------------------------------

def test_render_carries_the_job_so_gates_can_cite_it():
    out = _b(job="42", ref_name="cuDNN", hardware="H100").render()
    assert "[MEASURED, job 42]" in out
    assert "cuDNN" in out and "H100" in out


def test_render_states_when_there_is_no_ncu_path():
    """Template A.4: a workload with no NCU path carries no NCU-anchored gate line,
    so the teacher has to be told that rather than left to invent one."""
    out = _b(ncu={}).render()
    assert "no collection path" in out
    assert not _b(ncu={}).has_ncu


def test_render_lists_ncu_metrics_when_present():
    out = _b(ncu={"tensor_pct": 61.0, "spill_bytes": 0}).render()
    assert "tensor_pct: 61.0" in out and "spill_bytes: 0" in out


# --- bench-output parsing --------------------------------------------------

def test_parses_nested_json_from_bench_output():
    text = 'noise\n=== JSON ===\n{"performance": {"median_ms": 2.3641, "mfu_pct": 36.7}}\n'
    assert parse_ref_stdout(text) == {"ms": 2.3641, "mfu": 36.7}


def test_parses_prose_median_when_there_is_no_json():
    assert parse_ref_stdout("cuDNN: median 2.3641 ms over 100 iters")["ms"] == 2.3641


def test_picks_tflops_and_mfu_when_present():
    got = parse_ref_stdout('{"ms": 1.5, "tflops": 363.4, "mfu": 36.7}')
    assert got == {"ms": 1.5, "tflops": 363.4, "mfu": 36.7}


def test_device_window_is_read_from_the_winning_block_only():
    """A ref script that times both windows reports them side by side; the device
    number must come from the same backend block as the e2e number, and a block
    that reports only a device window must not win the e2e slot."""
    text = json.dumps({"backends": {
        "flash": {"median_ms": 3.8319, "device_median_ms": 3.7001},
        "cudnn": {"median_ms": 0.2796, "device_median_ms": 0.2491, "tflops": 491.5},
        "stray": {"device_median_ms": 0.0001}}})  # no e2e key: not a timing block
    got = parse_ref_stdout(text)
    assert got["ms"] == 0.2796 and got["ms_device"] == 0.2491


def test_single_window_ref_carries_no_device_number():
    assert "ms_device" not in parse_ref_stdout('{"ms": 1.5, "tflops": 363.4}')


def test_render_shows_both_windows_and_labels_them():
    """The teacher anchors gates to this block; if the two windows are not labeled,
    the e2e number gets quoted as a kernel-quality number (observed live: the same
    cuDNN kernel is 0.28 ms e2e and 0.25 ms device — a silent 12% convention gap)."""
    b = RefBaseline(op="attn_fwd", ms=0.2796, ms_device=0.2491, job="703021")
    text = b.render()
    assert "0.2796" in text and "0.2491" in text
    assert "host" in text and "device" in text


def test_render_without_device_window_stays_quiet_about_it():
    # ... on an e2e-gated run. Under the default (device) window the same archive
    # must instead warn loudly that the anchor reading is missing.
    b = RefBaseline(op="attn_fwd", ms=2.0, job="1")
    assert "device median" not in b.render("e2e")
    assert "not measured" in b.render()
    assert "must not be substituted with the end-to-end reading" in b.render()


def test_multi_backend_output_picks_the_fastest():
    """A ref bench times several backends; the ref is the fastest of them.

    Taking whichever the JSON lists first would anchor every gate to the slowest
    backend — measure_target.py reports flash before cudnn, and flash is ~1.6x
    slower on the bwd shape.
    """
    text = json.dumps({"backends": {
        "flash": {"median_ms": 3.8319, "tflops": 200.1, "mfu_pct": 20.2},
        "cudnn": {"median_ms": 2.3641, "tflops": 363.4, "mfu_pct": 36.7},
        "efficient": {"error": "NotImplementedError"}}})
    assert parse_ref_stdout(text) == {"ms": 2.3641, "tflops": 363.4, "mfu": 36.7}


def test_throughput_comes_from_the_same_backend_as_the_timing():
    """Mixing the fastest ms with another backend's TFLOPs would be incoherent."""
    text = json.dumps({"a": {"median_ms": 5.0, "tflops": 100.0},
                       "b": {"median_ms": 2.0, "tflops": 250.0}})
    assert parse_ref_stdout(text) == {"ms": 2.0, "tflops": 250.0}


def test_no_latency_found_returns_empty_so_caller_can_refuse():
    """Archiving a baseline with no ms would silently anchor gates to nothing."""
    assert "ms" not in parse_ref_stdout("build finished, no numbers here")


def test_booleans_are_not_mistaken_for_measurements():
    assert "ms" not in parse_ref_stdout('{"ms": true}')


# --- lost-job retry ---------------------------------------------------------

class _FakeBackend:
    """Returns each queued stdout in turn, recording how many submits it saw."""

    def __init__(self, outs):
        self.outs = list(outs)
        self.submits = 0
        self.last_job_id = ""

    def submit_gpu(self, branch, mode):
        self.submits += 1
        self.last_job_id = str(655000 + self.submits)
        return self.outs.pop(0) if self.outs else ""


def _patch_backend(monkeypatch, fake):
    import forgekernel.loop.cli as loop_cli
    monkeypatch.setattr(loop_cli, "RealBackend", lambda *a, **k: fake)


def test_a_lost_job_is_retried(monkeypatch):
    """A killed job reaches a terminal status with no numbers in its log. Archiving that would anchor every gate to a
    partial measurement."""
    fake = _FakeBackend(["node died, no output", json.dumps({"median_ms": 2.36})])
    _patch_backend(monkeypatch, fake)
    perf, _, job, samples, _ = _measure("demo", "bench.py", None, repeat=1)
    assert fake.submits == 2
    assert parse_ref_stdout(perf)["ms"] == 2.36
    assert samples == [2.36]
    assert job == "655002", "the job id must belong to the attempt that produced numbers"


def test_retries_are_bounded(monkeypatch):
    fake = _FakeBackend(["nothing"] * 9)
    _patch_backend(monkeypatch, fake)
    _measure("demo", "bench.py", None, repeat=1, attempts=3)
    assert fake.submits == 3


def test_a_good_first_run_is_not_resubmitted(monkeypatch):
    fake = _FakeBackend([json.dumps({"median_ms": 2.36})])
    _patch_backend(monkeypatch, fake)
    _measure("demo", "bench.py", None, repeat=1)
    assert fake.submits == 1


def test_repeat_keeps_the_fastest_sample(monkeypatch):
    """Measured four times on one operator, the same cuDNN backward returned
    2.3688 / 2.3645 / 2.3555 / 2.8141 ms — the outlier is node contention, not a
    different kernel. A single sample there puts target_ms 19% loose."""
    fake = _FakeBackend([json.dumps({"median_ms": ms})
                         for ms in (2.8141, 2.3555, 2.3688)])
    _patch_backend(monkeypatch, fake)
    perf, _, job, samples, _ = _measure("demo", "bench.py", None, repeat=3)
    assert fake.submits == 3
    assert parse_ref_stdout(perf)["ms"] == 2.3555
    assert samples == [2.8141, 2.3555, 2.3688], "every sample is kept, not just the best"
    assert job == "655002", "the job id must belong to the fastest run"


def test_repeat_survives_a_run_that_never_produced_numbers(monkeypatch):
    """One dead run must not lose the samples that did land."""
    fake = _FakeBackend([json.dumps({"median_ms": 2.5})] + ["dead"] * 3
                        + [json.dumps({"median_ms": 2.4})])
    _patch_backend(monkeypatch, fake)
    _, _, _, samples, _ = _measure("demo", "bench.py", None, repeat=3, attempts=3)
    assert samples == [2.5, 2.4]


def test_ncu_csv_recognised_by_its_header():
    assert _looks_like_ncu_csv('"ID","Process ID","Kernel Name"\n"0","1","foo"')
    assert not _looks_like_ncu_csv("==ERROR== Profiling failed")


# --- spread reporting -------------------------------------------------------

def test_spread_is_none_below_two_samples():
    assert _b(samples_ms=[]).spread_pct is None
    assert _b(samples_ms=[2.0]).spread_pct is None


def test_spread_is_reported_relative_to_the_fastest():
    assert _b(samples_ms=[2.0, 2.4]).spread_pct == pytest.approx(20.0)


def test_render_shows_the_samples_so_variance_is_not_hidden():
    out = _b(ms=2.3555, samples_ms=[2.8141, 2.3555, 2.3688]).render()
    assert "2.8141" in out and "spread" in out


# --- the device window's precondition, on the reference side ------------------
#
# A device reading is device time only while the host keeps the pipeline fed. A
# reference with a thick python wrapper breaks that, and the archived number then
# looks exactly like a clean one -- which is why the basis is archived alongside it.

def _hostbound(**kw):
    """A reference whose wrapper paces the pipeline (enqueue clock ~ the window)."""
    base = dict(op="op", ms=0.60, ms_device=0.4889, host_enqueue_ms=0.4712,
                device_window_basis=BASIS_PLAIN)
    base.update(kw)
    return RefBaseline(**base)


def test_a_host_bound_reference_refuses_to_yield_a_device_target():
    with pytest.raises(ValueError) as e:
        _hostbound().target_ms(1.0, "device")
    assert "HOST-BOUND" in str(e.value)


def test_an_archived_graph_reading_becomes_the_device_anchor():
    b = _hostbound(ms_device_plain=0.4889, ms_device_graph=0.2598)
    assert b.device_anchor_ms == 0.2598
    assert b.device_anchor_basis == BASIS_GRAPH
    assert b.target_ms(1.0, "device") == 0.2598


def test_the_override_exists_and_is_explicit():
    assert _hostbound().target_ms(1.0, "device", allow_host_bound=True) == 0.4889


def test_a_gpu_bound_reference_is_unaffected():
    b = RefBaseline(op="op", ms=0.6, ms_device=0.40, host_enqueue_ms=0.004)
    assert b.host_bound.host_bound is False
    assert b.target_ms(1.0, "device") == 0.40


def test_a_graph_reading_declared_in_ms_device_is_not_read_as_host_bound():
    """The prescribed fix must not be gated by the check that prescribes it.

    An archive may carry the host-free reading in ms_device and say so via the
    basis — which is exactly what declaring arch_ref_device_ms together with
    arch_ref_device_basis='cuda_graph' in loop.toml produces. The enqueue clock
    then has no plain-window reading to divide by: dividing it by a graph replay
    compares the host against a number the host was never in, lands near 1.0, and
    would refuse a reference that is in fact host-free.
    """
    b = RefBaseline(op="op", ms=0.5698, ms_device=0.2598,
                    device_window_basis=BASIS_GRAPH, host_enqueue_ms=0.22)
    assert b.device_anchor_basis == BASIS_GRAPH
    assert b.host_bound.host_bound is False
    assert b.host_bound.measured is False        # nothing to check, not a pass
    assert b.target_ms(1.0, "device") == 0.2598


def test_a_plain_reading_alongside_a_graph_anchor_is_still_checked():
    """Declaring the graph anchor does not switch the precondition check off: with
    a plain reading archived too, the enqueue clock still has its proper divisor."""
    b = RefBaseline(op="op", ms=0.60, ms_device=0.2598,
                    ms_device_plain=0.4889, device_window_basis=BASIS_GRAPH,
                    host_enqueue_ms=0.4712)
    assert b.host_bound.host_bound is True       # the plain window IS host-paced
    assert b.target_ms(1.0, "device") == 0.2598  # but the anchor is host-free


# --- CUPTI kernel span: the second host-free basis (dsa_sparse on B200) ----------
#
# The reference launcher (tvm-ffi) never lands on torch's capture stream, so a
# graph capture comes back EMPTY; the per-launch device span CUPTI records is the
# host-free reading instead. The live archive: plain window 0.01917 ms with the
# enqueue clock at 0.01911 (ratio 0.997 — the python path IS the pipeline on a
# 9.5 µs kernel), span 0.0095 ms declared as the basis of ms_device.

DSA_SPAN = dict(op="dsa", ms=0.03059, ms_device=0.0095, ms_device_plain=0.01917,
                device_window_basis=BASIS_CUPTI, host_enqueue_ms=0.01911,
                host_bound_ratio=0.997)


def test_the_cupti_span_is_a_host_free_basis():
    assert BASIS_CUPTI in HOST_FREE_BASES
    assert BASIS_GRAPH in HOST_FREE_BASES
    assert BASIS_PLAIN not in HOST_FREE_BASES


def test_a_span_anchor_with_a_host_paced_plain_window_still_yields_the_target():
    b = RefBaseline(**DSA_SPAN)
    assert b.device_anchor_basis == BASIS_CUPTI
    assert b.host_bound.host_bound is True          # the plain window IS host-paced
    assert b.target_ms(1.05, "device") == pytest.approx(0.0095 / 1.05)  # anchor is host-free


def test_a_span_reading_declared_in_ms_device_is_not_read_as_host_bound():
    # Same exemption the graph basis has: no plain reading -> nothing to divide by.
    b = RefBaseline(op="dsa", ms=0.03059, ms_device=0.0095,
                    device_window_basis=BASIS_CUPTI, host_enqueue_ms=0.01911)
    assert b.host_bound.host_bound is False
    assert b.host_bound.measured is False
    assert b.target_ms(1.0, "device") == 0.0095


def test_the_render_block_names_the_span_basis():
    text = RefBaseline(**DSA_SPAN).render("device")
    assert "CUPTI kernel span" in text
    assert "host excluded" in text


def test_an_archive_without_the_enqueue_clock_still_derives_a_target():
    # Every baseline written before the check existed is in this state. Refusing
    # them would stop every run instead of the broken ones; the verdict is simply
    # absent, and the CLI says so.
    b = RefBaseline(op="op", ms=0.6, ms_device=0.40)
    assert b.host_bound.measured is False
    assert b.target_ms(1.0, "device") == 0.40
    assert b.device_anchor_basis == BASIS_PLAIN     # unknown is treated as plain


def test_the_render_block_marks_which_window_the_number_came_from():
    b = _hostbound(ms_device_plain=0.4889, ms_device_graph=0.2598, ref_name="lib")
    text = b.render("device")
    assert "0.2598" in text and "graph" in text.lower()
    # and warns the teacher off distilling a host-orchestration lesson from it
    assert "host" in text


def test_new_fields_do_not_break_an_existing_archive(tmp_path):
    # Adding optional fields must not bump the schema: a bump would invalidate every
    # archived baseline in the repo and force a re-measure of all of them.
    b = RefBaseline(op="op", ms=1.0, ms_device=0.9)
    save_baseline(tmp_path, b)
    loaded = load_baseline(tmp_path)
    assert loaded.host_enqueue_ms is None and loaded.device_window_basis == ""
    assert loaded.ms_device == 0.9


def test_ref_stdout_carries_the_window_provenance_into_the_archive():
    # (6): a ref script that reports the basis, the plain reading, the graph reading
    # and the enqueue clock must not have any of it dropped -- a device number whose
    # basis was not recorded cannot be audited afterwards.
    out = parse_ref_stdout(
        '{"performance": {"median_ms": 0.62, "device_median_ms": 0.2598, '
        '"device_window_basis": "cuda_graph", "device_plain_ms": 0.4889, '
        '"graph_device_median_ms": 0.2598, "host_enqueue_ms": 0.4712, '
        '"host_bound_ratio": 0.964}}')
    assert out["ms"] == 0.62
    assert out["ms_device"] == 0.2598
    assert out["device_window_basis"] == "cuda_graph"
    assert out["ms_device_plain"] == 0.4889
    assert out["ms_device_graph"] == 0.2598
    assert out["host_enqueue_ms"] == 0.4712
    assert out["host_bound_ratio"] == 0.964


def test_ref_stdout_without_the_new_keys_is_unchanged():
    out = parse_ref_stdout('{"median_ms": 2.36, "device_median_ms": 2.10}')
    assert out["ms"] == 2.36 and out["ms_device"] == 2.10
    assert "device_window_basis" not in out
    assert "host_enqueue_ms" not in out


def test_a_new_archive_records_the_plain_basis_by_default():
    # device_median_ms IS the pipelined batch window by the timing contract, so a
    # script that does not declare a basis still gets a recorded (not blank) one.
    from forgekernel.baseline_cli import _archived_basis
    assert _archived_basis(dict(ms_device=0.4)) == BASIS_PLAIN
    assert _archived_basis(dict(ms_device=0.4, device_window_basis="cuda_graph")) == BASIS_GRAPH
    assert _archived_basis(dict()) == ""
