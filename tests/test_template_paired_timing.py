"""``measure_paired`` in ``src/forgekernel/card_template/eval/timing.py``: the in-job
reference and the kernel on one warm plateau, mirrored.

Same stub-torch harness as ``test_template_timing.py`` (the CI host has no CUDA);
the fake GPUs here add the two drifts the paired measurement exists to cancel —
a cold part that runs faster for its first calls, and a part that heats up and
slows down along its own timeline.
"""

from __future__ import annotations

import pytest

from test_template_timing import FakeGpu, load_timing


class ColdBoostGpu(FakeGpu):
    """A cold part at full boost: the first ``boost_calls`` calls run ``boost``
    times their steady-state cost, then the plateau."""

    def __init__(self, device_ms: float, *, boost: float = 0.97, boost_calls: int = 300):
        super().__init__(device_ms)
        self.boost = boost
        self.boost_calls = boost_calls

    def call(self):
        self.calls += 1
        factor = self.boost if self.calls <= self.boost_calls else 1.0
        self.clock_ms += self.device_ms * factor


class ThermalDriftGpu(FakeGpu):
    """A part that heats: per-call cost grows linearly with its own timeline."""

    def __init__(self, device_ms: float, *, slope_per_ms: float = 2e-5):
        super().__init__(device_ms)
        self.slope = slope_per_ms

    def call(self):
        self.calls += 1
        self.clock_ms += self.device_ms * (1.0 + self.slope * self.clock_ms)


def _self_ratio_naive(timing, gpu) -> float:
    """What a bench gets from two bare measure_windows calls, ref first."""
    ref = timing.measure_windows(gpu.call)
    kernel = timing.measure_windows(gpu.call)
    return ref["device_median_ms"] / kernel["device_median_ms"]


# --- order and burn-in -------------------------------------------------------

def test_windows_run_ref_kernel_kernel_ref():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_paired(gpu.call, gpu.call, burn_in_s=0)
    assert out["order"] == ["ref", "kernel", "kernel", "ref"]
    assert out["kernel"]["device_readings_ms"] == pytest.approx([1.0, 1.0])
    assert out["ref"]["device_readings_ms"] == pytest.approx([1.0, 1.0])


def test_burn_in_heats_with_the_kernel_before_the_first_window():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    seen_at_first_ref_call = []

    def ref():
        if not seen_at_first_ref_call:
            seen_at_first_ref_call.append(gpu.calls)
        gpu.call()

    out = timing.measure_paired(gpu.call, ref, burn_in_s=0.01)
    assert out["burn_in_rounds"] >= 1
    # the kernel had already run (burn-in) when the reference's first window began
    assert seen_at_first_ref_call[0] >= out["burn_in_rounds"] * timing.BURN_IN_CALLS_PER_ROUND


def test_burn_in_of_zero_seconds_runs_nothing():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    assert timing.burn_in(gpu.call, seconds=0) == 0
    assert gpu.calls == 0


# --- the two drifts it cancels ------------------------------------------------

def test_cold_boost_favours_the_first_side_without_burn_in():
    # The fake models the bias the paired measurement exists for: the same
    # callable, measured as "ref" then as "kernel", does not read 1.0.
    gpu = ColdBoostGpu(device_ms=1.0)
    timing = load_timing(gpu)
    ratio = _self_ratio_naive(timing, gpu)
    assert ratio < 0.995


def test_burn_in_puts_the_same_kernel_on_both_sides_at_one():
    gpu = ColdBoostGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_paired(gpu.call, gpu.call, burn_in_s=0.01)
    perf = timing.paired_performance(out)
    assert perf["vs_ref_device"] == pytest.approx(1.0, abs=1e-6)


def test_thermal_drift_biases_a_single_pair_and_mirroring_cancels_it():
    naive_gpu = ThermalDriftGpu(device_ms=1.0)
    timing = load_timing(naive_gpu)
    naive = _self_ratio_naive(timing, naive_gpu)
    assert abs(naive - 1.0) > 0.005          # ref measured earlier = cooler = faster

    paired_gpu = ThermalDriftGpu(device_ms=1.0)
    timing = load_timing(paired_gpu)
    out = timing.measure_paired(paired_gpu.call, paired_gpu.call, burn_in_s=0)
    ratio = out["ref"]["device_median_ms"] / out["kernel"]["device_median_ms"]
    assert abs(ratio - 1.0) < 0.001          # both sides at the same mean position


# --- the reading each side keeps ---------------------------------------------

def test_device_is_the_mean_of_the_pair_and_e2e_keeps_the_first_window():
    gpu = ThermalDriftGpu(device_ms=1.0, slope_per_ms=1e-4)
    timing = load_timing(gpu)
    out = timing.measure_paired(gpu.call, None, burn_in_s=0)
    k = out["kernel"]
    a, b = k["device_readings_ms"]
    assert b > a
    assert k["device_median_ms"] == pytest.approx((a + b) / 2)
    # host_bound_ratio is recomputed against the mean the gate reads
    assert k["host_bound_ratio"] == pytest.approx(k["host_enqueue_ms"] / k["device_median_ms"])


def test_a_faster_kernel_reads_above_one():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)

    def slower_ref():
        gpu.calls += 1
        gpu.clock_ms += 1.1

    perf = timing.paired_performance(timing.measure_paired(gpu.call, slower_ref, burn_in_s=0))
    assert perf["vs_ref_device"] == pytest.approx(1.1)
    assert perf["ref_device_ms"] == pytest.approx(1.1)
    assert perf["device_median_ms"] == pytest.approx(1.0)
    assert perf["in_job_ref"] == "paired"
    assert perf["vs_ref"] == pytest.approx(perf["ref_median_ms"] / perf["bwd_median_ms"])


def test_the_e2e_key_name_is_the_bench_author_s():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    perf = timing.paired_performance(timing.measure_paired(gpu.call, None, burn_in_s=0),
                                     e2e_key="fwd_median_ms")
    assert "fwd_median_ms" in perf and "bwd_median_ms" not in perf


# --- the single-ref contract when the reference fails ------------------------

def test_no_reference_means_no_ref_keys():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_paired(gpu.call, None, burn_in_s=0)
    assert out["ref"] is None and out["ref_error"] is None
    assert out["order"] == ["kernel", "kernel"]
    perf = timing.paired_performance(out)
    assert not [k for k in perf if k.startswith("ref_") or k.startswith("vs_")]
    assert perf["in_job_ref"] == "absent"


def test_a_reference_that_raises_leaves_the_kernel_reading_intact():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)

    def broken_ref():
        raise RuntimeError("no key in the job environment")

    out = timing.measure_paired(gpu.call, broken_ref, burn_in_s=0)
    assert out["ref"] is None
    assert "RuntimeError" in out["ref_error"]
    assert out["order"] == ["kernel", "kernel"]
    assert out["kernel"]["device_median_ms"] == pytest.approx(1.0)
    perf = timing.paired_performance(out)
    assert "vs_ref_device" not in perf
    assert perf["in_job_ref"].startswith("RuntimeError")


def test_a_reference_that_fails_its_second_window_yields_no_half_pair():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    ref_calls = []

    def flaky_ref():
        ref_calls.append(1)
        if len(ref_calls) > 500:        # past its first window
            raise RuntimeError("device lost")
        gpu.call()

    out = timing.measure_paired(gpu.call, flaky_ref, burn_in_s=0)
    assert out["order"] == ["ref", "kernel", "kernel"]
    assert out["ref"] is None
    assert "device lost" in out["ref_error"]


def test_describe_paired_lists_the_four_readings_in_run_order():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_paired(gpu.call, gpu.call, burn_in_s=0)
    line = timing.describe_paired(out)
    assert line.startswith("mirrored device readings (ms): ref 1.0000 / kernel 1.0000 / "
                           "kernel 1.0000 / ref 1.0000")
    out = timing.measure_paired(gpu.call, None, burn_in_s=0)
    assert "ref n/a" in timing.describe_paired(out)


def test_a_rotation_of_buffers_is_accepted_on_both_sides():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_paired([gpu.call, gpu.call, gpu.call], [gpu.call, gpu.call],
                                burn_in_s=0)
    assert out["ref"]["device_median_ms"] == pytest.approx(1.0)
    assert out["kernel"]["device_median_ms"] == pytest.approx(1.0)


def test_no_kernel_callable_is_a_usage_error():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    with pytest.raises(ValueError):
        timing.measure_paired([], gpu.call, burn_in_s=0)
