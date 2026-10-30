"""The template bench timing module, exercised WITHOUT a GPU.

``src/forgekernel/card_template/eval/timing.py`` is copied verbatim into every operator card
and into every generated bench, so it is the one file whose behaviour is shared by
every measurement the project makes -- and until now nothing tested it, because it
imports torch and the CI host has no CUDA. A stub torch fixes that: the pieces under
test are the loop structure and the verdict logic, and both are hardware-independent.

The stub models the one thing that matters here: a pipelined window measures
max(device cost, host cost) per call, because whichever side is slower is the
pipeline. That is exactly the asymmetry the host-bound check exists to detect.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
import types
import warnings
from pathlib import Path

import pytest

TIMING_PY = (Path(__file__).resolve().parents[1]
             / "src" / "forgekernel" / "card_template" / "eval" / "timing.py")

EMPTY_GRAPH_WARNING = (
    "The CUDA Graph is empty. This usually means that the graph was attempted to be "
    "captured on wrong device or stream.")


class FakeGpu:
    """A device whose per-call cost is known, driven by a host of known cost."""

    def __init__(self, device_ms: float, host_ms: float = 0.0,
                 graph_mode: str = "ok", graph_ms: float | None = None):
        self.device_ms = device_ms
        self.host_ms = host_ms
        self.graph_mode = graph_mode
        self.graph_ms = device_ms if graph_ms is None else graph_ms
        self.clock_ms = 0.0      # device timeline the fake events read
        self.calls = 0
        self.replays = 0
        self.captured = 0

    def call(self):
        """One call of the thing under test: host pays, then the device pays."""
        self.calls += 1
        if self.host_ms:
            time.sleep(self.host_ms / 1000.0)
        # The window advances by whichever side is the pipeline.
        self.clock_ms += max(self.device_ms, self.host_ms)

    def replay(self):
        self.replays += 1
        self.clock_ms += self.graph_ms


def _fake_torch(gpu: FakeGpu):
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")

    class Event:
        def __init__(self, enable_timing=False):
            self.t = 0.0

        def record(self):
            self.t = gpu.clock_ms

        def elapsed_time(self, other):
            return other.t - self.t

    class Stream:
        def wait_stream(self, other):
            return None

    class CUDAGraph:
        def replay(self):
            gpu.replay()

    @contextlib.contextmanager
    def stream(_s):
        yield

    @contextlib.contextmanager
    def graph(_g):
        if gpu.graph_mode == "raise":
            raise RuntimeError("capture failed: stream is not capturing")
        yield
        gpu.captured += 1
        if gpu.graph_mode == "empty":
            warnings.warn(EMPTY_GRAPH_WARNING)

    cuda.synchronize = lambda: None
    cuda.Event = Event
    cuda.Stream = Stream
    cuda.current_stream = Stream
    cuda.stream = stream
    cuda.CUDAGraph = CUDAGraph
    cuda.graph = graph
    torch.cuda = cuda
    torch.equal = lambda a, b: getattr(a, "data", a) == getattr(b, "data", b)
    return torch


def load_timing(gpu: FakeGpu):
    """Import the template module fresh against a stub torch."""
    saved = sys.modules.get("torch")
    sys.modules["torch"] = _fake_torch(gpu)
    try:
        spec = importlib.util.spec_from_file_location("_tmpl_timing", TIMING_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


# --- the device window's precondition ----------------------------------------

def test_gpu_bound_call_reports_a_small_host_bound_ratio():
    # 1 ms of device work behind a thin python path: the host runs ahead, the
    # pipeline stays full, and the device reading is the kernel.
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_windows(gpu.call)
    assert out["device_median_ms"] == pytest.approx(1.0)
    assert out["host_enqueue_ms"] - out["device_median_ms"] < 0
    assert out["host_bound_ratio"] - timing.HOST_BOUND_RATIO_WARN < 0


def test_host_bound_call_reports_a_ratio_near_one():
    # A thick wrapper (0.4 ms of python) in front of a light kernel (0.05 ms):
    # the stream drains every call, so the "device" reading is the wrapper.
    gpu = FakeGpu(device_ms=0.05, host_ms=0.4)
    timing = load_timing(gpu)
    out = timing.measure_windows(gpu.call)
    assert out["host_bound_ratio"] - timing.HOST_BOUND_RATIO_WARN >= 0
    # At or past 1: the host IS the pipeline, so nothing of the kernel is visible
    # in the device reading. (Sleep granularity puts it slightly over 1 here.)
    assert out["host_bound_ratio"] - 0.8 >= 0


def test_the_two_historical_windows_are_unchanged_by_the_new_key():
    # The e2e/device numbers are the anchors every archived result is comparable
    # against; the precondition check is additive and must not touch them.
    gpu = FakeGpu(device_ms=2.0)
    timing = load_timing(gpu)
    out = timing.measure_windows(gpu.call)
    assert sorted(out) == ["device_median_ms", "host_bound_ratio", "host_enqueue_ms",
                           "median_ms", "min_ms"]
    assert out["device_median_ms"] == pytest.approx(2.0)


def test_host_bound_ratio_is_absent_rather_than_zero_without_a_device_reading():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    assert timing.host_bound_ratio(0.5, None) is None
    assert timing.host_bound_ratio(None, 0.5) is None
    assert timing.host_bound_ratio(0.5, 1.0) == pytest.approx(0.5)


def test_the_enqueue_clock_does_not_synchronize_inside_its_loop():
    # If it synchronized per call it would measure the DEVICE, and a GPU-bound
    # kernel would look host-bound -- the check would invert.
    src = TIMING_PY.read_text(encoding="utf-8")
    body = src.split("def time_host_enqueue")[1].split("def host_bound_ratio")[0]
    loop = body.split("t0 = time.perf_counter()")[1].split("elapsed_ms")[0]
    assert "synchronize" not in loop


# --- the graph window ---------------------------------------------------------
#
# Its whole value is as the reading to trust when the pipelined one is measuring
# python, so a graph reading that cannot be defended is worse than none at all.

def test_graph_replay_is_available_and_removes_the_host():
    gpu = FakeGpu(device_ms=0.4, host_ms=0.4, graph_mode="ok", graph_ms=0.2)
    timing = load_timing(gpu)
    out = timing.time_device_graph(gpu.call, reference_ms=0.4)
    assert out["available"] is True
    assert out["device_median_ms"] == pytest.approx(0.2)
    assert gpu.replays


def test_an_empty_capture_is_rejected_not_reported_as_a_fast_kernel():
    # PyTorch only WARNS on an empty capture, and replaying nothing is very fast:
    # a live CuTe DSL kernel captured empty and "measured" 0.0013 ms against its
    # true 0.4 ms. Taking that number would have been the worst possible outcome.
    gpu = FakeGpu(device_ms=0.4, graph_mode="empty", graph_ms=0.0013)
    timing = load_timing(gpu)
    out = timing.time_device_graph(gpu.call, reference_ms=0.4)
    assert out["available"] is False
    assert out["device_median_ms"] is None
    assert "empty" in out["reason"].lower()


def test_an_implausibly_fast_replay_is_rejected_even_without_a_warning():
    # Second line of defence: removing the host can only take the reading down to
    # the device time it was hiding, so 1000x faster is not a measurement.
    gpu = FakeGpu(device_ms=0.4, graph_mode="ok", graph_ms=0.0004)
    timing = load_timing(gpu)
    out = timing.time_device_graph(gpu.call, reference_ms=0.4)
    assert out["available"] is False
    assert "empty" in out["reason"].lower()


def test_a_capture_that_raises_degrades_gracefully():
    gpu = FakeGpu(device_ms=0.4, graph_mode="raise")
    timing = load_timing(gpu)
    out = timing.time_device_graph(gpu.call, reference_ms=0.4)
    assert out["available"] is False
    assert "RuntimeError" in out["reason"]


def test_measure_windows_with_graph_always_states_the_graph_status():
    gpu = FakeGpu(device_ms=0.4, graph_mode="empty", graph_ms=0.0013)
    timing = load_timing(gpu)
    out = timing.measure_windows_with_graph(gpu.call)
    # the plain windows still there, the graph reading absent, and WHY it is absent
    # reported rather than silently one window short
    assert out["device_median_ms"] == pytest.approx(0.4)
    assert "graph_device_median_ms" not in out
    assert "empty" in out["graph_status"].lower()


def test_measure_windows_with_graph_carries_a_defensible_reading():
    gpu = FakeGpu(device_ms=0.4, host_ms=0.4, graph_mode="ok", graph_ms=0.2)
    timing = load_timing(gpu)
    out = timing.measure_windows_with_graph(gpu.call)
    assert out["graph_device_median_ms"] == pytest.approx(0.2)


def test_plausibility_floor_is_silent_without_a_reference():
    gpu = FakeGpu(device_ms=0.4)
    timing = load_timing(gpu)
    assert timing.implausible_graph_reason(0.0001, None) == ""
    assert timing.implausible_graph_reason(0.2, 0.4) == ""
    assert timing.implausible_graph_reason(0.001, 0.4) != ""


# --- the prep step: work the timed call receives from OUTSIDE the window -------

class FakeTensor:
    def __init__(self, shape, dtype, data=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data = data

    def clone(self):
        return FakeTensor(self.shape, self.dtype, self.data)


def test_prep_runs_once_outside_the_windows_and_is_described_and_timed():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    seen = []

    def prep():
        gpu.clock_ms += 0.25      # the prep step's own device cost, on the same clock
        return {"A": FakeTensor((4, 32, 64), "torch.bfloat16"),
                "g_cumsum": FakeTensor((4, 32), "torch.float32")}

    def timed(saved):
        seen.append(sorted(saved))
        gpu.call()

    out = timing.measure_windows(timed, prep=prep)
    # the timed call got the prep's tensors, every call
    assert seen and all(s == ["A", "g_cumsum"] for s in seen)
    # the windows read the timed call alone: prep's cost is not in them
    assert out["device_median_ms"] == pytest.approx(1.0)
    # and the prep block describes what left the window, dtype without "torch."
    assert out["prep"]["outputs"] == {"A": {"shape": [4, 32, 64], "dtype": "bfloat16"},
                                      "g_cumsum": {"shape": [4, 32], "dtype": "float32"}}
    assert out["prep"]["median_ms"] == pytest.approx(0.25)
    assert out["prep"]["device_median_ms"] == pytest.approx(0.25)
    assert out["prep"]["mutated"] == []      # read-only tensors, untouched


@pytest.mark.parametrize("tamper, expect", [
    (lambda saved: setattr(saved["A"], "data", 7), ["A"]),          # wrote into the buffer
    (lambda saved: saved.__setitem__("A", FakeTensor((4, 32, 64), "torch.bfloat16", 9)),
     ["A"]),                                                          # swapped the entry
    (lambda saved: saved.__setitem__("_cache", FakeTensor((1,), "x")), ["_cache"]),  # new key
    (lambda saved: setattr(saved["A"], "_stash", 1), ["A"]),         # hung state on the tensor
])
def test_a_timed_call_that_keeps_state_in_the_prep_tensors_is_reported(tamper, expect):
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)

    def prep():
        return {"A": FakeTensor((4, 32, 64), "torch.bfloat16", 3)}

    def timed(saved):
        tamper(saved)
        gpu.call()

    out = timing.measure_windows(timed, prep=prep)
    assert out["prep"]["mutated"] == expect
    # the report describes the tensors as prep produced them, not as tampered
    assert out["prep"]["outputs"] == {"A": {"shape": [4, 32, 64], "dtype": "bfloat16"}}


def test_prep_must_return_a_named_dict():
    timing = load_timing(FakeGpu(device_ms=1.0))
    with pytest.raises(TypeError, match="dict"):
        timing.measure_windows(lambda saved: None, prep=lambda: [FakeTensor((1,), "x")])


def test_without_prep_the_windows_are_unchanged_and_report_no_prep_block():
    gpu = FakeGpu(device_ms=1.0)
    timing = load_timing(gpu)
    out = timing.measure_windows(gpu.call)
    assert "prep" not in out and out["device_median_ms"] == pytest.approx(1.0)


def test_rotating_and_graph_variants_take_the_prep_step_too():
    gpu = FakeGpu(device_ms=0.4, graph_mode="ok", graph_ms=0.3)
    timing = load_timing(gpu)
    prep = lambda: {"A": FakeTensor((2, 2), "torch.float16")}  # noqa: E731
    got = []
    fns = [lambda saved: (got.append(sorted(saved)), gpu.call()) for _ in range(2)]
    out = timing.measure_windows_rotating(fns, prep=prep)
    assert out["prep"]["outputs"]["A"]["dtype"] == "float16" and got and got[0] == ["A"]
    out = timing.measure_windows_with_graph(lambda saved: gpu.call(), prep=prep)
    assert out["prep"]["outputs"]["A"]["shape"] == [2, 2] and out["prep"]["mutated"] == []
    assert out["graph_device_median_ms"] == pytest.approx(0.3)
