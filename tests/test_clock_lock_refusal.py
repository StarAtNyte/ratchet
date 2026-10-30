"""A GPU that cannot hold the board's clock lock is a machine fault, not a kernel verdict.

A leaderboard card's gpu_entry.sh refuses every timing task when ``nvidia-smi -lgc`` is
accepted but not honoured (rc=3, no eval JSON). Handed back, parse_eval would read that as
"correctness FAIL", the driver would record a REVERT and spend an iteration, and the
confirmation and meta tools would spend their attempts on it. RealBackend.submit_gpu
recognises the refusal (jobs.CLOCK_LOCK_REFUSED_MARKER), never hands such a log to a
caller, resubmits a few times, then stops with ClockLockRefusedError (a NoRunError).
"""

import pytest

from forgekernel.loop import cli
from forgekernel.loop.driver import NoRunError
from forgekernel.loop.parse import parse_eval
from forgekernel.submit import jobs

REFUSED_LOG = """\
=== gpu_entry.sh (010_gemm_n6144_k4096, sol-execbench) ===
  task:      all
  GPU:       NVIDIA B200, 1155 MHz, 3996 MHz, 1965 MHz
  clocks:    lock not held (asked sm=1500 mem=3996, read 1155,3996)

[gpu_entry.sh] REFUSED: this card times under the board's clock lock, and it did not hold here (lock not held (asked sm=1500 mem=3996, read 1155,3996)). LB_REQUIRE_CLOCK_LOCK=0 times anyway (the readings are then not comparable to the board's).
"""
LOCKED_LOG = """\
=== gpu_entry.sh (010_gemm_n6144_k4096, sol-execbench) ===
  task:      all
  clocks:    locked sm=1500 mem=3996
=== JSON ===
{"correctness": true, "bwd_median_ms": 0.5}
=== gpu_entry.sh done (task=all) ===
"""


def _backend(monkeypatch, logs):
    be = cli.RealBackend("op", None, None, "t", "t@t")
    be.clock_lock_retry_sleep_s = 0
    seen = []

    def once(branch, mode):
        seen.append(mode)
        be.last_job_id = str(len(seen))
        return logs.pop(0)
    monkeypatch.setattr(be, "_submit_gpu_once", once)
    return be, seen


def test_the_refusal_line_is_read_at_the_start_of_a_line_only():
    assert jobs.clock_lock_refusal(REFUSED_LOG).startswith(jobs.CLOCK_LOCK_REFUSED_MARKER)
    nested = "\n".join("  " + ln for ln in REFUSED_LOG.splitlines())
    assert jobs.clock_lock_refusal(nested) is None
    assert jobs.clock_lock_refusal(LOCKED_LOG) is None


def test_a_refused_job_is_resubmitted_and_never_handed_back(monkeypatch):
    be, seen = _backend(monkeypatch, [REFUSED_LOG, LOCKED_LOG])
    assert be.submit_gpu("b", "all") == LOCKED_LOG
    assert seen == ["all", "all"]


def test_a_gpu_that_keeps_refusing_stops_the_submit_without_a_verdict(monkeypatch):
    be, seen = _backend(monkeypatch, [REFUSED_LOG] * 5)
    with pytest.raises(cli.ClockLockRefusedError) as e:
        be.submit_gpu("b", "all")
    assert isinstance(e.value, NoRunError)
    assert len(seen) == be.clock_lock_retries and "read 1155,3996" in str(e.value)


def test_parse_eval_names_the_refusal_rather_than_a_dead_job():
    ev = parse_eval(REFUSED_LOG)
    assert not ev.correctness and "clock lock did not hold" in ev.error
