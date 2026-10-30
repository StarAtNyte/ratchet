"""Node hygiene across the seeded probe scripts.

The lever_pricing_run half lives in test_lever_pricing_run.py; this file pins
the timeline probe's copy of the same contract (the script is seeded standalone,
so the helpers are duplicated, and both copies must kill a timed-out command as a
PROCESS GROUP and report a GPU that stays held as NODE-DIRTY).
"""
import os
import time

import forgekernel.skills.timeline_probe as tp


def test_timeline_sh_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(tp, "ROOT", str(tmp_path))
    monkeypatch.setattr(tp, "_gpu_held_pids", lambda: [])
    pidfile = tmp_path / "child.pid"
    rc, _out, err = tp._sh(f"sh -c 'echo $$ > {pidfile}; exec sleep 30' & sleep 30", timeout=1)
    assert rc == 124 and "TIMEOUT" in err
    pid = int(pidfile.read_text().strip())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        raise AssertionError("the grandchild survived the timeout (no group kill)")


def test_timeline_sweep_prints_node_dirty_when_the_gpu_stays_held(monkeypatch, capsys):
    monkeypatch.setattr(tp, "_looks_like_ours", lambda pid: False)
    monkeypatch.setattr(tp, "_gpu_held_pids", lambda: [31])
    assert tp._sweep_gpu("after the stamped bench timed out", wait_s=0.0, sleep=lambda s: None) == [31]
    assert "NODE-DIRTY after the stamped bench timed out: pid(s) [31] still hold the GPU" in capsys.readouterr().out
    seq = iter([[31], []])
    monkeypatch.setattr(tp, "_gpu_held_pids", lambda: next(seq))
    assert tp._sweep_gpu("x", wait_s=5.0, sleep=lambda s: None) == []
    assert "GPU released" in capsys.readouterr().out
    monkeypatch.setattr(tp, "_gpu_held_pids", lambda: [])
    assert tp._sweep_gpu("x") == [] and capsys.readouterr().out == ""


def test_looks_like_ours_never_names_the_runner_itself():
    # nvidia-smi reports driver-side pids: our own pid, or a number that is not
    # a python/ncu in our namespace, must never receive the SIGKILL
    assert tp._looks_like_ours(os.getpid()) is False
    assert tp._looks_like_ours(2 ** 22 + 12345) is False   # no such /proc entry


def test_timeline_sweep_treats_an_unavailable_nvidia_smi_as_node_dirty(monkeypatch, capsys):
    monkeypatch.setattr(tp, "_gpu_held_pids", lambda: None)
    assert tp._sweep_gpu("after the stamped bench timed out", wait_s=0.0, sleep=lambda s: None) == []
    out = capsys.readouterr().out
    assert "NODE-DIRTY after the stamped bench timed out: nvidia-smi unavailable" in out
