"""forgekernel.runs (run directories) and forgekernel.localenv (the machine)."""

from __future__ import annotations

import json

import pytest

from forgekernel import paths
from forgekernel.localenv import MachineError, load_machine, read_env_file
from forgekernel.runs import RunError, list_runs, new_run, open_run
from forgekernel.task import load_task

from test_task import MINIMAL, _kb, _task


# -- local.env ------------------------------------------------------------------------------


def test_no_local_env_means_this_machine_gpu_0(tmp_path):
    m = load_machine(tmp_path, environ={})
    assert (m.delivery, m.gpu, m.remote_host) == ("local", "0", "")
    assert m.describe() == "this machine (GPU 0)"


def test_local_env_parses_comments_exports_and_quotes(tmp_path):
    (tmp_path / "local.env").write_text(
        "# machine\nFK_DELIVERY=ssh\nexport REMOTE_HOST=gpu-box\n"
        "REMOTE_WORKDIR='/data/fk'\nANTHROPIC_API_KEY=\"sk-test x\"\n\n")
    m = load_machine(tmp_path, environ={})
    assert (m.delivery, m.remote_host, m.remote_workdir) == ("ssh", "gpu-box", "/data/fk")
    assert m.env["ANTHROPIC_API_KEY"] == "sk-test x"


def test_the_environment_wins_over_the_file(tmp_path):
    (tmp_path / "local.env").write_text("FK_DELIVERY=ssh\nREMOTE_HOST=a\nREMOTE_WORKDIR=/d/fk\n")
    m = load_machine(tmp_path, environ={"FK_DELIVERY": "local", "FK_GPU": "3"})
    assert (m.delivery, m.gpu) == ("local", "3")


def test_ssh_needs_a_host_and_a_dedicated_workdir(tmp_path):
    with pytest.raises(MachineError) as e:
        load_machine(tmp_path, environ={"FK_DELIVERY": "ssh", "REMOTE_WORKDIR": "/tmp"})
    text = "\n".join(e.value.problems)
    assert "needs REMOTE_HOST" in text and "needs REMOTE_WORKDIR" in text


def test_a_malformed_line_is_refused_not_dropped(tmp_path):
    (tmp_path / "local.env").write_text("ANTHROPIC_API_KEY sk-oops\n")
    with pytest.raises(MachineError, match="line 1 is not KEY=VALUE"):
        read_env_file(tmp_path / "local.env")


def test_unknown_delivery_is_refused(tmp_path):
    with pytest.raises(MachineError, match="FK_DELIVERY must be one of"):
        load_machine(tmp_path, environ={"FK_DELIVERY": "carrier-pigeon"})


# -- runs -----------------------------------------------------------------------------------


def test_a_new_run_snapshots_the_task(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    run = new_run(t, root=tmp_path / "runs", machine=load_machine(tmp_path, environ={}))
    assert run.dir.parent == tmp_path / "runs" and run.dir.name.startswith("rmsnorm_")
    assert (run.task_dir / "task.toml").read_text() == MINIMAL
    rec = run.record()
    assert rec["task"] == "rmsnorm" and rec["status"] == "created"
    assert rec["machine"] == {"delivery": "local", "gpu": "0", "host": None}
    # the snapshot, not the source, is what a resumed run reads
    reopened = open_run(run.dir)
    assert reopened == run
    assert load_task(reopened.task_dir, kb_root=_kb(tmp_path), name=run.task_name).name == "rmsnorm"


def test_two_runs_of_one_task_get_their_own_directories(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    a = new_run(t, root=tmp_path / "runs", now=lambda: 0)
    b = new_run(t, root=tmp_path / "runs", now=lambda: 0)
    assert a.dir != b.dir
    assert [r.name for r in list_runs(tmp_path / "runs")] == sorted([a.name, b.name])


def test_a_run_whose_snapshot_was_edited_cannot_be_resumed(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    run = new_run(t, root=tmp_path / "runs")
    (run.task_dir / "task.toml").write_text(MINIMAL + "[target]\nratio = 2.0\n")
    with pytest.raises(RunError, match="edited after launch"):
        open_run(run.dir)


def test_open_run_refuses_a_plain_directory(tmp_path):
    with pytest.raises(RunError, match="not a run directory"):
        open_run(tmp_path)


def test_update_keeps_the_record_whole(tmp_path):
    t = load_task(_task(tmp_path, MINIMAL), kb_root=_kb(tmp_path))
    run = new_run(t, root=tmp_path / "runs")
    run.update(status="running")
    rec = json.loads((run.dir / "run.json").read_text())
    assert rec["status"] == "running" and rec["task"] == "rmsnorm"


# -- paths ----------------------------------------------------------------------------------


def test_operators_dir_follows_the_run(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.OPERATORS_ENV, raising=False)
    assert paths.operators_dir(tmp_path) == tmp_path / "operators"
    monkeypatch.setenv(paths.OPERATORS_ENV, str(tmp_path / "run" / "operators"))
    assert paths.operators_dir(tmp_path) == (tmp_path / "run" / "operators").resolve()
