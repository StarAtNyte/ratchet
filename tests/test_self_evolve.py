"""scripts/self_evolve.py: each generation's best kernel becomes the next one's reference."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from forgekernel.task import dump_task_toml, load_task, read_task_toml

from test_task import _kb

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("self_evolve", REPO / "scripts" / "self_evolve.py")
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)

KERNEL = "import torch\n\ndef run(x, w):\n    return x * w\n"


def _task(root: Path, name="rmsnorm") -> Path:
    d = root / "tasks" / name
    d.mkdir(parents=True)
    (d / "task.toml").write_text('[task]\nlanguage = "cutedsl"\narch = "sm100"\n'
                                 '[agent.teacher]\nmodel = "claude-opus-5-5"\neffort = "high"\n'
                                 '[budget]\nhours = 2.5\n')
    (d / "description.md").write_text("y = x * w\n")
    (d / "shapes.json").write_text('[{"name": "s", "axes": {"m": 4}}]')
    return d


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """Generations stood in for: each 'run' reports the next scripted speedup."""
    kb = _kb(tmp_path)
    monkeypatch.setattr(se, "load_task", lambda d: load_task(d, kb_root=kb))
    monkeypatch.setattr(se.launch, "repo_root", lambda: tmp_path)
    ran: list[Path] = []
    gains: list = []
    monkeypatch.setattr(se.launch, "cmd_new", lambda argv: ran.append(Path(argv[0])) or 0)
    monkeypatch.setattr(se.launch, "cmd_run", lambda argv: 0)
    monkeypatch.setattr(se, "latest_run", lambda name: SimpleNamespace(dir=tmp_path / "run"))
    def best_of(run, task):
        g = gains[len(ran) - 1]
        if g is None:
            return None
        speedup, ratios = g if isinstance(g, tuple) else (g, [])
        return {"sha": "abc123def456", "speedup": speedup, "ratios": ratios, "kernel": KERNEL}
    monkeypatch.setattr(se.launch, "best_of", best_of)
    orig = se.next_generation
    monkeypatch.setattr(se, "next_generation", lambda task, best, dest, step: orig(
        task, best, dest, step) and dest)
    return tmp_path, ran, gains, kb


def test_dump_task_toml_round_trips(tmp_path):
    d = _task(tmp_path)
    cfg = read_task_toml(d)
    assert tomllib.loads(dump_task_toml(cfg)) == cfg


def test_a_gain_writes_the_next_generation_and_runs_it(chain):
    root, ran, gains, kb = chain
    gains += [1.30, 1.01]
    first = _task(root)
    assert se.main([str(first), "--max-gen", "4"]) == 0
    g2 = root / "tasks" / "rmsnorm-g2"
    assert ran == [first.resolve(), g2]
    assert (g2 / "reference" / "kernel.py").read_text() == KERNEL
    t2 = load_task(g2, kb_root=kb)
    assert t2.mode == "source" and t2.target.ratio == 1.05
    assert t2.reference.run_fn == "run" and t2.budget.hours == 2.5
    assert t2.teacher.effort == "high"
    assert not (root / "tasks" / "rmsnorm-g3").exists()


def test_the_chain_stops_at_max_gen(chain):
    root, ran, gains, _ = chain
    gains += [1.5, 1.5, 1.5]
    assert se.main([str(_task(root)), "--max-gen", "2"]) == 0
    assert [p.name for p in ran] == ["rmsnorm", "rmsnorm-g2"]


def test_there_is_no_generation_cap_by_default(chain):
    root, ran, gains, _ = chain
    gains += [1.2] * 6 + [1.005]
    assert se.main([str(_task(root))]) == 0
    assert [p.name for p in ran] == ["rmsnorm"] + [f"rmsnorm-g{k}" for k in range(2, 8)]


def test_the_chain_stops_at_the_noise_floor(chain):
    root, ran, gains, _ = chain
    # 1.03x clears the 1.5% minimum, but the generation's own same-job ratios spread 4%
    gains += [(1.03, [1.01, 1.03, 1.0504])]
    assert se.main([str(_task(root))]) == 0
    assert len(ran) == 1
    assert not (root / "tasks" / "rmsnorm-g2").exists()


def test_noise_floor():
    assert se.noise_floor({"ratios": []}) == se.NOISE_FLOOR_MIN
    assert se.noise_floor({"ratios": [1.10, 1.101]}) == se.NOISE_FLOOR_MIN
    assert se.noise_floor({"ratios": [1.0, 1.05, 1.02]}) == pytest.approx(0.05)


def test_a_generation_without_a_best_ends_the_chain(chain):
    root, ran, gains, _ = chain
    gains += [None]
    assert se.main([str(_task(root))]) == 1
    assert len(ran) == 1


def test_an_existing_next_generation_is_not_overwritten(chain):
    root, ran, gains, _ = chain
    gains += [1.3]
    first = _task(root)
    (root / "tasks" / "rmsnorm-g2").mkdir()
    with pytest.raises(FileExistsError):
        se.main([str(first)])
