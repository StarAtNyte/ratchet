"""Tests for Ratchet's kernel-engine integration."""
from pathlib import Path
from types import SimpleNamespace

from forgekernel.agents import agent_command
from forgekernel.localenv import load_machine
from forgekernel.scaffold import new_operator


def test_ratchet_machine_config_is_external(tmp_path):
    config = tmp_path / 'forge.env'
    config.write_text('FK_DELIVERY=modal\nFK_MODAL_GPU=B200\n')
    machine = load_machine(environ={'RATCHET_FORGE_ENV': str(config)})
    assert machine.delivery == 'modal'
    assert machine.modal_gpu == 'B200'


def test_ratchet_account_model_overrides_task_model(monkeypatch):
    monkeypatch.setenv('RATCHET_KERNEL_MODEL', 'account-model')
    spec = SimpleNamespace(backend='claude', provider='native', model='task-model', effort='', timeout_s=60)
    command = agent_command(spec, 'student')
    assert 'account-model' in command
    assert 'task-model' not in command


def test_packaged_gitignore_template_is_usable(tmp_path):
    import shutil
    root = Path(__file__).resolve().parents[1]
    templates = tmp_path / 'templates'
    shutil.copytree(root / 'src' / 'forgekernel' / 'card_template', templates / 'src' / 'forgekernel' / 'card_template')
    shutil.copytree(root / 'meta', templates / 'meta')
    (templates / 'src' / 'forgekernel' / 'card_template' / '.gitignore').unlink(missing_ok=True)
    new_operator('demo', base_dir=tmp_path / 'run', templates_root=templates)
    assert (tmp_path / 'run' / 'operators' / 'demo' / '.gitignore').is_file()
