"""Modal configuration is carried from local.env into the GPU job."""
from types import SimpleNamespace
from forgekernel.localenv import MachineError, load_machine
from forgekernel.submit.config import load_config
from forgekernel.taskcard import write_submit
import pytest


def test_modal_settings_reach_submit(tmp_path):
    (tmp_path / 'local.env').write_text('FK_DELIVERY=modal\nFK_MODAL_GPU=H200\nFK_MODAL_TIMEOUT=123\n')
    machine = load_machine(tmp_path, environ={})
    assert machine.describe().startswith('Modal H200')
    op = tmp_path / 'operators' / 'op'
    op.mkdir(parents=True)
    write_submit(op, SimpleNamespace(arch='sm90'), machine, timeout=1800)
    cfg = load_config('op', base_dir=tmp_path)
    assert cfg.delivery == 'modal'
    assert cfg.timeout == 123
    assert cfg.modal_gpu == 'H200'
    assert cfg.cluster.startswith('modal:H200:')


def test_modal_timeout_rejects_nonpositive(tmp_path):
    (tmp_path / 'local.env').write_text('FK_DELIVERY=modal\nFK_MODAL_TIMEOUT=0\n')
    with pytest.raises(MachineError, match='FK_MODAL_TIMEOUT'):
        load_machine(tmp_path, environ={})
