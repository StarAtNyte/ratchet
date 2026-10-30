"""Submission config model and loader.

A ``SubmitConfig`` is read from ``operators/<op>/submit.toml`` (plus an optional
gitignored ``submit.local.toml`` merged over it). ``forge run`` writes both from the
machine's ``local.env``:

* ``[repo] delivery`` -- ``local`` (this machine) or ``ssh`` (one ssh host);
  ``branch`` -- the tree's branch a job runs; ``entry_script`` -- gpu_entry.sh;
* ``[exec]`` -- ``gpu`` (the CUDA device index), and for ssh ``host`` (an ssh alias that
  logs in without a prompt), ``workdir`` (an absolute directory on the host) and an
  optional ``env_file`` the job sources first;
* ``[job] timeout`` -- seconds one gpu_entry task may take;
* ``[env]`` -- environment for the job (secrets in ``submit.local.toml``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.paths import operators_dir

DELIVERIES = ("local", "ssh", "modal")

# [env] names that are secrets: a sealed reference's key (FK_REF_KEY, or the name a bench
# gives it). The ssh transport keeps them off every command line it builds (read off
# stdin), and the tree's gpu_entry.sh moves them out of the job environment
# (forgekernel.sealref.key_preamble).
SECRET_ENV_SUFFIX = "_REF_KEY"


def is_secret_env(name: str) -> bool:
    return str(name).endswith(SECRET_ENV_SUFFIX)


@dataclass
class SubmitConfig:
    branch: str
    entry_script: str = "gpu_entry.sh"
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 3600
    delivery: str = "local"
    op_dir: str = ""     # the operator tree (the ``git archive`` source)
    base_dir: str = ""   # the directory holding operators/
    exec_host: str = ""
    exec_workdir: str = ""
    exec_env_file: str = ""
    exec_gpu: str = "0"
    modal_gpu: str = "H200"
    modal_image: str = "lmsysorg/sglang:dev-muse-spark-open"

    @property
    def cluster(self) -> str:
        """Where this tree's jobs run ("local", or the ssh host): readings taken on two
        machines are not compared as one."""
        return self.exec_host if self.delivery == "ssh" else (f"modal:{self.modal_gpu}:{self.modal_image}" if self.delivery == "modal" else "local")


def _repo_root() -> Path:
    # src/forgekernel/submit/config.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _ops_dir(base_dir: str | Path | None) -> Path:
    """``<base_dir>/operators`` when one is named, else the current run's operator trees
    (forgekernel.paths.operators_dir)."""
    if base_dir is not None:
        return Path(base_dir) / "operators"
    return operators_dir(_repo_root())


def load_config(op: str, base_dir: str | Path | None = None) -> SubmitConfig:
    base = Path(base_dir) if base_dir is not None else _repo_root()
    op_dir = _ops_dir(base_dir) / op
    main = op_dir / "submit.toml"
    if not main.exists():
        raise FileNotFoundError(f"missing submit config: {main}")
    with open(main, "rb") as fh:
        data = tomllib.load(fh)
    local = op_dir / "submit.local.toml"
    if local.exists():
        with open(local, "rb") as fh:
            _merge(data, tomllib.load(fh))

    job = data.get("job", {})
    repo = data.get("repo", {})
    ex = data.get("exec", {})
    delivery = repo.get("delivery", "local")
    if delivery not in DELIVERIES:
        raise ValueError(f"invalid [repo] delivery {delivery!r}; expected one of {DELIVERIES}")
    host = str(ex.get("host", ""))
    workdir = str(ex.get("workdir", ""))
    if delivery == "ssh" and (not host or not workdir.startswith("/")):
        raise ValueError(f"{main}: ssh delivery needs [exec] host and an absolute "
                         f"[exec] workdir")
    return SubmitConfig(
        branch=repo["branch"],
        entry_script=repo.get("entry_script", "gpu_entry.sh"),
        env={str(k): str(v) for k, v in data.get("env", {}).items()},
        timeout=int(job.get("timeout", 3600)),
        delivery=delivery,
        op_dir=str(op_dir),
        base_dir=str(base),
        exec_host=host,
        exec_workdir=workdir,
        exec_env_file=str(ex.get("env_file", "")),
        exec_gpu=str(ex.get("gpu", "0")),
        modal_gpu=str(ex.get("modal_gpu", "H200")),
        modal_image=str(ex.get("modal_image", "lmsysorg/sglang:dev-muse-spark-open")),
    )


def _merge(base: dict, override: dict) -> None:
    """Shallow per-table merge: override tables update base tables in place."""
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            base[key].update(val)
        else:
            base[key] = val
