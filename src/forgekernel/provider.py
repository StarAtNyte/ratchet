"""Bind one agent turn to a provider: the only reader of global_config/backends.toml.

``native`` inherits the environment untouched: the agent CLI's own login, or the credentials
local.env puts in the environment. Any other name is a table of backends.toml that points
the CLI at another endpoint, with the key read from the variable the table names.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from forgekernel.paths import global_config_dir

BACKENDS_FILE = "backends.toml"
NATIVE = "native"
# The wire protocol each agent CLI speaks.
_BACKEND_WIRE = {"claude": "anthropic"}
# What the turn's launcher reads (scripts/agent-claude.sh): the endpoint, and the name of
# the variable holding its key -- the key itself never appears on a command line.
BASE_URL_ENV = "ANTHROPIC_BASE_URL"
KEY_ENV_ENV = "FK_PROVIDER_KEY_ENV"


class ProviderError(ValueError):
    pass


def _table(root: Path | None = None) -> dict:
    p = (root or global_config_dir()) / BACKENDS_FILE
    if not p.is_file():
        return {}
    return tomllib.loads(p.read_text(encoding="utf-8"))


def names(root: Path | None = None) -> tuple[str, ...]:
    return (NATIVE, *_table(root))


def grant(backend: str, name: str, env: dict | None = None,
          root: Path | None = None) -> dict[str, str]:
    """The environment additions that point ``backend`` at provider ``name``; ProviderError
    when the provider does not fit the backend or its key is not set."""
    if name == NATIVE:
        return {}
    table = _table(root)
    if name not in table:
        raise ProviderError(f"provider {name!r} is not in global_config/{BACKENDS_FILE}")
    p = table[name]
    wire = _BACKEND_WIRE.get(backend)
    if wire is None:
        raise ProviderError(f"the {backend} CLI has no provider channel; use provider = "
                            f"{NATIVE!r}")
    if p.get("wire") != wire:
        raise ProviderError(f"provider {name!r} speaks {p.get('wire')}, the {backend} CLI "
                            f"speaks {wire}")
    for key in ("base_url", "key_env"):
        if not isinstance(p.get(key), str) or not p[key].strip():
            raise ProviderError(f"provider {name!r} has no {key}")
    env = os.environ if env is None else env
    if not env.get(p["key_env"]):
        raise ProviderError(f"provider {name!r} needs {p['key_env']}; set it in local.env or "
                            f"the environment")
    return {BASE_URL_ENV: p["base_url"], KEY_ENV_ENV: p["key_env"]}
