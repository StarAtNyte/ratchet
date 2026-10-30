"""Correctness monitor loader. Python imports ``sitecustomize`` at start-up from the
first directory on sys.path that has one; the monitor's runner puts this directory
first on PYTHONPATH for the card's bench. It installs the hook (a no-op unless
FK_MON_PROFILE is set) and then hands over to whatever ``sitecustomize`` the image
itself ships, so the job environment is otherwise unchanged."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.insert(0, _HERE)
    import fk_monitor_hook

    fk_monitor_hook.install()
except Exception as _e:  # noqa: BLE001 — the monitor never costs the bench
    sys.stderr.write(f"[fk-monitor] hook not installed: {_e}\n")
finally:
    try:
        sys.path.remove(_HERE)
    except ValueError:
        pass


def _chain() -> None:
    import importlib

    saved = list(sys.path)
    ours = sys.modules.pop("sitecustomize", None)
    try:
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.curdir) != _HERE]
        try:
            importlib.import_module("sitecustomize")
        except ImportError:
            if ours is not None:
                sys.modules["sitecustomize"] = ours
    finally:
        sys.path[:] = saved


_chain()
