"""GPU jobs for an operator tree: on this machine or one ssh host.

Each operator tree carries its submission config in ``operators/<op>/submit.toml``
(forgekernel.submit.config); a job is a detached runner with its state in plain files
(forgekernel.submit.jobs) that runs one gpu_entry task (forgekernel.submit.direct).
"""

from forgekernel.submit.config import SubmitConfig, load_config

__all__ = ["SubmitConfig", "load_config"]
