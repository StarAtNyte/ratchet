"""Can NCU profile on this machine? (`forge run` asks once per launch.)

    ncu --csv --metrics gpu__time_active.sum python3 eval/ncu_probe.py

One small kernel. A machine that does not let this user read the GPU's performance
counters (ERR_NVGPUCTRPERM) prints no metric row; the run then degrades to timing only.
"""

import torch

x = torch.ones(1 << 20, device="cuda")
torch.cuda.synchronize()
print("ncu_probe:", float((x * 2).sum().item()))
