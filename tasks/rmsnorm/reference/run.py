"""RMSNorm reference: PyTorch, float32 statistics."""

import torch


def run(x, w):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w.float()
    return y.to(x.dtype)
