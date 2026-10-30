"""Reference: the promoted Muse prefill boundary kernel (mode 0, gpost enabled)."""
import torch
import submission as _mod

_ext = None

def run(branch, stream, gpost, alpha, beta, gpre):
    global _ext
    if _ext is None:
        _ext = _mod._build(True)
    sout = torch.empty_like(stream)
    nout = torch.empty_like(branch)
    zero = torch.empty((0,), device=branch.device, dtype=torch.int32)
    _ext.boundary(branch, stream, gpost, alpha, beta, gpre, sout, nout, True, zero)
    return sout, nout
