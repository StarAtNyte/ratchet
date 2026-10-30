"""DeepSeek-V4.1 prefill: Q RoPE + sparse MQA attention with sink + conjugate O RoPE + FP8 cast (one kernel, sm_100a).

dsa.cu is JIT-compiled once on import (outside timing).
Q_PERM: per token, q is consumed in 16-byte pieces ordered [cta r][col half c][TMEM quadrant w][piece k][lane]:
piece k of lane l in quadrant w holds head 64r + (32w + l) % 64, dims 256*(w // 2) + 128c + 8k .. +8, so each warp
load instruction reads 512 contiguous bytes.
O_PERM: within a group of 8 heads the stored channel 256*b + 32*j + e is head j, dim 32*b + e (8 heads of one
32-dim block are 256 contiguous bytes); scale blocks are natural 32-dim blocks of one head.
"""
import hashlib
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/usr/local/cuda"):
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
_BUILD = os.path.join(os.path.expanduser("~"), ".cache", "dsa_sol_build")
os.makedirs(_BUILD, exist_ok=True)

_TRACE = os.environ.get("DSA_TRACE") == "1"
_SRC = [os.path.join(_HERE, f) for f in ("dsa.cu", "dsa_ptx.cuh")]
_TAG = hashlib.md5(b"".join(open(f, "rb").read() for f in _SRC)).hexdigest()[:10]
_mod = load(
    name="dsa_sol_" + _TAG + ("_tr" if _TRACE else "") + ("_spin" if os.environ.get("DSA_SPIN") == "1" else "") + ("_wd" if os.environ.get("DSA_WD") == "1" else ""),
    sources=[os.path.join(_HERE, "dsa.cu")],
    extra_include_paths=[_HERE],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo"]
    + (["-DDSA_TRACE"] if _TRACE else []) + (["-DDSA_SPIN"] if os.environ.get("DSA_SPIN") == "1" else []) + (["-DDSA_WD"] if os.environ.get("DSA_WD") == "1" else []),
    extra_ldflags=["-lcuda"],
    build_directory=_BUILD,
    verbose=False,
)

Q_PERM = [(64 * r + (32 * w + l) % 64) * 512 + 256 * (w // 2) + 128 * c + 8 * k + e
          for r in range(2) for c in range(2) for w in range(4) for k in range(16) for l in range(32) for e in range(8)]
O_PERM = [512 * j + 32 * b + e for b in range(16) for j in range(8) for e in range(32)]

dbg = 0
trace = None  # debug: int64 cuda tensor [8192] (DSA_TRACE=1 builds)
_NCL = {}


def run(q, kv, indices, token_positions, cos_sin_cache, attn_sink, topk_length, sm_scale, rms_norm_eps, enable_q_norm):
    if enable_q_norm:
        raise NotImplementedError("Q RMSNorm is off for DeepSeek-V4.1")
    dev = q.device
    s_q = q.shape[0]
    if dev not in _NCL:
        _NCL[dev] = torch.cuda.get_device_properties(dev).multi_processor_count // 2
    out = torch.empty((s_q, 16, 4096), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((16, 32, s_q), dtype=torch.int32, device=dev).permute(2, 0, 1)
    ml = torch.empty((s_q, 128), dtype=torch.float32, device=dev)
    lse = torch.empty((s_q, 128), dtype=torch.float32, device=dev)
    _mod.launch(q, kv, indices, token_positions, cos_sin_cache, attn_sink, topk_length, float(sm_scale),
                out, sf, ml, lse, _NCL[dev], dbg, trace)
    return out, sf, ml, lse
