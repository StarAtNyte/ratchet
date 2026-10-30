# Muse prefill residual and normalization boundary

Given contiguous CUDA tensors, return `(sout, nout)`. Inputs in order:

- `branch`: BF16 `[m, h]`.
- `stream`: FP32 `[m, h]`.
- `gpost`, `gpre`: BF16 `[h]`.
- `alpha`, `beta`: FP32 `[h]`.

For each row, in FP32, compute `b = float(branch) * (1 + float(gpost))`; `r = rsqrt(mean(b*b) + 1e-8)`; `sout = alpha*stream + beta*b*r`; `nout = bf16(sout * rsqrt(mean(sout*sout) + 1e-5) * (1 + float(gpre)))`. Return the FP32 stream and BF16 normalized result, both fresh. Inputs are finite, arbitrary values, and all shapes are contiguous. The operation is the second boundary of our Muse prefill layer for turn sizes below the recompute threshold. Input values change between calls; do not cache outputs or inputs. Preserve BF16 input and FP32 accumulation. The challenge validates relative L2 and individual rows; use stable accumulation and cover near-zero, outlier and ordinary inputs. CUDA graph replay and cold L2 matter for final challenge validation. Official checks are required before integrating a proposed optimization.
