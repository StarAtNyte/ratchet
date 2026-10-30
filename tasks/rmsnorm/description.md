# rmsnorm

RMSNorm over the last axis, as used before every attention and MLP block of a transformer.

## Signature

- inputs, in order:
  - `x`: `[m, n]`, bfloat16 -- the activations, one row per token
  - `w`: `[n]`, bfloat16 -- the learned scale
- output:
  - `y`: `[m, n]`, bfloat16

## Semantics

For every row `i`: `y[i, :] = x[i, :] / sqrt(mean_j(x[i, j]^2) + 1e-6) * w`.
The mean and the square root are taken in float32 (at least); the result is rounded to
bfloat16 once, at the end.

## Legal inputs

Any finite bfloat16 values. Real activations have occasional large outlier channels (a
few columns tens of times larger than the rest); rows near zero must not produce NaN (the
epsilon keeps the denominator positive). `w` is positive in practice but the operator
does not assume it.
