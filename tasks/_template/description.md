# The operator

The operator, completely: the teacher writes the correctness judge from this file alone.

## Signature

- inputs (in order): name, shape over the axes in shapes.json, dtype
- outputs: name, shape, dtype

## Semantics

The formula or a precise description, including every convention (masking, padding, what
an out-of-range index means, accumulation precision if it is part of the contract).

## Legal inputs

What a valid input looks like (index ranges, sorted tensors, sparsity), so generated inputs
stay legal.
