# Hardware knowledge base (hw_kb)

An operator-agnostic library of GPU-certified hardware-interface recipes, one tree per kernel
language and, inside it, one tree per GPU architecture:

```
hw_kb/
  cutedsl/           # CuTe DSL runs (task language "cutedsl")
    README.md        # rules, layout and fill strategy
    TEMPLATE.md
    sm90/  sm100/
  cuda/              # CUDA C++ runs (task language "cuda")
    README.md
    TEMPLATE.md
    sm100/
```

A run reads and writes exactly one tree, `hw_kb/<language>/<arch>/`, picked by the task's `language`
and `arch` (`src/forgekernel/loop/lang_mode.py`). The two language trees are independent: a reading
certified with one language's probe is not carried over to the other. See each tree's README for
the rules.
