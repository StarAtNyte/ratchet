# sass

Static reads of a kernel's SASS, all from the page one `ncu --set source` collection (or a
`cuobjdump -sass` dump) already gives. Every tool checks itself and ends in one machine
verdict line; the driver opens a gate only on the OK form, never on prose.

| script | reads | verdict |
|---|---|---|
| `sass_tools.py` | parsing (both dump formats), opcode histogram and ISA families of the right kernel | `SASS-OK` / `SASS-FAIL` |
| `sass_census.py` | MMA A-operand source (register vs shared-memory descriptor) and WGMMA serialization | `SASS-OPERANDS ...` |
| `sched_live_tools.py` | the loop body as an ordered event stream (MMA chains, wait depths) and peak live registers | `SCHED-OK` / `SCHED-FAIL`, `LIVE-OK` / `LIVE-FAIL` |
| `struct_tools.py` | how each phase is built: warps, tile granule, internal precision, hand-off | `STRUCT-OK` / `STRUCT-FAIL` |
| `launch_tools.py` | grid, block, cluster shape from the launch metrics, never from the grid | `LAUNCH-OK` / `LAUNCH-PARTIAL` |

In a probe (the driver has seeded the files into `notes/`):

```python
import sass_tools
# the SASS page, the kernel NCU profiled, and every (kernel, duration) launch it saw
sass_tools.analyze(page_text, profiled_kernel, launch_pairs)   # prints SASS-OK / SASS-FAIL
```

Never hand-roll a SASS parser: the format differs between NCU and cuobjdump, and the first
launched kernel is often not the one you mean (`sass_tools.pick_kernel`).
