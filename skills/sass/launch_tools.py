"""Launch configuration off NCU output, with the cluster shape made explicit.

Reconnaissance §1 (shape + launch configuration) used to be read by eye off the
``Grid Size`` / ``Block Size`` columns, and the cluster shape inferred from the
grid's *shape* ("dim0 of (2,1,66) is the 2-CTA cluster"). That inference is
wrong whenever the launch uses a cluster attribute instead of a grid dimension:
the dsa_sparse B200 reference launches a flat grid of 128 CTAs with
``launch__cluster_dim_x = 16`` (8 tokens x 16 CTAs, DSMEM merge across the
cluster), and the report's §1 said "no cluster" while the same NCU log carried
the metric. The candidates table then reverse-engineered a dense 256x128x64
tile with cluster=(1,1,1), the teacher's first milestone copied it, and the
student's first kernel was 16x slower than the reference.

So the cluster shape is read from the launch metrics NCU already reports —
``launch__cluster_dim_x/y/z``, ``launch__cluster_max_active`` — and never
derived from the grid. ``report`` prints one machine line the driver reads:

    LAUNCH-OK kernel=<name|n/a> grid=(x,y,z) block=(x,y,z) ctas=N cluster=(x,y,z)
              cluster_max_active=<N|n/a> regs=<N|n/a> smem_dyn=<N|n/a>

and refuses (``LAUNCH-PARTIAL reason=no-cluster-metrics``) when the page carries
launch statistics but no cluster metric at all — absence of the metric on an
older collection is not evidence of "no cluster", and the line must not say
(1,1,1) on that basis. The driver cross-checks the report's prose against the
reading (``RECON-CONTRADICTION cluster``) at write-up time.

Accepted inputs (any mix, one text): the NCU ``--page raw`` dump
(``    launch__cluster_dim_x    16``), ``--csv --metrics`` rows
(``"...","launch__cluster_dim_x","cluster","16"``), ``key = value`` lines a
probe printed itself, and the ``--page details`` Launch Statistics rows
(``Grid Size``, ``Block Size``, ``Cluster Size``, ``Registers Per Thread``,
``Dynamic Shared Memory Per Block``).

Like ``sass_tools`` this file is seeded verbatim into the recon tree's
``notes/`` by the driver, so probes reach it with a bare import.
"""

from __future__ import annotations

import re
from typing import Dict, NamedTuple, Optional, Tuple

# ``launch__<metric>`` followed (after any non-digit noise: the raw page's
# column padding, '=', ':', a csv unit column) by its numeric value. The noise
# bound keeps a metric name from picking up a number from the next line; the
# same line never carries two metrics on the raw page, and a csv row's own
# value is the first digit after the name. Word-valued metrics (a scheduling
# policy) are read separately so a csv unit column ("cluster") never masquerades
# as the value of a numeric one.
_METRIC = re.compile(r"\b(launch__[a-z_]+)\b[^\d\n-]{0,120}?(-?\d+(?:\.\d+)?)(?=[\s\",]|$)", re.M)
_METRIC_WORD = re.compile(r"\b(launch__[a-z_]+)\b[\s=:\"',]{1,40}?([A-Za-z][A-Za-z_]+)")
# --page details "Launch Statistics" rows: label, then the value
_DETAIL = re.compile(r"^\s*(Grid Size|Block Size|Cluster Size|Registers Per Thread|"
                     r"Dynamic Shared Memory Per Block|Static Shared Memory Per Block)"
                     r"\s{2,}(?:[A-Za-z/]+\s+)?([\d.,]+)\s*$", re.M)
_KERNEL = re.compile(r"^\s*(?:Kernel Name|kernel)\s*[:=]\s*(\S.*?)\s*$", re.M)

_DIMS = ("x", "y", "z")
_INT_METRICS = {
    "launch__grid_dim_x", "launch__grid_dim_y", "launch__grid_dim_z",
    "launch__grid_size", "launch__block_dim_x", "launch__block_dim_y",
    "launch__block_dim_z", "launch__block_size", "launch__cluster_dim_x",
    "launch__cluster_dim_y", "launch__cluster_dim_z", "launch__cluster_size",
    "launch__cluster_max_active", "launch__cluster_max_potential_size",
    "launch__registers_per_thread", "launch__shared_mem_per_block_dynamic",
    "launch__shared_mem_per_block_static", "launch__shared_mem_per_block",
    "launch__waves_per_multiprocessor", "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem",
}


class LaunchFacts(NamedTuple):
    grid: Optional[Tuple[int, int, int]]
    block: Optional[Tuple[int, int, int]]
    cluster: Optional[Tuple[int, int, int]]     # None = no cluster metric on the page
    cluster_max_active: Optional[int]
    regs: Optional[int]
    smem_dynamic: Optional[int]
    kernel: Optional[str]
    raw: Dict[str, object]                      # every launch__ metric seen, verbatim

    @property
    def ctas(self) -> Optional[int]:
        if self.grid is None:
            return None
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def clustered(self) -> Optional[bool]:
        """True/False when the metric was read, None when it was not."""
        if self.cluster is None:
            return None
        return max(self.cluster) > 1


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def kernel_section(text: str, kernel: str = "") -> str:
    """The part of ``text`` that belongs to one kernel: the span after the header
    naming ``kernel`` and before the next header, or the LAST header's span when no
    name is given.

    A log with several kernels in it (ncu without ``-k`` profiles the warmup / RNG /
    torch elementwise kernels first, and a probe may print two pages) used to be read
    whole, so every metric came from whichever kernel appeared FIRST — a review
    measured ``report(text, kernel="dsa_kernel")`` printing
    ``grid=(64,1,1) cluster=(0,0,0)`` off a leading ``helper_kernel`` section, i.e.
    the tool certifying exactly the "no cluster" reading the gate exists to catch.
    Text with no header at all is returned unchanged (a single-page dump)."""
    heads = list(_KERNEL.finditer(text or ""))
    if not heads:
        return text
    def span(i: int) -> str:
        start = heads[i].start()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        return text[start:end]
    want = (kernel or "").strip()
    if want:
        for i, h in enumerate(heads):
            if h.group(1).strip() == want:
                return span(i)
        # an unread name must not silently fall back onto another kernel's numbers
        return ""
    return span(len(heads) - 1)


def parse_launch_metrics(text: str) -> Dict[str, object]:
    """Every ``launch__*`` metric in ``text`` (first occurrence wins), plus the
    ``--page details`` rows mapped onto the same metric names. Callers holding a
    multi-kernel log pass one kernel's section (see ``kernel_section``)."""
    raw: Dict[str, object] = {}
    for m in _METRIC.finditer(text):
        key, val = m.group(1), m.group(2)
        if key in raw:
            continue
        v = _num(val)
        raw[key] = int(v) if isinstance(v, float) and v.is_integer() else v
    for m in _METRIC_WORD.finditer(text):
        key, val = m.group(1), m.group(2)
        if key in raw or key in _INT_METRICS:
            continue
        raw[key] = val
    details = {"Grid Size": "launch__grid_size", "Block Size": "launch__block_size",
               "Cluster Size": "launch__cluster_size",
               "Registers Per Thread": "launch__registers_per_thread",
               "Dynamic Shared Memory Per Block": "launch__shared_mem_per_block_dynamic",
               "Static Shared Memory Per Block": "launch__shared_mem_per_block_static"}
    for m in _DETAIL.finditer(text):
        key = details[m.group(1)]
        if key not in raw:
            raw[key] = _num(m.group(2).replace(",", ""))
    return raw


def _dims(raw: Dict[str, object], stem: str, size_key: str) -> Optional[Tuple[int, int, int]]:
    xyz = [raw.get(f"{stem}_{d}") for d in _DIMS]
    if all(isinstance(v, int) for v in xyz):
        return (xyz[0], xyz[1], xyz[2])
    if isinstance(xyz[0], int):
        return (xyz[0], 1 if xyz[1] is None else int(xyz[1]),
                1 if xyz[2] is None else int(xyz[2]))
    size = raw.get(size_key)
    if isinstance(size, int):
        return (size, 1, 1)
    return None


def parse_launch(text: str, kernel: str = "") -> LaunchFacts:
    section = kernel_section(text, kernel)
    raw = parse_launch_metrics(section)
    k = kernel or None
    if k is None:
        m = _KERNEL.search(text)
        k = m.group(1) if m else None
        if m is not None and len(_KERNEL.findall(text)) > 1:
            k = _KERNEL.findall(text)[-1]      # the section the metrics came from
    regs = raw.get("launch__registers_per_thread")
    smem = raw.get("launch__shared_mem_per_block_dynamic")
    cma = raw.get("launch__cluster_max_active")
    return LaunchFacts(
        grid=_dims(raw, "launch__grid_dim", "launch__grid_size"),
        block=_dims(raw, "launch__block_dim", "launch__block_size"),
        cluster=_dims(raw, "launch__cluster_dim", "launch__cluster_size"),
        cluster_max_active=cma if isinstance(cma, int) else None,
        regs=regs if isinstance(regs, int) else None,
        smem_dynamic=smem if isinstance(smem, int) else None,
        kernel=k, raw=raw)


def _t(v) -> str:
    return "n/a" if v is None else (f"({v[0]},{v[1]},{v[2]})" if isinstance(v, tuple) else str(v))


def launch_line(f: LaunchFacts) -> str:
    """The verdict line for ``f``: LAUNCH-OK only with grid, block AND a cluster
    reading; LAUNCH-PARTIAL names what is missing."""
    fields = (f"kernel={f.kernel or 'n/a'} grid={_t(f.grid)} block={_t(f.block)} "
              f"ctas={_t(f.ctas)} cluster={_t(f.cluster)} "
              f"cluster_max_active={_t(f.cluster_max_active)} regs={_t(f.regs)} "
              f"smem_dyn={_t(f.smem_dynamic)}")
    if f.grid is None or f.block is None:
        return f"LAUNCH-PARTIAL reason=no-launch-metrics {fields}"
    if f.cluster is None:
        return f"LAUNCH-PARTIAL reason=no-cluster-metrics {fields}"
    return f"LAUNCH-OK {fields}"


def report(text: str, kernel: str = "", out=print) -> bool:
    """Print the §1 launch table and its verdict line; True on LAUNCH-OK.

    The cluster is the load-bearing field: a launch that reports a cluster of
    (1,1,1) is a measured fact, a page with no cluster metric is not — collect
    ``--metrics launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,``
    ``launch__cluster_max_active`` (or ``--page raw``) and rerun."""
    f = parse_launch(text, kernel)
    out("[launch] metrics seen: " + (", ".join(f"{k}={v}" for k, v in sorted(f.raw.items()))
                                    or "none"))
    if f.cluster is not None:
        out(f"[launch] cluster={_t(f.cluster)} "
            + ("(clustered launch: the cluster shape is a launch attribute, NOT a grid "
               "dimension — read tile / CTA-count candidates against it)"
               if f.clustered else "(no cluster: every CTA is its own cluster)"))
    else:
        out("[launch] no launch__cluster_dim_* metric on this page — absence is not "
            "evidence of 'no cluster'; collect it with --metrics "
            "launch__cluster_dim_x,launch__cluster_dim_y,launch__cluster_dim_z,"
            "launch__cluster_max_active or --page raw")
    line = launch_line(f)
    out(line)
    return line.startswith("LAUNCH-OK")
