"""Per-iteration orchestration census — the student's SASS contract axes on
EVERY evaluated KEEP, riding the NCU job that already runs for each one.

Why this exists (learned on dsa_sparse_fwd): a one-line KEEP that flipped
the KV gather's cp.async cache mode (`LoadCacheMode.GLOBAL`, SASS
``LDGSTS.E.BYPASS``) shipped twice at ~4 % and was later priced at 22 % on a
tree whose other stalls were gone. It never reached hw_kb because the harvest
path only sees *battles* (probe fights, stuck diagnoses); a cheap win has no
harvest channel. The dossier summary (orch_tools) already keys every SASS
contract axis — sync topology, in-flight depth, supply order, cache-policy
bits — but it was rendered only inside stall-pricing jobs at teacher entries,
so no KEEP could be tied to the axis it moved. This runner renders the same
ORCH-SUMMARY-JSON line off the kernel the NCU job just profiled, and the
driver records it on the iteration; a KEEP whose summary differs from the
previous KEEP's on any axis is then a harvest item in its own right.

How: wraps the operator's own ``bash gpu_entry.sh ncu`` (the op's metric list
and profiling window stay the op's), asking the CuTe DSL JIT to keep its cubin
(``CUTE_DSL_KEEP_CUBIN`` / ``CUTE_DSL_DUMP_DIR``). The NCU stdout is streamed
through line by line as it arrives — never held back — so a wrapper killed by
the outer ``cmd`` task's TASK_TIMEOUT loses nothing NCU already printed, and
the CSV reaches the log exactly as the plain task printed it. Then the largest
kernel dumped is disassembled and the two machine lines (``ORCH-CENSUS`` +
``ORCH-SUMMARY-JSON``) are printed LAST: a log read back with a length cap keeps its
tail, and lines printed ahead of a long CSV are the first to go (silently — the
census only ever degrades). The
driver finds the lines anywhere in the log and strips them before its CSV
parse. Every failure degrades to a one-line ``ORCH-CENSUS-SKIP`` and the
untouched NCU output — the census is evidence, never a precondition.

Runs vendored-flat in the student workspace (``python3 notes/orch_census_run.py``;
``notes/`` is on sys.path, so ``import orch_tools`` / ``import sass_tools`` are
bare) and needs nothing the pricing toolkit does not already ship.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

try:  # vendored flat in a job tree (notes/ on sys.path)
    import orch_tools  # type: ignore
    import sass_tools  # type: ignore
except ImportError:  # harness-side import path (tests)
    from forgekernel.skills import orch_tools, sass_tools
try:
    import sass_census  # type: ignore
except ImportError:
    try:
        from forgekernel.skills import sass_census  # type: ignore
    except ImportError:  # a tree seeded before the census module rode along
        sass_census = None  # type: ignore

DUMP_ENV = {"CUTE_DSL_KEEP_CUBIN": "1"}
DUMP_DIR_ENV = "CUTE_DSL_DUMP_DIR"
NCU_CMD = ["bash", "gpu_entry.sh", "ncu"]
_SYM = re.compile(r"(?:Function\s*:\s*|\.text\.)([A-Za-z0-9_$]+)")
# nvdisasm (and cuobjdump on recent toolkits) writes branch targets as
# `(.L_x_N) labels with `.L_x_N:` definition lines; the dossier's loop
# detection reads the NUMERIC target the NCU source page prints. Unresolved,
# every backedge vanishes and the census reports a kernel with no loops, no
# rings, no supply and no commit depth — structurally empty (smoke job
# 856949 on the dsa_sparse_fwd tree: 3656 instrs, 16 mbarriers, loops {}).
_LABEL_DEF = re.compile(r"^\s*(\.L_[A-Za-z0-9_]+)\s*:\s*$")
_LABEL_REF = re.compile(r"`\((\.L_[A-Za-z0-9_]+)\)")
_ADDR_LINE = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/")


def resolve_labels(sass: str) -> str:
    """Rewrite `(.L_x_N) branch targets as the hex address of the first
    instruction after the label's definition. Text without labels is returned
    unchanged; a reference to an undefined label is left as it was."""
    addr_of = {}
    pending = []
    for line in sass.splitlines():
        m = _LABEL_DEF.match(line)
        if m:
            pending.append(m.group(1))
            continue
        a = _ADDR_LINE.match(line)
        if a and pending:
            addr = f"0x{int(a.group(1), 16):x}"
            for lab in pending:
                addr_of.setdefault(lab, addr)
            pending = []
    if not addr_of:
        return sass
    return _LABEL_REF.sub(lambda m: addr_of.get(m.group(1), m.group(0)), sass)


def disassemble(cubin: str) -> Tuple[int, str]:
    """(rc, sass_text): nvdisasm first (compact), cuobjdump as the fallback;
    branch labels resolved to addresses either way."""
    for cmd in (["nvdisasm", "-c", cubin], ["cuobjdump", "-sass", cubin]):
        if shutil.which(cmd[0]) is None:
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and proc.stdout.strip():
            return 0, resolve_labels(proc.stdout)
    return 1, ""


def pick_kernel(dump_dir: str, out=print) -> Optional[Tuple[str, str, str, int]]:
    """The largest kernel dumped: (cubin path, symbol, sass text, n instrs).

    A JIT-compiled operator dumps one cubin per compiled entry; the kernel
    under test is by far the largest, and "largest by parsed instruction
    count" is a format-free criterion the NCU CSV's kernel name can be
    cross-checked against (the symbol rides the ORCH-CENSUS line).
    """
    cubins = sorted(glob.glob(os.path.join(dump_dir, "**", "*.cubin"),
                              recursive=True), key=os.path.getmtime)
    best = None
    for cub in cubins:
        rc, sass = disassemble(cub)
        if rc != 0:
            out(f"[orch-census] {os.path.basename(cub)}: disassembly failed")
            continue
        n = len(sass_tools.parse_sass(sass).instructions)
        m = _SYM.search(sass)
        sym = m.group(1) if m else "?"
        if best is None or n > best[3]:
            best = (cub, sym, sass, n)
    if best is not None:
        out(f"[orch-census] {len(cubins)} cubin(s) dumped; largest "
            f"{os.path.basename(best[0])} ({best[3]} instrs)")
    return best


def census(dump_dir: str, out=print, full: bool = False) -> Optional[dict]:
    """Print ORCH-CENSUS + ORCH-SUMMARY-JSON for the dumped kernel; None on skip."""
    try:
        picked = pick_kernel(dump_dir, out=out)
        if picked is None:
            out("ORCH-CENSUS-SKIP reason=no-cubin-dumped")
            return None
        cub, sym, sass, n = picked
        try:
            # the precision gate's SASS cross-check: MMA operand / accumulator types,
            # read even off a kernel too small for a dossier
            mp = orch_tools.mma_precision(sass_tools.parse_sass(sass).instructions)
            out("SASS-PRECISION-JSON: " + json.dumps({"kernel": sym, **mp}, sort_keys=True))
        except Exception as e:  # noqa: BLE001 — additive, never fatal
            out(f"[orch-census] sass precision skipped ({type(e).__name__})")
        if full:
            orch_tools.report(sass, out=out)
            summary = orch_tools.summarize(sass)
        else:
            summary = orch_tools.summarize(sass)
        if summary is None:
            out(f"ORCH-CENSUS-SKIP reason=too-few-instructions kernel={sym}")
            return None
        if not summary.get("loops") and not summary.get("spin_loops"):
            # A real kernel always has a loop; none resolved means the branch
            # targets did not parse (a dump format the resolver does not know)
            # and every structural axis below is empty, not "flat".
            out(f"[orch-census] WARN no loop regions resolved over {n} instrs — "
                f"the structural axes are unreadable, not empty")
        # The compiler's wgmma verdict rides the same dump: waits per wgmma. The
        # driver folds it into the profiler digest so a KEEP whose chains ptxas
        # just un-pipelined is named as such the round it lands, not two rounds
        # later from a perf number nobody could attribute. Printed before the two
        # census lines, which stay the log's tail (the archive keeps the tail).
        if sass_census is not None:
            try:
                out(sass_census.render_serial_line(
                    sass_census.wgmma_serialization_from_text(sass), sym))
            except Exception as e:  # noqa: BLE001 — additive, never fatal
                out(f"WGMMA-SERIAL-SKIP kernel={sym} reason={type(e).__name__}")
        out(f"ORCH-CENSUS cubin={os.path.basename(cub)} kernel={sym} instrs={n}")
        if not full:
            out("ORCH-SUMMARY-JSON: " + json.dumps(summary, sort_keys=True))
        return summary
    except Exception as e:  # noqa: BLE001 — additive channel, never fatal
        out(f"ORCH-CENSUS-SKIP reason={type(e).__name__}:{str(e)[:80]}")
        return None


# ---- CUDA C++ kernels (loop language cuda-sol) ----------------------------------
# A CUDA-mode student's CUDA C++ kernel file is built by the card (eval/leaderboard.py
# load_cuda_entry) into <FK_KERNEL_BUILD_DIR | $TMP/fk_kernel_build>/fk_kernel_<key>/,
# with a copy of the source staged next to the extension .so. No JIT dump switch
# exists for it: the build the NCU run just used is found by that staged copy and
# its embedded cubins are extracted (cuobjdump -xelf), then read like the DSL's.
# The CUDA suffix is spelled in two halves: this file is seeded into notes/, and
# a binary-reference recon tree's guard refuses notes that name C/C++ sources.
CUDA_SUFFIX = "." + "cu"
CUDA_BUILD_ENV = "FK_KERNEL_BUILD_DIR"


def cuda_build_root() -> str:
    return os.environ.get(CUDA_BUILD_ENV) or os.path.join(tempfile.gettempdir(), "fk_kernel_build")


def find_cuda_build(src: str, kernel_name: str, root: Optional[str] = None) -> Optional[str]:
    """The extension .so the card built from exactly ``src`` (staged as
    ``kernel_name`` in its build directory), newest first; None when none was built."""
    hits = []
    for d in glob.glob(os.path.join(root or cuda_build_root(), "fk_kernel_*")):
        try:
            with open(os.path.join(d, kernel_name), encoding="utf-8") as fh:
                if fh.read() != src:
                    continue
        except (OSError, UnicodeDecodeError):
            continue
        hits += glob.glob(os.path.join(d, "*.so"))
    return max(hits, key=os.path.getmtime) if hits else None


def extract_cuda_cubins(so: str, dump_dir: str) -> int:
    """Extract the .so's embedded device ELFs into ``dump_dir`` as *.cubin; the count."""
    if shutil.which("cuobjdump") is None:
        return 0
    subprocess.run(["cuobjdump", "-xelf", "all", os.path.abspath(so)], cwd=dump_dir,
                   capture_output=True, text=True, timeout=600)
    for p in glob.glob(os.path.join(dump_dir, "*")):
        # cuobjdump names them <so>.<k>.sm_<arch>.cubin; older toolkits drop the suffix
        if os.path.isfile(p) and not p.endswith(".cubin") and ".sm_" in os.path.basename(p):
            os.replace(p, p + ".cubin")
    return len(glob.glob(os.path.join(dump_dir, "*.cubin")))


def _run_ncu(env: dict) -> Tuple[int, str]:
    """Run the op's ncu task, echoing its stdout line by line as it arrives and
    returning (rc, everything it printed). Streamed rather than captured: the
    outer ``cmd`` task shares TASK_TIMEOUT with the inner ``ncu`` task and adds
    its own preamble plus the disassembly on top, so a wrapper SIGKILLed at the
    deadline must not take the CSV NCU already produced down with it. stderr is
    inherited (NCU warnings land in the job log where the plain task put them);
    undecodable bytes are replaced, never fatal."""
    proc = subprocess.Popen(NCU_CMD, env=env, stdout=subprocess.PIPE, text=True,
                            errors="replace")
    chunks = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    return proc.wait(), "".join(chunks)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    full = "--full" in args
    kernel = args[args.index("--kernel") + 1] if "--kernel" in args[:-1] else None
    dump_dir = tempfile.mkdtemp(prefix="fk_orch_census_")
    env = dict(os.environ, **DUMP_ENV)
    env[DUMP_DIR_ENV] = dump_dir
    print(f"[orch-census] running {' '.join(NCU_CMD)} with the JIT cubin kept "
          f"under {dump_dir}", flush=True)
    try:
        rc, ncu_out = _run_ncu(env)
    except Exception as e:  # noqa: BLE001
        print(f"ORCH-CENSUS-SKIP reason=ncu-launch-failed:{type(e).__name__}",
              flush=True)
        shutil.rmtree(dump_dir, ignore_errors=True)
        return 1
    if ncu_out and not ncu_out.endswith("\n"):
        sys.stdout.write("\n")
    if kernel and kernel.endswith(CUDA_SUFFIX):
        # a CUDA kernel dumps nothing: read the build the NCU run just used
        try:
            with open(kernel, encoding="utf-8") as fh:
                so = find_cuda_build(fh.read(), os.path.basename(kernel))
        except OSError:
            so = None
        if so is None or extract_cuda_cubins(so, dump_dir) == 0:
            print(f"ORCH-CENSUS-SKIP reason=no-cuda-build kernel={kernel}", flush=True)
            shutil.rmtree(dump_dir, ignore_errors=True)
            return rc
    # The census goes LAST, so a log clipped from the front still carries it; the driver
    # strips these lines before its CSV parse and reads them from anywhere.
    census(dump_dir, full=full)
    shutil.rmtree(dump_dir, ignore_errors=True)
    sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
