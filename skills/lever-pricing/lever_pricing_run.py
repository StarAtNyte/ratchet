"""Driver-seeded LEVER PRICING run — price N candidate levers on the current
kernel, single-add / single-remove / all-on, in one GPU job.

Submitted by the loop driver at a teacher entry that has a priced tree (the
plateau diagnosis and the M0 audit): `cmd:python3 notes/lever_pricing_run.py
notes/levers.json`, cwd = repo root, before the teacher is asked for a
cause_class, so the diagnosis reads MEASURED prices instead of estimating them
off the stall ledger.  Not a student tool: students keep using PROBE_REQUEST.
This file exists so every commit after seeding carries the frozen recipe (git
archive ships committed files only).

Why this exists: an audit that names one locus off the ledger has no price on
the locus it names nor on the levers it passes over.  Pricing the candidates by
ablation on the best tree — each alone, all together, all minus each — is the
only reading that also shows a lever that is worthless alone but required by
the others (it reads ~0 when added, negative when removed from the stack), and
one that only becomes visible once the others are subtracted.

Lever file (JSON, written by the teacher into notes/levers.json; the driver
adds ``kernel_file`` and ``rounds``):

    {"kernel_file": "kernel.py", "rounds": 3,
     "levers": [
       {"name": "ring_per_stage",
        "edits": [{"old": "<exact text in the kernel file>", "new": "<replacement>"}, ...],
        "expect": "producer refills each stage as its consumer releases it",
        "in_stack": true}
       , ...],
     "variants": [
       {"name": "ring+lane", "levers": ["ring_per_stage", "lane_owned"],
        "question": "is lane_owned's coupled reading a dependency on the ring?"}
       , ...]}

``in_stack: false`` prices a lever alone only — it is left out of ``all`` /
``all-L`` and of every subset (the teacher suspects it harmful or exclusive
with the rest; its single-add reading stands, its class has no remove leg).
``variants`` are the teacher's own subsets: the mechanical 2N+2 design cannot
say WHICH lever a ``coupled`` one depends on, nor whether the stack would be
positive without one harmful member — a named subset answers exactly the
question the teacher writes beside it. Each is timed against base and printed
as a VARIANT line the diagnosis may cite (``priced_by=<variant name>``).
The base and every single-add are always measured, whatever the teacher
asks for: they are the floor under every citation.

``ablations`` are the other kind of experiment the same job can run: an edit
that REMOVES or degrades a mechanism the tree already has (drop a barrier,
serialise a pipeline, stub a stage) to read what that mechanism costs today.
An ablation is a conviction instrument, not a candidate: it is applied to
base alone, never stacked, never enters ``all`` or any subset, and its
correctness is allowed to break — the reading comes from the NCU kernel
clock (``gpu__time_duration`` of the dominant launch, base re-timed in the
same pass) so a kernel that computes garbage is still timed. Its delta is
the price the ablated mechanism is charging right now: an upper bound on
what fixing it can return, citable as ``ablation:<name>``. The ablation
pass is bounded: each variant runs ncu FIRST (NCU_TIME_TIMEOUT); a timeout
marks it ``hung`` — no bench, no further rounds — and its line reads
``n/a``; the bench (only for the device-ms side reading) has its own short
timeout; the whole pass has a budget (``ablation_budget_s``) after which the
remaining variants read ``n/a status=budget``. The lever table is printed,
both copies, before the pass starts — but a job that outlives the driver's
poll window loses its log whole, which is what the budget is for.

    {"ablations": [
       {"name": "no_kv_barrier",
        "edits": [{"old": "...", "new": "..."}],
        "hypothesis": "the per-stage KV barrier is the exposed wait",
        "decisions": ["D1.2"]}
       , ...]}

Each lever is a list of exact-text replacements on the kernel file (the tree
as committed).  Every `old` must occur exactly once in the file it is applied
to, and the levers must be composable: applying them all in order must also
find every `old` exactly once.  A lever whose edit does not apply is REFUSED
(reported, never silently skipped) — a no-op variant is the one way this
instrument lies.

Variants built:  base | +L (each lever alone) | all | all-L (each lever
removed from all)  = 2N+2 variants, plus the teacher's subsets, plus — when
some stack lever reads ``harmful`` and at least two stack levers remain — an
automatic ``all-pos`` (all minus the harmful levers), timed in a second pass
TOGETHER WITH A FRESH ``base`` (interleaved, `rounds` more times; its delta is
read against that pass's own base, never the first pass's — on a bench without
a same-job reference the metric is device ms, and the node drift between the
two passes would otherwise land straight in the all-pos reading): one harmful
member drags every all-L reading with it, so the stack's real value is what is
left once it is out.  All timed interleaved, `rounds` times,
in ONE process per variant call (eval/bench.py --mode all), so the same-job
ratio to the reference (`vs_ref_device` when the bench carries one) is the
primary reading and node bimodality cancels.  The timing table is printed as
soon as the rounds are done; then per variant one `stall_price_run.py --pm`
for the role exposure ledger (spin / producer / consumer exposed ns, tensor
plateau) — the overlap account, so a lever's price can be attributed to a
role, not just to milliseconds — and the table is printed again with the
role columns filled.  A job killed during the role pass still leaves the
timing table in the log; the driver keeps the LAST copy of each line.
``--no-roles`` skips the role pass (the driver passes it on a group card's
non-representative workloads: the role columns are read on the representative).

Machine-readable output (the driver parses these):

    LEVER <name> add=<+x.xx%> remove=<-y.yy%> add_ms=<..> remove_ms=<..> class=<independent|coupled|dependent|null|harmful|unstacked|noop|broken|unmeasured> spin_add=<ns> spin_all_minus=<ns>
    VARIANT <name> levers=<a,b,..> delta=<+x.xx%> ms=<..> spin=<ns> q=<the question it answers>
    STACK all=<+z.zz%> sum_singles=<+s.ss%> coupling=<z-s pp> all_pos=<+p.pp%|n/a> all_pos_levers=<a,b|none> dropped=<c|none>
    PRICE-SCOPE workloads=representative rep=<wl name> of=<n>   (group card: priced on its
                                                  representative workload only)
    PRICE noise_floor=<n.nn%> metric=<vs_ref_device|device_ms>
    ABLATE <name> delta=<+x.xx%> ncu_ns=<..> base_ncu_ns=<..> correctness=<pass|fail|nojson|skipped> device_delta=<+x.xx%|n/a> kernel=<dominant launch, one token> status=<ok|hung|nodata|budget> h=<hypothesis>
    ABLATE-OK ablations=<n> rounds=<r> base_ncu_ns=<..>   /  ABLATE-FAIL reason=<...>
    PRICE ablation_noise_floor=<x.xx>% metric=ncu_ns
    PRICE-TIMED levers=<n> rounds=<r>            (timing table complete, roles pending)
    PRICE-OK levers=<n> rounds=<r> base_ms=<..> all_ms=<..> ...
    PRICE-FAIL reason=<...>

`add` is the ratio delta of base+L vs base; `remove` is the delta of all-L vs
all (negative = removing it costs).  Classes:
  independent  add > noise and |remove| ≈ add        (price it alone)
  coupled      add ≈ 0 but remove < -noise           (worthless alone, needed by
                                                      others: package it with them)
  dependent    add > noise but remove ≈ 0            (its gain is subsumed once
                                                      the others are in)
  null         add ≈ 0 and remove ≈ 0                (drop it)
  harmful      add < -noise and remove ≈ 0 (or        (the direction is wrong:
               unmeasured)                            base does not have it,
                                                      so its reverse is not a
                                                      lever — drop it)
               A negative single-add with remove < -noise is NOT harmful: alone
               it costs, but the stack loses more without it — that is
               `coupled` (package it with what depends on it).
  unstacked    add > noise but the `all` variant     (single reading only; its
               failed correctness                     coupling is unmeasured)
  noop         every +L cubin is byte-identical to base's (the edit changed
               nothing the compiler kept) — only when BOTH sides dumped, and
               only while add / remove stay inside the floor: an identical
               cubin with a reading beyond it is a host-side / launch-config
               change (grid, cluster, stream order live outside the cubin) and
               is classified by the reading like any other lever
Noise floor = max(0.5%, 3σ of the difference of two R-round medians), σ the
per-round spread of device ms pooled over every variant (a variant whose own
spread is > 4× the median variant's is left out of the pool and named).
"""
import glob
import hashlib
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KERNEL = "kernel.py"
NOISE_FLOOR_PCT = 0.5
# The lever floor is a NOISE_SIGMAS test on the difference of two medians of R
# rounds: sd(median of R) ≈ 1.2533 σ/√R for normal draws. The old floor, 3 ×
# the widest single variant's range, sat near 9σ on a 15-variant table: real
# +2% levers read null there (and one wild draw lifted it to 553%).
NOISE_SIGMAS = 3.0
_MEDIAN_SD_FACTOR = 1.2533
# a variant whose own round spread is this many times the median variant's is
# an outlier draw, not the node's noise: it stays out of the pooled σ
_OUTLIER_SPREAD_FACTOR = 4.0
MAX_SUBSETS = 6
# A group card (one kernel tuned on a size class of workloads) is priced one workload
# per job: the whole group x rounds x 2N+2 variants outlived the job clock (8
# workloads were killed at TASK_TIMEOUT with no verdict). The card's leaderboard reads this
# variable (== leaderboard.WORKLOAD_SCOPE_ENV) and tags its JSON; a card without group
# support ignores it and is timed whole. The table then says so (PRICE-SCOPE). The
# workload is the representative, or the one `--workload <name>` names: the driver
# prices each of the card's meta_workloads in its own job and merges the tables
# (loop/meta_shapes.py) -- priced on the representative alone, a lever's reading agreed
# in direction with the rest of the group only 61% of the time a change was real.
# Every bench, the role pass and the ablations' NCU clock read that one workload.
WORKLOAD_SCOPE_ENV = "FK_LB_WORKLOAD_SCOPE"
WORKLOAD_SCOPE = "representative"
_SCOPE_SEEN: dict = {}
_SCOPE_WANT = {"name": WORKLOAD_SCOPE}
PROBLEM_FILE = os.path.join("eval", "leaderboard", "problem.json")
AUTO_ALL_POS = "all-pos"
_RESERVED = ("base", "all", AUTO_ALL_POS)
MAX_ABLATIONS = 4
ABLATION_ROUNDS = 2
# the NCU kernel clock of the launch target, nothing else collected: an
# ablated kernel may compute garbage, the bench then refuses to time it, and
# this is the one reading that does not care
NCU_TIME_CMD = ("ncu --metrics gpu__time_duration.sum --profile-from-start off "
                "--target-processes all --csv python3 eval/ncu_run.py")
NCU_TIME_TIMEOUT = 900          # one ncu collection of the launch target
ABLATION_BENCH_TIMEOUT = 900    # the bench only records whether the ablation stayed correct
ABLATION_BUDGET_S = 3600        # the whole pass, all variants, all rounds
# The role pass (one `stall_price_run.py --pm` per timed variant, between the
# PRICE-TIMED and PRICE-OK copies of the table) runs under the JOB's clock:
# gpu_entry kills the whole job at TASK_TIMEOUT with SIGKILL, and a job killed
# mid-pass prints no PRICE-OK (12-15 variants x 1-2 min of NCU sampling
# outlived an 1800 s job on B200, four jobs in a row). The pass is
# budgeted from the job's remaining time: a variant is priced only when the
# time left covers it (the longest --pm seen so far, or ROLE_MIN_CALL_S before
# any), the call's own timeout is clamped to the time left, and the variants
# the budget cannot cover read n/a — the PRICE-OK copy is always printed.
ROLE_PM_TIMEOUT = 3000          # one --pm call when the job's clock is unknown
ROLE_PASS_MARGIN_S = 180        # kept back for the final table, the tree restore, the entry close
ROLE_MIN_CALL_S = 120           # the first --pm needs at least this much of the job's clock
TASK_TIMEOUT_ENV = "TASK_TIMEOUT"   # gpu_entry's kill clock (seconds), exported into the job
_NCU_UNIT_NS = {"nsecond": 1.0, "ns": 1.0, "usecond": 1e3, "us": 1e3,
                "msecond": 1e6, "ms": 1e6, "second": 1e9, "s": 1e9}


# ---------------------------------------------------------------- node hygiene
# A timed-out command is killed as a PROCESS GROUP, never as one child.
# ``subprocess.run(shell=True, timeout=)`` kills the shell (or ncu) alone; the
# launch target ncu spawned outlives it — with a deadlocked kernel resident on the
# GPU. On the same machine every later non-NCU reading then shares
# the GPU with that orphan: one hung ablation and the bench read 2.3x slow for the
# rest of the run, while ncu — which suspends other contexts while it replays —
# kept reading the same kernel time as before, and the confirmation archived the
# inflated number as the run's verdict. After the group kill the GPU is swept for
# compute processes still holding it; whatever survives is printed as NODE-DIRTY:
# the pass stops (its later readings would be garbage) and the driver re-prices the
# design once.
NODE_DIRTY_MARKER = "NODE-DIRTY"
NODE_SWEEP_WAIT_S = 30.0
_node_dirty: list = []


def node_dirty():
    """The NODE-DIRTY reason a timed-out command left behind, or None."""
    return _node_dirty[-1] if _node_dirty else None


def gpu_compute_pids():
    """PIDs holding a CUDA context on this GPU per nvidia-smi; None = unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [int(t) for t in re.findall(r"\b(\d+)\b", out)]


def _looks_like_ours(pid: int) -> bool:
    """Only a python / ncu process visible in OUR pid namespace gets a SIGKILL:
    nvidia-smi reports driver-side (host) pids, and inside a container that
    number can belong to an unrelated process — sshd, the runner itself."""
    if pid in (os.getpid(), os.getppid()):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ")
    except OSError:
        return False
    return b"python" in cmd or b"ncu" in cmd


def kill_process_group(p, grace_s: float = 5.0) -> None:
    """SIGKILL the whole group a timed-out ``Popen(start_new_session=True)``
    started (the shell, ncu, AND ncu's launch target), then reap."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        p.kill()
    try:
        p.communicate(timeout=grace_s)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass


def sweep_gpu_orphans(context: str, wait_s: float = NODE_SWEEP_WAIT_S, sleep=None):
    """Kill what still holds the GPU and wait for it to let go.

    Returns the pids still resident after ``wait_s`` (empty = clean). A process
    that survives SIGKILL this long is stuck in the driver behind a deadlocked
    kernel (or is not ours to kill): only a fresh node fixes that, so the
    survivors are recorded and printed as NODE-DIRTY."""
    sleep = sleep or time.sleep
    pids = gpu_compute_pids()
    if pids is None:
        # nvidia-smi itself hung or is missing — the very state that follows a
        # deadlocked kernel (which wedges the driver, which hangs the query).
        # "Cannot tell" is NOT "clean": a sweep that cannot confirm the GPU is
        # free must treat the node as dirty, or the orphan it could not see skews
        # every later reading. Nothing to kill (the query is already wedged), so
        # record and print NODE-DIRTY so the pass stops.
        reason = f"{context}: nvidia-smi unavailable after the command timed out " \
                 f"({wait_s:.0f}s sweep could not query the GPU) — cannot confirm it is free"
        _node_dirty.append(reason)
        print(f"NODE-HYGIENE {context}: nvidia-smi unavailable — cannot sweep the GPU", flush=True)
        print(f"{NODE_DIRTY_MARKER} {reason}", flush=True)
        return []
    if not pids:
        return []
    print(f"NODE-HYGIENE {context}: GPU still held by pid(s) {pids} — killing", flush=True)
    for pid in pids:
        if _looks_like_ours(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    deadline = time.monotonic() + wait_s
    left = pids
    while left and time.monotonic() < deadline:
        sleep(2.0)
        left = gpu_compute_pids() or []
    if left:
        reason = f"{context}: pid(s) {left} still hold the GPU {wait_s:.0f}s after SIGKILL"
        _node_dirty.append(reason)
        print(f"{NODE_DIRTY_MARKER} {reason}", flush=True)
    else:
        print(f"NODE-HYGIENE {context}: GPU released", flush=True)
    return left


def sh(cmd, timeout=2400, env=None):
    print("\n$ %s" % cmd[:200], flush=True)
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.Popen(cmd, shell=True, cwd=ROOT, env=e, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        kill_process_group(p)
        sweep_gpu_orphans(f"after `{cmd[:60]}` timed out ({timeout}s)")
        return 124, "", "TIMEOUT after %ds" % timeout


# ---------------------------------------------------------------- CUDA C++ kernels
# A CUDA-mode kernel (CUDA C++, loop language cuda-sol) is built by the card with
# nvcc + CUTLASS headers (~35 s a build). Every reading below is a fresh process, and
# the card caches a build per source, so each variant compiles once per job — but
# that once landed inside the timed pass, serially: 2N+2 variants x 35 s on top of
# the bench pushed CUDA pricing jobs past gpu_entry's
# TASK_TIMEOUT (11 of 17 SIGKILLed, no PRICE-TIMED table). Here every variant is
# built up front, in parallel, staged under the kernel file's own name (the card
# stages its build copy under that name, so the timed runs find the builds), and the
# timed rounds stop once the job clock cannot cover another one.
CUDA_PREBUILD_TIMEOUT = 1200    # the whole parallel prebuild
CUDA_PREBUILD_WORKERS = 8
# The prebuild is clamped to a share of the job's clock (prebuild_budget); on B200
# 7-17 variants built in 41-96 s, so the share only bites on a stuck or huge build.
PREBUILD_JOB_SHARE = 0.4
PREBUILD_MIN_S = 120
# A CUDA job keeps time for its ablation pass ahead of the role pass (ablation_reserve_s):
# the role pass (one ~2-4 min --pm per variant) used to spend the whole clock first, and
# CUDA pricing jobs left the ablations 6.7 s / 34.6 s / 95 s -- two of ten jobs
# carried no ablation reading at all. The roles of the last variants (all-L, subsets)
# read n/a instead; the ablations answer what each component of the kernel costs.
ABLATION_RESERVE_CAP_S = 600
_PREBUILD_SNIPPET = ("import sys; sys.path.insert(0, 'eval'); import leaderboard as L\n"
                     "try:\n    L.load_cuda_entry(sys.argv[1], '__fk_prebuild__')\n"
                     "except AttributeError:\n    pass\n")


# The CUDA suffix is spelled in two halves: this file is seeded into notes/, and
# a binary-reference recon tree's guard refuses notes that name C/C++ sources.
CUDA_SUFFIX = "." + "cu"


def is_cuda_kernel(kernel: str) -> bool:
    return str(kernel).endswith(CUDA_SUFFIX)


def prebuild_cuda(sources: dict, kernel: str, timeout=CUDA_PREBUILD_TIMEOUT,
                  workers=CUDA_PREBUILD_WORKERS) -> dict:
    """Build every variant source of a CUDA kernel through the card's own loader,
    ``workers`` at a time; ``{variant: ok}``. A variant that does not build reads
    ok=False here and COMPILE_ERROR in its bench, as it would have anyway."""
    import tempfile
    t0 = time.monotonic()
    stage = tempfile.mkdtemp(prefix="lp_prebuild_")
    todo = []
    for k, (name, src) in enumerate(sources.items()):
        d = os.path.join(stage, f"v{k}")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, os.path.basename(kernel))
        with open(path, "w") as fh:
            fh.write(src)
        todo.append((name, path))
    ok: dict = {}
    running: list = []
    errs: dict = {}
    while todo or running:
        while todo and len(running) < max(1, workers):
            name, path = todo.pop(0)
            # stderr goes to a file, never to a pipe nobody reads: a CUTLASS template
            # error runs to hundreds of KB, and a builder blocked on a full pipe held
            # the whole prebuild to its timeout (240 KB -> stuck 1200 s)
            errs[name] = os.path.join(os.path.dirname(path), "build.err")
            with open(errs[name], "w") as err:
                p = subprocess.Popen([sys.executable, "-c", _PREBUILD_SNIPPET, path], cwd=ROOT,
                                     stdout=subprocess.DEVNULL, stderr=err,
                                     start_new_session=True)
            running.append((name, p))
        for name, p in list(running):
            if p.poll() is not None:
                ok[name] = p.returncode == 0
                running.remove((name, p))
        if time.monotonic() - t0 > timeout:
            for name, p in running:
                kill_process_group(p)
                ok[name] = False
            for name, _ in todo:
                ok[name] = False
            break
        time.sleep(0.5)
    for name in [n for n, v in ok.items() if not v and n in errs][:PREBUILD_ERR_SHOWN]:
        # indented: a compiler message must never start a line the driver parses
        tail = "\n".join("  | " + ln for ln in _tail(errs[name], PREBUILD_ERR_TAIL).splitlines())
        print(f"PRICE PREBUILD {name} failed; build stderr tail:\n{tail}", flush=True)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"PRICE PREBUILD variants={len(ok)} built={sum(ok.values())} "
          f"failed={' '.join(n for n, v in ok.items() if not v) or 'none'} "
          f"secs={time.monotonic() - t0:.0f}", flush=True)
    return ok


PREBUILD_ERR_TAIL = 1500    # chars of a failed variant's build stderr echoed into the log
PREBUILD_ERR_SHOWN = 3      # failed variants whose tails are echoed (the rest are named)


def _tail(path: str, n: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - n))
            return fh.read().decode("utf-8", errors="replace").strip() or "(empty)"
    except OSError:
        return "(unreadable)"


def cuda_sass_sha(src: str, kernel: str):
    """``cubin_sha`` for a CUDA kernel: one sha over the SASS of the build the card
    made from exactly ``src`` (cuobjdump -sass of its extension .so); None when
    no build or no cuobjdump."""
    try:
        from orch_census_run import find_cuda_build  # type: ignore  # vendored flat in notes/
    except ImportError:
        from forgekernel.skills.orch_census_run import find_cuda_build
    so = find_cuda_build(src, os.path.basename(kernel))
    if so is None or shutil.which("cuobjdump") is None:
        return None
    proc = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return hashlib.sha256(sass_code_text(proc.stdout).encode()).hexdigest()[:12]


# nvcc names a translation unit's anonymous namespace after a hash of the source
# (`_GLOBAL__N__c5a99a67_9_kernel_cu_d698c71f`), and a kernel in it carries that name in
# every mangled symbol: a comment-only variant reads as different SASS (B200 smoke:
# a comment lever came back cubin_same_as_base=False). The name is not code.
_ANON_NS = re.compile(r"\d*_GLOBAL__N__[0-9A-Fa-f]+_\d+_[A-Za-z0-9_]*?_[0-9A-Fa-f]{8}")


def sass_code_text(dump: str) -> str:
    """The code of a ``cuobjdump -sass`` dump: function headers and instruction lines
    only (the fatbin headers can name the build's source path, which differs per
    variant build directory), anonymous-namespace hashes normalised."""
    return "\n".join(_ANON_NS.sub("_GLOBAL__N_", ln.strip()) for ln in dump.splitlines()
                     if re.match(r"\s*(/\*[0-9a-fA-F]+\*/|Function\s*:)", ln))


def round_fits(t0, last_round_s, clock=None, env=None) -> bool:
    """Whether the job clock still covers one more timed round (as long as the last
    one) plus ROLE_PASS_MARGIN_S for the table; True when the clock is unknown."""
    left = job_seconds_left(t0, clock=clock, env=env)
    return left is None or left - ROLE_PASS_MARGIN_S >= last_round_s


def lever_name(lever: dict) -> str:
    """The name as it appears on the LEVER line: one token (whitespace -> _)."""
    return re.sub(r"\s+", "_", str(lever.get("name", "")).strip()) or "unnamed"


# ---------------------------------------------------------------- variants
def apply_lever(src: str, lever: dict) -> str:
    for i, ed in enumerate(lever.get("edits", [])):
        old, new = ed["old"], ed["new"]
        n = src.count(old)
        if n != 1:
            raise ValueError(f"lever {lever_name(lever)!r} edit #{i + 1}: `old` occurs "
                             f"{n} times (need exactly 1)")
        src = src.replace(old, new)
    return src


def in_stack(lever: dict) -> bool:
    return lever.get("in_stack", True) is not False


def stack_names(levers: list) -> list:
    return [lever_name(lv) for lv in levers if in_stack(lv)]


def compose(base: str, levers: list, selected) -> str:
    """base with the selected levers applied, in lever-list order."""
    sel = set(selected)
    s = base
    for lv in levers:
        if lever_name(lv) in sel:
            s = apply_lever(s, lv)
    return s


def subset_name(sub: dict) -> str:
    return re.sub(r"\s+", "_", str(sub.get("name", "")).strip())


def check_subsets(subsets: list, names: list, lever_names: list | None = None) -> list:
    """The teacher's subsets, validated: a one-token name that is not a built
    variant's and not a lever's (``lever_names``: every lever of the spec, solo
    ones included — the driver keys its table by name across LEVER and VARIANT
    rows, so a subset wearing a lever's name would shadow that lever's price),
    at least one member, members are levers of this spec (``names``: the stack
    ones — a solo lever is priced alone by definition), no duplicates.
    Raises on the first bad subset (the design turn is re-prompted with it)."""
    seen = set()
    taken = set(lever_names if lever_names is not None else names)
    out = []
    for sub in (subsets or [])[:MAX_SUBSETS]:
        name = subset_name(sub)
        members = [re.sub(r"\s+", "_", str(m).strip()) for m in (sub.get("levers") or [])]
        if not name or name in _RESERVED or name.startswith("+") or name.startswith("all-"):
            raise ValueError(f"subset name {name!r} is empty or reserved (base/all/+L/all-L/{AUTO_ALL_POS})")
        if name in taken:
            raise ValueError(f"subset name {name!r} is a lever's name (a subset must not "
                             f"shadow a lever on the table)")
        if name in seen:
            raise ValueError(f"duplicate subset name {name!r}")
        if not members:
            raise ValueError(f"subset {name!r} names no levers")
        bad = [m for m in members if m not in names]
        if bad:
            raise ValueError(f"subset {name!r} names unknown / solo lever(s) {bad}")
        if len(set(members)) != len(members):
            raise ValueError(f"subset {name!r} repeats a lever")
        seen.add(name)
        out.append({"name": name, "levers": members,
                    "question": re.sub(r"\s+", " ", str(sub.get("question", ""))).strip()[:200]})
    return out


def build_variants(base: str, levers: list, subsets: list | None = None) -> dict:
    """name -> source.  Refuses (raises) on any non-unique edit, a no-op
    variant, or a malformed subset.  base and every +L always; all / all-L
    over the stack levers (``in_stack`` not false); then the teacher's subsets."""
    out = {"base": base}
    names = [lever_name(lv) for lv in levers]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate lever names {names}")
    for lv in levers:
        out["+" + lever_name(lv)] = apply_lever(base, lv)
    stack = stack_names(levers)
    if stack:
        out["all"] = compose(base, levers, stack)
        if len(stack) > 1:
            for skip in stack:
                out["all-" + skip] = compose(base, levers, [n for n in stack if n != skip])
    for sub in check_subsets(subsets, stack, names):
        out[sub["name"]] = compose(base, levers, sub["levers"])
    for k, v in out.items():
        if k != "base" and v == base:
            raise ValueError(f"variant {k} is byte-identical to base (no-op lever)")
    return out


def check_ablations(ablations: list, taken: list) -> list:
    """The ablation list normalised (one-token names), or raises: at most
    MAX_ABLATIONS, names unique, none reserved, none shadowing a lever or a
    subset (the driver's table is keyed by name across every row kind)."""
    if not isinstance(ablations, list):
        raise ValueError("ablations must be a list")
    if len(ablations) > MAX_ABLATIONS:
        raise ValueError(f"{len(ablations)} ablations (max {MAX_ABLATIONS})")
    out, seen = [], set()
    for a in ablations:
        if not isinstance(a, dict):
            raise ValueError("an ablation must be an object")
        a = dict(a)
        a["name"] = lever_name(a)
        if not a["name"] or a["name"] == "unnamed":
            raise ValueError("an ablation has no name")
        if a["name"] in _RESERVED or a["name"].startswith(("+", "all-", "~")):
            raise ValueError(f"ablation name {a['name']!r} is reserved")
        if a["name"] in taken:
            raise ValueError(f"ablation {a['name']!r} shadows a lever or subset name")
        if a["name"] in seen:
            raise ValueError(f"duplicate ablation name {a['name']!r}")
        seen.add(a["name"])
        if not a.get("edits"):
            raise ValueError(f"ablation {a['name']!r} has no edits")
        out.append(a)
    return out


def build_ablation_variants(base: str, ablations: list) -> dict:
    """``~<name>`` -> source: each ablation applied to base ALONE. Refuses a
    non-unique edit or a no-op exactly as build_variants does — an ablation
    that changed nothing would read as "this mechanism costs 0%"."""
    out = {}
    for a in ablations:
        src = apply_lever(base, a)
        if src == base:
            raise ValueError(f"ablation {a['name']!r} is byte-identical to base (no-op)")
        out["~" + a["name"]] = src
    return out


# ---------------------------------------------------------------- readings
def _dump_dir(variant):
    return os.path.join("/tmp", "lp_dump_" + re.sub(r"[^A-Za-z0-9_]", "_", variant))


def _scope_env():
    return {WORKLOAD_SCOPE_ENV: _SCOPE_WANT["name"]}


def scoped_workload_refused(name, root=None):
    """Why ``--workload <name>`` cannot be priced on this card, or None: the card must be
    a group card that lists its meta workloads (problem.json ``meta_workloads``, written
    by the forge revision whose card runtime understands a named scope -- an older card
    would ignore the name and time its whole group) and ``name`` one of its entries."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or ""):
        return f"bad workload name {name!r}"
    try:
        problem = json.load(open(os.path.join(root or ROOT, PROBLEM_FILE)))
    except (OSError, json.JSONDecodeError) as e:
        return f"no readable {PROBLEM_FILE} ({e})"
    if not problem.get("meta_workloads"):
        return "the card lists no meta_workloads (not a group card, or built before named scopes)"
    names = [str(g.get("name")) for g in (problem.get("group") or []) + (problem.get("check_only") or [])]
    if name not in names:
        return f"{name} is not a workload of the group ({', '.join(names)})"
    return None


def run_bench(variant, keep_cubin, timeout=2400):
    dump = _dump_dir(variant)
    env = _scope_env()
    if keep_cubin:
        shutil.rmtree(dump, ignore_errors=True)
        os.makedirs(dump, exist_ok=True)
        env.update({"CUTE_DSL_KEEP_CUBIN": "1", "CUTE_DSL_DUMP_DIR": dump})
    rc, out, err = sh("python3 eval/bench.py --mode all", env=env, timeout=timeout)
    m = re.search(r"=== JSON ===\s*(\{.*)", out, re.S)
    if not m:
        print("[bench:%s] rc=%d no JSON; stderr tail:\n%s" % (variant, rc, err[-2500:]),
              flush=True)
        return None
    try:
        j = json.loads(m.group(1))
    except json.JSONDecodeError:
        j = json.loads(m.group(1)[:m.group(1).rindex("}") + 1])
    if isinstance(j, dict) and isinstance(j.get("workload_scope"), dict) and not _SCOPE_SEEN:
        _SCOPE_SEEN.update(j["workload_scope"])
    return j


def scope_line():
    """``PRICE-SCOPE`` when the card was timed on part of its group, else None."""
    if not _SCOPE_SEEN:
        return None
    return (f"PRICE-SCOPE workloads={_SCOPE_SEEN.get('workloads', WORKLOAD_SCOPE)} "
            f"rep={_SCOPE_SEEN.get('rep', '?')} of={_SCOPE_SEEN.get('of', '?')}")


def cubin_sha(variant):
    """One sha over EVERY cubin the variant's bench dumped (each file's content
    hash, sorted, hashed together — names and dump order do not matter).  It
    used to hash only the largest cubin, so a lever that changed a smaller
    entry read `noop` next to readings of -33% / +49%.  None when nothing was
    dumped."""
    cubins = glob.glob(os.path.join(_dump_dir(variant), "**", "*.cubin"), recursive=True)
    if not cubins:
        return None
    digests = sorted(hashlib.sha256(open(p, "rb").read()).hexdigest() for p in cubins)
    return hashlib.sha256("\n".join(digests).encode()).hexdigest()[:12]


def job_seconds_left(t0, clock=None, env=None):
    """Seconds of the job's kill clock (TASK_TIMEOUT, exported by the job
    runner into the job) still unspent, counted from ``t0`` — None when
    the clock is unknown (no / unparseable TASK_TIMEOUT: a hand run)."""
    raw = (os.environ if env is None else env).get(TASK_TIMEOUT_ENV, "")
    try:
        total = float(raw)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    clock = clock or time.monotonic
    return total - (clock() - t0)


def role_pass_plan(t0, seen_s, clock=None, env=None, reserve_s=0.0):
    """(run, timeout_s, left_s) for the next --pm call: ``run`` False when the
    job's remaining clock (minus ROLE_PASS_MARGIN_S and ``reserve_s``, the time
    kept for the ablation pass) cannot cover it — the longest call seen so far,
    or ROLE_MIN_CALL_S before any; ``timeout_s`` is the call's own cap clamped
    to what is left. Unbounded (ROLE_PM_TIMEOUT) when the job's clock is unknown."""
    left = job_seconds_left(t0, clock=clock, env=env)
    if left is None:
        return True, ROLE_PM_TIMEOUT, None
    usable = left - ROLE_PASS_MARGIN_S - max(0.0, reserve_s)
    need = max([ROLE_MIN_CALL_S] + list(seen_s))
    if usable < need:
        return False, 0, left
    return True, max(1, min(ROLE_PM_TIMEOUT, int(usable))), left


def prebuild_budget(t0, clock=None, env=None) -> float:
    """The CUDA prebuild's own timeout: CUDA_PREBUILD_TIMEOUT, but never more than
    PREBUILD_JOB_SHARE of what the job's clock leaves (the timed rounds, the role
    pass and the ablations need the rest; a 1200 s prebuild in an 1800 s job left
    round 0 to be SIGKILLed with no table), and never under PREBUILD_MIN_S."""
    left = job_seconds_left(t0, clock=clock, env=env)
    if left is None:
        return float(CUDA_PREBUILD_TIMEOUT)
    return float(min(CUDA_PREBUILD_TIMEOUT, max(PREBUILD_MIN_S, left * PREBUILD_JOB_SHARE)))


def ablation_reserve_s(n_ablations: int, rounds: int, bench_s) -> float:
    """Seconds the role pass leaves the ablation pass of a CUDA job: base and every
    ablation, an NCU call each round plus round 0's bench, each about as long as the
    timed pass's median bench call (``bench_s``), capped at ABLATION_RESERVE_CAP_S.
    0 with no ablations or no bench reading."""
    if n_ablations <= 0 or not bench_s:
        return 0.0
    per = statistics.median(bench_s)
    calls = (n_ablations + 1) * (max(1, rounds) + 1)
    return float(min(ABLATION_RESERVE_CAP_S, per * calls))


def run_price(variant, timeout=ROLE_PM_TIMEOUT):
    # roles only: no -lineinfo build of every variant (one more ~40 s nvcc run each)
    rc, out, err = sh("python3 notes/stall_price_run.py --pm", timeout=timeout,
                      env={**_scope_env(), "FK_CUDA_LINEINFO": "0"})
    roles = None
    for ln in (out + "\n" + err).splitlines():
        if ln.startswith("ROLE-JSON: "):
            try:
                roles = json.loads(ln[len("ROLE-JSON: "):])
            except json.JSONDecodeError:
                pass
    return roles


def parse_ncu_time_csv(text: str):
    """(median_ns, kernel_name, launches) of the dominant kernel in an ncu
    --csv page carrying gpu__time_duration.sum, or None. Dominant = the
    kernel name whose launches sum to the most time (the stall-pricing rule),
    so a launch storm of small auxiliaries cannot out-vote the kernel."""
    import csv
    import io
    rows = list(csv.reader(io.StringIO(text)))
    head = next((i for i, r in enumerate(rows) if "Kernel Name" in r and "Metric Value" in r), None)
    if head is None:
        return None
    cols = rows[head]
    ik, im, iu, iv = (cols.index("Kernel Name"), cols.index("Metric Name"),
                      cols.index("Metric Unit"), cols.index("Metric Value"))
    per: dict = {}
    for r in rows[head + 1:]:
        if len(r) <= max(ik, im, iu, iv) or r[im] != "gpu__time_duration.sum":
            continue
        try:
            val = float(r[iv].replace(",", "")) * _NCU_UNIT_NS.get(r[iu].strip().lower(), 1.0)
        except ValueError:
            continue
        per.setdefault(r[ik], []).append(val)
    if not per:
        return None
    name = max(per, key=lambda k: sum(per[k]))
    return statistics.median(per[name]), name, len(per[name])


class AblationHung(Exception):
    """The launch target did not finish inside NCU_TIME_TIMEOUT: the ablated
    kernel deadlocked (or never returns). Distinct from "no duration rows"
    so the pass can stop spending on the variant."""


def kernel_token(name) -> str:
    """A kernel name as one token for the ABLATE line: ncu reports demangled
    names (``void foo<int>(int *, float)``) whose spaces would end the
    ``kernel=`` field early and unparse the whole row."""
    return re.sub(r"\s+", "_", str(name or "")).strip("_") or "none"


def run_ncu_time(variant):
    """The NCU kernel clock of the tree as written: (median_ns, kernel, launches)
    or None (no ncu, no launch, no rows). Raises AblationHung on timeout."""
    rc, out, err = sh(NCU_TIME_CMD, timeout=NCU_TIME_TIMEOUT, env=_scope_env())
    if rc == 124:
        raise AblationHung(f"{variant}: ncu did not return in {NCU_TIME_TIMEOUT}s")
    got = parse_ncu_time_csv(out)
    if got is None:
        print("[ncu-time:%s] rc=%d no duration rows; tail:\n%s" % (variant, rc, (err or out)[-1500:]),
              flush=True)
    return got


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def spread_pct(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    return 100.0 * (max(xs) - min(xs)) / statistics.median(xs)


def role_ns(roles, role):
    try:
        return roles["roles"][role]["exposed_ns"]
    except (KeyError, TypeError):
        return None


def ratio_reading(perf: dict):
    """The same-job ratio to the reference: the bench contract's `vs_ref_device`
    (the legacy `vs_cublas_device` spelling is accepted as a fallback)."""
    r = perf.get("vs_ref_device")
    return perf.get("vs_cublas_device") if r is None else r


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    t_job0 = time.monotonic()       # the job's kill clock, as near its start as this process sees
    args = list(sys.argv[1:] if argv is None else argv)
    _SCOPE_WANT["name"] = WORKLOAD_SCOPE
    _SCOPE_SEEN.clear()
    if "--workload" in args:
        # one workload of a group card per job (the driver merges the per-workload tables)
        i = args.index("--workload")
        want = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
        if (why := scoped_workload_refused(want)) is not None:
            print(f"PRICE-FAIL reason=workload-refused ({why})", flush=True)
            return 0
        _SCOPE_WANT["name"] = want
    no_roles = "--no-roles" in args
    if no_roles:
        # a group card's non-representative workload (the driver passes it): the role
        # columns are read on the representative alone -- on the others they were never
        # quoted (0 of 207 per-shape role readings) and cost a third of the job
        args.remove("--no-roles")
    spec_path = args[0] if args else os.path.join("notes", "levers.json")
    try:
        spec = json.load(open(os.path.join(ROOT, spec_path)))
    except (OSError, json.JSONDecodeError) as e:
        print(f"PRICE-FAIL reason=levers-unreadable ({e})", flush=True)
        return 0
    levers = spec.get("levers") or []
    ablations = spec.get("ablations") or []
    rounds = int(spec.get("rounds", 3))
    abl_rounds = int(spec.get("ablation_rounds", ABLATION_ROUNDS))
    abl_budget = float(spec.get("ablation_budget_s", ABLATION_BUDGET_S))
    price_roles = bool(spec.get("roles", True)) and not no_roles
    if no_roles:
        print("PRICE ROLES skipped: --no-roles (a non-representative workload of a group card)",
              flush=True)
    kernel = str(spec.get("kernel_file") or DEFAULT_KERNEL)
    if not levers and not ablations:
        print("PRICE-FAIL reason=no-levers", flush=True)
        return 0
    kpath = os.path.join(ROOT, kernel)
    try:
        base_src = open(kpath).read()
    except OSError as e:
        print(f"PRICE-FAIL reason=kernel-unreadable ({kernel}: {e})", flush=True)
        return 0
    try:
        subsets = check_subsets(spec.get("variants") or [], stack_names(levers),
                                [lever_name(lv) for lv in levers]) if levers else []
        variants = build_variants(base_src, levers, subsets) if levers else {"base": base_src}
        ablations = check_ablations(ablations, [lever_name(lv) for lv in levers]
                                    + [sb["name"] for sb in subsets])
        abl_variants = build_ablation_variants(base_src, ablations)
    except ValueError as e:
        print(f"PRICE-FAIL reason=lever-edit ({e})", flush=True)
        return 0
    order = list(variants)          # base, +L..., all, all-L..., subsets...
    print(f"PRICE kernel_file={kernel} variants: " + " ".join(order)
          + (" | ablations: " + " ".join(abl_variants) if abl_variants else ""), flush=True)
    cuda = is_cuda_kernel(kernel)
    if cuda:
        # every build up front and in parallel: the timed rounds below then load
        # cached modules instead of compiling inside the job clock
        prebuild_cuda({**variants, **abl_variants}, kernel, timeout=prebuild_budget(t_job0))
    if not levers:
        # ablation-only spec (a verdict probe written as a JSON file): no lever
        # table to build, the base is timed inside the ablation pass
        left = job_seconds_left(t_job0)
        if left is not None:
            abl_budget = max(0.0, min(abl_budget, left - ROLE_PASS_MARGIN_S))
        try:
            time_ablations(ablations, abl_variants, abl_rounds, kpath, base_src, abl_budget)
        finally:
            open(kpath, "w").write(base_src)
        return 0

    dev = {v: [] for v in order}
    ratio = {v: [] for v in order}
    ok = {v: True for v in order}
    dead = set()
    roles = {}
    auto = {}
    timed = False
    dumped: set = set()
    bench_s: list = []              # seconds per timed bench call (a CUDA job's ablation reserve)

    def time_variants(names, tag, dev_map, ratio_map, ok_map, dead_set):
        """One pass: every variant in ``names`` interleaved, ``rounds`` times,
        readings into the dicts given (a pass owns its readings — a second
        pass must not pool its base with the first's). False when base died."""
        t_round = None
        for rnd in range(rounds):
            if cuda and rnd > 0 and t_round is not None \
                    and not round_fits(t_job0, time.monotonic() - t_round):
                # a CUDA job's clock cannot cover another round: the table is
                # read off the rounds already timed instead of the job dying
                # under gpu_entry's SIGKILL with no table at all
                print(f"PRICE ROUNDS cut at {rnd}/{rounds} pass_tag={tag}: job clock has "
                      f"{job_seconds_left(t_job0):.0f}s left", flush=True)
                break
            t_round = time.monotonic()
            for v in names:
                if v in dead_set:
                    continue
                open(kpath, "w").write(variants[v])
                t_bench = time.monotonic()
                j = run_bench(v, keep_cubin=(v not in dumped))
                bench_s.append(time.monotonic() - t_bench)
                dumped.add(v)
                if node_dirty():
                    # a timed-out bench left the GPU held: every reading from
                    # here on would be the orphan's share of the GPU, not the
                    # variant's — stop the pass; the driver re-prices on a
                    # fresh node
                    print(f"PRICE-FAIL reason=node-dirty ({node_dirty()})", flush=True)
                    return False
                if j is None or not j.get("correctness", {}).get("pass"):
                    ok_map[v] = False
                    dead_set.add(v)
                    print(f"PRICE ROUND {rnd} {v} pass=False pass_tag={tag}", flush=True)
                    if v == "base":
                        return False
                    continue
                perf = j.get("performance") or {}
                dev_map[v].append(perf.get("device_median_ms"))
                ratio_map[v].append(ratio_reading(perf))
                print(f"PRICE ROUND {rnd} {v} pass=True device={perf.get('device_median_ms')} "
                      f"vs_ref_device={ratio_reading(perf)} pass_tag={tag}", flush=True)
        return True

    try:
        if not time_variants(order, "main", dev, ratio, ok, dead):
            # nothing can be priced against a dead base: stop here instead of
            # timing 2N+1 variants for a table that cannot be built (a dirty
            # node has already printed its own PRICE-FAIL)
            if not node_dirty():
                print("PRICE-FAIL reason=base-failed-correctness", flush=True)
            return 0
        # second pass: the stack minus the harmful levers. One harmful member
        # drags every all-L reading with it (a table with one lever at -17.7%
        # alone read all=-9% and every remove leg negative), so the stack's
        # real value is only visible once it is out. The pass re-times base
        # alongside: its delta is read against THIS pass's base, so the node
        # drift between the passes cancels the way it does inside the first.
        auto = plan_all_pos(levers, variants, order, ok, dev, ratio)
        if auto.get("levers"):
            variants[AUTO_ALL_POS] = compose(base_src, levers, auto["levers"])
            if cuda:
                prebuild_cuda({AUTO_ALL_POS: variants[AUTO_ALL_POS]}, kernel,
                              timeout=prebuild_budget(t_job0))
            order.append(AUTO_ALL_POS)
            dev[AUTO_ALL_POS], ratio[AUTO_ALL_POS], ok[AUTO_ALL_POS] = [], [], True
            print(f"PRICE {AUTO_ALL_POS}: stack minus {','.join(auto['dropped'])} "
                  f"(harmful alone) = {','.join(auto['levers'])}; second pass re-times "
                  f"base alongside", flush=True)
            dev2 = {"base": [], AUTO_ALL_POS: []}
            ratio2 = {"base": [], AUTO_ALL_POS: []}
            ok2 = {"base": True, AUTO_ALL_POS: True}
            time_variants(["base", AUTO_ALL_POS], AUTO_ALL_POS, dev2, ratio2, ok2, set())
            if node_dirty():
                return 0   # its PRICE-FAIL is printed; no table off a held GPU
            ok[AUTO_ALL_POS] = ok2[AUTO_ALL_POS]
            auto.update({"dev": dev2, "ratio": ratio2, "ok": ok2})
        shas = ({v: cuda_sass_sha(variants[v], kernel) for v in order} if cuda
                else {v: cubin_sha(v) for v in order})
        # the timing table first: a job killed in the (long) role pass still
        # leaves a parseable price table in the log
        _emit_table(levers, variants, order, ok, dev, ratio, shas, roles, final=False,
                    subsets=subsets, auto=auto)
        timed = True
        if price_roles:
            seen_s: list = []       # how long each --pm took: the next one is assumed as long
            reserve = (ablation_reserve_s(len(abl_variants), abl_rounds, bench_s)
                       if cuda and abl_variants else 0.0)
            if reserve:
                print(f"PRICE ROLES reserve={reserve:.0f}s kept for {len(abl_variants)} "
                      f"ablation(s) x {abl_rounds} round(s)", flush=True)
            for v in order:
                if not ok[v]:
                    continue
                run, tmo, left = role_pass_plan(t_job0, seen_s, reserve_s=reserve)
                if not run:
                    # the job's clock cannot cover another --pm: the variants
                    # left read n/a in the PRICE-OK copy below rather than
                    # the whole job dying under gpu_entry's SIGKILL with no
                    # PRICE-OK at all (four jobs, four tables lost)
                    pending = [w for w in order[order.index(v):] if ok[w]]
                    print(f"PRICE ROLES {v} skipped: job clock has {left:.0f}s left "
                          f"(margin {ROLE_PASS_MARGIN_S}s, ablation reserve {reserve:.0f}s, a --pm takes up to "
                          f"{max([ROLE_MIN_CALL_S] + seen_s):.0f}s) — {len(pending)} variant(s) "
                          f"unpriced for roles: {' '.join(pending)}", flush=True)
                    break
                open(kpath, "w").write(variants[v])
                t_call = time.monotonic()
                roles[v] = run_price(v, timeout=tmo)
                seen_s.append(time.monotonic() - t_call)
                if node_dirty():
                    # a timed-out role command left the GPU held: the ROLE rows
                    # gathered so far are worth keeping (the table below is
                    # emitted with final=False) but the FINAL table and the
                    # ablation pass would read the orphan's share of the GPU —
                    # stop, and let the driver re-price on a fresh node
                    print(f"PRICE-FAIL reason=node-dirty ({node_dirty()})", flush=True)
                    timed = False
                    break
                r = roles[v] or {}
                print(f"PRICE ROLES {v} tensor={r.get('tensor_active')} "
                      f"spin_ns={role_ns(r, 'spin')} producer_ns={role_ns(r, 'copy-producer')} "
                      f"consumer_ns={role_ns(r, 'mma-consumer')}", flush=True)
        if timed:
            _emit_table(levers, variants, order, ok, dev, ratio, shas, roles, final=True,
                        subsets=subsets, auto=auto)
        # the ablation pass LAST: the lever table is on the log in both copies
        # before any correctness-breaking variant runs, and a hung ablation
        # dies on its own timeout without taking the table with it (never on a
        # node the role pass already found dirty)
        if abl_variants and not node_dirty():
            # the pass's own budget, and never more than the job's clock leaves
            # (the ABLATE-OK copy must land before gpu_entry's kill)
            left = job_seconds_left(t_job0)
            if left is not None:
                abl_budget = max(0.0, min(abl_budget, left - ROLE_PASS_MARGIN_S))
            time_ablations(ablations, abl_variants, abl_rounds, kpath, base_src, abl_budget)
    finally:
        open(kpath, "w").write(base_src)      # leave the tree as we found it
    return 0


def time_ablations(ablations, abl_variants, rounds, kpath, base_src, budget_s=None,
                   clock=None):
    """The ablation pass: base and every ablation interleaved ``rounds``
    times on the NCU kernel clock. Per variant and round: ncu FIRST — a
    timeout marks the variant ``hung`` and it is never run again (no bench,
    no later round); then, in round 0 only (and later rounds only while it
    keeps passing), the bench with its own short timeout, so an ablation that
    happens to stay correct also gets a device-ms delta. ``budget_s`` caps the
    whole pass: once spent, every remaining variant reads ``status=budget``.
    Prints one ABLATE line per ablation and ABLATE-OK / ABLATE-FAIL."""
    import time
    clock = clock or time.monotonic
    budget_s = ABLATION_BUDGET_S if budget_s is None else budget_s
    t0 = clock()
    names = ["base"] + list(abl_variants)
    ncu = {v: [] for v in names}
    dev = {v: [] for v in names}
    kern = {v: None for v in names}
    corr = {v: "skipped" for v in names}
    status = {v: "ok" for v in names}
    for rnd in range(rounds):
        for v in names:
            if status[v] == "hung":
                continue
            if clock() - t0 > budget_s:
                if not any(x is not None for x in ncu[v]):
                    status[v] = "budget"
                print(f"ABLATE ROUND {rnd} {v} skipped: pass budget {budget_s}s spent", flush=True)
                continue
            open(kpath, "w").write(base_src if v == "base" else abl_variants[v])
            try:
                got = run_ncu_time(v)
            except AblationHung as e:
                status[v] = "hung"
                ncu[v].append(None)
                print(f"ABLATE ROUND {rnd} {v} HUNG ({e}) — no bench, no further rounds", flush=True)
                if node_dirty():
                    # the hung target is still on the GPU: no later reading in
                    # this pass can be trusted (the bench shares the GPU with
                    # it) — stop here; the driver reruns on a fresh node
                    print(f"ABLATE-FAIL reason=node-dirty ({node_dirty()})", flush=True)
                    return
                if v == "base":
                    break
                continue
            ncu[v].append(got[0] if got else None)
            if got:
                kern[v] = got[1]
            elif status[v] == "ok":
                status[v] = "nodata"
            if rnd == 0 or corr[v] == "pass":
                j = run_bench(v, keep_cubin=False, timeout=ABLATION_BENCH_TIMEOUT)
                if node_dirty():
                    # the bench timed out and left the GPU held: every later
                    # reading in this pass would be the orphan's share — stop
                    # here; the driver reruns the ablation on a fresh node
                    print(f"ABLATE-FAIL reason=node-dirty ({node_dirty()})", flush=True)
                    return
                if j is None:
                    corr[v] = "nojson"
                elif j.get("correctness", {}).get("pass"):
                    corr[v] = "pass"
                    dev[v].append((j.get("performance") or {}).get("device_median_ms"))
                else:
                    corr[v] = "fail"
            print(f"ABLATE ROUND {rnd} {v} ncu_ns={got[0] if got else None} "
                  f"kernel={kernel_token(got[1]) if got else None} correctness={corr[v]}", flush=True)
        if status["base"] == "hung":
            break
    base_ns = med(ncu["base"])
    if base_ns is None:
        print("ABLATE-FAIL reason=" + ("base-hung (the unablated tree did not return under ncu)"
                                       if status["base"] == "hung" else
                                       "base-not-timed (ncu produced no duration for base)"), flush=True)
        return
    spreads = [spread_pct(ncu[v]) for v in names if any(x is not None for x in ncu[v])]
    noise = max(NOISE_FLOOR_PCT, 3.0 * max(spreads)) if spreads else NOISE_FLOOR_PCT
    base_ms = med(dev["base"])
    n_ok = 0
    for a in ablations:
        v = "~" + a["name"]
        ns = med(ncu[v])
        # positive = the ablated tree is FASTER = the mechanism costs this much today
        d = None if ns is None or not ns else 100.0 * (base_ns / ns - 1.0)
        ms = med(dev[v])
        dd = None if (ms is None or base_ms is None or not ms) else 100.0 * (base_ms / ms - 1.0)
        n_ok += d is not None
        st = status[v] if d is None else "ok"
        print(f"ABLATE {a['name']} delta={_pct(d)} ncu_ns={ns} base_ncu_ns={base_ns} "
              f"correctness={corr[v]} device_delta={_pct(dd)} kernel={kernel_token(kern[v])} "
              f"status={st} h={str(a.get('hypothesis', '')).replace(chr(10), ' ')}", flush=True)
    print(f"PRICE ablation_noise_floor={noise:.2f}% metric=ncu_ns", flush=True)
    print(f"ABLATE-OK ablations={n_ok} rounds={rounds} base_ncu_ns={base_ns}", flush=True)


# ---------------------------------------------------------------- scoring
def make_scorer(order, ok, dev, ratio):
    """(score, delta, use_ratio): the same-job ratio when the bench carries a
    reference, else inverted device ms so "higher = better" holds."""
    use_ratio = bool(ratio.get("base")) and all(x is not None for x in ratio["base"])

    def score(v):
        if not ok.get(v):
            return None
        return med(ratio[v]) if use_ratio else (1.0 / med(dev[v]) if med(dev[v]) else None)

    def delta(a, b):
        sa, sb = score(a), score(b)
        if sa is None or sb is None or not sb:
            return None
        return 100.0 * (sa / sb - 1.0)

    return score, delta, use_ratio


def pooled_sigma(order, ok, dev):
    """(σ %, rounds, outliers): the per-round relative spread of device ms,
    pooled over every variant with at least two readings. A variant whose own
    spread is more than _OUTLIER_SPREAD_FACTOR × the median variant's (with at
    least four variants to take a median over) is one wild draw, not the node's
    noise — it stays out of the pool and is named. rounds = the fewest readings
    any pooled variant has (the conservative R for the floor). σ None when no
    variant repeated."""
    per = {}
    for v in order:
        xs = [x for x in (dev.get(v) or []) if x] if ok.get(v) else []
        if len(xs) < 2:
            continue
        m = statistics.mean(xs)
        per[v] = (sum((x / m - 1.0) ** 2 for x in xs), len(xs) - 1, len(xs))
    if not per:
        return None, 0, []
    sd = {v: math.sqrt(ss / df) for v, (ss, df, _) in per.items()}
    typical = statistics.median(sd.values())
    outliers = [v for v in per if len(per) >= 4 and typical > 0
                and sd[v] > _OUTLIER_SPREAD_FACTOR * typical]
    pooled = [v for v in per if v not in outliers]
    df = sum(per[v][1] for v in pooled)
    sigma = 100.0 * math.sqrt(sum(per[v][0] for v in pooled) / df)
    return sigma, min(per[v][2] for v in pooled), outliers


def noise_floor(order, ok, dev):
    """The floor a lever's delta must clear, from the STUDENT's own repeat
    spread (device ms): the same-job ratio inherits the reference's
    round-to-round jitter too, which can triple into a floor that reads a real
    1-2% lever as null while device ms barely moved.

    NOISE_SIGMAS × the sd of a difference of two R-round medians,
    √2 × 1.2533 σ/√R, never below NOISE_FLOOR_PCT."""
    sigma, rounds, _ = pooled_sigma(order, ok, dev)
    if sigma is None:
        return NOISE_FLOOR_PCT
    floor = NOISE_SIGMAS * math.sqrt(2.0) * _MEDIAN_SD_FACTOR * sigma / math.sqrt(rounds)
    return max(NOISE_FLOOR_PCT, floor)


def lever_class(add, rem, *, noise, ok_add=True, same=False, all_in_spec=False,
                all_ok=True) -> str:
    """The class ladder, one place for the table and for the all-pos plan.

    ``add`` = +L vs base, ``rem`` = all-L vs all (None when not built / not
    measured). ``harmful`` needs BOTH a loss alone AND no loss on removal
    (rem None or ≈ 0): a lever that costs alone but whose removal from the
    stack costs more is ``coupled`` — the stack needs it — never harmful.

    ``same`` (identical cubins) is ``noop`` only while the readings agree: a
    launch-config or host-side edit leaves every cubin byte-identical and still
    moves the time, so a reading beyond the floor is classified as one."""
    if not ok_add:
        return "broken"
    if same and (add is None or abs(add) <= noise) and (rem is None or abs(rem) <= noise):
        return "noop"
    if add is None:
        return "unmeasured"
    if add < -noise and (rem is None or rem >= -noise):
        # a lever that HURTS alone and is not missed by the stack: the reading
        # says the direction is wrong, and the stack sum carries its loss
        return "harmful"
    if add > noise and rem is None and all_in_spec and not all_ok:
        # the single reading stands, the stack broke correctness: whether the
        # others subsume it was never measured
        return "unstacked"
    if add > noise and (rem is None or rem < -noise * 0.5):
        return "independent"
    if add <= noise and rem is not None and rem < -noise:
        return "coupled"
    if add > noise and rem is not None and rem >= -noise * 0.5:
        return "dependent"
    return "null"


def plan_all_pos(levers, variants, order, ok, dev, ratio) -> dict:
    """{"levers": [...], "dropped": [...]} when an all-pos pass is worth buying:
    at least one stack lever reads ``harmful`` (lever_class: costs alone AND
    the stack does not miss it — a coupled lever stays in), and at least two
    stack levers with a measured single-add remain."""
    _, delta, _ = make_scorer(order, ok, dev, ratio)
    noise = noise_floor(order, ok, dev)
    dropped, keep = [], []
    for n in stack_names(levers):
        add = delta("+" + n, "base")
        if add is None:
            continue
        rem = delta("all-" + n, "all") if ("all-" + n) in variants else None
        cls = lever_class(add, rem, noise=noise, ok_add=ok.get("+" + n, True),
                          all_in_spec=("all-" + n) in variants, all_ok=ok.get("all", True))
        (dropped if cls == "harmful" else keep).append(n)
    if dropped and len(keep) >= 2 and AUTO_ALL_POS not in variants:
        return {"levers": keep, "dropped": dropped}
    return {"levers": [], "dropped": dropped}


def _emit_table(levers, variants, order, ok, dev, ratio, shas, roles, *, final,
                subsets=None, auto=None):
    rounds = len(dev["base"])
    score, delta, use_ratio = make_scorer(order, ok, dev, ratio)
    noise = noise_floor(order, ok, dev)
    base_sha = shas.get("base")
    names = [lever_name(lv) for lv in levers]
    has_all = "all" in variants
    total_single = 0.0
    for name in names:
        add = delta("+" + name, "base")
        rem = delta("all-" + name, "all") if ("all-" + name) in variants else None
        add_ms = med(dev["+" + name])
        rem_ms = med(dev.get("all-" + name, []))
        sha = shas.get("+" + name)
        # `noop` needs BOTH cubins: two missing dumps are not two equal kernels
        same = sha is not None and base_sha is not None and sha == base_sha
        cls = lever_class(add, rem, noise=noise, ok_add=bool(ok.get("+" + name)), same=same,
                          all_in_spec=("all-" + name) in variants, all_ok=bool(ok.get("all")))
        if add is not None:
            total_single += add
        sp_add = role_ns(roles.get("+" + name), "spin")
        sp_rem = role_ns(roles.get("all-" + name), "spin")
        print(f"LEVER {name} add={_pct(add)} remove={_pct(rem)} add_ms={add_ms} "
              f"remove_ms={rem_ms} class={cls} spin_add={sp_add} spin_all_minus={sp_rem} "
              f"cubin_same_as_base={same if (sha and base_sha) else 'unknown'}", flush=True)
    auto = auto or {}
    # all-pos is read on its own pass (base re-timed alongside it): the first
    # pass's base is a different moment on the node
    pos_d = pos_ms = pos_base_ms = None
    if auto.get("levers") and auto.get("dev"):
        _, delta2, _ = make_scorer(["base", AUTO_ALL_POS], auto.get("ok", {}),
                                   auto["dev"], auto["ratio"])
        pos_d = delta2(AUTO_ALL_POS, "base")
        pos_ms = med(auto["dev"].get(AUTO_ALL_POS, []))
        pos_base_ms = med(auto["dev"].get("base", []))
    for sub in list(subsets or []) + ([{"name": AUTO_ALL_POS, "levers": auto["levers"],
                                        "question": "auto: the stack without the harmful levers "
                                                    f"(dropped {','.join(auto['dropped'])}; "
                                                    f"second pass, base re-timed at {pos_base_ms} ms)"}]
                                      if auto.get("levers") else []):
        v = sub["name"]
        if v not in variants:
            continue
        d, ms = (pos_d, pos_ms) if v == AUTO_ALL_POS else (delta(v, "base"), med(dev.get(v, [])))
        print(f"VARIANT {v} levers={','.join(sub['levers'])} delta={_pct(d)} "
              f"ms={ms} spin={role_ns(roles.get(v), 'spin')} "
              f"q={sub.get('question', '')}", flush=True)
    all_d = delta("all", "base") if has_all else None
    print(f"STACK all={_pct(all_d)} sum_singles={_pct(total_single)} "
          f"coupling={_pct(None if all_d is None else all_d - total_single)} "
          f"all_pos={_pct(pos_d)} all_pos_levers={','.join(auto.get('levers') or []) or 'none'} "
          f"dropped={','.join(auto.get('dropped') or []) or 'none'}", flush=True)
    sigma, sig_rounds, outliers = pooled_sigma(order, ok, dev)
    if (sl := scope_line()) is not None:
        print(sl, flush=True)
    print(f"PRICE noise_floor={noise:.2f}% metric={'vs_ref_device' if use_ratio else 'device_ms'} "
          f"sigma={'n/a' if sigma is None else f'{sigma:.2f}%'} rounds={sig_rounds} "
          f"outliers={','.join(outliers) or 'none'}", flush=True)
    if not final:
        print(f"PRICE-TIMED levers={len(levers)} rounds={rounds}", flush=True)
        return
    print(f"PRICE-OK levers={len(levers)} rounds={rounds} base_ms={med(dev['base'])} "
          f"all_ms={med(dev['all']) if (has_all and ok.get('all')) else None} "
          f"spin_base={role_ns(roles.get('base'), 'spin')} spin_all={role_ns(roles.get('all'), 'spin')} "
          f"tensor_base={(roles.get('base') or {}).get('tensor_active')} "
          f"tensor_all={(roles.get('all') or {}).get('tensor_active')}", flush=True)


def _pct(x):
    return "n/a" if x is None else f"{x:+.2f}%"


if __name__ == "__main__":
    sys.exit(main())
