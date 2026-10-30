"""The judge: the PyTorch statement of a task's operator, written by the teacher.

Before anything else in a run, a teacher turn reads the task (``description.md``, the
shapes, the reference's source when the task is in source mode) and writes three files into
``<run>/check/``:

    definition.json   the signature: {"inputs": {name: {"shape": [...], "dtype": ...}},
                      "outputs": {...}} -- shapes over the task's axes (or integers),
                      inputs in the order ``run`` takes them
    judge.py          ``run(<inputs>)`` -> the outputs (a tensor, or a tuple in output
                      order): the operator's semantics, written for clarity, never speed;
                      ``make_inputs(axes, group, seed, device)`` -> {input name: tensor}
                      for the input groups the teacher picks
    groups.json       {shape name: [{"name", "intent", "seed"}]}: the input groups the
                      teacher adds so every shape has at least three

``check_problems`` validates them without a GPU; the run then calibrates the judge against
the reference on the GPU (eval/leaderboard.py ``--mode calibrate``) and hands the result
back to the teacher until they agree. A task whose reference a /goal session writes gets a
judge written from the description alone: the two are independent statements of the same
operator, and the generated reference is accepted only when the judge passes it.
"""

from __future__ import annotations

import ast
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from forgekernel.task import MIN_INPUT_GROUPS

CHECK_DIR = "check"
DEFINITION = "definition.json"
JUDGE = "judge.py"
GROUPS = "groups.json"
ATTEMPTS = 3

DTYPES = ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2",
          "int64", "int32", "int16", "int8", "bool")
# safetensors header dtype codes -> the signature's names
_ST_DTYPES = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
              "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "I64": "int64",
              "I32": "int32", "I16": "int16", "I8": "int8", "BOOL": "bool"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GROUP = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class Check:
    definition: dict
    judge_src: str
    groups: dict = field(default_factory=dict)   # shape -> [{"name", "intent", "seed"}]

    @property
    def inputs(self) -> list[str]:
        return list(self.definition["inputs"])

    @property
    def outputs(self) -> list[str]:
        return list(self.definition["outputs"])


def load_check(check_dir: Path) -> Check:
    d = Path(check_dir)
    return Check(definition=json.loads((d / DEFINITION).read_text(encoding="utf-8")),
                 judge_src=(d / JUDGE).read_text(encoding="utf-8"),
                 groups=json.loads((d / GROUPS).read_text(encoding="utf-8")))


def check_problems(check_dir: Path, task) -> list[str]:
    """Everything wrong with the teacher's files, statically (empty = ready to calibrate)."""
    d = Path(check_dir)
    out: list[str] = []
    for name in (DEFINITION, JUDGE, GROUPS):
        if not (d / name).is_file():
            out.append(f"{CHECK_DIR}/{name} is missing")
    if out:
        return out
    try:
        defn = json.loads((d / DEFINITION).read_text(encoding="utf-8"))
        groups = json.loads((d / GROUPS).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{CHECK_DIR}: a JSON file does not parse: {e}"]
    axes = set(task.shapes[0].axes)
    out += _definition_problems(defn, axes)
    src = (d / JUDGE).read_text(encoding="utf-8")
    inputs = list((defn.get("inputs") or {}) if isinstance(defn, dict) else {})
    out += _judge_problems(src, inputs)
    out += _group_problems(groups, task)
    if not out:
        out += _file_group_problems(task, defn)
    return out


def _definition_problems(defn: object, axes: set) -> list[str]:
    out = []
    if not isinstance(defn, dict) or set(defn) - {"inputs", "outputs", "description"}:
        return [f"{DEFINITION} must be {{\"inputs\": ..., \"outputs\": ...}}"]
    for side in ("inputs", "outputs"):
        specs = defn.get(side)
        if not isinstance(specs, dict) or not specs:
            out.append(f"{DEFINITION}: {side} must be a non-empty object")
            continue
        for name, spec in specs.items():
            where = f"{DEFINITION} {side}.{name}"
            if not _IDENT.match(name):
                out.append(f"{where}: not a Python identifier")
            if not isinstance(spec, dict):
                out.append(f"{where}: must be an object with shape and dtype")
                continue
            if spec.get("dtype") not in DTYPES:
                out.append(f"{where}: dtype must be one of {list(DTYPES)}, got {spec.get('dtype')!r}")
            shape = spec.get("shape")
            if shape is None and side == "inputs":
                continue          # a python scalar input
            if not isinstance(shape, list):
                out.append(f"{where}: shape must be a list of axis names / integers")
                continue
            for dim in shape:
                if isinstance(dim, int) and not isinstance(dim, bool) and dim > 0:
                    continue
                if isinstance(dim, str) and dim in axes:
                    continue
                out.append(f"{where}: dimension {dim!r} is neither a positive integer nor one of "
                           f"the task's axes {sorted(axes)}")
    if isinstance(defn.get("inputs"), dict) and isinstance(defn.get("outputs"), dict):
        clash = set(defn["inputs"]) & set(defn["outputs"])
        if clash:
            out.append(f"{DEFINITION}: names used for both an input and an output: {sorted(clash)}")
    return out


def _judge_problems(src: str, inputs: list[str]) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"{JUDGE} does not parse: {e}"]
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    out = []
    if "run" not in fns:
        out.append(f"{JUDGE} defines no top-level run(...)")
    else:
        args = [a.arg for a in fns["run"].args.args]
        if args != inputs:
            out.append(f"{JUDGE}: run{tuple(args)} must take the definition's inputs in order "
                       f"{tuple(inputs)}")
    if "make_inputs" not in fns:
        out.append(f"{JUDGE} defines no top-level make_inputs(axes, group, seed, device)")
    else:
        args = [a.arg for a in fns["make_inputs"].args.args]
        if args[:4] != ["axes", "group", "seed", "device"]:
            out.append(f"{JUDGE}: make_inputs must be make_inputs(axes, group, seed, device), "
                       f"got make_inputs({', '.join(args)})")
    return out


def _group_problems(groups: object, task) -> list[str]:
    if not isinstance(groups, dict):
        return [f"{GROUPS} must map shape names to lists of input groups"]
    out = []
    known = {s.name: s for s in task.shapes}
    for name in groups:
        if name not in known:
            out.append(f"{GROUPS}: {name!r} is not one of the task's shapes {sorted(known)}")
    for s in task.shapes:
        rows = groups.get(s.name, [])
        if not isinstance(rows, list):
            out.append(f"{GROUPS}[{s.name}] must be a list")
            continue
        taken = {g.name for g in s.inputs}
        for j, row in enumerate(rows):
            where = f"{GROUPS}[{s.name}][{j}]"
            if not isinstance(row, dict) or not _GROUP.match(str(row.get("name", ""))):
                out.append(f"{where}: needs a short \"name\"")
                continue
            if row["name"] in taken:
                out.append(f"{where}: input group name {row['name']!r} is already used on this shape")
            taken.add(row["name"])
            if not str(row.get("intent", "")).strip():
                out.append(f"{where} ({row['name']}): say what the group covers in \"intent\"")
            if "seed" in row and not (isinstance(row["seed"], int) and not isinstance(row["seed"], bool)):
                out.append(f"{where} ({row['name']}): seed must be an integer")
        if len(taken) < MIN_INPUT_GROUPS:
            out.append(f"{GROUPS}[{s.name}]: the shape has {len(taken)} input group(s); it needs "
                       f"at least {MIN_INPUT_GROUPS} (the task gives {len(s.inputs)})")
    return out


def safetensors_header(path: Path) -> dict:
    """name -> {"dtype", "shape"} of a safetensors file, read from its header (no torch)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        if n <= 0 or n > 100_000_000:
            raise ValueError("not a safetensors file")
        head = json.loads(fh.read(n))
    return {k: {"dtype": _ST_DTYPES.get(v.get("dtype"), v.get("dtype")), "shape": v.get("shape")}
            for k, v in head.items() if k != "__metadata__"}


def _file_group_problems(task, defn: dict) -> list[str]:
    """The task's own input files against the teacher's signature."""
    out = []
    for s in task.shapes:
        dims = dict(s.axes)
        for g in s.inputs:
            where = f"shape {s.name} input group {g.name} ({g.file.name})"
            try:
                head = safetensors_header(g.file)
            except (OSError, ValueError, struct.error) as e:
                out.append(f"{where}: {e}")
                continue
            for name, spec in defn["inputs"].items():
                if spec.get("shape") is None:
                    continue
                want = [d if isinstance(d, int) else dims[d] for d in spec["shape"]]
                got = head.get(name)
                if got is None:
                    out.append(f"{where}: has no tensor {name!r}")
                elif got["shape"] != want or got["dtype"] != spec["dtype"]:
                    out.append(f"{where}: {name} is {got['shape']} {got['dtype']}, the signature "
                               f"says {want} {spec['dtype']}")
    return out


def judge_prompt(task, check_dir: Path, *, reference_source: Path | None,
                 feedback: str = "") -> str:
    """The teacher's brief for writing (or fixing) the judge."""
    shapes = "\n".join(
        f"  - {s.name}: axes {json.dumps(s.axes)}; input groups given by the task: "
        + (", ".join(g.name for g in s.inputs) or "none")
        + f"; add {s.groups_to_add}" for s in task.shapes)
    if reference_source is not None:
        ref = (f"The task's reference implementation is at {reference_source}. Read it: the judge "
               f"must compute what it computes (the same outputs, in the same order and dtypes), "
               f"stated plainly in PyTorch. Do not copy its optimizations.")
    else:
        ref = ("There is no reference implementation to read. Write the judge from the "
               "description alone; an implementation written separately will be held to it.")
    fix = f"\n\nYour previous files were rejected:\n{feedback}\nFix them.\n" if feedback else ""
    return f"""You are writing the correctness judge for the operator task "{task.name}".

The task's description:

<description>
{task.description.strip()}
</description>

Shapes (every shape is one assignment of the axes; one kernel will serve all of them):
{shapes}

{ref}

Write exactly three files into {check_dir} (create nothing else):

1. {DEFINITION}: {{"inputs": {{name: {{"shape": [...], "dtype": ...}}}}, "outputs": {{...}}}}.
   Dimensions are the task's axis names ({", ".join(task.shapes[0].axes)}) or positive
   integers; dtypes are one of {", ".join(DTYPES)}. List the inputs in the order run() takes
   them. A python scalar input has "shape": null.

2. {JUDGE}: a module with
   - run(<inputs, same names and order>) returning the outputs (one tensor, or a tuple in the
     order of "outputs"), in the declared dtypes. Compute in higher precision inside if that
     states the operator more faithfully. It is the definition of correct: clarity over speed,
     no custom kernels, nothing but torch.
   - make_inputs(axes, group, seed, device) returning {{input name: tensor}} for an input group
     you define (below), at the shape's axes, deterministic in seed, honoring every constraint
     the description puts on legal inputs (index ranges, masks, monotonic tensors, ...).

3. {GROUPS}: {{shape name: [{{"name": ..., "intent": ..., "seed": ...}}]}} -- for each shape, the
   input groups you add so it has at least three in total. Pick groups that are representative
   AND different from each other: a typical distribution, extremes inside the legal range, and
   the structure that matters for this operator (sparsity, masking, skewed lengths, outliers,
   ...). Say in "intent" what each one covers. Use names not taken by the task's own groups.

The judge is frozen once it agrees with the reference, and every kernel the run writes is
held to it: make the signature complete and the semantics exact.{fix}
"""
