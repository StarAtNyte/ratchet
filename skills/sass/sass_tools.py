"""SASS extraction that fails loudly instead of succeeding quietly.

Every helper here exists because a probe once did the opposite and a run paid
for it (observed in a blackbox attn_fwd reconnaissance):

* probe1 picked the right kernel but parsed with a format guessed from memory
  (cuobjdump's ``/*0028*/`` comments); the node's NCU emits a bare hex address
  column, so 2615 correct lines yielded 0 opcodes.
* probe2 fixed the parser but took ``kernels[0]`` — the first *launched* kernel,
  which is ``torch.randn``'s RNG — and "perfectly" histogrammed the wrong kernel.
  Its all-ABSENT decisive-opcode table was self-consistent and unfalsifiable by eye.

Hence the two design rules of this module:

1. **Parsing is format-agnostic.** Both observed dump styles are recognised, and
   the mnemonic — not the line shape — is what identifies an instruction. Opcodes
   keep their full dotted form (``HGMMA.64x128x16.F32.BF16.BF16``): the MMA shape
   lives after the first dot, truncating there discards the payload.
2. **Silence is failure.** ``analyze`` ends in exactly one of two machine-readable
   verdict lines, ``SASS-OK ...`` or ``SASS-FAIL ...``, and the driver's finish
   gate looks for the former. No verdict line is printed unless the self-checks
   (right kernel, plausible instruction count, expected ISA families, no swallowed
   profiler errors) all pass, and any failure dumps a raw sample so the next turn
   debugs the evidence rather than the summary.

Stdlib only: this file is copied verbatim into the recon tree and imported by
probe scripts next to it (``python3 notes/<probe>.py`` puts ``notes/`` on
``sys.path``).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

# A real sm90 tensor kernel disassembles to thousands of lines; the RNG kernel that
# probe2 mistook for one had 400. The floor sits between those clusters.
MIN_INSTRUCTIONS = 500

# Families whose presence separates a Hopper tensor kernel from bystanders. BAR is
# deliberately absent: every kernel has BAR.SYNC (the RNG kernel had exactly one),
# so it identifies nothing.
SM90_TENSOR_FAMILIES = (
    "HGMMA", "QGMMA", "IGMMA", "BGMMA",
    "UTMALDG", "UTMASTG", "UTMACCTL",
    "USETMAXREG", "WARPGROUP",
)

# Lines the profiler injects into its own output stream. Counting these as opcodes
# is how a probe once reported ``opcodes={ERROR: 2}`` — the two "instructions" were
# ``==ERROR== Could not open report file``.
_TOOL_NOISE = re.compile(r"==(ERROR|WARNING)==|Could not open|^\s*$")

# cuobjdump style:   /*0028*/  IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;  /* 0x... */
_CUOBJDUMP = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*)$")
# ncu source-page style:   0x7fc07f7a1800           LDC R1, c[0x0][0x28]
_NCU_ADDR = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+(\S.*)$")
# The instruction body shared by both: optional (uniform) predicate, then a
# dotted mnemonic, then operands. The tail class is case-insensitive because MMA
# shape segments spell their separators in lowercase (``HGMMA.64x128x16.F32``).
_BODY = re.compile(r"^(@!?U?P(?:T|\d+)\s+)?([A-Z][A-Za-z0-9_.]{2,})\b\s*(.*)$")


class Instr(NamedTuple):
    addr: Optional[str]
    opcode: str          # full dotted mnemonic, never truncated
    operands: str


class ParseResult(NamedTuple):
    instructions: List[Instr]
    errors: List[str]    # profiler noise that must not masquerade as instructions
    unparsed: int        # non-empty lines that matched no pattern (headers, banners)


def parse_sass(text: str) -> ParseResult:
    """Disassembly text -> instructions, whatever tool rendered it.

    A line is an instruction if it carries an address in either known style and its
    body starts with a mnemonic. Lines without an address are headers or banners
    and are counted, not guessed at: a parser that guesses is how format drift
    turns into silent zero.
    """
    instructions: List[Instr] = []
    errors: List[str] = []
    unparsed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if _TOOL_NOISE.search(line):
            errors.append(line.strip())
            continue
        m = _CUOBJDUMP.match(line) or _NCU_ADDR.match(line)
        if not m:
            unparsed += 1
            continue
        addr, body = m.group(1), m.group(2)
        # cuobjdump suffixes the encoding as a second /* ... */ comment; drop it
        # before the body regex sees it.
        body = re.sub(r"/\*.*?\*/", "", body).strip().rstrip(";").strip()
        bm = _BODY.match(body)
        if not bm:
            unparsed += 1
            continue
        instructions.append(Instr(addr, bm.group(2), bm.group(3).strip()))
    return ParseResult(instructions, errors, unparsed)


def opcode_histogram(instructions: Iterable[Instr]) -> "Counter[str]":
    return Counter(i.opcode for i in instructions)


def family_counts(instructions: Iterable[Instr],
                  families: Sequence[str] = SM90_TENSOR_FAMILIES) -> "Counter[str]":
    counts: "Counter[str]" = Counter()
    for i in instructions:
        root = i.opcode.split(".", 1)[0]
        if root in families:
            counts[root] += 1
    return counts


def pick_kernel(pairs: Sequence[Tuple[str, float]]) -> Tuple[str, float]:
    """The kernel under study is the one that dominates *time*, never the one that
    launched first — the first launch is routinely ``torch.randn``'s RNG kernel.

    ``pairs`` is ``[(kernel_name, duration), ...]`` in any order.
    """
    if not pairs:
        raise ValueError("pick_kernel: no kernels to choose from")
    ranked = sorted(pairs, key=lambda p: -float(p[1]))
    print("kernel ranking by duration:")
    for name, dur in ranked[:5]:
        print(f"  {dur:>14.0f}  {name[:110]}")
    return ranked[0][0], float(ranked[0][1])


def self_check(profiled_kernel: str,
               expected_kernel: str,
               parsed: ParseResult,
               min_instructions: int = MIN_INSTRUCTIONS,
               families: Sequence[str] = SM90_TENSOR_FAMILIES) -> List[str]:
    """Everything that must hold before a histogram may be believed.

    Returns the list of violated expectations; empty means trustworthy. Each check
    is one of the §5 failure modes, mechanised.
    """
    problems: List[str] = []
    if _short(profiled_kernel) != _short(expected_kernel):
        problems.append(
            f"profiled kernel {_short(profiled_kernel)!r} is not the top-duration "
            f"kernel {_short(expected_kernel)!r} — you histogrammed a bystander")
    if parsed.errors:
        problems.append(
            f"{len(parsed.errors)} profiler error/warning line(s) in the stream, "
            f"first: {parsed.errors[0]!r}")
    if not parsed.instructions:
        problems.append(
            "parsed 0 instructions — the parser, not the kernel, is the story")
    elif len(parsed.instructions) < min_instructions:
        problems.append(
            f"only {len(parsed.instructions)} instructions (< {min_instructions}) — "
            f"a real tensor kernel is thousands; this smells like a helper kernel")
    if parsed.instructions and families and not family_counts(parsed.instructions,
                                                              families):
        problems.append(
            "none of " + "/".join(families[:4]) + "/... present — either this is "
            "not the tensor kernel or the dump is truncated")
    return problems


def dump_raw(text: str, head: int = 30, tail: int = 30) -> None:
    """Raw evidence beats a wrong summary: one report's §1.2 was rebuilt
    entirely from the 50 raw lines a failing probe had the discipline to print."""
    lines = text.splitlines()
    print(f"--- raw dump ({len(lines)} lines total) ---")
    for line in lines[:head]:
        print(line)
    if len(lines) > head + tail:
        print(f"... [{len(lines) - head - tail} lines omitted] ...")
    for line in lines[max(head, len(lines) - tail):]:
        print(line)
    print("--- end raw dump ---")


def analyze(sass_text: str,
            profiled_kernel: str,
            launch_pairs: Sequence[Tuple[str, float]],
            min_instructions: int = MIN_INSTRUCTIONS,
            families: Sequence[str] = SM90_TENSOR_FAMILIES,
            top: int = 40) -> bool:
    """Parse, histogram, self-check, and speak the verdict line the driver gates on.

    Print-only by design: the job's stdout is the only channel that returns from
    the node, so everything of value must go through it, aggregated.
    """
    expected, _ = pick_kernel(launch_pairs)
    parsed = parse_sass(sass_text)
    print(f"parsed: {len(parsed.instructions)} instructions, "
          f"{parsed.unparsed} unparsed lines, {len(parsed.errors)} tool-noise lines")
    if parsed.instructions:
        hist = opcode_histogram(parsed.instructions)
        print(f"opcode histogram (top {top} of {len(hist)}):")
        for op, n in hist.most_common(top):
            print(f"  {n:>6}  {op}")
        fams = family_counts(parsed.instructions, families)
        print("decisive families:", dict(fams) if fams else "NONE")
    problems = self_check(profiled_kernel, expected, parsed,
                          min_instructions=min_instructions, families=families)
    if problems:
        for p in problems:
            print(f"self-check FAILED: {p}")
        dump_raw(sass_text)
        print(f"SASS-FAIL {'; '.join(problems)}")
        return False
    # Spaces stripped from the kernel field: a demangled C++ template name
    # carries them ("void foo<Bar<512, 1>, ...") and a space inside the field
    # broke the driver's full-field acceptance pattern once already (in a
    # sparse-attention recon). The driver side now tolerates spaces too; keeping the
    # field a single token protects every other consumer that splits on
    # whitespace.
    print(f"SASS-OK kernel={'_'.join(_short(profiled_kernel).split())} "
          f"instructions={len(parsed.instructions)} "
          f"opcodes={len(opcode_histogram(parsed.instructions))}")
    return True


def _short(name: str) -> str:
    """Kernel names from different sources differ in mangling decoration and get
    truncated around 180 chars; compare on a prefix long enough to distinguish
    siblings and short enough to survive both."""
    return name.strip()[:80]


def main(argv: Sequence[str]) -> int:
    """Standalone: ``python3 sass_tools.py <sass_file> <profiled_kernel_name>``.

    The launch ranking degenerates to the profiled kernel itself, so only the
    format/floor/family checks apply — enough for offline re-parsing of a dump.
    """
    if len(argv) != 3:
        print("usage: sass_tools.py <sass_file> <profiled_kernel_name>")
        return 2
    with open(argv[1], encoding="utf-8", errors="replace") as f:
        text = f.read()
    ok = analyze(text, argv[2], [(argv[2], 1.0)])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
