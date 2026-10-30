#!/bin/bash
# blackbox-guard.sh — the closed-source run's in-job tripwire, as ONE shared copy.
#
# A blackbox run treats its reference as a binary: it may be run, timed, profiled and
# disassembled, but its SOURCE is off limits for teacher and student alike. The
# reference tree's `gpu_entry.sh cmd` is the channel the teacher drives during
# reconnaissance, so that is where the tripwire sits — it refuses a probe that names
# the sealed blob, the key, the unsealing machinery, the temp tree the JIT compiles
# from, source introspection on the reference module, the upstream repositories, or
# NCU's line-resolved source. SASS stays open: that IS black-box reconnaissance.
#
# Why this file exists: hand-copied guard logic in each gpu_entry.sh inherits the
# copies' bugs, and a broken copy can burn a run's whole reconnaissance budget. One
# copy, one test suite, one place to fix.
#
# How it reaches the node: a job only ever sees `git archive` of the reference tree
# itself — this repository's scripts/ is NOT there. The driver therefore seeds this
# file into the reference tree's notes/ (beside sass_tools.py and the other vendored
# tools) on a blackbox run and commits it there; gpu_entry.sh sources THAT copy and
# carries no guard logic of its own.
#
# Usage, from a reference tree's gpu_entry.sh (after W is set):
#
#     BLACKBOX_EXTRA='myref_impl|myref_runtime'    # reference-specific names, optional
#     source "$W/notes/blackbox-guard.sh"
#     ...
#     cmd) blackbox_guard "$*"; echo "[task] blackbox guard: passed"
#          run_bounded bash -c "$*" ;;
#
# `blackbox_guard <cmd-string>` scans the string AND every notes/*.py|*.sh under $W
# (the teacher's probe scripts ride there). A refusal prints BLACKBOX-GUARD REFUSED
# with the matched tokens and exits 3.
#
# BLACKBOX_EXTRA must not contain a substring of the reference KERNEL'S SYMBOL NAME.
# That symbol is public — NCU prints it, the harness writes it into ref_baseline.json
# and the reconnaissance prompt, and a legitimate probe filters on it (`-k`,
# `--kernel-name`). Listing `attn_core` for a reference whose main kernel is
# `kernel_cutlass_kernel_attn_coreForwardChunked` means the first probe that
# filters on that name refuses, and because notes/ is scanned cumulatively, that
# ONE resident file refuses every later round too. Module and blob names are
# the right EXTRA tokens; anything that also occurs in a mangled kernel name is not.
#
# BLACKBOX_EXTRA must not name the reference TREE'S OWN measurement wrapper either —
# the `<tree>/<name>.py` that the tree's README tells probes to import for
# make_inputs() / ref_runner(). That wrapper is the sanctioned handle on the
# reference, not the reference: listing it refuses every probe that does the sanctioned
# thing, and pushes the teacher to route around the scan by putting a wrapper outside
# notes/. The right tokens are the sealed blob, the unsealed entry and
# implementation module names (fkref_* is already in the base list). A collision is
# reported when this file is sourced, before any probe runs.

# What a task's own reference is called is task-specific; set BLACKBOX_EXTRA to add
# its module/blob names. Everything below is common to every sealed reference.
#
# NOT on this list, deliberately: tempfile / mkdtemp / tarfile. The unsealer uses
# them, but they are stdlib and a profiling probe legitimately needs a temp file for
# its .ncu-rep — banning them refuses ordinary NCU probes for the line
# `import tempfile` while protecting nothing, since what actually guards the
# unsealed tree is its `fkref` prefix plus the blob, the key, the module object and
# the source-introspection helpers, all still listed here.
BLACKBOX_FORBID_BASE='closed/|\.enc\b|_REF_KEY|keystream|unseal|decrypt|fkref|closed_ref|ref_payload|_REF_DIR|_REF_PATH|_SRC_PATH|_module\(\)|linecache|getsource|getsourcefile|inspect\.getfile|inspect\.getmodule|sys\.modules\[|_COMPILE_CACHE|py_compile|\bdis\.|print-source[= ]+(cuda|all|sass,cuda|cuda,sass)|import-source|--source-folders|find / .*-name|find /tmp|ls .*/tmp|mit-han-lab|kernel-design-agents|NVlabs|mlsys2026|contest-solution|flashinfer-bench|\bstrings\b|xxd|hexdump|base64 -d'

# This file, as it is being sourced. The driver seeds it into notes/, and notes/ is
# what blackbox_guard scans — while this file's own comments and forbid list carry
# every forbidden token, so scanning it would refuse EVERY `cmd` before a single probe
# ran (unit tests that source it from scripts/ cannot show the self-hit). blackbox_guard skips a notes/ file only when it is this
# guard byte for byte, so a probe cannot hide under the guard's file name.
BLACKBOX_GUARD_SELF="${BASH_SOURCE[0]}"

# An EXTRA token that names the tree's own measurement wrapper (a <tree>/*.py module
# probes are told to import) refuses every sanctioned probe — see the header. Say so
# at source time, in every job log, instead of letting the first refusal find out.
blackbox_extra_collisions() {
    [ -n "${BLACKBOX_EXTRA:-}" ] || return 0
    local f m out=""
    for f in "${W:-$PWD}"/*.py; do
        [ -f "$f" ] || continue
        m="$(basename "$f" .py)"
        printf '%s' "$m" | grep -qE "$BLACKBOX_EXTRA" && out="${out:+$out }$m"
    done
    printf '%s' "$out"
}
_bb_collide="$(blackbox_extra_collisions)"
if [ -n "$_bb_collide" ]; then
    echo "BLACKBOX-GUARD WARNING: BLACKBOX_EXTRA matches this tree's own wrapper module(s): ${_bb_collide} — every probe that imports it for make_inputs()/ref_runner() will be REFUSED. Remove it from BLACKBOX_EXTRA; list the sealed blob / unsealed entry / implementation names instead."
fi
unset _bb_collide

blackbox_forbid_pattern() {
    if [ -n "${BLACKBOX_EXTRA:-}" ]; then
        printf '%s|%s' "$BLACKBOX_FORBID_BASE" "$BLACKBOX_EXTRA"
    else
        printf '%s' "$BLACKBOX_FORBID_BASE"
    fi
}

# A refusal MUST be loud. With an unquoted `notes/*.sh` glob and no .sh files the glob
# stays literal, `cat` on it returns non-zero, and under the caller's `set -euo pipefail`
# the job dies at the aggregation line BEFORE printing why — a bare `rc=1` the teacher
# cannot tell from its own probe crashing. Hence: the file list is built by testing each
# candidate, and every aggregation pipeline ends in `|| true`.
blackbox_guard() {
    local cmd="${1:-}" why="" hit hits f
    local pattern; pattern="$(blackbox_forbid_pattern)"
    local dir="${BLACKBOX_NOTES_DIR:-${W:-$PWD}/notes}"

    local files=()
    for f in "$dir"/*.py "$dir"/*.sh; do
        [ -f "$f" ] || continue
        # The running guard itself, seeded into notes/ (see BLACKBOX_GUARD_SELF). Only an
        # exact copy is skipped: a different file under the same name is scanned.
        if [ -n "${BLACKBOX_GUARD_SELF:-}" ] && cmp -s "$f" "$BLACKBOX_GUARD_SELF"; then
            continue
        fi
        files+=("$f")
    done

    if printf '%s' "$cmd" | grep -qE "$pattern"; then why="cmd string"; fi
    if [ "${#files[@]}" -gt 0 ]; then
        hit=$(grep -lE "$pattern" "${files[@]}" 2>/dev/null | head -3 | tr '\n' ' ' || true)
        [ -n "$hit" ] && why="${why:+$why; }notes: $hit"
    fi
    [ -n "$why" ] || return 0

    hits=$({ printf '%s\n' "$cmd"; [ "${#files[@]}" -gt 0 ] && cat "${files[@]}"; } 2>/dev/null \
           | grep -oE "$pattern" | sort | uniq -c | sort -rn | head -6 | tr '\n' ' ' || true)
    # A reference-specific token that also occurs in the reference's kernel symbol refuses
    # every probe that filters on that kernel (see the header). Say so when the hit
    # came from BLACKBOX_EXTRA, so the task author fixes the list rather than the probe.
    local extra_hint=""
    if [ -n "${BLACKBOX_EXTRA:-}" ] && printf '%s' "$hits" | grep -qE "$BLACKBOX_EXTRA"; then
        extra_hint=" [reference-specific token from BLACKBOX_EXTRA — if it is part of the reference kernel's symbol name (NCU -k filter, ref_baseline.json), it does not belong in BLACKBOX_EXTRA; a refusal that names a notes/ file you did not write this round means a resident earlier probe carries the token — delete or rewrite that file]"
        local collide; collide="$(blackbox_extra_collisions)"
        if [ -n "$collide" ]; then
            extra_hint="${extra_hint} [BLACKBOX_EXTRA also matches this tree's own wrapper module(s): ${collide} — importing it is the sanctioned way to drive the reference, so this refusal is a run configuration error, not a probe error: report it, do not work around it]"
        fi
    fi
    echo "BLACKBOX-GUARD REFUSED ($why; matched: $hits): this run treats the reference as a closed binary — do not read, locate or unseal the reference kernel source (closed/, the key, the temp tree it is compiled from, the unsealed module object, its module names, getsource / linecache / sys.modules on it), do not fetch its upstream repository, and do not ask NCU for line-resolved source (--print-source sass only; cuobjdump -sass on the JIT cubin is fine). Drive the reference only through the tree's ref runner (make_inputs / ref_runner / bench / ncu). Introspecting the CuTe DSL package itself (cutlass.__file__, importlib.metadata, its .py sources) is fine. Rewrite the probe without the matched tokens and resend.${extra_hint}"
    exit 3
}
