#!/usr/bin/env python3
"""PreToolUse hook: refuse a tool call that reaches the network from an offline role, or
that names the reference implementation's repository (see
src/forgekernel/loop/agent_policy.py).

Installed by the harness into an agent's project settings; invoked by Claude Code as

    python3 agent_policy_hook.py <workspace>/.claude/forgekernel-agent-policy.json

with the tool call as JSON on stdin. Exit 2 blocks the call and hands stderr to the
agent; exit 0 lets it through. Standard library only: it runs under whatever
interpreter the harness had, before any tool does.

This is a **tripwire, not a boundary**. What actually holds is the Bash sandbox (no
outbound connection at all for the student, no read of a local reference tree for
either role) and the deny rules. The hook adds the two things those cannot express:

- a message that names the rule when an offline turn reaches for curl / pip / git
  clone, instead of an opaque sandbox error;
- the web side of "no reference source" for a closed-source teacher, which keeps the
  network on purpose. That side can only be matched on what the call spells out, so a
  fetch that never names the repository -- a shortener, a vendored mirror, a script
  that builds the URL at runtime -- walks past it.

Reference matching normalises ``:`` to ``/`` and lowercases, then looks for the
declared fragment and for its ``org/repo`` pair, so the forms that serve the same
source all match:

    github.com/org/RefLib            raw.githubusercontent.com/org/reflib/main/k.cuh
    api.github.com/repos/org/reflib  cdn.jsdelivr.net/gh/org/reflib@main/k.cuh
    git@github.com:org/RefLib.git    gh repo clone org/RefLib

A package manager asked for the repository by name (``pip download ref-lib``) matches
too: the repo name with ``-``, ``_`` and ``.`` removed, against a command that names a
fetching tool.
"""

from __future__ import annotations

import json
import re
import sys

# Commands that reach the network. The offline check refuses these outright; the
# reference check uses them to decide whether a bare package name is a fetch.
# A quote counts as a word start so the name is still seen inside a wrapper --
# `bash -lc 'ssh ...'`, `tmux new -d 'ssh ...'`. Only a boundary is required after the
# name, so a file that merely starts with it (`ssh_helper.py`) is not matched.
_NET_CMD = re.compile(r"""(?:^|[\s;&|(`$'"])(?:
      curl | wget | aria2c | scp | sftp | rsync | ssh | ncat | telnet | ftp
    | nc\s+-\S*
    | git\s+(?:clone|fetch|pull|ls-remote|remote)
    | (?:pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+(?:install|download)
    | uv\s+(?:add|sync|tool\s+install)
    | (?:npm|pnpm|yarn|bun)\s+(?:install|add|i)\b
    | npx
    | gh\s+(?:repo|api|release|gist)
    | (?:huggingface-cli|hf)\s+download
    | conda\s+install | poetry\s+add | cargo\s+(?:add|fetch) | brew\s+install
    )(?:\s|$)""", re.I | re.X)
_NET_URL = re.compile(r"\b(?:https?|ftps?|git|ssh|rsync)://", re.I)
# The ways to the GPU machine, for a student that keeps the network (no_cluster without
# offline): also by path (`/usr/bin/ssh`), which the deny rules do not match.
_CLUSTER_CMD = re.compile(
    r"""(?:^|[\s;&|(`$'"/])(?:ssh|scp|rsync|python3?\s+-m\s+forgekernel\.submit)(?:\s|$)""",
    re.I)


def _texts(tool_name: str, tool_input: dict) -> list[str]:
    if tool_name == "WebFetch":
        return [str(tool_input.get("url", ""))]
    if tool_name == "WebSearch":
        return [str(tool_input.get("query", ""))]
    if tool_name == "Bash":
        return [str(tool_input.get("command", ""))]
    return []


def _norm(text: str) -> str:
    """Lowercased, with ``:`` read as a path separator so ``git@host:org/repo`` and
    ``https://host/org/repo`` normalise to the same thing."""
    return text.lower().replace(":", "/")


def _squash(text: str) -> str:
    return re.sub(r"[-_.]", "", text)


def _repo_pair(pattern: str) -> str | None:
    """The ``org/repo`` pair a declared fragment names, when it names one."""
    parts = [p for p in _norm(pattern).split("/") if p]
    if len(parts) >= 3 and "." in parts[0]:   # host / org / repo ...
        return "/".join(parts[1:3])
    if len(parts) == 2:                       # already org/repo
        return "/".join(parts)
    return None


def _reference_hit(patterns: list[str], tool: str, text: str) -> str | None:
    """The fragment this call names, or None."""
    low = _norm(text)
    squashed = _squash(low)
    fetches = tool != "Bash" or bool(_NET_CMD.search(text))
    for pat in patterns:
        norm = _norm(pat)
        if norm and norm in low:
            return pat
        pair = _repo_pair(pat)
        if pair and pair in low:
            return pat
        # `pip download ref-lib` / `uv add ref_lib`: the repository asked for by name.
        repo = _squash((pair or norm).rsplit("/", 1)[-1])
        if fetches and len(repo) >= 5 and repo in squashed:
            return pat
    return None


def decide(policy: dict, event: dict) -> str | None:
    """The refusal message for this tool call, or None to allow it."""
    tool = event.get("tool_name", "")
    texts = _texts(tool, event.get("tool_input") or {})
    role = policy.get("role")

    if policy.get("offline"):
        if tool in ("WebFetch", "WebSearch"):
            return (f"BLOCKED by the ForgeKernel agent policy: the {role} works offline "
                    f"in this run, so {tool} is not available. Interface facts come from "
                    f"the KB, the DSL sources on this machine and your own probes.")
        for text in texts:
            if _NET_CMD.search(text) or _NET_URL.search(text):
                return (f"BLOCKED by the ForgeKernel agent policy: the {role} works "
                        f"offline in this run and this command reaches the network. "
                        f"Nothing outside this machine is available this turn — the "
                        f"sandbox refuses the connection as well.")
    if (policy.get("offline") or policy.get("no_cluster")) and tool == "Bash":
        for text in texts:
            if _CLUSTER_CMD.search(text):
                return (f"BLOCKED by the ForgeKernel agent policy: the {role} does not drive "
                        f"the GPU machine (ssh / its own jobs). GPU work goes through the "
                        f"evaluation command your task names; the loop submits and records it.")

    patterns = [p for p in policy.get("deny_urls", []) if p]
    if patterns:
        for text in texts:
            hit = _reference_hit(patterns, tool, text)
            if hit:
                return (f"BLOCKED by the ForgeKernel agent policy: this {tool} call names "
                        f"`{hit}`, the reference implementation's repository. The reference "
                        f"source is off limits in this run (role {role}, "
                        f"{policy.get('oracle_mode')} mode) — on this machine and on the web.")
    return None


def main(argv: list[str]) -> int:
    try:
        policy = json.loads(open(argv[1], encoding="utf-8").read())
    except (IndexError, OSError, ValueError) as e:
        # A policy that cannot be read must not silently allow everything.
        sys.stderr.write(f"BLOCKED: ForgeKernel agent policy unreadable ({e})\n")
        return 2
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    msg = decide(policy, event)
    if msg:
        sys.stderr.write(msg + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
