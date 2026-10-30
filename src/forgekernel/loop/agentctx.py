"""Context-window hygiene for headless claude-CLI agents (``--agent-cmd``).

The claude CLI decides its context budget client-side from the model id. Behind
a gateway (``ANTHROPIC_BASE_URL`` set) it cannot verify a 1M window, so a *bare*
claude id (``claude-opus-5``, ``opus``, ...) is budgeted at 200k and the session
is auto-compacted at ~150-170k tokens: the whole transcript is replaced by a
summary the model writes about itself, and the turn carries on from that summary.
``<model>[1m]`` declares the window explicitly; ``CLAUDE_CODE_AUTO_COMPACT_WINDOW``
raises the threshold. The clamp is the CLI's client-side default, not the gateway's
limit — the same gateway passes 350k+ contexts for ``fable[1m]`` without incident.

The launch gate is per *leg*, where a leg is one concrete model id a headless
claude invocation may send, together with the window/gateway context of the
invocation that sends it. ``--agent-cmd`` is often a wrapper whose branches are
mutually exclusive (``case "$ROLE"`` in a fallback wrapper): the teacher
branch may declare ``[1m]`` while the student branch sends a bare id. A single
global window cannot be tied to a leg, so a teacher's 1M would clear the
student's finding. Because the reader is static it cannot know which branch
runs, so a base-model flag variable resolves to the *set* of every value it is
assigned and each value becomes its own leg — a risky value is a finding even
if the script also has a safe one (fail-safe: refuse rather than let a silent
auto-compaction run and only be discoverable afterwards).

No leg needs fixing when it is first-party: the official endpoint (`unset` /
``env -u ANTHROPIC_BASE_URL``) grants the 1M window, so those are not findings.

``inspect_agent_cmd`` builds the leg table; ``window_findings`` reads it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

AUTO_COMPACT_WINDOW_ENV = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
GATEWAY_ENV = "ANTHROPIC_BASE_URL"
MODEL_ENV = "ANTHROPIC_MODEL"

# A claude model id the CLI recognizes client-side: a tier alias or a full
# ``claude-<tier>-<ver>`` id. Channel aliases (``CLAUDE_ew682d``) and foreign
# ids are unknown-model cases, a different rule
# (CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT, see the doc).
_CLAUDE_ID = re.compile(r"^(?:claude-)?(?:opus|sonnet|fable|haiku)(?:[-.][0-9][0-9a-z.-]*)?$", re.I)
_ONE_M = re.compile(r"\[1m\]$", re.I)

# Flag variables a wrapper uses to hold the base model. ``ANTHROPIC_MODEL`` is
# deliberately absent: the ``\b`` before MODEL fails on the underscore, and that
# spelling is read from the environment / settings.json instead.
_FLAG_VARS = ("MODEL_FLAG", "STUDENT_FLAG", "TEACHER_FLAG")
_VAR_ASSIGN = re.compile(
    r"\b(?:export\s+)?(MODEL_FLAG|STUDENT_FLAG|TEACHER_FLAG)="
    r"(\"[^\"\n]*\"|'[^'\n]*'|[^\s;&|]+)")

# ``--model X`` / ``--model=X``. The value is a quoted or bare token, possibly a
# ``$VAR`` / ``${VAR}`` reference; the loop unquotes it and resolves the variable.
_USE = re.compile(r"--model(?:=|\s+)(\"[^\"]*\"|'[^']*'|[^\s;&|]+)")

# Generic ``VAR=value`` shell assignment, used to resolve config-dir and flag
# variables (including inside case branches).
_SH_ASSIGN = re.compile(r"\b(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(\"[^\"\n]*\"|'[^'\n]*'|[^\s;&|]+)")
_CONFIG_DIR_ASSIGN = re.compile(r"\bCLAUDE_CONFIG_DIR=(\"[^\"\n]*\"|'[^'\n]*'|[^\s;&|]+)")
_WINDOW_ASSIGN = re.compile(r"\bCLAUDE_CODE_AUTO_COMPACT_WINDOW=['\"]?([0-9_]+)")
_SCRIPT_TOKEN = re.compile(r"(?<![\w/.-])((?:~|\$HOME|/)?[\w./~$-]*\.(?:sh|bash|zsh))\b")

# ``unset ANTHROPIC_BASE_URL`` / ``env -u ANTHROPIC_BASE_URL`` and their
# variable forms make a leg first-party (official endpoint, 1M window, no clamp).
# `unset` governs every claude call after it in the source — the first-party
# recipe, where the wrapper strips the gateway vars once and the whole file runs
# first-party. `env -u` governs only the one statement it prefixes.
# ``unset`` takes a whitespace-separated name list (backslash continuations
# already joined by ``_join_continuations``); ANTHROPIC_BASE_URL may sit anywhere
# in it. ``env`` may carry several ``-u NAME`` before the command.
_UNSET_BASE = re.compile(
    r"\bunset\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)*ANTHROPIC_BASE_URL\b")
_ENV_U_BASE = re.compile(
    r"\benv\s+(?:-[ui]\s+[A-Za-z_][A-Za-z0-9_]*\s+|-[A-Za-z]+\s+)*-u\s+ANTHROPIC_BASE_URL\b")
_CONTINUATION = re.compile(r"\\\n\s*")

_STATEMENT_SPLIT = re.compile(r";|&&|\|\||\n")


@dataclass
class Leg:
    model: str                 # one concrete model id this leg may send
    source: str                # "cmd", the script path, "<config>/settings.json", or "environment"
    window: int | None         # the >200k budget governing this leg (None = unknown)
    on_gateway: bool           # ANTHROPIC_BASE_URL is set on this leg
    first_party: bool          # the leg's own shell unsets ANTHROPIC_BASE_URL
    config_dir: str | None = None  # the config dir this leg reads (settings.json)
    inline_window: int | None = None
    variable: str = ""         # flag var it resolved from ("" for a literal / env / settings id)

    @property
    def is_claude(self) -> bool:
        return bool(_CLAUDE_ID.match(_ONE_M.sub("", self.model)))

    @property
    def declares_1m(self) -> bool:
        return bool(_ONE_M.search(self.model))


@dataclass
class AgentCmdFacts:
    cmd: str
    scripts: list[str] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)
    config_dirs: list[str] = field(default_factory=list)
    gateway: list[str] = field(default_factory=list)  # where BASE_URL was seen
    unreadable: list[str] = field(default_factory=list)


def _unquote(v: str) -> str:
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _expand(value: str, assigns: Mapping[str, str], env: Mapping[str, str]) -> str:
    """``$NAME`` / ``${NAME}`` / ``~`` expansion against the script's own
    assignments first, then the environment. Unknown names are left alone."""
    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name in assigns:
            return assigns[name]
        return env.get(name, m.group(0))
    out = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", sub, value)
    if out.startswith("~"):
        out = env.get("HOME", os.path.expanduser("~")) + out[1:]
    return out


def _settings_env(config_dir: str) -> dict[str, str] | None:
    p = Path(config_dir) / "settings.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    env = data.get("env") if isinstance(data, dict) else None
    return {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}


def _window_value(env: Mapping[str, str] | None) -> int | None:
    if not env:
        return None
    raw = env.get(AUTO_COMPACT_WINDOW_ENV, "").replace("_", "")
    return int(raw) if raw.isdigit() else None


def _resolve_window(inline: int | None, config_dir: str | None,
                    env: Mapping[str, str]) -> int | None:
    """Window precedence for a leg: inline on its statement, else its config
    dir's settings, else the process environment."""
    if inline is not None:
        return inline
    if config_dir:
        w = _window_value(_settings_env(config_dir))
        if w is not None:
            return w
    return _window_value(env)


def _resolve_gateway(config_dir: str | None, env: Mapping[str, str],
                     first_party: bool, source_unset: bool) -> bool:
    """Whether a leg runs behind a gateway.

    A leg that names its own config dir is governed solely by that dir's
    settings.json — a source-level ``unset ANTHROPIC_BASE_URL`` does NOT reach
    it, because the wrapper's fallback re-injects the gateway via the config
    dir (e.g. a fallback wrapper: the first-party branch unsets the vars, the
    gateway branch sets CLAUDE_CONFIG_DIR to a dir whose env has BASE_URL).
    A leg with no config dir is governed by the process environment, unless the
    source unsets BASE_URL (then it is first-party)."""
    if config_dir:
        s = _settings_env(config_dir)
        if s is not None:
            return bool(s.get(GATEWAY_ENV))
    if first_party:
        return False
    return bool(env.get(GATEWAY_ENV))


def _join_continuations(text: str) -> str:
    """Fold ``\\<newline>`` so a statement split over several source lines
    (``CLAUDE_CONFIG_DIR=x \\`` / ``claude ...``) is read as one statement."""
    return _CONTINUATION.sub(" ", text)


def _statements(text: str) -> list[str]:
    return [s for s in _STATEMENT_SPLIT.split(_join_continuations(text)) if s.strip()]


def inspect_agent_cmd(cmd: str | None, env: Mapping[str, str] | None = None,
                      cwd: str | os.PathLike | None = None) -> AgentCmdFacts:
    """Build the leg table from an ``--agent-cmd`` string, the shell scripts it
    names (one level), the config dirs they select, and the process environment.
    ``env`` is what the agent inherits — pass ``os.environ`` for a real run,
    ``{}`` in tests."""
    env = dict(env or {})
    facts = AgentCmdFacts(cmd=cmd or "")
    if not cmd:
        return facts
    texts: list[tuple[str, str]] = [("cmd", cmd)]
    for m in _SCRIPT_TOKEN.finditer(cmd):
        raw = m.group(1)
        path = _expand(raw, {}, env)
        if not os.path.isabs(path) and cwd is not None:
            path = os.path.join(os.fspath(cwd), path)
        if os.path.isfile(path):
            try:
                texts.append((path, Path(path).read_text(encoding="utf-8", errors="replace")))
                facts.scripts.append(path)
            except OSError:
                facts.unreadable.append(path)

    # Assignment context per source (merged across branches: the reader cannot
    # select a case branch, so every assignment in the source is in scope). This
    # resolves config-dir and flag variables wherever they are used.
    source_assigns: dict[str, dict[str, str]] = {}
    source_flags: dict[str, dict[str, list[str]]] = {}
    source_unset_base: dict[str, bool] = {}
    for source, text in texts:
        assigns = {k: _unquote(v) for k, v in _SH_ASSIGN.findall(text)}
        for _ in range(2):
            assigns = {k: _expand(v, assigns, env) for k, v in assigns.items()}
        source_assigns[source] = assigns
        flags: dict[str, list[str]] = {}
        for vm in _VAR_ASSIGN.finditer(text):
            name, val = vm.group(1), _unquote(vm.group(2))
            resolved = _expand(val, assigns, env)
            if not resolved.startswith("$"):
                flags.setdefault(name, []).append(resolved)
        source_flags[source] = flags
        # `unset`/`env -u ANTHROPIC_BASE_URL` anywhere in the source makes every
        # claude call in it first-party (splitting on statements would miss the
        # unset, which governs everything after it).
        source_unset_base[source] = bool(_UNSET_BASE.search(_join_continuations(text)))

    # Statement-level scan: a leg is one model id sent by one claude invocation,
    # sharing that statement's inline window / config-dir / first-party flag.
    seen: set[tuple] = set()
    dirs_with_model_flag: set[str] = set()
    for source, text in texts:
        assigns = source_assigns[source]
        flags = source_flags[source]
        src_unset = source_unset_base[source]
        for stmt in _statements(text):
            stmt_assigns = {**assigns,
                            **{k: _unquote(v) for k, v in _SH_ASSIGN.findall(stmt)}}
            for _ in range(2):
                stmt_assigns = {k: _expand(v, stmt_assigns, env)
                                for k, v in stmt_assigns.items()}
            cfg_dir = None
            for dm in _CONFIG_DIR_ASSIGN.finditer(stmt):
                d = _expand(_unquote(dm.group(1)), stmt_assigns, env)
                if not d.startswith("$"):
                    cfg_dir = d
            inline_win = None
            for wm in _WINDOW_ASSIGN.finditer(stmt):
                inline_win = int(wm.group(1).replace("_", ""))
            first_party = src_unset or bool(_ENV_U_BASE.search(stmt))
            uses = list(_USE.finditer(stmt))
            if cfg_dir and uses:
                # --model on this statement overrides the dir's ANTHROPIC_MODEL, so
                # that settings value never reaches the API from this leg.
                dirs_with_model_flag.add(cfg_dir)
            for um in uses:
                token = _unquote(um.group(1))
                raw = token
                if token.startswith("$"):
                    # a $-reference: every value the flag var may hold. The var is
                    # usually assigned in the wrapper; if not, it falls through to
                    # the environment ("") and produces no leg rather than a bogus "".
                    name = token.lstrip("$").lstrip("{").rstrip("}")
                    values = flags.get(name) or ([env[name]] if name in env else [])
                    for value in values:
                        v = _expand(value, stmt_assigns, env)
                        if not v or v.startswith("$"):
                            continue
                        win = _resolve_window(inline_win, cfg_dir, env)
                        gw = _resolve_gateway(cfg_dir, env, first_party, src_unset)
                        key = (v, source, win, gw, first_party)
                        if key not in seen:
                            seen.add(key)
                            facts.legs.append(Leg(v, source, win, gw, first_party,
                                                  cfg_dir, inline_win, name))
                else:
                    value = _expand(raw, stmt_assigns, env)
                    if value.startswith("$"):
                        continue
                    win = _resolve_window(inline_win, cfg_dir, env)
                    gw = _resolve_gateway(cfg_dir, env, first_party, src_unset)
                    key = (value, source, win, gw, first_party)
                    if key not in seen:
                        seen.add(key)
                        facts.legs.append(Leg(value, source, win, gw, first_party,
                                              cfg_dir, inline_win))

    # Config dirs actually selected (an explicit one suppresses the default).
    dirs: list[str] = []
    explicit_dir = False
    for source, text in texts:
        for stmt in _statements(text):
            for dm in _CONFIG_DIR_ASSIGN.finditer(stmt):
                d = _expand(_unquote(dm.group(1)), source_assigns[source], env)
                if not d.startswith("$") and d not in dirs:
                    dirs.append(d)
                    explicit_dir = True
    if not explicit_dir and env.get(CONFIG_DIR_ENV):
        if env[CONFIG_DIR_ENV] not in dirs:
            dirs.append(env[CONFIG_DIR_ENV])
    elif not explicit_dir:
        default_dir = os.path.join(env.get("HOME", os.path.expanduser("~")), ".claude")
        if default_dir not in dirs:
            dirs.append(default_dir)
    facts.config_dirs = dirs

    # settings.json legs: a model id recorded there, with that dir's own
    # gateway / window. Reached only when a dir is selected above, and only for
    # a dir whose claude call sends no --model (the flag overrides ANTHROPIC_MODEL,
    # so a fixed `--model 'x[1m]'` must not be re-flagged by a stale bare id
    # in the same dir's settings.json).
    seen_src = {(l.model, l.source) for l in facts.legs}
    for d in dirs:
        senv = _settings_env(d)
        if senv is None:
            continue
        src = os.path.join(d, "settings.json")
        if senv.get(GATEWAY_ENV):
            facts.gateway.append(src)
        m = senv.get(MODEL_ENV)
        if d in dirs_with_model_flag:
            m = None
        if m and (m, src) not in seen_src:
            seen_src.add((m, src))
            win = _window_value(senv)
            facts.legs.append(Leg(m, src, win, bool(senv.get(GATEWAY_ENV)), False,
                                  config_dir=d))

    # Process-environment leg: ANTHROPIC_MODEL may be inherited (an ``export`` in
    # the launcher) rather than written in settings.json.
    m = env.get(MODEL_ENV)
    if m and (m, "environment") not in seen_src:
        seen_src.add((m, "environment"))
        facts.legs.append(Leg(m, "environment", _window_value(env),
                              bool(env.get(GATEWAY_ENV)), False))

    if env.get(GATEWAY_ENV):
        facts.gateway.append("environment")
    return facts


def window_findings(facts: AgentCmdFacts) -> list[str]:
    """Launch-gate findings: a bare claude id that will run behind a gateway with
    no declared window. Empty means the command is safe as far as we can read it.

    Per leg: a model fronted by ``[1m]`` or its own >200k window is exempt; a
    bare id behind a gateway with no >200k window is a finding. A window or
    ``[1m]`` on one leg never exempts a different leg's bare id."""
    if not facts.cmd:
        return []
    out: list[str] = []
    for leg in facts.legs:
        if not leg.on_gateway:
            continue
        if not leg.is_claude:
            continue
        if leg.declares_1m:
            continue
        if leg.window and leg.window > 200_000:
            continue
        out.append(
            f"model '{leg.model}' ({leg.source}) is a bare claude id and its leg runs "
            f"behind a gateway: the claude CLI budgets it at 200k and auto-compacts "
            f"the session at ~150-170k. Write '{leg.model}[1m]' or set "
            f"{AUTO_COMPACT_WINDOW_ENV}=1000000 for that leg.")
    return out
