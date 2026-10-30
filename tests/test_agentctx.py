"""agentctx: the context-window launch gate for --agent-cmd.

The claude CLI clamps a bare claude model id to a 200k budget behind a gateway
and auto-compacts the session at ~150-170k. These tests pin the leg-level
reader that turns that fact into a launch finding, including the wrapper-script
shape (a ``case`` with a 1M teacher branch and a bare student branch) that the
original flat model list + global window could not tell apart.
"""

import json
from pathlib import Path

from forgekernel.loop.agentctx import (
    inspect_agent_cmd,
    window_findings,
)


def _env(tmp_path: Path, **extra) -> dict:
    """An agent environment whose HOME has no ~/.claude/settings.json."""
    return {"HOME": str(tmp_path / "home"), **extra}


def _cfg(tmp_path: Path, name: str, env: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "settings.json").write_text(json.dumps({"env": env}), encoding="utf-8")
    return d


def _models(facts):
    return [leg.model for leg in facts.legs]


def test_bare_claude_id_behind_gateway_is_a_finding(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm",
                                 "ANTHROPIC_API_KEY": "sk-x"})
    cmd = f"CLAUDE_CONFIG_DIR={cfg} claude -p --model claude-opus-5 --dangerously-skip-permissions"
    facts = inspect_agent_cmd(cmd, env=_env(tmp_path))
    assert _models(facts) == ["claude-opus-5"]
    assert str(cfg) in facts.config_dirs
    assert any("settings.json" in g for g in facts.gateway)
    findings = window_findings(facts)
    assert len(findings) == 1
    assert "claude-opus-5[1m]" in findings[0]
    assert "200k" in findings[0]


def test_1m_suffix_clears_the_finding(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm"})
    cmd = f"CLAUDE_CONFIG_DIR={cfg} claude -p --model 'fable[1m]'"
    facts = inspect_agent_cmd(cmd, env=_env(tmp_path))
    assert facts.legs[0].declares_1m and facts.legs[0].is_claude
    assert window_findings(facts) == []


def test_autocompact_window_env_clears_the_finding(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm"})
    inline = f"CLAUDE_CONFIG_DIR={cfg} CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000 claude -p --model claude-opus-5"
    assert window_findings(inspect_agent_cmd(inline, env=_env(tmp_path))) == []
    # ... but a window that is not above 200k does not help
    low = f"CLAUDE_CONFIG_DIR={cfg} CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000 claude -p --model claude-opus-5"
    assert len(window_findings(inspect_agent_cmd(low, env=_env(tmp_path)))) == 1
    # the settings.json env is honoured too
    cfg2 = _cfg(tmp_path, "cfg2", {"ANTHROPIC_BASE_URL": "https://gw.example/llm",
                                   "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000"})
    assert window_findings(inspect_agent_cmd(f"CLAUDE_CONFIG_DIR={cfg2} claude -p --model opus", env=_env(tmp_path))) == []


def test_no_gateway_means_no_finding(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_MODEL": "claude-opus-5"})  # first-party endpoint
    facts = inspect_agent_cmd(f"CLAUDE_CONFIG_DIR={cfg} claude -p --model claude-opus-5", env=_env(tmp_path))
    assert facts.gateway == []
    assert window_findings(facts) == []


def test_gateway_from_process_env_counts(tmp_path):
    env = _env(tmp_path, ANTHROPIC_BASE_URL="https://gw.example/llm")
    facts = inspect_agent_cmd("claude -p --model claude-sonnet-5", env=env)
    assert facts.gateway == ["environment"]
    assert len(window_findings(facts)) == 1
    # a wrapper that strips the variable before exec is not on a gateway
    facts2 = inspect_agent_cmd("unset ANTHROPIC_BASE_URL; claude -p --model claude-sonnet-5", env=env)
    assert window_findings(facts2) == []


def test_channel_alias_and_foreign_ids_are_not_claude_ids(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm",
                                 "ANTHROPIC_MODEL": "CLAUDE_ew682d"})
    facts = inspect_agent_cmd(f"CLAUDE_CONFIG_DIR={cfg} claude -p --model GLM_2squr5", env=_env(tmp_path))
    # --model on the call overrides the dir's ANTHROPIC_MODEL, so only the flag is a leg
    assert {leg.model for leg in facts.legs} == {"GLM_2squr5"}
    assert not any(leg.is_claude for leg in facts.legs)
    bare = inspect_agent_cmd(f"CLAUDE_CONFIG_DIR={cfg} claude -p", env=_env(tmp_path))
    assert {leg.model for leg in bare.legs} == {"CLAUDE_ew682d"}
    assert not any(leg.is_claude for leg in bare.legs) and window_findings(bare) == []
    assert window_findings(facts) == []


def test_wrapper_case_splits_teacher_and_student_legs(tmp_path):
    # regression: a 1M window / fable[1m] teacher branch must not clear a
    # bare student branch. This is exactly the agent-with-fallback.sh shape.
    home = tmp_path / "home"
    llmc = home / ".config" / "claude-config-proxy"
    llmc.mkdir(parents=True)
    (llmc / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw/llm"}}))
    wrap = tmp_path / "agent-with-fallback.sh"
    wrap.write_text(
        '#!/usr/bin/env bash\n'
        'ROLE=$1\n'
        'case "$ROLE" in\n'
        "  teacher) MODEL_FLAG='fable[1m]'; CFG_LLMC=\"$HOME/.config/claude-config-proxy\" ;;\n"
        "  student) MODEL_FLAG='claude-opus-5'; CFG_LLMC=\"$HOME/.config/claude-config-proxy\" ;;\n"
        'esac\n'
        'unset ANTHROPIC_BASE_URL\n'
        'claude -p --model "$MODEL_FLAG" < prompt || \\\n'
        'CLAUDE_CONFIG_DIR="$CFG_LLMC" claude -p --model "$MODEL_FLAG" < prompt\n',
        encoding="utf-8")
    facts = inspect_agent_cmd(f"bash {wrap} student", env={"HOME": str(home)})
    assert facts.scripts == [str(wrap)]
    assert str(llmc) in facts.config_dirs
    models = {leg.model for leg in facts.legs}
    assert models == {"fable[1m]", "claude-opus-5"}
    findings = window_findings(facts)
    assert len(findings) == 1 and "claude-opus-5" in findings[0] and str(wrap) in findings[0]
    # the fable[1m] leg itself is clean even though it shares the source
    clean = [leg for leg in facts.legs if leg.declares_1m]
    assert clean and clean[0].source == str(wrap)


def test_env_unset_makes_leg_first_party(tmp_path):
    # regression: `env -u ANTHROPIC_BASE_URL` (the direct-API opus-5
    # recipe) is first-party and must NOT be a finding, even with the var set in
    # the inherited environment.
    env = _env(tmp_path, ANTHROPIC_BASE_URL="https://gw.example/llm")
    inline = "env -u ANTHROPIC_BASE_URL claude -p --model claude-opus-5"
    facts = inspect_agent_cmd(inline, env=env)
    # the leg is first-party, so no finding even though the env has a gateway
    assert window_findings(facts) == []


def test_env_unset_precedes_statement(tmp_path):
    # the `env -u` applies to the claude call on the same statement; a later
    # statement that does not unset is governed by the (gateway) environment.
    env = _env(tmp_path, ANTHROPIC_BASE_URL="https://gw.example/llm")
    facts = inspect_agent_cmd("env -u ANTHROPIC_BASE_URL claude -p --model claude-opus-5; "
                              "claude -p --model claude-opus-5", env=env)
    findings = window_findings(facts)
    assert len(findings) == 1  # the second (non-unsets) claude is the finding


def test_anthropic_model_from_environment_is_a_leg(tmp_path):
    # regression: ANTHROPIC_MODEL inherited from the environment (not
    # settings.json) must be read as a leg. The old reader missed it and let a
    # bare claude id through.
    env = _env(tmp_path, ANTHROPIC_BASE_URL="https://gw.example/llm",
               ANTHROPIC_MODEL="claude-opus-5")
    facts = inspect_agent_cmd("claude -p", env=env)
    assert "claude-opus-5" in _models(facts)
    assert len(window_findings(facts)) == 1


def test_anthropic_model_only_no_gateway_is_fine(tmp_path):
    # ANTHROPIC_MODEL set but no gateway (first-party) is not a finding.
    env = _env(tmp_path, ANTHROPIC_MODEL="claude-opus-5")
    facts = inspect_agent_cmd("claude -p", env=env)
    assert window_findings(facts) == []


def test_inline_window_wins_over_settings(tmp_path):
    # regression: an inline CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000 must beat
    # a 200000 in the config dir's settings.json, so a correct launch is not
    # refused.
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm",
                                 "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000"})
    cmd = f"CLAUDE_CONFIG_DIR={cfg} CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000 claude -p --model claude-opus-5"
    assert window_findings(inspect_agent_cmd(cmd, env=_env(tmp_path))) == []


def test_no_agent_cmd_is_silent(tmp_path):
    facts = inspect_agent_cmd(None, env=_env(tmp_path, ANTHROPIC_BASE_URL="https://gw/llm"))
    assert facts.legs == [] and window_findings(facts) == []


def test_settings_model_is_not_a_leg_when_the_call_passes_model(tmp_path):
    # regression: a gateway's config dirs record a bare
    # ANTHROPIC_MODEL in settings.json. --model on the claude call overrides it,
    # so a fixed `--model 'claude-opus-5[1m]'` must not be re-flagged by that
    # stale bare id — otherwise the gate forces an edit of every config dir.
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm",
                                 "ANTHROPIC_MODEL": "claude-opus-5"})
    fixed = f"CLAUDE_CONFIG_DIR={cfg} claude -p --model 'claude-opus-5[1m]'"
    assert window_findings(inspect_agent_cmd(fixed, env=_env(tmp_path))) == []
    # ... while a bare `claude -p` on the same dir does send ANTHROPIC_MODEL
    bare = f"CLAUDE_CONFIG_DIR={cfg} claude -p"
    findings = window_findings(inspect_agent_cmd(bare, env=_env(tmp_path)))
    assert len(findings) == 1 and "settings.json" in findings[0]
    # ... and a bare --model on the dir is exactly one finding, not two
    dup = f"CLAUDE_CONFIG_DIR={cfg} claude -p --model claude-opus-5"
    assert len(window_findings(inspect_agent_cmd(dup, env=_env(tmp_path)))) == 1


def test_unset_recognised_anywhere_in_name_list(tmp_path):
    # regression: the real wrapper unsets a whole list with
    # ANTHROPIC_BASE_URL in the middle, across a backslash continuation, ending
    # in `|| true`. That is first-party.
    env = _env(tmp_path, ANTHROPIC_BASE_URL="https://gw.example/llm")
    wrapper = ("unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_MODEL ANTHROPIC_API_KEY \\\n"
               "      CLAUDE_CONFIG_DIR CLAUDECODE 2>/dev/null || true\n"
               "claude -p --model claude-opus-5\n")
    assert window_findings(inspect_agent_cmd(wrapper, env=env)) == []
    # a list that does not contain BASE_URL is not first-party
    other = "unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL\nclaude -p --model claude-opus-5\n"
    assert len(window_findings(inspect_agent_cmd(other, env=env))) == 1
    # several -u flags on env, BASE_URL not first
    multi = "env -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL claude -p --model claude-opus-5"
    assert window_findings(inspect_agent_cmd(multi, env=env)) == []


def test_backslash_continuation_keeps_config_dir_on_the_statement(tmp_path):
    cfg = _cfg(tmp_path, "cfg", {"ANTHROPIC_BASE_URL": "https://gw.example/llm"})
    cmd = f"CLAUDE_CONFIG_DIR={cfg} \\\n    claude -p --model claude-opus-5"
    facts = inspect_agent_cmd(cmd, env=_env(tmp_path))
    assert [leg.config_dir for leg in facts.legs] == [str(cfg)]
    assert len(window_findings(facts)) == 1
