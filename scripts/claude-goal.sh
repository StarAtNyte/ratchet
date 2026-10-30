#!/usr/bin/env bash
# claude-goal.sh — one agent turn served by an interactive `claude` driven by /goal
# (scripts/claude-goal.py has the whole story).
#
#   bash claude-goal.sh <expected-model> <model-flag> [extra claude args...]
#     prompt on stdin, reply on stdout.
#
# scripts/agent-claude.sh execs this when FK_AGENT_DRIVER=goal (set by a caller that
# wants a goal session, e.g. loop/goal_ref.py).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The helper is stdlib-only and 3.9-clean, so the system python3 serves when no
# interpreter is named.
exec "${FK_AGENT_PY:-python3}" "$HERE/claude-goal.py" "$@"
