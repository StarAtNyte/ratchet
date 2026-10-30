#!/usr/bin/env python3
"""A stand-in for the interactive `claude` that scripts/claude-goal.py drives.

It does what the leg reads, and nothing else: writes the session transcript (goal
sentinel, assistant records with a model and usage, API-error records, goal verdicts,
turn ends) under $CLAUDE_CONFIG_DIR/projects/<slug>/<sid>.jsonl, prints the CLI's goal
status lines to its terminal (the tmux pane), writes the deliverable, and reads lines
typed into the pane. STUB_MODE picks the scenario; see tests/test_leg_claude_goal.py.
"""
import json
import os
import re
import sys
import time

args = sys.argv[1:]
if args[:1] == ["--version"]:
    print("2.1.304 (Claude Code)")
    sys.exit(0)
if os.environ.get("STUB_ARGV_LOG"):
    with open(os.environ["STUB_ARGV_LOG"], "a") as f:
        f.write(json.dumps(args) + "\n")


def opt(name):
    return args[args.index(name) + 1] if name in args else None


sid = opt("--session-id") or opt("--resume")
resumed = "--resume" in args
goal = next(a for a in reversed(args) if a.startswith("/goal "))
cond = goal[len("/goal "):]
cfg = os.environ["CLAUDE_CONFIG_DIR"]
slug = re.sub(r"[^A-Za-z0-9]", "-", os.getcwd())
os.makedirs(os.path.join(cfg, "projects", slug), exist_ok=True)
path = os.path.join(cfg, "projects", slug, sid + ".jsonl")
turn_dir = os.environ.get("FK_GOAL_TURN_DIR")
if turn_dir:
    reply = json.load(open(os.path.join(turn_dir, "spec.json")))["reply"]
else:
    reply = re.search(r"to `([^`]*reply\.md)`", cond).group(1)
served = os.environ.get("STUB_SERVED_MODEL", "claude-test-1")
mode = os.environ.get("STUB_MODE", "ok")
n_msg = [0]


def rec(d):
    with open(path, "a") as f:
        f.write(json.dumps(d) + "\n")


def say(text):
    print(text, flush=True)


def assistant(text, model=None):
    n_msg[0] += 1
    rec({"type": "assistant", "message": {
        "id": f"msg_{sid[:6]}_{n_msg[0]}", "model": model or served,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 20,
                  "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000}}})


def api_error(text, kind="rate_limit"):
    rec({"type": "assistant", "isApiErrorMessage": True, "error": kind,
         "message": {"model": "<synthetic>", "content": [{"type": "text", "text": text}]}})


def turn_end():
    rec({"type": "system", "subtype": "turn_duration", "durationMs": 1000})


def met(iterations=1):
    rec({"type": "attachment", "attachment": {"type": "goal_status", "met": True, "condition": cond,
                                              "reason": "check printed OK", "iterations": iterations,
                                              "durationMs": 1234, "tokens": 999}})


def deliver(iterations=1):
    with open(reply, "w") as f:
        f.write(os.environ.get("STUB_REPLY", "EXHAUSTED: stub reply\n"))
    assistant("wrote the reply and ran the check")
    met(iterations)
    turn_end()
    say("✔ Goal achieved (1s · 1 turn · 999 tokens)")


def wait_line():
    # cbreak, like the real TUI's raw input: a canonical-mode tty drops a line longer
    # than MAX_CANON (1024 on macOS), and a pasted /goal condition is ~2 kB
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    buf = []
    try:
        while True:
            ch = os.read(fd, 1).decode("utf-8", "replace")
            if ch in ("\r", "\n", ""):
                break
            buf.append(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    line = "".join(buf) + "\n"
    if os.environ.get("STUB_LINES_LOG"):
        with open(os.environ["STUB_LINES_LOG"], "a") as f:
            f.write(line)
    return line


def idle():
    while True:
        time.sleep(60)


rec({"type": "attachment", "attachment": {"type": "goal_status", "met": False, "sentinel": True,
                                          "condition": cond}})
say("Goal set: " + cond[:60])

if mode in ("ok", "mismatch"):
    deliver()
    idle()
elif mode == "limit" or (mode == "resume_ok" and not resumed):
    assistant("working")
    if os.environ.get("STUB_REPLY_BEFORE_LIMIT"):     # the reply was done; the judge was not
        with open(reply, "w") as f:
            f.write(os.environ.get("STUB_REPLY", "EXHAUSTED: stub reply\n"))
    api_error("You've hit your session limit · resets 7pm (Asia/Shanghai)")
    turn_end()
    say("Goal paused · usage limit reached · send a message after it resets to continue")
    idle()
elif mode == "resume_ok":
    deliver()
    idle()
elif mode == "cleared":
    api_error("Please run /login · API Error: 401 OAuth access token is invalid.", "authentication_failed")
    turn_end()
    say('Goal cleared after an unrecoverable error (authentication failed): "FORGEKERNEL". '
        'Run /goal again to continue.')
    idle()
elif mode == "pause_then_ok":
    assistant("working")
    api_error("API Error: 429 rate_limit_error")
    turn_end()
    say("Goal paused · the request was rate limited · send a message to retry")
    if "/goal" in wait_line():
        say("Goal set: " + cond[:60])
        deliver(iterations=2)
    idle()
elif mode == "false_met":
    assistant("I am done")       # no reply.md written: the judge was wrong
    met()
    turn_end()
    say("✔ Goal achieved (1s · 1 turn · 999 tokens)")
    while "/goal" in wait_line():          # every re-sent goal: deliver (STUB_REPLY) again
        say("Goal set: " + cond[:60])
        deliver(iterations=2)
    idle()
elif mode == "exit":
    assistant("crashing")
    sys.exit(3)
elif mode == "hang":
    assistant("thinking for a long time")
    idle()
