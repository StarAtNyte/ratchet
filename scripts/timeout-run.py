#!/usr/bin/env python3
"""Run `<cmd...>` under a wall-clock timeout; kill the whole process group on expiry.

Why: `claude -p` has no idle timeout of its own. An API gateway's SSE stream can
establish the TCP connection and then never deliver a terminal frame nor close —
the CLI waits forever and the turn never finishes (observed: a turn hung 3h with
zero bytes written). The
model is a separate process family, so wait() can't bound it without a kill.

Usage: timeout-run.py <seconds> <cmd> [args...]
Exit:  the child's exit status, or 124 if the timeout fired (like GNU timeout).
"""
import os
import signal
import subprocess
import sys
import time


def main() -> int:
    secs = float(sys.argv[1])
    argv = sys.argv[2:]
    # start_new_session: the child becomes a session leader, so its pgid == its
    # pid and os.killpg hits the child plus anything it spawned (Claude's own
    # helpers) without touching the shared proxy daemon (a different session).
    proc = subprocess.Popen(argv, start_new_session=True)

    # The child's own session is out of reach of a signal sent to THIS process's group:
    # the loop ends an agent turn by SIGTERM to the group it started (agent-claude.sh and
    # this helper), and without forwarding the turn below — a goal session owning a tmux
    # session and an interactive claude — kept running unsupervised until its deadline.
    def _forward(signum, frame):
        try:
            os.killpg(proc.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        sys.exit(128 + signum)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _forward)
    try:
        return proc.wait(timeout=secs)
    except subprocess.TimeoutExpired:
        # On macOS, signaling a process group whose leader has already died
        # (SIGTERM killed it during the grace window) raises EPERM, not ESRCH —
        # and EPERM surfaces as PermissionError. Treat it as "group gone".
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        # Grace period; escalate if the child ignores SIGTERM.
        time.sleep(3)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.wait()
        return 124


if __name__ == "__main__":
    sys.exit(main())
