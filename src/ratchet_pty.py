"""Minimal PTY bridge for Ratchet's shell tab (no native Node modules needed).

Runs the user's login shell on a pseudo-terminal and relays raw bytes:
stdin -> pty, pty -> stdout. A control frame b"\\x00\\x1fR<cols>,<rows>\\n" on stdin
resizes the terminal instead of being typed.
"""
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios

FRAME = b"\x00\x1f"


def main():
    cols = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    shell = os.environ.get("SHELL") or "/bin/bash"
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(shell, [shell, "-l"])

    def resize(c, r):
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", r, c, 0, 0))
        try:
            os.kill(pid, signal.SIGWINCH)
        except ProcessLookupError:
            pass

    resize(cols, rows)
    inp = sys.stdin.fileno()
    out = sys.stdout.fileno()
    pending = b""
    while True:
        try:
            ready, _, _ = select.select([fd, inp], [], [])
        except InterruptedError:
            continue
        if fd in ready:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            os.write(out, data)
        if inp in ready:
            data = os.read(inp, 65536)
            if not data:
                break
            pending += data
            while pending:
                i = pending.find(FRAME)
                if i < 0:
                    # a lone trailing \x00 may be the start of a frame split across reads
                    keep = 1 if pending.endswith(b"\x00") else 0
                    os.write(fd, pending[: len(pending) - keep])
                    pending = pending[len(pending) - keep:]
                    break
                if i > 0:
                    os.write(fd, pending[:i])
                    pending = pending[i:]
                j = pending.find(b"\n")
                if j < 0:
                    break
                frame = pending[2:j].decode(errors="ignore")
                pending = pending[j + 1:]
                if frame.startswith("R"):
                    try:
                        c, r = frame[1:].split(",")
                        resize(int(c), int(r))
                    except ValueError:
                        pass
    try:
        os.kill(pid, signal.SIGHUP)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    main()
