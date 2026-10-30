"""scripts/timeout-run.py: the agent leg's wall-clock bound, and the hop a termination of
the loop's turn has to cross to reach the leg (which runs in a session of its own)."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "timeout-run.py"


def test_timeout_run_forwards_a_termination_to_the_leg(tmp_path):
    """The loop ends a turn by SIGTERM to the group it started; the leg runs in the
    session timeout-run.py made for it and only hears of it by forwarding (a goal leg
    then takes its tmux session down)."""
    mark = tmp_path / "got-term"
    child = tmp_path / "child.py"
    child.write_text("import signal, sys, time, pathlib\n"
                     f"signal.signal(signal.SIGTERM, lambda *a: (pathlib.Path({str(mark)!r}).write_text('x'), sys.exit(0)))\n"
                     "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(HELPER), "60", sys.executable, str(child)],
                            start_new_session=True)
    time.sleep(1.5)
    os.killpg(proc.pid, signal.SIGTERM)
    assert proc.wait(timeout=20) == 128 + signal.SIGTERM
    assert mark.read_text() == "x"


def test_timeout_run_still_bounds_the_child(tmp_path):
    proc = subprocess.run([sys.executable, str(HELPER), "1", sys.executable, "-c",
                           "import time; time.sleep(30)"], timeout=30)
    assert proc.returncode == 124
