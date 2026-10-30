"""forge -- ForgeKernel's command line (forge --version prints the version).

  forge init <op>                      -> tasks/<op>/ from tasks/_template/
  forge init <op> --leaderboard <suite> --dataset ... --definition ... [--workload ...]
                                       -> a task that IS one board problem
  forge new tasks/<op> [--dry-run]     -> a new run of the task: .artifacts/<op>_<id>/
  forge run .artifacts/<op>_<id>       -> run it, or continue it where it stopped
  forge status [.artifacts/<op>_<id>]  -> where a run is (every run when omitted)
  forge harvest .artifacts/<op>_<id>   -> the knowledge-base write-up of a stopped run
  forge export .artifacts/<op>_<id>    -> the best kernel (a board task's as the board's solution)

`forge run` exits 0 when the final confirmation meets the target (and the correctness
monitor raised no alarm on that kernel), and 2 otherwise.

`forge help internal` lists the lower-level commands the loop itself uses.
"""

from __future__ import annotations

import sys

INTERNAL_HELP = """\
Lower-level commands. They act on the operator trees a run generates under
.artifacts/<run>/operators/ -- point FORGEKERNEL_OPERATORS_DIR at that directory first.
`forge run` drives all of them; they are listed for debugging a run, not as a second way
to configure one.

  forge init <op> --kernel-file ...    -> scaffold an operator repo directly
  forge <op> baseline --ref-cmd ...    -> measure + archive the ref (do this first)
  forge <op> loop --workspace ...      -> run the two-layer loop
  forge <op> ncu-trace                 -> per-iteration NCU metric evolution (meta view)
  forge <op> window                    -> check window.toml (timed-window contract) + ledger
  forge <op> leaderboard show|export   -> the card's problem / the board's solution JSON
  forge <op> monitor [--sha ...]       -> correctness monitor: the card's own check under
                                          stressed inputs + kernel-output watch (advisory)
  forge <op> submit|status|logs|stop   -> GPU-job harness (forgekernel.submit)
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:2] == ["help", "internal"]:
        print(INTERNAL_HELP.strip())
        return 0
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    if argv[0] in ("-V", "--version"):
        from forgekernel import __version__

        print(f"forgekernel {__version__}")
        return 0
    if argv and argv[0] in ("init", "new", "run", "status", "harvest", "export"):
        from forgekernel.launch import COMMANDS

        return COMMANDS[argv[0]](argv[1:])

    if len(argv) >= 2 and argv[1] == "ncu-trace":
        from forgekernel.inner.ncu import main as ncu_main

        return ncu_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "ncu-report":
        from forgekernel.inner.ncu import report_main

        return report_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "baseline":
        from forgekernel.baseline_cli import main as baseline_main

        return baseline_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "window":
        from forgekernel.window_contract import main as window_main

        return window_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "leaderboard":
        from forgekernel.leaderboard import main as leaderboard_main

        return leaderboard_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "monitor":
        from forgekernel.monitor.cli import main as monitor_main

        return monitor_main([argv[0], *argv[2:]])

    if len(argv) >= 2 and argv[1] == "loop":
        from forgekernel.loop.cli import main as loop_main

        return loop_main([argv[0], *argv[2:]])

    from forgekernel.submit.cli import main as submit_main

    return submit_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
