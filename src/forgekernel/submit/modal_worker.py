"""Run one archived ForgeKernel GPU job in an ephemeral Modal container."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    import modal

    archive = Path(sys.argv[1]).read_bytes()
    job = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    image = modal.Image.from_registry(job["image"]).run_commands(
        "python -c 'import sysconfig; from pathlib import Path; "
        "Path(sysconfig.get_paths()[\"purelib\"], \"modal_system_sites.pth\").write_text("
        "\"/usr/local/lib/python3.12/dist-packages\\n/usr/lib/python3/dist-packages\\n\")'"
    )
    app = modal.App("forgekernel-gpu-job", image=image)

    @app.function(gpu=job["gpu"], timeout=int(job["timeout"]) + 60, scaledown_window=2, serialized=True)
    def execute(payload: bytes, argv: list[str], env: dict[str, str], timeout: int):
        import io
        import os
        import signal
        import shlex
        import shutil
        import subprocess
        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory(prefix="forge-job-") as tree:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as tf:
                tf.extractall(tree, filter="data")
            job_env = {**os.environ, **env}
            # Modal GPUs generally cannot change clocks. NCU can still collect
            # counters when asked to leave clocks alone.
            ncu = shutil.which("ncu", path=job_env.get("PATH"))
            if ncu:
                bindir = os.path.join(tree, ".forge-bin")
                os.mkdir(bindir)
                wrapper = os.path.join(bindir, "ncu")
                with open(wrapper, "w", encoding="utf-8") as fh:
                    fh.write("#!/bin/sh\nexec " + shlex.quote(ncu) + " --clock-control none \"$@\"\n")
                os.chmod(wrapper, 0o755)
                job_env["PATH"] = bindir + os.pathsep + job_env.get("PATH", "")
                # gpu_entry prepends CUDA/bin to PATH; an exported Bash
                # function keeps the no-clock-control option when it does so.
                job_env["BASH_FUNC_ncu%%"] = '() { command ' + shlex.quote(ncu) + ' --clock-control none "$@"; }'
            proc = subprocess.Popen(argv, cwd=tree, env=job_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                output, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate()
                return 124, output.decode(errors="replace") + "\n[modal] job timed out\n"
            return proc.returncode, output.decode(errors="replace")

    try:
        with app.run():
            rc, output = execute.remote(archive, job["argv"], job["env"], int(job["timeout"]))
        print("FK_JOB_BEGIN", flush=True)
        print(output, end="", flush=True)
        return int(rc)
    except Exception as exc:
        print(f"[jobs] TRANSPORT ERROR (the job could not be delivered): {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
