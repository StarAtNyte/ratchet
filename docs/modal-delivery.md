# Modal GPU delivery

Install the optional client in the ForgeKernel virtual environment and configure `local.env`:

```sh
uv pip install --python .venv/bin/python -e '.[modal]'
# Authenticate Modal using its normal CLI login, if not already authenticated.
# local.env:
FK_DELIVERY=modal
FK_MODAL_GPU=H200
FK_MODAL_IMAGE=lmsysorg/sglang:dev-muse-spark-open
FK_MODAL_TIMEOUT=300
```

Each GPU job uploads the same Git archive used by local and SSH delivery to a fresh ephemeral Modal container, runs `gpu_entry.sh` on one GPU, returns the combined stdout/stderr, and stops its app. The subprocess timeout is bounded by `FK_MODAL_TIMEOUT` seconds; a small fixed allowance covers container startup. Jobs are serialized by a local ForgeKernel lock to avoid unexpectedly launching multiple billable GPUs. `forge status <run>` reports campaign progress; `python -m forgekernel.submit <operator> status <job>` and `logs <job>` report individual jobs. Logs are returned when the job finishes, rather than streamed.

Set `FK_TASK_CUDA_GRAPH=1` in `local.env` to measure ordinary ForgeKernel tasks with eight cold-L2 CUDA graph replays per input group, alternating reference and candidate. Both sides use the same method; this is a local screen and does not reproduce the official judge's overhead subtraction or fresh-input protocol. The Modal worker invokes Nsight Compute with `--clock-control none`, because Modal does not permit its default GPU clock lock.

The public Muse image is not the official judge image. Confirm correctness and performance with the official challenge harness. GPU containers are short lived, so their filesystem and compilation caches are not preserved across jobs. The selected image must contain Python, CUDA-enabled PyTorch, build tools and any kernel dependencies required by the task; Nsight Compute may be unavailable. Job environment values, including sealed-reference keys when present, are transmitted to Modal as function inputs and never placed in command-line arguments or logs by this transport; Modal receives them as part of the job. Modal authentication and billing use the existing local Modal account. Stopping the job terminates the local client; the app is ephemeral and the remote function also has an explicit timeout.
