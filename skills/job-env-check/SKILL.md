# job-env-check

Whether a GPU job's bare `python3` can run the other skills. They are started as
`python3 notes/<tool>.py` and re-spawn `python3 eval/bench.py`, so they all need that
interpreter to import torch; on an ssh host the job runs in a non-login shell, where
`python3` may be the system one even though the kernel itself evaluates fine through the
card's own interpreter.

The driver runs it once per loop start (`python3 notes/job_env_check.py`). It prints exactly
one line, `JOB-ENV-OK python3=<path> torch=<version>` or
`JOB-ENV-FAIL python3=<path> <reason>`, with the failing import's own text above a FAIL.
