"""Running an experiment or stage entrypoint as a subprocess of the service.

Both local adapters launch a repository script the same way and have to react
to the same two interruptions - an operator cancel and the run timeout - within
the same terminate-then-kill budget. That behaviour lives here once so the two
adapters cannot answer a cancel differently.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

# How long a terminated child gets to exit before it is killed.
TERMINATE_GRACE_SECONDS = 10
_POLL_INTERVAL_SECONDS = 0.4

CANCELLED_EXIT_CODE = 130
TIMEOUT_EXIT_CODE = 124


@dataclass(frozen=True)
class LocalProcessOutcome:
    exit_code: int
    cancelled: bool
    timed_out: bool


def build_subprocess_env(project_root: Path, **extra: str) -> dict[str, str]:
    """The service environment plus this repository's import roots.

    Includes the required import roots (compute sources, shared contracts, the
    repository itself) so a script imports the same packages either way. An
    inherited PYTHONPATH is appended rather than dropped, so it can never
    shadow the repository's own packages.
    """

    env = os.environ.copy()
    pythonpath_parts = [
        str(project_root / "compute" / "src"),
        str(project_root / "shared" / "contracts" / "src"),
        str(project_root),
    ]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.update(extra)
    return env


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()


def run_local_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file,
    cancel_event: Event,
    timeout_seconds: float,
) -> LocalProcessOutcome:
    """Run `command` to completion, cancellation or timeout.

    stderr is folded into `log_file` with stdout so one file is the whole
    story of the run.
    """

    deadline = time.monotonic() + timeout_seconds
    cancelled = False
    timed_out = False
    exit_code = 1
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    while True:
        poll_result = process.poll()
        process_alive = poll_result is None

        if cancel_event.is_set() and process_alive:
            cancelled = True
            _stop(process)
            break

        if time.monotonic() >= deadline and process_alive:
            timed_out = True
            _stop(process)
            break

        if not process_alive:
            exit_code = int(poll_result)
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    if cancelled:
        exit_code = CANCELLED_EXIT_CODE
    elif timed_out:
        exit_code = TIMEOUT_EXIT_CODE
    return LocalProcessOutcome(
        exit_code=exit_code,
        cancelled=cancelled,
        timed_out=timed_out,
    )
