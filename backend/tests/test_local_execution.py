from __future__ import annotations

import sys
from pathlib import Path
from threading import Event, Timer
from types import SimpleNamespace

import pytest

from backend.app.adapters.local_process import (
    CANCELLED_EXIT_CODE,
    run_local_process,
)
from backend.app.adapters.local_stage_adapter import require_compute_device
from backend.app.stage_registry import Device


def test_cuda_request_fails_with_actionable_message_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    with pytest.raises(RuntimeError, match="no usable CUDA device is visible"):
        require_compute_device(Device.CUDA)


def test_cpu_request_does_not_require_a_cuda_runtime() -> None:
    require_compute_device(Device.CPU)


def test_local_process_cancellation_terminates_the_child(tmp_path: Path) -> None:
    cancelled = Event()
    log_path = tmp_path / "process.log"
    timer = Timer(0.2, cancelled.set)
    timer.start()
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            outcome = run_local_process(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=tmp_path,
                env={},
                log_file=log_file,
                cancel_event=cancelled,
                timeout_seconds=20,
            )
    finally:
        timer.cancel()

    assert outcome.cancelled is True
    assert outcome.timed_out is False
    assert outcome.exit_code == CANCELLED_EXIT_CODE
