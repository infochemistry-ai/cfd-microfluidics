"""Adapter protocol for calling existing compute pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Protocol

from microfluidics_contracts import SubmitRunRequestV1


@dataclass
class AdapterRunOutcome:
    exit_code: int
    started_at: str
    finished_at: str
    duration_seconds: float
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None


class ComputeAdapter(Protocol):
    name: str

    def run(
        self,
        run_id: str,
        request: SubmitRunRequestV1,
        cancel_event: Event,
        run_work_dir: Path,
    ) -> AdapterRunOutcome:
        """Execute a compute run using existing core pipelines."""
