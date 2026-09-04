"""Compute runner that maps adapter output to synchronous service contracts."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from backend.app.adapters import (
    ComputeAdapter,
    LocalStageAdapter,
)
from microfluidics_contracts import ResultPayloadV1, RuntimeSettings, SubmitRunRequestV1


class ComputeRunner:
    """Runs a submit request via adapter and returns normalized result payload."""

    def __init__(
        self,
        project_root: Path,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self.project_root = project_root
        self.settings = settings or RuntimeSettings.from_env()
        self.adapter: ComputeAdapter = LocalStageAdapter(
            project_root=project_root,
            settings=self.settings,
        )

    def run(
        self,
        run_id: str,
        request: SubmitRunRequestV1,
        cancel_event: Event,
        run_work_dir: Path,
    ) -> ResultPayloadV1:
        outcome = self.adapter.run(
            run_id=run_id,
            request=request,
            cancel_event=cancel_event,
            run_work_dir=run_work_dir,
        )

        return ResultPayloadV1(
            run_id=run_id,
            adapter_name=self.adapter.name,
            experiment_id=request.experiment_id,
            exit_code=outcome.exit_code,
            started_at=outcome.started_at,
            finished_at=outcome.finished_at,
            duration_seconds=outcome.duration_seconds,
            artifacts=outcome.artifacts,
            log_path=outcome.log_path,
            metadata=outcome.metadata,
        )
