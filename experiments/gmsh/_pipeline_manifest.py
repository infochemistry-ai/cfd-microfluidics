"""Fail-fast manifest integration for experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from experiments.gmsh._path_utils import _normalize_user_path
from microfluidics.pipeline import create_pipeline_manifest_file, upsert_pipeline_run


def add_pipeline_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pipeline-manifest",
        type=str,
        default="",
        help=(
            "Optional path to pipeline_manifest.json. If provided, this run must "
            "successfully record manifest updates."
        ),
    )
    parser.add_argument(
        "--pipeline-run-id",
        type=str,
        default="",
        help=(
            "Optional logical run id used inside the manifest. Defaults to the "
            "created run directory name."
        ),
    )
    parser.add_argument(
        "--pipeline-parent-run-id",
        action="append",
        default=[],
        help=(
            "Optional parent run id in the manifest. May be repeated, and each "
            "value may also contain comma-separated ids."
        ),
    )


def build_pipeline_manifest_recorder(
    args: argparse.Namespace,
    *,
    stage_type: str,
    run_dir: Path,
) -> "OptionalPipelineManifestRecorder":
    raw_manifest = str(getattr(args, "pipeline_manifest", "")).strip()
    if not raw_manifest:
        return OptionalPipelineManifestRecorder.disabled()
    raw_run_id = str(getattr(args, "pipeline_run_id", "")).strip() or run_dir.name
    raw_parent_values = list(getattr(args, "pipeline_parent_run_id", []) or [])
    parent_run_ids = _parse_parent_run_ids(raw_parent_values)
    manifest_path = _normalize_user_path(raw_manifest).resolve()
    return OptionalPipelineManifestRecorder(
        manifest_path=manifest_path,
        stage_type=str(stage_type).strip(),
        run_id=raw_run_id,
        parent_run_ids=parent_run_ids,
        run_dir=run_dir.resolve(),
    )


def _parse_parent_run_ids(raw_values: list[str]) -> list[str]:
    parent_ids: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for item in str(raw).split(","):
            value = item.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            parent_ids.append(value)
    return parent_ids


class OptionalPipelineManifestRecorder:
    """Fail-fast manifest writer for experiment scripts."""

    def __init__(
        self,
        *,
        manifest_path: Path | None,
        stage_type: str,
        run_id: str,
        parent_run_ids: list[str],
        run_dir: Path | None,
    ) -> None:
        self._manifest_path = manifest_path
        self._stage_type = stage_type
        self._run_id = run_id
        self._parent_run_ids = list(parent_run_ids)
        self._run_dir = run_dir

    @classmethod
    def disabled(cls) -> "OptionalPipelineManifestRecorder":
        return cls(
            manifest_path=None,
            stage_type="",
            run_id="",
            parent_run_ids=[],
            run_dir=None,
        )

    @property
    def enabled(self) -> bool:
        return self._manifest_path is not None

    @property
    def run_id(self) -> str:
        return self._run_id

    def record_started(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._record(
            status="running",
            inputs=inputs,
            artifacts=artifacts,
            metadata=metadata,
        )

    def record_completed(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._record(
            status="completed",
            inputs=inputs,
            outputs=outputs,
            artifacts=artifacts,
            metadata=metadata,
        )

    def record_failed(
        self,
        *,
        error: BaseException,
        inputs: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("exception_type", type(error).__name__)
        merged_metadata.setdefault("exception_message", str(error))
        self._record(
            status="failed",
            inputs=inputs,
            artifacts=artifacts,
            metadata=merged_metadata,
        )

    def _ensure_manifest_exists(self) -> None:
        if self._manifest_path is None:
            return
        if self._manifest_path.exists():
            return
        pipeline_id = self._manifest_path.parent.name or self._manifest_path.stem
        try:
            create_pipeline_manifest_file(
                self._manifest_path,
                pipeline_id=pipeline_id,
                pipeline_root=self._manifest_path.parent,
            )
        except FileExistsError:
            return

    def _record(
        self,
        *,
        status: str,
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._manifest_path is None or self._run_dir is None:
            return
        self._ensure_manifest_exists()
        upsert_pipeline_run(
            self._manifest_path,
            run_id=self._run_id,
            stage_type=self._stage_type,
            status=status,
            run_dir=self._run_dir,
            display_name=self._run_dir.name,
            parent_run_ids=self._parent_run_ids,
            inputs=inputs,
            outputs=outputs,
            artifacts=artifacts,
            metadata=metadata,
        )
