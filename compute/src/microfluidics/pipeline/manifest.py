"""Minimal DAG-friendly manifest helpers for multi-stage pipeline runs.

A manifest stores run lineage as a graph keyed by run id, which lets one
upstream run fan out into multiple downstream runs such as transport, thermal,
or future chemistry stages.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PIPELINE_MANIFEST_SCHEMA_VERSION = 1
_ALLOWED_RUN_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}


class PipelineManifestError(ValueError):
    """Raised when a manifest payload is invalid or cannot be updated safely."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return str(value)


def _looks_like_path_string(value: str) -> bool:
    text = str(value).strip()
    if not text or "://" in text:
        return False
    candidate = Path(text)
    return (
        candidate.is_absolute() or "/" in text or "\\" in text or text.startswith(".")
    )


def _discover_repo_root(start_dir: Path) -> Path:
    current = start_dir.resolve(strict=False)
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return current


def _normalize_mapping_paths(value: Any, *, pipeline_root: Path) -> Any:
    if isinstance(value, Path):
        return _portable_path(value, pipeline_root=pipeline_root)
    if isinstance(value, str):
        if _looks_like_path_string(value):
            return _portable_path(value, pipeline_root=pipeline_root)
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_mapping_paths(item, pipeline_root=pipeline_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _normalize_mapping_paths(item, pipeline_root=pipeline_root)
            for item in value
        ]
    return _json_ready(value)


def _portable_path(path: str | Path, *, pipeline_root: Path) -> str:
    repo_root = _discover_repo_root(pipeline_root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = pipeline_root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(pipeline_root)
    except ValueError:
        try:
            return resolved.relative_to(repo_root).as_posix()
        except ValueError:
            return resolved.as_posix()
    return relative.as_posix() or "."


def resolve_manifest_path_reference(
    manifest_path: str | Path,
    stored_path: str | Path,
) -> Path:
    """Resolve a stored manifest path reference to a local filesystem path."""

    manifest_root = Path(manifest_path).resolve(strict=False).parent
    repo_root = _discover_repo_root(manifest_root)
    raw_value = str(stored_path).strip()
    if raw_value in {"", "."}:
        return manifest_root
    direct_candidate = Path(raw_value)
    if direct_candidate.is_absolute():
        return direct_candidate.resolve(strict=False)
    posix_path = PurePosixPath(raw_value)
    candidate = Path(posix_path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    artifact_relative = (manifest_root / candidate).resolve(strict=False)
    repo_relative = (repo_root / candidate).resolve(strict=False)
    if artifact_relative.exists():
        return artifact_relative
    if repo_relative.exists():
        return repo_relative
    if raw_value.startswith(("results/", "data/")):
        return repo_relative
    return artifact_relative


def _resolve_pipeline_root(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    stored_root = str(manifest.get("pipeline_root", ".")).strip() or "."
    return resolve_manifest_path_reference(manifest_path, stored_root)


def _validate_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not run_id:
        raise PipelineManifestError("run_id must be a non-empty string.")
    return run_id


def _validate_stage_type(value: str) -> str:
    stage_type = str(value).strip()
    if not stage_type:
        raise PipelineManifestError("stage_type must be a non-empty string.")
    return stage_type


def _validate_status(value: str) -> str:
    status = str(value).strip().lower()
    if status not in _ALLOWED_RUN_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_RUN_STATUSES))
        raise PipelineManifestError(
            f"Unsupported run status {value!r}. Allowed values: {allowed}."
        )
    return status


def _validate_parents(
    parent_run_ids: Sequence[str] | None,
    *,
    manifest_runs: Mapping[str, Any],
) -> list[str]:
    if parent_run_ids is None:
        return []
    seen: set[str] = set()
    parents: list[str] = []
    for raw_value in parent_run_ids:
        run_id = _validate_run_id(raw_value)
        if run_id in seen:
            continue
        if run_id not in manifest_runs:
            raise PipelineManifestError(
                f"Unknown parent run id {run_id!r}; register parent runs first."
            )
        seen.add(run_id)
        parents.append(run_id)
    return parents


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f"{path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def save_pipeline_manifest(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist a manifest atomically and return a normalized copy."""

    path = Path(manifest_path).resolve(strict=False)
    normalized = load_pipeline_manifest_from_payload(path, manifest)
    _atomic_write_json(path, normalized)
    return normalized


def load_pipeline_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest file."""

    path = Path(manifest_path).resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineManifestError(
            f"Pipeline manifest is not valid JSON: {path}"
        ) from exc
    return load_pipeline_manifest_from_payload(path, payload)


def load_pipeline_manifest_from_payload(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a manifest payload without reading from disk."""

    if not isinstance(payload, Mapping):
        raise PipelineManifestError("Pipeline manifest payload must be a JSON object.")

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != PIPELINE_MANIFEST_SCHEMA_VERSION:
        raise PipelineManifestError(
            "Unsupported pipeline manifest schema_version "
            f"{schema_version!r}; expected {PIPELINE_MANIFEST_SCHEMA_VERSION}."
        )

    pipeline_id = str(payload.get("pipeline_id", "")).strip()
    if not pipeline_id:
        raise PipelineManifestError("Pipeline manifest must define pipeline_id.")

    pipeline_root = str(payload.get("pipeline_root", ".")).strip() or "."
    metadata = payload.get("metadata", {})
    runs = payload.get("runs", {})
    if not isinstance(metadata, Mapping):
        raise PipelineManifestError("Pipeline manifest metadata must be an object.")
    if not isinstance(runs, Mapping):
        raise PipelineManifestError("Pipeline manifest runs must be an object.")

    normalized_runs: dict[str, Any] = {}
    for raw_run_id, raw_run in runs.items():
        run_id = _validate_run_id(str(raw_run_id))
        if not isinstance(raw_run, Mapping):
            raise PipelineManifestError(
                f"Run entry {run_id!r} must be an object in the manifest."
            )
        stage_type = _validate_stage_type(str(raw_run.get("stage_type", "")))
        status = _validate_status(str(raw_run.get("status", "pending")))
        parents_raw = raw_run.get("parent_run_ids", [])
        if not isinstance(parents_raw, Sequence) or isinstance(parents_raw, str):
            raise PipelineManifestError(
                f"Run entry {run_id!r} parent_run_ids must be an array."
            )
        parents = [_validate_run_id(parent) for parent in parents_raw]
        for field_name in ("inputs", "outputs", "artifacts", "metadata"):
            field_value = raw_run.get(field_name, {})
            if not isinstance(field_value, Mapping):
                raise PipelineManifestError(
                    f"Run entry {run_id!r} field {field_name!r} must be an object."
                )
        run_dir = raw_run.get("run_dir")
        if run_dir is not None and not str(run_dir).strip():
            run_dir = None

        normalized_runs[run_id] = {
            "run_id": run_id,
            "stage_type": stage_type,
            "status": status,
            "display_name": str(raw_run.get("display_name", "")).strip() or None,
            "parent_run_ids": parents,
            "run_dir": str(run_dir).strip() if run_dir is not None else None,
            "created_at": str(raw_run.get("created_at", "")).strip() or None,
            "updated_at": str(raw_run.get("updated_at", "")).strip() or None,
            "inputs": _json_ready(dict(raw_run.get("inputs", {}))),
            "outputs": _json_ready(dict(raw_run.get("outputs", {}))),
            "artifacts": _json_ready(dict(raw_run.get("artifacts", {}))),
            "metadata": _json_ready(dict(raw_run.get("metadata", {}))),
        }

    return {
        "schema_version": PIPELINE_MANIFEST_SCHEMA_VERSION,
        "pipeline_id": pipeline_id,
        "pipeline_root": pipeline_root,
        "created_at": str(payload.get("created_at", "")).strip() or None,
        "updated_at": str(payload.get("updated_at", "")).strip() or None,
        "metadata": _json_ready(dict(metadata)),
        "runs": normalized_runs,
    }


def create_pipeline_manifest_file(
    manifest_path: str | Path,
    *,
    pipeline_id: str,
    pipeline_root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create an empty manifest file on disk.

    The manifest uses a generic run graph keyed by run id, which supports
    fan-out from one upstream stage into multiple downstream stages.
    """

    path = Path(manifest_path).resolve(strict=False)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Pipeline manifest already exists: {path}")

    manifest_root = path.parent.resolve(strict=False)
    resolved_pipeline_root = (
        Path(pipeline_root).resolve(strict=False)
        if pipeline_root is not None
        else manifest_root
    )
    now = _utc_now_iso()
    payload = {
        "schema_version": PIPELINE_MANIFEST_SCHEMA_VERSION,
        "pipeline_id": str(pipeline_id).strip(),
        "pipeline_root": _portable_path(
            resolved_pipeline_root,
            pipeline_root=manifest_root,
        ),
        "created_at": now,
        "updated_at": now,
        "metadata": _json_ready(dict(metadata or {})),
        "runs": {},
    }
    return save_pipeline_manifest(path, payload)


def upsert_pipeline_run(
    manifest_path: str | Path,
    *,
    run_id: str,
    stage_type: str,
    status: str,
    run_dir: str | Path | None = None,
    display_name: str | None = None,
    parent_run_ids: Sequence[str] | None = None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert or update a run node inside a manifest."""

    path = Path(manifest_path).resolve(strict=False)
    manifest = load_pipeline_manifest(path)
    pipeline_root = _resolve_pipeline_root(path, manifest)
    manifest_runs = dict(manifest["runs"])

    normalized_run_id = _validate_run_id(run_id)
    normalized_stage_type = _validate_stage_type(stage_type)
    normalized_status = _validate_status(status)
    existing = manifest_runs.get(normalized_run_id)

    if existing is not None and existing["stage_type"] != normalized_stage_type:
        raise PipelineManifestError(
            f"Run {normalized_run_id!r} already exists with stage_type "
            f"{existing['stage_type']!r}; got {normalized_stage_type!r}."
        )

    normalized_parents = (
        _validate_parents(parent_run_ids, manifest_runs=manifest_runs)
        if parent_run_ids is not None
        else list(existing.get("parent_run_ids", []))
        if existing
        else []
    )

    now = _utc_now_iso()
    created_at = existing.get("created_at") if existing else now
    next_run = {
        "run_id": normalized_run_id,
        "stage_type": normalized_stage_type,
        "status": normalized_status,
        "display_name": (
            str(display_name).strip()
            if display_name is not None
            else existing.get("display_name")
        )
        if existing is not None or display_name is not None
        else None,
        "parent_run_ids": normalized_parents,
        "run_dir": (
            _portable_path(run_dir, pipeline_root=pipeline_root)
            if run_dir is not None
            else existing.get("run_dir")
            if existing
            else None
        ),
        "created_at": created_at,
        "updated_at": now,
        "inputs": (
            _normalize_mapping_paths(inputs, pipeline_root=pipeline_root)
            if inputs is not None
            else existing.get("inputs", {})
            if existing
            else {}
        ),
        "outputs": (
            _normalize_mapping_paths(outputs, pipeline_root=pipeline_root)
            if outputs is not None
            else existing.get("outputs", {})
            if existing
            else {}
        ),
        "artifacts": (
            _normalize_mapping_paths(artifacts, pipeline_root=pipeline_root)
            if artifacts is not None
            else existing.get("artifacts", {})
            if existing
            else {}
        ),
        "metadata": (
            _normalize_mapping_paths(metadata, pipeline_root=pipeline_root)
            if metadata is not None
            else existing.get("metadata", {})
            if existing
            else {}
        ),
    }
    manifest_runs[normalized_run_id] = next_run
    manifest["runs"] = manifest_runs
    manifest["updated_at"] = now
    return save_pipeline_manifest(path, manifest)


def list_pipeline_runs(
    manifest: Mapping[str, Any],
    *,
    stage_type: str | None = None,
    parent_run_id: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """Return manifest runs filtered by stage, parent, and/or status."""

    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping):
        raise PipelineManifestError("Manifest runs must be an object.")

    normalized_status = _validate_status(status) if status is not None else None
    normalized_stage_type = (
        _validate_stage_type(stage_type) if stage_type is not None else None
    )
    normalized_parent = (
        _validate_run_id(parent_run_id) if parent_run_id is not None else None
    )

    filtered: list[dict[str, Any]] = []
    for run in runs.values():
        if not isinstance(run, Mapping):
            continue
        if (
            normalized_stage_type is not None
            and run.get("stage_type") != normalized_stage_type
        ):
            continue
        if normalized_status is not None and run.get("status") != normalized_status:
            continue
        if normalized_parent is not None and normalized_parent not in run.get(
            "parent_run_ids", []
        ):
            continue
        filtered.append(dict(run))
    filtered.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("run_id") or ""),
        )
    )
    return filtered


def get_pipeline_run(manifest: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """Return a manifest run entry by run id."""

    runs = manifest.get("runs", {})
    if not isinstance(runs, Mapping):
        raise PipelineManifestError("Manifest runs must be an object.")
    normalized_run_id = _validate_run_id(run_id)
    run = runs.get(normalized_run_id)
    if not isinstance(run, Mapping):
        raise PipelineManifestError(
            f"Run {normalized_run_id!r} was not found in the pipeline manifest."
        )
    return dict(run)
