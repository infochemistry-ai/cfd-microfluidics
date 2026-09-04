from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_sha256(raw: object, *, label: str) -> str:
    value = str(raw).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value


def _load_json_mapping(path: Path, *, label: str) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _resolve_reference_path(raw_value: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        # Flow summaries may retain the original run directory. A staged
        # artifact set keeps the fixed basenames next
        # to the summary, so make that portable layout usable downstream.
        portable = (base_dir / candidate.name).resolve()
        if portable.exists():
            return portable
        return resolved
    return (base_dir / candidate).resolve()


def load_flow_coupling_payload(flow_run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(flow_run_dir).resolve()
    metadata_path = run_dir / "flow_coupling_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"flow coupling metadata not found: {metadata_path}")
    metadata = _load_json_mapping(metadata_path, label="flow coupling metadata")

    flux_path = run_dir / "final_corrected_face_flux.npy"
    if not flux_path.exists():
        legacy_flux = run_dir / "corrected_face_flux.npy"
        if legacy_flux.exists():
            flux_path = legacy_flux
        else:
            raise FileNotFoundError(
                f"missing final corrected face flux file: {flux_path}"
            )

    cell_velocity_path = run_dir / "final_cell_velocity.npy"
    pressure_path = run_dir / "final_pressure.npy"
    face_to_cells_path = run_dir / "face_to_cells.npy"
    cell_volumes_path = run_dir / "cell_volumes.npy"
    face_groups_path = run_dir / "face_groups.npy"

    return {
        "flow_run_dir": str(run_dir),
        "metadata_path": str(metadata_path),
        "flux_path": str(flux_path),
        "metadata": metadata,
        "face_flux": np.asarray(
            np.load(flux_path, allow_pickle=False), dtype=np.float64
        ),
        "cell_velocity": (
            np.asarray(
                np.load(cell_velocity_path, allow_pickle=False), dtype=np.float64
            )
            if cell_velocity_path.exists()
            else None
        ),
        "pressure": (
            np.asarray(np.load(pressure_path, allow_pickle=False), dtype=np.float64)
            if pressure_path.exists()
            else None
        ),
        "face_to_cells": (
            np.asarray(np.load(face_to_cells_path, allow_pickle=False), dtype=np.int64)
            if face_to_cells_path.exists()
            else None
        ),
        "cell_volumes": (
            np.asarray(np.load(cell_volumes_path, allow_pickle=False), dtype=np.float64)
            if cell_volumes_path.exists()
            else None
        ),
        "face_groups": (
            np.asarray(np.load(face_groups_path, allow_pickle=False), dtype=np.int32)
            if face_groups_path.exists()
            else None
        ),
    }


def load_flow_summary_payload(
    flow_summary_json: str | Path,
    *,
    require_artifact_hashes: bool = False,
) -> Dict[str, Any]:
    summary_path = Path(flow_summary_json).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"flow summary json not found: {summary_path}")
    summary = _load_json_mapping(summary_path, label="flow summary json")
    artifacts = (
        dict(summary.get("artifacts", {}))
        if isinstance(summary.get("artifacts", {}), dict)
        else {}
    )

    metadata_raw = str(artifacts.get("flow_coupling_metadata_json", "")).strip()
    if metadata_raw:
        metadata_path = _resolve_reference_path(
            metadata_raw, base_dir=summary_path.parent
        )
        run_dir = metadata_path.parent
    else:
        run_dir = summary_path.parent
        metadata_path = run_dir / "flow_coupling_metadata.json"

    payload = load_flow_coupling_payload(run_dir)
    payload["summary_path"] = str(summary_path)
    payload["summary"] = summary
    payload["resolved_metadata_path"] = str(metadata_path.resolve())
    if require_artifact_hashes:
        loaded_metadata_path = Path(str(payload["metadata_path"])).resolve()
        if metadata_path.resolve() != loaded_metadata_path:
            raise ValueError(
                "flow summary metadata reference does not resolve to the loaded "
                "flow_coupling_metadata.json"
            )
        recorded_hashes = summary.get("flow_coupling_artifact_sha256")
        if not isinstance(recorded_hashes, Mapping):
            raise ValueError(
                "flow summary lacks required flow_coupling_artifact_sha256 provenance"
            )
        artifact_paths = {
            "flow_coupling_metadata_json": loaded_metadata_path,
            "final_corrected_face_flux_npy": Path(str(payload["flux_path"])).resolve(),
            "face_to_cells_npy": loaded_metadata_path.parent / "face_to_cells.npy",
            "cell_volumes_npy": loaded_metadata_path.parent / "cell_volumes.npy",
        }
        verified_hashes: dict[str, str] = {}
        for artifact_name, artifact_path in artifact_paths.items():
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    f"required flow coupling artifact not found: {artifact_path}"
                )
            expected = _recorded_sha256(
                recorded_hashes.get(artifact_name),
                label=f"flow_coupling_artifact_sha256.{artifact_name}",
            )
            actual = sha256_file(artifact_path)
            if actual != expected:
                raise ValueError(
                    f"flow coupling artifact SHA-256 mismatch for {artifact_name}: "
                    f"expected {expected}, got {actual}"
                )
            verified_hashes[artifact_name] = actual
        payload["verified_artifact_sha256"] = verified_hashes
    return payload


def check_flow_mesh_compatibility(
    *,
    mesh: Any,
    flow_payload: Mapping[str, Any],
    mesh_sha256: str | None = None,
    require_topology: bool = False,
) -> Dict[str, Any]:
    metadata = dict(flow_payload.get("metadata", {}))
    mesh_stats = (
        dict(metadata.get("mesh_stats", {}))
        if isinstance(metadata.get("mesh_stats", {}), dict)
        else {}
    )
    face_flux = np.asarray(
        flow_payload.get("face_flux", np.zeros((0,), dtype=np.float64)),
        dtype=np.float64,
    )
    face_to_cells_flow = flow_payload.get("face_to_cells")
    cell_volumes_flow = flow_payload.get("cell_volumes")
    summary = (
        dict(flow_payload.get("summary", {}))
        if isinstance(flow_payload.get("summary", {}), Mapping)
        else {}
    )

    mesh_hash_required = mesh_sha256 is not None
    recorded_mesh_hash_raw = summary.get("mesh_sha256")
    metadata_mesh_hash_raw = metadata.get("mesh_sha256")
    mesh_hash_present = bool(
        isinstance(recorded_mesh_hash_raw, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", recorded_mesh_hash_raw.strip())
    )
    mesh_hash_ok = bool(
        not mesh_hash_required
        or (
            mesh_hash_present
            and str(recorded_mesh_hash_raw).strip().lower()
            == str(mesh_sha256).strip().lower()
        )
    )
    metadata_mesh_hash_present = bool(
        isinstance(metadata_mesh_hash_raw, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", metadata_mesh_hash_raw.strip())
    )
    metadata_mesh_hash_ok = bool(
        not mesh_hash_required
        or (
            metadata_mesh_hash_present
            and str(metadata_mesh_hash_raw).strip().lower()
            == str(mesh_sha256).strip().lower()
        )
    )

    face_count_ok = int(face_flux.shape[0]) == int(mesh.face_vertices.shape[0])
    cell_count_ok = int(mesh.tetrahedra.shape[0]) == int(
        mesh_stats.get("tetra_count", mesh.tetrahedra.shape[0])
    )
    metadata_face_count_ok = int(mesh.face_vertices.shape[0]) == int(
        mesh_stats.get("face_count", mesh.face_vertices.shape[0])
    )
    metadata_node_count_ok = int(mesh.points.shape[0]) == int(
        mesh_stats.get("node_count", mesh.points.shape[0])
    )

    if isinstance(face_to_cells_flow, np.ndarray):
        face_to_cells_shape_ok = tuple(face_to_cells_flow.shape) == tuple(
            np.asarray(mesh.face_to_cells).shape
        )
        face_to_cells_equal = bool(
            face_to_cells_shape_ok
            and np.array_equal(face_to_cells_flow, np.asarray(mesh.face_to_cells))
        )
    else:
        face_to_cells_shape_ok = not require_topology
        face_to_cells_equal = not require_topology

    if isinstance(cell_volumes_flow, np.ndarray):
        cell_volumes_shape_ok = tuple(cell_volumes_flow.shape) == tuple(
            np.asarray(mesh.cell_volumes).shape
        )
        cell_volumes_close = bool(
            cell_volumes_shape_ok
            and np.allclose(
                cell_volumes_flow,
                np.asarray(mesh.cell_volumes, dtype=np.float64),
                rtol=1e-9,
                atol=1e-18,
            )
        )
    else:
        cell_volumes_shape_ok = not require_topology
        cell_volumes_close = not require_topology

    compatible = bool(
        mesh_hash_ok
        and metadata_mesh_hash_ok
        and face_count_ok
        and cell_count_ok
        and metadata_face_count_ok
        and metadata_node_count_ok
        and face_to_cells_shape_ok
        and face_to_cells_equal
        and cell_volumes_shape_ok
        and cell_volumes_close
    )
    return {
        "compatible": bool(compatible),
        "mesh_hash_required": bool(mesh_hash_required),
        "mesh_hash_present": bool(mesh_hash_present),
        "mesh_hash_ok": bool(mesh_hash_ok),
        "metadata_mesh_hash_present": bool(metadata_mesh_hash_present),
        "metadata_mesh_hash_ok": bool(metadata_mesh_hash_ok),
        "face_count_ok": bool(face_count_ok),
        "cell_count_ok": bool(cell_count_ok),
        "metadata_face_count_ok": bool(metadata_face_count_ok),
        "metadata_node_count_ok": bool(metadata_node_count_ok),
        "face_to_cells_shape_ok": bool(face_to_cells_shape_ok),
        "face_to_cells_equal": bool(face_to_cells_equal),
        "cell_volumes_shape_ok": bool(cell_volumes_shape_ok),
        "cell_volumes_close": bool(cell_volumes_close),
    }


def validate_source_flow_readiness(
    metadata: Mapping[str, Any],
    *,
    strict: bool,
    strict_label: str = "source flow",
) -> Dict[str, Any]:
    metadata_dict = dict(metadata)
    stage_status = (
        dict(metadata_dict.get("stage_status", {}))
        if isinstance(metadata_dict.get("stage_status", {}), dict)
        else {}
    )
    if "ready_for_next_stage" in stage_status:
        ready = bool(stage_status.get("ready_for_next_stage", False))
    elif "ready_for_next_stage" in metadata_dict:
        ready = bool(metadata_dict.get("ready_for_next_stage", False))
    else:
        ready = bool(metadata_dict.get("ready_for_flow_to_transport_coupling", False))

    nonphysical_fix = bool(metadata_dict.get("nonphysical_flux_fix_used", False))
    convective_auto_damping_used_any = bool(
        metadata_dict.get("convective_auto_damping_used_any", False)
    )
    reason = str(
        stage_status.get(
            "stage_status_reason",
            metadata_dict.get("stage_status_reason", ""),
        )
    )
    inconsistent_ready_reasons: list[str] = []
    if ready and nonphysical_fix:
        inconsistent_ready_reasons.append("nonphysical_flux_fix_used")
    if ready and convective_auto_damping_used_any:
        inconsistent_ready_reasons.append("convective_auto_damping_used_any")
    if inconsistent_ready_reasons:
        ready = False
        if not reason:
            reason = "ready metadata contradicts debug/nonphysical flags: " + ", ".join(
                inconsistent_ready_reasons
            )
    if not ready and strict:
        raise RuntimeError(
            f"{strict_label} readiness is false."
            + (f" reason={reason}" if reason else "")
        )
    return {
        "ready": bool(ready),
        "nonphysical_flux_fix_used": bool(nonphysical_fix),
        "convective_auto_damping_used_any": bool(convective_auto_damping_used_any),
        "run_completed": bool(
            stage_status.get("run_completed", metadata_dict.get("run_completed", False))
        ),
        "numerically_stable": bool(
            stage_status.get(
                "numerically_stable",
                metadata_dict.get("numerically_stable", False),
            )
        ),
        "physically_ready": bool(
            stage_status.get(
                "physically_ready",
                metadata_dict.get("physically_ready", False),
            )
        ),
        "ready_for_long_run": bool(
            stage_status.get(
                "ready_for_long_run",
                metadata_dict.get("ready_for_long_run", False),
            )
        ),
        "stage_status_reason": str(reason),
        "strict_enforced": bool(strict),
        "inconsistent_ready_reasons": inconsistent_ready_reasons,
        "warning": bool(not ready),
    }
