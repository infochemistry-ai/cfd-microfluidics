"""Loader for imported Gmsh tetra meshes saved as `*_imported_mesh.npz` artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh


def _read_text_array(npz: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in npz:
        return None
    value = npz[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _load_json_from_npz(npz: np.lib.npyio.NpzFile, key: str) -> Dict[str, Any]:
    payload = _read_text_array(npz, key)
    if payload is None or payload.strip() == "":
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_int_dict(mapping: Dict[str, Any]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for key, value in mapping.items():
        try:
            int_key = int(key)
        except (TypeError, ValueError):
            continue
        result[int_key] = str(value)
    return result


def _coerce_physical_groups(mapping: Dict[str, Any]) -> Dict[str, tuple[int, int]]:
    result: Dict[str, tuple[int, int]] = {}
    for name, raw_value in mapping.items():
        if not isinstance(raw_value, (list, tuple)) or len(raw_value) < 2:
            continue
        try:
            result[str(name)] = (int(raw_value[0]), int(raw_value[1]))
        except (TypeError, ValueError):
            continue
    return result


def _load_summary_fallback(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_boundary_names(
    npz: np.lib.npyio.NpzFile,
    summary_payload: dict,
) -> Dict[int, str]:
    from_npz = _coerce_int_dict(_load_json_from_npz(npz, "boundary_face_names_json"))
    if from_npz:
        return from_npz
    groups = summary_payload.get("boundary_groups", {})
    if not isinstance(groups, dict):
        return {}
    tag_to_name = groups.get("tag_to_name", {})
    if not isinstance(tag_to_name, dict):
        return {}
    return _coerce_int_dict(tag_to_name)


def _resolve_physical_groups(
    npz: np.lib.npyio.NpzFile,
    summary_payload: dict,
) -> Dict[str, tuple[int, int]]:
    from_npz = _coerce_physical_groups(_load_json_from_npz(npz, "physical_groups_json"))
    if from_npz:
        return from_npz
    names = summary_payload.get("boundary_groups", {}).get("physical_group_names", [])
    if not isinstance(names, list):
        return {}
    return {str(name): (-1, 2) for name in names}


def _resolve_diagnostics(
    npz: np.lib.npyio.NpzFile, summary_payload: dict
) -> Dict[str, Any]:
    from_npz = _load_json_from_npz(npz, "diagnostics_json")
    if from_npz:
        return from_npz
    summary = summary_payload.get("summary", {})
    if isinstance(summary, dict):
        diagnostics = summary.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            return diagnostics
    return {}


def _resolve_source_path(npz: np.lib.npyio.NpzFile, npz_path: Path) -> Path:
    text = _read_text_array(npz, "source_path")
    if text:
        return Path(text)
    return npz_path


def load_imported_tetra_mesh_npz(
    npz_path: str | Path,
    *,
    summary_path: str | Path | None = None,
) -> ImportedTetraMesh:
    """Load a mesh produced by `run_import_gmsh_mesh.py` and reconstruct the mesh dataclass."""

    path = Path(npz_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Imported mesh npz file not found: {path}")

    resolved_summary = (
        Path(summary_path).resolve() if summary_path else path.parent / "summary.json"
    )
    summary_payload = _load_summary_fallback(resolved_summary)

    with np.load(path, allow_pickle=False) as npz:
        boundary_face_names = _resolve_boundary_names(npz, summary_payload)
        physical_groups = _resolve_physical_groups(npz, summary_payload)
        diagnostics = _resolve_diagnostics(npz, summary_payload)
        case_config = _load_json_from_npz(npz, "case_config_json")
        preprocessor_metadata = _load_json_from_npz(npz, "preprocessor_metadata_json")
        source_path = _resolve_source_path(npz, path)

        unresolved_key = (
            "unresolved_faces"
            if "unresolved_faces" in npz
            else "boundary_unresolved_faces"
        )
        if unresolved_key not in npz:
            unresolved_faces = np.zeros((0,), dtype=np.int64)
        else:
            unresolved_faces = np.asarray(npz[unresolved_key], dtype=np.int64)

        return ImportedTetraMesh(
            source_path=source_path,
            points=np.asarray(npz["points"], dtype=np.float64),
            tetrahedra=np.asarray(npz["tetrahedra"], dtype=np.int64),
            boundary_triangles=np.asarray(npz["boundary_triangles"], dtype=np.int64),
            boundary_face_tags=np.asarray(npz["boundary_face_tags"], dtype=np.int32),
            volume_tag_per_cell=(
                np.asarray(npz["volume_tag_per_cell"], dtype=np.int32)
                if "volume_tag_per_cell" in npz
                else np.full(
                    (np.asarray(npz["tetrahedra"]).shape[0],),
                    -1,
                    dtype=np.int32,
                )
            ),
            physical_groups=physical_groups,
            boundary_face_names=boundary_face_names,
            case_config=case_config,
            preprocessor_metadata=preprocessor_metadata,
            cell_centers=np.asarray(npz["cell_centers"], dtype=np.float64),
            cell_volumes=np.asarray(npz["cell_volumes"], dtype=np.float64),
            face_vertices=np.asarray(npz["face_vertices"], dtype=np.int64),
            face_centers=np.asarray(npz["face_centers"], dtype=np.float64),
            face_areas=np.asarray(npz["face_areas"], dtype=np.float64),
            face_normals=np.asarray(npz["face_normals"], dtype=np.float64),
            face_to_cells=np.asarray(npz["face_to_cells"], dtype=np.int64),
            cell_to_faces=np.asarray(npz["cell_to_faces"], dtype=np.int64),
            boundary_tag_per_face=np.asarray(
                npz["boundary_tag_per_face"], dtype=np.int32
            ),
            interior_face_indices=np.asarray(
                npz["interior_face_indices"], dtype=np.int64
            ),
            boundary_face_indices=np.asarray(
                npz["boundary_face_indices"], dtype=np.int64
            ),
            inlet_faces=np.asarray(npz["inlet_faces"], dtype=np.int64),
            outlet_faces=np.asarray(npz["outlet_faces"], dtype=np.int64),
            wall_faces=np.asarray(npz["wall_faces"], dtype=np.int64),
            boundary_unresolved_faces=unresolved_faces,
            diagnostics=diagnostics,
        )
