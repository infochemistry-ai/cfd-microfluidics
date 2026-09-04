"""Import pipeline for external Gmsh tetrahedral meshes via meshio."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import (
    GmshBoundaryClassification,
    ImportedTetraMesh,
)

def _load_meshio():
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError(
            "meshio is required for Gmsh import. Install it with `pip install meshio`."
        ) from exc
    return meshio


def _cell_data_for_block(mesh, key: str, block_idx: int) -> np.ndarray | None:
    entries = mesh.cell_data.get(key)
    if entries is None or block_idx >= len(entries):
        return None
    return np.asarray(entries[block_idx])


def _extract_cells_from_meshio(
    mesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tetra_blocks: list[np.ndarray] = []
    tetra_tags: list[np.ndarray] = []
    tri_blocks: list[np.ndarray] = []
    tri_tags: list[np.ndarray] = []

    for block_idx, block in enumerate(mesh.cells):
        cell_type = str(block.type).lower()
        data = np.asarray(block.data, dtype=np.int64)
        if data.size == 0:
            continue
        physical = _cell_data_for_block(mesh, "gmsh:physical", block_idx)

        if cell_type.startswith("tetra"):
            tetra_blocks.append(np.asarray(data[:, :4], dtype=np.int64))
            if physical is None:
                tetra_tags.append(np.full((data.shape[0],), -1, dtype=np.int32))
            else:
                tetra_tags.append(np.asarray(physical, dtype=np.int32))
            continue

        if cell_type.startswith("triangle"):
            tri_blocks.append(np.asarray(data[:, :3], dtype=np.int64))
            if physical is None:
                tri_tags.append(np.full((data.shape[0],), -1, dtype=np.int32))
            else:
                tri_tags.append(np.asarray(physical, dtype=np.int32))

    if not tetra_blocks:
        raise ValueError("No tetrahedral cells found in Gmsh mesh.")
    if not tri_blocks:
        raise ValueError("No triangular boundary faces found in Gmsh mesh.")

    tetrahedra = np.concatenate(tetra_blocks, axis=0).astype(np.int64, copy=False)
    volume_tag_per_cell = np.concatenate(tetra_tags, axis=0).astype(
        np.int32, copy=False
    )
    boundary_triangles = np.concatenate(tri_blocks, axis=0).astype(np.int64, copy=False)
    boundary_face_tags = np.concatenate(tri_tags, axis=0).astype(np.int32, copy=False)
    return tetrahedra, volume_tag_per_cell, boundary_triangles, boundary_face_tags


def _canonical_triangle(face_vertices: Iterable[int]) -> tuple[int, int, int]:
    a, b, c = sorted(int(v) for v in face_vertices)
    return a, b, c


def _tetra_local_faces(nodes: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    a, b, c, d = (int(nodes[0]), int(nodes[1]), int(nodes[2]), int(nodes[3]))
    # Deterministic local order. Final normal orientation is fixed later.
    return ((b, c, d), (a, d, c), (a, b, d), (a, c, b))


def _build_face_topology(
    tetrahedra: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[tuple[int, int, int], int]
]:
    face_map: Dict[tuple[int, int, int], int] = {}
    face_vertices: list[np.ndarray] = []
    face_to_cells: list[list[int]] = []
    cell_to_faces = np.full((tetrahedra.shape[0], 4), -1, dtype=np.int64)

    for cell_idx, tet in enumerate(tetrahedra):
        for local_idx, oriented_face in enumerate(_tetra_local_faces(tet)):
            key = _canonical_triangle(oriented_face)
            face_idx = face_map.get(key)
            if face_idx is None:
                face_idx = len(face_vertices)
                face_map[key] = face_idx
                face_vertices.append(np.asarray(oriented_face, dtype=np.int64))
                face_to_cells.append([cell_idx, -1])
            elif face_to_cells[face_idx][1] == -1:
                face_to_cells[face_idx][1] = cell_idx
            cell_to_faces[cell_idx, local_idx] = face_idx

    face_vertices_arr = np.asarray(face_vertices, dtype=np.int64)
    face_to_cells_arr = np.asarray(face_to_cells, dtype=np.int64)
    interior_face_indices = np.flatnonzero(face_to_cells_arr[:, 1] >= 0).astype(
        np.int64
    )
    return (
        face_vertices_arr,
        face_to_cells_arr,
        cell_to_faces,
        interior_face_indices,
        face_map,
    )


def _compute_cell_geometry(
    points: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tetra_pts = points[tetrahedra]
    cell_centers = np.mean(tetra_pts, axis=1)
    edge1 = tetra_pts[:, 1, :] - tetra_pts[:, 0, :]
    edge2 = tetra_pts[:, 2, :] - tetra_pts[:, 0, :]
    edge3 = tetra_pts[:, 3, :] - tetra_pts[:, 0, :]
    signed6 = np.einsum("ij,ij->i", np.cross(edge1, edge2), edge3)
    cell_volumes = np.abs(signed6) / 6.0
    return cell_centers.astype(np.float64), cell_volumes.astype(np.float64)


def _compute_face_geometry(
    points: np.ndarray,
    face_vertices: np.ndarray,
    face_to_cells: np.ndarray,
    cell_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = points[face_vertices[:, 0]]
    p1 = points[face_vertices[:, 1]]
    p2 = points[face_vertices[:, 2]]
    face_centers = (p0 + p1 + p2) / 3.0
    raw_normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(raw_normal, axis=1)
    safe_norm = np.where(norm > 0.0, norm, 1.0)
    unit = raw_normal / safe_norm[:, None]
    face_areas = 0.5 * norm

    for face_idx in range(face_vertices.shape[0]):
        c0 = int(face_to_cells[face_idx, 0])
        c1 = int(face_to_cells[face_idx, 1])
        if c1 >= 0:
            direction = cell_centers[c1] - cell_centers[c0]
        else:
            direction = face_centers[face_idx] - cell_centers[c0]
        if float(np.dot(unit[face_idx], direction)) < 0.0:
            unit[face_idx] *= -1.0

    return (
        face_centers.astype(np.float64),
        face_areas.astype(np.float64),
        unit.astype(np.float64),
    )


def _tag_to_name_from_field_data(field_data: Dict[str, np.ndarray]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for name, values in field_data.items():
        arr = np.asarray(values).reshape(-1)
        if arr.size < 2:
            continue
        physical_tag = int(arr[0])
        dimension = int(arr[1])
        if dimension == 2:
            mapping[physical_tag] = str(name)
    return mapping


def _normalize_boundary_name(name: str) -> str:
    cleaned = name.strip().lower()
    for token in (" ", "-", ".", "/"):
        cleaned = cleaned.replace(token, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _semantic_from_name(normalized: str) -> str:
    if "outlet" in normalized or "outflow" in normalized:
        return "outlet"
    if "inlet" in normalized or "inflow" in normalized:
        return "inlet"
    if "wall" in normalized or "cavity" in normalized:
        return "wall"
    return "unresolved"


def _classify_boundary_faces(
    boundary_face_indices: np.ndarray,
    boundary_tag_per_face: np.ndarray,
    tag_to_name: Dict[int, str],
) -> GmshBoundaryClassification:
    inlet: list[int] = []
    outlet: list[int] = []
    wall: list[int] = []
    unresolved: list[int] = []
    tag_counts: Dict[int, int] = {}
    name_counts: Dict[str, int] = {}
    semantic_counts: Dict[str, int] = {
        "inlet": 0,
        "outlet": 0,
        "wall": 0,
        "unresolved": 0,
    }
    tag_semantic_map: Dict[str, str] = {}
    notes: list[str] = []

    for face_idx in boundary_face_indices.tolist():
        tag = int(boundary_tag_per_face[face_idx])
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        name = tag_to_name.get(tag)
        if name is None:
            unresolved.append(face_idx)
            semantic_counts["unresolved"] += 1
            continue

        normalized = _normalize_boundary_name(name)
        name_counts[normalized] = name_counts.get(normalized, 0) + 1
        semantic = _semantic_from_name(normalized)
        tag_semantic_map[str(tag)] = semantic
        semantic_counts[semantic] += 1

        if semantic == "inlet":
            inlet.append(face_idx)
        elif semantic == "outlet":
            outlet.append(face_idx)
        elif semantic == "wall":
            wall.append(face_idx)
        else:
            unresolved.append(face_idx)

    missing = [key for key in ("inlet", "outlet", "wall") if semantic_counts[key] == 0]
    if missing:
        notes.append(
            "Boundary classes not resolved from physical names for: "
            + ", ".join(missing)
        )
    if unresolved:
        notes.append(
            f"Unresolved boundary faces: {len(unresolved)}. "
            "Check physical names/tags in the .msh file."
        )
    if not tag_to_name:
        notes.append(
            "No 2D physical names found in field_data. "
            "Boundary tags cannot be mapped by semantic name."
        )

    notes.append(f"Tag-to-semantic mapping used: {tag_semantic_map}")
    return GmshBoundaryClassification(
        inlet_face_indices=np.asarray(sorted(inlet), dtype=np.int64),
        outlet_face_indices=np.asarray(sorted(outlet), dtype=np.int64),
        wall_face_indices=np.asarray(sorted(wall), dtype=np.int64),
        unresolved_face_indices=np.asarray(sorted(unresolved), dtype=np.int64),
        tag_to_name=dict(tag_to_name),
        tag_counts=tag_counts,
        name_counts=name_counts,
        semantic_counts=semantic_counts,
        tag_semantic_map=tag_semantic_map,
        notes=tuple(notes),
    )


def _boundary_tag_per_reconstructed_face(
    boundary_triangles: np.ndarray,
    boundary_face_tags: np.ndarray,
    face_map: Dict[tuple[int, int, int], int],
    face_count: int,
) -> tuple[np.ndarray, int]:
    boundary_tag_per_face = np.full((face_count,), -1, dtype=np.int32)
    unmatched = 0
    for tri, tag in zip(boundary_triangles, boundary_face_tags, strict=False):
        face_idx = face_map.get(_canonical_triangle(tri))
        if face_idx is None:
            unmatched += 1
            continue
        boundary_tag_per_face[face_idx] = int(tag)
    return boundary_tag_per_face, unmatched


def _reorient_imported_boundary_triangles(
    *,
    points: np.ndarray,
    boundary_triangles: np.ndarray,
    face_map: Dict[tuple[int, int, int], int],
    face_to_cells: np.ndarray,
    cell_centers: np.ndarray,
    eps: float = 1e-14,
) -> tuple[np.ndarray, Dict[str, object]]:
    reoriented = 0
    ambiguous = 0
    reoriented_indices: list[int] = []
    ambiguous_indices: list[int] = []
    oriented = np.array(boundary_triangles, copy=True)

    for tri_idx, tri in enumerate(oriented):
        face_idx = face_map.get(_canonical_triangle(tri))
        if face_idx is None:
            ambiguous += 1
            ambiguous_indices.append(int(tri_idx))
            continue

        c0 = int(face_to_cells[face_idx, 0])
        c1 = int(face_to_cells[face_idx, 1])
        if c0 < 0 or c1 >= 0:
            ambiguous += 1
            ambiguous_indices.append(int(tri_idx))
            continue

        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        n_norm = float(np.linalg.norm(normal))
        if n_norm <= eps:
            ambiguous += 1
            ambiguous_indices.append(int(tri_idx))
            continue
        center = (p0 + p1 + p2) / 3.0
        outward_hint = center - cell_centers[c0]
        dot = float(np.dot(normal, outward_hint))
        if abs(dot) <= eps:
            ambiguous += 1
            ambiguous_indices.append(int(tri_idx))
            continue
        if dot < 0.0:
            oriented[tri_idx, 1], oriented[tri_idx, 2] = (
                oriented[tri_idx, 2],
                oriented[tri_idx, 1],
            )
            reoriented += 1
            reoriented_indices.append(int(tri_idx))

    return oriented, {
        "reoriented_count": int(reoriented),
        "ambiguous_count": int(ambiguous),
        "ambiguous_triangle_indices": ambiguous_indices,
        "total_boundary_triangles": int(boundary_triangles.shape[0]),
        "reoriented_triangle_indices": reoriented_indices,
    }


def _histogram(
    values: np.ndarray, bins: int = 10
) -> Dict[str, list[float] | list[int]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "bin_edges": [float(v) for v in edges.tolist()],
        "counts": [int(v) for v in counts.tolist()],
    }


def _tetra_quality_audit(
    *,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    cell_volumes: np.ndarray,
    face_areas: np.ndarray,
) -> Dict[str, object]:
    tetra_pts = points[tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_arrays = [
        np.linalg.norm(tetra_pts[:, b, :] - tetra_pts[:, a, :], axis=1)
        for a, b in edge_pairs
    ]
    edge_stack = np.stack(edge_arrays, axis=1)
    edge_lengths = edge_stack.reshape(-1)
    per_tet_max = np.max(edge_stack, axis=1)
    per_tet_min = np.min(edge_stack, axis=1)
    eps = 1e-16
    aspect_ratio = per_tet_max / np.maximum(per_tet_min, eps)
    squared_edge_sum = np.sum(edge_stack * edge_stack, axis=1)
    # This normalized volume metric is 1 for a regular tetrahedron and tends
    # to 0 as a sliver collapses, even when all edge lengths remain comparable.
    mean_ratio = np.divide(
        12.0 * np.power(3.0 * cell_volumes, 2.0 / 3.0),
        squared_edge_sum,
        out=np.zeros_like(cell_volumes),
        where=squared_edge_sum > 0.0,
    )
    mean_ratio = np.clip(mean_ratio, 0.0, 1.0)

    volume_ref = float(np.median(cell_volumes)) if cell_volumes.size else 0.0
    near_zero_threshold = max(1e-18, volume_ref * 1e-6)
    invalid_count = int(np.count_nonzero(cell_volumes <= 0.0))
    near_zero_count = int(np.count_nonzero(cell_volumes <= near_zero_threshold))
    bad_aspect_threshold = 12.0
    bad_aspect_count = int(np.count_nonzero(aspect_ratio > bad_aspect_threshold))
    invalid_indices = np.flatnonzero(cell_volumes <= 0.0).astype(np.int64)
    near_zero_indices = np.flatnonzero(cell_volumes <= near_zero_threshold).astype(
        np.int64
    )
    bad_aspect_indices = np.flatnonzero(aspect_ratio > bad_aspect_threshold).astype(
        np.int64
    )
    zero_area_indices = np.flatnonzero(face_areas <= 0.0).astype(np.int64)

    warnings: list[str] = []
    if invalid_count > 0:
        warnings.append(f"Invalid tetra count (<=0 volume): {invalid_count}")
    if near_zero_count > 0:
        warnings.append(
            f"Near-zero tetra count (<= {near_zero_threshold:.3e}): {near_zero_count}"
        )
    if bad_aspect_count > 0:
        warnings.append(
            f"High-aspect tetra count (> {bad_aspect_threshold:.1f}): {bad_aspect_count}"
        )

    return {
        "cell_volume_stats": {
            "min": float(np.min(cell_volumes)),
            "max": float(np.max(cell_volumes)),
            "mean": float(np.mean(cell_volumes)),
        },
        "edge_length_stats": {
            "min": float(np.min(edge_lengths)),
            "max": float(np.max(edge_lengths)),
            "mean": float(np.mean(edge_lengths)),
        },
        "face_area_stats": {
            "min": float(np.min(face_areas)),
            "max": float(np.max(face_areas)),
            "mean": float(np.mean(face_areas)),
        },
        "tetra_aspect_ratio_proxy": {
            "min": float(np.min(aspect_ratio)),
            "max": float(np.max(aspect_ratio)),
            "mean": float(np.mean(aspect_ratio)),
            "bad_threshold": float(bad_aspect_threshold),
            "bad_count": int(bad_aspect_count),
        },
        "tetra_mean_ratio": {
            "min": float(np.min(mean_ratio)),
            "max": float(np.max(mean_ratio)),
            "mean": float(np.mean(mean_ratio)),
        },
        "invalid_tetra_count": int(invalid_count),
        "invalid_tetra_indices": invalid_indices.tolist(),
        "near_zero_tetra_count": int(near_zero_count),
        "near_zero_tetra_indices": near_zero_indices.tolist(),
        "near_zero_volume_threshold": float(near_zero_threshold),
        "high_aspect_tetra_indices": bad_aspect_indices.tolist(),
        "zero_area_face_count": int(zero_area_indices.size),
        "zero_area_face_indices": zero_area_indices.tolist(),
        "histograms": {
            "cell_volume": _histogram(cell_volumes, bins=12),
            "edge_length": _histogram(edge_lengths, bins=12),
            "tetra_aspect_ratio": _histogram(aspect_ratio, bins=12),
        },
        "quality_warnings": warnings,
    }


def _mesh_scale_audit(
    *,
    points: np.ndarray,
    source_path: Path,
) -> Dict[str, object]:
    pts = np.asarray(points, dtype=np.float64)
    bbox_min = np.min(pts, axis=0)
    bbox_max = np.max(pts, axis=0)
    bbox_extent = bbox_max - bbox_min
    positive_extent = bbox_extent[np.abs(bbox_extent) > 1e-30]
    aspect_ratio = (
        float(np.max(positive_extent) / np.min(positive_extent))
        if positive_extent.size
        else 0.0
    )
    warnings: list[str] = []
    if not np.all(np.isfinite(bbox_extent)):
        warnings.append("Non-finite bounding-box extents detected.")
    if positive_extent.size != bbox_extent.size:
        warnings.append("At least one mesh extent is effectively zero.")

    return {
        "bbox_min": [float(v) for v in bbox_min.tolist()],
        "bbox_max": [float(v) for v in bbox_max.tolist()],
        "bbox_extents": [float(v) for v in bbox_extent.tolist()],
        "bbox_aspect_ratio_max_over_min": float(aspect_ratio),
        "warnings": warnings,
    }


def build_imported_tetra_mesh(
    *,
    source_path: Path,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    boundary_triangles: np.ndarray,
    boundary_face_tags: np.ndarray,
    field_data: Dict[str, np.ndarray],
    volume_tag_per_cell: np.ndarray | None = None,
) -> ImportedTetraMesh:
    """Build internal tetra mesh contract from raw arrays."""

    source_path = Path(source_path)
    pts = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    btris = np.asarray(boundary_triangles, dtype=np.int64)
    btags = np.asarray(boundary_face_tags, dtype=np.int32)
    volume_tags = (
        np.full((tets.shape[0],), -1, dtype=np.int32)
        if volume_tag_per_cell is None
        else np.asarray(volume_tag_per_cell, dtype=np.int32).reshape(-1)
    )
    if volume_tags.shape[0] != tets.shape[0]:
        raise ValueError("volume_tag_per_cell length must match tetrahedra row count.")

    (
        face_vertices,
        face_to_cells,
        cell_to_faces,
        interior_face_indices,
        face_map,
    ) = _build_face_topology(tets)
    boundary_face_indices = np.flatnonzero(face_to_cells[:, 1] < 0).astype(np.int64)
    cell_centers, cell_volumes = _compute_cell_geometry(pts, tets)
    face_centers, face_areas, face_normals = _compute_face_geometry(
        pts,
        face_vertices,
        face_to_cells,
        cell_centers,
    )

    oriented_btris, orientation_diag = _reorient_imported_boundary_triangles(
        points=pts,
        boundary_triangles=btris,
        face_map=face_map,
        face_to_cells=face_to_cells,
        cell_centers=cell_centers,
    )
    boundary_tag_per_face, unmatched_boundary_triangles = (
        _boundary_tag_per_reconstructed_face(
            oriented_btris,
            btags,
            face_map,
            face_vertices.shape[0],
        )
    )
    tag_to_name = _tag_to_name_from_field_data(field_data)
    classification = _classify_boundary_faces(
        boundary_face_indices,
        boundary_tag_per_face,
        tag_to_name,
    )
    quality = _tetra_quality_audit(
        points=pts,
        tetrahedra=tets,
        cell_volumes=cell_volumes,
        face_areas=face_areas,
    )
    scale_audit = _mesh_scale_audit(points=pts, source_path=source_path)

    diagnostics = {
        "node_count": int(pts.shape[0]),
        "tetra_count": int(tets.shape[0]),
        "boundary_triangle_count_imported": int(btris.shape[0]),
        "face_count_reconstructed": int(face_vertices.shape[0]),
        "interior_face_count": int(interior_face_indices.shape[0]),
        "boundary_face_count_reconstructed": int(boundary_face_indices.shape[0]),
        "unmatched_imported_boundary_triangles": int(unmatched_boundary_triangles),
        "boundary_orientation": orientation_diag,
        "boundary_tags_found": sorted(int(v) for v in np.unique(btags).tolist()),
        "volume_tags_found": sorted(
            int(v) for v in np.unique(volume_tags).tolist() if int(v) >= 0
        ),
        "boundary_tag_counts": classification.tag_counts,
        "boundary_name_counts": classification.name_counts,
        "boundary_semantic_counts": classification.semantic_counts,
        "boundary_tag_to_name": classification.tag_to_name,
        "boundary_tag_semantic_map": classification.tag_semantic_map,
        "classification_notes": list(classification.notes),
        "tetra_quality": quality,
        "mesh_scale_audit": scale_audit,
    }

    physical_groups = {
        key: tuple(np.asarray(values, dtype=np.int32).tolist())
        for key, values in field_data.items()
    }
    return ImportedTetraMesh(
        source_path=source_path,
        points=pts,
        tetrahedra=tets,
        boundary_triangles=oriented_btris,
        boundary_face_tags=btags,
        volume_tag_per_cell=volume_tags,
        physical_groups=physical_groups,
        boundary_face_names=classification.tag_to_name,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        face_vertices=face_vertices,
        face_centers=face_centers,
        face_areas=face_areas,
        face_normals=face_normals,
        face_to_cells=face_to_cells,
        cell_to_faces=cell_to_faces,
        boundary_tag_per_face=boundary_tag_per_face,
        interior_face_indices=interior_face_indices,
        boundary_face_indices=boundary_face_indices,
        inlet_faces=classification.inlet_face_indices,
        outlet_faces=classification.outlet_face_indices,
        wall_faces=classification.wall_face_indices,
        boundary_unresolved_faces=classification.unresolved_face_indices,
        diagnostics=diagnostics,
    )


def import_gmsh_tetra_mesh(msh_path: str | Path) -> ImportedTetraMesh:
    """Read a `.msh` file and build internal tetrahedral mesh contract."""
    path = Path(msh_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Gmsh mesh file not found: {path}")

    meshio = _load_meshio()
    try:
        raw_mesh = meshio.read(path)
        points = np.asarray(raw_mesh.points, dtype=np.float64)
        (
            tetrahedra,
            volume_tag_per_cell,
            boundary_triangles,
            boundary_face_tags,
        ) = _extract_cells_from_meshio(raw_mesh)
        field_data = {
            key: np.asarray(value) for key, value in raw_mesh.field_data.items()
        }
    except ValueError as exc:
        message = str(exc)
        if "Incompatible cell data 'gmsh:physical'" not in message:
            raise
        (
            points,
            tetrahedra,
            volume_tag_per_cell,
            boundary_triangles,
            boundary_face_tags,
            field_data,
        ) = _read_msh41_ascii_fallback(path)

    return build_imported_tetra_mesh(
        source_path=path,
        points=points,
        tetrahedra=tetrahedra,
        volume_tag_per_cell=volume_tag_per_cell,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _read_msh41_ascii_fallback(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    """Read ASCII MSH 4.1 file when meshio fails on gmsh:physical block mismatch."""

    lines = path.read_text(encoding="utf-8").splitlines()
    physical_names: list[tuple[int, str, int]] = []
    surface_entity_to_phys: dict[int, int] = {}
    volume_entity_to_phys: dict[int, int] = {}

    def _seek(section: str) -> int:
        needle = f"${section}"
        for i, line in enumerate(lines):
            if line.strip() == needle:
                return i
        raise ValueError(f"Section {needle} not found in {path}")

    # PhysicalNames
    p0 = _seek("PhysicalNames")
    n_phys = int(lines[p0 + 1].strip())
    for i in range(n_phys):
        parts = lines[p0 + 2 + i].strip().split(maxsplit=2)
        dim = int(parts[0])
        tag = int(parts[1])
        name = parts[2].strip().strip('"')
        physical_names.append((tag, name, dim))

    # Entities
    e0 = _seek("Entities")
    header = [int(v) for v in lines[e0 + 1].strip().split()]
    n_points, n_curves, n_surfaces, n_volumes = header[:4]
    idx = e0 + 2 + n_points + n_curves
    # surfaces
    for _ in range(n_surfaces):
        parts = lines[idx].strip().split()
        idx += 1
        tag = int(parts[0])
        # tag + 6 bbox + numPhysicalTags ...
        n_phys_tags = int(parts[7])
        phys_tag = int(parts[8]) if n_phys_tags > 0 else 0
        surface_entity_to_phys[tag] = phys_tag
    # volumes
    for _ in range(n_volumes):
        parts = lines[idx].strip().split()
        idx += 1
        tag = int(parts[0])
        n_phys_tags = int(parts[7])
        phys_tag = int(parts[8]) if n_phys_tags > 0 else 0
        volume_entity_to_phys[tag] = phys_tag

    # Nodes
    n0 = _seek("Nodes")
    nb, _, _, _ = [int(v) for v in lines[n0 + 1].strip().split()]
    node_coords: dict[int, tuple[float, float, float]] = {}
    idx = n0 + 2
    for _ in range(nb):
        entity_dim, entity_tag, parametric, n_in_block = [
            int(v) for v in lines[idx].strip().split()
        ]
        idx += 1
        tags = [int(lines[idx + j].strip()) for j in range(n_in_block)]
        idx += n_in_block
        for j in range(n_in_block):
            xyz = [float(v) for v in lines[idx + j].strip().split()[:3]]
            node_coords[tags[j]] = (xyz[0], xyz[1], xyz[2])
        idx += n_in_block

    sorted_tags = sorted(node_coords.keys())
    tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
    points = np.asarray([node_coords[tag] for tag in sorted_tags], dtype=np.float64)

    # Elements
    el0 = _seek("Elements")
    nb, _, _, _ = [int(v) for v in lines[el0 + 1].strip().split()]
    idx = el0 + 2
    tetra: list[list[int]] = []
    tetra_tags: list[int] = []
    btri: list[list[int]] = []
    btags: list[int] = []
    for _ in range(nb):
        entity_dim, entity_tag, elem_type, n_in_block = [
            int(v) for v in lines[idx].strip().split()
        ]
        idx += 1
        for _ in range(n_in_block):
            parts = lines[idx].strip().split()
            idx += 1
            conn = [tag_to_idx[int(v)] for v in parts[1:]]
            if elem_type == 4 and entity_dim == 3 and len(conn) == 4:
                tetra.append(conn)
                tetra_tags.append(int(volume_entity_to_phys.get(entity_tag, -1)))
            elif elem_type == 2 and entity_dim == 2 and len(conn) == 3:
                btri.append(conn)
                btags.append(int(surface_entity_to_phys.get(entity_tag, 0)))

    field_data: dict[str, np.ndarray] = {}
    for tag, name, dim in physical_names:
        field_data[name] = np.asarray([int(tag), int(dim)], dtype=np.int32)

    return (
        points,
        np.asarray(tetra, dtype=np.int64),
        np.asarray(tetra_tags, dtype=np.int32),
        np.asarray(btri, dtype=np.int64),
        np.asarray(btags, dtype=np.int32),
        field_data,
    )
