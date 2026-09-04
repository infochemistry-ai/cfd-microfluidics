"""Typed data structures for the Gmsh tetrahedral mesh import pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np


def _empty_int64() -> np.ndarray:
    return np.zeros((0,), dtype=np.int64)


def _empty_int32() -> np.ndarray:
    return np.zeros((0,), dtype=np.int32)


@dataclass(frozen=True)
class GmshBoundaryClassification:
    """Boundary face classification resolved from physical names/tags."""

    inlet_face_indices: np.ndarray
    outlet_face_indices: np.ndarray
    wall_face_indices: np.ndarray
    unresolved_face_indices: np.ndarray
    tag_to_name: Dict[int, str] = field(default_factory=dict)
    tag_counts: Dict[int, int] = field(default_factory=dict)
    name_counts: Dict[str, int] = field(default_factory=dict)
    semantic_counts: Dict[str, int] = field(default_factory=dict)
    tag_semantic_map: Dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass
class ImportedTetraMesh:
    """Internal tetrahedral mesh format for future solver integration.

    Notes:
    - `boundary_triangles` / `boundary_face_tags` come from imported Gmsh surface elements.
    - `volume_tag_per_cell` is aligned with `tetrahedra` and contains the Gmsh
      physical volume tag for each tetrahedron, or `-1` when the source mesh did
      not provide one.
    - `face_*` topology/metrics are reconstructed from tetra connectivity.
    - `boundary_tag_per_face` is aligned with reconstructed faces (`face_vertices`) and may contain
      `-1` for unresolved/untagged boundary faces.
    """

    source_path: Path
    points: np.ndarray
    tetrahedra: np.ndarray
    boundary_triangles: np.ndarray
    boundary_face_tags: np.ndarray
    volume_tag_per_cell: np.ndarray = field(default_factory=_empty_int32)
    physical_groups: Dict[str, tuple[int, int]] = field(default_factory=dict)
    boundary_face_names: Dict[int, str] = field(default_factory=dict)
    case_config: Dict[str, object] = field(default_factory=dict)
    preprocessor_metadata: Dict[str, object] = field(default_factory=dict)

    cell_centers: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    cell_volumes: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )

    face_vertices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.int64)
    )
    face_centers: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    face_areas: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )
    face_normals: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64)
    )
    face_to_cells: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64)
    )
    cell_to_faces: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 4), dtype=np.int64)
    )

    boundary_tag_per_face: np.ndarray = field(default_factory=_empty_int32)
    interior_face_indices: np.ndarray = field(default_factory=_empty_int64)
    boundary_face_indices: np.ndarray = field(default_factory=_empty_int64)
    inlet_faces: np.ndarray = field(default_factory=_empty_int64)
    outlet_faces: np.ndarray = field(default_factory=_empty_int64)
    wall_faces: np.ndarray = field(default_factory=_empty_int64)
    boundary_unresolved_faces: np.ndarray = field(default_factory=_empty_int64)

    diagnostics: Dict[str, object] = field(default_factory=dict)
