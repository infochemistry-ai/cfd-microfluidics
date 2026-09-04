from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import (
    _read_msh41_ascii_fallback,
    build_imported_tetra_mesh,
)
from microfluidics.gmsh.gmsh_mesh_validation import validate_imported_tetra_mesh


def test_build_imported_tetra_mesh_with_named_boundaries() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    boundary_tags = np.asarray([1, 2, 3, 3], dtype=np.int32)
    field_data = {
        "left_inlet": np.asarray([1, 2], dtype=np.int32),
        "outlet": np.asarray([2, 2], dtype=np.int32),
        "wall": np.asarray([3, 2], dtype=np.int32),
        "fluid": np.asarray([10, 3], dtype=np.int32),
    }

    mesh = build_imported_tetra_mesh(
        source_path="dummy.msh",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_tags,
        field_data=field_data,
        volume_tag_per_cell=np.asarray([10], dtype=np.int32),
    )
    report = validate_imported_tetra_mesh(mesh)

    assert report.is_valid
    assert mesh.tetrahedra.shape == (1, 4)
    assert mesh.face_vertices.shape[0] == 4
    assert mesh.boundary_face_indices.shape[0] == 4
    assert mesh.inlet_faces.shape[0] == 1
    assert mesh.outlet_faces.shape[0] == 1
    assert mesh.wall_faces.shape[0] == 2
    assert mesh.volume_tag_per_cell.tolist() == [10]
    assert report.summary["volume_tags"] == [10]
    assert float(mesh.cell_volumes[0]) > 0.0
    assert "tetra_quality" in mesh.diagnostics
    assert "boundary_orientation" in mesh.diagnostics
    assert mesh.diagnostics["boundary_semantic_counts"]["inlet"] == 1
    assert mesh.diagnostics["boundary_semantic_counts"]["outlet"] == 1
    assert mesh.diagnostics["boundary_semantic_counts"]["wall"] == 2
    assert report.summary["tetra_quality"]["invalid_tetra_count"] == 0


def test_unmatched_boundary_triangle_reported() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray([[0, 1, 4]], dtype=np.int64)
    boundary_tags = np.asarray([7], dtype=np.int32)
    field_data = {"wall": np.asarray([7, 2], dtype=np.int32)}

    mesh = build_imported_tetra_mesh(
        source_path="dummy.msh",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_tags,
        field_data=field_data,
    )
    assert int(mesh.diagnostics["unmatched_imported_boundary_triangles"]) == 1


def test_build_imported_tetra_mesh_rejects_misaligned_volume_tags() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="volume_tag_per_cell length"):
        build_imported_tetra_mesh(
            source_path="dummy.msh",
            points=points,
            tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            boundary_triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
            boundary_face_tags=np.asarray([1], dtype=np.int32),
            field_data={"wall": np.asarray([1, 2], dtype=np.int32)},
            volume_tag_per_cell=np.asarray([], dtype=np.int32),
        )


def test_msh41_fallback_preserves_same_tag_across_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "same-tag-different-dimensions.msh"
    path.write_text(
        """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "inlet"
3 1 "fluid"
$EndPhysicalNames
$Entities
0 0 1 1
1 0 0 0 1 1 0 1 1 0
1 0 0 0 1 1 1 1 1 0
$EndEntities
$Nodes
1 4 1 4
3 1 0 4
1
2
3
4
0 0 0
1 0 0
0 1 0
0 0 1
$EndNodes
$Elements
2 2 1 2
2 1 2 1
1 1 2 3
3 1 4 1
2 1 2 3 4
$EndElements
""",
        encoding="utf-8",
    )

    _, _, volume_tags, _, boundary_tags, field_data = _read_msh41_ascii_fallback(path)

    assert field_data["inlet"].tolist() == [1, 2]
    assert field_data["fluid"].tolist() == [1, 3]
    assert boundary_tags.tolist() == [1]
    assert volume_tags.tolist() == [1]
