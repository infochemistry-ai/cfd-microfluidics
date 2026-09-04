from __future__ import annotations

from pathlib import Path

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor import MeshQualityPolicy, evaluate_mesh_quality_gate


def _quality_mesh() -> ImportedTetraMesh:
    return ImportedTetraMesh(
        source_path=Path("bad.msh"),
        points=np.asarray(
            [[0, 0, 0], [100, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
            dtype=np.float64,
        ),
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        boundary_triangles=np.asarray([[0, 1, 2]], dtype=np.int64),
        boundary_face_tags=np.asarray([-1], dtype=np.int32),
        volume_tag_per_cell=np.asarray([10], dtype=np.int32),
        cell_centers=np.asarray([[25, 0.0025, 0.0025]], dtype=np.float64),
        cell_volumes=np.asarray([1e-21], dtype=np.float64),
        face_centers=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
        face_areas=np.asarray([0.0, 1.0], dtype=np.float64),
        boundary_tag_per_face=np.asarray([-1, 2], dtype=np.int32),
        boundary_face_indices=np.asarray([0, 1], dtype=np.int64),
        boundary_unresolved_faces=np.asarray([0], dtype=np.int64),
        diagnostics={"boundary_orientation": {"ambiguous_triangle_indices": [0]}},
    )


def test_quality_gate_reports_actionable_entity_ids() -> None:
    report = evaluate_mesh_quality_gate(
        _quality_mesh(),
        MeshQualityPolicy(
            min_cell_volume_m3=1e-18,
            min_face_area_m2=0.0,
            max_tetra_aspect_ratio=10.0,
        ),
    )

    assert report.is_acceptable is False
    assert report.total_counts == {
        "small_cell_volume": 1,
        "degenerate_face_area": 1,
        "high_tetra_aspect_ratio": 1,
        "ambiguous_boundary_orientation": 1,
        "unresolved_boundary_tag": 1,
    }
    by_code = {finding.code: finding for finding in report.findings}
    assert by_code["small_cell_volume"].entity_id == 0
    assert by_code["degenerate_face_area"].entity_id == 0
    assert by_code["high_tetra_aspect_ratio"].value > 10.0
    assert report.to_dict()["is_acceptable"] is False


def test_quality_gate_can_promote_warnings_and_truncate_details() -> None:
    mesh = _quality_mesh()
    mesh.cell_volumes[:] = 1.0
    mesh.face_areas[:] = 1.0
    mesh.boundary_unresolved_faces = np.zeros((0,), dtype=np.int64)
    mesh.diagnostics = {"boundary_orientation": {}}

    report = evaluate_mesh_quality_gate(
        mesh,
        MeshQualityPolicy(
            max_tetra_aspect_ratio=2.0,
            max_reported_findings=1,
            fail_on_warnings=True,
        ),
    )

    assert report.errors == ()
    assert len(report.warnings) == 1
    assert report.is_acceptable is False
    assert report.truncated is False


def test_quality_gate_accepts_warning_when_policy_allows_it() -> None:
    mesh = _quality_mesh()
    mesh.cell_volumes[:] = 1.0
    mesh.face_areas[:] = 1.0
    mesh.boundary_unresolved_faces = np.zeros((0,), dtype=np.int64)
    mesh.diagnostics = {"boundary_orientation": {}}

    report = evaluate_mesh_quality_gate(
        mesh,
        MeshQualityPolicy(max_tetra_aspect_ratio=2.0, fail_on_warnings=False),
    )

    assert report.warnings
    assert report.is_acceptable is True


def test_quality_gate_never_hides_errors_after_truncated_warnings() -> None:
    mesh = _quality_mesh()
    mesh.cell_volumes[:] = 1.0
    mesh.face_areas[:] = 1.0
    mesh.boundary_unresolved_faces = np.asarray([0], dtype=np.int64)
    mesh.diagnostics = {"boundary_orientation": {}}

    report = evaluate_mesh_quality_gate(
        mesh,
        MeshQualityPolicy(
            max_tetra_aspect_ratio=2.0,
            max_reported_findings=1,
            fail_on_warnings=False,
        ),
    )

    assert [finding.severity for finding in report.findings] == ["warning"]
    assert report.total_errors == 1
    assert report.total_warnings == 1
    assert report.truncated is True
    assert report.is_acceptable is False
