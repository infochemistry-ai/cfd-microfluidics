from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor import (
    BoundaryConditionError,
    ZoneResolutionError,
    case_config_from_mapping,
    compile_boundary_conditions,
    pair_periodic_faces,
    resolve_case_zones,
)


def _tagged_tetra_mesh() -> ImportedTetraMesh:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return build_imported_tetra_mesh(
        source_path=Path("dummy.msh"),
        points=points,
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        boundary_triangles=np.asarray(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64
        ),
        boundary_face_tags=np.asarray([1, 2, 3, 4], dtype=np.int32),
        volume_tag_per_cell=np.asarray([10], dtype=np.int32),
        field_data={
            "inlet": np.asarray([1, 2], dtype=np.int32),
            "outlet": np.asarray([2, 2], dtype=np.int32),
            "wall_a": np.asarray([3, 2], dtype=np.int32),
            "wall_b": np.asarray([4, 2], dtype=np.int32),
            "fluid": np.asarray([10, 3], dtype=np.int32),
        },
    )


def _case(*, overlap: bool = False):
    return case_config_from_mapping(
        {
            "schema_version": "case_config_v1",
            "case_id": "zones",
            "mesh": {"path": "dummy.msh"},
            "zones": [
                {
                    "id": "fluid",
                    "kind": "volume",
                    "physical_names": ["fluid"],
                },
                {
                    "id": "inlet",
                    "kind": "surface",
                    "physical_names": ["inlet"],
                    "allow_overlap": overlap,
                },
                {
                    "id": "outlet",
                    "kind": "surface",
                    "physical_tags": [2],
                },
                {
                    "id": "walls",
                    "kind": "surface",
                    "physical_names": ["wall_a", "wall_b"],
                },
            ],
            "materials": [
                {
                    "id": "water",
                    "zone": "fluid",
                    "properties": {"density_kg_per_m3": 1000},
                }
            ],
            "boundary_conditions": [
                {
                    "id": "flow-in",
                    "zone": "inlet",
                    "kind": "velocity_inlet",
                    "normal_speed_m_per_s": 0.1,
                },
                {
                    "id": "flow-out",
                    "zone": "outlet",
                    "kind": "pressure_outlet",
                    "pressure_pa": 0,
                },
                {
                    "id": "flow-wall",
                    "zone": "walls",
                    "kind": "wall",
                    "wall_mode": "no_slip",
                },
                {
                    "id": "thermal-wall",
                    "zone": "walls",
                    "kind": "neumann",
                    "field": "temperature",
                    "flux": 0,
                },
            ],
        }
    )


def test_resolve_case_zones_maps_surface_and_volume_entities() -> None:
    resolved = resolve_case_zones(_tagged_tetra_mesh(), _case())

    assert resolved.zones["fluid"].entity_indices.tolist() == [0]
    assert resolved.zones["inlet"].physical_tags == (1,)
    assert resolved.zones["outlet"].entity_indices.size == 1
    assert resolved.zones["walls"].entity_indices.size == 2
    assert resolved.unassigned_boundary_faces.size == 0
    assert resolved.unassigned_cells.size == 0


def test_zone_resolution_rejects_unknown_name() -> None:
    payload = {
        "schema_version": "case_config_v1",
        "case_id": "bad",
        "mesh": {"path": "dummy.msh"},
        "zones": [{"id": "missing", "kind": "surface", "physical_names": ["missing"]}],
        "boundary_conditions": [
            {"id": "wall", "zone": "missing", "kind": "wall", "wall_mode": "slip"}
        ],
    }
    case = case_config_from_mapping(payload)

    with pytest.raises(ZoneResolutionError, match="unknown physical name"):
        resolve_case_zones(_tagged_tetra_mesh(), case)


def test_zone_resolution_rejects_dimension_mismatch() -> None:
    payload = {
        "schema_version": "case_config_v1",
        "case_id": "bad-dim",
        "mesh": {"path": "dummy.msh"},
        "zones": [{"id": "wrong", "kind": "surface", "physical_names": ["fluid"]}],
        "boundary_conditions": [
            {"id": "wall", "zone": "wrong", "kind": "wall", "wall_mode": "slip"}
        ],
    }

    with pytest.raises(ZoneResolutionError, match="Gmsh dimension 3"):
        resolve_case_zones(_tagged_tetra_mesh(), case_config_from_mapping(payload))


def test_zone_resolution_rejects_accidental_overlap() -> None:
    payload = {
        "schema_version": "case_config_v1",
        "case_id": "overlap",
        "mesh": {"path": "dummy.msh"},
        "zones": [
            {"id": "a", "kind": "surface", "physical_tags": [1]},
            {"id": "b", "kind": "surface", "physical_names": ["inlet"]},
        ],
        "boundary_conditions": [
            {"id": "a-bc", "zone": "a", "kind": "wall", "wall_mode": "slip"},
            {"id": "b-bc", "zone": "b", "kind": "symmetry"},
        ],
    }

    with pytest.raises(ZoneResolutionError, match="overlap"):
        resolve_case_zones(_tagged_tetra_mesh(), case_config_from_mapping(payload))


def test_compile_boundary_conditions_allows_different_fields_on_same_faces() -> None:
    compiled = compile_boundary_conditions(_tagged_tetra_mesh(), _case())

    assert len(compiled.conditions) == 4
    assert compiled.by_kind("wall")[0].face_indices.size == 2
    assert compiled.by_kind("neumann")[0].parameters["field"] == "temperature"


def test_compile_boundary_conditions_rejects_flow_conflict() -> None:
    case = _case()
    conditions = list(case.boundary_conditions)
    conditions.append(
        conditions[0].__class__(
            id="duplicate-flow",
            zone="inlet",
            kind="symmetry",
            parameters={},
        )
    )
    conflicting = case.__class__(
        schema_version=case.schema_version,
        case_id=case.case_id,
        mesh=case.mesh,
        zones=case.zones,
        materials=case.materials,
        boundary_conditions=tuple(conditions),
        mesh_quality=case.mesh_quality,
    )

    with pytest.raises(BoundaryConditionError, match="conflict"):
        compile_boundary_conditions(_tagged_tetra_mesh(), conflicting)


def _periodic_mesh() -> ImportedTetraMesh:
    return ImportedTetraMesh(
        source_path=Path("periodic.msh"),
        points=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
        tetrahedra=np.asarray([[0, 0, 0, 0]], dtype=np.int64),
        boundary_triangles=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        boundary_face_tags=np.asarray([1, 2], dtype=np.int32),
        volume_tag_per_cell=np.asarray([10], dtype=np.int32),
        physical_groups={"periodic_a": (1, 2), "periodic_b": (2, 2)},
        face_vertices=np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
        face_centers=np.asarray([[0, 0.5, 0.5], [1, 0.5, 0.5]], dtype=np.float64),
        face_areas=np.asarray([1.0, 1.0], dtype=np.float64),
        face_normals=np.asarray([[-1, 0, 0], [1, 0, 0]], dtype=np.float64),
        face_to_cells=np.asarray([[0, -1], [0, -1]], dtype=np.int64),
        boundary_tag_per_face=np.asarray([1, 2], dtype=np.int32),
        boundary_face_indices=np.asarray([0, 1], dtype=np.int64),
    )


def test_pair_periodic_faces_builds_translational_bijection() -> None:
    pairs = pair_periodic_faces(
        _periodic_mesh(),
        np.asarray([0]),
        np.asarray([1]),
        translation_m=(1.0, 0.0, 0.0),
    )

    assert [(pair.source_face, pair.target_face) for pair in pairs] == [(0, 1)]


def test_pair_periodic_faces_rejects_non_opposite_normals() -> None:
    mesh = _periodic_mesh()
    mesh.face_normals[1] = [-1, 0, 0]

    with pytest.raises(BoundaryConditionError, match="opposite normals"):
        pair_periodic_faces(mesh, np.asarray([0]), np.asarray([1]))


def test_periodic_faces_reject_other_scalar_boundary_conditions() -> None:
    case = case_config_from_mapping(
        {
            "schema_version": "case_config_v1",
            "case_id": "periodic-conflict",
            "mesh": {"path": "periodic.msh"},
            "zones": [
                {"id": "periodic-a", "kind": "surface", "physical_tags": [1]},
                {"id": "periodic-b", "kind": "surface", "physical_tags": [2]},
            ],
            "boundary_conditions": [
                {
                    "id": "periodic",
                    "zone": "periodic-a",
                    "kind": "periodic",
                    "paired_zone": "periodic-b",
                    "translation_m": [1.0, 0.0, 0.0],
                },
                {
                    "id": "temperature",
                    "zone": "periodic-b",
                    "kind": "dirichlet",
                    "field": "temperature",
                    "value": 300.0,
                },
            ],
        }
    )

    with pytest.raises(BoundaryConditionError, match="periodic face"):
        compile_boundary_conditions(_periodic_mesh(), case)
