from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.preprocessor import (
    BoundaryConditionError,
    MaterialAssignmentError,
    apply_flow_profile_to_mesh,
    case_config_from_mapping,
    compile_flow_runtime_profile,
    compile_material_cell_assignment,
    compile_scalar_runtime_profile,
    compile_uniform_material_properties,
    scalar_profile_to_diffusion_kwargs,
)


def _mesh():
    return build_imported_tetra_mesh(
        source_path=Path("runtime.msh"),
        points=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        ),
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        boundary_triangles=np.asarray(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64
        ),
        boundary_face_tags=np.asarray([1, 2, 3, 3], dtype=np.int32),
        volume_tag_per_cell=np.asarray([10], dtype=np.int32),
        field_data={
            "inlet": np.asarray([1, 2]),
            "outlet": np.asarray([2, 2]),
            "walls": np.asarray([3, 2]),
            "fluid": np.asarray([10, 3]),
        },
    )


def _case(*, velocity_vector: bool = False, wall_mode: str = "no_slip"):
    inlet = (
        {"velocity_m_per_s": [1, 0, 0]}
        if velocity_vector
        else {"normal_speed_m_per_s": 0.2}
    )
    return case_config_from_mapping(
        {
            "schema_version": "case_config_v1",
            "case_id": "runtime",
            "mesh": {"path": "runtime.msh"},
            "zones": [
                {"id": "fluid", "kind": "volume", "physical_names": ["fluid"]},
                {"id": "inlet", "kind": "surface", "physical_names": ["inlet"]},
                {"id": "outlet", "kind": "surface", "physical_names": ["outlet"]},
                {"id": "walls", "kind": "surface", "physical_names": ["walls"]},
            ],
            "materials": [
                {
                    "id": "water",
                    "zone": "fluid",
                    "properties": {
                        "density_kg_per_m3": 998.0,
                        "kinematic_viscosity_m2_per_s": 1.1e-6,
                    },
                }
            ],
            "boundary_conditions": [
                {"id": "in", "zone": "inlet", "kind": "velocity_inlet", **inlet},
                {
                    "id": "out",
                    "zone": "outlet",
                    "kind": "pressure_outlet",
                    "pressure_pa": 12.0,
                },
                {
                    "id": "walls-flow",
                    "zone": "walls",
                    "kind": "wall",
                    "wall_mode": wall_mode,
                },
                {
                    "id": "scalar-in",
                    "zone": "inlet",
                    "kind": "dirichlet",
                    "field": "concentration",
                    "value": 1.0,
                },
                {
                    "id": "scalar-out",
                    "zone": "outlet",
                    "kind": "neumann",
                    "field": "concentration",
                    "flux": 0.25,
                },
                {
                    "id": "scalar-wall",
                    "zone": "walls",
                    "kind": "robin",
                    "field": "concentration",
                    "alpha": 2.0,
                    "beta": 3.0,
                    "gamma": 4.0,
                },
            ],
        }
    )


def test_flow_runtime_profile_applies_case_parameters_and_faces() -> None:
    mesh = _mesh()
    profile = compile_flow_runtime_profile(mesh, _case())

    assert profile.inlet_speed_m_per_s == 0.2
    assert profile.outlet_pressure_pa == 12.0
    assert profile.wall_mode == "no_slip"
    assert profile.density_kg_per_m3 == 998.0
    assert profile.kinematic_viscosity_m2_per_s == 1.1e-6
    assert apply_flow_profile_to_mesh(mesh, profile).wall_faces.size == 2
    assert compile_uniform_material_properties(_case())["density_kg_per_m3"] == 998.0
    assignment = compile_material_cell_assignment(mesh, _case())
    assert assignment.counts() == {"water": 1}
    assert assignment.material_index_per_cell.tolist() == [0]


def test_scalar_runtime_profile_compiles_dirichlet_neumann_and_robin() -> None:
    mesh = _mesh()
    profile = compile_scalar_runtime_profile(mesh, _case(), "concentration")

    assert set(profile.dirichlet_by_face.values()) == {1.0}
    assert set(profile.neumann_flux_by_face.values()) == {0.25}
    assert len(profile.robin_by_face) == 2
    assert next(iter(profile.robin_by_face.values())).beta == 3.0
    kwargs = scalar_profile_to_diffusion_kwargs(profile)
    assert kwargs["diffusion_boundary_dirichlet"] == profile.dirichlet_by_face
    assert set(kwargs["diffusion_boundary_robin"].values()) == {(2.0, 3.0, 4.0)}


def test_flow_runtime_rejects_vector_inlet_until_facewise_operator_exists() -> None:
    with pytest.raises(BoundaryConditionError, match="normal_speed_m_per_s"):
        compile_flow_runtime_profile(_mesh(), _case(velocity_vector=True))


def test_flow_runtime_accepts_pure_symmetry_as_slip_wall() -> None:
    case = _case(wall_mode="slip")
    conditions = tuple(
        item for item in case.boundary_conditions if item.id != "walls-flow"
    ) + (
        case.boundary_conditions[0].__class__(
            id="symmetry",
            zone="walls",
            kind="symmetry",
            parameters={},
        ),
    )
    case = case.__class__(
        schema_version=case.schema_version,
        case_id=case.case_id,
        mesh=case.mesh,
        zones=case.zones,
        materials=case.materials,
        boundary_conditions=conditions,
        mesh_quality=case.mesh_quality,
    )

    assert compile_flow_runtime_profile(_mesh(), case).wall_mode == "slip"


def test_flow_runtime_derives_kinematic_viscosity_from_dynamic_value() -> None:
    case = _case()
    material = case.materials[0]
    case = replace(
        case,
        materials=(
            replace(
                material,
                properties={
                    "density_kg_per_m3": 1000.0,
                    "dynamic_viscosity_pa_s": 1e-3,
                },
            ),
        ),
    )

    assert compile_flow_runtime_profile(
        _mesh(), case
    ).kinematic_viscosity_m2_per_s == pytest.approx(1e-6)


def test_material_assignment_rejects_overlapping_material_cells() -> None:
    case = _case()
    fluid = replace(case.zones[0], allow_overlap=True)
    duplicate = replace(fluid, id="fluid-copy")
    second_material = replace(case.materials[0], id="water-copy", zone="fluid-copy")
    case = replace(
        case,
        zones=(fluid, duplicate, *case.zones[1:]),
        materials=(case.materials[0], second_material),
    )

    with pytest.raises(MaterialAssignmentError, match="overlaps"):
        compile_material_cell_assignment(_mesh(), case)


def test_uniform_material_properties_reject_partial_property_coverage() -> None:
    case = _case()
    water = case.materials[0]
    oil = replace(
        water,
        id="oil",
        properties={"dynamic_viscosity_pa_s": 0.01},
    )
    case = replace(case, materials=(water, oil))

    with pytest.raises(BoundaryConditionError, match="defined for every material"):
        compile_uniform_material_properties(case)
