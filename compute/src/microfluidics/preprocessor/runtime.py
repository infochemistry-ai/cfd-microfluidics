"""Compile case contracts into profiles consumed by the current CFD runtimes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor.boundary import (
    CompiledBoundaryConditions,
    PeriodicFacePair,
    compile_boundary_conditions,
)
from microfluidics.preprocessor.errors import (
    BoundaryConditionError,
    MaterialAssignmentError,
)
from microfluidics.preprocessor.models import CaseConfigV1
from microfluidics.preprocessor.zones import ResolvedCaseZones, resolve_case_zones


@dataclass(frozen=True, slots=True)
class FlowRuntimeProfile:
    inlet_faces: np.ndarray
    outlet_faces: np.ndarray
    wall_faces: np.ndarray
    inlet_speed_m_per_s: float
    outlet_pressure_pa: float
    wall_mode: str
    density_kg_per_m3: float | None
    kinematic_viscosity_m2_per_s: float | None


@dataclass(frozen=True, slots=True)
class RobinBoundaryValue:
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True, slots=True)
class ScalarRuntimeProfile:
    field: str
    dirichlet_by_face: dict[int, float]
    neumann_flux_by_face: dict[int, float]
    robin_by_face: dict[int, RobinBoundaryValue]
    periodic_pairs: tuple[PeriodicFacePair, ...]


@dataclass(frozen=True, slots=True)
class MaterialCellAssignment:
    material_ids: tuple[str, ...]
    material_index_per_cell: np.ndarray
    properties_by_cell: dict[str, np.ndarray]

    def counts(self) -> dict[str, int]:
        return {
            material_id: int(np.count_nonzero(self.material_index_per_cell == index))
            for index, material_id in enumerate(self.material_ids)
        }


def scalar_profile_to_diffusion_kwargs(
    profile: ScalarRuntimeProfile,
) -> dict[str, object]:
    """Translate a scalar case profile to ``build_scalar_backend_precompute`` args."""

    return {
        "diffusion_boundary_dirichlet": dict(profile.dirichlet_by_face),
        "diffusion_boundary_neumann": dict(profile.neumann_flux_by_face),
        "diffusion_boundary_robin": {
            face: (value.alpha, value.beta, value.gamma)
            for face, value in profile.robin_by_face.items()
        },
        "diffusion_periodic_face_pairs": tuple(
            (pair.source_face, pair.target_face) for pair in profile.periodic_pairs
        ),
    }


def _uniform(values: list[float], label: str) -> float:
    if not values:
        raise BoundaryConditionError(f"case does not define {label}")
    first = float(values[0])
    if not all(np.isclose(first, value, rtol=0.0, atol=1e-14) for value in values[1:]):
        raise BoundaryConditionError(
            f"current homogeneous runtime requires one uniform {label}"
        )
    return first


def _uniform_material_property(case: CaseConfigV1, property_name: str) -> float | None:
    materials_with_property = [
        material for material in case.materials if property_name in material.properties
    ]
    if not materials_with_property:
        return None
    if len(materials_with_property) != len(case.materials):
        missing = [
            material.id
            for material in case.materials
            if property_name not in material.properties
        ]
        raise BoundaryConditionError(
            f"material property {property_name} must be defined for every material; "
            f"missing on {missing}"
        )
    values = [
        float(material.properties[property_name])
        for material in materials_with_property
    ]
    return _uniform(values, f"material property {property_name}")


def compile_uniform_material_properties(case: CaseConfigV1) -> dict[str, float]:
    """Compile properties for runtimes that currently support one homogeneous fluid."""

    property_names = {
        name for material in case.materials for name in material.properties
    }
    return {
        name: value
        for name in sorted(property_names)
        if (value := _uniform_material_property(case, name)) is not None
    }


def compile_material_cell_assignment(
    mesh: ImportedTetraMesh,
    case: CaseConfigV1,
    *,
    resolved_zones: ResolvedCaseZones | None = None,
) -> MaterialCellAssignment:
    """Map every tetrahedron to exactly one configured material."""

    if not case.materials:
        raise MaterialAssignmentError("case does not define materials")
    zones = resolved_zones or resolve_case_zones(mesh, case)
    n_cells = int(mesh.tetrahedra.shape[0])
    material_index = np.full(n_cells, -1, dtype=np.int32)
    property_names = sorted(
        {name for material in case.materials for name in material.properties}
    )
    properties = {
        name: np.full(n_cells, np.nan, dtype=np.float64) for name in property_names
    }
    for index, material in enumerate(case.materials):
        cells = np.asarray(zones.zones[material.zone].entity_indices, dtype=np.int64)
        overlap = cells[material_index[cells] >= 0]
        if overlap.size:
            raise MaterialAssignmentError(
                f"material {material.id!r} overlaps another material on cells "
                f"{overlap[:20].tolist()}"
            )
        material_index[cells] = index
        for name, value in material.properties.items():
            properties[name][cells] = float(value)
    unassigned = np.flatnonzero(material_index < 0).astype(np.int64)
    if unassigned.size:
        raise MaterialAssignmentError(
            f"material assignment leaves cells unassigned: {unassigned[:20].tolist()}"
        )
    return MaterialCellAssignment(
        material_ids=tuple(material.id for material in case.materials),
        material_index_per_cell=material_index,
        properties_by_cell=properties,
    )


def compile_flow_runtime_profile(
    mesh: ImportedTetraMesh,
    case: CaseConfigV1,
    *,
    compiled: CompiledBoundaryConditions | None = None,
) -> FlowRuntimeProfile:
    """Compile flow BCs to the homogeneous profile supported by the solver."""

    boundaries = compiled or compile_boundary_conditions(mesh, case)
    velocity_conditions = boundaries.by_kind("velocity_inlet")
    pressure_conditions = boundaries.by_kind("pressure_outlet")
    wall_conditions = boundaries.by_kind("wall")
    symmetry_conditions = boundaries.by_kind("symmetry")
    if boundaries.by_kind("periodic"):
        raise BoundaryConditionError(
            "periodic flow BC is validated by the preprocessor but is not supported "
            "by the current pressure/velocity runtime"
        )
    inlet_speeds: list[float] = []
    for condition in velocity_conditions:
        if "velocity_m_per_s" in condition.parameters:
            raise BoundaryConditionError(
                "current flow runtime accepts normal_speed_m_per_s; vector velocity "
                "profiles require a face-wise inlet operator"
            )
        inlet_speeds.append(float(condition.parameters["normal_speed_m_per_s"]))
    inlet_speed = _uniform(inlet_speeds, "inlet normal speed")
    outlet_pressure = _uniform(
        [float(item.parameters["pressure_pa"]) for item in pressure_conditions],
        "outlet pressure",
    )
    wall_modes = {str(item.parameters["wall_mode"]) for item in wall_conditions}
    if symmetry_conditions:
        wall_modes.add("slip")
    if not wall_modes:
        raise BoundaryConditionError("case does not define wall or symmetry BCs")
    if len(wall_modes) != 1:
        raise BoundaryConditionError(
            "current flow runtime cannot mix slip/symmetry and no-slip walls in one run"
        )
    inlet_faces = np.unique(
        np.concatenate([item.face_indices for item in velocity_conditions])
    ).astype(np.int64)
    outlet_faces = np.unique(
        np.concatenate([item.face_indices for item in pressure_conditions])
    ).astype(np.int64)
    wall_sources = [item.face_indices for item in wall_conditions] + [
        item.face_indices for item in symmetry_conditions
    ]
    wall_faces = np.unique(np.concatenate(wall_sources)).astype(np.int64)
    density = _uniform_material_property(case, "density_kg_per_m3")
    kinematic_viscosity = _uniform_material_property(
        case, "kinematic_viscosity_m2_per_s"
    )
    dynamic_viscosity = _uniform_material_property(case, "dynamic_viscosity_pa_s")
    if kinematic_viscosity is None and dynamic_viscosity is not None:
        if density is None:
            raise BoundaryConditionError(
                "dynamic viscosity requires density_kg_per_m3 to derive kinematic viscosity"
            )
        kinematic_viscosity = dynamic_viscosity / density
    elif (
        kinematic_viscosity is not None
        and dynamic_viscosity is not None
        and density is not None
        and not np.isclose(
            dynamic_viscosity,
            density * kinematic_viscosity,
            rtol=1e-8,
            atol=0.0,
        )
    ):
        raise BoundaryConditionError(
            "dynamic and kinematic viscosity are inconsistent with density"
        )
    return FlowRuntimeProfile(
        inlet_faces=inlet_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        inlet_speed_m_per_s=inlet_speed,
        outlet_pressure_pa=outlet_pressure,
        wall_mode=next(iter(wall_modes)),
        density_kg_per_m3=density,
        kinematic_viscosity_m2_per_s=kinematic_viscosity,
    )


def apply_flow_profile_to_mesh(
    mesh: ImportedTetraMesh, profile: FlowRuntimeProfile
) -> ImportedTetraMesh:
    """Apply explicit case semantics while preserving the imported topology."""

    mesh.inlet_faces = np.asarray(profile.inlet_faces, dtype=np.int64)
    mesh.outlet_faces = np.asarray(profile.outlet_faces, dtype=np.int64)
    mesh.wall_faces = np.asarray(profile.wall_faces, dtype=np.int64)
    return mesh


def compile_scalar_runtime_profile(
    mesh: ImportedTetraMesh,
    case: CaseConfigV1,
    field: str,
    *,
    compiled: CompiledBoundaryConditions | None = None,
) -> ScalarRuntimeProfile:
    boundaries = compiled or compile_boundary_conditions(mesh, case)
    dirichlet: dict[int, float] = {}
    neumann: dict[int, float] = {}
    robin: dict[int, RobinBoundaryValue] = {}
    periodic: list[PeriodicFacePair] = []
    for condition in boundaries.conditions:
        condition_field = condition.parameters.get("field")
        if condition.kind == "periodic":
            periodic.extend(condition.periodic_pairs)
            continue
        if condition_field != field:
            continue
        if condition.kind == "dirichlet":
            for face in condition.face_indices.tolist():
                dirichlet[int(face)] = float(condition.parameters["value"])
        elif condition.kind == "neumann":
            for face in condition.face_indices.tolist():
                neumann[int(face)] = float(condition.parameters["flux"])
        elif condition.kind == "robin":
            value = RobinBoundaryValue(
                alpha=float(condition.parameters["alpha"]),
                beta=float(condition.parameters["beta"]),
                gamma=float(condition.parameters["gamma"]),
            )
            for face in condition.face_indices.tolist():
                robin[int(face)] = value
    return ScalarRuntimeProfile(
        field=field,
        dirichlet_by_face=dirichlet,
        neumann_flux_by_face=neumann,
        robin_by_face=robin,
        periodic_pairs=tuple(periodic),
    )
