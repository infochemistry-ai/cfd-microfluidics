"""Compile solver-neutral boundary conditions onto concrete mesh faces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor.errors import BoundaryConditionError
from microfluidics.preprocessor.models import BoundaryConditionSpec, CaseConfigV1
from microfluidics.preprocessor.zones import ResolvedCaseZones, resolve_case_zones

_FLOW_KINDS = frozenset(
    {"velocity_inlet", "pressure_outlet", "wall", "symmetry", "periodic"}
)


@dataclass(frozen=True, slots=True)
class PeriodicFacePair:
    source_face: int
    target_face: int


@dataclass(frozen=True, slots=True)
class CompiledBoundaryCondition:
    id: str
    zone: str
    kind: str
    face_indices: np.ndarray
    parameters: dict[str, object]
    periodic_pairs: tuple[PeriodicFacePair, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledBoundaryConditions:
    conditions: tuple[CompiledBoundaryCondition, ...]

    def by_kind(self, kind: str) -> tuple[CompiledBoundaryCondition, ...]:
        return tuple(item for item in self.conditions if item.kind == kind)


def _periodic_translation(
    mesh: ImportedTetraMesh,
    source: np.ndarray,
    target: np.ndarray,
    configured: object | None,
) -> np.ndarray:
    if configured is not None:
        return np.asarray(configured, dtype=np.float64)
    return np.mean(mesh.face_centers[target], axis=0) - np.mean(
        mesh.face_centers[source], axis=0
    )


def pair_periodic_faces(
    mesh: ImportedTetraMesh,
    source_faces: np.ndarray,
    target_faces: np.ndarray,
    *,
    translation_m: object | None = None,
    tolerance_m: float | None = None,
) -> tuple[PeriodicFacePair, ...]:
    """Build a bijection for congruent translational periodic surfaces."""

    source = np.asarray(source_faces, dtype=np.int64)
    target = np.asarray(target_faces, dtype=np.int64)
    if source.size != target.size:
        raise BoundaryConditionError(
            "periodic zones must contain the same number of boundary faces"
        )
    if source.size == 0:
        raise BoundaryConditionError("periodic zones must not be empty")
    translation = _periodic_translation(mesh, source, target, translation_m)
    extent = np.ptp(np.asarray(mesh.points, dtype=np.float64), axis=0)
    scale = max(float(np.linalg.norm(extent)), 1.0)
    tolerance = float(tolerance_m) if tolerance_m is not None else scale * 1e-8
    available = set(target.tolist())
    pairs: list[PeriodicFacePair] = []
    for source_face in source.tolist():
        candidates = np.asarray(sorted(available), dtype=np.int64)
        expected_center = mesh.face_centers[source_face] + translation
        distances = np.linalg.norm(
            mesh.face_centers[candidates] - expected_center, axis=1
        )
        position = int(np.argmin(distances))
        target_face = int(candidates[position])
        if float(distances[position]) > tolerance:
            raise BoundaryConditionError(
                f"periodic face {source_face} has no target within tolerance {tolerance:.3e} m"
            )
        source_area = float(mesh.face_areas[source_face])
        target_area = float(mesh.face_areas[target_face])
        if not np.isclose(source_area, target_area, rtol=1e-6, atol=tolerance**2):
            raise BoundaryConditionError(
                f"periodic faces {source_face} and {target_face} have different areas"
            )
        normal_dot = float(
            np.dot(mesh.face_normals[source_face], mesh.face_normals[target_face])
        )
        if normal_dot > -1.0 + 1e-6:
            raise BoundaryConditionError(
                f"periodic faces {source_face} and {target_face} do not have opposite normals"
            )
        available.remove(target_face)
        pairs.append(PeriodicFacePair(source_face, target_face))
    return tuple(pairs)


def _conflict_group(condition: BoundaryConditionSpec) -> str:
    if condition.kind in _FLOW_KINDS:
        return "flow"
    return f"field:{condition.parameters['field']}"


def compile_boundary_conditions(
    mesh: ImportedTetraMesh,
    case: CaseConfigV1,
    *,
    resolved_zones: ResolvedCaseZones | None = None,
) -> CompiledBoundaryConditions:
    zones = resolved_zones or resolve_case_zones(mesh, case)
    compiled: list[CompiledBoundaryCondition] = []
    claimed: dict[tuple[str, int], str] = {}
    claimed_by_face: dict[int, set[str]] = {}
    periodic_claimed: dict[int, str] = {}
    for condition in case.boundary_conditions:
        zone = zones.zones[condition.zone]
        faces = np.asarray(zone.entity_indices, dtype=np.int64)
        periodic_pairs: tuple[PeriodicFacePair, ...] = ()
        occupied_faces = faces
        if condition.kind == "periodic":
            paired_zone_id = str(condition.parameters["paired_zone"])
            target = np.asarray(
                zones.zones[paired_zone_id].entity_indices, dtype=np.int64
            )
            periodic_pairs = pair_periodic_faces(
                mesh,
                faces,
                target,
                translation_m=condition.parameters.get("translation_m"),
            )
            occupied_faces = np.unique(np.concatenate((faces, target))).astype(np.int64)

        group = _conflict_group(condition)
        for face in occupied_faces.tolist():
            face = int(face)
            if condition.kind == "periodic":
                previous_conditions = claimed_by_face.get(face, set())
                if previous_conditions:
                    previous = sorted(previous_conditions)[0]
                    raise BoundaryConditionError(
                        f"boundary conditions {previous!r} and {condition.id!r} "
                        f"conflict on periodic face {face}"
                    )
            else:
                previous_periodic = periodic_claimed.get(face)
                if previous_periodic is not None:
                    raise BoundaryConditionError(
                        f"boundary conditions {previous_periodic!r} and "
                        f"{condition.id!r} conflict on periodic face {face}"
                    )
            key = (group, face)
            previous = claimed.get(key)
            if previous is not None:
                raise BoundaryConditionError(
                    f"boundary conditions {previous!r} and {condition.id!r} conflict "
                    f"on face {face} for {group}"
                )
            claimed[key] = condition.id
            claimed_by_face.setdefault(face, set()).add(condition.id)
            if condition.kind == "periodic":
                periodic_claimed[face] = condition.id
        compiled.append(
            CompiledBoundaryCondition(
                id=condition.id,
                zone=condition.zone,
                kind=condition.kind,
                face_indices=faces,
                parameters=dict(condition.parameters),
                periodic_pairs=periodic_pairs,
            )
        )
    return CompiledBoundaryConditions(conditions=tuple(compiled))
