"""Resolve explicit physics zones against imported Gmsh entity tags."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor.errors import ZoneResolutionError
from microfluidics.preprocessor.models import CaseConfigV1, ZoneSpec


@dataclass(frozen=True, slots=True)
class ResolvedZone:
    id: str
    kind: str
    entity_indices: np.ndarray
    physical_tags: tuple[int, ...]
    physical_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedCaseZones:
    zones: dict[str, ResolvedZone]
    unassigned_boundary_faces: np.ndarray
    unassigned_cells: np.ndarray


def _tags_for_zone(mesh: ImportedTetraMesh, zone: ZoneSpec) -> tuple[int, ...]:
    expected_dim = 2 if zone.kind == "surface" else 3
    tags = set(zone.physical_tags)
    for name in zone.physical_names:
        group = mesh.physical_groups.get(name)
        if group is None:
            raise ZoneResolutionError(
                f"zone {zone.id!r} references unknown physical name {name!r}"
            )
        tag, dimension = int(group[0]), int(group[1])
        if dimension != expected_dim:
            raise ZoneResolutionError(
                f"zone {zone.id!r} expects a {zone.kind} group, but {name!r} "
                f"has Gmsh dimension {dimension}"
            )
        tags.add(tag)
    return tuple(sorted(tags))


def _entities_for_zone(
    mesh: ImportedTetraMesh, zone: ZoneSpec, tags: tuple[int, ...]
) -> np.ndarray:
    if zone.kind == "surface":
        candidate = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
        tag_values = np.asarray(mesh.boundary_tag_per_face, dtype=np.int32)
        entities = candidate[np.isin(tag_values[candidate], tags)]
    else:
        tag_values = np.asarray(mesh.volume_tag_per_cell, dtype=np.int32)
        entities = np.flatnonzero(np.isin(tag_values, tags)).astype(np.int64)
    if entities.size == 0:
        raise ZoneResolutionError(
            f"zone {zone.id!r} resolved no {zone.kind} entities for tags {list(tags)}"
        )
    return np.asarray(np.unique(entities), dtype=np.int64)


def resolve_case_zones(
    mesh: ImportedTetraMesh, case: CaseConfigV1
) -> ResolvedCaseZones:
    """Resolve every configured zone and reject accidental entity overlap."""

    resolved: dict[str, ResolvedZone] = {}
    claimed: dict[tuple[str, int], ZoneSpec] = {}
    for zone in case.zones:
        tags = _tags_for_zone(mesh, zone)
        entities = _entities_for_zone(mesh, zone, tags)
        for entity in entities.tolist():
            key = (zone.kind, int(entity))
            previous = claimed.get(key)
            if previous is not None and not (
                previous.allow_overlap and zone.allow_overlap
            ):
                raise ZoneResolutionError(
                    f"zones {previous.id!r} and {zone.id!r} overlap on "
                    f"{zone.kind} entity {entity}; set allow_overlap=true on both "
                    "zones only when the overlap is intentional"
                )
            claimed[key] = zone
        resolved[zone.id] = ResolvedZone(
            id=zone.id,
            kind=zone.kind,
            entity_indices=entities,
            physical_tags=tags,
            physical_names=zone.physical_names,
        )

    assigned_surface = {
        entity for (kind, entity), _zone in claimed.items() if kind == "surface"
    }
    assigned_volume = {
        entity for (kind, entity), _zone in claimed.items() if kind == "volume"
    }
    boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    cells = np.arange(mesh.tetrahedra.shape[0], dtype=np.int64)
    return ResolvedCaseZones(
        zones=resolved,
        unassigned_boundary_faces=np.asarray(
            [value for value in boundary.tolist() if value not in assigned_surface],
            dtype=np.int64,
        ),
        unassigned_cells=np.asarray(
            [value for value in cells.tolist() if value not in assigned_volume],
            dtype=np.int64,
        ),
    )
