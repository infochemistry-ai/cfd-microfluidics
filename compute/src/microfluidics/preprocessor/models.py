"""Versioned, solver-neutral contracts for CFD preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

ZoneKind = Literal["surface", "volume"]
BoundaryKind = Literal[
    "velocity_inlet",
    "pressure_outlet",
    "wall",
    "symmetry",
    "periodic",
    "dirichlet",
    "neumann",
    "robin",
]


@dataclass(frozen=True, slots=True)
class MeshSpec:
    path: str


@dataclass(frozen=True, slots=True)
class ZoneSpec:
    id: str
    kind: ZoneKind
    physical_names: tuple[str, ...] = ()
    physical_tags: tuple[int, ...] = ()
    allow_overlap: bool = False


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    id: str
    zone: str
    properties: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BoundaryConditionSpec:
    id: str
    zone: str
    kind: BoundaryKind
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeshQualityPolicy:
    min_cell_volume_m3: float = 1e-18
    min_face_area_m2: float = 0.0
    max_tetra_aspect_ratio: float = 12.0
    max_reported_findings: int = 100
    fail_on_warnings: bool = False


@dataclass(frozen=True, slots=True)
class CaseConfigV1:
    schema_version: Literal["case_config_v1"]
    case_id: str
    mesh: MeshSpec
    zones: tuple[ZoneSpec, ...]
    materials: tuple[MaterialSpec, ...]
    boundary_conditions: tuple[BoundaryConditionSpec, ...]
    mesh_quality: MeshQualityPolicy
