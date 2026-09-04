"""Detailed, policy-driven mesh quality findings for preprocessing gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.preprocessor.models import MeshQualityPolicy
from microfluidics.preprocessor.zones import ResolvedCaseZones


@dataclass(frozen=True, slots=True)
class MeshQualityFinding:
    severity: Literal["error", "warning"]
    code: str
    entity_type: Literal["cell", "face", "boundary_triangle", "mesh"]
    entity_id: int | None
    value: float | None
    threshold: float | None
    center_xyz: tuple[float, float, float] | None
    message: str


@dataclass(frozen=True, slots=True)
class MeshQualityGateReport:
    findings: tuple[MeshQualityFinding, ...]
    total_counts: dict[str, int]
    total_errors: int
    total_warnings: int
    truncated: bool
    fail_on_warnings: bool

    @property
    def errors(self) -> tuple[MeshQualityFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[MeshQualityFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "warning")

    @property
    def is_acceptable(self) -> bool:
        return self.total_errors == 0 and (
            not self.fail_on_warnings or self.total_warnings == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "is_acceptable": self.is_acceptable,
            "fail_on_warnings": self.fail_on_warnings,
            "total_counts": dict(self.total_counts),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "truncated": self.truncated,
            "findings": [asdict(item) for item in self.findings],
        }


def _center(values: np.ndarray, entity_id: int) -> tuple[float, float, float] | None:
    if entity_id < 0 or entity_id >= values.shape[0]:
        return None
    row = np.asarray(values[entity_id], dtype=np.float64).reshape(-1)
    if row.size != 3:
        return None
    return tuple(float(value) for value in row)  # type: ignore[return-value]


def _tetra_aspect_ratios(mesh: ImportedTetraMesh) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=np.float64)
    tetrahedra = np.asarray(mesh.tetrahedra, dtype=np.int64)
    if tetrahedra.size == 0:
        return np.zeros((0,), dtype=np.float64)
    tetra_points = points[tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edges = np.stack(
        [
            np.linalg.norm(tetra_points[:, right] - tetra_points[:, left], axis=1)
            for left, right in edge_pairs
        ],
        axis=1,
    )
    return np.max(edges, axis=1) / np.maximum(np.min(edges, axis=1), 1e-30)


def evaluate_mesh_quality_gate(
    mesh: ImportedTetraMesh,
    policy: MeshQualityPolicy,
    *,
    resolved_zones: ResolvedCaseZones | None = None,
) -> MeshQualityGateReport:
    """Evaluate configurable quality thresholds and retain actionable entity IDs."""

    all_findings: list[MeshQualityFinding] = []
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    for entity_id in np.flatnonzero(volumes <= policy.min_cell_volume_m3).tolist():
        all_findings.append(
            MeshQualityFinding(
                severity="error",
                code="small_cell_volume",
                entity_type="cell",
                entity_id=int(entity_id),
                value=float(volumes[entity_id]),
                threshold=float(policy.min_cell_volume_m3),
                center_xyz=_center(mesh.cell_centers, int(entity_id)),
                message="Tetrahedron volume is at or below the configured minimum.",
            )
        )

    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    for entity_id in np.flatnonzero(areas <= policy.min_face_area_m2).tolist():
        all_findings.append(
            MeshQualityFinding(
                severity="error",
                code="degenerate_face_area",
                entity_type="face",
                entity_id=int(entity_id),
                value=float(areas[entity_id]),
                threshold=float(policy.min_face_area_m2),
                center_xyz=_center(mesh.face_centers, int(entity_id)),
                message="Face area is at or below the configured minimum.",
            )
        )

    aspect = _tetra_aspect_ratios(mesh)
    for entity_id in np.flatnonzero(aspect > policy.max_tetra_aspect_ratio).tolist():
        all_findings.append(
            MeshQualityFinding(
                severity="warning",
                code="high_tetra_aspect_ratio",
                entity_type="cell",
                entity_id=int(entity_id),
                value=float(aspect[entity_id]),
                threshold=float(policy.max_tetra_aspect_ratio),
                center_xyz=_center(mesh.cell_centers, int(entity_id)),
                message="Tetrahedron aspect-ratio proxy exceeds the configured maximum.",
            )
        )

    orientation = mesh.diagnostics.get("boundary_orientation", {})
    if isinstance(orientation, dict):
        ambiguous_indices = orientation.get("ambiguous_triangle_indices", [])
        if not isinstance(ambiguous_indices, list):
            ambiguous_indices = []
        for entity_id in ambiguous_indices:
            all_findings.append(
                MeshQualityFinding(
                    severity="error",
                    code="ambiguous_boundary_orientation",
                    entity_type="boundary_triangle",
                    entity_id=int(entity_id),
                    value=None,
                    threshold=None,
                    center_xyz=None,
                    message="Boundary triangle orientation could not be resolved.",
                )
            )

    unresolved_faces = set(
        np.asarray(mesh.boundary_unresolved_faces, dtype=np.int64).tolist()
    )
    if resolved_zones is not None:
        assigned = {
            int(entity)
            for zone in resolved_zones.zones.values()
            if zone.kind == "surface"
            for entity in zone.entity_indices.tolist()
        }
        unresolved_faces -= assigned
    for entity_id in sorted(unresolved_faces):
        all_findings.append(
            MeshQualityFinding(
                severity="error",
                code="unresolved_boundary_tag",
                entity_type="face",
                entity_id=int(entity_id),
                value=float(mesh.boundary_tag_per_face[entity_id]),
                threshold=None,
                center_xyz=_center(mesh.face_centers, int(entity_id)),
                message="Boundary face has no resolved physical boundary class.",
            )
        )

    counts: dict[str, int] = {}
    for finding in all_findings:
        counts[finding.code] = counts.get(finding.code, 0) + 1
    limit = int(policy.max_reported_findings)
    return MeshQualityGateReport(
        findings=tuple(all_findings[:limit]),
        total_counts=counts,
        total_errors=sum(item.severity == "error" for item in all_findings),
        total_warnings=sum(item.severity == "warning" for item in all_findings),
        truncated=len(all_findings) > limit,
        fail_on_warnings=bool(policy.fail_on_warnings),
    )
