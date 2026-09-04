"""Validation utilities for imported external Gmsh tetrahedral meshes."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh


@dataclass(frozen=True)
class GmshMeshValidationReport:
    """Validation result with summary and structured findings."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    summary: Dict[str, object] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def _shape_check(
    name: str,
    array: np.ndarray,
    expected_ndim: int,
    expected_last: int,
    errors: list[str],
) -> None:
    if array.ndim != expected_ndim:
        errors.append(f"{name} must be rank {expected_ndim}, got rank {array.ndim}.")
        return
    if expected_ndim > 1 and array.shape[-1] != expected_last:
        errors.append(
            f"{name} last dimension must be {expected_last}, got {array.shape[-1]}."
        )


def validate_imported_tetra_mesh(
    mesh: ImportedTetraMesh,
    *,
    min_positive_volume: float = 1e-18,
    require_legacy_boundary_classes: bool = True,
) -> GmshMeshValidationReport:
    """Run consistency checks required before solver integration."""
    errors: list[str] = []
    warnings: list[str] = []

    _shape_check("points", mesh.points, 2, 3, errors)
    _shape_check("tetrahedra", mesh.tetrahedra, 2, 4, errors)
    _shape_check("boundary_triangles", mesh.boundary_triangles, 2, 3, errors)
    _shape_check("cell_centers", mesh.cell_centers, 2, 3, errors)
    _shape_check("face_vertices", mesh.face_vertices, 2, 3, errors)
    _shape_check("face_centers", mesh.face_centers, 2, 3, errors)
    _shape_check("face_normals", mesh.face_normals, 2, 3, errors)
    _shape_check("face_to_cells", mesh.face_to_cells, 2, 2, errors)
    _shape_check("cell_to_faces", mesh.cell_to_faces, 2, 4, errors)

    if mesh.boundary_face_tags.ndim != 1:
        errors.append("boundary_face_tags must be rank 1.")
    if mesh.boundary_face_tags.shape[0] != mesh.boundary_triangles.shape[0]:
        errors.append(
            "boundary_face_tags length must match boundary_triangles row count."
        )

    if mesh.cell_volumes.ndim != 1:
        errors.append("cell_volumes must be rank 1.")
    if mesh.cell_volumes.shape[0] != mesh.tetrahedra.shape[0]:
        errors.append("cell_volumes length must match tetrahedra row count.")

    if mesh.volume_tag_per_cell.ndim != 1:
        errors.append("volume_tag_per_cell must be rank 1.")
    if mesh.volume_tag_per_cell.shape[0] != mesh.tetrahedra.shape[0]:
        errors.append("volume_tag_per_cell length must match tetrahedra row count.")

    if mesh.face_areas.ndim != 1:
        errors.append("face_areas must be rank 1.")
    if mesh.face_areas.shape[0] != mesh.face_vertices.shape[0]:
        errors.append("face_areas length must match face_vertices row count.")

    if mesh.boundary_tag_per_face.ndim != 1:
        errors.append("boundary_tag_per_face must be rank 1.")
    if mesh.boundary_tag_per_face.shape[0] != mesh.face_vertices.shape[0]:
        errors.append("boundary_tag_per_face length must match face count.")

    node_count = int(mesh.points.shape[0])
    tetra_count = int(mesh.tetrahedra.shape[0])
    face_count = int(mesh.face_vertices.shape[0])

    if mesh.tetrahedra.size > 0:
        tet_min = int(np.min(mesh.tetrahedra))
        tet_max = int(np.max(mesh.tetrahedra))
        if tet_min < 0 or tet_max >= node_count:
            errors.append(
                f"tetrahedra indices out of bounds: min={tet_min}, max={tet_max}, "
                f"node_count={node_count}."
            )
    if mesh.boundary_triangles.size > 0:
        tri_min = int(np.min(mesh.boundary_triangles))
        tri_max = int(np.max(mesh.boundary_triangles))
        if tri_min < 0 or tri_max >= node_count:
            errors.append(
                f"boundary_triangles indices out of bounds: min={tri_min}, max={tri_max}, "
                f"node_count={node_count}."
            )

    if mesh.cell_to_faces.size > 0:
        face_min = int(np.min(mesh.cell_to_faces))
        face_max = int(np.max(mesh.cell_to_faces))
        if face_min < 0 or face_max >= face_count:
            errors.append(
                f"cell_to_faces indices out of bounds: min={face_min}, max={face_max}, "
                f"face_count={face_count}."
            )

    if mesh.face_to_cells.size > 0:
        c0 = mesh.face_to_cells[:, 0]
        c1 = mesh.face_to_cells[:, 1]
        if np.any(c0 < 0) or np.any(c0 >= tetra_count):
            errors.append("face_to_cells[:,0] must contain valid cell ids.")
        invalid_c1 = (c1 >= tetra_count) | (c1 < -1)
        if np.any(invalid_c1):
            errors.append(
                "face_to_cells[:,1] must contain valid cell ids or -1 for boundaries."
            )

    zero_or_neg_vol = int(np.count_nonzero(mesh.cell_volumes <= min_positive_volume))
    if zero_or_neg_vol > 0:
        errors.append(
            f"Detected {zero_or_neg_vol} tetrahedra with non-positive/small volume "
            f"(<= {min_positive_volume:.3e})."
        )

    zero_area = int(np.count_nonzero(mesh.face_areas <= 0.0))
    if zero_area > 0:
        errors.append(f"Detected {zero_area} faces with zero area.")

    computed_boundary = np.flatnonzero(mesh.face_to_cells[:, 1] < 0).astype(np.int64)
    if computed_boundary.shape[0] == 0:
        errors.append("No boundary faces found from reconstructed face adjacency.")

    if mesh.boundary_face_indices.shape[0] != computed_boundary.shape[0]:
        warnings.append(
            "boundary_face_indices length differs from reconstructed boundary count."
        )

    if require_legacy_boundary_classes:
        if mesh.inlet_faces.size == 0:
            errors.append("No inlet faces classified from boundary groups.")
        if mesh.outlet_faces.size == 0:
            errors.append("No outlet faces classified from boundary groups.")
        if mesh.wall_faces.size == 0:
            errors.append("No wall faces classified from boundary groups.")

    unresolved_count = int(mesh.boundary_unresolved_faces.shape[0])
    if unresolved_count > 0:
        warnings.append(
            f"Unresolved boundary faces: {unresolved_count}. Check physical names/tags mapping."
        )

    quality = dict(mesh.diagnostics.get("tetra_quality", {}))
    for item in quality.get("quality_warnings", []):
        warnings.append(f"quality: {item}")
    orientation = dict(mesh.diagnostics.get("boundary_orientation", {}))
    if int(orientation.get("ambiguous_count", 0)) > 0:
        warnings.append(
            "Boundary face orientation has ambiguous triangles: "
            f"{int(orientation.get('ambiguous_count', 0))}"
        )

    boundary_semantic = dict(mesh.diagnostics.get("boundary_semantic_counts", {}))
    boundary_tags = sorted(int(v) for v in np.unique(mesh.boundary_face_tags).tolist())

    summary = {
        "source_path": str(mesh.source_path),
        "node_count": node_count,
        "tetra_count": tetra_count,
        "volume_tags": sorted(
            int(v) for v in np.unique(mesh.volume_tag_per_cell).tolist() if int(v) >= 0
        ),
        "imported_boundary_triangle_count": int(mesh.boundary_triangles.shape[0]),
        "face_count": face_count,
        "interior_face_count": int(mesh.interior_face_indices.shape[0]),
        "boundary_face_count": int(mesh.boundary_face_indices.shape[0]),
        "inlet_face_count": int(mesh.inlet_faces.shape[0]),
        "outlet_face_count": int(mesh.outlet_faces.shape[0]),
        "wall_face_count": int(mesh.wall_faces.shape[0]),
        "unresolved_boundary_face_count": unresolved_count,
        "physical_group_names": sorted(mesh.physical_groups.keys()),
        "boundary_tags": boundary_tags,
        "boundary_tag_name_map": {
            str(k): v for k, v in mesh.boundary_face_names.items()
        },
        "boundary_tag_counts": mesh.diagnostics.get("boundary_tag_counts", {}),
        "boundary_name_counts": mesh.diagnostics.get("boundary_name_counts", {}),
        "boundary_semantic_counts": boundary_semantic,
        "boundary_tag_semantic_map": mesh.diagnostics.get(
            "boundary_tag_semantic_map", {}
        ),
        "boundary_orientation": orientation,
        "tetra_quality": quality,
        "diagnostics": mesh.diagnostics,
    }
    return GmshMeshValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        summary=summary,
    )


def format_validation_report(report: GmshMeshValidationReport) -> str:
    """Build a readable text report for logs and artifacts."""
    lines: list[str] = []
    lines.append("Gmsh Mesh Validation Report")
    lines.append("=" * 32)
    lines.append(f"valid: {report.is_valid}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 32)
    for key, value in report.summary.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    lines.append("Errors")
    lines.append("-" * 32)
    if report.errors:
        for item in report.errors:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("Warnings")
    lines.append("-" * 32)
    if report.warnings:
        for item in report.warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)
