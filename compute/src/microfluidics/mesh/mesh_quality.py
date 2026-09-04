"""Quality diagnostics for mesh-oriented T-junction scaffolds."""

from dataclasses import dataclass
from typing import Dict, List

from microfluidics.mesh.mesh_builder import MeshDataModel
from microfluidics.mesh.mesh_config import MeshRefinementConfig
from microfluidics.mesh.mesh_geometry import MeshGeometryModel3D


@dataclass(frozen=True)
class MeshQualityReport:
    point_count: int
    cell_count: int
    boundary_face_counts: Dict[str, int]
    dx: float
    dy: float
    dz: float
    aspect_ratio_proxy: float
    cavity_depth_cells_x: float
    inlet_width_cells_y: float
    channel_height_cells_z: float
    junction_width_cells_x: float
    wall_layer_coverage: float
    warnings: List[str]
    summary_text: str


def evaluate_mesh_quality(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
    refinement: MeshRefinementConfig,
) -> MeshQualityReport:
    """Compute quality indicators for mesh scaffold readiness."""
    del refinement  # Reserved for future adaptive remeshing diagnostics.

    dx = float(mesh.metadata.get("dx", 0.0))
    dy = float(mesh.metadata.get("dy", 0.0))
    dz = float(mesh.metadata.get("dz", 0.0))
    min_h = min(value for value in (dx, dy, dz) if value > 0.0)
    max_h = max(dx, dy, dz)
    aspect_ratio_proxy = max_h / min_h if min_h > 0.0 else float("inf")

    cfg = geometry.config
    cavity_depth_cells_x = cfg.cavity_depth / dx if dx > 0 else float("inf")
    inlet_width_cells_y = cfg.inlet_width / dy if dy > 0 else float("inf")
    channel_height_cells_z = cfg.channel_height / dz if dz > 0 else float("inf")
    junction_width_cells_x = cfg.outlet_width / dx if dx > 0 else float("inf")
    wall_layer = mesh.cell_data.get("wall_layer_id")
    wall_layer_coverage = (
        float((wall_layer >= 0).mean()) if wall_layer is not None else 0.0
    )

    warnings: List[str] = []
    if cfg.include_cavities and cavity_depth_cells_x < 4.0:
        warnings.append("Cavity depth is weakly resolved (< 4 cells in x).")
    if inlet_width_cells_y < 8.0:
        warnings.append("Inlet width is weakly resolved (< 8 cells in y).")
    if channel_height_cells_z < 8.0:
        warnings.append("Channel height is weakly resolved (< 8 cells in z).")
    if aspect_ratio_proxy > 3.0:
        warnings.append("Cell size anisotropy is high (aspect ratio proxy > 3).")
    if wall_layer_coverage < 0.05:
        warnings.append("Near-wall layer coverage is low (< 5% of fluid cells).")

    boundary_counts = {
        tag: int(faces.shape[0]) for tag, faces in mesh.boundary_faces.items()
    }
    summary_lines = [
        "Mesh quality summary",
        f"- points: {mesh.points.shape[0]}",
        f"- cells: {mesh.cells.shape[0]}",
        f"- spacing [dx, dy, dz] = [{dx:.6e}, {dy:.6e}, {dz:.6e}] m",
        f"- aspect ratio proxy: {aspect_ratio_proxy:.3f}",
        f"- cavity depth cells (x): {cavity_depth_cells_x:.2f}",
        f"- inlet width cells (y): {inlet_width_cells_y:.2f}",
        f"- channel height cells (z): {channel_height_cells_z:.2f}",
        f"- junction width cells (x): {junction_width_cells_x:.2f}",
        f"- wall-layer coverage: {wall_layer_coverage:.3f}",
        "- boundary faces: "
        + ", ".join(f"{name}={count}" for name, count in boundary_counts.items()),
    ]
    if warnings:
        summary_lines.append("- warnings:")
        summary_lines.extend(f"  * {message}" for message in warnings)
    else:
        summary_lines.append("- warnings: none")
    summary_text = "\n".join(summary_lines)

    return MeshQualityReport(
        point_count=int(mesh.points.shape[0]),
        cell_count=int(mesh.cells.shape[0]),
        boundary_face_counts=boundary_counts,
        dx=dx,
        dy=dy,
        dz=dz,
        aspect_ratio_proxy=float(aspect_ratio_proxy),
        cavity_depth_cells_x=float(cavity_depth_cells_x),
        inlet_width_cells_y=float(inlet_width_cells_y),
        channel_height_cells_z=float(channel_height_cells_z),
        junction_width_cells_x=float(junction_width_cells_x),
        wall_layer_coverage=float(wall_layer_coverage),
        warnings=warnings,
        summary_text=summary_text,
    )
