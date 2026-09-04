"""Mesh scaffold builder for the mesh-oriented 3D branch.

This module intentionally provides a reliable mesh data model + diagnostics
foundation. It does not claim to be a full unstructured Navier-Stokes solver.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import numpy as np

from microfluidics.mesh.mesh_config import (
    MeshGeometryConfig3D,
    MeshRefinementConfig,
    Resolution3D,
)
from microfluidics.mesh.mesh_geometry import (
    MeshGeometryModel3D,
    build_mesh_geometry_3d,
    points_in_union,
)

if TYPE_CHECKING:
    from microfluidics.cad.config import GeometryPipelineConfig
    from microfluidics.cad.pipeline import CadMeshPipelineResult

VTK_HEXAHEDRON = 12

BOUNDARY_TAG_IDS = {
    "left_inlet": 1,
    "right_inlet": 2,
    "outlet": 3,
    "wall": 4,
    "cavity_wall": 5,
}

ZONE_IDS = {
    "left_inlet_channel": 1,
    "right_inlet_channel": 2,
    "mixing_channel": 3,
    "left_cavity": 4,
    "right_cavity": 5,
}


@dataclass(frozen=True)
class MeshBuildSummary:
    total_candidate_cells: int
    fluid_cell_count: int
    point_count: int
    boundary_face_counts: Dict[str, int]
    wall_layer_coverage: float


@dataclass
class MeshDataModel:
    """Mesh connectivity, metadata, and optional fields."""

    points: np.ndarray
    cells: np.ndarray
    cell_types: np.ndarray
    cell_centers: np.ndarray
    cell_indices: np.ndarray
    cell_data: Dict[str, np.ndarray] = field(default_factory=dict)
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    boundary_faces: Dict[str, np.ndarray] = field(default_factory=dict)
    boundary_face_centers: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    summary: MeshBuildSummary | None = None


@dataclass(frozen=True)
class GeometryPipelineBuildResult:
    """Result of an explicitly selected CAD or legacy geometry source."""

    source_kind: str
    cad_mesh: "CadMeshPipelineResult | None" = None
    legacy_mesh: MeshDataModel | None = None


def _cell_center_grid(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    return x_centers, y_centers, z_centers


def _point_linear_index(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k


def _classify_zone_ids(
    centers: np.ndarray,
    geometry: MeshGeometryModel3D,
) -> np.ndarray:
    zone_ids = np.zeros(centers.shape[0], dtype=np.int32)
    for component_name, zone_id in ZONE_IDS.items():
        component = geometry.fluid_components.get(component_name)
        if component is None:
            continue
        mask = component.contains_points(centers)
        zone_ids[mask] = zone_id
    return zone_ids


def _is_cavity_wall_face(
    face_center: Tuple[float, float, float],
    geometry: MeshGeometryModel3D,
    eps: float = 1e-12,
) -> bool:
    cfg = geometry.config
    if not cfg.include_cavities:
        return False
    x_abs = abs(face_center[0])
    y_coord = face_center[1]
    half_outlet = 0.5 * cfg.outlet_width
    x_outer = half_outlet + cfg.cavity_depth
    y_min = cfg.cavity_offset_from_junction
    y_max = cfg.cavity_offset_from_junction + cfg.cavity_length
    return (
        half_outlet - eps <= x_abs <= x_outer + eps
        and y_min - eps <= y_coord <= y_max + eps
    )


def _classify_boundary_face_tag(
    face_center: Tuple[float, float, float],
    axis: str,
    direction: int,
    geometry: MeshGeometryModel3D,
) -> str:
    left_inlet = geometry.boundary_patches["left_inlet"]
    right_inlet = geometry.boundary_patches["right_inlet"]
    outlet = geometry.boundary_patches["outlet"]
    if axis == "x" and direction < 0 and left_inlet.contains_point(face_center):
        return "left_inlet"
    if axis == "x" and direction > 0 and right_inlet.contains_point(face_center):
        return "right_inlet"
    if axis == "y" and direction > 0 and outlet.contains_point(face_center):
        return "outlet"
    if _is_cavity_wall_face(face_center, geometry):
        return "cavity_wall"
    return "wall"


def _create_edge_arrays(
    geometry: MeshGeometryModel3D,
    resolution: Resolution3D,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain = geometry.domain
    nx, ny, nz = resolution
    x_edges = np.linspace(domain.x[0], domain.x[1], nx + 1, dtype=np.float64)
    y_edges = np.linspace(domain.y[0], domain.y[1], ny + 1, dtype=np.float64)
    z_edges = np.linspace(domain.z[0], domain.z[1], nz + 1, dtype=np.float64)
    return x_edges, y_edges, z_edges


def _face_vertices_for_direction(
    i: int,
    j: int,
    k: int,
    direction_key: str,
    ny: int,
    nz: int,
) -> np.ndarray:
    p000 = _point_linear_index(i, j, k, ny, nz)
    p100 = _point_linear_index(i + 1, j, k, ny, nz)
    p110 = _point_linear_index(i + 1, j + 1, k, ny, nz)
    p010 = _point_linear_index(i, j + 1, k, ny, nz)
    p001 = _point_linear_index(i, j, k + 1, ny, nz)
    p101 = _point_linear_index(i + 1, j, k + 1, ny, nz)
    p111 = _point_linear_index(i + 1, j + 1, k + 1, ny, nz)
    p011 = _point_linear_index(i, j + 1, k + 1, ny, nz)
    if direction_key == "x-":
        return np.array([p000, p001, p011, p010], dtype=np.int64)
    if direction_key == "x+":
        return np.array([p100, p110, p111, p101], dtype=np.int64)
    if direction_key == "y-":
        return np.array([p000, p100, p101, p001], dtype=np.int64)
    if direction_key == "y+":
        return np.array([p010, p011, p111, p110], dtype=np.int64)
    if direction_key == "z-":
        return np.array([p000, p010, p110, p100], dtype=np.int64)
    if direction_key == "z+":
        return np.array([p001, p101, p111, p011], dtype=np.int64)
    raise ValueError(f"Unknown direction key: {direction_key}")


def _compute_wall_layers(
    fluid_cell_ijk: np.ndarray,
    fluid_mask: np.ndarray,
    wall_seed_ids: Iterable[int],
    max_layers: int,
) -> np.ndarray:
    layer_ids = -np.ones(fluid_cell_ijk.shape[0], dtype=np.int32)
    if max_layers <= 0:
        return layer_ids
    if not wall_seed_ids:
        return layer_ids

    nx, ny, nz = fluid_mask.shape
    id_grid = -np.ones((nx, ny, nz), dtype=np.int64)
    id_grid[tuple(fluid_cell_ijk.T)] = np.arange(
        fluid_cell_ijk.shape[0], dtype=np.int64
    )
    frontier = np.array(sorted(set(wall_seed_ids)), dtype=np.int64)
    layer_ids[frontier] = 0
    visited = set(frontier.tolist())
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    for layer in range(1, max_layers):
        next_frontier_set: set[int] = set()
        for cell_id in frontier.tolist():
            i, j, k = fluid_cell_ijk[cell_id]
            for di, dj, dk in offsets:
                ni, nj, nk = i + di, j + dj, k + dk
                if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                    continue
                if not fluid_mask[ni, nj, nk]:
                    continue
                neigh_id = int(id_grid[ni, nj, nk])
                if neigh_id < 0 or neigh_id in visited:
                    continue
                next_frontier_set.add(neigh_id)
        if not next_frontier_set:
            break
        frontier = np.array(sorted(next_frontier_set), dtype=np.int64)
        layer_ids[frontier] = layer
        visited.update(next_frontier_set)
    return layer_ids


def build_mesh_scaffold(
    geometry: MeshGeometryModel3D,
    resolution: Resolution3D,
    refinement: MeshRefinementConfig,
) -> MeshDataModel:
    """Build a structured connectivity scaffold tagged for mesh workflows.

    The returned model stores explicit points/cells/boundary-tags and can be
    exported to VTK for ParaView inspection. This is a pre-solver stage.
    """
    nx, ny, nz = resolution
    x_edges, y_edges, z_edges = _create_edge_arrays(geometry, resolution)
    x_centers, y_centers, z_centers = _cell_center_grid(x_edges, y_edges, z_edges)

    center_grid = np.stack(
        np.meshgrid(x_centers, y_centers, z_centers, indexing="ij"),
        axis=-1,
    )
    centers_flat = center_grid.reshape(-1, 3)
    fluid_mask_flat = points_in_union(centers_flat, geometry.fluid_components.values())
    fluid_mask = fluid_mask_flat.reshape(nx, ny, nz)
    fluid_cell_ijk = np.argwhere(fluid_mask)
    fluid_centers = center_grid[fluid_mask]
    zone_ids = _classify_zone_ids(fluid_centers, geometry)

    cells_old: list[np.ndarray] = []
    boundary_faces_old: Dict[str, list[np.ndarray]] = {
        name: [] for name in BOUNDARY_TAG_IDS
    }
    boundary_centers: Dict[str, list[Tuple[float, float, float]]] = {
        name: [] for name in BOUNDARY_TAG_IDS
    }
    wall_seed_ids: list[int] = []

    directions = (
        ("x-", (-1, 0, 0), "x", -1),
        ("x+", (1, 0, 0), "x", 1),
        ("y-", (0, -1, 0), "y", -1),
        ("y+", (0, 1, 0), "y", 1),
        ("z-", (0, 0, -1), "z", -1),
        ("z+", (0, 0, 1), "z", 1),
    )

    for cell_id, (i, j, k) in enumerate(fluid_cell_ijk.tolist()):
        p000 = _point_linear_index(i, j, k, ny, nz)
        p100 = _point_linear_index(i + 1, j, k, ny, nz)
        p110 = _point_linear_index(i + 1, j + 1, k, ny, nz)
        p010 = _point_linear_index(i, j + 1, k, ny, nz)
        p001 = _point_linear_index(i, j, k + 1, ny, nz)
        p101 = _point_linear_index(i + 1, j, k + 1, ny, nz)
        p111 = _point_linear_index(i + 1, j + 1, k + 1, ny, nz)
        p011 = _point_linear_index(i, j + 1, k + 1, ny, nz)
        cells_old.append(np.array([p000, p100, p110, p010, p001, p101, p111, p011]))

        touches_wall = False
        for direction_key, (di, dj, dk), axis, sign in directions:
            ni, nj, nk = i + di, j + dj, k + dk
            neighbor_is_fluid = (
                0 <= ni < nx
                and 0 <= nj < ny
                and 0 <= nk < nz
                and bool(fluid_mask[ni, nj, nk])
            )
            if neighbor_is_fluid:
                continue

            if direction_key == "x-":
                face_center = (x_edges[i], y_centers[j], z_centers[k])
            elif direction_key == "x+":
                face_center = (x_edges[i + 1], y_centers[j], z_centers[k])
            elif direction_key == "y-":
                face_center = (x_centers[i], y_edges[j], z_centers[k])
            elif direction_key == "y+":
                face_center = (x_centers[i], y_edges[j + 1], z_centers[k])
            elif direction_key == "z-":
                face_center = (x_centers[i], y_centers[j], z_edges[k])
            else:
                face_center = (x_centers[i], y_centers[j], z_edges[k + 1])

            tag = _classify_boundary_face_tag(face_center, axis, sign, geometry)
            face_vertices = _face_vertices_for_direction(i, j, k, direction_key, ny, nz)
            boundary_faces_old[tag].append(face_vertices)
            boundary_centers[tag].append(face_center)
            if tag in {"wall", "cavity_wall"}:
                touches_wall = True
        if touches_wall:
            wall_seed_ids.append(cell_id)

    if not cells_old:
        raise RuntimeError(
            "No fluid cells were generated; check geometry or resolution."
        )

    cells_old_arr = np.asarray(cells_old, dtype=np.int64)
    total_points = (nx + 1) * (ny + 1) * (nz + 1)
    used_old_point_ids = np.unique(cells_old_arr.reshape(-1))
    point_map = -np.ones(total_points, dtype=np.int64)
    point_map[used_old_point_ids] = np.arange(used_old_point_ids.size, dtype=np.int64)
    cells = point_map[cells_old_arr]

    stride_i = (ny + 1) * (nz + 1)
    stride_j = nz + 1
    ii = used_old_point_ids // stride_i
    rem = used_old_point_ids % stride_i
    jj = rem // stride_j
    kk = rem % stride_j
    points = np.column_stack((x_edges[ii], y_edges[jj], z_edges[kk]))

    boundary_faces: Dict[str, np.ndarray] = {}
    boundary_face_centers: Dict[str, np.ndarray] = {}
    boundary_counts: Dict[str, int] = {}
    for tag, face_list in boundary_faces_old.items():
        if face_list:
            face_old_arr = np.asarray(face_list, dtype=np.int64)
            boundary_faces[tag] = point_map[face_old_arr]
            boundary_face_centers[tag] = np.asarray(
                boundary_centers[tag], dtype=np.float64
            )
        else:
            boundary_faces[tag] = np.zeros((0, 4), dtype=np.int64)
            boundary_face_centers[tag] = np.zeros((0, 3), dtype=np.float64)
        boundary_counts[tag] = int(boundary_faces[tag].shape[0])

    layer_ids = _compute_wall_layers(
        fluid_cell_ijk=fluid_cell_ijk,
        fluid_mask=fluid_mask,
        wall_seed_ids=wall_seed_ids,
        max_layers=max(refinement.wall_refinement_levels, 0),
    )
    layer_spacing = min(
        (x_edges[1] - x_edges[0]),
        (y_edges[1] - y_edges[0]),
        (z_edges[1] - z_edges[0]),
    )
    wall_distance_hint = np.where(layer_ids >= 0, (layer_ids + 1) * layer_spacing, -1.0)

    cell_data = {
        "zone_id": zone_ids.astype(np.int32),
        "wall_layer_id": layer_ids.astype(np.int32),
        "wall_distance_hint": wall_distance_hint.astype(np.float64),
    }
    cell_types = np.full(cells.shape[0], VTK_HEXAHEDRON, dtype=np.uint8)
    wall_coverage = float(np.mean(layer_ids >= 0)) if layer_ids.size else 0.0
    summary = MeshBuildSummary(
        total_candidate_cells=int(nx * ny * nz),
        fluid_cell_count=int(cells.shape[0]),
        point_count=int(points.shape[0]),
        boundary_face_counts=boundary_counts,
        wall_layer_coverage=wall_coverage,
    )
    metadata = {
        "resolution": tuple(int(v) for v in resolution),
        "dx": float(x_edges[1] - x_edges[0]),
        "dy": float(y_edges[1] - y_edges[0]),
        "dz": float(z_edges[1] - z_edges[0]),
        "boundary_tag_ids": dict(BOUNDARY_TAG_IDS),
        "zone_ids": dict(ZONE_IDS),
        "stage": (
            "pre-solver mesh scaffold with explicit connectivity and tags; "
            "not a full unstructured CFD solver"
        ),
    }
    return MeshDataModel(
        points=points.astype(np.float64),
        cells=cells.astype(np.int64),
        cell_types=cell_types,
        cell_centers=fluid_centers.astype(np.float64),
        cell_indices=fluid_cell_ijk.astype(np.int32),
        cell_data=cell_data,
        point_data={},
        boundary_faces=boundary_faces,
        boundary_face_centers=boundary_face_centers,
        metadata=metadata,
        summary=summary,
    )


def build_mesh_from_geometry_config(
    config: "GeometryPipelineConfig",
    output_directory: str | Path,
    *,
    gmsh_executable: str | Path | None = None,
) -> GeometryPipelineBuildResult:
    """Build the explicitly configured CAD/Gmsh or legacy mesh path.

    CAD and Gmsh failures are propagated. ``legacy_boxes`` is never selected as
    an automatic fallback; it must be requested by ``source.kind``.
    """

    config.validate()
    if config.source.kind != "legacy_boxes":
        from microfluidics.cad.pipeline import generate_cad_mesh

        return GeometryPipelineBuildResult(
            source_kind=config.source.kind,
            cad_mesh=generate_cad_mesh(
                config,
                output_directory,
                gmsh_executable=gmsh_executable,
            ),
        )

    params = config.geometry
    legacy_config = MeshGeometryConfig3D(
        inlet_width=params.inlet_width,
        outlet_width=params.outlet_width,
        channel_height=params.channel_height,
        inlet_length=params.inlet_length,
        outlet_length=params.outlet_length,
        include_cavities=params.include_cavities,
        cavity_depth=params.cavity_depth,
        cavity_length=params.cavity_length,
        cavity_offset_from_junction=params.cavity_offset_from_junction,
    )
    geometry = build_mesh_geometry_3d(legacy_config)
    legacy_mesh = build_mesh_scaffold(
        geometry,
        config.legacy_resolution,
        MeshRefinementConfig(volume_resolution=config.legacy_resolution),
    )
    return GeometryPipelineBuildResult(
        source_kind="legacy_boxes",
        legacy_mesh=legacy_mesh,
    )
