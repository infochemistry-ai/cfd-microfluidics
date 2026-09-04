"""Mesh-native operator base for cell-centered fields.

Assumptions of this operator layer:
1) Hexahedral scaffold with orthogonal axes (x, y, z).
2) Cell-centered finite-volume / finite-difference-like stencils.
3) Uniform spacing per axis from mesh metadata (dx, dy, dz).
4) Fluid domain can be non-rectangular through masked/absent neighbors.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple

import numpy as np

from microfluidics.mesh.mesh_builder import MeshDataModel

BoundaryMode = Literal["neumann0", "dirichlet"]

FACE_DIRECTIONS = ("x-", "x+", "y-", "y+", "z-", "z+")
FACE_NORMALS = np.array(
    [
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=np.float64,
)
OFFSETS = np.array(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class MeshTopology:
    """Neighbor graph and geometric data for mesh operators."""

    resolution: Tuple[int, int, int]
    id_grid: np.ndarray
    neighbor_ids: np.ndarray
    dx: float
    dy: float
    dz: float
    interior_mask: np.ndarray
    face_normals: np.ndarray = field(default_factory=lambda: FACE_NORMALS.copy())

    @property
    def n_cells(self) -> int:
        return int(self.neighbor_ids.shape[0])

    @property
    def inv_h2(self) -> Tuple[float, float, float]:
        return (
            1.0 / (self.dx * self.dx),
            1.0 / (self.dy * self.dy),
            1.0 / (self.dz * self.dz),
        )


def build_mesh_topology(mesh: MeshDataModel) -> MeshTopology:
    """Build mesh topology arrays from scaffold metadata."""
    if "resolution" not in mesh.metadata:
        raise ValueError(
            "mesh.metadata['resolution'] is required for mesh-native operators."
        )
    resolution = tuple(int(v) for v in mesh.metadata["resolution"])
    if len(resolution) != 3:
        raise ValueError(f"Expected 3D resolution in metadata, got {resolution}.")
    dx = float(mesh.metadata.get("dx", 0.0))
    dy = float(mesh.metadata.get("dy", 0.0))
    dz = float(mesh.metadata.get("dz", 0.0))
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"Invalid cell spacing metadata: dx={dx}, dy={dy}, dz={dz}.")

    nx, ny, nz = resolution
    id_grid = -np.ones((nx, ny, nz), dtype=np.int64)
    ijk = np.asarray(mesh.cell_indices, dtype=np.int64)
    id_grid[tuple(ijk.T)] = np.arange(ijk.shape[0], dtype=np.int64)
    neighbor_ids = -np.ones((ijk.shape[0], 6), dtype=np.int64)
    for face_idx, (di, dj, dk) in enumerate(OFFSETS):
        ni = ijk[:, 0] + di
        nj = ijk[:, 1] + dj
        nk = ijk[:, 2] + dk
        in_bounds = (
            (0 <= ni) & (ni < nx) & (0 <= nj) & (nj < ny) & (0 <= nk) & (nk < nz)
        )
        neigh = -np.ones(ijk.shape[0], dtype=np.int64)
        neigh[in_bounds] = id_grid[ni[in_bounds], nj[in_bounds], nk[in_bounds]]
        neighbor_ids[:, face_idx] = neigh
    interior_mask = np.all(neighbor_ids >= 0, axis=1)
    return MeshTopology(
        resolution=resolution,
        id_grid=id_grid,
        neighbor_ids=neighbor_ids,
        dx=dx,
        dy=dy,
        dz=dz,
        interior_mask=interior_mask,
    )


def cell_volumes(mesh: MeshDataModel) -> np.ndarray:
    dx = float(mesh.metadata["dx"])
    dy = float(mesh.metadata["dy"])
    dz = float(mesh.metadata["dz"])
    volume = dx * dy * dz
    return np.full(mesh.cells.shape[0], volume, dtype=np.float64)


def face_areas(mesh: MeshDataModel) -> np.ndarray:
    """Return per-cell face areas ordered as x-,x+,y-,y+,z-,z+."""
    dx = float(mesh.metadata["dx"])
    dy = float(mesh.metadata["dy"])
    dz = float(mesh.metadata["dz"])
    areas = np.array(
        [dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], dtype=np.float64
    )
    return np.repeat(areas[None, :], mesh.cells.shape[0], axis=0)


def cell_to_face_scalar(
    values: np.ndarray,
    topology: MeshTopology,
    boundary_mode: BoundaryMode = "neumann0",
    dirichlet_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != topology.n_cells:
        raise ValueError("cell_to_face_scalar expects one scalar per cell.")
    face_values = np.zeros((topology.n_cells, 6), dtype=np.float64)
    for face_idx in range(6):
        neigh = topology.neighbor_ids[:, face_idx]
        has_neigh = neigh >= 0
        face_values[has_neigh, face_idx] = 0.5 * (
            values[has_neigh] + values[neigh[has_neigh]]
        )
        if boundary_mode == "dirichlet":
            face_values[~has_neigh, face_idx] = 0.5 * (
                values[~has_neigh] + dirichlet_value
            )
        else:
            face_values[~has_neigh, face_idx] = values[~has_neigh]
    return face_values


def face_to_cell_scalar(face_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(face_values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise ValueError("face_to_cell_scalar expects array with shape (n_cells, 6).")
    return np.mean(arr, axis=1)


def _axis_derivative(
    values: np.ndarray,
    minus_neigh: np.ndarray,
    plus_neigh: np.ndarray,
    spacing: float,
    boundary_mode: BoundaryMode,
    dirichlet_value: float,
) -> np.ndarray:
    deriv = np.zeros(values.shape[0], dtype=np.float64)
    has_minus = minus_neigh >= 0
    has_plus = plus_neigh >= 0
    both = has_minus & has_plus
    deriv[both] = (values[plus_neigh[both]] - values[minus_neigh[both]]) / (
        2.0 * spacing
    )

    only_plus = (~has_minus) & has_plus
    only_minus = has_minus & (~has_plus)
    if boundary_mode == "dirichlet":
        deriv[only_plus] = (values[plus_neigh[only_plus]] - dirichlet_value) / (
            2.0 * spacing
        )
        deriv[only_minus] = (dirichlet_value - values[minus_neigh[only_minus]]) / (
            2.0 * spacing
        )
    else:
        deriv[only_plus] = (values[plus_neigh[only_plus]] - values[only_plus]) / spacing
        deriv[only_minus] = (
            values[only_minus] - values[minus_neigh[only_minus]]
        ) / spacing
    return deriv


def gradient_cell_scalar(
    values: np.ndarray,
    topology: MeshTopology,
    boundary_mode: BoundaryMode = "neumann0",
    dirichlet_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != topology.n_cells:
        raise ValueError("gradient_cell_scalar expects one scalar per cell.")

    dphidx = _axis_derivative(
        values,
        topology.neighbor_ids[:, 0],
        topology.neighbor_ids[:, 1],
        topology.dx,
        boundary_mode,
        dirichlet_value,
    )
    dphidy = _axis_derivative(
        values,
        topology.neighbor_ids[:, 2],
        topology.neighbor_ids[:, 3],
        topology.dy,
        boundary_mode,
        dirichlet_value,
    )
    dphidz = _axis_derivative(
        values,
        topology.neighbor_ids[:, 4],
        topology.neighbor_ids[:, 5],
        topology.dz,
        boundary_mode,
        dirichlet_value,
    )
    return np.column_stack((dphidx, dphidy, dphidz))


def divergence_cell_vector(
    vectors: np.ndarray,
    topology: MeshTopology,
    boundary_mode: BoundaryMode = "neumann0",
    dirichlet_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("divergence_cell_vector expects shape (n_cells, 3).")
    if arr.shape[0] != topology.n_cells:
        raise ValueError("divergence_cell_vector expects one vector per mesh cell.")
    duxdx = _axis_derivative(
        arr[:, 0],
        topology.neighbor_ids[:, 0],
        topology.neighbor_ids[:, 1],
        topology.dx,
        boundary_mode,
        dirichlet_vector[0],
    )
    duydy = _axis_derivative(
        arr[:, 1],
        topology.neighbor_ids[:, 2],
        topology.neighbor_ids[:, 3],
        topology.dy,
        boundary_mode,
        dirichlet_vector[1],
    )
    duzdz = _axis_derivative(
        arr[:, 2],
        topology.neighbor_ids[:, 4],
        topology.neighbor_ids[:, 5],
        topology.dz,
        boundary_mode,
        dirichlet_vector[2],
    )
    return duxdx + duydy + duzdz


def _axis_second_derivative(
    values: np.ndarray,
    minus_neigh: np.ndarray,
    plus_neigh: np.ndarray,
    inv_h2: float,
    boundary_mode: BoundaryMode,
    dirichlet_value: float,
) -> np.ndarray:
    second = np.zeros(values.shape[0], dtype=np.float64)
    has_minus = minus_neigh >= 0
    has_plus = plus_neigh >= 0
    both = has_minus & has_plus
    second[both] = (
        values[plus_neigh[both]] - 2.0 * values[both] + values[minus_neigh[both]]
    ) * inv_h2

    only_plus = (~has_minus) & has_plus
    only_minus = has_minus & (~has_plus)
    if boundary_mode == "dirichlet":
        second[only_plus] = (
            values[plus_neigh[only_plus]] - 2.0 * values[only_plus] + dirichlet_value
        ) * inv_h2
        second[only_minus] = (
            dirichlet_value - 2.0 * values[only_minus] + values[minus_neigh[only_minus]]
        ) * inv_h2
    else:
        second[only_plus] = (values[plus_neigh[only_plus]] - values[only_plus]) * inv_h2
        second[only_minus] = (
            values[minus_neigh[only_minus]] - values[only_minus]
        ) * inv_h2
    return second


def laplacian_cell_scalar(
    values: np.ndarray,
    topology: MeshTopology,
    boundary_mode: BoundaryMode = "neumann0",
    dirichlet_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != topology.n_cells:
        raise ValueError("laplacian_cell_scalar expects one scalar per cell.")
    inv_dx2, inv_dy2, inv_dz2 = topology.inv_h2
    d2x = _axis_second_derivative(
        values,
        topology.neighbor_ids[:, 0],
        topology.neighbor_ids[:, 1],
        inv_dx2,
        boundary_mode,
        dirichlet_value,
    )
    d2y = _axis_second_derivative(
        values,
        topology.neighbor_ids[:, 2],
        topology.neighbor_ids[:, 3],
        inv_dy2,
        boundary_mode,
        dirichlet_value,
    )
    d2z = _axis_second_derivative(
        values,
        topology.neighbor_ids[:, 4],
        topology.neighbor_ids[:, 5],
        inv_dz2,
        boundary_mode,
        dirichlet_value,
    )
    return d2x + d2y + d2z


def compute_operator_diagnostics(
    mesh: MeshDataModel,
    topology: MeshTopology,
) -> Dict[str, float]:
    """Run verification diagnostics on analytical fields."""
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    interior = topology.interior_mask
    if not np.any(interior):
        raise RuntimeError("No interior cells found for operator diagnostics.")

    linear = 2.0 * centers[:, 0] - 3.0 * centers[:, 1] + 0.5 * centers[:, 2]
    grad = gradient_cell_scalar(linear, topology, boundary_mode="neumann0")
    grad_exact = np.array([2.0, -3.0, 0.5], dtype=np.float64)
    grad_error = np.max(np.abs(grad[interior] - grad_exact[None, :]))

    quadratic = centers[:, 0] ** 2 + centers[:, 1] ** 2 + centers[:, 2] ** 2
    lap = laplacian_cell_scalar(quadratic, topology, boundary_mode="neumann0")
    lap_error = np.max(np.abs(lap[interior] - 6.0))

    vector = np.column_stack((centers[:, 0], centers[:, 1], centers[:, 2]))
    div = divergence_cell_vector(vector, topology, boundary_mode="neumann0")
    div_error = np.max(np.abs(div[interior] - 3.0))

    return {
        "interior_cell_count": float(np.count_nonzero(interior)),
        "gradient_linear_max_abs_error": float(grad_error),
        "laplacian_quadratic_max_abs_error": float(lap_error),
        "divergence_linear_max_abs_error": float(div_error),
    }
