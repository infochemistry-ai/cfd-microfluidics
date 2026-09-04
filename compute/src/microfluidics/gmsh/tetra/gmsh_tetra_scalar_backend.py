"""Shared scalar backend helpers for tetra transport and thermal solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Sequence

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    GradientMethod,
    LaplacianMethod,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
    DiffusionStencil,
    build_diffusion_boundary_maps,
    build_diffusion_stencil,
    resolve_inlet_face_groups,
)

ScalarLimiterScheme = Literal["upwind", "bounded_upwind"]


BOUNDARY_GROUP_CODE = {
    "unresolved": 0,
    "left_inlet": 1,
    "right_inlet": 2,
    "outlet": 3,
    "wall": 4,
}

BOUNDARY_GROUP_NAME = {code: name for name, code in BOUNDARY_GROUP_CODE.items()}


@dataclass(frozen=True)
class ScalarBackendPrecompute:
    mesh: ImportedTetraMesh
    face_normal_velocity: np.ndarray
    face_to_cells: np.ndarray
    face_areas: np.ndarray
    cell_volumes: np.ndarray
    boundary_face_groups: np.ndarray
    left_faces: np.ndarray
    right_faces: np.ndarray
    outlet_faces: np.ndarray
    wall_faces: np.ndarray
    boundary_faces: np.ndarray
    unresolved_faces: np.ndarray
    left_cells: np.ndarray
    right_cells: np.ndarray
    outlet_cells: np.ndarray
    left_zone_mask: np.ndarray
    right_zone_mask: np.ndarray
    corner_zone_mask: np.ndarray
    boundary_dirichlet: Dict[int, float]
    boundary_no_flux_faces: np.ndarray
    diffusion_stencil: DiffusionStencil | None
    diffusion_data: dict[str, object] | None
    torch_backend: dict[str, object] | None


def build_boundary_face_groups(
    mesh: ImportedTetraMesh,
    *,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> np.ndarray:
    groups = np.full(
        mesh.face_vertices.shape[0], BOUNDARY_GROUP_CODE["unresolved"], dtype=np.int32
    )
    groups[np.asarray(wall_faces, dtype=np.int64)] = BOUNDARY_GROUP_CODE["wall"]
    groups[np.asarray(outlet_faces, dtype=np.int64)] = BOUNDARY_GROUP_CODE["outlet"]
    groups[np.asarray(left_faces, dtype=np.int64)] = BOUNDARY_GROUP_CODE["left_inlet"]
    groups[np.asarray(right_faces, dtype=np.int64)] = BOUNDARY_GROUP_CODE["right_inlet"]
    return groups


def build_inlet_zone_masks(
    mesh: ImportedTetraMesh,
    *,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = mesh.tetrahedra.shape[0]
    left_mask = np.zeros(n_cells, dtype=bool)
    right_mask = np.zeros(n_cells, dtype=bool)
    corner_mask = np.zeros(n_cells, dtype=bool)
    wall_mask = np.zeros(n_cells, dtype=bool)

    if left_faces.size:
        left_cells = np.unique(
            mesh.face_to_cells[np.asarray(left_faces, dtype=np.int64), 0]
        )
        left_mask[left_cells] = True
    if right_faces.size:
        right_cells = np.unique(
            mesh.face_to_cells[np.asarray(right_faces, dtype=np.int64), 0]
        )
        right_mask[right_cells] = True
    if wall_faces.size:
        wall_cells = np.unique(
            mesh.face_to_cells[np.asarray(wall_faces, dtype=np.int64), 0]
        )
        wall_mask[wall_cells] = True
    corner_mask = wall_mask & (left_mask | right_mask)
    return left_mask, right_mask, corner_mask


def _build_diffusion_backend_data(
    mesh: ImportedTetraMesh,
    *,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
    diffusion_stencil: DiffusionStencil | None,
) -> dict[str, object] | None:
    if diffusion_stencil is None:
        return None

    stencil = diffusion_stencil
    n_cells = mesh.tetrahedra.shape[0]
    reg = 1e-14
    mat = np.zeros((n_cells, 3, 3), dtype=np.float64)
    eq_count = np.zeros((n_cells,), dtype=np.int32)
    c0_i = np.asarray(stencil.interior_c0, dtype=np.int64)
    c1_i = np.asarray(stencil.interior_c1, dtype=np.int64)
    delta_x = mesh.cell_centers[c1_i] - mesh.cell_centers[c0_i]
    w_i = 1.0 / np.maximum(np.einsum("ij,ij->i", delta_x, delta_x), 1e-20)
    if c0_i.size:
        outer_terms = w_i[:, None, None] * np.einsum("ni,nj->nij", delta_x, delta_x)
        np.add.at(mat, c0_i, outer_terms)
        np.add.at(mat, c1_i, outer_terms)
        np.add.at(eq_count, c0_i, 1)
        np.add.at(eq_count, c1_i, 1)

    db_owner = np.asarray(stencil.dirichlet_c0, dtype=np.int64)
    db_delta = np.asarray(stencil.dirichlet_cell_to_face_delta, dtype=np.float64)
    db_value = np.asarray(stencil.dirichlet_value, dtype=np.float64)
    db_norm2 = (
        np.einsum("ij,ij->i", db_delta, db_delta)
        if db_delta.size
        else np.zeros((0,), dtype=np.float64)
    )
    db_w = np.where(db_norm2 > 1e-20, 1.0 / db_norm2, 0.0)
    if db_owner.size:
        db_outer_terms = db_w[:, None, None] * np.einsum(
            "ni,nj->nij", db_delta, db_delta
        )
        np.add.at(mat, db_owner, db_outer_terms)
        np.add.at(eq_count, db_owner, (db_w > 0.0).astype(np.int32))

    mat += reg * np.eye(3, dtype=np.float64)[None, :, :]
    solve_mask = eq_count >= 3
    inv = np.zeros_like(mat)
    if np.any(solve_mask):
        inv[solve_mask] = np.linalg.inv(mat[solve_mask])

    no_flux_mask_np = np.zeros((mesh.face_vertices.shape[0],), dtype=bool)
    no_flux_mask_np[np.asarray(boundary_no_flux_faces, dtype=np.int64)] = True

    boundary_face_ids = np.arange(mesh.face_vertices.shape[0], dtype=np.int64)
    boundary_has_dirichlet = np.isin(
        boundary_face_ids,
        np.asarray(list(boundary_dirichlet.keys()), dtype=np.int64),
    )
    boundary_dirichlet_values = np.asarray(
        [boundary_dirichlet.get(int(face_idx), 0.0) for face_idx in boundary_face_ids],
        dtype=np.float64,
    )

    return {
        "st_interior_c0": np.asarray(stencil.interior_c0, dtype=np.int64),
        "st_interior_c1": np.asarray(stencil.interior_c1, dtype=np.int64),
        "st_interior_alpha": np.asarray(stencil.interior_alpha, dtype=np.float64),
        "st_interior_correction": np.asarray(
            stencil.interior_correction_vec, dtype=np.float64
        ),
        "st_dirichlet_c0": np.asarray(stencil.dirichlet_c0, dtype=np.int64),
        "st_dirichlet_coeff": np.asarray(stencil.dirichlet_coeff, dtype=np.float64),
        "st_dirichlet_value": np.asarray(stencil.dirichlet_value, dtype=np.float64),
        "st_neumann_c0": np.asarray(stencil.neumann_c0, dtype=np.int64),
        "st_neumann_area": np.asarray(stencil.neumann_area, dtype=np.float64),
        "st_neumann_gradient": np.asarray(
            stencil.neumann_normal_gradient, dtype=np.float64
        ),
        "st_robin_c0": np.asarray(stencil.robin_c0, dtype=np.int64),
        "st_robin_area": np.asarray(stencil.robin_area, dtype=np.float64),
        "st_robin_distance": np.asarray(stencil.robin_distance, dtype=np.float64),
        "st_robin_alpha": np.asarray(stencil.robin_alpha, dtype=np.float64),
        "st_robin_beta": np.asarray(stencil.robin_beta, dtype=np.float64),
        "st_robin_gamma": np.asarray(stencil.robin_gamma, dtype=np.float64),
        "gg_no_flux_mask": no_flux_mask_np,
        "gg_boundary_has_dirichlet": boundary_has_dirichlet,
        "gg_boundary_dirichlet_values": boundary_dirichlet_values,
        "gg_weighted_normals": np.asarray(
            mesh.face_normals * mesh.face_areas[:, None], dtype=np.float64
        ),
        "gg_c0": np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64),
        "gg_c1": np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64),
        "lsq_c0": c0_i,
        "lsq_c1": c1_i,
        "lsq_r": delta_x,
        "lsq_w": w_i,
        "lsq_db_owner": db_owner,
        "lsq_db_r": db_delta,
        "lsq_db_w": db_w,
        "lsq_db_value": db_value,
        "lsq_inv": inv,
        "lsq_solve_mask": solve_mask,
    }


def _build_torch_diffusion_data(
    diffusion_data: dict[str, object] | None,
    *,
    torch: Any,
    device: Any,
    dtype: Any,
) -> dict[str, object] | None:
    if diffusion_data is None:
        return None
    return {
        "st_interior_c0": torch.as_tensor(
            diffusion_data["st_interior_c0"], dtype=torch.long, device=device
        ),
        "st_interior_c1": torch.as_tensor(
            diffusion_data["st_interior_c1"], dtype=torch.long, device=device
        ),
        "st_interior_alpha": torch.as_tensor(
            diffusion_data["st_interior_alpha"], dtype=dtype, device=device
        ),
        "st_interior_correction": torch.as_tensor(
            diffusion_data["st_interior_correction"], dtype=dtype, device=device
        ),
        "st_dirichlet_c0": torch.as_tensor(
            diffusion_data["st_dirichlet_c0"], dtype=torch.long, device=device
        ),
        "st_dirichlet_coeff": torch.as_tensor(
            diffusion_data["st_dirichlet_coeff"], dtype=dtype, device=device
        ),
        "st_dirichlet_value": torch.as_tensor(
            diffusion_data["st_dirichlet_value"], dtype=dtype, device=device
        ),
        "st_neumann_c0": torch.as_tensor(
            diffusion_data["st_neumann_c0"], dtype=torch.long, device=device
        ),
        "st_neumann_area": torch.as_tensor(
            diffusion_data["st_neumann_area"], dtype=dtype, device=device
        ),
        "st_neumann_gradient": torch.as_tensor(
            diffusion_data["st_neumann_gradient"], dtype=dtype, device=device
        ),
        "st_robin_c0": torch.as_tensor(
            diffusion_data["st_robin_c0"], dtype=torch.long, device=device
        ),
        "st_robin_area": torch.as_tensor(
            diffusion_data["st_robin_area"], dtype=dtype, device=device
        ),
        "st_robin_distance": torch.as_tensor(
            diffusion_data["st_robin_distance"], dtype=dtype, device=device
        ),
        "st_robin_alpha": torch.as_tensor(
            diffusion_data["st_robin_alpha"], dtype=dtype, device=device
        ),
        "st_robin_beta": torch.as_tensor(
            diffusion_data["st_robin_beta"], dtype=dtype, device=device
        ),
        "st_robin_gamma": torch.as_tensor(
            diffusion_data["st_robin_gamma"], dtype=dtype, device=device
        ),
        "gg_no_flux_mask": torch.as_tensor(
            diffusion_data["gg_no_flux_mask"], dtype=torch.bool, device=device
        ),
        "gg_boundary_has_dirichlet": torch.as_tensor(
            diffusion_data["gg_boundary_has_dirichlet"],
            dtype=torch.bool,
            device=device,
        ),
        "gg_boundary_dirichlet_values": torch.as_tensor(
            diffusion_data["gg_boundary_dirichlet_values"], dtype=dtype, device=device
        ),
        "gg_weighted_normals": torch.as_tensor(
            diffusion_data["gg_weighted_normals"], dtype=dtype, device=device
        ),
        "gg_c0": torch.as_tensor(
            diffusion_data["gg_c0"], dtype=torch.long, device=device
        ),
        "gg_c1": torch.as_tensor(
            diffusion_data["gg_c1"], dtype=torch.long, device=device
        ),
        "lsq_c0": torch.as_tensor(
            diffusion_data["lsq_c0"], dtype=torch.long, device=device
        ),
        "lsq_c1": torch.as_tensor(
            diffusion_data["lsq_c1"], dtype=torch.long, device=device
        ),
        "lsq_r": torch.as_tensor(diffusion_data["lsq_r"], dtype=dtype, device=device),
        "lsq_w": torch.as_tensor(diffusion_data["lsq_w"], dtype=dtype, device=device),
        "lsq_db_owner": torch.as_tensor(
            diffusion_data["lsq_db_owner"], dtype=torch.long, device=device
        ),
        "lsq_db_r": torch.as_tensor(
            diffusion_data["lsq_db_r"], dtype=dtype, device=device
        ),
        "lsq_db_w": torch.as_tensor(
            diffusion_data["lsq_db_w"], dtype=dtype, device=device
        ),
        "lsq_db_value": torch.as_tensor(
            diffusion_data["lsq_db_value"], dtype=dtype, device=device
        ),
        "lsq_inv": torch.as_tensor(
            diffusion_data["lsq_inv"], dtype=dtype, device=device
        ),
        "lsq_solve_mask": torch.as_tensor(
            diffusion_data["lsq_solve_mask"], dtype=torch.bool, device=device
        ),
    }


def build_scalar_backend_precompute(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    backend: Literal["numpy", "torch"],
    torch_device: str = "cpu",
    diffusion_dirichlet_left_value: float | None = None,
    diffusion_dirichlet_right_value: float | None = None,
    diffusion_boundary_dirichlet: Mapping[int, float] | None = None,
    diffusion_boundary_neumann: Mapping[int, float] | None = None,
    diffusion_boundary_robin: Mapping[int, tuple[float, float, float]] | None = None,
    diffusion_periodic_face_pairs: Sequence[tuple[int, int]] = (),
) -> ScalarBackendPrecompute:
    inlet_groups = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet_groups["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet_groups["right_faces"], dtype=np.int64)
    left_cells = np.asarray(inlet_groups["left_cells"], dtype=np.int64)
    right_cells = np.asarray(inlet_groups["right_cells"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    boundary_faces = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    outlet_cells = (
        np.unique(mesh.face_to_cells[outlet_faces, 0])
        if outlet_faces.size
        else np.zeros((0,), dtype=np.int64)
    )
    boundary_face_groups = build_boundary_face_groups(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    unresolved_faces = boundary_faces[
        boundary_face_groups[boundary_faces] == BOUNDARY_GROUP_CODE["unresolved"]
    ]
    left_zone_mask, right_zone_mask, corner_zone_mask = build_inlet_zone_masks(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        wall_faces=wall_faces,
    )

    boundary_dirichlet: Dict[int, float] = {}
    boundary_no_flux_faces = np.zeros((0,), dtype=np.int64)
    diffusion_stencil: DiffusionStencil | None = None
    explicit_boundary_contract = any(
        item is not None
        for item in (
            diffusion_boundary_dirichlet,
            diffusion_boundary_neumann,
            diffusion_boundary_robin,
        )
    ) or bool(diffusion_periodic_face_pairs)
    if explicit_boundary_contract:
        boundary_dirichlet = {
            int(face): float(value)
            for face, value in (diffusion_boundary_dirichlet or {}).items()
        }
        neumann = {
            int(face): float(value)
            for face, value in (diffusion_boundary_neumann or {}).items()
        }
        robin = {
            int(face): (float(value[0]), float(value[1]), float(value[2]))
            for face, value in (diffusion_boundary_robin or {}).items()
        }
        periodic_faces = {
            int(face) for pair in diffusion_periodic_face_pairs for face in pair
        }
        non_dirichlet_claimed = set(neumann) | set(robin) | periodic_faces
        if diffusion_dirichlet_left_value is not None:
            for face in left_faces.tolist():
                if int(face) not in non_dirichlet_claimed:
                    boundary_dirichlet.setdefault(
                        int(face), float(diffusion_dirichlet_left_value)
                    )
        if diffusion_dirichlet_right_value is not None:
            for face in right_faces.tolist():
                if int(face) not in non_dirichlet_claimed:
                    boundary_dirichlet.setdefault(
                        int(face), float(diffusion_dirichlet_right_value)
                    )
        claimed = set(boundary_dirichlet) | set(neumann) | set(robin) | periodic_faces
        boundary_no_flux_faces = np.asarray(
            sorted(set(boundary_faces.tolist()) - claimed), dtype=np.int64
        )
        diffusion_stencil = build_diffusion_stencil(
            mesh,
            boundary_dirichlet=boundary_dirichlet,
            boundary_no_flux_faces=boundary_no_flux_faces,
            boundary_neumann=neumann,
            boundary_robin=robin,
            periodic_face_pairs=diffusion_periodic_face_pairs,
        )
    elif (
        diffusion_dirichlet_left_value is not None
        and diffusion_dirichlet_right_value is not None
    ):
        boundary_dirichlet, boundary_no_flux_faces = build_diffusion_boundary_maps(
            mesh,
            inlet_groups,
            left_value=float(diffusion_dirichlet_left_value),
            right_value=float(diffusion_dirichlet_right_value),
        )
        if diffusion_boundary_dirichlet:
            extra_faces = np.asarray(
                list(diffusion_boundary_dirichlet.keys()), dtype=np.int64
            )
            boundary_dirichlet.update(
                {
                    int(face_idx): float(value)
                    for face_idx, value in diffusion_boundary_dirichlet.items()
                }
            )
            boundary_no_flux_faces = boundary_no_flux_faces[
                ~np.isin(boundary_no_flux_faces, extra_faces)
            ]
        diffusion_stencil = build_diffusion_stencil(
            mesh,
            boundary_dirichlet=boundary_dirichlet,
            boundary_no_flux_faces=boundary_no_flux_faces,
        )

    diffusion_data = _build_diffusion_backend_data(
        mesh,
        boundary_dirichlet=boundary_dirichlet,
        boundary_no_flux_faces=boundary_no_flux_faces,
        diffusion_stencil=diffusion_stencil,
    )

    torch_backend = None
    if backend == "torch":
        torch_backend = _build_torch_backend(
            mesh,
            face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
            boundary_face_groups=boundary_face_groups,
            left_faces=left_faces,
            right_faces=right_faces,
            outlet_faces=outlet_faces,
            boundary_faces=boundary_faces,
            unresolved_faces=unresolved_faces,
            left_cells=left_cells,
            right_cells=right_cells,
            outlet_cells=outlet_cells,
            left_zone_mask=left_zone_mask,
            right_zone_mask=right_zone_mask,
            corner_zone_mask=corner_zone_mask,
            boundary_dirichlet=boundary_dirichlet,
            boundary_no_flux_faces=boundary_no_flux_faces,
            diffusion_stencil=diffusion_stencil,
            diffusion_data=diffusion_data,
            torch_device=torch_device,
        )

    return ScalarBackendPrecompute(
        mesh=mesh,
        face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
        face_to_cells=np.asarray(mesh.face_to_cells, dtype=np.int64),
        face_areas=np.asarray(mesh.face_areas, dtype=np.float64),
        cell_volumes=np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-16),
        boundary_face_groups=boundary_face_groups,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        boundary_faces=boundary_faces,
        unresolved_faces=unresolved_faces,
        left_cells=left_cells,
        right_cells=right_cells,
        outlet_cells=np.asarray(outlet_cells, dtype=np.int64),
        left_zone_mask=left_zone_mask,
        right_zone_mask=right_zone_mask,
        corner_zone_mask=corner_zone_mask,
        boundary_dirichlet=boundary_dirichlet,
        boundary_no_flux_faces=np.asarray(boundary_no_flux_faces, dtype=np.int64),
        diffusion_stencil=diffusion_stencil,
        diffusion_data=diffusion_data,
        torch_backend=torch_backend,
    )


def compute_outgoing_flux_sum(precompute: ScalarBackendPrecompute) -> np.ndarray:
    out_flux_sum = np.zeros(precompute.cell_volumes.shape[0], dtype=np.float64)
    c0 = precompute.face_to_cells[:, 0]
    c1 = precompute.face_to_cells[:, 1]
    q = np.asarray(precompute.face_normal_velocity, dtype=np.float64) * np.asarray(
        precompute.face_areas, dtype=np.float64
    )
    interior = c1 >= 0
    if np.any(interior):
        interior_faces = np.flatnonzero(interior)
        q_interior = q[interior_faces]
        pos = q_interior >= 0.0
        donor = np.where(pos, c0[interior_faces], c1[interior_faces]).astype(
            np.int64, copy=False
        )
        np.add.at(out_flux_sum, donor, np.abs(q_interior))
    boundary_out = (~interior) & (q > 0.0)
    if np.any(boundary_out):
        np.add.at(out_flux_sum, c0[boundary_out], q[boundary_out])
    return out_flux_sum


def compute_cfl_metrics(
    precompute: ScalarBackendPrecompute,
    dt: float,
) -> tuple[np.ndarray, float]:
    out_flux_sum = compute_outgoing_flux_sum(precompute)
    cfl = dt * out_flux_sum / np.maximum(precompute.cell_volumes, 1e-16)
    return cfl, float(np.max(cfl))


def estimate_stable_dt(
    precompute: ScalarBackendPrecompute, *, eps: float = 1e-20
) -> float:
    out_flux_sum = compute_outgoing_flux_sum(precompute)
    active = out_flux_sum > eps
    if not np.any(active):
        return float("inf")
    return float(np.min(precompute.cell_volumes[active] / out_flux_sum[active]))


def estimate_diffusion_dt_limit(
    stencil: DiffusionStencil,
    diffusivity: float,
) -> float:
    if diffusivity <= 0.0:
        return float("inf")
    coeff_sum = np.zeros(stencil.volumes.shape[0], dtype=np.float64)
    np.add.at(coeff_sum, stencil.interior_c0, stencil.interior_coeff)
    np.add.at(coeff_sum, stencil.interior_c1, stencil.interior_coeff)
    np.add.at(coeff_sum, stencil.dirichlet_c0, stencil.dirichlet_coeff)
    if stencil.robin_c0.size:
        denominator = stencil.robin_beta + stencil.robin_alpha * stencil.robin_distance
        robin_coeff = np.abs(stencil.robin_area * stencil.robin_alpha / denominator)
        np.add.at(coeff_sum, stencil.robin_c0, robin_coeff)
    positive = coeff_sum > 0.0
    if not np.any(positive):
        return float("inf")
    local_dt = stencil.volumes[positive] / (diffusivity * coeff_sum[positive])
    return float(0.5 * np.min(local_dt))


def assemble_advection_fluxes_numpy(
    precompute: ScalarBackendPrecompute,
    scalar: np.ndarray,
    *,
    left_inlet_value: float,
    right_inlet_value: float,
    dt: float,
) -> Dict[str, np.ndarray | float]:
    """Assemble upwind scalar fluxes and per-cell incoming/outgoing masses."""
    volumes = np.maximum(precompute.cell_volumes, 1e-16)
    n_cells = volumes.shape[0]
    c0 = np.asarray(precompute.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(precompute.face_to_cells[:, 1], dtype=np.int64)
    q = np.asarray(precompute.face_normal_velocity, dtype=np.float64) * np.asarray(
        precompute.face_areas, dtype=np.float64
    )
    abs_q = np.abs(q)
    boundary_face_groups = np.asarray(precompute.boundary_face_groups, dtype=np.int32)
    n_faces = q.shape[0]

    out_mass = np.zeros(n_cells, dtype=np.float64)
    in_mass = np.zeros(n_cells, dtype=np.float64)
    boundary_in_mass = np.zeros(n_cells, dtype=np.float64)
    boundary_out_mass = np.zeros(n_cells, dtype=np.float64)
    out_vol_rate = np.zeros(n_cells, dtype=np.float64)
    in_vol_rate = np.zeros(n_cells, dtype=np.float64)

    face_scalar_flux_raw = np.zeros(n_faces, dtype=np.float64)
    face_volumetric_flux_q = q.astype(np.float64, copy=True)
    face_upwind_scalar = np.asarray(scalar, dtype=np.float64)[c0].astype(
        np.float64, copy=True
    )
    face_mass_delta_owner = np.zeros(n_faces, dtype=np.float64)
    face_mass_delta_neighbor = np.zeros(n_faces, dtype=np.float64)
    face_contribution_c_owner = np.zeros(n_faces, dtype=np.float64)
    face_contribution_c_neighbor = np.zeros(n_faces, dtype=np.float64)
    face_owner = c0.astype(np.int64, copy=True)
    face_neighbor = c1.astype(np.int64, copy=True)
    face_boundary_group = boundary_face_groups.astype(np.int32, copy=True)

    scalar_arr = np.asarray(scalar, dtype=np.float64)
    interior_mask = c1 >= 0
    boundary_mask = ~interior_mask

    interior_faces_arr = np.flatnonzero(interior_mask).astype(np.int64)
    donor_cells = np.zeros((0,), dtype=np.int64)
    recv_cells = np.zeros((0,), dtype=np.int64)
    transfer_mass_raw = np.zeros((0,), dtype=np.float64)
    if interior_faces_arr.size:
        owner_i = c0[interior_faces_arr]
        neigh_i = c1[interior_faces_arr]
        q_i = q[interior_faces_arr]
        q_i_pos = q_i >= 0.0
        donor_cells = np.where(q_i_pos, owner_i, neigh_i).astype(np.int64, copy=False)
        recv_cells = np.where(q_i_pos, neigh_i, owner_i).astype(np.int64, copy=False)
        face_upwind_scalar[interior_faces_arr] = scalar_arr[donor_cells]
        phi_i = q_i * face_upwind_scalar[interior_faces_arr]
        transfer_mass_raw = dt * np.abs(phi_i)
        face_scalar_flux_raw[interior_faces_arr] = phi_i
        np.add.at(out_mass, donor_cells, transfer_mass_raw)
        np.add.at(in_mass, recv_cells, transfer_mass_raw)
        np.add.at(out_vol_rate, donor_cells, np.abs(q_i))
        np.add.at(in_vol_rate, recv_cells, np.abs(q_i))
        face_mass_delta_owner[interior_faces_arr] = -dt * phi_i
        face_mass_delta_neighbor[interior_faces_arr] = dt * phi_i
        face_contribution_c_owner[interior_faces_arr] = (
            face_mass_delta_owner[interior_faces_arr] / volumes[owner_i]
        )
        face_contribution_c_neighbor[interior_faces_arr] = (
            face_mass_delta_neighbor[interior_faces_arr] / volumes[neigh_i]
        )

    left_in = (
        boundary_mask
        & (boundary_face_groups == BOUNDARY_GROUP_CODE["left_inlet"])
        & (q < 0.0)
    )
    right_in = (
        boundary_mask
        & (boundary_face_groups == BOUNDARY_GROUP_CODE["right_inlet"])
        & (q < 0.0)
    )
    left_out = (
        boundary_mask
        & (boundary_face_groups == BOUNDARY_GROUP_CODE["left_inlet"])
        & (q >= 0.0)
    )
    right_out = (
        boundary_mask
        & (boundary_face_groups == BOUNDARY_GROUP_CODE["right_inlet"])
        & (q >= 0.0)
    )
    outlet_out = (
        boundary_mask
        & (boundary_face_groups == BOUNDARY_GROUP_CODE["outlet"])
        & (q > 0.0)
    )
    unresolved_out = (
        boundary_mask
        & (boundary_face_groups != BOUNDARY_GROUP_CODE["wall"])
        & (boundary_face_groups != BOUNDARY_GROUP_CODE["left_inlet"])
        & (boundary_face_groups != BOUNDARY_GROUP_CODE["right_inlet"])
        & (boundary_face_groups != BOUNDARY_GROUP_CODE["outlet"])
        & (q > 0.0)
    )

    def _apply_boundary_inflow(mask: np.ndarray, value: float) -> None:
        if not np.any(mask):
            return
        owner = c0[mask]
        phi = q[mask] * float(value)
        m = -dt * phi
        face_scalar_flux_raw[mask] = phi
        face_upwind_scalar[mask] = float(value)
        np.add.at(in_mass, owner, m)
        np.add.at(boundary_in_mass, owner, m)
        np.add.at(in_vol_rate, owner, abs_q[mask])
        face_mass_delta_owner[mask] = -dt * phi
        face_contribution_c_owner[mask] = face_mass_delta_owner[mask] / volumes[owner]

    def _apply_boundary_outflow(
        mask: np.ndarray,
        *,
        accumulate_outlet_flux: bool = False,
    ) -> None:
        if not np.any(mask):
            return
        owner = c0[mask]
        phi = q[mask] * scalar_arr[owner]
        m = dt * phi
        face_scalar_flux_raw[mask] = phi
        face_upwind_scalar[mask] = scalar_arr[owner]
        np.add.at(out_mass, owner, m)
        np.add.at(boundary_out_mass, owner, m)
        np.add.at(out_vol_rate, owner, abs_q[mask])
        face_mass_delta_owner[mask] = -dt * phi
        face_contribution_c_owner[mask] = face_mass_delta_owner[mask] / volumes[owner]
        if accumulate_outlet_flux:
            nonlocal_outlet[0] += float(np.sum(phi))

    inlet_scalar_flux_in = 0.0
    outlet_scalar_flux_out = 0.0
    nonlocal_outlet = [0.0]

    _apply_boundary_inflow(left_in, left_inlet_value)
    _apply_boundary_inflow(right_in, right_inlet_value)
    inlet_scalar_flux_in = float(
        -np.sum(face_scalar_flux_raw[left_in]) - np.sum(face_scalar_flux_raw[right_in])
    )
    _apply_boundary_outflow(left_out)
    _apply_boundary_outflow(right_out)
    _apply_boundary_outflow(outlet_out, accumulate_outlet_flux=True)
    _apply_boundary_outflow(unresolved_out)
    outlet_scalar_flux_out = nonlocal_outlet[0]

    boundary_in_faces = np.flatnonzero(left_in | right_in).astype(np.int64)
    boundary_out_faces = np.flatnonzero(
        left_out | right_out | outlet_out | unresolved_out
    ).astype(np.int64)
    boundary_in_cells = (
        c0[boundary_in_faces]
        if boundary_in_faces.size
        else np.zeros((0,), dtype=np.int64)
    )
    boundary_out_cells = (
        c0[boundary_out_faces]
        if boundary_out_faces.size
        else np.zeros((0,), dtype=np.int64)
    )
    boundary_in_mass_list = (
        face_mass_delta_owner[boundary_in_faces]
        if boundary_in_faces.size
        else np.zeros((0,), dtype=np.float64)
    )
    boundary_out_mass_list = (
        -face_mass_delta_owner[boundary_out_faces]
        if boundary_out_faces.size
        else np.zeros((0,), dtype=np.float64)
    )

    pairwise_err = np.abs(
        face_mass_delta_owner[interior_faces_arr]
        + face_mass_delta_neighbor[interior_faces_arr]
    )
    pairwise_max = float(np.max(pairwise_err)) if pairwise_err.size else 0.0
    pairwise_mean = float(np.mean(pairwise_err)) if pairwise_err.size else 0.0

    return {
        "face_scalar_flux_raw": face_scalar_flux_raw,
        "face_volumetric_flux_q": face_volumetric_flux_q,
        "face_upwind_scalar": face_upwind_scalar,
        "face_mass_delta_owner": face_mass_delta_owner,
        "face_mass_delta_neighbor": face_mass_delta_neighbor,
        "face_contribution_c_owner": face_contribution_c_owner,
        "face_contribution_c_neighbor": face_contribution_c_neighbor,
        "face_owner": face_owner,
        "face_neighbor": face_neighbor,
        "face_boundary_group": face_boundary_group,
        "out_mass_raw": out_mass,
        "in_mass_raw": in_mass,
        "out_vol_rate_raw": out_vol_rate,
        "in_vol_rate_raw": in_vol_rate,
        "boundary_in_mass_raw": boundary_in_mass,
        "boundary_out_mass_raw": boundary_out_mass,
        "inlet_scalar_flux_in": inlet_scalar_flux_in,
        "outlet_scalar_flux_out": outlet_scalar_flux_out,
        "interior_faces": interior_faces_arr,
        "donor_cells": np.asarray(donor_cells, dtype=np.int64),
        "recv_cells": np.asarray(recv_cells, dtype=np.int64),
        "transfer_mass_raw": np.asarray(transfer_mass_raw, dtype=np.float64),
        "boundary_in_faces": np.asarray(boundary_in_faces, dtype=np.int64),
        "boundary_out_faces": np.asarray(boundary_out_faces, dtype=np.int64),
        "boundary_in_cells": np.asarray(boundary_in_cells, dtype=np.int64),
        "boundary_out_cells": np.asarray(boundary_out_cells, dtype=np.int64),
        "boundary_in_mass_list": np.asarray(boundary_in_mass_list, dtype=np.float64),
        "boundary_out_mass_list": np.asarray(boundary_out_mass_list, dtype=np.float64),
        "pairwise_conservation_error_max_abs": pairwise_max,
        "pairwise_conservation_error_mean_abs": pairwise_mean,
    }


def apply_bounded_limiter_numpy(
    precompute: ScalarBackendPrecompute,
    scalar: np.ndarray,
    assembled: Dict[str, np.ndarray | float],
    *,
    scheme: ScalarLimiterScheme,
    lower_bound: float,
    upper_bound: float,
    collect_diagnostics: bool = True,
) -> Dict[str, np.ndarray | float]:
    """Apply bounded flux limiter in mass space before the FV update."""

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Limiter bounds must be finite.")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound.")

    n_cells = precompute.cell_volumes.shape[0]
    volumes = np.maximum(precompute.cell_volumes, 1e-16)

    face_flux_raw = np.asarray(assembled["face_scalar_flux_raw"], dtype=np.float64)
    out_raw = np.asarray(assembled["out_mass_raw"], dtype=np.float64)
    in_raw = np.asarray(assembled["in_mass_raw"], dtype=np.float64)
    interior_faces = np.asarray(assembled["interior_faces"], dtype=np.int64)
    donor_cells = np.asarray(assembled["donor_cells"], dtype=np.int64)
    recv_cells = np.asarray(assembled["recv_cells"], dtype=np.int64)
    transfer_mass_raw = np.asarray(assembled["transfer_mass_raw"], dtype=np.float64)
    b_in_faces = np.asarray(assembled["boundary_in_faces"], dtype=np.int64)
    b_out_faces = np.asarray(assembled["boundary_out_faces"], dtype=np.int64)
    b_in_cells = np.asarray(assembled["boundary_in_cells"], dtype=np.int64)
    b_out_cells = np.asarray(assembled["boundary_out_cells"], dtype=np.int64)
    b_in_mass_raw = np.asarray(assembled["boundary_in_mass_list"], dtype=np.float64)
    b_out_mass_raw = np.asarray(assembled["boundary_out_mass_list"], dtype=np.float64)
    face_boundary_group = np.asarray(assembled["face_boundary_group"], dtype=np.int32)

    mass = (scalar - lower_bound) * volumes
    capacity = (upper_bound - scalar) * volumes
    mass = np.maximum(mass, 0.0)
    capacity = np.maximum(capacity, 0.0)

    raw_state = scalar + (in_raw - out_raw) / volumes
    if collect_diagnostics:
        raw_overshoot = np.maximum(raw_state - upper_bound, 0.0)
        raw_undershoot = np.maximum(lower_bound - raw_state, 0.0)
    else:
        raw_overshoot = np.zeros_like(raw_state)
        raw_undershoot = np.zeros_like(raw_state)

    face_flux_limited = np.array(face_flux_raw, copy=True)
    out_limited = np.array(out_raw, copy=True)
    in_limited = np.array(in_raw, copy=True)

    limiter_scale_out = np.ones(n_cells, dtype=np.float64)
    limiter_scale_in = np.ones(n_cells, dtype=np.float64)
    limiter_pair = np.ones(interior_faces.shape[0], dtype=np.float64)
    limiter_b_in = np.ones(b_in_faces.shape[0], dtype=np.float64)
    limiter_b_out = np.ones(b_out_faces.shape[0], dtype=np.float64)
    bounded_enabled = scheme == "bounded_upwind"

    if bounded_enabled:
        with np.errstate(divide="ignore", invalid="ignore"):
            limiter_scale_out = np.where(
                out_raw > 0.0, np.minimum(1.0, mass / out_raw), 1.0
            )
            limiter_scale_in = np.where(
                in_raw > 0.0, np.minimum(1.0, capacity / in_raw), 1.0
            )

        out_limited = np.zeros_like(out_raw)
        in_limited = np.zeros_like(in_raw)
        face_flux_limited[:] = 0.0

        for i in range(interior_faces.shape[0]):
            donor = int(donor_cells[i])
            recv = int(recv_cells[i])
            f_idx = int(interior_faces[i])
            s = float(min(limiter_scale_out[donor], limiter_scale_in[recv]))
            limiter_pair[i] = s
            m_lim = s * float(transfer_mass_raw[i])
            out_limited[donor] += m_lim
            in_limited[recv] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

        for i in range(b_in_faces.shape[0]):
            cell = int(b_in_cells[i])
            f_idx = int(b_in_faces[i])
            s = float(limiter_scale_in[cell])
            limiter_b_in[i] = s
            m_lim = s * float(b_in_mass_raw[i])
            in_limited[cell] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

        for i in range(b_out_faces.shape[0]):
            cell = int(b_out_cells[i])
            f_idx = int(b_out_faces[i])
            s = float(limiter_scale_out[cell])
            limiter_b_out[i] = s
            m_lim = s * float(b_out_mass_raw[i])
            out_limited[cell] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

    limited_state = scalar + (in_limited - out_limited) / volumes
    if collect_diagnostics:
        limited_overshoot = np.maximum(limited_state - upper_bound, 0.0)
        limited_undershoot = np.maximum(lower_bound - limited_state, 0.0)

        limiter_active = (limiter_scale_out < 1.0 - 1e-15) | (
            limiter_scale_in < 1.0 - 1e-15
        )
        limiter_active_count = int(np.count_nonzero(limiter_active))
        limiter_active_fraction = float(limiter_active_count / max(1, n_cells))
        min_scale = float(
            min(
                np.min(limiter_scale_out) if limiter_scale_out.size else 1.0,
                np.min(limiter_scale_in) if limiter_scale_in.size else 1.0,
            )
        )
        limiter_max_flux_scale_down = float(1.0 - min_scale)
        limiter_mass_correction_total = float(
            np.sum(np.abs((in_raw - out_raw) - (in_limited - out_limited)))
        )

        total_raw = float(np.sum(in_raw) - np.sum(out_raw))
        total_limited = float(np.sum(in_limited) - np.sum(out_limited))
        conservation_error = float(total_limited - total_raw)
    else:
        limited_overshoot = np.zeros_like(limited_state)
        limited_undershoot = np.zeros_like(limited_state)
        limiter_active_count = 0
        limiter_active_fraction = 0.0
        limiter_max_flux_scale_down = 0.0
        limiter_mass_correction_total = 0.0
        conservation_error = 0.0

    inlet_face_mask = (face_boundary_group == BOUNDARY_GROUP_CODE["left_inlet"]) | (
        face_boundary_group == BOUNDARY_GROUP_CODE["right_inlet"]
    )
    outlet_face_mask = face_boundary_group == BOUNDARY_GROUP_CODE["outlet"]
    raw_inlet_scalar_flux_in = float(
        -np.sum(np.minimum(face_flux_raw[inlet_face_mask], 0.0))
    )
    raw_outlet_scalar_flux_out = float(
        np.sum(np.maximum(face_flux_raw[outlet_face_mask], 0.0))
    )
    limited_inlet_scalar_flux_in = float(
        -np.sum(np.minimum(face_flux_limited[inlet_face_mask], 0.0))
    )
    limited_outlet_scalar_flux_out = float(
        np.sum(np.maximum(face_flux_limited[outlet_face_mask], 0.0))
    )

    if not collect_diagnostics:
        return {
            "bounded_limiter_enabled": bounded_enabled,
            "limited_state_before_clip": limited_state,
            "limiter_scale_out": limiter_scale_out,
            "limiter_scale_in": limiter_scale_in,
        }

    return {
        "bounded_limiter_enabled": bounded_enabled,
        "face_scalar_flux_raw": face_flux_raw,
        "face_scalar_flux_limited": face_flux_limited,
        "out_mass_raw": out_raw,
        "in_mass_raw": in_raw,
        "out_mass_limited": out_limited,
        "in_mass_limited": in_limited,
        "raw_state_before_limiter": raw_state,
        "limited_state_before_clip": limited_state,
        "raw_overshoot": raw_overshoot,
        "raw_undershoot": raw_undershoot,
        "limited_overshoot": limited_overshoot,
        "limited_undershoot": limited_undershoot,
        "limiter_scale_out": limiter_scale_out,
        "limiter_scale_in": limiter_scale_in,
        "limiter_pair": limiter_pair,
        "limiter_active_count": limiter_active_count,
        "limiter_active_fraction": limiter_active_fraction,
        "limiter_max_flux_scale_down": limiter_max_flux_scale_down,
        "limiter_mass_correction_total": limiter_mass_correction_total,
        "conservation_error_after_limiter": conservation_error,
        "raw_inlet_scalar_flux_in": raw_inlet_scalar_flux_in,
        "raw_outlet_scalar_flux_out": raw_outlet_scalar_flux_out,
        "effective_inlet_scalar_flux_in": limited_inlet_scalar_flux_in,
        "effective_outlet_scalar_flux_out": limited_outlet_scalar_flux_out,
    }


def assemble_advection_fluxes_torch(
    precompute: ScalarBackendPrecompute,
    scalar: Any,
    *,
    left_inlet_value: float,
    right_inlet_value: float,
    dt: float,
) -> dict[str, object]:
    tb = _require_torch_backend(precompute)
    torch = tb["torch"]
    dtype = tb["dtype"]
    c0 = tb["c0"]
    q_all = tb["q_all"]
    donor_i = tb["donor_i"]
    recv_i = tb["recv_i"]
    q_i = tb["q_i"]
    left_faces_t = tb["left_faces_t"]
    right_faces_t = tb["right_faces_t"]
    outlet_faces_t = tb["outlet_faces_t"]
    unresolved_faces_t = tb["unresolved_faces_t"]
    interior_faces = tb["interior_faces"]
    n_cells = int(precompute.cell_volumes.shape[0])
    n_faces = int(precompute.face_to_cells.shape[0])

    out_mass = torch.zeros((n_cells,), dtype=dtype, device=tb["device"])
    in_mass = torch.zeros((n_cells,), dtype=dtype, device=tb["device"])
    out_vol_rate = torch.zeros((n_cells,), dtype=dtype, device=tb["device"])
    in_vol_rate = torch.zeros((n_cells,), dtype=dtype, device=tb["device"])
    face_flux_raw = torch.zeros((n_faces,), dtype=dtype, device=tb["device"])

    c_up_i = scalar[donor_i]
    phi_i = q_i * c_up_i
    m_i = dt * torch.abs(phi_i)
    out_mass.scatter_add_(0, donor_i, m_i)
    in_mass.scatter_add_(0, recv_i, m_i)
    out_vol_rate.scatter_add_(0, donor_i, torch.abs(q_i))
    in_vol_rate.scatter_add_(0, recv_i, torch.abs(q_i))
    face_flux_raw[interior_faces] = phi_i

    zero_t = torch.tensor(0.0, dtype=dtype, device=tb["device"])
    inlet_scalar_flux_in = zero_t.clone()
    outlet_scalar_flux_out = zero_t.clone()

    def _apply_boundary_group(
        face_ids_t: Any,
        prescribed_value: float | None,
        *,
        track_outlet_flux: bool = False,
    ) -> tuple[Any, Any]:
        nonlocal inlet_scalar_flux_in, outlet_scalar_flux_out
        if int(face_ids_t.numel()) == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=tb["device"]),
                torch.zeros((0,), dtype=dtype, device=tb["device"]),
            )

        owner = c0[face_ids_t]
        q = q_all[face_ids_t]
        inflow = q < 0.0
        outflow = q > 0.0

        face_mass = torch.zeros_like(q)
        face_flux = torch.zeros_like(q)

        if prescribed_value is not None:
            phi_in = q * float(prescribed_value)
            m_in = torch.where(inflow, -dt * phi_in, torch.zeros_like(q))
            in_mass.scatter_add_(0, owner, m_in)
            in_vol_rate.scatter_add_(
                0, owner, torch.where(inflow, torch.abs(q), zero_t)
            )
            inlet_scalar_flux_in = inlet_scalar_flux_in + torch.sum(
                torch.where(inflow, -phi_in, zero_t)
            )

            phi_out = q * scalar[owner]
            m_out = torch.where(outflow, dt * phi_out, torch.zeros_like(q))
            out_mass.scatter_add_(0, owner, m_out)
            out_vol_rate.scatter_add_(
                0, owner, torch.where(outflow, torch.abs(q), zero_t)
            )
            face_flux = torch.where(
                inflow, phi_in, torch.where(outflow, phi_out, face_flux)
            )
            face_mass = torch.where(
                inflow, m_in, torch.where(outflow, m_out, face_mass)
            )
            face_flux_raw[face_ids_t] = face_flux
            return owner, face_mass

        phi = q * scalar[owner]
        m = torch.where(outflow, dt * phi, torch.zeros_like(q))
        out_mass.scatter_add_(0, owner, m)
        out_vol_rate.scatter_add_(0, owner, torch.where(outflow, torch.abs(q), zero_t))
        face_flux = torch.where(outflow, phi, face_flux)
        face_mass = torch.where(outflow, m, face_mass)
        if track_outlet_flux:
            outlet_scalar_flux_out = outlet_scalar_flux_out + torch.sum(
                torch.where(outflow, phi, zero_t)
            )
        face_flux_raw[face_ids_t] = face_flux
        return owner, face_mass

    left_owner, left_mass = _apply_boundary_group(left_faces_t, left_inlet_value)
    right_owner, right_mass = _apply_boundary_group(right_faces_t, right_inlet_value)
    outlet_owner, outlet_mass = _apply_boundary_group(
        outlet_faces_t,
        None,
        track_outlet_flux=True,
    )
    unresolved_owner, unresolved_mass = _apply_boundary_group(unresolved_faces_t, None)

    return {
        "face_scalar_flux_raw": face_flux_raw,
        "out_mass_raw": out_mass,
        "in_mass_raw": in_mass,
        "out_vol_rate_raw": out_vol_rate,
        "in_vol_rate_raw": in_vol_rate,
        "inlet_scalar_flux_in": inlet_scalar_flux_in,
        "outlet_scalar_flux_out": outlet_scalar_flux_out,
        "interior_faces": interior_faces,
        "transfer_mass_raw": m_i,
        "left_inlet_value": float(left_inlet_value),
        "right_inlet_value": float(right_inlet_value),
        "left_owner": left_owner,
        "left_mass_raw": left_mass,
        "right_owner": right_owner,
        "right_mass_raw": right_mass,
        "outlet_owner": outlet_owner,
        "outlet_mass_raw": outlet_mass,
        "unresolved_owner": unresolved_owner,
        "unresolved_mass_raw": unresolved_mass,
    }


def apply_bounded_limiter_torch(
    precompute: ScalarBackendPrecompute,
    scalar: Any,
    assembled: dict[str, object],
    *,
    scheme: ScalarLimiterScheme,
    lower_bound: float,
    upper_bound: float,
    collect_diagnostics: bool = True,
) -> dict[str, object]:
    tb = _require_torch_backend(precompute)
    torch = tb["torch"]
    donor_i = tb["donor_i"]
    recv_i = tb["recv_i"]
    left_faces_t = tb["left_faces_t"]
    right_faces_t = tb["right_faces_t"]
    outlet_faces_t = tb["outlet_faces_t"]
    unresolved_faces_t = tb["unresolved_faces_t"]
    c0 = tb["c0"]
    q_all = tb["q_all"]
    vol = tb["vol"]

    out_mass = assembled["out_mass_raw"]
    in_mass = assembled["in_mass_raw"]
    face_flux_raw = assembled["face_scalar_flux_raw"]
    m_i = assembled["transfer_mass_raw"]
    interior_faces = assembled["interior_faces"]
    left_mass_raw = assembled["left_mass_raw"]
    right_mass_raw = assembled["right_mass_raw"]
    outlet_mass_raw = assembled["outlet_mass_raw"]
    unresolved_mass_raw = assembled["unresolved_mass_raw"]

    mass_old = (scalar - float(lower_bound)) * vol
    capacity = (float(upper_bound) - scalar) * vol
    mass_old = torch.clamp(mass_old, min=0.0)
    capacity = torch.clamp(capacity, min=0.0)

    raw_state = scalar + (in_mass - out_mass) / vol
    if collect_diagnostics:
        raw_overshoot = torch.clamp(raw_state - float(upper_bound), min=0.0)
        raw_undershoot = torch.clamp(float(lower_bound) - raw_state, min=0.0)
    else:
        raw_overshoot = torch.zeros_like(raw_state)
        raw_undershoot = torch.zeros_like(raw_state)

    if scheme == "bounded_upwind":
        limiter_out = torch.ones_like(scalar)
        limiter_in = torch.ones_like(scalar)
        limiter_out = torch.where(
            out_mass > 0.0,
            torch.minimum(
                torch.ones_like(scalar), mass_old / torch.clamp(out_mass, min=1e-30)
            ),
            limiter_out,
        )
        limiter_in = torch.where(
            in_mass > 0.0,
            torch.minimum(
                torch.ones_like(scalar), capacity / torch.clamp(in_mass, min=1e-30)
            ),
            limiter_in,
        )
    else:
        limiter_out = torch.ones_like(scalar)
        limiter_in = torch.ones_like(scalar)

    pair_scale = torch.minimum(limiter_out[donor_i], limiter_in[recv_i])
    m_i_lim = m_i * pair_scale
    out_lim = torch.zeros_like(scalar)
    in_lim = torch.zeros_like(scalar)
    out_lim.scatter_add_(0, donor_i, m_i_lim)
    in_lim.scatter_add_(0, recv_i, m_i_lim)

    face_flux_limited = torch.zeros_like(face_flux_raw)
    face_flux_limited[interior_faces] = pair_scale * face_flux_raw[interior_faces]

    inlet_scalar_flux_in_effective = torch.zeros(
        (), dtype=scalar.dtype, device=tb["device"]
    )
    outlet_scalar_flux_out_effective = torch.zeros(
        (), dtype=scalar.dtype, device=tb["device"]
    )

    def _apply_boundary_limited_actual(
        face_ids_t: Any,
        boundary_mass: Any,
        prescribed_value: float | None,
        *,
        track_outlet_flux: bool = False,
    ) -> None:
        nonlocal inlet_scalar_flux_in_effective, outlet_scalar_flux_out_effective
        if int(face_ids_t.numel()) == 0:
            return
        owner = c0[face_ids_t]
        q = q_all[face_ids_t]
        inflow = q < 0.0
        outflow = q > 0.0
        zero_boundary = torch.zeros_like(q)
        if prescribed_value is not None:
            phi_in = q * float(prescribed_value)
            scale_in = limiter_in[owner]
            in_limited = torch.where(
                inflow,
                boundary_mass * scale_in,
                zero_boundary,
            )
            in_lim.scatter_add_(0, owner, in_limited)
            inlet_scalar_flux_in_effective = inlet_scalar_flux_in_effective + torch.sum(
                torch.where(inflow, (-phi_in) * scale_in, zero_boundary)
            )

            phi_out = q * scalar[owner]
            scale_out = limiter_out[owner]
            out_limited = torch.where(
                outflow,
                boundary_mass * scale_out,
                zero_boundary,
            )
            out_lim.scatter_add_(0, owner, out_limited)
            face_flux_limited[face_ids_t] = torch.where(
                inflow,
                phi_in * scale_in,
                torch.where(outflow, phi_out * scale_out, zero_boundary),
            )
            return

        phi = q * scalar[owner]
        scale = limiter_out[owner]
        out_limited = torch.where(
            outflow,
            boundary_mass * scale,
            zero_boundary,
        )
        out_lim.scatter_add_(0, owner, out_limited)
        face_flux_limited[face_ids_t] = torch.where(outflow, phi * scale, zero_boundary)
        if track_outlet_flux:
            outlet_scalar_flux_out_effective = (
                outlet_scalar_flux_out_effective
                + torch.sum(torch.where(outflow, phi * scale, zero_boundary))
            )

    _apply_boundary_limited_actual(
        left_faces_t,
        left_mass_raw,
        assembled.get("left_inlet_value"),
    )
    _apply_boundary_limited_actual(
        right_faces_t,
        right_mass_raw,
        assembled.get("right_inlet_value"),
    )
    _apply_boundary_limited_actual(
        outlet_faces_t,
        outlet_mass_raw,
        None,
        track_outlet_flux=True,
    )
    _apply_boundary_limited_actual(unresolved_faces_t, unresolved_mass_raw, None)

    limited_state = scalar + (in_lim - out_lim) / vol
    if collect_diagnostics:
        limited_overshoot = torch.clamp(limited_state - float(upper_bound), min=0.0)
        limited_undershoot = torch.clamp(float(lower_bound) - limited_state, min=0.0)

        limiter_active_mask = (torch.abs(limiter_out - 1.0) > 1e-12) | (
            torch.abs(limiter_in - 1.0) > 1e-12
        )
        limiter_active_count = int(torch.count_nonzero(limiter_active_mask).item())
        limiter_active_fraction = float(
            limiter_active_count / max(1, int(precompute.cell_volumes.shape[0]))
        )
        limiter_max_flux_scale_down = float(
            torch.max(1.0 - torch.minimum(limiter_out, limiter_in)).item()
        )
        limiter_mass_correction_total = float(
            torch.sum(torch.abs((in_mass - out_mass) - (in_lim - out_lim))).item()
        )
        conservation_error = float(
            (
                torch.sum(in_lim)
                - torch.sum(out_lim)
                - (torch.sum(in_mass) - torch.sum(out_mass))
            ).item()
        )
    else:
        limited_overshoot = torch.zeros_like(limited_state)
        limited_undershoot = torch.zeros_like(limited_state)
        limiter_active_count = 0
        limiter_active_fraction = 0.0
        limiter_max_flux_scale_down = 0.0
        limiter_mass_correction_total = 0.0
        conservation_error = 0.0

    if not collect_diagnostics:
        return {
            "bounded_limiter_enabled": bool(scheme == "bounded_upwind"),
            "limited_state_before_clip": limited_state,
            "limiter_scale_out": limiter_out,
            "limiter_scale_in": limiter_in,
        }

    return {
        "bounded_limiter_enabled": bool(scheme == "bounded_upwind"),
        "face_scalar_flux_raw": face_flux_raw,
        "face_scalar_flux_limited": face_flux_limited,
        "out_mass_raw": out_mass,
        "in_mass_raw": in_mass,
        "out_mass_limited": out_lim,
        "in_mass_limited": in_lim,
        "raw_state_before_limiter": raw_state,
        "limited_state_before_clip": limited_state,
        "raw_overshoot": raw_overshoot,
        "raw_undershoot": raw_undershoot,
        "limited_overshoot": limited_overshoot,
        "limited_undershoot": limited_undershoot,
        "limiter_scale_out": limiter_out,
        "limiter_scale_in": limiter_in,
        "limiter_active_count": limiter_active_count,
        "limiter_active_fraction": limiter_active_fraction,
        "limiter_max_flux_scale_down": limiter_max_flux_scale_down,
        "limiter_mass_correction_total": limiter_mass_correction_total,
        "conservation_error_after_limiter": conservation_error,
        "effective_inlet_scalar_flux_in": inlet_scalar_flux_in_effective,
        "effective_outlet_scalar_flux_out": outlet_scalar_flux_out_effective,
    }


def laplacian_numpy(
    precompute: ScalarBackendPrecompute,
    scalar: np.ndarray,
    *,
    gradient_method: GradientMethod,
    laplacian_method: LaplacianMethod,
) -> np.ndarray:
    diffusion_data = precompute.diffusion_data
    if diffusion_data is None:
        raise ValueError("NumPy diffusion requested without diffusion precompute.")

    vol = np.maximum(precompute.cell_volumes, 1e-16)
    n_cells = int(precompute.cell_volumes.shape[0])
    scalar_arr = np.asarray(scalar, dtype=np.float64)

    def _add_prescribed_boundary_flux(lap: np.ndarray) -> None:
        neumann_c0 = np.asarray(diffusion_data["st_neumann_c0"], dtype=np.int64)
        if neumann_c0.size:
            area = np.asarray(diffusion_data["st_neumann_area"], dtype=np.float64)
            gradient = np.asarray(
                diffusion_data["st_neumann_gradient"], dtype=np.float64
            )
            np.add.at(lap, neumann_c0, area * gradient / vol[neumann_c0])
        robin_c0 = np.asarray(diffusion_data["st_robin_c0"], dtype=np.int64)
        if robin_c0.size:
            area = np.asarray(diffusion_data["st_robin_area"], dtype=np.float64)
            distance = np.asarray(diffusion_data["st_robin_distance"], dtype=np.float64)
            alpha = np.asarray(diffusion_data["st_robin_alpha"], dtype=np.float64)
            beta = np.asarray(diffusion_data["st_robin_beta"], dtype=np.float64)
            gamma = np.asarray(diffusion_data["st_robin_gamma"], dtype=np.float64)
            gradient = (gamma - alpha * scalar_arr[robin_c0]) / (
                beta + alpha * distance
            )
            np.add.at(lap, robin_c0, area * gradient / vol[robin_c0])

    def _gradient_green_gauss_numpy(values: np.ndarray) -> np.ndarray:
        weighted_normals = np.asarray(
            diffusion_data["gg_weighted_normals"], dtype=np.float64
        )
        fc0 = np.asarray(diffusion_data["gg_c0"], dtype=np.int64)
        fc1 = np.asarray(diffusion_data["gg_c1"], dtype=np.int64)
        face_values = np.zeros((precompute.face_to_cells.shape[0],), dtype=np.float64)
        interior = fc1 >= 0
        if np.any(interior):
            face_values[interior] = 0.5 * (
                values[fc0[interior]] + values[fc1[interior]]
            )
        if np.any(~interior):
            face_values[~interior] = values[fc0[~interior]]
        no_flux = np.asarray(diffusion_data["gg_no_flux_mask"], dtype=bool)
        if np.any(no_flux):
            face_values[no_flux] = values[fc0[no_flux]]
        has_dir = np.asarray(diffusion_data["gg_boundary_has_dirichlet"], dtype=bool)
        if np.any(has_dir):
            face_values[has_dir] = np.asarray(
                diffusion_data["gg_boundary_dirichlet_values"], dtype=np.float64
            )[has_dir]
        gradients = np.zeros((n_cells, 3), dtype=np.float64)
        flux_vec = face_values[:, None] * weighted_normals
        np.add.at(gradients, fc0, flux_vec)
        if np.any(interior):
            np.add.at(gradients, fc1[interior], -flux_vec[interior])
        gradients /= vol[:, None]
        return gradients

    def _gradient_lsq_numpy(values: np.ndarray) -> np.ndarray:
        if gradient_method == "face":
            return _gradient_green_gauss_numpy(values)
        rhs = np.zeros((n_cells, 3), dtype=np.float64)
        lsq_c0 = np.asarray(diffusion_data["lsq_c0"], dtype=np.int64)
        if lsq_c0.size:
            lsq_c1 = np.asarray(diffusion_data["lsq_c1"], dtype=np.int64)
            lsq_r = np.asarray(diffusion_data["lsq_r"], dtype=np.float64)
            lsq_w = np.asarray(diffusion_data["lsq_w"], dtype=np.float64)
            dphi = values[lsq_c1] - values[lsq_c0]
            vec = (lsq_w[:, None] * lsq_r) * dphi[:, None]
            np.add.at(rhs, lsq_c0, vec)
            np.add.at(rhs, lsq_c1, vec)
        db_owner = np.asarray(diffusion_data["lsq_db_owner"], dtype=np.int64)
        if db_owner.size:
            db_r = np.asarray(diffusion_data["lsq_db_r"], dtype=np.float64)
            db_w = np.asarray(diffusion_data["lsq_db_w"], dtype=np.float64)
            db_value = np.asarray(diffusion_data["lsq_db_value"], dtype=np.float64)
            vec_b = (db_w[:, None] * db_r) * (db_value - values[db_owner])[:, None]
            np.add.at(rhs, db_owner, vec_b)
        gradients = _gradient_green_gauss_numpy(values)
        solve_mask = np.asarray(diffusion_data["lsq_solve_mask"], dtype=bool)
        if np.any(solve_mask):
            inv = np.asarray(diffusion_data["lsq_inv"], dtype=np.float64)
            gradients[solve_mask] = np.matmul(
                inv[solve_mask], rhs[solve_mask][..., None]
            ).squeeze(-1)
        return gradients

    if laplacian_method == "tpfa":
        lap = np.zeros((n_cells,), dtype=np.float64)
        st_c0 = np.asarray(diffusion_data["st_interior_c0"], dtype=np.int64)
        if st_c0.size:
            st_c1 = np.asarray(diffusion_data["st_interior_c1"], dtype=np.int64)
            coeff = np.abs(
                np.asarray(diffusion_data["st_interior_alpha"], dtype=np.float64)
            )
            flux = coeff * (scalar_arr[st_c1] - scalar_arr[st_c0])
            np.add.at(lap, st_c0, flux / vol[st_c0])
            np.add.at(lap, st_c1, -flux / vol[st_c1])
        db_c0 = np.asarray(diffusion_data["st_dirichlet_c0"], dtype=np.int64)
        if db_c0.size:
            coeff_b = np.asarray(diffusion_data["st_dirichlet_coeff"], dtype=np.float64)
            value_b = np.asarray(diffusion_data["st_dirichlet_value"], dtype=np.float64)
            flux_b = coeff_b * (value_b - scalar_arr[db_c0])
            np.add.at(lap, db_c0, flux_b / vol[db_c0])
        _add_prescribed_boundary_flux(lap)
        return lap

    gradients = _gradient_lsq_numpy(scalar_arr)
    lap = np.zeros((n_cells,), dtype=np.float64)
    st_c0 = np.asarray(diffusion_data["st_interior_c0"], dtype=np.int64)
    if st_c0.size:
        st_c1 = np.asarray(diffusion_data["st_interior_c1"], dtype=np.int64)
        alpha = np.asarray(diffusion_data["st_interior_alpha"], dtype=np.float64)
        corr = np.asarray(diffusion_data["st_interior_correction"], dtype=np.float64)
        grad_face = 0.5 * (gradients[st_c0] + gradients[st_c1])
        orth_flux = alpha * (scalar_arr[st_c1] - scalar_arr[st_c0])
        corr_flux = np.einsum("ij,ij->i", grad_face, corr)
        flux = orth_flux + corr_flux
        np.add.at(lap, st_c0, flux / vol[st_c0])
        np.add.at(lap, st_c1, -flux / vol[st_c1])
    db_c0 = np.asarray(diffusion_data["st_dirichlet_c0"], dtype=np.int64)
    if db_c0.size:
        coeff_b = np.asarray(diffusion_data["st_dirichlet_coeff"], dtype=np.float64)
        value_b = np.asarray(diffusion_data["st_dirichlet_value"], dtype=np.float64)
        flux_b = coeff_b * (value_b - scalar_arr[db_c0])
        np.add.at(lap, db_c0, flux_b / vol[db_c0])
    _add_prescribed_boundary_flux(lap)
    return lap


def laplacian_torch(
    precompute: ScalarBackendPrecompute,
    scalar: Any,
    *,
    gradient_method: GradientMethod,
    laplacian_method: LaplacianMethod,
) -> Any:
    tb = _require_torch_backend(precompute)
    diffusion_data = tb.get("diffusion_data")
    if diffusion_data is None:
        raise ValueError("Torch diffusion requested without diffusion precompute.")

    torch = tb["torch"]
    vol = tb["vol"]
    n_cells = int(precompute.cell_volumes.shape[0])

    def _add_prescribed_boundary_flux(lap: Any) -> None:
        neumann_c0 = diffusion_data["st_neumann_c0"]
        if int(neumann_c0.numel()) > 0:
            flux = (
                diffusion_data["st_neumann_area"]
                * diffusion_data["st_neumann_gradient"]
            )
            lap.index_add_(0, neumann_c0, flux / vol[neumann_c0])
        robin_c0 = diffusion_data["st_robin_c0"]
        if int(robin_c0.numel()) > 0:
            alpha = diffusion_data["st_robin_alpha"]
            denominator = (
                diffusion_data["st_robin_beta"]
                + alpha * diffusion_data["st_robin_distance"]
            )
            gradient = (
                diffusion_data["st_robin_gamma"] - alpha * scalar[robin_c0]
            ) / denominator
            flux = diffusion_data["st_robin_area"] * gradient
            lap.index_add_(0, robin_c0, flux / vol[robin_c0])

    def _gradient_green_gauss_torch(scalar_t: Any) -> Any:
        weighted_normals = diffusion_data["gg_weighted_normals"]
        fc0 = diffusion_data["gg_c0"]
        fc1 = diffusion_data["gg_c1"]
        face_values = torch.zeros(
            (precompute.face_to_cells.shape[0],), dtype=tb["dtype"], device=tb["device"]
        )
        interior = fc1 >= 0
        face_values[interior] = 0.5 * (
            scalar_t[fc0[interior]] + scalar_t[fc1[interior]]
        )
        face_values[~interior] = scalar_t[fc0[~interior]]
        no_flux = diffusion_data["gg_no_flux_mask"]
        face_values[no_flux] = scalar_t[fc0[no_flux]]
        has_dir = diffusion_data["gg_boundary_has_dirichlet"]
        if bool(torch.any(has_dir)):
            face_values = torch.where(
                has_dir,
                diffusion_data["gg_boundary_dirichlet_values"],
                face_values,
            )
        gradients = torch.zeros((n_cells, 3), dtype=tb["dtype"], device=tb["device"])
        flux_vec = face_values[:, None] * weighted_normals
        gradients.index_add_(0, fc0, flux_vec)
        interior_faces_gg = torch.nonzero(interior, as_tuple=False).reshape(-1)
        if int(interior_faces_gg.numel()) > 0:
            gradients.index_add_(
                0, fc1[interior_faces_gg], -flux_vec[interior_faces_gg]
            )
        gradients = gradients / vol[:, None]
        return gradients

    def _gradient_lsq_torch(scalar_t: Any) -> Any:
        if gradient_method == "face":
            return _gradient_green_gauss_torch(scalar_t)
        rhs = torch.zeros((n_cells, 3), dtype=tb["dtype"], device=tb["device"])
        lsq_c0 = diffusion_data["lsq_c0"]
        lsq_c1 = diffusion_data["lsq_c1"]
        lsq_r = diffusion_data["lsq_r"]
        lsq_w = diffusion_data["lsq_w"]
        if int(lsq_c0.numel()) > 0:
            dphi = scalar_t[lsq_c1] - scalar_t[lsq_c0]
            vec = (lsq_w[:, None] * lsq_r) * dphi[:, None]
            rhs.index_add_(0, lsq_c0, vec)
            rhs.index_add_(0, lsq_c1, vec)
        db_owner = diffusion_data["lsq_db_owner"]
        if int(db_owner.numel()) > 0:
            db_r = diffusion_data["lsq_db_r"]
            db_w = diffusion_data["lsq_db_w"]
            db_value = diffusion_data["lsq_db_value"]
            vec_b = (db_w[:, None] * db_r) * (db_value - scalar_t[db_owner])[:, None]
            rhs.index_add_(0, db_owner, vec_b)
        gradients = _gradient_green_gauss_torch(scalar_t)
        solve_mask = diffusion_data["lsq_solve_mask"]
        if bool(torch.any(solve_mask)):
            inv = diffusion_data["lsq_inv"]
            solved = torch.matmul(
                inv[solve_mask], rhs[solve_mask].unsqueeze(-1)
            ).squeeze(-1)
            gradients[solve_mask] = solved
        return gradients

    if laplacian_method == "tpfa":
        lap = torch.zeros((n_cells,), dtype=tb["dtype"], device=tb["device"])
        st_c0 = diffusion_data["st_interior_c0"]
        st_c1 = diffusion_data["st_interior_c1"]
        if int(st_c0.numel()) > 0:
            coeff = torch.abs(diffusion_data["st_interior_alpha"])
            flux = coeff * (scalar[st_c1] - scalar[st_c0])
            lap.index_add_(0, st_c0, flux / vol[st_c0])
            lap.index_add_(0, st_c1, -flux / vol[st_c1])
        db_c0 = diffusion_data["st_dirichlet_c0"]
        if int(db_c0.numel()) > 0:
            coeff_b = diffusion_data["st_dirichlet_coeff"]
            value_b = diffusion_data["st_dirichlet_value"]
            flux_b = coeff_b * (value_b - scalar[db_c0])
            lap.index_add_(0, db_c0, flux_b / vol[db_c0])
        _add_prescribed_boundary_flux(lap)
        return lap

    gradients = _gradient_lsq_torch(scalar)
    lap = torch.zeros((n_cells,), dtype=tb["dtype"], device=tb["device"])
    st_c0 = diffusion_data["st_interior_c0"]
    st_c1 = diffusion_data["st_interior_c1"]
    if int(st_c0.numel()) > 0:
        alpha = diffusion_data["st_interior_alpha"]
        corr = diffusion_data["st_interior_correction"]
        grad_face = 0.5 * (gradients[st_c0] + gradients[st_c1])
        orth_flux = alpha * (scalar[st_c1] - scalar[st_c0])
        corr_flux = torch.sum(grad_face * corr, dim=1)
        flux = orth_flux + corr_flux
        lap.index_add_(0, st_c0, flux / vol[st_c0])
        lap.index_add_(0, st_c1, -flux / vol[st_c1])
    db_c0 = diffusion_data["st_dirichlet_c0"]
    if int(db_c0.numel()) > 0:
        coeff_b = diffusion_data["st_dirichlet_coeff"]
        value_b = diffusion_data["st_dirichlet_value"]
        flux_b = coeff_b * (value_b - scalar[db_c0])
        lap.index_add_(0, db_c0, flux_b / vol[db_c0])
    _add_prescribed_boundary_flux(lap)
    return lap


def build_backend_execution_diagnostics(
    *,
    requested_backend: str,
    stepping_backend: str,
    device: str,
    torch_device: str,
    used_numpy_fallback: bool,
    all_core_arrays_on_cuda: bool,
    per_step_large_cpu_gpu_transfer_warning: bool = False,
    cpu_gpu_transfer_notes: list[str] | None = None,
) -> dict[str, object]:
    return {
        "requested_backend": str(requested_backend),
        "stepping_backend": str(stepping_backend),
        "device": str(device),
        "torch_device": str(torch_device),
        "used_numpy_fallback": bool(used_numpy_fallback),
        "all_core_arrays_on_cuda": bool(all_core_arrays_on_cuda),
        "per_step_large_cpu_gpu_transfer_warning": bool(
            per_step_large_cpu_gpu_transfer_warning
        ),
        "cpu_gpu_transfer_notes": list(cpu_gpu_transfer_notes or []),
    }


def _build_torch_backend(
    mesh: ImportedTetraMesh,
    *,
    face_normal_velocity: np.ndarray,
    boundary_face_groups: np.ndarray,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    boundary_faces: np.ndarray,
    unresolved_faces: np.ndarray,
    left_cells: np.ndarray,
    right_cells: np.ndarray,
    outlet_cells: np.ndarray,
    left_zone_mask: np.ndarray,
    right_zone_mask: np.ndarray,
    corner_zone_mask: np.ndarray,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
    diffusion_stencil: DiffusionStencil | None,
    diffusion_data: dict[str, object] | None,
    torch_device: str,
) -> dict[str, object]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Torch backend requested but torch is not installed."
        ) from exc

    device = torch.device(torch_device)
    dtype = torch.float64

    c0 = torch.as_tensor(mesh.face_to_cells[:, 0], dtype=torch.long, device=device)
    c1 = torch.as_tensor(mesh.face_to_cells[:, 1], dtype=torch.long, device=device)
    area = torch.as_tensor(mesh.face_areas, dtype=dtype, device=device)
    vn = torch.as_tensor(face_normal_velocity, dtype=dtype, device=device)
    q_all = vn * area
    vol = torch.as_tensor(
        np.maximum(mesh.cell_volumes, 1e-16), dtype=dtype, device=device
    )
    groups = torch.as_tensor(boundary_face_groups, dtype=torch.long, device=device)
    face_ids = torch.arange(
        mesh.face_vertices.shape[0], device=device, dtype=torch.long
    )
    interior_mask = c1 >= 0
    interior_faces = face_ids[interior_mask]
    owner_i = c0[interior_faces]
    neigh_i = c1[interior_faces]
    q_i = q_all[interior_faces]
    q_i_pos = q_i >= 0.0
    donor_i = torch.where(q_i_pos, owner_i, neigh_i)
    recv_i = torch.where(q_i_pos, neigh_i, owner_i)

    diffusion_data_torch = _build_torch_diffusion_data(
        diffusion_data,
        torch=torch,
        device=device,
        dtype=dtype,
    )

    return {
        "torch": torch,
        "device": device,
        "dtype": dtype,
        "c0": c0,
        "c1": c1,
        "area": area,
        "vn": vn,
        "q_all": q_all,
        "groups": groups,
        "vol": vol,
        "interior_faces": interior_faces,
        "owner_i": owner_i,
        "neigh_i": neigh_i,
        "q_i": q_i,
        "donor_i": donor_i,
        "recv_i": recv_i,
        "left_faces_t": torch.as_tensor(left_faces, dtype=torch.long, device=device),
        "right_faces_t": torch.as_tensor(right_faces, dtype=torch.long, device=device),
        "outlet_faces_t": torch.as_tensor(
            outlet_faces, dtype=torch.long, device=device
        ),
        "boundary_faces_t": torch.as_tensor(
            boundary_faces, dtype=torch.long, device=device
        ),
        "unresolved_faces_t": torch.as_tensor(
            unresolved_faces, dtype=torch.long, device=device
        ),
        "left_cells_t": torch.as_tensor(left_cells, dtype=torch.long, device=device),
        "right_cells_t": torch.as_tensor(right_cells, dtype=torch.long, device=device),
        "outlet_cells_t": torch.as_tensor(
            outlet_cells, dtype=torch.long, device=device
        ),
        "left_zone_mask_t": torch.as_tensor(
            left_zone_mask, dtype=torch.bool, device=device
        ),
        "right_zone_mask_t": torch.as_tensor(
            right_zone_mask, dtype=torch.bool, device=device
        ),
        "corner_zone_mask_t": torch.as_tensor(
            corner_zone_mask, dtype=torch.bool, device=device
        ),
        "diffusion_data": diffusion_data_torch,
    }


def _require_torch_backend(precompute: ScalarBackendPrecompute) -> dict[str, object]:
    if precompute.torch_backend is None:
        raise RuntimeError("Torch backend precompute is not available.")
    return precompute.torch_backend
