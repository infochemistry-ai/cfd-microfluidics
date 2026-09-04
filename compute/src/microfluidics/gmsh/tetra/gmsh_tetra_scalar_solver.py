"""Diffusion-only scalar debug solver on imported tetrahedral Gmsh meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Sequence, Tuple

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    GradientMethod,
    LaplacianMethod,
    scalar_gradient,
)

SolverBackend = Literal["numpy", "torch"]


@dataclass(frozen=True)
class GmshTetraScalarConfig:
    dt: float = 5e-4
    steps: int = 120
    diffusivity: float = 3e-10
    left_inlet_value: float = 0.0
    right_inlet_value: float = 1.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    progress_every: int = 20
    gradient_method: GradientMethod = "least_squares"
    laplacian_method: LaplacianMethod = "lsq_flux"
    backend: SolverBackend = "numpy"
    torch_device: str = "cpu"


@dataclass(frozen=True)
class DiffusionStencil:
    interior_c0: np.ndarray
    interior_c1: np.ndarray
    interior_coeff: np.ndarray
    interior_alpha: np.ndarray
    interior_area: np.ndarray
    interior_normals: np.ndarray
    interior_correction_vec: np.ndarray
    dirichlet_c0: np.ndarray
    dirichlet_coeff: np.ndarray
    dirichlet_area: np.ndarray
    dirichlet_normal: np.ndarray
    dirichlet_cell_to_face_delta: np.ndarray
    dirichlet_value: np.ndarray
    neumann_c0: np.ndarray
    neumann_area: np.ndarray
    neumann_normal_gradient: np.ndarray
    robin_c0: np.ndarray
    robin_area: np.ndarray
    robin_distance: np.ndarray
    robin_alpha: np.ndarray
    robin_beta: np.ndarray
    robin_gamma: np.ndarray
    volumes: np.ndarray


def _normalize_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    for token in (" ", "-", ".", "/"):
        cleaned = cleaned.replace(token, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def resolve_inlet_face_groups(mesh: ImportedTetraMesh) -> Dict[str, object]:
    """Resolve left/right inlet faces from names; fallback to x-coordinate split."""

    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    if inlet_faces.size == 0:
        raise ValueError(
            "Mesh has no inlet faces; cannot set Dirichlet concentrations."
        )

    boundary_tag = np.asarray(mesh.boundary_tag_per_face, dtype=np.int32)
    tag_to_name = {
        int(k): _normalize_name(v) for k, v in mesh.boundary_face_names.items()
    }
    left_tags = {
        tag for tag, name in tag_to_name.items() if "inlet" in name and "left" in name
    }
    right_tags = {
        tag for tag, name in tag_to_name.items() if "inlet" in name and "right" in name
    }

    mode = "named"
    notes: list[str] = []
    if left_tags and right_tags:
        left_faces = inlet_faces[np.isin(boundary_tag[inlet_faces], list(left_tags))]
        right_faces = inlet_faces[np.isin(boundary_tag[inlet_faces], list(right_tags))]
    else:
        mode = "x_split_fallback"
        inlet_x = mesh.face_centers[inlet_faces, 0]
        split = float(np.median(inlet_x))
        left_faces = inlet_faces[inlet_x <= split]
        right_faces = inlet_faces[inlet_x > split]
        notes.append(
            "Left/right inlet names not fully resolved; fallback split by inlet face-center x."
        )

    if left_faces.size == 0 or right_faces.size == 0:
        inlet_x = mesh.face_centers[inlet_faces, 0]
        order = np.argsort(inlet_x)
        half = max(1, inlet_faces.size // 2)
        left_faces = inlet_faces[order[:half]]
        right_faces = inlet_faces[order[half:]]
        mode = "ordered_fallback"
        notes.append("Secondary fallback by sorted inlet x order.")

    left_cells = np.unique(mesh.face_to_cells[left_faces, 0]).astype(np.int64)
    right_cells = np.unique(mesh.face_to_cells[right_faces, 0]).astype(np.int64)
    return {
        "left_faces": left_faces.astype(np.int64),
        "right_faces": right_faces.astype(np.int64),
        "left_cells": left_cells,
        "right_cells": right_cells,
        "mode": mode,
        "notes": notes,
    }


def build_diffusion_boundary_maps(
    mesh: ImportedTetraMesh,
    inlet_groups: Dict[str, object],
    *,
    left_value: float,
    right_value: float,
) -> Tuple[Dict[int, float], np.ndarray]:
    """Create boundary Dirichlet map and no-flux face set for diffusion laplacian."""

    left_faces = np.asarray(inlet_groups["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet_groups["right_faces"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    boundary_dirichlet: Dict[int, float] = {}
    for face_idx in left_faces.tolist():
        boundary_dirichlet[int(face_idx)] = float(left_value)
    for face_idx in right_faces.tolist():
        boundary_dirichlet[int(face_idx)] = float(right_value)

    boundary_no_flux = np.unique(np.concatenate((wall_faces, outlet_faces))).astype(
        np.int64
    )
    return boundary_dirichlet, boundary_no_flux


def build_diffusion_stencil(
    mesh: ImportedTetraMesh,
    *,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
    boundary_neumann: Mapping[int, float] | None = None,
    boundary_robin: Mapping[int, tuple[float, float, float]] | None = None,
    periodic_face_pairs: Sequence[tuple[int, int]] = (),
) -> DiffusionStencil:
    """Precompute diffusion geometry.

    Neumann values are outward normal derivatives ``d(phi)/dn``. Robin values
    use ``alpha*phi + beta*d(phi)/dn = gamma`` at the boundary face.
    """

    no_flux_set = set(np.asarray(boundary_no_flux_faces, dtype=np.int64).tolist())
    neumann_map = {int(k): float(v) for k, v in (boundary_neumann or {}).items()}
    robin_map = {
        int(k): (float(v[0]), float(v[1]), float(v[2]))
        for k, v in (boundary_robin or {}).items()
    }
    periodic_pairs = [
        (int(source), int(target)) for source, target in periodic_face_pairs
    ]
    periodic_faces = {face for pair in periodic_pairs for face in pair}
    assignments = (
        set(boundary_dirichlet),
        no_flux_set,
        set(neumann_map),
        set(robin_map),
        periodic_faces,
    )
    claimed: set[int] = set()
    for assigned in assignments:
        overlap = claimed.intersection(assigned)
        if overlap:
            raise ValueError(
                f"diffusion boundary faces have conflicting conditions: {sorted(overlap)}"
            )
        claimed.update(assigned)
    interior_c0: list[int] = []
    interior_c1: list[int] = []
    interior_coeff: list[float] = []
    interior_alpha: list[float] = []
    interior_area: list[float] = []
    interior_normals: list[np.ndarray] = []
    interior_correction_vec: list[np.ndarray] = []
    dirichlet_c0: list[int] = []
    dirichlet_coeff: list[float] = []
    dirichlet_area: list[float] = []
    dirichlet_normal: list[np.ndarray] = []
    dirichlet_delta: list[np.ndarray] = []
    dirichlet_value: list[float] = []
    neumann_c0: list[int] = []
    neumann_area: list[float] = []
    neumann_gradient: list[float] = []
    robin_c0: list[int] = []
    robin_area: list[float] = []
    robin_distance: list[float] = []
    robin_alpha: list[float] = []
    robin_beta: list[float] = []
    robin_gamma: list[float] = []

    for face_idx in range(mesh.face_vertices.shape[0]):
        c0 = int(mesh.face_to_cells[face_idx, 0])
        c1 = int(mesh.face_to_cells[face_idx, 1])
        area = float(mesh.face_areas[face_idx])
        normal = np.asarray(mesh.face_normals[face_idx], dtype=np.float64)
        if c1 >= 0:
            delta = mesh.cell_centers[c1] - mesh.cell_centers[c0]
            dist = max(abs(float(np.dot(delta, normal))), 1e-12)
            s_vec = area * normal
            d_norm2 = max(float(np.dot(delta, delta)), 1e-20)
            alpha = float(np.dot(s_vec, delta)) / d_norm2
            correction_vec = s_vec - alpha * delta
            interior_c0.append(c0)
            interior_c1.append(c1)
            interior_coeff.append(area / dist)
            interior_alpha.append(alpha)
            interior_area.append(area)
            interior_normals.append(normal)
            interior_correction_vec.append(correction_vec)
            continue
        if face_idx in periodic_faces or face_idx in no_flux_set:
            continue
        delta_b = mesh.face_centers[face_idx] - mesh.cell_centers[c0]
        dist_b = max(abs(float(np.dot(delta_b, normal))), 1e-12)
        if face_idx in boundary_dirichlet:
            dirichlet_c0.append(c0)
            dirichlet_coeff.append(area / dist_b)
            dirichlet_area.append(area)
            dirichlet_normal.append(normal)
            dirichlet_delta.append(delta_b)
            dirichlet_value.append(float(boundary_dirichlet[face_idx]))
        elif face_idx in neumann_map:
            neumann_c0.append(c0)
            neumann_area.append(area)
            neumann_gradient.append(neumann_map[face_idx])
        elif face_idx in robin_map:
            alpha, beta, gamma = robin_map[face_idx]
            denominator = beta + alpha * dist_b
            if abs(denominator) <= 1e-14:
                raise ValueError(
                    f"Robin boundary denominator is zero on face {face_idx}"
                )
            robin_c0.append(c0)
            robin_area.append(area)
            robin_distance.append(dist_b)
            robin_alpha.append(alpha)
            robin_beta.append(beta)
            robin_gamma.append(gamma)

    for source_face, target_face in periodic_pairs:
        source_owner = int(mesh.face_to_cells[source_face, 0])
        target_owner = int(mesh.face_to_cells[target_face, 0])
        if source_owner < 0 or target_owner < 0:
            raise ValueError("periodic faces must be boundary faces with owner cells")
        source_normal = np.asarray(mesh.face_normals[source_face], dtype=np.float64)
        effective_delta = (
            mesh.face_centers[source_face]
            - mesh.cell_centers[source_owner]
            + mesh.cell_centers[target_owner]
            - mesh.face_centers[target_face]
        )
        distance = max(abs(float(np.dot(effective_delta, source_normal))), 1e-12)
        area = 0.5 * (
            float(mesh.face_areas[source_face]) + float(mesh.face_areas[target_face])
        )
        s_vec = area * source_normal
        d_norm2 = max(float(np.dot(effective_delta, effective_delta)), 1e-20)
        interior_c0.append(source_owner)
        interior_c1.append(target_owner)
        interior_coeff.append(area / distance)
        interior_alpha.append(float(np.dot(s_vec, effective_delta)) / d_norm2)
        interior_area.append(area)
        interior_normals.append(source_normal)
        interior_correction_vec.append(
            s_vec - (float(np.dot(s_vec, effective_delta)) / d_norm2) * effective_delta
        )

    return DiffusionStencil(
        interior_c0=np.asarray(interior_c0, dtype=np.int64),
        interior_c1=np.asarray(interior_c1, dtype=np.int64),
        interior_coeff=np.asarray(interior_coeff, dtype=np.float64),
        interior_alpha=np.asarray(interior_alpha, dtype=np.float64),
        interior_area=np.asarray(interior_area, dtype=np.float64),
        interior_normals=np.asarray(interior_normals, dtype=np.float64).reshape(-1, 3),
        interior_correction_vec=np.asarray(
            interior_correction_vec,
            dtype=np.float64,
        ).reshape(-1, 3),
        dirichlet_c0=np.asarray(dirichlet_c0, dtype=np.int64),
        dirichlet_coeff=np.asarray(dirichlet_coeff, dtype=np.float64),
        dirichlet_area=np.asarray(dirichlet_area, dtype=np.float64),
        dirichlet_normal=np.asarray(dirichlet_normal, dtype=np.float64).reshape(-1, 3),
        dirichlet_cell_to_face_delta=np.asarray(
            dirichlet_delta, dtype=np.float64
        ).reshape(-1, 3),
        dirichlet_value=np.asarray(dirichlet_value, dtype=np.float64),
        neumann_c0=np.asarray(neumann_c0, dtype=np.int64),
        neumann_area=np.asarray(neumann_area, dtype=np.float64),
        neumann_normal_gradient=np.asarray(neumann_gradient, dtype=np.float64),
        robin_c0=np.asarray(robin_c0, dtype=np.int64),
        robin_area=np.asarray(robin_area, dtype=np.float64),
        robin_distance=np.asarray(robin_distance, dtype=np.float64),
        robin_alpha=np.asarray(robin_alpha, dtype=np.float64),
        robin_beta=np.asarray(robin_beta, dtype=np.float64),
        robin_gamma=np.asarray(robin_gamma, dtype=np.float64),
        volumes=np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-16),
    )


def _estimate_stable_dt_explicit(
    stencil: DiffusionStencil, diffusivity: float
) -> float:
    """Conservative explicit stability estimate using TPFA-like coefficient sum."""

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


def _add_prescribed_boundary_flux_numpy(
    lap: np.ndarray, scalar: np.ndarray, stencil: DiffusionStencil
) -> None:
    if stencil.neumann_c0.size:
        flux = stencil.neumann_area * stencil.neumann_normal_gradient
        np.add.at(lap, stencil.neumann_c0, flux / stencil.volumes[stencil.neumann_c0])
    if stencil.robin_c0.size:
        denominator = stencil.robin_beta + stencil.robin_alpha * stencil.robin_distance
        gradient = (
            stencil.robin_gamma - stencil.robin_alpha * scalar[stencil.robin_c0]
        ) / denominator
        flux = stencil.robin_area * gradient
        np.add.at(lap, stencil.robin_c0, flux / stencil.volumes[stencil.robin_c0])


def _enforce_dirichlet_on_cells(
    values: np.ndarray,
    *,
    left_cells: np.ndarray,
    right_cells: np.ndarray,
    left_value: float,
    right_value: float,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[left_cells] = float(left_value)
    out[right_cells] = float(right_value)
    return out


def _laplacian_from_stencil_tpfa_numpy(
    scalar: np.ndarray, stencil: DiffusionStencil
) -> np.ndarray:
    lap = np.zeros_like(scalar, dtype=np.float64)
    if stencil.interior_c0.size:
        diff = scalar[stencil.interior_c1] - scalar[stencil.interior_c0]
        flux = stencil.interior_coeff * diff
        np.add.at(lap, stencil.interior_c0, flux / stencil.volumes[stencil.interior_c0])
        np.add.at(
            lap, stencil.interior_c1, -flux / stencil.volumes[stencil.interior_c1]
        )
    if stencil.dirichlet_c0.size:
        flux_b = stencil.dirichlet_coeff * (
            stencil.dirichlet_value - scalar[stencil.dirichlet_c0]
        )
        np.add.at(
            lap, stencil.dirichlet_c0, flux_b / stencil.volumes[stencil.dirichlet_c0]
        )
    _add_prescribed_boundary_flux_numpy(lap, scalar, stencil)
    return lap


def _laplacian_from_stencil_lsq_flux_numpy(
    scalar: np.ndarray,
    mesh: ImportedTetraMesh,
    stencil: DiffusionStencil,
    *,
    gradient_method: GradientMethod,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
) -> tuple[np.ndarray, bool]:
    gradients = scalar_gradient(
        scalar,
        mesh,
        method=gradient_method,
        boundary_face_values=boundary_dirichlet if boundary_dirichlet else None,
        boundary_no_flux_faces=boundary_no_flux_faces,
    )
    lap = np.zeros_like(scalar, dtype=np.float64)
    if stencil.interior_c0.size:
        grad_face = 0.5 * (
            gradients[stencil.interior_c0] + gradients[stencil.interior_c1]
        )
        orth_flux = stencil.interior_alpha * (
            scalar[stencil.interior_c1] - scalar[stencil.interior_c0]
        )
        corr_flux = np.einsum("ij,ij->i", grad_face, stencil.interior_correction_vec)
        flux = orth_flux + corr_flux
        np.add.at(lap, stencil.interior_c0, flux / stencil.volumes[stencil.interior_c0])
        np.add.at(
            lap, stencil.interior_c1, -flux / stencil.volumes[stencil.interior_c1]
        )
    if stencil.dirichlet_c0.size:
        flux_b = stencil.dirichlet_coeff * (
            stencil.dirichlet_value - scalar[stencil.dirichlet_c0]
        )
        np.add.at(
            lap, stencil.dirichlet_c0, flux_b / stencil.volumes[stencil.dirichlet_c0]
        )
    _add_prescribed_boundary_flux_numpy(lap, scalar, stencil)
    return lap, False


def _laplacian_from_stencil_tpfa_torch(
    scalar: np.ndarray,
    stencil: DiffusionStencil,
    *,
    device: str,
) -> tuple[np.ndarray, bool]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return _laplacian_from_stencil_tpfa_numpy(scalar, stencil), True

    scalar_t = torch.as_tensor(scalar, dtype=torch.float64, device=device)
    lap = torch.zeros_like(scalar_t)
    volumes = torch.as_tensor(stencil.volumes, dtype=torch.float64, device=device)
    if stencil.interior_c0.size:
        c0 = torch.as_tensor(stencil.interior_c0, dtype=torch.long, device=device)
        c1 = torch.as_tensor(stencil.interior_c1, dtype=torch.long, device=device)
        coeff = torch.as_tensor(
            stencil.interior_coeff, dtype=torch.float64, device=device
        )
        flux = coeff * (scalar_t[c1] - scalar_t[c0])
        lap.index_add_(0, c0, flux / volumes[c0])
        lap.index_add_(0, c1, -flux / volumes[c1])
    if stencil.dirichlet_c0.size:
        cb = torch.as_tensor(stencil.dirichlet_c0, dtype=torch.long, device=device)
        coeff_b = torch.as_tensor(
            stencil.dirichlet_coeff, dtype=torch.float64, device=device
        )
        value_b = torch.as_tensor(
            stencil.dirichlet_value, dtype=torch.float64, device=device
        )
        flux_b = coeff_b * (value_b - scalar_t[cb])
        lap.index_add_(0, cb, flux_b / volumes[cb])
    if stencil.neumann_c0.size:
        nb = torch.as_tensor(stencil.neumann_c0, dtype=torch.long, device=device)
        area = torch.as_tensor(stencil.neumann_area, dtype=torch.float64, device=device)
        gradient = torch.as_tensor(
            stencil.neumann_normal_gradient, dtype=torch.float64, device=device
        )
        lap.index_add_(0, nb, area * gradient / volumes[nb])
    if stencil.robin_c0.size:
        rb = torch.as_tensor(stencil.robin_c0, dtype=torch.long, device=device)
        area = torch.as_tensor(stencil.robin_area, dtype=torch.float64, device=device)
        distance = torch.as_tensor(
            stencil.robin_distance, dtype=torch.float64, device=device
        )
        alpha = torch.as_tensor(stencil.robin_alpha, dtype=torch.float64, device=device)
        beta = torch.as_tensor(stencil.robin_beta, dtype=torch.float64, device=device)
        gamma = torch.as_tensor(stencil.robin_gamma, dtype=torch.float64, device=device)
        gradient = (gamma - alpha * scalar_t[rb]) / (beta + alpha * distance)
        lap.index_add_(0, rb, area * gradient / volumes[rb])
    return lap.detach().cpu().numpy(), False


def _laplacian_from_stencil_lsq_flux_torch(
    scalar: np.ndarray,
    mesh: ImportedTetraMesh,
    stencil: DiffusionStencil,
    *,
    gradient_method: GradientMethod,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
    device: str,
) -> tuple[np.ndarray, bool]:
    """Torch accumulation for lsq_flux; gradient reconstruction is currently numpy-based."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return _laplacian_from_stencil_lsq_flux_numpy(
            scalar,
            mesh,
            stencil,
            gradient_method=gradient_method,
            boundary_dirichlet=boundary_dirichlet,
            boundary_no_flux_faces=boundary_no_flux_faces,
        )[0], True

    gradients = scalar_gradient(
        scalar,
        mesh,
        method=gradient_method,
        boundary_face_values=boundary_dirichlet if boundary_dirichlet else None,
        boundary_no_flux_faces=boundary_no_flux_faces,
    )
    used_numpy_fallback = True  # gradient reconstruction is numpy-only in this version

    scalar_t = torch.as_tensor(scalar, dtype=torch.float64, device=device)
    lap = torch.zeros_like(scalar_t)
    volumes = torch.as_tensor(stencil.volumes, dtype=torch.float64, device=device)
    grad_t = torch.as_tensor(gradients, dtype=torch.float64, device=device)

    if stencil.interior_c0.size:
        c0 = torch.as_tensor(stencil.interior_c0, dtype=torch.long, device=device)
        c1 = torch.as_tensor(stencil.interior_c1, dtype=torch.long, device=device)
        alpha = torch.as_tensor(
            stencil.interior_alpha, dtype=torch.float64, device=device
        )
        correction = torch.as_tensor(
            stencil.interior_correction_vec,
            dtype=torch.float64,
            device=device,
        )
        grad_face = 0.5 * (grad_t[c0] + grad_t[c1])
        orth_flux = alpha * (scalar_t[c1] - scalar_t[c0])
        corr_flux = torch.sum(grad_face * correction, dim=1)
        flux = orth_flux + corr_flux
        lap.index_add_(0, c0, flux / volumes[c0])
        lap.index_add_(0, c1, -flux / volumes[c1])

    if stencil.dirichlet_c0.size:
        cb = torch.as_tensor(stencil.dirichlet_c0, dtype=torch.long, device=device)
        coeff_b = torch.as_tensor(
            stencil.dirichlet_coeff, dtype=torch.float64, device=device
        )
        value_b = torch.as_tensor(
            stencil.dirichlet_value, dtype=torch.float64, device=device
        )
        flux_b = coeff_b * (value_b - scalar_t[cb])
        lap.index_add_(0, cb, flux_b / volumes[cb])
    if stencil.neumann_c0.size:
        nb = torch.as_tensor(stencil.neumann_c0, dtype=torch.long, device=device)
        area = torch.as_tensor(stencil.neumann_area, dtype=torch.float64, device=device)
        gradient = torch.as_tensor(
            stencil.neumann_normal_gradient, dtype=torch.float64, device=device
        )
        lap.index_add_(0, nb, area * gradient / volumes[nb])
    if stencil.robin_c0.size:
        rb = torch.as_tensor(stencil.robin_c0, dtype=torch.long, device=device)
        area = torch.as_tensor(stencil.robin_area, dtype=torch.float64, device=device)
        distance = torch.as_tensor(
            stencil.robin_distance, dtype=torch.float64, device=device
        )
        alpha = torch.as_tensor(stencil.robin_alpha, dtype=torch.float64, device=device)
        beta = torch.as_tensor(stencil.robin_beta, dtype=torch.float64, device=device)
        gamma = torch.as_tensor(stencil.robin_gamma, dtype=torch.float64, device=device)
        gradient = (gamma - alpha * scalar_t[rb]) / (beta + alpha * distance)
        lap.index_add_(0, rb, area * gradient / volumes[rb])
    return lap.detach().cpu().numpy(), used_numpy_fallback


def run_scalar_diffusion_debug(
    mesh: ImportedTetraMesh,
    config: GmshTetraScalarConfig,
) -> Dict[str, object]:
    """Run explicit diffusion-only prototype on imported tetra mesh."""

    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.dt <= 0:
        raise ValueError("dt must be positive.")
    if config.diffusivity < 0:
        raise ValueError("diffusivity must be non-negative.")

    inlet_groups = resolve_inlet_face_groups(mesh)
    boundary_dirichlet, boundary_no_flux = build_diffusion_boundary_maps(
        mesh,
        inlet_groups,
        left_value=config.left_inlet_value,
        right_value=config.right_inlet_value,
    )
    stencil = build_diffusion_stencil(
        mesh,
        boundary_dirichlet=boundary_dirichlet,
        boundary_no_flux_faces=boundary_no_flux,
    )
    stable_dt = _estimate_stable_dt_explicit(stencil, config.diffusivity)
    stability_ratio = (
        float(config.dt / stable_dt)
        if np.isfinite(stable_dt) and stable_dt > 0
        else 0.0
    )
    stable_warning = bool(np.isfinite(stable_dt) and config.dt > stable_dt)

    left_cells = np.asarray(inlet_groups["left_cells"], dtype=np.int64)
    right_cells = np.asarray(inlet_groups["right_cells"], dtype=np.int64)
    scalar = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    scalar = _enforce_dirichlet_on_cells(
        scalar,
        left_cells=left_cells,
        right_cells=right_cells,
        left_value=config.left_inlet_value,
        right_value=config.right_inlet_value,
    )

    used_numpy_fallback = False
    clipping_used = False
    overshoot_max = 0.0
    undershoot_max = 0.0
    history: list[Dict[str, float | int | bool]] = []
    mass_prev = float(np.sum(scalar * mesh.cell_volumes))
    for step in range(1, config.steps + 1):
        scalar = _enforce_dirichlet_on_cells(
            scalar,
            left_cells=left_cells,
            right_cells=right_cells,
            left_value=config.left_inlet_value,
            right_value=config.right_inlet_value,
        )

        if config.backend == "torch":
            if config.laplacian_method == "tpfa":
                lap, fallback = _laplacian_from_stencil_tpfa_torch(
                    scalar,
                    stencil,
                    device=config.torch_device,
                )
            else:
                lap, fallback = _laplacian_from_stencil_lsq_flux_torch(
                    scalar,
                    mesh,
                    stencil,
                    gradient_method=config.gradient_method,
                    boundary_dirichlet=boundary_dirichlet,
                    boundary_no_flux_faces=boundary_no_flux,
                    device=config.torch_device,
                )
            used_numpy_fallback = used_numpy_fallback or fallback
        else:
            if config.laplacian_method == "tpfa":
                lap = _laplacian_from_stencil_tpfa_numpy(scalar, stencil)
            else:
                lap, _ = _laplacian_from_stencil_lsq_flux_numpy(
                    scalar,
                    mesh,
                    stencil,
                    gradient_method=config.gradient_method,
                    boundary_dirichlet=boundary_dirichlet,
                    boundary_no_flux_faces=boundary_no_flux,
                )

        updated_raw = scalar + config.dt * config.diffusivity * lap
        updated_raw = _enforce_dirichlet_on_cells(
            updated_raw,
            left_cells=left_cells,
            right_cells=right_cells,
            left_value=config.left_inlet_value,
            right_value=config.right_inlet_value,
        )
        local_overshoot = float(max(np.max(updated_raw - config.clip_max), 0.0))
        local_undershoot = float(max(np.max(config.clip_min - updated_raw), 0.0))
        overshoot_max = max(overshoot_max, local_overshoot)
        undershoot_max = max(undershoot_max, local_undershoot)
        if local_overshoot > 0.0 or local_undershoot > 0.0:
            clipping_used = True
        updated = np.clip(updated_raw, config.clip_min, config.clip_max)

        max_change = float(np.max(np.abs(updated - scalar)))
        scalar = updated
        if not np.all(np.isfinite(scalar)):
            raise FloatingPointError(f"Scalar field became non-finite at step {step}.")

        mass = float(np.sum(scalar * mesh.cell_volumes))
        left_vals = scalar[left_cells] if left_cells.size else np.asarray([np.nan])
        right_vals = scalar[right_cells] if right_cells.size else np.asarray([np.nan])
        entry: Dict[str, float | int | bool] = {
            "step": int(step),
            "min": float(np.min(scalar)),
            "max": float(np.max(scalar)),
            "mean": float(np.mean(scalar)),
            "total_mass_proxy": mass,
            "mass_delta": float(mass - mass_prev),
            "max_change": max_change,
            "left_inlet_min": float(np.min(left_vals)),
            "left_inlet_max": float(np.max(left_vals)),
            "right_inlet_min": float(np.min(right_vals)),
            "right_inlet_max": float(np.max(right_vals)),
            "stability_warning": stable_warning,
            "stability_ratio_dt_to_limit": stability_ratio,
            "overshoot_before_clip": local_overshoot,
            "undershoot_before_clip": local_undershoot,
        }
        history.append(entry)
        mass_prev = mass
        if config.progress_every > 0 and step % config.progress_every == 0:
            print(
                "[gmsh-tetra-scalar] "
                f"step {step}/{config.steps}, min={entry['min']:.6f}, max={entry['max']:.6f}"
            )

    gradient = scalar_gradient(
        scalar,
        mesh,
        method=config.gradient_method,
        boundary_face_values=boundary_dirichlet if boundary_dirichlet else None,
        boundary_no_flux_faces=boundary_no_flux,
    )
    gradient_norm = np.linalg.norm(gradient, axis=1)
    return {
        "scalar": scalar,
        "gradient": gradient,
        "gradient_norm": gradient_norm,
        "history": history,
        "stable_dt_estimate": stable_dt,
        "stability_ratio_dt_to_limit": stability_ratio,
        "stability_warning": stable_warning,
        "boundary_setup": {
            "inlet_resolution_mode": inlet_groups["mode"],
            "inlet_resolution_notes": inlet_groups["notes"],
            "left_inlet_faces": int(np.asarray(inlet_groups["left_faces"]).size),
            "right_inlet_faces": int(np.asarray(inlet_groups["right_faces"]).size),
            "left_inlet_cells": int(left_cells.size),
            "right_inlet_cells": int(right_cells.size),
            "outlet_faces": int(mesh.outlet_faces.size),
            "wall_faces": int(mesh.wall_faces.size),
        },
        "backend_execution": {
            "requested_backend": config.backend,
            "torch_device": config.torch_device,
            "used_numpy_fallback": bool(used_numpy_fallback),
        },
        "methods": {
            "gradient_method": config.gradient_method,
            "laplacian_method": config.laplacian_method,
            "stability_estimate_method": "tpfa_like_upper_bound",
        },
        "clipping": {
            "enabled": True,
            "used": clipping_used,
            "max_overshoot_before_clip": overshoot_max,
            "max_undershoot_before_clip": undershoot_max,
        },
        "final_stats": {
            "min": float(np.min(scalar)),
            "max": float(np.max(scalar)),
            "mean": float(np.mean(scalar)),
            "gradient_norm_max": float(np.max(gradient_norm)),
            "gradient_norm_mean": float(np.mean(gradient_norm)),
        },
    }
