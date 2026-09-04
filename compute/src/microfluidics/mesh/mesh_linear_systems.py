"""Linear-system helpers for mesh-native scalar diffusion prototypes."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from microfluidics.mesh.mesh_operators import MeshTopology


@dataclass(frozen=True)
class DiffusionSystem:
    diag: np.ndarray
    offdiag: np.ndarray  # shape (n_cells, 6), non-negative coefficients
    rhs_dirichlet: np.ndarray


@dataclass(frozen=True)
class PoissonSystem:
    diag: np.ndarray
    offdiag: np.ndarray  # shape (n_cells, 6), non-negative coefficients
    rhs_dirichlet: np.ndarray


def build_diffusion_system(
    topology: MeshTopology,
    diffusivity: float,
    dt: float,
    boundary_mode: str = "neumann0",
    dirichlet_value: float = 0.0,
) -> DiffusionSystem:
    """Build per-cell coefficients for implicit diffusion solve."""
    if diffusivity < 0:
        raise ValueError("diffusivity must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    n_cells = topology.n_cells
    offdiag = np.zeros((n_cells, 6), dtype=np.float64)
    rhs_dirichlet = np.zeros(n_cells, dtype=np.float64)

    coeff_x = diffusivity * dt / (topology.dx * topology.dx)
    coeff_y = diffusivity * dt / (topology.dy * topology.dy)
    coeff_z = diffusivity * dt / (topology.dz * topology.dz)
    face_coeffs = np.array(
        [coeff_x, coeff_x, coeff_y, coeff_y, coeff_z, coeff_z],
        dtype=np.float64,
    )
    diag = np.ones(n_cells, dtype=np.float64)

    for face_idx in range(6):
        neigh = topology.neighbor_ids[:, face_idx]
        coeff = face_coeffs[face_idx]
        has_neigh = neigh >= 0
        offdiag[has_neigh, face_idx] = coeff
        diag[has_neigh] += coeff
        if boundary_mode == "dirichlet":
            missing = ~has_neigh
            diag[missing] += coeff
            rhs_dirichlet[missing] += coeff * dirichlet_value
    return DiffusionSystem(diag=diag, offdiag=offdiag, rhs_dirichlet=rhs_dirichlet)


def solve_diffusion_implicit_jacobi(
    rhs_values: np.ndarray,
    topology: MeshTopology,
    system: DiffusionSystem,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    max_iters: int = 300,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, int, float]:
    """Solve implicit diffusion linear system with Jacobi iterations."""
    rhs = np.asarray(rhs_values, dtype=np.float64).reshape(-1)
    if rhs.shape[0] != topology.n_cells:
        raise ValueError("rhs_values size must match topology.n_cells.")
    x = rhs.copy()
    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed_mask.shape[0] != topology.n_cells:
            raise ValueError("fixed_mask size must match topology.n_cells.")
    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
        if fixed_values.shape[0] != topology.n_cells:
            raise ValueError("fixed_values size must match topology.n_cells.")

    for iteration in range(1, max_iters + 1):
        rhs_eff = rhs + system.rhs_dirichlet
        neighbor_sum = np.zeros_like(rhs_eff)
        for face_idx in range(6):
            neigh = topology.neighbor_ids[:, face_idx]
            coeff = system.offdiag[:, face_idx]
            has_neigh = neigh >= 0
            neighbor_sum[has_neigh] += coeff[has_neigh] * x[neigh[has_neigh]]
        x_new = (rhs_eff + neighbor_sum) / system.diag
        if fixed_mask is not None and fixed_values is not None:
            x_new[fixed_mask] = fixed_values[fixed_mask]
        residual = float(np.max(np.abs(x_new - x)))
        x = x_new
        if residual <= tol:
            return x, iteration, residual
    return x, max_iters, residual


def build_poisson_system(
    topology: MeshTopology,
    boundary_mode: str = "neumann0",
    dirichlet_value: float = 0.0,
) -> PoissonSystem:
    """Build coefficients for laplacian(p) = rhs."""
    n_cells = topology.n_cells
    offdiag = np.zeros((n_cells, 6), dtype=np.float64)
    rhs_dirichlet = np.zeros(n_cells, dtype=np.float64)
    coeff_x = 1.0 / (topology.dx * topology.dx)
    coeff_y = 1.0 / (topology.dy * topology.dy)
    coeff_z = 1.0 / (topology.dz * topology.dz)
    face_coeffs = np.array(
        [coeff_x, coeff_x, coeff_y, coeff_y, coeff_z, coeff_z],
        dtype=np.float64,
    )
    diag = np.zeros(n_cells, dtype=np.float64)
    for face_idx in range(6):
        neigh = topology.neighbor_ids[:, face_idx]
        coeff = face_coeffs[face_idx]
        has_neigh = neigh >= 0
        offdiag[has_neigh, face_idx] = coeff
        diag[has_neigh] += coeff
        if boundary_mode == "dirichlet":
            missing = ~has_neigh
            diag[missing] += coeff
            rhs_dirichlet[missing] += coeff * dirichlet_value
    return PoissonSystem(diag=diag, offdiag=offdiag, rhs_dirichlet=rhs_dirichlet)


def poisson_residual(
    pressure: np.ndarray,
    rhs: np.ndarray,
    topology: MeshTopology,
    system: PoissonSystem,
) -> np.ndarray:
    """Compute residual for laplacian(p) = rhs."""
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if p.shape[0] != topology.n_cells or rhs_arr.shape[0] != topology.n_cells:
        raise ValueError("pressure/rhs size must match topology.n_cells.")
    lhs = np.zeros_like(p)
    for face_idx in range(6):
        neigh = topology.neighbor_ids[:, face_idx]
        coeff = system.offdiag[:, face_idx]
        has_neigh = neigh >= 0
        lhs[has_neigh] += coeff[has_neigh] * (p[neigh[has_neigh]] - p[has_neigh])
    lhs += system.rhs_dirichlet
    return lhs - rhs_arr


def solve_poisson_jacobi(
    rhs_values: np.ndarray,
    topology: MeshTopology,
    system: PoissonSystem,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    max_iters: int = 600,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, int, float]:
    """Solve laplacian(p)=rhs using Jacobi with optional gauge constraints."""
    rhs = np.asarray(rhs_values, dtype=np.float64).reshape(-1)
    if rhs.shape[0] != topology.n_cells:
        raise ValueError("rhs_values size must match topology.n_cells.")
    if initial_guess is None:
        p = np.zeros_like(rhs)
    else:
        p = np.asarray(initial_guess, dtype=np.float64).reshape(-1).copy()
        if p.shape[0] != topology.n_cells:
            raise ValueError("initial_guess size must match topology.n_cells.")
    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed_mask.shape[0] != topology.n_cells:
            raise ValueError("fixed_mask size must match topology.n_cells.")
    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
        if fixed_values.shape[0] != topology.n_cells:
            raise ValueError("fixed_values size must match topology.n_cells.")

    for iteration in range(1, max_iters + 1):
        neighbour_term = np.zeros_like(p)
        for face_idx in range(6):
            neigh = topology.neighbor_ids[:, face_idx]
            coeff = system.offdiag[:, face_idx]
            has_neigh = neigh >= 0
            neighbour_term[has_neigh] += coeff[has_neigh] * p[neigh[has_neigh]]

        safe_diag = np.where(system.diag > 0.0, system.diag, 1.0)
        p_new = (neighbour_term + system.rhs_dirichlet - rhs) / safe_diag
        if fixed_mask is not None and fixed_values is not None:
            p_new[fixed_mask] = fixed_values[fixed_mask]
        delta = np.max(np.abs(p_new - p))
        p = p_new
        if float(delta) <= tol:
            residual_vec = poisson_residual(p, rhs, topology, system)
            return p, iteration, float(np.max(np.abs(residual_vec)))

    residual_vec = poisson_residual(p, rhs, topology, system)
    return p, max_iters, float(np.max(np.abs(residual_vec)))


def _apply_poisson_operator(
    pressure: np.ndarray,
    topology: MeshTopology,
    system: PoissonSystem,
) -> np.ndarray:
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if p.shape[0] != topology.n_cells:
        raise ValueError("pressure size must match topology.n_cells.")

    out = system.diag * p
    for face_idx in range(6):
        neigh = topology.neighbor_ids[:, face_idx]
        coeff = system.offdiag[:, face_idx]
        has_neigh = neigh >= 0
        out[has_neigh] -= coeff[has_neigh] * p[neigh[has_neigh]]
    return out


def solve_poisson_pcg(
    rhs_values: np.ndarray,
    topology: MeshTopology,
    system: PoissonSystem,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
    initial_guess: np.ndarray | None = None,
    max_iters: int = 600,
    tol: float = 1e-9,
    breakdown_eps: float = 1e-30,
) -> Tuple[np.ndarray, int, float]:
    """Solve laplacian(p)=rhs using diagonal-preconditioned conjugate gradient."""
    rhs = np.asarray(rhs_values, dtype=np.float64).reshape(-1)
    if rhs.shape[0] != topology.n_cells:
        raise ValueError("rhs_values size must match topology.n_cells.")

    if initial_guess is None:
        p = np.zeros_like(rhs)
    else:
        p = np.asarray(initial_guess, dtype=np.float64).reshape(-1).copy()
        if p.shape[0] != topology.n_cells:
            raise ValueError("initial_guess size must match topology.n_cells.")

    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed_mask.shape[0] != topology.n_cells:
            raise ValueError("fixed_mask size must match topology.n_cells.")
    if fixed_values is not None:
        fixed_values = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
        if fixed_values.shape[0] != topology.n_cells:
            raise ValueError("fixed_values size must match topology.n_cells.")

    b = system.rhs_dirichlet - rhs
    safe_diag = np.where(system.diag > 0.0, system.diag, 1.0)

    if fixed_mask is not None and fixed_values is not None:
        p[fixed_mask] = fixed_values[fixed_mask]

    residual = b - _apply_poisson_operator(p, topology, system)
    if fixed_mask is not None:
        residual[fixed_mask] = 0.0
    residual_max = float(np.max(np.abs(residual)))
    if residual_max <= tol:
        return p, 0, residual_max

    z = residual / safe_diag
    if fixed_mask is not None:
        z[fixed_mask] = 0.0
    direction = z.copy()
    rz_old = float(np.dot(residual, z))
    if abs(rz_old) <= breakdown_eps:
        return p, 0, residual_max

    iterations = max_iters
    for iteration in range(1, max_iters + 1):
        ap = _apply_poisson_operator(direction, topology, system)
        if fixed_mask is not None:
            ap[fixed_mask] = 0.0
        denom = float(np.dot(direction, ap))
        if abs(denom) <= breakdown_eps:
            iterations = iteration - 1
            break

        alpha = rz_old / denom
        p = p + alpha * direction
        if fixed_mask is not None and fixed_values is not None:
            p[fixed_mask] = fixed_values[fixed_mask]

        residual = residual - alpha * ap
        if fixed_mask is not None:
            residual[fixed_mask] = 0.0
        residual_max = float(np.max(np.abs(residual)))
        iterations = iteration
        if residual_max <= tol:
            break

        z = residual / safe_diag
        if fixed_mask is not None:
            z[fixed_mask] = 0.0
        rz_new = float(np.dot(residual, z))
        if abs(rz_new) <= breakdown_eps:
            break
        beta = rz_new / rz_old
        direction = z + beta * direction
        if fixed_mask is not None:
            direction[fixed_mask] = 0.0
        rz_old = rz_new

    residual_vec = poisson_residual(p, rhs, topology, system)
    return p, iterations, float(np.max(np.abs(residual_vec)))
