"""Minimal mesh-native scalar solver prototype (diffusion-only)."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from microfluidics.mesh.mesh_builder import MeshDataModel
from microfluidics.mesh.mesh_fields import MeshScalarField, constant_scalar_field
from microfluidics.mesh.mesh_geometry import MeshGeometryModel3D
from microfluidics.mesh.mesh_linear_systems import (
    build_diffusion_system,
    solve_diffusion_implicit_jacobi,
)
from microfluidics.mesh.mesh_operators import (
    MeshTopology,
    build_mesh_topology,
    laplacian_cell_scalar,
)


@dataclass(frozen=True)
class MeshScalarDiffusionConfig:
    dt: float = 5e-4
    diffusivity: float = 3e-10
    steps: int = 100
    left_inlet_value: float = 0.0
    right_inlet_value: float = 1.0
    use_implicit: bool = True
    max_iters: int = 250
    tol: float = 1e-10
    progress_every: int = 10
    clip_min: float = 0.0
    clip_max: float = 1.0


def _expanded_contains(
    centers: np.ndarray,
    box,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    eps_x = 0.5 * dx + 1e-12
    eps_y = 0.5 * dy + 1e-12
    eps_z = 0.5 * dz + 1e-12
    return (
        (centers[:, 0] >= box.x[0] - eps_x)
        & (centers[:, 0] <= box.x[1] + eps_x)
        & (centers[:, 1] >= box.y[0] - eps_y)
        & (centers[:, 1] <= box.y[1] + eps_y)
        & (centers[:, 2] >= box.z[0] - eps_z)
        & (centers[:, 2] <= box.z[1] + eps_z)
    )


def identify_boundary_cells(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
) -> Dict[str, np.ndarray]:
    """Identify cell groups attached to tagged boundaries."""
    centers = mesh.cell_centers
    dx = float(mesh.metadata["dx"])
    dy = float(mesh.metadata["dy"])
    dz = float(mesh.metadata["dz"])
    left_mask = _expanded_contains(
        centers, geometry.boundary_patches["left_inlet"], dx, dy, dz
    )
    right_mask = _expanded_contains(
        centers, geometry.boundary_patches["right_inlet"], dx, dy, dz
    )
    outlet_mask = _expanded_contains(
        centers, geometry.boundary_patches["outlet"], dx, dy, dz
    )

    boundary_union = left_mask | right_mask | outlet_mask

    wall_layer = mesh.cell_data.get("wall_layer_id")
    if wall_layer is None:
        wall_mask = np.zeros(centers.shape[0], dtype=bool)
    else:
        wall_mask = (np.asarray(wall_layer) == 0) & (~boundary_union)
    cavity_zone = mesh.cell_data.get("zone_id")
    if cavity_zone is None:
        cavity_wall_mask = np.zeros(centers.shape[0], dtype=bool)
    else:
        cavity_zone = np.asarray(cavity_zone)
        cavity_wall_mask = wall_mask & np.isin(cavity_zone, [4, 5])
    return {
        "left_inlet": np.flatnonzero(left_mask),
        "right_inlet": np.flatnonzero(right_mask),
        "outlet": np.flatnonzero(outlet_mask),
        "wall": np.flatnonzero(wall_mask),
        "cavity_wall": np.flatnonzero(cavity_wall_mask),
    }


def apply_inlet_dirichlet(
    field: MeshScalarField,
    boundary_cells: Dict[str, np.ndarray],
    left_value: float,
    right_value: float,
) -> MeshScalarField:
    values = field.values.copy()
    values[boundary_cells["left_inlet"]] = left_value
    values[boundary_cells["right_inlet"]] = right_value
    return MeshScalarField(field.mesh, values, name=field.name)


def advance_diffusion_explicit(
    field: MeshScalarField,
    topology: MeshTopology,
    dt: float,
    diffusivity: float,
) -> MeshScalarField:
    lap = laplacian_cell_scalar(field.values, topology, boundary_mode="neumann0")
    values = field.values + dt * diffusivity * lap
    return MeshScalarField(field.mesh, values, name=field.name)


def advance_diffusion_implicit(
    field: MeshScalarField,
    topology: MeshTopology,
    dt: float,
    diffusivity: float,
    fixed_mask: np.ndarray,
    fixed_values: np.ndarray,
    max_iters: int,
    tol: float,
) -> tuple[MeshScalarField, int, float]:
    system = build_diffusion_system(
        topology,
        diffusivity=diffusivity,
        dt=dt,
        boundary_mode="neumann0",
    )
    values, iterations, residual = solve_diffusion_implicit_jacobi(
        rhs_values=field.values,
        topology=topology,
        system=system,
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
        max_iters=max_iters,
        tol=tol,
    )
    return MeshScalarField(field.mesh, values, name=field.name), iterations, residual


def run_mesh_scalar_diffusion_prototype(
    mesh: MeshDataModel,
    geometry: MeshGeometryModel3D,
    config: MeshScalarDiffusionConfig,
    initial_field: MeshScalarField | None = None,
) -> dict:
    """Run diffusion-only scalar prototype with inlet Dirichlet boundary values."""
    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.dt <= 0:
        raise ValueError("dt must be positive.")
    if config.diffusivity < 0:
        raise ValueError("diffusivity must be non-negative.")

    topology = build_mesh_topology(mesh)
    boundary_cells = identify_boundary_cells(mesh, geometry)
    if initial_field is None:
        field = constant_scalar_field(mesh, 0.0, name="concentration_native")
    else:
        field = initial_field.copy()

    fixed_mask = np.zeros(mesh.cells.shape[0], dtype=bool)
    fixed_mask[boundary_cells["left_inlet"]] = True
    fixed_mask[boundary_cells["right_inlet"]] = True
    fixed_values = field.values.copy()
    fixed_values[boundary_cells["left_inlet"]] = config.left_inlet_value
    fixed_values[boundary_cells["right_inlet"]] = config.right_inlet_value

    field = apply_inlet_dirichlet(
        field,
        boundary_cells=boundary_cells,
        left_value=config.left_inlet_value,
        right_value=config.right_inlet_value,
    )
    residual_history: list[float] = []
    iter_history: list[int] = []

    for step in range(config.steps):
        field = apply_inlet_dirichlet(
            field,
            boundary_cells=boundary_cells,
            left_value=config.left_inlet_value,
            right_value=config.right_inlet_value,
        )
        if config.use_implicit:
            field, iterations, residual = advance_diffusion_implicit(
                field,
                topology=topology,
                dt=config.dt,
                diffusivity=config.diffusivity,
                fixed_mask=fixed_mask,
                fixed_values=fixed_values,
                max_iters=config.max_iters,
                tol=config.tol,
            )
        else:
            field = advance_diffusion_explicit(
                field,
                topology=topology,
                dt=config.dt,
                diffusivity=config.diffusivity,
            )
            iterations = 1
            residual = float("nan")
        field = apply_inlet_dirichlet(
            field,
            boundary_cells=boundary_cells,
            left_value=config.left_inlet_value,
            right_value=config.right_inlet_value,
        )
        clipped = np.clip(field.values, config.clip_min, config.clip_max)
        field = MeshScalarField(mesh, clipped, name=field.name)
        residual_history.append(float(residual))
        iter_history.append(int(iterations))
        if config.progress_every > 0 and ((step + 1) % config.progress_every == 0):
            print(
                f"[native-scalar] step {step + 1}/{config.steps}, "
                f"min={field.values.min():.6f}, max={field.values.max():.6f}"
            )

    return {
        "field": field,
        "topology": topology,
        "boundary_cells": boundary_cells,
        "residual_history": residual_history,
        "iteration_history": iter_history,
        "stats": {
            "min": float(np.min(field.values)),
            "max": float(np.max(field.values)),
            "mean": float(np.mean(field.values)),
        },
    }
