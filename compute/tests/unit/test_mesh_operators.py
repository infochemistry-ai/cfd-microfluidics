from dataclasses import replace

import numpy as np

from microfluidics.mesh.mesh_builder import build_mesh_scaffold
from microfluidics.mesh.mesh_config import MeshRefinementConfig, default_mesh_run_config
from microfluidics.mesh.mesh_geometry import build_mesh_geometry_3d
from microfluidics.mesh.mesh_operators import (
    build_mesh_topology,
    divergence_cell_vector,
    gradient_cell_scalar,
    laplacian_cell_scalar,
)
from microfluidics.mesh.mesh_scalar_solver import (
    MeshScalarDiffusionConfig,
    run_mesh_scalar_diffusion_prototype,
)


def _small_mesh(include_cavities: bool = False):
    run_cfg = default_mesh_run_config(case_tag="TJ", include_cavities=include_cavities)
    resolution = (36, 24, 10)
    run_cfg = replace(
        run_cfg,
        refinement=MeshRefinementConfig(
            volume_resolution=resolution,
            wall_refinement_thickness=run_cfg.refinement.wall_refinement_thickness,
            wall_refinement_levels=run_cfg.refinement.wall_refinement_levels,
            cavity_target_size=run_cfg.refinement.cavity_target_size,
            junction_target_size=run_cfg.refinement.junction_target_size,
        ),
    )
    geometry = build_mesh_geometry_3d(run_cfg.geometry)
    mesh = build_mesh_scaffold(
        geometry, resolution=resolution, refinement=run_cfg.refinement
    )
    topology = build_mesh_topology(mesh)
    return geometry, mesh, topology


def test_gradient_linear_field_on_interior_cells():
    _, mesh, topology = _small_mesh(include_cavities=False)
    centers = mesh.cell_centers
    phi = 2.0 * centers[:, 0] - 3.0 * centers[:, 1] + 0.5 * centers[:, 2]
    grad = gradient_cell_scalar(phi, topology, boundary_mode="neumann0")
    interior = topology.interior_mask
    assert np.any(interior)
    target = np.array([2.0, -3.0, 0.5], dtype=np.float64)
    err = np.max(np.abs(grad[interior] - target[None, :]))
    assert err < 1e-10


def test_laplacian_quadratic_field_on_interior_cells():
    _, mesh, topology = _small_mesh(include_cavities=False)
    centers = mesh.cell_centers
    phi = centers[:, 0] ** 2 + centers[:, 1] ** 2 + centers[:, 2] ** 2
    lap = laplacian_cell_scalar(phi, topology, boundary_mode="neumann0")
    interior = topology.interior_mask
    assert np.any(interior)
    err = np.max(np.abs(lap[interior] - 6.0))
    assert err < 1e-8


def test_divergence_linear_vector_on_interior_cells():
    _, mesh, topology = _small_mesh(include_cavities=False)
    centers = mesh.cell_centers
    vec = np.column_stack((centers[:, 0], centers[:, 1], centers[:, 2]))
    div = divergence_cell_vector(vec, topology, boundary_mode="neumann0")
    interior = topology.interior_mask
    assert np.any(interior)
    err = np.max(np.abs(div[interior] - 3.0))
    assert err < 1e-10


def test_native_scalar_diffusion_keeps_inlet_dirichlet_and_bounds():
    geometry, mesh, _ = _small_mesh(include_cavities=False)
    result = run_mesh_scalar_diffusion_prototype(
        mesh=mesh,
        geometry=geometry,
        config=MeshScalarDiffusionConfig(
            dt=2e-4,
            diffusivity=3e-10,
            steps=20,
            left_inlet_value=0.0,
            right_inlet_value=1.0,
            use_implicit=True,
            progress_every=0,
        ),
    )
    field = result["field"]
    left = result["boundary_cells"]["left_inlet"]
    right = result["boundary_cells"]["right_inlet"]
    assert np.min(field.values) >= -1e-12
    assert np.max(field.values) <= 1.0 + 1e-12
    assert np.allclose(field.values[left], 0.0, atol=1e-8)
    assert np.allclose(field.values[right], 1.0, atol=1e-8)
