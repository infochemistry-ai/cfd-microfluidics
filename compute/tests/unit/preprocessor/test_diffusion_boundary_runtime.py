from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_backend import (
    build_scalar_backend_precompute,
    laplacian_numpy,
    laplacian_torch,
)
from microfluidics.gmsh.tetra.gmsh_tetra_thermal_solver import (
    GmshTetraThermalConfig,
    run_tetra_thermal_debug,
)
from microfluidics.gmsh.tetra.gmsh_tetra_transport_solver import (
    GmshTetraTransportConfig,
    run_tetra_transport_debug,
)


def _single_tetra_mesh():
    return build_imported_tetra_mesh(
        source_path=Path("bc-runtime.msh"),
        points=np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        ),
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        boundary_triangles=np.asarray(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64
        ),
        boundary_face_tags=np.asarray([1, 2, 3, 4], dtype=np.int32),
        field_data={
            "left_inlet": np.asarray([1, 2]),
            "right_inlet": np.asarray([2, 2]),
            "outlet": np.asarray([3, 2]),
            "walls": np.asarray([4, 2]),
        },
    )


def _precompute(mesh, **boundary):
    return build_scalar_backend_precompute(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        backend="numpy",
        **boundary,
    )


def test_neumann_is_outward_normal_derivative_in_discrete_laplacian() -> None:
    mesh = _single_tetra_mesh()
    face = int(mesh.boundary_face_indices[0])
    gradient = 2.5
    precompute = _precompute(
        mesh,
        diffusion_boundary_dirichlet={},
        diffusion_boundary_neumann={face: gradient},
        diffusion_boundary_robin={},
    )

    laplacian = laplacian_numpy(
        precompute,
        np.asarray([0.4]),
        gradient_method="least_squares",
        laplacian_method="tpfa",
    )

    expected = mesh.face_areas[face] * gradient / mesh.cell_volumes[0]
    assert laplacian[0] == pytest.approx(expected)


def test_robin_uses_face_linearization_and_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    mesh = _single_tetra_mesh()
    face = int(mesh.boundary_face_indices[1])
    alpha, beta, gamma = 2.0, 3.0, 4.0
    value = np.asarray([0.25], dtype=np.float64)
    kwargs = {
        "diffusion_boundary_dirichlet": {},
        "diffusion_boundary_neumann": {},
        "diffusion_boundary_robin": {face: (alpha, beta, gamma)},
    }
    numpy_precompute = _precompute(mesh, **kwargs)
    torch_precompute = build_scalar_backend_precompute(
        mesh,
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
        backend="torch",
        torch_device="cpu",
        **kwargs,
    )

    numpy_laplacian = laplacian_numpy(
        numpy_precompute,
        value,
        gradient_method="least_squares",
        laplacian_method="tpfa",
    )
    torch_laplacian = (
        laplacian_torch(
            torch_precompute,
            torch.as_tensor(value, dtype=torch.float64),
            gradient_method="least_squares",
            laplacian_method="tpfa",
        )
        .detach()
        .cpu()
        .numpy()
    )

    stencil = numpy_precompute.diffusion_stencil
    assert stencil is not None
    distance = float(stencil.robin_distance[0])
    normal_gradient = (gamma - alpha * value[0]) / (beta + alpha * distance)
    expected = mesh.face_areas[face] * normal_gradient / mesh.cell_volumes[0]
    assert numpy_laplacian[0] == pytest.approx(expected)
    np.testing.assert_allclose(torch_laplacian, numpy_laplacian, rtol=1e-13, atol=1e-13)


def test_diffusion_boundary_kinds_cannot_overlap_on_a_face() -> None:
    mesh = _single_tetra_mesh()
    face = int(mesh.boundary_face_indices[0])

    with pytest.raises(ValueError, match="conflicting conditions"):
        _precompute(
            mesh,
            diffusion_boundary_dirichlet={face: 1.0},
            diffusion_boundary_neumann={face: 0.0},
            diffusion_boundary_robin={},
        )


def test_transport_and_thermal_solvers_consume_generic_diffusion_contract() -> None:
    mesh = _single_tetra_mesh()
    face = int(mesh.boundary_face_indices[0])
    gradient = 2.0
    diffusivity = 1e-3
    dt = 1e-5
    boundary_kwargs = {
        "diffusion_boundary_dirichlet": {},
        "diffusion_boundary_neumann": {face: gradient},
        "diffusion_boundary_robin": {},
    }
    velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    expected_increment = (
        dt * diffusivity * mesh.face_areas[face] * gradient / mesh.cell_volumes[0]
    )

    transport = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(
            steps=1,
            dt=dt,
            dt_mode="manual",
            transport_mode="advection_diffusion",
            diffusivity=diffusivity,
            clipping_enabled=False,
            progress_every=0,
        ),
        face_normal_velocity=velocity,
        flux_diagnostics={},
        diffusion_boundary_kwargs=boundary_kwargs,
    )
    assert np.asarray(transport["scalar"])[0] == pytest.approx(expected_increment)

    thermal = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=1,
            dt=dt,
            dt_mode="manual",
            thermal_diffusivity=diffusivity,
            initial_temperature=300.0,
            clipping_enabled=False,
            min_temperature=None,
            max_temperature=None,
            progress_every=0,
            backend="numpy",
        ),
        face_normal_velocity=velocity,
        diffusion_boundary_kwargs=boundary_kwargs,
    )
    assert np.asarray(thermal["temperature"])[0] == pytest.approx(
        300.0 + expected_increment
    )
