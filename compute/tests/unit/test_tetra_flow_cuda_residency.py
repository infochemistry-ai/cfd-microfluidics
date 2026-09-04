from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
import weakref

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
import microfluidics.gmsh.tetra.gmsh_tetra_flow_solver as flow_solver


@pytest.fixture(autouse=True)
def _clear_solver_caches() -> None:
    flow_solver._GEOMETRY_CACHE.clear()
    flow_solver.clear_tetra_flow_torch_caches()
    yield
    flow_solver._GEOMETRY_CACHE.clear()
    flow_solver.clear_tetra_flow_torch_caches()


def _two_cell_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 4],
            [2, 3, 4],
            [1, 4, 3],
        ],
        dtype=np.int64,
    )
    return build_imported_tetra_mesh(
        source_path=Path("two_cell_unit.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=np.full(6, 5, dtype=np.int32),
        field_data={"walls": np.asarray([5, 2], dtype=np.int32)},
    )


def _scalar_stencil_as_matrices(mesh):
    face_ids, neighbor_ids, neighbor_weights, neighbor_weight_sums = (
        flow_solver._build_face_flux_laplacian_stencil(mesh)
    )
    active = np.asarray(
        [int(fid) for fid in face_ids if neighbor_ids[int(fid)].size > 0],
        dtype=np.int64,
    )
    width = max((int(neighbor_ids[int(fid)].size) for fid in active), default=0)
    neighbors = np.repeat(active[:, None], width, axis=1)
    weights = np.zeros((active.size, width), dtype=np.float64)
    weight_sums = np.zeros((active.size,), dtype=np.float64)
    for row, face_id in enumerate(active):
        ids = neighbor_ids[int(face_id)]
        values = neighbor_weights[int(face_id)]
        neighbors[row, : ids.size] = ids
        weights[row, : values.size] = values
        weight_sums[row] = neighbor_weight_sums[int(face_id)]
    return active, neighbors, weights, weight_sums


def test_vector_stencil_matches_independent_scalar_reference() -> None:
    mesh = _two_cell_mesh()

    expected = _scalar_stencil_as_matrices(mesh)
    actual = flow_solver._build_face_flux_laplacian_vector_stencil(mesh)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(actual[3], expected[3], rtol=1e-15, atol=0.0)


def test_vector_stencil_does_not_build_python_neighbor_dictionaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _two_cell_mesh()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the vectorized path called the scalar stencil builder")

    monkeypatch.setattr(
        flow_solver, "_cached_face_flux_laplacian_stencil", fail_if_called
    )

    active, neighbors, weights, weight_sums = (
        flow_solver._build_face_flux_laplacian_vector_stencil(mesh)
    )

    assert active.size == 1
    assert neighbors.shape == weights.shape == (1, 6)
    assert weight_sums.shape == (1,)


def test_vector_stencil_returns_empty_matrices_without_interior_faces() -> None:
    mesh = _two_cell_mesh()
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64).copy()
    face_to_cells[:, 1] = -1

    active, neighbors, weights, weight_sums = (
        flow_solver._build_face_flux_laplacian_vector_stencil(
            replace(mesh, face_to_cells=face_to_cells)
        )
    )

    assert active.shape == (0,)
    assert neighbors.shape == weights.shape == (0, 0)
    assert weight_sums.shape == (0,)


def test_vector_stencil_rejects_non_tetrahedral_cell_face_shape() -> None:
    mesh = _two_cell_mesh()

    with pytest.raises(ValueError, match=r"cell_to_faces must have shape \(N, 4\)"):
        flow_solver._build_face_flux_laplacian_vector_stencil(
            replace(mesh, cell_to_faces=np.zeros((2, 3), dtype=np.int64))
        )


def test_dynamic_cuda_handoff_cache_requires_array_identity_and_device() -> None:
    array = np.arange(6, dtype=np.float64)
    tensor_marker = object()

    flow_solver._remember_torch_array(array, tensor_marker, device="cuda:0")

    assert array.flags.writeable is False
    assert flow_solver._find_torch_array(array, device="cuda:0") is tensor_marker
    with pytest.raises(ValueError, match="read-only"):
        array[0] = -1.0
    array.setflags(write=True)
    array[0] = -1.0
    assert flow_solver._find_torch_array(array, device="cuda:0") is None
    mutable_copy = array.copy()
    assert mutable_copy.flags.writeable is True
    assert flow_solver._find_torch_array(mutable_copy, device="cuda:0") is None
    assert flow_solver._find_torch_array(array, device="cuda:1") is None


def test_torch_geometry_caches_are_bounded_to_active_mesh() -> None:
    first_mesh = _two_cell_mesh()
    second_mesh = _two_cell_mesh()

    flow_solver._activate_torch_cache_context(first_mesh, device="cuda:0")
    flow_solver._TORCH_PRESSURE_CACHE[("pressure",)] = {"marker": 1}
    flow_solver._TORCH_CONVECTION_CACHE[("convection",)] = {"marker": 1}
    flow_solver._TORCH_VISCOSITY_CACHE[("viscosity",)] = {"marker": 1}

    flow_solver._activate_torch_cache_context(first_mesh, device="cuda:0")
    assert flow_solver._TORCH_PRESSURE_CACHE
    assert flow_solver._TORCH_CONVECTION_CACHE
    assert flow_solver._TORCH_VISCOSITY_CACHE

    flow_solver._activate_torch_cache_context(second_mesh, device="cuda:0")
    assert flow_solver._TORCH_PRESSURE_CACHE == {}
    assert flow_solver._TORCH_CONVECTION_CACHE == {}
    assert flow_solver._TORCH_VISCOSITY_CACHE == {}
    assert flow_solver._ACTIVE_TORCH_CACHE_CONTEXT == (
        flow_solver._mesh_geometry_cache_key(second_mesh),
        "cuda:0",
    )

    flow_solver._TORCH_PRESSURE_CACHE[("pressure",)] = {"marker": 2}
    flow_solver._activate_torch_cache_context(second_mesh, device="cuda:1")
    assert flow_solver._TORCH_PRESSURE_CACHE == {}


def test_clear_tetra_flow_torch_caches_resets_lifecycle() -> None:
    mesh = _two_cell_mesh()
    flow_solver._activate_torch_cache_context(mesh, device="cuda:0")
    flow_solver._TORCH_PRESSURE_CACHE[("pressure",)] = {"marker": 1}

    flow_solver.clear_tetra_flow_torch_caches()

    assert flow_solver._TORCH_PRESSURE_CACHE == {}
    assert flow_solver._ACTIVE_TORCH_CACHE_CONTEXT is None


def test_dynamic_cuda_handoff_cache_releases_tensor_with_numpy_peer() -> None:
    array = np.arange(3, dtype=np.float64)
    array_reference = weakref.ref(array)
    cache_key = (id(array), "cuda:0")

    flow_solver._remember_torch_array(array, object(), device="cuda:0")
    assert cache_key in flow_solver._TORCH_DYNAMIC_ARRAY_CACHE

    del array
    gc.collect()

    assert array_reference() is None
    assert cache_key not in flow_solver._TORCH_DYNAMIC_ARRAY_CACHE


def test_cached_slip_reconstruction_map_matches_numpy_solver() -> None:
    mesh = _two_cell_mesh()
    face_flux = np.linspace(-0.25, 0.5, mesh.face_vertices.shape[0])
    geometry = flow_solver._cached_slip_velocity_reconstruction_geometry(mesh)

    signed_flux = face_flux[geometry["cell_faces"]] * geometry["orientation_sign"]
    mapped_velocity = np.sum(
        geometry["reconstruction_map"] * signed_flux[:, None, :], axis=2
    )
    reference_velocity = (
        flow_solver._reconstruct_cell_velocity_from_face_flux_numpy_direct(
            mesh, face_flux, wall_velocity_boundary_mode="slip"
        )
    )

    np.testing.assert_allclose(
        mapped_velocity, reference_velocity, rtol=2e-15, atol=2e-15
    )


def test_slip_reconstruction_uses_cached_map_and_matches_direct_solve() -> None:
    mesh = _two_cell_mesh()
    face_flux = np.linspace(-0.5, 0.75, mesh.face_vertices.shape[0])

    expected = flow_solver._reconstruct_cell_velocity_from_face_flux_numpy_direct(
        mesh, face_flux, wall_velocity_boundary_mode="slip"
    )
    actual = flow_solver._reconstruct_cell_velocity_from_face_flux_numpy(
        mesh, face_flux, wall_velocity_boundary_mode="slip"
    )

    assert any(
        key[-1] == "slip_velocity_reconstruction" for key in flow_solver._GEOMETRY_CACHE
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize(
    "wall_mode",
    ["no_slip", "no_slip_tangential", "no_slip_legacy_isotropic"],
)
def test_no_slip_reconstruction_keeps_direct_solver(wall_mode: str) -> None:
    mesh = _two_cell_mesh()
    face_flux = np.linspace(-0.5, 0.75, mesh.face_vertices.shape[0])

    expected = flow_solver._reconstruct_cell_velocity_from_face_flux_numpy_direct(
        mesh,
        face_flux,
        wall_velocity_boundary_mode=wall_mode,
        wall_tangential_no_slip_strength=0.6,
    )
    actual = flow_solver._reconstruct_cell_velocity_from_face_flux_numpy(
        mesh,
        face_flux,
        wall_velocity_boundary_mode=wall_mode,
        wall_tangential_no_slip_strength=0.6,
    )

    assert not any(
        key[-1] == "slip_velocity_reconstruction" for key in flow_solver._GEOMETRY_CACHE
    )
    np.testing.assert_array_equal(actual, expected)
