from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
from microfluidics.gmsh.tetra.gmsh_tetra_mesh_loader import load_imported_tetra_mesh_npz
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    laplacian_scalar,
    run_operator_diagnostics,
    scalar_gradient,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
    GmshTetraScalarConfig,
    run_scalar_diffusion_debug,
)


def _structured_tetra_mesh(skew: bool = True) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = 3, 3, 3
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    points = np.array(
        [[x, y, z] for z in zs for y in ys for x in xs],
        dtype=np.float64,
    )

    if skew:
        rng = np.random.default_rng(0)
        for idx, (x, y, z) in enumerate(points):
            interior = (
                1e-9 < x < 1.0 - 1e-9
                and 1e-9 < y < 1.0 - 1e-9
                and 1e-9 < z < 1.0 - 1e-9
            )
            if interior:
                points[idx] += 0.3 * (rng.random(3) - 0.5)

    def node(i: int, j: int, k: int) -> int:
        return k * (nx * ny) + j * nx + i

    tets: list[list[int]] = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v000 = node(i, j, k)
                v100 = node(i + 1, j, k)
                v010 = node(i, j + 1, k)
                v110 = node(i + 1, j + 1, k)
                v001 = node(i, j, k + 1)
                v101 = node(i + 1, j, k + 1)
                v011 = node(i, j + 1, k + 1)
                v111 = node(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                    ]
                )
    return points, np.asarray(tets, dtype=np.int64)


def _extract_boundary_triangles(
    points: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in tetrahedra:
        faces = (
            (int(tet[1]), int(tet[2]), int(tet[3])),
            (int(tet[0]), int(tet[3]), int(tet[2])),
            (int(tet[0]), int(tet[1]), int(tet[3])),
            (int(tet[0]), int(tet[2]), int(tet[1])),
        )
        for tri in faces:
            face_map.setdefault(tuple(sorted(tri)), []).append(tri)

    boundary_triangles: list[list[int]] = []
    boundary_tags: list[int] = []
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    eps = 1e-12
    for entries in face_map.values():
        if len(entries) != 1:
            continue
        tri = entries[0]
        center = np.mean(points[np.asarray(tri, dtype=np.int64)], axis=0)
        x, y, _ = center.tolist()
        if y <= y_min + eps:
            tag = 1 if x <= 0.5 else 2
        elif y >= y_max - eps:
            tag = 3
        else:
            tag = 4
        boundary_triangles.append([tri[0], tri[1], tri[2]])
        boundary_tags.append(tag)

    field_data = {
        "left_inlet": np.asarray([1, 2], dtype=np.int32),
        "right_inlet": np.asarray([2, 2], dtype=np.int32),
        "outlet": np.asarray([3, 2], dtype=np.int32),
        "walls": np.asarray([4, 2], dtype=np.int32),
    }
    return (
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(boundary_tags, dtype=np.int32),
        field_data,
    )


def _build_synthetic_imported_mesh(skew: bool = True):
    points, tetrahedra = _structured_tetra_mesh(skew=skew)
    boundary_triangles, boundary_tags, field_data = _extract_boundary_triangles(
        points, tetrahedra
    )
    return build_imported_tetra_mesh(
        source_path=Path("synthetic.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_tags,
        field_data=field_data,
    )


def test_gmsh_tetra_loader_roundtrip(tmp_path: Path) -> None:
    mesh = _build_synthetic_imported_mesh()
    npz_path = tmp_path / "synthetic_imported_mesh.npz"
    np.savez_compressed(
        npz_path,
        points=mesh.points,
        tetrahedra=mesh.tetrahedra,
        boundary_triangles=mesh.boundary_triangles,
        boundary_face_tags=mesh.boundary_face_tags,
        volume_tag_per_cell=np.full(mesh.tetrahedra.shape[0], 7, dtype=np.int32),
        cell_centers=mesh.cell_centers,
        cell_volumes=mesh.cell_volumes,
        face_vertices=mesh.face_vertices,
        face_centers=mesh.face_centers,
        face_areas=mesh.face_areas,
        face_normals=mesh.face_normals,
        face_to_cells=mesh.face_to_cells,
        cell_to_faces=mesh.cell_to_faces,
        boundary_tag_per_face=mesh.boundary_tag_per_face,
        interior_face_indices=mesh.interior_face_indices,
        boundary_face_indices=mesh.boundary_face_indices,
        inlet_faces=mesh.inlet_faces,
        outlet_faces=mesh.outlet_faces,
        wall_faces=mesh.wall_faces,
        unresolved_faces=mesh.boundary_unresolved_faces,
        boundary_face_names_json=np.asarray(
            json.dumps({str(k): v for k, v in mesh.boundary_face_names.items()})
        ),
        physical_groups_json=np.asarray(
            json.dumps(
                {
                    name: [int(values[0]), int(values[1])]
                    for name, values in mesh.physical_groups.items()
                }
            )
        ),
        diagnostics_json=np.asarray(json.dumps({})),
        source_path=np.asarray(str(tmp_path / "source.msh")),
        case_config_json=np.asarray(
            json.dumps({"schema_version": "case_config_v1", "case_id": "roundtrip"})
        ),
        preprocessor_metadata_json=np.asarray(json.dumps({"quality_gate": "passed"})),
    )
    loaded = load_imported_tetra_mesh_npz(npz_path)
    assert loaded.points.shape == mesh.points.shape
    assert loaded.tetrahedra.shape == mesh.tetrahedra.shape
    assert loaded.inlet_faces.size > 0
    assert loaded.outlet_faces.size > 0
    assert loaded.wall_faces.size > 0
    assert set(loaded.volume_tag_per_cell.tolist()) == {7}
    assert loaded.case_config["case_id"] == "roundtrip"
    assert loaded.preprocessor_metadata["quality_gate"] == "passed"
    assert loaded.source_path == tmp_path / "source.msh"


def test_constant_gradient_near_zero_for_both_methods() -> None:
    mesh = _build_synthetic_imported_mesh()
    constant = np.full(mesh.tetrahedra.shape[0], 0.5, dtype=np.float64)
    grad_face = scalar_gradient(constant, mesh, method="face")
    grad_ls = scalar_gradient(constant, mesh, method="least_squares")
    assert float(np.max(np.linalg.norm(grad_face, axis=1))) < 1e-10
    assert float(np.max(np.linalg.norm(grad_ls, axis=1))) < 1e-10


def test_least_squares_linear_gradient_better_than_face() -> None:
    mesh = _build_synthetic_imported_mesh(skew=True)
    coeff = np.array([1.2, -0.7, 0.3], dtype=np.float64)
    field = mesh.cell_centers @ coeff + 0.05
    grad_face = scalar_gradient(field, mesh, method="face")
    grad_ls = scalar_gradient(field, mesh, method="least_squares")
    err_face = np.linalg.norm(grad_face - coeff[None, :], axis=1)
    err_ls = np.linalg.norm(grad_ls - coeff[None, :], axis=1)
    assert float(np.mean(err_ls)) < float(np.mean(err_face))


def test_operator_diagnostics_reports_both_methods() -> None:
    mesh = _build_synthetic_imported_mesh(skew=True)
    diag = run_operator_diagnostics(mesh, gradient_method="least_squares")
    assert "gradient_face" in diag
    assert "gradient_least_squares" in diag
    assert "gradient_selected" in diag
    assert diag["selected_gradient_method"] == "least_squares"
    ls_mean = diag["gradient_least_squares"]["linear_gradient_mean_abs_error"]
    face_mean = diag["gradient_face"]["linear_gradient_mean_abs_error"]
    assert float(ls_mean) < float(face_mean)
    assert "laplacian_tpfa" in diag
    assert "laplacian_lsq_flux" in diag


def test_laplacian_constant_is_near_zero() -> None:
    mesh = _build_synthetic_imported_mesh()
    values = np.full(mesh.tetrahedra.shape[0], 0.333, dtype=np.float64)
    lap_tpfa = laplacian_scalar(values, mesh, method="tpfa")
    lap_lsq = laplacian_scalar(values, mesh, method="lsq_flux")
    assert float(np.max(np.abs(lap_tpfa))) < 1e-10
    assert float(np.max(np.abs(lap_lsq))) < 1e-10


def test_laplacian_linear_near_zero_for_both_methods() -> None:
    mesh = _build_synthetic_imported_mesh(skew=True)
    coeff = np.array([0.9, -0.6, 0.4], dtype=np.float64)
    linear = mesh.cell_centers @ coeff + 0.11
    lap_tpfa = laplacian_scalar(linear, mesh, method="tpfa")
    lap_lsq = laplacian_scalar(linear, mesh, method="lsq_flux")
    assert float(np.mean(np.abs(lap_lsq))) <= float(np.mean(np.abs(lap_tpfa)))


def test_lsq_flux_quadratic_better_than_tpfa_on_skew_mesh() -> None:
    mesh = _build_synthetic_imported_mesh(skew=True)
    diag = run_operator_diagnostics(mesh, gradient_method="least_squares")
    tpfa = diag["laplacian_tpfa"]["quadratic"]["mean_abs_error_interior_ref"]
    lsq = diag["laplacian_lsq_flux"]["quadratic"]["mean_abs_error_interior_ref"]
    assert float(lsq) < float(tpfa)


def test_diffusion_only_preserves_bounds_and_dirichlet_inlets() -> None:
    mesh = _build_synthetic_imported_mesh(skew=True)
    cfg = GmshTetraScalarConfig(
        dt=2e-3,
        steps=12,
        diffusivity=2e-4,
        left_inlet_value=0.0,
        right_inlet_value=1.0,
        progress_every=0,
        gradient_method="least_squares",
        laplacian_method="lsq_flux",
        backend="numpy",
    )
    result = run_scalar_diffusion_debug(mesh, cfg)
    scalar = np.asarray(result["scalar"], dtype=np.float64)
    assert np.all(np.isfinite(scalar))
    assert float(np.min(scalar)) >= -1e-12
    assert float(np.max(scalar)) <= 1.0 + 1e-12
    assert result["boundary_setup"]["left_inlet_cells"] > 0
    assert result["boundary_setup"]["right_inlet_cells"] > 0
    history = result["history"]
    assert len(history) == cfg.steps
    assert history[-1]["left_inlet_max"] <= 1e-8
    assert history[-1]["right_inlet_min"] >= 1.0 - 1e-8


def test_backend_selection_auto_without_torch() -> None:
    backend = select_backend(
        "auto",
        torch_available_override=False,
        torch_cuda_available_override=False,
    )
    assert backend.selected_backend == "numpy"
    assert backend.device == "cpu"
