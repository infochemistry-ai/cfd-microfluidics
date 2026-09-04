"""Finite-volume operators on imported tetrahedral meshes."""

from __future__ import annotations

from typing import Dict, Iterable, Literal

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh

GradientMethod = Literal["face", "least_squares"]
LaplacianMethod = Literal["tpfa", "lsq_flux"]


def _normalize_faces(indices: Iterable[int] | np.ndarray | None) -> np.ndarray:
    if indices is None:
        return np.zeros((0,), dtype=np.int64)
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.unique(arr) if arr.size else arr


def _distance_along_normal(
    delta: np.ndarray,
    normals: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    projected = np.abs(np.einsum("ij,ij->i", delta, normals))
    return np.maximum(projected, eps)


def interpolate_cell_to_face(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_face_values: Dict[int, float] | None = None,
    boundary_no_flux_faces: Iterable[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate cell-centered scalar values to faces."""

    cell_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if cell_values.shape[0] != mesh.tetrahedra.shape[0]:
        raise ValueError("interpolate_cell_to_face expects one scalar per tetra cell.")

    face_values = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    c0 = mesh.face_to_cells[:, 0]
    c1 = mesh.face_to_cells[:, 1]
    interior = c1 >= 0

    face_values[interior] = 0.5 * (
        cell_values[c0[interior]] + cell_values[c1[interior]]
    )
    face_values[~interior] = cell_values[c0[~interior]]

    no_flux_faces = _normalize_faces(boundary_no_flux_faces)
    if no_flux_faces.size > 0:
        face_values[no_flux_faces] = cell_values[c0[no_flux_faces]]

    if boundary_face_values:
        for face_idx, value in boundary_face_values.items():
            fid = int(face_idx)
            if fid < 0 or fid >= face_values.shape[0]:
                continue
            if mesh.face_to_cells[fid, 1] >= 0:
                continue
            face_values[fid] = float(value)
    return face_values


def scalar_gradient_green_gauss(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_face_values: Dict[int, float] | None = None,
    boundary_no_flux_faces: Iterable[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Legacy face-based Green-Gauss gradient reconstruction."""

    face_values = interpolate_cell_to_face(
        values,
        mesh,
        boundary_face_values=boundary_face_values,
        boundary_no_flux_faces=boundary_no_flux_faces,
    )
    gradients = np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64)
    weighted_normals = mesh.face_normals * mesh.face_areas[:, None]

    for face_idx in range(mesh.face_vertices.shape[0]):
        c0 = int(mesh.face_to_cells[face_idx, 0])
        c1 = int(mesh.face_to_cells[face_idx, 1])
        flux_vector = face_values[face_idx] * weighted_normals[face_idx]
        gradients[c0] += flux_vector
        if c1 >= 0:
            gradients[c1] -= flux_vector

    volumes = np.maximum(mesh.cell_volumes, 1e-16)
    gradients /= volumes[:, None]
    return gradients


def scalar_gradient_least_squares(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_face_values: Dict[int, float] | None = None,
    regularization: float = 1e-14,
) -> np.ndarray:
    """Least-squares cell-centered gradient reconstruction."""

    scalar = np.asarray(values, dtype=np.float64).reshape(-1)
    n_cells = mesh.tetrahedra.shape[0]
    if scalar.shape[0] != n_cells:
        raise ValueError(
            "scalar_gradient_least_squares expects one scalar per tetra cell."
        )

    mat = np.zeros((n_cells, 3, 3), dtype=np.float64)
    rhs = np.zeros((n_cells, 3), dtype=np.float64)
    equation_count = np.zeros((n_cells,), dtype=np.int32)

    interior_faces = mesh.interior_face_indices
    c0 = mesh.face_to_cells[interior_faces, 0]
    c1 = mesh.face_to_cells[interior_faces, 1]
    delta_x = mesh.cell_centers[c1] - mesh.cell_centers[c0]
    dphi = scalar[c1] - scalar[c0]
    weight = 1.0 / np.maximum(np.einsum("ij,ij->i", delta_x, delta_x), 1e-20)

    for row in range(interior_faces.shape[0]):
        i0 = int(c0[row])
        i1 = int(c1[row])
        r = delta_x[row]
        w = float(weight[row])
        outer = w * np.outer(r, r)
        vec = w * r * float(dphi[row])
        mat[i0] += outer
        mat[i1] += outer
        rhs[i0] += vec
        rhs[i1] += vec
        equation_count[i0] += 1
        equation_count[i1] += 1

    if boundary_face_values:
        for face_idx_raw, face_value in boundary_face_values.items():
            face_idx = int(face_idx_raw)
            if face_idx < 0 or face_idx >= mesh.face_vertices.shape[0]:
                continue
            owner = int(mesh.face_to_cells[face_idx, 0])
            if int(mesh.face_to_cells[face_idx, 1]) >= 0:
                continue
            r = mesh.face_centers[face_idx] - mesh.cell_centers[owner]
            norm2 = float(np.dot(r, r))
            if norm2 <= 1e-20:
                continue
            w = 1.0 / norm2
            mat[owner] += w * np.outer(r, r)
            rhs[owner] += w * r * (float(face_value) - float(scalar[owner]))
            equation_count[owner] += 1

    fallback = scalar_gradient_green_gauss(values, mesh)
    gradients = np.array(fallback, copy=True)
    eye = np.eye(3, dtype=np.float64)
    for cell_idx in range(n_cells):
        if equation_count[cell_idx] < 3:
            continue
        local = mat[cell_idx] + regularization * eye
        try:
            gradients[cell_idx] = np.linalg.solve(local, rhs[cell_idx])
        except np.linalg.LinAlgError:
            continue
    return gradients


def scalar_gradient(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    method: GradientMethod = "least_squares",
    boundary_face_values: Dict[int, float] | None = None,
    boundary_no_flux_faces: Iterable[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Unified gradient API with legacy and least-squares methods."""

    if method == "face":
        return scalar_gradient_green_gauss(
            values,
            mesh,
            boundary_face_values=boundary_face_values,
            boundary_no_flux_faces=boundary_no_flux_faces,
        )
    if method == "least_squares":
        return scalar_gradient_least_squares(
            values,
            mesh,
            boundary_face_values=boundary_face_values,
        )
    raise ValueError(f"Unsupported gradient method: {method!r}")


def divergence_from_face_flux(
    face_flux: np.ndarray, mesh: ImportedTetraMesh
) -> np.ndarray:
    """Compute cell divergence from face-normal flux values."""

    values = np.asarray(face_flux, dtype=np.float64).reshape(-1)
    if values.shape[0] != mesh.face_vertices.shape[0]:
        raise ValueError("divergence_from_face_flux expects one value per face.")

    divergence = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    face_area_flux = values * mesh.face_areas
    volumes = np.maximum(mesh.cell_volumes, 1e-16)
    for face_idx in range(mesh.face_vertices.shape[0]):
        c0 = int(mesh.face_to_cells[face_idx, 0])
        c1 = int(mesh.face_to_cells[face_idx, 1])
        signed = face_area_flux[face_idx]
        divergence[c0] += signed / volumes[c0]
        if c1 >= 0:
            divergence[c1] -= signed / volumes[c1]
    return divergence


def face_normal_velocity_from_cell_vector(
    vectors: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_face_vector_values: Dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Compute face-normal velocity by interpolation of cell-centered vectors."""

    cell_vectors = np.asarray(vectors, dtype=np.float64)
    if cell_vectors.ndim != 2 or cell_vectors.shape[1] != 3:
        raise ValueError(
            "face_normal_velocity_from_cell_vector expects shape (n_cells, 3)."
        )
    if cell_vectors.shape[0] != mesh.tetrahedra.shape[0]:
        raise ValueError(
            "face_normal_velocity_from_cell_vector expects one vector per cell."
        )

    face_velocity = np.zeros((mesh.face_vertices.shape[0], 3), dtype=np.float64)
    c0 = mesh.face_to_cells[:, 0]
    c1 = mesh.face_to_cells[:, 1]
    interior = c1 >= 0
    face_velocity[interior] = 0.5 * (
        cell_vectors[c0[interior]] + cell_vectors[c1[interior]]
    )
    face_velocity[~interior] = cell_vectors[c0[~interior]]

    if boundary_face_vector_values:
        for face_idx, value in boundary_face_vector_values.items():
            fid = int(face_idx)
            if fid < 0 or fid >= face_velocity.shape[0]:
                continue
            if mesh.face_to_cells[fid, 1] >= 0:
                continue
            face_velocity[fid] = np.asarray(value, dtype=np.float64).reshape(3)
    return np.einsum("ij,ij->i", face_velocity, mesh.face_normals)


def build_face_normal_flux_from_velocity(
    mesh: ImportedTetraMesh,
    cell_velocity: np.ndarray,
    *,
    boundary_face_velocity_overrides: Dict[int, np.ndarray] | None = None,
    left_inlet_faces: np.ndarray | None = None,
    right_inlet_faces: np.ndarray | None = None,
    outlet_faces: np.ndarray | None = None,
    wall_faces: np.ndarray | None = None,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Build face-normal velocity/flux and diagnostics for transport stepping."""

    n_cells = mesh.tetrahedra.shape[0]
    velocity = np.asarray(cell_velocity, dtype=np.float64)
    if velocity.shape != (n_cells, 3):
        raise ValueError(
            f"cell_velocity must have shape ({n_cells}, 3), got {velocity.shape}."
        )
    face_to_cells = mesh.face_to_cells
    c0 = face_to_cells[:, 0]
    c1 = face_to_cells[:, 1]
    face_velocity = np.zeros((mesh.face_vertices.shape[0], 3), dtype=np.float64)
    interior = c1 >= 0
    face_velocity[interior] = 0.5 * (velocity[c0[interior]] + velocity[c1[interior]])
    face_velocity[~interior] = velocity[c0[~interior]]

    if boundary_face_velocity_overrides:
        for face_idx, vec in boundary_face_velocity_overrides.items():
            fid = int(face_idx)
            if fid < 0 or fid >= face_velocity.shape[0]:
                continue
            if int(c1[fid]) >= 0:
                continue
            face_velocity[fid] = np.asarray(vec, dtype=np.float64).reshape(3)

    normal_velocity = np.einsum("ij,ij->i", face_velocity, mesh.face_normals)
    volumetric_flux = normal_velocity * mesh.face_areas

    left = np.asarray(
        left_inlet_faces if left_inlet_faces is not None else [], dtype=np.int64
    )
    right = np.asarray(
        right_inlet_faces if right_inlet_faces is not None else [], dtype=np.int64
    )
    outlet = np.asarray(
        outlet_faces if outlet_faces is not None else [], dtype=np.int64
    )
    wall = np.asarray(wall_faces if wall_faces is not None else [], dtype=np.int64)
    if left.size or right.size:
        inlet = np.unique(np.concatenate((left, right)))
    else:
        inlet = np.zeros((0,), dtype=np.int64)

    if inlet.size:
        inlet_flux_in = float(
            np.sum(np.maximum(-normal_velocity[inlet], 0.0) * mesh.face_areas[inlet])
        )
    else:
        inlet_flux_in = 0.0
    if outlet.size:
        outlet_flux_out = float(
            np.sum(np.maximum(normal_velocity[outlet], 0.0) * mesh.face_areas[outlet])
        )
    else:
        outlet_flux_out = 0.0
    net_boundary_flux = float(
        np.sum(volumetric_flux[np.asarray(mesh.boundary_face_indices, dtype=np.int64)])
    )
    wall_flux_max_abs = (
        float(np.max(np.abs(normal_velocity[wall]))) if wall.size else 0.0
    )

    diagnostics = {
        "face_normal_velocity_min": float(np.min(normal_velocity)),
        "face_normal_velocity_max": float(np.max(normal_velocity)),
        "volumetric_flux_min": float(np.min(volumetric_flux)),
        "volumetric_flux_max": float(np.max(volumetric_flux)),
        "total_inlet_flux_in": inlet_flux_in,
        "total_outlet_flux_out": outlet_flux_out,
        "net_boundary_flux": net_boundary_flux,
        "wall_flux_max_abs": wall_flux_max_abs,
    }
    return normal_velocity, diagnostics


def _laplacian_tpfa(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
) -> np.ndarray:
    cell_values = np.asarray(values, dtype=np.float64).reshape(-1)
    lap = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    volumes = np.maximum(mesh.cell_volumes, 1e-16)
    no_flux_set = set(np.asarray(boundary_no_flux_faces, dtype=np.int64).tolist())

    for face_idx in range(mesh.face_vertices.shape[0]):
        c0 = int(mesh.face_to_cells[face_idx, 0])
        c1 = int(mesh.face_to_cells[face_idx, 1])
        area = float(mesh.face_areas[face_idx])
        normal = mesh.face_normals[face_idx]
        if c1 >= 0:
            delta = mesh.cell_centers[c1] - mesh.cell_centers[c0]
            dist = _distance_along_normal(delta.reshape(1, 3), normal.reshape(1, 3))[0]
            flux = (area / dist) * (cell_values[c1] - cell_values[c0])
            lap[c0] += flux / volumes[c0]
            lap[c1] -= flux / volumes[c1]
            continue
        if face_idx in no_flux_set or face_idx not in boundary_dirichlet:
            continue
        delta_b = mesh.face_centers[face_idx] - mesh.cell_centers[c0]
        dist_b = _distance_along_normal(delta_b.reshape(1, 3), normal.reshape(1, 3))[0]
        flux_b = (area / dist_b) * (boundary_dirichlet[face_idx] - cell_values[c0])
        lap[c0] += flux_b / volumes[c0]
    return lap


def _laplacian_lsq_flux(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    gradient_method: GradientMethod,
    boundary_dirichlet: Dict[int, float],
    boundary_no_flux_faces: np.ndarray,
) -> np.ndarray:
    cell_values = np.asarray(values, dtype=np.float64).reshape(-1)
    lap = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    volumes = np.maximum(mesh.cell_volumes, 1e-16)
    no_flux_set = set(np.asarray(boundary_no_flux_faces, dtype=np.int64).tolist())
    gradients = scalar_gradient(
        cell_values,
        mesh,
        method=gradient_method,
        boundary_face_values=boundary_dirichlet if boundary_dirichlet else None,
        boundary_no_flux_faces=boundary_no_flux_faces,
    )

    for face_idx in range(mesh.face_vertices.shape[0]):
        c0 = int(mesh.face_to_cells[face_idx, 0])
        c1 = int(mesh.face_to_cells[face_idx, 1])
        area = float(mesh.face_areas[face_idx])
        normal = mesh.face_normals[face_idx]
        if c1 >= 0:
            s_vec = area * normal
            d_vec = mesh.cell_centers[c1] - mesh.cell_centers[c0]
            d_norm2 = max(float(np.dot(d_vec, d_vec)), 1e-20)
            alpha = float(np.dot(s_vec, d_vec)) / d_norm2
            orth_flux = alpha * (cell_values[c1] - cell_values[c0])
            correction_vec = s_vec - alpha * d_vec
            grad_face = 0.5 * (gradients[c0] + gradients[c1])
            flux = orth_flux + float(np.dot(grad_face, correction_vec))
            lap[c0] += flux / volumes[c0]
            lap[c1] -= flux / volumes[c1]
            continue
        if face_idx in no_flux_set:
            continue
        if face_idx not in boundary_dirichlet:
            continue
        delta_b = mesh.face_centers[face_idx] - mesh.cell_centers[c0]
        dist_b = _distance_along_normal(delta_b.reshape(1, 3), normal.reshape(1, 3))[0]
        flux_b = (area / dist_b) * (boundary_dirichlet[face_idx] - cell_values[c0])
        lap[c0] += flux_b / volumes[c0]
    return lap


def laplacian_scalar(
    values: np.ndarray,
    mesh: ImportedTetraMesh,
    *,
    boundary_dirichlet: Dict[int, float] | None = None,
    boundary_no_flux_faces: Iterable[int] | np.ndarray | None = None,
    method: LaplacianMethod = "tpfa",
    gradient_method: GradientMethod = "least_squares",
) -> np.ndarray:
    """Laplacian approximations on tetra cells (`tpfa` and `lsq_flux`)."""

    cell_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if cell_values.shape[0] != mesh.tetrahedra.shape[0]:
        raise ValueError("laplacian_scalar expects one scalar per tetra cell.")
    dirichlet = (
        {}
        if boundary_dirichlet is None
        else {int(k): float(v) for k, v in boundary_dirichlet.items()}
    )
    no_flux = _normalize_faces(boundary_no_flux_faces)
    if method == "tpfa":
        return _laplacian_tpfa(
            cell_values,
            mesh,
            boundary_dirichlet=dirichlet,
            boundary_no_flux_faces=no_flux,
        )
    if method == "lsq_flux":
        return _laplacian_lsq_flux(
            cell_values,
            mesh,
            gradient_method=gradient_method,
            boundary_dirichlet=dirichlet,
            boundary_no_flux_faces=no_flux,
        )
    raise ValueError(f"Unsupported laplacian method: {method!r}")


def face_normal_orientation_diagnostics(mesh: ImportedTetraMesh) -> Dict[str, float]:
    face_to_cells = mesh.face_to_cells
    interior = face_to_cells[:, 1] >= 0
    boundary = ~interior
    interior_vector = (
        mesh.cell_centers[face_to_cells[interior, 1]]
        - mesh.cell_centers[face_to_cells[interior, 0]]
    )
    interior_dots = np.einsum("ij,ij->i", mesh.face_normals[interior], interior_vector)
    boundary_dots = np.einsum(
        "ij,ij->i",
        mesh.face_normals[boundary],
        mesh.face_centers[boundary] - mesh.cell_centers[face_to_cells[boundary, 0]],
    )
    return {
        "interior_min_dot": float(np.min(interior_dots)) if interior_dots.size else 0.0,
        "interior_negative_count": int(np.count_nonzero(interior_dots < 0.0)),
        "boundary_min_dot": float(np.min(boundary_dots)) if boundary_dots.size else 0.0,
        "boundary_negative_count": int(np.count_nonzero(boundary_dots < 0.0)),
    }


def _error_stats(
    values: np.ndarray, expected: float, mask: np.ndarray
) -> Dict[str, float]:
    err = np.abs(values - expected)
    ref = err[mask]
    return {
        "max_abs_error_all": float(np.max(err)),
        "mean_abs_error_all": float(np.mean(err)),
        "l2_error_all": float(np.sqrt(np.mean(err**2))),
        "max_abs_error_interior_ref": float(np.max(ref)),
        "mean_abs_error_interior_ref": float(np.mean(ref)),
        "l2_error_interior_ref": float(np.sqrt(np.mean(ref**2))),
    }


def _gradient_error_metrics(
    gradient_constant: np.ndarray,
    gradient_linear: np.ndarray,
    *,
    linear_exact: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    const_err = np.linalg.norm(gradient_constant, axis=1)
    linear_err = np.linalg.norm(gradient_linear - linear_exact[None, :], axis=1)
    return {
        "constant_gradient_max_abs": float(np.max(const_err[mask])),
        "constant_gradient_mean_abs": float(np.mean(const_err[mask])),
        "linear_gradient_max_abs_error": float(np.max(linear_err[mask])),
        "linear_gradient_mean_abs_error": float(np.mean(linear_err[mask])),
        "linear_gradient_l2_error": float(np.sqrt(np.mean(linear_err[mask] ** 2))),
    }


def run_operator_diagnostics(
    mesh: ImportedTetraMesh,
    *,
    gradient_method: GradientMethod = "least_squares",
) -> Dict[str, object]:
    """Evaluate operator sanity metrics for gradients/divergence/laplacians."""

    centers = mesh.cell_centers
    n_cells = mesh.tetrahedra.shape[0]
    boundary_cells = np.unique(mesh.face_to_cells[mesh.boundary_face_indices, 0])
    ref_mask = np.ones(n_cells, dtype=bool)
    ref_mask[boundary_cells] = False
    if not np.any(ref_mask):
        ref_mask[:] = True

    constant = np.full(n_cells, 0.42, dtype=np.float64)
    linear_coeff = np.array([1.2, -0.7, 0.3], dtype=np.float64)
    linear = centers @ linear_coeff + 0.05
    quadratic = centers[:, 0] ** 2 + centers[:, 1] ** 2 + centers[:, 2] ** 2

    grad_face_const = scalar_gradient(constant, mesh, method="face")
    grad_face_linear = scalar_gradient(linear, mesh, method="face")
    grad_ls_const = scalar_gradient(constant, mesh, method="least_squares")
    grad_ls_linear = scalar_gradient(linear, mesh, method="least_squares")

    face_metrics = _gradient_error_metrics(
        grad_face_const,
        grad_face_linear,
        linear_exact=linear_coeff,
        mask=ref_mask,
    )
    ls_metrics = _gradient_error_metrics(
        grad_ls_const,
        grad_ls_linear,
        linear_exact=linear_coeff,
        mask=ref_mask,
    )
    selected = ls_metrics if gradient_method == "least_squares" else face_metrics

    lap_tpfa_const = laplacian_scalar(constant, mesh, method="tpfa")
    lap_tpfa_linear = laplacian_scalar(linear, mesh, method="tpfa")
    lap_tpfa_quad = laplacian_scalar(quadratic, mesh, method="tpfa")

    lap_lsq_const = laplacian_scalar(
        constant,
        mesh,
        method="lsq_flux",
        gradient_method="least_squares",
    )
    lap_lsq_linear = laplacian_scalar(
        linear,
        mesh,
        method="lsq_flux",
        gradient_method="least_squares",
    )
    lap_lsq_quad = laplacian_scalar(
        quadratic,
        mesh,
        method="lsq_flux",
        gradient_method="least_squares",
    )

    vector_linear = centers.copy()
    boundary_vectors: Dict[int, np.ndarray] = {
        int(fid): mesh.face_centers[int(fid)]
        for fid in mesh.boundary_face_indices.tolist()
    }
    face_flux = face_normal_velocity_from_cell_vector(
        vector_linear,
        mesh,
        boundary_face_vector_values=boundary_vectors,
    )
    div_linear = divergence_from_face_flux(face_flux, mesh)
    div_error = np.abs(div_linear - 3.0)
    div_ref = div_error[ref_mask]

    orient = face_normal_orientation_diagnostics(mesh)
    return {
        "selected_gradient_method": gradient_method,
        "selected_laplacian_method": "lsq_flux",
        "interior_reference_cell_count": int(np.count_nonzero(ref_mask)),
        "gradient_face": face_metrics,
        "gradient_least_squares": ls_metrics,
        "gradient_selected": selected,
        "divergence_linear": {
            "max_abs_error": float(np.max(div_ref)),
            "mean_abs_error": float(np.mean(div_ref)),
            "l2_error": float(np.sqrt(np.mean(div_ref**2))),
        },
        "laplacian_tpfa": {
            "method": "tpfa",
            "constant": _error_stats(lap_tpfa_const, 0.0, ref_mask),
            "linear": _error_stats(lap_tpfa_linear, 0.0, ref_mask),
            "quadratic": _error_stats(lap_tpfa_quad, 6.0, ref_mask),
        },
        "laplacian_lsq_flux": {
            "method": "lsq_flux",
            "constant": _error_stats(lap_lsq_const, 0.0, ref_mask),
            "linear": _error_stats(lap_lsq_linear, 0.0, ref_mask),
            "quadratic": _error_stats(lap_lsq_quad, 6.0, ref_mask),
        },
        "orientation": orient,
    }


def diagnostics_to_dict(diag: Dict[str, object]) -> Dict[str, object]:
    return diag
