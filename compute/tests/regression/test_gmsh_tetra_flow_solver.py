from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.gmsh.run_gmsh_tetra_flow_debug as flow_debug_module
import microfluidics.gmsh.tetra.gmsh_tetra_flow_solver as flow_solver_module
import microfluidics.gmsh.tetra.pressure_determinism_diagnostics as pressure_diagnostics_module
from experiments.gmsh.run_gmsh_tetra_flow_debug import (
    DEFAULT_TETRA_FLOW_DEBUG_PRESSURE_SOLVER,
    DEFAULT_STARTUP_BOOTSTRAP_MAX_STEPS,
    FLOW_RESUME_MANIFEST_SCHEMA_VERSION,
    FLOW_RESUME_SOLVER_CONTRACT_VERSION,
    FLOW_RESUME_STATE_FILENAMES,
    STARTUP_BOOTSTRAP_LEGACY_SEARCH_BUDGET,
    STARTUP_BOOTSTRAP_QUALIFICATION_TAIL,
    _build_expected_artifacts_map,
    _best_effort_environment_metadata,
    _best_effort_git_metadata,
    _build_flow_resume_manifest,
    _clamp_flow_dt_to_stop_time,
    _build_projection_acceptance_step_record,
    _collect_boundary_flux_policy,
    _collect_stabilization_audit,
    _collect_warning_aggregation,
    _convective_history_stats,
    _export_flow_coupling_bundle,
    _evaluate_convective_readiness,
    _evaluate_ns_coupling_readiness,
    _build_viscous_predictor_audit_from_history,
    _evaluate_flow_progression_acceptance,
    _evaluate_stokes_ready_for_advection,
    _face_group_codes,
    _finalize_flow_resume_manifest,
    _load_flow_resume_state,
    _mesh_npz_fingerprint,
    _next_snapshot_time,
    _evaluate_viscous_progression_acceptance,
    _parse_flow_modes,
    _parse_snapshot_steps,
    _parse_convective_stabilization_modes,
    _parse_flow_dt_mode,
    _projection_boundary_contract_runtime_payload,
    _resolve_flow_config_for_postsolve_audit,
    _projection_failed_criteria,
    _projection_strict_diagnostic_criteria_not_met,
    _pressure_iteration_telemetry,
    _pressure_matvec_telemetry,
    _record_cuda_synchronization,
    _parse_viscous_predictor_modes,
    _resolve_viscous_predictor_mode,
    _projection_acceptance,
    _run_startup_bootstrap,
    _save_velocity_vectors_xy_grid_binned,
    _seal_flow_resume_manifest,
    _source_runtime_fingerprint,
    _summarize_startup_bootstrap,
    _sync_flow_diagnostics_with_final_artifacts,
    _timing_mode_synchronizes_components,
    _timing_stats,
    _validate_flow_resume_manifest,
    _velocity_region_audit,
    _wall_shear_stress_metrics,
)
from microfluidics.gmsh.gmsh_mesh_import import (
    build_imported_tetra_mesh,
    import_gmsh_tetra_mesh,
)
from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
    TetraFlowConfig,
    TetraFlowState,
    _apply_face_flux_boundary_conditions_inplace,
    _build_pressure_system_coefficients,
    _build_wall_flux_stokes_resistance,
    _build_wall_tangential_no_slip_operator,
    _compute_cell_flux_sum,
    _locally_conserve_interior_face_flux_cell_sums,
    _matvec_pressure_numpy,
    _pressure_matrix_explicit_audit,
    _pressure_matrixfree_vs_explicit_audit,
    _pressure_operator_spd_audit,
    _solve_pressure_cg_numpy,
    _solve_pressure_reference_explicit,
    apply_tetra_convective_predictor,
    apply_tetra_flow_boundary_conditions,
    apply_tetra_stokes_viscous_predictor,
    compute_tetra_convective_cfl_rate,
    compute_tetra_flux_divergence,
    initialize_tetra_flow_state,
    solve_tetra_pressure_projection,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import resolve_inlet_face_groups


def _resume_manifest(
    *,
    mesh_sha256: str = "a" * 64,
    flow_config: dict | None = None,
) -> dict:
    return _seal_flow_resume_manifest(
        {
            "schema_version": FLOW_RESUME_MANIFEST_SCHEMA_VERSION,
            "solver_contract_version": FLOW_RESUME_SOLVER_CONTRACT_VERSION,
            "mesh_sha256": mesh_sha256,
            "input_mesh_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "runtime_identifier": "example-runtime@sha256:" + "1" * 64,
            "request_fingerprint": "d" * 64,
            "flow_config": flow_config or {"pressure_solver": "pcg_diag"},
        }
    )


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_wall_shear_metric_uses_tangential_owner_velocity_and_wall_distance() -> None:
    mesh = SimpleNamespace(
        wall_faces=np.array([0, 1], dtype=np.int64),
        face_to_cells=np.array([[0, -1], [1, -1]], dtype=np.int64),
        face_centers=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        cell_centers=np.zeros((2, 3)),
        face_normals=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        face_areas=np.array([1.0, 3.0]),
    )
    velocity = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])

    result = _wall_shear_stress_metrics(
        mesh,
        velocity,
        dynamic_viscosity=1e-3,
    )

    assert result["area_weighted_mean_pa"] == pytest.approx(0.002375)
    assert result["max_pa"] == pytest.approx(0.0025)


def _structured_tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = 3, 3, 3
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)

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


def _build_synthetic_mesh():
    points, tetrahedra = _structured_tetra_mesh()
    btri, btags, field_data = _extract_boundary_triangles(points, tetrahedra)
    return build_imported_tetra_mesh(
        source_path=Path("flow_synth.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=btri,
        boundary_face_tags=btags,
        field_data=field_data,
    )


def _build_two_cell_mesh():
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
    boundary_face_tags = np.asarray([5, 5, 5, 5, 5, 5], dtype=np.int32)
    field_data = {"walls": np.asarray([5, 2], dtype=np.int32)}
    return build_imported_tetra_mesh(
        source_path=Path("two_cell_flow.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _build_one_cell_outlet_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    boundary_face_tags = np.asarray([3, 4, 4, 4], dtype=np.int32)
    field_data = {
        "outlet": np.asarray([3, 2], dtype=np.int32),
        "walls": np.asarray([4, 2], dtype=np.int32),
    }
    return build_imported_tetra_mesh(
        source_path=Path("one_cell_outlet.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _wall_owner_cells(mesh) -> np.ndarray:
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall_faces.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.asarray(mesh.face_to_cells[wall_faces, 0], dtype=np.int64))


def test_interior_face_flux_conservation_equal_opposite() -> None:
    mesh = _build_two_cell_mesh()
    flux = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    interior_face = int(mesh.interior_face_indices[0])
    flux[interior_face] = 0.123
    diag = compute_tetra_flux_divergence(mesh, flux)
    div = np.asarray(diag["divergence"], dtype=np.float64)
    total = float(np.sum(div * mesh.cell_volumes))
    assert abs(total) <= 1e-12


def test_balanced_internal_flux_zero_global_divergence() -> None:
    mesh = _build_two_cell_mesh()
    flux = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    flux[int(mesh.interior_face_indices[0])] = 1e-3
    diag = compute_tetra_flux_divergence(mesh, flux)
    assert abs(float(np.sum(diag["divergence"] * mesh.cell_volumes))) <= 1e-12
    assert abs(float(diag["net_boundary_flux"])) <= 1e-12


def test_wall_faces_preserve_zero_flux() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(inlet_speed=0.15)
    rng = np.random.default_rng(0)
    state = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=rng.standard_normal(mesh.face_vertices.shape[0]),
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )
    out = apply_tetra_flow_boundary_conditions(mesh, state, cfg)
    if mesh.wall_faces.size:
        assert float(np.max(np.abs(out.face_flux[mesh.wall_faces]))) <= 1e-12


def test_viscous_boundary_contract_can_preserve_outlet_flux() -> None:
    mesh = _build_synthetic_mesh()
    inlet = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    assert outlet_faces.size > 0

    base = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    base[outlet_faces] = 1.234e-9
    preserve = base.copy()
    match = base.copy()

    _apply_face_flux_boundary_conditions_inplace(
        mesh,
        preserve,
        inlet_speed=0.15,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        outlet_contract_mode="preserve",
    )
    _apply_face_flux_boundary_conditions_inplace(
        mesh,
        match,
        inlet_speed=0.15,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        outlet_contract_mode="match_inlet",
    )

    np.testing.assert_allclose(preserve[outlet_faces], base[outlet_faces])
    assert not np.allclose(match[outlet_faces], base[outlet_faces])


def test_pressure_projection_outlet_contract_default_matches_legacy_mode() -> None:
    mesh = _build_synthetic_mesh()
    common = {
        "inlet_speed": 0.15,
        "projection_dt": 5e-4,
        "max_pressure_iterations": 80,
        "pressure_relative_tolerance": 1e-3,
        "pressure_solver": "pcg_diag",
        "enable_sign_comparison": False,
        "backend": "numpy",
    }
    cfg_default = TetraFlowConfig(**common)  # type: ignore[arg-type]
    cfg_legacy = TetraFlowConfig(
        **common,  # type: ignore[arg-type]
        pressure_projection_outlet_contract_mode="match_inlet",
    )
    face_flux = np.linspace(
        -0.07,
        0.11,
        mesh.face_vertices.shape[0],
        dtype=np.float64,
    )
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=face_flux,
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )

    out_default = solve_tetra_pressure_projection(mesh, state0, cfg_default)
    out_legacy = solve_tetra_pressure_projection(mesh, state0, cfg_legacy)

    np.testing.assert_allclose(out_default.face_flux, out_legacy.face_flux)
    np.testing.assert_allclose(out_default.cell_velocity, out_legacy.cell_velocity)
    np.testing.assert_allclose(out_default.pressure, out_legacy.pressure)
    default_star = np.asarray(
        out_default.diagnostics["face_flux_primary"]["face_flux_star"],
        dtype=np.float64,
    )
    legacy_star = np.asarray(
        out_legacy.diagnostics["face_flux_primary"]["face_flux_star"],
        dtype=np.float64,
    )
    np.testing.assert_allclose(default_star, legacy_star)
    inlet = resolve_inlet_face_groups(mesh)
    inlet_faces = np.concatenate(
        (
            np.asarray(inlet["left_faces"], dtype=np.int64),
            np.asarray(inlet["right_faces"], dtype=np.int64),
        )
    )
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    inlet_flux = cfg_default.inlet_speed * float(np.sum(mesh.face_areas[inlet_faces]))
    legacy_outlet_speed = inlet_flux / float(np.sum(mesh.face_areas[outlet_faces]))
    np.testing.assert_allclose(
        default_star[outlet_faces],
        legacy_outlet_speed * mesh.face_areas[outlet_faces],
    )
    assert not np.allclose(default_star[outlet_faces], face_flux[outlet_faces])
    assert (
        out_default.diagnostics["projection"][
            "pressure_projection_outlet_contract_mode"
        ]
        == "match_inlet"
    )


def test_pressure_projection_preserves_outlet_star_and_leaves_correction_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    inlet = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    assert outlet_faces.size > 0

    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=20,
        pressure_tolerance=1e-8,
        backend="numpy",
        enable_sign_comparison=False,
        pressure_projection_outlet_contract_mode="preserve",
        outlet_projection_mode="outlet_pressure_dirichlet",
    )
    face_flux = np.linspace(
        -0.2,
        0.3,
        mesh.face_vertices.shape[0],
        dtype=np.float64,
    )
    predicted_outlet = np.linspace(
        0.01,
        0.02,
        outlet_faces.size,
        dtype=np.float64,
    )
    face_flux[outlet_faces] = predicted_outlet
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=face_flux,
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )

    def _fake_solve_pressure_system(
        coeff: object,
        *,
        rhs: np.ndarray,
        p0: np.ndarray,
        config: TetraFlowConfig,
        backend: object,
    ) -> tuple[np.ndarray, dict[str, object], bool]:
        del coeff, rhs, config, backend
        return (
            np.zeros_like(np.asarray(p0, dtype=np.float64)),
            {
                "pressure_solved": True,
                "stopping_reason": "converged_relative_l2",
                "residual_ratio_to_rhs_l2": 0.0,
                "residual_ratio_to_rhs_max": 0.0,
            },
            False,
        )

    def _fake_pressure_face_gradient_flux(
        *args: object, **kwargs: object
    ) -> np.ndarray:
        del args, kwargs
        grad = np.full(mesh.face_vertices.shape[0], 0.25, dtype=np.float64)
        grad[outlet_faces] = 0.5
        return grad

    monkeypatch.setattr(
        flow_solver_module,
        "_solve_pressure_system",
        _fake_solve_pressure_system,
    )
    monkeypatch.setattr(
        flow_solver_module,
        "_pressure_face_gradient_flux",
        _fake_pressure_face_gradient_flux,
    )

    out = solve_tetra_pressure_projection(mesh, state0, cfg)
    primary = dict(out.diagnostics["face_flux_primary"])
    star = np.asarray(primary["face_flux_star"], dtype=np.float64)
    raw = np.asarray(primary["correction_flux_raw_pre_constraint"], dtype=np.float64)
    constrained = np.asarray(
        primary["correction_flux_constrained_pre_limiter"],
        dtype=np.float64,
    )
    effective = np.asarray(
        primary["correction_flux_effective_post_outlet_policy"],
        dtype=np.float64,
    )

    np.testing.assert_allclose(star[outlet_faces], predicted_outlet)
    np.testing.assert_allclose(raw[outlet_faces], -0.5)
    np.testing.assert_allclose(constrained[outlet_faces], raw[outlet_faces])
    np.testing.assert_allclose(effective[outlet_faces], raw[outlet_faces])
    np.testing.assert_allclose(
        np.asarray(out.face_flux)[outlet_faces],
        predicted_outlet + raw[outlet_faces],
    )
    for face_ids in (left_faces, right_faces, wall_faces):
        if not face_ids.size:
            continue
        np.testing.assert_allclose(constrained[face_ids], 0.0, atol=1e-12)
        np.testing.assert_allclose(effective[face_ids], 0.0, atol=1e-12)
        np.testing.assert_allclose(out.face_flux[face_ids], star[face_ids], atol=1e-12)
    if left_faces.size:
        np.testing.assert_allclose(
            star[left_faces],
            -cfg.inlet_speed * mesh.face_areas[left_faces],
        )
    if right_faces.size:
        np.testing.assert_allclose(
            star[right_faces],
            -cfg.inlet_speed * mesh.face_areas[right_faces],
        )
    if wall_faces.size:
        np.testing.assert_allclose(star[wall_faces], 0.0, atol=1e-12)
    pinned = out.diagnostics["projection_pinned_face_constraints"]
    expected_pinned = int(left_faces.size + right_faces.size + wall_faces.size)
    assert pinned["raw_pre_constraint"]["pinned_face_count"] == expected_pinned
    assert (
        out.diagnostics["projection"]["pressure_projection_outlet_contract_mode"]
        == "preserve"
    )


def _coherent_no_slip_test_config() -> TetraFlowConfig:
    return TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=100,
        pressure_relative_tolerance=1e-4,
        projection_dt=5e-2,
        enable_sign_comparison=False,
        kinematic_viscosity=1e-2,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="no_slip",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
        projection_cell_velocity_update_mode="momentum_pressure_corrected",
    )


def test_projection_cell_velocity_update_default_keeps_legacy_path_bitwise() -> None:
    mesh = _build_synthetic_mesh()
    common = {
        "pressure_solver": "pcg_diag",
        "max_pressure_iterations": 80,
        "pressure_relative_tolerance": 1e-3,
        "projection_dt": 5e-4,
        "enable_sign_comparison": False,
        "wall_velocity_boundary_mode": "slip",
    }
    cfg_default = TetraFlowConfig(**common)  # type: ignore[arg-type]
    cfg_legacy = TetraFlowConfig(
        **common,  # type: ignore[arg-type]
        projection_cell_velocity_update_mode="legacy_reconstruct",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg_default)

    pred_default = apply_tetra_stokes_viscous_predictor(
        mesh,
        state0,
        cfg_default,
        flow_dt=5e-4,
    )
    pred_legacy = apply_tetra_stokes_viscous_predictor(
        mesh,
        state0,
        cfg_legacy,
        flow_dt=5e-4,
    )
    np.testing.assert_array_equal(pred_default.face_flux, pred_legacy.face_flux)
    np.testing.assert_array_equal(
        pred_default.cell_velocity,
        pred_legacy.cell_velocity,
    )

    out_default = solve_tetra_pressure_projection(mesh, pred_default, cfg_default)
    out_legacy = solve_tetra_pressure_projection(mesh, pred_legacy, cfg_legacy)
    np.testing.assert_array_equal(out_default.face_flux, out_legacy.face_flux)
    np.testing.assert_array_equal(out_default.cell_velocity, out_legacy.cell_velocity)
    np.testing.assert_array_equal(out_default.pressure, out_legacy.pressure)
    expected_velocity = (
        flow_solver_module._reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            out_default.face_flux,
            wall_velocity_boundary_mode="slip",
        )
    )
    np.testing.assert_array_equal(out_default.cell_velocity, expected_velocity)


def test_auto_numerical_profile_is_physical_only_for_tangential_no_slip() -> None:
    slip = flow_solver_module.resolve_tetra_flow_numerical_profile(
        TetraFlowConfig(wall_velocity_boundary_mode="slip")
    )
    assert slip.viscous_predictor_outlet_contract_mode == "match_inlet"
    assert slip.pressure_projection_outlet_contract_mode == "match_inlet"
    assert slip.projection_cell_velocity_update_mode == "legacy_reconstruct"
    assert slip.pressure_nonorthogonal_correction_mode == "none"
    assert slip.viscous_nonorthogonal_correction_mode == "none"

    no_slip = flow_solver_module.resolve_tetra_flow_numerical_profile(
        TetraFlowConfig(wall_velocity_boundary_mode="no_slip")
    )
    assert no_slip.viscous_predictor_outlet_contract_mode == "preserve"
    assert no_slip.pressure_projection_outlet_contract_mode == "preserve"
    assert no_slip.projection_cell_velocity_update_mode == "momentum_pressure_corrected"
    assert no_slip.pressure_nonorthogonal_correction_mode == "deferred_lsq"
    assert no_slip.pressure_nonorthogonal_correction_sweeps == 4
    assert no_slip.viscous_nonorthogonal_correction_mode == "deferred_lsq"

    explicit_legacy = flow_solver_module.resolve_tetra_flow_numerical_profile(
        TetraFlowConfig(
            wall_velocity_boundary_mode="no_slip",
            viscous_predictor_outlet_contract_mode="match_inlet",
            pressure_projection_outlet_contract_mode="match_inlet",
            projection_cell_velocity_update_mode="legacy_reconstruct",
            pressure_nonorthogonal_correction_mode="none",
            viscous_nonorthogonal_correction_mode="none",
        )
    )
    assert explicit_legacy.viscous_predictor_outlet_contract_mode == "match_inlet"
    assert explicit_legacy.pressure_projection_outlet_contract_mode == "match_inlet"
    assert explicit_legacy.projection_cell_velocity_update_mode == "legacy_reconstruct"
    assert explicit_legacy.pressure_nonorthogonal_correction_mode == "none"
    assert explicit_legacy.viscous_nonorthogonal_correction_mode == "none"


def test_boundary_flux_helper_rejects_unresolved_auto_contract() -> None:
    mesh = _build_synthetic_mesh()
    with pytest.raises(ValueError, match="must be resolved"):
        flow_solver_module._apply_face_flux_boundary_conditions_inplace(
            mesh,
            np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
            inlet_speed=0.15,
            left_inlet_faces=np.asarray(mesh.inlet_faces, dtype=np.int64),
            right_inlet_faces=np.zeros(0, dtype=np.int64),
            outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
            wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
            outlet_contract_mode="auto",
        )


def test_debug_postsolve_audit_resolves_no_slip_auto_profile() -> None:
    effective = _resolve_flow_config_for_postsolve_audit(
        TetraFlowConfig(wall_velocity_boundary_mode="no_slip")
    )

    assert effective.viscous_predictor_outlet_contract_mode == "preserve"
    assert effective.pressure_projection_outlet_contract_mode == "preserve"
    assert effective.projection_cell_velocity_update_mode == (
        "momentum_pressure_corrected"
    )
    assert effective.pressure_nonorthogonal_correction_mode == "deferred_lsq"
    assert effective.viscous_nonorthogonal_correction_mode == "deferred_lsq"


def test_slip_auto_profile_is_multistep_bitwise_legacy() -> None:
    mesh = _build_synthetic_mesh()
    common = {
        "wall_velocity_boundary_mode": "slip",
        "viscous_predictor_mode": (
            "explicit_cell_velocity_laplacian_substepped_conservative"
        ),
        "pressure_solver": "pcg_diag",
        "max_pressure_iterations": 100,
        "pressure_relative_tolerance": 1e-5,
        "projection_dt": 5e-4,
        "enable_sign_comparison": False,
    }
    cfg_auto = TetraFlowConfig(**common)  # type: ignore[arg-type]
    cfg_legacy = TetraFlowConfig(
        **common,  # type: ignore[arg-type]
        viscous_predictor_outlet_contract_mode="match_inlet",
        pressure_projection_outlet_contract_mode="match_inlet",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        pressure_nonorthogonal_correction_mode="none",
        viscous_nonorthogonal_correction_mode="none",
    )
    auto_state = initialize_tetra_flow_state(mesh, cfg_auto)
    legacy_state = initialize_tetra_flow_state(mesh, cfg_legacy)

    for _ in range(5):
        auto_state = apply_tetra_stokes_viscous_predictor(
            mesh, auto_state, cfg_auto, flow_dt=cfg_auto.projection_dt
        )
        legacy_state = apply_tetra_stokes_viscous_predictor(
            mesh, legacy_state, cfg_legacy, flow_dt=cfg_legacy.projection_dt
        )
        auto_state = solve_tetra_pressure_projection(mesh, auto_state, cfg_auto)
        legacy_state = solve_tetra_pressure_projection(mesh, legacy_state, cfg_legacy)
        np.testing.assert_array_equal(auto_state.face_flux, legacy_state.face_flux)
        np.testing.assert_array_equal(
            auto_state.cell_velocity, legacy_state.cell_velocity
        )
        np.testing.assert_array_equal(auto_state.pressure, legacy_state.pressure)


def test_coherent_predictor_interpolates_complete_momentum_state_to_interior_flux() -> (
    None
):
    mesh = _build_synthetic_mesh()
    cfg = _coherent_no_slip_test_config()
    state0 = initialize_tetra_flow_state(mesh, cfg)
    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        state0,
        cfg,
        flow_dt=cfg.projection_dt,
    )
    interpolated = flow_solver_module._face_flux_from_cell_velocity_numpy(
        mesh,
        pred.cell_velocity,
    )
    interior = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64) >= 0
    np.testing.assert_array_equal(pred.face_flux[interior], interpolated[interior])

    diag = pred.diagnostics["viscous_predictor"]
    assert diag["momentum_predictor_preserved_for_projection"] is True
    assert diag["coherent_momentum_face_flux_interpolation_enabled"] is True
    assert diag["local_conservative_flux_correction_enabled"] is False
    assert (
        diag["wall_tangential_cell_velocity_flux_delta_conservative_applied"] is False
    )
    assert diag["face_flux_predictor_source"] == (
        "interpolation_of_complete_momentum_predictor"
    )


def test_momentum_projection_applies_effective_pressure_increment_to_u_star() -> None:
    mesh = _build_synthetic_mesh()
    cfg = _coherent_no_slip_test_config()
    state0 = initialize_tetra_flow_state(mesh, cfg)
    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        state0,
        cfg,
        flow_dt=cfg.projection_dt,
    )
    out = solve_tetra_pressure_projection(mesh, pred, cfg)
    correction_flux = np.asarray(
        out.diagnostics["face_flux_primary"]["correction_flux"],
        dtype=np.float64,
    )
    assert float(np.max(np.abs(correction_flux))) > 0.0
    pressure_increment = (
        flow_solver_module._reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            correction_flux,
            wall_velocity_boundary_mode="slip",
        )
    )
    np.testing.assert_allclose(
        out.cell_velocity,
        pred.cell_velocity + pressure_increment,
        rtol=1e-13,
        atol=1e-15,
    )

    update = out.diagnostics["projection_velocity_update"]
    assert update["mode"] == "momentum_pressure_corrected"
    assert update["momentum_state_preserved"] is True
    assert "interior_faces" in update["pressure_increment_face_flux_consistency"]
    assert "boundary_faces" in update["pressure_increment_face_flux_consistency"]


def test_zero_pressure_correction_preserves_wall_damped_momentum_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = replace(
        _coherent_no_slip_test_config(),
        backend="numpy",
        device="cpu",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        state0,
        cfg,
        flow_dt=cfg.projection_dt,
    )
    wall_cells = _wall_owner_cells(mesh)
    assert float(
        np.mean(np.linalg.norm(pred.cell_velocity[wall_cells], axis=1))
    ) < float(np.mean(np.linalg.norm(state0.cell_velocity[wall_cells], axis=1)))

    def zero_pressure_solver(
        coeff: dict[str, np.ndarray],
        *,
        rhs: np.ndarray,
        p0: np.ndarray,
        config: TetraFlowConfig,
        backend: object,
    ) -> tuple[np.ndarray, dict[str, object], bool]:
        del coeff, rhs, config, backend
        return np.zeros_like(p0), {"pressure_solved": True}, False

    monkeypatch.setattr(
        flow_solver_module,
        "_solve_pressure_system",
        zero_pressure_solver,
    )
    out = solve_tetra_pressure_projection(mesh, pred, cfg)
    np.testing.assert_array_equal(out.cell_velocity, pred.cell_velocity)
    np.testing.assert_array_equal(
        out.diagnostics["face_flux_primary"]["correction_flux"],
        np.zeros_like(pred.face_flux),
    )
    update = out.diagnostics["projection_velocity_update"]
    assert update["zero_effective_correction"] is True
    assert update["actual_cell_velocity_change"]["max_abs"] == 0.0


def test_wall_velocity_boundary_mode_defaults_to_slip_behavior() -> None:
    mesh = _build_synthetic_mesh()
    cfg_default = TetraFlowConfig(inlet_speed=0.15)
    cfg_slip = TetraFlowConfig(inlet_speed=0.15, wall_velocity_boundary_mode="slip")
    state_default = initialize_tetra_flow_state(mesh, cfg_default)
    state_slip = initialize_tetra_flow_state(mesh, cfg_slip)
    np.testing.assert_allclose(state_default.face_flux, state_slip.face_flux)
    np.testing.assert_allclose(state_default.cell_velocity, state_slip.cell_velocity)


def test_no_slip_mode_reduces_wall_adjacent_velocity_after_projection() -> None:
    mesh = _build_synthetic_mesh()
    wall_cells = _wall_owner_cells(mesh)
    assert wall_cells.size > 0
    cfg_slip = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="slip",
    )
    cfg_no_slip = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="no_slip",
        wall_tangential_shear_face_flux_enabled=False,
    )

    slip_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_slip),
        cfg_slip,
        flow_dt=5e-4,
    )
    no_slip_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_no_slip),
        cfg_no_slip,
        flow_dt=5e-4,
    )
    assert (
        slip_pred.diagnostics["viscous_predictor"][
            "wall_tangential_shear_face_flux_enabled"
        ]
        is False
    )
    assert (
        no_slip_pred.diagnostics["viscous_predictor"][
            "wall_tangential_cell_velocity_momentum_enabled"
        ]
        is True
    )
    assert (
        no_slip_pred.diagnostics["viscous_predictor"][
            "wall_tangential_operator_active_cells"
        ]
        > 0
    )
    assert (
        no_slip_pred.diagnostics["viscous_predictor"][
            "wall_tangential_operator_effective_nu_dt_max_abs"
        ]
        > 0.0
    )
    slip_proj = solve_tetra_pressure_projection(mesh, slip_pred, cfg_slip)
    no_slip_proj = solve_tetra_pressure_projection(mesh, no_slip_pred, cfg_no_slip)
    slip_update = slip_proj.diagnostics["projection_velocity_update"]
    no_slip_update = no_slip_proj.diagnostics["projection_velocity_update"]
    assert slip_update["cached_slip_reconstruction_used"] is True
    assert slip_update["slip_dense_reconstruction_solves_avoided"] == 2
    assert no_slip_update["cached_slip_reconstruction_used"] is False
    assert no_slip_update["slip_dense_reconstruction_solves_avoided"] == 0

    slip_wall_speed = np.linalg.norm(slip_proj.cell_velocity[wall_cells], axis=1)
    no_slip_wall_speed = np.linalg.norm(no_slip_proj.cell_velocity[wall_cells], axis=1)
    assert float(np.mean(no_slip_wall_speed)) <= float(np.mean(slip_wall_speed)) + 1e-9
    if mesh.wall_faces.size:
        assert float(np.max(np.abs(no_slip_proj.face_flux[mesh.wall_faces]))) <= 1e-12


def test_no_slip_tangential_mode_reduces_wall_adjacent_velocity_after_projection() -> (
    None
):
    mesh = _build_synthetic_mesh()
    wall_cells = _wall_owner_cells(mesh)
    assert wall_cells.size > 0
    cfg_slip = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="slip",
    )
    cfg_tangential = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="no_slip_tangential",
        wall_tangential_shear_face_flux_enabled=False,
    )

    slip_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_slip),
        cfg_slip,
        flow_dt=5e-4,
    )
    tangential_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_tangential),
        cfg_tangential,
        flow_dt=5e-4,
    )
    slip_proj = solve_tetra_pressure_projection(mesh, slip_pred, cfg_slip)
    tangential_proj = solve_tetra_pressure_projection(
        mesh, tangential_pred, cfg_tangential
    )

    slip_wall_speed = np.linalg.norm(slip_proj.cell_velocity[wall_cells], axis=1)
    tangential_wall_speed = np.linalg.norm(
        tangential_proj.cell_velocity[wall_cells], axis=1
    )
    assert (
        float(np.mean(tangential_wall_speed)) <= float(np.mean(slip_wall_speed)) + 1e-9
    )
    if mesh.wall_faces.size:
        assert (
            float(np.max(np.abs(tangential_proj.face_flux[mesh.wall_faces]))) <= 1e-12
        )


def test_no_slip_mode_matches_tangential_wall_implementation() -> None:
    mesh = _build_synthetic_mesh()
    cfg_no_slip = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
    )
    cfg_tangential = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip_tangential",
    )

    no_slip_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_no_slip),
        cfg_no_slip,
        flow_dt=5e-4,
    )
    tangential_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_tangential),
        cfg_tangential,
        flow_dt=5e-4,
    )

    np.testing.assert_allclose(no_slip_pred.face_flux, tangential_pred.face_flux)
    np.testing.assert_allclose(
        no_slip_pred.cell_velocity, tangential_pred.cell_velocity
    )
    assert (
        no_slip_pred.diagnostics["viscous_predictor"][
            "wall_velocity_boundary_implementation"
        ]
        == "tangential_zero_velocity"
    )


def test_wall_tangential_no_slip_strength_preserves_no_slip_alias() -> None:
    mesh = _build_synthetic_mesh()
    cfg_no_slip = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        wall_tangential_no_slip_strength=2.0,
    )
    cfg_tangential = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip_tangential",
        wall_tangential_no_slip_strength=2.0,
    )

    no_slip_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_no_slip),
        cfg_no_slip,
        flow_dt=5e-4,
    )
    tangential_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_tangential),
        cfg_tangential,
        flow_dt=5e-4,
    )

    np.testing.assert_allclose(no_slip_pred.face_flux, tangential_pred.face_flux)
    assert (
        no_slip_pred.diagnostics["viscous_predictor"][
            "wall_tangential_no_slip_strength"
        ]
        == 2.0
    )


def test_wall_flux_stokes_resistance_is_opt_in_and_reports_diagnostics() -> None:
    mesh = _build_synthetic_mesh()
    wall_cells = _wall_owner_cells(mesh)
    cfg_base = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        wall_flux_stokes_resistance_enabled=False,
    )
    cfg_exp = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        wall_flux_stokes_resistance_enabled=True,
        wall_flux_stokes_resistance_strength=1.0,
    )

    base_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_base),
        cfg_base,
        flow_dt=5e-4,
    )
    exp_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_exp),
        cfg_exp,
        flow_dt=5e-4,
    )

    base_wall_speed = np.linalg.norm(base_pred.cell_velocity[wall_cells], axis=1)
    exp_wall_speed = np.linalg.norm(exp_pred.cell_velocity[wall_cells], axis=1)
    assert float(np.mean(exp_wall_speed)) <= float(np.mean(base_wall_speed))
    assert (
        exp_pred.diagnostics["viscous_predictor"]["wall_flux_stokes_resistance_enabled"]
        is True
    )
    assert (
        exp_pred.diagnostics["viscous_predictor"][
            "wall_flux_stokes_resistance_active_faces"
        ]
        > 0
    )


def test_wall_flux_stokes_resistance_operator_uses_global_flux_signs() -> None:
    mesh = _build_synthetic_mesh()
    bundle = _build_wall_flux_stokes_resistance(mesh, mesh.wall_faces)
    cell_ids = np.asarray(bundle["cell_ids"], dtype=np.int64)
    assert cell_ids.size > 0

    wall_operator = _build_wall_tangential_no_slip_operator(mesh, mesh.wall_faces)
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.face_areas, dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)

    checked = False
    for row_idx, cell_idx_raw in enumerate(cell_ids.tolist()):
        cell_idx = int(cell_idx_raw)
        faces = np.asarray(bundle["cell_face_ids"][row_idx], dtype=np.int64)
        signs = np.ones((faces.size,), dtype=np.float64)
        oriented_normals = np.zeros((faces.size, 3), dtype=np.float64)
        has_neighbor_owned_face = False
        for local_idx, fid_raw in enumerate(faces.tolist()):
            fid = int(fid_raw)
            owner = int(face_to_cells[fid, 0])
            neigh = int(face_to_cells[fid, 1])
            if owner == cell_idx:
                oriented_normals[local_idx] = face_normals[fid]
                signs[local_idx] = 1.0
            elif neigh == cell_idx:
                oriented_normals[local_idx] = -face_normals[fid]
                signs[local_idx] = -1.0
                has_neighbor_owned_face = True
            else:
                oriented_normals[local_idx] = 0.0
                signs[local_idx] = 0.0
        if not has_neighbor_owned_face:
            continue

        mat = 1e-14 * eye3
        for local_idx, fid_raw in enumerate(faces.tolist()):
            n_vec = oriented_normals[local_idx]
            mat += max(float(face_areas[int(fid_raw)]), 1e-16) * np.outer(n_vec, n_vec)
        recon_map = np.linalg.solve(mat, oriented_normals.T)
        expected_local = recon_map.T @ wall_operator[cell_idx] @ recon_map
        expected_local = 0.5 * (expected_local + expected_local.T)
        expected_global = signs[:, None] * expected_local * signs[None, :]
        variable_mask = np.asarray(bundle["variable_face_mask"][row_idx], dtype=bool)
        variable_ids = np.flatnonzero(variable_mask)
        if variable_ids.size > 1:
            constraint = signs[variable_ids]
            denom = float(np.dot(constraint, constraint))
            if denom > 0.0:
                proj = (
                    np.eye(variable_ids.size, dtype=np.float64)
                    - np.outer(constraint, constraint) / denom
                )
                fixed_ids = np.flatnonzero(~variable_mask)
                op_vv = expected_global[np.ix_(variable_ids, variable_ids)]
                expected_global[np.ix_(variable_ids, variable_ids)] = (
                    proj @ op_vv @ proj
                )
                if fixed_ids.size:
                    op_vf = expected_global[np.ix_(variable_ids, fixed_ids)]
                    op_fv = expected_global[np.ix_(fixed_ids, variable_ids)]
                    expected_global[np.ix_(variable_ids, fixed_ids)] = proj @ op_vf
                    expected_global[np.ix_(fixed_ids, variable_ids)] = op_fv @ proj

        np.testing.assert_allclose(
            np.asarray(bundle["operator"][row_idx], dtype=np.float64),
            expected_global + 1e-14 * np.eye(faces.size, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        op_vv = np.asarray(bundle["operator"][row_idx], dtype=np.float64)[
            np.ix_(variable_ids, variable_ids)
        ]
        constraint = signs[variable_ids]
        np.testing.assert_allclose(op_vv @ constraint, 0.0, atol=1e-10)
        checked = True
        break
    assert checked


def test_wall_tangential_no_slip_shear_can_be_split_for_diagnostics() -> None:
    mesh = _build_synthetic_mesh()
    cfg_shear_only = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        wall_tangential_shear_face_flux_enabled=True,
    )
    cfg_shear_disabled = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        wall_tangential_shear_face_flux_enabled=False,
    )

    shear_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_shear_only),
        cfg_shear_only,
        flow_dt=5e-4,
    )
    disabled_pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_shear_disabled),
        cfg_shear_disabled,
        flow_dt=5e-4,
    )

    shear_diag = shear_pred.diagnostics["viscous_predictor"]
    disabled_diag = disabled_pred.diagnostics["viscous_predictor"]
    assert shear_diag["wall_tangential_shear_face_flux_requested"] is True
    assert shear_diag["wall_tangential_shear_face_flux_enabled"] is True
    assert disabled_diag["wall_tangential_shear_face_flux_requested"] is False
    assert disabled_diag["wall_tangential_shear_face_flux_enabled"] is False
    assert shear_diag["wall_tangential_operator_active_cells"] > 0
    assert disabled_diag["wall_tangential_operator_active_cells"] == 0


def test_wall_tangential_cell_velocity_momentum_uses_wall_operator() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
    )

    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg),
        cfg,
        flow_dt=5e-4,
    )
    diag = pred.diagnostics["viscous_predictor"]
    assert diag["wall_tangential_shear_face_flux_requested"] is False
    assert diag["wall_tangential_cell_velocity_momentum_enabled"] is True
    assert diag["wall_tangential_operator_active_cells"] > 0
    assert diag["wall_tangential_operator_effective_nu_dt_max_abs"] > 0.0


def test_wall_tangential_stability_uses_spectral_radius_not_trace() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        viscous_predictor_outlet_contract_mode="match_inlet",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        viscous_nonorthogonal_correction_mode="none",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
    )

    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg),
        cfg,
        flow_dt=5e-4,
    )
    diag = pred.diagnostics["viscous_predictor"]

    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    c0 = face_to_cells[:, 0]
    c1 = face_to_cells[:, 1]
    interior = c1 >= 0
    own = c0[interior]
    nei = c1[interior]
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    row_coef = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
    if own.size:
        d_ij = np.maximum(np.linalg.norm(centers[nei] - centers[own], axis=1), 1e-12)
        w = np.asarray(mesh.face_areas[interior], dtype=np.float64) / d_ij
        np.add.at(row_coef, own, w / vol[own])
        np.add.at(row_coef, nei, w / vol[nei])
    wall_operator = _build_wall_tangential_no_slip_operator(mesh, mesh.wall_faces)
    row_coef += np.maximum(np.linalg.eigvalsh(wall_operator)[:, -1], 0.0)
    expected_metric = float(cfg.kinematic_viscosity * 5e-4 * np.max(row_coef))

    assert diag["viscous_stability_metric"] == pytest.approx(expected_metric)


def test_local_conservative_flux_correction_reduces_cell_sum_residual() -> None:
    mesh = _build_synthetic_mesh()
    rng = np.random.default_rng(12)
    base_flux = rng.standard_normal(mesh.face_vertices.shape[0]) * 1e-6
    perturbed = base_flux.copy()
    interior = np.asarray(mesh.interior_face_indices, dtype=np.int64)
    perturbed[interior] += rng.standard_normal(interior.size) * 1e-5
    target = _compute_cell_flux_sum(mesh, base_flux)

    corrected, diag = _locally_conserve_interior_face_flux_cell_sums(
        mesh,
        perturbed,
        target_cell_flux_sum=target,
        iterations=12,
    )

    boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    before_residual = target - _compute_cell_flux_sum(mesh, perturbed)
    after_residual = target - _compute_cell_flux_sum(mesh, corrected)
    assert float(np.linalg.norm(after_residual)) < float(
        np.linalg.norm(before_residual)
    )
    assert diag["residual_l2_after"] < diag["residual_l2_before"]
    np.testing.assert_allclose(corrected[boundary], perturbed[boundary])


def test_conservative_cell_velocity_predictor_reports_local_flux_correction() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-5,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        viscous_nonorthogonal_correction_mode="none",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
    )

    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg),
        cfg,
        flow_dt=5e-4,
    )
    diag = pred.diagnostics["viscous_predictor"]

    assert diag["local_conservative_flux_correction_enabled"] is True
    assert diag["local_conservative_flux_correction_iterations"] > 0
    assert (
        diag["local_conservative_flux_correction_residual_l2_after"]
        <= diag["local_conservative_flux_correction_residual_l2_before"]
    )
    if mesh.wall_faces.size:
        assert float(np.max(np.abs(pred.face_flux[mesh.wall_faces]))) <= 1e-12


def test_conservative_cell_velocity_predictor_preserves_flux_baseline_without_delta() -> (
    None
):
    mesh = _build_synthetic_mesh()
    rng = np.random.default_rng(123)
    face_flux = rng.standard_normal(mesh.face_vertices.shape[0]) * 1e-6
    cell_velocity = rng.standard_normal((mesh.tetrahedra.shape[0], 3)) * 0.01
    state = TetraFlowState(
        cell_velocity=cell_velocity,
        face_flux=face_flux.copy(),
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )
    cfg = TetraFlowConfig(
        kinematic_viscosity=0.0,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        viscous_nonorthogonal_correction_mode="none",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
    )

    pred = apply_tetra_stokes_viscous_predictor(mesh, state, cfg, flow_dt=5e-4)
    arrays = pred.diagnostics["viscous_predictor"]["arrays"]

    np.testing.assert_allclose(
        arrays["face_flux_after_predictor_before_contract"],
        face_flux,
    )


def test_conservative_no_slip_wall_momentum_preserves_cell_flux_sum_for_all_cells() -> (
    None
):
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-5,
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
        viscous_predictor_outlet_contract_mode="match_inlet",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="legacy_reconstruct",
        viscous_nonorthogonal_correction_mode="none",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
    )

    state0 = initialize_tetra_flow_state(mesh, cfg)
    pred = apply_tetra_stokes_viscous_predictor(mesh, state0, cfg, flow_dt=5e-4)
    diag = pred.diagnostics["viscous_predictor"]

    assert (
        diag["local_conservative_flux_correction_target_mode"] == "preserve_all_cells"
    )
    assert diag["wall_tangential_cell_velocity_flux_delta_conservative_applied"] is True
    assert (
        diag["wall_tangential_cell_velocity_flux_delta_residual_l2_after"]
        <= diag["wall_tangential_cell_velocity_flux_delta_residual_l2_before"]
    )

    target = _compute_cell_flux_sum(mesh, state0.face_flux)
    corrected = _compute_cell_flux_sum(mesh, pred.face_flux)
    residual = target - corrected
    wall_cells = _wall_owner_cells(mesh)

    assert float(np.max(np.abs(residual))) <= 5e-10
    if wall_cells.size:
        assert float(np.max(np.abs(residual[wall_cells]))) <= 5e-10


def test_legacy_isotropic_no_slip_mode_remains_available() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip_legacy_isotropic",
    )

    pred = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg),
        cfg,
        flow_dt=5e-4,
    )

    assert (
        pred.diagnostics["viscous_predictor"]["wall_velocity_boundary_implementation"]
        == "legacy_isotropic_velocity_sink"
    )


def test_face_flux_laplacian_vectorized_matches_scalar_fallback() -> None:
    mesh = _build_synthetic_mesh()
    cfg_vector = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        viscous_face_flux_laplacian_vectorized=True,
    )
    cfg_scalar = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        wall_velocity_boundary_mode="no_slip",
        viscous_face_flux_laplacian_vectorized=False,
    )

    pred_vector = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_vector),
        cfg_vector,
        flow_dt=5e-4,
    )
    pred_scalar = apply_tetra_stokes_viscous_predictor(
        mesh,
        initialize_tetra_flow_state(mesh, cfg_scalar),
        cfg_scalar,
        flow_dt=5e-4,
    )

    np.testing.assert_allclose(pred_vector.face_flux, pred_scalar.face_flux)
    np.testing.assert_allclose(pred_vector.cell_velocity, pred_scalar.cell_velocity)
    assert (
        pred_vector.diagnostics["viscous_predictor"][
            "viscous_face_flux_laplacian_vectorized"
        ]
        is True
    )
    assert (
        pred_scalar.diagnostics["viscous_predictor"][
            "viscous_face_flux_laplacian_vectorized"
        ]
        is False
    )


def test_inlet_flux_sign_is_inflow_negative() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(inlet_speed=0.15)
    state = initialize_tetra_flow_state(mesh, cfg)
    inlet = resolve_inlet_face_groups(mesh)
    faces = np.unique(
        np.concatenate(
            (
                np.asarray(inlet["left_faces"], dtype=np.int64),
                np.asarray(inlet["right_faces"], dtype=np.int64),
            )
        )
    )
    assert faces.size > 0
    assert bool(np.all(state.face_flux[faces] <= 0.0))


def test_tetra_flow_step_does_not_preapply_boundary_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    rng = np.random.default_rng(19)
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=rng.standard_normal(mesh.face_vertices.shape[0]),
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )

    def _unexpected_preapply(*args: object, **kwargs: object) -> TetraFlowState:
        raise AssertionError(
            "tetra_flow_step() should not pre-apply boundary conditions"
        )

    monkeypatch.setattr(
        flow_solver_module,
        "apply_tetra_flow_boundary_conditions",
        _unexpected_preapply,
    )
    out = flow_solver_module.tetra_flow_step(mesh, state0, cfg)
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall.size:
        assert (
            float(np.max(np.abs(np.asarray(out.face_flux, dtype=np.float64)[wall])))
            <= 1e-12
        )


def test_tetra_flow_step_preserves_boundary_contract_on_public_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    inlet = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    def _unexpected_preapply(*args: object, **kwargs: object) -> TetraFlowState:
        raise AssertionError(
            "tetra_flow_step() should preserve the boundary contract without pre-apply"
        )

    monkeypatch.setattr(
        flow_solver_module,
        "apply_tetra_flow_boundary_conditions",
        _unexpected_preapply,
    )

    out = flow_solver_module.tetra_flow_step(mesh, state0, cfg)
    primary = dict(out.diagnostics["face_flux_primary"])
    contract = dict(out.diagnostics["projection_correction_boundary_contract"])
    star = np.asarray(primary["face_flux_star"], dtype=np.float64)
    constrained = np.asarray(
        primary["correction_flux_constrained_pre_limiter"],
        dtype=np.float64,
    )
    constrained_post_limiter = np.asarray(
        primary["correction_flux_constrained_post_limiter_pre_outlet_policy"],
        dtype=np.float64,
    )
    effective = np.asarray(
        primary["correction_flux_effective_post_outlet_policy"],
        dtype=np.float64,
    )
    final_flux = np.asarray(out.face_flux, dtype=np.float64)

    np.testing.assert_allclose(star, np.asarray(state0.face_flux, dtype=np.float64))
    for face_ids in (left_faces, right_faces, wall_faces):
        if face_ids.size == 0:
            continue
        np.testing.assert_allclose(final_flux[face_ids], star[face_ids], atol=1e-12)
        assert float(np.max(np.abs(constrained[face_ids]))) <= 1e-12
        assert float(np.max(np.abs(constrained_post_limiter[face_ids]))) <= 1e-12
        assert float(np.max(np.abs(effective[face_ids]))) <= 1e-12
    if left_faces.size or right_faces.size:
        inlet_faces = np.unique(np.concatenate((left_faces, right_faces)))
        assert bool(np.all(final_flux[inlet_faces] <= 0.0))
    if wall_faces.size:
        assert float(np.max(np.abs(final_flux[wall_faces]))) <= 1e-12
    assert (
        contract["constrained_post_limiter_pre_outlet_policy"]["walls"]["max_abs"]
        <= 1e-12
    )
    assert contract["effective_post_outlet_policy"]["walls"]["max_abs"] <= 1e-12


def test_projection_reduces_divergence_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=200,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    rng = np.random.default_rng(7)
    face_flux = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    interior = np.asarray(mesh.interior_face_indices, dtype=np.int64)
    face_flux[interior] = 1e-3 * rng.standard_normal(interior.size)
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=face_flux,
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    proj = state1.diagnostics["projection"]
    assert float(proj["final_divergence_max_abs"]) < float(
        proj["initial_divergence_max_abs"]
    )


def test_projection_pins_inlet_and_wall_faces_in_correction_space() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    rng = np.random.default_rng(23)
    face_flux = rng.standard_normal(mesh.face_vertices.shape[0])
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=face_flux,
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )

    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    inlet = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    proj = dict(state1.diagnostics["projection"])
    primary = dict(state1.diagnostics["face_flux_primary"])
    pinned = dict(state1.diagnostics["projection_pinned_face_constraints"])
    limited = np.asarray(
        primary["correction_flux_constrained_post_limiter_pre_outlet_policy"],
        dtype=np.float64,
    )
    star = np.asarray(primary["face_flux_star"], dtype=np.float64)
    final_flux = np.asarray(state1.face_flux, dtype=np.float64)

    expected_pinned_count = int(left_faces.size + right_faces.size + wall_faces.size)
    assert (
        int(pinned["raw_pre_constraint"]["pinned_face_count"]) == expected_pinned_count
    )
    assert (
        int(pinned["limiter_output_pre_reconstraint"]["pinned_face_count"])
        == expected_pinned_count
    )
    if left_faces.size:
        assert float(np.max(np.abs(limited[left_faces]))) <= 1e-12
        np.testing.assert_allclose(final_flux[left_faces], star[left_faces], atol=1e-12)
    if right_faces.size:
        assert float(np.max(np.abs(limited[right_faces]))) <= 1e-12
        np.testing.assert_allclose(
            final_flux[right_faces], star[right_faces], atol=1e-12
        )
    if wall_faces.size:
        assert float(np.max(np.abs(limited[wall_faces]))) <= 1e-12
        np.testing.assert_allclose(final_flux[wall_faces], star[wall_faces], atol=1e-12)
    np.testing.assert_allclose(
        final_flux,
        np.asarray(primary["face_flux_corrected"], dtype=np.float64),
        atol=1e-12,
    )
    assert float(proj["wall_flux_max_abs_after"]) <= 1e-12


def test_projection_boundary_diagnostics_preserve_raw_then_repin_limiter_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=20,
        pressure_tolerance=1e-8,
        backend="numpy",
        outlet_projection_mode="outlet_mass_balance_rescale",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    inlet = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)

    def _fake_solve_pressure_system(
        coeff: object,
        *,
        rhs: np.ndarray,
        p0: np.ndarray,
        config: TetraFlowConfig,
        backend: object,
    ) -> tuple[np.ndarray, dict[str, object], bool]:
        del coeff, config, backend
        return (
            np.zeros_like(np.asarray(p0, dtype=np.float64)),
            {
                "pressure_solved": True,
                "stopping_reason": "converged_relative_l2",
                "residual_ratio_to_rhs_l2": 0.0,
                "residual_ratio_to_rhs_max": 0.0,
            },
            False,
        )

    def _fake_pressure_face_gradient_flux(
        *args: object, **kwargs: object
    ) -> np.ndarray:
        del args, kwargs
        grad = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
        if left_faces.size:
            grad[left_faces] = 1.25
        if right_faces.size:
            grad[right_faces] = -2.5
        if wall_faces.size:
            grad[wall_faces] = 3.75
        if outlet_faces.size:
            grad[outlet_faces] = 0.5
        interior = np.asarray(mesh.interior_face_indices, dtype=np.int64)
        if interior.size:
            grad[interior[: min(4, interior.size)]] = np.array(
                [0.2, -0.3, 0.4, -0.1][: min(4, interior.size)],
                dtype=np.float64,
            )
        return grad

    def _fake_limiter(**kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        corr = np.asarray(kwargs["correction_flux_raw"], dtype=np.float64).copy()
        if left_faces.size:
            corr[left_faces] = 11.0
        if right_faces.size:
            corr[right_faces] = -7.0
        if wall_faces.size:
            corr[wall_faces] = 5.0
        return (
            corr,
            {
                "projection_correction_limit_mode": "synthetic_test_override",
                "number_of_limited_cells": 0,
                "number_of_limited_faces": int(
                    left_faces.size + right_faces.size + wall_faces.size
                ),
                "limited_cell_indices": np.zeros((0,), dtype=np.int64),
                "limited_face_indices": np.zeros((0,), dtype=np.int64),
                "conservation_audit": {},
            },
        )

    monkeypatch.setattr(
        flow_solver_module,
        "_solve_pressure_system",
        _fake_solve_pressure_system,
    )
    monkeypatch.setattr(
        flow_solver_module,
        "_pressure_face_gradient_flux",
        _fake_pressure_face_gradient_flux,
    )
    monkeypatch.setattr(
        flow_solver_module,
        "_apply_projection_correction_limiter",
        _fake_limiter,
    )

    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    primary = dict(state1.diagnostics["face_flux_primary"])
    pinned = dict(state1.diagnostics["projection_pinned_face_constraints"])
    contract = dict(state1.diagnostics["projection_correction_boundary_contract"])

    raw = np.asarray(primary["correction_flux_raw_pre_constraint"], dtype=np.float64)
    constrained = np.asarray(
        primary["correction_flux_constrained_pre_limiter"], dtype=np.float64
    )
    limiter_output = np.asarray(
        primary["correction_flux_limiter_output_pre_reconstraint"], dtype=np.float64
    )
    constrained_post_limiter = np.asarray(
        primary["correction_flux_constrained_post_limiter_pre_outlet_policy"],
        dtype=np.float64,
    )
    effective = np.asarray(
        primary["correction_flux_effective_post_outlet_policy"], dtype=np.float64
    )

    if left_faces.size:
        assert float(np.max(np.abs(raw[left_faces]))) > 0.0
    if right_faces.size:
        assert float(np.max(np.abs(raw[right_faces]))) > 0.0
    if wall_faces.size:
        assert float(np.max(np.abs(raw[wall_faces]))) > 0.0

    for face_ids in (left_faces, right_faces, wall_faces):
        if face_ids.size == 0:
            continue
        assert float(np.max(np.abs(limiter_output[face_ids]))) > 0.0
        assert float(np.max(np.abs(constrained[face_ids]))) <= 1e-12
        assert float(np.max(np.abs(constrained_post_limiter[face_ids]))) <= 1e-12
        assert float(np.max(np.abs(effective[face_ids]))) <= 1e-12

    assert contract["raw_pre_constraint"]["left_inlet"]["nonzero_count"] == int(
        left_faces.size
    )
    assert contract["raw_pre_constraint"]["right_inlet"]["nonzero_count"] == int(
        right_faces.size
    )
    assert contract["raw_pre_constraint"]["walls"]["nonzero_count"] == int(
        wall_faces.size
    )
    assert contract["constrained_pre_limiter"]["left_inlet"]["max_abs"] <= 1e-12
    assert contract["constrained_pre_limiter"]["right_inlet"]["max_abs"] <= 1e-12
    assert contract["constrained_pre_limiter"]["walls"]["max_abs"] <= 1e-12
    assert contract["limiter_output_pre_reconstraint"]["left_inlet"][
        "nonzero_count"
    ] == int(left_faces.size)
    assert contract["limiter_output_pre_reconstraint"]["right_inlet"][
        "nonzero_count"
    ] == int(right_faces.size)
    assert contract["limiter_output_pre_reconstraint"]["walls"]["nonzero_count"] == int(
        wall_faces.size
    )
    assert (
        contract["constrained_post_limiter_pre_outlet_policy"]["left_inlet"]["max_abs"]
        <= 1e-12
    )
    assert (
        contract["constrained_post_limiter_pre_outlet_policy"]["right_inlet"]["max_abs"]
        <= 1e-12
    )
    assert (
        contract["constrained_post_limiter_pre_outlet_policy"]["walls"]["max_abs"]
        <= 1e-12
    )
    assert contract["effective_post_outlet_policy"]["left_inlet"]["max_abs"] <= 1e-12
    assert contract["effective_post_outlet_policy"]["right_inlet"]["max_abs"] <= 1e-12
    assert contract["effective_post_outlet_policy"]["walls"]["max_abs"] <= 1e-12
    assert pinned["raw_pre_constraint"]["pinned_face_nonzero_before_count"] > 0
    assert (
        pinned["limiter_output_pre_reconstraint"]["pinned_face_nonzero_before_count"]
        > 0
    )
    assert pinned["limiter_reintroduced_pinned_faces"] is True
    assert (
        contract["limiter_output_pre_reconstraint"]["outlet"]["nonzero_count"]
        == (
            contract["constrained_post_limiter_pre_outlet_policy"]["outlet"][
                "nonzero_count"
            ]
        )
    )
    assert (
        contract["limiter_output_pre_reconstraint"]["interior"]["nonzero_count"]
        == (
            contract["constrained_post_limiter_pre_outlet_policy"]["interior"][
                "nonzero_count"
            ]
        )
    )


def test_projection_sign_comparison_present() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_sign="minus",
        enable_sign_comparison=True,
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    comp = state1.diagnostics.get("projection_sign_comparison_fixed_rhs", {})
    assert "minus" in comp
    assert "plus" in comp
    assert "recommended_projection_sign" in comp


def test_cg_solver_path_reduces_divergence() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-10,
        pressure_relative_tolerance=1e-4,
        pressure_solver="cg",
        backend="numpy",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    proj = state1.diagnostics["projection"]
    assert float(proj["final_divergence_max_abs"]) < float(
        proj["initial_divergence_max_abs"]
    )
    press = state1.diagnostics["pressure"]
    assert str(press["stopping_reason"]) in {
        "converged_relative_l2",
        "converged_relative_max",
        "converged_absolute",
        "stagnated",
        "breakdown_near_converged",
        "breakdown_not_converged",
        "max_iterations",
        "nan_or_inf",
    }
    assert "pressure_solved" in press


def test_pcg_diag_solver_path_reduces_divergence() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-10,
        pressure_relative_tolerance=1e-4,
        pressure_solver="pcg_diag",
        backend="numpy",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    proj = state1.diagnostics["projection"]
    press = state1.diagnostics["pressure"]
    assert float(proj["final_divergence_max_abs"]) < float(
        proj["initial_divergence_max_abs"]
    )
    assert str(press.get("preconditioner", "")) == "diag"
    assert "pressure_solved" in press
    assert float(press["solve_wall_seconds"]) >= 0.0
    assert float(press["solve_wall_seconds_per_iteration"]) >= 0.0


def test_amg_pcg_solver_path_reduces_divergence_and_reuses_hierarchy() -> None:
    flow_solver_module._AMG_PRESSURE_CACHE.clear()
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-10,
        pressure_relative_tolerance=1e-6,
        pressure_solver="amg_pcg",
        backend="numpy",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    state2 = solve_tetra_pressure_projection(mesh, state0, cfg)

    projection = state1.diagnostics["projection"]
    pressure1 = state1.diagnostics["pressure"]
    pressure2 = state2.diagnostics["pressure"]
    assert float(projection["final_divergence_max_abs"]) < float(
        projection["initial_divergence_max_abs"]
    )
    assert pressure1["preconditioner"] == "smoothed_aggregation_amg"
    assert pressure1["pressure_solved"] is True
    assert pressure1["amg_cache_hit"] is False
    assert pressure2["amg_cache_hit"] is True
    assert pressure1["amg_setup_wall_seconds"] >= 0.0


def test_pcg_l2_guard_rejects_false_relative_max_warm_start() -> None:
    n = 100
    coeff = {
        "diag": np.ones(n, dtype=np.float64),
        "int_owner": np.asarray([], dtype=np.int64),
        "int_neigh": np.asarray([], dtype=np.int64),
        "int_k": np.asarray([], dtype=np.float64),
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
    }
    rhs = np.zeros(n, dtype=np.float64)
    rhs[0] = 1000.0
    residual = np.full(n, 0.05, dtype=np.float64)
    p0 = rhs - residual
    legacy_cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        pressure_relative_tolerance=1e-4,
        backend="numpy",
    )
    guarded_cfg = replace(
        legacy_cfg,
        pcg_require_relative_l2_convergence=True,
    )

    legacy_p, legacy_diag = flow_solver_module._solve_pressure_pcg_diag_numpy(
        coeff, rhs=rhs, p0=p0, config=legacy_cfg
    )
    guarded_p, guarded_diag = flow_solver_module._solve_pressure_pcg_diag_numpy(
        coeff, rhs=rhs, p0=p0, config=guarded_cfg
    )

    np.testing.assert_array_equal(legacy_p, p0)
    assert legacy_diag["stopping_reason"] == "converged_relative_max"
    assert legacy_diag["actual_iterations"] == 0
    np.testing.assert_allclose(guarded_p, rhs, rtol=0.0, atol=1e-12)
    assert guarded_diag["stopping_reason"] == "converged_relative_l2"
    assert guarded_diag["actual_iterations"] == 1
    assert guarded_diag["true_residual_recompute_count"] == 1
    assert guarded_diag["final_true_residual_l2"] <= 1e-12


def test_pcg_l2_guard_verifies_true_final_residual() -> None:
    coeff = {
        "diag": np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        "int_owner": np.asarray([0, 1], dtype=np.int64),
        "int_neigh": np.asarray([1, 2], dtype=np.int64),
        "int_k": np.asarray([1.0, 1.0], dtype=np.float64),
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
    }
    expected = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, expected)
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=200,
        pressure_relative_tolerance=1e-10,
        cg_breakdown_eps=1e-40,
        pcg_require_relative_l2_convergence=True,
        backend="numpy",
    )

    pressure, diag = flow_solver_module._solve_pressure_pcg_diag_numpy(
        coeff, rhs=rhs, p0=np.zeros_like(rhs), config=cfg
    )
    true_residual = rhs - _matvec_pressure_numpy(coeff, pressure)
    true_l2 = float(np.sqrt(np.mean(true_residual**2)))

    np.testing.assert_allclose(pressure, expected, rtol=0.0, atol=1e-8)
    assert diag["true_residual_recompute_count"] >= 1
    assert diag["true_residual_restart_count"] == 0
    assert diag["final_true_residual_l2"] == pytest.approx(true_l2)


def test_pcg_l2_guard_restarts_once_when_stagnation_is_detected(monkeypatch) -> None:
    coeff = {
        "diag": np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        "int_owner": np.asarray([0, 1], dtype=np.int64),
        "int_neigh": np.asarray([1, 2], dtype=np.int64),
        "int_k": np.asarray([1.0, 1.0], dtype=np.float64),
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
    }
    expected = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, expected)
    calls = 0

    def detect_once(**_: object) -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    monkeypatch.setattr(flow_solver_module, "_cg_detect_stagnation", detect_once)
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=200,
        pressure_relative_tolerance=1e-10,
        cg_breakdown_eps=1e-40,
        pcg_require_relative_l2_convergence=True,
        backend="numpy",
    )

    pressure, diag = flow_solver_module._solve_pressure_pcg_diag_numpy(
        coeff, rhs=rhs, p0=np.zeros_like(rhs), config=cfg
    )

    np.testing.assert_allclose(pressure, expected, rtol=0.0, atol=1e-8)
    assert diag["true_residual_restart_count"] == 1
    assert diag["stopping_reason"] == "converged_relative_l2"


def test_default_flow_device_defers_to_backend_selection() -> None:
    assert TetraFlowConfig().device == ""


def test_torch_pcg_diag_can_disable_per_iteration_history() -> None:
    pytest.importorskip("torch")
    coeff = {
        "diag": np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        "int_owner": np.asarray([0, 1], dtype=np.int64),
        "int_neigh": np.asarray([1, 2], dtype=np.int64),
        "int_k": np.asarray([1.0, 1.0], dtype=np.float64),
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
    }
    p_true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, p_true)
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=200,
        pressure_tolerance=1e-14,
        pressure_relative_tolerance=1e-12,
        cg_breakdown_eps=1e-40,
        pcg_require_relative_l2_convergence=True,
        backend="torch",
        device="cpu",
        debug_store_history=False,
    )

    pressure, diag, _ = flow_solver_module._solve_pressure_pcg_diag_torch(
        coeff,
        rhs=rhs,
        p0=np.zeros_like(rhs),
        config=cfg,
        device="cpu",
    )

    np.testing.assert_allclose(pressure, p_true, atol=1e-8)
    assert diag["pressure_history"] == []
    assert diag["true_residual_recompute_count"] >= 1
    assert diag["final_true_residual_l2"] <= 1e-10


def test_torch_pcg_diag_can_return_resident_tensor() -> None:
    torch = pytest.importorskip("torch")
    coeff = _small_pressure_coefficients_with_cached_geometry()
    p_true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, p_true)
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=100,
        pressure_tolerance=1e-14,
        pressure_relative_tolerance=1e-12,
        cg_breakdown_eps=1e-40,
        backend="torch",
        device="cpu",
        debug_store_history=False,
    )

    pressure, diag, all_cuda = flow_solver_module._solve_pressure_pcg_diag_torch(
        coeff,
        rhs=torch.as_tensor(rhs, dtype=torch.float64),
        p0=torch.zeros_like(torch.as_tensor(rhs, dtype=torch.float64)),
        config=cfg,
        device="cpu",
        _return_tensor=True,
    )

    assert isinstance(pressure, torch.Tensor)
    np.testing.assert_allclose(pressure.numpy(), p_true, atol=1e-10, rtol=1e-10)
    assert diag["pressure_solved"] is True
    assert all_cuda is False


def _small_pressure_coefficients_with_cached_geometry() -> dict[str, np.ndarray]:
    base_diag = np.asarray([3.0, 2.0, 3.0], dtype=np.float64)
    base_int_k = np.asarray([1.0, 1.0], dtype=np.float64)
    scale = 0.25
    return {
        "diag": base_diag * scale,
        "int_owner": np.asarray([0, 1], dtype=np.int64),
        "int_neigh": np.asarray([1, 2], dtype=np.int64),
        "int_k": base_int_k * scale,
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
        "base_diag": base_diag,
        "base_int_k": base_int_k,
        "k_scale": scale,
    }


def test_torch_cpu_pcg_keeps_index_add_pressure_matvec() -> None:
    pytest.importorskip("torch")
    coeff = _small_pressure_coefficients_with_cached_geometry()
    p_true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, p_true)
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=100,
        pressure_tolerance=1e-14,
        pressure_relative_tolerance=1e-12,
        cg_breakdown_eps=1e-40,
        backend="torch",
        device="cpu",
        debug_store_history=False,
    )

    pressure, diag, all_cuda = flow_solver_module._solve_pressure_pcg_diag_torch(
        coeff,
        rhs=rhs,
        p0=np.zeros_like(rhs),
        config=cfg,
        device="cpu",
    )

    np.testing.assert_allclose(pressure, p_true, atol=1e-10, rtol=1e-10)
    assert all_cuda is False
    assert diag["pressure_matvec_backend"] == "torch_index_add"
    assert diag["pressure_matvec_sparse_csr_used"] is False
    assert diag["pressure_matvec_fallback_reason"] == ""


def test_torch_cuda_pcg_csr_matches_cpu_reference_and_reuses_matrix() -> None:
    torch = pytest.importorskip("torch")
    if not bool(torch.cuda.is_available()):
        pytest.skip("CUDA unavailable")
    flow_solver_module._TORCH_PRESSURE_CACHE.clear()
    coeff = _small_pressure_coefficients_with_cached_geometry()
    p_true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, p_true)
    base_cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=100,
        pressure_tolerance=1e-14,
        pressure_relative_tolerance=1e-12,
        cg_breakdown_eps=1e-40,
        backend="torch",
        device="cuda:0",
        debug_store_history=True,
    )

    reference, reference_diag, reference_all_cuda = (
        flow_solver_module._solve_pressure_pcg_diag_torch(
            coeff,
            rhs=rhs,
            p0=np.zeros_like(rhs),
            config=replace(base_cfg, device="cpu"),
            device="cpu",
        )
    )
    csr_first, first_diag, _ = flow_solver_module._solve_pressure_pcg_diag_torch(
        coeff,
        rhs=rhs,
        p0=np.zeros_like(rhs),
        config=base_cfg,
        device="cuda:0",
    )
    csr_second, second_diag, _ = flow_solver_module._solve_pressure_pcg_diag_torch(
        coeff,
        rhs=rhs,
        p0=np.zeros_like(rhs),
        config=base_cfg,
        device="cuda:0",
    )

    np.testing.assert_allclose(csr_first, reference, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(csr_second, csr_first)
    assert reference_all_cuda is False
    assert first_diag["actual_iterations"] == reference_diag["actual_iterations"]
    assert first_diag["stopping_reason"] == reference_diag["stopping_reason"]
    assert first_diag["pressure_matvec_backend"] == "torch_sparse_csr"
    assert first_diag["pressure_matvec_sparse_csr_used"] is True
    assert first_diag["pressure_matvec_matrix_cached"] is False
    assert second_diag["pressure_matvec_matrix_cached"] is True
    assert first_diag["pressure_matvec_fallback_reason"] == ""


def test_pressure_history_contains_rhs_stats() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=50,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    hist = state1.diagnostics.get("pressure_solver_history", {})
    assert float(hist.get("rhs_stats", {}).get("max_abs", 0.0)) >= 0.0
    assert "actual_iterations" in hist
    assert "stopping_reason" in hist


def test_torch_projection_smoke_if_cuda_available() -> None:
    torch = pytest.importorskip("torch")
    if not bool(torch.cuda.is_available()):
        pytest.skip("CUDA unavailable")
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="torch",
        device="cuda:0",
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    be = state1.diagnostics["backend_execution"]
    assert be["selected_backend"] == "torch"
    assert str(be["device"]).startswith("cuda")
    assert be["used_numpy_fallback"] is False
    assert be["all_core_arrays_on_cuda"] is True


def test_projection_limiter_none_matches_default_behavior() -> None:
    mesh = _build_synthetic_mesh()
    cfg_default = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=80,
        pressure_tolerance=1e-8,
        backend="numpy",
    )
    cfg_none = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=80,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_correction_limit_mode="none",
    )
    s0 = initialize_tetra_flow_state(mesh, cfg_default)
    a = solve_tetra_pressure_projection(mesh, s0, cfg_default)
    b = solve_tetra_pressure_projection(mesh, s0, cfg_none)
    assert np.allclose(a.face_flux, b.face_flux, rtol=0.0, atol=1e-14)
    assert (
        a.diagnostics.get("correction_limiter", {}).get(
            "projection_correction_limit_mode", ""
        )
        == "none"
    )


def test_face_flux_cap_limits_correction_flux_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_correction_limit_mode="face_flux_cap",
        projection_face_correction_over_volume_cap=1e-4,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s1 = solve_tetra_pressure_projection(mesh, s0, cfg)
    lim = dict(s1.diagnostics.get("correction_limiter", {}))
    assert int(lim.get("number_of_limited_faces", 0)) > 0
    q_lim = np.asarray(
        s1.diagnostics.get("face_flux_primary", {}).get(
            "correction_flux_limiter_output_pre_reconstraint", []
        ),
        dtype=np.float64,
    )
    ids = np.asarray(lim.get("limited_face_indices", []), dtype=np.int64)
    if ids.size:
        c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
        c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
        vol = np.asarray(mesh.cell_volumes, dtype=np.float64)
        for fid in ids.tolist():
            if c1[fid] < 0:
                continue
            bound = float(cfg.projection_face_correction_over_volume_cap) * min(
                float(vol[c0[fid]]),
                float(vol[c1[fid]]),
            )
            assert abs(float(q_lim[fid])) <= bound + 1e-18


def test_cell_divergence_cap_reduces_hotspot_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    rng = np.random.default_rng(3)
    face_flux = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    interior = np.asarray(mesh.interior_face_indices, dtype=np.int64)
    face_flux[interior] = 3e-3 * rng.standard_normal(interior.size)
    state0 = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=face_flux,
        pressure=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        diagnostics={},
    )
    cfg_none = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_correction_limit_mode="none",
    )
    cfg_cap = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=120,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_correction_limit_mode="cell_divergence_cap",
        projection_divergence_cap_factor=0.5,
        projection_divergence_floor=1e-8,
    )
    s_none = solve_tetra_pressure_projection(mesh, state0, cfg_none)
    s_cap = solve_tetra_pressure_projection(mesh, state0, cfg_cap)
    d_none = float(s_none.diagnostics["projection"]["final_divergence_max_abs"])
    d_cap = float(s_cap.diagnostics["projection"]["final_divergence_max_abs"])
    assert d_cap <= d_none


def test_limiter_conservation_audit_fields_present() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=60,
        pressure_tolerance=1e-8,
        backend="numpy",
        projection_correction_limit_mode="face_flux_cap",
        projection_face_correction_over_volume_cap=100.0,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s1 = solve_tetra_pressure_projection(mesh, s0, cfg)
    audit = dict(s1.diagnostics.get("correction_limiter_conservation_audit", {}))
    required = {
        "interior_pairwise_flux_conservation_error_max_abs",
        "interior_pairwise_flux_conservation_error_l2",
        "total_boundary_flux_before_limiter",
        "total_boundary_flux_after_limiter",
        "inlet_flux_before_limiter",
        "inlet_flux_after_limiter",
        "outlet_flux_before_limiter",
        "outlet_flux_after_limiter",
        "wall_flux_before_limiter",
        "wall_flux_after_limiter",
        "total_cell_flux_sum_before_limiter",
        "total_cell_flux_sum_after_limiter",
        "limiter_flux_delta_total",
    }
    assert required.issubset(set(audit.keys()))


def test_explicit_matrix_audit_two_cell_is_symmetric_and_nonnegative_energy() -> None:
    mesh = _build_two_cell_mesh()
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=5e-4,
        density=1000.0,
        outlet_faces=np.asarray([], dtype=np.int64),
    )
    audit = _pressure_matrix_explicit_audit(coeff, int(mesh.tetrahedra.shape[0]))
    assert float(audit["symmetry_max_abs"]) <= 1e-14
    assert int(audit["number_of_negative_diag"]) == 0
    op_cmp = _pressure_matrixfree_vs_explicit_audit(
        coeff,
        n_cells=int(mesh.tetrahedra.shape[0]),
        pressure_solution=np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64),
        random_count=4,
        seed=123,
    )
    assert float(op_cmp["max_abs_diff"]) <= 1e-14
    spd = _pressure_operator_spd_audit(
        coeff,
        n_cells=int(mesh.tetrahedra.shape[0]),
        random_count=16,
        seed=123,
    )
    assert int(spd["negative_energy_count"]) == 0


def test_explicit_matrix_one_cell_outlet_dirichlet_positive_diagonal() -> None:
    mesh = _build_one_cell_outlet_mesh()
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=5e-4,
        density=1000.0,
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
    )
    audit = _pressure_matrix_explicit_audit(coeff, int(mesh.tetrahedra.shape[0]))
    assert float(audit["diag_stats"]["min"]) > 0.0
    cmp_res = _pressure_matrixfree_vs_explicit_audit(
        coeff,
        n_cells=int(mesh.tetrahedra.shape[0]),
        pressure_solution=np.asarray([1.0], dtype=np.float64),
        random_count=2,
        seed=7,
    )
    assert float(cmp_res["max_abs_diff"]) <= 1e-14


def test_cg_converges_on_three_cell_chain_known_solution() -> None:
    coeff = {
        "diag": np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        "int_owner": np.asarray([0, 1], dtype=np.int64),
        "int_neigh": np.asarray([1, 2], dtype=np.int64),
        "int_k": np.asarray([1.0, 1.0], dtype=np.float64),
        "out_owner": np.asarray([], dtype=np.int64),
        "out_k": np.asarray([], dtype=np.float64),
        "out_face": np.asarray([], dtype=np.int64),
    }
    p_true = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = _matvec_pressure_numpy(coeff, p_true)
    cfg = TetraFlowConfig(
        pressure_solver="cg",
        max_pressure_iterations=200,
        pressure_tolerance=1e-14,
        pressure_relative_tolerance=1e-12,
        cg_breakdown_eps=1e-40,
        backend="numpy",
    )
    p, diag = _solve_pressure_cg_numpy(
        coeff,
        rhs=rhs,
        p0=np.zeros_like(rhs),
        config=cfg,
    )
    err = np.max(np.abs(np.asarray(p, dtype=np.float64) - p_true))
    assert err <= 1e-8
    assert str(diag.get("stopping_reason", "")) in {
        "converged_relative_l2",
        "converged_relative_max",
        "converged_absolute",
        "breakdown_near_converged",
    }


def test_reference_solver_explicit_scipy_or_graceful_unavailable() -> None:
    mesh = _build_two_cell_mesh()
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=5e-4,
        density=1000.0,
        outlet_faces=np.asarray([], dtype=np.int64),
    )
    rhs = np.asarray([1.0, -1.0], dtype=np.float64)
    out = _solve_pressure_reference_explicit(
        coeff,
        rhs=rhs,
        x0=np.zeros_like(rhs),
        rtol=1e-8,
        maxiter=1000,
    )
    if not bool(out.get("scipy_available", False)):
        pytest.skip("scipy unavailable for reference explicit solver")
    methods = list(out.get("methods", []))
    assert len(methods) >= 1


def test_projection_acceptance_allows_stagnated_with_small_residual() -> None:
    projection = {
        "final_divergence_l2": 0.7,
        "final_divergence_max_abs": 12.0,
        "divergence_reduction_ratio_l2": 0.002,
        "divergence_reduction_ratio": 0.002,
        "inlet_flux_total_after": 3.0e-7,
        "outlet_flux_total_after": 2.999e-7,
        "net_boundary_flux_after": 1.0e-10,
        "wall_flux_max_abs_after": 0.0,
    }
    pressure = {
        "stopping_reason": "stagnated",
        "residual_ratio_to_rhs_l2": 2.5e-3,
        "residual_ratio_to_rhs_max": 2.0e-3,
    }
    acc = _projection_acceptance(projection=projection, pressure=pressure)
    assert acc["pressure_linear_solved_strict"] is False
    assert acc["pressure_linear_accepted"] is True
    assert acc["projection_accepted"] is True
    assert acc["projection_solved"] is True


def _cuda_backend_execution() -> dict[str, object]:
    return {
        "used_numpy_fallback": False,
        "selected_backend": "torch",
        "device": "cuda:0",
        "all_core_arrays_on_cuda": True,
    }


def test_fail_if_numpy_fallback_accepts_cuda_stages_and_ignores_disabled() -> None:
    flow_debug_module._assert_no_numpy_fallback(
        backend_execution=_cuda_backend_execution(),
        step_history=[
            {
                "step": 1,
                "convective_predictor_used": True,
                "convective_torch_cuda_used": True,
                "convective_numpy_fallback_reason": "",
                "viscous_predictor_used": True,
                "viscous_torch_cuda_used": True,
                "viscous_numpy_fallback_reason": "",
            },
            {
                "step": 2,
                "convective_predictor_used": False,
                "convective_torch_cuda_used": False,
                "viscous_predictor_used": False,
                "viscous_torch_cuda_used": False,
            },
        ],
    )


@pytest.mark.parametrize(
    ("stage", "row"),
    [
        (
            "convection",
            {
                "convective_predictor_used": True,
                "convective_torch_cuda_used": False,
                "convective_numpy_fallback_reason": "unsupported stabilization",
            },
        ),
        (
            "viscosity",
            {
                "viscous_predictor_used": True,
                "viscous_torch_cuda_used": False,
                "viscous_numpy_fallback_reason": "unsupported predictor",
            },
        ),
    ],
)
def test_fail_if_numpy_fallback_rejects_stage_reason(
    stage: str, row: dict[str, object]
) -> None:
    with pytest.raises(RuntimeError, match=rf"{stage} used numpy fallback at step 7"):
        flow_debug_module._assert_no_numpy_fallback(
            backend_execution=_cuda_backend_execution(),
            step_history=[{"step": 7, **row}],
        )


@pytest.mark.parametrize(
    ("stage", "row"),
    [
        (
            "convection",
            {
                "convective_predictor_used": True,
                "convective_torch_cuda_used": False,
                "convective_numpy_fallback_reason": "",
            },
        ),
        (
            "viscosity",
            {
                "viscous_predictor_used": True,
                "viscous_torch_cuda_used": False,
                "viscous_numpy_fallback_reason": "",
            },
        ),
    ],
)
def test_fail_if_numpy_fallback_rejects_missing_cuda_dispatch(
    stage: str, row: dict[str, object]
) -> None:
    with pytest.raises(RuntimeError, match=rf"{stage} did not execute on Torch CUDA"):
        flow_debug_module._assert_no_numpy_fallback(
            backend_execution=_cuda_backend_execution(),
            step_history=[{"step": 3, **row}],
        )


def test_projection_acceptance_fails_when_outlet_flux_collapses() -> None:
    projection = {
        "final_divergence_l2": 0.5,
        "final_divergence_max_abs": 8.0,
        "divergence_reduction_ratio_l2": 0.002,
        "divergence_reduction_ratio": 0.002,
        "inlet_flux_total_after": 3.0e-7,
        "outlet_flux_total_after": 1.0e-8,
        "net_boundary_flux_after": -2.9e-7,
        "wall_flux_max_abs_after": 0.0,
    }
    pressure = {
        "stopping_reason": "stagnated",
        "residual_ratio_to_rhs_l2": 2.5e-3,
        "residual_ratio_to_rhs_max": 2.0e-3,
    }
    acc = _projection_acceptance(projection=projection, pressure=pressure)
    assert acc["pressure_linear_accepted"] is True
    assert acc["projection_accepted"] is False
    assert acc["projection_solved"] is False


def test_projection_acceptance_fails_when_divergence_high() -> None:
    projection = {
        "final_divergence_l2": 35.0,
        "final_divergence_max_abs": 400.0,
        "divergence_reduction_ratio_l2": 0.08,
        "divergence_reduction_ratio": 0.06,
        "inlet_flux_total_after": 3.0e-7,
        "outlet_flux_total_after": 2.99e-7,
        "net_boundary_flux_after": 1e-10,
        "wall_flux_max_abs_after": 0.0,
    }
    pressure = {
        "stopping_reason": "stagnated",
        "residual_ratio_to_rhs_l2": 2.5e-3,
        "residual_ratio_to_rhs_max": 2.0e-3,
    }
    acc = _projection_acceptance(projection=projection, pressure=pressure)
    assert acc["pressure_linear_accepted"] is True
    assert acc["projection_accepted"] is False
    assert acc["projection_solved"] is False


def test_projection_acceptance_fails_when_pressure_residual_is_too_large() -> None:
    projection = {
        "final_divergence_l2": 0.4,
        "final_divergence_max_abs": 6.0,
        "divergence_reduction_ratio_l2": 1e-3,
        "divergence_reduction_ratio": 1e-3,
        "inlet_flux_total_after": 3.0e-7,
        "outlet_flux_total_after": 2.999e-7,
        "net_boundary_flux_after": 1e-10,
        "wall_flux_max_abs_after": 0.0,
    }
    pressure = {
        "stopping_reason": "max_iterations",
        "residual_ratio_to_rhs_l2": 0.25,
        "residual_ratio_to_rhs_max": 0.15,
    }
    acc = _projection_acceptance(projection=projection, pressure=pressure)
    assert acc["pressure_linear_accepted"] is False
    assert acc["projection_accepted"] is True
    assert acc["projection_solved"] is False


def test_projection_acceptance_fails_on_nonfinite_metrics() -> None:
    projection = {
        "final_divergence_l2": float("nan"),
        "final_divergence_max_abs": 6.0,
        "divergence_reduction_ratio_l2": 1e-3,
        "divergence_reduction_ratio": float("inf"),
        "inlet_flux_total_after": 3.0e-7,
        "outlet_flux_total_after": 2.999e-7,
        "net_boundary_flux_after": 1e-10,
        "wall_flux_max_abs_after": 0.0,
    }
    pressure = {
        "stopping_reason": "converged_relative_l2",
        "residual_ratio_to_rhs_l2": 1e-4,
        "residual_ratio_to_rhs_max": 1e-4,
    }
    acc = _projection_acceptance(projection=projection, pressure=pressure)
    assert acc["projection_accepted"] is False
    assert acc["projection_solved"] is False
    assert acc["checklist"]["finite_fields"] is False


def test_runner_default_pressure_solver_is_pcg_diag() -> None:
    assert DEFAULT_TETRA_FLOW_DEBUG_PRESSURE_SOLVER == "pcg_diag"


def test_runner_default_startup_bootstrap_cap_includes_qualification_tail() -> None:
    assert STARTUP_BOOTSTRAP_LEGACY_SEARCH_BUDGET == 20
    assert STARTUP_BOOTSTRAP_QUALIFICATION_TAIL == 2
    assert DEFAULT_STARTUP_BOOTSTRAP_MAX_STEPS == 22
    source_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_flow_debug.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "--startup-bootstrap-max-steps":
            continue
        defaults = {
            keyword.arg: keyword.value
            for keyword in node.keywords
            if keyword.arg is not None
        }
        assert isinstance(defaults.get("default"), ast.Name)
        assert defaults["default"].id == "DEFAULT_STARTUP_BOOTSTRAP_MAX_STEPS"
        return
    raise AssertionError("Did not find --startup-bootstrap-max-steps CLI option")


def test_runner_pressure_projection_outlet_contract_cli_defaults_and_forwards() -> None:
    source_path = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    option_found = False
    config_forwarded = False
    summary_key_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == "pressure_projection_outlet_contract_mode"
        ):
            summary_key_count += 1
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--pressure-projection-outlet-contract-mode"
            ):
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                assert ast.literal_eval(keywords["choices"]) == (
                    "auto",
                    "match_inlet",
                    "preserve",
                )
                assert ast.literal_eval(keywords["default"]) == "auto"
                option_found = True
        if isinstance(node.func, ast.Name) and node.func.id == "TetraFlowConfig":
            config_forwarded = config_forwarded or any(
                keyword.arg == "pressure_projection_outlet_contract_mode"
                for keyword in node.keywords
            )
    assert option_found
    assert config_forwarded
    assert summary_key_count >= 3


def test_runner_projection_cell_velocity_update_cli_defaults_and_forwards() -> None:
    source_path = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    option_found = False
    config_forwarded = False
    summary_key_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == "projection_cell_velocity_update_mode"
        ):
            summary_key_count += 1
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--projection-cell-velocity-update-mode"
            ):
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                assert ast.literal_eval(keywords["choices"]) == (
                    "auto",
                    "legacy_reconstruct",
                    "momentum_pressure_corrected",
                )
                assert ast.literal_eval(keywords["default"]) == "auto"
                option_found = True
        if isinstance(node.func, ast.Name) and node.func.id == "TetraFlowConfig":
            config_forwarded = config_forwarded or any(
                keyword.arg == "projection_cell_velocity_update_mode"
                for keyword in node.keywords
            )
    assert option_found
    assert config_forwarded
    assert summary_key_count >= 3


def test_runner_rejects_negative_startup_bootstrap_cap_before_mesh_validation() -> None:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_flow_debug.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(source_path),
            "--flow-steps",
            "1",
            "--startup-bootstrap-max-steps",
            "-1",
        ],
        cwd=source_path.parents[2],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "--startup-bootstrap-max-steps must be non-negative." in completed.stderr


def test_parse_snapshot_steps_basic() -> None:
    assert _parse_snapshot_steps("1,5,10,20") == {1, 5, 10, 20}
    assert _parse_snapshot_steps(" ,2,abc,3,, ") == {2, 3}
    assert _parse_snapshot_steps("") == set()


def test_next_snapshot_time_is_global_across_resume() -> None:
    assert _next_snapshot_time(0.0, 0.005) == pytest.approx(0.005)
    assert _next_snapshot_time(0.03934987867815835, 0.005) == pytest.approx(0.04)
    assert _next_snapshot_time(0.04, 0.005) == pytest.approx(0.045)


def test_flow_dt_is_unaffected_by_snapshot_boundary_and_clamped_to_stop_time() -> None:
    assert _clamp_flow_dt_to_stop_time(
        physical_time=0.039,
        dt_candidate=0.01,
        stop_physical_time=0.05,
    ) == pytest.approx(0.01)
    assert _clamp_flow_dt_to_stop_time(
        physical_time=0.049,
        dt_candidate=0.01,
        stop_physical_time=0.05,
    ) == pytest.approx(0.001)


def test_time_snapshots_do_not_change_dt_or_final_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.gmsh.run_import_gmsh_mesh import _save_npz

    mesh_path = tmp_path / "synthetic_imported_mesh.npz"
    _save_npz(_build_synthetic_mesh(), mesh_path)
    monkeypatch.setattr(flow_debug_module, "_POSTPROCESSING_MODE", "full")
    monkeypatch.setattr(flow_debug_module, "_save_scatter", lambda **_kwargs: None)
    monkeypatch.setattr(
        flow_debug_module,
        "_save_vectors_sparse_normalized",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(flow_debug_module, "_export_vtu", lambda **_kwargs: None)

    def run_case(output_root: Path, *extra_args: str) -> Path:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_gmsh_tetra_flow_debug.py",
                "--mesh-npz",
                str(mesh_path),
                "--output-root",
                str(output_root),
                "--backend",
                "numpy",
                "--flow-execution-backend",
                "numpy",
                "--flow-steps",
                "1",
                "--flow-dt-mode",
                "manual",
                "--flow-dt",
                "1e-5",
                "--max-pressure-iterations",
                "1",
                "--postprocessing-mode",
                "minimal",
                *extra_args,
            ],
        )
        flow_debug_module.main()
        summaries = list(output_root.glob("*/summary.json"))
        assert len(summaries) == 1
        return summaries[0].parent

    baseline_dir = run_case(tmp_path / "baseline")
    snapshot_dir = run_case(
        tmp_path / "snapshots",
        "--snapshot-time-interval",
        "5e-6",
    )

    baseline_history = json.loads(
        (baseline_dir / "flow_progression_history.json").read_text("utf-8")
    )
    snapshot_history = json.loads(
        (snapshot_dir / "flow_progression_history.json").read_text("utf-8")
    )
    assert [row["used_flow_dt"] for row in snapshot_history["steps"]] == [
        row["used_flow_dt"] for row in baseline_history["steps"]
    ]
    assert snapshot_history["steps"][0]["used_flow_dt"] == pytest.approx(1e-5)
    for filename in FLOW_RESUME_STATE_FILENAMES:
        np.testing.assert_array_equal(
            np.load(snapshot_dir / filename),
            np.load(baseline_dir / filename),
        )

    metadata_paths = list(snapshot_dir.glob("snapshots/*/snapshot_metadata.json"))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text("utf-8"))
    assert metadata["requested_snapshot_time"] == pytest.approx(5e-6)
    assert metadata["actual_snapshot_time"] == pytest.approx(1e-5)


def test_flow_progression_acceptance_true_for_stable_history() -> None:
    hist = [
        {
            "step": 1,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0002,
            "wall_flux_max_abs_after": 0.0,
            "final_divergence_max_abs": 12.0,
        },
        {
            "step": 2,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0001,
            "wall_flux_max_abs_after": 0.0,
            "final_divergence_max_abs": 11.0,
        },
    ]
    out = _evaluate_flow_progression_acceptance(
        history=hist,
        allow_projection_warning_steps=0,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )
    assert out["flow_progression_solved"] is True


def test_flow_progression_acceptance_startup_warning_tolerated() -> None:
    hist = [
        {
            "step": 1,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0001,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1000.0,
            "final_divergence_max_abs": 80.0,
        },
        {
            "step": 2,
            "projection_solved": False,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0002,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1000.0,
            "final_divergence_max_abs": 25.0,
        },
        {
            "step": 3,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0001,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1000.0,
            "final_divergence_max_abs": 10.0,
        },
    ]
    out = _evaluate_flow_progression_acceptance(
        history=hist,
        allow_projection_warning_steps=0,
        startup_warning_steps=3,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )
    assert out["flow_progression_solved"] is True
    assert out["startup_warning_steps_observed"] == [2]
    assert out["nonstartup_failed_steps"] == []


def test_flow_progression_acceptance_nonstartup_failed_step_fails() -> None:
    hist = [
        {
            "step": 1,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0001,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1000.0,
            "final_divergence_max_abs": 30.0,
        },
        {
            "step": 4,
            "projection_solved": False,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0001,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1000.0,
            "final_divergence_max_abs": 25.0,
        },
    ]
    out = _evaluate_flow_progression_acceptance(
        history=hist,
        allow_projection_warning_steps=0,
        startup_warning_steps=3,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )
    assert out["flow_progression_solved"] is False
    assert out["nonstartup_failed_steps"] == [4]


def test_flow_progression_never_allows_nonstartup_pressure_failure() -> None:
    history = [
        {
            "step": 11,
            "projection_solved": False,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 1.0,
            "wall_flux_max_abs_after": 0.0,
            "initial_divergence_max_abs": 1.0,
            "final_divergence_max_abs": 0.1,
        }
    ]
    result = _evaluate_flow_progression_acceptance(
        history=history,
        allow_projection_warning_steps=5,
        startup_warning_steps=10,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )

    assert result["flow_progression_solved"] is False
    assert result["nonstartup_failed_steps"] == [11]


def test_flow_progression_acceptance_false_on_nonfinite_or_flux_collapse() -> None:
    hist_bad = [
        {
            "step": 1,
            "projection_solved": True,
            "finite_fields": False,
            "outlet_inlet_flux_ratio": 1.0,
            "wall_flux_max_abs_after": 0.0,
            "final_divergence_max_abs": 10.0,
        }
    ]
    out_bad = _evaluate_flow_progression_acceptance(
        history=hist_bad,
        allow_projection_warning_steps=0,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )
    assert out_bad["flow_progression_solved"] is False

    hist_collapse = [
        {
            "step": 1,
            "projection_solved": True,
            "finite_fields": True,
            "outlet_inlet_flux_ratio": 0.1,
            "wall_flux_max_abs_after": 0.0,
            "final_divergence_max_abs": 10.0,
        }
    ]
    out_col = _evaluate_flow_progression_acceptance(
        history=hist_collapse,
        allow_projection_warning_steps=0,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
    )
    assert out_col["flow_progression_solved"] is False


def test_projection_failed_criteria_reports_blocking_acceptance_failures_in_stable_order() -> (
    None
):
    acc = {
        "blocking_checks": {
            "pressure_linear_residual_acceptance": False,
            "finite_fields": True,
            "divergence_l2_acceptance": True,
            "divergence_linf_acceptance": False,
            "outlet_inlet_ratio": False,
            "net_boundary_flux_relative": True,
            "wall_flux": False,
        },
        "checklist": {
            "residual_l2": False,
            "residual_max": True,
            "div_l2": False,
            "div_linf": True,
            "div_reduction_l2": False,
            "div_reduction_linf": True,
            "outlet_inlet_ratio": False,
            "net_boundary_flux_relative": True,
            "wall_flux": False,
            "finite_fields": True,
        },
    }
    assert _projection_failed_criteria(acc) == [
        "pressure_linear_residual_acceptance",
        "divergence_linf_acceptance",
        "outlet_inlet_ratio",
        "wall_flux",
    ]


def test_projection_strict_diagnostics_are_reported_separately_from_acceptance_failures() -> (
    None
):
    acc = {
        "blocking_checks": {
            "pressure_linear_residual_acceptance": True,
            "finite_fields": True,
            "divergence_l2_acceptance": True,
            "divergence_linf_acceptance": True,
            "outlet_inlet_ratio": True,
            "net_boundary_flux_relative": True,
            "wall_flux": True,
        },
        "checklist": {
            "residual_l2": False,
            "residual_max": True,
            "div_l2": False,
            "div_linf": True,
            "div_reduction_l2": False,
            "div_reduction_linf": True,
            "inlet_flux": True,
            "outlet_flux": True,
            "outlet_inlet_ratio": True,
            "net_boundary_flux_relative": True,
            "wall_flux": True,
            "finite_fields": True,
        },
    }
    assert _projection_failed_criteria(acc) == []
    assert _projection_strict_diagnostic_criteria_not_met(acc) == [
        "residual_l2",
        "div_l2",
        "div_reduction_l2",
    ]


def test_build_projection_acceptance_step_record_exposes_root_cause_metrics() -> None:
    row = _build_projection_acceptance_step_record(
        step=3,
        acceptance={
            "projection_solved": False,
            "projection_accepted": False,
            "pressure_linear_accepted": False,
            "pressure_linear_solved_strict": False,
            "reason": "pressure linear system residual acceptance failed",
            "metrics": {
                "residual_ratio_to_rhs_l2": 0.12,
                "residual_ratio_to_rhs_max": 0.05,
                "outlet_inlet_flux_ratio": 1.1,
                "net_boundary_flux_after": 2e-9,
                "net_boundary_flux_relative": 1e-3,
                "wall_flux_max_abs_after": 0.0,
            },
            "blocking_checks": {
                "pressure_linear_residual_acceptance": False,
                "finite_fields": True,
                "divergence_l2_acceptance": True,
                "divergence_linf_acceptance": True,
                "outlet_inlet_ratio": False,
                "net_boundary_flux_relative": True,
                "wall_flux": True,
            },
            "checklist": {
                "residual_l2": False,
                "residual_max": True,
                "div_l2": True,
                "div_linf": True,
                "div_reduction_l2": True,
                "div_reduction_linf": True,
                "outlet_inlet_ratio": False,
                "net_boundary_flux_relative": True,
                "wall_flux": True,
                "finite_fields": True,
            },
        },
        projection={
            "initial_divergence_max_abs": 10.0,
            "final_divergence_max_abs": 25.0,
            "initial_divergence_l2": 2.0,
            "final_divergence_l2": 1.2,
        },
        pressure={"stopping_reason": "stagnated", "actual_iterations": 77},
        used_flow_dt=1e-5,
    )
    assert row["step"] == 3
    assert row["projection_acceptance_reason"] == (
        "pressure linear system residual acceptance failed"
    )
    assert row["projection_failed_criteria"] == [
        "pressure_linear_residual_acceptance",
        "outlet_inlet_ratio",
    ]
    assert row["strict_diagnostic_criteria_not_met"] == [
        "residual_l2",
        "outlet_inlet_ratio",
    ]
    assert row["pressure_iterations"] == 77
    assert row["pressure_residual_ratio_to_rhs_l2"] == 0.12
    assert row["used_flow_dt"] == 1e-5


def test_summarize_startup_bootstrap_reports_raw_initialization_root_cause() -> None:
    report = _summarize_startup_bootstrap(
        bootstrap_history=[
            {
                "step": 1,
                "projection_solved": False,
                "projection_failed_criteria": [
                    "pressure_linear_residual_acceptance",
                    "divergence_linf_acceptance",
                ],
                "strict_diagnostic_criteria_not_met": [
                    "residual_l2",
                    "div_linf",
                ],
            },
            {
                "step": 2,
                "projection_solved": True,
                "projection_failed_criteria": [],
                "strict_diagnostic_criteria_not_met": [],
            },
            {
                "step": 3,
                "projection_solved": True,
                "projection_failed_criteria": [],
                "strict_diagnostic_criteria_not_met": [],
            },
            {
                "step": 4,
                "projection_solved": True,
                "projection_failed_criteria": [],
                "strict_diagnostic_criteria_not_met": [],
            },
        ],
        initial_divergence={
            "divergence_max_abs": 123.0,
            "divergence_l2": 4.5,
        },
        requested_max_steps=20,
        bootstrap_required=True,
    )
    assert report["bootstrap_enabled"] is True
    assert report["bootstrap_converged"] is True
    assert report["bootstrap_requested_max_steps"] == 20
    assert report["bootstrap_physical_time_advanced"] == 0.0
    assert report["bootstrap_steps"] == 4
    assert report["bootstrap_required_consecutive_accepted_iterations"] == 3
    assert report["bootstrap_achieved_consecutive_accepted_iterations"] == 3
    assert report["dominant_failed_criteria"] == [
        "divergence_linf_acceptance",
        "pressure_linear_residual_acceptance",
    ]
    assert report["dominant_strict_diagnostic_criteria_not_met"] == [
        "div_linf",
        "residual_l2",
    ] or report["dominant_strict_diagnostic_criteria_not_met"] == [
        "residual_l2",
        "div_linf",
    ]
    assert report["initial_state_divergence_max_abs"] == 123.0


def test_summarize_startup_bootstrap_disabled_blocks_physical_progression() -> None:
    report = _summarize_startup_bootstrap(
        bootstrap_history=[],
        initial_divergence={
            "divergence_max_abs": 12.0,
            "divergence_l2": 1.5,
        },
        requested_max_steps=0,
        bootstrap_required=True,
    )
    assert report["bootstrap_enabled"] is False
    assert report["bootstrap_converged"] is False
    assert report["physical_progression_allowed"] is False
    assert report["bootstrap_reason"].startswith(
        "raw initialized face-flux state may be non-ready"
    )


def test_summarize_startup_bootstrap_cap_exhaustion_is_explicitly_blocking() -> None:
    report = _summarize_startup_bootstrap(
        bootstrap_history=[
            {
                "step": 1,
                "projection_solved": False,
                "projection_failed_criteria": ["divergence_l2_acceptance"],
                "strict_diagnostic_criteria_not_met": ["div_l2"],
            },
            {
                "step": 2,
                "projection_solved": False,
                "projection_failed_criteria": ["divergence_l2_acceptance"],
                "strict_diagnostic_criteria_not_met": ["div_l2"],
            },
        ],
        initial_divergence={
            "divergence_max_abs": 50.0,
            "divergence_l2": 5.0,
        },
        requested_max_steps=2,
        bootstrap_required=True,
    )
    assert report["bootstrap_cap_reached"] is True
    assert report["bootstrap_converged"] is False
    assert report["physical_progression_allowed"] is False
    assert report["bootstrap_reason"].endswith(
        "did not converge within the configured cap"
    )


def test_summarize_startup_bootstrap_requires_consecutive_accepted_steps() -> None:
    report = _summarize_startup_bootstrap(
        bootstrap_history=[
            {"step": 1, "projection_solved": True},
            {"step": 2, "projection_solved": False},
            {"step": 3, "projection_solved": True},
            {"step": 4, "projection_solved": True},
        ],
        initial_divergence={},
        requested_max_steps=4,
    )
    assert report["bootstrap_converged"] is False
    assert report["bootstrap_cap_reached"] is True
    assert report["bootstrap_achieved_consecutive_accepted_iterations"] == 2
    assert report["bootstrap_physical_time_advanced"] == 0.0


def test_startup_bootstrap_qualifies_three_consecutive_full_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter([True, False, True, True, True])
    state = TetraFlowState(
        cell_velocity=np.zeros((1, 3), dtype=np.float64),
        face_flux=np.zeros(1, dtype=np.float64),
        pressure=np.zeros(1, dtype=np.float64),
        diagnostics={},
    )
    mesh = type(
        "Mesh",
        (),
        {
            "outlet_faces": np.empty(0, dtype=np.int64),
            "wall_faces": np.empty(0, dtype=np.int64),
        },
    )()

    monkeypatch.setattr(
        flow_solver_module,
        "compute_tetra_convective_cfl_rate",
        lambda *_args, **_kwargs: {"cfl_rate_max": 1.0},
    )
    monkeypatch.setattr(
        flow_solver_module,
        "compute_tetra_flux_divergence",
        lambda *_args, **_kwargs: {"divergence_max_abs": 0.0, "divergence_l2": 0.0},
    )
    monkeypatch.setattr(
        flow_solver_module,
        "apply_tetra_convective_predictor",
        lambda _mesh, state, _cfg, **_kwargs: state,
    )
    monkeypatch.setattr(
        flow_solver_module,
        "apply_tetra_stokes_viscous_predictor",
        lambda _mesh, state, _cfg, **_kwargs: state,
    )
    monkeypatch.setattr(
        flow_solver_module,
        "solve_tetra_pressure_projection",
        lambda _mesh, state, _cfg: state,
    )
    monkeypatch.setattr(
        flow_debug_module,
        "_projection_acceptance",
        lambda **_kwargs: {"projection_solved": next(outcomes)},
    )
    monkeypatch.setattr(
        "microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver.resolve_inlet_face_groups",
        lambda _mesh: {"left_faces": [], "right_faces": []},
    )

    _state, history, report = _run_startup_bootstrap(
        mesh=mesh,
        state0=state,
        cfg=TetraFlowConfig(),
        flow_mode="navier_stokes_projection_debug",
        requested_flow_dt=1e-4,
        flow_dt_mode="manual",
        flow_dt_min=1e-7,
        flow_dt_max=1e-4,
        convective_cfl_target=0.5,
        acceptance_thresholds={},
        wall_strength_start=1.0,
        max_steps=5,
    )

    assert [row["projection_solved"] for row in history] == [
        True,
        False,
        True,
        True,
        True,
    ]
    assert all(row["iteration_kind"] == "pseudo_time_qualification" for row in history)
    assert all(row["physical_time_advanced"] == 0.0 for row in history)
    assert report["bootstrap_converged"] is True
    assert report["bootstrap_achieved_consecutive_accepted_iterations"] == 3
    assert report["initial_state_divergence_l2"] == 0.0


def test_velocity_region_audit_contains_required_regions_and_fields() -> None:
    centers = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.2, 0.4, 0.0],
            [-0.2, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    vel = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.3, 0.0],
            [0.2, 0.4, 0.0],
            [0.1, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    masks = {
        "left_inlet_branch": np.asarray([True, False, False, False, False, False]),
        "right_inlet_branch": np.asarray([False, True, False, False, False, False]),
        "junction": np.asarray([False, False, False, True, False, False]),
        "outlet_branch": np.asarray([False, False, True, False, True, True]),
        "boundary_adjacent": np.asarray([True, True, True, False, False, False]),
        "interior_core": np.asarray([False, False, False, True, True, True]),
    }
    audit = _velocity_region_audit(centers=centers, velocity=vel, region_masks=masks)
    for key in (
        "left_inlet_branch",
        "right_inlet_branch",
        "junction",
        "outlet_branch",
        "boundary_adjacent",
        "interior_core",
    ):
        assert key in audit
        assert "cell_count" in audit[key]
        assert "speed_p95" in audit[key]
        assert "mostly_expected_direction_fraction" in audit[key]
        assert "transverse_ratio_p95" in audit[key]


def test_binned_quiver_plot_does_not_crash(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(300, 3)).astype(np.float64)
    vel = rng.normal(size=(300, 3)).astype(np.float64)
    out = tmp_path / "velocity_vectors_xy_grid_binned.png"
    path = _save_velocity_vectors_xy_grid_binned(
        centers=centers,
        velocity=vel,
        out_path=out,
        bins_x=16,
        bins_y=16,
    )
    assert Path(path).exists()


def test_parse_flow_modes_basic() -> None:
    assert _parse_flow_modes(
        "projection_only,stokes_viscous_projection,navier_stokes_projection_debug"
    ) == [
        "projection_only",
        "stokes_viscous_projection",
        "navier_stokes_projection_debug",
    ]
    assert _parse_flow_modes("foo,projection_only,foo") == ["projection_only"]
    assert _parse_flow_modes("") == []


def test_resolve_viscous_predictor_mode_defaults_to_face_flux_for_stokes() -> None:
    mode, reason = _resolve_viscous_predictor_mode(
        flow_mode="stokes_viscous_projection",
        predictor_mode_cli="explicit_cell_velocity_laplacian_substepped",
        predictor_mode_explicit=False,
        wall_velocity_boundary_mode="slip",
    )
    assert mode == "face_flux_laplacian_substepped"
    assert "defaults to face_flux_laplacian_substepped" in reason


def test_resolve_viscous_predictor_mode_defaults_to_cell_velocity_for_no_slip() -> None:
    mode, reason = _resolve_viscous_predictor_mode(
        flow_mode="navier_stokes_projection_debug",
        predictor_mode_cli="explicit_cell_velocity_laplacian_substepped",
        predictor_mode_explicit=False,
        wall_velocity_boundary_mode="no_slip",
    )
    assert mode == "explicit_cell_velocity_laplacian_substepped_conservative"
    assert "cell-velocity wall momentum" in reason


def test_resolve_viscous_predictor_mode_respects_explicit_cli_override() -> None:
    mode, reason = _resolve_viscous_predictor_mode(
        flow_mode="stokes_viscous_projection",
        predictor_mode_cli="none",
        predictor_mode_explicit=True,
        wall_velocity_boundary_mode="no_slip",
    )
    assert mode == "none"
    assert "explicitly by CLI" in reason


def test_stokes_viscous_predictor_runs_without_nan_and_preserves_wall_flux_zero() -> (
    None
):
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s_pred = apply_tetra_stokes_viscous_predictor(mesh, s0, cfg, flow_dt=5e-4)
    vd = dict(s_pred.diagnostics.get("viscous_predictor", {}))
    assert vd.get("viscous_predictor_used", False) is True
    assert "viscous_stability_metric" in vd
    assert np.all(np.isfinite(np.asarray(s_pred.cell_velocity, dtype=np.float64)))
    s1 = solve_tetra_pressure_projection(mesh, s_pred, cfg)
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall.size:
        assert (
            float(np.max(np.abs(np.asarray(s1.face_flux, dtype=np.float64)[wall])))
            <= 1e-12
        )


def test_convective_predictor_runs_without_nan_and_preserves_wall_flux_after_projection() -> (
    None
):
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        enable_convective_predictor=True,
        convective_cfl_limit=0.5,
        convective_predictor_damping=1.0,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s_conv = apply_tetra_convective_predictor(mesh, s0, cfg, flow_dt=5e-4)
    cd = dict(s_conv.diagnostics.get("convective_predictor", {}))
    assert cd.get("convective_predictor_used", False) is True
    assert "convective_cfl_max" in cd
    assert np.all(np.isfinite(np.asarray(s_conv.cell_velocity, dtype=np.float64)))
    s_pred = apply_tetra_stokes_viscous_predictor(mesh, s_conv, cfg, flow_dt=5e-4)
    s1 = solve_tetra_pressure_projection(mesh, s_pred, cfg)
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall.size:
        assert (
            float(np.max(np.abs(np.asarray(s1.face_flux, dtype=np.float64)[wall])))
            <= 1e-12
        )


@pytest.mark.parametrize(
    "wall_mode",
    ("slip", "no_slip", "no_slip_tangential"),
)
def test_convective_predictor_dispatches_cuda_for_supported_wall_modes(
    monkeypatch: pytest.MonkeyPatch,
    wall_mode: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device="cuda:0",
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
    )
    state = initialize_tetra_flow_state(mesh, cfg)
    expected = apply_tetra_convective_predictor(
        mesh,
        state,
        replace(cfg, backend="numpy", device="cpu"),
        flow_dt=5e-4,
    )
    calls: list[str] = []

    def _fake_torch_step(
        mesh_arg,
        *,
        velocity,
        face_flux,
        cell_volume,
        update_scale,
        device,
    ):
        calls.append(str(device))
        conv_div = flow_solver_module._convective_divergence_term(
            mesh_arg,
            velocity=velocity,
            face_flux=face_flux,
            cell_volume=cell_volume,
        )
        velocity_next = np.asarray(velocity) - float(update_scale) * conv_div
        flux_next = flow_solver_module._face_flux_from_cell_velocity_numpy(
            mesh_arg, velocity_next
        )
        return velocity_next, flux_next

    monkeypatch.setattr(
        flow_solver_module, "_convective_predictor_step_torch", _fake_torch_step
    )
    result = apply_tetra_convective_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["convective_predictor"]
    np.testing.assert_array_equal(result.cell_velocity, expected.cell_velocity)
    np.testing.assert_array_equal(result.face_flux, expected.face_flux)
    assert calls == ["cuda:0"]
    assert diag["convective_execution_backend"] == "torch"
    assert diag["convective_execution_device"] == "cuda:0"
    assert diag["convective_torch_cuda_used"] is True
    assert diag["convective_cuda_handoff_available"] is False
    assert diag["convective_numpy_fallback_reason"] == ""


def test_convective_predictor_keeps_legacy_isotropic_no_slip_on_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device="cuda:0",
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        wall_velocity_boundary_mode="no_slip_legacy_isotropic",
    )
    state = initialize_tetra_flow_state(mesh, cfg)

    def _unexpected_torch_step(*_args, **_kwargs):
        raise AssertionError(
            "legacy isotropic no-slip convection must remain on the NumPy path"
        )

    monkeypatch.setattr(
        flow_solver_module,
        "_convective_predictor_step_torch",
        _unexpected_torch_step,
    )
    result = apply_tetra_convective_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["convective_predictor"]
    assert diag["convective_execution_backend"] == "numpy"
    assert diag["convective_torch_cuda_used"] is False
    assert diag["convective_numpy_fallback_reason"] == (
        "CUDA convection preserves the NumPy path for legacy isotropic no-slip walls"
    )


@pytest.mark.parametrize(
    ("device", "stabilization_mode", "fallback_reason"),
    (
        ("cpu", "auto_damping", "flow execution device is not CUDA"),
        (
            "cuda:0",
            "substepping",
            "CUDA convection currently preserves the NumPy path for substepping",
        ),
    ),
)
def test_convective_predictor_reports_numpy_fallback_for_unsupported_torch_modes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    stabilization_mode: str,
    fallback_reason: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device=device,
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        convective_stabilization_mode=stabilization_mode,  # type: ignore[arg-type]
        wall_velocity_boundary_mode="slip",
    )
    state = initialize_tetra_flow_state(mesh, cfg)

    def _unexpected_torch_step(*_args, **_kwargs):
        raise AssertionError("unsupported Torch modes must retain the NumPy path")

    monkeypatch.setattr(
        flow_solver_module,
        "_convective_predictor_step_torch",
        _unexpected_torch_step,
    )
    result = apply_tetra_convective_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["convective_predictor"]
    assert diag["convective_execution_backend"] == "numpy"
    assert diag["convective_execution_device"] == "cpu"
    assert diag["convective_torch_cuda_used"] is False
    assert diag["convective_numpy_fallback_reason"] == fallback_reason


@pytest.mark.parametrize(
    "wall_mode",
    ("slip", "no_slip", "no_slip_tangential"),
)
def test_convective_predictor_torch_cuda_matches_numpy_when_available(
    wall_mode: str,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = _build_synthetic_mesh()
    cfg_numpy = TetraFlowConfig(
        backend="numpy",
        device="cpu",
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
    )
    cfg_cuda = replace(cfg_numpy, backend="torch", device="cuda:0")
    state = initialize_tetra_flow_state(mesh, cfg_numpy)
    result_numpy = apply_tetra_convective_predictor(
        mesh, state, cfg_numpy, flow_dt=5e-4
    )
    result_cuda = apply_tetra_convective_predictor(mesh, state, cfg_cuda, flow_dt=5e-4)
    np.testing.assert_allclose(
        result_cuda.cell_velocity,
        result_numpy.cell_velocity,
        rtol=2e-13,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result_cuda.face_flux,
        result_numpy.face_flux,
        rtol=2e-13,
        atol=2e-15,
    )
    diag = result_cuda.diagnostics["convective_predictor"]
    assert diag["convective_torch_cuda_used"] is True


def test_viscous_predictor_dispatches_cuda_for_vectorized_slip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device="cuda:0",
        wall_velocity_boundary_mode="slip",
        viscous_predictor_mode="face_flux_laplacian_substepped",
        viscous_face_flux_laplacian_vectorized=True,
    )
    state = initialize_tetra_flow_state(mesh, cfg)
    calls: list[str] = []

    def _fake_torch_step(
        mesh_arg,
        *,
        face_flux,
        nu,
        substep_dt,
        substeps,
        divergence_impact_cap,
        device,
    ):
        calls.append(str(device))
        face_ids, nb_mat, w_mat, w_sum = (
            flow_solver_module._cached_face_flux_laplacian_vector_stencil(mesh_arg)
        )
        q = np.asarray(face_flux, dtype=np.float64).copy()
        c0 = np.asarray(mesh_arg.face_to_cells[:, 0], dtype=np.int64)
        c1 = np.asarray(mesh_arg.face_to_cells[:, 1], dtype=np.int64)
        vol = np.maximum(np.asarray(mesh_arg.cell_volumes), 1e-30)
        dq_cap = float(divergence_impact_cap) * np.minimum(
            vol[c0[face_ids]], vol[c1[face_ids]]
        )
        capped_any = np.zeros(face_ids.size, dtype=bool)
        capped_updates = 0
        for _ in range(int(substeps)):
            q_face = q[face_ids]
            lap_q = np.sum(w_mat * (q[nb_mat] - q_face[:, None]), axis=1) / np.maximum(
                w_sum, 1e-30
            )
            dq = float(nu) * float(substep_dt) * lap_q
            capped = np.abs(dq) > dq_cap
            capped_any |= capped
            capped_updates += int(np.count_nonzero(capped))
            dq = np.minimum(np.maximum(dq, -dq_cap), dq_cap)
            q_next = q.copy()
            q_next[face_ids] = q_face + dq
            q = q_next
        return (
            q,
            capped_updates,
            int(face_ids.size * int(substeps)),
            face_ids[capped_any],
        )

    monkeypatch.setattr(
        flow_solver_module,
        "_face_flux_viscous_predictor_step_torch",
        _fake_torch_step,
    )
    result = apply_tetra_stokes_viscous_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["viscous_predictor"]
    assert calls == ["cuda:0"]
    assert diag["viscous_execution_backend"] == "torch"
    assert diag["viscous_execution_device"] == "cuda:0"
    assert diag["viscous_torch_cuda_used"] is True
    assert diag["viscous_cuda_input_reused"] is False
    assert diag["viscous_cuda_finalization_used"] is False
    assert diag["viscous_numpy_fallback_reason"] == ""


@pytest.mark.parametrize("wall_mode", ("no_slip", "no_slip_tangential"))
def test_viscous_predictor_dispatches_cuda_for_conservative_no_slip(
    monkeypatch: pytest.MonkeyPatch,
    wall_mode: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device="cuda:0",
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
    )
    state = initialize_tetra_flow_state(mesh, cfg)
    calls: list[str] = []

    def _fake_torch_step(mesh_arg, *, velocity, device, **_kwargs):
        calls.append(str(device))
        n_cells = int(mesh_arg.tetrahedra.shape[0])
        n_faces = int(mesh_arg.face_vertices.shape[0])
        velocity_out = np.asarray(velocity, dtype=np.float64).copy()
        return {
            "velocity": velocity_out,
            "velocity_no_wall": velocity_out.copy(),
            "nonorthogonal_flux": np.zeros((n_faces, 3), dtype=np.float64),
            "nonorthogonal_laplacian": np.zeros((n_cells, 3), dtype=np.float64),
            "nonorthogonal_update_max": 0.0,
            "nonorthogonal_update_l2": 0.0,
            "operator_energy_rate": 0.0,
            "input_device_resident_reused": False,
        }

    monkeypatch.setattr(
        flow_solver_module,
        "_conservative_no_slip_viscous_predictor_step_torch",
        _fake_torch_step,
    )
    result = apply_tetra_stokes_viscous_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["viscous_predictor"]
    assert calls == ["cuda:0"]
    assert diag["viscous_execution_backend"] == "torch"
    assert diag["viscous_execution_device"] == "cuda:0"
    assert diag["viscous_torch_cuda_used"] is True
    assert diag["viscous_cuda_no_slip_used"] is True
    assert diag["viscous_cuda_input_reused"] is False
    assert diag["viscous_cuda_finalization_used"] is False
    assert diag["viscous_cuda_residency_scope"] == (
        "conservative_no_slip_velocity_update"
    )
    assert diag["viscous_numpy_fallback_reason"] == ""


@pytest.mark.parametrize(
    ("wall_mode", "predictor_mode", "nonorthogonal_mode", "enabled"),
    (
        (
            "no_slip",
            "explicit_cell_velocity_laplacian_substepped",
            "deferred_lsq",
            True,
        ),
        (
            "no_slip",
            "explicit_cell_velocity_laplacian_substepped_conservative",
            "none",
            True,
        ),
        (
            "no_slip_legacy_isotropic",
            "explicit_cell_velocity_laplacian_substepped_conservative",
            "none",
            True,
        ),
        (
            "no_slip",
            "explicit_cell_velocity_laplacian_substepped_conservative",
            "deferred_lsq",
            False,
        ),
    ),
)
def test_viscous_predictor_keeps_unsupported_no_slip_modes_on_numpy(
    monkeypatch: pytest.MonkeyPatch,
    wall_mode: str,
    predictor_mode: str,
    nonorthogonal_mode: str,
    enabled: bool,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device="cuda:0",
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
        viscous_predictor_mode=predictor_mode,  # type: ignore[arg-type]
        viscous_nonorthogonal_correction_mode=nonorthogonal_mode,  # type: ignore[arg-type]
        torch_cuda_viscosity_enabled=enabled,
    )
    state = initialize_tetra_flow_state(mesh, cfg)

    def _unexpected_torch_step(*_args, **_kwargs):
        raise AssertionError("unsupported no-slip mode must retain the NumPy path")

    monkeypatch.setattr(
        flow_solver_module,
        "_conservative_no_slip_viscous_predictor_step_torch",
        _unexpected_torch_step,
    )
    result = apply_tetra_stokes_viscous_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["viscous_predictor"]
    assert diag["viscous_execution_backend"] == "numpy"
    assert diag["viscous_torch_cuda_used"] is False
    assert diag["viscous_cuda_no_slip_used"] is False
    assert diag["viscous_numpy_fallback_reason"]


@pytest.mark.parametrize(
    ("device", "vectorized", "cuda_viscosity_enabled", "fallback_reason"),
    (
        ("cpu", True, True, "flow execution device is not CUDA"),
        (
            "cuda:0",
            False,
            True,
            "CUDA viscosity requires the vectorized face-flux Laplacian",
        ),
        (
            "cuda:0",
            True,
            False,
            "CUDA viscosity is disabled by configuration",
        ),
    ),
)
def test_viscous_predictor_reports_numpy_fallback_for_unsupported_torch_modes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    vectorized: bool,
    cuda_viscosity_enabled: bool,
    fallback_reason: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        backend="torch",
        device=device,
        wall_velocity_boundary_mode="slip",
        viscous_predictor_mode="face_flux_laplacian_substepped",
        viscous_face_flux_laplacian_vectorized=vectorized,
        torch_cuda_viscosity_enabled=cuda_viscosity_enabled,
    )
    state = initialize_tetra_flow_state(mesh, cfg)

    def _unexpected_torch_step(*_args, **_kwargs):
        raise AssertionError("unsupported Torch modes must retain the NumPy path")

    monkeypatch.setattr(
        flow_solver_module,
        "_face_flux_viscous_predictor_step_torch",
        _unexpected_torch_step,
    )
    result = apply_tetra_stokes_viscous_predictor(mesh, state, cfg, flow_dt=5e-4)
    diag = result.diagnostics["viscous_predictor"]
    assert diag["viscous_execution_backend"] == "numpy"
    assert diag["viscous_torch_cuda_used"] is False
    assert diag["viscous_numpy_fallback_reason"] == fallback_reason


def test_viscous_predictor_torch_cuda_matches_numpy_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = _build_synthetic_mesh()
    cfg_numpy = TetraFlowConfig(
        backend="numpy",
        device="cpu",
        wall_velocity_boundary_mode="slip",
        viscous_predictor_mode="face_flux_laplacian_substepped",
        viscous_face_flux_laplacian_vectorized=True,
    )
    cfg_cuda = replace(cfg_numpy, backend="torch", device="cuda:0")
    state = initialize_tetra_flow_state(mesh, cfg_numpy)
    result_numpy = apply_tetra_stokes_viscous_predictor(
        mesh, state, cfg_numpy, flow_dt=5e-4
    )
    result_cuda = apply_tetra_stokes_viscous_predictor(
        mesh, state, cfg_cuda, flow_dt=5e-4
    )
    np.testing.assert_allclose(
        result_cuda.cell_velocity,
        result_numpy.cell_velocity,
        rtol=2e-13,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result_cuda.face_flux,
        result_numpy.face_flux,
        rtol=2e-13,
        atol=2e-15,
    )
    diag = result_cuda.diagnostics["viscous_predictor"]
    assert diag["viscous_torch_cuda_used"] is True


@pytest.mark.parametrize("wall_mode", ("no_slip", "no_slip_tangential"))
def test_conservative_no_slip_viscosity_torch_cuda_matches_numpy_when_available(
    wall_mode: str,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = _build_synthetic_mesh()
    cfg_numpy = TetraFlowConfig(
        backend="numpy",
        device="cpu",
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
        viscous_predictor_mode="explicit_cell_velocity_laplacian_substepped_conservative",
    )
    cfg_cuda = replace(cfg_numpy, backend="torch", device="cuda:0")
    state = initialize_tetra_flow_state(mesh, cfg_numpy)
    conv_numpy = apply_tetra_convective_predictor(mesh, state, cfg_numpy, flow_dt=5e-4)
    expected = apply_tetra_stokes_viscous_predictor(
        mesh, conv_numpy, cfg_numpy, flow_dt=5e-4
    )
    conv_cuda = apply_tetra_convective_predictor(mesh, state, cfg_cuda, flow_dt=5e-4)
    actual = apply_tetra_stokes_viscous_predictor(
        mesh, conv_cuda, cfg_cuda, flow_dt=5e-4
    )

    np.testing.assert_allclose(
        actual.cell_velocity,
        expected.cell_velocity,
        rtol=5e-13,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        actual.face_flux,
        expected.face_flux,
        rtol=5e-13,
        atol=5e-15,
    )
    expected_diag = expected.diagnostics["viscous_predictor"]
    actual_diag = actual.diagnostics["viscous_predictor"]
    np.testing.assert_allclose(
        actual_diag["arrays"]["viscous_nonorthogonal_laplacian"],
        expected_diag["arrays"]["viscous_nonorthogonal_laplacian"],
        rtol=5e-13,
        atol=5e-15,
    )
    assert actual_diag["viscous_execution_backend"] == "torch"
    assert actual_diag["viscous_torch_cuda_used"] is True
    assert actual_diag["viscous_cuda_no_slip_used"] is True
    assert actual_diag["viscous_cuda_input_reused"] is True
    assert actual_diag["viscous_cuda_host_to_device_bytes_avoided"] == int(
        conv_cuda.cell_velocity.nbytes
    )
    assert actual_diag["viscous_cuda_residency_scope"] == (
        "convection_handoff_through_conservative_no_slip_velocity_update"
    )


def test_slip_cuda_predictor_chain_reuses_device_state_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = _build_synthetic_mesh()
    cfg_numpy = TetraFlowConfig(
        backend="numpy",
        device="cpu",
        enable_convective_predictor=True,
        disable_convective_auto_damping=True,
        wall_velocity_boundary_mode="slip",
        viscous_predictor_mode="face_flux_laplacian_substepped",
        viscous_face_flux_laplacian_vectorized=True,
    )
    cfg_cuda = replace(cfg_numpy, backend="torch", device="cuda:0")
    state = initialize_tetra_flow_state(mesh, cfg_numpy)

    conv_numpy = apply_tetra_convective_predictor(mesh, state, cfg_numpy, flow_dt=5e-4)
    result_numpy = apply_tetra_stokes_viscous_predictor(
        mesh, conv_numpy, cfg_numpy, flow_dt=5e-4
    )
    conv_cuda = apply_tetra_convective_predictor(mesh, state, cfg_cuda, flow_dt=5e-4)
    result_cuda = apply_tetra_stokes_viscous_predictor(
        mesh, conv_cuda, cfg_cuda, flow_dt=5e-4
    )

    np.testing.assert_allclose(
        result_cuda.cell_velocity,
        result_numpy.cell_velocity,
        rtol=3e-13,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        result_cuda.face_flux,
        result_numpy.face_flux,
        rtol=3e-13,
        atol=3e-15,
    )
    conv_diag = conv_cuda.diagnostics["convective_predictor"]
    viscous_diag = result_cuda.diagnostics["viscous_predictor"]
    assert conv_diag["convective_cuda_handoff_available"] is True
    assert viscous_diag["viscous_cuda_input_reused"] is True
    assert viscous_diag["viscous_cuda_finalization_used"] is True
    assert viscous_diag["viscous_cuda_host_to_device_bytes_avoided"] == int(
        conv_cuda.face_flux.nbytes
    )
    assert viscous_diag["viscous_cuda_cpu_reconstruction_solves_avoided"] == 3
    assert viscous_diag["viscous_cuda_residency_scope"] == (
        "convection_handoff_through_slip_boundary_and_velocity_reconstruction"
    )


def test_convective_substepping_computes_substeps_and_reports_cfl() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=80,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        kinematic_viscosity=1e-6,
        enable_convective_predictor=True,
        convective_cfl_limit=0.5,
        convective_predictor_damping=1.0,
        convective_stabilization_mode="substepping",
        convective_substep_boundary_contract="end_only",
        max_convective_substeps=256,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s_conv = apply_tetra_convective_predictor(mesh, s0, cfg, flow_dt=5e-3)
    cd = dict(s_conv.diagnostics.get("convective_predictor", {}))
    assert cd.get("convective_substepping_used", False) is True
    assert int(cd.get("convective_substep_count", 0)) >= 1
    assert (
        float(cd.get("convective_cfl_per_substep_max", 0.0))
        <= float(cfg.convective_cfl_limit) + 1e-8
    )


@pytest.mark.parametrize(
    ("wall_mode", "effective_outlet_contract"),
    (("slip", "match_inlet"), ("no_slip", "preserve")),
)
def test_convective_substep_auto_uses_effective_predictor_outlet_contract(
    wall_mode: str,
    effective_outlet_contract: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg_auto = TetraFlowConfig(
        wall_velocity_boundary_mode=wall_mode,  # type: ignore[arg-type]
        enable_convective_predictor=True,
        convective_stabilization_mode="substepping",
        convective_substep_boundary_contract="end_only",
        convective_cfl_limit=0.5,
        max_convective_substeps=256,
    )
    cfg_explicit = replace(
        cfg_auto,
        viscous_predictor_outlet_contract_mode=effective_outlet_contract,  # type: ignore[arg-type]
    )
    state_auto = initialize_tetra_flow_state(mesh, cfg_auto)
    state_explicit = initialize_tetra_flow_state(mesh, cfg_explicit)

    out_auto = apply_tetra_convective_predictor(
        mesh, state_auto, cfg_auto, flow_dt=5e-3
    )
    out_explicit = apply_tetra_convective_predictor(
        mesh, state_explicit, cfg_explicit, flow_dt=5e-3
    )

    np.testing.assert_array_equal(out_auto.face_flux, out_explicit.face_flux)
    np.testing.assert_array_equal(out_auto.cell_velocity, out_explicit.cell_velocity)
    assert (
        out_auto.diagnostics["convective_predictor"][
            "convective_predictor_outlet_contract_mode"
        ]
        == effective_outlet_contract
    )


def test_convective_predictor_emits_top_cfl_audit_payloads() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        enable_convective_predictor=True,
        convective_cfl_limit=0.5,
        convective_predictor_damping=1.0,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    s_conv = apply_tetra_convective_predictor(mesh, s0, cfg, flow_dt=5e-4)
    cd = dict(s_conv.diagnostics.get("convective_predictor", {}))
    assert "top_convective_cfl_cells" in cd
    assert "top_convective_cfl_faces" in cd
    assert "convective_cfl_definition_report" in cd
    assert isinstance(cd.get("top_convective_cfl_cells", {}), dict)
    assert isinstance(cd.get("top_convective_cfl_faces", {}), dict)
    assert isinstance(cd.get("convective_cfl_definition_report", {}), dict)


def test_convective_history_stats_and_readiness_split_debug_vs_physical() -> None:
    history = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0001,
            "final_divergence_l2": 1.0e-3,
            "final_divergence_max_abs": 1.0e-2,
            "velocity_magnitude_max": 0.2,
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 6.0,
            "convective_cfl_raw_p95": 3.0,
            "convective_cfl_effective_max": 0.4,
            "convective_cfl_effective_p95": 0.2,
            "convective_cfl_warning_raw": True,
            "convective_cfl_warning_effective": False,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 0.05,
            "convective_auto_damping_used": True,
            "convective_dt_effective": 5e-5,
        },
        {
            "step": 2,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0001,
            "final_divergence_l2": 1.0e-3,
            "final_divergence_max_abs": 1.0e-2,
            "velocity_magnitude_max": 0.2,
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 8.0,
            "convective_cfl_raw_p95": 4.0,
            "convective_cfl_effective_max": 0.45,
            "convective_cfl_effective_p95": 0.3,
            "convective_cfl_warning_raw": True,
            "convective_cfl_warning_effective": False,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 0.06,
            "convective_auto_damping_used": True,
            "convective_dt_effective": 6e-5,
        },
    ]
    stats = _convective_history_stats(history)
    assert stats["raw_cfl_warning_any"] is True
    assert stats["effective_cfl_warning_any"] is False
    assert stats["auto_damping_used_any"] is True
    readiness = _evaluate_convective_readiness(
        convective_prototype_accepted=True,
        convective_prototype_acceptance_reason="ok",
        history=history,
        flow_dt_mode="manual",
        convective_cfl_target=0.5,
    )
    assert readiness["ready_for_long_ns_run_debug"] is True
    assert readiness["ready_for_long_ns_run_physical"] is False
    assert "debug-stable only" in str(readiness["readiness_reason"])


def test_convective_history_stats_readiness_physical_when_raw_and_effective_clean() -> (
    None
):
    history = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0,
            "final_divergence_l2": 1.0e-4,
            "final_divergence_max_abs": 1.0e-3,
            "velocity_magnitude_max": 0.2,
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 0.2,
            "convective_cfl_raw_p95": 0.15,
            "convective_cfl_effective_max": 0.2,
            "convective_cfl_effective_p95": 0.15,
            "convective_cfl_warning_raw": False,
            "convective_cfl_warning_effective": False,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 1.0,
            "convective_auto_damping_used": False,
            "convective_dt_effective": 1e-3,
        }
    ]
    readiness = _evaluate_convective_readiness(
        convective_prototype_accepted=True,
        convective_prototype_acceptance_reason="ok",
        history=history,
        flow_dt_mode="manual",
        convective_cfl_target=0.5,
    )
    assert readiness["ready_for_long_ns_run_debug"] is True
    assert readiness["ready_for_long_ns_run_physical"] is True


def test_convective_readiness_can_be_physical_with_substepping_on_raw_warning() -> None:
    history = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0001,
            "final_divergence_l2": 1.0e-4,
            "final_divergence_max_abs": 5.0e-3,
            "velocity_magnitude_max": 0.25,
            "convective_stabilization_mode": "substepping",
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 3.0,
            "convective_cfl_raw_p95": 1.2,
            "convective_cfl_effective_max": 0.45,
            "convective_cfl_effective_p95": 0.2,
            "convective_cfl_warning_raw": True,
            "convective_cfl_warning_effective": False,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 1.0,
            "convective_auto_damping_used": False,
            "convective_substepping_used": True,
            "convective_substep_count": 8,
            "convective_substep_count_unclamped": 8,
            "convective_substep_cap_hit": False,
            "convective_cfl_per_substep_max": 0.45,
            "convective_cfl_per_substep_p95": 0.2,
            "convective_dt_effective": 1.25e-4,
        }
    ]
    stokes = {
        "final_divergence_l2": 1e-5,
        "final_divergence_max_abs": 1e-3,
    }
    readiness = _evaluate_convective_readiness(
        convective_prototype_accepted=True,
        convective_prototype_acceptance_reason="ok",
        history=history,
        stokes_baseline=stokes,
        max_convective_substeps=128,
        flow_dt_mode="manual",
        convective_cfl_target=0.5,
    )
    assert readiness["ready_for_long_ns_run_debug"] is True
    assert readiness["ready_for_long_ns_run_physical"] is True


def test_convective_readiness_can_be_physical_for_auto_cfl_without_auto_damping() -> (
    None
):
    history = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0001,
            "final_divergence_l2": 1.0e-4,
            "final_divergence_max_abs": 2.0e-3,
            "velocity_magnitude_max": 0.2,
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 6.0,
            "convective_cfl_raw_p95": 2.0,
            "convective_cfl_effective_max": 0.5,
            "convective_cfl_effective_p95": 0.2,
            "convective_cfl_warning_raw": True,
            "convective_cfl_warning_effective": False,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 1.0,
            "convective_auto_damping_used": False,
            "raw_cfl_max_before_dt_selection": 6.0,
            "raw_cfl_max_after_dt_selection": 0.5,
            "raw_cfl_p95_after_dt_selection": 0.2,
            "auto_dt_min_hit": False,
            "auto_dt_max_hit": False,
            "used_flow_dt": 1.5e-5,
            "auto_dt_scale_factor": 0.015,
        }
    ]
    readiness = _evaluate_convective_readiness(
        convective_prototype_accepted=True,
        convective_prototype_acceptance_reason="ok",
        history=history,
        flow_dt_mode="auto_cfl",
        convective_cfl_target=0.5,
    )
    assert readiness["ready_for_long_ns_run_debug"] is True
    assert readiness["ready_for_long_ns_run_physical"] is True


def test_convective_readiness_eps_tolerates_tiny_effective_cfl_excess() -> None:
    history = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0,
            "final_divergence_l2": 1e-4,
            "final_divergence_max_abs": 1e-3,
            "velocity_magnitude_max": 0.2,
            "convective_cfl_limit": 0.5,
            "convective_cfl_raw_max": 5.0,
            "convective_cfl_raw_p95": 2.0,
            "convective_cfl_effective_max": 0.5000000002,
            "convective_cfl_effective_p95": 0.2,
            "convective_predictor_damping_requested": 1.0,
            "convective_predictor_damping_effective": 1.0,
            "convective_auto_damping_used": False,
            "raw_cfl_max_after_dt_selection": 0.5000000002,
            "raw_cfl_p95_after_dt_selection": 0.2,
            "auto_dt_min_hit": False,
        }
    ]
    readiness = _evaluate_convective_readiness(
        convective_prototype_accepted=True,
        convective_prototype_acceptance_reason="ok",
        history=history,
        flow_dt_mode="auto_cfl",
        convective_cfl_target=0.5,
        convective_cfl_acceptance_eps=1e-9,
    )
    assert readiness["ready_for_long_ns_run_physical"] is True


def test_warning_aggregation_uses_epsilon_aware_flags() -> None:
    history = [
        {
            "step": 1,
            "convective_cfl_warning_effective": True,
            "effective_cfl_warning_with_eps": False,
            "raw_cfl_max_after_dt_selection": 0.5000000002,
            "convective_cfl_limit": 0.5,
            "raw_cfl_after_dt_selection_warning_with_eps": False,
        },
        {
            "step": 2,
            "convective_cfl_warning_effective": False,
            "effective_cfl_warning_with_eps": False,
            "raw_cfl_max_after_dt_selection": 0.49,
            "convective_cfl_limit": 0.5,
            "raw_cfl_after_dt_selection_warning_with_eps": False,
        },
    ]
    agg = _collect_warning_aggregation(history)
    assert agg["strict_warning_step_count"] == 1
    assert agg["epsilon_aware_warning_step_count"] == 0
    assert agg["warning_aggregation_mode"] == "epsilon_aware"
    assert agg["warning_aggregation_consistent"] is True


def test_nonphysical_outlet_rescale_blocks_flow_to_transport_readiness() -> None:
    policy = _collect_boundary_flux_policy(
        projection={
            "outlet_projection_mode": "outlet_mass_balance_rescale",
            "outlet_flux_rescale_used": True,
            "outlet_flux_rescale_factor": 0.9,
            "outlet_flux_rescale_reason": "rescaled for diagnostic balance",
            "nonphysical_flux_fix_used": True,
        },
        flow_mode="navier_stokes_projection_debug",
    )
    assert policy["nonphysical_flux_fix_used"] is True
    out = _evaluate_ns_coupling_readiness(
        flow_progression_solved=True,
        ready_for_long_ns_run_physical=True,
        nonphysical_flux_fix_used=True,
        convective_auto_damping_used_any=False,
        convective_substep_cap_hit_any=False,
        finite_fields=True,
        wall_flux_max_abs_after=0.0,
        outlet_inlet_flux_ratio=1.0,
        final_div_l2=0.01,
        final_div_max_abs=0.1,
        wall_flux_abs_tolerance=1e-14,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        projection_final_div_l2_tolerance=1.0,
        projection_final_div_max_tolerance=20.0,
        epsilon_aware_warning_step_count=0,
    )
    assert out["ready_for_flow_to_transport_coupling"] is False
    assert out["ns_baseline_physical_clean"] is False


def test_clean_auto_cfl_no_damping_can_be_ready_for_flow_to_transport_coupling() -> (
    None
):
    out = _evaluate_ns_coupling_readiness(
        flow_progression_solved=True,
        ready_for_long_ns_run_physical=True,
        nonphysical_flux_fix_used=False,
        convective_auto_damping_used_any=False,
        convective_substep_cap_hit_any=False,
        finite_fields=True,
        wall_flux_max_abs_after=0.0,
        outlet_inlet_flux_ratio=1.0001,
        final_div_l2=0.01,
        final_div_max_abs=0.1,
        wall_flux_abs_tolerance=1e-14,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        projection_final_div_l2_tolerance=1.0,
        projection_final_div_max_tolerance=20.0,
        epsilon_aware_warning_step_count=0,
    )
    assert out["ready_for_flow_to_transport_coupling"] is True
    assert out["ns_baseline_physical_clean"] is True


def test_large_final_divergence_blocks_flow_to_transport_readiness() -> None:
    out = _evaluate_ns_coupling_readiness(
        flow_progression_solved=True,
        ready_for_long_ns_run_physical=True,
        nonphysical_flux_fix_used=False,
        convective_auto_damping_used_any=False,
        convective_substep_cap_hit_any=False,
        finite_fields=True,
        wall_flux_max_abs_after=0.0,
        outlet_inlet_flux_ratio=1.0,
        final_div_l2=12066.0,
        final_div_max_abs=739965.0,
        wall_flux_abs_tolerance=1e-14,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        projection_final_div_l2_tolerance=1.0,
        projection_final_div_max_tolerance=20.0,
        epsilon_aware_warning_step_count=0,
    )
    assert out["ready_for_flow_to_transport_coupling"] is False
    checks = out["ns_baseline_physical_clean_checks"]
    assert checks["final_div_l2_ok"] is False
    assert checks["final_div_max_ok"] is False


def test_flow_coupling_export_writes_required_files(tmp_path: Path) -> None:
    mesh = _build_synthetic_mesh()
    face_flux = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    velocity = np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64)
    pressure = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    face_codes = _face_group_codes(mesh)
    out = _export_flow_coupling_bundle(
        run_dir=tmp_path,
        mesh=mesh,
        mesh_name="synthetic",
        mesh_sha256="a" * 64,
        face_flux=face_flux,
        cell_velocity=velocity,
        pressure=pressure,
        face_group_codes=face_codes,
        flow_mode="navier_stokes_projection_debug",
        flow_dt_mode="auto_cfl",
        flow_steps=10,
        physical_time_final=1e-3,
        run_completed=True,
        numerically_stable=True,
        physically_ready=True,
        ready_for_next_stage=True,
        ready_for_long_run=True,
        stage_status_reason="unit-test",
        ready_for_flow_to_transport_coupling=True,
        ns_baseline_physical_clean=True,
        outlet_flux_rescale_used=False,
        nonphysical_flux_fix_used=False,
        convective_auto_damping_used_any=False,
        wall_flux_max_abs_after=0.0,
        outlet_inlet_flux_ratio=1.0,
        final_div_l2=1e-6,
        final_div_max=1e-5,
    )
    artifacts = dict(out.get("artifacts", {}))
    required = [
        "flow_coupling_metadata_json",
        "final_corrected_face_flux_npy",
        "final_cell_velocity_npy",
        "final_pressure_npy",
        "face_centers_npy",
        "face_normals_npy",
        "face_to_cells_npy",
        "cell_centers_npy",
        "cell_volumes_npy",
        "face_groups_npy",
    ]
    for key in required:
        assert key in artifacts
        assert Path(str(artifacts[key])).exists()
    metadata = json.loads(
        Path(str(artifacts["flow_coupling_metadata_json"])).read_text(encoding="utf-8")
    )
    assert metadata["ready_for_flow_to_transport_coupling"] is True
    assert metadata["mesh_sha256"] == "a" * 64
    assert metadata["stage_status"]["ready_for_next_stage"] is True
    assert metadata["stage_status"]["stage_status_reason"] == "unit-test"
    assert metadata["convective_auto_damping_used_any"] is False
    artifact_hashes = out["artifact_sha256"]
    assert set(artifact_hashes) == {
        "flow_coupling_metadata_json",
        "final_corrected_face_flux_npy",
        "face_to_cells_npy",
        "cell_volumes_npy",
    }
    for artifact_name, sha256_hex in artifact_hashes.items():
        assert len(sha256_hex) == 64
        assert artifact_name in artifacts


def test_load_flow_resume_state_restores_time_step_and_arrays(
    tmp_path: Path,
) -> None:
    mesh = _build_synthetic_mesh()
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    step = 1250
    physical_time = 0.020196383976679833
    face_flux = np.linspace(0.0, 1.0, mesh.face_vertices.shape[0])
    velocity = np.arange(mesh.tetrahedra.shape[0] * 3, dtype=np.float64).reshape(-1, 3)
    pressure = np.linspace(-2.0, 3.0, mesh.tetrahedra.shape[0])
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "flow_steps_completed": 250,
                "flow_steps_completed_total": step,
                "physical_time_final": physical_time,
            }
        ),
        encoding="utf-8",
    )
    np.save(run_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(run_dir / "final_cell_velocity.npy", velocity)
    np.save(run_dir / "final_pressure.npy", pressure)
    manifest = _finalize_flow_resume_manifest(
        _resume_manifest(),
        run_dir=run_dir,
        completed_step=step,
        physical_time=physical_time,
    )
    (run_dir / "resume_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    state, restored_step, restored_time, metadata = _load_flow_resume_state(
        mesh,
        run_dir,
        expected_manifest=manifest,
    )

    assert restored_step == step
    assert restored_time == pytest.approx(physical_time)
    np.testing.assert_array_equal(state.face_flux, face_flux)
    np.testing.assert_array_equal(state.cell_velocity, velocity)
    np.testing.assert_array_equal(state.pressure, pressure)
    assert metadata["source_flow_steps_completed"] == step
    assert metadata["source_physical_time_final"] == pytest.approx(physical_time)
    assert state.diagnostics["resume"] == metadata


def test_mesh_npz_fingerprint_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    changed = tmp_path / "changed.npz"
    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    cells = np.arange(8, dtype=np.int64).reshape(2, 4)
    np.savez_compressed(first, points=points, cells=cells)
    np.savez_compressed(second, cells=cells, points=points)
    np.savez_compressed(changed, points=points, cells=cells[::-1])

    assert _mesh_npz_fingerprint(first) == _mesh_npz_fingerprint(second)
    assert _mesh_npz_fingerprint(first) != _mesh_npz_fingerprint(changed)


def test_source_runtime_fingerprint_includes_solver_modules(tmp_path: Path) -> None:
    solver = tmp_path / "compute" / "src" / "microfluidics" / "solver.py"
    runner = tmp_path / "experiments" / "gmsh" / "runner.py"
    solver.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    solver.write_text("VALUE = 1\n", encoding="utf-8")
    runner.write_text("RUNNER = 1\n", encoding="utf-8")
    original = _source_runtime_fingerprint(tmp_path)

    solver.write_text("VALUE = 2\n", encoding="utf-8")

    assert _source_runtime_fingerprint(tmp_path) != original


def test_resume_manifest_rejects_flow_mode_change_with_same_solver_config(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "mesh.npz"
    np.savez(mesh_path, points=np.zeros((4, 3)), tetrahedra=np.zeros((1, 4)))
    arguments = {
        "mesh_npz": mesh_path,
        "cfg": TetraFlowConfig(),
        "source_sha256": "a" * 64,
        "runtime_identifier": "example-runtime@sha256:" + "1" * 64,
        "input_mesh_sha256": "b" * 64,
        "request_fingerprint": "c" * 64,
        "flow_dt_mode": "auto_cfl",
        "requested_flow_dt": 1e-4,
        "flow_dt_min": 1e-7,
        "flow_dt_max": 1e-4,
        "convective_cfl_target": 0.45,
        "wall_strength_ramp_start": 1.0,
        "wall_strength_ramp_target": 1.0,
        "wall_strength_ramp_steps": 0,
    }
    projection_only = _build_flow_resume_manifest(
        flow_mode="projection_only",
        **arguments,
    )
    stokes = _build_flow_resume_manifest(
        flow_mode="stokes_viscous_projection",
        **arguments,
    )

    assert projection_only["flow_config"]["flow_mode"] == "projection_only"
    with pytest.raises(ValueError, match="flow_config"):
        _validate_flow_resume_manifest(projection_only, stokes)


def test_direct_resume_requires_explicit_complete_source_digest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments/gmsh/run_gmsh_tetra_flow_debug.py"),
            "--resume-flow-run-dir",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--resume-flow-run-dir requires an explicit --run-source-sha256" in (
        result.stderr
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("summary", "completed step"),
        ("state", "final_pressure.npy"),
    ],
)
def test_load_flow_resume_state_rejects_mixed_checkpoint(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    mesh = _build_synthetic_mesh()
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    step = 50
    physical_time = 0.25
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "flow_steps_completed_total": step,
                "physical_time_final": physical_time,
            }
        ),
        encoding="utf-8",
    )
    np.save(
        run_dir / "final_corrected_face_flux.npy",
        np.zeros(mesh.face_vertices.shape[0]),
    )
    np.save(
        run_dir / "final_cell_velocity.npy",
        np.zeros((mesh.tetrahedra.shape[0], 3)),
    )
    np.save(run_dir / "final_pressure.npy", np.zeros(mesh.tetrahedra.shape[0]))
    manifest = _finalize_flow_resume_manifest(
        _resume_manifest(),
        run_dir=run_dir,
        completed_step=step,
        physical_time=physical_time,
    )
    (run_dir / "resume_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    if tamper == "summary":
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "flow_steps_completed_total": 999999,
                    "physical_time_final": 12345.0,
                }
            ),
            encoding="utf-8",
        )
    else:
        np.save(
            run_dir / "final_pressure.npy",
            np.ones(mesh.tetrahedra.shape[0]),
        )

    with pytest.raises(ValueError, match=message):
        _load_flow_resume_state(
            mesh,
            run_dir,
            expected_manifest=_resume_manifest(),
        )


@pytest.mark.parametrize(
    ("recorded_manifest", "message"),
    [
        (_resume_manifest(mesh_sha256="e" * 64), "mesh_sha256"),
        (
            _resume_manifest(flow_config={"pressure_solver": "jacobi"}),
            "flow_config",
        ),
    ],
)
def test_load_flow_resume_state_rejects_incompatible_provenance_with_matching_shapes(
    tmp_path: Path,
    recorded_manifest: dict,
    message: str,
) -> None:
    mesh = _build_synthetic_mesh()
    run_dir = tmp_path / "resume_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps({"flow_steps_completed": 10, "physical_time_final": 1e-3}),
        encoding="utf-8",
    )
    (run_dir / "resume_manifest.json").write_text(
        json.dumps(recorded_manifest),
        encoding="utf-8",
    )
    np.save(
        run_dir / "final_corrected_face_flux.npy",
        np.zeros(mesh.face_vertices.shape[0]),
    )
    np.save(
        run_dir / "final_cell_velocity.npy",
        np.zeros((mesh.tetrahedra.shape[0], 3)),
    )
    np.save(run_dir / "final_pressure.npy", np.zeros(mesh.tetrahedra.shape[0]))

    with pytest.raises(ValueError, match=message):
        _load_flow_resume_state(
            mesh,
            run_dir,
            expected_manifest=_resume_manifest(),
        )


def test_convective_auto_damping_marks_stabilization_non_physical_baseline() -> None:
    conv_stats = {
        "auto_damping_step_count": 5,
        "auto_damping_used_any": True,
        "damping_effective_min": 0.1,
        "damping_effective_mean": 0.2,
        "damping_effective_max": 1.0,
        "substepping_used_any": False,
        "substep_cap_hit_any": False,
    }
    history = [{"capped_predictor_updates_fraction": 0.0}]
    stab = _collect_stabilization_audit(conv_stats=conv_stats, history=history)
    assert stab["convective_auto_damping_used_any"] is True
    assert stab["stabilization_is_physical_baseline"] is False
    assert "convective auto damping used" in stab["stabilization_debug_only_reasons"]


def test_evaluate_viscous_progression_acceptance_uses_projection_baseline() -> None:
    hist = [
        {
            "step": 1,
            "finite_fields": True,
            "wall_flux_max_abs_after": 0.0,
            "outlet_inlet_flux_ratio": 1.0002,
            "final_divergence_l2": 8e-1,
            "final_divergence_max_abs": 15.0,
            "velocity_magnitude_max": 0.2,
        }
    ]
    baseline = {"velocity_magnitude_max_final": 0.15}
    out = _evaluate_viscous_progression_acceptance(
        history=hist,
        projection_only_baseline=baseline,
        outlet_inlet_flux_ratio_tolerance=5e-3,
        wall_flux_abs_tolerance=1e-14,
        projection_final_div_l2_tolerance=1.0,
        projection_final_div_max_tolerance=20.0,
        velocity_blowup_ratio_tolerance=2.5,
    )
    assert out["viscous_progression_accepted"] is True
    assert out["checks"]["velocity_blowup_ok"] is True


def test_parse_viscous_predictor_modes_basic() -> None:
    assert _parse_viscous_predictor_modes(
        "none,no_viscous_debug_copy,explicit_cell_velocity_laplacian_substepped,face_flux_laplacian_substepped"
    ) == [
        "none",
        "no_viscous_debug_copy",
        "explicit_cell_velocity_laplacian_substepped",
        "face_flux_laplacian_substepped",
    ]


def test_parse_convective_stabilization_modes_basic() -> None:
    assert _parse_convective_stabilization_modes(
        "auto_damping,substepping,foo,substepping"
    ) == ["auto_damping", "substepping"]


def test_parse_flow_dt_mode_basic() -> None:
    assert _parse_flow_dt_mode("manual") == "manual"
    assert _parse_flow_dt_mode("auto_cfl") == "auto_cfl"
    assert _parse_flow_dt_mode("AUTO_CFL") == "auto_cfl"
    assert _parse_flow_dt_mode("bad_mode") == "manual"


def test_auto_cfl_dt_selection_clips_to_dt_max_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(inlet_speed=0.15)
    s0 = initialize_tetra_flow_state(mesh, cfg)
    rate = compute_tetra_convective_cfl_rate(
        mesh, np.asarray(s0.face_flux, dtype=np.float64)
    )
    rate_max = float(rate.get("cfl_rate_max", 0.0))
    target = 0.5
    dt_min = 1e-7
    dt_max = 1e-3
    dt_raw = target / max(rate_max, 1e-30) if rate_max > 0.0 else dt_max
    dt_used = min(max(dt_raw, dt_min), dt_max)
    assert dt_used <= dt_max + 1e-30
    assert dt_used >= dt_min - 1e-30
    raw_after = dt_used * rate_max
    assert raw_after <= target + 1e-8
    assert _parse_viscous_predictor_modes("foo,none,foo") == ["none"]
    assert _parse_viscous_predictor_modes(
        "none,no_viscous_debug_copy,explicit_cell_velocity_laplacian_substepped,face_flux_laplacian_substepped"
    ) == [
        "none",
        "no_viscous_debug_copy",
        "explicit_cell_velocity_laplacian_substepped",
        "face_flux_laplacian_substepped",
    ]


def test_no_viscous_debug_copy_close_to_projection_only_synthetic() -> None:
    mesh = _build_synthetic_mesh()
    cfg_proj = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=120,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        viscous_predictor_mode="none",
    )
    s0 = initialize_tetra_flow_state(mesh, cfg_proj)
    s_proj = solve_tetra_pressure_projection(mesh, s0, cfg_proj)

    cfg_copy = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=120,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        viscous_predictor_mode="no_viscous_debug_copy",
    )
    s_pred = apply_tetra_stokes_viscous_predictor(mesh, s0, cfg_copy, flow_dt=5e-4)
    s_stokes = solve_tetra_pressure_projection(mesh, s_pred, cfg_copy)
    diff = np.max(
        np.abs(
            np.asarray(s_proj.face_flux, dtype=np.float64)
            - np.asarray(s_stokes.face_flux, dtype=np.float64)
        )
    )
    assert diff <= 1e-10


def test_viscous_predictor_audit_contains_stage_metrics() -> None:
    history = [
        {
            "step": 1,
            "flow_mode": "stokes_viscous_projection",
            "viscous_predictor_mode": "explicit_cell_velocity_laplacian_substepped",
            "viscous_predictor_used": True,
            "divergence_before_predictor_max": 10.0,
            "divergence_before_predictor_l2": 2.0,
            "divergence_after_predictor_before_boundary_contract_max": 12.0,
            "divergence_after_predictor_before_boundary_contract_l2": 2.2,
            "divergence_after_boundary_contract_before_projection_max": 11.0,
            "divergence_after_boundary_contract_before_projection_l2": 2.1,
            "divergence_after_projection_max": 1.0,
            "divergence_after_projection_l2": 0.2,
            "net_boundary_flux_before_predictor": 0.0,
            "net_boundary_flux_after_predictor_before_contract": 1e-9,
            "net_boundary_flux_after_contract": 1e-10,
            "net_boundary_flux_after_projection": 1e-12,
            "wall_flux_max_after_predictor_before_contract": 1e-10,
            "wall_flux_max_after_contract": 0.0,
            "wall_flux_max_after_projection": 0.0,
            "outlet_inlet_ratio_before_predictor": 1.0,
            "outlet_inlet_ratio_after_predictor_before_contract": 0.9,
            "outlet_inlet_ratio_after_contract": 1.0,
            "outlet_inlet_flux_ratio": 1.0,
            "kinetic_energy_before_predictor": 0.5,
            "kinetic_energy_after_predictor": 0.45,
            "kinetic_energy_after_contract": 0.46,
            "kinetic_energy_after_projection": 0.47,
            "viscous_delta_velocity_max": 0.1,
            "viscous_delta_velocity_l2": 0.02,
            "face_flux_delta_predictor_max": 1e-7,
            "face_flux_delta_predictor_l2": 1e-8,
            "face_flux_delta_contract_max": 2e-7,
            "face_flux_delta_contract_l2": 2e-8,
            "wall_velocity_boundary_mode": "no_slip",
            "wall_velocity_boundary_implementation": "tangential_zero_velocity",
            "wall_tangential_no_slip_strength": 1.0,
            "wall_tangential_shear_face_flux_requested": True,
            "wall_tangential_cell_velocity_momentum_enabled": False,
            "wall_tangential_operator_active_cells": 4,
            "wall_tangential_operator_max_abs": 3.0,
            "wall_tangential_operator_trace_mean": 5.0,
            "wall_tangential_operator_trace_max": 7.0,
            "wall_tangential_operator_effective_nu_dt_max_abs": 1e-4,
            "wall_tangential_operator_effective_nu_subdt_max_abs": 5e-5,
            "wall_flux_stokes_resistance_enabled": False,
            "wall_flux_stokes_resistance_strength": 1.0,
            "wall_flux_stokes_resistance_active_faces": 0,
            "wall_flux_stokes_resistance_solver_iterations": 0,
            "wall_flux_stokes_resistance_solver_converged": True,
            "wall_flux_stokes_resistance_solver_residual_l2": 0.0,
            "wall_flux_stokes_resistance_solver_method": "disabled",
            "wall_tangential_shear_face_flux_enabled": True,
            "wall_tangential_shear_face_flux_applications": 1,
            "wall_tangential_shear_face_flux_active_cells": 4,
            "wall_tangential_shear_face_flux_delta_l2": 1e-9,
            "wall_tangential_shear_face_flux_wall_speed_mean_before": 0.1,
            "wall_tangential_shear_face_flux_wall_speed_mean_after": 0.05,
        }
    ]
    audit = _build_viscous_predictor_audit_from_history(history)
    assert "steps" in audit and len(audit["steps"]) == 1
    row = audit["steps"][0]
    assert "divergence_after_predictor_before_boundary_contract_max" in row
    assert "face_flux_delta_contract_l2" in row
    assert row["wall_tangential_shear_face_flux_enabled"] is True
    assert row["wall_tangential_shear_face_flux_requested"] is True
    assert row["wall_tangential_cell_velocity_momentum_enabled"] is False
    assert row["wall_tangential_operator_active_cells"] == 4
    assert row["wall_tangential_operator_effective_nu_dt_max_abs"] == 1e-4
    assert "summary" in audit


def test_stokes_ready_false_when_divergence_degradation_large() -> None:
    ok = _evaluate_stokes_ready_for_advection(
        damage_ratio_l2=20.0,
        damage_ratio_linf=50.0,
        predictor_damages_divergence=True,
    )
    assert ok is False


def test_face_flux_predictor_preserves_wall_and_boundary_contract_after_projection() -> (
    None
):
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        pressure_solver="pcg_diag",
        max_pressure_iterations=120,
        pressure_relative_tolerance=1e-3,
        projection_dt=5e-4,
        viscous_predictor_mode="face_flux_laplacian_substepped",
        viscous_face_flux_divergence_impact_cap=1.0,
    )
    s0 = initialize_tetra_flow_state(mesh, cfg)
    sp = apply_tetra_stokes_viscous_predictor(mesh, s0, cfg, flow_dt=5e-4)
    s1 = solve_tetra_pressure_projection(mesh, sp, cfg)
    diag = compute_tetra_flux_divergence(
        mesh,
        np.asarray(s1.face_flux, dtype=np.float64),
        left_inlet_faces=np.asarray(
            resolve_inlet_face_groups(mesh)["left_faces"], dtype=np.int64
        ),
        right_inlet_faces=np.asarray(
            resolve_inlet_face_groups(mesh)["right_faces"], dtype=np.int64
        ),
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
    )
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall.size:
        assert (
            float(np.max(np.abs(np.asarray(s1.face_flux, dtype=np.float64)[wall])))
            <= 1e-12
        )
    inlet = float(diag.get("inlet_flux_total", 0.0))
    outlet = float(diag.get("outlet_flux_total", 0.0))
    assert abs(outlet / max(abs(inlet), 1e-30) - 1.0) <= 5e-3


def test_expected_artifacts_map_contains_required_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="stokes_viscous_projection",
        flow_steps=100,
        compare_viscous_predictor_modes=True,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=True,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
    )
    required = [
        run_dir / "divergence_stage_comparison_step_0100.png",
        run_dir / "velocity_magnitude_before_after_predictor_step_0100.png",
        run_dir / "face_flux_delta_predictor_xy_step_0100.png",
        run_dir / "velocity_vectors_xy_grid_binned_step_0100.png",
        run_dir / "velocity_magnitude_p95_clipped_xy_step_0100.png",
        run_dir / "viscous_predictor_audit.json",
        run_dir / "viscous_predictor_mode_comparison.json",
        run_dir / "stokes_baseline_sensitivity_sweep.json",
        run_dir / "stokes_baseline_sensitivity_sweep.csv",
        run_dir / "flow_progression_history.json",
        run_dir / "summary.json",
    ]
    for p in required:
        assert str(p) in amap
    assert amap[str(run_dir / "summary.json")] is True


def test_expected_artifacts_map_is_mode_aware_for_comparison_only_run(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_cmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="projection_only",
        flow_steps=100,
        compare_viscous_predictor_modes=True,
        compare_flow_modes=True,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
    )
    assert str(run_dir / "viscous_predictor_mode_comparison.json") in amap
    assert str(run_dir / "flow_mode_comparison.json") in amap
    assert str(run_dir / "divergence_stage_comparison_step_0100.png") not in amap


def test_expected_artifacts_map_omits_visualizations_in_minimal_mode(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_minimal"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="navier_stokes_projection_debug",
        flow_steps=100,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
        postprocessing_mode="minimal",
    )
    assert str(run_dir / "divergence_stage_comparison_step_0100.png") not in amap
    assert str(run_dir / "velocity_vectors_xy_grid_binned_step_0100.png") not in amap
    assert str(run_dir / "summary.json") in amap
    assert str(run_dir / "acceptance_report.json") in amap


def test_minimal_postprocessing_plot_guard_skips_plot_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flow_debug_module, "_POSTPROCESSING_MODE", "minimal")
    assert flow_debug_module._save_scatter() is None


def test_minimal_postprocessing_disables_visualization_artifact_copies() -> None:
    assert flow_debug_module._postprocessing_writes_visualizations("minimal") is False
    assert flow_debug_module._postprocessing_writes_visualizations("full") is True


def test_full_postprocessing_plot_guard_preserves_plot_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _plot_body(*args, **kwargs):
        calls.append((args, kwargs))
        return "plot.png"

    monkeypatch.setattr(flow_debug_module, "_POSTPROCESSING_MODE", "full")
    guarded = flow_debug_module._guard_plot_output(_plot_body)
    assert guarded("value", out_path="plot.png") == "plot.png"
    assert calls == [(("value",), {"out_path": "plot.png"})]


def test_expected_artifacts_map_includes_navier_audit_for_navier_mode(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_navier"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="navier_stokes_projection_debug",
        flow_steps=100,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
    )
    assert str(run_dir / "navier_stokes_prototype_audit.json") in amap


def test_expected_artifacts_map_includes_convective_sweep_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_conv_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="navier_stokes_projection_debug",
        flow_steps=20,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=True,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
    )
    assert str(run_dir / "convective_sensitivity_sweep.json") in amap
    assert str(run_dir / "convective_sensitivity_sweep.csv") in amap


def test_expected_artifacts_map_includes_convective_cfl_audit_and_stabilization_compare(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_conv_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="navier_stokes_projection_debug",
        flow_steps=20,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=True,
        compare_convective_stabilization_modes=True,
        compare_ns_dt_modes=False,
    )
    assert str(run_dir / "top_convective_cfl_cells.json") in amap
    assert str(run_dir / "top_convective_cfl_faces.json") in amap
    assert str(run_dir / "convective_cfl_definition_report.json") in amap
    assert str(run_dir / "convective_stabilization_comparison.json") in amap


def test_expected_artifacts_map_includes_ns_dt_mode_comparison(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_ns_dt_cmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="navier_stokes_projection_debug",
        flow_steps=20,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
        compare_ns_dt_modes=True,
    )
    assert str(run_dir / "ns_dt_mode_comparison.json") in amap


def test_expected_artifacts_map_includes_projection_contract_runtime_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_projection_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    amap = _build_expected_artifacts_map(
        run_dir,
        flow_mode="stokes_projection",
        flow_steps=20,
        compare_viscous_predictor_modes=False,
        compare_flow_modes=False,
        compare_pressure_solvers=False,
        run_stokes_sensitivity_sweep=False,
        run_convective_sensitivity_sweep=False,
        audit_convective_cfl=False,
        compare_convective_stabilization_modes=False,
        compare_ns_dt_modes=False,
    )
    assert str(run_dir / "acceptance_report.json") in amap
    assert str(run_dir / "flow_diagnostics.json") in amap
    assert str(run_dir / "startup_bootstrap_history.json") in amap
    assert str(run_dir / "startup_root_cause_report.json") in amap


def test_projection_boundary_contract_runtime_payload_preserves_stage_contract() -> (
    None
):
    diag = {
        "projection_pinned_face_constraints": {
            "raw_pre_constraint": {"pinned_face_count": 3}
        },
        "projection_correction_boundary_contract": {
            "raw_pre_constraint": {"walls": {"max_abs": 0.0}}
        },
        "projection_correction_stage_codebook": {"raw_pre_constraint": "raw stage"},
        "face_flux_primary_stage_codebook": {
            "correction_flux_raw_pre_constraint": "raw stage"
        },
    }

    payload = _projection_boundary_contract_runtime_payload(diag)

    assert (
        payload["projection_pinned_face_constraints"]["raw_pre_constraint"][
            "pinned_face_count"
        ]
        == 3
    )
    assert (
        payload["projection_correction_boundary_contract"]["raw_pre_constraint"][
            "walls"
        ]["max_abs"]
        == 0.0
    )
    assert payload["projection_correction_stage_codebook"]["raw_pre_constraint"] == (
        "raw stage"
    )
    assert (
        payload["face_flux_primary_stage_codebook"][
            "correction_flux_raw_pre_constraint"
        ]
        == "raw stage"
    )


def test_flow_diagnostics_final_sync_uses_acceptance_summary_truth() -> None:
    stale_flow_diag = {
        "ready_for_long_ns_run": False,
        "ready_for_long_ns_run_debug": False,
        "ready_for_long_ns_run_physical": False,
        "ns_auto_dt_accepted": False,
        "raw_cfl_after_dt_selection_max": 0.0,
        "raw_cfl_after_dt_selection_p95": 0.0,
        "effective_cfl_limit_excess_max": 99.0,
    }
    summary = {
        "ready_for_long_ns_run": True,
        "ready_for_long_ns_run_debug": True,
        "ready_for_long_ns_run_physical": True,
        "ns_auto_dt_accepted": True,
        "raw_cfl_after_dt_selection_max": 0.5,
        "raw_cfl_after_dt_selection_p95": 0.12,
        "effective_cfl_limit_excess_max": 1e-16,
    }
    acceptance_report = {
        "ready_for_long_ns_run": True,
        "ready_for_long_ns_run_debug": True,
        "ready_for_long_ns_run_physical": True,
        "ns_auto_dt_accepted": True,
        "raw_cfl_after_dt_selection_max": 0.5,
        "raw_cfl_after_dt_selection_p95": 0.12,
    }

    synced = _sync_flow_diagnostics_with_final_artifacts(
        stale_flow_diag,
        acceptance_report=acceptance_report,
        summary=summary,
    )

    for key in (
        "ready_for_long_ns_run",
        "ready_for_long_ns_run_debug",
        "ready_for_long_ns_run_physical",
        "ns_auto_dt_accepted",
        "raw_cfl_after_dt_selection_max",
        "raw_cfl_after_dt_selection_p95",
        "effective_cfl_limit_excess_max",
    ):
        assert synced[key] == summary[key]
    assert synced["final_artifact_source_of_truth"] == "acceptance_report"


def test_pressure_nonorthogonal_default_none_matches_explicit_none_bitwise() -> None:
    mesh = _build_synthetic_mesh()
    common = {
        "inlet_speed": 0.15,
        "projection_dt": 5e-4,
        "max_pressure_iterations": 120,
        "pressure_relative_tolerance": 1e-6,
        "pressure_solver": "pcg_diag",
        "enable_sign_comparison": False,
        "backend": "numpy",
        "pressure_projection_outlet_contract_mode": "preserve",
    }
    cfg_default = TetraFlowConfig(**common)  # type: ignore[arg-type]
    cfg_explicit = TetraFlowConfig(
        **common,  # type: ignore[arg-type]
        pressure_nonorthogonal_correction_mode="none",
    )
    rng = np.random.default_rng(20260722)
    state = TetraFlowState(
        cell_velocity=rng.standard_normal((mesh.tetrahedra.shape[0], 3)),
        face_flux=rng.standard_normal(mesh.face_vertices.shape[0]),
        pressure=rng.standard_normal(mesh.tetrahedra.shape[0]),
        diagnostics={},
    )

    out_default = solve_tetra_pressure_projection(mesh, state, cfg_default)
    out_explicit = solve_tetra_pressure_projection(mesh, state, cfg_explicit)

    np.testing.assert_array_equal(out_default.pressure, out_explicit.pressure)
    np.testing.assert_array_equal(out_default.face_flux, out_explicit.face_flux)
    np.testing.assert_array_equal(
        out_default.cell_velocity,
        out_explicit.cell_velocity,
    )
    nonorth = dict(out_default.diagnostics["pressure_nonorthogonal_correction"])
    assert nonorth["mode"] == "none"
    assert nonorth["enabled"] is False
    assert nonorth["actual_sweeps"] == 0


def test_pressure_nonorthogonal_lsq_constant_and_linear_manufactured_fields() -> None:
    mesh = _build_synthetic_mesh()
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    geometry = flow_solver_module._cached_pressure_nonorthogonal_geometry(
        mesh,
        outlet_faces,
    )
    dt = 0.2
    density = 1.7
    pressure_outlet_value = 1.75

    pressure_constant = np.full(
        mesh.tetrahedra.shape[0],
        pressure_outlet_value,
        dtype=np.float64,
    )
    constant_nonorth, constant_gradient = (
        flow_solver_module._pressure_nonorthogonal_gradient_flux_numpy(
            mesh,
            pressure_constant,
            dt=dt,
            density=density,
            pressure_outlet_value=pressure_outlet_value,
            geometry=geometry,
        )
    )
    constant_full = flow_solver_module._pressure_face_gradient_flux(
        mesh,
        pressure_constant,
        dt=dt,
        density=density,
        outlet_faces=outlet_faces,
        pressure_outlet_value=pressure_outlet_value,
        frozen_nonorthogonal_gradient_flux=constant_nonorth,
    )
    np.testing.assert_allclose(constant_gradient, 0.0, atol=1e-14)
    np.testing.assert_allclose(constant_nonorth, 0.0, atol=1e-14)
    np.testing.assert_allclose(constant_full, 0.0, atol=1e-14)

    exact_gradient = np.asarray([0.0, 2.5, 0.0], dtype=np.float64)
    outlet_plane_point = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    pressure_linear = (
        pressure_outlet_value
        + (np.asarray(mesh.cell_centers, dtype=np.float64) - outlet_plane_point)
        @ exact_gradient
    )
    linear_nonorth, reconstructed_gradient = (
        flow_solver_module._pressure_nonorthogonal_gradient_flux_numpy(
            mesh,
            pressure_linear,
            dt=dt,
            density=density,
            pressure_outlet_value=pressure_outlet_value,
            geometry=geometry,
        )
    )
    orthogonal_flux = flow_solver_module._pressure_face_gradient_flux(
        mesh,
        pressure_linear,
        dt=dt,
        density=density,
        outlet_faces=outlet_faces,
        pressure_outlet_value=pressure_outlet_value,
    )
    full_flux = orthogonal_flux + linear_nonorth
    active_faces = np.unique(
        np.concatenate(
            (
                np.asarray(mesh.interior_face_indices, dtype=np.int64),
                outlet_faces,
            )
        )
    )
    exact_flux = (
        (dt / density)
        * np.asarray(mesh.face_areas, dtype=np.float64)
        * np.einsum(
            "ij,j->i",
            np.asarray(mesh.face_normals, dtype=np.float64),
            exact_gradient,
            optimize=True,
        )
    )
    orthogonal_error = float(
        np.sqrt(
            np.mean((orthogonal_flux[active_faces] - exact_flux[active_faces]) ** 2)
        )
    )
    corrected_error = float(
        np.sqrt(np.mean((full_flux[active_faces] - exact_flux[active_faces]) ** 2))
    )
    sufficiently_constrained = (
        np.asarray(geometry["lsq_equation_count"], dtype=np.int32) >= 3
    )
    # The linear axial field intentionally does not satisfy homogeneous
    # pressure-correction Neumann data on the inlet.  Exactness is therefore
    # asserted only on cells whose LSQ system contains no non-outlet boundary
    # equation; boundary-adjacent cells correctly honor the projection BC.
    sufficiently_constrained[
        np.unique(np.asarray(geometry["neumann_owner"], dtype=np.int64))
    ] = False
    assert np.any(sufficiently_constrained)
    np.testing.assert_allclose(
        reconstructed_gradient[sufficiently_constrained],
        np.broadcast_to(
            exact_gradient,
            reconstructed_gradient[sufficiently_constrained].shape,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    assert orthogonal_error > 0.0
    assert corrected_error < 0.45 * orthogonal_error


def test_pressure_nonorthogonal_rhs_term_respects_source_units() -> None:
    mesh = _build_synthetic_mesh()
    face_flux = np.linspace(
        -0.025,
        0.035,
        mesh.face_vertices.shape[0],
        dtype=np.float64,
    )
    integrated = _compute_cell_flux_sum(mesh, face_flux)
    as_integrated = flow_solver_module._pressure_nonorthogonal_rhs_term(
        mesh,
        face_flux,
        rhs_mode="volume_integrated_flux",
    )
    as_divergence = flow_solver_module._pressure_nonorthogonal_rhs_term(
        mesh,
        face_flux,
        rhs_mode="divergence_per_volume",
    )

    np.testing.assert_array_equal(as_integrated, integrated)
    np.testing.assert_allclose(
        as_divergence,
        integrated / np.asarray(mesh.cell_volumes, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_pressure_nonorthogonal_torch_operators_match_numpy_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    mesh = _build_synthetic_mesh()
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    geometry = flow_solver_module._cached_pressure_nonorthogonal_geometry(
        mesh, outlet_faces
    )
    geometry_t = flow_solver_module._cached_pressure_nonorthogonal_geometry_torch(
        mesh, geometry, device="cpu"
    )
    rng = np.random.default_rng(20260807)
    pressure = rng.normal(size=mesh.tetrahedra.shape[0])
    dt = 5e-4
    density = 997.0
    outlet_value = 0.25

    expected_flux, expected_gradient = (
        flow_solver_module._pressure_nonorthogonal_gradient_flux_numpy(
            mesh,
            pressure,
            dt=dt,
            density=density,
            pressure_outlet_value=outlet_value,
            geometry=geometry,
        )
    )
    actual_flux_t, actual_gradient_t = (
        flow_solver_module._pressure_nonorthogonal_gradient_flux_torch(
            torch.as_tensor(pressure, dtype=torch.float64),
            n_faces=mesh.face_vertices.shape[0],
            dt=dt,
            density=density,
            pressure_outlet_value=outlet_value,
            geometry=geometry_t,
        )
    )
    actual_rhs_t = flow_solver_module._pressure_nonorthogonal_rhs_term_torch(
        actual_flux_t,
        rhs_mode="volume_integrated_flux",
        geometry=geometry_t,
    )

    np.testing.assert_allclose(
        actual_gradient_t.numpy(), expected_gradient, rtol=2e-15, atol=2e-15
    )
    np.testing.assert_allclose(
        actual_flux_t.numpy(), expected_flux, rtol=2e-15, atol=2e-15
    )
    np.testing.assert_allclose(
        actual_rhs_t.numpy(),
        flow_solver_module._pressure_nonorthogonal_rhs_term(
            mesh,
            expected_flux,
            rhs_mode="volume_integrated_flux",
        ),
        rtol=2e-15,
        atol=2e-15,
    )


def test_deferred_lsq_first_sweep_relaxes_persisted_frozen_flux() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=500,
        pressure_tolerance=1e-12,
        pressure_relative_tolerance=1e-10,
        pressure_solver="pcg_diag",
        enable_sign_comparison=False,
        backend="numpy",
        pressure_projection_outlet_contract_mode="preserve",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=1,
        pressure_nonorthogonal_correction_relaxation=0.2,
    )
    state0 = initialize_tetra_flow_state(mesh, cfg)
    state1 = solve_tetra_pressure_projection(mesh, state0, cfg)
    previous_frozen = np.asarray(
        state1.diagnostics["face_flux_primary"][
            "pressure_gradient_flux_nonorthogonal_frozen"
        ],
        dtype=np.float64,
    )
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    geometry = flow_solver_module._cached_pressure_nonorthogonal_geometry(
        mesh,
        outlet_faces,
    )
    cfg_changed_dt = replace(cfg, projection_dt=2.0 * cfg.projection_dt)
    raw_from_incoming_pressure, _ = (
        flow_solver_module._pressure_nonorthogonal_gradient_flux_numpy(
            mesh,
            state1.pressure,
            dt=cfg_changed_dt.projection_dt,
            density=cfg_changed_dt.density,
            pressure_outlet_value=cfg_changed_dt.pressure_outlet_value,
            geometry=geometry,
        )
    )
    expected_frozen = (
        (1.0 - cfg_changed_dt.pressure_nonorthogonal_correction_relaxation)
        * previous_frozen
        + cfg_changed_dt.pressure_nonorthogonal_correction_relaxation
        * raw_from_incoming_pressure
    )

    state2 = solve_tetra_pressure_projection(mesh, state1, cfg_changed_dt)
    actual_frozen = np.asarray(
        state2.diagnostics["face_flux_primary"][
            "pressure_gradient_flux_nonorthogonal_frozen"
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual_frozen, expected_frozen, rtol=0.0, atol=0.0)
    nonorth = dict(state2.diagnostics["pressure_nonorthogonal_correction"])
    assert nonorth["initial_frozen_gradient_flux_source"].startswith(
        "previous_state_frozen_face_flux"
    )
    assert nonorth["initial_frozen_gradient_flux"]["l2"] == pytest.approx(
        float(np.sqrt(np.mean(previous_frozen * previous_frozen))),
        rel=0.0,
        abs=0.0,
    )
    assert nonorth["previous_frozen_gradient_flux_scaling"].startswith("none")


@pytest.mark.parametrize("projection_sign", ["minus", "plus"])
def test_deferred_lsq_projection_uses_frozen_flux_in_rhs_and_continuity(
    projection_sign: str,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=500,
        pressure_tolerance=1e-12,
        pressure_relative_tolerance=1e-10,
        pressure_solver="pcg_diag",
        pcg_require_relative_l2_convergence=True,
        projection_sign=projection_sign,  # type: ignore[arg-type]
        enable_sign_comparison=False,
        backend="numpy",
        pressure_projection_outlet_contract_mode="preserve",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=4,
        pressure_nonorthogonal_correction_relaxation=0.5,
        projection_rhs_mode="volume_integrated_flux",
    )
    rng = np.random.default_rng(42)
    state = TetraFlowState(
        cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
        face_flux=rng.normal(
            scale=0.01,
            size=mesh.face_vertices.shape[0],
        ),
        pressure=np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64),
        diagnostics={},
    )

    projected = solve_tetra_pressure_projection(mesh, state, cfg)
    divergence = compute_tetra_flux_divergence(mesh, projected.face_flux)
    assert float(divergence["divergence_l2"]) < 1e-8
    assert abs(float(divergence["net_boundary_flux"])) < 1e-9
    assert projected.diagnostics["pressure"]["pressure_solved"] is True

    nonorth = dict(projected.diagnostics["pressure_nonorthogonal_correction"])
    assert nonorth["actual_sweeps"] == 4
    assert nonorth["pressure_solve_count"] == 4
    assert nonorth["execution_backend"] == "numpy_vectorized"
    assert nonorth["cuda_used"] is False
    assert nonorth["cuda_fallback_reason"] == "flow execution backend is not Torch"
    assert nonorth["total_true_residual_recomputes"] > 0
    assert nonorth["total_true_residual_restarts"] >= 0
    assert all(
        "true_residual_recompute_count" in row["pressure_solver"]
        for row in nonorth["sweeps"]
    )
    defect = float(nonorth["outer_fixed_point_defect_relative_l2"])
    threshold = float(nonorth["stability_warning_threshold_outer_defect_relative_l2"])
    assert nonorth["stability_warning"] is bool(
        not np.isfinite(defect) or defect > threshold
    )
    assert "startup-step warning" in nonorth["stability_note"]
    primary = dict(projected.diagnostics["face_flux_primary"])
    frozen = np.asarray(
        primary["pressure_gradient_flux_nonorthogonal_frozen"],
        dtype=np.float64,
    )
    orthogonal = np.asarray(
        primary["pressure_gradient_flux_orthogonal"],
        dtype=np.float64,
    )
    full = np.asarray(
        primary["pressure_gradient_flux_full"],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(full, orthogonal + frozen)

    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=cfg.projection_dt,
        density=cfg.density,
        outlet_faces=outlet_faces,
    )
    face_flux_star = np.asarray(primary["face_flux_star"], dtype=np.float64)
    rhs_base, _ = flow_solver_module._assemble_poisson_rhs(
        _compute_cell_flux_sum(mesh, face_flux_star),
        coeff,
        pressure_outlet_value=cfg.pressure_outlet_value,
        projection_sign=cfg.projection_sign,
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
        rhs_mode=cfg.projection_rhs_mode,
    )
    rhs_expected = rhs_base + flow_solver_module._pressure_nonorthogonal_rhs_term(
        mesh,
        frozen,
        rhs_mode=cfg.projection_rhs_mode,
    )
    ap = _matvec_pressure_numpy(coeff, projected.pressure)
    np.testing.assert_allclose(ap, rhs_expected, rtol=1e-8, atol=1e-10)
    consistency = dict(projected.diagnostics["operator_consistency_audit"])
    assert consistency["rhs_from_code_vs_rhs_expected"]["max_abs"] == 0.0


def test_deferred_lsq_projection_torch_cuda_matches_numpy_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh = _build_synthetic_mesh()
    common = dict(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=500,
        pressure_tolerance=1e-12,
        pressure_relative_tolerance=1e-10,
        pressure_solver="pcg_diag",
        enable_sign_comparison=False,
        pressure_projection_outlet_contract_mode="preserve",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=4,
        pressure_nonorthogonal_correction_relaxation=0.5,
        projection_rhs_mode="volume_integrated_flux",
    )
    cfg_numpy = TetraFlowConfig(**common, backend="numpy", device="cpu")
    cfg_cuda = TetraFlowConfig(**common, backend="torch", device="cuda:0")
    rng = np.random.default_rng(20260807)
    state = TetraFlowState(
        cell_velocity=rng.normal(size=(mesh.tetrahedra.shape[0], 3)),
        face_flux=rng.normal(scale=0.01, size=mesh.face_vertices.shape[0]),
        pressure=rng.normal(scale=0.1, size=mesh.tetrahedra.shape[0]),
        diagnostics={},
    )

    expected = solve_tetra_pressure_projection(mesh, state, cfg_numpy)
    actual = solve_tetra_pressure_projection(mesh, state, cfg_cuda)

    np.testing.assert_allclose(
        actual.pressure, expected.pressure, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(
        actual.face_flux, expected.face_flux, rtol=2e-10, atol=2e-12
    )
    np.testing.assert_allclose(
        actual.cell_velocity, expected.cell_velocity, rtol=2e-10, atol=2e-12
    )
    nonorth = dict(actual.diagnostics["pressure_nonorthogonal_correction"])
    backend = dict(actual.diagnostics["backend_execution"])
    assert nonorth["execution_backend"] == "torch"
    assert nonorth["cuda_used"] is True
    assert nonorth["cuda_fallback_reason"] == ""
    assert nonorth["cuda_residency"]["inter_sweep_full_array_host_transfers"] == 0
    assert backend["pressure_nonorthogonal_execution_backend"] == "torch"
    assert backend["pressure_nonorthogonal_cuda_used"] is True
    assert backend["mixed_backend_pressure_nonorthogonal_correction"] is False
    assert backend["all_core_arrays_on_cuda"] is True


def test_deferred_lsq_keeps_unsupported_cuda_pressure_solver_on_legacy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        projection_dt=5e-4,
        max_pressure_iterations=200,
        pressure_tolerance=1e-12,
        pressure_relative_tolerance=1e-10,
        pressure_solver="cg",
        enable_sign_comparison=False,
        backend="torch",
        device="cuda:0",
        pressure_projection_outlet_contract_mode="preserve",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=2,
        projection_rhs_mode="volume_integrated_flux",
    )
    fake_backend = flow_solver_module.BackendSelection(
        requested_backend="torch",
        selected_backend="torch",
        device="cuda:0",
        torch_available=True,
        torch_version="test",
        torch_cuda_available=True,
        torch_device_count=1,
        torch_gpu_name="test",
        used_numpy_fallback=False,
        notes=(),
        cpu_threads={},
    )

    def _fake_pressure_system(coeff, *, rhs, p0, config, backend):
        pressure, diagnostics = flow_solver_module._solve_pressure_cg_numpy(
            coeff,
            rhs=rhs,
            p0=p0,
            config=config,
        )
        return pressure, diagnostics, False

    def _unexpected_resident_solve(*_args, **_kwargs):
        raise AssertionError("unsupported pressure solvers must retain the legacy path")

    monkeypatch.setattr(
        flow_solver_module, "_resolve_backend", lambda _cfg: fake_backend
    )
    monkeypatch.setattr(
        flow_solver_module, "_solve_pressure_system", _fake_pressure_system
    )
    monkeypatch.setattr(
        flow_solver_module,
        "_solve_pressure_deferred_lsq_pcg_diag_torch",
        _unexpected_resident_solve,
    )

    state = initialize_tetra_flow_state(
        mesh, replace(cfg, backend="numpy", device="cpu")
    )
    projected = solve_tetra_pressure_projection(mesh, state, cfg)
    nonorth = dict(projected.diagnostics["pressure_nonorthogonal_correction"])

    assert nonorth["execution_backend"] == "numpy_vectorized"
    assert nonorth["cuda_used"] is False
    assert nonorth["cuda_fallback_reason"] == (
        "CUDA deferred LSQ residency requires pressure_solver='pcg_diag'"
    )


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        (
            {"pressure_nonorthogonal_correction_mode": "unknown"},
            "pressure_nonorthogonal_correction_mode",
        ),
        (
            {"pressure_nonorthogonal_correction_sweeps": 0},
            "pressure_nonorthogonal_correction_sweeps",
        ),
        (
            {"pressure_nonorthogonal_correction_relaxation": 0.0},
            "pressure_nonorthogonal_correction_relaxation",
        ),
        (
            {"pressure_nonorthogonal_correction_relaxation": 1.01},
            "pressure_nonorthogonal_correction_relaxation",
        ),
        (
            {
                "pressure_nonorthogonal_correction_mode": "deferred_lsq",
                "projection_rhs_mode": "divergence_per_volume",
            },
            "volume_integrated_flux",
        ),
    ],
)
def test_pressure_nonorthogonal_config_validation(
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    mesh = _build_synthetic_mesh()
    config = TetraFlowConfig(**config_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        initialize_tetra_flow_state(mesh, config)


def _manufactured_full_viscous_flux(
    mesh,
    velocity: np.ndarray,
    *,
    wall_face_velocity: np.ndarray,
    inlet_face_velocity: np.ndarray,
    outlet_normal_gradient: np.ndarray,
    geometry: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    correction, gradient = flow_solver_module._viscous_nonorthogonal_face_flux_numpy(
        mesh,
        velocity,
        wall_face_velocity=wall_face_velocity,
        inlet_face_velocity=inlet_face_velocity,
        outlet_normal_gradient=outlet_normal_gradient,
        geometry=geometry,
    )
    full_flux = np.asarray(correction, dtype=np.float64).copy()
    for faces_key, owner_key, value, t_key in (
        (
            "wall_faces",
            "wall_owner",
            wall_face_velocity,
            "wall_t",
        ),
        (
            "inlet_faces",
            "inlet_owner",
            inlet_face_velocity,
            "inlet_t",
        ),
    ):
        face_ids = np.asarray(geometry[faces_key], dtype=np.int64)
        owner = np.asarray(geometry[owner_key], dtype=np.int64)
        t_coefficient = np.asarray(geometry[t_key], dtype=np.float64)
        full_flux[face_ids] += t_coefficient[:, None] * (value - velocity[owner])
    interior_faces = np.asarray(geometry["interior_faces"], dtype=np.int64)
    interior_owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
    interior_neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
    interior_t = np.asarray(geometry["interior_t"], dtype=np.float64)
    full_flux[interior_faces] += interior_t[:, None] * (
        velocity[interior_neighbor] - velocity[interior_owner]
    )
    return full_flux, gradient


def test_viscous_nonorthogonal_slip_auto_matches_explicit_none_bitwise() -> None:
    mesh = _build_synthetic_mesh()
    common = {
        "inlet_speed": 0.15,
        "projection_dt": 5e-4,
        "kinematic_viscosity": 1e-3,
        "backend": "numpy",
        "viscous_predictor_mode": (
            "explicit_cell_velocity_laplacian_substepped_conservative"
        ),
        "viscous_predictor_outlet_contract_mode": "preserve",
        "wall_velocity_boundary_mode": "slip",
    }
    cfg_default = TetraFlowConfig(**common)  # type: ignore[arg-type]
    cfg_explicit = TetraFlowConfig(
        **common,  # type: ignore[arg-type]
        viscous_nonorthogonal_correction_mode="none",
    )
    rng = np.random.default_rng(20260723)
    velocity = rng.normal(scale=0.05, size=(mesh.tetrahedra.shape[0], 3))
    face_flux = flow_solver_module._face_flux_from_cell_velocity_numpy(mesh, velocity)
    state = TetraFlowState(
        cell_velocity=velocity,
        face_flux=face_flux,
        pressure=rng.normal(size=mesh.tetrahedra.shape[0]),
        diagnostics={},
    )

    out_default = apply_tetra_stokes_viscous_predictor(
        mesh, state, cfg_default, flow_dt=cfg_default.projection_dt
    )
    out_explicit = apply_tetra_stokes_viscous_predictor(
        mesh, state, cfg_explicit, flow_dt=cfg_explicit.projection_dt
    )

    np.testing.assert_array_equal(out_default.cell_velocity, out_explicit.cell_velocity)
    np.testing.assert_array_equal(out_default.face_flux, out_explicit.face_flux)
    np.testing.assert_array_equal(out_default.pressure, out_explicit.pressure)
    assert (
        out_default.diagnostics["viscous_predictor"][
            "viscous_nonorthogonal_correction_mode"
        ]
        == "none"
    )


def test_viscous_nonorthogonal_flux_is_constant_and_linear_exact() -> None:
    mesh = _build_synthetic_mesh()
    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    geometry = flow_solver_module._cached_viscous_nonorthogonal_geometry(
        mesh,
        inlet_faces=inlet_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)

    constant = np.asarray([0.7, -1.1, 2.3], dtype=np.float64)
    constant_cell = np.broadcast_to(constant, (centers.shape[0], 3)).copy()
    constant_wall = np.broadcast_to(constant, (wall_faces.size, 3)).copy()
    constant_inlet = np.broadcast_to(constant, (inlet_faces.size, 3)).copy()
    constant_outlet_gradient = np.zeros((outlet_faces.size, 3), dtype=np.float64)
    constant_flux, constant_gradient = _manufactured_full_viscous_flux(
        mesh,
        constant_cell,
        wall_face_velocity=constant_wall,
        inlet_face_velocity=constant_inlet,
        outlet_normal_gradient=constant_outlet_gradient,
        geometry=geometry,
    )
    np.testing.assert_allclose(constant_gradient, 0.0, atol=2e-14)
    np.testing.assert_allclose(constant_flux, 0.0, atol=2e-14)

    exact_gradient = np.asarray(
        [
            [0.25, -0.70, 1.10],
            [1.30, 0.40, -0.55],
            [-0.80, 0.95, 0.35],
        ],
        dtype=np.float64,
    )
    intercept = np.asarray([0.2, -0.4, 0.9], dtype=np.float64)
    velocity = intercept + centers @ exact_gradient.T
    wall_velocity = intercept + face_centers[wall_faces] @ exact_gradient.T
    inlet_velocity = intercept + face_centers[inlet_faces] @ exact_gradient.T
    outlet_normal_gradient = np.einsum(
        "cj,fj->fc", exact_gradient, normals[outlet_faces], optimize=True
    )
    full_flux, reconstructed_gradient = _manufactured_full_viscous_flux(
        mesh,
        velocity,
        wall_face_velocity=wall_velocity,
        inlet_face_velocity=inlet_velocity,
        outlet_normal_gradient=outlet_normal_gradient,
        geometry=geometry,
    )
    expected_flux = areas[:, None] * np.einsum(
        "cj,fj->fc", exact_gradient, normals, optimize=True
    )

    np.testing.assert_allclose(
        reconstructed_gradient,
        np.broadcast_to(exact_gradient, reconstructed_gradient.shape),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(full_flux, expected_flux, rtol=2e-12, atol=2e-12)
    laplacian = flow_solver_module._vector_face_flux_laplacian_numpy(mesh, full_flux)
    integrated_laplacian = np.sum(
        np.asarray(mesh.cell_volumes)[:, None] * laplacian, axis=0
    )
    boundary_faces = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    np.testing.assert_allclose(
        integrated_laplacian,
        np.sum(full_flux[boundary_faces], axis=0),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "mesh_name",
    ["vertical_pipe_500.msh"],
)
def test_viscous_nonorthogonal_quadratic_pipe_residual_and_energy_improve(
    mesh_name: str,
) -> None:
    mesh = import_gmsh_tetra_mesh(PROJECT_ROOT / "data" / "meshes" / "gmsh" / mesh_name)
    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    geometry = flow_solver_module._cached_viscous_nonorthogonal_geometry(
        mesh,
        inlet_faces=inlet_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)

    def area_center(face_ids: np.ndarray) -> np.ndarray:
        return np.sum(face_centers[face_ids] * areas[face_ids, None], axis=0) / np.sum(
            areas[face_ids]
        )

    inlet_center = area_center(inlet_faces)
    outlet_center = area_center(outlet_faces)
    axis_delta = outlet_center - inlet_center
    length = float(np.linalg.norm(axis_delta))
    axis = axis_delta / length
    inlet_vertices = np.unique(
        np.asarray(mesh.face_vertices[inlet_faces], dtype=np.int64)
    )
    inlet_vertex_delta = (
        np.asarray(mesh.points[inlet_vertices], dtype=np.float64) - inlet_center
    )
    inlet_vertex_radial = (
        inlet_vertex_delta - (inlet_vertex_delta @ axis)[:, None] * axis[None, :]
    )
    radius = float(np.max(np.linalg.norm(inlet_vertex_radial, axis=1)))
    cell_delta = centers - inlet_center
    axial = cell_delta @ axis
    radial_vector = cell_delta - axial[:, None] * axis[None, :]
    radial = np.linalg.norm(radial_vector, axis=1)
    scalar_profile = 1.0 - (radial / radius) ** 2
    velocity = scalar_profile[:, None] * axis[None, :]

    inlet_delta = face_centers[inlet_faces] - inlet_center
    inlet_axial = inlet_delta @ axis
    inlet_radial_vector = inlet_delta - inlet_axial[:, None] * axis[None, :]
    inlet_radial = np.linalg.norm(inlet_radial_vector, axis=1)
    inlet_velocity = (1.0 - (inlet_radial / radius) ** 2)[:, None] * axis[None, :]
    corrected_flux, _ = _manufactured_full_viscous_flux(
        mesh,
        velocity,
        wall_face_velocity=np.zeros((wall_faces.size, 3), dtype=np.float64),
        inlet_face_velocity=inlet_velocity,
        outlet_normal_gradient=np.zeros((outlet_faces.size, 3), dtype=np.float64),
        geometry=geometry,
    )
    tpfa_flux = np.zeros_like(corrected_flux)
    interior_faces = np.asarray(geometry["interior_faces"], dtype=np.int64)
    owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
    neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
    tpfa_flux[interior_faces] = np.asarray(geometry["interior_t"], dtype=np.float64)[
        :, None
    ] * (velocity[neighbor] - velocity[owner])
    wall_owner = face_to_cells[wall_faces, 0]
    tpfa_flux[wall_faces] = np.asarray(geometry["wall_t"], dtype=np.float64)[
        :, None
    ] * (-velocity[wall_owner])
    inlet_owner = face_to_cells[inlet_faces, 0]
    tpfa_flux[inlet_faces] = np.asarray(geometry["inlet_t"], dtype=np.float64)[
        :, None
    ] * (inlet_velocity - velocity[inlet_owner])

    tpfa_laplacian = flow_solver_module._vector_face_flux_laplacian_numpy(
        mesh, tpfa_flux
    )
    corrected_laplacian = flow_solver_module._vector_face_flux_laplacian_numpy(
        mesh, corrected_flux
    )
    exact_laplacian = -4.0 / radius**2
    middle = (axial >= 0.25 * length) & (axial <= 0.75 * length)

    def middle_ratio(laplacian: np.ndarray) -> float:
        axial_laplacian = laplacian @ axis
        return float(
            np.sum(axial_laplacian[middle] * volumes[middle])
            / (exact_laplacian * np.sum(volumes[middle]))
        )

    tpfa_ratio = middle_ratio(tpfa_laplacian)
    corrected_ratio = middle_ratio(corrected_laplacian)
    exact_energy = float(np.sum(4.0 * radial**2 / radius**4 * volumes))
    tpfa_energy_ratio = float(
        -np.sum(volumes * scalar_profile * (tpfa_laplacian @ axis)) / exact_energy
    )
    corrected_energy_ratio = float(
        -np.sum(volumes * scalar_profile * (corrected_laplacian @ axis)) / exact_energy
    )

    assert abs(corrected_ratio - 1.0) < abs(tpfa_ratio - 1.0)
    assert abs(corrected_energy_ratio - 1.0) < abs(tpfa_energy_ratio - 1.0)


def test_viscous_nonorthogonal_nonaxisymmetric_predictor_is_dissipative() -> None:
    mesh = _build_synthetic_mesh()
    config = TetraFlowConfig(
        inlet_speed=0.0,
        kinematic_viscosity=1e-3,
        projection_dt=1e-4,
        backend="numpy",
        viscous_predictor_mode=(
            "explicit_cell_velocity_laplacian_substepped_conservative"
        ),
        viscous_predictor_outlet_contract_mode="preserve",
        viscous_nonorthogonal_correction_mode="deferred_lsq",
        wall_velocity_boundary_mode="no_slip",
        projection_cell_velocity_update_mode="momentum_pressure_corrected",
    )
    rng = np.random.default_rng(9317)
    velocity = rng.normal(scale=0.03, size=(mesh.tetrahedra.shape[0], 3))
    state = TetraFlowState(
        cell_velocity=velocity,
        face_flux=flow_solver_module._face_flux_from_cell_velocity_numpy(
            mesh, velocity
        ),
        pressure=np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64),
        diagnostics={},
    )
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    energy_before = float(np.sum(volumes[:, None] * velocity * velocity))

    predicted = apply_tetra_stokes_viscous_predictor(
        mesh, state, config, flow_dt=config.projection_dt
    )
    energy_after = float(
        np.sum(volumes[:, None] * np.asarray(predicted.cell_velocity) ** 2)
    )
    diag = predicted.diagnostics["viscous_predictor"]

    assert np.all(np.isfinite(predicted.cell_velocity))
    assert energy_after < energy_before
    assert diag["viscous_nonorthogonal_correction_enabled"] is True
    assert diag["viscous_nonorthogonal_wall_projector"] == "identity"
    assert diag["viscous_nonorthogonal_lsq_rank_min"] == 3
    assert diag["viscous_nonorthogonal_lsq_full_rank_fraction"] == pytest.approx(1.0)
    assert diag["viscous_stability_warning"] is False
    assert diag["viscous_nonorthogonal_stability_bound_per_substep"] <= diag[
        "viscous_nonorthogonal_stability_target"
    ] * (1.0 + 1e-12)


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        (
            {"viscous_nonorthogonal_correction_mode": "unknown"},
            "viscous_nonorthogonal_correction_mode",
        ),
        (
            {
                "viscous_nonorthogonal_correction_mode": "deferred_lsq",
                "viscous_predictor_mode": "face_flux_laplacian_substepped",
                "wall_velocity_boundary_mode": "no_slip",
            },
            "cell-velocity Laplacian",
        ),
        (
            {
                "viscous_nonorthogonal_correction_mode": "deferred_lsq",
                "viscous_predictor_mode": (
                    "explicit_cell_velocity_laplacian_substepped_conservative"
                ),
                "wall_velocity_boundary_mode": "slip",
            },
            "no-slip",
        ),
    ],
)
def test_viscous_nonorthogonal_config_validation(
    config_kwargs: dict[str, object],
    message: str,
) -> None:
    mesh = _build_synthetic_mesh()
    config = TetraFlowConfig(**config_kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        initialize_tetra_flow_state(mesh, config)


def test_deferred_lsq_no_slip_multistep_remains_finite_and_bounded() -> None:
    mesh = _build_synthetic_mesh()
    cfg = TetraFlowConfig(
        inlet_speed=0.15,
        pressure_solver="pcg_diag",
        max_pressure_iterations=300,
        pressure_tolerance=1e-10,
        pressure_relative_tolerance=1e-8,
        projection_dt=5e-4,
        enable_sign_comparison=False,
        backend="numpy",
        kinematic_viscosity=1e-3,
        viscous_predictor_mode=(
            "explicit_cell_velocity_laplacian_substepped_conservative"
        ),
        viscous_predictor_outlet_contract_mode="preserve",
        pressure_projection_outlet_contract_mode="preserve",
        wall_velocity_boundary_mode="no_slip",
        wall_tangential_shear_face_flux_enabled=False,
        wall_tangential_cell_velocity_momentum_enabled=True,
        projection_cell_velocity_update_mode="momentum_pressure_corrected",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=2,
        pressure_nonorthogonal_correction_relaxation=0.2,
    )
    state = initialize_tetra_flow_state(mesh, cfg)

    for _ in range(20):
        state = apply_tetra_stokes_viscous_predictor(
            mesh,
            state,
            cfg,
            flow_dt=cfg.projection_dt,
        )
        state = solve_tetra_pressure_projection(mesh, state, cfg)
        assert np.all(np.isfinite(state.pressure))
        assert np.all(np.isfinite(state.face_flux))
        assert np.all(np.isfinite(state.cell_velocity))
        assert float(np.max(np.abs(state.pressure))) < 1e7
        assert float(np.max(np.abs(state.cell_velocity))) < 1.0
        assert state.diagnostics["pressure"]["pressure_solved"] is True

    final_divergence = compute_tetra_flux_divergence(mesh, state.face_flux)
    assert float(final_divergence["divergence_l2"]) < 1e-8
    nonorthogonal = state.diagnostics["pressure_nonorthogonal_correction"]
    assert nonorthogonal["actual_sweeps"] == 2
    assert float(nonorthogonal["total_pressure_solve_wall_seconds"]) >= 0.0
    assert all(
        float(sweep["pressure_solver"]["solve_wall_seconds"]) >= 0.0
        for sweep in nonorthogonal["sweeps"]
    )
    assert float(
        state.diagnostics["pressure"]["nonorthogonal_total_solve_wall_seconds"]
    ) == pytest.approx(
        float(nonorthogonal["total_pressure_solve_wall_seconds"]),
        rel=1e-12,
        abs=1e-12,
    )


def test_runner_pressure_nonorthogonal_cli_defaults_and_forwards() -> None:
    source_path = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    expected_options = {
        "--pressure-nonorthogonal-correction-mode": (
            ("auto", "none", "deferred_lsq"),
            "auto",
        ),
        "--pressure-nonorthogonal-correction-sweeps": (None, 4),
        "--pressure-nonorthogonal-correction-relaxation": (None, 1.0),
    }
    found_options: set[str] = set()
    config_forwarded: set[str] = set()
    summary_key_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("pressure_nonorthogonal_")
        ):
            summary_key_count += 1
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in expected_options
            ):
                option = str(node.args[0].value)
                choices_expected, default_expected = expected_options[option]
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                assert ast.literal_eval(keywords["default"]) == default_expected
                if choices_expected is not None:
                    assert ast.literal_eval(keywords["choices"]) == choices_expected
                found_options.add(option)
        if isinstance(node.func, ast.Name) and node.func.id == "TetraFlowConfig":
            for keyword in node.keywords:
                if keyword.arg in {
                    "pressure_nonorthogonal_correction_mode",
                    "pressure_nonorthogonal_correction_sweeps",
                    "pressure_nonorthogonal_correction_relaxation",
                }:
                    config_forwarded.add(str(keyword.arg))

    assert found_options == set(expected_options)
    assert config_forwarded == {
        "pressure_nonorthogonal_correction_mode",
        "pressure_nonorthogonal_correction_sweeps",
        "pressure_nonorthogonal_correction_relaxation",
    }
    assert summary_key_count >= 12


def test_runner_viscous_nonorthogonal_cli_defaults_and_forwards() -> None:
    source_path = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    option_found = False
    config_forwarded = False
    summary_key_count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and node.value == "viscous_nonorthogonal_correction_mode"
        ):
            summary_key_count += 1
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--viscous-nonorthogonal-correction-mode"
            ):
                keywords = {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
                assert ast.literal_eval(keywords["choices"]) == (
                    "auto",
                    "none",
                    "deferred_lsq",
                )
                assert ast.literal_eval(keywords["default"]) == "auto"
                option_found = True
        if isinstance(node.func, ast.Name) and node.func.id == "TetraFlowConfig":
            config_forwarded = config_forwarded or any(
                keyword.arg == "viscous_nonorthogonal_correction_mode"
                for keyword in node.keywords
            )

    assert option_found
    assert config_forwarded
    assert summary_key_count >= 2


def test_timing_stats_handles_empty_single_and_multiple_samples() -> None:
    empty = _timing_stats([])
    assert empty == {
        "mean_seconds": 0.0,
        "median_seconds": 0.0,
        "p95_seconds": 0.0,
        "min_seconds": 0.0,
        "max_seconds": 0.0,
    }

    single = _timing_stats([2.0])
    assert single["mean_seconds"] == 2.0
    assert single["median_seconds"] == 2.0
    assert single["p95_seconds"] == 2.0

    multiple = _timing_stats([1.0, 2.0, 3.0, 4.0])
    assert multiple["median_seconds"] == 2.5
    assert multiple["p95_seconds"] == pytest.approx(3.85)


def test_pressure_iteration_telemetry_aggregates_physical_history() -> None:
    telemetry = _pressure_iteration_telemetry(
        [
            {"pressure_iterations": 2, "pressure_stopping_reason": "relative"},
            {"pressure_iterations": 4, "pressure_stopping_reason": "relative"},
            {"pressure_iterations": 8, "pressure_stopping_reason": "max"},
        ]
    )
    assert telemetry["pressure_iterations_total"] == 14
    assert telemetry["pressure_iterations_median"] == 4.0
    assert telemetry["pressure_iterations_p95"] == pytest.approx(7.6)
    assert telemetry["pressure_stopping_reason_counts"] == {"relative": 2, "max": 1}


def test_pressure_matvec_telemetry_reports_csr_cache_and_fallbacks() -> None:
    telemetry = _pressure_matvec_telemetry(
        [
            {
                "pressure_matvec_backend": "torch_sparse_csr",
                "pressure_matvec_sparse_csr_used": True,
                "pressure_matvec_matrix_cached": False,
                "pressure_matvec_fallback_reason": "",
            },
            {
                "pressure_matvec_backend": "torch_sparse_csr",
                "pressure_matvec_sparse_csr_used": True,
                "pressure_matvec_matrix_cached": True,
                "pressure_matvec_fallback_reason": "",
            },
            {
                "pressure_matvec_backend": "torch_index_add",
                "pressure_matvec_sparse_csr_used": False,
                "pressure_matvec_matrix_cached": False,
                "pressure_matvec_fallback_reason": "RuntimeError: unsupported",
            },
        ]
    )
    assert telemetry["pressure_matvec_backend_counts"] == {
        "torch_sparse_csr": 2,
        "torch_index_add": 1,
    }
    assert telemetry["pressure_matvec_sparse_csr_steps"] == 2
    assert telemetry["pressure_matvec_cached_matrix_steps"] == 1
    assert telemetry["pressure_matvec_fallback_steps"] == 1


def test_timing_stats_discards_nonfinite_samples() -> None:
    stats = _timing_stats([1.0, float("nan"), float("inf")])
    assert stats["mean_seconds"] == 1.0
    assert all(np.isfinite(value) for value in stats.values())


def test_environment_metadata_uses_actual_flow_execution_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flow_debug_module,
        "_best_effort_git_metadata",
        lambda: {"git_commit": None, "git_dirty": None},
    )
    mesh = _build_synthetic_mesh()
    backend = type("Backend", (), {"torch_version": "test-torch"})()
    metadata = _best_effort_environment_metadata(
        backend_requested="torch",
        backend_selected="torch",
        flow_execution_backend_requested="numpy",
        flow_execution_backend_selected="numpy",
        flow_execution_device_selected="cpu",
        backend=backend,
        mesh=mesh,
    )
    assert metadata["backend"] == "numpy"
    assert metadata["device"] == "cpu"
    assert metadata["backend_selected"] == "torch"
    assert metadata["flow_execution_backend_selected"] == "numpy"
    assert metadata["cuda_version"] is None
    assert metadata["gpu_name"] is None


def test_cuda_synchronization_telemetry_is_cpu_safe() -> None:
    telemetry = {
        "cuda_active": False,
        "setup_boundary_synchronization": False,
        "cuda_synchronization_count": 0,
    }
    assert not _record_cuda_synchronization(
        telemetry, scope="setup_boundary", backend="numpy", device="cpu"
    )
    assert not _record_cuda_synchronization(
        telemetry, scope="setup_boundary", backend="torch", device="cpu"
    )
    assert telemetry == {
        "cuda_active": False,
        "setup_boundary_synchronization": False,
        "cuda_synchronization_count": 0,
    }


def test_cuda_synchronization_telemetry_records_mocked_cuda_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        flow_debug_module, "_synchronize_cuda_if_active", lambda **_kwargs: True
    )
    telemetry = {
        "cuda_active": False,
        "flow_boundary_synchronization": False,
        "component_boundary_synchronization": False,
        "cuda_synchronization_count": 0,
    }
    _record_cuda_synchronization(
        telemetry, scope="flow_boundary", backend="torch", device="cuda:1"
    )
    _record_cuda_synchronization(
        telemetry, scope="component_boundary", backend="torch", device="cuda:1"
    )
    assert telemetry["cuda_active"] is True
    assert telemetry["flow_boundary_synchronization"] is True
    assert telemetry["component_boundary_synchronization"] is True
    assert telemetry["cuda_synchronization_count"] == 2


def test_basic_and_detailed_component_synchronization_policy() -> None:
    assert _timing_mode_synchronizes_components("basic") is False
    assert _timing_mode_synchronizes_components("detailed") is True


def test_git_metadata_failure_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 1.0)

    monkeypatch.setattr(flow_debug_module.subprocess, "check_output", _raise_timeout)
    assert _best_effort_git_metadata() == {"git_commit": None, "git_dirty": None}


def test_flow_runner_writes_finite_basic_timing_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.gmsh.run_import_gmsh_mesh import _save_npz

    mesh_path = tmp_path / "synthetic_imported_mesh.npz"
    _save_npz(_build_synthetic_mesh(), mesh_path)
    output_root = tmp_path / "flow_runs"

    def _unexpected_vtu_export(*_args, **_kwargs):
        raise AssertionError("minimal postprocessing must skip VTU export")

    monkeypatch.setattr(flow_debug_module, "_export_vtu", _unexpected_vtu_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gmsh_tetra_flow_debug.py",
            "--mesh-npz",
            str(mesh_path),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--flow-execution-backend",
            "numpy",
            "--flow-steps",
            "1",
            "--max-pressure-iterations",
            "1",
            "--postprocessing-mode",
            "minimal",
        ],
    )

    flow_debug_module.main()

    summaries = list(output_root.glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    timing = summary["timing"]
    assert summary["mesh_sha256"] == pressure_diagnostics_module.sha256_file(mesh_path)
    assert summary["runtime_seconds"] >= 0.0
    assert summary["runtime_seconds"] == timing["wall_total_seconds"]
    assert timing["finished_at_utc"]
    assert timing["timing_mode"] == "basic"
    assert timing["postprocessing_mode"] == "minimal"
    assert timing["steps"]["completed"] == summary["flow_steps_completed"]
    assert timing["environment"]["backend"] == "numpy"
    assert timing["environment"]["device"] == "cpu"
    assert timing["cuda_synchronization"]["cuda_active"] is False
    assert timing["cuda_synchronization"]["cuda_synchronization_count"] == 0
    assert all(
        np.isfinite(value) and value >= 0.0 for value in timing["phases"].values()
    )
    json.dumps(summary, allow_nan=False)
    config = json.loads((summaries[0].parent / "config.json").read_text("utf-8"))
    assert config["timing_mode"] == "basic"
    assert config["postprocessing_mode"] == "minimal"
    history = json.loads(
        (summaries[0].parent / "flow_progression_history.json").read_text("utf-8")
    )
    assert history["steps"][0]["viscous_execution_backend"] == "numpy"
    assert history["steps"][0]["viscous_execution_device"] == "cpu"
    assert history["steps"][0]["viscous_torch_cuda_used"] is False
    assert history["steps"][0]["convective_cuda_handoff_available"] is False
    assert history["steps"][0]["viscous_cuda_input_reused"] is False
    assert history["steps"][0]["viscous_cuda_no_slip_used"] is False
    assert history["steps"][0]["viscous_cuda_finalization_used"] is False
    assert history["steps"][0]["viscous_cuda_host_to_device_bytes_avoided"] == 0
    assert history["steps"][0]["viscous_cuda_residency_scope"] == "none"
    assert history["steps"][0]["cached_slip_reconstruction_used"] is True
    assert history["steps"][0]["slip_dense_reconstruction_solves_avoided"] == 2
    assert history["steps"][0]["viscous_numpy_fallback_reason"] == ""
    assert list(summaries[0].parent.glob("*.png")) == []
    assert list(summaries[0].parent.glob("*.vtu")) == []


def test_pressure_determinism_run_completes_pipeline_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.gmsh.run_import_gmsh_mesh import _save_npz

    mesh = _build_synthetic_mesh()
    mesh_path = tmp_path / "synthetic_imported_mesh.npz"
    _save_npz(mesh, mesh_path)
    mesh_sha256 = pressure_diagnostics_module.sha256_file(mesh_path)

    source_run = tmp_path / "source_flow"
    source_run.mkdir()
    (source_run / "summary.json").write_text(
        json.dumps(
            {
                "resolved_mesh_npz": str(mesh_path),
                "mesh_sha256": mesh_sha256,
                "flow_steps_completed": 1,
                "physical_time_final": 1.0e-5,
            }
        ),
        encoding="utf-8",
    )
    np.save(
        source_run / "final_corrected_face_flux.npy",
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
    )
    np.save(
        source_run / "final_cell_velocity.npy",
        np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
    )
    np.save(
        source_run / "final_pressure.npy",
        np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64),
    )

    def _fake_pressure_diagnostic(*, output_dir: Path, **_kwargs):
        report = {"first_divergent_stage": None}
        (output_dir / "pressure_determinism_report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        np.savez_compressed(output_dir / "pressure_system_inputs.npz", rhs=np.zeros(1))
        (output_dir / "cpu_residual_history.json").write_text(
            json.dumps({"history": []}),
            encoding="utf-8",
        )
        for index in range(1, 4):
            (output_dir / f"gpu_residual_history_{index}.json").write_text(
                json.dumps({"history": []}),
                encoding="utf-8",
            )
        return report

    monkeypatch.setattr(
        pressure_diagnostics_module,
        "run_pressure_determinism_diagnostic",
        _fake_pressure_diagnostic,
    )
    output_root = tmp_path / "diagnostic_runs"
    manifest_path = tmp_path / "pipeline" / "pipeline_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gmsh_tetra_flow_debug.py",
            "--mesh-npz",
            str(mesh_path),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--flow-execution-backend",
            "numpy",
            "--fixed-work-source-run-dir",
            str(source_run),
            "--pressure-determinism-diagnostic",
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "diagnostic-001",
        ],
    )

    flow_debug_module.main()

    run_dirs = list(output_root.glob("*_synthetic"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["runs"]["diagnostic-001"]
    assert run["status"] == "completed"
    assert run["metadata"]["manifest_role"] == "pressure_determinism_diagnostic"
    assert run["metadata"]["ready_for_next_stage"] is False
    assert "acceptance_report_json" not in run["artifacts"]
    assert set(run["artifacts"]) == {
        "run_log",
        "config_json",
        "summary_json",
        "fixed_work_manifest_json",
        "pressure_determinism_report_json",
        "pressure_system_inputs_npz",
        "cpu_residual_history_json",
        "gpu_residual_history_1_json",
        "gpu_residual_history_2_json",
        "gpu_residual_history_3_json",
    }
    assert all(
        (run_dir / Path(value).name).is_file() for value in run["artifacts"].values()
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_type"] == "pressure_determinism_diagnostic"
    assert summary["mesh_sha256"] == mesh_sha256
