from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import experiments.gmsh.run_gmsh_tetra_transport_debug as transport_debug_module
from experiments.gmsh.run_gmsh_tetra_transport_debug import (
    _check_flow_transport_mesh_compatibility,
    _compute_cleanliness_flags,
    _evaluate_transport_front_observation,
    _load_flow_coupling_payload,
    _parse_child_run_directory,
    _run_transport_with_checkpointing,
    _save_transport_checkpoint,
    _load_transport_checkpoint,
    _snapshot_front_diagnostics,
    _validate_source_flow_readiness,
)
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
from microfluidics.gmsh.tetra.gmsh_tetra_regime_guardrail import (
    DEFAULT_MAX_GRID_PECLET,
    DEFAULT_MAX_SCHMIDT,
)
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    build_face_normal_flux_from_velocity,
)
from microfluidics.gmsh.tetra.gmsh_tetra_transport_solver import (
    BOUNDARY_GROUP_CODE,
    GmshTetraTransportConfig,
    _apply_bounded_limiter,
    _assemble_advection_fluxes,
    _build_boundary_face_groups,
    resolve_transport_dt_control,
    run_tetra_transport_debug,
)
from microfluidics.gmsh.tetra.threshold_keys import (
    format_threshold_key,
    read_threshold_value,
)
from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (
    build_prescribed_velocity_field,
    compute_velocity_field_diagnostics,
)

_RESOLVABLE_DEFAULTS = {
    "DEFAULT_MAX_GRID_PECLET": DEFAULT_MAX_GRID_PECLET,
    "DEFAULT_MAX_SCHMIDT": DEFAULT_MAX_SCHMIDT,
}


def _collect_transport_debug_cli_defaults() -> dict[str, object]:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            if isinstance(keyword.value, ast.Constant):
                defaults[option] = keyword.value.value
            elif isinstance(keyword.value, ast.Name):
                resolved = _RESOLVABLE_DEFAULTS.get(keyword.value.id)
                if resolved is not None:
                    defaults[option] = resolved

    return defaults


def test_transport_debug_cli_defaults_match_stage3_profile() -> None:
    defaults = _collect_transport_debug_cli_defaults()

    assert defaults["--velocity-source"] == ""
    assert defaults["--transport-mode"] == "advection_diffusion"
    assert defaults["--transport-scheme"] == "bounded_upwind"
    assert defaults["--steps"] is None
    assert defaults["--transport-end-time"] is None
    assert defaults["--dt-mode"] == "auto"
    assert defaults["--dt"] == 1.5e-5
    assert defaults["--cfl-target"] == 0.5
    assert defaults["--auto-dt-percentile"] == 5.0
    assert defaults["--max-transport-substeps"] is None
    assert defaults["--transport-substep-warning-threshold"] == 32
    assert "--fail-on-transport-substep-cap" not in defaults
    assert defaults["--cfl-limit"] == 0.8
    assert defaults["--diffusivity"] == 3e-10
    assert defaults["--kinematic-viscosity"] == 1e-6
    assert defaults["--max-supported-grid-peclet"] == 2.0
    assert defaults["--max-supported-schmidt"] == 1_000.0
    assert defaults["--gradient-method"] == "least_squares"
    assert defaults["--laplacian-method"] == "tpfa"
    assert defaults["--left-inlet-value"] == 0.0
    assert defaults["--right-inlet-value"] == 1.0
    assert defaults["--inlet-speed"] == 0.15
    assert defaults["--velocity-comparison"] is False
    assert defaults["--transport-scheme-comparison"] is False


def _structured_tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = 3, 3, 3
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)

    rng = np.random.default_rng(0)
    for idx, (x, y, z) in enumerate(points):
        interior = (
            1e-9 < x < 1.0 - 1e-9 and 1e-9 < y < 1.0 - 1e-9 and 1e-9 < z < 1.0 - 1e-9
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


def _build_synthetic_mesh():
    points, tetrahedra = _structured_tetra_mesh()
    btri, btags, field_data = _extract_boundary_triangles(points, tetrahedra)
    return build_imported_tetra_mesh(
        source_path=Path("synthetic.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=btri,
        boundary_face_tags=btags,
        field_data=field_data,
    )


def _build_two_cell_mesh():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
            [1.0, 1.0, 1.0],  # 4
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
        source_path=Path("two_cell.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _build_one_cell_inlet_outlet_mesh():
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
            [1, 2, 3],  # inlet
            [0, 3, 2],  # outlet
            [0, 1, 3],  # wall
            [0, 2, 1],  # wall
        ],
        dtype=np.int64,
    )
    boundary_face_tags = np.asarray([2, 4, 5, 5], dtype=np.int32)
    field_data = {
        "left_inlet": np.asarray([2, 2], dtype=np.int32),
        "outlet": np.asarray([4, 2], dtype=np.int32),
        "walls": np.asarray([5, 2], dtype=np.int32),
    }
    return build_imported_tetra_mesh(
        source_path=Path("one_cell.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _build_transport_flux(
    field_name: str = "two_inlets_to_outlet_tj_axis_aligned_clean",
    *,
    inlet_speed: float = 0.15,
):
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name=field_name,
        inlet_speed=inlet_speed,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    return mesh, face_normal_velocity, flux_diag


def test_wall_flux_zero_with_prescribed_velocity_field() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    assert face_normal_velocity.shape[0] == mesh.face_vertices.shape[0]
    assert flux_diag["wall_flux_max_abs"] <= 1e-12


def test_parse_child_run_directory_extracts_explicit_stdout_path() -> None:
    stdout = (
        "hello\n[gmsh-tetra-transport] run directory: C:/tmp/transport-run\nworld\n"
    )
    parsed = _parse_child_run_directory(
        stdout,
        prefix="[gmsh-tetra-transport] run directory:",
        variant="cpu",
    )
    assert parsed == Path("C:/tmp/transport-run").resolve()


def test_parse_child_run_directory_fails_without_explicit_stdout_path() -> None:
    with pytest.raises(RuntimeError, match="Could not resolve child run directory"):
        _parse_child_run_directory(
            "hello\nworld\n",
            prefix="[gmsh-tetra-transport] run directory:",
            variant="gpu",
        )


def test_transport_advection_runs_without_nan_and_cfl_positive() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=12,
        dt=2e-3,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    scalar = np.asarray(result["scalar"], dtype=np.float64)
    assert np.all(np.isfinite(scalar))
    assert result["cfl_max"] > 0.0
    assert result["cfl_max"] <= 0.8 + 1e-9
    assert np.min(scalar) >= 0.0 - 1e-12
    assert np.max(scalar) <= 1.0 + 1e-12
    dt_control = result["dt_control"]
    assert dt_control["dt_mode"] == "auto"
    assert dt_control["used_dt"] > 0.0
    step0 = result["history"][0]
    assert "clipped_cell_count" in step0
    assert "overshoot_cell_count" in step0
    assert "undershoot_cell_count" in step0


def test_transport_dt_limiter_audit_identifies_limiting_cell_and_faces() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(steps=1, dt_mode="auto", progress_every=0),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    audit = result["dt_control"]["dt_limiter_audit"]
    limiting = audit["limiting_cell"]
    assert isinstance(limiting, dict)
    assert audit["active_outflow_cell_count"] > 0
    assert limiting["stable_dt"] == pytest.approx(audit["stable_dt_estimate"])
    assert audit["stable_dt_estimate"] == pytest.approx(
        result["dt_control"]["stable_dt_estimate"]
    )
    assert limiting["outgoing_volume_flux_sum"] > 0.0
    assert limiting["outgoing_face_count"] > 0
    assert limiting["outgoing_face_contributions"]


def test_transport_outlet_threshold_keys_are_canonical() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux(
        "two_inlets_to_outlet_tj_axis_aligned_sanity"
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=2,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
        backend="numpy",
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    thresholds = (1e-6, 1e-4, 1e-3, 1e-2)
    canonical_keys = {
        format_threshold_key("outlet_frac_gt", thr, canonical=True)
        for thr in thresholds
    }
    legacy_keys = {
        format_threshold_key("outlet_frac_gt", thr, canonical=False)
        for thr in thresholds
    }
    legacy_only_keys = legacy_keys - canonical_keys

    history_entry = dict(result["history"][0])
    assert canonical_keys.issubset(history_entry.keys())
    for key in legacy_only_keys:
        assert key not in history_entry

    outlet_arrival = dict(result["outlet_arrival"])
    assert canonical_keys.issubset(outlet_arrival.keys())
    for key in legacy_only_keys:
        assert key not in outlet_arrival


def test_read_threshold_value_supports_legacy_fallback_and_canonical_priority() -> None:
    canonical_key = format_threshold_key("outlet_frac_gt", 1e-4, canonical=True)
    legacy_key = format_threshold_key("outlet_frac_gt", 1e-4, canonical=False)

    legacy_only = {legacy_key: 0.25}
    assert (
        read_threshold_value(legacy_only, "outlet_frac_gt", 1e-4, default=0.0) == 0.25
    )

    canonical_and_legacy = {canonical_key: 0.75, legacy_key: 0.25}
    assert (
        read_threshold_value(canonical_and_legacy, "outlet_frac_gt", 1e-4, default=0.0)
        == 0.75
    )

    assert read_threshold_value({}, "outlet_frac_gt", 1e-4, default=0.5) == 0.5


def test_auto_dt_uses_same_used_dt_for_advection_diffusion_and_source() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    common = dict(
        transport_mode="advection_diffusion",
        transport_scheme="upwind",
        steps=1,
        dt_mode="auto",
        cfl_target=0.2,
        diffusivity=1e-2,
        source_term=0.25,
        clipping_enabled=False,
        progress_every=0,
        backend="numpy",
    )
    small_requested_dt = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, dt=0.01),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    large_requested_dt = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, dt=1.0),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert small_requested_dt["dt_control"]["dt_mode"] == "auto"
    assert (
        small_requested_dt["dt_control"]["configured_dt"]
        != large_requested_dt["dt_control"]["configured_dt"]
    )
    assert (
        small_requested_dt["dt_control"]["requested_outer_dt"]
        == large_requested_dt["dt_control"]["requested_outer_dt"]
    )
    assert (
        small_requested_dt["dt_control"]["requested_dt"]
        == large_requested_dt["dt_control"]["requested_dt"]
    )
    assert (
        small_requested_dt["dt_control"]["used_dt"]
        == large_requested_dt["dt_control"]["used_dt"]
    )
    np.testing.assert_allclose(
        np.asarray(small_requested_dt["scalar"], dtype=np.float64),
        np.asarray(large_requested_dt["scalar"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_auto_dt_uses_strict_stability_limit_without_substeps() -> None:
    mesh, face_normal_velocity, _ = _build_transport_flux()
    cfg = GmshTetraTransportConfig(
        dt_mode="auto",
        cfl_target=0.9,
        auto_dt_percentile=90.0,
        max_transport_substeps=None,
        progress_every=0,
    )

    dt_control = resolve_transport_dt_control(
        mesh,
        face_normal_velocity,
        config=cfg,
    )

    assert (
        dt_control["stable_dt_percentile_estimate"] >= dt_control["stable_dt_estimate"]
    )
    assert dt_control["stable_dt_robust_estimate"] >= dt_control["stable_dt_estimate"]
    assert dt_control["auto_dt_selection"] == "strict_stability_limit"
    assert dt_control["required_transport_substep_count"] == 1
    assert (
        dt_control["executed_transport_substep_count"]
        == dt_control["required_transport_substep_count"]
    )
    assert dt_control["dt_substep"] <= dt_control["explicit_stability_dt_limit"] + 1e-12
    assert dt_control["actual_advanced_dt"] == pytest.approx(
        dt_control["requested_outer_dt"]
    )
    assert dt_control["requested_outer_dt"] == pytest.approx(
        0.9 * dt_control["explicit_stability_dt_limit"]
    )


def test_transport_dt_preserves_requested_outer_interval_without_default_hard_cap() -> (
    None
):
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=1,
        dt_mode="manual",
        dt=30.0,
        max_transport_substeps=None,
        transport_substep_warning_threshold=32,
        clipping_enabled=False,
        cfl_limit=1.05,
        progress_every=0,
        backend="numpy",
    )

    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    dt_control = result["dt_control"]

    assert dt_control["transport_substep_cap_hit"] is False
    assert dt_control["transport_substep_warning"] is True
    assert dt_control["transport_dt_controller_status"] == "warning_substepping"
    assert dt_control["requested_outer_dt"] == pytest.approx(30.0)
    assert dt_control["used_dt"] == pytest.approx(dt_control["requested_outer_dt"])
    assert dt_control["actual_advanced_dt"] == pytest.approx(
        dt_control["requested_outer_dt"]
    )
    assert dt_control["required_transport_substep_count"] > 32
    assert (
        dt_control["executed_transport_substep_count"]
        == dt_control["required_transport_substep_count"]
    )
    assert result["history"][0]["dt_substep"] == pytest.approx(dt_control["dt_substep"])
    assert result["history"][0]["dt_used"] == pytest.approx(
        dt_control["requested_outer_dt"]
    )
    assert result["cfl_warning"] is False


def test_transport_dt_substep_warning_is_not_suppressed_by_legacy_threshold() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    common = dict(
        transport_mode="advection",
        steps=1,
        dt_mode="manual",
        dt=30.0,
        max_transport_substeps=None,
        progress_every=0,
        backend="numpy",
        clipping_enabled=False,
        cfl_limit=1.05,
    )
    warning_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(
            **common,
            transport_substep_warning_threshold=1,
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    quiet_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(
            **common,
            transport_substep_warning_threshold=128,
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert warning_result["dt_control"]["transport_substep_warning"] is True
    assert quiet_result["dt_control"]["transport_substep_warning"] is True
    assert warning_result["dt_control"]["used_dt"] == pytest.approx(
        quiet_result["dt_control"]["used_dt"]
    )
    assert warning_result["dt_control"]["dt_substep"] == pytest.approx(
        quiet_result["dt_control"]["dt_substep"]
    )
    assert warning_result["dt_control"][
        "required_transport_substep_count"
    ] == pytest.approx(quiet_result["dt_control"]["required_transport_substep_count"])
    assert warning_result["cfl_warning"] is False
    assert quiet_result["cfl_warning"] is False


def test_transport_dt_legacy_substep_cap_is_informational_not_blocking() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=1,
        dt_mode="manual",
        dt=30.0,
        max_transport_substeps=8,
        progress_every=0,
        backend="numpy",
        clipping_enabled=False,
        cfl_limit=1.05,
    )

    dt_control = resolve_transport_dt_control(
        mesh,
        face_normal_velocity,
        config=cfg,
    )

    assert dt_control["transport_substep_cap_hit"] is True
    assert dt_control["transport_dt_controller_blocked"] is False
    assert dt_control["transport_dt_controller_status"] == "warning_substepping"
    assert dt_control["requested_outer_dt"] == pytest.approx(30.0)
    assert dt_control["used_dt"] == pytest.approx(30.0)
    assert dt_control["actual_advanced_dt"] == pytest.approx(30.0)
    assert dt_control["required_transport_substep_count"] > 8
    assert (
        dt_control["executed_transport_substep_count"]
        == dt_control["required_transport_substep_count"]
    )

    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    assert result["dt_control"]["transport_substep_cap_hit"] is True


def test_transport_dt_control_accounts_for_diffusion_limit() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="upwind",
        steps=1,
        dt_mode="manual",
        dt=1.0,
        diffusivity=1.0,
        max_transport_substeps=None,
        clipping_enabled=False,
        progress_every=0,
        backend="numpy",
    )

    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    dt_control = result["dt_control"]

    assert np.isfinite(float(dt_control["diffusion_dt_limit"]))
    assert dt_control["dt_substep"] <= dt_control["diffusion_dt_limit"] + 1e-12


def test_transport_time_estimate_reports_legacy_cap_without_blocking() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_clean",
        inlet_speed=0.15,
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=8,
        dt_mode="manual",
        dt=30.0,
        max_transport_substeps=8,
        progress_every=0,
    )
    expected_dt_control = resolve_transport_dt_control(
        mesh,
        np.asarray(face_normal_velocity, dtype=np.float64),
        config=cfg,
    )

    time_estimate = transport_debug_module._estimate_transport_run_time(
        mesh=mesh,
        cell_velocity=np.asarray(velocity.cell_velocity, dtype=np.float64),
        face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
        flux_diag=flux_diag,
        transport_config=cfg,
        resolve_transport_dt_control_fn=resolve_transport_dt_control,
        resume_state=None,
        breakthrough_turnover_threshold=0.2,
        breakthrough_travel_fraction=0.35,
        max_walltime_seconds=0.0,
    )

    assert time_estimate["transport_substep_cap_hit"] is True
    assert time_estimate["transport_dt_controller_blocked"] is False
    assert "breakthrough_expected" in time_estimate
    for observation_key in (
        "breakthrough_detected",
        "observation",
        "observation_reason",
        "outlet_frac_gt_1e-6",
        "outlet_fraction_detection_threshold",
    ):
        assert observation_key not in time_estimate
    assert time_estimate["used_dt"] == pytest.approx(30.0)
    assert time_estimate["dt_control"]["requested_outer_dt"] == pytest.approx(30.0)
    assert (
        time_estimate["dt_control"]["required_transport_substep_count"]
        == expected_dt_control["required_transport_substep_count"]
    )
    assert (
        time_estimate["dt_control"]["executed_transport_substep_count"]
        == expected_dt_control["executed_transport_substep_count"]
    )
    assert (
        time_estimate["dt_control"]["transport_dt_controller_status"]
        == expected_dt_control["transport_dt_controller_status"]
    )


def test_transport_end_time_plans_short_final_step_and_records_progress(
    tmp_path: Path,
) -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_clean",
        inlet_speed=0.15,
    )
    base_cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=1,
        dt_mode="auto",
        cfl_target=0.5,
        progress_every=0,
        backend="numpy",
    )
    base_control = resolve_transport_dt_control(
        mesh,
        face_normal_velocity,
        config=base_cfg,
    )
    target_end_time = 2.5 * float(base_control["used_dt"])
    estimate = transport_debug_module._estimate_transport_run_time(
        mesh=mesh,
        cell_velocity=np.asarray(velocity.cell_velocity, dtype=np.float64),
        face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
        flux_diag=flux_diag,
        transport_config=base_cfg,
        resolve_transport_dt_control_fn=resolve_transport_dt_control,
        resume_state=None,
        breakthrough_turnover_threshold=0.2,
        breakthrough_travel_fraction=0.35,
        max_walltime_seconds=0.0,
        transport_end_time=target_end_time,
    )

    assert estimate["run_horizon_mode"] == "end_time"
    assert estimate["planned_total_steps"] == 3
    assert estimate["final_outer_dt"] == pytest.approx(0.5 * base_control["used_dt"])

    result = _run_transport_with_checkpointing(
        mesh=mesh,
        base_config=replace(base_cfg, steps=int(estimate["planned_total_steps"])),
        run_transport_fn=run_tetra_transport_debug,
        face_normal_velocity=face_normal_velocity,
        flux_diag=flux_diag,
        snapshot_steps=(),
        checkpoint_every=0,
        max_walltime_seconds=0.0,
        run_dir=tmp_path,
        resume_state=None,
        run_horizon_mode="end_time",
        transport_end_time=target_end_time,
        final_outer_dt=float(estimate["final_outer_dt"]),
    )

    assert result["run_control"]["completed_steps"] == 3
    assert result["run_control"]["physical_time_final"] == pytest.approx(
        target_end_time
    )
    assert result["history"][-1]["physical_time"] == pytest.approx(target_end_time)
    assert result["history"][-1]["step_progress_percent"] == pytest.approx(100.0)
    assert result["history"][-1]["time_progress_percent"] == pytest.approx(100.0)


def test_numpy_and_torch_auto_dt_control_parity() -> None:
    pytest.importorskip("torch")

    mesh, face_normal_velocity, flux_diag = _build_transport_flux(
        "two_inlets_to_outlet_tj_axis_aligned_sanity"
    )
    common = dict(
        transport_mode="advection",
        transport_scheme="upwind",
        steps=2,
        dt_mode="auto",
        cfl_target=0.9,
        auto_dt_percentile=90.0,
        max_transport_substeps=None,
        clipping_enabled=False,
        cfl_limit=0.8,
        progress_every=0,
    )
    numpy_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, backend="numpy"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    torch_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, backend="torch", torch_device="cpu"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert numpy_result["dt_control"]["used_dt"] == pytest.approx(
        torch_result["dt_control"]["used_dt"]
    )
    assert numpy_result["dt_control"]["dt_substep"] == pytest.approx(
        torch_result["dt_control"]["dt_substep"]
    )
    assert numpy_result["dt_control"]["transport_substep_count"] == pytest.approx(
        torch_result["dt_control"]["transport_substep_count"]
    )
    np.testing.assert_allclose(
        np.asarray(numpy_result["scalar"], dtype=np.float64),
        np.asarray(torch_result["scalar"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_bounded_upwind_mass_diagnostics_use_effective_limited_boundary_flux() -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    volume = float(mesh.cell_volumes[0])
    dt = 1.0
    q = 3.0 * volume / dt
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    inlet_face = int(mesh.inlet_faces[0])
    outlet_face = int(mesh.outlet_faces[0])
    face_normal_velocity[inlet_face] = -q / float(mesh.face_areas[inlet_face])
    face_normal_velocity[outlet_face] = q / float(mesh.face_areas[outlet_face])
    flux_diag = {
        "total_inlet_flux_in": q,
        "total_outlet_flux_out": q,
        "net_boundary_flux": 0.0,
        "wall_flux_max_abs": 0.0,
    }
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        transport_scheme="bounded_upwind",
        steps=1,
        dt=dt,
        dt_mode="manual",
        left_inlet_value=1.0,
        right_inlet_value=1.0,
        clipping_enabled=False,
        cfl_limit=10.0,
        progress_every=0,
        backend="numpy",
    )

    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    entry = result["history"][0]
    masses = result["masses"]
    mass_delta = float(masses["scalar_mass_final"]) - float(
        masses["scalar_mass_initial"]
    )

    assert entry["limiter_active_cell_count"] == 1
    assert abs(mass_delta - float(entry["net_mass_change_from_flux"])) <= 1e-12
    assert float(entry["raw_mass_in"]) > float(entry["mass_in"])
    assert float(entry["raw_net_mass_change_from_flux"]) > float(
        entry["net_mass_change_from_flux"]
    )
    assert abs(float(masses["mass_balance_error"])) <= 1e-12
    assert abs(float(masses["raw_mass_balance_error"])) > 1e-3


def test_diffusion_laplacian_metric_reports_active_nonzero_laplacian() -> None:
    mesh, face_normal_velocity, flux_diag = _build_transport_flux()
    cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="upwind",
        steps=2,
        dt_mode="auto",
        cfl_target=0.2,
        diffusivity=1e-2,
        clipping_enabled=False,
        progress_every=0,
        backend="numpy",
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    values = [
        float(entry["diffusion_laplacian_max_abs"]) for entry in result["history"]
    ]
    assert max(values) > 0.0


def test_numpy_and_torch_transport_source_term_parity() -> None:
    pytest.importorskip("torch")

    mesh, face_normal_velocity, flux_diag = _build_transport_flux(
        "two_inlets_to_outlet_tj_axis_aligned_sanity"
    )
    common = dict(
        transport_mode="advection",
        transport_scheme="upwind",
        steps=3,
        dt=1e-3,
        dt_mode="manual",
        source_term=0.2,
        clipping_enabled=False,
        cfl_limit=0.8,
        progress_every=0,
    )
    numpy_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, backend="numpy"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    torch_result = run_tetra_transport_debug(
        mesh,
        GmshTetraTransportConfig(**common, backend="torch", torch_device="cpu"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    np.testing.assert_allclose(
        np.asarray(numpy_result["scalar"], dtype=np.float64),
        np.asarray(torch_result["scalar"], dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_snapshot_front_diagnostics_uses_canonical_threshold_keys() -> None:
    centers = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=np.float64)
    concentration = np.asarray([0.0, 2e-4], dtype=np.float64)

    front = _snapshot_front_diagnostics(centers, concentration)

    assert "max_y_where_C_gt_1e-4" in front
    assert "max_y_where_C_gt_1e-04" not in front
    assert float(front["max_y_where_C_gt_1e-4"]) == 2.0


def test_manual_dt_uses_stable_substeps_before_reporting_cfl() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=2,
        dt=1.0,
        dt_mode="manual",
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    assert result["cfl_warning"] is False
    assert result["dt_control"]["transport_substep_count"] >= 2
    assert result["history"][0]["dt_substep"] < result["history"][0]["dt_used"]


def test_inlet_outlet_groups_present() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh, field_name="constant_x", inlet_speed=0.12
    )
    assert velocity.boundary_groups["left_inlet_faces"].size > 0
    assert velocity.boundary_groups["right_inlet_faces"].size > 0
    assert velocity.boundary_groups["outlet_faces"].size > 0
    assert velocity.boundary_groups["wall_faces"].size > 0


def test_backend_auto_not_failing_without_torch() -> None:
    backend = select_backend(
        "auto",
        torch_available_override=False,
        torch_cuda_available_override=False,
    )
    assert backend.selected_backend == "numpy"
    assert backend.device == "cpu"


def test_torch_transport_backend_reports_no_numpy_fallback_for_advection() -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return

    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_sanity",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    device = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        steps=3,
        dt=1e-3,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
        backend="torch",
        torch_device=device,
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    be = result["backend_execution"]
    assert be["stepping_backend"] == "torch"
    assert be["used_numpy_fallback"] is False
    if device.startswith("cuda"):
        assert be["all_core_arrays_on_cuda"] is True


def test_advection_diffusion_torch_path_reports_no_numpy_fallback() -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return

    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_sanity",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    device = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"
    cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=3,
        dt=1e-3,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=False,
        cfl_limit=0.8,
        progress_every=0,
        diffusivity=3e-10,
        gradient_method="least_squares",
        laplacian_method="lsq_flux",
        backend="torch",
        torch_device=device,
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    scalar = np.asarray(result["scalar"], dtype=np.float64)
    assert np.all(np.isfinite(scalar))
    be = result["backend_execution"]
    assert be["stepping_backend"] == "torch"
    assert be["used_numpy_fallback"] is False
    assert "diffusion_diagnostics" in result
    if device.startswith("cuda"):
        assert be["all_core_arrays_on_cuda"] is True


def test_diffusion_only_constant_state_remains_constant_with_zero_flux() -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    flux_diag = {
        "total_inlet_flux_in": 0.0,
        "total_outlet_flux_out": 0.0,
        "net_boundary_flux": 0.0,
        "wall_flux_max_abs": 0.0,
    }
    cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=2,
        dt=1e-3,
        dt_mode="manual",
        cfl_target=0.5,
        clipping_enabled=False,
        cfl_limit=0.8,
        progress_every=0,
        diffusivity=3e-10,
        gradient_method="least_squares",
        laplacian_method="lsq_flux",
        backend="numpy",
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    hist = result["history"]
    assert len(hist) == 2
    assert np.all(np.isfinite(np.asarray(result["scalar"], dtype=np.float64)))
    assert "diffusion_laplacian_max_abs" in hist[-1]


def test_transport_tpfa_default_avoids_lsq_flux_startup_clipping_on_balanced_case() -> (
    None
):
    mesh, face_normal_velocity, flux_diag = _build_transport_flux(
        "two_inlets_to_outlet_tj_balanced"
    )

    lsq_cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=1,
        dt=1e-3,
        dt_mode="manual",
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
        diffusivity=3e-10,
        gradient_method="least_squares",
        laplacian_method="lsq_flux",
        backend="numpy",
    )
    tpfa_cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=1,
        dt=1e-3,
        dt_mode="manual",
        clipping_enabled=True,
        cfl_limit=0.8,
        progress_every=0,
        diffusivity=3e-10,
        gradient_method="least_squares",
        laplacian_method="tpfa",
        backend="numpy",
    )

    lsq_result = run_tetra_transport_debug(
        mesh,
        lsq_cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    tpfa_result = run_tetra_transport_debug(
        mesh,
        tpfa_cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert lsq_result["clipping"]["post_step_clipping_used"] is True
    assert float(lsq_result["history"][0]["undershoot_before_clip"]) > 1e-12
    assert tpfa_result["clipping"]["post_step_clipping_used"] is False
    assert float(tpfa_result["history"][0]["undershoot_before_clip"]) == 0.0


def test_two_cell_interior_face_pairwise_conservation_and_bounded_raw_update() -> None:
    mesh = _build_two_cell_mesh()
    scalar = np.asarray([1.0, 0.0], dtype=np.float64)
    boundary_groups = _build_boundary_face_groups(
        mesh,
        left_faces=np.zeros((0,), dtype=np.int64),
        right_faces=np.zeros((0,), dtype=np.int64),
        outlet_faces=np.zeros((0,), dtype=np.int64),
        wall_faces=np.asarray(mesh.boundary_face_indices, dtype=np.int64),
    )

    interior_face = int(mesh.interior_face_indices[0])
    owner = int(mesh.face_to_cells[interior_face, 0])
    neigh = int(mesh.face_to_cells[interior_face, 1])
    min_vol = float(min(mesh.cell_volumes[owner], mesh.cell_volumes[neigh]))
    dt = 0.2
    q = 0.25 * min_vol / dt
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    face_normal_velocity[interior_face] = q / float(mesh.face_areas[interior_face])

    assembled = _assemble_advection_fluxes(
        mesh,
        scalar,
        face_normal_velocity,
        boundary_face_groups=boundary_groups,
        left_inlet_value=0.0,
        right_inlet_value=1.0,
        dt=dt,
    )
    limited = _apply_bounded_limiter(mesh, scalar, assembled, scheme="upwind", dt=dt)
    raw_state = np.asarray(limited["raw_state_before_limiter"], dtype=np.float64)

    mass_old = float(np.sum(scalar * mesh.cell_volumes))
    mass_new = float(np.sum(raw_state * mesh.cell_volumes))
    assert abs(mass_new - mass_old) <= 1e-12
    assert float(np.max(raw_state)) <= 1.0 + 1e-12
    assert float(np.min(raw_state)) >= -1e-12
    assert float(assembled["pairwise_conservation_error_max_abs"]) <= 1e-12


def test_one_cell_inlet_outlet_stirred_cell_formula_for_raw_update() -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    scalar = np.asarray([0.0], dtype=np.float64)
    dt = 0.1
    volume = float(mesh.cell_volumes[0])
    q = 0.2 * volume / dt
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    boundary_groups = np.full(
        mesh.face_vertices.shape[0],
        BOUNDARY_GROUP_CODE["wall"],
        dtype=np.int32,
    )

    inlet_face = int(mesh.inlet_faces[0])
    outlet_face = int(mesh.outlet_faces[0])
    boundary_groups[inlet_face] = BOUNDARY_GROUP_CODE["right_inlet"]
    boundary_groups[outlet_face] = BOUNDARY_GROUP_CODE["outlet"]
    face_normal_velocity[inlet_face] = -q / float(
        mesh.face_areas[inlet_face]
    )  # inflow into cell
    face_normal_velocity[outlet_face] = q / float(
        mesh.face_areas[outlet_face]
    )  # outflow

    assembled = _assemble_advection_fluxes(
        mesh,
        scalar,
        face_normal_velocity,
        boundary_face_groups=boundary_groups,
        left_inlet_value=0.0,
        right_inlet_value=1.0,
        dt=dt,
    )
    limited = _apply_bounded_limiter(mesh, scalar, assembled, scheme="upwind", dt=dt)
    raw_state = np.asarray(limited["raw_state_before_limiter"], dtype=np.float64)

    expected = (dt * q * 1.0) / volume
    assert np.isfinite(raw_state[0])
    assert abs(float(raw_state[0]) - expected) <= 1e-12
    assert float(raw_state[0]) <= 1.0 + 1e-12


def test_mass_diagnostics_fields_present_and_consistent() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection",
        transport_scheme="bounded_upwind",
        steps=10,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=False,
        progress_every=0,
        backend="numpy",
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    masses = result["masses"]
    scalar_mass_initial = float(masses["scalar_mass_initial"])
    scalar_mass_final = float(masses["scalar_mass_final"])
    assert np.isfinite(scalar_mass_initial)
    assert np.isfinite(scalar_mass_final)
    assert float(masses["initial_mass_proxy"]) == scalar_mass_initial
    assert float(masses["final_mass_proxy"]) == scalar_mass_final
    assert float(masses["total_mass_proxy_final"]) == scalar_mass_final


def test_axis_aligned_clean_velocity_field_builds_and_has_outlet_diagnostics() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_clean",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    assert face_normal_velocity.shape[0] == mesh.face_vertices.shape[0]
    assert float(flux_diag["wall_flux_max_abs"]) <= 1e-10
    diag = compute_velocity_field_diagnostics(
        mesh, velocity, flux_diagnostics=flux_diag
    )
    assert "outlet_branch_direction" in diag


def test_tolerance_cleanliness_passes_when_strict_fails() -> None:
    flags = _compute_cleanliness_flags(
        finite=True,
        cfl_warning=False,
        diffusion_stability_warning=False,
        overshoot_max=3.7e-7,
        undershoot_max=7.5e-7,
        clipped_count=42,
        tolerance=1e-6,
    )
    assert flags["strict_numerically_clean_transport"] is False
    assert flags["tolerance_numerically_clean_transport"] is True


def test_safety_clamp_after_diffusion_reports_metrics() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_clean",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=4,
        dt_mode="auto",
        cfl_target=0.5,
        clipping_enabled=False,
        safety_clamp_after_diffusion=True,
        diffusivity=3e-10,
        progress_every=0,
        backend="numpy",
    )
    result = run_tetra_transport_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    scalar = np.asarray(result["scalar"], dtype=np.float64)
    assert np.min(scalar) >= -1e-12
    assert np.max(scalar) <= 1.0 + 1e-12
    clipping = result["clipping"]
    assert "safety_clamp_mass_delta" in clipping
    assert "safety_clamp_after_diffusion_used" in clipping


def test_transport_refuses_incompatible_flow_run_mesh() -> None:
    mesh = _build_synthetic_mesh()
    bad_flux = np.zeros(mesh.face_vertices.shape[0] + 3, dtype=np.float64)
    payload = {
        "metadata": {
            "mesh_stats": {
                "tetra_count": int(mesh.tetrahedra.shape[0]),
                "face_count": int(mesh.face_vertices.shape[0] + 3),
            }
        },
        "face_flux": bad_flux,
        "face_to_cells": np.asarray(mesh.face_to_cells, dtype=np.int64),
        "cell_volumes": np.asarray(mesh.cell_volumes, dtype=np.float64),
    }
    compat = _check_flow_transport_mesh_compatibility(mesh=mesh, flow_payload=payload)
    assert compat["compatible"] is False
    assert compat["face_count_ok"] is False


def test_transport_loads_face_flux_from_flow_run_payload(tmp_path: Path) -> None:
    mesh = _build_synthetic_mesh()
    flux = np.linspace(0.0, 1.0, mesh.face_vertices.shape[0], dtype=np.float64)
    run_dir = tmp_path / "flow_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "final_corrected_face_flux.npy", flux)
    np.save(
        run_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        run_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    md = {
        "ready_for_flow_to_transport_coupling": True,
        "mesh_stats": {
            "tetra_count": int(mesh.tetrahedra.shape[0]),
            "face_count": int(mesh.face_vertices.shape[0]),
        },
    }
    (run_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(md), encoding="utf-8"
    )
    payload = _load_flow_coupling_payload(run_dir)
    assert np.array_equal(np.asarray(payload["face_flux"], dtype=np.float64), flux)
    compat = _check_flow_transport_mesh_compatibility(mesh=mesh, flow_payload=payload)
    assert compat["compatible"] is True


def _save_mesh_npz(mesh, out_path: Path) -> None:
    np.savez(
        out_path,
        source_path=str(mesh.source_path),
        points=np.asarray(mesh.points, dtype=np.float64),
        tetrahedra=np.asarray(mesh.tetrahedra, dtype=np.int64),
        boundary_triangles=np.asarray(mesh.boundary_triangles, dtype=np.int64),
        boundary_face_tags=np.asarray(mesh.boundary_face_tags, dtype=np.int32),
        cell_centers=np.asarray(mesh.cell_centers, dtype=np.float64),
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
        face_vertices=np.asarray(mesh.face_vertices, dtype=np.int64),
        face_centers=np.asarray(mesh.face_centers, dtype=np.float64),
        face_areas=np.asarray(mesh.face_areas, dtype=np.float64),
        face_normals=np.asarray(mesh.face_normals, dtype=np.float64),
        face_to_cells=np.asarray(mesh.face_to_cells, dtype=np.int64),
        cell_to_faces=np.asarray(mesh.cell_to_faces, dtype=np.int64),
        boundary_tag_per_face=np.asarray(mesh.boundary_tag_per_face, dtype=np.int32),
        interior_face_indices=np.asarray(mesh.interior_face_indices, dtype=np.int64),
        boundary_face_indices=np.asarray(mesh.boundary_face_indices, dtype=np.int64),
        inlet_faces=np.asarray(mesh.inlet_faces, dtype=np.int64),
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
        unresolved_faces=np.asarray(mesh.boundary_unresolved_faces, dtype=np.int64),
        boundary_face_names_json=json.dumps(
            {str(k): str(v) for k, v in mesh.boundary_face_names.items()}
        ),
        physical_groups_json=json.dumps(
            {str(k): [int(v[0]), int(v[1])] for k, v in mesh.physical_groups.items()}
        ),
        diagnostics_json=json.dumps({}),
    )


def test_clean_flow_metadata_allows_coupling_in_strict_mode() -> None:
    out = _validate_source_flow_readiness(
        {"ready_for_flow_to_transport_coupling": True},
        strict=True,
    )
    assert out["ready"] is True
    assert out["warning"] is False


def test_nonphysical_flow_metadata_blocks_coupling_in_strict_mode() -> None:
    try:
        _validate_source_flow_readiness(
            {
                "ready_for_flow_to_transport_coupling": False,
                "nonphysical_flux_fix_used": True,
            },
            strict=True,
        )
    except RuntimeError:
        return
    raise AssertionError("Expected RuntimeError for strict non-ready source flow")


def test_ready_flow_metadata_with_nonphysical_fix_is_rejected_in_strict_mode() -> None:
    with pytest.raises(RuntimeError, match="nonphysical_flux_fix_used"):
        _validate_source_flow_readiness(
            {
                "ready_for_flow_to_transport_coupling": True,
                "nonphysical_flux_fix_used": True,
            },
            strict=True,
        )


def test_ready_flow_metadata_with_convective_auto_damping_is_rejected_in_strict_mode() -> (
    None
):
    with pytest.raises(RuntimeError, match="convective_auto_damping_used_any"):
        _validate_source_flow_readiness(
            {
                "ready_for_flow_to_transport_coupling": True,
                "convective_auto_damping_used_any": True,
            },
            strict=True,
        )


def test_transport_readiness_wrapper_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_shared_validate(
        metadata: dict[str, object],
        *,
        strict: bool,
        strict_label: str,
    ) -> dict[str, object]:
        calls.append(
            {
                "metadata": dict(metadata),
                "strict": bool(strict),
                "strict_label": str(strict_label),
            }
        )
        return {"ready": True, "warning": False}

    monkeypatch.setattr(
        transport_debug_module,
        "_shared_validate_source_flow_readiness",
        _fake_shared_validate,
    )

    out = transport_debug_module._validate_source_flow_readiness(
        {"ready_for_flow_to_transport_coupling": True},
        strict=True,
    )

    assert out == {"ready": True, "warning": False}
    assert calls == [
        {
            "metadata": {"ready_for_flow_to_transport_coupling": True},
            "strict": True,
            "strict_label": "transport source flow",
        }
    ]


def test_flow_stage_status_ready_flag_overrides_legacy_ready_flag() -> None:
    out = _validate_source_flow_readiness(
        {
            "ready_for_flow_to_transport_coupling": False,
            "stage_status": {
                "run_completed": True,
                "numerically_stable": True,
                "physically_ready": True,
                "ready_for_next_stage": True,
                "ready_for_long_run": True,
                "stage_status_reason": "unit",
            },
        },
        strict=True,
    )
    assert out["ready"] is True
    assert out["run_completed"] is True
    assert out["numerically_stable"] is True
    assert out["physically_ready"] is True
    assert out["ready_for_long_run"] is True
    assert out["stage_status_reason"] == "unit"


def test_transport_front_observation_records_short_run_without_rejection() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    _, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    observation = _evaluate_transport_front_observation(
        mesh=mesh,
        cell_velocity=np.asarray(velocity.cell_velocity, dtype=np.float64),
        flux_diag=flux_diag,
        physical_time_final=1e-8,
        outlet_arrival={},
        outlet_c_mean=0.0,
    )
    assert observation["breakthrough_expected"] is False
    assert observation["breakthrough_detected"] is False
    assert observation["observation"] == "breakthrough_not_detected"
    assert "outlet_fraction_detection_threshold" in observation


def test_transport_front_observation_detects_breakthrough_when_expected() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    _, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    outlet_arrival = {format_threshold_key("outlet_frac_gt", 1e-6): 1e-3}
    observation = _evaluate_transport_front_observation(
        mesh=mesh,
        cell_velocity=np.asarray(velocity.cell_velocity, dtype=np.float64),
        flux_diag=flux_diag,
        physical_time_final=1.0e4,
        outlet_arrival=outlet_arrival,
        outlet_c_mean=1e-4,
    )
    assert observation["breakthrough_expected"] is True
    assert observation["breakthrough_detected"] is True
    assert observation["observation"] == "breakthrough_detected"


def test_transport_checkpoint_save_and_load_roundtrip(tmp_path: Path) -> None:
    c = np.linspace(0.0, 1.0, 7, dtype=np.float64)
    state_path = _save_transport_checkpoint(
        run_dir=tmp_path,
        step=25,
        concentration=c,
        cumulative_mass_in=1.23,
        cumulative_mass_out=0.77,
        physical_time=0.005,
        used_dt=2e-5,
        history_tail={"step": 25, "cfl_max": 0.5},
        extra_metadata={"tag": "unit"},
    )
    loaded = _load_transport_checkpoint(state_path)
    assert loaded["step"] == 25
    assert np.allclose(loaded["concentration"], c)
    assert abs(float(loaded["cumulative_mass_in"]) - 1.23) < 1e-15
    assert abs(float(loaded["cumulative_mass_out"]) - 0.77) < 1e-15
    assert loaded["metadata"]["tag"] == "unit"


def test_resume_from_checkpoint_preserves_cumulative_mass_diagnostics(
    tmp_path: Path,
) -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    base_cfg = GmshTetraTransportConfig(
        transport_mode="advection_diffusion",
        transport_scheme="bounded_upwind",
        steps=6,
        dt_mode="auto",
        cfl_target=0.5,
        progress_every=0,
        backend="numpy",
        snapshot_steps=(3, 6),
    )
    direct = run_tetra_transport_debug(
        mesh,
        base_cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    run_a = _run_transport_with_checkpointing(
        mesh=mesh,
        base_config=base_cfg,
        run_transport_fn=run_tetra_transport_debug,
        face_normal_velocity=face_normal_velocity,
        flux_diag=flux_diag,
        snapshot_steps=(3, 6),
        checkpoint_every=3,
        max_walltime_seconds=0.0,
        run_dir=tmp_path / "phase_a",
        resume_state=None,
    )
    ckpts = list(run_a["run_control"]["checkpoint_paths"])
    assert len(ckpts) >= 1
    resume = _load_transport_checkpoint(Path(ckpts[0]))
    run_b = _run_transport_with_checkpointing(
        mesh=mesh,
        base_config=base_cfg,
        run_transport_fn=run_tetra_transport_debug,
        face_normal_velocity=face_normal_velocity,
        flux_diag=flux_diag,
        snapshot_steps=(3, 6),
        checkpoint_every=0,
        max_walltime_seconds=0.0,
        run_dir=tmp_path / "phase_b",
        resume_state=resume,
    )
    m_direct = dict(direct["masses"])
    m_resumed = dict(run_b["masses"])
    assert (
        abs(
            float(m_direct["cumulative_mass_in"])
            - float(m_resumed["cumulative_mass_in"])
        )
        < 1e-12
    )
    assert (
        abs(
            float(m_direct["cumulative_mass_out"])
            - float(m_resumed["cumulative_mass_out"])
        )
        < 1e-12
    )


def test_flow_run_mode_writes_snapshots_summary_json(tmp_path: Path) -> None:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, _ = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )
    flow_dir = tmp_path / "flow_run"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    np.save(
        flow_dir / "face_groups.npy",
        np.zeros(mesh.face_vertices.shape[0], dtype=np.int32),
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": True,
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    cmd = [
        sys.executable,
        str(script),
        "--mesh-npz",
        str(mesh_npz),
        "--import-root",
        str(import_root),
        "--output-root",
        str(output_root),
        "--backend",
        "numpy",
        "--transport-execution-backend",
        "numpy",
        "--velocity-source",
        "flow_run",
        "--flow-run-dir",
        str(flow_dir),
        "--transport-mode",
        "advection_diffusion",
        "--transport-scheme",
        "bounded_upwind",
        "--steps",
        "5",
        "--snapshot-steps",
        "2,5",
        "--max-supported-grid-peclet",
        "1e12",
        "--max-supported-schmidt",
        "5000",
        "--progress-every",
        "0",
        "--no-velocity-comparison",
        "--no-transport-scheme-comparison",
    ]
    subprocess.run(cmd, check=True)
    run_dirs = sorted(output_root.glob("*_synthetic_imported_mesh"))
    assert run_dirs, "expected at least one transport run dir"
    run_dir = run_dirs[-1]
    snapshots_summary_path = run_dir / "snapshots_summary.json"
    assert snapshots_summary_path.exists()
    snapshots_summary = json.loads(snapshots_summary_path.read_text(encoding="utf-8"))
    assert isinstance(snapshots_summary.get("snapshots", []), list)
    assert len(snapshots_summary.get("snapshots", [])) >= 1

    acceptance_report_path = run_dir / "acceptance_report.json"
    stage_status_path = run_dir / "stage_status.json"
    regime_audit_path = run_dir / "transport_regime_audit.json"
    flow_snapshot_path = run_dir / "flow_coupling_metadata_snapshot.json"
    assert acceptance_report_path.exists()
    assert stage_status_path.exists()
    assert regime_audit_path.exists()
    assert flow_snapshot_path.exists()

    acceptance_report = json.loads(acceptance_report_path.read_text(encoding="utf-8"))
    stage_status = json.loads(stage_status_path.read_text(encoding="utf-8"))
    regime_audit = json.loads(regime_audit_path.read_text(encoding="utf-8"))
    flow_snapshot = json.loads(flow_snapshot_path.read_text(encoding="utf-8"))
    assert "ready_for_next_stage" in acceptance_report
    assert acceptance_report["transport_accuracy_supported"] is True
    assert acceptance_report["transport_regime_audit"]["support_status"] == "supported"
    assert "stage_status_reason" in stage_status
    assert stage_status["stage_status_checks"]["transport_accuracy_supported"] is True
    assert regime_audit["support_status"] == "supported"
    assert "source_flow_metadata" in flow_snapshot


def test_transport_cli_default_regime_warnings_do_not_block_readiness(
    tmp_path: Path,
) -> None:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, _ = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )
    flow_dir = tmp_path / "flow_run_ready"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    np.save(
        flow_dir / "final_cell_velocity.npy",
        np.asarray(velocity.cell_velocity, dtype=np.float64),
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": True,
                "stage_status": {
                    "run_completed": True,
                    "numerically_stable": True,
                    "physically_ready": True,
                    "ready_for_next_stage": True,
                    "ready_for_long_run": True,
                    "stage_status_reason": "unit-ready",
                },
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_dir),
            "--steps",
            "1",
            "--dt",
            "1e-7",
            "--progress-every",
            "0",
            "--no-velocity-comparison",
            "--no-transport-scheme-comparison",
        ],
        check=True,
    )

    run_dirs = sorted(output_root.glob("*_synthetic_imported_mesh"))
    assert run_dirs, "expected at least one transport run dir"
    run_dir = run_dirs[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    stage_status = json.loads(
        (run_dir / "stage_status.json").read_text(encoding="utf-8")
    )
    acceptance_report = json.loads(
        (run_dir / "acceptance_report.json").read_text(encoding="utf-8")
    )

    assert summary["transport_accuracy_supported"] is False
    assert summary["ready_for_next_stage"] is True
    assert summary["ready_for_long_run"] is False
    assert stage_status["stage_status_checks"]["transport_accuracy_supported"] is False
    assert stage_status["stage_status_checks"]["transport_regime_warning"] is True
    assert (
        stage_status["stage_status_checks"]["transport_regime_blocking_error"] is False
    )
    assert "transport_regime_warning_codes" not in stage_status["stage_status_checks"]
    assert "warning_high_schmidt" in stage_status["stage_status_warnings"]
    assert acceptance_report["ready_for_next_stage"] is True
    assert "warning_high_schmidt" in acceptance_report["transport_regime_warning_codes"]


def test_transport_cli_rejects_unready_flow_by_default(tmp_path: Path) -> None:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    flow_dir = tmp_path / "flow_run_unready"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        flow_dir / "final_corrected_face_flux.npy",
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
    )
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": False,
                "stage_status": {
                    "run_completed": True,
                    "numerically_stable": True,
                    "physically_ready": False,
                    "ready_for_next_stage": False,
                    "ready_for_long_run": False,
                    "stage_status_reason": "unit-unready",
                },
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_dir),
            "--steps",
            "5",
            "--progress-every",
            "0",
            "--no-velocity-comparison",
            "--no-transport-scheme-comparison",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "transport source flow readiness is false" in completed.stderr


def test_transport_cli_debug_override_allows_unready_flow(tmp_path: Path) -> None:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, _ = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )

    flow_dir = tmp_path / "flow_run_override"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    np.save(
        flow_dir / "final_cell_velocity.npy",
        np.asarray(velocity.cell_velocity, dtype=np.float64),
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": False,
                "stage_status": {
                    "run_completed": True,
                    "numerically_stable": True,
                    "physically_ready": False,
                    "ready_for_next_stage": False,
                    "ready_for_long_run": False,
                    "stage_status_reason": "unit-unready",
                },
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_dir),
            "--steps",
            "5",
            "--max-supported-grid-peclet",
            "1e12",
            "--max-supported-schmidt",
            "5000",
            "--allow-unready-flow",
            "--progress-every",
            "0",
            "--no-velocity-comparison",
            "--no-transport-scheme-comparison",
        ],
        check=True,
    )
    run_dirs = sorted(output_root.glob("*_synthetic_imported_mesh"))
    assert run_dirs, "expected at least one transport run dir"
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["source_flow_ready_for_transport"] is False
    assert summary["breakthrough_detected"] is False
    assert summary["transport_time_estimate"]["breakthrough_expected"] is False


def test_transport_cli_allows_short_physical_exposure(
    tmp_path: Path,
) -> None:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, _ = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )

    flow_dir = tmp_path / "flow_run_ready"
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    np.save(
        flow_dir / "final_cell_velocity.npy",
        np.asarray(velocity.cell_velocity, dtype=np.float64),
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": True,
                "stage_status": {
                    "run_completed": True,
                    "numerically_stable": True,
                    "physically_ready": True,
                    "ready_for_next_stage": True,
                    "ready_for_long_run": True,
                    "stage_status_reason": "unit-ready",
                },
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    script = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_transport_debug.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(output_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_dir),
            "--steps",
            "5",
            "--max-supported-grid-peclet",
            "1e12",
            "--max-supported-schmidt",
            "5000",
            "--progress-every",
            "0",
            "--no-velocity-comparison",
            "--no-transport-scheme-comparison",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    run_dirs = sorted(output_root.glob("*_synthetic_imported_mesh"))
    assert run_dirs, "expected a completed short transport run"
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["breakthrough_detected"] is False
    assert summary["transport_time_estimate"]["breakthrough_expected"] is False
