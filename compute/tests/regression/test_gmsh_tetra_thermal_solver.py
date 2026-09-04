from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra import gmsh_tetra_thermal_solver as thermal_solver_module
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    build_face_normal_flux_from_velocity,
)
from microfluidics.gmsh.tetra.gmsh_tetra_thermal_solver import (
    GmshTetraThermalConfig,
    run_tetra_thermal_debug,
)
from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (
    build_prescribed_velocity_field,
)


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
        source_path=Path("thermal_synth.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=btri,
        boundary_face_tags=btags,
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
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
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
        source_path=Path("one_cell_thermal.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data=field_data,
    )


def _build_slab_mesh(nx: int = 9):
    xs = np.linspace(0.0, 1.0, nx)
    points = np.asarray(
        [[x, y, z] for x in xs for z in (0.0, 1.0) for y in (0.0, 1.0)],
        dtype=np.float64,
    )

    def node(i: int, y: int, z: int) -> int:
        return i * 4 + z * 2 + y

    tetrahedra: list[list[int]] = []
    for i in range(nx - 1):
        v000 = node(i, 0, 0)
        v100 = node(i + 1, 0, 0)
        v010 = node(i, 1, 0)
        v110 = node(i + 1, 1, 0)
        v001 = node(i, 0, 1)
        v101 = node(i + 1, 0, 1)
        v011 = node(i, 1, 1)
        v111 = node(i + 1, 1, 1)
        tetrahedra.extend(
            [
                [v000, v100, v110, v111],
                [v000, v110, v010, v111],
                [v000, v010, v011, v111],
                [v000, v011, v001, v111],
                [v000, v001, v101, v111],
                [v000, v101, v100, v111],
            ]
        )
    tets = np.asarray(tetrahedra, dtype=np.int64)
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in tets:
        for tri in (
            (int(tet[1]), int(tet[2]), int(tet[3])),
            (int(tet[0]), int(tet[3]), int(tet[2])),
            (int(tet[0]), int(tet[1]), int(tet[3])),
            (int(tet[0]), int(tet[2]), int(tet[1])),
        ):
            face_map.setdefault(tuple(sorted(tri)), []).append(tri)
    boundary_triangles: list[tuple[int, int, int]] = []
    boundary_tags: list[int] = []
    for entries in face_map.values():
        if len(entries) != 1:
            continue
        tri = entries[0]
        center = np.mean(points[np.asarray(tri, dtype=np.int64)], axis=0)
        if center[0] <= 1e-12:
            tag = 1 if center[2] < 0.5 else 2
        elif center[0] >= 1.0 - 1e-12:
            tag = 5
        else:
            tag = 4
        boundary_triangles.append(tri)
        boundary_tags.append(tag)
    return build_imported_tetra_mesh(
        source_path=Path("thermal_slab.msh"),
        points=points,
        tetrahedra=tets,
        boundary_triangles=np.asarray(boundary_triangles, dtype=np.int64),
        boundary_face_tags=np.asarray(boundary_tags, dtype=np.int32),
        field_data={
            "left_inlet": np.asarray([1, 2], dtype=np.int32),
            "right_inlet": np.asarray([2, 2], dtype=np.int32),
            "walls": np.asarray([4, 2], dtype=np.int32),
            "heated_wall": np.asarray([5, 2], dtype=np.int32),
        },
    )


def _prescribed_velocity_inputs(mesh):
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
    return face_normal_velocity, flux_diag


def _backend_config_kwargs(backend: str) -> dict[str, object]:
    if backend == "torch":
        try:
            __import__("torch")
        except ModuleNotFoundError:
            pytest.skip("torch is not available")
        return {"backend": "torch", "torch_device": "cpu"}
    return {"backend": "numpy"}


def test_thermal_bc_one_cell_advective_mixing_formula() -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    inlet_face = int(mesh.inlet_faces[0])
    outlet_face = int(mesh.outlet_faces[0])

    dt = 0.1
    volume = float(mesh.cell_volumes[0])
    q = 0.2 * volume / dt
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    face_normal_velocity[inlet_face] = -q / float(mesh.face_areas[inlet_face])
    face_normal_velocity[outlet_face] = q / float(mesh.face_areas[outlet_face])

    cfg = GmshTetraThermalConfig(
        steps=1,
        dt=dt,
        dt_mode="manual",
        thermal_diffusivity=0.0,
        heat_source=0.0,
        initial_temperature=300.0,
        left_inlet_temperature=350.0,
        right_inlet_temperature=350.0,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh, cfg, face_normal_velocity=face_normal_velocity
    )
    temperature = np.asarray(result["temperature"], dtype=np.float64)

    expected = 300.0 + 0.2 * (350.0 - 300.0)
    assert abs(float(temperature[0]) - expected) <= 1e-10
    assert abs(float(result["history"][0]["raw_balance_error"])) <= 1e-12


def test_thermal_auto_dt_respects_advection_and_diffusion_limits() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    cfg = GmshTetraThermalConfig(
        steps=4,
        dt=1.0,
        dt_mode="auto",
        cfl_target=0.45,
        cfl_limit=0.8,
        diffusion_stability_factor=0.9,
        thermal_diffusivity=1e-5,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    dt_control = result["dt_control"]
    used_dt = float(dt_control["used_dt"])
    adv_lim = float(dt_control["advection_dt_limit"])
    diff_lim = float(dt_control["diffusion_dt_limit"])

    if np.isfinite(adv_lim):
        assert used_dt <= cfg.cfl_target * adv_lim + 1e-14
    if np.isfinite(diff_lim):
        assert used_dt <= cfg.diffusion_stability_factor * diff_lim + 1e-14

    assert np.all(np.isfinite(np.asarray(result["temperature"], dtype=np.float64)))


def test_thermal_reports_model_capabilities_and_observables() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    cfg = GmshTetraThermalConfig(
        steps=3,
        dt=1e-3,
        dt_mode="manual",
        thermal_diffusivity=0.0,
        heat_source=0.0,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    capabilities = result["model_capabilities"]
    observables = result["thermal_observables"]

    assert capabilities["contract_version"] == "thermal_v2"
    assert capabilities["model_family"] == "passive_scalar_advection_diffusion"
    assert "wall_fixed_temperature" in capabilities["supported_features"]
    assert "wall_prescribed_heat_flux" in capabilities["supported_features"]
    assert "conjugate_heat_transfer" in capabilities["unsupported_features"]
    assert (
        "buoyancy_or_temperature_to_flow_coupling"
        in capabilities["unsupported_features"]
    )
    assert observables["temperature_units"] == "K"
    assert np.isfinite(float(observables["domain_volume_weighted_temperature"]))
    assert observables["outlet_sample_face_count"] == int(mesh.outlet_faces.size)
    assert observables["outlet_sample_cell_count"] > 0
    assert observables["outlet_temperature_cell_mean"] is not None


def test_thermal_boundedness_clipping_prevents_overshoot() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)

    cfg = GmshTetraThermalConfig(
        steps=3,
        dt=0.5,
        dt_mode="manual",
        thermal_diffusivity=0.0,
        heat_source=1e8,
        rho=1000.0,
        cp=1000.0,
        initial_temperature=299.0,
        min_temperature=295.0,
        max_temperature=305.0,
        clipping_enabled=True,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh, cfg, face_normal_velocity=face_normal_velocity
    )
    temperature = np.asarray(result["temperature"], dtype=np.float64)

    assert np.all(np.isfinite(temperature))
    assert float(np.max(temperature)) <= 305.0 + 1e-12
    assert float(np.min(temperature)) >= 295.0 - 1e-12

    non_adv = result["non_advective_limiter"]
    assert bool(non_adv["enabled"]) is True
    assert int(non_adv["final_active_cell_count"]) > 0
    clipping = result["clipping"]
    assert int(clipping["final_clipped_cell_count"]) >= 0


def test_thermal_energy_proxy_consistency_without_clamp() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    cfg = GmshTetraThermalConfig(
        steps=8,
        dt=1e-3,
        dt_mode="auto",
        cfl_target=0.5,
        thermal_diffusivity=0.0,
        heat_source=0.0,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    energy = result["energy_proxy"]
    assert abs(float(energy["cumulative_clamp_integral_change"])) <= 1e-14

    final_energy = float(energy["final_integral_T_dV"])
    expected = float(energy["expected_final_without_clamp"])
    err = abs(final_energy - expected)
    scale = max(abs(final_energy), abs(expected), 1.0)
    assert err / scale <= 1e-10


def test_thermal_smoke_advection_diffusion_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    cfg = GmshTetraThermalConfig(
        steps=5,
        dt=2e-3,
        dt_mode="auto",
        cfl_target=0.5,
        cfl_limit=0.8,
        thermal_diffusivity=1.4e-7,
        heat_source=2e5,
        clipping_enabled=True,
        min_temperature=280.0,
        max_temperature=420.0,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    temperature = np.asarray(result["temperature"], dtype=np.float64)

    assert temperature.shape[0] == mesh.tetrahedra.shape[0]
    assert np.all(np.isfinite(temperature))
    assert "history" in result
    assert [int(step["step"]) for step in result["history"]] == [1, cfg.steps]
    assert "dt_control" in result
    assert "boundary_setup" in result
    assert "local_cfl_audit" in result
    assert "non_advective_limiter" in result


def test_thermal_bounded_case_reduces_mass_clipping_on_synthetic_mesh() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    cfg = GmshTetraThermalConfig(
        steps=20,
        dt=5e-3,
        dt_mode="auto",
        cfl_target=0.5,
        thermal_diffusivity=1.4e-7,
        heat_source=2.5e5,
        clipping_enabled=True,
        min_temperature=290.0,
        max_temperature=400.0,
        progress_every=0,
        history_stride=1,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    history = result["history"]
    clipped_avg = float(
        np.mean([float(step["clipped_cell_count"]) for step in history])
    )
    n_cells = float(mesh.tetrahedra.shape[0])
    assert clipped_avg / max(1.0, n_cells) < 0.5
    assert float(result["non_advective_limiter"]["final_scale_min"]) >= 0.0


def test_thermal_normalized_bounded_advection_assembles_fluxes_once_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    original = thermal_solver_module.assemble_advection_fluxes_numpy
    call_count = {"value": 0}

    def _counted(*args, **kwargs):
        call_count["value"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        thermal_solver_module,
        "assemble_advection_fluxes_numpy",
        _counted,
    )

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=1,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=0.0,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            normalize_bounded_advection=True,
            limiter_scheme="bounded_upwind",
            progress_every=0,
            backend="numpy",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert call_count["value"] == 1
    assert result["history"][0]["advection_flux_assemblies"] == 1
    assert result["performance_diagnostics"]["advection_flux_assemblies_per_step"] == 1


def test_thermal_velocity_input_modes_prescribed_vs_face_flux_equivalent() -> None:
    mesh = _build_synthetic_mesh()
    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_axis_aligned_clean",
        inlet_speed=0.15,
    )
    vn_prescribed, flux_diag = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = vn_prescribed * np.asarray(mesh.face_areas, dtype=np.float64)
    vn_flow_solver_mode = face_flux / np.maximum(mesh.face_areas, 1e-16)

    cfg = GmshTetraThermalConfig(
        steps=6,
        dt=2e-3,
        dt_mode="auto",
        cfl_target=0.5,
        thermal_diffusivity=1.4e-7,
        clipping_enabled=True,
        min_temperature=280.0,
        max_temperature=420.0,
        progress_every=0,
    )

    result_a = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=vn_prescribed,
        flux_diagnostics=flux_diag,
        velocity_metadata={"velocity_source": "prescribed"},
    )
    result_b = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=vn_flow_solver_mode,
        flux_diagnostics=flux_diag,
        velocity_metadata={"velocity_source": "flow_solver"},
    )

    ta = np.asarray(result_a["temperature"], dtype=np.float64)
    tb = np.asarray(result_b["temperature"], dtype=np.float64)
    assert np.max(np.abs(ta - tb)) <= 1e-12


def test_thermal_backend_auto_uses_numpy_when_torch_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    original_select_backend = thermal_solver_module.select_backend

    def _fake_select_backend(requested: str = "auto"):
        return original_select_backend(
            requested,
            torch_available_override=False,
            torch_cuda_available_override=False,
        )

    monkeypatch.setattr(thermal_solver_module, "select_backend", _fake_select_backend)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=2,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            progress_every=0,
            backend="auto",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    be = result["backend_execution"]
    assert be["stepping_backend"] == "numpy"
    assert be["used_numpy_fallback"] is True


def test_thermal_production_profile_rejects_numpy_without_override() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    with pytest.raises(RuntimeError, match="production profile requires torch backend"):
        run_tetra_thermal_debug(
            mesh,
            GmshTetraThermalConfig(
                steps=2,
                dt=1e-3,
                dt_mode="manual",
                thermal_diffusivity=1.4e-7,
                progress_every=0,
                backend="numpy",
                execution_profile="production",
            ),
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diag,
        )


def test_thermal_production_profile_allows_numpy_with_explicit_override() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=2,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            progress_every=0,
            backend="numpy",
            execution_profile="production",
            allow_numpy_production_fallback=True,
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    policy = result["execution_policy"]
    assert policy["execution_profile"] == "production"
    assert policy["selected_backend"] == "numpy"
    assert policy["production_backend_satisfied"] is False
    assert policy["degraded_execution_mode"] is True


def test_thermal_torch_backend_reports_no_numpy_fallback() -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return

    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    device = "cuda:0" if bool(torch.cuda.is_available()) else "cpu"

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="auto",
            cfl_target=0.5,
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            progress_every=0,
            backend="torch",
            torch_device=device,
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    be = result["backend_execution"]
    assert be["stepping_backend"] == "torch"
    assert be["used_numpy_fallback"] is False
    assert result["diffusion_diagnostics"]["diffusion_backend"] == "torch"
    if device.startswith("cuda"):
        assert be["all_core_arrays_on_cuda"] is True


def test_thermal_numpy_torch_parity_on_synthetic_mesh() -> None:
    try:
        __import__("torch")
    except ModuleNotFoundError:
        return

    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    common = dict(
        steps=5,
        dt=2e-3,
        dt_mode="auto",
        cfl_target=0.5,
        cfl_limit=0.8,
        diffusion_stability_factor=1.0,
        thermal_diffusivity=1.4e-7,
        heat_source=2e5,
        clipping_enabled=True,
        min_temperature=280.0,
        max_temperature=420.0,
        progress_every=0,
    )

    numpy_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="numpy"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    torch_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="torch", torch_device="cpu"),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    np.testing.assert_allclose(
        np.asarray(numpy_result["temperature"], dtype=np.float64),
        np.asarray(torch_result["temperature"], dtype=np.float64),
        rtol=1e-6,
        atol=1e-8,
    )
    assert float(numpy_result["dt_control"]["used_dt"]) == float(
        torch_result["dt_control"]["used_dt"]
    )
    assert pytest.approx(
        float(numpy_result["energy_proxy"]["final_integral_T_dV"]),
        rel=1e-6,
        abs=1e-8,
    ) == float(torch_result["energy_proxy"]["final_integral_T_dV"])


def test_thermal_smoke_t_junction_no_nan_and_reasonable_clipping() -> None:
    root = Path(__file__).resolve().parents[3]
    msh = root / "data/meshes/gmsh" / "t_junction.msh"
    if not msh.exists():
        return

    mesh = import_gmsh_tetra_mesh(msh)
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
    cfg = GmshTetraThermalConfig(
        steps=20,
        dt=5e-4,
        dt_mode="auto",
        cfl_target=0.5,
        thermal_diffusivity=1.4354066985645933e-7,
        heat_source=2.5e5,
        initial_temperature=298.15,
        left_inlet_temperature=303.15,
        right_inlet_temperature=303.15,
        clipping_enabled=True,
        min_temperature=290.0,
        max_temperature=400.0,
        progress_every=0,
    )
    result = run_tetra_thermal_debug(
        mesh,
        cfg,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
        velocity_metadata={"velocity_source": "prescribed"},
    )
    t = np.asarray(result["temperature"], dtype=np.float64)
    assert np.all(np.isfinite(t))
    assert float(np.min(t)) >= 290.0 - 1e-12
    assert float(np.max(t)) <= 400.0 + 1e-12
    clipped_fraction = float(result["clipping"]["final_clipped_cell_count"]) / max(
        1, t.size
    )
    assert clipped_fraction < 0.5


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_collect_history_false_keeps_summary_and_snapshots(
    backend: str,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=6,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            collect_history=False,
            snapshot_steps=(2, 6),
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert result["history"] == []
    assert result["history_settings"]["recorded_steps"] == []
    assert set(result["snapshots"]) == {2, 6}
    assert "final_stats" in result
    assert "energy_proxy" in result
    assert np.all(np.isfinite(np.asarray(result["temperature"], dtype=np.float64)))


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_history_stride_and_snapshot_steps_are_respected(
    backend: str,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=10,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            history_stride=4,
            snapshot_steps=(3, 9),
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert result["history_settings"]["recorded_steps"] == [1, 3, 4, 8, 9, 10]
    assert [int(step["step"]) for step in result["history"]] == [1, 3, 4, 8, 9, 10]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_compact_history_mode_records_arrays_without_dict_entries(
    backend: str,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=5,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            history_mode="compact",
            history_stride=2,
            snapshot_steps=(3,),
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert result["history"] == []
    assert result["history_settings"]["resolved_history_mode"] == "compact"
    assert result["history_settings"]["recorded_steps"] == [1, 2, 3, 4, 5]
    assert result["history_compact"]["step"] == [1, 2, 3, 4, 5]
    assert len(result["history_compact"]["temperature_min"]) == 5
    assert len(result["history_compact"]["temperature_max"]) == 5
    assert len(result["history_compact"]["cfl_max"]) == 5


def test_thermal_fast_diagnostics_mode_keeps_history_lightweight() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=4,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            diagnostics_mode="fast",
            history_mode="dict",
            history_stride=1,
            progress_every=0,
            backend="numpy",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    entry = result["history"][0]
    assert result["history_settings"]["resolved_diagnostics_mode"] == "fast"
    assert "temperature_min" in entry
    assert "temperature_max" in entry
    assert "cfl_max" in entry
    assert "raw_balance_error" not in entry
    assert "clipped_cell_count" not in entry
    assert "limiter_active_fraction" not in entry


def test_thermal_torch_compile_step_body_uses_compile_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("torch is not available")

    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    compile_calls: list[object] = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile, raising=False)
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            diagnostics_mode="fast",
            history_mode="compact",
            history_stride=1,
            progress_every=0,
            backend="torch",
            torch_device="cpu",
            torch_compile_step_body="on",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    perf = result["performance_diagnostics"]
    assert len(compile_calls) == 1
    assert perf["resolved_torch_compile_step_body"] == "on"
    assert perf["torch_compile_step_requested"] is True
    assert perf["torch_compile_step_used"] is True
    assert perf["torch_compile_step_reason"] == "compiled with torch.compile"


def test_thermal_torch_compile_auto_stays_off_on_cpu() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("torch is not available")

    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=2,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            diagnostics_mode="fast",
            history_mode="compact",
            history_stride=1,
            progress_every=0,
            backend="torch",
            torch_device="cpu",
            torch_compile_step_body="auto",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    perf = result["performance_diagnostics"]
    assert perf["resolved_torch_compile_step_body"] == "off"
    assert perf["torch_compile_step_requested"] is False
    assert perf["torch_compile_step_used"] is False
    assert (
        perf["torch_compile_step_reason"] == "auto disabled outside CUDA torch device"
    )


def test_thermal_torch_compile_auto_stays_off_when_cuda_triton_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        thermal_solver_module,
        "find_spec",
        lambda name: None if name == "triton" else find_spec(name),
    )
    cfg = GmshTetraThermalConfig(
        backend="torch",
        torch_device="cuda:0",
        execution_profile="production",
        diagnostics_mode="fast",
        torch_compile_step_body="auto",
    )

    resolved = thermal_solver_module._resolve_thermal_torch_compile_mode(
        cfg,
        actual_backend="torch",
        device_type="cuda",
    )
    reason = thermal_solver_module._thermal_torch_compile_disabled_reason(
        cfg,
        resolved_compile_mode=resolved,
        actual_backend="torch",
        device_type="cuda",
    )

    assert resolved == "off"
    assert reason == "torch.compile CUDA inductor unavailable: triton is not importable"


def test_thermal_torch_compile_on_skips_cuda_compile_when_triton_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("torch is not available")

    compile_calls: list[object] = []

    def _fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(
        thermal_solver_module,
        "find_spec",
        lambda name: None if name == "triton" else find_spec(name),
    )
    monkeypatch.setattr(torch, "compile", _fake_compile, raising=False)

    step_fn, metadata = thermal_solver_module._maybe_compile_thermal_torch_step_body(
        torch,
        lambda value: value,
        compile_mode="on",
        device_type="cuda",
    )

    assert step_fn("value") == "value"
    assert compile_calls == []
    assert metadata["torch_compile_step_requested"] is True
    assert metadata["torch_compile_step_used"] is False
    assert (
        metadata["torch_compile_step_reason"]
        == "torch.compile CUDA inductor unavailable: triton is not importable"
    )


def test_thermal_compact_history_omits_full_diagnostics_fields() -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            history_stride=1,
            full_step_diagnostics=False,
            progress_every=0,
            backend="numpy",
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    entry = result["history"][0]
    assert "raw_balance_error" in entry
    assert "clipped_cell_count" in entry
    assert "advective_mass_in" not in entry
    assert "diffusion_integral_change_unlimited" not in entry
    assert "limiter_active_fraction" not in entry


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_full_history_keeps_extended_step_diagnostics(backend: str) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            history_stride=1,
            full_step_diagnostics=True,
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    entry = result["history"][0]
    assert "advective_mass_in" in entry
    assert "diffusion_integral_change_unlimited" in entry
    assert "limiter_active_fraction" in entry
    assert "cumulative_clamp" in entry


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_diagnostics_stride_limits_extended_step_metrics(backend: str) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=5,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            history_stride=1,
            diagnostics_stride=0,
            full_step_diagnostics=True,
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    history_by_step = {int(entry["step"]): entry for entry in result["history"]}
    assert "advective_mass_in" in history_by_step[1]
    assert "advective_mass_in" not in history_by_step[2]
    assert "advective_mass_in" not in history_by_step[3]
    assert "advective_mass_in" not in history_by_step[4]
    assert "advective_mass_in" in history_by_step[5]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_thermal_debug_artifacts_flag_controls_final_step_payload(
    backend: str,
) -> None:
    mesh = _build_synthetic_mesh()
    face_normal_velocity, flux_diag = _prescribed_velocity_inputs(mesh)

    without_debug = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            progress_every=0,
            debug_artifacts=False,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )
    with_debug = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=3,
            dt=1e-3,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            clipping_enabled=True,
            min_temperature=280.0,
            max_temperature=420.0,
            progress_every=0,
            debug_artifacts=True,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diag,
    )

    assert without_debug["final_step_debug"] == {}
    assert "raw_state_before_limiter" in with_debug["final_step_debug"]


def _stationary_face_velocity(mesh) -> np.ndarray:
    return np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "wall_boundary_mode": "fixed_heat_flux",
                "heated_wall_physical_groups": ("missing",),
                "wall_heat_flux": 1.0,
            },
            "Unknown heated wall physical group",
        ),
        (
            {
                "wall_boundary_mode": "fixed_heat_flux",
                "heated_wall_physical_groups": ("left_inlet",),
                "wall_heat_flux": 1.0,
            },
            "not a subset of mesh.wall_faces",
        ),
        (
            {
                "wall_boundary_mode": "fixed_heat_flux",
                "heated_wall_physical_groups": (),
                "wall_heat_flux": 1.0,
            },
            "heated_wall_physical_groups is required",
        ),
        (
            {
                "wall_boundary_mode": "fixed_temperature",
                "heated_wall_physical_groups": ("walls",),
            },
            "wall_temperature is required",
        ),
        (
            {
                "wall_boundary_mode": "fixed_heat_flux",
                "heated_wall_physical_groups": ("walls",),
            },
            "wall_heat_flux is required",
        ),
        (
            {"wall_temperature": 320.0, "wall_heat_flux": 10.0},
            "cannot both be set",
        ),
    ],
)
def test_wall_boundary_validation_errors(kwargs: dict, message: str) -> None:
    mesh = _build_synthetic_mesh()
    with pytest.raises(ValueError, match=message):
        run_tetra_thermal_debug(
            mesh,
            GmshTetraThermalConfig(steps=1, progress_every=0, **kwargs),
            face_normal_velocity=_stationary_face_velocity(mesh),
        )


def test_wall_boundary_rejects_empty_physical_group() -> None:
    mesh = _build_synthetic_mesh()
    mesh.physical_groups["empty_wall"] = (99, 2)
    with pytest.raises(ValueError, match="contains no boundary faces"):
        run_tetra_thermal_debug(
            mesh,
            GmshTetraThermalConfig(
                steps=1,
                progress_every=0,
                wall_boundary_mode="fixed_heat_flux",
                heated_wall_physical_groups=("empty_wall",),
                wall_heat_flux=1.0,
            ),
            face_normal_velocity=_stationary_face_velocity(mesh),
        )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_adiabatic_wall_bc_is_backward_compatible(backend: str) -> None:
    mesh = _build_synthetic_mesh()
    velocity, flux_diag = _prescribed_velocity_inputs(mesh)
    common = dict(
        steps=4,
        dt=1e-3,
        dt_mode="manual",
        progress_every=0,
        **_backend_config_kwargs(backend),
    )
    legacy = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common),
        face_normal_velocity=velocity,
        flux_diagnostics=flux_diag,
    )
    explicit = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            **common,
            wall_boundary_mode="adiabatic",
            heated_wall_physical_groups=(),
        ),
        face_normal_velocity=velocity,
        flux_diagnostics=flux_diag,
    )
    np.testing.assert_array_equal(legacy["temperature"], explicit["temperature"])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_fixed_heat_flux_closed_volume_energy_identity(backend: str) -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    dt = 0.025
    steps = 8
    rho = 2.0
    cp = 3.0
    heat_flux = -123.5
    cfg = GmshTetraThermalConfig(
        steps=steps,
        dt=dt,
        dt_mode="manual",
        thermal_diffusivity=0.0,
        rho=rho,
        cp=cp,
        initial_temperature=310.0,
        heat_source=0.0,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
        wall_boundary_mode="fixed_heat_flux",
        heated_wall_physical_groups=("walls",),
        wall_heat_flux=heat_flux,
        **_backend_config_kwargs(backend),
    )
    result = run_tetra_thermal_debug(
        mesh, cfg, face_normal_velocity=_stationary_face_velocity(mesh)
    )
    area = float(np.sum(mesh.face_areas[mesh.wall_faces]))
    volume = float(mesh.cell_volumes[0])
    lhs = rho * cp * volume * (float(result["temperature"][0]) - 310.0)
    rhs = heat_flux * area * dt * steps
    assert abs(lhs - rhs) / max(abs(rhs), 1e-30) <= 1e-10
    wall = result["wall_heat_transfer"]
    assert wall["cumulative_requested_wall_energy_j"] == pytest.approx(rhs)
    assert wall["cumulative_applied_wall_energy_j"] == pytest.approx(rhs)
    assert wall["limiter_clipping_correction_j"] == pytest.approx(0.0, abs=1e-8)
    assert result["global_energy_balance"]["wall_energy_j"] == pytest.approx(rhs)
    assert result["global_energy_balance"]["balance_error_j"] == pytest.approx(
        0.0, abs=1e-10
    )


@pytest.mark.parametrize("limit_non_advective_update", [True, False])
def test_fixed_heat_flux_reports_limiter_and_clipping_energy_correction(
    limit_non_advective_update: bool,
) -> None:
    mesh = _build_one_cell_inlet_outlet_mesh()
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=1,
            dt=1.0,
            dt_mode="manual",
            thermal_diffusivity=0.0,
            rho=1.0,
            cp=1.0,
            initial_temperature=300.0,
            min_temperature=299.0,
            max_temperature=301.0,
            progress_every=0,
            limit_non_advective_update=limit_non_advective_update,
            wall_boundary_mode="fixed_heat_flux",
            heated_wall_physical_groups=("walls",),
            wall_heat_flux=100.0,
            backend="numpy",
        ),
        face_normal_velocity=_stationary_face_velocity(mesh),
    )
    wall = result["wall_heat_transfer"]
    assert wall["cumulative_requested_energy_j"] == pytest.approx(100.0)
    assert wall["cumulative_applied_energy_j"] < 100.0
    assert wall["limiter_clipping_correction_energy_j"] == pytest.approx(
        wall["cumulative_requested_energy_j"] - wall["cumulative_applied_energy_j"]
    )
    balance = result["global_energy_balance"]
    if limit_non_advective_update:
        assert wall["limiter_correction_energy_j"] > 0.0
        assert wall["clipping_correction_energy_j"] == pytest.approx(0.0)
        assert balance["clipping_energy_j"] == pytest.approx(0.0)
    else:
        assert wall["limiter_correction_energy_j"] == pytest.approx(0.0)
        assert wall["clipping_correction_energy_j"] > 0.0
        assert balance["wall_boundary_energy_j"] > balance["wall_energy_j"]
        assert balance["clipping_energy_j"] < 0.0
    assert balance["balance_error_j"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_global_energy_balance_includes_inlet_dirichlet_diffusion(
    backend: str,
) -> None:
    mesh = _build_synthetic_mesh()
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=5,
            dt=1e-4,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            initial_temperature=298.15,
            left_inlet_temperature=330.0,
            right_inlet_temperature=330.0,
            clipping_enabled=False,
            min_temperature=None,
            max_temperature=None,
            progress_every=0,
            **_backend_config_kwargs(backend),
        ),
        face_normal_velocity=_stationary_face_velocity(mesh),
    )
    balance = result["global_energy_balance"]
    assert balance["non_wall_diffusion_energy_j"] > 0.0
    assert balance["wall_boundary_energy_j"] == pytest.approx(0.0)
    assert balance["delta_energy_j"] == pytest.approx(
        balance["accounted_energy_j"], rel=1e-6, abs=1e-9
    )
    assert balance["balance_error_j"] == pytest.approx(0.0, abs=1e-7)


def test_global_energy_balance_combines_inlet_diffusion_and_fixed_wall_flux() -> None:
    mesh = _build_synthetic_mesh()
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=5,
            dt=1e-4,
            dt_mode="manual",
            thermal_diffusivity=1.4e-7,
            initial_temperature=298.15,
            left_inlet_temperature=330.0,
            right_inlet_temperature=330.0,
            clipping_enabled=False,
            min_temperature=None,
            max_temperature=None,
            progress_every=0,
            wall_boundary_mode="fixed_heat_flux",
            heated_wall_physical_groups=("walls",),
            wall_heat_flux=250.0,
            backend="numpy",
        ),
        face_normal_velocity=_stationary_face_velocity(mesh),
    )
    balance = result["global_energy_balance"]
    assert balance["non_wall_diffusion_energy_j"] > 0.0
    assert balance["wall_boundary_energy_j"] > 0.0
    assert balance["balance_error_j"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("diagnostics_mode", ["debug", "fast"])
def test_fixed_heat_flux_numpy_torch_parity(diagnostics_mode: str) -> None:
    pytest.importorskip("torch")
    mesh = _build_synthetic_mesh()
    common = dict(
        steps=5,
        dt=2e-4,
        dt_mode="manual",
        thermal_diffusivity=1.4e-7,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
        wall_boundary_mode="fixed_heat_flux",
        heated_wall_physical_groups=("walls",),
        wall_heat_flux=250.0,
        diagnostics_mode=diagnostics_mode,
    )
    velocity = _stationary_face_velocity(mesh)
    numpy_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="numpy"),
        face_normal_velocity=velocity,
    )
    torch_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="torch", torch_device="cpu"),
        face_normal_velocity=velocity,
    )
    np.testing.assert_allclose(
        numpy_result["temperature"],
        torch_result["temperature"],
        rtol=1e-10,
        atol=1e-10,
    )
    assert numpy_result["wall_heat_transfer"][
        "cumulative_applied_wall_energy_j"
    ] == pytest.approx(
        torch_result["wall_heat_transfer"]["cumulative_applied_wall_energy_j"],
        rel=1e-10,
        abs=1e-10,
    )


def test_fixed_temperature_slab_converges_to_linear_profile_and_flux() -> None:
    mesh = _build_slab_mesh()
    cold = 300.0
    hot = 400.0
    rho = 2.0
    cp = 3.0
    alpha = 1.0
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(
            steps=1500,
            dt=1e-3,
            dt_mode="auto",
            thermal_diffusivity=alpha,
            rho=rho,
            cp=cp,
            initial_temperature=cold,
            left_inlet_temperature=cold,
            right_inlet_temperature=cold,
            clipping_enabled=False,
            min_temperature=None,
            max_temperature=None,
            progress_every=0,
            laplacian_method="lsq_flux",
            wall_boundary_mode="fixed_temperature",
            heated_wall_physical_groups=("heated_wall",),
            wall_temperature=hot,
            backend="numpy",
        ),
        face_normal_velocity=_stationary_face_velocity(mesh),
    )
    expected = cold + (hot - cold) * mesh.cell_centers[:, 0]
    relative_l2 = float(
        np.linalg.norm(np.asarray(result["temperature"]) - expected)
        / np.linalg.norm(expected - cold)
    )
    expected_power = alpha * rho * cp * (hot - cold)
    actual_power = float(
        result["wall_heat_transfer"]["instantaneous_requested_power_w"]
    )
    assert result["wall_heat_transfer"]["face_count"] == 2
    assert result["boundary_setup"]["diffusion_bc"]["no_flux_faces"] == (
        int(mesh.wall_faces.size) - 2
    )
    assert relative_l2 <= 0.02
    assert abs(actual_power - expected_power) / expected_power <= 0.02


@pytest.mark.parametrize("diagnostics_mode", ["debug", "fast"])
def test_fixed_temperature_numpy_torch_energy_parity(
    diagnostics_mode: str,
) -> None:
    pytest.importorskip("torch")
    mesh = _build_slab_mesh()
    velocity = _stationary_face_velocity(mesh)
    common = dict(
        steps=20,
        dt=1e-3,
        dt_mode="auto",
        thermal_diffusivity=1.0,
        rho=2.0,
        cp=3.0,
        initial_temperature=300.0,
        left_inlet_temperature=300.0,
        right_inlet_temperature=300.0,
        clipping_enabled=False,
        min_temperature=None,
        max_temperature=None,
        progress_every=0,
        laplacian_method="lsq_flux",
        wall_boundary_mode="fixed_temperature",
        heated_wall_physical_groups=("heated_wall",),
        wall_temperature=400.0,
        diagnostics_mode=diagnostics_mode,
    )
    numpy_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="numpy"),
        face_normal_velocity=velocity,
    )
    torch_result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(**common, backend="torch", torch_device="cpu"),
        face_normal_velocity=velocity,
    )
    np.testing.assert_allclose(
        numpy_result["temperature"],
        torch_result["temperature"],
        rtol=1e-10,
        atol=1e-10,
    )
    for key in (
        "cumulative_energy_after_limiter_before_clipping_j",
        "cumulative_applied_wall_energy_j",
    ):
        assert numpy_result["wall_heat_transfer"][key] == pytest.approx(
            torch_result["wall_heat_transfer"][key], rel=1e-10, abs=1e-10
        )
    for key in (
        "wall_boundary_energy_j",
        "non_wall_diffusion_energy_j",
        "accounted_energy_j",
        "balance_error_j",
    ):
        assert numpy_result["global_energy_balance"][key] == pytest.approx(
            torch_result["global_energy_balance"][key], rel=1e-10, abs=1e-10
        )


def test_outlet_uniformity_is_none_without_positive_flux() -> None:
    mesh = _build_synthetic_mesh()
    result = run_tetra_thermal_debug(
        mesh,
        GmshTetraThermalConfig(steps=1, progress_every=0),
        face_normal_velocity=_stationary_face_velocity(mesh),
    )
    observables = result["thermal_observables"]
    assert observables["outlet_temperature_flow_weighted_mean"] is None
    assert observables["outlet_temperature_flow_weighted_std"] is None
    assert observables["outlet_temperature_min"] is not None
    assert observables["outlet_temperature_max"] is not None
    assert observables["outlet_temperature_range"] is not None
