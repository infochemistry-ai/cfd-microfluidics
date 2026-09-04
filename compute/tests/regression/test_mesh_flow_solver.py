from dataclasses import replace

import numpy as np
import pytest

from microfluidics.mesh.mesh_builder import build_mesh_scaffold
from microfluidics.mesh.mesh_config import MeshRefinementConfig, default_mesh_run_config
from microfluidics.mesh.mesh_geometry import build_mesh_geometry_3d
from microfluidics.mesh.mesh_native_flow_solver import (
    MeshNativeFlowConfig,
    _solve_poisson_jacobi_numpy,
    _solve_poisson_pcg_numpy,
    advance_mesh_native_flow_step,
    build_mesh_native_flow_context,
    compute_face_fluxes,
    divergence_from_face_fluxes,
    initialize_mesh_native_flow_state,
    projection_operator_consistency_diagnostics,
    run_mesh_native_flow,
)
from microfluidics.mesh.mesh_operators import divergence_cell_vector


def _build_setup(*, include_cavities: bool, channel_like: bool):
    run_cfg = default_mesh_run_config(case_tag="TJ", include_cavities=include_cavities)
    if channel_like:
        run_cfg = replace(
            run_cfg,
            geometry=replace(
                run_cfg.geometry,
                inlet_width=1.0e-3,
                outlet_width=1.2e-3,
                inlet_length=2.0e-3,
                outlet_length=3.0e-3,
            ),
        )
    resolution = (28, 20, 8)
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
    # The native solver has no convective term, so tests keep it in a low-Re regime.
    solver_reynolds = 0.5
    inlet_speed = (
        solver_reynolds
        * run_cfg.kinematic_viscosity
        / run_cfg.geometry.hydraulic_diameter
    )
    cfg = MeshNativeFlowConfig(
        dt=1e-4,
        viscosity=run_cfg.kinematic_viscosity,
        inlet_speed=inlet_speed,
        steps=16,
        pressure_max_iters=500,
        pressure_tol=1e-8,
        jacobi_omega=0.55,
        projection_relaxation=1.0,
        progress_every=0,
        enable_safeguards=True,
        blowup_abs_limit=1e9,
        max_poisson_residual=1e12,
        max_divergence=1e9,
    )
    return mesh, geometry, cfg


def _small_native_flow_setup(include_cavities: bool = False):
    return _build_setup(include_cavities=include_cavities, channel_like=False)


def _small_channel_like_setup():
    return _build_setup(include_cavities=False, channel_like=True)


def test_native_flow_reduces_divergence_max_abs():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    result = run_mesh_native_flow(mesh=mesh, geometry=geometry, cfg=cfg)
    history = result["state"].diagnostics_history
    first = history[0]
    assert np.isfinite(result["initial_divergence_max_abs"])
    assert np.isfinite(result["final_divergence_max_abs"])
    assert not result["stopped_early"]
    assert first["projection_reduction_factor_max_abs"] < 1.0
    assert first["projection_reduction_factor_l2"] < 1.0
    assert np.isfinite(first["poisson_residual_max_abs"])


def test_projection_step_reduces_divergence_on_controlled_case():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    cfg = replace(
        cfg,
        steps=1,
        post_projection_enforce_inlets=False,
        post_projection_enforce_outlet=True,
    )
    context = build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=cfg)
    state = initialize_mesh_native_flow_state(context)
    state = advance_mesh_native_flow_step(state, context)
    diag = state.diagnostics_history[-1]
    assert np.isfinite(diag["poisson_residual_max_abs"])
    assert np.isfinite(diag["divergence_star_max_abs"])
    assert np.isfinite(diag["divergence_projected_max_abs"])
    assert np.isfinite(diag["poisson_rhs_mean_raw"])
    assert np.isfinite(diag["poisson_rhs_mean_corrected"])
    rhs_ref = max(1.0, abs(diag["poisson_rhs_mean_raw"]))
    assert abs(diag["poisson_rhs_mean_corrected"]) <= 1e-9 * rhs_ref
    assert diag["projection_reduction_factor_max_abs"] < 1.0
    assert diag["projection_reduction_factor_l2"] < 1.0
    assert diag["projection_reduction_factor_max_abs_post_bc"] < 1.0
    assert diag["projection_reduction_factor_l2_post_bc"] < 1.0


def test_native_flow_rejects_under_relaxed_projection_contract() -> None:
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    bad_cfg = replace(cfg, projection_relaxation=0.5)
    with pytest.raises(ValueError, match="projection_relaxation=1.0"):
        build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=bad_cfg)


def test_projection_operator_laplace_div_grad_consistency_is_finite():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    context = build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=cfg)
    diag = projection_operator_consistency_diagnostics(context)
    assert np.isfinite(diag["max_abs_diff"])
    assert np.isfinite(diag["mean_abs_diff"])
    assert np.isfinite(diag["interior"]["max_abs"])
    assert np.isfinite(diag["near_boundary"]["max_abs"])
    assert np.isfinite(diag["regional"]["outlet_near_zone"]["max_abs"])


def test_native_flow_enforces_boundary_conditions_and_pressure_gauge():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=True)
    cfg = replace(
        cfg, post_projection_enforce_inlets=True, post_projection_enforce_walls=True
    )
    context = build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=cfg)
    state = initialize_mesh_native_flow_state(context)
    state = advance_mesh_native_flow_step(state, context)

    pressure = state.fields.pressure.values
    velocity = state.fields.velocity.values
    bc = context.boundary_cells
    bc_face = context.boundary_face_cells
    assert np.isclose(pressure[context.gauge_index], 0.0, atol=1e-12)
    assert np.allclose(velocity[bc_face["left_inlet"], 0], cfg.inlet_speed, atol=1e-12)
    assert np.allclose(
        velocity[bc_face["right_inlet"], 0], -cfg.inlet_speed, atol=1e-12
    )
    if bc["wall"].size > 0:
        wall_speed = np.linalg.norm(velocity[bc["wall"]], axis=1)
        assert float(np.max(wall_speed)) < 1e-10


def test_projection_chain_diagnostics_present_and_finite():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    cfg = replace(cfg, steps=1)
    result = run_mesh_native_flow(mesh=mesh, geometry=geometry, cfg=cfg)
    diag = result["state"].diagnostics_history[-1]
    assert np.isfinite(diag["projection_chain_flux_reconstruction_mismatch_max_abs"])
    assert np.isfinite(
        diag["projection_chain_divergence_reconstruction_mismatch_max_abs"]
    )
    assert isinstance(diag["outlet_flux_stages"], dict)
    assert isinstance(diag["outlet_near_divergence_chain"], dict)


def test_native_flow_face_flux_divergence_is_finite():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    context = build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=cfg)
    state = initialize_mesh_native_flow_state(context)
    state = advance_mesh_native_flow_step(state, context)

    fluxes = compute_face_fluxes(state.fields.velocity.values, context)
    flux_div = divergence_from_face_fluxes(fluxes, context)
    centered_div = divergence_cell_vector(
        state.fields.velocity.values,
        context.topology,
        "neumann0",
    )

    assert np.all(np.isfinite(flux_div))
    assert np.all(np.isfinite(centered_div))
    assert float(np.max(np.abs(flux_div))) > 0.0


def test_native_flow_small_tj_stays_finite_for_debug_horizon():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    result = run_mesh_native_flow(mesh=mesh, geometry=geometry, cfg=cfg)
    state = result["state"]
    velocity, pressure = state.to_numpy()
    assert not result["stopped_early"]
    assert np.all(np.isfinite(velocity))
    assert np.all(np.isfinite(pressure))
    div = divergence_cell_vector(velocity, result["context"].topology, "neumann0")
    assert np.all(np.isfinite(div))
    assert float(np.max(np.abs(div))) < 1e6
    assert (
        result["final_divergence_max_abs"]
        <= 10.0 * result["initial_divergence_max_abs"]
    )


def test_native_flow_pressure_pcg_improves_on_legacy_jacobi() -> None:
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    context = build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=cfg)
    state = initialize_mesh_native_flow_state(context)
    velocity, pressure = state.to_numpy()
    velocity_star = velocity + context.cfg.dt * context.cfg.viscosity * 0.0
    velocity_star = velocity_star.copy()
    flux_star = compute_face_fluxes(velocity_star, context)
    div_star = divergence_from_face_fluxes(flux_star, context)
    rhs_raw = div_star / context.cfg.dt
    rhs = rhs_raw - float(np.mean(rhs_raw))

    jacobi_pressure, jacobi_iters, jacobi_res = _solve_poisson_jacobi_numpy(
        rhs,
        pressure,
        context,
    )
    pcg_pressure, pcg_iters, pcg_res = _solve_poisson_pcg_numpy(
        rhs,
        pressure,
        context,
    )

    assert np.isfinite(jacobi_res)
    assert np.isfinite(pcg_res)
    assert np.isclose(jacobi_pressure[context.gauge_index], 0.0, atol=1e-12)
    assert np.isclose(pcg_pressure[context.gauge_index], 0.0, atol=1e-12)
    assert pcg_res <= jacobi_res
    assert pcg_iters <= context.cfg.pressure_max_iters
    assert jacobi_iters <= context.cfg.pressure_max_iters


def test_native_flow_channel_like_case_does_not_blow_up():
    mesh, geometry, cfg = _small_channel_like_setup()
    result = run_mesh_native_flow(mesh=mesh, geometry=geometry, cfg=cfg)
    state = result["state"]
    velocity, pressure = state.to_numpy()
    assert not result["stopped_early"]
    assert np.all(np.isfinite(velocity))
    assert np.all(np.isfinite(pressure))
    assert result["final_divergence_max_abs"] < 1e6
    assert (
        result["final_divergence_max_abs"]
        <= 10.0 * result["initial_divergence_max_abs"]
    )


def test_native_flow_rejects_high_reynolds_without_convective_model():
    mesh, geometry, cfg = _small_native_flow_setup(include_cavities=False)
    high_re_cfg = replace(
        cfg,
        inlet_speed=200.0 * cfg.viscosity / geometry.config.hydraulic_diameter,
    )
    with pytest.raises(ValueError, match="omits the convective term"):
        build_mesh_native_flow_context(mesh=mesh, geometry=geometry, cfg=high_re_cfg)
