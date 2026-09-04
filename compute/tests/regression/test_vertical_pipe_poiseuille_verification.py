from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tools.run_vertical_pipe_poiseuille_verification import (
    _flow_config,
    _parse_args,
    _tail_stationarity,
    convergence_summary,
    evaluate_acceptance,
    poiseuille_profile,
    poiseuille_reference,
    pressure_interior_metrics,
    run_case,
    weighted_linear_fit,
    weighted_relative_l2,
)


def test_harness_defaults_select_automatic_no_slip_profile() -> None:
    args = _parse_args([])

    assert args.outlet_contracts == ("auto",)
    assert args.projection_cell_velocity_update_modes == ("auto",)
    assert args.pressure_nonorthogonal_correction_modes == ("auto",)
    assert args.viscous_nonorthogonal_correction_modes == ("auto",)
    assert args.pressure_nonorthogonal_correction_sweeps == 4
    assert args.pressure_nonorthogonal_correction_relaxation == pytest.approx(1.0)


def test_flow_config_forwards_coherent_nonorthogonal_variant() -> None:
    config, outlet_fields = _flow_config(
        outlet_contract="preserve",
        projection_cell_velocity_update_mode="momentum_pressure_corrected",
        pressure_nonorthogonal_correction_mode="deferred_lsq",
        viscous_nonorthogonal_correction_mode="deferred_lsq",
        pressure_nonorthogonal_correction_sweeps=3,
        pressure_nonorthogonal_correction_relaxation=0.4,
        inlet_speed=5e-4,
        density=1000.0,
        kinematic_viscosity=1e-6,
        flow_dt=5e-4,
        max_pressure_iterations=1000,
        pressure_relative_tolerance=1e-5,
    )

    assert set(outlet_fields) == {
        "viscous_predictor_outlet_contract_mode",
        "pressure_projection_outlet_contract_mode",
    }
    assert config.projection_cell_velocity_update_mode == "momentum_pressure_corrected"
    assert config.pressure_nonorthogonal_correction_mode == "deferred_lsq"
    assert config.viscous_nonorthogonal_correction_mode == "deferred_lsq"
    assert config.pressure_nonorthogonal_correction_sweeps == 3
    assert config.pressure_nonorthogonal_correction_relaxation == pytest.approx(0.4)


def test_convergence_summary_keeps_numerical_variants_separate() -> None:
    rows = []
    for velocity_mode, nonorthogonal_mode in (
        ("legacy_reconstruct", "none"),
        ("momentum_pressure_corrected", "deferred_lsq"),
    ):
        rows.append(
            {
                "status": "completed",
                "case": {
                    "outlet_contract": "preserve",
                    "projection_cell_velocity_update_mode": velocity_mode,
                    "pressure_nonorthogonal_correction_mode": nonorthogonal_mode,
                    "viscous_nonorthogonal_correction_mode": "none",
                },
                "mesh": {
                    "name": "pipe.msh",
                    "cells": 100,
                    "characteristic_cell_size": 1e-4,
                },
                "metrics": {
                    "pressure": {"fit_pressure_gradient_relative_error": 0.1},
                    "velocity_profile": {"relative_l2_error": 0.2},
                },
                "metrics_by_radius": {
                    "nominal": {
                        "pressure": {"fit_pressure_gradient_relative_error": 0.1},
                        "velocity_profile": {"relative_l2_error": 0.2},
                    },
                    "effective_area": {
                        "pressure": {"fit_pressure_gradient_relative_error": 0.05},
                        "velocity_profile": {"relative_l2_error": 0.08},
                    },
                },
                "stationarity": {"converged": True},
            }
        )

    summaries = convergence_summary(rows)

    assert len(summaries) == 2
    assert {
        (
            row["projection_cell_velocity_update_mode"],
            row["pressure_nonorthogonal_correction_mode"],
            row["viscous_nonorthogonal_correction_mode"],
        )
        for row in summaries
    } == {
        ("legacy_reconstruct", "none", "none"),
        ("momentum_pressure_corrected", "deferred_lsq", "none"),
    }
    assert all(
        summary["points"][0]["normalizations"]["effective_area"][
            "profile_relative_l2_error"
        ]
        == pytest.approx(0.08)
        for summary in summaries
    )


def test_acceptance_requires_physics_conservation_and_outer_convergence() -> None:
    result = {
        "metrics": {
            "pressure": {"fit_pressure_gradient_relative_error": 0.08},
            "velocity_profile": {"relative_l2_error": 0.20},
            "wall_shear": {"available": True, "relative_error": 0.10},
        },
        "mass_balance": {
            "outlet_inlet_ratio": 1.0001,
            "wall_flux_max_abs": 0.0,
        },
        "stationarity": {"converged": True},
        "case": {
            "effective_numerical_profile": {
                "viscous_predictor_outlet_contract_mode": "preserve",
                "pressure_projection_outlet_contract_mode": "preserve",
                "projection_cell_velocity_update_mode": "momentum_pressure_corrected",
                "pressure_nonorthogonal_correction_mode": "deferred_lsq",
                "viscous_nonorthogonal_correction_mode": "deferred_lsq",
            }
        },
        "solver_diagnostics": {
            "state_arrays_all_finite_all_steps": True,
            "projection_reduced_divergence_l2_every_step": True,
            "projection_final_divergence_l2": 1e-8,
            "projection_divergence_reduction_ratio_l2": 1e-5,
            "outlet_flux_rescale_used_any_step": False,
            "nonphysical_flux_fix_used_any_step": False,
            "face_flux_update_identity_max_abs_all_steps": 0.0,
            "returned_face_flux_identity_max_abs_all_steps": 0.0,
            "pressure_nonorthogonal_outer_defect_relative_l2": 1e-4,
            "pressure_stopping_reason": "converged_relative_l2",
        },
    }

    accepted = evaluate_acceptance(result)
    assert accepted["passed"] is True
    assert all(accepted["checks"].values())
    assert accepted["reference_normalization"] == "nominal"
    assert accepted["thresholds"]["velocity_profile_relative_l2_max"] == pytest.approx(
        0.25
    )
    assert (
        "empirical regression bounds"
        in accepted["threshold_basis"]["percentage_error_gates"]
    )

    result["metrics"]["wall_shear"]["relative_error"] = 2.0
    wall_shear_diagnostic = evaluate_acceptance(result)
    assert wall_shear_diagnostic["passed"] is True
    assert wall_shear_diagnostic["diagnostics"]["wall_shear_status"] == (
        "non_gating_first_order_owner_cell_estimate"
    )

    result["solver_diagnostics"]["pressure_nonorthogonal_outer_defect_relative_l2"] = (
        0.2
    )
    rejected = evaluate_acceptance(result)
    assert rejected["passed"] is False
    assert rejected["checks"]["pressure_outer_fixed_point"] is False


@pytest.mark.parametrize(
    ("field", "bad_value", "failed_check"),
    [
        ("state_arrays_all_finite_all_steps", False, "finite_state"),
        (
            "projection_reduced_divergence_l2_every_step",
            False,
            "projection_reduces_divergence",
        ),
        ("outlet_flux_rescale_used_any_step", True, "no_outlet_flux_rescale"),
        ("nonphysical_flux_fix_used_any_step", True, "no_nonphysical_flux_fix"),
        (
            "face_flux_update_identity_max_abs_all_steps",
            1e-10,
            "face_flux_update_identity",
        ),
        (
            "returned_face_flux_identity_max_abs_all_steps",
            1e-10,
            "no_post_projection_flux_add_back",
        ),
    ],
)
def test_acceptance_rejects_hidden_runtime_flux_fixes(
    field: str,
    bad_value: bool | float,
    failed_check: str,
) -> None:
    result = {
        "metrics": {
            "pressure": {"fit_pressure_gradient_relative_error": 0.08},
            "velocity_profile": {"relative_l2_error": 0.20},
            "wall_shear": {"available": True, "relative_error": 0.10},
        },
        "mass_balance": {
            "outlet_inlet_ratio": 1.0001,
            "wall_flux_max_abs": 0.0,
        },
        "stationarity": {"converged": True},
        "case": {
            "effective_numerical_profile": {
                "viscous_predictor_outlet_contract_mode": "preserve",
                "pressure_projection_outlet_contract_mode": "preserve",
                "projection_cell_velocity_update_mode": "momentum_pressure_corrected",
                "pressure_nonorthogonal_correction_mode": "deferred_lsq",
                "viscous_nonorthogonal_correction_mode": "deferred_lsq",
            }
        },
        "solver_diagnostics": {
            "state_arrays_all_finite_all_steps": True,
            "projection_reduced_divergence_l2_every_step": True,
            "projection_final_divergence_l2": 1e-8,
            "projection_divergence_reduction_ratio_l2": 1e-5,
            "outlet_flux_rescale_used_any_step": False,
            "nonphysical_flux_fix_used_any_step": False,
            "face_flux_update_identity_max_abs_all_steps": 0.0,
            "returned_face_flux_identity_max_abs_all_steps": 0.0,
            "pressure_nonorthogonal_outer_defect_relative_l2": 1e-4,
            "pressure_stopping_reason": "converged_relative_l2",
        },
    }
    result["solver_diagnostics"][field] = bad_value

    rejected = evaluate_acceptance(result)

    assert rejected["passed"] is False
    assert rejected["checks"][failed_check] is False


def test_poiseuille_reference_uses_mean_velocity_circular_pipe_laws() -> None:
    reference = poiseuille_reference(
        radius=1e-3,
        length=18e-3,
        mean_speed=5e-4,
        density=1000.0,
        kinematic_viscosity=1e-6,
    )

    assert reference["dynamic_viscosity"] == pytest.approx(1e-3)
    assert reference["reynolds_number"] == pytest.approx(1.0)
    assert reference["pressure_gradient_magnitude"] == pytest.approx(4.0)
    assert reference["full_length_pressure_drop"] == pytest.approx(0.072)
    assert reference["wall_shear_stress_magnitude"] == pytest.approx(0.002)
    assert reference["centerline_speed"] == pytest.approx(1e-3)
    assert reference["volumetric_flow_rate"] == pytest.approx(math.pi * 5e-10)


def test_poiseuille_profile_and_weighted_relative_l2() -> None:
    radial = np.asarray([0.0, 0.5e-3, 1e-3])
    expected = np.asarray([1e-3, 7.5e-4, 0.0])

    actual = poiseuille_profile(radial, radius=1e-3, mean_speed=5e-4)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert weighted_relative_l2(actual, expected, np.ones(3)) == pytest.approx(0.0)
    assert weighted_relative_l2(1.1 * actual, expected, np.ones(3)) == pytest.approx(
        0.1
    )


def test_weighted_linear_fit_recovers_exact_pressure_slope() -> None:
    axial = np.linspace(0.0, 0.018, 31)
    pressure = 2.5 - 4.0 * axial
    weights = np.linspace(1.0, 3.0, axial.size)

    fit = weighted_linear_fit(axial, pressure, weights)

    assert fit["slope"] == pytest.approx(-4.0)
    assert fit["intercept"] == pytest.approx(2.5)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_pressure_metrics_exclude_ends_and_recover_interior_gradient() -> None:
    length = 0.018
    axial = np.linspace(0.0, length, 181)
    pressure = 1.0 - 4.0 * axial
    # Deliberately corrupt the entrance and outlet regions. The fit interval and
    # both pressure planes lie in the untouched interior.
    pressure[axial < 0.1 * length] += 50.0
    pressure[axial > 0.9 * length] -= 50.0

    metrics = pressure_interior_metrics(
        axial,
        pressure,
        np.ones_like(axial),
        length=length,
        plane_fractions=(0.35, 0.65),
        plane_half_width_fraction=0.01,
        fit_fraction=(0.25, 0.75),
    )

    assert metrics["fit_pressure_gradient_magnitude"] == pytest.approx(4.0)
    assert metrics["plane_pressure_gradient_magnitude"] == pytest.approx(4.0)
    assert metrics["plane_pressure_drop"] == pytest.approx(4.0 * 0.3 * length)
    assert metrics["fit_r_squared"] == pytest.approx(1.0)


def test_tail_stationarity_requires_a_full_stable_window() -> None:
    short = [
        {
            "step": 20,
            "pressure_gradient_magnitude": 4.0,
            "profile_relative_l2_error": 0.1,
        }
    ]
    assert not _tail_stationarity(short, window=3, relative_tolerance=1e-3)["converged"]

    stable = [
        {
            "step": step,
            "pressure_gradient_magnitude": 4.0 + delta,
            "profile_relative_l2_error": 0.1 + 0.1 * delta,
        }
        for step, delta in ((20, 0.0), (40, 1e-5), (60, -1e-5))
    ]
    result = _tail_stationarity(stable, window=3, relative_tolerance=1e-3)
    assert result["converged"]
    assert result["reason"] == "tail_stable"


def test_production_no_slip_auto_profile_meets_poiseuille_contract() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = run_case(
        project_root / "data" / "meshes" / "gmsh" / "vertical_pipe_500.msh",
        outlet_contract="auto",
        projection_cell_velocity_update_mode="auto",
        pressure_nonorthogonal_correction_mode="auto",
        viscous_nonorthogonal_correction_mode="auto",
        pressure_nonorthogonal_correction_sweeps=2,
        pressure_nonorthogonal_correction_relaxation=1.0,
        steps=700,
        flow_dt=5e-4,
        inlet_speed=5e-4,
        density=1000.0,
        kinematic_viscosity=1e-6,
        radius_override=None,
        sample_every=20,
        steady_window=5,
        steady_relative_tolerance=2e-3,
        minimum_diffusive_time=0.25,
        stop_when_steady=True,
        min_steps=500,
        max_pressure_iterations=1000,
        pressure_relative_tolerance=1e-5,
        plane_fractions=(0.35, 0.65),
        plane_half_width_fraction=0.035,
        fit_fraction=(0.25, 0.75),
        profile_fraction=0.5,
    )

    assert result["acceptance"]["passed"] is True
    assert result["geometry"]["nominal_radius"] == pytest.approx(1e-3)
    assert result["geometry"]["effective_area_radius"] == pytest.approx(
        9.488499966575888e-4
    )
    assert set(result["analytic_references"]) == {"nominal", "effective_area"}
    assert set(result["metrics_by_radius"]) == {"nominal", "effective_area"}
    assert result["metrics"] is result["metrics_by_radius"]["nominal"]
    assert result["radius_normalization_comparison"][
        "effective_area_profile_vs_nominal_reference_relative_l2"
    ] == pytest.approx(0.0808, abs=5e-4)
    assert (
        result["metrics_by_radius"]["effective_area"]["velocity_profile"][
            "relative_l2_error"
        ]
        < result["metrics_by_radius"]["nominal"]["velocity_profile"][
            "relative_l2_error"
        ]
    )
    for normalization in ("nominal", "effective_area"):
        shear = result["metrics_by_radius"][normalization]["wall_shear"]
        assert shear["analytic_profile_estimator_baseline"] > 0.0
        assert math.isfinite(shear["analytic_profile_estimator_relative_bias"])
        assert math.isfinite(shear["numerical_relative_error_vs_estimator_baseline"])
    assert result["case"]["effective_numerical_profile"] == {
        "viscous_predictor_outlet_contract_mode": "preserve",
        "pressure_projection_outlet_contract_mode": "preserve",
        "projection_cell_velocity_update_mode": "momentum_pressure_corrected",
        "pressure_nonorthogonal_correction_mode": "deferred_lsq",
        "viscous_nonorthogonal_correction_mode": "deferred_lsq",
    }
    assert result["solver_diagnostics"]["state_arrays_all_finite_all_steps"] is True
    assert (
        result["solver_diagnostics"]["projection_reduced_divergence_l2_every_step"]
        is True
    )
    assert result["solver_diagnostics"]["outlet_flux_rescale_used_any_step"] is False
    assert result["solver_diagnostics"]["nonphysical_flux_fix_used_any_step"] is False
    assert result["solver_diagnostics"]["pressure_solve_count_total"] == (
        result["case"]["completed_steps"] * 2
    )
    assert result["solver_diagnostics"]["pressure_iterations_total"] > 0
    assert result["solver_diagnostics"]["pressure_solve_wall_seconds_total"] >= 0.0
    assert result["solver_diagnostics"]["pressure_solve_wall_seconds_mean"] >= 0.0
    assert (
        result["solver_diagnostics"]["pressure_solve_wall_seconds_per_iteration"] >= 0.0
    )
    assert (
        result["solver_diagnostics"]["face_flux_update_identity_max_abs_all_steps"]
        <= 1e-14
    )
    assert (
        result["solver_diagnostics"]["returned_face_flux_identity_max_abs_all_steps"]
        <= 1e-14
    )
