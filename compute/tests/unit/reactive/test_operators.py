"""Regression-oriented unit tests for reusable reactive spatial operators."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_backend import BOUNDARY_GROUP_CODE
from microfluidics.gmsh.tetra.gmsh_tetra_transport_solver import (
    _apply_bounded_limiter,
    _assemble_advection_fluxes,
)
from microfluidics.reactive import (
    ReactiveCaseValidationError,
    ReactiveWalltimeLimitError,
    TransportSubstepCapError,
    advance_spatial_fields,
    build_reactive_spatial_precompute,
    reactive_case_from_mapping,
    run_reactive_transport,
)
from microfluidics.reactive.operators import _boundary_adjusted_conservation_error
import microfluidics.reactive.solver as reactive_solver_module


def _mesh() -> ImportedTetraMesh:
    return ImportedTetraMesh(
        source_path=Path("synthetic.msh"),
        points=np.zeros((5, 3), dtype=np.float64),
        tetrahedra=np.zeros((2, 4), dtype=np.int64),
        boundary_triangles=np.zeros((3, 3), dtype=np.int64),
        boundary_face_tags=np.array([10, 20, 30], dtype=np.int32),
        physical_groups={
            "inlet-left": (2, 10),
            "outlet": (2, 20),
            "wall": (2, 30),
        },
        boundary_face_names={10: "Inlet-Left", 20: "Outlet", 30: "Wall"},
        cell_centers=np.array([[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        cell_volumes=np.ones(2, dtype=np.float64),
        face_vertices=np.zeros((4, 3), dtype=np.int64),
        face_centers=np.array(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.5, 0.0],
            ]
        ),
        face_areas=np.ones(4, dtype=np.float64),
        face_normals=np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        face_to_cells=np.array([[0, 1], [0, -1], [1, -1], [0, -1]]),
        cell_to_faces=np.array([[0, 1, 3, 3], [0, 2, 2, 2]]),
        boundary_tag_per_face=np.array([-1, 10, 20, 30], dtype=np.int32),
        interior_face_indices=np.array([0], dtype=np.int64),
        boundary_face_indices=np.array([1, 2, 3], dtype=np.int64),
        inlet_faces=np.array([1], dtype=np.int64),
        outlet_faces=np.array([2], dtype=np.int64),
        wall_faces=np.array([3], dtype=np.int64),
        boundary_unresolved_faces=np.zeros(0, dtype=np.int64),
    )


def _case(*, max_transport_substeps: int = 16):
    return reactive_case_from_mapping(
        {
            "schema_version": 1,
            "case_id": "operator",
            "mode": "off",
            "mechanism": {
                "version": 1,
                "metadata": {
                    "name": "operator",
                    "temperature_range": [250.0, 1000.0],
                },
                "species": [
                    {
                        "name": "A",
                        "phase": "liquid",
                        "molecular_weight": 10.0,
                        "composition": {"X": 1.0},
                    },
                    {
                        "name": "B",
                        "phase": "liquid",
                        "molecular_weight": 10.0,
                        "composition": {"X": 1.0},
                    },
                ],
                "reactions": [
                    {
                        "id": "R1",
                        "equation": "A -> B",
                        "kinetics": {"A": "1 1/s", "Ea": 0.0},
                    }
                ],
            },
            "initial_state": {
                "temperature_k": 300.0,
                "operating_pressure_pa": 101325.0,
                "concentrations_mol_per_m3": {},
            },
            "inlets": {
                "inlet left": {
                    "temperature_k": 300.0,
                    "concentrations_mol_per_m3": {"A": 100.0, "B": 200.0},
                }
            },
            "material": {
                "density_kg_per_m3": 1000.0,
                "heat_capacity_j_per_kg_k": 4000.0,
                "thermal_diffusivity_m2_s": 0.0,
                "species_diffusivity_m2_s": {"A": 0.0, "B": 0.0},
            },
            "time": {
                "num_steps": 1,
                "dt_mode": "manual",
                "dt_s": 1.0,
                "max_dt_s": 1.0,
                "cfl_target": 0.5,
                "diffusion_stability_factor": 0.5,
                "max_transport_substeps": max_transport_substeps,
            },
            "chemistry_integrator": {
                "relative_tolerance": 1.0e-6,
                "concentration_absolute_tolerance_mol_per_m3": 1.0e-12,
                "temperature_absolute_tolerance_k": 1.0e-8,
                "max_temperature_change_per_substep_k": 1.0,
                "min_substep_fraction": 1.0e-12,
                "max_substeps_per_half_step": 1000,
                "cell_batch_size": 100,
            },
            "output": {"history_stride": 1, "snapshot_steps": []},
        }
    )


def _legacy_upwind_step(
    values: np.ndarray, *, inlet_value: float, dt_s: float
) -> np.ndarray:
    mesh = _mesh()
    face_flux = np.array([0.1, -0.1, 0.1, 0.0])
    groups = np.array(
        [
            BOUNDARY_GROUP_CODE["unresolved"],
            BOUNDARY_GROUP_CODE["left_inlet"],
            BOUNDARY_GROUP_CODE["outlet"],
            BOUNDARY_GROUP_CODE["wall"],
        ],
        dtype=np.int32,
    )
    assembled = _assemble_advection_fluxes(
        mesh,
        values,
        face_flux / mesh.face_areas,
        boundary_face_groups=groups,
        left_inlet_value=inlet_value,
        right_inlet_value=inlet_value,
        dt=dt_s,
    )
    legacy = _apply_bounded_limiter(
        mesh,
        values,
        assembled,
        scheme="upwind",
        dt=dt_s,
        lower_bound=0.0,
        upper_bound=max(float(np.max(values)), inlet_value),
    )
    return np.asarray(legacy["limited_state_before_clip"], dtype=np.float64)


def test_matrix_transport_has_no_hidden_unit_upper_bound() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    result = advance_spatial_fields(
        precompute,
        case,
        np.zeros((2, 2), dtype=np.float64),
        np.full(2, 300.0),
        outer_dt_s=1.0,
    )

    assert np.max(result.concentrations_mol_per_m3[:, 0]) > 1.0
    assert np.max(result.concentrations_mol_per_m3[:, 1]) > 1.0
    np.testing.assert_allclose(result.concentrations_mol_per_m3[0], [10.0, 20.0])
    np.testing.assert_array_equal(result.temperature_k, np.full(2, 300.0))


def test_one_species_advection_matches_existing_scalar_operator() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    concentrations = np.array([[10.0, 0.0], [2.0, 0.0]])
    result = advance_spatial_fields(
        precompute,
        case,
        concentrations,
        np.full(2, 300.0),
        outer_dt_s=1.0,
    )

    np.testing.assert_allclose(
        result.concentrations_mol_per_m3[:, 0],
        _legacy_upwind_step(concentrations[:, 0], inlet_value=100.0, dt_s=1.0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_thermal_advection_matches_existing_scalar_operator() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    temperature = np.array([300.0, 310.0])
    result = advance_spatial_fields(
        precompute,
        case,
        np.zeros((2, 2)),
        temperature,
        outer_dt_s=1.0,
    )

    np.testing.assert_allclose(
        result.temperature_k,
        _legacy_upwind_step(temperature, inlet_value=300.0, dt_s=1.0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_uniform_fields_are_preserved_and_species_are_independent() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    concentrations = np.array([[100.0, 200.0], [100.0, 200.0]])
    result = advance_spatial_fields(
        precompute,
        case,
        concentrations,
        np.full(2, 300.0),
        outer_dt_s=1.0,
    )
    np.testing.assert_allclose(result.concentrations_mol_per_m3, concentrations)
    assert result.diagnostics.pairwise_conservation_error <= 1.0e-12


def test_pairwise_conservation_diagnostic_detects_an_inconsistent_ledger() -> None:
    error = _boundary_adjusted_conservation_error(
        np.array([0.75, -0.5]),
        advective_in=0.0,
        advective_out=0.0,
        diffusive_in=0.0,
        diffusive_out=0.0,
    )

    assert error == pytest.approx(0.25)


def test_zero_velocity_and_diffusion_do_not_change_state() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), case)
    concentrations = np.array([[2.0, 3.0], [4.0, 5.0]])
    result = advance_spatial_fields(
        precompute,
        case,
        concentrations,
        np.array([300.0, 301.0]),
        outer_dt_s=1.0,
    )
    np.testing.assert_array_equal(result.concentrations_mol_per_m3, concentrations)
    np.testing.assert_array_equal(result.temperature_k, [300.0, 301.0])


def test_thermal_transport_has_no_inlet_maximum_limiter() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.0, -0.1, 0.0, 0.0]), case
    )
    result = advance_spatial_fields(
        precompute,
        case,
        np.zeros((2, 2)),
        np.full(2, 300.0),
        outer_dt_s=1.0,
    )

    assert result.temperature_k[0] > 300.0
    assert result.diagnostics.thermal_roundoff_normalization_k_m3 == 0.0


def test_transport_substep_cap_blocks_without_shortening_time() -> None:
    case = _case(max_transport_substeps=1)
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    with pytest.raises(TransportSubstepCapError, match="blocked"):
        advance_spatial_fields(
            precompute,
            case,
            np.zeros((2, 2)),
            np.full(2, 300.0),
            outer_dt_s=20.0,
        )


def test_spatial_subcycling_honors_expired_walltime_deadline() -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), case)

    with pytest.raises(ReactiveWalltimeLimitError, match="walltime limit"):
        advance_spatial_fields(
            precompute,
            case,
            np.zeros((2, 2)),
            np.full(2, 300.0),
            outer_dt_s=1.0,
            walltime_deadline_monotonic=0.0,
        )


def test_interrupted_strang_step_does_not_commit_partial_state_or_balances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case()
    precompute = build_reactive_spatial_precompute(
        _mesh(), np.array([0.1, -0.1, 0.1, 0.0]), case
    )
    original_advance_reaction = reactive_solver_module.advance_reaction
    call_count = 0

    def interrupt_second_reaction(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ReactiveWalltimeLimitError("walltime limit")
        return original_advance_reaction(*args, **kwargs)

    monkeypatch.setattr(
        reactive_solver_module,
        "advance_reaction",
        interrupt_second_reaction,
    )
    result = run_reactive_transport(precompute, case)

    assert result.summary["executed_steps"] == 0
    assert result.summary["physical_time_s"] == 0.0
    assert result.summary["readiness"]["status"] == "blocked_walltime_limit"
    np.testing.assert_array_equal(
        result.concentrations_mol_per_m3,
        np.zeros((2, 2)),
    )
    assert result.summary["transport"]["substeps_total"] == 0
    assert result.summary["balances"]["species"]["A"]["advective_in_moles"] == 0.0


def test_unknown_named_inlet_is_rejected_without_geometric_fallback() -> None:
    case = _case()
    mesh = _mesh()
    mesh.boundary_face_names[10] = "different inlet"
    with pytest.raises(ReactiveCaseValidationError, match="missing"):
        build_reactive_spatial_precompute(mesh, np.zeros(4), case)


def test_strang_solver_modes_and_balances() -> None:
    off_case = _case()
    off_precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), off_case)
    off_result = run_reactive_transport(off_precompute, off_case)
    np.testing.assert_array_equal(
        off_result.species_sources_mol_per_m3_s, np.zeros((2, 2))
    )
    assert off_result.summary["readiness"]["physically_ready"] is True
    assert "elemental_inventories_moles" in off_result.history[0]
    assert "balance_residuals" in off_result.history[0]
    assert off_result.summary["runtime"]["peak_rss_bytes"] > 0

    payload = off_case.normalized_payload
    payload["mode"] = "nonisothermal"
    payload["mechanism"]["reactions"][0]["thermodynamics"] = {"dH": -20000.0}
    payload["initial_state"]["concentrations_mol_per_m3"] = {
        "A": 10.0,
        "B": 0.0,
    }
    payload["inlets"]["inlet left"]["concentrations_mol_per_m3"] = {
        "A": 10.0,
        "B": 0.0,
    }
    payload["time"]["dt_s"] = 0.01
    payload["time"]["max_dt_s"] = 0.01
    nonisothermal = reactive_case_from_mapping(payload)
    precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), nonisothermal)
    hot = run_reactive_transport(precompute, nonisothermal)
    assert np.all(hot.concentrations_mol_per_m3[:, 0] < 10.0)
    assert np.all(hot.concentrations_mol_per_m3[:, 1] > 0.0)
    assert np.all(hot.temperature_k > 300.0)
    assert hot.summary["balances"]["species"]["A"]["relative_residual"] <= 1.0e-10
    assert hot.summary["balances"]["energy"]["relative_residual"] <= 1.0e-10

    payload["mode"] = "isothermal"
    isothermal = reactive_case_from_mapping(payload)
    precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), isothermal)
    fixed_temperature = run_reactive_transport(precompute, isothermal)
    np.testing.assert_array_equal(fixed_temperature.temperature_k, np.full(2, 300.0))
    assert np.all(fixed_temperature.concentrations_mol_per_m3[:, 0] < 10.0)


def test_reaction_heat_coupling_mismatch_blocks_physical_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _case().normalized_payload
    payload["mode"] = "nonisothermal"
    payload["mechanism"]["reactions"][0]["thermodynamics"] = {"dH": -20000.0}
    payload["initial_state"]["concentrations_mol_per_m3"] = {
        "A": 10.0,
        "B": 0.0,
    }
    payload["inlets"]["inlet left"]["concentrations_mol_per_m3"] = {
        "A": 10.0,
        "B": 0.0,
    }
    payload["time"]["dt_s"] = 0.01
    payload["time"]["max_dt_s"] = 0.01
    case = reactive_case_from_mapping(payload)
    precompute = build_reactive_spatial_precompute(_mesh(), np.zeros(4), case)
    original_advance_reaction = reactive_solver_module.advance_reaction

    def corrupt_sensible_energy_ledger(*args, **kwargs):
        advanced = original_advance_reaction(*args, **kwargs)
        return replace(
            advanced,
            sensible_energy_change_j_per_m3=(
                advanced.sensible_energy_change_j_per_m3 + 10.0
            ),
        )

    monkeypatch.setattr(
        reactive_solver_module,
        "advance_reaction",
        corrupt_sensible_energy_ledger,
    )

    result = run_reactive_transport(precompute, case)
    energy = result.summary["balances"]["energy"]

    assert energy["relative_residual"] <= 1.0e-10
    assert energy["reaction_coupling_relative_residual"] > 1.0e-5
    assert result.summary["readiness"]["energy_balance_ok"] is False
    assert result.summary["readiness"]["physically_ready"] is False
    assert "energy_balance_failed" in result.summary["readiness"]["blocking_reasons"]
