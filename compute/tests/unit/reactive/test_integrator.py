"""Analytic and mode tests for the Euler-Heun reaction integrator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from microfluidics.chemistry import compile_mechanism, mechanism_from_mapping
from microfluidics.reactive import (
    ChemistryIntegrationError,
    ChemistryIntegratorV1,
    ReactiveMaterialV1,
    ReactiveWalltimeLimitError,
)
from microfluidics.reactive.integrator import advance_reaction


def _chemistry(*, delta_h: float = -20000.0):
    return compile_mechanism(
        mechanism_from_mapping(
            {
                "metadata": {
                    "name": "first order",
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
                        "kinetics": {"A": "2 1/s", "Ea": 0.0},
                        "thermodynamics": {"dH": delta_h},
                    }
                ],
            }
        )
    )


def _bimolecular_chemistry():
    return compile_mechanism(
        mechanism_from_mapping(
            {
                "metadata": {
                    "name": "closed bimolecular",
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
                        "molecular_weight": 20.0,
                        "composition": {"Y": 1.0},
                    },
                    {
                        "name": "C",
                        "phase": "liquid",
                        "molecular_weight": 30.0,
                        "composition": {"X": 1.0, "Y": 1.0},
                    },
                ],
                "reactions": [
                    {
                        "id": "R1",
                        "equation": "A + B -> C",
                        "kinetics": {"A": "1e-3 m^3/(mol*s)", "Ea": 0.0},
                        "thermodynamics": {"dH": -20000.0},
                    }
                ],
            }
        )
    )


SETTINGS = ChemistryIntegratorV1(
    relative_tolerance=1.0e-8,
    concentration_absolute_tolerance_mol_per_m3=1.0e-12,
    temperature_absolute_tolerance_k=1.0e-10,
    max_temperature_change_per_substep_k=1.0,
    min_substep_fraction=1.0e-12,
    max_substeps_per_half_step=100000,
    cell_batch_size=2,
)
MATERIAL = ReactiveMaterialV1(
    density_kg_per_m3=1000.0,
    heat_capacity_j_per_kg_k=4000.0,
    thermal_diffusivity_m2_s=0.0,
    species_diffusivity_m2_s=(0.0, 0.0),
)


def test_isothermal_first_order_matches_analytic_batch() -> None:
    chemistry = _chemistry()
    initial = np.array([[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
    result = advance_reaction(
        chemistry,
        initial,
        np.full(3, 300.0),
        pressure_pa=101325.0,
        interval_s=0.2,
        mode="isothermal",
        material=MATERIAL,
        settings=SETTINGS,
    )

    expected_a = initial[:, 0] * math.exp(-0.4)
    np.testing.assert_allclose(result.concentrations_mol_per_m3[:, 0], expected_a)
    np.testing.assert_allclose(
        result.concentrations_mol_per_m3[:, 1], initial[:, 0] - expected_a
    )
    np.testing.assert_array_equal(result.temperature_k, np.full(3, 300.0))
    np.testing.assert_allclose(
        np.sum(result.concentrations_mol_per_m3, axis=1),
        np.sum(initial, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.stats.accepted_substeps > 0
    assert result.stats.rejected_substeps > 0


def test_isothermal_closed_bimolecular_batch_matches_analytic_solution() -> None:
    initial_value = 100.0
    interval_s = 2.0
    result = advance_reaction(
        _bimolecular_chemistry(),
        np.array([[initial_value, initial_value, 0.0]]),
        np.array([300.0]),
        pressure_pa=101325.0,
        interval_s=interval_s,
        mode="isothermal",
        material=ReactiveMaterialV1(
            density_kg_per_m3=1000.0,
            heat_capacity_j_per_kg_k=4000.0,
            thermal_diffusivity_m2_s=0.0,
            species_diffusivity_m2_s=(0.0, 0.0, 0.0),
        ),
        settings=SETTINGS,
    )

    expected_reactant = initial_value / (1.0 + 1.0e-3 * initial_value * interval_s)
    np.testing.assert_allclose(
        result.concentrations_mol_per_m3[0],
        [
            expected_reactant,
            expected_reactant,
            initial_value - expected_reactant,
        ],
        rtol=5.0e-5,
        atol=1.0e-10,
    )
    molecular_weights = np.array([0.010, 0.020, 0.030])
    assert (
        abs(
            float(
                (result.concentrations_mol_per_m3[0] - [100.0, 100.0, 0.0])
                @ molecular_weights
            )
        )
        <= 1.0e-10
    )


@pytest.mark.parametrize(
    ("delta_h", "direction"),
    [(-20000.0, 1.0), (20000.0, -1.0)],
)
def test_nonisothermal_heat_sign(delta_h: float, direction: float) -> None:
    result = advance_reaction(
        _chemistry(delta_h=delta_h),
        np.array([[10.0, 0.0]]),
        np.array([300.0]),
        pressure_pa=101325.0,
        interval_s=0.01,
        mode="nonisothermal",
        material=MATERIAL,
        settings=SETTINGS,
    )
    assert direction * (result.temperature_k[0] - 300.0) > 0.0
    np.testing.assert_allclose(
        result.sensible_energy_change_j_per_m3,
        MATERIAL.density_kg_per_m3
        * MATERIAL.heat_capacity_j_per_kg_k
        * (result.temperature_k - 300.0),
    )
    assert result.heat_release_integral_j_per_m3[0] != 0.0
    np.testing.assert_allclose(
        result.heat_release_integral_j_per_m3,
        result.sensible_energy_change_j_per_m3,
        rtol=1.0e-8,
        atol=1.0e-12,
    )


def test_off_mode_does_not_evaluate_chemistry() -> None:
    initial = np.array([[1.0, 0.0]])
    result = advance_reaction(
        _chemistry(),
        initial,
        np.array([300.0]),
        pressure_pa=101325.0,
        interval_s=1.0,
        mode="off",
        material=MATERIAL,
        settings=SETTINGS,
    )
    np.testing.assert_array_equal(result.concentrations_mol_per_m3, initial)
    np.testing.assert_array_equal(result.heat_release_integral_j_per_m3, np.zeros(1))
    assert result.stats.evaluations == 0


def test_stiffness_attempt_cap_fails_closed() -> None:
    capped = ChemistryIntegratorV1(
        relative_tolerance=1.0e-12,
        concentration_absolute_tolerance_mol_per_m3=1.0e-14,
        temperature_absolute_tolerance_k=1.0e-12,
        max_temperature_change_per_substep_k=1.0,
        min_substep_fraction=1.0e-12,
        max_substeps_per_half_step=1,
        cell_batch_size=2,
    )
    with pytest.raises(ChemistryIntegrationError, match="max_substeps"):
        advance_reaction(
            _chemistry(),
            np.array([[1.0, 0.0]]),
            np.array([300.0]),
            pressure_pa=101325.0,
            interval_s=1.0,
            mode="isothermal",
            material=MATERIAL,
            settings=capped,
        )


def test_chemistry_subcycling_honors_expired_walltime_deadline() -> None:
    with pytest.raises(ReactiveWalltimeLimitError, match="walltime limit"):
        advance_reaction(
            _chemistry(),
            np.array([[1.0, 0.0]]),
            np.array([300.0]),
            pressure_pa=101325.0,
            interval_s=1.0,
            mode="isothermal",
            material=MATERIAL,
            settings=SETTINGS,
            walltime_deadline_monotonic=0.0,
        )
