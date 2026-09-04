"""Unit tests for compile-once standalone chemistry evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from microfluidics.chemistry import (
    ChemistryEvaluationError,
    MechanismValidationError,
    compile_mechanism,
    load_mechanism,
    mechanism_sha256,
    mechanism_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = REPO_ROOT / "data" / "examples" / "chemistry"


def _compiled_example():
    return compile_mechanism(load_mechanism(EXAMPLE_ROOT / "exothermic_ab.yaml"))


def _ab_payload(*, molecular_weight_b: float = 10.0) -> dict[str, object]:
    return {
        "metadata": {"name": "A B", "temperature_range": [250.0, 1000.0]},
        "species": [
            {
                "name": "A",
                "phase": "gas",
                "molecular_weight": 10.0,
                "composition": {"X": 1.0},
            },
            {
                "name": "B",
                "phase": "gas",
                "molecular_weight": molecular_weight_b,
                "composition": {"X": 1.0},
            },
        ],
        "reactions": [
            {
                "id": "R1",
                "equation": "A -> B",
                "kinetics": {"A": "2 1/s", "Ea": 0.0},
                "thermodynamics": {"dH": -20000.0},
            }
        ],
    }


def test_irreversible_sources_heat_sign_and_mass_balance() -> None:
    compiled = _compiled_example()

    result = compiled.evaluate(
        {"A": 100.0, "B": 50.0},
        temperature_k=350.0,
    )

    rate = result.reaction_rates_mol_per_m3_s[0, 0]
    assert rate > 0.0
    assert result.forward_reaction_rates_mol_per_m3_s[0, 0] == pytest.approx(rate)
    assert result.reverse_reaction_rates_mol_per_m3_s[0, 0] == 0.0
    np.testing.assert_allclose(
        result.species_sources_mol_per_m3_s[0],
        [-rate, -rate, rate],
    )
    np.testing.assert_allclose(
        result.species_creation_rates_mol_per_m3_s[0],
        [0.0, 0.0, rate],
    )
    np.testing.assert_allclose(
        result.species_destruction_rates_mol_per_m3_s[0],
        [rate, rate, 0.0],
    )
    np.testing.assert_allclose(
        result.species_creation_rates_mol_per_m3_s
        - result.species_destruction_rates_mol_per_m3_s,
        result.species_sources_mol_per_m3_s,
    )
    assert result.reaction_heat_release_w_per_m3 is not None
    assert result.reaction_heat_release_w_per_m3[0, 0] == pytest.approx(50000.0 * rate)
    assert result.heat_release_w_per_m3 is not None
    assert result.heat_release_w_per_m3[0] == pytest.approx(50000.0 * rate)
    assert result.mass_balance_residual_kg_per_m3_s[0] == pytest.approx(0.0)


def test_lightweight_sources_match_full_evaluation_for_batch() -> None:
    compiled = _compiled_example()
    concentrations = np.array([[100.0, 50.0, 0.0], [20.0, 10.0, 1.0]], dtype=np.float64)
    temperatures = np.array([350.0, 400.0], dtype=np.float64)

    full = compiled.evaluate(
        concentrations,
        temperature_k=temperatures,
        require_heat_release=True,
    )
    lightweight = compiled.evaluate_sources(
        concentrations,
        temperature_k=temperatures,
        require_heat_release=True,
    )

    np.testing.assert_allclose(
        lightweight.species_sources_mol_per_m3_s,
        full.species_sources_mol_per_m3_s,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lightweight.heat_release_w_per_m3,
        full.heat_release_w_per_m3,
        rtol=1e-12,
        atol=1e-14,
    )


def test_temperature_is_runtime_vector_not_compile_option() -> None:
    compiled = _compiled_example()
    concentrations = np.asarray([[100.0, 50.0, 0.0], [100.0, 50.0, 0.0]])

    result = compiled.evaluate(
        concentrations,
        temperature_k=np.asarray([300.0, 600.0]),
    )

    assert result.forward_rate_constants[1, 0] > result.forward_rate_constants[0, 0]
    assert (
        result.reaction_rates_mol_per_m3_s[1, 0]
        > result.reaction_rates_mol_per_m3_s[0, 0]
    )


def test_scalar_concentrations_broadcast_across_temperature_vector() -> None:
    compiled = _compiled_example()

    result = compiled.evaluate(
        {"A": 100.0, "B": 50.0},
        temperature_k=np.asarray([300.0, 600.0]),
    )

    assert result.concentrations_mol_per_m3.shape == (2, 3)
    np.testing.assert_allclose(
        result.concentrations_mol_per_m3[0],
        result.concentrations_mol_per_m3[1],
    )
    assert (
        result.reaction_rates_mol_per_m3_s[1, 0]
        > result.reaction_rates_mol_per_m3_s[0, 0]
    )


def test_mapping_arrays_are_vectorized_and_missing_species_default_to_zero() -> None:
    compiled = _compiled_example()

    result = compiled.evaluate(
        {"A": np.asarray([10.0, 20.0]), "B": 5.0},
        temperature_k=350.0,
    )

    assert not result.scalar_input
    np.testing.assert_allclose(result.concentrations_mol_per_m3[:, 2], 0.0)
    assert result.reaction_rates_mol_per_m3_s[1, 0] == pytest.approx(
        2.0 * result.reaction_rates_mol_per_m3_s[0, 0]
    )


def test_vectorized_result_matches_scalar_evaluations() -> None:
    compiled = _compiled_example()
    states = np.asarray([[10.0, 4.0, 0.0], [20.0, 8.0, 0.0]])
    temperatures = np.asarray([325.0, 425.0])

    vectorized = compiled.evaluate(states, temperature_k=temperatures)
    scalar_rates = [
        compiled.evaluate(state, temperature_k=temperature).reaction_rates_mol_per_m3_s[
            0
        ]
        for state, temperature in zip(states, temperatures, strict=True)
    ]

    np.testing.assert_allclose(
        vectorized.reaction_rates_mol_per_m3_s,
        np.asarray(scalar_rates),
    )


def test_explicit_reversible_rate_and_custom_orders() -> None:
    payload = _ab_payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "A <=> B",
        "kinetics": {
            # Fractional-order constants use an explicitly canonical numeric A;
            # a first-order unit literal would be dimensionally inconsistent.
            "A_forward": 2.0,
            "Ea_forward": 0.0,
            "A_reverse": "1 1/s",
            "Ea_reverse": 0.0,
            "orders": {"A": 0.5},
            "reverse_orders": {"B": 1.0},
        },
        "thermodynamics": {"dH": 0.0},
    }
    compiled = compile_mechanism(mechanism_from_mapping(payload))

    result = compiled.evaluate({"A": 4.0, "B": 1.0}, temperature_k=300.0)

    assert result.forward_reaction_rates_mol_per_m3_s[0, 0] == pytest.approx(4.0)
    assert result.reverse_reaction_rates_mol_per_m3_s[0, 0] == pytest.approx(1.0)
    assert result.reaction_rates_mol_per_m3_s[0, 0] == pytest.approx(3.0)
    np.testing.assert_allclose(
        result.species_creation_rates_mol_per_m3_s[0], [1.0, 4.0]
    )
    np.testing.assert_allclose(
        result.species_destruction_rates_mol_per_m3_s[0], [4.0, 1.0]
    )
    np.testing.assert_allclose(result.species_sources_mol_per_m3_s[0], [-3.0, 3.0])


def test_thermodynamic_reverse_matches_forward_over_equilibrium_constant() -> None:
    payload = _ab_payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "A <=> B",
        "kinetics": {"A": "2 1/s", "Ea": 0.0},
        "thermodynamics": {"dH": 1000.0, "dS": 2.0},
    }
    compiled = compile_mechanism(mechanism_from_mapping(payload))
    temperature = 400.0

    result = compiled.evaluate({"A": 1.0, "B": 1.0}, temperature_k=temperature)
    expected_k_eq = math.exp(
        -(1000.0 - temperature * 2.0) / (8.31446261815324 * temperature)
    )

    assert result.reverse_rate_constants[0, 0] == pytest.approx(2.0 / expected_k_eq)


def test_thermodynamic_reverse_rejects_dimensionally_different_orders() -> None:
    payload = _ab_payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "2 A <=> B",
        "kinetics": {"A": 2.0, "Ea": 0.0},
        "thermodynamics": {"dH": 1000.0, "dS": 2.0},
    }

    with pytest.raises(MechanismValidationError, match="total orders differ"):
        mechanism_from_mapping(payload)


def test_reference_temperature_form_returns_authored_value_at_reference() -> None:
    payload = _ab_payload()
    payload["reactions"][0]["kinetics"] = {
        "A": "3 1/s",
        "Ea": 10000.0,
        "n": 1.5,
        "T_ref": 400.0,
    }
    compiled = compile_mechanism(mechanism_from_mapping(payload))

    result = compiled.evaluate({"A": 1.0}, temperature_k=400.0)

    assert result.forward_rate_constants[0, 0] == pytest.approx(3.0)


def test_missing_enthalpy_requires_explicit_kinetic_only_mode() -> None:
    mechanism = load_mechanism(EXAMPLE_ROOT / "esterification_kinetics.yaml")
    with pytest.raises(MechanismValidationError, match="requires dH"):
        compile_mechanism(mechanism)

    compiled = compile_mechanism(mechanism, require_reaction_enthalpy=False)
    result = compiled.evaluate(
        {"C2H4O2": 10.0, "C2H5OH": 10.0},
        temperature_k=343.15,
        require_heat_release=False,
    )

    assert result.heat_release_w_per_m3 is None
    assert result.missing_enthalpy_reactions == ("r1",)


def test_unbalanced_molecular_weights_fail_by_default_but_remain_diagnostic() -> None:
    mechanism = mechanism_from_mapping(_ab_payload(molecular_weight_b=11.0))

    with pytest.raises(MechanismValidationError, match="molecular-weight balance"):
        compile_mechanism(mechanism)

    compiled = compile_mechanism(mechanism, validate_mass_balance=False)
    result = compiled.evaluate({"A": 1.0}, temperature_k=300.0)
    assert result.mass_balance_residual_kg_per_m3_s[0] != 0.0


def test_elemental_imbalance_fails_even_when_molecular_weights_match() -> None:
    payload = _ab_payload()
    payload["species"][1]["composition"] = {"Y": 1.0}

    with pytest.raises(MechanismValidationError, match="elemental conservation"):
        compile_mechanism(mechanism_from_mapping(payload))


def test_surface_and_multiphase_mechanisms_are_out_of_v1_scope() -> None:
    surface = _ab_payload()
    surface["species"][0]["phase"] = "surface"
    with pytest.raises(MechanismValidationError, match="does not support"):
        compile_mechanism(mechanism_from_mapping(surface))

    multiphase = _ab_payload()
    multiphase["species"][1]["phase"] = "liquid"
    with pytest.raises(MechanismValidationError, match="cannot mix gas and liquid"):
        compile_mechanism(mechanism_from_mapping(multiphase))


def test_authored_temperature_range_is_enforced_and_can_be_diagnostic() -> None:
    mechanism = mechanism_from_mapping(_ab_payload())
    compiled = compile_mechanism(mechanism)

    with pytest.raises(ChemistryEvaluationError, match="outside authored"):
        compiled.evaluate({"A": 1.0}, temperature_k=1200.0)

    diagnostic = compile_mechanism(mechanism, strict_temperature_range=False)
    result = diagnostic.evaluate({"A": 1.0}, temperature_k=1200.0)
    assert result.temperature_k[0] == 1200.0


@pytest.mark.parametrize(
    ("concentrations", "temperature", "message"),
    [
        ({"Z": 1.0}, 300.0, "unknown concentration species"),
        ({"A": -1.0}, 300.0, "non-negative"),
        ({"A": 1.0}, 0.0, "temperature_k values must be positive"),
        (np.asarray([1.0, 2.0, 3.0]), 300.0, "must have length 2"),
    ],
)
def test_invalid_evaluation_states_fail_clearly(
    concentrations,
    temperature: float,
    message: str,
) -> None:
    compiled = compile_mechanism(mechanism_from_mapping(_ab_payload()))

    with pytest.raises(ChemistryEvaluationError, match=message):
        compiled.evaluate(concentrations, temperature_k=temperature)


def test_scalar_and_vector_json_contracts_are_finite_and_explicit() -> None:
    compiled = _compiled_example()
    scalar = compiled.evaluate({"A": 1.0, "B": 2.0}, temperature_k=350.0).to_dict()
    vector = compiled.evaluate(
        np.asarray([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]]),
        temperature_k=np.asarray([350.0, 360.0]),
    ).to_dict()

    assert scalar["contract_version"] == "chemistry_evaluation_v1"
    assert scalar["mechanism_provenance"]["version"] == "1.0"
    assert len(scalar["mechanism_provenance"]["sha256"]) == 64
    assert scalar["rate_constants"]["R1"]["reverse"] is None
    assert scalar["reverse_reaction_rates_mol_per_m3_s"]["R1"] == 0.0
    assert scalar["reaction_heat_release_w_per_m3"]["R1"] > 0.0
    assert (
        scalar["species_creation_rates_mol_per_m3_s"]["C"]
        == scalar["forward_reaction_rates_mol_per_m3_s"]["R1"]
    )
    assert isinstance(vector["state"]["temperature_k"], list)
    assert isinstance(vector["forward_reaction_rates_mol_per_m3_s"]["R1"], list)
    assert vector["unit_basis"]["heat_release"] == "W/m^3"


def test_mechanism_fingerprint_is_stable_and_tracks_computational_changes() -> None:
    payload = _ab_payload()
    first = mechanism_from_mapping(payload, source_label="first/location.yaml")
    second = mechanism_from_mapping(payload, source_label="second/location.yaml")

    first_hash = mechanism_sha256(first)
    assert first_hash == mechanism_sha256(second)
    assert len(first_hash) == 64
    assert set(first_hash) <= set("0123456789abcdef")

    changed = _ab_payload()
    changed["reactions"][0]["kinetics"]["A"] = "3 1/s"
    assert mechanism_sha256(mechanism_from_mapping(changed)) != first_hash


def test_multiple_reactions_report_per_reaction_heat_and_species_breakdown() -> None:
    payload = _ab_payload()
    payload["metadata"]["name"] = "A to B to C"
    payload["species"].append(
        {
            "name": "C",
            "phase": "gas",
            "molecular_weight": 10.0,
            "composition": {"X": 1.0},
        }
    )
    payload["reactions"].append(
        {
            "id": "R2",
            "equation": "B -> C",
            "kinetics": {"A": "0.5 1/s", "Ea": 0.0},
            "thermodynamics": {"dH": 10000.0},
        }
    )
    compiled = compile_mechanism(mechanism_from_mapping(payload))

    result = compiled.evaluate({"A": 2.0, "B": 4.0}, temperature_k=300.0)

    np.testing.assert_allclose(result.forward_reaction_rates_mol_per_m3_s[0], [4, 2])
    np.testing.assert_allclose(result.reverse_reaction_rates_mol_per_m3_s[0], [0, 0])
    np.testing.assert_allclose(result.reaction_rates_mol_per_m3_s[0], [4, 2])
    np.testing.assert_allclose(result.species_sources_mol_per_m3_s[0], [-4, 2, 2])
    assert result.reaction_heat_release_w_per_m3 is not None
    np.testing.assert_allclose(
        result.reaction_heat_release_w_per_m3[0], [80000.0, -20000.0]
    )
    assert result.heat_release_w_per_m3 is not None
    assert result.heat_release_w_per_m3[0] == pytest.approx(60000.0)
