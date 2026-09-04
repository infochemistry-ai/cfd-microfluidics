"""Unit tests for mechanism loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microfluidics.chemistry import (
    MechanismValidationError,
    load_mechanism,
    mechanism_from_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = REPO_ROOT / "data" / "examples" / "chemistry"


def _payload() -> dict[str, object]:
    return {
        "version": 1,
        "metadata": {"name": "loader test"},
        "units": {"activation-energy": "kJ/mol"},
        "species": [
            {
                "name": "reactant",
                "formula": "A",
                "phase": "liquid",
                "molecular_weight": 10.0,
            },
            {
                "name": "product",
                "formula": "B",
                "phase": "liquid",
                "molecular_weight": 10.0,
            },
        ],
        "reactions": [
            {
                "id": "R1",
                "equation": "A -> B",
                "kinetics": {"A": "2.0 1/s", "Ea": 5.0},
                "thermodynamics": {"dH": "-20 kJ/mol"},
            }
        ],
    }


def test_loads_tracked_exothermic_example() -> None:
    mechanism = load_mechanism(EXAMPLE_ROOT / "exothermic_ab.yaml")

    assert mechanism.name == "Exothermic homogeneous A + B -> C"
    assert mechanism.species_names == ("A", "B", "C")
    assert mechanism.reactions[0].forward.pre_exponential == pytest.approx(1e-3)
    assert mechanism.reactions[0].thermodynamics is not None
    assert mechanism.reactions[0].thermodynamics.delta_h_j_per_mol == -50000.0


def test_formula_aliases_are_canonicalized_to_species_names() -> None:
    mechanism = mechanism_from_mapping(_payload())
    reaction = mechanism.reactions[0]

    assert dict(reaction.reactants) == {"reactant": 1.0}
    assert dict(reaction.products) == {"product": 1.0}
    assert reaction.forward.activation_energy_j_per_mol == 5000.0


def test_unitless_numeric_strings_follow_schema_coercion() -> None:
    payload = _payload()
    payload["species"][0]["molecular_weight"] = "10.0"
    payload["reactions"][0]["kinetics"]["n"] = "0.5"

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.species[0].molar_mass_kg_per_mol == 0.01
    assert mechanism.reactions[0].forward.temperature_exponent == 0.5


def test_liter_second_order_pre_exponential_converts_to_si() -> None:
    payload = _payload()
    payload["species"] = [
        {"name": "A", "molecular_weight": 10.0},
        {"name": "B", "molecular_weight": 20.0},
        {"name": "C", "molecular_weight": 30.0},
    ]
    payload["reactions"] = [
        {
            "id": "R1",
            "equation": "A + B -> C",
            "kinetics": {"A": "0.005 L/(mol*s)", "Ea": 0.0},
            "thermodynamics": {"dH": -1.0},
        }
    ]

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.reactions[0].forward.pre_exponential == pytest.approx(5e-6)


def test_explicit_energy_literal_overrides_document_unit() -> None:
    payload = _payload()
    payload["reactions"][0]["kinetics"]["Ea"] = "2 kcal/mol"

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.reactions[0].forward.activation_energy_j_per_mol == 8368.0


def test_pressure_range_mpa_is_canonicalized_to_pa() -> None:
    payload = _payload()
    payload["metadata"]["pressure_range"] = [0.1, 2.0]

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.pressure_range_pa == (100000.0, 2000000.0)


def test_charged_species_do_not_break_plus_separator_parsing() -> None:
    payload = {
        "metadata": {"name": "neutralization syntax"},
        "species": [
            {"name": "H+", "molecular_weight": 1.0},
            {"name": "OH-", "molecular_weight": 17.0},
            {"name": "H2O", "molecular_weight": 18.0},
        ],
        "reactions": [
            {
                "id": "R1",
                "equation": "H+ + OH- -> H2O",
                "kinetics": {"A": 1.0, "Ea": 0.0},
                "thermodynamics": {"dH": -57000.0},
            }
        ],
    }

    mechanism = mechanism_from_mapping(payload)

    assert dict(mechanism.reactions[0].reactants) == {"H+": 1.0, "OH-": 1.0}


def test_spaced_plus_inside_species_parentheses_is_not_a_separator() -> None:
    payload = {
        "metadata": {"name": "parenthesized species"},
        "species": [
            {"name": "M-P(n + 1)", "molecular_weight": 10.0},
            {"name": "B", "molecular_weight": 20.0},
            {"name": "C", "molecular_weight": 30.0},
        ],
        "reactions": [
            {
                "id": "R1",
                "equation": "M-P(n + 1) + B -> C",
                "kinetics": {"A": 1.0, "Ea": 0.0},
                "thermodynamics": {"dH": -1.0},
            }
        ],
    }

    mechanism = mechanism_from_mapping(payload)

    assert dict(mechanism.reactions[0].reactants) == {
        "M-P(n + 1)": 1.0,
        "B": 1.0,
    }


def test_reversible_reaction_accepts_explicit_reverse_kinetics() -> None:
    payload = _payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "A <=> B",
        "kinetics": {
            "A_forward": "2 1/s",
            "Ea_forward": 0.0,
            "A_reverse": "1 1/s",
            "Ea_reverse": 0.0,
        },
        "thermodynamics": {"dH": 0.0},
    }

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.reactions[0].reversible
    assert mechanism.reactions[0].reverse is not None


def test_legacy_reversible_arrow_is_not_misparsed_as_irreversible() -> None:
    payload = _payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "A <-> B",
        "kinetics": {
            "A_forward": "2 1/s",
            "Ea_forward": 0.0,
            "A_reverse": "1 1/s",
            "Ea_reverse": 0.0,
        },
        "thermodynamics": {"dH": 0.0},
    }

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.reactions[0].reversible


def test_reversible_reaction_accepts_thermodynamic_reverse() -> None:
    payload = _payload()
    payload["reactions"][0] = {
        "id": "R1",
        "equation": "A <=> B",
        "kinetics": {"A": "2 1/s", "Ea": 0.0},
        "thermodynamics": {"dH": 1000.0, "dS": "2 J/(mol*K)"},
    }

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.reactions[0].reverse is None
    assert mechanism.reactions[0].thermodynamics is not None
    assert mechanism.reactions[0].thermodynamics.delta_s_j_per_mol_k == 2.0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(version=2), "unsupported mechanism version"),
        (
            lambda value: value["reactions"][0].update(equation="Z -> B"),
            "unknown species token",
        ),
        (
            lambda value: value["reactions"][0].update(reversible=True),
            "disagrees with equation arrow",
        ),
        (
            lambda value: value["reactions"][0]["kinetics"].update(A="1 L/(mol*s)"),
            "unit implies reaction order",
        ),
    ],
)
def test_rejects_invalid_mechanism_variants(mutator, message: str) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(MechanismValidationError, match=message):
        mechanism_from_mapping(payload)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["metadata"].update(temperatur_range=[250.0, 1500.0]),
        lambda value: value["species"][0].update(molecular_weigth=10.0),
        lambda value: value["reactions"][0].update(equaton="A -> B"),
        lambda value: value["reactions"][0]["kinetics"].update(n_foward=2.5),
        lambda value: value["reactions"][0]["thermodynamics"].update(
            delta_heat=-20000.0
        ),
    ],
)
def test_unknown_and_misspelled_fields_are_rejected(mutator) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(MechanismValidationError, match="unknown fields"):
        mechanism_from_mapping(payload)


def test_arbitrary_provenance_is_allowed_only_in_explicit_extensions() -> None:
    payload = _payload()
    payload["metadata"]["extensions"] = {
        "vendor": {"dataset": "reviewed-v3", "ticket": 42}
    }

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.metadata["extensions"]["vendor"]["ticket"] == 42


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(name="top-level name"),
        lambda value: value["metadata"].update(
            temperature_range=[250.0, 1500.0],
            temperature_range_k=[300.0, 1200.0],
        ),
        lambda value: value["metadata"].update(
            pressure_range=[0.1, 2.0],
            pressure_range_pa=[100000.0, 2000000.0],
        ),
        lambda value: value["metadata"].update(activation_energy_unit="J/mol"),
        lambda value: value["species"][0].update(molar_mass_kg_per_mol=0.01),
        lambda value: value["reactions"][0]["kinetics"].update(A_forward=3.0),
        lambda value: value["reactions"][0]["kinetics"].update(Ea_forward=6.0),
        lambda value: value["reactions"][0]["kinetics"].update(n=1.0, n_forward=2.0),
        lambda value: value["reactions"][0]["kinetics"].update(
            T_ref=300.0,
            T_ref_forward=400.0,
        ),
        lambda value: value["reactions"][0]["thermodynamics"].update(delta_h=-10000.0),
        lambda value: value["reactions"][0]["thermodynamics"].update(
            dS=1.0,
            delta_s=2.0,
        ),
    ],
)
def test_conflicting_alias_fields_are_rejected(mutator) -> None:
    payload = _payload()
    mutator(payload)

    with pytest.raises(MechanismValidationError, match="mutually exclusive"):
        mechanism_from_mapping(payload)


def test_schema_version_and_mechanism_revision_are_distinct() -> None:
    payload = _payload()
    payload["metadata"]["version"] = "reviewed-2026.07"

    mechanism = mechanism_from_mapping(payload)

    assert mechanism.version == "reviewed-2026.07"


def test_duplicate_json_mechanism_field_is_rejected(tmp_path: Path) -> None:
    text = json.dumps(_payload())
    text = text.replace('"Ea": 5.0', '"Ea": 0.0, "Ea": 5.0', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MechanismValidationError, match="duplicate key 'Ea'"):
        load_mechanism(path)


def test_duplicate_yaml_mechanism_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """
version: 1
metadata:
  name: duplicate
species:
  - name: A
    phase: liquid
    molecular_weight: 10
  - name: B
    phase: liquid
    molecular_weight: 10
reactions:
  - id: R1
    equation: A -> B
    kinetics:
      A: 1
      A: 2
      Ea: 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(MechanismValidationError, match="duplicate key 'A'"):
        load_mechanism(path)


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MechanismValidationError, match="does not exist"):
        load_mechanism(tmp_path / "missing.yaml")


def test_esterification_example_preserves_kinetic_only_status() -> None:
    mechanism = load_mechanism(EXAMPLE_ROOT / "esterification_kinetics.yaml")

    assert mechanism.reactions[0].forward.pre_exponential == pytest.approx(5e-6)
    assert mechanism.reactions[0].thermodynamics is None
    assert mechanism.species_names == ("C2H4O2", "C2H5OH", "H2O", "Et-Ac")
