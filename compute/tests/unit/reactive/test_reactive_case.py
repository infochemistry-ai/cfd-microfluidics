"""Unit tests for the strict reactive-case v1 contract."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from microfluidics.chemistry import MechanismValidationError
from microfluidics.reactive import (
    ReactiveCaseValidationError,
    load_reactive_case,
    reactive_case_from_mapping,
)


def _mechanism(*, include_enthalpy: bool = True) -> dict[str, object]:
    reaction: dict[str, object] = {
        "id": "R1",
        "equation": "A -> B",
        "kinetics": {"A": "2 1/s", "Ea": 0.0},
    }
    if include_enthalpy:
        reaction["thermodynamics"] = {"dH": -20000.0}
    return {
        "version": 1,
        "metadata": {
            "name": "case test",
            "version": "1",
            "temperature_range": [250.0, 1000.0],
        },
        "units": {},
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
        "reactions": [reaction],
    }


def valid_case(*, mode: str = "nonisothermal") -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "test_case",
        "mode": mode,
        "mechanism": _mechanism(),
        "initial_state": {
            "temperature_k": 300.0,
            "operating_pressure_pa": 101325.0,
            "concentrations_mol_per_m3": {"A": 1.0},
        },
        "inlets": {
            "Inlet Left": {
                "temperature_k": 300.0,
                "concentrations_mol_per_m3": {"A": 1.0},
            }
        },
        "material": {
            "density_kg_per_m3": 1000.0,
            "heat_capacity_j_per_kg_k": 4000.0,
            "thermal_diffusivity_m2_s": 1.0e-7,
            "species_diffusivity_m2_s": {"A": 1.0e-9, "B": 1.0e-9},
        },
        "time": {
            "num_steps": 10,
            "dt_mode": "auto",
            "dt_s": 1.0e-3,
            "max_dt_s": 1.0e-3,
            "cfl_target": 0.5,
            "diffusion_stability_factor": 0.5,
            "max_transport_substeps": 16,
        },
        "chemistry_integrator": {
            "relative_tolerance": 1.0e-6,
            "concentration_absolute_tolerance_mol_per_m3": 1.0e-12,
            "temperature_absolute_tolerance_k": 1.0e-8,
            "max_temperature_change_per_substep_k": 1.0,
            "min_substep_fraction": 1.0e-12,
            "max_substeps_per_half_step": 10000,
            "cell_batch_size": 1024,
        },
        "output": {"history_stride": 2, "snapshot_steps": [5, 10]},
    }


def test_duplicate_json_case_field_is_rejected(tmp_path: Path) -> None:
    text = json.dumps(valid_case())
    text = text.replace(
        '"mode": "nonisothermal"',
        '"mode": "off", "mode": "nonisothermal"',
        1,
    )
    path = tmp_path / "duplicate_case.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(
        ReactiveCaseValidationError, match="duplicate JSON field 'mode'"
    ):
        load_reactive_case(path)


def test_case_normalizes_species_and_fingerprint_deterministically() -> None:
    first = reactive_case_from_mapping(valid_case())
    reordered = copy.deepcopy(valid_case())
    reordered["initial_state"]["concentrations_mol_per_m3"] = {"B": 0.0, "A": 1.0}
    second = reactive_case_from_mapping(reordered)

    assert first.species_names == ("A", "B")
    assert first.initial_state.concentrations_mol_per_m3 == (1.0, 0.0)
    assert first.inlets[0].normalized_name == "inlet_left"
    assert first.reactive_case_sha256 == second.reactive_case_sha256


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (lambda payload: payload.update(extra=True), "unknown fields"),
        (
            lambda payload: payload["material"]["species_diffusivity_m2_s"].pop("B"),
            "diffusivity",
        ),
        (
            lambda payload: payload["material"].update(density_kg_per_m3=-1.0),
            "density",
        ),
        (
            lambda payload: payload["initial_state"][
                "concentrations_mol_per_m3"
            ].update(unknown=1.0),
            "unknown species",
        ),
    ],
)
def test_invalid_case_is_rejected(mutator, message: str) -> None:
    payload = valid_case()
    mutator(payload)
    with pytest.raises(ReactiveCaseValidationError, match=message):
        reactive_case_from_mapping(payload)


def test_nonisothermal_requires_enthalpy() -> None:
    payload = valid_case()
    payload["mechanism"] = _mechanism(include_enthalpy=False)
    with pytest.raises(ReactiveCaseValidationError, match="requires dH"):
        reactive_case_from_mapping(payload)


def test_isothermal_requires_uniform_temperature() -> None:
    payload = valid_case(mode="isothermal")
    payload["inlets"]["Inlet Left"]["temperature_k"] = 301.0
    with pytest.raises(ReactiveCaseValidationError, match="identical"):
        reactive_case_from_mapping(payload)


@pytest.mark.parametrize("field", ["case_id", "mode"])
def test_string_contract_fields_reject_non_strings(field: str) -> None:
    payload = valid_case()
    payload[field] = 123
    with pytest.raises(ReactiveCaseValidationError, match=field):
        reactive_case_from_mapping(payload)


def test_nonfinite_mechanism_number_uses_mechanism_error_type() -> None:
    payload = valid_case()
    payload["mechanism"]["reactions"][0]["kinetics"]["Ea"] = math.nan
    with pytest.raises(MechanismValidationError, match="finite"):
        reactive_case_from_mapping(payload)
