"""Deterministic provenance helpers for standalone chemistry mechanisms."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from microfluidics.chemistry.models import (
    ArrheniusParameters,
    Mechanism,
    ReactionThermodynamics,
)

MECHANISM_FINGERPRINT_SCHEMA = "chemistry_mechanism_fingerprint_v1"


def canonical_mechanism_payload(mechanism: Mechanism) -> dict[str, object]:
    """Return the normalized, computationally relevant mechanism payload.

    Free-form metadata and source paths are deliberately excluded. The same
    resolved mechanism therefore receives the same fingerprint when it is
    loaded from a different location, while changes to species, kinetics,
    thermodynamics, validity ranges, ordering, or mechanism version change the
    fingerprint.
    """
    return {
        "fingerprint_schema": MECHANISM_FINGERPRINT_SCHEMA,
        "name": mechanism.name,
        "version": mechanism.version,
        "temperature_range_k": mechanism.temperature_range_k,
        "pressure_range_pa": mechanism.pressure_range_pa,
        "species": [
            {
                "name": species.name,
                "formula": species.formula,
                "molar_mass_kg_per_mol": species.molar_mass_kg_per_mol,
                "phase": species.phase.value,
                "elemental_composition": _sorted_mapping(species.elemental_composition),
                "aliases": list(species.aliases),
            }
            for species in mechanism.species
        ],
        "reactions": [
            {
                "id": reaction.id,
                "equation": reaction.equation,
                "reactants": _sorted_mapping(reaction.reactants),
                "products": _sorted_mapping(reaction.products),
                "forward": _arrhenius_payload(reaction.forward),
                "reversible": reaction.reversible,
                "reverse": _arrhenius_payload(reaction.reverse),
                "forward_orders": _sorted_mapping(reaction.forward_orders),
                "reverse_orders": _sorted_mapping(reaction.reverse_orders),
                "thermodynamics": _thermodynamics_payload(reaction.thermodynamics),
                "phase": reaction.phase.value if reaction.phase is not None else None,
            }
            for reaction in mechanism.reactions
        ],
    }


def mechanism_sha256(mechanism: Mechanism) -> str:
    """Return a stable SHA-256 fingerprint of a normalized mechanism."""
    canonical_json = json.dumps(
        canonical_mechanism_payload(mechanism),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def _arrhenius_payload(
    parameters: ArrheniusParameters | None,
) -> dict[str, float | None] | None:
    if parameters is None:
        return None
    return {
        "pre_exponential": parameters.pre_exponential,
        "activation_energy_j_per_mol": parameters.activation_energy_j_per_mol,
        "temperature_exponent": parameters.temperature_exponent,
        "reference_temperature_k": parameters.reference_temperature_k,
    }


def _thermodynamics_payload(
    thermodynamics: ReactionThermodynamics | None,
) -> dict[str, float | None] | None:
    if thermodynamics is None:
        return None
    return {
        "delta_h_j_per_mol": thermodynamics.delta_h_j_per_mol,
        "delta_s_j_per_mol_k": thermodynamics.delta_s_j_per_mol_k,
    }


def _sorted_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {key: float(values[key]) for key in sorted(values)}
