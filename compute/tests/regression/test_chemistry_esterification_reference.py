"""Reference locks for reversible esterification kinetics."""

from __future__ import annotations

import numpy as np
import pytest

from microfluidics.chemistry import compile_mechanism, mechanism_from_mapping


def test_reversible_rate_matches_esterification_reference() -> None:
    """Match the locked reversible custom-order rate.

    The expected value is a fixed numerical baseline at 343.15 K for the
    mechanism declared below.
    """
    mechanism = mechanism_from_mapping(
        {
            "metadata": {"name": "Esterification kinetics reference"},
            "species": [
                {
                    "name": "Acetic acid",
                    "formula": "C2H4O2",
                    "phase": "liquid",
                    "molecular_weight": 60.0,
                    "composition": {"C": 2, "H": 4, "O": 2},
                },
                {
                    "name": "Ethanol",
                    "formula": "C2H6O",
                    "phase": "liquid",
                    "molecular_weight": 46.0,
                    "composition": {"C": 2, "H": 6, "O": 1},
                },
                {
                    "name": "Ethyl acetate",
                    "formula": "C4H8O2",
                    "phase": "liquid",
                    "molecular_weight": 88.0,
                    "composition": {"C": 4, "H": 8, "O": 2},
                },
                {
                    "name": "Water",
                    "formula": "H2O",
                    "phase": "liquid",
                    "molecular_weight": 18.0,
                    "composition": {"H": 2, "O": 1},
                },
            ],
            "reactions": [
                {
                    "id": "R1",
                    "equation": ("Acetic acid + Ethanol <=> Ethyl acetate + Water"),
                    "kinetics": {
                        "A_forward": 2.145,
                        "Ea_forward": 13494.0,
                        "A_reverse": 0.0545,
                        "Ea_reverse": 9314.0,
                        "orders": {"Acetic acid": 0.8, "Ethanol": 1.2},
                        "reverse_orders": {
                            "Ethyl acetate": 1.1,
                            "Water": 0.9,
                        },
                    },
                }
            ],
        },
        source_label="esterification regression",
    )
    compiled = compile_mechanism(mechanism, require_reaction_enthalpy=False)

    result = compiled.evaluate(
        np.asarray([4.0, 6.0, 1.5, 0.75]),
        temperature_k=343.15,
        require_heat_release=False,
    )

    assert result.reaction_rates_mol_per_m3_s[0, 0] == pytest.approx(
        0.4904656320123772,
        abs=1e-9,
    )
    assert (
        result.forward_reaction_rates_mol_per_m3_s[0, 0]
        > result.reverse_reaction_rates_mol_per_m3_s[0, 0]
        > 0.0
    )
    np.testing.assert_allclose(
        result.species_sources_mol_per_m3_s[0],
        [
            -0.4904656320123772,
            -0.4904656320123772,
            0.4904656320123772,
            0.4904656320123772,
        ],
        atol=1e-9,
    )
