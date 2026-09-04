"""Compile-once, runtime-temperature chemistry evaluation kernels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from microfluidics.chemistry.errors import (
    ChemistryEvaluationError,
    MechanismValidationError,
)
from microfluidics.chemistry.models import Mechanism, SpeciesPhase
from microfluidics.chemistry.provenance import (
    MECHANISM_FINGERPRINT_SCHEMA,
    mechanism_sha256,
)

R_J_PER_MOL_K = 8.31446261815324
CHEMISTRY_CONTRACT_VERSION = "chemistry_evaluation_v1"


@dataclass(slots=True)
class ChemistrySources:
    """Lightweight source-only chemistry result for solver coupling."""

    species_sources_mol_per_m3_s: np.ndarray
    heat_release_w_per_m3: np.ndarray | None
    scalar_input: bool


@dataclass(slots=True)
class ChemistryEvaluation:
    """One scalar or vectorized standalone chemistry evaluation."""

    mechanism_name: str
    mechanism_version: str
    mechanism_sha256: str
    mechanism_temperature_range_k: tuple[float, float] | None
    mechanism_pressure_range_pa: tuple[float, float] | None
    species_names: tuple[str, ...]
    reaction_ids: tuple[str, ...]
    temperature_k: np.ndarray
    pressure_pa: np.ndarray
    concentrations_mol_per_m3: np.ndarray
    forward_rate_constants: np.ndarray
    reverse_rate_constants: np.ndarray
    reversible_mask: np.ndarray
    forward_reaction_rates_mol_per_m3_s: np.ndarray
    reverse_reaction_rates_mol_per_m3_s: np.ndarray
    reaction_rates_mol_per_m3_s: np.ndarray
    species_creation_rates_mol_per_m3_s: np.ndarray
    species_destruction_rates_mol_per_m3_s: np.ndarray
    species_sources_mol_per_m3_s: np.ndarray
    reaction_heat_release_w_per_m3: np.ndarray | None
    heat_release_w_per_m3: np.ndarray | None
    mass_balance_residual_kg_per_m3_s: np.ndarray
    reaction_mass_balance_relative_error: np.ndarray
    missing_enthalpy_reactions: tuple[str, ...]
    scalar_input: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, unit-explicit payload."""
        if self.scalar_input:
            state: dict[str, object] = {
                "temperature_k": float(self.temperature_k[0]),
                "pressure_pa": float(self.pressure_pa[0]),
                "concentrations_mol_per_m3": {
                    name: float(self.concentrations_mol_per_m3[0, index])
                    for index, name in enumerate(self.species_names)
                },
            }
            rate_constants = {
                reaction_id: {
                    "forward": float(self.forward_rate_constants[0, index]),
                    "reverse": (
                        float(self.reverse_rate_constants[0, index])
                        if bool(self.reversible_mask[index])
                        else None
                    ),
                }
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            reaction_rates: object = {
                reaction_id: float(self.reaction_rates_mol_per_m3_s[0, index])
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            forward_reaction_rates: object = {
                reaction_id: float(self.forward_reaction_rates_mol_per_m3_s[0, index])
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            reverse_reaction_rates: object = {
                reaction_id: float(self.reverse_reaction_rates_mol_per_m3_s[0, index])
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            species_sources: object = {
                name: float(self.species_sources_mol_per_m3_s[0, index])
                for index, name in enumerate(self.species_names)
            }
            species_creation_rates: object = {
                name: float(self.species_creation_rates_mol_per_m3_s[0, index])
                for index, name in enumerate(self.species_names)
            }
            species_destruction_rates: object = {
                name: float(self.species_destruction_rates_mol_per_m3_s[0, index])
                for index, name in enumerate(self.species_names)
            }
            reaction_heat_release: object = (
                {
                    reaction_id: float(self.reaction_heat_release_w_per_m3[0, index])
                    for index, reaction_id in enumerate(self.reaction_ids)
                }
                if self.reaction_heat_release_w_per_m3 is not None
                else None
            )
            heat_release: object = (
                float(self.heat_release_w_per_m3[0])
                if self.heat_release_w_per_m3 is not None
                else None
            )
            mass_residual: object = float(self.mass_balance_residual_kg_per_m3_s[0])
        else:
            state = {
                "temperature_k": self.temperature_k.tolist(),
                "pressure_pa": self.pressure_pa.tolist(),
                "concentrations_mol_per_m3": {
                    name: self.concentrations_mol_per_m3[:, index].tolist()
                    for index, name in enumerate(self.species_names)
                },
            }
            rate_constants = {
                reaction_id: {
                    "forward": self.forward_rate_constants[:, index].tolist(),
                    "reverse": (
                        self.reverse_rate_constants[:, index].tolist()
                        if bool(self.reversible_mask[index])
                        else None
                    ),
                }
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            reaction_rates = {
                reaction_id: self.reaction_rates_mol_per_m3_s[:, index].tolist()
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            forward_reaction_rates = {
                reaction_id: self.forward_reaction_rates_mol_per_m3_s[:, index].tolist()
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            reverse_reaction_rates = {
                reaction_id: self.reverse_reaction_rates_mol_per_m3_s[:, index].tolist()
                for index, reaction_id in enumerate(self.reaction_ids)
            }
            species_sources = {
                name: self.species_sources_mol_per_m3_s[:, index].tolist()
                for index, name in enumerate(self.species_names)
            }
            species_creation_rates = {
                name: self.species_creation_rates_mol_per_m3_s[:, index].tolist()
                for index, name in enumerate(self.species_names)
            }
            species_destruction_rates = {
                name: self.species_destruction_rates_mol_per_m3_s[:, index].tolist()
                for index, name in enumerate(self.species_names)
            }
            reaction_heat_release = (
                {
                    reaction_id: self.reaction_heat_release_w_per_m3[:, index].tolist()
                    for index, reaction_id in enumerate(self.reaction_ids)
                }
                if self.reaction_heat_release_w_per_m3 is not None
                else None
            )
            heat_release = (
                self.heat_release_w_per_m3.tolist()
                if self.heat_release_w_per_m3 is not None
                else None
            )
            mass_residual = self.mass_balance_residual_kg_per_m3_s.tolist()

        return {
            "contract_version": CHEMISTRY_CONTRACT_VERSION,
            "mechanism": self.mechanism_name,
            "mechanism_provenance": {
                "name": self.mechanism_name,
                "version": self.mechanism_version,
                "sha256": self.mechanism_sha256,
                "fingerprint_schema": MECHANISM_FINGERPRINT_SCHEMA,
                "temperature_range_k": (
                    list(self.mechanism_temperature_range_k)
                    if self.mechanism_temperature_range_k is not None
                    else None
                ),
                "pressure_range_pa": (
                    list(self.mechanism_pressure_range_pa)
                    if self.mechanism_pressure_range_pa is not None
                    else None
                ),
            },
            "unit_basis": {
                "temperature": "K",
                "pressure": "Pa",
                "concentration": "mol/m^3",
                "reaction_rate": "mol/(m^3*s)",
                "species_source": "mol/(m^3*s)",
                "reaction_enthalpy": "J/mol",
                "heat_release": "W/m^3",
            },
            "state": state,
            "rate_constants": rate_constants,
            "forward_reaction_rates_mol_per_m3_s": forward_reaction_rates,
            "reverse_reaction_rates_mol_per_m3_s": reverse_reaction_rates,
            "reaction_rates_mol_per_m3_s": reaction_rates,
            "species_creation_rates_mol_per_m3_s": species_creation_rates,
            "species_destruction_rates_mol_per_m3_s": species_destruction_rates,
            "species_sources_mol_per_m3_s": species_sources,
            "reaction_heat_release_w_per_m3": reaction_heat_release,
            "heat_release_w_per_m3": heat_release,
            "mass_balance_residual_kg_per_m3_s": mass_residual,
            "diagnostics": {
                "scalar_input": self.scalar_input,
                "evaluation_count": int(self.temperature_k.size),
                "missing_enthalpy_reactions": list(self.missing_enthalpy_reactions),
                "max_abs_mass_balance_residual_kg_per_m3_s": float(
                    np.max(np.abs(self.mass_balance_residual_kg_per_m3_s))
                ),
                "reaction_mass_balance_relative_error": {
                    reaction_id: float(self.reaction_mass_balance_relative_error[index])
                    for index, reaction_id in enumerate(self.reaction_ids)
                },
            },
        }


@dataclass(slots=True)
class CompiledChemistry:
    """Mechanism topology compiled once; kinetic constants remain runtime-T values."""

    mechanism: Mechanism
    mechanism_sha256: str
    species_index: dict[str, int]
    stoichiometric_matrix: np.ndarray
    reactant_stoichiometric_matrix: np.ndarray
    product_stoichiometric_matrix: np.ndarray
    forward_order_matrix: np.ndarray
    reverse_order_matrix: np.ndarray
    forward_pre_exponential: np.ndarray
    forward_activation_energy_j_per_mol: np.ndarray
    forward_temperature_exponent: np.ndarray
    forward_reference_temperature_k: np.ndarray
    reverse_pre_exponential: np.ndarray
    reverse_activation_energy_j_per_mol: np.ndarray
    reverse_temperature_exponent: np.ndarray
    reverse_reference_temperature_k: np.ndarray
    reversible_mask: np.ndarray
    derived_reverse_mask: np.ndarray
    reaction_enthalpy_j_per_mol: np.ndarray
    reaction_entropy_j_per_mol_k: np.ndarray
    molecular_weights_kg_per_mol: np.ndarray
    reaction_mass_balance_relative_error: np.ndarray
    missing_enthalpy_reactions: tuple[str, ...]
    strict_temperature_range: bool = True
    strict_pressure_range: bool = True

    @property
    def species_names(self) -> tuple[str, ...]:
        return self.mechanism.species_names

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        return self.mechanism.reaction_ids

    def evaluate(
        self,
        concentrations_mol_per_m3: Mapping[str, object] | np.ndarray,
        *,
        temperature_k: float | np.ndarray,
        pressure_pa: float | np.ndarray = 101325.0,
        require_heat_release: bool = True,
    ) -> ChemistryEvaluation:
        """Evaluate scalar or batched chemistry without any mesh/solver coupling.

        Array concentrations use shape ``(n_species,)`` or
        ``(n_evaluations, n_species)``.  Mapping values may be scalars or
        one-dimensional arrays and are broadcast to a common evaluation count.
        """
        concentrations, concentration_scalar = self._normalize_concentrations(
            concentrations_mol_per_m3
        )
        concentration_count = concentrations.shape[0]
        count = max(
            concentration_count,
            _state_value_count(temperature_k, "temperature_k"),
            _state_value_count(pressure_pa, "pressure_pa"),
        )
        if concentration_count == 1 and count > 1:
            concentrations = np.repeat(concentrations, count, axis=0)
        elif concentration_count != count:
            raise ChemistryEvaluationError(
                "concentrations evaluation count "
                f"{concentration_count} cannot broadcast to {count}"
            )
        temperatures, temperature_scalar = _broadcast_state_value(
            temperature_k,
            count,
            "temperature_k",
        )
        pressures, pressure_scalar = _broadcast_state_value(
            pressure_pa,
            count,
            "pressure_pa",
        )
        if np.any(temperatures <= 0.0):
            raise ChemistryEvaluationError("temperature_k values must be positive")
        if np.any(pressures <= 0.0):
            raise ChemistryEvaluationError("pressure_pa values must be positive")
        self._validate_authored_ranges(temperatures, pressures)

        forward_constants = _arrhenius_values(
            temperatures,
            self.forward_pre_exponential,
            self.forward_activation_energy_j_per_mol,
            self.forward_temperature_exponent,
            self.forward_reference_temperature_k,
        )
        reverse_constants = np.zeros_like(forward_constants)
        explicit_reverse = self.reversible_mask & ~self.derived_reverse_mask
        if np.any(explicit_reverse):
            reverse_constants[:, explicit_reverse] = _arrhenius_values(
                temperatures,
                self.reverse_pre_exponential[explicit_reverse],
                self.reverse_activation_energy_j_per_mol[explicit_reverse],
                self.reverse_temperature_exponent[explicit_reverse],
                self.reverse_reference_temperature_k[explicit_reverse],
            )
        if np.any(self.derived_reverse_mask):
            indexes = self.derived_reverse_mask
            delta_h = self.reaction_enthalpy_j_per_mol[indexes]
            delta_s = self.reaction_entropy_j_per_mol_k[indexes]
            exponent = -(
                delta_h[None, :] - temperatures[:, None] * delta_s[None, :]
            ) / (R_J_PER_MOL_K * temperatures[:, None])
            equilibrium_constants = _checked_exp(exponent, "equilibrium constants")
            reverse_constants[:, indexes] = (
                forward_constants[:, indexes] / equilibrium_constants
            )

        forward_reaction_rates = np.zeros(
            (count, len(self.reaction_ids)), dtype=np.float64
        )
        reverse_reaction_rates = np.zeros_like(forward_reaction_rates)
        for reaction_index in range(len(self.reaction_ids)):
            forward_rate = forward_constants[:, reaction_index].copy()
            for species_index, order in enumerate(
                self.forward_order_matrix[reaction_index]
            ):
                if order == 0.0:
                    continue
                base = concentrations[:, species_index]
                if order < 0.0:
                    base = np.maximum(base, 1e-30)
                forward_rate *= np.power(base, order)

            reverse_rate = np.zeros(count, dtype=np.float64)
            if self.reversible_mask[reaction_index]:
                reverse_rate = reverse_constants[:, reaction_index].copy()
                for species_index, order in enumerate(
                    self.reverse_order_matrix[reaction_index]
                ):
                    if order == 0.0:
                        continue
                    base = concentrations[:, species_index]
                    if order < 0.0:
                        base = np.maximum(base, 1e-30)
                    reverse_rate *= np.power(base, order)
            forward_reaction_rates[:, reaction_index] = forward_rate
            reverse_reaction_rates[:, reaction_index] = reverse_rate

        reaction_rates = forward_reaction_rates - reverse_reaction_rates

        if not (
            np.isfinite(forward_reaction_rates).all()
            and np.isfinite(reverse_reaction_rates).all()
            and np.isfinite(reaction_rates).all()
        ):
            raise ChemistryEvaluationError(
                "reaction-rate evaluation produced non-finite values; check kinetics, "
                "orders, concentrations and temperature"
            )

        species_sources = reaction_rates @ self.stoichiometric_matrix.T
        species_creation_rates = (
            forward_reaction_rates @ self.product_stoichiometric_matrix.T
            + reverse_reaction_rates @ self.reactant_stoichiometric_matrix.T
        )
        species_destruction_rates = (
            forward_reaction_rates @ self.reactant_stoichiometric_matrix.T
            + reverse_reaction_rates @ self.product_stoichiometric_matrix.T
        )
        mass_residual = species_sources @ self.molecular_weights_kg_per_mol
        reaction_heat_release = None
        heat_release = None
        if self.missing_enthalpy_reactions:
            if require_heat_release:
                missing = ", ".join(self.missing_enthalpy_reactions)
                raise ChemistryEvaluationError(
                    "heat release requires explicit dH for every reaction; missing: "
                    f"{missing}"
                )
        else:
            reaction_heat_release = -(
                reaction_rates * self.reaction_enthalpy_j_per_mol[None, :]
            )
            heat_release = np.sum(reaction_heat_release, axis=1)

        scalar_input = concentration_scalar and temperature_scalar and pressure_scalar
        return ChemistryEvaluation(
            mechanism_name=self.mechanism.name,
            mechanism_version=self.mechanism.version,
            mechanism_sha256=self.mechanism_sha256,
            mechanism_temperature_range_k=self.mechanism.temperature_range_k,
            mechanism_pressure_range_pa=self.mechanism.pressure_range_pa,
            species_names=self.species_names,
            reaction_ids=self.reaction_ids,
            temperature_k=temperatures,
            pressure_pa=pressures,
            concentrations_mol_per_m3=concentrations,
            forward_rate_constants=forward_constants,
            reverse_rate_constants=reverse_constants,
            reversible_mask=self.reversible_mask.copy(),
            forward_reaction_rates_mol_per_m3_s=forward_reaction_rates,
            reverse_reaction_rates_mol_per_m3_s=reverse_reaction_rates,
            reaction_rates_mol_per_m3_s=reaction_rates,
            species_creation_rates_mol_per_m3_s=species_creation_rates,
            species_destruction_rates_mol_per_m3_s=species_destruction_rates,
            species_sources_mol_per_m3_s=species_sources,
            reaction_heat_release_w_per_m3=reaction_heat_release,
            heat_release_w_per_m3=heat_release,
            mass_balance_residual_kg_per_m3_s=mass_residual,
            reaction_mass_balance_relative_error=(
                self.reaction_mass_balance_relative_error.copy()
            ),
            missing_enthalpy_reactions=self.missing_enthalpy_reactions,
            scalar_input=scalar_input,
        )

    def evaluate_sources(
        self,
        concentrations_mol_per_m3: Mapping[str, object] | np.ndarray,
        *,
        temperature_k: float | np.ndarray,
        pressure_pa: float | np.ndarray = 101325.0,
        require_heat_release: bool = True,
    ) -> ChemistrySources:
        """Evaluate only species and heat sources without diagnostic arrays."""

        concentrations, concentration_scalar = self._normalize_concentrations(
            concentrations_mol_per_m3
        )
        concentration_count = concentrations.shape[0]
        count = max(
            concentration_count,
            _state_value_count(temperature_k, "temperature_k"),
            _state_value_count(pressure_pa, "pressure_pa"),
        )
        if concentration_count == 1 and count > 1:
            concentrations = np.repeat(concentrations, count, axis=0)
        elif concentration_count != count:
            raise ChemistryEvaluationError(
                "concentrations evaluation count "
                f"{concentration_count} cannot broadcast to {count}"
            )
        temperatures, temperature_scalar = _broadcast_state_value(
            temperature_k, count, "temperature_k"
        )
        pressures, pressure_scalar = _broadcast_state_value(
            pressure_pa, count, "pressure_pa"
        )
        if np.any(temperatures <= 0.0):
            raise ChemistryEvaluationError("temperature_k values must be positive")
        if np.any(pressures <= 0.0):
            raise ChemistryEvaluationError("pressure_pa values must be positive")
        self._validate_authored_ranges(temperatures, pressures)

        forward_constants = _arrhenius_values(
            temperatures,
            self.forward_pre_exponential,
            self.forward_activation_energy_j_per_mol,
            self.forward_temperature_exponent,
            self.forward_reference_temperature_k,
        )
        reverse_constants = np.zeros_like(forward_constants)
        explicit_reverse = self.reversible_mask & ~self.derived_reverse_mask
        if np.any(explicit_reverse):
            reverse_constants[:, explicit_reverse] = _arrhenius_values(
                temperatures,
                self.reverse_pre_exponential[explicit_reverse],
                self.reverse_activation_energy_j_per_mol[explicit_reverse],
                self.reverse_temperature_exponent[explicit_reverse],
                self.reverse_reference_temperature_k[explicit_reverse],
            )
        if np.any(self.derived_reverse_mask):
            indexes = self.derived_reverse_mask
            delta_h = self.reaction_enthalpy_j_per_mol[indexes]
            delta_s = self.reaction_entropy_j_per_mol_k[indexes]
            exponent = -(
                delta_h[None, :] - temperatures[:, None] * delta_s[None, :]
            ) / (R_J_PER_MOL_K * temperatures[:, None])
            reverse_constants[:, indexes] = forward_constants[
                :, indexes
            ] / _checked_exp(exponent, "equilibrium constants")

        reaction_rates = np.zeros((count, len(self.reaction_ids)), dtype=np.float64)
        for reaction_index in range(len(self.reaction_ids)):
            forward_rate = forward_constants[:, reaction_index].copy()
            for species_index, order in enumerate(
                self.forward_order_matrix[reaction_index]
            ):
                if order != 0.0:
                    base = concentrations[:, species_index]
                    if order < 0.0:
                        base = np.maximum(base, 1e-30)
                    forward_rate *= np.power(base, order)

            reverse_rate = np.zeros(count, dtype=np.float64)
            if self.reversible_mask[reaction_index]:
                reverse_rate = reverse_constants[:, reaction_index].copy()
                for species_index, order in enumerate(
                    self.reverse_order_matrix[reaction_index]
                ):
                    if order != 0.0:
                        base = concentrations[:, species_index]
                        if order < 0.0:
                            base = np.maximum(base, 1e-30)
                        reverse_rate *= np.power(base, order)
            reaction_rates[:, reaction_index] = forward_rate - reverse_rate

        if not np.isfinite(reaction_rates).all():
            raise ChemistryEvaluationError(
                "reaction-rate evaluation produced non-finite values; check kinetics, "
                "orders, concentrations and temperature"
            )
        species_sources = reaction_rates @ self.stoichiometric_matrix.T
        heat_release = None
        if self.missing_enthalpy_reactions:
            if require_heat_release:
                missing = ", ".join(self.missing_enthalpy_reactions)
                raise ChemistryEvaluationError(
                    "heat release requires explicit dH for every reaction; missing: "
                    f"{missing}"
                )
        else:
            heat_release = np.sum(
                -(reaction_rates * self.reaction_enthalpy_j_per_mol[None, :]),
                axis=1,
            )
        return ChemistrySources(
            species_sources_mol_per_m3_s=species_sources,
            heat_release_w_per_m3=heat_release,
            scalar_input=(
                concentration_scalar and temperature_scalar and pressure_scalar
            ),
        )

    def _normalize_concentrations(
        self,
        raw: Mapping[str, object] | np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        species_count = len(self.species_names)
        if isinstance(raw, Mapping):
            unknown = sorted(set(raw) - set(self.species_names))
            if unknown:
                raise ChemistryEvaluationError(
                    f"unknown concentration species {unknown}; known: {list(self.species_names)}"
                )
            arrays: dict[str, np.ndarray] = {}
            count = 1
            scalar = True
            for name, value in raw.items():
                try:
                    array = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ChemistryEvaluationError(
                        f"concentration for {name} must be numeric"
                    ) from exc
                if array.ndim == 0:
                    array = array.reshape(1)
                elif array.ndim == 1:
                    scalar = False
                else:
                    raise ChemistryEvaluationError(
                        f"concentration mapping value for {name} must be scalar or 1-D"
                    )
                if array.size == 0:
                    raise ChemistryEvaluationError(
                        f"concentration mapping value for {name} cannot be empty"
                    )
                count = max(count, int(array.size))
                arrays[name] = array

            concentrations = np.zeros((count, species_count), dtype=np.float64)
            for name, array in arrays.items():
                if array.size not in {1, count}:
                    raise ChemistryEvaluationError(
                        f"concentration for {name} has length {array.size}; expected 1 or {count}"
                    )
                concentrations[:, self.species_index[name]] = (
                    float(array[0]) if array.size == 1 else array
                )
        else:
            try:
                array = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ChemistryEvaluationError(
                    "concentrations must be numeric"
                ) from exc
            scalar = array.ndim == 1
            if array.ndim == 1:
                if array.shape != (species_count,):
                    raise ChemistryEvaluationError(
                        "one-dimensional concentrations must have length "
                        f"{species_count}, got {array.shape}"
                    )
                concentrations = array.reshape(1, species_count).copy()
            elif array.ndim == 2:
                if array.shape[1] != species_count or array.shape[0] == 0:
                    raise ChemistryEvaluationError(
                        "two-dimensional concentrations must have shape "
                        f"(n, {species_count}), got {array.shape}"
                    )
                concentrations = array.copy()
            else:
                raise ChemistryEvaluationError(
                    "concentrations must be a mapping, 1-D vector or 2-D matrix"
                )

        if not np.isfinite(concentrations).all():
            raise ChemistryEvaluationError("concentrations must contain finite values")
        if np.any(concentrations < 0.0):
            raise ChemistryEvaluationError("concentrations must be non-negative")
        return concentrations, scalar

    def _validate_authored_ranges(
        self,
        temperatures: np.ndarray,
        pressures: np.ndarray,
    ) -> None:
        temperature_range = self.mechanism.temperature_range_k
        if self.strict_temperature_range and temperature_range is not None:
            lower, upper = temperature_range
            if np.any((temperatures < lower) | (temperatures > upper)):
                raise ChemistryEvaluationError(
                    "temperature_k is outside authored mechanism range "
                    f"[{lower}, {upper}] K"
                )
        pressure_range = self.mechanism.pressure_range_pa
        if self.strict_pressure_range and pressure_range is not None:
            lower, upper = pressure_range
            if np.any((pressures < lower) | (pressures > upper)):
                raise ChemistryEvaluationError(
                    "pressure_pa is outside authored mechanism range "
                    f"[{lower}, {upper}] Pa"
                )


def compile_mechanism(
    mechanism: Mechanism,
    *,
    require_reaction_enthalpy: bool = True,
    validate_mass_balance: bool = True,
    mass_balance_relative_tolerance: float = 1e-6,
    validate_element_balance: bool = True,
    strict_temperature_range: bool = True,
    strict_pressure_range: bool = True,
) -> CompiledChemistry:
    """Compile a homogeneous mechanism without freezing temperature-dependent rates."""
    if mass_balance_relative_tolerance < 0.0:
        raise MechanismValidationError(
            "mass_balance_relative_tolerance must be non-negative"
        )
    _validate_homogeneous_scope(mechanism)

    species_index = {name: index for index, name in enumerate(mechanism.species_names)}
    species_count = len(mechanism.species)
    reaction_count = len(mechanism.reactions)
    stoichiometric_matrix = np.zeros((species_count, reaction_count), dtype=np.float64)
    reactant_stoichiometry = np.zeros_like(stoichiometric_matrix)
    product_stoichiometry = np.zeros_like(stoichiometric_matrix)
    forward_orders = np.zeros((reaction_count, species_count), dtype=np.float64)
    reverse_orders = np.zeros((reaction_count, species_count), dtype=np.float64)

    forward_a = np.empty(reaction_count, dtype=np.float64)
    forward_ea = np.empty(reaction_count, dtype=np.float64)
    forward_n = np.empty(reaction_count, dtype=np.float64)
    forward_t_ref = np.full(reaction_count, np.nan, dtype=np.float64)
    reverse_a = np.full(reaction_count, np.nan, dtype=np.float64)
    reverse_ea = np.full(reaction_count, np.nan, dtype=np.float64)
    reverse_n = np.full(reaction_count, np.nan, dtype=np.float64)
    reverse_t_ref = np.full(reaction_count, np.nan, dtype=np.float64)
    reversible = np.zeros(reaction_count, dtype=bool)
    derived_reverse = np.zeros(reaction_count, dtype=bool)
    enthalpy = np.full(reaction_count, np.nan, dtype=np.float64)
    entropy = np.full(reaction_count, np.nan, dtype=np.float64)

    for reaction_index, reaction in enumerate(mechanism.reactions):
        for species, coefficient in reaction.reactants.items():
            reactant_stoichiometry[species_index[species], reaction_index] = coefficient
        for species, coefficient in reaction.products.items():
            product_stoichiometry[species_index[species], reaction_index] = coefficient
        for species, coefficient in reaction.net_stoichiometry.items():
            stoichiometric_matrix[species_index[species], reaction_index] = coefficient
        for species, order in reaction.forward_orders.items():
            forward_orders[reaction_index, species_index[species]] = order
        for species, order in reaction.reverse_orders.items():
            reverse_orders[reaction_index, species_index[species]] = order

        forward_a[reaction_index] = reaction.forward.pre_exponential
        forward_ea[reaction_index] = reaction.forward.activation_energy_j_per_mol
        forward_n[reaction_index] = reaction.forward.temperature_exponent
        if reaction.forward.reference_temperature_k is not None:
            forward_t_ref[reaction_index] = reaction.forward.reference_temperature_k

        reversible[reaction_index] = reaction.reversible
        if reaction.reverse is not None:
            reverse_a[reaction_index] = reaction.reverse.pre_exponential
            reverse_ea[reaction_index] = reaction.reverse.activation_energy_j_per_mol
            reverse_n[reaction_index] = reaction.reverse.temperature_exponent
            if reaction.reverse.reference_temperature_k is not None:
                reverse_t_ref[reaction_index] = reaction.reverse.reference_temperature_k
        elif reaction.reversible:
            derived_reverse[reaction_index] = True

        if reaction.thermodynamics is not None:
            enthalpy[reaction_index] = reaction.thermodynamics.delta_h_j_per_mol
            if reaction.thermodynamics.delta_s_j_per_mol_k is not None:
                entropy[reaction_index] = reaction.thermodynamics.delta_s_j_per_mol_k

    missing_enthalpy = tuple(
        reaction_id
        for reaction_id, value in zip(mechanism.reaction_ids, enthalpy, strict=True)
        if not math.isfinite(float(value))
    )
    if require_reaction_enthalpy and missing_enthalpy:
        raise MechanismValidationError(
            "standalone heat-release evaluation requires dH for every reaction; missing: "
            + ", ".join(missing_enthalpy)
        )

    molecular_weights = np.asarray(
        [item.molar_mass_kg_per_mol for item in mechanism.species],
        dtype=np.float64,
    )
    reaction_mass_residual = molecular_weights @ stoichiometric_matrix
    reaction_mass_scale = np.empty(reaction_count, dtype=np.float64)
    for index, reaction in enumerate(mechanism.reactions):
        reactant_mass = sum(
            coefficient * molecular_weights[species_index[species]]
            for species, coefficient in reaction.reactants.items()
        )
        product_mass = sum(
            coefficient * molecular_weights[species_index[species]]
            for species, coefficient in reaction.products.items()
        )
        reaction_mass_scale[index] = max(reactant_mass, product_mass, 1e-30)
    relative_mass_error = np.abs(reaction_mass_residual) / reaction_mass_scale
    if validate_mass_balance:
        bad = np.flatnonzero(relative_mass_error > mass_balance_relative_tolerance)
        if bad.size:
            details = ", ".join(
                f"{mechanism.reaction_ids[index]}={relative_mass_error[index]:.3e}"
                for index in bad.tolist()
            )
            raise MechanismValidationError(
                "reaction molecular-weight balance exceeds relative tolerance "
                f"{mass_balance_relative_tolerance:g}: {details}"
            )

    if validate_element_balance:
        _validate_element_conservation(mechanism)

    return CompiledChemistry(
        mechanism=mechanism,
        mechanism_sha256=mechanism_sha256(mechanism),
        species_index=species_index,
        stoichiometric_matrix=stoichiometric_matrix,
        reactant_stoichiometric_matrix=reactant_stoichiometry,
        product_stoichiometric_matrix=product_stoichiometry,
        forward_order_matrix=forward_orders,
        reverse_order_matrix=reverse_orders,
        forward_pre_exponential=forward_a,
        forward_activation_energy_j_per_mol=forward_ea,
        forward_temperature_exponent=forward_n,
        forward_reference_temperature_k=forward_t_ref,
        reverse_pre_exponential=reverse_a,
        reverse_activation_energy_j_per_mol=reverse_ea,
        reverse_temperature_exponent=reverse_n,
        reverse_reference_temperature_k=reverse_t_ref,
        reversible_mask=reversible,
        derived_reverse_mask=derived_reverse,
        reaction_enthalpy_j_per_mol=enthalpy,
        reaction_entropy_j_per_mol_k=entropy,
        molecular_weights_kg_per_mol=molecular_weights,
        reaction_mass_balance_relative_error=relative_mass_error,
        missing_enthalpy_reactions=missing_enthalpy,
        strict_temperature_range=strict_temperature_range,
        strict_pressure_range=strict_pressure_range,
    )


def _validate_homogeneous_scope(mechanism: Mechanism) -> None:
    unsupported = {
        item.name: item.phase.value
        for item in mechanism.species
        if item.phase
        in {
            SpeciesPhase.SOLID,
            SpeciesPhase.SURFACE,
            SpeciesPhase.POLYMER,
        }
    }
    if unsupported:
        raise MechanismValidationError(
            "homogeneous chemistry v1 does not support solid, surface or polymer "
            f"species: {unsupported}"
        )
    unsupported_reactions = {
        reaction.id: reaction.phase.value
        for reaction in mechanism.reactions
        if reaction.phase
        in {SpeciesPhase.SOLID, SpeciesPhase.SURFACE, SpeciesPhase.POLYMER}
    }
    if unsupported_reactions:
        raise MechanismValidationError(
            "homogeneous chemistry v1 does not support solid, surface or polymer "
            f"reaction phases: {unsupported_reactions}"
        )
    explicit_phases = {
        item.phase
        for item in mechanism.species
        if item.phase in {SpeciesPhase.GAS, SpeciesPhase.LIQUID}
    }
    if len(explicit_phases) > 1:
        raise MechanismValidationError(
            "homogeneous chemistry v1 cannot mix gas and liquid phase species"
        )


def _validate_element_conservation(mechanism: Mechanism) -> None:
    species_by_name = {item.name: item for item in mechanism.species}
    for reaction in mechanism.reactions:
        involved = [species_by_name[name] for name in reaction.net_stoichiometry]
        if not involved or not all(item.elemental_composition for item in involved):
            continue
        elements = sorted(
            {element for item in involved for element in item.elemental_composition}
        )
        residuals = {
            element: sum(
                coefficient
                * float(
                    species_by_name[species].elemental_composition.get(element, 0.0)
                )
                for species, coefficient in reaction.net_stoichiometry.items()
            )
            for element in elements
        }
        bad = {
            element: value for element, value in residuals.items() if abs(value) > 1e-12
        }
        if bad:
            raise MechanismValidationError(
                f"reaction {reaction.id} violates elemental conservation: {bad}"
            )


def _broadcast_state_value(
    raw: float | np.ndarray,
    count: int,
    label: str,
) -> tuple[np.ndarray, bool]:
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ChemistryEvaluationError(f"{label} must be numeric") from exc
    scalar = array.ndim == 0
    if array.ndim == 0:
        values = np.full(count, float(array), dtype=np.float64)
    elif array.ndim == 1 and array.size in {1, count}:
        values = (
            np.full(count, float(array[0]), dtype=np.float64)
            if array.size == 1
            else array.copy()
        )
    else:
        raise ChemistryEvaluationError(
            f"{label} must be scalar or a 1-D array of length 1 or {count}"
        )
    if not np.isfinite(values).all():
        raise ChemistryEvaluationError(f"{label} must contain finite values")
    return values, scalar


def _state_value_count(raw: float | np.ndarray, label: str) -> int:
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ChemistryEvaluationError(f"{label} must be numeric") from exc
    if array.ndim == 0:
        return 1
    if array.ndim == 1 and array.size > 0:
        return int(array.size)
    raise ChemistryEvaluationError(f"{label} must be scalar or a non-empty 1-D array")


def _arrhenius_values(
    temperatures: np.ndarray,
    pre_exponential: np.ndarray,
    activation_energy: np.ndarray,
    temperature_exponent: np.ndarray,
    reference_temperature: np.ndarray,
) -> np.ndarray:
    temperature_matrix = temperatures[:, None]
    result = np.empty((temperatures.size, pre_exponential.size), dtype=np.float64)
    reference_mask = np.isfinite(reference_temperature)
    if np.any(~reference_mask):
        indexes = ~reference_mask
        exponent = -activation_energy[indexes][None, :] / (
            R_J_PER_MOL_K * temperature_matrix
        )
        result[:, indexes] = (
            pre_exponential[indexes][None, :]
            * np.power(temperature_matrix, temperature_exponent[indexes][None, :])
            * _checked_exp(exponent, "Arrhenius constants")
        )
    if np.any(reference_mask):
        indexes = reference_mask
        references = reference_temperature[indexes][None, :]
        exponent = (
            -activation_energy[indexes][None, :]
            / R_J_PER_MOL_K
            * (1.0 / temperature_matrix - 1.0 / references)
        )
        result[:, indexes] = (
            pre_exponential[indexes][None, :]
            * np.power(
                temperature_matrix / references,
                temperature_exponent[indexes][None, :],
            )
            * _checked_exp(exponent, "reference Arrhenius constants")
        )
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ChemistryEvaluationError(
            "Arrhenius evaluation produced invalid rate constants"
        )
    return result


def _checked_exp(exponent: np.ndarray, label: str) -> np.ndarray:
    if np.any(exponent > math.log(np.finfo(np.float64).max)):
        raise ChemistryEvaluationError(f"{label} overflowed the float64 range")
    result = np.exp(exponent)
    if not np.isfinite(result).all():
        raise ChemistryEvaluationError(f"{label} produced non-finite values")
    return result
