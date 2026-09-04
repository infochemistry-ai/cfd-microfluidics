"""Provider-neutral contracts for homogeneous chemical kinetics.

The contracts are independent from reactors, meshes, transport and thermal
solvers so the
chemistry package can be tested and evolved on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from microfluidics.chemistry.errors import MechanismValidationError


class SpeciesPhase(str, Enum):
    """High-level phase hint for a chemical species."""

    GAS = "gas"
    LIQUID = "liquid"
    SOLID = "solid"
    SURFACE = "surface"
    POLYMER = "polymer"
    PSEUDO = "pseudo"
    UNKNOWN = "unknown"


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MechanismValidationError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MechanismValidationError(f"{label} must be finite")
    return numeric


def _non_empty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MechanismValidationError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Species:
    """One chemical species in canonical SI units."""

    name: str
    formula: str
    molar_mass_kg_per_mol: float
    phase: SpeciesPhase = SpeciesPhase.UNKNOWN
    elemental_composition: Mapping[str, float] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "species name"))
        object.__setattr__(self, "formula", _non_empty(self.formula, "species formula"))
        molar_mass = _finite(self.molar_mass_kg_per_mol, "molar mass")
        if molar_mass <= 0.0:
            raise MechanismValidationError("molar mass must be positive")
        object.__setattr__(self, "molar_mass_kg_per_mol", molar_mass)
        try:
            phase = (
                self.phase
                if isinstance(self.phase, SpeciesPhase)
                else SpeciesPhase(str(self.phase).strip().lower())
            )
        except ValueError as exc:
            raise MechanismValidationError(
                f"species {self.name} has unsupported phase {self.phase!r}"
            ) from exc
        object.__setattr__(self, "phase", phase)

        composition: dict[str, float] = {}
        for element, count in self.elemental_composition.items():
            symbol = _non_empty(element, "element symbol")
            numeric = _finite(count, f"elemental composition for {symbol}")
            if numeric < 0.0:
                raise MechanismValidationError(
                    f"elemental composition for {symbol} must be non-negative"
                )
            composition[symbol] = numeric
        object.__setattr__(self, "elemental_composition", MappingProxyType(composition))

        aliases = tuple(_non_empty(alias, "species alias") for alias in self.aliases)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class ArrheniusParameters:
    """Arrhenius parameters with canonical activation energy in J/mol.

    ``pre_exponential`` is canonical for concentrations in mol/m^3.  Its
    dimensions depend on the total reaction order.
    """

    pre_exponential: float
    activation_energy_j_per_mol: float
    temperature_exponent: float = 0.0
    reference_temperature_k: float | None = None

    def __post_init__(self) -> None:
        pre_exponential = _finite(self.pre_exponential, "pre-exponential factor")
        if pre_exponential <= 0.0:
            raise MechanismValidationError("pre-exponential factor must be positive")
        activation_energy = _finite(
            self.activation_energy_j_per_mol,
            "activation energy",
        )
        if activation_energy < 0.0:
            raise MechanismValidationError("activation energy must be non-negative")
        exponent = _finite(self.temperature_exponent, "temperature exponent")
        reference = self.reference_temperature_k
        if reference is not None:
            reference = _finite(reference, "reference temperature")
            if reference <= 0.0:
                raise MechanismValidationError("reference temperature must be positive")

        object.__setattr__(self, "pre_exponential", pre_exponential)
        object.__setattr__(self, "activation_energy_j_per_mol", activation_energy)
        object.__setattr__(self, "temperature_exponent", exponent)
        object.__setattr__(self, "reference_temperature_k", reference)


@dataclass(frozen=True, slots=True)
class ReactionThermodynamics:
    """Reaction thermodynamics in canonical SI units."""

    delta_h_j_per_mol: float
    delta_s_j_per_mol_k: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_h_j_per_mol",
            _finite(self.delta_h_j_per_mol, "reaction enthalpy"),
        )
        if self.delta_s_j_per_mol_k is not None:
            object.__setattr__(
                self,
                "delta_s_j_per_mol_k",
                _finite(self.delta_s_j_per_mol_k, "reaction entropy"),
            )


@dataclass(frozen=True, slots=True)
class Reaction:
    """A homogeneous power-law reaction."""

    id: str
    equation: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    forward: ArrheniusParameters
    reversible: bool = False
    reverse: ArrheniusParameters | None = None
    forward_orders: Mapping[str, float] = field(default_factory=dict)
    reverse_orders: Mapping[str, float] = field(default_factory=dict)
    thermodynamics: ReactionThermodynamics | None = None
    phase: SpeciesPhase | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_empty(self.id, "reaction id"))
        object.__setattr__(
            self, "equation", _non_empty(self.equation, "reaction equation")
        )
        if not isinstance(self.reversible, bool):
            raise MechanismValidationError(
                f"reaction {self.id} reversible must be boolean"
            )
        if self.phase is not None:
            try:
                phase = (
                    self.phase
                    if isinstance(self.phase, SpeciesPhase)
                    else SpeciesPhase(str(self.phase).strip().lower())
                )
            except ValueError as exc:
                raise MechanismValidationError(
                    f"reaction {self.id} has unsupported phase {self.phase!r}"
                ) from exc
            object.__setattr__(self, "phase", phase)

        reactants = self._validate_positive_map(self.reactants, "reactant coefficient")
        products = self._validate_positive_map(self.products, "product coefficient")
        if not reactants or not products:
            raise MechanismValidationError(
                f"reaction {self.id} must have reactants and products"
            )
        object.__setattr__(self, "reactants", reactants)
        object.__setattr__(self, "products", products)

        forward_orders = self._validate_order_map(
            self.forward_orders or reactants,
            "forward order",
        )
        reverse_orders = self._validate_order_map(
            self.reverse_orders or (products if self.reversible else {}),
            "reverse order",
        )
        object.__setattr__(self, "forward_orders", forward_orders)
        object.__setattr__(self, "reverse_orders", reverse_orders)

        if not self.reversible and self.reverse is not None:
            raise MechanismValidationError(
                f"irreversible reaction {self.id} cannot define reverse kinetics"
            )
        if not self.reversible and reverse_orders:
            raise MechanismValidationError(
                f"irreversible reaction {self.id} cannot define reverse orders"
            )
        if self.reversible and self.reverse is None:
            thermo = self.thermodynamics
            if thermo is None or thermo.delta_s_j_per_mol_k is None:
                raise MechanismValidationError(
                    f"reversible reaction {self.id} requires reverse Arrhenius "
                    "parameters or both dH and dS"
                )
            forward_order = sum(forward_orders.values())
            reverse_order = sum(reverse_orders.values())
            if not math.isclose(
                forward_order,
                reverse_order,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise MechanismValidationError(
                    f"reversible reaction {self.id} cannot derive reverse kinetics "
                    "from a dimensionless equilibrium constant when forward and "
                    "reverse total orders differ"
                )

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @staticmethod
    def _validate_positive_map(
        values: Mapping[str, float],
        label: str,
    ) -> Mapping[str, float]:
        normalized: dict[str, float] = {}
        for species, value in values.items():
            name = _non_empty(species, "stoichiometric species")
            numeric = _finite(value, f"{label} for {name}")
            if numeric <= 0.0:
                raise MechanismValidationError(f"{label} for {name} must be positive")
            normalized[name] = numeric
        return MappingProxyType(normalized)

    @staticmethod
    def _validate_order_map(
        values: Mapping[str, float],
        label: str,
    ) -> Mapping[str, float]:
        normalized: dict[str, float] = {}
        for species, value in values.items():
            name = _non_empty(species, "reaction-order species")
            normalized[name] = _finite(value, f"{label} for {name}")
        return MappingProxyType(normalized)

    @property
    def net_stoichiometry(self) -> Mapping[str, float]:
        """Signed product-positive stoichiometry."""
        net: dict[str, float] = {}
        for species, coefficient in self.reactants.items():
            net[species] = net.get(species, 0.0) - coefficient
        for species, coefficient in self.products.items():
            net[species] = net.get(species, 0.0) + coefficient
        return MappingProxyType(net)


@dataclass(frozen=True, slots=True)
class Mechanism:
    """Complete standalone homogeneous chemistry mechanism."""

    name: str
    species: tuple[Species, ...]
    reactions: tuple[Reaction, ...]
    version: str = "1.0"
    description: str | None = None
    temperature_range_k: tuple[float, float] | None = None
    pressure_range_pa: tuple[float, float] | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "mechanism name"))
        object.__setattr__(
            self, "version", _non_empty(str(self.version), "mechanism version")
        )
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reactions", tuple(self.reactions))
        if not self.species:
            raise MechanismValidationError("mechanism must define at least one species")
        if not self.reactions:
            raise MechanismValidationError(
                "mechanism must define at least one reaction"
            )

        species_names = [item.name for item in self.species]
        reaction_ids = [item.id for item in self.reactions]
        self._require_unique(species_names, "species name")
        self._require_unique(reaction_ids, "reaction id")
        known_species = set(species_names)
        for reaction in self.reactions:
            referenced = (
                set(reaction.net_stoichiometry)
                | set(reaction.forward_orders)
                | set(reaction.reverse_orders)
            )
            unknown = sorted(referenced - known_species)
            if unknown:
                raise MechanismValidationError(
                    f"reaction {reaction.id} references unknown species: {unknown}"
                )

        object.__setattr__(
            self,
            "temperature_range_k",
            self._validate_range(self.temperature_range_k, "temperature range"),
        )
        object.__setattr__(
            self,
            "pressure_range_pa",
            self._validate_range(self.pressure_range_pa, "pressure range"),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @staticmethod
    def _require_unique(values: list[str], label: str) -> None:
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise MechanismValidationError(f"duplicate {label}s: {duplicates}")

    @staticmethod
    def _validate_range(
        value: tuple[float, float] | None,
        label: str,
    ) -> tuple[float, float] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise MechanismValidationError(f"{label} must contain two values")
        lower = _finite(value[0], f"{label} lower bound")
        upper = _finite(value[1], f"{label} upper bound")
        if lower <= 0.0 or upper <= lower:
            raise MechanismValidationError(f"{label} must satisfy 0 < lower < upper")
        return (lower, upper)

    @property
    def species_names(self) -> tuple[str, ...]:
        """Species names in deterministic state-vector order."""
        return tuple(item.name for item in self.species)

    @property
    def reaction_ids(self) -> tuple[str, ...]:
        """Reaction IDs in deterministic rate-vector order."""
        return tuple(item.id for item in self.reactions)
