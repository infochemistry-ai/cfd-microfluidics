"""YAML/JSON mechanism loading and SI normalization."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Mapping

import yaml

from microfluidics.chemistry.errors import MechanismValidationError
from microfluidics.chemistry.models import (
    ArrheniusParameters,
    Mechanism,
    Reaction,
    ReactionThermodynamics,
    Species,
    SpeciesPhase,
)

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_QUANTITY_PATTERN = re.compile(rf"^\s*({_NUMBER})\s*(.*?)\s*$")
_SPACED_COEFFICIENT_PATTERN = re.compile(rf"^({_NUMBER})\s+(?:\*\s*)?(.+)$")
_COMPACT_COEFFICIENT_PATTERN = re.compile(rf"^({_NUMBER})(?:\*\s*)?([A-Za-z(*].*)$")

_ENERGY_FACTORS = {
    "j/mol": 1.0,
    "kj/mol": 1000.0,
    "cal/mol": 4.184,
    "kcal/mol": 4184.0,
}
_ENTROPY_FACTORS = {
    "j/(mol*k)": 1.0,
    "j/mol/k": 1.0,
    "kj/(mol*k)": 1000.0,
    "kj/mol/k": 1000.0,
}
_PRE_EXPONENTIAL_UNITS = {
    "1/s": (1.0, 1.0),
    "s^-1": (1.0, 1.0),
    "s-1": (1.0, 1.0),
    "m^3/(mol*s)": (1.0, 2.0),
    "m3/(mol*s)": (1.0, 2.0),
    "m^3/mol/s": (1.0, 2.0),
    "m3/mol/s": (1.0, 2.0),
    "l/(mol*s)": (1e-3, 2.0),
    "l/mol/s": (1e-3, 2.0),
    "m^6/(mol^2*s)": (1.0, 3.0),
    "m6/(mol2*s)": (1.0, 3.0),
    "m^6/mol^2/s": (1.0, 3.0),
    "l^2/(mol^2*s)": (1e-6, 3.0),
    "l2/(mol2*s)": (1e-6, 3.0),
    "l^2/mol^2/s": (1e-6, 3.0),
}
_TOP_LEVEL_KEYS = frozenset(
    {"version", "name", "metadata", "units", "species", "reactions", "extensions"}
)
_METADATA_KEYS = frozenset(
    {
        "name",
        "version",
        "description",
        "mechanism_family",
        "temperature_range",
        "temperature_range_k",
        "pressure_range",
        "pressure_range_pa",
        "activation_energy_unit",
        "provenance",
        "extensions",
    }
)
_UNITS_KEYS = frozenset({"activation-energy", "extensions"})
_SPECIES_KEYS = frozenset(
    {
        "name",
        "formula",
        "phase",
        "molecular_weight",
        "molar_mass_kg_per_mol",
        "composition",
        "aliases",
        "extensions",
    }
)
_REACTION_KEYS = frozenset(
    {
        "id",
        "equation",
        "reversible",
        "kinetics",
        "thermodynamics",
        "phase",
        "notes",
        "extensions",
    }
)
_KINETICS_KEYS = frozenset(
    {
        "A",
        "Ea",
        "n",
        "T_ref",
        "A_forward",
        "Ea_forward",
        "n_forward",
        "T_ref_forward",
        "A_reverse",
        "Ea_reverse",
        "n_reverse",
        "T_ref_reverse",
        "orders",
        "reverse_orders",
        "extensions",
    }
)
_THERMODYNAMICS_KEYS = frozenset({"dH", "delta_h", "dS", "delta_s", "extensions"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_mechanism(path: str | Path) -> Mechanism:
    """Load one versioned YAML or JSON mechanism file."""
    source = Path(path)
    if not source.is_file():
        raise MechanismValidationError(f"mechanism file does not exist: {source}")
    try:
        payload = yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, yaml.YAMLError) as exc:
        raise MechanismValidationError(
            f"cannot read mechanism {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise MechanismValidationError("mechanism document must contain an object")
    return mechanism_from_mapping(payload, source_label=str(source))


def mechanism_from_mapping(
    payload: Mapping[str, object],
    *,
    source_label: str = "mechanism payload",
) -> Mechanism:
    """Normalize a mechanism mapping into standalone chemistry contracts."""
    _validate_keys(
        payload,
        source_label,
        allowed=_TOP_LEVEL_KEYS,
        required=frozenset({"species", "reactions"}),
    )
    raw_schema_version = payload.get("version", 1)
    if str(raw_schema_version).split(".", 1)[0] != "1":
        raise MechanismValidationError(
            f"{source_label} uses unsupported mechanism version "
            f"{raw_schema_version!r} for the document schema"
        )

    metadata = _mapping(payload.get("metadata", {}), "metadata")
    units = _mapping(payload.get("units", {}), "units")
    _validate_keys(metadata, "metadata", allowed=_METADATA_KEYS)
    _validate_keys(units, "units", allowed=_UNITS_KEYS)
    _reject_alias_conflicts(
        metadata,
        "metadata",
        ("temperature_range", "temperature_range_k"),
        ("pressure_range", "pressure_range_pa"),
    )
    if "name" in payload and "name" in metadata:
        raise MechanismValidationError(
            f"{source_label} contains mutually exclusive mechanism name fields: "
            "'name' and 'metadata.name'"
        )
    if "activation-energy" in units and "activation_energy_unit" in metadata:
        raise MechanismValidationError(
            "mechanism contains mutually exclusive activation-energy unit fields: "
            "'units.activation-energy' and 'metadata.activation_energy_unit'"
        )
    activation_energy_unit = _normalize_unit(
        units.get("activation-energy", metadata.get("activation_energy_unit", "J/mol"))
    )
    if activation_energy_unit not in _ENERGY_FACTORS:
        allowed = ", ".join(sorted(_ENERGY_FACTORS))
        raise MechanismValidationError(
            f"unsupported activation-energy unit {activation_energy_unit!r}; "
            f"supported units: {allowed}"
        )

    raw_species = _sequence(payload.get("species"), "species")
    species = tuple(
        _parse_species(item, index) for index, item in enumerate(raw_species)
    )
    alias_map = _build_species_alias_map(species)

    raw_reactions = _sequence(payload.get("reactions"), "reactions")
    reactions = tuple(
        _parse_reaction(
            item,
            index,
            alias_map=alias_map,
            activation_energy_unit=activation_energy_unit,
        )
        for index, item in enumerate(raw_reactions)
    )

    name = _required_string(
        metadata.get("name", payload.get("name", "standalone chemistry")),
        "mechanism name",
    )
    mechanism_version = str(metadata.get("version", raw_schema_version)).strip()
    if not mechanism_version:
        raise MechanismValidationError("metadata.version must not be empty")
    description_raw = metadata.get("description")
    temperature_range = _optional_range(
        metadata.get("temperature_range", metadata.get("temperature_range_k")),
        "metadata.temperature_range",
    )
    if metadata.get("pressure_range_pa") is not None:
        pressure_range = _optional_range(
            metadata.get("pressure_range_pa"),
            "metadata.pressure_range_pa",
        )
    else:
        # Schema v1 authors pressure_range in MPa while the runtime contract
        # remains canonical Pa.
        pressure_range_mpa = _optional_range(
            metadata.get("pressure_range"),
            "metadata.pressure_range",
        )
        pressure_range = (
            tuple(value * 1e6 for value in pressure_range_mpa)
            if pressure_range_mpa is not None
            else None
        )
    extra_metadata = dict(metadata)
    extra_metadata["source"] = source_label
    extra_metadata["activation_energy_unit_authored"] = activation_energy_unit

    return Mechanism(
        name=name,
        # Top-level ``version`` selects the supported document schema.
        # ``metadata.version`` is the independent authored mechanism revision.
        version=mechanism_version,
        description=str(description_raw) if description_raw is not None else None,
        temperature_range_k=temperature_range,
        pressure_range_pa=pressure_range,
        species=species,
        reactions=reactions,
        metadata=extra_metadata,
    )


def _parse_species(raw: object, index: int) -> Species:
    entry = _mapping(raw, f"species[{index}]")
    _validate_keys(
        entry,
        f"species[{index}]",
        allowed=_SPECIES_KEYS,
        required=frozenset({"name"}),
    )
    _reject_alias_conflicts(
        entry,
        f"species[{index}]",
        ("molecular_weight", "molar_mass_kg_per_mol"),
    )
    name = _required_string(entry.get("name"), f"species[{index}].name")
    formula = _required_string(entry.get("formula", name), f"species[{index}].formula")
    phase_raw = str(entry.get("phase", "unknown")).strip().lower()
    try:
        phase = SpeciesPhase(phase_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SpeciesPhase)
        raise MechanismValidationError(
            f"species {name} has unsupported phase {phase_raw!r}; allowed: {allowed}"
        ) from exc

    if "molar_mass_kg_per_mol" in entry:
        molar_mass = _number(
            entry["molar_mass_kg_per_mol"],
            f"species {name} molar_mass_kg_per_mol",
        )
    else:
        molecular_weight = _number(
            entry.get("molecular_weight"),
            f"species {name} molecular_weight",
        )
        molar_mass = molecular_weight / 1000.0

    composition_raw = _mapping(
        entry.get("composition", {}), f"species {name} composition"
    )
    composition = {
        str(element): _number(count, f"species {name} composition[{element}]")
        for element, count in composition_raw.items()
    }
    aliases_raw = entry.get("aliases", ())
    aliases = tuple(
        str(item).strip() for item in _sequence(aliases_raw, f"species {name} aliases")
    )
    return Species(
        name=name,
        formula=formula,
        phase=phase,
        molar_mass_kg_per_mol=molar_mass,
        elemental_composition=composition,
        aliases=aliases,
        metadata=dict(entry),
    )


def _build_species_alias_map(species: tuple[Species, ...]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    formulas = Counter(item.formula for item in species)
    for item in species:
        candidates = [item.name, *item.aliases]
        if formulas[item.formula] == 1:
            candidates.append(item.formula)
        for alias in candidates:
            previous = alias_map.get(alias)
            if previous is not None and previous != item.name:
                raise MechanismValidationError(
                    f"species alias {alias!r} is ambiguous between {previous!r} "
                    f"and {item.name!r}"
                )
            alias_map[alias] = item.name
    return alias_map


def _parse_reaction(
    raw: object,
    index: int,
    *,
    alias_map: Mapping[str, str],
    activation_energy_unit: str,
) -> Reaction:
    entry = _mapping(raw, f"reactions[{index}]")
    _validate_keys(
        entry,
        f"reactions[{index}]",
        allowed=_REACTION_KEYS,
        required=frozenset({"id", "equation", "kinetics"}),
    )
    reaction_id = _required_string(entry.get("id"), f"reactions[{index}].id")
    equation = _required_string(
        entry.get("equation"), f"reaction {reaction_id} equation"
    )
    reactants, products, arrow_reversible = _parse_equation(equation, alias_map)

    reversible_raw = entry.get("reversible")
    reversible = (
        arrow_reversible
        if reversible_raw is None
        else _boolean(
            reversible_raw,
            f"reaction {reaction_id} reversible",
        )
    )
    if reversible != arrow_reversible:
        raise MechanismValidationError(
            f"reaction {reaction_id} reversible flag disagrees with equation arrow"
        )

    kinetics = _mapping(entry.get("kinetics"), f"reaction {reaction_id} kinetics")
    _validate_keys(
        kinetics,
        f"reaction {reaction_id} kinetics",
        allowed=_KINETICS_KEYS,
    )
    _reject_alias_conflicts(
        kinetics,
        f"reaction {reaction_id} kinetics",
        ("A", "A_forward"),
        ("Ea", "Ea_forward"),
    )
    _reject_fallback_directional_conflict(
        kinetics,
        f"reaction {reaction_id} kinetics",
        fallback="n",
        directional=("n_forward", "n_reverse"),
    )
    _reject_fallback_directional_conflict(
        kinetics,
        f"reaction {reaction_id} kinetics",
        fallback="T_ref",
        directional=("T_ref_forward", "T_ref_reverse"),
    )
    forward_orders = _parse_order_map(
        kinetics.get("orders", reactants),
        alias_map,
        f"reaction {reaction_id} orders",
    )
    reverse_orders = _parse_order_map(
        kinetics.get("reverse_orders", products if reversible else {}),
        alias_map,
        f"reaction {reaction_id} reverse_orders",
    )

    forward_a = kinetics.get("A_forward", kinetics.get("A"))
    forward_ea = kinetics.get("Ea_forward", kinetics.get("Ea"))
    if forward_a is None or forward_ea is None:
        raise MechanismValidationError(
            f"reaction {reaction_id} requires A/Ea or A_forward/Ea_forward"
        )
    forward = ArrheniusParameters(
        pre_exponential=_pre_exponential(
            forward_a,
            total_order=sum(forward_orders.values()),
            label=f"reaction {reaction_id} forward A",
        ),
        activation_energy_j_per_mol=_energy(
            forward_ea,
            activation_energy_unit,
            f"reaction {reaction_id} forward Ea",
        ),
        temperature_exponent=_number(
            kinetics.get("n_forward", kinetics.get("n", 0.0)),
            f"reaction {reaction_id} forward n",
        ),
        reference_temperature_k=_optional_number(
            kinetics.get("T_ref_forward", kinetics.get("T_ref")),
            f"reaction {reaction_id} forward T_ref",
        ),
    )

    reverse = None
    reverse_a = kinetics.get("A_reverse")
    reverse_ea = kinetics.get("Ea_reverse")
    if reverse_a is not None or reverse_ea is not None:
        if reverse_a is None or reverse_ea is None:
            raise MechanismValidationError(
                f"reaction {reaction_id} reverse kinetics require both A_reverse and Ea_reverse"
            )
        reverse = ArrheniusParameters(
            pre_exponential=_pre_exponential(
                reverse_a,
                total_order=sum(reverse_orders.values()),
                label=f"reaction {reaction_id} reverse A",
            ),
            activation_energy_j_per_mol=_energy(
                reverse_ea,
                activation_energy_unit,
                f"reaction {reaction_id} reverse Ea",
            ),
            temperature_exponent=_number(
                kinetics.get("n_reverse", kinetics.get("n", 0.0)),
                f"reaction {reaction_id} reverse n",
            ),
            reference_temperature_k=_optional_number(
                kinetics.get("T_ref_reverse", kinetics.get("T_ref")),
                f"reaction {reaction_id} reverse T_ref",
            ),
        )

    thermodynamics = _parse_thermodynamics(entry.get("thermodynamics"), reaction_id)
    phase_raw = entry.get("phase")
    try:
        phase = (
            SpeciesPhase(str(phase_raw).strip().lower())
            if phase_raw is not None
            else None
        )
    except ValueError as exc:
        raise MechanismValidationError(
            f"reaction {reaction_id} has unsupported phase {phase_raw!r}"
        ) from exc
    return Reaction(
        id=reaction_id,
        equation=equation,
        reactants=reactants,
        products=products,
        reversible=reversible,
        forward=forward,
        reverse=reverse,
        forward_orders=forward_orders,
        reverse_orders=reverse_orders,
        thermodynamics=thermodynamics,
        phase=phase,
        metadata=dict(entry),
    )


def _parse_thermodynamics(
    raw: object, reaction_id: str
) -> ReactionThermodynamics | None:
    if raw is None:
        return None
    entry = _mapping(raw, f"reaction {reaction_id} thermodynamics")
    _validate_keys(
        entry,
        f"reaction {reaction_id} thermodynamics",
        allowed=_THERMODYNAMICS_KEYS,
    )
    _reject_alias_conflicts(
        entry,
        f"reaction {reaction_id} thermodynamics",
        ("dH", "delta_h"),
        ("dS", "delta_s"),
    )
    delta_h_raw = entry.get("dH", entry.get("delta_h"))
    if delta_h_raw is None:
        raise MechanismValidationError(
            f"reaction {reaction_id} thermodynamics requires dH"
        )
    delta_s_raw = entry.get("dS", entry.get("delta_s"))
    return ReactionThermodynamics(
        delta_h_j_per_mol=_energy(
            delta_h_raw,
            "j/mol",
            f"reaction {reaction_id} dH",
        ),
        delta_s_j_per_mol_k=(
            _entropy(delta_s_raw, f"reaction {reaction_id} dS")
            if delta_s_raw is not None
            else None
        ),
    )


def _parse_equation(
    equation: str,
    alias_map: Mapping[str, str],
) -> tuple[dict[str, float], dict[str, float], bool]:
    # Check reversible arrows first because ``->`` is a substring of ``<->``.
    if "<=>" in equation:
        separator = "<=>"
    elif "<->" in equation:
        separator = "<->"
    elif "->" in equation:
        separator = "->"
    else:
        raise MechanismValidationError(
            f"reaction equation must contain exactly one ->, <=> or <->: {equation!r}"
        )
    if equation.count(separator) != 1:
        raise MechanismValidationError(
            f"reaction equation must contain exactly one ->, <=> or <->: {equation!r}"
        )
    parts = equation.split(separator)
    if len(parts) != 2:
        raise MechanismValidationError(f"invalid reaction equation: {equation!r}")
    reactants = _parse_equation_side(parts[0], alias_map, equation)
    products = _parse_equation_side(parts[1], alias_map, equation)
    return reactants, products, separator != "->"


def _parse_equation_side(
    side: str,
    alias_map: Mapping[str, str],
    equation: str,
) -> dict[str, float]:
    stripped = side.strip()
    if not stripped:
        raise MechanismValidationError(f"empty side in reaction equation {equation!r}")
    if stripped in alias_map:
        raw_terms = [stripped]
    elif re.search(r"\s+\+\s+", stripped):
        raw_terms = re.split(r"\s+\+\s+(?![^()]*\))", stripped)
    else:
        raw_terms = re.split(r"\s*\+\s*(?![^()]*\))", stripped)

    parsed: dict[str, float] = {}
    for raw_term in raw_terms:
        coefficient, token = _parse_term(raw_term.strip())
        canonical = alias_map.get(token)
        if canonical is None:
            known = ", ".join(sorted(alias_map))
            raise MechanismValidationError(
                f"unknown species token {token!r} in {equation!r}; known aliases: {known}"
            )
        parsed[canonical] = parsed.get(canonical, 0.0) + coefficient
    return parsed


def _parse_term(term: str) -> tuple[float, str]:
    spaced = _SPACED_COEFFICIENT_PATTERN.match(term)
    if spaced:
        return float(spaced.group(1)), spaced.group(2).strip()
    compact = _COMPACT_COEFFICIENT_PATTERN.match(term)
    if compact:
        return float(compact.group(1)), compact.group(2).strip()
    return 1.0, term


def _parse_order_map(
    raw: object,
    alias_map: Mapping[str, str],
    label: str,
) -> dict[str, float]:
    entry = _mapping(raw, label)
    normalized: dict[str, float] = {}
    for alias, value in entry.items():
        canonical = alias_map.get(str(alias))
        if canonical is None:
            raise MechanismValidationError(
                f"{label} references unknown species {alias!r}"
            )
        normalized[canonical] = _number(value, f"{label}[{alias}]")
    return normalized


def _pre_exponential(raw: object, *, total_order: float, label: str) -> float:
    if isinstance(raw, bool):
        raise MechanismValidationError(f"{label} must be numeric")
    if isinstance(raw, int | float):
        return _number(raw, label)
    if not isinstance(raw, str):
        raise MechanismValidationError(f"{label} must be numeric or a unit literal")
    match = _QUANTITY_PATTERN.match(raw)
    if match is None:
        raise MechanismValidationError(f"invalid {label} literal: {raw!r}")
    value = float(match.group(1))
    unit = _normalize_rate_unit(match.group(2))
    if not unit:
        return value
    resolved = _PRE_EXPONENTIAL_UNITS.get(unit)
    if resolved is None:
        allowed = ", ".join(sorted(_PRE_EXPONENTIAL_UNITS))
        raise MechanismValidationError(
            f"unsupported {label} unit {match.group(2)!r}; supported units: {allowed}"
        )
    factor, unit_order = resolved
    if not math.isclose(total_order, unit_order, rel_tol=0.0, abs_tol=1e-12):
        raise MechanismValidationError(
            f"{label} unit implies reaction order {unit_order:g}, "
            f"but authored orders sum to {total_order:g}"
        )
    return value * factor


def _energy(raw: object, fallback_unit: str, label: str) -> float:
    value, explicit_unit = _quantity(raw, label)
    unit = _normalize_unit(explicit_unit or fallback_unit)
    factor = _ENERGY_FACTORS.get(unit)
    if factor is None:
        raise MechanismValidationError(f"unsupported energy unit {unit!r} for {label}")
    return value * factor


def _entropy(raw: object, label: str) -> float:
    value, explicit_unit = _quantity(raw, label)
    unit = _normalize_entropy_unit(explicit_unit or "J/(mol*K)")
    factor = _ENTROPY_FACTORS.get(unit)
    if factor is None:
        raise MechanismValidationError(f"unsupported entropy unit {unit!r} for {label}")
    return value * factor


def _quantity(raw: object, label: str) -> tuple[float, str | None]:
    if isinstance(raw, bool):
        raise MechanismValidationError(f"{label} must be numeric")
    if isinstance(raw, int | float):
        return _number(raw, label), None
    if not isinstance(raw, str):
        raise MechanismValidationError(f"{label} must be numeric or a unit literal")
    match = _QUANTITY_PATTERN.match(raw)
    if match is None:
        raise MechanismValidationError(f"invalid {label} literal: {raw!r}")
    return float(match.group(1)), match.group(2) or None


def _mapping(raw: object, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise MechanismValidationError(f"{label} must be an object")
    return raw


def _validate_keys(
    entry: Mapping[str, object],
    label: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> None:
    keys = set(entry)
    unknown = keys - allowed
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise MechanismValidationError(f"{label} contains unknown fields: {rendered}")
    missing = required - keys
    if missing:
        rendered = ", ".join(sorted(missing))
        raise MechanismValidationError(
            f"{label} is missing required fields: {rendered}"
        )
    if "extensions" in entry:
        _mapping(entry["extensions"], f"{label}.extensions")


def _reject_alias_conflicts(
    entry: Mapping[str, object],
    label: str,
    *alias_groups: tuple[str, ...],
) -> None:
    for aliases in alias_groups:
        present = [name for name in aliases if name in entry]
        if len(present) > 1:
            rendered = ", ".join(repr(name) for name in present)
            raise MechanismValidationError(
                f"{label} contains mutually exclusive alias fields: {rendered}"
            )


def _reject_fallback_directional_conflict(
    entry: Mapping[str, object],
    label: str,
    *,
    fallback: str,
    directional: tuple[str, ...],
) -> None:
    present_directional = [name for name in directional if name in entry]
    if fallback in entry and present_directional:
        rendered = ", ".join(repr(name) for name in (fallback, *present_directional))
        raise MechanismValidationError(
            f"{label} contains mutually exclusive fallback/directional fields: "
            f"{rendered}"
        )


def _sequence(raw: object, label: str) -> list[object] | tuple[object, ...]:
    if not isinstance(raw, list | tuple):
        raise MechanismValidationError(f"{label} must be an array")
    return raw


def _required_string(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise MechanismValidationError(f"{label} must be a non-empty string")
    return raw.strip()


def _number(raw: object, label: str) -> float:
    if isinstance(raw, bool):
        raise MechanismValidationError(f"{label} must be numeric")
    if isinstance(raw, str) and re.fullmatch(_NUMBER, raw.strip()):
        value = float(raw)
    elif isinstance(raw, int | float):
        value = float(raw)
    else:
        raise MechanismValidationError(f"{label} must be numeric")
    if not math.isfinite(value):
        raise MechanismValidationError(f"{label} must be finite")
    return value


def _optional_number(raw: object, label: str) -> float | None:
    return None if raw is None else _number(raw, label)


def _boolean(raw: object, label: str) -> bool:
    if not isinstance(raw, bool):
        raise MechanismValidationError(f"{label} must be boolean")
    return raw


def _optional_range(raw: object, label: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    values = _sequence(raw, label)
    if len(values) != 2:
        raise MechanismValidationError(f"{label} must contain two values")
    return (_number(values[0], label), _number(values[1], label))


def _normalize_unit(raw: object) -> str:
    return str(raw).strip().lower().replace(" ", "").replace("·", "*")


def _normalize_entropy_unit(raw: object) -> str:
    return _normalize_unit(raw).replace("°", "")


def _normalize_rate_unit(raw: object) -> str:
    return _normalize_unit(raw).replace("³", "^3").replace("²", "^2")
