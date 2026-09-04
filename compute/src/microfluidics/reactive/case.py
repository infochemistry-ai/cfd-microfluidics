"""Strict immutable reactive-case JSON v1 contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from microfluidics.chemistry import (
    CompiledChemistry,
    Mechanism,
    SpeciesPhase,
    compile_mechanism,
    mechanism_from_mapping,
)
from microfluidics.reactive.errors import ReactiveCaseValidationError

REACTIVE_CASE_SCHEMA_VERSION = 1
REACTIVE_CASE_CONTRACT_VERSION = "reactive_case_v1"
REACTIVE_TRANSPORT_CONTRACT_VERSION = "reactive_transport_v1"
REACTIVE_MODES = frozenset({"off", "isothermal", "nonisothermal"})


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReactiveCaseValidationError(
                f"reactive case contains duplicate JSON field {key!r}"
            )
        result[key] = value
    return result


def normalize_group_name(value: str) -> str:
    """Normalize physical-group spelling without geometric fallbacks."""

    normalized = str(value).strip().lower()
    for token in (" ", "-", ".", "/"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _object(
    raw: object,
    label: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ReactiveCaseValidationError(f"{label} must be a JSON object")
    values = {str(key): value for key, value in raw.items()}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ReactiveCaseValidationError(f"{label} contains unknown fields: {unknown}")
    missing = sorted(required - set(values))
    if missing:
        raise ReactiveCaseValidationError(f"{label} is missing fields: {missing}")
    return values


def _number(raw: object, label: str, *, positive: bool = False) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ReactiveCaseValidationError(f"{label} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ReactiveCaseValidationError(f"{label} must be finite")
    if positive and value <= 0.0:
        raise ReactiveCaseValidationError(f"{label} must be positive")
    return value


def _integer(raw: object, label: str, *, positive: bool = False) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ReactiveCaseValidationError(f"{label} must be an integer")
    value = int(raw)
    if positive and value <= 0:
        raise ReactiveCaseValidationError(f"{label} must be positive")
    return value


def _concentrations(
    raw: object,
    label: str,
    species_names: tuple[str, ...],
) -> tuple[float, ...]:
    if not isinstance(raw, Mapping):
        raise ReactiveCaseValidationError(f"{label} must be a JSON object")
    authored = {str(key): value for key, value in raw.items()}
    unknown = sorted(set(authored) - set(species_names))
    if unknown:
        raise ReactiveCaseValidationError(
            f"{label} contains unknown species: {unknown}"
        )
    values: list[float] = []
    for name in species_names:
        value = _number(authored.get(name, 0.0), f"{label}.{name}")
        if value < 0.0:
            raise ReactiveCaseValidationError(f"{label}.{name} must be non-negative")
        values.append(value)
    return tuple(values)


def _validate_state_range(
    mechanism: Mechanism,
    *,
    temperature_k: float,
    pressure_pa: float,
    label: str,
) -> None:
    if mechanism.temperature_range_k is not None:
        lower, upper = mechanism.temperature_range_k
        if not lower <= temperature_k <= upper:
            raise ReactiveCaseValidationError(
                f"{label}.temperature_k={temperature_k} is outside authored "
                f"range [{lower}, {upper}]"
            )
    if mechanism.pressure_range_pa is not None:
        lower, upper = mechanism.pressure_range_pa
        if not lower <= pressure_pa <= upper:
            raise ReactiveCaseValidationError(
                f"{label}.operating_pressure_pa={pressure_pa} is outside authored "
                f"range [{lower}, {upper}]"
            )


@dataclass(frozen=True, slots=True)
class ReactiveStateV1:
    temperature_k: float
    operating_pressure_pa: float
    concentrations_mol_per_m3: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReactiveInletV1:
    name: str
    normalized_name: str
    temperature_k: float
    concentrations_mol_per_m3: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReactiveMaterialV1:
    density_kg_per_m3: float
    heat_capacity_j_per_kg_k: float
    thermal_diffusivity_m2_s: float
    species_diffusivity_m2_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReactiveTimeV1:
    num_steps: int
    dt_mode: str
    dt_s: float
    max_dt_s: float
    cfl_target: float
    diffusion_stability_factor: float
    max_transport_substeps: int


@dataclass(frozen=True, slots=True)
class ChemistryIntegratorV1:
    relative_tolerance: float
    concentration_absolute_tolerance_mol_per_m3: float
    temperature_absolute_tolerance_k: float
    max_temperature_change_per_substep_k: float
    min_substep_fraction: float
    max_substeps_per_half_step: int
    cell_batch_size: int


@dataclass(frozen=True, slots=True)
class ReactiveOutputV1:
    history_stride: int
    snapshot_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ReactiveCaseV1:
    schema_version: int
    case_id: str
    mode: str
    mechanism: Mechanism
    compiled_chemistry: CompiledChemistry
    mechanism_payload_json: str
    initial_state: ReactiveStateV1
    inlets: tuple[ReactiveInletV1, ...]
    material: ReactiveMaterialV1
    time: ReactiveTimeV1
    chemistry_integrator: ChemistryIntegratorV1
    output: ReactiveOutputV1

    @property
    def species_names(self) -> tuple[str, ...]:
        return self.compiled_chemistry.species_names

    @property
    def mechanism_sha256(self) -> str:
        return self.compiled_chemistry.mechanism_sha256

    @property
    def normalized_payload(self) -> dict[str, object]:
        species = self.species_names
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "mode": self.mode,
            "mechanism": json.loads(self.mechanism_payload_json),
            "initial_state": {
                "temperature_k": self.initial_state.temperature_k,
                "operating_pressure_pa": self.initial_state.operating_pressure_pa,
                "concentrations_mol_per_m3": dict(
                    zip(
                        species,
                        self.initial_state.concentrations_mol_per_m3,
                        strict=True,
                    )
                ),
            },
            "inlets": {
                inlet.name: {
                    "temperature_k": inlet.temperature_k,
                    "concentrations_mol_per_m3": dict(
                        zip(
                            species,
                            inlet.concentrations_mol_per_m3,
                            strict=True,
                        )
                    ),
                }
                for inlet in self.inlets
            },
            "material": {
                "density_kg_per_m3": self.material.density_kg_per_m3,
                "heat_capacity_j_per_kg_k": (self.material.heat_capacity_j_per_kg_k),
                "thermal_diffusivity_m2_s": (self.material.thermal_diffusivity_m2_s),
                "species_diffusivity_m2_s": dict(
                    zip(
                        species,
                        self.material.species_diffusivity_m2_s,
                        strict=True,
                    )
                ),
            },
            "time": {
                "num_steps": self.time.num_steps,
                "dt_mode": self.time.dt_mode,
                "dt_s": self.time.dt_s,
                "max_dt_s": self.time.max_dt_s,
                "cfl_target": self.time.cfl_target,
                "diffusion_stability_factor": (self.time.diffusion_stability_factor),
                "max_transport_substeps": self.time.max_transport_substeps,
            },
            "chemistry_integrator": {
                "relative_tolerance": self.chemistry_integrator.relative_tolerance,
                "concentration_absolute_tolerance_mol_per_m3": (
                    self.chemistry_integrator.concentration_absolute_tolerance_mol_per_m3
                ),
                "temperature_absolute_tolerance_k": (
                    self.chemistry_integrator.temperature_absolute_tolerance_k
                ),
                "max_temperature_change_per_substep_k": (
                    self.chemistry_integrator.max_temperature_change_per_substep_k
                ),
                "min_substep_fraction": (
                    self.chemistry_integrator.min_substep_fraction
                ),
                "max_substeps_per_half_step": (
                    self.chemistry_integrator.max_substeps_per_half_step
                ),
                "cell_batch_size": self.chemistry_integrator.cell_batch_size,
            },
            "output": {
                "history_stride": self.output.history_stride,
                "snapshot_steps": list(self.output.snapshot_steps),
            },
        }

    @property
    def reactive_case_sha256(self) -> str:
        encoded = json.dumps(
            self.normalized_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def reactive_case_from_mapping(
    raw: Mapping[str, object],
    *,
    source_label: str = "reactive case",
) -> ReactiveCaseV1:
    """Validate and normalize one reactive-case JSON mapping."""

    top = _object(
        raw,
        source_label,
        allowed=frozenset(
            {
                "schema_version",
                "case_id",
                "mode",
                "mechanism",
                "initial_state",
                "inlets",
                "material",
                "time",
                "chemistry_integrator",
                "output",
            }
        ),
        required=frozenset(
            {
                "schema_version",
                "case_id",
                "mode",
                "mechanism",
                "initial_state",
                "inlets",
                "material",
                "time",
                "chemistry_integrator",
                "output",
            }
        ),
    )
    schema_version = _integer(top["schema_version"], "schema_version")
    if schema_version != REACTIVE_CASE_SCHEMA_VERSION:
        raise ReactiveCaseValidationError(
            f"schema_version must equal {REACTIVE_CASE_SCHEMA_VERSION}"
        )
    if not isinstance(top["case_id"], str) or not top["case_id"].strip():
        raise ReactiveCaseValidationError("case_id must be a non-empty string")
    case_id = top["case_id"].strip()
    if not isinstance(top["mode"], str):
        raise ReactiveCaseValidationError("mode must be a string")
    mode = top["mode"].strip().lower()
    if mode not in REACTIVE_MODES:
        raise ReactiveCaseValidationError(
            f"mode must be one of {sorted(REACTIVE_MODES)}"
        )

    mechanism_raw = top["mechanism"]
    if not isinstance(mechanism_raw, Mapping):
        raise ReactiveCaseValidationError("mechanism must be a JSON object")
    mechanism = mechanism_from_mapping(
        mechanism_raw, source_label=f"{source_label}.mechanism"
    )
    try:
        mechanism_payload_json = json.dumps(
            mechanism_raw,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReactiveCaseValidationError(
            f"mechanism must contain JSON-compatible finite values: {exc}"
        ) from exc
    if any(item.phase is not SpeciesPhase.LIQUID for item in mechanism.species):
        raise ReactiveCaseValidationError(
            "reactive case v1 supports only homogeneous liquid species"
        )
    compiled = compile_mechanism(mechanism, require_reaction_enthalpy=False)
    if mode == "nonisothermal" and compiled.missing_enthalpy_reactions:
        raise ReactiveCaseValidationError(
            "nonisothermal mode requires dH for every reaction; missing: "
            + ", ".join(compiled.missing_enthalpy_reactions)
        )
    species_names = compiled.species_names

    initial_raw = _object(
        top["initial_state"],
        "initial_state",
        allowed=frozenset(
            {
                "temperature_k",
                "operating_pressure_pa",
                "concentrations_mol_per_m3",
            }
        ),
        required=frozenset(
            {
                "temperature_k",
                "operating_pressure_pa",
                "concentrations_mol_per_m3",
            }
        ),
    )
    initial = ReactiveStateV1(
        temperature_k=_number(
            initial_raw["temperature_k"], "initial_state.temperature_k", positive=True
        ),
        operating_pressure_pa=_number(
            initial_raw["operating_pressure_pa"],
            "initial_state.operating_pressure_pa",
            positive=True,
        ),
        concentrations_mol_per_m3=_concentrations(
            initial_raw["concentrations_mol_per_m3"],
            "initial_state.concentrations_mol_per_m3",
            species_names,
        ),
    )
    _validate_state_range(
        mechanism,
        temperature_k=initial.temperature_k,
        pressure_pa=initial.operating_pressure_pa,
        label="initial_state",
    )

    inlets_raw = top["inlets"]
    if not isinstance(inlets_raw, Mapping) or not inlets_raw:
        raise ReactiveCaseValidationError("inlets must be a non-empty JSON object")
    inlets: list[ReactiveInletV1] = []
    normalized_names: set[str] = set()
    for raw_name, raw_inlet in inlets_raw.items():
        name = str(raw_name).strip()
        normalized_name = normalize_group_name(name)
        if not name or not normalized_name:
            raise ReactiveCaseValidationError("inlet names must be non-empty")
        if normalized_name in normalized_names:
            raise ReactiveCaseValidationError(
                f"inlet names collide after normalization: {name!r}"
            )
        normalized_names.add(normalized_name)
        values = _object(
            raw_inlet,
            f"inlets.{name}",
            allowed=frozenset({"temperature_k", "concentrations_mol_per_m3"}),
            required=frozenset({"temperature_k", "concentrations_mol_per_m3"}),
        )
        inlet = ReactiveInletV1(
            name=name,
            normalized_name=normalized_name,
            temperature_k=_number(
                values["temperature_k"],
                f"inlets.{name}.temperature_k",
                positive=True,
            ),
            concentrations_mol_per_m3=_concentrations(
                values["concentrations_mol_per_m3"],
                f"inlets.{name}.concentrations_mol_per_m3",
                species_names,
            ),
        )
        _validate_state_range(
            mechanism,
            temperature_k=inlet.temperature_k,
            pressure_pa=initial.operating_pressure_pa,
            label=f"inlets.{name}",
        )
        inlets.append(inlet)
    if mode == "isothermal":
        for inlet in inlets:
            if not math.isclose(
                inlet.temperature_k,
                initial.temperature_k,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ReactiveCaseValidationError(
                    "isothermal mode requires identical initial and inlet temperatures"
                )

    material_raw = _object(
        top["material"],
        "material",
        allowed=frozenset(
            {
                "density_kg_per_m3",
                "heat_capacity_j_per_kg_k",
                "thermal_diffusivity_m2_s",
                "species_diffusivity_m2_s",
            }
        ),
        required=frozenset(
            {
                "density_kg_per_m3",
                "heat_capacity_j_per_kg_k",
                "thermal_diffusivity_m2_s",
                "species_diffusivity_m2_s",
            }
        ),
    )
    diffusivity_raw = material_raw["species_diffusivity_m2_s"]
    if not isinstance(diffusivity_raw, Mapping):
        raise ReactiveCaseValidationError(
            "material.species_diffusivity_m2_s must be a JSON object"
        )
    unknown_diffusivity = sorted(set(map(str, diffusivity_raw)) - set(species_names))
    missing_diffusivity = sorted(set(species_names) - set(map(str, diffusivity_raw)))
    if unknown_diffusivity or missing_diffusivity:
        raise ReactiveCaseValidationError(
            "species diffusivity keys must exactly match mechanism species; "
            f"missing={missing_diffusivity}, unknown={unknown_diffusivity}"
        )
    diffusivities = tuple(
        _number(
            diffusivity_raw[name],
            f"material.species_diffusivity_m2_s.{name}",
        )
        for name in species_names
    )
    if any(value < 0.0 for value in diffusivities):
        raise ReactiveCaseValidationError("species diffusivities must be non-negative")
    thermal_diffusivity = _number(
        material_raw["thermal_diffusivity_m2_s"],
        "material.thermal_diffusivity_m2_s",
    )
    if thermal_diffusivity < 0.0:
        raise ReactiveCaseValidationError(
            "material.thermal_diffusivity_m2_s must be non-negative"
        )
    material = ReactiveMaterialV1(
        density_kg_per_m3=_number(
            material_raw["density_kg_per_m3"],
            "material.density_kg_per_m3",
            positive=True,
        ),
        heat_capacity_j_per_kg_k=_number(
            material_raw["heat_capacity_j_per_kg_k"],
            "material.heat_capacity_j_per_kg_k",
            positive=True,
        ),
        thermal_diffusivity_m2_s=thermal_diffusivity,
        species_diffusivity_m2_s=diffusivities,
    )

    time_raw = _object(
        top["time"],
        "time",
        allowed=frozenset(
            {
                "num_steps",
                "dt_mode",
                "dt_s",
                "max_dt_s",
                "cfl_target",
                "diffusion_stability_factor",
                "max_transport_substeps",
            }
        ),
        required=frozenset(
            {
                "num_steps",
                "dt_mode",
                "dt_s",
                "max_dt_s",
                "cfl_target",
                "diffusion_stability_factor",
                "max_transport_substeps",
            }
        ),
    )
    if not isinstance(time_raw["dt_mode"], str):
        raise ReactiveCaseValidationError("time.dt_mode must be a string")
    dt_mode = time_raw["dt_mode"].strip().lower()
    if dt_mode not in {"auto", "manual"}:
        raise ReactiveCaseValidationError("time.dt_mode must be auto or manual")
    time = ReactiveTimeV1(
        num_steps=_integer(time_raw["num_steps"], "time.num_steps", positive=True),
        dt_mode=dt_mode,
        dt_s=_number(time_raw["dt_s"], "time.dt_s", positive=True),
        max_dt_s=_number(time_raw["max_dt_s"], "time.max_dt_s", positive=True),
        cfl_target=_number(time_raw["cfl_target"], "time.cfl_target", positive=True),
        diffusion_stability_factor=_number(
            time_raw["diffusion_stability_factor"],
            "time.diffusion_stability_factor",
            positive=True,
        ),
        max_transport_substeps=_integer(
            time_raw["max_transport_substeps"],
            "time.max_transport_substeps",
            positive=True,
        ),
    )

    integrator_raw = _object(
        top["chemistry_integrator"],
        "chemistry_integrator",
        allowed=frozenset(
            {
                "relative_tolerance",
                "concentration_absolute_tolerance_mol_per_m3",
                "temperature_absolute_tolerance_k",
                "max_temperature_change_per_substep_k",
                "min_substep_fraction",
                "max_substeps_per_half_step",
                "cell_batch_size",
            }
        ),
        required=frozenset(
            {
                "relative_tolerance",
                "concentration_absolute_tolerance_mol_per_m3",
                "temperature_absolute_tolerance_k",
                "max_temperature_change_per_substep_k",
                "min_substep_fraction",
                "max_substeps_per_half_step",
                "cell_batch_size",
            }
        ),
    )
    integrator = ChemistryIntegratorV1(
        relative_tolerance=_number(
            integrator_raw["relative_tolerance"],
            "chemistry_integrator.relative_tolerance",
            positive=True,
        ),
        concentration_absolute_tolerance_mol_per_m3=_number(
            integrator_raw["concentration_absolute_tolerance_mol_per_m3"],
            "chemistry_integrator.concentration_absolute_tolerance_mol_per_m3",
            positive=True,
        ),
        temperature_absolute_tolerance_k=_number(
            integrator_raw["temperature_absolute_tolerance_k"],
            "chemistry_integrator.temperature_absolute_tolerance_k",
            positive=True,
        ),
        max_temperature_change_per_substep_k=_number(
            integrator_raw["max_temperature_change_per_substep_k"],
            "chemistry_integrator.max_temperature_change_per_substep_k",
            positive=True,
        ),
        min_substep_fraction=_number(
            integrator_raw["min_substep_fraction"],
            "chemistry_integrator.min_substep_fraction",
            positive=True,
        ),
        max_substeps_per_half_step=_integer(
            integrator_raw["max_substeps_per_half_step"],
            "chemistry_integrator.max_substeps_per_half_step",
            positive=True,
        ),
        cell_batch_size=_integer(
            integrator_raw["cell_batch_size"],
            "chemistry_integrator.cell_batch_size",
            positive=True,
        ),
    )
    if integrator.min_substep_fraction >= 1.0:
        raise ReactiveCaseValidationError(
            "chemistry_integrator.min_substep_fraction must be below 1"
        )

    output_raw = _object(
        top["output"],
        "output",
        allowed=frozenset({"history_stride", "snapshot_steps"}),
        required=frozenset({"history_stride", "snapshot_steps"}),
    )
    snapshots_raw = output_raw["snapshot_steps"]
    if not isinstance(snapshots_raw, list):
        raise ReactiveCaseValidationError("output.snapshot_steps must be an array")
    snapshots = tuple(
        _integer(value, "output.snapshot_steps item", positive=True)
        for value in snapshots_raw
    )
    if len(set(snapshots)) != len(snapshots):
        raise ReactiveCaseValidationError("output.snapshot_steps must be unique")
    if any(value > time.num_steps for value in snapshots):
        raise ReactiveCaseValidationError(
            "output.snapshot_steps cannot exceed time.num_steps"
        )
    output = ReactiveOutputV1(
        history_stride=_integer(
            output_raw["history_stride"], "output.history_stride", positive=True
        ),
        snapshot_steps=tuple(sorted(snapshots)),
    )

    return ReactiveCaseV1(
        schema_version=schema_version,
        case_id=case_id,
        mode=mode,
        mechanism=mechanism,
        compiled_chemistry=compiled,
        mechanism_payload_json=mechanism_payload_json,
        initial_state=initial,
        inlets=tuple(inlets),
        material=material,
        time=time,
        chemistry_integrator=integrator,
        output=output,
    )


def load_reactive_case(path: str | Path) -> ReactiveCaseV1:
    """Load one strict reactive-case JSON document."""

    source = Path(path)
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReactiveCaseValidationError(
            f"cannot read reactive case {source}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ReactiveCaseValidationError("reactive case must contain a JSON object")
    return reactive_case_from_mapping(raw, source_label=str(source))
