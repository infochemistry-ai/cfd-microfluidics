"""Strang-split reactive transport coupler for tetrahedral CFD fields."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from microfluidics.reactive.case import (
    REACTIVE_CASE_CONTRACT_VERSION,
    REACTIVE_TRANSPORT_CONTRACT_VERSION,
    ReactiveCaseV1,
)
from microfluidics.reactive.errors import (
    ChemistryIntegrationError,
    ReactiveWalltimeLimitError,
    TransportSubstepCapError,
)
from microfluidics.reactive.integrator import (
    ChemistryIntegrationStats,
    ReactionAdvanceResult,
    advance_reaction,
)
from microfluidics.reactive.operators import (
    ReactiveSpatialPrecompute,
    advance_spatial_fields,
)


@dataclass(frozen=True, slots=True)
class ReactiveRunResult:
    concentrations_mol_per_m3: np.ndarray
    temperature_k: np.ndarray
    species_sources_mol_per_m3_s: np.ndarray
    heat_release_w_per_m3: np.ndarray | None
    history: tuple[dict[str, object], ...]
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]]
    summary: dict[str, object]


def _species_inventory(
    concentrations_mol_per_m3: np.ndarray, volumes_m3: np.ndarray
) -> np.ndarray:
    return np.asarray(concentrations_mol_per_m3, dtype=np.float64).T @ np.asarray(
        volumes_m3, dtype=np.float64
    )


def _sensible_energy_j(
    temperature_k: np.ndarray,
    volumes_m3: np.ndarray,
    *,
    density_kg_per_m3: float,
    heat_capacity_j_per_kg_k: float,
) -> float:
    return float(
        density_kg_per_m3 * heat_capacity_j_per_kg_k * np.dot(temperature_k, volumes_m3)
    )


def _peak_rss_bytes() -> int | None:
    """Return process peak RSS without adding a runtime dependency."""

    if os.name == "nt":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
                ("quota_nonpaged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory.restype = ctypes.c_int
        return (
            int(counters.peak_working_set_size)
            if get_memory(handle, ctypes.byref(counters), counters.cb)
            else None
        )
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _chemistry_stats_payload(
    *,
    evaluations: int,
    accepted: int,
    rejected: int,
    minimum_dt_s: float,
    maximum_dt_s: float,
    limiting_cell: int | None,
    maximum_error: float,
    maximum_delta_t: float,
) -> dict[str, object]:
    return {
        "evaluations": evaluations,
        "accepted_substeps": accepted,
        "rejected_substeps": rejected,
        "minimum_dt_s": minimum_dt_s,
        "maximum_dt_s": maximum_dt_s,
        "limiting_cell": limiting_cell,
        "maximum_normalized_error": maximum_error,
        "maximum_temperature_change_k": maximum_delta_t,
    }


def run_reactive_transport(
    precompute: ReactiveSpatialPrecompute,
    case: ReactiveCaseV1,
    *,
    upstream_flow_ready: bool = True,
    mesh_flow_compatible: bool = True,
    walltime_limit_s: float | None = None,
    backflow_relative_threshold: float = 0.05,
    balance_relative_tolerance: float = 1.0e-5,
) -> ReactiveRunResult:
    """Run the CPU v1 reactive stage against a fixed flow face-flux field."""

    started = time.monotonic()
    if walltime_limit_s is not None and (
        not math.isfinite(walltime_limit_s) or walltime_limit_s <= 0.0
    ):
        raise ValueError("walltime_limit_s must be finite and positive when provided")
    walltime_deadline = (
        started + walltime_limit_s if walltime_limit_s is not None else None
    )
    volumes = np.asarray(precompute.mesh.cell_volumes, dtype=np.float64)
    cell_count = volumes.size
    species_count = len(case.species_names)
    concentrations = np.repeat(
        np.asarray(case.initial_state.concentrations_mol_per_m3, dtype=np.float64)[
            None, :
        ],
        cell_count,
        axis=0,
    )
    temperature = np.full(
        cell_count, case.initial_state.temperature_k, dtype=np.float64
    )
    initial_moles = _species_inventory(concentrations, volumes)
    elements = tuple(
        sorted(
            {
                element
                for species in case.mechanism.species
                for element in species.elemental_composition
            }
        )
    )
    element_matrix = np.asarray(
        [
            [species.elemental_composition.get(element, 0.0) for element in elements]
            for species in case.mechanism.species
        ],
        dtype=np.float64,
    )
    initial_elements = initial_moles @ element_matrix
    initial_energy = _sensible_energy_j(
        temperature,
        volumes,
        density_kg_per_m3=case.material.density_kg_per_m3,
        heat_capacity_j_per_kg_k=case.material.heat_capacity_j_per_kg_k,
    )

    boundary_in = np.zeros(species_count, dtype=np.float64)
    boundary_out = np.zeros(species_count, dtype=np.float64)
    boundary_adv_in = np.zeros(species_count, dtype=np.float64)
    boundary_adv_out = np.zeros(species_count, dtype=np.float64)
    boundary_diff_in = np.zeros(species_count, dtype=np.float64)
    boundary_diff_out = np.zeros(species_count, dtype=np.float64)
    reaction_change = np.zeros(species_count, dtype=np.float64)
    chemistry_roundoff = np.zeros(species_count, dtype=np.float64)
    transport_roundoff = np.zeros(species_count, dtype=np.float64)
    reaction_heat_integral = 0.0
    reaction_sensible_energy_change = 0.0
    energy_adv_in = 0.0
    energy_adv_out = 0.0
    energy_diff_in = 0.0
    energy_diff_out = 0.0
    energy_roundoff = 0.0
    cumulative_backflow_m3 = 0.0
    history: list[dict[str, object]] = []
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    physical_time_s = 0.0
    executed_steps = 0
    failure_status: str | None = None
    dt_values: list[float] = []
    transport_substeps_total = 0
    transport_substeps_max = 0
    limiter_active_transfers = 0
    limiter_min_scale = 1.0
    pairwise_conservation_error = 0.0
    chemistry_evaluations = 0
    chemistry_accepted = 0
    chemistry_rejected = 0
    chemistry_min_dt = math.inf
    chemistry_max_dt = 0.0
    chemistry_limiting_cell: int | None = None
    chemistry_max_error = 0.0
    chemistry_max_delta_t = 0.0

    def collect_chemistry_stats(stats: ChemistryIntegrationStats) -> None:
        nonlocal chemistry_evaluations
        nonlocal chemistry_accepted
        nonlocal chemistry_rejected
        nonlocal chemistry_min_dt
        nonlocal chemistry_max_dt
        nonlocal chemistry_limiting_cell
        nonlocal chemistry_max_error
        nonlocal chemistry_max_delta_t
        chemistry_evaluations += stats.evaluations
        chemistry_accepted += stats.accepted_substeps
        chemistry_rejected += stats.rejected_substeps
        if stats.minimum_dt_s > 0.0:
            chemistry_min_dt = min(chemistry_min_dt, stats.minimum_dt_s)
        chemistry_max_dt = max(chemistry_max_dt, stats.maximum_dt_s)
        if stats.maximum_normalized_error >= chemistry_max_error:
            chemistry_max_error = stats.maximum_normalized_error
            chemistry_limiting_cell = stats.limiting_cell
        chemistry_max_delta_t = max(
            chemistry_max_delta_t, stats.maximum_temperature_change_k
        )

    def reaction_advance(
        source_concentrations: np.ndarray,
        source_temperature: np.ndarray,
        interval_s: float,
        outer_dt_s: float,
    ) -> ReactionAdvanceResult:
        return advance_reaction(
            case.compiled_chemistry,
            source_concentrations,
            source_temperature,
            pressure_pa=case.initial_state.operating_pressure_pa,
            interval_s=interval_s,
            mode=case.mode,
            material=case.material,
            settings=case.chemistry_integrator,
            minimum_reference_dt_s=outer_dt_s,
            walltime_deadline_monotonic=walltime_deadline,
        )

    def collect_reaction_accounting(
        *advances: ReactionAdvanceResult,
    ) -> None:
        nonlocal reaction_heat_integral
        nonlocal reaction_sensible_energy_change
        for advanced in advances:
            actual_change = advanced.concentration_change_mol_per_m3.T @ volumes
            roundoff_change = advanced.roundoff_normalization_mol_per_m3.T @ volumes
            reaction_change[:] += actual_change - roundoff_change
            chemistry_roundoff[:] += roundoff_change
            reaction_heat_integral += float(
                np.dot(advanced.heat_release_integral_j_per_m3, volumes)
            )
            reaction_sensible_energy_change += float(
                np.dot(advanced.sensible_energy_change_j_per_m3, volumes)
            )
            collect_chemistry_stats(advanced.stats)

    heat_capacity_volume = (
        case.material.density_kg_per_m3 * case.material.heat_capacity_j_per_kg_k
    )
    for outer_step in range(1, case.time.num_steps + 1):
        if walltime_deadline is not None and time.monotonic() >= walltime_deadline:
            failure_status = "blocked_walltime_limit"
            break
        outer_dt_s = (
            case.time.dt_s if case.time.dt_mode == "manual" else case.time.max_dt_s
        )
        try:
            first_reaction = reaction_advance(
                concentrations,
                temperature,
                0.5 * outer_dt_s,
                outer_dt_s,
            )
            spatial = advance_spatial_fields(
                precompute,
                case,
                first_reaction.concentrations_mol_per_m3,
                first_reaction.temperature_k,
                outer_dt_s=outer_dt_s,
                walltime_deadline_monotonic=walltime_deadline,
            )
            second_reaction = reaction_advance(
                spatial.concentrations_mol_per_m3,
                spatial.temperature_k,
                0.5 * outer_dt_s,
                outer_dt_s,
            )
        except ReactiveWalltimeLimitError:
            failure_status = "blocked_walltime_limit"
            break
        except TransportSubstepCapError:
            failure_status = "blocked_transport_substep_cap"
            break
        except ChemistryIntegrationError:
            failure_status = "chemistry_stiffness_unsupported"
            break

        concentrations = second_reaction.concentrations_mol_per_m3
        temperature = second_reaction.temperature_k
        collect_reaction_accounting(first_reaction, second_reaction)
        diagnostics = spatial.diagnostics
        adv_in = np.asarray(diagnostics.species_advective_in_mol, dtype=np.float64)
        adv_out = np.asarray(diagnostics.species_advective_out_mol, dtype=np.float64)
        diff_in = np.asarray(diagnostics.species_diffusive_in_mol, dtype=np.float64)
        diff_out = np.asarray(diagnostics.species_diffusive_out_mol, dtype=np.float64)
        boundary_adv_in += adv_in
        boundary_adv_out += adv_out
        boundary_diff_in += diff_in
        boundary_diff_out += diff_out
        boundary_in += adv_in + diff_in
        boundary_out += adv_out + diff_out
        transport_roundoff += np.asarray(
            diagnostics.roundoff_normalization_mol, dtype=np.float64
        )
        energy_adv_in += heat_capacity_volume * diagnostics.thermal_advective_in_k_m3
        energy_adv_out += heat_capacity_volume * diagnostics.thermal_advective_out_k_m3
        energy_diff_in += heat_capacity_volume * diagnostics.thermal_diffusive_in_k_m3
        energy_diff_out += heat_capacity_volume * diagnostics.thermal_diffusive_out_k_m3
        energy_roundoff += (
            heat_capacity_volume * diagnostics.thermal_roundoff_normalization_k_m3
        )
        cumulative_backflow_m3 += diagnostics.outlet_backflow_m3
        transport_substeps_total += diagnostics.substeps
        transport_substeps_max = max(transport_substeps_max, diagnostics.substeps)
        limiter_active_transfers += diagnostics.limiter_active_transfers
        limiter_min_scale = min(limiter_min_scale, diagnostics.limiter_min_scale)
        pairwise_conservation_error = max(
            pairwise_conservation_error,
            diagnostics.pairwise_conservation_error,
        )

        executed_steps = outer_step
        physical_time_s += outer_dt_s
        dt_values.append(outer_dt_s)
        if outer_step in case.output.snapshot_steps:
            snapshots[outer_step] = (
                concentrations.copy(),
                temperature.copy(),
            )
        if (
            outer_step % case.output.history_stride == 0
            or outer_step == case.time.num_steps
        ):
            inventory = _species_inventory(concentrations, volumes)
            species_residual_now = (
                inventory
                - initial_moles
                - boundary_in
                + boundary_out
                - reaction_change
                - chemistry_roundoff
                - transport_roundoff
            )
            element_inventory = inventory @ element_matrix
            element_residual_now = (
                element_inventory
                - initial_elements
                - boundary_in @ element_matrix
                + boundary_out @ element_matrix
                - reaction_change @ element_matrix
                - chemistry_roundoff @ element_matrix
                - transport_roundoff @ element_matrix
            )
            history.append(
                {
                    "outer_step": outer_step,
                    "physical_time_s": physical_time_s,
                    "dt_s": outer_dt_s,
                    "total_moles": dict(
                        zip(
                            case.species_names,
                            (float(value) for value in inventory),
                            strict=True,
                        )
                    ),
                    "total_mass_kg": float(
                        inventory @ case.compiled_chemistry.molecular_weights_kg_per_mol
                    ),
                    "elemental_inventories_moles": dict(
                        zip(
                            elements,
                            (float(value) for value in element_inventory),
                            strict=True,
                        )
                    ),
                    "temperature_k": {
                        "minimum": float(np.min(temperature)),
                        "maximum": float(np.max(temperature)),
                        "mean": float(np.mean(temperature)),
                    },
                    "cumulative_reaction_heat_j": reaction_heat_integral,
                    "balance_residuals": {
                        "species_moles": dict(
                            zip(
                                case.species_names,
                                (float(value) for value in species_residual_now),
                                strict=True,
                            )
                        ),
                        "elements_moles": dict(
                            zip(
                                elements,
                                (float(value) for value in element_residual_now),
                                strict=True,
                            )
                        ),
                    },
                    "chemistry": {
                        "evaluations": chemistry_evaluations,
                        "accepted_substeps": chemistry_accepted,
                        "rejected_substeps": chemistry_rejected,
                    },
                }
            )

    final_moles = _species_inventory(concentrations, volumes)
    species_residual = (
        final_moles
        - initial_moles
        - boundary_in
        + boundary_out
        - reaction_change
        - chemistry_roundoff
        - transport_roundoff
    )
    species_denominator = np.maximum(
        np.abs(initial_moles)
        + np.abs(boundary_in)
        + np.abs(boundary_out)
        + np.abs(reaction_change)
        + np.abs(chemistry_roundoff)
        + np.abs(transport_roundoff),
        1.0e-30,
    )
    species_relative_residual = np.abs(species_residual) / species_denominator
    molecular_weights = case.compiled_chemistry.molecular_weights_kg_per_mol
    initial_mass = float(initial_moles @ molecular_weights)
    final_mass = float(final_moles @ molecular_weights)
    mass_in = float(boundary_in @ molecular_weights)
    mass_out = float(boundary_out @ molecular_weights)
    mass_reaction_change = float(reaction_change @ molecular_weights)
    mass_chemistry_roundoff = float(chemistry_roundoff @ molecular_weights)
    mass_transport_roundoff = float(transport_roundoff @ molecular_weights)
    mass_residual = (
        final_mass
        - initial_mass
        - mass_in
        + mass_out
        - mass_reaction_change
        - mass_chemistry_roundoff
        - mass_transport_roundoff
    )
    mass_scale = max(
        abs(initial_mass)
        + abs(mass_in)
        + abs(mass_out)
        + abs(mass_reaction_change)
        + abs(mass_chemistry_roundoff)
        + abs(mass_transport_roundoff),
        1.0e-30,
    )
    mass_relative_residual = abs(mass_residual) / mass_scale
    reaction_mass_scale = max(
        float(np.sum(np.abs(reaction_change * molecular_weights))), 1.0e-30
    )
    reaction_mass_relative_residual = abs(mass_reaction_change) / reaction_mass_scale

    final_elements = final_moles @ element_matrix
    element_in = boundary_in @ element_matrix
    element_out = boundary_out @ element_matrix
    element_reaction = reaction_change @ element_matrix
    element_chemistry_roundoff = chemistry_roundoff @ element_matrix
    element_transport_roundoff = transport_roundoff @ element_matrix
    element_residual = (
        final_elements
        - initial_elements
        - element_in
        + element_out
        - element_reaction
        - element_chemistry_roundoff
        - element_transport_roundoff
    )
    element_scale = np.maximum(
        np.abs(initial_elements)
        + np.abs(element_in)
        + np.abs(element_out)
        + np.abs(element_reaction)
        + np.abs(element_chemistry_roundoff)
        + np.abs(element_transport_roundoff),
        1.0e-30,
    )
    element_relative_residual = np.abs(element_residual) / element_scale
    reaction_element_scale = np.maximum(
        np.abs(reaction_change) @ element_matrix, 1.0e-30
    )
    reaction_element_relative_residual = (
        np.abs(element_reaction) / reaction_element_scale
    )

    final_energy = _sensible_energy_j(
        temperature,
        volumes,
        density_kg_per_m3=case.material.density_kg_per_m3,
        heat_capacity_j_per_kg_k=case.material.heat_capacity_j_per_kg_k,
    )
    energy_residual = (
        final_energy
        - initial_energy
        - energy_adv_in
        + energy_adv_out
        - energy_diff_in
        + energy_diff_out
        - reaction_heat_integral
        - energy_roundoff
    )
    energy_scale = max(
        abs(initial_energy)
        + abs(energy_adv_in)
        + abs(energy_adv_out)
        + abs(energy_diff_in)
        + abs(energy_diff_out)
        + abs(reaction_heat_integral)
        + abs(energy_roundoff),
        1.0e-30,
    )
    energy_relative_residual = abs(energy_residual) / energy_scale
    reaction_coupling_residual = (
        reaction_sensible_energy_change - reaction_heat_integral
    )
    reaction_coupling_scale = max(
        abs(reaction_sensible_energy_change),
        abs(reaction_heat_integral),
        1.0e-30,
    )
    reaction_coupling_relative_residual = (
        abs(reaction_coupling_residual) / reaction_coupling_scale
    )
    energy_balance_available = case.mode != "isothermal"

    finite = bool(np.isfinite(concentrations).all() and np.isfinite(temperature).all())
    positive_temperature = bool(np.all(temperature > 0.0))
    authored_temperature_range = case.mechanism.temperature_range_k
    temperature_in_mechanism_range = bool(
        authored_temperature_range is None
        or (
            np.all(temperature >= authored_temperature_range[0])
            and np.all(temperature <= authored_temperature_range[1])
        )
    )
    nonnegative = bool(
        np.all(
            concentrations
            >= -case.chemistry_integrator.concentration_absolute_tolerance_mol_per_m3
        )
    )
    run_completed = executed_steps == case.time.num_steps
    numerically_stable = bool(
        run_completed
        and failure_status is None
        and finite
        and nonnegative
        and positive_temperature
    )
    positive_outlet_rate = float(
        np.sum(
            np.maximum(precompute.face_volume_flux_m3_s[precompute.outlet_faces], 0.0)
        )
    )
    backflow_ratio = precompute.outlet_backflow_m3_s / max(
        positive_outlet_rate, 1.0e-30
    )
    material_balances_ok = bool(
        np.max(species_relative_residual, initial=0.0) <= balance_relative_tolerance
        and mass_relative_residual <= balance_relative_tolerance
        and np.max(element_relative_residual, initial=0.0) <= balance_relative_tolerance
        and reaction_mass_relative_residual <= balance_relative_tolerance
        and np.max(reaction_element_relative_residual, initial=0.0)
        <= balance_relative_tolerance
    )
    energy_balance_ok = bool(
        not energy_balance_available
        or (
            energy_relative_residual <= balance_relative_tolerance
            and reaction_coupling_relative_residual <= balance_relative_tolerance
        )
    )
    physically_ready = bool(
        numerically_stable
        and upstream_flow_ready
        and mesh_flow_compatible
        and backflow_ratio <= backflow_relative_threshold
        and material_balances_ok
        and energy_balance_ok
        and temperature_in_mechanism_range
    )
    readiness_blockers: list[str] = []
    if not run_completed:
        readiness_blockers.append(failure_status or "incomplete")
    if not finite:
        readiness_blockers.append("nonfinite_state")
    if not nonnegative:
        readiness_blockers.append("negative_concentration")
    if not positive_temperature:
        readiness_blockers.append("nonpositive_temperature")
    if not upstream_flow_ready:
        readiness_blockers.append("upstream_flow_unready")
    if not mesh_flow_compatible:
        readiness_blockers.append("mesh_flow_incompatible")
    if backflow_ratio > backflow_relative_threshold:
        readiness_blockers.append("outlet_backflow_above_threshold")
    if not material_balances_ok:
        readiness_blockers.append("material_balance_failed")
    if not energy_balance_ok:
        readiness_blockers.append("energy_balance_failed")
    if not temperature_in_mechanism_range:
        readiness_blockers.append("temperature_outside_mechanism_range")
    if failure_status is not None:
        readiness_status = failure_status
    elif physically_ready:
        readiness_status = "completed"
    elif run_completed:
        readiness_status = "completed_not_ready"
    else:
        readiness_status = "incomplete"

    if case.mode == "off":
        final_sources = np.zeros_like(concentrations)
        final_heat: np.ndarray | None = np.zeros_like(temperature)
    else:
        sources = case.compiled_chemistry.evaluate_sources(
            concentrations,
            temperature_k=temperature,
            pressure_pa=case.initial_state.operating_pressure_pa,
            require_heat_release=(case.mode == "nonisothermal"),
        )
        final_sources = sources.species_sources_mol_per_m3_s
        final_heat = sources.heat_release_w_per_m3

    outlet_faces = precompute.outlet_faces
    outlet_q = np.maximum(precompute.face_volume_flux_m3_s[outlet_faces], 0.0)
    outlet_owner = precompute.owner[outlet_faces]
    outlet_q_total = float(np.sum(outlet_q))
    outlet_composition = (
        np.sum(outlet_q[:, None] * concentrations[outlet_owner], axis=0)
        / outlet_q_total
        if outlet_q_total > 0.0
        else np.zeros(species_count, dtype=np.float64)
    )
    runtime_s = time.monotonic() - started
    peak_rss_bytes = _peak_rss_bytes()
    species_index = {
        name: index for index, name in enumerate(case.compiled_chemistry.species_names)
    }
    reactant_names = sorted(
        {name for reaction in case.mechanism.reactions for name in reaction.reactants}
    )
    product_names = sorted(
        {name for reaction in case.mechanism.reactions for name in reaction.products}
    )
    conversions = {
        name: (
            max(-float(reaction_change[species_index[name]]), 0.0)
            / max(
                float(
                    initial_moles[species_index[name]]
                    + boundary_in[species_index[name]]
                ),
                1.0e-30,
            )
        )
        for name in reactant_names
        if initial_moles[species_index[name]] + boundary_in[species_index[name]] > 0.0
    }
    positive_product_generation = {
        name: max(float(reaction_change[species_index[name]]), 0.0)
        for name in product_names
    }
    total_product_generation = sum(positive_product_generation.values())
    selectivities = (
        {
            name: value / total_product_generation
            for name, value in positive_product_generation.items()
        }
        if total_product_generation > 0.0
        else {}
    )
    summary: dict[str, object] = {
        "contract_version": REACTIVE_TRANSPORT_CONTRACT_VERSION,
        "reactive_case_contract_version": REACTIVE_CASE_CONTRACT_VERSION,
        "case_id": case.case_id,
        "mode": case.mode,
        "reactive_case_sha256": case.reactive_case_sha256,
        "mechanism_sha256": case.mechanism_sha256,
        "species_names": list(case.species_names),
        "cell_count": cell_count,
        "species_count": species_count,
        "reaction_count": len(case.compiled_chemistry.reaction_ids),
        "requested_steps": case.time.num_steps,
        "executed_steps": executed_steps,
        "physical_time_s": physical_time_s,
        "dt": {
            "mode": case.time.dt_mode,
            "minimum_s": min(dt_values) if dt_values else 0.0,
            "maximum_s": max(dt_values) if dt_values else 0.0,
            "mean_s": float(np.mean(dt_values)) if dt_values else 0.0,
        },
        "transport": {
            "substeps_total": transport_substeps_total,
            "substeps_max": transport_substeps_max,
            "limiter_active_transfers": limiter_active_transfers,
            "limiter_min_scale": limiter_min_scale,
            "pairwise_conservation_error": pairwise_conservation_error,
            "physical_clipping_used": False,
        },
        "chemistry": _chemistry_stats_payload(
            evaluations=chemistry_evaluations,
            accepted=chemistry_accepted,
            rejected=chemistry_rejected,
            minimum_dt_s=(0.0 if math.isinf(chemistry_min_dt) else chemistry_min_dt),
            maximum_dt_s=chemistry_max_dt,
            limiting_cell=chemistry_limiting_cell,
            maximum_error=chemistry_max_error,
            maximum_delta_t=chemistry_max_delta_t,
        ),
        "species": {
            name: {
                "minimum_mol_per_m3": float(np.min(concentrations[:, index])),
                "maximum_mol_per_m3": float(np.max(concentrations[:, index])),
                "mean_mol_per_m3": float(np.mean(concentrations[:, index])),
                "total_moles": float(final_moles[index]),
            }
            for index, name in enumerate(case.species_names)
        },
        "temperature_k": {
            "minimum": float(np.min(temperature)),
            "maximum": float(np.max(temperature)),
            "mean": float(np.mean(temperature)),
        },
        "outlet_flux_weighted_concentrations_mol_per_m3": dict(
            zip(
                case.species_names,
                (float(value) for value in outlet_composition),
                strict=True,
            )
        ),
        "conversion": {
            "definition": "net reaction consumption / (initial + boundary-in moles)",
            "by_reactant": conversions,
        },
        "selectivity": {
            "definition": "net molar reaction production / total positive product production",
            "by_product": selectivities,
        },
        "final_heat_release_w_per_m3": (
            {
                "available": True,
                "minimum": float(np.min(final_heat)),
                "maximum": float(np.max(final_heat)),
            }
            if final_heat is not None
            else {"available": False}
        ),
        "balances": {
            "species": {
                name: {
                    "initial_moles": float(initial_moles[index]),
                    "final_moles": float(final_moles[index]),
                    "advective_in_moles": float(boundary_adv_in[index]),
                    "advective_out_moles": float(boundary_adv_out[index]),
                    "diffusive_in_moles": float(boundary_diff_in[index]),
                    "diffusive_out_moles": float(boundary_diff_out[index]),
                    "reaction_change_moles": float(reaction_change[index]),
                    "chemistry_roundoff_normalization_moles": float(
                        chemistry_roundoff[index]
                    ),
                    "transport_roundoff_normalization_moles": float(
                        transport_roundoff[index]
                    ),
                    "residual_moles": float(species_residual[index]),
                    "relative_residual": float(species_relative_residual[index]),
                }
                for index, name in enumerate(case.species_names)
            },
            "mass": {
                "initial_kg": initial_mass,
                "final_kg": final_mass,
                "in_kg": mass_in,
                "out_kg": mass_out,
                "reaction_change_kg": mass_reaction_change,
                "reaction_relative_residual": reaction_mass_relative_residual,
                "chemistry_roundoff_normalization_kg": mass_chemistry_roundoff,
                "transport_roundoff_normalization_kg": mass_transport_roundoff,
                "residual_kg": mass_residual,
                "relative_residual": mass_relative_residual,
            },
            "elements": {
                element: {
                    "initial_moles": float(initial_elements[index]),
                    "final_moles": float(final_elements[index]),
                    "reaction_change_moles": float(element_reaction[index]),
                    "chemistry_roundoff_normalization_moles": float(
                        element_chemistry_roundoff[index]
                    ),
                    "transport_roundoff_normalization_moles": float(
                        element_transport_roundoff[index]
                    ),
                    "residual_moles": float(element_residual[index]),
                    "relative_residual": float(element_relative_residual[index]),
                }
                for index, element in enumerate(elements)
            },
            "maximum_element_relative_residual": float(
                np.max(element_relative_residual, initial=0.0)
            ),
            "maximum_reaction_element_relative_residual": float(
                np.max(reaction_element_relative_residual, initial=0.0)
            ),
            "energy": {
                "available": energy_balance_available,
                "initial_j": initial_energy,
                "final_j": final_energy,
                "advective_in_j": energy_adv_in,
                "advective_out_j": energy_adv_out,
                "diffusive_in_j": energy_diff_in,
                "diffusive_out_j": energy_diff_out,
                "reaction_heat_j": reaction_heat_integral,
                "reaction_heat_integral_j": reaction_heat_integral,
                "reaction_sensible_energy_change_j": (reaction_sensible_energy_change),
                "reaction_coupling_residual_j": reaction_coupling_residual,
                "reaction_coupling_relative_residual": (
                    reaction_coupling_relative_residual
                ),
                "roundoff_normalization_j": energy_roundoff,
                "residual_j": energy_residual,
                "relative_residual": energy_relative_residual,
            },
        },
        "backflow": {
            "outlet_backflow_m3": cumulative_backflow_m3,
            "relative_rate": backflow_ratio,
            "threshold": backflow_relative_threshold,
        },
        "runtime": {
            "wall_seconds": runtime_s,
            "walltime_limit_s": walltime_limit_s,
            "soft_walltime_enabled": walltime_limit_s is not None,
            "outer_steps_per_second": executed_steps / max(runtime_s, 1.0e-30),
            "chemistry_evaluations_per_second": chemistry_evaluations
            / max(runtime_s, 1.0e-30),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_gib": (
                peak_rss_bytes / (1024**3) if peak_rss_bytes is not None else None
            ),
        },
        "readiness": {
            "run_completed": run_completed,
            "numerically_stable": numerically_stable,
            "physically_ready": physically_ready,
            "ready_for_next_stage": physically_ready,
            "status": readiness_status,
            "blocking_reasons": readiness_blockers,
            "upstream_flow_ready": upstream_flow_ready,
            "mesh_flow_compatible": mesh_flow_compatible,
            "material_balances_ok": material_balances_ok,
            "energy_balance_ok": energy_balance_ok,
            "positive_temperature": positive_temperature,
            "temperature_in_mechanism_range": temperature_in_mechanism_range,
        },
    }
    return ReactiveRunResult(
        concentrations_mol_per_m3=concentrations,
        temperature_k=temperature,
        species_sources_mol_per_m3_s=final_sources,
        heat_release_w_per_m3=final_heat,
        history=tuple(history),
        snapshots=snapshots,
        summary=summary,
    )
