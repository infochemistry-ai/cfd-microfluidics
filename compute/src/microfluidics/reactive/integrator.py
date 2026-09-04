"""Adaptive mesh-agnostic Euler-Heun chemistry integration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from microfluidics.chemistry import CompiledChemistry
from microfluidics.chemistry.errors import ChemistryEvaluationError
from microfluidics.reactive.case import ChemistryIntegratorV1, ReactiveMaterialV1
from microfluidics.reactive.errors import (
    ChemistryIntegrationError,
    ReactiveWalltimeLimitError,
)


@dataclass(frozen=True, slots=True)
class ChemistryIntegrationStats:
    evaluations: int
    accepted_substeps: int
    rejected_substeps: int
    minimum_dt_s: float
    maximum_dt_s: float
    limiting_cell: int | None
    maximum_normalized_error: float
    maximum_temperature_change_k: float
    roundoff_normalization_mol_per_m3: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReactionAdvanceResult:
    concentrations_mol_per_m3: np.ndarray
    temperature_k: np.ndarray
    concentration_change_mol_per_m3: np.ndarray
    heat_release_integral_j_per_m3: np.ndarray
    sensible_energy_change_j_per_m3: np.ndarray
    roundoff_normalization_mol_per_m3: np.ndarray
    stats: ChemistryIntegrationStats


def _empty_stats(species_count: int) -> ChemistryIntegrationStats:
    return ChemistryIntegrationStats(
        evaluations=0,
        accepted_substeps=0,
        rejected_substeps=0,
        minimum_dt_s=0.0,
        maximum_dt_s=0.0,
        limiting_cell=None,
        maximum_normalized_error=0.0,
        maximum_temperature_change_k=0.0,
        roundoff_normalization_mol_per_m3=(0.0,) * species_count,
    )


def _temperature_in_range(
    chemistry: CompiledChemistry, temperature_k: np.ndarray
) -> bool:
    authored = chemistry.mechanism.temperature_range_k
    if authored is None:
        return True
    lower, upper = authored
    return bool(np.all((temperature_k >= lower) & (temperature_k <= upper)))


def advance_reaction(
    chemistry: CompiledChemistry,
    concentrations_mol_per_m3: np.ndarray,
    temperature_k: np.ndarray,
    *,
    pressure_pa: float,
    interval_s: float,
    mode: str,
    material: ReactiveMaterialV1,
    settings: ChemistryIntegratorV1,
    minimum_reference_dt_s: float | None = None,
    walltime_deadline_monotonic: float | None = None,
) -> ReactionAdvanceResult:
    """Advance independent homogeneous cell states over one reaction interval."""

    concentrations = np.asarray(concentrations_mol_per_m3, dtype=np.float64)
    temperatures = np.asarray(temperature_k, dtype=np.float64)
    if concentrations.ndim != 2:
        raise ValueError("concentrations must have shape (n_cells, n_species)")
    if temperatures.shape != (concentrations.shape[0],):
        raise ValueError("temperature must have shape (n_cells,)")
    if concentrations.shape[1] != len(chemistry.species_names):
        raise ValueError("concentration columns do not match chemistry species")
    if not math.isfinite(interval_s) or interval_s < 0.0:
        raise ValueError("interval_s must be finite and non-negative")
    if mode not in {"off", "isothermal", "nonisothermal"}:
        raise ValueError(f"unsupported reactive mode {mode!r}")

    initial_c = np.array(concentrations, copy=True)
    initial_t = np.array(temperatures, copy=True)
    if interval_s == 0.0 or mode == "off":
        return ReactionAdvanceResult(
            concentrations_mol_per_m3=initial_c,
            temperature_k=initial_t,
            concentration_change_mol_per_m3=np.zeros_like(initial_c),
            heat_release_integral_j_per_m3=np.zeros_like(initial_t),
            sensible_energy_change_j_per_m3=np.zeros_like(initial_t),
            roundoff_normalization_mol_per_m3=np.zeros_like(initial_c),
            stats=_empty_stats(initial_c.shape[1]),
        )

    result_c = np.array(initial_c, copy=True)
    result_t = np.array(initial_t, copy=True)
    heat_release_integral = np.zeros_like(result_t)
    roundoff_field = np.zeros_like(result_c)
    species_count = result_c.shape[1]
    roundoff = np.zeros(species_count, dtype=np.float64)
    evaluations = 0
    accepted = 0
    rejected = 0
    min_h = math.inf
    max_h = 0.0
    max_error = 0.0
    max_delta_t = 0.0
    limiting_cell: int | None = None
    heat_capacity_volume = (
        material.density_kg_per_m3 * material.heat_capacity_j_per_kg_k
    )
    atol_c = settings.concentration_absolute_tolerance_mol_per_m3
    rtol = settings.relative_tolerance
    reference_dt = (
        float(minimum_reference_dt_s)
        if minimum_reference_dt_s is not None
        else float(interval_s)
    )
    minimum_h = reference_dt * settings.min_substep_fraction

    for batch_start in range(0, result_c.shape[0], settings.cell_batch_size):
        if (
            walltime_deadline_monotonic is not None
            and time.monotonic() >= walltime_deadline_monotonic
        ):
            raise ReactiveWalltimeLimitError(
                "reactive transport reached its walltime limit during chemistry subcycling"
            )
        batch_end = min(result_c.shape[0], batch_start + settings.cell_batch_size)
        c = np.array(result_c[batch_start:batch_end], copy=True)
        t = np.array(result_t[batch_start:batch_end], copy=True)
        batch_heat_release_integral = np.zeros_like(t)
        elapsed = 0.0
        h = float(interval_s)
        attempts = 0

        while elapsed < interval_s:
            if (
                walltime_deadline_monotonic is not None
                and time.monotonic() >= walltime_deadline_monotonic
            ):
                raise ReactiveWalltimeLimitError(
                    "reactive transport reached its walltime limit during chemistry subcycling"
                )
            remaining = interval_s - elapsed
            h = min(h, remaining)
            if h < minimum_h and not math.isclose(
                h, remaining, rel_tol=0.0, abs_tol=np.finfo(np.float64).eps
            ):
                raise ChemistryIntegrationError(
                    "chemistry_stiffness_unsupported: substep fell below "
                    "outer_dt * min_substep_fraction"
                )
            attempts += 1
            if attempts > settings.max_substeps_per_half_step:
                raise ChemistryIntegrationError(
                    "chemistry_stiffness_unsupported: "
                    "max_substeps_per_half_step exceeded"
                )

            try:
                source0 = chemistry.evaluate_sources(
                    c,
                    temperature_k=t,
                    pressure_pa=pressure_pa,
                    require_heat_release=(mode == "nonisothermal"),
                )
            except ChemistryEvaluationError as exc:
                raise ChemistryIntegrationError(
                    f"chemistry_stiffness_unsupported: {exc}"
                ) from exc
            evaluations += c.shape[0]
            dc0 = source0.species_sources_mol_per_m3_s
            if mode == "nonisothermal":
                assert source0.heat_release_w_per_m3 is not None
                dt0 = source0.heat_release_w_per_m3 / heat_capacity_volume
            else:
                dt0 = np.zeros_like(t)

            c_euler = c + h * dc0
            t_euler = t + h * dt0
            predictor_valid = bool(
                np.isfinite(c_euler).all()
                and np.isfinite(t_euler).all()
                and np.all(c_euler >= 0.0)
                and np.all(t_euler > 0.0)
                and _temperature_in_range(chemistry, t_euler)
            )
            if predictor_valid:
                try:
                    source1 = chemistry.evaluate_sources(
                        c_euler,
                        temperature_k=t_euler,
                        pressure_pa=pressure_pa,
                        require_heat_release=(mode == "nonisothermal"),
                    )
                except ChemistryEvaluationError:
                    predictor_valid = False
                else:
                    evaluations += c.shape[0]

            if predictor_valid:
                dc1 = source1.species_sources_mol_per_m3_s
                if mode == "nonisothermal":
                    assert source1.heat_release_w_per_m3 is not None
                    dt1 = source1.heat_release_w_per_m3 / heat_capacity_volume
                else:
                    dt1 = np.zeros_like(t)
                c_heun = c + 0.5 * h * (dc0 + dc1)
                t_heun = t + 0.5 * h * (dt0 + dt1)

                c_scale = atol_c + rtol * np.maximum(np.abs(c), np.abs(c_heun))
                c_error_values = np.abs(c_heun - c_euler) / c_scale
                local_flat = int(np.argmax(c_error_values))
                error = float(c_error_values.reshape(-1)[local_flat])
                local_cell = local_flat // species_count
                if mode == "nonisothermal":
                    t_scale = (
                        settings.temperature_absolute_tolerance_k
                        + rtol * np.maximum(np.abs(t), np.abs(t_heun))
                    )
                    t_error_values = np.abs(t_heun - t_euler) / t_scale
                    t_local = int(np.argmax(t_error_values))
                    t_error = float(t_error_values[t_local])
                    if t_error > error:
                        error = t_error
                        local_cell = t_local
                delta_t_max = float(np.max(np.abs(t_heun - t)))
                state_valid = bool(
                    np.isfinite(c_heun).all()
                    and np.isfinite(t_heun).all()
                    and np.all(c_heun >= -atol_c)
                    and np.all(t_heun > 0.0)
                    and _temperature_in_range(chemistry, t_heun)
                    and delta_t_max <= settings.max_temperature_change_per_substep_k
                )
            else:
                error = math.inf
                local_cell = 0
                delta_t_max = math.inf
                state_valid = False

            if math.isfinite(error) and error > max_error:
                max_error = error
                limiting_cell = batch_start + local_cell

            if state_valid and error <= 1.0:
                negative_roundoff = (c_heun < 0.0) & (c_heun >= -atol_c)
                if np.any(negative_roundoff):
                    correction = np.where(negative_roundoff, -c_heun, 0.0)
                    roundoff += np.sum(correction, axis=0)
                    roundoff_field[batch_start:batch_end] += correction
                    c_heun = np.where(negative_roundoff, 0.0, c_heun)
                c = c_heun
                if mode == "nonisothermal":
                    batch_heat_release_integral += (
                        0.5
                        * h
                        * (
                            source0.heat_release_w_per_m3
                            + source1.heat_release_w_per_m3
                        )
                    )
                    t = t_heun
                elapsed += h
                accepted += 1
                min_h = min(min_h, h)
                max_h = max(max_h, h)
                max_delta_t = max(max_delta_t, delta_t_max)
                factor = 2.0 if error == 0.0 else 0.9 / math.sqrt(error)
                h *= min(2.0, max(0.2, factor))
                continue

            rejected += 1
            factor = 0.2 if not math.isfinite(error) else 0.9 / math.sqrt(error)
            h *= min(1.0, max(0.2, factor))
            if h < minimum_h:
                raise ChemistryIntegrationError(
                    "chemistry_stiffness_unsupported: substep fell below "
                    "outer_dt * min_substep_fraction"
                )

        result_c[batch_start:batch_end] = c
        result_t[batch_start:batch_end] = t
        heat_release_integral[batch_start:batch_end] = batch_heat_release_integral

    if not np.isfinite(result_c).all() or not np.isfinite(result_t).all():
        raise ChemistryIntegrationError(
            "chemistry_stiffness_unsupported: non-finite accepted state"
        )
    return ReactionAdvanceResult(
        concentrations_mol_per_m3=result_c,
        temperature_k=result_t,
        concentration_change_mol_per_m3=result_c - initial_c,
        heat_release_integral_j_per_m3=heat_release_integral,
        sensible_energy_change_j_per_m3=(heat_capacity_volume * (result_t - initial_t)),
        roundoff_normalization_mol_per_m3=roundoff_field,
        stats=ChemistryIntegrationStats(
            evaluations=evaluations,
            accepted_substeps=accepted,
            rejected_substeps=rejected,
            minimum_dt_s=(0.0 if math.isinf(min_h) else min_h),
            maximum_dt_s=max_h,
            limiting_cell=limiting_cell,
            maximum_normalized_error=max_error,
            maximum_temperature_change_k=max_delta_t,
            roundoff_normalization_mol_per_m3=tuple(float(value) for value in roundoff),
        ),
    )
