"""Reusable vectorized finite-volume operators for reactive fields."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.reactive.case import ReactiveCaseV1, normalize_group_name
from microfluidics.reactive.errors import (
    ReactiveCaseValidationError,
    ReactiveTransportError,
    ReactiveWalltimeLimitError,
    TransportSubstepCapError,
)


@dataclass(frozen=True, slots=True)
class ReactiveSpatialPrecompute:
    mesh: ImportedTetraMesh
    face_volume_flux_m3_s: np.ndarray
    owner: np.ndarray
    neighbor: np.ndarray
    interior_faces: np.ndarray
    inlet_faces: np.ndarray
    outlet_faces: np.ndarray
    wall_faces: np.ndarray
    inlet_index_per_face: np.ndarray
    interior_conductance_m: np.ndarray
    inlet_conductance_m: np.ndarray
    outgoing_volume_rate_m3_s: np.ndarray
    diffusion_conductance_sum_m: np.ndarray
    outlet_backflow_m3_s: float


@dataclass(frozen=True, slots=True)
class ScalarStepDiagnostics:
    advective_in: float
    advective_out: float
    diffusive_in: float
    diffusive_out: float
    limiter_active_transfers: int
    limiter_min_scale: float
    roundoff_normalization_volume_weighted: float
    pairwise_conservation_error: float


@dataclass(frozen=True, slots=True)
class SpatialAdvanceDiagnostics:
    substeps: int
    stable_dt_s: float
    species_advective_in_mol: tuple[float, ...]
    species_advective_out_mol: tuple[float, ...]
    species_diffusive_in_mol: tuple[float, ...]
    species_diffusive_out_mol: tuple[float, ...]
    thermal_advective_in_k_m3: float
    thermal_advective_out_k_m3: float
    thermal_diffusive_in_k_m3: float
    thermal_diffusive_out_k_m3: float
    limiter_active_transfers: int
    limiter_min_scale: float
    roundoff_normalization_mol: tuple[float, ...]
    thermal_roundoff_normalization_k_m3: float
    pairwise_conservation_error: float
    outlet_backflow_m3: float


@dataclass(frozen=True, slots=True)
class SpatialAdvanceResult:
    concentrations_mol_per_m3: np.ndarray
    temperature_k: np.ndarray
    diagnostics: SpatialAdvanceDiagnostics


def _boundary_adjusted_conservation_error(
    delta_mass: np.ndarray,
    *,
    advective_in: float,
    advective_out: float,
    diffusive_in: float,
    diffusive_out: float,
) -> float:
    """Return internal-transfer closure after accounting for boundary exchange."""

    inventory_change = float(np.sum(delta_mass, dtype=np.float64))
    boundary_change = math.fsum(
        (advective_in, diffusive_in, -advective_out, -diffusive_out)
    )
    return abs(inventory_change - boundary_change)


def _face_set(values: np.ndarray) -> set[int]:
    return set(np.asarray(values, dtype=np.int64).reshape(-1).tolist())


def _check_walltime(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise ReactiveWalltimeLimitError(
            "reactive transport reached its walltime limit during spatial subcycling"
        )


def build_reactive_spatial_precompute(
    mesh: ImportedTetraMesh,
    face_volume_flux_m3_s: np.ndarray,
    case: ReactiveCaseV1,
) -> ReactiveSpatialPrecompute:
    """Validate mesh boundaries and precompute geometry used by every field."""

    if np.asarray(mesh.boundary_unresolved_faces).size:
        raise ReactiveCaseValidationError(
            "reactive transport forbids unresolved boundary faces"
        )
    q = np.asarray(face_volume_flux_m3_s, dtype=np.float64).reshape(-1)
    face_count = mesh.face_vertices.shape[0]
    if q.shape != (face_count,):
        raise ReactiveTransportError(
            "flow face-flux shape does not match the reactive mesh"
        )
    if not np.isfinite(q).all():
        raise ReactiveTransportError("flow face flux contains non-finite values")

    owner = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    neighbor = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    interior_faces = np.asarray(mesh.interior_face_indices, dtype=np.int64)
    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    boundary_faces = _face_set(mesh.boundary_face_indices)
    classified = (
        _face_set(inlet_faces) | _face_set(outlet_faces) | _face_set(wall_faces)
    )
    if boundary_faces != classified:
        missing = sorted(boundary_faces - classified)
        raise ReactiveCaseValidationError(
            f"reactive transport has unclassified boundary faces: {missing[:20]}"
        )

    inlet_by_normalized_name = {
        inlet.normalized_name: index for index, inlet in enumerate(case.inlets)
    }
    mesh_inlet_names: dict[str, str] = {}
    inlet_index_per_face = np.full(face_count, -1, dtype=np.int64)
    boundary_tags = np.asarray(mesh.boundary_tag_per_face, dtype=np.int32)
    for face_idx in inlet_faces.tolist():
        tag = int(boundary_tags[face_idx])
        raw_name = str(mesh.boundary_face_names.get(tag, "")).strip()
        normalized = normalize_group_name(raw_name)
        if not normalized:
            raise ReactiveCaseValidationError(
                f"inlet face {face_idx} has no resolvable physical-group name"
            )
        mesh_inlet_names[normalized] = raw_name
        if normalized not in inlet_by_normalized_name:
            raise ReactiveCaseValidationError(
                f"mesh inlet group {raw_name!r} is missing from reactive case"
            )
        inlet_index_per_face[face_idx] = inlet_by_normalized_name[normalized]
    unknown_case_groups = sorted(set(inlet_by_normalized_name) - set(mesh_inlet_names))
    if unknown_case_groups:
        raise ReactiveCaseValidationError(
            f"reactive case contains unknown inlet groups: {unknown_case_groups}"
        )

    c0 = owner[interior_faces]
    c1 = neighbor[interior_faces]
    delta = mesh.cell_centers[c1] - mesh.cell_centers[c0]
    distance = np.maximum(
        np.abs(
            np.einsum(
                "ij,ij->i",
                delta,
                np.asarray(mesh.face_normals[interior_faces], dtype=np.float64),
            )
        ),
        1e-20,
    )
    interior_conductance = (
        np.asarray(mesh.face_areas[interior_faces], dtype=np.float64) / distance
    )
    inlet_owner = owner[inlet_faces]
    inlet_delta = mesh.face_centers[inlet_faces] - mesh.cell_centers[inlet_owner]
    inlet_distance = np.maximum(
        np.abs(
            np.einsum(
                "ij,ij->i",
                inlet_delta,
                np.asarray(mesh.face_normals[inlet_faces], dtype=np.float64),
            )
        ),
        1e-20,
    )
    inlet_conductance = (
        np.asarray(mesh.face_areas[inlet_faces], dtype=np.float64) / inlet_distance
    )

    cell_count = mesh.tetrahedra.shape[0]
    outgoing = np.zeros(cell_count, dtype=np.float64)
    q_interior = q[interior_faces]
    donor = np.where(q_interior >= 0.0, c0, c1)
    np.add.at(outgoing, donor, np.abs(q_interior))
    boundary_out = np.concatenate(
        (
            inlet_faces[q[inlet_faces] > 0.0],
            outlet_faces[q[outlet_faces] > 0.0],
        )
    )
    if boundary_out.size:
        np.add.at(outgoing, owner[boundary_out], q[boundary_out])

    conductance_sum = np.zeros(cell_count, dtype=np.float64)
    np.add.at(conductance_sum, c0, interior_conductance)
    np.add.at(conductance_sum, c1, interior_conductance)
    if inlet_faces.size:
        np.add.at(conductance_sum, inlet_owner, inlet_conductance)
    outlet_backflow_rate = float(np.sum(np.maximum(-q[outlet_faces], 0.0)))
    return ReactiveSpatialPrecompute(
        mesh=mesh,
        face_volume_flux_m3_s=q,
        owner=owner,
        neighbor=neighbor,
        interior_faces=interior_faces,
        inlet_faces=inlet_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        inlet_index_per_face=inlet_index_per_face,
        interior_conductance_m=interior_conductance,
        inlet_conductance_m=inlet_conductance,
        outgoing_volume_rate_m3_s=outgoing,
        diffusion_conductance_sum_m=conductance_sum,
        outlet_backflow_m3_s=outlet_backflow_rate,
    )


def stable_spatial_dt(
    precompute: ReactiveSpatialPrecompute,
    *,
    diffusivity_m2_s: float,
    cfl_target: float,
    diffusion_stability_factor: float,
) -> float:
    """Return the explicit advection/diffusion stability limit."""

    volumes = np.asarray(precompute.mesh.cell_volumes, dtype=np.float64)
    advective_rate = np.divide(
        precompute.outgoing_volume_rate_m3_s,
        volumes,
        out=np.zeros_like(volumes),
        where=volumes > 0.0,
    )
    max_advective_rate = float(np.max(advective_rate))
    advective_dt = (
        cfl_target / max_advective_rate if max_advective_rate > 0.0 else math.inf
    )
    diffusion_rate = np.divide(
        diffusivity_m2_s * precompute.diffusion_conductance_sum_m,
        volumes,
        out=np.zeros_like(volumes),
        where=volumes > 0.0,
    )
    max_diffusion_rate = float(np.max(diffusion_rate))
    diffusion_dt = (
        diffusion_stability_factor / max_diffusion_rate
        if max_diffusion_rate > 0.0
        else math.inf
    )
    return min(advective_dt, diffusion_dt)


def required_spatial_substeps(
    precompute: ReactiveSpatialPrecompute,
    case: ReactiveCaseV1,
    outer_dt_s: float,
) -> tuple[int, float]:
    """Resolve a common stable spatial substep count for C and T."""

    diffusivities = list(case.material.species_diffusivity_m2_s)
    if case.mode in {"off", "nonisothermal"}:
        diffusivities.append(case.material.thermal_diffusivity_m2_s)
    limits = [
        stable_spatial_dt(
            precompute,
            diffusivity_m2_s=value,
            cfl_target=case.time.cfl_target,
            diffusion_stability_factor=case.time.diffusion_stability_factor,
        )
        for value in diffusivities
    ]
    stable_dt = min(limits, default=math.inf)
    substeps = (
        1 if math.isinf(stable_dt) else max(1, int(math.ceil(outer_dt_s / stable_dt)))
    )
    if substeps > case.time.max_transport_substeps:
        raise TransportSubstepCapError(
            "blocked_transport_substep_cap: required "
            f"{substeps}, configured maximum {case.time.max_transport_substeps}"
        )
    return substeps, stable_dt


def _append_transfers(
    donors: list[np.ndarray],
    receivers: list[np.ndarray],
    masses: list[np.ndarray],
    kinds: list[np.ndarray],
    donor: np.ndarray,
    receiver: np.ndarray,
    mass: np.ndarray,
    kind: int,
) -> None:
    if mass.size == 0:
        return
    donors.append(np.asarray(donor, dtype=np.int64))
    receivers.append(np.asarray(receiver, dtype=np.int64))
    masses.append(np.asarray(mass, dtype=np.float64))
    kinds.append(np.full(mass.shape, kind, dtype=np.int8))


def _advance_scalar(
    precompute: ReactiveSpatialPrecompute,
    values: np.ndarray,
    inlet_values: np.ndarray,
    *,
    diffusivity_m2_s: float,
    dt_s: float,
    lower_bound: float,
    upper_bound: float | None,
    walltime_deadline_monotonic: float | None = None,
) -> tuple[np.ndarray, ScalarStepDiagnostics]:
    _check_walltime(walltime_deadline_monotonic)
    mesh = precompute.mesh
    state = np.asarray(values, dtype=np.float64)
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    donors: list[np.ndarray] = []
    receivers: list[np.ndarray] = []
    masses: list[np.ndarray] = []
    kinds: list[np.ndarray] = []

    faces = precompute.interior_faces
    owner = precompute.owner[faces]
    neighbor = precompute.neighbor[faces]
    q = precompute.face_volume_flux_m3_s[faces]
    donor = np.where(q >= 0.0, owner, neighbor)
    receiver = np.where(q >= 0.0, neighbor, owner)
    _append_transfers(
        donors,
        receivers,
        masses,
        kinds,
        donor,
        receiver,
        dt_s * np.abs(q) * state[donor],
        0,
    )
    if diffusivity_m2_s > 0.0 and faces.size:
        signed_rate = (
            diffusivity_m2_s
            * precompute.interior_conductance_m
            * (state[owner] - state[neighbor])
        )
        donor = np.where(signed_rate >= 0.0, owner, neighbor)
        receiver = np.where(signed_rate >= 0.0, neighbor, owner)
        _append_transfers(
            donors,
            receivers,
            masses,
            kinds,
            donor,
            receiver,
            dt_s * np.abs(signed_rate),
            0,
        )

    faces = precompute.inlet_faces
    owner = precompute.owner[faces]
    q = precompute.face_volume_flux_m3_s[faces]
    configured = inlet_values[precompute.inlet_index_per_face[faces]]
    inflow = q < 0.0
    _append_transfers(
        donors,
        receivers,
        masses,
        kinds,
        np.full(np.count_nonzero(inflow), -1, dtype=np.int64),
        owner[inflow],
        dt_s * (-q[inflow]) * configured[inflow],
        1,
    )
    outflow = ~inflow
    _append_transfers(
        donors,
        receivers,
        masses,
        kinds,
        owner[outflow],
        np.full(np.count_nonzero(outflow), -1, dtype=np.int64),
        dt_s * q[outflow] * state[owner[outflow]],
        2,
    )
    if diffusivity_m2_s > 0.0 and faces.size:
        signed_rate = (
            diffusivity_m2_s
            * precompute.inlet_conductance_m
            * (state[owner] - configured)
        )
        outward = signed_rate >= 0.0
        _append_transfers(
            donors,
            receivers,
            masses,
            kinds,
            owner[outward],
            np.full(np.count_nonzero(outward), -1, dtype=np.int64),
            dt_s * signed_rate[outward],
            4,
        )
        inward = ~outward
        _append_transfers(
            donors,
            receivers,
            masses,
            kinds,
            np.full(np.count_nonzero(inward), -1, dtype=np.int64),
            owner[inward],
            dt_s * (-signed_rate[inward]),
            3,
        )

    faces = precompute.outlet_faces
    owner = precompute.owner[faces]
    q = precompute.face_volume_flux_m3_s[faces]
    outflow = q > 0.0
    _append_transfers(
        donors,
        receivers,
        masses,
        kinds,
        owner[outflow],
        np.full(np.count_nonzero(outflow), -1, dtype=np.int64),
        dt_s * q[outflow] * state[owner[outflow]],
        2,
    )

    if not masses:
        return np.array(state, copy=True), ScalarStepDiagnostics(
            advective_in=0.0,
            advective_out=0.0,
            diffusive_in=0.0,
            diffusive_out=0.0,
            limiter_active_transfers=0,
            limiter_min_scale=1.0,
            roundoff_normalization_volume_weighted=0.0,
            pairwise_conservation_error=0.0,
        )

    all_donor = np.concatenate(donors)
    all_receiver = np.concatenate(receivers)
    raw_mass = np.concatenate(masses)
    all_kind = np.concatenate(kinds)
    if not np.isfinite(raw_mass).all() or np.any(raw_mass < 0.0):
        raise ReactiveTransportError("spatial transfer produced invalid mass")

    internal_donor = all_donor >= 0
    internal_receiver = all_receiver >= 0
    scale = np.ones_like(raw_mass)
    scale_reference = max(1.0, float(np.max(np.abs(state))))
    if upper_bound is not None:
        scale_reference = max(scale_reference, abs(upper_bound))
    tolerance = 128.0 * np.finfo(np.float64).eps * scale_reference
    # Reduce only transfers that violate a final cell inventory. Recomputing
    # allowances from the already-limited opposite transfer makes the limiter
    # conservative on faces and preserves uniform through-flow states.
    # The correction propagates through connected transfers. Real tetrahedral
    # diagnostic flows can require substantially more than a handful of sweeps;
    # the strict postcondition below still fails closed if convergence is absent.
    for _ in range(256):
        _check_walltime(walltime_deadline_monotonic)
        trial_mass = raw_mass * scale
        outgoing = np.zeros_like(state)
        incoming = np.zeros_like(state)
        np.add.at(outgoing, all_donor[internal_donor], trial_mass[internal_donor])
        np.add.at(
            incoming,
            all_receiver[internal_receiver],
            trial_mass[internal_receiver],
        )
        trial_state = state + (incoming - outgoing) / volumes
        below = trial_state < lower_bound - tolerance
        above = (
            trial_state > upper_bound + tolerance
            if upper_bound is not None
            else np.zeros_like(trial_state, dtype=bool)
        )
        if not (np.any(below) or np.any(above)):
            break
        out_factor = np.ones_like(state)
        in_factor = np.ones_like(state)
        if np.any(below):
            allowed_out = np.maximum((state - lower_bound) * volumes + incoming, 0.0)
            candidate = np.ones_like(state)
            np.divide(
                allowed_out,
                outgoing,
                out=candidate,
                where=outgoing > 0.0,
            )
            out_factor[below] = np.minimum(candidate[below], 1.0)
        if upper_bound is not None and np.any(above):
            allowed_in = np.maximum((upper_bound - state) * volumes + outgoing, 0.0)
            candidate = np.ones_like(state)
            np.divide(
                allowed_in,
                incoming,
                out=candidate,
                where=incoming > 0.0,
            )
            in_factor[above] = np.minimum(candidate[above], 1.0)
        transfer_factor = np.ones_like(scale)
        transfer_factor[internal_donor] = np.minimum(
            transfer_factor[internal_donor],
            out_factor[all_donor[internal_donor]],
        )
        transfer_factor[internal_receiver] = np.minimum(
            transfer_factor[internal_receiver],
            in_factor[all_receiver[internal_receiver]],
        )
        scale *= transfer_factor

    limited_mass = raw_mass * scale
    delta_mass = np.zeros_like(state)
    np.add.at(delta_mass, all_donor[internal_donor], -limited_mass[internal_donor])
    np.add.at(
        delta_mass,
        all_receiver[internal_receiver],
        limited_mass[internal_receiver],
    )
    updated = state + delta_mass / volumes
    if float(np.min(updated)) < lower_bound - tolerance:
        raise ReactiveTransportError(
            "bounded spatial operator produced a negative state"
        )
    negative_roundoff = (updated < lower_bound) & (updated >= lower_bound - tolerance)
    roundoff = float(
        np.sum((lower_bound - updated[negative_roundoff]) * volumes[negative_roundoff])
    )
    updated[negative_roundoff] = lower_bound
    if upper_bound is not None and float(np.max(updated)) > upper_bound + tolerance:
        raise ReactiveTransportError(
            "bounded spatial operator exceeded its dynamic upper bound: "
            f"maximum={float(np.max(updated)):.17g}, "
            f"upper_bound={upper_bound:.17g}, tolerance={tolerance:.3g}"
        )

    advective_in = float(np.sum(limited_mass[all_kind == 1]))
    advective_out = float(np.sum(limited_mass[all_kind == 2]))
    diffusive_in = float(np.sum(limited_mass[all_kind == 3]))
    diffusive_out = float(np.sum(limited_mass[all_kind == 4]))
    pairwise_error = _boundary_adjusted_conservation_error(
        delta_mass,
        advective_in=advective_in,
        advective_out=advective_out,
        diffusive_in=diffusive_in,
        diffusive_out=diffusive_out,
    )
    return updated, ScalarStepDiagnostics(
        advective_in=advective_in,
        advective_out=advective_out,
        diffusive_in=diffusive_in,
        diffusive_out=diffusive_out,
        limiter_active_transfers=int(np.count_nonzero(scale < 1.0 - 1e-15)),
        limiter_min_scale=float(np.min(scale)) if scale.size else 1.0,
        roundoff_normalization_volume_weighted=roundoff,
        pairwise_conservation_error=pairwise_error,
    )


def advance_spatial_fields(
    precompute: ReactiveSpatialPrecompute,
    case: ReactiveCaseV1,
    concentrations_mol_per_m3: np.ndarray,
    temperature_k: np.ndarray,
    *,
    outer_dt_s: float,
    walltime_deadline_monotonic: float | None = None,
) -> SpatialAdvanceResult:
    """Advance all spatial fields over one shared outer interval."""

    concentrations = np.asarray(concentrations_mol_per_m3, dtype=np.float64).copy()
    temperature = np.asarray(temperature_k, dtype=np.float64).copy()
    if concentrations.shape != (
        precompute.mesh.tetrahedra.shape[0],
        len(case.species_names),
    ):
        raise ValueError("reactive concentration field has an invalid shape")
    if temperature.shape != (precompute.mesh.tetrahedra.shape[0],):
        raise ValueError("reactive temperature field has an invalid shape")
    substeps, stable_dt = required_spatial_substeps(precompute, case, outer_dt_s)
    substep_dt = outer_dt_s / substeps
    inlet_concentrations = np.asarray(
        [inlet.concentrations_mol_per_m3 for inlet in case.inlets],
        dtype=np.float64,
    )
    inlet_temperatures = np.asarray(
        [inlet.temperature_k for inlet in case.inlets], dtype=np.float64
    )
    species_adv_in = np.zeros(len(case.species_names), dtype=np.float64)
    species_adv_out = np.zeros_like(species_adv_in)
    species_diff_in = np.zeros_like(species_adv_in)
    species_diff_out = np.zeros_like(species_adv_in)
    roundoff_mol = np.zeros_like(species_adv_in)
    thermal_roundoff_k_m3 = 0.0
    thermal_adv_in = 0.0
    thermal_adv_out = 0.0
    thermal_diff_in = 0.0
    thermal_diff_out = 0.0
    limiter_active = 0
    limiter_min_scale = 1.0
    pairwise_error = 0.0

    for _ in range(substeps):
        _check_walltime(walltime_deadline_monotonic)
        for species_index, diffusivity in enumerate(
            case.material.species_diffusivity_m2_s
        ):
            upper_bound = max(
                float(np.max(concentrations[:, species_index])),
                float(np.max(inlet_concentrations[:, species_index])),
                np.finfo(np.float64).tiny,
            )
            updated, diagnostics = _advance_scalar(
                precompute,
                concentrations[:, species_index],
                inlet_concentrations[:, species_index],
                diffusivity_m2_s=diffusivity,
                dt_s=substep_dt,
                lower_bound=0.0,
                upper_bound=upper_bound,
                walltime_deadline_monotonic=walltime_deadline_monotonic,
            )
            concentrations[:, species_index] = updated
            species_adv_in[species_index] += diagnostics.advective_in
            species_adv_out[species_index] += diagnostics.advective_out
            species_diff_in[species_index] += diagnostics.diffusive_in
            species_diff_out[species_index] += diagnostics.diffusive_out
            roundoff_mol[species_index] += (
                diagnostics.roundoff_normalization_volume_weighted
            )
            limiter_active += diagnostics.limiter_active_transfers
            limiter_min_scale = min(limiter_min_scale, diagnostics.limiter_min_scale)
            pairwise_error = max(
                pairwise_error, diagnostics.pairwise_conservation_error
            )

        if case.mode in {"off", "nonisothermal"}:
            temperature, diagnostics = _advance_scalar(
                precompute,
                temperature,
                inlet_temperatures,
                diffusivity_m2_s=case.material.thermal_diffusivity_m2_s,
                dt_s=substep_dt,
                lower_bound=np.finfo(np.float64).tiny,
                # Unlike concentrations, temperature has no local maximum
                # principle in the reactive contract: reaction half-steps may
                # legitimately heat above or cool below inlet values.
                upper_bound=None,
                walltime_deadline_monotonic=walltime_deadline_monotonic,
            )
            thermal_adv_in += diagnostics.advective_in
            thermal_adv_out += diagnostics.advective_out
            thermal_diff_in += diagnostics.diffusive_in
            thermal_diff_out += diagnostics.diffusive_out
            thermal_roundoff_k_m3 += diagnostics.roundoff_normalization_volume_weighted
            limiter_active += diagnostics.limiter_active_transfers
            limiter_min_scale = min(limiter_min_scale, diagnostics.limiter_min_scale)
            pairwise_error = max(
                pairwise_error, diagnostics.pairwise_conservation_error
            )

    return SpatialAdvanceResult(
        concentrations_mol_per_m3=concentrations,
        temperature_k=temperature,
        diagnostics=SpatialAdvanceDiagnostics(
            substeps=substeps,
            stable_dt_s=stable_dt,
            species_advective_in_mol=tuple(float(value) for value in species_adv_in),
            species_advective_out_mol=tuple(float(value) for value in species_adv_out),
            species_diffusive_in_mol=tuple(float(value) for value in species_diff_in),
            species_diffusive_out_mol=tuple(float(value) for value in species_diff_out),
            thermal_advective_in_k_m3=thermal_adv_in,
            thermal_advective_out_k_m3=thermal_adv_out,
            thermal_diffusive_in_k_m3=thermal_diff_in,
            thermal_diffusive_out_k_m3=thermal_diff_out,
            limiter_active_transfers=limiter_active,
            limiter_min_scale=limiter_min_scale,
            roundoff_normalization_mol=tuple(float(value) for value in roundoff_mol),
            thermal_roundoff_normalization_k_m3=thermal_roundoff_k_m3,
            pairwise_conservation_error=pairwise_error,
            outlet_backflow_m3=precompute.outlet_backflow_m3_s * outer_dt_s,
        ),
    )
