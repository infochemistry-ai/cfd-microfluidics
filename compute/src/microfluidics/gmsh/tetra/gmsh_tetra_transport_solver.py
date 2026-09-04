"""Tetra-native transport prototype (advection / advection-diffusion) on imported mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    GradientMethod,
    LaplacianMethod,
    laplacian_scalar,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_backend import (
    BOUNDARY_GROUP_CODE,
    build_boundary_face_groups,
    build_inlet_zone_masks,
    build_scalar_backend_precompute,
    compute_outgoing_flux_sum as compute_scalar_outgoing_flux_sum,
    compute_cfl_metrics as compute_scalar_cfl_metrics,
    estimate_diffusion_dt_limit as estimate_scalar_diffusion_dt_limit,
    estimate_stable_dt as estimate_scalar_stable_dt,
    laplacian_numpy as scalar_laplacian_numpy,
    laplacian_torch as scalar_laplacian_torch,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
    build_diffusion_boundary_maps,
    build_diffusion_stencil,
    resolve_inlet_face_groups,
)
from microfluidics.gmsh.tetra.threshold_keys import (
    OUTLET_FRACTION_THRESHOLDS,
    format_threshold_key,
)

TransportMode = Literal["advection", "advection_diffusion"]
TransportScheme = Literal["upwind", "bounded_upwind"]


def _collect_transport_dt_candidates(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    eps: float = 1e-20,
) -> Dict[str, Any]:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    velocity = np.asarray(face_normal_velocity, dtype=np.float64)
    volumes_raw = np.asarray(mesh.cell_volumes, dtype=np.float64)
    volumes = np.maximum(volumes_raw, 1e-16)
    flux = velocity * areas
    interior = c1 >= 0
    contribution = np.zeros_like(flux)
    donor = np.full(flux.shape, -1, dtype=np.int64)
    interior_ids = np.flatnonzero(interior)
    if interior_ids.size:
        donor[interior_ids] = np.where(
            flux[interior_ids] >= 0.0, c0[interior_ids], c1[interior_ids]
        )
        contribution[interior_ids] = np.abs(flux[interior_ids])
    boundary_ids = np.flatnonzero((~interior) & (flux > 0.0))
    donor[boundary_ids] = c0[boundary_ids]
    contribution[boundary_ids] = flux[boundary_ids]
    outgoing = np.zeros(volumes.shape, dtype=np.float64)
    active_faces = donor >= 0
    np.add.at(outgoing, donor[active_faces], contribution[active_faces])
    active_cells = outgoing > float(eps)
    local_dt = np.full(volumes.shape, np.inf, dtype=np.float64)
    local_dt[active_cells] = volumes[active_cells] / outgoing[active_cells]
    ordered = np.flatnonzero(active_cells)[
        np.argsort(local_dt[active_cells], kind="stable")
    ]
    return {
        "c0": c0,
        "c1": c1,
        "areas": areas,
        "velocity": velocity,
        "volumes_raw": volumes_raw,
        "volumes": volumes,
        "flux": flux,
        "interior": interior,
        "donor": donor,
        "contribution": contribution,
        "outgoing": outgoing,
        "active_cells": active_cells,
        "local_dt": local_dt,
        "ordered": ordered,
    }


def _finite_positive_min(*values: float) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v)) and float(v) > 0.0]
    if not finite:
        return float("inf")
    return float(min(finite))


def build_transport_dt_limiter_audit(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    top_k: int = 10,
    eps: float = 1e-20,
) -> Dict[str, object]:
    """Return the cells and faces that constrain the explicit CFL time step."""
    dt_data = _collect_transport_dt_candidates(
        mesh,
        face_normal_velocity,
        eps=eps,
    )
    c0 = np.asarray(dt_data["c0"], dtype=np.int64)
    c1 = np.asarray(dt_data["c1"], dtype=np.int64)
    areas = np.asarray(dt_data["areas"], dtype=np.float64)
    velocity = np.asarray(dt_data["velocity"], dtype=np.float64)
    volumes_raw = np.asarray(dt_data["volumes_raw"], dtype=np.float64)
    volumes = np.asarray(dt_data["volumes"], dtype=np.float64)
    flux = np.asarray(dt_data["flux"], dtype=np.float64)
    interior = np.asarray(dt_data["interior"], dtype=bool)
    contribution = np.asarray(dt_data["contribution"], dtype=np.float64)
    donor = np.asarray(dt_data["donor"], dtype=np.int64)
    outgoing = np.asarray(dt_data["outgoing"], dtype=np.float64)
    active_cells = np.asarray(dt_data["active_cells"], dtype=bool)
    local_dt = np.asarray(dt_data["local_dt"], dtype=np.float64)
    ordered = np.asarray(dt_data["ordered"], dtype=np.int64)
    inlet = resolve_inlet_face_groups(mesh)
    groups = build_boundary_face_groups(
        mesh,
        left_faces=np.asarray(inlet["left_faces"], dtype=np.int64),
        right_faces=np.asarray(inlet["right_faces"], dtype=np.int64),
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
    )
    names = {code: name for name, code in BOUNDARY_GROUP_CODE.items()}
    median_volume = float(np.median(volumes)) if volumes.size else 0.0
    cells: list[Dict[str, object]] = []
    for cell_id in ordered[: max(0, int(top_k))].tolist():
        face_ids = np.flatnonzero(donor == cell_id)
        face_ids = face_ids[np.argsort(-contribution[face_ids], kind="stable")]
        faces = [
            {
                "face_id": int(face_id),
                "face_kind": "interior" if interior[face_id] else "boundary",
                "boundary_group": (
                    "interior"
                    if interior[face_id]
                    else names.get(int(groups[face_id]), "unresolved")
                ),
                "neighbour_cell_id": (
                    int(c1[face_id] if c0[face_id] == cell_id else c0[face_id])
                    if interior[face_id]
                    else None
                ),
                "face_area": float(areas[face_id]),
                "normal_velocity": float(velocity[face_id]),
                "signed_volume_flux": float(flux[face_id]),
                "outgoing_volume_flux_contribution": float(contribution[face_id]),
            }
            for face_id in face_ids.tolist()
        ]
        cells.append(
            {
                "cell_id": int(cell_id),
                "center_xyz": np.asarray(
                    mesh.cell_centers[cell_id], dtype=np.float64
                ).tolist(),
                "cell_volume_raw": float(volumes_raw[cell_id]),
                "cell_volume_used_for_cfl": float(volumes[cell_id]),
                "volume_ratio_to_median": float(
                    volumes[cell_id] / max(median_volume, 1e-30)
                ),
                "outgoing_volume_flux_sum": float(outgoing[cell_id]),
                "stable_dt": float(local_dt[cell_id]),
                "outgoing_face_count": int(face_ids.size),
                "outgoing_face_contributions": faces,
            }
        )
    return {
        "method": "explicit_advection_cfl_min_cell_volume_over_outgoing_flux",
        "stable_dt_estimate": float(local_dt[ordered[0]])
        if ordered.size
        else float("inf"),
        "active_outflow_cell_count": int(np.count_nonzero(active_cells)),
        "cell_volume_median": median_volume,
        "cell_volume_min": float(np.min(volumes)) if volumes.size else 0.0,
        "limiting_cell": cells[0] if cells else None,
        "top_limiting_cells": cells,
    }


@dataclass(frozen=True)
class GmshTetraTransportConfig:
    dt: float = 5e-4
    dt_mode: Literal["manual", "auto"] = "auto"
    cfl_target: float = 0.5
    # Retained only to deserialize earlier run configurations.  Auto mode now
    # always uses the strict local stability limit, so it never needs hidden
    # substeps.
    auto_dt_percentile: float = 5.0
    # Retained for backwards-compatible configuration parsing.  A configured
    # cap is reported but never prevents a numerically safe subcycled step.
    max_transport_substeps: int | None = None
    # Retained for backwards-compatible configuration parsing.  Any substep is
    # now user-visible, irrespective of this legacy threshold.
    transport_substep_warning_threshold: int = 32
    steps: int = 200
    progress_total_steps: int | None = None
    progress_target_end_time: float | None = None
    physical_time_start: float = 0.0
    transport_mode: TransportMode = "advection"
    transport_scheme: TransportScheme = "bounded_upwind"
    diffusivity: float = 3e-10
    cfl_limit: float = 0.8
    initial_value: float = 0.0
    left_inlet_value: float = 0.0
    right_inlet_value: float = 1.0
    source_term: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    clipping_enabled: bool = True
    safety_clamp_after_diffusion: bool = False
    boundedness_tolerance: float = 1e-6
    gradient_method: GradientMethod = "least_squares"
    laplacian_method: LaplacianMethod = "tpfa"
    progress_every: int = 20
    backend: Literal["numpy", "torch"] = "numpy"
    torch_device: str = "cpu"
    snapshot_steps: tuple[int, ...] = ()


BOUNDARY_GROUP_NAME = {code: name for name, code in BOUNDARY_GROUP_CODE.items()}


def _estimate_diffusion_dt_limit_from_stencil(
    *,
    volumes: np.ndarray,
    interior_c0: np.ndarray,
    interior_c1: np.ndarray,
    interior_coeff: np.ndarray,
    dirichlet_c0: np.ndarray,
    dirichlet_coeff: np.ndarray,
    diffusivity: float,
) -> float:
    if float(diffusivity) <= 0.0:
        return float("inf")
    coeff_sum = np.zeros(volumes.shape[0], dtype=np.float64)
    np.add.at(coeff_sum, np.asarray(interior_c0, dtype=np.int64), interior_coeff)
    np.add.at(coeff_sum, np.asarray(interior_c1, dtype=np.int64), interior_coeff)
    np.add.at(coeff_sum, np.asarray(dirichlet_c0, dtype=np.int64), dirichlet_coeff)
    positive = coeff_sum > 0.0
    if not np.any(positive):
        return float("inf")
    local_dt = np.asarray(volumes, dtype=np.float64)[positive] / (
        float(diffusivity) * coeff_sum[positive]
    )
    return float(0.5 * np.min(local_dt))


def compute_transport_diffusion_dt_limit(
    mesh: ImportedTetraMesh,
    config: GmshTetraTransportConfig,
) -> float:
    if config.transport_mode != "advection_diffusion" or config.diffusivity <= 0.0:
        return float("inf")
    inlet_groups = resolve_inlet_face_groups(mesh)
    boundary_dirichlet, boundary_no_flux = build_diffusion_boundary_maps(
        mesh,
        inlet_groups,
        left_value=config.left_inlet_value,
        right_value=config.right_inlet_value,
    )
    stencil = build_diffusion_stencil(
        mesh,
        boundary_dirichlet=boundary_dirichlet,
        boundary_no_flux_faces=boundary_no_flux,
    )
    return _estimate_diffusion_dt_limit_from_stencil(
        volumes=np.asarray(stencil.volumes, dtype=np.float64),
        interior_c0=np.asarray(stencil.interior_c0, dtype=np.int64),
        interior_c1=np.asarray(stencil.interior_c1, dtype=np.int64),
        interior_coeff=np.asarray(stencil.interior_coeff, dtype=np.float64),
        dirichlet_c0=np.asarray(stencil.dirichlet_c0, dtype=np.int64),
        dirichlet_coeff=np.asarray(stencil.dirichlet_coeff, dtype=np.float64),
        diffusivity=float(config.diffusivity),
    )


def resolve_transport_dt_control(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    config: GmshTetraTransportConfig,
    diffusion_dt_limit: float = float("inf"),
    eps: float = 1e-20,
) -> Dict[str, object]:
    if (
        not np.isfinite(diffusion_dt_limit)
        and config.transport_mode == "advection_diffusion"
        and config.diffusivity > 0.0
    ):
        diffusion_dt_limit = compute_transport_diffusion_dt_limit(mesh, config)
    dt_data = _collect_transport_dt_candidates(
        mesh,
        face_normal_velocity,
        eps=eps,
    )
    audit = build_transport_dt_limiter_audit(
        mesh,
        face_normal_velocity,
        eps=eps,
    )
    local_dt = np.asarray(dt_data["local_dt"], dtype=np.float64)
    active_cells = np.asarray(dt_data["active_cells"], dtype=bool)
    finite_local_dt = local_dt[active_cells & np.isfinite(local_dt)]
    stable_dt_estimate = float(audit["stable_dt_estimate"])
    percentile = float(np.clip(config.auto_dt_percentile, 0.0, 100.0))
    if finite_local_dt.size:
        stable_dt_percentile_estimate = float(
            np.percentile(finite_local_dt, percentile)
        )
    else:
        stable_dt_percentile_estimate = float("inf")
    stable_dt_robust_estimate = _finite_positive_min(
        stable_dt_percentile_estimate,
        diffusion_dt_limit,
    )
    if not np.isfinite(stable_dt_robust_estimate):
        stable_dt_robust_estimate = stable_dt_percentile_estimate

    explicit_stability_dt_limit = _finite_positive_min(
        stable_dt_estimate,
        diffusion_dt_limit,
    )
    requested_dt = float(config.dt)
    if config.dt_mode == "auto" and np.isfinite(explicit_stability_dt_limit):
        # ``auto`` must choose a step which is safe in every active cell.  The
        # percentile estimate remains diagnostic evidence only; choosing it as
        # an outer interval would reintroduce hidden subcycling.
        requested_outer_dt = float(config.cfl_target * explicit_stability_dt_limit)
        auto_dt_selection = "strict_stability_limit"
    else:
        requested_outer_dt = requested_dt
        auto_dt_selection = "manual_requested_dt"
    max_transport_substeps = (
        None
        if config.max_transport_substeps is None
        else int(config.max_transport_substeps)
    )
    if not np.isfinite(explicit_stability_dt_limit):
        required_substep_count = 1
        dt_substep = requested_outer_dt
    else:
        required_substep_count = max(
            1,
            int(
                np.ceil(
                    max(float(requested_outer_dt), 0.0)
                    / max(float(explicit_stability_dt_limit), 1e-30)
                    - 1e-12
                )
            ),
        )
        dt_substep = float(requested_outer_dt / max(1, required_substep_count))

    warning_hit = required_substep_count > 1
    cap_hit = (
        max_transport_substeps is not None
        and required_substep_count > max_transport_substeps
    )
    blocked = False
    actual_advanced_dt = requested_outer_dt
    executed_substep_count = required_substep_count
    if required_substep_count == 1:
        status = "ok"
        reason = "transport dt resolved without substepping"
    else:
        status = "warning_substepping"
        reason = (
            "requested transport dt is advanced through stable explicit substeps: "
            f"requested_outer_dt={requested_outer_dt:.6e}, "
            f"required_substeps={required_substep_count}, "
            f"dt_substep={dt_substep:.6e}, "
            f"explicit_stability_dt_limit={explicit_stability_dt_limit:.6e}"
        )
        if cap_hit:
            reason += (
                f"; legacy configured substep cap={max_transport_substeps} is "
                "informational and does not block the run"
            )

    return {
        "dt_mode": config.dt_mode,
        "configured_dt": requested_dt,
        "requested_dt": float(requested_outer_dt),
        "requested_outer_dt": float(requested_outer_dt),
        "nominal_used_dt": float(requested_outer_dt),
        "used_dt": float(actual_advanced_dt),
        "actual_advanced_dt": float(actual_advanced_dt),
        "dt_substep": float(dt_substep),
        "stable_dt_estimate": float(stable_dt_estimate),
        "stable_dt_percentile_estimate": float(stable_dt_percentile_estimate),
        "stable_dt_robust_estimate": float(stable_dt_robust_estimate),
        "diffusion_dt_limit": float(diffusion_dt_limit),
        "explicit_stability_dt_limit": float(explicit_stability_dt_limit),
        "transport_substep_count": int(executed_substep_count),
        "required_transport_substep_count": int(required_substep_count),
        "executed_transport_substep_count": int(executed_substep_count),
        "nominal_transport_substep_count": int(required_substep_count),
        "transport_substep_warning_threshold": 1,
        "transport_substep_warning": bool(warning_hit),
        "transport_substep_cap_hit": bool(cap_hit),
        "transport_dt_reduced": False,
        "max_transport_substeps": max_transport_substeps,
        "auto_dt_percentile": float(percentile),
        "auto_dt_selection": auto_dt_selection,
        "transport_dt_controller_status": str(status),
        "transport_dt_controller_blocked": bool(blocked),
        "transport_dt_controller_reason": str(reason),
        "dt_limiter_audit": audit,
    }


def _build_boundary_face_groups(
    mesh: ImportedTetraMesh,
    *,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> np.ndarray:
    return build_boundary_face_groups(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )


def _build_inlet_zone_masks(
    mesh: ImportedTetraMesh,
    *,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return build_inlet_zone_masks(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        wall_faces=wall_faces,
    )


def _compute_outgoing_flux_sum(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
) -> np.ndarray:
    precompute = build_scalar_backend_precompute(
        mesh,
        face_normal_velocity,
        backend="numpy",
    )
    return compute_scalar_outgoing_flux_sum(precompute)


def _compute_cfl_metrics(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    precompute = build_scalar_backend_precompute(
        mesh,
        face_normal_velocity,
        backend="numpy",
    )
    return compute_scalar_cfl_metrics(precompute, dt)


def estimate_stable_dt(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    eps: float = 1e-20,
) -> float:
    """Estimate explicit stability dt from outgoing flux / cell volume."""
    precompute = build_scalar_backend_precompute(
        mesh,
        face_normal_velocity,
        backend="numpy",
    )
    return estimate_scalar_stable_dt(precompute, eps=eps)


def _top_cell_metrics(
    values: np.ndarray, *, top_k: int = 10
) -> list[Dict[str, float | int]]:
    if values.size == 0:
        return []
    order = np.argsort(values)[::-1][:top_k]
    return [{"cell_index": int(idx), "value": float(values[idx])} for idx in order]


def _assemble_advection_fluxes(
    mesh: ImportedTetraMesh,
    scalar: np.ndarray,
    face_normal_velocity: np.ndarray,
    *,
    boundary_face_groups: np.ndarray,
    left_inlet_value: float,
    right_inlet_value: float,
    dt: float,
) -> Dict[str, np.ndarray | float]:
    """Assemble upwind scalar fluxes and per-cell incoming/outgoing masses."""

    n_cells = mesh.tetrahedra.shape[0]
    n_faces = mesh.face_vertices.shape[0]
    c0 = mesh.face_to_cells[:, 0]
    c1 = mesh.face_to_cells[:, 1]
    area = mesh.face_areas

    out_mass = np.zeros(
        n_cells, dtype=np.float64
    )  # mass of transported scalar for this dt
    in_mass = np.zeros(
        n_cells, dtype=np.float64
    )  # mass of transported scalar for this dt
    boundary_in_mass = np.zeros(
        n_cells, dtype=np.float64
    )  # scalar mass inflow for this dt
    boundary_out_mass = np.zeros(
        n_cells, dtype=np.float64
    )  # scalar mass outflow for this dt
    out_vol_rate = np.zeros(n_cells, dtype=np.float64)  # m^3/s
    in_vol_rate = np.zeros(n_cells, dtype=np.float64)  # m^3/s

    # Signed scalar volumetric transport rate through face [m^3/s].
    # Positive sign means outward w.r.t owner cell.
    face_scalar_flux_raw = np.zeros(n_faces, dtype=np.float64)
    face_volumetric_flux_q = np.zeros(n_faces, dtype=np.float64)  # [m^3/s]
    face_upwind_scalar = np.zeros(n_faces, dtype=np.float64)
    face_mass_delta_owner = np.zeros(n_faces, dtype=np.float64)  # [m^3] over dt
    face_mass_delta_neighbor = np.zeros(n_faces, dtype=np.float64)  # [m^3] over dt
    face_contribution_c_owner = np.zeros(n_faces, dtype=np.float64)  # delta C over dt
    face_contribution_c_neighbor = np.zeros(
        n_faces, dtype=np.float64
    )  # delta C over dt
    face_owner = c0.astype(np.int64, copy=True)
    face_neighbor = c1.astype(np.int64, copy=True)
    face_boundary_group = boundary_face_groups.astype(np.int32, copy=True)

    inlet_scalar_flux_in = 0.0
    outlet_scalar_flux_out = 0.0

    interior_faces = []
    donor_cells = []
    recv_cells = []
    transfer_mass_raw = []

    boundary_in_faces = []
    boundary_out_faces = []
    boundary_in_cells = []
    boundary_out_cells = []
    boundary_in_mass_list = []
    boundary_out_mass_list = []

    for face_idx in range(n_faces):
        owner = int(c0[face_idx])
        neigh = int(c1[face_idx])
        vn = float(face_normal_velocity[face_idx])
        a = float(area[face_idx])
        q = vn * a
        group = int(boundary_face_groups[face_idx])
        face_volumetric_flux_q[face_idx] = q

        if neigh >= 0:
            if vn >= 0.0:
                donor = owner
                recv = neigh
                c_up = float(scalar[owner])
            else:
                donor = neigh
                recv = owner
                c_up = float(scalar[neigh])
            phi = q * c_up
            m = dt * abs(phi)
            face_scalar_flux_raw[face_idx] = phi
            face_upwind_scalar[face_idx] = c_up
            out_mass[donor] += m
            in_mass[recv] += m
            out_vol_rate[donor] += abs(q)
            in_vol_rate[recv] += abs(q)
            face_mass_delta_owner[face_idx] = -dt * phi
            face_mass_delta_neighbor[face_idx] = dt * phi
            face_contribution_c_owner[face_idx] = face_mass_delta_owner[face_idx] / max(
                mesh.cell_volumes[owner], 1e-16
            )
            face_contribution_c_neighbor[face_idx] = face_mass_delta_neighbor[
                face_idx
            ] / max(mesh.cell_volumes[neigh], 1e-16)
            interior_faces.append(face_idx)
            donor_cells.append(donor)
            recv_cells.append(recv)
            transfer_mass_raw.append(m)
            continue

        # Boundary faces
        if group == BOUNDARY_GROUP_CODE["wall"]:
            face_scalar_flux_raw[face_idx] = 0.0
            face_upwind_scalar[face_idx] = float(scalar[owner])
            continue

        if group == BOUNDARY_GROUP_CODE["left_inlet"]:
            if vn < 0.0:
                c_face = float(left_inlet_value)
                phi = q * c_face
                m = -dt * phi
                face_scalar_flux_raw[face_idx] = phi
                face_upwind_scalar[face_idx] = c_face
                in_mass[owner] += m
                boundary_in_mass[owner] += m
                in_vol_rate[owner] += abs(q)
                inlet_scalar_flux_in += -phi
                face_mass_delta_owner[face_idx] = -dt * phi
                face_contribution_c_owner[face_idx] = face_mass_delta_owner[
                    face_idx
                ] / max(mesh.cell_volumes[owner], 1e-16)
                boundary_in_faces.append(face_idx)
                boundary_in_cells.append(owner)
                boundary_in_mass_list.append(m)
            else:
                c_face = float(scalar[owner])
                phi = q * c_face
                m = dt * phi
                face_scalar_flux_raw[face_idx] = phi
                face_upwind_scalar[face_idx] = c_face
                out_mass[owner] += m
                boundary_out_mass[owner] += m
                out_vol_rate[owner] += abs(q)
                face_mass_delta_owner[face_idx] = -dt * phi
                face_contribution_c_owner[face_idx] = face_mass_delta_owner[
                    face_idx
                ] / max(mesh.cell_volumes[owner], 1e-16)
                boundary_out_faces.append(face_idx)
                boundary_out_cells.append(owner)
                boundary_out_mass_list.append(m)
            continue

        if group == BOUNDARY_GROUP_CODE["right_inlet"]:
            if vn < 0.0:
                c_face = float(right_inlet_value)
                phi = q * c_face
                m = -dt * phi
                face_scalar_flux_raw[face_idx] = phi
                face_upwind_scalar[face_idx] = c_face
                in_mass[owner] += m
                boundary_in_mass[owner] += m
                in_vol_rate[owner] += abs(q)
                inlet_scalar_flux_in += -phi
                face_mass_delta_owner[face_idx] = -dt * phi
                face_contribution_c_owner[face_idx] = face_mass_delta_owner[
                    face_idx
                ] / max(mesh.cell_volumes[owner], 1e-16)
                boundary_in_faces.append(face_idx)
                boundary_in_cells.append(owner)
                boundary_in_mass_list.append(m)
            else:
                c_face = float(scalar[owner])
                phi = q * c_face
                m = dt * phi
                face_scalar_flux_raw[face_idx] = phi
                face_upwind_scalar[face_idx] = c_face
                out_mass[owner] += m
                boundary_out_mass[owner] += m
                out_vol_rate[owner] += abs(q)
                face_mass_delta_owner[face_idx] = -dt * phi
                face_contribution_c_owner[face_idx] = face_mass_delta_owner[
                    face_idx
                ] / max(mesh.cell_volumes[owner], 1e-16)
                boundary_out_faces.append(face_idx)
                boundary_out_cells.append(owner)
                boundary_out_mass_list.append(m)
            continue

        if group == BOUNDARY_GROUP_CODE["outlet"]:
            if vn > 0.0:
                c_face = float(scalar[owner])
                phi = q * c_face
                m = dt * phi
                face_scalar_flux_raw[face_idx] = phi
                face_upwind_scalar[face_idx] = c_face
                out_mass[owner] += m
                boundary_out_mass[owner] += m
                out_vol_rate[owner] += abs(q)
                outlet_scalar_flux_out += phi
                face_mass_delta_owner[face_idx] = -dt * phi
                face_contribution_c_owner[face_idx] = face_mass_delta_owner[
                    face_idx
                ] / max(mesh.cell_volumes[owner], 1e-16)
                boundary_out_faces.append(face_idx)
                boundary_out_cells.append(owner)
                boundary_out_mass_list.append(m)
            else:
                face_scalar_flux_raw[face_idx] = 0.0
                face_upwind_scalar[face_idx] = float(scalar[owner])
            continue

        # Unresolved boundaries: allow outward convective flux only.
        if vn > 0.0:
            c_face = float(scalar[owner])
            phi = q * c_face
            m = dt * phi
            face_scalar_flux_raw[face_idx] = phi
            face_upwind_scalar[face_idx] = c_face
            out_mass[owner] += m
            boundary_out_mass[owner] += m
            out_vol_rate[owner] += abs(q)
            face_mass_delta_owner[face_idx] = -dt * phi
            face_contribution_c_owner[face_idx] = face_mass_delta_owner[face_idx] / max(
                mesh.cell_volumes[owner], 1e-16
            )
            boundary_out_faces.append(face_idx)
            boundary_out_cells.append(owner)
            boundary_out_mass_list.append(m)
        else:
            face_scalar_flux_raw[face_idx] = 0.0
            face_upwind_scalar[face_idx] = float(scalar[owner])

    interior_faces_arr = np.asarray(interior_faces, dtype=np.int64)
    pairwise_err = np.abs(
        face_mass_delta_owner[interior_faces_arr]
        + face_mass_delta_neighbor[interior_faces_arr]
    )
    pairwise_max = float(np.max(pairwise_err)) if pairwise_err.size else 0.0
    pairwise_mean = float(np.mean(pairwise_err)) if pairwise_err.size else 0.0

    return {
        "face_scalar_flux_raw": face_scalar_flux_raw,
        "face_volumetric_flux_q": face_volumetric_flux_q,
        "face_upwind_scalar": face_upwind_scalar,
        "face_mass_delta_owner": face_mass_delta_owner,
        "face_mass_delta_neighbor": face_mass_delta_neighbor,
        "face_contribution_c_owner": face_contribution_c_owner,
        "face_contribution_c_neighbor": face_contribution_c_neighbor,
        "face_owner": face_owner,
        "face_neighbor": face_neighbor,
        "face_boundary_group": face_boundary_group,
        "out_mass_raw": out_mass,
        "in_mass_raw": in_mass,
        "out_vol_rate_raw": out_vol_rate,
        "in_vol_rate_raw": in_vol_rate,
        "boundary_in_mass_raw": boundary_in_mass,
        "boundary_out_mass_raw": boundary_out_mass,
        "inlet_scalar_flux_in": inlet_scalar_flux_in,
        "outlet_scalar_flux_out": outlet_scalar_flux_out,
        "interior_faces": interior_faces_arr,
        "donor_cells": np.asarray(donor_cells, dtype=np.int64),
        "recv_cells": np.asarray(recv_cells, dtype=np.int64),
        "transfer_mass_raw": np.asarray(transfer_mass_raw, dtype=np.float64),
        "boundary_in_faces": np.asarray(boundary_in_faces, dtype=np.int64),
        "boundary_out_faces": np.asarray(boundary_out_faces, dtype=np.int64),
        "boundary_in_cells": np.asarray(boundary_in_cells, dtype=np.int64),
        "boundary_out_cells": np.asarray(boundary_out_cells, dtype=np.int64),
        "boundary_in_mass_list": np.asarray(boundary_in_mass_list, dtype=np.float64),
        "boundary_out_mass_list": np.asarray(boundary_out_mass_list, dtype=np.float64),
        "pairwise_conservation_error_max_abs": pairwise_max,
        "pairwise_conservation_error_mean_abs": pairwise_mean,
    }


def _apply_bounded_limiter(
    mesh: ImportedTetraMesh,
    scalar: np.ndarray,
    assembled: Dict[str, np.ndarray | float],
    *,
    scheme: TransportScheme,
    dt: float,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> Dict[str, np.ndarray | float]:
    """Apply bounded flux limiter in mass space before the FV update."""

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        raise ValueError("Limiter bounds must be finite.")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound.")

    n_cells = mesh.tetrahedra.shape[0]
    volumes = np.maximum(mesh.cell_volumes, 1e-16)

    face_flux_raw = np.asarray(assembled["face_scalar_flux_raw"], dtype=np.float64)
    out_raw = np.asarray(assembled["out_mass_raw"], dtype=np.float64)
    in_raw = np.asarray(assembled["in_mass_raw"], dtype=np.float64)
    interior_faces = np.asarray(assembled["interior_faces"], dtype=np.int64)
    donor_cells = np.asarray(assembled["donor_cells"], dtype=np.int64)
    recv_cells = np.asarray(assembled["recv_cells"], dtype=np.int64)
    transfer_mass_raw = np.asarray(assembled["transfer_mass_raw"], dtype=np.float64)
    b_in_faces = np.asarray(assembled["boundary_in_faces"], dtype=np.int64)
    b_out_faces = np.asarray(assembled["boundary_out_faces"], dtype=np.int64)
    b_in_cells = np.asarray(assembled["boundary_in_cells"], dtype=np.int64)
    b_out_cells = np.asarray(assembled["boundary_out_cells"], dtype=np.int64)
    b_in_mass_raw = np.asarray(assembled["boundary_in_mass_list"], dtype=np.float64)
    b_out_mass_raw = np.asarray(assembled["boundary_out_mass_list"], dtype=np.float64)
    face_boundary_group = np.asarray(assembled["face_boundary_group"], dtype=np.int32)

    mass = (scalar - lower_bound) * volumes
    capacity = (upper_bound - scalar) * volumes
    mass = np.maximum(mass, 0.0)
    capacity = np.maximum(capacity, 0.0)

    raw_state = scalar + (in_raw - out_raw) / volumes
    raw_overshoot = np.maximum(raw_state - upper_bound, 0.0)
    raw_undershoot = np.maximum(lower_bound - raw_state, 0.0)

    face_flux_limited = np.array(face_flux_raw, copy=True)
    out_limited = np.array(out_raw, copy=True)
    in_limited = np.array(in_raw, copy=True)

    limiter_scale_out = np.ones(n_cells, dtype=np.float64)
    limiter_scale_in = np.ones(n_cells, dtype=np.float64)
    limiter_pair = np.ones(interior_faces.shape[0], dtype=np.float64)
    limiter_b_in = np.ones(b_in_faces.shape[0], dtype=np.float64)
    limiter_b_out = np.ones(b_out_faces.shape[0], dtype=np.float64)
    bounded_enabled = scheme == "bounded_upwind"

    if bounded_enabled:
        with np.errstate(divide="ignore", invalid="ignore"):
            limiter_scale_out = np.where(
                out_raw > 0.0, np.minimum(1.0, mass / out_raw), 1.0
            )
            limiter_scale_in = np.where(
                in_raw > 0.0, np.minimum(1.0, capacity / in_raw), 1.0
            )

        out_limited = np.zeros_like(out_raw)
        in_limited = np.zeros_like(in_raw)
        face_flux_limited[:] = 0.0

        # Interior transfers: conservative pairwise limiter.
        for i in range(interior_faces.shape[0]):
            donor = int(donor_cells[i])
            recv = int(recv_cells[i])
            f_idx = int(interior_faces[i])
            s = float(min(limiter_scale_out[donor], limiter_scale_in[recv]))
            limiter_pair[i] = s
            m_lim = s * float(transfer_mass_raw[i])
            out_limited[donor] += m_lim
            in_limited[recv] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

        # Boundary inflow (outside -> cell): limited by incoming capacity.
        for i in range(b_in_faces.shape[0]):
            cell = int(b_in_cells[i])
            f_idx = int(b_in_faces[i])
            s = float(limiter_scale_in[cell])
            limiter_b_in[i] = s
            m_lim = s * float(b_in_mass_raw[i])
            in_limited[cell] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

        # Boundary outflow (cell -> outside): limited by available mass.
        for i in range(b_out_faces.shape[0]):
            cell = int(b_out_cells[i])
            f_idx = int(b_out_faces[i])
            s = float(limiter_scale_out[cell])
            limiter_b_out[i] = s
            m_lim = s * float(b_out_mass_raw[i])
            out_limited[cell] += m_lim
            face_flux_limited[f_idx] = s * face_flux_raw[f_idx]

        # Faces with no active transfer remain zero, keep unchanged physical sign
        # where zero is expected.

    limited_state = scalar + (in_limited - out_limited) / volumes
    limited_overshoot = np.maximum(limited_state - upper_bound, 0.0)
    limited_undershoot = np.maximum(lower_bound - limited_state, 0.0)

    limiter_active = (limiter_scale_out < 1.0 - 1e-15) | (
        limiter_scale_in < 1.0 - 1e-15
    )
    limiter_active_count = int(np.count_nonzero(limiter_active))
    limiter_active_fraction = float(limiter_active_count / max(1, n_cells))
    min_scale = float(
        min(
            np.min(limiter_scale_out) if limiter_scale_out.size else 1.0,
            np.min(limiter_scale_in) if limiter_scale_in.size else 1.0,
        )
    )
    limiter_max_flux_scale_down = float(1.0 - min_scale)
    limiter_mass_correction_total = float(
        np.sum(np.abs((in_raw - out_raw) - (in_limited - out_limited)))
    )

    total_raw = float(np.sum(in_raw) - np.sum(out_raw))
    total_limited = float(np.sum(in_limited) - np.sum(out_limited))
    conservation_error = float(total_limited - total_raw)

    inlet_face_mask = (face_boundary_group == BOUNDARY_GROUP_CODE["left_inlet"]) | (
        face_boundary_group == BOUNDARY_GROUP_CODE["right_inlet"]
    )
    outlet_face_mask = face_boundary_group == BOUNDARY_GROUP_CODE["outlet"]
    raw_inlet_scalar_flux_in = float(
        -np.sum(np.minimum(face_flux_raw[inlet_face_mask], 0.0))
    )
    raw_outlet_scalar_flux_out = float(
        np.sum(np.maximum(face_flux_raw[outlet_face_mask], 0.0))
    )
    limited_inlet_scalar_flux_in = float(
        -np.sum(np.minimum(face_flux_limited[inlet_face_mask], 0.0))
    )
    limited_outlet_scalar_flux_out = float(
        np.sum(np.maximum(face_flux_limited[outlet_face_mask], 0.0))
    )

    return {
        "bounded_limiter_enabled": bounded_enabled,
        "face_scalar_flux_raw": face_flux_raw,
        "face_scalar_flux_limited": face_flux_limited,
        "out_mass_raw": out_raw,
        "in_mass_raw": in_raw,
        "out_mass_limited": out_limited,
        "in_mass_limited": in_limited,
        "raw_state_before_limiter": raw_state,
        "limited_state_before_clip": limited_state,
        "raw_overshoot": raw_overshoot,
        "raw_undershoot": raw_undershoot,
        "limited_overshoot": limited_overshoot,
        "limited_undershoot": limited_undershoot,
        "limiter_scale_out": limiter_scale_out,
        "limiter_scale_in": limiter_scale_in,
        "limiter_pair": limiter_pair,
        "limiter_active_count": limiter_active_count,
        "limiter_active_fraction": limiter_active_fraction,
        "limiter_max_flux_scale_down": limiter_max_flux_scale_down,
        "limiter_mass_correction_total": limiter_mass_correction_total,
        "conservation_error_after_limiter": conservation_error,
        "raw_inlet_scalar_flux_in": raw_inlet_scalar_flux_in,
        "raw_outlet_scalar_flux_out": raw_outlet_scalar_flux_out,
        "effective_inlet_scalar_flux_in": limited_inlet_scalar_flux_in,
        "effective_outlet_scalar_flux_out": limited_outlet_scalar_flux_out,
    }


def _run_tetra_transport_debug_torch(
    mesh: ImportedTetraMesh,
    config: GmshTetraTransportConfig,
    *,
    face_normal_velocity: np.ndarray,
    flux_diagnostics: Dict[str, float],
    initial_scalar: np.ndarray | None = None,
    start_step: int = 0,
    initial_cumulative_mass_in: float = 0.0,
    initial_cumulative_mass_out: float = 0.0,
    diffusion_boundary_kwargs: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Torch backend transport stepper for advection and advection-diffusion."""
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Torch backend requested but torch is not installed."
        ) from exc

    device = torch.device(config.torch_device)
    dtype = torch.float64
    n_cells = mesh.tetrahedra.shape[0]
    n_faces = mesh.face_vertices.shape[0]

    inlet_groups = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet_groups["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet_groups["right_faces"], dtype=np.int64)
    left_cells = np.asarray(inlet_groups["left_cells"], dtype=np.int64)
    right_cells = np.asarray(inlet_groups["right_cells"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    outlet_cells = (
        np.unique(mesh.face_to_cells[outlet_faces, 0])
        if outlet_faces.size
        else np.zeros((0,), dtype=np.int64)
    )

    boundary_face_groups_np = _build_boundary_face_groups(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    left_zone_mask_np, right_zone_mask_np, corner_zone_mask_np = (
        _build_inlet_zone_masks(
            mesh,
            left_faces=left_faces,
            right_faces=right_faces,
            wall_faces=wall_faces,
        )
    )

    # Core geometry tensors.
    c0 = torch.as_tensor(mesh.face_to_cells[:, 0], dtype=torch.long, device=device)
    c1 = torch.as_tensor(mesh.face_to_cells[:, 1], dtype=torch.long, device=device)
    area = torch.as_tensor(mesh.face_areas, dtype=dtype, device=device)
    vn = torch.as_tensor(face_normal_velocity, dtype=dtype, device=device)
    q_all = vn * area
    groups = torch.as_tensor(boundary_face_groups_np, dtype=torch.long, device=device)
    vol = torch.as_tensor(
        np.maximum(mesh.cell_volumes, 1e-16), dtype=dtype, device=device
    )

    face_ids = torch.arange(n_faces, device=device, dtype=torch.long)
    interior_mask = c1 >= 0
    interior_faces = face_ids[interior_mask]
    owner_i = c0[interior_faces]
    neigh_i = c1[interior_faces]
    q_i = q_all[interior_faces]
    q_i_pos = q_i >= 0.0
    donor_i = torch.where(q_i_pos, owner_i, neigh_i)
    recv_i = torch.where(q_i_pos, neigh_i, owner_i)

    left_faces_t = torch.as_tensor(left_faces, dtype=torch.long, device=device)
    right_faces_t = torch.as_tensor(right_faces, dtype=torch.long, device=device)
    outlet_faces_t = torch.as_tensor(outlet_faces, dtype=torch.long, device=device)
    boundary_faces_t = torch.as_tensor(
        mesh.boundary_face_indices, dtype=torch.long, device=device
    )
    unresolved_faces_t = boundary_faces_t[
        groups[boundary_faces_t] == BOUNDARY_GROUP_CODE["unresolved"]
    ]

    left_cells_t = torch.as_tensor(left_cells, dtype=torch.long, device=device)
    right_cells_t = torch.as_tensor(right_cells, dtype=torch.long, device=device)
    outlet_cells_t = torch.as_tensor(outlet_cells, dtype=torch.long, device=device)
    left_zone_mask_t = torch.as_tensor(
        left_zone_mask_np, dtype=torch.bool, device=device
    )
    right_zone_mask_t = torch.as_tensor(
        right_zone_mask_np, dtype=torch.bool, device=device
    )
    corner_zone_mask_t = torch.as_tensor(
        corner_zone_mask_np, dtype=torch.bool, device=device
    )

    # Diffusion precompute (torch-native) for advection_diffusion mode.
    diffusion_data: Dict[str, torch.Tensor] | None = None
    generic_diffusion_precompute = None
    diffusion_dt_limit = float("inf")
    if config.transport_mode == "advection_diffusion":
        boundary_dirichlet, boundary_no_flux = build_diffusion_boundary_maps(
            mesh,
            inlet_groups,
            left_value=config.left_inlet_value,
            right_value=config.right_inlet_value,
        )
        if diffusion_boundary_kwargs:
            generic_diffusion_precompute = build_scalar_backend_precompute(
                mesh,
                face_normal_velocity,
                backend="torch",
                torch_device=config.torch_device,
                **dict(diffusion_boundary_kwargs),
            )
            stencil = generic_diffusion_precompute.diffusion_stencil
            assert stencil is not None
        else:
            stencil = build_diffusion_stencil(
                mesh,
                boundary_dirichlet=boundary_dirichlet,
                boundary_no_flux_faces=boundary_no_flux,
            )
        diffusion_dt_limit = estimate_scalar_diffusion_dt_limit(
            stencil, float(config.diffusivity)
        )

        # Least-squares gradient precompute.
        reg = 1e-14
        mat = np.zeros((n_cells, 3, 3), dtype=np.float64)
        eq_count = np.zeros((n_cells,), dtype=np.int32)
        c0_i = np.asarray(stencil.interior_c0, dtype=np.int64)
        c1_i = np.asarray(stencil.interior_c1, dtype=np.int64)
        delta_x = mesh.cell_centers[c1_i] - mesh.cell_centers[c0_i]
        w_i = 1.0 / np.maximum(np.einsum("ij,ij->i", delta_x, delta_x), 1e-20)
        for row in range(c0_i.shape[0]):
            i0 = int(c0_i[row])
            i1 = int(c1_i[row])
            r = delta_x[row]
            w = float(w_i[row])
            outer = w * np.outer(r, r)
            mat[i0] += outer
            mat[i1] += outer
            eq_count[i0] += 1
            eq_count[i1] += 1
        db_owner = np.asarray(stencil.dirichlet_c0, dtype=np.int64)
        db_delta = np.asarray(stencil.dirichlet_cell_to_face_delta, dtype=np.float64)
        db_value = np.asarray(stencil.dirichlet_value, dtype=np.float64)
        db_w = np.zeros(db_owner.shape[0], dtype=np.float64)
        for row in range(db_owner.shape[0]):
            owner = int(db_owner[row])
            r = db_delta[row]
            norm2 = float(np.dot(r, r))
            if norm2 <= 1e-20:
                continue
            w = 1.0 / norm2
            db_w[row] = w
            mat[owner] += w * np.outer(r, r)
            eq_count[owner] += 1
        mat += reg * np.eye(3, dtype=np.float64)[None, :, :]
        solve_mask = eq_count >= 3
        inv = np.zeros_like(mat)
        if np.any(solve_mask):
            inv[solve_mask] = np.linalg.inv(mat[solve_mask])

        no_flux_mask_np = np.zeros((n_faces,), dtype=bool)
        no_flux_mask_np[np.asarray(boundary_no_flux, dtype=np.int64)] = True

        diffusion_data = {
            "st_interior_c0": torch.as_tensor(
                stencil.interior_c0, dtype=torch.long, device=device
            ),
            "st_interior_c1": torch.as_tensor(
                stencil.interior_c1, dtype=torch.long, device=device
            ),
            "st_interior_alpha": torch.as_tensor(
                stencil.interior_alpha, dtype=dtype, device=device
            ),
            "st_interior_correction": torch.as_tensor(
                stencil.interior_correction_vec, dtype=dtype, device=device
            ),
            "st_dirichlet_c0": torch.as_tensor(
                stencil.dirichlet_c0, dtype=torch.long, device=device
            ),
            "st_dirichlet_coeff": torch.as_tensor(
                stencil.dirichlet_coeff, dtype=dtype, device=device
            ),
            "st_dirichlet_value": torch.as_tensor(
                stencil.dirichlet_value, dtype=dtype, device=device
            ),
            "gg_no_flux_mask": torch.as_tensor(
                no_flux_mask_np, dtype=torch.bool, device=device
            ),
            "gg_boundary_has_dirichlet": torch.as_tensor(
                np.isin(
                    np.arange(n_faces),
                    np.asarray(list(boundary_dirichlet.keys()), dtype=np.int64),
                ),
                dtype=torch.bool,
                device=device,
            ),
            "gg_boundary_dirichlet_values": torch.as_tensor(
                np.asarray(
                    [boundary_dirichlet.get(i, 0.0) for i in range(n_faces)],
                    dtype=np.float64,
                ),
                dtype=dtype,
                device=device,
            ),
            "gg_weighted_normals": torch.as_tensor(
                mesh.face_normals * mesh.face_areas[:, None], dtype=dtype, device=device
            ),
            "gg_c0": c0,
            "gg_c1": c1,
            "lsq_c0": torch.as_tensor(c0_i, dtype=torch.long, device=device),
            "lsq_c1": torch.as_tensor(c1_i, dtype=torch.long, device=device),
            "lsq_r": torch.as_tensor(delta_x, dtype=dtype, device=device),
            "lsq_w": torch.as_tensor(w_i, dtype=dtype, device=device),
            "lsq_db_owner": torch.as_tensor(db_owner, dtype=torch.long, device=device),
            "lsq_db_r": torch.as_tensor(db_delta, dtype=dtype, device=device),
            "lsq_db_w": torch.as_tensor(db_w, dtype=dtype, device=device),
            "lsq_db_value": torch.as_tensor(db_value, dtype=dtype, device=device),
            "lsq_inv": torch.as_tensor(inv, dtype=dtype, device=device),
            "lsq_solve_mask": torch.as_tensor(
                solve_mask, dtype=torch.bool, device=device
            ),
        }

    dt_control = resolve_transport_dt_control(
        mesh,
        face_normal_velocity,
        config=config,
        diffusion_dt_limit=diffusion_dt_limit,
    )
    dt_control = {
        **dt_control,
        "cfl_target": float(config.cfl_target),
        "cfl_limit": float(config.cfl_limit),
    }
    used_dt = float(dt_control["used_dt"])
    dt_substep = float(dt_control["dt_substep"])
    transport_substep_count = int(dt_control["transport_substep_count"])
    diffusion_stability_warning = bool(
        np.isfinite(diffusion_dt_limit)
        and dt_substep > float(diffusion_dt_limit) + 1e-30
    )
    if bool(dt_control["transport_dt_controller_blocked"]):
        raise RuntimeError(str(dt_control["transport_dt_controller_reason"]))

    def _gradient_green_gauss_torch(scalar_t: torch.Tensor) -> torch.Tensor:
        weighted_normals = diffusion_data["gg_weighted_normals"]  # type: ignore[index]
        fc0 = diffusion_data["gg_c0"]  # type: ignore[index]
        fc1 = diffusion_data["gg_c1"]  # type: ignore[index]
        face_values = torch.zeros((n_faces,), dtype=dtype, device=device)
        interior = fc1 >= 0
        face_values[interior] = 0.5 * (
            scalar_t[fc0[interior]] + scalar_t[fc1[interior]]
        )
        face_values[~interior] = scalar_t[fc0[~interior]]
        no_flux = diffusion_data["gg_no_flux_mask"]  # type: ignore[index]
        face_values[no_flux] = scalar_t[fc0[no_flux]]
        has_dir = diffusion_data["gg_boundary_has_dirichlet"]  # type: ignore[index]
        if bool(torch.any(has_dir)):
            face_values = torch.where(
                has_dir,
                diffusion_data["gg_boundary_dirichlet_values"],  # type: ignore[index]
                face_values,
            )
        gradients = torch.zeros((n_cells, 3), dtype=dtype, device=device)
        flux_vec = face_values[:, None] * weighted_normals
        gradients.index_add_(0, fc0, flux_vec)
        interior_faces_gg = torch.nonzero(interior, as_tuple=False).reshape(-1)
        if int(interior_faces_gg.numel()) > 0:
            gradients.index_add_(
                0, fc1[interior_faces_gg], -flux_vec[interior_faces_gg]
            )
        gradients = gradients / vol[:, None]
        return gradients

    def _gradient_lsq_torch(scalar_t: torch.Tensor) -> torch.Tensor:
        if config.gradient_method == "face":
            return _gradient_green_gauss_torch(scalar_t)
        rhs = torch.zeros((n_cells, 3), dtype=dtype, device=device)
        lsq_c0 = diffusion_data["lsq_c0"]  # type: ignore[index]
        lsq_c1 = diffusion_data["lsq_c1"]  # type: ignore[index]
        lsq_r = diffusion_data["lsq_r"]  # type: ignore[index]
        lsq_w = diffusion_data["lsq_w"]  # type: ignore[index]
        if int(lsq_c0.numel()) > 0:
            dphi = scalar_t[lsq_c1] - scalar_t[lsq_c0]
            vec = (lsq_w[:, None] * lsq_r) * dphi[:, None]
            rhs.index_add_(0, lsq_c0, vec)
            rhs.index_add_(0, lsq_c1, vec)
        db_owner = diffusion_data["lsq_db_owner"]  # type: ignore[index]
        if int(db_owner.numel()) > 0:
            db_r = diffusion_data["lsq_db_r"]  # type: ignore[index]
            db_w = diffusion_data["lsq_db_w"]  # type: ignore[index]
            db_value = diffusion_data["lsq_db_value"]  # type: ignore[index]
            vec_b = (db_w[:, None] * db_r) * (db_value - scalar_t[db_owner])[:, None]
            rhs.index_add_(0, db_owner, vec_b)
        gradients = _gradient_green_gauss_torch(scalar_t)
        solve_mask = diffusion_data["lsq_solve_mask"]  # type: ignore[index]
        if bool(torch.any(solve_mask)):
            inv = diffusion_data["lsq_inv"]  # type: ignore[index]
            solved = torch.matmul(
                inv[solve_mask], rhs[solve_mask].unsqueeze(-1)
            ).squeeze(-1)
            gradients[solve_mask] = solved
        return gradients

    def _laplacian_torch(scalar_t: torch.Tensor) -> torch.Tensor:
        if generic_diffusion_precompute is not None:
            return scalar_laplacian_torch(
                generic_diffusion_precompute,
                scalar_t,
                gradient_method=config.gradient_method,
                laplacian_method=config.laplacian_method,
            )
        if config.laplacian_method == "tpfa":
            lap = torch.zeros((n_cells,), dtype=dtype, device=device)
            st_c0 = diffusion_data["st_interior_c0"]  # type: ignore[index]
            st_c1 = diffusion_data["st_interior_c1"]  # type: ignore[index]
            if int(st_c0.numel()) > 0:
                coeff = torch.abs(diffusion_data["st_interior_alpha"])  # type: ignore[index]
                flux = coeff * (scalar_t[st_c1] - scalar_t[st_c0])
                lap.index_add_(0, st_c0, flux / vol[st_c0])
                lap.index_add_(0, st_c1, -flux / vol[st_c1])
            db_c0 = diffusion_data["st_dirichlet_c0"]  # type: ignore[index]
            if int(db_c0.numel()) > 0:
                coeff_b = diffusion_data["st_dirichlet_coeff"]  # type: ignore[index]
                value_b = diffusion_data["st_dirichlet_value"]  # type: ignore[index]
                flux_b = coeff_b * (value_b - scalar_t[db_c0])
                lap.index_add_(0, db_c0, flux_b / vol[db_c0])
            return lap
        gradients = _gradient_lsq_torch(scalar_t)
        lap = torch.zeros((n_cells,), dtype=dtype, device=device)
        st_c0 = diffusion_data["st_interior_c0"]  # type: ignore[index]
        st_c1 = diffusion_data["st_interior_c1"]  # type: ignore[index]
        if int(st_c0.numel()) > 0:
            alpha = diffusion_data["st_interior_alpha"]  # type: ignore[index]
            corr = diffusion_data["st_interior_correction"]  # type: ignore[index]
            grad_face = 0.5 * (gradients[st_c0] + gradients[st_c1])
            orth_flux = alpha * (scalar_t[st_c1] - scalar_t[st_c0])
            corr_flux = torch.sum(grad_face * corr, dim=1)
            flux = orth_flux + corr_flux
            lap.index_add_(0, st_c0, flux / vol[st_c0])
            lap.index_add_(0, st_c1, -flux / vol[st_c1])
        db_c0 = diffusion_data["st_dirichlet_c0"]  # type: ignore[index]
        if int(db_c0.numel()) > 0:
            coeff_b = diffusion_data["st_dirichlet_coeff"]  # type: ignore[index]
            value_b = diffusion_data["st_dirichlet_value"]  # type: ignore[index]
            flux_b = coeff_b * (value_b - scalar_t[db_c0])
            lap.index_add_(0, db_c0, flux_b / vol[db_c0])
        return lap

    # Main stepping.
    if initial_scalar is None:
        scalar = torch.zeros((n_cells,), dtype=dtype, device=device)
    else:
        init = np.asarray(initial_scalar, dtype=np.float64)
        if init.shape != (n_cells,):
            raise ValueError(
                f"initial_scalar shape mismatch: expected {(n_cells,)}, got {init.shape}"
            )
        if not np.all(np.isfinite(init)):
            raise ValueError("initial_scalar contains non-finite values.")
        scalar = torch.as_tensor(init, dtype=dtype, device=device)
    history: list[Dict[str, float | int | bool]] = []
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_steps_set = set(int(s) for s in config.snapshot_steps if int(s) > 0)
    clipping_used = False
    thresholds = OUTLET_FRACTION_THRESHOLDS
    initial_mass = float(torch.sum(scalar * vol).item())
    mass_prev = initial_mass
    cumulative_mass_in = float(initial_cumulative_mass_in)
    cumulative_mass_out = float(initial_cumulative_mass_out)
    cumulative_raw_mass_in = float(initial_cumulative_mass_in)
    cumulative_raw_mass_out = float(initial_cumulative_mass_out)
    max_raw_overshoot = 0.0
    max_raw_undershoot = 0.0
    max_limited_overshoot = 0.0
    max_limited_undershoot = 0.0
    max_diffusion_overshoot = 0.0
    max_diffusion_undershoot = 0.0
    safety_clamp_used = False
    safety_clamp_cell_count_total = 0
    safety_clamp_cell_count_final = 0
    safety_clamp_mass_delta_total = 0.0
    safety_clamp_max_correction = 0.0

    final_raw_state = scalar.clone()
    final_limited_state = scalar.clone()
    final_updated_before_clip = scalar.clone()
    final_in_raw = torch.zeros_like(scalar)
    final_out_lim = torch.zeros_like(scalar)
    final_in_lim = torch.zeros_like(scalar)
    final_limiter_out = torch.ones_like(scalar)
    final_limiter_in = torch.ones_like(scalar)
    final_out_vol_rate_raw = torch.zeros_like(scalar)
    final_in_vol_rate_raw = torch.zeros_like(scalar)
    final_scalar_old = scalar.clone()

    left_inlet_value_t = float(config.left_inlet_value)
    right_inlet_value_t = float(config.right_inlet_value)
    zero_t = torch.tensor(0.0, dtype=dtype, device=device)

    for step_local in range(1, config.steps + 1):
        step = int(start_step + step_local)
        scalar_prev = scalar.clone()
        step_raw_mass_in = 0.0
        step_raw_mass_out = 0.0
        step_mass_in = 0.0
        step_mass_out = 0.0
        step_out_cfl_max = 0.0
        step_in_cfl_max = 0.0
        step_raw_overshoot = 0.0
        step_raw_undershoot = 0.0
        step_lim_overshoot = 0.0
        step_lim_undershoot = 0.0
        step_diff_overshoot = 0.0
        step_diff_undershoot = 0.0
        step_diffusion_lap_max_abs = 0.0
        step_max_raw_delta_c = 0.0
        step_limiter_active_count = 0
        step_limiter_active_fraction = 0.0
        step_limiter_max_flux_scale_down = 0.0
        step_limiter_mass_correction_total = 0.0
        step_conservation_error = 0.0
        step_overshoot_mask = torch.zeros((n_cells,), dtype=torch.bool, device=device)
        step_undershoot_mask = torch.zeros((n_cells,), dtype=torch.bool, device=device)
        step_clipped_mask = torch.zeros((n_cells,), dtype=torch.bool, device=device)

        for _substep in range(transport_substep_count):
            scalar_substep_prev = scalar.clone()
            out_mass = torch.zeros_like(scalar)
            in_mass = torch.zeros_like(scalar)
            out_vol_rate = torch.zeros_like(scalar)
            in_vol_rate = torch.zeros_like(scalar)

            c_up_i = scalar[donor_i]
            phi_i = q_i * c_up_i
            m_i = dt_substep * torch.abs(phi_i)
            out_mass.scatter_add_(0, donor_i, m_i)
            in_mass.scatter_add_(0, recv_i, m_i)
            out_vol_rate.scatter_add_(0, donor_i, torch.abs(q_i))
            in_vol_rate.scatter_add_(0, recv_i, torch.abs(q_i))

            inlet_scalar_flux_in = zero_t.clone()
            outlet_scalar_flux_out = zero_t.clone()

            def _apply_boundary_group(
                face_ids_t: torch.Tensor, prescribed_value: float | None
            ) -> None:
                nonlocal inlet_scalar_flux_in, outlet_scalar_flux_out
                if int(face_ids_t.numel()) == 0:
                    return
                owner = c0[face_ids_t]
                q = q_all[face_ids_t]
                inflow = q < 0.0
                outflow = q > 0.0
                if prescribed_value is not None:
                    if bool(torch.any(inflow)):
                        ow = owner[inflow]
                        qi = q[inflow]
                        c_face = torch.full_like(
                            qi, float(prescribed_value), dtype=dtype
                        )
                        phi = qi * c_face
                        m = -dt_substep * phi
                        in_mass.scatter_add_(0, ow, m)
                        in_vol_rate.scatter_add_(0, ow, torch.abs(qi))
                        inlet_scalar_flux_in = inlet_scalar_flux_in + torch.sum(-phi)
                    if bool(torch.any(outflow)):
                        ow = owner[outflow]
                        qo = q[outflow]
                        c_face = scalar[ow]
                        phi = qo * c_face
                        m = dt_substep * phi
                        out_mass.scatter_add_(0, ow, m)
                        out_vol_rate.scatter_add_(0, ow, torch.abs(qo))
                    return
                if bool(torch.any(outflow)):
                    ow = owner[outflow]
                    qo = q[outflow]
                    c_face = scalar[ow]
                    phi = qo * c_face
                    m = dt_substep * phi
                    out_mass.scatter_add_(0, ow, m)
                    out_vol_rate.scatter_add_(0, ow, torch.abs(qo))
                    if torch.equal(face_ids_t, outlet_faces_t):
                        outlet_scalar_flux_out = outlet_scalar_flux_out + torch.sum(phi)

            _apply_boundary_group(left_faces_t, left_inlet_value_t)
            _apply_boundary_group(right_faces_t, right_inlet_value_t)
            _apply_boundary_group(outlet_faces_t, None)
            _apply_boundary_group(unresolved_faces_t, None)

            mass_old = scalar * vol
            raw_state = (mass_old + in_mass - out_mass) / vol

            if config.transport_scheme == "bounded_upwind":
                limiter_out = torch.ones_like(scalar)
                limiter_in = torch.ones_like(scalar)
                mask_out = out_mass > 0.0
                mask_in = in_mass > 0.0
                limiter_out = torch.where(
                    mask_out,
                    torch.minimum(
                        torch.ones_like(scalar),
                        mass_old / torch.clamp(out_mass, min=1e-30),
                    ),
                    limiter_out,
                )
                capacity = (config.clip_max - scalar) * vol
                limiter_in = torch.where(
                    mask_in,
                    torch.minimum(
                        torch.ones_like(scalar),
                        capacity / torch.clamp(in_mass, min=1e-30),
                    ),
                    limiter_in,
                )
            else:
                limiter_out = torch.ones_like(scalar)
                limiter_in = torch.ones_like(scalar)

            pair_scale = torch.minimum(limiter_out[donor_i], limiter_in[recv_i])
            m_i_lim = m_i * pair_scale
            out_lim = torch.zeros_like(scalar)
            in_lim = torch.zeros_like(scalar)
            out_lim.scatter_add_(0, donor_i, m_i_lim)
            in_lim.scatter_add_(0, recv_i, m_i_lim)

            inlet_scalar_flux_in_effective = zero_t.clone()
            outlet_scalar_flux_out_effective = zero_t.clone()

            def _apply_boundary_limited(
                face_ids_t: torch.Tensor, prescribed_value: float | None
            ) -> None:
                nonlocal \
                    inlet_scalar_flux_in_effective, \
                    outlet_scalar_flux_out_effective
                if int(face_ids_t.numel()) == 0:
                    return
                owner = c0[face_ids_t]
                q = q_all[face_ids_t]
                inflow = q < 0.0
                outflow = q > 0.0
                if prescribed_value is not None:
                    if bool(torch.any(inflow)):
                        ow = owner[inflow]
                        qi = q[inflow]
                        phi = qi * float(prescribed_value)
                        m_raw = -dt_substep * phi
                        scale = limiter_in[ow]
                        in_lim.scatter_add_(0, ow, m_raw * scale)
                        inlet_scalar_flux_in_effective = (
                            inlet_scalar_flux_in_effective + torch.sum((-phi) * scale)
                        )
                    if bool(torch.any(outflow)):
                        ow = owner[outflow]
                        qo = q[outflow]
                        phi = qo * scalar[ow]
                        m_raw = dt_substep * phi
                        out_lim.scatter_add_(0, ow, m_raw * limiter_out[ow])
                    return
                if bool(torch.any(outflow)):
                    ow = owner[outflow]
                    qo = q[outflow]
                    phi = qo * scalar[ow]
                    m_raw = dt_substep * phi
                    scale = limiter_out[ow]
                    out_lim.scatter_add_(0, ow, m_raw * scale)
                    if torch.equal(face_ids_t, outlet_faces_t):
                        outlet_scalar_flux_out_effective = (
                            outlet_scalar_flux_out_effective + torch.sum(phi * scale)
                        )

            _apply_boundary_limited(left_faces_t, left_inlet_value_t)
            _apply_boundary_limited(right_faces_t, right_inlet_value_t)
            _apply_boundary_limited(outlet_faces_t, None)
            _apply_boundary_limited(unresolved_faces_t, None)

            limited_state = (mass_old + in_lim - out_lim) / vol
            updated_raw = limited_state
            diffusion_lap_max_abs = 0.0
            if config.transport_mode == "advection_diffusion":
                lap = _laplacian_torch(updated_raw)
                diffusion_lap_max_abs = (
                    float(torch.max(torch.abs(lap)).item()) if n_cells else 0.0
                )
                updated_raw = updated_raw + dt_substep * float(config.diffusivity) * lap

            if config.source_term != 0.0:
                updated_raw = updated_raw + dt_substep * float(config.source_term)

            if config.safety_clamp_after_diffusion:
                clamped = torch.clamp(
                    updated_raw, min=config.clip_min, max=config.clip_max
                )
                delta = clamped - updated_raw
                count = int(torch.count_nonzero(delta != 0.0).item())
                mass_delta = float(torch.sum(delta * vol).item())
                max_corr = float(torch.max(torch.abs(delta)).item()) if n_cells else 0.0
                safety_clamp_cell_count_total += count
                safety_clamp_cell_count_final = count
                safety_clamp_mass_delta_total += mass_delta
                safety_clamp_max_correction = max(safety_clamp_max_correction, max_corr)
                if count > 0:
                    safety_clamp_used = True
                updated_raw = clamped

            raw_overshoot = torch.clamp(raw_state - config.clip_max, min=0.0)
            raw_undershoot = torch.clamp(config.clip_min - raw_state, min=0.0)
            limited_overshoot = torch.clamp(limited_state - config.clip_max, min=0.0)
            limited_undershoot = torch.clamp(config.clip_min - limited_state, min=0.0)
            diffusion_overshoot = torch.clamp(updated_raw - config.clip_max, min=0.0)
            diffusion_undershoot = torch.clamp(config.clip_min - updated_raw, min=0.0)
            overshoot_mask = updated_raw > (config.clip_max + 1e-12)
            undershoot_mask = updated_raw < (config.clip_min - 1e-12)
            clipped_mask = overshoot_mask | undershoot_mask

            if config.clipping_enabled:
                if bool(torch.any(clipped_mask)):
                    clipping_used = True
                updated = torch.clamp(
                    updated_raw, min=config.clip_min, max=config.clip_max
                )
            else:
                updated = updated_raw
            scalar = updated

            raw_mass_in = dt_substep * float(inlet_scalar_flux_in.item())
            raw_mass_out = dt_substep * float(outlet_scalar_flux_out.item())
            mass_in = dt_substep * float(inlet_scalar_flux_in_effective.item())
            mass_out = dt_substep * float(outlet_scalar_flux_out_effective.item())
            step_raw_mass_in += raw_mass_in
            step_raw_mass_out += raw_mass_out
            step_mass_in += mass_in
            step_mass_out += mass_out

            local_outgoing_cfl = dt_substep * out_vol_rate / vol
            local_incoming_cfl = dt_substep * in_vol_rate / vol
            local_out_cfl_max = float(torch.max(local_outgoing_cfl).item())
            local_in_cfl_max = float(torch.max(local_incoming_cfl).item())
            step_out_cfl_max = max(step_out_cfl_max, local_out_cfl_max)
            step_in_cfl_max = max(step_in_cfl_max, local_in_cfl_max)

            local_raw_overshoot = (
                float(torch.max(raw_overshoot).item()) if n_cells else 0.0
            )
            local_raw_undershoot = (
                float(torch.max(raw_undershoot).item()) if n_cells else 0.0
            )
            local_lim_overshoot = (
                float(torch.max(limited_overshoot).item()) if n_cells else 0.0
            )
            local_lim_undershoot = (
                float(torch.max(limited_undershoot).item()) if n_cells else 0.0
            )
            local_diff_overshoot = (
                float(torch.max(diffusion_overshoot).item()) if n_cells else 0.0
            )
            local_diff_undershoot = (
                float(torch.max(diffusion_undershoot).item()) if n_cells else 0.0
            )
            step_raw_overshoot = max(step_raw_overshoot, local_raw_overshoot)
            step_raw_undershoot = max(step_raw_undershoot, local_raw_undershoot)
            step_lim_overshoot = max(step_lim_overshoot, local_lim_overshoot)
            step_lim_undershoot = max(step_lim_undershoot, local_lim_undershoot)
            step_diff_overshoot = max(step_diff_overshoot, local_diff_overshoot)
            step_diff_undershoot = max(step_diff_undershoot, local_diff_undershoot)
            step_diffusion_lap_max_abs = max(
                step_diffusion_lap_max_abs, diffusion_lap_max_abs
            )

            raw_delta = torch.abs(raw_state - scalar_substep_prev)
            local_max_raw_delta_c = (
                float(torch.max(raw_delta).item()) if n_cells else 0.0
            )
            step_max_raw_delta_c = max(step_max_raw_delta_c, local_max_raw_delta_c)
            limiter_active_mask = (torch.abs(limiter_out - 1.0) > 1e-12) | (
                torch.abs(limiter_in - 1.0) > 1e-12
            )
            limiter_active_count = int(torch.count_nonzero(limiter_active_mask).item())
            limiter_active_fraction = float(limiter_active_count / max(1, n_cells))
            limiter_max_flux_scale_down = float(
                torch.max(1.0 - torch.minimum(limiter_out, limiter_in)).item()
            )
            limiter_mass_correction_total = float(
                torch.sum(torch.abs(in_mass - in_lim)).item()
            )
            conservation_error = float(
                torch.sum((mass_old + in_lim - out_lim) - (raw_state * vol)).item()
            )
            step_limiter_active_count = max(
                step_limiter_active_count, limiter_active_count
            )
            step_limiter_active_fraction = max(
                step_limiter_active_fraction, limiter_active_fraction
            )
            step_limiter_max_flux_scale_down = max(
                step_limiter_max_flux_scale_down, limiter_max_flux_scale_down
            )
            step_limiter_mass_correction_total += limiter_mass_correction_total
            step_conservation_error = max(
                step_conservation_error, abs(conservation_error)
            )
            step_overshoot_mask = overshoot_mask
            step_undershoot_mask = undershoot_mask
            step_clipped_mask = clipped_mask

            final_scalar_old = scalar_substep_prev.clone()
            final_raw_state = raw_state.clone()
            final_limited_state = limited_state.clone()
            final_updated_before_clip = updated_raw.clone()
            final_in_raw = in_mass.clone()
            final_out_lim = out_lim.clone()
            final_in_lim = in_lim.clone()
            final_limiter_out = limiter_out.clone()
            final_limiter_in = limiter_in.clone()
            final_out_vol_rate_raw = out_vol_rate.clone()
            final_in_vol_rate_raw = in_vol_rate.clone()

        cumulative_mass_in += step_mass_in
        cumulative_mass_out += step_mass_out
        cumulative_raw_mass_in += step_raw_mass_in
        cumulative_raw_mass_out += step_raw_mass_out

        mass = float(torch.sum(scalar * vol).item())
        max_change = float(torch.max(torch.abs(scalar - scalar_prev)).item())
        mean_change = float(torch.mean(torch.abs(scalar - scalar_prev)).item())
        mean_ref = float(torch.mean(torch.abs(scalar_prev)).item())
        relative_change = float(mean_change / max(mean_ref, 1e-30))

        left_vals = (
            scalar[left_cells_t]
            if left_cells_t.numel() > 0
            else torch.full((1,), float("nan"), dtype=dtype, device=device)
        )
        right_vals = (
            scalar[right_cells_t]
            if right_cells_t.numel() > 0
            else torch.full((1,), float("nan"), dtype=dtype, device=device)
        )
        outlet_vals = (
            scalar[outlet_cells_t]
            if outlet_cells_t.numel() > 0
            else torch.full((1,), float("nan"), dtype=dtype, device=device)
        )

        outlet_arrival: Dict[str, float] = {}
        for thr in thresholds:
            key = format_threshold_key("outlet_frac_gt", thr)
            if outlet_cells_t.numel() > 0:
                outlet_arrival[key] = float(
                    torch.count_nonzero(outlet_vals > thr).item()
                    / max(1, int(outlet_vals.numel()))
                )
            else:
                outlet_arrival[key] = 0.0

        max_raw_overshoot = max(max_raw_overshoot, step_raw_overshoot)
        max_raw_undershoot = max(max_raw_undershoot, step_raw_undershoot)
        max_limited_overshoot = max(max_limited_overshoot, step_lim_overshoot)
        max_limited_undershoot = max(max_limited_undershoot, step_lim_undershoot)
        max_diffusion_overshoot = max(max_diffusion_overshoot, step_diff_overshoot)
        max_diffusion_undershoot = max(max_diffusion_undershoot, step_diff_undershoot)

        overshoot_count = int(torch.count_nonzero(step_overshoot_mask).item())
        undershoot_count = int(torch.count_nonzero(step_undershoot_mask).item())
        clipped_count = int(torch.count_nonzero(step_clipped_mask).item())
        clipped_fraction = float(clipped_count / max(1, n_cells))

        entry: Dict[str, float | int | bool] = {
            "step": int(step),
            "min": float(torch.min(scalar).item()),
            "max": float(torch.max(scalar).item()),
            "mean": float(torch.mean(scalar).item()),
            "total_mass_proxy": mass,
            "mass_delta": float(mass - mass_prev),
            "max_change": max_change,
            "mean_change": mean_change,
            "relative_change": relative_change,
            "left_inlet_min": float(torch.min(left_vals).item()),
            "left_inlet_max": float(torch.max(left_vals).item()),
            "right_inlet_min": float(torch.min(right_vals).item()),
            "right_inlet_max": float(torch.max(right_vals).item()),
            "cfl_max": step_out_cfl_max,
            "cfl_warning": bool(step_out_cfl_max > config.cfl_limit),
            "dt_used": used_dt,
            "dt_substep": dt_substep,
            "transport_substep_count": transport_substep_count,
            "raw_overshoot_before_limiter": step_raw_overshoot,
            "raw_undershoot_before_limiter": step_raw_undershoot,
            "overshoot_after_limiter_before_clip": step_lim_overshoot,
            "undershoot_after_limiter_before_clip": step_lim_undershoot,
            "overshoot_before_clip": step_diff_overshoot,
            "undershoot_before_clip": step_diff_undershoot,
            "overshoot_cell_count": overshoot_count,
            "undershoot_cell_count": undershoot_count,
            "clipped_cell_count": clipped_count,
            "clipped_cell_count_after_limiter": clipped_count,
            "clipped_fraction": clipped_fraction,
            "max_raw_delta_c": step_max_raw_delta_c,
            "left_inlet_zone_overshoot_cell_count": int(
                torch.count_nonzero(step_overshoot_mask & left_zone_mask_t).item()
            ),
            "right_inlet_zone_overshoot_cell_count": int(
                torch.count_nonzero(step_overshoot_mask & right_zone_mask_t).item()
            ),
            "inlet_wall_corner_zone_overshoot_cell_count": int(
                torch.count_nonzero(step_overshoot_mask & corner_zone_mask_t).item()
            ),
            "left_inlet_zone_undershoot_cell_count": int(
                torch.count_nonzero(step_undershoot_mask & left_zone_mask_t).item()
            ),
            "right_inlet_zone_undershoot_cell_count": int(
                torch.count_nonzero(step_undershoot_mask & right_zone_mask_t).item()
            ),
            "inlet_wall_corner_zone_undershoot_cell_count": int(
                torch.count_nonzero(step_undershoot_mask & corner_zone_mask_t).item()
            ),
            "left_inlet_zone_clipped_cell_count": int(
                torch.count_nonzero(step_clipped_mask & left_zone_mask_t).item()
            ),
            "right_inlet_zone_clipped_cell_count": int(
                torch.count_nonzero(step_clipped_mask & right_zone_mask_t).item()
            ),
            "inlet_wall_corner_zone_clipped_cell_count": int(
                torch.count_nonzero(step_clipped_mask & corner_zone_mask_t).item()
            ),
            "limiter_active_cell_count": step_limiter_active_count,
            "limiter_active_fraction": step_limiter_active_fraction,
            "limiter_max_flux_scale_down": step_limiter_max_flux_scale_down,
            "limiter_mass_correction_total": step_limiter_mass_correction_total,
            "conservation_error_after_limiter": step_conservation_error,
            "inlet_scalar_flux_in": float(step_mass_in / max(used_dt, 1e-30)),
            "outlet_scalar_flux_out": float(step_mass_out / max(used_dt, 1e-30)),
            "raw_inlet_scalar_flux_in": float(step_raw_mass_in / max(used_dt, 1e-30)),
            "raw_outlet_scalar_flux_out": float(
                step_raw_mass_out / max(used_dt, 1e-30)
            ),
            "effective_inlet_scalar_flux_in": float(step_mass_in / max(used_dt, 1e-30)),
            "effective_outlet_scalar_flux_out": float(
                step_mass_out / max(used_dt, 1e-30)
            ),
            "mass_in": step_mass_in,
            "mass_out": step_mass_out,
            "raw_mass_in": step_raw_mass_in,
            "raw_mass_out": step_raw_mass_out,
            "net_mass_change_from_flux": step_mass_in - step_mass_out,
            "raw_net_mass_change_from_flux": step_raw_mass_in - step_raw_mass_out,
            "cumulative_mass_in": cumulative_mass_in,
            "cumulative_mass_out": cumulative_mass_out,
            "cumulative_net_mass_change_from_flux": cumulative_mass_in
            - cumulative_mass_out,
            "cumulative_raw_mass_in": cumulative_raw_mass_in,
            "cumulative_raw_mass_out": cumulative_raw_mass_out,
            "cumulative_raw_net_mass_change_from_flux": cumulative_raw_mass_in
            - cumulative_raw_mass_out,
            "max_local_outgoing_cfl": step_out_cfl_max,
            "max_local_incoming_cfl": step_in_cfl_max,
            "pairwise_conservation_error_max_abs": 0.0,
            "pairwise_conservation_error_mean_abs": 0.0,
            "diffusion_laplacian_max_abs": step_diffusion_lap_max_abs,
        }
        entry.update(outlet_arrival)
        physical_time = float(config.physical_time_start + step_local * used_dt)
        total_progress_steps = int(
            config.progress_total_steps
            if config.progress_total_steps is not None
            else start_step + config.steps
        )
        entry["physical_time"] = physical_time
        entry["step_progress_percent"] = float(
            100.0 * step / max(1, total_progress_steps)
        )
        if config.progress_target_end_time is not None:
            entry["time_progress_percent"] = float(
                min(
                    100.0,
                    100.0
                    * physical_time
                    / max(float(config.progress_target_end_time), 1e-30),
                )
            )
        history.append(entry)
        mass_prev = mass

        if config.progress_every > 0 and step_local % config.progress_every == 0:
            print(
                "[gmsh-tetra-transport] "
                f"step {step}/{start_step + config.steps}, min={entry['min']:.6f}, "
                f"max={entry['max']:.6f}, cfl={entry['cfl_max']:.4f}, "
                f"substeps={transport_substep_count}, "
                f"progress={entry['step_progress_percent']:.1f}%"
            )
        if step in snapshot_steps_set:
            snapshots[int(step)] = scalar.detach().cpu().numpy().copy()

    scalar_np = scalar.detach().cpu().numpy()
    raw_state_np = final_raw_state.detach().cpu().numpy()
    lim_state_np = final_limited_state.detach().cpu().numpy()
    upd_raw_np = final_updated_before_clip.detach().cpu().numpy()
    overshoot_mask_np = upd_raw_np > (config.clip_max + 1e-12)
    undershoot_mask_np = upd_raw_np < (config.clip_min - 1e-12)
    clipped_mask_np = overshoot_mask_np | undershoot_mask_np

    outlet_arrival_final = {}
    if outlet_cells.size:
        out_vals = scalar_np[outlet_cells]
        for thr in thresholds:
            outlet_arrival_final[format_threshold_key("outlet_frac_gt", thr)] = float(
                np.count_nonzero(out_vals > thr) / out_vals.size
            )
    else:
        for thr in thresholds:
            outlet_arrival_final[format_threshold_key("outlet_frac_gt", thr)] = 0.0

    mass_final = float(np.sum(scalar_np * mesh.cell_volumes))
    max_out_cfl = float(torch.max(dt_substep * final_out_vol_rate_raw / vol).item())
    max_in_cfl = float(torch.max(dt_substep * final_in_vol_rate_raw / vol).item())

    diffusion_diag = {
        "diffusion_backend": "torch"
        if config.transport_mode == "advection_diffusion"
        else "none",
        "laplacian_method": config.laplacian_method,
        "gradient_method": config.gradient_method,
        "diffusivity": float(config.diffusivity),
        "diffusion_dt_limit": diffusion_dt_limit,
        "diffusion_stability_warning": bool(diffusion_stability_warning),
        "max_overshoot_after_diffusion_before_clip": float(max_diffusion_overshoot),
        "max_undershoot_after_diffusion_before_clip": float(max_diffusion_undershoot),
        "safety_clamp_after_diffusion_enabled": bool(
            config.safety_clamp_after_diffusion
        ),
        "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
    }

    return {
        "scalar": scalar_np,
        "history": history,
        "snapshots": snapshots,
        "cfl_values": (dt_substep * final_out_vol_rate_raw / vol)
        .detach()
        .cpu()
        .numpy(),
        "cfl_max": max_out_cfl,
        "cfl_warning": bool(max_out_cfl > config.cfl_limit),
        "dt_control": dt_control,
        "flux_diagnostics": flux_diagnostics,
        "boundary_setup": {
            "left_inlet_faces": int(left_faces.size),
            "right_inlet_faces": int(right_faces.size),
            "outlet_faces": int(outlet_faces.size),
            "wall_faces": int(wall_faces.size),
            "left_inlet_cells": int(left_cells.size),
            "right_inlet_cells": int(right_cells.size),
        },
        "transport_info": {
            "transport_mode": config.transport_mode,
            "transport_scheme": config.transport_scheme,
            "gradient_method": config.gradient_method,
            "laplacian_method": config.laplacian_method,
            "flow_solved": False,
            "pressure_solved": False,
            "velocity_source": "prescribed_debug_field",
            "scalar_bc_mode": "inflow_only_dirichlet_face_flux",
            "inlet_dirichlet_applied_on": "faces",
            "cell_level_inlet_overwrite_used": False,
            "post_step_cell_overwrite_used": False,
            "bounded_limiter_enabled": bool(
                config.transport_scheme == "bounded_upwind"
            ),
            "diffusion_backend": diffusion_diag["diffusion_backend"],
            "safety_clamp_after_diffusion_enabled": bool(
                config.safety_clamp_after_diffusion
            ),
            "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
        },
        "backend_execution": {
            "requested_backend": config.backend,
            "stepping_backend": "torch",
            "device": str(device),
            "torch_device": config.torch_device,
            "used_numpy_fallback": False,
            "all_core_arrays_on_cuda": bool(device.type == "cuda"),
            "per_step_large_cpu_gpu_transfer_warning": False,
            "cpu_gpu_transfer_notes": [],
        },
        "clipping": {
            "enabled": bool(config.clipping_enabled),
            "used": bool(clipping_used),
            "post_step_clipping_used": bool(clipping_used),
            "safety_clamp_after_diffusion_enabled": bool(
                config.safety_clamp_after_diffusion
            ),
            "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
            "safety_clamp_cell_count_total": int(safety_clamp_cell_count_total),
            "safety_clamp_cell_count_final_step": int(safety_clamp_cell_count_final),
            "safety_clamp_mass_delta": float(safety_clamp_mass_delta_total),
            "safety_clamp_max_correction": float(safety_clamp_max_correction),
            "max_overshoot_before_clip": float(np.max(upd_raw_np - config.clip_max)),
            "max_undershoot_before_clip": float(np.max(config.clip_min - upd_raw_np)),
            "max_overshoot_after_limiter_before_clip": float(max_limited_overshoot),
            "max_undershoot_after_limiter_before_clip": float(max_limited_undershoot),
            "final_clipped_cell_count": int(np.count_nonzero(clipped_mask_np)),
            "final_overshoot_cell_count": int(np.count_nonzero(overshoot_mask_np)),
            "final_undershoot_cell_count": int(np.count_nonzero(undershoot_mask_np)),
            "clipped_cell_count_after_limiter": int(np.count_nonzero(clipped_mask_np)),
            "final_left_inlet_zone_clipped_cell_count": int(
                np.count_nonzero(clipped_mask_np & left_zone_mask_np)
            ),
            "final_right_inlet_zone_clipped_cell_count": int(
                np.count_nonzero(clipped_mask_np & right_zone_mask_np)
            ),
            "final_inlet_wall_corner_zone_clipped_cell_count": int(
                np.count_nonzero(clipped_mask_np & corner_zone_mask_np)
            ),
            "final_left_inlet_zone_overshoot_cell_count": int(
                np.count_nonzero(overshoot_mask_np & left_zone_mask_np)
            ),
            "final_right_inlet_zone_overshoot_cell_count": int(
                np.count_nonzero(overshoot_mask_np & right_zone_mask_np)
            ),
            "final_inlet_wall_corner_zone_overshoot_cell_count": int(
                np.count_nonzero(overshoot_mask_np & corner_zone_mask_np)
            ),
            "final_left_inlet_zone_undershoot_cell_count": int(
                np.count_nonzero(undershoot_mask_np & left_zone_mask_np)
            ),
            "final_right_inlet_zone_undershoot_cell_count": int(
                np.count_nonzero(undershoot_mask_np & right_zone_mask_np)
            ),
            "final_inlet_wall_corner_zone_undershoot_cell_count": int(
                np.count_nonzero(undershoot_mask_np & corner_zone_mask_np)
            ),
            "final_clipped_fraction": float(
                np.count_nonzero(clipped_mask_np) / max(1, n_cells)
            ),
            "top_clipping_cells_final_step": [],
            "top_inlet_worst_cells_final_step": [],
        },
        "raw_update_audit": {
            "max_raw_delta_c": float(
                np.max(np.abs(raw_state_np - final_scalar_old.detach().cpu().numpy()))
            ),
            "mean_raw_delta_c": float(
                np.mean(np.abs(raw_state_np - final_scalar_old.detach().cpu().numpy()))
            ),
            "raw_overshoot_before_limiter": float(max_raw_overshoot),
            "raw_undershoot_before_limiter": float(max_raw_undershoot),
            "top_raw_update_offenders": [],
        },
        "conservation_audit": {
            "interior_pairwise_conservation_error_max_abs": 0.0,
            "interior_pairwise_conservation_error_mean_abs": 0.0,
            "boundary_scalar_flux_in_rate": float(history[-1]["inlet_scalar_flux_in"])
            if history
            else 0.0,
            "boundary_scalar_flux_out_rate": float(
                history[-1]["outlet_scalar_flux_out"]
            )
            if history
            else 0.0,
            "raw_boundary_scalar_flux_in_rate": float(
                history[-1]["raw_inlet_scalar_flux_in"]
            )
            if history
            else 0.0,
            "raw_boundary_scalar_flux_out_rate": float(
                history[-1]["raw_outlet_scalar_flux_out"]
            )
            if history
            else 0.0,
            "boundary_mass_in": float(history[-1]["mass_in"]) if history else 0.0,
            "boundary_mass_out": float(history[-1]["mass_out"]) if history else 0.0,
            "raw_boundary_mass_in": float(history[-1]["raw_mass_in"])
            if history
            else 0.0,
            "raw_boundary_mass_out": float(history[-1]["raw_mass_out"])
            if history
            else 0.0,
        },
        "local_cfl_audit": {
            "max_local_outgoing_cfl": max_out_cfl,
            "max_local_incoming_cfl": max_in_cfl,
            "top_cells_by_local_outgoing_cfl": _top_cell_metrics(
                (dt_substep * final_out_vol_rate_raw / vol).detach().cpu().numpy(),
                top_k=10,
            ),
            "top_cells_by_local_incoming_cfl": _top_cell_metrics(
                (dt_substep * final_in_vol_rate_raw / vol).detach().cpu().numpy(),
                top_k=10,
            ),
        },
        "limiter": {
            "limiter_active_count": int(
                np.count_nonzero(
                    (np.abs(final_limiter_out.detach().cpu().numpy() - 1.0) > 1e-12)
                    | (np.abs(final_limiter_in.detach().cpu().numpy() - 1.0) > 1e-12)
                )
            ),
            "limiter_active_fraction": float(
                np.count_nonzero(
                    (np.abs(final_limiter_out.detach().cpu().numpy() - 1.0) > 1e-12)
                    | (np.abs(final_limiter_in.detach().cpu().numpy() - 1.0) > 1e-12)
                )
                / max(1, n_cells)
            ),
            "limiter_max_flux_scale_down": float(
                np.max(
                    1.0
                    - np.minimum(
                        final_limiter_out.detach().cpu().numpy(),
                        final_limiter_in.detach().cpu().numpy(),
                    )
                )
            ),
            "limiter_mass_correction_total": float(
                np.sum(
                    np.abs(
                        final_in_raw.detach().cpu().numpy()
                        - final_in_lim.detach().cpu().numpy()
                    )
                )
            ),
            "conservation_error_after_limiter": float(
                np.sum(
                    (
                        final_scalar_old.detach().cpu().numpy() * mesh.cell_volumes
                        + final_in_lim.detach().cpu().numpy()
                        - final_out_lim.detach().cpu().numpy()
                    )
                    - (lim_state_np * mesh.cell_volumes)
                )
            ),
        },
        "masses": {
            "scalar_mass_initial": initial_mass,
            "scalar_mass_final": mass_final,
            "initial_mass_proxy": initial_mass,
            "final_mass_proxy": mass_final,
            "total_mass_proxy_final": mass_final,
            "cumulative_mass_in": cumulative_mass_in,
            "cumulative_mass_out": cumulative_mass_out,
            "cumulative_raw_mass_in": cumulative_raw_mass_in,
            "cumulative_raw_mass_out": cumulative_raw_mass_out,
            "cumulative_raw_net_mass_change_from_flux": float(
                cumulative_raw_mass_in - cumulative_raw_mass_out
            ),
            "mass_balance_error": float(
                mass_final - (initial_mass + cumulative_mass_in - cumulative_mass_out)
            ),
            "raw_mass_balance_error": float(
                mass_final
                - (initial_mass + cumulative_raw_mass_in - cumulative_raw_mass_out)
            ),
            "relative_mass_balance_error": float(
                abs(
                    mass_final
                    - (initial_mass + cumulative_mass_in - cumulative_mass_out)
                )
                / max(
                    abs(initial_mass),
                    abs(cumulative_mass_in),
                    abs(cumulative_mass_out),
                    abs(mass_final),
                    1e-30,
                )
            ),
        },
        "outlet_arrival": outlet_arrival_final,
        "diffusion_diagnostics": diffusion_diag,
        "final_stats": {
            "min": float(np.min(scalar_np)),
            "max": float(np.max(scalar_np)),
            "mean": float(np.mean(scalar_np)),
        },
        "final_step_debug": {
            "clipped_mask": clipped_mask_np,
            "overshoot_mask": overshoot_mask_np,
            "undershoot_mask": undershoot_mask_np,
            "left_inlet_cells": left_cells,
            "right_inlet_cells": right_cells,
            "outlet_cells": outlet_cells,
            "left_inlet_zone_mask": left_zone_mask_np,
            "right_inlet_zone_mask": right_zone_mask_np,
            "inlet_wall_corner_zone_mask": corner_zone_mask_np,
            "boundary_faces": np.asarray(mesh.boundary_face_indices, dtype=np.int64),
            "boundary_face_groups": boundary_face_groups_np,
        },
        "right_inlet_corner_debug": {
            "note": "torch path provides reduced corner diagnostics",
            "top_inlet_worst_cells_final_step": [],
        },
    }


def run_tetra_transport_debug(
    mesh: ImportedTetraMesh,
    config: GmshTetraTransportConfig,
    *,
    face_normal_velocity: np.ndarray,
    flux_diagnostics: Dict[str, float],
    initial_scalar: np.ndarray | None = None,
    start_step: int = 0,
    initial_cumulative_mass_in: float = 0.0,
    initial_cumulative_mass_out: float = 0.0,
    diffusion_boundary_kwargs: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Run transport-only prototype on tetra mesh with prescribed velocity fluxes."""

    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.dt <= 0:
        raise ValueError("dt must be positive.")
    if config.cfl_target <= 0 or config.cfl_target > 1.0:
        raise ValueError("cfl_target must be within (0, 1].")
    if not np.isfinite(config.auto_dt_percentile):
        raise ValueError("auto_dt_percentile must be finite.")
    if config.auto_dt_percentile < 0.0 or config.auto_dt_percentile > 100.0:
        raise ValueError("auto_dt_percentile must be within [0, 100].")
    if config.max_transport_substeps is not None and config.max_transport_substeps <= 0:
        raise ValueError("max_transport_substeps must be positive.")
    if config.transport_substep_warning_threshold <= 0:
        raise ValueError("transport_substep_warning_threshold must be positive.")
    if config.progress_total_steps is not None and config.progress_total_steps <= 0:
        raise ValueError("progress_total_steps must be positive when provided.")
    if (
        config.progress_target_end_time is not None
        and config.progress_target_end_time <= 0
    ):
        raise ValueError("progress_target_end_time must be positive when provided.")
    if config.physical_time_start < 0.0:
        raise ValueError("physical_time_start must be non-negative.")
    if config.dt_mode not in {"manual", "auto"}:
        raise ValueError(f"Unsupported dt_mode: {config.dt_mode!r}")
    if config.transport_mode not in {"advection", "advection_diffusion"}:
        raise ValueError(f"Unsupported transport_mode: {config.transport_mode!r}")
    if config.transport_scheme not in {"upwind", "bounded_upwind"}:
        raise ValueError(f"Unsupported transport_scheme: {config.transport_scheme!r}")
    if config.backend == "torch":
        return _run_tetra_transport_debug_torch(
            mesh,
            config,
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diagnostics,
            initial_scalar=initial_scalar,
            start_step=int(start_step),
            initial_cumulative_mass_in=float(initial_cumulative_mass_in),
            initial_cumulative_mass_out=float(initial_cumulative_mass_out),
            diffusion_boundary_kwargs=diffusion_boundary_kwargs,
        )

    inlet_groups = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet_groups["left_faces"], dtype=np.int64)
    right_faces = np.asarray(inlet_groups["right_faces"], dtype=np.int64)
    left_cells = np.asarray(inlet_groups["left_cells"], dtype=np.int64)
    right_cells = np.asarray(inlet_groups["right_cells"], dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    boundary_dirichlet, boundary_no_flux = build_diffusion_boundary_maps(
        mesh,
        inlet_groups,
        left_value=config.left_inlet_value,
        right_value=config.right_inlet_value,
    )

    if initial_scalar is None:
        scalar = np.zeros(mesh.tetrahedra.shape[0], dtype=np.float64)
    else:
        init = np.asarray(initial_scalar, dtype=np.float64)
        if init.shape != (mesh.tetrahedra.shape[0],):
            raise ValueError(
                "initial_scalar shape mismatch: "
                f"expected {(mesh.tetrahedra.shape[0],)}, got {init.shape}"
            )
        if not np.all(np.isfinite(init)):
            raise ValueError("initial_scalar contains non-finite values.")
        scalar = init.copy()
    boundary_face_groups = _build_boundary_face_groups(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    left_zone_mask, right_zone_mask, corner_zone_mask = _build_inlet_zone_masks(
        mesh,
        left_faces=left_faces,
        right_faces=right_faces,
        wall_faces=wall_faces,
    )
    inlet_zone_mask = left_zone_mask | right_zone_mask
    boundary_faces = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    cell_boundary_faces: list[list[int]] = [[] for _ in range(mesh.tetrahedra.shape[0])]
    for face_idx in boundary_faces.tolist():
        owner = int(mesh.face_to_cells[face_idx, 0])
        cell_boundary_faces[owner].append(face_idx)

    history: list[Dict[str, float | int | bool]] = []
    clipping_used = False
    overshoot_max = 0.0
    undershoot_max = 0.0
    overshoot_after_limiter_max = 0.0
    undershoot_after_limiter_max = 0.0
    safety_clamp_used = False
    safety_clamp_cell_count_total = 0
    safety_clamp_cell_count_final = 0
    safety_clamp_mass_delta_total = 0.0
    safety_clamp_max_correction = 0.0
    initial_mass = float(np.sum(scalar * mesh.cell_volumes))
    mass_prev = initial_mass
    cumulative_mass_in = float(initial_cumulative_mass_in)
    cumulative_mass_out = float(initial_cumulative_mass_out)
    cumulative_raw_mass_in = float(initial_cumulative_mass_in)
    cumulative_raw_mass_out = float(initial_cumulative_mass_out)
    thresholds = OUTLET_FRACTION_THRESHOLDS
    outlet_cells = (
        np.unique(mesh.face_to_cells[outlet_faces, 0])
        if outlet_faces.size
        else np.zeros((0,), dtype=np.int64)
    )

    diffusion_dt_limit = float("inf")
    generic_diffusion_precompute = None
    if config.transport_mode == "advection_diffusion" and config.diffusivity > 0.0:
        if diffusion_boundary_kwargs:
            generic_diffusion_precompute = build_scalar_backend_precompute(
                mesh,
                face_normal_velocity,
                backend="numpy",
                **dict(diffusion_boundary_kwargs),
            )
            diff_stencil = generic_diffusion_precompute.diffusion_stencil
            assert diff_stencil is not None
        else:
            diff_stencil = build_diffusion_stencil(
                mesh,
                boundary_dirichlet=boundary_dirichlet,
                boundary_no_flux_faces=boundary_no_flux,
            )
        diffusion_dt_limit = estimate_scalar_diffusion_dt_limit(
            diff_stencil, float(config.diffusivity)
        )

    dt_control = resolve_transport_dt_control(
        mesh,
        face_normal_velocity,
        config=config,
        diffusion_dt_limit=diffusion_dt_limit,
    )
    dt_control = {
        **dt_control,
        "cfl_target": float(config.cfl_target),
        "cfl_limit": float(config.cfl_limit),
    }
    used_dt = float(dt_control["used_dt"])
    dt_substep = float(dt_control["dt_substep"])
    transport_substep_count = int(dt_control["transport_substep_count"])
    diffusion_stability_warning = bool(
        np.isfinite(diffusion_dt_limit)
        and dt_substep > float(diffusion_dt_limit) + 1e-30
    )
    if bool(dt_control["transport_dt_controller_blocked"]):
        raise RuntimeError(str(dt_control["transport_dt_controller_reason"]))

    cfl_values, cfl_max = _compute_cfl_metrics(mesh, face_normal_velocity, dt_substep)
    cfl_warning = bool(cfl_max > config.cfl_limit)
    if config.dt_mode == "manual" and cfl_warning:
        print(
            "[gmsh-tetra-transport] WARNING: manual dt violates CFL limit, "
            f"used_dt={used_dt:.6e}, cfl_max={cfl_max:.6f}, cfl_limit={config.cfl_limit:.6f}"
        )

    final_raw_state = scalar.copy()
    final_limited_state = scalar.copy()
    final_updated_before_clip = scalar.copy()
    final_overshoot_mask = np.zeros_like(scalar, dtype=bool)
    final_undershoot_mask = np.zeros_like(scalar, dtype=bool)
    final_clipped_mask = np.zeros_like(scalar, dtype=bool)
    final_limiter_out = np.ones_like(scalar)
    final_limiter_in = np.ones_like(scalar)
    final_face_flux_raw = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_face_flux_limited = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_in_mass_raw = np.zeros_like(scalar)
    final_out_mass_raw = np.zeros_like(scalar)
    final_in_mass_lim = np.zeros_like(scalar)
    final_out_mass_lim = np.zeros_like(scalar)
    final_out_vol_rate_raw = np.zeros_like(scalar)
    final_in_vol_rate_raw = np.zeros_like(scalar)
    final_scalar_old = scalar.copy()
    final_face_q = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_face_c_up = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_face_mass_delta_owner = np.zeros(
        mesh.face_vertices.shape[0], dtype=np.float64
    )
    final_face_mass_delta_neighbor = np.zeros(
        mesh.face_vertices.shape[0], dtype=np.float64
    )
    final_face_contrib_owner = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_face_contrib_neighbor = np.zeros(
        mesh.face_vertices.shape[0], dtype=np.float64
    )
    final_pairwise_max = 0.0
    final_pairwise_mean = 0.0
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_steps_set = set(int(s) for s in config.snapshot_steps if int(s) > 0)

    for step_local in range(1, config.steps + 1):
        step = int(start_step + step_local)
        scalar_prev = scalar.copy()
        step_raw_mass_in = 0.0
        step_raw_mass_out = 0.0
        step_mass_in = 0.0
        step_mass_out = 0.0
        step_outgoing_cfl_max = 0.0
        step_incoming_cfl_max = 0.0
        step_raw_overshoot = 0.0
        step_raw_undershoot = 0.0
        step_limited_overshoot = 0.0
        step_limited_undershoot = 0.0
        step_overshoot = 0.0
        step_undershoot = 0.0
        step_diffusion_laplacian_max_abs = 0.0
        step_max_raw_delta_c = 0.0
        step_limiter_active_count = 0
        step_limiter_active_fraction = 0.0
        step_limiter_max_flux_scale_down = 0.0
        step_limiter_mass_correction_total = 0.0
        step_conservation_error_after_limiter = 0.0
        step_pairwise_max_abs = 0.0
        step_pairwise_mean_abs = 0.0
        step_overshoot_mask = np.zeros_like(scalar, dtype=bool)
        step_undershoot_mask = np.zeros_like(scalar, dtype=bool)
        step_clipped_mask = np.zeros_like(scalar, dtype=bool)

        for _substep in range(transport_substep_count):
            scalar_substep_prev = scalar.copy()
            assembled = _assemble_advection_fluxes(
                mesh,
                scalar,
                face_normal_velocity,
                boundary_face_groups=boundary_face_groups,
                left_inlet_value=config.left_inlet_value,
                right_inlet_value=config.right_inlet_value,
                dt=dt_substep,
            )
            limited = _apply_bounded_limiter(
                mesh,
                scalar,
                assembled,
                scheme=config.transport_scheme,
                dt=dt_substep,
            )

            updated_raw = np.asarray(
                limited["limited_state_before_clip"], dtype=np.float64
            )
            diffusion_laplacian_max_abs = 0.0
            if config.transport_mode == "advection_diffusion":
                if generic_diffusion_precompute is not None:
                    lap = scalar_laplacian_numpy(
                        generic_diffusion_precompute,
                        updated_raw,
                        gradient_method=config.gradient_method,
                        laplacian_method=config.laplacian_method,
                    )
                else:
                    lap = laplacian_scalar(
                        updated_raw,
                        mesh,
                        boundary_dirichlet=boundary_dirichlet,
                        boundary_no_flux_faces=boundary_no_flux,
                        method=config.laplacian_method,
                        gradient_method=config.gradient_method,
                    )
                diffusion_laplacian_max_abs = (
                    float(np.max(np.abs(lap))) if lap.size else 0.0
                )
                updated_raw = updated_raw + dt_substep * config.diffusivity * lap
            if config.source_term != 0.0:
                updated_raw = updated_raw + dt_substep * config.source_term

            if config.safety_clamp_after_diffusion:
                clamped = np.clip(updated_raw, config.clip_min, config.clip_max)
                delta = clamped - updated_raw
                count = int(np.count_nonzero(delta))
                mass_delta = float(np.sum(delta * mesh.cell_volumes))
                max_corr = float(np.max(np.abs(delta))) if delta.size else 0.0
                safety_clamp_cell_count_total += count
                safety_clamp_cell_count_final = count
                safety_clamp_mass_delta_total += mass_delta
                safety_clamp_max_correction = max(safety_clamp_max_correction, max_corr)
                if count > 0:
                    safety_clamp_used = True
                updated_raw = clamped

            clip_tol = 1e-12
            overshoot_mask = updated_raw > (config.clip_max + clip_tol)
            undershoot_mask = updated_raw < (config.clip_min - clip_tol)
            clipped_mask = overshoot_mask | undershoot_mask

            local_overshoot = float(max(np.max(updated_raw - config.clip_max), 0.0))
            local_undershoot = float(max(np.max(config.clip_min - updated_raw), 0.0))
            local_raw_overshoot = float(
                np.max(np.asarray(limited["raw_overshoot"], dtype=np.float64))
            )
            local_raw_undershoot = float(
                np.max(np.asarray(limited["raw_undershoot"], dtype=np.float64))
            )
            local_limited_overshoot = float(
                np.max(np.asarray(limited["limited_overshoot"], dtype=np.float64))
            )
            local_limited_undershoot = float(
                np.max(np.asarray(limited["limited_undershoot"], dtype=np.float64))
            )
            local_max_raw_delta_c = float(
                np.max(
                    np.abs(
                        np.asarray(
                            limited["raw_state_before_limiter"], dtype=np.float64
                        )
                        - np.asarray(scalar_substep_prev, dtype=np.float64)
                    )
                )
            )

            step_raw_overshoot = max(step_raw_overshoot, local_raw_overshoot)
            step_raw_undershoot = max(step_raw_undershoot, local_raw_undershoot)
            step_limited_overshoot = max(
                step_limited_overshoot, local_limited_overshoot
            )
            step_limited_undershoot = max(
                step_limited_undershoot, local_limited_undershoot
            )
            step_overshoot = max(step_overshoot, local_overshoot)
            step_undershoot = max(step_undershoot, local_undershoot)
            step_diffusion_laplacian_max_abs = max(
                step_diffusion_laplacian_max_abs, diffusion_laplacian_max_abs
            )
            step_max_raw_delta_c = max(step_max_raw_delta_c, local_max_raw_delta_c)

            if config.clipping_enabled:
                if local_overshoot > clip_tol or local_undershoot > clip_tol:
                    clipping_used = True
                updated = np.clip(updated_raw, config.clip_min, config.clip_max)
            else:
                updated = updated_raw
            scalar = updated
            if not np.all(np.isfinite(scalar)):
                raise FloatingPointError(
                    f"Scalar field became non-finite at step {step}."
                )

            raw_mass_in = dt_substep * float(limited["raw_inlet_scalar_flux_in"])
            raw_mass_out = dt_substep * float(limited["raw_outlet_scalar_flux_out"])
            mass_in = dt_substep * float(limited["effective_inlet_scalar_flux_in"])
            mass_out = dt_substep * float(limited["effective_outlet_scalar_flux_out"])
            out_vol_rate_raw = np.asarray(
                assembled["out_vol_rate_raw"], dtype=np.float64
            )
            in_vol_rate_raw = np.asarray(assembled["in_vol_rate_raw"], dtype=np.float64)
            local_outgoing_cfl = (
                dt_substep * out_vol_rate_raw / np.maximum(mesh.cell_volumes, 1e-16)
            )
            local_incoming_cfl = (
                dt_substep * in_vol_rate_raw / np.maximum(mesh.cell_volumes, 1e-16)
            )
            local_outgoing_cfl_max = float(np.max(local_outgoing_cfl))
            local_incoming_cfl_max = float(np.max(local_incoming_cfl))

            step_raw_mass_in += raw_mass_in
            step_raw_mass_out += raw_mass_out
            step_mass_in += mass_in
            step_mass_out += mass_out
            step_outgoing_cfl_max = max(step_outgoing_cfl_max, local_outgoing_cfl_max)
            step_incoming_cfl_max = max(step_incoming_cfl_max, local_incoming_cfl_max)
            step_limiter_active_count = max(
                step_limiter_active_count, int(limited["limiter_active_count"])
            )
            step_limiter_active_fraction = max(
                step_limiter_active_fraction,
                float(limited["limiter_active_fraction"]),
            )
            step_limiter_max_flux_scale_down = max(
                step_limiter_max_flux_scale_down,
                float(limited["limiter_max_flux_scale_down"]),
            )
            step_limiter_mass_correction_total += float(
                limited["limiter_mass_correction_total"]
            )
            step_conservation_error_after_limiter = max(
                step_conservation_error_after_limiter,
                abs(float(limited["conservation_error_after_limiter"])),
            )
            step_pairwise_max_abs = max(
                step_pairwise_max_abs,
                float(assembled["pairwise_conservation_error_max_abs"]),
            )
            step_pairwise_mean_abs = max(
                step_pairwise_mean_abs,
                float(assembled["pairwise_conservation_error_mean_abs"]),
            )
            step_overshoot_mask = overshoot_mask
            step_undershoot_mask = undershoot_mask
            step_clipped_mask = clipped_mask

            final_scalar_old = scalar_substep_prev
            final_raw_state = np.asarray(
                limited["raw_state_before_limiter"], dtype=np.float64
            )
            final_limited_state = np.asarray(
                limited["limited_state_before_clip"], dtype=np.float64
            )
            final_updated_before_clip = np.asarray(updated_raw, dtype=np.float64)
            final_overshoot_mask = overshoot_mask
            final_undershoot_mask = undershoot_mask
            final_clipped_mask = clipped_mask
            final_limiter_out = np.asarray(
                limited["limiter_scale_out"], dtype=np.float64
            )
            final_limiter_in = np.asarray(limited["limiter_scale_in"], dtype=np.float64)
            final_face_flux_raw = np.asarray(
                limited["face_scalar_flux_raw"], dtype=np.float64
            )
            final_face_flux_limited = np.asarray(
                limited["face_scalar_flux_limited"], dtype=np.float64
            )
            final_in_mass_raw = np.asarray(limited["in_mass_raw"], dtype=np.float64)
            final_out_mass_raw = np.asarray(limited["out_mass_raw"], dtype=np.float64)
            final_in_mass_lim = np.asarray(limited["in_mass_limited"], dtype=np.float64)
            final_out_mass_lim = np.asarray(
                limited["out_mass_limited"], dtype=np.float64
            )
            final_out_vol_rate_raw = out_vol_rate_raw
            final_in_vol_rate_raw = in_vol_rate_raw
            final_face_q = np.asarray(
                assembled["face_volumetric_flux_q"], dtype=np.float64
            )
            final_face_c_up = np.asarray(
                assembled["face_upwind_scalar"], dtype=np.float64
            )
            final_face_mass_delta_owner = np.asarray(
                assembled["face_mass_delta_owner"], dtype=np.float64
            )
            final_face_mass_delta_neighbor = np.asarray(
                assembled["face_mass_delta_neighbor"], dtype=np.float64
            )
            final_face_contrib_owner = np.asarray(
                assembled["face_contribution_c_owner"], dtype=np.float64
            )
            final_face_contrib_neighbor = np.asarray(
                assembled["face_contribution_c_neighbor"], dtype=np.float64
            )
            final_pairwise_max = float(assembled["pairwise_conservation_error_max_abs"])
            final_pairwise_mean = float(
                assembled["pairwise_conservation_error_mean_abs"]
            )

        overshoot_max = max(overshoot_max, step_overshoot)
        undershoot_max = max(undershoot_max, step_undershoot)
        overshoot_after_limiter_max = max(
            overshoot_after_limiter_max, step_limited_overshoot
        )
        undershoot_after_limiter_max = max(
            undershoot_after_limiter_max, step_limited_undershoot
        )

        cumulative_mass_in += step_mass_in
        cumulative_mass_out += step_mass_out
        cumulative_raw_mass_in += step_raw_mass_in
        cumulative_raw_mass_out += step_raw_mass_out

        max_change = float(np.max(np.abs(scalar - scalar_prev)))
        mean_change = float(np.mean(np.abs(scalar - scalar_prev)))
        mean_ref = float(np.mean(np.abs(scalar_prev)))
        relative_change = float(mean_change / max(mean_ref, 1e-30))
        mass = float(np.sum(scalar * mesh.cell_volumes))

        left_vals = scalar[left_cells] if left_cells.size else np.asarray([np.nan])
        right_vals = scalar[right_cells] if right_cells.size else np.asarray([np.nan])
        outlet_vals = (
            scalar[outlet_cells] if outlet_cells.size else np.asarray([np.nan])
        )
        outlet_arrival = {}
        for thr in thresholds:
            key = format_threshold_key("outlet_frac_gt", thr)
            if outlet_cells.size:
                outlet_arrival[key] = float(
                    np.count_nonzero(outlet_vals > thr) / outlet_vals.size
                )
            else:
                outlet_arrival[key] = 0.0

        overshoot_count = int(np.count_nonzero(step_overshoot_mask))
        undershoot_count = int(np.count_nonzero(step_undershoot_mask))
        clipped_count = int(np.count_nonzero(step_clipped_mask))
        clipped_fraction = float(clipped_count / max(1, scalar.size))

        left_zone_overshoot_count = int(
            np.count_nonzero(step_overshoot_mask & left_zone_mask)
        )
        right_zone_overshoot_count = int(
            np.count_nonzero(step_overshoot_mask & right_zone_mask)
        )
        corner_zone_overshoot_count = int(
            np.count_nonzero(step_overshoot_mask & corner_zone_mask)
        )
        left_zone_clipped_count = int(
            np.count_nonzero(step_clipped_mask & left_zone_mask)
        )
        right_zone_clipped_count = int(
            np.count_nonzero(step_clipped_mask & right_zone_mask)
        )
        corner_zone_clipped_count = int(
            np.count_nonzero(step_clipped_mask & corner_zone_mask)
        )
        left_zone_undershoot_count = int(
            np.count_nonzero(step_undershoot_mask & left_zone_mask)
        )
        right_zone_undershoot_count = int(
            np.count_nonzero(step_undershoot_mask & right_zone_mask)
        )
        corner_zone_undershoot_count = int(
            np.count_nonzero(step_undershoot_mask & corner_zone_mask)
        )

        entry: Dict[str, float | int | bool] = {
            "step": int(step),
            "min": float(np.min(scalar)),
            "max": float(np.max(scalar)),
            "mean": float(np.mean(scalar)),
            "total_mass_proxy": mass,
            "mass_delta": float(mass - mass_prev),
            "max_change": max_change,
            "mean_change": mean_change,
            "relative_change": relative_change,
            "left_inlet_min": float(np.min(left_vals)),
            "left_inlet_max": float(np.max(left_vals)),
            "right_inlet_min": float(np.min(right_vals)),
            "right_inlet_max": float(np.max(right_vals)),
            "cfl_max": step_outgoing_cfl_max,
            "cfl_warning": bool(step_outgoing_cfl_max > config.cfl_limit),
            "dt_used": used_dt,
            "dt_substep": dt_substep,
            "transport_substep_count": transport_substep_count,
            "raw_overshoot_before_limiter": step_raw_overshoot,
            "raw_undershoot_before_limiter": step_raw_undershoot,
            "overshoot_after_limiter_before_clip": step_limited_overshoot,
            "undershoot_after_limiter_before_clip": step_limited_undershoot,
            "overshoot_before_clip": step_overshoot,
            "undershoot_before_clip": step_undershoot,
            "overshoot_cell_count": overshoot_count,
            "undershoot_cell_count": undershoot_count,
            "clipped_cell_count": clipped_count,
            "clipped_cell_count_after_limiter": clipped_count,
            "clipped_fraction": clipped_fraction,
            "max_raw_delta_c": step_max_raw_delta_c,
            "left_inlet_zone_overshoot_cell_count": left_zone_overshoot_count,
            "right_inlet_zone_overshoot_cell_count": right_zone_overshoot_count,
            "inlet_wall_corner_zone_overshoot_cell_count": corner_zone_overshoot_count,
            "left_inlet_zone_undershoot_cell_count": left_zone_undershoot_count,
            "right_inlet_zone_undershoot_cell_count": right_zone_undershoot_count,
            "inlet_wall_corner_zone_undershoot_cell_count": corner_zone_undershoot_count,
            "left_inlet_zone_clipped_cell_count": left_zone_clipped_count,
            "right_inlet_zone_clipped_cell_count": right_zone_clipped_count,
            "inlet_wall_corner_zone_clipped_cell_count": corner_zone_clipped_count,
            "limiter_active_cell_count": step_limiter_active_count,
            "limiter_active_fraction": step_limiter_active_fraction,
            "limiter_max_flux_scale_down": step_limiter_max_flux_scale_down,
            "limiter_mass_correction_total": step_limiter_mass_correction_total,
            "conservation_error_after_limiter": step_conservation_error_after_limiter,
            "inlet_scalar_flux_in": float(step_mass_in / max(used_dt, 1e-30)),
            "outlet_scalar_flux_out": float(step_mass_out / max(used_dt, 1e-30)),
            "raw_inlet_scalar_flux_in": float(step_raw_mass_in / max(used_dt, 1e-30)),
            "raw_outlet_scalar_flux_out": float(
                step_raw_mass_out / max(used_dt, 1e-30)
            ),
            "effective_inlet_scalar_flux_in": float(step_mass_in / max(used_dt, 1e-30)),
            "effective_outlet_scalar_flux_out": float(
                step_mass_out / max(used_dt, 1e-30)
            ),
            "mass_in": step_mass_in,
            "mass_out": step_mass_out,
            "raw_mass_in": step_raw_mass_in,
            "raw_mass_out": step_raw_mass_out,
            "net_mass_change_from_flux": step_mass_in - step_mass_out,
            "raw_net_mass_change_from_flux": step_raw_mass_in - step_raw_mass_out,
            "cumulative_mass_in": cumulative_mass_in,
            "cumulative_mass_out": cumulative_mass_out,
            "cumulative_net_mass_change_from_flux": cumulative_mass_in
            - cumulative_mass_out,
            "cumulative_raw_mass_in": cumulative_raw_mass_in,
            "cumulative_raw_mass_out": cumulative_raw_mass_out,
            "cumulative_raw_net_mass_change_from_flux": cumulative_raw_mass_in
            - cumulative_raw_mass_out,
            "max_local_outgoing_cfl": step_outgoing_cfl_max,
            "max_local_incoming_cfl": step_incoming_cfl_max,
            "pairwise_conservation_error_max_abs": step_pairwise_max_abs,
            "pairwise_conservation_error_mean_abs": step_pairwise_mean_abs,
            "diffusion_laplacian_max_abs": step_diffusion_laplacian_max_abs,
        }
        entry.update(outlet_arrival)
        physical_time = float(config.physical_time_start + step_local * used_dt)
        total_progress_steps = int(
            config.progress_total_steps
            if config.progress_total_steps is not None
            else start_step + config.steps
        )
        entry["physical_time"] = physical_time
        entry["step_progress_percent"] = float(
            100.0 * step / max(1, total_progress_steps)
        )
        if config.progress_target_end_time is not None:
            entry["time_progress_percent"] = float(
                min(
                    100.0,
                    100.0
                    * physical_time
                    / max(float(config.progress_target_end_time), 1e-30),
                )
            )
        history.append(entry)
        mass_prev = mass

        if config.progress_every > 0 and step_local % config.progress_every == 0:
            print(
                "[gmsh-tetra-transport] "
                f"step {step}/{start_step + config.steps}, min={entry['min']:.6f}, "
                f"max={entry['max']:.6f}, cfl={entry['cfl_max']:.4f}, "
                f"substeps={transport_substep_count}, "
                f"progress={entry['step_progress_percent']:.1f}%"
            )
        if step in snapshot_steps_set:
            snapshots[int(step)] = scalar.copy()

    # Global top clipping diagnostics
    severity = np.maximum(
        final_updated_before_clip - config.clip_max,
        config.clip_min - final_updated_before_clip,
    )
    violating = np.where(final_clipped_mask)[0]
    if violating.size:
        worst = violating[np.argsort(severity[violating])[::-1][:10]]
    else:
        worst = np.zeros((0,), dtype=np.int64)
    top_clipping_cells = [
        {
            "cell_index": int(idx),
            "cell_center": mesh.cell_centers[int(idx)].tolist(),
            "value_before_clip": float(final_updated_before_clip[int(idx)]),
            "overshoot": float(
                max(final_updated_before_clip[int(idx)] - config.clip_max, 0.0)
            ),
            "undershoot": float(
                max(config.clip_min - final_updated_before_clip[int(idx)], 0.0)
            ),
        }
        for idx in worst
    ]

    # Raw finite-volume update offenders (before limiter)
    raw_delta = final_raw_state - final_scalar_old
    raw_delta_abs = np.abs(raw_delta)
    top_raw_indices = np.argsort(raw_delta_abs)[::-1][:30]
    top_raw_update_offenders: list[Dict[str, object]] = []
    for idx in top_raw_indices.tolist():
        face_ids = mesh.cell_to_faces[int(idx)].tolist()
        face_items: list[Dict[str, object]] = []
        boundary_groups = set()
        for fid in face_ids:
            owner = int(mesh.face_to_cells[fid, 0])
            neigh = int(mesh.face_to_cells[fid, 1])
            is_owner = owner == int(idx)
            signed_scalar_flux = (
                float(final_face_flux_raw[fid])
                if is_owner
                else -float(final_face_flux_raw[fid])
            )
            signed_dt_flux = -dt_substep * signed_scalar_flux
            face_item = {
                "face_index": int(fid),
                "neighbor_or_boundary": (
                    int(neigh if is_owner else owner) if neigh >= 0 else "boundary"
                ),
                "boundary_name": (
                    BOUNDARY_GROUP_NAME.get(int(boundary_face_groups[fid]), "unknown")
                    if neigh < 0
                    else None
                ),
                "u_dot_n_for_cell": float(
                    face_normal_velocity[fid]
                    if is_owner
                    else -face_normal_velocity[fid]
                ),
                "volumetric_flux_Q": float(
                    final_face_q[fid] if is_owner else -final_face_q[fid]
                ),
                "C_upwind": float(final_face_c_up[fid]),
                "scalar_flux": signed_scalar_flux,
                "dt_scalar_flux": signed_dt_flux,
                "contribution_to_C": float(
                    signed_dt_flux / max(mesh.cell_volumes[idx], 1e-16)
                ),
            }
            if neigh < 0:
                boundary_groups.add(str(face_item["boundary_name"]))
            face_items.append(face_item)
        incoming_mass = float(final_in_mass_raw[int(idx)])
        outgoing_mass = float(final_out_mass_raw[int(idx)])
        net_mass_delta = incoming_mass - outgoing_mass
        top_raw_update_offenders.append(
            {
                "cell_index": int(idx),
                "center": mesh.cell_centers[int(idx)].tolist(),
                "volume": float(mesh.cell_volumes[int(idx)]),
                "C_old": float(final_scalar_old[int(idx)]),
                "C_raw": float(final_raw_state[int(idx)]),
                "delta_C_raw": float(raw_delta[int(idx)]),
                "raw_mass_old": float(
                    final_scalar_old[int(idx)] * mesh.cell_volumes[int(idx)]
                ),
                "raw_mass_new": float(
                    final_raw_state[int(idx)] * mesh.cell_volumes[int(idx)]
                ),
                "incoming_mass_sum": incoming_mass,
                "outgoing_mass_sum": outgoing_mass,
                "net_mass_delta": net_mass_delta,
                "dt": float(dt_substep),
                "net_mass_delta_over_volume": float(
                    net_mass_delta / max(mesh.cell_volumes[int(idx)], 1e-16)
                ),
                "adjacent_boundary_groups": sorted(boundary_groups),
                "adjacent_faces": face_items,
            }
        )

    # Inlet corner focused diagnostics
    raw_overshoot_all = np.maximum(final_raw_state - config.clip_max, 0.0)
    corner_candidates = np.where(corner_zone_mask & (raw_overshoot_all > 0.0))[0]
    if corner_candidates.size:
        corner_worst = corner_candidates[
            np.argsort(raw_overshoot_all[corner_candidates])[::-1][:20]
        ]
    else:
        corner_worst = np.zeros((0,), dtype=np.int64)

    def _cell_prescribed_group(cell_idx: int) -> str:
        in_left = bool(left_zone_mask[cell_idx])
        in_right = bool(right_zone_mask[cell_idx])
        if in_left and in_right:
            return "left_and_right_inlet_zone"
        if in_left:
            return "left_inlet_zone"
        if in_right:
            return "right_inlet_zone"
        return "none"

    top_inlet_worst_cells: list[Dict[str, object]] = []
    for idx in corner_worst.tolist():
        face_ids = mesh.cell_to_faces[int(idx)].tolist()
        boundary_face_info = []
        interior_face_info = []
        boundary_groups = set()
        for fid in face_ids:
            owner = int(mesh.face_to_cells[fid, 0])
            neigh = int(mesh.face_to_cells[fid, 1])
            if neigh < 0:
                g_name = BOUNDARY_GROUP_NAME.get(
                    int(boundary_face_groups[fid]), "unknown"
                )
                boundary_groups.add(g_name)
                vn = float(face_normal_velocity[fid])
                boundary_face_info.append(
                    {
                        "face_index": int(fid),
                        "boundary_group": g_name,
                        "u_dot_n": vn,
                        "flow_sign": "inflow" if vn < 0.0 else "outflow_or_zero",
                        "scalar_flux_raw_signed_owner": float(final_face_flux_raw[fid]),
                        "scalar_flux_limited_signed_owner": float(
                            final_face_flux_limited[fid]
                        ),
                    }
                )
            else:
                vn_owner = float(
                    face_normal_velocity[fid]
                    if owner == int(idx)
                    else -face_normal_velocity[fid]
                )
                phi_raw_owner = float(
                    final_face_flux_raw[fid]
                    if owner == int(idx)
                    else -final_face_flux_raw[fid]
                )
                phi_lim_owner = float(
                    final_face_flux_limited[fid]
                    if owner == int(idx)
                    else -final_face_flux_limited[fid]
                )
                interior_face_info.append(
                    {
                        "face_index": int(fid),
                        "u_dot_n_for_cell": vn_owner,
                        "scalar_flux_raw_signed_cell": phi_raw_owner,
                        "scalar_flux_limited_signed_cell": phi_lim_owner,
                    }
                )
        limiter_scale = float(
            min(final_limiter_out[int(idx)], final_limiter_in[int(idx)])
        )
        top_inlet_worst_cells.append(
            {
                "cell_index": int(idx),
                "center": mesh.cell_centers[int(idx)].tolist(),
                "volume": float(mesh.cell_volumes[int(idx)]),
                "C_old": float(final_scalar_old[int(idx)]),
                "C_raw": float(final_raw_state[int(idx)]),
                "C_limited": float(final_limited_state[int(idx)]),
                "incoming_mass_sum_raw": float(final_in_mass_raw[int(idx)]),
                "outgoing_mass_sum_raw": float(final_out_mass_raw[int(idx)]),
                "incoming_mass_sum_limited": float(final_in_mass_lim[int(idx)]),
                "outgoing_mass_sum_limited": float(final_out_mass_lim[int(idx)]),
                "limiter_scale": limiter_scale,
                "prescribed_bc_group": _cell_prescribed_group(int(idx)),
                "adjacent_boundary_face_groups": sorted(boundary_groups),
                "adjacent_boundary_faces": boundary_face_info,
                "adjacent_interior_face_fluxes": interior_face_info,
                "raw_overshoot_before_limiter": float(raw_overshoot_all[int(idx)]),
            }
        )

    outlet_arrival_final: Dict[str, float] = {}
    if outlet_cells.size:
        outlet_final_vals = scalar[outlet_cells]
        for thr in thresholds:
            outlet_arrival_final[format_threshold_key("outlet_frac_gt", thr)] = float(
                np.count_nonzero(outlet_final_vals > thr) / outlet_final_vals.size
            )
    else:
        for thr in thresholds:
            outlet_arrival_final[format_threshold_key("outlet_frac_gt", thr)] = 0.0

    final_local_outgoing_cfl = (
        dt_substep * final_out_vol_rate_raw / np.maximum(mesh.cell_volumes, 1e-16)
    )
    final_local_incoming_cfl = (
        dt_substep * final_in_vol_rate_raw / np.maximum(mesh.cell_volumes, 1e-16)
    )
    max_local_outgoing_cfl = (
        float(np.max(final_local_outgoing_cfl))
        if final_local_outgoing_cfl.size
        else 0.0
    )
    max_local_incoming_cfl = (
        float(np.max(final_local_incoming_cfl))
        if final_local_incoming_cfl.size
        else 0.0
    )

    return {
        "scalar": scalar,
        "history": history,
        "cfl_values": final_local_outgoing_cfl,
        "cfl_max": max_local_outgoing_cfl,
        "cfl_warning": bool(max_local_outgoing_cfl > config.cfl_limit),
        "dt_control": dt_control,
        "flux_diagnostics": flux_diagnostics,
        "boundary_setup": {
            "left_inlet_faces": int(left_faces.size),
            "right_inlet_faces": int(right_faces.size),
            "outlet_faces": int(outlet_faces.size),
            "wall_faces": int(wall_faces.size),
            "left_inlet_cells": int(left_cells.size),
            "right_inlet_cells": int(right_cells.size),
        },
        "transport_info": {
            "transport_mode": config.transport_mode,
            "transport_scheme": config.transport_scheme,
            "gradient_method": config.gradient_method,
            "laplacian_method": config.laplacian_method,
            "flow_solved": False,
            "pressure_solved": False,
            "velocity_source": "prescribed_debug_field",
            "scalar_bc_mode": "inflow_only_dirichlet_face_flux",
            "inlet_dirichlet_applied_on": "faces",
            "cell_level_inlet_overwrite_used": False,
            "post_step_cell_overwrite_used": False,
            "bounded_limiter_enabled": bool(
                config.transport_scheme == "bounded_upwind"
            ),
            "safety_clamp_after_diffusion_enabled": bool(
                config.safety_clamp_after_diffusion
            ),
            "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
        },
        "backend_execution": {
            "requested_backend": config.backend,
            "stepping_backend": "numpy",
            "device": "cpu",
            "torch_device": config.torch_device,
            # Current transport stepping is numpy-based by design.
            "used_numpy_fallback": bool(config.backend == "torch"),
            "all_core_arrays_on_cuda": False,
            "per_step_large_cpu_gpu_transfer_warning": False,
            "cpu_gpu_transfer_notes": [],
        },
        "clipping": {
            "enabled": bool(config.clipping_enabled),
            "used": bool(clipping_used),
            "post_step_clipping_used": bool(clipping_used),
            "safety_clamp_after_diffusion_enabled": bool(
                config.safety_clamp_after_diffusion
            ),
            "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
            "safety_clamp_cell_count_total": int(safety_clamp_cell_count_total),
            "safety_clamp_cell_count_final_step": int(safety_clamp_cell_count_final),
            "safety_clamp_mass_delta": float(safety_clamp_mass_delta_total),
            "safety_clamp_max_correction": float(safety_clamp_max_correction),
            "max_overshoot_before_clip": float(overshoot_max),
            "max_undershoot_before_clip": float(undershoot_max),
            "max_overshoot_after_limiter_before_clip": float(
                overshoot_after_limiter_max
            ),
            "max_undershoot_after_limiter_before_clip": float(
                undershoot_after_limiter_max
            ),
            "final_clipped_cell_count": int(np.count_nonzero(final_clipped_mask)),
            "final_overshoot_cell_count": int(np.count_nonzero(final_overshoot_mask)),
            "final_undershoot_cell_count": int(np.count_nonzero(final_undershoot_mask)),
            "clipped_cell_count_after_limiter": int(
                np.count_nonzero(final_clipped_mask)
            ),
            "final_left_inlet_zone_clipped_cell_count": int(
                np.count_nonzero(final_clipped_mask & left_zone_mask)
            ),
            "final_right_inlet_zone_clipped_cell_count": int(
                np.count_nonzero(final_clipped_mask & right_zone_mask)
            ),
            "final_inlet_wall_corner_zone_clipped_cell_count": int(
                np.count_nonzero(final_clipped_mask & corner_zone_mask)
            ),
            "final_left_inlet_zone_overshoot_cell_count": int(
                np.count_nonzero(final_overshoot_mask & left_zone_mask)
            ),
            "final_right_inlet_zone_overshoot_cell_count": int(
                np.count_nonzero(final_overshoot_mask & right_zone_mask)
            ),
            "final_inlet_wall_corner_zone_overshoot_cell_count": int(
                np.count_nonzero(final_overshoot_mask & corner_zone_mask)
            ),
            "final_left_inlet_zone_undershoot_cell_count": int(
                np.count_nonzero(final_undershoot_mask & left_zone_mask)
            ),
            "final_right_inlet_zone_undershoot_cell_count": int(
                np.count_nonzero(final_undershoot_mask & right_zone_mask)
            ),
            "final_inlet_wall_corner_zone_undershoot_cell_count": int(
                np.count_nonzero(final_undershoot_mask & corner_zone_mask)
            ),
            "final_clipped_fraction": float(
                np.count_nonzero(final_clipped_mask) / max(1, scalar.size)
            ),
            "top_clipping_cells_final_step": top_clipping_cells,
            "top_inlet_worst_cells_final_step": top_inlet_worst_cells,
        },
        "raw_update_audit": {
            "max_raw_delta_c": float(np.max(raw_delta_abs))
            if raw_delta_abs.size
            else 0.0,
            "mean_raw_delta_c": float(np.mean(raw_delta_abs))
            if raw_delta_abs.size
            else 0.0,
            "raw_overshoot_before_limiter": float(np.max(raw_overshoot_all))
            if raw_overshoot_all.size
            else 0.0,
            "raw_undershoot_before_limiter": float(
                np.max(np.maximum(config.clip_min - final_raw_state, 0.0))
            )
            if final_raw_state.size
            else 0.0,
            "top_raw_update_offenders": top_raw_update_offenders,
        },
        "conservation_audit": {
            "interior_pairwise_conservation_error_max_abs": float(final_pairwise_max),
            "interior_pairwise_conservation_error_mean_abs": float(final_pairwise_mean),
            "boundary_scalar_flux_in_rate": float(history[-1]["inlet_scalar_flux_in"]),
            "boundary_scalar_flux_out_rate": float(
                history[-1]["outlet_scalar_flux_out"]
            ),
            "raw_boundary_scalar_flux_in_rate": float(
                history[-1]["raw_inlet_scalar_flux_in"]
            ),
            "raw_boundary_scalar_flux_out_rate": float(
                history[-1]["raw_outlet_scalar_flux_out"]
            ),
            "boundary_mass_in": float(history[-1]["mass_in"]),
            "boundary_mass_out": float(history[-1]["mass_out"]),
            "raw_boundary_mass_in": float(history[-1]["raw_mass_in"]),
            "raw_boundary_mass_out": float(history[-1]["raw_mass_out"]),
        },
        "local_cfl_audit": {
            "max_local_outgoing_cfl": max_local_outgoing_cfl,
            "max_local_incoming_cfl": max_local_incoming_cfl,
            "top_cells_by_local_outgoing_cfl": _top_cell_metrics(
                final_local_outgoing_cfl, top_k=10
            ),
            "top_cells_by_local_incoming_cfl": _top_cell_metrics(
                final_local_incoming_cfl, top_k=10
            ),
        },
        "limiter": {
            "limiter_active_cell_count": int(
                np.count_nonzero((final_limiter_out < 1.0) | (final_limiter_in < 1.0))
            ),
            "limiter_active_fraction": float(
                np.count_nonzero((final_limiter_out < 1.0) | (final_limiter_in < 1.0))
                / max(1, scalar.size)
            ),
            "limiter_max_flux_scale_down": float(
                1.0
                - min(float(np.min(final_limiter_out)), float(np.min(final_limiter_in)))
            ),
            "limiter_mass_correction_total": float(
                np.sum(
                    np.abs(
                        (final_in_mass_raw - final_out_mass_raw)
                        - (final_in_mass_lim - final_out_mass_lim)
                    )
                )
            ),
            "conservation_error_after_limiter": float(
                (np.sum(final_in_mass_lim) - np.sum(final_out_mass_lim))
                - (np.sum(final_in_mass_raw) - np.sum(final_out_mass_raw))
            ),
        },
        "masses": {
            "scalar_mass_initial": float(initial_mass),
            "scalar_mass_final": float(np.sum(scalar * mesh.cell_volumes)),
            "initial_mass_proxy": float(initial_mass),
            "final_mass_proxy": float(np.sum(scalar * mesh.cell_volumes)),
            "total_mass_proxy_final": float(np.sum(scalar * mesh.cell_volumes)),
            "cumulative_mass_in": float(cumulative_mass_in),
            "cumulative_mass_out": float(cumulative_mass_out),
            "cumulative_net_mass_change_from_flux": float(
                cumulative_mass_in - cumulative_mass_out
            ),
            "cumulative_raw_mass_in": float(cumulative_raw_mass_in),
            "cumulative_raw_mass_out": float(cumulative_raw_mass_out),
            "cumulative_raw_net_mass_change_from_flux": float(
                cumulative_raw_mass_in - cumulative_raw_mass_out
            ),
            "net_mass_change_from_state": float(
                np.sum(scalar * mesh.cell_volumes) - initial_mass
            ),
            "mass_balance_error": float(
                np.sum(scalar * mesh.cell_volumes)
                - (initial_mass + cumulative_mass_in - cumulative_mass_out)
            ),
            "raw_mass_balance_error": float(
                np.sum(scalar * mesh.cell_volumes)
                - (initial_mass + cumulative_raw_mass_in - cumulative_raw_mass_out)
            ),
            "relative_mass_balance_error": float(
                abs(
                    np.sum(scalar * mesh.cell_volumes)
                    - (initial_mass + cumulative_mass_in - cumulative_mass_out)
                )
                / max(
                    abs(initial_mass),
                    abs(cumulative_mass_in),
                    abs(cumulative_mass_out),
                    abs(np.sum(scalar * mesh.cell_volumes)),
                    1e-30,
                )
            ),
        },
        "outlet_arrival": outlet_arrival_final,
        "diffusion_diagnostics": {
            "diffusion_backend": "numpy"
            if config.transport_mode == "advection_diffusion"
            else "none",
            "laplacian_method": config.laplacian_method,
            "gradient_method": config.gradient_method,
            "diffusivity": float(config.diffusivity),
            "diffusion_dt_limit": diffusion_dt_limit,
            "diffusion_stability_warning": bool(diffusion_stability_warning),
            "max_overshoot_after_diffusion_before_clip": float(overshoot_max),
            "max_undershoot_after_diffusion_before_clip": float(undershoot_max),
            "safety_clamp_after_diffusion_enabled": bool(
                config.safety_clamp_after_diffusion
            ),
            "safety_clamp_after_diffusion_used": bool(safety_clamp_used),
        },
        "right_inlet_corner_debug": {
            "number_of_cells_in_right_inlet_corner_zone": int(
                np.count_nonzero(corner_zone_mask & right_zone_mask)
            ),
            "max_C_before_limiter": float(
                np.max(final_raw_state[corner_zone_mask & right_zone_mask])
            )
            if np.any(corner_zone_mask & right_zone_mask)
            else float("nan"),
            "max_C_after_limiter": float(
                np.max(final_limited_state[corner_zone_mask & right_zone_mask])
            )
            if np.any(corner_zone_mask & right_zone_mask)
            else float("nan"),
            "top_20_cells_by_raw_overshoot_before_limiter": top_inlet_worst_cells,
        },
        "final_step_debug": {
            "raw_state_before_limiter": final_raw_state,
            "limited_state_before_clip": final_limited_state,
            "updated_raw_before_clip": final_updated_before_clip,
            "overshoot_mask": final_overshoot_mask,
            "undershoot_mask": final_undershoot_mask,
            "clipped_mask": final_clipped_mask,
            "outlet_cells": outlet_cells,
            "left_inlet_cells": left_cells,
            "right_inlet_cells": right_cells,
            "left_inlet_zone_mask": left_zone_mask,
            "right_inlet_zone_mask": right_zone_mask,
            "inlet_wall_corner_zone_mask": corner_zone_mask,
            "inlet_zone_mask": inlet_zone_mask,
            "boundary_faces": boundary_faces,
            "boundary_face_groups": boundary_face_groups,
            "face_volumetric_flux_q": final_face_q,
            "face_scalar_flux_raw": final_face_flux_raw,
            "face_upwind_scalar": final_face_c_up,
            "face_mass_delta_owner": final_face_mass_delta_owner,
            "face_mass_delta_neighbor": final_face_mass_delta_neighbor,
            "face_contribution_c_owner": final_face_contrib_owner,
            "face_contribution_c_neighbor": final_face_contrib_neighbor,
        },
        "final_stats": {
            "min": float(np.min(scalar)),
            "max": float(np.max(scalar)),
            "mean": float(np.mean(scalar)),
        },
        "snapshots": snapshots,
    }
