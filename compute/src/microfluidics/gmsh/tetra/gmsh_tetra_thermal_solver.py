"""Tetra-native thermal advection-diffusion solver on imported Gmsh meshes."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Dict, Literal, Mapping

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_backend import (
    BackendSelection,
    select_backend,
)
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    GradientMethod,
    LaplacianMethod,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_backend import (
    apply_bounded_limiter_numpy,
    assemble_advection_fluxes_numpy,
    assemble_advection_fluxes_torch,
    apply_bounded_limiter_torch,
    build_backend_execution_diagnostics,
    build_scalar_backend_precompute,
    compute_cfl_metrics,
    estimate_diffusion_dt_limit,
    estimate_stable_dt,
    laplacian_numpy,
    laplacian_torch,
    ScalarBackendPrecompute,
)
from microfluidics.gmsh.tetra.gmsh_tetra_transport_solver import _top_cell_metrics

ThermalDtMode = Literal["manual", "auto"]
ThermalLimiterScheme = Literal["upwind", "bounded_upwind"]
ThermalExecutionProfile = Literal["production", "debug", "reference"]
ThermalDiagnosticsMode = Literal["auto", "debug", "fast"]
ThermalTorchCompileMode = Literal["auto", "on", "off"]
ThermalHistoryMode = Literal["auto", "dict", "compact", "off"]
ThermalWallBoundaryMode = Literal["adiabatic", "fixed_temperature", "fixed_heat_flux"]
ThermalStepHistoryEntry = Dict[str, float | int | bool]


@dataclass(frozen=True)
class GmshTetraThermalConfig:
    steps: int = 200
    dt: float = 5e-4
    dt_mode: ThermalDtMode = "auto"
    cfl_target: float = 0.5
    cfl_limit: float = 0.8
    diffusion_stability_factor: float = 1.0
    thermal_diffusivity: float = 1.4e-7
    rho: float = 1000.0
    cp: float = 4180.0
    heat_source: float = 0.0
    initial_temperature: float = 298.15
    left_inlet_temperature: float = 303.15
    right_inlet_temperature: float = 303.15
    wall_boundary_mode: ThermalWallBoundaryMode = "adiabatic"
    heated_wall_physical_groups: tuple[str, ...] = ()
    wall_temperature: float | None = None
    wall_heat_flux: float | None = None
    limiter_scheme: ThermalLimiterScheme = "bounded_upwind"
    normalize_bounded_advection: bool = True
    limit_non_advective_update: bool = True
    clipping_enabled: bool = True
    min_temperature: float | None = None
    max_temperature: float | None = None
    progress_every: int = 20
    gradient_method: GradientMethod = "least_squares"
    laplacian_method: LaplacianMethod = "lsq_flux"
    backend: Literal["auto", "numpy", "torch"] = "auto"
    torch_device: str = "cpu"
    execution_profile: ThermalExecutionProfile = "debug"
    allow_numpy_production_fallback: bool = False
    diagnostics_mode: ThermalDiagnosticsMode = "auto"
    torch_compile_step_body: ThermalTorchCompileMode = "auto"
    collect_history: bool = True
    history_mode: ThermalHistoryMode = "auto"
    history_stride: int = 100
    diagnostics_stride: int = 0
    full_step_diagnostics: bool = False
    debug_artifacts: bool = False
    snapshot_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ResolvedWallBoundary:
    faces: np.ndarray
    owners: np.ndarray
    areas: np.ndarray
    group_face_counts: dict[str, int]
    group_tags: dict[str, int]

    @property
    def area(self) -> float:
        return float(np.sum(self.areas))


def _thermal_v2_model_capabilities() -> dict[str, object]:
    return {
        "contract_version": "thermal_v2",
        "model_family": "passive_scalar_advection_diffusion",
        "supported_features": [
            "prescribed_velocity_field",
            "inflow_temperature_dirichlet_on_inflow_faces",
            "constant_thermal_diffusivity",
            "uniform_volumetric_heat_source",
            "adiabatic_walls_zero_diffusive_flux",
            "wall_fixed_temperature",
            "wall_prescribed_heat_flux",
            "advective_outflow_with_backflow_suppression",
        ],
        "unsupported_features": [
            "conjugate_heat_transfer",
            "temperature_dependent_material_properties",
            "buoyancy_or_temperature_to_flow_coupling",
            "reaction_heat_release_coupling",
            "turbulent_heat_transfer",
        ],
        "boundary_contract": {
            "inlet": "temperature Dirichlet is enforced only on inflow faces",
            "walls": "adiabatic, fixed temperature, or prescribed heat flux on selected physical groups",
            "outlet": "advective outflow with backflow suppression and zero diffusive flux",
        },
        "production_claim_note": (
            "Production performance claims require benchmark acceptance with "
            "explicit target mesh bounds, runtime and memory budgets, and "
            "compile-path usage requirements."
        ),
    }


def _build_thermal_observables(
    mesh: ImportedTetraMesh,
    temperature: np.ndarray,
    face_normal_velocity: np.ndarray,
) -> dict[str, object]:
    temperature_np = np.asarray(temperature, dtype=np.float64)
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    safe_volumes = np.maximum(volumes, 1e-16)
    total_volume = float(np.sum(safe_volumes))
    domain_volume_weighted_temperature = (
        float(np.sum(temperature_np * safe_volumes) / total_volume)
        if total_volume > 0.0
        else float(np.mean(temperature_np))
    )

    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    outlet_face_count = int(outlet_faces.size)
    outlet_sample_cell_count = 0
    outlet_temperature_flux_weighted: float | None = None
    outlet_temperature_flux_weighted_std: float | None = None
    outlet_temperature_cell_mean: float | None = None
    outlet_temperature_min: float | None = None
    outlet_temperature_max: float | None = None
    outlet_temperature_range: float | None = None
    outlet_outflow_rate = 0.0

    if outlet_face_count > 0:
        face_normal_velocity_np = np.asarray(face_normal_velocity, dtype=np.float64)
        outlet_cells = np.asarray(mesh.face_to_cells[outlet_faces, 0], dtype=np.int64)
        valid_outlet_mask = outlet_cells >= 0
        if np.any(valid_outlet_mask):
            unique_cells = np.unique(outlet_cells[valid_outlet_mask])
            outlet_sample_cell_count = int(unique_cells.size)
            outlet_cell_temperatures = temperature_np[unique_cells]
            outlet_temperature_cell_mean = float(np.mean(outlet_cell_temperatures))
            outlet_temperature_min = float(np.min(outlet_cell_temperatures))
            outlet_temperature_max = float(np.max(outlet_cell_temperatures))
            outlet_temperature_range = float(
                outlet_temperature_max - outlet_temperature_min
            )

        outlet_face_flux = face_normal_velocity_np[outlet_faces] * np.asarray(
            mesh.face_areas[outlet_faces], dtype=np.float64
        )
        outlet_positive_flux = np.maximum(outlet_face_flux, 0.0)
        if np.any(valid_outlet_mask):
            outlet_positive_flux = np.where(
                valid_outlet_mask, outlet_positive_flux, 0.0
            )
        outlet_outflow_rate = float(np.sum(outlet_positive_flux))
        if outlet_outflow_rate > 0.0 and np.any(valid_outlet_mask):
            outlet_temperature_flux_weighted = float(
                np.sum(
                    outlet_positive_flux[valid_outlet_mask]
                    * temperature_np[outlet_cells[valid_outlet_mask]]
                )
                / outlet_outflow_rate
            )
            outlet_temperature_flux_weighted_std = float(
                np.sqrt(
                    np.sum(
                        outlet_positive_flux[valid_outlet_mask]
                        * (
                            temperature_np[outlet_cells[valid_outlet_mask]]
                            - outlet_temperature_flux_weighted
                        )
                        ** 2
                    )
                    / outlet_outflow_rate
                )
            )

    return {
        "temperature_units": "K",
        "domain_volume_weighted_temperature": domain_volume_weighted_temperature,
        "outlet_temperature_flux_weighted": outlet_temperature_flux_weighted,
        "outlet_temperature_flow_weighted_mean": outlet_temperature_flux_weighted,
        "outlet_temperature_flow_weighted_std": outlet_temperature_flux_weighted_std,
        "outlet_temperature_cell_mean": outlet_temperature_cell_mean,
        "outlet_temperature_min": outlet_temperature_min,
        "outlet_temperature_max": outlet_temperature_max,
        "outlet_temperature_range": outlet_temperature_range,
        "outlet_owner_cell_temperature_min": outlet_temperature_min,
        "outlet_owner_cell_temperature_max": outlet_temperature_max,
        "outlet_owner_cell_temperature_range": outlet_temperature_range,
        "outlet_outflow_rate": float(outlet_outflow_rate),
        "outlet_sample_face_count": outlet_face_count,
        "outlet_sample_cell_count": int(outlet_sample_cell_count),
    }


def _compute_bounds_violation(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    overshoot = np.maximum(values - upper, 0.0)
    undershoot = np.maximum(lower - values, 0.0)
    overshoot_mask = overshoot > 0.0
    undershoot_mask = undershoot > 0.0
    clipped_mask = overshoot_mask | undershoot_mask
    return overshoot, undershoot, overshoot_mask, undershoot_mask, clipped_mask


def _wall_delta_numpy(
    mesh: ImportedTetraMesh,
    config: GmshTetraThermalConfig,
    wall: _ResolvedWallBoundary,
    temperature: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    delta = np.zeros_like(temperature, dtype=np.float64)
    if wall.faces.size == 0:
        return delta, 0.0

    volumes = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-16)
    if config.wall_boundary_mode == "fixed_heat_flux":
        face_power = float(config.wall_heat_flux) * wall.areas
    elif config.wall_boundary_mode == "fixed_temperature":
        face_delta = np.asarray(
            mesh.face_centers[wall.faces], dtype=np.float64
        ) - np.asarray(mesh.cell_centers[wall.owners], dtype=np.float64)
        normals = np.asarray(mesh.face_normals[wall.faces], dtype=np.float64)
        distance = np.maximum(np.abs(np.einsum("ij,ij->i", face_delta, normals)), 1e-12)
        coefficient = wall.areas / distance
        conductivity = (
            float(config.thermal_diffusivity) * float(config.rho) * float(config.cp)
        )
        face_power = (
            conductivity
            * coefficient
            * (float(config.wall_temperature) - temperature[wall.owners])
        )
    else:
        return delta, 0.0

    cell_power = np.zeros_like(temperature, dtype=np.float64)
    np.add.at(cell_power, wall.owners, face_power)
    delta = float(dt) * cell_power / (float(config.rho) * float(config.cp) * volumes)
    return delta, float(dt) * float(np.sum(face_power))


def _build_wall_heat_transfer(
    config: GmshTetraThermalConfig,
    wall: _ResolvedWallBoundary,
    *,
    final_requested_energy: float,
    final_applied_energy: float,
    cumulative_requested_energy: float,
    cumulative_energy_after_limiter_before_clipping: float,
    cumulative_applied_energy: float,
    elapsed_time: float,
) -> dict[str, object]:
    inverse_dt = 1.0 / elapsed_time if elapsed_time > 0.0 else 0.0
    return {
        "mode": str(config.wall_boundary_mode),
        "boundary_mode": str(config.wall_boundary_mode),
        "groups": [str(name) for name in config.heated_wall_physical_groups],
        "physical_groups": [str(name) for name in config.heated_wall_physical_groups],
        "resolved_group_tags": dict(wall.group_tags),
        "resolved_group_face_counts": dict(wall.group_face_counts),
        "face_count": int(wall.faces.size),
        "area_m2": float(wall.area),
        "area": float(wall.area),
        "instantaneous_requested_power_w": float(final_requested_energy)
        * inverse_dt
        * max(int(config.steps), 1),
        "instantaneous_power_w": float(final_applied_energy)
        * inverse_dt
        * max(int(config.steps), 1),
        "average_requested_power_w": float(cumulative_requested_energy) * inverse_dt,
        "average_power_w": float(cumulative_applied_energy) * inverse_dt,
        "instantaneous_applied_power_w": float(final_applied_energy)
        * inverse_dt
        * max(int(config.steps), 1),
        "average_applied_power_w": float(cumulative_applied_energy) * inverse_dt,
        "cumulative_requested_energy_j": float(cumulative_requested_energy),
        "cumulative_energy_after_limiter_before_clipping_j": float(
            cumulative_energy_after_limiter_before_clipping
        ),
        "cumulative_applied_energy_j": float(cumulative_applied_energy),
        "cumulative_requested_wall_energy_j": float(cumulative_requested_energy),
        "cumulative_applied_wall_energy_j": float(cumulative_applied_energy),
        "limiter_clipping_correction_energy_j": float(
            cumulative_requested_energy - cumulative_applied_energy
        ),
        "limiter_clipping_correction_j": float(
            cumulative_requested_energy - cumulative_applied_energy
        ),
        "limiter_correction_energy_j": float(
            cumulative_requested_energy
            - cumulative_energy_after_limiter_before_clipping
        ),
        "clipping_correction_energy_j": float(
            cumulative_energy_after_limiter_before_clipping - cumulative_applied_energy
        ),
    }


def _build_global_energy_balance(
    config: GmshTetraThermalConfig,
    *,
    initial_integral_t_dv: float,
    final_integral_t_dv: float,
    cumulative_advective_in: float,
    cumulative_advective_out: float,
    cumulative_advective_integral_change: float,
    cumulative_diffusion_integral_change: float,
    cumulative_source_integral_change: float,
    cumulative_clamp_integral_change: float,
    cumulative_wall_operator_energy_j: float,
    cumulative_wall_applied_energy_j: float,
) -> dict[str, float]:
    rho_cp = float(config.rho) * float(config.cp)
    delta_energy_j = rho_cp * (
        float(final_integral_t_dv) - float(initial_integral_t_dv)
    )
    advective_in_energy_j = rho_cp * float(cumulative_advective_in)
    advective_out_energy_j = rho_cp * float(cumulative_advective_out)
    advective_net_energy_j = rho_cp * float(cumulative_advective_integral_change)
    diffusion_energy_j = rho_cp * float(cumulative_diffusion_integral_change)
    wall_boundary_energy_j = float(cumulative_wall_operator_energy_j)
    non_wall_diffusion_energy_j = diffusion_energy_j
    if config.wall_boundary_mode == "fixed_temperature":
        non_wall_diffusion_energy_j -= wall_boundary_energy_j
    volume_energy_j = rho_cp * float(cumulative_source_integral_change)
    clipping_energy_j = rho_cp * float(cumulative_clamp_integral_change)
    accounted_energy_j = (
        advective_net_energy_j
        + non_wall_diffusion_energy_j
        + wall_boundary_energy_j
        + volume_energy_j
        + clipping_energy_j
    )
    return {
        "delta_energy_j": float(delta_energy_j),
        "advective_in_energy_j": float(advective_in_energy_j),
        "advective_out_energy_j": float(advective_out_energy_j),
        "advective_net_energy_j": float(advective_net_energy_j),
        "diffusion_energy_j": float(diffusion_energy_j),
        "non_wall_diffusion_energy_j": float(non_wall_diffusion_energy_j),
        "wall_boundary_energy_j": float(wall_boundary_energy_j),
        "volume_energy_j": float(volume_energy_j),
        "clipping_energy_j": float(clipping_energy_j),
        "accounted_energy_j": float(accounted_energy_j),
        # Backward-compatible diagnostic: wall energy after limiter and clipping.
        "wall_energy_j": float(cumulative_wall_applied_energy_j),
        "balance_error_j": float(delta_energy_j - accounted_energy_j),
    }


def _validate_thermal_config(config: GmshTetraThermalConfig) -> None:
    if config.steps <= 0:
        raise ValueError("steps must be positive.")
    if config.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if config.dt_mode not in {"manual", "auto"}:
        raise ValueError("dt_mode must be 'manual' or 'auto'.")
    if config.cfl_target <= 0.0:
        raise ValueError("cfl_target must be positive.")
    if config.cfl_limit <= 0.0:
        raise ValueError("cfl_limit must be positive.")
    if config.diffusion_stability_factor <= 0.0:
        raise ValueError("diffusion_stability_factor must be positive.")
    if config.thermal_diffusivity < 0.0:
        raise ValueError("thermal_diffusivity must be non-negative.")
    if config.rho <= 0.0:
        raise ValueError("rho must be positive.")
    if config.cp <= 0.0:
        raise ValueError("cp must be positive.")
    if config.wall_boundary_mode not in {
        "adiabatic",
        "fixed_temperature",
        "fixed_heat_flux",
    }:
        raise ValueError("Unsupported wall_boundary_mode.")
    if config.wall_temperature is not None and config.wall_heat_flux is not None:
        raise ValueError("wall_temperature and wall_heat_flux cannot both be set.")
    if config.wall_boundary_mode == "fixed_temperature":
        if not config.heated_wall_physical_groups:
            raise ValueError(
                "heated_wall_physical_groups is required for fixed_temperature."
            )
        if config.wall_temperature is None:
            raise ValueError("wall_temperature is required for fixed_temperature.")
    elif config.wall_boundary_mode == "fixed_heat_flux":
        if not config.heated_wall_physical_groups:
            raise ValueError(
                "heated_wall_physical_groups is required for fixed_heat_flux."
            )
        if config.wall_heat_flux is None:
            raise ValueError("wall_heat_flux is required for fixed_heat_flux.")
    if config.limiter_scheme not in {"upwind", "bounded_upwind"}:
        raise ValueError("Unsupported limiter_scheme.")
    if config.execution_profile not in {"production", "debug", "reference"}:
        raise ValueError("Unsupported execution_profile.")
    if config.diagnostics_mode not in {"auto", "debug", "fast"}:
        raise ValueError("Unsupported diagnostics_mode.")
    if config.torch_compile_step_body not in {"auto", "on", "off"}:
        raise ValueError("Unsupported torch_compile_step_body.")
    if config.history_mode not in {"auto", "dict", "compact", "off"}:
        raise ValueError("Unsupported history_mode.")
    if config.history_stride <= 0:
        raise ValueError("history_stride must be positive.")
    if config.diagnostics_stride < 0:
        raise ValueError("diagnostics_stride must be non-negative.")
    min_t = config.min_temperature
    max_t = config.max_temperature
    if min_t is not None and max_t is not None and float(min_t) >= float(max_t):
        raise ValueError("min_temperature must be smaller than max_temperature.")


def _resolve_wall_boundary(
    mesh: ImportedTetraMesh,
    config: GmshTetraThermalConfig,
) -> _ResolvedWallBoundary:
    if config.wall_boundary_mode == "adiabatic":
        return _ResolvedWallBoundary(
            faces=np.zeros((0,), dtype=np.int64),
            owners=np.zeros((0,), dtype=np.int64),
            areas=np.zeros((0,), dtype=np.float64),
            group_face_counts={},
            group_tags={},
        )

    boundary_tags = np.asarray(mesh.boundary_tag_per_face, dtype=np.int32)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    wall_face_set = set(int(face_idx) for face_idx in wall_faces.tolist())
    name_to_tag = {
        str(name): int(tag) for tag, name in mesh.boundary_face_names.items()
    }
    resolved_faces: list[np.ndarray] = []
    group_face_counts: dict[str, int] = {}
    group_tags: dict[str, int] = {}

    for requested_name in config.heated_wall_physical_groups:
        name = str(requested_name)
        if name not in mesh.physical_groups:
            raise ValueError(f"Unknown heated wall physical group: {name!r}.")
        tag = name_to_tag.get(name)
        if tag is None:
            raise ValueError(
                f"Heated physical group {name!r} contains no boundary faces."
            )
        faces = np.flatnonzero(boundary_tags == int(tag)).astype(np.int64)
        if faces.size == 0:
            raise ValueError(
                f"Heated physical group {name!r} contains no boundary faces."
            )
        non_wall = [
            int(face_idx)
            for face_idx in faces.tolist()
            if int(face_idx) not in wall_face_set
        ]
        if non_wall:
            raise ValueError(
                f"Heated physical group {name!r} is not a subset of mesh.wall_faces."
            )
        resolved_faces.append(faces)
        group_face_counts[name] = int(faces.size)
        group_tags[name] = int(tag)

    faces = np.unique(np.concatenate(resolved_faces)).astype(np.int64)
    return _ResolvedWallBoundary(
        faces=faces,
        owners=np.asarray(mesh.face_to_cells[faces, 0], dtype=np.int64),
        areas=np.asarray(mesh.face_areas[faces], dtype=np.float64),
        group_face_counts=group_face_counts,
        group_tags=group_tags,
    )


def _should_record_thermal_history_step(
    config: GmshTetraThermalConfig,
    step: int,
    *,
    snapshot_steps: set[int],
) -> bool:
    if not config.collect_history:
        return False
    if step == 1 or step == config.steps or step in snapshot_steps:
        return True
    return step % config.history_stride == 0


def _should_log_thermal_progress_step(
    config: GmshTetraThermalConfig,
    step: int,
) -> bool:
    return config.progress_every > 0 and step % config.progress_every == 0


def _should_collect_thermal_diagnostics_step(
    config: GmshTetraThermalConfig,
    step: int,
    *,
    snapshot_steps: set[int],
) -> bool:
    if step == 1 or step == config.steps or step in snapshot_steps:
        return True
    return config.diagnostics_stride > 0 and step % config.diagnostics_stride == 0


def _thermal_execution_mode(config: GmshTetraThermalConfig) -> str:
    if _resolve_thermal_diagnostics_mode(config) == "fast":
        return "production"
    if (
        config.collect_history
        or config.diagnostics_stride > 0
        or config.progress_every > 0
        or config.full_step_diagnostics
        or config.debug_artifacts
    ):
        return "debug"
    return "production"


def _resolve_thermal_diagnostics_mode(config: GmshTetraThermalConfig) -> str:
    if config.diagnostics_mode != "auto":
        return str(config.diagnostics_mode)
    if config.execution_profile == "production":
        return "fast"
    return "debug"


def _resolve_thermal_torch_compile_mode(
    config: GmshTetraThermalConfig,
    *,
    actual_backend: str | None = None,
    device_type: str | None = None,
) -> str:
    if config.torch_compile_step_body != "auto":
        return str(config.torch_compile_step_body)
    backend_name = str(actual_backend or config.backend)
    resolved_device = str(device_type or config.torch_device).split(":", 1)[0].lower()
    if (
        backend_name == "torch"
        and resolved_device == "cuda"
        and _resolve_thermal_diagnostics_mode(config) == "fast"
        and not config.debug_artifacts
    ):
        if _torch_inductor_cuda_compile_unavailable_reason(device_type=resolved_device):
            return "off"
        return "on"
    return "off"


def _torch_inductor_cuda_compile_unavailable_reason(
    *,
    device_type: str | None,
) -> str:
    resolved_device = str(device_type or "").split(":", 1)[0].lower()
    if resolved_device != "cuda":
        return ""
    try:
        triton_spec = find_spec("triton")
    except (ImportError, ValueError):
        triton_spec = None
    if triton_spec is None:
        return "torch.compile CUDA inductor unavailable: triton is not importable"
    return ""


def _thermal_torch_compile_disabled_reason(
    config: GmshTetraThermalConfig,
    *,
    resolved_compile_mode: str,
    actual_backend: str,
    device_type: str | None = None,
) -> str:
    if resolved_compile_mode != "off":
        return "disabled"
    if config.torch_compile_step_body == "off":
        return "disabled"
    backend_name = str(actual_backend)
    resolved_device = str(device_type or config.torch_device).split(":", 1)[0].lower()
    if backend_name != "torch":
        return "torch backend not active"
    if _resolve_thermal_diagnostics_mode(config) != "fast":
        return "auto disabled outside fast diagnostics mode"
    if config.debug_artifacts:
        return "auto disabled while debug artifacts are enabled"
    unavailable_reason = _torch_inductor_cuda_compile_unavailable_reason(
        device_type=resolved_device
    )
    if unavailable_reason:
        return unavailable_reason
    if resolved_device != "cuda":
        return "auto disabled outside CUDA torch device"
    return "disabled"


def _resolve_thermal_history_mode(config: GmshTetraThermalConfig) -> str:
    if not config.collect_history:
        return "off"
    if config.history_mode != "auto":
        return str(config.history_mode)
    if config.execution_profile == "production":
        return "compact"
    return "dict"


def _resolve_thermal_execution_policy(
    config: GmshTetraThermalConfig,
    backend_selection: BackendSelection,
) -> dict[str, object]:
    execution_profile = str(config.execution_profile)
    requested_backend = str(config.backend)
    selected_backend = str(backend_selection.selected_backend)
    production_backend = "torch"
    production_backend_satisfied = bool(selected_backend == production_backend)
    degraded_execution_mode = bool(
        execution_profile == "production" and not production_backend_satisfied
    )
    reason = "backend policy satisfied"
    if execution_profile == "production":
        if production_backend_satisfied:
            reason = "production profile running on torch backend"
        elif config.allow_numpy_production_fallback:
            reason = (
                "production profile fell back to numpy due to explicit override; "
                "treat results as degraded execution mode"
            )
        else:
            reason = (
                "production profile requires torch backend; selected backend is numpy. "
                "Use execution_profile=debug/reference or "
                "allow_numpy_production_fallback=True to override."
            )
            raise RuntimeError(reason)
    elif execution_profile == "debug":
        reason = f"debug profile allows backend {selected_backend}"
    else:
        reason = f"reference profile allows backend {selected_backend}"

    return {
        "execution_profile": execution_profile,
        "requested_backend": requested_backend,
        "selected_backend": selected_backend,
        "production_backend": production_backend,
        "production_backend_satisfied": bool(production_backend_satisfied),
        "allow_numpy_production_fallback": bool(config.allow_numpy_production_fallback),
        "degraded_execution_mode": bool(degraded_execution_mode),
        "used_numpy_fallback": bool(backend_selection.used_numpy_fallback),
        "reason": str(reason),
    }


def _build_compact_thermal_history_entry(
    *,
    step: int,
    used_dt: float,
    temperature_min: float,
    temperature_max: float,
    temperature_mean: float,
    max_change: float,
    advective_integral_change: float,
    diffusion_integral_change: float,
    source_integral_change: float,
    non_advective_limiter_integral_change: float,
    clamp_integral_change: float,
    raw_balance_error: float,
    energy_proxy: float,
    cfl_max: float,
    cfl_incoming_max: float,
    cfl_warning: bool,
    clipped_cell_count: int,
    limiter_active_cell_count: int,
    non_advective_limiter_active_cell_count: int,
) -> ThermalStepHistoryEntry:
    return {
        "step": int(step),
        "dt_used": float(used_dt),
        "temperature_min": float(temperature_min),
        "temperature_max": float(temperature_max),
        "temperature_mean": float(temperature_mean),
        "max_change": float(max_change),
        "advective_integral_change": float(advective_integral_change),
        "diffusion_integral_change": float(diffusion_integral_change),
        "source_integral_change": float(source_integral_change),
        "non_advective_limiter_integral_change": float(
            non_advective_limiter_integral_change
        ),
        "clamp_integral_change": float(clamp_integral_change),
        "raw_balance_error": float(raw_balance_error),
        "energy_proxy": float(energy_proxy),
        "cfl_max": float(cfl_max),
        "cfl_incoming_max": float(cfl_incoming_max),
        "cfl_warning": bool(cfl_warning),
        "clipped_cell_count": int(clipped_cell_count),
        "limiter_active_cell_count": int(limiter_active_cell_count),
        "non_advective_limiter_active_cell_count": int(
            non_advective_limiter_active_cell_count
        ),
        "advection_flux_assemblies": 1,
    }


def _build_fast_thermal_history_entry(
    *,
    step: int,
    used_dt: float,
    temperature_min: float,
    temperature_max: float,
    cfl_max: float,
    cfl_warning: bool,
) -> ThermalStepHistoryEntry:
    return {
        "step": int(step),
        "dt_used": float(used_dt),
        "temperature_min": float(temperature_min),
        "temperature_max": float(temperature_max),
        "cfl_max": float(cfl_max),
        "cfl_warning": bool(cfl_warning),
        "advection_flux_assemblies": 1,
    }


def _boundary_volume_flux_metrics(
    precompute: ScalarBackendPrecompute,
) -> tuple[float, float]:
    face_flux = np.asarray(
        precompute.face_normal_velocity * precompute.face_areas, dtype=np.float64
    )
    inlet_faces = np.concatenate((precompute.left_faces, precompute.right_faces))
    inlet_vol_flux_in = (
        float(-np.sum(np.minimum(face_flux[inlet_faces], 0.0)))
        if inlet_faces.size
        else 0.0
    )
    outlet_vol_flux_out = (
        float(np.sum(np.maximum(face_flux[precompute.outlet_faces], 0.0)))
        if precompute.outlet_faces.size
        else 0.0
    )
    return inlet_vol_flux_in, outlet_vol_flux_out


def _materialize_torch_scalar_batch(
    torch_module: object,
    scalar_metrics: dict[str, object],
    *,
    device: object,
) -> dict[str, float]:
    if not scalar_metrics:
        return {}

    names = list(scalar_metrics)
    torch_api = torch_module
    packed = []
    for name in names:
        value = scalar_metrics[name]
        if hasattr(value, "detach"):
            tensor = value.to(dtype=torch_api.float64).reshape(())
        else:
            tensor = torch_api.tensor(
                float(value), dtype=torch_api.float64, device=device
            )
        packed.append(tensor)
    values = torch_api.stack(packed).detach().cpu().numpy()
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def _thermal_torch_compile_metadata(
    *,
    compile_mode: str,
    reason: str,
    used: bool = False,
) -> dict[str, object]:
    return {
        "torch_compile_step_requested": bool(compile_mode == "on"),
        "torch_compile_step_used": bool(used),
        "torch_compile_step_reason": str(reason),
    }


def _maybe_compile_thermal_torch_step_body(
    torch_module: object,
    step_fn: object,
    *,
    compile_mode: str,
    device_type: str | None = None,
    disabled_reason: str = "disabled",
) -> tuple[object, dict[str, object]]:
    metadata = _thermal_torch_compile_metadata(
        compile_mode=compile_mode,
        reason=disabled_reason,
    )
    if compile_mode != "on":
        return step_fn, metadata

    if not hasattr(torch_module, "compile"):
        metadata["torch_compile_step_reason"] = "torch.compile unavailable"
        return step_fn, metadata
    unavailable_reason = _torch_inductor_cuda_compile_unavailable_reason(
        device_type=device_type
    )
    if unavailable_reason:
        metadata["torch_compile_step_reason"] = unavailable_reason
        return step_fn, metadata

    try:
        compiled = torch_module.compile(
            step_fn,
            backend="inductor",
            fullgraph=False,
            dynamic=False,
        )
    except Exception as exc:
        metadata["torch_compile_step_reason"] = f"compile failed: {exc!s}"
        return step_fn, metadata

    metadata["torch_compile_step_used"] = True
    metadata["torch_compile_step_reason"] = "compiled with torch.compile"
    return compiled, metadata


def _thermal_torch_fast_step_body(
    temperature: object,
    *,
    precompute: ScalarBackendPrecompute,
    gradient_method: GradientMethod,
    laplacian_method: LaplacianMethod,
    limiter_scheme: ThermalLimiterScheme,
    normalize_bounded_advection: bool,
    limit_non_advective_update: bool,
    clipping_enabled: bool,
    boundedness_enabled: bool,
    min_temperature: float | None,
    max_temperature: float | None,
    inlet_vol_flux_in_t: object,
    outlet_vol_flux_out_t: object,
    left_inlet_temperature: float,
    right_inlet_temperature: float,
    used_dt: float,
    thermal_diffusivity: float,
    source_term: float,
    wall_boundary_mode: ThermalWallBoundaryMode,
    wall_temperature: float | None,
    wall_heat_flux: float | None,
    rho_cp: float,
    wall_owners_t: object,
    wall_areas_t: object,
    wall_coefficients_t: object,
) -> tuple[object, ...]:
    tb = precompute.torch_backend
    if tb is None:
        raise RuntimeError("Torch backend precompute is not available.")

    torch = tb["torch"]
    volumes_t = tb["vol"]
    zero_scalar = torch.zeros((), dtype=torch.float64, device=tb["device"])

    lower_bound = float(min_temperature) if boundedness_enabled else -1.0e12
    upper_bound = float(max_temperature) if boundedness_enabled else 1.0e12
    scale = 1.0
    offset = 0.0
    advection_state = temperature
    left_inlet_for_advection = float(left_inlet_temperature)
    right_inlet_for_advection = float(right_inlet_temperature)
    limiter_lower_bound = lower_bound
    limiter_upper_bound = upper_bound

    if (
        boundedness_enabled
        and normalize_bounded_advection
        and limiter_scheme == "bounded_upwind"
    ):
        span = upper_bound - lower_bound
        if span <= 0.0:
            raise ValueError("Invalid thermal bounds span for advection limiter.")
        offset = lower_bound
        scale = span
        advection_state = (temperature - offset) / scale
        left_inlet_for_advection = (float(left_inlet_temperature) - offset) / scale
        right_inlet_for_advection = (float(right_inlet_temperature) - offset) / scale
        limiter_lower_bound = 0.0
        limiter_upper_bound = 1.0

    assembled = assemble_advection_fluxes_torch(
        precompute,
        advection_state,
        left_inlet_value=left_inlet_for_advection,
        right_inlet_value=right_inlet_for_advection,
        dt=used_dt,
    )
    limited = apply_bounded_limiter_torch(
        precompute,
        advection_state,
        assembled,
        scheme=limiter_scheme if boundedness_enabled else "upwind",
        lower_bound=limiter_lower_bound,
        upper_bound=limiter_upper_bound,
        collect_diagnostics=False,
    )

    limited_state_for_limiter = limited["limited_state_before_clip"]
    advected = offset + scale * limited_state_for_limiter
    old_integral = torch.sum(temperature * volumes_t)
    advected_integral = torch.sum(advected * volumes_t)
    advective_integral_change = advected_integral - old_integral

    lap = torch.zeros_like(advected)
    if thermal_diffusivity > 0.0:
        lap = laplacian_torch(
            precompute,
            advected,
            gradient_method=gradient_method,
            laplacian_method=laplacian_method,
        )
    diff_delta_unlimited = used_dt * float(thermal_diffusivity) * lap
    source_delta_unlimited = torch.full_like(advected, used_dt * source_term)
    wall_delta_unlimited = torch.zeros_like(advected)
    wall_requested_energy = zero_scalar
    if wall_boundary_mode != "adiabatic" and int(wall_owners_t.numel()) > 0:
        if wall_boundary_mode == "fixed_heat_flux":
            wall_face_power = float(wall_heat_flux) * wall_areas_t
        else:
            conductivity = float(thermal_diffusivity) * float(rho_cp)
            wall_face_power = (
                conductivity
                * wall_coefficients_t
                * (float(wall_temperature) - advected[wall_owners_t])
            )
        wall_cell_power = torch.zeros_like(advected)
        wall_cell_power.scatter_add_(0, wall_owners_t, wall_face_power)
        wall_delta_unlimited = (
            float(used_dt) * wall_cell_power / (float(rho_cp) * volumes_t)
        )
        wall_requested_energy = float(used_dt) * torch.sum(wall_face_power)
    non_advective_delta_unlimited = diff_delta_unlimited + source_delta_unlimited
    if wall_boundary_mode == "fixed_heat_flux":
        non_advective_delta_unlimited = (
            non_advective_delta_unlimited + wall_delta_unlimited
        )
    pre_non_advective_update = advected + non_advective_delta_unlimited

    non_advective_limiter_scale = torch.ones_like(advected)
    non_advective_limiter_active_mask = torch.zeros_like(advected, dtype=torch.bool)
    if boundedness_enabled and limit_non_advective_update:
        positive = non_advective_delta_unlimited > 0.0
        negative = non_advective_delta_unlimited < 0.0
        allowed_pos = torch.clamp(upper_bound - advected, min=0.0)
        allowed_neg = torch.clamp(lower_bound - advected, max=0.0)
        positive_scale = allowed_pos / torch.clamp(
            non_advective_delta_unlimited, min=1e-16
        )
        negative_scale = allowed_neg / torch.clamp(
            non_advective_delta_unlimited, max=-1e-16
        )
        candidate_scale = torch.ones_like(non_advective_limiter_scale)
        candidate_scale = torch.where(positive, positive_scale, candidate_scale)
        candidate_scale = torch.where(negative, negative_scale, candidate_scale)
        non_advective_limiter_scale = torch.minimum(
            non_advective_limiter_scale,
            candidate_scale,
        )
        non_advective_limiter_scale = torch.clamp(
            non_advective_limiter_scale, min=0.0, max=1.0
        )
        non_advective_limiter_active_mask = non_advective_limiter_scale < (1.0 - 1e-12)

    diff_delta = diff_delta_unlimited * non_advective_limiter_scale
    source_delta = source_delta_unlimited * non_advective_limiter_scale
    wall_delta = wall_delta_unlimited * non_advective_limiter_scale
    wall_operator_energy = float(rho_cp) * torch.sum(wall_delta * volumes_t)
    updated_raw = advected + diff_delta + source_delta
    if wall_boundary_mode == "fixed_heat_flux":
        updated_raw = updated_raw + wall_delta

    overshoot_mask = torch.zeros_like(updated_raw, dtype=torch.bool)
    undershoot_mask = torch.zeros_like(updated_raw, dtype=torch.bool)
    clipped_mask = torch.zeros_like(updated_raw, dtype=torch.bool)
    overshoot = torch.zeros_like(updated_raw)
    undershoot = torch.zeros_like(updated_raw)
    raw_overshoot_physical = torch.zeros_like(updated_raw)
    raw_undershoot_physical = torch.zeros_like(updated_raw)
    pre_non_adv_overshoot = torch.zeros_like(updated_raw)
    pre_non_adv_undershoot = torch.zeros_like(updated_raw)

    if boundedness_enabled:
        pre_non_adv_overshoot = torch.clamp(
            pre_non_advective_update - upper_bound, min=0.0
        )
        pre_non_adv_undershoot = torch.clamp(
            lower_bound - pre_non_advective_update, min=0.0
        )
        overshoot_mask = updated_raw > (upper_bound + 1e-12)
        undershoot_mask = updated_raw < (lower_bound - 1e-12)
        clipped_mask = overshoot_mask | undershoot_mask
        overshoot = torch.clamp(updated_raw - upper_bound, min=0.0)
        undershoot = torch.clamp(lower_bound - updated_raw, min=0.0)
        raw_state_physical = advected
        raw_overshoot_physical = torch.clamp(raw_state_physical - upper_bound, min=0.0)
        raw_undershoot_physical = torch.clamp(lower_bound - raw_state_physical, min=0.0)

    if boundedness_enabled and clipping_enabled:
        updated = torch.clamp(updated_raw, min=lower_bound, max=upper_bound)
        updated_without_wall = torch.clamp(
            updated_raw - wall_delta, min=lower_bound, max=upper_bound
        )
    else:
        updated = updated_raw
        updated_without_wall = updated_raw - wall_delta
    wall_applied_energy = float(rho_cp) * torch.sum(
        (updated - updated_without_wall) * volumes_t
    )

    clamp_delta = updated - updated_raw
    diffusion_integral_change = torch.sum(diff_delta * volumes_t)
    source_integral_change = torch.sum(source_delta * volumes_t)
    diffusion_integral_change_unlimited = torch.sum(diff_delta_unlimited * volumes_t)
    source_integral_change_unlimited = torch.sum(source_delta_unlimited * volumes_t)
    applied_non_advective_delta = diff_delta + source_delta
    if wall_boundary_mode == "fixed_heat_flux":
        applied_non_advective_delta = applied_non_advective_delta + wall_delta
    non_advective_limiter_delta = (
        non_advective_delta_unlimited - applied_non_advective_delta
    )
    non_advective_limiter_integral_change = torch.sum(
        non_advective_limiter_delta * volumes_t
    )

    limiter_scale_out = limited["limiter_scale_out"]
    limiter_scale_in = limited["limiter_scale_in"]
    advection_limiter_scale = torch.minimum(limiter_scale_out, limiter_scale_in)
    advection_limiter_active_mask = advection_limiter_scale < (1.0 - 1e-12)

    raw_inlet_scalar_flux_in = (
        float(offset) * inlet_vol_flux_in_t
        + float(scale) * assembled["inlet_scalar_flux_in"]
    )
    raw_outlet_scalar_flux_out = (
        float(offset) * outlet_vol_flux_out_t
        + float(scale) * assembled["outlet_scalar_flux_out"]
    )
    adv_mass_in = used_dt * raw_inlet_scalar_flux_in
    adv_mass_out = used_dt * raw_outlet_scalar_flux_out

    return (
        updated,
        assembled["out_vol_rate_raw"],
        assembled["in_vol_rate_raw"],
        adv_mass_in,
        adv_mass_out,
        advective_integral_change,
        diffusion_integral_change,
        source_integral_change,
        diffusion_integral_change_unlimited,
        source_integral_change_unlimited,
        torch.sum(clamp_delta * volumes_t),
        non_advective_limiter_integral_change,
        wall_requested_energy,
        wall_operator_energy,
        wall_applied_energy,
        clipped_mask,
        overshoot_mask,
        undershoot_mask,
        advection_limiter_active_mask,
        non_advective_limiter_active_mask,
        non_advective_limiter_scale,
        limiter_scale_out,
        limiter_scale_in,
        torch.max(overshoot) if boundedness_enabled else zero_scalar,
        torch.max(undershoot) if boundedness_enabled else zero_scalar,
        torch.max(raw_overshoot_physical) if boundedness_enabled else zero_scalar,
        torch.max(raw_undershoot_physical) if boundedness_enabled else zero_scalar,
        torch.max(pre_non_adv_overshoot) if boundedness_enabled else zero_scalar,
        torch.max(pre_non_adv_undershoot) if boundedness_enabled else zero_scalar,
    )


def _empty_compact_thermal_history() -> dict[str, list[float | int]]:
    return {
        "step": [],
        "dt_used": [],
        "temperature_min": [],
        "temperature_max": [],
        "cfl_max": [],
    }


def _append_compact_thermal_history_sample(
    compact_history: dict[str, list[float | int]],
    *,
    step: int,
    dt_used: float,
    temperature_min: float,
    temperature_max: float,
    cfl_max: float,
) -> None:
    compact_history["step"].append(int(step))
    compact_history["dt_used"].append(float(dt_used))
    compact_history["temperature_min"].append(float(temperature_min))
    compact_history["temperature_max"].append(float(temperature_max))
    compact_history["cfl_max"].append(float(cfl_max))


def _run_tetra_thermal_debug_numpy(
    mesh: ImportedTetraMesh,
    config: GmshTetraThermalConfig,
    *,
    precompute: ScalarBackendPrecompute | None = None,
    backend_selection: BackendSelection | None = None,
    face_normal_velocity: np.ndarray,
    flux_diagnostics: Dict[str, float] | None = None,
    velocity_metadata: Dict[str, object] | None = None,
    execution_policy: Dict[str, object] | None = None,
    wall_boundary: _ResolvedWallBoundary | None = None,
) -> Dict[str, object]:
    """Run thermal transport on tetra mesh using finite-volume advection+diffusion."""

    _validate_thermal_config(config)
    wall_boundary = wall_boundary or _resolve_wall_boundary(mesh, config)

    min_t = config.min_temperature
    max_t = config.max_temperature
    boundedness_enabled = min_t is not None and max_t is not None
    if precompute is None:
        precompute = build_scalar_backend_precompute(
            mesh,
            face_normal_velocity,
            backend="numpy",
            diffusion_dirichlet_left_value=float(config.left_inlet_temperature),
            diffusion_dirichlet_right_value=float(config.right_inlet_temperature),
        )
    boundary_face_groups = precompute.boundary_face_groups
    left_faces = precompute.left_faces
    right_faces = precompute.right_faces
    left_cells = precompute.left_cells
    right_cells = precompute.right_cells
    outlet_faces = precompute.outlet_faces
    wall_faces = precompute.wall_faces
    diffusion_stencil = precompute.diffusion_stencil
    if diffusion_stencil is None:
        raise ValueError("Thermal diffusion precompute is required.")
    inlet_vol_flux_in, outlet_vol_flux_out = _boundary_volume_flux_metrics(precompute)

    advection_dt_limit = estimate_stable_dt(precompute)
    diffusion_dt_limit = estimate_diffusion_dt_limit(
        diffusion_stencil, float(config.thermal_diffusivity)
    )

    requested_dt = float(config.dt)
    used_dt = requested_dt
    if config.dt_mode == "auto":
        dt_candidates: list[float] = []
        if np.isfinite(advection_dt_limit) and advection_dt_limit > 0.0:
            dt_candidates.append(float(config.cfl_target) * float(advection_dt_limit))
        if np.isfinite(diffusion_dt_limit) and diffusion_dt_limit > 0.0:
            dt_candidates.append(
                float(config.diffusion_stability_factor) * float(diffusion_dt_limit)
            )
        if dt_candidates:
            used_dt = float(min(dt_candidates))

    if used_dt <= 0.0:
        raise FloatingPointError("Resolved dt must be positive.")

    cfl_values_initial, cfl_max_initial = compute_cfl_metrics(precompute, used_dt)
    cfl_warning = bool(cfl_max_initial > float(config.cfl_limit))
    diffusion_stability_warning = bool(
        np.isfinite(diffusion_dt_limit)
        and diffusion_dt_limit > 0.0
        and used_dt > diffusion_dt_limit
    )

    temperature = np.full(
        mesh.tetrahedra.shape[0],
        float(config.initial_temperature),
        dtype=np.float64,
    )

    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    source_term = float(config.heat_source) / (float(config.rho) * float(config.cp))
    initial_energy = float(np.sum(np.asarray(temperature, dtype=np.float64) * volumes))
    resolved_diagnostics_mode = _resolve_thermal_diagnostics_mode(config)
    allow_expensive_step_diagnostics = resolved_diagnostics_mode == "debug"
    resolved_history_mode = _resolve_thermal_history_mode(config)
    resolved_torch_compile_mode = _resolve_thermal_torch_compile_mode(
        config,
        actual_backend="numpy",
    )

    history: list[ThermalStepHistoryEntry] = []
    compact_history = (
        _empty_compact_thermal_history() if resolved_history_mode == "compact" else {}
    )
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_steps = {int(s) for s in config.snapshot_steps if int(s) > 0}

    clipping_used = False
    max_overshoot_before_clip = 0.0
    max_undershoot_before_clip = 0.0
    max_overshoot_before_limiter = 0.0
    max_undershoot_before_limiter = 0.0
    cumulative_adv_in = 0.0
    cumulative_adv_out = 0.0
    cumulative_advective_integral_change = 0.0
    cumulative_source = 0.0
    cumulative_diffusion = 0.0
    cumulative_source_unlimited = 0.0
    cumulative_diffusion_unlimited = 0.0
    cumulative_clamp = 0.0
    cumulative_non_advective_limiter = 0.0
    cumulative_wall_requested_energy = 0.0
    cumulative_wall_operator_energy = 0.0
    cumulative_wall_applied_energy = 0.0
    final_wall_requested_energy = 0.0
    final_wall_applied_energy = 0.0
    max_overshoot_before_non_advective_limiter = 0.0
    max_undershoot_before_non_advective_limiter = 0.0

    final_lap = np.zeros_like(temperature)
    final_raw_state = np.array(temperature, copy=True)
    final_limited_state = np.array(temperature, copy=True)
    final_updated_before_clip = np.array(temperature, copy=True)
    final_updated_before_non_advective_limiter = np.array(temperature, copy=True)
    final_clipped_mask = np.zeros_like(temperature, dtype=bool)
    final_overshoot_mask = np.zeros_like(temperature, dtype=bool)
    final_undershoot_mask = np.zeros_like(temperature, dtype=bool)
    final_advection_limiter_active_mask = np.zeros_like(temperature, dtype=bool)
    final_non_advective_limiter_active_mask = np.zeros_like(temperature, dtype=bool)
    final_non_advective_limiter_scale = np.ones_like(temperature)
    final_advection_limiter_scale = np.ones_like(temperature)
    final_advection_delta = np.zeros_like(temperature)
    final_diffusion_delta = np.zeros_like(temperature)
    final_source_delta = np.zeros_like(temperature)
    final_clamp_delta = np.zeros_like(temperature)
    final_face_flux_raw = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_face_flux_limited = np.zeros(mesh.face_vertices.shape[0], dtype=np.float64)
    final_in_mass_raw = np.zeros_like(temperature)
    final_out_mass_raw = np.zeros_like(temperature)
    final_in_mass_lim = np.zeros_like(temperature)
    final_out_mass_lim = np.zeros_like(temperature)
    final_limiter_out = np.ones_like(temperature)
    final_limiter_in = np.ones_like(temperature)

    for step in range(1, config.steps + 1):
        old = np.asarray(temperature, dtype=np.float64)
        old_integral = float(np.sum(old * volumes))

        lower_bound = float(min_t) if boundedness_enabled else float(np.min(old) - 1e9)
        upper_bound = float(max_t) if boundedness_enabled else float(np.max(old) + 1e9)
        scale = 1.0
        offset = 0.0
        advection_state = old
        left_inlet_for_advection = float(config.left_inlet_temperature)
        right_inlet_for_advection = float(config.right_inlet_temperature)
        limiter_lower_bound = lower_bound
        limiter_upper_bound = upper_bound

        if (
            boundedness_enabled
            and config.normalize_bounded_advection
            and config.limiter_scheme == "bounded_upwind"
        ):
            span = upper_bound - lower_bound
            if span <= 0.0:
                raise ValueError("Invalid thermal bounds span for advection limiter.")
            offset = lower_bound
            scale = span
            advection_state = (old - offset) / scale
            left_inlet_for_advection = (
                float(config.left_inlet_temperature) - offset
            ) / scale
            right_inlet_for_advection = (
                float(config.right_inlet_temperature) - offset
            ) / scale
            limiter_lower_bound = 0.0
            limiter_upper_bound = 1.0

        assembled_for_limiter = assemble_advection_fluxes_numpy(
            precompute,
            advection_state,
            left_inlet_value=left_inlet_for_advection,
            right_inlet_value=right_inlet_for_advection,
            dt=used_dt,
        )
        record_history = _should_record_thermal_history_step(
            config, step, snapshot_steps=snapshot_steps
        )
        record_diagnostics = allow_expensive_step_diagnostics and (
            _should_collect_thermal_diagnostics_step(
                config, step, snapshot_steps=snapshot_steps
            )
        )
        log_progress = _should_log_thermal_progress_step(config, step)
        capture_final_state = bool(config.debug_artifacts or step == config.steps)

        limited = apply_bounded_limiter_numpy(
            precompute,
            advection_state,
            assembled_for_limiter,
            scheme=config.limiter_scheme if boundedness_enabled else "upwind",
            lower_bound=limiter_lower_bound,
            upper_bound=limiter_upper_bound,
            collect_diagnostics=bool(
                allow_expensive_step_diagnostics or capture_final_state
            ),
        )

        limited_state_for_limiter = np.asarray(
            limited["limited_state_before_clip"], dtype=np.float64
        )
        raw_state_for_limiter = np.asarray(
            limited.get("raw_state_before_limiter", limited_state_for_limiter),
            dtype=np.float64,
        )
        raw_state_physical = offset + scale * raw_state_for_limiter
        limited_state_physical = offset + scale * limited_state_for_limiter
        advected = np.asarray(limited_state_physical, dtype=np.float64)
        advected_integral = float(np.sum(advected * volumes))
        advective_integral_change = float(advected_integral - old_integral)

        lap = np.zeros_like(advected)
        if config.thermal_diffusivity > 0.0:
            lap = laplacian_numpy(
                precompute,
                advected,
                gradient_method=config.gradient_method,
                laplacian_method=config.laplacian_method,
            )
        diff_delta_unlimited = used_dt * float(config.thermal_diffusivity) * lap
        source_delta_unlimited = np.full_like(advected, used_dt * source_term)
        wall_delta_unlimited, wall_requested_energy = _wall_delta_numpy(
            mesh, config, wall_boundary, advected, used_dt
        )
        non_advective_delta_unlimited = diff_delta_unlimited + source_delta_unlimited
        if config.wall_boundary_mode == "fixed_heat_flux":
            non_advective_delta_unlimited = (
                non_advective_delta_unlimited + wall_delta_unlimited
            )

        non_advective_limiter_scale = np.ones_like(advected)
        non_advective_limiter_active_mask = np.zeros_like(advected, dtype=bool)
        pre_non_advective_update = advected + non_advective_delta_unlimited

        if boundedness_enabled:
            (
                pre_non_adv_overshoot,
                pre_non_adv_undershoot,
                _,
                _,
                _,
            ) = _compute_bounds_violation(
                pre_non_advective_update, lower=lower_bound, upper=upper_bound
            )
            if allow_expensive_step_diagnostics or capture_final_state:
                max_overshoot_before_non_advective_limiter = max(
                    max_overshoot_before_non_advective_limiter,
                    float(np.max(pre_non_adv_overshoot))
                    if pre_non_adv_overshoot.size
                    else 0.0,
                )
                max_undershoot_before_non_advective_limiter = max(
                    max_undershoot_before_non_advective_limiter,
                    float(np.max(pre_non_adv_undershoot))
                    if pre_non_adv_undershoot.size
                    else 0.0,
                )

            if config.limit_non_advective_update:
                positive = non_advective_delta_unlimited > 0.0
                negative = non_advective_delta_unlimited < 0.0
                if np.any(positive):
                    allowed_pos = np.maximum(upper_bound - advected, 0.0)
                    non_advective_limiter_scale[positive] = np.minimum(
                        1.0,
                        allowed_pos[positive]
                        / np.maximum(non_advective_delta_unlimited[positive], 1e-16),
                    )
                if np.any(negative):
                    allowed_neg = np.minimum(lower_bound - advected, 0.0)
                    non_advective_limiter_scale[negative] = np.minimum(
                        1.0,
                        allowed_neg[negative]
                        / np.minimum(non_advective_delta_unlimited[negative], -1e-16),
                    )
                non_advective_limiter_scale = np.clip(
                    non_advective_limiter_scale, 0.0, 1.0
                )
                non_advective_limiter_active_mask = non_advective_limiter_scale < (
                    1.0 - 1e-12
                )

        diff_delta = diff_delta_unlimited * non_advective_limiter_scale
        source_delta = source_delta_unlimited * non_advective_limiter_scale
        wall_delta = wall_delta_unlimited * non_advective_limiter_scale
        wall_operator_energy = (
            float(config.rho) * float(config.cp) * float(np.sum(wall_delta * volumes))
        )
        updated_raw = advected + diff_delta + source_delta
        if config.wall_boundary_mode == "fixed_heat_flux":
            updated_raw = updated_raw + wall_delta

        overshoot = np.zeros_like(updated_raw)
        undershoot = np.zeros_like(updated_raw)
        overshoot_mask = np.zeros_like(updated_raw, dtype=bool)
        undershoot_mask = np.zeros_like(updated_raw, dtype=bool)
        clipped_mask = np.zeros_like(updated_raw, dtype=bool)

        if boundedness_enabled:
            overshoot_mask = updated_raw > (upper_bound + 1e-12)
            undershoot_mask = updated_raw < (lower_bound - 1e-12)
            clipped_mask = overshoot_mask | undershoot_mask
            if allow_expensive_step_diagnostics or capture_final_state:
                (
                    overshoot,
                    undershoot,
                    _,
                    _,
                    _,
                ) = _compute_bounds_violation(
                    updated_raw, lower=lower_bound, upper=upper_bound
                )
                max_overshoot_before_clip = max(
                    max_overshoot_before_clip, float(np.max(overshoot))
                )
                max_undershoot_before_clip = max(
                    max_undershoot_before_clip, float(np.max(undershoot))
                )
                raw_overshoot_physical = np.maximum(
                    raw_state_physical - upper_bound, 0.0
                )
                raw_undershoot_physical = np.maximum(
                    lower_bound - raw_state_physical, 0.0
                )
                max_overshoot_before_limiter = max(
                    max_overshoot_before_limiter,
                    float(np.max(raw_overshoot_physical))
                    if raw_overshoot_physical.size
                    else 0.0,
                )
                max_undershoot_before_limiter = max(
                    max_undershoot_before_limiter,
                    float(np.max(raw_undershoot_physical))
                    if raw_undershoot_physical.size
                    else 0.0,
                )

        if boundedness_enabled and config.clipping_enabled:
            updated = np.clip(updated_raw, lower_bound, upper_bound)
            updated_without_wall = np.clip(
                updated_raw - wall_delta, lower_bound, upper_bound
            )
            if np.any(clipped_mask):
                clipping_used = True
        else:
            updated = updated_raw
            updated_without_wall = updated_raw - wall_delta

        wall_applied_energy = (
            float(config.rho)
            * float(config.cp)
            * float(np.sum((updated - updated_without_wall) * volumes))
        )
        cumulative_wall_requested_energy += float(wall_requested_energy)
        cumulative_wall_operator_energy += float(wall_operator_energy)
        cumulative_wall_applied_energy += float(wall_applied_energy)
        final_wall_requested_energy = float(wall_requested_energy)
        final_wall_applied_energy = float(wall_applied_energy)

        if not np.all(np.isfinite(updated)):
            raise FloatingPointError(f"Temperature became non-finite at step {step}.")

        raw_inlet_scalar_flux_in = float(assembled_for_limiter["inlet_scalar_flux_in"])
        raw_outlet_scalar_flux_out = float(
            assembled_for_limiter["outlet_scalar_flux_out"]
        )
        physical_inlet_scalar_flux_in = (
            float(offset) * inlet_vol_flux_in + float(scale) * raw_inlet_scalar_flux_in
        )
        physical_outlet_scalar_flux_out = (
            float(offset) * outlet_vol_flux_out
            + float(scale) * raw_outlet_scalar_flux_out
        )
        adv_mass_in = used_dt * physical_inlet_scalar_flux_in
        adv_mass_out = used_dt * physical_outlet_scalar_flux_out
        adv_net = adv_mass_in - adv_mass_out
        diffusion_integral_change = float(np.sum(diff_delta * volumes))
        source_integral_change = float(np.sum(source_delta * volumes))
        diffusion_integral_change_unlimited = float(
            np.sum(diff_delta_unlimited * volumes)
        )
        source_integral_change_unlimited = float(
            np.sum(source_delta_unlimited * volumes)
        )

        clamp_delta = np.asarray(updated, dtype=np.float64) - updated_raw
        clamp_integral_change = float(np.sum(clamp_delta * volumes))

        applied_non_advective_delta = diff_delta + source_delta
        if config.wall_boundary_mode == "fixed_heat_flux":
            applied_non_advective_delta = applied_non_advective_delta + wall_delta
        non_advective_limiter_delta = (
            non_advective_delta_unlimited - applied_non_advective_delta
        )
        non_advective_limiter_integral_change = float(
            np.sum(non_advective_limiter_delta * volumes)
        )

        cumulative_adv_in += adv_mass_in
        cumulative_adv_out += adv_mass_out
        cumulative_advective_integral_change += advective_integral_change
        cumulative_source += source_integral_change
        cumulative_diffusion += diffusion_integral_change
        cumulative_source_unlimited += source_integral_change_unlimited
        cumulative_diffusion_unlimited += diffusion_integral_change_unlimited
        cumulative_clamp += clamp_integral_change
        cumulative_non_advective_limiter += non_advective_limiter_integral_change

        temperature = updated
        advection_limiter_scale = np.minimum(
            np.asarray(
                limited.get("limiter_scale_out", np.ones_like(temperature)),
                dtype=np.float64,
            ),
            np.asarray(
                limited.get("limiter_scale_in", np.ones_like(temperature)),
                dtype=np.float64,
            ),
        )
        advection_limiter_active_mask = advection_limiter_scale < (1.0 - 1e-12)
        advection_delta = advected - old
        limiter_conservation_error = float(scale) * float(
            limited.get("conservation_error_after_limiter", 0.0)
        )
        limiter_mass_correction_total = float(scale) * float(
            limited.get("limiter_mass_correction_total", 0.0)
        )

        progress_temperature_min = 0.0
        progress_temperature_max = 0.0
        progress_cfl_max = 0.0
        if record_history or log_progress:
            out_vol_rate_raw = np.asarray(
                assembled_for_limiter["out_vol_rate_raw"], dtype=np.float64
            )
            in_vol_rate_raw = np.asarray(
                assembled_for_limiter["in_vol_rate_raw"], dtype=np.float64
            )
            local_outgoing_cfl = used_dt * out_vol_rate_raw / np.maximum(volumes, 1e-16)
            local_incoming_cfl = used_dt * in_vol_rate_raw / np.maximum(volumes, 1e-16)
            progress_temperature_min = (
                float(np.min(temperature)) if temperature.size else 0.0
            )
            progress_temperature_max = (
                float(np.max(temperature)) if temperature.size else 0.0
            )
            progress_cfl_max = (
                float(np.max(local_outgoing_cfl)) if local_outgoing_cfl.size else 0.0
            )

            if record_history:
                if resolved_history_mode == "compact":
                    _append_compact_thermal_history_sample(
                        compact_history,
                        step=step,
                        dt_used=float(used_dt),
                        temperature_min=progress_temperature_min,
                        temperature_max=progress_temperature_max,
                        cfl_max=progress_cfl_max,
                    )
                elif allow_expensive_step_diagnostics:
                    temperature_mean = (
                        float(np.mean(temperature)) if temperature.size else 0.0
                    )
                    max_change = (
                        float(np.max(np.abs(temperature - old)))
                        if temperature.size
                        else 0.0
                    )
                    raw_integral = float(np.sum(updated_raw * volumes))
                    expected_raw_integral = (
                        old_integral
                        + advective_integral_change
                        + diffusion_integral_change
                        + source_integral_change
                        + (
                            float(np.sum(wall_delta * volumes))
                            if config.wall_boundary_mode == "fixed_heat_flux"
                            else 0.0
                        )
                    )
                    raw_balance_error = float(raw_integral - expected_raw_integral)
                    new_integral = float(np.sum(updated * volumes))
                    cfl_incoming_max = (
                        float(np.max(local_incoming_cfl))
                        if local_incoming_cfl.size
                        else 0.0
                    )
                    cfl_warning_step = bool(progress_cfl_max > config.cfl_limit)
                    clipped_cell_count = int(np.count_nonzero(clipped_mask))
                    limiter_active_cell_count = int(
                        np.count_nonzero(advection_limiter_active_mask)
                    )
                    non_advective_limiter_active_cell_count = int(
                        np.count_nonzero(non_advective_limiter_active_mask)
                    )
                    entry = _build_compact_thermal_history_entry(
                        step=step,
                        used_dt=float(used_dt),
                        temperature_min=progress_temperature_min,
                        temperature_max=progress_temperature_max,
                        temperature_mean=temperature_mean,
                        max_change=max_change,
                        advective_integral_change=float(advective_integral_change),
                        diffusion_integral_change=float(diffusion_integral_change),
                        source_integral_change=float(source_integral_change),
                        non_advective_limiter_integral_change=float(
                            non_advective_limiter_integral_change
                        ),
                        clamp_integral_change=float(clamp_integral_change),
                        raw_balance_error=float(raw_balance_error),
                        energy_proxy=float(new_integral),
                        cfl_max=progress_cfl_max,
                        cfl_incoming_max=cfl_incoming_max,
                        cfl_warning=cfl_warning_step,
                        clipped_cell_count=clipped_cell_count,
                        limiter_active_cell_count=limiter_active_cell_count,
                        non_advective_limiter_active_cell_count=non_advective_limiter_active_cell_count,
                    )
                    if config.full_step_diagnostics and record_diagnostics:
                        entry.update(
                            {
                                "advective_mass_in": float(adv_mass_in),
                                "advective_mass_out": float(adv_mass_out),
                                "advective_net": float(adv_net),
                                "diffusion_integral_change_unlimited": float(
                                    diffusion_integral_change_unlimited
                                ),
                                "source_integral_change_unlimited": float(
                                    source_integral_change_unlimited
                                ),
                                "advection_delta_max_abs": float(
                                    np.max(np.abs(advection_delta))
                                ),
                                "diffusion_delta_max_abs": float(
                                    np.max(np.abs(diff_delta))
                                ),
                                "source_delta_max_abs": float(
                                    np.max(np.abs(source_delta))
                                ),
                                "clamp_delta_max_abs": float(
                                    np.max(np.abs(clamp_delta))
                                ),
                                "non_advective_limiter_scale_min": float(
                                    np.min(non_advective_limiter_scale)
                                ),
                                "non_advective_limiter_active_fraction": float(
                                    non_advective_limiter_active_cell_count
                                    / max(1, non_advective_limiter_active_mask.size)
                                ),
                                "overshoot_before_non_advective_limiter": float(
                                    np.max(pre_non_adv_overshoot)
                                )
                                if boundedness_enabled and pre_non_adv_overshoot.size
                                else 0.0,
                                "undershoot_before_non_advective_limiter": float(
                                    np.max(pre_non_adv_undershoot)
                                )
                                if boundedness_enabled and pre_non_adv_undershoot.size
                                else 0.0,
                                "diffusion_laplacian_max_abs": float(
                                    np.max(np.abs(lap))
                                )
                                if lap.size
                                else 0.0,
                                "overshoot_before_clip": float(np.max(overshoot))
                                if overshoot.size
                                else 0.0,
                                "undershoot_before_clip": float(np.max(undershoot))
                                if undershoot.size
                                else 0.0,
                                "overshoot_cell_count": int(
                                    np.count_nonzero(overshoot_mask)
                                ),
                                "undershoot_cell_count": int(
                                    np.count_nonzero(undershoot_mask)
                                ),
                                "limiter_active_fraction": float(
                                    limiter_active_cell_count
                                    / max(1, advection_limiter_active_mask.size)
                                ),
                                "limiter_max_flux_scale_down": float(
                                    limited["limiter_max_flux_scale_down"]
                                ),
                                "conservation_error_after_limiter": float(
                                    limiter_conservation_error
                                ),
                                "limiter_mass_correction_total": float(
                                    limiter_mass_correction_total
                                ),
                                "cumulative_advective_mass_in": float(
                                    cumulative_adv_in
                                ),
                                "cumulative_advective_mass_out": float(
                                    cumulative_adv_out
                                ),
                                "cumulative_source": float(cumulative_source),
                                "cumulative_source_unlimited": float(
                                    cumulative_source_unlimited
                                ),
                                "cumulative_diffusion": float(cumulative_diffusion),
                                "cumulative_diffusion_unlimited": float(
                                    cumulative_diffusion_unlimited
                                ),
                                "cumulative_non_advective_limiter": float(
                                    cumulative_non_advective_limiter
                                ),
                                "cumulative_clamp": float(cumulative_clamp),
                            }
                        )
                    history.append(entry)
                else:
                    history.append(
                        _build_fast_thermal_history_entry(
                            step=step,
                            used_dt=float(used_dt),
                            temperature_min=progress_temperature_min,
                            temperature_max=progress_temperature_max,
                            cfl_max=progress_cfl_max,
                            cfl_warning=bool(progress_cfl_max > config.cfl_limit),
                        )
                    )

        if step in snapshot_steps:
            snapshots[int(step)] = np.asarray(temperature, dtype=np.float64).copy()

        if config.debug_artifacts or step == config.steps:
            final_lap = lap
            final_raw_state = raw_state_physical
            final_limited_state = limited_state_physical
            final_updated_before_non_advective_limiter = pre_non_advective_update
            final_updated_before_clip = np.asarray(updated_raw, dtype=np.float64)
            final_clipped_mask = clipped_mask
            final_overshoot_mask = overshoot_mask
            final_undershoot_mask = undershoot_mask
            final_advection_limiter_active_mask = advection_limiter_active_mask
            final_non_advective_limiter_active_mask = non_advective_limiter_active_mask
            final_non_advective_limiter_scale = non_advective_limiter_scale
            final_advection_limiter_scale = advection_limiter_scale
            final_advection_delta = advection_delta
            final_diffusion_delta = diff_delta
            final_source_delta = source_delta
            final_clamp_delta = clamp_delta
            if "face_scalar_flux_raw" in limited:
                final_face_flux_raw = np.asarray(
                    limited["face_scalar_flux_raw"], dtype=np.float64
                ) * float(scale)
                final_face_flux_limited = np.asarray(
                    limited["face_scalar_flux_limited"], dtype=np.float64
                ) * float(scale)
                final_in_mass_raw = np.asarray(
                    limited["in_mass_raw"], dtype=np.float64
                ) * float(scale)
                final_out_mass_raw = np.asarray(
                    limited["out_mass_raw"], dtype=np.float64
                ) * float(scale)
                final_in_mass_lim = np.asarray(
                    limited["in_mass_limited"], dtype=np.float64
                ) * float(scale)
                final_out_mass_lim = np.asarray(
                    limited["out_mass_limited"], dtype=np.float64
                ) * float(scale)
            final_limiter_out = np.asarray(
                limited.get("limiter_scale_out", np.ones_like(temperature)),
                dtype=np.float64,
            )
            final_limiter_in = np.asarray(
                limited.get("limiter_scale_in", np.ones_like(temperature)),
                dtype=np.float64,
            )

        if log_progress:
            print(
                "[gmsh-tetra-thermal] "
                f"step {step}/{config.steps}, "
                f"T_min={progress_temperature_min:.6f}, "
                f"T_max={progress_temperature_max:.6f}, "
                f"CFL={progress_cfl_max:.4f}"
            )

    final_out_rate = np.asarray(
        assembled_for_limiter["out_vol_rate_raw"], dtype=np.float64
    )
    final_in_rate = np.asarray(
        assembled_for_limiter["in_vol_rate_raw"], dtype=np.float64
    )
    final_local_outgoing_cfl = used_dt * final_out_rate / np.maximum(volumes, 1e-16)
    final_local_incoming_cfl = used_dt * final_in_rate / np.maximum(volumes, 1e-16)

    final_energy = float(np.sum(np.asarray(temperature, dtype=np.float64) * volumes))
    wall_integral_change = cumulative_wall_applied_energy / (
        float(config.rho) * float(config.cp)
    )
    diffusion_non_wall_integral_change = cumulative_diffusion - (
        wall_integral_change
        if config.wall_boundary_mode == "fixed_temperature"
        else 0.0
    )
    expected_energy_no_clamp = (
        initial_energy
        + cumulative_advective_integral_change
        + diffusion_non_wall_integral_change
        + cumulative_source
        + wall_integral_change
    )

    source = flux_diagnostics if flux_diagnostics is not None else {}
    velocity_meta = velocity_metadata if velocity_metadata is not None else {}
    execution_policy_payload = execution_policy if execution_policy is not None else {}
    resolved_torch_compile_mode = _resolve_thermal_torch_compile_mode(
        config,
        actual_backend="numpy",
    )
    torch_compile_step_metadata = _thermal_torch_compile_metadata(
        compile_mode=resolved_torch_compile_mode,
        reason="torch backend not active",
    )
    recorded_steps = (
        [int(step) for step in compact_history.get("step", [])]
        if resolved_history_mode == "compact"
        else [int(entry["step"]) for entry in history]
    )
    history_settings = {
        "collect_history": bool(config.collect_history),
        "diagnostics_mode": str(config.diagnostics_mode),
        "resolved_diagnostics_mode": str(resolved_diagnostics_mode),
        "history_mode": str(config.history_mode),
        "resolved_history_mode": str(resolved_history_mode),
        "history_stride": int(config.history_stride),
        "diagnostics_stride": int(config.diagnostics_stride),
        "full_step_diagnostics": bool(config.full_step_diagnostics),
        "debug_artifacts": bool(config.debug_artifacts),
        "torch_compile_step_body": str(config.torch_compile_step_body),
        "resolved_torch_compile_step_body": str(resolved_torch_compile_mode),
        **torch_compile_step_metadata,
        "execution_mode": _thermal_execution_mode(config),
        "recorded_steps": recorded_steps,
    }
    elapsed_time = float(config.steps) * float(used_dt)
    wall_heat_transfer = _build_wall_heat_transfer(
        config,
        wall_boundary,
        final_requested_energy=final_wall_requested_energy,
        final_applied_energy=final_wall_applied_energy,
        cumulative_requested_energy=cumulative_wall_requested_energy,
        cumulative_energy_after_limiter_before_clipping=(
            cumulative_wall_operator_energy
        ),
        cumulative_applied_energy=cumulative_wall_applied_energy,
        elapsed_time=elapsed_time,
    )
    global_energy_balance = _build_global_energy_balance(
        config,
        initial_integral_t_dv=initial_energy,
        final_integral_t_dv=final_energy,
        cumulative_advective_in=cumulative_adv_in,
        cumulative_advective_out=cumulative_adv_out,
        cumulative_advective_integral_change=cumulative_advective_integral_change,
        cumulative_diffusion_integral_change=cumulative_diffusion,
        cumulative_source_integral_change=cumulative_source,
        cumulative_clamp_integral_change=cumulative_clamp,
        cumulative_wall_operator_energy_j=cumulative_wall_operator_energy,
        cumulative_wall_applied_energy_j=cumulative_wall_applied_energy,
    )

    return {
        "temperature": np.asarray(temperature, dtype=np.float64),
        "history": history,
        "history_compact": compact_history,
        "history_settings": history_settings,
        "execution_policy": execution_policy_payload,
        "model_capabilities": _thermal_v2_model_capabilities(),
        "thermal_observables": _build_thermal_observables(
            mesh,
            np.asarray(temperature, dtype=np.float64),
            np.asarray(face_normal_velocity, dtype=np.float64),
        ),
        "wall_heat_transfer": wall_heat_transfer,
        "global_energy_balance": global_energy_balance,
        "cfl_values": cfl_values_initial,
        "cfl_max": float(cfl_max_initial),
        "cfl_warning": bool(cfl_warning),
        "dt_control": {
            "dt_mode": config.dt_mode,
            "requested_dt": float(requested_dt),
            "used_dt": float(used_dt),
            "advection_dt_limit": float(advection_dt_limit),
            "diffusion_dt_limit": float(diffusion_dt_limit),
            "cfl_target": float(config.cfl_target),
            "diffusion_stability_factor": float(config.diffusion_stability_factor),
            "cfl_limit": float(config.cfl_limit),
            "diffusion_stability_warning": bool(diffusion_stability_warning),
        },
        "boundary_setup": {
            "left_inlet_faces": int(left_faces.size),
            "right_inlet_faces": int(right_faces.size),
            "outlet_faces": int(outlet_faces.size),
            "wall_faces": int(wall_faces.size),
            "heated_wall_faces": int(wall_boundary.faces.size),
            "heated_wall_area_m2": float(wall_boundary.area),
            "heated_wall_group_face_counts": dict(wall_boundary.group_face_counts),
            "left_inlet_cells": int(left_cells.size),
            "right_inlet_cells": int(right_cells.size),
            "diffusion_bc": {
                "dirichlet_faces": int(len(precompute.boundary_dirichlet)),
                "no_flux_faces": int(precompute.boundary_no_flux_faces.size),
                "outlet_mode": "advective_outflow_with_backflow_suppression_and_zero_diffusive_flux",
            },
        },
        "transport_info": {
            "equation": "dT/dt + u.grad(T) = alpha*laplacian(T) + Q/(rho*cp)",
            "velocity_source": str(velocity_meta.get("velocity_source", "unspecified")),
            "flow_solved": bool(velocity_meta.get("flow_solved", False)),
            "pressure_solved": bool(velocity_meta.get("pressure_solved", False)),
            "execution_profile": str(
                execution_policy_payload.get("execution_profile", "")
            ),
            "production_backend_satisfied": bool(
                execution_policy_payload.get("production_backend_satisfied", False)
            ),
            "advection_scheme": config.limiter_scheme,
            "normalize_bounded_advection": bool(config.normalize_bounded_advection),
            "limit_non_advective_update": bool(config.limit_non_advective_update),
            "gradient_method": config.gradient_method,
            "laplacian_method": config.laplacian_method,
            "inlet_temperature_mode": "face_dirichlet_for_inflow",
            "wall_temperature_mode": str(config.wall_boundary_mode),
            "outlet_temperature_mode": "advective_outflow_with_backflow_suppression_and_zero_diffusive_flux",
            "boundedness_enabled": bool(boundedness_enabled),
        },
        "performance_diagnostics": {
            "advection_flux_assemblies_per_step": 1,
            "normalized_bounded_advection_avoids_reassembly": True,
            "history_entries_recorded": int(len(recorded_steps)),
            "execution_mode": _thermal_execution_mode(config),
            "resolved_diagnostics_mode": str(resolved_diagnostics_mode),
            "resolved_torch_compile_step_body": str(resolved_torch_compile_mode),
            **torch_compile_step_metadata,
            "degraded_execution_mode": bool(
                execution_policy_payload.get("degraded_execution_mode", False)
            ),
        },
        "flux_diagnostics": source,
        "clipping": {
            "enabled": bool(config.clipping_enabled and boundedness_enabled),
            "used": bool(clipping_used),
            "max_overshoot_before_clip": float(max_overshoot_before_clip),
            "max_undershoot_before_clip": float(max_undershoot_before_clip),
            "max_overshoot_before_limiter": float(max_overshoot_before_limiter),
            "max_undershoot_before_limiter": float(max_undershoot_before_limiter),
            "max_overshoot_before_non_advective_limiter": float(
                max_overshoot_before_non_advective_limiter
            ),
            "max_undershoot_before_non_advective_limiter": float(
                max_undershoot_before_non_advective_limiter
            ),
            "final_clipped_cell_count": int(np.count_nonzero(final_clipped_mask)),
            "final_overshoot_cell_count": int(np.count_nonzero(final_overshoot_mask)),
            "final_undershoot_cell_count": int(np.count_nonzero(final_undershoot_mask)),
        },
        "limiter": {
            "limiter_active_cell_count": int(
                np.count_nonzero(final_advection_limiter_active_mask)
            ),
            "limiter_active_fraction": float(
                np.count_nonzero(final_advection_limiter_active_mask)
                / max(1, temperature.size)
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
        "non_advective_limiter": {
            "enabled": bool(boundedness_enabled and config.limit_non_advective_update),
            "final_active_cell_count": int(
                np.count_nonzero(final_non_advective_limiter_active_mask)
            ),
            "final_active_fraction": float(
                np.count_nonzero(final_non_advective_limiter_active_mask)
                / max(1, temperature.size)
            ),
            "final_scale_min": float(np.min(final_non_advective_limiter_scale)),
            "cumulative_integral_change": float(cumulative_non_advective_limiter),
        },
        "energy_proxy": {
            "initial_integral_T_dV": float(initial_energy),
            "final_integral_T_dV": float(final_energy),
            "cumulative_advective_in": float(cumulative_adv_in),
            "cumulative_advective_out": float(cumulative_adv_out),
            "cumulative_advective_integral_change": float(
                cumulative_advective_integral_change
            ),
            "cumulative_diffusion_integral_change": float(cumulative_diffusion),
            "cumulative_non_wall_diffusion_integral_change": float(
                diffusion_non_wall_integral_change
            ),
            "cumulative_wall_integral_change": float(wall_integral_change),
            "cumulative_wall_energy_j": float(cumulative_wall_applied_energy),
            "cumulative_diffusion_integral_change_unlimited": float(
                cumulative_diffusion_unlimited
            ),
            "cumulative_source_integral_change": float(cumulative_source),
            "cumulative_source_integral_change_unlimited": float(
                cumulative_source_unlimited
            ),
            "cumulative_non_advective_limiter_integral_change": float(
                cumulative_non_advective_limiter
            ),
            "cumulative_clamp_integral_change": float(cumulative_clamp),
            "expected_final_without_clamp": float(expected_energy_no_clamp),
            "balance_error_without_clamp": float(
                final_energy - cumulative_clamp - expected_energy_no_clamp
            ),
            "balance_error_with_clamp": float(
                final_energy - (expected_energy_no_clamp + cumulative_clamp)
            ),
        },
        "local_cfl_audit": {
            "max_local_outgoing_cfl": float(np.max(final_local_outgoing_cfl))
            if final_local_outgoing_cfl.size
            else 0.0,
            "max_local_incoming_cfl": float(np.max(final_local_incoming_cfl))
            if final_local_incoming_cfl.size
            else 0.0,
            "top_cells_by_local_outgoing_cfl": _top_cell_metrics(
                final_local_outgoing_cfl, top_k=10
            ),
            "top_cells_by_local_incoming_cfl": _top_cell_metrics(
                final_local_incoming_cfl, top_k=10
            ),
        },
        "final_step_debug": (
            {
                "raw_state_before_limiter": final_raw_state,
                "limited_state_before_clip": final_limited_state,
                "updated_raw_before_non_advective_limiter": final_updated_before_non_advective_limiter,
                "updated_raw_before_clip": final_updated_before_clip,
                "clipped_mask": final_clipped_mask,
                "overshoot_mask": final_overshoot_mask,
                "undershoot_mask": final_undershoot_mask,
                "advection_limiter_active_mask": final_advection_limiter_active_mask,
                "non_advective_limiter_active_mask": final_non_advective_limiter_active_mask,
                "advection_limiter_scale": final_advection_limiter_scale,
                "non_advective_limiter_scale": final_non_advective_limiter_scale,
                "advection_delta": final_advection_delta,
                "diffusion_delta": final_diffusion_delta,
                "source_delta": final_source_delta,
                "clamp_delta": final_clamp_delta,
                "face_scalar_flux_raw": final_face_flux_raw,
                "face_scalar_flux_limited": final_face_flux_limited,
                "laplacian": final_lap,
                "left_inlet_cells": left_cells,
                "right_inlet_cells": right_cells,
                "boundary_face_groups": boundary_face_groups,
            }
            if config.debug_artifacts
            else {}
        ),
        "diffusion_diagnostics": {
            "diffusivity": float(config.thermal_diffusivity),
            "diffusion_backend": "numpy"
            if config.thermal_diffusivity > 0.0
            else "none",
            "laplacian_method": config.laplacian_method,
            "gradient_method": config.gradient_method,
            "diffusion_dt_limit": float(diffusion_dt_limit),
            "diffusion_stability_warning": bool(diffusion_stability_warning),
            "max_laplacian_abs_final_step": float(np.max(np.abs(final_lap)))
            if final_lap.size
            else 0.0,
        },
        "final_stats": {
            "min": float(np.min(temperature)),
            "max": float(np.max(temperature)),
            "mean": float(np.mean(temperature)),
        },
        "backend_execution": build_backend_execution_diagnostics(
            requested_backend=str(config.backend),
            stepping_backend="numpy",
            device="cpu",
            torch_device=str(config.torch_device),
            used_numpy_fallback=bool(
                backend_selection is not None and backend_selection.used_numpy_fallback
            ),
            all_core_arrays_on_cuda=False,
            per_step_large_cpu_gpu_transfer_warning=False,
            cpu_gpu_transfer_notes=[],
        ),
        "snapshots": snapshots,
    }


def _run_tetra_thermal_debug_torch(
    mesh: ImportedTetraMesh,
    config: GmshTetraThermalConfig,
    *,
    precompute: ScalarBackendPrecompute,
    backend_selection: BackendSelection,
    face_normal_velocity: np.ndarray,
    flux_diagnostics: Dict[str, float] | None = None,
    velocity_metadata: Dict[str, object] | None = None,
    execution_policy: Dict[str, object] | None = None,
    wall_boundary: _ResolvedWallBoundary | None = None,
) -> Dict[str, object]:
    _validate_thermal_config(config)
    wall_boundary = wall_boundary or _resolve_wall_boundary(mesh, config)
    if precompute.torch_backend is None:
        raise RuntimeError("Torch backend precompute is not available.")

    tb = precompute.torch_backend
    torch = tb["torch"]
    device = tb["device"]
    volumes_t = tb["vol"]
    volumes = precompute.cell_volumes
    n_cells = int(volumes.shape[0])

    min_t = config.min_temperature
    max_t = config.max_temperature
    boundedness_enabled = min_t is not None and max_t is not None
    left_faces = precompute.left_faces
    right_faces = precompute.right_faces
    left_cells = precompute.left_cells
    right_cells = precompute.right_cells
    outlet_faces = precompute.outlet_faces
    wall_faces = precompute.wall_faces
    boundary_face_groups = precompute.boundary_face_groups
    diffusion_stencil = precompute.diffusion_stencil
    if diffusion_stencil is None:
        raise ValueError("Thermal diffusion precompute is required.")

    advection_dt_limit = estimate_stable_dt(precompute)
    diffusion_dt_limit = estimate_diffusion_dt_limit(
        diffusion_stencil, float(config.thermal_diffusivity)
    )

    requested_dt = float(config.dt)
    used_dt = requested_dt
    if config.dt_mode == "auto":
        dt_candidates: list[float] = []
        if np.isfinite(advection_dt_limit) and advection_dt_limit > 0.0:
            dt_candidates.append(float(config.cfl_target) * float(advection_dt_limit))
        if np.isfinite(diffusion_dt_limit) and diffusion_dt_limit > 0.0:
            dt_candidates.append(
                float(config.diffusion_stability_factor) * float(diffusion_dt_limit)
            )
        if dt_candidates:
            used_dt = float(min(dt_candidates))
    if used_dt <= 0.0:
        raise FloatingPointError("Resolved dt must be positive.")

    cfl_values_initial, cfl_max_initial = compute_cfl_metrics(precompute, used_dt)
    cfl_warning = bool(cfl_max_initial > float(config.cfl_limit))
    diffusion_stability_warning = bool(
        np.isfinite(diffusion_dt_limit)
        and diffusion_dt_limit > 0.0
        and used_dt > diffusion_dt_limit
    )

    temperature = torch.full(
        (n_cells,),
        float(config.initial_temperature),
        dtype=torch.float64,
        device=device,
    )
    inlet_vol_flux_in, outlet_vol_flux_out = _boundary_volume_flux_metrics(precompute)
    source_term = float(config.heat_source) / (float(config.rho) * float(config.cp))
    initial_energy = torch.sum(temperature * volumes_t)
    inlet_vol_flux_in_t = torch.full(
        (),
        float(inlet_vol_flux_in),
        dtype=torch.float64,
        device=device,
    )
    outlet_vol_flux_out_t = torch.full(
        (),
        float(outlet_vol_flux_out),
        dtype=torch.float64,
        device=device,
    )
    wall_owners_t = torch.as_tensor(
        wall_boundary.owners, dtype=torch.long, device=device
    )
    wall_areas_t = torch.as_tensor(
        wall_boundary.areas, dtype=torch.float64, device=device
    )
    if wall_boundary.faces.size:
        wall_face_delta = np.asarray(
            mesh.face_centers[wall_boundary.faces], dtype=np.float64
        ) - np.asarray(mesh.cell_centers[wall_boundary.owners], dtype=np.float64)
        wall_normals = np.asarray(
            mesh.face_normals[wall_boundary.faces], dtype=np.float64
        )
        wall_distance = np.maximum(
            np.abs(np.einsum("ij,ij->i", wall_face_delta, wall_normals)), 1e-12
        )
        wall_coefficients = wall_boundary.areas / wall_distance
    else:
        wall_coefficients = np.zeros((0,), dtype=np.float64)
    wall_coefficients_t = torch.as_tensor(
        wall_coefficients, dtype=torch.float64, device=device
    )
    resolved_diagnostics_mode = _resolve_thermal_diagnostics_mode(config)
    allow_expensive_step_diagnostics = resolved_diagnostics_mode == "debug"
    resolved_history_mode = _resolve_thermal_history_mode(config)
    resolved_torch_compile_mode = _resolve_thermal_torch_compile_mode(
        config,
        actual_backend="torch",
        device_type=str(getattr(device, "type", config.torch_device)),
    )

    history: list[ThermalStepHistoryEntry] = []
    compact_history = (
        _empty_compact_thermal_history() if resolved_history_mode == "compact" else {}
    )
    snapshots: Dict[int, np.ndarray] = {}
    snapshot_steps = {int(s) for s in config.snapshot_steps if int(s) > 0}

    zero_scalar = torch.zeros((), dtype=torch.float64, device=device)
    max_overshoot_before_clip = zero_scalar.clone()
    max_undershoot_before_clip = zero_scalar.clone()
    max_overshoot_before_limiter = zero_scalar.clone()
    max_undershoot_before_limiter = zero_scalar.clone()
    cumulative_adv_in = zero_scalar.clone()
    cumulative_adv_out = zero_scalar.clone()
    cumulative_advective_integral_change = zero_scalar.clone()
    cumulative_source = zero_scalar.clone()
    cumulative_diffusion = zero_scalar.clone()
    cumulative_source_unlimited = zero_scalar.clone()
    cumulative_diffusion_unlimited = zero_scalar.clone()
    cumulative_clamp = zero_scalar.clone()
    cumulative_non_advective_limiter = zero_scalar.clone()
    cumulative_wall_requested_energy = zero_scalar.clone()
    cumulative_wall_operator_energy = zero_scalar.clone()
    cumulative_wall_applied_energy = zero_scalar.clone()
    final_wall_requested_energy = zero_scalar.clone()
    final_wall_applied_energy = zero_scalar.clone()
    max_overshoot_before_non_advective_limiter = zero_scalar.clone()
    max_undershoot_before_non_advective_limiter = zero_scalar.clone()

    final_lap = torch.zeros_like(temperature)
    final_raw_state = temperature.clone()
    final_limited_state = temperature.clone()
    final_updated_before_clip = temperature.clone()
    final_updated_before_non_advective_limiter = temperature.clone()
    final_clipped_mask = torch.zeros_like(temperature, dtype=torch.bool)
    final_overshoot_mask = torch.zeros_like(temperature, dtype=torch.bool)
    final_undershoot_mask = torch.zeros_like(temperature, dtype=torch.bool)
    final_advection_limiter_active_mask = torch.zeros_like(
        temperature, dtype=torch.bool
    )
    final_non_advective_limiter_active_mask = torch.zeros_like(
        temperature, dtype=torch.bool
    )
    final_non_advective_limiter_scale = torch.ones_like(temperature)
    final_advection_limiter_scale = torch.ones_like(temperature)
    final_advection_delta = torch.zeros_like(temperature)
    final_diffusion_delta = torch.zeros_like(temperature)
    final_source_delta = torch.zeros_like(temperature)
    final_clamp_delta = torch.zeros_like(temperature)
    final_face_flux_raw = torch.zeros(
        (precompute.face_to_cells.shape[0],), dtype=torch.float64, device=device
    )
    final_face_flux_limited = torch.zeros_like(final_face_flux_raw)
    final_in_mass_raw = torch.zeros_like(temperature)
    final_out_mass_raw = torch.zeros_like(temperature)
    final_in_mass_lim = torch.zeros_like(temperature)
    final_out_mass_lim = torch.zeros_like(temperature)
    final_limiter_out = torch.ones_like(temperature)
    final_limiter_in = torch.ones_like(temperature)
    final_out_vol_rate = torch.zeros_like(temperature)
    final_in_vol_rate = torch.zeros_like(temperature)
    compiled_fast_step_fn = None
    torch_compile_step_metadata = _thermal_torch_compile_metadata(
        compile_mode=resolved_torch_compile_mode,
        reason="step-body compile only enabled for fast torch path",
    )
    if resolved_diagnostics_mode == "fast":

        def _fast_step_body(current_temperature: object) -> tuple[object, ...]:
            return _thermal_torch_fast_step_body(
                current_temperature,
                precompute=precompute,
                gradient_method=config.gradient_method,
                laplacian_method=config.laplacian_method,
                limiter_scheme=config.limiter_scheme,
                normalize_bounded_advection=config.normalize_bounded_advection,
                limit_non_advective_update=config.limit_non_advective_update,
                clipping_enabled=config.clipping_enabled,
                boundedness_enabled=boundedness_enabled,
                min_temperature=min_t,
                max_temperature=max_t,
                inlet_vol_flux_in_t=inlet_vol_flux_in_t,
                outlet_vol_flux_out_t=outlet_vol_flux_out_t,
                left_inlet_temperature=float(config.left_inlet_temperature),
                right_inlet_temperature=float(config.right_inlet_temperature),
                used_dt=float(used_dt),
                thermal_diffusivity=float(config.thermal_diffusivity),
                source_term=float(source_term),
                wall_boundary_mode=config.wall_boundary_mode,
                wall_temperature=config.wall_temperature,
                wall_heat_flux=config.wall_heat_flux,
                rho_cp=float(config.rho) * float(config.cp),
                wall_owners_t=wall_owners_t,
                wall_areas_t=wall_areas_t,
                wall_coefficients_t=wall_coefficients_t,
            )

        compiled_fast_step_fn, torch_compile_step_metadata = (
            _maybe_compile_thermal_torch_step_body(
                torch,
                _fast_step_body,
                compile_mode=resolved_torch_compile_mode,
                device_type=str(getattr(device, "type", config.torch_device)),
                disabled_reason=_thermal_torch_compile_disabled_reason(
                    config,
                    resolved_compile_mode=resolved_torch_compile_mode,
                    actual_backend="torch",
                    device_type=str(getattr(device, "type", config.torch_device)),
                ),
            )
        )
    elif resolved_torch_compile_mode == "on":
        torch_compile_step_metadata = _thermal_torch_compile_metadata(
            compile_mode=resolved_torch_compile_mode,
            reason="step-body compile is only wired for fast diagnostics mode",
        )

    for step in range(1, config.steps + 1):
        old = temperature
        record_history = _should_record_thermal_history_step(
            config, step, snapshot_steps=snapshot_steps
        )
        record_diagnostics = allow_expensive_step_diagnostics and (
            _should_collect_thermal_diagnostics_step(
                config, step, snapshot_steps=snapshot_steps
            )
        )
        log_progress = _should_log_thermal_progress_step(config, step)
        capture_final_state = bool(config.debug_artifacts or step == config.steps)

        if compiled_fast_step_fn is not None:
            try:
                (
                    updated,
                    out_vol_rate_raw,
                    in_vol_rate_raw,
                    adv_mass_in,
                    adv_mass_out,
                    advective_integral_change,
                    diffusion_integral_change,
                    source_integral_change,
                    diffusion_integral_change_unlimited,
                    source_integral_change_unlimited,
                    clamp_integral_change,
                    non_advective_limiter_integral_change,
                    wall_requested_energy,
                    wall_operator_energy,
                    wall_applied_energy,
                    clipped_mask,
                    overshoot_mask,
                    undershoot_mask,
                    advection_limiter_active_mask,
                    non_advective_limiter_active_mask,
                    non_advective_limiter_scale,
                    limiter_scale_out,
                    limiter_scale_in,
                    step_max_overshoot_before_clip,
                    step_max_undershoot_before_clip,
                    step_max_overshoot_before_limiter,
                    step_max_undershoot_before_limiter,
                    step_max_overshoot_before_non_advective_limiter,
                    step_max_undershoot_before_non_advective_limiter,
                ) = compiled_fast_step_fn(old)
            except Exception as exc:
                compiled_fast_step_fn = None
                torch_compile_step_metadata = _thermal_torch_compile_metadata(
                    compile_mode=resolved_torch_compile_mode,
                    reason=f"compile failed during execution: {exc!s}",
                )
                (
                    updated,
                    out_vol_rate_raw,
                    in_vol_rate_raw,
                    adv_mass_in,
                    adv_mass_out,
                    advective_integral_change,
                    diffusion_integral_change,
                    source_integral_change,
                    diffusion_integral_change_unlimited,
                    source_integral_change_unlimited,
                    clamp_integral_change,
                    non_advective_limiter_integral_change,
                    wall_requested_energy,
                    wall_operator_energy,
                    wall_applied_energy,
                    clipped_mask,
                    overshoot_mask,
                    undershoot_mask,
                    advection_limiter_active_mask,
                    non_advective_limiter_active_mask,
                    non_advective_limiter_scale,
                    limiter_scale_out,
                    limiter_scale_in,
                    step_max_overshoot_before_clip,
                    step_max_undershoot_before_clip,
                    step_max_overshoot_before_limiter,
                    step_max_undershoot_before_limiter,
                    step_max_overshoot_before_non_advective_limiter,
                    step_max_undershoot_before_non_advective_limiter,
                ) = _fast_step_body(old)

            if not bool(torch.all(torch.isfinite(updated)).item()):
                raise FloatingPointError(
                    f"Temperature became non-finite at step {step}."
                )

            if n_cells:
                max_overshoot_before_clip = torch.maximum(
                    max_overshoot_before_clip,
                    step_max_overshoot_before_clip,
                )
                max_undershoot_before_clip = torch.maximum(
                    max_undershoot_before_clip,
                    step_max_undershoot_before_clip,
                )
                max_overshoot_before_limiter = torch.maximum(
                    max_overshoot_before_limiter,
                    step_max_overshoot_before_limiter,
                )
                max_undershoot_before_limiter = torch.maximum(
                    max_undershoot_before_limiter,
                    step_max_undershoot_before_limiter,
                )
                max_overshoot_before_non_advective_limiter = torch.maximum(
                    max_overshoot_before_non_advective_limiter,
                    step_max_overshoot_before_non_advective_limiter,
                )
                max_undershoot_before_non_advective_limiter = torch.maximum(
                    max_undershoot_before_non_advective_limiter,
                    step_max_undershoot_before_non_advective_limiter,
                )

            cumulative_adv_in += adv_mass_in
            cumulative_adv_out += adv_mass_out
            cumulative_advective_integral_change += advective_integral_change
            cumulative_source += source_integral_change
            cumulative_diffusion += diffusion_integral_change
            cumulative_source_unlimited += source_integral_change_unlimited
            cumulative_diffusion_unlimited += diffusion_integral_change_unlimited
            cumulative_clamp += clamp_integral_change
            cumulative_non_advective_limiter += non_advective_limiter_integral_change
            cumulative_wall_requested_energy += wall_requested_energy
            cumulative_wall_operator_energy += wall_operator_energy
            cumulative_wall_applied_energy += wall_applied_energy
            final_wall_requested_energy = wall_requested_energy
            final_wall_applied_energy = wall_applied_energy

            temperature = updated
            advection_limiter_scale = torch.minimum(
                limiter_scale_out,
                limiter_scale_in,
            )

            progress_temperature_min = 0.0
            progress_temperature_max = 0.0
            progress_cfl_max = 0.0
            if record_history or log_progress:
                local_outgoing_cfl = used_dt * out_vol_rate_raw / volumes_t
                step_scalar_values = _materialize_torch_scalar_batch(
                    torch,
                    {
                        "temperature_min": (
                            torch.min(temperature) if n_cells else zero_scalar
                        ),
                        "temperature_max": (
                            torch.max(temperature) if n_cells else zero_scalar
                        ),
                        "cfl_max": (
                            torch.max(local_outgoing_cfl) if n_cells else zero_scalar
                        ),
                    },
                    device=device,
                )
                progress_temperature_min = float(step_scalar_values["temperature_min"])
                progress_temperature_max = float(step_scalar_values["temperature_max"])
                progress_cfl_max = float(step_scalar_values["cfl_max"])

                if record_history:
                    if resolved_history_mode == "compact":
                        _append_compact_thermal_history_sample(
                            compact_history,
                            step=step,
                            dt_used=float(used_dt),
                            temperature_min=progress_temperature_min,
                            temperature_max=progress_temperature_max,
                            cfl_max=progress_cfl_max,
                        )
                    elif resolved_history_mode == "dict":
                        history.append(
                            _build_fast_thermal_history_entry(
                                step=step,
                                used_dt=float(used_dt),
                                temperature_min=progress_temperature_min,
                                temperature_max=progress_temperature_max,
                                cfl_max=progress_cfl_max,
                                cfl_warning=bool(progress_cfl_max > config.cfl_limit),
                            )
                        )

            if step in snapshot_steps:
                snapshots[int(step)] = temperature.detach().cpu().numpy().copy()

            if capture_final_state:
                final_clipped_mask = clipped_mask.clone()
                final_overshoot_mask = overshoot_mask.clone()
                final_undershoot_mask = undershoot_mask.clone()
                final_advection_limiter_active_mask = (
                    advection_limiter_active_mask.clone()
                )
                final_non_advective_limiter_active_mask = (
                    non_advective_limiter_active_mask.clone()
                )
                final_non_advective_limiter_scale = non_advective_limiter_scale.clone()
                final_advection_limiter_scale = advection_limiter_scale.clone()
                final_limiter_out = limiter_scale_out.clone()
                final_limiter_in = limiter_scale_in.clone()
                final_out_vol_rate = out_vol_rate_raw.clone()
                final_in_vol_rate = in_vol_rate_raw.clone()

            if log_progress:
                print(
                    "[gmsh-tetra-thermal] "
                    f"step {step}/{config.steps}, "
                    f"T_min={progress_temperature_min:.6f}, "
                    f"T_max={progress_temperature_max:.6f}, "
                    f"CFL={progress_cfl_max:.4f}"
                )
            continue

        old_integral = torch.sum(old * volumes_t)

        lower_bound = float(min_t) if boundedness_enabled else -1.0e12
        upper_bound = float(max_t) if boundedness_enabled else 1.0e12
        scale = 1.0
        offset = 0.0
        advection_state = old
        left_inlet_for_advection = float(config.left_inlet_temperature)
        right_inlet_for_advection = float(config.right_inlet_temperature)
        limiter_lower_bound = lower_bound
        limiter_upper_bound = upper_bound

        if (
            boundedness_enabled
            and config.normalize_bounded_advection
            and config.limiter_scheme == "bounded_upwind"
        ):
            span = upper_bound - lower_bound
            if span <= 0.0:
                raise ValueError("Invalid thermal bounds span for advection limiter.")
            offset = lower_bound
            scale = span
            advection_state = (old - offset) / scale
            left_inlet_for_advection = (
                float(config.left_inlet_temperature) - offset
            ) / scale
            right_inlet_for_advection = (
                float(config.right_inlet_temperature) - offset
            ) / scale
            limiter_lower_bound = 0.0
            limiter_upper_bound = 1.0

        assembled_for_limiter = assemble_advection_fluxes_torch(
            precompute,
            advection_state,
            left_inlet_value=left_inlet_for_advection,
            right_inlet_value=right_inlet_for_advection,
            dt=used_dt,
        )
        limited = apply_bounded_limiter_torch(
            precompute,
            advection_state,
            assembled_for_limiter,
            scheme=config.limiter_scheme if boundedness_enabled else "upwind",
            lower_bound=limiter_lower_bound,
            upper_bound=limiter_upper_bound,
            collect_diagnostics=bool(
                allow_expensive_step_diagnostics or capture_final_state
            ),
        )

        limited_state_for_limiter = limited["limited_state_before_clip"]
        raw_state_for_limiter = limited.get(
            "raw_state_before_limiter", limited_state_for_limiter
        )
        raw_state_physical = offset + scale * raw_state_for_limiter
        limited_state_physical = offset + scale * limited_state_for_limiter
        advected = limited_state_physical
        advected_integral = torch.sum(advected * volumes_t)
        advective_integral_change = advected_integral - old_integral

        lap = torch.zeros_like(advected)
        if config.thermal_diffusivity > 0.0:
            lap = laplacian_torch(
                precompute,
                advected,
                gradient_method=config.gradient_method,
                laplacian_method=config.laplacian_method,
            )
        diff_delta_unlimited = used_dt * float(config.thermal_diffusivity) * lap
        source_delta_unlimited = torch.full_like(advected, used_dt * source_term)
        wall_delta_unlimited = torch.zeros_like(advected)
        wall_requested_energy = zero_scalar
        if config.wall_boundary_mode != "adiabatic" and wall_boundary.faces.size:
            if config.wall_boundary_mode == "fixed_heat_flux":
                wall_face_power = float(config.wall_heat_flux) * wall_areas_t
            else:
                conductivity = (
                    float(config.thermal_diffusivity)
                    * float(config.rho)
                    * float(config.cp)
                )
                wall_face_power = (
                    conductivity
                    * wall_coefficients_t
                    * (float(config.wall_temperature) - advected[wall_owners_t])
                )
            wall_cell_power = torch.zeros_like(advected)
            wall_cell_power.scatter_add_(0, wall_owners_t, wall_face_power)
            wall_delta_unlimited = (
                used_dt
                * wall_cell_power
                / (float(config.rho) * float(config.cp) * volumes_t)
            )
            wall_requested_energy = used_dt * torch.sum(wall_face_power)
        non_advective_delta_unlimited = diff_delta_unlimited + source_delta_unlimited
        if config.wall_boundary_mode == "fixed_heat_flux":
            non_advective_delta_unlimited = (
                non_advective_delta_unlimited + wall_delta_unlimited
            )
        pre_non_advective_update = advected + non_advective_delta_unlimited

        non_advective_limiter_scale = torch.ones_like(advected)
        non_advective_limiter_active_mask = torch.zeros_like(advected, dtype=torch.bool)
        if boundedness_enabled:
            pre_non_adv_overshoot = torch.clamp(
                pre_non_advective_update - upper_bound, min=0.0
            )
            pre_non_adv_undershoot = torch.clamp(
                lower_bound - pre_non_advective_update, min=0.0
            )
            if (allow_expensive_step_diagnostics or capture_final_state) and n_cells:
                max_overshoot_before_non_advective_limiter = torch.maximum(
                    max_overshoot_before_non_advective_limiter,
                    torch.max(pre_non_adv_overshoot),
                )
                max_undershoot_before_non_advective_limiter = torch.maximum(
                    max_undershoot_before_non_advective_limiter,
                    torch.max(pre_non_adv_undershoot),
                )
            if config.limit_non_advective_update:
                positive = non_advective_delta_unlimited > 0.0
                negative = non_advective_delta_unlimited < 0.0
                allowed_pos = torch.clamp(upper_bound - advected, min=0.0)
                allowed_neg = torch.clamp(lower_bound - advected, max=0.0)
                positive_scale = allowed_pos / torch.clamp(
                    non_advective_delta_unlimited, min=1e-16
                )
                negative_scale = allowed_neg / torch.clamp(
                    non_advective_delta_unlimited, max=-1e-16
                )
                candidate_scale = torch.ones_like(non_advective_limiter_scale)
                candidate_scale = torch.where(positive, positive_scale, candidate_scale)
                candidate_scale = torch.where(negative, negative_scale, candidate_scale)
                non_advective_limiter_scale = torch.minimum(
                    non_advective_limiter_scale,
                    candidate_scale,
                )
                non_advective_limiter_scale = torch.clamp(
                    non_advective_limiter_scale, min=0.0, max=1.0
                )
                non_advective_limiter_active_mask = non_advective_limiter_scale < (
                    1.0 - 1e-12
                )
        else:
            pre_non_adv_overshoot = torch.zeros_like(advected)
            pre_non_adv_undershoot = torch.zeros_like(advected)

        diff_delta = diff_delta_unlimited * non_advective_limiter_scale
        source_delta = source_delta_unlimited * non_advective_limiter_scale
        wall_delta = wall_delta_unlimited * non_advective_limiter_scale
        wall_operator_energy = (
            float(config.rho) * float(config.cp) * torch.sum(wall_delta * volumes_t)
        )
        updated_raw = advected + diff_delta + source_delta
        if config.wall_boundary_mode == "fixed_heat_flux":
            updated_raw = updated_raw + wall_delta

        if boundedness_enabled:
            overshoot_mask = updated_raw > (upper_bound + 1e-12)
            undershoot_mask = updated_raw < (lower_bound - 1e-12)
            clipped_mask = overshoot_mask | undershoot_mask
            if allow_expensive_step_diagnostics or capture_final_state:
                overshoot = torch.clamp(updated_raw - upper_bound, min=0.0)
                undershoot = torch.clamp(lower_bound - updated_raw, min=0.0)
                if n_cells:
                    max_overshoot_before_clip = torch.maximum(
                        max_overshoot_before_clip,
                        torch.max(overshoot),
                    )
                    max_undershoot_before_clip = torch.maximum(
                        max_undershoot_before_clip,
                        torch.max(undershoot),
                    )
                raw_overshoot_physical = torch.clamp(
                    raw_state_physical - upper_bound, min=0.0
                )
                raw_undershoot_physical = torch.clamp(
                    lower_bound - raw_state_physical, min=0.0
                )
                if n_cells:
                    max_overshoot_before_limiter = torch.maximum(
                        max_overshoot_before_limiter,
                        torch.max(raw_overshoot_physical),
                    )
                    max_undershoot_before_limiter = torch.maximum(
                        max_undershoot_before_limiter,
                        torch.max(raw_undershoot_physical),
                    )
            else:
                overshoot = torch.zeros_like(updated_raw)
                undershoot = torch.zeros_like(updated_raw)
        else:
            overshoot = torch.zeros_like(updated_raw)
            undershoot = torch.zeros_like(updated_raw)
            overshoot_mask = torch.zeros_like(updated_raw, dtype=torch.bool)
            undershoot_mask = torch.zeros_like(updated_raw, dtype=torch.bool)
            clipped_mask = torch.zeros_like(updated_raw, dtype=torch.bool)

        if boundedness_enabled and config.clipping_enabled:
            updated = torch.clamp(updated_raw, min=lower_bound, max=upper_bound)
            updated_without_wall = torch.clamp(
                updated_raw - wall_delta, min=lower_bound, max=upper_bound
            )
        else:
            updated = updated_raw
            updated_without_wall = updated_raw - wall_delta
        wall_applied_energy = (
            float(config.rho)
            * float(config.cp)
            * torch.sum((updated - updated_without_wall) * volumes_t)
        )

        if not bool(torch.all(torch.isfinite(updated)).item()):
            raise FloatingPointError(f"Temperature became non-finite at step {step}.")

        raw_inlet_scalar_flux_in = (
            float(offset) * inlet_vol_flux_in_t
            + float(scale) * assembled_for_limiter["inlet_scalar_flux_in"]
        )
        raw_outlet_scalar_flux_out = (
            float(offset) * outlet_vol_flux_out_t
            + float(scale) * assembled_for_limiter["outlet_scalar_flux_out"]
        )
        adv_mass_in = used_dt * raw_inlet_scalar_flux_in
        adv_mass_out = used_dt * raw_outlet_scalar_flux_out
        adv_net = adv_mass_in - adv_mass_out
        diffusion_integral_change = torch.sum(diff_delta * volumes_t)
        source_integral_change = torch.sum(source_delta * volumes_t)
        diffusion_integral_change_unlimited = torch.sum(
            diff_delta_unlimited * volumes_t
        )
        source_integral_change_unlimited = torch.sum(source_delta_unlimited * volumes_t)

        clamp_delta = updated - updated_raw
        clamp_integral_change = torch.sum(clamp_delta * volumes_t)
        applied_non_advective_delta = diff_delta + source_delta
        if config.wall_boundary_mode == "fixed_heat_flux":
            applied_non_advective_delta = applied_non_advective_delta + wall_delta
        non_advective_limiter_delta = (
            non_advective_delta_unlimited - applied_non_advective_delta
        )
        non_advective_limiter_integral_change = torch.sum(
            non_advective_limiter_delta * volumes_t
        )

        cumulative_adv_in += adv_mass_in
        cumulative_adv_out += adv_mass_out
        cumulative_advective_integral_change += advective_integral_change
        cumulative_source += source_integral_change
        cumulative_diffusion += diffusion_integral_change
        cumulative_source_unlimited += source_integral_change_unlimited
        cumulative_diffusion_unlimited += diffusion_integral_change_unlimited
        cumulative_clamp += clamp_integral_change
        cumulative_non_advective_limiter += non_advective_limiter_integral_change
        cumulative_wall_requested_energy += wall_requested_energy
        cumulative_wall_operator_energy += wall_operator_energy
        cumulative_wall_applied_energy += wall_applied_energy
        final_wall_requested_energy = wall_requested_energy
        final_wall_applied_energy = wall_applied_energy

        temperature = updated
        limiter_scale_out = limited.get(
            "limiter_scale_out", torch.ones_like(temperature)
        )
        limiter_scale_in = limited.get("limiter_scale_in", torch.ones_like(temperature))
        advection_limiter_scale = torch.minimum(limiter_scale_out, limiter_scale_in)
        advection_limiter_active_mask = advection_limiter_scale < (1.0 - 1e-12)
        advection_delta = advected - old
        out_vol_rate_raw = assembled_for_limiter["out_vol_rate_raw"]
        in_vol_rate_raw = assembled_for_limiter["in_vol_rate_raw"]
        progress_temperature_min = 0.0
        progress_temperature_max = 0.0
        progress_cfl_max = 0.0
        if record_history or log_progress:
            local_outgoing_cfl = used_dt * out_vol_rate_raw / volumes_t
            local_incoming_cfl = used_dt * in_vol_rate_raw / volumes_t
            step_scalar_metrics: dict[str, object] = {
                "temperature_min": torch.min(temperature) if n_cells else zero_scalar,
                "temperature_max": torch.max(temperature) if n_cells else zero_scalar,
                "cfl_max": torch.max(local_outgoing_cfl) if n_cells else zero_scalar,
            }

            if (
                record_history
                and resolved_history_mode == "dict"
                and allow_expensive_step_diagnostics
            ):
                raw_integral = torch.sum(updated_raw * volumes_t)
                expected_raw_integral = (
                    old_integral
                    + advective_integral_change
                    + diffusion_integral_change
                    + source_integral_change
                    + (
                        torch.sum(wall_delta * volumes_t)
                        if config.wall_boundary_mode == "fixed_heat_flux"
                        else zero_scalar
                    )
                )
                raw_balance_error = raw_integral - expected_raw_integral
                new_integral = torch.sum(updated * volumes_t)
                step_scalar_metrics.update(
                    {
                        "temperature_mean": (
                            torch.mean(temperature) if n_cells else zero_scalar
                        ),
                        "max_change": (
                            torch.max(torch.abs(temperature - old))
                            if n_cells
                            else zero_scalar
                        ),
                        "raw_balance_error": raw_balance_error,
                        "energy_proxy": new_integral,
                        "cfl_incoming_max": (
                            torch.max(local_incoming_cfl) if n_cells else zero_scalar
                        ),
                        "clipped_cell_count": torch.count_nonzero(clipped_mask),
                        "limiter_active_cell_count": torch.count_nonzero(
                            advection_limiter_active_mask
                        ),
                        "non_advective_limiter_active_cell_count": torch.count_nonzero(
                            non_advective_limiter_active_mask
                        ),
                        "advective_integral_change": advective_integral_change,
                        "diffusion_integral_change": diffusion_integral_change,
                        "source_integral_change": source_integral_change,
                        "non_advective_limiter_integral_change": non_advective_limiter_integral_change,
                        "clamp_integral_change": clamp_integral_change,
                    }
                )
                if config.full_step_diagnostics and record_diagnostics:
                    step_scalar_metrics.update(
                        {
                            "advective_mass_in": adv_mass_in,
                            "advective_mass_out": adv_mass_out,
                            "advective_net": adv_net,
                            "diffusion_integral_change_unlimited": diffusion_integral_change_unlimited,
                            "source_integral_change_unlimited": source_integral_change_unlimited,
                            "advection_delta_max_abs": (
                                torch.max(torch.abs(advection_delta))
                                if n_cells
                                else zero_scalar
                            ),
                            "diffusion_delta_max_abs": (
                                torch.max(torch.abs(diff_delta))
                                if n_cells
                                else zero_scalar
                            ),
                            "source_delta_max_abs": (
                                torch.max(torch.abs(source_delta))
                                if n_cells
                                else zero_scalar
                            ),
                            "clamp_delta_max_abs": (
                                torch.max(torch.abs(clamp_delta))
                                if n_cells
                                else zero_scalar
                            ),
                            "non_advective_limiter_scale_min": (
                                torch.min(non_advective_limiter_scale)
                                if n_cells
                                else zero_scalar
                            ),
                            "overshoot_before_non_advective_limiter": (
                                torch.max(pre_non_adv_overshoot)
                                if boundedness_enabled and n_cells
                                else zero_scalar
                            ),
                            "undershoot_before_non_advective_limiter": (
                                torch.max(pre_non_adv_undershoot)
                                if boundedness_enabled and n_cells
                                else zero_scalar
                            ),
                            "diffusion_laplacian_max_abs": (
                                torch.max(torch.abs(lap)) if n_cells else zero_scalar
                            ),
                            "overshoot_before_clip": (
                                torch.max(overshoot) if n_cells else zero_scalar
                            ),
                            "undershoot_before_clip": (
                                torch.max(undershoot) if n_cells else zero_scalar
                            ),
                            "overshoot_cell_count": torch.count_nonzero(overshoot_mask),
                            "undershoot_cell_count": torch.count_nonzero(
                                undershoot_mask
                            ),
                            "cumulative_advective_mass_in": cumulative_adv_in,
                            "cumulative_advective_mass_out": cumulative_adv_out,
                            "cumulative_source": cumulative_source,
                            "cumulative_source_unlimited": cumulative_source_unlimited,
                            "cumulative_diffusion": cumulative_diffusion,
                            "cumulative_diffusion_unlimited": cumulative_diffusion_unlimited,
                            "cumulative_non_advective_limiter": cumulative_non_advective_limiter,
                            "cumulative_clamp": cumulative_clamp,
                            "limiter_max_flux_scale_down": limited.get(
                                "limiter_max_flux_scale_down", zero_scalar
                            ),
                            "conservation_error_after_limiter": float(scale)
                            * limited.get(
                                "conservation_error_after_limiter", zero_scalar
                            ),
                            "limiter_mass_correction_total": float(scale)
                            * limited.get("limiter_mass_correction_total", zero_scalar),
                        }
                    )
            step_scalar_values = _materialize_torch_scalar_batch(
                torch,
                step_scalar_metrics,
                device=device,
            )
            progress_temperature_min = float(step_scalar_values["temperature_min"])
            progress_temperature_max = float(step_scalar_values["temperature_max"])
            progress_cfl_max = float(step_scalar_values["cfl_max"])

            if record_history:
                if resolved_history_mode == "compact":
                    _append_compact_thermal_history_sample(
                        compact_history,
                        step=step,
                        dt_used=float(used_dt),
                        temperature_min=progress_temperature_min,
                        temperature_max=progress_temperature_max,
                        cfl_max=progress_cfl_max,
                    )
                elif allow_expensive_step_diagnostics:
                    temperature_mean = float(step_scalar_values["temperature_mean"])
                    max_change = float(step_scalar_values["max_change"])
                    cfl_incoming_max = float(step_scalar_values["cfl_incoming_max"])
                    cfl_warning_step = bool(progress_cfl_max > config.cfl_limit)
                    clipped_cell_count = int(step_scalar_values["clipped_cell_count"])
                    limiter_active_cell_count = int(
                        step_scalar_values["limiter_active_cell_count"]
                    )
                    non_advective_limiter_active_cell_count = int(
                        step_scalar_values["non_advective_limiter_active_cell_count"]
                    )
                    entry = _build_compact_thermal_history_entry(
                        step=step,
                        used_dt=float(used_dt),
                        temperature_min=progress_temperature_min,
                        temperature_max=progress_temperature_max,
                        temperature_mean=temperature_mean,
                        max_change=max_change,
                        advective_integral_change=float(
                            step_scalar_values["advective_integral_change"]
                        ),
                        diffusion_integral_change=float(
                            step_scalar_values["diffusion_integral_change"]
                        ),
                        source_integral_change=float(
                            step_scalar_values["source_integral_change"]
                        ),
                        non_advective_limiter_integral_change=float(
                            step_scalar_values["non_advective_limiter_integral_change"]
                        ),
                        clamp_integral_change=float(
                            step_scalar_values["clamp_integral_change"]
                        ),
                        raw_balance_error=float(
                            step_scalar_values["raw_balance_error"]
                        ),
                        energy_proxy=float(step_scalar_values["energy_proxy"]),
                        cfl_max=progress_cfl_max,
                        cfl_incoming_max=cfl_incoming_max,
                        cfl_warning=cfl_warning_step,
                        clipped_cell_count=clipped_cell_count,
                        limiter_active_cell_count=limiter_active_cell_count,
                        non_advective_limiter_active_cell_count=non_advective_limiter_active_cell_count,
                    )
                    if config.full_step_diagnostics and record_diagnostics:
                        entry.update(
                            {
                                "advective_mass_in": float(
                                    step_scalar_values["advective_mass_in"]
                                ),
                                "advective_mass_out": float(
                                    step_scalar_values["advective_mass_out"]
                                ),
                                "advective_net": float(
                                    step_scalar_values["advective_net"]
                                ),
                                "diffusion_integral_change_unlimited": float(
                                    step_scalar_values[
                                        "diffusion_integral_change_unlimited"
                                    ]
                                ),
                                "source_integral_change_unlimited": float(
                                    step_scalar_values[
                                        "source_integral_change_unlimited"
                                    ]
                                ),
                                "advection_delta_max_abs": float(
                                    step_scalar_values["advection_delta_max_abs"]
                                ),
                                "diffusion_delta_max_abs": float(
                                    step_scalar_values["diffusion_delta_max_abs"]
                                ),
                                "source_delta_max_abs": float(
                                    step_scalar_values["source_delta_max_abs"]
                                ),
                                "clamp_delta_max_abs": float(
                                    step_scalar_values["clamp_delta_max_abs"]
                                ),
                                "non_advective_limiter_scale_min": float(
                                    step_scalar_values[
                                        "non_advective_limiter_scale_min"
                                    ]
                                ),
                                "non_advective_limiter_active_fraction": float(
                                    non_advective_limiter_active_cell_count
                                    / max(
                                        1,
                                        int(non_advective_limiter_active_mask.numel()),
                                    )
                                ),
                                "overshoot_before_non_advective_limiter": float(
                                    step_scalar_values[
                                        "overshoot_before_non_advective_limiter"
                                    ]
                                )
                                if boundedness_enabled and n_cells
                                else 0.0,
                                "undershoot_before_non_advective_limiter": float(
                                    step_scalar_values[
                                        "undershoot_before_non_advective_limiter"
                                    ]
                                )
                                if boundedness_enabled and n_cells
                                else 0.0,
                                "diffusion_laplacian_max_abs": float(
                                    step_scalar_values["diffusion_laplacian_max_abs"]
                                )
                                if n_cells
                                else 0.0,
                                "overshoot_before_clip": float(
                                    step_scalar_values["overshoot_before_clip"]
                                )
                                if n_cells
                                else 0.0,
                                "undershoot_before_clip": float(
                                    step_scalar_values["undershoot_before_clip"]
                                )
                                if n_cells
                                else 0.0,
                                "overshoot_cell_count": int(
                                    step_scalar_values["overshoot_cell_count"]
                                ),
                                "undershoot_cell_count": int(
                                    step_scalar_values["undershoot_cell_count"]
                                ),
                                "limiter_active_fraction": float(
                                    limiter_active_cell_count
                                    / max(1, int(advection_limiter_active_mask.numel()))
                                ),
                                "limiter_max_flux_scale_down": float(
                                    step_scalar_values["limiter_max_flux_scale_down"]
                                ),
                                "conservation_error_after_limiter": float(
                                    step_scalar_values[
                                        "conservation_error_after_limiter"
                                    ]
                                ),
                                "limiter_mass_correction_total": float(
                                    step_scalar_values["limiter_mass_correction_total"]
                                ),
                                "cumulative_advective_mass_in": float(
                                    step_scalar_values["cumulative_advective_mass_in"]
                                ),
                                "cumulative_advective_mass_out": float(
                                    step_scalar_values["cumulative_advective_mass_out"]
                                ),
                                "cumulative_source": float(
                                    step_scalar_values["cumulative_source"]
                                ),
                                "cumulative_source_unlimited": float(
                                    step_scalar_values["cumulative_source_unlimited"]
                                ),
                                "cumulative_diffusion": float(
                                    step_scalar_values["cumulative_diffusion"]
                                ),
                                "cumulative_diffusion_unlimited": float(
                                    step_scalar_values["cumulative_diffusion_unlimited"]
                                ),
                                "cumulative_non_advective_limiter": float(
                                    step_scalar_values[
                                        "cumulative_non_advective_limiter"
                                    ]
                                ),
                                "cumulative_clamp": float(
                                    step_scalar_values["cumulative_clamp"]
                                ),
                            }
                        )
                    history.append(entry)
                else:
                    history.append(
                        _build_fast_thermal_history_entry(
                            step=step,
                            used_dt=float(used_dt),
                            temperature_min=progress_temperature_min,
                            temperature_max=progress_temperature_max,
                            cfl_max=progress_cfl_max,
                            cfl_warning=bool(progress_cfl_max > config.cfl_limit),
                        )
                    )

        if step in snapshot_steps:
            snapshots[int(step)] = temperature.detach().cpu().numpy().copy()

        if config.debug_artifacts or step == config.steps:
            final_lap = lap.clone()
            final_raw_state = raw_state_physical.clone()
            final_limited_state = limited_state_physical.clone()
            final_updated_before_non_advective_limiter = (
                pre_non_advective_update.clone()
            )
            final_updated_before_clip = updated_raw.clone()
            final_clipped_mask = clipped_mask.clone()
            final_overshoot_mask = overshoot_mask.clone()
            final_undershoot_mask = undershoot_mask.clone()
            final_advection_limiter_active_mask = advection_limiter_active_mask.clone()
            final_non_advective_limiter_active_mask = (
                non_advective_limiter_active_mask.clone()
            )
            final_non_advective_limiter_scale = non_advective_limiter_scale.clone()
            final_advection_limiter_scale = advection_limiter_scale.clone()
            final_advection_delta = advection_delta.clone()
            final_diffusion_delta = diff_delta.clone()
            final_source_delta = source_delta.clone()
            final_clamp_delta = clamp_delta.clone()
            if "face_scalar_flux_raw" in limited:
                final_face_flux_raw = limited["face_scalar_flux_raw"].clone() * float(
                    scale
                )
                final_face_flux_limited = limited[
                    "face_scalar_flux_limited"
                ].clone() * float(scale)
                final_in_mass_raw = limited["in_mass_raw"].clone() * float(scale)
                final_out_mass_raw = limited["out_mass_raw"].clone() * float(scale)
                final_in_mass_lim = limited["in_mass_limited"].clone() * float(scale)
                final_out_mass_lim = limited["out_mass_limited"].clone() * float(scale)
            final_limiter_out = limiter_scale_out.clone()
            final_limiter_in = limiter_scale_in.clone()
            final_out_vol_rate = out_vol_rate_raw.clone()
            final_in_vol_rate = in_vol_rate_raw.clone()

        if log_progress:
            print(
                "[gmsh-tetra-thermal] "
                f"step {step}/{config.steps}, "
                f"T_min={progress_temperature_min:.6f}, "
                f"T_max={progress_temperature_max:.6f}, "
                f"CFL={progress_cfl_max:.4f}"
            )

    final_local_outgoing_cfl = used_dt * final_out_vol_rate / volumes_t
    final_local_incoming_cfl = used_dt * final_in_vol_rate / volumes_t
    final_scalar_values = _materialize_torch_scalar_batch(
        torch,
        {
            "final_energy": torch.sum(temperature * volumes_t),
            "max_overshoot_before_clip": max_overshoot_before_clip,
            "max_undershoot_before_clip": max_undershoot_before_clip,
            "max_overshoot_before_limiter": max_overshoot_before_limiter,
            "max_undershoot_before_limiter": max_undershoot_before_limiter,
            "max_overshoot_before_non_advective_limiter": max_overshoot_before_non_advective_limiter,
            "max_undershoot_before_non_advective_limiter": max_undershoot_before_non_advective_limiter,
        },
        device=device,
    )
    final_energy = float(final_scalar_values["final_energy"])
    max_overshoot_before_clip_value = float(
        final_scalar_values["max_overshoot_before_clip"]
    )
    max_undershoot_before_clip_value = float(
        final_scalar_values["max_undershoot_before_clip"]
    )
    max_overshoot_before_limiter_value = float(
        final_scalar_values["max_overshoot_before_limiter"]
    )
    max_undershoot_before_limiter_value = float(
        final_scalar_values["max_undershoot_before_limiter"]
    )
    max_overshoot_before_non_advective_limiter_value = float(
        final_scalar_values["max_overshoot_before_non_advective_limiter"]
    )
    max_undershoot_before_non_advective_limiter_value = float(
        final_scalar_values["max_undershoot_before_non_advective_limiter"]
    )
    clipping_used = bool(
        config.clipping_enabled
        and boundedness_enabled
        and (
            max_overshoot_before_clip_value > 0.0
            or max_undershoot_before_clip_value > 0.0
        )
    )
    wall_integral_change = float(
        cumulative_wall_applied_energy.detach().cpu().item()
    ) / (float(config.rho) * float(config.cp))
    diffusion_non_wall_integral_change = float(
        cumulative_diffusion.detach().cpu().item()
    ) - (
        wall_integral_change
        if config.wall_boundary_mode == "fixed_temperature"
        else 0.0
    )
    expected_energy_no_clamp = (
        float(initial_energy.detach().cpu().item())
        + float(cumulative_advective_integral_change.detach().cpu().item())
        + diffusion_non_wall_integral_change
        + float(cumulative_source.detach().cpu().item())
        + wall_integral_change
    )

    temperature_np = temperature.detach().cpu().numpy()
    source = flux_diagnostics if flux_diagnostics is not None else {}
    velocity_meta = velocity_metadata if velocity_metadata is not None else {}
    execution_policy_payload = execution_policy if execution_policy is not None else {}
    recorded_steps = (
        [int(step) for step in compact_history.get("step", [])]
        if resolved_history_mode == "compact"
        else [int(entry["step"]) for entry in history]
    )
    history_settings = {
        "collect_history": bool(config.collect_history),
        "diagnostics_mode": str(config.diagnostics_mode),
        "resolved_diagnostics_mode": str(resolved_diagnostics_mode),
        "history_mode": str(config.history_mode),
        "resolved_history_mode": str(resolved_history_mode),
        "history_stride": int(config.history_stride),
        "diagnostics_stride": int(config.diagnostics_stride),
        "full_step_diagnostics": bool(config.full_step_diagnostics),
        "debug_artifacts": bool(config.debug_artifacts),
        "torch_compile_step_body": str(config.torch_compile_step_body),
        "resolved_torch_compile_step_body": str(resolved_torch_compile_mode),
        **torch_compile_step_metadata,
        "execution_mode": _thermal_execution_mode(config),
        "recorded_steps": recorded_steps,
    }

    elapsed_time = float(config.steps) * float(used_dt)
    cumulative_wall_requested_energy_value = float(
        cumulative_wall_requested_energy.detach().cpu().item()
    )
    cumulative_wall_operator_energy_value = float(
        cumulative_wall_operator_energy.detach().cpu().item()
    )
    cumulative_wall_applied_energy_value = float(
        cumulative_wall_applied_energy.detach().cpu().item()
    )
    wall_heat_transfer = _build_wall_heat_transfer(
        config,
        wall_boundary,
        final_requested_energy=float(final_wall_requested_energy.detach().cpu().item()),
        final_applied_energy=float(final_wall_applied_energy.detach().cpu().item()),
        cumulative_requested_energy=cumulative_wall_requested_energy_value,
        cumulative_energy_after_limiter_before_clipping=(
            cumulative_wall_operator_energy_value
        ),
        cumulative_applied_energy=cumulative_wall_applied_energy_value,
        elapsed_time=elapsed_time,
    )
    initial_energy_value = float(initial_energy.detach().cpu().item())
    global_energy_balance = _build_global_energy_balance(
        config,
        initial_integral_t_dv=initial_energy_value,
        final_integral_t_dv=final_energy,
        cumulative_advective_in=float(cumulative_adv_in.detach().cpu().item()),
        cumulative_advective_out=float(cumulative_adv_out.detach().cpu().item()),
        cumulative_advective_integral_change=float(
            cumulative_advective_integral_change.detach().cpu().item()
        ),
        cumulative_diffusion_integral_change=float(
            cumulative_diffusion.detach().cpu().item()
        ),
        cumulative_source_integral_change=float(
            cumulative_source.detach().cpu().item()
        ),
        cumulative_clamp_integral_change=float(cumulative_clamp.detach().cpu().item()),
        cumulative_wall_operator_energy_j=cumulative_wall_operator_energy_value,
        cumulative_wall_applied_energy_j=cumulative_wall_applied_energy_value,
    )

    return {
        "temperature": temperature_np,
        "history": history,
        "history_compact": compact_history,
        "history_settings": history_settings,
        "execution_policy": execution_policy_payload,
        "model_capabilities": _thermal_v2_model_capabilities(),
        "thermal_observables": _build_thermal_observables(
            mesh,
            temperature_np,
            np.asarray(face_normal_velocity, dtype=np.float64),
        ),
        "wall_heat_transfer": wall_heat_transfer,
        "global_energy_balance": global_energy_balance,
        "cfl_values": cfl_values_initial,
        "cfl_max": float(cfl_max_initial),
        "cfl_warning": bool(cfl_warning),
        "dt_control": {
            "dt_mode": config.dt_mode,
            "requested_dt": float(requested_dt),
            "used_dt": float(used_dt),
            "advection_dt_limit": float(advection_dt_limit),
            "diffusion_dt_limit": float(diffusion_dt_limit),
            "cfl_target": float(config.cfl_target),
            "diffusion_stability_factor": float(config.diffusion_stability_factor),
            "cfl_limit": float(config.cfl_limit),
            "diffusion_stability_warning": bool(diffusion_stability_warning),
        },
        "boundary_setup": {
            "left_inlet_faces": int(left_faces.size),
            "right_inlet_faces": int(right_faces.size),
            "outlet_faces": int(outlet_faces.size),
            "wall_faces": int(wall_faces.size),
            "heated_wall_faces": int(wall_boundary.faces.size),
            "heated_wall_area_m2": float(wall_boundary.area),
            "heated_wall_group_face_counts": dict(wall_boundary.group_face_counts),
            "left_inlet_cells": int(left_cells.size),
            "right_inlet_cells": int(right_cells.size),
            "diffusion_bc": {
                "dirichlet_faces": int(len(precompute.boundary_dirichlet)),
                "no_flux_faces": int(precompute.boundary_no_flux_faces.size),
                "outlet_mode": "advective_outflow_with_backflow_suppression_and_zero_diffusive_flux",
            },
        },
        "transport_info": {
            "equation": "dT/dt + u.grad(T) = alpha*laplacian(T) + Q/(rho*cp)",
            "velocity_source": str(velocity_meta.get("velocity_source", "unspecified")),
            "flow_solved": bool(velocity_meta.get("flow_solved", False)),
            "pressure_solved": bool(velocity_meta.get("pressure_solved", False)),
            "execution_profile": str(
                execution_policy_payload.get("execution_profile", "")
            ),
            "production_backend_satisfied": bool(
                execution_policy_payload.get("production_backend_satisfied", False)
            ),
            "advection_scheme": config.limiter_scheme,
            "normalize_bounded_advection": bool(config.normalize_bounded_advection),
            "limit_non_advective_update": bool(config.limit_non_advective_update),
            "gradient_method": config.gradient_method,
            "laplacian_method": config.laplacian_method,
            "inlet_temperature_mode": "face_dirichlet_for_inflow",
            "wall_temperature_mode": str(config.wall_boundary_mode),
            "outlet_temperature_mode": "advective_outflow_with_backflow_suppression_and_zero_diffusive_flux",
            "boundedness_enabled": bool(boundedness_enabled),
            "diffusion_backend": "torch"
            if config.thermal_diffusivity > 0.0
            else "none",
        },
        "performance_diagnostics": {
            "advection_flux_assemblies_per_step": 1,
            "normalized_bounded_advection_avoids_reassembly": True,
            "history_entries_recorded": int(len(recorded_steps)),
            "execution_mode": _thermal_execution_mode(config),
            "resolved_diagnostics_mode": str(resolved_diagnostics_mode),
            "resolved_torch_compile_step_body": str(resolved_torch_compile_mode),
            **torch_compile_step_metadata,
            "degraded_execution_mode": bool(
                execution_policy_payload.get("degraded_execution_mode", False)
            ),
        },
        "flux_diagnostics": source,
        "clipping": {
            "enabled": bool(config.clipping_enabled and boundedness_enabled),
            "used": bool(clipping_used),
            "max_overshoot_before_clip": float(max_overshoot_before_clip_value),
            "max_undershoot_before_clip": float(max_undershoot_before_clip_value),
            "max_overshoot_before_limiter": float(max_overshoot_before_limiter_value),
            "max_undershoot_before_limiter": float(max_undershoot_before_limiter_value),
            "max_overshoot_before_non_advective_limiter": float(
                max_overshoot_before_non_advective_limiter_value
            ),
            "max_undershoot_before_non_advective_limiter": float(
                max_undershoot_before_non_advective_limiter_value
            ),
            "final_clipped_cell_count": int(
                torch.count_nonzero(final_clipped_mask).item()
            ),
            "final_overshoot_cell_count": int(
                torch.count_nonzero(final_overshoot_mask).item()
            ),
            "final_undershoot_cell_count": int(
                torch.count_nonzero(final_undershoot_mask).item()
            ),
        },
        "limiter": {
            "limiter_active_cell_count": int(
                torch.count_nonzero(final_advection_limiter_active_mask).item()
            ),
            "limiter_active_fraction": float(
                torch.count_nonzero(final_advection_limiter_active_mask).item()
                / max(1, temperature_np.size)
            ),
            "limiter_max_flux_scale_down": float(
                1.0
                - min(
                    float(torch.min(final_limiter_out).item()),
                    float(torch.min(final_limiter_in).item()),
                )
            ),
            "limiter_mass_correction_total": float(
                torch.sum(
                    torch.abs(
                        (final_in_mass_raw - final_out_mass_raw)
                        - (final_in_mass_lim - final_out_mass_lim)
                    )
                ).item()
            ),
            "conservation_error_after_limiter": float(
                (
                    torch.sum(final_in_mass_lim)
                    - torch.sum(final_out_mass_lim)
                    - (torch.sum(final_in_mass_raw) - torch.sum(final_out_mass_raw))
                ).item()
            ),
        },
        "non_advective_limiter": {
            "enabled": bool(boundedness_enabled and config.limit_non_advective_update),
            "final_active_cell_count": int(
                torch.count_nonzero(final_non_advective_limiter_active_mask).item()
            ),
            "final_active_fraction": float(
                torch.count_nonzero(final_non_advective_limiter_active_mask).item()
                / max(1, temperature_np.size)
            ),
            "final_scale_min": float(
                torch.min(final_non_advective_limiter_scale).item()
            ),
            "cumulative_integral_change": float(cumulative_non_advective_limiter),
        },
        "energy_proxy": {
            "initial_integral_T_dV": float(initial_energy),
            "final_integral_T_dV": float(final_energy),
            "cumulative_advective_in": float(cumulative_adv_in),
            "cumulative_advective_out": float(cumulative_adv_out),
            "cumulative_advective_integral_change": float(
                cumulative_advective_integral_change
            ),
            "cumulative_diffusion_integral_change": float(cumulative_diffusion),
            "cumulative_non_wall_diffusion_integral_change": float(
                diffusion_non_wall_integral_change
            ),
            "cumulative_wall_integral_change": float(wall_integral_change),
            "cumulative_wall_energy_j": float(
                cumulative_wall_applied_energy.detach().cpu().item()
            ),
            "cumulative_diffusion_integral_change_unlimited": float(
                cumulative_diffusion_unlimited
            ),
            "cumulative_source_integral_change": float(cumulative_source),
            "cumulative_source_integral_change_unlimited": float(
                cumulative_source_unlimited
            ),
            "cumulative_non_advective_limiter_integral_change": float(
                cumulative_non_advective_limiter
            ),
            "cumulative_clamp_integral_change": float(cumulative_clamp),
            "expected_final_without_clamp": float(expected_energy_no_clamp),
            "balance_error_without_clamp": float(
                final_energy - cumulative_clamp - expected_energy_no_clamp
            ),
            "balance_error_with_clamp": float(
                final_energy - (expected_energy_no_clamp + cumulative_clamp)
            ),
        },
        "local_cfl_audit": {
            "max_local_outgoing_cfl": float(torch.max(final_local_outgoing_cfl).item())
            if n_cells
            else 0.0,
            "max_local_incoming_cfl": float(torch.max(final_local_incoming_cfl).item())
            if n_cells
            else 0.0,
            "top_cells_by_local_outgoing_cfl": _top_cell_metrics(
                final_local_outgoing_cfl.detach().cpu().numpy(), top_k=10
            ),
            "top_cells_by_local_incoming_cfl": _top_cell_metrics(
                final_local_incoming_cfl.detach().cpu().numpy(), top_k=10
            ),
        },
        "final_step_debug": (
            {
                "raw_state_before_limiter": final_raw_state.detach().cpu().numpy(),
                "limited_state_before_clip": final_limited_state.detach().cpu().numpy(),
                "updated_raw_before_non_advective_limiter": final_updated_before_non_advective_limiter.detach()
                .cpu()
                .numpy(),
                "updated_raw_before_clip": final_updated_before_clip.detach()
                .cpu()
                .numpy(),
                "clipped_mask": final_clipped_mask.detach().cpu().numpy(),
                "overshoot_mask": final_overshoot_mask.detach().cpu().numpy(),
                "undershoot_mask": final_undershoot_mask.detach().cpu().numpy(),
                "advection_limiter_active_mask": final_advection_limiter_active_mask.detach()
                .cpu()
                .numpy(),
                "non_advective_limiter_active_mask": final_non_advective_limiter_active_mask.detach()
                .cpu()
                .numpy(),
                "advection_limiter_scale": final_advection_limiter_scale.detach()
                .cpu()
                .numpy(),
                "non_advective_limiter_scale": final_non_advective_limiter_scale.detach()
                .cpu()
                .numpy(),
                "advection_delta": final_advection_delta.detach().cpu().numpy(),
                "diffusion_delta": final_diffusion_delta.detach().cpu().numpy(),
                "source_delta": final_source_delta.detach().cpu().numpy(),
                "clamp_delta": final_clamp_delta.detach().cpu().numpy(),
                "face_scalar_flux_raw": final_face_flux_raw.detach().cpu().numpy(),
                "face_scalar_flux_limited": final_face_flux_limited.detach()
                .cpu()
                .numpy(),
                "laplacian": final_lap.detach().cpu().numpy(),
                "left_inlet_cells": left_cells,
                "right_inlet_cells": right_cells,
                "boundary_face_groups": boundary_face_groups,
            }
            if config.debug_artifacts
            else {}
        ),
        "diffusion_diagnostics": {
            "diffusivity": float(config.thermal_diffusivity),
            "diffusion_backend": "torch"
            if config.thermal_diffusivity > 0.0
            else "none",
            "laplacian_method": config.laplacian_method,
            "gradient_method": config.gradient_method,
            "diffusion_dt_limit": float(diffusion_dt_limit),
            "diffusion_stability_warning": bool(diffusion_stability_warning),
            "max_laplacian_abs_final_step": float(
                torch.max(torch.abs(final_lap)).item()
            )
            if n_cells
            else 0.0,
        },
        "final_stats": {
            "min": float(np.min(temperature_np)),
            "max": float(np.max(temperature_np)),
            "mean": float(np.mean(temperature_np)),
        },
        "backend_execution": build_backend_execution_diagnostics(
            requested_backend=str(config.backend),
            stepping_backend="torch",
            device=str(device),
            torch_device=str(config.torch_device),
            used_numpy_fallback=bool(backend_selection.used_numpy_fallback),
            all_core_arrays_on_cuda=bool(getattr(device, "type", "cpu") == "cuda"),
            per_step_large_cpu_gpu_transfer_warning=False,
            cpu_gpu_transfer_notes=[],
        ),
        "snapshots": snapshots,
    }


def run_tetra_thermal_debug(
    mesh: ImportedTetraMesh,
    config: GmshTetraThermalConfig,
    *,
    face_normal_velocity: np.ndarray,
    flux_diagnostics: Dict[str, float] | None = None,
    velocity_metadata: Dict[str, object] | None = None,
    diffusion_boundary_kwargs: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    _validate_thermal_config(config)
    wall_boundary = _resolve_wall_boundary(mesh, config)
    backend_selection = select_backend(config.backend)
    execution_policy = _resolve_thermal_execution_policy(config, backend_selection)
    stepping_backend = backend_selection.selected_backend
    precompute_boundary_kwargs = dict(diffusion_boundary_kwargs or {})
    if config.wall_boundary_mode == "fixed_temperature":
        configured_dirichlet = precompute_boundary_kwargs.get(
            "diffusion_boundary_dirichlet", {}
        )
        if not isinstance(configured_dirichlet, Mapping):
            raise TypeError("diffusion_boundary_dirichlet must be a mapping")
        merged_dirichlet = {
            int(face_idx): float(value)
            for face_idx, value in configured_dirichlet.items()
        }
        merged_dirichlet.update(
            {
                int(face_idx): float(config.wall_temperature)
                for face_idx in wall_boundary.faces.tolist()
            }
        )
        precompute_boundary_kwargs["diffusion_boundary_dirichlet"] = merged_dirichlet
    precompute = build_scalar_backend_precompute(
        mesh,
        face_normal_velocity,
        backend=stepping_backend,
        torch_device=config.torch_device,
        diffusion_dirichlet_left_value=float(config.left_inlet_temperature),
        diffusion_dirichlet_right_value=float(config.right_inlet_temperature),
        **precompute_boundary_kwargs,
    )
    if stepping_backend == "torch":
        return _run_tetra_thermal_debug_torch(
            mesh,
            config,
            precompute=precompute,
            backend_selection=backend_selection,
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diagnostics,
            velocity_metadata=velocity_metadata,
            execution_policy=execution_policy,
            wall_boundary=wall_boundary,
        )
    return _run_tetra_thermal_debug_numpy(
        mesh,
        config,
        precompute=precompute,
        backend_selection=backend_selection,
        face_normal_velocity=face_normal_velocity,
        flux_diagnostics=flux_diagnostics,
        velocity_metadata=velocity_metadata,
        execution_policy=execution_policy,
        wall_boundary=wall_boundary,
    )
