"""Minimal tetra-native pressure-projection sanity solver.

This module implements a narrow projection-only flow debug path:
- primary mass state: face flux
- pressure: cell-centered
- velocity: derived in the legacy path, with an opt-in momentum-state update
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal
import warnings
import weakref

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.tetra.gmsh_tetra_backend import (
    BackendSelection,
    select_backend,
)
from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
    resolve_inlet_face_groups,
)

_GEOMETRY_CACHE: dict[tuple[Any, ...], Any] = {}
_AMG_PRESSURE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TORCH_PRESSURE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TORCH_CONVECTION_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TORCH_VISCOSITY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_TORCH_DYNAMIC_ARRAY_CACHE: dict[
    tuple[int, str], tuple[weakref.ReferenceType[np.ndarray], Any]
] = {}
_ACTIVE_TORCH_CACHE_CONTEXT: tuple[tuple[int, ...], str] | None = None

FlowBackend = Literal["auto", "numpy", "torch"]
PressureSolver = Literal["jacobi", "cg", "pcg_diag", "amg_pcg"]
ProjectionSign = Literal["minus", "plus"]
ProjectionRhsMode = Literal["divergence_per_volume", "volume_integrated_flux"]
ProjectionCorrectionLimitMode = Literal[
    "none",
    "cell_divergence_cap",
    "face_flux_cap",
    "redistribute_local",
]
ViscousPredictorMode = Literal[
    "none",
    "no_viscous_debug_copy",
    "explicit_cell_velocity_laplacian_substepped",
    "explicit_cell_velocity_laplacian_substepped_conservative",
    "face_flux_laplacian_substepped",
]
OutletProjectionMode = Literal[
    "outlet_pressure_dirichlet",
    "outlet_flux_preserve",
    "outlet_mass_balance_rescale",
]
ConvectiveStabilizationMode = Literal["auto_damping", "substepping"]
ConvectiveSubstepBoundaryContractMode = Literal["end_only", "every_substep"]
ViscousPredictorOutletContractMode = Literal["auto", "match_inlet", "preserve"]
PressureProjectionOutletContractMode = Literal["auto", "match_inlet", "preserve"]
PressureNonorthogonalCorrectionMode = Literal["auto", "none", "deferred_lsq"]
ViscousNonorthogonalCorrectionMode = Literal["auto", "none", "deferred_lsq"]
ProjectionCellVelocityUpdateMode = Literal[
    "auto",
    "legacy_reconstruct",
    "momentum_pressure_corrected",
]
WallVelocityBoundaryMode = Literal[
    "slip",
    "no_slip",
    "no_slip_tangential",
    "no_slip_legacy_isotropic",
]


@dataclass(frozen=True)
class TetraFlowConfig:
    density: float = 1000.0
    kinematic_viscosity: float = 1e-6
    inlet_speed: float = 0.15
    pressure_outlet_value: float = 0.0
    max_pressure_iterations: int = 500
    pressure_tolerance: float = 1e-8
    pressure_relative_tolerance: float = 1e-3
    divergence_tolerance: float = 1e-8
    relaxation_omega: float = 1.0
    backend: FlowBackend = "auto"
    device: str = ""
    pressure_solver: PressureSolver = "jacobi"
    debug_store_history: bool = True
    projection_dt: float = 5e-4
    projection_sign: ProjectionSign = "minus"
    enable_sign_comparison: bool = True
    outlet_projection_mode: OutletProjectionMode = "outlet_pressure_dirichlet"
    pressure_projection_outlet_contract_mode: PressureProjectionOutletContractMode = (
        "auto"
    )
    pressure_nonorthogonal_correction_mode: PressureNonorthogonalCorrectionMode = "auto"
    pressure_nonorthogonal_correction_sweeps: int = 4
    pressure_nonorthogonal_correction_relaxation: float = 1.0
    projection_cell_velocity_update_mode: ProjectionCellVelocityUpdateMode = "auto"
    cg_breakdown_eps: float = 1e-30
    cg_stagnation_window: int = 25
    cg_stagnation_ratio: float = 0.995
    pcg_require_relative_l2_convergence: bool = False
    projection_rhs_mode: ProjectionRhsMode = "volume_integrated_flux"
    projection_correction_damping: float = 1.0
    projection_correction_limit_mode: ProjectionCorrectionLimitMode = "none"
    projection_divergence_cap_factor: float = 2.0
    projection_divergence_floor: float = 1e-12
    projection_face_correction_over_volume_cap: float = 8_000.0
    viscous_predictor_mode: ViscousPredictorMode = (
        "explicit_cell_velocity_laplacian_substepped"
    )
    viscous_predictor_outlet_contract_mode: ViscousPredictorOutletContractMode = "auto"
    viscous_nonorthogonal_correction_mode: ViscousNonorthogonalCorrectionMode = "auto"
    viscous_face_flux_divergence_impact_cap: float = 1.0
    viscous_face_flux_laplacian_vectorized: bool = True
    torch_cuda_viscosity_enabled: bool = True
    enable_convective_predictor: bool = False
    disable_convective_predictor: bool = False
    convective_cfl_limit: float = 0.5
    convective_predictor_damping: float = 1.0
    disable_convective_auto_damping: bool = False
    convective_stabilization_mode: ConvectiveStabilizationMode = "auto_damping"
    convective_substep_boundary_contract: ConvectiveSubstepBoundaryContractMode = (
        "end_only"
    )
    max_convective_substeps: int = 128
    fail_on_convective_substep_cap: bool = False
    wall_velocity_boundary_mode: WallVelocityBoundaryMode = "slip"
    wall_tangential_no_slip_strength: float = 1.0
    wall_tangential_shear_face_flux_enabled: bool = True
    wall_tangential_cell_velocity_momentum_enabled: bool = True
    wall_flux_stokes_resistance_enabled: bool = False
    wall_flux_stokes_resistance_strength: float = 1.0


@dataclass
class TetraFlowState:
    face_flux: np.ndarray
    cell_velocity: np.ndarray
    pressure: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _remember_torch_array(
    array: np.ndarray,
    tensor: Any,
    *,
    device: str,
) -> None:
    """Keep an immutable NumPy/device pair alive while the NumPy peer exists."""
    arr = np.asarray(array)
    arr.setflags(write=False)
    key = (id(arr), str(device))

    def _discard(reference: weakref.ReferenceType[np.ndarray]) -> None:
        current = _TORCH_DYNAMIC_ARRAY_CACHE.get(key)
        if current is not None and current[0] is reference:
            _TORCH_DYNAMIC_ARRAY_CACHE.pop(key, None)

    reference = weakref.ref(arr, _discard)
    _TORCH_DYNAMIC_ARRAY_CACHE[key] = (reference, tensor)


def _find_torch_array(array: np.ndarray, *, device: str) -> Any | None:
    arr = np.asarray(array)
    key = (id(arr), str(device))
    cached = _TORCH_DYNAMIC_ARRAY_CACHE.get(key)
    if cached is None:
        return None
    if arr.flags.writeable:
        _TORCH_DYNAMIC_ARRAY_CACHE.pop(key, None)
        return None
    reference, tensor = cached
    if reference() is not arr:
        _TORCH_DYNAMIC_ARRAY_CACHE.pop(key, None)
        return None
    return tensor


def clear_tetra_flow_torch_caches() -> None:
    """Release cached Torch tensors at a worker or mesh lifecycle boundary."""
    global _ACTIVE_TORCH_CACHE_CONTEXT

    _TORCH_PRESSURE_CACHE.clear()
    _TORCH_CONVECTION_CACHE.clear()
    _TORCH_VISCOSITY_CACHE.clear()
    _TORCH_DYNAMIC_ARRAY_CACHE.clear()
    _ACTIVE_TORCH_CACHE_CONTEXT = None


def _activate_torch_cache_context(mesh: ImportedTetraMesh, *, device: str) -> None:
    """Keep Torch geometry caches bounded to one active mesh and device."""
    global _ACTIVE_TORCH_CACHE_CONTEXT

    context = (_mesh_geometry_cache_key(mesh), str(device))
    if context == _ACTIVE_TORCH_CACHE_CONTEXT:
        return
    _TORCH_PRESSURE_CACHE.clear()
    _TORCH_CONVECTION_CACHE.clear()
    _TORCH_VISCOSITY_CACHE.clear()
    _TORCH_DYNAMIC_ARRAY_CACHE.clear()
    _ACTIVE_TORCH_CACHE_CONTEXT = context


def _validate_config(mesh: ImportedTetraMesh, config: TetraFlowConfig) -> None:
    if mesh.tetrahedra.shape[0] <= 0:
        raise ValueError("Mesh has no tetra cells.")
    if mesh.face_vertices.shape[0] <= 0:
        raise ValueError("Mesh has no faces.")
    if config.density <= 0:
        raise ValueError("density must be positive.")
    if config.max_pressure_iterations <= 0:
        raise ValueError("max_pressure_iterations must be positive.")
    if config.pressure_tolerance <= 0:
        raise ValueError("pressure_tolerance must be positive.")
    if config.pressure_relative_tolerance <= 0:
        raise ValueError("pressure_relative_tolerance must be positive.")
    if config.projection_dt <= 0:
        raise ValueError("projection_dt must be positive.")
    if config.relaxation_omega <= 0:
        raise ValueError("relaxation_omega must be positive.")
    if config.pressure_solver not in ("jacobi", "cg", "pcg_diag", "amg_pcg"):
        raise ValueError(
            "pressure_solver must be 'jacobi', 'cg', 'pcg_diag' or 'amg_pcg'."
        )
    if config.projection_sign not in ("minus", "plus"):
        raise ValueError("projection_sign must be 'minus' or 'plus'.")
    if config.outlet_projection_mode not in (
        "outlet_pressure_dirichlet",
        "outlet_flux_preserve",
        "outlet_mass_balance_rescale",
    ):
        raise ValueError("Unknown outlet_projection_mode.")
    if config.pressure_projection_outlet_contract_mode not in (
        "match_inlet",
        "preserve",
    ):
        raise ValueError(
            "pressure_projection_outlet_contract_mode must be "
            "'match_inlet' or 'preserve'."
        )
    if config.pressure_nonorthogonal_correction_mode not in (
        "none",
        "deferred_lsq",
    ):
        raise ValueError(
            "pressure_nonorthogonal_correction_mode must be 'none' or 'deferred_lsq'."
        )
    if config.pressure_nonorthogonal_correction_sweeps < 1:
        raise ValueError("pressure_nonorthogonal_correction_sweeps must be >= 1.")
    if not (0.0 < config.pressure_nonorthogonal_correction_relaxation <= 1.0):
        raise ValueError(
            "pressure_nonorthogonal_correction_relaxation must be in (0, 1]."
        )
    if (
        config.pressure_nonorthogonal_correction_mode == "deferred_lsq"
        and config.projection_rhs_mode == "divergence_per_volume"
    ):
        raise ValueError(
            "pressure_nonorthogonal_correction_mode='deferred_lsq' requires "
            "projection_rhs_mode='volume_integrated_flux'; the legacy "
            "divergence_per_volume RHS is not consistent with the existing "
            "volume-integrated SPD pressure matrix."
        )
    if config.projection_cell_velocity_update_mode not in (
        "legacy_reconstruct",
        "momentum_pressure_corrected",
    ):
        raise ValueError(
            "projection_cell_velocity_update_mode must be "
            "'legacy_reconstruct' or 'momentum_pressure_corrected'."
        )
    if config.cg_breakdown_eps <= 0.0:
        raise ValueError("cg_breakdown_eps must be positive.")
    if config.cg_stagnation_window < 2:
        raise ValueError("cg_stagnation_window must be >= 2.")
    if not (0.0 < config.cg_stagnation_ratio <= 1.0):
        raise ValueError("cg_stagnation_ratio must be in (0, 1].")
    if config.projection_rhs_mode not in (
        "divergence_per_volume",
        "volume_integrated_flux",
    ):
        raise ValueError(
            "projection_rhs_mode must be 'divergence_per_volume' or 'volume_integrated_flux'."
        )
    if config.projection_correction_damping <= 0.0:
        raise ValueError("projection_correction_damping must be positive.")
    if config.projection_correction_limit_mode not in (
        "none",
        "cell_divergence_cap",
        "face_flux_cap",
        "redistribute_local",
    ):
        raise ValueError("Unknown projection_correction_limit_mode.")
    if config.projection_divergence_cap_factor <= 0.0:
        raise ValueError("projection_divergence_cap_factor must be positive.")
    if config.projection_divergence_floor <= 0.0:
        raise ValueError("projection_divergence_floor must be positive.")
    if config.projection_face_correction_over_volume_cap <= 0.0:
        raise ValueError("projection_face_correction_over_volume_cap must be positive.")
    if config.viscous_predictor_mode not in (
        "none",
        "no_viscous_debug_copy",
        "explicit_cell_velocity_laplacian_substepped",
        "explicit_cell_velocity_laplacian_substepped_conservative",
        "face_flux_laplacian_substepped",
    ):
        raise ValueError("Unknown viscous_predictor_mode.")
    if config.viscous_predictor_outlet_contract_mode not in (
        "match_inlet",
        "preserve",
    ):
        raise ValueError(
            "viscous_predictor_outlet_contract_mode must be "
            "'match_inlet' or 'preserve'."
        )
    if config.viscous_nonorthogonal_correction_mode not in (
        "none",
        "deferred_lsq",
    ):
        raise ValueError(
            "viscous_nonorthogonal_correction_mode must be 'none' or 'deferred_lsq'."
        )
    if (
        config.viscous_nonorthogonal_correction_mode == "deferred_lsq"
        and config.viscous_predictor_mode
        not in {
            "explicit_cell_velocity_laplacian_substepped",
            "explicit_cell_velocity_laplacian_substepped_conservative",
        }
    ):
        raise ValueError(
            "viscous_nonorthogonal_correction_mode='deferred_lsq' requires "
            "a cell-velocity Laplacian viscous predictor."
        )
    if config.viscous_nonorthogonal_correction_mode == "deferred_lsq" and not (
        _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)
        or _uses_legacy_isotropic_no_slip_wall(config.wall_velocity_boundary_mode)
    ):
        raise ValueError(
            "viscous_nonorthogonal_correction_mode='deferred_lsq' requires "
            "a no-slip wall velocity boundary mode."
        )
    if config.viscous_face_flux_divergence_impact_cap <= 0.0:
        raise ValueError("viscous_face_flux_divergence_impact_cap must be positive.")
    if config.convective_cfl_limit <= 0.0:
        raise ValueError("convective_cfl_limit must be positive.")
    if config.convective_predictor_damping <= 0.0:
        raise ValueError("convective_predictor_damping must be positive.")
    if config.convective_stabilization_mode not in ("auto_damping", "substepping"):
        raise ValueError(
            "convective_stabilization_mode must be 'auto_damping' or 'substepping'."
        )
    if config.convective_substep_boundary_contract not in ("end_only", "every_substep"):
        raise ValueError(
            "convective_substep_boundary_contract must be 'end_only' or 'every_substep'."
        )
    if config.max_convective_substeps < 1:
        raise ValueError("max_convective_substeps must be >= 1.")
    if config.wall_velocity_boundary_mode not in (
        "slip",
        "no_slip",
        "no_slip_tangential",
        "no_slip_legacy_isotropic",
    ):
        raise ValueError(
            "wall_velocity_boundary_mode must be "
            "'slip', 'no_slip', 'no_slip_tangential' or "
            "'no_slip_legacy_isotropic'."
        )
    if config.wall_tangential_no_slip_strength < 0.0:
        raise ValueError("wall_tangential_no_slip_strength must be non-negative.")
    if config.wall_flux_stokes_resistance_strength < 0.0:
        raise ValueError("wall_flux_stokes_resistance_strength must be non-negative.")


def _resolve_backend(config: TetraFlowConfig) -> BackendSelection:
    return select_backend(config.backend)


def _uses_tangential_no_slip_wall(mode: WallVelocityBoundaryMode | str) -> bool:
    return str(mode) in {"no_slip", "no_slip_tangential"}


def resolve_tetra_flow_numerical_profile(
    config: TetraFlowConfig,
) -> TetraFlowConfig:
    """Resolve ``auto`` without changing the established slip discretization.

    The wall mode is the user-facing opt-in.  A tangential no-slip wall needs the
    coherent momentum/flux update and both non-orthogonal corrections; the default
    slip wall retains the historical projection and TPFA operators bit for bit.
    Explicit mode selections always override this profile.
    """

    physical_no_slip = _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)
    updates: dict[str, str] = {}
    if config.viscous_predictor_outlet_contract_mode == "auto":
        updates["viscous_predictor_outlet_contract_mode"] = (
            "preserve" if physical_no_slip else "match_inlet"
        )
    if config.pressure_projection_outlet_contract_mode == "auto":
        updates["pressure_projection_outlet_contract_mode"] = (
            "preserve" if physical_no_slip else "match_inlet"
        )
    if config.projection_cell_velocity_update_mode == "auto":
        updates["projection_cell_velocity_update_mode"] = (
            "momentum_pressure_corrected" if physical_no_slip else "legacy_reconstruct"
        )
    if config.pressure_nonorthogonal_correction_mode == "auto":
        updates["pressure_nonorthogonal_correction_mode"] = (
            "deferred_lsq" if physical_no_slip else "none"
        )
    if config.viscous_nonorthogonal_correction_mode == "auto":
        cell_velocity_viscous_mode = config.viscous_predictor_mode in {
            "explicit_cell_velocity_laplacian_substepped",
            "explicit_cell_velocity_laplacian_substepped_conservative",
        }
        updates["viscous_nonorthogonal_correction_mode"] = (
            "deferred_lsq"
            if physical_no_slip and cell_velocity_viscous_mode
            else "none"
        )
    return replace(config, **updates) if updates else config


def _numerical_profile_resolution_diagnostics(
    requested: TetraFlowConfig,
    effective: TetraFlowConfig,
) -> dict[str, Any]:
    fields = (
        "viscous_predictor_outlet_contract_mode",
        "pressure_projection_outlet_contract_mode",
        "projection_cell_velocity_update_mode",
        "pressure_nonorthogonal_correction_mode",
        "viscous_nonorthogonal_correction_mode",
    )
    return {
        "wall_velocity_boundary_mode": str(effective.wall_velocity_boundary_mode),
        "physical_no_slip_profile": bool(
            _uses_tangential_no_slip_wall(effective.wall_velocity_boundary_mode)
        ),
        "requested": {name: str(getattr(requested, name)) for name in fields},
        "effective": {name: str(getattr(effective, name)) for name in fields},
    }


def _wall_reconstruction_boundary_mode(
    config: TetraFlowConfig,
) -> WallVelocityBoundaryMode:
    if _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode):
        return "slip"
    return config.wall_velocity_boundary_mode


def _uses_legacy_isotropic_no_slip_wall(
    mode: WallVelocityBoundaryMode | str,
) -> bool:
    return str(mode) == "no_slip_legacy_isotropic"


def _uses_wall_flux_stokes_resistance(config: TetraFlowConfig) -> bool:
    return bool(
        config.wall_flux_stokes_resistance_enabled
    ) and _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)


def _mesh_geometry_cache_key(mesh: ImportedTetraMesh) -> tuple[int, ...]:
    return (
        id(mesh),
        id(mesh.face_to_cells),
        id(mesh.face_centers),
        id(mesh.cell_to_faces),
        id(mesh.wall_faces),
        int(mesh.tetrahedra.shape[0]),
        int(mesh.face_vertices.shape[0]),
    )


def _inlet_face_sets(mesh: ImportedTetraMesh) -> tuple[np.ndarray, np.ndarray]:
    inlet = resolve_inlet_face_groups(mesh)
    return (
        np.asarray(inlet["left_faces"], dtype=np.int64),
        np.asarray(inlet["right_faces"], dtype=np.int64),
    )


def _vector_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "l2": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "l2": float(np.sqrt(np.mean(arr * arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def _safe_ratio(num: float, den: float) -> float:
    return float(num / max(abs(den), 1e-30))


def _compute_outlet_flux_from_inlet(
    mesh: ImportedTetraMesh,
    inlet_flux_in: float,
    outlet_faces: np.ndarray,
) -> float:
    if outlet_faces.size == 0:
        return 0.0
    outlet_area = float(np.sum(mesh.face_areas[outlet_faces]))
    if outlet_area <= 1e-20:
        return 0.0
    return inlet_flux_in / outlet_area


def _apply_face_flux_boundary_conditions_inplace(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    inlet_speed: float,
    left_inlet_faces: np.ndarray,
    right_inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    outlet_contract_mode: ViscousPredictorOutletContractMode = "match_inlet",
) -> dict[str, float]:
    if outlet_contract_mode not in {"match_inlet", "preserve"}:
        raise ValueError(
            "outlet_contract_mode must be resolved to 'match_inlet' or 'preserve'."
        )
    if left_inlet_faces.size:
        face_flux[left_inlet_faces] = (
            -float(inlet_speed) * mesh.face_areas[left_inlet_faces]
        )
    if right_inlet_faces.size:
        face_flux[right_inlet_faces] = (
            -float(inlet_speed) * mesh.face_areas[right_inlet_faces]
        )
    if wall_faces.size:
        face_flux[wall_faces] = 0.0

    inlet_faces = np.concatenate((left_inlet_faces, right_inlet_faces))
    inlet_flux_in = 0.0
    if inlet_faces.size:
        inlet_flux_in = float(np.sum(np.maximum(-face_flux[inlet_faces], 0.0)))
    outlet_speed = _compute_outlet_flux_from_inlet(mesh, inlet_flux_in, outlet_faces)
    if outlet_faces.size and str(outlet_contract_mode) == "match_inlet":
        face_flux[outlet_faces] = outlet_speed * mesh.face_areas[outlet_faces]

    outlet_flux_out = (
        float(np.sum(np.maximum(face_flux[outlet_faces], 0.0)))
        if outlet_faces.size
        else 0.0
    )
    return {
        "inlet_flux_total": inlet_flux_in,
        "outlet_flux_total": outlet_flux_out,
        "outlet_speed": float(outlet_speed),
    }


def _reconstruct_cell_velocity_from_face_flux_numpy_direct(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    wall_velocity_boundary_mode: WallVelocityBoundaryMode = "slip",
    wall_tangential_no_slip_strength: float = 1.0,
) -> np.ndarray:
    cell_faces = np.asarray(mesh.cell_to_faces, dtype=np.int64)
    n_cells = mesh.tetrahedra.shape[0]
    eye = np.eye(3, dtype=np.float64)
    wall_penalty = np.zeros((n_cells,), dtype=np.float64)
    if _uses_tangential_no_slip_wall(wall_velocity_boundary_mode):
        wall_velocity_boundary_mode = "slip"
    elif _uses_legacy_isotropic_no_slip_wall(wall_velocity_boundary_mode):
        wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
        if wall_faces.size:
            wall_owner = np.asarray(mesh.face_to_cells[wall_faces, 0], dtype=np.int64)
            wall_area = np.asarray(mesh.face_areas[wall_faces], dtype=np.float64)
            np.add.at(wall_penalty, wall_owner, wall_area)

    if cell_faces.ndim != 2 or cell_faces.shape[1] != 4:
        raise ValueError("Imported tetra mesh cell_to_faces must have shape (N, 4).")
    cell_ids = np.arange(n_cells, dtype=np.int64)[:, None]
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.maximum(np.asarray(mesh.face_areas, dtype=np.float64), 1e-16)
    flux = np.asarray(face_flux, dtype=np.float64)

    owner = face_to_cells[cell_faces, 0]
    owner_oriented = owner == cell_ids
    oriented_normals = np.where(
        owner_oriented[:, :, None],
        face_normals[cell_faces],
        -face_normals[cell_faces],
    )
    signed_flux = np.where(owner_oriented, flux[cell_faces], -flux[cell_faces])
    weights = face_areas[cell_faces]
    mat = (
        np.einsum(
            "cf,cfi,cfj->cij",
            weights,
            oriented_normals,
            oriented_normals,
            optimize=True,
        )
        + 1e-14 * eye[None, :, :]
    )
    if np.any(wall_penalty > 0.0):
        mat = mat + wall_penalty[:, None, None] * eye[None, :, :]
    rhs = np.einsum("cfi,cf->ci", oriented_normals, signed_flux, optimize=True)
    try:
        return np.linalg.solve(mat, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        out = np.zeros((n_cells, 3), dtype=np.float64)
        for cell_idx in range(n_cells):
            try:
                out[cell_idx] = np.linalg.solve(mat[cell_idx], rhs[cell_idx])
            except np.linalg.LinAlgError:
                out[cell_idx] = 0.0
        return out


def _reconstruct_cell_velocity_from_face_flux_numpy(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    wall_velocity_boundary_mode: WallVelocityBoundaryMode = "slip",
    wall_tangential_no_slip_strength: float = 1.0,
) -> np.ndarray:
    if str(wall_velocity_boundary_mode) != "slip":
        return _reconstruct_cell_velocity_from_face_flux_numpy_direct(
            mesh,
            face_flux,
            wall_velocity_boundary_mode=wall_velocity_boundary_mode,
            wall_tangential_no_slip_strength=wall_tangential_no_slip_strength,
        )

    geometry = _cached_slip_velocity_reconstruction_geometry(mesh)
    flux = np.asarray(face_flux, dtype=np.float64)
    signed_flux = flux[geometry["cell_faces"]] * geometry["orientation_sign"]
    return np.einsum(
        "cif,cf->ci",
        geometry["reconstruction_map"],
        signed_flux,
        optimize=True,
    )


def _build_wall_tangential_projection(face_normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(face_normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("face_normals must have shape (N, 3).")
    normal_norm = np.linalg.norm(normals, axis=1)
    normal_hat = normals / np.maximum(normal_norm[:, None], 1e-30)
    eye = np.eye(3, dtype=np.float64)[None, :, :]
    outer = normal_hat[:, :, None] * normal_hat[:, None, :]
    return eye - outer


def _build_wall_no_slip_sink_coefficient(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
) -> np.ndarray:
    n_cells = mesh.tetrahedra.shape[0]
    sink = np.zeros((n_cells,), dtype=np.float64)
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    if wall_ids.size == 0:
        return sink
    owner = np.asarray(mesh.face_to_cells[wall_ids, 0], dtype=np.int64)
    cell_centers = np.asarray(mesh.cell_centers, dtype=np.float64)[owner]
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)[wall_ids]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)[wall_ids]
    face_area = np.asarray(mesh.face_areas, dtype=np.float64)[wall_ids]
    cell_volume = np.maximum(
        np.asarray(mesh.cell_volumes, dtype=np.float64)[owner],
        1e-30,
    )
    normal_norm = np.linalg.norm(face_normals, axis=1)
    normal_hat = face_normals / np.maximum(normal_norm[:, None], 1e-30)
    center_delta = face_centers - cell_centers
    normal_distance = np.abs(np.sum(center_delta * normal_hat, axis=1))
    center_distance = np.linalg.norm(center_delta, axis=1)
    distance = np.where(normal_distance > 1e-12, normal_distance, center_distance)
    distance = np.maximum(distance, 1e-12)
    coeff = face_area / distance
    np.add.at(sink, owner, coeff / cell_volume)
    return sink


def _cached_wall_no_slip_sink_coefficient(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
) -> np.ndarray:
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    key = (*_mesh_geometry_cache_key(mesh), "wall_no_slip_sink", int(wall_ids.size))
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_wall_no_slip_sink_coefficient(mesh, wall_ids)
        _GEOMETRY_CACHE[key] = cached
    return np.asarray(cached, dtype=np.float64)


def _build_wall_flux_stokes_resistance(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
    *,
    strength: float = 1.0,
) -> dict[str, np.ndarray]:
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    if wall_ids.size == 0:
        return {
            "cell_ids": np.zeros((0,), dtype=np.int64),
            "cell_face_ids": np.zeros((0, 4), dtype=np.int64),
            "variable_face_mask": np.zeros((0, 4), dtype=bool),
            "cell_active_pos": np.zeros((0, 4), dtype=np.int64),
            "operator": np.zeros((0, 4, 4), dtype=np.float64),
            "active_face_ids": np.zeros((0,), dtype=np.int64),
            "active_diag": np.zeros((0,), dtype=np.float64),
            "matrix_rows": np.zeros((0,), dtype=np.int64),
            "matrix_cols": np.zeros((0,), dtype=np.int64),
            "matrix_vals": np.zeros((0,), dtype=np.float64),
            "fixed_rows": np.zeros((0,), dtype=np.int64),
            "fixed_face_ids": np.zeros((0,), dtype=np.int64),
            "fixed_vals": np.zeros((0,), dtype=np.float64),
        }
    wall_operator = _build_wall_tangential_no_slip_operator(
        mesh,
        wall_ids,
        strength=float(strength),
    )
    candidate_cells = np.where(np.max(np.abs(wall_operator), axis=(1, 2)) > 0.0)[0]
    if candidate_cells.size == 0:
        return {
            "cell_ids": np.zeros((0,), dtype=np.int64),
            "cell_face_ids": np.zeros((0, 4), dtype=np.int64),
            "variable_face_mask": np.zeros((0, 4), dtype=bool),
            "cell_active_pos": np.zeros((0, 4), dtype=np.int64),
            "operator": np.zeros((0, 4, 4), dtype=np.float64),
            "active_face_ids": np.zeros((0,), dtype=np.int64),
            "active_diag": np.zeros((0,), dtype=np.float64),
            "matrix_rows": np.zeros((0,), dtype=np.int64),
            "matrix_cols": np.zeros((0,), dtype=np.int64),
            "matrix_vals": np.zeros((0,), dtype=np.float64),
            "fixed_rows": np.zeros((0,), dtype=np.int64),
            "fixed_face_ids": np.zeros((0,), dtype=np.int64),
            "fixed_vals": np.zeros((0,), dtype=np.float64),
        }
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.face_areas, dtype=np.float64)
    local_face_ids: list[np.ndarray] = []
    local_variable_mask: list[np.ndarray] = []
    local_operator: list[np.ndarray] = []
    active_cells: list[int] = []
    active_faces: set[int] = set()
    eye4 = np.eye(4, dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for cell_idx in candidate_cells.tolist():
        faces = np.asarray(mesh.cell_to_faces[cell_idx], dtype=np.int64)
        if faces.size != 4:
            continue
        oriented_normals = np.zeros((4, 3), dtype=np.float64)
        local_flux_sign = np.ones((4,), dtype=np.float64)
        variable_mask = np.zeros((4,), dtype=bool)
        for local_idx, fid in enumerate(faces.tolist()):
            owner = int(face_to_cells[fid, 0])
            neigh = int(face_to_cells[fid, 1])
            if owner == cell_idx:
                oriented_normals[local_idx] = face_normals[fid]
                local_flux_sign[local_idx] = 1.0
            elif neigh == cell_idx:
                oriented_normals[local_idx] = -face_normals[fid]
                local_flux_sign[local_idx] = -1.0
            else:
                oriented_normals[local_idx] = 0.0
                local_flux_sign[local_idx] = 0.0
            variable_mask[local_idx] = neigh >= 0
        if not np.any(variable_mask):
            continue
        area = np.maximum(face_areas[faces], 1e-16)
        mat = 1e-14 * eye3
        for local_idx in range(4):
            n_vec = oriented_normals[local_idx]
            mat = mat + area[local_idx] * np.outer(n_vec, n_vec)
        rhs_map = oriented_normals.T
        try:
            recon_map = np.linalg.solve(mat, rhs_map)
        except np.linalg.LinAlgError:
            continue
        op_local = recon_map.T @ wall_operator[cell_idx] @ recon_map
        op_local = 0.5 * (op_local + op_local.T)
        op_local = local_flux_sign[:, None] * op_local * local_flux_sign[None, :]
        variable_ids = np.flatnonzero(variable_mask)
        if variable_ids.size > 1:
            constraint = local_flux_sign[variable_ids]
            denom = float(np.dot(constraint, constraint))
            if denom > 0.0:
                proj = (
                    np.eye(variable_ids.size, dtype=np.float64)
                    - np.outer(constraint, constraint) / denom
                )
                fixed_ids = np.flatnonzero(~variable_mask)
                op_vv = op_local[np.ix_(variable_ids, variable_ids)]
                op_local[np.ix_(variable_ids, variable_ids)] = proj @ op_vv @ proj
                if fixed_ids.size:
                    op_vf = op_local[np.ix_(variable_ids, fixed_ids)]
                    op_fv = op_local[np.ix_(fixed_ids, variable_ids)]
                    op_local[np.ix_(variable_ids, fixed_ids)] = proj @ op_vf
                    op_local[np.ix_(fixed_ids, variable_ids)] = op_fv @ proj
        if not np.any(np.abs(op_local) > 0.0):
            continue
        local_face_ids.append(faces.copy())
        local_variable_mask.append(variable_mask.copy())
        local_operator.append(op_local + 1e-14 * eye4)
        active_cells.append(int(cell_idx))
        active_faces.update(int(fid) for fid in faces[variable_mask].tolist())
    if not active_cells:
        return {
            "cell_ids": np.zeros((0,), dtype=np.int64),
            "cell_face_ids": np.zeros((0, 4), dtype=np.int64),
            "variable_face_mask": np.zeros((0, 4), dtype=bool),
            "cell_active_pos": np.zeros((0, 4), dtype=np.int64),
            "operator": np.zeros((0, 4, 4), dtype=np.float64),
            "active_face_ids": np.zeros((0,), dtype=np.int64),
            "active_diag": np.zeros((0,), dtype=np.float64),
            "matrix_rows": np.zeros((0,), dtype=np.int64),
            "matrix_cols": np.zeros((0,), dtype=np.int64),
            "matrix_vals": np.zeros((0,), dtype=np.float64),
            "fixed_rows": np.zeros((0,), dtype=np.int64),
            "fixed_face_ids": np.zeros((0,), dtype=np.int64),
            "fixed_vals": np.zeros((0,), dtype=np.float64),
        }
    active_face_ids = np.asarray(sorted(active_faces), dtype=np.int64)
    face_to_active = -np.ones((mesh.face_vertices.shape[0],), dtype=np.int64)
    face_to_active[active_face_ids] = np.arange(active_face_ids.size, dtype=np.int64)
    cell_face_arr = np.asarray(local_face_ids, dtype=np.int64)
    variable_mask_arr = np.asarray(local_variable_mask, dtype=bool)
    op_arr = np.asarray(local_operator, dtype=np.float64)
    cell_active_pos = -np.ones_like(cell_face_arr, dtype=np.int64)
    active_entries = variable_mask_arr & (face_to_active[cell_face_arr] >= 0)
    cell_active_pos[active_entries] = face_to_active[cell_face_arr[active_entries]]
    active_diag = np.zeros((active_face_ids.size,), dtype=np.float64)
    matrix_rows: list[int] = []
    matrix_cols: list[int] = []
    matrix_vals: list[float] = []
    fixed_rows: list[int] = []
    fixed_face_ids: list[int] = []
    fixed_vals: list[float] = []
    for local_idx in range(cell_face_arr.shape[0]):
        pos = cell_active_pos[local_idx]
        mask = pos >= 0
        if not np.any(mask):
            continue
        local_ids = np.flatnonzero(mask)
        np.add.at(active_diag, pos[mask], np.diag(op_arr[local_idx])[local_ids])
        fixed_ids = np.flatnonzero(~mask)
        for i_local in local_ids.tolist():
            row = int(pos[i_local])
            for j_local in local_ids.tolist():
                val = float(op_arr[local_idx, i_local, j_local])
                if abs(val) <= 0.0:
                    continue
                matrix_rows.append(row)
                matrix_cols.append(int(pos[j_local]))
                matrix_vals.append(val)
            for j_local in fixed_ids.tolist():
                val = float(op_arr[local_idx, i_local, j_local])
                if abs(val) <= 0.0:
                    continue
                fixed_rows.append(row)
                fixed_face_ids.append(int(cell_face_arr[local_idx, j_local]))
                fixed_vals.append(val)
    return {
        "cell_ids": np.asarray(active_cells, dtype=np.int64),
        "cell_face_ids": cell_face_arr,
        "variable_face_mask": variable_mask_arr,
        "cell_active_pos": cell_active_pos,
        "operator": op_arr,
        "active_face_ids": active_face_ids,
        "active_diag": active_diag,
        "matrix_rows": np.asarray(matrix_rows, dtype=np.int64),
        "matrix_cols": np.asarray(matrix_cols, dtype=np.int64),
        "matrix_vals": np.asarray(matrix_vals, dtype=np.float64),
        "fixed_rows": np.asarray(fixed_rows, dtype=np.int64),
        "fixed_face_ids": np.asarray(fixed_face_ids, dtype=np.int64),
        "fixed_vals": np.asarray(fixed_vals, dtype=np.float64),
    }


def _cached_wall_flux_stokes_resistance(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
    *,
    strength: float = 1.0,
) -> dict[str, np.ndarray]:
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    key = (
        *_mesh_geometry_cache_key(mesh),
        "wall_flux_stokes_resistance",
        int(wall_ids.size),
        float(strength),
    )
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_wall_flux_stokes_resistance(
            mesh,
            wall_ids,
            strength=float(strength),
        )
        _GEOMETRY_CACHE[key] = cached
    return {
        "cell_ids": np.asarray(cached["cell_ids"], dtype=np.int64),
        "cell_face_ids": np.asarray(cached["cell_face_ids"], dtype=np.int64),
        "variable_face_mask": np.asarray(cached["variable_face_mask"], dtype=bool),
        "cell_active_pos": np.asarray(cached["cell_active_pos"], dtype=np.int64),
        "operator": np.asarray(cached["operator"], dtype=np.float64),
        "active_face_ids": np.asarray(cached["active_face_ids"], dtype=np.int64),
        "active_diag": np.asarray(cached["active_diag"], dtype=np.float64),
        "matrix_rows": np.asarray(cached["matrix_rows"], dtype=np.int64),
        "matrix_cols": np.asarray(cached["matrix_cols"], dtype=np.int64),
        "matrix_vals": np.asarray(cached["matrix_vals"], dtype=np.float64),
        "fixed_rows": np.asarray(cached["fixed_rows"], dtype=np.int64),
        "fixed_face_ids": np.asarray(cached["fixed_face_ids"], dtype=np.int64),
        "fixed_vals": np.asarray(cached["fixed_vals"], dtype=np.float64),
    }


def _build_wall_tangential_no_slip_operator(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    n_cells = mesh.tetrahedra.shape[0]
    mats = np.zeros((n_cells, 3, 3), dtype=np.float64)
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    if wall_ids.size == 0:
        return mats
    owner = np.asarray(mesh.face_to_cells[wall_ids, 0], dtype=np.int64)
    cell_centers = np.asarray(mesh.cell_centers, dtype=np.float64)[owner]
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)[wall_ids]
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)[wall_ids]
    face_area = np.asarray(mesh.face_areas, dtype=np.float64)[wall_ids]
    cell_volume = np.maximum(
        np.asarray(mesh.cell_volumes, dtype=np.float64)[owner],
        1e-30,
    )
    normal_hat = face_normals / np.maximum(
        np.linalg.norm(face_normals, axis=1)[:, None],
        1e-30,
    )
    center_delta = face_centers - cell_centers
    normal_distance = np.abs(np.sum(center_delta * normal_hat, axis=1))
    center_distance = np.linalg.norm(center_delta, axis=1)
    distance = np.where(normal_distance > 1e-12, normal_distance, center_distance)
    distance = np.maximum(distance, 1e-12)
    coeff = (face_area / distance) / cell_volume
    tangent_proj = _build_wall_tangential_projection(face_normals)
    contrib = float(strength) * coeff[:, None, None] * tangent_proj
    np.add.at(mats, owner, contrib)
    return mats


def _cached_wall_tangential_no_slip_operator(
    mesh: ImportedTetraMesh,
    wall_faces: np.ndarray,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    key = (
        *_mesh_geometry_cache_key(mesh),
        "wall_tangential_no_slip_operator",
        int(wall_ids.size),
        float(strength),
    )
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_wall_tangential_no_slip_operator(
            mesh,
            wall_ids,
            strength=float(strength),
        )
        _GEOMETRY_CACHE[key] = cached
    return np.asarray(cached, dtype=np.float64)


def _apply_wall_no_slip_velocity_sink(
    velocity: np.ndarray,
    *,
    wall_sink: np.ndarray,
    nu: float,
    dt: float,
) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    if vel.size == 0 or np.all(wall_sink <= 0.0) or nu <= 0.0 or dt <= 0.0:
        return vel
    damping = np.clip(1.0 - (float(nu) * float(dt) * wall_sink), 0.0, 1.0)
    return vel * damping[:, None]


def _apply_wall_tangential_no_slip_implicit(
    velocity: np.ndarray,
    *,
    wall_operator: np.ndarray,
    nu: float,
    dt: float,
) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    op = np.asarray(wall_operator, dtype=np.float64)
    if vel.size == 0 or op.size == 0 or nu <= 0.0 or dt <= 0.0:
        return vel
    active = np.where(np.max(np.abs(op), axis=(1, 2)) > 0.0)[0]
    if active.size == 0:
        return vel
    out = vel.copy()
    eye = np.eye(3, dtype=np.float64)[None, :, :]
    lhs = eye + (float(nu) * float(dt)) * op[active]
    rhs = vel[active][:, :, None]
    out[active] = np.linalg.solve(lhs, rhs)[:, :, 0]
    return out


def _apply_wall_tangential_shear_to_face_flux(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    wall_operator: np.ndarray,
    nu: float,
    dt: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    flux = np.asarray(face_flux, dtype=np.float64)
    op = np.asarray(wall_operator, dtype=np.float64)
    active = (
        np.where(np.max(np.abs(op), axis=(1, 2)) > 0.0)[0]
        if op.size
        else np.zeros((0,), dtype=np.int64)
    )
    if flux.size == 0 or active.size == 0 or nu <= 0.0 or dt <= 0.0:
        return flux, {
            "enabled": False,
            "active_cells": int(active.size),
            "delta_l2": 0.0,
            "wall_speed_mean_before": 0.0,
            "wall_speed_mean_after": 0.0,
        }

    velocity_before = _reconstruct_cell_velocity_from_face_flux_numpy(
        mesh,
        flux,
        wall_velocity_boundary_mode="slip",
    )
    velocity_after = _apply_wall_tangential_no_slip_implicit(
        velocity_before,
        wall_operator=op,
        nu=float(nu),
        dt=float(dt),
    )
    out = _face_flux_from_cell_velocity_numpy(mesh, velocity_after)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall_faces.size:
        out[wall_faces] = 0.0
    delta = np.asarray(out, dtype=np.float64) - flux
    speed_before = np.linalg.norm(velocity_before[active], axis=1)
    speed_after = np.linalg.norm(velocity_after[active], axis=1)
    return out, {
        "enabled": True,
        "active_cells": int(active.size),
        "delta_l2": float(np.linalg.norm(delta)),
        "wall_speed_mean_before": float(np.mean(speed_before))
        if speed_before.size
        else 0.0,
        "wall_speed_mean_after": float(np.mean(speed_after))
        if speed_after.size
        else 0.0,
    }


def _wall_flux_stokes_resistance_matvec(
    x_active: np.ndarray,
    *,
    resistance_bundle: dict[str, np.ndarray],
) -> np.ndarray:
    x = np.asarray(x_active, dtype=np.float64)
    cell_active_pos = np.asarray(resistance_bundle["cell_active_pos"], dtype=np.int64)
    operator = np.asarray(resistance_bundle["operator"], dtype=np.float64)
    if x.size == 0 or cell_active_pos.size == 0 or operator.size == 0:
        return np.asarray(x, dtype=np.float64)
    gather_pos = np.maximum(cell_active_pos, 0)
    local_vals = x[gather_pos]
    local_vals[cell_active_pos < 0] = 0.0
    contrib = np.einsum("nij,nj->ni", operator, local_vals, optimize=True)
    out = np.zeros_like(x)
    active_mask = cell_active_pos >= 0
    np.add.at(out, cell_active_pos[active_mask], contrib[active_mask])
    return out


def _prepare_wall_flux_stokes_resistance_solver(
    resistance_bundle: dict[str, np.ndarray],
    *,
    alpha: float,
) -> dict[str, Any]:
    active_face_ids = np.asarray(resistance_bundle["active_face_ids"], dtype=np.int64)
    rows = np.asarray(resistance_bundle["matrix_rows"], dtype=np.int64)
    cols = np.asarray(resistance_bundle["matrix_cols"], dtype=np.int64)
    vals = np.asarray(resistance_bundle["matrix_vals"], dtype=np.float64)
    fixed_rows = np.asarray(resistance_bundle["fixed_rows"], dtype=np.int64)
    fixed_face_ids = np.asarray(resistance_bundle["fixed_face_ids"], dtype=np.int64)
    fixed_vals = np.asarray(resistance_bundle["fixed_vals"], dtype=np.float64)
    n_active = int(active_face_ids.size)
    prepared: dict[str, Any] = {
        "available": False,
        "method": "matrix_free_fallback",
        "alpha": float(alpha),
        "active_face_ids": active_face_ids,
        "fixed_rows": fixed_rows,
        "fixed_face_ids": fixed_face_ids,
        "fixed_vals": fixed_vals,
    }
    if n_active == 0 or rows.size == 0 or vals.size == 0 or alpha <= 0.0:
        prepared["available"] = True
        prepared["method"] = "identity"
        return prepared
    try:
        import scipy.sparse as sp  # type: ignore
        import scipy.sparse.linalg as spla  # type: ignore
    except Exception:
        return prepared
    mat = sp.coo_matrix((vals, (rows, cols)), shape=(n_active, n_active)).tocsr()
    eye = sp.eye(n_active, dtype=np.float64, format="csr")
    system = eye + float(alpha) * mat
    prepared["available"] = True
    prepared["method"] = "scipy_factorized"
    prepared["system"] = system
    try:
        prepared["solve"] = spla.factorized(system.tocsc())
    except Exception:
        prepared["method"] = "scipy_gmres"
        prepared["diag"] = np.asarray(system.diagonal(), dtype=np.float64)
    return prepared


def _apply_wall_flux_stokes_resistance_global_implicit(
    face_flux: np.ndarray,
    *,
    resistance_bundle: dict[str, np.ndarray],
    nu: float,
    dt: float,
    strength: float,
    prepared_solver: dict[str, Any] | None = None,
    max_iter: int = 24,
    rel_tol: float = 1e-6,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    flux = np.asarray(face_flux, dtype=np.float64)
    active_face_ids = np.asarray(resistance_bundle["active_face_ids"], dtype=np.int64)
    cell_face_ids = np.asarray(resistance_bundle["cell_face_ids"], dtype=np.int64)
    cell_active_pos = np.asarray(resistance_bundle["cell_active_pos"], dtype=np.int64)
    operator = np.asarray(resistance_bundle["operator"], dtype=np.float64)
    active_diag = np.asarray(resistance_bundle["active_diag"], dtype=np.float64)
    if (
        flux.size == 0
        or active_face_ids.size == 0
        or cell_face_ids.size == 0
        or cell_active_pos.size == 0
        or operator.size == 0
        or nu <= 0.0
        or dt <= 0.0
        or strength <= 0.0
    ):
        return flux, {
            "iterations": 0,
            "converged": True,
            "residual_l2": 0.0,
        }
    alpha = float(nu) * float(dt) * float(strength)
    rhs = np.asarray(flux[active_face_ids], dtype=np.float64).copy()

    diag = 1.0 + alpha * np.maximum(active_diag, 0.0)
    prepared = prepared_solver
    if prepared is None or abs(float(prepared.get("alpha", -1.0)) - alpha) > 0.0:
        prepared = _prepare_wall_flux_stokes_resistance_solver(
            resistance_bundle,
            alpha=alpha,
        )
    if (
        bool(prepared.get("available", False))
        and str(prepared.get("method")) == "identity"
    ):
        return flux, {
            "iterations": 0,
            "converged": True,
            "residual_l2": 0.0,
            "method": "identity",
        }

    fixed_rows = np.asarray(prepared.get("fixed_rows", np.zeros((0,), dtype=np.int64)))
    fixed_face_ids = np.asarray(
        prepared.get("fixed_face_ids", np.zeros((0,), dtype=np.int64))
    )
    fixed_vals_sparse = np.asarray(
        prepared.get("fixed_vals", np.zeros((0,), dtype=np.float64))
    )
    if fixed_rows.size:
        rhs_adjust_sparse = np.bincount(
            fixed_rows,
            weights=fixed_vals_sparse * flux[fixed_face_ids],
            minlength=active_face_ids.size,
        )
        rhs = np.asarray(flux[active_face_ids], dtype=np.float64) - alpha * np.asarray(
            rhs_adjust_sparse, dtype=np.float64
        )

    def matvec(vec: np.ndarray) -> np.ndarray:
        return vec + alpha * _wall_flux_stokes_resistance_matvec(
            vec,
            resistance_bundle=resistance_bundle,
        )

    x = np.asarray(flux[active_face_ids], dtype=np.float64).copy()
    method = "matrix_free_fallback"
    if bool(prepared.get("available", False)):
        method = str(prepared.get("method", method))
        if method == "scipy_factorized":
            try:
                solve = prepared["solve"]
                x = np.asarray(solve(rhs), dtype=np.float64)
                residual_vec = matvec(x) - rhs
                residual = float(np.linalg.norm(residual_vec))
                out = flux.copy()
                out[active_face_ids] = x
                return out, {
                    "iterations": 1,
                    "converged": True,
                    "residual_l2": float(residual),
                    "method": method,
                }
            except Exception:
                method = "matrix_free_fallback"
        elif method == "scipy_gmres":
            try:
                import scipy.sparse.linalg as spla  # type: ignore

                system = prepared["system"]
                diag_pre = np.asarray(prepared["diag"], dtype=np.float64)

                def precond(v: np.ndarray) -> np.ndarray:
                    return np.asarray(v, dtype=np.float64) / np.maximum(diag_pre, 1e-12)

                m = spla.LinearOperator(system.shape, matvec=precond, dtype=np.float64)
                x, info = spla.gmres(
                    system,
                    rhs,
                    x0=x,
                    rtol=float(rel_tol),
                    atol=0.0,
                    restart=min(100, max(20, active_face_ids.size)),
                    maxiter=int(max_iter),
                    M=m,
                )
                x = np.asarray(x, dtype=np.float64)
                residual_vec = np.asarray(system @ x - rhs, dtype=np.float64)
                residual = float(np.linalg.norm(residual_vec))
                out = flux.copy()
                out[active_face_ids] = x
                return out, {
                    "iterations": int(max_iter if info > 0 else 1),
                    "converged": bool(info == 0),
                    "residual_l2": float(residual),
                    "method": method,
                }
            except Exception:
                method = "matrix_free_fallback"

    r = rhs - matvec(x)
    rhs_norm = float(np.linalg.norm(rhs))
    tol_abs = float(rel_tol) * max(rhs_norm, 1e-30)
    z = r / np.maximum(diag, 1e-12)
    p = z.copy()
    rz_old = float(np.dot(r, z))
    residual = float(np.linalg.norm(r))
    iterations = 0
    converged = residual <= tol_abs
    for k in range(max(1, int(max_iter))):
        if converged:
            break
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if abs(denom) <= 1e-30:
            break
        alpha_k = rz_old / denom
        x = x + alpha_k * p
        r = r - alpha_k * ap
        residual = float(np.linalg.norm(r))
        iterations = k + 1
        if residual <= tol_abs:
            converged = True
            break
        z = r / np.maximum(diag, 1e-12)
        rz_new = float(np.dot(r, z))
        if abs(rz_old) <= 1e-30:
            break
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    out = flux.copy()
    out[active_face_ids] = x
    return out, {
        "iterations": int(iterations),
        "converged": bool(converged),
        "residual_l2": float(residual),
        "method": method,
    }


def _face_flux_from_cell_velocity_numpy(
    mesh: ImportedTetraMesh,
    velocity: np.ndarray,
) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    n = np.asarray(mesh.face_normals, dtype=np.float64)
    area = np.asarray(mesh.face_areas, dtype=np.float64)
    face_velocity = np.asarray(vel[c0], dtype=np.float64).copy()
    interior = c1 >= 0
    if np.any(interior):
        face_velocity[interior] = 0.5 * (vel[c0[interior]] + vel[c1[interior]])
    return np.einsum("ij,ij->i", face_velocity, n, optimize=True) * area


def _face_flux_reinterpolation_consistency_audit(
    mesh: ImportedTetraMesh,
    *,
    reference_face_flux: np.ndarray,
    reinterpolated_face_flux: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference_face_flux, dtype=np.float64)
    reinterpolated = np.asarray(reinterpolated_face_flux, dtype=np.float64)
    if reference.shape != reinterpolated.shape:
        raise ValueError("Face-flux consistency arrays must have matching shapes.")
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    interior = c1 >= 0
    boundary = ~interior
    mismatch = reinterpolated - reference

    def region(mask: np.ndarray) -> dict[str, Any]:
        reference_region = reference[mask]
        reinterpolated_region = reinterpolated[mask]
        mismatch_region = mismatch[mask]
        reference_stats = _vector_stats(reference_region)
        mismatch_stats = _vector_stats(mismatch_region)
        return {
            "face_count": int(np.count_nonzero(mask)),
            "reference": reference_stats,
            "reinterpolated": _vector_stats(reinterpolated_region),
            "mismatch": mismatch_stats,
            "mismatch_relative_l2": _safe_ratio(
                float(mismatch_stats["l2"]),
                float(reference_stats["l2"]),
            ),
        }

    all_faces = np.ones(reference.shape, dtype=bool)
    return {
        "all_faces": region(all_faces),
        "interior_faces": region(interior),
        "boundary_faces": region(boundary),
    }


def _build_face_flux_laplacian_stencil(
    mesh: ImportedTetraMesh,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    interior_face_ids = np.flatnonzero(c1 >= 0).astype(np.int64)
    neighbor_ids: dict[int, np.ndarray] = {}
    neighbor_w: dict[int, np.ndarray] = {}
    neighbor_w_sum: dict[int, float] = {}
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    for fid in interior_face_ids.tolist():
        i = int(c0[fid])
        j = int(c1[fid])
        cand = np.unique(
            np.concatenate(
                (
                    np.asarray(mesh.cell_to_faces[i], dtype=np.int64),
                    np.asarray(mesh.cell_to_faces[j], dtype=np.int64),
                )
            )
        )
        cand = cand[cand != int(fid)]
        if cand.size == 0:
            continue
        d = np.linalg.norm(face_centers[cand] - face_centers[int(fid)], axis=1)
        d = np.maximum(d, 1e-12)
        ww = 1.0 / (d * d)
        neighbor_ids[int(fid)] = cand
        neighbor_w[int(fid)] = ww
        neighbor_w_sum[int(fid)] = float(np.sum(ww))
    return interior_face_ids, neighbor_ids, neighbor_w, neighbor_w_sum


def _cached_face_flux_laplacian_stencil(
    mesh: ImportedTetraMesh,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float]]:
    key = (*_mesh_geometry_cache_key(mesh), "face_flux_laplacian_stencil")
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_face_flux_laplacian_stencil(mesh)
        _GEOMETRY_CACHE[key] = cached
    return cached


def _build_face_flux_laplacian_vector_stencil(
    mesh: ImportedTetraMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    cell_to_faces = np.asarray(mesh.cell_to_faces, dtype=np.int64)
    if cell_to_faces.ndim != 2 or cell_to_faces.shape[1] != 4:
        raise ValueError("Imported tetra mesh cell_to_faces must have shape (N, 4).")

    face_ids = np.flatnonzero(face_to_cells[:, 1] >= 0).astype(np.int64)
    if face_ids.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    owner = face_to_cells[face_ids, 0]
    neighbor = face_to_cells[face_ids, 1]
    candidates = np.concatenate((cell_to_faces[owner], cell_to_faces[neighbor]), axis=1)
    candidates.sort(axis=1)
    unique = np.ones(candidates.shape, dtype=bool)
    unique[:, 1:] = candidates[:, 1:] != candidates[:, :-1]
    valid = unique & (candidates != face_ids[:, None])
    neighbor_counts = np.count_nonzero(valid, axis=1).astype(np.int64)
    active_rows = neighbor_counts > 0
    if not np.any(active_rows):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    sentinel = int(mesh.face_vertices.shape[0])
    candidates[~valid] = sentinel
    candidates.sort(axis=1)
    active = face_ids[active_rows]
    active_counts = neighbor_counts[active_rows]
    max_neighbors = int(np.max(active_counts))
    packed = candidates[active_rows, :max_neighbors]
    valid_columns = (
        np.arange(max_neighbors, dtype=np.int64)[None, :] < active_counts[:, None]
    )
    nb_mat = np.where(valid_columns, packed, active[:, None]).astype(
        np.int64, copy=False
    )
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    distance = np.linalg.norm(
        face_centers[nb_mat] - face_centers[active, None, :], axis=2
    )
    distance = np.maximum(distance, 1e-12)
    w_mat = np.where(valid_columns, 1.0 / (distance * distance), 0.0)
    w_sum = np.sum(w_mat, axis=1)
    return active, nb_mat, w_mat, w_sum


def _cached_face_flux_laplacian_vector_stencil(
    mesh: ImportedTetraMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = (*_mesh_geometry_cache_key(mesh), "face_flux_laplacian_vector_stencil")
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_face_flux_laplacian_vector_stencil(mesh)
        _GEOMETRY_CACHE[key] = cached
    return cached


def _cached_viscous_tpfa_geometry(mesh: ImportedTetraMesh) -> dict[str, np.ndarray]:
    key = (*_mesh_geometry_cache_key(mesh), "viscous_tpfa_geometry")
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
        c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
        area = np.asarray(mesh.face_areas, dtype=np.float64)
        centers = np.asarray(mesh.cell_centers, dtype=np.float64)
        vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
        interior = c1 >= 0
        own = np.asarray(c0[interior], dtype=np.int64)
        nei = np.asarray(c1[interior], dtype=np.int64)
        if own.size:
            distance = np.linalg.norm(centers[nei] - centers[own], axis=1)
            distance = np.maximum(distance, 1e-12)
            weight = np.asarray(area[interior], dtype=np.float64) / distance
        else:
            weight = np.zeros((0,), dtype=np.float64)
        row_coefficient = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
        if own.size:
            np.add.at(row_coefficient, own, weight / vol[own])
            np.add.at(row_coefficient, nei, weight / vol[nei])
        cached = {
            "c0": c0,
            "c1": c1,
            "area": area,
            "centers": centers,
            "volume": vol,
            "interior": interior,
            "owner": own,
            "neighbor": nei,
            "weight": weight,
            "row_coefficient": row_coefficient,
        }
        _GEOMETRY_CACHE[key] = cached
    return cached


def _cached_slip_velocity_reconstruction_geometry(
    mesh: ImportedTetraMesh,
) -> dict[str, np.ndarray]:
    key = (*_mesh_geometry_cache_key(mesh), "slip_velocity_reconstruction")
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cell_faces = np.asarray(mesh.cell_to_faces, dtype=np.int64)
        n_cells = int(mesh.tetrahedra.shape[0])
        if cell_faces.ndim != 2 or cell_faces.shape != (n_cells, 4):
            raise ValueError(
                "Imported tetra mesh cell_to_faces must have shape (N, 4)."
            )
        cell_ids = np.arange(n_cells, dtype=np.int64)[:, None]
        face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        face_areas = np.maximum(np.asarray(mesh.face_areas, dtype=np.float64), 1e-16)
        owner = face_to_cells[cell_faces, 0]
        owner_oriented = owner == cell_ids
        orientation_sign = np.where(owner_oriented, 1.0, -1.0)
        oriented_normals = np.where(
            owner_oriented[:, :, None],
            face_normals[cell_faces],
            -face_normals[cell_faces],
        )
        weights = face_areas[cell_faces]
        matrix = (
            np.einsum(
                "cf,cfi,cfj->cij",
                weights,
                oriented_normals,
                oriented_normals,
                optimize=True,
            )
            + 1e-14 * np.eye(3, dtype=np.float64)[None, :, :]
        )
        rhs_map = np.swapaxes(oriented_normals, 1, 2)
        try:
            inverse_matrix = np.linalg.inv(matrix)
            reconstruction_map = np.einsum(
                "cij,cjf->cif", inverse_matrix, rhs_map, optimize=True
            )
        except np.linalg.LinAlgError:
            reconstruction_map = np.zeros((n_cells, 3, 4), dtype=np.float64)
            for cell_idx in range(n_cells):
                try:
                    reconstruction_map[cell_idx] = (
                        np.linalg.inv(matrix[cell_idx]) @ rhs_map[cell_idx]
                    )
                except np.linalg.LinAlgError:
                    reconstruction_map[cell_idx] = 0.0
        cached = {
            "cell_faces": cell_faces,
            "orientation_sign": np.asarray(orientation_sign, dtype=np.float64),
            "reconstruction_map": np.asarray(reconstruction_map, dtype=np.float64),
        }
        _GEOMETRY_CACHE[key] = cached
    return cached


def _face_flux_viscous_predictor_step_torch(
    mesh: ImportedTetraMesh,
    *,
    face_flux: np.ndarray,
    nu: float,
    substep_dt: float,
    substeps: int,
    divergence_impact_cap: float,
    device: str,
) -> tuple[np.ndarray, int, int, np.ndarray, bool]:
    """Apply the existing vectorized slip face-flux Laplacian on CUDA."""
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Torch viscosity requested but torch is unavailable."
        ) from exc

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("Torch viscosity acceleration requires a CUDA device.")
    dtype = torch.float64
    face_ids, nb_mat, w_mat, w_sum = _cached_face_flux_laplacian_vector_stencil(mesh)
    if face_ids.size == 0:
        return (
            np.asarray(face_flux, dtype=np.float64).copy(),
            0,
            0,
            np.zeros((0,), dtype=np.int64),
            False,
        )

    cache_key = (
        *_mesh_geometry_cache_key(mesh),
        "face_flux_viscous_laplacian",
        id(face_ids),
        id(nb_mat),
        id(w_mat),
        id(w_sum),
        str(dev),
        str(dtype),
    )
    geometry = _TORCH_VISCOSITY_CACHE.get(cache_key)
    if geometry is None:
        c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
        c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
        vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
        active_min_volume = np.minimum(vol[c0[face_ids]], vol[c1[face_ids]])
        valid_neighbors = np.asarray(w_mat, dtype=np.float64) > 0.0
        row_counts = np.count_nonzero(valid_neighbors, axis=1).astype(np.int64)
        crow_indices = np.zeros((face_ids.size + 1,), dtype=np.int64)
        crow_indices[1:] = np.cumsum(row_counts, dtype=np.int64)
        column_indices = np.asarray(nb_mat, dtype=np.int64)[valid_neighbors]
        normalized_weights = (
            np.asarray(w_mat, dtype=np.float64)
            / np.maximum(np.asarray(w_sum, dtype=np.float64), 1e-30)[:, None]
        )[valid_neighbors]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            normalized_stencil = torch.sparse_csr_tensor(
                torch.as_tensor(crow_indices, dtype=torch.long, device=dev),
                torch.as_tensor(column_indices, dtype=torch.long, device=dev),
                torch.as_tensor(normalized_weights, dtype=dtype, device=dev),
                size=(int(face_ids.size), int(mesh.face_vertices.shape[0])),
                dtype=dtype,
                device=dev,
            )
        geometry = {
            "face_ids": torch.as_tensor(face_ids, dtype=torch.long, device=dev),
            "normalized_stencil": normalized_stencil,
            "active_min_volume": torch.as_tensor(
                active_min_volume, dtype=dtype, device=dev
            ),
        }
        _TORCH_VISCOSITY_CACHE[cache_key] = geometry

    cached_input = _find_torch_array(face_flux, device=str(dev))
    input_device_resident_reused = bool(cached_input is not None)
    if cached_input is None:
        q = torch.as_tensor(
            np.asarray(face_flux, dtype=np.float64), dtype=dtype, device=dev
        ).clone()
    else:
        q = cached_input.clone()
    active = geometry["face_ids"]
    normalized_stencil = geometry["normalized_stencil"]
    dq_cap = float(divergence_impact_cap) * geometry["active_min_volume"]
    capped_any = torch.zeros_like(active, dtype=torch.bool)
    capped_updates = torch.zeros((), dtype=torch.int64, device=dev)
    update_scale = float(nu) * float(substep_dt)
    for _ in range(int(substeps)):
        q_face = q[active]
        neighbor_average = torch.sparse.mm(normalized_stencil, q[:, None])[:, 0]
        lap_q = neighbor_average - q_face
        dq = update_scale * lap_q
        capped = torch.abs(dq) > dq_cap
        capped_any |= capped
        capped_updates += torch.count_nonzero(capped)
        dq = torch.minimum(torch.maximum(dq, -dq_cap), dq_cap)
        q_next = q.clone()
        q_next[active] = q_face + dq
        q = q_next

    capped_face_ids = active[capped_any]
    q_numpy = q.detach().cpu().numpy()
    _remember_torch_array(q_numpy, q, device=str(dev))
    return (
        q_numpy,
        int(capped_updates.detach().cpu().item()),
        int(face_ids.size * int(substeps)),
        capped_face_ids.detach().cpu().numpy(),
        bool(input_device_resident_reused),
    )


def _finalize_slip_viscous_predictor_torch(
    mesh: ImportedTetraMesh,
    *,
    face_flux_raw: np.ndarray,
    inlet_speed: float,
    left_inlet_faces: np.ndarray,
    right_inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    outlet_contract_mode: ViscousPredictorOutletContractMode,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Finish the supported slip predictor on CUDA without a CPU round trip."""
    q_raw = _find_torch_array(face_flux_raw, device=str(device))
    if q_raw is None:
        return None
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        return None

    dev = torch.device(device)
    dtype = torch.float64
    reconstruction = _cached_slip_velocity_reconstruction_geometry(mesh)
    cache_key = (
        *_mesh_geometry_cache_key(mesh),
        "slip_velocity_reconstruction_torch",
        id(reconstruction["cell_faces"]),
        str(dev),
        str(dtype),
    )
    geometry = _TORCH_VISCOSITY_CACHE.get(cache_key)
    if geometry is None:
        geometry = {
            "cell_faces": torch.as_tensor(
                reconstruction["cell_faces"], dtype=torch.long, device=dev
            ),
            "orientation_sign": torch.as_tensor(
                reconstruction["orientation_sign"], dtype=dtype, device=dev
            ),
            "reconstruction_map": torch.as_tensor(
                reconstruction["reconstruction_map"], dtype=dtype, device=dev
            ),
            "face_areas": torch.as_tensor(
                np.asarray(mesh.face_areas, dtype=np.float64),
                dtype=dtype,
                device=dev,
            ),
            "left_inlet_faces": torch.as_tensor(
                np.asarray(left_inlet_faces, dtype=np.int64),
                dtype=torch.long,
                device=dev,
            ),
            "right_inlet_faces": torch.as_tensor(
                np.asarray(right_inlet_faces, dtype=np.int64),
                dtype=torch.long,
                device=dev,
            ),
            "outlet_faces": torch.as_tensor(
                np.asarray(outlet_faces, dtype=np.int64),
                dtype=torch.long,
                device=dev,
            ),
            "wall_faces": torch.as_tensor(
                np.asarray(wall_faces, dtype=np.int64),
                dtype=torch.long,
                device=dev,
            ),
        }
        _TORCH_VISCOSITY_CACHE[cache_key] = geometry

    def _reconstruct(q: Any) -> Any:
        signed_flux = q[geometry["cell_faces"]] * geometry["orientation_sign"]
        return torch.sum(
            geometry["reconstruction_map"] * signed_flux[:, None, :], dim=2
        )

    velocity_raw = _reconstruct(q_raw)
    q_contract = q_raw.clone()
    for inlet_key in ("left_inlet_faces", "right_inlet_faces"):
        inlet_ids = geometry[inlet_key]
        if int(inlet_ids.numel()) > 0:
            q_contract[inlet_ids] = (
                -float(inlet_speed) * geometry["face_areas"][inlet_ids]
            )
    wall_ids = geometry["wall_faces"]
    if int(wall_ids.numel()) > 0:
        q_contract[wall_ids] = 0.0
    outlet_ids = geometry["outlet_faces"]
    if int(outlet_ids.numel()) > 0 and str(outlet_contract_mode) == "match_inlet":
        inlet_ids = torch.cat(
            (geometry["left_inlet_faces"], geometry["right_inlet_faces"])
        )
        inlet_flux = torch.sum(torch.clamp(-q_contract[inlet_ids], min=0.0))
        outlet_area = torch.sum(geometry["face_areas"][outlet_ids])
        outlet_speed = inlet_flux / torch.clamp(outlet_area, min=1e-20)
        q_contract[outlet_ids] = outlet_speed * geometry["face_areas"][outlet_ids]
    velocity_contract = _reconstruct(q_contract)

    q_contract_numpy = q_contract.detach().cpu().numpy()
    velocity_raw_numpy = velocity_raw.detach().cpu().numpy()
    velocity_contract_numpy = velocity_contract.detach().cpu().numpy()
    _remember_torch_array(q_contract_numpy, q_contract, device=str(dev))
    _remember_torch_array(velocity_contract_numpy, velocity_contract, device=str(dev))
    return q_contract_numpy, velocity_raw_numpy, velocity_contract_numpy


def _conservative_no_slip_viscous_predictor_step_torch(
    mesh: ImportedTetraMesh,
    *,
    velocity: np.ndarray,
    nu: float,
    substep_dt: float,
    substeps: int,
    inlet_source: np.ndarray,
    wall_sink: np.ndarray,
    inlet_sink: np.ndarray,
    wall_face_velocity: np.ndarray,
    inlet_face_velocity: np.ndarray,
    outlet_normal_gradient: np.ndarray,
    nonorthogonal_geometry: dict[str, np.ndarray | float],
    device: str,
) -> dict[str, Any]:
    """Run the established conservative no-slip velocity update on CUDA."""
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Torch no-slip viscosity requested but torch is unavailable."
        ) from exc

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("Torch no-slip viscosity acceleration requires a CUDA device.")
    dtype = torch.float64
    tpfa = _cached_viscous_tpfa_geometry(mesh)
    cache_key = (
        *_mesh_geometry_cache_key(mesh),
        "conservative_no_slip_viscosity",
        id(nonorthogonal_geometry),
        str(dev),
        str(dtype),
    )
    geometry = _TORCH_VISCOSITY_CACHE.get(cache_key)
    if geometry is None:
        owner = np.asarray(tpfa["owner"], dtype=np.int64)
        neighbor = np.asarray(tpfa["neighbor"], dtype=np.int64)
        volume = np.asarray(tpfa["volume"], dtype=np.float64)
        weight = np.asarray(tpfa["weight"], dtype=np.float64)

        def _long(key: str) -> Any:
            return torch.as_tensor(
                np.asarray(nonorthogonal_geometry[key], dtype=np.int64),
                dtype=torch.long,
                device=dev,
            )

        def _float(key: str) -> Any:
            return torch.as_tensor(
                np.asarray(nonorthogonal_geometry[key], dtype=np.float64),
                dtype=dtype,
                device=dev,
            )

        geometry = {
            "owner": torch.as_tensor(owner, dtype=torch.long, device=dev),
            "neighbor": torch.as_tensor(neighbor, dtype=torch.long, device=dev),
            "face_owner": torch.as_tensor(
                np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64),
                dtype=torch.long,
                device=dev,
            ),
            "owner_coefficient": torch.as_tensor(
                weight / volume[owner], dtype=dtype, device=dev
            ),
            "neighbor_coefficient": torch.as_tensor(
                weight / volume[neighbor], dtype=dtype, device=dev
            ),
            "inverse_volume": torch.as_tensor(1.0 / volume, dtype=dtype, device=dev),
            "volume": torch.as_tensor(volume, dtype=dtype, device=dev),
            "interior_faces": _long("interior_faces"),
            "interior_owner": _long("interior_owner"),
            "interior_neighbor": _long("interior_neighbor"),
            "interior_delta": _float("interior_delta"),
            "interior_weight": _float("interior_weight"),
            "interior_lambda": _float("interior_lambda"),
            "interior_k": _float("interior_k"),
            "wall_faces": _long("wall_faces"),
            "wall_owner": _long("wall_owner"),
            "wall_delta": _float("wall_delta"),
            "wall_weight": _float("wall_weight"),
            "wall_k": _float("wall_k"),
            "inlet_faces": _long("inlet_faces"),
            "inlet_owner": _long("inlet_owner"),
            "inlet_delta": _float("inlet_delta"),
            "inlet_weight": _float("inlet_weight"),
            "inlet_k": _float("inlet_k"),
            "outlet_faces": _long("outlet_faces"),
            "outlet_owner": _long("outlet_owner"),
            "outlet_normal": _float("outlet_normal"),
            "outlet_area": _float("outlet_area"),
            "lsq_pseudoinverse": _float("lsq_pseudoinverse"),
            "n_faces": int(np.asarray(nonorthogonal_geometry["n_faces"]).item()),
        }
        _TORCH_VISCOSITY_CACHE[cache_key] = geometry

    cached_velocity = _find_torch_array(velocity, device=str(dev))
    input_device_resident_reused = bool(cached_velocity is not None)
    if cached_velocity is None:
        initial = torch.as_tensor(
            np.asarray(velocity, dtype=np.float64), dtype=dtype, device=dev
        )
    else:
        initial = cached_velocity
    v_pred = initial.clone()
    v_pred_no_wall = initial.clone()
    inlet_source_t = torch.as_tensor(
        np.asarray(inlet_source, dtype=np.float64), dtype=dtype, device=dev
    )
    wall_sink_t = torch.as_tensor(
        np.asarray(wall_sink, dtype=np.float64), dtype=dtype, device=dev
    )
    inlet_sink_t = torch.as_tensor(
        np.asarray(inlet_sink, dtype=np.float64), dtype=dtype, device=dev
    )
    wall_face_velocity_t = torch.as_tensor(
        np.asarray(wall_face_velocity, dtype=np.float64), dtype=dtype, device=dev
    )
    inlet_face_velocity_t = torch.as_tensor(
        np.asarray(inlet_face_velocity, dtype=np.float64), dtype=dtype, device=dev
    )
    outlet_normal_gradient_t = torch.as_tensor(
        np.asarray(outlet_normal_gradient, dtype=np.float64), dtype=dtype, device=dev
    )
    update_scale = float(nu) * float(substep_dt)
    boundary_denominator = 1.0 + update_scale * (wall_sink_t + inlet_sink_t)
    update_max = torch.zeros((), dtype=dtype, device=dev)
    update_l2 = torch.zeros((), dtype=dtype, device=dev)
    operator_energy_rate = torch.zeros((), dtype=dtype, device=dev)
    nonorthogonal_flux = torch.zeros(
        (int(geometry["n_faces"]), 3), dtype=dtype, device=dev
    )
    nonorthogonal_gradient = torch.zeros(
        (int(v_pred.shape[0]), 3, 3), dtype=dtype, device=dev
    )
    nonorthogonal_laplacian = torch.zeros_like(v_pred)

    owner = geometry["owner"]
    neighbor = geometry["neighbor"]
    for _ in range(int(substeps)):
        lap_no_wall = torch.zeros_like(v_pred_no_wall)
        if int(owner.numel()) > 0:
            delta_no_wall = v_pred_no_wall[neighbor] - v_pred_no_wall[owner]
            lap_no_wall.index_add_(
                0, owner, geometry["owner_coefficient"][:, None] * delta_no_wall
            )
            lap_no_wall.index_add_(
                0,
                neighbor,
                -geometry["neighbor_coefficient"][:, None] * delta_no_wall,
            )
        v_pred_no_wall = v_pred_no_wall + update_scale * lap_no_wall

        lap = torch.zeros_like(v_pred)
        if int(owner.numel()) > 0:
            delta = v_pred[neighbor] - v_pred[owner]
            lap.index_add_(0, owner, geometry["owner_coefficient"][:, None] * delta)
            lap.index_add_(
                0, neighbor, -geometry["neighbor_coefficient"][:, None] * delta
            )

        rhs = torch.zeros(
            (int(v_pred.shape[0]), int(v_pred.shape[1]), 3),
            dtype=dtype,
            device=dev,
        )
        interior_owner = geometry["interior_owner"]
        interior_neighbor = geometry["interior_neighbor"]
        if int(interior_owner.numel()) > 0:
            velocity_delta = v_pred[interior_neighbor] - v_pred[interior_owner]
            contribution = (
                geometry["interior_weight"][:, None, None]
                * velocity_delta[:, :, None]
                * geometry["interior_delta"][:, None, :]
            )
            rhs.index_add_(0, interior_owner, contribution)
            rhs.index_add_(0, interior_neighbor, contribution)
        for boundary_owner, boundary_value, delta_key, weight_key in (
            (
                geometry["wall_owner"],
                wall_face_velocity_t,
                "wall_delta",
                "wall_weight",
            ),
            (
                geometry["inlet_owner"],
                inlet_face_velocity_t,
                "inlet_delta",
                "inlet_weight",
            ),
        ):
            if int(boundary_owner.numel()) > 0:
                value_delta = boundary_value - v_pred[boundary_owner]
                contribution = (
                    geometry[weight_key][:, None, None]
                    * value_delta[:, :, None]
                    * geometry[delta_key][:, None, :]
                )
                rhs.index_add_(0, boundary_owner, contribution)
        outlet_owner = geometry["outlet_owner"]
        if int(outlet_owner.numel()) > 0:
            contribution = (
                outlet_normal_gradient_t[:, :, None]
                * geometry["outlet_normal"][:, None, :]
            )
            rhs.index_add_(0, outlet_owner, contribution)
        nonorthogonal_gradient = torch.einsum(
            "nij,ncj->nci", geometry["lsq_pseudoinverse"], rhs
        )

        nonorthogonal_flux = torch.zeros(
            (int(geometry["n_faces"]), int(v_pred.shape[1])),
            dtype=dtype,
            device=dev,
        )
        interior_faces = geometry["interior_faces"]
        if int(interior_faces.numel()) > 0:
            interpolation = geometry["interior_lambda"]
            face_gradient = (1.0 - interpolation)[
                :, None, None
            ] * nonorthogonal_gradient[interior_owner] + interpolation[
                :, None, None
            ] * nonorthogonal_gradient[interior_neighbor]
            nonorthogonal_flux[interior_faces] = torch.einsum(
                "fj,fcj->fc", geometry["interior_k"], face_gradient
            )
        for ids_key, owner_key, k_key in (
            ("wall_faces", "wall_owner", "wall_k"),
            ("inlet_faces", "inlet_owner", "inlet_k"),
        ):
            face_ids = geometry[ids_key]
            if int(face_ids.numel()) > 0:
                nonorthogonal_flux[face_ids] = torch.einsum(
                    "fj,fcj->fc",
                    geometry[k_key],
                    nonorthogonal_gradient[geometry[owner_key]],
                )
        outlet_faces = geometry["outlet_faces"]
        if int(outlet_faces.numel()) > 0:
            nonorthogonal_flux[outlet_faces] = (
                geometry["outlet_area"][:, None] * outlet_normal_gradient_t
            )

        nonorthogonal_laplacian = torch.zeros_like(v_pred)
        face_owner = geometry["interior_owner"]
        face_neighbor = geometry["interior_neighbor"]
        all_face_owner = geometry["face_owner"]
        nonorthogonal_laplacian.index_add_(
            0,
            all_face_owner,
            nonorthogonal_flux * geometry["inverse_volume"][all_face_owner][:, None],
        )
        if int(face_owner.numel()) > 0:
            nonorthogonal_laplacian.index_add_(
                0,
                face_neighbor,
                -nonorthogonal_flux[interior_faces]
                * geometry["inverse_volume"][face_neighbor][:, None],
            )
        lap = lap + nonorthogonal_laplacian
        full_operator_laplacian = (
            lap + inlet_source_t - (wall_sink_t + inlet_sink_t)[:, None] * v_pred
        )
        operator_energy_rate = torch.sum(
            geometry["volume"][:, None] * v_pred * full_operator_laplacian
        )
        nonorthogonal_update_magnitude = torch.linalg.vector_norm(
            update_scale * nonorthogonal_laplacian, dim=1
        )
        if int(nonorthogonal_update_magnitude.numel()) > 0:
            update_max = torch.maximum(
                update_max, torch.max(nonorthogonal_update_magnitude)
            )
            update_l2 = torch.maximum(
                update_l2,
                torch.sqrt(
                    torch.mean(
                        nonorthogonal_update_magnitude * nonorthogonal_update_magnitude
                    )
                ),
            )
        explicit_velocity = v_pred + update_scale * lap
        v_pred = (
            explicit_velocity + update_scale * inlet_source_t
        ) / boundary_denominator[:, None]

    v_pred_numpy = v_pred.detach().cpu().numpy()
    _remember_torch_array(v_pred_numpy, v_pred, device=str(dev))
    return {
        "velocity": v_pred_numpy,
        "velocity_no_wall": v_pred_no_wall.detach().cpu().numpy(),
        "nonorthogonal_flux": nonorthogonal_flux.detach().cpu().numpy(),
        "nonorthogonal_laplacian": nonorthogonal_laplacian.detach().cpu().numpy(),
        "nonorthogonal_update_max": float(update_max.detach().cpu().item()),
        "nonorthogonal_update_l2": float(update_l2.detach().cpu().item()),
        "operator_energy_rate": float(operator_energy_rate.detach().cpu().item()),
        "input_device_resident_reused": input_device_resident_reused,
    }


def _compute_cell_flux_sum(
    mesh: ImportedTetraMesh, face_flux: np.ndarray
) -> np.ndarray:
    flux = np.asarray(face_flux, dtype=np.float64)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    n_cells = mesh.tetrahedra.shape[0]
    summed = np.zeros((n_cells,), dtype=np.float64)
    np.add.at(summed, c0, flux)
    interior = c1 >= 0
    if np.any(interior):
        np.add.at(summed, c1[interior], -flux[interior])
    return summed


def _compute_divergence_numpy(
    mesh: ImportedTetraMesh, face_flux: np.ndarray
) -> np.ndarray:
    summed = _compute_cell_flux_sum(mesh, face_flux)
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-16)
    return summed / vol


def _locally_conserve_interior_face_flux_cell_sums(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    target_cell_flux_sum: np.ndarray,
    eligible_cell_mask: np.ndarray | None = None,
    iterations: int = 8,
) -> tuple[np.ndarray, dict[str, float | int]]:
    out = np.asarray(face_flux, dtype=np.float64).copy()
    target = np.asarray(target_cell_flux_sum, dtype=np.float64)
    if out.size == 0 or target.size != mesh.tetrahedra.shape[0]:
        return out, {
            "iterations": 0,
            "residual_l2_before": 0.0,
            "residual_l2_after": 0.0,
            "residual_max_abs_before": 0.0,
            "residual_max_abs_after": 0.0,
            "delta_l2": 0.0,
            "delta_max_abs": 0.0,
        }

    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    owner = face_to_cells[:, 0]
    neigh = face_to_cells[:, 1]
    interior_faces = np.flatnonzero(neigh >= 0).astype(np.int64)
    if eligible_cell_mask is not None and interior_faces.size:
        eligible = np.asarray(eligible_cell_mask, dtype=bool)
        if eligible.size == mesh.tetrahedra.shape[0]:
            interior_faces = interior_faces[
                eligible[owner[interior_faces]] & eligible[neigh[interior_faces]]
            ]
    if interior_faces.size == 0:
        return out, {
            "iterations": 0,
            "residual_l2_before": 0.0,
            "residual_l2_after": 0.0,
            "residual_max_abs_before": 0.0,
            "residual_max_abs_after": 0.0,
            "delta_l2": 0.0,
            "delta_max_abs": 0.0,
        }

    n_cells = int(mesh.tetrahedra.shape[0])
    degree = np.zeros((n_cells,), dtype=np.float64)
    np.add.at(degree, owner[interior_faces], 1.0)
    np.add.at(degree, neigh[interior_faces], 1.0)
    degree = np.maximum(degree, 1.0)

    before = out.copy()
    residual = target - _compute_cell_flux_sum(mesh, out)
    residual_l2_before = float(np.sqrt(np.mean(residual * residual)))
    residual_max_before = float(np.max(np.abs(residual))) if residual.size else 0.0
    for _ in range(max(0, int(iterations))):
        residual = target - _compute_cell_flux_sum(mesh, out)
        corr = 0.5 * (
            residual[owner[interior_faces]] / degree[owner[interior_faces]]
            - residual[neigh[interior_faces]] / degree[neigh[interior_faces]]
        )
        out[interior_faces] = out[interior_faces] + corr
    residual_after = target - _compute_cell_flux_sum(mesh, out)
    delta = out - before
    return out, {
        "iterations": int(max(0, int(iterations))),
        "residual_l2_before": float(residual_l2_before),
        "residual_l2_after": float(np.sqrt(np.mean(residual_after * residual_after)))
        if residual_after.size
        else 0.0,
        "residual_max_abs_before": float(residual_max_before),
        "residual_max_abs_after": float(np.max(np.abs(residual_after)))
        if residual_after.size
        else 0.0,
        "delta_l2": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "delta_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _group_flux_metrics(face_flux: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    ids = np.asarray(faces, dtype=np.int64)
    if ids.size == 0:
        return {
            "face_count": 0.0,
            "sum_flux": 0.0,
            "inflow_total": 0.0,
            "outflow_total": 0.0,
            "max_abs": 0.0,
            "mean": 0.0,
        }
    vals = np.asarray(face_flux[ids], dtype=np.float64)
    return {
        "face_count": float(ids.size),
        "sum_flux": float(np.sum(vals)),
        "inflow_total": float(np.sum(np.maximum(-vals, 0.0))),
        "outflow_total": float(np.sum(np.maximum(vals, 0.0))),
        "max_abs": float(np.max(np.abs(vals))),
        "mean": float(np.mean(vals)),
    }


def _boundary_flux_audit(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    left_inlet_faces: np.ndarray,
    right_inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> dict[str, Any]:
    boundary_faces = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    group = {
        "left_inlet": _group_flux_metrics(face_flux, left_inlet_faces),
        "right_inlet": _group_flux_metrics(face_flux, right_inlet_faces),
        "outlet": _group_flux_metrics(face_flux, outlet_faces),
        "walls": _group_flux_metrics(face_flux, wall_faces),
        "boundary_total": _group_flux_metrics(face_flux, boundary_faces),
    }

    tag_map: dict[str, Any] = {}
    if boundary_faces.size:
        tags = np.asarray(mesh.boundary_tag_per_face[boundary_faces], dtype=np.int32)
        for tag in np.unique(tags).tolist():
            if int(tag) < 0:
                continue
            faces = boundary_faces[tags == int(tag)]
            name = mesh.boundary_face_names.get(int(tag), f"tag_{int(tag)}")
            tag_map[str(name)] = _group_flux_metrics(face_flux, faces)
            tag_map[str(name)]["tag"] = int(tag)
    group["per_physical_group"] = tag_map
    return group


def _build_region_masks(
    mesh: ImportedTetraMesh,
    *,
    left_inlet_faces: np.ndarray,
    right_inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> dict[str, np.ndarray]:
    n_cells = mesh.tetrahedra.shape[0]
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    boundary_faces = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    boundary_adj = np.zeros((n_cells,), dtype=bool)
    inlet_adj = np.zeros((n_cells,), dtype=bool)
    outlet_adj = np.zeros((n_cells,), dtype=bool)
    wall_adj = np.zeros((n_cells,), dtype=bool)

    if boundary_faces.size:
        boundary_adj[np.unique(c0[boundary_faces])] = True
    inlet_faces = np.unique(np.concatenate((left_inlet_faces, right_inlet_faces)))
    if inlet_faces.size:
        inlet_adj[np.unique(c0[inlet_faces])] = True
    if outlet_faces.size:
        outlet_adj[np.unique(c0[np.asarray(outlet_faces, dtype=np.int64)])] = True
    if wall_faces.size:
        wall_adj[np.unique(c0[np.asarray(wall_faces, dtype=np.int64)])] = True

    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    y_in = (
        float(np.median(mesh.face_centers[inlet_faces, 1]))
        if inlet_faces.size
        else float(np.min(y))
    )
    y_out = (
        float(np.median(mesh.face_centers[np.asarray(outlet_faces, dtype=np.int64), 1]))
        if outlet_faces.size
        else float(np.max(y))
    )
    y_span = max(y_out - y_in, 1e-12)
    junction_y = y_in + 0.12 * y_span
    x_abs = np.abs(x)
    x_thresh = float(np.percentile(x_abs, 35.0)) if x_abs.size else 0.0
    junction = (np.abs(y - junction_y) <= 0.12 * y_span) & (
        x_abs <= max(x_thresh, 1e-12)
    )
    if not np.any(junction):
        junction = np.abs(y - junction_y) <= 0.16 * y_span

    interior_core = ~boundary_adj
    return {
        "interior_core": interior_core,
        "boundary_adjacent": boundary_adj,
        "inlet_adjacent": inlet_adj,
        "outlet_adjacent": outlet_adj,
        "wall_adjacent": wall_adj,
        "junction_zone": junction,
    }


def _masked_divergence_stats(div: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(div, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return {"count": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "l2": 0.0}
    return {
        "count": float(values.size),
        "max_abs": float(np.max(np.abs(values))),
        "mean_abs": float(np.mean(np.abs(values))),
        "l2": float(np.sqrt(np.mean(values * values))),
    }


def _region_divergence_audit(
    div_star: np.ndarray,
    div_corr: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        s = _masked_divergence_stats(div_star, mask)
        c = _masked_divergence_stats(div_corr, mask)
        out[name] = {
            "star": s,
            "corrected": c,
            "reduction_ratio_linf": _safe_ratio(c["max_abs"], s["max_abs"]),
            "reduction_ratio_l2": _safe_ratio(c["l2"], s["l2"]),
        }
    return out


def compute_tetra_flux_divergence(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
    *,
    left_inlet_faces: np.ndarray | None = None,
    right_inlet_faces: np.ndarray | None = None,
    outlet_faces: np.ndarray | None = None,
    wall_faces: np.ndarray | None = None,
) -> dict[str, Any]:
    left = np.asarray(
        left_inlet_faces if left_inlet_faces is not None else [], dtype=np.int64
    )
    right = np.asarray(
        right_inlet_faces if right_inlet_faces is not None else [], dtype=np.int64
    )
    outlet = np.asarray(
        outlet_faces if outlet_faces is not None else [], dtype=np.int64
    )
    wall = np.asarray(wall_faces if wall_faces is not None else [], dtype=np.int64)

    flux = np.asarray(face_flux, dtype=np.float64)
    div = _compute_divergence_numpy(mesh, flux)
    boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    inlet = (
        np.unique(np.concatenate((left, right)))
        if (left.size or right.size)
        else np.zeros((0,), dtype=np.int64)
    )

    inlet_flux = float(np.sum(np.maximum(-flux[inlet], 0.0))) if inlet.size else 0.0
    outlet_flux = float(np.sum(np.maximum(flux[outlet], 0.0))) if outlet.size else 0.0
    net_boundary_flux = float(np.sum(flux[boundary])) if boundary.size else 0.0
    wall_flux_max_abs = float(np.max(np.abs(flux[wall]))) if wall.size else 0.0

    return {
        "divergence": div,
        "divergence_max_abs": float(np.max(np.abs(div))) if div.size else 0.0,
        "divergence_mean_abs": float(np.mean(np.abs(div))) if div.size else 0.0,
        "divergence_l2": float(np.sqrt(np.mean(div**2))) if div.size else 0.0,
        "net_boundary_flux": net_boundary_flux,
        "wall_flux_max_abs": wall_flux_max_abs,
        "inlet_flux_total": inlet_flux,
        "outlet_flux_total": outlet_flux,
    }


def initialize_tetra_flow_state(
    mesh: ImportedTetraMesh,
    config: TetraFlowConfig,
) -> TetraFlowState:
    requested_config = config
    config = resolve_tetra_flow_numerical_profile(config)
    _validate_config(mesh, config)
    n_cells = mesh.tetrahedra.shape[0]
    n_faces = mesh.face_vertices.shape[0]
    left_faces, right_faces = _inlet_face_sets(mesh)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    face_flux = np.zeros((n_faces,), dtype=np.float64)
    bc_diag = _apply_face_flux_boundary_conditions_inplace(
        mesh,
        face_flux,
        inlet_speed=float(config.inlet_speed),
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    pressure = np.zeros((n_cells,), dtype=np.float64)
    cell_velocity = _reconstruct_cell_velocity_from_face_flux_numpy(
        mesh,
        face_flux,
        wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
        wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
    )
    div_diag = compute_tetra_flux_divergence(
        mesh,
        face_flux,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    diagnostics = {
        "flow_solved": False,
        "pressure_solved": False,
        "numerical_profile_resolution": _numerical_profile_resolution_diagnostics(
            requested_config,
            config,
        ),
        "initialization": {
            "inlet_speed": float(config.inlet_speed),
            "boundary_flux": bc_diag,
        },
        "divergence": {
            "initial_max_abs": float(div_diag["divergence_max_abs"]),
            "initial_l2": float(div_diag["divergence_l2"]),
        },
    }
    return TetraFlowState(
        cell_velocity=cell_velocity,
        face_flux=face_flux,
        pressure=pressure,
        diagnostics=diagnostics,
    )


def _build_pressure_system_coefficients(
    mesh: ImportedTetraMesh,
    *,
    dt: float,
    density: float,
    outlet_faces: np.ndarray,
) -> dict[str, Any]:
    n_cells = mesh.tetrahedra.shape[0]
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    area = np.asarray(mesh.face_areas, dtype=np.float64)
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    k_scale = float(dt / density)

    outlet_ids = np.unique(np.asarray(outlet_faces, dtype=np.int64).reshape(-1))
    outlet_ids = outlet_ids[
        (outlet_ids >= 0) & (outlet_ids < int(mesh.face_vertices.shape[0]))
    ]
    outlet_ids = outlet_ids[c1[outlet_ids] < 0]
    geometry_key = (
        *_mesh_geometry_cache_key(mesh),
        "pressure_system_geometry",
        tuple(int(fid) for fid in outlet_ids.tolist()),
    )
    geometry = _GEOMETRY_CACHE.get(geometry_key)
    if geometry is None:
        int_face = np.flatnonzero(c1 >= 0).astype(np.int64)
        int_owner = np.asarray(c0[int_face], dtype=np.int64)
        int_neigh = np.asarray(c1[int_face], dtype=np.int64)
        int_distance = np.maximum(
            np.linalg.norm(centers[int_neigh] - centers[int_owner], axis=1),
            1e-12,
        )
        base_int_k = np.asarray(area[int_face] / int_distance, dtype=np.float64)

        out_face = np.asarray(outlet_ids, dtype=np.int64)
        out_owner = np.asarray(c0[out_face], dtype=np.int64)
        out_distance = np.maximum(
            np.linalg.norm(face_centers[out_face] - centers[out_owner], axis=1),
            1e-12,
        )
        base_out_k = np.asarray(area[out_face] / out_distance, dtype=np.float64)

        base_diag = np.zeros((n_cells,), dtype=np.float64)
        if int_owner.size:
            np.add.at(base_diag, int_owner, base_int_k)
            np.add.at(base_diag, int_neigh, base_int_k)
        if out_owner.size:
            np.add.at(base_diag, out_owner, base_out_k)
        geometry = {
            "base_diag": base_diag,
            "int_face": int_face,
            "int_owner": int_owner,
            "int_neigh": int_neigh,
            "base_int_k": base_int_k,
            "out_owner": out_owner,
            "base_out_k": base_out_k,
            "out_face": out_face,
        }
        _GEOMETRY_CACHE[geometry_key] = geometry

    return {
        "diag": np.asarray(geometry["base_diag"] * k_scale, dtype=np.float64),
        "int_face": geometry["int_face"],
        "int_owner": geometry["int_owner"],
        "int_neigh": geometry["int_neigh"],
        "int_k": np.asarray(geometry["base_int_k"] * k_scale, dtype=np.float64),
        "out_owner": geometry["out_owner"],
        "out_k": np.asarray(geometry["base_out_k"] * k_scale, dtype=np.float64),
        "out_face": geometry["out_face"],
        "base_diag": geometry["base_diag"],
        "base_int_k": geometry["base_int_k"],
        "k_scale": k_scale,
    }


def _build_pressure_nonorthogonal_geometry(
    mesh: ImportedTetraMesh,
    outlet_faces: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Precompute vectorized LSQ and ``K=S-T*d`` pressure geometry."""

    started = perf_counter()
    n_cells = int(mesh.tetrahedra.shape[0])
    n_faces = int(mesh.face_vertices.shape[0])
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    owner_all = face_to_cells[:, 0]
    neighbor_all = face_to_cells[:, 1]
    interior_faces = np.flatnonzero(neighbor_all >= 0).astype(np.int64)
    interior_owner = owner_all[interior_faces]
    interior_neighbor = neighbor_all[interior_faces]

    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_areas = np.asarray(mesh.face_areas, dtype=np.float64)

    interior_delta = centers[interior_neighbor] - centers[interior_owner]
    interior_distance2 = np.maximum(
        np.einsum("ij,ij->i", interior_delta, interior_delta, optimize=True),
        1e-24,
    )
    interior_distance = np.sqrt(interior_distance2)
    interior_weight = 1.0 / interior_distance2
    interior_t = face_areas[interior_faces] / interior_distance
    interior_s = face_areas[interior_faces, None] * face_normals[interior_faces]
    interior_nonorthogonal_vector = interior_s - interior_t[:, None] * interior_delta
    face_plane_denominator = np.einsum(
        "ij,ij->i",
        interior_delta,
        face_normals[interior_faces],
        optimize=True,
    )
    safe_plane_denominator = np.where(
        np.abs(face_plane_denominator) > 1e-30,
        face_plane_denominator,
        np.where(face_plane_denominator < 0.0, -1e-30, 1e-30),
    )
    interior_lambda = (
        np.einsum(
            "ij,ij->i",
            face_centers[interior_faces] - centers[interior_owner],
            face_normals[interior_faces],
            optimize=True,
        )
        / safe_plane_denominator
    )

    outlet_ids = np.asarray(outlet_faces, dtype=np.int64).reshape(-1)
    outlet_owner = owner_all[outlet_ids]
    outlet_delta = face_centers[outlet_ids] - centers[outlet_owner]
    outlet_distance2 = np.maximum(
        np.einsum("ij,ij->i", outlet_delta, outlet_delta, optimize=True),
        1e-24,
    )
    outlet_distance = np.sqrt(outlet_distance2)
    outlet_weight = 1.0 / outlet_distance2
    outlet_t = face_areas[outlet_ids] / outlet_distance
    outlet_s = face_areas[outlet_ids, None] * face_normals[outlet_ids]
    outlet_nonorthogonal_vector = outlet_s - outlet_t[:, None] * outlet_delta

    boundary_mask = neighbor_all < 0
    if outlet_ids.size:
        boundary_mask[outlet_ids] = False
    neumann_faces = np.flatnonzero(boundary_mask).astype(np.int64)
    neumann_owner = owner_all[neumann_faces]
    neumann_normal = face_normals[neumann_faces]

    normal_matrix = np.zeros((n_cells, 3, 3), dtype=np.float64)
    interior_outer = (
        interior_weight[:, None, None]
        * interior_delta[:, :, None]
        * interior_delta[:, None, :]
    )
    if interior_faces.size:
        np.add.at(normal_matrix, interior_owner, interior_outer)
        np.add.at(normal_matrix, interior_neighbor, interior_outer)
    if outlet_ids.size:
        outlet_outer = (
            outlet_weight[:, None, None]
            * outlet_delta[:, :, None]
            * outlet_delta[:, None, :]
        )
        np.add.at(normal_matrix, outlet_owner, outlet_outer)
    if neumann_faces.size:
        # The projection pins pressure-correction flux on every non-outlet
        # boundary face.  Add the matching homogeneous Neumann equations
        # grad(p) dot n = 0 to the LSQ reconstruction.  Without these rows,
        # boundary-cell gradients are underconstrained and the deferred
        # fixed-point spectrum becomes strongly mesh dependent.
        neumann_outer = neumann_normal[:, :, None] * neumann_normal[:, None, :]
        np.add.at(normal_matrix, neumann_owner, neumann_outer)
    normal_matrix_eigenvalues = np.linalg.eigvalsh(normal_matrix)
    normal_matrix_max_eigenvalue = np.maximum(
        normal_matrix_eigenvalues[:, -1],
        0.0,
    )
    normal_matrix_rank = np.sum(
        normal_matrix_eigenvalues
        > 1e-12 * np.maximum(normal_matrix_max_eigenvalue[:, None], 1e-30),
        axis=1,
    ).astype(np.int32)
    normal_matrix_min_effective_eigenvalue = np.maximum(
        normal_matrix_eigenvalues[:, 0],
        1e-30,
    )
    normal_matrix_condition = (
        normal_matrix_max_eigenvalue / normal_matrix_min_effective_eigenvalue
    )
    lsq_pseudoinverse = np.linalg.pinv(
        normal_matrix,
        rcond=1e-12,
        hermitian=True,
    )
    equation_count = np.zeros((n_cells,), dtype=np.int32)
    if interior_faces.size:
        np.add.at(equation_count, interior_owner, 1)
        np.add.at(equation_count, interior_neighbor, 1)
    if outlet_ids.size:
        np.add.at(equation_count, outlet_owner, 1)
    if neumann_faces.size:
        np.add.at(equation_count, neumann_owner, 1)

    return {
        "n_faces": np.asarray(n_faces, dtype=np.int64),
        "interior_faces": interior_faces,
        "interior_owner": interior_owner,
        "interior_neighbor": interior_neighbor,
        "interior_delta": interior_delta,
        "interior_weight": interior_weight,
        "interior_lambda": interior_lambda,
        "interior_nonorthogonal_vector": interior_nonorthogonal_vector,
        "outlet_faces": outlet_ids,
        "outlet_owner": outlet_owner,
        "outlet_delta": outlet_delta,
        "outlet_weight": outlet_weight,
        "outlet_nonorthogonal_vector": outlet_nonorthogonal_vector,
        "neumann_faces": neumann_faces,
        "neumann_owner": neumann_owner,
        "neumann_normal": neumann_normal,
        "lsq_pseudoinverse": lsq_pseudoinverse,
        "lsq_equation_count": equation_count,
        "lsq_normal_matrix_rank": normal_matrix_rank,
        "lsq_normal_matrix_condition": normal_matrix_condition,
        "lsq_normal_matrix_eigenvalues": normal_matrix_eigenvalues,
        "build_seconds": float(perf_counter() - started),
    }


def _cached_pressure_nonorthogonal_geometry(
    mesh: ImportedTetraMesh,
    outlet_faces: np.ndarray,
) -> dict[str, np.ndarray | float]:
    outlet_ids = np.asarray(outlet_faces, dtype=np.int64).reshape(-1)
    key = (
        *_mesh_geometry_cache_key(mesh),
        id(mesh.cell_centers),
        id(mesh.face_normals),
        id(mesh.face_areas),
        "pressure_nonorthogonal_lsq_geometry",
        hash(outlet_ids.tobytes()),
    )
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_pressure_nonorthogonal_geometry(mesh, outlet_ids)
        _GEOMETRY_CACHE[key] = cached
    return cached


def _build_viscous_nonorthogonal_geometry(
    mesh: ImportedTetraMesh,
    *,
    inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Build LSQ and ``K=S-T*d`` geometry for the vector viscous flux.

    The opt-in operator uses complete Dirichlet momentum conditions at no-slip
    walls and prescribed-velocity inlets.  The pressure outlet is homogeneous
    Neumann.  This is deliberately separate from the legacy tangential wall
    sink so ``mode='none'`` keeps the historical predictor bitwise unchanged.
    """

    started = perf_counter()
    n_cells = int(mesh.tetrahedra.shape[0])
    n_faces = int(mesh.face_vertices.shape[0])
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    owner_all = face_to_cells[:, 0]
    neighbor_all = face_to_cells[:, 1]
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    volumes = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)

    interior_faces = np.flatnonzero(neighbor_all >= 0).astype(np.int64)
    interior_owner = owner_all[interior_faces]
    interior_neighbor = neighbor_all[interior_faces]
    interior_delta = centers[interior_neighbor] - centers[interior_owner]
    interior_distance2 = np.maximum(
        np.einsum("ij,ij->i", interior_delta, interior_delta, optimize=True),
        1e-24,
    )
    interior_distance = np.sqrt(interior_distance2)
    interior_weight = 1.0 / interior_distance2
    interior_t = areas[interior_faces] / interior_distance
    interior_k = (
        areas[interior_faces, None] * normals[interior_faces]
        - interior_t[:, None] * interior_delta
    )
    face_plane_denominator = np.einsum(
        "ij,ij->i", interior_delta, normals[interior_faces], optimize=True
    )
    safe_plane_denominator = np.where(
        np.abs(face_plane_denominator) > 1e-30,
        face_plane_denominator,
        np.where(face_plane_denominator < 0.0, -1e-30, 1e-30),
    )
    interior_lambda = (
        np.einsum(
            "ij,ij->i",
            face_centers[interior_faces] - centers[interior_owner],
            normals[interior_faces],
            optimize=True,
        )
        / safe_plane_denominator
    )

    inlet_ids = np.asarray(inlet_faces, dtype=np.int64).reshape(-1)
    outlet_ids = np.asarray(outlet_faces, dtype=np.int64).reshape(-1)
    wall_ids = np.asarray(wall_faces, dtype=np.int64).reshape(-1)

    def dirichlet_geometry(ids: np.ndarray) -> tuple[np.ndarray, ...]:
        boundary_owner = owner_all[ids]
        boundary_delta = face_centers[ids] - centers[boundary_owner]
        distance2 = np.maximum(
            np.einsum("ij,ij->i", boundary_delta, boundary_delta, optimize=True),
            1e-24,
        )
        weight = 1.0 / distance2
        normal_distance = np.maximum(
            np.abs(np.einsum("ij,ij->i", boundary_delta, normals[ids], optimize=True)),
            1e-12,
        )
        t_coefficient = areas[ids] / normal_distance
        k_vector = (
            areas[ids, None] * normals[ids] - t_coefficient[:, None] * boundary_delta
        )
        return (
            boundary_owner,
            boundary_delta,
            weight,
            t_coefficient,
            k_vector,
        )

    (
        wall_owner,
        wall_delta,
        wall_weight,
        wall_t,
        wall_k,
    ) = dirichlet_geometry(wall_ids)
    (
        inlet_owner,
        inlet_delta,
        inlet_weight,
        inlet_t,
        inlet_k,
    ) = dirichlet_geometry(inlet_ids)
    outlet_owner = owner_all[outlet_ids]
    outlet_normal = normals[outlet_ids]

    normal_matrix = np.zeros((n_cells, 3, 3), dtype=np.float64)
    interior_outer = (
        interior_weight[:, None, None]
        * interior_delta[:, :, None]
        * interior_delta[:, None, :]
    )
    if interior_faces.size:
        np.add.at(normal_matrix, interior_owner, interior_outer)
        np.add.at(normal_matrix, interior_neighbor, interior_outer)
    for boundary_owner, boundary_delta, boundary_weight in (
        (wall_owner, wall_delta, wall_weight),
        (inlet_owner, inlet_delta, inlet_weight),
    ):
        if boundary_owner.size:
            boundary_outer = (
                boundary_weight[:, None, None]
                * boundary_delta[:, :, None]
                * boundary_delta[:, None, :]
            )
            np.add.at(normal_matrix, boundary_owner, boundary_outer)
    if outlet_ids.size:
        outlet_outer = outlet_normal[:, :, None] * outlet_normal[:, None, :]
        np.add.at(normal_matrix, outlet_owner, outlet_outer)

    eigenvalues = np.linalg.eigvalsh(normal_matrix)
    max_eigenvalue = np.maximum(eigenvalues[:, -1], 0.0)
    rank = np.sum(
        eigenvalues > 1e-12 * np.maximum(max_eigenvalue[:, None], 1e-30),
        axis=1,
    ).astype(np.int32)
    min_effective_eigenvalue = np.maximum(eigenvalues[:, 0], 1e-30)
    condition = max_eigenvalue / min_effective_eigenvalue
    pseudoinverse = np.linalg.pinv(
        normal_matrix,
        rcond=1e-12,
        hermitian=True,
    )

    equation_count = np.zeros((n_cells,), dtype=np.int32)
    if interior_faces.size:
        np.add.at(equation_count, interior_owner, 1)
        np.add.at(equation_count, interior_neighbor, 1)
    for boundary_owner in (wall_owner, inlet_owner, outlet_owner):
        if boundary_owner.size:
            np.add.at(equation_count, boundary_owner, 1)

    # Conservative induced-infinity-norm bound for the explicit K*grad term.
    # For ||phi||_inf <= 1, every interior difference is bounded by 2 and each
    # fixed Dirichlet boundary difference by 1.  Propagate those componentwise
    # bounds through the LSQ pseudoinverse and the oriented face correction.
    rhs_absolute_bound = np.zeros((n_cells, 3), dtype=np.float64)
    if interior_faces.size:
        term = interior_weight[:, None] * np.abs(interior_delta)
        np.add.at(rhs_absolute_bound, interior_owner, 2.0 * term)
        np.add.at(rhs_absolute_bound, interior_neighbor, 2.0 * term)
    for boundary_owner, boundary_delta, boundary_weight in (
        (wall_owner, wall_delta, wall_weight),
        (inlet_owner, inlet_delta, inlet_weight),
    ):
        if boundary_owner.size:
            term = boundary_weight[:, None] * np.abs(boundary_delta)
            np.add.at(rhs_absolute_bound, boundary_owner, term)
    gradient_absolute_bound = np.einsum(
        "nij,nj->ni",
        np.abs(pseudoinverse),
        rhs_absolute_bound,
        optimize=True,
    )
    correction_flux_bound = np.zeros((n_faces,), dtype=np.float64)
    if interior_faces.size:
        gradient_face_bound = (
            np.abs(1.0 - interior_lambda)[:, None]
            * gradient_absolute_bound[interior_owner]
            + np.abs(interior_lambda)[:, None]
            * gradient_absolute_bound[interior_neighbor]
        )
        correction_flux_bound[interior_faces] = np.einsum(
            "ij,ij->i", np.abs(interior_k), gradient_face_bound, optimize=True
        )
    for ids, boundary_owner, boundary_k in (
        (wall_ids, wall_owner, wall_k),
        (inlet_ids, inlet_owner, inlet_k),
    ):
        if ids.size:
            correction_flux_bound[ids] = np.einsum(
                "ij,ij->i",
                np.abs(boundary_k),
                gradient_absolute_bound[boundary_owner],
                optimize=True,
            )
    correction_laplacian_bound = np.zeros((n_cells,), dtype=np.float64)
    if interior_faces.size:
        np.add.at(
            correction_laplacian_bound,
            interior_owner,
            correction_flux_bound[interior_faces],
        )
        np.add.at(
            correction_laplacian_bound,
            interior_neighbor,
            correction_flux_bound[interior_faces],
        )
    for ids, boundary_owner in (
        (wall_ids, wall_owner),
        (inlet_ids, inlet_owner),
    ):
        if ids.size:
            np.add.at(
                correction_laplacian_bound,
                boundary_owner,
                correction_flux_bound[ids],
            )
    correction_laplacian_bound /= volumes

    interior_row_coefficient = np.zeros((n_cells,), dtype=np.float64)
    if interior_faces.size:
        np.add.at(
            interior_row_coefficient,
            interior_owner,
            interior_t / volumes[interior_owner],
        )
        np.add.at(
            interior_row_coefficient,
            interior_neighbor,
            interior_t / volumes[interior_neighbor],
        )
    wall_sink = np.zeros((n_cells,), dtype=np.float64)
    inlet_sink = np.zeros((n_cells,), dtype=np.float64)
    if wall_ids.size:
        np.add.at(wall_sink, wall_owner, wall_t / volumes[wall_owner])
    if inlet_ids.size:
        np.add.at(inlet_sink, inlet_owner, inlet_t / volumes[inlet_owner])

    return {
        "n_faces": np.asarray(n_faces, dtype=np.int64),
        "interior_faces": interior_faces,
        "interior_owner": interior_owner,
        "interior_neighbor": interior_neighbor,
        "interior_delta": interior_delta,
        "interior_weight": interior_weight,
        "interior_t": interior_t,
        "interior_k": interior_k,
        "interior_lambda": interior_lambda,
        "wall_faces": wall_ids,
        "wall_owner": wall_owner,
        "wall_delta": wall_delta,
        "wall_weight": wall_weight,
        "wall_t": wall_t,
        "wall_k": wall_k,
        "inlet_faces": inlet_ids,
        "inlet_owner": inlet_owner,
        "inlet_delta": inlet_delta,
        "inlet_weight": inlet_weight,
        "inlet_t": inlet_t,
        "inlet_k": inlet_k,
        "inlet_normal": normals[inlet_ids],
        "outlet_faces": outlet_ids,
        "outlet_owner": outlet_owner,
        "outlet_normal": outlet_normal,
        "outlet_area": areas[outlet_ids],
        "lsq_pseudoinverse": pseudoinverse,
        "lsq_equation_count": equation_count,
        "lsq_normal_matrix_rank": rank,
        "lsq_normal_matrix_condition": condition,
        "lsq_normal_matrix_eigenvalues": eigenvalues,
        "gradient_absolute_bound": gradient_absolute_bound,
        "correction_flux_infinity_norm_bound": correction_flux_bound,
        "correction_laplacian_infinity_norm_bound": correction_laplacian_bound,
        "interior_row_coefficient": interior_row_coefficient,
        "wall_orthogonal_sink_per_volume": wall_sink,
        "inlet_orthogonal_sink_per_volume": inlet_sink,
        "build_seconds": float(perf_counter() - started),
    }


def _cached_viscous_nonorthogonal_geometry(
    mesh: ImportedTetraMesh,
    *,
    inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> dict[str, np.ndarray | float]:
    inlet_ids = np.asarray(inlet_faces, dtype=np.int64).reshape(-1)
    outlet_ids = np.asarray(outlet_faces, dtype=np.int64).reshape(-1)
    wall_ids = np.asarray(wall_faces, dtype=np.int64).reshape(-1)
    key = (
        *_mesh_geometry_cache_key(mesh),
        id(mesh.cell_centers),
        id(mesh.face_normals),
        id(mesh.face_areas),
        "viscous_nonorthogonal_lsq_geometry",
        hash(inlet_ids.tobytes()),
        hash(outlet_ids.tobytes()),
        hash(wall_ids.tobytes()),
    )
    cached = _GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = _build_viscous_nonorthogonal_geometry(
            mesh,
            inlet_faces=inlet_ids,
            outlet_faces=outlet_ids,
            wall_faces=wall_ids,
        )
        _GEOMETRY_CACHE[key] = cached
    return cached


def _viscous_lsq_cell_gradient_numpy(
    velocity: np.ndarray,
    *,
    wall_face_velocity: np.ndarray,
    inlet_face_velocity: np.ndarray,
    outlet_normal_gradient: np.ndarray,
    geometry: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Reconstruct ``grad(u_component)`` in every cell by weighted LSQ."""

    vel = np.asarray(velocity, dtype=np.float64)
    if vel.ndim != 2:
        raise ValueError("velocity must be a two-dimensional cell array")
    n_cells, n_components = vel.shape
    wall_value = np.asarray(wall_face_velocity, dtype=np.float64)
    inlet_value = np.asarray(inlet_face_velocity, dtype=np.float64)
    outlet_gradient = np.asarray(outlet_normal_gradient, dtype=np.float64)
    wall_owner = np.asarray(geometry["wall_owner"], dtype=np.int64)
    inlet_owner = np.asarray(geometry["inlet_owner"], dtype=np.int64)
    outlet_owner = np.asarray(geometry["outlet_owner"], dtype=np.int64)
    expected_shapes = (
        (wall_owner.size, n_components),
        (inlet_owner.size, n_components),
        (outlet_owner.size, n_components),
    )
    if (
        wall_value.shape,
        inlet_value.shape,
        outlet_gradient.shape,
    ) != expected_shapes:
        raise ValueError("viscous LSQ boundary arrays have inconsistent shapes")

    rhs = np.zeros((n_cells, n_components, 3), dtype=np.float64)
    interior_owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
    interior_neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
    if interior_owner.size:
        delta = np.asarray(geometry["interior_delta"], dtype=np.float64)
        weight = np.asarray(geometry["interior_weight"], dtype=np.float64)
        velocity_delta = vel[interior_neighbor] - vel[interior_owner]
        contribution = (
            weight[:, None, None] * velocity_delta[:, :, None] * delta[:, None, :]
        )
        np.add.at(rhs, interior_owner, contribution)
        np.add.at(rhs, interior_neighbor, contribution)
    for boundary_owner, boundary_value, delta_key, weight_key in (
        (wall_owner, wall_value, "wall_delta", "wall_weight"),
        (inlet_owner, inlet_value, "inlet_delta", "inlet_weight"),
    ):
        if boundary_owner.size:
            delta = np.asarray(geometry[delta_key], dtype=np.float64)
            weight = np.asarray(geometry[weight_key], dtype=np.float64)
            value_delta = boundary_value - vel[boundary_owner]
            contribution = (
                weight[:, None, None] * value_delta[:, :, None] * delta[:, None, :]
            )
            np.add.at(rhs, boundary_owner, contribution)
    if outlet_owner.size:
        outlet_normal = np.asarray(geometry["outlet_normal"], dtype=np.float64)
        contribution = outlet_gradient[:, :, None] * outlet_normal[:, None, :]
        np.add.at(rhs, outlet_owner, contribution)
    pseudoinverse = np.asarray(geometry["lsq_pseudoinverse"], dtype=np.float64)
    return np.einsum("nij,ncj->nci", pseudoinverse, rhs, optimize=True)


def _viscous_nonorthogonal_face_flux_numpy(
    mesh: ImportedTetraMesh,
    velocity: np.ndarray,
    *,
    wall_face_velocity: np.ndarray,
    inlet_face_velocity: np.ndarray,
    outlet_normal_gradient: np.ndarray,
    geometry: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative vector K*grad fluxes and the reconstructed gradient."""

    gradient = _viscous_lsq_cell_gradient_numpy(
        velocity,
        wall_face_velocity=wall_face_velocity,
        inlet_face_velocity=inlet_face_velocity,
        outlet_normal_gradient=outlet_normal_gradient,
        geometry=geometry,
    )
    n_faces = int(np.asarray(geometry["n_faces"]).item())
    n_components = int(np.asarray(velocity).shape[1])
    flux = np.zeros((n_faces, n_components), dtype=np.float64)
    interior_faces = np.asarray(geometry["interior_faces"], dtype=np.int64)
    if interior_faces.size:
        owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
        neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
        interpolation = np.asarray(geometry["interior_lambda"], dtype=np.float64)
        face_gradient = (1.0 - interpolation)[:, None, None] * gradient[
            owner
        ] + interpolation[:, None, None] * gradient[neighbor]
        k_vector = np.asarray(geometry["interior_k"], dtype=np.float64)
        flux[interior_faces] = np.einsum(
            "fj,fcj->fc", k_vector, face_gradient, optimize=True
        )
    for ids_key, owner_key, k_key in (
        ("wall_faces", "wall_owner", "wall_k"),
        ("inlet_faces", "inlet_owner", "inlet_k"),
    ):
        ids = np.asarray(geometry[ids_key], dtype=np.int64)
        if ids.size:
            owner = np.asarray(geometry[owner_key], dtype=np.int64)
            k_vector = np.asarray(geometry[k_key], dtype=np.float64)
            flux[ids] = np.einsum(
                "fj,fcj->fc", k_vector, gradient[owner], optimize=True
            )
    outlet_faces = np.asarray(geometry["outlet_faces"], dtype=np.int64)
    if outlet_faces.size:
        outlet_area = np.asarray(geometry["outlet_area"], dtype=np.float64)
        flux[outlet_faces] = outlet_area[:, None] * np.asarray(
            outlet_normal_gradient, dtype=np.float64
        )
    return flux, gradient


def _vector_face_flux_laplacian_numpy(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
) -> np.ndarray:
    """Accumulate an oriented vector face flux and divide by cell volume."""

    flux = np.asarray(face_flux, dtype=np.float64)
    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    owner = face_to_cells[:, 0]
    neighbor = face_to_cells[:, 1]
    accumulated = np.zeros((mesh.tetrahedra.shape[0], flux.shape[1]), dtype=np.float64)
    np.add.at(accumulated, owner, flux)
    interior = neighbor >= 0
    if np.any(interior):
        np.add.at(accumulated, neighbor[interior], -flux[interior])
    volumes = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    return accumulated / volumes[:, None]


def _pressure_lsq_cell_gradient_numpy(
    pressure: np.ndarray,
    *,
    pressure_outlet_value: float,
    geometry: dict[str, np.ndarray | float],
) -> np.ndarray:
    """Reconstruct cell pressure gradients without per-cell Python loops."""

    p = np.asarray(pressure, dtype=np.float64)
    pseudoinverse = np.asarray(geometry["lsq_pseudoinverse"], dtype=np.float64)
    rhs = np.zeros((p.shape[0], 3), dtype=np.float64)

    interior_owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
    interior_neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
    if interior_owner.size:
        interior_delta = np.asarray(geometry["interior_delta"], dtype=np.float64)
        interior_weight = np.asarray(geometry["interior_weight"], dtype=np.float64)
        pressure_delta = p[interior_neighbor] - p[interior_owner]
        contribution = (
            interior_weight[:, None] * interior_delta * pressure_delta[:, None]
        )
        np.add.at(rhs, interior_owner, contribution)
        np.add.at(rhs, interior_neighbor, contribution)

    outlet_owner = np.asarray(geometry["outlet_owner"], dtype=np.int64)
    if outlet_owner.size:
        outlet_delta = np.asarray(geometry["outlet_delta"], dtype=np.float64)
        outlet_weight = np.asarray(geometry["outlet_weight"], dtype=np.float64)
        outlet_pressure_delta = float(pressure_outlet_value) - p[outlet_owner]
        outlet_contribution = (
            outlet_weight[:, None] * outlet_delta * outlet_pressure_delta[:, None]
        )
        np.add.at(rhs, outlet_owner, outlet_contribution)

    return np.einsum("nij,nj->ni", pseudoinverse, rhs, optimize=True)


def _pressure_nonorthogonal_gradient_flux_numpy(
    mesh: ImportedTetraMesh,
    pressure: np.ndarray,
    *,
    dt: float,
    density: float,
    pressure_outlet_value: float,
    geometry: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return deferred pressure-gradient flux and the LSQ cell gradient."""

    gradient = _pressure_lsq_cell_gradient_numpy(
        pressure,
        pressure_outlet_value=float(pressure_outlet_value),
        geometry=geometry,
    )
    flux = np.zeros((mesh.face_vertices.shape[0],), dtype=np.float64)
    scale = float(dt / density)

    interior_faces = np.asarray(geometry["interior_faces"], dtype=np.int64)
    if interior_faces.size:
        interior_owner = np.asarray(geometry["interior_owner"], dtype=np.int64)
        interior_neighbor = np.asarray(geometry["interior_neighbor"], dtype=np.int64)
        interpolation = np.asarray(geometry["interior_lambda"], dtype=np.float64)
        gradient_face = (1.0 - interpolation)[:, None] * gradient[
            interior_owner
        ] + interpolation[:, None] * gradient[interior_neighbor]
        nonorthogonal_vector = np.asarray(
            geometry["interior_nonorthogonal_vector"], dtype=np.float64
        )
        flux[interior_faces] = scale * np.einsum(
            "ij,ij->i",
            nonorthogonal_vector,
            gradient_face,
            optimize=True,
        )

    outlet_faces = np.asarray(geometry["outlet_faces"], dtype=np.int64)
    if outlet_faces.size:
        outlet_owner = np.asarray(geometry["outlet_owner"], dtype=np.int64)
        outlet_nonorthogonal_vector = np.asarray(
            geometry["outlet_nonorthogonal_vector"], dtype=np.float64
        )
        flux[outlet_faces] = scale * np.einsum(
            "ij,ij->i",
            outlet_nonorthogonal_vector,
            gradient[outlet_owner],
            optimize=True,
        )
    return flux, gradient


def _pressure_nonorthogonal_rhs_term(
    mesh: ImportedTetraMesh,
    nonorthogonal_gradient_flux: np.ndarray,
    *,
    rhs_mode: ProjectionRhsMode,
) -> np.ndarray:
    """Map deferred face flux to the pressure RHS in matching source units."""

    term = _compute_cell_flux_sum(mesh, nonorthogonal_gradient_flux)
    if rhs_mode == "divergence_per_volume":
        volume = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
        term = term / volume
    return term


def _cached_pressure_nonorthogonal_geometry_torch(
    mesh: ImportedTetraMesh,
    geometry: dict[str, np.ndarray | float],
    *,
    device: str,
) -> dict[str, Any]:
    """Cache the immutable deferred-LSQ geometry on a Torch device."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable.") from exc

    dev = torch.device(device)
    dtype = torch.float64
    key = (
        "pressure_nonorthogonal_lsq_geometry",
        id(geometry["lsq_pseudoinverse"]),
        id(mesh.face_to_cells),
        id(mesh.cell_volumes),
        str(dev),
        str(dtype),
    )
    cached = _TORCH_PRESSURE_CACHE.get(key)
    if cached is not None:
        return cached

    face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
    neighbor = face_to_cells[:, 1]
    interior_faces_all = np.flatnonzero(neighbor >= 0).astype(np.int64)
    float_keys = (
        "interior_delta",
        "interior_weight",
        "interior_lambda",
        "interior_nonorthogonal_vector",
        "outlet_delta",
        "outlet_weight",
        "outlet_nonorthogonal_vector",
        "lsq_pseudoinverse",
    )
    index_keys = (
        "interior_faces",
        "interior_owner",
        "interior_neighbor",
        "outlet_faces",
        "outlet_owner",
    )
    cached = {
        name: torch.as_tensor(geometry[name], dtype=dtype, device=dev)
        for name in float_keys
    }
    cached.update(
        {
            name: torch.as_tensor(geometry[name], dtype=torch.long, device=dev)
            for name in index_keys
        }
    )
    cached.update(
        {
            "owner_all": torch.as_tensor(
                face_to_cells[:, 0], dtype=torch.long, device=dev
            ),
            "interior_faces_all": torch.as_tensor(
                interior_faces_all, dtype=torch.long, device=dev
            ),
            "interior_neighbor_all": torch.as_tensor(
                neighbor[interior_faces_all], dtype=torch.long, device=dev
            ),
            "cell_volumes": torch.as_tensor(
                np.asarray(mesh.cell_volumes, dtype=np.float64),
                dtype=dtype,
                device=dev,
            ),
        }
    )
    _TORCH_PRESSURE_CACHE[key] = cached
    return cached


def _pressure_nonorthogonal_gradient_flux_torch(
    pressure: Any,
    *,
    n_faces: int,
    dt: float,
    density: float,
    pressure_outlet_value: float,
    geometry: dict[str, Any],
) -> tuple[Any, Any]:
    """Torch equivalent of the vectorized deferred-LSQ pressure correction."""

    import torch  # type: ignore

    p = pressure
    rhs = torch.zeros((p.numel(), 3), dtype=p.dtype, device=p.device)
    interior_owner = geometry["interior_owner"]
    if interior_owner.numel():
        interior_neighbor = geometry["interior_neighbor"]
        pressure_delta = p[interior_neighbor] - p[interior_owner]
        contribution = (
            geometry["interior_weight"][:, None]
            * geometry["interior_delta"]
            * pressure_delta[:, None]
        )
        rhs.index_add_(0, interior_owner, contribution)
        rhs.index_add_(0, interior_neighbor, contribution)

    outlet_owner = geometry["outlet_owner"]
    if outlet_owner.numel():
        outlet_pressure_delta = float(pressure_outlet_value) - p[outlet_owner]
        outlet_contribution = (
            geometry["outlet_weight"][:, None]
            * geometry["outlet_delta"]
            * outlet_pressure_delta[:, None]
        )
        rhs.index_add_(0, outlet_owner, outlet_contribution)

    gradient = torch.bmm(geometry["lsq_pseudoinverse"], rhs.unsqueeze(-1)).squeeze(-1)
    flux = torch.zeros((int(n_faces),), dtype=p.dtype, device=p.device)
    interior_faces = geometry["interior_faces"]
    if interior_faces.numel():
        interpolation = geometry["interior_lambda"]
        gradient_face = (1.0 - interpolation)[:, None] * gradient[
            interior_owner
        ] + interpolation[:, None] * gradient[geometry["interior_neighbor"]]
        flux[interior_faces] = float(dt / density) * torch.sum(
            geometry["interior_nonorthogonal_vector"] * gradient_face,
            dim=1,
        )

    outlet_faces = geometry["outlet_faces"]
    if outlet_faces.numel():
        flux[outlet_faces] = float(dt / density) * torch.sum(
            geometry["outlet_nonorthogonal_vector"] * gradient[outlet_owner],
            dim=1,
        )
    return flux, gradient


def _pressure_nonorthogonal_rhs_term_torch(
    nonorthogonal_gradient_flux: Any,
    *,
    rhs_mode: ProjectionRhsMode,
    geometry: dict[str, Any],
) -> Any:
    """Accumulate deferred face flux without returning through host memory."""

    import torch  # type: ignore

    flux = nonorthogonal_gradient_flux
    term = torch.zeros(
        (geometry["cell_volumes"].numel(),), dtype=flux.dtype, device=flux.device
    )
    term.index_add_(0, geometry["owner_all"], flux)
    interior_faces = geometry["interior_faces_all"]
    if interior_faces.numel():
        term.index_add_(
            0,
            geometry["interior_neighbor_all"],
            -flux[interior_faces],
        )
    if rhs_mode == "divergence_per_volume":
        term = term / torch.clamp(geometry["cell_volumes"], min=1e-30)
    return term


def _vector_stats_torch(values: Any) -> dict[str, float]:
    """Compute the existing scalar diagnostics without copying full arrays."""

    import torch  # type: ignore

    arr = values.reshape(-1)
    if arr.numel() == 0:
        return _vector_stats(np.zeros((0,), dtype=np.float64))
    absolute = torch.abs(arr)
    scalars = (
        torch.stack(
            (
                torch.min(arr),
                torch.max(arr),
                torch.mean(arr),
                torch.sqrt(torch.mean(arr * arr)),
                torch.max(absolute),
                torch.mean(absolute),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    return dict(zip(("min", "max", "mean", "l2", "max_abs", "mean_abs"), scalars))


def _assemble_poisson_rhs(
    cell_flux_sum: np.ndarray,
    coeff: dict[str, np.ndarray],
    *,
    pressure_outlet_value: float,
    projection_sign: ProjectionSign,
    cell_volumes: np.ndarray | None = None,
    rhs_mode: ProjectionRhsMode = "volume_integrated_flux",
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(cell_flux_sum, dtype=np.float64).copy()
    if rhs_mode == "divergence_per_volume":
        if cell_volumes is None:
            raise ValueError(
                "cell_volumes is required for rhs_mode='divergence_per_volume'."
            )
        vol = np.maximum(np.asarray(cell_volumes, dtype=np.float64), 1e-30)
        source = source / vol
    if projection_sign == "minus":
        rhs = -source
    else:
        rhs = source
    outlet_rhs = np.zeros_like(rhs)
    if coeff["out_owner"].size:
        out_term = np.asarray(coeff["out_k"], dtype=np.float64)
        if rhs_mode == "divergence_per_volume":
            if cell_volumes is None:
                raise ValueError(
                    "cell_volumes is required for rhs_mode='divergence_per_volume'."
                )
            owner_vol = np.maximum(
                np.asarray(cell_volumes, dtype=np.float64)[coeff["out_owner"]], 1e-30
            )
            out_term = out_term / owner_vol
        np.add.at(
            outlet_rhs, coeff["out_owner"], out_term * float(pressure_outlet_value)
        )
        rhs += outlet_rhs
    return rhs, outlet_rhs


def _matvec_pressure_numpy(coeff: dict[str, np.ndarray], p: np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    out = np.asarray(coeff["diag"], dtype=np.float64) * p_arr
    if coeff["int_owner"].size:
        np.add.at(out, coeff["int_owner"], -coeff["int_k"] * p_arr[coeff["int_neigh"]])
        np.add.at(out, coeff["int_neigh"], -coeff["int_k"] * p_arr[coeff["int_owner"]])
    return out


def _pressure_matrix_explicit_entries(
    coeff: dict[str, np.ndarray],
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diag = np.asarray(coeff["diag"], dtype=np.float64)
    rows = np.arange(int(n_cells), dtype=np.int64)
    cols = np.arange(int(n_cells), dtype=np.int64)
    vals = diag.copy()
    owner = np.asarray(coeff["int_owner"], dtype=np.int64)
    neigh = np.asarray(coeff["int_neigh"], dtype=np.int64)
    k = np.asarray(coeff["int_k"], dtype=np.float64)
    if owner.size:
        rows = np.concatenate((rows, owner, neigh))
        cols = np.concatenate((cols, neigh, owner))
        vals = np.concatenate((vals, -k, -k))
    return rows, cols, vals


def _compress_sparse_entries(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(rows, dtype=np.int64)
    c = np.asarray(cols, dtype=np.int64)
    v = np.asarray(vals, dtype=np.float64)
    if r.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    keys = r * int(n_cells) + c
    order = np.argsort(keys)
    keys_sorted = keys[order]
    vals_sorted = v[order]
    unique_keys, first = np.unique(keys_sorted, return_index=True)
    reduced_vals = np.add.reduceat(vals_sorted, first)
    rows_u = unique_keys // int(n_cells)
    cols_u = unique_keys % int(n_cells)
    return (
        rows_u.astype(np.int64),
        cols_u.astype(np.int64),
        reduced_vals.astype(np.float64),
    )


def _explicit_pressure_matvec(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    p: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    out = np.zeros((int(n_cells),), dtype=np.float64)
    np.add.at(
        out,
        np.asarray(rows, dtype=np.int64),
        np.asarray(vals, dtype=np.float64)
        * np.asarray(p, dtype=np.float64)[np.asarray(cols, dtype=np.int64)],
    )
    return out


def _pressure_matrix_explicit_audit(
    coeff: dict[str, np.ndarray],
    n_cells: int,
) -> dict[str, Any]:
    rows_raw, cols_raw, vals_raw = _pressure_matrix_explicit_entries(coeff, n_cells)
    rows, cols, vals = _compress_sparse_entries(rows_raw, cols_raw, vals_raw, n_cells)
    nnz = int(vals.size)
    is_diag = rows == cols
    diag_vals = vals[is_diag]
    offdiag_vals = vals[~is_diag]
    row_sum = np.bincount(rows, weights=vals, minlength=int(n_cells))
    row_abs_sum = np.bincount(rows, weights=np.abs(vals), minlength=int(n_cells))
    row_nnz = np.bincount(rows, minlength=int(n_cells))
    number_of_isolated_rows = int(np.count_nonzero(row_nnz == 0))
    number_of_negative_diag = int(np.count_nonzero(diag_vals < 0.0))
    number_of_positive_offdiag = int(np.count_nonzero(offdiag_vals > 0.0))
    has_dirichlet_anchor = bool(np.asarray(coeff["out_owner"], dtype=np.int64).size > 0)

    # Symmetry audit on compressed entries: compare A_ij and A_ji.
    key = rows * int(n_cells) + cols
    val_map: dict[int, float] = {
        int(k): float(v) for k, v in zip(key.tolist(), vals.tolist())
    }
    max_abs = 0.0
    sq = 0.0
    count = 0
    for k_int, v in val_map.items():
        i = int(k_int // int(n_cells))
        j = int(k_int % int(n_cells))
        k_t = int(j * int(n_cells) + i)
        vt = float(val_map.get(k_t, 0.0))
        d = float(v - vt)
        ad = abs(d)
        if ad > max_abs:
            max_abs = ad
        sq += d * d
        count += 1
    symmetry_l2 = float(np.sqrt(sq / max(count, 1)))

    diag_abs = np.abs(diag_vals)
    offdiag_abs_sum = np.zeros((int(n_cells),), dtype=np.float64)
    if np.any(~is_diag):
        np.add.at(offdiag_abs_sum, rows[~is_diag], np.abs(offdiag_vals))
    dom_ratio = (
        diag_abs / np.maximum(offdiag_abs_sum, 1e-30)
        if diag_abs.size
        else np.zeros((0,), dtype=np.float64)
    )
    estimated_condition_proxy = (
        float(np.max(diag_vals) / max(np.min(diag_vals), 1e-30))
        if diag_vals.size
        else 0.0
    )

    return {
        "n_cells": int(n_cells),
        "nnz": int(nnz),
        "diag_stats": _vector_stats(diag_vals),
        "offdiag_stats": _vector_stats(offdiag_vals),
        "row_sum_stats": _vector_stats(row_sum),
        "row_abs_sum_stats": _vector_stats(row_abs_sum),
        "diagonal_dominance_proxy_stats": _vector_stats(dom_ratio),
        "symmetry_max_abs": float(max_abs),
        "symmetry_l2": float(symmetry_l2),
        "number_of_isolated_rows": int(number_of_isolated_rows),
        "number_of_negative_diag": int(number_of_negative_diag),
        "number_of_positive_offdiag": int(number_of_positive_offdiag),
        "has_dirichlet_anchor": bool(has_dirichlet_anchor),
        "estimated_condition_proxy": float(estimated_condition_proxy),
        "notes": {
            "interior_faces": "Aii+=k, Ajj+=k, Aij-=k, Aji-=k",
            "outlet_dirichlet_faces": "included via diagonal coefficient in coeff['diag'] and rhs outlet term",
            "inlet_wall_neumann": "no direct Dirichlet matrix term",
        },
        "matrix_entries": {
            "rows": rows,
            "cols": cols,
            "vals": vals,
        },
    }


def _pressure_matrixfree_vs_explicit_audit(
    coeff: dict[str, np.ndarray],
    *,
    n_cells: int,
    pressure_solution: np.ndarray,
    random_count: int = 8,
    seed: int = 20260504,
) -> dict[str, Any]:
    audit = _pressure_matrix_explicit_audit(coeff, n_cells)
    rows = np.asarray(audit["matrix_entries"]["rows"], dtype=np.int64)
    cols = np.asarray(audit["matrix_entries"]["cols"], dtype=np.int64)
    vals = np.asarray(audit["matrix_entries"]["vals"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    vectors: list[tuple[str, np.ndarray]] = [
        ("pressure_solution", np.asarray(pressure_solution, dtype=np.float64))
    ]
    for i in range(int(max(random_count, 1))):
        vectors.append(
            (f"random_{i}", rng.standard_normal(int(n_cells)).astype(np.float64))
        )
    worst_abs = 0.0
    worst_l2 = 0.0
    worst_rel = 0.0
    worst_name = ""
    worst_diff_vec = np.zeros((int(n_cells),), dtype=np.float64)
    rows_out: list[dict[str, Any]] = []
    for name, p in vectors:
        mf = _matvec_pressure_numpy(coeff, p)
        ex = _explicit_pressure_matvec(rows, cols, vals, p, int(n_cells))
        diff = mf - ex
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        l2 = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
        rel = _safe_ratio(l2, float(np.sqrt(np.mean(ex * ex))) if ex.size else 0.0)
        rows_out.append(
            {
                "vector_name": name,
                "max_abs_diff": float(max_abs),
                "l2_diff": float(l2),
                "relative_l2_diff": float(rel),
            }
        )
        if max_abs > worst_abs:
            worst_abs = max_abs
            worst_l2 = l2
            worst_rel = rel
            worst_name = str(name)
            worst_diff_vec = diff.copy()
    top = np.argsort(-np.abs(worst_diff_vec))[:30]
    return {
        "vectors": rows_out,
        "max_abs_diff": float(worst_abs),
        "l2_diff": float(worst_l2),
        "relative_l2_diff": float(worst_rel),
        "worst_vector_name": str(worst_name),
        "pass_tolerance": bool(worst_abs <= 1e-12 and worst_rel <= 1e-10),
        "worst_cells": [
            {
                "cell_index": int(cid),
                "abs_diff": float(abs(worst_diff_vec[cid])),
                "diff": float(worst_diff_vec[cid]),
            }
            for cid in top.tolist()
        ],
        "worst_diff_vector": worst_diff_vec,
    }


def _pressure_operator_spd_audit(
    coeff: dict[str, np.ndarray],
    *,
    n_cells: int,
    random_count: int = 40,
    seed: int = 20260504,
) -> dict[str, Any]:
    mat = _pressure_matrix_explicit_audit(coeff, n_cells)
    rows = np.asarray(mat["matrix_entries"]["rows"], dtype=np.int64)
    cols = np.asarray(mat["matrix_entries"]["cols"], dtype=np.int64)
    vals = np.asarray(mat["matrix_entries"]["vals"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    energies: list[float] = []
    rayleigh: list[float] = []
    negative = 0
    near_zero = 0
    for _ in range(int(max(random_count, 1))):
        x = rng.standard_normal(int(n_cells)).astype(np.float64)
        ax = _explicit_pressure_matvec(rows, cols, vals, x, int(n_cells))
        e = float(np.dot(x, ax))
        energies.append(e)
        xn = float(np.dot(x, x))
        rayleigh.append(e / max(xn, 1e-30))
        if e < -1e-12:
            negative += 1
        if abs(e) <= 1e-12:
            near_zero += 1

    eig = {}
    spd_likely = bool(
        negative == 0 and float(mat.get("symmetry_max_abs", 0.0)) <= 1e-12
    )
    reason = "random_energy_nonnegative_and_symmetric"
    try:
        import scipy.sparse as sp  # type: ignore
        import scipy.sparse.linalg as spla  # type: ignore

        A = sp.coo_matrix(
            (vals, (rows, cols)), shape=(int(n_cells), int(n_cells))
        ).tocsr()
        k_small = min(3, max(int(n_cells) - 2, 1))
        small = spla.eigsh(A, k=k_small, which="SA", return_eigenvectors=False)
        large = spla.eigsh(A, k=1, which="LA", return_eigenvectors=False)
        eig = {
            "smallest_eigenvalues": [
                float(v) for v in np.asarray(small, dtype=np.float64).tolist()
            ],
            "largest_eigenvalue": float(
                np.asarray(large, dtype=np.float64).reshape(-1)[0]
            ),
        }
        if np.min(np.asarray(small, dtype=np.float64)) < -1e-10:
            spd_likely = False
            reason = "negative_smallest_eigenvalue"
    except Exception:
        eig = {
            "smallest_eigenvalues": None,
            "largest_eigenvalue": None,
            "note": "scipy eigsh unavailable or failed; used random-energy audit only",
        }
    if float(mat.get("symmetry_max_abs", 0.0)) > 1e-10:
        spd_likely = False
        reason = "matrix_not_symmetric"
    if negative > 0:
        spd_likely = False
        reason = "negative_random_energy_detected"
    return {
        "random_energy_min": float(min(energies) if energies else 0.0),
        "random_energy_max": float(max(energies) if energies else 0.0),
        "negative_energy_count": int(negative),
        "near_zero_energy_count": int(near_zero),
        "rayleigh_stats": _vector_stats(np.asarray(rayleigh, dtype=np.float64)),
        "SPD_likely": bool(spd_likely),
        "reason": str(reason),
        **eig,
    }


def _solve_pressure_reference_explicit(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    rtol: float = 1e-6,
    maxiter: int = 5000,
) -> dict[str, Any]:
    n_cells = int(np.asarray(rhs, dtype=np.float64).size)
    mat = _pressure_matrix_explicit_audit(coeff, n_cells)
    rows = np.asarray(mat["matrix_entries"]["rows"], dtype=np.int64)
    cols = np.asarray(mat["matrix_entries"]["cols"], dtype=np.int64)
    vals = np.asarray(mat["matrix_entries"]["vals"], dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    x0_arr = (
        np.asarray(x0, dtype=np.float64) if x0 is not None else np.zeros_like(rhs_arr)
    )
    try:
        import scipy.sparse as sp  # type: ignore
        import scipy.sparse.linalg as spla  # type: ignore
    except Exception:
        return {
            "scipy_available": False,
            "methods": [],
            "note": "scipy is unavailable",
        }
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    rhs_l2 = float(np.sqrt(np.mean(rhs_arr * rhs_arr))) if rhs_arr.size else 0.0
    rhs_max = float(np.max(np.abs(rhs_arr))) if rhs_arr.size else 0.0

    methods: list[dict[str, Any]] = []

    def _record(name: str, x: np.ndarray, info: int) -> None:
        r = A @ x - rhs_arr
        r_max = float(np.max(np.abs(r))) if r.size else 0.0
        r_l2 = float(np.sqrt(np.mean(r * r))) if r.size else 0.0
        methods.append(
            {
                "solver": str(name),
                "converged": bool(info == 0),
                "info": int(info),
                "residual_ratio_to_rhs_max": _safe_ratio(r_max, rhs_max),
                "residual_ratio_to_rhs_l2": _safe_ratio(r_l2, rhs_l2),
                "residual_max_abs": float(r_max),
                "residual_l2": float(r_l2),
                "pressure_min": float(np.min(x)) if x.size else 0.0,
                "pressure_max": float(np.max(x)) if x.size else 0.0,
                "pressure_mean": float(np.mean(x)) if x.size else 0.0,
                "pressure_solution": x,
            }
        )

    try:
        x_cg, info_cg = spla.cg(
            A, rhs_arr, x0=x0_arr, rtol=float(rtol), atol=0.0, maxiter=int(maxiter)
        )
        _record("scipy_cg", np.asarray(x_cg, dtype=np.float64), int(info_cg))
    except Exception as exc:
        methods.append(
            {"solver": "scipy_cg", "converged": False, "info": -999, "error": str(exc)}
        )
    try:
        x_minres, info_minres = spla.minres(
            A, rhs_arr, x0=x0_arr, rtol=float(rtol), maxiter=int(maxiter)
        )
        _record(
            "scipy_minres", np.asarray(x_minres, dtype=np.float64), int(info_minres)
        )
    except Exception as exc:
        methods.append(
            {
                "solver": "scipy_minres",
                "converged": False,
                "info": -999,
                "error": str(exc),
            }
        )
    try:
        x_gmres, info_gmres = spla.gmres(
            A,
            rhs_arr,
            x0=x0_arr,
            rtol=float(rtol),
            atol=0.0,
            restart=200,
            maxiter=int(maxiter),
        )
        _record("scipy_gmres", np.asarray(x_gmres, dtype=np.float64), int(info_gmres))
    except Exception as exc:
        methods.append(
            {
                "solver": "scipy_gmres",
                "converged": False,
                "info": -999,
                "error": str(exc),
            }
        )

    return {
        "scipy_available": True,
        "methods": methods,
        "matrix_nnz": int(A.nnz),
    }


def _solve_pressure_jacobi_numpy(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    diag = np.maximum(np.asarray(coeff["diag"], dtype=np.float64), 1e-20)
    p = np.asarray(p0, dtype=np.float64).copy()

    res0_vec = _matvec_pressure_numpy(coeff, p) - rhs_arr
    res0_max = float(np.max(np.abs(res0_vec)))
    res0_l2 = float(np.sqrt(np.mean(res0_vec**2)))
    rhs_max = float(np.max(np.abs(rhs_arr)))
    rhs_l2 = float(np.sqrt(np.mean(rhs_arr**2)))
    rhs_mean = float(np.mean(rhs_arr))

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    actual_iterations = 0
    update_max_last = 0.0
    res_max_last = res0_max
    res_l2_last = res0_l2

    if rhs_max <= 1e-30 and res0_max <= config.pressure_tolerance:
        stopping_reason = "rhs_already_zero"
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            off = np.zeros_like(rhs_arr)
            if coeff["int_owner"].size:
                np.add.at(
                    off, coeff["int_owner"], -coeff["int_k"] * p[coeff["int_neigh"]]
                )
                np.add.at(
                    off, coeff["int_neigh"], -coeff["int_k"] * p[coeff["int_owner"]]
                )
            p_new = (rhs_arr - off) / diag
            p_next = (
                1.0 - config.relaxation_omega
            ) * p + config.relaxation_omega * p_new
            update = p_next - p
            update_max = float(np.max(np.abs(update)))
            p = p_next

            res_vec = _matvec_pressure_numpy(coeff, p) - rhs_arr
            res_max = float(np.max(np.abs(res_vec)))
            res_l2 = float(np.sqrt(np.mean(res_vec**2)))
            res_ratio = _safe_ratio(res_max, rhs_max)
            update_max_last = update_max
            res_max_last = res_max
            res_l2_last = res_l2
            actual_iterations = it

            if it <= 10:
                history.append(
                    {
                        "iteration": float(it),
                        "residual_max_abs": res_max,
                        "residual_l2": res_l2,
                        "residual_ratio_to_rhs_max": res_ratio,
                        "pressure_update_max_abs": update_max,
                    }
                )
            if not np.isfinite(res_max) or not np.isfinite(update_max):
                stopping_reason = "nan_detected"
                break
            if rhs_max > 1e-30:
                if res_ratio <= config.pressure_relative_tolerance:
                    stopping_reason = "relative_residual_below_tolerance"
                    break
            elif res_max <= config.pressure_tolerance:
                stopping_reason = "residual_below_tolerance"
                break

    if actual_iterations > 10:
        history.append(
            {
                "iteration": float(actual_iterations),
                "residual_max_abs": res_max_last,
                "residual_l2": res_l2_last,
                "residual_ratio_to_rhs_max": _safe_ratio(res_max_last, rhs_max),
                "pressure_update_max_abs": update_max_last,
            }
        )

    return p, {
        "poisson_iterations": int(actual_iterations),
        "actual_iterations": int(actual_iterations),
        "stopping_reason": stopping_reason,
        "rhs_norm_l2": rhs_l2,
        "rhs_max_abs": rhs_max,
        "rhs_mean": rhs_mean,
        "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
        "initial_residual_max_abs": res0_max,
        "initial_residual_l2": res0_l2,
        "poisson_residual_initial": res0_max,
        "poisson_residual_final": res_max_last,
        "final_residual_l2": res_l2_last,
        "residual_ratio_to_rhs_max": _safe_ratio(res_max_last, rhs_max),
        "residual_ratio_to_rhs_l2": _safe_ratio(res_l2_last, rhs_l2),
        "pressure_history": history,
        "pressure_update_max_abs_last": update_max_last,
        "detected_nan_or_inf": bool(stopping_reason == "nan_detected"),
    }


def _solve_pressure_jacobi_torch(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable.") from exc

    dev = torch.device(device)
    dtype = torch.float64
    rhs_t = torch.as_tensor(rhs, dtype=dtype, device=dev)
    diag = torch.as_tensor(np.maximum(coeff["diag"], 1e-20), dtype=dtype, device=dev)
    p = torch.as_tensor(p0, dtype=dtype, device=dev).clone()
    int_o = torch.as_tensor(coeff["int_owner"], dtype=torch.long, device=dev)
    int_n = torch.as_tensor(coeff["int_neigh"], dtype=torch.long, device=dev)
    int_k = torch.as_tensor(coeff["int_k"], dtype=dtype, device=dev)

    def matvec(pt: Any) -> Any:
        out = diag * pt
        if int_o.numel() > 0:
            out.index_add_(0, int_o, -int_k * pt[int_n])
            out.index_add_(0, int_n, -int_k * pt[int_o])
        return out

    res0_vec = matvec(p) - rhs_t
    res0_max = float(torch.max(torch.abs(res0_vec)).item())
    res0_l2 = float(torch.sqrt(torch.mean(res0_vec * res0_vec)).item())
    rhs_max = float(torch.max(torch.abs(rhs_t)).item())
    rhs_l2 = float(torch.sqrt(torch.mean(rhs_t * rhs_t)).item())
    rhs_mean = float(torch.mean(rhs_t).item())

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    actual_iterations = 0
    update_max_last = 0.0
    res_max_last = res0_max
    res_l2_last = res0_l2

    if rhs_max <= 1e-30 and res0_max <= config.pressure_tolerance:
        stopping_reason = "rhs_already_zero"
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            off = torch.zeros_like(rhs_t)
            if int_o.numel() > 0:
                off.index_add_(0, int_o, -int_k * p[int_n])
                off.index_add_(0, int_n, -int_k * p[int_o])
            p_new = (rhs_t - off) / diag
            p_next = (
                1.0 - config.relaxation_omega
            ) * p + config.relaxation_omega * p_new
            update = p_next - p
            update_max = float(torch.max(torch.abs(update)).item())
            p = p_next

            res_vec = matvec(p) - rhs_t
            res_max = float(torch.max(torch.abs(res_vec)).item())
            res_l2 = float(torch.sqrt(torch.mean(res_vec * res_vec)).item())
            res_ratio = _safe_ratio(res_max, rhs_max)
            update_max_last = update_max
            res_max_last = res_max
            res_l2_last = res_l2
            actual_iterations = it

            if it <= 10:
                history.append(
                    {
                        "iteration": float(it),
                        "residual_max_abs": res_max,
                        "residual_l2": res_l2,
                        "residual_ratio_to_rhs_max": res_ratio,
                        "pressure_update_max_abs": update_max,
                    }
                )
            if not np.isfinite(res_max) or not np.isfinite(update_max):
                stopping_reason = "nan_detected"
                break
            if rhs_max > 1e-30:
                if res_ratio <= config.pressure_relative_tolerance:
                    stopping_reason = "relative_residual_below_tolerance"
                    break
            elif res_max <= config.pressure_tolerance:
                stopping_reason = "residual_below_tolerance"
                break

    if actual_iterations > 10:
        history.append(
            {
                "iteration": float(actual_iterations),
                "residual_max_abs": res_max_last,
                "residual_l2": res_l2_last,
                "residual_ratio_to_rhs_max": _safe_ratio(res_max_last, rhs_max),
                "pressure_update_max_abs": update_max_last,
            }
        )

    return (
        p.detach().cpu().numpy(),
        {
            "poisson_iterations": int(actual_iterations),
            "actual_iterations": int(actual_iterations),
            "stopping_reason": stopping_reason,
            "rhs_norm_l2": rhs_l2,
            "rhs_max_abs": rhs_max,
            "rhs_mean": rhs_mean,
            "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
            "initial_residual_max_abs": res0_max,
            "initial_residual_l2": res0_l2,
            "poisson_residual_initial": res0_max,
            "poisson_residual_final": res_max_last,
            "final_residual_l2": res_l2_last,
            "residual_ratio_to_rhs_max": _safe_ratio(res_max_last, rhs_max),
            "residual_ratio_to_rhs_l2": _safe_ratio(res_l2_last, rhs_l2),
            "pressure_history": history,
            "pressure_update_max_abs_last": update_max_last,
            "detected_nan_or_inf": bool(stopping_reason == "nan_detected"),
        },
        bool(dev.type == "cuda"),
    )


def _cg_check_convergence(
    *,
    res_l2: float,
    res_max: float,
    rhs_l2: float,
    rhs_max: float,
    config: TetraFlowConfig,
    allow_relative_max: bool = True,
) -> str | None:
    ratio_l2 = _safe_ratio(res_l2, rhs_l2)
    ratio_max = _safe_ratio(res_max, rhs_max)
    if ratio_l2 <= float(config.pressure_relative_tolerance):
        return "converged_relative_l2"
    if allow_relative_max and ratio_max <= float(config.pressure_relative_tolerance):
        return "converged_relative_max"
    # Use absolute convergence only when rhs itself is near machine-zero.
    if (rhs_max <= 1e-30) and (res_max <= float(config.pressure_tolerance)):
        return "converged_absolute"
    return None


def _cg_detect_stagnation(
    *,
    residual_l2_history: list[float],
    rhs_l2: float,
    config: TetraFlowConfig,
) -> bool:
    w = int(config.cg_stagnation_window)
    if len(residual_l2_history) < max(w, 200):
        return False
    old = float(residual_l2_history[-w])
    new = float(residual_l2_history[-1])
    if old <= 1e-30:
        return False
    ratio = new / old
    if ratio < float(config.cg_stagnation_ratio):
        return False
    near_target = _safe_ratio(new, rhs_l2) <= max(
        10.0 * float(config.pressure_relative_tolerance), 1e-3
    )
    long_plateau = len(residual_l2_history) >= max(4 * w, 400)
    return bool(near_target or long_plateau)


def _cg_pressure_solved_from_reason(reason: str) -> bool:
    return reason in {
        "converged_relative_l2",
        "converged_relative_max",
        "converged_absolute",
        "breakdown_near_converged",
    }


def _solve_pressure_cg_numpy(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64).copy()

    res = rhs_arr - _matvec_pressure_numpy(coeff, p)
    rhs_l2 = float(np.sqrt(np.mean(rhs_arr**2)))
    rhs_max = float(np.max(np.abs(rhs_arr)))
    res0_l2 = float(np.sqrt(np.mean(res**2)))
    res0_max = float(np.max(np.abs(res)))
    p_dir = res.copy()
    rr_old = float(np.dot(res, res))

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    breakdown_reason = ""
    actual_iterations = 0
    res_l2 = res0_l2
    res_max = res0_max
    nan_detected = False
    update_max_last = 0.0
    alpha_last = 0.0
    beta_last = 0.0
    p_ap_last = 0.0
    rr_num_last = rr_old
    residual_l2_history: list[float] = [res0_l2]
    residual_max_history: list[float] = [res0_max]

    early_reason = _cg_check_convergence(
        res_l2=res0_l2,
        res_max=res0_max,
        rhs_l2=rhs_l2,
        rhs_max=rhs_max,
        config=config,
    )
    if rhs_l2 <= 1e-30 and res0_l2 <= config.pressure_tolerance:
        stopping_reason = "converged_absolute"
    elif early_reason is not None:
        stopping_reason = early_reason
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            ap = _matvec_pressure_numpy(coeff, p_dir)
            denom = float(np.dot(p_dir, ap))
            p_ap_last = denom
            rr_num_last = rr_old
            if not np.isfinite(denom):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "pAp_not_finite"
                break
            if abs(denom) <= float(config.cg_breakdown_eps):
                converged = _cg_check_convergence(
                    res_l2=res_l2,
                    res_max=res_max,
                    rhs_l2=rhs_l2,
                    rhs_max=rhs_max,
                    config=config,
                )
                stopping_reason = (
                    "breakdown_near_converged"
                    if converged is not None
                    else "breakdown_not_converged"
                )
                breakdown_reason = (
                    f"|pAp|<=eps ({abs(denom):.3e} <= {config.cg_breakdown_eps:.3e})"
                )
                break
            alpha = rr_old / denom
            alpha_last = float(alpha)
            if not np.isfinite(alpha):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "alpha_not_finite"
                break
            p_next = p + alpha * p_dir
            update = p_next - p
            update_max = float(np.max(np.abs(update)))
            p = p_next
            res = res - alpha * ap

            rr_new = float(np.dot(res, res))
            if rr_new < 0.0 or not np.isfinite(rr_new):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "rr_new_invalid"
                break
            res_l2 = float(np.sqrt(np.mean(res**2)))
            res_max = float(np.max(np.abs(res)))
            ratio_l2 = _safe_ratio(res_l2, rhs_l2)
            ratio_max = _safe_ratio(res_max, rhs_max)
            residual_l2_history.append(res_l2)
            residual_max_history.append(res_max)
            actual_iterations = it
            update_max_last = update_max

            beta = 0.0
            if rr_old > 1e-30:
                beta = rr_new / rr_old
            beta_last = float(beta)
            history.append(
                {
                    "iteration": float(it),
                    "rr_numerator": float(rr_old),
                    "pAp_denominator": float(denom),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "residual_l2": res_l2,
                    "residual_max_abs": res_max,
                    "residual_ratio_to_rhs_l2": ratio_l2,
                    "residual_ratio_to_rhs_max": ratio_max,
                    "pressure_update_max_abs": update_max,
                }
            )
            if not np.isfinite(res_l2) or not np.isfinite(res_max):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "residual_not_finite"
                break
            converged = _cg_check_convergence(
                res_l2=res_l2,
                res_max=res_max,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
            )
            if converged is not None:
                stopping_reason = converged
                break
            if _cg_detect_stagnation(
                residual_l2_history=residual_l2_history,
                rhs_l2=rhs_l2,
                config=config,
            ):
                stopping_reason = "stagnated"
                break
            p_dir = res + beta * p_dir
            rr_old = rr_new

    return p, {
        "poisson_iterations": int(actual_iterations),
        "actual_iterations": int(actual_iterations),
        "stopping_reason": stopping_reason,
        "breakdown_reason": breakdown_reason,
        "rhs_norm_l2": rhs_l2,
        "rhs_max_abs": rhs_max,
        "rhs_mean": float(np.mean(rhs_arr)),
        "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
        "pressure_tolerance": float(config.pressure_tolerance),
        "cg_breakdown_eps": float(config.cg_breakdown_eps),
        "cg_stagnation_window": int(config.cg_stagnation_window),
        "cg_stagnation_ratio": float(config.cg_stagnation_ratio),
        "initial_residual_max_abs": res0_max,
        "initial_residual_l2": res0_l2,
        "poisson_residual_initial": res0_max,
        "poisson_residual_final": res_max,
        "final_residual_l2": res_l2,
        "residual_ratio_to_rhs_max": _safe_ratio(res_max, rhs_max),
        "residual_ratio_to_rhs_l2": _safe_ratio(res_l2, rhs_l2),
        "pressure_history": history,
        "pressure_update_max_abs_last": update_max_last,
        "last_alpha": float(alpha_last),
        "last_beta": float(beta_last),
        "last_pAp_denominator": float(p_ap_last),
        "last_rr_numerator": float(rr_num_last),
        "detected_nan_or_inf": bool(nan_detected),
        "pressure_solved": bool(_cg_pressure_solved_from_reason(stopping_reason)),
    }


def _solve_pressure_pcg_diag_numpy(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64).copy()
    diag = np.maximum(np.asarray(coeff["diag"], dtype=np.float64), 1e-20)
    inv_diag = 1.0 / diag

    res = rhs_arr - _matvec_pressure_numpy(coeff, p)
    z = inv_diag * res
    rhs_l2 = float(np.sqrt(np.mean(rhs_arr**2)))
    rhs_max = float(np.max(np.abs(rhs_arr)))
    res0_l2 = float(np.sqrt(np.mean(res**2)))
    res0_max = float(np.max(np.abs(res)))
    p_dir = z.copy()
    rz_old = float(np.dot(res, z))

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    breakdown_reason = ""
    actual_iterations = 0
    res_l2 = res0_l2
    res_max = res0_max
    nan_detected = False
    update_max_last = 0.0
    alpha_last = 0.0
    beta_last = 0.0
    p_ap_last = 0.0
    rz_num_last = rz_old
    residual_l2_history: list[float] = [res0_l2]
    allow_relative_max = not bool(config.pcg_require_relative_l2_convergence)
    verify_true_residual = bool(config.pcg_require_relative_l2_convergence)
    true_residual_recompute_count = 0
    true_residual_restart_count = 0
    recursive_true_mismatch_l2_max = 0.0
    recursive_true_mismatch_max_abs_max = 0.0

    early_reason = _cg_check_convergence(
        res_l2=res0_l2,
        res_max=res0_max,
        rhs_l2=rhs_l2,
        rhs_max=rhs_max,
        config=config,
        allow_relative_max=allow_relative_max,
    )
    if rhs_l2 <= 1e-30 and res0_l2 <= config.pressure_tolerance:
        stopping_reason = "converged_absolute"
    elif early_reason is not None:
        stopping_reason = early_reason
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            ap = _matvec_pressure_numpy(coeff, p_dir)
            denom = float(np.dot(p_dir, ap))
            p_ap_last = denom
            rz_num_last = rz_old
            if not np.isfinite(denom):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "pAp_not_finite"
                break
            if abs(denom) <= float(config.cg_breakdown_eps):
                converged = _cg_check_convergence(
                    res_l2=res_l2,
                    res_max=res_max,
                    rhs_l2=rhs_l2,
                    rhs_max=rhs_max,
                    config=config,
                    allow_relative_max=allow_relative_max,
                )
                stopping_reason = (
                    "breakdown_near_converged"
                    if converged is not None
                    else "breakdown_not_converged"
                )
                breakdown_reason = (
                    f"|pAp|<=eps ({abs(denom):.3e} <= {config.cg_breakdown_eps:.3e})"
                )
                break
            alpha = rz_old / denom
            alpha_last = float(alpha)
            if not np.isfinite(alpha):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "alpha_not_finite"
                break
            p_next = p + alpha * p_dir
            update = p_next - p
            update_max = float(np.max(np.abs(update)))
            p = p_next
            res = res - alpha * ap

            if not np.all(np.isfinite(res)):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "residual_not_finite"
                break
            res_l2 = float(np.sqrt(np.mean(res**2)))
            res_max = float(np.max(np.abs(res)))
            residual_l2_history.append(res_l2)
            stagnation_detected = _cg_detect_stagnation(
                residual_l2_history=residual_l2_history,
                rhs_l2=rhs_l2,
                config=config,
            )
            candidate_reason = _cg_check_convergence(
                res_l2=res_l2,
                res_max=res_max,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
                allow_relative_max=allow_relative_max,
            )
            residual_recomputed = bool(
                verify_true_residual
                and (candidate_reason is not None or stagnation_detected)
            )
            if residual_recomputed:
                recursive_res = res
                res = rhs_arr - _matvec_pressure_numpy(coeff, p)
                mismatch = res - recursive_res
                recursive_true_mismatch_l2_max = max(
                    recursive_true_mismatch_l2_max,
                    float(np.sqrt(np.mean(mismatch**2))),
                )
                recursive_true_mismatch_max_abs_max = max(
                    recursive_true_mismatch_max_abs_max,
                    float(np.max(np.abs(mismatch))),
                )
                true_residual_recompute_count += 1
                res_l2 = float(np.sqrt(np.mean(res**2)))
                res_max = float(np.max(np.abs(res)))
                residual_l2_history[-1] = res_l2
            z = inv_diag * res
            rz_new = float(np.dot(res, z))
            if rz_new < 0.0 or not np.isfinite(rz_new):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "rz_new_invalid"
                break

            ratio_l2 = _safe_ratio(res_l2, rhs_l2)
            ratio_max = _safe_ratio(res_max, rhs_max)
            actual_iterations = it
            update_max_last = update_max

            beta = 0.0
            if not residual_recomputed and abs(rz_old) > float(config.cg_breakdown_eps):
                beta = rz_new / rz_old
            beta_last = float(beta)
            history.append(
                {
                    "iteration": float(it),
                    "rr_numerator": float(rz_old),
                    "pAp_denominator": float(denom),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "residual_l2": res_l2,
                    "residual_max_abs": res_max,
                    "residual_ratio_to_rhs_l2": ratio_l2,
                    "residual_ratio_to_rhs_max": ratio_max,
                    "pressure_update_max_abs": update_max,
                    "true_residual_recomputed": float(residual_recomputed),
                    "stagnation_detected": float(stagnation_detected),
                }
            )
            converged = _cg_check_convergence(
                res_l2=res_l2,
                res_max=res_max,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
                allow_relative_max=allow_relative_max,
            )
            if converged is not None:
                stopping_reason = converged
                break
            if residual_recomputed:
                p_dir = z.copy()
                rz_old = rz_new
                true_residual_restart_count += 1
                residual_l2_history = [res_l2]
                continue
            if stagnation_detected:
                stopping_reason = "stagnated"
                break
            p_dir = z + beta * p_dir
            rz_old = rz_new

    final_true_res_l2: float | None = None
    final_true_res_max: float | None = None
    if verify_true_residual:
        final_true_res = rhs_arr - _matvec_pressure_numpy(coeff, p)
        final_true_res_l2 = float(np.sqrt(np.mean(final_true_res**2)))
        final_true_res_max = float(np.max(np.abs(final_true_res)))

    return p, {
        "poisson_iterations": int(actual_iterations),
        "actual_iterations": int(actual_iterations),
        "stopping_reason": stopping_reason,
        "breakdown_reason": breakdown_reason,
        "rhs_norm_l2": rhs_l2,
        "rhs_max_abs": rhs_max,
        "rhs_mean": float(np.mean(rhs_arr)),
        "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
        "pressure_tolerance": float(config.pressure_tolerance),
        "cg_breakdown_eps": float(config.cg_breakdown_eps),
        "cg_stagnation_window": int(config.cg_stagnation_window),
        "cg_stagnation_ratio": float(config.cg_stagnation_ratio),
        "pcg_require_relative_l2_convergence": bool(
            config.pcg_require_relative_l2_convergence
        ),
        "pcg_true_residual_verification_enabled": verify_true_residual,
        "true_residual_recompute_count": int(true_residual_recompute_count),
        "true_residual_restart_count": int(true_residual_restart_count),
        "recursive_true_residual_mismatch_l2_max": float(
            recursive_true_mismatch_l2_max
        ),
        "recursive_true_residual_mismatch_max_abs_max": float(
            recursive_true_mismatch_max_abs_max
        ),
        "initial_residual_max_abs": res0_max,
        "initial_residual_l2": res0_l2,
        "poisson_residual_initial": res0_max,
        "poisson_residual_final": res_max,
        "final_residual_l2": res_l2,
        "final_true_residual_l2": final_true_res_l2,
        "final_true_residual_max_abs": final_true_res_max,
        "residual_ratio_to_rhs_max": _safe_ratio(res_max, rhs_max),
        "residual_ratio_to_rhs_l2": _safe_ratio(res_l2, rhs_l2),
        "pressure_history": history,
        "pressure_update_max_abs_last": update_max_last,
        "last_alpha": float(alpha_last),
        "last_beta": float(beta_last),
        "last_pAp_denominator": float(p_ap_last),
        "last_rr_numerator": float(rz_num_last),
        "detected_nan_or_inf": bool(nan_detected),
        "preconditioner": "diag",
        "pressure_solved": bool(_cg_pressure_solved_from_reason(stopping_reason)),
    }


def _solve_pressure_cg_torch(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable.") from exc

    dev = torch.device(device)
    dtype = torch.float64
    rhs_t = torch.as_tensor(rhs, dtype=dtype, device=dev)
    p = torch.as_tensor(p0, dtype=dtype, device=dev).clone()
    diag = torch.as_tensor(
        np.asarray(coeff["diag"], dtype=np.float64), dtype=dtype, device=dev
    )
    int_o = torch.as_tensor(coeff["int_owner"], dtype=torch.long, device=dev)
    int_n = torch.as_tensor(coeff["int_neigh"], dtype=torch.long, device=dev)
    int_k = torch.as_tensor(coeff["int_k"], dtype=dtype, device=dev)

    def matvec(pt: Any) -> Any:
        out = diag * pt
        if int_o.numel() > 0:
            out.index_add_(0, int_o, -int_k * pt[int_n])
            out.index_add_(0, int_n, -int_k * pt[int_o])
        return out

    res = rhs_t - matvec(p)
    rhs_l2 = float(torch.sqrt(torch.mean(rhs_t * rhs_t)).item())
    rhs_max = float(torch.max(torch.abs(rhs_t)).item())
    res0_l2 = float(torch.sqrt(torch.mean(res * res)).item())
    res0_max = float(torch.max(torch.abs(res)).item())
    p_dir = res.clone()
    rr_old = float(torch.dot(res, res).item())

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    breakdown_reason = ""
    actual_iterations = 0
    res_l2 = res0_l2
    res_max = res0_max
    nan_detected = False
    update_max_last = 0.0
    alpha_last = 0.0
    beta_last = 0.0
    p_ap_last = 0.0
    rr_num_last = rr_old
    residual_l2_history: list[float] = [res0_l2]

    early_reason = _cg_check_convergence(
        res_l2=res0_l2,
        res_max=res0_max,
        rhs_l2=rhs_l2,
        rhs_max=rhs_max,
        config=config,
    )
    if rhs_l2 <= 1e-30 and res0_l2 <= config.pressure_tolerance:
        stopping_reason = "converged_absolute"
    elif early_reason is not None:
        stopping_reason = early_reason
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            ap = matvec(p_dir)
            denom = float(torch.dot(p_dir, ap).item())
            p_ap_last = denom
            rr_num_last = rr_old
            if not np.isfinite(denom):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "pAp_not_finite"
                break
            if abs(denom) <= float(config.cg_breakdown_eps):
                converged = _cg_check_convergence(
                    res_l2=res_l2,
                    res_max=res_max,
                    rhs_l2=rhs_l2,
                    rhs_max=rhs_max,
                    config=config,
                )
                stopping_reason = (
                    "breakdown_near_converged"
                    if converged is not None
                    else "breakdown_not_converged"
                )
                breakdown_reason = (
                    f"|pAp|<=eps ({abs(denom):.3e} <= {config.cg_breakdown_eps:.3e})"
                )
                break
            alpha = rr_old / denom
            alpha_last = float(alpha)
            if not np.isfinite(alpha):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "alpha_not_finite"
                break
            p_next = p + alpha * p_dir
            update = p_next - p
            update_max = float(torch.max(torch.abs(update)).item())
            p = p_next
            res = res - alpha * ap
            rr_new = float(torch.dot(res, res).item())
            if rr_new < 0.0 or not np.isfinite(rr_new):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "rr_new_invalid"
                break
            res_l2 = float(torch.sqrt(torch.mean(res * res)).item())
            res_max = float(torch.max(torch.abs(res)).item())
            ratio_l2 = _safe_ratio(res_l2, rhs_l2)
            ratio_max = _safe_ratio(res_max, rhs_max)
            residual_l2_history.append(res_l2)
            actual_iterations = it
            update_max_last = update_max

            beta = 0.0
            if rr_old > float(config.cg_breakdown_eps):
                beta = rr_new / rr_old
            beta_last = float(beta)
            history.append(
                {
                    "iteration": float(it),
                    "rr_numerator": float(rr_old),
                    "pAp_denominator": float(denom),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "residual_l2": res_l2,
                    "residual_max_abs": res_max,
                    "residual_ratio_to_rhs_l2": ratio_l2,
                    "residual_ratio_to_rhs_max": ratio_max,
                    "pressure_update_max_abs": update_max,
                }
            )
            if not np.isfinite(res_l2) or not np.isfinite(res_max):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "residual_not_finite"
                break
            converged = _cg_check_convergence(
                res_l2=res_l2,
                res_max=res_max,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
            )
            if converged is not None:
                stopping_reason = converged
                break
            if _cg_detect_stagnation(
                residual_l2_history=residual_l2_history,
                rhs_l2=rhs_l2,
                config=config,
            ):
                stopping_reason = "stagnated"
                break
            p_dir = res + beta * p_dir
            rr_old = rr_new

    return (
        p.detach().cpu().numpy(),
        {
            "poisson_iterations": int(actual_iterations),
            "actual_iterations": int(actual_iterations),
            "stopping_reason": stopping_reason,
            "breakdown_reason": breakdown_reason,
            "rhs_norm_l2": rhs_l2,
            "rhs_max_abs": rhs_max,
            "rhs_mean": float(torch.mean(rhs_t).item()),
            "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
            "pressure_tolerance": float(config.pressure_tolerance),
            "cg_breakdown_eps": float(config.cg_breakdown_eps),
            "cg_stagnation_window": int(config.cg_stagnation_window),
            "cg_stagnation_ratio": float(config.cg_stagnation_ratio),
            "initial_residual_max_abs": res0_max,
            "initial_residual_l2": res0_l2,
            "poisson_residual_initial": res0_max,
            "poisson_residual_final": res_max,
            "final_residual_l2": res_l2,
            "residual_ratio_to_rhs_max": _safe_ratio(res_max, rhs_max),
            "residual_ratio_to_rhs_l2": _safe_ratio(res_l2, rhs_l2),
            "pressure_history": history,
            "pressure_update_max_abs_last": update_max_last,
            "last_alpha": float(alpha_last),
            "last_beta": float(beta_last),
            "last_pAp_denominator": float(p_ap_last),
            "last_rr_numerator": float(rr_num_last),
            "detected_nan_or_inf": bool(nan_detected),
            "pressure_solved": bool(_cg_pressure_solved_from_reason(stopping_reason)),
        },
        bool(dev.type == "cuda"),
    )


def _solve_pressure_pcg_diag_torch(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
    device: str,
    _return_tensor: bool = False,
) -> tuple[Any, dict[str, Any], bool]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable.") from exc

    dev = torch.device(device)
    dtype = torch.float64
    rhs_t = torch.as_tensor(rhs, dtype=dtype, device=dev)
    p = torch.as_tensor(p0, dtype=dtype, device=dev).clone()
    base_diag = coeff.get("base_diag")
    base_int_k = coeff.get("base_int_k")
    scale = 1.0
    pressure_matvec_backend = "torch_index_add"
    pressure_matvec_fallback_reason = ""
    pressure_matvec_matrix_cached = False
    sparse_csr_matrix = None
    if base_diag is not None and base_int_k is not None:
        tensor_cache_key = (
            "pcg_diag",
            id(base_diag),
            id(base_int_k),
            id(coeff["int_owner"]),
            id(coeff["int_neigh"]),
            str(dev),
            str(dtype),
        )
        tensor_geometry = _TORCH_PRESSURE_CACHE.get(tensor_cache_key)
        if tensor_geometry is None:
            tensor_geometry = {
                "base_diag": torch.as_tensor(base_diag, dtype=dtype, device=dev),
                "base_int_k": torch.as_tensor(base_int_k, dtype=dtype, device=dev),
                "int_owner": torch.as_tensor(
                    coeff["int_owner"], dtype=torch.long, device=dev
                ),
                "int_neigh": torch.as_tensor(
                    coeff["int_neigh"], dtype=torch.long, device=dev
                ),
            }
            _TORCH_PRESSURE_CACHE[tensor_cache_key] = tensor_geometry
        if dev.type == "cuda":
            sparse_csr_matrix = tensor_geometry.get("base_sparse_csr")
            pressure_matvec_matrix_cached = sparse_csr_matrix is not None
            if sparse_csr_matrix is None:
                try:
                    base_diag_t = tensor_geometry["base_diag"]
                    base_int_k_t = tensor_geometry["base_int_k"]
                    base_int_o_t = tensor_geometry["int_owner"]
                    base_int_n_t = tensor_geometry["int_neigh"]
                    cell_ids = torch.arange(
                        base_diag_t.numel(), dtype=torch.long, device=dev
                    )
                    row = torch.cat((cell_ids, base_int_o_t, base_int_n_t))
                    col = torch.cat((cell_ids, base_int_n_t, base_int_o_t))
                    values = torch.cat((base_diag_t, -base_int_k_t, -base_int_k_t))
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="Sparse CSR tensor support is in beta state.*",
                            category=UserWarning,
                        )
                        sparse_csr_matrix = (
                            torch.sparse_coo_tensor(
                                torch.stack((row, col)),
                                values,
                                (base_diag_t.numel(), base_diag_t.numel()),
                                dtype=dtype,
                                device=dev,
                                check_invariants=False,
                            )
                            .coalesce()
                            .to_sparse_csr()
                        )
                    tensor_geometry["base_sparse_csr"] = sparse_csr_matrix
                except (RuntimeError, NotImplementedError) as exc:
                    sparse_csr_matrix = None
                    pressure_matvec_fallback_reason = f"{type(exc).__name__}: {exc}"
            if sparse_csr_matrix is not None:
                pressure_matvec_backend = "torch_sparse_csr"
        scale = float(coeff.get("k_scale", 1.0))
        diag = tensor_geometry["base_diag"] * scale
        int_k = tensor_geometry["base_int_k"] * scale
        int_o = tensor_geometry["int_owner"]
        int_n = tensor_geometry["int_neigh"]
    else:
        diag = torch.as_tensor(
            np.asarray(coeff["diag"], dtype=np.float64), dtype=dtype, device=dev
        )
        int_o = torch.as_tensor(coeff["int_owner"], dtype=torch.long, device=dev)
        int_n = torch.as_tensor(coeff["int_neigh"], dtype=torch.long, device=dev)
        int_k = torch.as_tensor(coeff["int_k"], dtype=dtype, device=dev)
    inv_diag = 1.0 / torch.clamp(diag, min=1e-20)

    def matvec(pt: Any) -> Any:
        if sparse_csr_matrix is not None:
            return float(scale) * torch.mv(sparse_csr_matrix, pt)
        out = diag * pt
        if int_o.numel() > 0:
            out.index_add_(0, int_o, -int_k * pt[int_n])
            out.index_add_(0, int_n, -int_k * pt[int_o])
        return out

    res = rhs_t - matvec(p)
    z = inv_diag * res
    rhs_l2_t = torch.sqrt(torch.mean(rhs_t * rhs_t))
    rhs_max_t = torch.max(torch.abs(rhs_t))
    res0_l2_t = torch.sqrt(torch.mean(res * res))
    res0_max_t = torch.max(torch.abs(res))
    p_dir = z.clone()
    rz_old_t = torch.dot(res, z)
    rhs_l2, rhs_max, res0_l2, res0_max, rz_old = (
        float(value)
        for value in torch.stack((rhs_l2_t, rhs_max_t, res0_l2_t, res0_max_t, rz_old_t))
        .detach()
        .cpu()
        .tolist()
    )

    history: list[dict[str, float]] = []
    stopping_reason = "max_iterations"
    breakdown_reason = ""
    actual_iterations = 0
    res_l2 = res0_l2
    res_max = res0_max
    nan_detected = False
    update_max_last = 0.0
    alpha_last = 0.0
    beta_last = 0.0
    p_ap_last = 0.0
    rz_num_last = rz_old
    residual_l2_history: list[float] = [res0_l2]
    allow_relative_max = not bool(config.pcg_require_relative_l2_convergence)
    verify_true_residual = bool(config.pcg_require_relative_l2_convergence)
    true_residual_recompute_count = 0
    true_residual_restart_count = 0
    recursive_true_mismatch_l2_max = 0.0
    recursive_true_mismatch_max_abs_max = 0.0

    early_reason = _cg_check_convergence(
        res_l2=res0_l2,
        res_max=res0_max,
        rhs_l2=rhs_l2,
        rhs_max=rhs_max,
        config=config,
        allow_relative_max=allow_relative_max,
    )
    if rhs_l2 <= 1e-30 and res0_l2 <= config.pressure_tolerance:
        stopping_reason = "converged_absolute"
    elif early_reason is not None:
        stopping_reason = early_reason
    else:
        for it in range(1, int(config.max_pressure_iterations) + 1):
            ap = matvec(p_dir)
            denom_t = torch.dot(p_dir, ap)
            alpha_t = rz_old_t / denom_t
            p_next = p + alpha_t * p_dir
            update_max_t = torch.max(torch.abs(p_next - p))
            res_next = res - alpha_t * ap
            z_next = inv_diag * res_next
            rz_new_t = torch.dot(res_next, z_next)
            res_l2_t = torch.sqrt(torch.mean(res_next * res_next))
            res_max_t = torch.max(torch.abs(res_next))
            if abs(rz_old) > float(config.cg_breakdown_eps):
                beta_t = rz_new_t / rz_old_t
            else:
                beta_t = torch.zeros_like(rz_new_t)
            (
                denom,
                alpha,
                update_max,
                rz_new,
                res_l2_next,
                res_max_next,
                beta,
            ) = (
                float(value)
                for value in torch.stack(
                    (
                        denom_t,
                        alpha_t,
                        update_max_t,
                        rz_new_t,
                        res_l2_t,
                        res_max_t,
                        beta_t,
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            p_ap_last = denom
            rz_num_last = rz_old
            if not np.isfinite(denom):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "pAp_not_finite"
                break
            if abs(denom) <= float(config.cg_breakdown_eps):
                converged = _cg_check_convergence(
                    res_l2=res_l2,
                    res_max=res_max,
                    rhs_l2=rhs_l2,
                    rhs_max=rhs_max,
                    config=config,
                    allow_relative_max=allow_relative_max,
                )
                stopping_reason = (
                    "breakdown_near_converged"
                    if converged is not None
                    else "breakdown_not_converged"
                )
                breakdown_reason = (
                    f"|pAp|<=eps ({abs(denom):.3e} <= {config.cg_breakdown_eps:.3e})"
                )
                break
            alpha_last = float(alpha)
            if not np.isfinite(alpha):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "alpha_not_finite"
                break
            if (
                not np.isfinite(res_l2_next)
                or not np.isfinite(res_max_next)
                or not np.isfinite(rz_new)
            ):
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "residual_not_finite"
                break
            if rz_new < 0.0:
                stopping_reason = "nan_or_inf"
                nan_detected = True
                breakdown_reason = "rz_new_invalid"
                break
            p = p_next
            residual_l2_history.append(res_l2_next)
            stagnation_detected = _cg_detect_stagnation(
                residual_l2_history=residual_l2_history,
                rhs_l2=rhs_l2,
                config=config,
            )
            candidate_reason = _cg_check_convergence(
                res_l2=res_l2_next,
                res_max=res_max_next,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
                allow_relative_max=allow_relative_max,
            )
            residual_recomputed = bool(
                verify_true_residual
                and (candidate_reason is not None or stagnation_detected)
            )
            if residual_recomputed:
                recursive_res = res_next
                res_next = rhs_t - matvec(p)
                mismatch = res_next - recursive_res
                z_next = inv_diag * res_next
                rz_new_t = torch.dot(res_next, z_next)
                res_l2_t = torch.sqrt(torch.mean(res_next * res_next))
                res_max_t = torch.max(torch.abs(res_next))
                mismatch_l2_t = torch.sqrt(torch.mean(mismatch * mismatch))
                mismatch_max_t = torch.max(torch.abs(mismatch))
                (
                    rz_new,
                    res_l2_next,
                    res_max_next,
                    mismatch_l2,
                    mismatch_max,
                ) = (
                    float(value)
                    for value in torch.stack(
                        (
                            rz_new_t,
                            res_l2_t,
                            res_max_t,
                            mismatch_l2_t,
                            mismatch_max_t,
                        )
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                if (
                    not np.isfinite(rz_new)
                    or not np.isfinite(res_l2_next)
                    or not np.isfinite(res_max_next)
                    or rz_new < 0.0
                ):
                    stopping_reason = "nan_or_inf"
                    nan_detected = True
                    breakdown_reason = "true_residual_not_finite"
                    break
                recursive_true_mismatch_l2_max = max(
                    recursive_true_mismatch_l2_max, mismatch_l2
                )
                recursive_true_mismatch_max_abs_max = max(
                    recursive_true_mismatch_max_abs_max, mismatch_max
                )
                true_residual_recompute_count += 1
                beta_t = torch.zeros_like(rz_new_t)
                beta = 0.0
                residual_l2_history[-1] = res_l2_next
            res = res_next
            z = z_next
            res_l2 = res_l2_next
            res_max = res_max_next
            ratio_l2 = _safe_ratio(res_l2, rhs_l2)
            ratio_max = _safe_ratio(res_max, rhs_max)
            actual_iterations = it
            update_max_last = update_max

            beta_last = float(beta)
            if config.debug_store_history:
                history.append(
                    {
                        "iteration": float(it),
                        "rr_numerator": float(rz_old),
                        "pAp_denominator": float(denom),
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "residual_l2": res_l2,
                        "residual_max_abs": res_max,
                        "residual_ratio_to_rhs_l2": ratio_l2,
                        "residual_ratio_to_rhs_max": ratio_max,
                        "pressure_update_max_abs": update_max,
                        "true_residual_recomputed": float(residual_recomputed),
                        "stagnation_detected": float(stagnation_detected),
                    }
                )
            converged = _cg_check_convergence(
                res_l2=res_l2,
                res_max=res_max,
                rhs_l2=rhs_l2,
                rhs_max=rhs_max,
                config=config,
                allow_relative_max=allow_relative_max,
            )
            if converged is not None:
                stopping_reason = converged
                break
            if residual_recomputed:
                p_dir = z.clone()
                rz_old_t = rz_new_t
                rz_old = rz_new
                true_residual_restart_count += 1
                residual_l2_history = [res_l2]
                continue
            if stagnation_detected:
                stopping_reason = "stagnated"
                break
            p_dir = z + beta_t * p_dir
            rz_old_t = rz_new_t
            rz_old = rz_new

    final_true_res_l2: float | None = None
    final_true_res_max: float | None = None
    if verify_true_residual:
        final_true_res = rhs_t - matvec(p)
        final_true_values = (
            torch.stack(
                (
                    torch.sqrt(torch.mean(final_true_res * final_true_res)),
                    torch.max(torch.abs(final_true_res)),
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        final_true_res_l2, final_true_res_max = (
            float(value) for value in final_true_values
        )

    return (
        p if _return_tensor else p.detach().cpu().numpy(),
        {
            "poisson_iterations": int(actual_iterations),
            "actual_iterations": int(actual_iterations),
            "stopping_reason": stopping_reason,
            "breakdown_reason": breakdown_reason,
            "rhs_norm_l2": rhs_l2,
            "rhs_max_abs": rhs_max,
            "rhs_mean": float(torch.mean(rhs_t).item()),
            "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
            "pressure_tolerance": float(config.pressure_tolerance),
            "cg_breakdown_eps": float(config.cg_breakdown_eps),
            "cg_stagnation_window": int(config.cg_stagnation_window),
            "cg_stagnation_ratio": float(config.cg_stagnation_ratio),
            "pcg_require_relative_l2_convergence": bool(
                config.pcg_require_relative_l2_convergence
            ),
            "pcg_true_residual_verification_enabled": verify_true_residual,
            "true_residual_recompute_count": int(true_residual_recompute_count),
            "true_residual_restart_count": int(true_residual_restart_count),
            "recursive_true_residual_mismatch_l2_max": float(
                recursive_true_mismatch_l2_max
            ),
            "recursive_true_residual_mismatch_max_abs_max": float(
                recursive_true_mismatch_max_abs_max
            ),
            "initial_residual_max_abs": res0_max,
            "initial_residual_l2": res0_l2,
            "poisson_residual_initial": res0_max,
            "poisson_residual_final": res_max,
            "final_residual_l2": res_l2,
            "final_true_residual_l2": final_true_res_l2,
            "final_true_residual_max_abs": final_true_res_max,
            "residual_ratio_to_rhs_max": _safe_ratio(res_max, rhs_max),
            "residual_ratio_to_rhs_l2": _safe_ratio(res_l2, rhs_l2),
            "pressure_history": history,
            "pressure_update_max_abs_last": update_max_last,
            "last_alpha": float(alpha_last),
            "last_beta": float(beta_last),
            "last_pAp_denominator": float(p_ap_last),
            "last_rr_numerator": float(rz_num_last),
            "detected_nan_or_inf": bool(nan_detected),
            "preconditioner": "diag",
            "pressure_matvec_backend": pressure_matvec_backend,
            "pressure_matvec_sparse_csr_used": bool(
                pressure_matvec_backend == "torch_sparse_csr"
            ),
            "pressure_matvec_matrix_cached": bool(pressure_matvec_matrix_cached),
            "pressure_matvec_fallback_reason": pressure_matvec_fallback_reason,
            "pressure_solved": bool(_cg_pressure_solved_from_reason(stopping_reason)),
        },
        bool(dev.type == "cuda"),
    )


def _solve_pressure_deferred_lsq_pcg_diag_torch(
    mesh: ImportedTetraMesh,
    coeff: dict[str, np.ndarray],
    *,
    rhs_base: np.ndarray,
    p0: np.ndarray,
    previous_frozen_gradient_flux: np.ndarray,
    geometry: dict[str, np.ndarray | float],
    config: TetraFlowConfig,
    device: str,
) -> dict[str, Any]:
    """Keep deferred-LSQ sweeps and PCG state resident on a CUDA device."""

    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Torch backend requested but torch is unavailable.") from exc

    dev = torch.device(device)
    dtype = torch.float64
    geometry_t = _cached_pressure_nonorthogonal_geometry_torch(
        mesh,
        geometry,
        device=str(dev),
    )
    p = torch.as_tensor(p0, dtype=dtype, device=dev).clone()
    rhs_base_t = torch.as_tensor(rhs_base, dtype=dtype, device=dev)
    relaxed_flux = torch.as_tensor(
        previous_frozen_gradient_flux, dtype=dtype, device=dev
    ).clone()
    relaxation = float(config.pressure_nonorthogonal_correction_relaxation)
    sweep_rows: list[dict[str, Any]] = []
    total_pressure_iterations = 0
    total_true_residual_recomputes = 0
    total_true_residual_restarts = 0
    recursive_true_mismatch_l2_max = 0.0
    recursive_true_mismatch_max_abs_max = 0.0
    total_pressure_solve_wall_seconds = 0.0
    p_diag: dict[str, Any] = {}
    rhs_nonorthogonal_term = torch.zeros_like(rhs_base_t)
    rhs_sweep = rhs_base_t
    all_sweeps_on_cuda = bool(dev.type == "cuda")

    for sweep_idx in range(int(config.pressure_nonorthogonal_correction_sweeps)):
        raw_flux, gradient_before_solve = _pressure_nonorthogonal_gradient_flux_torch(
            p,
            n_faces=int(mesh.face_vertices.shape[0]),
            dt=float(config.projection_dt),
            density=float(config.density),
            pressure_outlet_value=float(config.pressure_outlet_value),
            geometry=geometry_t,
        )
        relaxed_before = relaxed_flux
        relaxed_flux = (1.0 - relaxation) * relaxed_before + relaxation * raw_flux
        rhs_nonorthogonal_term = _pressure_nonorthogonal_rhs_term_torch(
            relaxed_flux,
            rhs_mode=config.projection_rhs_mode,
            geometry=geometry_t,
        )
        rhs_sweep = rhs_base_t + rhs_nonorthogonal_term
        pressure_before = p
        solve_started = perf_counter()
        p, sweep_pressure_diag, sweep_cuda = _solve_pressure_pcg_diag_torch(
            coeff,
            rhs=rhs_sweep,
            p0=pressure_before,
            config=config,
            device=str(dev),
            _return_tensor=True,
        )
        solve_wall_seconds = float(perf_counter() - solve_started)
        sweep_pressure_diag["solve_wall_seconds"] = solve_wall_seconds
        actual_iterations = int(sweep_pressure_diag.get("actual_iterations", 0))
        sweep_pressure_diag["solve_wall_seconds_per_iteration"] = (
            float(solve_wall_seconds / actual_iterations)
            if actual_iterations > 0
            else 0.0
        )
        total_pressure_solve_wall_seconds += solve_wall_seconds
        all_sweeps_on_cuda = bool(all_sweeps_on_cuda and sweep_cuda)
        p_diag = sweep_pressure_diag
        total_pressure_iterations += int(
            sweep_pressure_diag.get("actual_iterations", 0)
        )
        total_true_residual_recomputes += int(
            sweep_pressure_diag.get("true_residual_recompute_count", 0)
        )
        total_true_residual_restarts += int(
            sweep_pressure_diag.get("true_residual_restart_count", 0)
        )
        recursive_true_mismatch_l2_max = max(
            recursive_true_mismatch_l2_max,
            float(
                sweep_pressure_diag.get("recursive_true_residual_mismatch_l2_max", 0.0)
            ),
        )
        recursive_true_mismatch_max_abs_max = max(
            recursive_true_mismatch_max_abs_max,
            float(
                sweep_pressure_diag.get(
                    "recursive_true_residual_mismatch_max_abs_max", 0.0
                )
            ),
        )
        sweep_rows.append(
            {
                "sweep": int(sweep_idx + 1),
                "raw_gradient_flux": _vector_stats_torch(raw_flux),
                "frozen_gradient_flux_before": _vector_stats_torch(relaxed_before),
                "frozen_gradient_flux_used": _vector_stats_torch(relaxed_flux),
                "frozen_flux_change": _vector_stats_torch(
                    relaxed_flux - relaxed_before
                ),
                "lsq_cell_gradient": _vector_stats_torch(gradient_before_solve),
                "rhs_nonorthogonal_term": _vector_stats_torch(rhs_nonorthogonal_term),
                "rhs_full": _vector_stats_torch(rhs_sweep),
                "pressure_change": _vector_stats_torch(p - pressure_before),
                "pressure_solver": {
                    "actual_iterations": int(
                        sweep_pressure_diag.get("actual_iterations", 0)
                    ),
                    "stopping_reason": str(
                        sweep_pressure_diag.get("stopping_reason", "")
                    ),
                    "pressure_solved": bool(
                        sweep_pressure_diag.get("pressure_solved", False)
                    ),
                    "solve_wall_seconds": float(
                        sweep_pressure_diag.get("solve_wall_seconds", 0.0)
                    ),
                    "residual_ratio_to_rhs_l2": float(
                        sweep_pressure_diag.get("residual_ratio_to_rhs_l2", 0.0)
                    ),
                    "residual_ratio_to_rhs_max": float(
                        sweep_pressure_diag.get("residual_ratio_to_rhs_max", 0.0)
                    ),
                    "true_residual_recompute_count": int(
                        sweep_pressure_diag.get("true_residual_recompute_count", 0)
                    ),
                    "true_residual_restart_count": int(
                        sweep_pressure_diag.get("true_residual_restart_count", 0)
                    ),
                    "recursive_true_residual_mismatch_l2_max": float(
                        sweep_pressure_diag.get(
                            "recursive_true_residual_mismatch_l2_max", 0.0
                        )
                    ),
                    "recursive_true_residual_mismatch_max_abs_max": float(
                        sweep_pressure_diag.get(
                            "recursive_true_residual_mismatch_max_abs_max", 0.0
                        )
                    ),
                },
            }
        )

    final_raw_flux, nonorthogonal_gradient = (
        _pressure_nonorthogonal_gradient_flux_torch(
            p,
            n_faces=int(mesh.face_vertices.shape[0]),
            dt=float(config.projection_dt),
            density=float(config.density),
            pressure_outlet_value=float(config.pressure_outlet_value),
            geometry=geometry_t,
        )
    )
    outer_defect = final_raw_flux - relaxed_flux
    host_values = tuple(
        value.detach().cpu().numpy()
        for value in (
            p,
            rhs_sweep,
            rhs_nonorthogonal_term,
            relaxed_flux,
            final_raw_flux,
            nonorthogonal_gradient,
            outer_defect,
        )
    )
    return {
        "pressure": host_values[0],
        "rhs": host_values[1],
        "rhs_nonorthogonal_term": host_values[2],
        "frozen_nonorthogonal_gradient_flux": host_values[3],
        "final_raw_flux": host_values[4],
        "nonorthogonal_gradient": host_values[5],
        "outer_defect": host_values[6],
        "pressure_diagnostics": p_diag,
        "sweeps": sweep_rows,
        "total_pressure_iterations": int(total_pressure_iterations),
        "total_true_residual_recomputes": int(total_true_residual_recomputes),
        "total_true_residual_restarts": int(total_true_residual_restarts),
        "recursive_true_residual_mismatch_l2_max": float(
            recursive_true_mismatch_l2_max
        ),
        "recursive_true_residual_mismatch_max_abs_max": float(
            recursive_true_mismatch_max_abs_max
        ),
        "total_pressure_solve_wall_seconds": float(total_pressure_solve_wall_seconds),
        "all_sweeps_on_cuda": bool(all_sweeps_on_cuda),
        "geometry_cache_device": str(dev),
        "host_to_device_full_array_transfers": 3,
        "device_to_host_full_array_transfers": len(host_values),
        "inter_sweep_full_array_host_transfers": 0,
    }


def _cached_amg_pressure_preconditioner(
    coeff: dict[str, np.ndarray],
) -> tuple[Any, Any, dict[str, Any]]:
    """Build one AMG hierarchy per unscaled pressure-operator geometry."""

    try:
        import pyamg  # type: ignore
        import scipy.sparse as sparse  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging contract
        raise RuntimeError(
            "pressure_solver='amg_pcg' requires scipy and pyamg"
        ) from exc

    diagonal = np.asarray(coeff.get("base_diag", coeff["diag"]), dtype=np.float64)
    transmissibility = np.asarray(
        coeff.get("base_int_k", coeff["int_k"]), dtype=np.float64
    )
    owner = np.asarray(coeff["int_owner"], dtype=np.int64)
    neighbor = np.asarray(coeff["int_neigh"], dtype=np.int64)
    key = (
        id(diagonal),
        id(transmissibility),
        id(owner),
        id(neighbor),
        int(diagonal.size),
        int(transmissibility.size),
    )
    cached = _AMG_PRESSURE_CACHE.get(key)
    if cached is not None:
        return (
            cached["matrix"],
            cached["preconditioner"],
            {
                **cached["diagnostics"],
                "amg_cache_hit": True,
            },
        )

    indices = np.arange(diagonal.size, dtype=np.int64)
    rows = np.concatenate((indices, owner, neighbor))
    columns = np.concatenate((indices, neighbor, owner))
    values = np.concatenate((diagonal, -transmissibility, -transmissibility))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(diagonal.size, diagonal.size)
    ).tocsr()
    setup_started = perf_counter()
    hierarchy = pyamg.smoothed_aggregation_solver(matrix, symmetry="symmetric")
    preconditioner = hierarchy.aspreconditioner(cycle="V")
    diagnostics = {
        "amg_cache_hit": False,
        "amg_setup_wall_seconds": float(perf_counter() - setup_started),
        "amg_levels": int(len(hierarchy.levels)),
        "amg_operator_complexity": float(hierarchy.operator_complexity()),
        "amg_grid_complexity": float(hierarchy.grid_complexity()),
    }
    _AMG_PRESSURE_CACHE[key] = {
        "matrix": matrix,
        "preconditioner": preconditioner,
        "diagnostics": diagnostics,
    }
    return matrix, preconditioner, diagnostics


def _solve_pressure_amg_pcg_numpy(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the SPD pressure system with cached smoothed-aggregation AMG-PCG."""

    try:
        import scipy.sparse.linalg as sparse_linalg  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging contract
        raise RuntimeError(
            "pressure_solver='amg_pcg' requires scipy and pyamg"
        ) from exc

    rhs_arr = np.asarray(rhs, dtype=np.float64)
    initial = np.asarray(p0, dtype=np.float64)
    scale = float(coeff.get("k_scale", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("pressure operator scale must be finite and positive")
    matrix, preconditioner, amg_diagnostics = _cached_amg_pressure_preconditioner(coeff)
    normalized_rhs = rhs_arr / scale
    rhs_l2 = float(np.sqrt(np.mean(rhs_arr**2)))
    rhs_max = float(np.max(np.abs(rhs_arr)))
    initial_residual = rhs_arr - _matvec_pressure_numpy(coeff, initial)
    initial_l2 = float(np.sqrt(np.mean(initial_residual**2)))
    initial_max = float(np.max(np.abs(initial_residual)))
    iteration_count = 0
    history: list[dict[str, float]] = []

    def _record_iteration(solution: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1
        if config.debug_store_history:
            residual = rhs_arr - _matvec_pressure_numpy(coeff, solution)
            residual_l2 = float(np.sqrt(np.mean(residual**2)))
            history.append(
                {
                    "iteration": float(iteration_count),
                    "residual_l2": residual_l2,
                    "residual_max_abs": float(np.max(np.abs(residual))),
                    "residual_ratio_to_rhs_l2": _safe_ratio(residual_l2, rhs_l2),
                }
            )

    pressure, info = sparse_linalg.cg(
        matrix,
        normalized_rhs,
        x0=initial,
        rtol=float(config.pressure_relative_tolerance),
        atol=(
            float(config.pressure_tolerance)
            * np.sqrt(float(max(rhs_arr.size, 1)))
            / scale
        ),
        maxiter=int(config.max_pressure_iterations),
        M=preconditioner,
        callback=_record_iteration,
    )
    pressure_arr = np.asarray(pressure, dtype=np.float64)
    residual = rhs_arr - _matvec_pressure_numpy(coeff, pressure_arr)
    residual_l2 = float(np.sqrt(np.mean(residual**2)))
    residual_max = float(np.max(np.abs(residual)))
    ratio_l2 = _safe_ratio(residual_l2, rhs_l2)
    if residual_l2 <= float(config.pressure_tolerance):
        stopping_reason = "converged_absolute"
    elif ratio_l2 <= float(config.pressure_relative_tolerance):
        stopping_reason = "converged_relative_l2"
    elif int(info) > 0:
        stopping_reason = "max_iterations"
    else:
        stopping_reason = "breakdown_not_converged"
    pressure_solved = bool(
        stopping_reason in {"converged_absolute", "converged_relative_l2"}
    )
    return pressure_arr, {
        "poisson_iterations": int(iteration_count),
        "actual_iterations": int(iteration_count),
        "stopping_reason": stopping_reason,
        "breakdown_reason": "" if int(info) >= 0 else f"scipy_cg_info={int(info)}",
        "rhs_norm_l2": rhs_l2,
        "rhs_max_abs": rhs_max,
        "rhs_mean": float(np.mean(rhs_arr)),
        "pressure_relative_tolerance": float(config.pressure_relative_tolerance),
        "pressure_tolerance": float(config.pressure_tolerance),
        "initial_residual_max_abs": initial_max,
        "initial_residual_l2": initial_l2,
        "poisson_residual_initial": initial_max,
        "poisson_residual_final": residual_max,
        "final_residual_l2": residual_l2,
        "final_true_residual_l2": residual_l2,
        "final_true_residual_max_abs": residual_max,
        "residual_ratio_to_rhs_max": _safe_ratio(residual_max, rhs_max),
        "residual_ratio_to_rhs_l2": ratio_l2,
        "pressure_history": history,
        "detected_nan_or_inf": bool(
            not np.all(np.isfinite(pressure_arr)) or not np.all(np.isfinite(residual))
        ),
        "preconditioner": "smoothed_aggregation_amg",
        "scipy_cg_info": int(info),
        "pressure_solved": pressure_solved,
        **amg_diagnostics,
    }


def _solve_pressure_system(
    coeff: dict[str, np.ndarray],
    *,
    rhs: np.ndarray,
    p0: np.ndarray,
    config: TetraFlowConfig,
    backend: BackendSelection,
) -> tuple[np.ndarray, dict[str, Any], bool]:
    solve_started = perf_counter()
    all_core_arrays_on_cuda = False
    solver = str(config.pressure_solver)
    if solver == "amg_pcg":
        p, diag = _solve_pressure_amg_pcg_numpy(
            coeff,
            rhs=rhs,
            p0=p0,
            config=config,
        )
    elif solver == "cg":
        if backend.selected_backend == "torch":
            p, diag, all_core_arrays_on_cuda = _solve_pressure_cg_torch(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
                device=config.device if config.device else backend.device,
            )
        else:
            p, diag = _solve_pressure_cg_numpy(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
            )
    elif solver == "pcg_diag":
        if backend.selected_backend == "torch":
            p, diag, all_core_arrays_on_cuda = _solve_pressure_pcg_diag_torch(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
                device=config.device if config.device else backend.device,
            )
        else:
            p, diag = _solve_pressure_pcg_diag_numpy(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
            )
    else:
        if backend.selected_backend == "torch":
            p, diag, all_core_arrays_on_cuda = _solve_pressure_jacobi_torch(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
                device=config.device if config.device else backend.device,
            )
        else:
            p, diag = _solve_pressure_jacobi_numpy(
                coeff,
                rhs=rhs,
                p0=p0,
                config=config,
            )
    if "pressure_solved" not in diag:
        reason = str(diag.get("stopping_reason", ""))
        diag["pressure_solved"] = bool(
            reason
            in {
                "relative_residual_below_tolerance",
                "residual_below_tolerance",
                "rhs_already_zero",
                "converged_relative_l2",
                "converged_relative_max",
                "converged_absolute",
                "breakdown_near_converged",
            }
        )
    diag["pressure_solver"] = solver
    solve_wall_seconds = float(perf_counter() - solve_started)
    diag["solve_wall_seconds"] = solve_wall_seconds
    actual_iterations = int(diag.get("actual_iterations", 0))
    diag["solve_wall_seconds_per_iteration"] = (
        float(solve_wall_seconds / actual_iterations) if actual_iterations > 0 else 0.0
    )
    return p, diag, all_core_arrays_on_cuda


def _pressure_face_gradient_flux(
    mesh: ImportedTetraMesh,
    pressure: np.ndarray,
    *,
    dt: float,
    density: float,
    outlet_faces: np.ndarray,
    pressure_outlet_value: float,
    frozen_nonorthogonal_gradient_flux: np.ndarray | None = None,
    coeff: dict[str, Any] | None = None,
) -> np.ndarray:
    grad_flux = np.zeros((mesh.face_vertices.shape[0],), dtype=np.float64)
    pressure_coeff = (
        coeff
        if coeff is not None
        else _build_pressure_system_coefficients(
            mesh,
            dt=float(dt),
            density=float(density),
            outlet_faces=np.asarray(outlet_faces, dtype=np.int64),
        )
    )
    int_face = np.asarray(pressure_coeff.get("int_face", []), dtype=np.int64)
    int_owner = np.asarray(pressure_coeff["int_owner"], dtype=np.int64)
    int_neigh = np.asarray(pressure_coeff["int_neigh"], dtype=np.int64)
    if int_face.size:
        grad_flux[int_face] = np.asarray(pressure_coeff["int_k"], dtype=np.float64) * (
            pressure[int_neigh] - pressure[int_owner]
        )
    out_face = np.asarray(pressure_coeff["out_face"], dtype=np.int64)
    out_owner = np.asarray(pressure_coeff["out_owner"], dtype=np.int64)
    if out_face.size:
        grad_flux[out_face] = np.asarray(pressure_coeff["out_k"], dtype=np.float64) * (
            float(pressure_outlet_value) - pressure[out_owner]
        )
    if frozen_nonorthogonal_gradient_flux is not None:
        nonorthogonal = np.asarray(
            frozen_nonorthogonal_gradient_flux,
            dtype=np.float64,
        )
        if nonorthogonal.shape != grad_flux.shape:
            raise ValueError(
                "frozen_nonorthogonal_gradient_flux must match the face count."
            )
        grad_flux = grad_flux + nonorthogonal
    return grad_flux


def _apply_projection_correction(
    face_flux_star: np.ndarray,
    grad_flux: np.ndarray,
    *,
    projection_sign: ProjectionSign,
    damping: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if projection_sign == "minus":
        correction_flux = -np.asarray(grad_flux, dtype=np.float64)
    else:
        correction_flux = np.asarray(grad_flux, dtype=np.float64)
    correction_flux = float(damping) * correction_flux
    corrected = np.asarray(face_flux_star, dtype=np.float64) + correction_flux
    return corrected, correction_flux


def _pin_projection_correction_faces(
    correction_flux: np.ndarray,
    *,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    pinned = np.asarray(correction_flux, dtype=np.float64).copy()
    pinned_mask = np.zeros_like(pinned, dtype=bool)
    if left_faces.size:
        pinned_mask[np.asarray(left_faces, dtype=np.int64)] = True
    if right_faces.size:
        pinned_mask[np.asarray(right_faces, dtype=np.int64)] = True
    if wall_faces.size:
        pinned_mask[np.asarray(wall_faces, dtype=np.int64)] = True
    max_abs_before = (
        float(np.max(np.abs(pinned[pinned_mask]))) if np.any(pinned_mask) else 0.0
    )
    if np.any(pinned_mask):
        pinned[pinned_mask] = 0.0
    return pinned, {
        "pinned_face_count": int(np.count_nonzero(pinned_mask)),
        "pinned_face_nonzero_before_count": int(
            np.count_nonzero(
                np.abs(np.asarray(correction_flux, dtype=np.float64)[pinned_mask]) > 0.0
            )
        ),
        "pinned_face_max_abs_before_zeroing": float(max_abs_before),
    }


def _face_subset_correction_stats(
    correction_flux: np.ndarray,
    face_ids: np.ndarray,
    *,
    nonzero_tolerance: float = 1e-14,
) -> dict[str, Any]:
    ids = np.asarray(face_ids, dtype=np.int64)
    values = np.asarray(correction_flux, dtype=np.float64)
    if ids.size == 0:
        subset = np.zeros((0,), dtype=np.float64)
    else:
        subset = values[ids]
    abs_subset = np.abs(subset)
    return {
        "face_count": int(ids.size),
        "nonzero_count": int(np.count_nonzero(abs_subset > float(nonzero_tolerance))),
        "max_abs": float(np.max(abs_subset)) if abs_subset.size else 0.0,
        "mean_abs": float(np.mean(abs_subset)) if abs_subset.size else 0.0,
        "l2": float(np.sqrt(np.mean(subset * subset))) if subset.size else 0.0,
        "sum_signed": float(np.sum(subset)) if subset.size else 0.0,
        "sum_abs": float(np.sum(abs_subset)) if abs_subset.size else 0.0,
    }


def _projection_correction_boundary_contract_audit(
    correction_flux: np.ndarray,
    *,
    n_faces: int,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    nonzero_tolerance: float = 1e-14,
) -> dict[str, Any]:
    left_ids = np.asarray(left_faces, dtype=np.int64)
    right_ids = np.asarray(right_faces, dtype=np.int64)
    outlet_ids = np.asarray(outlet_faces, dtype=np.int64)
    wall_ids = np.asarray(wall_faces, dtype=np.int64)
    boundary_mask = np.zeros((int(n_faces),), dtype=bool)
    for ids in (left_ids, right_ids, outlet_ids, wall_ids):
        if ids.size:
            boundary_mask[ids] = True
    interior_ids = np.flatnonzero(~boundary_mask).astype(np.int64)
    return {
        "nonzero_tolerance": float(nonzero_tolerance),
        "left_inlet": _face_subset_correction_stats(
            correction_flux,
            left_ids,
            nonzero_tolerance=nonzero_tolerance,
        ),
        "right_inlet": _face_subset_correction_stats(
            correction_flux,
            right_ids,
            nonzero_tolerance=nonzero_tolerance,
        ),
        "walls": _face_subset_correction_stats(
            correction_flux,
            wall_ids,
            nonzero_tolerance=nonzero_tolerance,
        ),
        "outlet": _face_subset_correction_stats(
            correction_flux,
            outlet_ids,
            nonzero_tolerance=nonzero_tolerance,
        ),
        "interior": _face_subset_correction_stats(
            correction_flux,
            interior_ids,
            nonzero_tolerance=nonzero_tolerance,
        ),
    }


PROJECTION_CORRECTION_STAGE_CODEBOOK = {
    "raw_pre_constraint": "pressure correction before any boundary pinning",
    "constrained_pre_limiter": "boundary-pinned correction before limiter",
    "limiter_output_pre_reconstraint": (
        "direct limiter output before pinned-face re-constraint"
    ),
    "constrained_post_limiter_pre_outlet_policy": (
        "re-constrained correction after limiter and before outlet policy"
    ),
    "effective_post_outlet_policy": "final effective correction after outlet policy",
}


FACE_FLUX_PRIMARY_STAGE_CODEBOOK = {
    "correction_flux_raw_pre_constraint": (
        "pressure correction before any boundary pinning"
    ),
    "correction_flux_constrained_pre_limiter": (
        "boundary-pinned correction before limiter"
    ),
    "correction_flux_limiter_output_pre_reconstraint": (
        "direct limiter output before pinned-face re-constraint"
    ),
    "correction_flux_constrained_post_limiter_pre_outlet_policy": (
        "re-constrained correction after limiter and before outlet policy"
    ),
    "correction_flux_effective_post_outlet_policy": (
        "final effective correction after outlet policy"
    ),
}


def _interior_pairwise_flux_conservation_error(
    face_flux: np.ndarray, face_to_cells: np.ndarray
) -> np.ndarray:
    c1 = np.asarray(face_to_cells[:, 1], dtype=np.int64)
    interior = c1 >= 0
    vals = np.asarray(face_flux, dtype=np.float64)[interior]
    # By representation, interior face flux is a single shared scalar.
    # Pairwise owner/neighbor conservation error is q + (-q) = 0.
    return vals + (-vals)


def _apply_projection_correction_limiter(
    *,
    mesh: ImportedTetraMesh,
    flux_star: np.ndarray,
    correction_flux_raw: np.ndarray,
    div_star: np.ndarray,
    masks: dict[str, np.ndarray],
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    config: TetraFlowConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(config.projection_correction_limit_mode)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    n_cells = int(mesh.tetrahedra.shape[0])
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    interior_core = np.asarray(
        masks.get("interior_core", np.zeros((n_cells,), dtype=bool)), dtype=bool
    )
    boundary_adjacent = np.asarray(
        masks.get("boundary_adjacent", np.ones((n_cells,), dtype=bool)),
        dtype=bool,
    )
    eligible_cells = interior_core & (~boundary_adjacent)
    interior_faces = c1 >= 0
    eligible_faces = interior_faces & eligible_cells[c0] & eligible_cells[c1]

    corr_before = np.asarray(correction_flux_raw, dtype=np.float64).copy()
    corr_after = corr_before.copy()
    limited_faces_mask = np.zeros_like(corr_after, dtype=bool)
    limited_cells_mask = np.zeros((n_cells,), dtype=bool)

    flux_before = np.asarray(flux_star, dtype=np.float64) + corr_before
    div_before = _compute_divergence_numpy(mesh, flux_before)
    median_abs_div_star = float(
        np.median(np.abs(np.asarray(div_star, dtype=np.float64)))
    )
    div_floor = max(float(config.projection_divergence_floor), 1e-30)

    if mode == "face_flux_cap":
        cap = float(config.projection_face_correction_over_volume_cap)
        face_ids = np.where(interior_faces)[0]
        for fid in face_ids.tolist():
            owner = int(c0[fid])
            neigh = int(c1[fid])
            min_vol = min(float(vol[owner]), float(vol[neigh]))
            allowed = cap * max(min_vol, 1e-30)
            val = float(corr_after[fid])
            if abs(val) > allowed:
                corr_after[fid] = np.sign(val) * allowed
                limited_faces_mask[fid] = True
                limited_cells_mask[owner] = True
                limited_cells_mask[neigh] = True
    elif mode == "cell_divergence_cap":
        for _ in range(2):
            flux_now = np.asarray(flux_star, dtype=np.float64) + corr_after
            div_now = _compute_divergence_numpy(mesh, flux_now)
            cell_order = np.argsort(-np.abs(div_now))
            for cid in cell_order.tolist():
                if not bool(eligible_cells[cid]):
                    continue
                ref = max(abs(float(div_star[cid])), median_abs_div_star, div_floor)
                cap_val = float(config.projection_divergence_cap_factor) * ref
                dcur = float(div_now[cid])
                if abs(dcur) <= cap_val:
                    continue
                face_ids = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
                cand = [
                    int(fid)
                    for fid in face_ids.tolist()
                    if bool(eligible_faces[int(fid)])
                ]
                if not cand:
                    continue
                target_div = float(np.sign(dcur) * cap_val)
                delta_flux_sum = (target_div - dcur) * float(vol[cid])
                delta_each = delta_flux_sum / float(len(cand))
                for fid in cand:
                    sign = 1.0 if int(c0[fid]) == cid else -1.0
                    corr_after[fid] += sign * delta_each
                    limited_faces_mask[fid] = True
                limited_cells_mask[cid] = True
    elif mode == "redistribute_local":
        flux_now = np.asarray(flux_star, dtype=np.float64) + corr_after
        div_now = _compute_divergence_numpy(mesh, flux_now)
        cell_order = np.argsort(-np.abs(div_now))
        for cid in cell_order.tolist():
            if not bool(eligible_cells[cid]):
                continue
            ref = max(abs(float(div_star[cid])), median_abs_div_star, div_floor)
            cap_val = float(config.projection_divergence_cap_factor) * ref
            dcur = float(div_now[cid])
            if abs(dcur) <= cap_val:
                continue
            face_ids = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
            cand: list[int] = []
            neigh_abs_div: list[float] = []
            signs: list[float] = []
            for fid in face_ids.tolist():
                if not bool(eligible_faces[int(fid)]):
                    continue
                owner = int(c0[fid])
                neigh = int(c1[fid])
                other = neigh if owner == cid else owner
                cand.append(int(fid))
                signs.append(1.0 if owner == cid else -1.0)
                neigh_abs_div.append(abs(float(div_now[other])) if other >= 0 else 0.0)
            if not cand:
                continue
            target_div = float(np.sign(dcur) * cap_val)
            delta_flux_sum = (target_div - dcur) * float(vol[cid])
            w = 1.0 / np.maximum(np.asarray(neigh_abs_div, dtype=np.float64), div_floor)
            w = w / max(float(np.sum(w)), 1e-30)
            for i, fid in enumerate(cand):
                corr_after[fid] += float(signs[i] * delta_flux_sum * w[i])
                limited_faces_mask[fid] = True
            limited_cells_mask[cid] = True
            flux_now = np.asarray(flux_star, dtype=np.float64) + corr_after
            div_now = _compute_divergence_numpy(mesh, flux_now)

    flux_after = np.asarray(flux_star, dtype=np.float64) + corr_after
    div_after = _compute_divergence_numpy(mesh, flux_after)
    cell_sum_before = _compute_cell_flux_sum(mesh, flux_before)
    cell_sum_after = _compute_cell_flux_sum(mesh, flux_after)
    bnd_before = _boundary_flux_audit(
        mesh,
        flux_before,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    bnd_after = _boundary_flux_audit(
        mesh,
        flux_after,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    pair_err = _interior_pairwise_flux_conservation_error(
        flux_after, np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    limiter_delta = corr_after - corr_before
    limiter_diag = {
        "projection_correction_limit_mode": mode,
        "projection_limit_experimental": bool(mode != "none"),
        "projection_divergence_cap_factor": float(
            config.projection_divergence_cap_factor
        ),
        "projection_divergence_floor": float(config.projection_divergence_floor),
        "projection_face_correction_over_volume_cap": float(
            config.projection_face_correction_over_volume_cap
        ),
        "number_of_limited_cells": int(np.count_nonzero(limited_cells_mask)),
        "number_of_limited_faces": int(np.count_nonzero(limited_faces_mask)),
        "limited_cell_indices": np.where(limited_cells_mask)[0].astype(np.int64),
        "limited_face_indices": np.where(limited_faces_mask)[0].astype(np.int64),
        "correction_flux_before_stats": _vector_stats(corr_before),
        "correction_flux_after_stats": _vector_stats(corr_after),
        "correction_flux_limiter_delta_stats": _vector_stats(limiter_delta),
        "correction_flux_delta_total_signed": float(np.sum(limiter_delta)),
        "correction_flux_delta_total_abs": float(np.sum(np.abs(limiter_delta))),
        "divergence_before_limiter_stats": _vector_stats(div_before),
        "divergence_after_limiter_stats": _vector_stats(div_after),
        "pairwise_interior_conservation_preserved": bool(
            float(np.max(np.abs(pair_err))) <= 1e-30
        ),
        "conservation_audit": {
            "interior_pairwise_flux_conservation_error_max_abs": float(
                np.max(np.abs(pair_err)) if pair_err.size else 0.0
            ),
            "interior_pairwise_flux_conservation_error_l2": float(
                np.sqrt(np.mean(pair_err * pair_err)) if pair_err.size else 0.0
            ),
            "total_boundary_flux_before_limiter": float(
                bnd_before["boundary_total"]["sum_flux"]
            ),
            "total_boundary_flux_after_limiter": float(
                bnd_after["boundary_total"]["sum_flux"]
            ),
            "inlet_flux_before_limiter": float(
                bnd_before["left_inlet"]["inflow_total"]
                + bnd_before["right_inlet"]["inflow_total"]
            ),
            "inlet_flux_after_limiter": float(
                bnd_after["left_inlet"]["inflow_total"]
                + bnd_after["right_inlet"]["inflow_total"]
            ),
            "outlet_flux_before_limiter": float(bnd_before["outlet"]["outflow_total"]),
            "outlet_flux_after_limiter": float(bnd_after["outlet"]["outflow_total"]),
            "wall_flux_before_limiter": float(bnd_before["walls"]["sum_flux"]),
            "wall_flux_after_limiter": float(bnd_after["walls"]["sum_flux"]),
            "total_cell_flux_sum_before_limiter": float(np.sum(cell_sum_before)),
            "total_cell_flux_sum_after_limiter": float(np.sum(cell_sum_after)),
            "limiter_flux_delta_total": float(np.sum(limiter_delta)),
        },
    }
    return corr_after, limiter_diag


def _apply_outlet_projection_mode(
    *,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    outlet_faces: np.ndarray,
    inlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    mode: OutletProjectionMode,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.asarray(flux_corrected, dtype=np.float64).copy()
    out_ids = np.asarray(outlet_faces, dtype=np.int64)
    diag: dict[str, Any] = {
        "outlet_projection_mode": str(mode),
        "outlet_flux_rescale_used": False,
        "outlet_flux_rescale_factor": 1.0,
        "outlet_flux_rescale_reason": "outlet pressure dirichlet baseline",
        "nonphysical_flux_fix_used": bool(
            str(mode) in {"outlet_flux_preserve", "outlet_mass_balance_rescale"}
        ),
    }
    if out_ids.size == 0:
        diag["outlet_flux_rescale_reason"] = "no outlet faces"
        return out, diag
    if mode == "outlet_flux_preserve":
        before = np.asarray(out[out_ids], dtype=np.float64).copy()
        out[out_ids] = flux_star[out_ids]
        changed = (
            bool(np.max(np.abs(out[out_ids] - before)) > 0.0) if out_ids.size else False
        )
        diag["outlet_flux_rescale_used"] = bool(changed)
        diag["outlet_flux_rescale_reason"] = (
            "outlet flux overwritten by star flux (diagnostic-only policy)"
        )
        return out, diag
    if mode == "outlet_mass_balance_rescale":
        target = float(
            np.sum(np.maximum(-flux_star[np.asarray(inlet_faces, dtype=np.int64)], 0.0))
        )
        if wall_faces.size:
            target -= float(
                np.sum(np.maximum(out[np.asarray(wall_faces, dtype=np.int64)], 0.0))
                - np.sum(np.maximum(-out[np.asarray(wall_faces, dtype=np.int64)], 0.0))
            )
        current = float(np.sum(np.maximum(out[out_ids], 0.0)))
        scale_factor = 1.0
        if current > 1e-30:
            scale_factor = float(target / current)
            out[out_ids] *= scale_factor
            diag["outlet_flux_rescale_reason"] = (
                "outlet mass-balance rescale applied to match inlet/wall flux target"
            )
        else:
            uniform = target / float(out_ids.size) if out_ids.size else 0.0
            out[out_ids] = uniform
            diag["outlet_flux_rescale_reason"] = (
                "outlet mass-balance fallback applied from zero current outflow"
            )
        diag["outlet_flux_rescale_factor"] = float(scale_factor)
        diag["outlet_flux_rescale_used"] = bool(abs(scale_factor - 1.0) > 1e-15)
        return out, diag
    return out, diag


def _outlet_projection_audit(
    *,
    mesh: ImportedTetraMesh,
    outlet_faces: np.ndarray,
    flux_star: np.ndarray,
    correction_flux: np.ndarray,
    flux_corrected: np.ndarray,
    pressure: np.ndarray,
    pressure_outlet_value: float,
    expected_outlet_flux: float,
) -> dict[str, Any]:
    out_ids = np.asarray(outlet_faces, dtype=np.int64)
    if out_ids.size == 0:
        return {
            "outlet_face_count": 0,
            "pressure_outlet_value": float(pressure_outlet_value),
            "star_outlet_flux_total": 0.0,
            "corrected_outlet_flux_total": 0.0,
            "expected_outlet_flux_total": float(expected_outlet_flux),
            "outlet_flux_error": -float(expected_outlet_flux),
            "relative_outlet_flux_error": 0.0,
            "top_outlet_faces_by_correction": [],
        }
    owner = np.asarray(mesh.face_to_cells[out_ids, 0], dtype=np.int64)
    p_owner = np.asarray(pressure[owner], dtype=np.float64)
    corr = np.asarray(correction_flux[out_ids], dtype=np.float64)
    star = np.asarray(flux_star[out_ids], dtype=np.float64)
    corr_flux = np.asarray(flux_corrected[out_ids], dtype=np.float64)
    grad = np.asarray(corr, dtype=np.float64)
    star_out = float(np.sum(np.maximum(star, 0.0)))
    corr_out = float(np.sum(np.maximum(corr_flux, 0.0)))
    err = float(corr_out - expected_outlet_flux)
    rel_err = _safe_ratio(err, expected_outlet_flux)

    order = np.argsort(-np.abs(corr))
    top: list[dict[str, Any]] = []
    for ridx in order[:30].tolist():
        fid = int(out_ids[ridx])
        top.append(
            {
                "face_index": fid,
                "area": float(mesh.face_areas[fid]),
                "normal": np.asarray(mesh.face_normals[fid], dtype=np.float64).tolist(),
                "center": np.asarray(mesh.face_centers[fid], dtype=np.float64).tolist(),
                "owner_cell": int(owner[ridx]),
                "owner_pressure": float(p_owner[ridx]),
                "boundary_pressure": float(pressure_outlet_value),
                "star_flux": float(star[ridx]),
                "correction_flux": float(corr[ridx]),
                "corrected_flux": float(corr_flux[ridx]),
            }
        )
    return {
        "outlet_face_count": int(out_ids.size),
        "pressure_outlet_value": float(pressure_outlet_value),
        "owner_pressure_stats_near_outlet": _vector_stats(p_owner),
        "pressure_normal_gradient_flux_stats": _vector_stats(grad),
        "correction_flux_stats_on_outlet": _vector_stats(corr),
        "star_outlet_flux_total": star_out,
        "corrected_outlet_flux_total": corr_out,
        "expected_outlet_flux_total": float(expected_outlet_flux),
        "outlet_flux_error": err,
        "relative_outlet_flux_error": rel_err,
        "top_outlet_faces_by_correction": top,
    }


def _convective_percentiles(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def _convective_threshold_fractions(
    values: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"above_{thr:g}": 0.0 for thr in thresholds}
    return {
        f"above_{thr:g}": float(np.count_nonzero(arr > float(thr)) / arr.size)
        for thr in thresholds
    }


def _estimate_cell_regions(
    mesh: ImportedTetraMesh,
    *,
    left_inlet_faces: np.ndarray,
    right_inlet_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
) -> np.ndarray:
    n_cells = int(mesh.tetrahedra.shape[0])
    labels = np.full((n_cells,), "interior", dtype=object)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)

    def _cells_adjacent_to_faces(face_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(face_ids, dtype=np.int64)
        mask = np.zeros((n_cells,), dtype=bool)
        if ids.size == 0:
            return mask
        own = c0[ids]
        nei = c1[ids]
        mask[own] = True
        interior = nei >= 0
        if np.any(interior):
            mask[nei[interior]] = True
        return mask

    left_adj = _cells_adjacent_to_faces(left_inlet_faces)
    right_adj = _cells_adjacent_to_faces(right_inlet_faces)
    outlet_adj = _cells_adjacent_to_faces(outlet_faces)
    wall_adj = _cells_adjacent_to_faces(wall_faces)

    labels[wall_adj] = "wall-adjacent"
    labels[outlet_adj] = "outlet"
    labels[right_adj] = "right inlet"
    labels[left_adj] = "left inlet"

    unresolved = labels == "interior"
    if np.any(unresolved):
        face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
        jc_parts: list[np.ndarray] = []
        if left_inlet_faces.size:
            jc_parts.append(np.mean(face_centers[left_inlet_faces], axis=0))
        if right_inlet_faces.size:
            jc_parts.append(np.mean(face_centers[right_inlet_faces], axis=0))
        if outlet_faces.size:
            jc_parts.append(np.mean(face_centers[outlet_faces], axis=0))
        if jc_parts:
            junction_center = np.mean(np.asarray(jc_parts, dtype=np.float64), axis=0)
            cell_centers = np.asarray(mesh.cell_centers, dtype=np.float64)
            box_diag = float(
                np.linalg.norm(
                    np.max(np.asarray(mesh.points, dtype=np.float64), axis=0)
                    - np.min(np.asarray(mesh.points, dtype=np.float64), axis=0)
                )
            )
            radius = max(1e-20, 0.18 * box_diag)
            dist = np.linalg.norm(cell_centers - junction_center[None, :], axis=1)
            jmask = unresolved & (dist <= radius)
            labels[jmask] = "junction"
    return np.asarray(labels, dtype=object)


def _build_top_convective_cell_audit(
    mesh: ImportedTetraMesh,
    *,
    vel: np.ndarray,
    out_rate: np.ndarray,
    length_scale: np.ndarray,
    cfl_raw: np.ndarray,
    region_labels: np.ndarray,
    top_k: int = 200,
) -> dict[str, Any]:
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    vol = np.asarray(mesh.cell_volumes, dtype=np.float64)
    speed = np.linalg.norm(np.asarray(vel, dtype=np.float64), axis=1)
    cfl = np.asarray(cfl_raw, dtype=np.float64)
    ls = np.asarray(length_scale, dtype=np.float64)
    reg = np.asarray(region_labels, dtype=object)
    order = np.argsort(-cfl)
    rows: list[dict[str, Any]] = []
    for idx in order[: int(max(top_k, 0))].tolist():
        rows.append(
            {
                "cell_index": int(idx),
                "center": centers[idx].tolist(),
                "cell_volume": float(vol[idx]),
                "local_speed": float(speed[idx]),
                "local_length_scale": float(ls[idx])
                if np.isfinite(ls[idx])
                else float("inf"),
                "outgoing_flux_rate": float(out_rate[idx]),
                "raw_cfl": float(cfl[idx]),
                "region": str(reg[idx]),
            }
        )
    unique_regions = sorted({str(x) for x in reg.tolist()})
    region_counts = {r: int(np.count_nonzero(reg == r)) for r in unique_regions}
    return {
        "cell_count": int(cfl.size),
        "threshold_fractions": _convective_threshold_fractions(cfl),
        "raw_cfl_percentiles": _convective_percentiles(cfl),
        "region_counts": region_counts,
        "top_cells": rows,
    }


def _build_top_convective_face_audit(
    mesh: ImportedTetraMesh,
    *,
    flux: np.ndarray,
    face_cfl_raw: np.ndarray,
    face_length_scale: np.ndarray,
    face_velocity_normal_abs: np.ndarray,
    wall_faces: np.ndarray,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    top_k: int = 200,
) -> dict[str, Any]:
    n_faces = int(mesh.face_vertices.shape[0])
    labels = np.full((n_faces,), "interior", dtype=object)
    if wall_faces.size:
        labels[np.asarray(wall_faces, dtype=np.int64)] = "wall-adjacent"
    if outlet_faces.size:
        labels[np.asarray(outlet_faces, dtype=np.int64)] = "outlet"
    if right_faces.size:
        labels[np.asarray(right_faces, dtype=np.int64)] = "right inlet"
    if left_faces.size:
        labels[np.asarray(left_faces, dtype=np.int64)] = "left inlet"
    cfl = np.asarray(face_cfl_raw, dtype=np.float64)
    ls = np.asarray(face_length_scale, dtype=np.float64)
    un = np.asarray(face_velocity_normal_abs, dtype=np.float64)
    centers = np.asarray(mesh.face_centers, dtype=np.float64)
    area = np.asarray(mesh.face_areas, dtype=np.float64)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)

    order = np.argsort(-cfl)
    rows: list[dict[str, Any]] = []
    for fid in order[: int(max(top_k, 0))].tolist():
        rows.append(
            {
                "face_index": int(fid),
                "center": centers[fid].tolist(),
                "area": float(area[fid]),
                "owner_cell": int(c0[fid]),
                "neighbor_cell": int(c1[fid]),
                "local_speed_normal_abs": float(un[fid]),
                "local_length_scale": float(ls[fid])
                if np.isfinite(ls[fid])
                else float("inf"),
                "flux": float(np.asarray(flux, dtype=np.float64)[fid]),
                "raw_cfl": float(cfl[fid]),
                "region": str(labels[fid]),
            }
        )
    unique_regions = sorted({str(x) for x in labels.tolist()})
    region_counts = {r: int(np.count_nonzero(labels == r)) for r in unique_regions}
    return {
        "face_count": int(cfl.size),
        "threshold_fractions": _convective_threshold_fractions(cfl),
        "raw_cfl_percentiles": _convective_percentiles(cfl),
        "region_counts": region_counts,
        "top_faces": rows,
    }


def _convective_divergence_term(
    mesh: ImportedTetraMesh,
    *,
    velocity: np.ndarray,
    face_flux: np.ndarray,
    cell_volume: np.ndarray,
) -> np.ndarray:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    vol = np.maximum(np.asarray(cell_volume, dtype=np.float64), 1e-30)
    vel = np.asarray(velocity, dtype=np.float64)
    flux = np.asarray(face_flux, dtype=np.float64)
    n_cells = int(mesh.tetrahedra.shape[0])
    conv_div = np.zeros((n_cells, 3), dtype=np.float64)
    for comp in range(3):
        sum_comp = np.zeros((n_cells,), dtype=np.float64)
        phi = vel[:, comp]
        for fid in range(mesh.face_vertices.shape[0]):
            owner = int(c0[fid])
            neigh = int(c1[fid])
            q = float(flux[fid])
            if neigh >= 0:
                phi_up = float(phi[owner]) if q >= 0.0 else float(phi[neigh])
            else:
                phi_up = float(phi[owner])
            flux_phi = q * phi_up
            sum_comp[owner] += flux_phi
            if neigh >= 0:
                sum_comp[neigh] -= flux_phi
        conv_div[:, comp] = sum_comp / vol
    return conv_div


def _convective_predictor_step_torch(
    mesh: ImportedTetraMesh,
    *,
    velocity: np.ndarray,
    face_flux: np.ndarray,
    cell_volume: np.ndarray,
    update_scale: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Torch convection requested but torch is unavailable."
        ) from exc

    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError("Torch convection acceleration requires a CUDA device.")
    dtype = torch.float64
    cache_key = (
        id(mesh.face_to_cells),
        id(mesh.face_normals),
        id(mesh.face_areas),
        id(mesh.cell_volumes),
        str(dev),
        str(dtype),
    )
    geometry = _TORCH_CONVECTION_CACHE.get(cache_key)
    if geometry is None:
        face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64)
        owner = torch.as_tensor(face_to_cells[:, 0], dtype=torch.long, device=dev)
        neigh = torch.as_tensor(face_to_cells[:, 1], dtype=torch.long, device=dev)
        interior = neigh >= 0
        geometry = {
            "owner": owner,
            "neigh": neigh,
            "neigh_safe": torch.clamp(neigh, min=0),
            "interior": interior,
            "has_interior": bool(np.any(face_to_cells[:, 1] >= 0)),
            "normals": torch.as_tensor(
                np.asarray(mesh.face_normals, dtype=np.float64),
                dtype=dtype,
                device=dev,
            ),
            "areas": torch.as_tensor(
                np.asarray(mesh.face_areas, dtype=np.float64),
                dtype=dtype,
                device=dev,
            ),
            "inverse_volume": 1.0
            / torch.clamp(
                torch.as_tensor(
                    np.asarray(cell_volume, dtype=np.float64),
                    dtype=dtype,
                    device=dev,
                ),
                min=1e-30,
            ),
        }
        _TORCH_CONVECTION_CACHE[cache_key] = geometry

    vel_t = torch.as_tensor(
        np.asarray(velocity, dtype=np.float64), dtype=dtype, device=dev
    )
    flux_t = torch.as_tensor(
        np.asarray(face_flux, dtype=np.float64), dtype=dtype, device=dev
    )
    owner = geometry["owner"]
    neigh = geometry["neigh"]
    neigh_safe = geometry["neigh_safe"]
    interior = geometry["interior"]

    owner_upwind = (flux_t >= 0.0) | (~interior)
    upwind_velocity = torch.where(
        owner_upwind[:, None],
        vel_t[owner],
        vel_t[neigh_safe],
    )
    transported = flux_t[:, None] * upwind_velocity
    cell_sum = torch.zeros_like(vel_t)
    cell_sum.index_add_(0, owner, transported)
    if geometry["has_interior"]:
        cell_sum.index_add_(0, neigh[interior], -transported[interior])
    convective_divergence = cell_sum * geometry["inverse_volume"][:, None]
    velocity_next = vel_t - float(update_scale) * convective_divergence

    face_velocity = velocity_next[owner].clone()
    if geometry["has_interior"]:
        face_velocity[interior] = 0.5 * (
            velocity_next[owner[interior]] + velocity_next[neigh[interior]]
        )
    flux_next = (
        torch.sum(face_velocity * geometry["normals"], dim=1) * geometry["areas"]
    )
    velocity_numpy = velocity_next.detach().cpu().numpy()
    flux_numpy = flux_next.detach().cpu().numpy()
    _remember_torch_array(velocity_numpy, velocity_next, device=str(dev))
    _remember_torch_array(flux_numpy, flux_next, device=str(dev))
    return velocity_numpy, flux_numpy


def _convective_cell_cfl_from_flux(
    mesh: ImportedTetraMesh,
    *,
    face_flux: np.ndarray,
    flow_dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    flux = np.asarray(face_flux, dtype=np.float64)
    dt = float(flow_dt)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    n_cells = int(mesh.tetrahedra.shape[0])
    out_rate = np.zeros((n_cells,), dtype=np.float64)
    np.add.at(out_rate, c0, np.maximum(flux, 0.0))
    interior = c1 >= 0
    if np.any(interior):
        np.add.at(out_rate, c1[interior], np.maximum(-flux[interior], 0.0))
    cfl_cell = (dt * out_rate) / vol
    return out_rate, cfl_cell


def compute_tetra_convective_cfl_rate(
    mesh: ImportedTetraMesh,
    face_flux: np.ndarray,
) -> dict[str, Any]:
    out_rate, cfl_rate = _convective_cell_cfl_from_flux(
        mesh, face_flux=face_flux, flow_dt=1.0
    )
    arr = np.asarray(cfl_rate, dtype=np.float64)
    return {
        "cell_cfl_rate": np.asarray(arr, dtype=np.float64),
        "outgoing_flux_rate": np.asarray(out_rate, dtype=np.float64),
        "cfl_rate_max": float(np.max(arr)) if arr.size else 0.0,
        "cfl_rate_p95": float(np.percentile(arr, 95.0)) if arr.size else 0.0,
    }


def apply_tetra_convective_predictor(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
    config: TetraFlowConfig,
    *,
    flow_dt: float,
) -> TetraFlowState:
    requested_config = config
    config = resolve_tetra_flow_numerical_profile(config)
    _activate_torch_cache_context(mesh, device=str(config.device))
    left_faces, right_faces = _inlet_face_sets(mesh)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    predictor_enabled = bool(config.enable_convective_predictor) and (
        not bool(config.disable_convective_predictor)
    )
    vel0 = np.asarray(state.cell_velocity, dtype=np.float64)
    if vel0.shape != (mesh.tetrahedra.shape[0], 3):
        vel0 = _reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            np.asarray(state.face_flux, dtype=np.float64),
            wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
            wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
        )
    flux0 = np.asarray(state.face_flux, dtype=np.float64).copy()
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    dt = float(flow_dt)

    div_before_diag = compute_tetra_flux_divergence(
        mesh,
        flux0,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    kin_before = float(np.mean(0.5 * np.sum(vel0 * vel0, axis=1))) if vel0.size else 0.0
    kin_before_volume_integral = (
        float(np.sum(vol * 0.5 * np.sum(vel0 * vel0, axis=1))) if vel0.size else 0.0
    )

    out_rate, cfl_cell = _convective_cell_cfl_from_flux(
        mesh, face_flux=flux0, flow_dt=dt
    )
    cfl_raw_max = float(np.max(cfl_cell)) if cfl_cell.size else 0.0
    cfl_raw_p95 = float(np.percentile(cfl_cell, 95.0)) if cfl_cell.size else 0.0
    cfl_limit = float(config.convective_cfl_limit)
    damping_cli = float(config.convective_predictor_damping)
    stabilization_mode = str(config.convective_stabilization_mode)
    boundary_contract_mode = str(config.convective_substep_boundary_contract)
    cfl_scale = 1.0
    damping_effective = damping_cli
    substepping_used = False
    substep_count_unclamped = 1
    substep_count = 1
    substep_dt = dt
    substep_cap_hit = False
    substep_cfl_max_values: list[float] = []
    substep_cfl_p95_values: list[float] = []
    substep_runtime = 0.0
    convective_execution_backend = "numpy"
    convective_execution_device = "cpu"
    convective_torch_cuda_used = False
    convective_numpy_fallback_reason = ""
    cuda_convection_wall_supported = bool(
        str(config.wall_velocity_boundary_mode) == "slip"
        or _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)
    )
    torch_cuda_eligible = bool(
        predictor_enabled
        and stabilization_mode == "auto_damping"
        and str(config.backend) == "torch"
        and str(config.device).startswith("cuda")
        and cuda_convection_wall_supported
    )
    if predictor_enabled and str(config.backend) == "torch" and not torch_cuda_eligible:
        if stabilization_mode != "auto_damping":
            convective_numpy_fallback_reason = (
                "CUDA convection currently preserves the NumPy path for substepping"
            )
        elif not str(config.device).startswith("cuda"):
            convective_numpy_fallback_reason = "flow execution device is not CUDA"
        elif not cuda_convection_wall_supported:
            convective_numpy_fallback_reason = (
                "CUDA convection preserves the NumPy path for legacy isotropic "
                "no-slip walls"
            )

    vel_conv = vel0.copy()
    flux_conv = flux0.copy()
    if predictor_enabled and stabilization_mode == "substepping":
        substepping_used = True
        substep_count_unclamped = int(
            max(1, np.ceil(cfl_raw_max / max(cfl_limit, 1e-30)))
        )
        substep_count = int(
            min(substep_count_unclamped, int(config.max_convective_substeps))
        )
        substep_cap_hit = bool(
            substep_count_unclamped > int(config.max_convective_substeps)
        )
        if substep_cap_hit and bool(config.fail_on_convective_substep_cap):
            raise RuntimeError(
                "convective substep cap exceeded: required "
                f"{substep_count_unclamped}, max {int(config.max_convective_substeps)}"
            )
        substep_dt = float(dt / max(substep_count, 1))
        damping_effective = float(damping_cli)
        cfl_scale = 1.0
        t_sub0 = perf_counter()
        vel_curr = vel0.copy()
        flux_curr = flux0.copy()
        for _ in range(substep_count):
            conv_div = _convective_divergence_term(
                mesh,
                velocity=vel_curr,
                face_flux=flux_curr,
                cell_volume=vol,
            )
            vel_curr = vel_curr - (damping_cli * substep_dt) * conv_div
            flux_curr = _face_flux_from_cell_velocity_numpy(mesh, vel_curr)
            if boundary_contract_mode == "every_substep":
                _apply_face_flux_boundary_conditions_inplace(
                    mesh,
                    flux_curr,
                    inlet_speed=float(config.inlet_speed),
                    left_inlet_faces=left_faces,
                    right_inlet_faces=right_faces,
                    outlet_faces=outlet_faces,
                    wall_faces=wall_faces,
                    outlet_contract_mode=(
                        config.viscous_predictor_outlet_contract_mode
                    ),
                )
                vel_curr = _reconstruct_cell_velocity_from_face_flux_numpy(
                    mesh,
                    flux_curr,
                    wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(
                        config
                    ),
                    wall_tangential_no_slip_strength=(
                        config.wall_tangential_no_slip_strength
                    ),
                )
            out_rate_sub, cfl_sub = _convective_cell_cfl_from_flux(
                mesh, face_flux=flux_curr, flow_dt=substep_dt * damping_cli
            )
            _ = out_rate_sub
            substep_cfl_max_values.append(
                float(np.max(cfl_sub)) if cfl_sub.size else 0.0
            )
            substep_cfl_p95_values.append(
                float(np.percentile(cfl_sub, 95.0)) if cfl_sub.size else 0.0
            )
        substep_runtime = float(perf_counter() - t_sub0)
        vel_conv = vel_curr
        flux_conv = flux_curr
        if boundary_contract_mode == "end_only":
            _apply_face_flux_boundary_conditions_inplace(
                mesh,
                flux_conv,
                inlet_speed=float(config.inlet_speed),
                left_inlet_faces=left_faces,
                right_inlet_faces=right_faces,
                outlet_faces=outlet_faces,
                wall_faces=wall_faces,
                outlet_contract_mode=config.viscous_predictor_outlet_contract_mode,
            )
            vel_conv = _reconstruct_cell_velocity_from_face_flux_numpy(
                mesh,
                flux_conv,
                wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
                wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
            )
    elif predictor_enabled:
        if bool(config.disable_convective_auto_damping):
            cfl_scale = 1.0
        else:
            cfl_scale = (
                1.0
                if cfl_raw_max <= cfl_limit
                else float(cfl_limit / max(cfl_raw_max, 1e-30))
            )
        damping_effective = float(damping_cli * cfl_scale)
        if torch_cuda_eligible:
            vel_conv, flux_conv = _convective_predictor_step_torch(
                mesh,
                velocity=vel0,
                face_flux=flux0,
                cell_volume=vol,
                update_scale=float(damping_effective * dt),
                device=str(config.device),
            )
            convective_execution_backend = "torch"
            convective_execution_device = str(config.device)
            convective_torch_cuda_used = True
        else:
            conv_div = _convective_divergence_term(
                mesh,
                velocity=vel0,
                face_flux=flux0,
                cell_volume=vol,
            )
            vel_conv = vel0 - damping_effective * dt * conv_div
            flux_conv = _face_flux_from_cell_velocity_numpy(mesh, vel_conv)
    else:
        damping_effective = 0.0
        cfl_scale = 1.0

    cfl_cell_effective = cfl_cell * (
        damping_effective if stabilization_mode == "auto_damping" else 1.0
    )
    if substepping_used:
        cfl_effective_max = (
            max(substep_cfl_max_values) if substep_cfl_max_values else 0.0
        )
        cfl_effective_p95 = (
            max(substep_cfl_p95_values) if substep_cfl_p95_values else 0.0
        )
    else:
        cfl_effective_max = (
            float(np.max(cfl_cell_effective)) if cfl_cell_effective.size else 0.0
        )
        cfl_effective_p95 = (
            float(np.percentile(cfl_cell_effective, 95.0))
            if cfl_cell_effective.size
            else 0.0
        )
    cfl_warning_raw = bool(cfl_raw_max > cfl_limit)
    cfl_warning_effective = bool(cfl_effective_max > cfl_limit)
    auto_damping_used = bool(
        predictor_enabled
        and stabilization_mode == "auto_damping"
        and (not bool(config.disable_convective_auto_damping))
        and (cfl_scale < (1.0 - 1e-12))
    )
    if not predictor_enabled:
        auto_damping_reason = "convective predictor disabled"
    elif stabilization_mode == "substepping":
        auto_damping_reason = (
            f"substepping mode: N={substep_count} (required={substep_count_unclamped}, "
            f"cap={int(config.max_convective_substeps)})"
        )
    elif bool(config.disable_convective_auto_damping):
        auto_damping_reason = "convective auto damping disabled by config"
    elif auto_damping_used:
        auto_damping_reason = (
            f"raw CFL {cfl_raw_max:.6g} exceeds limit {cfl_limit:.6g}; "
            f"auto scale {cfl_scale:.6g}"
        )
    else:
        auto_damping_reason = "raw CFL within limit; auto damping not used"
    convective_dt_effective = (
        float(substep_dt * damping_effective)
        if (predictor_enabled and substepping_used)
        else (float(dt * damping_effective) if predictor_enabled else 0.0)
    )

    div_after_conv_diag = compute_tetra_flux_divergence(
        mesh,
        flux_conv,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )

    dv = vel_conv - vel0
    dv_mag = np.linalg.norm(dv, axis=1)
    kin_after = (
        float(np.mean(0.5 * np.sum(vel_conv * vel_conv, axis=1)))
        if vel_conv.size
        else 0.0
    )
    kin_after_volume_integral = (
        float(np.sum(vol * 0.5 * np.sum(vel_conv * vel_conv, axis=1)))
        if vel_conv.size
        else 0.0
    )

    diag = dict(state.diagnostics)
    diag["numerical_profile_resolution"] = _numerical_profile_resolution_diagnostics(
        requested_config, config
    )
    speed0 = np.linalg.norm(vel0, axis=1)
    out_area_equiv = np.where(speed0 > 1e-30, out_rate / np.maximum(speed0, 1e-30), 0.0)
    local_length_scale = np.where(
        out_area_equiv > 1e-30,
        vol / np.maximum(out_area_equiv, 1e-30),
        np.inf,
    )
    area = np.maximum(np.asarray(mesh.face_areas, dtype=np.float64), 1e-30)
    c1_local = np.asarray(c1, dtype=np.int64)
    neigh_ok = c1_local >= 0
    vol_owner = np.asarray(vol[c0], dtype=np.float64)
    vol_nei = np.where(neigh_ok, vol[c1_local], vol_owner)
    vol_min = np.minimum(vol_owner, vol_nei)
    face_vel_normal_abs = np.abs(np.asarray(flux0, dtype=np.float64)) / area
    face_length_scale = vol_min / area
    face_cfl_raw = dt * face_vel_normal_abs / np.maximum(face_length_scale, 1e-30)
    cell_region_labels = _estimate_cell_regions(
        mesh,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    cfl_definition_report = {
        "name": "outgoing_volume_fraction_cfl",
        "raw_cfl_formula": "raw_cfl_cell = dt * outgoing_flux_rate_cell / cell_volume",
        "outgoing_flux_rate_formula": (
            "outgoing_flux_rate_cell = sum(max(q_face,0) for owner faces) + "
            "sum(max(-q_face,0) for neighbor faces)"
        ),
        "local_length_scale_formula": (
            "h_local = cell_volume / (outgoing_flux_rate_cell / max(|u_cell|, eps))"
        ),
        "effective_cfl_formula": (
            "auto_damping: effective_cfl = raw_cfl * damping_effective; "
            "substepping: per-substep effective_cfl computed with dt_sub and requested damping"
        ),
        "dt_used_for_raw_cfl": float(dt),
        "dt_effective_used": float(convective_dt_effective),
        "velocity_measure": "cell velocity magnitude for local-length diagnostic, face-normal velocity for face audit",
        "face_relation": "face_cfl_raw = dt * |q_face| / min(volume_owner, volume_neighbor_or_owner)",
        "stabilization_mode": str(stabilization_mode),
        "boundary_contract_mode": str(boundary_contract_mode),
    }
    top_cfl_cells = _build_top_convective_cell_audit(
        mesh,
        vel=vel0,
        out_rate=out_rate,
        length_scale=local_length_scale,
        cfl_raw=cfl_cell,
        region_labels=cell_region_labels,
    )
    top_cfl_faces = _build_top_convective_face_audit(
        mesh,
        flux=flux0,
        face_cfl_raw=face_cfl_raw,
        face_length_scale=face_length_scale,
        face_velocity_normal_abs=face_vel_normal_abs,
        wall_faces=wall_faces,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
    )
    diag["convective_predictor"] = {
        "convective_predictor_used": bool(predictor_enabled),
        "convective_predictor_disabled_by_flag": bool(
            config.disable_convective_predictor
        ),
        "convective_auto_damping_disabled": bool(
            config.disable_convective_auto_damping
        ),
        "convective_stabilization_mode": str(stabilization_mode),
        "convective_substep_boundary_contract_mode": str(boundary_contract_mode),
        "convective_predictor_outlet_contract_mode": str(
            config.viscous_predictor_outlet_contract_mode
        ),
        "convective_execution_backend": str(convective_execution_backend),
        "convective_execution_device": str(convective_execution_device),
        "convective_torch_cuda_used": bool(convective_torch_cuda_used),
        "convective_cuda_handoff_available": bool(
            convective_torch_cuda_used
            and _find_torch_array(flux_conv, device=str(config.device)) is not None
        ),
        "convective_numpy_fallback_reason": str(convective_numpy_fallback_reason),
        "convective_cfl_limit": float(cfl_limit),
        "convective_cfl_raw_max": float(cfl_raw_max),
        "convective_cfl_raw_p95": float(cfl_raw_p95),
        "convective_cfl_effective_max": float(cfl_effective_max),
        "convective_cfl_effective_p95": float(cfl_effective_p95),
        "convective_cfl_warning_raw": bool(cfl_warning_raw),
        "convective_cfl_warning_effective": bool(cfl_warning_effective),
        # Backward-compat aliases for existing reports/tests.
        "convective_cfl_max": float(cfl_raw_max),
        "convective_cfl_p95": float(cfl_raw_p95),
        "convective_cfl_warning": bool(cfl_warning_raw),
        "convective_predictor_damping_requested": float(damping_cli),
        "convective_predictor_damping": float(damping_cli),
        "convective_predictor_cfl_scale": float(cfl_scale),
        "convective_predictor_damping_effective": float(damping_effective),
        "convective_auto_damping_used": bool(auto_damping_used),
        "convective_auto_damping_reason": str(auto_damping_reason),
        "convective_substepping_used": bool(substepping_used),
        "convective_substep_count": int(substep_count),
        "convective_substep_count_unclamped": int(substep_count_unclamped),
        "convective_substep_cap_hit": bool(substep_cap_hit),
        "convective_substep_dt": float(substep_dt),
        "convective_cfl_per_substep_max": float(cfl_effective_max),
        "convective_cfl_per_substep_p95": float(cfl_effective_p95),
        "convective_substepping_runtime_seconds": float(substep_runtime),
        "convective_dt": float(dt),
        "convective_dt_effective": float(convective_dt_effective),
        "convective_delta_velocity_max": float(np.max(dv_mag)) if dv_mag.size else 0.0,
        "convective_delta_velocity_l2": float(np.sqrt(np.mean(dv_mag * dv_mag)))
        if dv_mag.size
        else 0.0,
        "kinetic_energy_before_convection": float(kin_before),
        "kinetic_energy_after_convection": float(kin_after),
        "kinetic_energy_volume_integral_before_convection_m5_s2": float(
            kin_before_volume_integral
        ),
        "kinetic_energy_volume_integral_after_convection_m5_s2": float(
            kin_after_volume_integral
        ),
        "divergence_before_convection_max": float(
            div_before_diag.get("divergence_max_abs", 0.0)
        ),
        "divergence_before_convection_l2": float(
            div_before_diag.get("divergence_l2", 0.0)
        ),
        "divergence_after_convection_before_projection_max": float(
            div_after_conv_diag.get("divergence_max_abs", 0.0)
        ),
        "divergence_after_convection_before_projection_l2": float(
            div_after_conv_diag.get("divergence_l2", 0.0)
        ),
        "convective_cfl_definition_report": cfl_definition_report,
        "top_convective_cfl_cells": top_cfl_cells,
        "top_convective_cfl_faces": top_cfl_faces,
        "arrays": {
            "velocity_before_convection": np.asarray(vel0, dtype=np.float64),
            "velocity_after_convection": np.asarray(vel_conv, dtype=np.float64),
            "face_flux_before_convection": np.asarray(flux0, dtype=np.float64),
            "face_flux_after_convection": np.asarray(flux_conv, dtype=np.float64),
            "convective_cfl_cell_raw": np.asarray(cfl_cell, dtype=np.float64),
            "convective_cfl_face_raw": np.asarray(face_cfl_raw, dtype=np.float64),
            "convective_local_speed_cell": np.asarray(speed0, dtype=np.float64),
            "convective_local_length_scale_cell": np.asarray(
                local_length_scale, dtype=np.float64
            ),
            "convective_cell_outgoing_flux_rate": np.asarray(
                out_rate, dtype=np.float64
            ),
            "convective_cell_region_labels": np.asarray(
                cell_region_labels, dtype=object
            ),
        },
    }
    return TetraFlowState(
        cell_velocity=vel_conv,
        face_flux=flux_conv,
        pressure=np.asarray(state.pressure, dtype=np.float64).copy(),
        diagnostics=diag,
    )


def apply_tetra_stokes_viscous_predictor(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
    config: TetraFlowConfig,
    *,
    flow_dt: float,
) -> TetraFlowState:
    requested_config = config
    config = resolve_tetra_flow_numerical_profile(config)
    _activate_torch_cache_context(mesh, device=str(config.device))
    left_faces, right_faces = _inlet_face_sets(mesh)
    inlet_faces = np.unique(np.concatenate((left_faces, right_faces))).astype(np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    mode = str(config.viscous_predictor_mode)
    viscous_execution_backend = "numpy"
    viscous_execution_device = "cpu"
    viscous_torch_cuda_used = False
    viscous_numpy_fallback_reason = ""
    viscous_cuda_input_reused = False
    viscous_cuda_finalization_used = False
    viscous_cuda_host_to_device_bytes_avoided = 0
    viscous_cuda_no_slip_used = False
    cuda_flux_contract: np.ndarray | None = None
    cuda_velocity_raw: np.ndarray | None = None
    cuda_velocity_contract: np.ndarray | None = None
    viscous_torch_cuda_eligible = bool(
        mode == "face_flux_laplacian_substepped"
        and bool(config.viscous_face_flux_laplacian_vectorized)
        and bool(config.torch_cuda_viscosity_enabled)
        and str(config.backend) == "torch"
        and str(config.device).startswith("cuda")
        and str(config.wall_velocity_boundary_mode) == "slip"
    )
    viscous_no_slip_torch_cuda_eligible = bool(
        mode == "explicit_cell_velocity_laplacian_substepped_conservative"
        and config.viscous_nonorthogonal_correction_mode == "deferred_lsq"
        and bool(config.torch_cuda_viscosity_enabled)
        and str(config.backend) == "torch"
        and str(config.device).startswith("cuda")
        and _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)
    )
    if (
        mode not in {"none", "no_viscous_debug_copy"}
        and str(config.backend) == "torch"
        and not (viscous_torch_cuda_eligible or viscous_no_slip_torch_cuda_eligible)
    ):
        if not str(config.device).startswith("cuda"):
            viscous_numpy_fallback_reason = "flow execution device is not CUDA"
        elif not bool(config.torch_cuda_viscosity_enabled):
            viscous_numpy_fallback_reason = (
                "CUDA viscosity is disabled by configuration"
            )
        elif _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode):
            viscous_numpy_fallback_reason = (
                "CUDA no-slip viscosity requires the conservative cell-velocity "
                "predictor with deferred LSQ correction"
            )
        elif mode != "face_flux_laplacian_substepped":
            viscous_numpy_fallback_reason = "CUDA viscosity currently preserves the NumPy path for this predictor mode"
        elif not bool(config.viscous_face_flux_laplacian_vectorized):
            viscous_numpy_fallback_reason = (
                "CUDA viscosity requires the vectorized face-flux Laplacian"
            )
    viscous_nonorthogonal_enabled = bool(
        config.viscous_nonorthogonal_correction_mode == "deferred_lsq"
        and mode
        in {
            "explicit_cell_velocity_laplacian_substepped",
            "explicit_cell_velocity_laplacian_substepped_conservative",
        }
    )
    momentum_pressure_state_enabled = bool(
        config.projection_cell_velocity_update_mode == "momentum_pressure_corrected"
    )
    vel0 = np.asarray(state.cell_velocity, dtype=np.float64)
    if vel0.shape != (mesh.tetrahedra.shape[0], 3):
        vel0 = _reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            np.asarray(state.face_flux, dtype=np.float64),
            wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
            wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
        )
    v_before = vel0.copy()
    state_face_flux = np.asarray(state.face_flux, dtype=np.float64)
    flux_before = state_face_flux.copy()
    ke_before = 0.5 * np.sum(v_before * v_before, axis=1)
    kin_before = float(np.mean(ke_before)) if ke_before.size else 0.0

    viscous_geometry = _cached_viscous_tpfa_geometry(mesh)
    c0 = np.asarray(viscous_geometry["c0"], dtype=np.int64)
    c1 = np.asarray(viscous_geometry["c1"], dtype=np.int64)
    vol = np.asarray(viscous_geometry["volume"], dtype=np.float64)
    nu = float(config.kinematic_viscosity)
    dt = float(flow_dt)
    own = np.asarray(viscous_geometry["owner"], dtype=np.int64)
    nei = np.asarray(viscous_geometry["neighbor"], dtype=np.int64)
    w = np.asarray(viscous_geometry["weight"], dtype=np.float64)
    row_coef = np.asarray(viscous_geometry["row_coefficient"], dtype=np.float64).copy()
    wall_sink = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
    wall_tangential_operator = np.zeros(
        (mesh.tetrahedra.shape[0], 3, 3), dtype=np.float64
    )
    wall_flux_resistance_bundle: dict[str, np.ndarray] = {
        "cell_ids": np.zeros((0,), dtype=np.int64),
        "cell_face_ids": np.zeros((0, 4), dtype=np.int64),
        "variable_face_mask": np.zeros((0, 4), dtype=bool),
        "operator": np.zeros((0, 4, 4), dtype=np.float64),
        "active_face_ids": np.zeros((0,), dtype=np.int64),
    }
    tangential_wall_mode = _uses_tangential_no_slip_wall(
        config.wall_velocity_boundary_mode
    )
    shear_face_flux_enabled = bool(
        tangential_wall_mode and config.wall_tangential_shear_face_flux_enabled
    )
    cell_velocity_wall_momentum_enabled = bool(
        tangential_wall_mode
        and config.wall_tangential_cell_velocity_momentum_enabled
        and mode
        in {
            "explicit_cell_velocity_laplacian_substepped",
            "explicit_cell_velocity_laplacian_substepped_conservative",
        }
    )
    wall_operator_needed = bool(
        tangential_wall_mode
        and (
            shear_face_flux_enabled
            or cell_velocity_wall_momentum_enabled
            or _uses_wall_flux_stokes_resistance(config)
        )
    )
    if wall_operator_needed:
        wall_tangential_operator = _cached_wall_tangential_no_slip_operator(
            mesh,
            wall_faces,
            strength=float(config.wall_tangential_no_slip_strength),
        )
        if cell_velocity_wall_momentum_enabled and not viscous_nonorthogonal_enabled:
            wall_row_coef = np.maximum(
                np.linalg.eigvalsh(wall_tangential_operator)[:, -1],
                0.0,
            )
            row_coef = row_coef + wall_row_coef
        if _uses_wall_flux_stokes_resistance(config):
            wall_flux_resistance_bundle = _cached_wall_flux_stokes_resistance(
                mesh,
                wall_faces,
                strength=float(config.wall_tangential_no_slip_strength),
            )
    elif (
        _uses_legacy_isotropic_no_slip_wall(config.wall_velocity_boundary_mode)
        and not viscous_nonorthogonal_enabled
    ):
        wall_sink = _cached_wall_no_slip_sink_coefficient(mesh, wall_faces)
        row_coef = row_coef + wall_sink
    viscous_nonorthogonal_geometry: dict[str, np.ndarray | float] | None = None
    viscous_wall_face_velocity = np.zeros((wall_faces.size, 3), dtype=np.float64)
    viscous_inlet_face_velocity = np.zeros((inlet_faces.size, 3), dtype=np.float64)
    viscous_outlet_normal_gradient = np.zeros((outlet_faces.size, 3), dtype=np.float64)
    viscous_wall_sink = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
    viscous_inlet_sink = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
    viscous_inlet_source = np.zeros_like(v_before)
    viscous_nonorthogonal_stability_bound_raw = 0.0
    viscous_nonorthogonal_stability_target = 0.16
    viscous_base_stability_target = 0.05
    viscous_nonorthogonal_bound_substeps = 1
    viscous_base_bound_substeps = 1
    if viscous_nonorthogonal_enabled:
        viscous_nonorthogonal_geometry = _cached_viscous_nonorthogonal_geometry(
            mesh,
            inlet_faces=inlet_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
        )
        viscous_wall_sink = np.asarray(
            viscous_nonorthogonal_geometry["wall_orthogonal_sink_per_volume"],
            dtype=np.float64,
        )
        viscous_inlet_sink = np.asarray(
            viscous_nonorthogonal_geometry["inlet_orthogonal_sink_per_volume"],
            dtype=np.float64,
        )
        viscous_inlet_face_velocity = -float(config.inlet_speed) * np.asarray(
            viscous_nonorthogonal_geometry["inlet_normal"], dtype=np.float64
        )
        inlet_owner_nonorthogonal = np.asarray(
            viscous_nonorthogonal_geometry["inlet_owner"], dtype=np.int64
        )
        inlet_t_nonorthogonal = np.asarray(
            viscous_nonorthogonal_geometry["inlet_t"], dtype=np.float64
        )
        if inlet_owner_nonorthogonal.size:
            np.add.at(
                viscous_inlet_source,
                inlet_owner_nonorthogonal,
                (inlet_t_nonorthogonal / vol[inlet_owner_nonorthogonal])[:, None]
                * viscous_inlet_face_velocity,
            )
        row_coef = (
            np.asarray(
                viscous_nonorthogonal_geometry["interior_row_coefficient"],
                dtype=np.float64,
            )
            + viscous_wall_sink
            + viscous_inlet_sink
        )
        explicit_operator_bound = 2.0 * np.asarray(
            viscous_nonorthogonal_geometry["interior_row_coefficient"],
            dtype=np.float64,
        ) + np.asarray(
            viscous_nonorthogonal_geometry["correction_laplacian_infinity_norm_bound"],
            dtype=np.float64,
        )
        viscous_nonorthogonal_stability_bound_raw = (
            float(nu * dt * np.max(explicit_operator_bound))
            if explicit_operator_bound.size
            else 0.0
        )
    stability_metric_raw = float(nu * dt * np.max(row_coef)) if row_coef.size else 0.0
    if viscous_nonorthogonal_enabled:
        viscous_base_bound_substeps = max(
            1,
            int(np.ceil(stability_metric_raw / viscous_base_stability_target)),
        )
        viscous_nonorthogonal_bound_substeps = max(
            1,
            int(
                np.ceil(
                    viscous_nonorthogonal_stability_bound_raw
                    / viscous_nonorthogonal_stability_target
                )
            ),
        )
        substeps = max(
            viscous_base_bound_substeps,
            viscous_nonorthogonal_bound_substeps,
        )
        target_metric = viscous_base_stability_target
    else:
        target_metric = (
            0.2
            if mode != "explicit_cell_velocity_laplacian_substepped_conservative"
            else 0.08
        )
        substeps = (
            max(1, int(np.ceil(stability_metric_raw / target_metric)))
            if stability_metric_raw > 0.0
            else 1
        )
    sub_dt = float(dt / substeps)
    wall_flux_resistance_prepared_solver: dict[str, Any] | None = None
    if _uses_wall_flux_stokes_resistance(config):
        wall_flux_resistance_prepared_solver = (
            _prepare_wall_flux_stokes_resistance_solver(
                wall_flux_resistance_bundle,
                alpha=float(nu)
                * float(sub_dt)
                * float(config.wall_flux_stokes_resistance_strength),
            )
        )

    v_pred = v_before.copy()
    v_pred_no_wall = v_before.copy()
    flux_pred_raw = flux_before.copy()
    predictor_used = bool(mode not in {"none"})
    capped_updates = 0
    total_updates = 0
    capped_faces_unique: set[int] = set()
    wall_flux_resistance_iterations = 0
    wall_flux_resistance_converged = True
    wall_flux_resistance_residual_l2 = 0.0
    wall_flux_resistance_method = "disabled"
    wall_shear_face_flux_applications = 0
    wall_shear_face_flux_delta_l2 = 0.0
    wall_shear_face_flux_active_cells = 0
    wall_shear_face_flux_speed_before = 0.0
    wall_shear_face_flux_speed_after = 0.0
    local_conservation_diag: dict[str, float | int] = {
        "iterations": 0,
        "residual_l2_before": 0.0,
        "residual_l2_after": 0.0,
        "residual_max_abs_before": 0.0,
        "residual_max_abs_after": 0.0,
        "delta_l2": 0.0,
        "delta_max_abs": 0.0,
    }
    local_conservative_target_mode = "preserve_non_wall_cells"
    wall_momentum_flux_delta_diag: dict[str, float | int | bool] = {
        "enabled": False,
        "iterations": 0,
        "residual_l2_before": 0.0,
        "residual_l2_after": 0.0,
        "residual_max_abs_before": 0.0,
        "residual_max_abs_after": 0.0,
        "delta_l2": 0.0,
        "delta_max_abs": 0.0,
    }
    viscous_nonorthogonal_flux = np.zeros(
        (mesh.face_vertices.shape[0], 3), dtype=np.float64
    )
    viscous_lsq_gradient = np.zeros((mesh.tetrahedra.shape[0], 3, 3), dtype=np.float64)
    viscous_nonorthogonal_laplacian = np.zeros_like(v_before)
    viscous_nonorthogonal_update_max = 0.0
    viscous_nonorthogonal_update_l2 = 0.0
    viscous_operator_energy_rate = 0.0
    if mode in {
        "explicit_cell_velocity_laplacian_substepped",
        "explicit_cell_velocity_laplacian_substepped_conservative",
    }:
        if (own.size or viscous_nonorthogonal_enabled) and nu > 0.0:
            if viscous_no_slip_torch_cuda_eligible:
                if viscous_nonorthogonal_geometry is None:
                    raise RuntimeError(
                        "viscous non-orthogonal geometry was not initialized"
                    )
                torch_result = _conservative_no_slip_viscous_predictor_step_torch(
                    mesh,
                    velocity=vel0,
                    nu=nu,
                    substep_dt=sub_dt,
                    substeps=substeps,
                    inlet_source=viscous_inlet_source,
                    wall_sink=viscous_wall_sink,
                    inlet_sink=viscous_inlet_sink,
                    wall_face_velocity=viscous_wall_face_velocity,
                    inlet_face_velocity=viscous_inlet_face_velocity,
                    outlet_normal_gradient=viscous_outlet_normal_gradient,
                    nonorthogonal_geometry=viscous_nonorthogonal_geometry,
                    device=str(config.device),
                )
                v_pred = np.asarray(torch_result["velocity"], dtype=np.float64)
                v_pred_no_wall = np.asarray(
                    torch_result["velocity_no_wall"], dtype=np.float64
                )
                viscous_nonorthogonal_flux = np.asarray(
                    torch_result["nonorthogonal_flux"], dtype=np.float64
                )
                viscous_nonorthogonal_laplacian = np.asarray(
                    torch_result["nonorthogonal_laplacian"], dtype=np.float64
                )
                viscous_nonorthogonal_update_max = float(
                    torch_result["nonorthogonal_update_max"]
                )
                viscous_nonorthogonal_update_l2 = float(
                    torch_result["nonorthogonal_update_l2"]
                )
                viscous_operator_energy_rate = float(
                    torch_result["operator_energy_rate"]
                )
                viscous_cuda_input_reused = bool(
                    torch_result["input_device_resident_reused"]
                )
                if viscous_cuda_input_reused:
                    viscous_cuda_host_to_device_bytes_avoided = int(vel0.nbytes)
                viscous_execution_backend = "torch"
                viscous_execution_device = str(config.device)
                viscous_torch_cuda_used = True
                viscous_cuda_no_slip_used = True
            else:
                for _ in range(substeps):
                    lap_no_wall = np.zeros_like(v_pred_no_wall)
                    if own.size:
                        dv_ij_no_wall = v_pred_no_wall[nei] - v_pred_no_wall[own]
                        np.add.at(
                            lap_no_wall,
                            own,
                            (w / vol[own])[:, None] * dv_ij_no_wall,
                        )
                        np.add.at(
                            lap_no_wall,
                            nei,
                            (w / vol[nei])[:, None] * (-dv_ij_no_wall),
                        )
                    v_pred_no_wall = v_pred_no_wall + (nu * sub_dt) * lap_no_wall
                    lap = np.zeros_like(v_pred)
                    if own.size:
                        dv_ij = v_pred[nei] - v_pred[own]
                        np.add.at(lap, own, (w / vol[own])[:, None] * dv_ij)
                        np.add.at(lap, nei, (w / vol[nei])[:, None] * (-dv_ij))
                    if viscous_nonorthogonal_enabled:
                        if viscous_nonorthogonal_geometry is None:
                            raise RuntimeError(
                                "viscous non-orthogonal geometry was not initialized"
                            )
                        (
                            viscous_nonorthogonal_flux,
                            viscous_lsq_gradient,
                        ) = _viscous_nonorthogonal_face_flux_numpy(
                            mesh,
                            v_pred,
                            wall_face_velocity=viscous_wall_face_velocity,
                            inlet_face_velocity=viscous_inlet_face_velocity,
                            outlet_normal_gradient=viscous_outlet_normal_gradient,
                            geometry=viscous_nonorthogonal_geometry,
                        )
                        viscous_nonorthogonal_laplacian = (
                            _vector_face_flux_laplacian_numpy(
                                mesh, viscous_nonorthogonal_flux
                            )
                        )
                        lap = lap + viscous_nonorthogonal_laplacian
                        full_operator_laplacian = (
                            lap
                            + viscous_inlet_source
                            - (viscous_wall_sink + viscous_inlet_sink)[:, None] * v_pred
                        )
                        viscous_operator_energy_rate = float(
                            np.sum(vol[:, None] * v_pred * full_operator_laplacian)
                        )
                        nonorthogonal_update = (
                            nu * sub_dt * viscous_nonorthogonal_laplacian
                        )
                        nonorthogonal_update_magnitude = np.linalg.norm(
                            nonorthogonal_update, axis=1
                        )
                        viscous_nonorthogonal_update_max = max(
                            viscous_nonorthogonal_update_max,
                            float(np.max(nonorthogonal_update_magnitude))
                            if nonorthogonal_update_magnitude.size
                            else 0.0,
                        )
                        viscous_nonorthogonal_update_l2 = max(
                            viscous_nonorthogonal_update_l2,
                            float(
                                np.sqrt(
                                    np.mean(
                                        nonorthogonal_update_magnitude
                                        * nonorthogonal_update_magnitude
                                    )
                                )
                            )
                            if nonorthogonal_update_magnitude.size
                            else 0.0,
                        )
                        explicit_velocity = v_pred + (nu * sub_dt) * lap
                        boundary_denominator = 1.0 + (nu * sub_dt) * (
                            viscous_wall_sink + viscous_inlet_sink
                        )
                        v_pred = (
                            explicit_velocity + (nu * sub_dt) * viscous_inlet_source
                        ) / boundary_denominator[:, None]
                    else:
                        v_pred = v_pred + (nu * sub_dt) * lap
                    if (
                        cell_velocity_wall_momentum_enabled
                        and not viscous_nonorthogonal_enabled
                    ):
                        v_pred = _apply_wall_tangential_no_slip_implicit(
                            v_pred,
                            wall_operator=wall_tangential_operator,
                            nu=nu,
                            dt=sub_dt,
                        )
                    elif (
                        _uses_legacy_isotropic_no_slip_wall(
                            config.wall_velocity_boundary_mode
                        )
                        and not viscous_nonorthogonal_enabled
                    ):
                        v_pred = _apply_wall_no_slip_velocity_sink(
                            v_pred,
                            wall_sink=wall_sink,
                            nu=nu,
                            dt=sub_dt,
                        )
        if (
            mode == "explicit_cell_velocity_laplacian_substepped_conservative"
            and not momentum_pressure_state_enabled
        ):
            flux_delta = _face_flux_from_cell_velocity_numpy(
                mesh,
                v_pred_no_wall - v_before,
            )
            if cell_velocity_wall_momentum_enabled:
                wall_flux_delta = _face_flux_from_cell_velocity_numpy(
                    mesh,
                    v_pred - v_pred_no_wall,
                )
                wall_flux_delta, wall_flux_delta_diag_raw = (
                    _locally_conserve_interior_face_flux_cell_sums(
                        mesh,
                        wall_flux_delta,
                        target_cell_flux_sum=np.zeros(
                            (mesh.tetrahedra.shape[0],), dtype=np.float64
                        ),
                        iterations=8,
                    )
                )
                wall_momentum_flux_delta_diag = {
                    "enabled": True,
                    **dict(wall_flux_delta_diag_raw),
                }
                flux_delta = np.asarray(flux_delta, dtype=np.float64) + np.asarray(
                    wall_flux_delta,
                    dtype=np.float64,
                )
            else:
                flux_delta = np.asarray(flux_delta, dtype=np.float64) + (
                    _face_flux_from_cell_velocity_numpy(mesh, v_pred - v_pred_no_wall)
                )
            flux_pred_raw = np.asarray(flux_before, dtype=np.float64) + np.asarray(
                flux_delta,
                dtype=np.float64,
            )
        else:
            flux_pred_raw = _face_flux_from_cell_velocity_numpy(mesh, v_pred)
    elif mode == "face_flux_laplacian_substepped":
        q = (
            state_face_flux
            if viscous_torch_cuda_eligible
            else np.asarray(flux_before, dtype=np.float64).copy()
        )
        cap = float(config.viscous_face_flux_divergence_impact_cap)
        if bool(config.viscous_face_flux_laplacian_vectorized):
            face_ids, nb_mat, w_mat, w_sum = _cached_face_flux_laplacian_vector_stencil(
                mesh
            )
            if face_ids.size:
                if viscous_torch_cuda_eligible:
                    torch_step = _face_flux_viscous_predictor_step_torch(
                        mesh,
                        face_flux=q,
                        nu=nu,
                        substep_dt=sub_dt,
                        substeps=substeps,
                        divergence_impact_cap=cap,
                        device=str(config.device),
                    )
                    (
                        q,
                        capped_updates,
                        total_updates,
                        capped_face_ids,
                    ) = torch_step[:4]
                    viscous_cuda_input_reused = bool(
                        torch_step[4] if len(torch_step) > 4 else False
                    )
                    if viscous_cuda_input_reused:
                        viscous_cuda_host_to_device_bytes_avoided = int(
                            state_face_flux.nbytes
                        )
                    cuda_finalization = (
                        _finalize_slip_viscous_predictor_torch(
                            mesh,
                            face_flux_raw=q,
                            inlet_speed=float(config.inlet_speed),
                            left_inlet_faces=left_faces,
                            right_inlet_faces=right_faces,
                            outlet_faces=outlet_faces,
                            wall_faces=wall_faces,
                            outlet_contract_mode=(
                                config.viscous_predictor_outlet_contract_mode
                            ),
                            device=str(config.device),
                        )
                        if not momentum_pressure_state_enabled
                        else None
                    )
                    if cuda_finalization is not None:
                        (
                            cuda_flux_contract,
                            cuda_velocity_raw,
                            cuda_velocity_contract,
                        ) = cuda_finalization
                        viscous_cuda_finalization_used = True
                    capped_faces_unique.update(
                        int(fid)
                        for fid in np.asarray(capped_face_ids, dtype=np.int64).tolist()
                    )
                    viscous_execution_backend = "torch"
                    viscous_execution_device = str(config.device)
                    viscous_torch_cuda_used = True
                else:
                    dq_cap = cap * np.minimum(vol[c0[face_ids]], vol[c1[face_ids]])
                    capped_mask_any = np.zeros((face_ids.size,), dtype=bool)
                    denom = np.maximum(w_sum, 1e-30)
                    for _ in range(substeps):
                        qn = q.copy()
                        q_face = q[face_ids]
                        lap_q = (
                            np.sum(w_mat * (q[nb_mat] - q_face[:, None]), axis=1)
                            / denom
                        )
                        dq = (nu * sub_dt) * lap_q
                        capped = np.abs(dq) > dq_cap
                        if np.any(capped):
                            dq = dq.copy()
                            dq[capped] = np.sign(dq[capped]) * dq_cap[capped]
                            capped_updates += int(np.count_nonzero(capped))
                            capped_mask_any |= capped
                        total_updates += int(face_ids.size)
                        qn[face_ids] = q_face + dq
                        if wall_flux_resistance_bundle["cell_ids"].size:
                            qn, wall_flux_diag = (
                                _apply_wall_flux_stokes_resistance_global_implicit(
                                    qn,
                                    resistance_bundle=wall_flux_resistance_bundle,
                                    nu=nu,
                                    dt=sub_dt,
                                    strength=float(
                                        config.wall_flux_stokes_resistance_strength
                                    ),
                                    prepared_solver=wall_flux_resistance_prepared_solver,
                                )
                            )
                            wall_flux_resistance_iterations = max(
                                wall_flux_resistance_iterations,
                                int(wall_flux_diag["iterations"]),
                            )
                            wall_flux_resistance_converged = (
                                wall_flux_resistance_converged
                                and bool(wall_flux_diag["converged"])
                            )
                            wall_flux_resistance_residual_l2 = max(
                                wall_flux_resistance_residual_l2,
                                float(wall_flux_diag["residual_l2"]),
                            )
                            wall_flux_resistance_method = str(
                                wall_flux_diag.get(
                                    "method", wall_flux_resistance_method
                                )
                            )
                        q = qn
                    if np.any(capped_mask_any):
                        capped_faces_unique.update(
                            int(fid) for fid in face_ids[capped_mask_any].tolist()
                        )
        else:
            (
                interior_face_ids,
                neighbor_ids,
                neighbor_w,
                neighbor_w_sum,
            ) = _cached_face_flux_laplacian_stencil(mesh)
            for _ in range(substeps):
                qn = q.copy()
                for fid in interior_face_ids.tolist():
                    nb = neighbor_ids.get(int(fid), None)
                    ww = neighbor_w.get(int(fid), None)
                    if nb is None or ww is None or nb.size == 0:
                        continue
                    total_updates += 1
                    lap_q = float(
                        np.sum(ww * (q[nb] - q[int(fid)]))
                        / max(float(neighbor_w_sum.get(int(fid), 0.0)), 1e-30)
                    )
                    dq = float(nu * sub_dt * lap_q)
                    i = int(c0[fid])
                    j = int(c1[fid])
                    dq_cap = float(cap * min(float(vol[i]), float(vol[j])))
                    if abs(dq) > dq_cap:
                        dq = float(np.sign(dq) * dq_cap)
                        capped_updates += 1
                        capped_faces_unique.add(int(fid))
                    qn[int(fid)] = q[int(fid)] + dq
                if wall_flux_resistance_bundle["cell_ids"].size:
                    qn, wall_flux_diag = (
                        _apply_wall_flux_stokes_resistance_global_implicit(
                            qn,
                            resistance_bundle=wall_flux_resistance_bundle,
                            nu=nu,
                            dt=sub_dt,
                            strength=float(config.wall_flux_stokes_resistance_strength),
                            prepared_solver=wall_flux_resistance_prepared_solver,
                        )
                    )
                    wall_flux_resistance_iterations = max(
                        wall_flux_resistance_iterations,
                        int(wall_flux_diag["iterations"]),
                    )
                    wall_flux_resistance_converged = (
                        wall_flux_resistance_converged
                        and bool(wall_flux_diag["converged"])
                    )
                    wall_flux_resistance_residual_l2 = max(
                        wall_flux_resistance_residual_l2,
                        float(wall_flux_diag["residual_l2"]),
                    )
                    wall_flux_resistance_method = str(
                        wall_flux_diag.get("method", wall_flux_resistance_method)
                    )
                q = qn
        flux_pred_raw = np.asarray(q, dtype=np.float64)
        if shear_face_flux_enabled and not _uses_wall_flux_stokes_resistance(config):
            flux_pred_raw, wall_shear_diag = _apply_wall_tangential_shear_to_face_flux(
                mesh,
                flux_pred_raw,
                wall_operator=wall_tangential_operator,
                nu=nu,
                dt=dt,
            )
            if bool(wall_shear_diag["enabled"]):
                wall_shear_face_flux_applications += 1
            wall_shear_face_flux_delta_l2 = max(
                wall_shear_face_flux_delta_l2,
                float(wall_shear_diag["delta_l2"]),
            )
            wall_shear_face_flux_active_cells = max(
                wall_shear_face_flux_active_cells,
                int(wall_shear_diag["active_cells"]),
            )
            wall_shear_face_flux_speed_before = max(
                wall_shear_face_flux_speed_before,
                float(wall_shear_diag["wall_speed_mean_before"]),
            )
            wall_shear_face_flux_speed_after = max(
                wall_shear_face_flux_speed_after,
                float(wall_shear_diag["wall_speed_mean_after"]),
            )
            v_pred = _reconstruct_cell_velocity_from_face_flux_numpy(
                mesh,
                flux_pred_raw,
                wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
                wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
            )
        elif _uses_legacy_isotropic_no_slip_wall(config.wall_velocity_boundary_mode):
            v_pred = _reconstruct_cell_velocity_from_face_flux_numpy(
                mesh,
                flux_pred_raw,
                wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
                wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
            )
            for _ in range(substeps):
                v_pred = _apply_wall_no_slip_velocity_sink(
                    v_pred,
                    wall_sink=wall_sink,
                    nu=nu,
                    dt=sub_dt,
                )
            flux_pred_raw = _face_flux_from_cell_velocity_numpy(mesh, v_pred)
        elif cuda_velocity_raw is not None:
            v_pred = np.asarray(cuda_velocity_raw, dtype=np.float64)
        else:
            v_pred = _reconstruct_cell_velocity_from_face_flux_numpy(
                mesh,
                flux_pred_raw,
                wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
                wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
            )
    elif mode == "no_viscous_debug_copy":
        v_pred = v_before.copy()
        flux_pred_raw = flux_before.copy()
    elif mode == "none":
        v_pred = v_before.copy()
        flux_pred_raw = flux_before.copy()
    else:
        raise ValueError(f"Unknown viscous_predictor_mode: {mode}")

    if momentum_pressure_state_enabled:
        # In the coherent state path q* is the mass interpolation of the complete
        # momentum predictor u*, including wall friction. Do not split out the
        # wall-only velocity delta or force it to have a zero cell-flux sum.
        flux_pred_raw = _face_flux_from_cell_velocity_numpy(mesh, v_pred)
        local_conservative_target_mode = "coherent_momentum_interpolation"

    if cuda_flux_contract is not None:
        flux_contract = np.asarray(cuda_flux_contract, dtype=np.float64)
    else:
        flux_contract = np.asarray(flux_pred_raw, dtype=np.float64).copy()
        _apply_face_flux_boundary_conditions_inplace(
            mesh,
            flux_contract,
            inlet_speed=float(config.inlet_speed),
            left_inlet_faces=left_faces,
            right_inlet_faces=right_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
            outlet_contract_mode=config.viscous_predictor_outlet_contract_mode,
        )
    local_conservative_flux_correction_enabled = bool(
        mode == "explicit_cell_velocity_laplacian_substepped_conservative"
        and tangential_wall_mode
        and not momentum_pressure_state_enabled
    )
    if local_conservative_flux_correction_enabled:
        target_cell_flux_sum = _compute_cell_flux_sum(mesh, flux_before)
        wall_tangent_abs_for_target = (
            np.max(np.abs(wall_tangential_operator), axis=(1, 2))
            if wall_tangential_operator.size
            else np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
        )
        wall_target_cells = wall_tangent_abs_for_target > 0.0
        eligible_cell_mask: np.ndarray | None = None
        if cell_velocity_wall_momentum_enabled:
            # Tangential wall momentum is a viscous momentum update, not a mass
            # source. Preserve the pre-predictor cell flux sum for every cell so
            # the predictor does not inject a new divergence RHS near the wall.
            local_conservative_target_mode = "preserve_all_cells"
        elif np.any(wall_target_cells):
            current_contract_sum = _compute_cell_flux_sum(mesh, flux_contract)
            target_cell_flux_sum = np.asarray(target_cell_flux_sum, dtype=np.float64)
            target_cell_flux_sum[wall_target_cells] = current_contract_sum[
                wall_target_cells
            ]
            eligible_cell_mask = ~np.asarray(wall_target_cells, dtype=bool)
        flux_contract, local_conservation_diag = (
            _locally_conserve_interior_face_flux_cell_sums(
                mesh,
                flux_contract,
                target_cell_flux_sum=target_cell_flux_sum,
                eligible_cell_mask=eligible_cell_mask,
                iterations=8,
            )
        )
        _apply_face_flux_boundary_conditions_inplace(
            mesh,
            flux_contract,
            inlet_speed=float(config.inlet_speed),
            left_inlet_faces=left_faces,
            right_inlet_faces=right_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
            outlet_contract_mode=config.viscous_predictor_outlet_contract_mode,
        )
    if cuda_velocity_contract is not None and cuda_velocity_raw is not None:
        v_after = np.asarray(cuda_velocity_contract, dtype=np.float64)
        v_after_pred = np.asarray(cuda_velocity_raw, dtype=np.float64)
    else:
        v_after = _reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            flux_contract,
            wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
            wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
        )
        v_after_pred = _reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            flux_pred_raw,
            wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
            wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
        )
    preserve_momentum_predictor = momentum_pressure_state_enabled
    cell_velocity_state = v_pred if preserve_momentum_predictor else v_after
    velocity_state_vs_flux_reconstruction = np.asarray(
        cell_velocity_state,
        dtype=np.float64,
    ) - np.asarray(v_after, dtype=np.float64)

    dv = v_pred - v_before
    dv_mag = np.linalg.norm(dv, axis=1)
    v_after_pred_mag2 = np.sum(v_after_pred * v_after_pred, axis=1)
    v_after_mag2 = np.sum(v_after * v_after, axis=1)
    kin_after_predictor = (
        float(np.mean(0.5 * v_after_pred_mag2)) if v_after_pred_mag2.size else 0.0
    )
    kin_after_contract = (
        float(np.mean(0.5 * v_after_mag2)) if v_after_mag2.size else 0.0
    )
    cell_state_after_predictor_mag2 = np.sum(v_pred * v_pred, axis=1)
    kin_cell_state_after_predictor = (
        float(np.mean(0.5 * cell_state_after_predictor_mag2))
        if cell_state_after_predictor_mag2.size
        else 0.0
    )
    kin_volume_integral_before_predictor = (
        float(np.sum(vol * ke_before)) if ke_before.size else 0.0
    )
    kin_volume_integral_after_predictor = (
        float(np.sum(vol * 0.5 * cell_state_after_predictor_mag2))
        if cell_state_after_predictor_mag2.size
        else 0.0
    )
    kin_volume_integral_after_contract_reconstruction = (
        float(np.sum(vol * 0.5 * v_after_mag2)) if v_after_mag2.size else 0.0
    )
    stability_metric = float(stability_metric_raw)
    viscous_nonorthogonal_bound_per_substep = float(
        viscous_nonorthogonal_stability_bound_raw / max(substeps, 1)
    )
    stability_warning = (
        bool(
            not np.isfinite(viscous_nonorthogonal_bound_per_substep)
            or viscous_nonorthogonal_bound_per_substep
            > viscous_nonorthogonal_stability_target * (1.0 + 1e-12)
        )
        if viscous_nonorthogonal_enabled
        else bool(stability_metric_raw > 0.5)
    )

    viscous_nonorthogonal_flux_stats: dict[str, Any] = {
        "all_faces": _vector_stats(np.zeros((0,), dtype=np.float64)),
        "interior_faces": _vector_stats(np.zeros((0,), dtype=np.float64)),
        "wall_faces": _vector_stats(np.zeros((0,), dtype=np.float64)),
        "inlet_faces": _vector_stats(np.zeros((0,), dtype=np.float64)),
        "outlet_faces": _vector_stats(np.zeros((0,), dtype=np.float64)),
    }
    viscous_lsq_rank_min = 0
    viscous_lsq_full_rank_fraction = 0.0
    viscous_lsq_condition_max = 0.0
    viscous_lsq_condition_p95 = 0.0
    viscous_lsq_equation_count_min = 0
    viscous_geometry_build_seconds = 0.0
    if viscous_nonorthogonal_enabled and viscous_nonorthogonal_geometry is not None:
        region_ids = {
            "interior_faces": np.asarray(
                viscous_nonorthogonal_geometry["interior_faces"], dtype=np.int64
            ),
            "wall_faces": np.asarray(
                viscous_nonorthogonal_geometry["wall_faces"], dtype=np.int64
            ),
            "inlet_faces": np.asarray(
                viscous_nonorthogonal_geometry["inlet_faces"], dtype=np.int64
            ),
            "outlet_faces": np.asarray(
                viscous_nonorthogonal_geometry["outlet_faces"], dtype=np.int64
            ),
        }
        viscous_nonorthogonal_flux_stats["all_faces"] = _vector_stats(
            viscous_nonorthogonal_flux
        )
        for region_name, region_face_ids in region_ids.items():
            viscous_nonorthogonal_flux_stats[region_name] = _vector_stats(
                viscous_nonorthogonal_flux[region_face_ids]
            )
        lsq_rank = np.asarray(
            viscous_nonorthogonal_geometry["lsq_normal_matrix_rank"],
            dtype=np.int32,
        )
        lsq_condition = np.asarray(
            viscous_nonorthogonal_geometry["lsq_normal_matrix_condition"],
            dtype=np.float64,
        )
        lsq_equation_count = np.asarray(
            viscous_nonorthogonal_geometry["lsq_equation_count"], dtype=np.int32
        )
        viscous_lsq_rank_min = int(np.min(lsq_rank)) if lsq_rank.size else 0
        viscous_lsq_full_rank_fraction = (
            float(np.mean(lsq_rank == 3)) if lsq_rank.size else 0.0
        )
        finite_condition = lsq_condition[np.isfinite(lsq_condition)]
        viscous_lsq_condition_max = (
            float(np.max(finite_condition)) if finite_condition.size else float("inf")
        )
        viscous_lsq_condition_p95 = (
            float(np.percentile(finite_condition, 95.0))
            if finite_condition.size
            else float("inf")
        )
        viscous_lsq_equation_count_min = (
            int(np.min(lsq_equation_count)) if lsq_equation_count.size else 0
        )
        viscous_geometry_build_seconds = float(
            viscous_nonorthogonal_geometry["build_seconds"]
        )

    div_before_diag = compute_tetra_flux_divergence(
        mesh,
        flux_before,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    div_after_pred_diag = compute_tetra_flux_divergence(
        mesh,
        flux_pred_raw,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    div_after_contract_diag = compute_tetra_flux_divergence(
        mesh,
        flux_contract,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    inlet_b = float(div_before_diag.get("inlet_flux_total", 0.0))
    inlet_p = float(div_after_pred_diag.get("inlet_flux_total", 0.0))
    inlet_c = float(div_after_contract_diag.get("inlet_flux_total", 0.0))
    outlet_b = float(div_before_diag.get("outlet_flux_total", 0.0))
    outlet_p = float(div_after_pred_diag.get("outlet_flux_total", 0.0))
    outlet_c = float(div_after_contract_diag.get("outlet_flux_total", 0.0))

    dflux_pred = np.asarray(flux_pred_raw, dtype=np.float64) - np.asarray(
        flux_before, dtype=np.float64
    )
    dflux_contract = np.asarray(flux_contract, dtype=np.float64) - np.asarray(
        flux_pred_raw, dtype=np.float64
    )
    wall_tangent_abs = (
        np.max(np.abs(wall_tangential_operator), axis=(1, 2))
        if wall_tangential_operator.size
        else np.zeros((0,), dtype=np.float64)
    )
    wall_tangent_active = wall_tangent_abs > 0.0
    wall_tangent_trace = (
        np.trace(wall_tangential_operator, axis1=1, axis2=2)
        if wall_tangential_operator.size
        else np.zeros((0,), dtype=np.float64)
    )
    wall_tangent_trace_active = wall_tangent_trace[wall_tangent_active]
    wall_tangent_max_abs = (
        float(np.max(wall_tangent_abs[wall_tangent_active]))
        if np.any(wall_tangent_active)
        else 0.0
    )
    wall_tangent_trace_max = (
        float(np.max(wall_tangent_trace_active))
        if wall_tangent_trace_active.size
        else 0.0
    )
    wall_tangent_trace_mean = (
        float(np.mean(wall_tangent_trace_active))
        if wall_tangent_trace_active.size
        else 0.0
    )

    diag = dict(state.diagnostics)
    diag["numerical_profile_resolution"] = _numerical_profile_resolution_diagnostics(
        requested_config, config
    )
    diag["viscous_predictor"] = {
        "viscous_predictor_used": bool(predictor_used),
        "viscous_predictor_mode": str(mode),
        "viscous_execution_backend": str(viscous_execution_backend),
        "viscous_execution_device": str(viscous_execution_device),
        "viscous_torch_cuda_used": bool(viscous_torch_cuda_used),
        "viscous_cuda_no_slip_used": bool(viscous_cuda_no_slip_used),
        "viscous_cuda_input_reused": bool(viscous_cuda_input_reused),
        "viscous_cuda_finalization_used": bool(viscous_cuda_finalization_used),
        "viscous_cuda_host_to_device_bytes_avoided": int(
            viscous_cuda_host_to_device_bytes_avoided
        ),
        "viscous_cuda_cpu_reconstruction_solves_avoided": int(
            3 if viscous_cuda_finalization_used else 0
        ),
        "viscous_cuda_residency_scope": (
            "convection_handoff_through_conservative_no_slip_velocity_update"
            if viscous_cuda_no_slip_used and viscous_cuda_input_reused
            else (
                "conservative_no_slip_velocity_update"
                if viscous_cuda_no_slip_used
                else (
                    "convection_handoff_through_slip_boundary_and_velocity_reconstruction"
                    if viscous_cuda_finalization_used and viscous_cuda_input_reused
                    else (
                        "slip_boundary_and_velocity_reconstruction"
                        if viscous_cuda_finalization_used
                        else "none"
                    )
                )
            )
        ),
        "viscous_numpy_fallback_reason": str(viscous_numpy_fallback_reason),
        "viscous_predictor_outlet_contract_mode": str(
            config.viscous_predictor_outlet_contract_mode
        ),
        "projection_cell_velocity_update_mode": str(
            config.projection_cell_velocity_update_mode
        ),
        "cell_velocity_state_source": (
            "momentum_predictor"
            if preserve_momentum_predictor
            else "face_flux_reconstruction_after_contract"
        ),
        "momentum_predictor_preserved_for_projection": bool(
            preserve_momentum_predictor
        ),
        "face_flux_predictor_source": (
            "interpolation_of_complete_momentum_predictor"
            if momentum_pressure_state_enabled
            else "legacy_predictor_flux_path"
        ),
        "coherent_momentum_face_flux_interpolation_enabled": bool(
            momentum_pressure_state_enabled
        ),
        "cell_velocity_state_vs_flux_reconstruction": _vector_stats(
            velocity_state_vs_flux_reconstruction
        ),
        "viscous_face_flux_laplacian_vectorized": bool(
            config.viscous_face_flux_laplacian_vectorized
        ),
        "wall_velocity_boundary_mode": str(config.wall_velocity_boundary_mode),
        "wall_velocity_boundary_implementation": (
            "full_vector_dirichlet_identity"
            if viscous_nonorthogonal_enabled
            else (
                "tangential_zero_velocity"
                if _uses_tangential_no_slip_wall(config.wall_velocity_boundary_mode)
                else (
                    "legacy_isotropic_velocity_sink"
                    if _uses_legacy_isotropic_no_slip_wall(
                        config.wall_velocity_boundary_mode
                    )
                    else "slip_zero_normal_flux"
                )
            )
        ),
        "viscous_nonorthogonal_correction_mode": str(
            config.viscous_nonorthogonal_correction_mode
        ),
        "viscous_nonorthogonal_correction_enabled": bool(viscous_nonorthogonal_enabled),
        "viscous_nonorthogonal_scheme": (
            "cell_lsq_deferred_K_dot_grad"
            if viscous_nonorthogonal_enabled
            else "legacy_tpfa"
        ),
        "viscous_nonorthogonal_wall_projector": (
            "identity" if viscous_nonorthogonal_enabled else "legacy"
        ),
        "viscous_nonorthogonal_wall_boundary_condition": (
            "dirichlet_zero_velocity" if viscous_nonorthogonal_enabled else "legacy"
        ),
        "viscous_nonorthogonal_inlet_boundary_condition": (
            "dirichlet_prescribed_normal_velocity"
            if viscous_nonorthogonal_enabled
            else "legacy_no_diffusive_face_flux"
        ),
        "viscous_nonorthogonal_outlet_boundary_condition": (
            "homogeneous_neumann"
            if viscous_nonorthogonal_enabled
            else "legacy_no_diffusive_face_flux"
        ),
        "viscous_nonorthogonal_wall_strength_calibration_used": False,
        "viscous_nonorthogonal_flux": viscous_nonorthogonal_flux_stats,
        "viscous_nonorthogonal_update_max": float(viscous_nonorthogonal_update_max),
        "viscous_nonorthogonal_update_l2": float(viscous_nonorthogonal_update_l2),
        "viscous_nonorthogonal_operator_power": float(viscous_operator_energy_rate),
        "viscous_nonorthogonal_lsq_rank_min": int(viscous_lsq_rank_min),
        "viscous_nonorthogonal_lsq_full_rank_fraction": float(
            viscous_lsq_full_rank_fraction
        ),
        "viscous_nonorthogonal_lsq_condition_p95": float(viscous_lsq_condition_p95),
        "viscous_nonorthogonal_lsq_condition_max": float(viscous_lsq_condition_max),
        "viscous_nonorthogonal_lsq_equation_count_min": int(
            viscous_lsq_equation_count_min
        ),
        "viscous_nonorthogonal_geometry_build_seconds": float(
            viscous_geometry_build_seconds
        ),
        "wall_tangential_no_slip_strength": float(
            config.wall_tangential_no_slip_strength
        ),
        "wall_tangential_shear_face_flux_requested": bool(shear_face_flux_enabled),
        "wall_tangential_cell_velocity_momentum_enabled": bool(
            cell_velocity_wall_momentum_enabled
        ),
        "wall_tangential_operator_active_cells": int(
            np.count_nonzero(wall_tangent_active)
        ),
        "wall_tangential_operator_max_abs": float(wall_tangent_max_abs),
        "wall_tangential_operator_trace_mean": float(wall_tangent_trace_mean),
        "wall_tangential_operator_trace_max": float(wall_tangent_trace_max),
        "wall_tangential_operator_effective_nu_dt_max_abs": float(
            float(nu) * float(dt) * wall_tangent_max_abs
        ),
        "wall_tangential_operator_effective_nu_subdt_max_abs": float(
            float(nu) * float(sub_dt) * wall_tangent_max_abs
        ),
        "wall_flux_stokes_resistance_enabled": bool(
            _uses_wall_flux_stokes_resistance(config)
        ),
        "wall_flux_stokes_resistance_strength": float(
            config.wall_flux_stokes_resistance_strength
        ),
        "wall_flux_stokes_resistance_active_faces": int(
            np.asarray(
                wall_flux_resistance_bundle["active_face_ids"], dtype=np.int64
            ).size
        ),
        "wall_flux_stokes_resistance_solver_iterations": int(
            wall_flux_resistance_iterations
        ),
        "wall_flux_stokes_resistance_solver_converged": bool(
            wall_flux_resistance_converged
        ),
        "wall_flux_stokes_resistance_solver_residual_l2": float(
            wall_flux_resistance_residual_l2
        ),
        "wall_flux_stokes_resistance_solver_method": str(wall_flux_resistance_method),
        "wall_tangential_shear_face_flux_enabled": bool(
            wall_shear_face_flux_applications > 0
        ),
        "wall_tangential_shear_face_flux_applications": int(
            wall_shear_face_flux_applications
        ),
        "wall_tangential_shear_face_flux_active_cells": int(
            wall_shear_face_flux_active_cells
        ),
        "wall_tangential_shear_face_flux_delta_l2": float(
            wall_shear_face_flux_delta_l2
        ),
        "wall_tangential_shear_face_flux_wall_speed_mean_before": float(
            wall_shear_face_flux_speed_before
        ),
        "wall_tangential_shear_face_flux_wall_speed_mean_after": float(
            wall_shear_face_flux_speed_after
        ),
        "local_conservative_flux_correction_enabled": bool(
            local_conservative_flux_correction_enabled
        ),
        "local_conservative_flux_correction_target_mode": str(
            local_conservative_target_mode
        ),
        "local_conservative_flux_correction_iterations": int(
            local_conservation_diag["iterations"]
        ),
        "local_conservative_flux_correction_residual_l2_before": float(
            local_conservation_diag["residual_l2_before"]
        ),
        "local_conservative_flux_correction_residual_l2_after": float(
            local_conservation_diag["residual_l2_after"]
        ),
        "local_conservative_flux_correction_residual_max_abs_before": float(
            local_conservation_diag["residual_max_abs_before"]
        ),
        "local_conservative_flux_correction_residual_max_abs_after": float(
            local_conservation_diag["residual_max_abs_after"]
        ),
        "local_conservative_flux_correction_delta_l2": float(
            local_conservation_diag["delta_l2"]
        ),
        "local_conservative_flux_correction_delta_max_abs": float(
            local_conservation_diag["delta_max_abs"]
        ),
        "wall_tangential_cell_velocity_flux_delta_conservative_applied": bool(
            wall_momentum_flux_delta_diag["enabled"]
        ),
        "wall_tangential_cell_velocity_flux_delta_conservative_iterations": int(
            wall_momentum_flux_delta_diag["iterations"]
        ),
        "wall_tangential_cell_velocity_flux_delta_residual_l2_before": float(
            wall_momentum_flux_delta_diag["residual_l2_before"]
        ),
        "wall_tangential_cell_velocity_flux_delta_residual_l2_after": float(
            wall_momentum_flux_delta_diag["residual_l2_after"]
        ),
        "wall_tangential_cell_velocity_flux_delta_residual_max_abs_before": float(
            wall_momentum_flux_delta_diag["residual_max_abs_before"]
        ),
        "wall_tangential_cell_velocity_flux_delta_residual_max_abs_after": float(
            wall_momentum_flux_delta_diag["residual_max_abs_after"]
        ),
        "wall_tangential_cell_velocity_flux_delta_conservative_delta_l2": float(
            wall_momentum_flux_delta_diag["delta_l2"]
        ),
        "wall_tangential_cell_velocity_flux_delta_conservative_delta_max_abs": float(
            wall_momentum_flux_delta_diag["delta_max_abs"]
        ),
        "kinematic_viscosity": float(config.kinematic_viscosity),
        "viscous_dt": float(dt),
        "viscous_substeps": int(substeps),
        "viscous_substep_dt": float(sub_dt),
        "viscous_face_flux_divergence_impact_cap": float(
            config.viscous_face_flux_divergence_impact_cap
        ),
        "viscous_stability_metric": float(stability_metric),
        "viscous_stability_warning": bool(stability_warning),
        "viscous_base_stability_target": float(
            viscous_base_stability_target
            if viscous_nonorthogonal_enabled
            else target_metric
        ),
        "viscous_base_stability_bound_substeps": int(viscous_base_bound_substeps),
        "viscous_nonorthogonal_stability_bound": float(
            viscous_nonorthogonal_stability_bound_raw
        ),
        "viscous_nonorthogonal_stability_bound_per_substep": float(
            viscous_nonorthogonal_bound_per_substep
        ),
        "viscous_nonorthogonal_stability_target": float(
            viscous_nonorthogonal_stability_target
        ),
        "viscous_nonorthogonal_stability_bound_substeps": int(
            viscous_nonorthogonal_bound_substeps
        ),
        "capped_predictor_updates_count": int(capped_updates),
        "total_predictor_updates_count": int(total_updates),
        "capped_predictor_updates_fraction": float(
            capped_updates / max(total_updates, 1)
        ),
        "capped_predictor_faces_count": int(len(capped_faces_unique)),
        "viscous_delta_velocity_max": float(np.max(dv_mag)) if dv_mag.size else 0.0,
        "viscous_delta_velocity_l2": float(np.sqrt(np.mean(dv_mag * dv_mag)))
        if dv_mag.size
        else 0.0,
        "kinetic_energy_before_predictor": float(kin_before),
        "kinetic_energy_after_predictor": float(kin_after_predictor),
        "kinetic_energy_after_contract": float(kin_after_contract),
        "kinetic_energy_cell_velocity_state_before_predictor": float(kin_before),
        "kinetic_energy_cell_velocity_state_after_predictor": float(
            kin_cell_state_after_predictor
        ),
        "kinetic_energy_volume_integral_before_predictor_m5_s2": float(
            kin_volume_integral_before_predictor
        ),
        "kinetic_energy_volume_integral_after_predictor_m5_s2": float(
            kin_volume_integral_after_predictor
        ),
        "kinetic_energy_volume_integral_after_contract_reconstruction_m5_s2": float(
            kin_volume_integral_after_contract_reconstruction
        ),
        "kinetic_energy_cell_velocity_state_output": float(
            np.mean(
                0.5
                * np.sum(
                    np.asarray(cell_velocity_state, dtype=np.float64) ** 2,
                    axis=1,
                )
            )
        )
        if np.asarray(cell_velocity_state).size
        else 0.0,
        "face_flux_delta_predictor_max": float(np.max(np.abs(dflux_pred)))
        if dflux_pred.size
        else 0.0,
        "face_flux_delta_predictor_l2": float(np.sqrt(np.mean(dflux_pred * dflux_pred)))
        if dflux_pred.size
        else 0.0,
        "face_flux_delta_contract_max": float(np.max(np.abs(dflux_contract)))
        if dflux_contract.size
        else 0.0,
        "face_flux_delta_contract_l2": float(
            np.sqrt(np.mean(dflux_contract * dflux_contract))
        )
        if dflux_contract.size
        else 0.0,
        "divergence_before_predictor_max": float(
            div_before_diag.get("divergence_max_abs", 0.0)
        ),
        "divergence_before_predictor_l2": float(
            div_before_diag.get("divergence_l2", 0.0)
        ),
        "divergence_after_predictor_before_boundary_contract_max": float(
            div_after_pred_diag.get("divergence_max_abs", 0.0)
        ),
        "divergence_after_predictor_before_boundary_contract_l2": float(
            div_after_pred_diag.get("divergence_l2", 0.0)
        ),
        "divergence_after_boundary_contract_before_projection_max": float(
            div_after_contract_diag.get("divergence_max_abs", 0.0)
        ),
        "divergence_after_boundary_contract_before_projection_l2": float(
            div_after_contract_diag.get("divergence_l2", 0.0)
        ),
        "net_boundary_flux_before_predictor": float(
            div_before_diag.get("net_boundary_flux", 0.0)
        ),
        "net_boundary_flux_after_predictor_before_contract": float(
            div_after_pred_diag.get("net_boundary_flux", 0.0)
        ),
        "net_boundary_flux_after_contract": float(
            div_after_contract_diag.get("net_boundary_flux", 0.0)
        ),
        "wall_flux_max_after_predictor_before_contract": float(
            div_after_pred_diag.get("wall_flux_max_abs", 0.0)
        ),
        "wall_flux_max_after_contract": float(
            div_after_contract_diag.get("wall_flux_max_abs", 0.0)
        ),
        "outlet_inlet_ratio_before_predictor": float(
            outlet_b / max(abs(inlet_b), 1e-30)
        ),
        "outlet_inlet_ratio_after_predictor_before_contract": float(
            outlet_p / max(abs(inlet_p), 1e-30)
        ),
        "outlet_inlet_ratio_after_contract": float(outlet_c / max(abs(inlet_c), 1e-30)),
        "arrays": {
            "divergence_before_predictor": np.asarray(
                div_before_diag["divergence"], dtype=np.float64
            ),
            "divergence_after_predictor_before_boundary_contract": np.asarray(
                div_after_pred_diag["divergence"], dtype=np.float64
            ),
            "divergence_after_boundary_contract_before_projection": np.asarray(
                div_after_contract_diag["divergence"], dtype=np.float64
            ),
            "velocity_before_predictor": np.asarray(v_before, dtype=np.float64),
            "velocity_after_predictor": np.asarray(v_after_pred, dtype=np.float64),
            "velocity_after_contract": np.asarray(v_after, dtype=np.float64),
            "momentum_velocity_after_predictor": np.asarray(
                v_pred,
                dtype=np.float64,
            ),
            "cell_velocity_state_output": np.asarray(
                cell_velocity_state,
                dtype=np.float64,
            ),
            "face_flux_before_predictor": np.asarray(flux_before, dtype=np.float64),
            "face_flux_after_predictor_before_contract": np.asarray(
                flux_pred_raw, dtype=np.float64
            ),
            "face_flux_after_contract": np.asarray(flux_contract, dtype=np.float64),
            "viscous_nonorthogonal_face_flux": np.asarray(
                viscous_nonorthogonal_flux, dtype=np.float64
            ),
            "viscous_nonorthogonal_laplacian": np.asarray(
                viscous_nonorthogonal_laplacian, dtype=np.float64
            ),
        },
    }
    return TetraFlowState(
        cell_velocity=cell_velocity_state,
        face_flux=flux_contract,
        pressure=np.asarray(state.pressure, dtype=np.float64).copy(),
        diagnostics=diag,
    )


def apply_tetra_flow_boundary_conditions(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
    config: TetraFlowConfig,
) -> TetraFlowState:
    requested_config = config
    config = resolve_tetra_flow_numerical_profile(config)
    left_faces, right_faces = _inlet_face_sets(mesh)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    flux = np.asarray(state.face_flux, dtype=np.float64).copy()
    _apply_face_flux_boundary_conditions_inplace(
        mesh,
        flux,
        inlet_speed=float(config.inlet_speed),
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        outlet_contract_mode=config.viscous_predictor_outlet_contract_mode,
    )
    velocity = _reconstruct_cell_velocity_from_face_flux_numpy(
        mesh,
        flux,
        wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
        wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
    )
    diagnostics = dict(state.diagnostics)
    diagnostics["numerical_profile_resolution"] = (
        _numerical_profile_resolution_diagnostics(requested_config, config)
    )
    return TetraFlowState(
        cell_velocity=velocity,
        face_flux=flux,
        pressure=np.asarray(state.pressure, dtype=np.float64).copy(),
        diagnostics=diagnostics,
    )


def compute_tetra_velocity_divergence(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
) -> np.ndarray:
    return _compute_divergence_numpy(
        mesh, np.asarray(state.face_flux, dtype=np.float64)
    )


def _run_sign_trial(
    mesh: ImportedTetraMesh,
    *,
    flux_star: np.ndarray,
    grad_flux: np.ndarray,
    sign: ProjectionSign,
    left_faces: np.ndarray,
    right_faces: np.ndarray,
    outlet_faces: np.ndarray,
    wall_faces: np.ndarray,
    damping: float,
) -> dict[str, Any]:
    flux_corr, corr = _apply_projection_correction(
        flux_star,
        grad_flux,
        projection_sign=sign,
        damping=float(damping),
    )
    if left_faces.size:
        flux_corr[left_faces] = flux_star[left_faces]
    if right_faces.size:
        flux_corr[right_faces] = flux_star[right_faces]
    if wall_faces.size:
        flux_corr[wall_faces] = 0.0

    div_star = _compute_divergence_numpy(mesh, flux_star)
    div_corr = _compute_divergence_numpy(mesh, flux_corr)
    bnd = _boundary_flux_audit(
        mesh,
        flux_corr,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    return {
        "projection_sign": sign,
        "initial_divergence_max_abs": float(np.max(np.abs(div_star))),
        "final_divergence_max_abs": float(np.max(np.abs(div_corr))),
        "divergence_reduction_ratio_linf": _safe_ratio(
            float(np.max(np.abs(div_corr))),
            float(np.max(np.abs(div_star))),
        ),
        "initial_divergence_l2": float(np.sqrt(np.mean(div_star**2))),
        "final_divergence_l2": float(np.sqrt(np.mean(div_corr**2))),
        "divergence_reduction_ratio_l2": _safe_ratio(
            float(np.sqrt(np.mean(div_corr**2))),
            float(np.sqrt(np.mean(div_star**2))),
        ),
        "net_boundary_flux_after": float(bnd["boundary_total"]["sum_flux"]),
        "inlet_flux_after": float(
            bnd["left_inlet"]["inflow_total"] + bnd["right_inlet"]["inflow_total"]
        ),
        "outlet_flux_after": float(bnd["outlet"]["outflow_total"]),
        "correction_flux_stats": _vector_stats(corr),
    }


def _top_divergence_cells(
    *,
    mesh: ImportedTetraMesh,
    div_star: np.ndarray,
    div_corr: np.ndarray,
    face_flux_corrected: np.ndarray,
    masks: dict[str, np.ndarray],
    top_k: int = 50,
) -> dict[str, Any]:
    abs_corr = np.abs(np.asarray(div_corr, dtype=np.float64))
    order = np.argsort(-abs_corr)
    take = min(int(top_k), int(order.size))
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    boundary_ids = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    boundary_set = set(boundary_ids.tolist())
    flux_all = np.asarray(face_flux_corrected, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    boundary_adj = np.asarray(
        masks.get("boundary_adjacent", np.zeros_like(abs_corr, dtype=bool))
    )
    outlet_adj = np.asarray(
        masks.get("outlet_adjacent", np.zeros_like(abs_corr, dtype=bool))
    )
    inlet_adj = np.asarray(
        masks.get("inlet_adjacent", np.zeros_like(abs_corr, dtype=bool))
    )
    wall_adj = np.asarray(
        masks.get("wall_adjacent", np.zeros_like(abs_corr, dtype=bool))
    )
    junction_adj = np.asarray(
        masks.get("junction_zone", np.zeros_like(abs_corr, dtype=bool))
    )
    interior_adj = np.asarray(
        masks.get("interior_core", np.zeros_like(abs_corr, dtype=bool))
    )
    for idx in order[:take].tolist():
        region_labels = [name for name, mask in masks.items() if bool(mask[idx])]
        face_ids = np.asarray(mesh.cell_to_faces[idx], dtype=np.int64)
        boundary_groups: list[str] = []
        for fid in face_ids.tolist():
            if int(fid) in boundary_set:
                tag = int(mesh.boundary_tag_per_face[fid])
                name = mesh.boundary_face_names.get(tag, f"tag_{tag}")
                if name not in boundary_groups:
                    boundary_groups.append(name)
        local_flux = (
            flux_all[face_ids] if face_ids.size else np.zeros((0,), dtype=np.float64)
        )
        local_pos = local_flux[local_flux > 0.0]
        local_neg = local_flux[local_flux < 0.0]
        if bool(outlet_adj[idx]):
            region = "outlet_adjacent"
        elif bool(inlet_adj[idx]):
            region = "inlet_adjacent"
        elif bool(wall_adj[idx]):
            region = "wall_adjacent"
        elif bool(junction_adj[idx]):
            region = "junction_zone"
        elif bool(interior_adj[idx]):
            region = "interior_core"
        else:
            region = "unclassified"
        rows.append(
            {
                "cell_index": int(idx),
                "center_xyz": np.asarray(
                    mesh.cell_centers[idx], dtype=np.float64
                ).tolist(),
                "volume": float(mesh.cell_volumes[idx]),
                "div_star": float(div_star[idx]),
                "div_corrected": float(div_corr[idx]),
                "abs_div_corrected": float(abs(div_corr[idx])),
                "region": region,
                "region_labels": region_labels,
                "adjacent_boundary_groups": boundary_groups,
                "adjacent_face_count": int(face_ids.size),
                "neighbor_count": int(np.sum(c1[face_ids] >= 0)),
                "max_face_flux_magnitude": float(np.max(np.abs(local_flux)))
                if local_flux.size
                else 0.0,
                "sum_positive_flux": float(np.sum(local_pos))
                if local_pos.size
                else 0.0,
                "sum_negative_flux": float(np.sum(local_neg))
                if local_neg.size
                else 0.0,
                "local_net_flux": float(np.sum(local_flux)) if local_flux.size else 0.0,
                "near_inlet": bool(inlet_adj[idx]),
                "near_outlet": bool(outlet_adj[idx]),
                "near_wall": bool(wall_adj[idx]),
                "near_corner": bool(inlet_adj[idx] and wall_adj[idx]),
                "boundary_adjacent": bool(boundary_adj[idx]),
            }
        )
    if rows:
        top_ids = np.asarray([int(r["cell_index"]) for r in rows], dtype=np.int64)
    else:
        top_ids = np.zeros((0,), dtype=np.int64)
    median_vol = float(np.median(np.asarray(mesh.cell_volumes, dtype=np.float64)))
    max_hotspot_vol = (
        float(np.max(np.asarray(mesh.cell_volumes[top_ids], dtype=np.float64)))
        if top_ids.size
        else 0.0
    )
    boundary_hotspots = (
        int(np.count_nonzero(boundary_adj[top_ids])) if top_ids.size else 0
    )
    outlet_hotspots = int(np.count_nonzero(outlet_adj[top_ids])) if top_ids.size else 0
    interior_hotspots = (
        int(np.count_nonzero(interior_adj[top_ids])) if top_ids.size else 0
    )
    if outlet_hotspots >= max(1, top_ids.size // 2):
        suspected = "mostly outlet-adjacent pressure projection residue"
    elif boundary_hotspots >= max(1, top_ids.size // 2):
        suspected = "boundary-condition-localized residue"
    else:
        suspected = "interior operator/discretization residue"
    return {
        "cells": rows,
        "hotspot_summary": {
            "top_k": int(top_ids.size),
            "boundary_adjacent_count": int(boundary_hotspots),
            "outlet_adjacent_count": int(outlet_hotspots),
            "interior_count": int(interior_hotspots),
            "max_hotspot_volume_ratio_vs_median": _safe_ratio(
                max_hotspot_vol, median_vol
            ),
            "suspected_cause": suspected,
        },
    }


def solve_tetra_pressure_projection(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
    config: TetraFlowConfig,
) -> TetraFlowState:
    requested_config = config
    config = resolve_tetra_flow_numerical_profile(config)
    _activate_torch_cache_context(mesh, device=str(config.device))
    _validate_config(mesh, config)
    backend = _resolve_backend(config)
    left_faces, right_faces = _inlet_face_sets(mesh)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    flux_star = np.asarray(state.face_flux, dtype=np.float64).copy()
    _apply_face_flux_boundary_conditions_inplace(
        mesh,
        flux_star,
        inlet_speed=float(config.inlet_speed),
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        outlet_contract_mode=config.pressure_projection_outlet_contract_mode,
    )
    div_star = _compute_divergence_numpy(mesh, flux_star)
    star_sum = _compute_cell_flux_sum(mesh, flux_star)
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=float(config.projection_dt),
        density=float(config.density),
        outlet_faces=outlet_faces,
    )
    rhs, outlet_rhs = _assemble_poisson_rhs(
        star_sum,
        coeff,
        pressure_outlet_value=float(config.pressure_outlet_value),
        projection_sign=config.projection_sign,
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
        rhs_mode=config.projection_rhs_mode,
    )
    rhs_base = rhs.copy()
    rhs_nonorthogonal_term = np.zeros_like(rhs_base)
    rhs_expected = rhs.copy()
    p0 = np.asarray(state.pressure, dtype=np.float64)

    used_numpy_fallback = bool(backend.used_numpy_fallback)
    frozen_nonorthogonal_gradient_flux = np.zeros_like(flux_star)
    previous_frozen_gradient_flux = np.zeros_like(flux_star)
    previous_frozen_gradient_flux_source = "zero_no_compatible_previous_state"
    current_gradient_flux_scale = float(config.projection_dt / config.density)
    if config.pressure_nonorthogonal_correction_mode == "deferred_lsq":
        previous_primary = state.diagnostics.get("face_flux_primary", {})
        if isinstance(previous_primary, dict):
            previous_candidate = np.asarray(
                previous_primary.get(
                    "pressure_gradient_flux_nonorthogonal_frozen",
                    np.zeros((0,), dtype=np.float64),
                ),
                dtype=np.float64,
            )
            if previous_candidate.shape == flux_star.shape and np.all(
                np.isfinite(previous_candidate)
            ):
                # Carry the accepted face-volume flux itself.  Both A and the
                # raw non-orthogonal pressure flux scale with dt/rho, while the
                # solved pressure correction scales inversely, so rescaling the
                # frozen face flux when dt changes would double-count that scale.
                previous_frozen_gradient_flux = previous_candidate.copy()
                previous_frozen_gradient_flux_source = (
                    "previous_state_frozen_face_flux_unscaled"
                )
    nonorthogonal_gradient = np.zeros(
        (mesh.tetrahedra.shape[0], 3),
        dtype=np.float64,
    )
    nonorthogonal_correction_diag: dict[str, Any] = {
        "mode": str(config.pressure_nonorthogonal_correction_mode),
        "enabled": False,
        "configured_sweeps": int(config.pressure_nonorthogonal_correction_sweeps),
        "actual_sweeps": 0,
        "relaxation": float(config.pressure_nonorthogonal_correction_relaxation),
        "implicit_transmissibility": "legacy_A_over_center_distance",
        "implicit_matrix_unchanged": True,
        "spd_matrix_preserved": True,
        "rhs_term_sign": "plus_divergence_of_frozen_nonorthogonal_gradient_flux",
        "supported_rhs_mode": "volume_integrated_flux",
        "rhs_mode_contract_note": (
            "deferred_lsq rejects the legacy divergence_per_volume path because "
            "that RHS is not consistent with the existing volume-integrated SPD "
            "pressure matrix"
        ),
        "fixed_point_relaxation_scheme": (
            "persistent_frozen_flux_picard: "
            "c_next=(1-r)*c_previous+r*C(p_current) on every sweep, "
            "including the first sweep of each timestep"
        ),
        "fixed_point_spectral_model": (
            "stationary error factor H=(1-r)I+r*A^-1*D*C; persistent carry "
            "avoids the unstable warm-pressure polynomial from resetting the "
            "first sweep to r=1"
        ),
        "sweeps": [],
    }
    pressure_nonorthogonal_execution_backend = "disabled"
    pressure_nonorthogonal_cuda_used = False
    pressure_nonorthogonal_cuda_fallback_reason = ""
    pressure_nonorthogonal_residency: dict[str, Any] = {
        "inter_sweep_full_array_host_transfers": 0,
    }
    if config.pressure_nonorthogonal_correction_mode == "none":
        # Keep the legacy/default path as one unchanged pressure solve.
        p, p_diag, all_core_arrays_on_cuda = _solve_pressure_system(
            coeff,
            rhs=rhs,
            p0=p0,
            config=config,
            backend=backend,
        )
    else:
        # Warm-start the deferred fixed point from the incoming pressure state.
        # The first-ever step (p0=0) naturally reduces to a TPFA solve, while
        # later timesteps retain the previous non-orthogonal pressure structure.
        geometry = _cached_pressure_nonorthogonal_geometry(mesh, outlet_faces)
        total_true_residual_recomputes = 0
        total_true_residual_restarts = 0
        recursive_true_mismatch_l2_max = 0.0
        recursive_true_mismatch_max_abs_max = 0.0
        total_pressure_solve_wall_seconds = 0.0
        pressure_nonorthogonal_cuda_eligible = bool(
            backend.selected_backend == "torch"
            and str(config.device if config.device else backend.device).startswith(
                "cuda"
            )
            and config.pressure_solver == "pcg_diag"
        )
        if pressure_nonorthogonal_cuda_eligible:
            device = str(config.device if config.device else backend.device)
            resident_result = _solve_pressure_deferred_lsq_pcg_diag_torch(
                mesh,
                coeff,
                rhs_base=rhs_base,
                p0=p0,
                previous_frozen_gradient_flux=previous_frozen_gradient_flux,
                geometry=geometry,
                config=config,
                device=device,
            )
            p = np.asarray(resident_result["pressure"], dtype=np.float64)
            p_diag = dict(resident_result["pressure_diagnostics"])
            rhs = np.asarray(resident_result["rhs"], dtype=np.float64)
            rhs_nonorthogonal_term = np.asarray(
                resident_result["rhs_nonorthogonal_term"], dtype=np.float64
            )
            frozen_nonorthogonal_gradient_flux = np.asarray(
                resident_result["frozen_nonorthogonal_gradient_flux"],
                dtype=np.float64,
            )
            final_raw_flux = np.asarray(
                resident_result["final_raw_flux"], dtype=np.float64
            )
            nonorthogonal_gradient = np.asarray(
                resident_result["nonorthogonal_gradient"], dtype=np.float64
            )
            outer_defect = np.asarray(resident_result["outer_defect"], dtype=np.float64)
            sweep_rows = list(resident_result["sweeps"])
            total_pressure_iterations = int(
                resident_result["total_pressure_iterations"]
            )
            total_pressure_solve_wall_seconds = float(
                resident_result["total_pressure_solve_wall_seconds"]
            )
            total_true_residual_recomputes = int(
                resident_result["total_true_residual_recomputes"]
            )
            total_true_residual_restarts = int(
                resident_result["total_true_residual_restarts"]
            )
            recursive_true_mismatch_l2_max = float(
                resident_result["recursive_true_residual_mismatch_l2_max"]
            )
            recursive_true_mismatch_max_abs_max = float(
                resident_result["recursive_true_residual_mismatch_max_abs_max"]
            )
            all_core_arrays_on_cuda = bool(resident_result["all_sweeps_on_cuda"])
            pressure_nonorthogonal_execution_backend = "torch"
            pressure_nonorthogonal_cuda_used = True
            pressure_nonorthogonal_residency = {
                "geometry_cache_device": str(resident_result["geometry_cache_device"]),
                "host_to_device_full_array_transfers": int(
                    resident_result["host_to_device_full_array_transfers"]
                ),
                "device_to_host_full_array_transfers": int(
                    resident_result["device_to_host_full_array_transfers"]
                ),
                "inter_sweep_full_array_host_transfers": int(
                    resident_result["inter_sweep_full_array_host_transfers"]
                ),
            }
        else:
            p = np.asarray(p0, dtype=np.float64).copy()
            p_diag = {}
            all_core_arrays_on_cuda = bool(backend.selected_backend == "torch")
            relaxation = float(config.pressure_nonorthogonal_correction_relaxation)
            relaxed_flux = previous_frozen_gradient_flux.copy()
            sweep_rows = []
            total_pressure_iterations = 0
            for sweep_idx in range(
                int(config.pressure_nonorthogonal_correction_sweeps)
            ):
                raw_flux, gradient_before_solve = (
                    _pressure_nonorthogonal_gradient_flux_numpy(
                        mesh,
                        p,
                        dt=float(config.projection_dt),
                        density=float(config.density),
                        pressure_outlet_value=float(config.pressure_outlet_value),
                        geometry=geometry,
                    )
                )
                relaxed_before = relaxed_flux.copy()
                relaxed_flux = (
                    1.0 - relaxation
                ) * relaxed_before + relaxation * raw_flux
                rhs_nonorthogonal_term = _pressure_nonorthogonal_rhs_term(
                    mesh,
                    relaxed_flux,
                    rhs_mode=config.projection_rhs_mode,
                )
                rhs_sweep = rhs_base + rhs_nonorthogonal_term
                pressure_before = np.asarray(p, dtype=np.float64).copy()
                p, sweep_pressure_diag, sweep_cuda = _solve_pressure_system(
                    coeff,
                    rhs=rhs_sweep,
                    p0=pressure_before,
                    config=config,
                    backend=backend,
                )
                all_core_arrays_on_cuda = bool(all_core_arrays_on_cuda and sweep_cuda)
                p_diag = sweep_pressure_diag
                total_pressure_iterations += int(
                    sweep_pressure_diag.get("actual_iterations", 0)
                )
                total_pressure_solve_wall_seconds += float(
                    sweep_pressure_diag.get("solve_wall_seconds", 0.0)
                )
                total_true_residual_recomputes += int(
                    sweep_pressure_diag.get("true_residual_recompute_count", 0)
                )
                total_true_residual_restarts += int(
                    sweep_pressure_diag.get("true_residual_restart_count", 0)
                )
                recursive_true_mismatch_l2_max = max(
                    recursive_true_mismatch_l2_max,
                    float(
                        sweep_pressure_diag.get(
                            "recursive_true_residual_mismatch_l2_max", 0.0
                        )
                    ),
                )
                recursive_true_mismatch_max_abs_max = max(
                    recursive_true_mismatch_max_abs_max,
                    float(
                        sweep_pressure_diag.get(
                            "recursive_true_residual_mismatch_max_abs_max", 0.0
                        )
                    ),
                )
                sweep_rows.append(
                    {
                        "sweep": int(sweep_idx + 1),
                        "raw_gradient_flux": _vector_stats(raw_flux),
                        "frozen_gradient_flux_before": _vector_stats(relaxed_before),
                        "frozen_gradient_flux_used": _vector_stats(relaxed_flux),
                        "frozen_flux_change": _vector_stats(
                            relaxed_flux - relaxed_before
                        ),
                        "lsq_cell_gradient": _vector_stats(gradient_before_solve),
                        "rhs_nonorthogonal_term": _vector_stats(rhs_nonorthogonal_term),
                        "rhs_full": _vector_stats(rhs_sweep),
                        "pressure_change": _vector_stats(p - pressure_before),
                        "pressure_solver": {
                            "actual_iterations": int(
                                sweep_pressure_diag.get("actual_iterations", 0)
                            ),
                            "stopping_reason": str(
                                sweep_pressure_diag.get("stopping_reason", "")
                            ),
                            "pressure_solved": bool(
                                sweep_pressure_diag.get("pressure_solved", False)
                            ),
                            "solve_wall_seconds": float(
                                sweep_pressure_diag.get("solve_wall_seconds", 0.0)
                            ),
                            "residual_ratio_to_rhs_l2": float(
                                sweep_pressure_diag.get("residual_ratio_to_rhs_l2", 0.0)
                            ),
                            "residual_ratio_to_rhs_max": float(
                                sweep_pressure_diag.get(
                                    "residual_ratio_to_rhs_max", 0.0
                                )
                            ),
                            "true_residual_recompute_count": int(
                                sweep_pressure_diag.get(
                                    "true_residual_recompute_count", 0
                                )
                            ),
                            "true_residual_restart_count": int(
                                sweep_pressure_diag.get(
                                    "true_residual_restart_count", 0
                                )
                            ),
                            "recursive_true_residual_mismatch_l2_max": float(
                                sweep_pressure_diag.get(
                                    "recursive_true_residual_mismatch_l2_max", 0.0
                                )
                            ),
                            "recursive_true_residual_mismatch_max_abs_max": float(
                                sweep_pressure_diag.get(
                                    "recursive_true_residual_mismatch_max_abs_max", 0.0
                                )
                            ),
                        },
                    }
                )
                rhs = rhs_sweep
                frozen_nonorthogonal_gradient_flux = relaxed_flux.copy()

            final_raw_flux, nonorthogonal_gradient = (
                _pressure_nonorthogonal_gradient_flux_numpy(
                    mesh,
                    p,
                    dt=float(config.projection_dt),
                    density=float(config.density),
                    pressure_outlet_value=float(config.pressure_outlet_value),
                    geometry=geometry,
                )
            )
            outer_defect = final_raw_flux - frozen_nonorthogonal_gradient_flux
            pressure_nonorthogonal_execution_backend = "numpy_vectorized"
            if backend.selected_backend != "torch":
                pressure_nonorthogonal_cuda_fallback_reason = (
                    "flow execution backend is not Torch"
                )
            elif not str(config.device if config.device else backend.device).startswith(
                "cuda"
            ):
                pressure_nonorthogonal_cuda_fallback_reason = (
                    "flow execution device is not CUDA"
                )
            else:
                pressure_nonorthogonal_cuda_fallback_reason = (
                    "CUDA deferred LSQ residency requires pressure_solver='pcg_diag'"
                )
        rhs_expected = rhs_base + rhs_nonorthogonal_term
        equation_count = np.asarray(geometry["lsq_equation_count"], dtype=np.int32)
        normal_matrix_rank = np.asarray(
            geometry["lsq_normal_matrix_rank"], dtype=np.int32
        )
        normal_matrix_condition = np.asarray(
            geometry["lsq_normal_matrix_condition"], dtype=np.float64
        )
        interpolation = np.asarray(geometry["interior_lambda"], dtype=np.float64)
        outer_defect_relative_l2 = _safe_ratio(
            float(np.sqrt(np.mean(outer_defect * outer_defect))),
            max(
                float(
                    np.sqrt(
                        np.mean(
                            frozen_nonorthogonal_gradient_flux
                            * frozen_nonorthogonal_gradient_flux
                        )
                    )
                ),
                float(np.sqrt(np.mean(final_raw_flux * final_raw_flux))),
                1e-30,
            ),
        )
        outer_defect_warning_threshold = 0.03
        outer_defect_stability_warning = bool(
            not np.isfinite(outer_defect_relative_l2)
            or outer_defect_relative_l2 > outer_defect_warning_threshold
        )
        nonorthogonal_correction_diag = {
            **nonorthogonal_correction_diag,
            "enabled": True,
            "actual_sweeps": int(len(sweep_rows)),
            "execution_backend": str(pressure_nonorthogonal_execution_backend),
            "cuda_used": bool(pressure_nonorthogonal_cuda_used),
            "cuda_fallback_reason": str(pressure_nonorthogonal_cuda_fallback_reason),
            "cuda_residency": dict(pressure_nonorthogonal_residency),
            "stability_warning": bool(outer_defect_stability_warning),
            "stability_warning_threshold_outer_defect_relative_l2": float(
                outer_defect_warning_threshold
            ),
            "stability_warning_reason": (
                "final outer fixed-point defect exceeds 0.03"
                if outer_defect_stability_warning
                else "final outer fixed-point defect is within 0.03"
            ),
            "stability_note": (
                "warning is based on the final normalized outer fixed-point "
                "defect; a startup-step warning is diagnostic and is not by "
                "itself a run failure"
            ),
            "stability_reference": {
                "operator": (
                    "LSQ with homogeneous Neumann normal rows on every non-outlet "
                    "boundary face"
                ),
                "mesh_cell_counts": [632, 1110, 2426, 5144],
                "estimated_spectral_radius_A_inverse_D_C": [
                    0.268609,
                    0.353262,
                    0.359847,
                    0.361178,
                ],
                "selected_default_relaxation": 1.0,
                "note": (
                    "for a stationary eigenmode lambda, persistent relaxation "
                    "amplifies frozen-flux error by 1-r+r*lambda per sweep; all "
                    "measured corrected-operator spectral radii are below one"
                ),
            },
            "mixed_backend_reason": (
                ""
                if pressure_nonorthogonal_cuda_used
                else "deferred LSQ geometry, gradients, and RHS assembly execute on CPU"
            ),
            "geometry_build_seconds": float(geometry["build_seconds"]),
            "lsq_equation_count": {
                "min": int(np.min(equation_count)) if equation_count.size else 0,
                "max": int(np.max(equation_count)) if equation_count.size else 0,
                "mean": float(np.mean(equation_count)) if equation_count.size else 0.0,
                "cells_with_fewer_than_three_equations": int(
                    np.count_nonzero(equation_count < 3)
                ),
            },
            "lsq_boundary_conditions": {
                "outlet_dirichlet_equation_count": int(
                    np.asarray(geometry["outlet_faces"], dtype=np.int64).size
                ),
                "nonoutlet_zero_neumann_equation_count": int(
                    np.asarray(geometry["neumann_faces"], dtype=np.int64).size
                ),
                "nonoutlet_condition": "grad_pressure_dot_outward_normal_equals_zero",
                "matches_pinned_projection_correction_flux": True,
            },
            "lsq_normal_matrix_rank": {
                "min": int(np.min(normal_matrix_rank))
                if normal_matrix_rank.size
                else 0,
                "max": int(np.max(normal_matrix_rank))
                if normal_matrix_rank.size
                else 0,
                "full_rank_cell_count": int(np.count_nonzero(normal_matrix_rank == 3)),
                "rank_deficient_cell_count": int(
                    np.count_nonzero(normal_matrix_rank < 3)
                ),
            },
            "lsq_normal_matrix_condition": _vector_stats(normal_matrix_condition),
            "face_interpolation_lambda": _vector_stats(interpolation),
            "initial_pressure_source": "incoming_state.pressure",
            "initial_pressure": _vector_stats(p0),
            "initial_frozen_gradient_flux_source": str(
                previous_frozen_gradient_flux_source
            ),
            "initial_frozen_gradient_flux": _vector_stats(
                previous_frozen_gradient_flux
            ),
            "gradient_flux_scale_dt_over_density": float(current_gradient_flux_scale),
            "previous_frozen_gradient_flux_scaling": (
                "none; accepted pressure-correction face-volume flux is carried "
                "unchanged across dt updates"
            ),
            "pressure_solve_count": int(len(sweep_rows)),
            "total_pressure_iterations": int(total_pressure_iterations),
            "total_true_residual_recomputes": int(total_true_residual_recomputes),
            "total_true_residual_restarts": int(total_true_residual_restarts),
            "recursive_true_residual_mismatch_l2_max": float(
                recursive_true_mismatch_l2_max
            ),
            "recursive_true_residual_mismatch_max_abs_max": float(
                recursive_true_mismatch_max_abs_max
            ),
            "total_pressure_solve_wall_seconds": float(
                total_pressure_solve_wall_seconds
            ),
            "sweeps": sweep_rows,
            "frozen_gradient_flux_used_for_final_rhs": _vector_stats(
                frozen_nonorthogonal_gradient_flux
            ),
            "raw_gradient_flux_from_final_pressure": _vector_stats(final_raw_flux),
            "outer_fixed_point_defect": _vector_stats(outer_defect),
            "outer_fixed_point_defect_relative_l2": float(outer_defect_relative_l2),
            "final_lsq_cell_gradient": _vector_stats(nonorthogonal_gradient),
            "rhs_nonorthogonal_term": _vector_stats(rhs_nonorthogonal_term),
        }
        if not pressure_nonorthogonal_cuda_used:
            # Pressure linear solves may still run on CUDA, but the complete
            # deferred correction remains mixed-backend on this fallback path.
            all_core_arrays_on_cuda = False
        p_diag = {
            **p_diag,
            "nonorthogonal_total_actual_iterations": int(total_pressure_iterations),
            "nonorthogonal_total_solve_wall_seconds": float(
                total_pressure_solve_wall_seconds
            ),
            "nonorthogonal_correction_sweeps": int(len(sweep_rows)),
        }

    grad_flux = _pressure_face_gradient_flux(
        mesh,
        p,
        dt=float(config.projection_dt),
        density=float(config.density),
        outlet_faces=outlet_faces,
        pressure_outlet_value=float(config.pressure_outlet_value),
        frozen_nonorthogonal_gradient_flux=(
            frozen_nonorthogonal_gradient_flux
            if config.pressure_nonorthogonal_correction_mode == "deferred_lsq"
            else None
        ),
        coeff=coeff,
    )
    orthogonal_grad_flux = (
        grad_flux - frozen_nonorthogonal_gradient_flux
        if config.pressure_nonorthogonal_correction_mode == "deferred_lsq"
        else np.asarray(grad_flux, dtype=np.float64).copy()
    )
    _, correction_flux_raw = _apply_projection_correction(
        flux_star,
        grad_flux,
        projection_sign=config.projection_sign,
        damping=float(config.projection_correction_damping),
    )
    correction_flux_raw_pre_constraint = np.asarray(
        correction_flux_raw,
        dtype=np.float64,
    ).copy()
    correction_flux_constrained, pinned_raw_diag = _pin_projection_correction_faces(
        correction_flux_raw,
        left_faces=left_faces,
        right_faces=right_faces,
        wall_faces=wall_faces,
    )
    masks = _build_region_masks(
        mesh,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    correction_flux_limited, limiter_diag = _apply_projection_correction_limiter(
        mesh=mesh,
        flux_star=flux_star,
        correction_flux_raw=correction_flux_constrained,
        div_star=div_star,
        masks=masks,
        left_faces=left_faces,
        right_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
        config=config,
    )
    correction_flux_limiter_output_pre_reconstraint = np.asarray(
        correction_flux_limited,
        dtype=np.float64,
    ).copy()
    (
        correction_flux_constrained_post_limiter,
        pinned_limiter_output_diag,
    ) = _pin_projection_correction_faces(
        correction_flux_limiter_output_pre_reconstraint,
        left_faces=left_faces,
        right_faces=right_faces,
        wall_faces=wall_faces,
    )
    flux_corr_limited_raw = np.asarray(flux_star, dtype=np.float64) + np.asarray(
        correction_flux_constrained_post_limiter,
        dtype=np.float64,
    )
    inlet_faces = np.unique(np.concatenate((left_faces, right_faces)))
    flux_corr, outlet_policy_diag = _apply_outlet_projection_mode(
        flux_star=flux_star,
        flux_corrected=flux_corr_limited_raw,
        outlet_faces=outlet_faces,
        inlet_faces=inlet_faces,
        wall_faces=wall_faces,
        mode=config.outlet_projection_mode,
    )
    correction_flux_effective = np.asarray(flux_corr, dtype=np.float64) - np.asarray(
        flux_star,
        dtype=np.float64,
    )
    correction_boundary_contract = {
        "raw_pre_constraint": _projection_correction_boundary_contract_audit(
            correction_flux_raw_pre_constraint,
            n_faces=flux_star.shape[0],
            left_faces=left_faces,
            right_faces=right_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
        ),
        "constrained_pre_limiter": _projection_correction_boundary_contract_audit(
            correction_flux_constrained,
            n_faces=flux_star.shape[0],
            left_faces=left_faces,
            right_faces=right_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
        ),
        "limiter_output_pre_reconstraint": (
            _projection_correction_boundary_contract_audit(
                correction_flux_limiter_output_pre_reconstraint,
                n_faces=flux_star.shape[0],
                left_faces=left_faces,
                right_faces=right_faces,
                outlet_faces=outlet_faces,
                wall_faces=wall_faces,
            )
        ),
        "constrained_post_limiter_pre_outlet_policy": (
            _projection_correction_boundary_contract_audit(
                correction_flux_constrained_post_limiter,
                n_faces=flux_star.shape[0],
                left_faces=left_faces,
                right_faces=right_faces,
                outlet_faces=outlet_faces,
                wall_faces=wall_faces,
            )
        ),
        "effective_post_outlet_policy": _projection_correction_boundary_contract_audit(
            correction_flux_effective,
            n_faces=flux_star.shape[0],
            left_faces=left_faces,
            right_faces=right_faces,
            outlet_faces=outlet_faces,
            wall_faces=wall_faces,
        ),
    }

    div_corr = _compute_divergence_numpy(mesh, flux_corr)
    cell_velocity_star = np.asarray(state.cell_velocity, dtype=np.float64)
    pressure_velocity_increment = _reconstruct_cell_velocity_from_face_flux_numpy(
        mesh,
        correction_flux_effective,
        wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
        wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
    )
    momentum_pressure_corrected = bool(
        config.projection_cell_velocity_update_mode == "momentum_pressure_corrected"
    )
    cached_slip_reconstruction_used = bool(
        str(config.wall_velocity_boundary_mode) == "slip"
    )
    if momentum_pressure_corrected:
        if np.any(correction_flux_effective != 0.0):
            speed_vel = cell_velocity_star + pressure_velocity_increment
        else:
            speed_vel = cell_velocity_star.copy()
    else:
        # Keep the historical default path numerically unchanged. In this mode
        # cell velocity remains a diagnostic reconstruction of the final mass flux.
        speed_vel = _reconstruct_cell_velocity_from_face_flux_numpy(
            mesh,
            flux_corr,
            wall_velocity_boundary_mode=_wall_reconstruction_boundary_mode(config),
            wall_tangential_no_slip_strength=config.wall_tangential_no_slip_strength,
        )
    cell_velocity_change = np.asarray(speed_vel, dtype=np.float64) - cell_velocity_star
    flux_from_velocity_star = _face_flux_from_cell_velocity_numpy(
        mesh,
        cell_velocity_star,
    )
    flux_from_pressure_velocity_increment = _face_flux_from_cell_velocity_numpy(
        mesh,
        pressure_velocity_increment,
    )
    flux_from_velocity_corrected = _face_flux_from_cell_velocity_numpy(
        mesh,
        speed_vel,
    )
    speed_mag = np.linalg.norm(speed_vel, axis=1)
    projection_velocity_update = {
        "mode": str(config.projection_cell_velocity_update_mode),
        "cached_slip_reconstruction_used": bool(cached_slip_reconstruction_used),
        "slip_dense_reconstruction_solves_avoided": int(
            (1 if momentum_pressure_corrected else 2)
            if cached_slip_reconstruction_used
            else 0
        ),
        "cell_velocity_star_source": "incoming_state.cell_velocity",
        "cell_velocity_output_source": (
            "cell_velocity_star_plus_effective_face_flux_correction_increment"
            if momentum_pressure_corrected
            else "legacy_reconstruction_from_corrected_face_flux"
        ),
        "momentum_state_preserved": bool(momentum_pressure_corrected),
        "zero_effective_correction": bool(not np.any(correction_flux_effective != 0.0)),
        "effective_face_flux_correction": _vector_stats(correction_flux_effective),
        "pressure_velocity_increment": _vector_stats(pressure_velocity_increment),
        "actual_cell_velocity_change": _vector_stats(cell_velocity_change),
        "cell_velocity_star_face_flux_consistency": (
            _face_flux_reinterpolation_consistency_audit(
                mesh,
                reference_face_flux=flux_star,
                reinterpolated_face_flux=flux_from_velocity_star,
            )
        ),
        "pressure_increment_face_flux_consistency": (
            _face_flux_reinterpolation_consistency_audit(
                mesh,
                reference_face_flux=correction_flux_effective,
                reinterpolated_face_flux=flux_from_pressure_velocity_increment,
            )
        ),
        "cell_velocity_output_face_flux_consistency": (
            _face_flux_reinterpolation_consistency_audit(
                mesh,
                reference_face_flux=flux_corr,
                reinterpolated_face_flux=flux_from_velocity_corrected,
            )
        ),
    }

    before_flux_audit = _boundary_flux_audit(
        mesh,
        flux_star,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )
    after_flux_audit = _boundary_flux_audit(
        mesh,
        flux_corr,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=outlet_faces,
        wall_faces=wall_faces,
    )

    region_audit = _region_divergence_audit(div_star, div_corr, masks)
    expected_outlet_flux = float(
        before_flux_audit["left_inlet"]["inflow_total"]
        + before_flux_audit["right_inlet"]["inflow_total"]
        - before_flux_audit["walls"]["sum_flux"]
    )
    outlet_audit = _outlet_projection_audit(
        mesh=mesh,
        outlet_faces=outlet_faces,
        flux_star=flux_star,
        correction_flux=correction_flux_effective,
        flux_corrected=flux_corr,
        pressure=p,
        pressure_outlet_value=float(config.pressure_outlet_value),
        expected_outlet_flux=expected_outlet_flux,
    )
    top_div_cells = _top_divergence_cells(
        mesh=mesh,
        div_star=div_star,
        div_corr=div_corr,
        face_flux_corrected=flux_corr,
        masks=masks,
        top_k=50,
    )

    ap = _matvec_pressure_numpy(coeff, p)
    residual = ap - rhs
    rhs_diff = rhs - rhs_expected
    div_check = _compute_divergence_numpy(mesh, flux_corr)
    div_check_diff = div_check - div_corr

    projection_audit = {
        "sign_convention": {
            "projection_sign": config.projection_sign,
            "projection_rhs_mode": config.projection_rhs_mode,
            "pressure_nonorthogonal_correction_mode": str(
                config.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_sweeps": int(
                config.pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                config.pressure_nonorthogonal_correction_relaxation
            ),
            "projection_correction_damping": float(
                config.projection_correction_damping
            ),
            "projection_correction_limit_mode": config.projection_correction_limit_mode,
            "projection_divergence_cap_factor": float(
                config.projection_divergence_cap_factor
            ),
            "projection_face_correction_over_volume_cap": float(
                config.projection_face_correction_over_volume_cap
            ),
            "correction_formula": (
                "q_corr = q_star - damping * grad_flux"
                if config.projection_sign == "minus"
                else "q_corr = q_star + damping * grad_flux"
            ),
            "rhs_formula": (
                "rhs = -source + outlet_dirichlet_term + D(nonorth_flux_frozen)"
                if config.projection_sign == "minus"
                else "rhs = +source + outlet_dirichlet_term + D(nonorth_flux_frozen)"
            ),
            "rhs_source_definition": (
                "source = cell_flux_sum / cell_volume"
                if config.projection_rhs_mode == "divergence_per_volume"
                else "source = cell_flux_sum"
            ),
        },
        "stage_metrics": {
            "face_flux_star": _vector_stats(flux_star),
            "div_star": _vector_stats(div_star),
            "poisson_rhs_base": _vector_stats(rhs_base),
            "poisson_rhs_nonorthogonal_term": _vector_stats(rhs_nonorthogonal_term),
            "poisson_rhs": _vector_stats(rhs),
            "pressure": _vector_stats(p),
            "pressure_gradient_orthogonal_flux": _vector_stats(orthogonal_grad_flux),
            "pressure_gradient_nonorthogonal_flux_frozen": _vector_stats(
                frozen_nonorthogonal_gradient_flux
            ),
            "pressure_gradient_normal_flux": _vector_stats(grad_flux),
            "correction_flux_raw_pre_constraint": _vector_stats(
                correction_flux_raw_pre_constraint
            ),
            "correction_flux_constrained_pre_limiter": _vector_stats(
                correction_flux_constrained
            ),
            "correction_flux_limiter_output_pre_reconstraint": _vector_stats(
                correction_flux_limiter_output_pre_reconstraint
            ),
            "correction_flux_constrained_post_limiter_pre_outlet_policy": (
                _vector_stats(correction_flux_constrained_post_limiter)
            ),
            "correction_flux_effective_post_outlet_policy": _vector_stats(
                correction_flux_effective
            ),
            "correction_flux_raw": _vector_stats(correction_flux_raw_pre_constraint),
            "correction_flux_limited_pre_bc": _vector_stats(
                correction_flux_constrained_post_limiter
            ),
            "correction_flux_effective_post_bc": _vector_stats(
                correction_flux_effective
            ),
            "face_flux_corrected": _vector_stats(flux_corr),
            "div_corrected": _vector_stats(div_corr),
            "divergence_reduction_ratio_linf": _safe_ratio(
                float(np.max(np.abs(div_corr))),
                float(np.max(np.abs(div_star))),
            ),
            "divergence_reduction_ratio_l2": _safe_ratio(
                float(np.sqrt(np.mean(div_corr**2))),
                float(np.sqrt(np.mean(div_star**2))),
            ),
        },
        "consistency_checks": {
            "rhs_from_code_vs_rhs_expected": _vector_stats(rhs_diff),
            "laplacian_pressure_minus_rhs_residual": _vector_stats(residual),
            "div_after_formula_check": {
                "max_abs_diff": float(np.max(np.abs(div_check_diff))),
                "mean_abs_diff": float(np.mean(np.abs(div_check_diff))),
                "l2_diff": float(np.sqrt(np.mean(div_check_diff**2))),
            },
            "operator_consistency_max_abs": float(np.max(np.abs(residual))),
            "operator_consistency_l2": float(np.sqrt(np.mean(residual**2))),
            "operator_consistency_mean_abs": float(np.mean(np.abs(residual))),
        },
    }

    sign_comparison = {}
    if config.enable_sign_comparison:
        sign_comparison = {
            "minus": _run_sign_trial(
                mesh,
                flux_star=flux_star,
                grad_flux=grad_flux,
                sign="minus",
                left_faces=left_faces,
                right_faces=right_faces,
                outlet_faces=outlet_faces,
                wall_faces=wall_faces,
                damping=float(config.projection_correction_damping),
            ),
            "plus": _run_sign_trial(
                mesh,
                flux_star=flux_star,
                grad_flux=grad_flux,
                sign="plus",
                left_faces=left_faces,
                right_faces=right_faces,
                outlet_faces=outlet_faces,
                wall_faces=wall_faces,
                damping=float(config.projection_correction_damping),
            ),
        }
        sign_comparison["fixed_rhs_pressure_source"] = (
            "single rhs + single pressure solution reused for both signs"
        )
        if (
            sign_comparison["minus"]["final_divergence_max_abs"]
            <= sign_comparison["plus"]["final_divergence_max_abs"]
        ):
            sign_comparison["recommended_projection_sign"] = "minus"
        else:
            sign_comparison["recommended_projection_sign"] = "plus"

    d0 = float(np.max(np.abs(div_star)))
    d1 = float(np.max(np.abs(div_corr)))
    reduction = _safe_ratio(d1, d0)
    pressure_solved = bool(p_diag.get("pressure_solved", False))
    diagnostics = {
        "flow_solved": False,
        "pressure_solved": pressure_solved,
        "numerical_profile_resolution": _numerical_profile_resolution_diagnostics(
            requested_config,
            config,
        ),
        "pressure_solver": config.pressure_solver,
        "projection_rhs_mode": config.projection_rhs_mode,
        "pressure_nonorthogonal_correction_mode": str(
            config.pressure_nonorthogonal_correction_mode
        ),
        "pressure_nonorthogonal_correction_sweeps": int(
            config.pressure_nonorthogonal_correction_sweeps
        ),
        "pressure_nonorthogonal_correction_relaxation": float(
            config.pressure_nonorthogonal_correction_relaxation
        ),
        "projection_correction_damping": float(config.projection_correction_damping),
        "projection_correction_limit_mode": config.projection_correction_limit_mode,
        "projection_cell_velocity_update_mode": str(
            config.projection_cell_velocity_update_mode
        ),
        "projection_limit_experimental": bool(
            config.projection_correction_limit_mode != "none"
        ),
        "backend_execution": {
            "requested_backend": config.backend,
            "selected_backend": backend.selected_backend,
            "device": config.device if config.device else backend.device,
            "used_numpy_fallback": bool(used_numpy_fallback),
            "all_core_arrays_on_cuda": bool(all_core_arrays_on_cuda),
            "pressure_nonorthogonal_execution_backend": str(
                pressure_nonorthogonal_execution_backend
            ),
            "pressure_nonorthogonal_cuda_used": bool(pressure_nonorthogonal_cuda_used),
            "pressure_nonorthogonal_cuda_fallback_reason": str(
                pressure_nonorthogonal_cuda_fallback_reason
            ),
            "mixed_backend_pressure_nonorthogonal_correction": bool(
                config.pressure_nonorthogonal_correction_mode == "deferred_lsq"
                and not pressure_nonorthogonal_cuda_used
            ),
            "torch_available": backend.torch_available,
            "torch_version": backend.torch_version,
            "torch_cuda_available": backend.torch_cuda_available,
            "torch_gpu_name": backend.torch_gpu_name,
        },
        "projection": {
            "initial_divergence_max_abs": d0,
            "final_divergence_max_abs": d1,
            "divergence_reduction_ratio": reduction,
            "initial_divergence_l2": float(np.sqrt(np.mean(div_star**2))),
            "final_divergence_l2": float(np.sqrt(np.mean(div_corr**2))),
            "divergence_reduction_ratio_l2": _safe_ratio(
                float(np.sqrt(np.mean(div_corr**2))),
                float(np.sqrt(np.mean(div_star**2))),
            ),
            "net_boundary_flux_before": float(
                before_flux_audit["boundary_total"]["sum_flux"]
            ),
            "net_boundary_flux_after": float(
                after_flux_audit["boundary_total"]["sum_flux"]
            ),
            "inlet_flux_total_before": float(
                before_flux_audit["left_inlet"]["inflow_total"]
                + before_flux_audit["right_inlet"]["inflow_total"]
            ),
            "inlet_flux_total_after": float(
                after_flux_audit["left_inlet"]["inflow_total"]
                + after_flux_audit["right_inlet"]["inflow_total"]
            ),
            "outlet_flux_total_before": float(
                before_flux_audit["outlet"]["outflow_total"]
            ),
            "outlet_flux_total_after": float(
                after_flux_audit["outlet"]["outflow_total"]
            ),
            "wall_flux_max_abs_after": float(after_flux_audit["walls"]["max_abs"]),
            "outlet_projection_mode": str(config.outlet_projection_mode),
            "pressure_projection_outlet_contract_mode": str(
                config.pressure_projection_outlet_contract_mode
            ),
            "pressure_nonorthogonal_correction_mode": str(
                config.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_sweeps": int(
                config.pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                config.pressure_nonorthogonal_correction_relaxation
            ),
            "projection_cell_velocity_update_mode": str(
                config.projection_cell_velocity_update_mode
            ),
            "momentum_cell_velocity_state_preserved": bool(momentum_pressure_corrected),
            "outlet_flux_rescale_used": bool(
                outlet_policy_diag.get("outlet_flux_rescale_used", False)
            ),
            "outlet_flux_rescale_factor": float(
                outlet_policy_diag.get("outlet_flux_rescale_factor", 1.0)
            ),
            "outlet_flux_rescale_reason": str(
                outlet_policy_diag.get("outlet_flux_rescale_reason", "")
            ),
            "nonphysical_flux_fix_used": bool(
                outlet_policy_diag.get("nonphysical_flux_fix_used", False)
            ),
        },
        "pressure": {
            "min": float(np.min(p)),
            "max": float(np.max(p)),
            "mean": float(np.mean(p)),
            "l2": float(np.sqrt(np.mean(p * p))),
            **p_diag,
        },
        "velocity": {
            "magnitude_min": float(np.min(speed_mag)),
            "magnitude_max": float(np.max(speed_mag)),
            "magnitude_mean": float(np.mean(speed_mag)),
            "projection_cell_velocity_update_mode": str(
                config.projection_cell_velocity_update_mode
            ),
        },
        "projection_velocity_update": projection_velocity_update,
        "pressure_nonorthogonal_correction": nonorthogonal_correction_diag,
        "thresholds": {
            "pressure_tolerance": float(config.pressure_tolerance),
            "divergence_tolerance": float(config.divergence_tolerance),
        },
        "projection_audit": projection_audit,
        "projection_correction_stage_codebook": dict(
            PROJECTION_CORRECTION_STAGE_CODEBOOK
        ),
        "face_flux_primary_stage_codebook": dict(FACE_FLUX_PRIMARY_STAGE_CODEBOOK),
        "correction_limiter": limiter_diag,
        "correction_limiter_conservation_audit": dict(
            limiter_diag.get("conservation_audit", {})
        ),
        "projection_pinned_face_constraints": {
            "raw_pre_constraint": pinned_raw_diag,
            "limiter_output_pre_reconstraint": pinned_limiter_output_diag,
            "limiter_reintroduced_pinned_faces": bool(
                int(
                    pinned_limiter_output_diag.get(
                        "pinned_face_nonzero_before_count",
                        0,
                    )
                )
                > 0
            ),
        },
        "projection_correction_boundary_contract": correction_boundary_contract,
        "pressure_solver_history": {
            "rhs_stats": _vector_stats(rhs),
            "rhs_base_stats": _vector_stats(rhs_base),
            "rhs_nonorthogonal_term_stats": _vector_stats(rhs_nonorthogonal_term),
            "rhs_outlet_term_stats": _vector_stats(outlet_rhs),
            "rhs_mode": config.projection_rhs_mode,
            "pressure_nonorthogonal_correction_mode": str(
                config.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_sweeps": int(
                config.pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                config.pressure_nonorthogonal_correction_relaxation
            ),
            "projection_correction_damping": float(
                config.projection_correction_damping
            ),
            **p_diag,
        },
        "boundary_flux_audit": {
            "before_projection": before_flux_audit,
            "after_projection": after_flux_audit,
        },
        "region_divergence_audit": region_audit,
        "operator_consistency_audit": {
            "laplacian_pressure_minus_rhs": _vector_stats(residual),
            "rhs_from_code_vs_rhs_expected": _vector_stats(rhs_diff),
            "rhs_base": _vector_stats(rhs_base),
            "rhs_nonorthogonal_term": _vector_stats(rhs_nonorthogonal_term),
            "operator_consistency_max_abs": float(np.max(np.abs(residual))),
            "operator_consistency_mean_abs": float(np.mean(np.abs(residual))),
            "operator_consistency_l2": float(np.sqrt(np.mean(residual**2))),
            "divergence_formula_check_max_abs_diff": float(
                np.max(np.abs(div_check_diff))
            ),
            "divergence_formula_check_l2_diff": float(
                np.sqrt(np.mean(div_check_diff**2))
            ),
        },
        "projection_sign_comparison_fixed_rhs": sign_comparison,
        "outlet_projection_audit": outlet_audit,
        "outlet_projection_policy": outlet_policy_diag,
        "face_flux_primary": {
            "face_flux_star": flux_star,
            "face_flux_corrected": flux_corr,
            "pressure_gradient_flux_orthogonal": orthogonal_grad_flux,
            "pressure_gradient_flux_nonorthogonal_frozen": (
                frozen_nonorthogonal_gradient_flux
            ),
            "pressure_gradient_flux_full": grad_flux,
            "correction_flux": correction_flux_effective,
            "correction_flux_effective_post_outlet_policy": correction_flux_effective,
            "correction_flux_effective_post_bc": correction_flux_effective,
            "correction_flux_raw_pre_constraint": correction_flux_raw_pre_constraint,
            "correction_flux_constrained_pre_limiter": correction_flux_constrained,
            "correction_flux_limiter_output_pre_reconstraint": (
                correction_flux_limiter_output_pre_reconstraint
            ),
            "correction_flux_constrained_post_limiter_pre_outlet_policy": (
                correction_flux_constrained_post_limiter
            ),
        },
        "top_divergence_cells": top_div_cells,
    }

    return TetraFlowState(
        cell_velocity=speed_vel,
        face_flux=flux_corr,
        pressure=p,
        diagnostics=diagnostics,
    )


def tetra_flow_step(
    mesh: ImportedTetraMesh,
    state: TetraFlowState,
    config: TetraFlowConfig,
) -> TetraFlowState:
    return solve_tetra_pressure_projection(mesh, state, config)
