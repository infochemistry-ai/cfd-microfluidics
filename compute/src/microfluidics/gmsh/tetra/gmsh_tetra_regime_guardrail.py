"""Warning-first scalar-transport regime guardrails for tetrahedral meshes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh

ScalarRegimeKind = Literal["mass", "thermal"]

REGIME_GUARDRAIL_POLICY_VERSION = "scalar_regime_guardrail_v2"
DEFAULT_MAX_GRID_PECLET = 2.0
DEFAULT_MAX_SCHMIDT = 1000.0
DEFAULT_MAX_PRANDTL = 100.0
_THRESHOLD_RTOL = 1e-12


def _within_threshold(value: float | None, threshold: float) -> bool:
    return bool(
        value is not None
        and np.isfinite(value)
        and value <= threshold * (1.0 + _THRESHOLD_RTOL)
    )


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, float(percentile)))


def _top_cell_records(values: np.ndarray, mesh: ImportedTetraMesh, top_k: int) -> list:
    numeric = np.asarray(values, dtype=np.float64)
    # Keep non-finite values visible at the head of the diagnostic records instead
    # of silently dropping them behind finite values.
    order_values = np.where(np.isfinite(numeric), numeric, np.inf)
    order = np.argsort(-order_values, kind="stable")
    records: list[dict[str, Any]] = []
    for cell_id in order[: max(0, int(top_k))].tolist():
        volume = float(mesh.cell_volumes[cell_id])
        records.append(
            {
                "cell_id": int(cell_id),
                "value": float(numeric[cell_id])
                if np.isfinite(numeric[cell_id])
                else None,
                "center_xyz": np.asarray(
                    mesh.cell_centers[cell_id], dtype=np.float64
                ).tolist(),
                "cell_volume_m3": volume if np.isfinite(volume) else None,
                "characteristic_length_m": (
                    float(volume ** (1.0 / 3.0))
                    if np.isfinite(volume) and volume > 0.0
                    else None
                ),
            }
        )
    return records


def _outgoing_volume_flux_by_cell(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
) -> np.ndarray:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )
    outgoing = np.zeros(np.asarray(mesh.cell_volumes).shape[0], dtype=np.float64)

    interior = c1 >= 0
    interior_ids = np.flatnonzero(interior)
    if interior_ids.size:
        donor = np.where(flux[interior_ids] >= 0.0, c0[interior_ids], c1[interior_ids])
        np.add.at(outgoing, donor, np.abs(flux[interior_ids]))

    # Boundary inflow does not contribute to outgoing conductance. The caller
    # validates the underlying velocity and area arrays before using this result.
    boundary_ids = np.flatnonzero(~interior)
    positive = boundary_ids[flux[boundary_ids] > 0.0]
    if positive.size:
        np.add.at(outgoing, c0[positive], flux[positive])
    return outgoing


def build_scalar_regime_audit(
    mesh: ImportedTetraMesh,
    face_normal_velocity: np.ndarray,
    *,
    diffusivity: float,
    kinematic_viscosity: float,
    scalar_kind: ScalarRegimeKind,
    max_grid_peclet: float = DEFAULT_MAX_GRID_PECLET,
    max_schmidt: float = DEFAULT_MAX_SCHMIDT,
    max_prandtl: float = DEFAULT_MAX_PRANDTL,
    top_k: int = 10,
) -> dict[str, Any]:
    """Audit scalar-regime quality without turning finite warnings into gates.

    ``Pe_grid = Q_out / (D * h)``, where ``h = V**(1/3)``, is the local
    finite-volume advection/diffusion conductance ratio.  It is not necessarily
    the classical ``|u| * h / D``: its geometric factor depends on cell shape
    and flow direction.  Finite high Pe/Sc/Pr values are advisory warnings;
    malformed or non-finite data are blocking errors.
    """

    if scalar_kind not in {"mass", "thermal"}:
        raise ValueError("scalar_kind must be 'mass' or 'thermal'.")
    for name, value in (
        ("max_grid_peclet", max_grid_peclet),
        ("max_schmidt", max_schmidt),
        ("max_prandtl", max_prandtl),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    volumes_raw = np.asarray(mesh.cell_volumes, dtype=np.float64)
    face_areas = np.asarray(mesh.face_areas, dtype=np.float64)
    face_velocity = np.asarray(face_normal_velocity, dtype=np.float64)
    face_count = int(face_areas.shape[0])
    error_codes: list[str] = []
    warning_codes: list[str] = []

    invalid_volume = (~np.isfinite(volumes_raw)) | (volumes_raw <= 0.0)
    nonfinite_volume_count = int(np.count_nonzero(~np.isfinite(volumes_raw)))
    invalid_volume_count = int(np.count_nonzero(invalid_volume))
    if invalid_volume_count:
        error_codes.append("invalid_cell_volume")

    invalid_face_area = (~np.isfinite(face_areas)) | (face_areas <= 0.0)
    nonfinite_face_area_count = int(np.count_nonzero(~np.isfinite(face_areas)))
    invalid_face_area_count = int(np.count_nonzero(invalid_face_area))
    if invalid_face_area_count:
        error_codes.append("invalid_face_area")

    velocity_shape_valid = bool(
        face_velocity.ndim == 1 and face_velocity.size == face_count
    )
    if not velocity_shape_valid:
        error_codes.append("invalid_face_normal_velocity_shape")
        velocity = np.full(face_count, np.nan, dtype=np.float64)
    else:
        velocity = face_velocity
    nonfinite_face_velocity_count = int(np.count_nonzero(~np.isfinite(velocity)))
    if nonfinite_face_velocity_count:
        error_codes.append("invalid_nonfinite_face_normal_velocity")

    h = np.full(volumes_raw.shape, np.nan, dtype=np.float64)
    valid_volumes = ~invalid_volume
    h[valid_volumes] = np.cbrt(volumes_raw[valid_volumes])
    outgoing = _outgoing_volume_flux_by_cell(mesh, velocity)
    nonfinite_outgoing_count = int(np.count_nonzero(~np.isfinite(outgoing)))
    if nonfinite_outgoing_count:
        error_codes.append("invalid_nonfinite_outgoing_volume_flux")

    diffusivity_value = float(diffusivity)
    viscosity_value = float(kinematic_viscosity)
    diffusivity_advection_only = bool(diffusivity_value == 0.0)
    diffusivity_valid = bool(
        np.isfinite(diffusivity_value) and diffusivity_value >= 0.0
    )
    viscosity_valid = bool(np.isfinite(viscosity_value) and viscosity_value >= 0.0)
    if not diffusivity_valid:
        error_codes.append("invalid_diffusivity")
    elif diffusivity_advection_only:
        warning_codes.append("warning_zero_diffusivity_advection_only")
    if not viscosity_valid:
        error_codes.append("invalid_kinematic_viscosity")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pe = outgoing / (diffusivity_value * h)
    nonfinite_grid_peclet_count = int(np.count_nonzero(~np.isfinite(pe)))
    if nonfinite_grid_peclet_count and not diffusivity_advection_only:
        error_codes.append("invalid_nonfinite_grid_peclet")

    pe_max = _percentile(pe, 100.0)
    pe_p95 = _percentile(pe, 95.0)
    pe_p99 = _percentile(pe, 99.0)
    pe_within_warning_threshold = _within_threshold(pe_max, float(max_grid_peclet))
    if pe_max is not None and np.isfinite(pe_max) and not pe_within_warning_threshold:
        warning_codes.append("warning_high_grid_peclet")

    with np.errstate(divide="ignore", invalid="ignore"):
        dimensionless_ratio = float(
            np.divide(np.float64(viscosity_value), np.float64(diffusivity_value))
        )
    if scalar_kind == "mass":
        ratio_name, ratio_symbol = "schmidt_number", "Sc"
        ratio_threshold, ratio_warning_code = float(max_schmidt), "warning_high_schmidt"
    else:
        ratio_name, ratio_symbol = "prandtl_number", "Pr"
        ratio_threshold, ratio_warning_code = float(max_prandtl), "warning_high_prandtl"
    ratio_within_warning_threshold = _within_threshold(
        dimensionless_ratio, ratio_threshold
    )
    if np.isfinite(dimensionless_ratio) and not ratio_within_warning_threshold:
        warning_codes.append(ratio_warning_code)

    blocking_error = bool(error_codes)
    supported_accuracy = bool(not warning_codes and not blocking_error)
    severity = "error" if blocking_error else "warning" if warning_codes else "ok"
    reason_codes = [*error_codes, *warning_codes]
    if blocking_error:
        human_readable_reason = (
            "Scalar-regime audit contains blocking invalid or non-finite data: "
            + ", ".join(error_codes)
        )
    elif warning_codes:
        human_readable_reason = (
            "Scalar-regime run may continue with advisory accuracy warnings: "
            + ", ".join(warning_codes)
        )
    else:
        human_readable_reason = (
            "Scalar regime is within the documented supported-accuracy policy."
        )

    return {
        "policy_version": REGIME_GUARDRAIL_POLICY_VERSION,
        "scalar_kind": scalar_kind,
        "support_status": "supported" if severity == "ok" else reason_codes[0],
        "supported_accuracy": supported_accuracy,
        "warning_codes": warning_codes,
        "error_codes": error_codes,
        "blocking_error": blocking_error,
        "severity": severity,
        "reason_codes": reason_codes,
        "human_readable_reason": human_readable_reason,
        "definitions": {
            "grid_peclet": "Pe_grid = Q_out / (diffusivity * h), h = cell_volume**(1/3)",
            "schmidt_number": "Sc = nu / D for mass transport",
            "prandtl_number": "Pr = nu / alpha for thermal transport",
            "note": "Pe_grid is a finite-volume conductance ratio, not necessarily classical |u|*h/D or a CFL metric.",
        },
        "units": {
            "kinematic_viscosity": "m^2/s",
            "diffusivity": "m^2/s",
            "thermal_diffusivity": "m^2/s",
            "outgoing_volume_flux": "m^3/s",
            "characteristic_length": "m",
        },
        "policy_thresholds": {
            "max_grid_peclet": float(max_grid_peclet),
            "max_schmidt": float(max_schmidt),
            "max_prandtl": float(max_prandtl),
        },
        "inputs": {
            "diffusivity": diffusivity_value,
            "kinematic_viscosity": viscosity_value,
            "dimensionless_ratio": (
                float(dimensionless_ratio) if np.isfinite(dimensionless_ratio) else None
            ),
            "dimensionless_ratio_name": ratio_name,
            "dimensionless_ratio_symbol": ratio_symbol,
            "dimensionless_ratio_is_undefined": bool(
                not np.isfinite(dimensionless_ratio)
            ),
            # Backward-compatible scalar-kind-specific aliases.
            ratio_name: float(dimensionless_ratio)
            if np.isfinite(dimensionless_ratio)
            else None,
        },
        "diagnostics": {
            "face_normal_velocity_shape": list(face_velocity.shape),
            "expected_face_normal_velocity_count": face_count,
            "nonfinite_face_velocity_count": nonfinite_face_velocity_count,
            "nonfinite_face_area_count": nonfinite_face_area_count,
            "invalid_face_area_count": invalid_face_area_count,
            "nonfinite_cell_volume_count": nonfinite_volume_count,
            "invalid_cell_volume_count": invalid_volume_count,
            "nonfinite_outgoing_volume_flux_count": nonfinite_outgoing_count,
            "nonfinite_grid_peclet_cell_count": nonfinite_grid_peclet_count,
        },
        "grid_peclet": {
            "max": pe_max,
            "p99": pe_p99,
            "p95": pe_p95,
            "active_cell_count": int(
                np.count_nonzero(np.isfinite(outgoing) & (outgoing > 0.0))
            ),
            "top_cells": _top_cell_records(pe, mesh, top_k),
        },
        "checks": {
            "audit_input_valid": not blocking_error,
            "blocking_error_absent": not blocking_error,
            "grid_peclet_within_warning_threshold": pe_within_warning_threshold,
            "dimensionless_ratio_within_warning_threshold": ratio_within_warning_threshold,
            f"{ratio_symbol.lower()}_within_warning_threshold": ratio_within_warning_threshold,
            # Backward-compatible v1 aliases for existing artifact consumers.
            "grid_peclet_supported": pe_within_warning_threshold,
            f"{ratio_symbol.lower()}_supported": ratio_within_warning_threshold,
            "supported_accuracy": supported_accuracy,
        },
    }
