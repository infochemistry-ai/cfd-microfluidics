"""Verify the tetra flow solver against circular-pipe Poiseuille flow.

The harness intentionally runs the existing solver without modifying it.  It imports
the tracked ``gmsh/vertical_pipe_*.msh`` meshes directly, runs a low-Re Stokes case,
and compares pressure gradient, velocity profile, and wall shear in an interior
region.  Sampling away from the inlet and pressure outlet prevents entrance and
outlet-cell artefacts from dominating the verification metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for _path in (PROJECT_ROOT, COMPUTE_SRC):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from microfluidics.gmsh.gmsh_mesh_import import (  # noqa: E402
    import_gmsh_tetra_mesh,
)
from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (  # noqa: E402
    TetraFlowConfig,
    apply_tetra_stokes_viscous_predictor,
    initialize_tetra_flow_state,
    solve_tetra_pressure_projection,
)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a finite weighted mean and reject malformed/empty samples."""

    value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value_arr.shape != weight_arr.shape:
        raise ValueError("values and weights must have identical shapes")
    finite = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr > 0.0)
    if not np.any(finite):
        raise ValueError("weighted mean requires at least one positive finite weight")
    value_arr = value_arr[finite]
    weight_arr = weight_arr[finite]
    return float(np.sum(value_arr * weight_arr) / np.sum(weight_arr))


def weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Fit ``y = intercept + slope*x`` and report a weighted :math:`R^2`."""

    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x_arr.shape != y_arr.shape or x_arr.shape != w_arr.shape:
        raise ValueError("x, y, and weights must have identical shapes")
    finite = (
        np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(w_arr) & (w_arr > 0.0)
    )
    if np.count_nonzero(finite) < 2:
        raise ValueError("linear fit requires at least two finite weighted points")
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    w_arr = w_arr[finite]
    x_mean = weighted_mean(x_arr, w_arr)
    y_mean = weighted_mean(y_arr, w_arr)
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    denominator = float(np.sum(w_arr * dx * dx))
    if denominator <= 0.0:
        raise ValueError("linear fit requires non-constant x")
    slope = float(np.sum(w_arr * dx * dy) / denominator)
    intercept = float(y_mean - slope * x_mean)
    residual = y_arr - (intercept + slope * x_arr)
    ss_res = float(np.sum(w_arr * residual * residual))
    ss_tot = float(np.sum(w_arr * dy * dy))
    r_squared = (
        1.0
        if ss_tot <= 1e-300 and ss_res <= 1e-300
        else 1.0 - ss_res / max(ss_tot, 1e-300)
    )
    result = {
        "slope": slope,
        "intercept": intercept,
        "r_squared": float(r_squared),
    }
    return result


def poiseuille_reference(
    *,
    radius: float,
    length: float,
    mean_speed: float,
    density: float,
    kinematic_viscosity: float,
) -> dict[str, float]:
    """Return the fully developed circular-pipe Poiseuille reference values."""

    if radius <= 0.0 or length <= 0.0:
        raise ValueError("radius and length must be positive")
    if mean_speed < 0.0:
        raise ValueError("mean_speed must be non-negative")
    if density <= 0.0 or kinematic_viscosity <= 0.0:
        raise ValueError("density and kinematic_viscosity must be positive")
    dynamic_viscosity = float(density * kinematic_viscosity)
    diameter = float(2.0 * radius)
    pressure_gradient = float(8.0 * dynamic_viscosity * mean_speed / radius**2)
    pressure_drop = float(pressure_gradient * length)
    wall_shear = float(4.0 * dynamic_viscosity * mean_speed / radius)
    flow_rate = float(math.pi * radius**2 * mean_speed)
    reynolds = float(mean_speed * diameter / kinematic_viscosity)
    return {
        "radius": float(radius),
        "diameter": diameter,
        "length": float(length),
        "mean_speed": float(mean_speed),
        "density": float(density),
        "kinematic_viscosity": float(kinematic_viscosity),
        "dynamic_viscosity": dynamic_viscosity,
        "reynolds_number": reynolds,
        "volumetric_flow_rate": flow_rate,
        "pressure_gradient_magnitude": pressure_gradient,
        "full_length_pressure_drop": pressure_drop,
        "wall_shear_stress_magnitude": wall_shear,
        "centerline_speed": float(2.0 * mean_speed),
    }


def poiseuille_profile(
    radius_coordinate: np.ndarray,
    *,
    radius: float,
    mean_speed: float,
) -> np.ndarray:
    """Evaluate the axial Poiseuille profile at arbitrary radial coordinates."""

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    radial = np.asarray(radius_coordinate, dtype=np.float64)
    normalized = np.clip(radial / float(radius), 0.0, 1.0)
    return 2.0 * float(mean_speed) * (1.0 - normalized * normalized)


def weighted_relative_l2(
    actual: np.ndarray,
    expected: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return ``||actual-expected||_W / ||expected||_W``."""

    actual_arr = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected_arr = np.asarray(expected, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if actual_arr.shape != expected_arr.shape or actual_arr.shape != weight_arr.shape:
        raise ValueError("actual, expected, and weights must have identical shapes")
    finite = (
        np.isfinite(actual_arr)
        & np.isfinite(expected_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
    )
    if not np.any(finite):
        raise ValueError("relative L2 requires finite values with positive weights")
    delta = actual_arr[finite] - expected_arr[finite]
    numerator = float(np.sum(weight_arr[finite] * delta * delta))
    denominator = float(
        np.sum(weight_arr[finite] * expected_arr[finite] * expected_arr[finite])
    )
    return float(math.sqrt(numerator / max(denominator, 1e-300)))


def pressure_interior_metrics(
    axial_coordinate: np.ndarray,
    pressure: np.ndarray,
    weights: np.ndarray,
    *,
    length: float,
    plane_fractions: tuple[float, float] = (0.35, 0.65),
    plane_half_width_fraction: float = 0.035,
    fit_fraction: tuple[float, float] = (0.25, 0.75),
) -> dict[str, float | int]:
    """Measure pressure drop/gradient using only the pipe interior."""

    axial = np.asarray(axial_coordinate, dtype=np.float64).reshape(-1)
    pressure_arr = np.asarray(pressure, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if axial.shape != pressure_arr.shape or axial.shape != weight_arr.shape:
        raise ValueError("axial_coordinate, pressure, and weights must match")
    if length <= 0.0:
        raise ValueError("length must be positive")
    f0, f1 = (float(plane_fractions[0]), float(plane_fractions[1]))
    fit0, fit1 = (float(fit_fraction[0]), float(fit_fraction[1]))
    if not (0.0 < f0 < f1 < 1.0):
        raise ValueError("plane fractions must satisfy 0 < first < second < 1")
    if not (0.0 <= fit0 < fit1 <= 1.0):
        raise ValueError("fit fractions must satisfy 0 <= first < second <= 1")
    half_width = float(plane_half_width_fraction) * float(length)
    if half_width <= 0.0:
        raise ValueError("plane_half_width_fraction must be positive")

    def _plane_mask(target: float) -> np.ndarray:
        selected = np.abs(axial - target) <= half_width
        if np.count_nonzero(selected) >= 2:
            return selected
        # Very coarse meshes still get a deterministic plane estimate from the
        # nearest cells.  This path is reported through the selected-cell count.
        order = np.argsort(np.abs(axial - target))
        selected = np.zeros(axial.shape, dtype=bool)
        selected[order[: min(4, axial.size)]] = True
        return selected

    s0 = f0 * float(length)
    s1 = f1 * float(length)
    mask0 = _plane_mask(s0)
    mask1 = _plane_mask(s1)
    p0 = weighted_mean(pressure_arr[mask0], weight_arr[mask0])
    p1 = weighted_mean(pressure_arr[mask1], weight_arr[mask1])
    plane_gradient = float((p0 - p1) / (s1 - s0))
    fit_mask = (axial >= fit0 * length) & (axial <= fit1 * length)
    fit = weighted_linear_fit(
        axial[fit_mask], pressure_arr[fit_mask], weight_arr[fit_mask]
    )
    return {
        "upstream_plane_fraction": f0,
        "downstream_plane_fraction": f1,
        "plane_half_width_fraction": float(plane_half_width_fraction),
        "upstream_plane_cell_count": int(np.count_nonzero(mask0)),
        "downstream_plane_cell_count": int(np.count_nonzero(mask1)),
        "upstream_plane_pressure": p0,
        "downstream_plane_pressure": p1,
        "plane_pressure_drop": float(p0 - p1),
        "plane_pressure_gradient_magnitude": plane_gradient,
        "fit_start_fraction": fit0,
        "fit_end_fraction": fit1,
        "fit_cell_count": int(np.count_nonzero(fit_mask)),
        "fit_pressure_gradient_magnitude": float(-fit["slope"]),
        "fit_pressure_slope": float(fit["slope"]),
        "fit_pressure_intercept": float(fit["intercept"]),
        "fit_r_squared": float(fit["r_squared"]),
    }


def _area_weighted_face_center(mesh: Any, face_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(face_ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("pipe geometry requires non-empty inlet and outlet groups")
    weights = np.asarray(mesh.face_areas[ids], dtype=np.float64)
    return np.sum(mesh.face_centers[ids] * weights[:, None], axis=0) / np.sum(weights)


def infer_pipe_geometry(
    mesh: Any, *, radius_override: float | None = None
) -> dict[str, Any]:
    """Infer pipe axis, length, and radius from inlet/outlet boundary groups."""

    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    inlet_center = _area_weighted_face_center(mesh, inlet_faces)
    outlet_center = _area_weighted_face_center(mesh, outlet_faces)
    axis_delta = outlet_center - inlet_center
    length = float(np.linalg.norm(axis_delta))
    if length <= 0.0:
        raise ValueError("inlet and outlet centers do not define a pipe axis")
    axis = axis_delta / length
    inlet_area = float(
        np.sum(np.asarray(mesh.face_areas[inlet_faces], dtype=np.float64))
    )
    outlet_area = float(
        np.sum(np.asarray(mesh.face_areas[outlet_faces], dtype=np.float64))
    )
    inlet_vertices = np.unique(
        np.asarray(mesh.face_vertices[inlet_faces], dtype=np.int64)
    )
    vertex_delta = (
        np.asarray(mesh.points[inlet_vertices], dtype=np.float64) - inlet_center
    )
    vertex_axial = vertex_delta @ axis
    vertex_radial = vertex_delta - vertex_axial[:, None] * axis[None, :]
    radius_vertex_max = float(np.max(np.linalg.norm(vertex_radial, axis=1)))
    radius_from_area = float(math.sqrt(inlet_area / math.pi))
    mesh_volume = float(np.sum(np.asarray(mesh.cell_volumes, dtype=np.float64)))
    radius_from_volume = float(math.sqrt(mesh_volume / (math.pi * length)))
    nominal_radius = (
        float(radius_override) if radius_override is not None else radius_vertex_max
    )
    if nominal_radius <= 0.0:
        raise ValueError("inferred/overridden radius must be positive")
    analytic_circle_area = float(math.pi * nominal_radius**2)
    return {
        "inlet_center": inlet_center,
        "outlet_center": outlet_center,
        "axis": axis,
        "length": length,
        # ``radius`` is retained as a compatibility alias for the nominal
        # (circumscribed/CLI) radius.  Metrics report this reference alongside
        # the equal-area radius of the polygonal inlet.
        "radius": nominal_radius,
        "nominal_radius": nominal_radius,
        "effective_area_radius": radius_from_area,
        "radius_source": "cli_override"
        if radius_override is not None
        else "inlet_vertex_max",
        "nominal_radius_source": "cli_override"
        if radius_override is not None
        else "inlet_vertex_max",
        "effective_area_radius_source": "sqrt_inlet_area_over_pi",
        "radius_inlet_vertex_max": radius_vertex_max,
        "radius_from_inlet_area": radius_from_area,
        "radius_from_mesh_volume": radius_from_volume,
        "inlet_area": inlet_area,
        "outlet_area": outlet_area,
        "analytic_circle_area": analytic_circle_area,
        "inlet_area_to_analytic_circle_ratio": float(inlet_area / analytic_circle_area),
        "mesh_volume": mesh_volume,
    }


def _cell_coordinates(
    mesh: Any, geometry: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    origin = np.asarray(geometry["inlet_center"], dtype=np.float64)
    axis = np.asarray(geometry["axis"], dtype=np.float64)
    delta = centers - origin[None, :]
    axial = delta @ axis
    radial_vector = delta - axial[:, None] * axis[None, :]
    radial = np.linalg.norm(radial_vector, axis=1)
    return axial, radial


def _profile_sample_mask(
    axial: np.ndarray,
    *,
    length: float,
    profile_fraction: float,
    plane_half_width_fraction: float,
) -> np.ndarray:
    axial_arr = np.asarray(axial, dtype=np.float64)
    target_s = float(profile_fraction) * float(length)
    profile_half_width = float(plane_half_width_fraction) * float(length)
    profile_mask = np.abs(axial_arr - target_s) <= profile_half_width
    if np.count_nonzero(profile_mask) < 4:
        order = np.argsort(np.abs(axial_arr - target_s))
        profile_mask = np.zeros(axial_arr.shape, dtype=bool)
        profile_mask[order[: min(12, axial_arr.size)]] = True
    return profile_mask


def _profile_reference_mismatch(
    mesh: Any,
    *,
    geometry: dict[str, Any],
    nominal_reference: dict[str, float],
    effective_area_reference: dict[str, float],
    plane_half_width_fraction: float,
    profile_fraction: float,
) -> dict[str, float]:
    """Quantify the geometric floor from comparing the two exact profiles."""

    axial, radial = _cell_coordinates(mesh, geometry)
    profile_mask = _profile_sample_mask(
        axial,
        length=float(geometry["length"]),
        profile_fraction=profile_fraction,
        plane_half_width_fraction=plane_half_width_fraction,
    )
    nominal_profile = poiseuille_profile(
        radial[profile_mask],
        radius=float(nominal_reference["radius"]),
        mean_speed=float(nominal_reference["mean_speed"]),
    )
    effective_profile = poiseuille_profile(
        radial[profile_mask],
        radius=float(effective_area_reference["radius"]),
        mean_speed=float(effective_area_reference["mean_speed"]),
    )
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    return {
        "effective_area_profile_vs_nominal_reference_relative_l2": (
            weighted_relative_l2(
                effective_profile,
                nominal_profile,
                volumes[profile_mask],
            )
        ),
        "effective_area_radius_to_nominal_radius": float(
            effective_area_reference["radius"] / nominal_reference["radius"]
        ),
    }


def _solution_metrics(
    mesh: Any,
    *,
    pressure: np.ndarray,
    cell_velocity: np.ndarray,
    geometry: dict[str, Any],
    reference: dict[str, float],
    plane_fractions: tuple[float, float],
    plane_half_width_fraction: float,
    fit_fraction: tuple[float, float],
    profile_fraction: float,
) -> dict[str, Any]:
    length = float(geometry["length"])
    radius = float(geometry["radius"])
    axis = np.asarray(geometry["axis"], dtype=np.float64)
    axial, radial = _cell_coordinates(mesh, geometry)
    volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    velocity = np.asarray(cell_velocity, dtype=np.float64)
    axial_velocity = velocity @ axis
    transverse_velocity = velocity - axial_velocity[:, None] * axis[None, :]
    pressure_metrics = pressure_interior_metrics(
        axial,
        np.asarray(pressure, dtype=np.float64),
        volumes,
        length=length,
        plane_fractions=plane_fractions,
        plane_half_width_fraction=plane_half_width_fraction,
        fit_fraction=fit_fraction,
    )
    profile_mask = _profile_sample_mask(
        axial,
        length=length,
        profile_fraction=profile_fraction,
        plane_half_width_fraction=plane_half_width_fraction,
    )
    expected_profile = poiseuille_profile(
        radial[profile_mask], radius=radius, mean_speed=float(reference["mean_speed"])
    )
    profile_relative_l2 = weighted_relative_l2(
        axial_velocity[profile_mask], expected_profile, volumes[profile_mask]
    )
    transverse_l2 = math.sqrt(
        weighted_mean(
            np.sum(transverse_velocity[profile_mask] ** 2, axis=1),
            volumes[profile_mask],
        )
    )
    profile_axial_mean = weighted_mean(
        axial_velocity[profile_mask], volumes[profile_mask]
    )
    center_mask = profile_mask & (radial <= 0.25 * radius)
    if not np.any(center_mask):
        center_mask = profile_mask & (radial <= np.min(radial[profile_mask]) + 1e-15)
    centerline_estimate = weighted_mean(
        axial_velocity[center_mask], volumes[center_mask]
    )

    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    wall_s = (
        np.asarray(mesh.face_centers[wall_faces]) - geometry["inlet_center"]
    ) @ axis
    wall_mask = (wall_s >= float(fit_fraction[0]) * length) & (
        wall_s <= float(fit_fraction[1]) * length
    )
    interior_wall_faces = wall_faces[wall_mask]
    wall_shear: dict[str, Any]
    if interior_wall_faces.size:
        owner = np.asarray(mesh.face_to_cells[interior_wall_faces, 0], dtype=np.int64)
        normal = np.asarray(mesh.face_normals[interior_wall_faces], dtype=np.float64)
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
        delta_to_face = np.asarray(
            mesh.face_centers[interior_wall_faces], dtype=np.float64
        ) - np.asarray(mesh.cell_centers[owner], dtype=np.float64)
        normal_distance = np.abs(np.einsum("ij,ij->i", delta_to_face, normal))
        owner_axial_speed = np.abs(axial_velocity[owner])
        shear_samples = (
            float(reference["dynamic_viscosity"])
            * owner_axial_speed
            / np.maximum(normal_distance, 1e-30)
        )
        wall_area = np.asarray(mesh.face_areas[interior_wall_faces], dtype=np.float64)
        numerical_wall_shear = weighted_mean(shear_samples, wall_area)
        analytic_owner_speed = poiseuille_profile(
            radial[owner],
            radius=radius,
            mean_speed=float(reference["mean_speed"]),
        )
        analytic_estimator_samples = (
            float(reference["dynamic_viscosity"])
            * analytic_owner_speed
            / np.maximum(normal_distance, 1e-30)
        )
        analytic_estimator_baseline = weighted_mean(
            analytic_estimator_samples, wall_area
        )
        analytic_wall_shear = float(reference["wall_shear_stress_magnitude"])
        wall_shear = {
            "available": True,
            "method": "wall_owner_linear_normal_gradient",
            "face_count": int(interior_wall_faces.size),
            "numerical_area_weighted_mean": numerical_wall_shear,
            "analytic": analytic_wall_shear,
            "relative_error": float(
                numerical_wall_shear / max(analytic_wall_shear, 1e-300) - 1.0
            ),
            "analytic_profile_estimator_baseline": analytic_estimator_baseline,
            "analytic_profile_estimator_relative_bias": float(
                analytic_estimator_baseline / max(analytic_wall_shear, 1e-300) - 1.0
            ),
            "numerical_relative_error_vs_estimator_baseline": float(
                numerical_wall_shear / max(analytic_estimator_baseline, 1e-300) - 1.0
            ),
            "normal_distance_mean": weighted_mean(normal_distance, wall_area),
        }
    else:
        wall_shear = {
            "available": False,
            "method": "wall_owner_linear_normal_gradient",
            "face_count": 0,
        }

    analytic_gradient = float(reference["pressure_gradient_magnitude"])
    measured_gradient = float(pressure_metrics["fit_pressure_gradient_magnitude"])
    plane_spacing = float(plane_fractions[1] - plane_fractions[0]) * length
    analytic_plane_drop = analytic_gradient * plane_spacing
    measured_plane_drop = float(pressure_metrics["plane_pressure_drop"])
    return {
        "pressure": pressure_metrics
        | {
            "analytic_pressure_gradient_magnitude": analytic_gradient,
            "fit_pressure_gradient_relative_error": float(
                measured_gradient / max(analytic_gradient, 1e-300) - 1.0
            ),
            "analytic_plane_pressure_drop": analytic_plane_drop,
            "plane_pressure_drop_relative_error": float(
                measured_plane_drop / max(analytic_plane_drop, 1e-300) - 1.0
            ),
        },
        "velocity_profile": {
            "plane_fraction": float(profile_fraction),
            "plane_half_width_fraction": float(plane_half_width_fraction),
            "cell_count": int(np.count_nonzero(profile_mask)),
            "relative_l2_error": float(profile_relative_l2),
            "numerical_axial_mean": float(profile_axial_mean),
            "analytic_mean": float(reference["mean_speed"]),
            "mean_relative_error": float(
                profile_axial_mean / max(float(reference["mean_speed"]), 1e-300) - 1.0
            ),
            "numerical_centerline_estimate": float(centerline_estimate),
            "analytic_centerline": float(reference["centerline_speed"]),
            "centerline_relative_error": float(
                centerline_estimate / max(float(reference["centerline_speed"]), 1e-300)
                - 1.0
            ),
            "transverse_velocity_l2": float(transverse_l2),
            "transverse_to_mean_speed": float(
                transverse_l2 / max(float(reference["mean_speed"]), 1e-300)
            ),
        },
        "wall_shear": wall_shear,
    }


def _solution_metrics_by_radius(
    mesh: Any,
    *,
    pressure: np.ndarray,
    cell_velocity: np.ndarray,
    geometry: dict[str, Any],
    references: dict[str, dict[str, float]],
    plane_fractions: tuple[float, float],
    plane_half_width_fraction: float,
    fit_fraction: tuple[float, float],
    profile_fraction: float,
) -> dict[str, dict[str, Any]]:
    return {
        name: _solution_metrics(
            mesh,
            pressure=pressure,
            cell_velocity=cell_velocity,
            geometry=geometry | {"radius": float(reference["radius"])},
            reference=reference,
            plane_fractions=plane_fractions,
            plane_half_width_fraction=plane_half_width_fraction,
            fit_fraction=fit_fraction,
            profile_fraction=profile_fraction,
        )
        for name, reference in references.items()
    }


def _tail_stationarity(
    history: Sequence[dict[str, float | int]],
    *,
    window: int,
    relative_tolerance: float,
) -> dict[str, Any]:
    tail = list(history[-max(int(window), 2) :])
    if len(tail) < max(int(window), 2):
        return {
            "converged": False,
            "reason": "insufficient_samples",
            "available_samples": len(tail),
            "required_samples": max(int(window), 2),
        }
    gradient = np.asarray([float(row["pressure_gradient_magnitude"]) for row in tail])
    profile_error = np.asarray(
        [float(row["profile_relative_l2_error"]) for row in tail]
    )
    gradient_scale = max(abs(float(np.mean(gradient))), 1e-300)
    gradient_relative_range = float(
        (np.max(gradient) - np.min(gradient)) / gradient_scale
    )
    gradient_relative_drift = float(abs(gradient[-1] - gradient[0]) / gradient_scale)
    profile_scale = max(abs(float(np.mean(profile_error))), 1e-12)
    profile_relative_drift = float(
        abs(profile_error[-1] - profile_error[0]) / profile_scale
    )
    converged = bool(
        gradient_relative_range <= relative_tolerance
        and gradient_relative_drift <= relative_tolerance
        and profile_relative_drift <= relative_tolerance
    )
    return {
        "converged": converged,
        "reason": "tail_stable" if converged else "tail_still_changing",
        "sample_count": len(tail),
        "relative_tolerance": float(relative_tolerance),
        "gradient_relative_range": gradient_relative_range,
        "gradient_relative_drift": gradient_relative_drift,
        "profile_error_relative_drift": profile_relative_drift,
    }


def _flow_config(
    *,
    outlet_contract: str,
    projection_cell_velocity_update_mode: str,
    pressure_nonorthogonal_correction_mode: str,
    viscous_nonorthogonal_correction_mode: str,
    pressure_nonorthogonal_correction_sweeps: int,
    pressure_nonorthogonal_correction_relaxation: float,
    inlet_speed: float,
    density: float,
    kinematic_viscosity: float,
    flow_dt: float,
    max_pressure_iterations: int,
    pressure_relative_tolerance: float,
) -> tuple[TetraFlowConfig, list[str]]:
    available = {field.name for field in fields(TetraFlowConfig)}
    kwargs: dict[str, Any] = {
        "backend": "numpy",
        "device": "cpu",
        "inlet_speed": float(inlet_speed),
        "density": float(density),
        "kinematic_viscosity": float(kinematic_viscosity),
        "projection_dt": float(flow_dt),
        "wall_velocity_boundary_mode": "no_slip",
        "wall_tangential_no_slip_strength": 1.0,
        "wall_tangential_shear_face_flux_enabled": True,
        "wall_tangential_cell_velocity_momentum_enabled": True,
        "wall_flux_stokes_resistance_enabled": False,
        "viscous_predictor_mode": (
            "explicit_cell_velocity_laplacian_substepped_conservative"
        ),
        "viscous_face_flux_divergence_impact_cap": 0.03,
        "viscous_face_flux_laplacian_vectorized": True,
        "enable_convective_predictor": False,
        "disable_convective_predictor": True,
        "pressure_solver": "pcg_diag",
        "max_pressure_iterations": int(max_pressure_iterations),
        "pressure_relative_tolerance": float(pressure_relative_tolerance),
        "projection_rhs_mode": "volume_integrated_flux",
        "projection_correction_limit_mode": "none",
        "enable_sign_comparison": False,
        "outlet_projection_mode": "outlet_pressure_dirichlet",
    }
    applied_fields: list[str] = []
    for name in (
        "viscous_predictor_outlet_contract_mode",
        "pressure_projection_outlet_contract_mode",
    ):
        if name in available:
            kwargs[name] = str(outlet_contract)
            applied_fields.append(name)
    if outlet_contract != "match_inlet" and not applied_fields:
        raise RuntimeError(
            "outlet-contract A/B requested, but TetraFlowConfig exposes no contract field"
        )
    optional_modes = {
        "projection_cell_velocity_update_mode": str(
            projection_cell_velocity_update_mode
        ),
        "pressure_nonorthogonal_correction_mode": str(
            pressure_nonorthogonal_correction_mode
        ),
        "viscous_nonorthogonal_correction_mode": str(
            viscous_nonorthogonal_correction_mode
        ),
        "pressure_nonorthogonal_correction_sweeps": int(
            pressure_nonorthogonal_correction_sweeps
        ),
        "pressure_nonorthogonal_correction_relaxation": float(
            pressure_nonorthogonal_correction_relaxation
        ),
    }
    missing = [name for name in optional_modes if name not in available]
    if missing:
        requested_nonlegacy = bool(
            projection_cell_velocity_update_mode != "legacy_reconstruct"
            or pressure_nonorthogonal_correction_mode != "none"
            or viscous_nonorthogonal_correction_mode != "none"
        )
        if requested_nonlegacy:
            raise RuntimeError(
                "requested projection verification modes are unavailable: "
                + ", ".join(missing)
            )
    for name, value in optional_modes.items():
        if name in available:
            kwargs[name] = value
    return TetraFlowConfig(**kwargs), applied_fields


def _mass_balance_metrics(mesh: Any, state: Any) -> dict[str, float]:
    projection = dict(state.diagnostics.get("projection", {}))
    inlet = float(projection.get("inlet_flux_total_after", 0.0) or 0.0)
    outlet = float(projection.get("outlet_flux_total_after", 0.0) or 0.0)
    net = float(projection.get("net_boundary_flux_after", 0.0) or 0.0)
    if inlet == 0.0:
        inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
        outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
        flux = np.asarray(state.face_flux, dtype=np.float64)
        inlet = float(np.sum(np.maximum(-flux[inlet_faces], 0.0)))
        outlet = float(np.sum(np.maximum(flux[outlet_faces], 0.0)))
        net = float(
            np.sum(flux[np.asarray(mesh.boundary_face_indices, dtype=np.int64)])
        )
    scale = max(abs(inlet), 1e-300)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    wall_max = (
        float(np.max(np.abs(np.asarray(state.face_flux)[wall_faces])))
        if wall_faces.size
        else 0.0
    )
    return {
        "inlet_flux": inlet,
        "outlet_flux": outlet,
        "outlet_inlet_ratio": float(outlet / scale),
        "net_boundary_flux": net,
        "net_boundary_flux_relative": float(net / scale),
        "wall_flux_max_abs": wall_max,
    }


def run_case(
    mesh_path: Path,
    *,
    outlet_contract: str,
    projection_cell_velocity_update_mode: str,
    pressure_nonorthogonal_correction_mode: str,
    viscous_nonorthogonal_correction_mode: str,
    pressure_nonorthogonal_correction_sweeps: int,
    pressure_nonorthogonal_correction_relaxation: float,
    steps: int,
    flow_dt: float,
    inlet_speed: float,
    density: float,
    kinematic_viscosity: float,
    radius_override: float | None,
    sample_every: int,
    steady_window: int,
    steady_relative_tolerance: float,
    minimum_diffusive_time: float,
    stop_when_steady: bool,
    min_steps: int,
    max_pressure_iterations: int,
    pressure_relative_tolerance: float,
    plane_fractions: tuple[float, float],
    plane_half_width_fraction: float,
    fit_fraction: tuple[float, float],
    profile_fraction: float,
) -> dict[str, Any]:
    started = perf_counter()
    mesh = import_gmsh_tetra_mesh(mesh_path)
    geometry = infer_pipe_geometry(mesh, radius_override=radius_override)
    nominal_reference = poiseuille_reference(
        radius=float(geometry["radius"]),
        length=float(geometry["length"]),
        mean_speed=float(inlet_speed),
        density=float(density),
        kinematic_viscosity=float(kinematic_viscosity),
    )
    effective_area_reference = poiseuille_reference(
        radius=float(geometry["effective_area_radius"]),
        length=float(geometry["length"]),
        mean_speed=float(inlet_speed),
        density=float(density),
        kinematic_viscosity=float(kinematic_viscosity),
    )
    references = {
        "nominal": nominal_reference,
        "effective_area": effective_area_reference,
    }
    radius_normalization_comparison = _profile_reference_mismatch(
        mesh,
        geometry=geometry,
        nominal_reference=nominal_reference,
        effective_area_reference=effective_area_reference,
        plane_half_width_fraction=plane_half_width_fraction,
        profile_fraction=profile_fraction,
    )
    config, contract_fields = _flow_config(
        outlet_contract=outlet_contract,
        projection_cell_velocity_update_mode=projection_cell_velocity_update_mode,
        pressure_nonorthogonal_correction_mode=(pressure_nonorthogonal_correction_mode),
        viscous_nonorthogonal_correction_mode=(viscous_nonorthogonal_correction_mode),
        pressure_nonorthogonal_correction_sweeps=(
            pressure_nonorthogonal_correction_sweeps
        ),
        pressure_nonorthogonal_correction_relaxation=(
            pressure_nonorthogonal_correction_relaxation
        ),
        inlet_speed=inlet_speed,
        density=density,
        kinematic_viscosity=kinematic_viscosity,
        flow_dt=flow_dt,
        max_pressure_iterations=max_pressure_iterations,
        pressure_relative_tolerance=pressure_relative_tolerance,
    )
    state = initialize_tetra_flow_state(mesh, config)
    samples: list[dict[str, float | int]] = []
    completed_steps = 0
    final_metrics: dict[str, Any] = {}
    final_metrics_by_radius: dict[str, dict[str, Any]] = {}
    stationarity: dict[str, Any] = {
        "converged": False,
        "reason": "not_sampled",
    }
    runtime_contract = {
        "state_arrays_all_finite_all_steps": True,
        "projection_reduced_divergence_l2_every_step": True,
        "outlet_flux_rescale_used_any_step": False,
        "nonphysical_flux_fix_used_any_step": False,
        "face_flux_update_identity_max_abs_all_steps": 0.0,
        "returned_face_flux_identity_max_abs_all_steps": 0.0,
    }
    pressure_cost = {
        "solve_count": 0,
        "iterations": 0,
        "solve_wall_seconds": 0.0,
    }
    for step in range(1, int(steps) + 1):
        state = apply_tetra_stokes_viscous_predictor(
            mesh, state, config, flow_dt=float(flow_dt)
        )
        viscous_diag = dict(state.diagnostics.get("viscous_predictor", {}))
        state = solve_tetra_pressure_projection(mesh, state, config)
        state.diagnostics["viscous_predictor"] = viscous_diag
        step_pressure = dict(state.diagnostics.get("pressure", {}))
        step_nonorthogonal = dict(
            state.diagnostics.get("pressure_nonorthogonal_correction", {})
        )
        step_solve_count = int(
            step_nonorthogonal.get(
                "pressure_solve_count",
                1,
            )
        )
        pressure_cost["solve_count"] += step_solve_count
        pressure_cost["iterations"] += int(
            step_pressure.get(
                "nonorthogonal_total_actual_iterations",
                step_pressure.get("actual_iterations", 0),
            )
        )
        pressure_cost["solve_wall_seconds"] += float(
            step_pressure.get(
                "nonorthogonal_total_solve_wall_seconds",
                step_pressure.get("solve_wall_seconds", 0.0),
            )
        )
        completed_steps = step
        finite_arrays = all(
            np.all(np.isfinite(values))
            for values in (state.pressure, state.cell_velocity, state.face_flux)
        )
        runtime_contract["state_arrays_all_finite_all_steps"] = bool(
            runtime_contract["state_arrays_all_finite_all_steps"] and finite_arrays
        )
        step_projection = dict(state.diagnostics.get("projection", {}))
        initial_divergence_l2 = float(
            step_projection.get("initial_divergence_l2", float("inf"))
        )
        final_divergence_l2 = float(
            step_projection.get("final_divergence_l2", float("inf"))
        )
        runtime_contract["projection_reduced_divergence_l2_every_step"] = bool(
            runtime_contract["projection_reduced_divergence_l2_every_step"]
            and math.isfinite(initial_divergence_l2)
            and math.isfinite(final_divergence_l2)
            and final_divergence_l2 < initial_divergence_l2
        )
        runtime_contract["outlet_flux_rescale_used_any_step"] = bool(
            runtime_contract["outlet_flux_rescale_used_any_step"]
            or step_projection.get("outlet_flux_rescale_used", False)
        )
        runtime_contract["nonphysical_flux_fix_used_any_step"] = bool(
            runtime_contract["nonphysical_flux_fix_used_any_step"]
            or step_projection.get("nonphysical_flux_fix_used", False)
        )
        primary_flux = dict(state.diagnostics.get("face_flux_primary", {}))
        face_flux_star = np.asarray(
            primary_flux.get("face_flux_star", np.zeros((0,), dtype=np.float64)),
            dtype=np.float64,
        )
        correction_flux = np.asarray(
            primary_flux.get("correction_flux", np.zeros((0,), dtype=np.float64)),
            dtype=np.float64,
        )
        face_flux_corrected = np.asarray(
            primary_flux.get("face_flux_corrected", np.zeros((0,), dtype=np.float64)),
            dtype=np.float64,
        )
        if (
            face_flux_star.shape == state.face_flux.shape
            and correction_flux.shape == state.face_flux.shape
            and face_flux_corrected.shape == state.face_flux.shape
        ):
            update_identity_error = float(
                np.max(np.abs(face_flux_corrected - (face_flux_star + correction_flux)))
            )
            returned_flux_error = float(
                np.max(
                    np.abs(
                        np.asarray(state.face_flux, dtype=np.float64)
                        - face_flux_corrected
                    )
                )
            )
        else:
            update_identity_error = float("inf")
            returned_flux_error = float("inf")
        runtime_contract["face_flux_update_identity_max_abs_all_steps"] = max(
            float(runtime_contract["face_flux_update_identity_max_abs_all_steps"]),
            update_identity_error,
        )
        runtime_contract["returned_face_flux_identity_max_abs_all_steps"] = max(
            float(runtime_contract["returned_face_flux_identity_max_abs_all_steps"]),
            returned_flux_error,
        )
        pressure_scale = max(
            abs(float(nominal_reference["full_length_pressure_drop"])), 1e-12
        )
        speed_scale = max(abs(float(inlet_speed)), 1e-12)
        pressure_max_abs = float(np.max(np.abs(state.pressure)))
        speed_max = float(
            np.max(np.linalg.norm(np.asarray(state.cell_velocity), axis=1))
        )
        if (
            not finite_arrays
            or pressure_max_abs > 1e6 * pressure_scale
            or speed_max > 1e6 * speed_scale
        ):
            raise FloatingPointError(
                "flow verification diverged at step "
                f"{step}: pressure_max_abs={pressure_max_abs:.6g}, "
                f"speed_max={speed_max:.6g}"
            )
        should_sample = (
            step == 1 or step == steps or step % max(int(sample_every), 1) == 0
        )
        if should_sample:
            final_metrics_by_radius = _solution_metrics_by_radius(
                mesh,
                pressure=state.pressure,
                cell_velocity=state.cell_velocity,
                geometry=geometry,
                references=references,
                plane_fractions=plane_fractions,
                plane_half_width_fraction=plane_half_width_fraction,
                fit_fraction=fit_fraction,
                profile_fraction=profile_fraction,
            )
            final_metrics = final_metrics_by_radius["nominal"]
            samples.append(
                {
                    "step": int(step),
                    "physical_time": float(step * flow_dt),
                    "pressure_gradient_magnitude": float(
                        final_metrics["pressure"]["fit_pressure_gradient_magnitude"]
                    ),
                    "pressure_gradient_relative_error": float(
                        final_metrics["pressure"][
                            "fit_pressure_gradient_relative_error"
                        ]
                    ),
                    "profile_relative_l2_error": float(
                        final_metrics["velocity_profile"]["relative_l2_error"]
                    ),
                }
            )
            stationarity = _tail_stationarity(
                samples,
                window=steady_window,
                relative_tolerance=steady_relative_tolerance,
            )
            diffusive_time = float(
                step
                * flow_dt
                * kinematic_viscosity
                / max(float(geometry["radius"]) ** 2, 1e-300)
            )
            stationarity["elapsed_diffusive_time"] = diffusive_time
            stationarity["minimum_diffusive_time"] = float(minimum_diffusive_time)
            if diffusive_time + 1e-12 < float(minimum_diffusive_time):
                stationarity["converged"] = False
                stationarity["reason"] = "insufficient_diffusive_time"
            if (
                stop_when_steady
                and step >= min_steps
                and bool(stationarity["converged"])
            ):
                break
    if not final_metrics or int(samples[-1]["step"]) != completed_steps:
        final_metrics_by_radius = _solution_metrics_by_radius(
            mesh,
            pressure=state.pressure,
            cell_velocity=state.cell_velocity,
            geometry=geometry,
            references=references,
            plane_fractions=plane_fractions,
            plane_half_width_fraction=plane_half_width_fraction,
            fit_fraction=fit_fraction,
            profile_fraction=profile_fraction,
        )
        final_metrics = final_metrics_by_radius["nominal"]
    pressure_diag = dict(state.diagnostics.get("pressure", {}))
    projection_diag = dict(state.diagnostics.get("projection", {}))
    viscous_diag = dict(state.diagnostics.get("viscous_predictor", {}))
    nonorthogonal_diag = dict(
        state.diagnostics.get("pressure_nonorthogonal_correction", {})
    )
    profile_resolution = dict(state.diagnostics.get("numerical_profile_resolution", {}))
    effective_profile = dict(profile_resolution.get("effective", {}))
    result = {
        "status": "completed",
        "mesh": {
            "path": str(mesh_path.resolve()),
            "name": mesh_path.name,
            "cells": int(mesh.tetrahedra.shape[0]),
            "faces": int(mesh.face_vertices.shape[0]),
            "inlet_faces": int(np.asarray(mesh.inlet_faces).size),
            "outlet_faces": int(np.asarray(mesh.outlet_faces).size),
            "wall_faces": int(np.asarray(mesh.wall_faces).size),
            "characteristic_cell_size": float(
                (float(geometry["mesh_volume"]) / mesh.tetrahedra.shape[0])
                ** (1.0 / 3.0)
            ),
        },
        "case": {
            "outlet_contract": str(outlet_contract),
            "contract_config_fields": contract_fields,
            "projection_cell_velocity_update_mode": str(
                projection_cell_velocity_update_mode
            ),
            "pressure_nonorthogonal_correction_mode": str(
                pressure_nonorthogonal_correction_mode
            ),
            "viscous_nonorthogonal_correction_mode": str(
                viscous_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_sweeps": int(
                pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                pressure_nonorthogonal_correction_relaxation
            ),
            "effective_numerical_profile": effective_profile,
            "requested_steps": int(steps),
            "completed_steps": int(completed_steps),
            "flow_dt": float(flow_dt),
            "physical_time": float(completed_steps * flow_dt),
            "equations": "unsteady_stokes_projection_no_convection",
            "backend": "numpy_cpu",
        },
        "geometry": geometry,
        # Compatibility aliases remain nominal; the explicitly named mappings
        # are authoritative for dual-normalization consumers.
        "analytic_reference": nominal_reference,
        "analytic_references": references,
        "metrics": final_metrics,
        "metrics_by_radius": final_metrics_by_radius,
        "radius_normalization_comparison": radius_normalization_comparison,
        "mass_balance": _mass_balance_metrics(mesh, state),
        "stationarity": stationarity,
        "history": samples,
        "solver_diagnostics": {
            "numerical_profile_resolution": profile_resolution,
            "pressure_stopping_reason": pressure_diag.get("stopping_reason"),
            "pressure_iterations": pressure_diag.get("actual_iterations"),
            "pressure_residual_ratio_to_rhs_l2": pressure_diag.get(
                "residual_ratio_to_rhs_l2"
            ),
            "pressure_nonorthogonal_total_iterations": pressure_diag.get(
                "nonorthogonal_total_actual_iterations"
            ),
            "pressure_nonorthogonal_outer_defect_relative_l2": (
                nonorthogonal_diag.get("outer_fixed_point_defect_relative_l2")
            ),
            "pressure_nonorthogonal_actual_sweeps": nonorthogonal_diag.get(
                "actual_sweeps"
            ),
            "pressure_solve_count_total": int(pressure_cost["solve_count"]),
            "pressure_iterations_total": int(pressure_cost["iterations"]),
            "pressure_solve_wall_seconds_total": float(
                pressure_cost["solve_wall_seconds"]
            ),
            "pressure_solve_wall_seconds_mean": float(
                pressure_cost["solve_wall_seconds"]
                / max(int(pressure_cost["solve_count"]), 1)
            ),
            "pressure_solve_wall_seconds_per_iteration": float(
                pressure_cost["solve_wall_seconds"]
                / max(int(pressure_cost["iterations"]), 1)
            ),
            "projection_final_divergence_l2": projection_diag.get(
                "final_divergence_l2"
            ),
            "projection_final_divergence_max_abs": projection_diag.get(
                "final_divergence_max_abs"
            ),
            "projection_initial_divergence_l2": projection_diag.get(
                "initial_divergence_l2"
            ),
            "projection_initial_divergence_max_abs": projection_diag.get(
                "initial_divergence_max_abs"
            ),
            "projection_divergence_reduction_ratio_l2": projection_diag.get(
                "divergence_reduction_ratio_l2"
            ),
            "projection_divergence_reduction_ratio_linf": projection_diag.get(
                "divergence_reduction_ratio"
            ),
            "outlet_projection_mode": projection_diag.get("outlet_projection_mode"),
            "pressure_projection_outlet_contract_mode": projection_diag.get(
                "pressure_projection_outlet_contract_mode"
            ),
            "outlet_flux_rescale_used": projection_diag.get("outlet_flux_rescale_used"),
            "nonphysical_flux_fix_used": projection_diag.get(
                "nonphysical_flux_fix_used"
            ),
            **runtime_contract,
            "viscous_substeps": viscous_diag.get("viscous_substeps"),
            "viscous_stability_metric": viscous_diag.get("viscous_stability_metric"),
            "viscous_nonorthogonal_stability_bound": viscous_diag.get(
                "viscous_nonorthogonal_stability_bound"
            ),
            "viscous_nonorthogonal_stability_bound_per_substep": viscous_diag.get(
                "viscous_nonorthogonal_stability_bound_per_substep"
            ),
            "viscous_nonorthogonal_update_l2": viscous_diag.get(
                "viscous_nonorthogonal_update_l2"
            ),
        },
        "elapsed_seconds": float(perf_counter() - started),
    }
    result["acceptance"] = evaluate_acceptance(result)
    return result


def evaluate_acceptance(result: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the discrete no-slip regression contract on a completed case.

    Analytic-error gates deliberately use the nominal-radius compatibility
    metrics.  Effective-area metrics and the first-order wall-shear estimate are
    reported as diagnostics rather than silently changing the acceptance basis.
    """

    normalized_metrics = dict(result.get("metrics_by_radius", {}))
    acceptance_metrics = dict(normalized_metrics.get("nominal", result["metrics"]))
    pressure_error = abs(
        float(acceptance_metrics["pressure"]["fit_pressure_gradient_relative_error"])
    )
    profile_error = float(acceptance_metrics["velocity_profile"]["relative_l2_error"])
    wall_shear = dict(acceptance_metrics.get("wall_shear", {}))
    wall_shear_error = abs(float(wall_shear.get("relative_error", float("inf"))))
    mass = dict(result["mass_balance"])
    solver = dict(result["solver_diagnostics"])
    outer_defect_raw = solver.get("pressure_nonorthogonal_outer_defect_relative_l2")
    outer_defect = (
        float(outer_defect_raw) if outer_defect_raw is not None else float("inf")
    )
    stopping_reason = str(solver.get("pressure_stopping_reason", ""))
    effective_profile = dict(
        result.get("case", {}).get("effective_numerical_profile", {})
    )
    expected_profile = {
        "viscous_predictor_outlet_contract_mode": "preserve",
        "pressure_projection_outlet_contract_mode": "preserve",
        "projection_cell_velocity_update_mode": "momentum_pressure_corrected",
        "pressure_nonorthogonal_correction_mode": "deferred_lsq",
        "viscous_nonorthogonal_correction_mode": "deferred_lsq",
    }
    thresholds = {
        "pressure_gradient_relative_error_max_abs": 0.10,
        "velocity_profile_relative_l2_max": 0.25,
        "outlet_inlet_ratio_error_max_abs": 0.005,
        "wall_flux_max_abs": 1e-14,
        "projection_divergence_l2_max": 1e-6,
        "face_flux_update_identity_max_abs": 1e-14,
        "pressure_nonorthogonal_outer_defect_relative_l2_max": 0.03,
    }
    checks = {
        "finite_state": bool(solver.get("state_arrays_all_finite_all_steps", False)),
        "effective_no_slip_profile": effective_profile == expected_profile,
        "stationary": bool(result["stationarity"].get("converged", False)),
        "pressure_gradient": pressure_error
        <= thresholds["pressure_gradient_relative_error_max_abs"],
        "velocity_profile": profile_error
        <= thresholds["velocity_profile_relative_l2_max"],
        "mass_balance": abs(float(mass["outlet_inlet_ratio"]) - 1.0)
        <= thresholds["outlet_inlet_ratio_error_max_abs"],
        "wall_impermeability": float(mass["wall_flux_max_abs"])
        <= thresholds["wall_flux_max_abs"],
        "projection_divergence": float(solver["projection_final_divergence_l2"])
        <= thresholds["projection_divergence_l2_max"],
        "projection_reduces_divergence": bool(
            solver.get("projection_reduced_divergence_l2_every_step", False)
        ),
        "no_outlet_flux_rescale": not bool(
            solver.get("outlet_flux_rescale_used_any_step", True)
        ),
        "no_nonphysical_flux_fix": not bool(
            solver.get("nonphysical_flux_fix_used_any_step", True)
        ),
        "face_flux_update_identity": float(
            solver.get("face_flux_update_identity_max_abs_all_steps", float("inf"))
        )
        <= thresholds["face_flux_update_identity_max_abs"],
        "no_post_projection_flux_add_back": float(
            solver.get("returned_face_flux_identity_max_abs_all_steps", float("inf"))
        )
        <= thresholds["face_flux_update_identity_max_abs"],
        "pressure_outer_fixed_point": outer_defect
        <= thresholds["pressure_nonorthogonal_outer_defect_relative_l2_max"],
        "pressure_linear_solve": stopping_reason.startswith("converged_"),
    }
    return {
        "passed": bool(all(checks.values())),
        "reference_normalization": "nominal",
        "checks": checks,
        "thresholds": thresholds,
        "threshold_basis": {
            "percentage_error_gates": (
                "empirical regression bounds selected from the validated mesh sweep; "
                "they are not independent accuracy guarantees"
            ),
            "conservation_gates": (
                "discrete conservation identities and numerical precision"
            ),
        },
        "diagnostics": {
            "wall_shear_available": bool(wall_shear.get("available", False)),
            "wall_shear_relative_error_abs": wall_shear_error,
            "wall_shear_status": (
                "non_gating_first_order_owner_cell_estimate"
                if wall_shear.get("available", False)
                else "unavailable"
            ),
        },
        "observed": {
            "pressure_gradient_relative_error_abs": pressure_error,
            "velocity_profile_relative_l2": profile_error,
            "wall_shear_relative_error_abs": wall_shear_error,
            "outlet_inlet_ratio_error_abs": abs(
                float(mass["outlet_inlet_ratio"]) - 1.0
            ),
            "wall_flux_max_abs": float(mass["wall_flux_max_abs"]),
            "projection_divergence_l2": float(solver["projection_final_divergence_l2"]),
            "projection_divergence_reduction_ratio_l2": float(
                solver.get("projection_divergence_reduction_ratio_l2", float("inf"))
            ),
            "state_arrays_all_finite_all_steps": bool(
                solver.get("state_arrays_all_finite_all_steps", False)
            ),
            "effective_numerical_profile": effective_profile,
            "outlet_flux_rescale_used_any_step": bool(
                solver.get("outlet_flux_rescale_used_any_step", True)
            ),
            "nonphysical_flux_fix_used_any_step": bool(
                solver.get("nonphysical_flux_fix_used_any_step", True)
            ),
            "face_flux_update_identity_max_abs_all_steps": float(
                solver.get("face_flux_update_identity_max_abs_all_steps", float("inf"))
            ),
            "returned_face_flux_identity_max_abs_all_steps": float(
                solver.get(
                    "returned_face_flux_identity_max_abs_all_steps", float("inf")
                )
            ),
            "pressure_nonorthogonal_outer_defect_relative_l2": outer_defect,
            "pressure_stopping_reason": stopping_reason,
        },
    }


def convergence_summary(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize error trends without mixing different numerical variants."""

    summaries: list[dict[str, Any]] = []
    variants = sorted(
        {
            (
                str(row["case"]["outlet_contract"]),
                str(row["case"]["projection_cell_velocity_update_mode"]),
                str(row["case"]["pressure_nonorthogonal_correction_mode"]),
                str(row["case"]["viscous_nonorthogonal_correction_mode"]),
            )
            for row in results
            if row.get("status") == "completed"
        }
    )
    for (
        contract,
        velocity_update_mode,
        nonorthogonal_mode,
        viscous_nonorthogonal_mode,
    ) in variants:
        rows = sorted(
            (
                row
                for row in results
                if row.get("status") == "completed"
                and str(row["case"]["outlet_contract"]) == contract
                and str(row["case"]["projection_cell_velocity_update_mode"])
                == velocity_update_mode
                and str(row["case"]["pressure_nonorthogonal_correction_mode"])
                == nonorthogonal_mode
                and str(row["case"]["viscous_nonorthogonal_correction_mode"])
                == viscous_nonorthogonal_mode
            ),
            key=lambda row: int(row["mesh"]["cells"]),
        )
        points = []
        for row in rows:
            normalized = dict(row.get("metrics_by_radius", {}))
            nominal_metrics = dict(normalized.get("nominal", row["metrics"]))
            effective_metrics = normalized.get("effective_area")
            points.append(
                {
                    "mesh": row["mesh"]["name"],
                    "cells": int(row["mesh"]["cells"]),
                    "h": float(row["mesh"]["characteristic_cell_size"]),
                    "pressure_gradient_absolute_relative_error": abs(
                        float(
                            nominal_metrics["pressure"][
                                "fit_pressure_gradient_relative_error"
                            ]
                        )
                    ),
                    "profile_relative_l2_error": float(
                        nominal_metrics["velocity_profile"]["relative_l2_error"]
                    ),
                    "normalizations": {
                        "nominal": {
                            "pressure_gradient_absolute_relative_error": abs(
                                float(
                                    nominal_metrics["pressure"][
                                        "fit_pressure_gradient_relative_error"
                                    ]
                                )
                            ),
                            "profile_relative_l2_error": float(
                                nominal_metrics["velocity_profile"]["relative_l2_error"]
                            ),
                        },
                        **(
                            {
                                "effective_area": {
                                    "pressure_gradient_absolute_relative_error": abs(
                                        float(
                                            effective_metrics["pressure"][
                                                "fit_pressure_gradient_relative_error"
                                            ]
                                        )
                                    ),
                                    "profile_relative_l2_error": float(
                                        effective_metrics["velocity_profile"][
                                            "relative_l2_error"
                                        ]
                                    ),
                                }
                            }
                            if effective_metrics is not None
                            else {}
                        ),
                    },
                    "profile_reference_mismatch_relative_l2": float(
                        row.get("radius_normalization_comparison", {}).get(
                            "effective_area_profile_vs_nominal_reference_relative_l2",
                            float("nan"),
                        )
                    ),
                    "stationary": bool(row["stationarity"].get("converged", False)),
                }
            )
        observed_orders: list[float] = []
        for coarse, fine in zip(points, points[1:]):
            e0 = float(coarse["pressure_gradient_absolute_relative_error"])
            e1 = float(fine["pressure_gradient_absolute_relative_error"])
            h0 = float(coarse["h"])
            h1 = float(fine["h"])
            if e0 > 0.0 and e1 > 0.0 and h0 > h1:
                observed_orders.append(float(math.log(e0 / e1) / math.log(h0 / h1)))
        summaries.append(
            {
                "outlet_contract": contract,
                "projection_cell_velocity_update_mode": velocity_update_mode,
                "pressure_nonorthogonal_correction_mode": nonorthogonal_mode,
                "viscous_nonorthogonal_correction_mode": (viscous_nonorthogonal_mode),
                "points": points,
                "pressure_gradient_observed_orders": observed_orders,
                "pressure_error_monotone_decreasing": bool(
                    len(points) >= 2
                    and all(
                        float(b["pressure_gradient_absolute_relative_error"])
                        <= float(a["pressure_gradient_absolute_relative_error"])
                        for a, b in zip(points, points[1:])
                    )
                ),
            }
        )
    return summaries


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Vertical-pipe Poiseuille verification",
        "",
        "The solver was run in unsteady Stokes mode (convection disabled). Pressure and velocity "
        "metrics use interior planes/slabs, not inlet- or outlet-adjacent cells.",
        "",
        "Analytic errors are shown for both the nominal circumscribed radius "
        "(`nom`) and the equal-area radius of the polygonal inlet (`eff`).",
        "",
        "| Mesh | Cells | Outlet | Velocity update | Pressure nonorth | Viscous nonorth | "
        "Rnom/Reff | dp/dz error nom/eff | Profile L2 nom/eff | Geometry floor | "
        "Wall shear error nom/eff | Qout/Qin | Stationary | Acceptance | Time, s |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|",
    ]
    for row in payload["results"]:
        if row.get("status") != "completed":
            lines.append(
                f"| {row.get('mesh_path', '?')} | - | {row.get('outlet_contract', '?')} | "
                f"{row.get('projection_cell_velocity_update_mode', '?')} | "
                f"{row.get('pressure_nonorthogonal_correction_mode', '?')} | "
                f"{row.get('viscous_nonorthogonal_correction_mode', '?')} | - | "
                f"ERROR: {row.get('error', 'unknown')} | - | - | - | - | no | - |"
            )
            continue
        normalized = dict(row.get("metrics_by_radius", {}))
        nominal_metrics = dict(normalized.get("nominal", row["metrics"]))
        effective_metrics = dict(normalized.get("effective_area", nominal_metrics))
        nominal_shear = nominal_metrics["wall_shear"]
        effective_shear = effective_metrics["wall_shear"]
        shear_error = " / ".join(
            (
                f"{100.0 * float(shear['relative_error']):+.2f}%"
                if shear.get("available")
                else "n/a"
            )
            for shear in (nominal_shear, effective_shear)
        )
        nominal_radius = float(
            row["geometry"].get("nominal_radius", row["geometry"]["radius"])
        )
        effective_radius = float(
            row["geometry"].get(
                "effective_area_radius", row["geometry"]["radius_from_inlet_area"]
            )
        )
        lines.append(
            "| {mesh} | {cells} | `{contract}` | `{velocity_update}` | `{nonorth}` | `{viscous_nonorth}` | "
            "{radius_ratio:.5f} | {pressure_nominal:+.2f}% / {pressure_effective:+.2f}% | "
            "{profile_nominal:.2f}% / {profile_effective:.2f}% | {geometry_floor:.2f}% | "
            "{shear} | {ratio:.6f} | {stationary} | {accepted} | {elapsed:.2f} |".format(
                mesh=row["mesh"]["name"],
                cells=int(row["mesh"]["cells"]),
                contract=row["case"]["outlet_contract"],
                velocity_update=row["case"]["projection_cell_velocity_update_mode"],
                nonorth=row["case"]["pressure_nonorthogonal_correction_mode"],
                viscous_nonorth=row["case"]["viscous_nonorthogonal_correction_mode"],
                radius_ratio=nominal_radius / effective_radius,
                pressure_nominal=100.0
                * float(
                    nominal_metrics["pressure"]["fit_pressure_gradient_relative_error"]
                ),
                pressure_effective=100.0
                * float(
                    effective_metrics["pressure"][
                        "fit_pressure_gradient_relative_error"
                    ]
                ),
                profile_nominal=100.0
                * float(nominal_metrics["velocity_profile"]["relative_l2_error"]),
                profile_effective=100.0
                * float(effective_metrics["velocity_profile"]["relative_l2_error"]),
                geometry_floor=100.0
                * float(
                    row.get("radius_normalization_comparison", {}).get(
                        "effective_area_profile_vs_nominal_reference_relative_l2",
                        0.0,
                    )
                ),
                shear=shear_error,
                ratio=float(row["mass_balance"]["outlet_inlet_ratio"]),
                stationary="yes" if row["stationarity"].get("converged") else "no",
                accepted="yes" if row["acceptance"]["passed"] else "no",
                elapsed=float(row["elapsed_seconds"]),
            )
        )
    lines.extend(["", "## Analytic contract", ""])
    cfg = payload["config"]
    lines.extend(
        [
            f"- Mean inlet speed: `{cfg['inlet_speed']:.9g} m/s`",
            f"- Density: `{cfg['density']:.9g} kg/m^3`",
            f"- Kinematic viscosity: `{cfg['kinematic_viscosity']:.9g} m^2/s`",
            "- `nom`: nominal circumscribed radius (or explicit `--radius` override); "
            "`eff`: `sqrt(inlet_area/pi)` for the polygonal inlet.",
            "- Geometry floor: weighted profile L2 obtained by treating the exact "
            "effective-area profile as numerical data and comparing it with the nominal "
            "reference on the same profile sample.",
            "- Circular-pipe laws: `|dp/dz| = 8 mu U / R^2`, "
            "`u(r) = 2 U (1-r^2/R^2)`, `|tau_w| = 4 mu U / R`.",
            f"- Pressure fit interval: `{cfg['fit_fraction'][0]:.3f}L.."
            f"{cfg['fit_fraction'][1]:.3f}L`.",
            f"- Pressure planes: `{cfg['plane_fractions'][0]:.3f}L` and "
            f"`{cfg['plane_fractions'][1]:.3f}L`.",
            "",
            "Acceptance uses the nominal-radius pressure/profile metrics. Its percentage-error "
            "bounds are empirical regression bounds from the validated mesh sweep, not "
            "independent accuracy guarantees. Wall shear is reported as a non-gating, "
            "first-order owner-cell diagnostic.",
            "",
            "## Mesh-convergence diagnostics",
            "",
        ]
    )
    for summary in payload["convergence"]:
        lines.append(
            "### outlet=`{}`, velocity=`{}`, pressure nonorth=`{}`, viscous nonorth=`{}`".format(
                summary["outlet_contract"],
                summary["projection_cell_velocity_update_mode"],
                summary["pressure_nonorthogonal_correction_mode"],
                summary["viscous_nonorthogonal_correction_mode"],
            )
        )
        lines.append("")
        lines.append(
            "Pressure error monotonically decreases: "
            + ("yes" if summary["pressure_error_monotone_decreasing"] else "no")
            + "."
        )
        orders = summary["pressure_gradient_observed_orders"]
        if orders:
            lines.append(
                "Observed adjacent-mesh pressure orders: "
                + ", ".join(f"{float(order):.3f}" for order in orders)
                + "."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "A fixed-step result is not a steady validation unless the Stationary column is `yes`. "
            "Use `--stop-when-steady` or increase `--steps` before drawing a mesh-convergence "
            "conclusion. The wall-shear estimate is first-order (owner-cell velocity divided by "
            "normal wall distance), so it is primarily a convergence diagnostic.",
            "",
        ]
    )
    return "\n".join(lines)


def _default_meshes() -> list[Path]:
    names = ["vertical_pipe_500.msh"]
    return [PROJECT_ROOT / "data" / "meshes" / "gmsh" / name for name in names]


def _parse_fraction_pair(raw: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(raw) != 2:
        raise ValueError(f"{name} requires exactly two values")
    return float(raw[0]), float(raw[1])


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        action="append",
        default=None,
        help="Gmsh .msh path; repeat for a mesh sweep. Defaults to the included 500-cell mesh.",
    )
    parser.add_argument(
        "--outlet-contracts",
        nargs="+",
        choices=("auto", "match_inlet", "preserve"),
        default=("auto",),
        help="Run the automatic physical profile or explicit outlet variants.",
    )
    parser.add_argument(
        "--projection-cell-velocity-update-modes",
        nargs="+",
        choices=("auto", "legacy_reconstruct", "momentum_pressure_corrected"),
        default=("auto",),
        help="Projection velocity-state variants; auto resolves from the no-slip wall.",
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-modes",
        nargs="+",
        choices=("auto", "none", "deferred_lsq"),
        default=("auto",),
        help="Pressure-gradient variants; auto selects deferred_lsq for no-slip.",
    )
    parser.add_argument(
        "--viscous-nonorthogonal-correction-modes",
        nargs="+",
        choices=("auto", "none", "deferred_lsq"),
        default=("auto",),
        help=(
            "Vector viscous-gradient variants; auto selects deferred_lsq for no-slip."
        ),
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-sweeps", type=int, default=4
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-relaxation", type=float, default=1.0
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--min-steps", type=int, default=200)
    parser.add_argument("--flow-dt", type=float, default=5e-4)
    parser.add_argument("--inlet-speed", type=float, default=5e-4)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--kinematic-viscosity", type=float, default=1e-6)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--steady-window", type=int, default=5)
    parser.add_argument("--steady-relative-tolerance", type=float, default=2e-3)
    parser.add_argument(
        "--minimum-diffusive-time",
        type=float,
        default=0.25,
        help="Require nu*t/R^2 to reach this value before declaring stationarity.",
    )
    parser.add_argument(
        "--stop-when-steady", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-acceptance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return a non-zero exit code when any completed case fails the physical contract.",
    )
    parser.add_argument("--max-pressure-iterations", type=int, default=1000)
    parser.add_argument("--pressure-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--plane-fractions", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--plane-half-width-fraction", type=float, default=0.035)
    parser.add_argument("--fit-fraction", type=float, nargs=2, default=(0.25, 0.75))
    parser.add_argument("--profile-fraction", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report.json and report.md.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.steps < 1 or args.min_steps < 1:
        raise SystemExit("--steps and --min-steps must be positive")
    if args.flow_dt <= 0.0 or args.inlet_speed <= 0.0:
        raise SystemExit("--flow-dt and --inlet-speed must be positive")
    if args.pressure_nonorthogonal_correction_sweeps < 1:
        raise SystemExit("--pressure-nonorthogonal-correction-sweeps must be positive")
    if not 0.0 < args.pressure_nonorthogonal_correction_relaxation <= 1.0:
        raise SystemExit(
            "--pressure-nonorthogonal-correction-relaxation must be in (0, 1]"
        )
    plane_fractions = _parse_fraction_pair(
        args.plane_fractions, name="--plane-fractions"
    )
    fit_fraction = _parse_fraction_pair(args.fit_fraction, name="--fit-fraction")
    meshes = list(args.mesh or _default_meshes())
    missing = [path for path in meshes if not path.exists()]
    if missing:
        raise SystemExit(
            "missing mesh files: " + ", ".join(str(path) for path in missing)
        )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            PROJECT_ROOT / "results" / "analysis" / f"poiseuille_verification_{stamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mesh_path in meshes:
        variants = product(
            args.outlet_contracts,
            args.projection_cell_velocity_update_modes,
            args.pressure_nonorthogonal_correction_modes,
            args.viscous_nonorthogonal_correction_modes,
        )
        for (
            outlet_contract,
            velocity_update_mode,
            nonorthogonal_mode,
            viscous_nonorthogonal_mode,
        ) in variants:
            print(
                f"[poiseuille] mesh={mesh_path.name} outlet_contract={outlet_contract} "
                f"velocity_update={velocity_update_mode} "
                f"pressure_nonorth={nonorthogonal_mode} "
                f"viscous_nonorth={viscous_nonorthogonal_mode} "
                f"steps={args.steps}",
                flush=True,
            )
            try:
                row = run_case(
                    mesh_path,
                    outlet_contract=str(outlet_contract),
                    projection_cell_velocity_update_mode=str(velocity_update_mode),
                    pressure_nonorthogonal_correction_mode=str(nonorthogonal_mode),
                    viscous_nonorthogonal_correction_mode=str(
                        viscous_nonorthogonal_mode
                    ),
                    pressure_nonorthogonal_correction_sweeps=int(
                        args.pressure_nonorthogonal_correction_sweeps
                    ),
                    pressure_nonorthogonal_correction_relaxation=float(
                        args.pressure_nonorthogonal_correction_relaxation
                    ),
                    steps=int(args.steps),
                    flow_dt=float(args.flow_dt),
                    inlet_speed=float(args.inlet_speed),
                    density=float(args.density),
                    kinematic_viscosity=float(args.kinematic_viscosity),
                    radius_override=args.radius,
                    sample_every=int(args.sample_every),
                    steady_window=int(args.steady_window),
                    steady_relative_tolerance=float(args.steady_relative_tolerance),
                    minimum_diffusive_time=float(args.minimum_diffusive_time),
                    stop_when_steady=bool(args.stop_when_steady),
                    min_steps=int(args.min_steps),
                    max_pressure_iterations=int(args.max_pressure_iterations),
                    pressure_relative_tolerance=float(args.pressure_relative_tolerance),
                    plane_fractions=plane_fractions,
                    plane_half_width_fraction=float(args.plane_half_width_fraction),
                    fit_fraction=fit_fraction,
                    profile_fraction=float(args.profile_fraction),
                )
                nominal_metrics = row["metrics_by_radius"]["nominal"]
                effective_metrics = row["metrics_by_radius"]["effective_area"]
                print(
                    f"[poiseuille] cells={row['mesh']['cells']} "
                    "dp/dz_error_nom/eff="
                    f"{100.0 * nominal_metrics['pressure']['fit_pressure_gradient_relative_error']:+.3f}%/"
                    f"{100.0 * effective_metrics['pressure']['fit_pressure_gradient_relative_error']:+.3f}% "
                    "profile_l2_nom/eff="
                    f"{100.0 * nominal_metrics['velocity_profile']['relative_l2_error']:.3f}%/"
                    f"{100.0 * effective_metrics['velocity_profile']['relative_l2_error']:.3f}% "
                    "geometry_floor="
                    f"{100.0 * row['radius_normalization_comparison']['effective_area_profile_vs_nominal_reference_relative_l2']:.3f}% "
                    f"stationary={row['stationarity'].get('converged', False)} "
                    f"elapsed={row['elapsed_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as exc:  # keep a multi-mesh sweep auditable
                row = {
                    "status": "error",
                    "mesh_path": str(mesh_path.resolve()),
                    "outlet_contract": str(outlet_contract),
                    "projection_cell_velocity_update_mode": str(velocity_update_mode),
                    "pressure_nonorthogonal_correction_mode": str(nonorthogonal_mode),
                    "viscous_nonorthogonal_correction_mode": str(
                        viscous_nonorthogonal_mode
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"[poiseuille] ERROR {type(exc).__name__}: {exc}", flush=True)
            results.append(row)
    payload = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "meshes": [str(path.resolve()) for path in meshes],
            "outlet_contracts": [str(value) for value in args.outlet_contracts],
            "projection_cell_velocity_update_modes": [
                str(value) for value in args.projection_cell_velocity_update_modes
            ],
            "pressure_nonorthogonal_correction_modes": [
                str(value) for value in args.pressure_nonorthogonal_correction_modes
            ],
            "viscous_nonorthogonal_correction_modes": [
                str(value) for value in args.viscous_nonorthogonal_correction_modes
            ],
            "pressure_nonorthogonal_correction_sweeps": int(
                args.pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                args.pressure_nonorthogonal_correction_relaxation
            ),
            "steps": int(args.steps),
            "min_steps": int(args.min_steps),
            "flow_dt": float(args.flow_dt),
            "inlet_speed": float(args.inlet_speed),
            "density": float(args.density),
            "kinematic_viscosity": float(args.kinematic_viscosity),
            "radius_override": args.radius,
            "sample_every": int(args.sample_every),
            "steady_window": int(args.steady_window),
            "steady_relative_tolerance": float(args.steady_relative_tolerance),
            "minimum_diffusive_time": float(args.minimum_diffusive_time),
            "stop_when_steady": bool(args.stop_when_steady),
            "require_acceptance": bool(args.require_acceptance),
            "plane_fractions": plane_fractions,
            "plane_half_width_fraction": float(args.plane_half_width_fraction),
            "fit_fraction": fit_fraction,
            "profile_fraction": float(args.profile_fraction),
            "stokes_mode": True,
            "convective_predictor_enabled": False,
        },
        "results": results,
        "convergence": convergence_summary(results),
    }
    markdown = render_markdown(payload)
    payload = _json_ready(payload)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"[poiseuille] wrote {json_path}")
    print(f"[poiseuille] wrote {markdown_path}")
    has_error = any(row.get("status") == "error" for row in results)
    has_rejection = any(
        row.get("status") == "completed"
        and not bool(dict(row.get("acceptance", {})).get("passed", False))
        for row in results
    )
    return 2 if has_error or (bool(args.require_acceptance) and has_rejection) else 0


if __name__ == "__main__":
    raise SystemExit(main())
