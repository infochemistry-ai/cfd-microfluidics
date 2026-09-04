"""Prescribed debug velocity fields for tetra transport experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np

from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh

VelocityFieldName = Literal[
    "constant_x",
    "two_inlets_to_outlet_tj",
    "two_inlets_to_outlet_tj_balanced",
    "two_inlets_to_outlet_tj_balanced_symmetric_profile",
    "two_inlets_to_outlet_tj_piecewise_centerline",
    "two_inlets_to_outlet_tj_axis_aligned_sanity",
    "two_inlets_to_outlet_tj_axis_aligned_clean",
]


@dataclass(frozen=True)
class VelocityFieldData:
    """Prescribed transport velocity data."""

    name: VelocityFieldName
    cell_velocity: np.ndarray
    boundary_face_velocity_overrides: Dict[int, np.ndarray]
    boundary_groups: Dict[str, np.ndarray]
    metadata: Dict[str, object]


def _owner_cells_for_faces(mesh: ImportedTetraMesh, face_ids: np.ndarray) -> np.ndarray:
    if face_ids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    owners = mesh.face_to_cells[np.asarray(face_ids, dtype=np.int64), 0]
    return np.asarray(np.unique(owners), dtype=np.int64)


def _normalize_name(name: str) -> str:
    cleaned = str(name).strip().lower()
    for token in (" ", "-", ".", "/"):
        cleaned = cleaned.replace(token, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def resolve_tj_boundary_groups(mesh: ImportedTetraMesh) -> Dict[str, np.ndarray]:
    """Resolve left/right inlet, outlet and wall face groups from imported tags."""

    tag_to_name = {
        int(tag): _normalize_name(name)
        for tag, name in mesh.boundary_face_names.items()
    }
    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    boundary_tag = np.asarray(mesh.boundary_tag_per_face, dtype=np.int32)

    left_tags = {
        tag for tag, name in tag_to_name.items() if "left" in name and "inlet" in name
    }
    right_tags = {
        tag for tag, name in tag_to_name.items() if "right" in name and "inlet" in name
    }

    if left_tags and right_tags:
        left = inlet_faces[np.isin(boundary_tag[inlet_faces], list(left_tags))]
        right = inlet_faces[np.isin(boundary_tag[inlet_faces], list(right_tags))]
        mode = "named"
    else:
        x = mesh.face_centers[inlet_faces, 0]
        split = float(np.median(x))
        left = inlet_faces[x <= split]
        right = inlet_faces[x > split]
        mode = "x_split_fallback"

    if left.size == 0 or right.size == 0:
        x = mesh.face_centers[inlet_faces, 0]
        order = np.argsort(x)
        half = max(1, inlet_faces.size // 2)
        left = inlet_faces[order[:half]]
        right = inlet_faces[order[half:]]
        mode = "ordered_fallback"

    return {
        "left_inlet_faces": np.asarray(np.unique(left), dtype=np.int64),
        "right_inlet_faces": np.asarray(np.unique(right), dtype=np.int64),
        "inlet_faces": np.asarray(np.unique(inlet_faces), dtype=np.int64),
        "outlet_faces": np.asarray(np.unique(mesh.outlet_faces), dtype=np.int64),
        "wall_faces": np.asarray(np.unique(mesh.wall_faces), dtype=np.int64),
        "mode": np.asarray(mode),
    }


def _build_boundary_velocity_overrides(
    *,
    mesh: ImportedTetraMesh,
    boundary_groups: Dict[str, np.ndarray],
    cell_velocity: np.ndarray,
    inlet_speed: float,
) -> Dict[int, np.ndarray]:
    overrides: Dict[int, np.ndarray] = {}
    wall_faces = np.asarray(boundary_groups["wall_faces"], dtype=np.int64)
    outlet_faces = np.asarray(boundary_groups["outlet_faces"], dtype=np.int64)
    inlet_faces = np.asarray(boundary_groups["inlet_faces"], dtype=np.int64)

    # No-penetration walls.
    for face_idx in wall_faces.tolist():
        overrides[int(face_idx)] = np.zeros(3, dtype=np.float64)

    # Prescribed inflow aligned opposite outward normal.
    for face_idx in inlet_faces.tolist():
        n = np.asarray(mesh.face_normals[int(face_idx)], dtype=np.float64)
        overrides[int(face_idx)] = -float(inlet_speed) * n

    # Outlet: allow outflow, suppress artificial backflow.
    for face_idx in outlet_faces.tolist():
        owner = int(mesh.face_to_cells[int(face_idx), 0])
        n = np.asarray(mesh.face_normals[int(face_idx)], dtype=np.float64)
        owner_v = np.asarray(cell_velocity[owner], dtype=np.float64)
        normal_component = max(float(np.dot(owner_v, n)), 0.0)
        overrides[int(face_idx)] = normal_component * n

    return overrides


def _principal_axis_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-30:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return axis / norm


def _symmetric_profile_inlet_overrides(
    *,
    mesh: ImportedTetraMesh,
    inlet_faces: np.ndarray,
    inlet_speed: float,
) -> tuple[Dict[int, np.ndarray], Dict[str, float]]:
    overrides: Dict[int, np.ndarray] = {}
    if inlet_faces.size == 0:
        return overrides, {"profile_width_xy": 0.0}

    faces = np.asarray(inlet_faces, dtype=np.int64)
    centers = np.asarray(mesh.face_centers[faces], dtype=np.float64)
    xy = centers[:, :2]
    axis = _principal_axis_xy(xy)
    xy_center = np.mean(xy, axis=0)
    s = (xy - xy_center) @ axis
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    span = max(s_max - s_min, 1e-12)
    s_norm = 2.0 * (s - s_min) / span - 1.0
    weights = np.clip(1.0 - s_norm * s_norm, 0.0, None)
    weights = np.maximum(weights, 1e-6)

    areas = np.asarray(mesh.face_areas[faces], dtype=np.float64)
    denom = float(np.sum(weights * areas))
    if denom <= 1e-30:
        alpha = 1.0
    else:
        alpha = float(np.sum(areas) / denom)
    speeds = float(inlet_speed) * alpha * weights

    for i, face_idx in enumerate(faces.tolist()):
        n = np.asarray(mesh.face_normals[int(face_idx)], dtype=np.float64)
        overrides[int(face_idx)] = -float(speeds[i]) * n

    return overrides, {
        "profile_width_xy": span,
        "profile_scale_alpha": alpha,
        "profile_speed_min": float(np.min(speeds)),
        "profile_speed_max": float(np.max(speeds)),
    }


def _scale_outlet_overrides(
    *,
    overrides: Dict[int, np.ndarray],
    outlet_faces: np.ndarray,
    scale: float,
) -> Dict[int, np.ndarray]:
    out = {int(k): np.asarray(v, dtype=np.float64).copy() for k, v in overrides.items()}
    for face_idx in np.asarray(outlet_faces, dtype=np.int64).tolist():
        if int(face_idx) in out:
            out[int(face_idx)] = float(scale) * out[int(face_idx)]
    return out


def _flux_diagnostics_from_overrides(
    *,
    mesh: ImportedTetraMesh,
    cell_velocity: np.ndarray,
    overrides: Dict[int, np.ndarray],
    boundary_groups: Dict[str, np.ndarray],
) -> Dict[str, float]:
    from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
        build_face_normal_flux_from_velocity,
    )

    _, flux = build_face_normal_flux_from_velocity(
        mesh,
        cell_velocity,
        boundary_face_velocity_overrides=overrides,
        left_inlet_faces=np.asarray(
            boundary_groups["left_inlet_faces"], dtype=np.int64
        ),
        right_inlet_faces=np.asarray(
            boundary_groups["right_inlet_faces"], dtype=np.int64
        ),
        outlet_faces=np.asarray(boundary_groups["outlet_faces"], dtype=np.int64),
        wall_faces=np.asarray(boundary_groups["wall_faces"], dtype=np.int64),
    )
    return flux


def _constant_x_velocity(mesh: ImportedTetraMesh, speed: float) -> np.ndarray:
    velocity = np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = float(speed)
    return velocity


def _two_inlets_to_outlet_velocity(mesh: ImportedTetraMesh, speed: float) -> np.ndarray:
    centers = mesh.cell_centers
    x = centers[:, 0]
    y = centers[:, 1]
    inlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 1])
    )
    outlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1])
    )
    y_span = max(outlet_y - inlet_y, 1e-6)
    y_ratio = np.clip((y - inlet_y) / y_span, 0.0, 1.0)

    x_extent = np.percentile(x, 95.0) - np.percentile(x, 5.0)
    x_scale = max(float(0.35 * x_extent), 1e-6)

    velocity = np.zeros((centers.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = -0.8 * float(speed) * np.tanh(x / x_scale) * (1.0 - y_ratio)
    velocity[:, 1] = float(speed) * (0.35 + 0.95 * y_ratio)
    return velocity


def _two_inlets_to_outlet_piecewise_centerline_velocity(
    mesh: ImportedTetraMesh, speed: float
) -> np.ndarray:
    centers = mesh.cell_centers
    x = centers[:, 0]
    y = centers[:, 1]
    inlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 1])
    )
    outlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1])
    )
    y_span = max(outlet_y - inlet_y, 1e-6)
    y_ratio = np.clip((y - inlet_y) / y_span, 0.0, 1.0)

    inlet_x = mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 0]
    x_scale = max(float(np.percentile(np.abs(inlet_x), 65.0)), 1e-6)

    # In inlet branches: left -> +x, right -> -x.
    ux_inlet = -float(speed) * np.tanh(x / x_scale)
    # Damp x-advection toward outlet branch.
    ux = ux_inlet * (1.0 - y_ratio)

    # Outlet-aligned acceleration along +y, stronger near centerline.
    center_weight = np.exp(-((x / max(0.7 * x_scale, 1e-6)) ** 2))
    uy = float(speed) * (0.05 + 0.95 * y_ratio + 0.35 * center_weight * y_ratio)

    velocity = np.zeros((centers.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = ux
    velocity[:, 1] = np.maximum(uy, 0.0)
    velocity[:, 2] = 0.0
    return velocity


def _two_inlets_to_outlet_axis_aligned_sanity_velocity(
    mesh: ImportedTetraMesh, speed: float
) -> np.ndarray:
    centers = mesh.cell_centers
    x = centers[:, 0]
    y = centers[:, 1]
    inlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 1])
    )
    outlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1])
    )
    y_span = max(outlet_y - inlet_y, 1e-6)

    outlet_face_x = mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 0]
    if outlet_face_x.size:
        outlet_half_width = max(
            0.5 * float(np.max(outlet_face_x) - np.min(outlet_face_x)), 1e-9
        )
    else:
        outlet_half_width = max(float(np.percentile(np.abs(x), 50.0)), 1e-9)

    # Keep inlet branches almost purely horizontal until close to junction.
    junction_y = inlet_y + 0.08 * y_span
    blend_half_height = max(0.06 * y_span, 2e-4)
    y0 = junction_y - blend_half_height
    y1 = junction_y + blend_half_height
    t = np.clip((y - y0) / max(y1 - y0, 1e-12), 0.0, 1.0)

    # Horizontal direction: left branch -> +x, right branch -> -x.
    sign_to_junction = -np.sign(x)
    sign_to_junction[np.abs(x) <= 1e-12] = 0.0
    ux_base = float(speed) * sign_to_junction

    # Blend across x near centerline so outlet branch is mostly +y.
    x_blend = np.clip(np.abs(x) / max(outlet_half_width, 1e-12), 0.0, 1.0)
    horizontal_weight = x_blend * (1.0 - t)
    vertical_weight = 1.0 - horizontal_weight

    velocity = np.zeros((centers.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = ux_base * horizontal_weight
    velocity[:, 1] = float(speed) * vertical_weight
    velocity[:, 2] = 0.0
    return velocity


def _smoothstep01(v: np.ndarray) -> np.ndarray:
    t = np.clip(v, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _two_inlets_to_outlet_axis_aligned_clean_velocity(
    mesh: ImportedTetraMesh, speed: float
) -> np.ndarray:
    centers = mesh.cell_centers
    x = centers[:, 0]
    y = centers[:, 1]

    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    inlet_y = float(np.median(mesh.face_centers[inlet_faces, 1]))
    outlet_y = float(np.median(mesh.face_centers[outlet_faces, 1]))
    y_span = max(outlet_y - inlet_y, 1e-6)
    junction_y = inlet_y + 0.08 * y_span

    outlet_face_x = mesh.face_centers[outlet_faces, 0]
    if outlet_face_x.size:
        outlet_width = float(np.max(outlet_face_x) - np.min(outlet_face_x))
        outlet_half_width = max(0.5 * outlet_width, 1e-9)
    else:
        outlet_half_width = max(float(np.percentile(np.abs(x), 45.0)), 1e-9)

    inlet_face_x = mesh.face_centers[inlet_faces, 0]
    if inlet_face_x.size:
        inlet_width = max(
            float(np.percentile(inlet_face_x, 95.0) - np.percentile(inlet_face_x, 5.0)),
            1e-9,
        )
    else:
        inlet_width = max(float(0.5 * outlet_half_width), 1e-9)

    sign_to_junction = -np.sign(x)
    sign_to_junction[np.abs(x) <= 1e-12] = 0.0
    ux = float(speed) * sign_to_junction
    uy = np.zeros_like(ux)

    # Keep inlet branches strictly horizontal until very close to junction.
    blend_start_y = junction_y - 0.15 * inlet_width
    blend_end_y = junction_y + 0.60 * inlet_width
    blend_len_y = max(blend_end_y - blend_start_y, 1e-12)
    t = _smoothstep01((y - blend_start_y) / blend_len_y)
    ux = ux * (1.0 - t)
    uy = float(speed) * t

    # In outlet branch (above blend): almost purely vertical.
    outlet_vertical = y >= blend_end_y
    ux[outlet_vertical] = 0.0
    uy[outlet_vertical] = float(speed)

    # In deep inlet branch (below blend): enforce horizontal plug flow.
    inlet_horizontal = y <= blend_start_y
    ux[inlet_horizontal] = float(speed) * sign_to_junction[inlet_horizontal]
    uy[inlet_horizontal] = 0.0

    velocity = np.zeros((centers.shape[0], 3), dtype=np.float64)
    velocity[:, 0] = ux
    velocity[:, 1] = np.maximum(uy, 0.0)
    velocity[:, 2] = 0.0
    return velocity


def build_prescribed_velocity_field(
    mesh: ImportedTetraMesh,
    *,
    field_name: VelocityFieldName = "two_inlets_to_outlet_tj",
    inlet_speed: float = 0.15,
) -> VelocityFieldData:
    """Build prescribed debug velocity for transport-only tetra experiments."""

    if field_name not in {
        "constant_x",
        "two_inlets_to_outlet_tj",
        "two_inlets_to_outlet_tj_balanced",
        "two_inlets_to_outlet_tj_balanced_symmetric_profile",
        "two_inlets_to_outlet_tj_piecewise_centerline",
        "two_inlets_to_outlet_tj_axis_aligned_sanity",
        "two_inlets_to_outlet_tj_axis_aligned_clean",
    }:
        raise ValueError(f"Unsupported velocity field: {field_name!r}")

    boundary_groups = resolve_tj_boundary_groups(mesh)
    if field_name == "constant_x":
        cell_velocity = _constant_x_velocity(mesh, inlet_speed)
    elif field_name == "two_inlets_to_outlet_tj_piecewise_centerline":
        cell_velocity = _two_inlets_to_outlet_piecewise_centerline_velocity(
            mesh, inlet_speed
        )
    elif field_name == "two_inlets_to_outlet_tj_axis_aligned_sanity":
        cell_velocity = _two_inlets_to_outlet_axis_aligned_sanity_velocity(
            mesh, inlet_speed
        )
    elif field_name == "two_inlets_to_outlet_tj_axis_aligned_clean":
        cell_velocity = _two_inlets_to_outlet_axis_aligned_clean_velocity(
            mesh, inlet_speed
        )
    else:
        cell_velocity = _two_inlets_to_outlet_velocity(mesh, inlet_speed)

    base_overrides = _build_boundary_velocity_overrides(
        mesh=mesh,
        boundary_groups=boundary_groups,
        cell_velocity=cell_velocity,
        inlet_speed=inlet_speed,
    )
    profile_diag: Dict[str, float | bool] = {
        "symmetric_inlet_profile_enabled": False,
    }
    if field_name == "two_inlets_to_outlet_tj_balanced_symmetric_profile":
        left_faces = np.asarray(boundary_groups["left_inlet_faces"], dtype=np.int64)
        right_faces = np.asarray(boundary_groups["right_inlet_faces"], dtype=np.int64)
        left_overrides, left_diag = _symmetric_profile_inlet_overrides(
            mesh=mesh, inlet_faces=left_faces, inlet_speed=inlet_speed
        )
        right_overrides, right_diag = _symmetric_profile_inlet_overrides(
            mesh=mesh, inlet_faces=right_faces, inlet_speed=inlet_speed
        )
        for face_idx, vec in left_overrides.items():
            base_overrides[int(face_idx)] = vec
        for face_idx, vec in right_overrides.items():
            base_overrides[int(face_idx)] = vec
        profile_diag = {
            "symmetric_inlet_profile_enabled": True,
            "left_profile_width_xy": float(left_diag.get("profile_width_xy", 0.0)),
            "right_profile_width_xy": float(right_diag.get("profile_width_xy", 0.0)),
            "left_profile_scale_alpha": float(
                left_diag.get("profile_scale_alpha", 1.0)
            ),
            "right_profile_scale_alpha": float(
                right_diag.get("profile_scale_alpha", 1.0)
            ),
            "left_profile_speed_min": float(left_diag.get("profile_speed_min", 0.0)),
            "left_profile_speed_max": float(left_diag.get("profile_speed_max", 0.0)),
            "right_profile_speed_min": float(right_diag.get("profile_speed_min", 0.0)),
            "right_profile_speed_max": float(right_diag.get("profile_speed_max", 0.0)),
        }
    balance_diag: Dict[str, float | bool] = {
        "balanced_velocity_enabled": False,
        "outlet_scale_factor": 1.0,
    }
    if field_name in {
        "two_inlets_to_outlet_tj_balanced",
        "two_inlets_to_outlet_tj_balanced_symmetric_profile",
        "two_inlets_to_outlet_tj_piecewise_centerline",
        "two_inlets_to_outlet_tj_axis_aligned_sanity",
        "two_inlets_to_outlet_tj_axis_aligned_clean",
    }:
        before = _flux_diagnostics_from_overrides(
            mesh=mesh,
            cell_velocity=cell_velocity,
            overrides=base_overrides,
            boundary_groups=boundary_groups,
        )
        inlet_before = float(before["total_inlet_flux_in"])
        outlet_before = float(before["total_outlet_flux_out"])
        if outlet_before > 1e-30:
            scale = inlet_before / outlet_before
        else:
            scale = 1.0
        overrides = _scale_outlet_overrides(
            overrides=base_overrides,
            outlet_faces=np.asarray(boundary_groups["outlet_faces"], dtype=np.int64),
            scale=scale,
        )
        after = _flux_diagnostics_from_overrides(
            mesh=mesh,
            cell_velocity=cell_velocity,
            overrides=overrides,
            boundary_groups=boundary_groups,
        )
        inlet_after = float(after["total_inlet_flux_in"])
        outlet_after = float(after["total_outlet_flux_out"])
        imbalance_after = float(
            abs(after["net_boundary_flux"]) / max(inlet_after, 1e-30)
        )
        balance_diag = {
            "balanced_velocity_enabled": True,
            "inlet_flux_before_balance": inlet_before,
            "outlet_flux_before_balance": outlet_before,
            "outlet_scale_factor": float(scale),
            "inlet_flux_after_balance": inlet_after,
            "outlet_flux_after_balance": outlet_after,
            "net_boundary_flux_after_balance": float(after["net_boundary_flux"]),
            "boundary_flux_imbalance_ratio_after_balance": imbalance_after,
            "wall_flux_max_abs_after_balance": float(after["wall_flux_max_abs"]),
        }
    else:
        overrides = base_overrides

    metadata: Dict[str, object] = {
        "velocity_source": "prescribed_debug_field",
        "flow_solved": False,
        "pressure_solved": False,
        "field_name": field_name,
        "inlet_speed": float(inlet_speed),
        "boundary_group_resolution_mode": str(
            np.asarray(boundary_groups["mode"]).item()
        ),
    }
    metadata.update(balance_diag)
    metadata.update(profile_diag)
    return VelocityFieldData(
        name=field_name,
        cell_velocity=cell_velocity,
        boundary_face_velocity_overrides=overrides,
        boundary_groups=boundary_groups,
        metadata=metadata,
    )


def compute_velocity_field_diagnostics(
    mesh: ImportedTetraMesh,
    velocity_field: VelocityFieldData,
    *,
    flux_diagnostics: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Compute debug diagnostics for prescribed TJ velocity consistency."""

    velocity = np.asarray(velocity_field.cell_velocity, dtype=np.float64)
    speed = np.linalg.norm(velocity, axis=1)

    left_faces = np.asarray(
        velocity_field.boundary_groups["left_inlet_faces"], dtype=np.int64
    )
    right_faces = np.asarray(
        velocity_field.boundary_groups["right_inlet_faces"], dtype=np.int64
    )
    outlet_faces = np.asarray(
        velocity_field.boundary_groups["outlet_faces"], dtype=np.int64
    )

    left_cells = _owner_cells_for_faces(mesh, left_faces)
    right_cells = _owner_cells_for_faces(mesh, right_faces)
    outlet_cells = _owner_cells_for_faces(mesh, outlet_faces)

    inlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 1])
    )
    outlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1])
    )
    y_span = max(outlet_y - inlet_y, 1e-12)
    junction_y = inlet_y + 0.08 * y_span
    x_abs = np.abs(mesh.cell_centers[:, 0])
    x_thresh = float(np.percentile(x_abs, 35.0))
    near_junction = (np.abs(mesh.cell_centers[:, 1] - junction_y) <= 0.12 * y_span) & (
        x_abs <= x_thresh
    )
    if not np.any(near_junction):
        near_junction = np.abs(mesh.cell_centers[:, 1] - junction_y) <= 0.15 * y_span

    avg_left = float(np.mean(speed[left_cells])) if left_cells.size else 0.0
    avg_right = float(np.mean(speed[right_cells])) if right_cells.size else 0.0
    avg_outlet = float(np.mean(speed[outlet_cells])) if outlet_cells.size else 0.0
    max_junction = (
        float(np.max(speed[near_junction]))
        if np.any(near_junction)
        else float(np.max(speed))
    )

    right_face_x = (
        mesh.face_centers[right_faces, 0] if right_faces.size else np.zeros((0,))
    )
    left_face_x = (
        mesh.face_centers[left_faces, 0] if left_faces.size else np.zeros((0,))
    )
    inlet_axis_abs = max(
        float(np.mean(np.abs(right_face_x))) if right_face_x.size else 0.0,
        float(np.mean(np.abs(left_face_x))) if left_face_x.size else 0.0,
    )
    if inlet_axis_abs <= 1e-12:
        inlet_axis_abs = max(
            float(np.percentile(np.abs(mesh.cell_centers[:, 0]), 80.0)), 1e-12
        )
    branch_x_cut = 0.6 * inlet_axis_abs
    branch_y_limit = junction_y - 0.01 * y_span
    x_abs = np.abs(mesh.cell_centers[:, 0])
    x_thresh_lr = float(np.percentile(x_abs, 35.0))
    outlet_face_x = (
        mesh.face_centers[outlet_faces, 0] if outlet_faces.size else np.zeros((0,))
    )
    if outlet_face_x.size:
        outlet_half_width = max(
            0.5 * float(np.max(outlet_face_x) - np.min(outlet_face_x)),
            1e-12,
        )
    else:
        outlet_half_width = max(x_thresh_lr, 1e-12)
    right_mask = (mesh.cell_centers[:, 0] >= branch_x_cut) & (
        mesh.cell_centers[:, 1] <= branch_y_limit
    )
    left_mask = (mesh.cell_centers[:, 0] <= -branch_x_cut) & (
        mesh.cell_centers[:, 1] <= branch_y_limit
    )
    junction_mask = near_junction
    outlet_mask = (mesh.cell_centers[:, 1] > (junction_y + 0.10 * y_span)) & (
        x_abs <= 1.1 * outlet_half_width
    )
    if not np.any(right_mask):
        right_mask = mesh.cell_centers[:, 0] > 0.0
    if not np.any(left_mask):
        left_mask = mesh.cell_centers[:, 0] < 0.0
    if not np.any(outlet_mask):
        outlet_mask = mesh.cell_centers[:, 1] > junction_y

    def _zone_stats(mask: np.ndarray) -> Dict[str, float]:
        if not np.any(mask):
            return {
                "cell_count": 0,
                "speed_min": float("nan"),
                "speed_max": float("nan"),
                "speed_mean": float("nan"),
                "u_x_min": float("nan"),
                "u_x_max": float("nan"),
                "u_x_mean": float("nan"),
                "u_y_min": float("nan"),
                "u_y_max": float("nan"),
                "u_y_mean": float("nan"),
                "u_z_min": float("nan"),
                "u_z_max": float("nan"),
                "u_z_mean": float("nan"),
            }
        u = velocity[mask]
        s = speed[mask]
        return {
            "cell_count": int(np.count_nonzero(mask)),
            "speed_min": float(np.min(s)),
            "speed_max": float(np.max(s)),
            "speed_mean": float(np.mean(s)),
            "u_x_min": float(np.min(u[:, 0])),
            "u_x_max": float(np.max(u[:, 0])),
            "u_x_mean": float(np.mean(u[:, 0])),
            "u_y_min": float(np.min(u[:, 1])),
            "u_y_max": float(np.max(u[:, 1])),
            "u_y_mean": float(np.mean(u[:, 1])),
            "u_z_min": float(np.min(u[:, 2])),
            "u_z_max": float(np.max(u[:, 2])),
            "u_z_mean": float(np.mean(u[:, 2])),
        }

    def _inlet_direction_stats(
        mask: np.ndarray, toward_negative_x: bool
    ) -> Dict[str, float]:
        if not np.any(mask):
            return {
                "toward_junction_fraction": float("nan"),
                "away_from_junction_fraction": float("nan"),
                "near_zero_speed_fraction": float("nan"),
                "mean_abs_uy_over_abs_ux": float("nan"),
                "mean_abs_uz_over_abs_ux": float("nan"),
                "u_x_min": float("nan"),
                "u_x_max": float("nan"),
                "u_x_mean": float("nan"),
                "u_y_min": float("nan"),
                "u_y_max": float("nan"),
                "u_y_mean": float("nan"),
                "u_z_min": float("nan"),
                "u_z_max": float("nan"),
                "u_z_mean": float("nan"),
            }
        u = velocity[mask]
        s = speed[mask]
        ux = u[:, 0]
        uy = u[:, 1]
        uz = u[:, 2]
        toward = ux < 0.0 if toward_negative_x else ux > 0.0
        away = ux > 0.0 if toward_negative_x else ux < 0.0
        near_zero = s <= max(1e-12, 1e-3 * max(float(np.mean(s)), 1e-12))
        denom = np.maximum(np.abs(ux), 1e-12)
        return {
            "toward_junction_fraction": float(np.mean(toward)),
            "away_from_junction_fraction": float(np.mean(away)),
            "near_zero_speed_fraction": float(np.mean(near_zero)),
            "mean_abs_uy_over_abs_ux": float(np.mean(np.abs(uy) / denom)),
            "mean_abs_uz_over_abs_ux": float(np.mean(np.abs(uz) / denom)),
            "mostly_horizontal_fraction": float(np.mean(np.abs(ux) > 5.0 * np.abs(uy))),
            "transverse_bad_fraction": float(np.mean(np.abs(uy) > np.abs(ux))),
            "u_x_min": float(np.min(ux)),
            "u_x_max": float(np.max(ux)),
            "u_x_mean": float(np.mean(ux)),
            "u_y_min": float(np.min(uy)),
            "u_y_max": float(np.max(uy)),
            "u_y_mean": float(np.mean(uy)),
            "u_z_min": float(np.min(uz)),
            "u_z_max": float(np.max(uz)),
            "u_z_mean": float(np.mean(uz)),
        }

    def _outlet_direction_stats(mask: np.ndarray) -> Dict[str, float]:
        if not np.any(mask):
            return {
                "mostly_vertical_fraction": float("nan"),
                "mean_abs_ux_over_abs_uy": float("nan"),
                "mean_abs_uz_over_abs_uy": float("nan"),
                "transverse_bad_fraction": float("nan"),
                "near_zero_speed_fraction": float("nan"),
                "u_x_mean": float("nan"),
                "u_y_mean": float("nan"),
                "u_z_mean": float("nan"),
            }
        u = velocity[mask]
        s = speed[mask]
        ux = u[:, 0]
        uy = u[:, 1]
        uz = u[:, 2]
        denom = np.maximum(np.abs(uy), 1e-12)
        near_zero = s <= max(1e-12, 1e-3 * max(float(np.mean(s)), 1e-12))
        return {
            "mostly_vertical_fraction": float(np.mean(np.abs(uy) > 5.0 * np.abs(ux))),
            "mean_abs_ux_over_abs_uy": float(np.mean(np.abs(ux) / denom)),
            "mean_abs_uz_over_abs_uy": float(np.mean(np.abs(uz) / denom)),
            "transverse_bad_fraction": float(np.mean(np.abs(ux) > np.abs(uy))),
            "near_zero_speed_fraction": float(np.mean(near_zero)),
            "u_x_mean": float(np.mean(ux)),
            "u_y_mean": float(np.mean(uy)),
            "u_z_mean": float(np.mean(uz)),
        }

    imbalance_ratio = 0.0
    imbalance_warning = False
    if flux_diagnostics is not None:
        inlet_flux = float(abs(flux_diagnostics.get("total_inlet_flux_in", 0.0)))
        net_flux = float(abs(flux_diagnostics.get("net_boundary_flux", 0.0)))
        denom = max(inlet_flux, 1e-30)
        imbalance_ratio = net_flux / denom
        imbalance_warning = bool(imbalance_ratio > 0.05)

    return {
        "avg_speed_left_inlet_branch": avg_left,
        "avg_speed_right_inlet_branch": avg_right,
        "avg_speed_outlet_branch": avg_outlet,
        "max_speed_near_junction": max_junction,
        "boundary_flux_imbalance_ratio": imbalance_ratio,
        "boundary_flux_imbalance_warning": imbalance_warning,
        "boundary_flux_imbalance_threshold": 0.05,
        "cell_velocity_magnitude_stats": {
            "min": float(np.min(speed)),
            "max": float(np.max(speed)),
            "mean": float(np.mean(speed)),
        },
        "zones": {
            "right_inlet_branch": _zone_stats(right_mask),
            "left_inlet_branch": _zone_stats(left_mask),
            "junction_zone": _zone_stats(junction_mask),
            "outlet_branch": _zone_stats(outlet_mask),
        },
        "right_inlet_branch_direction": _inlet_direction_stats(
            right_mask, toward_negative_x=True
        ),
        "left_inlet_branch_direction": _inlet_direction_stats(
            left_mask, toward_negative_x=False
        ),
        "outlet_branch_direction": _outlet_direction_stats(outlet_mask),
    }
