"""Geometry primitives and builders for the mesh-oriented 3D branch.

This module enforces a strict geometry contract for solver-facing entities
while safely skipping optional degenerate wall subregions.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from microfluidics.mesh.mesh_config import MeshGeometryConfig3D
from microfluidics.mesh.mesh_validation import validate_mesh_geometry_config


@dataclass(frozen=True)
class AxisAlignedBox3D:
    """Simple axis-aligned box used by the mesh branch."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    def contains_points(self, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        return (
            (points[:, 0] >= self.x[0] - eps)
            & (points[:, 0] <= self.x[1] + eps)
            & (points[:, 1] >= self.y[0] - eps)
            & (points[:, 1] <= self.y[1] + eps)
            & (points[:, 2] >= self.z[0] - eps)
            & (points[:, 2] <= self.z[1] + eps)
        )

    def contains_point(
        self, point: Tuple[float, float, float], eps: float = 1e-12
    ) -> bool:
        x, y, z = point
        return (
            self.x[0] - eps <= x <= self.x[1] + eps
            and self.y[0] - eps <= y <= self.y[1] + eps
            and self.z[0] - eps <= z <= self.z[1] + eps
        )


@dataclass(frozen=True)
class MeshGeometryModel3D:
    """Normalized geometry model for mesh generation."""

    config: MeshGeometryConfig3D
    domain: AxisAlignedBox3D
    fluid_components: Dict[str, AxisAlignedBox3D]
    boundary_patches: Dict[str, AxisAlignedBox3D]
    wall_regions: Dict[str, AxisAlignedBox3D]
    validation_warnings: tuple[str, ...] = ()


def _validate_box(name: str, box: AxisAlignedBox3D) -> None:
    if box.x[1] <= box.x[0] or box.y[1] <= box.y[0] or box.z[1] <= box.z[0]:
        raise ValueError(f"Invalid box bounds for {name}.")


def _make_box(
    name: str,
    x: Tuple[float, float],
    y: Tuple[float, float],
    z: Tuple[float, float],
    *,
    allow_degenerate: bool = False,
) -> AxisAlignedBox3D | None:
    if x[1] <= x[0] or y[1] <= y[0] or z[1] <= z[0]:
        if allow_degenerate:
            return None
        raise ValueError(
            f"Invalid bounds for {name}: x={x}, y={y}, z={z}. Expected strictly increasing bounds."
        )
    return AxisAlignedBox3D(x=x, y=y, z=z)


def _base_channels_3d(cfg: MeshGeometryConfig3D) -> Dict[str, AxisAlignedBox3D]:
    half_inlet = 0.5 * cfg.inlet_width
    half_outlet = 0.5 * cfg.outlet_width
    z_span = (0.0, cfg.channel_height)
    return {
        "left_inlet_channel": AxisAlignedBox3D(
            x=(-cfg.inlet_length, 0.0),
            y=(-half_inlet, half_inlet),
            z=z_span,
        ),
        "right_inlet_channel": AxisAlignedBox3D(
            x=(0.0, cfg.inlet_length),
            y=(-half_inlet, half_inlet),
            z=z_span,
        ),
        "mixing_channel": AxisAlignedBox3D(
            x=(-half_outlet, half_outlet),
            y=(0.0, cfg.outlet_length),
            z=z_span,
        ),
    }


def _create_cavities_3d(cfg: MeshGeometryConfig3D) -> Dict[str, AxisAlignedBox3D]:
    half_outlet = 0.5 * cfg.outlet_width
    y_min = cfg.cavity_offset_from_junction
    y_max = y_min + cfg.cavity_length
    z_span = (0.0, cfg.channel_height)
    return {
        "left_cavity": AxisAlignedBox3D(
            x=(-half_outlet - cfg.cavity_depth, -half_outlet),
            y=(y_min, y_max),
            z=z_span,
        ),
        "right_cavity": AxisAlignedBox3D(
            x=(half_outlet, half_outlet + cfg.cavity_depth),
            y=(y_min, y_max),
            z=z_span,
        ),
    }


def _create_boundary_patches_3d(
    cfg: MeshGeometryConfig3D,
) -> Dict[str, AxisAlignedBox3D]:
    half_inlet = 0.5 * cfg.inlet_width
    half_outlet = 0.5 * cfg.outlet_width
    z_span = (0.0, cfg.channel_height)
    return {
        "left_inlet": AxisAlignedBox3D(
            x=(-cfg.inlet_length, -cfg.inlet_length + cfg.patch_depth),
            y=(-half_inlet, half_inlet),
            z=z_span,
        ),
        "right_inlet": AxisAlignedBox3D(
            x=(cfg.inlet_length - cfg.patch_depth, cfg.inlet_length),
            y=(-half_inlet, half_inlet),
            z=z_span,
        ),
        "outlet": AxisAlignedBox3D(
            x=(-half_outlet, half_outlet),
            y=(cfg.outlet_length - cfg.patch_depth, cfg.outlet_length),
            z=z_span,
        ),
    }


def _wall_regions_3d(cfg: MeshGeometryConfig3D) -> Dict[str, AxisAlignedBox3D]:
    half_inlet = 0.5 * cfg.inlet_width
    half_outlet = 0.5 * cfg.outlet_width
    half_span_y = max(half_inlet, half_outlet)
    half_span_x = max(cfg.inlet_length, half_outlet + cfg.cavity_depth)
    z_span = (0.0, cfg.channel_height)
    regions: Dict[str, AxisAlignedBox3D] = {}
    bottom_left = _make_box(
        "bottom_left_corner",
        x=(-half_span_x, 0.0),
        y=(-half_span_y, -half_inlet),
        z=z_span,
    )
    bottom_right = _make_box(
        "bottom_right_corner",
        x=(0.0, half_span_x),
        y=(-half_span_y, -half_inlet),
        z=z_span,
    )
    assert bottom_left is not None and bottom_right is not None  # strict boxes
    regions["bottom_left_corner"] = bottom_left
    regions["bottom_right_corner"] = bottom_right
    if not cfg.include_cavities:
        left_block = _make_box(
            "upper_left_block",
            x=(-half_span_x, -half_outlet),
            y=(half_inlet, cfg.outlet_length),
            z=z_span,
        )
        right_block = _make_box(
            "upper_right_block",
            x=(half_outlet, half_span_x),
            y=(half_inlet, cfg.outlet_length),
            z=z_span,
        )
        assert left_block is not None and right_block is not None  # strict boxes
        regions["upper_left_block"] = left_block
        regions["upper_right_block"] = right_block
        return regions

    cavity_y_min = cfg.cavity_offset_from_junction
    cavity_y_max = min(
        cfg.cavity_offset_from_junction + cfg.cavity_length, cfg.outlet_length
    )
    left_near_min = -half_outlet - cfg.cavity_depth
    left_near_max = -half_outlet
    right_near_min = half_outlet
    right_near_max = half_outlet + cfg.cavity_depth
    optional_regions = {
        "upper_left_outer": _make_box(
            "upper_left_outer",
            x=(-half_span_x, left_near_min),
            y=(half_inlet, cfg.outlet_length),
            z=z_span,
            allow_degenerate=True,
        ),
        "upper_left_before_cavity": _make_box(
            "upper_left_before_cavity",
            x=(left_near_min, left_near_max),
            y=(half_inlet, cavity_y_min),
            z=z_span,
            allow_degenerate=True,
        ),
        "upper_left_after_cavity": _make_box(
            "upper_left_after_cavity",
            x=(left_near_min, left_near_max),
            y=(cavity_y_max, cfg.outlet_length),
            z=z_span,
            allow_degenerate=True,
        ),
        "upper_right_outer": _make_box(
            "upper_right_outer",
            x=(right_near_max, half_span_x),
            y=(half_inlet, cfg.outlet_length),
            z=z_span,
            allow_degenerate=True,
        ),
        "upper_right_before_cavity": _make_box(
            "upper_right_before_cavity",
            x=(right_near_min, right_near_max),
            y=(half_inlet, cavity_y_min),
            z=z_span,
            allow_degenerate=True,
        ),
        "upper_right_after_cavity": _make_box(
            "upper_right_after_cavity",
            x=(right_near_min, right_near_max),
            y=(cavity_y_max, cfg.outlet_length),
            z=z_span,
            allow_degenerate=True,
        ),
    }
    for name, box in optional_regions.items():
        if box is not None:
            regions[name] = box
    return regions


def build_mesh_geometry_3d(config: MeshGeometryConfig3D) -> MeshGeometryModel3D:
    """Build mesh-oriented 3D geometry model for TJ / TJC."""
    validation = validate_mesh_geometry_config(config)

    half_inlet = 0.5 * config.inlet_width
    half_outlet = 0.5 * config.outlet_width
    half_span_y = max(half_inlet, half_outlet)
    half_span_x = max(config.inlet_length, half_outlet + config.cavity_depth)
    domain = AxisAlignedBox3D(
        x=(-half_span_x, half_span_x),
        y=(-half_span_y, config.outlet_length),
        z=(0.0, config.channel_height),
    )
    fluid_components = _base_channels_3d(config)
    if config.include_cavities:
        fluid_components.update(_create_cavities_3d(config))
    boundary_patches = _create_boundary_patches_3d(config)
    wall_regions = _wall_regions_3d(config)

    for name, box in fluid_components.items():
        _validate_box(name, box)
    for name, box in boundary_patches.items():
        _validate_box(name, box)
    for name, box in wall_regions.items():
        _validate_box(name, box)
    _validate_box("domain", domain)

    return MeshGeometryModel3D(
        config=config,
        domain=domain,
        fluid_components=fluid_components,
        boundary_patches=boundary_patches,
        wall_regions=wall_regions,
        validation_warnings=validation.warnings,
    )


def points_in_union(
    points: np.ndarray, boxes: Iterable[AxisAlignedBox3D]
) -> np.ndarray:
    """Mask points that lie inside a union of axis-aligned boxes."""
    mask = np.zeros(points.shape[0], dtype=bool)
    for box in boxes:
        mask |= box.contains_points(points)
    return mask


def geometry_resolution_diagnostics_3d(
    config: MeshGeometryConfig3D,
    resolution: Tuple[int, int, int],
) -> Dict[str, float | list[str]]:
    """Estimate how mesh resolution covers key geometric features."""
    if any(value <= 0 for value in resolution):
        raise ValueError("resolution values must be positive.")

    geometry = build_mesh_geometry_3d(config)
    dx = (geometry.domain.x[1] - geometry.domain.x[0]) / resolution[0]
    dy = (geometry.domain.y[1] - geometry.domain.y[0]) / resolution[1]
    dz = (geometry.domain.z[1] - geometry.domain.z[0]) / resolution[2]
    diagnostics = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "cavity_depth_cells_x": config.cavity_depth / dx,
        "inlet_width_cells_y": config.inlet_width / dy,
        "channel_height_cells_z": config.channel_height / dz,
        "junction_width_cells_x": config.outlet_width / dx,
    }
    warnings_list: list[str] = []
    if config.include_cavities and diagnostics["cavity_depth_cells_x"] < 4.0:
        warnings_list.append("Cavity depth is under-resolved (< 4 cells in x).")
    if diagnostics["inlet_width_cells_y"] < 8.0:
        warnings_list.append("Inlet width is under-resolved (< 8 cells in y).")
    if diagnostics["channel_height_cells_z"] < 8.0:
        warnings_list.append("Channel height is under-resolved (< 8 cells in z).")
    diagnostics["warnings"] = warnings_list
    return diagnostics
