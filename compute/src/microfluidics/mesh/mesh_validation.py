"""Validation helpers for mesh-native geometry and solver contracts."""

from dataclasses import dataclass

from microfluidics.mesh.mesh_config import MeshGeometryConfig3D


@dataclass(frozen=True)
class MeshGeometryValidationResult:
    warnings: tuple[str, ...]


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _require_less(name: str, lower: float, upper_name: str, upper: float) -> None:
    if lower >= upper:
        raise ValueError(f"{name} must be < {upper_name}, got {lower} >= {upper}.")


def validate_mesh_geometry_config(
    config: MeshGeometryConfig3D,
) -> MeshGeometryValidationResult:
    """Validate geometry contract and provide non-fatal warnings.

    Strict failures raise ValueError with explicit messages.
    """
    _require_positive("inlet_width", config.inlet_width)
    _require_positive("outlet_width", config.outlet_width)
    _require_positive("channel_height", config.channel_height)
    _require_positive("inlet_length", config.inlet_length)
    _require_positive("outlet_length", config.outlet_length)
    _require_positive("patch_depth", config.patch_depth)
    _require_positive("hydraulic_diameter", config.hydraulic_diameter)
    _require_less(
        "patch_depth", config.patch_depth, "inlet_length", config.inlet_length
    )
    _require_less(
        "patch_depth", config.patch_depth, "outlet_length", config.outlet_length
    )

    warnings: list[str] = []
    if config.include_cavities:
        _require_positive("cavity_depth", config.cavity_depth)
        _require_positive("cavity_length", config.cavity_length)
        if config.cavity_offset_from_junction < 0:
            raise ValueError(
                "cavity_offset_from_junction must be non-negative, "
                f"got {config.cavity_offset_from_junction}."
            )
        cavity_y_max = config.cavity_offset_from_junction + config.cavity_length
        if cavity_y_max <= 0:
            raise ValueError(
                "Cavity window is fully outside mixing channel (y<=0). "
                f"offset={config.cavity_offset_from_junction}, length={config.cavity_length}."
            )
        if config.cavity_offset_from_junction >= config.outlet_length:
            raise ValueError(
                "cavity_offset_from_junction must be < outlet_length to intersect mixing channel."
            )
        if cavity_y_max > config.outlet_length:
            warnings.append(
                "Cavity extends beyond outlet_length; upper cavity wall segment will be clipped."
            )

        half_inlet = 0.5 * config.inlet_width
        if config.cavity_offset_from_junction < half_inlet:
            warnings.append(
                "cavity_offset_from_junction is below inlet half-width; "
                "some pre-cavity wall subregions can be degenerate and will be skipped."
            )
    return MeshGeometryValidationResult(warnings=tuple(warnings))
