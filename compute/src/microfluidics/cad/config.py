"""Strict configuration contract for CAD-backed geometry and meshing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


PHYSICAL_GROUP_NAMES = ("fluid", "left_inlet", "right_inlet", "outlet", "walls")
SOURCE_KINDS = ("procedural_cad", "external_step", "legacy_boxes")


class GeometryConfigError(ValueError):
    """Raised when a geometry pipeline configuration is invalid."""


@dataclass(frozen=True)
class CadParameters:
    inlet_width: float = 1.0e-3
    outlet_width: float = 2.0e-3
    channel_height: float = 1.0e-3
    inlet_length: float = 12.0e-3
    outlet_length: float = 20.0e-3
    include_cavities: bool = False
    cavity_depth: float = 0.49875e-3
    cavity_length: float = 1.995e-3
    cavity_offset_from_junction: float = 0.5e-3
    fillet_radius: float = 0.0

    def validate(self) -> None:
        positive = (
            "inlet_width",
            "outlet_width",
            "channel_height",
            "inlet_length",
            "outlet_length",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise GeometryConfigError(
                    f"geometry.{name} must be finite and positive"
                )
        if self.inlet_length <= 0.5 * self.outlet_width:
            raise GeometryConfigError(
                "geometry.inlet_length must be greater than half of "
                "geometry.outlet_width so both inlet terminals remain exposed"
            )
        if self.outlet_length <= 0.5 * self.inlet_width:
            raise GeometryConfigError(
                "geometry.outlet_length must be greater than half of "
                "geometry.inlet_width so the outlet terminal remains exposed"
            )
        if not math.isfinite(self.fillet_radius) or self.fillet_radius < 0:
            raise GeometryConfigError(
                "geometry.fillet_radius must be finite and non-negative"
            )
        if self.include_cavities:
            for name in ("cavity_depth", "cavity_length"):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise GeometryConfigError(
                        f"geometry.{name} must be finite and positive when cavities are enabled"
                    )
            cavity_end = self.cavity_offset_from_junction + self.cavity_length
            if (
                not math.isfinite(self.cavity_offset_from_junction)
                or self.cavity_offset_from_junction < 0
                or cavity_end >= self.outlet_length
                or math.isclose(cavity_end, self.outlet_length, rel_tol=1e-12)
            ):
                raise GeometryConfigError(
                    "geometry cavity interval must end before the outlet terminal"
                )
        if self.fillet_radius > 0:
            limiting_dimensions = [
                self.inlet_width,
                self.outlet_width,
                self.channel_height,
            ]
            if self.include_cavities:
                limiting_dimensions.extend((self.cavity_depth, self.cavity_length))
            if self.fillet_radius >= 0.5 * min(limiting_dimensions):
                raise GeometryConfigError(
                    "geometry.fillet_radius must be smaller than half of every local feature"
                )

    def expected_bbox(self) -> tuple[float, float, float, float, float, float]:
        half_outlet = 0.5 * self.outlet_width
        x_extent = self.inlet_length
        if self.include_cavities:
            x_extent = max(x_extent, half_outlet + self.cavity_depth)
        return (
            -x_extent,
            -0.5 * self.inlet_width,
            0.0,
            x_extent,
            self.outlet_length,
            self.channel_height,
        )

    def expected_unfilleted_volume(self) -> float:
        """Return the exact union volume of the configured box primitives."""

        half_inlet = 0.5 * self.inlet_width
        half_outlet = 0.5 * self.outlet_width
        rectangles = [
            (-self.inlet_length, 0.0, -half_inlet, half_inlet),
            (0.0, self.inlet_length, -half_inlet, half_inlet),
            (-half_outlet, half_outlet, 0.0, self.outlet_length),
        ]
        if self.include_cavities:
            cavity_ymax = self.cavity_offset_from_junction + self.cavity_length
            rectangles.extend(
                (
                    (
                        -half_outlet - self.cavity_depth,
                        -half_outlet,
                        self.cavity_offset_from_junction,
                        cavity_ymax,
                    ),
                    (
                        half_outlet,
                        half_outlet + self.cavity_depth,
                        self.cavity_offset_from_junction,
                        cavity_ymax,
                    ),
                )
            )

        x_coordinates = sorted(
            {value for rectangle in rectangles for value in rectangle[:2]}
        )
        area = 0.0
        for xmin, xmax in zip(x_coordinates, x_coordinates[1:]):
            midpoint = 0.5 * (xmin + xmax)
            intervals = sorted(
                (ymin, ymax)
                for rect_xmin, rect_xmax, ymin, ymax in rectangles
                if rect_xmin < midpoint < rect_xmax
            )
            if not intervals:
                continue
            covered_y = 0.0
            current_min, current_max = intervals[0]
            for interval_min, interval_max in intervals[1:]:
                if interval_min > current_max:
                    covered_y += current_max - current_min
                    current_min, current_max = interval_min, interval_max
                else:
                    current_max = max(current_max, interval_max)
            covered_y += current_max - current_min
            area += (xmax - xmin) * covered_y
        return area * self.channel_height


@dataclass(frozen=True)
class BoundaryBoxSelector:
    """Axis-aligned selection box used by generated Gmsh GEO source."""

    bounds: tuple[float, float, float, float, float, float]

    def validate(self, name: str) -> None:
        if len(self.bounds) != 6 or not all(math.isfinite(v) for v in self.bounds):
            raise GeometryConfigError(
                f"source.boundary_selectors.{name} must contain six finite values"
            )
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounds
        if xmin > xmax or ymin > ymax or zmin > zmax:
            raise GeometryConfigError(
                f"source.boundary_selectors.{name} has reversed bounds"
            )


@dataclass(frozen=True)
class GeometrySourceConfig:
    kind: str
    step_path: Path | None = None
    boundary_selectors: Mapping[str, BoundaryBoxSelector] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in SOURCE_KINDS:
            raise GeometryConfigError(
                f"source.kind must be one of {SOURCE_KINDS}, got {self.kind!r}"
            )
        if self.kind == "external_step":
            if self.step_path is None:
                raise GeometryConfigError(
                    "source.step_path is required for external_step"
                )
            if not self.step_path.is_file():
                raise GeometryConfigError(
                    f"external STEP does not exist: {self.step_path}"
                )
            expected = set(PHYSICAL_GROUP_NAMES) - {"fluid", "walls"}
            actual = set(self.boundary_selectors)
            if actual != expected:
                raise GeometryConfigError(
                    "external_step boundary_selectors must define exactly "
                    f"{sorted(expected)}, got {sorted(actual)}"
                )
        elif self.step_path is not None or self.boundary_selectors:
            raise GeometryConfigError(
                "step_path and boundary_selectors are only valid for external_step"
            )
        for name, selector in self.boundary_selectors.items():
            selector.validate(name)


@dataclass(frozen=True)
class CadMeshConfig:
    element_size: float = 0.2e-3
    optimize: bool = True
    timeout_seconds: float = 300.0
    max_aspect_ratio: float = 12.0
    min_tetra_mean_ratio: float = 0.05
    min_positive_volume: float = 1e-18

    def validate(self) -> None:
        for name in (
            "element_size",
            "timeout_seconds",
            "max_aspect_ratio",
            "min_tetra_mean_ratio",
            "min_positive_volume",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise GeometryConfigError(f"mesh.{name} must be finite and positive")
        if self.min_tetra_mean_ratio > 1.0:
            raise GeometryConfigError("mesh.min_tetra_mean_ratio must be at most 1")


@dataclass(frozen=True)
class GeometryPipelineConfig:
    source: GeometrySourceConfig
    geometry: CadParameters = field(default_factory=CadParameters)
    mesh: CadMeshConfig = field(default_factory=CadMeshConfig)
    legacy_resolution: tuple[int, int, int] = (220, 140, 28)

    def validate(self) -> None:
        self.source.validate()
        self.geometry.validate()
        self.mesh.validate()
        if self.source.kind == "legacy_boxes" and self.geometry.fillet_radius > 0:
            raise GeometryConfigError(
                "legacy_boxes does not support fillet_radius; select procedural_cad"
            )
        if len(self.legacy_resolution) != 3 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in self.legacy_resolution
        ):
            raise GeometryConfigError(
                "legacy_resolution must contain three positive integers"
            )


def _reject_unknown(raw: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise GeometryConfigError(f"{label} has unknown fields: {unknown}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GeometryConfigError(f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GeometryConfigError(f"{label} must be finite")
    return result


def geometry_pipeline_config_from_mapping(
    raw: Mapping[str, object], *, base_directory: Path
) -> GeometryPipelineConfig:
    _reject_unknown(
        raw,
        {"schema_version", "source", "geometry", "mesh", "legacy_resolution"},
        "geometry config",
    )
    if raw.get("schema_version") != "geometry_pipeline_v1":
        raise GeometryConfigError("schema_version must be 'geometry_pipeline_v1'")

    if "source" not in raw:
        raise GeometryConfigError("source is required")
    source_raw = _mapping(raw.get("source", {}), "source")
    _reject_unknown(source_raw, {"kind", "step_path", "boundary_selectors"}, "source")
    if "kind" not in source_raw or not isinstance(source_raw["kind"], str):
        raise GeometryConfigError("source.kind is required and must be a string")
    kind = source_raw["kind"]
    raw_step_path = source_raw.get("step_path")
    step_path = None
    if raw_step_path is not None:
        if not isinstance(raw_step_path, str) or not raw_step_path.strip():
            raise GeometryConfigError("source.step_path must be a non-empty string")
        step_path = (base_directory / raw_step_path).resolve()
    selectors_raw = _mapping(
        source_raw.get("boundary_selectors", {}), "source.boundary_selectors"
    )
    selectors: dict[str, BoundaryBoxSelector] = {}
    for name, value in selectors_raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise GeometryConfigError(
                f"source.boundary_selectors.{name} must be an array of six numbers"
            )
        try:
            bounds = tuple(
                _finite_number(item, f"source.boundary_selectors.{name}")
                for item in value
            )
        except GeometryConfigError as exc:
            raise GeometryConfigError(
                f"source.boundary_selectors.{name} must contain numbers"
            ) from exc
        selectors[str(name)] = BoundaryBoxSelector(bounds=bounds)  # type: ignore[arg-type]
    source = GeometrySourceConfig(
        kind=kind,
        step_path=step_path,
        boundary_selectors=selectors,
    )

    geometry_raw = _mapping(raw.get("geometry", {}), "geometry")
    geometry_fields = set(CadParameters.__dataclass_fields__)
    _reject_unknown(geometry_raw, geometry_fields, "geometry")
    geometry_values: dict[str, object] = {}
    for name, value in geometry_raw.items():
        if name == "include_cavities":
            if not isinstance(value, bool):
                raise GeometryConfigError("geometry.include_cavities must be boolean")
            geometry_values[name] = value
        else:
            geometry_values[name] = _finite_number(value, f"geometry.{name}")
    geometry = CadParameters(**geometry_values)  # type: ignore[arg-type]

    mesh_raw = _mapping(raw.get("mesh", {}), "mesh")
    mesh_fields = set(CadMeshConfig.__dataclass_fields__)
    _reject_unknown(mesh_raw, mesh_fields, "mesh")
    mesh_values: dict[str, object] = {}
    for name, value in mesh_raw.items():
        if name == "optimize":
            if not isinstance(value, bool):
                raise GeometryConfigError("mesh.optimize must be boolean")
            mesh_values[name] = value
        else:
            mesh_values[name] = _finite_number(value, f"mesh.{name}")
    mesh = CadMeshConfig(**mesh_values)  # type: ignore[arg-type]

    resolution_raw = raw.get("legacy_resolution", (220, 140, 28))
    if not isinstance(resolution_raw, (list, tuple)) or len(resolution_raw) != 3:
        raise GeometryConfigError(
            "legacy_resolution must be an array of three integers"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in resolution_raw
    ):
        raise GeometryConfigError("legacy_resolution must contain integers")

    config = GeometryPipelineConfig(
        source=source,
        geometry=geometry,
        mesh=mesh,
        legacy_resolution=tuple(resolution_raw),  # type: ignore[arg-type]
    )
    config.validate()
    return config


def load_geometry_pipeline_config(path: str | Path) -> GeometryPipelineConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GeometryConfigError(
            f"cannot read geometry config {source_path}: {exc}"
        ) from exc
    try:
        if source_path.suffix.lower() == ".json":
            payload = json.loads(text)
        elif source_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            raise GeometryConfigError("geometry config must be JSON or YAML")
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise GeometryConfigError(
            f"cannot parse geometry config {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise GeometryConfigError("geometry config must contain an object")
    return geometry_pipeline_config_from_mapping(
        payload,
        base_directory=source_path.parent,
    )


__all__ = [
    "BoundaryBoxSelector",
    "CadMeshConfig",
    "CadParameters",
    "GeometryConfigError",
    "GeometryPipelineConfig",
    "GeometrySourceConfig",
    "PHYSICAL_GROUP_NAMES",
    "SOURCE_KINDS",
    "geometry_pipeline_config_from_mapping",
    "load_geometry_pipeline_config",
]
