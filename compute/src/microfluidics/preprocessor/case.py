"""Strict JSON loader for the versioned CFD case contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from microfluidics.preprocessor.errors import CaseConfigError
from microfluidics.preprocessor.models import (
    BoundaryConditionSpec,
    CaseConfigV1,
    MaterialSpec,
    MeshQualityPolicy,
    MeshSpec,
    ZoneSpec,
)

_MATERIAL_PROPERTIES = frozenset(
    {
        "density_kg_per_m3",
        "dynamic_viscosity_pa_s",
        "kinematic_viscosity_m2_per_s",
        "thermal_conductivity_w_per_m_k",
        "specific_heat_capacity_j_per_kg_k",
        "thermal_diffusivity_m2_per_s",
        "scalar_diffusivity_m2_per_s",
    }
)
_BOUNDARY_KINDS = frozenset(
    {
        "velocity_inlet",
        "pressure_outlet",
        "wall",
        "symmetry",
        "periodic",
        "dirichlet",
        "neumann",
        "robin",
    }
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaseConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _object(
    value: object,
    label: str,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaseConfigError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise CaseConfigError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise CaseConfigError(f"{label} is missing required fields: {missing}")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise CaseConfigError(f"{label} must be an array")
    return value


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaseConfigError(f"{label} must be a non-empty string")
    return value.strip()


def _number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CaseConfigError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CaseConfigError(f"{label} must be finite")
    if positive and result <= 0.0:
        raise CaseConfigError(f"{label} must be positive")
    if non_negative and result < 0.0:
        raise CaseConfigError(f"{label} must be non-negative")
    return result


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaseConfigError(f"{label} must be an integer")
    if positive and value <= 0:
        raise CaseConfigError(f"{label} must be positive")
    return int(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CaseConfigError(f"{label} must be boolean")
    return value


def _string_array(value: object, label: str) -> tuple[str, ...]:
    values = tuple(_non_empty(item, f"{label} item") for item in _array(value, label))
    if len(set(values)) != len(values):
        raise CaseConfigError(f"{label} must contain unique values")
    return values


def _tag_array(value: object, label: str) -> tuple[int, ...]:
    values = tuple(
        _integer(item, f"{label} item", positive=True) for item in _array(value, label)
    )
    if len(set(values)) != len(values):
        raise CaseConfigError(f"{label} must contain unique values")
    return values


def _vector3(value: object, label: str) -> tuple[float, float, float]:
    values = _array(value, label)
    if len(values) != 3:
        raise CaseConfigError(f"{label} must contain exactly three values")
    return tuple(_number(item, f"{label} item") for item in values)  # type: ignore[return-value]


def _parse_zone(raw: object, index: int) -> ZoneSpec:
    label = f"zones[{index}]"
    item = _object(
        raw,
        label,
        allowed=frozenset(
            {"id", "kind", "physical_names", "physical_tags", "allow_overlap"}
        ),
        required=frozenset({"id", "kind"}),
    )
    kind = _non_empty(item["kind"], f"{label}.kind")
    if kind not in {"surface", "volume"}:
        raise CaseConfigError(f"{label}.kind must be 'surface' or 'volume'")
    names = _string_array(item.get("physical_names", []), f"{label}.physical_names")
    tags = _tag_array(item.get("physical_tags", []), f"{label}.physical_tags")
    if not names and not tags:
        raise CaseConfigError(
            f"{label} must define physical_names and/or physical_tags"
        )
    return ZoneSpec(
        id=_non_empty(item["id"], f"{label}.id"),
        kind=kind,  # type: ignore[arg-type]
        physical_names=names,
        physical_tags=tags,
        allow_overlap=_boolean(
            item.get("allow_overlap", False), f"{label}.allow_overlap"
        ),
    )


def _parse_material(raw: object, index: int) -> MaterialSpec:
    label = f"materials[{index}]"
    item = _object(
        raw,
        label,
        allowed=frozenset({"id", "zone", "properties"}),
        required=frozenset({"id", "zone", "properties"}),
    )
    properties_raw = _object(
        item["properties"],
        f"{label}.properties",
        allowed=_MATERIAL_PROPERTIES,
    )
    if not properties_raw:
        raise CaseConfigError(f"{label}.properties must not be empty")
    properties = {
        key: _number(value, f"{label}.properties.{key}", positive=True)
        for key, value in properties_raw.items()
    }
    return MaterialSpec(
        id=_non_empty(item["id"], f"{label}.id"),
        zone=_non_empty(item["zone"], f"{label}.zone"),
        properties=properties,
    )


def _parse_boundary(raw: object, index: int) -> BoundaryConditionSpec:
    label = f"boundary_conditions[{index}]"
    item = _object(
        raw,
        label,
        allowed=frozenset(
            {
                "id",
                "zone",
                "kind",
                "normal_speed_m_per_s",
                "velocity_m_per_s",
                "pressure_pa",
                "wall_mode",
                "paired_zone",
                "translation_m",
                "field",
                "value",
                "flux",
                "alpha",
                "beta",
                "gamma",
            }
        ),
        required=frozenset({"id", "zone", "kind"}),
    )
    kind = _non_empty(item["kind"], f"{label}.kind")
    if kind not in _BOUNDARY_KINDS:
        raise CaseConfigError(f"{label}.kind is unsupported: {kind!r}")

    common = {"id", "zone", "kind"}
    required: set[str]
    allowed: set[str]
    if kind == "velocity_inlet":
        required = set()
        allowed = {"normal_speed_m_per_s", "velocity_m_per_s"}
        present = allowed & set(item)
        if len(present) != 1:
            raise CaseConfigError(
                f"{label} must define exactly one of normal_speed_m_per_s or velocity_m_per_s"
            )
    elif kind == "pressure_outlet":
        required, allowed = {"pressure_pa"}, {"pressure_pa"}
    elif kind == "wall":
        required, allowed = {"wall_mode"}, {"wall_mode"}
    elif kind == "symmetry":
        required, allowed = set(), set()
    elif kind == "periodic":
        required, allowed = {"paired_zone"}, {"paired_zone", "translation_m"}
    elif kind == "dirichlet":
        required, allowed = {"field", "value"}, {"field", "value"}
    elif kind == "neumann":
        required, allowed = {"field", "flux"}, {"field", "flux"}
    else:
        required = {"field", "alpha", "beta", "gamma"}
        allowed = set(required)

    extra = sorted(set(item) - common - allowed)
    missing = sorted(required - set(item))
    if extra:
        raise CaseConfigError(f"{label} has fields invalid for {kind}: {extra}")
    if missing:
        raise CaseConfigError(
            f"{label} is missing fields required for {kind}: {missing}"
        )

    parameters: dict[str, object] = {}
    for key in allowed & set(item):
        value = item[key]
        if key in {"velocity_m_per_s", "translation_m"}:
            parameters[key] = _vector3(value, f"{label}.{key}")
        elif key in {"field", "paired_zone", "wall_mode"}:
            parameters[key] = _non_empty(value, f"{label}.{key}")
        else:
            parameters[key] = _number(value, f"{label}.{key}")
    if (
        kind == "velocity_inlet"
        and "normal_speed_m_per_s" in parameters
        and float(parameters["normal_speed_m_per_s"]) < 0.0
    ):
        raise CaseConfigError(f"{label}.normal_speed_m_per_s must be non-negative")
    if kind == "wall" and parameters["wall_mode"] not in {"slip", "no_slip"}:
        raise CaseConfigError(f"{label}.wall_mode must be 'slip' or 'no_slip'")
    if (
        kind == "robin"
        and abs(float(parameters["alpha"])) + abs(float(parameters["beta"])) == 0.0
    ):
        raise CaseConfigError(f"{label} Robin alpha and beta cannot both be zero")
    return BoundaryConditionSpec(
        id=_non_empty(item["id"], f"{label}.id"),
        zone=_non_empty(item["zone"], f"{label}.zone"),
        kind=kind,  # type: ignore[arg-type]
        parameters=parameters,
    )


def _parse_quality(raw: object) -> MeshQualityPolicy:
    item = _object(
        raw,
        "mesh_quality",
        allowed=frozenset(
            {
                "min_cell_volume_m3",
                "min_face_area_m2",
                "max_tetra_aspect_ratio",
                "max_reported_findings",
                "fail_on_warnings",
            }
        ),
    )
    return MeshQualityPolicy(
        min_cell_volume_m3=_number(
            item.get("min_cell_volume_m3", 1e-18),
            "mesh_quality.min_cell_volume_m3",
            non_negative=True,
        ),
        min_face_area_m2=_number(
            item.get("min_face_area_m2", 0.0),
            "mesh_quality.min_face_area_m2",
            non_negative=True,
        ),
        max_tetra_aspect_ratio=_number(
            item.get("max_tetra_aspect_ratio", 12.0),
            "mesh_quality.max_tetra_aspect_ratio",
            positive=True,
        ),
        max_reported_findings=_integer(
            item.get("max_reported_findings", 100),
            "mesh_quality.max_reported_findings",
            positive=True,
        ),
        fail_on_warnings=_boolean(
            item.get("fail_on_warnings", False),
            "mesh_quality.fail_on_warnings",
        ),
    )


def _require_unique(values: list[str], label: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise CaseConfigError(f"duplicate {label}s: {duplicates}")


def case_config_from_mapping(raw: Mapping[str, object]) -> CaseConfigV1:
    top = _object(
        raw,
        "case config",
        allowed=frozenset(
            {
                "schema_version",
                "case_id",
                "mesh",
                "zones",
                "materials",
                "boundary_conditions",
                "mesh_quality",
            }
        ),
        required=frozenset(
            {"schema_version", "case_id", "mesh", "zones", "boundary_conditions"}
        ),
    )
    schema = _non_empty(top["schema_version"], "schema_version")
    if schema != "case_config_v1":
        raise CaseConfigError("schema_version must be 'case_config_v1'")
    mesh_raw = _object(
        top["mesh"],
        "mesh",
        allowed=frozenset({"path"}),
        required=frozenset({"path"}),
    )
    zones = tuple(
        _parse_zone(item, index)
        for index, item in enumerate(_array(top["zones"], "zones"))
    )
    if not zones:
        raise CaseConfigError("zones must not be empty")
    materials = tuple(
        _parse_material(item, index)
        for index, item in enumerate(_array(top.get("materials", []), "materials"))
    )
    boundaries = tuple(
        _parse_boundary(item, index)
        for index, item in enumerate(
            _array(top["boundary_conditions"], "boundary_conditions")
        )
    )
    if not boundaries:
        raise CaseConfigError("boundary_conditions must not be empty")
    _require_unique([zone.id for zone in zones], "zone id")
    _require_unique([material.id for material in materials], "material id")
    _require_unique([bc.id for bc in boundaries], "boundary condition id")
    zone_by_id = {zone.id: zone for zone in zones}
    for material in materials:
        zone = zone_by_id.get(material.zone)
        if zone is None:
            raise CaseConfigError(
                f"material {material.id!r} references unknown zone {material.zone!r}"
            )
        if zone.kind != "volume":
            raise CaseConfigError(
                f"material {material.id!r} must reference a volume zone"
            )
    for bc in boundaries:
        zone = zone_by_id.get(bc.zone)
        if zone is None:
            raise CaseConfigError(
                f"boundary condition {bc.id!r} references unknown zone {bc.zone!r}"
            )
        if zone.kind != "surface":
            raise CaseConfigError(
                f"boundary condition {bc.id!r} must reference a surface zone"
            )
        paired = bc.parameters.get("paired_zone")
        if paired is not None:
            paired_zone = zone_by_id.get(str(paired))
            if paired_zone is None or paired_zone.kind != "surface":
                raise CaseConfigError(
                    f"periodic boundary {bc.id!r} references invalid paired_zone {paired!r}"
                )
            if paired_zone.id == zone.id:
                raise CaseConfigError(
                    f"periodic boundary {bc.id!r} cannot pair a zone with itself"
                )

    return CaseConfigV1(
        schema_version="case_config_v1",
        case_id=_non_empty(top["case_id"], "case_id"),
        mesh=MeshSpec(path=_non_empty(mesh_raw["path"], "mesh.path")),
        zones=zones,
        materials=materials,
        boundary_conditions=boundaries,
        mesh_quality=_parse_quality(top.get("mesh_quality", {})),
    )


def load_case_config(path: str | Path) -> CaseConfigV1:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except CaseConfigError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise CaseConfigError(f"cannot read case config {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise CaseConfigError("case config must contain a JSON object")
    return case_config_from_mapping(payload)


def resolve_case_mesh_path(case: CaseConfigV1, case_config_path: str | Path) -> Path:
    """Resolve ``mesh.path`` relative to the case JSON that declares it."""

    config_path = Path(case_config_path).resolve()
    mesh_path = Path(case.mesh.path)
    return (
        mesh_path.resolve()
        if mesh_path.is_absolute()
        else (config_path.parent / mesh_path).resolve()
    )


def case_config_to_dict(case: CaseConfigV1) -> dict[str, object]:
    """Return a deterministic JSON-ready representation for provenance artifacts."""

    return {
        "schema_version": case.schema_version,
        "case_id": case.case_id,
        "mesh": asdict(case.mesh),
        "zones": [
            {
                "id": zone.id,
                "kind": zone.kind,
                "physical_names": list(zone.physical_names),
                "physical_tags": list(zone.physical_tags),
                "allow_overlap": zone.allow_overlap,
            }
            for zone in case.zones
        ],
        "materials": [
            {
                "id": material.id,
                "zone": material.zone,
                "properties": dict(material.properties),
            }
            for material in case.materials
        ],
        "boundary_conditions": [
            {
                "id": condition.id,
                "zone": condition.zone,
                "kind": condition.kind,
                **{
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in condition.parameters.items()
                },
            }
            for condition in case.boundary_conditions
        ],
        "mesh_quality": asdict(case.mesh_quality),
    }
