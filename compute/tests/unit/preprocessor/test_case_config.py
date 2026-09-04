from __future__ import annotations

import json

import pytest

from microfluidics.preprocessor import (
    CaseConfigError,
    case_config_from_mapping,
    case_config_to_dict,
    load_case_config,
    resolve_case_mesh_path,
)


def _valid_case() -> dict[str, object]:
    return {
        "schema_version": "case_config_v1",
        "case_id": "t-mixer",
        "mesh": {"path": "data/meshes/gmsh/t_junction.msh"},
        "zones": [
            {
                "id": "fluid",
                "kind": "volume",
                "physical_names": ["fluid"],
            },
            {
                "id": "left_inlet",
                "kind": "surface",
                "physical_names": ["left_inlet"],
            },
            {
                "id": "outlet",
                "kind": "surface",
                "physical_tags": [3],
            },
            {
                "id": "walls",
                "kind": "surface",
                "physical_names": ["walls"],
            },
        ],
        "materials": [
            {
                "id": "water",
                "zone": "fluid",
                "properties": {
                    "density_kg_per_m3": 1000.0,
                    "kinematic_viscosity_m2_per_s": 1e-6,
                },
            }
        ],
        "boundary_conditions": [
            {
                "id": "left-flow",
                "zone": "left_inlet",
                "kind": "velocity_inlet",
                "normal_speed_m_per_s": 0.15,
            },
            {
                "id": "outlet-pressure",
                "zone": "outlet",
                "kind": "pressure_outlet",
                "pressure_pa": 0.0,
            },
            {
                "id": "wall-flow",
                "zone": "walls",
                "kind": "wall",
                "wall_mode": "no_slip",
            },
            {
                "id": "wall-temperature",
                "zone": "walls",
                "kind": "robin",
                "field": "temperature",
                "alpha": 2.0,
                "beta": 1.0,
                "gamma": 600.0,
            },
        ],
        "mesh_quality": {
            "min_cell_volume_m3": 1e-20,
            "min_face_area_m2": 1e-12,
            "max_tetra_aspect_ratio": 8.0,
            "max_reported_findings": 20,
            "fail_on_warnings": True,
        },
    }


def test_case_config_parses_all_contract_sections() -> None:
    case = case_config_from_mapping(_valid_case())

    assert case.schema_version == "case_config_v1"
    assert case.mesh.path.endswith("t_junction.msh")
    assert [zone.id for zone in case.zones] == [
        "fluid",
        "left_inlet",
        "outlet",
        "walls",
    ]
    assert case.materials[0].properties["density_kg_per_m3"] == 1000.0
    assert case.boundary_conditions[0].parameters["normal_speed_m_per_s"] == 0.15
    assert case.boundary_conditions[-1].parameters["field"] == "temperature"
    assert case.mesh_quality.fail_on_warnings is True


def test_normalized_case_roundtrips_through_public_json_schema() -> None:
    case = case_config_from_mapping(_valid_case())
    normalized = case_config_to_dict(case)

    assert "parameters" not in normalized["boundary_conditions"][0]
    assert case_config_from_mapping(normalized) == case


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda case: case.update(schema_version="v2"), "schema_version"),
        (
            lambda case: case["zones"][0].update(physical_names=[]),
            "physical_names and/or physical_tags",
        ),
        (
            lambda case: case["materials"][0].update(zone="walls"),
            "must reference a volume zone",
        ),
        (
            lambda case: case["boundary_conditions"][0].update(zone="fluid"),
            "must reference a surface zone",
        ),
        (
            lambda case: case["boundary_conditions"][0].update(
                velocity_m_per_s=[1, 0, 0]
            ),
            "exactly one",
        ),
        (
            lambda case: case["boundary_conditions"][-1].update(alpha=0, beta=0),
            "cannot both be zero",
        ),
    ],
)
def test_case_config_rejects_invalid_contracts(mutate, message: str) -> None:
    payload = _valid_case()
    mutate(payload)
    with pytest.raises(CaseConfigError, match=message):
        case_config_from_mapping(payload)


def test_case_config_rejects_kind_specific_extra_fields() -> None:
    payload = _valid_case()
    payload["boundary_conditions"][1]["flux"] = 1.0

    with pytest.raises(CaseConfigError, match="invalid for pressure_outlet"):
        case_config_from_mapping(payload)


def test_load_case_config_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "case.json"
    path.write_text('{"schema_version":"case_config_v1","schema_version":"x"}')

    with pytest.raises(CaseConfigError, match="duplicate JSON key"):
        load_case_config(path)


def test_load_case_config_reads_valid_json(tmp_path) -> None:
    path = tmp_path / "case.json"
    path.write_text(json.dumps(_valid_case()), encoding="utf-8")

    loaded = load_case_config(path)
    assert loaded.case_id == "t-mixer"
    assert (
        resolve_case_mesh_path(loaded, path)
        == (tmp_path / "data/meshes/gmsh/t_junction.msh").resolve()
    )
