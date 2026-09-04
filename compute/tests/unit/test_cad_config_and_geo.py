from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microfluidics.cad.config import (
    CadMeshConfig,
    CadParameters,
    GeometryConfigError,
    GeometryPipelineConfig,
    GeometrySourceConfig,
    load_geometry_pipeline_config,
)
from microfluidics.cad.geo import build_geo_source, procedural_boundary_selectors
from microfluidics.cad.pipeline import (
    CadArtifacts,
    CadPipelineValidationError,
    MESH_BBOX_ABSOLUTE_TOLERANCE,
    MESH_BBOX_RELATIVE_TOLERANCE,
    MESH_VOLUME_ABSOLUTE_TOLERANCE,
    MESH_VOLUME_RELATIVE_TOLERANCE,
    _assert_mesh_matches_cad,
    _assert_physical_groups,
    _validate_quality,
    assert_msh41_binary,
)
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.mesh.mesh_builder import build_mesh_from_geometry_config


def _regular_tetra_mesh() -> ImportedTetraMesh:
    return build_imported_tetra_mesh(
        source_path=Path("regular.msh"),
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0e-3, 0.0, 0.0],
                [0.0, 2.0e-3, 0.0],
                [0.0, 0.0, 3.0e-3],
            ],
            dtype=np.float64,
        ),
        tetrahedra=np.asarray([[0, 1, 2, 3]], dtype=np.int64),
        boundary_triangles=np.asarray(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            dtype=np.int64,
        ),
        boundary_face_tags=np.asarray([2, 3, 4, 5], dtype=np.int32),
        field_data={
            "fluid": np.asarray([1, 3], dtype=np.int32),
            "left_inlet": np.asarray([2, 2], dtype=np.int32),
            "right_inlet": np.asarray([3, 2], dtype=np.int32),
            "outlet": np.asarray([4, 2], dtype=np.int32),
            "walls": np.asarray([5, 2], dtype=np.int32),
        },
        volume_tag_per_cell=np.asarray([1], dtype=np.int32),
    )


def _cad_artifacts_for_mesh(mesh: ImportedTetraMesh) -> CadArtifacts:
    bbox = tuple(
        float(value)
        for value in np.concatenate(
            (np.min(mesh.points, axis=0), np.max(mesh.points, axis=0))
        )
    )
    return CadArtifacts(
        brep_path=Path("model.brep"),
        step_path=Path("display.step"),
        geo_path=Path("geometry_generated.geo"),
        bbox=bbox,
        volume=float(np.sum(mesh.cell_volumes)),
        step_roundtrip_bbox=bbox,
        step_roundtrip_volume=float(np.sum(mesh.cell_volumes)),
    )


def test_cad_parameters_report_expected_tjunction_bbox() -> None:
    plain = CadParameters()
    cavities = CadParameters(
        inlet_length=1.1e-3,
        outlet_width=2.0e-3,
        include_cavities=True,
        cavity_depth=0.5e-3,
        cavity_length=1.0e-3,
        cavity_offset_from_junction=1.0e-3,
    )

    assert plain.expected_bbox() == pytest.approx(
        (-0.012, -0.0005, 0.0, 0.012, 0.02, 0.001)
    )
    assert cavities.expected_bbox() == pytest.approx(
        (-0.0015, -0.0005, 0.0, 0.0015, 0.02, 0.001)
    )


def test_cad_parameters_report_exact_unfilleted_union_volume() -> None:
    plain = CadParameters()
    cavities = CadParameters(
        include_cavities=True,
        cavity_depth=0.5e-3,
        cavity_length=2.0e-3,
        cavity_offset_from_junction=2.0e-3,
    )

    assert plain.expected_unfilleted_volume() == pytest.approx(6.3e-8)
    assert cavities.expected_unfilleted_volume() == pytest.approx(6.5e-8)


def test_cad_parameters_reject_invalid_fillet_and_cavity_window() -> None:
    with pytest.raises(GeometryConfigError, match="fillet_radius"):
        CadParameters(fillet_radius=0.0005).validate()
    with pytest.raises(GeometryConfigError, match="cavity interval"):
        CadParameters(
            include_cavities=True,
            cavity_offset_from_junction=0.019,
            cavity_length=0.002,
        ).validate()
    with pytest.raises(GeometryConfigError, match="cavity interval"):
        CadParameters(
            include_cavities=True,
            cavity_offset_from_junction=0.018,
            cavity_length=0.002,
        ).validate()


def test_cad_mesh_quality_rejects_nearly_coplanar_sliver() -> None:
    points = np.asarray(
        [
            [1.0e-3, 0.0, 0.0],
            [-1.0e-3, 0.0, 0.0],
            [0.0, 1.0e-3, 1.0e-7],
            [0.0, -1.0e-3, 1.0e-7],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
        dtype=np.int64,
    )
    mesh = build_imported_tetra_mesh(
        source_path=Path("sliver.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=np.asarray([2, 3, 4, 5], dtype=np.int32),
        field_data={
            "fluid": np.asarray([1, 3], dtype=np.int32),
            "left_inlet": np.asarray([2, 2], dtype=np.int32),
            "right_inlet": np.asarray([3, 2], dtype=np.int32),
            "outlet": np.asarray([4, 2], dtype=np.int32),
            "walls": np.asarray([5, 2], dtype=np.int32),
        },
        volume_tag_per_cell=np.asarray([1], dtype=np.int32),
    )

    quality = mesh.diagnostics["tetra_quality"]
    assert mesh.cell_volumes[0] > 1.0e-18
    assert quality["tetra_aspect_ratio_proxy"]["max"] < 2.0
    assert quality["tetra_mean_ratio"]["min"] < 0.01
    with pytest.raises(CadPipelineValidationError, match="mean ratio"):
        _validate_quality(mesh, CadMeshConfig(min_tetra_mean_ratio=0.05))


def test_cad_mesh_geometry_audit_accepts_matching_bbox_and_volume() -> None:
    mesh = _regular_tetra_mesh()
    artifacts = _cad_artifacts_for_mesh(mesh)

    audit = _assert_mesh_matches_cad(mesh, artifacts)

    assert audit["mesh_bbox"] == pytest.approx(artifacts.bbox)
    assert audit["mesh_tetrahedral_volume_m3"] == pytest.approx(artifacts.volume)
    assert audit["bbox_relative_tolerance"] == MESH_BBOX_RELATIVE_TOLERANCE
    assert audit["bbox_absolute_tolerance_m"] == MESH_BBOX_ABSOLUTE_TOLERANCE
    assert audit["volume_relative_tolerance"] == MESH_VOLUME_RELATIVE_TOLERANCE
    assert audit["volume_absolute_tolerance_m3"] == MESH_VOLUME_ABSOLUTE_TOLERANCE


def test_cad_mesh_geometry_audit_rejects_scale_mismatch() -> None:
    mesh = _regular_tetra_mesh()
    artifacts = _cad_artifacts_for_mesh(mesh)
    mesh.points *= 1_000.0

    with pytest.raises(CadPipelineValidationError, match="mesh bbox differs"):
        _assert_mesh_matches_cad(mesh, artifacts)


def test_cad_mesh_geometry_audit_rejects_incomplete_tetrahedral_volume() -> None:
    mesh = _regular_tetra_mesh()
    artifacts = _cad_artifacts_for_mesh(mesh)
    mesh.cell_volumes *= 0.5

    with pytest.raises(CadPipelineValidationError, match="tetrahedral volume differs"):
        _assert_mesh_matches_cad(mesh, artifacts)


def test_cad_mesh_quality_rejects_unresolved_reconstructed_boundary_faces() -> None:
    mesh = _regular_tetra_mesh()
    mesh.boundary_unresolved_faces = np.asarray([0], dtype=np.int64)

    with pytest.raises(CadPipelineValidationError, match="unresolved reconstructed"):
        _validate_quality(mesh, CadMeshConfig())


def test_cad_mesh_quality_rejects_unmatched_imported_boundary_triangles() -> None:
    mesh = _regular_tetra_mesh()
    mesh.diagnostics["unmatched_imported_boundary_triangles"] = 1

    with pytest.raises(CadPipelineValidationError, match="unmatched imported"):
        _validate_quality(mesh, CadMeshConfig())


def test_cad_mesh_physical_groups_require_every_tetrahedron_in_fluid() -> None:
    mesh = _regular_tetra_mesh()
    mesh.volume_tag_per_cell[0] = -1

    with pytest.raises(CadPipelineValidationError, match="fluid=1"):
        _assert_physical_groups(mesh)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            CadParameters(inlet_length=0.5e-3, outlet_width=1.0e-3),
            "inlet_length",
        ),
        (
            CadParameters(inlet_length=0.4e-3, outlet_width=1.0e-3),
            "inlet_length",
        ),
        (
            CadParameters(outlet_length=0.5e-3, inlet_width=1.0e-3),
            "outlet_length",
        ),
        (
            CadParameters(outlet_length=0.4e-3, inlet_width=1.0e-3),
            "outlet_length",
        ),
    ],
)
def test_channel_lengths_keep_terminal_planes_exposed(
    parameters: CadParameters,
    message: str,
) -> None:
    with pytest.raises(GeometryConfigError, match=message):
        parameters.validate()


def test_load_json_geometry_config_resolves_external_step_relative_to_config(
    tmp_path: Path,
) -> None:
    step_path = tmp_path / "input.step"
    step_path.write_text("fixture", encoding="utf-8")
    payload = {
        "schema_version": "geometry_pipeline_v1",
        "source": {
            "kind": "external_step",
            "step_path": "input.step",
            "boundary_selectors": {
                "left_inlet": [0, 0, 0, 0, 1, 1],
                "right_inlet": [2, 0, 0, 2, 1, 1],
                "outlet": [0, 2, 0, 2, 2, 1],
            },
        },
    }
    config_path = tmp_path / "geometry.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_geometry_pipeline_config(config_path)

    assert config.source.kind == "external_step"
    assert config.source.step_path == step_path.resolve()


def test_load_geometry_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "geometry.yaml"
    path.write_text(
        "schema_version: geometry_pipeline_v1\nsource:\n  kind: procedural_cad\nworkaround: true\n",
        encoding="utf-8",
    )
    with pytest.raises(GeometryConfigError, match="unknown fields"):
        load_geometry_pipeline_config(path)


def test_file_config_requires_explicit_source_kind(tmp_path: Path) -> None:
    path = tmp_path / "geometry.json"
    path.write_text('{"schema_version":"geometry_pipeline_v1"}', encoding="utf-8")
    with pytest.raises(GeometryConfigError, match="source is required"):
        load_geometry_pipeline_config(path)


def test_generated_geo_imports_brep_and_owns_all_physical_groups() -> None:
    params = CadParameters(include_cavities=True)
    source = build_geo_source(
        cad_file_name="model.brep",
        boundary_selectors=procedural_boundary_selectors(params),
        mesh=CadMeshConfig(element_size=0.0003),
    )

    assert 'fluid() = ShapeFromFile("model.brep");' in source
    assert 'Physical Volume("fluid", 1)' in source
    for name in ("left_inlet", "right_inlet", "outlet", "walls"):
        assert f'Physical Surface("{name}"' in source
    assert "walls() -= {leftInlet(), rightInlet(), outlet()};" in source
    assert "Mesh.MshFileVersion = 4.1;" in source
    assert "Mesh.Binary = 1;" in source


@pytest.mark.parametrize(
    ("header", "valid"),
    [
        (b"$MeshFormat\n4.1 1 8\n", True),
        (b"$MeshFormat\n4.1 1 4\n", False),
        (b"$MeshFormat\n4.1 1 8 extra\n", False),
        (b"$MeshFormat\n4.1 0 8\n", False),
        (b"$MeshFormat\n2.2 1 8\n", False),
    ],
)
def test_msh41_binary_contract(tmp_path: Path, header: bytes, valid: bool) -> None:
    path = tmp_path / "mesh.msh"
    path.write_bytes(header)
    if valid:
        assert_msh41_binary(path)
    else:
        with pytest.raises(CadPipelineValidationError, match="MSH 4.1 Binary"):
            assert_msh41_binary(path)


def test_legacy_boxes_requires_explicit_source_and_still_builds(tmp_path: Path) -> None:
    config = GeometryPipelineConfig(
        source=GeometrySourceConfig(kind="legacy_boxes"),
        legacy_resolution=(12, 8, 4),
    )

    result = build_mesh_from_geometry_config(config, tmp_path)

    assert result.source_kind == "legacy_boxes"
    assert result.cad_mesh is None
    assert result.legacy_mesh is not None
    assert result.legacy_mesh.cells.shape[0] > 0


def test_legacy_boxes_rejects_unsupported_fillet() -> None:
    config = GeometryPipelineConfig(
        source=GeometrySourceConfig(kind="legacy_boxes"),
        geometry=CadParameters(fillet_radius=0.00005),
    )
    with pytest.raises(GeometryConfigError, match="does not support fillet_radius"):
        config.validate()


def test_cad_failure_is_not_replaced_by_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("CAD failed")

    monkeypatch.setattr("microfluidics.cad.pipeline.generate_cad_mesh", fail)
    config = GeometryPipelineConfig(source=GeometrySourceConfig(kind="procedural_cad"))

    with pytest.raises(RuntimeError, match="CAD failed"):
        build_mesh_from_geometry_config(config, tmp_path)
