from __future__ import annotations

import re
import shutil
from pathlib import Path

import meshio
import numpy as np
import pytest

from microfluidics.cad.config import (
    CadMeshConfig,
    CadParameters,
    GeometryPipelineConfig,
    GeometrySourceConfig,
    load_geometry_pipeline_config,
)
from microfluidics.cad.geo import procedural_boundary_selectors
from microfluidics.cad.occ import build_tjunction, read_step
from microfluidics.cad.pipeline import (
    MESH_BBOX_ABSOLUTE_TOLERANCE,
    MESH_BBOX_RELATIVE_TOLERANCE,
    MESH_VOLUME_ABSOLUTE_TOLERANCE,
    MESH_VOLUME_RELATIVE_TOLERANCE,
    generate_cad_artifacts,
    generate_cad_mesh,
)
from microfluidics.gmsh import GmshCli


pytestmark = pytest.mark.integration


def _require_occ() -> None:
    pytest.importorskip("OCC.Core")


def test_pythonocc_tjunction_is_one_valid_solid_with_expected_bbox_and_step_roundtrip(
    tmp_path: Path,
) -> None:
    _require_occ()
    params = CadParameters(
        inlet_length=0.003,
        outlet_length=0.004,
        include_cavities=True,
        cavity_depth=0.0004,
        cavity_length=0.001,
        cavity_offset_from_junction=0.0015,
        fillet_radius=0.00005,
    )
    config = GeometryPipelineConfig(
        source=GeometrySourceConfig(kind="procedural_cad"),
        geometry=params,
    )

    shape = build_tjunction(params)
    unfilleted_params = CadParameters(
        inlet_length=0.003,
        outlet_length=0.004,
        include_cavities=True,
        cavity_depth=0.0004,
        cavity_length=0.001,
        cavity_offset_from_junction=0.0015,
    )
    unfilleted_shape = build_tjunction(unfilleted_params)
    artifacts = generate_cad_artifacts(config, tmp_path)
    roundtrip = read_step(artifacts.step_path)

    assert shape.solid_count == 1
    assert shape.bbox == pytest.approx(params.expected_bbox(), abs=1e-9)
    assert shape.volume > 0
    assert roundtrip.solid_count == 1
    assert roundtrip.bbox == pytest.approx(shape.bbox, abs=1e-9)
    assert roundtrip.volume == pytest.approx(shape.volume, rel=1e-8)
    assert unfilleted_shape.volume == pytest.approx(
        unfilleted_params.expected_unfilleted_volume(), rel=1e-8
    )
    assert not np.isclose(
        shape.volume,
        unfilleted_shape.volume,
        rtol=1e-10,
        atol=1e-18,
    )
    for path in (artifacts.brep_path, artifacts.step_path, artifacts.geo_path):
        assert path.is_file() and path.stat().st_size > 0
    step_source = re.sub(r"\s+", "", artifacts.step_path.read_text(encoding="ascii"))
    assert "SI_UNIT($,.METRE.)" in step_source
    assert "SI_UNIT(.MILLI.,.METRE.)" not in step_source

    external = GeometryPipelineConfig(
        source=GeometrySourceConfig(
            kind="external_step",
            step_path=artifacts.step_path,
            boundary_selectors=procedural_boundary_selectors(params),
        ),
        geometry=params,
    )
    external_artifacts = generate_cad_artifacts(external, tmp_path / "external")
    assert external_artifacts.bbox == pytest.approx(shape.bbox, abs=1e-9)
    assert external_artifacts.volume == pytest.approx(shape.volume, rel=1e-8)


def test_external_gmsh_reads_declared_step_dimensions_in_metres(tmp_path: Path) -> None:
    _require_occ()
    gmsh = shutil.which("gmsh") or shutil.which("gmsh.exe")
    if gmsh is None:
        pytest.skip("external Gmsh executable is not installed")
    params = CadParameters(
        inlet_width=0.001,
        outlet_width=0.001,
        channel_height=0.001,
        inlet_length=0.002,
        outlet_length=0.003,
    )
    artifacts = generate_cad_artifacts(
        GeometryPipelineConfig(
            source=GeometrySourceConfig(kind="procedural_cad"),
            geometry=params,
        ),
        tmp_path,
    )
    geo_path = tmp_path / "step_external_check.geo"
    geo_path.write_text(
        "\n".join(
            [
                'SetFactory("OpenCASCADE");',
                'Geometry.OCCTargetUnit = "M";',
                f'fluid() = ShapeFromFile("{artifacts.step_path.name}");',
                'Physical Volume("fluid", 1) = {fluid()};',
                "Mesh.CharacteristicLengthMin = 0.0005;",
                "Mesh.CharacteristicLengthMax = 0.0005;",
            ]
        ),
        encoding="utf-8",
    )
    msh_path = tmp_path / "step_external_check.msh"
    GmshCli(gmsh, working_directory=tmp_path).generate_mesh(
        geo_path,
        msh_path,
        dimension=3,
        mesh_format="msh4",
        binary=True,
    )
    mesh = meshio.read(msh_path)

    actual_bbox = (*np.min(mesh.points, axis=0), *np.max(mesh.points, axis=0))
    assert actual_bbox == pytest.approx(params.expected_bbox(), abs=1e-9)


def test_read_step_converts_declared_millimetres_to_contract_metres(
    tmp_path: Path,
) -> None:
    _require_occ()
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Controller,
        STEPControl_Writer,
    )

    assert STEPControl_Controller.Init()
    assert Interface_Static.SetCVal("xstep.cascade.unit", "MM")
    assert Interface_Static.SetCVal("write.step.unit", "MM")
    writer = STEPControl_Writer()
    assert (
        writer.Transfer(BRepPrimAPI_MakeBox(2.0, 3.0, 1.0).Shape(), STEPControl_AsIs)
        == IFSelect_RetDone
    )
    step_path = tmp_path / "external_mm.step"
    assert writer.Write(str(step_path)) == IFSelect_RetDone

    imported = read_step(step_path)

    assert imported.bbox == pytest.approx((0.0, 0.0, 0.0, 0.002, 0.003, 0.001))


@pytest.mark.slow
def test_brep_to_gmsh_mesh_matches_cad_ignores_hostile_config_and_runs_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_occ()
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        TetraFlowConfig,
        initialize_tetra_flow_state,
        tetra_flow_step,
    )

    gmsh = shutil.which("gmsh") or shutil.which("gmsh.exe")
    if gmsh is None:
        pytest.skip("external Gmsh executable is not installed")
    user_home = tmp_path / "hostile-user-home"
    user_home.mkdir()
    for file_name in (".gmshrc", ".gmsh-options"):
        (user_home / file_name).write_text("Exit;\n", encoding="utf-8")
    monkeypatch.setenv("GMSH_HOME", str(user_home))
    monkeypatch.setenv("HOME", str(user_home))
    config = GeometryPipelineConfig(
        source=GeometrySourceConfig(kind="procedural_cad"),
        geometry=CadParameters(
            inlet_width=0.001,
            outlet_width=0.001,
            channel_height=0.001,
            inlet_length=0.002,
            outlet_length=0.003,
        ),
        mesh=CadMeshConfig(element_size=0.0005, max_aspect_ratio=12.0),
    )

    result = generate_cad_mesh(config, tmp_path, gmsh_executable=gmsh)
    flow_config = TetraFlowConfig(
        inlet_speed=0.01,
        projection_dt=1e-5,
        max_pressure_iterations=200,
        pressure_tolerance=1e-7,
        backend="numpy",
    )
    state = initialize_tetra_flow_state(result.mesh, flow_config)
    next_state = tetra_flow_step(result.mesh, state, flow_config)

    assert result.validation.is_valid
    assert not result.mesh.boundary_unresolved_faces.size
    assert result.mesh.diagnostics["unmatched_imported_boundary_triangles"] == 0
    mesh_bbox = np.concatenate(
        (np.min(result.mesh.points, axis=0), np.max(result.mesh.points, axis=0))
    )
    assert mesh_bbox == pytest.approx(
        result.artifacts.bbox,
        rel=MESH_BBOX_RELATIVE_TOLERANCE,
        abs=MESH_BBOX_ABSOLUTE_TOLERANCE,
    )
    assert float(np.sum(result.mesh.cell_volumes)) == pytest.approx(
        result.artifacts.volume,
        rel=MESH_VOLUME_RELATIVE_TOLERANCE,
        abs=MESH_VOLUME_ABSOLUTE_TOLERANCE,
    )
    assert result.mesh.diagnostics["tetra_quality"]["invalid_tetra_count"] == 0
    assert not (tmp_path / ".gmsh-home").exists()
    assert np.all(np.isfinite(next_state.face_flux))
    assert np.all(np.isfinite(next_state.pressure))


@pytest.mark.slow
def test_tracked_cavities_and_fillet_example_passes_full_cad_pipeline(
    tmp_path: Path,
) -> None:
    _require_occ()
    gmsh = shutil.which("gmsh") or shutil.which("gmsh.exe")
    if gmsh is None:
        pytest.skip("external Gmsh executable is not installed")
    project_root = Path(__file__).resolve().parents[3]
    config = load_geometry_pipeline_config(
        project_root / "data/examples/geometry/t_junction_cad.json"
    )

    result = generate_cad_mesh(config, tmp_path, gmsh_executable=gmsh)

    quality = result.mesh.diagnostics["tetra_quality"]
    assert quality["tetra_mean_ratio"]["min"] >= config.mesh.min_tetra_mean_ratio
    assert quality["tetra_aspect_ratio_proxy"]["max"] <= config.mesh.max_aspect_ratio
    assert np.all(result.mesh.volume_tag_per_cell == 1)
    assert not result.mesh.boundary_unresolved_faces.size
    assert result.mesh.diagnostics["unmatched_imported_boundary_triangles"] == 0
