"""End-to-end CAD artifact and external Gmsh CLI pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from microfluidics.cad.config import (
    CadMeshConfig,
    GeometryPipelineConfig,
    PHYSICAL_GROUP_NAMES,
)
from microfluidics.cad.geo import build_geo_source, procedural_boundary_selectors
from microfluidics.cad.occ import (
    CadShape,
    build_tjunction,
    ensure_tessellatable,
    read_step,
    write_brep,
    write_step,
)
from microfluidics.gmsh.gmsh_cli import (
    GmshCli,
    GmshMeshGenerationResult,
    isolated_gmsh_environment,
)
from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh
from microfluidics.gmsh.gmsh_mesh_types import ImportedTetraMesh
from microfluidics.gmsh.gmsh_mesh_validation import (
    GmshMeshValidationReport,
    validate_imported_tetra_mesh,
)


class CadPipelineValidationError(RuntimeError):
    """Raised when generated CAD or mesh artifacts violate the contract."""


# Gmsh facets curved CAD boundaries, so its tetrahedral volume is not bitwise
# equal to the OpenCASCADE volume. A 0.5% allowance covers that discretization
# while still rejecting unit-scale errors and materially incomplete meshes.
MESH_BBOX_RELATIVE_TOLERANCE = 1.0e-8
MESH_BBOX_ABSOLUTE_TOLERANCE = 1.0e-12
MESH_VOLUME_RELATIVE_TOLERANCE = 5.0e-3
MESH_VOLUME_ABSOLUTE_TOLERANCE = 1.0e-24


@dataclass(frozen=True)
class CadArtifacts:
    brep_path: Path
    step_path: Path
    geo_path: Path
    bbox: tuple[float, float, float, float, float, float]
    volume: float
    step_roundtrip_bbox: tuple[float, float, float, float, float, float]
    step_roundtrip_volume: float


@dataclass(frozen=True)
class CadMeshPipelineResult:
    artifacts: CadArtifacts
    generation: GmshMeshGenerationResult
    mesh: ImportedTetraMesh
    validation: GmshMeshValidationReport


def _assert_roundtrip(original: CadShape, roundtrip: CadShape) -> None:
    scale = max(max(abs(value) for value in original.bbox), 1.0)
    bbox_tolerance = scale * 1e-8
    if not np.allclose(original.bbox, roundtrip.bbox, rtol=0.0, atol=bbox_tolerance):
        raise CadPipelineValidationError(
            f"STEP round-trip changed bbox: {original.bbox} -> {roundtrip.bbox}"
        )
    if not np.isclose(original.volume, roundtrip.volume, rtol=1e-8, atol=1e-18):
        raise CadPipelineValidationError(
            f"STEP round-trip changed volume: {original.volume} -> {roundtrip.volume}"
        )


def _assert_expected_bbox(
    actual: tuple[float, float, float, float, float, float],
    expected: tuple[float, float, float, float, float, float],
) -> None:
    scale = max(max(abs(value) for value in expected), 1.0)
    if not np.allclose(actual, expected, rtol=0.0, atol=scale * 1e-8):
        raise CadPipelineValidationError(
            f"procedural CAD bbox differs from configured dimensions: {actual} != {expected}"
        )


def _assert_expected_volume(actual: float, expected: float) -> None:
    if not np.isclose(actual, expected, rtol=1e-8, atol=max(expected * 1e-10, 1e-24)):
        raise CadPipelineValidationError(
            "procedural CAD volume differs from the configured primitive union: "
            f"{actual} != {expected}"
        )


def generate_cad_artifacts(
    config: GeometryPipelineConfig,
    output_directory: str | Path,
) -> CadArtifacts:
    """Create model.brep, display.step and geometry_generated.geo."""

    config.validate()
    if config.source.kind == "legacy_boxes":
        raise ValueError("legacy_boxes does not produce CAD artifacts")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if config.source.kind == "procedural_cad":
        cad_shape = build_tjunction(config.geometry)
        _assert_expected_bbox(cad_shape.bbox, config.geometry.expected_bbox())
        if config.geometry.fillet_radius == 0:
            _assert_expected_volume(
                cad_shape.volume,
                config.geometry.expected_unfilleted_volume(),
            )
        selectors = procedural_boundary_selectors(config.geometry)
    else:
        assert config.source.step_path is not None
        cad_shape = read_step(config.source.step_path)
        selectors = config.source.boundary_selectors

    brep_path = write_brep(cad_shape, output / "model.brep")
    step_path = write_step(cad_shape, output / "display.step")
    roundtrip = read_step(step_path)
    _assert_roundtrip(cad_shape, roundtrip)
    ensure_tessellatable(roundtrip)

    geo_path = output / "geometry_generated.geo"
    geo_path.write_text(
        build_geo_source(
            cad_file_name=brep_path.name,
            boundary_selectors=selectors,
            mesh=config.mesh,
        ),
        encoding="utf-8",
    )
    artifacts = CadArtifacts(
        brep_path=brep_path,
        step_path=step_path,
        geo_path=geo_path,
        bbox=cad_shape.bbox,
        volume=cad_shape.volume,
        step_roundtrip_bbox=roundtrip.bbox,
        step_roundtrip_volume=roundtrip.volume,
    )
    manifest = {
        "schema_version": "cad_artifacts_v1",
        "source_kind": config.source.kind,
        "artifacts": {
            "brep": brep_path.name,
            "step": step_path.name,
            "geo": geo_path.name,
        },
        "solid_count": 1,
        "bbox": list(artifacts.bbox),
        "volume": artifacts.volume,
        "step_roundtrip_bbox": list(artifacts.step_roundtrip_bbox),
        "step_roundtrip_volume": artifacts.step_roundtrip_volume,
        "physical_groups": list(PHYSICAL_GROUP_NAMES),
    }
    (output / "cad_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifacts


def assert_msh41_binary(path: str | Path) -> None:
    mesh_path = Path(path)
    with mesh_path.open("rb") as stream:
        first = stream.readline().strip()
        format_line = stream.readline().strip()
    if first != b"$MeshFormat":
        raise CadPipelineValidationError(f"not a Gmsh MSH file: {mesh_path}")
    fields = format_line.split()
    if fields != [b"4.1", b"1", b"8"]:
        raise CadPipelineValidationError(
            f"solver mesh must be MSH 4.1 Binary, got {format_line!r}"
        )


def _assert_physical_groups(mesh: ImportedTetraMesh) -> None:
    expected = {
        "fluid": (1, 3),
        "left_inlet": (2, 2),
        "right_inlet": (3, 2),
        "outlet": (4, 2),
        "walls": (5, 2),
    }
    actual = {name: tuple(values) for name, values in mesh.physical_groups.items()}
    if actual != expected:
        raise CadPipelineValidationError(
            f"generated mesh physical groups differ from contract: {actual}"
        )
    volume_tags = np.asarray(mesh.volume_tag_per_cell, dtype=np.int32)
    if volume_tags.shape != (mesh.tetrahedra.shape[0],) or not np.all(
        volume_tags == expected["fluid"][0]
    ):
        raise CadPipelineValidationError(
            "generated mesh tetrahedra must all belong to Physical Volume "
            f"fluid=1, got tags {sorted(int(tag) for tag in np.unique(volume_tags))}"
        )


def _validate_quality(
    mesh: ImportedTetraMesh,
    mesh_config: CadMeshConfig,
) -> GmshMeshValidationReport:
    report = validate_imported_tetra_mesh(
        mesh,
        min_positive_volume=mesh_config.min_positive_volume,
    )
    if not report.is_valid:
        raise CadPipelineValidationError(
            "generated CAD mesh failed validation: " + "; ".join(report.errors)
        )
    unresolved_count = int(mesh.boundary_unresolved_faces.size)
    if unresolved_count:
        raise CadPipelineValidationError(
            "generated CAD mesh has unresolved reconstructed boundary faces: "
            f"{unresolved_count}"
        )
    unmatched_count = int(
        mesh.diagnostics.get("unmatched_imported_boundary_triangles", 0)
    )
    if unmatched_count:
        raise CadPipelineValidationError(
            "generated CAD mesh has unmatched imported boundary triangles: "
            f"{unmatched_count}"
        )
    quality = mesh.diagnostics.get("tetra_quality", {})
    aspect = quality.get("tetra_aspect_ratio_proxy", {})
    max_aspect = float(aspect.get("max", float("inf")))
    if max_aspect > mesh_config.max_aspect_ratio:
        raise CadPipelineValidationError(
            f"generated CAD mesh max aspect ratio {max_aspect:.6g} exceeds "
            f"configured threshold {mesh_config.max_aspect_ratio:.6g}"
        )
    mean_ratio = quality.get("tetra_mean_ratio", {})
    min_mean_ratio = float(mean_ratio.get("min", float("-inf")))
    if (
        not np.isfinite(min_mean_ratio)
        or min_mean_ratio < mesh_config.min_tetra_mean_ratio
    ):
        raise CadPipelineValidationError(
            f"generated CAD mesh minimum tetra mean ratio {min_mean_ratio:.6g} is below "
            f"configured threshold {mesh_config.min_tetra_mean_ratio:.6g}"
        )
    return report


def _assert_mesh_matches_cad(
    mesh: ImportedTetraMesh,
    artifacts: CadArtifacts,
) -> dict[str, object]:
    """Reject a mesh whose spatial extent or integrated volume differs from CAD."""

    mesh_bbox = np.concatenate(
        (np.min(mesh.points, axis=0), np.max(mesh.points, axis=0))
    )
    cad_bbox = np.asarray(artifacts.bbox, dtype=np.float64)
    if not np.allclose(
        mesh_bbox,
        cad_bbox,
        rtol=MESH_BBOX_RELATIVE_TOLERANCE,
        atol=MESH_BBOX_ABSOLUTE_TOLERANCE,
    ):
        raise CadPipelineValidationError(
            "generated mesh bbox differs from CAD bbox: "
            f"{tuple(mesh_bbox.tolist())} != {artifacts.bbox}"
        )

    mesh_volume = float(np.sum(mesh.cell_volumes, dtype=np.float64))
    if not np.isclose(
        mesh_volume,
        artifacts.volume,
        rtol=MESH_VOLUME_RELATIVE_TOLERANCE,
        atol=MESH_VOLUME_ABSOLUTE_TOLERANCE,
    ):
        raise CadPipelineValidationError(
            "generated mesh tetrahedral volume differs from CAD volume: "
            f"{mesh_volume} != {artifacts.volume}"
        )
    return {
        "cad_bbox": list(artifacts.bbox),
        "mesh_bbox": mesh_bbox.tolist(),
        "bbox_relative_tolerance": MESH_BBOX_RELATIVE_TOLERANCE,
        "bbox_absolute_tolerance_m": MESH_BBOX_ABSOLUTE_TOLERANCE,
        "cad_volume_m3": artifacts.volume,
        "mesh_tetrahedral_volume_m3": mesh_volume,
        "volume_relative_tolerance": MESH_VOLUME_RELATIVE_TOLERANCE,
        "volume_absolute_tolerance_m3": MESH_VOLUME_ABSOLUTE_TOLERANCE,
    }


def generate_cad_mesh(
    config: GeometryPipelineConfig,
    output_directory: str | Path,
    *,
    gmsh_executable: str | Path | None = None,
) -> CadMeshPipelineResult:
    """Run CAD -> GEO -> external Gmsh CLI -> validated MSH 4.1 Binary."""

    artifacts = generate_cad_artifacts(config, output_directory)
    with TemporaryDirectory(prefix="microfluidics-cad-gmsh-") as gmsh_home:
        client = GmshCli(
            gmsh_executable,
            working_directory=artifacts.geo_path.parent,
            environment=isolated_gmsh_environment(gmsh_home),
        )
        generation = client.generate_mesh(
            artifacts.geo_path,
            artifacts.geo_path.parent / "mesh.msh",
            dimension=3,
            mesh_format="msh4",
            additional_args=("-optimize_netgen",) if config.mesh.optimize else (),
            binary=True,
            timeout_seconds=config.mesh.timeout_seconds,
        )
    assert_msh41_binary(generation.msh_path)
    mesh = import_gmsh_tetra_mesh(generation.msh_path)
    _assert_physical_groups(mesh)
    validation = _validate_quality(mesh, config.mesh)
    geometry_consistency = _assert_mesh_matches_cad(mesh, artifacts)
    mesh_manifest = {
        "schema_version": "cad_mesh_validation_v1",
        "msh": generation.msh_path.name,
        "msh_version": "4.1",
        "binary": True,
        "node_count": int(mesh.points.shape[0]),
        "tetra_count": int(mesh.tetrahedra.shape[0]),
        "physical_groups": {
            name: {"tag": int(values[0]), "dimension": int(values[1])}
            for name, values in sorted(mesh.physical_groups.items())
        },
        "geometry_consistency": geometry_consistency,
        "tetra_quality": mesh.diagnostics.get("tetra_quality", {}),
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "valid": validation.is_valid,
    }
    (artifacts.geo_path.parent / "mesh_validation.json").write_text(
        json.dumps(mesh_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CadMeshPipelineResult(
        artifacts=artifacts,
        generation=generation,
        mesh=mesh,
        validation=validation,
    )


__all__ = [
    "CadArtifacts",
    "CadMeshPipelineResult",
    "CadPipelineValidationError",
    "assert_msh41_binary",
    "generate_cad_artifacts",
    "generate_cad_mesh",
]
