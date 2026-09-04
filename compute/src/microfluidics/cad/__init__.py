"""Procedural OpenCASCADE geometry and CAD-to-mesh pipeline."""

from microfluidics.cad.config import (
    BoundaryBoxSelector,
    CadMeshConfig,
    CadParameters,
    GeometryConfigError,
    GeometryPipelineConfig,
    GeometrySourceConfig,
    load_geometry_pipeline_config,
)
from microfluidics.cad.occ import (
    CadBuildError,
    CadShape,
    OpenCascadeUnavailableError,
    build_tjunction,
    ensure_tessellatable,
    read_step,
    write_brep,
    write_step,
)
from microfluidics.cad.pipeline import (
    CadArtifacts,
    CadMeshPipelineResult,
    CadPipelineValidationError,
    generate_cad_artifacts,
    generate_cad_mesh,
)

__all__ = [
    "BoundaryBoxSelector",
    "CadArtifacts",
    "CadBuildError",
    "CadMeshConfig",
    "CadMeshPipelineResult",
    "CadParameters",
    "CadPipelineValidationError",
    "CadShape",
    "GeometryConfigError",
    "GeometryPipelineConfig",
    "GeometrySourceConfig",
    "OpenCascadeUnavailableError",
    "build_tjunction",
    "ensure_tessellatable",
    "generate_cad_artifacts",
    "generate_cad_mesh",
    "load_geometry_pipeline_config",
    "read_step",
    "write_brep",
    "write_step",
]
