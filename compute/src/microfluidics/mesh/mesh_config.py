"""Configuration models for mesh-oriented T-junction infrastructure.

This module is intentionally independent from the structured-grid solver stack.
It defines settings for mesh generation, quality diagnostics, and export.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from microfluidics.path_contract import MESH_NATIVE_FLOW_RUNS_ROOT_REL

Resolution3D = Tuple[int, int, int]


@dataclass(frozen=True)
class MeshGeometryConfig3D:
    """Article-oriented 3D geometry inputs for mesh scaffolding."""

    inlet_width: float = 1.0e-3
    outlet_width: float = 2.0e-3
    channel_height: float = 1.0e-3
    inlet_length: float = 12.0e-3
    outlet_length: float = 20.0e-3
    patch_depth: float = 0.5e-3
    include_cavities: bool = False
    cavity_depth: float = 0.49875e-3
    cavity_length: float = 1.99500e-3
    cavity_offset_from_junction: float = 0.49875e-3
    hydraulic_diameter: float = 1.33e-3


@dataclass(frozen=True)
class MeshRefinementConfig:
    """Refinement targets for the mesh-oriented branch.

    Note:
    Current implementation stores refinement metadata and layer diagnostics.
    It does not yet run a full adaptive tetra/hex remesher.
    """

    volume_resolution: Resolution3D = (220, 140, 28)
    wall_refinement_thickness: float = 0.15e-3
    wall_refinement_levels: int = 3
    cavity_target_size: float = 0.08e-3
    junction_target_size: float = 0.10e-3


@dataclass(frozen=True)
class MeshOutputConfig:
    """Output settings for mesh runs."""

    base_output_dir: Path = MESH_NATIVE_FLOW_RUNS_ROOT_REL
    export_volume_vtu: bool = True
    export_surface_vtp: bool = True
    write_preview_png: bool = True
    write_quality_report: bool = True


@dataclass(frozen=True)
class MeshRunConfig:
    """Top-level run configuration for mesh generation baseline."""

    case_tag: str = "TJ"
    reynolds: float = 200.0
    kinematic_viscosity: float = 1e-6
    geometry: MeshGeometryConfig3D = field(default_factory=MeshGeometryConfig3D)
    refinement: MeshRefinementConfig = field(default_factory=MeshRefinementConfig)
    output: MeshOutputConfig = field(default_factory=MeshOutputConfig)
    stage_note: str = (
        "mesh scaffold diagnostics plus a low-Re mesh-native flow prototype; "
        "structured-grid bridge components have been removed"
    )


def default_mesh_run_config(
    case_tag: str = "TJ",
    include_cavities: bool = False,
) -> MeshRunConfig:
    geometry = MeshGeometryConfig3D(include_cavities=include_cavities)
    return MeshRunConfig(case_tag=case_tag, geometry=geometry)
