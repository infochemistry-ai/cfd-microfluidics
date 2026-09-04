"""Public CFD case preprocessing contracts."""

from microfluidics.preprocessor.case import (
    case_config_from_mapping,
    case_config_to_dict,
    load_case_config,
    resolve_case_mesh_path,
)
from microfluidics.preprocessor.boundary import (
    CompiledBoundaryCondition,
    CompiledBoundaryConditions,
    PeriodicFacePair,
    compile_boundary_conditions,
    pair_periodic_faces,
)
from microfluidics.preprocessor.errors import (
    BoundaryConditionError,
    CaseConfigError,
    MaterialAssignmentError,
    ZoneResolutionError,
)
from microfluidics.preprocessor.models import (
    BoundaryConditionSpec,
    CaseConfigV1,
    MaterialSpec,
    MeshQualityPolicy,
    MeshSpec,
    ZoneSpec,
)
from microfluidics.preprocessor.quality import (
    MeshQualityFinding,
    MeshQualityGateReport,
    evaluate_mesh_quality_gate,
)
from microfluidics.preprocessor.runtime import (
    FlowRuntimeProfile,
    MaterialCellAssignment,
    RobinBoundaryValue,
    ScalarRuntimeProfile,
    apply_flow_profile_to_mesh,
    compile_flow_runtime_profile,
    compile_material_cell_assignment,
    compile_scalar_runtime_profile,
    compile_uniform_material_properties,
    scalar_profile_to_diffusion_kwargs,
)
from microfluidics.preprocessor.zones import (
    ResolvedCaseZones,
    ResolvedZone,
    resolve_case_zones,
)

__all__ = [
    "BoundaryConditionError",
    "BoundaryConditionSpec",
    "CompiledBoundaryCondition",
    "CompiledBoundaryConditions",
    "CaseConfigError",
    "CaseConfigV1",
    "MaterialSpec",
    "MaterialAssignmentError",
    "MeshQualityPolicy",
    "MeshQualityFinding",
    "MeshQualityGateReport",
    "MeshSpec",
    "PeriodicFacePair",
    "FlowRuntimeProfile",
    "MaterialCellAssignment",
    "ResolvedCaseZones",
    "ResolvedZone",
    "RobinBoundaryValue",
    "ScalarRuntimeProfile",
    "ZoneResolutionError",
    "ZoneSpec",
    "case_config_from_mapping",
    "case_config_to_dict",
    "apply_flow_profile_to_mesh",
    "compile_boundary_conditions",
    "compile_flow_runtime_profile",
    "compile_material_cell_assignment",
    "compile_scalar_runtime_profile",
    "compile_uniform_material_properties",
    "scalar_profile_to_diffusion_kwargs",
    "evaluate_mesh_quality_gate",
    "load_case_config",
    "pair_periodic_faces",
    "resolve_case_zones",
    "resolve_case_mesh_path",
]
