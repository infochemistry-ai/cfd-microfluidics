"""Reactive transport contracts, chemistry integration, and CFD coupling."""

from microfluidics.reactive.case import (
    REACTIVE_CASE_CONTRACT_VERSION,
    REACTIVE_CASE_SCHEMA_VERSION,
    REACTIVE_TRANSPORT_CONTRACT_VERSION,
    ChemistryIntegratorV1,
    ReactiveCaseV1,
    ReactiveInletV1,
    ReactiveMaterialV1,
    ReactiveOutputV1,
    ReactiveStateV1,
    ReactiveTimeV1,
    load_reactive_case,
    normalize_group_name,
    reactive_case_from_mapping,
)
from microfluidics.reactive.errors import (
    ChemistryIntegrationError,
    ReactiveCaseValidationError,
    ReactiveTransportError,
    ReactiveWalltimeLimitError,
    TransportSubstepCapError,
)
from microfluidics.reactive.integrator import (
    ChemistryIntegrationStats,
    ReactionAdvanceResult,
    advance_reaction,
)
from microfluidics.reactive.operators import (
    ReactiveSpatialPrecompute,
    SpatialAdvanceDiagnostics,
    SpatialAdvanceResult,
    advance_spatial_fields,
    build_reactive_spatial_precompute,
    required_spatial_substeps,
    stable_spatial_dt,
)
from microfluidics.reactive.solver import ReactiveRunResult, run_reactive_transport

__all__ = [
    "REACTIVE_CASE_CONTRACT_VERSION",
    "REACTIVE_CASE_SCHEMA_VERSION",
    "REACTIVE_TRANSPORT_CONTRACT_VERSION",
    "ChemistryIntegrationError",
    "ChemistryIntegrationStats",
    "ChemistryIntegratorV1",
    "ReactiveCaseV1",
    "ReactiveCaseValidationError",
    "ReactiveInletV1",
    "ReactiveMaterialV1",
    "ReactiveOutputV1",
    "ReactiveRunResult",
    "ReactiveStateV1",
    "ReactiveSpatialPrecompute",
    "ReactiveTimeV1",
    "ReactiveTransportError",
    "ReactiveWalltimeLimitError",
    "ReactionAdvanceResult",
    "SpatialAdvanceDiagnostics",
    "SpatialAdvanceResult",
    "TransportSubstepCapError",
    "load_reactive_case",
    "advance_reaction",
    "advance_spatial_fields",
    "build_reactive_spatial_precompute",
    "normalize_group_name",
    "reactive_case_from_mapping",
    "required_spatial_substeps",
    "stable_spatial_dt",
    "run_reactive_transport",
]
