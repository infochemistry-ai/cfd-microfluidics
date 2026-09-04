"""Standalone homogeneous chemistry interfaces and numerical kernels.

This package deliberately has no imports from Gmsh, flow, transport or thermal
solvers.  ``compile_mechanism`` compiles topology once and
``CompiledChemistry.evaluate`` accepts scalar states or mesh-agnostic batches.
"""

from microfluidics.chemistry.errors import (
    ChemistryError,
    ChemistryEvaluationError,
    MechanismValidationError,
)
from microfluidics.chemistry.kernel import (
    CHEMISTRY_CONTRACT_VERSION,
    CompiledChemistry,
    ChemistryEvaluation,
    ChemistrySources,
    compile_mechanism,
)
from microfluidics.chemistry.loader import load_mechanism, mechanism_from_mapping
from microfluidics.chemistry.models import (
    ArrheniusParameters,
    Mechanism,
    Reaction,
    ReactionThermodynamics,
    Species,
    SpeciesPhase,
)
from microfluidics.chemistry.provenance import (
    MECHANISM_FINGERPRINT_SCHEMA,
    canonical_mechanism_payload,
    mechanism_sha256,
)

__all__ = [
    "CHEMISTRY_CONTRACT_VERSION",
    "ArrheniusParameters",
    "ChemistryError",
    "ChemistryEvaluation",
    "ChemistryEvaluationError",
    "ChemistrySources",
    "CompiledChemistry",
    "Mechanism",
    "MechanismValidationError",
    "MECHANISM_FINGERPRINT_SCHEMA",
    "Reaction",
    "ReactionThermodynamics",
    "Species",
    "SpeciesPhase",
    "compile_mechanism",
    "canonical_mechanism_payload",
    "load_mechanism",
    "mechanism_sha256",
    "mechanism_from_mapping",
]
