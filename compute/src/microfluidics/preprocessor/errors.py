"""Errors raised while loading and compiling CFD case configuration."""


class CaseConfigError(ValueError):
    """The case document is malformed or inconsistent with the mesh."""


class ZoneResolutionError(CaseConfigError):
    """A configured physics zone cannot be resolved unambiguously."""


class BoundaryConditionError(CaseConfigError):
    """Boundary conditions overlap or cannot be compiled."""


class MaterialAssignmentError(CaseConfigError):
    """Volume zones do not produce one material assignment per cell."""
