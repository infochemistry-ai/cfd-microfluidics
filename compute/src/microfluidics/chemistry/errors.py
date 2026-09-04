"""Errors raised by the standalone chemistry subsystem."""


class ChemistryError(ValueError):
    """Base error for invalid chemistry inputs or unsupported calculations."""


class MechanismValidationError(ChemistryError):
    """Raised when a chemistry mechanism is incomplete or inconsistent."""


class ChemistryEvaluationError(ChemistryError):
    """Raised when a compiled mechanism cannot evaluate a supplied state."""
