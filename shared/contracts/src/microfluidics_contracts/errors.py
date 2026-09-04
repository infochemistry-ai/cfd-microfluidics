"""Contract-specific errors."""


class ContractValidationError(ValueError):
    """Raised when request or response payloads do not match contracts."""
