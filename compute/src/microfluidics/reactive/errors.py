"""Errors raised by reactive-transport contracts and numerical coupling."""


class ReactiveTransportError(RuntimeError):
    """Base reactive-transport failure."""


class ReactiveCaseValidationError(ValueError):
    """Reactive-case JSON violates the v1 contract."""


class ChemistryIntegrationError(ReactiveTransportError):
    """Adaptive explicit chemistry cannot advance the requested interval."""


class TransportSubstepCapError(ReactiveTransportError):
    """Stable spatial advancement exceeds the configured substep cap."""


class ReactiveWalltimeLimitError(ReactiveTransportError):
    """Reactive advancement reached its soft walltime deadline."""
