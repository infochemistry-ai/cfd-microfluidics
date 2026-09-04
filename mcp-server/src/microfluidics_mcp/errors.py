"""Domain errors surfaced to MCP clients without a second error vocabulary."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from mcp.server.fastmcp.exceptions import ToolError

from backend.app.input_errors import StageInputError
from backend.app.request_validation import RequestValidationError
from backend.app.service import IdempotencyConflictError, RunCapacityError
from backend.app.stage_registry import StageParametersError
from microfluidics_contracts import ContractValidationError

logger = logging.getLogger(__name__)

_STATIC_CODES: tuple[tuple[type[Exception], str], ...] = (
    (IdempotencyConflictError, "idempotency_conflict"),
    (RunCapacityError, "capacity_exceeded"),
    (StageParametersError, "invalid_parameters"),
    (ContractValidationError, "invalid_contract"),
)

# RequestValidationError, StageParametersError and ContractValidationError
# are already ValueError subclasses, so the bare ValueError entry below is
# redundant for them; it exists only to catch *unrelated* ValueErrors that
# escape future tool code (a bad int(), a malformed unpack). Those get
# repackaged as "invalid_request: <message>" and shown to the agent as if
# it were bad user input, which risks hiding a real bug. The trade-off we
# accept: guard() logs every caught exception with its traceback via
# logger.exception before converting it, so an operator can still tell a
# genuine programming error apart from legitimate domain validation.
_HANDLED = (
    RequestValidationError,
    StageInputError,
    IdempotencyConflictError,
    RunCapacityError,
    StageParametersError,
    ContractValidationError,
    ValueError,
)


def tool_error(exc: Exception) -> ToolError:
    """Preserve the service's own error code instead of inventing a new one."""

    code = getattr(exc, "code", None)
    if not isinstance(code, str) or not code:
        code = "invalid_request"
        for exc_type, static_code in _STATIC_CODES:
            if isinstance(exc, exc_type):
                code = static_code
                break
    message = getattr(exc, "message", None)
    if not isinstance(message, str) or not message:
        message = str(exc).strip() or exc.__class__.__name__
    return ToolError(f"{code}: {message}")


@contextmanager
def guard() -> Iterator[None]:
    try:
        yield
    except _HANDLED as exc:
        logger.exception("guard() converting %s to ToolError", exc.__class__.__name__)
        raise tool_error(exc) from exc
