"""Backend-specific semantic validation for synchronous compute requests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from backend.app.stage_registry import (
    STAGE_REGISTRY,
    StageParametersError,
    parse_stage_parameters,
    stage_input_prefixes,
)
from microfluidics_contracts import RuntimeSettings, SubmitRunRequestV1

__all__ = [
    "RequestValidationError",
    "prepare_submit_request",
    "stage_input_prefixes",
]


class RequestValidationError(ValueError):
    """Semantic validation failure for a submit request."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status


def prepare_submit_request(
    project_root: Path,
    settings: RuntimeSettings,
    request: SubmitRunRequestV1,
) -> SubmitRunRequestV1:
    """Validate backend policy and return a canonicalized compute request."""

    stage = STAGE_REGISTRY.get(request.experiment_id)
    if stage is None:
        allowed_ids = sorted(STAGE_REGISTRY)
        allowed = ", ".join(allowed_ids)
        raise RequestValidationError(
            code="unknown_experiment",
            message=f"Unknown experiment_id={request.experiment_id!r}. Allowed values: {allowed}.",
            details={
                "experiment_id": request.experiment_id,
                "allowed_experiment_ids": allowed_ids,
            },
        )

    try:
        normalized_parameters = parse_stage_parameters(
            request.experiment_id,
            request.parameters,
            input_prefixes=stage_input_prefixes(settings),
        ).to_dict()
    except (ValueError, StageParametersError) as exc:
        raise RequestValidationError(
            code="invalid_parameters",
            message=str(exc),
            details={
                "experiment_id": request.experiment_id,
                "allowed_parameters": sorted(stage.parameters_model.allowed_fields),
            },
            http_status=422,
        ) from exc

    return replace(
        request,
        parameters=normalized_parameters,
    )
