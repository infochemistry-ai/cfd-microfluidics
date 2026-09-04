"""Versioned DTOs used by the local API and compute worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .errors import ContractValidationError

CONTRACT_VERSION = "v1"
_SUBMIT_REQUEST_FIELDS = frozenset(
    {"contract_version", "experiment_id", "parameters", "request_id"}
)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    # RUNNING is part of the vocabulary but is never emitted by the stateless
    # compute service: a claimed request is PENDING until its record becomes
    # ready, and ready always carries a terminal outcome. Neither the compute
    # API nor the MCP surface documents it as a state a poller will observe,
    # and nothing should start emitting it without documenting the change.
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubmitRunRequestV1:
    experiment_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubmitRunRequestV1":
        if not isinstance(payload, dict):
            raise ContractValidationError("Submit payload must be a JSON object.")

        unsupported_fields = sorted(set(payload) - _SUBMIT_REQUEST_FIELDS)
        if unsupported_fields:
            raise ContractValidationError(
                "Unsupported submit fields: " + ", ".join(unsupported_fields)
            )

        contract_version = payload.get("contract_version", CONTRACT_VERSION)
        if contract_version != CONTRACT_VERSION:
            raise ContractValidationError(
                f"Unsupported contract_version={contract_version!r}. "
                f"Expected {CONTRACT_VERSION!r}."
            )

        experiment_id = payload.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise ContractValidationError("'experiment_id' must be a non-empty string.")

        parameters = payload.get("parameters", {})
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ContractValidationError(
                "'parameters' must be an object when provided."
            )

        request_id = payload.get("request_id")
        if request_id is not None:
            if not isinstance(request_id, str):
                raise ContractValidationError(
                    "'request_id' must be a string when provided."
                )
            request_id = request_id.strip()
            if not request_id:
                raise ContractValidationError(
                    "'request_id' must not be empty when provided."
                )

        return cls(
            experiment_id=experiment_id.strip(),
            parameters=parameters,
            request_id=request_id,
            contract_version=CONTRACT_VERSION,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "experiment_id": self.experiment_id,
            "parameters": self.parameters,
            "request_id": self.request_id,
        }


@dataclass
class ErrorPayloadV1:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ResultPayloadV1:
    run_id: str
    adapter_name: str
    experiment_id: str
    exit_code: int
    started_at: str
    finished_at: str
    duration_seconds: float
    request_id: str | None = None
    artifacts: list[str] = field(default_factory=list)
    log_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "adapter_name": self.adapter_name,
            "experiment_id": self.experiment_id,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "request_id": self.request_id,
            "artifacts": self.artifacts,
            "log_path": self.log_path,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResponseV1:
    request_id: str
    run_id: str
    status: RunStatus
    result: ResultPayloadV1 | None = None
    error: ErrorPayloadV1 | None = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass
class StatusResponseV1:
    run_id: str
    status: RunStatus
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    result: ResultPayloadV1 | None = None
    error: ErrorPayloadV1 | None = None
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error.to_dict() if self.error else None,
        }
