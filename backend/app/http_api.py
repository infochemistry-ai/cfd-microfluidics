"""Minimal HTTP API for stateless compute execution, using stdlib only."""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import MISSING, fields
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from microfluidics_contracts import (
    ContractValidationError,
    ErrorPayloadV1,
    RuntimeSettings,
    SubmitRunRequestV1,
)

from .input_errors import StageInputError
from .request_validation import RequestValidationError, stage_input_prefixes
from .service import ComputeExecutionService, IdempotencyConflictError
from .stage_registry import (
    STAGE_REGISTRY,
    Device,
    FlowNumericalProfile,
    InputPrefixes,
    input_path_schema_pattern,
)


logger = logging.getLogger(__name__)


class RequestBodyError(Exception):
    def __init__(self, *, status: HTTPStatus, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def api_key_matches(presented: str, configured: str) -> bool:
    """Constant-time key comparison that cannot raise on a hostile header.

    Header bytes reach both doors as latin-1/iso-8859-1 text (the stdlib
    parser here, the ASGI middleware in the MCP transport), so a presented
    key carrying non-ASCII bytes arrives as a non-ASCII `str`.
    `secrets.compare_digest` raises TypeError for those, which would leave
    the handler as an unauthenticated 500 with a traceback instead of the
    normal 401. Such a key simply is not the configured key, so it takes the
    rejection path. Configured keys are expected to be ASCII; a non-ASCII
    SERVICE_API_KEY therefore matches nothing and fails closed.

    Shared with `microfluidics_mcp.http_app` so the two doors cannot drift
    apart on what counts as a valid key.
    """

    if not presented.isascii() or not configured.isascii():
        return False
    return secrets.compare_digest(presented, configured)


def build_openapi_schema(*, input_prefixes: InputPrefixes = "") -> dict[str, Any]:
    """Expose the strict discriminated CFD request union for generated clients."""

    steps_schema = {"type": "integer", "minimum": 1, "maximum": 1_000_000}
    device_schema = {
        "type": "string",
        "enum": [device.value for device in Device],
        "default": Device.CPU.value,
    }

    def parameters(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

    parameter_schemas: dict[str, Any] = {}
    request_schemas: dict[str, Any] = {}
    stage_models: dict[str, tuple[str, str]] = {}
    for stage_id, definition in STAGE_REGISTRY.items():
        parameter_name = definition.parameters_model.__name__
        request_name = parameter_name.replace("Parameters", "ComputeRequest")
        stage_models[stage_id] = (request_name, parameter_name)
        bindings = {binding.parameter_name: binding for binding in definition.inputs}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for model_field in fields(definition.parameters_model):
            binding = bindings.get(model_field.name)
            if binding is not None:
                properties[model_field.name] = {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1024,
                    "pattern": input_path_schema_pattern(
                        suffix=binding.required_suffix,
                        input_prefixes=input_prefixes,
                    ),
                }
            elif model_field.name == "num_steps":
                properties[model_field.name] = steps_schema
            elif model_field.name == "device":
                properties[model_field.name] = (
                    {
                        "type": "string",
                        "enum": [Device.CPU.value],
                        "default": Device.CPU.value,
                    }
                    if stage_id == "stage_reactive_transport"
                    else device_schema
                )
            elif model_field.name == "numerical_profile":
                properties[model_field.name] = {
                    "type": "string",
                    "enum": [profile.value for profile in FlowNumericalProfile],
                    "default": FlowNumericalProfile.DEFAULT.value,
                }
            elif model_field.name in {
                "flow_stop_physical_time",
                "snapshot_time_interval",
            }:
                properties[model_field.name] = {
                    "type": ["number", "null"],
                    "exclusiveMinimum": 0.0,
                    "default": None,
                }
            elif model_field.name == "max_walltime_seconds":
                properties[model_field.name] = {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.0,
                }
            else:
                raise RuntimeError(
                    f"No OpenAPI schema registered for stage field {model_field.name!r}."
                )
            if (
                model_field.default is MISSING
                and model_field.default_factory is MISSING
            ):
                required.append(model_field.name)
        parameter_schemas[parameter_name] = parameters(properties, required)
        request_schemas[request_name] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["experiment_id", "parameters"],
            "properties": {
                "request_id": {"type": "string", "minLength": 1},
                "experiment_id": {"type": "string", "const": stage_id},
                "parameters": {"$ref": f"#/components/schemas/{parameter_name}"},
            },
        }
    union_refs = [
        {"$ref": f"#/components/schemas/{request_name}"}
        for request_name, _ in stage_models.values()
    ]
    return {
        "openapi": "3.1.0",
        "info": {"title": "Microfluidics Compute API", "version": "1.0.0"},
        "paths": {
            "/api/v1/compute": {
                "post": {
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "oneOf": union_refs,
                                    "discriminator": {
                                        "propertyName": "experiment_id",
                                        "mapping": {
                                            stage_id: f"#/components/schemas/{request_name}"
                                            for stage_id, (
                                                request_name,
                                                _,
                                            ) in stage_models.items()
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {"200": {"description": "Terminal compute response"}},
                }
            },
            "/api/v1/compute/{request_id}": {
                "get": {
                    "parameters": [
                        {
                            "name": "request_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "minLength": 1},
                        }
                    ],
                    "responses": {
                        "200": {"description": "Current or terminal compute response"},
                        "404": {"description": "Unknown request ID"},
                    },
                }
            },
            "/api/v1/compute/{request_id}/cancel": {
                "post": {
                    "parameters": [
                        {
                            "name": "request_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string", "minLength": 1},
                        }
                    ],
                    "responses": {
                        "202": {"description": "Cancellation accepted"},
                        "404": {"description": "Unknown request ID"},
                    },
                }
            },
            "/health": {"get": {"responses": {"200": {"description": "Liveness OK"}}}},
            "/ready": {
                "get": {
                    "responses": {
                        "200": {"description": "Service ready"},
                        "503": {"description": "Service not ready"},
                    }
                }
            },
        },
        "components": {
            "schemas": {
                **parameter_schemas,
                **request_schemas,
            }
        },
    }


class ComputeRequestHandler(BaseHTTPRequestHandler):
    """API handler for the stateless compute surface."""

    server_version = "MicrofluidicsCompute/1.0"

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RequestBodyError(
                status=HTTPStatus.BAD_REQUEST,
                code="invalid_content_length",
                message="Content-Length header must be a valid integer.",
            ) from exc
        if length < 0:
            raise RequestBodyError(
                status=HTTPStatus.BAD_REQUEST,
                code="invalid_content_length",
                message="Content-Length must not be negative.",
            )
        if length > int(self.settings.service_max_request_bytes):
            raise RequestBodyError(
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                code="request_too_large",
                message=(
                    "Request body exceeds SERVICE_MAX_REQUEST_BYTES="
                    f"{int(self.settings.service_max_request_bytes)}."
                ),
            )
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _request_path(self) -> str:
        return urlsplit(self.path).path

    def _route_parts(self) -> list[str]:
        return [part for part in self._request_path().split("/") if part]

    def _extract_bearer_token(self) -> str | None:
        authorization = self.headers.get("Authorization", "").strip()
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            return token or None
        return None

    def _ensure_authorized(self) -> bool:
        configured_key = str(self.settings.service_api_key).strip()
        if not configured_key:
            return True
        presented_key = (
            self._extract_bearer_token()
            or self.headers.get("X-API-Key", "").strip()
            or None
        )
        if presented_key and api_key_matches(presented_key, configured_key):
            return True
        payload = ErrorPayloadV1(
            code="unauthorized",
            message=(
                "Missing or invalid service API key. "
                "Use Authorization: Bearer <key> or X-API-Key."
            ),
        ).to_dict()
        body = _json_bytes(payload)
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)
        return False

    @property
    def service(self) -> ComputeExecutionService:
        return self.server.service  # type: ignore[attr-defined]

    @property
    def settings(self) -> RuntimeSettings:
        return self.server.settings  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        path = self._request_path()
        parts = self._route_parts()

        if path == "/api/v1/compute":
            self._handle_compute()
            return

        if (
            len(parts) == 5
            and parts[:3] == ["api", "v1", "compute"]
            and parts[4] == "cancel"
        ):
            self._handle_cancel(unquote(parts[3]))
            return

        self._write_json(
            HTTPStatus.NOT_FOUND,
            ErrorPayloadV1(
                code="not_found",
                message=f"Unknown endpoint: {self.path}",
            ).to_dict(),
        )

    def do_GET(self) -> None:  # noqa: N802
        path = self._request_path()
        parts = self._route_parts()
        if path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/ready":
            self._handle_ready()
            return
        if path == "/openapi.json":
            self._write_json(
                HTTPStatus.OK,
                # Same rule as submit validation, so the advertised pattern
                # cannot promise a constraint the service does not enforce
                # (or demand a prefix it will not accept) in either
                # runtime.
                build_openapi_schema(
                    input_prefixes=stage_input_prefixes(self.settings),
                ),
            )
            return

        if len(parts) == 4 and parts[:3] == ["api", "v1", "compute"]:
            self._handle_get(unquote(parts[3]))
            return

        self._write_json(
            HTTPStatus.NOT_FOUND,
            ErrorPayloadV1(
                code="not_found",
                message=f"Unknown endpoint: {self.path}",
            ).to_dict(),
        )

    def _handle_ready(self) -> None:
        """Readiness for every door this process publishes, not just this one.

        `/health` stays a pure liveness signal: it must not fail because MCP
        is gone, or a process supervisor could interrupt active compute work.
        `/ready` is the separate signal used to stop accepting new work when
        the advertised MCP endpoint is unavailable.
        """

        checks = {"compute_api": "ok"}
        ready = True
        if self.settings.mcp_enabled:
            probe = getattr(self.server, "mcp_ready", None)
            serving = False
            if probe is not None:
                try:
                    serving = bool(probe())
                except Exception:  # noqa: BLE001 - a probe must not 500
                    logger.exception("MCP readiness probe raised")
            checks["mcp_transport"] = "ok" if serving else "down"
            ready = serving

        payload: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        }
        if not ready:
            payload["message"] = (
                "MCP_ENABLED is true but the MCP transport is not serving on "
                f"{self.settings.mcp_host}:{self.settings.mcp_port}. Check the "
                "compute log for 'MCP HTTP transport stopped' or 'Failed to "
                "start MCP HTTP transport'."
            )
        self._write_json(
            HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE,
            payload,
        )

    def _ensure_enabled(self) -> bool:
        if self.settings.service_enabled:
            return True
        self._write_json(
            HTTPStatus.SERVICE_UNAVAILABLE,
            ErrorPayloadV1(
                code="service_disabled",
                message="SERVICE_ENABLED=false. Stateless compute service is disabled.",
            ).to_dict(),
        )
        return False

    def _handle_compute(self) -> None:
        if not self._ensure_enabled():
            return
        if not self._ensure_authorized():
            return

        try:
            payload = self._read_json()
            request = SubmitRunRequestV1.from_dict(payload)
        except RequestBodyError as exc:
            self._write_json(
                exc.status,
                ErrorPayloadV1(
                    code=exc.code,
                    message=exc.message,
                ).to_dict(),
            )
            return
        except ContractValidationError as exc:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                ErrorPayloadV1(
                    code="invalid_contract",
                    message=str(exc),
                ).to_dict(),
            )
            return
        except json.JSONDecodeError:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                ErrorPayloadV1(
                    code="invalid_json",
                    message="Request body must be valid JSON.",
                ).to_dict(),
            )
            return

        try:
            response = self.service.execute(request)
        except RequestValidationError as exc:
            self._write_json(
                exc.http_status,
                ErrorPayloadV1(
                    code=exc.code,
                    message=exc.message,
                    details=exc.details,
                ).to_dict(),
            )
            return
        except IdempotencyConflictError as exc:
            self._write_json(
                HTTPStatus.CONFLICT,
                ErrorPayloadV1(code="idempotency_conflict", message=str(exc)).to_dict(),
            )
            return
        except StageInputError as exc:
            self._write_json(
                exc.http_status,
                ErrorPayloadV1(code=exc.code, message=str(exc)).to_dict(),
            )
            return
        self._write_json(HTTPStatus.OK, response.to_dict())

    def _handle_get(self, request_id: str) -> None:
        if not self._ensure_enabled() or not self._ensure_authorized():
            return
        response = self.service.get(request_id)
        if response is None:
            self._write_json(
                HTTPStatus.NOT_FOUND,
                ErrorPayloadV1(
                    code="request_not_found",
                    message=f"No completed compute request with request_id={request_id!r}.",
                ).to_dict(),
            )
            return
        self._write_json(HTTPStatus.OK, response.to_dict())

    def _handle_cancel(self, request_id: str) -> None:
        if not self._ensure_enabled():
            return
        if not self._ensure_authorized():
            return

        summary = self.service.cancel(request_id)

        if (
            not summary["cancelled"]
            and not summary["already_terminal"]
            and not summary.get("cancellation_requested")
        ):
            self._write_json(
                HTTPStatus.NOT_FOUND,
                ErrorPayloadV1(
                    code="request_not_found",
                    message=(f"No local run is known for request_id={request_id!r}."),
                ).to_dict(),
            )
            return

        self._write_json(HTTPStatus.ACCEPTED, summary)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)


class ComputeApiServer(ThreadingHTTPServer):
    service: ComputeExecutionService
    settings: RuntimeSettings
    # Answers "is the MCP transport serving?" for /ready. None while
    # MCP_ENABLED is true means the transport never started at all, which is
    # not ready either.
    mcp_ready: Callable[[], bool] | None


def create_http_server(
    settings: RuntimeSettings,
    service: ComputeExecutionService,
    mcp_ready: Callable[[], bool] | None = None,
) -> ComputeApiServer:
    server = ComputeApiServer(
        (settings.service_host, settings.service_port),
        ComputeRequestHandler,
    )
    server.service = service
    server.settings = settings
    server.mcp_ready = mcp_ready
    return server
