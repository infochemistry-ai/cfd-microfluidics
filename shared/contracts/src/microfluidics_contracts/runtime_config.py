"""Runtime settings for the standalone local compute service."""

from __future__ import annotations

import os
from dataclasses import dataclass


TRUTHY = {"1", "true", "yes", "on"}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in TRUTHY


def _strict_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _strict_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number.")
    return value


def _ascii_value(name: str, value: str) -> str:
    if value.isascii():
        return value
    raise ValueError(f"{name} must contain ASCII characters only.")


@dataclass(frozen=True)
class RuntimeSettings:
    """Centralized settings for local HTTP and MCP execution."""

    service_enabled: bool = False
    service_host: str = "127.0.0.1"
    service_port: int = 8091
    service_api_key: str = ""
    service_max_request_bytes: int = 1_048_576
    service_run_root: str = "results/service_runs"
    service_max_concurrent_runs: int = 1
    run_timeout_seconds: float = 14_400.0
    mcp_enabled: bool = False
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8092
    mcp_streamable_http_path: str = "/mcp"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        service_api_key = _ascii_value(
            "SERVICE_API_KEY", os.getenv("SERVICE_API_KEY", "").strip()
        )
        return cls(
            service_enabled=_as_bool(os.getenv("SERVICE_ENABLED"), False),
            service_host=os.getenv("SERVICE_HOST", "127.0.0.1").strip()
            or "127.0.0.1",
            service_port=_strict_positive_int_env("SERVICE_PORT", 8091),
            service_api_key=service_api_key,
            service_max_request_bytes=_strict_positive_int_env(
                "SERVICE_MAX_REQUEST_BYTES", 1_048_576
            ),
            service_run_root=os.getenv(
                "SERVICE_RUN_ROOT", "results/service_runs"
            ).strip()
            or "results/service_runs",
            service_max_concurrent_runs=_strict_positive_int_env(
                "SERVICE_MAX_CONCURRENT_RUNS", 1
            ),
            run_timeout_seconds=_strict_positive_float_env(
                "SERVICE_RUN_TIMEOUT_SECONDS", 14_400.0
            ),
            mcp_enabled=_as_bool(os.getenv("MCP_ENABLED"), False),
            mcp_host=os.getenv("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1",
            mcp_port=_strict_positive_int_env("MCP_PORT", 8092),
            mcp_streamable_http_path=os.getenv(
                "MCP_STREAMABLE_HTTP_PATH", "/mcp"
            ).strip()
            or "/mcp",
        )
