"""Shared dependencies handed to every tool implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from backend.app.service import ComputeExecutionService
from microfluidics_contracts import RuntimeSettings

from .storage import LocalMeshStore


@dataclass(frozen=True)
class ServerContext:
    service: ComputeExecutionService
    settings: RuntimeSettings
    project_root: Path
    mesh_store: LocalMeshStore

    @property
    def is_standalone(self) -> bool:
        return True

    def require_service_enabled(self) -> None:
        """Refuse service work while SERVICE_ENABLED=false.

        MCP and the compute API are two doors onto one
        ComputeExecutionService, so the operator's kill switch has to mean
        the same thing at both. `backend/app/http_api.py::_ensure_enabled`
        gates exactly three routes - submit (POST /api/v1/compute), status
        (GET /api/v1/compute/{request_id}) and cancel (POST
        /api/v1/compute/{request_id}/cancel) - and answers 503 with code
        "service_disabled"; the same code is reused here rather than
        inventing a second vocabulary for one policy.

        Routes that never touch the service (/health, /openapi.json) stay
        up on the HTTP side, and the mesh tools - which only talk to the
        mesh store - stay callable here for the same reason.

        Called from the three helpers every affected tool funnels through:
        `tools_stages.submit_stage`, `tools_runs._require_response` and
        `tools_runs.cancel_run`, mirroring the HTTP handler's three call
        sites one for one.
        """

        if self.settings.service_enabled:
            return
        raise ToolError(
            "service_disabled: SERVICE_ENABLED=false. Stateless compute "
            "service is disabled. Set SERVICE_ENABLED=true to enable it."
        )
