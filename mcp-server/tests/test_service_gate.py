"""SERVICE_ENABLED must mean the same thing on the MCP door as on the HTTP one.

`backend/app/http_api.py::_ensure_enabled` gates exactly three routes -
submit, status and cancel - with 503 `service_disabled`, and leaves
/health and /openapi.json up. The MCP transport shares one
ComputeExecutionService with that API, so the same flag has to refuse the
same operations here; otherwise SERVICE_ENABLED=false plus MCP_ENABLED=true
leaves a second, unguarded entry point to the compute service.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from microfluidics_contracts import RuntimeSettings
from microfluidics_mcp.context import ServerContext
from microfluidics_mcp.server import build_mcp_server
from microfluidics_mcp.storage import LocalMeshStore
from microfluidics_mcp.tools_meshes import list_meshes, register_local_mesh
from microfluidics_mcp.tools_runs import (
    cancel_run,
    get_artifact,
    get_run,
    list_artifacts,
)
from microfluidics_mcp.tools_stages import submit_stage

# The literal the compute API answers with for this case
# (http_api.py::_ensure_enabled). Reused rather than reinvented so an agent
# meets one error vocabulary across both doors.
_CODE = "service_disabled"


class _ExplodingService:
    """Reaching this service at all is the failure being tested for.

    A disabled service must be refused before any call is made, not after
    the service politely declines - the point of the flag is that no work
    is started.
    """

    def submit_async(self, request):
        raise AssertionError("submit_async reached a disabled service")

    def get(self, request_id):
        raise AssertionError("get reached a disabled service")

    def cancel(self, request_id):
        raise AssertionError("cancel reached a disabled service")


class _RecordingService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def submit_async(self, request):
        self.calls.append("submit_async")
        return request.request_id or "generated-id"

    def get(self, request_id):
        self.calls.append("get")
        return None

    def cancel(self, request_id):
        self.calls.append("cancel")
        # A cancelled job, not an empty summary: this fake stands for a service
        # that was reached, and cancel_run reads an empty summary as
        # request_not_found (the compute API's HTTP 404 for the same call).
        return {
            "request_id": request_id,
            "cancelled": ["job-1"],
            "already_terminal": [],
        }


def _mesh_root(tmp_path: Path) -> Path:
    mesh_root = tmp_path / "data" / "meshes"
    mesh_root.mkdir(parents=True, exist_ok=True)
    (mesh_root / "pipe.msh").write_text("mesh", encoding="utf-8")
    return mesh_root


def _context(tmp_path: Path, *, enabled: bool, service=None) -> ServerContext:
    return ServerContext(
        service=service if service is not None else _ExplodingService(),
        settings=replace(RuntimeSettings.from_env(), service_enabled=enabled),
        project_root=tmp_path,
        mesh_store=LocalMeshStore(
            project_root=tmp_path,
            mesh_root=_mesh_root(tmp_path),
        ),
    )


def test_submit_is_refused_while_the_service_is_disabled(tmp_path) -> None:
    ctx = _context(tmp_path, enabled=False)

    with pytest.raises(ToolError, match=_CODE) as excinfo:
        submit_stage(
            ctx,
            experiment_id="stage_import",
            parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
            request_id="import-1",
        )

    assert "SERVICE_ENABLED" in str(excinfo.value)


def test_status_is_refused_while_the_service_is_disabled(tmp_path) -> None:
    ctx = _context(tmp_path, enabled=False)

    with pytest.raises(ToolError, match=_CODE):
        get_run(ctx, request_id="flow-1")


def test_cancel_is_refused_while_the_service_is_disabled(tmp_path) -> None:
    ctx = _context(tmp_path, enabled=False)

    with pytest.raises(ToolError, match=_CODE):
        cancel_run(ctx, request_id="flow-1")


def test_artifact_reads_are_refused_while_the_service_is_disabled(tmp_path) -> None:
    """Both artifact tools resolve the run through the same service.get()
    the gated GET /api/v1/compute/{request_id} route serves."""

    ctx = _context(tmp_path, enabled=False)

    with pytest.raises(ToolError, match=_CODE):
        list_artifacts(ctx, request_id="flow-1")
    with pytest.raises(ToolError, match=_CODE):
        get_artifact(ctx, request_id="flow-1", key="summary.json")


def test_the_refusal_names_the_setting_an_operator_must_change(tmp_path) -> None:
    ctx = _context(tmp_path, enabled=False)

    with pytest.raises(ToolError) as excinfo:
        get_run(ctx, request_id="flow-1")

    message = str(excinfo.value)
    assert message.startswith(f"{_CODE}: ")
    assert "SERVICE_ENABLED=false" in message
    assert "SERVICE_ENABLED=true" in message


_DISABLED_TOOL_CALLS = (
    ("cfd_run_import", {"mesh_path": "data/inputs/gmsh/pipe.msh"}),
    (
        "cfd_run_flow",
        {"mesh_npz_path": "in/mesh.npz", "num_steps": 10},
    ),
    (
        "cfd_run_transport",
        {
            "mesh_npz_path": "in/mesh.npz",
            "flow_coupling_metadata_path": "in/meta.json",
            "flow_face_flux_path": "in/flux.npy",
            "num_steps": 10,
        },
    ),
    (
        "cfd_run_thermal",
        {
            "mesh_path": "in/pipe.msh",
            "flow_summary_path": "in/summary.json",
            "flow_coupling_metadata_path": "in/meta.json",
            "flow_face_flux_path": "in/flux.npy",
            "flow_face_to_cells_path": "in/face_to_cells.npy",
            "flow_cell_volumes_path": "in/cell_volumes.npy",
            "num_steps": 10,
        },
    ),
    (
        "cfd_run_reactive_transport",
        {
            "mesh_npz_path": "in/mesh.npz",
            "flow_summary_path": "in/summary.json",
            "flow_coupling_metadata_path": "in/meta.json",
            "flow_face_flux_path": "in/flux.npy",
            "flow_face_to_cells_path": "in/face_to_cells.npy",
            "flow_cell_volumes_path": "in/cell_volumes.npy",
            "reactive_case_path": "in/reactive_case.json",
        },
    ),
    ("cfd_get_run", {"request_id": "flow-1"}),
    ("cfd_list_artifacts", {"request_id": "flow-1"}),
    ("cfd_cancel_run", {"request_id": "flow-1"}),
    ("cfd_get_artifact", {"request_id": "flow-1", "key": "summary.json"}),
)


def _server(tmp_path: Path, *, enabled: bool, service=None):
    return build_mcp_server(
        service=service if service is not None else _ExplodingService(),
        settings=replace(RuntimeSettings.from_env(), service_enabled=enabled),
        project_root=tmp_path,
        mesh_store=LocalMeshStore(
            project_root=tmp_path,
            mesh_root=_mesh_root(tmp_path),
        ),
    )


@pytest.mark.parametrize(("tool_name", "arguments"), _DISABLED_TOOL_CALLS)
def test_every_service_tool_is_refused_through_the_tool_layer(
    tmp_path, tool_name, arguments
) -> None:
    mcp = _server(tmp_path, enabled=False)

    with pytest.raises(ToolError, match=_CODE):
        asyncio.run(mcp.call_tool(tool_name, arguments))


def test_the_gate_covers_every_tool_that_can_reach_the_service(tmp_path) -> None:
    """Guards the parametrisation above: a new service-backed tool added
    without a gate would otherwise be tested by nobody. Storage discovery,
    ingress, and stateless reactive-case validation are exempt, matching
    /health and /openapi.json staying up while SERVICE_ENABLED=false."""

    mcp = _server(tmp_path, enabled=False)
    exposed = {tool.name for tool in asyncio.run(mcp.list_tools())}
    non_service_tools = {
        "cfd_list_meshes",
        "cfd_register_local_mesh",
        "cfd_validate_reactive_case",
    }

    assert exposed - non_service_tools == {name for name, _ in _DISABLED_TOOL_CALLS}


def test_mesh_tools_stay_available_while_the_service_is_disabled(tmp_path) -> None:
    """They never touch the service, so disabling it must not hide them -
    the HTTP door keeps /health and /openapi.json answering too."""

    ctx = _context(tmp_path, enabled=False)

    assert list_meshes(ctx, limit=10)["storage"] == "local"
    assert register_local_mesh(ctx, path="data/meshes/pipe.msh")["key"] == (
        "data/meshes/pipe.msh"
    )


def test_an_enabled_service_still_reaches_the_service(tmp_path) -> None:
    service = _RecordingService()
    ctx = _context(tmp_path, enabled=True, service=service)

    submit_stage(
        ctx,
        experiment_id="stage_import",
        parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
        request_id="import-1",
    )
    cancel_run(ctx, request_id="import-1")

    assert service.calls == ["submit_async", "get", "cancel"]
