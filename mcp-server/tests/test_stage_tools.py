from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from backend.app.stage_registry import Device
from microfluidics_contracts import ExecutionResponseV1, RunStatus, RuntimeSettings
from microfluidics_mcp.context import ServerContext
from microfluidics_mcp.storage import LocalMeshStore
from microfluidics_mcp.tools_stages import register, submit_stage


class _RecordingService:
    """Records submitted requests; `get()` reports no observable run.

    Matches the shape of ComputeExecutionService.get(), which returns None
    for a request_id it has never seen. Tests that care about a specific
    terminal status use `_TerminalService` instead.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def submit_async(self, request):
        self.calls.append((request.experiment_id, dict(request.parameters)))
        return request.request_id or "generated-id"

    def get(self, request_id):
        return None


def _enabled_settings() -> RuntimeSettings:
    """The stage tools are gated on SERVICE_ENABLED, exactly as the compute
    API's submit route is, so these tests ask for an enabled service
    explicitly instead of inheriting whatever the environment exports.
    test_service_gate.py owns the disabled half."""

    return replace(RuntimeSettings.from_env(), service_enabled=True)


def _context(tmp_path: Path) -> ServerContext:
    settings = _enabled_settings()
    return ServerContext(
        service=_RecordingService(),
        settings=settings,
        project_root=tmp_path,
        mesh_store=LocalMeshStore(project_root=tmp_path, mesh_root=tmp_path),
    )


def test_submit_stage_returns_a_pending_handle(tmp_path) -> None:
    ctx = _context(tmp_path)

    result = submit_stage(
        ctx,
        experiment_id="stage_import",
        parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
        request_id="import-1",
    )

    assert result == {
        "request_id": "import-1",
        "experiment_id": "stage_import",
        "status": "pending",
    }
    assert ctx.service.calls == [
        ("stage_import", {"mesh_path": "data/inputs/gmsh/pipe.msh"})
    ]


def test_submit_stage_reports_a_generated_request_id(tmp_path) -> None:
    ctx = _context(tmp_path)

    result = submit_stage(
        ctx,
        experiment_id="stage_import",
        parameters={"mesh_path": "data/inputs/gmsh/pipe.msh"},
        request_id=None,
    )

    assert result["request_id"] == "generated-id"


def test_submit_stage_surfaces_validation_errors_verbatim(tmp_path) -> None:
    class _RejectingService:
        def submit_async(self, request):
            from backend.app.request_validation import RequestValidationError

            raise RequestValidationError(
                code="invalid_parameters",
                message="'num_steps' is required.",
                http_status=422,
            )

    settings = _enabled_settings()
    ctx = ServerContext(
        service=_RejectingService(),
        settings=settings,
        project_root=tmp_path,
        mesh_store=LocalMeshStore(project_root=tmp_path, mesh_root=tmp_path),
    )

    with pytest.raises(ToolError, match="invalid_parameters: 'num_steps' is required."):
        submit_stage(
            ctx,
            experiment_id="stage_flow",
            parameters={},
            request_id=None,
        )


def test_submit_stage_drops_unset_optional_parameters(tmp_path) -> None:
    ctx = _context(tmp_path)

    submit_stage(
        ctx,
        experiment_id="stage_flow",
        parameters={
            "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
            "num_steps": 100,
            "device": None,
        },
        request_id="flow-1",
    )

    assert ctx.service.calls[0][1] == {
        "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
        "num_steps": 100,
    }


class _TerminalService:
    """A service whose run is already finished by the time submit returns.

    Reproduces ComputeExecutionService.submit_async's early return for a
    request_id that maps to an existing record (service.py:335-336), and
    _claim() handing back a record whose run has already reached a terminal
    RunStatus (service.py:159-163)."""

    def __init__(self, status: RunStatus) -> None:
        self._status = status
        self.calls: list[tuple[str, dict]] = []

    def submit_async(self, request):
        self.calls.append((request.experiment_id, dict(request.parameters)))
        return request.request_id or "generated-id"

    def get(self, request_id):
        return ExecutionResponseV1(
            request_id=request_id,
            run_id="run-1",
            status=self._status,
        )


def test_submit_stage_reports_the_services_real_terminal_status(tmp_path) -> None:
    settings = _enabled_settings()
    ctx = ServerContext(
        service=_TerminalService(RunStatus.SUCCEEDED),
        settings=settings,
        project_root=tmp_path,
        mesh_store=LocalMeshStore(project_root=tmp_path, mesh_root=tmp_path),
    )

    result = submit_stage(
        ctx,
        experiment_id="stage_flow",
        parameters={
            "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
            "num_steps": 100,
        },
        request_id="flow-1",
    )

    assert result["status"] == "succeeded"
    assert result["status"] != "pending"


def _register_context(tmp_path: Path) -> tuple[ServerContext, FastMCP]:
    ctx = _context(tmp_path)
    mcp = FastMCP("test")
    register(mcp, ctx)
    return ctx, mcp


def test_register_exposes_exactly_the_five_stage_tools(tmp_path) -> None:
    _, mcp = _register_context(tmp_path)

    tools = asyncio.run(mcp.list_tools())

    assert {tool.name for tool in tools} == {
        "cfd_run_import",
        "cfd_run_flow",
        "cfd_run_transport",
        "cfd_run_thermal",
        "cfd_run_reactive_transport",
    }


def test_cfd_run_flow_reaches_the_service_through_the_tool_layer(tmp_path) -> None:
    ctx, mcp = _register_context(tmp_path)

    # cfd_run_flow returns `dict[str, Any]`, so FastMCP hands call_tool()
    # both the rendered text content and the structured payload; the
    # structured half is what we assert on.
    _content, payload = asyncio.run(
        mcp.call_tool(
            "cfd_run_flow",
            {
                "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
                "num_steps": 100,
                "device": "cuda",
                "numerical_profile": "no_slip_tjunction_validation_v1",
                "flow_stop_physical_time": 0.62,
                "snapshot_time_interval": 0.02,
                "request_id": "flow-1",
            },
        )
    )

    assert payload["request_id"] == "flow-1"
    assert payload["experiment_id"] == "stage_flow"
    assert ctx.service.calls == [
        (
            "stage_flow",
            {
                "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
                "num_steps": 100,
                "device": "cuda",
                "numerical_profile": "no_slip_tjunction_validation_v1",
                "flow_stop_physical_time": 0.62,
                "snapshot_time_interval": 0.02,
            },
        )
    ]


def test_cfd_run_flow_omitted_device_does_not_reach_the_service_as_none(
    tmp_path,
) -> None:
    """The default lives on the tool signature; only the tool layer can prove
    it never becomes an explicit `device=None` sent on to the service."""

    ctx, mcp = _register_context(tmp_path)

    asyncio.run(
        mcp.call_tool(
            "cfd_run_flow",
            {
                "mesh_npz_path": "data/inputs/cfd/import-1/mesh.npz",
                "num_steps": 100,
                "request_id": "flow-2",
            },
        )
    )

    _, parameters = ctx.service.calls[-1]
    assert "device" not in parameters


def test_device_schema_mirrors_the_stage_registry_device_enum(tmp_path) -> None:
    """cfd_run_flow/transport/thermal's `device` schema must enumerate exactly
    backend.app.stage_registry.Device's members, so a future third device
    cannot drift the tool schema and the registry apart silently."""

    _, mcp = _register_context(tmp_path)
    expected = {member.value for member in Device}

    tools = asyncio.run(mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    for name in ("cfd_run_flow", "cfd_run_transport", "cfd_run_thermal"):
        device_schema = by_name[name].inputSchema["properties"]["device"]
        enum_branch = next(
            branch for branch in device_schema["anyOf"] if "enum" in branch
        )
        assert set(enum_branch["enum"]) == expected


def test_flow_profile_schema_is_closed_and_exposes_time_controls(tmp_path) -> None:
    _, mcp = _register_context(tmp_path)
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    properties = tools["cfd_run_flow"].inputSchema["properties"]
    profile_schema = properties["numerical_profile"]
    enum_branch = next(branch for branch in profile_schema["anyOf"] if "enum" in branch)

    assert set(enum_branch["enum"]) == {
        "default",
        "no_slip_tjunction_validation_v1",
    }
    assert "flow_stop_physical_time" in properties
    assert "snapshot_time_interval" in properties
