"""The MCP door must refuse the arguments the compute API refuses.

`@mcp.tool()` builds a pydantic model from each tool's signature and that
model, left as the SDK builds it, silently drops keys it does not know: a run
started on defaults while the agent believed it had asked for something else.
`POST /api/v1/compute` answers the identical body with `invalid_parameters`.
These tests go through `FastMCP.call_tool` and `FastMCP.list_tools` - what a
client actually reaches - rather than the tool functions, because the tool
functions are exactly the layer that never sees the offending key.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from backend.app.stage_registry import StageParametersError, parse_stage_parameters
from microfluidics_contracts import RuntimeSettings
from microfluidics_mcp.server import build_mcp_server
from microfluidics_mcp.storage import LocalMeshStore
from microfluidics_mcp.strict_tools import unsupported_parameters_message

# One valid call per tool of the standalone shape, so the unknown-argument case
# differs from the accepted one by exactly the bogus key. The values need not
# name anything real: the refusal must happen before the service is reached,
# which _ExplodingService turns into a test failure rather than a pass.
_VALID_ARGUMENTS: dict[str, dict] = {
    "cfd_list_meshes": {"limit": 1},
    "cfd_register_local_mesh": {"path": "data/meshes/pipe.msh"},
    "cfd_validate_reactive_case": {"reactive_case": {}},
    "cfd_run_import": {"mesh_path": "in/pipe.msh"},
    "cfd_run_flow": {"mesh_npz_path": "in/mesh.npz", "num_steps": 10},
    "cfd_run_transport": {
        "mesh_npz_path": "in/mesh.npz",
        "flow_coupling_metadata_path": "in/meta.json",
        "flow_face_flux_path": "in/flux.npy",
        "num_steps": 10,
    },
    "cfd_run_thermal": {
        "mesh_path": "in/pipe.msh",
        "flow_summary_path": "in/summary.json",
        "flow_coupling_metadata_path": "in/meta.json",
        "flow_face_flux_path": "in/flux.npy",
        "flow_face_to_cells_path": "in/face_to_cells.npy",
        "flow_cell_volumes_path": "in/cell_volumes.npy",
        "num_steps": 10,
    },
    "cfd_run_reactive_transport": {
        "mesh_npz_path": "in/mesh.npz",
        "flow_summary_path": "in/summary.json",
        "flow_coupling_metadata_path": "in/meta.json",
        "flow_face_flux_path": "in/flux.npy",
        "flow_face_to_cells_path": "in/face_to_cells.npy",
        "flow_cell_volumes_path": "in/cell_volumes.npy",
        "reactive_case_path": "in/reactive_case.json",
    },
    "cfd_get_run": {"request_id": "flow-1"},
    "cfd_list_artifacts": {"request_id": "flow-1"},
    "cfd_cancel_run": {"request_id": "flow-1"},
    "cfd_get_artifact": {"request_id": "flow-1", "key": "summary.json"},
}


class _ExplodingService:
    """Any tool that reaches the service has already accepted the call."""

    def submit_async(self, request):
        raise AssertionError("an unknown argument reached the service")

    def get(self, request_id):
        raise AssertionError("an unknown argument reached the service")

    def cancel(self, request_id):
        raise AssertionError("an unknown argument reached the service")


def _server(tmp_path: Path, monkeypatch):
    """Build the local-only server."""
    settings = replace(RuntimeSettings.from_env(), service_enabled=True)
    mesh_store = LocalMeshStore(
        project_root=tmp_path,
        mesh_root=tmp_path / "data" / "meshes",
    )
    return build_mcp_server(
        service=_ExplodingService(),
        settings=settings,
        project_root=tmp_path,
        mesh_store=mesh_store,
    )


def _tool_names(mcp) -> list[str]:
    return [tool.name for tool in asyncio.run(mcp.list_tools())]


def test_every_published_tool_forbids_additional_properties(
    tmp_path, monkeypatch
) -> None:
    """The schema must say what the local server enforces."""

    mcp = _server(tmp_path, monkeypatch)

    tools = asyncio.run(mcp.list_tools())

    assert tools, "the runtime published no tools at all"
    for tool in tools:
        assert tool.inputSchema.get("additionalProperties") is False, tool.name
        # Per-field typing is what lets an agent call these tools correctly;
        # forbidding extras must not have flattened it away.
        assert tool.inputSchema.get("properties"), tool.name


def test_every_standalone_tool_refuses_an_unknown_argument(
    tmp_path, monkeypatch
) -> None:
    mcp = _server(tmp_path, monkeypatch)
    names = _tool_names(mcp)

    assert set(names) <= set(_VALID_ARGUMENTS), "a new tool has no case here"
    for name in names:
        arguments = dict(_VALID_ARGUMENTS[name], bogus_parameter="x")
        with pytest.raises(ToolError) as excinfo:
            asyncio.run(mcp.call_tool(name, arguments))
        assert str(excinfo.value) == (
            "invalid_parameters: Unsupported parameter(s): 'bogus_parameter'."
        ), name


def test_a_misspelled_optional_argument_is_reported_not_ignored(
    tmp_path, monkeypatch
) -> None:
    """The case that costs an agent a run: 'devise' for 'device' used to start
    a run on the default device while the agent believed it had chosen one."""

    mcp = _server(tmp_path, monkeypatch)

    with pytest.raises(ToolError) as excinfo:
        asyncio.run(
            mcp.call_tool(
                "cfd_run_flow",
                {"mesh_npz_path": "in/mesh.npz", "num_steps": 10, "devise": "cuda"},
            )
        )

    assert "'devise'" in str(excinfo.value)


def test_several_unknown_arguments_are_all_named(tmp_path, monkeypatch) -> None:
    mcp = _server(tmp_path, monkeypatch)

    with pytest.raises(ToolError) as excinfo:
        asyncio.run(
            mcp.call_tool(
                "cfd_run_import",
                {"mesh_path": "in/pipe.msh", "zebra": 1, "alpha": 2},
            )
        )

    assert str(excinfo.value) == (
        "invalid_parameters: Unsupported parameter(s): 'alpha', 'zebra'."
    )


def test_the_refusal_matches_the_compute_api_word_for_word() -> None:
    """Both doors of this process answer the same body the same way.

    `parse_stage_parameters` is what `POST /api/v1/compute` runs; if its
    wording moves, the MCP message must move with it rather than the two
    quietly diverging.
    """

    with pytest.raises(StageParametersError) as excinfo:
        parse_stage_parameters(
            "stage_import",
            {"mesh_path": "in/pipe.msh", "bogus_parameter": "x"},
        )

    assert str(excinfo.value) == unsupported_parameters_message(["bogus_parameter"])


def test_the_unknown_tool_error_is_left_to_the_tool_manager(
    tmp_path, monkeypatch
) -> None:
    """A tool this runtime does not register must still say so, not be
    reported as a parameter problem."""

    mcp = _server(tmp_path, monkeypatch)

    with pytest.raises(ToolError, match="Unknown tool"):
        asyncio.run(mcp.call_tool("unsupported_tool", {"filename": "a.msh"}))


def test_a_declared_argument_is_still_accepted(tmp_path, monkeypatch) -> None:
    """The guard must not have made the tools unusable."""

    (tmp_path / "data" / "meshes").mkdir(parents=True)
    mcp = _server(tmp_path, monkeypatch)

    result = asyncio.run(mcp.call_tool("cfd_list_meshes", {"limit": 1}))

    assert result[1] == {"meshes": [], "storage": "local"}
