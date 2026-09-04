"""Keeps the MCP tool surface honest against backend.app.stage_registry.

_STAGE_TOOL_BY_ID below is itself hand-maintained, so it can drift from
STAGE_REGISTRY the same way the tools it checks can: someone adds a fifth
stage to the registry and forgets this file exists. Without a dedicated
check, that shows up as a bare `KeyError: 'stage_new'` raised from inside
a dict subscript in the two tests below - a failure that names no
suspect and points nowhere near backend/app/stage_registry.py or this
file's mapping. test_stage_tool_mapping_covers_the_full_registry exists
to fail first and loudly, with a message that says exactly what to edit.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.app.stage_registry import STAGE_REGISTRY
from microfluidics_contracts import RuntimeSettings
from microfluidics_mcp.outputs import (
    CONSUMER_PARAMETERS,
    STAGE_PRODUCTS,
    supplied_parameters,
)
from microfluidics_mcp.server import build_mcp_server

_STAGE_TOOL_BY_ID = {
    "stage_import": "cfd_run_import",
    "stage_flow": "cfd_run_flow",
    "stage_transport": "cfd_run_transport",
    "stage_thermal": "cfd_run_thermal",
    "stage_reactive_transport": "cfd_run_reactive_transport",
}


def _tools(tmp_path: Path, monkeypatch) -> dict:
    """Build a local-only server and return its tools, keyed by name."""
    mcp = build_mcp_server(
        service=object(),
        settings=RuntimeSettings.from_env(),
        project_root=tmp_path,
    )
    return {tool.name: tool for tool in asyncio.run(mcp.list_tools())}


def _tool_name_for(stage_id: str) -> str:
    """Guarded lookup so a new stage id fails with an actionable message
    instead of a bare KeyError pointing at neither file."""

    try:
        return _STAGE_TOOL_BY_ID[stage_id]
    except KeyError as exc:
        raise AssertionError(
            f"Stage {stage_id!r} is in STAGE_REGISTRY but "
            "test_registry_drift.py's _STAGE_TOOL_BY_ID does not know its "
            "tool name. Add stage_id -> tool name to _STAGE_TOOL_BY_ID in "
            "mcp-server/tests/test_registry_drift.py."
        ) from exc


def test_stage_tool_mapping_covers_the_full_registry() -> None:
    """Runs before the tests that index _STAGE_TOOL_BY_ID, so a stage added
    to STAGE_REGISTRY without a matching entry here fails on this test, with
    a message naming the missing stage id and the file to edit - not on a
    bare KeyError raised from inside one of the tests below."""

    missing = sorted(set(STAGE_REGISTRY) - set(_STAGE_TOOL_BY_ID))
    assert not missing, (
        f"STAGE_REGISTRY has stage id(s) {missing} that "
        "test_registry_drift.py's _STAGE_TOOL_BY_ID does not map to a tool "
        "name. Add each one to _STAGE_TOOL_BY_ID in "
        "mcp-server/tests/test_registry_drift.py, then confirm "
        "tools_stages.register exposes a matching MCP tool."
    )


def test_every_registered_stage_has_a_tool(tmp_path, monkeypatch) -> None:
    tools = _tools(tmp_path, monkeypatch)

    for stage_id in STAGE_REGISTRY:
        tool_name = _tool_name_for(stage_id)
        assert tool_name in tools, (
            f"Stage {stage_id!r} has no MCP tool. Add it to tools_stages.register."
        )


def test_stage_tool_arguments_match_the_registry(tmp_path, monkeypatch) -> None:
    tools = _tools(tmp_path, monkeypatch)

    for stage_id, definition in STAGE_REGISTRY.items():
        tool_name = _tool_name_for(stage_id)
        assert tool_name in tools, (
            f"Stage {stage_id!r} has no MCP tool. Add it to tools_stages.register."
        )
        schema = tools[tool_name].inputSchema
        exposed = set(schema["properties"]) - {"request_id"}
        assert exposed == set(definition.parameters_model.allowed_fields), (
            f"Tool for {stage_id!r} drifted from the stage registry."
        )


def test_every_registered_stage_has_a_producer_entry() -> None:
    """`STAGE_PRODUCTS` is hand-maintained because the registry does not know
    what a stage's entrypoint calls its own files - `staged_path` is the name
    the *consumer* mounts. A fifth stage added to STAGE_REGISTRY without an
    entry here would answer `cfd_get_run` with an empty `outputs` and no
    explanation."""

    missing = sorted(set(STAGE_REGISTRY) - set(STAGE_PRODUCTS))
    assert not missing, (
        f"STAGE_REGISTRY has stage id(s) {missing} with no entry in "
        "STAGE_PRODUCTS. Read that stage's entrypoint in experiments/gmsh/, "
        "then add stage_id -> StageProducts(produced=..., echoed=...) to "
        "mcp-server/src/microfluidics_mcp/outputs.py."
    )


def test_every_stage_input_can_be_supplied_by_a_stage() -> None:
    """The registry is the source of truth for which parameters exist; the
    producer table has to keep up. A stage that gains an input no stage
    produces and no stage echoes is a link the agent cannot follow: it would
    sit in `missing` on every run forever."""

    unsupplied = sorted(set(CONSUMER_PARAMETERS) - supplied_parameters())
    assert not unsupplied, (
        f"Stage input parameter(s) {unsupplied} are declared in STAGE_REGISTRY "
        "but no stage produces or echoes them. Find the entrypoint in "
        "experiments/gmsh/ that writes each one, and add it to the producing "
        "stage's StageProducts.produced in "
        "mcp-server/src/microfluidics_mcp/outputs.py (or to the consuming "
        "stage's .echoed if only the agent can supply it)."
    )


def test_standalone_server_exposes_local_mesh_tools(tmp_path, monkeypatch) -> None:
    tools = _tools(tmp_path, monkeypatch)

    assert "cfd_validate_reactive_case" in tools
    assert "cfd_register_local_mesh" in tools
