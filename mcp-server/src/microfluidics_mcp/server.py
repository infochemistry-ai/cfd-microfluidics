"""Assembles the MCP server from the service, settings and mesh storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from microfluidics_contracts import RuntimeSettings

from . import tools_meshes, tools_reactive_cases, tools_runs, tools_stages
from .context import ServerContext
from .storage import LocalMeshStore, build_mesh_store
from .strict_tools import StrictArgumentsFastMCP

_INSTRUCTIONS = """\
Runs microfluidics CFD stages as local subprocesses on this machine.

First get a mesh path: call cfd_list_meshes, or register a .msh file already
inside the project with cfd_register_local_mesh.

Order: cfd_run_import -> cfd_run_flow -> cfd_run_transport,
cfd_run_thermal, and/or cfd_run_reactive_transport.
Every cfd_run_* tool returns immediately with a request_id; poll cfd_get_run
until status reaches a terminal value: 'succeeded', 'failed' or 'cancelled'.
'pending' is the only non-terminal status - keep polling while you see it. Copy
the fields of cfd_get_run's 'outputs' object verbatim into the next stage -
their names are the next stage's parameters. Each run's 'outputs' holds only
what that stage can hand on, so keep the earlier runs' outputs. Thermal takes
mesh_path from the import run and its five flow_* keys from the flow run;
reactive transport takes mesh_npz_path from import, the same five flow_* keys
from flow, and a repository-relative reactive_case_path. Validate a case
with cfd_validate_reactive_case before starting reactive transport.
"""


def build_mcp_server(
    *,
    service: Any,
    settings: RuntimeSettings,
    project_root: Path,
    mesh_store: LocalMeshStore | None = None,
) -> FastMCP:
    ctx = ServerContext(
        service=service,
        settings=settings,
        project_root=Path(project_root),
        mesh_store=mesh_store or build_mesh_store(settings, Path(project_root)),
    )
    # StrictArgumentsFastMCP, not FastMCP: an argument no tool declares is a
    # mistake the agent must be told about, exactly as the compute API tells
    # an HTTP caller. See strict_tools.py for why the SDK cannot do it.
    mcp = StrictArgumentsFastMCP(
        name="microfluidics-cfd",
        instructions=_INSTRUCTIONS,
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=settings.mcp_streamable_http_path,
    )
    tools_meshes.register(mcp, ctx)
    tools_reactive_cases.register(mcp, ctx)
    tools_stages.register(mcp, ctx)
    tools_runs.register(mcp, ctx)
    return mcp
