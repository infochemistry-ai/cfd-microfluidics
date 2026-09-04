"""Local mesh discovery and registration."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from .context import ServerContext
from .errors import guard

_MAX_MESH_PAGE = 200


def list_meshes(ctx: ServerContext, *, limit: int = 50) -> dict[str, Any]:
    # A non-positive limit asks for nothing and gets nothing: both mesh stores
    # already return [] for it, and cfd_list_artifacts answers limit=0 with an
    # empty page. Clamping up to 1 here would make the two tools disagree and
    # hand back a mesh the caller never asked for.
    page_size = min(limit, _MAX_MESH_PAGE)
    with guard():
        refs = ctx.mesh_store.list_meshes(page_size)
    return {
        "meshes": [ref.to_dict() for ref in refs],
        "storage": "local",
    }


def register_local_mesh(ctx: ServerContext, *, path: str) -> dict[str, Any]:
    with guard():
        return ctx.mesh_store.register(path).to_dict()


def register(mcp: FastMCP, ctx: ServerContext) -> None:
    @mcp.tool(
        name="cfd_list_meshes",
        description=(
            "List the .msh meshes this runtime can already run. `limit` is "
            "capped at 200; a limit of 0 or less returns no meshes."
        ),
    )
    def cfd_list_meshes(limit: int = 50) -> dict[str, Any]:
        return list_meshes(ctx, limit=limit)

    _ = cfd_list_meshes

    @mcp.tool(
        name="cfd_register_local_mesh",
        description=(
            "Register a .msh file already present in the project directory "
            "and return the repository-relative path cfd_run_import expects."
        ),
    )
    def cfd_register_local_mesh(path: str) -> dict[str, Any]:
        return register_local_mesh(ctx, path=path)

    _ = cfd_register_local_mesh
