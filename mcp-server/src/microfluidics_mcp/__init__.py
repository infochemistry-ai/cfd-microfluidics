"""MCP surface over the stateless CFD compute service."""

from __future__ import annotations

from .context import ServerContext
from .server import build_mcp_server

__all__ = ["ServerContext", "build_mcp_server", "__version__"]

__version__ = "0.1.0"
