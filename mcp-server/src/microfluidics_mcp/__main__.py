"""Stdio entrypoint: `python -m microfluidics_mcp`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "backend",
    PROJECT_ROOT / "compute" / "src",
    PROJECT_ROOT / "shared" / "contracts" / "src",
    PROJECT_ROOT / "mcp-server" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def main() -> None:
    # stdout is the MCP transport; every log line must go to stderr.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from backend.app.service import ComputeExecutionService
    from microfluidics_contracts import RuntimeSettings
    from microfluidics_mcp.server import build_mcp_server

    settings = RuntimeSettings.from_env()
    service = ComputeExecutionService(project_root=PROJECT_ROOT, settings=settings)
    build_mcp_server(
        service=service,
        settings=settings,
        project_root=PROJECT_ROOT,
    ).run(transport="stdio")


if __name__ == "__main__":
    main()
