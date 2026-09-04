"""Entrypoint for the stateless compute integration API."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
SHARED_CONTRACTS_SRC = PROJECT_ROOT / "shared" / "contracts" / "src"
MCP_SRC = PROJECT_ROOT / "mcp-server" / "src"

for path in (PROJECT_ROOT, BACKEND_ROOT, COMPUTE_SRC, SHARED_CONTRACTS_SRC, MCP_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_compute_service(settings):
    from backend.app.service import ComputeExecutionService

    return ComputeExecutionService(
        project_root=PROJECT_ROOT,
        settings=settings,
    )


def main() -> None:
    from microfluidics_contracts import RuntimeSettings
    from backend.app.http_api import create_http_server

    settings = RuntimeSettings.from_env()
    service = build_compute_service(settings)

    logger.info(
        "Starting local compute API at http://%s:%s "
        "(SERVICE_ENABLED=%s, RUN_ROOT=%s)",
        settings.service_host,
        settings.service_port,
        settings.service_enabled,
        settings.service_run_root,
    )

    # A failed MCP transport must not take the compute API down mid-run, so it
    # is still started best-effort - but it must not be published as healthy
    # either, so `/ready` reports it and the readiness probe fails instead.
    mcp_ready = None
    if settings.mcp_enabled:
        try:
            from microfluidics_mcp.http_app import start_http_server_thread
            from microfluidics_mcp.server import build_mcp_server

            transport = start_http_server_thread(
                build_mcp_server(
                    service=service,
                    settings=settings,
                    project_root=PROJECT_ROOT,
                ),
                settings,
            )
            mcp_ready = transport.is_serving
            logger.info(
                "Starting MCP streamable HTTP at http://%s:%s%s",
                settings.mcp_host,
                settings.mcp_port,
                settings.mcp_streamable_http_path,
            )
        except Exception:  # noqa: BLE001 - compute API must still start
            logger.exception("Failed to start MCP HTTP transport")

    server = create_http_server(settings, service, mcp_ready=mcp_ready)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down compute API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
