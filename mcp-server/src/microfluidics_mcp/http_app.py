"""Streamable HTTP transport, guarded by the same key as the compute API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.app.http_api import api_key_matches

from microfluidics_contracts import RuntimeSettings

logger = logging.getLogger(__name__)

_UNAUTHORIZED_BODY = json.dumps(
    {
        "contract_version": "v1",
        "code": "unauthorized",
        "message": (
            "Missing or invalid service API key. "
            "Use Authorization: Bearer <key> or X-API-Key."
        ),
        "details": {},
    }
).encode("utf-8")


class BearerAuthMiddleware:
    """Rejects unauthenticated calls before they reach the MCP session layer."""

    def __init__(self, app: Any, api_key: str) -> None:
        self.app = app
        self.api_key = str(api_key).strip()

    def _header(self, scope: dict[str, Any], name: bytes) -> str:
        """First occurrence wins, as in the compute API.

        The stdlib header parser backing `backend/app/http_api.py` returns
        the first of a repeated header (`email.message.Message.get`), so a
        dict comprehension over the raw list - which keeps the last - would
        let a duplicated Authorization header tell the two doors apart.
        """

        for raw_name, raw_value in scope.get("headers", []):
            if raw_name.lower() == name:
                return raw_value.decode("latin-1").strip()
        return ""

    def _presented_key(self, scope: dict[str, Any]) -> str | None:
        authorization = self._header(scope, b"authorization")
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            if token:
                return token
        return self._header(scope, b"x-api-key") or None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self.api_key:
            await self.app(scope, receive, send)
            return
        presented = self._presented_key(scope)
        if presented and api_key_matches(presented, self.api_key):
            await self.app(scope, receive, send)
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode("ascii")),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def build_http_app(mcp: FastMCP, settings: RuntimeSettings) -> Any:
    return BearerAuthMiddleware(
        mcp.streamable_http_app(),
        api_key=settings.service_api_key,
    )


@dataclass(frozen=True)
class McpTransport:
    """Handle on the background transport, for the compute API's /ready probe.

    `uvicorn.Server.started` is set only after the socket is bound and the
    protocol factory is installed, and a bind failure raises SystemExit out of
    `serve()` before that, ending the thread. The conjunction of the two is
    therefore true exactly while the transport is answering, and false for
    both failure shapes: never bound, and bound then stopped.
    """

    thread: threading.Thread
    server: Any

    def is_serving(self) -> bool:
        return self.thread.is_alive() and bool(getattr(self.server, "started", False))


def start_http_server_thread(
    mcp: FastMCP,
    settings: RuntimeSettings,
) -> McpTransport:
    """Serve MCP beside the compute API; uvicorn skips signal setup off-main."""

    import uvicorn

    config = uvicorn.Config(
        build_http_app(mcp, settings),
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        try:
            asyncio.run(server.serve())
        except BaseException:  # noqa: BLE001 - the compute API must survive this
            # uvicorn.Server.startup raises SystemExit when the port cannot
            # be bound, and `except Exception` does not catch it: threading's
            # default excepthook then discards SystemExit silently, so this
            # log line never fired and uvicorn's own message was the only
            # trace. Catching BaseException keeps the compute API alive
            # either way and leaves one explicit record of why MCP is gone.
            logger.exception("MCP HTTP transport stopped")

    thread = threading.Thread(target=_serve, name="mcp-http", daemon=True)
    thread.start()
    return McpTransport(thread=thread, server=server)
