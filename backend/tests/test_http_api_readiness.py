"""`/ready` reports every door this process publishes; `/health` reports one.

An operator marks the service ready from a single probe, and the compute API and
the MCP transport live in one process on two ports. A probe that only asks the
compute port lets a pod whose MCP port failed to bind go Ready anyway, so the
Service publishes a port where every connection fails.

The split matters in both directions:

* readiness must fail when MCP is enabled and not serving, so the pod leaves
  the service instead of advertising a
  dead port;
* liveness must NOT fail for the same reason, or a process supervisor could
  interrupt a synchronous compute request.

The handler is driven directly, as in test_http_api_auth.py, so no socket is
bound and the response can be read straight out of `wfile`.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from email.message import Message

from backend.app.http_api import ComputeRequestHandler
from microfluidics_contracts import RuntimeSettings


class _StubServer:
    def __init__(self, settings: RuntimeSettings, mcp_ready) -> None:
        self.settings = settings
        self.service = None
        self.mcp_ready = mcp_ready


def _get(path: str, *, mcp_enabled: bool, mcp_ready=None) -> tuple[int, dict]:
    handler = ComputeRequestHandler.__new__(ComputeRequestHandler)
    handler.headers = Message()
    handler.server = _StubServer(
        replace(
            RuntimeSettings.from_env(),
            mcp_enabled=mcp_enabled,
            mcp_host="0.0.0.0",
            mcp_port=8092,
        ),
        mcp_ready,
    )
    handler.path = path
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)

    handler.do_GET()

    raw = handler.wfile.getvalue().decode("utf-8")
    head, _, body = raw.partition("\r\n\r\n")
    status = int(head.splitlines()[0].split(" ")[1])
    return status, json.loads(body)


def test_ready_is_ok_when_mcp_is_disabled() -> None:
    status, payload = _get("/ready", mcp_enabled=False)

    assert status == 200
    assert payload["status"] == "ready"
    # Nothing is claimed about a transport this runtime does not run.
    assert "mcp_transport" not in payload["checks"]


def test_ready_is_ok_when_mcp_is_enabled_and_serving() -> None:
    status, payload = _get("/ready", mcp_enabled=True, mcp_ready=lambda: True)

    assert status == 200
    assert payload["status"] == "ready"
    assert payload["checks"]["mcp_transport"] == "ok"


def test_ready_fails_when_the_enabled_mcp_transport_stopped() -> None:
    status, payload = _get("/ready", mcp_enabled=True, mcp_ready=lambda: False)

    assert status == 503
    assert payload["status"] == "not_ready"
    assert payload["checks"]["mcp_transport"] == "down"
    # The operator has to be told which port to look at and what to grep for.
    assert "0.0.0.0:8092" in payload["message"]


def test_ready_fails_when_mcp_is_enabled_but_never_started() -> None:
    """`build_mcp_server` raising leaves no transport at all. That is the same
    outage as a transport that stopped, and must not read as ready."""

    status, payload = _get("/ready", mcp_enabled=True, mcp_ready=None)

    assert status == 503
    assert payload["checks"]["mcp_transport"] == "down"


def test_ready_fails_rather_than_500s_when_the_probe_raises() -> None:
    def _explode() -> bool:
        raise RuntimeError("probe blew up")

    status, payload = _get("/ready", mcp_enabled=True, mcp_ready=_explode)

    assert status == 503
    assert payload["checks"]["mcp_transport"] == "down"


def test_health_stays_up_while_the_mcp_transport_is_down() -> None:
    """Liveness is deliberately blind to MCP so active work is not interrupted
    when the separate readiness signal already reports the missing endpoint."""

    status, payload = _get("/health", mcp_enabled=True, mcp_ready=lambda: False)

    assert status == 200
    assert payload == {"status": "ok"}
