"""`McpTransport.is_serving` is what the compute API's /ready probe reports.

The transport runs in a daemon thread, so nothing about the process tells
an operator whether it came up. Before this, a bind failure was logged and the
pod went Ready anyway, publishing an MCP port where every connection failed.
These tests pin the two answers the probe has to get right against a real
uvicorn server rather than a mock of one: a bound transport, and a port
somebody else already holds.
"""

from __future__ import annotations

import socket
from time import monotonic, sleep

import pytest
from mcp.server.fastmcp import FastMCP

from microfluidics_contracts import RuntimeSettings
from microfluidics_mcp.http_app import McpTransport, start_http_server_thread

_STARTUP_TIMEOUT_SECONDS = 20.0


def _wait_for(predicate, *, timeout: float = _STARTUP_TIMEOUT_SECONDS) -> bool:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return True
        sleep(0.02)
    return False


def _settings(port: int) -> RuntimeSettings:
    return RuntimeSettings(
        service_api_key="",
        mcp_host="127.0.0.1",
        mcp_port=port,
    )


def _stop(transport: McpTransport) -> None:
    transport.server.should_exit = True
    transport.thread.join(timeout=_STARTUP_TIMEOUT_SECONDS)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture()
def occupied_port():
    """A port held by a live listener, so uvicorn's bind cannot succeed."""

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    try:
        yield holder.getsockname()[1]
    finally:
        holder.close()


def test_a_bound_transport_reports_itself_as_serving() -> None:
    transport = None
    try:
        transport = start_http_server_thread(FastMCP("test"), _settings(_free_port()))
        assert _wait_for(transport.is_serving), "transport never reported serving"
    finally:
        if transport is not None:
            _stop(transport)

    # A transport that has stopped is not serving either: the thread is gone,
    # even though uvicorn leaves `started` True after a clean shutdown.
    assert transport.is_serving() is False


def test_a_transport_that_cannot_bind_never_reports_itself_as_serving(
    occupied_port,
) -> None:
    """The failure the readiness probe exists for. uvicorn raises SystemExit
    out of its own bind, which `start_http_server_thread` catches so the
    compute API survives - so nothing but this flag can tell an operator."""

    transport = start_http_server_thread(FastMCP("test"), _settings(occupied_port))
    transport.thread.join(timeout=_STARTUP_TIMEOUT_SECONDS)

    assert transport.thread.is_alive() is False
    assert transport.is_serving() is False
