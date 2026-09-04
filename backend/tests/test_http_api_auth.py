"""Authorization handling for the stdlib compute API.

Exercises ComputeRequestHandler._ensure_authorized without binding a socket:
the handler is instantiated directly and given the pieces
BaseHTTPRequestHandler would normally have set up during a real request, so
the 401 it writes can be read straight out of `wfile`.
"""

from __future__ import annotations

import io
from dataclasses import replace
from email.message import Message

import pytest

from backend.app.http_api import ComputeRequestHandler, api_key_matches
from microfluidics_contracts import RuntimeSettings

# Wire bytes for `Authorization: Bearer évil`, as the client would send
# them. The stdlib parses headers as iso-8859-1, so these arrive as the
# non-ASCII str `Bearer Ã©vil`.
_NON_ASCII_BEARER = b"Bearer \xc3\xa9vil".decode("iso-8859-1")


class _StubServer:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self.service = None


def _handler(*, api_key: str, headers: list[tuple[str, str]]) -> ComputeRequestHandler:
    handler = ComputeRequestHandler.__new__(ComputeRequestHandler)
    message = Message()
    for name, value in headers:
        message[name] = value
    handler.headers = message
    handler.server = _StubServer(
        replace(RuntimeSettings.from_env(), service_api_key=api_key)
    )
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /api/v1/compute HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    return handler


def _response(handler: ComputeRequestHandler) -> str:
    return handler.wfile.getvalue().decode("utf-8")


def test_a_non_ascii_bearer_token_yields_401_not_an_exception() -> None:
    """secrets.compare_digest raises TypeError for non-ASCII str, which used
    to escape the handler and render as an unauthenticated 500 with a
    traceback. A key that cannot be the configured key is just invalid."""

    handler = _handler(
        api_key="secret",
        headers=[("Authorization", _NON_ASCII_BEARER)],
    )

    assert handler._ensure_authorized() is False
    body = _response(handler)
    assert " 401 " in body.splitlines()[0]
    assert '"code": "unauthorized"' in body


def test_a_non_ascii_x_api_key_yields_401_not_an_exception() -> None:
    handler = _handler(
        api_key="secret",
        headers=[("X-API-Key", b"\xc3\xa9vil".decode("iso-8859-1"))],
    )

    assert handler._ensure_authorized() is False
    assert '"code": "unauthorized"' in _response(handler)


def test_a_valid_key_is_accepted_and_writes_nothing() -> None:
    handler = _handler(api_key="secret", headers=[("Authorization", "Bearer secret")])

    assert handler._ensure_authorized() is True
    assert _response(handler) == ""


def test_a_duplicated_authorization_header_takes_the_first_occurrence() -> None:
    """Pins the stdlib behaviour the MCP middleware is written to match:
    email.message.Message.get returns the first of a repeated header."""

    accepted = _handler(
        api_key="secret",
        headers=[
            ("Authorization", "Bearer secret"),
            ("Authorization", "Bearer nope"),
        ],
    )
    rejected = _handler(
        api_key="secret",
        headers=[
            ("Authorization", "Bearer nope"),
            ("Authorization", "Bearer secret"),
        ],
    )

    assert accepted._ensure_authorized() is True
    assert rejected._ensure_authorized() is False


@pytest.mark.parametrize(
    ("presented", "configured", "expected"),
    [
        ("secret", "secret", True),
        ("nope", "secret", False),
        (_NON_ASCII_BEARER[7:], "secret", False),
        ("secret", "sécret", False),
        ("", "secret", False),
    ],
)
def test_api_key_matches_never_raises(presented, configured, expected) -> None:
    assert api_key_matches(presented, configured) is expected
