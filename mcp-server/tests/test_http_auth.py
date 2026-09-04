from __future__ import annotations

import asyncio
import json

from microfluidics_mcp.http_app import BearerAuthMiddleware


def _call(middleware, headers) -> list[dict]:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": headers,
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


async def _ok_app(scope, receive, send):
    _ = (scope, receive)
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_missing_token_is_rejected() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[])

    assert sent[0]["status"] == 401


def test_wrong_token_is_rejected() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"authorization", b"Bearer nope")])

    assert sent[0]["status"] == 401


def test_valid_token_reaches_the_application() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"authorization", b"Bearer secret")])

    assert sent[0]["status"] == 200


def test_empty_api_key_disables_authentication() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="")

    sent = _call(middleware, headers=[])

    assert sent[0]["status"] == 200


def test_x_api_key_header_is_accepted() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"x-api-key", b"secret")])

    assert sent[0]["status"] == 200


def test_the_scheme_is_matched_case_insensitively() -> None:
    """RFC 7235 makes the auth scheme case-insensitive, so a client sending
    `BEARER` must be authenticated, not silently rejected."""

    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"authorization", b"BEARER secret")])

    assert sent[0]["status"] == 200


def test_a_non_ascii_key_is_rejected_rather_than_raising() -> None:
    """Header bytes are decoded latin-1, so non-ASCII bytes arrive as a
    non-ASCII str; secrets.compare_digest raises TypeError for those, which
    would escape __call__ as an unauthenticated 500 with a traceback. Such a
    key is simply invalid and takes the normal 401 path."""

    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"authorization", b"Bearer \xc3\xa9vil")])

    assert sent[0]["status"] == 401


def test_a_non_ascii_x_api_key_is_rejected_rather_than_raising() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[(b"x-api-key", b"\xc3\xa9vil")])

    assert sent[0]["status"] == 401


def test_a_duplicated_authorization_header_takes_the_first_occurrence() -> None:
    """The compute API's stdlib parser (email.message.Message.get) returns
    the first of a repeated header. The middleware must agree, or the two
    doors can be told apart by sending the header twice."""

    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    accepted = _call(
        middleware,
        headers=[
            (b"authorization", b"Bearer secret"),
            (b"authorization", b"Bearer nope"),
        ],
    )
    rejected = _call(
        middleware,
        headers=[
            (b"authorization", b"Bearer nope"),
            (b"authorization", b"Bearer secret"),
        ],
    )

    assert accepted[0]["status"] == 200
    assert rejected[0]["status"] == 401


def test_the_401_carries_a_well_formed_error_body() -> None:
    middleware = BearerAuthMiddleware(_ok_app, api_key="secret")

    sent = _call(middleware, headers=[])

    start, body_message = sent
    headers = {name.lower(): value for name, value in start["headers"]}
    body = body_message["body"]
    assert headers[b"content-type"] == b"application/json"
    assert headers[b"content-length"] == str(len(body)).encode("ascii")
    assert headers[b"www-authenticate"] == b"Bearer"
    assert json.loads(body.decode("utf-8")) == {
        "contract_version": "v1",
        "code": "unauthorized",
        "message": (
            "Missing or invalid service API key. "
            "Use Authorization: Bearer <key> or X-API-Key."
        ),
        "details": {},
    }
