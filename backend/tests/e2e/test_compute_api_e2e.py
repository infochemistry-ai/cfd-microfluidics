from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
E2E_EXPERIMENT_ID = "stage_import"
E2E_PARAMETERS = {"mesh_path": "data/meshes/gmsh/t_junction.msh"}


@dataclass
class RunningServer:
    process: subprocess.Popen[str]
    base_url: str
    log_path: Path
    log_file: TextIO


def _read_json(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _http_json(
    base_url: str,
    *,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    data = None
    request_headers: dict[str, str] = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(
        f"{base_url}{path}",
        method=method,
        data=data,
        headers=request_headers,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            return int(response.status), _read_json(body)
    except HTTPError as exc:
        return int(exc.code), _read_json(exc.read())


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_log(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


def _wait_for_health(server: RunningServer, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            log_text = _server_log(server.log_path)
            raise AssertionError(
                "Compute service exited before /health became ready.\n"
                f"Log output:\n{log_text}"
            )
        try:
            status_code, payload = _http_json(
                server.base_url,
                method="GET",
                path="/health",
                timeout_seconds=1.0,
            )
        except URLError:
            status_code, payload = 0, {}
        if status_code == 200 and payload.get("status") == "ok":
            return
        time.sleep(0.2)

    log_text = _server_log(server.log_path)
    raise AssertionError(
        f"/health did not become ready within {timeout_seconds:.1f}s.\n"
        f"Last log output:\n{log_text}"
    )


def _start_server(
    *,
    service_enabled: bool,
    logs_dir: Path,
    api_key: str = "",
    max_request_bytes: int | None = None,
) -> RunningServer:
    port = _get_free_port()
    env = os.environ.copy()
    env.update(
        {
            "SERVICE_ENABLED": "true" if service_enabled else "false",
            "SERVICE_HOST": "127.0.0.1",
            "SERVICE_PORT": str(port),
            "SERVICE_API_KEY": api_key,
        }
    )
    if max_request_bytes is not None:
        env["SERVICE_MAX_REQUEST_BYTES"] = str(max_request_bytes)

    log_path = logs_dir / f"compute_{port}.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "backend/app/app.py"],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    server = RunningServer(
        process=process,
        base_url=f"http://127.0.0.1:{port}",
        log_path=log_path,
        log_file=log_file,
    )

    try:
        _wait_for_health(server)
    except Exception:
        _stop_server(server)
        raise
    return server


def _stop_server(server: RunningServer) -> None:
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=5)
    server.log_file.close()


@pytest.fixture
def enabled_server(tmp_path_factory: pytest.TempPathFactory) -> RunningServer:
    logs_dir = tmp_path_factory.mktemp("compute-e2e-logs")
    server = _start_server(service_enabled=True, logs_dir=logs_dir)
    try:
        yield server
    finally:
        _stop_server(server)


@pytest.fixture
def disabled_server(tmp_path_factory: pytest.TempPathFactory) -> RunningServer:
    logs_dir = tmp_path_factory.mktemp("compute-e2e-logs-disabled")
    server = _start_server(service_enabled=False, logs_dir=logs_dir)
    try:
        yield server
    finally:
        _stop_server(server)


def test_health_endpoint_returns_ok(enabled_server: RunningServer) -> None:
    status_code, payload = _http_json(
        enabled_server.base_url,
        method="GET",
        path="/health",
    )
    assert status_code == 200
    assert payload == {"status": "ok"}


def test_compute_returns_503_when_service_disabled(
    disabled_server: RunningServer,
) -> None:
    status_code, payload = _http_json(
        disabled_server.base_url,
        method="POST",
        path="/api/v1/compute",
        payload={"experiment_id": E2E_EXPERIMENT_ID},
    )
    assert status_code == 503
    assert payload.get("code") == "service_disabled"


def test_compute_returns_400_for_invalid_json(enabled_server: RunningServer) -> None:
    request = Request(
        f"{enabled_server.base_url}/api/v1/compute",
        method="POST",
        data=b"{",
        headers={"Content-Type": "application/json"},
    )
    try:
        urlopen(request, timeout=5.0)
    except HTTPError as exc:
        assert int(exc.code) == 400
        payload = _read_json(exc.read())
        assert payload.get("code") == "invalid_json"
    else:
        raise AssertionError("Expected invalid JSON request to fail with 400.")


def test_compute_returns_400_for_unknown_experiment(
    enabled_server: RunningServer,
) -> None:
    status_code, payload = _http_json(
        enabled_server.base_url,
        method="POST",
        path="/api/v1/compute",
        payload={
            "contract_version": "v1",
            "experiment_id": "unknown_experiment",
            "parameters": {},
        },
    )
    assert status_code == 400
    assert payload.get("code") == "unknown_experiment"


def test_compute_rejects_unknown_submit_field(enabled_server: RunningServer) -> None:
    status_code, payload = _http_json(
        enabled_server.base_url,
        method="POST",
        path="/api/v1/compute",
        payload={
            "contract_version": "v1",
            "experiment_id": E2E_EXPERIMENT_ID,
            "parameters": {},
            "unsupported_option": True,
        },
    )
    assert status_code == 400
    assert payload.get("code") == "invalid_contract"


def test_compute_returns_413_for_oversized_request(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    logs_dir = tmp_path_factory.mktemp("compute-e2e-logs-max")
    server = _start_server(
        service_enabled=True,
        logs_dir=logs_dir,
        max_request_bytes=64,
    )
    try:
        status_code, payload = _http_json(
            server.base_url,
            method="POST",
            path="/api/v1/compute",
            payload={
                "contract_version": "v1",
                "experiment_id": E2E_EXPERIMENT_ID,
                "parameters": {},
                "request_id": "x" * 200,
            },
        )
        assert status_code == 413
        assert payload.get("code") == "request_too_large"
    finally:
        _stop_server(server)


def test_compute_requires_api_key_when_configured(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    logs_dir = tmp_path_factory.mktemp("compute-e2e-logs-auth")
    server = _start_server(
        service_enabled=True,
        logs_dir=logs_dir,
        api_key="secret",
    )
    try:
        status_code, payload = _http_json(
            server.base_url,
            method="POST",
            path="/api/v1/compute",
            payload={
                "contract_version": "v1",
                "experiment_id": E2E_EXPERIMENT_ID,
                "parameters": {},
            },
        )
        assert status_code == 401
        assert payload.get("code") == "unauthorized"
    finally:
        _stop_server(server)


def test_compute_executes_import_run_and_returns_terminal_payload(
    enabled_server: RunningServer,
) -> None:
    status_code, payload = _http_json(
        enabled_server.base_url,
        method="POST",
        path="/api/v1/compute",
        payload={
            "contract_version": "v1",
            "request_id": "local-job-123",
            "experiment_id": E2E_EXPERIMENT_ID,
            "parameters": E2E_PARAMETERS,
        },
        timeout_seconds=120.0,
    )
    assert status_code == 200
    assert payload["request_id"] == "local-job-123"
    assert isinstance(payload.get("run_id"), str) and payload["run_id"]
    assert payload["status"] == "succeeded"
    assert payload["error"] is None
    result = payload["result"]
    assert result is not None
    assert result["request_id"] == "local-job-123"
    assert result["experiment_id"] == E2E_EXPERIMENT_ID
    assert result["exit_code"] == 0


def test_cancel_endpoint_reports_unknown_local_request(
    enabled_server: RunningServer,
) -> None:
    status_code, payload = _http_json(
        enabled_server.base_url,
        method="POST",
        path="/api/v1/compute/some-request-id/cancel",
    )
    assert status_code == 404
    assert payload.get("code") == "request_not_found"
