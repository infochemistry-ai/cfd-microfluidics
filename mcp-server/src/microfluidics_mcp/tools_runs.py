"""Observe, page, cancel and locate artifacts of local runs."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .context import ServerContext
from .errors import guard
from .outputs import resolve_stage_outputs

_MAX_ARTIFACT_PAGE = 200
LOCAL_PATH = "local_path"


def _require_response(ctx: ServerContext, request_id: str):
    ctx.require_service_enabled()
    response = ctx.service.get(request_id)
    if response is None:
        raise ToolError(
            f"request_not_found: no run is known for request_id={request_id!r}."
        )
    return response


def _artifact_listing(response) -> list[str]:
    result = getattr(response, "result", None)
    return list(getattr(result, "artifacts", None) or []) if result else []


def _request_parameters(response) -> dict[str, Any]:
    result = getattr(response, "result", None)
    metadata = getattr(result, "metadata", None) or {}
    parameters = metadata.get("request_parameters")
    return parameters if isinstance(parameters, dict) else {}


def _execution(response) -> dict[str, Any]:
    result = getattr(response, "result", None)
    if result is None:
        return {}
    metadata = getattr(result, "metadata", None) or {}
    execution: dict[str, Any] = {
        "adapter": str(getattr(result, "adapter_name", "") or ""),
        "exit_code": getattr(result, "exit_code", None),
        "started_at": getattr(result, "started_at", None),
        "finished_at": getattr(result, "finished_at", None),
        "duration_seconds": getattr(result, "duration_seconds", None),
    }
    for name in ("timed_out", "timeout_seconds"):
        if metadata.get(name) is not None:
            execution[name] = metadata[name]
    return execution


def _experiment_id(response) -> str:
    result = getattr(response, "result", None)
    return str(getattr(result, "experiment_id", "") or "")


def get_run(ctx: ServerContext, *, request_id: str) -> dict[str, Any]:
    response = _require_response(ctx, request_id)
    artifacts = _artifact_listing(response)
    resolved = resolve_stage_outputs(
        experiment_id=_experiment_id(response),
        artifacts=artifacts,
        request_parameters=_request_parameters(response),
    )
    error = getattr(response, "error", None)
    return {
        "request_id": request_id,
        "run_id": getattr(response, "run_id", ""),
        "status": str(getattr(response.status, "value", response.status)),
        "artifact_location": LOCAL_PATH,
        "outputs": resolved.outputs,
        "missing": resolved.missing,
        "ambiguous": resolved.ambiguous,
        "artifact_count": resolved.artifact_count,
        "error": error.to_dict() if error is not None else None,
        "execution": _execution(response),
    }


def list_artifacts(
    ctx: ServerContext,
    *,
    request_id: str,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    response = _require_response(ctx, request_id)
    artifacts = _artifact_listing(response)
    start = max(0, offset)
    page_size = min(limit, _MAX_ARTIFACT_PAGE) if limit > 0 else 0
    page = artifacts[start : start + page_size]
    end = start + len(page)
    return {
        "request_id": request_id,
        "artifact_location": LOCAL_PATH,
        "total": len(artifacts),
        "offset": start,
        "keys": page,
        "next_offset": end if page and end < len(artifacts) else None,
    }


def cancel_run(ctx: ServerContext, *, request_id: str) -> dict[str, Any]:
    ctx.require_service_enabled()
    with guard():
        summary = ctx.service.cancel(request_id)
    if (
        not summary["cancelled"]
        and not summary["already_terminal"]
        and not summary.get("cancellation_requested")
    ):
        raise ToolError(
            f"request_not_found: no local run is known for request_id={request_id!r}."
        )
    return summary


def get_artifact(
    ctx: ServerContext,
    *,
    request_id: str,
    key: str,
) -> dict[str, Any]:
    response = _require_response(ctx, request_id)
    artifacts = _artifact_listing(response)
    if key not in artifacts:
        raise ToolError(
            f"artifact_not_found: {key!r} is not an artifact of "
            f"request_id={request_id!r}."
        )
    return {
        "request_id": request_id,
        "key": key,
        "artifact_location": LOCAL_PATH,
        "local_path": key,
    }


def register(mcp: FastMCP, ctx: ServerContext) -> None:
    @mcp.tool(name="cfd_get_run", description="Poll a local CFD run and obtain next-stage paths.")
    def cfd_get_run(request_id: str) -> dict[str, Any]:
        return get_run(ctx, request_id=request_id)

    @mcp.tool(name="cfd_list_artifacts", description="Page through local artifacts produced by a run.")
    def cfd_list_artifacts(
        request_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        return list_artifacts(ctx, request_id=request_id, offset=offset, limit=limit)

    @mcp.tool(name="cfd_cancel_run", description="Stop a running local solver subprocess.")
    def cfd_cancel_run(request_id: str) -> dict[str, Any]:
        return cancel_run(ctx, request_id=request_id)

    @mcp.tool(name="cfd_get_artifact", description="Return the local path of one run artifact.")
    def cfd_get_artifact(request_id: str, key: str) -> dict[str, Any]:
        return get_artifact(ctx, request_id=request_id, key=key)

    _ = (cfd_get_run, cfd_list_artifacts, cfd_cancel_run, cfd_get_artifact)
