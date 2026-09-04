"""Tools that start one CFD stage each.

Parameter policy lives in `backend.app.stage_registry`; these tools only shape
the call and hand the payload to the service for validation."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from microfluidics_contracts import SubmitRunRequestV1

from .context import ServerContext
from .errors import guard

# Mirrors backend.app.stage_registry.Device. Kept as a Literal (rather than
# plain `str`) so the constraint reaches the agent through the tool's JSON
# schema instead of only being enforced later by the registry; a test in
# test_stage_tools.py asserts this stays in lockstep with the registry enum.
_Device = Literal["cpu", "cuda"]
_CPUDevice = Literal["cpu"]
_FlowNumericalProfile = Literal["default", "no_slip_tjunction_validation_v1"]

# What request_id actually guarantees. Two things narrow it, and an agent that
# believes the unqualified version will retry and silently start a duplicate.
# Records live in process memory, are capped and are evicted oldest-first, so
# "reusing it never starts a second run" is a promise this surface cannot keep.
# And only succeeded and cancelled outcomes are replayed at all
# (ComputeExecutionService._is_failed_outcome): a failure is deliberately
# retryable, which is what makes a capacity_exceeded refusal resubmittable.
_REQUEST_ID_NOTE = (
    "request_id is a caller-owned idempotency key. While the run is in "
    "flight, and afterwards for as long as the service still holds a "
    "succeeded or cancelled record of it, reusing it returns that same run "
    "instead of starting a second one; reusing it with different arguments is "
    "rejected. A run that failed is not replayed - reusing its request_id "
    "deliberately starts a fresh run. Records are capped and evicted "
    "oldest-first and are lost on restart, so a request_id reused long after "
    "its run finished may start a fresh run."
)


def _without_unset(parameters: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in parameters.items() if value is not None}


def submit_stage(
    ctx: ServerContext,
    *,
    experiment_id: str,
    parameters: dict[str, Any],
    request_id: str | None,
) -> dict[str, Any]:
    """Validate and start a stage; returns as soon as the run is observable."""

    ctx.require_service_enabled()
    with guard():
        submitted = ctx.service.submit_async(
            SubmitRunRequestV1(
                experiment_id=experiment_id,
                parameters=_without_unset(parameters),
                request_id=request_id,
            )
        )
        record = ctx.service.get(submitted)
    if record is None:
        status = "pending"
    else:
        raw_status = record.status
        status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
    return {
        "request_id": submitted,
        "experiment_id": experiment_id,
        "status": status,
    }


def register(mcp: FastMCP, ctx: ServerContext) -> None:
    @mcp.tool(
        name="cfd_run_import",
        description=(
            "Import a Gmsh .msh mesh into the solver's .npz format. Returns a "
            "request_id immediately; poll cfd_get_run for the outcome. "
            + _REQUEST_ID_NOTE
        ),
    )
    def cfd_run_import(
        mesh_path: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return submit_stage(
            ctx,
            experiment_id="stage_import",
            parameters={"mesh_path": mesh_path},
            request_id=request_id,
        )

    @mcp.tool(
        name="cfd_run_flow",
        description=(
            "Run the flow solver on an imported mesh. mesh_npz_path comes from "
            "a completed cfd_run_import. Returns a request_id immediately and "
            "does not wait for the run to finish; poll cfd_get_run for the "
            "outcome. " + _REQUEST_ID_NOTE
        ),
    )
    def cfd_run_flow(
        mesh_npz_path: str,
        num_steps: int,
        device: _Device | None = None,
        numerical_profile: _FlowNumericalProfile | None = None,
        flow_stop_physical_time: float | None = None,
        snapshot_time_interval: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return submit_stage(
            ctx,
            experiment_id="stage_flow",
            parameters={
                "mesh_npz_path": mesh_npz_path,
                "num_steps": num_steps,
                "device": device,
                "numerical_profile": numerical_profile,
                "flow_stop_physical_time": flow_stop_physical_time,
                "snapshot_time_interval": snapshot_time_interval,
            },
            request_id=request_id,
        )

    @mcp.tool(
        name="cfd_run_transport",
        description=(
            "Run scalar transport on a completed flow field. mesh_npz_path "
            "comes from the cfd_run_import that produced the mesh; "
            "flow_coupling_metadata_path and flow_face_flux_path are "
            "copied verbatim from the outputs of the completed cfd_run_flow "
            "run that produced them (fetch them via cfd_get_run). Returns a "
            "request_id immediately and does not wait for the run to finish; "
            "poll cfd_get_run for the outcome. " + _REQUEST_ID_NOTE
        ),
    )
    def cfd_run_transport(
        mesh_npz_path: str,
        flow_coupling_metadata_path: str,
        flow_face_flux_path: str,
        num_steps: int,
        device: _Device | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return submit_stage(
            ctx,
            experiment_id="stage_transport",
            parameters={
                "mesh_npz_path": mesh_npz_path,
                "flow_coupling_metadata_path": flow_coupling_metadata_path,
                "flow_face_flux_path": flow_face_flux_path,
                "num_steps": num_steps,
                "device": device,
            },
            request_id=request_id,
        )

    @mcp.tool(
        name="cfd_run_thermal",
        description=(
            "Run passive-scalar thermal transport on a completed flow field. "
            "mesh_path is the original .msh used for the cfd_run_import that "
            "produced the mesh; flow_summary_path, "
            "flow_coupling_metadata_path, flow_face_flux_path, "
            "flow_face_to_cells_path, and flow_cell_volumes_path are "
            "copied verbatim from the outputs of the completed cfd_run_flow "
            "run that produced them (fetch them via cfd_get_run). Returns a "
            "request_id immediately and does not wait for the run to finish; "
            "poll cfd_get_run for the outcome. " + _REQUEST_ID_NOTE
        ),
    )
    def cfd_run_thermal(
        mesh_path: str,
        flow_summary_path: str,
        flow_coupling_metadata_path: str,
        flow_face_flux_path: str,
        flow_face_to_cells_path: str,
        flow_cell_volumes_path: str,
        num_steps: int,
        device: _Device | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return submit_stage(
            ctx,
            experiment_id="stage_thermal",
            parameters={
                "mesh_path": mesh_path,
                "flow_summary_path": flow_summary_path,
                "flow_coupling_metadata_path": flow_coupling_metadata_path,
                "flow_face_flux_path": flow_face_flux_path,
                "flow_face_to_cells_path": flow_face_to_cells_path,
                "flow_cell_volumes_path": flow_cell_volumes_path,
                "num_steps": num_steps,
                "device": device,
            },
            request_id=request_id,
        )

    @mcp.tool(
        name="cfd_run_reactive_transport",
        description=(
            "Run reactive transport on a completed flow field. "
            "mesh_npz_path comes from cfd_run_import; the five flow keys "
            "come from the completed cfd_run_flow; reactive_case_path names "
            "the strict reactive-case JSON. Reactive transport v1 is CPU-only. "
            "Returns a request_id immediately and does not wait for the run to "
            "finish; poll cfd_get_run for the outcome. " + _REQUEST_ID_NOTE
        ),
    )
    def cfd_run_reactive_transport(
        mesh_npz_path: str,
        flow_summary_path: str,
        flow_coupling_metadata_path: str,
        flow_face_flux_path: str,
        flow_face_to_cells_path: str,
        flow_cell_volumes_path: str,
        reactive_case_path: str,
        device: _CPUDevice | None = None,
        max_walltime_seconds: float | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return submit_stage(
            ctx,
            experiment_id="stage_reactive_transport",
            parameters={
                "mesh_npz_path": mesh_npz_path,
                "flow_summary_path": flow_summary_path,
                "flow_coupling_metadata_path": flow_coupling_metadata_path,
                "flow_face_flux_path": flow_face_flux_path,
                "flow_face_to_cells_path": flow_face_to_cells_path,
                "flow_cell_volumes_path": flow_cell_volumes_path,
                "reactive_case_path": reactive_case_path,
                "device": device,
                "max_walltime_seconds": max_walltime_seconds,
            },
            request_id=request_id,
        )
