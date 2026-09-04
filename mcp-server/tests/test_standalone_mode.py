"""Acceptance tests for the standalone MCP surface.

The tests call the tools exposed by ``FastMCP`` and assemble the server from
environment-derived settings. The numerical runner is replaced with a small
fixture so these tests stay focused on the MCP contract.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Iterator

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from backend.app.service import ComputeExecutionService
from microfluidics_contracts import ResultPayloadV1, RuntimeSettings, SubmitRunRequestV1
from microfluidics_mcp.server import build_mcp_server
from microfluidics_mcp.storage import LocalMeshStore, build_mesh_store

_POLL_TIMEOUT_SECONDS = 30.0
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})

MESH_KEY = "data/meshes/pipe.msh"

# What each stubbed stage leaves on disk, under the names the real stages use,
# so `resolve_stage_outputs` maps them onto the next stage's parameters exactly
# as it would for a real run. `input_imported_mesh.npz` is
# `run_import_gmsh_mesh.py:287`'s `f"{msh_path.stem}_imported_mesh.npz"` for
# the staged `/job/work/input.msh`; the registry's `mesh.npz` is only what the
# flow stage mounts that file as.
_STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "stage_import": ("input_imported_mesh.npz",),
    "stage_flow": (
        "summary.json",
        "flow_coupling_metadata.json",
        "final_corrected_face_flux.npy",
        "face_to_cells.npy",
        "cell_volumes.npy",
        "logs/run.log",
    ),
}


class _LocalStageRunner:
    """Small deterministic runner used by the MCP contract tests.

    The service still validates, schedules and records the run. Artifacts are
    written under the project root and reported as repository-relative POSIX
    paths in ``ResultPayloadV1.artifacts``."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.requests: list[SubmitRunRequestV1] = []

    def run(
        self,
        run_id: str,
        request: SubmitRunRequestV1,
        cancel_event: Any,
        run_work_dir: Path,
    ) -> ResultPayloadV1:
        _ = cancel_event
        self.requests.append(request)
        artifacts: list[str] = []
        for name in _STAGE_ARTIFACTS[request.experiment_id]:
            path = Path(run_work_dir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stub artifact")
            artifacts.append(path.resolve().relative_to(self.project_root).as_posix())
        return ResultPayloadV1(
            run_id=run_id,
            adapter_name="local-stage-adapter",
            experiment_id=request.experiment_id,
            exit_code=0,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:10+00:00",
            duration_seconds=10.0,
            artifacts=artifacts,
            log_path=None,
            metadata={"request_parameters": dict(request.parameters)},
        )


@dataclass(frozen=True)
class _StandaloneRuntime:
    """The standalone runtime as an agent reaches it: tools and nothing else."""

    mcp: FastMCP
    service: ComputeExecutionService
    runner: _LocalStageRunner
    project_root: Path

    def call(self, name: str, **arguments: Any) -> Any:
        _content, payload = asyncio.run(self.mcp.call_tool(name, arguments))
        return payload

    def tool_names(self) -> set[str]:
        return {tool.name for tool in asyncio.run(self.mcp.list_tools())}

    def poll(self, request_id: str) -> dict[str, Any]:
        """Poll cfd_get_run to a terminal state, as the instructions tell an
        agent to. Nothing here reads the service directly."""

        deadline = monotonic() + _POLL_TIMEOUT_SECONDS
        while True:
            payload = self.call("cfd_get_run", request_id=request_id)
            if payload["status"] in _TERMINAL:
                return payload
            assert payload["status"] in {"pending", "running"}, payload["status"]
            assert monotonic() < deadline, (
                f"run {request_id} never reached a terminal state"
            )
            sleep(0.01)

    def run_stage(self, tool: str, **arguments: Any) -> dict[str, Any]:
        submitted = self.call(tool, **arguments)
        assert submitted["status"] not in {"failed", "cancelled"}, submitted
        return self.poll(submitted["request_id"])


@pytest.fixture()
def standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_StandaloneRuntime]:
    # The run work directory has to stay under this test's project root: an
    # exported SERVICE_RUN_ROOT would put run artifacts outside tmp_path,
    # where they can no longer be expressed as repository-relative paths.
    monkeypatch.delenv("SERVICE_RUN_ROOT", raising=False)
    # The operator's kill switch: the run tools refuse submit, status and
    # cancel while it is false, so a standalone runtime must turn it on.
    monkeypatch.setenv("SERVICE_ENABLED", "true")

    project_root = tmp_path.resolve()
    (project_root / "data" / "meshes").mkdir(parents=True)
    (project_root / MESH_KEY).write_bytes(b"$MeshFormat\n")

    settings = RuntimeSettings.from_env()
    runner = _LocalStageRunner(project_root)
    service = ComputeExecutionService(
        project_root=project_root,
        settings=settings,
        runner=runner,
    )
    yield _StandaloneRuntime(
        mcp=build_mcp_server(
            service=service,
            settings=settings,
            project_root=project_root,
        ),
        service=service,
        runner=runner,
        project_root=project_root,
    )


def test_standalone_server_exposes_only_tools_that_work_here(standalone) -> None:
    """The tool list contains the complete supported standalone surface."""

    names = standalone.tool_names()

    assert names == {
        "cfd_list_meshes",
        "cfd_register_local_mesh",
        "cfd_validate_reactive_case",
        "cfd_run_import",
        "cfd_run_flow",
        "cfd_run_transport",
        "cfd_run_thermal",
        "cfd_run_reactive_transport",
        "cfd_get_run",
        "cfd_list_artifacts",
        "cfd_get_artifact",
        "cfd_cancel_run",
    }


def test_standalone_server_builds_local_storage_from_the_environment(
    standalone,
) -> None:
    """`build_mcp_server` was handed no mesh store: the fallback inside
    `build_mesh_store` is what has to notice there is local storage, and
    the agent has to be told which world it is in."""

    store = build_mesh_store(standalone.service.settings, standalone.project_root)
    listed = standalone.call("cfd_list_meshes")

    assert isinstance(store, LocalMeshStore)
    assert store.mesh_root == standalone.project_root / "data" / "meshes"
    assert listed["storage"] == "local"


def test_standalone_agent_discovers_and_registers_a_mesh_from_disk(
    standalone,
) -> None:
    listed = standalone.call("cfd_list_meshes", limit=10)
    registered = standalone.call("cfd_register_local_mesh", path=MESH_KEY)

    assert [item["key"] for item in listed["meshes"]] == [MESH_KEY]
    assert listed["meshes"][0]["size"] == len(b"$MeshFormat\n")
    assert registered["key"] == MESH_KEY
    assert (standalone.project_root / registered["key"]).is_file()


def test_standalone_mesh_registration_stays_inside_the_project(standalone) -> None:
    """Local mesh ingress is a project-root-scoped door, not a file reader for
    the whole machine."""

    outside = standalone.project_root.parent / "outside_the_project.msh"
    outside.write_bytes(b"$MeshFormat\n")

    with pytest.raises(ToolError):
        standalone.call(
            "cfd_register_local_mesh",
            path=f"../{outside.name}",
        )


def test_standalone_run_reaches_a_terminal_state_through_the_tool_layer(
    standalone,
) -> None:
    """The whole point: a mesh on disk, started through the tools an agent
    sees, polled through the tool an agent is told to poll."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]

    submitted = standalone.call(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-1",
    )
    final = standalone.poll(submitted["request_id"])

    assert submitted["request_id"] == "standalone-import-1"
    assert submitted["experiment_id"] == "stage_import"
    assert final["status"] == "succeeded"
    assert final["error"] is None
    # The key the mesh tool handed the agent is the key the service ran, with
    # no rewriting in between.
    assert [
        request.parameters["mesh_path"] for request in standalone.runner.requests
    ] == [mesh_key]


def test_standalone_outputs_resolve_to_files_on_disk(standalone) -> None:
    """``outputs`` contains repository-relative files a next stage can use."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]

    final = standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-2",
    )

    assert final["status"] == "succeeded"
    assert final["artifact_location"] == "local_path"
    assert final["artifact_count"] == 1

    # The local filesystem is what produced these values.
    stored = standalone.service.get("standalone-import-2")
    assert stored.result.artifacts == [final["outputs"]["mesh_npz_path"]]

    mesh_npz_key = final["outputs"]["mesh_npz_path"]
    assert mesh_npz_key.endswith("/input_imported_mesh.npz")
    assert (standalone.project_root / mesh_npz_key).is_file()
    # The mesh the agent supplied is echoed back so it can be carried to a
    # stage that still needs it (stage_thermal takes the original .msh).
    assert final["outputs"]["mesh_path"] == mesh_key
    assert final["ambiguous"] == []


def test_standalone_outputs_feed_the_next_stage_verbatim(standalone) -> None:
    """The contract an agent is told to follow: copy `outputs` into the next
    stage's arguments. That has to be true of local paths too, or the
    standalone runtime stops after one stage."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]
    imported = standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-3",
    )

    flow = standalone.run_stage(
        "cfd_run_flow",
        mesh_npz_path=imported["outputs"]["mesh_npz_path"],
        num_steps=1,
        request_id="standalone-flow-1",
    )

    assert flow["status"] == "succeeded"
    assert flow["error"] is None
    assert flow["artifact_location"] == "local_path"
    outputs = flow["outputs"]
    for parameter, filename in (
        ("flow_summary_path", "summary.json"),
        ("flow_coupling_metadata_path", "flow_coupling_metadata.json"),
        ("flow_face_flux_path", "final_corrected_face_flux.npy"),
        ("flow_face_to_cells_path", "face_to_cells.npy"),
        ("flow_cell_volumes_path", "cell_volumes.npy"),
    ):
        value = outputs[parameter]
        assert value.endswith(f"/{filename}")
        assert (standalone.project_root / value).is_file()
    # The .npz the agent passed in is echoed for the stage that needs it next.
    assert outputs["mesh_npz_path"] == imported["outputs"]["mesh_npz_path"]
    # `outputs` is scoped to what a flow run can supply. stage_thermal's
    # original .msh is neither produced nor consumed here, so it is absent
    # rather than reported as something this run failed to deliver; the agent
    # carries it from the import run, whose `outputs` echoed it.
    assert flow["missing"] == []
    assert "mesh_path" not in outputs
    assert imported["outputs"]["mesh_path"] == mesh_key


def test_standalone_artifacts_are_reachable_from_a_completed_run(
    standalone,
) -> None:
    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]
    imported = standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-4",
    )
    flow = standalone.run_stage(
        "cfd_run_flow",
        mesh_npz_path=imported["outputs"]["mesh_npz_path"],
        num_steps=1,
        request_id="standalone-flow-2",
    )

    page = standalone.call(
        "cfd_list_artifacts",
        request_id="standalone-flow-2",
        offset=0,
        limit=50,
    )
    summary_key = flow["outputs"]["flow_summary_path"]
    fetched = standalone.call(
        "cfd_get_artifact",
        request_id="standalone-flow-2",
        key=summary_key,
    )

    assert page["artifact_location"] == "local_path"
    assert page["total"] == len(_STAGE_ARTIFACTS["stage_flow"])
    assert page["next_offset"] is None
    assert summary_key in page["keys"]
    # Every listed key is a path the operator can open, including the log the
    # stage wrote, which is not an input to any stage and so never appears in
    # `outputs`.
    for key in page["keys"]:
        assert (standalone.project_root / key).is_file()
    assert any(key.endswith("/logs/run.log") for key in page["keys"])
    assert fetched == {
        "request_id": "standalone-flow-2",
        "key": summary_key,
        "artifact_location": "local_path",
        "local_path": summary_key,
    }
    assert (standalone.project_root / fetched["local_path"]).read_bytes() == (
        b"stub artifact"
    )


def test_standalone_get_artifact_refuses_a_key_the_run_never_produced(
    standalone,
) -> None:
    """Local paths must not turn cfd_get_artifact into a file reader for the
    whole project root."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]
    standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-5",
    )

    with pytest.raises(ToolError, match="artifact_not_found"):
        standalone.call(
            "cfd_get_artifact",
            request_id="standalone-import-5",
            key=MESH_KEY,
        )


def test_standalone_cancel_reports_completed_run_as_terminal(standalone) -> None:
    """A cancellation request must accurately report an already-finished run."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]
    standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="standalone-import-6",
    )

    summary = standalone.call(
        "cfd_cancel_run", request_id="standalone-import-6"
    )
    assert summary["cancelled"] == []
    assert summary["already_terminal"] == ["standalone-import-6"]


def test_standalone_unknown_request_is_reported_not_invented(standalone) -> None:
    with pytest.raises(ToolError, match="request_not_found"):
        standalone.call("cfd_get_run", request_id="never-submitted")
