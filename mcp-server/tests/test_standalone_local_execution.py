"""The standalone runtime executing stages for real, end to end.

test_standalone_mode.py stubs the runner to accept the MCP-facing contract.
This file removes that stub: the service builds its own ComputeRunner, which
in a checkout hands CFD stages to LocalStageAdapter, which runs
`experiments/gmsh/` entrypoints as subprocesses. Only the entrypoints
themselves are stubs - there is no arm64 wheel for torch==2.5.1+cpu, so the
solvers cannot run on this machine - and they are stubs *on disk*, in the
project root the runtime was pointed at, so the adapter resolves, stages,
executes and collects exactly as it would for the real ones.

What this pins down that a stubbed runner cannot: the artifacts a local stage
leaves behind are found, reported as repository-relative paths, and mapped
onto the next stage's parameters by name - i.e. the chain an agent is told to
follow actually chains.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Iterator

import pytest
from mcp.server.fastmcp import FastMCP

from backend.app.adapters.local_stage_adapter import LocalStageAdapter
from backend.app.service import ComputeExecutionService
from microfluidics_contracts import RuntimeSettings
from microfluidics_mcp.server import build_mcp_server

_CLEARED_ENV = (
    # Run artifacts have to land under this test's project root.
    "SERVICE_RUN_ROOT",
)

_POLL_TIMEOUT_SECONDS = 60.0
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})

MESH_KEY = "data/meshes/pipe.msh"

_STUB_PREAMBLE = """
import sys
from pathlib import Path

argv = sys.argv[1:]
output_root = Path(argv[argv.index("--output-root") + 1])
run_dir = output_root / "20260101_000000_stub"
run_dir.mkdir(parents=True, exist_ok=True)
"""

# Each stub writes the files the real stage writes, under the real names, from
# the input it was actually handed - so an empty or unstaged input fails here.
#
# The import stub derives its archive name the way run_import_gmsh_mesh.py:287
# does, `f"{msh_path.stem}_imported_mesh.npz"`, rather than naming a file: the
# registry's `mesh.npz` is what the *flow* stage mounts this file as, and a
# stub that wrote it would be asserting a filename the real script cannot
# produce.
_STUB_SCRIPTS = {
    "run_import_gmsh_mesh.py": _STUB_PREAMBLE
    + """
mesh = Path(argv[argv.index("--msh") + 1])
assert mesh.read_bytes() == b"$MeshFormat\\n", mesh
(run_dir / f"{mesh.stem}_imported_mesh.npz").write_bytes(b"stub npz")
(run_dir / "summary.json").write_text("{}", encoding="utf-8")
""",
    "run_gmsh_tetra_flow_debug.py": _STUB_PREAMBLE
    + """
mesh_npz = Path(argv[argv.index("--mesh-npz") + 1])
assert mesh_npz.read_bytes() == b"stub npz", mesh_npz
for name in (
    "summary.json",
    "flow_coupling_metadata.json",
    "final_corrected_face_flux.npy",
    "face_to_cells.npy",
    "cell_volumes.npy",
):
    (run_dir / name).write_text("{}", encoding="utf-8")
""",
}


@dataclass(frozen=True)
class _StandaloneRuntime:
    mcp: FastMCP
    service: ComputeExecutionService
    project_root: Path

    def call(self, name: str, **arguments: Any) -> Any:
        _content, payload = asyncio.run(self.mcp.call_tool(name, arguments))
        return payload

    def poll(self, request_id: str) -> dict[str, Any]:
        deadline = monotonic() + _POLL_TIMEOUT_SECONDS
        while True:
            payload = self.call("cfd_get_run", request_id=request_id)
            if payload["status"] in _TERMINAL:
                return payload
            assert monotonic() < deadline, (
                f"run {request_id} never reached a terminal state"
            )
            sleep(0.05)

    def run_stage(self, tool: str, **arguments: Any) -> dict[str, Any]:
        submitted = self.call(tool, **arguments)
        return self.poll(submitted["request_id"])


@pytest.fixture()
def standalone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_StandaloneRuntime]:
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERVICE_ENABLED", "true")

    project_root = tmp_path.resolve()
    (project_root / "data" / "meshes").mkdir(parents=True)
    (project_root / MESH_KEY).write_bytes(b"$MeshFormat\n")
    gmsh_root = project_root / "experiments" / "gmsh"
    gmsh_root.mkdir(parents=True)
    for filename, source in _STUB_SCRIPTS.items():
        (gmsh_root / filename).write_text(source, encoding="utf-8")

    settings = RuntimeSettings.from_env()

    # No runner is injected: the service builds the one this runtime would
    # really use.
    service = ComputeExecutionService(project_root=project_root, settings=settings)
    assert isinstance(service.runner.adapter, LocalStageAdapter)

    yield _StandaloneRuntime(
        mcp=build_mcp_server(
            service=service,
            settings=settings,
            project_root=project_root,
        ),
        service=service,
        project_root=project_root,
    )


def test_standalone_import_runs_the_entrypoint_and_reports_its_files(
    standalone,
) -> None:
    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]

    final = standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="local-import-1",
    )

    assert final["status"] == "succeeded", final
    assert final["artifact_location"] == "local_path"
    mesh_npz_key = final["outputs"]["mesh_npz_path"]
    # The registry stages this file as `mesh.npz` for the flow stage; the
    # importer writes it under the stem of the `.msh` it was handed.
    assert mesh_npz_key.endswith("/input_imported_mesh.npz")
    assert not mesh_npz_key.startswith("/")
    assert (standalone.project_root / mesh_npz_key).read_bytes() == b"stub npz"
    # The mesh the agent supplied is echoed for the stage that still needs it.
    assert final["outputs"]["mesh_path"] == mesh_key
    assert final["missing"] == []
    # The import run also wrote a summary.json, as every entrypoint does. It is
    # not the flow summary stage_thermal consumes, so it is not offered as one.
    assert "flow_summary_path" not in final["outputs"]

    stored = standalone.service.get("local-import-1")
    assert stored.result.adapter_name == "local-stage-adapter"
    assert (standalone.project_root / stored.result.log_path).is_file()


def test_standalone_flow_consumes_the_import_outputs_verbatim(standalone) -> None:
    """The contract an agent follows: copy `outputs` into the next call. The
    import stage's output path has to be a key the flow stage accepts, resolve
    to the file it names, and arrive at the solver as its staged mesh."""

    mesh_key = standalone.call("cfd_register_local_mesh", path=MESH_KEY)["key"]
    imported = standalone.run_stage(
        "cfd_run_import",
        mesh_path=mesh_key,
        request_id="local-import-2",
    )

    flow = standalone.run_stage(
        "cfd_run_flow",
        mesh_npz_path=imported["outputs"]["mesh_npz_path"],
        num_steps=1,
        request_id="local-flow-2",
    )

    assert flow["status"] == "succeeded", flow
    assert flow["ambiguous"] == []
    for parameter, filename in (
        ("flow_summary_path", "summary.json"),
        ("flow_coupling_metadata_path", "flow_coupling_metadata.json"),
        ("flow_face_flux_path", "final_corrected_face_flux.npy"),
        ("flow_face_to_cells_path", "face_to_cells.npy"),
        ("flow_cell_volumes_path", "cell_volumes.npy"),
    ):
        value = flow["outputs"][parameter]
        assert value.endswith(f"/{filename}")
        assert (standalone.project_root / value).is_file()
    assert flow["outputs"]["mesh_npz_path"] == imported["outputs"]["mesh_npz_path"]
    # A flow run supplies everything a flow run can supply. stage_thermal's
    # `mesh_path` is not a flow parameter and is not this run's to miss: the
    # agent takes it from the import run, which echoed it.
    assert flow["missing"] == []
    assert "mesh_path" not in flow["outputs"]


def test_standalone_run_fails_when_the_input_is_outside_the_project(
    symlinks,
    standalone,
) -> None:
    """A key that is safe as a key but resolves outside the checkout is
    refused, and the refusal reaches the agent as a named failure rather than
    a stage that quietly read someone else's file."""

    outside = standalone.project_root.parent / "outside.msh"
    outside.write_bytes(b"$MeshFormat\n")
    (standalone.project_root / "data" / "meshes" / "link.msh").symlink_to(outside)

    submitted = standalone.call(
        "cfd_run_import",
        mesh_path="data/meshes/link.msh",
        request_id="local-import-escape",
    )
    final = standalone.poll(submitted["request_id"])

    assert final["status"] == "failed"
    assert final["error"]["code"] == "input_outside_project_root"
    assert final["artifact_count"] == 0
