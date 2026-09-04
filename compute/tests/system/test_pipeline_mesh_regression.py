from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_REGRESSION = os.environ.get("RUN_PIPELINE_MESH_REGRESSION", "0") == "1"


def _latest_run_dir(root: Path, mesh_name: str) -> Path:
    candidates = sorted(
        [p for p in root.glob(f"*_{mesh_name}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory found for mesh={mesh_name} in {root}"
        )
    return candidates[-1]


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n--- stdout ---\n"
            + completed.stdout
            + "\n--- stderr ---\n"
            + completed.stderr
        )


@pytest.mark.skipif(
    not RUN_REGRESSION,
    reason=(
        "Set RUN_PIPELINE_MESH_REGRESSION=1 to run multi-mesh flow->transport "
        "regression suite."
    ),
)
@pytest.mark.parametrize("mesh_name", ["TJ_test2", "t_mixer", "t_mixer_2"])
def test_pipeline_status_consistency_on_reference_meshes(
    tmp_path: Path,
    mesh_name: str,
) -> None:
    import_root = tmp_path / "import_runs"
    flow_root = tmp_path / "flow_runs"
    transport_root = tmp_path / "transport_runs"
    import_root.mkdir(parents=True, exist_ok=True)
    flow_root.mkdir(parents=True, exist_ok=True)
    transport_root.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "gmsh" / "run_import_gmsh_mesh.py"),
            "--msh",
            f"{mesh_name}.msh",
            "--output-root",
            str(import_root),
        ]
    )
    import_dir = _latest_run_dir(import_root, mesh_name)
    mesh_npz = import_dir / f"{mesh_name}_imported_mesh.npz"
    assert mesh_npz.exists()

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"),
            "--mesh-npz",
            str(mesh_npz),
            "--output-root",
            str(flow_root),
            "--backend",
            "numpy",
            "--flow-mode",
            "navier_stokes_projection_debug",
            "--flow-steps",
            "8",
            "--flow-dt-mode",
            "auto_cfl",
            "--convective-stabilization-mode",
            "substepping",
        ]
    )
    flow_dir = _latest_run_dir(flow_root, mesh_name)
    flow_summary = json.loads((flow_dir / "summary.json").read_text(encoding="utf-8"))

    _run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "experiments"
                / "gmsh"
                / "run_gmsh_tetra_transport_debug.py"
            ),
            "--mesh-npz",
            str(mesh_npz),
            "--output-root",
            str(transport_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_dir),
            "--steps",
            "120",
            "--dt-mode",
            "auto",
        ]
    )
    transport_dir = _latest_run_dir(transport_root, mesh_name)
    transport_summary = json.loads(
        (transport_dir / "summary.json").read_text(encoding="utf-8")
    )

    for key in (
        "run_completed",
        "numerically_stable",
        "physically_ready",
        "ready_for_next_stage",
        "ready_for_long_run",
    ):
        assert key in flow_summary
        assert key in transport_summary
        assert isinstance(flow_summary[key], bool)
        assert isinstance(transport_summary[key], bool)

    if flow_summary["ready_for_next_stage"]:
        assert flow_summary["run_completed"]
        assert flow_summary["numerically_stable"]
        assert flow_summary["physically_ready"]
    if flow_summary["ready_for_long_run"]:
        assert flow_summary["ready_for_next_stage"]

    if transport_summary["ready_for_next_stage"]:
        assert transport_summary["run_completed"]
        assert transport_summary["numerically_stable"]
        assert transport_summary["physically_ready"]
    if transport_summary["ready_for_long_run"]:
        assert transport_summary["ready_for_next_stage"]

    if transport_summary["long_run_status"] in {
        "success",
        "success_with_tiny_boundedness_tolerance",
    }:
        assert transport_summary["ready_for_flow_transport_pipeline"]
        assert transport_summary["physically_ready"]
