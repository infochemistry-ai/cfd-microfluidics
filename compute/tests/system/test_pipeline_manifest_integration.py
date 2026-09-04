from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from microfluidics.pipeline.manifest import (
    load_pipeline_manifest,
    resolve_manifest_path_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_GMSH = PROJECT_ROOT / "data/meshes/gmsh"
IMPORT_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_import_gmsh_mesh.py"
FLOW_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
TRANSPORT_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_transport_debug.py"
)
THERMAL_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_thermal_debug.py"
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
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
    return completed


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


def test_optional_pipeline_manifest_records_core_stage_dag(tmp_path: Path) -> None:
    mesh_name = "t_junction"
    msh_path = DATA_GMSH / f"{mesh_name}.msh"

    pipeline_root = tmp_path / "results" / "manifest_pipeline"
    import_root = pipeline_root / "import_runs"
    flow_root = pipeline_root / "flow_runs"
    transport_root = pipeline_root / "transport_runs"
    thermal_root = pipeline_root / "thermal_runs"
    manifest_path = pipeline_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            "--msh",
            str(msh_path),
            "--output-root",
            str(import_root),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "import-001",
        ]
    )
    import_dir = _latest_run_dir(import_root, mesh_name)
    mesh_npz = import_dir / f"{mesh_name}_imported_mesh.npz"
    assert mesh_npz.exists()

    _run(
        [
            sys.executable,
            str(FLOW_SCRIPT),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(flow_root),
            "--backend",
            "numpy",
            "--flow-execution-backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "flow-001",
            "--pipeline-parent-run-id",
            "import-001",
        ]
    )
    flow_dir = _latest_run_dir(flow_root, mesh_name)
    flow_summary = flow_dir / "summary.json"
    flow_flux = flow_dir / "final_corrected_face_flux.npy"
    assert flow_summary.exists()
    assert flow_flux.exists()

    _run(
        [
            sys.executable,
            str(TRANSPORT_SCRIPT),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
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
            "1",
            "--allow-unready-flow",
            "--dt-mode",
            "manual",
            "--dt",
            "1e-4",
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "transport-001",
            "--pipeline-parent-run-id",
            "flow-001",
        ]
    )
    transport_dir = _latest_run_dir(transport_root, mesh_name)
    assert (transport_dir / "summary.json").exists()

    _run(
        [
            sys.executable,
            str(THERMAL_SCRIPT),
            "--msh",
            str(msh_path),
            "--output-root",
            str(thermal_root),
            "--backend",
            "numpy",
            "--velocity-source",
            "flow_solver",
            "--flow-summary-json",
            str(flow_summary),
            "--allow-unready-flow",
            "--steps",
            "1",
            "--dt-mode",
            "manual",
            "--dt",
            "1e-4",
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "thermal-001",
            "--pipeline-parent-run-id",
            "flow-001",
        ]
    )
    thermal_dir = _latest_run_dir(thermal_root, mesh_name)
    assert (thermal_dir / "summary.json").exists()

    manifest = load_pipeline_manifest(manifest_path)
    assert set(manifest["runs"]) == {
        "import-001",
        "flow-001",
        "transport-001",
        "thermal-001",
    }
    assert manifest["runs"]["flow-001"]["parent_run_ids"] == ["import-001"]
    assert manifest["runs"]["transport-001"]["parent_run_ids"] == ["flow-001"]
    assert manifest["runs"]["thermal-001"]["parent_run_ids"] == ["flow-001"]

    assert (
        resolve_manifest_path_reference(
            manifest_path,
            manifest["runs"]["import-001"]["run_dir"],
        )
        == import_dir.resolve()
    )
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            manifest["runs"]["flow-001"]["run_dir"],
        )
        == flow_dir.resolve()
    )
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            manifest["runs"]["transport-001"]["run_dir"],
        )
        == transport_dir.resolve()
    )
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            manifest["runs"]["thermal-001"]["run_dir"],
        )
        == thermal_dir.resolve()
    )

    assert (
        manifest["runs"]["flow-001"]["outputs"]["final_corrected_face_flux_npy"]
        == "flow_runs/" + flow_dir.name + "/final_corrected_face_flux.npy"
    )
    assert (
        manifest["runs"]["transport-001"]["inputs"]["flow_run_dir"]
        == "flow_runs/" + flow_dir.name
    )
    assert (
        manifest["runs"]["thermal-001"]["inputs"]["flow_summary_json"]
        == "flow_runs/" + flow_dir.name + "/summary.json"
    )
