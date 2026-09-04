from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
SCALAR_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_scalar_debug.py"


def _win_path(path: Path) -> str:
    return str(path).replace("/", "\\")


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


def test_gmsh_pipeline_normalizes_windows_style_paths(tmp_path: Path) -> None:
    mesh_name = "t_junction"
    msh_path = DATA_GMSH / f"{mesh_name}.msh"

    import_root = tmp_path / "results" / "gmsh_import_runs"
    flow_root = tmp_path / "results" / "gmsh_tetra_flow_runs"
    transport_root = tmp_path / "results" / "gmsh_tetra_transport_runs"
    thermal_root = tmp_path / "results" / "gmsh_tetra_thermal_runs"
    scalar_root = tmp_path / "results" / "gmsh_tetra_scalar_runs"

    _run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            "--msh",
            _win_path(msh_path),
            "--output-root",
            _win_path(import_root),
        ]
    )
    import_dir = _latest_run_dir(import_root, mesh_name)
    import_summary = json.loads(
        (import_dir / "summary.json").read_text(encoding="utf-8")
    )
    mesh_npz = import_dir / f"{mesh_name}_imported_mesh.npz"
    assert mesh_npz.exists()
    assert import_summary["artifacts"]["mesh_npz"] == str(mesh_npz)

    _run(
        [
            sys.executable,
            str(FLOW_SCRIPT),
            "--mesh-npz",
            _win_path(mesh_npz),
            "--import-root",
            _win_path(import_root),
            "--output-root",
            _win_path(flow_root),
            "--backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
        ]
    )
    flow_dir = _latest_run_dir(flow_root, mesh_name)
    flow_summary = json.loads((flow_dir / "summary.json").read_text(encoding="utf-8"))
    assert flow_summary["artifacts"]["run_log"] == str(flow_dir / "run.log")

    _run(
        [
            sys.executable,
            str(TRANSPORT_SCRIPT),
            "--mesh-npz",
            _win_path(mesh_npz),
            "--import-root",
            _win_path(import_root),
            "--output-root",
            _win_path(transport_root),
            "--backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            _win_path(flow_dir),
            "--steps",
            "1",
            "--allow-unready-flow",
            "--dt-mode",
            "manual",
            "--dt",
            "1e-4",
        ]
    )
    transport_dir = _latest_run_dir(transport_root, mesh_name)
    transport_summary = json.loads(
        (transport_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert transport_summary["artifacts"]["run_log"] == str(transport_dir / "run.log")

    _run(
        [
            sys.executable,
            str(THERMAL_SCRIPT),
            "--msh",
            _win_path(msh_path),
            "--output-root",
            _win_path(thermal_root),
            "--backend",
            "numpy",
            "--velocity-source",
            "prescribed",
            "--steps",
            "1",
            "--dt-mode",
            "manual",
            "--dt",
            "1e-4",
        ]
    )
    thermal_dir = _latest_run_dir(thermal_root, mesh_name)
    thermal_summary = json.loads(
        (thermal_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert thermal_summary["artifacts"]["run_log"] == str(thermal_dir / "run.log")

    _run(
        [
            sys.executable,
            str(SCALAR_SCRIPT),
            "--mesh-npz",
            _win_path(mesh_npz),
            "--import-root",
            _win_path(import_root),
            "--output-root",
            _win_path(scalar_root),
            "--backend",
            "numpy",
            "--steps",
            "1",
            "--dt",
            "1e-4",
        ]
    )
    scalar_dir = _latest_run_dir(scalar_root, mesh_name)
    scalar_summary = json.loads(
        (scalar_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert scalar_summary["artifacts"]["run_log"] == str(scalar_dir / "run.log")
    assert (scalar_dir / "summary.json").exists()
    assert (scalar_dir / "run.log").exists()

    if os.name != "nt":
        assert not (tmp_path / r"results\gmsh_import_runs").exists()
        assert not (tmp_path / r"results\gmsh_tetra_flow_runs").exists()
        assert not (tmp_path / r"results\gmsh_tetra_transport_runs").exists()
        assert not (tmp_path / r"results\gmsh_tetra_thermal_runs").exists()
        assert not (tmp_path / r"results\gmsh_tetra_scalar_runs").exists()
