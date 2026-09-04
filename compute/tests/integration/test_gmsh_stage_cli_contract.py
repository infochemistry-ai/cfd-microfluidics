from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMPORT_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_import_gmsh_mesh.py"
FLOW_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"
TRANSPORT_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_transport_debug.py"
)
THERMAL_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_thermal_debug.py"
)
SCALAR_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_scalar_debug.py"


def _run_raw(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )


def test_flow_debug_requires_explicit_mesh_npz() -> None:
    completed = _run_raw([sys.executable, str(FLOW_SCRIPT), "--flow-steps", "1"])
    assert completed.returncode != 0
    assert "--mesh-npz is required" in completed.stderr


def test_pressure_determinism_requires_fixed_work_source() -> None:
    completed = _run_raw(
        [
            sys.executable,
            str(FLOW_SCRIPT),
            "--mesh-npz",
            "unused.npz",
            "--pressure-determinism-diagnostic",
        ]
    )
    assert completed.returncode != 0
    assert "requires --fixed-work-source-run-dir" in completed.stderr


def test_import_debug_requires_explicit_msh() -> None:
    completed = _run_raw([sys.executable, str(IMPORT_SCRIPT)])
    assert completed.returncode != 0
    assert "--msh is required" in completed.stderr


def test_transport_debug_requires_explicit_mesh_npz() -> None:
    completed = _run_raw([sys.executable, str(TRANSPORT_SCRIPT), "--steps", "1"])
    assert completed.returncode != 0
    assert "--mesh-npz is required" in completed.stderr


def test_transport_debug_requires_explicit_velocity_source(tmp_path: Path) -> None:
    mesh_npz = tmp_path / "dummy_imported_mesh.npz"
    mesh_npz.write_bytes(b"placeholder")
    completed = _run_raw(
        [
            sys.executable,
            str(TRANSPORT_SCRIPT),
            "--mesh-npz",
            str(mesh_npz),
            "--steps",
            "1",
        ]
    )
    assert completed.returncode != 0
    assert "--velocity-source is required" in completed.stderr


def test_thermal_debug_requires_explicit_msh() -> None:
    completed = _run_raw(
        [
            sys.executable,
            str(THERMAL_SCRIPT),
            "--velocity-source",
            "prescribed",
            "--steps",
            "1",
        ]
    )
    assert completed.returncode != 0
    assert "--msh is required" in completed.stderr


def test_thermal_debug_requires_explicit_velocity_source() -> None:
    completed = _run_raw(
        [
            sys.executable,
            str(THERMAL_SCRIPT),
            "--msh",
            str(PROJECT_ROOT / "data" / "meshes" / "gmsh" / "t_junction.msh"),
            "--steps",
            "1",
        ]
    )
    assert completed.returncode != 0
    assert "--velocity-source is required" in completed.stderr


def test_scalar_debug_requires_explicit_mesh_npz() -> None:
    completed = _run_raw([sys.executable, str(SCALAR_SCRIPT), "--steps", "1"])
    assert completed.returncode != 0
    assert "--mesh-npz is required" in completed.stderr
