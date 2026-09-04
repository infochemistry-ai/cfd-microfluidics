from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GMSH_EXPERIMENTS = PROJECT_ROOT / "experiments" / "gmsh"
MANIFEST_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_pipeline_manifest.py"
MANIFEST_TOOL_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_pipeline_manifest_tool.py"
IMPORT_SCRIPT = GMSH_EXPERIMENTS / "run_import_gmsh_mesh.py"
FLOW_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_tetra_flow_debug.py"
TRANSPORT_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_tetra_transport_debug.py"
THERMAL_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_tetra_thermal_debug.py"
SCALAR_SCRIPT = GMSH_EXPERIMENTS / "run_gmsh_tetra_scalar_debug.py"
THERMAL_BENCHMARK_SCRIPT = (
    GMSH_EXPERIMENTS / "run_gmsh_tetra_thermal_compile_benchmark.py"
)
SCRIPTS = (
    MANIFEST_SCRIPT,
    MANIFEST_TOOL_SCRIPT,
    IMPORT_SCRIPT,
    FLOW_SCRIPT,
    TRANSPORT_SCRIPT,
    THERMAL_SCRIPT,
    SCALAR_SCRIPT,
    THERMAL_BENCHMARK_SCRIPT,
)


def test_scripts_expose_help_surface() -> None:
    for script in SCRIPTS:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_stage_and_special_purpose_help_surfaces_expose_role_markers() -> None:
    expected_markers = {
        MANIFEST_SCRIPT: "supported manifest-first",
        MANIFEST_TOOL_SCRIPT: "manifest-first pipeline",
        IMPORT_SCRIPT: "direct import stage runner",
        FLOW_SCRIPT: "direct flow stage runner",
        TRANSPORT_SCRIPT: "direct transport stage runner",
        THERMAL_SCRIPT: "direct thermal stage runner",
        SCALAR_SCRIPT: "special-purpose",
        THERMAL_BENCHMARK_SCRIPT: "special-purpose",
    }

    for script, marker in expected_markers.items():
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert marker in completed.stdout.lower(), completed.stdout
