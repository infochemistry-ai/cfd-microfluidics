"""Portable repo-relative path contract for local compute code."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

RESULTS_ROOT_REL = Path("results")
DATA_ROOT = Path("data")
GMSH_MESHES_ROOT = DATA_ROOT / "meshes" / "gmsh"
GMSH_GEOMETRY_ROOT = DATA_ROOT / "geometry" / "gmsh"
DATA_EXAMPLES_ROOT = DATA_ROOT / "examples"
GMSH_IMPORT_RUNS_ROOT_REL = RESULTS_ROOT_REL / "gmsh_import_runs"
GMSH_TETRA_FLOW_RUNS_ROOT_REL = RESULTS_ROOT_REL / "gmsh_tetra_flow_runs"
GMSH_TETRA_TRANSPORT_RUNS_ROOT_REL = RESULTS_ROOT_REL / "gmsh_tetra_transport_runs"
GMSH_TETRA_THERMAL_RUNS_ROOT_REL = RESULTS_ROOT_REL / "gmsh_tetra_thermal_runs"
GMSH_TETRA_SCALAR_RUNS_ROOT_REL = RESULTS_ROOT_REL / "gmsh_tetra_scalar_runs"
MESH_NATIVE_FLOW_RUNS_ROOT_REL = RESULTS_ROOT_REL / "mesh_native_flow_runs"
SERVICE_RUNS_ROOT_REL = RESULTS_ROOT_REL / "service_runs"
EXPERIMENT_RUNS_ROOT_REL = RESULTS_ROOT_REL / "experiment_runs"


def resolve_repo_path(project_root: Path, relative_path: Path) -> Path:
    """Resolve a repo-relative path against a project root."""

    if relative_path.is_absolute():
        return relative_path.resolve()
    return (project_root / relative_path).resolve()


def normalize_user_path(value: str | Path) -> Path:
    """Normalize user-provided paths while preserving platform conventions."""

    raw = str(value)
    if os.name != "nt":
        raw = raw.replace("\\", "/")
    return Path(raw)


def resolve_service_runs_root(project_root: Path, run_root: str | Path) -> Path:
    """The directory the compute service creates per-run work directories in.

    Shared by `backend.app.service` (which creates the directories) and
    `backend.app.adapters.local_stage_adapter` (which has to know whether the
    files it writes there can be named as repository-relative paths), so the
    two cannot disagree about where runs land.
    """

    raw = str(run_root).strip() or str(SERVICE_RUNS_ROOT_REL)
    return (Path(project_root).resolve() / normalize_user_path(raw)).resolve()


def create_timestamped_run_dir(base_dir: Path, stem: str) -> Path:
    """Create a timestamped run directory using the current naming contract."""

    base_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = str(stem).strip() or "run"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{stamp}_{safe_stem}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{stamp}_{safe_stem}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
