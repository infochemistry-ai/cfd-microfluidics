from __future__ import annotations

import ast
import json
from pathlib import Path

from experiments.gmsh.run_gmsh_pipeline_manifest import _flow_ready_for_downstream
from experiments.gmsh.run_gmsh_tetra_flow_debug import (
    _resolve_convective_predictor_setting,
    _resolve_viscous_predictor_mode,
)


def _write_manifest(
    tmp_path: Path,
    *,
    ready_for_next_stage: bool,
    physically_ready: bool,
    ready_for_long_run_debug: bool,
) -> Path:
    manifest_path = tmp_path / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pipeline_id": "unit",
                "pipeline_root": ".",
                "metadata": {},
                "runs": {
                    "flow-1": {
                        "run_id": "flow-1",
                        "stage_type": "flow",
                        "status": "completed",
                        "display_name": None,
                        "parent_run_ids": [],
                        "run_dir": "results/flow",
                        "created_at": "2026-06-24T00:00:00+00:00",
                        "updated_at": "2026-06-24T00:00:00+00:00",
                        "inputs": {},
                        "outputs": {},
                        "artifacts": {},
                        "metadata": {
                            "ready_for_next_stage": ready_for_next_stage,
                            "physically_ready": physically_ready,
                            "ready_for_long_run_debug": ready_for_long_run_debug,
                            "stage_status_reason": "unit-ready-state",
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_navier_debug_mode_enables_convective_predictor_by_default() -> None:
    enabled, reason = _resolve_convective_predictor_setting(
        flow_mode="navier_stokes_projection_debug",
        enable_convective_predictor=False,
        disable_convective_predictor=False,
    )
    assert enabled is True
    assert "enables convective predictor by default" in reason


def _collect_flow_debug_cli_defaults() -> dict[str, object]:
    source_path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_flow_debug.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    constants: dict[str, object] = {}
    defaults: dict[str, object] = {}

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = node.value
            if isinstance(value, ast.Constant):
                constants[node.targets[0].id] = value.value

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        option = node.args[0].value
        if not isinstance(option, str) or not option.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            value = keyword.value
            if isinstance(value, ast.Constant):
                defaults[option] = value.value
            elif isinstance(value, ast.Name) and value.id in constants:
                defaults[option] = constants[value.id]

    return defaults


def test_direct_flow_debug_cli_defaults_match_stage2_profile() -> None:
    defaults = _collect_flow_debug_cli_defaults()

    assert defaults["--flow-steps"] == 20
    assert defaults["--flow-dt-mode"] == "auto_cfl"
    assert defaults["--flow-mode"] == "navier_stokes_projection_debug"
    assert defaults["--wall-velocity-boundary-mode"] == "slip"
    assert defaults["--disable-convective-auto-damping"] is True
    assert defaults["--convective-stabilization-mode"] == "auto_damping"
    assert defaults["--viscous-predictor-mode"] == ""
    assert defaults["--viscous-face-flux-divergence-impact-cap"] == 0.03
    assert defaults["--startup-warning-steps"] == 10
    assert defaults["--max-pressure-iterations"] == 1000
    assert defaults["--pressure-relative-tolerance"] == 1e-4
    assert defaults["--pressure-solver"] == "pcg_diag"


def test_slip_wall_resolves_face_flux_viscous_predictor_by_default() -> None:
    resolved, reason = _resolve_viscous_predictor_mode(
        flow_mode="navier_stokes_projection_debug",
        predictor_mode_cli="",
        predictor_mode_explicit=False,
        wall_velocity_boundary_mode="slip",
    )

    assert resolved == "face_flux_laplacian_substepped"
    assert "face_flux_laplacian_substepped" in reason


def test_no_slip_wall_resolver_still_works_when_mode_not_explicit() -> None:
    resolved, reason = _resolve_viscous_predictor_mode(
        flow_mode="stokes_viscous_projection",
        predictor_mode_cli="",
        predictor_mode_explicit=False,
        wall_velocity_boundary_mode="no_slip",
    )

    assert resolved == "explicit_cell_velocity_laplacian_substepped_conservative"
    assert "no-slip wall mode defaults" in reason


def test_explicit_viscous_predictor_override_keeps_priority() -> None:
    resolved, reason = _resolve_viscous_predictor_mode(
        flow_mode="navier_stokes_projection_debug",
        predictor_mode_cli="face_flux_laplacian_substepped",
        predictor_mode_explicit=True,
        wall_velocity_boundary_mode="no_slip",
    )

    assert resolved == "face_flux_laplacian_substepped"
    assert reason == "viscous predictor set explicitly by CLI"


def test_downstream_flow_readiness_uses_physical_ready_gate(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        ready_for_next_stage=False,
        physically_ready=False,
        ready_for_long_run_debug=True,
    )

    ready, reason = _flow_ready_for_downstream(manifest_path, flow_run_id="flow-1")

    assert ready is False
    assert reason == "unit-ready-state"
