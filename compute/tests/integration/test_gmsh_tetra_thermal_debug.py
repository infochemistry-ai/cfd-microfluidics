from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.gmsh import run_gmsh_tetra_thermal_debug as thermal_runner
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from experiments.gmsh.run_gmsh_tetra_thermal_debug import _resolve_flow_face_flux_path
from microfluidics.gmsh.tetra.gmsh_tetra_regime_guardrail import (
    DEFAULT_MAX_GRID_PECLET,
    DEFAULT_MAX_PRANDTL,
)


def _collect_thermal_debug_cli_defaults() -> dict[str, object]:
    interesting_flags = {
        "--msh",
        "--velocity-source",
        "--steps",
        "--dt-mode",
        "--dt",
        "--cfl-target",
        "--cfl-limit",
        "--thermal-conductivity",
        "--thermal-diffusivity",
        "--kinematic-viscosity",
        "--max-supported-grid-peclet",
        "--max-supported-prandtl",
        "--heat-source",
        "--left-inlet-temperature",
        "--right-inlet-temperature",
    }
    source = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "gmsh"
        / "run_gmsh_tetra_thermal_debug.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults: dict[str, object] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in interesting_flags
        ):
            flag = node.args[0].value
            for keyword in node.keywords:
                if keyword.arg != "default":
                    continue
                if isinstance(keyword.value, ast.Name):
                    resolved = {
                        "DEFAULT_MAX_GRID_PECLET": DEFAULT_MAX_GRID_PECLET,
                        "DEFAULT_MAX_PRANDTL": DEFAULT_MAX_PRANDTL,
                    }.get(keyword.value.id)
                    if resolved is not None:
                        defaults[flag] = resolved
                else:
                    defaults[flag] = ast.literal_eval(keyword.value)
                break
    return defaults


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_tiny_thermal_mesh(tmp_path: Path):
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    boundary_face_tags = np.asarray([1, 3, 4, 4], dtype=np.int32)
    return build_imported_tetra_mesh(
        source_path=tmp_path / "tiny.msh",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data={
            "left_inlet": np.asarray([1, 2], dtype=np.int32),
            "outlet": np.asarray([3, 2], dtype=np.int32),
            "walls": np.asarray([4, 2], dtype=np.int32),
        },
    )


def _write_flow_payload(
    tmp_path: Path,
    mesh,
    *,
    ready_for_next_stage: bool,
    face_to_cells: np.ndarray | None = None,
    cell_volumes: np.ndarray | None = None,
) -> Path:
    flow_dir = tmp_path / "flow_payload"
    flow_dir.mkdir(parents=True, exist_ok=True)
    flux = np.linspace(0.05, 0.2, mesh.face_vertices.shape[0], dtype=np.float64)
    np.save(flow_dir / "final_corrected_face_flux.npy", flux)
    np.save(
        flow_dir / "face_to_cells.npy",
        np.asarray(
            face_to_cells if face_to_cells is not None else mesh.face_to_cells,
            dtype=np.int64,
        ),
    )
    np.save(
        flow_dir / "cell_volumes.npy",
        np.asarray(
            cell_volumes if cell_volumes is not None else mesh.cell_volumes,
            dtype=np.float64,
        ),
    )
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
                "stage_status": {
                    "run_completed": True,
                    "numerically_stable": True,
                    "physically_ready": bool(ready_for_next_stage),
                    "ready_for_next_stage": bool(ready_for_next_stage),
                    "ready_for_long_run": bool(ready_for_next_stage),
                    "stage_status_reason": (
                        "unit-stage-ready"
                        if ready_for_next_stage
                        else "unit-stage-unready"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    summary_path = flow_dir / "summary.json"
    _write_summary(
        summary_path,
        {
            "flow_mode": "unit",
            "artifacts": {
                "flow_coupling_metadata_json": str(
                    flow_dir / "flow_coupling_metadata.json"
                )
            },
        },
    )
    return summary_path


def test_resolve_flow_face_flux_accepts_valid_summary_payload(tmp_path: Path) -> None:
    mesh = _build_tiny_thermal_mesh(tmp_path)
    summary_path = _write_flow_payload(
        tmp_path,
        mesh,
        ready_for_next_stage=True,
    )

    face_flux, meta, summary_resolved = _resolve_flow_face_flux_path(
        explicit_path="",
        flow_summary_json=str(summary_path),
        run_dir=tmp_path,
        mesh=mesh,
    )
    assert face_flux.shape[0] == mesh.face_vertices.shape[0]
    assert summary_resolved == summary_path.resolve()
    assert bool(meta["flow_solved"]) is True
    assert bool(meta["ready_for_next_stage"]) is True
    assert bool(meta["manual_flow_input_override"]) is False
    assert str(meta["stage_status_reason"]) == "unit-stage-ready"


def test_resolve_flow_face_flux_rejects_unready_payload_by_default(
    tmp_path: Path,
) -> None:
    mesh = _build_tiny_thermal_mesh(tmp_path)
    summary_path = _write_flow_payload(
        tmp_path,
        mesh,
        ready_for_next_stage=False,
    )

    with pytest.raises(RuntimeError, match="thermal source flow readiness is false"):
        _resolve_flow_face_flux_path(
            explicit_path="",
            flow_summary_json=str(summary_path),
            run_dir=tmp_path,
            mesh=mesh,
        )


def test_resolve_flow_face_flux_rejects_incompatible_payload(tmp_path: Path) -> None:
    mesh = _build_tiny_thermal_mesh(tmp_path)
    bad_face_to_cells = np.asarray(mesh.face_to_cells, dtype=np.int64).copy()
    bad_face_to_cells[0, 0] = -1
    summary_path = _write_flow_payload(
        tmp_path,
        mesh,
        ready_for_next_stage=True,
        face_to_cells=bad_face_to_cells,
    )

    with pytest.raises(RuntimeError, match="incompatible"):
        _resolve_flow_face_flux_path(
            explicit_path="",
            flow_summary_json=str(summary_path),
            run_dir=tmp_path,
            mesh=mesh,
        )


def test_runner_wires_backend_and_summary_backend_execution(tmp_path: Path) -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    boundary_triangles = np.asarray(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
    boundary_face_tags = np.asarray([1, 3, 4, 4], dtype=np.int32)
    mesh = build_imported_tetra_mesh(
        source_path=tmp_path / "tiny.msh",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=boundary_face_tags,
        field_data={
            "left_inlet": np.asarray([1, 2], dtype=np.int32),
            "outlet": np.asarray([3, 2], dtype=np.int32),
            "walls": np.asarray([4, 2], dtype=np.int32),
        },
    )
    run_dir = tmp_path / "run"
    msh_path = tmp_path / "tiny.msh"
    msh_path.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="utf-8")

    sys_argv_prev = sys.argv[:]
    try:
        sys.argv = [
            "run_gmsh_tetra_thermal_debug.py",
            "--msh",
            str(msh_path),
            "--output-root",
            str(tmp_path),
            "--steps",
            "1",
            "--dt",
            "1e-4",
            "--dt-mode",
            "manual",
            "--velocity-source",
            "prescribed",
            "--backend",
            "torch",
            "--torch-device",
            "cpu",
            "--execution-profile",
            "production",
            "--allow-numpy-production-fallback",
            "--diagnostics-mode",
            "auto",
            "--history-mode",
            "auto",
            "--thermal-diffusivity",
            "1.4e-7",
            "--kinematic-viscosity",
            "1e-6",
            "--max-supported-prandtl",
            "1.0",
            "--history-stride",
            "2",
            "--diagnostics-stride",
            "3",
            "--full-step-diagnostics",
            "--debug-artifacts",
            "--wall-boundary-mode",
            "fixed_heat_flux",
            "--heated-wall-group",
            "walls",
            "--wall-heat-flux",
            "125.0",
        ]

        import microfluidics.gmsh.gmsh_mesh_import as mesh_import_module
        import microfluidics.gmsh.gmsh_mesh_validation as validation_module
        import microfluidics.gmsh.tetra.gmsh_tetra_operators as operators_module
        import microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields as velocity_module

        original_import_mesh = mesh_import_module.import_gmsh_tetra_mesh
        original_validate = validation_module.validate_imported_tetra_mesh
        original_format = validation_module.format_validation_report
        original_build_velocity = velocity_module.build_prescribed_velocity_field
        original_build_flux = operators_module.build_face_normal_flux_from_velocity
        original_run_operator_diag = operators_module.run_operator_diagnostics
        original_create_run_dir = thermal_runner.create_timestamped_run_dir
        original_export_vtu = thermal_runner._export_thermal_vtu
        original_save_previews = thermal_runner._save_temperature_previews

        mesh_import_module.import_gmsh_tetra_mesh = lambda _: mesh
        validation_module.validate_imported_tetra_mesh = lambda _: SimpleNamespace(
            is_valid=True, summary={"is_valid": True}
        )
        validation_module.format_validation_report = lambda _: "ok"
        velocity_module.build_prescribed_velocity_field = lambda *args, **kwargs: (
            SimpleNamespace(
                cell_velocity=np.zeros((mesh.tetrahedra.shape[0], 3), dtype=np.float64),
                boundary_face_velocity_overrides={},
                boundary_groups={
                    "left_inlet_faces": np.asarray(mesh.inlet_faces, dtype=np.int64),
                    "right_inlet_faces": np.zeros((0,), dtype=np.int64),
                    "outlet_faces": np.asarray(mesh.outlet_faces, dtype=np.int64),
                    "wall_faces": np.asarray(mesh.wall_faces, dtype=np.int64),
                },
                name="unit",
                metadata={},
            )
        )
        operators_module.build_face_normal_flux_from_velocity = lambda *args, **kwargs: (
            np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
            {
                "total_inlet_flux_in": 0.0,
                "total_outlet_flux_out": 0.0,
                "net_boundary_flux": 0.0,
                "wall_flux_max_abs": 0.0,
            },
        )
        operators_module.run_operator_diagnostics = lambda *args, **kwargs: {}
        thermal_runner.create_timestamped_run_dir = lambda *args, **kwargs: run_dir
        thermal_runner._export_thermal_vtu = lambda **kwargs: run_dir / "result.vtu"
        thermal_runner._save_temperature_previews = lambda **kwargs: {
            "temperature_xy": str(run_dir / "preview.png")
        }

        thermal_runner.main()
    finally:
        sys.argv = sys_argv_prev
        mesh_import_module.import_gmsh_tetra_mesh = original_import_mesh
        validation_module.validate_imported_tetra_mesh = original_validate
        validation_module.format_validation_report = original_format
        velocity_module.build_prescribed_velocity_field = original_build_velocity
        operators_module.build_face_normal_flux_from_velocity = original_build_flux
        operators_module.run_operator_diagnostics = original_run_operator_diag
        thermal_runner.create_timestamped_run_dir = original_create_run_dir
        thermal_runner._export_thermal_vtu = original_export_vtu
        thermal_runner._save_temperature_previews = original_save_previews

    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    thermal = payload["thermal"]
    assert thermal["config"]["backend"] == "torch"
    assert thermal["config"]["torch_device"] == "cpu"
    assert thermal["config"]["execution_profile"] == "production"
    assert thermal["config"]["diagnostics_mode"] == "auto"
    assert thermal["config"]["history_mode"] == "auto"
    assert thermal["config"]["history_stride"] == 2
    assert thermal["config"]["diagnostics_stride"] == 3
    assert thermal["config"]["full_step_diagnostics"] is True
    assert thermal["config"]["debug_artifacts"] is True
    assert thermal["config"]["wall_boundary_mode"] == "fixed_heat_flux"
    assert thermal["config"]["heated_wall_physical_groups"] == ["walls"]
    assert thermal["config"]["wall_heat_flux"] == 125.0
    assert thermal["backend_execution"]["requested_backend"] == "torch"
    assert "stepping_backend" in thermal["backend_execution"]
    assert thermal["execution_policy"]["execution_profile"] == "production"
    assert thermal["execution_policy"]["selected_backend"] in {"torch", "numpy"}
    assert thermal["execution_policy"]["production_backend_satisfied"] is (
        thermal["execution_policy"]["selected_backend"] == "torch"
    )
    assert thermal["history_settings"]["history_stride"] == 2
    assert thermal["history_settings"]["resolved_diagnostics_mode"] == "fast"
    assert thermal["history_settings"]["resolved_history_mode"] == "compact"
    assert thermal["history_settings"]["diagnostics_stride"] == 3
    assert thermal["history_settings"]["full_step_diagnostics"] is True
    assert thermal["history_settings"]["debug_artifacts"] is True
    assert (
        thermal["diffusion_backend"]
        == thermal["diffusion_diagnostics"]["diffusion_backend"]
    )
    assert thermal["thermal_accuracy_supported"] is False
    assert thermal["thermal_accuracy_support_status"] == "warning_high_prandtl"
    assert thermal["thermal_regime_warning"] is True
    assert thermal["thermal_regime_warning_codes"] == ["warning_high_prandtl"]
    assert thermal["thermal_regime_blocking_error"] is False
    assert thermal["thermal_regime_audit"]["scalar_kind"] == "thermal"
    assert "prandtl_number" in thermal["thermal_regime_audit"]["inputs"]
    assert "schmidt_number" not in thermal["thermal_regime_audit"]["inputs"]
    assert thermal["wall_heat_transfer"]["resolved_group_face_counts"] == {"walls": 2}
    assert thermal["wall_heat_transfer"]["cumulative_requested_wall_energy_j"] > 0.0
    assert "outlet_temperature_flow_weighted_std" in thermal["thermal_observables"]
    assert "stage_status" in payload
    assert payload["stage_status"]["physically_ready"] is True
    assert (
        "thermal_regime_warning_codes"
        not in payload["stage_status"]["stage_status_checks"]
    )
    assert payload["stage_status"]["stage_status_warnings"] == ["warning_high_prandtl"]
    assert (run_dir / "thermal_regime_audit.json").exists()
    assert (run_dir / "stage_status.json").exists()
    assert (run_dir / "acceptance_report.json").exists()


def test_thermal_debug_cli_defaults_match_stage4_profile() -> None:
    defaults = _collect_thermal_debug_cli_defaults()

    assert defaults["--msh"] == ""
    assert defaults["--velocity-source"] == ""
    assert defaults["--steps"] == 200
    assert defaults["--dt-mode"] == "auto"
    assert defaults["--dt"] == 5e-4
    assert defaults["--cfl-target"] == 0.5
    assert defaults["--cfl-limit"] == 0.8
    assert defaults["--thermal-conductivity"] == 0.6
    assert defaults["--thermal-diffusivity"] == -1.0
    assert defaults["--kinematic-viscosity"] == 1e-6
    assert defaults["--max-supported-grid-peclet"] == 2.0
    assert defaults["--max-supported-prandtl"] == 100.0
    assert defaults["--heat-source"] == 0.0
    assert defaults["--left-inlet-temperature"] == 303.15
    assert defaults["--right-inlet-temperature"] == 303.15
