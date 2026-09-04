from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from microfluidics.pipeline import (
    get_pipeline_run,
    load_pipeline_manifest,
    save_pipeline_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_RUNNER_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_pipeline_manifest.py"
)
IMPORT_SCRIPT = PROJECT_ROOT / "experiments" / "gmsh" / "run_import_gmsh_mesh.py"


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


def _run_raw(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )


def test_manifest_pipeline_runner_runs_exact_stage_handoffs(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_run"
    manifest_path = run_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--mesh-name",
            "t_junction",
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
            "--allow-unready-flow-for-downstream",
            "--transport-backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--transport-steps",
            "1",
            "--run-thermal",
            "--thermal-backend",
            "numpy",
            "--thermal-steps",
            "1",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    runs = manifest["runs"]
    assert len(runs) == 4

    import_run = next(run for run in runs.values() if run.get("stage_type") == "import")
    flow_run = next(run for run in runs.values() if run.get("stage_type") == "flow")
    transport_run = next(
        run for run in runs.values() if run.get("stage_type") == "transport"
    )
    thermal_run = next(
        run for run in runs.values() if run.get("stage_type") == "thermal"
    )

    assert get_pipeline_run(manifest, import_run["run_id"])["status"] == "completed"
    assert flow_run["status"] == "completed"
    assert transport_run["status"] == "completed"
    assert thermal_run["status"] == "completed"

    assert flow_run["parent_run_ids"] == [import_run["run_id"]]
    assert transport_run["parent_run_ids"] == [flow_run["run_id"]]
    assert thermal_run["parent_run_ids"] == [flow_run["run_id"]]

    flow_outputs = dict(flow_run.get("outputs", {}))
    transport_outputs = dict(transport_run.get("outputs", {}))
    transport_inputs = dict(transport_run.get("inputs", {}))
    thermal_inputs = dict(thermal_run.get("inputs", {}))
    thermal_outputs = dict(thermal_run.get("outputs", {}))

    assert flow_outputs["summary_json"].startswith("flow_runs/")
    assert flow_outputs["final_corrected_face_flux_npy"].startswith("flow_runs/")
    assert transport_inputs["flow_run_dir"].startswith("flow_runs/")
    assert thermal_inputs["flow_summary_json"] == flow_outputs["summary_json"]
    assert thermal_inputs["flow_face_flux_npy"] == ""

    transport_config_path = run_root / transport_outputs["config_json"]
    transport_config = json.loads(transport_config_path.read_text(encoding="utf-8"))
    assert transport_config["transport_mode"] == "advection_diffusion"
    assert transport_config["transport_scheme"] == "bounded_upwind"
    assert transport_config["dt_mode"] == "auto"
    assert transport_config["dt"] == 1.5e-5
    assert transport_config["cfl_target"] == 0.5
    assert transport_config["cfl_limit"] == 0.8
    assert transport_config["diffusivity"] == 3e-10
    assert transport_config["gradient_method"] == "least_squares"
    assert transport_config["laplacian_method"] == "tpfa"
    assert transport_config["left_inlet_value"] == 0.0
    assert transport_config["right_inlet_value"] == 1.0
    assert transport_config["inlet_speed"] == 0.15
    assert transport_config["checkpoint_every"] == 5000
    assert transport_config["progress_every"] == 500
    assert transport_config["no_velocity_comparison"] is True
    assert transport_config["no_transport_scheme_comparison"] is True
    assert transport_config["fail_if_numpy_fallback"] is False

    thermal_config_path = run_root / thermal_outputs["config_json"]
    thermal_config = json.loads(thermal_config_path.read_text(encoding="utf-8"))
    assert thermal_config["velocity_source"] == "flow_solver"
    assert thermal_config["steps"] == 1
    assert thermal_config["dt_mode"] == "auto"
    assert thermal_config["dt"] == 5e-4
    assert thermal_config["cfl_target"] == 0.5
    assert thermal_config["cfl_limit"] == 0.8
    assert thermal_config["heat_source"] == 0.0

    import_summary_path = run_root / import_run["outputs"]["summary_json"]
    import_summary = json.loads(import_summary_path.read_text(encoding="utf-8"))
    scale_audit = import_summary.get("mesh_scale_audit", {})
    assert all(float(value) > 0.0 for value in scale_audit["bbox_extents"])


def test_manifest_pipeline_runner_runs_mutually_exclusive_reactive_branch(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "manifest_reactive"
    case = json.loads(
        (
            PROJECT_ROOT
            / "data"
            / "examples"
            / "reactive"
            / "t_mixer_exothermic_ab.json"
        ).read_text(encoding="utf-8")
    )
    case["mode"] = "nonisothermal"
    case["initial_state"]["concentrations_mol_per_m3"] = {
        "A": 10.0,
        "B": 5.0,
        "C": 0.0,
    }
    for inlet in case["inlets"].values():
        inlet["concentrations_mol_per_m3"] = {
            "A": 100.0,
            "B": 50.0,
            "C": 1.0,
        }
    case["mechanism"]["reactions"][0]["kinetics"] = {
        "A": "0.1 m^3/(mol*s)",
        "Ea": "0.0 J/mol",
    }
    case["time"]["num_steps"] = 2
    case["time"]["dt_mode"] = "manual"
    case["time"]["dt_s"] = 1.0e-5
    case["time"]["max_dt_s"] = 1.0e-5
    case["output"]["history_stride"] = 1
    case_path = tmp_path / "reactive_case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
            "--allow-unready-flow-for-downstream",
            "--reactive-case",
            str(case_path),
            "--reactive-max-walltime-seconds",
            "120",
        ]
    )

    manifest = load_pipeline_manifest(run_root / "pipeline_manifest.json")
    stage_types = [run["stage_type"] for run in manifest["runs"].values()]
    assert sorted(stage_types) == ["flow", "import", "reactive_transport"]
    flow_run = next(
        run for run in manifest["runs"].values() if run["stage_type"] == "flow"
    )
    reactive_run = next(
        run
        for run in manifest["runs"].values()
        if run["stage_type"] == "reactive_transport"
    )
    assert reactive_run["parent_run_ids"] == [flow_run["run_id"]]
    assert reactive_run["status"] == "completed"
    flow_flux_path = run_root / flow_run["outputs"]["final_corrected_face_flux_npy"]
    assert np.any(np.abs(np.load(flow_flux_path)) > 0.0)

    summary_path = run_root / reactive_run["outputs"]["summary_json"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "nonisothermal"
    assert summary["readiness"]["run_completed"] is True
    assert summary["readiness"]["numerically_stable"] is True
    assert (
        summary["readiness"]["ready_for_next_stage"]
        is summary["readiness"]["physically_ready"]
    )
    assert summary["readiness"]["cli_success"] is True
    assert summary["runtime"]["walltime_limit_s"] == 120.0
    assert (
        max(
            item["relative_residual"]
            for item in summary["balances"]["species"].values()
        )
        <= 1.0e-5
    )
    assert summary["balances"]["energy"]["relative_residual"] <= 1.0e-5

    fields_path = Path(summary["artifacts"]["reactive_fields_npz"])
    with np.load(fields_path, allow_pickle=False) as fields:
        concentrations = fields["concentrations_mol_per_m3"]
        temperature = fields["temperature_k"]
        heat_release = fields["heat_release_w_per_m3"]
    assert np.max(concentrations[:, 2]) > 0.0
    assert np.max(temperature) > 298.15
    assert np.max(heat_release) > 0.0


def test_manifest_pipeline_runner_stage2_flow_defaults_reach_flow_stage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_stage2_defaults"
    manifest_path = run_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--mesh-name",
            "t_junction",
            "--flow-backend",
            "numpy",
            "--flow-steps",
            "1",
            "--skip-transport",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    flow_run = next(
        run for run in manifest["runs"].values() if run["stage_type"] == "flow"
    )
    flow_config = json.loads(
        (run_root / flow_run["outputs"]["config_json"]).read_text(encoding="utf-8")
    )
    flow_summary = json.loads(
        (run_root / flow_run["outputs"]["summary_json"]).read_text(encoding="utf-8")
    )
    cli_args = flow_config["cli_args"]

    assert cli_args["flow_mode"] == "navier_stokes_projection_debug"
    assert cli_args["flow_steps"] == 1
    assert cli_args["inlet_speed"] == 0.15
    assert cli_args["flow_dt_mode"] == "auto_cfl"
    assert cli_args["flow_dt"] is None
    assert cli_args["flow_dt_min"] == 1e-7
    assert cli_args["flow_dt_max"] is None
    assert cli_args["convective_cfl_target"] == 0.5
    assert cli_args["wall_velocity_boundary_mode"] == "slip"
    assert cli_args["pressure_solver"] == "pcg_diag"
    assert cli_args["max_pressure_iterations"] == 1000
    assert cli_args["pressure_relative_tolerance"] == 1e-4
    assert cli_args["startup_warning_steps"] == 10
    assert cli_args["convective_stabilization_mode"] == "auto_damping"
    assert cli_args["disable_convective_auto_damping"] is True
    assert cli_args["viscous_face_flux_divergence_impact_cap"] == 0.03
    assert cli_args["viscous_predictor_mode"] == ""
    assert cli_args["viscous_predictor_outlet_contract_mode"] == "auto"
    assert cli_args["pressure_projection_outlet_contract_mode"] == "auto"
    assert cli_args["projection_cell_velocity_update_mode"] == "auto"
    assert cli_args["pressure_nonorthogonal_correction_mode"] == "auto"
    assert cli_args["viscous_nonorthogonal_correction_mode"] == "auto"
    assert cli_args["pressure_nonorthogonal_correction_sweeps"] == 4
    assert cli_args["pressure_nonorthogonal_correction_relaxation"] == 1.0

    assert flow_summary["flow_mode"] == "navier_stokes_projection_debug"
    assert flow_summary["flow_dt_mode"] == "auto_cfl"
    assert flow_summary["flow_dt_min"] == 1e-7
    assert flow_summary["convective_cfl_target"] == 0.5
    assert flow_summary["startup_warning_steps_allowed"] == 10
    assert (
        flow_summary["flow_config"]["pressure_projection_outlet_contract_mode"]
        == "auto"
    )
    assert flow_summary["flow_config"]["projection_cell_velocity_update_mode"] == "auto"
    assert (
        flow_summary["flow_config"]["pressure_nonorthogonal_correction_mode"] == "auto"
    )
    assert (
        flow_summary["flow_config"]["viscous_nonorthogonal_correction_mode"] == "auto"
    )
    assert flow_summary["flow_config"]["pressure_nonorthogonal_correction_sweeps"] == 4
    assert (
        flow_summary["flow_config"]["pressure_nonorthogonal_correction_relaxation"]
        == 1.0
    )
    assert flow_summary["numerical_profile_resolution"]["effective"] == {
        "viscous_predictor_outlet_contract_mode": "match_inlet",
        "pressure_projection_outlet_contract_mode": "match_inlet",
        "projection_cell_velocity_update_mode": "legacy_reconstruct",
        "pressure_nonorthogonal_correction_mode": "none",
        "viscous_nonorthogonal_correction_mode": "none",
    }


def test_manifest_pipeline_runner_flow_overrides_reach_flow_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_stage2_overrides"
    manifest_path = run_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--mesh-name",
            "t_junction",
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "navier_stokes_projection_debug",
            "--flow-steps",
            "1",
            "--flow-wall-velocity-boundary-mode",
            "no_slip",
            "--flow-dt-mode",
            "manual",
            "--flow-dt",
            "2e-5",
            "--flow-dt-min",
            "2e-7",
            "--flow-dt-max",
            "4e-5",
            "--flow-convective-cfl-target",
            "0.25",
            "--flow-convective-stabilization-mode",
            "substepping",
            "--no-flow-disable-convective-auto-damping",
            "--flow-max-pressure-iterations",
            "77",
            "--flow-pressure-relative-tolerance",
            "0.002",
            "--flow-pressure-solver",
            "cg",
            "--flow-startup-warning-steps",
            "4",
            "--flow-viscous-face-flux-divergence-impact-cap",
            "0.02",
            "--flow-pressure-projection-outlet-contract-mode",
            "preserve",
            "--flow-projection-cell-velocity-update-mode",
            "momentum_pressure_corrected",
            "--flow-pressure-nonorthogonal-correction-mode",
            "auto",
            "--flow-viscous-nonorthogonal-correction-mode",
            "deferred_lsq",
            "--flow-pressure-nonorthogonal-correction-sweeps",
            "3",
            "--flow-pressure-nonorthogonal-correction-relaxation",
            "0.6",
            "--skip-transport",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    flow_run = next(
        run for run in manifest["runs"].values() if run["stage_type"] == "flow"
    )
    flow_config = json.loads(
        (run_root / flow_run["outputs"]["config_json"]).read_text(encoding="utf-8")
    )
    flow_summary = json.loads(
        (run_root / flow_run["outputs"]["summary_json"]).read_text(encoding="utf-8")
    )
    cli_args = flow_config["cli_args"]

    assert cli_args["flow_mode"] == "navier_stokes_projection_debug"
    assert cli_args["wall_velocity_boundary_mode"] == "no_slip"
    assert cli_args["flow_dt_mode"] == "manual"
    assert cli_args["flow_dt"] == 2e-5
    assert cli_args["flow_dt_min"] == 2e-7
    assert cli_args["flow_dt_max"] == 4e-5
    assert cli_args["convective_cfl_target"] == 0.25
    assert cli_args["convective_stabilization_mode"] == "substepping"
    assert cli_args["disable_convective_auto_damping"] is False
    assert cli_args["max_pressure_iterations"] == 77
    assert cli_args["pressure_relative_tolerance"] == 0.002
    assert cli_args["pressure_solver"] == "cg"
    assert cli_args["startup_warning_steps"] == 4
    assert cli_args["viscous_face_flux_divergence_impact_cap"] == 0.02
    assert cli_args["pressure_projection_outlet_contract_mode"] == "preserve"
    assert (
        cli_args["projection_cell_velocity_update_mode"]
        == "momentum_pressure_corrected"
    )
    assert cli_args["pressure_nonorthogonal_correction_mode"] == "auto"
    assert cli_args["viscous_nonorthogonal_correction_mode"] == "deferred_lsq"
    assert cli_args["pressure_nonorthogonal_correction_sweeps"] == 3
    assert cli_args["pressure_nonorthogonal_correction_relaxation"] == 0.6

    assert flow_summary["flow_mode"] == "navier_stokes_projection_debug"
    assert flow_summary["flow_dt_mode"] == "manual"
    assert flow_summary["flow_dt_min"] == 2e-7
    assert flow_summary["flow_dt_max"] == 4e-5
    assert flow_summary["convective_cfl_target"] == 0.25
    assert flow_summary["startup_warning_steps_allowed"] == 4
    assert (
        flow_summary["flow_config"]["pressure_projection_outlet_contract_mode"]
        == "preserve"
    )
    assert (
        flow_summary["flow_config"]["projection_cell_velocity_update_mode"]
        == "momentum_pressure_corrected"
    )
    assert (
        flow_summary["flow_config"]["pressure_nonorthogonal_correction_mode"] == "auto"
    )
    assert (
        flow_summary["flow_config"]["viscous_nonorthogonal_correction_mode"]
        == "deferred_lsq"
    )
    assert flow_summary["flow_config"]["pressure_nonorthogonal_correction_sweeps"] == 3
    assert (
        flow_summary["flow_config"]["pressure_nonorthogonal_correction_relaxation"]
        == 0.6
    )
    assert (
        flow_summary["flow_config"]["effective_numerical_profile"][
            "pressure_nonorthogonal_correction_mode"
        ]
        == "deferred_lsq"
    )
    projection_equation_residual = json.loads(
        Path(flow_summary["artifacts"]["projection_equation_residual_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert projection_equation_residual["numerical_profile"] == {
        "pressure_nonorthogonal_correction_mode_requested": "auto",
        "pressure_nonorthogonal_correction_mode_effective": "deferred_lsq",
        "pressure_nonorthogonal_correction_enabled": True,
    }
    flow_progression = json.loads(
        Path(flow_summary["artifacts"]["flow_progression_history_json"]).read_text(
            encoding="utf-8"
        )
    )
    assert flow_progression["steps"][0]["convective_predictor_used"] is True
    assert (
        flow_progression["steps"][0]["convective_predictor_outlet_contract_mode"]
        == "preserve"
    )


def test_manifest_pipeline_runner_requires_explicit_msh_for_fresh_runs(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_missing_mesh"
    completed = _run_raw(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--skip-transport",
        ]
    )

    assert completed.returncode != 0
    assert "Fresh manifest pipeline runs require --msh" in completed.stderr
    assert not (run_root / "pipeline_manifest.json").exists()


def test_manifest_pipeline_runner_can_reuse_existing_flow_run(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_reuse"
    manifest_path = run_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--mesh-name",
            "t_junction",
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
            "--skip-transport",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    flow_run = next(
        run for run in manifest["runs"].values() if run["stage_type"] == "flow"
    )
    flow_run_id = str(flow_run["run_id"])
    flow_summary = json.loads(
        (run_root / flow_run["outputs"]["summary_json"]).read_text(encoding="utf-8")
    )
    assert flow_summary["resolved_mesh_npz"]
    flow_outputs = dict(flow_run.get("outputs", {}))
    flow_outputs.pop("mesh_npz", None)
    manifest["runs"][flow_run_id]["outputs"] = flow_outputs
    save_pipeline_manifest(manifest_path, manifest)

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--from-flow-run-id",
            flow_run_id,
            "--allow-unready-flow-for-downstream",
            "--transport-backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--transport-steps",
            "1",
            "--transport-dt-mode",
            "manual",
            "--transport-dt",
            "1e-4",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    runs = manifest["runs"]
    assert len(runs) == 3
    transport_run = next(
        run for run in runs.values() if run.get("stage_type") == "transport"
    )
    assert transport_run["parent_run_ids"] == [flow_run_id]
    assert transport_run["status"] == "completed"


def test_manifest_pipeline_runner_can_reuse_existing_import_run(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_import_reuse"
    manifest_path = run_root / "pipeline_manifest.json"
    import_root = run_root / "import_runs"

    _run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            "--msh",
            "t_junction.msh",
            "--output-root",
            str(import_root),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "import-001",
        ]
    )

    _run(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--from-import-run-id",
            "import-001",
            "--flow-backend",
            "numpy",
            "--flow-steps",
            "1",
            "--skip-transport",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    runs = manifest["runs"]
    assert len(runs) == 2

    import_run = get_pipeline_run(manifest, "import-001")
    flow_run = next(run for run in runs.values() if run.get("stage_type") == "flow")
    assert import_run["status"] == "completed"
    assert flow_run["status"] == "completed"
    assert flow_run["parent_run_ids"] == ["import-001"]
    assert (
        dict(flow_run.get("outputs", {}))["mesh_npz"]
        == dict(import_run.get("outputs", {}))["mesh_npz"]
    )


def test_manifest_pipeline_runner_can_gate_unready_flow_downstream(tmp_path: Path) -> None:
    run_root = tmp_path / "results" / "manifest_pipeline_runner_gate"
    manifest_path = run_root / "pipeline_manifest.json"

    completed = _run_raw(
        [
            sys.executable,
            str(PIPELINE_RUNNER_SCRIPT),
            "--run-root",
            str(run_root),
            "--msh",
            "t_junction.msh",
            "--mesh-name",
            "t_junction",
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "projection_only",
            "--flow-steps",
            "1",
            "--transport-backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--transport-steps",
            "1",
            "--transport-dt-mode",
            "manual",
            "--transport-dt",
            "1e-4",
        ]
    )
    assert completed.returncode != 0
    assert "not ready for downstream stages" in completed.stderr

    manifest = load_pipeline_manifest(manifest_path)
    stage_types = {run["stage_type"] for run in manifest["runs"].values()}
    assert stage_types == {"import", "flow"}
