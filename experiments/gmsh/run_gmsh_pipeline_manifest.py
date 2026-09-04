"""Run the supported manifest-first Gmsh import->flow->transport->thermal pipeline."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from microfluidics.pipeline import (  # noqa: E402
    create_pipeline_manifest_file,
    get_pipeline_run,
    load_pipeline_manifest,
    resolve_manifest_path_reference,
)
from experiments.gmsh._path_utils import option_was_explicitly_provided  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _run_stage(cmd: list[str]) -> None:
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Pipeline stage failed:\n"
            + " ".join(cmd)
            + "\n--- stdout ---\n"
            + completed.stdout
            + "\n--- stderr ---\n"
            + completed.stderr
        )


def _resolve_output_path(
    manifest_path: Path,
    *,
    run_id: str,
    output_key: str,
) -> Path:
    manifest = load_pipeline_manifest(manifest_path)
    run = get_pipeline_run(manifest, run_id)
    if str(run.get("status", "")) != "completed":
        raise RuntimeError(
            f"Manifest run {run_id!r} is not completed; status={run.get('status')!r}."
        )
    outputs = dict(run.get("outputs", {}))
    raw_value = str(outputs.get(output_key, "")).strip()
    if not raw_value:
        raise RuntimeError(
            f"Manifest run {run_id!r} does not provide output {output_key!r}."
        )
    return resolve_manifest_path_reference(manifest_path, raw_value)


def _resolve_completed_stage_run(
    manifest_path: Path,
    *,
    run_id: str,
    stage_type: str,
) -> dict[str, object]:
    manifest = load_pipeline_manifest(manifest_path)
    run = get_pipeline_run(manifest, run_id)
    if str(run.get("status", "")) != "completed":
        raise RuntimeError(
            f"{stage_type.capitalize()} run {run_id!r} is not completed; "
            f"status={run.get('status')!r}."
        )
    if str(run.get("stage_type", "")) != stage_type:
        raise RuntimeError(
            f"Run {run_id!r} is not a {stage_type} stage in the pipeline manifest."
        )
    return run


def _resolve_run_dir(manifest_path: Path, *, run_id: str) -> Path:
    manifest = load_pipeline_manifest(manifest_path)
    run = get_pipeline_run(manifest, run_id)
    if str(run.get("status", "")) != "completed":
        raise RuntimeError(
            f"Manifest run {run_id!r} is not completed; status={run.get('status')!r}."
        )
    raw_value = str(run.get("run_dir", "")).strip()
    if not raw_value:
        raise RuntimeError(f"Manifest run {run_id!r} does not define run_dir.")
    return resolve_manifest_path_reference(manifest_path, raw_value)


def _read_run_summary(
    manifest_path: Path,
    *,
    run_id: str,
) -> dict[str, object]:
    summary_path = _resolve_output_path(
        manifest_path,
        run_id=run_id,
        output_key="summary_json",
    )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_flow_stage_inputs(
    manifest_path: Path,
    *,
    flow_run_id: str,
) -> dict[str, Path]:
    flow_run = _resolve_completed_stage_run(
        manifest_path,
        run_id=flow_run_id,
        stage_type="flow",
    )
    outputs = dict(flow_run.get("outputs", {}))
    resolved_mesh_npz = str(outputs.get("mesh_npz", "")).strip()
    if not resolved_mesh_npz:
        summary = _read_run_summary(manifest_path, run_id=flow_run_id)
        resolved_mesh_npz = str(summary.get("resolved_mesh_npz", "")).strip()
    if not resolved_mesh_npz:
        raise RuntimeError(
            f"Flow run {flow_run_id!r} does not expose output 'mesh_npz' and its "
            "summary does not define 'resolved_mesh_npz'."
        )
    return {
        "mesh_npz": resolve_manifest_path_reference(manifest_path, resolved_mesh_npz),
        "run_dir": _resolve_run_dir(manifest_path, run_id=flow_run_id),
        "summary_json": _resolve_output_path(
            manifest_path,
            run_id=flow_run_id,
            output_key="summary_json",
        ),
        "final_corrected_face_flux_npy": _resolve_output_path(
            manifest_path,
            run_id=flow_run_id,
            output_key="final_corrected_face_flux_npy",
        ),
    }


def _find_ancestor_run_by_stage_type(
    manifest_path: Path,
    *,
    start_run_id: str,
    stage_type: str,
) -> dict[str, object] | None:
    manifest = load_pipeline_manifest(manifest_path)
    seen: set[str] = set()
    queue: deque[str] = deque([start_run_id])
    while queue:
        run_id = queue.popleft()
        if run_id in seen:
            continue
        seen.add(run_id)
        run = get_pipeline_run(manifest, run_id)
        if str(run.get("stage_type", "")) == stage_type:
            return run
        for parent_run_id in run.get("parent_run_ids", []):
            queue.append(str(parent_run_id))
    return None


def _infer_msh_argument_from_import_run(
    manifest_path: Path,
    *,
    import_run_id: str,
) -> str:
    import_run = _resolve_completed_stage_run(
        manifest_path,
        run_id=import_run_id,
        stage_type="import",
    )
    inputs = dict(import_run.get("inputs", {}))
    resolved_msh_path = str(inputs.get("resolved_msh_path", "")).strip()
    if resolved_msh_path:
        return str(resolve_manifest_path_reference(manifest_path, resolved_msh_path))
    original_input = str(inputs.get("original_input", "")).strip()
    if original_input:
        return original_input
    raise RuntimeError(
        f"Import run {import_run_id!r} does not expose a mesh source path."
    )


def _infer_msh_argument_from_flow_run(
    manifest_path: Path,
    *,
    flow_run_id: str,
) -> str:
    import_run = _find_ancestor_run_by_stage_type(
        manifest_path,
        start_run_id=flow_run_id,
        stage_type="import",
    )
    if import_run is None:
        raise RuntimeError(
            f"Could not infer mesh path for flow run {flow_run_id!r}: "
            "no import ancestor found."
        )
    return _infer_msh_argument_from_import_run(
        manifest_path,
        import_run_id=str(import_run.get("run_id", "")),
    )


def _resolve_import_stage_inputs(
    manifest_path: Path,
    *,
    import_run_id: str,
) -> dict[str, object]:
    import_run = _resolve_completed_stage_run(
        manifest_path,
        run_id=import_run_id,
        stage_type="import",
    )
    outputs = dict(import_run.get("outputs", {}))
    resolved_mesh_npz = str(outputs.get("mesh_npz", "")).strip()
    if not resolved_mesh_npz:
        raise RuntimeError(
            f"Import run {import_run_id!r} does not expose output 'mesh_npz'."
        )
    return {
        "mesh_npz": resolve_manifest_path_reference(manifest_path, resolved_mesh_npz),
        "msh": _infer_msh_argument_from_import_run(
            manifest_path,
            import_run_id=import_run_id,
        ),
    }


def _flow_ready_for_downstream(
    manifest_path: Path,
    *,
    flow_run_id: str,
) -> tuple[bool, str]:
    manifest = load_pipeline_manifest(manifest_path)
    flow_run = get_pipeline_run(manifest, flow_run_id)
    if str(flow_run.get("status", "")) != "completed":
        return False, f"flow status is {flow_run.get('status')!r}"
    metadata = dict(flow_run.get("metadata", {}))
    ready = metadata.get("ready_for_next_stage", False)
    reason = str(metadata.get("stage_status_reason", "")).strip()
    return bool(ready), reason


def main() -> None:
    raw_argv = list(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "gmsh_manifest_pipeline_runs"),
        help="Root directory for import/flow/transport/thermal/reactive run folders.",
    )
    parser.add_argument(
        "--pipeline-id",
        type=str,
        default="",
        help="Optional logical pipeline id. Defaults to the run-root folder name.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch child stage scripts.",
    )
    parser.add_argument(
        "--msh",
        type=str,
        default="",
        help=(
            "Mesh path or mesh filename for a fresh manifest pipeline run. "
            "Required unless --from-import-run-id or --from-flow-run-id is used."
        ),
    )
    parser.add_argument(
        "--case-config",
        type=str,
        default="",
        help=(
            "Optional case_config_v1 JSON. For a fresh run it may provide mesh.path "
            "instead of --msh and is embedded into the imported mesh artifact."
        ),
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="",
        help=(
            "Retained for wrapper compatibility and metadata. Exact stage handoff "
            "comes from the manifest."
        ),
    )
    parser.add_argument(
        "--flow-mode",
        type=str,
        default="navier_stokes_projection_debug",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=700,
    )
    parser.add_argument(
        "--flow-inlet-speed",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--flow-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--flow-device",
        type=str,
        default="",
        help="Optional device passed to the flow stage, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--resume-flow-run-dir",
        type=str,
        default="",
        help="Optional previous flow run directory passed to the flow stage.",
    )
    parser.add_argument(
        "--flow-wall-velocity-boundary-mode",
        type=str,
        choices=("slip", "no_slip", "no_slip_tangential", "no_slip_legacy_isotropic"),
        default="slip",
    )
    parser.add_argument(
        "--flow-wall-tangential-no-slip-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--flow-wall-tangential-no-slip-strength-ramp-start",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flow-wall-tangential-no-slip-strength-ramp-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--flow-wall-tangential-shear-face-flux-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flow-wall-tangential-cell-velocity-momentum-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--flow-wall-flux-stokes-resistance-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--flow-wall-flux-stokes-resistance-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--flow-dt-mode",
        type=str,
        choices=("manual", "auto_cfl"),
        default="auto_cfl",
    )
    parser.add_argument("--flow-dt", type=float, default=None)
    parser.add_argument("--flow-dt-min", type=float, default=1e-7)
    parser.add_argument("--flow-dt-max", type=float, default=None)
    parser.add_argument(
        "--flow-convective-cfl-target",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--flow-viscous-face-flux-divergence-impact-cap",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--flow-viscous-predictor-mode",
        type=str,
        choices=(
            "none",
            "no_viscous_debug_copy",
            "explicit_cell_velocity_laplacian_substepped",
            "explicit_cell_velocity_laplacian_substepped_conservative",
            "face_flux_laplacian_substepped",
        ),
        default="",
    )
    parser.add_argument(
        "--flow-viscous-predictor-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="",
    )
    parser.add_argument(
        "--flow-pressure-projection-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="",
    )
    parser.add_argument(
        "--flow-projection-cell-velocity-update-mode",
        type=str,
        choices=("auto", "legacy_reconstruct", "momentum_pressure_corrected"),
        default="",
    )
    parser.add_argument(
        "--flow-pressure-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="",
    )
    parser.add_argument(
        "--flow-viscous-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="",
    )
    parser.add_argument(
        "--flow-pressure-nonorthogonal-correction-sweeps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--flow-pressure-nonorthogonal-correction-relaxation",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--flow-convective-stabilization-mode",
        type=str,
        choices=("auto_damping", "substepping"),
        default="auto_damping",
    )
    parser.add_argument(
        "--flow-disable-convective-auto-damping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--flow-max-pressure-iterations", type=int, default=1000)
    parser.add_argument("--flow-pressure-relative-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--flow-pressure-solver",
        type=str,
        choices=("jacobi", "cg", "pcg_diag", "amg_pcg"),
        default="pcg_diag",
    )
    parser.add_argument("--flow-startup-warning-steps", type=int, default=10)
    parser.add_argument(
        "--transport-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--transport-end-time",
        type=float,
        default=None,
        help="Target physical transport time in seconds; mutually exclusive with --transport-steps.",
    )
    parser.add_argument(
        "--transport-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--transport-execution-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--transport-torch-device",
        type=str,
        default="",
        help="Optional torch device passed to transport when execution backend is torch.",
    )
    parser.add_argument(
        "--transport-dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    parser.add_argument("--transport-dt", type=float, default=1.5e-5)
    parser.add_argument("--transport-snapshot-every", type=int, default=5000)
    parser.add_argument("--transport-checkpoint-every", type=int, default=5000)
    parser.add_argument("--transport-progress-every", type=int, default=500)
    parser.add_argument(
        "--transport-mode",
        type=str,
        choices=("advection", "advection_diffusion"),
        default="advection_diffusion",
    )
    parser.add_argument(
        "--transport-scheme",
        type=str,
        choices=("upwind", "bounded_upwind"),
        default="bounded_upwind",
    )
    parser.add_argument("--transport-cfl-target", type=float, default=0.5)
    parser.add_argument("--transport-cfl-limit", type=float, default=0.8)
    parser.add_argument("--transport-diffusivity", type=float, default=3e-10)
    parser.add_argument("--transport-kinematic-viscosity", type=float, default=1e-6)
    parser.add_argument(
        "--transport-max-supported-grid-peclet", type=float, default=2.0
    )
    parser.add_argument("--transport-max-supported-schmidt", type=float, default=1000.0)
    parser.add_argument(
        "--transport-gradient-method",
        type=str,
        choices=("face", "least_squares"),
        default="least_squares",
    )
    parser.add_argument(
        "--transport-laplacian-method",
        type=str,
        choices=("tpfa", "lsq_flux"),
        default="tpfa",
    )
    parser.add_argument("--transport-left-inlet-value", type=float, default=0.0)
    parser.add_argument("--transport-right-inlet-value", type=float, default=1.0)
    parser.add_argument("--transport-inlet-speed", type=float, default=0.15)
    parser.add_argument(
        "--transport-no-velocity-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--transport-no-transport-scheme-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--transport-fail-if-numpy-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-transport",
        action="store_true",
        help="Skip transport and stop after flow unless thermal is requested.",
    )
    parser.add_argument(
        "--reactive-case",
        type=str,
        default="",
        help=(
            "Run reactive_transport directly after flow using this exact case JSON. "
            "Ordinary transport and thermal are mutually exclusive with this mode."
        ),
    )
    parser.add_argument(
        "--reactive-max-walltime-seconds",
        type=_nonnegative_finite_float,
        default=0.0,
        help=(
            "Optional soft walltime budget passed to reactive_transport; zero "
            "disables the limit."
        ),
    )
    parser.add_argument(
        "--from-flow-run-id",
        type=str,
        default="",
        help=(
            "Reuse an existing flow run from the manifest and launch downstream "
            "stages from it without rerunning import/flow."
        ),
    )
    parser.add_argument(
        "--from-import-run-id",
        type=str,
        default="",
        help=(
            "Reuse an existing import run from the manifest and launch a new flow "
            "stage from its exact imported mesh without rerunning import."
        ),
    )
    parser.add_argument(
        "--require-ready-flow-for-downstream",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Readiness is now enforced by default "
            "unless --allow-unready-flow-for-downstream is set."
        ),
    )
    parser.add_argument(
        "--allow-unready-flow-for-downstream",
        action="store_true",
        help=(
            "Unsafe debug override: allow downstream stages to launch from a flow "
            "run that is not ready_for_next_stage."
        ),
    )
    parser.add_argument(
        "--run-thermal",
        action="store_true",
        help="Run the thermal stage after transport.",
    )
    parser.add_argument("--thermal-steps", type=int, default=20000)
    parser.add_argument(
        "--thermal-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--thermal-torch-device",
        type=str,
        default="",
        help="Optional torch device passed to the thermal stage.",
    )
    parser.add_argument(
        "--thermal-dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    parser.add_argument("--thermal-dt", type=float, default=5e-4)
    parser.add_argument("--thermal-cfl-target", type=float, default=0.5)
    parser.add_argument("--thermal-cfl-limit", type=float, default=0.8)
    parser.add_argument("--thermal-heat-source", type=float, default=0.0)
    args = parser.parse_args()
    if args.transport_steps is not None and args.transport_end_time is not None:
        parser.error(
            "--transport-steps and --transport-end-time are mutually exclusive."
        )
    if args.transport_end_time is not None and args.transport_end_time <= 0.0:
        parser.error("--transport-end-time must be positive.")
    if args.transport_steps is None and args.transport_end_time is None:
        args.transport_steps = 20000
    msh_explicitly_provided = option_was_explicitly_provided(raw_argv, "--msh")
    reactive_case = str(args.reactive_case).strip()
    explicitly_requested_transport = any(
        item.startswith("--transport-") for item in raw_argv
    )
    if reactive_case and (bool(args.run_thermal) or explicitly_requested_transport):
        parser.error(
            "--reactive-case is mutually exclusive with ordinary transport and thermal options."
        )

    from_flow_run_id = str(args.from_flow_run_id).strip()
    from_import_run_id = str(args.from_import_run_id).strip()
    if (
        (not from_flow_run_id)
        and (not from_import_run_id)
        and (not str(args.msh).strip())
        and (not str(args.case_config).strip())
    ):
        parser.error(
            "Fresh manifest pipeline runs require --msh or --case-config. Pass a "
            "mesh/case path or reuse an existing manifest run via "
            "--from-import-run-id or --from-flow-run-id."
        )

    run_root = Path(args.run_root).resolve()
    import_root = run_root / "import_runs"
    flow_root = run_root / "flow_runs"
    transport_root = run_root / "transport_runs"
    thermal_root = run_root / "thermal_runs"
    reactive_root = run_root / "reactive_transport_runs"
    for root in (import_root, flow_root, transport_root, thermal_root, reactive_root):
        root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "pipeline_manifest.json"
    pipeline_id = str(args.pipeline_id).strip() or run_root.name
    if not manifest_path.exists():
        create_pipeline_manifest_file(
            manifest_path,
            pipeline_id=pipeline_id,
            pipeline_root=run_root,
            metadata={"pipeline_runner": "run_gmsh_pipeline_manifest.py"},
        )

    if from_flow_run_id and from_import_run_id:
        parser.error(
            "--from-flow-run-id and --from-import-run-id are mutually exclusive."
        )
    import_run_id = f"import-{_utc_stamp()}"
    flow_run_id = f"flow-{_utc_stamp()}"
    transport_run_id = f"transport-{_utc_stamp()}"
    thermal_run_id = f"thermal-{_utc_stamp()}"
    reactive_run_id = f"reactive-transport-{_utc_stamp()}"

    if from_flow_run_id:
        flow_run_id = from_flow_run_id
        flow_inputs = _resolve_flow_stage_inputs(
            manifest_path,
            flow_run_id=flow_run_id,
        )
        mesh_npz = flow_inputs["mesh_npz"]
        flow_run_dir = flow_inputs["run_dir"]
        flow_summary_json = flow_inputs["summary_json"]
    else:
        if from_import_run_id:
            import_run_id = from_import_run_id
            import_inputs = _resolve_import_stage_inputs(
                manifest_path,
                import_run_id=import_run_id,
            )
            mesh_npz = Path(str(import_inputs["mesh_npz"]))
        else:
            import_cmd = [
                str(Path(args.python_exe)),
                str(PROJECT_ROOT / "experiments" / "gmsh" / "run_import_gmsh_mesh.py"),
                "--output-root",
                str(import_root),
                "--pipeline-manifest",
                str(manifest_path),
                "--pipeline-run-id",
                import_run_id,
            ]
            if str(args.msh).strip():
                import_cmd.extend(["--msh", str(args.msh)])
            if str(args.case_config).strip():
                import_cmd.extend(["--case-config", str(args.case_config)])
            _run_stage(import_cmd)
            mesh_npz = _resolve_output_path(
                manifest_path,
                run_id=import_run_id,
                output_key="mesh_npz",
            )

        flow_cmd = [
            str(Path(args.python_exe)),
            str(PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_flow_debug.py"),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(flow_root),
            "--inlet-speed",
            str(args.flow_inlet_speed),
            "--flow-steps",
            str(args.flow_steps),
            "--flow-mode",
            str(args.flow_mode),
            "--wall-velocity-boundary-mode",
            str(args.flow_wall_velocity_boundary_mode),
            "--wall-tangential-no-slip-strength",
            str(args.flow_wall_tangential_no_slip_strength),
            "--flow-dt-mode",
            str(args.flow_dt_mode),
            "--flow-dt-min",
            str(args.flow_dt_min),
            "--convective-cfl-target",
            str(args.flow_convective_cfl_target),
            "--startup-warning-steps",
            str(args.flow_startup_warning_steps),
            "--backend",
            str(args.flow_backend),
            "--flow-execution-backend",
            str(args.flow_backend),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            flow_run_id,
            "--pipeline-parent-run-id",
            import_run_id,
        ]
        if str(args.case_config).strip():
            flow_cmd.extend(["--case-config", str(args.case_config)])
        if args.flow_wall_tangential_no_slip_strength_ramp_start is not None:
            flow_cmd.extend(
                [
                    "--wall-tangential-no-slip-strength-ramp-start",
                    str(args.flow_wall_tangential_no_slip_strength_ramp_start),
                ]
            )
        if int(args.flow_wall_tangential_no_slip_strength_ramp_steps) > 0:
            flow_cmd.extend(
                [
                    "--wall-tangential-no-slip-strength-ramp-steps",
                    str(args.flow_wall_tangential_no_slip_strength_ramp_steps),
                ]
            )
        if str(args.flow_device).strip():
            flow_cmd.extend(["--device", str(args.flow_device).strip()])
        if str(args.resume_flow_run_dir).strip():
            flow_cmd.extend(
                ["--resume-flow-run-dir", str(args.resume_flow_run_dir).strip()]
            )
        if args.flow_dt is not None:
            flow_cmd.extend(["--flow-dt", str(args.flow_dt)])
        if args.flow_dt_max is not None:
            flow_cmd.extend(["--flow-dt-max", str(args.flow_dt_max)])
        if args.flow_viscous_face_flux_divergence_impact_cap is not None:
            flow_cmd.extend(
                [
                    "--viscous-face-flux-divergence-impact-cap",
                    str(args.flow_viscous_face_flux_divergence_impact_cap),
                ]
            )
        if str(args.flow_viscous_predictor_mode).strip():
            flow_cmd.extend(
                [
                    "--viscous-predictor-mode",
                    str(args.flow_viscous_predictor_mode).strip(),
                ]
            )
        if str(args.flow_viscous_predictor_outlet_contract_mode).strip():
            flow_cmd.extend(
                [
                    "--viscous-predictor-outlet-contract-mode",
                    str(args.flow_viscous_predictor_outlet_contract_mode).strip(),
                ]
            )
        if str(args.flow_pressure_projection_outlet_contract_mode).strip():
            flow_cmd.extend(
                [
                    "--pressure-projection-outlet-contract-mode",
                    str(args.flow_pressure_projection_outlet_contract_mode).strip(),
                ]
            )
        if str(args.flow_projection_cell_velocity_update_mode).strip():
            flow_cmd.extend(
                [
                    "--projection-cell-velocity-update-mode",
                    str(args.flow_projection_cell_velocity_update_mode).strip(),
                ]
            )
        if str(args.flow_pressure_nonorthogonal_correction_mode).strip():
            flow_cmd.extend(
                [
                    "--pressure-nonorthogonal-correction-mode",
                    str(args.flow_pressure_nonorthogonal_correction_mode).strip(),
                ]
            )
        if str(args.flow_viscous_nonorthogonal_correction_mode).strip():
            flow_cmd.extend(
                [
                    "--viscous-nonorthogonal-correction-mode",
                    str(args.flow_viscous_nonorthogonal_correction_mode).strip(),
                ]
            )
        if args.flow_pressure_nonorthogonal_correction_sweeps is not None:
            flow_cmd.extend(
                [
                    "--pressure-nonorthogonal-correction-sweeps",
                    str(args.flow_pressure_nonorthogonal_correction_sweeps),
                ]
            )
        if args.flow_pressure_nonorthogonal_correction_relaxation is not None:
            flow_cmd.extend(
                [
                    "--pressure-nonorthogonal-correction-relaxation",
                    str(args.flow_pressure_nonorthogonal_correction_relaxation),
                ]
            )
        if str(args.flow_convective_stabilization_mode).strip():
            flow_cmd.extend(
                [
                    "--convective-stabilization-mode",
                    str(args.flow_convective_stabilization_mode).strip(),
                ]
            )
        if args.flow_disable_convective_auto_damping is not None:
            if bool(args.flow_disable_convective_auto_damping):
                flow_cmd.append("--disable-convective-auto-damping")
            else:
                flow_cmd.append("--no-disable-convective-auto-damping")
        if args.flow_max_pressure_iterations is not None:
            flow_cmd.extend(
                ["--max-pressure-iterations", str(args.flow_max_pressure_iterations)]
            )
        if args.flow_pressure_relative_tolerance is not None:
            flow_cmd.extend(
                [
                    "--pressure-relative-tolerance",
                    str(args.flow_pressure_relative_tolerance),
                ]
            )
        if str(args.flow_pressure_solver).strip():
            flow_cmd.extend(
                ["--pressure-solver", str(args.flow_pressure_solver).strip()]
            )
        if bool(args.flow_wall_tangential_shear_face_flux_enabled):
            flow_cmd.append("--wall-tangential-shear-face-flux-enabled")
        else:
            flow_cmd.append("--no-wall-tangential-shear-face-flux-enabled")
        if bool(args.flow_wall_tangential_cell_velocity_momentum_enabled):
            flow_cmd.append("--wall-tangential-cell-velocity-momentum-enabled")
        else:
            flow_cmd.append("--no-wall-tangential-cell-velocity-momentum-enabled")
        if args.flow_wall_flux_stokes_resistance_enabled is not None:
            if bool(args.flow_wall_flux_stokes_resistance_enabled):
                flow_cmd.append("--wall-flux-stokes-resistance-enabled")
            else:
                flow_cmd.append("--no-wall-flux-stokes-resistance-enabled")
        if args.flow_disable_convective_auto_damping is not None:
            if bool(args.flow_disable_convective_auto_damping):
                flow_cmd.append("--disable-convective-auto-damping")
            else:
                flow_cmd.append("--no-disable-convective-auto-damping")
        flow_cmd.extend(
            [
                "--wall-flux-stokes-resistance-strength",
                str(args.flow_wall_flux_stokes_resistance_strength),
            ]
        )
        _run_stage(flow_cmd)
        flow_run_dir = _resolve_run_dir(manifest_path, run_id=flow_run_id)
        flow_summary_json = _resolve_output_path(
            manifest_path,
            run_id=flow_run_id,
            output_key="summary_json",
        )

    wants_downstream = (
        bool(reactive_case) or (not bool(args.skip_transport)) or bool(args.run_thermal)
    )
    if (not bool(args.allow_unready_flow_for_downstream)) and wants_downstream:
        flow_ready, ready_reason = _flow_ready_for_downstream(
            manifest_path,
            flow_run_id=flow_run_id,
        )
        if not flow_ready:
            suffix = f" Reason: {ready_reason}" if ready_reason else ""
            raise RuntimeError(
                f"Flow run {flow_run_id!r} is not ready for downstream stages.{suffix}"
            )

    if reactive_case:
        reactive_cmd = [
            str(Path(args.python_exe)),
            str(
                PROJECT_ROOT
                / "experiments"
                / "gmsh"
                / "run_gmsh_tetra_reactive_debug.py"
            ),
            "--mesh-npz",
            str(mesh_npz),
            "--flow-run-dir",
            str(flow_run_dir),
            "--reactive-case",
            reactive_case,
            "--output-root",
            str(reactive_root),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            reactive_run_id,
            "--pipeline-parent-run-id",
            flow_run_id,
        ]
        if bool(args.allow_unready_flow_for_downstream):
            reactive_cmd.extend(
                [
                    "--allow-unready-flow",
                    "--allow-physically-unready-result",
                ]
            )
        if float(args.reactive_max_walltime_seconds) > 0.0:
            reactive_cmd.extend(
                [
                    "--max-walltime-seconds",
                    str(args.reactive_max_walltime_seconds),
                ]
            )
        _run_stage(reactive_cmd)
    elif not bool(args.skip_transport):
        transport_cmd = [
            str(Path(args.python_exe)),
            str(
                PROJECT_ROOT
                / "experiments"
                / "gmsh"
                / "run_gmsh_tetra_transport_debug.py"
            ),
            "--mesh-npz",
            str(mesh_npz),
            "--import-root",
            str(import_root),
            "--output-root",
            str(transport_root),
            "--backend",
            str(args.transport_backend),
            "--transport-execution-backend",
            str(args.transport_execution_backend),
            "--velocity-source",
            "flow_run",
            "--flow-run-dir",
            str(flow_run_dir),
            "--transport-mode",
            str(args.transport_mode),
            "--transport-scheme",
            str(args.transport_scheme),
            "--snapshot-every",
            str(args.transport_snapshot_every),
            "--checkpoint-every",
            str(args.transport_checkpoint_every),
            "--progress-every",
            str(args.transport_progress_every),
            "--cfl-target",
            str(args.transport_cfl_target),
            "--cfl-limit",
            str(args.transport_cfl_limit),
            "--diffusivity",
            str(args.transport_diffusivity),
            "--kinematic-viscosity",
            str(args.transport_kinematic_viscosity),
            "--max-supported-grid-peclet",
            str(args.transport_max_supported_grid_peclet),
            "--max-supported-schmidt",
            str(args.transport_max_supported_schmidt),
            "--gradient-method",
            str(args.transport_gradient_method),
            "--laplacian-method",
            str(args.transport_laplacian_method),
            "--left-inlet-value",
            str(args.transport_left_inlet_value),
            "--right-inlet-value",
            str(args.transport_right_inlet_value),
            "--inlet-speed",
            str(args.transport_inlet_speed),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            transport_run_id,
            "--pipeline-parent-run-id",
            flow_run_id,
        ]
        if args.transport_end_time is None:
            transport_cmd.extend(
                [
                    "--steps",
                    str(args.transport_steps),
                    "--snapshot-steps",
                    str(args.transport_steps),
                ]
            )
        else:
            transport_cmd.extend(["--transport-end-time", str(args.transport_end_time)])
        if str(args.transport_torch_device).strip():
            transport_cmd.extend(
                ["--torch-device", str(args.transport_torch_device).strip()]
            )
        if str(args.transport_dt_mode) == "manual":
            transport_cmd.extend(
                ["--dt-mode", "manual", "--dt", str(args.transport_dt)]
            )
        else:
            transport_cmd.extend(["--dt-mode", "auto"])
        transport_requests_torch = str(args.transport_execution_backend) == "torch" or (
            str(args.transport_backend) == "torch"
        )
        if bool(args.transport_no_velocity_comparison):
            transport_cmd.append("--no-velocity-comparison")
        if bool(args.transport_no_transport_scheme_comparison):
            transport_cmd.append("--no-transport-scheme-comparison")
        if bool(args.transport_fail_if_numpy_fallback) and transport_requests_torch:
            transport_cmd.append("--fail-if-numpy-fallback")
        if bool(args.allow_unready_flow_for_downstream):
            transport_cmd.append("--allow-unready-flow")
        if str(args.case_config).strip():
            transport_cmd.extend(["--case-config", str(args.case_config)])
        _run_stage(transport_cmd)

    if args.run_thermal:
        if msh_explicitly_provided:
            thermal_msh_value = str(args.msh)
        elif from_flow_run_id:
            thermal_msh_value = _infer_msh_argument_from_flow_run(
                manifest_path,
                flow_run_id=flow_run_id,
            )
        elif from_import_run_id:
            thermal_msh_value = _infer_msh_argument_from_import_run(
                manifest_path,
                import_run_id=import_run_id,
            )
        else:
            thermal_msh_value = str(args.msh)
        thermal_cmd = [
            str(Path(args.python_exe)),
            str(
                PROJECT_ROOT
                / "experiments"
                / "gmsh"
                / "run_gmsh_tetra_thermal_debug.py"
            ),
            "--msh",
            thermal_msh_value,
            "--output-root",
            str(thermal_root),
            "--backend",
            str(args.thermal_backend),
            "--velocity-source",
            "flow_solver",
            "--flow-summary-json",
            str(flow_summary_json),
            "--steps",
            str(args.thermal_steps),
            "--cfl-target",
            str(args.thermal_cfl_target),
            "--cfl-limit",
            str(args.thermal_cfl_limit),
            "--heat-source",
            str(args.thermal_heat_source),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            thermal_run_id,
            "--pipeline-parent-run-id",
            flow_run_id,
        ]
        if str(args.case_config).strip():
            thermal_cmd.extend(["--case-config", str(args.case_config)])
        if str(args.thermal_torch_device).strip():
            thermal_cmd.extend(
                ["--torch-device", str(args.thermal_torch_device).strip()]
            )
        if str(args.thermal_dt_mode) == "manual":
            thermal_cmd.extend(["--dt-mode", "manual", "--dt", str(args.thermal_dt)])
        else:
            thermal_cmd.extend(["--dt-mode", "auto"])
        if bool(args.allow_unready_flow_for_downstream):
            thermal_cmd.append("--allow-unready-flow")
        _run_stage(thermal_cmd)


if __name__ == "__main__":
    main()
