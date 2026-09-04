"""Inspect manifest-first pipeline runs and launch supported stages from recorded manifest lineage."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.gmsh._flow_coupling import (  # noqa: E402
    load_flow_summary_payload,
    validate_source_flow_readiness,
)
from microfluidics.pipeline import (  # noqa: E402
    create_pipeline_manifest_file,
    get_pipeline_run,
    list_pipeline_runs,
    load_pipeline_manifest,
    resolve_manifest_path_reference,
    upsert_pipeline_run,
)

DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / "gmsh_manifest_pipeline_runs"
PIPELINE_RUNNER_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_pipeline_manifest.py"
).resolve()


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _normalize_manifest_path(
    *,
    run_root: str | Path,
    pipeline_manifest: str,
) -> Path:
    raw_manifest = str(pipeline_manifest).strip()
    if raw_manifest:
        return Path(raw_manifest).resolve()
    return (Path(run_root).resolve() / "pipeline_manifest.json").resolve()


def _resolve_cli_context(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = _normalize_manifest_path(
        run_root=getattr(args, "run_root", DEFAULT_RUN_ROOT),
        pipeline_manifest=str(getattr(args, "pipeline_manifest", "") or ""),
    )
    manifest = load_pipeline_manifest(manifest_path)
    run_root = manifest_path.parent.resolve()
    return manifest_path, run_root, manifest


def _resolve_registration_context(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any]]:
    manifest_path = _normalize_manifest_path(
        run_root=getattr(args, "run_root", DEFAULT_RUN_ROOT),
        pipeline_manifest=str(getattr(args, "pipeline_manifest", "") or ""),
    )
    run_root = manifest_path.parent.resolve()
    if not manifest_path.exists():
        pipeline_id = str(getattr(args, "pipeline_id", "")).strip() or run_root.name
        create_pipeline_manifest_file(
            manifest_path,
            pipeline_id=pipeline_id,
            pipeline_root=run_root,
            metadata={"pipeline_runner": "run_gmsh_pipeline_manifest_tool.py"},
        )
    manifest = load_pipeline_manifest(manifest_path)
    return manifest_path, run_root, manifest


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _extract_readiness(run: Mapping[str, Any]) -> dict[str, Any]:
    metadata = (
        dict(run.get("metadata", {}))
        if isinstance(run.get("metadata", {}), Mapping)
        else {}
    )
    return {
        "run_completed": _bool_or_none(metadata.get("run_completed")),
        "numerically_stable": _bool_or_none(metadata.get("numerically_stable")),
        "physically_ready": _bool_or_none(metadata.get("physically_ready")),
        "ready_for_next_stage": bool(metadata.get("ready_for_next_stage", False)),
        "ready_for_long_run": bool(metadata.get("ready_for_long_run", False)),
        "stage_status_reason": str(metadata.get("stage_status_reason", "")).strip(),
    }


def _mesh_name_for_run(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
) -> str:
    metadata = (
        dict(run.get("metadata", {}))
        if isinstance(run.get("metadata", {}), Mapping)
        else {}
    )
    mesh_name = str(metadata.get("mesh_name", "")).strip()
    if mesh_name:
        return mesh_name

    outputs = (
        dict(run.get("outputs", {}))
        if isinstance(run.get("outputs", {}), Mapping)
        else {}
    )
    mesh_npz = str(outputs.get("mesh_npz", "")).strip()
    if mesh_npz:
        stem = Path(mesh_npz).stem
        return stem.removesuffix("_imported_mesh")

    import_run = _find_ancestor_run_by_stage_type(
        manifest,
        start_run_id=str(run.get("run_id", "")),
        stage_type="import",
    )
    if import_run is None:
        return ""

    import_inputs = (
        dict(import_run.get("inputs", {}))
        if isinstance(import_run.get("inputs", {}), Mapping)
        else {}
    )
    original_input = str(import_inputs.get("original_input", "")).strip()
    if original_input:
        return Path(original_input).stem

    resolved_msh_path = str(import_inputs.get("resolved_msh_path", "")).strip()
    if not resolved_msh_path:
        return ""
    return resolve_manifest_path_reference(manifest_path, resolved_msh_path).stem


def _find_ancestor_run_by_stage_type(
    manifest: Mapping[str, Any],
    *,
    start_run_id: str,
    stage_type: str,
) -> dict[str, Any] | None:
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


def _infer_msh_argument(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    flow_run_id: str,
) -> str:
    import_run = _find_ancestor_run_by_stage_type(
        manifest,
        start_run_id=flow_run_id,
        stage_type="import",
    )
    if import_run is None:
        raise RuntimeError(
            f"Could not infer mesh path for flow run {flow_run_id!r}: no import ancestor found."
        )

    inputs = (
        dict(import_run.get("inputs", {}))
        if isinstance(import_run.get("inputs", {}), Mapping)
        else {}
    )
    resolved_msh_path = str(inputs.get("resolved_msh_path", "")).strip()
    if resolved_msh_path:
        return str(resolve_manifest_path_reference(manifest_path, resolved_msh_path))

    original_input = str(inputs.get("original_input", "")).strip()
    if original_input:
        return original_input

    raise RuntimeError(
        f"Could not infer mesh path for flow run {flow_run_id!r}: import inputs are incomplete."
    )


def _build_flow_rows(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    ready_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in list_pipeline_runs(manifest, stage_type="flow"):
        readiness = _extract_readiness(run)
        if ready_only and not readiness["ready_for_next_stage"]:
            continue
        rows.append(
            {
                "run_id": str(run.get("run_id", "")),
                "status": str(run.get("status", "")),
                "mesh": _mesh_name_for_run(manifest_path, manifest, run),
                "ready_for_next_stage": bool(readiness["ready_for_next_stage"]),
                "ready_for_long_run": bool(readiness["ready_for_long_run"]),
                "stage_status_reason": str(readiness["stage_status_reason"]),
                "run_dir": str(run.get("run_dir", "") or ""),
            }
        )
    return rows


def _print_flow_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No flow runs found.")
        return

    headers = [
        "run_id",
        "status",
        "mesh",
        "ready_for_next_stage",
        "ready_for_long_run",
        "stage_status_reason",
        "run_dir",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def _render(values: Iterable[str]) -> str:
        return "  ".join(
            value.ljust(widths[header]) for header, value in zip(headers, values)
        )

    print(_render(headers))
    print(_render(["-" * widths[header] for header in headers]))
    for row in rows:
        print(_render([str(row.get(header, "")) for header in headers]))


def _show_run_payload(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    run = get_pipeline_run(manifest, run_id)
    children = list_pipeline_runs(manifest, parent_run_id=run_id)
    payload = {
        "pipeline_id": str(manifest.get("pipeline_id", "")),
        "manifest_path": str(manifest_path),
        "run": run,
        "resolved_run_dir": (
            str(resolve_manifest_path_reference(manifest_path, run["run_dir"]))
            if str(run.get("run_dir", "")).strip()
            else None
        ),
        "child_run_ids": [str(child.get("run_id", "")) for child in children],
        "readiness": _extract_readiness(run),
    }
    return payload


def _validate_flow_source(
    manifest: Mapping[str, Any],
    *,
    flow_run_id: str,
) -> dict[str, Any]:
    run = get_pipeline_run(manifest, flow_run_id)
    if str(run.get("stage_type", "")) != "flow":
        raise RuntimeError(
            f"Run {flow_run_id!r} is stage_type={run.get('stage_type')!r}, expected 'flow'."
        )
    if str(run.get("status", "")) != "completed":
        raise RuntimeError(
            f"Run {flow_run_id!r} is status={run.get('status')!r}, expected 'completed'."
        )
    return run


def _validate_import_source(
    manifest: Mapping[str, Any],
    *,
    import_run_id: str,
) -> dict[str, Any]:
    run = get_pipeline_run(manifest, import_run_id)
    if str(run.get("stage_type", "")) != "import":
        raise RuntimeError(
            f"Run {import_run_id!r} is stage_type={run.get('stage_type')!r}, expected 'import'."
        )
    if str(run.get("status", "")) != "completed":
        raise RuntimeError(
            f"Run {import_run_id!r} is status={run.get('status')!r}, expected 'completed'."
        )
    return run


def _resolve_external_flow_source(source_path: str | Path) -> tuple[Path, Path]:
    candidate = Path(str(source_path)).resolve()
    if candidate.is_dir():
        run_dir = candidate
        summary_path = run_dir / "summary.json"
    else:
        summary_path = candidate
        run_dir = candidate.parent
    if not summary_path.exists():
        raise FileNotFoundError(f"Flow summary json not found: {summary_path}")
    return run_dir, summary_path


def _infer_external_flow_mesh_name(
    *,
    summary: Mapping[str, Any],
    mesh_npz: Path,
    run_dir: Path,
) -> str:
    mesh_name = str(summary.get("mesh_name", "")).strip()
    if mesh_name:
        return mesh_name
    stem = mesh_npz.stem
    if stem.endswith("_imported_mesh"):
        return stem.removesuffix("_imported_mesh")
    return run_dir.name


def _validate_external_flow_registration_source(
    source_path: str | Path,
) -> dict[str, Any]:
    run_dir, summary_path = _resolve_external_flow_source(source_path)
    payload = load_flow_summary_payload(summary_path)
    summary = (
        dict(payload.get("summary", {}))
        if isinstance(payload.get("summary", {}), Mapping)
        else {}
    )
    metadata = (
        dict(payload.get("metadata", {}))
        if isinstance(payload.get("metadata", {}), Mapping)
        else {}
    )
    readiness = validate_source_flow_readiness(
        metadata,
        strict=True,
        strict_label="external source flow",
    )
    resolved_mesh_raw = str(summary.get("resolved_mesh_npz", "")).strip()
    if not resolved_mesh_raw:
        raise RuntimeError(
            f"Flow summary {summary_path} does not define 'resolved_mesh_npz'."
        )
    mesh_npz = Path(resolved_mesh_raw).resolve()
    if not mesh_npz.exists():
        raise FileNotFoundError(f"Resolved mesh npz not found: {mesh_npz}")

    required_summary_true = {
        "projection_solved": summary.get("projection_solved"),
        "pressure_linear_accepted": summary.get("pressure_linear_accepted"),
        "flow_progression_solved": summary.get("flow_progression_solved"),
        "flow_solved": summary.get("flow_solved"),
    }
    for field_name, raw_value in required_summary_true.items():
        if raw_value is not True:
            raise RuntimeError(
                f"External source flow summary requires {field_name}=true; "
                f"got {raw_value!r} from {summary_path}."
            )

    for field_name in ("run_completed", "numerically_stable", "physically_ready"):
        if readiness.get(field_name) is not True:
            raise RuntimeError(
                f"External source flow readiness requires {field_name}=true; "
                f"got {readiness.get(field_name)!r}."
            )

    summary_ready = summary.get("ready_for_next_stage")
    if summary_ready is not None and bool(summary_ready) is not bool(
        readiness["ready"]
    ):
        raise RuntimeError(
            "External source flow readiness is contradictory between summary.json "
            "and flow_coupling_metadata.json."
        )

    source_run_dir = Path(str(payload.get("flow_run_dir", run_dir))).resolve()
    metadata_path = Path(
        str(payload.get("resolved_metadata_path", payload.get("metadata_path", "")))
    ).resolve()
    flux_path = Path(str(payload.get("flux_path", ""))).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Flow coupling metadata not found: {metadata_path}")
    if not flux_path.exists():
        raise FileNotFoundError(
            f"Final corrected face flux file not found: {flux_path}"
        )

    config_json = source_run_dir / "config.json"
    acceptance_report_json = source_run_dir / "acceptance_report.json"
    mesh_name = _infer_external_flow_mesh_name(
        summary=summary,
        mesh_npz=mesh_npz,
        run_dir=source_run_dir,
    )
    return {
        "run_dir": source_run_dir,
        "summary_path": summary_path.resolve(),
        "summary": summary,
        "metadata": metadata,
        "readiness": readiness,
        "mesh_name": mesh_name,
        "mesh_npz": mesh_npz,
        "flow_coupling_metadata_json": metadata_path,
        "final_corrected_face_flux_npy": flux_path,
        "config_json": config_json.resolve() if config_json.exists() else None,
        "acceptance_report_json": (
            acceptance_report_json.resolve()
            if acceptance_report_json.exists()
            else None
        ),
    }


def _run_subprocess(cmd: list[str], *, dry_run: bool) -> int:
    rendered = " ".join(cmd)
    print(f"[pipeline-manifest-tool] launching: {rendered}")
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def _common_launch_prefix(
    *,
    python_exe: str,
    run_root: Path,
    source_flag: str,
    source_run_id: str,
    require_ready_flow_for_downstream: bool,
    allow_unready_flow_for_downstream: bool,
) -> list[str]:
    cmd = [
        str(Path(python_exe)),
        str(PIPELINE_RUNNER_SCRIPT),
        "--run-root",
        str(run_root),
        source_flag,
        str(source_run_id),
    ]
    if allow_unready_flow_for_downstream:
        cmd.append("--allow-unready-flow-for-downstream")
    elif require_ready_flow_for_downstream:
        cmd.append("--require-ready-flow-for-downstream")
    return cmd


def handle_list_flows(args: argparse.Namespace) -> int:
    manifest_path, _, manifest = _resolve_cli_context(args)
    rows = _build_flow_rows(
        manifest_path,
        manifest,
        ready_only=bool(args.ready_only),
    )
    if bool(args.json):
        print(_json_dump(rows))
    else:
        _print_flow_rows(rows)
    return 0


def handle_show_run(args: argparse.Namespace) -> int:
    manifest_path, _, manifest = _resolve_cli_context(args)
    payload = _show_run_payload(
        manifest_path,
        manifest,
        run_id=str(args.run_id),
    )
    print(_json_dump(payload))
    return 0


def handle_register_external_flow(args: argparse.Namespace) -> int:
    manifest_path, _, manifest = _resolve_registration_context(args)
    source = _validate_external_flow_registration_source(str(args.source_path))
    run_id = str(args.run_id).strip() or f"flow-external-{source['run_dir'].name}"
    display_name = str(args.display_name).strip() or (
        f"External flow source {source['mesh_name']}"
    )

    existing_run = manifest.get("runs", {}).get(run_id)
    if isinstance(existing_run, Mapping):
        existing_metadata = (
            dict(existing_run.get("metadata", {}))
            if isinstance(existing_run.get("metadata", {}), Mapping)
            else {}
        )
        existing_source_run_dir = str(
            existing_metadata.get("source_run_dir", existing_run.get("run_dir", ""))
        ).strip()
        if (
            existing_source_run_dir
            and Path(existing_source_run_dir).resolve()
            != Path(source["run_dir"]).resolve()
        ):
            raise RuntimeError(
                f"Manifest run {run_id!r} already points at a different source run: "
                f"{existing_source_run_dir}"
            )

    outputs = {
        "mesh_npz": source["mesh_npz"],
        "summary_json": source["summary_path"],
        "flow_coupling_metadata_json": source["flow_coupling_metadata_json"],
        "final_corrected_face_flux_npy": source["final_corrected_face_flux_npy"],
    }
    artifacts = dict(outputs)
    if source["config_json"] is not None:
        outputs["config_json"] = source["config_json"]
        artifacts["config_json"] = source["config_json"]
    if source["acceptance_report_json"] is not None:
        outputs["acceptance_report_json"] = source["acceptance_report_json"]
        artifacts["acceptance_report_json"] = source["acceptance_report_json"]

    readiness = dict(source["readiness"])
    stage_status_reason = str(readiness.get("stage_status_reason", "")).strip()
    if not stage_status_reason:
        stage_status_reason = "registered external accepted flow source"
    provenance = {
        "kind": "external_flow_run",
        "source_run_dir": source["run_dir"],
        "source_summary_json": source["summary_path"],
        "source_flow_coupling_metadata_json": source["flow_coupling_metadata_json"],
        "source_final_corrected_face_flux_npy": source["final_corrected_face_flux_npy"],
        "source_resolved_mesh_npz": source["mesh_npz"],
    }
    upsert_pipeline_run(
        manifest_path,
        run_id=run_id,
        stage_type="flow",
        status="completed",
        run_dir=source["run_dir"],
        display_name=display_name,
        parent_run_ids=[],
        inputs={
            "registration_kind": "external_flow_run",
            "registration_source_path": Path(str(args.source_path)).resolve(),
            "resolved_mesh_npz": source["mesh_npz"],
        },
        outputs=outputs,
        artifacts=artifacts,
        metadata={
            "mesh_name": source["mesh_name"],
            "run_completed": bool(readiness["run_completed"]),
            "numerically_stable": bool(readiness["numerically_stable"]),
            "physically_ready": bool(readiness["physically_ready"]),
            "ready_for_next_stage": True,
            "ready_for_long_run": bool(readiness["ready_for_long_run"]),
            "stage_status_reason": stage_status_reason,
            "source_kind": "external_flow_run",
            "source_run_dir": source["run_dir"],
            "source_summary_json": source["summary_path"],
            "source_flow_coupling_metadata_json": source["flow_coupling_metadata_json"],
            "source_final_corrected_face_flux_npy": source[
                "final_corrected_face_flux_npy"
            ],
            "source_resolved_mesh_npz": source["mesh_npz"],
            "source_provenance": provenance,
        },
    )
    registered_manifest = load_pipeline_manifest(manifest_path)
    payload = {
        "manifest_path": str(manifest_path),
        "run_id": run_id,
        "run": get_pipeline_run(registered_manifest, run_id),
    }
    if bool(args.json):
        print(_json_dump(payload))
    else:
        print(
            f"Registered external flow source {run_id} in {manifest_path} "
            f"from {source['run_dir']}."
        )
    return 0


def handle_run_flow(args: argparse.Namespace) -> int:
    _, run_root, manifest = _resolve_cli_context(args)
    import_run_id = str(args.import_run_id).strip()
    _validate_import_source(manifest, import_run_id=import_run_id)

    cmd = _common_launch_prefix(
        python_exe=str(args.python_exe),
        run_root=run_root,
        source_flag="--from-import-run-id",
        source_run_id=import_run_id,
        require_ready_flow_for_downstream=False,
        allow_unready_flow_for_downstream=False,
    )
    cmd.extend(
        [
            "--skip-transport",
            "--flow-backend",
            str(args.flow_backend),
            "--flow-mode",
            str(args.flow_mode),
            "--flow-steps",
            str(args.flow_steps),
            "--flow-inlet-speed",
            str(args.flow_inlet_speed),
            "--flow-dt-mode",
            str(args.flow_dt_mode),
            "--flow-dt-min",
            str(args.flow_dt_min),
            "--flow-convective-cfl-target",
            str(args.flow_convective_cfl_target),
            "--flow-wall-velocity-boundary-mode",
            str(args.flow_wall_velocity_boundary_mode),
            "--flow-wall-tangential-no-slip-strength",
            str(args.flow_wall_tangential_no_slip_strength),
            "--flow-wall-flux-stokes-resistance-strength",
            str(args.flow_wall_flux_stokes_resistance_strength),
            "--flow-startup-warning-steps",
            str(args.flow_startup_warning_steps),
        ]
    )
    if args.flow_wall_tangential_no_slip_strength_ramp_start is not None:
        cmd.extend(
            [
                "--flow-wall-tangential-no-slip-strength-ramp-start",
                str(args.flow_wall_tangential_no_slip_strength_ramp_start),
            ]
        )
    if int(args.flow_wall_tangential_no_slip_strength_ramp_steps) > 0:
        cmd.extend(
            [
                "--flow-wall-tangential-no-slip-strength-ramp-steps",
                str(args.flow_wall_tangential_no_slip_strength_ramp_steps),
            ]
        )
    if args.flow_dt is not None:
        cmd.extend(["--flow-dt", str(args.flow_dt)])
    if args.flow_dt_max is not None:
        cmd.extend(["--flow-dt-max", str(args.flow_dt_max)])
    if args.flow_viscous_face_flux_divergence_impact_cap is not None:
        cmd.extend(
            [
                "--flow-viscous-face-flux-divergence-impact-cap",
                str(args.flow_viscous_face_flux_divergence_impact_cap),
            ]
        )
    if str(args.flow_viscous_predictor_mode).strip():
        cmd.extend(
            [
                "--flow-viscous-predictor-mode",
                str(args.flow_viscous_predictor_mode).strip(),
            ]
        )
    if str(args.flow_viscous_predictor_outlet_contract_mode).strip():
        cmd.extend(
            [
                "--flow-viscous-predictor-outlet-contract-mode",
                str(args.flow_viscous_predictor_outlet_contract_mode).strip(),
            ]
        )
    if str(args.flow_pressure_projection_outlet_contract_mode).strip():
        cmd.extend(
            [
                "--flow-pressure-projection-outlet-contract-mode",
                str(args.flow_pressure_projection_outlet_contract_mode).strip(),
            ]
        )
    if str(args.flow_projection_cell_velocity_update_mode).strip():
        cmd.extend(
            [
                "--flow-projection-cell-velocity-update-mode",
                str(args.flow_projection_cell_velocity_update_mode).strip(),
            ]
        )
    if str(args.flow_pressure_nonorthogonal_correction_mode).strip():
        cmd.extend(
            [
                "--flow-pressure-nonorthogonal-correction-mode",
                str(args.flow_pressure_nonorthogonal_correction_mode).strip(),
            ]
        )
    if str(args.flow_viscous_nonorthogonal_correction_mode).strip():
        cmd.extend(
            [
                "--flow-viscous-nonorthogonal-correction-mode",
                str(args.flow_viscous_nonorthogonal_correction_mode).strip(),
            ]
        )
    if args.flow_pressure_nonorthogonal_correction_sweeps is not None:
        cmd.extend(
            [
                "--flow-pressure-nonorthogonal-correction-sweeps",
                str(args.flow_pressure_nonorthogonal_correction_sweeps),
            ]
        )
    if args.flow_pressure_nonorthogonal_correction_relaxation is not None:
        cmd.extend(
            [
                "--flow-pressure-nonorthogonal-correction-relaxation",
                str(args.flow_pressure_nonorthogonal_correction_relaxation),
            ]
        )
    if str(args.flow_convective_stabilization_mode).strip():
        cmd.extend(
            [
                "--flow-convective-stabilization-mode",
                str(args.flow_convective_stabilization_mode).strip(),
            ]
        )
    if args.flow_disable_convective_auto_damping is not None:
        if bool(args.flow_disable_convective_auto_damping):
            cmd.append("--flow-disable-convective-auto-damping")
        else:
            cmd.append("--no-flow-disable-convective-auto-damping")
    if args.flow_max_pressure_iterations is not None:
        cmd.extend(
            ["--flow-max-pressure-iterations", str(args.flow_max_pressure_iterations)]
        )
    if args.flow_pressure_relative_tolerance is not None:
        cmd.extend(
            [
                "--flow-pressure-relative-tolerance",
                str(args.flow_pressure_relative_tolerance),
            ]
        )
    if str(args.flow_pressure_solver).strip():
        cmd.extend(["--flow-pressure-solver", str(args.flow_pressure_solver).strip()])
    if str(args.flow_device).strip():
        cmd.extend(["--flow-device", str(args.flow_device).strip()])
    if str(args.resume_flow_run_dir).strip():
        cmd.extend(["--resume-flow-run-dir", str(args.resume_flow_run_dir).strip()])
    if bool(args.flow_wall_tangential_shear_face_flux_enabled):
        cmd.append("--flow-wall-tangential-shear-face-flux-enabled")
    else:
        cmd.append("--no-flow-wall-tangential-shear-face-flux-enabled")
    if bool(args.flow_wall_tangential_cell_velocity_momentum_enabled):
        cmd.append("--flow-wall-tangential-cell-velocity-momentum-enabled")
    else:
        cmd.append("--no-flow-wall-tangential-cell-velocity-momentum-enabled")
    if args.flow_wall_flux_stokes_resistance_enabled is not None:
        if bool(args.flow_wall_flux_stokes_resistance_enabled):
            cmd.append("--flow-wall-flux-stokes-resistance-enabled")
        else:
            cmd.append("--no-flow-wall-flux-stokes-resistance-enabled")

    return _run_subprocess(cmd, dry_run=bool(args.dry_run))


def handle_run_transport(args: argparse.Namespace) -> int:
    _, run_root, manifest = _resolve_cli_context(args)
    if args.transport_steps is not None and args.transport_end_time is not None:
        raise ValueError(
            "--transport-steps and --transport-end-time are mutually exclusive."
        )
    if args.transport_end_time is not None and args.transport_end_time <= 0.0:
        raise ValueError("--transport-end-time must be positive.")
    if args.transport_steps is None and args.transport_end_time is None:
        args.transport_steps = 20000
    flow_run_id = str(args.flow_run_id).strip()
    _validate_flow_source(manifest, flow_run_id=flow_run_id)

    cmd = _common_launch_prefix(
        python_exe=str(args.python_exe),
        run_root=run_root,
        source_flag="--from-flow-run-id",
        source_run_id=flow_run_id,
        require_ready_flow_for_downstream=bool(args.require_ready_flow_for_downstream),
        allow_unready_flow_for_downstream=bool(args.allow_unready_flow_for_downstream),
    )
    cmd.extend(
        [
            "--transport-backend",
            str(args.transport_backend),
            "--transport-execution-backend",
            str(args.transport_execution_backend),
            "--transport-dt-mode",
            str(args.transport_dt_mode),
            "--transport-dt",
            str(args.transport_dt),
            "--transport-snapshot-every",
            str(args.transport_snapshot_every),
            "--transport-checkpoint-every",
            str(args.transport_checkpoint_every),
            "--transport-progress-every",
            str(args.transport_progress_every),
            "--transport-mode",
            str(args.transport_mode),
            "--transport-scheme",
            str(args.transport_scheme),
            "--transport-cfl-target",
            str(args.transport_cfl_target),
            "--transport-cfl-limit",
            str(args.transport_cfl_limit),
            "--transport-diffusivity",
            str(args.transport_diffusivity),
            "--transport-kinematic-viscosity",
            str(args.transport_kinematic_viscosity),
            "--transport-max-supported-grid-peclet",
            str(args.transport_max_supported_grid_peclet),
            "--transport-max-supported-schmidt",
            str(args.transport_max_supported_schmidt),
            "--transport-gradient-method",
            str(args.transport_gradient_method),
            "--transport-laplacian-method",
            str(args.transport_laplacian_method),
            "--transport-left-inlet-value",
            str(args.transport_left_inlet_value),
            "--transport-right-inlet-value",
            str(args.transport_right_inlet_value),
            "--transport-inlet-speed",
            str(args.transport_inlet_speed),
        ]
    )
    if args.transport_end_time is None:
        cmd.extend(["--transport-steps", str(args.transport_steps)])
    else:
        cmd.extend(["--transport-end-time", str(args.transport_end_time)])
    if str(args.transport_torch_device).strip():
        cmd.extend(
            ["--transport-torch-device", str(args.transport_torch_device).strip()]
        )
    transport_requests_torch = str(args.transport_execution_backend) == "torch" or (
        str(args.transport_backend) == "torch"
    )
    if bool(args.transport_no_velocity_comparison):
        cmd.append("--transport-no-velocity-comparison")
    if bool(args.transport_no_transport_scheme_comparison):
        cmd.append("--transport-no-transport-scheme-comparison")
    if bool(args.transport_fail_if_numpy_fallback) and transport_requests_torch:
        cmd.append("--transport-fail-if-numpy-fallback")
    return _run_subprocess(cmd, dry_run=bool(args.dry_run))


def handle_run_thermal(args: argparse.Namespace) -> int:
    manifest_path, run_root, manifest = _resolve_cli_context(args)
    flow_run_id = str(args.flow_run_id).strip()
    _validate_flow_source(manifest, flow_run_id=flow_run_id)

    msh_value = str(args.msh).strip() or _infer_msh_argument(
        manifest_path,
        manifest,
        flow_run_id=flow_run_id,
    )
    cmd = _common_launch_prefix(
        python_exe=str(args.python_exe),
        run_root=run_root,
        source_flag="--from-flow-run-id",
        source_run_id=flow_run_id,
        require_ready_flow_for_downstream=bool(args.require_ready_flow_for_downstream),
        allow_unready_flow_for_downstream=bool(args.allow_unready_flow_for_downstream),
    )
    cmd.extend(
        [
            "--skip-transport",
            "--run-thermal",
            "--msh",
            msh_value,
            "--thermal-backend",
            str(args.thermal_backend),
            "--thermal-steps",
            str(args.thermal_steps),
            "--thermal-dt-mode",
            str(args.thermal_dt_mode),
            "--thermal-dt",
            str(args.thermal_dt),
            "--thermal-cfl-target",
            str(args.thermal_cfl_target),
            "--thermal-cfl-limit",
            str(args.thermal_cfl_limit),
            "--thermal-heat-source",
            str(args.thermal_heat_source),
        ]
    )
    if str(args.thermal_torch_device).strip():
        cmd.extend(["--thermal-torch-device", str(args.thermal_torch_device).strip()])

    return _run_subprocess(cmd, dry_run=bool(args.dry_run))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_manifest_context(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--run-root",
            type=str,
            default=str(DEFAULT_RUN_ROOT),
            help="Pipeline run root that contains pipeline_manifest.json.",
        )
        subparser.add_argument(
            "--pipeline-manifest",
            type=str,
            default="",
            help="Optional explicit path to pipeline_manifest.json.",
        )

    list_flows = subparsers.add_parser(
        "list-flows",
        help="List flow runs from the manifest.",
    )
    add_manifest_context(list_flows)
    list_flows.add_argument("--json", action="store_true")
    list_flows.set_defaults(func=handle_list_flows, ready_only=False)

    list_ready_flows = subparsers.add_parser(
        "list-ready-flows",
        help="List only flow runs that are ready for downstream stages.",
    )
    add_manifest_context(list_ready_flows)
    list_ready_flows.add_argument("--json", action="store_true")
    list_ready_flows.set_defaults(func=handle_list_flows, ready_only=True)

    show_run = subparsers.add_parser(
        "show-run",
        help="Show a manifest run with derived lineage details.",
    )
    add_manifest_context(show_run)
    show_run.add_argument("run_id", type=str)
    show_run.set_defaults(func=handle_show_run)

    register_external_flow = subparsers.add_parser(
        "register-external-flow",
        help=(
            "Register an existing accepted flow run as an immutable external source "
            "inside a manifest."
        ),
    )
    add_manifest_context(register_external_flow)
    register_external_flow.add_argument(
        "--pipeline-id",
        type=str,
        default="",
        help="Optional pipeline id used when creating a new manifest.",
    )
    register_external_flow.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional manifest run id. Defaults to flow-external-<source-dir>.",
    )
    register_external_flow.add_argument(
        "--display-name",
        type=str,
        default="",
        help="Optional display name stored in the manifest.",
    )
    register_external_flow.add_argument("--json", action="store_true")
    register_external_flow.add_argument(
        "source_path",
        type=str,
        help="Existing flow run directory or its summary.json path.",
    )
    register_external_flow.set_defaults(func=handle_register_external_flow)

    run_flow = subparsers.add_parser(
        "run-flow",
        help="Launch a new flow run from an existing manifest import run.",
    )
    add_manifest_context(run_flow)
    run_flow.add_argument("import_run_id", type=str)
    run_flow.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the existing pipeline runner.",
    )
    run_flow.add_argument("--dry-run", action="store_true")
    run_flow.add_argument(
        "--flow-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    run_flow.add_argument("--flow-device", type=str, default="")
    run_flow.add_argument(
        "--flow-mode",
        type=str,
        default="navier_stokes_projection_debug",
    )
    run_flow.add_argument("--flow-steps", type=int, default=700)
    run_flow.add_argument("--flow-inlet-speed", type=float, default=0.15)
    run_flow.add_argument(
        "--flow-dt-mode",
        type=str,
        choices=("manual", "auto_cfl"),
        default="auto_cfl",
    )
    run_flow.add_argument("--flow-dt", type=float, default=None)
    run_flow.add_argument("--flow-dt-min", type=float, default=1e-7)
    run_flow.add_argument("--flow-dt-max", type=float, default=None)
    run_flow.add_argument("--flow-convective-cfl-target", type=float, default=0.5)
    run_flow.add_argument("--resume-flow-run-dir", type=str, default="")
    run_flow.add_argument(
        "--flow-wall-velocity-boundary-mode",
        type=str,
        choices=("slip", "no_slip", "no_slip_tangential", "no_slip_legacy_isotropic"),
        default="slip",
    )
    run_flow.add_argument(
        "--flow-wall-tangential-no-slip-strength",
        type=float,
        default=1.0,
    )
    run_flow.add_argument(
        "--flow-wall-tangential-no-slip-strength-ramp-start",
        type=float,
        default=None,
    )
    run_flow.add_argument(
        "--flow-wall-tangential-no-slip-strength-ramp-steps",
        type=int,
        default=0,
    )
    run_flow.add_argument(
        "--flow-wall-tangential-shear-face-flux-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_flow.add_argument(
        "--flow-wall-tangential-cell-velocity-momentum-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_flow.add_argument(
        "--flow-wall-flux-stokes-resistance-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    run_flow.add_argument(
        "--flow-wall-flux-stokes-resistance-strength",
        type=float,
        default=1.0,
    )
    run_flow.add_argument(
        "--flow-viscous-face-flux-divergence-impact-cap",
        type=float,
        default=0.03,
    )
    run_flow.add_argument(
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
    run_flow.add_argument(
        "--flow-viscous-predictor-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="",
    )
    run_flow.add_argument(
        "--flow-pressure-projection-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="",
    )
    run_flow.add_argument(
        "--flow-projection-cell-velocity-update-mode",
        type=str,
        choices=("auto", "legacy_reconstruct", "momentum_pressure_corrected"),
        default="",
    )
    run_flow.add_argument(
        "--flow-pressure-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="",
    )
    run_flow.add_argument(
        "--flow-viscous-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="",
    )
    run_flow.add_argument(
        "--flow-pressure-nonorthogonal-correction-sweeps",
        type=int,
        default=None,
    )
    run_flow.add_argument(
        "--flow-pressure-nonorthogonal-correction-relaxation",
        type=float,
        default=None,
    )
    run_flow.add_argument(
        "--flow-convective-stabilization-mode",
        type=str,
        choices=("auto_damping", "substepping"),
        default="auto_damping",
    )
    run_flow.add_argument(
        "--flow-disable-convective-auto-damping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_flow.add_argument("--flow-max-pressure-iterations", type=int, default=1000)
    run_flow.add_argument(
        "--flow-pressure-relative-tolerance", type=float, default=1e-4
    )
    run_flow.add_argument(
        "--flow-pressure-solver",
        type=str,
        choices=("jacobi", "cg", "pcg_diag", "amg_pcg"),
        default="pcg_diag",
    )
    run_flow.add_argument("--flow-startup-warning-steps", type=int, default=10)
    run_flow.set_defaults(func=handle_run_flow)

    run_transport = subparsers.add_parser(
        "run-transport",
        help="Launch a new transport run from an existing manifest flow run.",
    )
    add_manifest_context(run_transport)
    run_transport.add_argument("flow_run_id", type=str)
    run_transport.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the existing pipeline runner.",
    )
    run_transport.add_argument(
        "--require-ready-flow-for-downstream",
        action="store_true",
    )
    run_transport.add_argument(
        "--allow-unready-flow-for-downstream",
        action="store_true",
    )
    run_transport.add_argument("--dry-run", action="store_true")
    run_transport.add_argument(
        "--transport-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    run_transport.add_argument(
        "--transport-execution-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    run_transport.add_argument("--transport-torch-device", type=str, default="")
    run_transport.add_argument("--transport-steps", type=int, default=None)
    run_transport.add_argument("--transport-end-time", type=float, default=None)
    run_transport.add_argument(
        "--transport-dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    run_transport.add_argument("--transport-dt", type=float, default=1.5e-5)
    run_transport.add_argument("--transport-snapshot-every", type=int, default=5000)
    run_transport.add_argument("--transport-checkpoint-every", type=int, default=5000)
    run_transport.add_argument("--transport-progress-every", type=int, default=500)
    run_transport.add_argument(
        "--transport-mode",
        type=str,
        choices=("advection", "advection_diffusion"),
        default="advection_diffusion",
    )
    run_transport.add_argument(
        "--transport-scheme",
        type=str,
        choices=("upwind", "bounded_upwind"),
        default="bounded_upwind",
    )
    run_transport.add_argument("--transport-cfl-target", type=float, default=0.5)
    run_transport.add_argument("--transport-cfl-limit", type=float, default=0.8)
    run_transport.add_argument("--transport-diffusivity", type=float, default=3e-10)
    run_transport.add_argument(
        "--transport-kinematic-viscosity", type=float, default=1e-6
    )
    run_transport.add_argument(
        "--transport-max-supported-grid-peclet", type=float, default=2.0
    )
    run_transport.add_argument(
        "--transport-max-supported-schmidt", type=float, default=1000.0
    )
    run_transport.add_argument(
        "--transport-gradient-method",
        type=str,
        choices=("face", "least_squares"),
        default="least_squares",
    )
    run_transport.add_argument(
        "--transport-laplacian-method",
        type=str,
        choices=("tpfa", "lsq_flux"),
        default="tpfa",
    )
    run_transport.add_argument("--transport-left-inlet-value", type=float, default=0.0)
    run_transport.add_argument("--transport-right-inlet-value", type=float, default=1.0)
    run_transport.add_argument("--transport-inlet-speed", type=float, default=0.15)
    run_transport.add_argument(
        "--transport-no-velocity-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_transport.add_argument(
        "--transport-no-transport-scheme-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_transport.add_argument(
        "--transport-fail-if-numpy-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run_transport.set_defaults(func=handle_run_transport)

    run_thermal = subparsers.add_parser(
        "run-thermal",
        help="Launch a new thermal run from an existing manifest flow run.",
    )
    add_manifest_context(run_thermal)
    run_thermal.add_argument("flow_run_id", type=str)
    run_thermal.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the existing pipeline runner.",
    )
    run_thermal.add_argument(
        "--require-ready-flow-for-downstream",
        action="store_true",
    )
    run_thermal.add_argument(
        "--allow-unready-flow-for-downstream",
        action="store_true",
    )
    run_thermal.add_argument("--dry-run", action="store_true")
    run_thermal.add_argument(
        "--msh",
        type=str,
        default="",
        help="Optional mesh path. When omitted, the helper infers it from the import ancestor.",
    )
    run_thermal.add_argument(
        "--thermal-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    run_thermal.add_argument("--thermal-torch-device", type=str, default="")
    run_thermal.add_argument("--thermal-steps", type=int, default=20000)
    run_thermal.add_argument(
        "--thermal-dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    run_thermal.add_argument("--thermal-dt", type=float, default=5e-4)
    run_thermal.add_argument("--thermal-cfl-target", type=float, default=0.5)
    run_thermal.add_argument("--thermal-cfl-limit", type=float, default=0.8)
    run_thermal.add_argument("--thermal-heat-source", type=float, default=0.0)
    run_thermal.set_defaults(func=handle_run_thermal)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = int(args.func(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
