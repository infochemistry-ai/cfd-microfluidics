"""Run CPU v1 reactive transport on a staged tetra mesh and fixed flow field."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import meshio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for import_path in (PROJECT_ROOT, COMPUTE_SRC):
    import_text = str(import_path)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from experiments.gmsh._flow_coupling import (  # noqa: E402
    check_flow_mesh_compatibility,
    load_flow_summary_payload,
    sha256_file,
    validate_source_flow_readiness,
)
from experiments.gmsh._pipeline_manifest import (  # noqa: E402
    add_pipeline_manifest_arguments,
    build_pipeline_manifest_recorder,
)
from microfluidics.gmsh.tetra.gmsh_tetra_mesh_loader import (  # noqa: E402
    load_imported_tetra_mesh_npz,
)
from microfluidics.reactive import (  # noqa: E402
    build_reactive_spatial_precompute,
    load_reactive_case,
    run_reactive_transport,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return normalized or "species"


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _species_field_names(species_names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for name in species_names:
        base = f"concentration_{_safe_name(name)}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        result[name] = candidate
    return result


def _export_vtu(
    path: Path,
    *,
    mesh,
    concentrations: np.ndarray,
    temperature: np.ndarray,
    heat_release: np.ndarray | None,
    species_names: tuple[str, ...],
    field_names: dict[str, str],
) -> Path:
    cell_data = {
        field_names[name]: [concentrations[:, index]]
        for index, name in enumerate(species_names)
    }
    cell_data["temperature_k"] = [temperature]
    if heat_release is not None:
        cell_data["heat_release_w_per_m3"] = [heat_release]
    result = meshio.Mesh(
        points=np.asarray(mesh.points, dtype=np.float64),
        cells=[("tetra", np.asarray(mesh.tetrahedra, dtype=np.int64))],
        cell_data=cell_data,
    )
    meshio.write(path, result)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Strang-split reactive transport on a tetrahedral CFD flow."
    )
    parser.add_argument("--mesh-npz", required=True)
    parser.add_argument("--flow-run-dir", required=True)
    parser.add_argument("--reactive-case", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--max-walltime-seconds",
        type=_nonnegative_finite_float,
        default=0.0,
        help=(
            "Optional soft walltime budget. Zero disables the limit; a positive "
            "value stops cleanly between substeps and preserves complete outer steps."
        ),
    )
    parser.add_argument(
        "--allow-unready-flow",
        action="store_true",
        help="Diagnostic override; the reactive result remains physically unready.",
    )
    parser.add_argument(
        "--allow-physically-unready-result",
        action="store_true",
        help=(
            "Diagnostic override: return success for a numerically completed result "
            "whose readiness checks failed. ready_for_next_stage remains false."
        ),
    )
    add_pipeline_manifest_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    case_path = Path(args.reactive_case).resolve()

    # Contract validation intentionally precedes flow artifact loading and
    # allocation of fields sized by the tetra count.
    case = load_reactive_case(case_path)

    mesh_path = Path(args.mesh_npz).resolve()
    flow_run_dir = Path(args.flow_run_dir).resolve()
    flow_summary_path = flow_run_dir / "summary.json"
    mesh = load_imported_tetra_mesh_npz(mesh_path)
    flow_payload = load_flow_summary_payload(
        flow_summary_path,
        require_artifact_hashes=True,
    )
    compatibility = check_flow_mesh_compatibility(
        mesh=mesh,
        flow_payload=flow_payload,
        mesh_sha256=sha256_file(mesh_path),
        require_topology=True,
    )
    if not bool(compatibility["compatible"]):
        raise RuntimeError(
            "reactive mesh/flow compatibility validation failed: "
            + json.dumps(compatibility, sort_keys=True)
        )
    readiness = validate_source_flow_readiness(
        flow_payload["metadata"],
        strict=not bool(args.allow_unready_flow),
        strict_label="source flow for reactive transport",
    )
    precompute = build_reactive_spatial_precompute(
        mesh,
        np.asarray(flow_payload["face_flux"], dtype=np.float64),
        case,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).resolve() / f"{stamp}_{_safe_name(case.case_id)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    recorder = build_pipeline_manifest_recorder(
        args, stage_type="reactive_transport", run_dir=run_dir
    )
    inputs = {
        "mesh_npz": str(mesh_path),
        "flow_summary_json": str(flow_summary_path),
        "flow_coupling_metadata_json": str(flow_payload["metadata_path"]),
        "flow_face_flux_npy": str(flow_payload["flux_path"]),
        "flow_face_to_cells_npy": str(flow_run_dir / "face_to_cells.npy"),
        "flow_cell_volumes_npy": str(flow_run_dir / "cell_volumes.npy"),
        "reactive_case_json": str(case_path),
    }
    recorder.record_started(
        inputs=inputs,
        metadata={
            "case_id": case.case_id,
            "mode": case.mode,
            "reactive_case_sha256": case.reactive_case_sha256,
        },
    )
    try:
        result = run_reactive_transport(
            precompute,
            case,
            upstream_flow_ready=bool(readiness["ready"]),
            mesh_flow_compatible=bool(compatibility["compatible"]),
            walltime_limit_s=(
                float(args.max_walltime_seconds)
                if float(args.max_walltime_seconds) > 0.0
                else None
            ),
        )
        field_names = _species_field_names(case.species_names)
        normalized_case_path = _write_json(
            run_dir / "normalized_reactive_case.json", case.normalized_payload
        )
        provenance = {
            "contract_version": "reactive_provenance_v1",
            "reactive_case_sha256": case.reactive_case_sha256,
            "mechanism_sha256": case.mechanism_sha256,
            "mechanism_missing_enthalpy_reactions": list(
                case.compiled_chemistry.missing_enthalpy_reactions
            ),
            "mode": case.mode,
            "inputs": inputs,
            "mesh_flow_compatibility": compatibility,
            "source_flow_readiness": readiness,
            "run_control": {
                "max_walltime_seconds": float(args.max_walltime_seconds),
                "allow_physically_unready_result": bool(
                    args.allow_physically_unready_result
                ),
            },
            "scientific_scope": {
                "fixed_velocity_field": True,
                "homogeneous_liquid_phase_only": True,
                "synthetic_thermal_mechanisms_are_not_physical_property_datasets": True,
            },
        }
        provenance_path = _write_json(run_dir / "provenance.json", provenance)
        history_path = _write_json(run_dir / "history.json", list(result.history))
        npz_payload: dict[str, np.ndarray] = {
            "concentrations_mol_per_m3": result.concentrations_mol_per_m3,
            "temperature_k": result.temperature_k,
            "species_sources_mol_per_m3_s": (result.species_sources_mol_per_m3_s),
            "cell_volumes_m3": np.asarray(mesh.cell_volumes, dtype=np.float64),
            "species_names_json": np.asarray(
                json.dumps(case.species_names, ensure_ascii=False)
            ),
            "mechanism_sha256": np.asarray(case.mechanism_sha256),
            "reactive_case_sha256": np.asarray(case.reactive_case_sha256),
            "physical_time_s": np.asarray(
                float(result.summary["physical_time_s"]), dtype=np.float64
            ),
        }
        if result.heat_release_w_per_m3 is not None:
            npz_payload["heat_release_w_per_m3"] = result.heat_release_w_per_m3
        fields_path = run_dir / "reactive_fields.npz"
        np.savez_compressed(fields_path, **npz_payload)
        vtu_path = _export_vtu(
            run_dir / "reactive_result.vtu",
            mesh=mesh,
            concentrations=result.concentrations_mol_per_m3,
            temperature=result.temperature_k,
            heat_release=result.heat_release_w_per_m3,
            species_names=case.species_names,
            field_names=field_names,
        )
        snapshot_paths: dict[str, str] = {}
        for step, (snapshot_c, snapshot_t) in sorted(result.snapshots.items()):
            snapshot_path = run_dir / f"snapshot_step_{step:08d}.npz"
            np.savez_compressed(
                snapshot_path,
                concentrations_mol_per_m3=snapshot_c,
                temperature_k=snapshot_t,
                physical_time_s=np.asarray(
                    step
                    * (
                        case.time.dt_s
                        if case.time.dt_mode == "manual"
                        else case.time.max_dt_s
                    ),
                    dtype=np.float64,
                ),
            )
            snapshot_paths[str(step)] = str(snapshot_path)

        summary = dict(result.summary)
        summary["lineage"] = {
            "mesh_npz": str(mesh_path),
            "flow_run_dir": str(flow_run_dir),
            "flow_summary_json": str(flow_summary_path),
            "flow_coupling_metadata_json": str(flow_payload["metadata_path"]),
            "flow_face_flux_npy": str(flow_payload["flux_path"]),
            "flow_face_to_cells_npy": str(flow_run_dir / "face_to_cells.npy"),
            "flow_cell_volumes_npy": str(flow_run_dir / "cell_volumes.npy"),
        }
        summary["species_field_names"] = field_names
        run_completed = bool(summary["readiness"]["run_completed"])
        physically_ready = bool(summary["readiness"]["physically_ready"])
        stage_succeeded = bool(
            run_completed
            and (physically_ready or bool(args.allow_physically_unready_result))
        )
        summary["readiness"]["cli_success"] = stage_succeeded
        summary["artifacts"] = {
            "summary_json": str(run_dir / "summary.json"),
            "reactive_fields_npz": str(fields_path),
            "reactive_result_vtu": str(vtu_path),
            "history_json": str(history_path),
            "normalized_reactive_case_json": str(normalized_case_path),
            "provenance_json": str(provenance_path),
            "validation_report_txt": str(run_dir / "validation_report.txt"),
            "snapshots": snapshot_paths,
        }
        summary_path = _write_json(run_dir / "summary.json", summary)
        validation_lines = [
            "Reactive transport validation report",
            f"case_id={case.case_id}",
            f"mode={case.mode}",
            f"run_completed={summary['readiness']['run_completed']}",
            f"numerically_stable={summary['readiness']['numerically_stable']}",
            f"physically_ready={summary['readiness']['physically_ready']}",
            f"ready_for_next_stage={summary['readiness']['ready_for_next_stage']}",
            f"cli_success={summary['readiness']['cli_success']}",
            f"status={summary['readiness']['status']}",
            f"mesh_flow_compatible={compatibility['compatible']}",
            f"upstream_flow_ready={readiness['ready']}",
            "physical_clipping_used=false",
        ]
        (run_dir / "validation_report.txt").write_text(
            "\n".join(validation_lines) + "\n", encoding="utf-8"
        )
        artifacts = dict(summary["artifacts"])
        manifest_metadata = {
            "case_id": case.case_id,
            "mode": case.mode,
            "status": summary["readiness"]["status"],
            "run_completed": run_completed,
            "physically_ready": physically_ready,
            "ready_for_next_stage": physically_ready,
        }
        if stage_succeeded:
            recorder.record_completed(
                inputs=inputs,
                outputs={
                    "summary_json": str(summary_path),
                    "run_completed": run_completed,
                    "physically_ready": physically_ready,
                    "ready_for_next_stage": physically_ready,
                },
                artifacts=artifacts,
                metadata=manifest_metadata,
            )
        else:
            recorder.record_failed(
                error=RuntimeError(str(summary["readiness"]["status"])),
                inputs=inputs,
                artifacts=artifacts,
                metadata=manifest_metadata,
            )
        print(f"[gmsh-tetra-reactive] summary written: {summary_path}")
        return 0 if stage_succeeded else 2
    except BaseException as exc:
        recorder.record_failed(error=exc, inputs=inputs)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
