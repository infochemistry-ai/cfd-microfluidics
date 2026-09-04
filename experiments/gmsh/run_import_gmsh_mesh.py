"""Direct import stage runner for Gmsh tetra mesh ingestion and validation.

Useful for explicit stage launches and diagnostics, but the supported orchestration
path remains the manifest-first pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GMSH_MESHES_DIR = PROJECT_ROOT / "data" / "meshes" / "gmsh"
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.gmsh._path_utils import _normalize_user_path  # noqa: E402
from experiments.gmsh._pipeline_manifest import (  # noqa: E402
    add_pipeline_manifest_arguments,
    build_pipeline_manifest_recorder,
)
from microfluidics.path_contract import (  # noqa: E402
    GMSH_IMPORT_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)


class _TeeWriter:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield


def _to_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def _resolve_msh_input(msh_input: str | Path) -> tuple[Path, list[Path]]:
    raw = Path(str(msh_input))
    searched: list[Path] = []

    if raw.is_absolute():
        candidate = raw.resolve()
        searched.append(candidate)
        if candidate.exists():
            return candidate, searched
        raise FileNotFoundError(
            "Gmsh mesh file not found at absolute path: "
            f"{candidate}. "
            "For filename-only input, place .msh in data/meshes/gmsh/."
        )

    is_filename_only = raw.parent in (Path("."), Path(""))
    if is_filename_only:
        candidates = (
            (GMSH_MESHES_DIR / raw.name).resolve(),
            (Path.cwd() / raw.name).resolve(),
            (PROJECT_ROOT / raw.name).resolve(),
        )
    else:
        candidates = (
            (Path.cwd() / raw).resolve(),
            (PROJECT_ROOT / raw).resolve(),
        )

    for candidate in candidates:
        searched.append(candidate)
        if candidate.exists():
            return candidate, searched

    searched_lines = "\n".join(f"- {path}" for path in searched)
    raise FileNotFoundError(
        "Gmsh mesh file not found.\n"
        f"Original input: {raw}\n"
        "Searched paths:\n"
        f"{searched_lines}\n"
        f"Hint: place mesh file in {GMSH_MESHES_DIR} or pass an absolute path."
    )


def _save_npz(mesh, output_path: Path) -> Path:
    boundary_face_names_json = json.dumps(
        {str(k): str(v) for k, v in mesh.boundary_face_names.items()}
    )
    physical_groups_json = json.dumps(
        {
            str(name): [int(values[0]), int(values[1])]
            for name, values in mesh.physical_groups.items()
        }
    )
    diagnostics_json = json.dumps(_to_serializable(mesh.diagnostics))
    case_config_json = json.dumps(_to_serializable(mesh.case_config))
    preprocessor_metadata_json = json.dumps(
        _to_serializable(mesh.preprocessor_metadata)
    )
    np.savez_compressed(
        output_path,
        points=mesh.points,
        tetrahedra=mesh.tetrahedra,
        boundary_triangles=mesh.boundary_triangles,
        boundary_face_tags=mesh.boundary_face_tags,
        volume_tag_per_cell=mesh.volume_tag_per_cell,
        cell_centers=mesh.cell_centers,
        cell_volumes=mesh.cell_volumes,
        face_vertices=mesh.face_vertices,
        face_centers=mesh.face_centers,
        face_areas=mesh.face_areas,
        face_normals=mesh.face_normals,
        face_to_cells=mesh.face_to_cells,
        cell_to_faces=mesh.cell_to_faces,
        boundary_tag_per_face=mesh.boundary_tag_per_face,
        interior_face_indices=mesh.interior_face_indices,
        boundary_face_indices=mesh.boundary_face_indices,
        inlet_faces=mesh.inlet_faces,
        outlet_faces=mesh.outlet_faces,
        wall_faces=mesh.wall_faces,
        unresolved_faces=mesh.boundary_unresolved_faces,
        source_path=np.asarray(str(mesh.source_path)),
        boundary_face_names_json=np.asarray(boundary_face_names_json),
        physical_groups_json=np.asarray(physical_groups_json),
        diagnostics_json=np.asarray(diagnostics_json),
        case_config_json=np.asarray(case_config_json),
        preprocessor_metadata_json=np.asarray(preprocessor_metadata_json),
    )
    return output_path


def main() -> None:
    from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh
    from microfluidics.gmsh.gmsh_mesh_preview import (
        export_imported_mesh_boundary_vtp,
        export_imported_mesh_vtu,
        save_import_previews,
    )
    from microfluidics.gmsh.gmsh_mesh_validation import (
        format_validation_report,
        validate_imported_tetra_mesh,
    )
    from microfluidics.preprocessor import (
        case_config_to_dict,
        compile_boundary_conditions,
        compile_material_cell_assignment,
        evaluate_mesh_quality_gate,
        load_case_config,
        resolve_case_zones,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--msh",
        type=str,
        default="",
        help="Mesh path or filename. Filename-only input is searched in ./data/meshes/gmsh/.",
    )
    parser.add_argument(
        "--case-config",
        type=str,
        default="",
        help=(
            "Optional case_config_v1 JSON. Its mesh.path is used when --msh is "
            "omitted, and its zones, BCs and quality policy are validated at import."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_IMPORT_RUNS_ROOT_REL)),
        help="Base directory for per-run artifacts.",
    )
    parser.add_argument(
        "--max-preview-points",
        type=int,
        default=60000,
        help="Max points/faces used in matplotlib previews.",
    )
    parser.add_argument(
        "--fail-on-validation-error",
        action="store_true",
        default=False,
        help="Exit with non-zero status if validation reports errors.",
    )
    parser.add_argument(
        "--allow-known-mesh-scale-mismatch",
        action="store_true",
        default=False,
        help=(
            "Unsafe debug override: allow import to continue even when a known mesh "
            "fails its reference scale audit."
        ),
    )
    add_pipeline_manifest_arguments(parser)
    args = parser.parse_args()
    case_config_path = (
        _normalize_user_path(args.case_config).resolve()
        if str(args.case_config).strip()
        else None
    )
    case_config = load_case_config(case_config_path) if case_config_path else None
    if not str(args.msh).strip() and case_config is None:
        parser.error(
            "--msh is required for import runs unless --case-config is provided. "
            "Automatic default mesh selection is no longer supported."
        )

    if str(args.msh).strip():
        original_input = str(args.msh)
        msh_input = _normalize_user_path(original_input)
    else:
        assert case_config is not None and case_config_path is not None
        original_input = case_config.mesh.path
        configured_path = Path(case_config.mesh.path)
        msh_input = (
            configured_path
            if configured_path.is_absolute()
            else (case_config_path.parent / configured_path).resolve()
        )
    output_root = _normalize_user_path(args.output_root).resolve()
    msh_path, searched_paths = _resolve_msh_input(msh_input)
    if case_config is not None and str(args.msh).strip():
        assert case_config_path is not None
        configured_path = Path(case_config.mesh.path)
        configured_path = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (case_config_path.parent / configured_path).resolve()
        )
        if configured_path != msh_path:
            parser.error(
                f"--msh resolves to {msh_path}, but case config mesh.path resolves "
                f"to {configured_path}"
            )
    run_dir = create_timestamped_run_dir(output_root, msh_path.stem)
    manifest_recorder = build_pipeline_manifest_recorder(
        args,
        stage_type="import",
        run_dir=run_dir,
    )
    manifest_inputs = {
        "original_input": original_input,
        "resolved_msh_path": str(msh_path),
        "searched_paths": [str(path) for path in searched_paths],
        "output_root": str(output_root),
        "case_config": str(case_config_path) if case_config_path else "",
    }
    manifest_artifacts = {
        "run_log": str(run_dir / "run.log"),
        "config_json": str(run_dir / "config.json"),
        "summary_json": str(run_dir / "summary.json"),
        "validation_report_txt": str(run_dir / "validation_report.txt"),
    }
    if case_config is not None:
        manifest_artifacts["normalized_case_config_json"] = str(
            run_dir / "case_config.normalized.json"
        )
    manifest_recorder.record_started(
        inputs=manifest_inputs,
        artifacts=manifest_artifacts,
        metadata={"manifest_role": "upstream_mesh_import"},
    )
    try:
        with _tee_logging(run_dir / "run.log"):
            print(f"[gmsh-import] run directory: {run_dir}")
            print(f"[gmsh-import] original input: {original_input}")
            print(f"[gmsh-import] resolved mesh path: {msh_path}")
            print(f"[gmsh-import] gmsh meshes dir: {GMSH_MESHES_DIR}")
            print("[gmsh-import] searched paths:")
            for path in searched_paths:
                print(f"  - {path}")
            config = {
                "original_input": original_input,
                "resolved_msh_path": str(msh_path),
                "searched_paths": [str(path) for path in searched_paths],
                "gmsh_meshes_dir": str(GMSH_MESHES_DIR),
                "output_root": str(output_root),
                "max_preview_points": int(args.max_preview_points),
                "case_config": str(case_config_path) if case_config_path else "",
                "fail_on_validation_error": bool(args.fail_on_validation_error),
                "allow_known_mesh_scale_mismatch": bool(
                    args.allow_known_mesh_scale_mismatch
                ),
            }
            _write_json(run_dir / "config.json", config)

            mesh = import_gmsh_tetra_mesh(msh_path)
            scale_audit = dict(mesh.diagnostics.get("mesh_scale_audit", {}))
            known_reference = scale_audit.get("known_mesh_reference")
            if (
                isinstance(known_reference, dict)
                and bool(known_reference)
                and not bool(known_reference.get("matches_reference", True))
                and not bool(args.allow_known_mesh_scale_mismatch)
            ):
                raise SystemExit(
                    "[gmsh-import] known-mesh scale audit failed. "
                    "Pass --allow-known-mesh-scale-mismatch only for explicit debug work."
                )
            resolved_zones = None
            compiled_boundaries = None
            quality_gate = None
            material_assignment = None
            if case_config is not None:
                resolved_zones = resolve_case_zones(mesh, case_config)
                compiled_boundaries = compile_boundary_conditions(
                    mesh,
                    case_config,
                    resolved_zones=resolved_zones,
                )
                quality_gate = evaluate_mesh_quality_gate(
                    mesh,
                    case_config.mesh_quality,
                    resolved_zones=resolved_zones,
                )
                if case_config.materials:
                    material_assignment = compile_material_cell_assignment(
                        mesh,
                        case_config,
                        resolved_zones=resolved_zones,
                    )
                _write_json(
                    run_dir / "case_config.normalized.json",
                    case_config_to_dict(case_config),
                )
                mesh.case_config = case_config_to_dict(case_config)
                flow_inlet_faces = [
                    item.face_indices
                    for item in compiled_boundaries.by_kind("velocity_inlet")
                ]
                flow_outlet_faces = [
                    item.face_indices
                    for item in compiled_boundaries.by_kind("pressure_outlet")
                ]
                flow_wall_faces = [
                    item.face_indices
                    for kind in ("wall", "symmetry")
                    for item in compiled_boundaries.by_kind(kind)
                ]
                if flow_inlet_faces:
                    mesh.inlet_faces = np.unique(
                        np.concatenate(flow_inlet_faces)
                    ).astype(np.int64)
                if flow_outlet_faces:
                    mesh.outlet_faces = np.unique(
                        np.concatenate(flow_outlet_faces)
                    ).astype(np.int64)
                if flow_wall_faces:
                    mesh.wall_faces = np.unique(np.concatenate(flow_wall_faces)).astype(
                        np.int64
                    )
                mesh.preprocessor_metadata = {
                    "case_id": case_config.case_id,
                    "schema_version": case_config.schema_version,
                    "flow_semantics_applied": bool(
                        flow_inlet_faces or flow_outlet_faces or flow_wall_faces
                    ),
                    "material_cell_counts": (
                        material_assignment.counts()
                        if material_assignment is not None
                        else {}
                    ),
                }
            report = validate_imported_tetra_mesh(
                mesh,
                require_legacy_boundary_classes=case_config is None,
            )
            report_text = format_validation_report(report)
            (run_dir / "validation_report.txt").write_text(
                report_text, encoding="utf-8"
            )
            print(report_text)

            npz_path = _save_npz(mesh, run_dir / f"{msh_path.stem}_imported_mesh.npz")
            volume_vtu_path = export_imported_mesh_vtu(
                mesh, run_dir / f"{msh_path.stem}_volume.vtu"
            )
            boundary_vtp = export_imported_mesh_boundary_vtp(
                mesh,
                run_dir,
                file_prefix=msh_path.stem,
            )
            previews = save_import_previews(
                mesh,
                run_dir,
                file_prefix=msh_path.stem,
                max_points=args.max_preview_points,
            )

            preprocessor_summary: dict[str, object] = {}
            if case_config is not None:
                assert resolved_zones is not None
                assert compiled_boundaries is not None
                assert quality_gate is not None
                preprocessor_summary = {
                    "case_id": case_config.case_id,
                    "schema_version": case_config.schema_version,
                    "zones": {
                        zone_id: {
                            "kind": zone.kind,
                            "entity_count": int(zone.entity_indices.size),
                            "physical_tags": list(zone.physical_tags),
                            "physical_names": list(zone.physical_names),
                        }
                        for zone_id, zone in resolved_zones.zones.items()
                    },
                    "unassigned_boundary_face_count": int(
                        resolved_zones.unassigned_boundary_faces.size
                    ),
                    "unassigned_cell_count": int(resolved_zones.unassigned_cells.size),
                    "boundary_conditions": [
                        {
                            "id": condition.id,
                            "zone": condition.zone,
                            "kind": condition.kind,
                            "face_count": int(condition.face_indices.size),
                            "parameters": condition.parameters,
                            "periodic_pair_count": len(condition.periodic_pairs),
                        }
                        for condition in compiled_boundaries.conditions
                    ],
                    "mesh_quality_gate": quality_gate.to_dict(),
                    "material_cell_counts": (
                        material_assignment.counts()
                        if material_assignment is not None
                        else {}
                    ),
                }
            import_is_valid = bool(
                report.is_valid and (quality_gate is None or quality_gate.is_acceptable)
            )
            summary = {
                "source_msh": str(msh_path),
                "original_input": original_input,
                "searched_paths": [str(path) for path in searched_paths],
                "gmsh_meshes_dir": str(GMSH_MESHES_DIR),
                "is_valid": import_is_valid,
                "error_count": len(report.errors),
                "warning_count": len(report.warnings),
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "summary": report.summary,
                "boundary_groups": {
                    "physical_group_names": report.summary.get(
                        "physical_group_names", []
                    ),
                    "boundary_tags": report.summary.get("boundary_tags", []),
                    "counts_per_tag": report.summary.get("boundary_tag_counts", {}),
                    "counts_per_semantic_class": report.summary.get(
                        "boundary_semantic_counts",
                        {},
                    ),
                    "tag_to_name": report.summary.get("boundary_tag_name_map", {}),
                    "tag_to_semantic": report.summary.get(
                        "boundary_tag_semantic_map",
                        {},
                    ),
                },
                "mesh_scale_audit": scale_audit,
                "preprocessor": preprocessor_summary,
                "artifacts": {
                    "run_log": str(run_dir / "run.log"),
                    "config_json": str(run_dir / "config.json"),
                    "summary_json": str(run_dir / "summary.json"),
                    "validation_report": str(run_dir / "validation_report.txt"),
                    "normalized_case_config_json": (
                        str(run_dir / "case_config.normalized.json")
                        if case_config is not None
                        else ""
                    ),
                    "mesh_npz": str(npz_path),
                    "volume_vtu": str(volume_vtu_path),
                    "boundary_vtp": boundary_vtp,
                    "preview_png": previews,
                },
            }
            _write_json(run_dir / "summary.json", summary)
            manifest_recorder.record_completed(
                inputs=manifest_inputs,
                outputs={
                    "mesh_npz": str(npz_path),
                    "summary_json": str(run_dir / "summary.json"),
                    "config_json": str(run_dir / "config.json"),
                    "validation_report_txt": str(run_dir / "validation_report.txt"),
                    "volume_vtu": str(volume_vtu_path),
                    "boundary_vtp": boundary_vtp,
                },
                artifacts=summary["artifacts"],
                metadata={
                    "is_valid": import_is_valid,
                    "error_count": int(len(report.errors)),
                    "warning_count": int(len(report.warnings)),
                    "mesh_scale_reference_ok": bool(
                        (known_reference or {}).get("matches_reference", True)
                    ),
                },
            )
            print(f"[gmsh-import] summary written: {run_dir / 'summary.json'}")
            print(
                f"[gmsh-import] imported nodes/tetra: {mesh.points.shape[0]}/{mesh.tetrahedra.shape[0]}"
            )
            print(
                "[gmsh-import] boundary faces inlet/outlet/wall/unresolved: "
                f"{mesh.inlet_faces.shape[0]}/{mesh.outlet_faces.shape[0]}/"
                f"{mesh.wall_faces.shape[0]}/{mesh.boundary_unresolved_faces.shape[0]}"
            )
            if args.fail_on_validation_error and not report.is_valid:
                raise SystemExit(
                    "[gmsh-import] validation failed. See validation_report.txt"
                )
            if quality_gate is not None and not quality_gate.is_acceptable:
                raise RuntimeError(
                    "Case mesh-quality gate failed. See summary.json preprocessor findings."
                )
    except Exception as exc:
        manifest_recorder.record_failed(
            error=exc,
            inputs=manifest_inputs,
            artifacts=manifest_artifacts,
        )
        raise


if __name__ == "__main__":
    main()
