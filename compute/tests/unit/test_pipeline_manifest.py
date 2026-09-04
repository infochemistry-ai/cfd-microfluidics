from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from experiments.gmsh._pipeline_manifest import build_pipeline_manifest_recorder
from microfluidics.pipeline.manifest import (
    PipelineManifestError,
    create_pipeline_manifest_file,
    list_pipeline_runs,
    load_pipeline_manifest,
    resolve_manifest_path_reference,
    upsert_pipeline_run,
)


def test_pipeline_manifest_supports_fan_out_graphs(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "results" / "demo_pipeline"
    pipeline_root.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_root / "pipeline_manifest.json"

    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id="demo_pipeline",
        metadata={"scenario": "gmsh_tj", "owner": "tests"},
    )

    import_run_dir = pipeline_root / "import_runs" / "20260621_150000_mesh"
    flow_run_dir = pipeline_root / "flow_runs" / "20260621_150100_flow"
    transport_a_dir = pipeline_root / "transport_runs" / "20260621_150200_transport_a"
    transport_b_dir = pipeline_root / "transport_runs" / "20260621_150300_transport_b"
    thermal_dir = pipeline_root / "thermal_runs" / "20260621_150400_thermal"
    chemistry_dir = pipeline_root / "chemistry_runs" / "20260621_150500_chemistry"

    upsert_pipeline_run(
        manifest_path,
        run_id="import-001",
        stage_type="import",
        status="completed",
        run_dir=import_run_dir,
        outputs={
            "mesh_npz": import_run_dir / "mesh_imported.npz",
            "summary_json": import_run_dir / "summary.json",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="flow-001",
        stage_type="flow",
        status="completed",
        run_dir=flow_run_dir,
        parent_run_ids=["import-001"],
        inputs={"mesh_npz": import_run_dir / "mesh_imported.npz"},
        outputs={
            "summary_json": flow_run_dir / "summary.json",
            "face_flux_npy": flow_run_dir / "final_corrected_face_flux.npy",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="transport-001",
        stage_type="transport",
        status="completed",
        run_dir=transport_a_dir,
        parent_run_ids=["flow-001"],
        inputs={
            "mesh_npz": import_run_dir / "mesh_imported.npz",
            "flow_run_dir": flow_run_dir,
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="transport-002",
        stage_type="transport",
        status="running",
        run_dir=transport_b_dir,
        parent_run_ids=["flow-001"],
        inputs={
            "mesh_npz": import_run_dir / "mesh_imported.npz",
            "flow_run_dir": flow_run_dir,
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="thermal-001",
        stage_type="thermal",
        status="pending",
        run_dir=thermal_dir,
        parent_run_ids=["flow-001"],
        inputs={
            "flow_summary_json": flow_run_dir / "summary.json",
            "flow_face_flux_npy": flow_run_dir / "final_corrected_face_flux.npy",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="chemistry-001",
        stage_type="chemistry",
        status="pending",
        run_dir=chemistry_dir,
        parent_run_ids=["flow-001", "transport-001"],
        metadata={"module_family": "future_extension"},
    )

    manifest = load_pipeline_manifest(manifest_path)
    assert manifest["pipeline_id"] == "demo_pipeline"
    assert manifest["pipeline_root"] == "."

    flow_run = manifest["runs"]["flow-001"]
    assert flow_run["parent_run_ids"] == ["import-001"]
    assert flow_run["run_dir"] == "flow_runs/20260621_150100_flow"
    assert (
        flow_run["outputs"]["summary_json"]
        == "flow_runs/20260621_150100_flow/summary.json"
    )

    transport_runs = list_pipeline_runs(manifest, stage_type="transport")
    assert [run["run_id"] for run in transport_runs] == [
        "transport-001",
        "transport-002",
    ]

    downstream_runs = list_pipeline_runs(manifest, parent_run_id="flow-001")
    assert {run["run_id"] for run in downstream_runs} == {
        "transport-001",
        "transport-002",
        "thermal-001",
        "chemistry-001",
    }

    resolved_transport_dir = resolve_manifest_path_reference(
        manifest_path,
        manifest["runs"]["transport-001"]["run_dir"],
    )
    assert resolved_transport_dir == transport_a_dir.resolve()


def test_pipeline_manifest_rejects_unknown_parent_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "pipeline_manifest.json"
    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id="unknown_parent_check",
    )

    with pytest.raises(PipelineManifestError, match="Unknown parent run id"):
        upsert_pipeline_run(
            manifest_path,
            run_id="transport-001",
            stage_type="transport",
            status="pending",
            parent_run_ids=["flow-404"],
        )


def test_resolve_manifest_path_reference_keeps_absolute_paths_absolute(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "pipeline" / "pipeline_manifest.json"
    absolute_target = (tmp_path.parent / "external-data" / "mesh.msh").resolve()

    resolved = resolve_manifest_path_reference(
        manifest_path,
        absolute_target.as_posix(),
    )

    assert resolved == absolute_target


def test_pipeline_manifest_recorder_fails_fast_when_manifest_path_is_invalid(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "manifest-blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    run_dir = tmp_path / "run-dir"
    run_dir.mkdir()

    recorder = build_pipeline_manifest_recorder(
        Namespace(
            pipeline_manifest=str(blocker / "pipeline_manifest.json"),
            pipeline_run_id="demo-run",
            pipeline_parent_run_id=[],
        ),
        stage_type="import",
        run_dir=run_dir,
    )

    with pytest.raises(OSError):
        recorder.record_started(inputs={"demo": "value"})
