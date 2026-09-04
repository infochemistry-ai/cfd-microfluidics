"""Integration test for reactive CLI staging and artifact contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.gmsh.run_import_gmsh_mesh import _save_npz
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "gmsh" / "run_gmsh_tetra_reactive_debug.py"
PIPELINE_RUNNER = REPO_ROOT / "experiments" / "gmsh" / "run_gmsh_pipeline_manifest.py"
EXAMPLE_CASE = (
    REPO_ROOT / "data" / "examples" / "reactive" / "t_mixer_exothermic_ab.json"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_flow_artifact_hashes(flow_dir: Path) -> None:
    summary_path = flow_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["flow_coupling_artifact_sha256"] = {
        name: _sha256_file(flow_dir / filename)
        for name, filename in {
            "flow_coupling_metadata_json": "flow_coupling_metadata.json",
            "final_corrected_face_flux_npy": "final_corrected_face_flux.npy",
            "face_to_cells_npy": "face_to_cells.npy",
            "cell_volumes_npy": "cell_volumes.npy",
        }.items()
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
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
        [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64
    )
    mesh = build_imported_tetra_mesh(
        source_path=tmp_path / "tiny.msh",
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=boundary_triangles,
        boundary_face_tags=np.asarray([1, 2, 3, 4], dtype=np.int32),
        field_data={
            "left_inlet": np.asarray([1, 2], dtype=np.int32),
            "right_inlet": np.asarray([2, 2], dtype=np.int32),
            "outlet": np.asarray([3, 2], dtype=np.int32),
            "wall": np.asarray([4, 2], dtype=np.int32),
            "fluid": np.asarray([5, 3], dtype=np.int32),
        },
    )
    mesh_path = _save_npz(mesh, tmp_path / "mesh.npz")

    flow_dir = tmp_path / "flow_run"
    flow_dir.mkdir()
    np.save(
        flow_dir / "final_corrected_face_flux.npy",
        np.zeros(mesh.face_vertices.shape[0], dtype=np.float64),
    )
    np.save(flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells))
    np.save(flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes))
    metadata = {
        "mesh_sha256": _sha256_file(mesh_path),
        "mesh_stats": {
            "tetra_count": int(mesh.tetrahedra.shape[0]),
            "face_count": int(mesh.face_vertices.shape[0]),
            "node_count": int(mesh.points.shape[0]),
        },
        "stage_status": {
            "ready_for_next_stage": True,
            "run_completed": True,
            "stage_status_reason": "integration fixture",
        },
    }
    metadata_path = flow_dir / "flow_coupling_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    artifact_paths = {
        "flow_coupling_metadata_json": metadata_path,
        "final_corrected_face_flux_npy": (flow_dir / "final_corrected_face_flux.npy"),
        "face_to_cells_npy": flow_dir / "face_to_cells.npy",
        "cell_volumes_npy": flow_dir / "cell_volumes.npy",
    }
    (flow_dir / "summary.json").write_text(
        json.dumps(
            {
                "mesh_sha256": _sha256_file(mesh_path),
                "artifacts": {
                    "flow_coupling_metadata_json": str(metadata_path),
                },
                "flow_coupling_artifact_sha256": {
                    name: _sha256_file(path) for name, path in artifact_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )

    case = json.loads(EXAMPLE_CASE.read_text(encoding="utf-8"))
    case["mode"] = "off"
    case["time"]["num_steps"] = 1
    case["time"]["dt_mode"] = "manual"
    case["time"]["dt_s"] = 1.0e-4
    case["output"]["history_stride"] = 1
    case_path = tmp_path / "reactive_case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    return mesh_path, flow_dir, case_path


def test_reactive_cli_rejects_same_counts_with_different_topology_and_hash(
    tmp_path: Path,
) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    with np.load(mesh_path, allow_pickle=False) as source:
        mismatched_arrays = {name: np.array(source[name], copy=True) for name in source}
    mismatched_arrays["face_to_cells"][0, 0] = -1
    mismatched_mesh_path = tmp_path / "mismatched_mesh.npz"
    np.savez_compressed(mismatched_mesh_path, **mismatched_arrays)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh-npz",
            str(mismatched_mesh_path),
            "--flow-run-dir",
            str(flow_dir),
            "--reactive-case",
            str(case_path),
            "--output-root",
            str(tmp_path / "reactive"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "reactive mesh/flow compatibility validation failed" in completed.stderr
    assert '"mesh_hash_ok": false' in completed.stderr
    assert '"face_to_cells_equal": false' in completed.stderr


@pytest.mark.parametrize(
    "artifact_name",
    ["flow_coupling_metadata.json", "final_corrected_face_flux.npy"],
)
def test_reactive_cli_rejects_flow_artifact_hash_mismatch(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    artifact_path = flow_dir / artifact_name
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh-npz",
            str(mesh_path),
            "--flow-run-dir",
            str(flow_dir),
            "--reactive-case",
            str(case_path),
            "--output-root",
            str(tmp_path / "reactive"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "flow coupling artifact SHA-256 mismatch" in completed.stderr


def test_reactive_cli_writes_complete_no_pickle_artifacts(tmp_path: Path) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    output_root = tmp_path / "reactive"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh-npz",
            str(mesh_path),
            "--flow-run-dir",
            str(flow_dir),
            "--reactive-case",
            str(case_path),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["contract_version"] == "reactive_transport_v1"
    assert summary["readiness"]["run_completed"] is True
    assert summary["readiness"]["numerically_stable"] is True
    assert summary["transport"]["physical_clipping_used"] is False
    assert (run_dir / "reactive_result.vtu").is_file()
    assert (run_dir / "history.json").is_file()
    assert (run_dir / "normalized_reactive_case.json").is_file()
    assert (run_dir / "provenance.json").is_file()
    assert (run_dir / "validation_report.txt").is_file()

    with np.load(run_dir / "reactive_fields.npz", allow_pickle=False) as fields:
        assert fields["concentrations_mol_per_m3"].shape == (1, 3)
        assert fields["temperature_k"].shape == (1,)
        assert fields["species_sources_mol_per_m3_s"].shape == (1, 3)
        np.testing.assert_array_equal(
            fields["species_sources_mol_per_m3_s"], np.zeros((1, 3))
        )
        np.testing.assert_array_equal(fields["heat_release_w_per_m3"], np.zeros(1))
        assert fields["species_names_json"].dtype.kind == "U"


def test_manifest_reactive_mode_rejects_ordinary_transport_options(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_RUNNER),
            "--run-root",
            str(tmp_path / "pipeline"),
            "--msh",
            "placeholder.msh",
            "--reactive-case",
            str(EXAMPLE_CASE),
            "--transport-steps",
            "1",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "mutually exclusive" in completed.stderr


def test_incomplete_reactive_run_is_failed_in_manifest(tmp_path: Path) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    flux_path = flow_dir / "final_corrected_face_flux.npy"
    np.save(flux_path, np.ones_like(np.load(flux_path)))
    _refresh_flow_artifact_hashes(flow_dir)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["time"]["dt_s"] = 1.0
    case["time"]["max_transport_substeps"] = 1
    case_path.write_text(json.dumps(case), encoding="utf-8")
    manifest_path = tmp_path / "pipeline" / "pipeline_manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh-npz",
            str(mesh_path),
            "--flow-run-dir",
            str(flow_dir),
            "--reactive-case",
            str(case_path),
            "--output-root",
            str(tmp_path / "reactive"),
            "--pipeline-manifest",
            str(manifest_path),
            "--pipeline-run-id",
            "reactive-incomplete",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["runs"]["reactive-incomplete"]
    assert run["status"] == "failed"
    assert run["metadata"]["status"] == "blocked_transport_substep_cap"


@pytest.mark.parametrize(
    ("allow_unready_result", "expected_returncode", "expected_manifest_status"),
    [(False, 2, "failed"), (True, 0, "completed")],
)
def test_physically_unready_result_requires_explicit_diagnostic_override(
    tmp_path: Path,
    allow_unready_result: bool,
    expected_returncode: int,
    expected_manifest_status: str,
) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    metadata_path = flow_dir / "flow_coupling_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stage_status"]["ready_for_next_stage"] = False
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_flow_artifact_hashes(flow_dir)
    manifest_path = tmp_path / "pipeline" / "pipeline_manifest.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--mesh-npz",
        str(mesh_path),
        "--flow-run-dir",
        str(flow_dir),
        "--reactive-case",
        str(case_path),
        "--output-root",
        str(tmp_path / "reactive"),
        "--allow-unready-flow",
        "--pipeline-manifest",
        str(manifest_path),
        "--pipeline-run-id",
        "reactive-unready",
    ]
    if allow_unready_result:
        command.append("--allow-physically-unready-result")

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest["runs"]["reactive-unready"]
    assert run["status"] == expected_manifest_status
    summary_path = next((tmp_path / "reactive").glob("*/summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["readiness"]["run_completed"] is True
    assert summary["readiness"]["physically_ready"] is False
    assert summary["readiness"]["ready_for_next_stage"] is False
    assert summary["readiness"]["cli_success"] is allow_unready_result
    assert summary["readiness"]["status"] == "completed_not_ready"


def test_reactive_cli_soft_walltime_writes_incomplete_artifacts(
    tmp_path: Path,
) -> None:
    mesh_path, flow_dir, case_path = _write_inputs(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh-npz",
            str(mesh_path),
            "--flow-run-dir",
            str(flow_dir),
            "--reactive-case",
            str(case_path),
            "--output-root",
            str(tmp_path / "reactive"),
            "--max-walltime-seconds",
            "1e-300",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2, completed.stderr
    summary_path = next((tmp_path / "reactive").glob("*/summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["executed_steps"] == 0
    assert summary["readiness"]["status"] == "blocked_walltime_limit"
    assert summary["readiness"]["cli_success"] is False
