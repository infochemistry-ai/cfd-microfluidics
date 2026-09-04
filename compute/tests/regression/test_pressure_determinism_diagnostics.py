from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from microfluidics.gmsh.tetra.pressure_determinism_diagnostics import (
    array_summary,
    compare_arrays,
    configure_cuda_determinism,
    gauge_adjusted_comparison,
    load_fixed_work_state,
    sha256_file,
    write_json,
)


def _mesh() -> SimpleNamespace:
    return SimpleNamespace(
        face_vertices=np.zeros((5, 3), dtype=np.int64),
        tetrahedra=np.zeros((2, 4), dtype=np.int64),
    )


def _source(tmp_path, mesh_path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "resolved_mesh_npz": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "final_corrected_face_flux.npy", np.zeros(5, dtype=np.float64))
    np.save(tmp_path / "final_cell_velocity.npy", np.zeros((2, 3), dtype=np.float64))
    np.save(tmp_path / "final_pressure.npy", np.zeros(2, dtype=np.float64))


def test_off_determinism_does_not_import_or_change_torch_settings(monkeypatch) -> None:
    def fail_import(name: str):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("builtins.__import__", fail_import)
    report = configure_cuda_determinism("off")
    assert report["requested_mode"] == "off"
    assert report["effective_mode"] == "off"


def test_fixed_work_manifest_hashes_and_validates_state(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    mesh_path.write_bytes(b"mesh identity")
    source = tmp_path / "source"
    source.mkdir()
    _source(source, mesh_path)

    state, manifest = load_fixed_work_state(
        source_run_dir=source, mesh=_mesh(), mesh_npz=mesh_path
    )

    assert state.face_flux.dtype == np.float64
    assert manifest["mesh_identity_equal"] is True
    assert manifest["source_mesh_path_matches_recorded"] is True
    assert manifest["input_arrays"]["pressure"]["shape"] == [2]
    assert len(manifest["input_arrays"]["face_flux"]["sha256"]) == 64


def test_fixed_work_rejects_wrong_shapes_and_mesh_identity(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    mesh_path.write_bytes(b"mesh identity")
    source = tmp_path / "source"
    source.mkdir()
    _source(source, mesh_path)
    np.save(source / "final_pressure.npy", np.zeros(3, dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        load_fixed_work_state(source_run_dir=source, mesh=_mesh(), mesh_npz=mesh_path)

    _source(source, mesh_path)
    other_mesh = tmp_path / "other_mesh.npz"
    other_mesh.write_bytes(b"different mesh")
    with pytest.raises(ValueError, match="does not match"):
        load_fixed_work_state(source_run_dir=source, mesh=_mesh(), mesh_npz=other_mesh)


def test_fixed_work_rejects_mesh_replaced_at_recorded_source_path(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    mesh_path.write_bytes(b"original mesh identity")
    source = tmp_path / "source"
    source.mkdir()
    _source(source, mesh_path)

    mesh_path.write_bytes(b"replacement mesh identity")

    with pytest.raises(ValueError, match="recorded at source-run time"):
        load_fixed_work_state(
            source_run_dir=source,
            mesh=_mesh(),
            mesh_npz=mesh_path,
        )


def test_fixed_work_rejects_source_without_recorded_mesh_hash(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    mesh_path.write_bytes(b"mesh identity")
    source = tmp_path / "source"
    source.mkdir()
    _source(source, mesh_path)
    (source / "summary.json").write_text(
        json.dumps({"resolved_mesh_npz": str(mesh_path)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lacks a valid recorded mesh_sha256"):
        load_fixed_work_state(
            source_run_dir=source,
            mesh=_mesh(),
            mesh_npz=mesh_path,
        )


def test_fixed_work_accepts_explicit_hash_for_legacy_source(tmp_path) -> None:
    mesh_path = tmp_path / "mesh.npz"
    mesh_path.write_bytes(b"mesh identity")
    source = tmp_path / "source"
    source.mkdir()
    _source(source, mesh_path)
    (source / "summary.json").write_text(
        json.dumps({"resolved_mesh_npz": "/previous/location/mesh.npz"}),
        encoding="utf-8",
    )

    _, manifest = load_fixed_work_state(
        source_run_dir=source,
        mesh=_mesh(),
        mesh_npz=mesh_path,
        legacy_source_mesh_sha256=sha256_file(mesh_path),
    )

    assert manifest["mesh_identity_equal"] is True
    assert manifest["source_mesh_sha256_provenance"] == "explicit_legacy_override"
    assert manifest["source_mesh_path_matches_recorded"] is None


def test_array_comparison_handles_zero_reference_and_nonfinite_json(tmp_path) -> None:
    comparison = compare_arrays(np.zeros(2), np.array([3.0, 4.0]))
    assert comparison["relative_l2"] == 5.0
    summary = array_summary(np.array([np.nan, np.inf]))
    assert summary["finite_count"] == 0
    output = tmp_path / "report.json"
    write_json(
        output,
        {
            "summary": summary,
            "bad": float("nan"),
            "nested": {
                "numpy_scalar": np.float64("inf"),
                "numpy_array": np.array([np.float64("nan"), np.float64("-inf")]),
            },
        },
    )
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["summary"]["min"] is None
    assert parsed["bad"] is None
    assert parsed["nested"] == {"numpy_scalar": None, "numpy_array": [None, None]}


def test_array_comparison_aggregates_only_finite_differences() -> None:
    comparison = compare_arrays(
        np.array([3.0, 4.0, np.inf, 1.0]), np.array([6.0, 8.0, np.inf, np.inf])
    )

    assert comparison["finite_count"] == 2
    assert comparison["nan_count"] == 1
    assert comparison["inf_count"] == 1
    assert comparison["max_abs"] == 4.0
    assert comparison["mean_abs"] == 3.5
    assert comparison["relative_l2"] == 1.0


def test_array_comparison_reports_no_aggregates_without_finite_differences() -> None:
    comparison = compare_arrays(np.array([np.nan]), np.array([np.nan]))

    assert comparison["finite_count"] == 0
    assert comparison["nan_count"] == 1
    assert comparison["inf_count"] == 0
    assert comparison["max_abs"] is None
    assert comparison["mean_abs"] is None
    assert comparison["relative_l2"] is None


def test_gauge_adjustment_uses_only_paired_finite_values() -> None:
    comparison = gauge_adjusted_comparison(
        np.array([1.0, 2.0, np.inf]),
        np.array([2.0, 4.0, np.inf]),
    )

    assert comparison["removed_constant_offset"] == 1.5
    assert comparison["finite_count"] == 2
    assert comparison["nan_count"] == 1
    assert comparison["max_abs"] == 0.5
