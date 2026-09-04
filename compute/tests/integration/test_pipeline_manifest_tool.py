from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from microfluidics.pipeline import load_pipeline_manifest
from microfluidics.pipeline.manifest import (
    create_pipeline_manifest_file,
    get_pipeline_run,
    resolve_manifest_path_reference,
    upsert_pipeline_run,
)
from microfluidics.gmsh.gmsh_mesh_import import build_imported_tetra_mesh
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
    build_face_normal_flux_from_velocity,
)
from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (
    build_prescribed_velocity_field,
)
from experiments.gmsh._path_utils import option_was_explicitly_provided

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HELPER_SCRIPT = (
    PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_pipeline_manifest_tool.py"
)


def test_option_explicitness_supports_split_and_equals_forms() -> None:
    raw_argv = [
        "--msh=mesh.msh",
        "--startup-bootstrap-max-steps",
        "12",
        "--pressure-solver=pcg_diag",
        "--viscous-predictor-mode",
        "none",
        "--flow-dt-min=1e-7",
    ]

    for option in (
        "--msh",
        "--startup-bootstrap-max-steps",
        "--pressure-solver",
        "--viscous-predictor-mode",
        "--flow-dt-min",
    ):
        assert option_was_explicitly_provided(raw_argv, option)


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


def _load_helper_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_gmsh_pipeline_manifest_tool",
        HELPER_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_manifest_with_ready_flow(run_root: Path) -> Path:
    manifest_path = run_root / "pipeline_manifest.json"
    import_run_dir = run_root / "import_runs" / "mesh"
    flow_run_dir = run_root / "flow_runs" / "flow-ready-001"
    import_run_dir.mkdir(parents=True, exist_ok=True)
    flow_run_dir.mkdir(parents=True, exist_ok=True)

    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id=run_root.name,
        pipeline_root=run_root,
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="import-001",
        stage_type="import",
        status="completed",
        run_dir=import_run_dir,
        inputs={
            "original_input": "t_junction.msh",
            "resolved_msh_path": PROJECT_ROOT / "data/meshes/gmsh" / "t_junction.msh",
        },
        outputs={
            "mesh_npz": import_run_dir / "t_junction_imported_mesh.npz",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="flow-ready-001",
        stage_type="flow",
        status="completed",
        run_dir=flow_run_dir,
        parent_run_ids=["import-001"],
        metadata={
            "mesh_name": "t_junction",
            "run_completed": True,
            "ready_for_next_stage": True,
            "ready_for_long_run": False,
            "stage_status_reason": "ready for downstream reuse",
        },
        outputs={
            "mesh_npz": flow_run_dir / "t_junction_imported_mesh.npz",
            "summary_json": flow_run_dir / "summary.json",
            "final_corrected_face_flux_npy": flow_run_dir
            / "final_corrected_face_flux.npy",
        },
    )
    return manifest_path


def _structured_tetra_mesh() -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = 3, 3, 3
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    zs = np.linspace(0.0, 1.0, nz)
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)

    rng = np.random.default_rng(0)
    for idx, (x, y, z) in enumerate(points):
        interior = (
            1e-9 < x < 1.0 - 1e-9 and 1e-9 < y < 1.0 - 1e-9 and 1e-9 < z < 1.0 - 1e-9
        )
        if interior:
            points[idx] += 0.3 * (rng.random(3) - 0.5)

    def node(i: int, j: int, k: int) -> int:
        return k * (nx * ny) + j * nx + i

    tets: list[list[int]] = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v000 = node(i, j, k)
                v100 = node(i + 1, j, k)
                v010 = node(i, j + 1, k)
                v110 = node(i + 1, j + 1, k)
                v001 = node(i, j, k + 1)
                v101 = node(i + 1, j, k + 1)
                v011 = node(i, j + 1, k + 1)
                v111 = node(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                    ]
                )
    return points, np.asarray(tets, dtype=np.int64)


def _extract_boundary_triangles(
    points: np.ndarray,
    tetrahedra: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    for tet in tetrahedra:
        faces = (
            (int(tet[1]), int(tet[2]), int(tet[3])),
            (int(tet[0]), int(tet[3]), int(tet[2])),
            (int(tet[0]), int(tet[1]), int(tet[3])),
            (int(tet[0]), int(tet[2]), int(tet[1])),
        )
        for tri in faces:
            face_map.setdefault(tuple(sorted(tri)), []).append(tri)

    boundary_triangles: list[list[int]] = []
    boundary_tags: list[int] = []
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    eps = 1e-12
    for entries in face_map.values():
        if len(entries) != 1:
            continue
        tri = entries[0]
        center = np.mean(points[np.asarray(tri, dtype=np.int64)], axis=0)
        x, y, _ = center.tolist()
        if y <= y_min + eps:
            tag = 1 if x <= 0.5 else 2
        elif y >= y_max - eps:
            tag = 3
        else:
            tag = 4
        boundary_triangles.append([tri[0], tri[1], tri[2]])
        boundary_tags.append(tag)

    field_data = {
        "left_inlet": np.asarray([1, 2], dtype=np.int32),
        "right_inlet": np.asarray([2, 2], dtype=np.int32),
        "outlet": np.asarray([3, 2], dtype=np.int32),
        "walls": np.asarray([4, 2], dtype=np.int32),
    }
    return (
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(boundary_tags, dtype=np.int32),
        field_data,
    )


def _build_synthetic_mesh():
    points, tetrahedra = _structured_tetra_mesh()
    btri, btags, field_data = _extract_boundary_triangles(points, tetrahedra)
    return build_imported_tetra_mesh(
        source_path=Path("synthetic.msh"),
        points=points,
        tetrahedra=tetrahedra,
        boundary_triangles=btri,
        boundary_face_tags=btags,
        field_data=field_data,
    )


def _save_mesh_npz(mesh, out_path: Path) -> None:
    np.savez(
        out_path,
        source_path=str(mesh.source_path),
        points=np.asarray(mesh.points, dtype=np.float64),
        tetrahedra=np.asarray(mesh.tetrahedra, dtype=np.int64),
        boundary_triangles=np.asarray(mesh.boundary_triangles, dtype=np.int64),
        boundary_face_tags=np.asarray(mesh.boundary_face_tags, dtype=np.int32),
        cell_centers=np.asarray(mesh.cell_centers, dtype=np.float64),
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
        face_vertices=np.asarray(mesh.face_vertices, dtype=np.int64),
        face_centers=np.asarray(mesh.face_centers, dtype=np.float64),
        face_areas=np.asarray(mesh.face_areas, dtype=np.float64),
        face_normals=np.asarray(mesh.face_normals, dtype=np.float64),
        face_to_cells=np.asarray(mesh.face_to_cells, dtype=np.int64),
        cell_to_faces=np.asarray(mesh.cell_to_faces, dtype=np.int64),
        boundary_tag_per_face=np.asarray(mesh.boundary_tag_per_face, dtype=np.int32),
        interior_face_indices=np.asarray(mesh.interior_face_indices, dtype=np.int64),
        boundary_face_indices=np.asarray(mesh.boundary_face_indices, dtype=np.int64),
        inlet_faces=np.asarray(mesh.inlet_faces, dtype=np.int64),
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
        unresolved_faces=np.asarray(mesh.boundary_unresolved_faces, dtype=np.int64),
        boundary_face_names_json=json.dumps(
            {str(k): str(v) for k, v in mesh.boundary_face_names.items()}
        ),
        physical_groups_json=json.dumps(
            {str(k): [int(v[0]), int(v[1])] for k, v in mesh.physical_groups.items()}
        ),
        diagnostics_json=json.dumps({}),
    )


def _build_external_flow_fixture(
    tmp_path: Path,
    *,
    ready: bool,
) -> tuple[Path, Path]:
    mesh = _build_synthetic_mesh()
    import_root = tmp_path / "imports"
    import_root.mkdir(parents=True, exist_ok=True)
    mesh_npz = import_root / "synthetic_imported_mesh.npz"
    _save_mesh_npz(mesh, mesh_npz)

    velocity = build_prescribed_velocity_field(
        mesh,
        field_name="two_inlets_to_outlet_tj_balanced",
        inlet_speed=0.15,
    )
    face_normal_velocity, _ = build_face_normal_flux_from_velocity(
        mesh,
        velocity.cell_velocity,
        boundary_face_velocity_overrides=velocity.boundary_face_velocity_overrides,
        left_inlet_faces=velocity.boundary_groups["left_inlet_faces"],
        right_inlet_faces=velocity.boundary_groups["right_inlet_faces"],
        outlet_faces=velocity.boundary_groups["outlet_faces"],
        wall_faces=velocity.boundary_groups["wall_faces"],
    )
    face_flux = np.asarray(face_normal_velocity, dtype=np.float64) * np.asarray(
        mesh.face_areas, dtype=np.float64
    )

    flow_dir = tmp_path / ("flow_ready" if ready else "flow_unready")
    flow_dir.mkdir(parents=True, exist_ok=True)
    np.save(flow_dir / "final_corrected_face_flux.npy", face_flux)
    np.save(
        flow_dir / "face_to_cells.npy", np.asarray(mesh.face_to_cells, dtype=np.int64)
    )
    np.save(
        flow_dir / "cell_volumes.npy", np.asarray(mesh.cell_volumes, dtype=np.float64)
    )
    np.save(
        flow_dir / "final_cell_velocity.npy",
        np.asarray(velocity.cell_velocity, dtype=np.float64),
    )

    stage_status = {
        "run_completed": True,
        "numerically_stable": bool(ready),
        "physically_ready": bool(ready),
        "ready_for_next_stage": bool(ready),
        "ready_for_long_run": bool(ready),
        "stage_status_reason": "unit-ready" if ready else "unit-unready",
    }
    (flow_dir / "flow_coupling_metadata.json").write_text(
        json.dumps(
            {
                "ready_for_flow_to_transport_coupling": bool(ready),
                "ready_for_next_stage": bool(ready),
                "nonphysical_flux_fix_used": False,
                "convective_auto_damping_used_any": False,
                "stage_status_reason": stage_status["stage_status_reason"],
                "stage_status": stage_status,
                "mesh_stats": {
                    "tetra_count": int(mesh.tetrahedra.shape[0]),
                    "face_count": int(mesh.face_vertices.shape[0]),
                    "node_count": int(mesh.points.shape[0]),
                },
            }
        ),
        encoding="utf-8",
    )
    (flow_dir / "config.json").write_text(
        json.dumps({"mesh_name": "synthetic", "inlet_speed": 0.15}, indent=2),
        encoding="utf-8",
    )
    (flow_dir / "acceptance_report.json").write_text(
        json.dumps(
            {
                "projection_solved": True,
                "pressure_linear_accepted": True,
                "flow_progression_solved": bool(ready),
                "ready_for_flow_to_transport_coupling": bool(ready),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (flow_dir / "summary.json").write_text(
        json.dumps(
            {
                "mesh_name": "synthetic",
                "projection_solved": True,
                "pressure_linear_accepted": True,
                "flow_progression_solved": bool(ready),
                "flow_solved": bool(ready),
                "ready_for_next_stage": bool(ready),
                "resolved_mesh_npz": str(mesh_npz.resolve()),
                "artifacts": {
                    "flow_coupling_metadata_json": str(
                        (flow_dir / "flow_coupling_metadata.json").resolve()
                    ),
                    "config_json": str((flow_dir / "config.json").resolve()),
                    "acceptance_report_json": str(
                        (flow_dir / "acceptance_report.json").resolve()
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return flow_dir, mesh_npz


def test_manifest_tool_lists_ready_flows_and_shows_lineage(tmp_path: Path) -> None:
    pipeline_root = tmp_path / "results" / "manifest_tool_listing"
    manifest_path = pipeline_root / "pipeline_manifest.json"
    import_dir = pipeline_root / "import_runs" / "mesh"
    flow_ready_dir = pipeline_root / "flow_runs" / "flow_ready"
    flow_unready_dir = pipeline_root / "flow_runs" / "flow_unready"
    transport_dir = pipeline_root / "transport_runs" / "transport_ready_child"

    for path in (import_dir, flow_ready_dir, flow_unready_dir, transport_dir):
        path.mkdir(parents=True, exist_ok=True)

    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id="manifest_tool_listing",
        pipeline_root=pipeline_root,
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="import-001",
        stage_type="import",
        status="completed",
        run_dir=import_dir,
        inputs={
            "original_input": "t_junction.msh",
            "resolved_msh_path": PROJECT_ROOT / "data/meshes/gmsh" / "t_junction.msh",
        },
        outputs={"mesh_npz": import_dir / "t_junction_imported_mesh.npz"},
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="flow-ready-001",
        stage_type="flow",
        status="completed",
        run_dir=flow_ready_dir,
        parent_run_ids=["import-001"],
        metadata={
            "mesh_name": "t_junction",
            "run_completed": True,
            "ready_for_next_stage": True,
            "ready_for_long_run": False,
            "stage_status_reason": "ready for downstream reuse",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="flow-unready-001",
        stage_type="flow",
        status="completed",
        run_dir=flow_unready_dir,
        parent_run_ids=["import-001"],
        metadata={
            "mesh_name": "t_junction",
            "run_completed": True,
            "ready_for_next_stage": False,
            "ready_for_long_run": False,
            "stage_status_reason": "coupling gate not satisfied",
        },
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="transport-001",
        stage_type="transport",
        status="completed",
        run_dir=transport_dir,
        parent_run_ids=["flow-ready-001"],
    )

    list_completed = _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "list-flows",
            "--pipeline-manifest",
            str(manifest_path),
            "--json",
        ]
    )
    flow_rows = json.loads(list_completed.stdout)
    assert [row["run_id"] for row in flow_rows] == [
        "flow-ready-001",
        "flow-unready-001",
    ]
    assert flow_rows[0]["mesh"] == "t_junction"
    assert flow_rows[0]["ready_for_next_stage"] is True
    assert flow_rows[1]["ready_for_next_stage"] is False

    ready_completed = _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "list-ready-flows",
            "--pipeline-manifest",
            str(manifest_path),
            "--json",
        ]
    )
    ready_rows = json.loads(ready_completed.stdout)
    assert ready_rows == [
        {
            "mesh": "t_junction",
            "ready_for_long_run": False,
            "ready_for_next_stage": True,
            "run_dir": "flow_runs/flow_ready",
            "run_id": "flow-ready-001",
            "stage_status_reason": "ready for downstream reuse",
            "status": "completed",
        }
    ]

    show_completed = _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "show-run",
            "--pipeline-manifest",
            str(manifest_path),
            "flow-ready-001",
        ]
    )
    payload = json.loads(show_completed.stdout)
    assert payload["pipeline_id"] == "manifest_tool_listing"
    assert payload["run"]["stage_type"] == "flow"
    assert payload["run"]["parent_run_ids"] == ["import-001"]
    assert payload["child_run_ids"] == ["transport-001"]
    assert payload["readiness"]["ready_for_next_stage"] is True
    assert payload["readiness"]["stage_status_reason"] == "ready for downstream reuse"


def test_manifest_tool_run_flow_builds_expected_pipeline_runner_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    helper_module = _load_helper_module()
    run_root = tmp_path / "results" / "manifest_tool_run_flow"
    manifest_path = run_root / "pipeline_manifest.json"
    import_run_dir = run_root / "import_runs" / "mesh"
    import_run_dir.mkdir(parents=True, exist_ok=True)
    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id="manifest_tool_run_flow",
        pipeline_root=run_root,
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="import-001",
        stage_type="import",
        status="completed",
        run_dir=import_run_dir,
        inputs={
            "original_input": "t_junction.msh",
            "resolved_msh_path": PROJECT_ROOT / "data/meshes/gmsh" / "t_junction.msh",
        },
        outputs={"mesh_npz": import_run_dir / "t_junction_imported_mesh.npz"},
    )

    captured: list[list[str]] = []

    def _capture(cmd: list[str], *, dry_run: bool) -> int:
        assert dry_run is False
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(helper_module, "_run_subprocess", _capture)
    parser = helper_module.build_parser()
    args = parser.parse_args(
        [
            "run-flow",
            "--pipeline-manifest",
            str(manifest_path),
            "import-001",
            "--python-exe",
            sys.executable,
            "--flow-backend",
            "numpy",
            "--flow-mode",
            "stokes_viscous_projection",
            "--flow-steps",
            "1",
            "--flow-inlet-speed",
            "0.25",
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
            "--flow-wall-velocity-boundary-mode",
            "no_slip",
            "--flow-wall-tangential-no-slip-strength-ramp-start",
            "0.0",
            "--flow-wall-tangential-no-slip-strength-ramp-steps",
            "5",
            "--flow-viscous-face-flux-divergence-impact-cap",
            "0.02",
            "--flow-pressure-projection-outlet-contract-mode",
            "preserve",
            "--flow-projection-cell-velocity-update-mode",
            "momentum_pressure_corrected",
            "--flow-pressure-nonorthogonal-correction-mode",
            "deferred_lsq",
            "--flow-viscous-nonorthogonal-correction-mode",
            "deferred_lsq",
            "--flow-pressure-nonorthogonal-correction-sweeps",
            "3",
            "--flow-pressure-nonorthogonal-correction-relaxation",
            "0.6",
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
            "--no-flow-wall-flux-stokes-resistance-enabled",
        ]
    )

    assert int(args.func(args)) == 0
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:5] == [
        sys.executable,
        str(helper_module.PIPELINE_RUNNER_SCRIPT),
        "--run-root",
        str(run_root.resolve()),
        "--from-import-run-id",
    ]
    assert "import-001" in cmd
    assert "--skip-transport" in cmd
    assert "--flow-backend" in cmd
    assert "--flow-mode" in cmd
    assert cmd[cmd.index("--flow-mode") + 1] == "stokes_viscous_projection"
    assert "--flow-dt-mode" in cmd
    assert cmd[cmd.index("--flow-dt-mode") + 1] == "manual"
    assert "--flow-dt" in cmd
    assert cmd[cmd.index("--flow-dt") + 1] == "2e-05"
    assert "--flow-dt-min" in cmd
    assert cmd[cmd.index("--flow-dt-min") + 1] == "2e-07"
    assert "--flow-dt-max" in cmd
    assert cmd[cmd.index("--flow-dt-max") + 1] == "4e-05"
    assert "--flow-convective-cfl-target" in cmd
    assert cmd[cmd.index("--flow-convective-cfl-target") + 1] == "0.25"
    assert "--flow-wall-velocity-boundary-mode" in cmd
    assert cmd[cmd.index("--flow-wall-velocity-boundary-mode") + 1] == "no_slip"
    assert "--flow-wall-tangential-no-slip-strength-ramp-start" in cmd
    assert (
        cmd[cmd.index("--flow-wall-tangential-no-slip-strength-ramp-start") + 1]
        == "0.0"
    )
    assert "--flow-wall-tangential-no-slip-strength-ramp-steps" in cmd
    assert cmd[cmd.index("--flow-wall-tangential-no-slip-strength-ramp-steps") + 1] == (
        "5"
    )
    assert "--flow-viscous-face-flux-divergence-impact-cap" in cmd
    assert (
        cmd[cmd.index("--flow-viscous-face-flux-divergence-impact-cap") + 1] == "0.02"
    )
    assert "--flow-pressure-projection-outlet-contract-mode" in cmd
    assert (
        cmd[cmd.index("--flow-pressure-projection-outlet-contract-mode") + 1]
        == "preserve"
    )
    assert "--flow-projection-cell-velocity-update-mode" in cmd
    assert (
        cmd[cmd.index("--flow-projection-cell-velocity-update-mode") + 1]
        == "momentum_pressure_corrected"
    )
    assert "--flow-pressure-nonorthogonal-correction-mode" in cmd
    assert (
        cmd[cmd.index("--flow-pressure-nonorthogonal-correction-mode") + 1]
        == "deferred_lsq"
    )
    assert "--flow-viscous-nonorthogonal-correction-mode" in cmd
    assert (
        cmd[cmd.index("--flow-viscous-nonorthogonal-correction-mode") + 1]
        == "deferred_lsq"
    )
    assert "--flow-pressure-nonorthogonal-correction-sweeps" in cmd
    assert cmd[cmd.index("--flow-pressure-nonorthogonal-correction-sweeps") + 1] == "3"
    assert "--flow-pressure-nonorthogonal-correction-relaxation" in cmd
    assert (
        cmd[cmd.index("--flow-pressure-nonorthogonal-correction-relaxation") + 1]
        == "0.6"
    )
    assert "--flow-convective-stabilization-mode" in cmd
    assert cmd[cmd.index("--flow-convective-stabilization-mode") + 1] == "substepping"
    assert "--no-flow-disable-convective-auto-damping" in cmd
    assert "--flow-max-pressure-iterations" in cmd
    assert cmd[cmd.index("--flow-max-pressure-iterations") + 1] == "77"
    assert "--flow-pressure-relative-tolerance" in cmd
    assert cmd[cmd.index("--flow-pressure-relative-tolerance") + 1] == "0.002"
    assert "--flow-pressure-solver" in cmd
    assert cmd[cmd.index("--flow-pressure-solver") + 1] == "cg"
    assert "--flow-startup-warning-steps" in cmd
    assert cmd[cmd.index("--flow-startup-warning-steps") + 1] == "4"
    assert "--no-flow-wall-flux-stokes-resistance-enabled" in cmd


def test_manifest_tool_run_flow_parser_defaults_match_stage2_profile() -> None:
    helper_module = _load_helper_module()
    parser = helper_module.build_parser()
    args = parser.parse_args(["run-flow", "import-001"])

    assert args.flow_mode == "navier_stokes_projection_debug"
    assert args.flow_steps == 700
    assert args.flow_inlet_speed == 0.15
    assert args.flow_dt_mode == "auto_cfl"
    assert args.flow_dt is None
    assert args.flow_dt_min == 1e-7
    assert args.flow_dt_max is None
    assert args.flow_convective_cfl_target == 0.5
    assert args.flow_wall_velocity_boundary_mode == "slip"
    assert args.flow_viscous_face_flux_divergence_impact_cap == 0.03
    assert args.flow_viscous_predictor_outlet_contract_mode == ""
    assert args.flow_pressure_projection_outlet_contract_mode == ""
    assert args.flow_projection_cell_velocity_update_mode == ""
    assert args.flow_pressure_nonorthogonal_correction_mode == ""
    assert args.flow_viscous_nonorthogonal_correction_mode == ""
    assert args.flow_pressure_nonorthogonal_correction_sweeps is None
    assert args.flow_pressure_nonorthogonal_correction_relaxation is None
    assert args.flow_convective_stabilization_mode == "auto_damping"
    assert args.flow_disable_convective_auto_damping is True
    assert args.flow_max_pressure_iterations == 1000
    assert args.flow_pressure_relative_tolerance == 1e-4
    assert args.flow_pressure_solver == "pcg_diag"
    assert args.flow_startup_warning_steps == 10


def test_manifest_tool_run_thermal_parser_defaults_match_stage4_profile() -> None:
    helper_module = _load_helper_module()
    parser = helper_module.build_parser()
    args = parser.parse_args(["run-thermal", "flow-001"])

    assert args.thermal_steps == 20000
    assert args.thermal_dt_mode == "auto"
    assert args.thermal_dt == 5e-4
    assert args.thermal_cfl_target == 0.5
    assert args.thermal_cfl_limit == 0.8
    assert args.thermal_heat_source == 0.0


def test_manifest_tool_run_transport_and_thermal_build_expected_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    helper_module = _load_helper_module()
    run_root = tmp_path / "results" / "manifest_tool_run_downstream"
    manifest_path = _build_manifest_with_ready_flow(run_root)
    captured: list[list[str]] = []

    def _capture(cmd: list[str], *, dry_run: bool) -> int:
        assert dry_run is False
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(helper_module, "_run_subprocess", _capture)
    parser = helper_module.build_parser()

    transport_args = parser.parse_args(
        [
            "run-transport",
            "--pipeline-manifest",
            str(manifest_path),
            "flow-ready-001",
            "--python-exe",
            sys.executable,
            "--transport-backend",
            "torch",
            "--transport-execution-backend",
            "torch",
            "--allow-unready-flow-for-downstream",
        ]
    )
    thermal_args = parser.parse_args(
        [
            "run-thermal",
            "--pipeline-manifest",
            str(manifest_path),
            "flow-ready-001",
            "--python-exe",
            sys.executable,
            "--thermal-backend",
            "numpy",
            "--allow-unready-flow-for-downstream",
        ]
    )

    assert int(transport_args.func(transport_args)) == 0
    assert int(thermal_args.func(thermal_args)) == 0
    assert len(captured) == 2

    transport_cmd, thermal_cmd = captured
    assert transport_cmd[:5] == [
        sys.executable,
        str(helper_module.PIPELINE_RUNNER_SCRIPT),
        "--run-root",
        str(run_root.resolve()),
        "--from-flow-run-id",
    ]
    assert "flow-ready-001" in transport_cmd
    assert "--transport-backend" in transport_cmd
    assert "--transport-steps" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-steps") + 1] == "20000"
    assert "--transport-dt-mode" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-dt-mode") + 1] == "auto"
    assert "--transport-dt" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-dt") + 1] == "1.5e-05"
    assert "--transport-mode" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-mode") + 1] == (
        "advection_diffusion"
    )
    assert "--transport-scheme" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-scheme") + 1] == (
        "bounded_upwind"
    )
    assert "--transport-cfl-target" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-cfl-target") + 1] == "0.5"
    assert "--transport-cfl-limit" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-cfl-limit") + 1] == "0.8"
    assert "--transport-diffusivity" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-diffusivity") + 1] == (
        "3e-10"
    )
    assert (
        transport_cmd[transport_cmd.index("--transport-kinematic-viscosity") + 1]
        == "1e-06"
    )
    assert (
        transport_cmd[transport_cmd.index("--transport-max-supported-grid-peclet") + 1]
        == "2.0"
    )
    assert (
        transport_cmd[transport_cmd.index("--transport-max-supported-schmidt") + 1]
        == "1000.0"
    )
    assert "--transport-gradient-method" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-gradient-method") + 1] == (
        "least_squares"
    )
    assert "--transport-laplacian-method" in transport_cmd
    assert (
        transport_cmd[transport_cmd.index("--transport-laplacian-method") + 1] == "tpfa"
    )
    assert "--transport-left-inlet-value" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-left-inlet-value") + 1] == (
        "0.0"
    )
    assert "--transport-right-inlet-value" in transport_cmd
    assert (
        transport_cmd[transport_cmd.index("--transport-right-inlet-value") + 1] == "1.0"
    )
    assert "--transport-inlet-speed" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-inlet-speed") + 1] == "0.15"
    assert "--transport-checkpoint-every" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-checkpoint-every") + 1] == (
        "5000"
    )
    assert "--transport-progress-every" in transport_cmd
    assert transport_cmd[transport_cmd.index("--transport-progress-every") + 1] == (
        "500"
    )
    assert "--transport-no-velocity-comparison" in transport_cmd
    assert "--transport-no-transport-scheme-comparison" in transport_cmd
    assert "--transport-fail-if-numpy-fallback" in transport_cmd
    assert "--allow-unready-flow-for-downstream" in transport_cmd

    assert thermal_cmd[:5] == [
        sys.executable,
        str(helper_module.PIPELINE_RUNNER_SCRIPT),
        "--run-root",
        str(run_root.resolve()),
        "--from-flow-run-id",
    ]
    assert "flow-ready-001" in thermal_cmd
    assert "--run-thermal" in thermal_cmd
    assert "--skip-transport" in thermal_cmd
    assert "--allow-unready-flow-for-downstream" in thermal_cmd
    assert "--msh" in thermal_cmd
    msh_value = thermal_cmd[thermal_cmd.index("--msh") + 1]
    assert Path(msh_value).resolve() == (
        PROJECT_ROOT / "data" / "meshes" / "gmsh" / "t_junction.msh"
    )
    assert "--thermal-steps" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-steps") + 1] == "20000"
    assert "--thermal-dt-mode" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-dt-mode") + 1] == "auto"
    assert "--thermal-dt" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-dt") + 1] == "0.0005"
    assert "--thermal-cfl-target" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-cfl-target") + 1] == "0.5"
    assert "--thermal-cfl-limit" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-cfl-limit") + 1] == "0.8"
    assert "--thermal-heat-source" in thermal_cmd
    assert thermal_cmd[thermal_cmd.index("--thermal-heat-source") + 1] == "0.0"


def test_manifest_tool_forwards_transport_end_time_without_step_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    helper_module = _load_helper_module()
    run_root = tmp_path / "results" / "manifest_tool_transport_end_time"
    manifest_path = _build_manifest_with_ready_flow(run_root)
    captured: list[list[str]] = []

    monkeypatch.setattr(
        helper_module,
        "_run_subprocess",
        lambda cmd, *, dry_run: captured.append(list(cmd)) or 0,
    )
    args = helper_module.build_parser().parse_args(
        [
            "run-transport",
            "--pipeline-manifest",
            str(manifest_path),
            "--transport-end-time",
            "0.3",
            "flow-ready-001",
        ]
    )

    assert int(args.func(args)) == 0
    assert len(captured) == 1
    assert "--transport-end-time" in captured[0]
    assert captured[0][captured[0].index("--transport-end-time") + 1] == "0.3"
    assert "--transport-steps" not in captured[0]


def test_manifest_tool_run_transport_can_launch_diagnostic_override_path(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "results" / "manifest_tool_unready_transport"
    manifest_path = run_root / "pipeline_manifest.json"

    _run(
        [
            sys.executable,
            str(
                PROJECT_ROOT / "experiments" / "gmsh" / "run_gmsh_pipeline_manifest.py"
            ),
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
        run for run in manifest["runs"].values() if run.get("stage_type") == "flow"
    )
    flow_run_id = str(flow_run["run_id"])

    _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "run-transport",
            "--pipeline-manifest",
            str(manifest_path),
            flow_run_id,
            "--python-exe",
            sys.executable,
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
            "--allow-unready-flow-for-downstream",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    transport_run = next(
        run for run in manifest["runs"].values() if run.get("stage_type") == "transport"
    )
    assert transport_run["parent_run_ids"] == [flow_run_id]
    assert transport_run["status"] == "completed"


def test_manifest_tool_run_flow_from_existing_import_preserves_manifest_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    helper_module = _load_helper_module()
    run_root = tmp_path / "results" / "manifest_tool_state"
    manifest_path = run_root / "pipeline_manifest.json"
    import_run_dir = run_root / "import_runs" / "mesh"
    import_run_dir.mkdir(parents=True, exist_ok=True)
    create_pipeline_manifest_file(
        manifest_path,
        pipeline_id="manifest_tool_state",
        pipeline_root=run_root,
    )
    upsert_pipeline_run(
        manifest_path,
        run_id="import-001",
        stage_type="import",
        status="completed",
        run_dir=import_run_dir,
        inputs={
            "original_input": "t_junction.msh",
            "resolved_msh_path": PROJECT_ROOT / "data/meshes/gmsh" / "t_junction.msh",
        },
        outputs={"mesh_npz": import_run_dir / "t_junction_imported_mesh.npz"},
    )

    monkeypatch.setattr(helper_module, "_run_subprocess", lambda cmd, dry_run: 0)
    parser = helper_module.build_parser()
    args = parser.parse_args(
        [
            "run-flow",
            "--pipeline-manifest",
            str(manifest_path),
            "import-001",
        ]
    )

    assert int(args.func(args)) == 0
    manifest = load_pipeline_manifest(manifest_path)
    assert set(manifest["runs"]) == {"import-001"}


def test_manifest_tool_can_register_external_ready_flow_and_run_transport(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external_source"
    flow_dir, mesh_npz = _build_external_flow_fixture(source_root, ready=True)
    run_root = tmp_path / "results" / "manifest_tool_external_source"
    manifest_path = run_root / "pipeline_manifest.json"

    registration = _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "register-external-flow",
            "--run-root",
            str(run_root),
            "--pipeline-id",
            "manifest_tool_external_source",
            "--json",
            str(flow_dir / "summary.json"),
        ]
    )
    payload = json.loads(registration.stdout)
    flow_run_id = str(payload["run_id"])

    manifest = load_pipeline_manifest(manifest_path)
    flow_run = get_pipeline_run(manifest, flow_run_id)
    assert flow_run["status"] == "completed"
    assert flow_run["parent_run_ids"] == []
    assert flow_run["metadata"]["ready_for_next_stage"] is True
    assert flow_run["metadata"]["source_kind"] == "external_flow_run"
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            flow_run["outputs"]["mesh_npz"],
        )
        == mesh_npz.resolve()
    )
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            flow_run["outputs"]["summary_json"],
        )
        == (flow_dir / "summary.json").resolve()
    )
    assert (
        resolve_manifest_path_reference(
            manifest_path,
            flow_run["outputs"]["flow_coupling_metadata_json"],
        )
        == (flow_dir / "flow_coupling_metadata.json").resolve()
    )

    _run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "run-transport",
            "--pipeline-manifest",
            str(manifest_path),
            flow_run_id,
            "--python-exe",
            sys.executable,
            "--transport-backend",
            "numpy",
            "--transport-execution-backend",
            "numpy",
            "--transport-steps",
            "5",
            "--transport-dt-mode",
            "manual",
            "--transport-dt",
            "1e-4",
            "--transport-progress-every",
            "0",
        ]
    )

    manifest = load_pipeline_manifest(manifest_path)
    transport_run = next(
        run for run in manifest["runs"].values() if run["stage_type"] == "transport"
    )
    assert transport_run["parent_run_ids"] == [flow_run_id]
    assert transport_run["status"] == "completed"


def test_manifest_tool_rejects_unready_external_flow_registration(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "external_source"
    flow_dir, _ = _build_external_flow_fixture(source_root, ready=False)
    run_root = tmp_path / "results" / "manifest_tool_external_source_reject"
    manifest_path = run_root / "pipeline_manifest.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "register-external-flow",
            "--run-root",
            str(run_root),
            str(flow_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "readiness is false" in completed.stderr
    assert (
        not manifest_path.exists()
        or load_pipeline_manifest(manifest_path)["runs"] == {}
    )
