"""Direct thermal stage runner for tetra-native thermal diagnostics.

Useful for explicit stage launches and debugging, but not a replacement for the
supported manifest-first pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
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
from experiments.gmsh._flow_coupling import (  # noqa: E402
    check_flow_mesh_compatibility,
    load_flow_summary_payload,
    validate_source_flow_readiness,
)
from microfluidics.path_contract import (  # noqa: E402
    GMSH_TETRA_THERMAL_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)
from microfluidics.gmsh.tetra.gmsh_tetra_regime_guardrail import (  # noqa: E402
    DEFAULT_MAX_GRID_PECLET,
    DEFAULT_MAX_PRANDTL,
    build_scalar_regime_audit,
)

_ACTIVE_PIPELINE_MANIFEST_RECORDER = None
_ACTIVE_PIPELINE_MANIFEST_INPUTS: dict[str, Any] | None = None
_ACTIVE_PIPELINE_MANIFEST_ARTIFACTS: dict[str, Any] | None = None


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _build_stage_status(
    *,
    run_completed: bool,
    numerically_stable: bool,
    physically_ready: bool,
    ready_for_next_stage: bool,
    ready_for_long_run: bool,
    reason: str,
    checks: Dict[str, bool],
) -> Dict[str, Any]:
    return {
        "run_completed": bool(run_completed),
        "numerically_stable": bool(numerically_stable),
        "physically_ready": bool(physically_ready),
        "ready_for_next_stage": bool(ready_for_next_stage),
        "ready_for_long_run": bool(ready_for_long_run),
        "stage_status_reason": str(reason),
        "stage_status_checks": {str(k): bool(v) for k, v in checks.items()},
    }


def _resolve_msh_input(msh_input: str | Path) -> tuple[Path, list[Path]]:
    raw = Path(str(msh_input))
    searched: list[Path] = []

    if raw.is_absolute():
        candidate = raw.resolve()
        searched.append(candidate)
        if candidate.exists():
            return candidate, searched
        raise FileNotFoundError(f"Gmsh mesh file not found: {candidate}")

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


def _resolve_thermal_diffusivity(
    *,
    thermal_diffusivity: float | None,
    rho: float,
    cp: float,
    thermal_conductivity: float | None,
) -> float:
    if thermal_diffusivity is not None:
        if thermal_diffusivity <= 0.0:
            raise ValueError("thermal_diffusivity must be positive.")
        return float(thermal_diffusivity)
    if thermal_conductivity is None:
        raise ValueError(
            "Provide either --thermal-diffusivity or --thermal-conductivity to resolve alpha."
        )
    if thermal_conductivity <= 0.0 or rho <= 0.0 or cp <= 0.0:
        raise ValueError("thermal_conductivity, rho, cp must be positive.")
    return float(thermal_conductivity / (rho * cp))


def _save_temperature_previews(
    *,
    centers: np.ndarray,
    temperature: np.ndarray,
    velocity_magnitude: np.ndarray,
    run_dir: Path,
    file_prefix: str,
    velocity_label: str,
) -> Dict[str, str]:
    previews: Dict[str, str] = {}

    fig_t, ax_t = plt.subplots(figsize=(7.2, 5.6))
    sc_t = ax_t.scatter(
        centers[:, 0],
        centers[:, 1],
        c=temperature,
        s=2.0,
        cmap="inferno",
        alpha=0.85,
        linewidths=0,
    )
    ax_t.set_title("Temperature XY (cell centers)")
    ax_t.set_xlabel("x [m]")
    ax_t.set_ylabel("y [m]")
    ax_t.set_aspect("equal", adjustable="box")
    fig_t.colorbar(sc_t, ax=ax_t, label="T [K]")
    fig_t.tight_layout()
    path_t = run_dir / f"{file_prefix}_temperature_xy.png"
    fig_t.savefig(path_t, dpi=180)
    plt.close(fig_t)
    previews["temperature_xy"] = str(path_t)

    z = centers[:, 2]
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
    z_span = float(np.max(z) - np.min(z))
    tol = max(0.02 * z_span, 1e-6)
    mask_mid = np.abs(z - z_mid) <= tol
    if not np.any(mask_mid):
        idx = np.argsort(np.abs(z - z_mid))[: max(100, centers.shape[0] // 50)]
        mask_mid = np.zeros(centers.shape[0], dtype=bool)
        mask_mid[idx] = True

    fig_tm, ax_tm = plt.subplots(figsize=(7.2, 5.6))
    sc_tm = ax_tm.scatter(
        centers[mask_mid, 0],
        centers[mask_mid, 1],
        c=temperature[mask_mid],
        s=4.0,
        cmap="inferno",
        alpha=0.9,
        linewidths=0,
    )
    ax_tm.set_title("Temperature Mid-Z XY")
    ax_tm.set_xlabel("x [m]")
    ax_tm.set_ylabel("y [m]")
    ax_tm.set_aspect("equal", adjustable="box")
    fig_tm.colorbar(sc_tm, ax=ax_tm, label="T [K]")
    fig_tm.tight_layout()
    path_tm = run_dir / f"{file_prefix}_temperature_mid_z_xy.png"
    fig_tm.savefig(path_tm, dpi=180)
    plt.close(fig_tm)
    previews["temperature_mid_z_xy"] = str(path_tm)

    fig_v, ax_v = plt.subplots(figsize=(7.2, 5.6))
    sc_v = ax_v.scatter(
        centers[:, 0],
        centers[:, 1],
        c=velocity_magnitude,
        s=2.0,
        cmap="magma",
        alpha=0.85,
        linewidths=0,
    )
    ax_v.set_title(f"Velocity Magnitude XY ({velocity_label})")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")
    ax_v.set_aspect("equal", adjustable="box")
    fig_v.colorbar(sc_v, ax=ax_v, label="|u| [m/s]")
    fig_v.tight_layout()
    path_v = run_dir / f"{file_prefix}_velocity_magnitude_xy.png"
    fig_v.savefig(path_v, dpi=180)
    plt.close(fig_v)
    previews["velocity_magnitude_xy"] = str(path_v)
    return previews


def _export_thermal_vtu(
    *,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    temperature: np.ndarray,
    velocity: np.ndarray,
    debug_cell_data: Dict[str, np.ndarray] | None,
    output_path: Path,
) -> Path:
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "meshio is required to export VTU thermal results. Install with `pip install meshio`."
        ) from exc

    speed = np.linalg.norm(velocity, axis=1)
    cell_data: Dict[str, list[np.ndarray]] = {
        "temperature": [temperature.astype(np.float64, copy=False)],
        "velocity": [velocity.astype(np.float64, copy=False)],
        "velocity_magnitude": [speed.astype(np.float64, copy=False)],
    }
    if debug_cell_data is not None:
        n_cells = int(temperature.shape[0])
        for key, value in debug_cell_data.items():
            arr = np.asarray(value)
            if arr.shape[0] != n_cells:
                continue
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8, copy=False)
            elif arr.dtype.kind not in {"f", "i", "u"}:
                continue
            cell_data[str(key)] = [arr]

    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", tetrahedra.astype(np.int64, copy=False))],
        cell_data=cell_data,
    )
    meshio.write(output_path, mesh)
    return output_path


def _resolve_flow_face_flux_path(
    *,
    explicit_path: str,
    flow_summary_json: str,
    run_dir: Path,
    mesh=None,
    allow_unready_flow: bool = False,
    allow_manual_flow_input: bool = False,
) -> tuple[np.ndarray, dict[str, object], Path | None]:
    summary_path = (
        _normalize_user_path(flow_summary_json).resolve()
        if str(flow_summary_json).strip()
        else None
    )
    explicit_flux_path = (
        _normalize_user_path(explicit_path).resolve()
        if str(explicit_path).strip()
        else None
    )

    if summary_path is not None:
        payload = load_flow_summary_payload(summary_path)
        readiness = validate_source_flow_readiness(
            dict(payload.get("metadata", {})),
            strict=not bool(allow_unready_flow),
            strict_label="thermal source flow",
        )
        compatibility: dict[str, object] = {}
        if mesh is not None:
            compatibility = check_flow_mesh_compatibility(
                mesh=mesh,
                flow_payload=payload,
            )
            if not bool(compatibility.get("compatible", False)):
                raise RuntimeError(
                    "Flow run mesh is incompatible with current thermal mesh: "
                    f"{compatibility}"
                )

        resolved_flux_path = Path(str(payload.get("flux_path", ""))).resolve()
        if explicit_flux_path is not None and explicit_flux_path != resolved_flux_path:
            raise RuntimeError(
                "Thermal flow summary resolves a different coupling flux than the "
                "explicit --flow-face-flux-npy path. Production flow-solver "
                "coupling must use the summary-resolved payload."
            )

        face_flux = np.asarray(payload["face_flux"], dtype=np.float64).reshape(-1)
        summary_payload = (
            dict(payload.get("summary", {}))
            if isinstance(payload.get("summary", {}), dict)
            else {}
        )
        metadata = (
            dict(payload.get("metadata", {}))
            if isinstance(payload.get("metadata", {}), dict)
            else {}
        )
        stage_status = (
            dict(metadata.get("stage_status", {}))
            if isinstance(metadata.get("stage_status", {}), dict)
            else {}
        )
        meta: dict[str, object] = {
            "velocity_source": "flow_solver",
            "flow_face_flux_path": str(resolved_flux_path),
            "flow_summary_json": str(summary_path),
            "flow_summary_payload_available": True,
            "flow_solved": bool(readiness.get("ready", False)),
            "pressure_solved": bool(metadata.get("pressure_solved", False)),
            "run_completed": bool(readiness.get("run_completed", False)),
            "numerically_stable": bool(readiness.get("numerically_stable", False)),
            "physically_ready": bool(readiness.get("physically_ready", False)),
            "ready_for_next_stage": bool(readiness.get("ready", False)),
            "ready_for_long_run": bool(readiness.get("ready_for_long_run", False)),
            "stage_status_reason": str(readiness.get("stage_status_reason", "")),
            "flow_mode": summary_payload.get("flow_mode"),
            "flow_acceptance_reason": summary_payload.get(
                "projection_acceptance_reason"
            ),
            "flow_artifacts": summary_payload.get("artifacts", {}),
            "flow_stage_status": stage_status,
            "flow_coupling_metadata_path": str(payload.get("metadata_path", "")),
            "flow_mesh_compatibility": compatibility,
            "manual_flow_input_override": False,
        }
        _write_json(run_dir / "flow_summary_snapshot.json", summary_payload)
        return face_flux, meta, summary_path

    if explicit_flux_path is None:
        raise ValueError(
            "Flow summary json is required for --velocity-source=flow_solver unless "
            "--allow-manual-flow-input is used with an explicit "
            "--flow-face-flux-npy."
        )
    if not bool(allow_manual_flow_input):
        raise ValueError(
            "Manual flow-face-flux input without coupling metadata is unsafe and "
            "requires --allow-manual-flow-input."
        )
    if not explicit_flux_path.exists():
        raise FileNotFoundError(f"Flow face flux file not found: {explicit_flux_path}")

    face_flux = np.asarray(np.load(explicit_flux_path), dtype=np.float64).reshape(-1)
    meta = {
        "velocity_source": "flow_solver",
        "flow_face_flux_path": str(explicit_flux_path),
        "flow_summary_json": None,
        "flow_summary_payload_available": False,
        "flow_solved": False,
        "pressure_solved": False,
        "run_completed": False,
        "numerically_stable": False,
        "physically_ready": False,
        "ready_for_next_stage": False,
        "ready_for_long_run": False,
        "stage_status_reason": "manual_flow_input_override",
        "manual_flow_input_override": True,
    }
    return face_flux, meta, None


def main() -> None:
    from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh
    from microfluidics.gmsh.gmsh_mesh_validation import (
        format_validation_report,
        validate_imported_tetra_mesh,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        _reconstruct_cell_velocity_from_face_flux_numpy,
        compute_tetra_flux_divergence,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
        build_face_normal_flux_from_velocity,
        run_operator_diagnostics,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
        resolve_inlet_face_groups,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_thermal_solver import (
        GmshTetraThermalConfig,
        run_tetra_thermal_debug,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (
        build_prescribed_velocity_field,
    )
    from microfluidics.preprocessor import (
        apply_flow_profile_to_mesh,
        compile_flow_runtime_profile,
        compile_scalar_runtime_profile,
        compile_uniform_material_properties,
        load_case_config,
        scalar_profile_to_diffusion_kwargs,
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
        help="Optional case_config_v1 JSON; mesh.path may replace --msh.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_TETRA_THERMAL_RUNS_ROOT_REL)),
    )
    add_pipeline_manifest_arguments(parser)
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument(
        "--execution-profile",
        type=str,
        choices=("production", "debug", "reference"),
        default="debug",
    )
    parser.add_argument(
        "--allow-numpy-production-fallback",
        action="store_true",
        help=(
            "Allow production execution_profile to continue on numpy when torch is "
            "unavailable or numpy is selected explicitly. Results will be marked as "
            "degraded execution mode."
        ),
    )
    parser.add_argument(
        "--velocity-source",
        type=str,
        choices=("prescribed", "flow_solver"),
        default="",
    )
    parser.add_argument(
        "--velocity-field",
        type=str,
        choices=(
            "constant_x",
            "two_inlets_to_outlet_tj",
            "two_inlets_to_outlet_tj_balanced",
            "two_inlets_to_outlet_tj_balanced_symmetric_profile",
            "two_inlets_to_outlet_tj_piecewise_centerline",
            "two_inlets_to_outlet_tj_axis_aligned_sanity",
            "two_inlets_to_outlet_tj_axis_aligned_clean",
        ),
        default="two_inlets_to_outlet_tj_axis_aligned_clean",
    )
    parser.add_argument("--inlet-speed", type=float, default=0.15)
    parser.add_argument("--flow-summary-json", type=str, default="")
    parser.add_argument("--flow-face-flux-npy", type=str, default="")
    parser.add_argument(
        "--allow-unready-flow",
        action="store_true",
        help=(
            "Unsafe debug override: allow thermal coupling to consume upstream flow "
            "even when source readiness is false."
        ),
    )
    parser.add_argument(
        "--allow-manual-flow-input",
        action="store_true",
        help=(
            "Unsafe debug override: allow --velocity-source=flow_solver to use an "
            "explicit --flow-face-flux-npy without summary-derived coupling metadata."
        ),
    )

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument(
        "--dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    parser.add_argument("--cfl-target", type=float, default=0.5)
    parser.add_argument("--cfl-limit", type=float, default=0.8)
    parser.add_argument("--diffusion-stability-factor", type=float, default=1.0)

    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--cp", type=float, default=4180.0)
    parser.add_argument("--thermal-conductivity", type=float, default=0.6)
    parser.add_argument("--thermal-diffusivity", type=float, default=-1.0)
    parser.add_argument(
        "--kinematic-viscosity",
        type=float,
        default=1e-6,
        help="Kinematic viscosity nu [m^2/s] used only for Pr support diagnostics.",
    )
    parser.add_argument(
        "--max-supported-grid-peclet",
        type=float,
        default=DEFAULT_MAX_GRID_PECLET,
        help=(
            "Warning threshold for thermal Pe_grid=max(Q_out/(alpha*h)); a finite "
            "exceedance does not block readiness and is not a CFL stability limit."
        ),
    )
    parser.add_argument(
        "--max-supported-prandtl",
        type=float,
        default=DEFAULT_MAX_PRANDTL,
        help="Warning threshold for Pr=nu/alpha; finite exceedance does not block readiness.",
    )
    parser.add_argument("--heat-source", type=float, default=0.0)
    parser.add_argument(
        "--wall-boundary-mode",
        choices=("adiabatic", "fixed_temperature", "fixed_heat_flux"),
        default="adiabatic",
    )
    parser.add_argument(
        "--heated-wall-group",
        action="append",
        default=[],
        help="Gmsh wall physical group to heat; repeat for multiple groups.",
    )
    parser.add_argument("--wall-temperature", type=float, default=None)
    parser.add_argument("--wall-heat-flux", type=float, default=None)
    parser.add_argument("--initial-temperature", type=float, default=298.15)
    parser.add_argument("--left-inlet-temperature", type=float, default=303.15)
    parser.add_argument("--right-inlet-temperature", type=float, default=303.15)
    parser.add_argument("--min-temperature", type=float, default=290.0)
    parser.add_argument("--max-temperature", type=float, default=400.0)
    parser.add_argument(
        "--limiter-scheme",
        type=str,
        choices=("upwind", "bounded_upwind"),
        default="bounded_upwind",
    )
    parser.add_argument("--no-clipping", action="store_true")
    parser.add_argument(
        "--gradient-method",
        type=str,
        choices=("face", "least_squares"),
        default="least_squares",
    )
    parser.add_argument(
        "--laplacian-method",
        type=str,
        choices=("tpfa", "lsq_flux"),
        default="lsq_flux",
    )
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--diagnostics-mode",
        type=str,
        choices=("auto", "debug", "fast"),
        default="auto",
    )
    parser.add_argument(
        "--history-mode",
        type=str,
        choices=("auto", "dict", "compact", "off"),
        default="auto",
    )
    parser.add_argument(
        "--torch-compile-step-body",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--history-stride", type=int, default=100)
    parser.add_argument("--diagnostics-stride", type=int, default=0)
    parser.add_argument("--full-step-diagnostics", action="store_true")
    parser.add_argument("--debug-artifacts", action="store_true")
    args = parser.parse_args()
    case_config_path = (
        _normalize_user_path(args.case_config).resolve()
        if str(args.case_config).strip()
        else None
    )
    case_config = load_case_config(case_config_path) if case_config_path else None
    if not str(args.msh).strip() and case_config is None:
        parser.error(
            "--msh is required for direct thermal runs unless --case-config is "
            "provided."
        )
    if not str(args.velocity_source).strip():
        parser.error(
            "--velocity-source is required. Choose --velocity-source prescribed or "
            "--velocity-source flow_solver."
        )

    if str(args.msh).strip():
        msh_input = _normalize_user_path(args.msh)
    else:
        assert case_config is not None and case_config_path is not None
        configured_mesh = Path(case_config.mesh.path)
        msh_input = (
            configured_mesh
            if configured_mesh.is_absolute()
            else (case_config_path.parent / configured_mesh).resolve()
        )
    output_root = _normalize_user_path(args.output_root).resolve()
    flow_summary_json = (
        _normalize_user_path(args.flow_summary_json).resolve()
        if str(args.flow_summary_json).strip()
        else None
    )
    flow_face_flux_npy = (
        _normalize_user_path(args.flow_face_flux_npy).resolve()
        if str(args.flow_face_flux_npy).strip()
        else None
    )
    msh_path, searched_paths = _resolve_msh_input(msh_input)
    if case_config is not None and str(args.msh).strip():
        assert case_config_path is not None
        configured_mesh = Path(case_config.mesh.path)
        configured_mesh = (
            configured_mesh.resolve()
            if configured_mesh.is_absolute()
            else (case_config_path.parent / configured_mesh).resolve()
        )
        if configured_mesh != msh_path:
            parser.error(
                f"--msh resolves to {msh_path}, but case config mesh.path resolves "
                f"to {configured_mesh}"
            )
    if case_config is not None:
        material = compile_uniform_material_properties(case_config)
        args.rho = material.get("density_kg_per_m3", args.rho)
        args.cp = material.get("specific_heat_capacity_j_per_kg_k", args.cp)
        args.thermal_conductivity = material.get(
            "thermal_conductivity_w_per_m_k", args.thermal_conductivity
        )
        args.thermal_diffusivity = material.get(
            "thermal_diffusivity_m2_per_s", args.thermal_diffusivity
        )
        args.kinematic_viscosity = material.get(
            "kinematic_viscosity_m2_per_s", args.kinematic_viscosity
        )
    run_dir = create_timestamped_run_dir(output_root, msh_path.stem)
    manifest_recorder = build_pipeline_manifest_recorder(
        args,
        stage_type="thermal",
        run_dir=run_dir,
    )
    manifest_inputs = {
        "msh_path": str(msh_path),
        "output_root": str(output_root),
        "velocity_source": str(args.velocity_source),
        "flow_summary_json": str(flow_summary_json) if flow_summary_json else "",
        "flow_face_flux_npy": str(flow_face_flux_npy) if flow_face_flux_npy else "",
        "allow_unready_flow": bool(args.allow_unready_flow),
        "allow_manual_flow_input": bool(args.allow_manual_flow_input),
        "kinematic_viscosity": float(args.kinematic_viscosity),
        "max_supported_grid_peclet": float(args.max_supported_grid_peclet),
        "max_supported_prandtl": float(args.max_supported_prandtl),
        "case_config": str(case_config_path) if case_config_path else "",
    }
    manifest_artifacts = {
        "run_log": str(run_dir / "run.log"),
        "config_json": str(run_dir / "config.json"),
        "summary_json": str(run_dir / "summary.json"),
        "acceptance_report_json": str(run_dir / "acceptance_report.json"),
        "stage_status_json": str(run_dir / "stage_status.json"),
        "thermal_regime_audit_json": str(run_dir / "thermal_regime_audit.json"),
        "validation_report_txt": str(run_dir / "validation_report.txt"),
        "thermal_history_json": str(run_dir / "thermal_history.json"),
        "thermal_fields_npz": str(run_dir / f"{msh_path.stem}_thermal_fields.npz"),
    }
    global _ACTIVE_PIPELINE_MANIFEST_RECORDER
    global _ACTIVE_PIPELINE_MANIFEST_INPUTS
    global _ACTIVE_PIPELINE_MANIFEST_ARTIFACTS
    _ACTIVE_PIPELINE_MANIFEST_RECORDER = manifest_recorder
    _ACTIVE_PIPELINE_MANIFEST_INPUTS = manifest_inputs
    _ACTIVE_PIPELINE_MANIFEST_ARTIFACTS = manifest_artifacts
    manifest_recorder.record_started(
        inputs=manifest_inputs,
        artifacts=manifest_artifacts,
        metadata={"manifest_role": "thermal_stage"},
    )

    with _tee_logging(run_dir / "run.log"):
        backend = select_backend(args.backend)
        for note in backend.notes:
            print(f"[gmsh-tetra-thermal] note: {note}")
        print(f"[gmsh-tetra-thermal] run directory: {run_dir}")
        print(f"[gmsh-tetra-thermal] mesh path: {msh_path}")
        print("[gmsh-tetra-thermal] searched paths:")
        for candidate in searched_paths:
            print(f"  - {candidate}")

        thermal_diffusivity_input = (
            None
            if float(args.thermal_diffusivity) <= 0.0
            else float(args.thermal_diffusivity)
        )
        alpha = _resolve_thermal_diffusivity(
            thermal_diffusivity=thermal_diffusivity_input,
            rho=float(args.rho),
            cp=float(args.cp),
            thermal_conductivity=float(args.thermal_conductivity),
        )

        config_payload = {
            "msh_path": str(msh_path),
            "output_root": str(output_root),
            "backend": backend.__dict__,
            "requested_backend": str(args.backend),
            "torch_device": str(args.torch_device),
            "execution_profile": str(args.execution_profile),
            "allow_numpy_production_fallback": bool(
                args.allow_numpy_production_fallback
            ),
            "diagnostics_mode": str(args.diagnostics_mode),
            "history_mode": str(args.history_mode),
            "torch_compile_step_body": str(args.torch_compile_step_body),
            "velocity_source": str(args.velocity_source),
            "velocity_field": str(args.velocity_field),
            "flow_summary_json": (
                str(flow_summary_json) if flow_summary_json is not None else ""
            ),
            "flow_face_flux_npy": (
                str(flow_face_flux_npy) if flow_face_flux_npy is not None else ""
            ),
            "steps": int(args.steps),
            "dt": float(args.dt),
            "dt_mode": str(args.dt_mode),
            "cfl_target": float(args.cfl_target),
            "cfl_limit": float(args.cfl_limit),
            "diffusion_stability_factor": float(args.diffusion_stability_factor),
            "rho": float(args.rho),
            "cp": float(args.cp),
            "thermal_conductivity": float(args.thermal_conductivity),
            "thermal_diffusivity": float(alpha),
            "kinematic_viscosity": float(args.kinematic_viscosity),
            "max_supported_grid_peclet": float(args.max_supported_grid_peclet),
            "max_supported_prandtl": float(args.max_supported_prandtl),
            "heat_source": float(args.heat_source),
            "wall_boundary_mode": str(args.wall_boundary_mode),
            "heated_wall_physical_groups": [
                str(name) for name in args.heated_wall_group
            ],
            "wall_temperature": (
                float(args.wall_temperature)
                if args.wall_temperature is not None
                else None
            ),
            "wall_heat_flux": (
                float(args.wall_heat_flux) if args.wall_heat_flux is not None else None
            ),
            "initial_temperature": float(args.initial_temperature),
            "left_inlet_temperature": float(args.left_inlet_temperature),
            "right_inlet_temperature": float(args.right_inlet_temperature),
            "min_temperature": float(args.min_temperature),
            "max_temperature": float(args.max_temperature),
            "limiter_scheme": str(args.limiter_scheme),
            "clipping_enabled": bool(not args.no_clipping),
            "gradient_method": str(args.gradient_method),
            "laplacian_method": str(args.laplacian_method),
            "collect_history": bool(not args.no_history),
            "history_stride": int(args.history_stride),
            "diagnostics_stride": int(args.diagnostics_stride),
            "full_step_diagnostics": bool(args.full_step_diagnostics),
            "debug_artifacts": bool(args.debug_artifacts),
        }
        _write_json(run_dir / "config.json", config_payload)

        mesh = import_gmsh_tetra_mesh(msh_path)
        thermal_diffusion_kwargs = None
        if case_config is not None:
            flow_profile = compile_flow_runtime_profile(mesh, case_config)
            apply_flow_profile_to_mesh(mesh, flow_profile)
            thermal_profile = compile_scalar_runtime_profile(
                mesh, case_config, "temperature"
            )
            thermal_diffusion_kwargs = scalar_profile_to_diffusion_kwargs(
                thermal_profile
            )
            inlet_groups_for_case = resolve_inlet_face_groups(mesh)
            for side, argument_name in (
                ("left_faces", "left_inlet_temperature"),
                ("right_faces", "right_inlet_temperature"),
            ):
                faces = np.asarray(inlet_groups_for_case[side], dtype=np.int64)
                values = [
                    thermal_profile.dirichlet_by_face.get(int(face)) for face in faces
                ]
                if not values or any(value is None for value in values):
                    raise ValueError(
                        f"case temperature BC must define Dirichlet values on all {side}"
                    )
                first = float(values[0])
                if not all(np.isclose(first, float(value)) for value in values[1:]):
                    raise ValueError(
                        f"current thermal advection runtime requires uniform {side} temperature"
                    )
                setattr(args, argument_name, first)
            config_payload["left_inlet_temperature"] = float(
                args.left_inlet_temperature
            )
            config_payload["right_inlet_temperature"] = float(
                args.right_inlet_temperature
            )
            config_payload["case_config"] = str(case_config_path)
            _write_json(run_dir / "config.json", config_payload)
        validation = validate_imported_tetra_mesh(mesh)
        (run_dir / "validation_report.txt").write_text(
            format_validation_report(validation),
            encoding="utf-8",
        )
        if not validation.is_valid:
            raise RuntimeError(
                "Loaded mesh failed validation. See validation_report.txt"
            )

        velocity_metadata: dict[str, object]
        flux_diag: dict[str, float]

        if args.velocity_source == "prescribed":
            velocity_field = build_prescribed_velocity_field(
                mesh,
                field_name=args.velocity_field,  # type: ignore[arg-type]
                inlet_speed=float(args.inlet_speed),
            )
            face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
                mesh,
                velocity_field.cell_velocity,
                boundary_face_velocity_overrides=velocity_field.boundary_face_velocity_overrides,
                left_inlet_faces=velocity_field.boundary_groups["left_inlet_faces"],
                right_inlet_faces=velocity_field.boundary_groups["right_inlet_faces"],
                outlet_faces=velocity_field.boundary_groups["outlet_faces"],
                wall_faces=velocity_field.boundary_groups["wall_faces"],
            )
            cell_velocity = np.asarray(velocity_field.cell_velocity, dtype=np.float64)
            velocity_metadata = {
                "velocity_source": "prescribed_debug_field",
                "flow_solved": False,
                "pressure_solved": False,
                "field_name": velocity_field.name,
                "inlet_speed": float(args.inlet_speed),
                **velocity_field.metadata,
            }
            velocity_label = "prescribed"
        else:
            face_flux, flow_meta, flow_summary_path = _resolve_flow_face_flux_path(
                explicit_path=(
                    str(flow_face_flux_npy) if flow_face_flux_npy is not None else ""
                ),
                flow_summary_json=(
                    str(flow_summary_json) if flow_summary_json is not None else ""
                ),
                run_dir=run_dir,
                mesh=mesh,
                allow_unready_flow=bool(args.allow_unready_flow),
                allow_manual_flow_input=bool(args.allow_manual_flow_input),
            )
            if face_flux.shape[0] != mesh.face_vertices.shape[0]:
                raise ValueError(
                    "Flow face flux array size does not match mesh face count: "
                    f"{face_flux.shape[0]} vs {mesh.face_vertices.shape[0]}"
                )
            face_normal_velocity = face_flux / np.maximum(mesh.face_areas, 1e-16)

            inlet_groups = resolve_inlet_face_groups(mesh)
            div_diag = compute_tetra_flux_divergence(
                mesh,
                face_flux,
                left_inlet_faces=np.asarray(inlet_groups["left_faces"], dtype=np.int64),
                right_inlet_faces=np.asarray(
                    inlet_groups["right_faces"], dtype=np.int64
                ),
                outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
                wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
            )
            flux_diag = {
                "total_inlet_flux_in": float(div_diag.get("inlet_flux_total", 0.0)),
                "total_outlet_flux_out": float(div_diag.get("outlet_flux_total", 0.0)),
                "net_boundary_flux": float(div_diag.get("net_boundary_flux", 0.0)),
                "wall_flux_max_abs": float(div_diag.get("wall_flux_max_abs", 0.0)),
            }
            velocity_metadata = dict(flow_meta)
            velocity_metadata["flow_summary_json"] = (
                str(flow_summary_path) if flow_summary_path is not None else None
            )
            cell_velocity = _reconstruct_cell_velocity_from_face_flux_numpy(
                mesh, face_flux
            )
            velocity_label = "flow_solver"

        _write_json(run_dir / "flux_diagnostics.json", flux_diag)
        _write_json(run_dir / "velocity_metadata.json", velocity_metadata)

        thermal_cfg = GmshTetraThermalConfig(
            steps=int(args.steps),
            dt=float(args.dt),
            dt_mode=args.dt_mode,  # type: ignore[arg-type]
            cfl_target=float(args.cfl_target),
            cfl_limit=float(args.cfl_limit),
            diffusion_stability_factor=float(args.diffusion_stability_factor),
            thermal_diffusivity=float(alpha),
            rho=float(args.rho),
            cp=float(args.cp),
            heat_source=float(args.heat_source),
            initial_temperature=float(args.initial_temperature),
            left_inlet_temperature=float(args.left_inlet_temperature),
            right_inlet_temperature=float(args.right_inlet_temperature),
            wall_boundary_mode=args.wall_boundary_mode,  # type: ignore[arg-type]
            heated_wall_physical_groups=tuple(
                str(name) for name in args.heated_wall_group
            ),
            wall_temperature=(
                float(args.wall_temperature)
                if args.wall_temperature is not None
                else None
            ),
            wall_heat_flux=(
                float(args.wall_heat_flux) if args.wall_heat_flux is not None else None
            ),
            limiter_scheme=args.limiter_scheme,  # type: ignore[arg-type]
            clipping_enabled=bool(not args.no_clipping),
            min_temperature=float(args.min_temperature),
            max_temperature=float(args.max_temperature),
            gradient_method=args.gradient_method,  # type: ignore[arg-type]
            laplacian_method=args.laplacian_method,  # type: ignore[arg-type]
            backend=args.backend,  # type: ignore[arg-type]
            torch_device=str(args.torch_device),
            execution_profile=args.execution_profile,  # type: ignore[arg-type]
            allow_numpy_production_fallback=bool(args.allow_numpy_production_fallback),
            diagnostics_mode=args.diagnostics_mode,  # type: ignore[arg-type]
            torch_compile_step_body=args.torch_compile_step_body,  # type: ignore[arg-type]
            collect_history=bool(not args.no_history),
            history_mode=args.history_mode,  # type: ignore[arg-type]
            history_stride=int(args.history_stride),
            diagnostics_stride=int(args.diagnostics_stride),
            full_step_diagnostics=bool(args.full_step_diagnostics),
            debug_artifacts=bool(args.debug_artifacts),
        )

        thermal_result = run_tetra_thermal_debug(
            mesh,
            thermal_cfg,
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diag,
            velocity_metadata=velocity_metadata,
            diffusion_boundary_kwargs=thermal_diffusion_kwargs,
        )
        regime_audit = build_scalar_regime_audit(
            mesh,
            np.asarray(face_normal_velocity, dtype=np.float64),
            diffusivity=float(alpha),
            kinematic_viscosity=float(args.kinematic_viscosity),
            scalar_kind="thermal",
            max_grid_peclet=float(args.max_supported_grid_peclet),
            max_prandtl=float(args.max_supported_prandtl),
        )
        regime_supported_accuracy = bool(regime_audit.get("supported_accuracy", False))
        regime_blocking_error = bool(regime_audit.get("blocking_error", True))
        regime_warning_codes = list(regime_audit.get("warning_codes", []))
        regime_error_codes = list(regime_audit.get("error_codes", []))
        regime_warning = bool(regime_warning_codes)

        temperature = np.asarray(thermal_result["temperature"], dtype=np.float64)
        velocity = np.asarray(cell_velocity, dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        debug_cell_data: dict[str, np.ndarray] = {}
        if bool(args.debug_artifacts):
            final_step_debug = (
                thermal_result.get("final_step_debug", {})
                if isinstance(thermal_result.get("final_step_debug", {}), dict)
                else {}
            )
            for key in (
                "raw_state_before_limiter",
                "limited_state_before_clip",
                "updated_raw_before_non_advective_limiter",
                "updated_raw_before_clip",
                "advection_limiter_scale",
                "non_advective_limiter_scale",
                "advection_delta",
                "diffusion_delta",
                "source_delta",
                "clamp_delta",
                "clipped_mask",
                "overshoot_mask",
                "undershoot_mask",
                "advection_limiter_active_mask",
                "non_advective_limiter_active_mask",
            ):
                value = final_step_debug.get(key)
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.ndim >= 1 and arr.shape[0] == temperature.shape[0]:
                    debug_cell_data[str(key)] = arr
        _write_json(
            run_dir / "thermal_history.json",
            {
                "history": thermal_result["history"],
                "history_compact": thermal_result.get("history_compact", {}),
                "history_settings": thermal_result.get("history_settings", {}),
            },
        )

        npz_path = run_dir / f"{msh_path.stem}_thermal_fields.npz"
        np.savez_compressed(
            npz_path,
            temperature=temperature,
            velocity=velocity,
            face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
            cell_centers=mesh.cell_centers,
            cell_volumes=mesh.cell_volumes,
            points=mesh.points,
            tetrahedra=mesh.tetrahedra,
            thermal_diffusivity=np.array(alpha, dtype=float),
            heat_source=np.array(float(args.heat_source), dtype=float),
            source_term=np.array(
                float(args.heat_source) / (float(args.rho) * float(args.cp)),
                dtype=float,
            ),
            dt=np.array(float(thermal_result["dt_control"]["used_dt"]), dtype=float),
            steps=np.array(int(args.steps), dtype=int),
            clipped_mask=np.asarray(
                debug_cell_data.get(
                    "clipped_mask", np.zeros_like(temperature, dtype=bool)
                )
            ),
            overshoot_mask=np.asarray(
                debug_cell_data.get(
                    "overshoot_mask", np.zeros_like(temperature, dtype=bool)
                )
            ),
            undershoot_mask=np.asarray(
                debug_cell_data.get(
                    "undershoot_mask", np.zeros_like(temperature, dtype=bool)
                )
            ),
            advection_limiter_active_mask=np.asarray(
                debug_cell_data.get(
                    "advection_limiter_active_mask",
                    np.zeros_like(temperature, dtype=bool),
                )
            ),
            non_advective_limiter_active_mask=np.asarray(
                debug_cell_data.get(
                    "non_advective_limiter_active_mask",
                    np.zeros_like(temperature, dtype=bool),
                )
            ),
            advection_limiter_scale=np.asarray(
                debug_cell_data.get(
                    "advection_limiter_scale", np.ones_like(temperature)
                )
            ),
            non_advective_limiter_scale=np.asarray(
                debug_cell_data.get(
                    "non_advective_limiter_scale", np.ones_like(temperature)
                )
            ),
        )

        vtu_path = _export_thermal_vtu(
            points=mesh.points,
            tetrahedra=mesh.tetrahedra,
            temperature=temperature,
            velocity=velocity,
            debug_cell_data=debug_cell_data,
            output_path=run_dir / f"{msh_path.stem}_thermal_result.vtu",
        )

        previews = (
            _save_temperature_previews(
                centers=mesh.cell_centers,
                temperature=temperature,
                velocity_magnitude=speed,
                run_dir=run_dir,
                file_prefix=msh_path.stem,
                velocity_label=velocity_label,
            )
            if bool(args.debug_artifacts)
            else {}
        )

        operator_diag = run_operator_diagnostics(
            mesh,
            gradient_method=args.gradient_method,  # type: ignore[arg-type]
        )
        finite_temperature = bool(np.all(np.isfinite(temperature)))
        cfl_warning = bool(thermal_result["cfl_warning"])
        diffusion_warning = bool(
            thermal_result["dt_control"].get("diffusion_stability_warning", False)
        )
        execution_policy = dict(thermal_result.get("execution_policy", {}))
        backend_execution = dict(thermal_result.get("backend_execution", {}))
        production_policy_satisfied = bool(
            execution_policy.get("production_backend_satisfied", False)
        )
        degraded_execution = bool(execution_policy.get("degraded_execution_mode", True))
        used_numpy_fallback = bool(backend_execution.get("used_numpy_fallback", True))
        numerically_stable = bool(
            finite_temperature and (not cfl_warning) and (not diffusion_warning)
        )
        source_ready = bool(
            velocity_metadata.get("ready_for_next_stage", True)
            if str(args.velocity_source) == "flow_solver"
            else True
        )
        physically_ready = bool(
            numerically_stable and source_ready and not regime_blocking_error
        )
        production_profile = str(args.execution_profile) == "production"
        ready_for_next_stage = bool(
            physically_ready
            and (
                not production_profile
                or (
                    production_policy_satisfied
                    and not degraded_execution
                    and not used_numpy_fallback
                )
            )
        )
        ready_for_long_run = bool(ready_for_next_stage and production_profile)
        if not numerically_stable:
            stage_reason = "thermal_numerical_stability_failed"
        elif not source_ready:
            stage_reason = "thermal_source_flow_not_ready"
        elif regime_blocking_error:
            stage_reason = str(
                regime_error_codes[0]
                if regime_error_codes
                else "thermal_regime_data_error"
            )
        elif production_profile and not production_policy_satisfied:
            stage_reason = "thermal_production_backend_policy_not_satisfied"
        elif production_profile and (degraded_execution or used_numpy_fallback):
            stage_reason = "thermal_production_execution_degraded"
        else:
            stage_reason = "stage readiness satisfied"
        stage_status = _build_stage_status(
            run_completed=True,
            numerically_stable=numerically_stable,
            physically_ready=physically_ready,
            ready_for_next_stage=ready_for_next_stage,
            ready_for_long_run=ready_for_long_run,
            reason=stage_reason,
            checks={
                "finite_temperature": finite_temperature,
                "cfl_warning": cfl_warning,
                "diffusion_stability_warning": diffusion_warning,
                "source_ready": source_ready,
                "thermal_accuracy_supported": regime_supported_accuracy,
                "thermal_regime_warning": regime_warning,
                "thermal_regime_blocking_error": regime_blocking_error,
                "production_policy_satisfied": production_policy_satisfied,
                "degraded_execution": degraded_execution,
                "used_numpy_fallback": used_numpy_fallback,
            },
        )
        if regime_warning:
            stage_status["stage_status_warnings"] = regime_warning_codes

        summary = {
            "mesh_path": str(msh_path),
            "backend": backend.__dict__,
            "velocity": {
                "source": str(args.velocity_source),
                "metadata": velocity_metadata,
            },
            "validation": validation.summary,
            "flux_diagnostics": flux_diag,
            "operator_diagnostics": operator_diag,
            "thermal": {
                "config": _json_ready(thermal_cfg.__dict__),
                "backend_execution": thermal_result["backend_execution"],
                "execution_policy": thermal_result.get("execution_policy", {}),
                "history_settings": thermal_result.get("history_settings", {}),
                "performance_diagnostics": thermal_result.get(
                    "performance_diagnostics", {}
                ),
                "dt_control": thermal_result["dt_control"],
                "boundary_setup": thermal_result["boundary_setup"],
                "transport_info": thermal_result["transport_info"],
                "clipping": thermal_result["clipping"],
                "limiter": thermal_result["limiter"],
                "non_advective_limiter": thermal_result.get(
                    "non_advective_limiter", {}
                ),
                "energy_proxy": thermal_result["energy_proxy"],
                "global_energy_balance": thermal_result.get(
                    "global_energy_balance", {}
                ),
                "wall_heat_transfer": thermal_result.get("wall_heat_transfer", {}),
                "thermal_observables": thermal_result.get("thermal_observables", {}),
                "local_cfl_audit": thermal_result["local_cfl_audit"],
                "diffusion_diagnostics": thermal_result["diffusion_diagnostics"],
                "diffusion_backend": thermal_result["diffusion_diagnostics"][
                    "diffusion_backend"
                ],
                "final_stats": thermal_result["final_stats"],
                "cfl_max": thermal_result["cfl_max"],
                "cfl_warning": thermal_result["cfl_warning"],
                "thermal_regime_audit": regime_audit,
                "thermal_accuracy_supported": bool(regime_supported_accuracy),
                "thermal_accuracy_support_status": str(
                    regime_audit.get("support_status", "unsupported")
                ),
                "thermal_accuracy_support_reason_codes": list(
                    regime_audit.get("reason_codes", [])
                ),
                "thermal_regime_warning": regime_warning,
                "thermal_regime_warning_codes": regime_warning_codes,
                "thermal_regime_blocking_error": regime_blocking_error,
                "thermal_regime_error_codes": regime_error_codes,
            },
            "stage_status": stage_status,
            "run_completed": bool(stage_status["run_completed"]),
            "numerically_stable": bool(stage_status["numerically_stable"]),
            "physically_ready": bool(stage_status["physically_ready"]),
            "ready_for_next_stage": bool(stage_status["ready_for_next_stage"]),
            "ready_for_long_run": bool(stage_status["ready_for_long_run"]),
            "stage_status_reason": str(stage_status["stage_status_reason"]),
            "stage_status_warnings": list(
                stage_status.get("stage_status_warnings", [])
            ),
            "artifacts": {
                "run_log": str(run_dir / "run.log"),
                "config_json": str(run_dir / "config.json"),
                "summary_json": str(run_dir / "summary.json"),
                "acceptance_report_json": str(run_dir / "acceptance_report.json"),
                "stage_status_json": str(run_dir / "stage_status.json"),
                "thermal_regime_audit_json": str(run_dir / "thermal_regime_audit.json"),
                "validation_report_txt": str(run_dir / "validation_report.txt"),
                "flux_diagnostics_json": str(run_dir / "flux_diagnostics.json"),
                "velocity_metadata_json": str(run_dir / "velocity_metadata.json"),
                "thermal_history_json": str(run_dir / "thermal_history.json"),
                "thermal_fields_npz": str(npz_path),
                "thermal_vtu": str(vtu_path),
                "preview_png": previews,
            },
        }
        acceptance_report = {
            "run_completed": bool(stage_status["run_completed"]),
            "numerically_stable": bool(stage_status["numerically_stable"]),
            "physically_ready": bool(stage_status["physically_ready"]),
            "ready_for_next_stage": bool(stage_status["ready_for_next_stage"]),
            "ready_for_long_run": bool(stage_status["ready_for_long_run"]),
            "stage_status_reason": str(stage_status["stage_status_reason"]),
            "thermal_accuracy_supported": bool(regime_supported_accuracy),
            "thermal_accuracy_support_status": str(
                regime_audit.get("support_status", "unsupported")
            ),
            "thermal_accuracy_support_reason_codes": list(
                regime_audit.get("reason_codes", [])
            ),
            "thermal_regime_warning": regime_warning,
            "thermal_regime_warning_codes": regime_warning_codes,
            "thermal_regime_blocking_error": regime_blocking_error,
            "thermal_regime_error_codes": regime_error_codes,
            "thermal_regime_audit": regime_audit,
        }
        _write_json(run_dir / "thermal_regime_audit.json", regime_audit)
        _write_json(run_dir / "stage_status.json", stage_status)
        _write_json(run_dir / "acceptance_report.json", acceptance_report)
        _write_json(run_dir / "summary.json", summary)
        manifest_recorder.record_completed(
            inputs=manifest_inputs,
            outputs={
                "summary_json": str(run_dir / "summary.json"),
                "config_json": str(run_dir / "config.json"),
                "acceptance_report_json": str(run_dir / "acceptance_report.json"),
                "stage_status_json": str(run_dir / "stage_status.json"),
                "thermal_regime_audit_json": str(run_dir / "thermal_regime_audit.json"),
                "thermal_history_json": str(run_dir / "thermal_history.json"),
                "thermal_fields_npz": str(npz_path),
                "thermal_vtu": str(vtu_path),
            },
            artifacts=summary.get("artifacts", {}),
            metadata={
                "velocity_source": str(args.velocity_source),
                "mesh_path": str(msh_path),
                "run_completed": bool(stage_status["run_completed"]),
                "ready_for_next_stage": bool(stage_status["ready_for_next_stage"]),
                "ready_for_long_run": bool(stage_status["ready_for_long_run"]),
                "stage_status_reason": str(stage_status["stage_status_reason"]),
                "thermal_accuracy_supported": bool(regime_supported_accuracy),
                "thermal_accuracy_support_status": str(
                    regime_audit.get("support_status", "unsupported")
                ),
                "thermal_regime_warning": regime_warning,
                "thermal_regime_blocking_error": regime_blocking_error,
            },
        )

        print(
            "[gmsh-tetra-thermal] final stats: "
            f"min={thermal_result['final_stats']['min']:.6f}, "
            f"max={thermal_result['final_stats']['max']:.6f}, "
            f"mean={thermal_result['final_stats']['mean']:.6f}"
        )
        print(
            "[gmsh-tetra-thermal] dt control: "
            f"requested={thermal_result['dt_control']['requested_dt']:.6e}, "
            f"used={thermal_result['dt_control']['used_dt']:.6e}, "
            f"adv_limit={thermal_result['dt_control']['advection_dt_limit']:.6e}, "
            f"diff_limit={thermal_result['dt_control']['diffusion_dt_limit']:.6e}"
        )
        print(
            "[gmsh-tetra-thermal] flux diagnostics: "
            f"inlet={flux_diag['total_inlet_flux_in']:.6e}, "
            f"outlet={flux_diag['total_outlet_flux_out']:.6e}, "
            f"wall_max_abs={flux_diag['wall_flux_max_abs']:.6e}"
        )
        print(f"[gmsh-tetra-thermal] summary written: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if _ACTIVE_PIPELINE_MANIFEST_RECORDER is not None:
            _ACTIVE_PIPELINE_MANIFEST_RECORDER.record_failed(
                error=exc,
                inputs=_ACTIVE_PIPELINE_MANIFEST_INPUTS,
                artifacts=_ACTIVE_PIPELINE_MANIFEST_ARTIFACTS,
            )
        raise
