"""Direct transport stage runner for tetra-native transport diagnostics.

Useful for explicit stage launches and debugging, but not a replacement for the
supported manifest-first pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    check_flow_mesh_compatibility as _shared_check_flow_mesh_compatibility,
    load_flow_coupling_payload as _shared_load_flow_coupling_payload,
    validate_source_flow_readiness as _shared_validate_source_flow_readiness,
)
from microfluidics.path_contract import (  # noqa: E402
    GMSH_IMPORT_RUNS_ROOT_REL,
    GMSH_TETRA_TRANSPORT_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)

from microfluidics.gmsh.tetra.threshold_keys import (  # noqa: E402
    format_threshold_key,
    read_threshold_value,
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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _parse_child_run_directory(stdout: str, *, prefix: str, variant: str) -> Path:
    """Extract the child run directory from machine-readable stdout."""

    for line in stdout.splitlines():
        if line.startswith(prefix):
            raw_path = line.split(":", 1)[1].strip()
            if raw_path:
                return Path(raw_path).resolve()
            break
    raise RuntimeError(
        "Could not resolve child run directory from subprocess stdout for "
        f"variant={variant}. Expected a line starting with {prefix!r}."
    )


def _safe_rel_diff(num: float, den: float, floor: float = 1e-30) -> float:
    return float(abs(num) / max(abs(den), floor))


def _read_outlet_fraction(
    metrics: Dict[str, Any], threshold: float, default: float = 0.0
) -> float:
    return read_threshold_value(
        metrics,
        "outlet_frac_gt",
        threshold,
        default=default,
    )


def _extract_scalar_mass_final(summary: Dict[str, Any]) -> float:
    masses = summary.get("transport", {}).get("masses", {})
    candidates = (
        masses.get("scalar_mass_final"),
        masses.get("final_mass_proxy"),
        masses.get("total_mass_proxy_final"),
    )
    for value in candidates:
        if value is not None:
            return float(value)
    return 0.0


def _compute_cleanliness_flags(
    *,
    finite: bool,
    cfl_warning: bool,
    diffusion_stability_warning: bool,
    overshoot_max: float,
    undershoot_max: float,
    clipped_count: int,
    tolerance: float,
) -> Dict[str, Any]:
    strict_clean = bool(
        finite
        and (not cfl_warning)
        and (not diffusion_stability_warning)
        and overshoot_max <= 0.0
        and undershoot_max <= 0.0
        and clipped_count == 0
    )
    tol_clean = bool(
        finite
        and (not cfl_warning)
        and (not diffusion_stability_warning)
        and overshoot_max <= tolerance
        and undershoot_max <= tolerance
    )
    if strict_clean:
        reason = "strict clean: boundedness is exactly within [0,1]"
    elif tol_clean:
        reason = "tolerance clean: tiny boundedness drift accepted"
    elif cfl_warning:
        reason = "not clean: CFL warning"
    elif diffusion_stability_warning:
        reason = "not clean: diffusion stability warning"
    elif not finite:
        reason = "not clean: non-finite values detected"
    else:
        reason = "not clean: boundedness drift exceeds tolerance"
    return {
        "strict_numerically_clean_transport": strict_clean,
        "tolerance_numerically_clean_transport": tol_clean,
        "cleanliness_reason": reason,
    }


def _build_stage_status(
    *,
    run_completed: bool,
    numerically_stable: bool,
    physically_ready: bool,
    ready_for_next_stage: bool,
    ready_for_long_run: bool,
    checks: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    check_map = {str(k): bool(v) for k, v in (checks or {}).items()}
    reason = "stage readiness satisfied"
    if not run_completed:
        reason = "stage did not complete requested steps"
    elif not numerically_stable:
        reason = "stage is not numerically stable"
    elif not physically_ready:
        reason = "stage is numerically stable but not physically ready"
    elif not ready_for_next_stage:
        reason = "stage is physically ready but next-stage gate is not satisfied"
    elif not ready_for_long_run:
        reason = "stage is ready for next stage but not for long-run profile"
    return {
        "run_completed": bool(run_completed),
        "numerically_stable": bool(numerically_stable),
        "physically_ready": bool(physically_ready),
        "ready_for_next_stage": bool(ready_for_next_stage),
        "ready_for_long_run": bool(ready_for_long_run),
        "stage_status_reason": str(reason),
        "stage_status_checks": check_map,
    }


def _estimate_transport_front_timing(
    *,
    mesh,
    cell_velocity: np.ndarray,
    flux_diag: Dict[str, float],
    physical_time_final: float,
    breakthrough_turnover_threshold: float = 0.20,
    breakthrough_travel_fraction: float = 0.35,
) -> Dict[str, Any]:
    """Return only front timing values available before a transport run starts."""
    cell_volumes = np.asarray(mesh.cell_volumes, dtype=np.float64)
    domain_volume = float(np.sum(cell_volumes))
    inlet_flux_in = max(float(flux_diag.get("total_inlet_flux_in", 0.0)), 0.0)
    turnover = float(
        inlet_flux_in * float(physical_time_final) / max(domain_volume, 1e-30)
    )

    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    if inlet_faces.size and outlet_faces.size:
        inlet_center = np.mean(
            np.asarray(mesh.face_centers[inlet_faces], dtype=np.float64), axis=0
        )
        outlet_center = np.mean(
            np.asarray(mesh.face_centers[outlet_faces], dtype=np.float64), axis=0
        )
        axis = np.asarray(outlet_center - inlet_center, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
    else:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        axis_norm = 0.0
    axis_unit = (
        axis / max(axis_norm, 1e-30)
        if axis_norm > 0.0
        else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    )
    vel = np.asarray(cell_velocity, dtype=np.float64)
    if vel.ndim == 2 and vel.shape[0] == cell_volumes.size:
        vel_proj = np.asarray(vel @ axis_unit, dtype=np.float64)
        pos_proj = vel_proj[vel_proj > 1e-12]
        if pos_proj.size:
            speed_char = float(np.percentile(pos_proj, 70.0))
        else:
            speed_char = float(np.percentile(np.abs(vel_proj), 70.0))
    else:
        speed_char = 0.0
    if speed_char <= 1e-12:
        speed_char = float(inlet_flux_in / max(domain_volume, 1e-30))

    travel_time_est = (
        float(axis_norm / max(speed_char, 1e-30))
        if axis_norm > 0.0
        else float(domain_volume / max(inlet_flux_in, 1e-30))
    )
    breakthrough_expected = bool(
        (turnover >= float(breakthrough_turnover_threshold))
        or (
            float(physical_time_final)
            >= float(breakthrough_travel_fraction) * max(travel_time_est, 1e-30)
        )
    )
    return {
        "domain_volume": float(domain_volume),
        "inlet_flux_in": float(inlet_flux_in),
        "transport_turnover_count": float(turnover),
        "breakthrough_turnover_threshold": float(breakthrough_turnover_threshold),
        "inlet_outlet_distance": float(axis_norm),
        "characteristic_streamwise_speed": float(speed_char),
        "travel_time_estimate": float(travel_time_est),
        "breakthrough_travel_fraction_threshold": float(breakthrough_travel_fraction),
        "breakthrough_expected": bool(breakthrough_expected),
    }


def _evaluate_transport_front_observation(
    *,
    mesh,
    cell_velocity: np.ndarray,
    flux_diag: Dict[str, float],
    physical_time_final: float,
    outlet_arrival: Dict[str, Any],
    outlet_c_mean: float,
    breakthrough_turnover_threshold: float = 0.20,
    breakthrough_travel_fraction: float = 0.35,
    outlet_fraction_detection_threshold: float = 1e-4,
) -> Dict[str, Any]:
    """Add post-run breakthrough observations to a pre-run timing estimate."""
    timing = _estimate_transport_front_timing(
        mesh=mesh,
        cell_velocity=cell_velocity,
        flux_diag=flux_diag,
        physical_time_final=physical_time_final,
        breakthrough_turnover_threshold=breakthrough_turnover_threshold,
        breakthrough_travel_fraction=breakthrough_travel_fraction,
    )
    outlet_frac = float(_read_outlet_fraction(outlet_arrival, 1e-6, default=0.0))
    breakthrough_detected = bool(
        outlet_frac >= float(outlet_fraction_detection_threshold)
        or float(outlet_c_mean) > 1e-9
    )
    observation = "breakthrough_detected"
    reason = "outlet breakthrough detected"
    if not breakthrough_detected:
        observation = "breakthrough_not_detected"
        reason = "outlet concentration remains below the breakthrough threshold"
    return {
        **timing,
        "outlet_fraction_detection_threshold": float(
            outlet_fraction_detection_threshold
        ),
        "breakthrough_detected": bool(breakthrough_detected),
        "outlet_frac_gt_1e-6": float(outlet_frac),
        "observation": str(observation),
        "observation_reason": str(reason),
    }


def _estimate_transport_run_time(
    *,
    mesh,
    cell_velocity: np.ndarray,
    face_normal_velocity: np.ndarray,
    flux_diag: Dict[str, float],
    transport_config,
    resolve_transport_dt_control_fn,
    resume_state: dict[str, Any] | None,
    breakthrough_turnover_threshold: float,
    breakthrough_travel_fraction: float,
    max_walltime_seconds: float,
    transport_end_time: float | None = None,
) -> Dict[str, Any]:
    dt_control = dict(
        resolve_transport_dt_control_fn(
            mesh,
            np.asarray(face_normal_velocity, dtype=np.float64),
            config=transport_config,
        )
    )
    requested_total_steps = int(transport_config.steps)
    requested_dt = float(dt_control.get("requested_dt", transport_config.dt))
    used_dt = float(dt_control.get("used_dt", 0.0))
    controller_blocked = bool(dt_control.get("transport_dt_controller_blocked", False))
    controller_cap_hit = bool(dt_control.get("transport_substep_cap_hit", False))
    controller_warning = bool(dt_control.get("transport_substep_warning", False))

    completed_steps = int(resume_state.get("step", 0)) if resume_state else 0
    existing_physical_time = (
        float(resume_state.get("physical_time", 0.0)) if resume_state else 0.0
    )
    if transport_end_time is None:
        run_horizon_mode = "steps"
        requested_end_time = None
        planned_total_steps = int(requested_total_steps)
        remaining_steps = max(0, planned_total_steps - completed_steps)
        final_outer_dt = None
        projected_additional_physical_time = float(remaining_steps * used_dt)
    else:
        run_horizon_mode = "end_time"
        requested_end_time = float(transport_end_time)
        remaining_physical_time = float(requested_end_time - existing_physical_time)
        if remaining_physical_time <= 1e-15:
            raise ValueError(
                "transport_end_time must be greater than the current checkpoint "
                "physical time."
            )
        if used_dt <= 0.0:
            raise RuntimeError(
                "Transport timestep controller produced non-positive dt."
            )
        full_steps = int(np.floor(remaining_physical_time / used_dt + 1e-12))
        remainder = float(remaining_physical_time - full_steps * used_dt)
        tolerance = max(1e-15, 1e-12 * max(used_dt, requested_end_time))
        if remainder <= tolerance:
            final_outer_dt = None
            remaining_steps = max(1, full_steps)
            projected_additional_physical_time = float(remaining_steps * used_dt)
        else:
            final_outer_dt = remainder
            remaining_steps = full_steps + 1
            projected_additional_physical_time = float(
                full_steps * used_dt + final_outer_dt
            )
        planned_total_steps = int(completed_steps + remaining_steps)
    projected_physical_time_final = float(
        existing_physical_time + projected_additional_physical_time
    )

    front_estimate = _estimate_transport_front_timing(
        mesh=mesh,
        cell_velocity=np.asarray(cell_velocity, dtype=np.float64),
        flux_diag=flux_diag,
        physical_time_final=projected_physical_time_final,
        breakthrough_turnover_threshold=float(breakthrough_turnover_threshold),
        breakthrough_travel_fraction=float(breakthrough_travel_fraction),
    )
    return {
        **dict(front_estimate),
        "requested_total_steps": int(requested_total_steps),
        "run_horizon_mode": run_horizon_mode,
        "requested_transport_end_time": requested_end_time,
        "planned_total_steps": int(planned_total_steps),
        "final_outer_dt": final_outer_dt,
        "resume_from_step": int(completed_steps),
        "remaining_steps": int(remaining_steps),
        "requested_dt": float(requested_dt),
        "used_dt": float(used_dt),
        "stable_dt_estimate": float(dt_control.get("stable_dt_estimate", float("nan"))),
        "dt_mode": str(dt_control.get("dt_mode", transport_config.dt_mode)),
        "cfl_target": float(dt_control.get("cfl_target", transport_config.cfl_target)),
        "existing_physical_time": float(existing_physical_time),
        "projected_additional_physical_time": float(projected_additional_physical_time),
        "projected_physical_time_final": float(projected_physical_time_final),
        "max_walltime_seconds": float(max(max_walltime_seconds, 0.0)),
        "dt_control": dt_control,
        "transport_substep_cap_hit": controller_cap_hit,
        "transport_dt_controller_blocked": controller_blocked,
        "transport_substep_warning": controller_warning,
    }


def _enforce_transport_dt_controller(
    *,
    time_estimate: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(time_estimate.get("transport_dt_controller_blocked", False)):
        reason = str(
            time_estimate.get("dt_control", {}).get(
                "transport_dt_controller_reason",
                "transport dt controller blocked requested run",
            )
        )
        raise RuntimeError(
            f"Transport timestep controller rejected requested run: {reason}."
        )
    return time_estimate


def _compute_face_flux_diagnostics_from_face_normal_velocity(
    mesh,
    face_normal_velocity: np.ndarray,
    *,
    left_inlet_faces: np.ndarray | None = None,
    right_inlet_faces: np.ndarray | None = None,
) -> Dict[str, float]:
    vn = np.asarray(face_normal_velocity, dtype=np.float64)
    face_areas = np.asarray(mesh.face_areas, dtype=np.float64)
    flux = vn * face_areas

    if left_inlet_faces is None or right_inlet_faces is None:
        inlet_fallback = np.asarray(getattr(mesh, "inlet_faces", []), dtype=np.int64)
        left_faces = inlet_fallback
        right_faces = np.asarray([], dtype=np.int64)
    else:
        left_faces = np.asarray(left_inlet_faces, dtype=np.int64)
        right_faces = np.asarray(right_inlet_faces, dtype=np.int64)
    inlet_faces = np.concatenate([left_faces, right_faces]).astype(np.int64, copy=False)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)

    inlet_in = float(-np.sum(flux[inlet_faces])) if inlet_faces.size else 0.0
    inlet_out = (
        float(np.sum(np.maximum(flux[inlet_faces], 0.0))) if inlet_faces.size else 0.0
    )
    outlet_out = float(np.sum(flux[outlet_faces])) if outlet_faces.size else 0.0
    outlet_in = (
        float(-np.sum(np.minimum(flux[outlet_faces], 0.0)))
        if outlet_faces.size
        else 0.0
    )
    net_boundary_flux = (
        float(np.sum(flux[inlet_faces]) + np.sum(flux[outlet_faces]))
        if (inlet_faces.size or outlet_faces.size)
        else 0.0
    )
    wall_flux_max_abs = (
        float(np.max(np.abs(flux[wall_faces]))) if wall_faces.size else 0.0
    )
    ref = max(abs(inlet_in), abs(outlet_out), 1e-30)
    imbalance_ratio = float(abs(net_boundary_flux) / ref)

    return {
        "total_inlet_flux_in": float(inlet_in),
        "total_inlet_flux_out": float(inlet_out),
        "total_outlet_flux_out": float(outlet_out),
        "total_outlet_flux_in": float(outlet_in),
        "net_boundary_flux": float(net_boundary_flux),
        "boundary_flux_imbalance_ratio": float(imbalance_ratio),
        "wall_flux_max_abs": float(wall_flux_max_abs),
    }


def _load_flow_coupling_payload(flow_run_dir: Path) -> Dict[str, Any]:
    return _shared_load_flow_coupling_payload(flow_run_dir)


def _check_flow_transport_mesh_compatibility(
    *,
    mesh,
    flow_payload: Dict[str, Any],
) -> Dict[str, Any]:
    return _shared_check_flow_mesh_compatibility(mesh=mesh, flow_payload=flow_payload)


def _validate_source_flow_readiness(
    metadata: Dict[str, Any],
    *,
    strict: bool,
) -> Dict[str, Any]:
    return _shared_validate_source_flow_readiness(
        metadata,
        strict=bool(strict),
        strict_label="transport source flow",
    )


def _split_inlet_faces_by_x(
    mesh, inlet_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(inlet_faces, dtype=np.int64)
    if faces.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    x = np.asarray(mesh.face_centers[faces, 0], dtype=np.float64)
    x_mid = float(np.median(x))
    left = faces[x <= x_mid]
    right = faces[x > x_mid]
    if left.size == 0 or right.size == 0:
        order = np.argsort(x)
        half = max(1, faces.size // 2)
        left = faces[order[:half]]
        right = faces[order[half:]]
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def _save_transport_checkpoint(
    *,
    run_dir: Path,
    step: int,
    concentration: np.ndarray,
    cumulative_mass_in: float,
    cumulative_mass_out: float,
    physical_time: float,
    used_dt: float,
    history_tail: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = run_dir / "checkpoints" / f"step_{int(step):06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    conc_path = ckpt_dir / "concentration.npy"
    np.save(conc_path, np.asarray(concentration, dtype=np.float64))
    payload: Dict[str, Any] = {
        "step": int(step),
        "concentration_npy": str(conc_path),
        "cumulative_mass_in": float(cumulative_mass_in),
        "cumulative_mass_out": float(cumulative_mass_out),
        "physical_time": float(physical_time),
        "used_dt": float(used_dt),
        "history_tail": dict(history_tail) if history_tail else {},
        "rng_state": None,
    }
    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)
    state_path = ckpt_dir / "checkpoint_state.json"
    _write_json(state_path, payload)
    return state_path


def _load_transport_checkpoint(path: Path) -> dict[str, Any]:
    p = Path(path).resolve()
    if p.is_dir():
        p = p / "checkpoint_state.json"
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint state not found: {p}")
    state = json.loads(p.read_text(encoding="utf-8"))
    conc_path = Path(str(state.get("concentration_npy", "")))
    if not conc_path.is_absolute():
        conc_path = (p.parent / conc_path).resolve()
    if not conc_path.exists():
        raise FileNotFoundError(f"Checkpoint concentration file not found: {conc_path}")
    concentration = np.asarray(np.load(conc_path), dtype=np.float64)
    return {
        "state_path": str(p),
        "step": int(state.get("step", 0)),
        "cumulative_mass_in": float(state.get("cumulative_mass_in", 0.0)),
        "cumulative_mass_out": float(state.get("cumulative_mass_out", 0.0)),
        "physical_time": float(state.get("physical_time", 0.0)),
        "used_dt": float(state.get("used_dt", 0.0)),
        "history_tail": dict(state.get("history_tail", {}))
        if isinstance(state.get("history_tail", {}), dict)
        else {},
        "metadata": dict(state.get("metadata", {}))
        if isinstance(state.get("metadata", {}), dict)
        else {},
        "concentration": concentration,
    }


def _save_transport_previews(
    *,
    centers: np.ndarray,
    concentration: np.ndarray,
    velocity_magnitude: np.ndarray,
    clipped_mask: np.ndarray | None,
    overshoot_mask: np.ndarray | None,
    undershoot_mask: np.ndarray | None,
    left_inlet_cells: np.ndarray | None,
    right_inlet_cells: np.ndarray | None,
    outlet_cells: np.ndarray | None,
    left_inlet_zone_mask: np.ndarray | None,
    right_inlet_zone_mask: np.ndarray | None,
    inlet_wall_corner_zone_mask: np.ndarray | None,
    boundary_faces: np.ndarray | None,
    boundary_face_groups: np.ndarray | None,
    run_dir: Path,
    file_prefix: str,
) -> Dict[str, str]:
    previews: Dict[str, str] = {}

    fig_c, ax_c = plt.subplots(figsize=(7.2, 5.6))
    sc_c = ax_c.scatter(
        centers[:, 0],
        centers[:, 1],
        c=concentration,
        s=2.0,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax_c.set_title("Concentration XY (cell centers)")
    ax_c.set_xlabel("x [m]")
    ax_c.set_ylabel("y [m]")
    ax_c.set_aspect("equal", adjustable="box")
    fig_c.colorbar(sc_c, ax=ax_c, label="C")
    fig_c.tight_layout()
    path_c = run_dir / f"{file_prefix}_concentration_xy.png"
    fig_c.savefig(path_c, dpi=180)
    plt.close(fig_c)
    previews["concentration_xy"] = str(path_c)

    z = centers[:, 2]
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
    z_span = float(np.max(z) - np.min(z))
    tol = max(0.02 * z_span, 1e-6)
    mask_mid = np.abs(z - z_mid) <= tol
    if not np.any(mask_mid):
        idx = np.argsort(np.abs(z - z_mid))[: max(100, centers.shape[0] // 50)]
        mask_mid = np.zeros(centers.shape[0], dtype=bool)
        mask_mid[idx] = True

    fig_cm, ax_cm = plt.subplots(figsize=(7.2, 5.6))
    sc_cm = ax_cm.scatter(
        centers[mask_mid, 0],
        centers[mask_mid, 1],
        c=concentration[mask_mid],
        s=4.0,
        cmap="viridis",
        alpha=0.9,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax_cm.set_title("Concentration Mid-Z XY")
    ax_cm.set_xlabel("x [m]")
    ax_cm.set_ylabel("y [m]")
    ax_cm.set_aspect("equal", adjustable="box")
    fig_cm.colorbar(sc_cm, ax=ax_cm, label="C")
    fig_cm.tight_layout()
    path_cm = run_dir / f"{file_prefix}_concentration_mid_z_xy.png"
    fig_cm.savefig(path_cm, dpi=180)
    plt.close(fig_cm)
    previews["concentration_mid_z_xy"] = str(path_cm)

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
    ax_v.set_title("Velocity Magnitude XY")
    ax_v.set_xlabel("x [m]")
    ax_v.set_ylabel("y [m]")
    ax_v.set_aspect("equal", adjustable="box")
    fig_v.colorbar(sc_v, ax=ax_v, label="|u| [m/s]")
    fig_v.tight_layout()
    path_v = run_dir / f"{file_prefix}_velocity_magnitude_xy.png"
    fig_v.savefig(path_v, dpi=180)
    plt.close(fig_v)
    previews["velocity_magnitude_xy"] = str(path_v)

    if clipped_mask is not None:
        cm = np.asarray(clipped_mask, dtype=bool)
        fig_clip, ax_clip = plt.subplots(figsize=(7.2, 5.6))
        ax_clip.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.4, alpha=0.25, linewidths=0
        )
        if np.any(cm):
            ax_clip.scatter(
                centers[cm, 0],
                centers[cm, 1],
                c="#d62728",
                s=7.0,
                alpha=0.9,
                linewidths=0,
            )
        ax_clip.set_title("Clipped Cells XY (final step)")
        ax_clip.set_xlabel("x [m]")
        ax_clip.set_ylabel("y [m]")
        ax_clip.set_aspect("equal", adjustable="box")
        fig_clip.tight_layout()
        path_clip = run_dir / f"{file_prefix}_clipped_cells_xy.png"
        fig_clip.savefig(path_clip, dpi=180)
        plt.close(fig_clip)
        previews["clipped_cells_xy"] = str(path_clip)

    if overshoot_mask is not None:
        om = np.asarray(overshoot_mask, dtype=bool)
        fig_over, ax_over = plt.subplots(figsize=(7.2, 5.6))
        ax_over.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.4, alpha=0.25, linewidths=0
        )
        if np.any(om):
            ax_over.scatter(
                centers[om, 0],
                centers[om, 1],
                c="#ff7f0e",
                s=7.0,
                alpha=0.9,
                linewidths=0,
            )
        ax_over.set_title("Overshoot Cells XY (final step)")
        ax_over.set_xlabel("x [m]")
        ax_over.set_ylabel("y [m]")
        ax_over.set_aspect("equal", adjustable="box")
        fig_over.tight_layout()
        path_over = run_dir / f"{file_prefix}_overshoot_cells_xy.png"
        fig_over.savefig(path_over, dpi=180)
        plt.close(fig_over)
        previews["overshoot_cells_xy"] = str(path_over)

    if undershoot_mask is not None:
        um = np.asarray(undershoot_mask, dtype=bool)
        fig_under, ax_under = plt.subplots(figsize=(7.2, 5.6))
        ax_under.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.4, alpha=0.25, linewidths=0
        )
        if np.any(um):
            ax_under.scatter(
                centers[um, 0],
                centers[um, 1],
                c="#1f77b4",
                s=7.0,
                alpha=0.9,
                linewidths=0,
            )
        ax_under.set_title("Undershoot Cells XY (final step)")
        ax_under.set_xlabel("x [m]")
        ax_under.set_ylabel("y [m]")
        ax_under.set_aspect("equal", adjustable="box")
        fig_under.tight_layout()
        path_under = run_dir / f"{file_prefix}_undershoot_cells_xy.png"
        fig_under.savefig(path_under, dpi=180)
        plt.close(fig_under)
        previews["undershoot_cells_xy"] = str(path_under)

    if (
        left_inlet_cells is not None
        and right_inlet_cells is not None
        and outlet_cells is not None
    ):
        left = np.asarray(left_inlet_cells, dtype=np.int64)
        right = np.asarray(right_inlet_cells, dtype=np.int64)
        outlet = np.asarray(outlet_cells, dtype=np.int64)
        fig_io, ax_io = plt.subplots(figsize=(7.2, 5.6))
        ax_io.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.2, alpha=0.18, linewidths=0
        )
        if left.size:
            ax_io.scatter(
                centers[left, 0],
                centers[left, 1],
                c="#17becf",
                s=7.0,
                linewidths=0,
                label="left inlet",
            )
        if right.size:
            ax_io.scatter(
                centers[right, 0],
                centers[right, 1],
                c="#2ca02c",
                s=7.0,
                linewidths=0,
                label="right inlet",
            )
        if outlet.size:
            ax_io.scatter(
                centers[outlet, 0],
                centers[outlet, 1],
                c="#9467bd",
                s=7.0,
                linewidths=0,
                label="outlet",
            )
        ax_io.set_title("Inlet / Outlet Boundary-Adjacent Cells XY")
        ax_io.set_xlabel("x [m]")
        ax_io.set_ylabel("y [m]")
        ax_io.set_aspect("equal", adjustable="box")
        if left.size or right.size or outlet.size:
            ax_io.legend(loc="best", frameon=False)
        fig_io.tight_layout()
        path_io = run_dir / "inlet_outlet_boundary_cells_xy.png"
        fig_io.savefig(path_io, dpi=180)
        plt.close(fig_io)
        previews["inlet_outlet_boundary_cells_xy"] = str(path_io)

        if right.size:
            x0 = float(np.min(centers[right, 0]))
            x1 = float(np.max(centers[right, 0]))
            y0 = float(np.min(centers[right, 1]))
            y1 = float(np.max(centers[right, 1]))
            x_pad = max(0.15 * (x1 - x0), 5e-4)
            y_pad = max(0.15 * (y1 - y0), 5e-4)
            x_low, x_high = x0 - x_pad, x1 + x_pad
            y_low, y_high = y0 - y_pad, y1 + y_pad
            zoom_mask = (
                (centers[:, 0] >= x_low)
                & (centers[:, 0] <= x_high)
                & (centers[:, 1] >= y_low)
                & (centers[:, 1] <= y_high)
            )
            if np.any(zoom_mask):
                fig_zoom, ax_zoom = plt.subplots(figsize=(7.2, 5.6))
                sc_zoom = ax_zoom.scatter(
                    centers[zoom_mask, 0],
                    centers[zoom_mask, 1],
                    c=concentration[zoom_mask],
                    s=5.0,
                    cmap="viridis",
                    alpha=0.92,
                    linewidths=0,
                    vmin=0.0,
                    vmax=1.0,
                )
                ax_zoom.set_title("Concentration Zoom Near Right Inlet XY")
                ax_zoom.set_xlabel("x [m]")
                ax_zoom.set_ylabel("y [m]")
                ax_zoom.set_aspect("equal", adjustable="box")
                fig_zoom.colorbar(sc_zoom, ax=ax_zoom, label="C")
                fig_zoom.tight_layout()
                path_zoom = run_dir / "concentration_final_zoom_right_inlet_xy.png"
                fig_zoom.savefig(path_zoom, dpi=180)
                plt.close(fig_zoom)
                previews["concentration_final_zoom_right_inlet_xy"] = str(path_zoom)

    if (
        clipped_mask is not None
        and overshoot_mask is not None
        and left_inlet_zone_mask is not None
        and right_inlet_zone_mask is not None
    ):
        inlet_zone = np.asarray(left_inlet_zone_mask, dtype=bool) | np.asarray(
            right_inlet_zone_mask, dtype=bool
        )
        clipped_inlet = np.asarray(clipped_mask, dtype=bool) & inlet_zone
        overshoot_inlet = np.asarray(overshoot_mask, dtype=bool) & inlet_zone

        fig_ic, ax_ic = plt.subplots(figsize=(7.2, 5.6))
        ax_ic.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.2, alpha=0.18, linewidths=0
        )
        if np.any(inlet_zone):
            ax_ic.scatter(
                centers[inlet_zone, 0],
                centers[inlet_zone, 1],
                c="#7f7f7f",
                s=2.0,
                alpha=0.5,
                linewidths=0,
            )
        if np.any(clipped_inlet):
            ax_ic.scatter(
                centers[clipped_inlet, 0],
                centers[clipped_inlet, 1],
                c="#d62728",
                s=8.0,
                alpha=0.95,
                linewidths=0,
            )
        ax_ic.set_title("Inlet-Zone Clipped Cells XY")
        ax_ic.set_xlabel("x [m]")
        ax_ic.set_ylabel("y [m]")
        ax_ic.set_aspect("equal", adjustable="box")
        fig_ic.tight_layout()
        path_ic = run_dir / "inlet_zone_clipped_cells_xy.png"
        fig_ic.savefig(path_ic, dpi=180)
        plt.close(fig_ic)
        previews["inlet_zone_clipped_cells_xy"] = str(path_ic)

        fig_iover, ax_iover = plt.subplots(figsize=(7.2, 5.6))
        ax_iover.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.2, alpha=0.18, linewidths=0
        )
        if np.any(inlet_zone):
            ax_iover.scatter(
                centers[inlet_zone, 0],
                centers[inlet_zone, 1],
                c="#7f7f7f",
                s=2.0,
                alpha=0.5,
                linewidths=0,
            )
        if np.any(overshoot_inlet):
            ax_iover.scatter(
                centers[overshoot_inlet, 0],
                centers[overshoot_inlet, 1],
                c="#ff7f0e",
                s=8.0,
                alpha=0.95,
                linewidths=0,
            )
        ax_iover.set_title("Inlet-Zone Overshoot Cells XY")
        ax_iover.set_xlabel("x [m]")
        ax_iover.set_ylabel("y [m]")
        ax_iover.set_aspect("equal", adjustable="box")
        fig_iover.tight_layout()
        path_iover = run_dir / "inlet_zone_overshoot_cells_xy.png"
        fig_iover.savefig(path_iover, dpi=180)
        plt.close(fig_iover)
        previews["inlet_zone_overshoot_cells_xy"] = str(path_iover)

    if (
        left_inlet_zone_mask is not None
        and right_inlet_zone_mask is not None
        and inlet_wall_corner_zone_mask is not None
    ):
        left_zone = np.asarray(left_inlet_zone_mask, dtype=bool)
        right_zone = np.asarray(right_inlet_zone_mask, dtype=bool)
        corner_zone = np.asarray(inlet_wall_corner_zone_mask, dtype=bool)

        if np.any(right_zone):
            xr0, xr1 = (
                float(np.min(centers[right_zone, 0])),
                float(np.max(centers[right_zone, 0])),
            )
            yr0, yr1 = (
                float(np.min(centers[right_zone, 1])),
                float(np.max(centers[right_zone, 1])),
            )
            xpad = max(0.2 * (xr1 - xr0), 5e-4)
            ypad = max(0.2 * (yr1 - yr0), 5e-4)
            right_zoom = (
                (centers[:, 0] >= xr0 - xpad)
                & (centers[:, 0] <= xr1 + xpad)
                & (centers[:, 1] >= yr0 - ypad)
                & (centers[:, 1] <= yr1 + ypad)
            )
            fig_rz, ax_rz = plt.subplots(figsize=(7.2, 5.6))
            ax_rz.scatter(
                centers[right_zoom, 0],
                centers[right_zoom, 1],
                c="#d0d0d0",
                s=2.0,
                alpha=0.3,
                linewidths=0,
            )
            ax_rz.scatter(
                centers[right_zone, 0],
                centers[right_zone, 1],
                c="#2ca02c",
                s=6.0,
                alpha=0.9,
                linewidths=0,
            )
            if np.any(corner_zone):
                cz = corner_zone & right_zoom
                if np.any(cz):
                    ax_rz.scatter(
                        centers[cz, 0],
                        centers[cz, 1],
                        c="#d62728",
                        s=8.0,
                        alpha=0.95,
                        linewidths=0,
                    )
            ax_rz.set_title("Right Inlet Corner Zoom XY")
            ax_rz.set_xlabel("x [m]")
            ax_rz.set_ylabel("y [m]")
            ax_rz.set_aspect("equal", adjustable="box")
            fig_rz.tight_layout()
            path_rz = run_dir / "right_inlet_corner_zoom_xy.png"
            fig_rz.savefig(path_rz, dpi=180)
            plt.close(fig_rz)
            previews["right_inlet_corner_zoom_xy"] = str(path_rz)

        if np.any(left_zone):
            xl0, xl1 = (
                float(np.min(centers[left_zone, 0])),
                float(np.max(centers[left_zone, 0])),
            )
            yl0, yl1 = (
                float(np.min(centers[left_zone, 1])),
                float(np.max(centers[left_zone, 1])),
            )
            xpad = max(0.2 * (xl1 - xl0), 5e-4)
            ypad = max(0.2 * (yl1 - yl0), 5e-4)
            left_zoom = (
                (centers[:, 0] >= xl0 - xpad)
                & (centers[:, 0] <= xl1 + xpad)
                & (centers[:, 1] >= yl0 - ypad)
                & (centers[:, 1] <= yl1 + ypad)
            )
            fig_lz, ax_lz = plt.subplots(figsize=(7.2, 5.6))
            ax_lz.scatter(
                centers[left_zoom, 0],
                centers[left_zoom, 1],
                c="#d0d0d0",
                s=2.0,
                alpha=0.3,
                linewidths=0,
            )
            ax_lz.scatter(
                centers[left_zone, 0],
                centers[left_zone, 1],
                c="#17becf",
                s=6.0,
                alpha=0.9,
                linewidths=0,
            )
            if np.any(corner_zone):
                cz = corner_zone & left_zoom
                if np.any(cz):
                    ax_lz.scatter(
                        centers[cz, 0],
                        centers[cz, 1],
                        c="#d62728",
                        s=8.0,
                        alpha=0.95,
                        linewidths=0,
                    )
            ax_lz.set_title("Left Inlet Corner Zoom XY")
            ax_lz.set_xlabel("x [m]")
            ax_lz.set_ylabel("y [m]")
            ax_lz.set_aspect("equal", adjustable="box")
            fig_lz.tight_layout()
            path_lz = run_dir / "left_inlet_corner_zoom_xy.png"
            fig_lz.savefig(path_lz, dpi=180)
            plt.close(fig_lz)
            previews["left_inlet_corner_zoom_xy"] = str(path_lz)

    if boundary_faces is not None and boundary_face_groups is not None:
        # boundary groups preview based on boundary-adjacent cell zones.
        fig_bg, ax_bg = plt.subplots(figsize=(7.2, 5.6))
        ax_bg.scatter(
            centers[:, 0], centers[:, 1], c="#d0d0d0", s=1.0, alpha=0.12, linewidths=0
        )
        if left_inlet_zone_mask is not None:
            lz = np.asarray(left_inlet_zone_mask, dtype=bool)
            ax_bg.scatter(
                centers[lz, 0],
                centers[lz, 1],
                c="#17becf",
                s=3.2,
                alpha=0.85,
                linewidths=0,
            )
        if right_inlet_zone_mask is not None:
            rz = np.asarray(right_inlet_zone_mask, dtype=bool)
            ax_bg.scatter(
                centers[rz, 0],
                centers[rz, 1],
                c="#2ca02c",
                s=3.2,
                alpha=0.85,
                linewidths=0,
            )
        if outlet_cells is not None:
            oz = np.asarray(outlet_cells, dtype=np.int64)
            if oz.size:
                ax_bg.scatter(
                    centers[oz, 0],
                    centers[oz, 1],
                    c="#9467bd",
                    s=3.2,
                    alpha=0.85,
                    linewidths=0,
                )
        if inlet_wall_corner_zone_mask is not None:
            cz = np.asarray(inlet_wall_corner_zone_mask, dtype=bool)
            if np.any(cz):
                ax_bg.scatter(
                    centers[cz, 0],
                    centers[cz, 1],
                    c="#d62728",
                    s=4.0,
                    alpha=0.95,
                    linewidths=0,
                )
        ax_bg.set_title("Boundary Face Groups XY (cell-adjacent zones)")
        ax_bg.set_xlabel("x [m]")
        ax_bg.set_ylabel("y [m]")
        ax_bg.set_aspect("equal", adjustable="box")
        fig_bg.tight_layout()
        path_bg = run_dir / f"{file_prefix}_boundary_face_groups_xy.png"
        fig_bg.savefig(path_bg, dpi=180)
        plt.close(fig_bg)
        previews["boundary_face_groups_xy"] = str(path_bg)
    return previews


def _save_concentration_xy_snapshot(
    *,
    centers: np.ndarray,
    concentration: np.ndarray,
    output_path: Path,
    title: str,
) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=concentration,
        s=2.0,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="C")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _save_snapshots_comparison_xy(
    *,
    centers: np.ndarray,
    snapshots: Dict[int, np.ndarray],
    output_path: Path,
) -> str | None:
    if len(snapshots) == 0:
        return None
    ordered_steps = sorted(snapshots)
    fig, axes = plt.subplots(
        1, len(ordered_steps), figsize=(6.2 * len(ordered_steps), 5.2)
    )
    if len(ordered_steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, ordered_steps):
        sc = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            c=snapshots[step],
            s=2.0,
            cmap="viridis",
            alpha=0.85,
            linewidths=0,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"step {step}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=axes, label="C")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _save_two_field_snapshot_comparison_xy(
    *,
    centers: np.ndarray,
    conc_old: np.ndarray,
    conc_new: np.ndarray,
    old_label: str,
    new_label: str,
    output_path: Path,
    step: int,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)
    im0 = axes[0].scatter(
        centers[:, 0],
        centers[:, 1],
        c=conc_old,
        s=1.8,
        cmap="viridis",
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(f"{old_label} at step {step}")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(
        centers[:, 0],
        centers[:, 1],
        c=conc_new,
        s=1.8,
        cmap="viridis",
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(f"{new_label} at step {step}")
    axes[1].set_xlabel("x [m]")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(im0, ax=axes, label="C")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _export_transport_vtu(
    *,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    concentration: np.ndarray,
    velocity: np.ndarray,
    output_path: Path,
) -> Path:
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "meshio is required to export VTU transport results. Install with `pip install meshio`."
        ) from exc

    speed = np.linalg.norm(velocity, axis=1)
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", tetrahedra.astype(np.int64, copy=False))],
        cell_data={
            "concentration": [concentration.astype(np.float64, copy=False)],
            "velocity": [velocity.astype(np.float64, copy=False)],
            "velocity_magnitude": [speed.astype(np.float64, copy=False)],
        },
    )
    meshio.write(output_path, mesh)
    return output_path


def _snapshot_front_diagnostics(
    centers: np.ndarray, concentration: np.ndarray
) -> Dict[str, float]:
    y = centers[:, 1]
    x = centers[:, 0]
    z = centers[:, 2]
    thresholds = (1e-6, 1e-4, 1e-3, 1e-2)
    out: Dict[str, float] = {}
    for thr in thresholds:
        mask = concentration > thr
        key = format_threshold_key("max_y_where_C_gt", thr)
        out[key] = float(np.max(y[mask])) if np.any(mask) else float("nan")
    mask_x = concentration > 1e-4
    if np.any(mask_x):
        out["min_x_where_C_gt_1e-4"] = float(np.min(x[mask_x]))
        out["max_x_extent_where_C_gt_1e-4"] = float(
            np.max(x[mask_x]) - np.min(x[mask_x])
        )
        out["max_x_where_C_gt_1e-4"] = float(np.max(x[mask_x]))
        out["min_y_where_C_gt_1e-4"] = float(np.min(y[mask_x]))
        out["max_y_where_C_gt_1e-4"] = float(np.max(y[mask_x]))
        out["min_z_where_C_gt_1e-4"] = float(np.min(z[mask_x]))
        out["max_z_where_C_gt_1e-4"] = float(np.max(z[mask_x]))
    else:
        out["min_x_where_C_gt_1e-4"] = float("nan")
        out["max_x_extent_where_C_gt_1e-4"] = float("nan")
        out["max_x_where_C_gt_1e-4"] = float("nan")
        out["min_y_where_C_gt_1e-4"] = float("nan")
        out["max_y_where_C_gt_1e-4"] = float("nan")
        out["min_z_where_C_gt_1e-4"] = float("nan")
        out["max_z_where_C_gt_1e-4"] = float("nan")
    w = np.maximum(concentration, 0.0)
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        out["concentration_centroid_x"] = float(np.sum(w * centers[:, 0]) / w_sum)
        out["concentration_centroid_y"] = float(np.sum(w * centers[:, 1]) / w_sum)
        out["concentration_centroid_z"] = float(np.sum(w * centers[:, 2]) / w_sum)
    else:
        out["concentration_centroid_x"] = float("nan")
        out["concentration_centroid_y"] = float("nan")
        out["concentration_centroid_z"] = float("nan")
    return out


def _principal_axis_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-30:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return axis / norm


def _save_inlet_face_scatter(
    *,
    centers: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=values,
        s=22.0,
        cmap=cmap,
        alpha=0.9,
        linewidths=0.2,
        edgecolors="black",
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _save_inlet_inflow_plot(
    *,
    centers: np.ndarray,
    inflow_mask: np.ndarray,
    corner_mask: np.ndarray,
    output_path: Path,
    title: str,
) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(
        centers[:, 0], centers[:, 1], c="#bdbdbd", s=22.0, alpha=0.7, linewidths=0
    )
    if np.any(inflow_mask):
        ax.scatter(
            centers[inflow_mask, 0],
            centers[inflow_mask, 1],
            c="#2ca02c",
            s=30.0,
            alpha=0.95,
            linewidths=0,
            label="inflow faces",
        )
    if np.any(corner_mask):
        ax.scatter(
            centers[corner_mask, 0],
            centers[corner_mask, 1],
            c="#d62728",
            s=42.0,
            alpha=0.95,
            linewidths=0,
            label="corner-adjacent",
        )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    if np.any(inflow_mask) or np.any(corner_mask):
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _save_inlet_profile_plot(
    *,
    local_s: np.ndarray,
    u_n: np.ndarray,
    output_path: Path,
    title: str,
) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(local_s, u_n, marker="o", linewidth=1.2, markersize=3.8, color="#1f77b4")
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("local transverse coordinate s [m]")
    ax.set_ylabel("u·n [m/s]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _inlet_group_audit(
    *,
    mesh,
    group_name: str,
    face_ids: np.ndarray,
    face_normal_velocity: np.ndarray,
    prescribed_scalar_value: float,
    wall_face_ids: np.ndarray,
    run_dir: Path | None,
    save_artifacts: bool = True,
) -> Dict[str, object]:
    faces = np.asarray(face_ids, dtype=np.int64)
    centers = np.asarray(mesh.face_centers[faces], dtype=np.float64)
    vn = np.asarray(face_normal_velocity[faces], dtype=np.float64)
    areas = np.asarray(mesh.face_areas[faces], dtype=np.float64)
    inflow_mask = vn < 0.0
    outflow_mask = vn >= 0.0
    inflow_weights = np.maximum(-vn, 0.0) * areas
    inflow_total = float(np.sum(inflow_weights))
    outflow_total = float(np.sum(np.maximum(vn, 0.0) * areas))
    unweighted_centroid = (
        np.mean(centers, axis=0) if centers.size else np.asarray([np.nan] * 3)
    )
    if inflow_total > 0.0:
        flux_centroid = np.sum(centers * inflow_weights[:, None], axis=0) / inflow_total
    else:
        flux_centroid = np.asarray([np.nan] * 3)

    owner_cells = np.asarray(mesh.face_to_cells[faces, 0], dtype=np.int64)
    wall_owners = set(
        np.asarray(
            mesh.face_to_cells[np.asarray(wall_face_ids, dtype=np.int64), 0]
        ).tolist()
    )
    corner_mask = np.asarray([int(c) in wall_owners for c in owner_cells], dtype=bool)

    prescribed_values = np.full(faces.shape[0], np.nan, dtype=np.float64)
    prescribed_values[inflow_mask] = float(prescribed_scalar_value)
    scalar_flux = inflow_weights * np.where(
        np.isnan(prescribed_values), 0.0, prescribed_values
    )
    scalar_flux_total = float(np.sum(scalar_flux))
    if scalar_flux_total > 0.0:
        scalar_centroid = (
            np.sum(centers * scalar_flux[:, None], axis=0) / scalar_flux_total
        )
    else:
        scalar_centroid = np.asarray([np.nan] * 3)

    xy = centers[:, :2]
    axis = _principal_axis_xy(xy)
    center_xy = np.mean(xy, axis=0)
    local_s = (xy - center_xy) @ axis
    order = np.argsort(local_s)
    profile_rows: list[Dict[str, object]] = []
    for i in order.tolist():
        profile_rows.append(
            {
                "local_s": float(local_s[i]),
                "x": float(centers[i, 0]),
                "y": float(centers[i, 1]),
                "z": float(centers[i, 2]),
                "face_area": float(areas[i]),
                "u_dot_n": float(vn[i]),
                "prescribed_scalar": (
                    None
                    if np.isnan(prescribed_values[i])
                    else float(prescribed_values[i])
                ),
                "is_inflow_face": bool(inflow_mask[i]),
                "is_corner_face": bool(corner_mask[i]),
            }
        )

    width = float(np.max(local_s) - np.min(local_s)) if local_s.size else 0.0
    if inflow_total > 0.0 and width > 1e-30:
        flux_centroid_xy = np.sum(xy * inflow_weights[:, None], axis=0) / inflow_total
        delta_xy = flux_centroid_xy - center_xy
        delta_transverse = float(np.dot(delta_xy, axis))
    else:
        delta_transverse = 0.0
    symmetry_shift_ratio = float(abs(delta_transverse) / max(width, 1e-30))
    symmetry_warning = bool(symmetry_shift_ratio > 0.10)

    artifacts: Dict[str, str] = {}
    if save_artifacts:
        if run_dir is None:
            raise ValueError("run_dir must be provided when save_artifacts=True.")
        stem = group_name.lower()
        artifacts = {
            "face_normal_velocity_xy": _save_inlet_face_scatter(
                centers=centers,
                values=vn,
                output_path=run_dir / f"{stem}_face_normal_velocity_xy.png",
                title=f"{group_name} Inlet Face Normal Velocity",
                cbar_label="u·n [m/s]",
                cmap="coolwarm",
            ),
            "prescribed_scalar_xy": _save_inlet_face_scatter(
                centers=centers,
                values=np.nan_to_num(prescribed_values, nan=-0.1),
                output_path=run_dir / f"{stem}_prescribed_scalar_xy.png",
                title=f"{group_name} Inlet Prescribed Scalar",
                cbar_label="prescribed scalar (inflow faces)",
                cmap="viridis",
            ),
            "inflow_faces_xy": _save_inlet_inflow_plot(
                centers=centers,
                inflow_mask=inflow_mask,
                corner_mask=inflow_mask & corner_mask,
                output_path=run_dir / f"{stem}_inflow_faces_xy.png",
                title=f"{group_name} Inflow Faces",
            ),
            "profile_u_n_png": _save_inlet_profile_plot(
                local_s=local_s[order],
                u_n=vn[order],
                output_path=run_dir / f"{stem}_profile_u_n.png",
                title=f"{group_name} Inlet u·n Profile",
            ),
        }

    return {
        "number_of_faces": int(faces.size),
        "inflow_face_count": int(np.count_nonzero(inflow_mask)),
        "outflow_face_count": int(np.count_nonzero(outflow_mask)),
        "total_inflow_flux": float(inflow_total),
        "total_outflow_flux": float(outflow_total),
        "flux_weighted_centroid_x": float(flux_centroid[0]),
        "flux_weighted_centroid_y": float(flux_centroid[1]),
        "flux_weighted_centroid_z": float(flux_centroid[2]),
        "unweighted_centroid_x": float(unweighted_centroid[0]),
        "unweighted_centroid_y": float(unweighted_centroid[1]),
        "unweighted_centroid_z": float(unweighted_centroid[2]),
        "min_face_normal_velocity": float(np.min(vn)) if vn.size else float("nan"),
        "max_face_normal_velocity": float(np.max(vn)) if vn.size else float("nan"),
        "mean_face_normal_velocity_on_inflow_faces": (
            float(np.mean(vn[inflow_mask])) if np.any(inflow_mask) else float("nan")
        ),
        "number_of_faces_with_prescribed_scalar": int(np.count_nonzero(inflow_mask)),
        "prescribed_scalar_flux_weighted_centroid_x": float(scalar_centroid[0]),
        "prescribed_scalar_flux_weighted_centroid_y": float(scalar_centroid[1]),
        "prescribed_scalar_flux_weighted_centroid_z": float(scalar_centroid[2]),
        "unique_prescribed_scalar_values_on_group": (
            sorted(
                {
                    float(v)
                    for v in prescribed_values[np.isfinite(prescribed_values)].tolist()
                }
            )
            if np.any(np.isfinite(prescribed_values))
            else []
        ),
        "geometric_center_x": float(center_xy[0]) if center_xy.size else float("nan"),
        "geometric_center_y": float(center_xy[1]) if center_xy.size else float("nan"),
        "flux_weighted_center_x": (
            float(np.sum(xy[:, 0] * inflow_weights) / inflow_total)
            if inflow_total > 0.0
            else float("nan")
        ),
        "flux_weighted_center_y": (
            float(np.sum(xy[:, 1] * inflow_weights) / inflow_total)
            if inflow_total > 0.0
            else float("nan")
        ),
        "delta_transverse_flux_center": float(delta_transverse),
        "inlet_width_transverse": float(width),
        "symmetry_shift_ratio": float(symmetry_shift_ratio),
        "symmetry_shift_warning": symmetry_warning,
        "profile_table": profile_rows,
        "artifacts": artifacts,
    }


def _run_inlet_symmetry_audit(
    *,
    mesh,
    velocity_field,
    face_normal_velocity: np.ndarray,
    run_dir: Path | None,
    left_scalar: float,
    right_scalar: float,
    save_artifacts: bool = True,
) -> Dict[str, object]:
    left_faces = np.asarray(
        velocity_field.boundary_groups["left_inlet_faces"], dtype=np.int64
    )
    right_faces = np.asarray(
        velocity_field.boundary_groups["right_inlet_faces"], dtype=np.int64
    )
    wall_faces = np.asarray(
        velocity_field.boundary_groups["wall_faces"], dtype=np.int64
    )
    left = _inlet_group_audit(
        mesh=mesh,
        group_name="left_inlet",
        face_ids=left_faces,
        face_normal_velocity=face_normal_velocity,
        prescribed_scalar_value=left_scalar,
        wall_face_ids=wall_faces,
        run_dir=run_dir,
        save_artifacts=save_artifacts,
    )
    right = _inlet_group_audit(
        mesh=mesh,
        group_name="right_inlet",
        face_ids=right_faces,
        face_normal_velocity=face_normal_velocity,
        prescribed_scalar_value=right_scalar,
        wall_face_ids=wall_faces,
        run_dir=run_dir,
        save_artifacts=save_artifacts,
    )
    return {
        "left_inlet": left,
        "right_inlet": right,
        "symmetry_warning_any": bool(
            left.get("symmetry_shift_warning", False)
            or right.get("symmetry_shift_warning", False)
        ),
    }


def _compact_inlet_group_metrics(group: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "number_of_faces",
        "inflow_face_count",
        "outflow_face_count",
        "total_inflow_flux",
        "total_outflow_flux",
        "flux_weighted_centroid_x",
        "flux_weighted_centroid_y",
        "flux_weighted_centroid_z",
        "unweighted_centroid_x",
        "unweighted_centroid_y",
        "unweighted_centroid_z",
        "mean_face_normal_velocity_on_inflow_faces",
        "delta_transverse_flux_center",
        "inlet_width_transverse",
        "symmetry_shift_ratio",
        "symmetry_shift_warning",
    ]
    out: Dict[str, object] = {}
    for k in keys:
        out[k] = group.get(k)
    return out


def _build_velocity_zone_masks(mesh) -> Dict[str, np.ndarray]:
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    inlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.inlet_faces, dtype=np.int64), 1])
    )
    outlet_y = float(
        np.median(mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1])
    )
    y_span = max(outlet_y - inlet_y, 1e-12)
    junction_y = inlet_y + 0.08 * y_span

    left_faces = np.asarray(mesh.left_inlet_faces, dtype=np.int64)
    right_faces = np.asarray(mesh.right_inlet_faces, dtype=np.int64)
    left_center_x = (
        float(np.mean(mesh.face_centers[left_faces, 0]))
        if left_faces.size
        else float(np.percentile(x, 25.0))
    )
    right_center_x = (
        float(np.mean(mesh.face_centers[right_faces, 0]))
        if right_faces.size
        else float(np.percentile(x, 75.0))
    )
    x_abs = np.abs(x)
    x_thresh = float(np.percentile(x_abs, 35.0))

    right_inlet_branch = (x >= right_center_x) & (y <= junction_y)
    left_inlet_branch = (x <= left_center_x) & (y <= junction_y)
    junction_zone = (np.abs(y - junction_y) <= 0.12 * y_span) & (x_abs <= x_thresh)
    outlet_branch = (y > junction_y) & (x_abs <= max(x_thresh, 1e-12))
    if not np.any(junction_zone):
        junction_zone = np.abs(y - junction_y) <= 0.15 * y_span
    if not np.any(outlet_branch):
        outlet_branch = y > junction_y

    return {
        "right_inlet_branch": right_inlet_branch,
        "left_inlet_branch": left_inlet_branch,
        "junction_zone": junction_zone,
        "outlet_branch": outlet_branch,
    }


def _velocity_direction_classes(
    centers: np.ndarray,
    velocity: np.ndarray,
    *,
    junction_point: np.ndarray,
) -> Dict[str, np.ndarray]:
    v = np.asarray(velocity, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    near_zero = speed <= max(1e-12, 1e-3 * max(float(np.mean(speed)), 1e-12))

    to_j = np.asarray(junction_point, dtype=np.float64)[None, :] - c
    to_j[:, 2] = 0.0
    v_xy = v.copy()
    v_xy[:, 2] = 0.0
    denom = np.maximum(
        np.linalg.norm(v_xy[:, :2], axis=1) * np.linalg.norm(to_j[:, :2], axis=1),
        1e-12,
    )
    cos = np.sum(v_xy[:, :2] * to_j[:, :2], axis=1) / denom

    toward = (~near_zero) & (cos >= 0.5)
    away = (~near_zero) & (cos <= -0.2)
    transverse = (~near_zero) & (~toward) & (~away)
    return {
        "toward_junction": toward,
        "away_from_junction": away,
        "mostly_transverse": transverse,
        "near_zero": near_zero,
    }


def _save_velocity_vector_previews(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    run_dir: Path,
) -> Dict[str, str]:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    y = c[:, 1]
    inlet_y = float(np.min(y))
    outlet_y = float(np.max(y))
    junction_y = inlet_y + 0.08 * max(outlet_y - inlet_y, 1e-12)
    junction_pt = np.asarray(
        [0.0, junction_y, float(np.mean(c[:, 2]))], dtype=np.float64
    )
    classes = _velocity_direction_classes(c, v, junction_point=junction_pt)

    outputs: Dict[str, str] = {}
    fig_m, ax_m = plt.subplots(figsize=(7.2, 5.6))
    sc_m = ax_m.scatter(
        c[:, 0], c[:, 1], c=speed, s=2.0, cmap="magma", alpha=0.85, linewidths=0
    )
    ax_m.set_title("Velocity Magnitude XY")
    ax_m.set_xlabel("x [m]")
    ax_m.set_ylabel("y [m]")
    ax_m.set_aspect("equal", adjustable="box")
    fig_m.colorbar(sc_m, ax=ax_m, label="|u| [m/s]")
    fig_m.tight_layout()
    path_m = run_dir / "velocity_magnitude_xy.png"
    fig_m.savefig(path_m, dpi=180)
    plt.close(fig_m)
    outputs["velocity_magnitude_xy"] = str(path_m)

    step = max(c.shape[0] // 3500, 1)
    idx = np.arange(0, c.shape[0], step, dtype=np.int64)
    fig_q, ax_q = plt.subplots(figsize=(7.2, 5.6))
    ax_q.quiver(
        c[idx, 0],
        c[idx, 1],
        v[idx, 0],
        v[idx, 1],
        np.linalg.norm(v[idx, :2], axis=1),
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0018,
    )
    ax_q.set_title("Velocity Vectors XY (downsampled)")
    ax_q.set_xlabel("x [m]")
    ax_q.set_ylabel("y [m]")
    ax_q.set_aspect("equal", adjustable="box")
    fig_q.tight_layout()
    path_q = run_dir / "velocity_vectors_xy_downsampled.png"
    fig_q.savefig(path_q, dpi=180)
    plt.close(fig_q)
    outputs["velocity_vectors_xy_downsampled"] = str(path_q)

    right_mask = c[:, 0] >= float(np.percentile(c[:, 0], 75.0))
    right_mask &= c[:, 1] <= float(junction_y + 0.02 * max(outlet_y - inlet_y, 1e-12))
    if not np.any(right_mask):
        right_mask = c[:, 0] >= float(np.percentile(c[:, 0], 70.0))
    ridx = np.where(right_mask)[0][:: max(np.count_nonzero(right_mask) // 700, 1)]
    fig_r, ax_r = plt.subplots(figsize=(7.2, 5.6))
    ax_r.quiver(
        c[ridx, 0],
        c[ridx, 1],
        v[ridx, 0],
        v[ridx, 1],
        np.linalg.norm(v[ridx, :2], axis=1),
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0022,
    )
    ax_r.set_title("Right Inlet Velocity Vectors Zoom XY")
    ax_r.set_xlabel("x [m]")
    ax_r.set_ylabel("y [m]")
    ax_r.set_aspect("equal", adjustable="box")
    fig_r.tight_layout()
    path_r = run_dir / "right_inlet_velocity_vectors_zoom_xy.png"
    fig_r.savefig(path_r, dpi=180)
    plt.close(fig_r)
    outputs["right_inlet_velocity_vectors_zoom_xy"] = str(path_r)

    left_mask = c[:, 0] <= float(np.percentile(c[:, 0], 25.0))
    left_mask &= c[:, 1] <= float(junction_y + 0.02 * max(outlet_y - inlet_y, 1e-12))
    if not np.any(left_mask):
        left_mask = c[:, 0] <= float(np.percentile(c[:, 0], 30.0))
    lidx = np.where(left_mask)[0][:: max(np.count_nonzero(left_mask) // 700, 1)]
    fig_l, ax_l = plt.subplots(figsize=(7.2, 5.6))
    ax_l.quiver(
        c[lidx, 0],
        c[lidx, 1],
        v[lidx, 0],
        v[lidx, 1],
        np.linalg.norm(v[lidx, :2], axis=1),
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0022,
    )
    ax_l.set_title("Left Inlet Velocity Vectors Zoom XY")
    ax_l.set_xlabel("x [m]")
    ax_l.set_ylabel("y [m]")
    ax_l.set_aspect("equal", adjustable="box")
    fig_l.tight_layout()
    path_l = run_dir / "left_inlet_velocity_vectors_zoom_xy.png"
    fig_l.savefig(path_l, dpi=180)
    plt.close(fig_l)
    outputs["left_inlet_velocity_vectors_zoom_xy"] = str(path_l)

    j_mask = np.abs(c[:, 1] - junction_y) <= 0.12 * max(outlet_y - inlet_y, 1e-12)
    jidx = np.where(j_mask)[0][:: max(np.count_nonzero(j_mask) // 800, 1)]
    fig_j, ax_j = plt.subplots(figsize=(7.2, 5.6))
    ax_j.quiver(
        c[jidx, 0],
        c[jidx, 1],
        v[jidx, 0],
        v[jidx, 1],
        np.linalg.norm(v[jidx, :2], axis=1),
        cmap="cividis",
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0022,
    )
    ax_j.set_title("Junction Velocity Vectors Zoom XY")
    ax_j.set_xlabel("x [m]")
    ax_j.set_ylabel("y [m]")
    ax_j.set_aspect("equal", adjustable="box")
    fig_j.tight_layout()
    path_j = run_dir / "junction_velocity_vectors_zoom_xy.png"
    fig_j.savefig(path_j, dpi=180)
    plt.close(fig_j)
    outputs["junction_velocity_vectors_zoom_xy"] = str(path_j)

    class_code = np.full(c.shape[0], -1, dtype=np.int32)
    class_code[classes["near_zero"]] = 0
    class_code[classes["toward_junction"]] = 1
    class_code[classes["away_from_junction"]] = 2
    class_code[classes["mostly_transverse"]] = 3
    cmap = plt.matplotlib.colors.ListedColormap(
        ["#bdbdbd", "#2ca02c", "#d62728", "#1f77b4"]
    )
    fig_cls, ax_cls = plt.subplots(figsize=(7.2, 5.6))
    ax_cls.scatter(
        c[:, 0],
        c[:, 1],
        c=class_code,
        s=3.0,
        cmap=cmap,
        alpha=0.92,
        linewidths=0,
    )
    ax_cls.set_title("Velocity Direction Classification XY")
    ax_cls.set_xlabel("x [m]")
    ax_cls.set_ylabel("y [m]")
    ax_cls.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="near_zero",
            markerfacecolor="#bdbdbd",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="toward_junction",
            markerfacecolor="#2ca02c",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="away_from_junction",
            markerfacecolor="#d62728",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="mostly_transverse",
            markerfacecolor="#1f77b4",
            markersize=7,
        ),
    ]
    ax_cls.legend(handles=handles, loc="upper right", frameon=True)
    fig_cls.tight_layout()
    path_cls = run_dir / "velocity_direction_classification_xy.png"
    fig_cls.savefig(path_cls, dpi=180)
    plt.close(fig_cls)
    outputs["velocity_direction_classification_xy"] = str(path_cls)
    return outputs


def _scalar_front_velocity_audit(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    concentration: np.ndarray,
) -> Dict[str, float]:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    s = np.asarray(concentration, dtype=np.float64)
    mask = s > 1e-4
    out: Dict[str, float] = {
        "scalar_cell_count_gt_1e-4": int(np.count_nonzero(mask)),
    }
    if not np.any(mask):
        out.update(
            {
                "mean_scalar_velocity_x": float("nan"),
                "mean_scalar_velocity_y": float("nan"),
                "mean_scalar_velocity_z": float("nan"),
                "fraction_scalar_velocity_toward_junction": float("nan"),
                "fraction_scalar_velocity_transverse_or_wallward": float("nan"),
                "max_distance_scalar_front_toward_junction": float("nan"),
            }
        )
        return out

    y = c[:, 1]
    inlet_y = float(np.min(y))
    outlet_y = float(np.max(y))
    junction_y = inlet_y + 0.08 * max(outlet_y - inlet_y, 1e-12)
    junction_pt = np.asarray(
        [0.0, junction_y, float(np.mean(c[:, 2]))], dtype=np.float64
    )
    classes = _velocity_direction_classes(c, v, junction_point=junction_pt)
    to_j = junction_pt[None, :] - c
    dist = np.linalg.norm(to_j[:, :2], axis=1)
    toward_proj = np.zeros_like(dist)
    vnorm = np.maximum(np.linalg.norm(v[:, :2], axis=1), 1e-12)
    dnorm = np.maximum(np.linalg.norm(to_j[:, :2], axis=1), 1e-12)
    toward_proj = np.sum(v[:, :2] * (to_j[:, :2] / dnorm[:, None]), axis=1)
    toward_dist = np.maximum(toward_proj, 0.0) * dist / np.maximum(np.max(vnorm), 1e-12)

    out.update(
        {
            "mean_scalar_velocity_x": float(np.mean(v[mask, 0])),
            "mean_scalar_velocity_y": float(np.mean(v[mask, 1])),
            "mean_scalar_velocity_z": float(np.mean(v[mask, 2])),
            "fraction_scalar_velocity_toward_junction": float(
                np.mean(classes["toward_junction"][mask])
            ),
            "fraction_scalar_velocity_transverse_or_wallward": float(
                np.mean(
                    (classes["mostly_transverse"] | classes["away_from_junction"])[mask]
                )
            ),
            "max_distance_scalar_front_toward_junction": float(
                np.max(toward_dist[mask])
            ),
        }
    )
    return out


def _compute_mass_diagnostics(
    *,
    domain_mass_initial: float,
    domain_mass_current: float,
    cumulative_scalar_mass_in: float,
    cumulative_scalar_mass_out: float,
) -> Dict[str, float | str | bool]:
    expected_domain_mass = float(
        domain_mass_initial + cumulative_scalar_mass_in - cumulative_scalar_mass_out
    )
    residual = float(domain_mass_current - expected_domain_mass)
    throughput_ref = max(
        abs(cumulative_scalar_mass_in),
        abs(cumulative_scalar_mass_out),
        1e-30,
    )
    domain_ref = max(abs(domain_mass_current), abs(domain_mass_initial), 1e-30)
    return {
        "mass_balance_formula_used": "domain_mass(t)=domain_mass_initial+cumulative_scalar_mass_in-cumulative_scalar_mass_out",
        "domain_mass_initial": float(domain_mass_initial),
        "domain_mass_current": float(domain_mass_current),
        "cumulative_scalar_mass_in": float(cumulative_scalar_mass_in),
        "cumulative_scalar_mass_out": float(cumulative_scalar_mass_out),
        "expected_domain_mass_current": float(expected_domain_mass),
        "domain_mass_balance_residual": float(residual),
        "domain_mass_balance_relative_to_throughput": float(
            abs(residual) / throughput_ref
        ),
        "domain_mass_balance_relative_to_domain_mass": float(
            abs(residual) / domain_ref
        ),
    }


def _compute_outlet_mixing_metrics(
    concentration: np.ndarray,
) -> Dict[str, Any]:
    c = np.asarray(concentration, dtype=np.float64)
    if c.size == 0:
        return {
            "outlet_cell_count": 0,
            "C_mean_outlet": float("nan"),
            "C_std_outlet": float("nan"),
            "C_min_outlet": float("nan"),
            "C_max_outlet": float("nan"),
            "std_reference": float("nan"),
            "normalized_unmixedness": float("nan"),
            "mixing_index_simple": float("nan"),
            "formula": {
                "std_reference": "sqrt(C_mean_outlet * (1 - C_mean_outlet))",
                "normalized_unmixedness": "C_std_outlet / std_reference",
                "mixing_index_simple": "1 - normalized_unmixedness",
            },
        }
    c_mean = float(np.mean(c))
    c_std = float(np.std(c))
    c_min = float(np.min(c))
    c_max = float(np.max(c))
    std_ref_sq = max(c_mean * (1.0 - c_mean), 0.0)
    std_reference = float(np.sqrt(std_ref_sq))
    if std_reference <= 1e-15:
        normalized = 0.0 if c_std <= 1e-15 else float("inf")
        mixing = 1.0 if c_std <= 1e-15 else 0.0
        note = "std_reference near zero; used guarded branch"
    else:
        normalized = float(c_std / std_reference)
        mixing = float(1.0 - normalized)
        note = "regular formula"
    return {
        "outlet_cell_count": int(c.size),
        "C_mean_outlet": float(c_mean),
        "C_std_outlet": float(c_std),
        "C_min_outlet": float(c_min),
        "C_max_outlet": float(c_max),
        "std_reference": float(std_reference),
        "normalized_unmixedness": float(normalized),
        "mixing_index_simple": float(mixing),
        "formula": {
            "std_reference": "sqrt(C_mean_outlet * (1 - C_mean_outlet))",
            "normalized_unmixedness": "C_std_outlet / std_reference",
            "mixing_index_simple": "1 - normalized_unmixedness",
        },
        "formula_note": str(note),
    }


def _write_outlet_profile_artifacts(
    *,
    run_dir: Path,
    centers: np.ndarray,
    concentration: np.ndarray,
    outlet_cells: np.ndarray,
) -> Dict[str, Any]:
    out_dir = run_dir / "outlet_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    outlet = np.asarray(outlet_cells, dtype=np.int64)
    c = np.asarray(concentration, dtype=np.float64)
    if outlet.size:
        c_out = c[outlet]
        xyz = np.asarray(centers[outlet], dtype=np.float64)
    else:
        c_out = np.asarray([], dtype=np.float64)
        xyz = np.zeros((0, 3), dtype=np.float64)

    csv_path = out_dir / "outlet_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("cell_index,x,y,z,concentration\n")
        for idx, coord, cv in zip(outlet.tolist(), xyz.tolist(), c_out.tolist()):
            f.write(
                f"{int(idx)},{coord[0]:.16e},{coord[1]:.16e},{coord[2]:.16e},{float(cv):.16e}\n"
            )

    profile_json = {
        "outlet_cell_count": int(outlet.size),
        "profile_columns": ["cell_index", "x", "y", "z", "concentration"],
        "csv_path": str(csv_path),
    }
    profile_json_path = out_dir / "outlet_profile.json"
    _write_json(profile_json_path, profile_json)

    mixing = _compute_outlet_mixing_metrics(c_out)
    mixing_json_path = out_dir / "outlet_mixing_metrics.json"
    _write_json(mixing_json_path, mixing)

    fig_prof, ax_prof = plt.subplots(figsize=(7.2, 5.6))
    if outlet.size:
        sc = ax_prof.scatter(
            xyz[:, 0],
            xyz[:, 2],
            c=c_out,
            cmap="viridis",
            s=22.0,
            alpha=0.9,
            linewidths=0,
            vmin=0.0,
            vmax=1.0,
        )
        fig_prof.colorbar(sc, ax=ax_prof, label="C")
    ax_prof.set_title("Outlet Concentration Profile (X-Z)")
    ax_prof.set_xlabel("x [m]")
    ax_prof.set_ylabel("z [m]")
    ax_prof.set_aspect("equal", adjustable="box")
    fig_prof.tight_layout()
    profile_png = out_dir / "outlet_concentration_profile.png"
    fig_prof.savefig(profile_png, dpi=180)
    plt.close(fig_prof)

    fig_hist, ax_hist = plt.subplots(figsize=(7.2, 5.6))
    if outlet.size:
        ax_hist.hist(c_out, bins=40, color="#1f77b4", alpha=0.85)
    ax_hist.set_title("Outlet Concentration Histogram")
    ax_hist.set_xlabel("C")
    ax_hist.set_ylabel("count")
    fig_hist.tight_layout()
    hist_png = out_dir / "outlet_concentration_histogram.png"
    fig_hist.savefig(hist_png, dpi=180)
    plt.close(fig_hist)

    return {
        "mixing_metrics": mixing,
        "artifacts": {
            "outlet_profile_csv": str(csv_path),
            "outlet_profile_json": str(profile_json_path),
            "outlet_mixing_metrics_json": str(mixing_json_path),
            "outlet_concentration_profile_png": str(profile_png),
            "outlet_concentration_histogram_png": str(hist_png),
        },
    }


def _run_transport_with_checkpointing(
    *,
    mesh,
    base_config,
    run_transport_fn,
    face_normal_velocity: np.ndarray,
    flux_diag: dict[str, Any],
    snapshot_steps: tuple[int, ...],
    checkpoint_every: int,
    max_walltime_seconds: float,
    run_dir: Path,
    resume_state: dict[str, Any] | None,
    run_horizon_mode: str = "steps",
    transport_end_time: float | None = None,
    final_outer_dt: float | None = None,
    diffusion_boundary_kwargs: dict[str, object] | None = None,
) -> dict[str, Any]:
    total_steps = int(base_config.steps)
    completed_steps = int(resume_state.get("step", 0)) if resume_state else 0
    if completed_steps >= total_steps:
        raise ValueError(
            f"resume checkpoint step={completed_steps} is already >= requested steps={total_steps}"
        )
    scalar_state = (
        np.asarray(resume_state.get("concentration"), dtype=np.float64)
        if resume_state is not None
        else None
    )
    cumulative_in = (
        float(resume_state.get("cumulative_mass_in", 0.0))
        if resume_state is not None
        else 0.0
    )
    cumulative_out = (
        float(resume_state.get("cumulative_mass_out", 0.0))
        if resume_state is not None
        else 0.0
    )
    domain_mass_initial = (
        float(resume_state.get("metadata", {}).get("domain_mass_initial", float("nan")))
        if resume_state is not None
        else float("nan")
    )
    physical_time = (
        float(resume_state.get("physical_time", 0.0))
        if resume_state is not None
        else 0.0
    )
    if not np.isfinite(domain_mass_initial):
        if scalar_state is None:
            domain_mass_initial = 0.0
        else:
            domain_mass_initial = float(
                np.sum(
                    np.asarray(scalar_state, dtype=np.float64)
                    * np.asarray(mesh.cell_volumes, dtype=np.float64)
                )
            )

    history_all: list[dict[str, Any]] = []
    snapshots_all: dict[int, np.ndarray] = {}
    checkpoint_paths: list[str] = []
    last_result: dict[str, Any] | None = None
    start_clock = time.perf_counter()
    stopped_due_walltime = False
    final_partial_step = int(total_steps) if final_outer_dt is not None else None

    while completed_steps < total_steps:
        elapsed = float(time.perf_counter() - start_clock)
        if max_walltime_seconds > 0.0 and elapsed >= max_walltime_seconds:
            stopped_due_walltime = True
            break
        remaining = total_steps - completed_steps
        if final_partial_step is not None and completed_steps < final_partial_step:
            # Preserve the final, shortened outer interval for a separate call.
            remaining = min(remaining, final_partial_step - completed_steps - 1)
            if remaining <= 0:
                remaining = 1
        chunk_steps = remaining
        if checkpoint_every > 0:
            to_next_checkpoint = checkpoint_every - (completed_steps % checkpoint_every)
            if to_next_checkpoint <= 0:
                to_next_checkpoint = checkpoint_every
            chunk_steps = min(chunk_steps, to_next_checkpoint)
        elif max_walltime_seconds > 0.0:
            chunk_steps = min(chunk_steps, 50)
        chunk_steps = max(1, int(chunk_steps))

        chunk_snapshot_steps = tuple(
            s
            for s in snapshot_steps
            if completed_steps < int(s) <= completed_steps + chunk_steps
        )
        is_final_partial_chunk = bool(
            final_partial_step is not None
            and completed_steps + chunk_steps == final_partial_step
            and completed_steps == final_partial_step - 1
        )
        chunk_cfg = replace(
            base_config,
            steps=int(chunk_steps),
            snapshot_steps=chunk_snapshot_steps,
            physical_time_start=float(physical_time),
            progress_total_steps=int(total_steps),
            progress_target_end_time=(
                float(transport_end_time)
                if run_horizon_mode == "end_time" and transport_end_time is not None
                else None
            ),
        )
        if is_final_partial_chunk:
            chunk_cfg = replace(
                chunk_cfg,
                dt_mode="manual",
                dt=float(final_outer_dt),
            )
        result = run_transport_fn(
            mesh,
            chunk_cfg,
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diag,
            initial_scalar=scalar_state,
            start_step=int(completed_steps),
            initial_cumulative_mass_in=float(cumulative_in),
            initial_cumulative_mass_out=float(cumulative_out),
            diffusion_boundary_kwargs=diffusion_boundary_kwargs,
        )
        last_result = dict(result)
        hist_chunk = [dict(h) for h in result.get("history", [])]
        history_all.extend(hist_chunk)
        for k, v in dict(result.get("snapshots", {})).items():
            snapshots_all[int(k)] = np.asarray(v, dtype=np.float64)
        if hist_chunk:
            completed_steps = int(hist_chunk[-1].get("step", completed_steps))
        else:
            completed_steps += int(chunk_steps)
        scalar_state = np.asarray(result.get("scalar"), dtype=np.float64)
        masses = dict(result.get("masses", {}))
        if not np.isfinite(domain_mass_initial):
            domain_mass_initial = float(masses.get("scalar_mass_initial", 0.0))
        cumulative_in = float(masses.get("cumulative_mass_in", cumulative_in))
        cumulative_out = float(masses.get("cumulative_mass_out", cumulative_out))
        used_dt = float(dict(result.get("dt_control", {})).get("used_dt", 0.0))
        if hist_chunk and "physical_time" in hist_chunk[-1]:
            physical_time = float(hist_chunk[-1]["physical_time"])
        else:
            physical_time += float(int(chunk_steps) * used_dt)

        should_checkpoint = bool(
            checkpoint_every > 0
            and (
                completed_steps % checkpoint_every == 0
                or completed_steps >= total_steps
            )
        )
        if should_checkpoint and scalar_state is not None:
            state_path = _save_transport_checkpoint(
                run_dir=run_dir,
                step=int(completed_steps),
                concentration=scalar_state,
                cumulative_mass_in=float(cumulative_in),
                cumulative_mass_out=float(cumulative_out),
                physical_time=float(physical_time),
                used_dt=float(used_dt),
                history_tail=hist_chunk[-1] if hist_chunk else {},
                extra_metadata={
                    "total_steps_requested": int(total_steps),
                    "transport_mode": str(base_config.transport_mode),
                    "transport_scheme": str(base_config.transport_scheme),
                    "domain_mass_initial": float(domain_mass_initial),
                },
            )
            checkpoint_paths.append(str(state_path))

    if last_result is None:
        raise RuntimeError("No transport steps were executed.")
    out = dict(last_result)
    out["history"] = history_all
    out["snapshots"] = snapshots_all
    out["run_control"] = {
        "total_steps_requested": int(total_steps),
        "completed_steps": int(completed_steps),
        "stopped_due_walltime": bool(stopped_due_walltime),
        "max_walltime_seconds": float(max_walltime_seconds),
        "checkpoint_every": int(checkpoint_every),
        "checkpoint_paths": checkpoint_paths,
        "resume_used": bool(resume_state is not None),
        "resume_from_step": int(resume_state.get("step", 0)) if resume_state else 0,
        "run_horizon_mode": str(run_horizon_mode),
        "requested_transport_end_time": transport_end_time,
        "final_outer_dt": final_outer_dt,
        "physical_time_final": float(physical_time),
        "domain_mass_initial": float(domain_mass_initial),
        "mass_balance_formula_used": "domain_mass(t)=domain_mass_initial+cumulative_scalar_mass_in-cumulative_scalar_mass_out",
    }
    return out


def main() -> None:
    from microfluidics.gmsh.gmsh_mesh_validation import (
        format_validation_report,
        validate_imported_tetra_mesh,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
    from microfluidics.gmsh.tetra.gmsh_tetra_mesh_loader import (
        load_imported_tetra_mesh_npz,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_operators import (
        build_face_normal_flux_from_velocity,
        run_operator_diagnostics,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_regime_guardrail import (
        DEFAULT_MAX_GRID_PECLET,
        DEFAULT_MAX_SCHMIDT,
        build_scalar_regime_audit,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_transport_solver import (
        GmshTetraTransportConfig,
        resolve_transport_dt_control,
        run_tetra_transport_debug,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (
        build_prescribed_velocity_field,
        compute_velocity_field_diagnostics,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
        resolve_inlet_face_groups,
    )
    from microfluidics.preprocessor import (
        apply_flow_profile_to_mesh,
        case_config_from_mapping,
        compile_flow_runtime_profile,
        compile_scalar_runtime_profile,
        compile_uniform_material_properties,
        load_case_config,
        resolve_case_mesh_path,
        scalar_profile_to_diffusion_kwargs,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-npz",
        type=str,
        default="",
        help="Explicit path to *_imported_mesh.npz.",
    )
    parser.add_argument(
        "--case-config",
        type=str,
        default="",
        help=(
            "Optional case_config_v1 JSON. When omitted, an embedded case from "
            "the imported mesh is used automatically."
        ),
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="",
        help="Deprecated compatibility flag. Transport debug runs now require --mesh-npz.",
    )
    parser.add_argument(
        "--import-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_IMPORT_RUNS_ROOT_REL)),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            resolve_repo_path(PROJECT_ROOT, GMSH_TETRA_TRANSPORT_RUNS_ROOT_REL)
        ),
    )
    add_pipeline_manifest_arguments(parser)
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument(
        "--transport-execution-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
        help="Execution backend for transport stepping.",
    )
    parser.add_argument(
        "--torch-device",
        type=str,
        default="",
        help="Optional torch device for transport execution, for example cpu or cuda:0.",
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
        default="two_inlets_to_outlet_tj_balanced",
    )
    parser.add_argument(
        "--velocity-source",
        type=str,
        choices=("prescribed", "flow_run"),
        default="",
        help=(
            "Velocity/flux source for transport. Must be chosen explicitly: "
            "prescribed or flow_run."
        ),
    )
    parser.add_argument(
        "--flow-run-dir",
        type=str,
        default="",
        help="Path to flow run directory that contains flow_coupling_metadata.json and final_corrected_face_flux.npy.",
    )
    parser.add_argument(
        "--allow-unready-flow",
        action="store_true",
        help=(
            "Unsafe debug override: allow transport to consume upstream flow even "
            "when source readiness is false."
        ),
    )
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
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Outer transport intervals to advance (default: 200 unless --transport-end-time is set).",
    )
    parser.add_argument(
        "--transport-end-time",
        type=float,
        default=None,
        help="Advance to this physical time in seconds; mutually exclusive with --steps.",
    )
    parser.add_argument(
        "--snapshot-steps",
        type=str,
        default="",
        help="Comma-separated snapshot steps (e.g. 200,600,1200).",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Save snapshots every N steps (merged with --snapshot-steps).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save transport checkpoint every N steps. 0 disables checkpointing.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="",
        help="Path to checkpoint_state.json to resume transport run.",
    )
    parser.add_argument(
        "--max-walltime-seconds",
        type=float,
        default=0.0,
        help="Optional walltime budget; run stops cleanly once budget is reached.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=20,
        help="Progress print cadence for transport stepping.",
    )
    parser.add_argument("--dt", type=float, default=1.5e-5)
    parser.add_argument(
        "--dt-mode",
        type=str,
        choices=("manual", "auto"),
        default="auto",
    )
    parser.add_argument("--cfl-target", type=float, default=0.5)
    parser.add_argument(
        "--auto-dt-percentile",
        type=float,
        default=5.0,
        help="Deprecated compatibility option; auto dt now uses the strict stability limit.",
    )
    parser.add_argument(
        "--max-transport-substeps",
        type=int,
        default=None,
        help="Deprecated compatibility option; a configured cap is reported but never blocks safe substeps.",
    )
    parser.add_argument(
        "--transport-substep-warning-threshold",
        type=int,
        default=32,
        help="Deprecated compatibility option; every use of substeps is now reported.",
    )
    parser.add_argument("--diffusivity", type=float, default=3e-10)
    parser.add_argument(
        "--kinematic-viscosity",
        type=float,
        default=1e-6,
        help="Kinematic viscosity nu [m^2/s] used only for Sc support diagnostics.",
    )
    parser.add_argument(
        "--max-supported-grid-peclet",
        type=float,
        default=DEFAULT_MAX_GRID_PECLET,
        help=(
            "Warning threshold for Pe_grid=max(Q_out/(D*h)); a finite exceedance "
            "does not block readiness and is not a CFL stability limit."
        ),
    )
    parser.add_argument(
        "--max-supported-schmidt",
        type=float,
        default=DEFAULT_MAX_SCHMIDT,
        help="Warning threshold for Sc=nu/D; finite exceedance does not block readiness.",
    )
    parser.add_argument("--left-inlet-value", type=float, default=0.0)
    parser.add_argument("--right-inlet-value", type=float, default=1.0)
    parser.add_argument("--cfl-limit", type=float, default=0.8)
    parser.add_argument("--boundedness-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--breakthrough-turnover-threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum turnover count (inlet_flux * time / volume) that makes outlet "
            "breakthrough physically expected."
        ),
    )
    parser.add_argument(
        "--breakthrough-travel-fraction",
        type=float,
        default=0.35,
        help=(
            "Physical time fraction of estimated inlet->outlet travel time needed "
            "before breakthrough is expected."
        ),
    )
    parser.add_argument(
        "--breakthrough-outlet-frac-threshold",
        type=float,
        default=1e-4,
        help="Outlet scalar fraction threshold used to declare breakthrough detection.",
    )
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
        default="tpfa",
    )
    parser.add_argument("--inlet-speed", type=float, default=0.15)
    parser.add_argument(
        "--sweep-manual-vs-auto",
        action="store_true",
        help="Run two back-to-back cases: manual dt and auto dt.",
    )
    parser.add_argument(
        "--velocity-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in old-vs-balanced prescribed velocity comparison sweep.",
    )
    parser.add_argument(
        "--transport-scheme-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in upwind-vs-bounded_upwind comparison sweep.",
    )
    parser.add_argument(
        "--disable-post-step-clipping",
        action="store_true",
        help="Disable post-step clipping safety guard.",
    )
    parser.add_argument(
        "--safety-clamp-after-diffusion",
        action="store_true",
        help="Apply optional safety clamp to [0,1] right after diffusion sub-step.",
    )
    parser.add_argument(
        "--compare-cpu-gpu",
        action="store_true",
        help="Run CPU and GPU transport runs and save comparison JSON.",
    )
    parser.add_argument(
        "--fail-if-numpy-fallback",
        action="store_true",
        help="Fail run if requested torch execution falls back to numpy stepping.",
    )
    args = parser.parse_args()
    if args.steps is not None and args.transport_end_time is not None:
        parser.error("--steps and --transport-end-time are mutually exclusive.")
    if args.transport_end_time is not None and args.transport_end_time <= 0.0:
        parser.error("--transport-end-time must be positive.")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive.")
    if args.steps is None and args.transport_end_time is None:
        args.steps = 200
    external_case_path = (
        _normalize_user_path(args.case_config).resolve()
        if str(args.case_config).strip()
        else None
    )
    import_root = _normalize_user_path(args.import_root).resolve()
    output_root = _normalize_user_path(args.output_root).resolve()
    flow_run_dir = (
        _normalize_user_path(args.flow_run_dir).resolve()
        if str(args.flow_run_dir).strip()
        else None
    )
    resume_from_checkpoint = (
        _normalize_user_path(args.resume_from_checkpoint).resolve()
        if str(args.resume_from_checkpoint).strip()
        else None
    )
    mesh_npz_input = (
        _normalize_user_path(args.mesh_npz).resolve()
        if str(args.mesh_npz).strip()
        else None
    )
    snapshot_steps_set = {
        int(s.strip()) for s in args.snapshot_steps.split(",") if s.strip()
    }
    snapshot_steps = tuple(sorted(s for s in snapshot_steps_set if int(s) > 0))

    if mesh_npz_input is None:
        parser.error(
            "--mesh-npz is required. Automatic import-root/mesh-name resolution "
            "is no longer supported."
        )
    if not str(args.velocity_source).strip():
        parser.error(
            "--velocity-source is required. Choose --velocity-source prescribed or "
            "--velocity-source flow_run."
        )
    mesh_npz = mesh_npz_input
    original_mesh_input = str(args.mesh_npz)
    if not mesh_npz.exists():
        raise FileNotFoundError(f"Mesh npz not found: {mesh_npz}")
    mesh_stem = mesh_npz.name.replace("_imported_mesh.npz", "")

    if bool(args.compare_cpu_gpu):
        compare_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compare_root = output_root / f"{compare_stamp}_{mesh_stem}_cpu_gpu"
        compare_root.mkdir(parents=True, exist_ok=True)

        def _run_variant(variant: str) -> Path:
            variant_root = compare_root / variant
            variant_root.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mesh-npz",
                str(mesh_npz),
                "--import-root",
                str(import_root),
                "--output-root",
                str(variant_root),
                "--velocity-field",
                str(args.velocity_field),
                "--velocity-source",
                str(args.velocity_source),
                "--transport-mode",
                str(args.transport_mode),
                "--transport-scheme",
                str(args.transport_scheme),
                "--dt-mode",
                str(args.dt_mode),
                "--cfl-target",
                str(args.cfl_target),
                "--cfl-limit",
                str(args.cfl_limit),
                "--boundedness-tolerance",
                str(args.boundedness_tolerance),
                "--snapshot-steps",
                ",".join(str(s) for s in snapshot_steps),
                "--snapshot-every",
                str(args.snapshot_every),
                "--dt",
                str(args.dt),
                "--diffusivity",
                str(args.diffusivity),
                "--gradient-method",
                str(args.gradient_method),
                "--laplacian-method",
                str(args.laplacian_method),
                "--inlet-speed",
                str(args.inlet_speed),
                "--no-velocity-comparison",
                "--no-transport-scheme-comparison",
            ]
            if args.transport_end_time is None:
                cmd.extend(["--steps", str(args.steps)])
            else:
                cmd.extend(["--transport-end-time", str(args.transport_end_time)])
            if flow_run_dir is not None:
                cmd.extend(["--flow-run-dir", str(flow_run_dir)])
            if external_case_path is not None:
                cmd.extend(["--case-config", str(external_case_path)])
            if bool(args.allow_unready_flow):
                cmd.append("--allow-unready-flow")
            if bool(args.disable_post_step_clipping):
                cmd.append("--disable-post-step-clipping")
            if bool(args.safety_clamp_after_diffusion):
                cmd.append("--safety-clamp-after-diffusion")
            if variant == "cpu":
                cmd.extend(
                    ["--backend", "numpy", "--transport-execution-backend", "numpy"]
                )
            else:
                cmd.extend(
                    ["--backend", "auto", "--transport-execution-backend", "torch"]
                )
                if str(args.torch_device).strip():
                    cmd.extend(["--torch-device", str(args.torch_device).strip()])
                if bool(args.fail_if_numpy_fallback):
                    cmd.append("--fail-if-numpy-fallback")
            started = time.perf_counter()
            run = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = float(time.perf_counter() - started)
            run_dir = _parse_child_run_directory(
                run.stdout,
                prefix="[gmsh-tetra-transport] run directory:",
                variant=variant,
            )
            marker = run_dir / "comparison_runtime_seconds.txt"
            marker.write_text(f"{elapsed:.9f}", encoding="utf-8")
            return run_dir

        cpu_run_dir = _run_variant("cpu")
        gpu_run_dir = _run_variant("gpu")

        cpu_summary = json.loads(
            (cpu_run_dir / "summary.json").read_text(encoding="utf-8")
        )
        gpu_summary = json.loads(
            (gpu_run_dir / "summary.json").read_text(encoding="utf-8")
        )
        cpu_time = float(
            (cpu_run_dir / "comparison_runtime_seconds.txt").read_text(encoding="utf-8")
        )
        gpu_time = float(
            (gpu_run_dir / "comparison_runtime_seconds.txt").read_text(encoding="utf-8")
        )
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

        cpu_final = np.load(cpu_run_dir / "final_concentration.npy")
        gpu_final = np.load(gpu_run_dir / "final_concentration.npy")
        diff = np.abs(cpu_final - gpu_final)
        max_abs_diff_final = float(np.max(diff))
        mean_abs_diff_final = float(np.mean(diff))
        l2_diff_final = float(np.sqrt(np.mean((cpu_final - gpu_final) ** 2)))

        snap_diffs: Dict[str, float] = {}
        for step in snapshot_steps:
            c_path = (
                cpu_run_dir / "snapshots" / f"step_{step:04d}" / "concentration.npy"
            )
            g_path = (
                gpu_run_dir / "snapshots" / f"step_{step:04d}" / "concentration.npy"
            )
            if c_path.exists() and g_path.exists():
                cd = np.load(c_path)
                gd = np.load(g_path)
                snap_diffs[str(step)] = float(np.max(np.abs(cd - gd)))

        cpu_backend_exec = cpu_summary.get("transport", {}).get("backend_execution", {})
        gpu_backend_exec = gpu_summary.get("transport", {}).get("backend_execution", {})
        gpu_valid = (
            str(gpu_backend_exec.get("stepping_backend", "")) == "torch"
            and str(gpu_backend_exec.get("device", "")).startswith("cuda")
            and not bool(gpu_backend_exec.get("used_numpy_fallback", True))
            and bool(gpu_backend_exec.get("all_core_arrays_on_cuda", False))
        )
        tol_pass = max_abs_diff_final <= 1e-8 and mean_abs_diff_final <= 1e-10
        conclusion = "pass" if (tol_pass and gpu_valid) else "fail"
        cpu_mass_final = _extract_scalar_mass_final(cpu_summary)
        gpu_mass_final = _extract_scalar_mass_final(gpu_summary)
        mass_diff = cpu_mass_final - gpu_mass_final

        comparison = {
            "mesh_stem": mesh_stem,
            "original_input": original_mesh_input,
            "resolved_mesh_npz": str(mesh_npz),
            "cpu_run_dir": str(cpu_run_dir),
            "gpu_run_dir": str(gpu_run_dir),
            "cpu_runtime_seconds": cpu_time,
            "gpu_runtime_seconds": gpu_time,
            "speedup_factor": speedup,
            "cpu_backend_execution": cpu_backend_exec,
            "gpu_backend_execution": gpu_backend_exec,
            "max_abs_diff_final_C": max_abs_diff_final,
            "mean_abs_diff_final_C": mean_abs_diff_final,
            "l2_diff_final_C": l2_diff_final,
            "max_abs_diff_snapshot_C": snap_diffs,
            "mass_diagnostics": {
                "cpu_total_mass_proxy_final": cpu_mass_final,
                "gpu_total_mass_proxy_final": gpu_mass_final,
                "abs_mass_diff": float(abs(mass_diff)),
                "relative_mass_diff": _safe_rel_diff(mass_diff, cpu_mass_final),
            },
            "boundedness": {
                "cpu": cpu_summary.get("transport", {}).get("clipping", {}),
                "gpu": gpu_summary.get("transport", {}).get("clipping", {}),
            },
            "cfl_comparison": {
                "cpu_cfl_max": float(
                    cpu_summary.get("transport", {}).get("cfl_max", 0.0)
                ),
                "gpu_cfl_max": float(
                    gpu_summary.get("transport", {}).get("cfl_max", 0.0)
                ),
            },
            "outlet_arrival_comparison": {
                "cpu": cpu_summary.get("transport", {}).get("outlet_arrival", {}),
                "gpu": gpu_summary.get("transport", {}).get("outlet_arrival", {}),
            },
            "conclusion": conclusion,
            "tolerance_pass": tol_pass,
            "gpu_backend_pass": gpu_valid,
        }
        comparison_path = compare_root / "cpu_gpu_transport_comparison.json"
        _write_json(comparison_path, comparison)
        print(f"[gmsh-tetra-transport] comparison root: {compare_root}")
        print(f"[gmsh-tetra-transport] comparison json: {comparison_path}")
        return

    run_dir = create_timestamped_run_dir(output_root, mesh_npz.stem)
    manifest_recorder = build_pipeline_manifest_recorder(
        args,
        stage_type="transport",
        run_dir=run_dir,
    )
    manifest_inputs = {
        "original_input": original_mesh_input,
        "resolved_mesh_npz": str(mesh_npz),
        "import_root": str(import_root),
        "output_root": str(output_root),
        "velocity_source": str(args.velocity_source),
        "flow_run_dir": str(args.flow_run_dir),
        "kinematic_viscosity": float(args.kinematic_viscosity),
        "max_supported_grid_peclet": float(args.max_supported_grid_peclet),
        "max_supported_schmidt": float(args.max_supported_schmidt),
        "case_config": str(external_case_path) if external_case_path else "",
    }
    manifest_artifacts = {
        "run_log": str(run_dir / "run.log"),
        "config_json": str(run_dir / "config.json"),
        "summary_json": str(run_dir / "summary.json"),
        "acceptance_report_json": str(run_dir / "acceptance_report.json"),
        "stage_status_json": str(run_dir / "stage_status.json"),
        "transport_regime_audit_json": str(run_dir / "transport_regime_audit.json"),
        "flow_coupling_metadata_snapshot_json": str(
            run_dir / "flow_coupling_metadata_snapshot.json"
        ),
        "validation_report_txt": str(run_dir / "validation_report.txt"),
        "snapshots_summary_json": str(run_dir / "snapshots_summary.json"),
        "final_concentration_npy": str(run_dir / "final_concentration.npy"),
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
        metadata={"manifest_role": "transport_stage"},
    )
    run_started = time.perf_counter()
    with _tee_logging(run_dir / "run.log"):
        backend = select_backend(args.backend)
        execution_backend_request = str(args.transport_execution_backend)
        requested_torch_device = str(args.torch_device).strip()
        if execution_backend_request == "auto":
            execution_backend = backend.selected_backend
            execution_device = (
                requested_torch_device
                if requested_torch_device and execution_backend == "torch"
                else backend.device
            )
        elif execution_backend_request == "numpy":
            execution_backend = "numpy"
            execution_device = "cpu"
        else:
            if backend.torch_available:
                execution_backend = "torch"
                execution_device = requested_torch_device or (
                    "cuda:0" if backend.torch_cuda_available else "cpu"
                )
            else:
                execution_backend = "numpy"
                execution_device = "cpu"
        print(f"[gmsh-tetra-transport] run directory: {run_dir}")
        print(f"[gmsh-tetra-transport] original input: {original_mesh_input}")
        print(f"[gmsh-tetra-transport] mesh npz: {mesh_npz}")
        print(f"[gmsh-tetra-transport] mesh stem: {mesh_stem}")
        print(
            "[gmsh-tetra-transport] backend: "
            f"requested={backend.requested_backend}, selected={backend.selected_backend}, "
            f"device={backend.device}"
        )
        print(
            "[gmsh-tetra-transport] transport execution backend: "
            f"requested={execution_backend_request}, selected={execution_backend}, "
            f"device={execution_device}"
        )
        print(
            "[gmsh-tetra-transport] torch: "
            f"available={backend.torch_available}, version={backend.torch_version}, "
            f"cuda_available={backend.torch_cuda_available}, "
            f"device_count={backend.torch_device_count}, gpu={backend.torch_gpu_name}"
        )
        for note in backend.notes:
            print(f"[gmsh-tetra-transport] note: {note}")

        config_payload = {
            "original_input": original_mesh_input,
            "resolved_mesh_npz": str(mesh_npz),
            "mesh_stem": mesh_stem,
            "mesh_npz": str(mesh_npz),
            "import_root": str(import_root),
            "output_root": str(output_root),
            "backend": backend.__dict__,
            "transport_execution_backend_requested": execution_backend_request,
            "transport_execution_backend_selected": execution_backend,
            "transport_execution_device": execution_device,
            "transport_torch_device_requested": requested_torch_device,
            "velocity_source": str(args.velocity_source),
            "velocity_field": str(args.velocity_field),
            "flow_run_dir": str(flow_run_dir) if flow_run_dir is not None else "",
            "allow_unready_flow": bool(args.allow_unready_flow),
            "transport_mode": str(args.transport_mode),
            "transport_scheme": str(args.transport_scheme),
            "steps": int(args.steps),
            "snapshot_steps": list(snapshot_steps),
            "snapshot_every": int(args.snapshot_every),
            "dt": float(args.dt),
            "dt_mode": str(args.dt_mode),
            "cfl_target": float(args.cfl_target),
            "auto_dt_percentile": float(args.auto_dt_percentile),
            "max_transport_substeps": (
                None
                if args.max_transport_substeps is None
                else int(args.max_transport_substeps)
            ),
            "transport_substep_warning_threshold": int(
                args.transport_substep_warning_threshold
            ),
            "diffusivity": float(args.diffusivity),
            "kinematic_viscosity": float(args.kinematic_viscosity),
            "max_supported_grid_peclet": float(args.max_supported_grid_peclet),
            "max_supported_schmidt": float(args.max_supported_schmidt),
            "left_inlet_value": float(args.left_inlet_value),
            "right_inlet_value": float(args.right_inlet_value),
            "cfl_limit": float(args.cfl_limit),
            "boundedness_tolerance": float(args.boundedness_tolerance),
            "breakthrough_turnover_threshold": float(
                args.breakthrough_turnover_threshold
            ),
            "breakthrough_travel_fraction": float(args.breakthrough_travel_fraction),
            "breakthrough_outlet_frac_threshold": float(
                args.breakthrough_outlet_frac_threshold
            ),
            "gradient_method": str(args.gradient_method),
            "laplacian_method": str(args.laplacian_method),
            "inlet_speed": float(args.inlet_speed),
            "sweep_manual_vs_auto": bool(args.sweep_manual_vs_auto),
            "velocity_comparison": bool(args.velocity_comparison),
            "transport_scheme_comparison": bool(args.transport_scheme_comparison),
            "no_velocity_comparison": not bool(args.velocity_comparison),
            "no_transport_scheme_comparison": not bool(
                args.transport_scheme_comparison
            ),
            "disable_post_step_clipping": bool(args.disable_post_step_clipping),
            "safety_clamp_after_diffusion": bool(args.safety_clamp_after_diffusion),
            "fail_if_numpy_fallback": bool(args.fail_if_numpy_fallback),
            "checkpoint_every": int(args.checkpoint_every),
            "resume_from_checkpoint": (
                str(resume_from_checkpoint)
                if resume_from_checkpoint is not None
                else ""
            ),
            "max_walltime_seconds": float(args.max_walltime_seconds),
            "progress_every": int(args.progress_every),
        }
        _write_json(run_dir / "config.json", config_payload)

        mesh = load_imported_tetra_mesh_npz(mesh_npz)
        runtime_case = None
        runtime_case_source = ""
        if external_case_path is not None:
            runtime_case = load_case_config(external_case_path)
            configured_mesh_path = resolve_case_mesh_path(
                runtime_case, external_case_path
            )
            if configured_mesh_path != mesh.source_path.resolve():
                raise ValueError(
                    f"case mesh resolves to {configured_mesh_path}, but imported "
                    f"mesh was built from {mesh.source_path.resolve()}"
                )
            runtime_case_source = str(external_case_path)
        elif mesh.case_config:
            runtime_case = case_config_from_mapping(mesh.case_config)
            runtime_case_source = "embedded:mesh_npz"
        transport_diffusion_kwargs = None
        if runtime_case is not None:
            flow_profile = compile_flow_runtime_profile(mesh, runtime_case)
            apply_flow_profile_to_mesh(mesh, flow_profile)
            scalar_profile = compile_scalar_runtime_profile(
                mesh, runtime_case, "concentration"
            )
            if scalar_profile.periodic_pairs:
                raise ValueError(
                    "periodic scalar diffusion is supported by the diffusion backend, "
                    "but periodic advection is not supported by the transport runtime"
                )
            transport_diffusion_kwargs = scalar_profile_to_diffusion_kwargs(
                scalar_profile
            )
            material = compile_uniform_material_properties(runtime_case)
            args.diffusivity = material.get(
                "scalar_diffusivity_m2_per_s", args.diffusivity
            )
            args.kinematic_viscosity = material.get(
                "kinematic_viscosity_m2_per_s", args.kinematic_viscosity
            )
            inlet_groups_for_case = resolve_inlet_face_groups(mesh)
            for side, argument_name in (
                ("left_faces", "left_inlet_value"),
                ("right_faces", "right_inlet_value"),
            ):
                faces = np.asarray(inlet_groups_for_case[side], dtype=np.int64)
                values = [
                    scalar_profile.dirichlet_by_face.get(int(face)) for face in faces
                ]
                if not values or any(value is None for value in values):
                    raise ValueError(
                        f"case concentration BC must define Dirichlet values on all {side}"
                    )
                first = float(values[0])
                if not all(np.isclose(first, float(value)) for value in values[1:]):
                    raise ValueError(
                        f"current transport advection runtime requires uniform {side} concentration"
                    )
                setattr(args, argument_name, first)
            config_payload["left_inlet_value"] = float(args.left_inlet_value)
            config_payload["right_inlet_value"] = float(args.right_inlet_value)
            config_payload["diffusivity"] = float(args.diffusivity)
            config_payload["kinematic_viscosity"] = float(args.kinematic_viscosity)
            config_payload["case_config"] = runtime_case_source
            manifest_inputs["case_config"] = runtime_case_source
            _write_json(run_dir / "config.json", config_payload)
        validation = validate_imported_tetra_mesh(mesh)
        (run_dir / "validation_report.txt").write_text(
            format_validation_report(validation),
            encoding="utf-8",
        )
        if not validation.is_valid:
            raise RuntimeError("Loaded imported mesh failed validation.")
        inlet_groups = resolve_inlet_face_groups(mesh)
        left_inlet_faces = np.asarray(
            inlet_groups.get("left_inlet_faces", []), dtype=np.int64
        )
        right_inlet_faces = np.asarray(
            inlet_groups.get("right_inlet_faces", []), dtype=np.int64
        )
        resume_state: dict[str, Any] | None = None
        if resume_from_checkpoint is not None:
            resume_state = _load_transport_checkpoint(resume_from_checkpoint)
            print(
                "[gmsh-tetra-transport] resume checkpoint: "
                f"{resume_state.get('state_path')} (step={resume_state.get('step')})"
            )

        run_variants = [args.dt_mode]
        if args.sweep_manual_vs_auto:
            run_variants = ["manual", "auto"]
        scheme_variants = [str(args.transport_scheme)]
        if (
            str(args.transport_scheme) == "bounded_upwind"
            and str(args.transport_mode) == "advection"
            and bool(args.transport_scheme_comparison)
        ):
            scheme_variants = ["upwind", "bounded_upwind"]

        velocity_source = str(args.velocity_source)
        source_flow_payload: Dict[str, Any] = {}
        source_flow_compatibility: Dict[str, Any] = {"compatible": False}
        readiness_info: Dict[str, Any] = {}
        source_flow_ready_for_transport = False
        coupling_flux_loaded_successfully = False

        velocity_fields_to_run = [str(args.velocity_field)]
        if velocity_source == "prescribed":
            if str(args.velocity_field) == "two_inlets_to_outlet_tj_balanced" and bool(
                args.velocity_comparison
            ):
                velocity_fields_to_run = [
                    "two_inlets_to_outlet_tj",
                    "two_inlets_to_outlet_tj_balanced",
                    "two_inlets_to_outlet_tj_balanced_symmetric_profile",
                ]
            if str(args.velocity_field) in {
                "two_inlets_to_outlet_tj_piecewise_centerline",
                "two_inlets_to_outlet_tj_axis_aligned_sanity",
            } and bool(args.velocity_comparison):
                velocity_fields_to_run = [
                    "two_inlets_to_outlet_tj_piecewise_centerline",
                    "two_inlets_to_outlet_tj_axis_aligned_sanity",
                ]
        else:
            if flow_run_dir is None:
                raise ValueError(
                    "--flow-run-dir is required when --velocity-source=flow_run."
                )
            source_flow_payload = _load_flow_coupling_payload(flow_run_dir)
            source_flow_compatibility = _check_flow_transport_mesh_compatibility(
                mesh=mesh,
                flow_payload=source_flow_payload,
            )
            if not bool(source_flow_compatibility.get("compatible", False)):
                raise RuntimeError(
                    "Flow run mesh is incompatible with current transport mesh: "
                    f"{source_flow_compatibility}"
                )
            readiness_info = _validate_source_flow_readiness(
                dict(source_flow_payload.get("metadata", {})),
                strict=not bool(args.allow_unready_flow),
            )
            source_flow_ready_for_transport = bool(readiness_info.get("ready", False))
            if bool(readiness_info.get("warning", False)):
                print(
                    "[gmsh-tetra-transport] WARNING: source flow stage-status is not "
                    "ready for transport coupling."
                )
            velocity_fields_to_run = ["flow_run_face_flux"]

        velocity_runs: Dict[str, Dict[str, object]] = {}
        for velocity_name in velocity_fields_to_run:
            if velocity_source == "flow_run":
                face_flux = np.asarray(
                    source_flow_payload["face_flux"], dtype=np.float64
                )
                if int(face_flux.shape[0]) != int(mesh.face_vertices.shape[0]):
                    raise RuntimeError(
                        "Flow run face flux length does not match current mesh face count."
                    )
                face_normal_velocity = face_flux / np.maximum(
                    np.asarray(mesh.face_areas, dtype=np.float64), 1e-30
                )
                coupling_flux_loaded_successfully = True
                face_groups_loaded = source_flow_payload.get("face_groups")
                inlet_faces_for_flow = np.asarray([], dtype=np.int64)
                outlet_faces_for_flow = np.asarray(mesh.outlet_faces, dtype=np.int64)
                wall_faces_for_flow = np.asarray(mesh.wall_faces, dtype=np.int64)
                left_faces_for_flow = left_inlet_faces
                right_faces_for_flow = right_inlet_faces
                if isinstance(face_groups_loaded, np.ndarray):
                    fg = np.asarray(face_groups_loaded, dtype=np.int32)
                    if fg.shape[0] == mesh.face_vertices.shape[0]:
                        inlet_faces_for_flow = np.flatnonzero(fg == 3).astype(np.int64)
                        outlet_faces_for_flow = np.flatnonzero(fg == 2).astype(np.int64)
                        wall_faces_for_flow = np.flatnonzero(fg == 1).astype(np.int64)
                        left_faces_for_flow, right_faces_for_flow = (
                            _split_inlet_faces_by_x(
                                mesh,
                                inlet_faces_for_flow,
                            )
                        )
                flow_cell_velocity = source_flow_payload.get("cell_velocity")
                if isinstance(flow_cell_velocity, np.ndarray) and tuple(
                    flow_cell_velocity.shape
                ) == tuple(np.asarray(mesh.cell_centers).shape):
                    cell_velocity = np.asarray(flow_cell_velocity, dtype=np.float64)
                else:
                    cell_velocity = np.zeros_like(
                        np.asarray(mesh.cell_centers, dtype=np.float64)
                    )
                velocity_field = SimpleNamespace(
                    name="flow_run_face_flux",
                    cell_velocity=cell_velocity,
                    boundary_face_velocity_overrides={},
                    boundary_groups={
                        "left_inlet_faces": left_faces_for_flow,
                        "right_inlet_faces": right_faces_for_flow,
                        "outlet_faces": outlet_faces_for_flow,
                        "wall_faces": wall_faces_for_flow,
                    },
                    metadata={
                        "velocity_source": "flow_run",
                        "source_flow_run_dir": str(
                            source_flow_payload.get("flow_run_dir", "")
                        ),
                        "source_flow_ready_for_transport": bool(
                            source_flow_ready_for_transport
                        ),
                    },
                )
                flux_diag = _compute_face_flux_diagnostics_from_face_normal_velocity(
                    mesh,
                    face_normal_velocity,
                    left_inlet_faces=left_faces_for_flow,
                    right_inlet_faces=right_faces_for_flow,
                )
                velocity_diagnostics = {
                    "boundary_flux_imbalance_ratio": float(
                        flux_diag.get("boundary_flux_imbalance_ratio", 0.0)
                    ),
                    "boundary_flux_imbalance_warning": bool(
                        float(flux_diag.get("boundary_flux_imbalance_ratio", 0.0))
                        > 5e-2
                    ),
                    "run_source": "flow_run",
                }
            else:
                velocity_field = build_prescribed_velocity_field(
                    mesh,
                    field_name=velocity_name,  # type: ignore[arg-type]
                    inlet_speed=float(args.inlet_speed),
                )
                face_normal_velocity, flux_diag = build_face_normal_flux_from_velocity(
                    mesh,
                    velocity_field.cell_velocity,
                    boundary_face_velocity_overrides=velocity_field.boundary_face_velocity_overrides,
                    left_inlet_faces=velocity_field.boundary_groups["left_inlet_faces"],
                    right_inlet_faces=velocity_field.boundary_groups[
                        "right_inlet_faces"
                    ],
                    outlet_faces=velocity_field.boundary_groups["outlet_faces"],
                    wall_faces=velocity_field.boundary_groups["wall_faces"],
                )
                velocity_diagnostics = compute_velocity_field_diagnostics(
                    mesh,
                    velocity_field,
                    flux_diagnostics=flux_diag,
                )
            cell_velocity = np.asarray(velocity_field.cell_velocity, dtype=np.float64)

            scheme_runs: Dict[str, Dict[str, object]] = {}
            selected_dt_variant = run_variants[-1]
            for scheme_name in scheme_variants:
                dt_runs: Dict[str, Dict[str, object]] = {}
                for dt_mode in run_variants:
                    transport_cfg = GmshTetraTransportConfig(
                        dt=float(args.dt),
                        dt_mode=dt_mode,  # type: ignore[arg-type]
                        cfl_target=float(args.cfl_target),
                        auto_dt_percentile=float(args.auto_dt_percentile),
                        max_transport_substeps=(
                            None
                            if args.max_transport_substeps is None
                            else int(args.max_transport_substeps)
                        ),
                        transport_substep_warning_threshold=int(
                            args.transport_substep_warning_threshold
                        ),
                        steps=int(args.steps) if args.steps is not None else 1,
                        transport_mode=args.transport_mode,  # type: ignore[arg-type]
                        transport_scheme=scheme_name,  # type: ignore[arg-type]
                        diffusivity=float(args.diffusivity),
                        left_inlet_value=float(args.left_inlet_value),
                        right_inlet_value=float(args.right_inlet_value),
                        cfl_limit=float(args.cfl_limit),
                        gradient_method=args.gradient_method,  # type: ignore[arg-type]
                        laplacian_method=args.laplacian_method,  # type: ignore[arg-type]
                        backend=execution_backend,  # type: ignore[arg-type]
                        torch_device=execution_device,
                        clipping_enabled=not bool(args.disable_post_step_clipping),
                        safety_clamp_after_diffusion=bool(
                            args.safety_clamp_after_diffusion
                        ),
                        boundedness_tolerance=float(args.boundedness_tolerance),
                        progress_every=int(args.progress_every),
                        snapshot_steps=snapshot_steps,
                    )
                    transport_time_estimate = _estimate_transport_run_time(
                        mesh=mesh,
                        cell_velocity=np.asarray(
                            velocity_field.cell_velocity, dtype=np.float64
                        ),
                        face_normal_velocity=np.asarray(
                            face_normal_velocity, dtype=np.float64
                        ),
                        flux_diag=flux_diag,
                        transport_config=transport_cfg,
                        resolve_transport_dt_control_fn=resolve_transport_dt_control,
                        resume_state=resume_state,
                        transport_end_time=args.transport_end_time,
                        breakthrough_turnover_threshold=float(
                            args.breakthrough_turnover_threshold
                        ),
                        breakthrough_travel_fraction=float(
                            args.breakthrough_travel_fraction
                        ),
                        max_walltime_seconds=float(
                            max(float(args.max_walltime_seconds), 0.0)
                        ),
                    )
                    planned_total_steps = int(
                        transport_time_estimate["planned_total_steps"]
                    )
                    planned_snapshot_steps_set = set(snapshot_steps_set)
                    if int(args.snapshot_every) > 0:
                        planned_snapshot_steps_set.update(
                            range(
                                int(args.snapshot_every),
                                planned_total_steps + 1,
                                int(args.snapshot_every),
                            )
                        )
                    planned_snapshot_steps = tuple(
                        sorted(s for s in planned_snapshot_steps_set if int(s) > 0)
                    )
                    transport_cfg = replace(
                        transport_cfg,
                        steps=planned_total_steps,
                        snapshot_steps=planned_snapshot_steps,
                    )
                    transport_time_estimate["requested_total_steps"] = (
                        planned_total_steps
                    )
                    _enforce_transport_dt_controller(
                        time_estimate=transport_time_estimate,
                    )
                    planned_dt_control = dict(
                        transport_time_estimate.get("dt_control", {})
                    )
                    if bool(planned_dt_control.get("transport_substep_warning", False)):
                        print(
                            "[gmsh-tetra-transport] WARNING: requested outer dt will "
                            "be split into stable substeps; "
                            f"requested_dt={float(planned_dt_control.get('used_dt', 0.0)):.6e}, "
                            f"dt_substep={float(planned_dt_control.get('dt_substep', 0.0)):.6e}, "
                            f"substeps={int(planned_dt_control.get('required_transport_substep_count', 0))}."
                        )
                    transport_result = _run_transport_with_checkpointing(
                        mesh=mesh,
                        base_config=transport_cfg,
                        run_transport_fn=run_tetra_transport_debug,
                        face_normal_velocity=face_normal_velocity,
                        flux_diag=flux_diag,
                        snapshot_steps=planned_snapshot_steps,
                        checkpoint_every=int(args.checkpoint_every),
                        max_walltime_seconds=float(
                            max(float(args.max_walltime_seconds), 0.0)
                        ),
                        run_dir=run_dir,
                        resume_state=resume_state,
                        run_horizon_mode=str(
                            transport_time_estimate["run_horizon_mode"]
                        ),
                        transport_end_time=args.transport_end_time,
                        final_outer_dt=transport_time_estimate["final_outer_dt"],
                        diffusion_boundary_kwargs=transport_diffusion_kwargs,
                    )
                    # Resume checkpoint must only be consumed once.
                    resume_state = None
                    dt_runs[dt_mode] = {
                        "config": transport_cfg,
                        "transport_time_estimate": transport_time_estimate,
                        "result": transport_result,
                    }
                    dt_ctrl = transport_result["dt_control"]
                    print(
                        "[gmsh-tetra-transport] "
                        f"velocity={velocity_name}, scheme={scheme_name}, dt-variant {dt_mode}: "
                        f"requested={dt_ctrl['requested_dt']:.6e}, "
                        f"used={dt_ctrl['used_dt']:.6e}, "
                        f"dt_substep={dt_ctrl['dt_substep']:.6e}, "
                        f"substeps={int(dt_ctrl['transport_substep_count'])}, "
                        f"stable_est={dt_ctrl['stable_dt_estimate']:.6e}, "
                        f"cfl_max={transport_result['cfl_max']:.6f}"
                    )
                scheme_runs[scheme_name] = {"dt_runs": dt_runs}

            velocity_runs[velocity_name] = {
                "velocity_field": velocity_field,
                "face_normal_velocity": face_normal_velocity,
                "flux_diagnostics": flux_diag,
                "velocity_diagnostics": velocity_diagnostics,
                "scheme_runs": scheme_runs,
                "selected_dt_variant": selected_dt_variant,
            }

        selected_velocity_name = (
            str(args.velocity_field)
            if velocity_source == "prescribed"
            else "flow_run_face_flux"
        )
        if selected_velocity_name not in velocity_runs:
            selected_velocity_name = velocity_fields_to_run[-1]

        selected_velocity = velocity_runs[selected_velocity_name]
        velocity_field = selected_velocity["velocity_field"]  # type: ignore[assignment]
        face_normal_velocity = selected_velocity["face_normal_velocity"]  # type: ignore[assignment]
        flux_diag = selected_velocity["flux_diagnostics"]  # type: ignore[assignment]
        velocity_diagnostics = selected_velocity["velocity_diagnostics"]  # type: ignore[assignment]
        selected_dt_variant = selected_velocity["selected_dt_variant"]  # type: ignore[assignment]
        selected_scheme_name = str(args.transport_scheme)
        if selected_scheme_name not in selected_velocity["scheme_runs"]:  # type: ignore[operator]
            selected_scheme_name = scheme_variants[-1]
        transport_cfg = selected_velocity["scheme_runs"][selected_scheme_name][
            "dt_runs"
        ][selected_dt_variant]["config"]  # type: ignore[index]
        transport_result = selected_velocity["scheme_runs"][selected_scheme_name][
            "dt_runs"
        ][selected_dt_variant]["result"]  # type: ignore[index]
        transport_time_estimate = selected_velocity["scheme_runs"][
            selected_scheme_name
        ]["dt_runs"][selected_dt_variant]["transport_time_estimate"]  # type: ignore[index]
        backend_exec_selected = transport_result.get("backend_execution", {})
        if bool(args.fail_if_numpy_fallback):
            if bool(backend_exec_selected.get("used_numpy_fallback", True)):
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: transport stepping used numpy fallback."
                )
            if str(backend_exec_selected.get("stepping_backend", "")) != "torch":
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: stepping_backend is not 'torch'."
                )
            if not str(backend_exec_selected.get("device", "")).startswith("cuda"):
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: stepping device is not CUDA."
                )
            if not bool(backend_exec_selected.get("all_core_arrays_on_cuda", False)):
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: core arrays are not reported on CUDA."
                )

        regime_audit = build_scalar_regime_audit(
            mesh,
            np.asarray(face_normal_velocity, dtype=np.float64),
            diffusivity=float(args.diffusivity),
            kinematic_viscosity=float(args.kinematic_viscosity),
            scalar_kind="mass",
            max_grid_peclet=float(args.max_supported_grid_peclet),
            max_schmidt=float(args.max_supported_schmidt),
        )
        regime_supported_accuracy = bool(regime_audit.get("supported_accuracy", False))
        regime_blocking_error = bool(regime_audit.get("blocking_error", True))
        regime_warning_codes = list(regime_audit.get("warning_codes", []))
        regime_error_codes = list(regime_audit.get("error_codes", []))
        regime_warning = bool(regime_warning_codes)

        velocity = np.asarray(velocity_field.cell_velocity, dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        velocity_previews = _save_velocity_vector_previews(
            centers=mesh.cell_centers,
            velocity=velocity,
            run_dir=run_dir,
        )
        right_branch = velocity_diagnostics.get("right_inlet_branch_direction", {})
        left_branch = velocity_diagnostics.get("left_inlet_branch_direction", {})
        outlet_branch = velocity_diagnostics.get("outlet_branch_direction", {})
        criteria = {
            "mostly_horizontal_fraction_min": 0.95,
            "transverse_bad_fraction_max": 0.05,
            "mean_abs_uy_over_abs_ux_max": 0.03,
            "mostly_vertical_fraction_min": 0.95,
            "mean_abs_ux_over_abs_uy_max": 0.03,
        }

        def _criteria_pass(branch: Dict[str, Any]) -> Dict[str, bool]:
            mh = float(branch.get("mostly_horizontal_fraction", float("nan")))
            tb = float(branch.get("transverse_bad_fraction", float("nan")))
            ryx = float(branch.get("mean_abs_uy_over_abs_ux", float("nan")))
            return {
                "mostly_horizontal_fraction_pass": bool(
                    mh > criteria["mostly_horizontal_fraction_min"]
                ),
                "transverse_bad_fraction_pass": bool(
                    tb < criteria["transverse_bad_fraction_max"]
                ),
                "mean_abs_uy_over_abs_ux_pass": bool(
                    ryx < criteria["mean_abs_uy_over_abs_ux_max"]
                ),
            }

        def _outlet_criteria_pass(branch: Dict[str, Any]) -> Dict[str, bool]:
            mv = float(branch.get("mostly_vertical_fraction", float("nan")))
            rxy = float(branch.get("mean_abs_ux_over_abs_uy", float("nan")))
            return {
                "mostly_vertical_fraction_pass": bool(
                    mv > criteria["mostly_vertical_fraction_min"]
                ),
                "mean_abs_ux_over_abs_uy_pass": bool(
                    rxy < criteria["mean_abs_ux_over_abs_uy_max"]
                ),
            }

        velocity_field_audit = {
            "velocity_field_name": str(velocity_field.name),
            "cell_velocity_magnitude_stats": {
                "min": float(np.min(speed)),
                "max": float(np.max(speed)),
                "mean": float(np.mean(speed)),
            },
            "zones": velocity_diagnostics.get("zones", {}),
            "criteria": criteria,
            "right_inlet_branch": {
                **right_branch,
                "criteria_eval": _criteria_pass(right_branch),  # type: ignore[arg-type]
            },
            "left_inlet_branch": {
                **left_branch,
                "criteria_eval": _criteria_pass(left_branch),  # type: ignore[arg-type]
            },
            "outlet_branch": {
                **outlet_branch,
                "criteria_eval": _outlet_criteria_pass(outlet_branch),  # type: ignore[arg-type]
            },
            "artifacts": velocity_previews,
        }
        _write_json(run_dir / "velocity_field_audit.json", velocity_field_audit)

        inlet_symmetry_audit = _run_inlet_symmetry_audit(
            mesh=mesh,
            velocity_field=velocity_field,
            face_normal_velocity=face_normal_velocity,
            run_dir=run_dir,
            left_scalar=float(transport_cfg.left_inlet_value),
            right_scalar=float(transport_cfg.right_inlet_value),
        )
        _write_json(run_dir / "inlet_symmetry_audit.json", inlet_symmetry_audit)

        flux_diag_payload = {
            **flux_diag,
            "velocity_field_name": str(velocity_field.name),
            "balanced_velocity_enabled": bool(
                velocity_field.metadata.get("balanced_velocity_enabled", False)
            ),
            "inlet_flux_before_balance": velocity_field.metadata.get(
                "inlet_flux_before_balance"
            ),
            "outlet_flux_before_balance": velocity_field.metadata.get(
                "outlet_flux_before_balance"
            ),
            "outlet_scale_factor": velocity_field.metadata.get("outlet_scale_factor"),
            "inlet_flux_after_balance": velocity_field.metadata.get(
                "inlet_flux_after_balance"
            ),
            "outlet_flux_after_balance": velocity_field.metadata.get(
                "outlet_flux_after_balance"
            ),
            "net_boundary_flux_after_balance": velocity_field.metadata.get(
                "net_boundary_flux_after_balance"
            ),
            "boundary_flux_imbalance_ratio_after_balance": velocity_field.metadata.get(
                "boundary_flux_imbalance_ratio_after_balance"
            ),
            "wall_flux_max_abs": flux_diag.get("wall_flux_max_abs"),
        }
        _write_json(run_dir / "flux_diagnostics.json", flux_diag_payload)

        comparison_payload: Dict[str, object] = {}
        if (
            "two_inlets_to_outlet_tj" in velocity_runs
            and "two_inlets_to_outlet_tj_balanced" in velocity_runs
        ):
            scheme_for_velocity_compare = "bounded_upwind"
            if (
                scheme_for_velocity_compare
                not in velocity_runs["two_inlets_to_outlet_tj"]["scheme_runs"]  # type: ignore[index]
            ):
                scheme_for_velocity_compare = scheme_variants[-1]
            old_result = velocity_runs["two_inlets_to_outlet_tj"]["scheme_runs"][
                scheme_for_velocity_compare
            ]["dt_runs"][selected_dt_variant]["result"]
            bal_result = velocity_runs["two_inlets_to_outlet_tj_balanced"][
                "scheme_runs"
            ][scheme_for_velocity_compare]["dt_runs"][selected_dt_variant]["result"]
            old_flux = velocity_runs["two_inlets_to_outlet_tj"]["flux_diagnostics"]
            bal_flux = velocity_runs["two_inlets_to_outlet_tj_balanced"][
                "flux_diagnostics"
            ]
            comparison_payload = {
                "selected_dt_variant": selected_dt_variant,
                "selected_scheme": scheme_for_velocity_compare,
                "old": {
                    "velocity_field": "two_inlets_to_outlet_tj",
                    "boundary_flux_imbalance_ratio": velocity_runs[
                        "two_inlets_to_outlet_tj"
                    ]["velocity_diagnostics"]["boundary_flux_imbalance_ratio"],
                    "net_boundary_flux": old_flux["net_boundary_flux"],
                    "clipped_cell_count_total": old_result["clipping"][
                        "final_clipped_cell_count"
                    ],
                    "overshoot_cell_count_total": old_result["clipping"][
                        "final_overshoot_cell_count"
                    ],
                    "undershoot_cell_count_total": old_result["clipping"][
                        "final_undershoot_cell_count"
                    ],
                    "overshoot_max": old_result["clipping"][
                        "max_overshoot_before_clip"
                    ],
                    "undershoot_max": old_result["clipping"][
                        "max_undershoot_before_clip"
                    ],
                },
                "balanced": {
                    "velocity_field": "two_inlets_to_outlet_tj_balanced",
                    "boundary_flux_imbalance_ratio": velocity_runs[
                        "two_inlets_to_outlet_tj_balanced"
                    ]["velocity_diagnostics"]["boundary_flux_imbalance_ratio"],
                    "net_boundary_flux": bal_flux["net_boundary_flux"],
                    "clipped_cell_count_total": bal_result["clipping"][
                        "final_clipped_cell_count"
                    ],
                    "overshoot_cell_count_total": bal_result["clipping"][
                        "final_overshoot_cell_count"
                    ],
                    "undershoot_cell_count_total": bal_result["clipping"][
                        "final_undershoot_cell_count"
                    ],
                    "overshoot_max": bal_result["clipping"][
                        "max_overshoot_before_clip"
                    ],
                    "undershoot_max": bal_result["clipping"][
                        "max_undershoot_before_clip"
                    ],
                },
            }
            old_clip = int(comparison_payload["old"]["clipped_cell_count_total"])  # type: ignore[index]
            bal_clip = int(comparison_payload["balanced"]["clipped_cell_count_total"])  # type: ignore[index]
            old_over = int(comparison_payload["old"]["overshoot_cell_count_total"])  # type: ignore[index]
            bal_over = int(comparison_payload["balanced"]["overshoot_cell_count_total"])  # type: ignore[index]
            if bal_clip < old_clip and bal_over < old_over:
                comparison_payload["conclusion"] = (
                    "balanced prescribed velocity improved clipping/overshoot vs old field"
                )
            else:
                comparison_payload["conclusion"] = (
                    "transport issue is likely scalar boundary update / upwind implementation, "
                    "not velocity mass imbalance"
                )
            _write_json(
                run_dir / "velocity_balance_comparison.json", comparison_payload
            )

        symmetry_profile_comparison: Dict[str, object] = {}
        if (
            "two_inlets_to_outlet_tj" in velocity_runs
            and "two_inlets_to_outlet_tj_balanced_symmetric_profile" in velocity_runs
        ):
            scheme_for_profile_compare = "bounded_upwind"
            if (
                scheme_for_profile_compare
                not in velocity_runs["two_inlets_to_outlet_tj"]["scheme_runs"]  # type: ignore[index]
            ):
                scheme_for_profile_compare = scheme_variants[-1]

            old_result = velocity_runs["two_inlets_to_outlet_tj"]["scheme_runs"][
                scheme_for_profile_compare
            ]["dt_runs"][selected_dt_variant]["result"]
            sym_result = velocity_runs[
                "two_inlets_to_outlet_tj_balanced_symmetric_profile"
            ]["scheme_runs"][scheme_for_profile_compare]["dt_runs"][
                selected_dt_variant
            ]["result"]

            def _snap_metrics(
                result_obj: Dict[str, object], step: int
            ) -> Dict[str, float]:
                snaps = {
                    int(k): np.asarray(v, dtype=np.float64)
                    for k, v in result_obj.get("snapshots", {}).items()  # type: ignore[union-attr]
                }
                conc = snaps.get(step)
                if conc is None:
                    return {}
                front = _snapshot_front_diagnostics(mesh.cell_centers, conc)
                return {
                    "concentration_centroid_y": float(
                        front["concentration_centroid_y"]
                    ),
                    "max_x_extent_where_C_gt_1e-4": float(
                        front["max_x_extent_where_C_gt_1e-4"]
                    ),
                }

            old_m50 = _snap_metrics(old_result, 50)
            old_m200 = _snap_metrics(old_result, 200)
            sym_m50 = _snap_metrics(sym_result, 50)
            sym_m200 = _snap_metrics(sym_result, 200)

            old_field = velocity_runs["two_inlets_to_outlet_tj"]["velocity_field"]
            old_vn = velocity_runs["two_inlets_to_outlet_tj"]["face_normal_velocity"]
            sym_field = velocity_runs[
                "two_inlets_to_outlet_tj_balanced_symmetric_profile"
            ]["velocity_field"]
            sym_vn = velocity_runs[
                "two_inlets_to_outlet_tj_balanced_symmetric_profile"
            ]["face_normal_velocity"]
            old_inlet = _run_inlet_symmetry_audit(
                mesh=mesh,
                velocity_field=old_field,
                face_normal_velocity=old_vn,
                run_dir=None,
                left_scalar=float(transport_cfg.left_inlet_value),
                right_scalar=float(transport_cfg.right_inlet_value),
                save_artifacts=False,
            )
            sym_inlet = _run_inlet_symmetry_audit(
                mesh=mesh,
                velocity_field=sym_field,
                face_normal_velocity=sym_vn,
                run_dir=None,
                left_scalar=float(transport_cfg.left_inlet_value),
                right_scalar=float(transport_cfg.right_inlet_value),
                save_artifacts=False,
            )
            symmetry_profile_comparison = {
                "selected_dt_variant": selected_dt_variant,
                "selected_scheme": scheme_for_profile_compare,
                "inlet_metrics_old": {
                    "left_inlet": _compact_inlet_group_metrics(old_inlet["left_inlet"]),
                    "right_inlet": _compact_inlet_group_metrics(
                        old_inlet["right_inlet"]
                    ),
                    "symmetry_warning_any": bool(old_inlet["symmetry_warning_any"]),
                },
                "inlet_metrics_symmetric": {
                    "left_inlet": _compact_inlet_group_metrics(sym_inlet["left_inlet"]),
                    "right_inlet": _compact_inlet_group_metrics(
                        sym_inlet["right_inlet"]
                    ),
                    "symmetry_warning_any": bool(sym_inlet["symmetry_warning_any"]),
                },
                "step_50": {"old": old_m50, "symmetric": sym_m50},
                "step_200": {"old": old_m200, "symmetric": sym_m200},
            }
            _write_json(
                run_dir / "inlet_symmetry_profile_comparison.json",
                symmetry_profile_comparison,
            )

        velocity_field_comparison: Dict[str, object] = {}
        if (
            "two_inlets_to_outlet_tj_piecewise_centerline" in velocity_runs
            and "two_inlets_to_outlet_tj_axis_aligned_sanity" in velocity_runs
        ):
            scheme_for_cmp = "bounded_upwind"
            if (
                scheme_for_cmp
                not in velocity_runs["two_inlets_to_outlet_tj_piecewise_centerline"][
                    "scheme_runs"
                ]  # type: ignore[index]
            ):
                scheme_for_cmp = scheme_variants[-1]

            old_result = velocity_runs["two_inlets_to_outlet_tj_piecewise_centerline"][
                "scheme_runs"
            ][scheme_for_cmp]["dt_runs"][selected_dt_variant]["result"]
            new_result = velocity_runs["two_inlets_to_outlet_tj_axis_aligned_sanity"][
                "scheme_runs"
            ][scheme_for_cmp]["dt_runs"][selected_dt_variant]["result"]
            old_flux = velocity_runs["two_inlets_to_outlet_tj_piecewise_centerline"][
                "flux_diagnostics"
            ]
            new_flux = velocity_runs["two_inlets_to_outlet_tj_axis_aligned_sanity"][
                "flux_diagnostics"
            ]
            old_clip = old_result["clipping"]
            new_clip = new_result["clipping"]
            old_diag = velocity_runs["two_inlets_to_outlet_tj_piecewise_centerline"][
                "velocity_diagnostics"
            ]
            new_diag = velocity_runs["two_inlets_to_outlet_tj_axis_aligned_sanity"][
                "velocity_diagnostics"
            ]

            steps_cmp = [50, 200, 400]
            old_snaps = {
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in old_result.get("snapshots", {}).items()  # type: ignore[union-attr]
            }
            new_snaps = {
                int(k): np.asarray(v, dtype=np.float64)
                for k, v in new_result.get("snapshots", {}).items()  # type: ignore[union-attr]
            }
            old_hist = {
                int(h["step"]): h
                for h in old_result["history"]  # type: ignore[index]
            }
            new_hist = {
                int(h["step"]): h
                for h in new_result["history"]  # type: ignore[index]
            }
            step_metrics: Dict[str, object] = {}
            for step_cmp in steps_cmp:
                old_c = old_snaps.get(step_cmp)
                new_c = new_snaps.get(step_cmp)
                if old_c is None or new_c is None:
                    continue
                old_front = _snapshot_front_diagnostics(mesh.cell_centers, old_c)
                new_front = _snapshot_front_diagnostics(mesh.cell_centers, new_c)
                old_h = old_hist.get(step_cmp, {})
                new_h = new_hist.get(step_cmp, {})
                step_metrics[str(step_cmp)] = {
                    "old_piecewise_centerline": {
                        "max_x_extent_where_C_gt_1e-4": float(
                            old_front["max_x_extent_where_C_gt_1e-4"]
                        ),
                        "max_y_where_C_gt_1e-4": float(
                            old_front["max_y_where_C_gt_1e-4"]
                        ),
                        "concentration_centroid_x": float(
                            old_front["concentration_centroid_x"]
                        ),
                        "concentration_centroid_y": float(
                            old_front["concentration_centroid_y"]
                        ),
                        "concentration_centroid_z": float(
                            old_front["concentration_centroid_z"]
                        ),
                        "outlet_frac_gt_1e-6": _read_outlet_fraction(old_h, 1e-6),
                        "outlet_frac_gt_1e-4": _read_outlet_fraction(old_h, 1e-4),
                        "outlet_frac_gt_1e-3": _read_outlet_fraction(old_h, 1e-3),
                        "outlet_frac_gt_1e-2": _read_outlet_fraction(old_h, 1e-2),
                    },
                    "new_axis_aligned_sanity": {
                        "max_x_extent_where_C_gt_1e-4": float(
                            new_front["max_x_extent_where_C_gt_1e-4"]
                        ),
                        "max_y_where_C_gt_1e-4": float(
                            new_front["max_y_where_C_gt_1e-4"]
                        ),
                        "concentration_centroid_x": float(
                            new_front["concentration_centroid_x"]
                        ),
                        "concentration_centroid_y": float(
                            new_front["concentration_centroid_y"]
                        ),
                        "concentration_centroid_z": float(
                            new_front["concentration_centroid_z"]
                        ),
                        "outlet_frac_gt_1e-6": _read_outlet_fraction(new_h, 1e-6),
                        "outlet_frac_gt_1e-4": _read_outlet_fraction(new_h, 1e-4),
                        "outlet_frac_gt_1e-3": _read_outlet_fraction(new_h, 1e-3),
                        "outlet_frac_gt_1e-2": _read_outlet_fraction(new_h, 1e-2),
                    },
                }

            improved = False
            if "400" in step_metrics:
                old_front = (
                    step_metrics["400"]  # type: ignore[union-attr]
                    .get("old_piecewise_centerline", {})
                    .get("max_y_where_C_gt_1e-4")
                )
                new_front = (
                    step_metrics["400"]
                    .get("new_axis_aligned_sanity", {})
                    .get("max_y_where_C_gt_1e-4")  # type: ignore[union-attr]
                )
                if isinstance(old_front, (float, int)) and isinstance(
                    new_front, (float, int)
                ):
                    improved = float(new_front) > float(old_front)

            old_step400 = old_snaps.get(400)
            new_step400 = new_snaps.get(400)
            side_by_side_png = None
            if old_step400 is not None and new_step400 is not None:
                side_by_side_png = _save_two_field_snapshot_comparison_xy(
                    centers=mesh.cell_centers,
                    conc_old=old_step400,
                    conc_new=new_step400,
                    old_label="piecewise_centerline",
                    new_label="axis_aligned_sanity",
                    output_path=run_dir / "concentration_old_vs_new_step_0400.png",
                    step=400,
                )

            velocity_field_comparison = {
                "selected_dt_variant": selected_dt_variant,
                "selected_scheme": scheme_for_cmp,
                "old_piecewise_centerline": {
                    "inlet_flux_balance": float(old_flux["total_inlet_flux_in"]),
                    "outlet_flux_balance": float(old_flux["total_outlet_flux_out"]),
                    "net_boundary_flux": float(old_flux["net_boundary_flux"]),
                    "wall_flux_max_abs": float(old_flux["wall_flux_max_abs"]),
                    "overshoot_cell_count_total": int(
                        old_clip["final_overshoot_cell_count"]
                    ),
                    "undershoot_cell_count_total": int(
                        old_clip["final_undershoot_cell_count"]
                    ),
                    "clipped_cell_count_total": int(
                        old_clip["final_clipped_cell_count"]
                    ),
                    "mean_abs_uy_over_abs_ux_right": float(
                        old_diag["right_inlet_branch_direction"][
                            "mean_abs_uy_over_abs_ux"
                        ]
                    ),
                    "mean_abs_uy_over_abs_ux_left": float(
                        old_diag["left_inlet_branch_direction"][
                            "mean_abs_uy_over_abs_ux"
                        ]
                    ),
                    "mostly_horizontal_fraction_right": float(
                        old_diag["right_inlet_branch_direction"][
                            "mostly_horizontal_fraction"
                        ]
                    ),
                    "mostly_horizontal_fraction_left": float(
                        old_diag["left_inlet_branch_direction"][
                            "mostly_horizontal_fraction"
                        ]
                    ),
                    "transverse_bad_fraction_right": float(
                        old_diag["right_inlet_branch_direction"][
                            "transverse_bad_fraction"
                        ]
                    ),
                    "transverse_bad_fraction_left": float(
                        old_diag["left_inlet_branch_direction"][
                            "transverse_bad_fraction"
                        ]
                    ),
                },
                "new_axis_aligned_sanity": {
                    "inlet_flux_balance": float(new_flux["total_inlet_flux_in"]),
                    "outlet_flux_balance": float(new_flux["total_outlet_flux_out"]),
                    "net_boundary_flux": float(new_flux["net_boundary_flux"]),
                    "wall_flux_max_abs": float(new_flux["wall_flux_max_abs"]),
                    "overshoot_cell_count_total": int(
                        new_clip["final_overshoot_cell_count"]
                    ),
                    "undershoot_cell_count_total": int(
                        new_clip["final_undershoot_cell_count"]
                    ),
                    "clipped_cell_count_total": int(
                        new_clip["final_clipped_cell_count"]
                    ),
                    "mean_abs_uy_over_abs_ux_right": float(
                        new_diag["right_inlet_branch_direction"][
                            "mean_abs_uy_over_abs_ux"
                        ]
                    ),
                    "mean_abs_uy_over_abs_ux_left": float(
                        new_diag["left_inlet_branch_direction"][
                            "mean_abs_uy_over_abs_ux"
                        ]
                    ),
                    "mostly_horizontal_fraction_right": float(
                        new_diag["right_inlet_branch_direction"][
                            "mostly_horizontal_fraction"
                        ]
                    ),
                    "mostly_horizontal_fraction_left": float(
                        new_diag["left_inlet_branch_direction"][
                            "mostly_horizontal_fraction"
                        ]
                    ),
                    "transverse_bad_fraction_right": float(
                        new_diag["right_inlet_branch_direction"][
                            "transverse_bad_fraction"
                        ]
                    ),
                    "transverse_bad_fraction_left": float(
                        new_diag["left_inlet_branch_direction"][
                            "transverse_bad_fraction"
                        ]
                    ),
                },
                "step_metrics": step_metrics,
                "conclusion": (
                    "axis_aligned_sanity improves concentration movement "
                    "from inlet toward junction/outlet"
                    if improved
                    else "axis_aligned_sanity does not show clear transport movement improvement"
                ),
                "artifacts": {
                    "concentration_old_vs_new_step_0400_png": side_by_side_png,
                },
            }
            _write_json(
                run_dir / "velocity_field_comparison.json", velocity_field_comparison
            )

        scheme_comparison_payload: Dict[str, object] = {}
        selected_velocity_for_scheme = velocity_runs[selected_velocity_name]
        if (
            "upwind" in selected_velocity_for_scheme["scheme_runs"]
            and "bounded_upwind" in selected_velocity_for_scheme["scheme_runs"]
        ):  # type: ignore[index]
            up = selected_velocity_for_scheme["scheme_runs"]["upwind"]["dt_runs"][
                selected_dt_variant
            ]["result"]  # type: ignore[index]
            bd = selected_velocity_for_scheme["scheme_runs"]["bounded_upwind"][
                "dt_runs"
            ][selected_dt_variant]["result"]  # type: ignore[index]
            up_clip = up["clipping"]
            bd_clip = bd["clipping"]
            scheme_comparison_payload = {
                "velocity_field": selected_velocity_name,
                "selected_dt_variant": selected_dt_variant,
                "upwind": {
                    "overshoot_max": up_clip["max_overshoot_before_clip"],
                    "undershoot_max": up_clip["max_undershoot_before_clip"],
                    "raw_overshoot_before_limiter": up["raw_update_audit"][
                        "raw_overshoot_before_limiter"
                    ],
                    "raw_undershoot_before_limiter": up["raw_update_audit"][
                        "raw_undershoot_before_limiter"
                    ],
                    "max_raw_delta_c": up["raw_update_audit"]["max_raw_delta_c"],
                    "clipped_cell_count": up_clip["final_clipped_cell_count"],
                    "clipped_fraction": up_clip["final_clipped_fraction"],
                    "mass_conservation_error": up["limiter"][
                        "conservation_error_after_limiter"
                    ],
                    "pairwise_conservation_error_max_abs": up["conservation_audit"][
                        "interior_pairwise_conservation_error_max_abs"
                    ],
                    "max_local_outgoing_cfl": up["local_cfl_audit"][
                        "max_local_outgoing_cfl"
                    ],
                    "max_local_incoming_cfl": up["local_cfl_audit"][
                        "max_local_incoming_cfl"
                    ],
                    "outlet_arrival": up["outlet_arrival"],
                    "final_stats": up["final_stats"],
                    "right_inlet_corner_overshoot_count": up_clip[
                        "final_inlet_wall_corner_zone_overshoot_cell_count"
                    ],
                },
                "bounded_upwind": {
                    "overshoot_max": bd_clip["max_overshoot_before_clip"],
                    "undershoot_max": bd_clip["max_undershoot_before_clip"],
                    "raw_overshoot_before_limiter": bd["raw_update_audit"][
                        "raw_overshoot_before_limiter"
                    ],
                    "raw_undershoot_before_limiter": bd["raw_update_audit"][
                        "raw_undershoot_before_limiter"
                    ],
                    "max_raw_delta_c": bd["raw_update_audit"]["max_raw_delta_c"],
                    "clipped_cell_count": bd_clip["final_clipped_cell_count"],
                    "clipped_fraction": bd_clip["final_clipped_fraction"],
                    "mass_conservation_error": bd["limiter"][
                        "conservation_error_after_limiter"
                    ],
                    "pairwise_conservation_error_max_abs": bd["conservation_audit"][
                        "interior_pairwise_conservation_error_max_abs"
                    ],
                    "max_local_outgoing_cfl": bd["local_cfl_audit"][
                        "max_local_outgoing_cfl"
                    ],
                    "max_local_incoming_cfl": bd["local_cfl_audit"][
                        "max_local_incoming_cfl"
                    ],
                    "outlet_arrival": bd["outlet_arrival"],
                    "final_stats": bd["final_stats"],
                    "right_inlet_corner_overshoot_count": bd_clip[
                        "final_inlet_wall_corner_zone_overshoot_cell_count"
                    ],
                },
            }
            _write_json(
                run_dir / "transport_scheme_comparison.json", scheme_comparison_payload
            )

        concentration = np.asarray(transport_result["scalar"], dtype=np.float64)
        np.save(run_dir / "final_concentration.npy", concentration)
        velocity = np.asarray(velocity_field.cell_velocity, dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=1)
        _write_json(
            run_dir / "transport_history.json", {"history": transport_result["history"]}
        )

        snapshot_scalars: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.float64)
            for k, v in transport_result.get("snapshots", {}).items()
        }
        snapshot_history = {
            int(entry["step"]): entry
            for entry in transport_result["history"]
            if int(entry["step"]) in snapshot_scalars
        }
        snapshots_dir = run_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshots_summary: Dict[str, object] = {"snapshots": []}
        run_control_for_snap = dict(transport_result.get("run_control", {}))
        domain_mass_initial_snap = float(
            run_control_for_snap.get(
                "domain_mass_initial",
                transport_result["masses"].get("initial_mass_proxy", 0.0),
            )
        )
        outlet_cells_snapshot = np.asarray(
            transport_result["final_step_debug"].get(
                "outlet_cells", np.zeros((0,), dtype=np.int64)
            ),
            dtype=np.int64,
        )
        for step in sorted(snapshot_scalars):
            scalar_step = snapshot_scalars[step]
            hist = snapshot_history.get(step, {})
            step_dir = snapshots_dir / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            np.save(step_dir / "concentration.npy", scalar_step)
            png_path = Path(
                _save_concentration_xy_snapshot(
                    centers=mesh.cell_centers,
                    concentration=scalar_step,
                    output_path=step_dir / "concentration_xy.png",
                    title=f"Concentration XY at step {step}",
                )
            )
            vtu_step = _export_transport_vtu(
                points=mesh.points,
                tetrahedra=mesh.tetrahedra,
                concentration=scalar_step,
                velocity=velocity,
                output_path=step_dir / "result.vtu",
            )
            dt_hist = float(
                hist.get("dt_used", transport_result["dt_control"]["used_dt"])
            )
            physical_time = float(hist.get("physical_time", step * dt_hist))
            min_c = float(np.min(scalar_step))
            max_c = float(np.max(scalar_step))
            mean_c = float(np.mean(scalar_step))
            total_mass = float(np.sum(scalar_step * mesh.cell_volumes))
            cumulative_in = float(hist.get("cumulative_mass_in", 0.0))
            cumulative_out = float(hist.get("cumulative_mass_out", 0.0))
            mass_diag_snap = _compute_mass_diagnostics(
                domain_mass_initial=domain_mass_initial_snap,
                domain_mass_current=total_mass,
                cumulative_scalar_mass_in=cumulative_in,
                cumulative_scalar_mass_out=cumulative_out,
            )
            mass_balance_error = float(
                mass_diag_snap.get("domain_mass_balance_residual", 0.0)
            )
            outlet_mix_snap = _compute_outlet_mixing_metrics(
                scalar_step[outlet_cells_snapshot]
                if outlet_cells_snapshot.size
                else np.asarray([], dtype=np.float64)
            )
            numerically_clean_snapshot = bool(
                np.all(np.isfinite(scalar_step))
                and (not bool(hist.get("cfl_warning", False)))
                and float(hist.get("overshoot_before_clip", 0.0)) <= 1e-12
                and float(hist.get("undershoot_before_clip", 0.0)) <= 1e-12
                and int(hist.get("clipped_cell_count", 0)) == 0
            )
            snap_item: Dict[str, object] = {
                "step": int(step),
                "physical_time": physical_time,
                "used_dt": float(
                    hist.get("dt_used", transport_result["dt_control"]["used_dt"])
                ),
                "cfl_max": float(hist.get("cfl_max", transport_result["cfl_max"])),
                "cfl_warning": bool(hist.get("cfl_warning", False)),
                "min_C": min_c,
                "max_C": max_c,
                "mean_C": mean_c,
                "total_mass_proxy": total_mass,
                "cumulative_mass_in": cumulative_in,
                "cumulative_mass_out": cumulative_out,
                "mass_balance_error": mass_balance_error,
                "domain_mass_initial": float(
                    mass_diag_snap.get("domain_mass_initial", domain_mass_initial_snap)
                ),
                "domain_mass_current": float(
                    mass_diag_snap.get("domain_mass_current", total_mass)
                ),
                "expected_domain_mass_current": float(
                    mass_diag_snap.get("expected_domain_mass_current", 0.0)
                ),
                "domain_mass_balance_residual": float(
                    mass_diag_snap.get("domain_mass_balance_residual", 0.0)
                ),
                "domain_mass_balance_relative_to_throughput": float(
                    mass_diag_snap.get(
                        "domain_mass_balance_relative_to_throughput", 0.0
                    )
                ),
                "domain_mass_balance_relative_to_domain_mass": float(
                    mass_diag_snap.get(
                        "domain_mass_balance_relative_to_domain_mass", 0.0
                    )
                ),
                "mass_balance_formula_used": str(
                    mass_diag_snap.get("mass_balance_formula_used", "")
                ),
                "mass_conservation_error": float(
                    hist.get("conservation_error_after_limiter", 0.0)
                ),
                "pairwise_conservation_error_max_abs": float(
                    hist.get("pairwise_conservation_error_max_abs", 0.0)
                ),
                "overshoot_max": float(hist.get("overshoot_before_clip", 0.0)),
                "undershoot_max": float(hist.get("undershoot_before_clip", 0.0)),
                "clipped_cell_count": int(hist.get("clipped_cell_count", 0)),
                "clipped_fraction": float(hist.get("clipped_fraction", 0.0)),
                "limiter_active_cell_count": int(
                    hist.get("limiter_active_cell_count", 0)
                ),
                "limiter_active_fraction": float(
                    hist.get("limiter_active_fraction", 0.0)
                ),
                "limiter_mass_correction_total": float(
                    hist.get("limiter_mass_correction_total", 0.0)
                ),
                "outlet_frac_gt_1e-6": _read_outlet_fraction(hist, 1e-6),
                "outlet_frac_gt_1e-4": _read_outlet_fraction(hist, 1e-4),
                "outlet_frac_gt_1e-3": _read_outlet_fraction(hist, 1e-3),
                "outlet_frac_gt_1e-2": _read_outlet_fraction(hist, 1e-2),
                "outlet_C_mean": float(
                    outlet_mix_snap.get("C_mean_outlet", float("nan"))
                ),
                "outlet_C_std": float(
                    outlet_mix_snap.get("C_std_outlet", float("nan"))
                ),
                "outlet_mixing_index_simple": float(
                    outlet_mix_snap.get("mixing_index_simple", float("nan"))
                ),
                "outlet_normalized_unmixedness": float(
                    outlet_mix_snap.get("normalized_unmixedness", float("nan"))
                ),
                "max_raw_delta_c": float(hist.get("max_raw_delta_c", 0.0)),
                "raw_overshoot_before_limiter": float(
                    hist.get("raw_overshoot_before_limiter", 0.0)
                ),
                "raw_undershoot_before_limiter": float(
                    hist.get("raw_undershoot_before_limiter", 0.0)
                ),
                "diffusion_laplacian_max_abs": float(
                    hist.get("diffusion_laplacian_max_abs", 0.0)
                ),
                "numerically_clean_transport": numerically_clean_snapshot,
                "artifacts": {
                    "concentration_xy_png": str(png_path),
                    "result_vtu": str(vtu_step),
                },
            }
            snap_item.update(
                _snapshot_front_diagnostics(mesh.cell_centers, scalar_step)
            )
            snap_item["front_location"] = {
                "min_x_where_C_gt_1e-4": float(
                    snap_item.get("min_x_where_C_gt_1e-4", float("nan"))
                ),
                "max_x_where_C_gt_1e-4": float(
                    snap_item.get("max_x_where_C_gt_1e-4", float("nan"))
                ),
                "max_y_where_C_gt_1e-4": float(
                    snap_item.get("max_y_where_C_gt_1e-4", float("nan"))
                ),
                "min_z_where_C_gt_1e-4": float(
                    snap_item.get("min_z_where_C_gt_1e-4", float("nan"))
                ),
                "max_z_where_C_gt_1e-4": float(
                    snap_item.get("max_z_where_C_gt_1e-4", float("nan"))
                ),
                "concentration_centroid_x": float(
                    snap_item.get("concentration_centroid_x", float("nan"))
                ),
                "concentration_centroid_y": float(
                    snap_item.get("concentration_centroid_y", float("nan"))
                ),
                "concentration_centroid_z": float(
                    snap_item.get("concentration_centroid_z", float("nan"))
                ),
            }
            snap_item["scalar_front_velocity_audit"] = _scalar_front_velocity_audit(
                centers=mesh.cell_centers,
                velocity=velocity,
                concentration=scalar_step,
            )
            snapshots_summary["snapshots"].append(snap_item)

        combined_png = _save_snapshots_comparison_xy(
            centers=mesh.cell_centers,
            snapshots=snapshot_scalars,
            output_path=run_dir / "snapshots_concentration_comparison_xy.png",
        )
        if combined_png is not None:
            snapshots_summary["combined_png"] = combined_png
        snapshots_summary["mass_balance_formula_used"] = (
            "domain_mass(t)=domain_mass_initial+cumulative_scalar_mass_in-cumulative_scalar_mass_out"
        )
        snapshots_summary["domain_mass_initial"] = float(domain_mass_initial_snap)
        _write_json(run_dir / "snapshots_summary.json", snapshots_summary)

        vtu_path = _export_transport_vtu(
            points=mesh.points,
            tetrahedra=mesh.tetrahedra,
            concentration=concentration,
            velocity=velocity,
            output_path=run_dir / f"{mesh_npz.stem}_transport_result.vtu",
        )
        previews = _save_transport_previews(
            centers=mesh.cell_centers,
            concentration=concentration,
            velocity_magnitude=speed,
            clipped_mask=transport_result["final_step_debug"]["clipped_mask"],  # type: ignore[index]
            overshoot_mask=transport_result["final_step_debug"]["overshoot_mask"],  # type: ignore[index]
            undershoot_mask=transport_result["final_step_debug"]["undershoot_mask"],  # type: ignore[index]
            left_inlet_cells=transport_result["final_step_debug"]["left_inlet_cells"],  # type: ignore[index]
            right_inlet_cells=transport_result["final_step_debug"]["right_inlet_cells"],  # type: ignore[index]
            outlet_cells=transport_result["final_step_debug"]["outlet_cells"],  # type: ignore[index]
            left_inlet_zone_mask=transport_result["final_step_debug"][
                "left_inlet_zone_mask"
            ],  # type: ignore[index]
            right_inlet_zone_mask=transport_result["final_step_debug"][
                "right_inlet_zone_mask"
            ],  # type: ignore[index]
            inlet_wall_corner_zone_mask=transport_result["final_step_debug"][
                "inlet_wall_corner_zone_mask"
            ],  # type: ignore[index]
            boundary_faces=transport_result["final_step_debug"]["boundary_faces"],  # type: ignore[index]
            boundary_face_groups=transport_result["final_step_debug"][
                "boundary_face_groups"
            ],  # type: ignore[index]
            run_dir=run_dir,
            file_prefix=mesh_npz.stem,
        )

        operator_diag = run_operator_diagnostics(
            mesh,
            gradient_method=args.gradient_method,  # type: ignore[arg-type]
        )

        clip_info = transport_result["clipping"]
        clip_fraction = float(clip_info["final_clipped_fraction"])
        overshoot_peak = float(clip_info["max_overshoot_before_clip"])
        undershoot_peak = float(clip_info["max_undershoot_before_clip"])
        overshoot_total = int(clip_info["final_overshoot_cell_count"])
        undershoot_total = int(clip_info["final_undershoot_cell_count"])
        clipped_total = int(clip_info["final_clipped_cell_count"])
        diffusion_diag = dict(transport_result.get("diffusion_diagnostics", {}))
        finite_values = bool(np.all(np.isfinite(concentration)))
        cleanliness_flags = _compute_cleanliness_flags(
            finite=finite_values,
            cfl_warning=bool(transport_result["cfl_warning"]),
            diffusion_stability_warning=bool(
                diffusion_diag.get("diffusion_stability_warning", False)
            ),
            overshoot_max=overshoot_peak,
            undershoot_max=undershoot_peak,
            clipped_count=clipped_total,
            tolerance=float(args.boundedness_tolerance),
        )
        numerically_clean_transport = bool(
            cleanliness_flags["strict_numerically_clean_transport"]
        )
        backend_execution = transport_result["backend_execution"]
        stepping_backend = str(backend_execution.get("stepping_backend", "unknown"))
        stepping_device = str(backend_execution.get("device", "unknown"))
        pure_gpu_run = bool(
            (not bool(args.compare_cpu_gpu))
            and execution_backend == "torch"
            and stepping_backend == "torch"
            and stepping_device.startswith("cuda")
        )
        run_mode = "gpu_transport_long_run" if pure_gpu_run else "transport_run"
        last_outer_dt = float(transport_result["dt_control"]["used_dt"])
        used_dt = float(transport_time_estimate.get("used_dt", last_outer_dt))
        run_control = dict(transport_result.get("run_control", {}))
        completed_steps = int(
            run_control.get("completed_steps", len(transport_result.get("history", [])))
        )
        total_steps_requested = int(
            run_control.get("total_steps_requested", transport_cfg.steps)
        )
        stopped_due_walltime = bool(run_control.get("stopped_due_walltime", False))
        runtime_seconds = float(time.perf_counter() - run_started)
        steps_per_second = (
            float(completed_steps / runtime_seconds) if runtime_seconds > 0 else 0.0
        )
        physical_time_final = float(
            run_control.get("physical_time_final", used_dt * completed_steps)
        )
        final_front = _snapshot_front_diagnostics(mesh.cell_centers, concentration)
        outlet_arrival = dict(transport_result["outlet_arrival"])
        outlet_target_y = (
            float(
                np.max(
                    mesh.face_centers[np.asarray(mesh.outlet_faces, dtype=np.int64), 1]
                )
            )
            if np.asarray(mesh.outlet_faces, dtype=np.int64).size
            else float(np.max(mesh.cell_centers[:, 1]))
        )
        current_front_y = float(final_front.get("max_y_where_C_gt_1e-4", float("nan")))
        snap_items = list(snapshots_summary.get("snapshots", []))
        front_progress_per_step = float("nan")
        front_progress_per_physical_time = float("nan")
        if len(snap_items) >= 2:
            first = dict(snap_items[0])
            last = dict(snap_items[-1])
            y0 = float(first.get("max_y_where_C_gt_1e-4", float("nan")))
            y1 = float(last.get("max_y_where_C_gt_1e-4", float("nan")))
            s0 = int(first.get("step", 0))
            s1 = int(last.get("step", 0))
            t0 = float(first.get("physical_time", 0.0))
            t1 = float(last.get("physical_time", 0.0))
            if np.isfinite(y0) and np.isfinite(y1) and s1 > s0:
                front_progress_per_step = float((y1 - y0) / max(1, s1 - s0))
            if np.isfinite(y0) and np.isfinite(y1) and (t1 - t0) > 0.0:
                front_progress_per_physical_time = float((y1 - y0) / (t1 - t0))
        if (
            np.isfinite(current_front_y)
            and np.isfinite(front_progress_per_step)
            and front_progress_per_step > 0.0
        ):
            remaining_y = max(outlet_target_y - current_front_y, 0.0)
            estimated_steps_to_outlet = float(remaining_y / front_progress_per_step)
        else:
            estimated_steps_to_outlet = float("nan")

        masses = dict(transport_result["masses"])
        scalar_mass_initial = float(
            masses.get("scalar_mass_initial", masses.get("initial_mass_proxy", 0.0))
        )
        scalar_mass_final = float(
            masses.get(
                "scalar_mass_final",
                masses.get(
                    "final_mass_proxy", masses.get("total_mass_proxy_final", 0.0)
                ),
            )
        )
        cumulative_mass_in = float(masses.get("cumulative_mass_in", 0.0))
        cumulative_mass_out = float(masses.get("cumulative_mass_out", 0.0))
        mass_balance_error = float(
            scalar_mass_final
            - scalar_mass_initial
            - cumulative_mass_in
            + cumulative_mass_out
        )
        mass_ref = max(
            abs(scalar_mass_initial),
            abs(cumulative_mass_in),
            abs(cumulative_mass_out),
            abs(scalar_mass_final),
            1e-30,
        )
        relative_mass_balance_error = float(abs(mass_balance_error) / mass_ref)
        outlet_cells_final = np.asarray(
            transport_result["final_step_debug"].get(
                "outlet_cells", np.zeros((0,), dtype=np.int64)
            ),
            dtype=np.int64,
        )
        outlet_profile = _write_outlet_profile_artifacts(
            run_dir=run_dir,
            centers=np.asarray(mesh.cell_centers, dtype=np.float64),
            concentration=concentration,
            outlet_cells=outlet_cells_final,
        )
        outlet_mixing_metrics = dict(outlet_profile.get("mixing_metrics", {}))
        mass_diag = _compute_mass_diagnostics(
            domain_mass_initial=float(
                run_control.get(
                    "domain_mass_initial",
                    transport_result["masses"].get("initial_mass_proxy", 0.0),
                )
            ),
            domain_mass_current=float(scalar_mass_final),
            cumulative_scalar_mass_in=float(cumulative_mass_in),
            cumulative_scalar_mass_out=float(cumulative_mass_out),
        )
        source_flow_md = (
            dict(source_flow_payload.get("metadata", {}))
            if velocity_source == "flow_run"
            else {}
        )
        if velocity_source == "flow_run":
            coupling_mesh_compatible = bool(
                source_flow_compatibility.get("compatible", False)
            )
            coupling_mesh_compatibility_summary = source_flow_compatibility
        else:
            coupling_mesh_compatible = True
            coupling_mesh_compatibility_summary = {
                "compatible": True,
                "mode": "prescribed",
            }
        source_flow_flux_balance = (
            dict(source_flow_md.get("source_flow_flux_balance", {}))
            if isinstance(source_flow_md.get("source_flow_flux_balance", {}), dict)
            else {}
        )

        summary = {
            "run_mode": run_mode,
            "original_input": original_mesh_input,
            "resolved_mesh_npz": str(mesh_npz),
            "mesh_stem": mesh_stem,
            "mesh_npz": str(mesh_npz),
            "mesh_name": mesh_stem,
            "requested_backend": str(args.backend),
            "selected_backend": str(backend.selected_backend),
            "transport_execution_backend": execution_backend,
            "stepping_backend": stepping_backend,
            "device": stepping_device,
            "used_numpy_fallback": bool(
                backend_execution.get("used_numpy_fallback", False)
            ),
            "all_core_arrays_on_cuda": bool(
                backend_execution.get("all_core_arrays_on_cuda", False)
            ),
            "per_step_large_cpu_gpu_transfer_warning": bool(
                backend_execution.get("per_step_large_cpu_gpu_transfer_warning", False)
            ),
            "pure_gpu_run": pure_gpu_run,
            "run_control": run_control,
            "steps_requested": int(total_steps_requested),
            "steps_completed": int(completed_steps),
            "run_completed_full_steps": bool(completed_steps >= total_steps_requested),
            "stopped_due_walltime": bool(stopped_due_walltime),
            "backend": backend.__dict__,
            "velocity_field": {
                "name": velocity_field.name,
                "metadata": velocity_field.metadata,
            },
            "velocity_diagnostics": velocity_diagnostics,
            "flow_solved": (
                bool(source_flow_ready_for_transport)
                if velocity_source == "flow_run"
                else False
            ),
            "pressure_solved": False,
            "velocity_source": str(velocity_source),
            "source_flow_run_dir": (
                str(source_flow_payload.get("flow_run_dir", ""))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_ready_for_transport": (
                bool(source_flow_ready_for_transport)
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_stage_status_reason": (
                str(readiness_info.get("stage_status_reason", ""))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_run_completed": (
                bool(readiness_info.get("run_completed", False))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_numerically_stable": (
                bool(readiness_info.get("numerically_stable", False))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_physically_ready": (
                bool(readiness_info.get("physically_ready", False))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_ready_for_long_run": (
                bool(readiness_info.get("ready_for_long_run", False))
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_physical_time_final": (
                source_flow_md.get("physical_time_final")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_final_div_l2": (
                source_flow_md.get("final_div_l2")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_final_div_max": (
                source_flow_md.get("final_div_max")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_outlet_inlet_flux_ratio": (
                source_flow_md.get("outlet_inlet_flux_ratio")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_nonphysical_flux_fix_used": (
                source_flow_md.get("nonphysical_flux_fix_used")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_convective_auto_damping_used_any": (
                source_flow_md.get("convective_auto_damping_used_any")
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_flux_balance": source_flow_flux_balance,
            "coupling_mesh_compatible": bool(coupling_mesh_compatible),
            "coupling_mesh_compatibility": coupling_mesh_compatibility_summary,
            "coupling_flux_loaded_successfully": bool(
                coupling_flux_loaded_successfully
                if velocity_source == "flow_run"
                else True
            ),
            "transport_time_estimate": dict(transport_time_estimate),
            "transport_regime_audit": regime_audit,
            "transport_accuracy_supported": bool(regime_supported_accuracy),
            "transport_accuracy_support_status": str(
                regime_audit.get("support_status", "unsupported")
            ),
            "transport_accuracy_support_reason_codes": list(
                regime_audit.get("reason_codes", [])
            ),
            "transport_regime_warning": bool(regime_warning),
            "transport_regime_warning_codes": regime_warning_codes,
            "transport_regime_blocking_error": bool(regime_blocking_error),
            "transport_regime_error_codes": regime_error_codes,
            "front_progress_per_step": float(front_progress_per_step),
            "front_progress_per_physical_time": float(front_progress_per_physical_time),
            "estimated_steps_to_outlet": float(estimated_steps_to_outlet),
            "current_min_x_where_C_gt_1e-4": float(
                final_front.get("min_x_where_C_gt_1e-4", float("nan"))
            ),
            "current_max_y_where_C_gt_1e-4": float(
                final_front.get("max_y_where_C_gt_1e-4", float("nan"))
            ),
            "outlet_mixing_metrics": outlet_mixing_metrics,
            "mass_balance_formula_used": str(
                mass_diag.get("mass_balance_formula_used", "")
            ),
            "domain_mass_initial": float(mass_diag.get("domain_mass_initial", 0.0)),
            "domain_mass_final": float(mass_diag.get("domain_mass_current", 0.0)),
            "cumulative_scalar_mass_in": float(
                mass_diag.get("cumulative_scalar_mass_in", 0.0)
            ),
            "cumulative_scalar_mass_out": float(
                mass_diag.get("cumulative_scalar_mass_out", 0.0)
            ),
            "expected_domain_mass_final": float(
                mass_diag.get("expected_domain_mass_current", 0.0)
            ),
            "domain_mass_balance_residual": float(
                mass_diag.get("domain_mass_balance_residual", 0.0)
            ),
            "domain_mass_balance_relative_to_throughput": float(
                mass_diag.get("domain_mass_balance_relative_to_throughput", 0.0)
            ),
            "domain_mass_balance_relative_to_domain_mass": float(
                mass_diag.get("domain_mass_balance_relative_to_domain_mass", 0.0)
            ),
            "local_pairwise_conservation_clean": bool(
                abs(
                    float(
                        transport_result["conservation_audit"][
                            "interior_pairwise_conservation_error_max_abs"
                        ]
                    )
                )
                <= 1e-12
            ),
            "validation": validation.summary,
            "flux_diagnostics": {
                **flux_diag,
                "velocity_field_name": str(velocity_field.name),
                "balanced_velocity_enabled": bool(
                    velocity_field.metadata.get("balanced_velocity_enabled", False)
                ),
                "inlet_flux_before_balance": velocity_field.metadata.get(
                    "inlet_flux_before_balance"
                ),
                "outlet_flux_before_balance": velocity_field.metadata.get(
                    "outlet_flux_before_balance"
                ),
                "outlet_scale_factor": velocity_field.metadata.get(
                    "outlet_scale_factor"
                ),
                "inlet_flux_after_balance": velocity_field.metadata.get(
                    "inlet_flux_after_balance"
                ),
                "outlet_flux_after_balance": velocity_field.metadata.get(
                    "outlet_flux_after_balance"
                ),
                "net_boundary_flux_after_balance": velocity_field.metadata.get(
                    "net_boundary_flux_after_balance"
                ),
                "boundary_flux_imbalance_ratio_after_balance": velocity_field.metadata.get(
                    "boundary_flux_imbalance_ratio_after_balance"
                ),
                "wall_flux_max_abs": flux_diag.get("wall_flux_max_abs"),
            },
            "operator_diagnostics": operator_diag,
            "transport": {
                "config": {
                    "mode": transport_cfg.transport_mode,
                    "transport_scheme": transport_cfg.transport_scheme,
                    "dt": transport_cfg.dt,
                    "dt_mode": transport_cfg.dt_mode,
                    "cfl_target": transport_cfg.cfl_target,
                    "auto_dt_percentile": transport_cfg.auto_dt_percentile,
                    "max_transport_substeps": transport_cfg.max_transport_substeps,
                    "transport_substep_warning_threshold": (
                        transport_cfg.transport_substep_warning_threshold
                    ),
                    "steps": transport_cfg.steps,
                    "snapshot_steps": list(transport_cfg.snapshot_steps),
                    "diffusivity": transport_cfg.diffusivity,
                    "cfl_limit": transport_cfg.cfl_limit,
                    "gradient_method": transport_cfg.gradient_method,
                    "laplacian_method": transport_cfg.laplacian_method,
                    "clipping_enabled": transport_cfg.clipping_enabled,
                    "safety_clamp_after_diffusion": transport_cfg.safety_clamp_after_diffusion,
                    "boundedness_tolerance": transport_cfg.boundedness_tolerance,
                },
                "cfl_max": transport_result["cfl_max"],
                "cfl_warning": transport_result["cfl_warning"],
                "dt_control": transport_result["dt_control"],
                "boundary_setup": transport_result["boundary_setup"],
                "clipping": transport_result["clipping"],
                "limiter": transport_result["limiter"],
                "masses": transport_result["masses"],
                "raw_update_audit": transport_result["raw_update_audit"],
                "conservation_audit": transport_result["conservation_audit"],
                "local_cfl_audit": transport_result["local_cfl_audit"],
                "outlet_arrival": transport_result["outlet_arrival"],
                "backend_execution": transport_result["backend_execution"],
                "diffusion_diagnostics": transport_result.get(
                    "diffusion_diagnostics", {}
                ),
                "scalar_bc_mode": transport_result["transport_info"]["scalar_bc_mode"],
                "inlet_dirichlet_applied_on": transport_result["transport_info"][
                    "inlet_dirichlet_applied_on"
                ],
                "post_step_cell_overwrite_used": transport_result["transport_info"][
                    "post_step_cell_overwrite_used"
                ],
                "cell_level_inlet_overwrite_used": transport_result["transport_info"][
                    "cell_level_inlet_overwrite_used"
                ],
                "final_stats": transport_result["final_stats"],
            },
            "numerical_cleanliness": {
                "numerically_clean_transport": bool(numerically_clean_transport),
                "strict_numerically_clean_transport": bool(
                    cleanliness_flags["strict_numerically_clean_transport"]
                ),
                "tolerance_numerically_clean_transport": bool(
                    cleanliness_flags["tolerance_numerically_clean_transport"]
                ),
                "boundedness_tolerance": float(args.boundedness_tolerance),
                "cleanliness_reason": str(cleanliness_flags["cleanliness_reason"]),
                "overshoot_cell_count_total": overshoot_total,
                "undershoot_cell_count_total": undershoot_total,
                "clipped_cell_count_total": clipped_total,
                "right_inlet_near_zone_overshoot_cell_count": int(
                    clip_info["final_right_inlet_zone_overshoot_cell_count"]
                ),
                "left_inlet_near_zone_overshoot_cell_count": int(
                    clip_info["final_left_inlet_zone_overshoot_cell_count"]
                ),
                "corner_zone_overshoot_cell_count": int(
                    clip_info["final_inlet_wall_corner_zone_overshoot_cell_count"]
                ),
                "right_inlet_near_zone_undershoot_cell_count": int(
                    clip_info["final_right_inlet_zone_undershoot_cell_count"]
                ),
                "left_inlet_near_zone_undershoot_cell_count": int(
                    clip_info["final_left_inlet_zone_undershoot_cell_count"]
                ),
                "corner_zone_undershoot_cell_count": int(
                    clip_info["final_inlet_wall_corner_zone_undershoot_cell_count"]
                ),
                "right_inlet_near_zone_clipped_cell_count": int(
                    clip_info["final_right_inlet_zone_clipped_cell_count"]
                ),
                "left_inlet_near_zone_clipped_cell_count": int(
                    clip_info["final_left_inlet_zone_clipped_cell_count"]
                ),
                "corner_zone_clipped_cell_count": int(
                    clip_info["final_inlet_wall_corner_zone_clipped_cell_count"]
                ),
                "final_clipped_fraction": clip_fraction,
                "max_overshoot_before_clip": overshoot_peak,
                "max_undershoot_before_clip": undershoot_peak,
                "overshoot_after_limiter_before_clip": float(
                    clip_info["max_overshoot_after_limiter_before_clip"]
                ),
                "undershoot_after_limiter_before_clip": float(
                    clip_info["max_undershoot_after_limiter_before_clip"]
                ),
                "clipped_cell_count_after_limiter": int(
                    clip_info["clipped_cell_count_after_limiter"]
                ),
                "post_step_clipping_used": bool(clip_info["post_step_clipping_used"]),
                "safety_clamp_after_diffusion_used": bool(
                    clip_info.get("safety_clamp_after_diffusion_used", False)
                ),
                "safety_clamp_cell_count_total": int(
                    clip_info.get("safety_clamp_cell_count_total", 0)
                ),
                "safety_clamp_mass_delta": float(
                    clip_info.get("safety_clamp_mass_delta", 0.0)
                ),
                "safety_clamp_max_correction": float(
                    clip_info.get("safety_clamp_max_correction", 0.0)
                ),
                "cfl_warning": bool(transport_result["cfl_warning"]),
                "diffusion_stability_warning": bool(
                    diffusion_diag.get("diffusion_stability_warning", False)
                ),
            },
            "long_run_status": "warning",
            "transport_variants": {
                v_name: {
                    scheme_name: {
                        dt_mode: {
                            "dt_control": v_data["scheme_runs"][scheme_name]["dt_runs"][
                                dt_mode
                            ]["result"]["dt_control"],  # type: ignore[index]
                            "cfl_max": v_data["scheme_runs"][scheme_name]["dt_runs"][
                                dt_mode
                            ]["result"]["cfl_max"],  # type: ignore[index]
                            "cfl_warning": v_data["scheme_runs"][scheme_name][
                                "dt_runs"
                            ][dt_mode]["result"]["cfl_warning"],  # type: ignore[index]
                            "clipping": v_data["scheme_runs"][scheme_name]["dt_runs"][
                                dt_mode
                            ]["result"]["clipping"],  # type: ignore[index]
                            "limiter": v_data["scheme_runs"][scheme_name]["dt_runs"][
                                dt_mode
                            ]["result"]["limiter"],  # type: ignore[index]
                            "final_stats": v_data["scheme_runs"][scheme_name][
                                "dt_runs"
                            ][dt_mode]["result"]["final_stats"],  # type: ignore[index]
                        }
                        for dt_mode in run_variants
                    }
                    for scheme_name in scheme_variants
                    if scheme_name in v_data["scheme_runs"]  # type: ignore[operator]
                }
                for v_name, v_data in velocity_runs.items()
            },
            "velocity_balance_comparison": comparison_payload,
            "velocity_field_audit": velocity_field_audit,
            "velocity_field_comparison": velocity_field_comparison,
            "transport_scheme_comparison": scheme_comparison_payload,
            "inlet_symmetry_audit": inlet_symmetry_audit,
            "inlet_symmetry_profile_comparison": symmetry_profile_comparison,
            "right_inlet_corner_debug": transport_result["right_inlet_corner_debug"],
            "artifacts": {
                "run_log": str(run_dir / "run.log"),
                "config_json": str(run_dir / "config.json"),
                "summary_json": str(run_dir / "summary.json"),
                "validation_report_txt": str(run_dir / "validation_report.txt"),
                "flux_diagnostics_json": str(run_dir / "flux_diagnostics.json"),
                "transport_history_json": str(run_dir / "transport_history.json"),
                "snapshots_summary_json": str(run_dir / "snapshots_summary.json"),
                "velocity_balance_comparison_json": (
                    str(run_dir / "velocity_balance_comparison.json")
                    if comparison_payload
                    else None
                ),
                "velocity_field_audit_json": str(run_dir / "velocity_field_audit.json"),
                "velocity_field_comparison_json": (
                    str(run_dir / "velocity_field_comparison.json")
                    if velocity_field_comparison
                    else None
                ),
                "inlet_symmetry_audit_json": str(run_dir / "inlet_symmetry_audit.json"),
                "inlet_symmetry_profile_comparison_json": (
                    str(run_dir / "inlet_symmetry_profile_comparison.json")
                    if symmetry_profile_comparison
                    else None
                ),
                "transport_scheme_comparison_json": (
                    str(run_dir / "transport_scheme_comparison.json")
                    if scheme_comparison_payload
                    else None
                ),
                "transport_vtu": str(vtu_path),
                "final_concentration_npy": str(run_dir / "final_concentration.npy"),
                "preview_png": previews,
                "velocity_preview_png": velocity_previews,
                "snapshots_dir": str(snapshots_dir),
                "snapshots_comparison_png": combined_png,
                "outlet_profile_csv": str(
                    outlet_profile.get("artifacts", {}).get("outlet_profile_csv", "")
                ),
                "outlet_profile_json": str(
                    outlet_profile.get("artifacts", {}).get("outlet_profile_json", "")
                ),
                "outlet_mixing_metrics_json": str(
                    outlet_profile.get("artifacts", {}).get(
                        "outlet_mixing_metrics_json", ""
                    )
                ),
                "outlet_concentration_profile_png": str(
                    outlet_profile.get("artifacts", {}).get(
                        "outlet_concentration_profile_png", ""
                    )
                ),
                "outlet_concentration_histogram_png": str(
                    outlet_profile.get("artifacts", {}).get(
                        "outlet_concentration_histogram_png", ""
                    )
                ),
                "checkpoints_dir": str(run_dir / "checkpoints"),
                "last_checkpoint_state_json": (
                    str(run_control.get("checkpoint_paths", [])[-1])
                    if run_control.get("checkpoint_paths", [])
                    else None
                ),
            },
        }
        summary["runtime_seconds"] = runtime_seconds
        summary["steps_per_second"] = steps_per_second
        summary["used_dt"] = used_dt
        summary["last_outer_dt"] = last_outer_dt
        summary["physical_time_final"] = physical_time_final
        summary["run_horizon_mode"] = str(run_control.get("run_horizon_mode", "steps"))
        summary["requested_transport_end_time"] = run_control.get(
            "requested_transport_end_time"
        )
        summary["progress_percent"] = float(
            100.0 * completed_steps / max(1, total_steps_requested)
            if str(run_control.get("run_horizon_mode", "steps")) == "steps"
            else min(
                100.0,
                100.0
                * physical_time_final
                / max(
                    float(run_control.get("requested_transport_end_time", 0.0)),
                    1e-30,
                ),
            )
        )
        summary["cfl_max"] = float(transport_result["cfl_max"])
        summary["cfl_warning"] = bool(transport_result["cfl_warning"])
        summary["final_C_min"] = float(np.min(concentration))
        summary["final_C_max"] = float(np.max(concentration))
        summary["final_C_mean"] = float(np.mean(concentration))
        summary["overshoot_count"] = overshoot_total
        summary["undershoot_count"] = undershoot_total
        summary["clipped_count"] = clipped_total
        summary["min_x_where_C_gt_1e-4"] = float(final_front["min_x_where_C_gt_1e-4"])
        summary["max_x_where_C_gt_1e-4"] = float(final_front["max_x_where_C_gt_1e-4"])
        summary["max_y_where_C_gt_1e-4"] = float(final_front["max_y_where_C_gt_1e-4"])
        summary["outlet_frac_gt_1e-6"] = _read_outlet_fraction(outlet_arrival, 1e-6)
        summary["outlet_frac_gt_1e-4"] = _read_outlet_fraction(outlet_arrival, 1e-4)
        summary["outlet_frac_gt_1e-3"] = _read_outlet_fraction(outlet_arrival, 1e-3)
        summary["outlet_frac_gt_1e-2"] = _read_outlet_fraction(outlet_arrival, 1e-2)
        summary["outlet_C_mean"] = float(
            outlet_mixing_metrics.get("C_mean_outlet", float("nan"))
        )
        summary["outlet_C_std"] = float(
            outlet_mixing_metrics.get("C_std_outlet", float("nan"))
        )
        summary["outlet_C_min"] = float(
            outlet_mixing_metrics.get("C_min_outlet", float("nan"))
        )
        summary["outlet_C_max"] = float(
            outlet_mixing_metrics.get("C_max_outlet", float("nan"))
        )
        summary["outlet_mixing_index_simple"] = float(
            outlet_mixing_metrics.get("mixing_index_simple", float("nan"))
        )
        summary["outlet_normalized_unmixedness"] = float(
            outlet_mixing_metrics.get("normalized_unmixedness", float("nan"))
        )

        masses["scalar_mass_initial"] = scalar_mass_initial
        masses["scalar_mass_final"] = scalar_mass_final
        masses["initial_mass_proxy"] = scalar_mass_initial
        masses["final_mass_proxy"] = scalar_mass_final
        masses["total_mass_proxy_final"] = scalar_mass_final
        masses["mass_balance_error"] = mass_balance_error
        masses["relative_mass_balance_error"] = relative_mass_balance_error
        summary["transport"]["masses"] = masses
        summary["scalar_mass_initial"] = scalar_mass_initial
        summary["scalar_mass_final"] = scalar_mass_final
        summary["cumulative_mass_in"] = cumulative_mass_in
        summary["cumulative_mass_out"] = cumulative_mass_out
        summary["mass_balance_error"] = float(
            mass_diag.get("domain_mass_balance_residual", mass_balance_error)
        )
        summary["relative_mass_balance_error"] = relative_mass_balance_error
        summary["relative_mass_balance_error_legacy"] = relative_mass_balance_error
        summary["mass_diagnostics_interpretation"] = (
            "Local pairwise conservation is checked separately (interior face cancellation). "
            "Domain mass residual compares domain_mass_final against "
            "domain_mass_initial + cumulative_scalar_mass_in - cumulative_scalar_mass_out. "
            "relative_mass_balance_error_legacy is retained for backward compatibility."
        )
        summary["transport_mass_balance_error"] = float(
            mass_diag.get("domain_mass_balance_residual", mass_balance_error)
        )
        diffusion_diag = dict(transport_result.get("diffusion_diagnostics", {}))
        summary["transport_mode"] = str(transport_cfg.transport_mode)
        summary["transport_scheme"] = str(transport_cfg.transport_scheme)
        summary["laplacian_method"] = str(transport_cfg.laplacian_method)
        summary["gradient_method"] = str(transport_cfg.gradient_method)
        summary["diffusivity"] = float(transport_cfg.diffusivity)
        summary["diffusion_backend"] = str(
            diffusion_diag.get("diffusion_backend", "none")
        )
        summary["diffusion_dt_limit"] = float(
            diffusion_diag.get("diffusion_dt_limit", float("inf"))
        )
        summary["diffusion_stability_warning"] = bool(
            diffusion_diag.get("diffusion_stability_warning", False)
        )
        summary["max_overshoot_after_diffusion_before_clip"] = float(
            diffusion_diag.get("max_overshoot_after_diffusion_before_clip", 0.0)
        )
        summary["max_undershoot_after_diffusion_before_clip"] = float(
            diffusion_diag.get("max_undershoot_after_diffusion_before_clip", 0.0)
        )
        summary["safety_clamp_after_diffusion_used"] = bool(
            clip_info.get("safety_clamp_after_diffusion_used", False)
        )
        summary["safety_clamp_mass_delta"] = float(
            clip_info.get("safety_clamp_mass_delta", 0.0)
        )
        summary["boundedness_min"] = float(np.min(concentration))
        summary["boundedness_max"] = float(np.max(concentration))
        summary["boundedness_overshoot"] = float(overshoot_peak)
        summary["boundedness_undershoot"] = float(undershoot_peak)
        hist_all = list(transport_result.get("history", []))
        if hist_all:
            last = dict(hist_all[-1])
            max_change_last_step = float(last.get("max_change", 0.0))
            mean_change_last_step = float(last.get("mean_change", 0.0))
            rel_change_last_step = float(last.get("relative_change", 0.0))
        else:
            max_change_last_step = float("nan")
            mean_change_last_step = float("nan")
            rel_change_last_step = float("nan")
        rolling_n = min(50, len(hist_all))
        if rolling_n > 0:
            tail = hist_all[-rolling_n:]
            rolling_mean_max_change = float(
                np.mean([float(r.get("max_change", 0.0)) for r in tail])
            )
            rolling_mean_mean_change = float(
                np.mean([float(r.get("mean_change", 0.0)) for r in tail])
            )
        else:
            rolling_mean_max_change = float("nan")
            rolling_mean_mean_change = float("nan")
        stationarity_candidate = bool(
            np.isfinite(max_change_last_step)
            and np.isfinite(mean_change_last_step)
            and max_change_last_step <= 1e-5
            and mean_change_last_step <= 1e-7
        )
        summary["max_change_last_step"] = float(max_change_last_step)
        summary["mean_change_last_step"] = float(mean_change_last_step)
        summary["relative_change_last_step"] = float(rel_change_last_step)
        summary["rolling_change_window"] = int(rolling_n)
        summary["rolling_mean_max_change"] = float(rolling_mean_max_change)
        summary["rolling_mean_mean_change"] = float(rolling_mean_mean_change)
        summary["stationarity_candidate"] = bool(stationarity_candidate)
        front_observation = _evaluate_transport_front_observation(
            mesh=mesh,
            cell_velocity=np.asarray(cell_velocity, dtype=np.float64),
            flux_diag=flux_diag,
            physical_time_final=float(summary.get("physical_time_final", 0.0)),
            outlet_arrival=outlet_arrival,
            outlet_c_mean=float(summary.get("outlet_C_mean", 0.0)),
            breakthrough_turnover_threshold=float(args.breakthrough_turnover_threshold),
            breakthrough_travel_fraction=float(args.breakthrough_travel_fraction),
            outlet_fraction_detection_threshold=float(
                args.breakthrough_outlet_frac_threshold
            ),
        )
        summary["transport_front_observation"] = dict(front_observation)
        summary["transport_turnover_count"] = float(
            front_observation.get("transport_turnover_count", 0.0)
        )
        summary["travel_time_estimate"] = float(
            front_observation.get("travel_time_estimate", 0.0)
        )
        summary["inlet_outlet_distance"] = float(
            front_observation.get("inlet_outlet_distance", 0.0)
        )
        summary["characteristic_streamwise_speed"] = float(
            front_observation.get("characteristic_streamwise_speed", 0.0)
        )
        summary["breakthrough_expected"] = bool(
            front_observation.get("breakthrough_expected", False)
        )
        summary["breakthrough_detected"] = bool(
            front_observation.get("breakthrough_detected", False)
        )
        summary["transport_front_observation_reason"] = str(
            front_observation.get("observation_reason", "")
        )
        mass_error_abs = abs(
            float(summary["transport"]["limiter"]["conservation_error_after_limiter"])
        )
        pairwise_err = abs(
            float(
                summary["transport"]["conservation_audit"][
                    "interior_pairwise_conservation_error_max_abs"
                ]
            )
        )
        strict_clean = bool(
            summary["numerical_cleanliness"]["strict_numerically_clean_transport"]
        )
        tol_clean = bool(
            summary["numerical_cleanliness"]["tolerance_numerically_clean_transport"]
        )
        cfl_warn = bool(summary["numerical_cleanliness"]["cfl_warning"])
        finite_final = bool(np.all(np.isfinite(concentration)))
        pairwise_conservation_ok = bool(pairwise_err <= 1e-12)
        limiter_mass_conservation_ok = bool(mass_error_abs <= 1e-10)
        numerically_stable = bool(
            finite_final
            and bool(tol_clean)
            and (not cfl_warn)
            and pairwise_conservation_ok
            and limiter_mass_conservation_ok
        )
        run_completed = bool(summary.get("run_completed_full_steps", False)) and (
            not bool(summary.get("stopped_due_walltime", False))
        )
        dt_control_selected = dict(transport_result.get("dt_control", {}))
        transport_dt_blocked = bool(
            dt_control_selected.get("transport_dt_controller_blocked", False)
        )
        transport_dt_warning = bool(
            dt_control_selected.get("transport_substep_warning", False)
        )
        coupling_ready = bool(
            coupling_mesh_compatible
            and (
                bool(coupling_flux_loaded_successfully)
                if velocity_source == "flow_run"
                else True
            )
        )
        if velocity_source == "flow_run":
            coupling_ready = bool(
                coupling_ready and bool(source_flow_ready_for_transport)
            )
        physically_ready = bool(
            run_completed
            and numerically_stable
            and coupling_ready
            and not regime_blocking_error
        )
        ready_for_next_stage = bool(physically_ready and not transport_dt_blocked)
        ready_for_long_run = bool(
            ready_for_next_stage
            and (not bool(summary.get("used_numpy_fallback", True)))
            and bool(summary.get("pure_gpu_run", False))
        )
        stage_status = _build_stage_status(
            run_completed=bool(run_completed),
            numerically_stable=bool(numerically_stable),
            physically_ready=bool(physically_ready),
            ready_for_next_stage=bool(ready_for_next_stage),
            ready_for_long_run=bool(ready_for_long_run),
            checks={
                "run_completed_full_steps": bool(
                    summary.get("run_completed_full_steps", False)
                ),
                "stopped_due_walltime": bool(
                    summary.get("stopped_due_walltime", False)
                ),
                "finite_fields_final": bool(finite_final),
                "tolerance_clean": bool(tol_clean),
                "strict_clean": bool(strict_clean),
                "cfl_warning": bool(cfl_warn),
                "pairwise_conservation_ok": bool(pairwise_conservation_ok),
                "limiter_mass_conservation_ok": bool(limiter_mass_conservation_ok),
                "breakthrough_expected": bool(
                    front_observation.get("breakthrough_expected", False)
                ),
                "breakthrough_detected": bool(
                    front_observation.get("breakthrough_detected", False)
                ),
                "projected_breakthrough_expected": bool(
                    transport_time_estimate.get("breakthrough_expected", False)
                ),
                "transport_accuracy_supported": bool(regime_supported_accuracy),
                "transport_regime_warning": bool(regime_warning),
                "transport_regime_blocking_error": bool(regime_blocking_error),
                "transport_dt_substep_cap_hit": bool(
                    dt_control_selected.get("transport_substep_cap_hit", False)
                ),
                "transport_dt_controller_blocked": bool(transport_dt_blocked),
                "transport_dt_substep_warning": bool(transport_dt_warning),
                "coupling_mesh_compatible": bool(coupling_mesh_compatible),
                "coupling_flux_loaded": bool(
                    coupling_flux_loaded_successfully
                    if velocity_source == "flow_run"
                    else True
                ),
                "source_flow_ready_for_transport": bool(
                    source_flow_ready_for_transport
                    if velocity_source == "flow_run"
                    else True
                ),
                "pure_gpu_run": bool(summary.get("pure_gpu_run", False)),
                "used_numpy_fallback": bool(summary.get("used_numpy_fallback", True)),
            },
        )
        if not bool(run_completed):
            stage_status["stage_status_reason"] = "transport_run_not_completed"
        elif not bool(numerically_stable):
            stage_status["stage_status_reason"] = "transport_numerical_stability_failed"
        elif bool(regime_blocking_error):
            stage_status["stage_status_reason"] = str(
                regime_error_codes[0]
                if regime_error_codes
                else "transport_regime_data_error"
            )
        elif not bool(coupling_ready):
            stage_status["stage_status_reason"] = "transport_coupling_failed"
        elif bool(transport_dt_blocked):
            stage_status["stage_status_reason"] = str(
                transport_time_estimate.get("dt_control", {}).get(
                    "transport_dt_controller_status", "blocked_transport_substep_cap"
                )
            )
        if bool(regime_warning):
            stage_status["stage_status_warnings"] = regime_warning_codes
        summary["run_completed"] = bool(stage_status.get("run_completed", False))
        summary["numerically_stable"] = bool(
            stage_status.get("numerically_stable", False)
        )
        summary["physically_ready"] = bool(stage_status.get("physically_ready", False))
        summary["ready_for_next_stage"] = bool(
            stage_status.get("ready_for_next_stage", False)
        )
        summary["ready_for_long_run"] = bool(
            stage_status.get("ready_for_long_run", False)
        )
        summary["stage_status_reason"] = str(
            stage_status.get("stage_status_reason", "")
        )
        summary["stage_status_checks"] = dict(
            stage_status.get("stage_status_checks", {})
        )
        summary["stage_status_warnings"] = list(
            stage_status.get("stage_status_warnings", [])
        )
        summary["ready_for_flow_transport_pipeline"] = bool(
            stage_status.get("ready_for_next_stage", False)
        )
        summary["ready_for_server_long_run"] = bool(
            stage_status.get("ready_for_long_run", False)
        )
        if (
            bool(ready_for_long_run)
            and bool(strict_clean)
            and bool(pairwise_conservation_ok)
            and bool(limiter_mass_conservation_ok)
        ):
            summary["long_run_status"] = "success"
        elif (
            bool(ready_for_long_run)
            and bool(tol_clean)
            and bool(pairwise_conservation_ok)
            and bool(limiter_mass_conservation_ok)
        ):
            summary["long_run_status"] = "success_with_tiny_boundedness_tolerance"
        elif bool(run_completed) and bool(numerically_stable):
            summary["long_run_status"] = "warning"
        else:
            summary["long_run_status"] = "failure"
        long_run_reasons: list[str] = []
        if not bool(run_completed):
            long_run_reasons.append("run did not complete requested steps")
        if bool(summary.get("stopped_due_walltime", False)):
            long_run_reasons.append("stopped by walltime")
        if not bool(numerically_stable):
            long_run_reasons.append("numerical stability gate failed")
        if not bool(coupling_ready):
            long_run_reasons.append("flow-transport coupling gate failed")
        if bool(regime_warning):
            long_run_reasons.append(
                str(regime_audit.get("human_readable_reason", "regime warning"))
            )
        if bool(summary.get("used_numpy_fallback", True)):
            long_run_reasons.append("numpy fallback used")
        if not bool(summary.get("pure_gpu_run", False)):
            long_run_reasons.append("not pure gpu run")
        if not long_run_reasons:
            long_run_reasons.append(
                str(stage_status.get("stage_status_reason", "ready"))
            )
        summary["long_run_reason"] = "; ".join(long_run_reasons)
        source_flow_metadata_snapshot = (
            dict(source_flow_payload.get("metadata", {}))
            if velocity_source == "flow_run"
            else {}
        )
        acceptance_report = {
            "run_completed": bool(stage_status.get("run_completed", False)),
            "numerically_stable": bool(stage_status.get("numerically_stable", False)),
            "physically_ready": bool(stage_status.get("physically_ready", False)),
            "ready_for_next_stage": bool(
                stage_status.get("ready_for_next_stage", False)
            ),
            "ready_for_long_run": bool(stage_status.get("ready_for_long_run", False)),
            "stage_status_reason": str(stage_status.get("stage_status_reason", "")),
            "stage_status_warnings": list(
                stage_status.get("stage_status_warnings", [])
            ),
            "clipping_used": bool(
                transport_result.get("clipping", {}).get(
                    "post_step_clipping_used", False
                )
            ),
            "cfl_warning": bool(summary.get("cfl_warning", False)),
            "used_numpy_fallback": bool(summary.get("used_numpy_fallback", True)),
            "pure_gpu_run": bool(summary.get("pure_gpu_run", False)),
            "physical_time_final": float(summary.get("physical_time_final", 0.0)),
            "transport_turnover_count": float(
                summary.get("transport_turnover_count", 0.0)
            ),
            "breakthrough_detected": bool(summary.get("breakthrough_detected", False)),
            "transport_accuracy_supported": bool(regime_supported_accuracy),
            "transport_accuracy_support_status": str(
                regime_audit.get("support_status", "unsupported")
            ),
            "transport_accuracy_support_reason_codes": list(
                regime_audit.get("reason_codes", [])
            ),
            "transport_regime_warning": bool(regime_warning),
            "transport_regime_warning_codes": regime_warning_codes,
            "transport_regime_blocking_error": bool(regime_blocking_error),
            "transport_regime_error_codes": regime_error_codes,
            "transport_regime_audit": regime_audit,
        }
        flow_coupling_metadata_snapshot = {
            "velocity_source": str(velocity_source),
            "source_flow_run_dir": (
                str(source_flow_payload.get("flow_run_dir", ""))
                if velocity_source == "flow_run"
                else ""
            ),
            "source_flow_ready_for_transport": (
                bool(source_flow_ready_for_transport)
                if velocity_source == "flow_run"
                else None
            ),
            "source_flow_metadata": source_flow_metadata_snapshot,
        }
        _write_json(run_dir / "stage_status.json", stage_status)
        _write_json(run_dir / "acceptance_report.json", acceptance_report)
        _write_json(run_dir / "transport_regime_audit.json", regime_audit)
        _write_json(
            run_dir / "flow_coupling_metadata_snapshot.json",
            flow_coupling_metadata_snapshot,
        )
        _write_json(run_dir / "summary.json", summary)
        manifest_recorder.record_completed(
            inputs=manifest_inputs,
            outputs={
                "summary_json": str(run_dir / "summary.json"),
                "config_json": str(run_dir / "config.json"),
                "acceptance_report_json": str(run_dir / "acceptance_report.json"),
                "stage_status_json": str(run_dir / "stage_status.json"),
                "transport_regime_audit_json": str(
                    run_dir / "transport_regime_audit.json"
                ),
                "flow_coupling_metadata_snapshot_json": str(
                    run_dir / "flow_coupling_metadata_snapshot.json"
                ),
                "snapshots_summary_json": str(run_dir / "snapshots_summary.json"),
                "final_concentration_npy": str(run_dir / "final_concentration.npy"),
                "transport_vtu": str(vtu_path),
            },
            artifacts=summary.get("artifacts", {}),
            metadata={
                "mesh_name": str(mesh_stem),
                "run_completed": bool(summary.get("run_completed", False)),
                "ready_for_next_stage": bool(
                    summary.get("ready_for_next_stage", False)
                ),
                "ready_for_long_run": bool(summary.get("ready_for_long_run", False)),
                "stage_status_reason": str(summary.get("stage_status_reason", "")),
                "transport_accuracy_supported": bool(regime_supported_accuracy),
                "transport_accuracy_support_status": str(
                    regime_audit.get("support_status", "unsupported")
                ),
                "transport_regime_warning": bool(regime_warning),
                "transport_regime_blocking_error": bool(regime_blocking_error),
            },
        )

        print(
            "[gmsh-tetra-transport] final stats: "
            f"min={transport_result['final_stats']['min']:.6f}, "
            f"max={transport_result['final_stats']['max']:.6f}, "
            f"mean={transport_result['final_stats']['mean']:.6f}"
        )
        if velocity_diagnostics["boundary_flux_imbalance_warning"]:
            print(
                "[gmsh-tetra-transport] WARNING: boundary flux imbalance ratio > 5%: "
                f"{velocity_diagnostics['boundary_flux_imbalance_ratio']:.4f}"
            )
        print(
            "[gmsh-tetra-transport] flux diagnostics: "
            f"inlet={flux_diag['total_inlet_flux_in']:.6e}, "
            f"outlet={flux_diag['total_outlet_flux_out']:.6e}, "
            f"wall_max_abs={flux_diag['wall_flux_max_abs']:.6e}"
        )
        print(
            "[gmsh-tetra-transport] run mode: "
            f"{summary['run_mode']}, stepping_backend={summary['stepping_backend']}, "
            f"device={summary['device']}, fallback={summary['used_numpy_fallback']}"
        )
        print(f"[gmsh-tetra-transport] summary written: {run_dir / 'summary.json'}")


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
