"""Direct flow stage runner for tetra-native flow diagnostics on imported Gmsh mesh.

Useful for explicit stage launches and debugging, but not a replacement for the
supported manifest-first pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.gmsh._path_utils import (  # noqa: E402
    _normalize_user_path,
    option_was_explicitly_provided,
)
from experiments.gmsh._flow_coupling import sha256_file as coupling_sha256_file  # noqa: E402
from experiments.gmsh._pipeline_manifest import (  # noqa: E402
    add_pipeline_manifest_arguments,
    build_pipeline_manifest_recorder,
)
from microfluidics.path_contract import (  # noqa: E402
    GMSH_IMPORT_RUNS_ROOT_REL,
    GMSH_TETRA_FLOW_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)

DEFAULT_TETRA_FLOW_DEBUG_PRESSURE_SOLVER = "pcg_diag"
STARTUP_BOOTSTRAP_REQUIRED_CONSECUTIVE_ACCEPTED_STEPS = 3
STARTUP_BOOTSTRAP_LEGACY_SEARCH_BUDGET = 20
STARTUP_BOOTSTRAP_QUALIFICATION_TAIL = (
    STARTUP_BOOTSTRAP_REQUIRED_CONSECUTIVE_ACCEPTED_STEPS - 1
)
DEFAULT_STARTUP_BOOTSTRAP_MAX_STEPS = (
    STARTUP_BOOTSTRAP_LEGACY_SEARCH_BUDGET + STARTUP_BOOTSTRAP_QUALIFICATION_TAIL
)
FLOW_RESUME_MANIFEST_SCHEMA_VERSION = 3
FLOW_RESUME_SOLVER_CONTRACT_VERSION = "gmsh-tetra-flow-resume-v1"
FLOW_RESUME_STATE_FILENAMES = (
    "final_pressure.npy",
    "final_cell_velocity.npy",
    "final_corrected_face_flux.npy",
)
_ACTIVE_PIPELINE_MANIFEST_RECORDER = None
_ACTIVE_PIPELINE_MANIFEST_INPUTS: dict[str, Any] | None = None
_ACTIVE_PIPELINE_MANIFEST_ARTIFACTS: dict[str, Any] | None = None
_POSTPROCESSING_MODE = "full"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _assert_no_numpy_fallback(
    *,
    backend_execution: dict[str, Any],
    step_history: list[dict[str, Any]],
) -> None:
    """Require every requested flow stage to have executed on Torch CUDA."""
    if bool(backend_execution.get("used_numpy_fallback", True)):
        raise RuntimeError(
            "fail_if_numpy_fallback triggered: pressure projection used numpy fallback."
        )
    if str(backend_execution.get("selected_backend", "")) != "torch":
        raise RuntimeError(
            "fail_if_numpy_fallback triggered: selected_backend is not 'torch'."
        )
    if not str(backend_execution.get("device", "")).startswith("cuda"):
        raise RuntimeError("fail_if_numpy_fallback triggered: device is not CUDA.")
    if not bool(backend_execution.get("all_core_arrays_on_cuda", False)):
        raise RuntimeError(
            "fail_if_numpy_fallback triggered: core arrays not reported on CUDA."
        )

    stage_contracts = (
        (
            "convection",
            "convective_predictor_used",
            "convective_torch_cuda_used",
            "convective_numpy_fallback_reason",
        ),
        (
            "viscosity",
            "viscous_predictor_used",
            "viscous_torch_cuda_used",
            "viscous_numpy_fallback_reason",
        ),
    )
    for history_index, row in enumerate(step_history, start=1):
        step = int(row.get("step", history_index))
        for stage, requested_key, cuda_key, fallback_key in stage_contracts:
            if not bool(row.get(requested_key, False)):
                continue
            fallback_reason = str(row.get(fallback_key, "")).strip()
            if fallback_reason:
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: "
                    f"{stage} used numpy fallback at step {step}: {fallback_reason}"
                )
            if not bool(row.get(cuda_key, False)):
                raise RuntimeError(
                    "fail_if_numpy_fallback triggered: "
                    f"{stage} did not execute on Torch CUDA at step {step}."
                )


def _pressure_determinism_manifest_artifacts(run_dir: Path) -> dict[str, str]:
    return {
        "run_log": str(run_dir / "run.log"),
        "config_json": str(run_dir / "config.json"),
        "summary_json": str(run_dir / "summary.json"),
        "fixed_work_manifest_json": str(run_dir / "fixed_work_manifest.json"),
        "pressure_determinism_report_json": str(
            run_dir / "pressure_determinism_report.json"
        ),
        "pressure_system_inputs_npz": str(run_dir / "pressure_system_inputs.npz"),
        "cpu_residual_history_json": str(run_dir / "cpu_residual_history.json"),
        "gpu_residual_history_1_json": str(run_dir / "gpu_residual_history_1.json"),
        "gpu_residual_history_2_json": str(run_dir / "gpu_residual_history_2.json"),
        "gpu_residual_history_3_json": str(run_dir / "gpu_residual_history_3.json"),
    }


def _complete_pressure_determinism_run(
    *,
    run_dir: Path,
    manifest_recorder: Any,
    manifest_inputs: dict[str, Any],
    manifest_artifacts: dict[str, str],
    mesh_npz: Path,
    mesh_sha256: str,
    cli_args: dict[str, Any],
    command_line: list[str],
    exec_backend: str,
    exec_device: str,
    determinism_report: dict[str, Any],
    fixed_work_manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    from microfluidics.gmsh.tetra.pressure_determinism_diagnostics import write_json

    outputs = {
        "mesh_npz": str(mesh_npz),
        **{key: value for key, value in manifest_artifacts.items() if key != "run_log"},
    }
    write_json(
        run_dir / "summary.json",
        {
            "run_type": "pressure_determinism_diagnostic",
            "resolved_mesh_npz": str(mesh_npz),
            "mesh_sha256": str(mesh_sha256),
            "fixed_work_manifest": str(run_dir / "fixed_work_manifest.json"),
            "pressure_determinism_report": str(
                run_dir / "pressure_determinism_report.json"
            ),
            "pressure_system_inputs": str(run_dir / "pressure_system_inputs.npz"),
            "cuda_determinism": determinism_report,
            "first_divergent_stage": report.get("first_divergent_stage"),
            "artifacts": dict(manifest_artifacts),
        },
    )
    write_json(
        run_dir / "config.json",
        {
            "cli_args": cli_args,
            "resolved_mesh_npz": str(mesh_npz),
            "mesh_sha256": str(mesh_sha256),
            "flow_backend_selected": exec_backend,
            "flow_device_selected": exec_device,
            "cuda_determinism": determinism_report,
            "fixed_work": fixed_work_manifest,
            "command_line": command_line,
        },
    )
    manifest_recorder.record_completed(
        inputs=manifest_inputs,
        outputs=outputs,
        artifacts=manifest_artifacts,
        metadata={
            "manifest_role": "pressure_determinism_diagnostic",
            "run_type": "pressure_determinism_diagnostic",
            "run_completed": True,
            "ready_for_next_stage": False,
            "ready_for_long_run": False,
            "stage_status_reason": (
                "diagnostic-only run is not a downstream flow source"
            ),
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_npz_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            array = np.ascontiguousarray(np.asarray(archive[key]))
            digest.update(key.encode("utf-8"))
            digest.update(b"\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(
                json.dumps(array.shape, separators=(",", ":")).encode("ascii")
            )
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _seal_flow_resume_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(_json_ready(payload))
    sealed.pop("fingerprint", None)
    canonical = json.dumps(
        sealed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    sealed["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return sealed


def _validated_sha256(value: str, *, label: str, fallback_path: Path) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        return _sha256_file(fallback_path)
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256.")
    return normalized


def _source_runtime_fingerprint(project_root: Path = PROJECT_ROOT) -> str:
    digest = hashlib.sha256()
    source_paths: list[Path] = []
    for relative_root in (
        Path("compute/src/microfluidics"),
        Path("experiments/gmsh"),
    ):
        source_root = project_root / relative_root
        if source_root.is_dir():
            source_paths.extend(source_root.rglob("*.py"))
    for relative_path in (
        Path("pyproject.toml"),
        Path("compute/pyproject.toml"),
        Path("uv.lock"),
    ):
        candidate = project_root / relative_path
        if candidate.is_file():
            source_paths.append(candidate)
    if not source_paths:
        raise ValueError(
            "Cannot fingerprint solver source/runtime: no source files found."
        )
    for path in sorted(
        source_paths, key=lambda item: item.relative_to(project_root).as_posix()
    ):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    runtime_identity = {
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("matplotlib", "numpy", "scipy", "torch")
            if importlib.util.find_spec(package) is not None
        },
    }
    digest.update(
        json.dumps(
            runtime_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _flow_resume_state_descriptor(path: Path) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    return {
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
    }


def _finalize_flow_resume_manifest(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    completed_step: int,
    physical_time: float,
) -> dict[str, Any]:
    if int(completed_step) < 0:
        raise ValueError("Resume flow step must be non-negative.")
    if not np.isfinite(physical_time) or float(physical_time) < 0.0:
        raise ValueError("Resume physical time must be finite and non-negative.")
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    payload["checkpoint"] = {
        "flow_steps_completed_total": int(completed_step),
        "physical_time_final": float(physical_time),
        "artifacts": {
            filename: _flow_resume_state_descriptor(run_dir / filename)
            for filename in FLOW_RESUME_STATE_FILENAMES
        },
    }
    return _seal_flow_resume_manifest(payload)


def _validate_flow_resume_checkpoint(
    recorded_manifest: dict[str, Any],
    *,
    run_dir: Path,
    summary: dict[str, Any],
) -> tuple[int, float]:
    checkpoint = recorded_manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume manifest is missing the sealed checkpoint.")
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        FLOW_RESUME_STATE_FILENAMES
    ):
        raise ValueError("Resume manifest checkpoint artifacts are incomplete.")
    sealed_step = int(checkpoint.get("flow_steps_completed_total", -1))
    sealed_time = float(checkpoint.get("physical_time_final", -1.0))
    summary_step = int(summary.get("flow_steps_completed_total", -1))
    summary_time = float(summary.get("physical_time_final", -1.0))
    if sealed_step < 0 or summary_step != sealed_step:
        raise ValueError("Resume summary completed step does not match the manifest.")
    if not np.isfinite(sealed_time) or sealed_time < 0.0 or summary_time != sealed_time:
        raise ValueError("Resume summary physical time does not match the manifest.")
    for filename in FLOW_RESUME_STATE_FILENAMES:
        if artifacts.get(filename) != _flow_resume_state_descriptor(run_dir / filename):
            raise ValueError(
                f"Resume state artifact {filename} does not match the manifest."
            )
    return sealed_step, sealed_time


def _build_flow_resume_manifest(
    *,
    mesh_npz: Path,
    cfg: Any,
    source_sha256: str,
    runtime_identifier: str,
    input_mesh_sha256: str,
    request_fingerprint: str,
    flow_mode: str,
    flow_dt_mode: str,
    requested_flow_dt: float,
    flow_dt_min: float,
    flow_dt_max: float,
    convective_cfl_target: float,
    wall_strength_ramp_start: float,
    wall_strength_ramp_target: float,
    wall_strength_ramp_steps: int,
) -> dict[str, Any]:
    mesh_sha256 = _mesh_npz_fingerprint(mesh_npz)
    source_digest = (
        _validated_sha256(
            source_sha256,
            label="--run-source-sha256",
            fallback_path=Path(__file__),
        )
        if str(source_sha256).strip()
        else _source_runtime_fingerprint()
    )
    input_mesh_digest = (
        _validated_sha256(
            input_mesh_sha256,
            label="--run-input-mesh-sha256",
            fallback_path=mesh_npz,
        )
        if str(input_mesh_sha256).strip()
        else mesh_sha256
    )
    request_digest = str(request_fingerprint).strip().lower()
    if request_digest and (
        len(request_digest) != 64
        or any(c not in "0123456789abcdef" for c in request_digest)
    ):
        raise ValueError(
            "--resume-request-fingerprint must be a 64-character hexadecimal SHA-256."
        )
    flow_config = dict(_json_ready(asdict(cfg)))
    flow_config.update(
        {
            "flow_mode": str(flow_mode),
            "flow_dt_mode": str(flow_dt_mode),
            "requested_flow_dt": float(requested_flow_dt),
            "flow_dt_min": float(flow_dt_min),
            "flow_dt_max": float(flow_dt_max),
            "convective_cfl_target": float(convective_cfl_target),
            "wall_strength_ramp_start": float(wall_strength_ramp_start),
            "wall_strength_ramp_target": float(wall_strength_ramp_target),
            "wall_strength_ramp_steps": int(wall_strength_ramp_steps),
        }
    )
    return _seal_flow_resume_manifest(
        {
            "schema_version": FLOW_RESUME_MANIFEST_SCHEMA_VERSION,
            "solver_contract_version": FLOW_RESUME_SOLVER_CONTRACT_VERSION,
            "mesh_sha256": mesh_sha256,
            "input_mesh_sha256": input_mesh_digest,
            "source_sha256": source_digest,
            "runtime_identifier": str(runtime_identifier).strip(),
            "request_fingerprint": request_digest,
            "flow_config": flow_config,
        }
    )


def _validate_flow_resume_manifest(
    recorded: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    if int(recorded.get("schema_version", 0)) != FLOW_RESUME_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Resume manifest has an unsupported or missing schema_version."
        )
    if _seal_flow_resume_manifest(recorded).get("fingerprint") != recorded.get(
        "fingerprint"
    ):
        raise ValueError("Resume manifest fingerprint is invalid.")
    for key in (
        "solver_contract_version",
        "mesh_sha256",
        "input_mesh_sha256",
        "source_sha256",
        "runtime_identifier",
        "request_fingerprint",
        "flow_config",
    ):
        if recorded.get(key) != expected.get(key):
            raise ValueError(f"Resume manifest {key} does not match the current run.")


FINAL_FLOW_DIAGNOSTIC_CANONICAL_KEYS = (
    "run_completed",
    "numerically_stable",
    "physically_ready",
    "ready_for_next_stage",
    "ready_for_long_run",
    "stage_status_reason",
    "stage_status_checks",
    "flow_progression_solved",
    "flow_progression_solved_with_startup_tolerance",
    "flow_steps_completed",
    "physical_time_final",
    "used_dt_min",
    "used_dt_mean",
    "used_dt_max",
    "auto_dt_scale_min",
    "auto_dt_scale_mean",
    "auto_dt_scale_max",
    "auto_dt_floor_min",
    "auto_dt_floor_mean",
    "auto_dt_floor_max",
    "auto_dt_min_hit_any",
    "auto_dt_max_hit_any",
    "raw_cfl_after_dt_selection_max",
    "raw_cfl_after_dt_selection_p95",
    "effective_cfl_limit_excess_max",
    "effective_cfl_warning_steps",
    "raw_cfl_after_dt_selection_warning_steps",
    "startup_warning_steps_allowed",
    "startup_warning_steps_observed",
    "nonstartup_failed_steps",
    "flow_progression_acceptance_reason",
    "convective_prototype_accepted",
    "convective_prototype_acceptance_reason",
    "convective_prototype_checks",
    "convective_readiness_checks",
    "readiness_reason",
    "ready_for_long_ns_run_debug",
    "ready_for_long_ns_run_physical",
    "ready_for_long_ns_run",
    "ns_auto_dt_accepted",
    "ns_auto_dt_warning",
    "epsilon_aware_warning_step_count",
    "epsilon_aware_warning_steps",
    "strict_warning_step_count",
    "strict_warning_steps",
    "convective_auto_damping_used_any",
    "convective_substep_cap_hit_any",
    "ns_baseline_physical_clean",
    "ns_baseline_physical_clean_reason",
    "ready_for_flow_to_transport_coupling",
    "ns_baseline_physical_clean_checks",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _ratio(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    den = _safe_float(denominator, 0.0)
    if abs(den) <= 1e-30:
        return float(default)
    return _safe_float(numerator, 0.0) / den


def _timing_stats(values: list[float]) -> dict[str, float]:
    """Return finite aggregate timing statistics without retaining a history."""
    samples = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if samples.size == 0:
        return {
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
        }
    return {
        "mean_seconds": float(np.mean(samples)),
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "min_seconds": float(np.min(samples)),
        "max_seconds": float(np.max(samples)),
    }


def _pressure_iteration_telemetry(history: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = [_safe_float(row.get("pressure_iterations", 0.0)) for row in history]
    stats = _timing_stats(iterations)
    reasons: dict[str, int] = {}
    for row in history:
        reason = str(row.get("pressure_stopping_reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "pressure_iterations_mean": stats["mean_seconds"],
        "pressure_iterations_median": stats["median_seconds"],
        "pressure_iterations_p95": stats["p95_seconds"],
        "pressure_iterations_max": int(stats["max_seconds"]),
        "pressure_iterations_total": int(sum(iterations)),
        "pressure_stopping_reason_counts": reasons,
    }


def _pressure_matvec_telemetry(history: list[dict[str, Any]]) -> dict[str, Any]:
    backend_counts: dict[str, int] = {}
    sparse_csr_steps = 0
    cached_matrix_steps = 0
    fallback_steps = 0
    for row in history:
        backend = str(row.get("pressure_matvec_backend", "") or "unknown")
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        sparse_csr_steps += int(bool(row.get("pressure_matvec_sparse_csr_used", False)))
        cached_matrix_steps += int(
            bool(row.get("pressure_matvec_matrix_cached", False))
        )
        fallback_steps += int(bool(row.get("pressure_matvec_fallback_reason", "")))
    return {
        "pressure_matvec_backend_counts": backend_counts,
        "pressure_matvec_sparse_csr_steps": int(sparse_csr_steps),
        "pressure_matvec_cached_matrix_steps": int(cached_matrix_steps),
        "pressure_matvec_fallback_steps": int(fallback_steps),
    }


def _best_effort_git_metadata() -> dict[str, Any]:
    """Read Git provenance without delaying or failing a simulation."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            ).strip()
        )
        return {"git_commit": commit or None, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def _best_effort_environment_metadata(
    *,
    backend_requested: str,
    backend_selected: str,
    flow_execution_backend_requested: str,
    flow_execution_backend_selected: str,
    flow_execution_device_selected: str,
    backend: Any,
    mesh: Any,
) -> dict[str, Any]:
    """Collect optional host metadata for the actual flow execution path."""
    cuda_version = None
    gpu_name = None
    if str(flow_execution_backend_selected) == "torch" and str(
        flow_execution_device_selected
    ).startswith("cuda"):
        try:
            import torch

            device = torch.device(flow_execution_device_selected)
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            pass
    return {
        "backend": str(flow_execution_backend_selected),
        "device": str(flow_execution_device_selected),
        "backend_requested": str(backend_requested),
        "backend_selected": str(backend_selected),
        "flow_execution_backend_requested": str(flow_execution_backend_requested),
        "flow_execution_backend_selected": str(flow_execution_backend_selected),
        "flow_execution_device_selected": str(flow_execution_device_selected),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "cpu_logical_count": int(os.cpu_count() or 0),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": getattr(backend, "torch_version", None),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "cell_count": int(mesh.tetrahedra.shape[0]),
        "face_count": int(mesh.face_vertices.shape[0]),
        **_best_effort_git_metadata(),
    }


def _synchronize_cuda_if_active(*, backend: str, device: str) -> bool:
    """Synchronize only explicit CUDA timing boundaries, never CPU-only runs."""
    if str(backend) != "torch" or not str(device).startswith("cuda"):
        return False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            return True
    except Exception:
        pass
    return False


def _record_cuda_synchronization(
    telemetry: dict[str, Any],
    *,
    scope: str,
    backend: str,
    device: str,
) -> bool:
    """Record only CUDA barriers that actually completed."""
    synchronized = _synchronize_cuda_if_active(backend=backend, device=device)
    if synchronized:
        telemetry["cuda_active"] = True
        telemetry["cuda_synchronization_count"] += 1
        telemetry[f"{scope}_synchronization"] = True
    return synchronized


def _timing_mode_synchronizes_components(timing_mode: str) -> bool:
    return str(timing_mode) == "detailed"


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(round(_safe_float(seconds))))
    minutes, seconds_part = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds_part}s"
    if minutes:
        return f"{minutes}m {seconds_part}s"
    return f"{seconds_part}s"


def _linear_ramp_value(
    *,
    start: float,
    target: float,
    local_step_idx: int,
    ramp_steps: int,
) -> float:
    steps = int(ramp_steps)
    if steps <= 0:
        return float(target)
    if steps == 1:
        return float(target)
    alpha = (int(local_step_idx) - 1) / float(max(steps - 1, 1))
    alpha = min(max(float(alpha), 0.0), 1.0)
    return float(start + alpha * (target - start))


def _build_stage_status(
    *,
    run_completed: bool,
    numerically_stable: bool,
    physically_ready: bool,
    ready_for_next_stage: bool,
    ready_for_long_run: bool,
    checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    check_map = {str(k): bool(v) for k, v in (checks or {}).items()}
    reason = "stage readiness satisfied"
    if not run_completed:
        reason = "stage did not complete requested steps"
    elif not numerically_stable:
        reason = "stage is not numerically stable"
    elif not physically_ready:
        reason = "stage is numerically stable but not physically ready"
    elif not ready_for_next_stage:
        reason = "stage is physically ready but coupling gate is not satisfied"
    elif not ready_for_long_run:
        reason = "stage passes next-stage gate but not long-run gate"
    return {
        "run_completed": bool(run_completed),
        "numerically_stable": bool(numerically_stable),
        "physically_ready": bool(physically_ready),
        "ready_for_next_stage": bool(ready_for_next_stage),
        "ready_for_long_run": bool(ready_for_long_run),
        "stage_status_reason": str(reason),
        "stage_status_checks": check_map,
    }


def _velocity_scale_from_history(
    *,
    history: list[dict[str, Any]],
    stokes_baseline: dict[str, Any] | None,
    inlet_speed: float,
    fallback: float = 1e-9,
) -> float:
    candidates: list[float] = [abs(float(inlet_speed))]
    if stokes_baseline is not None:
        candidates.append(
            abs(_safe_float(stokes_baseline.get("velocity_magnitude_max_final", 0.0)))
        )
        candidates.append(
            abs(_safe_float(stokes_baseline.get("velocity_magnitude_mean_final", 0.0)))
        )
    for item in history:
        candidates.append(abs(_safe_float(item.get("velocity_magnitude_max", 0.0))))
        candidates.append(abs(_safe_float(item.get("velocity_magnitude_mean", 0.0))))
        candidates.append(abs(_safe_float(item.get("velocity_magnitude_p95", 0.0))))
    candidates = [v for v in candidates if np.isfinite(v)]
    scale = max(candidates) if candidates else float(fallback)
    return max(float(scale), float(fallback))


def _sync_flow_diagnostics_with_final_artifacts(
    flow_diagnostics: dict[str, Any],
    *,
    acceptance_report: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Make flow_diagnostics.json reflect the final acceptance/summary values."""
    synced = dict(flow_diagnostics)
    for key in FINAL_FLOW_DIAGNOSTIC_CANONICAL_KEYS:
        if key in acceptance_report:
            synced[key] = acceptance_report[key]
        elif key in summary:
            synced[key] = summary[key]
    synced["final_artifact_source_of_truth"] = "acceptance_report"
    return synced


def _resolve_convective_predictor_setting(
    *,
    flow_mode: str,
    enable_convective_predictor: bool,
    disable_convective_predictor: bool,
) -> tuple[bool, str]:
    enabled = bool(enable_convective_predictor)
    if flow_mode == "navier_stokes_projection_debug":
        enabled = True
    if bool(disable_convective_predictor):
        enabled = False
    reason = (
        "navier_stokes_projection_debug enables convective predictor by default"
        if (
            flow_mode == "navier_stokes_projection_debug"
            and (not bool(enable_convective_predictor))
            and (not bool(disable_convective_predictor))
        )
        else (
            "convective predictor disabled by CLI flag"
            if bool(disable_convective_predictor)
            else (
                "convective predictor enabled by CLI flag"
                if bool(enable_convective_predictor)
                else "convective predictor disabled for non-navier flow mode"
            )
        )
    )
    return enabled, reason


def _build_expected_artifacts_map(
    run_dir: Path,
    *,
    flow_mode: str,
    flow_steps: int,
    compare_viscous_predictor_modes: bool,
    compare_flow_modes: bool,
    compare_pressure_solvers: bool,
    run_stokes_sensitivity_sweep: bool,
    run_convective_sensitivity_sweep: bool,
    audit_convective_cfl: bool,
    compare_convective_stabilization_modes: bool,
    compare_ns_dt_modes: bool = False,
    postprocessing_mode: str = "full",
) -> dict[str, bool]:
    expected_artifact_paths = [
        run_dir / "flow_progression_history.json",
        run_dir / "resume_manifest.json",
        run_dir / "summary.json",
        run_dir / "acceptance_report.json",
        run_dir / "flow_diagnostics.json",
        run_dir / "viscous_predictor_audit.json",
    ]
    if int(flow_steps) > 1:
        expected_artifact_paths.extend(
            [
                run_dir / "startup_bootstrap_history.json",
                run_dir / "startup_root_cause_report.json",
            ]
        )
    if str(flow_mode) == "navier_stokes_projection_debug":
        expected_artifact_paths.append(run_dir / "navier_stokes_prototype_audit.json")
    if str(postprocessing_mode) == "full" and str(flow_mode) in {
        "stokes_viscous_projection",
        "navier_stokes_projection_debug",
    }:
        expected_artifact_paths.extend(
            [
                run_dir / "divergence_stage_comparison_step_0100.png",
                run_dir / "velocity_magnitude_before_after_predictor_step_0100.png",
                run_dir / "face_flux_delta_predictor_xy_step_0100.png",
            ]
        )
        if int(flow_steps) >= 100:
            expected_artifact_paths.extend(
                [
                    run_dir / "velocity_vectors_xy_grid_binned_step_0100.png",
                    run_dir / "velocity_magnitude_p95_clipped_xy_step_0100.png",
                ]
            )
    if bool(compare_viscous_predictor_modes):
        expected_artifact_paths.append(
            run_dir / "viscous_predictor_mode_comparison.json"
        )
    if bool(compare_flow_modes):
        expected_artifact_paths.append(run_dir / "flow_mode_comparison.json")
    if bool(compare_pressure_solvers):
        expected_artifact_paths.append(run_dir / "pressure_solver_comparison.json")
    if bool(run_stokes_sensitivity_sweep):
        expected_artifact_paths.extend(
            [
                run_dir / "stokes_baseline_sensitivity_sweep.json",
                run_dir / "stokes_baseline_sensitivity_sweep.csv",
            ]
        )
    if bool(run_convective_sensitivity_sweep):
        expected_artifact_paths.extend(
            [
                run_dir / "convective_sensitivity_sweep.json",
                run_dir / "convective_sensitivity_sweep.csv",
            ]
        )
    if bool(audit_convective_cfl):
        expected_artifact_paths.extend(
            [
                run_dir / "top_convective_cfl_cells.json",
                run_dir / "top_convective_cfl_faces.json",
                run_dir / "convective_cfl_definition_report.json",
            ]
        )
    if bool(compare_convective_stabilization_modes):
        expected_artifact_paths.append(
            run_dir / "convective_stabilization_comparison.json"
        )
    if bool(compare_ns_dt_modes):
        expected_artifact_paths.append(run_dir / "ns_dt_mode_comparison.json")
    expected_artifacts = {str(p): bool(p.exists()) for p in expected_artifact_paths}
    expected_artifacts[str(run_dir / "resume_manifest.json")] = True
    expected_artifacts[str(run_dir / "summary.json")] = True
    return expected_artifacts


def _projection_boundary_contract_runtime_payload(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "projection_pinned_face_constraints": dict(
            diagnostics.get("projection_pinned_face_constraints", {})
        ),
        "projection_correction_boundary_contract": dict(
            diagnostics.get("projection_correction_boundary_contract", {})
        ),
        "projection_correction_stage_codebook": dict(
            diagnostics.get("projection_correction_stage_codebook", {})
        ),
        "face_flux_primary_stage_codebook": dict(
            diagnostics.get("face_flux_primary_stage_codebook", {})
        ),
    }


def _vector_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "l2": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "l2": float(np.sqrt(np.mean(arr * arr))),
        "max_abs": float(np.max(np.abs(arr))),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def _save_scatter(
    *,
    centers: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    out_path: Path,
    cmap: str = "viridis",
    log10_abs: bool = False,
) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    vals = np.asarray(values, dtype=np.float64)
    if log10_abs:
        vals = np.log10(np.maximum(np.abs(vals), 1e-20))
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=vals,
        s=2.0,
        cmap=cmap,
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_normalized(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    max_arrows: int = 3000,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    n = c.shape[0]
    step = max(1, n // max_arrows)
    idx = np.arange(0, n, step, dtype=np.int64)
    vx = v[idx, 0]
    vy = v[idx, 1]
    mag = np.sqrt(vx * vx + vy * vy)
    scale = np.maximum(mag, 1e-20)
    ux = vx / scale
    uy = vy / scale

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        c[idx, 0],
        c[idx, 1],
        ux,
        uy,
        angles="xy",
        scale_units="xy",
        scale=30.0,
        width=0.0012,
        alpha=0.9,
        color="#004c78",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (normalized)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_raw_clipped_scale(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    max_arrows: int = 3000,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    n = c.shape[0]
    step = max(1, n // max_arrows)
    idx = np.arange(0, n, step, dtype=np.int64)
    vx = np.asarray(v[idx, 0], dtype=np.float64)
    vy = np.asarray(v[idx, 1], dtype=np.float64)
    mag = np.sqrt(vx * vx + vy * vy)
    clip = float(np.percentile(mag, 95.0)) if mag.size else 1.0
    clip = max(clip, 1e-12)
    scale = np.minimum(1.0, clip / np.maximum(mag, 1e-20))
    vx_clip = vx * scale
    vy_clip = vy * scale

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        c[idx, 0],
        c[idx, 1],
        vx_clip,
        vy_clip,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0012,
        alpha=0.9,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (raw, clipped scale)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_downsampled(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    max_arrows: int = 3000,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    n = c.shape[0]
    step = max(1, n // max_arrows)
    idx = np.arange(0, n, step, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        c[idx, 0],
        c[idx, 1],
        v[idx, 0],
        v[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0012,
        alpha=0.9,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (downsampled)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_sparse_normalized(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    max_arrows: int = 1000,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    n = c.shape[0]
    if n <= max_arrows:
        idx = np.arange(n, dtype=np.int64)
    else:
        idx = np.linspace(0, n - 1, max_arrows, dtype=np.int64)
    vx = v[idx, 0]
    vy = v[idx, 1]
    mag = np.sqrt(vx * vx + vy * vy)
    ux = vx / np.maximum(mag, 1e-20)
    uy = vy / np.maximum(mag, 1e-20)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        c[idx, 0],
        c[idx, 1],
        ux,
        uy,
        angles="xy",
        scale_units="xy",
        scale=30.0,
        width=0.0012,
        alpha=0.9,
        color="#005f73",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (sparse normalized)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_by_region(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    masks: dict[str, np.ndarray],
    out_path: Path,
    max_per_region: int = 350,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    palette = {
        "inlet_adjacent": "#1d4ed8",
        "junction_zone": "#16a34a",
        "outlet_adjacent": "#dc2626",
    }
    for region, color in palette.items():
        mask = np.asarray(
            masks.get(region, np.zeros((c.shape[0],), dtype=bool)), dtype=bool
        )
        ids = np.flatnonzero(mask)
        if ids.size == 0:
            continue
        if ids.size > max_per_region:
            ids = ids[np.linspace(0, ids.size - 1, max_per_region, dtype=np.int64)]
        vx = v[ids, 0]
        vy = v[ids, 1]
        mag = np.sqrt(vx * vx + vy * vy)
        ux = vx / np.maximum(mag, 1e-20)
        uy = vy / np.maximum(mag, 1e-20)
        ax.quiver(
            c[ids, 0],
            c[ids, 1],
            ux,
            uy,
            angles="xy",
            scale_units="xy",
            scale=30.0,
            width=0.0015,
            alpha=0.9,
            color=color,
            label=region,
        )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors by Region (normalized)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_outlet_flux_faces(
    *,
    mesh,
    corrected_flux: np.ndarray,
    out_path: Path,
) -> str:
    outlet = np.asarray(mesh.outlet_faces, dtype=np.int64)
    if outlet.size == 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.text(0.5, 0.5, "No outlet faces", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return str(out_path)
    centers = np.asarray(mesh.face_centers[outlet], dtype=np.float64)
    vals = np.asarray(corrected_flux[outlet], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        centers[:, 0], centers[:, 1], c=vals, s=8.0, cmap="coolwarm", linewidths=0
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Outlet Face Flux (corrected)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="q_out [m^3/s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_pressure_outlet_zoom(
    *,
    mesh,
    pressure: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(mesh.cell_centers, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    outlet = np.asarray(mesh.outlet_faces, dtype=np.int64)
    if outlet.size == 0:
        return _save_scatter(
            centers=c,
            values=p,
            title="Pressure XY",
            label="p",
            out_path=out_path,
            cmap="plasma",
        )
    y_out = float(np.median(mesh.face_centers[outlet, 1]))
    y_span = max(float(np.max(c[:, 1]) - np.min(c[:, 1])), 1e-12)
    mask = c[:, 1] >= y_out - 0.2 * y_span
    if not np.any(mask):
        mask = np.ones((c.shape[0],), dtype=bool)
    return _save_scatter(
        centers=c[mask],
        values=p[mask],
        title="Pressure Outlet Zoom XY",
        label="p",
        out_path=out_path,
        cmap="plasma",
    )


def _save_divergence_hotspots(
    *,
    mesh,
    top_cells: list[dict[str, Any]],
    out_path: Path,
) -> str:
    c = np.asarray(mesh.cell_centers, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(c[:, 0], c[:, 1], s=0.6, c="#d1d5db", alpha=0.4, linewidths=0)
    if top_cells:
        ids = np.asarray([int(row["cell_index"]) for row in top_cells], dtype=np.int64)
        vals = np.asarray(
            [abs(float(row["div_corrected"])) for row in top_cells], dtype=np.float64
        )
        sc = ax.scatter(
            c[ids, 0],
            c[ids, 1],
            c=np.log10(np.maximum(vals, 1e-20)),
            s=16.0,
            cmap="inferno",
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, label="log10(|div_corrected|)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top Divergence Hotspot Cells")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_divergence_before_after_same_scale(
    *,
    centers: np.ndarray,
    div_before: np.ndarray,
    div_after: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    b = np.asarray(div_before, dtype=np.float64)
    a = np.asarray(div_after, dtype=np.float64)
    v = np.log10(np.maximum(np.concatenate((np.abs(b), np.abs(a))), 1e-20))
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axes[0].scatter(
        c[:, 0],
        c[:, 1],
        c=np.log10(np.maximum(np.abs(b), 1e-20)),
        s=2.0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    axes[0].set_title("Before: log10(|div|)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    sc1 = axes[1].scatter(
        c[:, 0],
        c[:, 1],
        c=np.log10(np.maximum(np.abs(a), 1e-20)),
        s=2.0,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    axes[1].set_title("After: log10(|div|)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist())
    cbar.set_label("log10|div|")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_correction_flux_hotspots_xy(
    *,
    mesh,
    correction_flux: np.ndarray,
    out_path: Path,
    max_faces: int = 1500,
) -> str:
    q = np.asarray(correction_flux, dtype=np.float64)
    centers = np.asarray(mesh.face_centers, dtype=np.float64)
    mag = np.abs(q)
    if q.size > max_faces:
        ids = np.argsort(-mag)[:max_faces]
    else:
        ids = np.arange(q.size, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        centers[ids, 0],
        centers[ids, 1],
        c=np.log10(np.maximum(mag[ids], 1e-20)),
        s=8.0,
        cmap="magma",
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Correction Flux Hotspots XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="log10(|correction_flux|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_pressure_correction_flux_magnitude_xy(
    *,
    mesh,
    pressure_gradient_flux: np.ndarray,
    out_path: Path,
) -> str:
    centers = np.asarray(mesh.face_centers, dtype=np.float64)
    mag = np.abs(np.asarray(pressure_gradient_flux, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=np.log10(np.maximum(mag, 1e-20)),
        s=2.0,
        cmap="cividis",
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Pressure Correction Flux Magnitude XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="log10(|grad-pressure flux|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_top_divergence_correction_breakdown_xy(
    *,
    mesh,
    breakdown: dict[str, Any],
    out_path: Path,
) -> str:
    c = np.asarray(mesh.cell_centers, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(c[:, 0], c[:, 1], s=0.6, c="#d1d5db", alpha=0.35, linewidths=0)
    cells = list(breakdown.get("cells", []))
    if cells:
        ids = np.asarray(
            [
                int(row.get("cell_index", -1))
                for row in cells
                if int(row.get("cell_index", -1)) >= 0
            ],
            dtype=np.int64,
        )
        vals = np.asarray(
            [
                abs(float(row.get("div_corrected", 0.0)))
                for row in cells
                if int(row.get("cell_index", -1)) >= 0
            ],
            dtype=np.float64,
        )
        if ids.size:
            sc = ax.scatter(
                c[ids, 0],
                c[ids, 1],
                c=np.log10(np.maximum(vals, 1e-20)),
                s=18.0,
                cmap="inferno",
                linewidths=0,
            )
            fig.colorbar(sc, ax=ax, label="log10(|div_corrected|)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top Divergence Correction Breakdown XY")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_boundary_policy_comparison_bar(
    *,
    comparison: dict[str, Any],
    out_path: Path,
) -> str:
    items = list(comparison.get("policies", []))
    labels = [str(it.get("policy", "")) for it in items]
    div_l2_ratio = [float(it.get("div_l2_ratio", 0.0)) for it in items]
    net_flux = [float(abs(it.get("net_boundary_flux_after", 0.0))) for it in items]
    x = np.arange(len(labels), dtype=np.float64)
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x - w / 2, div_l2_ratio, width=w, label="div_l2_ratio")
    ax.bar(x + w / 2, net_flux, width=w, label="|net_boundary_flux_after|")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title("Boundary Policy Comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_top_divergence_cells_before_after_limiter_xy(
    *,
    mesh,
    before_cells: list[dict[str, Any]],
    after_cells: list[dict[str, Any]],
    out_path: Path,
) -> str:
    c = np.asarray(mesh.cell_centers, dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axes[0].scatter(c[:, 0], c[:, 1], s=0.5, c="#d1d5db", alpha=0.3, linewidths=0)
    if before_cells:
        ids0 = np.asarray(
            [int(r.get("cell_index", -1)) for r in before_cells], dtype=np.int64
        )
        ids0 = ids0[ids0 >= 0]
        if ids0.size:
            axes[0].scatter(
                c[ids0, 0], c[ids0, 1], s=14.0, c="#b91c1c", alpha=0.9, linewidths=0
            )
    axes[0].set_title("Top Divergence Cells: Before Limiter")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(c[:, 0], c[:, 1], s=0.5, c="#d1d5db", alpha=0.3, linewidths=0)
    if after_cells:
        ids1 = np.asarray(
            [int(r.get("cell_index", -1)) for r in after_cells], dtype=np.int64
        )
        ids1 = ids1[ids1 >= 0]
        if ids1.size:
            axes[1].scatter(
                c[ids1, 0], c[ids1, 1], s=14.0, c="#0f766e", alpha=0.9, linewidths=0
            )
    axes[1].set_title("Top Divergence Cells: After Limiter")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_aspect("equal", adjustable="box")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_correction_flux_limited_faces_xy(
    *,
    mesh,
    correction_before: np.ndarray,
    correction_after: np.ndarray,
    limited_face_indices: np.ndarray,
    out_path: Path,
) -> str:
    centers = np.asarray(mesh.face_centers, dtype=np.float64)
    cb = np.asarray(correction_before, dtype=np.float64)
    ca = np.asarray(correction_after, dtype=np.float64)
    ids = np.asarray(limited_face_indices, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.scatter(
        centers[:, 0], centers[:, 1], s=0.5, c="#d1d5db", alpha=0.3, linewidths=0
    )
    if ids.size:
        delta = np.abs(ca[ids] - cb[ids])
        sc = ax.scatter(
            centers[ids, 0],
            centers[ids, 1],
            c=np.log10(np.maximum(delta, 1e-20)),
            s=10.0,
            cmap="magma",
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, label="log10(|delta correction|)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Correction Flux Limited Faces XY")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_limiter_effect_histogram(
    *,
    div_before: np.ndarray,
    div_after: np.ndarray,
    out_path: Path,
) -> str:
    b = np.abs(np.asarray(div_before, dtype=np.float64))
    a = np.abs(np.asarray(div_after, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(
        np.log10(np.maximum(b, 1e-20)),
        bins=90,
        alpha=0.55,
        color="#b91c1c",
        label="before limiter",
    )
    ax.hist(
        np.log10(np.maximum(a, 1e-20)),
        bins=90,
        alpha=0.55,
        color="#0f766e",
        label="after limiter",
    )
    ax.set_xlabel("log10(|div|)")
    ax.set_ylabel("cell count")
    ax.set_title("Limiter Effect Histogram")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_pressure_operator_symmetry_hotspots_xy(
    *,
    centers: np.ndarray,
    row_symmetry_error: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    e = np.asarray(row_symmetry_error, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        c[:, 0],
        c[:, 1],
        c=np.log10(np.maximum(np.abs(e), 1e-30)),
        s=2.0,
        cmap="inferno",
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Pressure Operator Symmetry Hotspots XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="log10(row symmetry error)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_pressure_operator_matrixfree_mismatch_xy(
    *,
    centers: np.ndarray,
    mismatch: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    m = np.asarray(mismatch, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        c[:, 0],
        c[:, 1],
        c=np.log10(np.maximum(np.abs(m), 1e-30)),
        s=2.0,
        cmap="cividis",
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Matrix-free vs Explicit Mismatch XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="log10(|A_mf p - A_exp p|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_explicit_vs_matrixfree_residual_histogram(
    *,
    mismatch: np.ndarray,
    out_path: Path,
) -> str:
    m = np.abs(np.asarray(mismatch, dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(np.log10(np.maximum(m, 1e-30)), bins=90, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("log10(|A_mf p - A_exp p|)")
    ax.set_ylabel("cell count")
    ax.set_title("Explicit vs Matrix-free Residual Histogram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _export_vtu(
    *,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    pressure: np.ndarray,
    divergence: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
) -> str:
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError:
        return ""
    vel_mag = np.linalg.norm(velocity, axis=1)
    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", tetrahedra)],
        cell_data={
            "pressure": [pressure],
            "divergence": [divergence],
            "velocity_magnitude": [vel_mag],
            "velocity": [velocity],
        },
    )
    mesh.write(out_path)
    return str(out_path)


def _cell_zone_masks(mesh) -> dict[str, np.ndarray]:
    n_cells = int(mesh.tetrahedra.shape[0])
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    inlet = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)

    masks = {
        "boundary_adjacent": np.zeros((n_cells,), dtype=bool),
        "inlet_adjacent": np.zeros((n_cells,), dtype=bool),
        "outlet_adjacent": np.zeros((n_cells,), dtype=bool),
        "wall_adjacent": np.zeros((n_cells,), dtype=bool),
    }
    if boundary.size:
        masks["boundary_adjacent"][np.unique(c0[boundary])] = True
    if inlet.size:
        masks["inlet_adjacent"][np.unique(c0[inlet])] = True
    if outlet.size:
        masks["outlet_adjacent"][np.unique(c0[outlet])] = True
    if wall.size:
        masks["wall_adjacent"][np.unique(c0[wall])] = True
    masks["interior_core"] = ~masks["boundary_adjacent"]

    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    y_in = (
        float(np.median(mesh.face_centers[inlet, 1]))
        if inlet.size
        else float(np.min(y))
    )
    y_out = (
        float(np.median(mesh.face_centers[outlet, 1]))
        if outlet.size
        else float(np.max(y))
    )
    y_span = max(y_out - y_in, 1e-12)
    junction_y = y_in + 0.12 * y_span
    x_abs = np.abs(x)
    x_thresh = float(np.percentile(x_abs, 35.0)) if x_abs.size else 0.0
    junction = (np.abs(y - junction_y) <= 0.12 * y_span) & (
        x_abs <= max(x_thresh, 1e-12)
    )
    if not np.any(junction):
        junction = np.abs(y - junction_y) <= 0.16 * y_span
    masks["junction_zone"] = junction
    return masks


def _build_velocity_region_masks(
    mesh, zone_masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    x = centers[:, 0]
    y = centers[:, 1]
    n_cells = centers.shape[0]
    inlet_faces = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet_faces = np.asarray(mesh.outlet_faces, dtype=np.int64)
    y_in = (
        float(np.median(mesh.face_centers[inlet_faces, 1]))
        if inlet_faces.size
        else float(np.min(y))
    )
    y_out = (
        float(np.median(mesh.face_centers[outlet_faces, 1]))
        if outlet_faces.size
        else float(np.max(y))
    )
    y_span = max(y_out - y_in, 1e-12)
    junction_y = y_in + 0.12 * y_span
    x_abs = np.abs(x)
    x_thr = float(np.percentile(x_abs, 45.0)) if x_abs.size else 0.0
    x_thr = max(0.45 * x_thr, 1e-12)
    near_inlet_y = y <= (junction_y + 0.20 * y_span)
    left_branch = near_inlet_y & (x < -x_thr)
    right_branch = near_inlet_y & (x > x_thr)
    outlet_branch = y >= (junction_y + 0.10 * y_span)
    junction_mask = np.asarray(
        zone_masks.get("junction_zone", np.zeros((n_cells,), dtype=bool)), dtype=bool
    )
    if not np.any(junction_mask):
        junction_mask = (np.abs(y - junction_y) <= 0.18 * y_span) & (
            np.abs(x) <= max(1.5 * x_thr, 1e-12)
        )
    out = {
        "left_inlet_branch": left_branch,
        "right_inlet_branch": right_branch,
        "junction": junction_mask,
        "outlet_branch": outlet_branch,
        "boundary_adjacent": np.asarray(
            zone_masks.get("boundary_adjacent", np.zeros((n_cells,), dtype=bool)),
            dtype=bool,
        ),
        "interior_core": np.asarray(
            zone_masks.get("interior_core", np.ones((n_cells,), dtype=bool)), dtype=bool
        ),
    }
    return out


def _binned_velocity_xy(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    bins_x: int = 42,
    bins_y: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if c.size == 0:
        z = np.zeros((0,), dtype=np.float64)
        iz = np.zeros((0,), dtype=np.int64)
        return z, z, z, z, iz
    x = c[:, 0]
    y = c[:, 1]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmax <= xmin:
        xmax = xmin + 1e-12
    if ymax <= ymin:
        ymax = ymin + 1e-12
    bx = max(int(bins_x), 2)
    by = max(int(bins_y), 2)
    ex = np.linspace(xmin, xmax, bx + 1)
    ey = np.linspace(ymin, ymax, by + 1)
    ix = np.clip(np.searchsorted(ex, x, side="right") - 1, 0, bx - 1)
    iy = np.clip(np.searchsorted(ey, y, side="right") - 1, 0, by - 1)
    bid = iy * bx + ix
    n_bins = bx * by
    cnt = np.bincount(bid, minlength=n_bins).astype(np.int64)
    sum_x = np.bincount(bid, weights=x, minlength=n_bins)
    sum_y = np.bincount(bid, weights=y, minlength=n_bins)
    sum_u = np.bincount(bid, weights=v[:, 0], minlength=n_bins)
    sum_v = np.bincount(bid, weights=v[:, 1], minlength=n_bins)
    nz = cnt > 0
    xc = sum_x[nz] / np.maximum(cnt[nz], 1)
    yc = sum_y[nz] / np.maximum(cnt[nz], 1)
    uc = sum_u[nz] / np.maximum(cnt[nz], 1)
    vc = sum_v[nz] / np.maximum(cnt[nz], 1)
    return xc, yc, uc, vc, cnt[nz]


def _save_velocity_vectors_xy_grid_binned(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    bins_x: int = 42,
    bins_y: int = 42,
) -> str:
    xc, yc, uc, vc, _ = _binned_velocity_xy(
        centers=centers, velocity=velocity, bins_x=bins_x, bins_y=bins_y
    )
    mag = np.sqrt(uc * uc + vc * vc)
    clip = max(float(np.percentile(mag, 95.0)) if mag.size else 1.0, 1e-12)
    scl = np.minimum(1.0, clip / np.maximum(mag, 1e-20))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        xc,
        yc,
        uc * scl,
        vc * scl,
        mag,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        alpha=0.92,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (grid-binned)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_velocity_vectors_xy_region_panels_binned(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    region_masks: dict[str, np.ndarray],
    out_path: Path,
    bins_x: int = 26,
    bins_y: int = 26,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    panels = [
        ("left inlet", "left_inlet_branch"),
        ("right inlet", "right_inlet_branch"),
        ("junction", "junction"),
        ("outlet", "outlet_branch"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    for ax, (title, key) in zip(axes.flat, panels):
        mask = np.asarray(
            region_masks.get(key, np.zeros((c.shape[0],), dtype=bool)), dtype=bool
        )
        ids = np.flatnonzero(mask)
        if ids.size == 0:
            ax.set_title(f"{title} (no cells)")
            ax.axis("off")
            continue
        xc, yc, uc, vc, _ = _binned_velocity_xy(
            centers=c[ids],
            velocity=v[ids],
            bins_x=bins_x,
            bins_y=bins_y,
        )
        mag = np.sqrt(uc * uc + vc * vc)
        clip = max(float(np.percentile(mag, 95.0)) if mag.size else 1.0, 1e-12)
        scl = np.minimum(1.0, clip / np.maximum(mag, 1e-20))
        ax.quiver(
            xc,
            yc,
            uc * scl,
            vc * scl,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="#1f2937",
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_velocity_vectors_xy_direction_colored(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    bins_x: int = 42,
    bins_y: int = 42,
) -> str:
    xc, yc, uc, vc, _ = _binned_velocity_xy(
        centers=centers, velocity=velocity, bins_x=bins_x, bins_y=bins_y
    )
    ang = np.arctan2(vc, uc)
    mag = np.sqrt(uc * uc + vc * vc)
    ux = uc / np.maximum(mag, 1e-20)
    uy = vc / np.maximum(mag, 1e-20)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    q = ax.quiver(
        xc,
        yc,
        ux,
        uy,
        ang,
        cmap="twilight",
        angles="xy",
        scale_units="xy",
        scale=24.0,
        width=0.002,
        alpha=0.94,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Direction XY (binned, colored)")
    ax.set_aspect("equal", adjustable="box")
    cb = fig.colorbar(q, ax=ax)
    cb.set_label("direction angle atan2(vy,vx) [rad]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_velocity_magnitude_p95_clipped_xy(
    *,
    centers: np.ndarray,
    speed: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    s = np.asarray(speed, dtype=np.float64)
    clip = max(float(np.percentile(s, 95.0)) if s.size else 1.0, 1e-20)
    s_clip = np.minimum(s, clip)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(c[:, 0], c[:, 1], c=s_clip, s=2.0, cmap="magma", linewidths=0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Magnitude XY (p95 clipped)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="|u| clipped")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_divergence_stage_comparison(
    *,
    centers: np.ndarray,
    div_before: np.ndarray,
    div_after_predictor: np.ndarray,
    div_after_contract: np.ndarray,
    div_after_projection: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    panels = [
        ("before predictor", np.asarray(div_before, dtype=np.float64)),
        ("after predictor", np.asarray(div_after_predictor, dtype=np.float64)),
        ("after boundary contract", np.asarray(div_after_contract, dtype=np.float64)),
        ("after projection", np.asarray(div_after_projection, dtype=np.float64)),
    ]
    vmax = 0.0
    for _, vals in panels:
        vmax = max(vmax, float(np.max(np.abs(vals))) if vals.size else 0.0)
    vmax = max(vmax, 1e-20)
    for ax, (title, vals) in zip(axes.flat, panels):
        sc = ax.scatter(
            c[:, 0],
            c[:, 1],
            c=vals,
            s=2.0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_velocity_magnitude_before_after_predictor(
    *,
    centers: np.ndarray,
    vel_before: np.ndarray,
    vel_after_predictor: np.ndarray,
    out_path: Path,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    vb = np.linalg.norm(np.asarray(vel_before, dtype=np.float64), axis=1)
    va = np.linalg.norm(np.asarray(vel_after_predictor, dtype=np.float64), axis=1)
    vmax = max(
        float(np.percentile(vb, 99.0)) if vb.size else 0.0,
        float(np.percentile(va, 99.0)) if va.size else 0.0,
    )
    vmax = max(vmax, 1e-20)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for ax, vals, title in (
        (axes[0], vb, "before predictor"),
        (axes[1], va, "after predictor"),
    ):
        sc = ax.scatter(
            c[:, 0],
            c[:, 1],
            c=np.minimum(vals, vmax),
            s=2.0,
            cmap="magma",
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_face_flux_delta_predictor_xy(
    *,
    mesh,
    face_flux_before: np.ndarray,
    face_flux_after_predictor: np.ndarray,
    out_path: Path,
) -> str:
    fc = np.asarray(mesh.face_centers, dtype=np.float64)
    dq = np.asarray(face_flux_after_predictor, dtype=np.float64) - np.asarray(
        face_flux_before, dtype=np.float64
    )
    vmax = max(float(np.percentile(np.abs(dq), 99.0)) if dq.size else 0.0, 1e-20)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        fc[:, 0],
        fc[:, 1],
        c=dq,
        s=2.0,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Face Flux Delta Predictor XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax, label="dq predictor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _top_velocity_cells(
    *,
    mesh,
    velocity: np.ndarray,
    top_k: int = 30,
) -> dict[str, Any]:
    v = np.asarray(velocity, dtype=np.float64)
    mag = np.linalg.norm(v, axis=1)
    order = np.argsort(-mag)
    n_take = min(top_k, int(order.size))
    zone_masks = _cell_zone_masks(mesh)
    zone_names = list(zone_masks.keys())
    records: list[dict[str, Any]] = []
    for idx in order[:n_take].tolist():
        labels = [name for name in zone_names if bool(zone_masks[name][idx])]
        records.append(
            {
                "cell_index": int(idx),
                "center": np.asarray(mesh.cell_centers[idx], dtype=np.float64).tolist(),
                "velocity": v[idx].tolist(),
                "magnitude": float(mag[idx]),
                "region_labels": labels,
            }
        )
    return {
        "top_k": int(n_take),
        "max_velocity_magnitude": float(mag[order[0]]) if order.size else 0.0,
        "records": records,
    }


def _save_vectors_sparse_normalized_seeded(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    out_path: Path,
    max_arrows: int = 300,
    seed: int = 1234,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    n = c.shape[0]
    if n <= max_arrows:
        idx = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_arrows, replace=False).astype(np.int64))
    vx = v[idx, 0]
    vy = v[idx, 1]
    mag = np.sqrt(vx * vx + vy * vy)
    ux = vx / np.maximum(mag, 1e-20)
    uy = vy / np.maximum(mag, 1e-20)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.quiver(
        c[idx, 0],
        c[idx, 1],
        ux,
        uy,
        angles="xy",
        scale_units="xy",
        scale=20.0,
        width=0.0015,
        alpha=0.9,
        color="#0f766e",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Velocity Vectors XY (sparse normalized, seeded)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _save_vectors_region_panels(
    *,
    mesh,
    centers: np.ndarray,
    velocity: np.ndarray,
    masks: dict[str, np.ndarray],
    out_path: Path,
    max_per_panel: int = 260,
) -> str:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    panel_specs = [
        (
            "left inlet",
            np.asarray(
                masks.get("inlet_adjacent", np.zeros((c.shape[0],), dtype=bool)),
                dtype=bool,
            )
            & (c[:, 0] < 0.0),
        ),
        (
            "right inlet",
            np.asarray(
                masks.get("inlet_adjacent", np.zeros((c.shape[0],), dtype=bool)),
                dtype=bool,
            )
            & (c[:, 0] >= 0.0),
        ),
        (
            "junction",
            np.asarray(
                masks.get("junction_zone", np.zeros((c.shape[0],), dtype=bool)),
                dtype=bool,
            ),
        ),
        (
            "outlet",
            np.asarray(
                masks.get("outlet_adjacent", np.zeros((c.shape[0],), dtype=bool)),
                dtype=bool,
            ),
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for ax, (title, mask) in zip(axes.flat, panel_specs):
        ids = np.flatnonzero(mask)
        if ids.size == 0:
            ax.set_title(f"{title} (no cells)")
            ax.axis("off")
            continue
        if ids.size > max_per_panel:
            ids = ids[np.linspace(0, ids.size - 1, max_per_panel, dtype=np.int64)]
        vx = v[ids, 0]
        vy = v[ids, 1]
        mag = np.sqrt(vx * vx + vy * vy)
        ux = vx / np.maximum(mag, 1e-20)
        uy = vy / np.maximum(mag, 1e-20)
        ax.quiver(
            c[ids, 0],
            c[ids, 1],
            ux,
            uy,
            angles="xy",
            scale_units="xy",
            scale=20.0,
            width=0.002,
            color="#1f2937",
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _face_group_codes(mesh) -> np.ndarray:
    codes = np.zeros((mesh.face_vertices.shape[0],), dtype=np.int32)
    codes[np.asarray(mesh.wall_faces, dtype=np.int64)] = 1
    codes[np.asarray(mesh.outlet_faces, dtype=np.int64)] = 2
    inlet = np.asarray(mesh.inlet_faces, dtype=np.int64)
    if inlet.size:
        codes[inlet] = 3
    return codes


def _export_flow_coupling_bundle(
    *,
    run_dir: Path,
    mesh,
    mesh_name: str,
    mesh_sha256: str,
    face_flux: np.ndarray,
    cell_velocity: np.ndarray,
    pressure: np.ndarray,
    face_group_codes: np.ndarray,
    flow_mode: str,
    flow_dt_mode: str,
    flow_steps: int,
    physical_time_final: float,
    run_completed: bool,
    numerically_stable: bool,
    physically_ready: bool,
    ready_for_next_stage: bool,
    ready_for_long_run: bool,
    stage_status_reason: str,
    ready_for_flow_to_transport_coupling: bool,
    ns_baseline_physical_clean: bool,
    outlet_flux_rescale_used: bool,
    nonphysical_flux_fix_used: bool,
    convective_auto_damping_used_any: bool,
    wall_flux_max_abs_after: float,
    outlet_inlet_flux_ratio: float,
    final_div_l2: float,
    final_div_max: float,
) -> dict[str, Any]:
    final_flux_path = run_dir / "final_corrected_face_flux.npy"
    final_cell_velocity_path = run_dir / "final_cell_velocity.npy"
    final_pressure_path = run_dir / "final_pressure.npy"
    face_centers_path = run_dir / "face_centers.npy"
    face_normals_path = run_dir / "face_normals.npy"
    face_to_cells_path = run_dir / "face_to_cells.npy"
    cell_centers_path = run_dir / "cell_centers.npy"
    cell_volumes_path = run_dir / "cell_volumes.npy"
    face_groups_path = run_dir / "face_groups.npy"
    metadata_path = run_dir / "flow_coupling_metadata.json"

    np.save(final_flux_path, np.asarray(face_flux, dtype=np.float64))
    np.save(final_cell_velocity_path, np.asarray(cell_velocity, dtype=np.float64))
    np.save(final_pressure_path, np.asarray(pressure, dtype=np.float64))
    np.save(face_centers_path, np.asarray(mesh.face_centers, dtype=np.float64))
    np.save(face_normals_path, np.asarray(mesh.face_normals, dtype=np.float64))
    np.save(face_to_cells_path, np.asarray(mesh.face_to_cells, dtype=np.int64))
    np.save(cell_centers_path, np.asarray(mesh.cell_centers, dtype=np.float64))
    np.save(cell_volumes_path, np.asarray(mesh.cell_volumes, dtype=np.float64))
    np.save(face_groups_path, np.asarray(face_group_codes, dtype=np.int32))

    metadata = {
        "source_run_dir": str(run_dir),
        "mesh_name": str(mesh_name),
        "mesh_sha256": str(mesh_sha256),
        "mesh_stats": {
            "tetra_count": int(mesh.tetrahedra.shape[0]),
            "face_count": int(mesh.face_vertices.shape[0]),
            "node_count": int(mesh.points.shape[0]),
        },
        "flow_mode": str(flow_mode),
        "flow_dt_mode": str(flow_dt_mode),
        "flow_steps": int(flow_steps),
        "physical_time_final": float(physical_time_final),
        "run_completed": bool(run_completed),
        "numerically_stable": bool(numerically_stable),
        "physically_ready": bool(physically_ready),
        "ready_for_next_stage": bool(ready_for_next_stage),
        "ready_for_long_run": bool(ready_for_long_run),
        "stage_status_reason": str(stage_status_reason),
        "ready_for_flow_to_transport_coupling": bool(
            ready_for_flow_to_transport_coupling
        ),
        "ns_baseline_physical_clean": bool(ns_baseline_physical_clean),
        "outlet_flux_rescale_used": bool(outlet_flux_rescale_used),
        "nonphysical_flux_fix_used": bool(nonphysical_flux_fix_used),
        "convective_auto_damping_used_any": bool(convective_auto_damping_used_any),
        "wall_flux_max_abs_after": float(wall_flux_max_abs_after),
        "outlet_inlet_flux_ratio": float(outlet_inlet_flux_ratio),
        "final_div_l2": float(final_div_l2),
        "final_div_max": float(final_div_max),
        "source_flow_flux_balance": {
            "wall_flux_max_abs_after": float(wall_flux_max_abs_after),
            "outlet_inlet_flux_ratio": float(outlet_inlet_flux_ratio),
        },
        "boundary_face_group_counts": {
            "wall": int(np.count_nonzero(np.asarray(face_group_codes) == 1)),
            "outlet": int(np.count_nonzero(np.asarray(face_group_codes) == 2)),
            "inlet": int(np.count_nonzero(np.asarray(face_group_codes) == 3)),
        },
        "stage_status": {
            "run_completed": bool(run_completed),
            "numerically_stable": bool(numerically_stable),
            "physically_ready": bool(physically_ready),
            "ready_for_next_stage": bool(ready_for_next_stage),
            "ready_for_long_run": bool(ready_for_long_run),
            "stage_status_reason": str(stage_status_reason),
        },
    }
    _write_json(metadata_path, metadata)
    artifact_sha256 = {
        "flow_coupling_metadata_json": coupling_sha256_file(metadata_path),
        "final_corrected_face_flux_npy": coupling_sha256_file(final_flux_path),
        "face_to_cells_npy": coupling_sha256_file(face_to_cells_path),
        "cell_volumes_npy": coupling_sha256_file(cell_volumes_path),
    }

    return {
        "metadata": metadata,
        "artifact_sha256": artifact_sha256,
        "artifacts": {
            "flow_coupling_metadata_json": str(metadata_path),
            "final_corrected_face_flux_npy": str(final_flux_path),
            "final_cell_velocity_npy": str(final_cell_velocity_path),
            "final_pressure_npy": str(final_pressure_path),
            "face_centers_npy": str(face_centers_path),
            "face_normals_npy": str(face_normals_path),
            "face_to_cells_npy": str(face_to_cells_path),
            "cell_centers_npy": str(cell_centers_path),
            "cell_volumes_npy": str(cell_volumes_path),
            "face_groups_npy": str(face_groups_path),
        },
    }


def _save_face_flux_stream_proxy(
    *,
    mesh,
    corrected_flux: np.ndarray,
    out_path: Path,
    max_faces: int = 1200,
) -> str:
    boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
    if boundary.size == 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.text(0.5, 0.5, "No boundary faces", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return str(out_path)
    q = np.asarray(corrected_flux[boundary], dtype=np.float64)
    mag = np.abs(q)
    if boundary.size > max_faces:
        order = np.argsort(-mag)
        pick = np.sort(order[:max_faces])
    else:
        pick = np.arange(boundary.size, dtype=np.int64)
    ids = boundary[pick]
    q_sel = q[pick]
    c = np.asarray(mesh.face_centers[ids], dtype=np.float64)
    n = np.asarray(mesh.face_normals[ids], dtype=np.float64)
    sign = np.sign(q_sel)
    sign[sign == 0.0] = 1.0
    u = n[:, 0] * sign
    v = n[:, 1] * sign
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    qv = ax.quiver(
        c[:, 0],
        c[:, 1],
        u,
        v,
        np.abs(q_sel),
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=120.0,
        width=0.0018,
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Face-Flux Stream Proxy XY")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(qv, ax=ax, label="|face flux| [m^3/s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _guard_plot_output(func):
    def guarded(*args, **kwargs):
        if _POSTPROCESSING_MODE == "minimal":
            return None
        return func(*args, **kwargs)

    return guarded


def _postprocessing_writes_visualizations(mode: str) -> bool:
    return str(mode) == "full"


for _plot_helper_name in (
    "_save_scatter",
    "_save_vectors_normalized",
    "_save_vectors_raw_clipped_scale",
    "_save_vectors_downsampled",
    "_save_vectors_sparse_normalized",
    "_save_vectors_by_region",
    "_save_outlet_flux_faces",
    "_save_pressure_outlet_zoom",
    "_save_divergence_hotspots",
    "_save_divergence_before_after_same_scale",
    "_save_correction_flux_hotspots_xy",
    "_save_pressure_correction_flux_magnitude_xy",
    "_save_top_divergence_correction_breakdown_xy",
    "_save_boundary_policy_comparison_bar",
    "_save_top_divergence_cells_before_after_limiter_xy",
    "_save_correction_flux_limited_faces_xy",
    "_save_limiter_effect_histogram",
    "_save_pressure_operator_symmetry_hotspots_xy",
    "_save_pressure_operator_matrixfree_mismatch_xy",
    "_save_explicit_vs_matrixfree_residual_histogram",
    "_save_velocity_vectors_xy_grid_binned",
    "_save_velocity_vectors_xy_region_panels_binned",
    "_save_velocity_vectors_xy_direction_colored",
    "_save_velocity_magnitude_p95_clipped_xy",
    "_save_divergence_stage_comparison",
    "_save_velocity_magnitude_before_after_predictor",
    "_save_face_flux_delta_predictor_xy",
    "_save_vectors_sparse_normalized_seeded",
    "_save_vectors_region_panels",
    "_save_face_flux_stream_proxy",
):
    globals()[_plot_helper_name] = _guard_plot_output(globals()[_plot_helper_name])


def _reconstruct_face_flux_from_cell_velocity(mesh, velocity: np.ndarray) -> np.ndarray:
    vel = np.asarray(velocity, dtype=np.float64)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    n = np.asarray(mesh.face_normals, dtype=np.float64)
    area = np.asarray(mesh.face_areas, dtype=np.float64)
    flux = np.zeros((mesh.face_vertices.shape[0],), dtype=np.float64)
    for fid in range(mesh.face_vertices.shape[0]):
        o = int(c0[fid])
        nb = int(c1[fid])
        if nb >= 0:
            vf = 0.5 * (vel[o] + vel[nb])
        else:
            vf = vel[o]
        flux[fid] = float(np.dot(vf, n[fid]) * area[fid])
    return flux


def _region_metric(mask: np.ndarray, values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if vals.size == 0:
        return {"count": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "l2": 0.0}
    return {
        "count": float(vals.size),
        "max_abs": float(np.max(np.abs(vals))),
        "mean_abs": float(np.mean(np.abs(vals))),
        "l2": float(np.sqrt(np.mean(vals * vals))),
    }


def _velocity_reconstruction_audit(
    *,
    mesh,
    corrected_face_flux: np.ndarray,
    reconstructed_cell_velocity: np.ndarray,
    masks: dict[str, np.ndarray],
    region_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        compute_tetra_flux_divergence,
    )

    q_corr = np.asarray(corrected_face_flux, dtype=np.float64)
    q_rebuilt = _reconstruct_face_flux_from_cell_velocity(
        mesh, reconstructed_cell_velocity
    )
    q_diff = q_rebuilt - q_corr
    div_corr = compute_tetra_flux_divergence(mesh, q_corr)["divergence"]
    div_rebuilt = compute_tetra_flux_divergence(mesh, q_rebuilt)["divergence"]

    top_faces_idx = np.argsort(-np.abs(q_diff))[:30]
    top_faces: list[dict[str, Any]] = []
    for fid in top_faces_idx.tolist():
        top_faces.append(
            {
                "face_index": int(fid),
                "center": np.asarray(mesh.face_centers[fid], dtype=np.float64).tolist(),
                "normal": np.asarray(mesh.face_normals[fid], dtype=np.float64).tolist(),
                "face_area": float(mesh.face_areas[fid]),
                "flux_corrected": float(q_corr[fid]),
                "flux_rebuilt_from_cell_velocity": float(q_rebuilt[fid]),
                "flux_mismatch": float(q_diff[fid]),
            }
        )
    speed = np.linalg.norm(
        np.asarray(reconstructed_cell_velocity, dtype=np.float64), axis=1
    )
    top_cells_idx = np.argsort(-speed)[:30]
    top_cells: list[dict[str, Any]] = []
    for cid in top_cells_idx.tolist():
        top_cells.append(
            {
                "cell_index": int(cid),
                "center": np.asarray(mesh.cell_centers[cid], dtype=np.float64).tolist(),
                "velocity": np.asarray(
                    reconstructed_cell_velocity[cid], dtype=np.float64
                ).tolist(),
                "velocity_magnitude": float(speed[cid]),
            }
        )

    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    region_metrics: dict[str, dict[str, float]] = {}
    for name, cell_mask in masks.items():
        cm = np.asarray(cell_mask, dtype=bool)
        fm = cm[c0]
        interior = c1 >= 0
        if np.any(interior):
            fm[interior] = fm[interior] | cm[c1[interior]]
        region_metrics[name] = _region_metric(fm, q_diff)
    tendency = np.zeros_like(np.asarray(reconstructed_cell_velocity, dtype=np.float64))
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    for fid in range(mesh.face_vertices.shape[0]):
        o = int(c0[fid])
        nb = int(c1[fid])
        q = float(q_corr[fid])
        a = max(float(areas[fid]), 1e-30)
        tendency[o] += (-q / a) * np.asarray(normals[fid], dtype=np.float64)
        if nb >= 0:
            tendency[nb] += (+q / a) * np.asarray(normals[fid], dtype=np.float64)
    v_cell = np.asarray(reconstructed_cell_velocity, dtype=np.float64)
    v_mag = np.linalg.norm(v_cell, axis=1)
    t_mag = np.linalg.norm(tendency, axis=1)
    valid = (v_mag > 1e-14) & (t_mag > 1e-14)
    dots = np.zeros((v_cell.shape[0],), dtype=np.float64)
    if np.any(valid):
        dots[valid] = np.einsum(
            "ij,ij->i",
            v_cell[valid] / v_mag[valid, None],
            tendency[valid] / t_mag[valid, None],
        )
    dir_agree = np.zeros((v_cell.shape[0],), dtype=bool)
    dir_agree[valid] = dots[valid] >= 0.0
    suspicious = np.zeros((v_cell.shape[0],), dtype=bool)
    if np.any(valid):
        speed_p95 = float(np.percentile(v_mag[valid], 95.0))
        suspicious = ((~dir_agree) & valid) | (
            (v_mag > max(speed_p95, 1e-14)) & (dots < -0.25)
        )
    suspicious_idx = np.flatnonzero(suspicious)
    if suspicious_idx.size > 30:
        suspicious_idx = suspicious_idx[np.argsort(dots[suspicious_idx])[:30]]
    top_suspicious_cells: list[dict[str, Any]] = []
    for cid in suspicious_idx.tolist():
        labels = [
            name
            for name, mask in masks.items()
            if bool(np.asarray(mask, dtype=bool)[cid])
        ]
        top_suspicious_cells.append(
            {
                "cell_index": int(cid),
                "center": np.asarray(mesh.cell_centers[cid], dtype=np.float64).tolist(),
                "velocity": np.asarray(v_cell[cid], dtype=np.float64).tolist(),
                "velocity_magnitude": float(v_mag[cid]),
                "flux_tendency_vector": np.asarray(
                    tendency[cid], dtype=np.float64
                ).tolist(),
                "flux_tendency_magnitude": float(t_mag[cid]),
                "direction_dot_velocity_vs_flux_tendency": float(dots[cid]),
                "region_labels": labels,
            }
        )

    regions = region_masks if region_masks is not None else masks
    direction_region: dict[str, float] = {}
    for rname, rmask in regions.items():
        rm = np.asarray(rmask, dtype=bool)
        rv = valid & rm
        if np.any(rv):
            direction_region[rname] = float(np.mean(dir_agree[rv]))
        else:
            direction_region[rname] = 0.0

    return {
        "velocity_reconstruction_method": "existing_cell_velocity_from_solver",
        "div_from_corrected_face_flux": {
            "max_abs": float(np.max(np.abs(div_corr))),
            "l2": float(np.sqrt(np.mean(div_corr**2))),
        },
        "div_from_reconstructed_velocity_flux": {
            "max_abs": float(np.max(np.abs(div_rebuilt))),
            "l2": float(np.sqrt(np.mean(div_rebuilt**2))),
        },
        "reconstruction_flux_mismatch": {
            "max_abs": float(np.max(np.abs(q_diff))),
            "l2": float(np.sqrt(np.mean(q_diff**2))),
            "mean_abs": float(np.mean(np.abs(q_diff))),
        },
        "top_30_faces_by_mismatch": top_faces,
        "top_30_cells_by_velocity_magnitude": top_cells,
        "direction_agreement_fraction_global": float(np.mean(dir_agree[valid]))
        if np.any(valid)
        else 0.0,
        "direction_agreement_fraction_by_region": direction_region,
        "suspicious_velocity_cells_count": int(np.count_nonzero(suspicious)),
        "top_suspicious_velocity_cells": top_suspicious_cells,
        "region_wise_mismatch": region_metrics,
        "arrays": {
            "rebuilt_face_flux_from_reconstructed_velocity": q_rebuilt,
            "flux_mismatch": q_diff,
        },
    }


def _velocity_region_audit(
    *,
    centers: np.ndarray,
    velocity: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    c = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    speed = np.linalg.norm(v, axis=1)
    eps = 1e-14

    def _stats(mask: np.ndarray, region: str) -> dict[str, Any]:
        m = np.asarray(mask, dtype=bool)
        ids = np.flatnonzero(m)
        if ids.size == 0:
            return {
                "cell_count": 0,
                "speed_min": 0.0,
                "speed_max": 0.0,
                "speed_mean": 0.0,
                "speed_p95": 0.0,
                "mostly_expected_direction_fraction": 0.0,
                "wrong_direction_fraction": 0.0,
                "transverse_ratio_mean": 0.0,
                "transverse_ratio_p95": 0.0,
                "near_zero_speed_fraction": 1.0,
                "high_speed_outlier_count": 0,
                "mean_direction_xy": [0.0, 0.0],
            }
        sv = speed[ids]
        vx = v[ids, 0]
        vy = v[ids, 1]
        vz = v[ids, 2]
        moving = sv > eps
        near_zero_fraction = float(np.mean(~moving))
        p95 = float(np.percentile(sv, 95.0))
        high_out = int(np.count_nonzero(sv > max(2.0 * p95, eps)))
        if region == "left_inlet_branch":
            expected = vx > 0.0
            wrong = vx < 0.0
            trans = np.sqrt(vy * vy + vz * vz) / np.maximum(np.abs(vx), eps)
        elif region == "right_inlet_branch":
            expected = vx < 0.0
            wrong = vx > 0.0
            trans = np.sqrt(vy * vy + vz * vz) / np.maximum(np.abs(vx), eps)
        elif region == "outlet_branch":
            expected = vy > 0.0
            wrong = vy < 0.0
            trans = np.sqrt(vx * vx + vz * vz) / np.maximum(np.abs(vy), eps)
        else:
            ang = np.arctan2(vy, vx)
            expected = np.cos(ang) >= 0.0
            wrong = np.cos(ang) < 0.0
            trans = np.sqrt(vx * vx + vz * vz) / np.maximum(np.abs(vy), eps)
        expected_frac = float(np.mean(expected[moving])) if np.any(moving) else 0.0
        wrong_frac = float(np.mean(wrong[moving])) if np.any(moving) else 0.0
        mean_xy = np.mean(v[ids, :2], axis=0)
        mean_xy_norm = mean_xy / max(float(np.linalg.norm(mean_xy)), eps)
        return {
            "cell_count": int(ids.size),
            "speed_min": float(np.min(sv)),
            "speed_max": float(np.max(sv)),
            "speed_mean": float(np.mean(sv)),
            "speed_p95": p95,
            "mostly_expected_direction_fraction": expected_frac,
            "wrong_direction_fraction": wrong_frac,
            "transverse_ratio_mean": float(np.mean(trans)),
            "transverse_ratio_p95": float(np.percentile(trans, 95.0)),
            "near_zero_speed_fraction": near_zero_fraction,
            "high_speed_outlier_count": high_out,
            "mean_direction_xy": [float(mean_xy_norm[0]), float(mean_xy_norm[1])],
        }

    out: dict[str, Any] = {}
    for key in (
        "left_inlet_branch",
        "right_inlet_branch",
        "junction",
        "outlet_branch",
        "boundary_adjacent",
        "interior_core",
    ):
        out[key] = _stats(
            region_masks.get(key, np.zeros((c.shape[0],), dtype=bool)), key
        )
    return out


def _finite_scalar(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _projection_acceptance(
    *,
    projection: dict[str, Any],
    pressure: dict[str, Any],
    backend_execution: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    t = dict(
        {
            "pressure_relative_tolerance_effective": 5e-3,
            "pressure_relative_tolerance_max_effective": 5e-3,
            "projection_divergence_reduction_l2_tolerance": 5e-3,
            "projection_divergence_reduction_linf_tolerance": 5e-3,
            "projection_final_div_l2_tolerance": 1.0,
            "projection_final_div_max_tolerance": 20.0,
            "outlet_inlet_flux_ratio_tolerance": 5e-3,
            "net_boundary_flux_relative_tolerance": 5e-3,
            "wall_flux_abs_tolerance": 1e-14,
        }
    )
    if thresholds is not None:
        t.update({str(k): float(v) for k, v in thresholds.items()})

    stop = str(pressure.get("stopping_reason", ""))
    strict_reasons = {
        "converged_relative_l2",
        "converged_relative_max",
        "converged_absolute",
    }
    pressure_linear_solved_strict = bool(stop in strict_reasons)
    res_ratio_l2 = float(pressure.get("residual_ratio_to_rhs_l2", float("inf")))
    res_ratio_max = float(pressure.get("residual_ratio_to_rhs_max", float("inf")))
    pressure_linear_accepted = bool(
        (res_ratio_l2 <= float(t["pressure_relative_tolerance_effective"]))
        or (res_ratio_max <= float(t["pressure_relative_tolerance_max_effective"]))
    )

    final_div_l2 = float(projection.get("final_divergence_l2", float("inf")))
    final_div_max = float(projection.get("final_divergence_max_abs", float("inf")))
    red_l2 = float(projection.get("divergence_reduction_ratio_l2", float("inf")))
    red_linf = float(projection.get("divergence_reduction_ratio", float("inf")))
    inlet_flux = float(projection.get("inlet_flux_total_after", 0.0))
    outlet_flux = float(projection.get("outlet_flux_total_after", 0.0))
    net_flux = float(projection.get("net_boundary_flux_after", float("inf")))
    wall_flux = float(projection.get("wall_flux_max_abs_after", float("inf")))
    outlet_inlet_ratio = outlet_flux / max(abs(inlet_flux), 1e-30)
    net_flux_relative = abs(net_flux) / max(abs(inlet_flux), 1e-30)

    finite_fields = bool(
        _finite_scalar(final_div_l2)
        and _finite_scalar(final_div_max)
        and _finite_scalar(red_l2)
        and _finite_scalar(red_linf)
        and _finite_scalar(outlet_inlet_ratio)
        and _finite_scalar(net_flux_relative)
        and _finite_scalar(res_ratio_l2)
        and _finite_scalar(res_ratio_max)
    )
    if backend_execution is not None:
        finite_fields = finite_fields and (
            not bool(backend_execution.get("used_numpy_fallback", False)) or True
        )

    div_l2_ok = bool(
        (final_div_l2 <= float(t["projection_final_div_l2_tolerance"]))
        or (red_l2 <= float(t["projection_divergence_reduction_l2_tolerance"]))
    )
    div_linf_ok = bool(
        (final_div_max <= float(t["projection_final_div_max_tolerance"]))
        or (red_linf <= float(t["projection_divergence_reduction_linf_tolerance"]))
    )
    ratio_tol = float(t["outlet_inlet_flux_ratio_tolerance"])
    outlet_ratio_ok = bool(abs(outlet_inlet_ratio - 1.0) <= ratio_tol)
    net_flux_ok = bool(
        net_flux_relative <= float(t["net_boundary_flux_relative_tolerance"])
    )
    wall_flux_ok = bool(abs(wall_flux) <= float(t["wall_flux_abs_tolerance"]))

    projection_accepted = bool(
        finite_fields
        and div_l2_ok
        and div_linf_ok
        and outlet_ratio_ok
        and net_flux_ok
        and wall_flux_ok
    )
    projection_solved = bool(pressure_linear_accepted and projection_accepted)
    blocking_checks = {
        "pressure_linear_residual_acceptance": bool(pressure_linear_accepted),
        "finite_fields": bool(finite_fields),
        "divergence_l2_acceptance": bool(div_l2_ok),
        "divergence_linf_acceptance": bool(div_linf_ok),
        "outlet_inlet_ratio": bool(outlet_ratio_ok),
        "net_boundary_flux_relative": bool(net_flux_ok),
        "wall_flux": bool(wall_flux_ok),
    }

    checklist = {
        "residual_l2": bool(
            res_ratio_l2 <= float(t["pressure_relative_tolerance_effective"])
        ),
        "residual_max": bool(
            res_ratio_max <= float(t["pressure_relative_tolerance_max_effective"])
        ),
        "div_l2": bool(final_div_l2 <= float(t["projection_final_div_l2_tolerance"])),
        "div_linf": bool(
            final_div_max <= float(t["projection_final_div_max_tolerance"])
        ),
        "div_reduction_l2": bool(
            red_l2 <= float(t["projection_divergence_reduction_l2_tolerance"])
        ),
        "div_reduction_linf": bool(
            red_linf <= float(t["projection_divergence_reduction_linf_tolerance"])
        ),
        "inlet_flux": bool(_finite_scalar(inlet_flux)),
        "outlet_flux": bool(_finite_scalar(outlet_flux)),
        "outlet_inlet_ratio": bool(outlet_ratio_ok),
        "net_boundary_flux_relative": bool(net_flux_ok),
        "wall_flux": bool(wall_flux_ok),
        "finite_fields": bool(finite_fields),
    }

    reason = "projection acceptance criteria satisfied"
    if not pressure_linear_accepted:
        reason = "pressure linear system residual acceptance failed"
    elif not finite_fields:
        reason = "non-finite fields in projection metrics"
    elif not div_l2_ok:
        reason = "divergence l2 criteria failed"
    elif not div_linf_ok:
        reason = "divergence linf criteria failed"
    elif not outlet_ratio_ok:
        reason = "outlet/inlet flux ratio criteria failed"
    elif not net_flux_ok:
        reason = "net boundary flux relative criteria failed"
    elif not wall_flux_ok:
        reason = "wall flux criteria failed"

    return {
        "pressure_linear_solved_strict": bool(pressure_linear_solved_strict),
        "pressure_linear_accepted": bool(pressure_linear_accepted),
        "projection_accepted": bool(projection_accepted),
        "projection_solved": bool(projection_solved),
        "reason": str(reason),
        "thresholds": t,
        "blocking_checks": blocking_checks,
        "metrics": {
            "residual_ratio_to_rhs_l2": float(res_ratio_l2),
            "residual_ratio_to_rhs_max": float(res_ratio_max),
            "final_div_l2": float(final_div_l2),
            "final_div_max_abs": float(final_div_max),
            "divergence_reduction_ratio_l2": float(red_l2),
            "divergence_reduction_ratio_linf": float(red_linf),
            "inlet_flux_after": float(inlet_flux),
            "outlet_flux_after": float(outlet_flux),
            "outlet_inlet_flux_ratio": float(outlet_inlet_ratio),
            "net_boundary_flux_after": float(net_flux),
            "net_boundary_flux_relative": float(net_flux_relative),
            "wall_flux_max_abs_after": float(wall_flux),
            "stopping_reason": stop,
        },
        "checklist": checklist,
    }


def _projection_failed_criteria(
    acceptance: dict[str, Any],
) -> list[str]:
    blocking_checks = dict(acceptance.get("blocking_checks", {}))
    if blocking_checks:
        ordered_keys = (
            "pressure_linear_residual_acceptance",
            "finite_fields",
            "divergence_l2_acceptance",
            "divergence_linf_acceptance",
            "outlet_inlet_ratio",
            "net_boundary_flux_relative",
            "wall_flux",
        )
        return [
            key
            for key in ordered_keys
            if (key in blocking_checks) and (not bool(blocking_checks.get(key, False)))
        ]
    checklist = dict(acceptance.get("checklist", {}))
    ordered_keys = (
        "residual_l2",
        "residual_max",
        "div_l2",
        "div_linf",
        "div_reduction_l2",
        "div_reduction_linf",
        "outlet_inlet_ratio",
        "net_boundary_flux_relative",
        "wall_flux",
        "finite_fields",
    )
    return [
        key
        for key in ordered_keys
        if (key in checklist) and (not bool(checklist.get(key, False)))
    ]


def _projection_strict_diagnostic_criteria_not_met(
    acceptance: dict[str, Any],
) -> list[str]:
    checklist = dict(acceptance.get("checklist", {}))
    ordered_keys = (
        "residual_l2",
        "residual_max",
        "div_l2",
        "div_linf",
        "div_reduction_l2",
        "div_reduction_linf",
        "inlet_flux",
        "outlet_flux",
        "outlet_inlet_ratio",
        "net_boundary_flux_relative",
        "wall_flux",
        "finite_fields",
    )
    return [
        key
        for key in ordered_keys
        if (key in checklist) and (not bool(checklist.get(key, False)))
    ]


def _build_projection_acceptance_step_record(
    *,
    step: int,
    acceptance: dict[str, Any],
    projection: dict[str, Any],
    pressure: dict[str, Any],
    used_flow_dt: float,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "projection_solved": bool(acceptance.get("projection_solved", False)),
        "projection_accepted": bool(acceptance.get("projection_accepted", False)),
        "pressure_linear_accepted": bool(
            acceptance.get("pressure_linear_accepted", False)
        ),
        "pressure_linear_solved_strict": bool(
            acceptance.get("pressure_linear_solved_strict", False)
        ),
        "projection_acceptance_reason": str(acceptance.get("reason", "")),
        "projection_failed_criteria": _projection_failed_criteria(acceptance),
        "strict_diagnostic_criteria_not_met": (
            _projection_strict_diagnostic_criteria_not_met(acceptance)
        ),
        "projection_blocking_checks": dict(acceptance.get("blocking_checks", {})),
        "projection_acceptance_checklist": dict(acceptance.get("checklist", {})),
        "pressure_stopping_reason": str(pressure.get("stopping_reason", "")),
        "pressure_iterations": int(pressure.get("actual_iterations", 0)),
        "pressure_residual_ratio_to_rhs_l2": float(
            acceptance.get("metrics", {}).get("residual_ratio_to_rhs_l2", float("inf"))
        ),
        "pressure_residual_ratio_to_rhs_max": float(
            acceptance.get("metrics", {}).get("residual_ratio_to_rhs_max", float("inf"))
        ),
        "initial_divergence_max_abs": float(
            projection.get("initial_divergence_max_abs", 0.0)
        ),
        "final_divergence_max_abs": float(
            projection.get("final_divergence_max_abs", 0.0)
        ),
        "initial_divergence_l2": float(projection.get("initial_divergence_l2", 0.0)),
        "final_divergence_l2": float(projection.get("final_divergence_l2", 0.0)),
        "outlet_inlet_flux_ratio": float(
            acceptance.get("metrics", {}).get("outlet_inlet_flux_ratio", 0.0)
        ),
        "net_boundary_flux_after": float(
            acceptance.get("metrics", {}).get("net_boundary_flux_after", 0.0)
        ),
        "net_boundary_flux_relative": float(
            acceptance.get("metrics", {}).get("net_boundary_flux_relative", 0.0)
        ),
        "wall_flux_max_abs_after": float(
            acceptance.get("metrics", {}).get("wall_flux_max_abs_after", 0.0)
        ),
        "used_flow_dt": float(used_flow_dt),
        "finite_fields": bool(
            acceptance.get("checklist", {}).get("finite_fields", False)
        ),
    }


def _summarize_startup_bootstrap(
    *,
    bootstrap_history: list[dict[str, Any]],
    initial_divergence: dict[str, Any],
    requested_max_steps: int,
    bootstrap_required: bool = True,
    not_run_reason: str = "bootstrap not run",
    uses_default_budget: bool = True,
) -> dict[str, Any]:
    failed_counts: dict[str, int] = {}
    strict_failed_counts: dict[str, int] = {}
    for row in bootstrap_history:
        for key in row.get("projection_failed_criteria", []):
            name = str(key)
            failed_counts[name] = int(failed_counts.get(name, 0) + 1)
        for key in row.get("strict_diagnostic_criteria_not_met", []):
            name = str(key)
            strict_failed_counts[name] = int(strict_failed_counts.get(name, 0) + 1)
    dominant_failures = [
        key
        for key, _count in sorted(
            failed_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    dominant_strict_failures = [
        key
        for key, _count in sorted(
            strict_failed_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]
    configured_max_steps = int(max(0, requested_max_steps))
    required_consecutive = STARTUP_BOOTSTRAP_REQUIRED_CONSECUTIVE_ACCEPTED_STEPS
    achieved_consecutive = 0
    for row in bootstrap_history:
        if bool(row.get("projection_solved", False)):
            achieved_consecutive += 1
        else:
            achieved_consecutive = 0
    converged = bool(
        bootstrap_required and achieved_consecutive >= required_consecutive
    )
    if not bootstrap_required:
        reason = str(not_run_reason)
    elif configured_max_steps == 0:
        reason = (
            "raw initialized face-flux state may be non-ready; startup bootstrap is "
            "disabled by configuration and physical progression is blocked"
        )
    elif converged:
        reason = (
            "raw initialized face-flux state is not projection-ready; bootstrap "
            "completed consecutive full-step qualification before physical time stepping"
        )
    else:
        reason = (
            "raw initialized face-flux state is not projection-ready; bootstrap did "
            "not converge within the configured cap"
        )
    physical_progression_allowed = bool((not bootstrap_required) or converged)
    cap_reached = bool(
        bootstrap_required
        and configured_max_steps > 0
        and (not converged)
        and (len(bootstrap_history) >= configured_max_steps)
    )
    return {
        "bootstrap_enabled": bool(bootstrap_required and configured_max_steps > 0),
        "bootstrap_required": bool(bootstrap_required),
        "bootstrap_requested_max_steps": int(configured_max_steps),
        "bootstrap_configured_max_steps": int(configured_max_steps),
        "bootstrap_cap_policy": (
            "default_legacy_search_plus_qualification_tail"
            if uses_default_budget
            else "explicit_total_cap"
        ),
        "bootstrap_legacy_search_budget": (
            int(STARTUP_BOOTSTRAP_LEGACY_SEARCH_BUDGET) if uses_default_budget else None
        ),
        "bootstrap_qualification_tail": (
            int(STARTUP_BOOTSTRAP_QUALIFICATION_TAIL) if uses_default_budget else None
        ),
        "bootstrap_effective_total_cap": int(configured_max_steps),
        "bootstrap_steps": int(len(bootstrap_history)),
        "bootstrap_converged": bool(converged),
        "bootstrap_cap_reached": bool(cap_reached),
        "bootstrap_reason": str(reason),
        "bootstrap_convergence_criterion": (
            "three consecutive full-step projection_solved qualifications according "
            "to production acceptance policy"
        ),
        "bootstrap_required_consecutive_accepted_iterations": int(required_consecutive),
        "bootstrap_achieved_consecutive_accepted_iterations": int(achieved_consecutive),
        "qualification_iteration_history": list(bootstrap_history),
        "bootstrap_physical_time_advanced": 0.0,
        "physical_progression_allowed": bool(physical_progression_allowed),
        "dominant_failed_criteria": dominant_failures,
        "dominant_blocking_criteria": dominant_failures,
        "dominant_strict_diagnostic_criteria_not_met": dominant_strict_failures,
        "initial_state_divergence_max_abs": float(
            initial_divergence.get("divergence_max_abs", 0.0)
        ),
        "initial_state_divergence_l2": float(
            initial_divergence.get("divergence_l2", 0.0)
        ),
        "final_bootstrap_step": dict(bootstrap_history[-1])
        if bootstrap_history
        else {},
    }


def _parse_snapshot_steps(raw: str) -> set[int]:
    txt = str(raw).strip()
    if not txt:
        return set()
    out: set[int] = set()
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            continue
        if v >= 1:
            out.add(v)
    return out


def _next_snapshot_time(physical_time: float, interval: float) -> float:
    """Return the first interval boundary strictly after ``physical_time``."""

    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("snapshot time interval must be finite and positive")
    quotient = np.floor((float(physical_time) + 1e-12 * interval) / interval)
    return float((quotient + 1.0) * interval)


def _clamp_flow_dt_to_stop_time(
    *,
    physical_time: float,
    dt_candidate: float,
    stop_physical_time: float | None,
) -> float:
    """Shorten one step only when needed to land on the final stop time."""

    remaining = [float(dt_candidate)]
    if stop_physical_time is not None:
        remaining.append(float(stop_physical_time) - float(physical_time))
    clamped = min(remaining)
    if not np.isfinite(clamped) or clamped <= 0.0:
        raise ValueError("flow timestep boundary clamp produced a non-positive dt")
    return float(clamped)


def _area_weighted_boundary_pressure(
    mesh: Any,
    pressure: np.ndarray,
    face_ids: np.ndarray,
) -> tuple[float, float]:
    faces = np.asarray(face_ids, dtype=np.int64)
    areas = np.asarray(mesh.face_areas[faces], dtype=np.float64)
    pairs = np.asarray(mesh.face_to_cells[faces], dtype=np.int64)
    owners = np.where(pairs[:, 0] >= 0, pairs[:, 0], pairs[:, 1])
    area = float(np.sum(areas, dtype=np.float64))
    if area <= 0.0 or np.any(owners < 0):
        raise ValueError("boundary pressure surface has invalid faces or area")
    value = float(
        np.sum(areas * np.asarray(pressure, dtype=np.float64)[owners], dtype=np.float64)
        / area
    )
    return value, area


def _wall_shear_stress_metrics(
    mesh: Any,
    cell_velocity: np.ndarray,
    *,
    dynamic_viscosity: float,
) -> dict[str, Any]:
    """Estimate no-slip wall shear from the owner-cell tangential gradient."""

    wall_faces = np.asarray(mesh.wall_faces, dtype=np.int64)
    if wall_faces.size == 0:
        raise ValueError("wall-shear metric requires at least one wall face")
    pairs = np.asarray(mesh.face_to_cells[wall_faces], dtype=np.int64)
    owners = np.where(pairs[:, 0] >= 0, pairs[:, 0], pairs[:, 1])
    if np.any(owners < 0):
        raise ValueError("wall-shear metric found a wall face without an owner")
    normals = np.asarray(mesh.face_normals[wall_faces], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-300)
    owner_velocity = np.asarray(cell_velocity, dtype=np.float64)[owners]
    normal_velocity = np.einsum("ij,ij->i", owner_velocity, normals)
    tangential_velocity = owner_velocity - normal_velocity[:, None] * normals
    tangential_speed = np.linalg.norm(tangential_velocity, axis=1)
    owner_to_face = np.asarray(
        mesh.face_centers[wall_faces], dtype=np.float64
    ) - np.asarray(mesh.cell_centers[owners], dtype=np.float64)
    normal_distance = np.abs(np.einsum("ij,ij->i", owner_to_face, normals))
    shear = (
        float(dynamic_viscosity)
        * tangential_speed
        / np.maximum(normal_distance, 1e-300)
    )
    areas = np.asarray(mesh.face_areas[wall_faces], dtype=np.float64)
    total_area = float(np.sum(areas, dtype=np.float64))
    if total_area <= 0.0:
        raise ValueError("wall-shear metric requires positive wall area")
    return {
        "method": "wall_owner_tangential_linear_normal_gradient",
        "face_count": int(wall_faces.size),
        "area_weighted_mean_pa": float(
            np.sum(areas * shear, dtype=np.float64) / total_area
        ),
        "max_pa": float(np.max(shear)),
    }


def _parse_flow_modes(raw: str) -> list[str]:
    txt = str(raw).strip()
    if not txt:
        return []
    allowed = {
        "projection_only",
        "stokes_viscous_projection",
        "navier_stokes_projection_debug",
    }
    out: list[str] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        if p in allowed and p not in out:
            out.append(p)
    return out


def _parse_viscous_predictor_modes(raw: str) -> list[str]:
    txt = str(raw).strip()
    if not txt:
        return []
    allowed = {
        "none",
        "no_viscous_debug_copy",
        "explicit_cell_velocity_laplacian_substepped",
        "explicit_cell_velocity_laplacian_substepped_conservative",
        "face_flux_laplacian_substepped",
    }
    out: list[str] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        if p in allowed and p not in out:
            out.append(p)
    return out


def _parse_convective_stabilization_modes(raw: str) -> list[str]:
    items = [s.strip().lower() for s in str(raw).split(",") if s.strip()]
    out: list[str] = []
    for item in items:
        if item in {"auto_damping", "substepping"} and item not in out:
            out.append(item)
    return out


def _parse_flow_dt_mode(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode in {"manual", "auto_cfl"}:
        return mode
    return "manual"


def _parse_float_list(raw: str) -> list[float]:
    txt = str(raw).strip()
    if not txt:
        return []
    out: list[float] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def _parse_cap_list(raw: str) -> list[float | None]:
    txt = str(raw).strip()
    if not txt:
        return []
    out: list[float | None] = []
    for part in txt.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p in {"default", "none", "null"}:
            out.append(None)
            continue
        try:
            out.append(float(p))
        except ValueError:
            continue
    return out


def _resolve_viscous_predictor_mode(
    *,
    flow_mode: str,
    predictor_mode_cli: str,
    predictor_mode_explicit: bool,
    wall_velocity_boundary_mode: str = "slip",
) -> tuple[str, str]:
    mode = str(predictor_mode_cli)
    if (not bool(predictor_mode_explicit)) and str(flow_mode) in {
        "stokes_viscous_projection",
        "navier_stokes_projection_debug",
    }:
        if str(wall_velocity_boundary_mode) in {"no_slip", "no_slip_tangential"}:
            return (
                "explicit_cell_velocity_laplacian_substepped_conservative",
                "no-slip wall mode defaults to conservative cell-velocity wall momentum predictor",
            )
        return (
            "face_flux_laplacian_substepped",
            "stokes/navier mode defaults to face_flux_laplacian_substepped",
        )
    if not bool(predictor_mode_explicit):
        return (
            "explicit_cell_velocity_laplacian_substepped",
            "default viscous predictor retained for non-stokes flow mode",
        )
    if bool(predictor_mode_explicit):
        return (mode, "viscous predictor set explicitly by CLI")
    return (mode, "default viscous predictor retained for non-stokes flow mode")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_ready(v) for k, v in row.items()})


def _evaluate_viscous_progression_acceptance(
    *,
    history: list[dict[str, Any]],
    projection_only_baseline: dict[str, Any] | None,
    outlet_inlet_flux_ratio_tolerance: float,
    wall_flux_abs_tolerance: float,
    projection_final_div_l2_tolerance: float,
    projection_final_div_max_tolerance: float,
    velocity_blowup_ratio_tolerance: float = 2.5,
) -> dict[str, Any]:
    if not history:
        return {
            "viscous_progression_accepted": False,
            "reason": "no history",
            "checks": {},
        }
    final = history[-1]
    finite_ok = bool(final.get("finite_fields", False))
    wall_ok = abs(float(final.get("wall_flux_max_abs_after", 0.0))) <= float(
        wall_flux_abs_tolerance
    )
    flux_ok = abs(float(final.get("outlet_inlet_flux_ratio", 0.0)) - 1.0) <= float(
        outlet_inlet_flux_ratio_tolerance
    )
    div_l2_ok = float(final.get("final_divergence_l2", float("inf"))) <= float(
        projection_final_div_l2_tolerance
    )
    div_max_ok = float(final.get("final_divergence_max_abs", float("inf"))) <= float(
        projection_final_div_max_tolerance
    )
    base_vmax = None
    vel_ok = True
    if projection_only_baseline is not None:
        base_vmax = float(
            projection_only_baseline.get("velocity_magnitude_max_final", 0.0)
        )
        stokes_vmax = float(final.get("velocity_magnitude_max", 0.0))
        vel_ok = stokes_vmax <= max(
            float(velocity_blowup_ratio_tolerance) * max(base_vmax, 1e-20), 1e-20
        )
    accepted = bool(
        finite_ok and wall_ok and flux_ok and div_l2_ok and div_max_ok and vel_ok
    )
    reason = "viscous progression accepted"
    if not finite_ok:
        reason = "non-finite fields in viscous progression"
    elif not wall_ok:
        reason = "wall flux tolerance violated"
    elif not flux_ok:
        reason = "outlet/inlet flux ratio violated"
    elif not div_l2_ok or not div_max_ok:
        reason = "divergence tolerance violated"
    elif not vel_ok:
        reason = "velocity magnitude blow-up vs projection_only baseline"
    return {
        "viscous_progression_accepted": bool(accepted),
        "reason": str(reason),
        "checks": {
            "finite_ok": bool(finite_ok),
            "wall_flux_ok": bool(wall_ok),
            "flux_ratio_ok": bool(flux_ok),
            "div_l2_ok": bool(div_l2_ok),
            "div_max_ok": bool(div_max_ok),
            "velocity_blowup_ok": bool(vel_ok),
            "projection_only_velocity_max_final": base_vmax,
            "stokes_velocity_max_final": float(
                final.get("velocity_magnitude_max", 0.0)
            ),
            "velocity_blowup_ratio_tolerance": float(velocity_blowup_ratio_tolerance),
        },
    }


def _evaluate_flow_progression_acceptance(
    *,
    history: list[dict[str, Any]],
    allow_projection_warning_steps: int,
    startup_warning_steps: int = 0,
    outlet_inlet_flux_ratio_tolerance: float,
    wall_flux_abs_tolerance: float,
    startup_catastrophic_divergence_ratio: float = 5.0,
    startup_catastrophic_divergence_abs: float = 1.0e6,
) -> dict[str, Any]:
    if not history:
        return {
            "flow_progression_solved": False,
            "reason": "no progression history",
            "warning_step_count": 0,
            "warning_steps": [],
            "finite_all_steps": False,
            "flux_ratio_all_steps_ok": False,
            "wall_flux_all_steps_ok": False,
            "steps_completed": 0,
            "final_step": {},
            "worst_step": {},
        }
    warning_steps = [
        int(r.get("step", -1))
        for r in history
        if not bool(r.get("projection_solved", False))
    ]
    startup_allowed = int(max(0, startup_warning_steps))
    startup_warning_steps_observed: list[int] = []
    nonstartup_failed_steps: list[int] = []
    startup_rejected_steps: list[int] = []
    for r in history:
        step = int(r.get("step", -1))
        if bool(r.get("projection_solved", False)):
            continue
        is_startup = step >= 1 and step <= startup_allowed
        final_div = abs(float(r.get("final_divergence_max_abs", 0.0)))
        init_div = abs(float(r.get("initial_divergence_max_abs", 0.0)))
        finite = bool(r.get("finite_fields", False))
        ratio_ok = abs(float(r.get("outlet_inlet_flux_ratio", 0.0)) - 1.0) <= float(
            outlet_inlet_flux_ratio_tolerance
        )
        wall_ok = abs(float(r.get("wall_flux_max_abs_after", 0.0))) <= float(
            wall_flux_abs_tolerance
        )
        catastrophic = bool(
            final_div
            > max(
                float(startup_catastrophic_divergence_abs),
                float(startup_catastrophic_divergence_ratio) * max(init_div, 1e-30),
            )
        )
        startup_ok = bool(finite and ratio_ok and wall_ok and (not catastrophic))
        if is_startup and startup_ok:
            startup_warning_steps_observed.append(step)
        elif is_startup:
            startup_rejected_steps.append(step)
        else:
            nonstartup_failed_steps.append(step)
    finite_all = all(bool(r.get("finite_fields", False)) for r in history)
    flux_ok_all = all(
        abs(float(r.get("outlet_inlet_flux_ratio", 0.0)) - 1.0)
        <= float(outlet_inlet_flux_ratio_tolerance)
        for r in history
    )
    wall_ok_all = all(
        abs(float(r.get("wall_flux_max_abs_after", 0.0)))
        <= float(wall_flux_abs_tolerance)
        for r in history
    )
    # A failed pressure solve is only admissible inside the explicitly bounded
    # startup window. Production steps must never inherit the legacy global
    # warning allowance.
    nonstartup_allowed_by_legacy = bool(len(nonstartup_failed_steps) == 0)
    warning_ok = bool(len(warning_steps) <= int(max(0, allow_projection_warning_steps)))
    solved = bool(
        finite_all
        and flux_ok_all
        and wall_ok_all
        and nonstartup_allowed_by_legacy
        and (len(startup_rejected_steps) == 0)
    )
    solved_with_startup_tolerance = bool(solved)
    reason = "flow progression acceptance satisfied"
    if not finite_all:
        reason = "non-finite fields detected in progression history"
    elif startup_rejected_steps:
        reason = (
            "startup warning step violates finite/flux/wall/divergence safety checks"
        )
    elif not nonstartup_allowed_by_legacy:
        reason = "non-startup projection failures exceed allowed warning count"
    elif (not warning_ok) and (startup_allowed <= 0):
        reason = "too many projection warning steps"
    elif not flux_ok_all:
        reason = "outlet/inlet flux ratio tolerance violated in progression"
    elif not wall_ok_all:
        reason = "wall flux tolerance violated in progression"
    worst = max(
        history, key=lambda r: abs(float(r.get("final_divergence_max_abs", 0.0)))
    )
    return {
        "flow_progression_solved": bool(solved),
        "flow_progression_solved_with_startup_tolerance": bool(
            solved_with_startup_tolerance
        ),
        "reason": str(reason),
        "warning_step_count": int(len(warning_steps)),
        "warning_steps": warning_steps,
        "startup_warning_steps_allowed": int(startup_allowed),
        "startup_warning_steps_observed": startup_warning_steps_observed,
        "startup_rejected_steps": startup_rejected_steps,
        "nonstartup_failed_steps": nonstartup_failed_steps,
        "finite_all_steps": bool(finite_all),
        "flux_ratio_all_steps_ok": bool(flux_ok_all),
        "wall_flux_all_steps_ok": bool(wall_ok_all),
        "steps_completed": int(len(history)),
        "final_step": history[-1],
        "worst_step": worst,
    }


def _build_viscous_predictor_audit_from_history(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not history:
        return {"steps": [], "summary": {}}
    steps = []
    for r in history:
        steps.append(
            {
                "step": int(r.get("step", -1)),
                "flow_mode": str(r.get("flow_mode", "")),
                "viscous_predictor_mode": str(r.get("viscous_predictor_mode", "none")),
                "viscous_predictor_outlet_contract_mode": str(
                    r.get("viscous_predictor_outlet_contract_mode", "")
                ),
                "viscous_predictor_used": bool(r.get("viscous_predictor_used", False)),
                "divergence_before_predictor_max": float(
                    r.get("divergence_before_predictor_max", 0.0)
                ),
                "divergence_before_predictor_l2": float(
                    r.get("divergence_before_predictor_l2", 0.0)
                ),
                "divergence_after_predictor_before_boundary_contract_max": float(
                    r.get(
                        "divergence_after_predictor_before_boundary_contract_max", 0.0
                    )
                ),
                "divergence_after_predictor_before_boundary_contract_l2": float(
                    r.get("divergence_after_predictor_before_boundary_contract_l2", 0.0)
                ),
                "divergence_after_boundary_contract_before_projection_max": float(
                    r.get(
                        "divergence_after_boundary_contract_before_projection_max", 0.0
                    )
                ),
                "divergence_after_boundary_contract_before_projection_l2": float(
                    r.get(
                        "divergence_after_boundary_contract_before_projection_l2", 0.0
                    )
                ),
                "divergence_after_projection_max": float(
                    r.get("divergence_after_projection_max", 0.0)
                ),
                "divergence_after_projection_l2": float(
                    r.get("divergence_after_projection_l2", 0.0)
                ),
                "net_boundary_flux_before_predictor": float(
                    r.get("net_boundary_flux_before_predictor", 0.0)
                ),
                "net_boundary_flux_after_predictor_before_contract": float(
                    r.get("net_boundary_flux_after_predictor_before_contract", 0.0)
                ),
                "net_boundary_flux_after_contract": float(
                    r.get("net_boundary_flux_after_contract", 0.0)
                ),
                "net_boundary_flux_after_projection": float(
                    r.get("net_boundary_flux_after_projection", 0.0)
                ),
                "wall_flux_max_after_predictor_before_contract": float(
                    r.get("wall_flux_max_after_predictor_before_contract", 0.0)
                ),
                "wall_flux_max_after_contract": float(
                    r.get("wall_flux_max_after_contract", 0.0)
                ),
                "wall_flux_max_after_projection": float(
                    r.get("wall_flux_max_after_projection", 0.0)
                ),
                "outlet_inlet_ratio_before_predictor": float(
                    r.get("outlet_inlet_ratio_before_predictor", 0.0)
                ),
                "outlet_inlet_ratio_after_predictor_before_contract": float(
                    r.get("outlet_inlet_ratio_after_predictor_before_contract", 0.0)
                ),
                "outlet_inlet_ratio_after_contract": float(
                    r.get("outlet_inlet_ratio_after_contract", 0.0)
                ),
                "outlet_inlet_ratio_after_projection": float(
                    r.get("outlet_inlet_flux_ratio", 0.0)
                ),
                "kinetic_energy_before_predictor": float(
                    r.get("kinetic_energy_before_predictor", 0.0)
                ),
                "kinetic_energy_after_predictor": float(
                    r.get("kinetic_energy_after_predictor", 0.0)
                ),
                "kinetic_energy_after_contract": float(
                    r.get("kinetic_energy_after_contract", 0.0)
                ),
                "kinetic_energy_after_projection": float(
                    r.get("kinetic_energy_after_projection", 0.0)
                ),
                "velocity_delta_predictor_max": float(
                    r.get("viscous_delta_velocity_max", 0.0)
                ),
                "velocity_delta_predictor_l2": float(
                    r.get("viscous_delta_velocity_l2", 0.0)
                ),
                "face_flux_delta_predictor_max": float(
                    r.get("face_flux_delta_predictor_max", 0.0)
                ),
                "face_flux_delta_predictor_l2": float(
                    r.get("face_flux_delta_predictor_l2", 0.0)
                ),
                "face_flux_delta_contract_max": float(
                    r.get("face_flux_delta_contract_max", 0.0)
                ),
                "face_flux_delta_contract_l2": float(
                    r.get("face_flux_delta_contract_l2", 0.0)
                ),
                "wall_velocity_boundary_mode": str(
                    r.get("wall_velocity_boundary_mode", "")
                ),
                "wall_velocity_boundary_implementation": str(
                    r.get("wall_velocity_boundary_implementation", "")
                ),
                "wall_tangential_no_slip_strength": float(
                    r.get("wall_tangential_no_slip_strength", 0.0)
                ),
                "wall_tangential_no_slip_strength_ramp_enabled": bool(
                    r.get("wall_tangential_no_slip_strength_ramp_enabled", False)
                ),
                "wall_tangential_no_slip_strength_ramp_start": float(
                    r.get("wall_tangential_no_slip_strength_ramp_start", 0.0)
                ),
                "wall_tangential_no_slip_strength_ramp_target": float(
                    r.get("wall_tangential_no_slip_strength_ramp_target", 0.0)
                ),
                "wall_tangential_no_slip_strength_ramp_steps": int(
                    r.get("wall_tangential_no_slip_strength_ramp_steps", 0)
                ),
                "wall_tangential_shear_face_flux_requested": bool(
                    r.get("wall_tangential_shear_face_flux_requested", False)
                ),
                "wall_tangential_cell_velocity_momentum_enabled": bool(
                    r.get("wall_tangential_cell_velocity_momentum_enabled", False)
                ),
                "wall_tangential_operator_active_cells": int(
                    r.get("wall_tangential_operator_active_cells", 0)
                ),
                "wall_tangential_operator_max_abs": float(
                    r.get("wall_tangential_operator_max_abs", 0.0)
                ),
                "wall_tangential_operator_trace_mean": float(
                    r.get("wall_tangential_operator_trace_mean", 0.0)
                ),
                "wall_tangential_operator_trace_max": float(
                    r.get("wall_tangential_operator_trace_max", 0.0)
                ),
                "wall_tangential_operator_effective_nu_dt_max_abs": float(
                    r.get("wall_tangential_operator_effective_nu_dt_max_abs", 0.0)
                ),
                "wall_tangential_operator_effective_nu_subdt_max_abs": float(
                    r.get("wall_tangential_operator_effective_nu_subdt_max_abs", 0.0)
                ),
                "wall_flux_stokes_resistance_enabled": bool(
                    r.get("wall_flux_stokes_resistance_enabled", False)
                ),
                "wall_flux_stokes_resistance_strength": float(
                    r.get("wall_flux_stokes_resistance_strength", 0.0)
                ),
                "wall_flux_stokes_resistance_active_faces": int(
                    r.get("wall_flux_stokes_resistance_active_faces", 0)
                ),
                "wall_flux_stokes_resistance_solver_iterations": int(
                    r.get("wall_flux_stokes_resistance_solver_iterations", 0)
                ),
                "wall_flux_stokes_resistance_solver_converged": bool(
                    r.get("wall_flux_stokes_resistance_solver_converged", True)
                ),
                "wall_flux_stokes_resistance_solver_residual_l2": float(
                    r.get("wall_flux_stokes_resistance_solver_residual_l2", 0.0)
                ),
                "wall_flux_stokes_resistance_solver_method": str(
                    r.get("wall_flux_stokes_resistance_solver_method", "")
                ),
                "wall_tangential_shear_face_flux_enabled": bool(
                    r.get("wall_tangential_shear_face_flux_enabled", False)
                ),
                "wall_tangential_shear_face_flux_applications": int(
                    r.get("wall_tangential_shear_face_flux_applications", 0)
                ),
                "wall_tangential_shear_face_flux_active_cells": int(
                    r.get("wall_tangential_shear_face_flux_active_cells", 0)
                ),
                "wall_tangential_shear_face_flux_delta_l2": float(
                    r.get("wall_tangential_shear_face_flux_delta_l2", 0.0)
                ),
                "wall_tangential_shear_face_flux_wall_speed_mean_before": float(
                    r.get("wall_tangential_shear_face_flux_wall_speed_mean_before", 0.0)
                ),
                "wall_tangential_shear_face_flux_wall_speed_mean_after": float(
                    r.get("wall_tangential_shear_face_flux_wall_speed_mean_after", 0.0)
                ),
            }
        )
    arr_before = np.asarray(
        [float(s["divergence_before_predictor_l2"]) for s in steps], dtype=np.float64
    )
    arr_pred = np.asarray(
        [
            float(s["divergence_after_predictor_before_boundary_contract_l2"])
            for s in steps
        ],
        dtype=np.float64,
    )
    arr_contract = np.asarray(
        [
            float(s["divergence_after_boundary_contract_before_projection_l2"])
            for s in steps
        ],
        dtype=np.float64,
    )
    arr_proj = np.asarray(
        [float(s["divergence_after_projection_l2"]) for s in steps], dtype=np.float64
    )
    return {
        "steps": steps,
        "summary": {
            "steps_count": int(len(steps)),
            "mean_divergence_l2_before_predictor": float(np.mean(arr_before))
            if arr_before.size
            else 0.0,
            "mean_divergence_l2_after_predictor_before_contract": float(
                np.mean(arr_pred)
            )
            if arr_pred.size
            else 0.0,
            "mean_divergence_l2_after_contract_before_projection": float(
                np.mean(arr_contract)
            )
            if arr_contract.size
            else 0.0,
            "mean_divergence_l2_after_projection": float(np.mean(arr_proj))
            if arr_proj.size
            else 0.0,
            "max_divergence_l2_after_predictor_before_contract": float(np.max(arr_pred))
            if arr_pred.size
            else 0.0,
            "max_divergence_l2_after_contract_before_projection": float(
                np.max(arr_contract)
            )
            if arr_contract.size
            else 0.0,
            "max_divergence_l2_after_projection": float(np.max(arr_proj))
            if arr_proj.size
            else 0.0,
        },
    }


def _run_startup_bootstrap(
    *,
    mesh,
    state0,
    cfg,
    flow_mode: str,
    requested_flow_dt: float,
    flow_dt_mode: str,
    flow_dt_min: float,
    flow_dt_max: float,
    convective_cfl_target: float,
    acceptance_thresholds: dict[str, float],
    wall_strength_start: float,
    max_steps: int = 20,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        apply_tetra_convective_predictor,
        apply_tetra_stokes_viscous_predictor,
        compute_tetra_convective_cfl_rate,
        compute_tetra_flux_divergence,
        solve_tetra_pressure_projection,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
        resolve_inlet_face_groups,
    )

    state_curr = state0
    history: list[dict[str, Any]] = []
    inlet_groups = resolve_inlet_face_groups(mesh)
    left_faces = np.asarray(inlet_groups.get("left_faces", []), dtype=np.int64)
    right_faces = np.asarray(inlet_groups.get("right_faces", []), dtype=np.int64)
    initial_divergence = compute_tetra_flux_divergence(
        mesh,
        state0.face_flux,
        left_inlet_faces=left_faces,
        right_inlet_faces=right_faces,
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
    )
    for step_idx in range(1, int(max_steps) + 1):
        cfl_rate_diag = compute_tetra_convective_cfl_rate(
            mesh, np.asarray(state_curr.face_flux, dtype=np.float64)
        )
        cfl_rate_max = float(cfl_rate_diag.get("cfl_rate_max", 0.0))
        dt_candidate = float(requested_flow_dt)
        if (
            str(flow_dt_mode) == "auto_cfl"
            and str(flow_mode) == "navier_stokes_projection_debug"
        ):
            dt_from_cfl = float(
                convective_cfl_target / max(cfl_rate_max, 1e-30)
                if cfl_rate_max > 0.0
                else flow_dt_max
            )
            auto_dt_floor = float(
                max(1e-12, min(float(flow_dt_min), 0.05 * dt_from_cfl))
            )
            dt_candidate = float(
                min(max(dt_from_cfl, auto_dt_floor), float(flow_dt_max))
            )
        step_cfg = replace(
            cfg,
            projection_dt=float(dt_candidate),
            wall_tangential_no_slip_strength=float(wall_strength_start),
        )
        state_after_conv = state_curr
        use_convective = bool(
            (str(flow_mode) == "navier_stokes_projection_debug")
            or step_cfg.enable_convective_predictor
        )
        if use_convective:
            state_after_conv = apply_tetra_convective_predictor(
                mesh,
                state_curr,
                step_cfg,
                flow_dt=float(dt_candidate),
            )
        convective_diagnostics = (
            state_after_conv.diagnostics.get("convective_predictor", {})
            if use_convective
            else {}
        )
        state_for_projection = state_after_conv
        use_viscous = str(flow_mode) in {
            "stokes_viscous_projection",
            "navier_stokes_projection_debug",
        }
        if use_viscous:
            state_for_projection = apply_tetra_stokes_viscous_predictor(
                mesh,
                state_after_conv,
                step_cfg,
                flow_dt=float(dt_candidate),
            )
        viscous_diagnostics = (
            state_for_projection.diagnostics.get("viscous_predictor", {})
            if use_viscous
            else {}
        )
        state_next = solve_tetra_pressure_projection(
            mesh,
            state_for_projection,
            step_cfg,
        )
        projection = dict(state_next.diagnostics.get("projection", {}))
        pressure = dict(state_next.diagnostics.get("pressure", {}))
        backend_execution = dict(state_next.diagnostics.get("backend_execution", {}))
        acceptance = _projection_acceptance(
            projection=projection,
            pressure=pressure,
            backend_execution=backend_execution,
            thresholds=acceptance_thresholds,
        )
        qualification_row = _build_projection_acceptance_step_record(
            step=step_idx,
            acceptance=acceptance,
            projection=projection,
            pressure=pressure,
            used_flow_dt=float(dt_candidate),
        )
        qualification_row["iteration_kind"] = "pseudo_time_qualification"
        qualification_row["physical_time_advanced"] = 0.0
        qualification_row.update(
            {
                "convective_predictor_used": bool(
                    convective_diagnostics.get("convective_predictor_used", False)
                ),
                "convective_torch_cuda_used": bool(
                    convective_diagnostics.get("convective_torch_cuda_used", False)
                ),
                "convective_numpy_fallback_reason": str(
                    convective_diagnostics.get("convective_numpy_fallback_reason", "")
                ),
                "viscous_predictor_used": bool(
                    viscous_diagnostics.get("viscous_predictor_used", False)
                ),
                "viscous_torch_cuda_used": bool(
                    viscous_diagnostics.get("viscous_torch_cuda_used", False)
                ),
                "viscous_numpy_fallback_reason": str(
                    viscous_diagnostics.get("viscous_numpy_fallback_reason", "")
                ),
            }
        )
        history.append(qualification_row)
        state_curr = state_next
        consecutive_accepted = 0
        for row in reversed(history):
            if not bool(row.get("projection_solved", False)):
                break
            consecutive_accepted += 1
        if (
            consecutive_accepted
            >= STARTUP_BOOTSTRAP_REQUIRED_CONSECUTIVE_ACCEPTED_STEPS
        ):
            break
    return (
        state_curr,
        history,
        _summarize_startup_bootstrap(
            bootstrap_history=history,
            initial_divergence=initial_divergence,
            requested_max_steps=int(max_steps),
            bootstrap_required=True,
        ),
    )


def _convective_history_stats(
    history: list[dict[str, Any]],
    *,
    convective_cfl_acceptance_eps: float = 0.0,
) -> dict[str, Any]:
    if not history:
        return {
            "steps_count": 0,
            "convective_cfl_limit": 0.0,
            "raw_cfl_max_max": 0.0,
            "raw_cfl_max_mean": 0.0,
            "raw_cfl_p95_max": 0.0,
            "raw_cfl_p95_mean": 0.0,
            "raw_cfl_before_dt_selection_max": 0.0,
            "raw_cfl_before_dt_selection_p95": 0.0,
            "raw_cfl_after_dt_selection_max": 0.0,
            "raw_cfl_after_dt_selection_p95": 0.0,
            "effective_cfl_max_max": 0.0,
            "effective_cfl_max_mean": 0.0,
            "effective_cfl_p95_max": 0.0,
            "effective_cfl_p95_mean": 0.0,
            "raw_cfl_warning_any": False,
            "effective_cfl_warning_any": False,
            "raw_cfl_warning_step_count": 0,
            "effective_cfl_warning_step_count": 0,
            "effective_cfl_limit_excess_max": 0.0,
            "effective_cfl_warning_steps": [],
            "raw_cfl_after_dt_selection_warning_steps": [],
            "auto_damping_used_any": False,
            "auto_damping_step_count": 0,
            "damping_requested_min": 0.0,
            "damping_requested_mean": 0.0,
            "damping_requested_max": 0.0,
            "damping_effective_min": 0.0,
            "damping_effective_mean": 0.0,
            "damping_effective_max": 0.0,
            "dt_effective_min": 0.0,
            "dt_effective_mean": 0.0,
            "dt_effective_max": 0.0,
            "used_dt_min": 0.0,
            "used_dt_mean": 0.0,
            "used_dt_max": 0.0,
            "auto_dt_scale_min": 0.0,
            "auto_dt_scale_mean": 0.0,
            "auto_dt_scale_max": 0.0,
            "auto_dt_floor_min": 0.0,
            "auto_dt_floor_mean": 0.0,
            "auto_dt_floor_max": 0.0,
            "auto_dt_min_hit_any": False,
            "auto_dt_max_hit_any": False,
            "substepping_used_any": False,
            "substepping_step_count": 0,
            "substep_count_mean": 0.0,
            "substep_count_max": 0,
            "substep_cap_hit_any": False,
            "substep_cfl_max_max": 0.0,
            "substep_cfl_p95_max": 0.0,
            "stabilization_modes_used": [],
            "strict_warning_step_count": 0,
            "epsilon_aware_warning_step_count": 0,
            "strict_warning_steps": [],
            "epsilon_aware_warning_steps": [],
            "warning_aggregation_mode": "epsilon_aware",
            "warning_aggregation_consistent": True,
        }

    eps = max(float(convective_cfl_acceptance_eps), 0.0)
    arr_limit = np.asarray(
        [float(r.get("convective_cfl_limit", 0.0)) for r in history], dtype=np.float64
    )
    limit = float(arr_limit[0]) if arr_limit.size else 0.0
    arr_raw_max = np.asarray(
        [
            float(
                r.get(
                    "convective_cfl_raw_max",
                    r.get("convective_cfl_max", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_raw_p95 = np.asarray(
        [
            float(
                r.get(
                    "convective_cfl_raw_p95",
                    r.get("convective_cfl_p95", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_eff_max = np.asarray(
        [
            float(
                r.get(
                    "convective_cfl_effective_max",
                    r.get("convective_cfl_max", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_eff_p95 = np.asarray(
        [
            float(
                r.get(
                    "convective_cfl_effective_p95",
                    r.get("convective_cfl_p95", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_damp_req = np.asarray(
        [
            float(
                r.get(
                    "convective_predictor_damping_requested",
                    r.get("convective_predictor_damping", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_damp_eff = np.asarray(
        [float(r.get("convective_predictor_damping_effective", 0.0)) for r in history],
        dtype=np.float64,
    )
    arr_dt_eff = np.asarray(
        [float(r.get("convective_dt_effective", 0.0)) for r in history],
        dtype=np.float64,
    )
    arr_used_dt = np.asarray(
        [
            float(
                r.get(
                    "used_flow_dt",
                    r.get("convective_dt", 0.0),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_auto_dt_scale = np.asarray(
        [float(r.get("auto_dt_scale_factor", 1.0)) for r in history], dtype=np.float64
    )
    arr_auto_dt_floor = np.asarray(
        [float(r.get("auto_dt_floor_used", 0.0)) for r in history], dtype=np.float64
    )
    arr_raw_before = np.asarray(
        [float(r.get("raw_cfl_max_before_dt_selection", 0.0)) for r in history],
        dtype=np.float64,
    )
    arr_raw_after = np.asarray(
        [
            float(
                r.get(
                    "raw_cfl_max_after_dt_selection",
                    r.get("convective_cfl_raw_max", r.get("convective_cfl_max", 0.0)),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    arr_raw_after_p95 = np.asarray(
        [
            float(
                r.get(
                    "raw_cfl_p95_after_dt_selection",
                    r.get("convective_cfl_raw_p95", r.get("convective_cfl_p95", 0.0)),
                )
            )
            for r in history
        ],
        dtype=np.float64,
    )
    auto = np.asarray(
        [bool(r.get("convective_auto_damping_used", False)) for r in history],
        dtype=bool,
    )
    substepping = np.asarray(
        [bool(r.get("convective_substepping_used", False)) for r in history], dtype=bool
    )
    arr_substeps = np.asarray(
        [int(r.get("convective_substep_count", 1)) for r in history], dtype=np.int64
    )
    arr_substep_cfl_max = np.asarray(
        [float(r.get("convective_cfl_per_substep_max", 0.0)) for r in history],
        dtype=np.float64,
    )
    arr_substep_cfl_p95 = np.asarray(
        [float(r.get("convective_cfl_per_substep_p95", 0.0)) for r in history],
        dtype=np.float64,
    )
    substep_cap = np.asarray(
        [bool(r.get("convective_substep_cap_hit", False)) for r in history], dtype=bool
    )
    auto_dt_min_hits = np.asarray(
        [bool(r.get("auto_dt_min_hit", False)) for r in history], dtype=bool
    )
    auto_dt_max_hits = np.asarray(
        [bool(r.get("auto_dt_max_hit", False)) for r in history], dtype=bool
    )
    stab_modes = sorted(
        {str(r.get("convective_stabilization_mode", "auto_damping")) for r in history}
    )
    raw_warn = arr_raw_max > arr_limit
    eff_threshold = arr_limit * (1.0 + eps)
    eff_warn = arr_eff_max > eff_threshold
    raw_after_warn = arr_raw_after > eff_threshold
    strict_warn = np.asarray(
        [
            bool(
                r.get(
                    "convective_cfl_warning_effective",
                    r.get("convective_cfl_warning", False),
                )
            )
            or bool(
                float(
                    r.get(
                        "raw_cfl_max_after_dt_selection",
                        r.get(
                            "convective_cfl_raw_max",
                            r.get("convective_cfl_max", 0.0),
                        ),
                    )
                )
                > float(r.get("convective_cfl_limit", 0.0))
            )
            for r in history
        ],
        dtype=bool,
    )
    eps_warn = np.asarray(
        [
            bool(r.get("effective_cfl_warning_with_eps", bool(eff_warn[i])))
            or bool(
                r.get(
                    "raw_cfl_after_dt_selection_warning_with_eps",
                    bool(raw_after_warn[i]),
                )
            )
            for i, r in enumerate(history)
        ],
        dtype=bool,
    )
    arr_eff_excess = arr_eff_max - arr_limit
    eff_warn_steps = [
        int(history[i].get("step", i + 1)) for i in np.flatnonzero(eff_warn).tolist()
    ]
    raw_after_warn_steps = [
        int(history[i].get("step", i + 1))
        for i in np.flatnonzero(raw_after_warn).tolist()
    ]
    strict_warn_steps = [
        int(history[i].get("step", i + 1)) for i in np.flatnonzero(strict_warn).tolist()
    ]
    eps_warn_steps = [
        int(history[i].get("step", i + 1)) for i in np.flatnonzero(eps_warn).tolist()
    ]
    return {
        "steps_count": int(len(history)),
        "convective_cfl_limit": float(limit),
        "raw_cfl_max_max": float(np.max(arr_raw_max)) if arr_raw_max.size else 0.0,
        "raw_cfl_max_mean": float(np.mean(arr_raw_max)) if arr_raw_max.size else 0.0,
        "raw_cfl_p95_max": float(np.max(arr_raw_p95)) if arr_raw_p95.size else 0.0,
        "raw_cfl_p95_mean": float(np.mean(arr_raw_p95)) if arr_raw_p95.size else 0.0,
        "raw_cfl_before_dt_selection_max": float(np.max(arr_raw_before))
        if arr_raw_before.size
        else 0.0,
        "raw_cfl_before_dt_selection_p95": float(np.percentile(arr_raw_before, 95.0))
        if arr_raw_before.size
        else 0.0,
        "raw_cfl_after_dt_selection_max": float(np.max(arr_raw_after))
        if arr_raw_after.size
        else 0.0,
        "raw_cfl_after_dt_selection_p95": float(np.max(arr_raw_after_p95))
        if arr_raw_after_p95.size
        else 0.0,
        "effective_cfl_max_max": float(np.max(arr_eff_max))
        if arr_eff_max.size
        else 0.0,
        "effective_cfl_max_mean": float(np.mean(arr_eff_max))
        if arr_eff_max.size
        else 0.0,
        "effective_cfl_p95_max": float(np.max(arr_eff_p95))
        if arr_eff_p95.size
        else 0.0,
        "effective_cfl_p95_mean": float(np.mean(arr_eff_p95))
        if arr_eff_p95.size
        else 0.0,
        "raw_cfl_warning_any": bool(np.any(raw_warn)),
        "effective_cfl_warning_any": bool(np.any(eff_warn)),
        "raw_cfl_warning_step_count": int(np.count_nonzero(raw_warn)),
        "effective_cfl_warning_step_count": int(np.count_nonzero(eff_warn)),
        "effective_cfl_limit_excess_max": float(np.max(np.maximum(arr_eff_excess, 0.0)))
        if arr_eff_excess.size
        else 0.0,
        "effective_cfl_warning_steps": eff_warn_steps,
        "raw_cfl_after_dt_selection_warning_steps": raw_after_warn_steps,
        "auto_damping_used_any": bool(np.any(auto)),
        "auto_damping_step_count": int(np.count_nonzero(auto)),
        "damping_requested_min": float(np.min(arr_damp_req))
        if arr_damp_req.size
        else 0.0,
        "damping_requested_mean": float(np.mean(arr_damp_req))
        if arr_damp_req.size
        else 0.0,
        "damping_requested_max": float(np.max(arr_damp_req))
        if arr_damp_req.size
        else 0.0,
        "damping_effective_min": float(np.min(arr_damp_eff))
        if arr_damp_eff.size
        else 0.0,
        "damping_effective_mean": float(np.mean(arr_damp_eff))
        if arr_damp_eff.size
        else 0.0,
        "damping_effective_max": float(np.max(arr_damp_eff))
        if arr_damp_eff.size
        else 0.0,
        "dt_effective_min": float(np.min(arr_dt_eff)) if arr_dt_eff.size else 0.0,
        "dt_effective_mean": float(np.mean(arr_dt_eff)) if arr_dt_eff.size else 0.0,
        "dt_effective_max": float(np.max(arr_dt_eff)) if arr_dt_eff.size else 0.0,
        "used_dt_min": float(np.min(arr_used_dt)) if arr_used_dt.size else 0.0,
        "used_dt_mean": float(np.mean(arr_used_dt)) if arr_used_dt.size else 0.0,
        "used_dt_max": float(np.max(arr_used_dt)) if arr_used_dt.size else 0.0,
        "auto_dt_scale_min": float(np.min(arr_auto_dt_scale))
        if arr_auto_dt_scale.size
        else 0.0,
        "auto_dt_scale_mean": float(np.mean(arr_auto_dt_scale))
        if arr_auto_dt_scale.size
        else 0.0,
        "auto_dt_scale_max": float(np.max(arr_auto_dt_scale))
        if arr_auto_dt_scale.size
        else 0.0,
        "auto_dt_floor_min": float(np.min(arr_auto_dt_floor))
        if arr_auto_dt_floor.size
        else 0.0,
        "auto_dt_floor_mean": float(np.mean(arr_auto_dt_floor))
        if arr_auto_dt_floor.size
        else 0.0,
        "auto_dt_floor_max": float(np.max(arr_auto_dt_floor))
        if arr_auto_dt_floor.size
        else 0.0,
        "auto_dt_min_hit_any": bool(np.any(auto_dt_min_hits)),
        "auto_dt_max_hit_any": bool(np.any(auto_dt_max_hits)),
        "substepping_used_any": bool(np.any(substepping)),
        "substepping_step_count": int(np.count_nonzero(substepping)),
        "substep_count_mean": float(np.mean(arr_substeps))
        if arr_substeps.size
        else 0.0,
        "substep_count_max": int(np.max(arr_substeps)) if arr_substeps.size else 0,
        "substep_cap_hit_any": bool(np.any(substep_cap)),
        "substep_cfl_max_max": float(np.max(arr_substep_cfl_max))
        if arr_substep_cfl_max.size
        else 0.0,
        "substep_cfl_p95_max": float(np.max(arr_substep_cfl_p95))
        if arr_substep_cfl_p95.size
        else 0.0,
        "stabilization_modes_used": stab_modes,
        "strict_warning_step_count": int(np.count_nonzero(strict_warn)),
        "epsilon_aware_warning_step_count": int(np.count_nonzero(eps_warn)),
        "strict_warning_steps": strict_warn_steps,
        "epsilon_aware_warning_steps": eps_warn_steps,
        "warning_aggregation_mode": "epsilon_aware",
        "warning_aggregation_consistent": bool(
            int(np.count_nonzero(eps_warn)) <= int(np.count_nonzero(strict_warn))
        ),
    }


def _build_navier_stokes_prototype_audit(
    history: list[dict[str, Any]],
    *,
    convective_cfl_acceptance_eps: float = 0.0,
) -> dict[str, Any]:
    if not history:
        return {"steps": [], "summary": {}}
    steps: list[dict[str, Any]] = []
    for r in history:
        steps.append(
            {
                "step": int(r.get("step", -1)),
                "flow_mode": str(r.get("flow_mode", "")),
                "convective_predictor_used": bool(
                    r.get("convective_predictor_used", False)
                ),
                "convective_stabilization_mode": str(
                    r.get("convective_stabilization_mode", "auto_damping")
                ),
                "convective_substep_boundary_contract_mode": str(
                    r.get("convective_substep_boundary_contract_mode", "end_only")
                ),
                "convective_cfl_limit": float(r.get("convective_cfl_limit", 0.0)),
                "convective_cfl_raw_max": float(
                    r.get("convective_cfl_raw_max", r.get("convective_cfl_max", 0.0))
                ),
                "convective_cfl_raw_p95": float(
                    r.get("convective_cfl_raw_p95", r.get("convective_cfl_p95", 0.0))
                ),
                "convective_cfl_effective_max": float(
                    r.get(
                        "convective_cfl_effective_max",
                        r.get("convective_cfl_max", 0.0),
                    )
                ),
                "convective_cfl_effective_p95": float(
                    r.get(
                        "convective_cfl_effective_p95",
                        r.get("convective_cfl_p95", 0.0),
                    )
                ),
                "convective_cfl_warning_raw": bool(
                    r.get(
                        "convective_cfl_warning_raw",
                        r.get("convective_cfl_warning", False),
                    )
                ),
                "convective_cfl_warning_effective": bool(
                    r.get(
                        "convective_cfl_warning_effective",
                        r.get("convective_cfl_warning", False),
                    )
                ),
                "convective_predictor_damping_requested": float(
                    r.get(
                        "convective_predictor_damping_requested",
                        r.get("convective_predictor_damping", 0.0),
                    )
                ),
                "convective_predictor_damping_effective": float(
                    r.get("convective_predictor_damping_effective", 0.0)
                ),
                "convective_auto_damping_used": bool(
                    r.get("convective_auto_damping_used", False)
                ),
                "convective_auto_damping_reason": str(
                    r.get("convective_auto_damping_reason", "")
                ),
                "convective_substepping_used": bool(
                    r.get("convective_substepping_used", False)
                ),
                "convective_substep_count": int(r.get("convective_substep_count", 1)),
                "convective_substep_count_unclamped": int(
                    r.get("convective_substep_count_unclamped", 1)
                ),
                "convective_substep_cap_hit": bool(
                    r.get("convective_substep_cap_hit", False)
                ),
                "convective_substep_dt": float(r.get("convective_substep_dt", 0.0)),
                "convective_cfl_per_substep_max": float(
                    r.get("convective_cfl_per_substep_max", 0.0)
                ),
                "convective_cfl_per_substep_p95": float(
                    r.get("convective_cfl_per_substep_p95", 0.0)
                ),
                "convective_substepping_runtime_seconds": float(
                    r.get("convective_substepping_runtime_seconds", 0.0)
                ),
                "convective_dt_effective": float(r.get("convective_dt_effective", 0.0)),
                "convective_delta_velocity_max": float(
                    r.get("convective_delta_velocity_max", 0.0)
                ),
                "convective_delta_velocity_l2": float(
                    r.get("convective_delta_velocity_l2", 0.0)
                ),
                "kinetic_energy_before_convection": float(
                    r.get("kinetic_energy_before_convection", 0.0)
                ),
                "kinetic_energy_after_convection": float(
                    r.get("kinetic_energy_after_convection", 0.0)
                ),
                "kinetic_energy_after_projection": float(
                    r.get("kinetic_energy_after_projection", 0.0)
                ),
                "divergence_before_convection_max": float(
                    r.get("divergence_before_convection_max", 0.0)
                ),
                "divergence_before_convection_l2": float(
                    r.get("divergence_before_convection_l2", 0.0)
                ),
                "divergence_after_convection_before_projection_max": float(
                    r.get("divergence_after_convection_before_projection_max", 0.0)
                ),
                "divergence_after_convection_before_projection_l2": float(
                    r.get("divergence_after_convection_before_projection_l2", 0.0)
                ),
                "final_divergence_max_abs": float(
                    r.get("final_divergence_max_abs", 0.0)
                ),
                "final_divergence_l2": float(r.get("final_divergence_l2", 0.0)),
                "outlet_inlet_flux_ratio": float(r.get("outlet_inlet_flux_ratio", 0.0)),
                "wall_flux_max_abs_after": float(r.get("wall_flux_max_abs_after", 0.0)),
                "projection_solved": bool(r.get("projection_solved", False)),
                "finite_fields": bool(r.get("finite_fields", False)),
            }
        )
    arr_du = np.asarray(
        [float(s["convective_delta_velocity_l2"]) for s in steps], dtype=np.float64
    )
    arr_div = np.asarray(
        [float(s["divergence_after_convection_before_projection_l2"]) for s in steps],
        dtype=np.float64,
    )
    conv_summary = _convective_history_stats(
        steps, convective_cfl_acceptance_eps=convective_cfl_acceptance_eps
    )
    return {
        "steps": steps,
        "summary": {
            "steps_count": int(len(steps)),
            "convective_predictor_used_any": bool(
                any(bool(s["convective_predictor_used"]) for s in steps)
            ),
            "convective_cfl_max_max": float(conv_summary.get("raw_cfl_max_max", 0.0)),
            "convective_cfl_max_mean": float(conv_summary.get("raw_cfl_max_mean", 0.0)),
            "convective_delta_velocity_l2_max": float(np.max(arr_du))
            if arr_du.size
            else 0.0,
            "convective_delta_velocity_l2_mean": float(np.mean(arr_du))
            if arr_du.size
            else 0.0,
            "divergence_after_convection_before_projection_l2_max": float(
                np.max(arr_div)
            )
            if arr_div.size
            else 0.0,
            "divergence_after_convection_before_projection_l2_mean": float(
                np.mean(arr_div)
            )
            if arr_div.size
            else 0.0,
            **conv_summary,
        },
    }


def _collect_warning_aggregation(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "warning_step_count": 0,
            "strict_warning_step_count": 0,
            "epsilon_aware_warning_step_count": 0,
            "warning_aggregation_mode": "epsilon_aware",
            "warning_aggregation_consistent": True,
            "strict_warning_steps": [],
            "epsilon_aware_warning_steps": [],
        }
    strict_steps: list[int] = []
    eps_steps: list[int] = []
    for idx, row in enumerate(history):
        step = int(row.get("step", idx + 1))
        strict_flag = bool(
            row.get(
                "convective_cfl_warning_effective",
                row.get("convective_cfl_warning", False),
            )
        ) or bool(
            row.get("raw_cfl_max_after_dt_selection", 0.0)
            > row.get("convective_cfl_limit", float("inf"))
        )
        eps_flag = bool(row.get("effective_cfl_warning_with_eps", False)) or bool(
            row.get("raw_cfl_after_dt_selection_warning_with_eps", False)
        )
        if strict_flag:
            strict_steps.append(step)
        if eps_flag:
            eps_steps.append(step)
    strict_steps = sorted(set(strict_steps))
    eps_steps = sorted(set(eps_steps))
    return {
        "warning_step_count": int(len(eps_steps)),
        "strict_warning_step_count": int(len(strict_steps)),
        "epsilon_aware_warning_step_count": int(len(eps_steps)),
        "warning_aggregation_mode": "epsilon_aware",
        "warning_aggregation_consistent": bool(len(eps_steps) <= len(strict_steps)),
        "strict_warning_steps": strict_steps,
        "epsilon_aware_warning_steps": eps_steps,
    }


def _collect_boundary_flux_policy(
    *,
    projection: dict[str, Any],
    flow_mode: str,
) -> dict[str, Any]:
    mode = str(projection.get("outlet_projection_mode", "outlet_pressure_dirichlet"))
    rescale_used = bool(projection.get("outlet_flux_rescale_used", False))
    rescale_factor = float(projection.get("outlet_flux_rescale_factor", 1.0))
    rescale_reason = str(projection.get("outlet_flux_rescale_reason", ""))
    nonphysical_flux_fix_used = bool(
        projection.get(
            "nonphysical_flux_fix_used",
            mode in {"outlet_flux_preserve", "outlet_mass_balance_rescale"},
        )
    )
    boundary_contract_used = bool(
        flow_mode in {"stokes_viscous_projection", "navier_stokes_projection_debug"}
    )
    boundary_contract_stage = (
        "viscous_predictor_post_update_enforcement"
        if boundary_contract_used
        else "none"
    )
    boundary_contract_is_physical_baseline = bool(
        boundary_contract_used
        and (not nonphysical_flux_fix_used)
        and mode == "outlet_pressure_dirichlet"
    )
    return {
        "boundary_flux_policy": mode,
        "outlet_flux_rescale_used": bool(rescale_used),
        "outlet_flux_rescale_factor": float(rescale_factor),
        "outlet_flux_rescale_reason": str(rescale_reason),
        "boundary_contract_used": bool(boundary_contract_used),
        "boundary_contract_stage": str(boundary_contract_stage),
        "boundary_contract_is_physical_baseline": bool(
            boundary_contract_is_physical_baseline
        ),
        "nonphysical_flux_fix_used": bool(nonphysical_flux_fix_used),
    }


def _collect_stabilization_audit(
    *,
    conv_stats: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    capped_fracs = np.asarray(
        [float(r.get("capped_predictor_updates_fraction", 0.0)) for r in history],
        dtype=np.float64,
    )
    visc_cap_used_any = bool(np.any(capped_fracs > 0.0)) if capped_fracs.size else False
    visc_capped_updates_fraction_max = (
        float(np.max(capped_fracs)) if capped_fracs.size else 0.0
    )
    debug_reasons: list[str] = []
    conv_auto_used = bool(conv_stats.get("auto_damping_used_any", False))
    conv_substep_used = bool(conv_stats.get("substepping_used_any", False))
    conv_substep_cap = bool(conv_stats.get("substep_cap_hit_any", False))
    conv_damp_min = float(conv_stats.get("damping_effective_min", 0.0))
    if conv_auto_used:
        debug_reasons.append("convective auto damping used")
    if conv_substep_used:
        debug_reasons.append("convective substepping used")
    if conv_substep_cap:
        debug_reasons.append("convective substep cap hit")
    if conv_damp_min < (1.0 - 1e-12):
        debug_reasons.append("effective convective damping below 1.0")
    if visc_cap_used_any:
        debug_reasons.append("viscous predictor cap used")
    stabilization_is_physical_baseline = bool(len(debug_reasons) == 0)
    return {
        "convective_auto_damping_step_count": int(
            conv_stats.get("auto_damping_step_count", 0)
        ),
        "convective_auto_damping_used_any": bool(conv_auto_used),
        "convective_effective_damping_min": float(
            conv_stats.get("damping_effective_min", 0.0)
        ),
        "convective_effective_damping_mean": float(
            conv_stats.get("damping_effective_mean", 0.0)
        ),
        "convective_effective_damping_max": float(
            conv_stats.get("damping_effective_max", 0.0)
        ),
        "convective_substepping_used_any": bool(conv_substep_used),
        "convective_substep_cap_hit_any": bool(conv_substep_cap),
        "viscous_cap_used_any": bool(visc_cap_used_any),
        "viscous_capped_updates_fraction_max": float(visc_capped_updates_fraction_max),
        "stabilization_is_physical_baseline": bool(stabilization_is_physical_baseline),
        "stabilization_debug_only_reasons": debug_reasons,
    }


def _evaluate_ns_coupling_readiness(
    *,
    flow_progression_solved: bool,
    ready_for_long_ns_run_physical: bool,
    nonphysical_flux_fix_used: bool,
    convective_auto_damping_used_any: bool,
    convective_substep_cap_hit_any: bool,
    finite_fields: bool,
    wall_flux_max_abs_after: float,
    outlet_inlet_flux_ratio: float,
    final_div_l2: float,
    final_div_max_abs: float,
    wall_flux_abs_tolerance: float,
    outlet_inlet_flux_ratio_tolerance: float,
    projection_final_div_l2_tolerance: float,
    projection_final_div_max_tolerance: float,
    epsilon_aware_warning_step_count: int,
) -> dict[str, Any]:
    checks = {
        "flow_progression_solved": bool(flow_progression_solved),
        "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
        "no_nonphysical_flux_fix": bool(not nonphysical_flux_fix_used),
        "no_convective_auto_damping": bool(not convective_auto_damping_used_any),
        "no_convective_substep_cap_hit": bool(not convective_substep_cap_hit_any),
        "finite_fields": bool(finite_fields),
        "wall_flux_ok": bool(
            abs(float(wall_flux_max_abs_after)) <= float(wall_flux_abs_tolerance)
        ),
        "outlet_inlet_flux_ratio_ok": bool(
            abs(float(outlet_inlet_flux_ratio) - 1.0)
            <= float(outlet_inlet_flux_ratio_tolerance)
        ),
        "final_div_l2_ok": bool(
            float(final_div_l2) <= float(projection_final_div_l2_tolerance)
        ),
        "final_div_max_ok": bool(
            float(final_div_max_abs) <= float(projection_final_div_max_tolerance)
        ),
        "epsilon_aware_warning_step_count_zero": bool(
            int(epsilon_aware_warning_step_count) == 0
        ),
    }
    ready = bool(all(bool(v) for v in checks.values()))
    reason = "NS baseline is physically clean and ready for flow-to-transport coupling"
    if not ready:
        failing = [k for k, v in checks.items() if not bool(v)]
        reason = "not physically clean: " + ", ".join(failing)
    return {
        "ns_baseline_physical_clean": bool(ready),
        "ns_baseline_physical_clean_reason": str(reason),
        "ready_for_flow_to_transport_coupling": bool(ready),
        "ns_baseline_physical_clean_checks": checks,
    }


def _evaluate_convective_prototype_acceptance(
    *,
    history: list[dict[str, Any]],
    stokes_baseline: dict[str, Any] | None,
    outlet_inlet_flux_ratio_tolerance: float,
    wall_flux_abs_tolerance: float,
    inlet_speed: float = 0.0,
    velocity_growth_ratio_limit: float = 15.0,
) -> dict[str, Any]:
    if not history:
        return {
            "convective_prototype_accepted": False,
            "reason": "no progression history",
            "checks": {},
        }
    final = history[-1]
    finite_ok = bool(final.get("finite_fields", False))
    wall_ok = abs(float(final.get("wall_flux_max_abs_after", 0.0))) <= float(
        wall_flux_abs_tolerance
    )
    flux_ok = abs(float(final.get("outlet_inlet_flux_ratio", 0.0)) - 1.0) <= float(
        outlet_inlet_flux_ratio_tolerance
    )
    div_l2 = float(final.get("final_divergence_l2", float("inf")))
    div_max = float(final.get("final_divergence_max_abs", float("inf")))
    if stokes_baseline is not None:
        st_l2 = float(stokes_baseline.get("final_divergence_l2", 1e-30))
        st_max = float(stokes_baseline.get("final_divergence_max_abs", 1e-30))
    else:
        st_l2 = max(div_l2, 1e-30)
        st_max = max(div_max, 1e-30)
    div_vs_stokes_l2 = float(div_l2 / max(st_l2, 1e-30))
    div_vs_stokes_max = float(div_max / max(st_max, 1e-30))
    div_ok = bool((div_vs_stokes_l2 <= 100.0) and (div_vs_stokes_max <= 100.0))
    vel_max = float(final.get("velocity_magnitude_max", 0.0))
    history_ref = history[:-1] if len(history) > 1 else []
    vel_scale = _velocity_scale_from_history(
        history=history_ref,
        stokes_baseline=stokes_baseline,
        inlet_speed=float(inlet_speed),
    )
    if not history_ref:
        vel_scale = max(
            float(vel_scale),
            abs(_safe_float(final.get("velocity_magnitude_mean", 0.0), 0.0)),
            abs(float(inlet_speed)),
            abs(float(vel_max)),
            1e-9,
        )
    vel_growth_ratio = float(vel_max / max(vel_scale, 1e-30))
    vel_ok = bool(
        np.isfinite(vel_max)
        and np.isfinite(vel_growth_ratio)
        and (vel_growth_ratio <= float(velocity_growth_ratio_limit))
    )
    accepted = bool(finite_ok and wall_ok and flux_ok and div_ok and vel_ok)
    reason = "convective prototype acceptance criteria satisfied"
    if not finite_ok:
        reason = "non-finite fields in convective prototype"
    elif not wall_ok:
        reason = "wall flux tolerance violated"
    elif not flux_ok:
        reason = "outlet/inlet flux ratio tolerance violated"
    elif not div_ok:
        reason = "divergence degradation vs stokes baseline exceeds tolerance"
    elif not vel_ok:
        reason = "velocity blow-up detected"
    return {
        "convective_prototype_accepted": bool(accepted),
        "reason": str(reason),
        "checks": {
            "finite_ok": bool(finite_ok),
            "wall_flux_ok": bool(wall_ok),
            "flux_ratio_ok": bool(flux_ok),
            "divergence_vs_stokes_l2_ratio": float(div_vs_stokes_l2),
            "divergence_vs_stokes_max_ratio": float(div_vs_stokes_max),
            "divergence_vs_stokes_ok": bool(div_ok),
            "velocity_max_ok": bool(vel_ok),
            "velocity_scale_reference": float(vel_scale),
            "velocity_growth_ratio": float(vel_growth_ratio),
            "velocity_growth_ratio_limit": float(velocity_growth_ratio_limit),
        },
    }


def _evaluate_convective_readiness(
    *,
    convective_prototype_accepted: bool,
    convective_prototype_acceptance_reason: str,
    history: list[dict[str, Any]],
    stokes_baseline: dict[str, Any] | None = None,
    outlet_inlet_flux_ratio_tolerance: float = 5e-3,
    wall_flux_abs_tolerance: float = 1e-14,
    max_convective_substeps: int = 128,
    divergence_vs_stokes_factor_limit: float = 100.0,
    flow_dt_mode: str = "manual",
    convective_cfl_target: float = 0.5,
    cfl_target_tolerance_factor: float = 1.05,
    convective_cfl_acceptance_eps: float = 1e-9,
    inlet_speed: float = 0.0,
    velocity_growth_ratio_limit: float = 15.0,
) -> dict[str, Any]:
    if not history:
        return {
            "ready_for_long_ns_run_debug": False,
            "ready_for_long_ns_run_physical": False,
            "readiness_reason": "no progression history",
            "convective_readiness_checks": {},
        }
    stats = _convective_history_stats(
        history, convective_cfl_acceptance_eps=convective_cfl_acceptance_eps
    )
    final = history[-1]
    effective_ok = bool(not bool(stats.get("effective_cfl_warning_any", False)))
    raw_warn_any = bool(stats.get("raw_cfl_warning_any", False))
    raw_after_max = float(
        stats.get("raw_cfl_after_dt_selection_max", stats.get("raw_cfl_max_max", 0.0))
    )
    raw_after_ok = bool(
        raw_after_max
        <= float(convective_cfl_target) * float(cfl_target_tolerance_factor)
    )
    auto_used_any = bool(stats.get("auto_damping_used_any", False))
    substepping_used_any = bool(stats.get("substepping_used_any", False))
    auto_dt_min_hit_any = bool(stats.get("auto_dt_min_hit_any", False))
    substep_cap_ok = bool(
        (not bool(stats.get("substep_cap_hit_any", False)))
        and (int(stats.get("substep_count_max", 0)) <= int(max_convective_substeps))
    )
    finite_ok = bool(final.get("finite_fields", False))
    wall_ok = abs(float(final.get("wall_flux_max_abs_after", 0.0))) <= float(
        wall_flux_abs_tolerance
    )
    flux_ok = abs(float(final.get("outlet_inlet_flux_ratio", 0.0)) - 1.0) <= float(
        outlet_inlet_flux_ratio_tolerance
    )
    vel_max = float(final.get("velocity_magnitude_max", 0.0))
    history_ref = history[:-1] if len(history) > 1 else []
    vel_scale = _velocity_scale_from_history(
        history=history_ref,
        stokes_baseline=stokes_baseline,
        inlet_speed=float(inlet_speed),
    )
    if not history_ref:
        vel_scale = max(
            float(vel_scale),
            abs(_safe_float(final.get("velocity_magnitude_mean", 0.0), 0.0)),
            abs(float(inlet_speed)),
            abs(float(vel_max)),
            1e-9,
        )
    vel_growth_ratio = float(vel_max / max(vel_scale, 1e-30))
    velocity_ok = bool(
        np.isfinite(vel_max)
        and np.isfinite(vel_growth_ratio)
        and (vel_growth_ratio <= float(velocity_growth_ratio_limit))
    )
    if stokes_baseline is not None:
        st_l2 = float(stokes_baseline.get("final_divergence_l2", 1e-30))
        st_linf = float(stokes_baseline.get("final_divergence_max_abs", 1e-30))
    else:
        st_l2 = max(float(final.get("final_divergence_l2", 1e-30)), 1e-30)
        st_linf = max(float(final.get("final_divergence_max_abs", 1e-30)), 1e-30)
    div_l2_ratio = float(final.get("final_divergence_l2", 0.0) / max(st_l2, 1e-30))
    div_linf_ratio = float(
        final.get("final_divergence_max_abs", 0.0) / max(st_linf, 1e-30)
    )
    divergence_ok = bool(
        (div_l2_ratio <= float(divergence_vs_stokes_factor_limit))
        and (div_linf_ratio <= float(divergence_vs_stokes_factor_limit))
    )

    ready_debug = bool(
        convective_prototype_accepted
        and effective_ok
        and finite_ok
        and wall_ok
        and flux_ok
        and velocity_ok
        and divergence_ok
    )
    physical_stability_ok = bool(
        substepping_used_any
        or ((not raw_warn_any) and (not auto_used_any))
        or (str(flow_dt_mode) == "auto_cfl" and raw_after_ok and (not auto_used_any))
    )
    ready_physical = bool(
        ready_debug
        and physical_stability_ok
        and substep_cap_ok
        and (not auto_dt_min_hit_any)
        and ((str(flow_dt_mode) != "auto_cfl") or raw_after_ok)
    )
    reason = "convective prototype physically ready for long NS runs"
    if ready_physical:
        reason = "convective prototype physically ready for long NS runs"
    elif not convective_prototype_accepted:
        reason = "convective prototype not accepted: " + str(
            convective_prototype_acceptance_reason
        )
    elif not effective_ok:
        reason = "effective CFL exceeds limit on one or more steps"
    elif not finite_ok:
        reason = "non-finite fields in progression history"
    elif not wall_ok:
        reason = "wall flux tolerance violated"
    elif not flux_ok:
        reason = "outlet/inlet flux ratio tolerance violated"
    elif not divergence_ok:
        reason = "divergence degradation vs stokes baseline exceeds tolerance"
    elif not velocity_ok:
        reason = "velocity blow-up detected"
    elif not substep_cap_ok:
        reason = "convective substep count exceeds configured cap"
    elif auto_dt_min_hit_any:
        reason = "auto CFL dt reached configured minimum on one or more steps"
    elif (str(flow_dt_mode) == "auto_cfl") and (not raw_after_ok):
        reason = "raw CFL after auto dt selection exceeds target tolerance"
    elif raw_warn_any and (not substepping_used_any):
        reason = "debug-stable only: raw CFL exceeds limit and stabilization relies on auto damping"
    elif auto_used_any and (not substepping_used_any):
        reason = "debug-stable only: auto damping used on one or more steps"
    return {
        "ready_for_long_ns_run_debug": bool(ready_debug),
        "ready_for_long_ns_run_physical": bool(ready_physical),
        "readiness_reason": str(reason),
        "convective_readiness_checks": {
            "convective_prototype_accepted": bool(convective_prototype_accepted),
            "raw_cfl_warning_any": bool(raw_warn_any),
            "effective_cfl_warning_any": bool(
                stats.get("effective_cfl_warning_any", False)
            ),
            "auto_damping_used_any": bool(auto_used_any),
            "flow_dt_mode": str(flow_dt_mode),
            "convective_cfl_acceptance_eps": float(convective_cfl_acceptance_eps),
            "raw_cfl_after_dt_selection_max": float(raw_after_max),
            "raw_cfl_after_dt_selection_ok": bool(raw_after_ok),
            "convective_cfl_target": float(convective_cfl_target),
            "auto_dt_min_hit_any": bool(auto_dt_min_hit_any),
            "substepping_used_any": bool(substepping_used_any),
            "substep_cap_ok": bool(substep_cap_ok),
            "finite_ok": bool(finite_ok),
            "wall_flux_ok": bool(wall_ok),
            "flux_ratio_ok": bool(flux_ok),
            "velocity_ok": bool(velocity_ok),
            "velocity_scale_reference": float(vel_scale),
            "velocity_growth_ratio": float(vel_growth_ratio),
            "velocity_growth_ratio_limit": float(velocity_growth_ratio_limit),
            "divergence_vs_stokes_l2_ratio": float(div_l2_ratio),
            "divergence_vs_stokes_max_ratio": float(div_linf_ratio),
            "divergence_vs_stokes_ok": bool(divergence_ok),
        },
    }


def _evaluate_stokes_ready_for_advection(
    *,
    damage_ratio_l2: float,
    damage_ratio_linf: float,
    predictor_damages_divergence: bool,
) -> bool:
    if predictor_damages_divergence:
        return False
    if float(damage_ratio_l2) > 10.0:
        return False
    if float(damage_ratio_linf) > 100.0:
        return False
    return True


def _evaluate_stokes_baseline_acceptance(
    *,
    final_div_l2: float,
    final_div_max: float,
    outlet_inlet_ratio: float,
    net_boundary_flux_relative: float,
    wall_flux_max_abs: float,
    damage_ratio_l2: float,
    damage_ratio_linf: float,
    finite: bool,
    outlet_ratio_tolerance: float = 5e-3,
    net_boundary_flux_rel_tolerance: float = 5e-3,
    wall_flux_tol: float = 1e-14,
) -> tuple[bool, str]:
    if not finite:
        return False, "non-finite fields"
    if abs(float(outlet_inlet_ratio) - 1.0) > float(outlet_ratio_tolerance):
        return False, "outlet/inlet ratio out of tolerance"
    if abs(float(net_boundary_flux_relative)) > float(net_boundary_flux_rel_tolerance):
        return False, "net boundary flux relative out of tolerance"
    if abs(float(wall_flux_max_abs)) > float(wall_flux_tol):
        return False, "wall flux out of tolerance"
    if float(final_div_l2) > 1.0:
        return False, "final divergence l2 too large"
    if float(final_div_max) > 20.0:
        return False, "final divergence linf too large"
    if float(damage_ratio_l2) > 10.0 or float(damage_ratio_linf) > 100.0:
        return False, "predictor damages divergence vs no-op baseline"
    return True, "stokes baseline acceptance satisfied"


def _recommend_stokes_baseline_config(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {}

    def _row_quality_key(r: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(r.get("final_div_l2", float("inf"))),
            float(r.get("final_div_max_abs", float("inf"))),
            float(abs(r.get("net_boundary_flux_relative", float("inf")))),
        )

    finite_rows = [
        r
        for r in rows
        if bool(r.get("flow_progression_solved", False))
        and bool(r.get("finite_fields_final", True))
        and abs(float(r.get("outlet_inlet_flux_ratio", 0.0)) - 1.0) <= 5e-3
        and abs(float(r.get("net_boundary_flux_relative", 0.0))) <= 5e-3
        and abs(float(r.get("wall_flux_max_abs_after", 0.0))) <= 1e-14
    ]
    candidates = finite_rows if finite_rows else rows
    best_quality = min(candidates, key=_row_quality_key)
    best_l2 = float(best_quality.get("final_div_l2", float("inf")))
    best_linf = float(best_quality.get("final_div_max_abs", float("inf")))
    near_best = [
        r
        for r in candidates
        if float(r.get("final_div_l2", float("inf"))) <= 1.25 * max(best_l2, 1e-30)
        and float(r.get("final_div_max_abs", float("inf")))
        <= 1.25 * max(best_linf, 1e-30)
    ]
    if not near_best:
        near_best = [best_quality]
    rec = max(
        near_best,
        key=lambda r: (
            float(r.get("flow_dt", 0.0)),
            float(r.get("steps_per_second", 0.0)),
            -float(r.get("runtime_seconds", float("inf"))),
        ),
    )
    return {
        "flow_dt": float(rec.get("flow_dt", 0.0)),
        "viscous_predictor_mode": str(rec.get("viscous_predictor_mode", "")),
        "viscous_face_flux_divergence_impact_cap": rec.get(
            "viscous_face_flux_divergence_impact_cap", None
        ),
        "resolved_cap_value": float(rec.get("resolved_cap_value", 0.0)),
        "reason": (
            "selected for near-best divergence/flux quality with preference for larger dt and faster runtime"
        ),
    }


def _recommend_convective_debug_config(
    rows: list[dict[str, Any]],
    *,
    stokes_baseline: dict[str, Any] | None,
    divergence_vs_stokes_factor_limit: float = 100.0,
) -> dict[str, Any]:
    if not rows:
        return {}

    stokes_l2 = (
        float(stokes_baseline.get("final_divergence_l2", 0.0))
        if stokes_baseline
        else 0.0
    )
    stokes_linf = (
        float(stokes_baseline.get("final_divergence_max_abs", 0.0))
        if stokes_baseline
        else 0.0
    )
    stokes_l2 = max(stokes_l2, 1e-30)
    stokes_linf = max(stokes_linf, 1e-30)

    def _row_is_clean(r: dict[str, Any]) -> bool:
        limit = float(r.get("convective_cfl_limit", 0.5))
        out_in = abs(float(r.get("outlet_inlet_flux_ratio", 0.0)) - 1.0)
        div_l2_ratio = float(
            r.get(
                "divergence_vs_stokes_l2_ratio",
                float(r.get("final_div_l2", float("inf"))) / stokes_l2,
            )
        )
        div_linf_ratio = float(
            r.get(
                "divergence_vs_stokes_max_ratio",
                float(r.get("final_div_max_abs", float("inf"))) / stokes_linf,
            )
        )
        return bool(
            bool(r.get("flow_progression_solved", False))
            and bool(r.get("convective_prototype_accepted", False))
            and bool(r.get("ready_for_long_ns_run_debug", False))
            and float(r.get("effective_cfl_max", float("inf"))) <= (limit + 1e-12)
            and out_in <= 5e-3
            and abs(float(r.get("net_boundary_flux_relative", float("inf")))) <= 5e-3
            and abs(float(r.get("wall_flux_max_abs_after", float("inf")))) <= 1e-14
            and div_l2_ratio <= float(divergence_vs_stokes_factor_limit)
            and div_linf_ratio <= float(divergence_vs_stokes_factor_limit)
        )

    preferred = [r for r in rows if _row_is_clean(r)]
    candidates = preferred if preferred else rows

    def _row_key(r: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        limit = max(float(r.get("convective_cfl_limit", 0.5)), 1e-30)
        raw_ratio = float(r.get("raw_cfl_max", float("inf"))) / limit
        damping_eff = float(r.get("mean_effective_damping", 0.0))
        out_in_penalty = abs(float(r.get("outlet_inlet_flux_ratio", 0.0)) - 1.0)
        return (
            1.0 if bool(r.get("ready_for_long_ns_run_physical", False)) else 0.0,
            1.0 if bool(r.get("ready_for_long_ns_run_debug", False)) else 0.0,
            -raw_ratio,
            damping_eff,
            float(r.get("flow_dt", 0.0)),
            -(float(r.get("final_div_l2", float("inf"))) + out_in_penalty),
        )

    rec = max(candidates, key=_row_key)
    reason = (
        "selected for effective CFL within limit, stable flux/wall diagnostics, and best tradeoff "
        "between larger dt and weaker auto damping"
    )
    if not preferred:
        reason = "selected as best-available debug candidate (no row satisfied all clean criteria)"
    return {
        "flow_dt": float(rec.get("flow_dt", 0.0)),
        "requested_damping": float(rec.get("requested_damping", 0.0)),
        "convective_cfl_limit": float(rec.get("convective_cfl_limit", 0.5)),
        "flow_steps": int(rec.get("flow_steps", 0)),
        "raw_cfl_max": float(rec.get("raw_cfl_max", 0.0)),
        "effective_cfl_max": float(rec.get("effective_cfl_max", 0.0)),
        "mean_effective_damping": float(rec.get("mean_effective_damping", 0.0)),
        "ready_for_long_ns_run_debug": bool(
            rec.get("ready_for_long_ns_run_debug", False)
        ),
        "ready_for_long_ns_run_physical": bool(
            rec.get("ready_for_long_ns_run_physical", False)
        ),
        "reason": str(reason),
    }


def _operator_region_metrics(
    values: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    arr = np.asarray(values, dtype=np.float64)
    for name, mask in masks.items():
        vals = arr[np.asarray(mask, dtype=bool)]
        if vals.size == 0:
            out[name] = {"count": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "l2": 0.0}
        else:
            out[name] = {
                "count": float(vals.size),
                "max_abs": float(np.max(np.abs(vals))),
                "mean_abs": float(np.mean(np.abs(vals))),
                "l2": float(np.sqrt(np.mean(vals * vals))),
            }
    return out


def _build_operator_identity_audit(
    *,
    mesh,
    coeff: dict[str, np.ndarray],
    dt: float,
    density: float,
    outlet_faces: np.ndarray,
    pressure_outlet_value: float,
    pressure_solution: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        _matvec_pressure_numpy,
        _pressure_face_gradient_flux,
        compute_tetra_flux_divergence,
    )

    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    fields = {
        "constant": np.full((centers.shape[0],), 1.23456789, dtype=np.float64),
        "linear_x": np.asarray(centers[:, 0], dtype=np.float64).copy(),
        "linear_y": np.asarray(centers[:, 1], dtype=np.float64).copy(),
        "linear_z": np.asarray(centers[:, 2], dtype=np.float64).copy(),
        "random_seeded": np.random.default_rng(20260503)
        .standard_normal(centers.shape[0])
        .astype(np.float64),
        "pressure_solution": np.asarray(pressure_solution, dtype=np.float64).copy(),
    }
    entries: dict[str, Any] = {}
    best_sign_global = "minus"
    worst_rel_l2 = 0.0
    for name, p in fields.items():
        ap = np.asarray(_matvec_pressure_numpy(coeff, p), dtype=np.float64)
        lap_from_ap = ap / vol
        grad_flux = np.asarray(
            _pressure_face_gradient_flux(
                mesh,
                p,
                dt=float(dt),
                density=float(density),
                outlet_faces=np.asarray(outlet_faces, dtype=np.int64),
                pressure_outlet_value=float(pressure_outlet_value),
            ),
            dtype=np.float64,
        )
        div_grad = np.asarray(
            compute_tetra_flux_divergence(mesh, grad_flux)["divergence"],
            dtype=np.float64,
        )
        diff_plus = lap_from_ap + div_grad
        diff_minus = lap_from_ap - div_grad
        l2_plus = float(np.sqrt(np.mean(diff_plus * diff_plus)))
        l2_minus = float(np.sqrt(np.mean(diff_minus * diff_minus)))
        if l2_minus <= l2_plus:
            chosen_sign = "minus"
            diff = diff_minus
        else:
            chosen_sign = "plus"
            diff = diff_plus
        if name in {"random_seeded", "pressure_solution"}:
            best_sign_global = chosen_sign
        rel_l2 = float(
            np.sqrt(np.mean(diff * diff))
            / max(float(np.sqrt(np.mean((lap_from_ap * lap_from_ap)))), 1e-30)
        )
        worst_rel_l2 = max(worst_rel_l2, rel_l2)
        entries[name] = {
            "chosen_sign": chosen_sign,
            "lap_from_assembled_operator_stats": _vector_stats(lap_from_ap),
            "div_of_pressure_gradient_flux_stats": _vector_stats(div_grad),
            "identity_error_stats": _vector_stats(diff),
            "identity_relative_l2": rel_l2,
            "identity_region_metrics": _operator_region_metrics(diff, masks),
        }
    return {
        "global_recommended_sign_for_identity": best_sign_global,
        "worst_identity_relative_l2": float(worst_rel_l2),
        "projection_operator_consistent": bool(worst_rel_l2 <= 0.05),
        "fields": entries,
    }


def _build_projection_equation_residual_audit(
    *,
    mesh,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    correction_flux_raw_pre_limiter: np.ndarray,
    correction_flux_limited_pre_bc: np.ndarray,
    correction_flux_effective_post_bc: np.ndarray,
    pressure_gradient_flux: np.ndarray,
    rhs_used: np.ndarray,
    projection_sign: str,
    rhs_mode: str,
    dt: float,
    density: float,
    step_number: int,
    stage_name: str,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        compute_tetra_flux_divergence,
    )

    q_star = np.asarray(flux_star, dtype=np.float64)
    q_corr = np.asarray(flux_corrected, dtype=np.float64)
    q_corr_eff = np.asarray(correction_flux_effective_post_bc, dtype=np.float64)
    q_corr_raw = np.asarray(correction_flux_raw_pre_limiter, dtype=np.float64)
    q_corr_limited = np.asarray(correction_flux_limited_pre_bc, dtype=np.float64)
    q_corr_post_bc = q_corr - q_star
    q_grad = np.asarray(pressure_gradient_flux, dtype=np.float64)

    div_star = np.asarray(
        compute_tetra_flux_divergence(mesh, q_star)["divergence"], dtype=np.float64
    )
    div_corr = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr)["divergence"], dtype=np.float64
    )
    div_corr_eff = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr_eff)["divergence"], dtype=np.float64
    )
    div_corr_raw = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr_raw)["divergence"], dtype=np.float64
    )
    div_corr_limited = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr_limited)["divergence"],
        dtype=np.float64,
    )
    div_corr_post_bc = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr_post_bc)["divergence"],
        dtype=np.float64,
    )
    div_grad = np.asarray(
        compute_tetra_flux_divergence(mesh, q_grad)["divergence"], dtype=np.float64
    )

    predicted_from_flux_additivity = div_star + div_corr_eff
    predicted_from_pressure_operator = div_star - div_grad
    residual_flux_additivity = div_corr - predicted_from_flux_additivity
    residual_pressure_operator = div_corr - predicted_from_pressure_operator
    residual_stage_raw = div_corr_raw + div_grad
    residual_stage_limited = div_corr_limited + div_grad
    residual_stage_final = div_corr_post_bc + div_grad
    rel_l2_raw = float(
        np.sqrt(np.mean(residual_stage_raw * residual_stage_raw))
        / max(float(np.sqrt(np.mean(div_corr_raw * div_corr_raw))), 1e-30)
    )
    rel_l2_limited = float(
        np.sqrt(np.mean(residual_stage_limited * residual_stage_limited))
        / max(float(np.sqrt(np.mean(div_corr_limited * div_corr_limited))), 1e-30)
    )
    rel_l2_final = float(
        np.sqrt(np.mean(residual_stage_final * residual_stage_final))
        / max(float(np.sqrt(np.mean(div_corr_post_bc * div_corr_post_bc))), 1e-30)
    )
    stage_rows = [
        {
            "stage": "raw_pre_limiter",
            "relative_l2": rel_l2_raw,
            "max_abs": float(np.max(np.abs(residual_stage_raw))),
            "l2": float(np.sqrt(np.mean(residual_stage_raw * residual_stage_raw))),
        },
        {
            "stage": "limited_pre_bc",
            "relative_l2": rel_l2_limited,
            "max_abs": float(np.max(np.abs(residual_stage_limited))),
            "l2": float(
                np.sqrt(np.mean(residual_stage_limited * residual_stage_limited))
            ),
        },
        {
            "stage": "final_post_bc",
            "relative_l2": rel_l2_final,
            "max_abs": float(np.max(np.abs(residual_stage_final))),
            "l2": float(np.sqrt(np.mean(residual_stage_final * residual_stage_final))),
        },
    ]
    best_stage = min(stage_rows, key=lambda x: float(x["relative_l2"]))
    stage_consistent_default = bool(float(best_stage["relative_l2"]) <= 0.05)
    stage_targets = {
        "raw_pre_limiter": div_corr_raw,
        "limited_pre_bc": div_corr_limited,
        "final_post_bc": div_corr_post_bc,
    }
    dt_over_rho = float(dt) / max(float(density), 1e-30)
    rho_over_dt = max(float(density), 1e-30) / max(float(dt), 1e-30)
    scaling_candidates = [
        ("1.0", 1.0),
        ("2.0", 2.0),
        ("0.5", 0.5),
        ("dt_over_rho", dt_over_rho),
        ("rho_over_dt", rho_over_dt),
    ]
    sign_candidates = [("minus", -1.0), ("plus", 1.0)]
    scaling_rows: list[dict[str, Any]] = []
    best_scaling_row: dict[str, Any] | None = None
    for stage_key, div_target in stage_targets.items():
        target_l2 = float(np.sqrt(np.mean(div_target * div_target)))
        for sign_name, sign_val in sign_candidates:
            for scale_name, scale_val in scaling_candidates:
                pred = sign_val * float(scale_val) * div_grad
                res = div_target - pred
                rel_l2_local = float(
                    np.sqrt(np.mean(res * res)) / max(float(target_l2), 1e-30)
                )
                row = {
                    "stage": str(stage_key),
                    "sign": str(sign_name),
                    "scale_label": str(scale_name),
                    "scale_value": float(scale_val),
                    "relative_l2": float(rel_l2_local),
                    "max_abs": float(np.max(np.abs(res))),
                    "l2": float(np.sqrt(np.mean(res * res))),
                }
                scaling_rows.append(row)
                if (best_scaling_row is None) or (
                    float(row["relative_l2"]) < float(best_scaling_row["relative_l2"])
                ):
                    best_scaling_row = row
    if best_scaling_row is None:
        best_scaling_row = {
            "stage": "raw_pre_limiter",
            "sign": "minus",
            "scale_label": "1.0",
            "scale_value": 1.0,
            "relative_l2": float("inf"),
            "max_abs": float("inf"),
            "l2": float("inf"),
        }
    stage_consistent = bool(float(best_scaling_row["relative_l2"]) <= 0.05)
    rel_l2 = rel_l2_final
    top = np.argsort(-np.abs(residual_pressure_operator))[:50]
    top_rows: list[dict[str, Any]] = []
    for cid in top.tolist():
        top_rows.append(
            {
                "cell_index": int(cid),
                "center_xyz": np.asarray(
                    mesh.cell_centers[cid], dtype=np.float64
                ).tolist(),
                "cell_volume": float(mesh.cell_volumes[cid]),
                "div_star": float(div_star[cid]),
                "div_correction_effective": float(div_corr_eff[cid]),
                "div_correction_raw": float(div_corr_raw[cid]),
                "div_from_pressure_grad_flux": float(div_grad[cid]),
                "predicted_div_corrected": float(predicted_from_pressure_operator[cid]),
                "actual_div_corrected": float(div_corr[cid]),
                "residual": float(residual_pressure_operator[cid]),
            }
        )
    residual_reason = "projection equation residual consistent at best operator stage"
    if not stage_consistent:
        residual_reason = (
            f"inconsistent at all tested scalings; best stage={best_scaling_row['stage']} "
            f"sign={best_scaling_row['sign']} scale={best_scaling_row['scale_label']} "
            f"relative_l2={float(best_scaling_row['relative_l2']):.3e}"
        )
    elif (
        str(best_scaling_row["stage"]) != "final_post_bc"
        or str(best_scaling_row["sign"]) != "minus"
        or str(best_scaling_row["scale_label"]) != "1.0"
    ):
        residual_reason = (
            "best consistency requires adjusted audit scaling/sign/stage; "
            "diagnostic mismatch likely from definition/stage mismatch, not physical projection failure."
        )
    scaling_diagnosis = (
        "default audit uses sign=minus scale=1.0 against div_from_pressure_grad_flux"
    )
    if (
        str(best_scaling_row["sign"]) == "minus"
        and str(best_scaling_row["scale_label"]) == "0.5"
    ):
        scaling_diagnosis = (
            "best fit is minus with scale 0.5: correction divergence magnitude is ~0.5 * grad-flux divergence; "
            "diagnostic likely comparing different correction definitions (factor-2 mismatch)."
        )
    elif (
        str(best_scaling_row["sign"]) == "minus"
        and str(best_scaling_row["scale_label"]) == "2.0"
    ):
        scaling_diagnosis = (
            "best fit is minus with scale 2.0: correction divergence magnitude is ~2x grad-flux divergence; "
            "diagnostic likely missing factor-2 in formula."
        )
    elif str(best_scaling_row["sign"]) == "plus":
        scaling_diagnosis = "best fit requires plus sign; this indicates sign-convention mismatch in diagnostic relation."
    elif str(best_scaling_row["scale_label"]) in {"dt_over_rho", "rho_over_dt"}:
        scaling_diagnosis = "best fit requires dt/rho scaling variant; diagnostic may mix pre-scaled and raw pressure-gradient definitions."
    rhs_arr = np.asarray(rhs_used, dtype=np.float64)
    return {
        "audit_context": {
            "stage_name": str(stage_name),
            "step_number": int(step_number),
            "projection_sign": str(projection_sign),
            "rhs_mode": str(rhs_mode),
            "dt": float(dt),
            "density": float(density),
        },
        "rhs_used_stats": _vector_stats(rhs_arr),
        "div_star_stats": _vector_stats(div_star),
        "div_corrected_stats": _vector_stats(div_corr),
        "div_correction_effective_stats": _vector_stats(div_corr_eff),
        "div_correction_raw_stats": _vector_stats(div_corr_raw),
        "div_correction_limited_pre_bc_stats": _vector_stats(div_corr_limited),
        "div_correction_post_bc_stats": _vector_stats(div_corr_post_bc),
        "div_pressure_gradient_flux_stats": _vector_stats(div_grad),
        "predicted_from_flux_additivity_stats": _vector_stats(
            predicted_from_flux_additivity
        ),
        "predicted_from_pressure_operator_stats": _vector_stats(
            predicted_from_pressure_operator
        ),
        "residual_flux_additivity_stats": _vector_stats(residual_flux_additivity),
        "residual_pressure_operator_stats": _vector_stats(residual_pressure_operator),
        "residual_pressure_operator_relative_l2": rel_l2,
        "stage_relative_residuals": stage_rows,
        "best_stage_for_consistency": best_stage,
        "projection_equation_default_stage_consistent": bool(stage_consistent_default),
        "projection_equation_consistent_final_post_bc": bool(rel_l2_final <= 0.05),
        "projection_equation_residual_consistent": bool(stage_consistent),
        "projection_equation_best_scaling": str(best_scaling_row["scale_label"]),
        "projection_equation_best_scaling_value": float(
            best_scaling_row["scale_value"]
        ),
        "projection_equation_best_sign": str(best_scaling_row["sign"]),
        "projection_equation_best_stage": str(best_scaling_row["stage"]),
        "projection_equation_best_relative_l2": float(best_scaling_row["relative_l2"]),
        "projection_equation_scaling_diagnosis": str(scaling_diagnosis),
        "projection_equation_scaling_candidates": scaling_rows,
        "projection_equation_residual_reason": str(residual_reason),
        "projection_equation_consistent": bool(stage_consistent),
        "region_metrics_pressure_operator_residual": _operator_region_metrics(
            residual_pressure_operator, masks
        ),
        "top_cells_by_residual": top_rows,
    }


def _build_pressure_projection_scale_audit(
    *,
    mesh,
    rhs: np.ndarray,
    star_sum: np.ndarray,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    correction_flux_raw: np.ndarray,
    pressure_gradient_flux: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    q_star = np.asarray(flux_star, dtype=np.float64)
    q_corr = np.asarray(flux_corrected, dtype=np.float64)
    q_grad = np.asarray(pressure_gradient_flux, dtype=np.float64)
    q_corr_raw = np.asarray(correction_flux_raw, dtype=np.float64)
    q_eff = q_corr - q_star
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    star = np.asarray(star_sum, dtype=np.float64)
    face_ratio = np.abs(q_corr_raw) / np.maximum(np.abs(q_star), 1e-30)
    face_ratio_grad = np.abs(q_grad) / np.maximum(np.abs(q_star), 1e-30)
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    cell_corr = np.zeros((mesh.tetrahedra.shape[0],), dtype=np.float64)
    np.add.at(cell_corr, c0, q_eff)
    interior = c1 >= 0
    if np.any(interior):
        np.add.at(cell_corr, c1[interior], -q_eff[interior])
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    cell_corr_per_vol = cell_corr / vol
    source_per_vol = star / vol
    scale_ratio = np.abs(cell_corr_per_vol) / np.maximum(np.abs(source_per_vol), 1e-30)
    top_faces = np.argsort(-face_ratio)[:30]
    top_cells = np.argsort(-scale_ratio)[:30]
    return {
        "rhs_stats": _vector_stats(rhs_arr),
        "divergence_source_cell_flux_sum_stats": _vector_stats(star),
        "face_flux_star_stats": _vector_stats(q_star),
        "face_flux_corrected_stats": _vector_stats(q_corr),
        "raw_correction_flux_stats": _vector_stats(q_corr_raw),
        "pressure_gradient_flux_stats": _vector_stats(q_grad),
        "effective_correction_flux_stats": _vector_stats(q_eff),
        "per_face_ratio_raw_correction_to_star_stats": _vector_stats(face_ratio),
        "per_face_ratio_grad_to_star_stats": _vector_stats(face_ratio_grad),
        "per_cell_correction_over_source_ratio_stats": _vector_stats(scale_ratio),
        "region_metrics_correction_over_source_ratio": _operator_region_metrics(
            scale_ratio, masks
        ),
        "top_30_faces_by_raw_correction_to_star_ratio": [
            {
                "face_index": int(fid),
                "ratio": float(face_ratio[fid]),
                "flux_star": float(q_star[fid]),
                "raw_correction_flux": float(q_corr_raw[fid]),
                "pressure_gradient_flux": float(q_grad[fid]),
                "center_xyz": np.asarray(
                    mesh.face_centers[fid], dtype=np.float64
                ).tolist(),
            }
            for fid in top_faces.tolist()
        ],
        "top_30_cells_by_correction_over_source_ratio": [
            {
                "cell_index": int(cid),
                "ratio": float(scale_ratio[cid]),
                "cell_volume": float(mesh.cell_volumes[cid]),
                "source_per_vol": float(source_per_vol[cid]),
                "correction_per_vol": float(cell_corr_per_vol[cid]),
                "center_xyz": np.asarray(
                    mesh.cell_centers[cid], dtype=np.float64
                ).tolist(),
            }
            for cid in top_cells.tolist()
        ],
    }


def _top_divergence_correction_breakdown(
    *,
    mesh,
    top_cells: list[dict[str, Any]],
    pressure: np.ndarray,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    correction_flux_raw: np.ndarray,
) -> dict[str, Any]:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    q_star = np.asarray(flux_star, dtype=np.float64)
    q_corr = np.asarray(flux_corrected, dtype=np.float64)
    q_raw = np.asarray(correction_flux_raw, dtype=np.float64)
    q_eff = q_corr - q_star
    rows: list[dict[str, Any]] = []
    for row in top_cells:
        cid = int(row.get("cell_index", -1))
        if cid < 0:
            continue
        face_ids = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
        by_face: list[dict[str, Any]] = []
        div_corr_from_faces = 0.0
        for fid in face_ids.tolist():
            owner = int(c0[fid])
            neigh = int(c1[fid])
            if owner == cid:
                sign = 1.0
                neighbor = neigh
                dist = (
                    float(np.linalg.norm(centers[cid] - centers[neigh]))
                    if neigh >= 0
                    else float(np.linalg.norm(face_centers[fid] - centers[cid]))
                )
            elif neigh == cid:
                sign = -1.0
                neighbor = owner
                dist = float(np.linalg.norm(centers[cid] - centers[owner]))
            else:
                continue
            div_contrib = (
                sign * float(q_eff[fid]) / max(float(mesh.cell_volumes[cid]), 1e-30)
            )
            div_corr_from_faces += div_contrib
            group = "interior"
            if neigh < 0:
                tag = int(mesh.boundary_tag_per_face[fid])
                group = mesh.boundary_face_names.get(tag, f"tag_{tag}")
            by_face.append(
                {
                    "face_index": int(fid),
                    "face_group": str(group),
                    "neighbor_cell": int(neighbor) if neighbor >= 0 else -1,
                    "face_area": float(areas[fid]),
                    "face_normal": face_normals[fid].tolist(),
                    "face_center_xyz": face_centers[fid].tolist(),
                    "center_to_center_or_face_distance": float(max(dist, 1e-30)),
                    "pressure_cell": float(p[cid]),
                    "pressure_neighbor_or_boundary": (
                        float(p[neighbor]) if neighbor >= 0 else None
                    ),
                    "flux_star": float(q_star[fid]),
                    "flux_corrected": float(q_corr[fid]),
                    "raw_correction_flux": float(q_raw[fid]),
                    "effective_correction_flux": float(q_eff[fid]),
                    "correction_div_contribution_over_volume": float(div_contrib),
                }
            )
        rows.append(
            {
                "cell_index": int(cid),
                "center_xyz": centers[cid].tolist(),
                "cell_volume": float(mesh.cell_volumes[cid]),
                "pressure_value": float(p[cid]),
                "div_star": float(row.get("div_star", 0.0)),
                "div_corrected": float(row.get("div_corrected", 0.0)),
                "div_from_face_correction_sum": float(div_corr_from_faces),
                "adjacent_faces": by_face,
            }
        )
    return {"cells": rows}


def _build_top_divergence_local_face_audit(
    *,
    mesh,
    top_cells: list[dict[str, Any]],
    pressure: np.ndarray,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    correction_flux_raw: np.ndarray,
    pressure_outlet_value: float,
) -> dict[str, Any]:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    q_star = np.asarray(flux_star, dtype=np.float64)
    q_corr = np.asarray(flux_corrected, dtype=np.float64)
    q_raw = np.asarray(correction_flux_raw, dtype=np.float64)
    q_eff = q_corr - q_star
    top_rows: list[dict[str, Any]] = []
    for row in top_cells:
        cid = int(row.get("cell_index", -1))
        if cid < 0:
            continue
        vol = max(float(mesh.cell_volumes[cid]), 1e-30)
        faces = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
        local_faces: list[dict[str, Any]] = []
        div_from_faces = 0.0
        for fid in faces.tolist():
            owner = int(c0[fid])
            neigh = int(c1[fid])
            if owner == cid:
                orient = "owner_outward"
                sign = 1.0
                neighbor_index = int(neigh)
                n_local = np.asarray(face_normals[fid], dtype=np.float64)
                p_owner = float(p[cid])
                if neigh >= 0:
                    p_other = float(p[neigh])
                    distance = float(np.linalg.norm(centers[neigh] - centers[cid]))
                    neighbor_label = "interior_cell"
                else:
                    tag = int(mesh.boundary_tag_per_face[fid])
                    group = mesh.boundary_face_names.get(tag, f"tag_{tag}")
                    neighbor_label = str(group)
                    if str(group) == "outlet":
                        p_other = float(pressure_outlet_value)
                    else:
                        p_other = float(p_owner)
                    distance = float(np.linalg.norm(face_centers[fid] - centers[cid]))
            elif neigh == cid:
                orient = "neighbor_outward"
                sign = -1.0
                neighbor_index = int(owner)
                n_local = -np.asarray(face_normals[fid], dtype=np.float64)
                p_owner = float(p[cid])
                p_other = float(p[owner])
                distance = float(np.linalg.norm(centers[owner] - centers[cid]))
                neighbor_label = "interior_cell"
            else:
                continue
            distance = max(distance, 1e-30)
            pressure_jump = float(p_other - p_owner)
            grad_p_normal = float(pressure_jump / distance)
            star_local = float(sign * q_star[fid])
            corr_local = float(sign * q_corr[fid])
            corr_flux_local = float(sign * q_eff[fid])
            div_contrib = corr_local / vol
            div_from_faces += div_contrib
            local_faces.append(
                {
                    "face_index": int(fid),
                    "neighbor_cell_index": int(neighbor_index)
                    if neighbor_index >= 0
                    else -1,
                    "neighbor_boundary_group": (
                        None if neighbor_index >= 0 else str(neighbor_label)
                    ),
                    "face_area": float(areas[fid]),
                    "face_normal": np.asarray(n_local, dtype=np.float64).tolist(),
                    "owner_neighbor_orientation": orient,
                    "distance_center_or_face": float(distance),
                    "pressure_owner": float(p_owner),
                    "pressure_neighbor_or_boundary_value": float(p_other),
                    "pressure_jump": float(pressure_jump),
                    "grad_p_normal": float(grad_p_normal),
                    "correction_flux": float(corr_flux_local),
                    "star_flux": float(star_local),
                    "corrected_flux": float(corr_local),
                    "raw_correction_flux_signed": float(sign * q_raw[fid]),
                    "contribution_to_div": float(div_contrib),
                }
            )
        top_rows.append(
            {
                "cell_index": int(cid),
                "center_xyz": np.asarray(centers[cid], dtype=np.float64).tolist(),
                "volume": float(vol),
                "div_star": float(row.get("div_star", 0.0)),
                "div_corrected": float(row.get("div_corrected", 0.0)),
                "div_delta": float(
                    row.get("div_corrected", 0.0) - row.get("div_star", 0.0)
                ),
                "adjacent_faces": local_faces,
                "div_corrected_reconstructed_from_faces": float(div_from_faces),
                "reconstruction_residual": float(
                    div_from_faces - float(row.get("div_corrected", 0.0))
                ),
            }
        )
    return {"cells": top_rows}


def _build_pressure_projection_hotspot_correlation(
    *,
    mesh,
    div_star: np.ndarray,
    div_corrected: np.ndarray,
    correction_flux_raw: np.ndarray,
) -> dict[str, Any]:
    c0 = np.asarray(mesh.face_to_cells[:, 0], dtype=np.int64)
    c1 = np.asarray(mesh.face_to_cells[:, 1], dtype=np.int64)
    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    face_centers = np.asarray(mesh.face_centers, dtype=np.float64)
    areas = np.asarray(mesh.face_areas, dtype=np.float64)
    vols = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    q_corr = np.asarray(correction_flux_raw, dtype=np.float64)
    div_s = np.asarray(div_star, dtype=np.float64)
    div_c = np.asarray(div_corrected, dtype=np.float64)
    abs_div = np.abs(div_c)
    abs_delta = np.abs(div_c - div_s)

    n_cells = int(mesh.tetrahedra.shape[0])
    min_face_area = np.zeros((n_cells,), dtype=np.float64)
    max_face_area = np.zeros((n_cells,), dtype=np.float64)
    mean_face_area = np.zeros((n_cells,), dtype=np.float64)
    min_neighbor_dist = np.zeros((n_cells,), dtype=np.float64)
    max_neighbor_dist = np.zeros((n_cells,), dtype=np.float64)
    max_corr_flux = np.zeros((n_cells,), dtype=np.float64)
    max_corr_flux_over_vol = np.zeros((n_cells,), dtype=np.float64)
    area_ratio = np.zeros((n_cells,), dtype=np.float64)
    dist_ratio = np.zeros((n_cells,), dtype=np.float64)
    geom_proxy = np.zeros((n_cells,), dtype=np.float64)
    for cid in range(n_cells):
        fids = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
        if fids.size == 0:
            continue
        af = np.asarray(areas[fids], dtype=np.float64)
        min_face_area[cid] = float(np.min(af))
        max_face_area[cid] = float(np.max(af))
        mean_face_area[cid] = float(np.mean(af))
        dvals: list[float] = []
        for fid in fids.tolist():
            neigh = int(c1[fid]) if int(c0[fid]) == cid else int(c0[fid])
            if neigh >= 0:
                d = float(np.linalg.norm(centers[neigh] - centers[cid]))
            else:
                d = float(np.linalg.norm(face_centers[fid] - centers[cid]))
            dvals.append(max(d, 1e-30))
        d_arr = np.asarray(dvals, dtype=np.float64)
        min_neighbor_dist[cid] = float(np.min(d_arr))
        max_neighbor_dist[cid] = float(np.max(d_arr))
        loc_corr = np.abs(np.asarray(q_corr[fids], dtype=np.float64))
        max_corr_flux[cid] = float(np.max(loc_corr))
        max_corr_flux_over_vol[cid] = float(np.max(loc_corr / vols[cid]))
        area_ratio[cid] = float(max_face_area[cid] / max(min_face_area[cid], 1e-30))
        dist_ratio[cid] = float(
            max_neighbor_dist[cid] / max(min_neighbor_dist[cid], 1e-30)
        )
        geom_proxy[cid] = float(area_ratio[cid] * dist_ratio[cid])

    def _corr(metric: np.ndarray, target: np.ndarray) -> float:
        x = np.asarray(metric, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.size == 0 or y.size == 0:
            return 0.0
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _top_cells(metric: np.ndarray, top_k: int = 50) -> list[dict[str, Any]]:
        order = np.argsort(-np.asarray(metric, dtype=np.float64))
        rows: list[dict[str, Any]] = []
        for cid in order[: min(top_k, order.size)].tolist():
            rows.append(
                {
                    "cell_index": int(cid),
                    "center_xyz": np.asarray(centers[cid], dtype=np.float64).tolist(),
                    "volume": float(vols[cid]),
                    "abs_div_corrected": float(abs_div[cid]),
                    "abs_div_delta": float(abs_delta[cid]),
                    "min_face_area": float(min_face_area[cid]),
                    "max_face_area": float(max_face_area[cid]),
                    "mean_face_area": float(mean_face_area[cid]),
                    "min_neighbor_distance": float(min_neighbor_dist[cid]),
                    "max_neighbor_distance": float(max_neighbor_dist[cid]),
                    "max_correction_flux_abs": float(max_corr_flux[cid]),
                    "max_correction_flux_over_volume": float(
                        max_corr_flux_over_vol[cid]
                    ),
                    "aspect_ratio_proxy": float(geom_proxy[cid]),
                    "metric_value": float(metric[cid]),
                }
            )
        return rows

    return {
        "metric_stats": {
            "volume": _vector_stats(vols),
            "min_face_area": _vector_stats(min_face_area),
            "max_face_area": _vector_stats(max_face_area),
            "mean_face_area": _vector_stats(mean_face_area),
            "min_neighbor_distance": _vector_stats(min_neighbor_dist),
            "max_neighbor_distance": _vector_stats(max_neighbor_dist),
            "max_correction_flux_abs": _vector_stats(max_corr_flux),
            "max_correction_flux_over_volume": _vector_stats(max_corr_flux_over_vol),
            "aspect_ratio_proxy": _vector_stats(geom_proxy),
            "abs_div_corrected": _vector_stats(abs_div),
            "abs_div_delta": _vector_stats(abs_delta),
        },
        "pearson_correlation_vs_abs_div_corrected": {
            "volume": _corr(vols, abs_div),
            "min_face_area": _corr(min_face_area, abs_div),
            "max_face_area": _corr(max_face_area, abs_div),
            "max_correction_flux_abs": _corr(max_corr_flux, abs_div),
            "max_correction_flux_over_volume": _corr(max_corr_flux_over_vol, abs_div),
            "aspect_ratio_proxy": _corr(geom_proxy, abs_div),
        },
        "top_cells_by_abs_div_corrected": _top_cells(abs_div),
        "top_cells_by_correction_flux_over_volume": _top_cells(max_corr_flux_over_vol),
        "top_cells_by_smallest_volume": _top_cells(1.0 / np.maximum(vols, 1e-30)),
        "top_cells_by_bad_geometry_proxy": _top_cells(geom_proxy),
    }


def _build_pressure_matrix_coefficient_audit(
    *,
    mesh,
    coeff: dict[str, np.ndarray],
    top_divergence_cells: list[dict[str, Any]],
) -> dict[str, Any]:
    n_cells = int(mesh.tetrahedra.shape[0])
    diag = np.asarray(coeff["diag"], dtype=np.float64)
    int_owner = np.asarray(coeff["int_owner"], dtype=np.int64)
    int_neigh = np.asarray(coeff["int_neigh"], dtype=np.int64)
    int_k = np.asarray(coeff["int_k"], dtype=np.float64)
    out_k = np.asarray(coeff["out_k"], dtype=np.float64)

    sum_abs_offdiag = np.zeros((n_cells,), dtype=np.float64)
    if int_k.size:
        np.add.at(sum_abs_offdiag, int_owner, np.abs(int_k))
        np.add.at(sum_abs_offdiag, int_neigh, np.abs(int_k))
    diag_dom = np.abs(diag) / np.maximum(sum_abs_offdiag, 1e-30)
    worst_ids = np.argsort(diag_dom)[: min(50, n_cells)]

    top_ids = {
        int(row.get("cell_index", -1))
        for row in top_divergence_cells
        if int(row.get("cell_index", -1)) >= 0
    }
    overlap = [int(cid) for cid in worst_ids.tolist() if int(cid) in top_ids]

    worst_rows: list[dict[str, Any]] = []
    for cid in worst_ids.tolist():
        worst_rows.append(
            {
                "cell_index": int(cid),
                "center_xyz": np.asarray(
                    mesh.cell_centers[cid], dtype=np.float64
                ).tolist(),
                "diag": float(diag[cid]),
                "sum_abs_offdiag": float(sum_abs_offdiag[cid]),
                "diag_dominance_proxy": float(diag_dom[cid]),
                "is_top_divergence_hotspot": bool(int(cid) in top_ids),
            }
        )

    return {
        "diag_coefficient_stats": _vector_stats(diag),
        "offdiagonal_coupling_k_stats": _vector_stats(int_k),
        "diagonal_dominance_proxy_stats": _vector_stats(diag_dom),
        "positive_offdiag_count_expected_zero": int(np.count_nonzero(int_k < 0.0)),
        "unexpected_nonpositive_diagonal_count": int(np.count_nonzero(diag <= 0.0)),
        "outlet_diagonal_contrib_stats": _vector_stats(out_k),
        "top_50_worst_diagonal_dominance_cells": worst_rows,
        "top_hotspot_overlap_with_worst_diagonal_dominance": {
            "worst_cell_count": int(len(worst_rows)),
            "hotspot_overlap_count": int(len(overlap)),
            "hotspot_overlap_cell_indices": overlap,
        },
    }


def _build_projection_volume_weighting_audit(
    *,
    mesh,
    coeff: dict[str, np.ndarray],
    pressure: np.ndarray,
    rhs: np.ndarray,
    rhs_outlet: np.ndarray,
    star_sum: np.ndarray,
    flux_star: np.ndarray,
    flux_corrected: np.ndarray,
    projection_sign: str,
) -> dict[str, Any]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        _matvec_pressure_numpy,
        compute_tetra_flux_divergence,
    )

    p = np.asarray(pressure, dtype=np.float64)
    rhs_arr = np.asarray(rhs, dtype=np.float64)
    rhs_out = np.asarray(rhs_outlet, dtype=np.float64)
    star = np.asarray(star_sum, dtype=np.float64)
    vol = np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
    ap = np.asarray(_matvec_pressure_numpy(coeff, p), dtype=np.float64)
    residual_flux_form = ap - rhs_arr
    residual_div_form = (ap / vol) - (rhs_arr / vol)

    q_star = np.asarray(flux_star, dtype=np.float64)
    q_corr = np.asarray(flux_corrected, dtype=np.float64)
    q_eff = q_corr - q_star
    div_star = np.asarray(
        compute_tetra_flux_divergence(mesh, q_star)["divergence"], dtype=np.float64
    )
    div_corr = np.asarray(
        compute_tetra_flux_divergence(mesh, q_corr)["divergence"], dtype=np.float64
    )
    div_eff = np.asarray(
        compute_tetra_flux_divergence(mesh, q_eff)["divergence"], dtype=np.float64
    )

    if str(projection_sign) == "minus":
        pred_flux = -ap
        pred_div = -(ap / vol)
    else:
        pred_flux = ap
        pred_div = ap / vol
    # Effective correction divergence from projection should align with sign-scaled A(p).
    corr_flux_sum = np.asarray(div_eff * vol, dtype=np.float64)
    corr_flux_residual_flux_form = corr_flux_sum - pred_flux
    corr_flux_residual_div_form = div_eff - pred_div

    return {
        "contract_interpretation": {
            "projection_sign": str(projection_sign),
            "volume_integrated_flux_form": "A(p) ~ signed effective correction cell-flux sum",
            "cell_divergence_form": "A(p)/V ~ signed effective divergence correction",
        },
        "raw_divergence_units": _vector_stats(div_star),
        "rhs_used_stats": _vector_stats(rhs_arr),
        "rhs_outlet_term_stats": _vector_stats(rhs_out),
        "divergence_source_cell_flux_sum_stats": _vector_stats(star),
        "residual_volume_integrated_flux_form_stats": _vector_stats(residual_flux_form),
        "residual_cell_divergence_form_stats": _vector_stats(residual_div_form),
        "effective_divergence_correction_stats": _vector_stats(div_eff),
        "actual_div_corrected_stats": _vector_stats(div_corr),
        "predicted_correction_flux_sum_from_operator_stats": _vector_stats(pred_flux),
        "predicted_correction_divergence_from_operator_stats": _vector_stats(pred_div),
        "correction_match_residual_volume_form_stats": _vector_stats(
            corr_flux_residual_flux_form
        ),
        "correction_match_residual_divergence_form_stats": _vector_stats(
            corr_flux_residual_div_form
        ),
    }


def _apply_policy_from_pressure(
    *,
    mesh,
    pressure: np.ndarray,
    flux_star: np.ndarray,
    projection_sign: str,
    projection_correction_damping: float,
    dt: float,
    density: float,
    pressure_outlet_value: float,
    policy: str,
    frozen_nonorthogonal_gradient_flux: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        _apply_projection_correction,
        _pressure_face_gradient_flux,
    )

    left = np.asarray(mesh.inlet_faces, dtype=np.int64)
    outlet = np.asarray(mesh.outlet_faces, dtype=np.int64)
    wall = np.asarray(mesh.wall_faces, dtype=np.int64)
    grad = np.asarray(
        _pressure_face_gradient_flux(
            mesh,
            np.asarray(pressure, dtype=np.float64),
            dt=float(dt),
            density=float(density),
            outlet_faces=outlet,
            pressure_outlet_value=float(pressure_outlet_value),
            frozen_nonorthogonal_gradient_flux=(frozen_nonorthogonal_gradient_flux),
        ),
        dtype=np.float64,
    )
    corrected, corr = _apply_projection_correction(
        np.asarray(flux_star, dtype=np.float64),
        grad,
        projection_sign=str(projection_sign),  # type: ignore[arg-type]
        damping=float(projection_correction_damping),
    )
    corrected_policy = np.asarray(corrected, dtype=np.float64).copy()
    if policy == "outlet_pressure_dirichlet":
        pass
    elif policy == "outlet_neumann_flux_preserving":
        corrected_policy[outlet] = np.asarray(flux_star, dtype=np.float64)[outlet]
    elif policy == "boundary_flux_preserving":
        boundary = np.asarray(mesh.boundary_face_indices, dtype=np.int64)
        corrected_policy[boundary] = np.asarray(flux_star, dtype=np.float64)[boundary]
    else:
        raise ValueError(f"Unknown policy: {policy}")
    corrected_policy[left] = np.asarray(flux_star, dtype=np.float64)[left]
    if wall.size:
        corrected_policy[wall] = 0.0
    return corrected_policy, corr


def _resolve_flow_config_for_postsolve_audit(config):
    """Use the solver's effective profile in all post-solve operator audits."""

    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        resolve_tetra_flow_numerical_profile,
    )

    return resolve_tetra_flow_numerical_profile(config)


def _load_flow_resume_state(
    mesh: Any,
    resume_flow_run_dir: str | Path,
    *,
    expected_manifest: dict[str, Any],
) -> tuple[Any, int, float, dict[str, Any]]:
    """Restore the exact state, step, and physical time of a prior flow run."""

    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import TetraFlowState

    resume_run_dir = _normalize_user_path(resume_flow_run_dir).resolve()
    resume_summary_path = resume_run_dir / "summary.json"
    resume_manifest_path = resume_run_dir / "resume_manifest.json"
    resume_flux_path = resume_run_dir / "final_corrected_face_flux.npy"
    resume_velocity_path = resume_run_dir / "final_cell_velocity.npy"
    resume_pressure_path = resume_run_dir / "final_pressure.npy"
    required_paths = (
        resume_summary_path,
        resume_manifest_path,
        resume_flux_path,
        resume_velocity_path,
        resume_pressure_path,
    )
    missing_resume_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_resume_paths:
        raise FileNotFoundError(
            "Cannot resume flow run; missing artifacts: "
            + ", ".join(missing_resume_paths)
        )

    resume_summary = json.loads(resume_summary_path.read_text(encoding="utf-8"))
    if not isinstance(resume_summary, dict):
        raise ValueError("Resume summary must contain a JSON object.")
    recorded_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(recorded_manifest, dict):
        raise ValueError("Resume manifest must contain a JSON object.")
    _validate_flow_resume_manifest(recorded_manifest, expected_manifest)
    resume_start_step, resume_start_time = _validate_flow_resume_checkpoint(
        recorded_manifest,
        run_dir=resume_run_dir,
        summary=resume_summary,
    )

    resume_face_flux = np.load(resume_flux_path, allow_pickle=False)
    resume_cell_velocity = np.load(resume_velocity_path, allow_pickle=False)
    resume_pressure = np.load(resume_pressure_path, allow_pickle=False)
    if resume_face_flux.shape != (mesh.face_vertices.shape[0],):
        raise ValueError(
            "Resume face flux shape does not match mesh faces: "
            f"{resume_face_flux.shape} vs {(mesh.face_vertices.shape[0],)}"
        )
    if resume_cell_velocity.shape != (mesh.tetrahedra.shape[0], 3):
        raise ValueError(
            "Resume cell velocity shape does not match mesh cells: "
            f"{resume_cell_velocity.shape} vs {(mesh.tetrahedra.shape[0], 3)}"
        )
    if resume_pressure.shape != (mesh.tetrahedra.shape[0],):
        raise ValueError(
            "Resume pressure shape does not match mesh cells: "
            f"{resume_pressure.shape} vs {(mesh.tetrahedra.shape[0],)}"
        )
    if not all(
        np.all(np.isfinite(array))
        for array in (resume_face_flux, resume_cell_velocity, resume_pressure)
    ):
        raise ValueError("Resume state arrays contain NaN or Inf.")

    resume_metadata = {
        "enabled": True,
        "source_run_dir": str(resume_run_dir),
        "source_summary_json": str(resume_summary_path),
        "source_resume_manifest_json": str(resume_manifest_path),
        "source_resume_fingerprint": str(recorded_manifest["fingerprint"]),
        "source_flow_steps_completed": int(resume_start_step),
        "source_physical_time_final": float(resume_start_time),
    }
    state = TetraFlowState(
        face_flux=np.asarray(resume_face_flux, dtype=np.float64),
        cell_velocity=np.asarray(resume_cell_velocity, dtype=np.float64),
        pressure=np.asarray(resume_pressure, dtype=np.float64),
        diagnostics={"resume": dict(resume_metadata)},
    )
    return state, resume_start_step, resume_start_time, resume_metadata


def main() -> None:
    process_started_at = datetime.now(timezone.utc)
    process_started_perf = perf_counter()
    from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
        TetraFlowConfig,
        _assemble_poisson_rhs,
        _build_pressure_system_coefficients,
        _compute_cell_flux_sum,
        _pressure_face_gradient_flux,
        _pressure_nonorthogonal_rhs_term,
        _pressure_matrix_explicit_audit,
        _pressure_matrixfree_vs_explicit_audit,
        _pressure_operator_spd_audit,
        _solve_pressure_reference_explicit,
        _top_divergence_cells,
        apply_tetra_convective_predictor,
        apply_tetra_stokes_viscous_predictor,
        compute_tetra_convective_cfl_rate,
        compute_tetra_flux_divergence,
        initialize_tetra_flow_state,
        solve_tetra_pressure_projection,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_mesh_loader import (
        load_imported_tetra_mesh_npz,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
        resolve_inlet_face_groups,
    )
    from microfluidics.preprocessor import (
        apply_flow_profile_to_mesh,
        case_config_from_mapping,
        compile_flow_runtime_profile,
        load_case_config,
        resolve_case_mesh_path,
    )

    raw_argv = list(sys.argv[1:])
    startup_bootstrap_cap_explicit = option_was_explicitly_provided(
        raw_argv, "--startup-bootstrap-max-steps"
    )
    pressure_solver_explicit = option_was_explicitly_provided(
        raw_argv, "--pressure-solver"
    )
    viscous_predictor_explicit = option_was_explicitly_provided(
        raw_argv, "--viscous-predictor-mode"
    )
    flow_dt_min_explicit = option_was_explicitly_provided(raw_argv, "--flow-dt-min")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="",
        help="Deprecated compatibility flag. Flow debug runs now require --mesh-npz.",
    )
    parser.add_argument("--mesh-npz", type=str, default="")
    parser.add_argument(
        "--case-config",
        type=str,
        default="",
        help=(
            "Optional case_config_v1 JSON. When omitted, an embedded case from the "
            "imported mesh is used automatically."
        ),
    )
    parser.add_argument(
        "--import-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_IMPORT_RUNS_ROOT_REL)),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_TETRA_FLOW_RUNS_ROOT_REL)),
    )
    parser.add_argument(
        "--resume-flow-run-dir",
        type=str,
        default="",
        help=(
            "Continue from a previous flow debug run directory containing "
            "final_corrected_face_flux.npy, final_cell_velocity.npy, final_pressure.npy, "
            "summary.json, and resume_manifest.json."
        ),
    )
    parser.add_argument("--run-source-sha256", type=str, default="")
    parser.add_argument("--run-runtime-identifier", type=str, default="")
    parser.add_argument("--run-input-mesh-sha256", type=str, default="")
    parser.add_argument("--resume-request-fingerprint", type=str, default="")
    parser.add_argument(
        "--backend", type=str, choices=("auto", "numpy", "torch"), default="auto"
    )
    parser.add_argument(
        "--flow-execution-backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--fail-if-numpy-fallback", action="store_true")
    parser.add_argument("--inlet-speed", type=float, default=0.15)
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--kinematic-viscosity", type=float, default=1e-6)
    parser.add_argument(
        "--wall-velocity-boundary-mode",
        type=str,
        choices=("slip", "no_slip", "no_slip_tangential", "no_slip_legacy_isotropic"),
        default="slip",
    )
    parser.add_argument("--wall-tangential-no-slip-strength", type=float, default=1.0)
    parser.add_argument(
        "--wall-tangential-no-slip-strength-ramp-start",
        type=float,
        default=None,
        help=(
            "Optional start value for a per-step linear ramp of wall tangential "
            "no-slip strength. Defaults to 0.0 when ramp steps are requested."
        ),
    )
    parser.add_argument(
        "--wall-tangential-no-slip-strength-ramp-steps",
        type=int,
        default=0,
        help=(
            "Number of local flow steps used to ramp wall tangential no-slip "
            "strength from the ramp start to --wall-tangential-no-slip-strength. "
            "Zero keeps the historical constant-strength behavior."
        ),
    )
    parser.add_argument(
        "--wall-tangential-shear-face-flux-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wall-tangential-cell-velocity-momentum-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wall-flux-stokes-resistance-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wall-flux-stokes-resistance-strength", type=float, default=1.0
    )
    parser.add_argument("--projection-dt", type=float, default=5e-4)
    parser.add_argument("--flow-steps", type=int, default=20)
    parser.add_argument(
        "--flow-stop-physical-time",
        type=float,
        default=None,
        help="Stop at this absolute physical time; --flow-steps remains a safety cap.",
    )
    parser.add_argument(
        "--startup-bootstrap-max-steps",
        type=int,
        default=DEFAULT_STARTUP_BOOTSTRAP_MAX_STEPS,
        help=(
            "Total bounded pseudo-time qualification cap. The default preserves the "
            "legacy 20-step search budget plus the fixed qualification tail."
        ),
    )
    parser.add_argument("--flow-dt", type=float, default=None)
    parser.add_argument(
        "--flow-dt-mode",
        type=str,
        choices=("manual", "auto_cfl"),
        default="auto_cfl",
    )
    parser.add_argument("--flow-dt-min", type=float, default=1e-7)
    parser.add_argument("--flow-dt-max", type=float, default=None)
    parser.add_argument("--convective-cfl-target", type=float, default=0.5)
    parser.add_argument(
        "--flow-mode",
        type=str,
        choices=(
            "projection_only",
            "stokes_viscous_projection",
            "navier_stokes_projection_debug",
        ),
        default="navier_stokes_projection_debug",
    )
    parser.add_argument(
        "--compare-flow-modes",
        type=str,
        default="",
        help=(
            "Comma-separated flow modes to compare, e.g. "
            "stokes_viscous_projection,navier_stokes_projection_debug"
        ),
    )
    parser.add_argument(
        "--compare-ns-dt-modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare manual dt+auto_damping vs auto_cfl modes for NS prototype.",
    )
    parser.add_argument(
        "--enable-convective-predictor",
        action="store_true",
        help="Enable explicit convective predictor (used by navier_stokes_projection_debug by default).",
    )
    parser.add_argument(
        "--disable-convective-predictor",
        action="store_true",
        help="Disable convective predictor regardless of flow mode.",
    )
    parser.add_argument(
        "--disable-convective-auto-damping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable convective CFL-based auto damping shrink in auto_damping stabilization mode.",
    )
    parser.add_argument("--convective-cfl-limit", type=float, default=0.5)
    parser.add_argument("--convective-cfl-acceptance-eps", type=float, default=1e-9)
    parser.add_argument("--convective-predictor-damping", type=float, default=1.0)
    parser.add_argument(
        "--convective-stabilization-mode",
        type=str,
        choices=("auto_damping", "substepping"),
        default="auto_damping",
    )
    parser.add_argument(
        "--convective-substep-boundary-contract",
        type=str,
        choices=("end_only", "every_substep"),
        default="end_only",
    )
    parser.add_argument("--max-convective-substeps", type=int, default=128)
    parser.add_argument(
        "--fail-on-convective-substep-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--audit-convective-cfl",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compare-convective-stabilization-modes",
        type=str,
        default="",
        help="Comma-separated stabilization modes to compare, e.g. auto_damping,substepping",
    )
    parser.add_argument(
        "--viscous-predictor-mode",
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
        "--viscous-predictor-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="auto",
        help=(
            "Viscous-predictor outlet flux contract; auto preserves the "
            "predicted outlet profile for no-slip and retains match_inlet for slip."
        ),
    )
    parser.add_argument(
        "--viscous-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="auto",
        help=(
            "Vector viscous face-gradient correction. deferred_lsq adds a "
            "cell-LSQ K·grad flux with full no-slip wall and prescribed-inlet "
            "Dirichlet momentum boundaries; auto selects it only for no-slip."
        ),
    )
    parser.add_argument(
        "--pressure-projection-outlet-contract-mode",
        type=str,
        choices=("auto", "match_inlet", "preserve"),
        default="auto",
        help=(
            "Pre-projection pressure-outlet face-flux contract: preserve the "
            "predicted outlet profile or replace it with legacy uniform match_inlet."
        ),
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-mode",
        type=str,
        choices=("auto", "none", "deferred_lsq"),
        default="auto",
        help=(
            "Pressure face-gradient geometry correction. deferred_lsq keeps the "
            "legacy SPD matrix and adds frozen LSQ non-orthogonal flux to the RHS."
        ),
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-sweeps",
        type=int,
        default=4,
        help="Deferred LSQ fixed-point pressure solves per projection step.",
    )
    parser.add_argument(
        "--pressure-nonorthogonal-correction-relaxation",
        type=float,
        default=1.0,
        help=(
            "Persistent deferred non-orthogonal face-flux relaxation in (0, 1] "
            "(default: 1.0; monitor the outer fixed-point defect on new meshes)."
        ),
    )
    parser.add_argument(
        "--viscous-face-flux-divergence-impact-cap",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--viscous-face-flux-laplacian-vectorized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-cuda-viscosity-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the Torch CUDA implementation for eligible slip face-flux "
            "viscosity; disable for controlled same-build NumPy comparisons."
        ),
    )
    parser.add_argument(
        "--run-stokes-sensitivity-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run stokes baseline sweep over flow dt and viscous face-flux cap values.",
    )
    parser.add_argument(
        "--sweep-flow-dts",
        type=str,
        default="1e-4,2.5e-4,5e-4,1e-3",
        help="Comma-separated flow_dt values for stokes baseline sweep.",
    )
    parser.add_argument(
        "--sweep-viscous-face-flux-caps",
        type=str,
        default="default,0.1,0.03,0.01",
        help="Comma-separated cap values for sweep; use default/none for current cap.",
    )
    parser.add_argument(
        "--run-convective-sensitivity-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run navier_stokes convective sensitivity sweep over flow_dt and damping.",
    )
    parser.add_argument(
        "--convective-sweep-flow-dts",
        type=str,
        default="1e-4,2.5e-4,5e-4,1e-3",
        help="Comma-separated flow_dt values for convective sensitivity sweep.",
    )
    parser.add_argument(
        "--convective-sweep-dampings",
        type=str,
        default="1.0,0.5,0.25",
        help="Comma-separated convective predictor damping values for sweep.",
    )
    add_pipeline_manifest_arguments(parser)
    parser.add_argument(
        "--compare-viscous-predictor-modes",
        type=str,
        default="",
        help="Comma-separated viscous predictor modes to compare in stokes flow mode.",
    )
    parser.add_argument("--snapshot-steps", type=str, default="")
    parser.add_argument(
        "--snapshot-time-interval",
        type=float,
        default=None,
        help="Save heavy snapshots when physical time crosses this interval.",
    )
    parser.add_argument("--allow-projection-warning-steps", type=int, default=0)
    parser.add_argument(
        "--startup-warning-steps",
        type=int,
        default=10,
        help="If >=0, treat failed steps 1..K as startup transients if safety checks pass. "
        "If -1, falls back to --allow-projection-warning-steps for compatibility.",
    )
    parser.add_argument("--max-pressure-iterations", type=int, default=1000)
    parser.add_argument("--pressure-tolerance", type=float, default=1e-8)
    parser.add_argument("--pressure-relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--divergence-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--pressure-solver",
        type=str,
        choices=("jacobi", "cg", "pcg_diag", "amg_pcg"),
        default=DEFAULT_TETRA_FLOW_DEBUG_PRESSURE_SOLVER,
    )
    parser.add_argument("--relaxation-omega", type=float, default=1.0)
    parser.add_argument(
        "--projection-sign", type=str, choices=("minus", "plus"), default="minus"
    )
    parser.add_argument(
        "--projection-rhs-mode",
        type=str,
        choices=("divergence_per_volume", "volume_integrated_flux"),
        default="volume_integrated_flux",
    )
    parser.add_argument("--projection-correction-damping", type=float, default=1.0)
    parser.add_argument(
        "--projection-cell-velocity-update-mode",
        type=str,
        choices=("auto", "legacy_reconstruct", "momentum_pressure_corrected"),
        default="auto",
        help=(
            "Cell-momentum state update across projection. auto preserves the "
            "legacy slip path and selects the coherent update for no-slip."
        ),
    )
    parser.add_argument(
        "--projection-correction-limit-mode",
        type=str,
        choices=("none", "cell_divergence_cap", "face_flux_cap", "redistribute_local"),
        default="none",
    )
    parser.add_argument("--projection-divergence-cap-factor", type=float, default=2.0)
    parser.add_argument("--projection-divergence-floor", type=float, default=1e-12)
    parser.add_argument(
        "--projection-face-correction-over-volume-cap", type=float, default=8000.0
    )
    parser.add_argument("--cg-breakdown-eps", type=float, default=1e-30)
    parser.add_argument("--cg-stagnation-window", type=int, default=25)
    parser.add_argument("--cg-stagnation-ratio", type=float, default=0.995)
    parser.add_argument(
        "--pcg-require-relative-l2-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the relative L2 residual target for PCG instead of accepting "
            "either the L2 or max-relative target, and verify convergence with "
            "a freshly computed r=b-Ap residual."
        ),
    )
    parser.add_argument(
        "--pressure-relative-tolerance-effective",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--pressure-relative-tolerance-max-effective",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--projection-divergence-reduction-l2-tolerance",
        type=float,
        default=5e-3,
    )
    parser.add_argument(
        "--projection-divergence-reduction-linf-tolerance",
        type=float,
        default=5e-3,
    )
    parser.add_argument("--projection-final-div-l2-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--projection-final-div-max-tolerance", type=float, default=20.0
    )
    parser.add_argument("--outlet-inlet-flux-ratio-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--net-boundary-flux-relative-tolerance", type=float, default=5e-3
    )
    parser.add_argument("--wall-flux-abs-tolerance", type=float, default=1e-14)
    parser.add_argument(
        "--compare-pressure-solvers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--audit-pressure-operator",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--compare-correction-limit-modes", action="store_true")
    parser.add_argument("--compare-outlet-projection-modes", action="store_true")
    parser.add_argument(
        "--timing-mode",
        choices=("basic", "detailed"),
        default="basic",
        help=(
            "basic records canonical wall-clock timing with no extra component "
            "CUDA synchronizations; detailed synchronizes component boundaries."
        ),
    )
    parser.add_argument(
        "--postprocessing-mode",
        choices=("full", "minimal"),
        default="full",
        help=(
            "full writes the existing plots and VTU output; minimal keeps numerical "
            "reports and reusable state arrays but skips visualization artifacts."
        ),
    )
    parser.add_argument(
        "--cuda-determinism",
        choices=("off", "warn", "error"),
        default="off",
        help=(
            "Opt-in Torch CUDA determinism diagnostics. warn records PyTorch "
            "nondeterminism warnings; error raises them. This is not a production "
            "benchmark mode."
        ),
    )
    parser.add_argument(
        "--fixed-work-source-run-dir",
        type=str,
        default="",
        help=(
            "Opt-in source run for a fixed-work resume benchmark. Its final state "
            "arrays are validated and hashed before use."
        ),
    )
    parser.add_argument(
        "--fixed-work-legacy-source-mesh-sha256",
        type=str,
        default="",
        help=(
            "Explicit mesh hash for a historical fixed-work source whose summary "
            "predates recorded mesh_sha256 provenance."
        ),
    )
    parser.add_argument(
        "--fixed-work-steps",
        type=int,
        default=0,
        help="Exact physical steps for --fixed-work-source-run-dir (must be positive).",
    )
    parser.add_argument(
        "--fixed-work-dt",
        type=float,
        default=1.0e-5,
        help="Manual fixed-work dt; default follows the prior benchmark's ~9.66e-6 auto-CFL mean.",
    )
    parser.add_argument(
        "--pressure-determinism-diagnostic",
        action="store_true",
        help=(
            "Run one CPU and three CUDA PCG/projection comparisons from a fixed-work "
            "source state, then write pressure diagnostic artifacts and exit."
        ),
    )
    args = parser.parse_args()
    global _POSTPROCESSING_MODE
    _POSTPROCESSING_MODE = str(args.postprocessing_mode)
    if _POSTPROCESSING_MODE == "full" and plt is None:
        parser.error("--postprocessing-mode full requires matplotlib")
    if (
        str(args.resume_flow_run_dir).strip()
        and not str(args.run_source_sha256).strip()
    ):
        parser.error(
            "--resume-flow-run-dir requires an explicit --run-source-sha256 "
            "for the complete staged solver source."
        )
    fixed_work_active = bool(str(args.fixed_work_source_run_dir).strip())
    if str(args.fixed_work_legacy_source_mesh_sha256).strip() and not fixed_work_active:
        parser.error(
            "--fixed-work-legacy-source-mesh-sha256 requires "
            "--fixed-work-source-run-dir."
        )
    if fixed_work_active:
        if int(args.fixed_work_steps) <= 0 and not bool(
            args.pressure_determinism_diagnostic
        ):
            parser.error(
                "--fixed-work-steps must be positive with --fixed-work-source-run-dir."
            )
        if float(args.fixed_work_dt) <= 0.0:
            parser.error("--fixed-work-dt must be positive.")
        if str(args.resume_flow_run_dir).strip():
            parser.error(
                "Fixed-work mode owns resume; do not also pass --resume-flow-run-dir."
            )
        if str(args.snapshot_steps).strip():
            parser.error("Fixed-work mode disables snapshots; omit --snapshot-steps.")
        if str(args.pressure_solver) != "pcg_diag":
            parser.error("Fixed-work diagnostics require --pressure-solver pcg_diag.")
        args.resume_flow_run_dir = str(args.fixed_work_source_run_dir)
        args.flow_steps = int(args.fixed_work_steps)
        args.flow_dt_mode = "manual"
        args.flow_dt = float(args.fixed_work_dt)
        args.projection_dt = float(args.fixed_work_dt)
        args.flow_dt_min = float(args.fixed_work_dt)
        args.flow_dt_max = float(args.fixed_work_dt)
    if bool(args.pressure_determinism_diagnostic) and not fixed_work_active:
        parser.error(
            "--pressure-determinism-diagnostic requires --fixed-work-source-run-dir."
        )
    diagnostic_cuda_device = "cuda:0"
    if bool(args.pressure_determinism_diagnostic) and str(args.device).strip():
        diagnostic_cuda_device = str(args.device)
        if not diagnostic_cuda_device.startswith("cuda"):
            parser.error("--pressure-determinism-diagnostic requires a CUDA --device.")
    from microfluidics.gmsh.tetra.pressure_determinism_diagnostics import (
        configure_cuda_determinism,
        sha256_file,
    )

    determinism_report = configure_cuda_determinism(str(args.cuda_determinism))
    if str(args.cuda_determinism) != "off" and not bool(
        determinism_report["diagnostic_success"]
    ):
        raise RuntimeError(
            "Unable to enable requested CUDA determinism diagnostics: "
            f"{determinism_report['exception_type']}: "
            f"{determinism_report['exception_message']}"
        )
    import_root = _normalize_user_path(args.import_root).resolve()
    output_root = _normalize_user_path(args.output_root).resolve()
    acceptance_thresholds_cli = {
        "pressure_relative_tolerance_effective": float(
            args.pressure_relative_tolerance_effective
        ),
        "pressure_relative_tolerance_max_effective": float(
            args.pressure_relative_tolerance_max_effective
        ),
        "projection_divergence_reduction_l2_tolerance": float(
            args.projection_divergence_reduction_l2_tolerance
        ),
        "projection_divergence_reduction_linf_tolerance": float(
            args.projection_divergence_reduction_linf_tolerance
        ),
        "projection_final_div_l2_tolerance": float(
            args.projection_final_div_l2_tolerance
        ),
        "projection_final_div_max_tolerance": float(
            args.projection_final_div_max_tolerance
        ),
        "outlet_inlet_flux_ratio_tolerance": float(
            args.outlet_inlet_flux_ratio_tolerance
        ),
        "net_boundary_flux_relative_tolerance": float(
            args.net_boundary_flux_relative_tolerance
        ),
        "wall_flux_abs_tolerance": float(args.wall_flux_abs_tolerance),
    }
    if int(args.startup_bootstrap_max_steps) < 0:
        parser.error("--startup-bootstrap-max-steps must be non-negative.")

    if not str(args.mesh_npz).strip():
        parser.error(
            "--mesh-npz is required. Automatic import-root/mesh-name resolution "
            "is no longer supported."
        )
    mesh_npz = _normalize_user_path(args.mesh_npz).resolve()
    original_input = str(args.mesh_npz)
    if not mesh_npz.exists():
        raise FileNotFoundError(f"Mesh npz not found: {mesh_npz}")
    mesh_sha256 = sha256_file(mesh_npz)
    mesh_stem = mesh_npz.name.replace("_imported_mesh.npz", "")
    mesh = load_imported_tetra_mesh_npz(mesh_npz)
    runtime_case = None
    runtime_case_source = ""
    if str(args.case_config).strip():
        runtime_case_path = _normalize_user_path(args.case_config).resolve()
        runtime_case = load_case_config(runtime_case_path)
        configured_mesh_path = resolve_case_mesh_path(runtime_case, runtime_case_path)
        if configured_mesh_path != mesh.source_path.resolve():
            raise ValueError(
                f"case mesh resolves to {configured_mesh_path}, but imported mesh "
                f"was built from {mesh.source_path.resolve()}"
            )
        runtime_case_source = str(runtime_case_path)
    elif mesh.case_config:
        runtime_case = case_config_from_mapping(mesh.case_config)
        runtime_case_source = "embedded:mesh_npz"
    case_flow_profile = None
    if runtime_case is not None:
        case_flow_profile = compile_flow_runtime_profile(mesh, runtime_case)
        apply_flow_profile_to_mesh(mesh, case_flow_profile)
        args.inlet_speed = float(case_flow_profile.inlet_speed_m_per_s)
        args.wall_velocity_boundary_mode = str(case_flow_profile.wall_mode)
        if case_flow_profile.density_kg_per_m3 is not None:
            args.density = float(case_flow_profile.density_kg_per_m3)
        if case_flow_profile.kinematic_viscosity_m2_per_s is not None:
            args.kinematic_viscosity = float(
                case_flow_profile.kinematic_viscosity_m2_per_s
            )
    run_dir = create_timestamped_run_dir(output_root, mesh_stem)
    manifest_recorder = build_pipeline_manifest_recorder(
        args,
        stage_type="flow",
        run_dir=run_dir,
    )
    manifest_inputs = {
        "original_input": original_input,
        "resolved_mesh_npz": str(mesh_npz),
        "mesh_sha256": mesh_sha256,
        "import_root": str(import_root),
        "output_root": str(output_root),
        "cuda_determinism": determinism_report,
        "fixed_work_enabled": bool(fixed_work_active),
        "case_config_source": runtime_case_source,
    }
    if bool(args.pressure_determinism_diagnostic):
        manifest_artifacts = _pressure_determinism_manifest_artifacts(run_dir)
        manifest_metadata = {"manifest_role": "pressure_determinism_diagnostic"}
    else:
        manifest_artifacts = {
            "run_log": str(run_dir / "run.log"),
            "config_json": str(run_dir / "config.json"),
            "summary_json": str(run_dir / "summary.json"),
            "resume_manifest_json": str(run_dir / "resume_manifest.json"),
            "acceptance_report_json": str(run_dir / "acceptance_report.json"),
            "flow_diagnostics_json": str(run_dir / "flow_diagnostics.json"),
            "startup_bootstrap_history_json": str(
                run_dir / "startup_bootstrap_history.json"
            ),
            "startup_root_cause_report_json": str(
                run_dir / "startup_root_cause_report.json"
            ),
            "flow_coupling_metadata_json": str(run_dir / "flow_coupling_metadata.json"),
            "final_corrected_face_flux_npy": str(
                run_dir / "final_corrected_face_flux.npy"
            ),
        }
        manifest_metadata = {"manifest_role": "flow_stage"}
    global _ACTIVE_PIPELINE_MANIFEST_RECORDER
    global _ACTIVE_PIPELINE_MANIFEST_INPUTS
    global _ACTIVE_PIPELINE_MANIFEST_ARTIFACTS
    _ACTIVE_PIPELINE_MANIFEST_RECORDER = manifest_recorder
    _ACTIVE_PIPELINE_MANIFEST_INPUTS = manifest_inputs
    _ACTIVE_PIPELINE_MANIFEST_ARTIFACTS = manifest_artifacts
    manifest_recorder.record_started(
        inputs=manifest_inputs,
        artifacts=manifest_artifacts,
        metadata=manifest_metadata,
    )
    flow_mode = str(args.flow_mode)
    (
        convective_predictor_enabled_resolved,
        convective_predictor_default_reason,
    ) = _resolve_convective_predictor_setting(
        flow_mode=flow_mode,
        enable_convective_predictor=bool(args.enable_convective_predictor),
        disable_convective_predictor=bool(args.disable_convective_predictor),
    )
    resolved_viscous_predictor_mode, viscous_predictor_default_reason = (
        _resolve_viscous_predictor_mode(
            flow_mode=flow_mode,
            predictor_mode_cli=str(args.viscous_predictor_mode),
            predictor_mode_explicit=bool(viscous_predictor_explicit),
            wall_velocity_boundary_mode=str(args.wall_velocity_boundary_mode),
        )
    )

    with _tee_logging(run_dir / "run.log"):
        print(
            "[gmsh-tetra-flow] CUDA determinism: "
            f"requested={determinism_report['requested_mode']}, "
            f"effective={determinism_report['effective_mode']}"
        )
        backend = select_backend(args.backend)
        req = str(args.flow_execution_backend)
        if req == "auto":
            exec_backend = backend.selected_backend
            exec_device = backend.device
        elif req == "numpy":
            exec_backend = "numpy"
            exec_device = "cpu"
        else:
            if backend.torch_available:
                exec_backend = "torch"
                exec_device = "cuda:0" if backend.torch_cuda_available else "cpu"
            else:
                exec_backend = "numpy"
                exec_device = "cpu"
        if args.device:
            exec_device = str(args.device)

        cfg = TetraFlowConfig(
            density=float(args.density),
            kinematic_viscosity=float(args.kinematic_viscosity),
            inlet_speed=float(args.inlet_speed),
            pressure_outlet_value=(
                float(case_flow_profile.outlet_pressure_pa)
                if case_flow_profile is not None
                else 0.0
            ),
            max_pressure_iterations=int(args.max_pressure_iterations),
            pressure_tolerance=float(args.pressure_tolerance),
            pressure_relative_tolerance=float(args.pressure_relative_tolerance),
            divergence_tolerance=float(args.divergence_tolerance),
            relaxation_omega=float(args.relaxation_omega),
            backend=exec_backend,  # type: ignore[arg-type]
            device=exec_device,
            pressure_solver=args.pressure_solver,  # type: ignore[arg-type]
            projection_dt=float(args.projection_dt),
            projection_sign=args.projection_sign,  # type: ignore[arg-type]
            projection_rhs_mode=args.projection_rhs_mode,  # type: ignore[arg-type]
            pressure_projection_outlet_contract_mode=str(
                args.pressure_projection_outlet_contract_mode
            ),  # type: ignore[arg-type]
            pressure_nonorthogonal_correction_mode=str(
                args.pressure_nonorthogonal_correction_mode
            ),  # type: ignore[arg-type]
            pressure_nonorthogonal_correction_sweeps=int(
                args.pressure_nonorthogonal_correction_sweeps
            ),
            pressure_nonorthogonal_correction_relaxation=float(
                args.pressure_nonorthogonal_correction_relaxation
            ),
            projection_correction_damping=float(args.projection_correction_damping),
            projection_cell_velocity_update_mode=str(
                args.projection_cell_velocity_update_mode
            ),  # type: ignore[arg-type]
            projection_correction_limit_mode=args.projection_correction_limit_mode,  # type: ignore[arg-type]
            projection_divergence_cap_factor=float(
                args.projection_divergence_cap_factor
            ),
            projection_divergence_floor=float(args.projection_divergence_floor),
            projection_face_correction_over_volume_cap=float(
                args.projection_face_correction_over_volume_cap
            ),
            viscous_predictor_mode=resolved_viscous_predictor_mode,  # type: ignore[arg-type]
            viscous_predictor_outlet_contract_mode=str(
                args.viscous_predictor_outlet_contract_mode
            ),  # type: ignore[arg-type]
            viscous_nonorthogonal_correction_mode=str(
                args.viscous_nonorthogonal_correction_mode
            ),  # type: ignore[arg-type]
            viscous_face_flux_divergence_impact_cap=float(
                args.viscous_face_flux_divergence_impact_cap
            ),
            viscous_face_flux_laplacian_vectorized=bool(
                args.viscous_face_flux_laplacian_vectorized
            ),
            torch_cuda_viscosity_enabled=bool(args.torch_cuda_viscosity_enabled),
            enable_convective_predictor=bool(convective_predictor_enabled_resolved),
            disable_convective_predictor=bool(args.disable_convective_predictor),
            convective_cfl_limit=float(args.convective_cfl_limit),
            convective_predictor_damping=float(args.convective_predictor_damping),
            disable_convective_auto_damping=bool(args.disable_convective_auto_damping),
            convective_stabilization_mode=str(args.convective_stabilization_mode),  # type: ignore[arg-type]
            convective_substep_boundary_contract=str(
                args.convective_substep_boundary_contract
            ),  # type: ignore[arg-type]
            max_convective_substeps=int(args.max_convective_substeps),
            fail_on_convective_substep_cap=bool(args.fail_on_convective_substep_cap),
            wall_velocity_boundary_mode=str(args.wall_velocity_boundary_mode),  # type: ignore[arg-type]
            wall_tangential_no_slip_strength=float(
                args.wall_tangential_no_slip_strength
            ),
            wall_tangential_shear_face_flux_enabled=bool(
                args.wall_tangential_shear_face_flux_enabled
            ),
            wall_tangential_cell_velocity_momentum_enabled=bool(
                args.wall_tangential_cell_velocity_momentum_enabled
            ),
            wall_flux_stokes_resistance_enabled=bool(
                args.wall_flux_stokes_resistance_enabled
            ),
            wall_flux_stokes_resistance_strength=float(
                args.wall_flux_stokes_resistance_strength
            ),
            enable_sign_comparison=True,
            cg_breakdown_eps=float(args.cg_breakdown_eps),
            cg_stagnation_window=int(args.cg_stagnation_window),
            cg_stagnation_ratio=float(args.cg_stagnation_ratio),
            pcg_require_relative_l2_convergence=bool(
                args.pcg_require_relative_l2_convergence
            ),
        )
        cfg_effective = _resolve_flow_config_for_postsolve_audit(cfg)

        print(f"[gmsh-tetra-flow] run directory: {run_dir}")
        print(f"[gmsh-tetra-flow] original input: {original_input}")
        print(f"[gmsh-tetra-flow] mesh npz: {mesh_npz}")
        print(
            "[gmsh-tetra-flow] backend: "
            f"requested={backend.requested_backend}, selected={backend.selected_backend}, "
            f"exec_backend={exec_backend}, exec_device={exec_device}"
        )
        for note in backend.notes:
            print(f"[gmsh-tetra-flow] note: {note}")

        wall_strength_ramp_steps = int(args.wall_tangential_no_slip_strength_ramp_steps)
        if wall_strength_ramp_steps < 0:
            raise ValueError(
                "wall_tangential_no_slip_strength_ramp_steps must be non-negative."
            )
        wall_strength_ramp_target = float(args.wall_tangential_no_slip_strength)
        wall_strength_ramp_start = (
            float(args.wall_tangential_no_slip_strength_ramp_start)
            if args.wall_tangential_no_slip_strength_ramp_start is not None
            else (0.0 if wall_strength_ramp_steps > 0 else wall_strength_ramp_target)
        )
        resume_requested_flow_dt = (
            float(args.flow_dt)
            if args.flow_dt is not None
            else float(cfg.projection_dt)
        )
        resume_flow_dt_max = (
            float(args.flow_dt_max)
            if args.flow_dt_max is not None
            else resume_requested_flow_dt
        )
        flow_resume_manifest = _build_flow_resume_manifest(
            mesh_npz=mesh_npz,
            cfg=cfg_effective,
            source_sha256=str(args.run_source_sha256),
            runtime_identifier=str(args.run_runtime_identifier),
            input_mesh_sha256=str(args.run_input_mesh_sha256),
            request_fingerprint=str(args.resume_request_fingerprint),
            flow_mode=flow_mode,
            flow_dt_mode=str(args.flow_dt_mode),
            requested_flow_dt=resume_requested_flow_dt,
            flow_dt_min=float(args.flow_dt_min),
            flow_dt_max=resume_flow_dt_max,
            convective_cfl_target=float(args.convective_cfl_target),
            wall_strength_ramp_start=wall_strength_ramp_start,
            wall_strength_ramp_target=wall_strength_ramp_target,
            wall_strength_ramp_steps=wall_strength_ramp_steps,
        )

        state0 = initialize_tetra_flow_state(mesh, cfg)
        startup_bootstrap_seconds = 0.0
        solver_step_durations: list[float] = []
        flow_loop_step_durations: list[float] = []
        component_seconds = {
            "convection_total_seconds": 0.0,
            "viscous_predictor_total_seconds": 0.0,
            "pressure_projection_total_seconds": 0.0,
        }
        synchronization_telemetry: dict[str, Any] = {
            "cuda_active": False,
            "setup_boundary_synchronization": False,
            "startup_bootstrap_boundary_synchronization": False,
            "flow_boundary_synchronization": False,
            "component_boundary_synchronization": False,
            "cuda_synchronization_count": 0,
        }
        resume_enabled = bool(str(args.resume_flow_run_dir).strip())
        resume_start_step = 0
        resume_start_time = 0.0
        resume_metadata: dict[str, Any] = {
            "enabled": bool(resume_enabled),
            "source_run_dir": "",
            "source_summary_json": "",
            "source_resume_manifest_json": "",
            "source_resume_fingerprint": "",
            "source_flow_steps_completed": 0,
            "source_physical_time_final": 0.0,
        }
        if resume_enabled and not fixed_work_active:
            (
                state0,
                resume_start_step,
                resume_start_time,
                resume_metadata,
            ) = _load_flow_resume_state(
                mesh,
                args.resume_flow_run_dir,
                expected_manifest=flow_resume_manifest,
            )
            print(
                "[gmsh-tetra-flow] resume from: "
                f"{resume_metadata['source_run_dir']} (step={resume_start_step}, "
                f"physical_time={resume_start_time:.12g})"
            )
        fixed_work_manifest: dict[str, Any] = {}
        if fixed_work_active:
            from microfluidics.gmsh.tetra.pressure_determinism_diagnostics import (
                load_fixed_work_state,
                write_json as write_determinism_json,
            )

            state0, fixed_work_manifest = load_fixed_work_state(
                source_run_dir=_normalize_user_path(
                    args.fixed_work_source_run_dir
                ).resolve(),
                mesh=mesh,
                mesh_npz=mesh_npz,
                legacy_source_mesh_sha256=(
                    str(args.fixed_work_legacy_source_mesh_sha256).strip() or None
                ),
            )
            fixed_work_summary_path = Path(
                str(fixed_work_manifest["source_summary_json"])
            )
            fixed_work_summary = json.loads(
                fixed_work_summary_path.read_text(encoding="utf-8")
            )
            resume_start_step = int(
                fixed_work_summary.get(
                    "flow_steps_completed_total",
                    fixed_work_summary.get("flow_steps_completed", 0),
                )
            )
            resume_start_time = float(
                fixed_work_summary.get("physical_time_final", 0.0)
            )
            if resume_start_step < 0:
                raise ValueError("Fixed-work source flow step must be non-negative.")
            if not np.isfinite(resume_start_time) or resume_start_time < 0.0:
                raise ValueError(
                    "Fixed-work source physical time must be finite and non-negative."
                )
            resume_metadata.update(
                {
                    "source_run_dir": str(fixed_work_manifest["source_run_dir"]),
                    "source_summary_json": str(fixed_work_summary_path),
                    "source_flow_steps_completed": resume_start_step,
                    "source_physical_time_final": resume_start_time,
                }
            )
            fixed_work_manifest.update(
                {
                    "manual_dt": float(args.fixed_work_dt),
                    "flow_steps": int(args.fixed_work_steps),
                    "pressure_solver": str(args.pressure_solver),
                    "wall_velocity_boundary_mode": str(
                        args.wall_velocity_boundary_mode
                    ),
                    "snapshots_disabled": True,
                    "bootstrap_skipped_via_resume": True,
                    "command_line": list(sys.argv),
                }
            )
            write_determinism_json(
                run_dir / "fixed_work_manifest.json", fixed_work_manifest
            )
        if bool(args.pressure_determinism_diagnostic):
            from microfluidics.gmsh.tetra.pressure_determinism_diagnostics import (
                run_pressure_determinism_diagnostic,
            )

            report = run_pressure_determinism_diagnostic(
                mesh=mesh,
                state=state0,
                config=cfg,
                output_dir=run_dir,
                cuda_device=diagnostic_cuda_device,
            )
            _complete_pressure_determinism_run(
                run_dir=run_dir,
                manifest_recorder=manifest_recorder,
                manifest_inputs=manifest_inputs,
                manifest_artifacts=manifest_artifacts,
                mesh_npz=mesh_npz,
                mesh_sha256=mesh_sha256,
                cli_args=vars(args),
                command_line=[sys.executable, *sys.argv],
                exec_backend=exec_backend,
                exec_device=exec_device,
                determinism_report=determinism_report,
                fixed_work_manifest=fixed_work_manifest,
                report=report,
            )
            print("[gmsh-tetra-flow] pressure determinism diagnostic complete")
            return
        inlet = resolve_inlet_face_groups(mesh)
        left_faces = np.asarray(inlet["left_faces"], dtype=np.int64)
        right_faces = np.asarray(inlet["right_faces"], dtype=np.int64)
        div_before = compute_tetra_flux_divergence(
            mesh,
            state0.face_flux,
            left_inlet_faces=left_faces,
            right_inlet_faces=right_faces,
            outlet_faces=mesh.outlet_faces,
            wall_faces=mesh.wall_faces,
        )
        flow_steps_requested = max(int(args.flow_steps), 1)
        requested_flow_dt = (
            float(args.flow_dt)
            if (args.flow_dt is not None)
            else float(cfg.projection_dt)
        )
        flow_dt_mode = _parse_flow_dt_mode(str(args.flow_dt_mode))
        flow_dt = float(requested_flow_dt)
        flow_dt_min = float(args.flow_dt_min)
        flow_dt_max = (
            float(args.flow_dt_max)
            if (args.flow_dt_max is not None)
            else float(requested_flow_dt)
        )
        convective_cfl_target = float(args.convective_cfl_target)
        startup_bootstrap_max_steps = int(args.startup_bootstrap_max_steps)
        snapshot_steps = _parse_snapshot_steps(str(args.snapshot_steps))
        flow_stop_physical_time = (
            float(args.flow_stop_physical_time)
            if args.flow_stop_physical_time is not None
            else None
        )
        if flow_stop_physical_time is not None and (
            not np.isfinite(flow_stop_physical_time)
            or flow_stop_physical_time <= resume_start_time
        ):
            raise ValueError(
                "flow_stop_physical_time must be finite and greater than the "
                "initial physical time"
            )
        snapshot_time_interval = (
            float(args.snapshot_time_interval)
            if args.snapshot_time_interval is not None
            else None
        )
        if snapshot_time_interval is not None and (
            not np.isfinite(snapshot_time_interval) or snapshot_time_interval <= 0.0
        ):
            raise ValueError("snapshot_time_interval must be finite and positive")
        next_snapshot_time = (
            _next_snapshot_time(resume_start_time, snapshot_time_interval)
            if snapshot_time_interval is not None
            else None
        )
        progression_cfg = replace(cfg, projection_dt=float(flow_dt))
        progression_history: list[dict[str, Any]] = []
        flow_progression_enabled = bool(flow_steps_requested > 1)
        startup_bootstrap_history: list[dict[str, Any]] = []
        startup_bootstrap_required = bool(
            flow_progression_enabled and (not resume_enabled)
        )
        startup_root_cause_report: dict[str, Any] = _summarize_startup_bootstrap(
            bootstrap_history=[],
            initial_divergence=div_before,
            requested_max_steps=int(startup_bootstrap_max_steps),
            bootstrap_required=bool(startup_bootstrap_required),
            uses_default_budget=not startup_bootstrap_cap_explicit,
            not_run_reason=(
                "bootstrap skipped because physical progression resumes from an "
                "existing flow run"
                if resume_enabled
                else "bootstrap not required for projection-only execution"
            ),
        )
        _record_cuda_synchronization(
            synchronization_telemetry,
            scope="setup_boundary",
            backend=exec_backend,
            device=exec_device,
        )
        setup_seconds = float(perf_counter() - process_started_perf)
        state_curr = state0
        if startup_bootstrap_required and startup_bootstrap_max_steps > 0:
            startup_started_perf = perf_counter()
            state_curr, startup_bootstrap_history, _ = _run_startup_bootstrap(
                mesh=mesh,
                state0=state0,
                cfg=progression_cfg,
                flow_mode=str(flow_mode),
                requested_flow_dt=float(requested_flow_dt),
                flow_dt_mode=str(flow_dt_mode),
                flow_dt_min=float(flow_dt_min),
                flow_dt_max=float(flow_dt_max),
                convective_cfl_target=float(convective_cfl_target),
                acceptance_thresholds=acceptance_thresholds_cli,
                wall_strength_start=float(wall_strength_ramp_start),
                max_steps=int(startup_bootstrap_max_steps),
            )
            _record_cuda_synchronization(
                synchronization_telemetry,
                scope="startup_bootstrap_boundary",
                backend=exec_backend,
                device=exec_device,
            )
            startup_bootstrap_seconds = float(perf_counter() - startup_started_perf)
            startup_root_cause_report = _summarize_startup_bootstrap(
                bootstrap_history=startup_bootstrap_history,
                initial_divergence=div_before,
                requested_max_steps=int(startup_bootstrap_max_steps),
                bootstrap_required=True,
                uses_default_budget=not startup_bootstrap_cap_explicit,
            )
        if startup_bootstrap_required:
            print(
                "[gmsh-tetra-flow] startup bootstrap: "
                f"requested_cap={startup_bootstrap_max_steps}, "
                f"steps={len(startup_bootstrap_history)}, "
                f"converged={startup_root_cause_report.get('bootstrap_converged', False)}, "
                f"physical_time_advanced="
                f"{startup_root_cause_report.get('bootstrap_physical_time_advanced', 0.0):.12g}"
            )
        convective_cfl_audit_peak: dict[str, Any] = {}
        convective_cfl_audit_peak_step = 0
        physical_time = float(resume_start_time)
        used_dt_values: list[float] = []
        auto_dt_scale_values: list[float] = []
        auto_dt_floor_values: list[float] = []
        auto_dt_min_hit_any = False
        auto_dt_max_hit_any = False
        run_physical_progression = bool(
            (not startup_bootstrap_required)
            or startup_root_cause_report.get("physical_progression_allowed", False)
        )
        if (not run_physical_progression) and flow_progression_enabled:
            print(
                "[gmsh-tetra-flow] physical progression blocked: "
                f"{startup_root_cause_report.get('bootstrap_reason', '')}"
            )
        flow_stepping_started_perf = perf_counter()
        for local_step_idx in range(1, flow_steps_requested + 1):
            if not run_physical_progression:
                break
            step_started_perf = perf_counter()
            if (
                flow_stop_physical_time is not None
                and physical_time
                >= flow_stop_physical_time - 1e-12 * max(flow_stop_physical_time, 1.0)
            ):
                break
            step_idx = int(resume_start_step + local_step_idx)
            cfl_rate_diag = compute_tetra_convective_cfl_rate(
                mesh, np.asarray(state_curr.face_flux, dtype=np.float64)
            )
            cfl_rate_max = float(cfl_rate_diag.get("cfl_rate_max", 0.0))
            cfl_rate_p95 = float(cfl_rate_diag.get("cfl_rate_p95", 0.0))
            raw_cfl_before_dt_selection = float(requested_flow_dt * cfl_rate_max)
            dt_candidate = float(requested_flow_dt)
            auto_dt_scale_factor = 1.0
            auto_dt_min_hit = False
            auto_dt_max_hit = False
            auto_dt_floor_used = float(flow_dt_min)
            if (
                flow_dt_mode == "auto_cfl"
                and flow_mode == "navier_stokes_projection_debug"
            ):
                dt_from_cfl = float(
                    convective_cfl_target / max(cfl_rate_max, 1e-30)
                    if cfl_rate_max > 0.0
                    else flow_dt_max
                )
                if bool(flow_dt_min_explicit):
                    auto_dt_floor_used = float(flow_dt_min)
                else:
                    auto_dt_floor_used = float(
                        max(1e-12, min(float(flow_dt_min), 0.05 * dt_from_cfl))
                    )
                auto_dt_min_hit = bool(dt_from_cfl < auto_dt_floor_used)
                auto_dt_max_hit = bool(dt_from_cfl > flow_dt_max)
                dt_candidate = float(
                    min(max(dt_from_cfl, auto_dt_floor_used), flow_dt_max)
                )
                auto_dt_scale_factor = float(
                    dt_candidate / max(requested_flow_dt, 1e-30)
                )
            if flow_stop_physical_time is not None:
                dt_candidate = _clamp_flow_dt_to_stop_time(
                    physical_time=physical_time,
                    dt_candidate=dt_candidate,
                    stop_physical_time=flow_stop_physical_time,
                )
                auto_dt_scale_factor = float(
                    dt_candidate / max(requested_flow_dt, 1e-30)
                )
            used_dt_values.append(float(dt_candidate))
            auto_dt_scale_values.append(float(auto_dt_scale_factor))
            auto_dt_floor_values.append(float(auto_dt_floor_used))
            auto_dt_min_hit_any = bool(auto_dt_min_hit_any or auto_dt_min_hit)
            auto_dt_max_hit_any = bool(auto_dt_max_hit_any or auto_dt_max_hit)
            physical_time += float(dt_candidate)
            raw_cfl_after_dt_selection_max = float(dt_candidate * cfl_rate_max)
            raw_cfl_after_dt_selection_p95 = float(dt_candidate * cfl_rate_p95)
            step_wall_strength = _linear_ramp_value(
                start=float(wall_strength_ramp_start),
                target=float(wall_strength_ramp_target),
                local_step_idx=int(local_step_idx),
                ramp_steps=int(wall_strength_ramp_steps),
            )
            step_cfg = replace(
                progression_cfg,
                projection_dt=float(dt_candidate),
                wall_tangential_no_slip_strength=float(step_wall_strength),
            )
            convective_step_diag: dict[str, Any] = {
                "convective_predictor_used": False,
                "convective_execution_backend": "numpy",
                "convective_execution_device": "cpu",
                "convective_torch_cuda_used": False,
                "convective_numpy_fallback_reason": "",
                "convective_predictor_disabled_by_flag": bool(
                    step_cfg.disable_convective_predictor
                ),
                "convective_stabilization_mode": str(
                    step_cfg.convective_stabilization_mode
                ),
                "convective_substep_boundary_contract_mode": str(
                    step_cfg.convective_substep_boundary_contract
                ),
                "convective_cfl_limit": float(step_cfg.convective_cfl_limit),
                "convective_cfl_raw_max": 0.0,
                "convective_cfl_raw_p95": 0.0,
                "convective_cfl_effective_max": 0.0,
                "convective_cfl_effective_p95": 0.0,
                "convective_cfl_warning_raw": False,
                "convective_cfl_warning_effective": False,
                "convective_cfl_max": 0.0,
                "convective_cfl_p95": 0.0,
                "convective_cfl_warning": False,
                "convective_predictor_damping_requested": float(
                    step_cfg.convective_predictor_damping
                ),
                "convective_predictor_damping": float(
                    step_cfg.convective_predictor_damping
                ),
                "convective_predictor_cfl_scale": 1.0,
                "convective_predictor_damping_effective": 0.0,
                "convective_auto_damping_used": False,
                "convective_auto_damping_reason": "convective predictor disabled",
                "convective_substepping_used": False,
                "convective_substep_count": 1,
                "convective_substep_count_unclamped": 1,
                "convective_substep_cap_hit": False,
                "convective_substep_dt": float(dt_candidate),
                "convective_cfl_per_substep_max": 0.0,
                "convective_cfl_per_substep_p95": 0.0,
                "convective_substepping_runtime_seconds": 0.0,
                "convective_dt": float(dt_candidate),
                "convective_dt_effective": 0.0,
                "convective_delta_velocity_max": 0.0,
                "convective_delta_velocity_l2": 0.0,
                "kinetic_energy_before_convection": 0.0,
                "kinetic_energy_after_convection": 0.0,
                "divergence_before_convection_max": 0.0,
                "divergence_before_convection_l2": 0.0,
                "divergence_after_convection_before_projection_max": 0.0,
                "divergence_after_convection_before_projection_l2": 0.0,
                "convective_cfl_definition_report": {},
                "top_convective_cfl_cells": {},
                "top_convective_cfl_faces": {},
            }
            viscous_step_diag: dict[str, Any] = {
                "viscous_predictor_used": False,
                "viscous_predictor_mode": "none",
                "viscous_execution_backend": "numpy",
                "viscous_execution_device": "cpu",
                "viscous_torch_cuda_used": False,
                "viscous_cuda_no_slip_used": False,
                "viscous_numpy_fallback_reason": "",
                "kinematic_viscosity": float(step_cfg.kinematic_viscosity),
                "viscous_dt": float(dt_candidate),
                "viscous_stability_metric": 0.0,
                "viscous_stability_warning": False,
                "viscous_delta_velocity_max": 0.0,
                "viscous_delta_velocity_l2": 0.0,
                "kinetic_energy_before_predictor": 0.0,
                "kinetic_energy_after_predictor": 0.0,
            }
            state_after_conv = state_curr
            use_convective = bool(
                (flow_mode == "navier_stokes_projection_debug")
                or step_cfg.enable_convective_predictor
            )
            if use_convective:
                if _timing_mode_synchronizes_components(args.timing_mode):
                    _record_cuda_synchronization(
                        synchronization_telemetry,
                        scope="component_boundary",
                        backend=exec_backend,
                        device=exec_device,
                    )
                component_started_perf = perf_counter()
                state_conv = apply_tetra_convective_predictor(
                    mesh,
                    state_curr,
                    step_cfg,
                    flow_dt=float(dt_candidate),
                )
                if _timing_mode_synchronizes_components(args.timing_mode):
                    _record_cuda_synchronization(
                        synchronization_telemetry,
                        scope="component_boundary",
                        backend=exec_backend,
                        device=exec_device,
                    )
                component_seconds["convection_total_seconds"] += float(
                    perf_counter() - component_started_perf
                )
                convective_step_diag = dict(
                    state_conv.diagnostics.get(
                        "convective_predictor", convective_step_diag
                    )
                )
                state_after_conv = state_conv
            state_for_projection = state_after_conv
            if flow_mode in {
                "stokes_viscous_projection",
                "navier_stokes_projection_debug",
            }:
                if _timing_mode_synchronizes_components(args.timing_mode):
                    _record_cuda_synchronization(
                        synchronization_telemetry,
                        scope="component_boundary",
                        backend=exec_backend,
                        device=exec_device,
                    )
                component_started_perf = perf_counter()
                state_pred = apply_tetra_stokes_viscous_predictor(
                    mesh,
                    state_after_conv,
                    step_cfg,
                    flow_dt=float(dt_candidate),
                )
                if _timing_mode_synchronizes_components(args.timing_mode):
                    _record_cuda_synchronization(
                        synchronization_telemetry,
                        scope="component_boundary",
                        backend=exec_backend,
                        device=exec_device,
                    )
                component_seconds["viscous_predictor_total_seconds"] += float(
                    perf_counter() - component_started_perf
                )
                viscous_step_diag = dict(
                    state_pred.diagnostics.get("viscous_predictor", viscous_step_diag)
                )
                state_for_projection = state_pred
            if _timing_mode_synchronizes_components(args.timing_mode):
                _record_cuda_synchronization(
                    synchronization_telemetry,
                    scope="component_boundary",
                    backend=exec_backend,
                    device=exec_device,
                )
            component_started_perf = perf_counter()
            state_next = solve_tetra_pressure_projection(
                mesh, state_for_projection, step_cfg
            )
            if _timing_mode_synchronizes_components(args.timing_mode):
                _record_cuda_synchronization(
                    synchronization_telemetry,
                    scope="component_boundary",
                    backend=exec_backend,
                    device=exec_device,
                )
            component_seconds["pressure_projection_total_seconds"] += float(
                perf_counter() - component_started_perf
            )
            state_next.diagnostics["convective_predictor"] = dict(convective_step_diag)
            state_next.diagnostics["viscous_predictor"] = dict(viscous_step_diag)
            d_step = dict(state_next.diagnostics)
            p_step = dict(d_step.get("projection", {}))
            pr_step = dict(d_step.get("pressure", {}))
            be_step = dict(d_step.get("backend_execution", {}))
            v_step = dict(d_step.get("velocity", {}))
            velocity_update_step = dict(d_step.get("projection_velocity_update", {}))
            conv_diag = dict(d_step.get("convective_predictor", {}))
            conv_raw_cfl_step = float(
                conv_diag.get(
                    "convective_cfl_raw_max", conv_diag.get("convective_cfl_max", 0.0)
                )
            )
            if conv_raw_cfl_step >= float(
                convective_cfl_audit_peak.get("convective_cfl_raw_max", -1.0)
            ):
                convective_cfl_audit_peak = {
                    "step": int(step_idx),
                    **conv_diag,
                }
                convective_cfl_audit_peak_step = int(step_idx)
            visc_diag = dict(d_step.get("viscous_predictor", {}))
            nonorth_step = dict(d_step.get("pressure_nonorthogonal_correction", {}))
            acc_step = _projection_acceptance(
                projection=p_step,
                pressure=pr_step,
                backend_execution=be_step,
                thresholds=acceptance_thresholds_cli,
            )
            vel_step = np.asarray(state_next.cell_velocity, dtype=np.float64)
            vel_before_projection = np.asarray(
                state_for_projection.cell_velocity,
                dtype=np.float64,
            )
            cell_volumes_step = np.asarray(mesh.cell_volumes, dtype=np.float64)
            kinetic_before_projection_cell_state = (
                float(
                    np.mean(
                        0.5
                        * np.sum(
                            vel_before_projection * vel_before_projection,
                            axis=1,
                        )
                    )
                )
                if vel_before_projection.size
                else 0.0
            )
            kinetic_after_projection = (
                float(np.mean(0.5 * np.sum(vel_step * vel_step, axis=1)))
                if vel_step.size
                else 0.0
            )
            kinetic_volume_integral_before_projection = (
                float(
                    np.sum(
                        cell_volumes_step
                        * 0.5
                        * np.sum(
                            vel_before_projection * vel_before_projection,
                            axis=1,
                        )
                    )
                )
                if vel_before_projection.size
                else 0.0
            )
            kinetic_volume_integral_after_projection = (
                float(
                    np.sum(
                        cell_volumes_step * 0.5 * np.sum(vel_step * vel_step, axis=1)
                    )
                )
                if vel_step.size
                else 0.0
            )
            pressure_step = np.asarray(state_next.pressure, dtype=np.float64)
            previous_pressure = np.asarray(state_curr.pressure, dtype=np.float64)
            pressure_jump_l2_pa = float(
                np.sqrt(
                    np.sum(
                        cell_volumes_step
                        * (pressure_step - previous_pressure)
                        * (pressure_step - previous_pressure),
                        dtype=np.float64,
                    )
                    / max(float(np.sum(cell_volumes_step)), 1e-300)
                )
            )
            left_pressure, left_area = _area_weighted_boundary_pressure(
                mesh, pressure_step, left_faces
            )
            right_pressure, right_area = _area_weighted_boundary_pressure(
                mesh, pressure_step, right_faces
            )
            outlet_pressure, _ = _area_weighted_boundary_pressure(
                mesh, pressure_step, np.asarray(mesh.outlet_faces, dtype=np.int64)
            )
            combined_inlet_pressure = float(
                (left_pressure * left_area + right_pressure * right_area)
                / (left_area + right_area)
            )
            pressure_drop_pa = combined_inlet_pressure - outlet_pressure
            velocity_volume_weighted_norm = float(
                np.sqrt(
                    np.sum(
                        cell_volumes_step * np.sum(vel_step * vel_step, axis=1),
                        dtype=np.float64,
                    )
                    / max(float(np.sum(cell_volumes_step)), 1e-300)
                )
            )
            wall_shear_step = _wall_shear_stress_metrics(
                mesh,
                vel_step,
                dynamic_viscosity=(
                    float(step_cfg.density) * float(step_cfg.kinematic_viscosity)
                ),
            )
            inlet_flux_left = -float(np.sum(state_next.face_flux[left_faces]))
            inlet_flux_right = -float(np.sum(state_next.face_flux[right_faces]))
            inlet_symmetry_relative = float(
                abs(inlet_flux_left - inlet_flux_right)
                / max(abs(inlet_flux_left) + abs(inlet_flux_right), 1e-300)
            )
            snapshot_due_time = bool(
                next_snapshot_time is not None
                and physical_time
                >= next_snapshot_time - 1e-12 * max(abs(next_snapshot_time), 1.0)
            )
            rec = {
                "step": int(step_idx),
                "time": float(physical_time),
                "physical_time": float(physical_time),
                "flow_dt_mode": str(flow_dt_mode),
                "requested_flow_dt": float(requested_flow_dt),
                "used_flow_dt": float(dt_candidate),
                "pressure_drop_pa": float(pressure_drop_pa),
                "left_inlet_pressure_pa": float(left_pressure),
                "right_inlet_pressure_pa": float(right_pressure),
                "combined_inlet_pressure_pa": float(combined_inlet_pressure),
                "outlet_pressure_pa": float(outlet_pressure),
                "pressure_l2_jump_pa": float(pressure_jump_l2_pa),
                "velocity_volume_weighted_norm": float(velocity_volume_weighted_norm),
                "wall_shear_stress_method": str(wall_shear_step["method"]),
                "wall_shear_stress_area_weighted_mean_pa": float(
                    wall_shear_step["area_weighted_mean_pa"]
                ),
                "wall_shear_stress_max_pa": float(wall_shear_step["max_pa"]),
                "inlet_flux_left": float(inlet_flux_left),
                "inlet_flux_right": float(inlet_flux_right),
                "inlet_symmetry_relative": float(inlet_symmetry_relative),
                "density": float(step_cfg.density),
                "convective_cfl_target": float(convective_cfl_target),
                "run_contract_fingerprint": str(flow_resume_manifest["fingerprint"]),
                "raw_cfl_max_before_dt_selection": float(raw_cfl_before_dt_selection),
                "raw_cfl_max_after_dt_selection": float(raw_cfl_after_dt_selection_max),
                "raw_cfl_p95_after_dt_selection": float(raw_cfl_after_dt_selection_p95),
                "auto_dt_scale_factor": float(auto_dt_scale_factor),
                "auto_dt_floor_used": float(auto_dt_floor_used),
                "auto_dt_min_hit": bool(auto_dt_min_hit),
                "auto_dt_max_hit": bool(auto_dt_max_hit),
                "flow_mode": str(flow_mode),
                "pressure_solver": str(
                    d_step.get("pressure_solver", step_cfg.pressure_solver)
                ),
                "pressure_projection_outlet_contract_mode": str(
                    p_step.get(
                        "pressure_projection_outlet_contract_mode",
                        step_cfg.pressure_projection_outlet_contract_mode,
                    )
                ),
                "projection_cell_velocity_update_mode": str(
                    p_step.get(
                        "projection_cell_velocity_update_mode",
                        step_cfg.projection_cell_velocity_update_mode,
                    )
                ),
                "cached_slip_reconstruction_used": bool(
                    velocity_update_step.get("cached_slip_reconstruction_used", False)
                ),
                "slip_dense_reconstruction_solves_avoided": int(
                    velocity_update_step.get(
                        "slip_dense_reconstruction_solves_avoided", 0
                    )
                ),
                "pressure_nonorthogonal_correction_mode": str(
                    nonorth_step.get(
                        "mode",
                        step_cfg.pressure_nonorthogonal_correction_mode,
                    )
                ),
                "pressure_nonorthogonal_correction_sweeps": int(
                    nonorth_step.get("actual_sweeps", 0)
                ),
                "pressure_nonorthogonal_total_pressure_iterations": int(
                    nonorth_step.get(
                        "total_pressure_iterations",
                        pr_step.get("actual_iterations", 0),
                    )
                ),
                "pressure_nonorthogonal_true_residual_recomputes": int(
                    nonorth_step.get("total_true_residual_recomputes", 0)
                ),
                "pressure_nonorthogonal_true_residual_restarts": int(
                    nonorth_step.get("total_true_residual_restarts", 0)
                ),
                "pressure_recursive_true_residual_mismatch_l2_max": float(
                    nonorth_step.get("recursive_true_residual_mismatch_l2_max", 0.0)
                ),
                "pressure_recursive_true_residual_mismatch_max_abs_max": float(
                    nonorth_step.get(
                        "recursive_true_residual_mismatch_max_abs_max", 0.0
                    )
                ),
                "pressure_nonorthogonal_outer_defect_relative_l2": float(
                    nonorth_step.get("outer_fixed_point_defect_relative_l2", 0.0)
                ),
                "pressure_stopping_reason": str(pr_step.get("stopping_reason", "")),
                "pressure_iterations": int(pr_step.get("actual_iterations", 0)),
                "pressure_matvec_backend": str(
                    pr_step.get("pressure_matvec_backend", "")
                ),
                "pressure_matvec_sparse_csr_used": bool(
                    pr_step.get("pressure_matvec_sparse_csr_used", False)
                ),
                "pressure_matvec_matrix_cached": bool(
                    pr_step.get("pressure_matvec_matrix_cached", False)
                ),
                "pressure_matvec_fallback_reason": str(
                    pr_step.get("pressure_matvec_fallback_reason", "")
                ),
                "pressure_residual_ratio_to_rhs_l2": float(
                    acc_step.get("metrics", {}).get(
                        "residual_ratio_to_rhs_l2",
                        float("inf"),
                    )
                ),
                "pressure_residual_ratio_to_rhs_max": float(
                    acc_step.get("metrics", {}).get(
                        "residual_ratio_to_rhs_max",
                        float("inf"),
                    )
                ),
                "pressure_rhs_norm_l2": float(pr_step.get("rhs_norm_l2", 0.0)),
                "pressure_residual_norm_l2": float(
                    pr_step.get("residual_norm_l2", 0.0)
                ),
                "pressure_linear_accepted": bool(
                    acc_step.get("pressure_linear_accepted", False)
                ),
                "pressure_linear_solved_strict": bool(
                    acc_step.get("pressure_linear_solved_strict", False)
                ),
                "projection_accepted": bool(acc_step.get("projection_accepted", False)),
                "projection_solved": bool(acc_step.get("projection_solved", False)),
                "projection_acceptance_reason": str(acc_step.get("reason", "")),
                "projection_failed_criteria": _projection_failed_criteria(acc_step),
                "strict_diagnostic_criteria_not_met": (
                    _projection_strict_diagnostic_criteria_not_met(acc_step)
                ),
                "projection_blocking_checks": dict(acc_step.get("blocking_checks", {})),
                "projection_acceptance_checklist": dict(acc_step.get("checklist", {})),
                "initial_divergence_max_abs": float(
                    p_step.get("initial_divergence_max_abs", 0.0)
                ),
                "final_divergence_max_abs": float(
                    p_step.get("final_divergence_max_abs", 0.0)
                ),
                "initial_divergence_l2": float(
                    p_step.get("initial_divergence_l2", 0.0)
                ),
                "final_divergence_l2": float(p_step.get("final_divergence_l2", 0.0)),
                "divergence_reduction_ratio_linf": float(
                    p_step.get("divergence_reduction_ratio", 0.0)
                ),
                "divergence_reduction_ratio_l2": float(
                    p_step.get("divergence_reduction_ratio_l2", 0.0)
                ),
                "inlet_flux_before": float(p_step.get("inlet_flux_total_before", 0.0)),
                "inlet_flux_after": float(p_step.get("inlet_flux_total_after", 0.0)),
                "outlet_flux_before": float(
                    p_step.get("outlet_flux_total_before", 0.0)
                ),
                "outlet_flux_after": float(p_step.get("outlet_flux_total_after", 0.0)),
                "outlet_inlet_flux_ratio": float(
                    float(p_step.get("outlet_flux_total_after", 0.0))
                    / max(abs(float(p_step.get("inlet_flux_total_after", 0.0))), 1e-30)
                ),
                "net_boundary_flux_after": float(
                    p_step.get("net_boundary_flux_after", 0.0)
                ),
                "net_boundary_flux_relative": float(
                    abs(float(p_step.get("net_boundary_flux_after", 0.0)))
                    / max(abs(float(p_step.get("inlet_flux_total_after", 0.0))), 1e-30)
                ),
                "wall_flux_max_abs_after": float(
                    p_step.get("wall_flux_max_abs_after", 0.0)
                ),
                "outlet_flux_rescale_used": bool(
                    p_step.get("outlet_flux_rescale_used", False)
                ),
                "nonphysical_flux_fix_used": bool(
                    p_step.get("nonphysical_flux_fix_used", False)
                ),
                "velocity_magnitude_min": float(v_step.get("magnitude_min", 0.0)),
                "velocity_magnitude_max": float(v_step.get("magnitude_max", 0.0)),
                "velocity_magnitude_mean": float(v_step.get("magnitude_mean", 0.0)),
                "convective_predictor_used": bool(
                    conv_diag.get("convective_predictor_used", False)
                ),
                "convective_execution_backend": str(
                    conv_diag.get("convective_execution_backend", "numpy")
                ),
                "convective_execution_device": str(
                    conv_diag.get("convective_execution_device", "cpu")
                ),
                "convective_torch_cuda_used": bool(
                    conv_diag.get("convective_torch_cuda_used", False)
                ),
                "convective_cuda_handoff_available": bool(
                    conv_diag.get("convective_cuda_handoff_available", False)
                ),
                "convective_numpy_fallback_reason": str(
                    conv_diag.get("convective_numpy_fallback_reason", "")
                ),
                "convective_stabilization_mode": str(
                    conv_diag.get(
                        "convective_stabilization_mode",
                        step_cfg.convective_stabilization_mode,
                    )
                ),
                "convective_substep_boundary_contract_mode": str(
                    conv_diag.get(
                        "convective_substep_boundary_contract_mode",
                        step_cfg.convective_substep_boundary_contract,
                    )
                ),
                "convective_predictor_outlet_contract_mode": str(
                    conv_diag.get(
                        "convective_predictor_outlet_contract_mode",
                        step_cfg.viscous_predictor_outlet_contract_mode,
                    )
                ),
                "convective_delta_velocity_max": float(
                    conv_diag.get("convective_delta_velocity_max", 0.0)
                ),
                "convective_delta_velocity_l2": float(
                    conv_diag.get("convective_delta_velocity_l2", 0.0)
                ),
                "convective_cfl_limit": float(
                    conv_diag.get("convective_cfl_limit", step_cfg.convective_cfl_limit)
                ),
                "convective_cfl_raw_max": float(
                    conv_diag.get(
                        "convective_cfl_raw_max",
                        conv_diag.get("convective_cfl_max", 0.0),
                    )
                ),
                "convective_cfl_raw_p95": float(
                    conv_diag.get(
                        "convective_cfl_raw_p95",
                        conv_diag.get("convective_cfl_p95", 0.0),
                    )
                ),
                "convective_cfl_effective_max": float(
                    conv_diag.get(
                        "convective_cfl_effective_max",
                        conv_diag.get("convective_cfl_max", 0.0),
                    )
                ),
                "convective_cfl_effective_p95": float(
                    conv_diag.get(
                        "convective_cfl_effective_p95",
                        conv_diag.get("convective_cfl_p95", 0.0),
                    )
                ),
                "effective_cfl_limit_excess": float(
                    conv_diag.get(
                        "convective_cfl_effective_max",
                        conv_diag.get("convective_cfl_max", 0.0),
                    )
                    - conv_diag.get(
                        "convective_cfl_limit", step_cfg.convective_cfl_limit
                    )
                ),
                "convective_cfl_warning_raw": bool(
                    conv_diag.get(
                        "convective_cfl_warning_raw",
                        conv_diag.get("convective_cfl_warning", False),
                    )
                ),
                "convective_cfl_warning_effective": bool(
                    conv_diag.get(
                        "convective_cfl_warning_effective",
                        conv_diag.get("convective_cfl_warning", False),
                    )
                ),
                "convective_cfl_max": float(
                    conv_diag.get(
                        "convective_cfl_raw_max",
                        conv_diag.get("convective_cfl_max", 0.0),
                    )
                ),
                "convective_cfl_p95": float(
                    conv_diag.get(
                        "convective_cfl_raw_p95",
                        conv_diag.get("convective_cfl_p95", 0.0),
                    )
                ),
                "convective_cfl_warning": bool(
                    conv_diag.get(
                        "convective_cfl_warning_raw",
                        conv_diag.get("convective_cfl_warning", False),
                    )
                ),
                "effective_cfl_warning_with_eps": bool(
                    float(
                        conv_diag.get(
                            "convective_cfl_effective_max",
                            conv_diag.get("convective_cfl_max", 0.0),
                        )
                    )
                    > float(
                        conv_diag.get(
                            "convective_cfl_limit", step_cfg.convective_cfl_limit
                        )
                    )
                    * (1.0 + float(args.convective_cfl_acceptance_eps))
                ),
                "raw_cfl_after_dt_selection_warning_with_eps": bool(
                    float(raw_cfl_after_dt_selection_max)
                    > float(step_cfg.convective_cfl_limit)
                    * (1.0 + float(args.convective_cfl_acceptance_eps))
                ),
                "convective_predictor_damping_requested": float(
                    conv_diag.get(
                        "convective_predictor_damping_requested",
                        conv_diag.get(
                            "convective_predictor_damping",
                            step_cfg.convective_predictor_damping,
                        ),
                    )
                ),
                "convective_predictor_damping": float(
                    conv_diag.get(
                        "convective_predictor_damping",
                        step_cfg.convective_predictor_damping,
                    )
                ),
                "convective_predictor_damping_effective": float(
                    conv_diag.get("convective_predictor_damping_effective", 0.0)
                ),
                "convective_predictor_cfl_scale": float(
                    conv_diag.get("convective_predictor_cfl_scale", 1.0)
                ),
                "convective_auto_damping_used": bool(
                    conv_diag.get("convective_auto_damping_used", False)
                ),
                "convective_auto_damping_reason": str(
                    conv_diag.get("convective_auto_damping_reason", "")
                ),
                "convective_substepping_used": bool(
                    conv_diag.get("convective_substepping_used", False)
                ),
                "convective_substep_count": int(
                    conv_diag.get("convective_substep_count", 1)
                ),
                "convective_substep_count_unclamped": int(
                    conv_diag.get("convective_substep_count_unclamped", 1)
                ),
                "convective_substep_cap_hit": bool(
                    conv_diag.get("convective_substep_cap_hit", False)
                ),
                "convective_substep_dt": float(
                    conv_diag.get("convective_substep_dt", dt_candidate)
                ),
                "convective_cfl_per_substep_max": float(
                    conv_diag.get("convective_cfl_per_substep_max", 0.0)
                ),
                "convective_cfl_per_substep_p95": float(
                    conv_diag.get("convective_cfl_per_substep_p95", 0.0)
                ),
                "convective_substepping_runtime_seconds": float(
                    conv_diag.get("convective_substepping_runtime_seconds", 0.0)
                ),
                "convective_dt_effective": float(
                    conv_diag.get("convective_dt_effective", 0.0)
                ),
                "kinetic_energy_before_convection": float(
                    conv_diag.get("kinetic_energy_before_convection", 0.0)
                ),
                "kinetic_energy_after_convection": float(
                    conv_diag.get("kinetic_energy_after_convection", 0.0)
                ),
                "kinetic_energy_volume_integral_before_convection_m5_s2": float(
                    conv_diag.get(
                        "kinetic_energy_volume_integral_before_convection_m5_s2",
                        0.0,
                    )
                ),
                "kinetic_energy_volume_integral_after_convection_m5_s2": float(
                    conv_diag.get(
                        "kinetic_energy_volume_integral_after_convection_m5_s2",
                        0.0,
                    )
                ),
                "divergence_before_convection_max": float(
                    conv_diag.get("divergence_before_convection_max", 0.0)
                ),
                "divergence_before_convection_l2": float(
                    conv_diag.get("divergence_before_convection_l2", 0.0)
                ),
                "divergence_after_convection_before_projection_max": float(
                    conv_diag.get(
                        "divergence_after_convection_before_projection_max", 0.0
                    )
                ),
                "divergence_after_convection_before_projection_l2": float(
                    conv_diag.get(
                        "divergence_after_convection_before_projection_l2", 0.0
                    )
                ),
                "viscous_predictor_used": bool(
                    visc_diag.get("viscous_predictor_used", False)
                ),
                "viscous_predictor_mode": str(
                    visc_diag.get("viscous_predictor_mode", "none")
                ),
                "viscous_execution_backend": str(
                    visc_diag.get("viscous_execution_backend", "numpy")
                ),
                "viscous_execution_device": str(
                    visc_diag.get("viscous_execution_device", "cpu")
                ),
                "viscous_torch_cuda_used": bool(
                    visc_diag.get("viscous_torch_cuda_used", False)
                ),
                "viscous_cuda_no_slip_used": bool(
                    visc_diag.get("viscous_cuda_no_slip_used", False)
                ),
                "viscous_cuda_input_reused": bool(
                    visc_diag.get("viscous_cuda_input_reused", False)
                ),
                "viscous_cuda_finalization_used": bool(
                    visc_diag.get("viscous_cuda_finalization_used", False)
                ),
                "viscous_cuda_host_to_device_bytes_avoided": int(
                    visc_diag.get("viscous_cuda_host_to_device_bytes_avoided", 0)
                ),
                "viscous_cuda_cpu_reconstruction_solves_avoided": int(
                    visc_diag.get("viscous_cuda_cpu_reconstruction_solves_avoided", 0)
                ),
                "viscous_cuda_residency_scope": str(
                    visc_diag.get("viscous_cuda_residency_scope", "none")
                ),
                "viscous_numpy_fallback_reason": str(
                    visc_diag.get("viscous_numpy_fallback_reason", "")
                ),
                "viscous_predictor_outlet_contract_mode": str(
                    visc_diag.get(
                        "viscous_predictor_outlet_contract_mode",
                        step_cfg.viscous_predictor_outlet_contract_mode,
                    )
                ),
                "kinematic_viscosity": float(step_cfg.kinematic_viscosity),
                "viscous_dt": float(dt_candidate),
                "viscous_stability_metric": float(
                    visc_diag.get("viscous_stability_metric", 0.0)
                ),
                "viscous_stability_warning": bool(
                    visc_diag.get("viscous_stability_warning", False)
                ),
                "viscous_substeps": int(visc_diag.get("viscous_substeps", 1)),
                "viscous_substep_dt": float(
                    visc_diag.get("viscous_substep_dt", dt_candidate)
                ),
                "viscous_face_flux_divergence_impact_cap": float(
                    visc_diag.get(
                        "viscous_face_flux_divergence_impact_cap",
                        args.viscous_face_flux_divergence_impact_cap,
                    )
                ),
                "wall_velocity_boundary_mode": str(
                    visc_diag.get(
                        "wall_velocity_boundary_mode",
                        step_cfg.wall_velocity_boundary_mode,
                    )
                ),
                "wall_velocity_boundary_implementation": str(
                    visc_diag.get("wall_velocity_boundary_implementation", "")
                ),
                "wall_tangential_no_slip_strength": float(
                    visc_diag.get(
                        "wall_tangential_no_slip_strength",
                        step_cfg.wall_tangential_no_slip_strength,
                    )
                ),
                "wall_tangential_no_slip_strength_ramp_enabled": bool(
                    wall_strength_ramp_steps > 0
                ),
                "wall_tangential_no_slip_strength_ramp_start": float(
                    wall_strength_ramp_start
                ),
                "wall_tangential_no_slip_strength_ramp_target": float(
                    wall_strength_ramp_target
                ),
                "wall_tangential_no_slip_strength_ramp_steps": int(
                    wall_strength_ramp_steps
                ),
                "wall_tangential_shear_face_flux_requested": bool(
                    visc_diag.get(
                        "wall_tangential_shear_face_flux_requested",
                        step_cfg.wall_tangential_shear_face_flux_enabled,
                    )
                ),
                "wall_tangential_cell_velocity_momentum_enabled": bool(
                    visc_diag.get(
                        "wall_tangential_cell_velocity_momentum_enabled",
                        step_cfg.wall_tangential_cell_velocity_momentum_enabled,
                    )
                ),
                "wall_tangential_operator_active_cells": int(
                    visc_diag.get("wall_tangential_operator_active_cells", 0)
                ),
                "wall_tangential_operator_max_abs": float(
                    visc_diag.get("wall_tangential_operator_max_abs", 0.0)
                ),
                "wall_tangential_operator_trace_mean": float(
                    visc_diag.get("wall_tangential_operator_trace_mean", 0.0)
                ),
                "wall_tangential_operator_trace_max": float(
                    visc_diag.get("wall_tangential_operator_trace_max", 0.0)
                ),
                "wall_tangential_operator_effective_nu_dt_max_abs": float(
                    visc_diag.get(
                        "wall_tangential_operator_effective_nu_dt_max_abs", 0.0
                    )
                ),
                "wall_tangential_operator_effective_nu_subdt_max_abs": float(
                    visc_diag.get(
                        "wall_tangential_operator_effective_nu_subdt_max_abs", 0.0
                    )
                ),
                "wall_flux_stokes_resistance_enabled": bool(
                    visc_diag.get("wall_flux_stokes_resistance_enabled", False)
                ),
                "wall_flux_stokes_resistance_strength": float(
                    visc_diag.get("wall_flux_stokes_resistance_strength", 0.0)
                ),
                "wall_flux_stokes_resistance_active_faces": int(
                    visc_diag.get("wall_flux_stokes_resistance_active_faces", 0)
                ),
                "wall_flux_stokes_resistance_solver_iterations": int(
                    visc_diag.get("wall_flux_stokes_resistance_solver_iterations", 0)
                ),
                "wall_flux_stokes_resistance_solver_converged": bool(
                    visc_diag.get("wall_flux_stokes_resistance_solver_converged", True)
                ),
                "wall_flux_stokes_resistance_solver_residual_l2": float(
                    visc_diag.get("wall_flux_stokes_resistance_solver_residual_l2", 0.0)
                ),
                "wall_flux_stokes_resistance_solver_method": str(
                    visc_diag.get("wall_flux_stokes_resistance_solver_method", "")
                ),
                "wall_tangential_shear_face_flux_enabled": bool(
                    visc_diag.get("wall_tangential_shear_face_flux_enabled", False)
                ),
                "wall_tangential_shear_face_flux_applications": int(
                    visc_diag.get("wall_tangential_shear_face_flux_applications", 0)
                ),
                "wall_tangential_shear_face_flux_active_cells": int(
                    visc_diag.get("wall_tangential_shear_face_flux_active_cells", 0)
                ),
                "wall_tangential_shear_face_flux_delta_l2": float(
                    visc_diag.get("wall_tangential_shear_face_flux_delta_l2", 0.0)
                ),
                "wall_tangential_shear_face_flux_wall_speed_mean_before": float(
                    visc_diag.get(
                        "wall_tangential_shear_face_flux_wall_speed_mean_before", 0.0
                    )
                ),
                "wall_tangential_shear_face_flux_wall_speed_mean_after": float(
                    visc_diag.get(
                        "wall_tangential_shear_face_flux_wall_speed_mean_after", 0.0
                    )
                ),
                "local_conservative_flux_correction_enabled": bool(
                    visc_diag.get("local_conservative_flux_correction_enabled", False)
                ),
                "local_conservative_flux_correction_iterations": int(
                    visc_diag.get("local_conservative_flux_correction_iterations", 0)
                ),
                "local_conservative_flux_correction_residual_l2_before": float(
                    visc_diag.get(
                        "local_conservative_flux_correction_residual_l2_before", 0.0
                    )
                ),
                "local_conservative_flux_correction_residual_l2_after": float(
                    visc_diag.get(
                        "local_conservative_flux_correction_residual_l2_after", 0.0
                    )
                ),
                "local_conservative_flux_correction_residual_max_abs_before": float(
                    visc_diag.get(
                        "local_conservative_flux_correction_residual_max_abs_before",
                        0.0,
                    )
                ),
                "local_conservative_flux_correction_residual_max_abs_after": float(
                    visc_diag.get(
                        "local_conservative_flux_correction_residual_max_abs_after",
                        0.0,
                    )
                ),
                "local_conservative_flux_correction_delta_l2": float(
                    visc_diag.get("local_conservative_flux_correction_delta_l2", 0.0)
                ),
                "local_conservative_flux_correction_delta_max_abs": float(
                    visc_diag.get(
                        "local_conservative_flux_correction_delta_max_abs", 0.0
                    )
                ),
                "viscous_delta_velocity_max": float(
                    visc_diag.get("viscous_delta_velocity_max", 0.0)
                ),
                "viscous_delta_velocity_l2": float(
                    visc_diag.get("viscous_delta_velocity_l2", 0.0)
                ),
                "viscous_nonorthogonal_operator_power": float(
                    visc_diag.get("viscous_nonorthogonal_operator_power", 0.0)
                ),
                "capped_predictor_updates_count": int(
                    visc_diag.get("capped_predictor_updates_count", 0)
                ),
                "total_predictor_updates_count": int(
                    visc_diag.get("total_predictor_updates_count", 0)
                ),
                "capped_predictor_updates_fraction": float(
                    visc_diag.get("capped_predictor_updates_fraction", 0.0)
                ),
                "capped_predictor_faces_count": int(
                    visc_diag.get("capped_predictor_faces_count", 0)
                ),
                "face_flux_delta_predictor_max": float(
                    visc_diag.get("face_flux_delta_predictor_max", 0.0)
                ),
                "face_flux_delta_predictor_l2": float(
                    visc_diag.get("face_flux_delta_predictor_l2", 0.0)
                ),
                "face_flux_delta_contract_max": float(
                    visc_diag.get("face_flux_delta_contract_max", 0.0)
                ),
                "face_flux_delta_contract_l2": float(
                    visc_diag.get("face_flux_delta_contract_l2", 0.0)
                ),
                "divergence_before_predictor_max": float(
                    visc_diag.get("divergence_before_predictor_max", 0.0)
                ),
                "divergence_before_predictor_l2": float(
                    visc_diag.get("divergence_before_predictor_l2", 0.0)
                ),
                "divergence_after_predictor_before_boundary_contract_max": float(
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_max", 0.0
                    )
                ),
                "divergence_after_predictor_before_boundary_contract_l2": float(
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_l2", 0.0
                    )
                ),
                "divergence_after_boundary_contract_before_projection_max": float(
                    visc_diag.get(
                        "divergence_after_boundary_contract_before_projection_max", 0.0
                    )
                ),
                "divergence_after_boundary_contract_before_projection_l2": float(
                    visc_diag.get(
                        "divergence_after_boundary_contract_before_projection_l2", 0.0
                    )
                ),
                "divergence_predictor_over_before_l2_ratio": _ratio(
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_l2", 0.0
                    ),
                    visc_diag.get("divergence_before_predictor_l2", 0.0),
                ),
                "divergence_contract_over_predictor_l2_ratio": _ratio(
                    visc_diag.get(
                        "divergence_after_boundary_contract_before_projection_l2", 0.0
                    ),
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_l2", 0.0
                    ),
                ),
                "divergence_predictor_over_before_max_ratio": _ratio(
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_max", 0.0
                    ),
                    visc_diag.get("divergence_before_predictor_max", 0.0),
                ),
                "divergence_contract_over_predictor_max_ratio": _ratio(
                    visc_diag.get(
                        "divergence_after_boundary_contract_before_projection_max", 0.0
                    ),
                    visc_diag.get(
                        "divergence_after_predictor_before_boundary_contract_max", 0.0
                    ),
                ),
                "divergence_after_projection_max": float(
                    p_step.get("final_divergence_max_abs", 0.0)
                ),
                "divergence_after_projection_l2": float(
                    p_step.get("final_divergence_l2", 0.0)
                ),
                "net_boundary_flux_before_predictor": float(
                    visc_diag.get("net_boundary_flux_before_predictor", 0.0)
                ),
                "net_boundary_flux_after_predictor_before_contract": float(
                    visc_diag.get(
                        "net_boundary_flux_after_predictor_before_contract", 0.0
                    )
                ),
                "net_boundary_flux_after_contract": float(
                    visc_diag.get("net_boundary_flux_after_contract", 0.0)
                ),
                "net_boundary_flux_after_projection": float(
                    p_step.get("net_boundary_flux_after", 0.0)
                ),
                "wall_flux_max_after_predictor_before_contract": float(
                    visc_diag.get("wall_flux_max_after_predictor_before_contract", 0.0)
                ),
                "wall_flux_max_after_contract": float(
                    visc_diag.get("wall_flux_max_after_contract", 0.0)
                ),
                "wall_flux_max_after_projection": float(
                    p_step.get("wall_flux_max_abs_after", 0.0)
                ),
                "outlet_inlet_ratio_before_predictor": float(
                    visc_diag.get("outlet_inlet_ratio_before_predictor", 0.0)
                ),
                "outlet_inlet_ratio_after_predictor_before_contract": float(
                    visc_diag.get(
                        "outlet_inlet_ratio_after_predictor_before_contract", 0.0
                    )
                ),
                "outlet_inlet_ratio_after_contract": float(
                    visc_diag.get("outlet_inlet_ratio_after_contract", 0.0)
                ),
                "kinetic_energy_before_predictor": float(
                    visc_diag.get("kinetic_energy_before_predictor", 0.0)
                ),
                "kinetic_energy_after_predictor": float(
                    visc_diag.get("kinetic_energy_after_predictor", 0.0)
                ),
                "kinetic_energy_after_contract": float(
                    visc_diag.get("kinetic_energy_after_contract", 0.0)
                ),
                "kinetic_energy_cell_velocity_state_before_predictor": float(
                    visc_diag.get(
                        "kinetic_energy_cell_velocity_state_before_predictor",
                        0.0,
                    )
                ),
                "kinetic_energy_cell_velocity_state_after_predictor": float(
                    visc_diag.get(
                        "kinetic_energy_cell_velocity_state_after_predictor",
                        0.0,
                    )
                ),
                "kinetic_energy_volume_integral_before_predictor_m5_s2": float(
                    visc_diag.get(
                        "kinetic_energy_volume_integral_before_predictor_m5_s2",
                        0.0,
                    )
                ),
                "kinetic_energy_volume_integral_after_predictor_m5_s2": float(
                    visc_diag.get(
                        "kinetic_energy_volume_integral_after_predictor_m5_s2",
                        0.0,
                    )
                ),
                "kinetic_energy_volume_integral_after_contract_reconstruction_m5_s2": float(
                    visc_diag.get(
                        "kinetic_energy_volume_integral_after_contract_reconstruction_m5_s2",
                        0.0,
                    )
                ),
                "kinetic_energy_before_projection_cell_velocity_state": float(
                    kinetic_before_projection_cell_state
                ),
                "kinetic_energy_after_projection": float(kinetic_after_projection),
                "kinetic_energy_volume_integral_before_projection_m5_s2": float(
                    kinetic_volume_integral_before_projection
                ),
                "kinetic_energy_volume_integral_after_projection_m5_s2": float(
                    kinetic_volume_integral_after_projection
                ),
                "pressure_min": float(pr_step.get("min", 0.0)),
                "pressure_max": float(pr_step.get("max", 0.0)),
                "pressure_mean": float(pr_step.get("mean", 0.0)),
                "pressure_l2": float(pr_step.get("l2", 0.0)),
                "finite_fields": bool(
                    acc_step.get("checklist", {}).get("finite_fields", False)
                ),
                "snapshot_due_physical_time": snapshot_due_time,
                "used_numpy_fallback": bool(be_step.get("used_numpy_fallback", True)),
                "all_core_arrays_on_cuda": bool(
                    be_step.get("all_core_arrays_on_cuda", False)
                ),
            }
            progression_history.append(rec)
            solver_step_durations.append(float(perf_counter() - step_started_perf))
            if step_idx in snapshot_steps or snapshot_due_time:
                snap_dir = run_dir / "snapshots" / f"step_{step_idx:04d}"
                snap_dir.mkdir(parents=True, exist_ok=True)
                (snap_dir / "snapshot_metadata.json").write_text(
                    json.dumps(
                        {
                            "step": step_idx,
                            "physical_time": physical_time,
                            "target_physical_time": (
                                next_snapshot_time if snapshot_due_time else None
                            ),
                            "requested_snapshot_time": (
                                next_snapshot_time if snapshot_due_time else None
                            ),
                            "actual_snapshot_time": physical_time,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                div_snap = np.asarray(
                    compute_tetra_flux_divergence(
                        mesh,
                        state_next.face_flux,
                        left_inlet_faces=left_faces,
                        right_inlet_faces=right_faces,
                        outlet_faces=mesh.outlet_faces,
                        wall_faces=mesh.wall_faces,
                    )["divergence"],
                    dtype=np.float64,
                )
                _save_scatter(
                    centers=mesh.cell_centers,
                    values=state_next.pressure,
                    title=f"Pressure XY step {step_idx}",
                    label="p",
                    out_path=snap_dir / "pressure_xy.png",
                    cmap="plasma",
                )
                _save_scatter(
                    centers=mesh.cell_centers,
                    values=np.linalg.norm(state_next.cell_velocity, axis=1),
                    title=f"Velocity Magnitude XY step {step_idx}",
                    label="|u|",
                    out_path=snap_dir / "velocity_magnitude_xy.png",
                    cmap="magma",
                )
                _save_scatter(
                    centers=mesh.cell_centers,
                    values=div_snap,
                    title=f"log10(|div|) XY step {step_idx}",
                    label="log10|div|",
                    out_path=snap_dir / "divergence_xy.png",
                    cmap="viridis",
                    log10_abs=True,
                )
                _save_vectors_sparse_normalized(
                    centers=mesh.cell_centers,
                    velocity=state_next.cell_velocity,
                    out_path=snap_dir / "velocity_vectors_xy_sparse_normalized.png",
                    max_arrows=400,
                )
                np.save(
                    snap_dir / "corrected_face_flux.npy",
                    np.asarray(state_next.face_flux, dtype=np.float64),
                )
                np.save(
                    snap_dir / "cell_velocity.npy",
                    np.asarray(state_next.cell_velocity, dtype=np.float64),
                )
                np.save(
                    snap_dir / "pressure.npy",
                    np.asarray(state_next.pressure, dtype=np.float64),
                )
                np.save(snap_dir / "divergence.npy", div_snap)
                _export_vtu(
                    points=mesh.points,
                    tetrahedra=mesh.tetrahedra,
                    pressure=np.asarray(state_next.pressure, dtype=np.float64),
                    divergence=div_snap,
                    velocity=np.asarray(state_next.cell_velocity, dtype=np.float64),
                    out_path=snap_dir / f"{mesh_stem}_step_{step_idx:04d}.vtu",
                )
            if snapshot_due_time and next_snapshot_time is not None:
                while next_snapshot_time <= physical_time + 1e-15:
                    next_snapshot_time += float(snapshot_time_interval)
            flow_loop_step_durations.append(float(perf_counter() - step_started_perf))
            state_curr = state_next

        _record_cuda_synchronization(
            synchronization_telemetry,
            scope="flow_boundary",
            backend=exec_backend,
            device=exec_device,
        )
        flow_stepping_seconds = float(perf_counter() - flow_stepping_started_perf)
        postprocessing_started_perf = perf_counter()
        state1 = state_curr
        diag = dict(state1.diagnostics)
        backend_exec = diag.get("backend_execution", {})
        flow_progression_enabled = bool(flow_steps_requested > 1)
        flow_progression_solved = False
        flow_progression_reason = ""
        flow_progression_final_metrics: dict[str, Any] = {}
        flow_progression_worst_step_metrics: dict[str, Any] = {}
        flow_prog_acc: dict[str, Any] = {}
        flow_mode_comparison: dict[str, Any] = {}
        viscous_predictor_mode_comparison: dict[str, Any] = {}
        viscous_progression_accepted = False
        viscous_progression_acceptance_reason = "not evaluated"
        viscous_progression_checks: dict[str, Any] = {}
        viscous_predictor_damages_divergence = False
        viscous_predictor_damage_ratio_l2 = 1.0
        viscous_predictor_damage_ratio_linf = 1.0
        viscous_predictor_best_mode = str(resolved_viscous_predictor_mode)
        stokes_ready_for_advection_term = False
        convective_prototype_accepted = False
        convective_prototype_acceptance_reason = "not evaluated"
        convective_prototype_checks: dict[str, Any] = {}
        convective_readiness_checks: dict[str, Any] = {}
        convective_readiness_reason = "not evaluated"
        ready_for_long_ns_run_debug = False
        ready_for_long_ns_run_physical = False
        ready_for_long_ns_run = False
        ns_auto_dt_accepted = False
        ns_auto_dt_warning = ""
        navier_stokes_prototype_audit: dict[str, Any] = {}
        convective_sensitivity_sweep: dict[str, Any] = {}
        convective_stabilization_comparison: dict[str, Any] = {}
        ns_dt_mode_comparison: dict[str, Any] = {}
        auto_cfl_no_damping_readiness_reason = ""
        convective_cfl_audit_written = False
        recommended_convective_debug_config: dict[str, Any] = {}
        stokes_baseline_sensitivity_sweep: dict[str, Any] = {}
        stokes_baseline_accepted = False
        stokes_baseline_acceptance_reason = "not evaluated"
        recommended_stokes_baseline_config: dict[str, Any] = {}
        ready_for_convective_prototype = False
        used_dt_min = (
            float(np.min(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        used_dt_mean = (
            float(np.mean(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        used_dt_max = (
            float(np.max(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        auto_dt_scale_min = (
            float(np.min(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        auto_dt_scale_mean = (
            float(np.mean(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        auto_dt_scale_max = (
            float(np.max(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        auto_dt_floor_min = (
            float(np.min(np.asarray(auto_dt_floor_values, dtype=np.float64)))
            if auto_dt_floor_values
            else 0.0
        )
        auto_dt_floor_mean = (
            float(np.mean(np.asarray(auto_dt_floor_values, dtype=np.float64)))
            if auto_dt_floor_values
            else 0.0
        )
        auto_dt_floor_max = (
            float(np.max(np.asarray(auto_dt_floor_values, dtype=np.float64)))
            if auto_dt_floor_values
            else 0.0
        )
        raw_cfl_after_dt_selection_max = 0.0
        raw_cfl_after_dt_selection_p95 = 0.0

        if bool(args.fail_if_numpy_fallback):
            _assert_no_numpy_fallback(
                backend_execution=backend_exec,
                step_history=[*startup_bootstrap_history, *progression_history],
            )

        def _projection_summary(
            label: str,
            cfg_used: TetraFlowConfig,
            state_proj,
        ) -> dict[str, Any]:
            cfg_used = _resolve_flow_config_for_postsolve_audit(cfg_used)
            d = dict(state_proj.diagnostics)
            p = dict(d.get("projection", {}))
            pr = dict(d.get("pressure", {}))
            be = dict(d.get("backend_execution", {}))
            lim = dict(d.get("correction_limiter", {}))
            lim_cons = dict(d.get("correction_limiter_conservation_audit", {}))
            nonorth = dict(d.get("pressure_nonorthogonal_correction", {}))
            acc = _projection_acceptance(
                projection=p,
                pressure=pr,
                backend_execution=be,
                thresholds=acceptance_thresholds_cli,
            )
            inlet_flux_after = float(p.get("inlet_flux_total_after", 0.0))
            outlet_flux_after = float(p.get("outlet_flux_total_after", 0.0))
            net_boundary_flux_after = float(p.get("net_boundary_flux_after", 0.0))
            return {
                "label": label,
                "pressure_solver": str(cfg_used.pressure_solver),
                "projection_rhs_mode": str(cfg_used.projection_rhs_mode),
                "projection_correction_damping": float(
                    cfg_used.projection_correction_damping
                ),
                "projection_correction_limit_mode": str(
                    cfg_used.projection_correction_limit_mode
                ),
                "projection_cell_velocity_update_mode": str(
                    cfg_used.projection_cell_velocity_update_mode
                ),
                "max_pressure_iterations": int(cfg_used.max_pressure_iterations),
                "pressure_tolerance": float(cfg_used.pressure_tolerance),
                "pressure_relative_tolerance": float(
                    cfg_used.pressure_relative_tolerance
                ),
                "cg_breakdown_eps": float(cfg_used.cg_breakdown_eps),
                "cg_stagnation_window": int(cfg_used.cg_stagnation_window),
                "cg_stagnation_ratio": float(cfg_used.cg_stagnation_ratio),
                "pcg_require_relative_l2_convergence": bool(
                    cfg_used.pcg_require_relative_l2_convergence
                ),
                "outlet_projection_mode": str(cfg_used.outlet_projection_mode),
                "pressure_projection_outlet_contract_mode": str(
                    cfg_used.pressure_projection_outlet_contract_mode
                ),
                "pressure_nonorthogonal_correction_mode": str(
                    cfg_used.pressure_nonorthogonal_correction_mode
                ),
                "pressure_nonorthogonal_correction_sweeps": int(
                    cfg_used.pressure_nonorthogonal_correction_sweeps
                ),
                "pressure_nonorthogonal_correction_relaxation": float(
                    cfg_used.pressure_nonorthogonal_correction_relaxation
                ),
                "pressure_nonorthogonal_actual_sweeps": int(
                    nonorth.get("actual_sweeps", 0)
                ),
                "pressure_nonorthogonal_total_pressure_iterations": int(
                    nonorth.get(
                        "total_pressure_iterations",
                        pr.get("actual_iterations", 0),
                    )
                ),
                "pressure_nonorthogonal_true_residual_recomputes": int(
                    nonorth.get("total_true_residual_recomputes", 0)
                ),
                "pressure_nonorthogonal_true_residual_restarts": int(
                    nonorth.get("total_true_residual_restarts", 0)
                ),
                "pressure_recursive_true_residual_mismatch_l2_max": float(
                    nonorth.get("recursive_true_residual_mismatch_l2_max", 0.0)
                ),
                "pressure_recursive_true_residual_mismatch_max_abs_max": float(
                    nonorth.get("recursive_true_residual_mismatch_max_abs_max", 0.0)
                ),
                "pressure_nonorthogonal_outer_defect_relative_l2": float(
                    nonorth.get("outer_fixed_point_defect_relative_l2", 0.0)
                ),
                "residual_final": float(pr.get("poisson_residual_final", 0.0)),
                "residual_ratio_to_rhs_max": float(
                    pr.get("residual_ratio_to_rhs_max", 0.0)
                ),
                "residual_ratio_to_rhs_l2": float(
                    pr.get("residual_ratio_to_rhs_l2", 0.0)
                ),
                "actual_iterations": int(pr.get("actual_iterations", 0)),
                "stopping_reason": str(pr.get("stopping_reason", "")),
                "pressure_solved": bool(pr.get("pressure_solved", False)),
                "pressure_linear_solved_strict": bool(
                    acc.get("pressure_linear_solved_strict", False)
                ),
                "pressure_linear_accepted": bool(
                    acc.get("pressure_linear_accepted", False)
                ),
                "final_div_max_abs": float(p.get("final_divergence_max_abs", 0.0)),
                "final_div_l2": float(p.get("final_divergence_l2", 0.0)),
                "divergence_reduction_ratio_linf": float(
                    p.get("divergence_reduction_ratio", 0.0)
                ),
                "divergence_reduction_ratio_l2": float(
                    p.get("divergence_reduction_ratio_l2", 0.0)
                ),
                "inlet_flux_after": inlet_flux_after,
                "outlet_flux_after": outlet_flux_after,
                "outlet_inlet_flux_ratio": float(
                    outlet_flux_after / max(abs(inlet_flux_after), 1e-30)
                ),
                "net_boundary_flux_after": net_boundary_flux_after,
                "net_boundary_flux_relative": float(
                    abs(net_boundary_flux_after) / max(abs(inlet_flux_after), 1e-30)
                ),
                "wall_flux_max_abs_after": float(p.get("wall_flux_max_abs_after", 0.0)),
                "flow_projection_accepted": bool(acc.get("projection_accepted", False)),
                "projection_accepted": bool(acc.get("projection_accepted", False)),
                "projection_solved": bool(acc.get("projection_solved", False)),
                "used_cuda": bool(be.get("all_core_arrays_on_cuda", False)),
                "top_hotspot_abs_div": float(
                    d.get("top_divergence_cells", {})
                    .get("cells", [{}])[0]
                    .get("abs_div_corrected", 0.0)
                    if isinstance(d.get("top_divergence_cells", {}), dict)
                    and d.get("top_divergence_cells", {}).get("cells")
                    else 0.0
                ),
                "number_of_limited_cells": int(lim.get("number_of_limited_cells", 0)),
                "number_of_limited_faces": int(lim.get("number_of_limited_faces", 0)),
                "correction_mass_or_flux_delta": float(
                    lim_cons.get("limiter_flux_delta_total", 0.0)
                ),
                "pairwise_interior_conservation_preserved": bool(
                    lim.get("pairwise_interior_conservation_preserved", True)
                ),
                "acceptance_reason": str(acc.get("reason", "")),
            }

        pressure_solver_comparison: dict[str, Any] = {}
        if bool(args.compare_pressure_solvers):
            comparison_items: list[dict[str, Any]] = []
            compare_cfgs = [
                (
                    "jacobi_2000",
                    replace(
                        cfg, pressure_solver="jacobi", max_pressure_iterations=2000
                    ),
                ),
                (
                    "pcg_diag_1000",
                    replace(
                        cfg, pressure_solver="pcg_diag", max_pressure_iterations=1000
                    ),
                ),
            ]
            for label, c_cfg in compare_cfgs:
                t0 = perf_counter()
                s_cmp = solve_tetra_pressure_projection(mesh, state0, c_cfg)
                dt = perf_counter() - t0
                item = _projection_summary(label, c_cfg, s_cmp)
                item["runtime_seconds"] = float(dt)
                comparison_items.append(item)
            accepted = [
                it
                for it in comparison_items
                if bool(it.get("flow_projection_accepted", False))
            ]
            if accepted:
                accepted_sorted = sorted(
                    accepted,
                    key=lambda it: (
                        float(it.get("final_div_l2", float("inf"))),
                        float(abs(it.get("net_boundary_flux_after", float("inf")))),
                        float(it.get("runtime_seconds", float("inf"))),
                    ),
                )
                recommended = str(accepted_sorted[0].get("pressure_solver", ""))
            else:
                fallback_sorted = sorted(
                    comparison_items,
                    key=lambda it: (
                        int(not bool(it.get("pressure_solved", False))),
                        float(it.get("residual_ratio_to_rhs_l2", float("inf"))),
                        float(it.get("final_div_l2", float("inf"))),
                    ),
                )
                recommended = str(fallback_sorted[0].get("pressure_solver", ""))
            pressure_solver_comparison = {
                "solvers": comparison_items,
                "recommended_pressure_solver": recommended,
            }
            _write_json(
                run_dir / "pressure_solver_comparison.json", pressure_solver_comparison
            )

        rhs_mode_comparison: dict[str, Any] = {}
        damping_comparison: dict[str, Any] = {}
        correction_limit_mode_comparison: dict[str, Any] = {}
        limit_mode_states: dict[str, Any] = {}
        if False and bool(args.compare_pressure_solvers):
            rhs_mode_items: list[dict[str, Any]] = []
            rhs_modes = [str(cfg.projection_rhs_mode)]
            alt_mode = (
                "divergence_per_volume"
                if str(cfg.projection_rhs_mode) == "volume_integrated_flux"
                else "volume_integrated_flux"
            )
            if alt_mode not in rhs_modes:
                rhs_modes.append(alt_mode)
            for mode in rhs_modes:
                c_cfg = replace(cfg, projection_rhs_mode=mode)  # type: ignore[arg-type]
                t0 = perf_counter()
                s_mode = solve_tetra_pressure_projection(mesh, state0, c_cfg)
                dt = perf_counter() - t0
                item = _projection_summary(f"rhs_mode_{mode}", c_cfg, s_mode)
                item["runtime_seconds"] = float(dt)
                rhs_mode_items.append(item)
            rhs_mode_comparison = {
                "modes": rhs_mode_items,
                "current_default_mode": str(cfg.projection_rhs_mode),
            }
            _write_json(
                run_dir / "projection_rhs_mode_comparison.json", rhs_mode_comparison
            )

            damping_items: list[dict[str, Any]] = []
            for damping in (1.0, 0.5, 0.25):
                c_cfg = replace(cfg, projection_correction_damping=float(damping))
                t0 = perf_counter()
                s_damp = solve_tetra_pressure_projection(mesh, state0, c_cfg)
                dt = perf_counter() - t0
                item = _projection_summary(f"damping_{damping:.2f}", c_cfg, s_damp)
                item["runtime_seconds"] = float(dt)
                damping_items.append(item)
            damping_comparison = {"items": damping_items}
            _write_json(
                run_dir / "projection_damping_comparison.json", damping_comparison
            )

        if bool(args.compare_correction_limit_modes):
            mode_items: list[dict[str, Any]] = []
            limit_modes = [
                "none",
                "cell_divergence_cap",
                "face_flux_cap",
                "redistribute_local",
            ]
            for mode in limit_modes:
                c_cfg = replace(
                    cfg,
                    projection_correction_limit_mode=mode,  # type: ignore[arg-type]
                )
                t0 = perf_counter()
                s_mode = solve_tetra_pressure_projection(mesh, state0, c_cfg)
                dt = perf_counter() - t0
                limit_mode_states[str(mode)] = s_mode
                item = _projection_summary(f"limit_mode_{mode}", c_cfg, s_mode)
                item["runtime_seconds"] = float(dt)
                mode_items.append(item)
            correction_limit_mode_comparison = {
                "modes": mode_items,
                "experimental": True,
            }
            _write_json(
                run_dir / "projection_correction_limit_comparison.json",
                correction_limit_mode_comparison,
            )

        outlet_mode_comparison: dict[str, Any] = {}
        if bool(args.compare_outlet_projection_modes):
            mode_items: list[dict[str, Any]] = []
            modes = [
                ("outlet_pressure_dirichlet", "production_candidate"),
                ("outlet_flux_preserve", "diagnostic_only"),
                ("outlet_mass_balance_rescale", "diagnostic_only"),
            ]
            for mode, notes in modes:
                c_cfg = replace(cfg, outlet_projection_mode=mode)  # type: ignore[arg-type]
                t0 = perf_counter()
                s_mode = solve_tetra_pressure_projection(mesh, state0, c_cfg)
                dt = perf_counter() - t0
                item = _projection_summary(mode, c_cfg, s_mode)
                reg = dict(s_mode.diagnostics.get("region_divergence_audit", {}))
                item["interior_core_div_max_abs"] = float(
                    reg.get("interior_core", {})
                    .get("corrected", {})
                    .get("max_abs", 0.0)
                )
                item["interior_core_div_l2"] = float(
                    reg.get("interior_core", {}).get("corrected", {}).get("l2", 0.0)
                )
                item["outlet_adjacent_div_max_abs"] = float(
                    reg.get("outlet_adjacent", {})
                    .get("corrected", {})
                    .get("max_abs", 0.0)
                )
                item["outlet_adjacent_div_l2"] = float(
                    reg.get("outlet_adjacent", {}).get("corrected", {}).get("l2", 0.0)
                )
                item["runtime_seconds"] = float(dt)
                item["flow_solved"] = bool(s_mode.diagnostics.get("flow_solved", False))
                item["notes"] = notes
                mode_items.append(item)
            outlet_mode_comparison = {"modes": mode_items}
            _write_json(run_dir / "outlet_mode_comparison.json", outlet_mode_comparison)

        divergence = compute_tetra_flux_divergence(
            mesh,
            state1.face_flux,
            left_inlet_faces=left_faces,
            right_inlet_faces=right_faces,
            outlet_faces=mesh.outlet_faces,
            wall_faces=mesh.wall_faces,
        )["divergence"]
        face_flux_primary = dict(diag.get("face_flux_primary", {}))
        face_flux_star = np.asarray(
            face_flux_primary.get("face_flux_star", np.zeros_like(state1.face_flux)),
            dtype=np.float64,
        )
        face_flux_corrected = np.asarray(state1.face_flux, dtype=np.float64)
        correction_flux = np.asarray(
            face_flux_primary.get(
                "correction_flux", face_flux_corrected - face_flux_star
            ),
            dtype=np.float64,
        )
        correction_flux_raw_pre_limiter = np.asarray(
            face_flux_primary.get(
                "correction_flux_raw_pre_constraint",
                correction_flux,
            ),
            dtype=np.float64,
        )
        correction_flux_limited_pre_bc = np.asarray(
            face_flux_primary.get(
                "correction_flux_constrained_post_limiter_pre_outlet_policy",
                face_flux_primary.get(
                    "correction_flux_limiter_output_pre_reconstraint",
                    correction_flux,
                ),
            ),
            dtype=np.float64,
        )
        speed = np.linalg.norm(state1.cell_velocity, axis=1)

        pressure_png = _save_scatter(
            centers=mesh.cell_centers,
            values=state1.pressure,
            title="Pressure XY",
            label="p",
            out_path=run_dir / "pressure_xy.png",
            cmap="plasma",
        )
        vel_mag_png = _save_scatter(
            centers=mesh.cell_centers,
            values=speed,
            title="Velocity Magnitude XY",
            label="|u|",
            out_path=run_dir / "velocity_magnitude_xy.png",
            cmap="magma",
        )
        div_png = _save_scatter(
            centers=mesh.cell_centers,
            values=divergence,
            title="log10(|divergence|) XY",
            label="log10|div|",
            out_path=run_dir / "divergence_abs_log10_xy.png",
            cmap="viridis",
            log10_abs=True,
        )
        vec_norm_png = _save_vectors_normalized(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_normalized.png",
        )
        vec_ds_png = _save_vectors_downsampled(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_downsampled.png",
        )
        vec_sparse_png = _save_vectors_sparse_normalized(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_sparse_normalized.png",
            max_arrows=1000,
        )
        vec_seeded_png = _save_vectors_sparse_normalized_seeded(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_sparse_normalized_seeded.png",
            max_arrows=300,
            seed=1234,
        )
        vec_raw_png = _save_vectors_raw_clipped_scale(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_raw_clipped_scale.png",
        )
        zone_masks = _cell_zone_masks(mesh)
        velocity_region_masks = _build_velocity_region_masks(mesh, zone_masks)
        vec_region_png = _save_vectors_by_region(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            masks=zone_masks,
            out_path=run_dir / "velocity_vectors_xy_by_region.png",
        )
        vec_region_panels_png = _save_vectors_region_panels(
            mesh=mesh,
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            masks=zone_masks,
            out_path=run_dir / "velocity_vectors_xy_region_panels.png",
            max_per_panel=260,
        )
        vec_grid_binned_png = _save_velocity_vectors_xy_grid_binned(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_grid_binned.png",
            bins_x=42,
            bins_y=42,
        )
        vec_region_panels_binned_png = _save_velocity_vectors_xy_region_panels_binned(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            region_masks=velocity_region_masks,
            out_path=run_dir / "velocity_vectors_xy_region_panels_binned.png",
            bins_x=26,
            bins_y=26,
        )
        vec_direction_colored_png = _save_velocity_vectors_xy_direction_colored(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            out_path=run_dir / "velocity_vectors_xy_direction_colored.png",
            bins_x=42,
            bins_y=42,
        )
        vel_mag_p95_clip_png = _save_velocity_magnitude_p95_clipped_xy(
            centers=mesh.cell_centers,
            speed=speed,
            out_path=run_dir / "velocity_magnitude_p95_clipped_xy.png",
        )
        outlet_flux_png = _save_outlet_flux_faces(
            mesh=mesh,
            corrected_flux=state1.face_flux,
            out_path=run_dir / "outlet_flux_faces_xy.png",
        )
        face_flux_proxy_png = _save_face_flux_stream_proxy(
            mesh=mesh,
            corrected_flux=face_flux_corrected,
            out_path=run_dir / "face_flux_stream_proxy_xy.png",
            max_faces=1200,
        )
        pressure_outlet_zoom_png = _save_pressure_outlet_zoom(
            mesh=mesh,
            pressure=state1.pressure,
            out_path=run_dir / "pressure_outlet_zoom_xy.png",
        )
        divergence_stage_comparison_step_0100_png: str | None = None
        velocity_magnitude_before_after_predictor_step_0100_png: str | None = None
        face_flux_delta_predictor_xy_step_0100_png: str | None = None
        viscous_predictor_stage_hotspots: dict[str, Any] = {}
        visc_final = dict(diag.get("viscous_predictor", {}))
        visc_arrays = dict(visc_final.get("arrays", {}))
        if (
            flow_mode in {"stokes_viscous_projection", "navier_stokes_projection_debug"}
            and visc_arrays
        ):
            div_b = np.asarray(
                visc_arrays.get(
                    "divergence_before_predictor",
                    np.asarray(divergence, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            div_p = np.asarray(
                visc_arrays.get(
                    "divergence_after_predictor_before_boundary_contract",
                    np.asarray(divergence, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            div_c = np.asarray(
                visc_arrays.get(
                    "divergence_after_boundary_contract_before_projection",
                    np.asarray(divergence, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            divergence_stage_comparison_step_0100_png = (
                _save_divergence_stage_comparison(
                    centers=mesh.cell_centers,
                    div_before=div_b,
                    div_after_predictor=div_p,
                    div_after_contract=div_c,
                    div_after_projection=np.asarray(divergence, dtype=np.float64),
                    out_path=run_dir / "divergence_stage_comparison_step_0100.png",
                )
            )
            vel_b = np.asarray(
                visc_arrays.get(
                    "velocity_before_predictor",
                    np.asarray(state1.cell_velocity, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            vel_p = np.asarray(
                visc_arrays.get(
                    "velocity_after_predictor",
                    np.asarray(state1.cell_velocity, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            velocity_magnitude_before_after_predictor_step_0100_png = (
                _save_velocity_magnitude_before_after_predictor(
                    centers=mesh.cell_centers,
                    vel_before=vel_b,
                    vel_after_predictor=vel_p,
                    out_path=run_dir
                    / "velocity_magnitude_before_after_predictor_step_0100.png",
                )
            )
            flux_b = np.asarray(
                visc_arrays.get(
                    "face_flux_before_predictor",
                    np.asarray(state1.face_flux, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            flux_p = np.asarray(
                visc_arrays.get(
                    "face_flux_after_predictor_before_contract",
                    np.asarray(state1.face_flux, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            flux_c = np.asarray(
                visc_arrays.get(
                    "face_flux_after_contract",
                    np.asarray(state1.face_flux, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            stage_rows = {
                "before_predictor": {
                    "divergence": div_b,
                    "face_flux": flux_b,
                    "baseline_divergence": div_b,
                },
                "after_predictor_before_boundary_contract": {
                    "divergence": div_p,
                    "face_flux": flux_p,
                    "baseline_divergence": div_b,
                },
                "after_boundary_contract_before_projection": {
                    "divergence": div_c,
                    "face_flux": flux_c,
                    "baseline_divergence": div_p,
                },
            }
            viscous_predictor_stage_hotspots = {
                "viscous_predictor_mode": str(
                    visc_final.get("viscous_predictor_mode", "")
                ),
                "viscous_predictor_outlet_contract_mode": str(
                    visc_final.get("viscous_predictor_outlet_contract_mode", "")
                ),
                "wall_velocity_boundary_mode": str(
                    visc_final.get("wall_velocity_boundary_mode", "")
                ),
                "wall_tangential_shear_face_flux_enabled": bool(
                    visc_final.get("wall_tangential_shear_face_flux_enabled", False)
                ),
                "wall_tangential_cell_velocity_momentum_enabled": bool(
                    visc_final.get(
                        "wall_tangential_cell_velocity_momentum_enabled", False
                    )
                ),
                "stage_stats": {
                    name: {
                        "divergence_l2": float(
                            np.sqrt(np.mean(np.asarray(row["divergence"]) ** 2))
                        )
                        if np.asarray(row["divergence"]).size
                        else 0.0,
                        "divergence_max_abs": float(
                            np.max(np.abs(np.asarray(row["divergence"])))
                        )
                        if np.asarray(row["divergence"]).size
                        else 0.0,
                    }
                    for name, row in stage_rows.items()
                },
                "top_divergence_cells_by_stage": {
                    name: _top_divergence_cells(
                        mesh=mesh,
                        div_star=np.asarray(
                            row["baseline_divergence"], dtype=np.float64
                        ),
                        div_corr=np.asarray(row["divergence"], dtype=np.float64),
                        face_flux_corrected=np.asarray(
                            row["face_flux"], dtype=np.float64
                        ),
                        masks=zone_masks,
                        top_k=30,
                    )
                    for name, row in stage_rows.items()
                },
            }
            stage_stats = dict(viscous_predictor_stage_hotspots["stage_stats"])
            before_stats = dict(stage_stats.get("before_predictor", {}))
            predictor_stats = dict(
                stage_stats.get("after_predictor_before_boundary_contract", {})
            )
            contract_stats = dict(
                stage_stats.get("after_boundary_contract_before_projection", {})
            )
            viscous_predictor_stage_hotspots["stage_ratios"] = {
                "predictor_over_before_l2": _ratio(
                    predictor_stats.get("divergence_l2", 0.0),
                    before_stats.get("divergence_l2", 0.0),
                ),
                "contract_over_predictor_l2": _ratio(
                    contract_stats.get("divergence_l2", 0.0),
                    predictor_stats.get("divergence_l2", 0.0),
                ),
                "predictor_over_before_max": _ratio(
                    predictor_stats.get("divergence_max_abs", 0.0),
                    before_stats.get("divergence_max_abs", 0.0),
                ),
                "contract_over_predictor_max": _ratio(
                    contract_stats.get("divergence_max_abs", 0.0),
                    predictor_stats.get("divergence_max_abs", 0.0),
                ),
            }
            face_flux_delta_predictor_xy_step_0100_png = (
                _save_face_flux_delta_predictor_xy(
                    mesh=mesh,
                    face_flux_before=flux_b,
                    face_flux_after_predictor=flux_p,
                    out_path=run_dir / "face_flux_delta_predictor_xy_step_0100.png",
                )
            )
        if flow_mode in {"stokes_viscous_projection", "navier_stokes_projection_debug"}:
            step50_src = run_dir / "snapshots" / "step_0050" / "divergence_xy.png"
            if step50_src.exists():
                (run_dir / "divergence_xy_step_0050.png").write_bytes(
                    step50_src.read_bytes()
                )
            step100_src = run_dir / "snapshots" / "step_0100" / "divergence_xy.png"
            if step100_src.exists():
                (run_dir / "divergence_xy_step_0100.png").write_bytes(
                    step100_src.read_bytes()
                )
            if (
                _postprocessing_writes_visualizations(args.postprocessing_mode)
                and flow_steps_requested >= 100
            ):
                (
                    run_dir / "velocity_magnitude_p95_clipped_xy_step_0100.png"
                ).write_bytes(
                    (run_dir / "velocity_magnitude_p95_clipped_xy.png").read_bytes()
                )
                (run_dir / "velocity_vectors_xy_grid_binned_step_0100.png").write_bytes(
                    (run_dir / "velocity_vectors_xy_grid_binned.png").read_bytes()
                )
                (
                    run_dir / "velocity_vectors_xy_region_panels_binned_step_0100.png"
                ).write_bytes(
                    (
                        run_dir / "velocity_vectors_xy_region_panels_binned.png"
                    ).read_bytes()
                )
        vtu_path = (
            _export_vtu(
                points=mesh.points,
                tetrahedra=mesh.tetrahedra,
                pressure=state1.pressure,
                divergence=divergence,
                velocity=state1.cell_velocity,
                out_path=run_dir / f"{mesh_stem}_flow_result.vtu",
            )
            if _postprocessing_writes_visualizations(args.postprocessing_mode)
            else None
        )

        projection_audit = dict(diag.get("projection_audit", {}))
        pressure_history = dict(diag.get("pressure_solver_history", {}))
        boundary_flux_audit = dict(diag.get("boundary_flux_audit", {}))
        region_divergence_audit = dict(diag.get("region_divergence_audit", {}))
        operator_consistency = dict(diag.get("operator_consistency_audit", {}))
        sign_comparison = dict(diag.get("projection_sign_comparison_fixed_rhs", {}))
        outlet_projection_audit = dict(diag.get("outlet_projection_audit", {}))
        correction_limiter = dict(diag.get("correction_limiter", {}))
        correction_limiter_conservation_audit = dict(
            diag.get("correction_limiter_conservation_audit", {})
        )
        pressure_diag = dict(diag.get("pressure", {}))
        top_velocity = _top_velocity_cells(mesh=mesh, velocity=state1.cell_velocity)
        raw_top_divergence = diag.get("top_divergence_cells", {})
        if isinstance(raw_top_divergence, list):
            top_divergence_cells = list(raw_top_divergence)
            top_divergence_summary = {}
        else:
            top_divergence = dict(raw_top_divergence)
            top_divergence_cells = list(top_divergence.get("cells", []))
            top_divergence_summary = dict(top_divergence.get("hotspot_summary", {}))

        coeff = _build_pressure_system_coefficients(
            mesh,
            dt=float(cfg.projection_dt),
            density=float(cfg.density),
            outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        )
        star_sum = _compute_cell_flux_sum(mesh, face_flux_star)
        rhs_used, rhs_outlet_used = _assemble_poisson_rhs(
            star_sum,
            coeff,
            pressure_outlet_value=float(cfg.pressure_outlet_value),
            projection_sign=cfg.projection_sign,
            cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
            rhs_mode=cfg.projection_rhs_mode,
        )
        frozen_nonorthogonal_gradient_flux = np.asarray(
            face_flux_primary.get(
                "pressure_gradient_flux_nonorthogonal_frozen",
                np.zeros_like(face_flux_star),
            ),
            dtype=np.float64,
        )
        nonorthogonal_correction_enabled = (
            cfg_effective.pressure_nonorthogonal_correction_mode == "deferred_lsq"
        )
        if nonorthogonal_correction_enabled:
            rhs_used = rhs_used + _pressure_nonorthogonal_rhs_term(
                mesh,
                frozen_nonorthogonal_gradient_flux,
                rhs_mode=cfg_effective.projection_rhs_mode,
            )
        grad_flux_solution = np.asarray(
            _pressure_face_gradient_flux(
                mesh,
                np.asarray(state1.pressure, dtype=np.float64),
                dt=float(cfg.projection_dt),
                density=float(cfg.density),
                outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
                pressure_outlet_value=float(cfg.pressure_outlet_value),
                frozen_nonorthogonal_gradient_flux=(
                    frozen_nonorthogonal_gradient_flux
                    if nonorthogonal_correction_enabled
                    else None
                ),
            ),
            dtype=np.float64,
        )
        operator_identity_audit = _build_operator_identity_audit(
            mesh=mesh,
            coeff=coeff,
            dt=float(cfg.projection_dt),
            density=float(cfg.density),
            outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
            pressure_outlet_value=float(cfg.pressure_outlet_value),
            pressure_solution=np.asarray(state1.pressure, dtype=np.float64),
            masks=zone_masks,
        )
        projection_equation_residual = _build_projection_equation_residual_audit(
            mesh=mesh,
            flux_star=face_flux_star,
            flux_corrected=face_flux_corrected,
            correction_flux_raw_pre_limiter=correction_flux_raw_pre_limiter,
            correction_flux_limited_pre_bc=correction_flux_limited_pre_bc,
            correction_flux_effective_post_bc=correction_flux,
            pressure_gradient_flux=grad_flux_solution,
            rhs_used=rhs_used,
            projection_sign=str(cfg.projection_sign),
            rhs_mode=str(cfg.projection_rhs_mode),
            dt=float(cfg.projection_dt),
            density=float(cfg.density),
            step_number=int(len(progression_history)),
            stage_name=(
                "progression_final_step"
                if bool(flow_progression_enabled)
                else "single_projection_step"
            ),
            masks=zone_masks,
        )
        projection_equation_residual["numerical_profile"] = {
            "pressure_nonorthogonal_correction_mode_requested": str(
                cfg.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_mode_effective": str(
                cfg_effective.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_enabled": bool(
                nonorthogonal_correction_enabled
            ),
        }
        scale_audit = _build_pressure_projection_scale_audit(
            mesh=mesh,
            rhs=rhs_used,
            star_sum=star_sum,
            flux_star=face_flux_star,
            flux_corrected=face_flux_corrected,
            correction_flux_raw=correction_flux_raw_pre_limiter,
            pressure_gradient_flux=grad_flux_solution,
            masks=zone_masks,
        )
        hotspot_correlation = _build_pressure_projection_hotspot_correlation(
            mesh=mesh,
            div_star=np.asarray(div_before["divergence"], dtype=np.float64),
            div_corrected=np.asarray(divergence, dtype=np.float64),
            correction_flux_raw=correction_flux_raw_pre_limiter,
        )
        matrix_coefficient_audit = _build_pressure_matrix_coefficient_audit(
            mesh=mesh,
            coeff=coeff,
            top_divergence_cells=top_divergence_cells,
        )
        volume_weighting_audit = _build_projection_volume_weighting_audit(
            mesh=mesh,
            coeff=coeff,
            pressure=np.asarray(state1.pressure, dtype=np.float64),
            rhs=rhs_used,
            rhs_outlet=rhs_outlet_used,
            star_sum=star_sum,
            flux_star=face_flux_star,
            flux_corrected=face_flux_corrected,
            projection_sign=str(cfg.projection_sign),
        )
        pressure_matrix_explicit_audit: dict[str, Any] = {}
        pressure_operator_matrixfree_vs_explicit: dict[str, Any] = {}
        pressure_operator_spd_audit: dict[str, Any] = {}
        pressure_reference_solver_comparison: dict[str, Any] = {}
        operator_symmetry_hotspots_png = ""
        operator_matrixfree_mismatch_png = ""
        explicit_vs_matrixfree_hist_png = ""
        if bool(args.audit_pressure_operator):
            pressure_matrix_explicit_audit = _pressure_matrix_explicit_audit(
                coeff,
                int(mesh.tetrahedra.shape[0]),
            )
            pressure_operator_matrixfree_vs_explicit = (
                _pressure_matrixfree_vs_explicit_audit(
                    coeff,
                    n_cells=int(mesh.tetrahedra.shape[0]),
                    pressure_solution=np.asarray(state1.pressure, dtype=np.float64),
                    random_count=10,
                    seed=20260504,
                )
            )
            pressure_operator_spd_audit = _pressure_operator_spd_audit(
                coeff,
                n_cells=int(mesh.tetrahedra.shape[0]),
                random_count=40,
                seed=20260504,
            )

            mat_entries = dict(pressure_matrix_explicit_audit.get("matrix_entries", {}))
            rows_e = np.asarray(mat_entries.get("rows", []), dtype=np.int64)
            cols_e = np.asarray(mat_entries.get("cols", []), dtype=np.int64)
            vals_e = np.asarray(mat_entries.get("vals", []), dtype=np.float64)
            n_cells = int(mesh.tetrahedra.shape[0])
            row_sym = np.zeros((n_cells,), dtype=np.float64)
            if rows_e.size:
                key = rows_e * n_cells + cols_e
                val_map = {
                    int(k): float(v) for k, v in zip(key.tolist(), vals_e.tolist())
                }
                for k_int, v in val_map.items():
                    i = int(k_int // n_cells)
                    j = int(k_int % n_cells)
                    vt = float(val_map.get(j * n_cells + i, 0.0))
                    row_sym[i] = max(row_sym[i], abs(v - vt))
            mismatch_vec = np.asarray(
                pressure_operator_matrixfree_vs_explicit.get(
                    "worst_diff_vector", np.zeros((n_cells,), dtype=np.float64)
                ),
                dtype=np.float64,
            )
            operator_symmetry_hotspots_png = (
                _save_pressure_operator_symmetry_hotspots_xy(
                    centers=mesh.cell_centers,
                    row_symmetry_error=row_sym,
                    out_path=run_dir / "pressure_operator_symmetry_hotspots_xy.png",
                )
            )
            operator_matrixfree_mismatch_png = (
                _save_pressure_operator_matrixfree_mismatch_xy(
                    centers=mesh.cell_centers,
                    mismatch=mismatch_vec,
                    out_path=run_dir / "pressure_operator_matrixfree_mismatch_xy.png",
                )
            )
            explicit_vs_matrixfree_hist_png = (
                _save_explicit_vs_matrixfree_residual_histogram(
                    mismatch=mismatch_vec,
                    out_path=run_dir / "explicit_vs_matrixfree_residual_histogram.png",
                )
            )

            ref = _solve_pressure_reference_explicit(
                coeff,
                rhs=np.asarray(rhs_used, dtype=np.float64),
                x0=np.zeros_like(np.asarray(rhs_used, dtype=np.float64)),
                rtol=float(cfg.pressure_relative_tolerance),
                maxiter=max(int(cfg.max_pressure_iterations), 5000),
            )
            ref_methods_out: list[dict[str, Any]] = []
            ref_methods = list(ref.get("methods", []))
            for method in ref_methods:
                p_ref = method.get("pressure_solution", None)
                row = dict(method)
                if isinstance(p_ref, np.ndarray):
                    flux_ref, _ = _apply_policy_from_pressure(
                        mesh=mesh,
                        pressure=np.asarray(p_ref, dtype=np.float64),
                        flux_star=face_flux_star,
                        projection_sign=str(cfg.projection_sign),
                        projection_correction_damping=float(
                            cfg.projection_correction_damping
                        ),
                        dt=float(cfg.projection_dt),
                        density=float(cfg.density),
                        pressure_outlet_value=float(cfg.pressure_outlet_value),
                        policy="outlet_pressure_dirichlet",
                        frozen_nonorthogonal_gradient_flux=(
                            frozen_nonorthogonal_gradient_flux
                            if nonorthogonal_correction_enabled
                            else None
                        ),
                    )
                    div_ref = compute_tetra_flux_divergence(
                        mesh,
                        flux_ref,
                        left_inlet_faces=left_faces,
                        right_inlet_faces=right_faces,
                        outlet_faces=mesh.outlet_faces,
                        wall_faces=mesh.wall_faces,
                    )
                    div_vals = np.asarray(div_ref["divergence"], dtype=np.float64)
                    row.update(
                        {
                            "final_div_max_abs": float(np.max(np.abs(div_vals))),
                            "final_div_l2": float(
                                np.sqrt(np.mean(div_vals * div_vals))
                            ),
                            "inlet_flux_after": float(
                                div_ref.get("inlet_flux_total", 0.0)
                            ),
                            "outlet_flux_after": float(
                                div_ref.get("outlet_flux_total", 0.0)
                            ),
                            "net_boundary_flux_after": float(
                                div_ref.get("net_boundary_flux", 0.0)
                            ),
                            "pressure_min": float(
                                np.min(np.asarray(p_ref, dtype=np.float64))
                            ),
                            "pressure_max": float(
                                np.max(np.asarray(p_ref, dtype=np.float64))
                            ),
                        }
                    )
                    row.pop("pressure_solution", None)
                ref_methods_out.append(row)
            current_methods: list[dict[str, Any]] = []
            current_methods.append(_projection_summary("current_main", cfg, state1))
            if pressure_solver_comparison:
                current_methods.extend(
                    list(pressure_solver_comparison.get("solvers", []))
                )
            pressure_reference_solver_comparison = {
                "scipy_available": bool(ref.get("scipy_available", False)),
                "matrix_nnz": int(ref.get("matrix_nnz", 0)),
                "current_solvers": current_methods,
                "reference_solvers": ref_methods_out,
                "notes": "reference solver uses same explicit A and outlet_pressure_dirichlet projection policy",
            }

            # Keep JSON compact.
            if "matrix_entries" in pressure_matrix_explicit_audit:
                pressure_matrix_explicit_audit = dict(pressure_matrix_explicit_audit)
                pressure_matrix_explicit_audit.pop("matrix_entries", None)
            pressure_operator_matrixfree_vs_explicit = dict(
                pressure_operator_matrixfree_vs_explicit
            )
            pressure_operator_matrixfree_vs_explicit.pop("worst_diff_vector", None)

        if bool(args.compare_pressure_solvers) and (
            not pressure_reference_solver_comparison
        ):
            ref = _solve_pressure_reference_explicit(
                coeff,
                rhs=np.asarray(rhs_used, dtype=np.float64),
                x0=np.zeros_like(np.asarray(rhs_used, dtype=np.float64)),
                rtol=float(cfg.pressure_relative_tolerance),
                maxiter=max(int(cfg.max_pressure_iterations), 5000),
            )
            ref_methods_out: list[dict[str, Any]] = []
            for method in list(ref.get("methods", [])):
                p_ref = method.get("pressure_solution", None)
                row = dict(method)
                if isinstance(p_ref, np.ndarray):
                    flux_ref, _ = _apply_policy_from_pressure(
                        mesh=mesh,
                        pressure=np.asarray(p_ref, dtype=np.float64),
                        flux_star=face_flux_star,
                        projection_sign=str(cfg.projection_sign),
                        projection_correction_damping=float(
                            cfg.projection_correction_damping
                        ),
                        dt=float(cfg.projection_dt),
                        density=float(cfg.density),
                        pressure_outlet_value=float(cfg.pressure_outlet_value),
                        policy="outlet_pressure_dirichlet",
                        frozen_nonorthogonal_gradient_flux=(
                            frozen_nonorthogonal_gradient_flux
                            if nonorthogonal_correction_enabled
                            else None
                        ),
                    )
                    div_ref = compute_tetra_flux_divergence(
                        mesh,
                        flux_ref,
                        left_inlet_faces=left_faces,
                        right_inlet_faces=right_faces,
                        outlet_faces=mesh.outlet_faces,
                        wall_faces=mesh.wall_faces,
                    )
                    dv = np.asarray(div_ref["divergence"], dtype=np.float64)
                    row.update(
                        {
                            "final_div_max_abs": float(np.max(np.abs(dv))),
                            "final_div_l2": float(np.sqrt(np.mean(dv * dv))),
                            "inlet_flux_after": float(
                                div_ref.get("inlet_flux_total", 0.0)
                            ),
                            "outlet_flux_after": float(
                                div_ref.get("outlet_flux_total", 0.0)
                            ),
                            "net_boundary_flux_after": float(
                                div_ref.get("net_boundary_flux", 0.0)
                            ),
                            "wall_flux_max_abs_after": float(
                                div_ref.get("wall_flux_max_abs", 0.0)
                            ),
                        }
                    )
                    row.pop("pressure_solution", None)
                ref_methods_out.append(row)
            pressure_reference_solver_comparison = {
                "scipy_available": bool(ref.get("scipy_available", False)),
                "matrix_nnz": int(ref.get("matrix_nnz", 0)),
                "reference_solvers": ref_methods_out,
                "notes": "reference solver uses same explicit A and outlet_pressure_dirichlet projection policy",
            }

        velocity_recon_audit = _velocity_reconstruction_audit(
            mesh=mesh,
            corrected_face_flux=face_flux_corrected,
            reconstructed_cell_velocity=state1.cell_velocity,
            masks=zone_masks,
            region_masks=velocity_region_masks,
        )
        velocity_region_audit = _velocity_region_audit(
            centers=mesh.cell_centers,
            velocity=state1.cell_velocity,
            region_masks=velocity_region_masks,
        )
        top_suspicious_velocity_cells = {
            "cells": list(
                velocity_recon_audit.get("top_suspicious_velocity_cells", [])
            ),
            "count": int(
                velocity_recon_audit.get("suspicious_velocity_cells_count", 0)
            ),
        }
        top_divergence_correction_breakdown = _top_divergence_correction_breakdown(
            mesh=mesh,
            top_cells=top_divergence_cells,
            pressure=np.asarray(state1.pressure, dtype=np.float64),
            flux_star=face_flux_star,
            flux_corrected=face_flux_corrected,
            correction_flux_raw=correction_flux_raw_pre_limiter,
        )
        top_divergence_local_face_audit = _build_top_divergence_local_face_audit(
            mesh=mesh,
            top_cells=top_divergence_cells,
            pressure=np.asarray(state1.pressure, dtype=np.float64),
            flux_star=face_flux_star,
            flux_corrected=face_flux_corrected,
            correction_flux_raw=correction_flux_raw_pre_limiter,
            pressure_outlet_value=float(cfg.pressure_outlet_value),
        )
        projection_hotspot_before_after_limiters: dict[str, Any] = {}
        if bool(args.compare_correction_limit_modes) and limit_mode_states:
            baseline_state = limit_mode_states.get("none", None)
            if baseline_state is None:
                baseline_state = state1
            base_diag = dict(getattr(baseline_state, "diagnostics", {}))
            base_top = (
                list(dict(base_diag.get("top_divergence_cells", {})).get("cells", []))[
                    :30
                ]
                if isinstance(base_diag.get("top_divergence_cells", {}), dict)
                else []
            )
            baseline_cell_ids = [
                int(r.get("cell_index", -1))
                for r in base_top
                if int(r.get("cell_index", -1)) >= 0
            ]
            mode_rows: list[dict[str, Any]] = []
            for mode_name, mode_state in limit_mode_states.items():
                md = dict(getattr(mode_state, "diagnostics", {}))
                face_primary_mode = dict(md.get("face_flux_primary", {}))
                q_star_mode = np.asarray(
                    face_primary_mode.get("face_flux_star", mode_state.face_flux),
                    dtype=np.float64,
                )
                q_raw_mode = np.asarray(
                    face_primary_mode.get(
                        "correction_flux_raw_pre_constraint",
                        face_primary_mode.get(
                            "correction_flux", np.zeros_like(q_star_mode)
                        ),
                    ),
                    dtype=np.float64,
                )
                q_lim_mode = np.asarray(
                    face_primary_mode.get(
                        "correction_flux_constrained_post_limiter_pre_outlet_policy",
                        face_primary_mode.get(
                            "correction_flux", np.zeros_like(q_star_mode)
                        ),
                    ),
                    dtype=np.float64,
                )
                div_before_lim_mode = _compute_cell_flux_sum(
                    mesh, q_star_mode + q_raw_mode
                ) / np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
                div_after_lim_mode = _compute_cell_flux_sum(
                    mesh, q_star_mode + q_lim_mode
                ) / np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
                lim_diag_mode = dict(md.get("correction_limiter", {}))
                limited_faces_mode = set(
                    np.asarray(
                        lim_diag_mode.get("limited_face_indices", []), dtype=np.int64
                    ).tolist()
                )
                top_mode = (
                    list(dict(md.get("top_divergence_cells", {})).get("cells", []))
                    if isinstance(md.get("top_divergence_cells", {}), dict)
                    else []
                )
                top_mode_index = (
                    int(top_mode[0].get("cell_index", -1)) if top_mode else -1
                )
                per_cell_rows: list[dict[str, Any]] = []
                for cid in baseline_cell_ids:
                    fids = np.asarray(mesh.cell_to_faces[cid], dtype=np.int64)
                    per_cell_rows.append(
                        {
                            "cell_index": int(cid),
                            "div_before_limiter": float(div_before_lim_mode[cid]),
                            "div_after_limiter": float(div_after_lim_mode[cid]),
                            "correction_fluxes_before": np.asarray(
                                q_raw_mode[fids], dtype=np.float64
                            ),
                            "correction_fluxes_after": np.asarray(
                                q_lim_mode[fids], dtype=np.float64
                            ),
                            "which_faces_limited": [
                                int(fid)
                                for fid in fids.tolist()
                                if int(fid) in limited_faces_mode
                            ],
                            "limited_face_count_in_cell": int(
                                np.count_nonzero(
                                    np.asarray(
                                        [
                                            int(fid) in limited_faces_mode
                                            for fid in fids.tolist()
                                        ],
                                        dtype=bool,
                                    )
                                )
                            ),
                        }
                    )
                mode_rows.append(
                    {
                        "mode": str(mode_name),
                        "top_hotspot_cell_index_after_mode": int(top_mode_index),
                        "did_hotspot_move_elsewhere": bool(
                            top_mode_index not in set(baseline_cell_ids)
                        ),
                        "cells": per_cell_rows,
                    }
                )
            projection_hotspot_before_after_limiters = {
                "baseline_mode": "none",
                "baseline_top_cell_indices": baseline_cell_ids,
                "modes": mode_rows,
            }
        boundary_policy_items: list[dict[str, Any]] = []
        for policy_name in (
            "outlet_pressure_dirichlet",
            "outlet_neumann_flux_preserving",
            "boundary_flux_preserving",
        ):
            flux_pol, _ = _apply_policy_from_pressure(
                mesh=mesh,
                pressure=np.asarray(state1.pressure, dtype=np.float64),
                flux_star=face_flux_star,
                projection_sign=str(cfg.projection_sign),
                projection_correction_damping=float(cfg.projection_correction_damping),
                dt=float(cfg.projection_dt),
                density=float(cfg.density),
                pressure_outlet_value=float(cfg.pressure_outlet_value),
                policy=policy_name,
                frozen_nonorthogonal_gradient_flux=(
                    frozen_nonorthogonal_gradient_flux
                    if nonorthogonal_correction_enabled
                    else None
                ),
            )
            pol_diag = compute_tetra_flux_divergence(
                mesh,
                flux_pol,
                left_inlet_faces=left_faces,
                right_inlet_faces=right_faces,
                outlet_faces=mesh.outlet_faces,
                wall_faces=mesh.wall_faces,
            )
            div_pol = np.asarray(pol_diag["divergence"], dtype=np.float64)
            inlet_flux = float(pol_diag.get("inlet_flux_total", 0.0))
            outlet_flux = float(pol_diag.get("outlet_flux_total", 0.0))
            p_metrics = {
                "divergence_reduction_ratio_l2": float(
                    np.sqrt(np.mean(div_pol * div_pol))
                    / max(
                        float(
                            np.sqrt(
                                np.mean(
                                    np.asarray(
                                        div_before["divergence"], dtype=np.float64
                                    )
                                    ** 2
                                )
                            )
                        ),
                        1e-30,
                    )
                ),
                "net_boundary_flux_after": float(
                    pol_diag.get("net_boundary_flux", 0.0)
                ),
                "inlet_flux_total_after": inlet_flux,
                "outlet_flux_total_after": outlet_flux,
                "wall_flux_max_abs_after": float(
                    pol_diag.get("wall_flux_max_abs", 0.0)
                ),
                "divergence_reduction_ratio": float(
                    np.max(np.abs(div_pol))
                    / max(
                        float(
                            np.max(
                                np.abs(
                                    np.asarray(
                                        div_before["divergence"], dtype=np.float64
                                    )
                                )
                            )
                        ),
                        1e-30,
                    )
                ),
            }
            acc_pol = _projection_acceptance(
                projection=p_metrics,
                pressure=pressure_diag,
                backend_execution=backend_exec,
                thresholds=acceptance_thresholds_cli,
            )
            boundary_policy_items.append(
                {
                    "policy": policy_name,
                    "final_div_linf": float(np.max(np.abs(div_pol))),
                    "final_div_l2": float(np.sqrt(np.mean(div_pol * div_pol))),
                    "div_linf_ratio": float(p_metrics["divergence_reduction_ratio"]),
                    "div_l2_ratio": float(p_metrics["divergence_reduction_ratio_l2"]),
                    "net_boundary_flux_after": float(
                        p_metrics["net_boundary_flux_after"]
                    ),
                    "inlet_flux_after": float(inlet_flux),
                    "outlet_flux_after": float(outlet_flux),
                    "wall_flux_max_abs_after": float(
                        p_metrics["wall_flux_max_abs_after"]
                    ),
                    "accepted": bool(acc_pol.get("projection_accepted", False)),
                    "pressure_solved": bool(
                        pressure_diag.get("pressure_solved", False)
                    ),
                }
            )
        boundary_policy_comparison = {"policies": boundary_policy_items}
        face_group_codes = _face_group_codes(mesh)
        corrected_face_flux_path = run_dir / "corrected_face_flux.npy"
        face_flux_star_path = run_dir / "face_flux_star.npy"
        correction_flux_path = run_dir / "correction_flux.npy"
        face_centers_path = run_dir / "face_centers.npy"
        face_normals_path = run_dir / "face_normals.npy"
        face_to_cells_path = run_dir / "face_to_cells.npy"
        face_groups_path = run_dir / "face_groups.npy"
        rebuilt_face_flux_path = (
            run_dir / "rebuilt_face_flux_from_reconstructed_velocity.npy"
        )
        flux_mismatch_path = run_dir / "flux_reconstruction_mismatch.npy"
        np.save(corrected_face_flux_path, face_flux_corrected)
        np.save(face_flux_star_path, face_flux_star)
        np.save(correction_flux_path, correction_flux)
        np.save(face_centers_path, np.asarray(mesh.face_centers, dtype=np.float64))
        np.save(face_normals_path, np.asarray(mesh.face_normals, dtype=np.float64))
        np.save(face_to_cells_path, np.asarray(mesh.face_to_cells, dtype=np.int64))
        np.save(face_groups_path, face_group_codes)
        np.save(
            rebuilt_face_flux_path,
            np.asarray(
                velocity_recon_audit.get("arrays", {}).get(
                    "rebuilt_face_flux_from_reconstructed_velocity",
                    np.zeros_like(face_flux_corrected),
                ),
                dtype=np.float64,
            ),
        )
        np.save(
            flux_mismatch_path,
            np.asarray(
                velocity_recon_audit.get("arrays", {}).get(
                    "flux_mismatch", np.zeros_like(face_flux_corrected)
                ),
                dtype=np.float64,
            ),
        )
        velocity_recon_audit["arrays"] = {
            "rebuilt_face_flux_from_reconstructed_velocity_npy": str(
                rebuilt_face_flux_path
            ),
            "flux_mismatch_npy": str(flux_mismatch_path),
        }

        _write_json(run_dir / "projection_audit.json", projection_audit)
        _write_json(run_dir / "pressure_solver_history.json", pressure_history)
        _write_json(run_dir / "boundary_flux_audit.json", boundary_flux_audit)
        _write_json(run_dir / "region_divergence_audit.json", region_divergence_audit)
        _write_json(run_dir / "operator_consistency_audit.json", operator_consistency)
        _write_json(
            run_dir / "projection_sign_comparison_fixed_rhs.json", sign_comparison
        )
        _write_json(run_dir / "outlet_projection_audit.json", outlet_projection_audit)
        _write_json(run_dir / "correction_limiter_audit.json", correction_limiter)
        _write_json(
            run_dir / "correction_limiter_conservation_audit.json",
            correction_limiter_conservation_audit,
        )
        _write_json(run_dir / "operator_identity_audit.json", operator_identity_audit)
        _write_json(
            run_dir / "projection_equation_residual.json", projection_equation_residual
        )
        projection_equation_inconsistency_details: dict[str, Any] = {}
        if not bool(
            projection_equation_residual.get(
                "projection_equation_residual_consistent", False
            )
        ):
            projection_equation_inconsistency_details = {
                "stage_name": str(
                    projection_equation_residual.get("audit_context", {}).get(
                        "stage_name", ""
                    )
                ),
                "step_number": int(
                    projection_equation_residual.get("audit_context", {}).get(
                        "step_number", 0
                    )
                ),
                "projection_sign": str(
                    projection_equation_residual.get("audit_context", {}).get(
                        "projection_sign", ""
                    )
                ),
                "rhs_mode": str(
                    projection_equation_residual.get("audit_context", {}).get(
                        "rhs_mode", ""
                    )
                ),
                "rhs_used_stats": dict(
                    projection_equation_residual.get("rhs_used_stats", {})
                ),
                "best_stage_for_consistency": dict(
                    projection_equation_residual.get("best_stage_for_consistency", {})
                ),
                "projection_equation_residual_reason": str(
                    projection_equation_residual.get(
                        "projection_equation_residual_reason", ""
                    )
                ),
                "projection_equation_best_scaling": str(
                    projection_equation_residual.get(
                        "projection_equation_best_scaling", ""
                    )
                ),
                "projection_equation_best_scaling_value": float(
                    projection_equation_residual.get(
                        "projection_equation_best_scaling_value", 0.0
                    )
                ),
                "projection_equation_best_sign": str(
                    projection_equation_residual.get(
                        "projection_equation_best_sign", ""
                    )
                ),
                "projection_equation_best_stage": str(
                    projection_equation_residual.get(
                        "projection_equation_best_stage", ""
                    )
                ),
                "projection_equation_best_relative_l2": float(
                    projection_equation_residual.get(
                        "projection_equation_best_relative_l2", 0.0
                    )
                ),
                "projection_equation_scaling_diagnosis": str(
                    projection_equation_residual.get(
                        "projection_equation_scaling_diagnosis", ""
                    )
                ),
                "projection_equation_consistent_final_post_bc": bool(
                    projection_equation_residual.get(
                        "projection_equation_consistent_final_post_bc", False
                    )
                ),
                "residual_pressure_operator_stats": dict(
                    projection_equation_residual.get(
                        "residual_pressure_operator_stats", {}
                    )
                ),
                "residual_pressure_operator_relative_l2": float(
                    projection_equation_residual.get(
                        "residual_pressure_operator_relative_l2", 0.0
                    )
                ),
                "stage_relative_residuals": list(
                    projection_equation_residual.get("stage_relative_residuals", [])
                ),
                "projection_equation_scaling_candidates": list(
                    projection_equation_residual.get(
                        "projection_equation_scaling_candidates", []
                    )
                ),
                "top_cells_by_residual": list(
                    projection_equation_residual.get("top_cells_by_residual", [])[:30]
                ),
            }
            _write_json(
                run_dir / "projection_equation_residual_inconsistency.json",
                projection_equation_inconsistency_details,
            )
        _write_json(run_dir / "pressure_projection_scale_audit.json", scale_audit)
        _write_json(
            run_dir / "pressure_projection_hotspot_correlation.json",
            hotspot_correlation,
        )
        _write_json(
            run_dir / "pressure_matrix_coefficient_audit.json", matrix_coefficient_audit
        )
        _write_json(
            run_dir / "projection_volume_weighting_audit.json", volume_weighting_audit
        )
        if bool(args.audit_pressure_operator):
            _write_json(
                run_dir / "pressure_matrix_explicit_audit.json",
                pressure_matrix_explicit_audit,
            )
            _write_json(
                run_dir / "pressure_operator_matrixfree_vs_explicit.json",
                pressure_operator_matrixfree_vs_explicit,
            )
            _write_json(
                run_dir / "pressure_operator_spd_audit.json",
                pressure_operator_spd_audit,
            )
            _write_json(
                run_dir / "pressure_reference_solver_comparison.json",
                pressure_reference_solver_comparison,
            )
        _write_json(
            run_dir / "boundary_policy_comparison.json", boundary_policy_comparison
        )
        _write_json(
            run_dir / "top_divergence_correction_breakdown.json",
            top_divergence_correction_breakdown,
        )
        _write_json(
            run_dir / "top_divergence_local_face_audit.json",
            top_divergence_local_face_audit,
        )
        if viscous_predictor_stage_hotspots:
            _write_json(
                run_dir / "viscous_predictor_stage_hotspots.json",
                viscous_predictor_stage_hotspots,
            )
        if projection_hotspot_before_after_limiters:
            _write_json(
                run_dir / "projection_hotspot_before_after_limiters.json",
                projection_hotspot_before_after_limiters,
            )
        _write_json(
            run_dir / "velocity_reconstruction_audit.json", velocity_recon_audit
        )
        _write_json(run_dir / "velocity_region_audit.json", velocity_region_audit)
        _write_json(
            run_dir / "top_suspicious_velocity_cells.json",
            top_suspicious_velocity_cells,
        )
        _write_json(run_dir / "top_velocity_cells.json", top_velocity)
        _write_json(
            run_dir / "top_divergence_cells.json",
            {
                "cells": top_divergence_cells,
                "hotspot_summary": top_divergence_summary,
            },
        )
        div_hotspot_png = _save_divergence_hotspots(
            mesh=mesh,
            top_cells=top_divergence_cells,
            out_path=run_dir / "divergence_hotspot_top_cells_xy.png",
        )
        div_before_after_png = _save_divergence_before_after_same_scale(
            centers=mesh.cell_centers,
            div_before=np.asarray(div_before["divergence"], dtype=np.float64),
            div_after=np.asarray(divergence, dtype=np.float64),
            out_path=run_dir / "divergence_before_after_same_scale.png",
        )
        div_before_limiter = _compute_cell_flux_sum(
            mesh,
            face_flux_star + correction_flux_raw_pre_limiter,
        ) / np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
        div_after_limiter = _compute_cell_flux_sum(
            mesh,
            face_flux_star + correction_flux_limited_pre_bc,
        ) / np.maximum(np.asarray(mesh.cell_volumes, dtype=np.float64), 1e-30)
        div_before_after_limiter_png = _save_divergence_before_after_same_scale(
            centers=mesh.cell_centers,
            div_before=div_before_limiter,
            div_after=div_after_limiter,
            out_path=run_dir / "divergence_before_after_limiter_same_scale.png",
        )
        top_before_cells = _top_divergence_cells(
            mesh=mesh,
            div_star=np.asarray(div_before["divergence"], dtype=np.float64),
            div_corr=div_before_limiter,
            face_flux_corrected=face_flux_star + correction_flux_raw_pre_limiter,
            masks=zone_masks,
            top_k=30,
        )
        top_after_cells = _top_divergence_cells(
            mesh=mesh,
            div_star=np.asarray(div_before["divergence"], dtype=np.float64),
            div_corr=div_after_limiter,
            face_flux_corrected=face_flux_star + correction_flux_limited_pre_bc,
            masks=zone_masks,
            top_k=30,
        )
        top_before_after_limiter_png = (
            _save_top_divergence_cells_before_after_limiter_xy(
                mesh=mesh,
                before_cells=list(top_before_cells.get("cells", [])),
                after_cells=list(top_after_cells.get("cells", [])),
                out_path=run_dir / "top_divergence_cells_before_after_limiter_xy.png",
            )
        )
        limited_face_indices = np.asarray(
            correction_limiter.get("limited_face_indices", []),
            dtype=np.int64,
        )
        correction_flux_limited_faces_png = _save_correction_flux_limited_faces_xy(
            mesh=mesh,
            correction_before=correction_flux_raw_pre_limiter,
            correction_after=correction_flux_limited_pre_bc,
            limited_face_indices=limited_face_indices,
            out_path=run_dir / "correction_flux_limited_faces_xy.png",
        )
        limiter_histogram_png = _save_limiter_effect_histogram(
            div_before=div_before_limiter,
            div_after=div_after_limiter,
            out_path=run_dir / "limiter_effect_histogram.png",
        )
        corr_hotspot_png = _save_correction_flux_hotspots_xy(
            mesh=mesh,
            correction_flux=correction_flux,
            out_path=run_dir / "correction_flux_hotspots_xy.png",
            max_faces=1500,
        )
        pressure_corr_flux_png = _save_pressure_correction_flux_magnitude_xy(
            mesh=mesh,
            pressure_gradient_flux=grad_flux_solution,
            out_path=run_dir / "pressure_correction_flux_magnitude_xy.png",
        )
        top_breakdown_png = _save_top_divergence_correction_breakdown_xy(
            mesh=mesh,
            breakdown=top_divergence_correction_breakdown,
            out_path=run_dir / "top_divergence_correction_breakdown_xy.png",
        )
        boundary_policy_bar_png = _save_boundary_policy_comparison_bar(
            comparison=boundary_policy_comparison,
            out_path=run_dir / "boundary_policy_comparison_bar.png",
        )
        final_step_idx = int(resume_start_step + flow_steps_requested)
        if final_step_idx in snapshot_steps:
            final_snap_dir = run_dir / "snapshots" / f"step_{final_step_idx:04d}"
            final_snap_dir.mkdir(parents=True, exist_ok=True)
            # Keep final-step velocity audit artifacts near the snapshot for easier inspection.
            _write_json(
                final_snap_dir / "velocity_reconstruction_audit.json",
                velocity_recon_audit,
            )
            _write_json(
                final_snap_dir / "velocity_region_audit.json", velocity_region_audit
            )
            _write_json(
                final_snap_dir / "top_suspicious_velocity_cells.json",
                top_suspicious_velocity_cells,
            )
            for src_name in (
                "velocity_vectors_xy_grid_binned.png",
                "velocity_vectors_xy_region_panels_binned.png",
                "velocity_vectors_xy_direction_colored.png",
                "velocity_magnitude_p95_clipped_xy.png",
            ):
                src_path = run_dir / src_name
                if src_path.exists():
                    (final_snap_dir / src_name).write_bytes(src_path.read_bytes())
        projection_operator_consistent = bool(
            operator_identity_audit.get("projection_operator_consistent", False)
        )
        projection_equation_consistent = bool(
            projection_equation_residual.get("projection_equation_consistent", False)
        )
        projection_equation_residual_consistent = bool(
            projection_equation_residual.get(
                "projection_equation_residual_consistent", False
            )
        )
        projection_equation_residual_reason = str(
            projection_equation_residual.get(
                "projection_equation_residual_reason",
                "projection equation residual audit not evaluated",
            )
        )
        startup_warning_steps_allowed = (
            int(args.startup_warning_steps)
            if int(args.startup_warning_steps) >= 0
            else int(max(0, args.allow_projection_warning_steps))
        )
        flow_progression_enabled = bool(flow_steps_requested > 1)
        flow_prog_acc = _evaluate_flow_progression_acceptance(
            history=progression_history,
            allow_projection_warning_steps=int(args.allow_projection_warning_steps),
            startup_warning_steps=int(startup_warning_steps_allowed),
            outlet_inlet_flux_ratio_tolerance=float(
                args.outlet_inlet_flux_ratio_tolerance
            ),
            wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
        )
        if (not run_physical_progression) and flow_progression_enabled:
            flow_prog_acc["reason"] = str(
                startup_root_cause_report.get("bootstrap_reason", "")
            )
            flow_prog_acc["bootstrap_blocked_progression"] = True
            flow_prog_acc["startup_bootstrap"] = dict(startup_root_cause_report)
        flow_progression_solved = bool(
            flow_prog_acc.get("flow_progression_solved", False)
        )
        flow_progression_reason = str(flow_prog_acc.get("reason", ""))
        flow_progression_final_metrics = dict(flow_prog_acc.get("final_step", {}))
        flow_progression_worst_step_metrics = dict(flow_prog_acc.get("worst_step", {}))
        conv_stats_main = _convective_history_stats(
            progression_history,
            convective_cfl_acceptance_eps=float(args.convective_cfl_acceptance_eps),
        )
        warning_agg = _collect_warning_aggregation(progression_history)
        boundary_flux_policy_audit = _collect_boundary_flux_policy(
            projection=dict(diag.get("projection", {})),
            flow_mode=str(flow_mode),
        )
        stabilization_audit = _collect_stabilization_audit(
            conv_stats=conv_stats_main,
            history=progression_history,
        )
        ns_coupling_audit = _evaluate_ns_coupling_readiness(
            flow_progression_solved=bool(flow_progression_solved),
            ready_for_long_ns_run_physical=bool(ready_for_long_ns_run_physical),
            nonphysical_flux_fix_used=bool(
                boundary_flux_policy_audit.get("nonphysical_flux_fix_used", False)
            ),
            convective_auto_damping_used_any=bool(
                stabilization_audit.get("convective_auto_damping_used_any", False)
            ),
            convective_substep_cap_hit_any=bool(
                stabilization_audit.get("convective_substep_cap_hit_any", False)
            ),
            finite_fields=bool(
                flow_progression_final_metrics.get("finite_fields", False)
            ),
            wall_flux_max_abs_after=float(
                flow_progression_final_metrics.get("wall_flux_max_abs_after", 0.0)
            ),
            outlet_inlet_flux_ratio=float(
                flow_progression_final_metrics.get("outlet_inlet_flux_ratio", 0.0)
            ),
            final_div_l2=float(
                flow_progression_final_metrics.get(
                    "final_div_l2",
                    flow_progression_final_metrics.get(
                        "final_divergence_l2", float("inf")
                    ),
                )
            ),
            final_div_max_abs=float(
                flow_progression_final_metrics.get(
                    "final_div_max_abs",
                    flow_progression_final_metrics.get(
                        "final_divergence_max_abs", float("inf")
                    ),
                )
            ),
            wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
            outlet_inlet_flux_ratio_tolerance=float(
                args.outlet_inlet_flux_ratio_tolerance
            ),
            projection_final_div_l2_tolerance=float(
                args.projection_final_div_l2_tolerance
            ),
            projection_final_div_max_tolerance=float(
                args.projection_final_div_max_tolerance
            ),
            epsilon_aware_warning_step_count=int(
                warning_agg.get("epsilon_aware_warning_step_count", 0)
            ),
        )
        flow_run_completed = bool(
            int(len(progression_history)) >= int(flow_steps_requested)
        )
        flow_numerically_stable = bool(
            flow_progression_solved
            and bool(flow_progression_final_metrics.get("finite_fields", False))
        )
        flow_physically_ready = bool(ready_for_long_ns_run_physical)
        flow_ready_for_next_stage = bool(
            ns_coupling_audit.get("ready_for_flow_to_transport_coupling", False)
        )
        flow_ready_for_long_run = bool(flow_physically_ready)
        flow_stage_status = _build_stage_status(
            run_completed=bool(flow_run_completed),
            numerically_stable=bool(flow_numerically_stable),
            physically_ready=bool(flow_physically_ready),
            ready_for_next_stage=bool(flow_ready_for_next_stage),
            ready_for_long_run=bool(flow_ready_for_long_run),
            checks={
                "flow_progression_solved": bool(flow_progression_solved),
                "finite_fields_final": bool(
                    flow_progression_final_metrics.get("finite_fields", False)
                ),
                "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
                "ready_for_flow_to_transport_coupling": bool(
                    ns_coupling_audit.get("ready_for_flow_to_transport_coupling", False)
                ),
                "startup_bootstrap_converged": bool(
                    startup_root_cause_report.get("bootstrap_converged", False)
                    or (not startup_root_cause_report.get("bootstrap_required", False))
                ),
            },
        )
        if (not run_physical_progression) and flow_progression_enabled:
            flow_stage_status["stage_status_reason"] = str(
                startup_root_cause_report.get("bootstrap_reason", "")
            )
        flow_solved = bool(flow_ready_for_next_stage)

        flow_diag = {
            "initial_divergence_stats": {
                "max_abs": float(div_before["divergence_max_abs"]),
                "l2": float(div_before["divergence_l2"]),
                "mean_abs": float(div_before["divergence_mean_abs"]),
            },
            "projection": diag.get("projection", {}),
            "pressure": diag.get("pressure", {}),
            "pressure_nonorthogonal_correction": diag.get(
                "pressure_nonorthogonal_correction", {}
            ),
            "velocity": diag.get("velocity", {}),
            "velocity_region_audit": velocity_region_audit,
            "velocity_reconstruction_audit_summary": {
                "direction_agreement_fraction_global": float(
                    velocity_recon_audit.get("direction_agreement_fraction_global", 0.0)
                ),
                "suspicious_velocity_cells_count": int(
                    velocity_recon_audit.get("suspicious_velocity_cells_count", 0)
                ),
            },
            "correction_limiter": correction_limiter,
            "correction_limiter_conservation_audit": correction_limiter_conservation_audit,
            "backend_execution": backend_exec,
            **_projection_boundary_contract_runtime_payload(diag),
            "face_flux_primary_note": "corrected face flux is the primary projection state",
            "projection_operator_consistent": bool(projection_operator_consistent),
            "projection_equation_consistent": bool(projection_equation_consistent),
            "projection_equation_residual_consistent": bool(
                projection_equation_residual_consistent
            ),
            "projection_equation_residual_reason": str(
                projection_equation_residual_reason
            ),
            "projection_equation_best_scaling": str(
                projection_equation_residual.get("projection_equation_best_scaling", "")
            ),
            "projection_equation_best_sign": str(
                projection_equation_residual.get("projection_equation_best_sign", "")
            ),
            "projection_equation_best_relative_l2": float(
                projection_equation_residual.get(
                    "projection_equation_best_relative_l2", 0.0
                )
            ),
            "flow_progression_enabled": bool(flow_progression_enabled),
            "startup_bootstrap": dict(startup_root_cause_report),
            "flow_solved": bool(flow_solved),
            "resume": dict(resume_metadata),
            "run_completed": bool(flow_stage_status.get("run_completed", False)),
            "numerically_stable": bool(
                flow_stage_status.get("numerically_stable", False)
            ),
            "physically_ready": bool(flow_stage_status.get("physically_ready", False)),
            "ready_for_next_stage": bool(
                flow_stage_status.get("ready_for_next_stage", False)
            ),
            "ready_for_long_run": bool(
                flow_stage_status.get("ready_for_long_run", False)
            ),
            "stage_status_reason": str(
                flow_stage_status.get("stage_status_reason", "")
            ),
            "stage_status_checks": dict(
                flow_stage_status.get("stage_status_checks", {})
            ),
            "flow_steps_requested": int(flow_steps_requested),
            "flow_steps_completed": int(len(progression_history)),
            "flow_start_step": int(resume_start_step),
            "flow_step_end": int(resume_start_step + len(progression_history)),
            "flow_steps_completed_total": int(
                resume_start_step + len(progression_history)
            ),
            "flow_stop_physical_time": flow_stop_physical_time,
            "snapshot_time_interval": snapshot_time_interval,
            "flow_dt": float(flow_dt),
            "flow_dt_mode": str(flow_dt_mode),
            "requested_flow_dt": float(requested_flow_dt),
            "flow_dt_min": float(flow_dt_min),
            "flow_dt_max": float(flow_dt_max),
            "convective_cfl_target": float(convective_cfl_target),
            "physical_time_initial": float(resume_start_time),
            "physical_time_advanced": float(physical_time - resume_start_time),
            "physical_time_final": float(physical_time),
            "used_dt_min": float(used_dt_min),
            "used_dt_mean": float(used_dt_mean),
            "used_dt_max": float(used_dt_max),
            "auto_dt_scale_min": float(auto_dt_scale_min),
            "auto_dt_scale_mean": float(auto_dt_scale_mean),
            "auto_dt_scale_max": float(auto_dt_scale_max),
            "auto_dt_floor_min": float(auto_dt_floor_min),
            "auto_dt_floor_mean": float(auto_dt_floor_mean),
            "auto_dt_floor_max": float(auto_dt_floor_max),
            "auto_dt_min_hit_any": bool(auto_dt_min_hit_any),
            "auto_dt_max_hit_any": bool(auto_dt_max_hit_any),
            "raw_cfl_after_dt_selection_max": float(raw_cfl_after_dt_selection_max),
            "raw_cfl_after_dt_selection_p95": float(raw_cfl_after_dt_selection_p95),
            "effective_cfl_limit_excess_max": float(
                conv_stats_main.get("effective_cfl_limit_excess_max", 0.0)
            ),
            "effective_cfl_warning_steps": list(
                conv_stats_main.get("effective_cfl_warning_steps", [])
            ),
            "raw_cfl_after_dt_selection_warning_steps": list(
                conv_stats_main.get("raw_cfl_after_dt_selection_warning_steps", [])
            ),
            "flow_mode": str(flow_mode),
            "flow_progression_solved": bool(flow_progression_solved),
            "flow_progression_solved_with_startup_tolerance": bool(
                flow_prog_acc.get(
                    "flow_progression_solved_with_startup_tolerance",
                    flow_progression_solved,
                )
            ),
            "startup_warning_steps_allowed": int(
                flow_prog_acc.get("startup_warning_steps_allowed", 0)
            ),
            "startup_warning_steps_observed": list(
                flow_prog_acc.get("startup_warning_steps_observed", [])
            ),
            "nonstartup_failed_steps": list(
                flow_prog_acc.get("nonstartup_failed_steps", [])
            ),
            "flow_progression_acceptance_reason": str(flow_progression_reason),
            "viscous_progression_accepted": bool(viscous_progression_accepted),
            "viscous_progression_acceptance_reason": str(
                viscous_progression_acceptance_reason
            ),
            "viscous_progression_checks": viscous_progression_checks,
            "flow_mode_comparison": flow_mode_comparison,
            "viscous_predictor_mode_comparison": viscous_predictor_mode_comparison,
            "ns_dt_mode_comparison": ns_dt_mode_comparison,
            "convective_stabilization_comparison": convective_stabilization_comparison,
            "viscous_predictor_audit_summary": _build_viscous_predictor_audit_from_history(
                progression_history
            ).get("summary", {}),
            "navier_stokes_prototype_audit_summary": navier_stokes_prototype_audit.get(
                "summary", {}
            ),
            "viscous_predictor_damages_divergence": bool(
                viscous_predictor_damages_divergence
            ),
            "viscous_predictor_damage_ratio_l2": float(
                viscous_predictor_damage_ratio_l2
            ),
            "viscous_predictor_damage_ratio_linf": float(
                viscous_predictor_damage_ratio_linf
            ),
            "viscous_predictor_best_mode": str(viscous_predictor_best_mode),
            "stokes_ready_for_advection_term": bool(stokes_ready_for_advection_term),
            "convective_predictor_used": bool(
                any(
                    bool(r.get("convective_predictor_used", False))
                    for r in progression_history
                )
            ),
            "convective_prototype_accepted": bool(convective_prototype_accepted),
            "convective_prototype_acceptance_reason": str(
                convective_prototype_acceptance_reason
            ),
            "convective_prototype_checks": convective_prototype_checks,
            "convective_readiness_checks": convective_readiness_checks,
            "readiness_reason": str(convective_readiness_reason),
            "ready_for_long_ns_run_debug": bool(ready_for_long_ns_run_debug),
            "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
            "ready_for_long_ns_run": bool(ready_for_long_ns_run),
            "ns_auto_dt_accepted": bool(ns_auto_dt_accepted),
            "ns_auto_dt_warning": str(ns_auto_dt_warning),
            "auto_cfl_no_damping_readiness_reason": str(
                auto_cfl_no_damping_readiness_reason
            ),
            **warning_agg,
            **boundary_flux_policy_audit,
            **stabilization_audit,
            **ns_coupling_audit,
            "stokes_baseline_accepted": bool(stokes_baseline_accepted),
            "stokes_baseline_acceptance_reason": str(stokes_baseline_acceptance_reason),
            "recommended_stokes_baseline_config": recommended_stokes_baseline_config,
            "ready_for_convective_prototype": bool(ready_for_convective_prototype),
            "stokes_baseline_sensitivity_sweep": stokes_baseline_sensitivity_sweep,
            "audit_pressure_operator_enabled": bool(args.audit_pressure_operator),
            "pressure_matrix_explicit_audit": pressure_matrix_explicit_audit,
            "pressure_operator_matrixfree_vs_explicit": pressure_operator_matrixfree_vs_explicit,
            "pressure_operator_spd_audit": pressure_operator_spd_audit,
            "pressure_reference_solver_comparison": pressure_reference_solver_comparison,
        }
        _write_json(run_dir / "flow_diagnostics.json", flow_diag)
        projection = dict(diag.get("projection", {}))
        pressure = dict(diag.get("pressure", {}))
        acc_main = _projection_acceptance(
            projection=projection,
            pressure=pressure,
            backend_execution=backend_exec,
            thresholds=acceptance_thresholds_cli,
        )
        pressure_linear_solved_strict = bool(
            acc_main.get("pressure_linear_solved_strict", False)
        )
        pressure_linear_accepted = bool(acc_main.get("pressure_linear_accepted", False))
        projection_accepted_physical = bool(acc_main.get("projection_accepted", False))
        projection_solved_physical = bool(acc_main.get("projection_solved", False))
        projection_accepted = bool(projection_accepted_physical)
        projection_solved = bool(projection_solved_physical)
        projection_reason = str(acc_main.get("reason", ""))
        projection_thresholds = dict(acc_main.get("thresholds", {}))
        projection_acceptance_metrics = dict(acc_main.get("metrics", {}))
        projection_checklist = dict(acc_main.get("checklist", {}))
        projection_limit_mode = str(
            diag.get(
                "projection_correction_limit_mode", cfg.projection_correction_limit_mode
            )
        )
        projection_limit_experimental = bool(
            diag.get("projection_limit_experimental", projection_limit_mode != "none")
        )
        if projection_solved and projection_limit_experimental:
            projection_reason = "projection passes metrics but uses experimental correction limiter mode"
        if projection_limit_experimental:
            projection_solved = False
        compare_modes = _parse_flow_modes(str(args.compare_flow_modes))
        if compare_modes:
            cmp_items: list[dict[str, Any]] = []
            mode_histories: dict[str, list[dict[str, Any]]] = {}
            for mode_name in compare_modes:
                mode_cfg = progression_cfg
                if (
                    mode_name
                    in {
                        "stokes_viscous_projection",
                        "navier_stokes_projection_debug",
                    }
                    and (not viscous_predictor_explicit)
                    and str(mode_cfg.viscous_predictor_mode)
                    != "face_flux_laplacian_substepped"
                ):
                    mode_cfg = replace(
                        mode_cfg,
                        viscous_predictor_mode="face_flux_laplacian_substepped",
                    )
                s_cmp = state0
                hist_cmp: list[dict[str, Any]] = []
                for step_idx in range(1, flow_steps_requested + 1):
                    conv_diag_cmp: dict[str, Any] = {
                        "convective_predictor_used": False,
                        "convective_cfl_limit": float(mode_cfg.convective_cfl_limit),
                        "convective_cfl_raw_max": 0.0,
                        "convective_cfl_raw_p95": 0.0,
                        "convective_cfl_effective_max": 0.0,
                        "convective_cfl_effective_p95": 0.0,
                        "convective_cfl_warning_raw": False,
                        "convective_cfl_warning_effective": False,
                        "convective_predictor_damping_requested": float(
                            mode_cfg.convective_predictor_damping
                        ),
                        "convective_predictor_damping_effective": 0.0,
                        "convective_auto_damping_used": False,
                        "convective_auto_damping_reason": "convective predictor disabled",
                        "convective_dt_effective": 0.0,
                        "convective_delta_velocity_max": 0.0,
                        "convective_delta_velocity_l2": 0.0,
                        "kinetic_energy_before_convection": 0.0,
                        "kinetic_energy_after_convection": 0.0,
                    }
                    visc_diag_cmp: dict[str, Any] = {
                        "viscous_predictor_used": False,
                        "viscous_predictor_mode": "none",
                        "kinematic_viscosity": float(
                            progression_cfg.kinematic_viscosity
                        ),
                        "viscous_dt": float(flow_dt),
                        "viscous_stability_metric": 0.0,
                        "viscous_stability_warning": False,
                        "viscous_delta_velocity_max": 0.0,
                        "viscous_delta_velocity_l2": 0.0,
                        "kinetic_energy_before_predictor": 0.0,
                        "kinetic_energy_after_predictor": 0.0,
                    }
                    s_after_conv_cmp = s_cmp
                    if mode_name == "navier_stokes_projection_debug":
                        s_conv_cmp = apply_tetra_convective_predictor(
                            mesh,
                            s_cmp,
                            mode_cfg,
                            flow_dt=float(flow_dt),
                        )
                        conv_diag_cmp = dict(
                            s_conv_cmp.diagnostics.get(
                                "convective_predictor", conv_diag_cmp
                            )
                        )
                        s_after_conv_cmp = s_conv_cmp
                    s_for_cmp = s_after_conv_cmp
                    if mode_name in {
                        "stokes_viscous_projection",
                        "navier_stokes_projection_debug",
                    }:
                        s_pred_cmp = apply_tetra_stokes_viscous_predictor(
                            mesh,
                            s_after_conv_cmp,
                            mode_cfg,
                            flow_dt=float(flow_dt),
                        )
                        visc_diag_cmp = dict(
                            s_pred_cmp.diagnostics.get(
                                "viscous_predictor", visc_diag_cmp
                            )
                        )
                        s_for_cmp = s_pred_cmp
                    s_next_cmp = solve_tetra_pressure_projection(
                        mesh, s_for_cmp, mode_cfg
                    )
                    s_next_cmp.diagnostics["convective_predictor"] = dict(conv_diag_cmp)
                    s_next_cmp.diagnostics["viscous_predictor"] = dict(visc_diag_cmp)
                    d_cmp = dict(s_next_cmp.diagnostics)
                    p_cmp = dict(d_cmp.get("projection", {}))
                    pr_cmp = dict(d_cmp.get("pressure", {}))
                    be_cmp = dict(d_cmp.get("backend_execution", {}))
                    v_cmp = dict(d_cmp.get("velocity", {}))
                    conv_out_cmp = dict(d_cmp.get("convective_predictor", {}))
                    acc_cmp = _projection_acceptance(
                        projection=p_cmp,
                        pressure=pr_cmp,
                        backend_execution=be_cmp,
                        thresholds=acceptance_thresholds_cli,
                    )
                    vel_cmp = np.asarray(s_next_cmp.cell_velocity, dtype=np.float64)
                    speed_cmp = np.linalg.norm(vel_cmp, axis=1)
                    kin_after_proj_cmp = (
                        float(np.mean(0.5 * speed_cmp * speed_cmp))
                        if speed_cmp.size
                        else 0.0
                    )
                    hist_cmp.append(
                        {
                            "step": int(step_idx),
                            "projection_solved": bool(
                                acc_cmp.get("projection_solved", False)
                            ),
                            "finite_fields": bool(
                                acc_cmp.get("checklist", {}).get("finite_fields", False)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                float(p_cmp.get("outlet_flux_total_after", 0.0))
                                / max(
                                    abs(
                                        float(p_cmp.get("inlet_flux_total_after", 0.0))
                                    ),
                                    1e-30,
                                )
                            ),
                            "wall_flux_max_abs_after": float(
                                p_cmp.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "initial_divergence_max_abs": float(
                                p_cmp.get("initial_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_max_abs": float(
                                p_cmp.get("final_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_l2": float(
                                p_cmp.get("final_divergence_l2", 0.0)
                            ),
                            "net_boundary_flux_after": float(
                                p_cmp.get("net_boundary_flux_after", 0.0)
                            ),
                            "velocity_magnitude_max": float(
                                v_cmp.get("magnitude_max", 0.0)
                            ),
                            "velocity_magnitude_mean": float(
                                v_cmp.get("magnitude_mean", 0.0)
                            ),
                            "velocity_magnitude_p95": float(
                                np.percentile(speed_cmp, 95.0)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "kinetic_energy_after_projection": float(
                                kin_after_proj_cmp
                            ),
                            "pressure_iterations": int(
                                pr_cmp.get("actual_iterations", 0)
                            ),
                            "convective_predictor_used": bool(
                                conv_out_cmp.get("convective_predictor_used", False)
                            ),
                            "convective_cfl_limit": float(
                                conv_out_cmp.get(
                                    "convective_cfl_limit",
                                    mode_cfg.convective_cfl_limit,
                                )
                            ),
                            "convective_cfl_raw_max": float(
                                conv_out_cmp.get(
                                    "convective_cfl_raw_max",
                                    conv_out_cmp.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_raw_p95": float(
                                conv_out_cmp.get(
                                    "convective_cfl_raw_p95",
                                    conv_out_cmp.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_effective_max": float(
                                conv_out_cmp.get(
                                    "convective_cfl_effective_max",
                                    conv_out_cmp.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_effective_p95": float(
                                conv_out_cmp.get(
                                    "convective_cfl_effective_p95",
                                    conv_out_cmp.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_warning_raw": bool(
                                conv_out_cmp.get(
                                    "convective_cfl_warning_raw",
                                    conv_out_cmp.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_cfl_warning_effective": bool(
                                conv_out_cmp.get(
                                    "convective_cfl_warning_effective",
                                    conv_out_cmp.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_predictor_damping_requested": float(
                                conv_out_cmp.get(
                                    "convective_predictor_damping_requested",
                                    conv_out_cmp.get(
                                        "convective_predictor_damping", 0.0
                                    ),
                                )
                            ),
                            "convective_predictor_damping_effective": float(
                                conv_out_cmp.get(
                                    "convective_predictor_damping_effective", 0.0
                                )
                            ),
                            "convective_predictor_cfl_scale": float(
                                conv_out_cmp.get("convective_predictor_cfl_scale", 1.0)
                            ),
                            "convective_auto_damping_used": bool(
                                conv_out_cmp.get("convective_auto_damping_used", False)
                            ),
                            "convective_auto_damping_reason": str(
                                conv_out_cmp.get("convective_auto_damping_reason", "")
                            ),
                            "convective_stabilization_mode": str(
                                conv_out_cmp.get(
                                    "convective_stabilization_mode",
                                    mode_cfg.convective_stabilization_mode,
                                )
                            ),
                            "convective_substep_boundary_contract_mode": str(
                                conv_out_cmp.get(
                                    "convective_substep_boundary_contract_mode",
                                    mode_cfg.convective_substep_boundary_contract,
                                )
                            ),
                            "convective_substepping_used": bool(
                                conv_out_cmp.get("convective_substepping_used", False)
                            ),
                            "convective_substep_count": int(
                                conv_out_cmp.get("convective_substep_count", 1)
                            ),
                            "convective_substep_count_unclamped": int(
                                conv_out_cmp.get(
                                    "convective_substep_count_unclamped", 1
                                )
                            ),
                            "convective_substep_cap_hit": bool(
                                conv_out_cmp.get("convective_substep_cap_hit", False)
                            ),
                            "convective_substep_dt": float(
                                conv_out_cmp.get("convective_substep_dt", flow_dt)
                            ),
                            "convective_cfl_per_substep_max": float(
                                conv_out_cmp.get("convective_cfl_per_substep_max", 0.0)
                            ),
                            "convective_cfl_per_substep_p95": float(
                                conv_out_cmp.get("convective_cfl_per_substep_p95", 0.0)
                            ),
                            "convective_dt_effective": float(
                                conv_out_cmp.get("convective_dt_effective", 0.0)
                            ),
                            "convective_cfl_max": float(
                                conv_out_cmp.get(
                                    "convective_cfl_raw_max",
                                    conv_out_cmp.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_p95": float(
                                conv_out_cmp.get(
                                    "convective_cfl_raw_p95",
                                    conv_out_cmp.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_delta_velocity_max": float(
                                conv_out_cmp.get("convective_delta_velocity_max", 0.0)
                            ),
                            "convective_delta_velocity_l2": float(
                                conv_out_cmp.get("convective_delta_velocity_l2", 0.0)
                            ),
                            "kinetic_energy_before_convection": float(
                                conv_out_cmp.get(
                                    "kinetic_energy_before_convection", 0.0
                                )
                            ),
                            "kinetic_energy_after_convection": float(
                                conv_out_cmp.get("kinetic_energy_after_convection", 0.0)
                            ),
                            "divergence_after_convection_before_projection_max": float(
                                conv_out_cmp.get(
                                    "divergence_after_convection_before_projection_max",
                                    0.0,
                                )
                            ),
                            "divergence_after_convection_before_projection_l2": float(
                                conv_out_cmp.get(
                                    "divergence_after_convection_before_projection_l2",
                                    0.0,
                                )
                            ),
                            "capped_predictor_updates_count": int(
                                visc_diag_cmp.get("capped_predictor_updates_count", 0)
                            ),
                            "total_predictor_updates_count": int(
                                visc_diag_cmp.get("total_predictor_updates_count", 0)
                            ),
                            "capped_predictor_updates_fraction": float(
                                visc_diag_cmp.get(
                                    "capped_predictor_updates_fraction", 0.0
                                )
                            ),
                        }
                    )
                    s_cmp = s_next_cmp
                acc_hist_cmp = _evaluate_flow_progression_acceptance(
                    history=hist_cmp,
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                final_cmp = hist_cmp[-1] if hist_cmp else {}
                iters = [int(r.get("pressure_iterations", 0)) for r in hist_cmp]
                conv_stats_cmp = _convective_history_stats(
                    hist_cmp,
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                )
                failed_steps = [
                    int(r.get("step", -1))
                    for r in hist_cmp
                    if not bool(r.get("projection_solved", False))
                ]
                cmp_items.append(
                    {
                        "flow_mode": str(mode_name),
                        "final_divergence_max_abs": float(
                            final_cmp.get("final_divergence_max_abs", 0.0)
                        ),
                        "final_divergence_l2": float(
                            final_cmp.get("final_divergence_l2", 0.0)
                        ),
                        "outlet_inlet_flux_ratio": float(
                            final_cmp.get("outlet_inlet_flux_ratio", 0.0)
                        ),
                        "net_boundary_flux_relative": float(
                            final_cmp.get("net_boundary_flux_relative", 0.0)
                        ),
                        "net_boundary_flux_after": float(
                            final_cmp.get("net_boundary_flux_after", 0.0)
                        ),
                        "wall_flux": float(
                            final_cmp.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "wall_flux_max_abs_after": float(
                            final_cmp.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "velocity_magnitude_max_final": float(
                            final_cmp.get("velocity_magnitude_max", 0.0)
                        ),
                        "velocity_magnitude_mean_final": float(
                            final_cmp.get("velocity_magnitude_mean", 0.0)
                        ),
                        "velocity_magnitude_p95_final": float(
                            final_cmp.get("velocity_magnitude_p95", 0.0)
                        ),
                        "kinetic_energy_final": float(
                            final_cmp.get("kinetic_energy_after_projection", 0.0)
                        ),
                        "convective_predictor_used": bool(
                            final_cmp.get("convective_predictor_used", False)
                        ),
                        "raw_cfl_max": float(
                            conv_stats_cmp.get("raw_cfl_max_max", 0.0)
                        ),
                        "raw_cfl_p95": float(
                            conv_stats_cmp.get("raw_cfl_p95_max", 0.0)
                        ),
                        "effective_cfl_max": float(
                            conv_stats_cmp.get("effective_cfl_max_max", 0.0)
                        ),
                        "effective_cfl_p95": float(
                            conv_stats_cmp.get("effective_cfl_p95_max", 0.0)
                        ),
                        "convective_cfl_max_mean": float(
                            conv_stats_cmp.get("raw_cfl_max_mean", 0.0)
                        ),
                        "convective_cfl_warning_raw_any": bool(
                            conv_stats_cmp.get("raw_cfl_warning_any", False)
                        ),
                        "convective_cfl_warning_effective_any": bool(
                            conv_stats_cmp.get("effective_cfl_warning_any", False)
                        ),
                        "auto_damping_used_any": bool(
                            conv_stats_cmp.get("auto_damping_used_any", False)
                        ),
                        "auto_damping_step_count": int(
                            conv_stats_cmp.get("auto_damping_step_count", 0)
                        ),
                        "auto_damping_effective_min": float(
                            conv_stats_cmp.get("damping_effective_min", 0.0)
                        ),
                        "auto_damping_effective_mean": float(
                            conv_stats_cmp.get("damping_effective_mean", 0.0)
                        ),
                        "auto_damping_effective_max": float(
                            conv_stats_cmp.get("damping_effective_max", 0.0)
                        ),
                        "convective_delta_velocity_max_final": float(
                            final_cmp.get("convective_delta_velocity_max", 0.0)
                        ),
                        "convective_delta_velocity_l2_final": float(
                            final_cmp.get("convective_delta_velocity_l2", 0.0)
                        ),
                        "convective_stabilization_mode": str(
                            final_cmp.get(
                                "convective_stabilization_mode", "auto_damping"
                            )
                        ),
                        "convective_substep_boundary_contract_mode": str(
                            final_cmp.get(
                                "convective_substep_boundary_contract_mode", "end_only"
                            )
                        ),
                        "convective_substepping_used_any": bool(
                            conv_stats_cmp.get("substepping_used_any", False)
                        ),
                        "convective_substep_count_mean": float(
                            conv_stats_cmp.get("substep_count_mean", 0.0)
                        ),
                        "convective_substep_count_max": int(
                            conv_stats_cmp.get("substep_count_max", 0)
                        ),
                        "convective_substep_cap_hit_any": bool(
                            conv_stats_cmp.get("substep_cap_hit_any", False)
                        ),
                        "pressure_iterations_mean": float(np.mean(iters))
                        if iters
                        else 0.0,
                        "pressure_iterations_max": int(np.max(iters)) if iters else 0,
                        "divergence_after_convection_before_projection_max_final": float(
                            final_cmp.get(
                                "divergence_after_convection_before_projection_max", 0.0
                            )
                        ),
                        "divergence_after_convection_before_projection_l2_final": float(
                            final_cmp.get(
                                "divergence_after_convection_before_projection_l2", 0.0
                            )
                        ),
                        "kinetic_energy_before_convection_final": float(
                            final_cmp.get("kinetic_energy_before_convection", 0.0)
                        ),
                        "kinetic_energy_after_convection_final": float(
                            final_cmp.get("kinetic_energy_after_convection", 0.0)
                        ),
                        "projection_solved_all": bool(len(failed_steps) == 0),
                        "failed_steps": failed_steps,
                        "startup_warning_steps": list(
                            acc_hist_cmp.get("startup_warning_steps_observed", [])
                        ),
                        "flow_progression_solved": bool(
                            acc_hist_cmp.get("flow_progression_solved", False)
                        ),
                        "convective_prototype_accepted": False,
                        "ready_for_long_ns_run_debug": False,
                        "ready_for_long_ns_run_physical": False,
                    }
                )
                mode_histories[str(mode_name)] = hist_cmp
            flow_mode_comparison = {"items": cmp_items}
            _write_json(run_dir / "flow_mode_comparison.json", flow_mode_comparison)
            mode_map = {str(it.get("flow_mode", "")): it for it in cmp_items}
            if "navier_stokes_projection_debug" in mode_histories:
                stokes_ref = mode_map.get("stokes_viscous_projection", None)
                conv_acc = _evaluate_convective_prototype_acceptance(
                    history=mode_histories.get("navier_stokes_projection_debug", []),
                    stokes_baseline=stokes_ref,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    inlet_speed=float(args.inlet_speed),
                )
                convective_prototype_accepted = bool(
                    conv_acc.get("convective_prototype_accepted", False)
                )
                convective_prototype_acceptance_reason = str(conv_acc.get("reason", ""))
                convective_prototype_checks = dict(conv_acc.get("checks", {}))
                readiness = _evaluate_convective_readiness(
                    convective_prototype_accepted=bool(convective_prototype_accepted),
                    convective_prototype_acceptance_reason=str(
                        convective_prototype_acceptance_reason
                    ),
                    history=mode_histories.get("navier_stokes_projection_debug", []),
                    stokes_baseline=stokes_ref,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    max_convective_substeps=int(args.max_convective_substeps),
                    flow_dt_mode=str(args.flow_dt_mode),
                    convective_cfl_target=float(args.convective_cfl_target),
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                    inlet_speed=float(args.inlet_speed),
                )
                ready_for_long_ns_run_debug = bool(
                    readiness.get("ready_for_long_ns_run_debug", False)
                )
                ready_for_long_ns_run_physical = bool(
                    readiness.get("ready_for_long_ns_run_physical", False)
                )
                ready_for_long_ns_run = bool(ready_for_long_ns_run_physical)
                convective_readiness_reason = str(
                    readiness.get("readiness_reason", "not evaluated")
                )
                convective_readiness_checks = dict(
                    readiness.get("convective_readiness_checks", {})
                )
                ns_item = mode_map.get("navier_stokes_projection_debug", None)
                if ns_item is not None:
                    ns_item["convective_prototype_accepted"] = bool(
                        convective_prototype_accepted
                    )
                    ns_item["ready_for_long_ns_run_debug"] = bool(
                        ready_for_long_ns_run_debug
                    )
                    ns_item["ready_for_long_ns_run_physical"] = bool(
                        ready_for_long_ns_run_physical
                    )
                    ns_item["readiness_reason"] = str(convective_readiness_reason)
            st_item = mode_map.get("stokes_viscous_projection", None)
            if st_item is not None:
                st_acc, st_reason = _evaluate_stokes_baseline_acceptance(
                    final_div_l2=float(
                        st_item.get("final_divergence_l2", float("inf"))
                    ),
                    final_div_max=float(
                        st_item.get("final_divergence_max_abs", float("inf"))
                    ),
                    outlet_inlet_ratio=float(
                        st_item.get("outlet_inlet_flux_ratio", float("inf"))
                    ),
                    net_boundary_flux_relative=float(
                        st_item.get("net_boundary_flux_relative", float("inf"))
                    ),
                    wall_flux_max_abs=float(st_item.get("wall_flux", float("inf"))),
                    damage_ratio_l2=1.0,
                    damage_ratio_linf=1.0,
                    finite=bool(st_item.get("flow_progression_solved", False)),
                )
                st_item["stokes_baseline_accepted"] = bool(st_acc)
                st_item["stokes_baseline_acceptance_reason"] = str(st_reason)
            if flow_mode == "stokes_viscous_projection":
                visc_acc = _evaluate_viscous_progression_acceptance(
                    history=progression_history,
                    projection_only_baseline=mode_map.get("projection_only", None),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    projection_final_div_l2_tolerance=float(
                        args.projection_final_div_l2_tolerance
                    ),
                    projection_final_div_max_tolerance=float(
                        args.projection_final_div_max_tolerance
                    ),
                )
                viscous_progression_accepted = bool(
                    visc_acc.get("viscous_progression_accepted", False)
                )
                viscous_progression_acceptance_reason = str(visc_acc.get("reason", ""))
                viscous_progression_checks = dict(visc_acc.get("checks", {}))
            if flow_mode == "navier_stokes_projection_debug":
                conv_acc = _evaluate_convective_prototype_acceptance(
                    history=progression_history,
                    stokes_baseline=mode_map.get("stokes_viscous_projection", None),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    inlet_speed=float(args.inlet_speed),
                )
                convective_prototype_accepted = bool(
                    conv_acc.get("convective_prototype_accepted", False)
                )
                convective_prototype_acceptance_reason = str(conv_acc.get("reason", ""))
                convective_prototype_checks = dict(conv_acc.get("checks", {}))
                readiness = _evaluate_convective_readiness(
                    convective_prototype_accepted=bool(convective_prototype_accepted),
                    convective_prototype_acceptance_reason=str(
                        convective_prototype_acceptance_reason
                    ),
                    history=progression_history,
                    stokes_baseline=mode_map.get("stokes_viscous_projection", None),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    max_convective_substeps=int(args.max_convective_substeps),
                    flow_dt_mode=str(args.flow_dt_mode),
                    convective_cfl_target=float(args.convective_cfl_target),
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                    inlet_speed=float(args.inlet_speed),
                )
                ready_for_long_ns_run_debug = bool(
                    readiness.get("ready_for_long_ns_run_debug", False)
                )
                ready_for_long_ns_run_physical = bool(
                    readiness.get("ready_for_long_ns_run_physical", False)
                )
                ready_for_long_ns_run = bool(ready_for_long_ns_run_physical)
                convective_readiness_reason = str(
                    readiness.get("readiness_reason", "not evaluated")
                )
                convective_readiness_checks = dict(
                    readiness.get("convective_readiness_checks", {})
                )
            _write_json(run_dir / "flow_mode_comparison.json", flow_mode_comparison)
        elif flow_mode == "stokes_viscous_projection":
            visc_acc = _evaluate_viscous_progression_acceptance(
                history=progression_history,
                projection_only_baseline=None,
                outlet_inlet_flux_ratio_tolerance=float(
                    args.outlet_inlet_flux_ratio_tolerance
                ),
                wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                projection_final_div_l2_tolerance=float(
                    args.projection_final_div_l2_tolerance
                ),
                projection_final_div_max_tolerance=float(
                    args.projection_final_div_max_tolerance
                ),
            )
            viscous_progression_accepted = bool(
                visc_acc.get("viscous_progression_accepted", False)
            )
            viscous_progression_acceptance_reason = str(visc_acc.get("reason", ""))
            viscous_progression_checks = dict(visc_acc.get("checks", {}))
        elif flow_mode == "navier_stokes_projection_debug":
            conv_acc = _evaluate_convective_prototype_acceptance(
                history=progression_history,
                stokes_baseline=None,
                outlet_inlet_flux_ratio_tolerance=float(
                    args.outlet_inlet_flux_ratio_tolerance
                ),
                wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                inlet_speed=float(args.inlet_speed),
            )
            convective_prototype_accepted = bool(
                conv_acc.get("convective_prototype_accepted", False)
            )
            convective_prototype_acceptance_reason = str(conv_acc.get("reason", ""))
            convective_prototype_checks = dict(conv_acc.get("checks", {}))
            readiness = _evaluate_convective_readiness(
                convective_prototype_accepted=bool(convective_prototype_accepted),
                convective_prototype_acceptance_reason=str(
                    convective_prototype_acceptance_reason
                ),
                history=progression_history,
                stokes_baseline=None,
                outlet_inlet_flux_ratio_tolerance=float(
                    args.outlet_inlet_flux_ratio_tolerance
                ),
                wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                max_convective_substeps=int(args.max_convective_substeps),
                flow_dt_mode=str(args.flow_dt_mode),
                convective_cfl_target=float(args.convective_cfl_target),
                convective_cfl_acceptance_eps=float(args.convective_cfl_acceptance_eps),
                inlet_speed=float(args.inlet_speed),
            )
            ready_for_long_ns_run_debug = bool(
                readiness.get("ready_for_long_ns_run_debug", False)
            )
            ready_for_long_ns_run_physical = bool(
                readiness.get("ready_for_long_ns_run_physical", False)
            )
            ready_for_long_ns_run = bool(ready_for_long_ns_run_physical)
            convective_readiness_reason = str(
                readiness.get("readiness_reason", "not evaluated")
            )
            convective_readiness_checks = dict(
                readiness.get("convective_readiness_checks", {})
            )
        compare_conv_stab_modes = _parse_convective_stabilization_modes(
            str(args.compare_convective_stabilization_modes)
        )
        if compare_conv_stab_modes:
            stokes_ref_for_readiness: dict[str, Any] | None = None
            if isinstance(flow_mode_comparison, dict):
                for _it in flow_mode_comparison.get("items", []):
                    if str(_it.get("flow_mode", "")) == "stokes_viscous_projection":
                        stokes_ref_for_readiness = dict(_it)
                        break
            scenario_cfgs: list[tuple[str, TetraFlowConfig]] = []
            for stab_mode in compare_conv_stab_modes:
                if stab_mode == "auto_damping":
                    scenario_cfgs.append(
                        (
                            "auto_damping",
                            replace(
                                progression_cfg,
                                convective_stabilization_mode="auto_damping",
                                convective_substep_boundary_contract="end_only",
                                enable_convective_predictor=True,
                                disable_convective_predictor=False,
                            ),
                        )
                    )
                elif stab_mode == "substepping":
                    scenario_cfgs.append(
                        (
                            "substepping_end_only",
                            replace(
                                progression_cfg,
                                convective_stabilization_mode="substepping",
                                convective_substep_boundary_contract="end_only",
                                enable_convective_predictor=True,
                                disable_convective_predictor=False,
                            ),
                        )
                    )
                    scenario_cfgs.append(
                        (
                            "substepping_every_substep",
                            replace(
                                progression_cfg,
                                convective_stabilization_mode="substepping",
                                convective_substep_boundary_contract="every_substep",
                                enable_convective_predictor=True,
                                disable_convective_predictor=False,
                            ),
                        )
                    )
            comp_items: list[dict[str, Any]] = []
            for scenario_name, cmp_cfg in scenario_cfgs:
                s_cmp = state0
                hist_cmp: list[dict[str, Any]] = []
                t_cmp0 = perf_counter()
                for step_idx in range(1, flow_steps_requested + 1):
                    s_conv = apply_tetra_convective_predictor(
                        mesh,
                        s_cmp,
                        cmp_cfg,
                        flow_dt=float(flow_dt),
                    )
                    conv_diag = dict(s_conv.diagnostics.get("convective_predictor", {}))
                    s_pred = apply_tetra_stokes_viscous_predictor(
                        mesh,
                        s_conv,
                        cmp_cfg,
                        flow_dt=float(flow_dt),
                    )
                    s_next = solve_tetra_pressure_projection(mesh, s_pred, cmp_cfg)
                    d_cmp = dict(s_next.diagnostics)
                    p_cmp = dict(d_cmp.get("projection", {}))
                    v_cmp = dict(d_cmp.get("velocity", {}))
                    pr_cmp = dict(d_cmp.get("pressure", {}))
                    be_cmp = dict(d_cmp.get("backend_execution", {}))
                    acc_cmp = _projection_acceptance(
                        projection=p_cmp,
                        pressure=pr_cmp,
                        backend_execution=be_cmp,
                        thresholds=acceptance_thresholds_cli,
                    )
                    vel_cmp = np.asarray(s_next.cell_velocity, dtype=np.float64)
                    speed_cmp = np.linalg.norm(vel_cmp, axis=1)
                    hist_cmp.append(
                        {
                            "step": int(step_idx),
                            "projection_solved": bool(
                                acc_cmp.get("projection_solved", False)
                            ),
                            "finite_fields": bool(
                                acc_cmp.get("checklist", {}).get("finite_fields", False)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                float(p_cmp.get("outlet_flux_total_after", 0.0))
                                / max(
                                    abs(
                                        float(p_cmp.get("inlet_flux_total_after", 0.0))
                                    ),
                                    1e-30,
                                )
                            ),
                            "net_boundary_flux_relative": float(
                                abs(float(p_cmp.get("net_boundary_flux_after", 0.0)))
                                / max(
                                    abs(
                                        float(p_cmp.get("inlet_flux_total_after", 0.0))
                                    ),
                                    1e-30,
                                )
                            ),
                            "wall_flux_max_abs_after": float(
                                p_cmp.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "final_divergence_max_abs": float(
                                p_cmp.get("final_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_l2": float(
                                p_cmp.get("final_divergence_l2", 0.0)
                            ),
                            "velocity_magnitude_max": float(
                                v_cmp.get("magnitude_max", 0.0)
                            ),
                            "velocity_magnitude_mean": float(
                                v_cmp.get("magnitude_mean", 0.0)
                            ),
                            "velocity_magnitude_p95": float(
                                np.percentile(speed_cmp, 95.0)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "kinetic_energy_after_projection": float(
                                np.mean(0.5 * speed_cmp * speed_cmp)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "pressure_iterations": int(
                                pr_cmp.get("actual_iterations", 0)
                            ),
                            "convective_predictor_used": bool(
                                conv_diag.get("convective_predictor_used", False)
                            ),
                            "convective_stabilization_mode": str(
                                conv_diag.get(
                                    "convective_stabilization_mode",
                                    cmp_cfg.convective_stabilization_mode,
                                )
                            ),
                            "convective_substep_boundary_contract_mode": str(
                                conv_diag.get(
                                    "convective_substep_boundary_contract_mode",
                                    cmp_cfg.convective_substep_boundary_contract,
                                )
                            ),
                            "convective_cfl_limit": float(
                                conv_diag.get(
                                    "convective_cfl_limit", cmp_cfg.convective_cfl_limit
                                )
                            ),
                            "convective_cfl_raw_max": float(
                                conv_diag.get(
                                    "convective_cfl_raw_max",
                                    conv_diag.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_raw_p95": float(
                                conv_diag.get(
                                    "convective_cfl_raw_p95",
                                    conv_diag.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_effective_max": float(
                                conv_diag.get(
                                    "convective_cfl_effective_max",
                                    conv_diag.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_effective_p95": float(
                                conv_diag.get(
                                    "convective_cfl_effective_p95",
                                    conv_diag.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_warning_raw": bool(
                                conv_diag.get(
                                    "convective_cfl_warning_raw",
                                    conv_diag.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_cfl_warning_effective": bool(
                                conv_diag.get(
                                    "convective_cfl_warning_effective",
                                    conv_diag.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_predictor_damping_effective": float(
                                conv_diag.get(
                                    "convective_predictor_damping_effective", 0.0
                                )
                            ),
                            "convective_auto_damping_used": bool(
                                conv_diag.get("convective_auto_damping_used", False)
                            ),
                            "convective_substepping_used": bool(
                                conv_diag.get("convective_substepping_used", False)
                            ),
                            "convective_substep_count": int(
                                conv_diag.get("convective_substep_count", 1)
                            ),
                            "convective_substep_cap_hit": bool(
                                conv_diag.get("convective_substep_cap_hit", False)
                            ),
                            "convective_cfl_per_substep_max": float(
                                conv_diag.get("convective_cfl_per_substep_max", 0.0)
                            ),
                            "convective_cfl_per_substep_p95": float(
                                conv_diag.get("convective_cfl_per_substep_p95", 0.0)
                            ),
                            "convective_delta_velocity_max": float(
                                conv_diag.get("convective_delta_velocity_max", 0.0)
                            ),
                            "convective_delta_velocity_l2": float(
                                conv_diag.get("convective_delta_velocity_l2", 0.0)
                            ),
                            "kinetic_energy_before_convection": float(
                                conv_diag.get("kinetic_energy_before_convection", 0.0)
                            ),
                            "kinetic_energy_after_convection": float(
                                conv_diag.get("kinetic_energy_after_convection", 0.0)
                            ),
                            "divergence_after_convection_before_projection_max": float(
                                conv_diag.get(
                                    "divergence_after_convection_before_projection_max",
                                    0.0,
                                )
                            ),
                            "divergence_after_convection_before_projection_l2": float(
                                conv_diag.get(
                                    "divergence_after_convection_before_projection_l2",
                                    0.0,
                                )
                            ),
                        }
                    )
                    s_cmp = s_next
                runtime_cmp = float(perf_counter() - t_cmp0)
                acc_hist_cmp = _evaluate_flow_progression_acceptance(
                    history=hist_cmp,
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                conv_acc_cmp = _evaluate_convective_prototype_acceptance(
                    history=hist_cmp,
                    stokes_baseline=stokes_ref_for_readiness,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    inlet_speed=float(args.inlet_speed),
                )
                readiness_cmp = _evaluate_convective_readiness(
                    convective_prototype_accepted=bool(
                        conv_acc_cmp.get("convective_prototype_accepted", False)
                    ),
                    convective_prototype_acceptance_reason=str(
                        conv_acc_cmp.get("reason", "")
                    ),
                    history=hist_cmp,
                    stokes_baseline=stokes_ref_for_readiness,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    max_convective_substeps=int(args.max_convective_substeps),
                    flow_dt_mode=str(args.flow_dt_mode),
                    convective_cfl_target=float(args.convective_cfl_target),
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                    inlet_speed=float(args.inlet_speed),
                )
                conv_stats_cmp = _convective_history_stats(
                    hist_cmp,
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                )
                final_cmp = hist_cmp[-1] if hist_cmp else {}
                iters_cmp = [
                    int(r.get("pressure_iterations", 0))
                    for r in hist_cmp
                    if int(r.get("pressure_iterations", 0)) > 0
                ]
                comp_items.append(
                    {
                        "mode": str(scenario_name),
                        "convective_stabilization_mode": str(
                            final_cmp.get(
                                "convective_stabilization_mode",
                                cmp_cfg.convective_stabilization_mode,
                            )
                        ),
                        "convective_substep_boundary_contract_mode": str(
                            final_cmp.get(
                                "convective_substep_boundary_contract_mode",
                                cmp_cfg.convective_substep_boundary_contract,
                            )
                        ),
                        "final_div_l2": float(
                            final_cmp.get("final_divergence_l2", 0.0)
                        ),
                        "final_div_max": float(
                            final_cmp.get("final_divergence_max_abs", 0.0)
                        ),
                        "outlet_inlet": float(
                            final_cmp.get("outlet_inlet_flux_ratio", 0.0)
                        ),
                        "wall_flux": float(
                            final_cmp.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "velocity_max": float(
                            final_cmp.get("velocity_magnitude_max", 0.0)
                        ),
                        "velocity_mean": float(
                            final_cmp.get("velocity_magnitude_mean", 0.0)
                        ),
                        "velocity_p95": float(
                            final_cmp.get("velocity_magnitude_p95", 0.0)
                        ),
                        "kinetic_energy": float(
                            final_cmp.get("kinetic_energy_after_projection", 0.0)
                        ),
                        "raw_cfl_max": float(
                            conv_stats_cmp.get("raw_cfl_max_max", 0.0)
                        ),
                        "raw_cfl_p95": float(
                            conv_stats_cmp.get("raw_cfl_p95_max", 0.0)
                        ),
                        "effective_cfl_max": float(
                            conv_stats_cmp.get("effective_cfl_max_max", 0.0)
                        ),
                        "effective_cfl_p95": float(
                            conv_stats_cmp.get("effective_cfl_p95_max", 0.0)
                        ),
                        "substep_cfl_max": float(
                            conv_stats_cmp.get("substep_cfl_max_max", 0.0)
                        ),
                        "substep_cfl_p95": float(
                            conv_stats_cmp.get("substep_cfl_p95_max", 0.0)
                        ),
                        "damping_effective_min": float(
                            conv_stats_cmp.get("damping_effective_min", 0.0)
                        ),
                        "damping_effective_mean": float(
                            conv_stats_cmp.get("damping_effective_mean", 0.0)
                        ),
                        "damping_effective_max": float(
                            conv_stats_cmp.get("damping_effective_max", 0.0)
                        ),
                        "substep_count_mean": float(
                            conv_stats_cmp.get("substep_count_mean", 0.0)
                        ),
                        "substep_count_max": int(
                            conv_stats_cmp.get("substep_count_max", 0)
                        ),
                        "convective_delta_velocity_max": float(
                            final_cmp.get("convective_delta_velocity_max", 0.0)
                        ),
                        "convective_delta_velocity_l2": float(
                            final_cmp.get("convective_delta_velocity_l2", 0.0)
                        ),
                        "kinetic_energy_before_convection": float(
                            final_cmp.get("kinetic_energy_before_convection", 0.0)
                        ),
                        "kinetic_energy_after_convection": float(
                            final_cmp.get("kinetic_energy_after_convection", 0.0)
                        ),
                        "divergence_after_convection_before_projection_l2": float(
                            final_cmp.get(
                                "divergence_after_convection_before_projection_l2", 0.0
                            )
                        ),
                        "flow_progression_solved": bool(
                            acc_hist_cmp.get("flow_progression_solved", False)
                        ),
                        "convective_prototype_accepted": bool(
                            conv_acc_cmp.get("convective_prototype_accepted", False)
                        ),
                        "ready_for_long_ns_run_debug": bool(
                            readiness_cmp.get("ready_for_long_ns_run_debug", False)
                        ),
                        "ready_for_long_ns_run_physical": bool(
                            readiness_cmp.get("ready_for_long_ns_run_physical", False)
                        ),
                        "readiness_reason": str(
                            readiness_cmp.get("readiness_reason", "")
                        ),
                        "pressure_iterations_mean": float(np.mean(iters_cmp))
                        if iters_cmp
                        else 0.0,
                        "pressure_iterations_max": int(np.max(iters_cmp))
                        if iters_cmp
                        else 0,
                        "runtime_seconds": float(runtime_cmp),
                        "steps_per_second": float(
                            flow_steps_requested / max(runtime_cmp, 1e-12)
                        ),
                    }
                )
            convective_stabilization_comparison = {
                "flow_mode": "navier_stokes_projection_debug",
                "flow_steps": int(flow_steps_requested),
                "flow_dt": float(flow_dt),
                "convective_cfl_limit": float(args.convective_cfl_limit),
                "items": comp_items,
            }
            _write_json(
                run_dir / "convective_stabilization_comparison.json",
                convective_stabilization_comparison,
            )
        compare_visc_modes = _parse_viscous_predictor_modes(
            str(args.compare_viscous_predictor_modes)
        )
        if bool(args.compare_ns_dt_modes):
            scenario_items: list[dict[str, Any]] = []
            scenario_defs = [
                {
                    "label": "manual_dt_auto_damping",
                    "flow_dt_mode": "manual",
                    "disable_convective_auto_damping": False,
                },
                {
                    "label": "auto_cfl_no_auto_damping",
                    "flow_dt_mode": "auto_cfl",
                    "disable_convective_auto_damping": True,
                },
                {
                    "label": "auto_cfl_auto_damping",
                    "flow_dt_mode": "auto_cfl",
                    "disable_convective_auto_damping": False,
                },
            ]
            for sc in scenario_defs:
                sc_cfg = replace(
                    progression_cfg,
                    enable_convective_predictor=True,
                    disable_convective_predictor=False,
                    convective_stabilization_mode="auto_damping",
                    disable_convective_auto_damping=bool(
                        sc["disable_convective_auto_damping"]
                    ),
                )
                s_cmp = state0
                hist_cmp: list[dict[str, Any]] = []
                used_dt_cmp: list[float] = []
                auto_dt_scale_cmp: list[float] = []
                physical_time_cmp = 0.0
                t0_cmp = perf_counter()
                for step_idx in range(1, flow_steps_requested + 1):
                    rate_diag = compute_tetra_convective_cfl_rate(
                        mesh, np.asarray(s_cmp.face_flux, dtype=np.float64)
                    )
                    rate_max = float(rate_diag.get("cfl_rate_max", 0.0))
                    rate_p95 = float(rate_diag.get("cfl_rate_p95", 0.0))
                    dt_req = float(requested_flow_dt)
                    dt_use = float(dt_req)
                    auto_dt_scale = 1.0
                    auto_dt_min_hit = False
                    auto_dt_max_hit = False
                    auto_dt_floor = float(flow_dt_min)
                    if str(sc["flow_dt_mode"]) == "auto_cfl":
                        dt_from_cfl = float(
                            convective_cfl_target / max(rate_max, 1e-30)
                            if rate_max > 0.0
                            else flow_dt_max
                        )
                        if bool(flow_dt_min_explicit):
                            auto_dt_floor = float(flow_dt_min)
                        else:
                            auto_dt_floor = float(
                                max(1e-12, min(float(flow_dt_min), 0.05 * dt_from_cfl))
                            )
                        auto_dt_min_hit = bool(dt_from_cfl < auto_dt_floor)
                        auto_dt_max_hit = bool(dt_from_cfl > flow_dt_max)
                        dt_use = float(
                            min(max(dt_from_cfl, auto_dt_floor), flow_dt_max)
                        )
                        auto_dt_scale = float(dt_use / max(dt_req, 1e-30))
                    used_dt_cmp.append(float(dt_use))
                    auto_dt_scale_cmp.append(float(auto_dt_scale))
                    physical_time_cmp += float(dt_use)
                    step_cfg_cmp = replace(sc_cfg, projection_dt=float(dt_use))
                    s_conv = apply_tetra_convective_predictor(
                        mesh,
                        s_cmp,
                        step_cfg_cmp,
                        flow_dt=float(dt_use),
                    )
                    conv_diag_cmp = dict(
                        s_conv.diagnostics.get("convective_predictor", {})
                    )
                    s_pred = apply_tetra_stokes_viscous_predictor(
                        mesh,
                        s_conv,
                        step_cfg_cmp,
                        flow_dt=float(dt_use),
                    )
                    s_next = solve_tetra_pressure_projection(mesh, s_pred, step_cfg_cmp)
                    d_cmp = dict(s_next.diagnostics)
                    p_cmp = dict(d_cmp.get("projection", {}))
                    v_cmp = dict(d_cmp.get("velocity", {}))
                    pr_cmp = dict(d_cmp.get("pressure", {}))
                    be_cmp = dict(d_cmp.get("backend_execution", {}))
                    acc_cmp = _projection_acceptance(
                        projection=p_cmp,
                        pressure=pr_cmp,
                        backend_execution=be_cmp,
                        thresholds=acceptance_thresholds_cli,
                    )
                    speed_cmp = np.linalg.norm(
                        np.asarray(s_next.cell_velocity, dtype=np.float64), axis=1
                    )
                    hist_cmp.append(
                        {
                            "step": int(step_idx),
                            "flow_dt_mode": str(sc["flow_dt_mode"]),
                            "requested_flow_dt": float(dt_req),
                            "used_flow_dt": float(dt_use),
                            "physical_time": float(physical_time_cmp),
                            "auto_dt_scale_factor": float(auto_dt_scale),
                            "auto_dt_floor_used": float(auto_dt_floor),
                            "auto_dt_min_hit": bool(auto_dt_min_hit),
                            "auto_dt_max_hit": bool(auto_dt_max_hit),
                            "convective_cfl_target": float(convective_cfl_target),
                            "raw_cfl_max_before_dt_selection": float(dt_req * rate_max),
                            "raw_cfl_max_after_dt_selection": float(dt_use * rate_max),
                            "raw_cfl_p95_after_dt_selection": float(dt_use * rate_p95),
                            "projection_solved": bool(
                                acc_cmp.get("projection_solved", False)
                            ),
                            "finite_fields": bool(
                                acc_cmp.get("checklist", {}).get("finite_fields", False)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                float(p_cmp.get("outlet_flux_total_after", 0.0))
                                / max(
                                    abs(
                                        float(p_cmp.get("inlet_flux_total_after", 0.0))
                                    ),
                                    1e-30,
                                )
                            ),
                            "net_boundary_flux_relative": float(
                                abs(float(p_cmp.get("net_boundary_flux_after", 0.0)))
                                / max(
                                    abs(
                                        float(p_cmp.get("inlet_flux_total_after", 0.0))
                                    ),
                                    1e-30,
                                )
                            ),
                            "wall_flux_max_abs_after": float(
                                p_cmp.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "final_divergence_max_abs": float(
                                p_cmp.get("final_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_l2": float(
                                p_cmp.get("final_divergence_l2", 0.0)
                            ),
                            "velocity_magnitude_max": float(
                                v_cmp.get("magnitude_max", 0.0)
                            ),
                            "velocity_magnitude_mean": float(
                                v_cmp.get("magnitude_mean", 0.0)
                            ),
                            "kinetic_energy_after_projection": float(
                                np.mean(0.5 * speed_cmp * speed_cmp)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "pressure_iterations": int(
                                pr_cmp.get("actual_iterations", 0)
                            ),
                            "convective_cfl_limit": float(
                                conv_diag_cmp.get(
                                    "convective_cfl_limit", sc_cfg.convective_cfl_limit
                                )
                            ),
                            "convective_cfl_raw_max": float(
                                conv_diag_cmp.get(
                                    "convective_cfl_raw_max",
                                    conv_diag_cmp.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_raw_p95": float(
                                conv_diag_cmp.get(
                                    "convective_cfl_raw_p95",
                                    conv_diag_cmp.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_effective_max": float(
                                conv_diag_cmp.get(
                                    "convective_cfl_effective_max",
                                    conv_diag_cmp.get("convective_cfl_max", 0.0),
                                )
                            ),
                            "convective_cfl_effective_p95": float(
                                conv_diag_cmp.get(
                                    "convective_cfl_effective_p95",
                                    conv_diag_cmp.get("convective_cfl_p95", 0.0),
                                )
                            ),
                            "convective_cfl_warning_raw": bool(
                                conv_diag_cmp.get(
                                    "convective_cfl_warning_raw",
                                    conv_diag_cmp.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_cfl_warning_effective": bool(
                                conv_diag_cmp.get(
                                    "convective_cfl_warning_effective",
                                    conv_diag_cmp.get("convective_cfl_warning", False),
                                )
                            ),
                            "convective_predictor_damping_effective": float(
                                conv_diag_cmp.get(
                                    "convective_predictor_damping_effective", 0.0
                                )
                            ),
                            "convective_auto_damping_used": bool(
                                conv_diag_cmp.get("convective_auto_damping_used", False)
                            ),
                        }
                    )
                    s_cmp = s_next
                runtime_cmp = float(perf_counter() - t0_cmp)
                acc_hist_cmp = _evaluate_flow_progression_acceptance(
                    history=hist_cmp,
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                conv_acc_cmp = _evaluate_convective_prototype_acceptance(
                    history=hist_cmp,
                    stokes_baseline=None,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    inlet_speed=float(args.inlet_speed),
                )
                readiness_cmp = _evaluate_convective_readiness(
                    convective_prototype_accepted=bool(
                        conv_acc_cmp.get("convective_prototype_accepted", False)
                    ),
                    convective_prototype_acceptance_reason=str(
                        conv_acc_cmp.get("reason", "")
                    ),
                    history=hist_cmp,
                    stokes_baseline=None,
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    max_convective_substeps=int(args.max_convective_substeps),
                    flow_dt_mode=str(sc["flow_dt_mode"]),
                    convective_cfl_target=float(convective_cfl_target),
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                    inlet_speed=float(args.inlet_speed),
                )
                conv_stats_cmp = _convective_history_stats(
                    hist_cmp,
                    convective_cfl_acceptance_eps=float(
                        args.convective_cfl_acceptance_eps
                    ),
                )
                final_cmp = hist_cmp[-1] if hist_cmp else {}
                scenario_items.append(
                    {
                        "scenario": str(sc["label"]),
                        "flow_dt_mode": str(sc["flow_dt_mode"]),
                        "disable_convective_auto_damping": bool(
                            sc["disable_convective_auto_damping"]
                        ),
                        "flow_progression_solved": bool(
                            acc_hist_cmp.get("flow_progression_solved", False)
                        ),
                        "ready_for_long_ns_run_debug": bool(
                            readiness_cmp.get("ready_for_long_ns_run_debug", False)
                        ),
                        "ready_for_long_ns_run_physical": bool(
                            readiness_cmp.get("ready_for_long_ns_run_physical", False)
                        ),
                        "final_div_l2": float(
                            final_cmp.get("final_divergence_l2", 0.0)
                        ),
                        "final_div_max": float(
                            final_cmp.get("final_divergence_max_abs", 0.0)
                        ),
                        "outlet_inlet": float(
                            final_cmp.get("outlet_inlet_flux_ratio", 0.0)
                        ),
                        "wall_flux": float(
                            final_cmp.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "used_dt_min": float(conv_stats_cmp.get("used_dt_min", 0.0)),
                        "used_dt_mean": float(conv_stats_cmp.get("used_dt_mean", 0.0)),
                        "used_dt_max": float(conv_stats_cmp.get("used_dt_max", 0.0)),
                        "physical_time_final": float(physical_time_cmp),
                        "runtime_seconds": float(runtime_cmp),
                        "steps_per_second": float(
                            flow_steps_requested / max(runtime_cmp, 1e-12)
                        ),
                        "raw_cfl_max_after_dt_selection": float(
                            conv_stats_cmp.get("raw_cfl_after_dt_selection_max", 0.0)
                        ),
                        "raw_cfl_p95_after_dt_selection": float(
                            conv_stats_cmp.get("raw_cfl_after_dt_selection_p95", 0.0)
                        ),
                        "effective_cfl_max": float(
                            conv_stats_cmp.get("effective_cfl_max_max", 0.0)
                        ),
                        "effective_cfl_limit_excess_max": float(
                            conv_stats_cmp.get("effective_cfl_limit_excess_max", 0.0)
                        ),
                        "warning_step_count": int(
                            conv_stats_cmp.get("effective_cfl_warning_step_count", 0)
                        ),
                        "effective_cfl_warning_steps": list(
                            conv_stats_cmp.get("effective_cfl_warning_steps", [])
                        ),
                        "effective_damping_min": float(
                            conv_stats_cmp.get("damping_effective_min", 0.0)
                        ),
                        "effective_damping_mean": float(
                            conv_stats_cmp.get("damping_effective_mean", 0.0)
                        ),
                        "effective_damping_max": float(
                            conv_stats_cmp.get("damping_effective_max", 0.0)
                        ),
                        "kinetic_energy_final": float(
                            final_cmp.get("kinetic_energy_after_projection", 0.0)
                        ),
                        "velocity_max_final": float(
                            final_cmp.get("velocity_magnitude_max", 0.0)
                        ),
                        "readiness_reason": str(
                            readiness_cmp.get("readiness_reason", "")
                        ),
                    }
                )
            ns_dt_mode_comparison = {
                "flow_mode": "navier_stokes_projection_debug",
                "flow_steps": int(flow_steps_requested),
                "requested_flow_dt": float(requested_flow_dt),
                "flow_dt_min": float(flow_dt_min),
                "flow_dt_max": float(flow_dt_max),
                "convective_cfl_target": float(convective_cfl_target),
                "items": scenario_items,
            }
            for _it in scenario_items:
                if str(_it.get("scenario", "")) == "auto_cfl_no_auto_damping":
                    auto_cfl_no_damping_readiness_reason = str(
                        _it.get("readiness_reason", "")
                    )
                    break
            _write_json(run_dir / "ns_dt_mode_comparison.json", ns_dt_mode_comparison)
        if compare_visc_modes:
            mode_items: list[dict[str, Any]] = []
            for vmode in compare_visc_modes:
                cmp_cfg = replace(progression_cfg, viscous_predictor_mode=str(vmode))
                s_cmp = state0
                hist_cmp: list[dict[str, Any]] = []
                for step_idx in range(1, flow_steps_requested + 1):
                    s_pred = apply_tetra_stokes_viscous_predictor(
                        mesh,
                        s_cmp,
                        cmp_cfg,
                        flow_dt=float(flow_dt),
                    )
                    visc_cmp = dict(s_pred.diagnostics.get("viscous_predictor", {}))
                    s_next = solve_tetra_pressure_projection(mesh, s_pred, cmp_cfg)
                    p_cmp = dict(s_next.diagnostics.get("projection", {}))
                    v_cmp = dict(s_next.diagnostics.get("velocity", {}))
                    pr_cmp = dict(s_next.diagnostics.get("pressure", {}))
                    inlet_after = float(p_cmp.get("inlet_flux_total_after", 0.0))
                    outlet_after = float(p_cmp.get("outlet_flux_total_after", 0.0))
                    speed_cmp = np.linalg.norm(
                        np.asarray(s_next.cell_velocity, dtype=np.float64), axis=1
                    )
                    hist_cmp.append(
                        {
                            "step": int(step_idx),
                            "projection_solved": bool(
                                _projection_acceptance(
                                    projection=p_cmp,
                                    pressure=pr_cmp,
                                    backend_execution=dict(
                                        s_next.diagnostics.get("backend_execution", {})
                                    ),
                                    thresholds=acceptance_thresholds_cli,
                                ).get("projection_solved", False)
                            ),
                            "divergence_after_predictor_before_projection_l2": float(
                                visc_cmp.get(
                                    "divergence_after_predictor_before_boundary_contract_l2",
                                    0.0,
                                )
                            ),
                            "divergence_after_predictor_before_projection_max": float(
                                visc_cmp.get(
                                    "divergence_after_predictor_before_boundary_contract_max",
                                    0.0,
                                )
                            ),
                            "divergence_after_projection_l2": float(
                                p_cmp.get("final_divergence_l2", 0.0)
                            ),
                            "divergence_after_projection_max": float(
                                p_cmp.get("final_divergence_max_abs", 0.0)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                outlet_after / max(abs(inlet_after), 1e-30)
                            ),
                            "net_boundary_flux_after": float(
                                p_cmp.get("net_boundary_flux_after", 0.0)
                            ),
                            "wall_flux_max_abs_after": float(
                                p_cmp.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "velocity_magnitude_max": float(
                                v_cmp.get("magnitude_max", 0.0)
                            ),
                            "velocity_magnitude_mean": float(
                                v_cmp.get("magnitude_mean", 0.0)
                            ),
                            "velocity_magnitude_p95": float(
                                np.percentile(speed_cmp, 95.0)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "kinetic_energy_after_projection": float(
                                np.mean(0.5 * speed_cmp * speed_cmp)
                            )
                            if speed_cmp.size
                            else 0.0,
                            "pressure_iterations": int(
                                pr_cmp.get("actual_iterations", 0)
                            ),
                        }
                    )
                    s_cmp = s_next
                acc_hist_cmp = _evaluate_flow_progression_acceptance(
                    history=[
                        {
                            "step": int(r["step"]),
                            "projection_solved": bool(r["projection_solved"]),
                            "finite_fields": True,
                            "outlet_inlet_flux_ratio": float(
                                r["outlet_inlet_flux_ratio"]
                            ),
                            "wall_flux_max_abs_after": float(
                                r["wall_flux_max_abs_after"]
                            ),
                            "final_divergence_max_abs": float(
                                r["divergence_after_projection_max"]
                            ),
                            "initial_divergence_max_abs": float(
                                r["divergence_after_predictor_before_projection_max"]
                            ),
                        }
                        for r in hist_cmp
                    ],
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                final_cmp = hist_cmp[-1] if hist_cmp else {}
                iters = [int(r.get("pressure_iterations", 0)) for r in hist_cmp]
                arr_pred_l2 = np.asarray(
                    [
                        float(
                            r.get(
                                "divergence_after_predictor_before_projection_l2", 0.0
                            )
                        )
                        for r in hist_cmp
                    ],
                    dtype=np.float64,
                )
                arr_pred_max = np.asarray(
                    [
                        float(
                            r.get(
                                "divergence_after_predictor_before_projection_max", 0.0
                            )
                        )
                        for r in hist_cmp
                    ],
                    dtype=np.float64,
                )
                arr_proj_l2 = np.asarray(
                    [
                        float(r.get("divergence_after_projection_l2", 0.0))
                        for r in hist_cmp
                    ],
                    dtype=np.float64,
                )
                arr_proj_max = np.asarray(
                    [
                        float(r.get("divergence_after_projection_max", 0.0))
                        for r in hist_cmp
                    ],
                    dtype=np.float64,
                )
                capped_updates_total = int(
                    np.sum(
                        np.asarray(
                            [
                                int(r.get("capped_predictor_updates_count", 0))
                                for r in hist_cmp
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                predictor_updates_total = int(
                    np.sum(
                        np.asarray(
                            [
                                int(r.get("total_predictor_updates_count", 0))
                                for r in hist_cmp
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                mode_items.append(
                    {
                        "viscous_predictor_mode": str(vmode),
                        "flow_progression_solved": bool(
                            acc_hist_cmp.get("flow_progression_solved", False)
                        ),
                        "final_div_max_abs": float(
                            final_cmp.get("divergence_after_projection_max", 0.0)
                        ),
                        "final_div_l2": float(
                            final_cmp.get("divergence_after_projection_l2", 0.0)
                        ),
                        "outlet_inlet_ratio": float(
                            final_cmp.get("outlet_inlet_flux_ratio", 0.0)
                        ),
                        "net_boundary_flux_after": float(
                            final_cmp.get("net_boundary_flux_after", 0.0)
                        ),
                        "net_boundary_flux_relative": float(
                            abs(float(final_cmp.get("net_boundary_flux_after", 0.0)))
                            / max(
                                abs(
                                    float(
                                        s_cmp.diagnostics.get("projection", {}).get(
                                            "inlet_flux_total_after", 0.0
                                        )
                                    )
                                ),
                                1e-30,
                            )
                        ),
                        "wall_flux_max_abs_after": float(
                            final_cmp.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "velocity_max_final": float(
                            final_cmp.get("velocity_magnitude_max", 0.0)
                        ),
                        "velocity_mean_final": float(
                            final_cmp.get("velocity_magnitude_mean", 0.0)
                        ),
                        "velocity_p95_final": float(
                            final_cmp.get("velocity_magnitude_p95", 0.0)
                        ),
                        "kinetic_energy_final": float(
                            final_cmp.get("kinetic_energy_after_projection", 0.0)
                        ),
                        "pressure_iterations_mean": float(np.mean(iters))
                        if iters
                        else 0.0,
                        "pressure_iterations_max": int(np.max(iters)) if iters else 0,
                        "failed_steps": [
                            int(r.get("step", -1))
                            for r in hist_cmp
                            if not bool(r.get("projection_solved", False))
                        ],
                        "startup_warning_steps": list(
                            acc_hist_cmp.get("startup_warning_steps_observed", [])
                        ),
                        "divergence_after_predictor_before_projection_mean_l2": float(
                            np.mean(arr_pred_l2)
                        )
                        if arr_pred_l2.size
                        else 0.0,
                        "divergence_after_predictor_before_projection_max_l2": float(
                            np.max(arr_pred_l2)
                        )
                        if arr_pred_l2.size
                        else 0.0,
                        "divergence_after_predictor_before_projection_mean_max": float(
                            np.mean(arr_pred_max)
                        )
                        if arr_pred_max.size
                        else 0.0,
                        "divergence_after_predictor_before_projection_max_max": float(
                            np.max(arr_pred_max)
                        )
                        if arr_pred_max.size
                        else 0.0,
                        "predictor_damages_divergence_significantly": bool(
                            (
                                np.mean(arr_pred_l2)
                                > 10.0 * max(np.mean(arr_proj_l2), 1e-30)
                            )
                            or (
                                np.mean(arr_pred_max)
                                > 100.0 * max(np.mean(arr_proj_max), 1e-30)
                            )
                        ),
                        "capped_predictor_updates_count_total": int(
                            capped_updates_total
                        ),
                        "total_predictor_updates_count_total": int(
                            predictor_updates_total
                        ),
                        "capped_predictor_updates_fraction_total": float(
                            capped_updates_total / max(predictor_updates_total, 1)
                        ),
                    }
                )
            mode_lookup_local = {
                str(it.get("viscous_predictor_mode", "")): it for it in mode_items
            }
            base_ref = mode_lookup_local.get(
                "no_viscous_debug_copy", None
            ) or mode_lookup_local.get("none", None)
            if base_ref is not None:
                base_l2 = float(base_ref.get("final_div_l2", 1e-30))
                base_linf = float(base_ref.get("final_div_max_abs", 1e-30))
                for it in mode_items:
                    ratio_l2 = float(it.get("final_div_l2", 0.0)) / max(base_l2, 1e-30)
                    ratio_linf = float(it.get("final_div_max_abs", 0.0)) / max(
                        base_linf, 1e-30
                    )
                    dmg = bool((ratio_l2 > 10.0) or (ratio_linf > 100.0))
                    dmg_target = bool((ratio_l2 > 100.0) or (ratio_linf > 1000.0))
                    it["damage_ratio_vs_noop_l2"] = float(ratio_l2)
                    it["damage_ratio_vs_noop_linf"] = float(ratio_linf)
                    it["predictor_damages_divergence_significantly"] = bool(dmg_target)
                    it["stokes_ready_for_advection_term"] = bool(
                        _evaluate_stokes_ready_for_advection(
                            damage_ratio_l2=float(ratio_l2),
                            damage_ratio_linf=float(ratio_linf),
                            predictor_damages_divergence=bool(dmg),
                        )
                    )
            viscous_predictor_mode_comparison = {"items": mode_items}
            _write_json(
                run_dir / "viscous_predictor_mode_comparison.json",
                viscous_predictor_mode_comparison,
            )
            mode_map = {
                str(it.get("viscous_predictor_mode", "")): it for it in mode_items
            }
            best = (
                min(
                    mode_items, key=lambda x: float(x.get("final_div_l2", float("inf")))
                )
                if mode_items
                else None
            )
            if best is not None:
                viscous_predictor_best_mode = str(
                    best.get("viscous_predictor_mode", resolved_viscous_predictor_mode)
                )
            base_none = mode_map.get("none", None) or mode_map.get(
                "no_viscous_debug_copy", None
            )
            cur = mode_map.get(str(resolved_viscous_predictor_mode), None)
            if (base_none is not None) and (cur is not None):
                viscous_predictor_damage_ratio_l2 = float(
                    cur.get("final_div_l2", 0.0)
                ) / max(float(base_none.get("final_div_l2", 1e-30)), 1e-30)
                viscous_predictor_damage_ratio_linf = float(
                    cur.get("final_div_max_abs", 0.0)
                ) / max(float(base_none.get("final_div_max_abs", 1e-30)), 1e-30)
                viscous_predictor_damages_divergence = bool(
                    (viscous_predictor_damage_ratio_l2 > 10.0)
                    or (viscous_predictor_damage_ratio_linf > 100.0)
                    or bool(
                        cur.get("predictor_damages_divergence_significantly", False)
                    )
                )
                stokes_ready_for_advection_term = _evaluate_stokes_ready_for_advection(
                    damage_ratio_l2=float(viscous_predictor_damage_ratio_l2),
                    damage_ratio_linf=float(viscous_predictor_damage_ratio_linf),
                    predictor_damages_divergence=bool(
                        viscous_predictor_damages_divergence
                    ),
                )
            else:
                stokes_ready_for_advection_term = False
        else:
            stokes_ready_for_advection_term = False

        if bool(args.run_stokes_sensitivity_sweep):
            sweep_flow_dts = _parse_float_list(str(args.sweep_flow_dts))
            if not sweep_flow_dts:
                sweep_flow_dts = [1e-4, 2.5e-4, 5e-4, 1e-3]
            sweep_caps = _parse_cap_list(str(args.sweep_viscous_face_flux_caps))
            if not sweep_caps:
                sweep_caps = [None, 0.1, 0.03, 0.01]
            # Deduplicate while preserving order.
            sweep_flow_dts = [float(x) for x in dict.fromkeys(sweep_flow_dts)]
            sweep_caps = list(dict.fromkeys(sweep_caps))

            def _run_stokes_variant(
                *,
                flow_dt_variant: float,
                cap_variant: float,
                predictor_mode_variant: str,
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                cfg_variant = replace(
                    cfg,
                    projection_dt=float(flow_dt_variant),
                    viscous_predictor_mode=str(predictor_mode_variant),  # type: ignore[arg-type]
                    viscous_face_flux_divergence_impact_cap=float(cap_variant),
                )
                state_variant = initialize_tetra_flow_state(mesh, cfg_variant)
                hist_variant: list[dict[str, Any]] = []
                t0_variant = perf_counter()
                for step_idx in range(1, flow_steps_requested + 1):
                    s_pred_variant = apply_tetra_stokes_viscous_predictor(
                        mesh,
                        state_variant,
                        cfg_variant,
                        flow_dt=float(flow_dt_variant),
                    )
                    visc_variant = dict(
                        s_pred_variant.diagnostics.get("viscous_predictor", {})
                    )
                    s_next_variant = solve_tetra_pressure_projection(
                        mesh, s_pred_variant, cfg_variant
                    )
                    p_variant = dict(s_next_variant.diagnostics.get("projection", {}))
                    pr_variant = dict(s_next_variant.diagnostics.get("pressure", {}))
                    be_variant = dict(
                        s_next_variant.diagnostics.get("backend_execution", {})
                    )
                    v_variant = dict(s_next_variant.diagnostics.get("velocity", {}))
                    acc_variant = _projection_acceptance(
                        projection=p_variant,
                        pressure=pr_variant,
                        backend_execution=be_variant,
                        thresholds=acceptance_thresholds_cli,
                    )
                    inlet_after_variant = float(
                        p_variant.get("inlet_flux_total_after", 0.0)
                    )
                    outlet_after_variant = float(
                        p_variant.get("outlet_flux_total_after", 0.0)
                    )
                    net_after_variant = float(
                        p_variant.get("net_boundary_flux_after", 0.0)
                    )
                    speed_variant = np.linalg.norm(
                        np.asarray(s_next_variant.cell_velocity, dtype=np.float64),
                        axis=1,
                    )
                    hist_variant.append(
                        {
                            "step": int(step_idx),
                            "projection_solved": bool(
                                acc_variant.get("projection_solved", False)
                            ),
                            "finite_fields": bool(
                                acc_variant.get("checklist", {}).get(
                                    "finite_fields", False
                                )
                            ),
                            "final_divergence_max_abs": float(
                                p_variant.get("final_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_l2": float(
                                p_variant.get("final_divergence_l2", 0.0)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                outlet_after_variant
                                / max(abs(inlet_after_variant), 1e-30)
                            ),
                            "net_boundary_flux_after": float(net_after_variant),
                            "net_boundary_flux_relative": float(
                                abs(net_after_variant)
                                / max(abs(inlet_after_variant), 1e-30)
                            ),
                            "wall_flux_max_abs_after": float(
                                p_variant.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "velocity_magnitude_max": float(
                                v_variant.get("magnitude_max", 0.0)
                            ),
                            "velocity_magnitude_mean": float(
                                v_variant.get("magnitude_mean", 0.0)
                            ),
                            "velocity_magnitude_p95": float(
                                np.percentile(speed_variant, 95.0)
                            )
                            if speed_variant.size
                            else 0.0,
                            "kinetic_energy_after_projection": float(
                                np.mean(0.5 * speed_variant * speed_variant)
                            )
                            if speed_variant.size
                            else 0.0,
                            "face_flux_delta_predictor_l2": float(
                                visc_variant.get("face_flux_delta_predictor_l2", 0.0)
                            ),
                            "pressure_iterations": int(
                                pr_variant.get("actual_iterations", 0)
                            ),
                            "capped_predictor_updates_count": int(
                                visc_variant.get("capped_predictor_updates_count", 0)
                            ),
                            "total_predictor_updates_count": int(
                                visc_variant.get("total_predictor_updates_count", 0)
                            ),
                            "capped_predictor_updates_fraction": float(
                                visc_variant.get(
                                    "capped_predictor_updates_fraction", 0.0
                                )
                            ),
                        }
                    )
                    state_variant = s_next_variant
                runtime_variant = perf_counter() - t0_variant
                acc_hist_variant = _evaluate_flow_progression_acceptance(
                    history=hist_variant,
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                final_variant = hist_variant[-1] if hist_variant else {}
                pressure_iters = [
                    int(r.get("pressure_iterations", 0))
                    for r in hist_variant
                    if int(r.get("pressure_iterations", 0)) > 0
                ]
                face_flux_delta_l2 = np.asarray(
                    [
                        float(r.get("face_flux_delta_predictor_l2", 0.0))
                        for r in hist_variant
                    ],
                    dtype=np.float64,
                )
                capped_updates_total_variant = int(
                    np.sum(
                        np.asarray(
                            [
                                int(r.get("capped_predictor_updates_count", 0))
                                for r in hist_variant
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                predictor_updates_total_variant = int(
                    np.sum(
                        np.asarray(
                            [
                                int(r.get("total_predictor_updates_count", 0))
                                for r in hist_variant
                            ],
                            dtype=np.int64,
                        )
                    )
                )
                row_variant = {
                    "flow_dt": float(flow_dt_variant),
                    "flow_steps": int(flow_steps_requested),
                    "viscous_predictor_mode": str(predictor_mode_variant),
                    "flow_progression_solved": bool(
                        acc_hist_variant.get("flow_progression_solved", False)
                    ),
                    "final_div_max_abs": float(
                        final_variant.get("final_divergence_max_abs", 0.0)
                    ),
                    "final_div_l2": float(
                        final_variant.get("final_divergence_l2", 0.0)
                    ),
                    "outlet_inlet_flux_ratio": float(
                        final_variant.get("outlet_inlet_flux_ratio", 0.0)
                    ),
                    "net_boundary_flux_relative": float(
                        final_variant.get("net_boundary_flux_relative", 0.0)
                    ),
                    "wall_flux_max_abs_after": float(
                        final_variant.get("wall_flux_max_abs_after", 0.0)
                    ),
                    "velocity_max_final": float(
                        final_variant.get("velocity_magnitude_max", 0.0)
                    ),
                    "velocity_mean_final": float(
                        final_variant.get("velocity_magnitude_mean", 0.0)
                    ),
                    "velocity_p95_final": float(
                        final_variant.get("velocity_magnitude_p95", 0.0)
                    ),
                    "kinetic_energy_final": float(
                        final_variant.get("kinetic_energy_after_projection", 0.0)
                    ),
                    "face_flux_delta_predictor_l2": float(np.mean(face_flux_delta_l2))
                    if face_flux_delta_l2.size
                    else 0.0,
                    "pressure_iterations_mean": float(np.mean(pressure_iters))
                    if pressure_iters
                    else 0.0,
                    "runtime_seconds": float(runtime_variant),
                    "steps_per_second": float(
                        flow_steps_requested / max(runtime_variant, 1e-12)
                    ),
                    "startup_warning_steps": list(
                        acc_hist_variant.get("startup_warning_steps_observed", [])
                    ),
                    "failed_steps": [
                        int(r.get("step", -1))
                        for r in hist_variant
                        if not bool(r.get("projection_solved", False))
                    ],
                    "finite_fields_final": bool(
                        final_variant.get("finite_fields", False)
                    ),
                    "capped_predictor_updates_count_total": int(
                        capped_updates_total_variant
                    ),
                    "total_predictor_updates_count_total": int(
                        predictor_updates_total_variant
                    ),
                    "capped_predictor_updates_fraction_total": float(
                        capped_updates_total_variant
                        / max(predictor_updates_total_variant, 1)
                    ),
                }
                return row_variant, hist_variant

            baseline_noop_by_dt: dict[float, dict[str, Any]] = {}
            for dt_candidate in sweep_flow_dts:
                noop_row, _ = _run_stokes_variant(
                    flow_dt_variant=float(dt_candidate),
                    cap_variant=float(args.viscous_face_flux_divergence_impact_cap),
                    predictor_mode_variant="no_viscous_debug_copy",
                )
                baseline_noop_by_dt[float(dt_candidate)] = noop_row

            sweep_rows: list[dict[str, Any]] = []
            for dt_candidate in sweep_flow_dts:
                for cap_candidate in sweep_caps:
                    resolved_cap = (
                        float(args.viscous_face_flux_divergence_impact_cap)
                        if cap_candidate is None
                        else float(cap_candidate)
                    )
                    row, _ = _run_stokes_variant(
                        flow_dt_variant=float(dt_candidate),
                        cap_variant=float(resolved_cap),
                        predictor_mode_variant="face_flux_laplacian_substepped",
                    )
                    row["viscous_face_flux_divergence_impact_cap"] = (
                        None if cap_candidate is None else float(cap_candidate)
                    )
                    row["resolved_cap_value"] = float(resolved_cap)
                    base_row = baseline_noop_by_dt.get(float(dt_candidate), None)
                    if base_row is not None:
                        dmg_l2 = float(row.get("final_div_l2", 0.0)) / max(
                            float(base_row.get("final_div_l2", 1e-30)), 1e-30
                        )
                        dmg_linf = float(row.get("final_div_max_abs", 0.0)) / max(
                            float(base_row.get("final_div_max_abs", 1e-30)), 1e-30
                        )
                    else:
                        dmg_l2 = 1.0
                        dmg_linf = 1.0
                    damages = bool((dmg_l2 > 10.0) or (dmg_linf > 100.0))
                    row["predictor_damages_divergence"] = bool(damages)
                    row["damage_ratio_vs_noop_l2"] = float(dmg_l2)
                    row["damage_ratio_vs_noop_linf"] = float(dmg_linf)
                    row["stokes_ready_for_advection_term"] = bool(
                        _evaluate_stokes_ready_for_advection(
                            damage_ratio_l2=float(dmg_l2),
                            damage_ratio_linf=float(dmg_linf),
                            predictor_damages_divergence=bool(damages),
                        )
                    )
                    sweep_rows.append(row)

            recommended_stokes_baseline_config = _recommend_stokes_baseline_config(
                sweep_rows
            )
            best_mode = "face_flux_laplacian_substepped"
            if recommended_stokes_baseline_config:
                best_mode = str(
                    recommended_stokes_baseline_config.get(
                        "viscous_predictor_mode", "face_flux_laplacian_substepped"
                    )
                )
            rec_match = None
            for row in sweep_rows:
                if float(row.get("flow_dt", -1.0)) == float(
                    recommended_stokes_baseline_config.get("flow_dt", -2.0)
                ) and float(row.get("resolved_cap_value", -1.0)) == float(
                    recommended_stokes_baseline_config.get("resolved_cap_value", -2.0)
                ):
                    rec_match = row
                    break
            if rec_match is not None:
                stokes_baseline_accepted, stokes_baseline_acceptance_reason = (
                    _evaluate_stokes_baseline_acceptance(
                        final_div_l2=float(rec_match.get("final_div_l2", float("inf"))),
                        final_div_max=float(
                            rec_match.get("final_div_max_abs", float("inf"))
                        ),
                        outlet_inlet_ratio=float(
                            rec_match.get("outlet_inlet_flux_ratio", float("inf"))
                        ),
                        net_boundary_flux_relative=float(
                            rec_match.get("net_boundary_flux_relative", float("inf"))
                        ),
                        wall_flux_max_abs=float(
                            rec_match.get("wall_flux_max_abs_after", float("inf"))
                        ),
                        damage_ratio_l2=float(
                            rec_match.get("damage_ratio_vs_noop_l2", float("inf"))
                        ),
                        damage_ratio_linf=float(
                            rec_match.get("damage_ratio_vs_noop_linf", float("inf"))
                        ),
                        finite=bool(rec_match.get("finite_fields_final", False)),
                    )
                )
                ready_for_convective_prototype = bool(
                    stokes_baseline_accepted
                    and bool(rec_match.get("stokes_ready_for_advection_term", False))
                )
            stokes_baseline_sensitivity_sweep = {
                "flow_mode": "stokes_viscous_projection",
                "viscous_predictor_mode": str(best_mode),
                "rows": sweep_rows,
                "recommended_stokes_baseline_config": recommended_stokes_baseline_config,
                "stokes_baseline_accepted": bool(stokes_baseline_accepted),
                "stokes_baseline_acceptance_reason": str(
                    stokes_baseline_acceptance_reason
                ),
                "ready_for_convective_prototype": bool(ready_for_convective_prototype),
            }
            _write_json(
                run_dir / "stokes_baseline_sensitivity_sweep.json",
                stokes_baseline_sensitivity_sweep,
            )
            _write_csv(run_dir / "stokes_baseline_sensitivity_sweep.csv", sweep_rows)

        if bool(args.run_convective_sensitivity_sweep):
            sweep_flow_dts_conv = _parse_float_list(str(args.convective_sweep_flow_dts))
            if not sweep_flow_dts_conv:
                sweep_flow_dts_conv = [1e-4, 2.5e-4, 5e-4, 1e-3]
            sweep_dampings_conv = _parse_float_list(str(args.convective_sweep_dampings))
            if not sweep_dampings_conv:
                sweep_dampings_conv = [1.0, 0.5, 0.25]
            sweep_flow_dts_conv = [float(x) for x in dict.fromkeys(sweep_flow_dts_conv)]
            sweep_dampings_conv = [float(x) for x in dict.fromkeys(sweep_dampings_conv)]
            sweep_steps_conv = int(flow_steps_requested)
            if sweep_steps_conv > 20:
                sweep_steps_conv = 20
            if sweep_steps_conv < 10:
                sweep_steps_conv = 10

            def _dt_key(dt_value: float) -> str:
                return f"{float(dt_value):.12g}"

            def _run_stokes_reference_for_dt(dt_variant: float) -> dict[str, Any]:
                cfg_st = replace(
                    cfg,
                    projection_dt=float(dt_variant),
                    viscous_predictor_mode=str(progression_cfg.viscous_predictor_mode),  # type: ignore[arg-type]
                    viscous_face_flux_divergence_impact_cap=float(
                        progression_cfg.viscous_face_flux_divergence_impact_cap
                    ),
                )
                s_curr = initialize_tetra_flow_state(mesh, cfg_st)
                hist_st: list[dict[str, Any]] = []
                for step_idx in range(1, sweep_steps_conv + 1):
                    s_pred_st = apply_tetra_stokes_viscous_predictor(
                        mesh, s_curr, cfg_st, flow_dt=float(dt_variant)
                    )
                    s_next_st = solve_tetra_pressure_projection(mesh, s_pred_st, cfg_st)
                    d_st = dict(s_next_st.diagnostics)
                    p_st = dict(d_st.get("projection", {}))
                    pr_st = dict(d_st.get("pressure", {}))
                    be_st = dict(d_st.get("backend_execution", {}))
                    acc_st = _projection_acceptance(
                        projection=p_st,
                        pressure=pr_st,
                        backend_execution=be_st,
                        thresholds=acceptance_thresholds_cli,
                    )
                    inlet_st = float(p_st.get("inlet_flux_total_after", 0.0))
                    outlet_st = float(p_st.get("outlet_flux_total_after", 0.0))
                    hist_st.append(
                        {
                            "step": int(step_idx),
                            "projection_solved": bool(
                                acc_st.get("projection_solved", False)
                            ),
                            "finite_fields": bool(
                                acc_st.get("checklist", {}).get("finite_fields", False)
                            ),
                            "final_divergence_max_abs": float(
                                p_st.get("final_divergence_max_abs", 0.0)
                            ),
                            "final_divergence_l2": float(
                                p_st.get("final_divergence_l2", 0.0)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                outlet_st / max(abs(inlet_st), 1e-30)
                            ),
                            "net_boundary_flux_relative": float(
                                abs(float(p_st.get("net_boundary_flux_after", 0.0)))
                                / max(abs(inlet_st), 1e-30)
                            ),
                            "wall_flux_max_abs_after": float(
                                p_st.get("wall_flux_max_abs_after", 0.0)
                            ),
                        }
                    )
                    s_curr = s_next_st
                acc_hist_st = _evaluate_flow_progression_acceptance(
                    history=hist_st,
                    allow_projection_warning_steps=int(
                        args.allow_projection_warning_steps
                    ),
                    startup_warning_steps=int(startup_warning_steps_allowed),
                    outlet_inlet_flux_ratio_tolerance=float(
                        args.outlet_inlet_flux_ratio_tolerance
                    ),
                    wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                )
                final_st = hist_st[-1] if hist_st else {}
                return {
                    "final_divergence_l2": float(
                        final_st.get("final_divergence_l2", float("inf"))
                    ),
                    "final_divergence_max_abs": float(
                        final_st.get("final_divergence_max_abs", float("inf"))
                    ),
                    "flow_progression_solved": bool(
                        acc_hist_st.get("flow_progression_solved", False)
                    ),
                }

            stokes_ref_by_dt: dict[str, dict[str, Any]] = {}
            for dt_candidate in sweep_flow_dts_conv:
                stokes_ref_by_dt[_dt_key(dt_candidate)] = _run_stokes_reference_for_dt(
                    float(dt_candidate)
                )

            sweep_rows_conv: list[dict[str, Any]] = []
            for dt_candidate in sweep_flow_dts_conv:
                for damping_candidate in sweep_dampings_conv:
                    cfg_conv = replace(
                        cfg,
                        projection_dt=float(dt_candidate),
                        enable_convective_predictor=True,
                        disable_convective_predictor=False,
                        convective_cfl_limit=float(args.convective_cfl_limit),
                        convective_predictor_damping=float(damping_candidate),
                        viscous_predictor_mode=str(
                            progression_cfg.viscous_predictor_mode
                        ),  # type: ignore[arg-type]
                        viscous_face_flux_divergence_impact_cap=float(
                            progression_cfg.viscous_face_flux_divergence_impact_cap
                        ),
                    )
                    s_curr = initialize_tetra_flow_state(mesh, cfg_conv)
                    hist_conv: list[dict[str, Any]] = []
                    t0_conv = perf_counter()
                    for step_idx in range(1, sweep_steps_conv + 1):
                        s_conv = apply_tetra_convective_predictor(
                            mesh, s_curr, cfg_conv, flow_dt=float(dt_candidate)
                        )
                        conv_diag = dict(
                            s_conv.diagnostics.get("convective_predictor", {})
                        )
                        s_pred = apply_tetra_stokes_viscous_predictor(
                            mesh, s_conv, cfg_conv, flow_dt=float(dt_candidate)
                        )
                        s_next = solve_tetra_pressure_projection(mesh, s_pred, cfg_conv)
                        d = dict(s_next.diagnostics)
                        p = dict(d.get("projection", {}))
                        pr = dict(d.get("pressure", {}))
                        be = dict(d.get("backend_execution", {}))
                        v = dict(d.get("velocity", {}))
                        acc = _projection_acceptance(
                            projection=p,
                            pressure=pr,
                            backend_execution=be,
                            thresholds=acceptance_thresholds_cli,
                        )
                        inlet_after = float(p.get("inlet_flux_total_after", 0.0))
                        outlet_after = float(p.get("outlet_flux_total_after", 0.0))
                        speed = np.linalg.norm(
                            np.asarray(s_next.cell_velocity, dtype=np.float64), axis=1
                        )
                        hist_conv.append(
                            {
                                "step": int(step_idx),
                                "projection_solved": bool(
                                    acc.get("projection_solved", False)
                                ),
                                "finite_fields": bool(
                                    acc.get("checklist", {}).get("finite_fields", False)
                                ),
                                "final_divergence_max_abs": float(
                                    p.get("final_divergence_max_abs", 0.0)
                                ),
                                "final_divergence_l2": float(
                                    p.get("final_divergence_l2", 0.0)
                                ),
                                "outlet_inlet_flux_ratio": float(
                                    outlet_after / max(abs(inlet_after), 1e-30)
                                ),
                                "net_boundary_flux_relative": float(
                                    abs(float(p.get("net_boundary_flux_after", 0.0)))
                                    / max(abs(inlet_after), 1e-30)
                                ),
                                "wall_flux_max_abs_after": float(
                                    p.get("wall_flux_max_abs_after", 0.0)
                                ),
                                "velocity_magnitude_max": float(
                                    v.get("magnitude_max", 0.0)
                                ),
                                "kinetic_energy_after_projection": float(
                                    np.mean(0.5 * speed * speed)
                                )
                                if speed.size
                                else 0.0,
                                "pressure_iterations": int(
                                    pr.get("actual_iterations", 0)
                                ),
                                "convective_predictor_used": bool(
                                    conv_diag.get("convective_predictor_used", False)
                                ),
                                "convective_cfl_limit": float(
                                    conv_diag.get(
                                        "convective_cfl_limit",
                                        cfg_conv.convective_cfl_limit,
                                    )
                                ),
                                "convective_cfl_raw_max": float(
                                    conv_diag.get(
                                        "convective_cfl_raw_max",
                                        conv_diag.get("convective_cfl_max", 0.0),
                                    )
                                ),
                                "convective_cfl_raw_p95": float(
                                    conv_diag.get(
                                        "convective_cfl_raw_p95",
                                        conv_diag.get("convective_cfl_p95", 0.0),
                                    )
                                ),
                                "convective_cfl_effective_max": float(
                                    conv_diag.get(
                                        "convective_cfl_effective_max",
                                        conv_diag.get("convective_cfl_max", 0.0),
                                    )
                                ),
                                "convective_cfl_effective_p95": float(
                                    conv_diag.get(
                                        "convective_cfl_effective_p95",
                                        conv_diag.get("convective_cfl_p95", 0.0),
                                    )
                                ),
                                "convective_cfl_warning_raw": bool(
                                    conv_diag.get(
                                        "convective_cfl_warning_raw",
                                        conv_diag.get("convective_cfl_warning", False),
                                    )
                                ),
                                "convective_cfl_warning_effective": bool(
                                    conv_diag.get(
                                        "convective_cfl_warning_effective",
                                        conv_diag.get("convective_cfl_warning", False),
                                    )
                                ),
                                "convective_predictor_damping_requested": float(
                                    conv_diag.get(
                                        "convective_predictor_damping_requested",
                                        conv_diag.get(
                                            "convective_predictor_damping",
                                            cfg_conv.convective_predictor_damping,
                                        ),
                                    )
                                ),
                                "convective_predictor_damping_effective": float(
                                    conv_diag.get(
                                        "convective_predictor_damping_effective", 0.0
                                    )
                                ),
                                "convective_auto_damping_used": bool(
                                    conv_diag.get("convective_auto_damping_used", False)
                                ),
                                "convective_auto_damping_reason": str(
                                    conv_diag.get("convective_auto_damping_reason", "")
                                ),
                                "convective_dt_effective": float(
                                    conv_diag.get("convective_dt_effective", 0.0)
                                ),
                                "convective_delta_velocity_max": float(
                                    conv_diag.get("convective_delta_velocity_max", 0.0)
                                ),
                                "convective_delta_velocity_l2": float(
                                    conv_diag.get("convective_delta_velocity_l2", 0.0)
                                ),
                                "kinetic_energy_before_convection": float(
                                    conv_diag.get(
                                        "kinetic_energy_before_convection", 0.0
                                    )
                                ),
                                "kinetic_energy_after_convection": float(
                                    conv_diag.get(
                                        "kinetic_energy_after_convection", 0.0
                                    )
                                ),
                                "divergence_after_convection_before_projection_max": float(
                                    conv_diag.get(
                                        "divergence_after_convection_before_projection_max",
                                        0.0,
                                    )
                                ),
                                "divergence_after_convection_before_projection_l2": float(
                                    conv_diag.get(
                                        "divergence_after_convection_before_projection_l2",
                                        0.0,
                                    )
                                ),
                            }
                        )
                        s_curr = s_next

                    runtime_conv = perf_counter() - t0_conv
                    acc_hist_conv = _evaluate_flow_progression_acceptance(
                        history=hist_conv,
                        allow_projection_warning_steps=int(
                            args.allow_projection_warning_steps
                        ),
                        startup_warning_steps=int(startup_warning_steps_allowed),
                        outlet_inlet_flux_ratio_tolerance=float(
                            args.outlet_inlet_flux_ratio_tolerance
                        ),
                        wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                    )
                    stokes_ref_variant = stokes_ref_by_dt.get(
                        _dt_key(dt_candidate), None
                    )
                    conv_acc = _evaluate_convective_prototype_acceptance(
                        history=hist_conv,
                        stokes_baseline=stokes_ref_variant,
                        outlet_inlet_flux_ratio_tolerance=float(
                            args.outlet_inlet_flux_ratio_tolerance
                        ),
                        wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                        inlet_speed=float(args.inlet_speed),
                    )
                    readiness_conv = _evaluate_convective_readiness(
                        convective_prototype_accepted=bool(
                            conv_acc.get("convective_prototype_accepted", False)
                        ),
                        convective_prototype_acceptance_reason=str(
                            conv_acc.get("reason", "")
                        ),
                        history=hist_conv,
                        stokes_baseline=stokes_ref_variant,
                        outlet_inlet_flux_ratio_tolerance=float(
                            args.outlet_inlet_flux_ratio_tolerance
                        ),
                        wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
                        max_convective_substeps=int(args.max_convective_substeps),
                        flow_dt_mode=str(args.flow_dt_mode),
                        convective_cfl_target=float(args.convective_cfl_target),
                        convective_cfl_acceptance_eps=float(
                            args.convective_cfl_acceptance_eps
                        ),
                        inlet_speed=float(args.inlet_speed),
                    )
                    conv_stats = _convective_history_stats(
                        hist_conv,
                        convective_cfl_acceptance_eps=float(
                            args.convective_cfl_acceptance_eps
                        ),
                    )
                    final_conv = hist_conv[-1] if hist_conv else {}
                    pressure_iters_conv = [
                        int(r.get("pressure_iterations", 0))
                        for r in hist_conv
                        if int(r.get("pressure_iterations", 0)) > 0
                    ]
                    div_vs_stokes_l2 = float(
                        final_conv.get("final_divergence_l2", float("inf"))
                    ) / max(
                        float(
                            (stokes_ref_variant or {}).get("final_divergence_l2", 1e-30)
                        ),
                        1e-30,
                    )
                    div_vs_stokes_linf = float(
                        final_conv.get("final_divergence_max_abs", float("inf"))
                    ) / max(
                        float(
                            (stokes_ref_variant or {}).get(
                                "final_divergence_max_abs", 1e-30
                            )
                        ),
                        1e-30,
                    )
                    sweep_rows_conv.append(
                        {
                            "flow_dt": float(dt_candidate),
                            "flow_steps": int(sweep_steps_conv),
                            "requested_damping": float(damping_candidate),
                            "convective_cfl_limit": float(args.convective_cfl_limit),
                            "min_effective_damping": float(
                                conv_stats.get("damping_effective_min", 0.0)
                            ),
                            "mean_effective_damping": float(
                                conv_stats.get("damping_effective_mean", 0.0)
                            ),
                            "max_effective_damping": float(
                                conv_stats.get("damping_effective_max", 0.0)
                            ),
                            "raw_cfl_max": float(
                                conv_stats.get("raw_cfl_max_max", 0.0)
                            ),
                            "raw_cfl_p95": float(
                                conv_stats.get("raw_cfl_p95_max", 0.0)
                            ),
                            "effective_cfl_max": float(
                                conv_stats.get("effective_cfl_max_max", 0.0)
                            ),
                            "effective_cfl_p95": float(
                                conv_stats.get("effective_cfl_p95_max", 0.0)
                            ),
                            "final_div_l2": float(
                                final_conv.get("final_divergence_l2", 0.0)
                            ),
                            "final_div_max_abs": float(
                                final_conv.get("final_divergence_max_abs", 0.0)
                            ),
                            "outlet_inlet_flux_ratio": float(
                                final_conv.get("outlet_inlet_flux_ratio", 0.0)
                            ),
                            "net_boundary_flux_relative": float(
                                final_conv.get("net_boundary_flux_relative", 0.0)
                            ),
                            "wall_flux_max_abs_after": float(
                                final_conv.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "wall_flux": float(
                                final_conv.get("wall_flux_max_abs_after", 0.0)
                            ),
                            "velocity_max": float(
                                final_conv.get("velocity_magnitude_max", 0.0)
                            ),
                            "kinetic_energy_final": float(
                                final_conv.get("kinetic_energy_after_projection", 0.0)
                            ),
                            "flow_progression_solved": bool(
                                acc_hist_conv.get("flow_progression_solved", False)
                            ),
                            "convective_prototype_accepted": bool(
                                conv_acc.get("convective_prototype_accepted", False)
                            ),
                            "ready_for_long_ns_run_debug": bool(
                                readiness_conv.get("ready_for_long_ns_run_debug", False)
                            ),
                            "ready_for_long_ns_run_physical": bool(
                                readiness_conv.get(
                                    "ready_for_long_ns_run_physical", False
                                )
                            ),
                            "divergence_vs_stokes_l2_ratio": float(div_vs_stokes_l2),
                            "divergence_vs_stokes_max_ratio": float(div_vs_stokes_linf),
                            "pressure_iterations_mean": float(
                                np.mean(pressure_iters_conv)
                            )
                            if pressure_iters_conv
                            else 0.0,
                            "pressure_iterations_max": int(np.max(pressure_iters_conv))
                            if pressure_iters_conv
                            else 0,
                            "runtime_seconds": float(runtime_conv),
                            "steps_per_second": float(
                                sweep_steps_conv / max(runtime_conv, 1e-12)
                            ),
                        }
                    )

            stokes_ref_main_dt = stokes_ref_by_dt.get(_dt_key(flow_dt), None)
            recommended_convective_debug_config = _recommend_convective_debug_config(
                sweep_rows_conv,
                stokes_baseline=stokes_ref_main_dt,
                divergence_vs_stokes_factor_limit=100.0,
            )
            convective_sensitivity_sweep = {
                "flow_mode": "navier_stokes_projection_debug",
                "flow_steps": int(sweep_steps_conv),
                "convective_cfl_limit": float(args.convective_cfl_limit),
                "rows": sweep_rows_conv,
                "stokes_reference_by_flow_dt": stokes_ref_by_dt,
                "recommended_convective_debug_config": recommended_convective_debug_config,
            }
            _write_json(
                run_dir / "convective_sensitivity_sweep.json",
                convective_sensitivity_sweep,
            )
            _write_csv(run_dir / "convective_sensitivity_sweep.csv", sweep_rows_conv)

        if not stokes_baseline_sensitivity_sweep:
            projection_main = dict(diag.get("projection", {}))
            inlet_main = float(projection_main.get("inlet_flux_total_after", 0.0))
            outlet_main = float(projection_main.get("outlet_flux_total_after", 0.0))
            net_main = float(projection_main.get("net_boundary_flux_after", 0.0))
            stokes_baseline_accepted, stokes_baseline_acceptance_reason = (
                _evaluate_stokes_baseline_acceptance(
                    final_div_l2=float(
                        projection_main.get("final_divergence_l2", float("inf"))
                    ),
                    final_div_max=float(
                        projection_main.get("final_divergence_max_abs", float("inf"))
                    ),
                    outlet_inlet_ratio=float(outlet_main / max(abs(inlet_main), 1e-30)),
                    net_boundary_flux_relative=float(
                        abs(net_main) / max(abs(inlet_main), 1e-30)
                    ),
                    wall_flux_max_abs=float(
                        projection_main.get("wall_flux_max_abs_after", float("inf"))
                    ),
                    damage_ratio_l2=float(viscous_predictor_damage_ratio_l2),
                    damage_ratio_linf=float(viscous_predictor_damage_ratio_linf),
                    finite=bool(
                        np.all(
                            np.isfinite(np.asarray(state1.pressure, dtype=np.float64))
                        )
                        and np.all(
                            np.isfinite(np.asarray(state1.face_flux, dtype=np.float64))
                        )
                        and np.all(
                            np.isfinite(
                                np.asarray(state1.cell_velocity, dtype=np.float64)
                            )
                        )
                    ),
                )
            )
            recommended_stokes_baseline_config = {
                "flow_dt": float(flow_dt),
                "viscous_face_flux_divergence_impact_cap": float(
                    args.viscous_face_flux_divergence_impact_cap
                ),
                "resolved_cap_value": float(
                    args.viscous_face_flux_divergence_impact_cap
                ),
                "viscous_predictor_mode": str(resolved_viscous_predictor_mode),
                "viscous_predictor_outlet_contract_mode": str(
                    args.viscous_predictor_outlet_contract_mode
                ),
                "viscous_nonorthogonal_correction_mode": str(
                    args.viscous_nonorthogonal_correction_mode
                ),
                "reason": "current run configuration",
            }
            ready_for_convective_prototype = bool(
                stokes_baseline_accepted and bool(stokes_ready_for_advection_term)
            )
        viscous_predictor_audit = _build_viscous_predictor_audit_from_history(
            progression_history
        )
        navier_stokes_prototype_audit = _build_navier_stokes_prototype_audit(
            progression_history,
            convective_cfl_acceptance_eps=float(args.convective_cfl_acceptance_eps),
        )
        conv_stats_main = _convective_history_stats(
            progression_history,
            convective_cfl_acceptance_eps=float(args.convective_cfl_acceptance_eps),
        )
        used_dt_min = (
            float(np.min(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        used_dt_mean = (
            float(np.mean(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        used_dt_max = (
            float(np.max(np.asarray(used_dt_values, dtype=np.float64)))
            if used_dt_values
            else 0.0
        )
        auto_dt_scale_min = (
            float(np.min(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        auto_dt_scale_mean = (
            float(np.mean(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        auto_dt_scale_max = (
            float(np.max(np.asarray(auto_dt_scale_values, dtype=np.float64)))
            if auto_dt_scale_values
            else 0.0
        )
        raw_cfl_after_dt_selection_max = float(
            conv_stats_main.get(
                "raw_cfl_after_dt_selection_max",
                conv_stats_main.get("raw_cfl_max_max", 0.0),
            )
        )
        raw_cfl_after_dt_selection_p95 = float(
            conv_stats_main.get(
                "raw_cfl_after_dt_selection_p95",
                conv_stats_main.get("raw_cfl_p95_max", 0.0),
            )
        )
        ns_auto_dt_accepted = bool(
            (flow_mode == "navier_stokes_projection_debug")
            and (flow_dt_mode == "auto_cfl")
            and bool(ready_for_long_ns_run_physical)
        )
        if ns_auto_dt_accepted and (used_dt_mean < 0.1 * max(flow_dt_max, 1e-30)):
            ns_auto_dt_warning = "physical-ready but expensive"
        elif bool(auto_dt_min_hit_any):
            ns_auto_dt_warning = "auto dt min bound was hit on one or more steps"
        for _row in progression_history:
            _row["ready_for_long_ns_run_physical"] = bool(
                ready_for_long_ns_run_physical
            )
            _row["ready_for_long_ns_run_debug"] = bool(ready_for_long_ns_run_debug)
        _write_json(
            run_dir / "startup_bootstrap_history.json",
            {
                "steps": startup_bootstrap_history,
                "bootstrap_summary": startup_root_cause_report,
            },
        )
        _write_json(
            run_dir / "startup_root_cause_report.json",
            startup_root_cause_report,
        )
        _write_json(
            run_dir / "flow_progression_history.json",
            {
                "steps": progression_history,
                "flow_dt_mode": str(flow_dt_mode),
                "convective_cfl_acceptance_eps": float(
                    args.convective_cfl_acceptance_eps
                ),
                "effective_cfl_limit_excess_max": float(
                    conv_stats_main.get("effective_cfl_limit_excess_max", 0.0)
                ),
                "effective_cfl_warning_steps": list(
                    conv_stats_main.get("effective_cfl_warning_steps", [])
                ),
                "raw_cfl_after_dt_selection_warning_steps": list(
                    conv_stats_main.get("raw_cfl_after_dt_selection_warning_steps", [])
                ),
                "auto_dt_floor_min": float(auto_dt_floor_min),
                "auto_dt_floor_mean": float(auto_dt_floor_mean),
                "auto_dt_floor_max": float(auto_dt_floor_max),
                "auto_cfl_no_damping_readiness_reason": str(
                    auto_cfl_no_damping_readiness_reason
                ),
            },
        )
        _write_json(run_dir / "viscous_predictor_audit.json", viscous_predictor_audit)
        _write_json(
            run_dir / "navier_stokes_prototype_audit.json",
            navier_stokes_prototype_audit,
        )
        convective_cfl_audit_written = False
        if bool(args.audit_convective_cfl) and bool(convective_cfl_audit_peak):
            top_cells_payload = dict(
                convective_cfl_audit_peak.get("top_convective_cfl_cells", {})
            )
            top_faces_payload = dict(
                convective_cfl_audit_peak.get("top_convective_cfl_faces", {})
            )
            cfl_def_payload = dict(
                convective_cfl_audit_peak.get("convective_cfl_definition_report", {})
            )
            top_cells_payload["peak_step"] = int(convective_cfl_audit_peak_step)
            top_cells_payload["peak_raw_cfl_max"] = float(
                convective_cfl_audit_peak.get(
                    "convective_cfl_raw_max",
                    convective_cfl_audit_peak.get("convective_cfl_max", 0.0),
                )
            )
            top_faces_payload["peak_step"] = int(convective_cfl_audit_peak_step)
            cfl_def_payload["peak_step"] = int(convective_cfl_audit_peak_step)
            _write_json(run_dir / "top_convective_cfl_cells.json", top_cells_payload)
            _write_json(run_dir / "top_convective_cfl_faces.json", top_faces_payload)
            _write_json(
                run_dir / "convective_cfl_definition_report.json", cfl_def_payload
            )
            convective_cfl_audit_written = True
        recommended_pressure_solver = str(cfg.pressure_solver)
        if pressure_solver_comparison:
            recommended_pressure_solver = str(
                pressure_solver_comparison.get(
                    "recommended_pressure_solver", cfg.pressure_solver
                )
            )
        projection_baseline_acceptance_comparison: dict[str, Any] = {}
        if bool(args.compare_pressure_solvers):
            rows_cmp: list[dict[str, Any]] = []
            for item in list(pressure_solver_comparison.get("solvers", [])):
                lbl = str(item.get("label", ""))
                if lbl not in {"jacobi_2000", "pcg_diag_1000"}:
                    continue
                rows_cmp.append(
                    {
                        "label": lbl,
                        "pressure_solver": str(item.get("pressure_solver", "")),
                        "pressure_linear_accepted": bool(
                            item.get("pressure_linear_accepted", False)
                        ),
                        "projection_accepted": bool(
                            item.get("projection_accepted", False)
                        ),
                        "projection_solved": bool(item.get("projection_solved", False)),
                        "final_div_max_abs": float(item.get("final_div_max_abs", 0.0)),
                        "final_div_l2": float(item.get("final_div_l2", 0.0)),
                        "divergence_reduction_ratio_linf": float(
                            item.get("divergence_reduction_ratio_linf", 0.0)
                        ),
                        "divergence_reduction_ratio_l2": float(
                            item.get("divergence_reduction_ratio_l2", 0.0)
                        ),
                        "outlet_flux_after": float(item.get("outlet_flux_after", 0.0)),
                        "inlet_flux_after": float(item.get("inlet_flux_after", 0.0)),
                        "outlet_inlet_flux_ratio": float(
                            item.get("outlet_inlet_flux_ratio", 0.0)
                        ),
                        "net_boundary_flux_after": float(
                            item.get("net_boundary_flux_after", 0.0)
                        ),
                        "net_boundary_flux_relative": float(
                            item.get("net_boundary_flux_relative", 0.0)
                        ),
                        "wall_flux_max_abs_after": float(
                            item.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "runtime_seconds": float(item.get("runtime_seconds", 0.0)),
                    }
                )
            for ref_item in list(
                pressure_reference_solver_comparison.get("reference_solvers", [])
            ):
                if str(ref_item.get("solver", "")) != "scipy_gmres":
                    continue
                inlet_flux_ref = float(ref_item.get("inlet_flux_after", 0.0))
                outlet_flux_ref = float(ref_item.get("outlet_flux_after", 0.0))
                net_ref = float(ref_item.get("net_boundary_flux_after", 0.0))
                out_in_ratio_ref = outlet_flux_ref / max(abs(inlet_flux_ref), 1e-30)
                net_rel_ref = abs(net_ref) / max(abs(inlet_flux_ref), 1e-30)
                p_metrics_ref = {
                    "final_divergence_l2": float(
                        ref_item.get("final_div_l2", float("inf"))
                    ),
                    "final_divergence_max_abs": float(
                        ref_item.get("final_div_max_abs", float("inf"))
                    ),
                    "divergence_reduction_ratio_l2": float(
                        ref_item.get("final_div_l2", float("inf"))
                    )
                    / max(float(projection.get("initial_divergence_l2", 1.0)), 1e-30),
                    "divergence_reduction_ratio": float(
                        ref_item.get("final_div_max_abs", float("inf"))
                    )
                    / max(
                        float(projection.get("initial_divergence_max_abs", 1.0)), 1e-30
                    ),
                    "inlet_flux_total_after": inlet_flux_ref,
                    "outlet_flux_total_after": outlet_flux_ref,
                    "net_boundary_flux_after": net_ref,
                    "wall_flux_max_abs_after": float(
                        ref_item.get("wall_flux_max_abs_after", 0.0)
                    ),
                }
                p_ref_acc = {
                    "residual_ratio_to_rhs_l2": float(
                        ref_item.get("residual_ratio_to_rhs_l2", float("inf"))
                    ),
                    "residual_ratio_to_rhs_max": float(
                        ref_item.get("residual_ratio_to_rhs_max", float("inf"))
                    ),
                    "stopping_reason": "scipy_converged"
                    if bool(ref_item.get("converged", False))
                    else "scipy_not_converged",
                }
                acc_ref = _projection_acceptance(
                    projection=p_metrics_ref,
                    pressure=p_ref_acc,
                    backend_execution=backend_exec,
                    thresholds=acceptance_thresholds_cli,
                )
                rows_cmp.append(
                    {
                        "label": "scipy_gmres_reference",
                        "pressure_solver": "scipy_gmres",
                        "pressure_linear_accepted": bool(
                            acc_ref.get("pressure_linear_accepted", False)
                        ),
                        "projection_accepted": bool(
                            acc_ref.get("projection_accepted", False)
                        ),
                        "projection_solved": bool(
                            acc_ref.get("projection_solved", False)
                        ),
                        "final_div_max_abs": float(
                            ref_item.get("final_div_max_abs", 0.0)
                        ),
                        "final_div_l2": float(ref_item.get("final_div_l2", 0.0)),
                        "divergence_reduction_ratio_linf": float(
                            p_metrics_ref["divergence_reduction_ratio"]
                        ),
                        "divergence_reduction_ratio_l2": float(
                            p_metrics_ref["divergence_reduction_ratio_l2"]
                        ),
                        "outlet_flux_after": float(outlet_flux_ref),
                        "inlet_flux_after": float(inlet_flux_ref),
                        "outlet_inlet_flux_ratio": float(out_in_ratio_ref),
                        "net_boundary_flux_after": float(net_ref),
                        "net_boundary_flux_relative": float(net_rel_ref),
                        "wall_flux_max_abs_after": float(
                            ref_item.get("wall_flux_max_abs_after", 0.0)
                        ),
                        "runtime_seconds": float(ref_item.get("runtime_seconds", 0.0)),
                    }
                )
            projection_baseline_acceptance_comparison = {"items": rows_cmp}
            _write_json(
                run_dir / "projection_baseline_acceptance_comparison.json",
                projection_baseline_acceptance_comparison,
            )
        pressure_solver_default_reason = (
            "pcg_diag selected after operator audit; Jacobi is diagnostic fallback"
            if (not pressure_solver_explicit)
            else "pressure solver set explicitly by CLI"
        )
        ns_coupling_audit = _evaluate_ns_coupling_readiness(
            flow_progression_solved=bool(flow_progression_solved),
            ready_for_long_ns_run_physical=bool(ready_for_long_ns_run_physical),
            nonphysical_flux_fix_used=bool(
                boundary_flux_policy_audit.get("nonphysical_flux_fix_used", False)
            ),
            convective_auto_damping_used_any=bool(
                stabilization_audit.get("convective_auto_damping_used_any", False)
            ),
            convective_substep_cap_hit_any=bool(
                stabilization_audit.get("convective_substep_cap_hit_any", False)
            ),
            finite_fields=bool(
                flow_progression_final_metrics.get("finite_fields", False)
            ),
            wall_flux_max_abs_after=float(
                flow_progression_final_metrics.get("wall_flux_max_abs_after", 0.0)
            ),
            outlet_inlet_flux_ratio=float(
                flow_progression_final_metrics.get("outlet_inlet_flux_ratio", 0.0)
            ),
            final_div_l2=float(
                flow_progression_final_metrics.get(
                    "final_div_l2",
                    flow_progression_final_metrics.get(
                        "final_divergence_l2", float("inf")
                    ),
                )
            ),
            final_div_max_abs=float(
                flow_progression_final_metrics.get(
                    "final_div_max_abs",
                    flow_progression_final_metrics.get(
                        "final_divergence_max_abs", float("inf")
                    ),
                )
            ),
            wall_flux_abs_tolerance=float(args.wall_flux_abs_tolerance),
            outlet_inlet_flux_ratio_tolerance=float(
                args.outlet_inlet_flux_ratio_tolerance
            ),
            projection_final_div_l2_tolerance=float(
                args.projection_final_div_l2_tolerance
            ),
            projection_final_div_max_tolerance=float(
                args.projection_final_div_max_tolerance
            ),
            epsilon_aware_warning_step_count=int(
                warning_agg.get("epsilon_aware_warning_step_count", 0)
            ),
        )
        flow_run_completed = bool(
            int(len(progression_history)) >= int(flow_steps_requested)
        )
        flow_numerically_stable = bool(
            flow_progression_solved
            and bool(flow_progression_final_metrics.get("finite_fields", False))
        )
        flow_physically_ready = bool(ready_for_long_ns_run_physical)
        flow_ready_for_next_stage = bool(
            ns_coupling_audit.get("ready_for_flow_to_transport_coupling", False)
        )
        flow_ready_for_long_run = bool(flow_physically_ready)
        flow_stage_status = _build_stage_status(
            run_completed=bool(flow_run_completed),
            numerically_stable=bool(flow_numerically_stable),
            physically_ready=bool(flow_physically_ready),
            ready_for_next_stage=bool(flow_ready_for_next_stage),
            ready_for_long_run=bool(flow_ready_for_long_run),
            checks={
                "flow_progression_solved": bool(flow_progression_solved),
                "finite_fields_final": bool(
                    flow_progression_final_metrics.get("finite_fields", False)
                ),
                "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
                "ready_for_flow_to_transport_coupling": bool(
                    ns_coupling_audit.get("ready_for_flow_to_transport_coupling", False)
                ),
            },
        )
        flow_solved = bool(flow_ready_for_next_stage)
        final_div_l2_for_coupling = float(
            flow_progression_final_metrics.get(
                "final_divergence_l2",
                projection_acceptance_metrics.get("final_divergence_l2", 0.0),
            )
        )
        final_div_max_for_coupling = float(
            flow_progression_final_metrics.get(
                "final_divergence_max_abs",
                projection_acceptance_metrics.get("final_divergence_max_abs", 0.0),
            )
        )
        flow_coupling_export = _export_flow_coupling_bundle(
            run_dir=run_dir,
            mesh=mesh,
            mesh_name=mesh_stem,
            mesh_sha256=mesh_sha256,
            face_flux=face_flux_corrected,
            cell_velocity=state1.cell_velocity,
            pressure=state1.pressure,
            face_group_codes=face_group_codes,
            flow_mode=str(flow_mode),
            flow_dt_mode=str(flow_dt_mode),
            flow_steps=int(flow_steps_requested),
            physical_time_final=float(physical_time),
            run_completed=bool(flow_stage_status.get("run_completed", False)),
            numerically_stable=bool(flow_stage_status.get("numerically_stable", False)),
            physically_ready=bool(flow_stage_status.get("physically_ready", False)),
            ready_for_next_stage=bool(
                flow_stage_status.get("ready_for_next_stage", False)
            ),
            ready_for_long_run=bool(flow_stage_status.get("ready_for_long_run", False)),
            stage_status_reason=str(flow_stage_status.get("stage_status_reason", "")),
            ready_for_flow_to_transport_coupling=bool(
                ns_coupling_audit.get("ready_for_flow_to_transport_coupling", False)
            ),
            ns_baseline_physical_clean=bool(
                ns_coupling_audit.get("ns_baseline_physical_clean", False)
            ),
            outlet_flux_rescale_used=bool(
                boundary_flux_policy_audit.get("outlet_flux_rescale_used", False)
            ),
            nonphysical_flux_fix_used=bool(
                boundary_flux_policy_audit.get("nonphysical_flux_fix_used", False)
            ),
            convective_auto_damping_used_any=bool(
                stabilization_audit.get("convective_auto_damping_used_any", False)
            ),
            wall_flux_max_abs_after=float(
                flow_progression_final_metrics.get("wall_flux_max_abs_after", 0.0)
            ),
            outlet_inlet_flux_ratio=float(
                flow_progression_final_metrics.get("outlet_inlet_flux_ratio", 0.0)
            ),
            final_div_l2=float(final_div_l2_for_coupling),
            final_div_max=float(final_div_max_for_coupling),
        )
        flow_resume_manifest = _finalize_flow_resume_manifest(
            flow_resume_manifest,
            run_dir=run_dir,
            completed_step=resume_start_step + len(progression_history),
            physical_time=physical_time,
        )

        postprocessing_seconds = float(perf_counter() - postprocessing_started_perf)
        solver_step_stats = _timing_stats(solver_step_durations)
        warm_solver_step_stats = _timing_stats(solver_step_durations[1:])
        flow_loop_step_stats = _timing_stats(flow_loop_step_durations)
        pressure_iteration_telemetry = _pressure_iteration_telemetry(
            progression_history
        )
        pressure_matvec_telemetry = _pressure_matvec_telemetry(progression_history)
        timing_environment = _best_effort_environment_metadata(
            backend_requested=str(args.backend),
            backend_selected=str(backend.selected_backend),
            flow_execution_backend_requested=str(args.flow_execution_backend),
            flow_execution_backend_selected=str(exec_backend),
            flow_execution_device_selected=str(exec_device),
            backend=backend,
            mesh=mesh,
        )
        summary = {
            "run_dir": str(run_dir),
            "original_input": original_input,
            "mesh_name": mesh_stem,
            "resolved_mesh_npz": str(mesh_npz),
            "mesh_sha256": mesh_sha256,
            "cuda_determinism": determinism_report,
            "fixed_work": fixed_work_manifest,
            "postprocessing_mode": str(args.postprocessing_mode),
            "resume_manifest": dict(flow_resume_manifest),
            "mesh_stats": {
                "node_count": int(mesh.points.shape[0]),
                "tetra_count": int(mesh.tetrahedra.shape[0]),
                "face_count": int(mesh.face_vertices.shape[0]),
                "inlet_face_count": int(mesh.inlet_faces.size),
                "outlet_face_count": int(mesh.outlet_faces.size),
                "wall_face_count": int(mesh.wall_faces.size),
            },
            "runtime_seconds": 0.0,
            "steps_per_second": 0.0,
            "steps_per_second_scope": (
                "flow loop throughput including requested snapshot output"
            ),
            "timing": {
                "schema_version": 1,
                "clock": "time.perf_counter",
                "started_at_utc": process_started_at.isoformat(),
                "finished_at_utc": None,
                "wall_total_seconds": 0.0,
                "timing_mode": str(args.timing_mode),
                "postprocessing_mode": str(args.postprocessing_mode),
                "synchronization_policy": (
                    "cuda synchronized at flow and component boundaries"
                    if synchronization_telemetry["component_boundary_synchronization"]
                    else (
                        "cuda synchronized at setup/startup/flow phase boundaries"
                        if synchronization_telemetry["cuda_active"]
                        else "no CUDA synchronization performed"
                    )
                ),
                "cuda_synchronization": synchronization_telemetry,
                "run_scope": "current process continuation only",
                "phases": {
                    "setup_seconds": float(setup_seconds),
                    "startup_bootstrap_seconds": float(startup_bootstrap_seconds),
                    "flow_stepping_seconds": float(flow_stepping_seconds),
                    "postprocessing_seconds": float(postprocessing_seconds),
                    "final_metadata_write_seconds": 0.0,
                    "unaccounted_seconds": 0.0,
                },
                "steps": {
                    "completed": int(len(progression_history)),
                    "solver_step_scope": "physical solver work before snapshot output",
                    "solver_step_first_seconds": (
                        float(solver_step_durations[0])
                        if solver_step_durations
                        else 0.0
                    ),
                    **{
                        f"solver_step_{key}": value
                        for key, value in solver_step_stats.items()
                    },
                    "warm_solver_step_mean_seconds": warm_solver_step_stats[
                        "mean_seconds"
                    ],
                    "warm_solver_step_median_seconds": warm_solver_step_stats[
                        "median_seconds"
                    ],
                    "warm_solver_step_p95_seconds": warm_solver_step_stats[
                        "p95_seconds"
                    ],
                    "flow_loop_step_scope": (
                        "complete physical loop iteration including requested snapshot output"
                    ),
                    **{
                        f"flow_loop_step_{key}": value
                        for key, value in flow_loop_step_stats.items()
                    },
                    "flow_loop_steps_per_second": 0.0,
                    "flow_loop_cell_steps_per_second": 0.0,
                },
                "components": {
                    **component_seconds,
                    "unattributed_flow_loop_seconds": 0.0,
                },
                "pressure_solver": {
                    "scope": "physical progression only; startup bootstrap excluded",
                    **pressure_iteration_telemetry,
                    **pressure_matvec_telemetry,
                },
                "environment": timing_environment,
            },
            "flow_config": {
                "numerical_profile_values": "requested",
                "effective_numerical_profile": {
                    "viscous_predictor_outlet_contract_mode": str(
                        cfg_effective.viscous_predictor_outlet_contract_mode
                    ),
                    "pressure_projection_outlet_contract_mode": str(
                        cfg_effective.pressure_projection_outlet_contract_mode
                    ),
                    "projection_cell_velocity_update_mode": str(
                        cfg_effective.projection_cell_velocity_update_mode
                    ),
                    "pressure_nonorthogonal_correction_mode": str(
                        cfg_effective.pressure_nonorthogonal_correction_mode
                    ),
                    "viscous_nonorthogonal_correction_mode": str(
                        cfg_effective.viscous_nonorthogonal_correction_mode
                    ),
                },
                "density": cfg.density,
                "kinematic_viscosity": cfg.kinematic_viscosity,
                "wall_velocity_boundary_mode": cfg.wall_velocity_boundary_mode,
                "wall_tangential_no_slip_strength": (
                    cfg.wall_tangential_no_slip_strength
                ),
                "wall_tangential_no_slip_strength_ramp_start": (
                    float(wall_strength_ramp_start)
                ),
                "wall_tangential_no_slip_strength_ramp_target": (
                    float(wall_strength_ramp_target)
                ),
                "wall_tangential_no_slip_strength_ramp_steps": (
                    int(wall_strength_ramp_steps)
                ),
                "wall_tangential_no_slip_strength_ramp_enabled": bool(
                    wall_strength_ramp_steps > 0
                ),
                "wall_tangential_shear_face_flux_enabled": bool(
                    cfg.wall_tangential_shear_face_flux_enabled
                ),
                "wall_tangential_cell_velocity_momentum_enabled": bool(
                    cfg.wall_tangential_cell_velocity_momentum_enabled
                ),
                "wall_flux_stokes_resistance_enabled": bool(
                    cfg.wall_flux_stokes_resistance_enabled
                ),
                "wall_flux_stokes_resistance_strength": float(
                    cfg.wall_flux_stokes_resistance_strength
                ),
                "inlet_speed": cfg.inlet_speed,
                "flow_steps": int(flow_steps_requested),
                "startup_bootstrap_max_steps": int(startup_bootstrap_max_steps),
                "flow_dt": float(flow_dt),
                "flow_dt_mode": str(args.flow_dt_mode),
                "flow_dt_min": float(args.flow_dt_min),
                "flow_dt_max": (
                    float(args.flow_dt_max) if (args.flow_dt_max is not None) else None
                ),
                "convective_cfl_target": float(args.convective_cfl_target),
                "flow_mode": str(flow_mode),
                "allow_projection_warning_steps": int(
                    args.allow_projection_warning_steps
                ),
                "startup_warning_steps": int(startup_warning_steps_allowed),
                "compare_flow_modes": str(args.compare_flow_modes),
                "compare_ns_dt_modes": bool(args.compare_ns_dt_modes),
                "enable_convective_predictor": bool(
                    convective_predictor_enabled_resolved
                ),
                "disable_convective_predictor": bool(args.disable_convective_predictor),
                "disable_convective_auto_damping": bool(
                    args.disable_convective_auto_damping
                ),
                "convective_predictor_default_reason": str(
                    convective_predictor_default_reason
                ),
                "convective_cfl_limit": float(args.convective_cfl_limit),
                "convective_cfl_acceptance_eps": float(
                    args.convective_cfl_acceptance_eps
                ),
                "convective_predictor_damping": float(
                    args.convective_predictor_damping
                ),
                "convective_stabilization_mode": str(
                    args.convective_stabilization_mode
                ),
                "convective_substep_boundary_contract": str(
                    args.convective_substep_boundary_contract
                ),
                "max_convective_substeps": int(args.max_convective_substeps),
                "fail_on_convective_substep_cap": bool(
                    args.fail_on_convective_substep_cap
                ),
                "audit_convective_cfl": bool(args.audit_convective_cfl),
                "compare_convective_stabilization_modes": str(
                    args.compare_convective_stabilization_modes
                ),
                "viscous_predictor_mode": str(resolved_viscous_predictor_mode),
                "viscous_predictor_outlet_contract_mode": str(
                    args.viscous_predictor_outlet_contract_mode
                ),
                "viscous_nonorthogonal_correction_mode": str(
                    cfg.viscous_nonorthogonal_correction_mode
                ),
                "viscous_predictor_default_reason": str(
                    viscous_predictor_default_reason
                ),
                "viscous_face_flux_divergence_impact_cap": float(
                    args.viscous_face_flux_divergence_impact_cap
                ),
                "viscous_face_flux_laplacian_vectorized": bool(
                    args.viscous_face_flux_laplacian_vectorized
                ),
                "torch_cuda_viscosity_enabled": bool(args.torch_cuda_viscosity_enabled),
                "compare_viscous_predictor_modes": str(
                    args.compare_viscous_predictor_modes
                ),
                "run_stokes_sensitivity_sweep": bool(args.run_stokes_sensitivity_sweep),
                "sweep_flow_dts": str(args.sweep_flow_dts),
                "sweep_viscous_face_flux_caps": str(args.sweep_viscous_face_flux_caps),
                "run_convective_sensitivity_sweep": bool(
                    args.run_convective_sensitivity_sweep
                ),
                "convective_sweep_flow_dts": str(args.convective_sweep_flow_dts),
                "convective_sweep_dampings": str(args.convective_sweep_dampings),
                "projection_dt": cfg.projection_dt,
                "projection_sign": cfg.projection_sign,
                "projection_rhs_mode": cfg.projection_rhs_mode,
                "outlet_projection_mode": str(cfg.outlet_projection_mode),
                "pressure_projection_outlet_contract_mode": str(
                    cfg.pressure_projection_outlet_contract_mode
                ),
                "pressure_nonorthogonal_correction_mode": str(
                    cfg.pressure_nonorthogonal_correction_mode
                ),
                "pressure_nonorthogonal_correction_sweeps": int(
                    cfg.pressure_nonorthogonal_correction_sweeps
                ),
                "pressure_nonorthogonal_correction_relaxation": float(
                    cfg.pressure_nonorthogonal_correction_relaxation
                ),
                "projection_correction_damping": cfg.projection_correction_damping,
                "projection_cell_velocity_update_mode": str(
                    cfg.projection_cell_velocity_update_mode
                ),
                "projection_correction_limit_mode": cfg.projection_correction_limit_mode,
                "projection_divergence_cap_factor": cfg.projection_divergence_cap_factor,
                "projection_divergence_floor": cfg.projection_divergence_floor,
                "projection_face_correction_over_volume_cap": (
                    cfg.projection_face_correction_over_volume_cap
                ),
                "pressure_outlet_value": cfg.pressure_outlet_value,
                "max_pressure_iterations": cfg.max_pressure_iterations,
                "pressure_tolerance": cfg.pressure_tolerance,
                "pressure_relative_tolerance": cfg.pressure_relative_tolerance,
                "divergence_tolerance": cfg.divergence_tolerance,
                "pressure_solver": cfg.pressure_solver,
                "pressure_solver_default_reason": pressure_solver_default_reason,
                "cg_breakdown_eps": cfg.cg_breakdown_eps,
                "cg_stagnation_window": cfg.cg_stagnation_window,
                "cg_stagnation_ratio": cfg.cg_stagnation_ratio,
                "pcg_require_relative_l2_convergence": (
                    cfg.pcg_require_relative_l2_convergence
                ),
                **acceptance_thresholds_cli,
            },
            "backend": backend.__dict__,
            "backend_execution": backend_exec,
            "numerical_profile_resolution": diag.get(
                "numerical_profile_resolution", {}
            ),
            "projection": diag.get("projection", {}),
            "pressure": diag.get("pressure", {}),
            "pressure_nonorthogonal_correction": diag.get(
                "pressure_nonorthogonal_correction", {}
            ),
            "velocity": diag.get("velocity", {}),
            "velocity_representation_note": "cell-centered velocity is derived/debug, not primary projection state",
            "velocity_region_audit": velocity_region_audit,
            "velocity_reconstruction_audit_summary": {
                "direction_agreement_fraction_global": float(
                    velocity_recon_audit.get("direction_agreement_fraction_global", 0.0)
                ),
                "direction_agreement_fraction_by_region": dict(
                    velocity_recon_audit.get(
                        "direction_agreement_fraction_by_region", {}
                    )
                ),
                "suspicious_velocity_cells_count": int(
                    velocity_recon_audit.get("suspicious_velocity_cells_count", 0)
                ),
            },
            "pressure_solver_comparison": pressure_solver_comparison,
            "projection_baseline_acceptance_comparison": projection_baseline_acceptance_comparison,
            "projection_rhs_mode_comparison": rhs_mode_comparison,
            "projection_damping_comparison": damping_comparison,
            "projection_correction_limit_comparison": correction_limit_mode_comparison,
            "flow_mode_comparison": flow_mode_comparison,
            "viscous_predictor_mode_comparison": viscous_predictor_mode_comparison,
            "audit_pressure_operator_enabled": bool(args.audit_pressure_operator),
            "pressure_matrix_explicit_audit": pressure_matrix_explicit_audit,
            "pressure_operator_matrixfree_vs_explicit": pressure_operator_matrixfree_vs_explicit,
            "pressure_operator_spd_audit": pressure_operator_spd_audit,
            "pressure_reference_solver_comparison": pressure_reference_solver_comparison,
            "recommended_pressure_solver": recommended_pressure_solver,
            "pressure_solver_default_reason": pressure_solver_default_reason,
            "outlet_mode_comparison": outlet_mode_comparison,
            "boundary_policy_comparison": boundary_policy_comparison,
            "divergence_hotspot_summary": top_divergence_summary,
            "face_flux_primary_output": {
                "description": "projection primary state is corrected face flux",
                "correction_stage_codebook": (
                    dict(diag.get("face_flux_primary_stage_codebook", {}))
                    or {
                        "correction_flux_raw_pre_constraint": (
                            "pressure correction before any boundary pinning"
                        ),
                        "correction_flux_constrained_pre_limiter": (
                            "boundary-pinned correction before limiter"
                        ),
                        "correction_flux_limiter_output_pre_reconstraint": (
                            "direct limiter output before pinned-face re-constraint"
                        ),
                        "correction_flux_constrained_post_limiter_pre_outlet_policy": (
                            "re-constrained correction after limiter and before outlet policy"
                        ),
                        "correction_flux_effective_post_outlet_policy": (
                            "final effective correction after outlet policy"
                        ),
                    }
                ),
                "face_group_codebook": {
                    "0": "interior_or_unclassified",
                    "1": "wall",
                    "2": "outlet",
                    "3": "inlet",
                },
            },
            "resume": dict(resume_metadata),
            "flow_solved": bool(flow_solved),
            "run_completed": bool(flow_stage_status.get("run_completed", False)),
            "numerically_stable": bool(
                flow_stage_status.get("numerically_stable", False)
            ),
            "physically_ready": bool(flow_stage_status.get("physically_ready", False)),
            "ready_for_next_stage": bool(
                flow_stage_status.get("ready_for_next_stage", False)
            ),
            "ready_for_long_run": bool(
                flow_stage_status.get("ready_for_long_run", False)
            ),
            "stage_status_reason": str(
                flow_stage_status.get("stage_status_reason", "")
            ),
            "stage_status_checks": dict(
                flow_stage_status.get("stage_status_checks", {})
            ),
            "flow_progression_enabled": bool(flow_progression_enabled),
            "startup_bootstrap": dict(startup_root_cause_report),
            "flow_steps_requested": int(flow_steps_requested),
            "flow_steps_completed": int(len(progression_history)),
            "flow_start_step": int(resume_start_step),
            "flow_step_end": int(resume_start_step + len(progression_history)),
            "flow_steps_completed_total": int(
                resume_start_step + len(progression_history)
            ),
            "flow_stop_physical_time": flow_stop_physical_time,
            "snapshot_time_interval": snapshot_time_interval,
            "flow_dt": float(flow_dt),
            "flow_dt_mode": str(flow_dt_mode),
            "requested_flow_dt": float(requested_flow_dt),
            "flow_dt_min": float(flow_dt_min),
            "flow_dt_max": float(flow_dt_max),
            "convective_cfl_target": float(convective_cfl_target),
            "physical_time_initial": float(resume_start_time),
            "physical_time_advanced": float(physical_time - resume_start_time),
            "physical_time_final": float(physical_time),
            "used_dt_min": float(used_dt_min),
            "used_dt_mean": float(used_dt_mean),
            "used_dt_max": float(used_dt_max),
            "auto_dt_scale_min": float(auto_dt_scale_min),
            "auto_dt_scale_mean": float(auto_dt_scale_mean),
            "auto_dt_scale_max": float(auto_dt_scale_max),
            "auto_dt_floor_min": float(auto_dt_floor_min),
            "auto_dt_floor_mean": float(auto_dt_floor_mean),
            "auto_dt_floor_max": float(auto_dt_floor_max),
            "auto_dt_min_hit_any": bool(auto_dt_min_hit_any),
            "auto_dt_max_hit_any": bool(auto_dt_max_hit_any),
            "raw_cfl_after_dt_selection_max": float(raw_cfl_after_dt_selection_max),
            "raw_cfl_after_dt_selection_p95": float(raw_cfl_after_dt_selection_p95),
            "effective_cfl_limit_excess_max": float(
                conv_stats_main.get("effective_cfl_limit_excess_max", 0.0)
            ),
            "effective_cfl_warning_steps": list(
                conv_stats_main.get("effective_cfl_warning_steps", [])
            ),
            "raw_cfl_after_dt_selection_warning_steps": list(
                conv_stats_main.get("raw_cfl_after_dt_selection_warning_steps", [])
            ),
            "flow_mode": str(flow_mode),
            "flow_progression_solved": bool(flow_progression_solved),
            "flow_progression_solved_with_startup_tolerance": bool(
                flow_prog_acc.get(
                    "flow_progression_solved_with_startup_tolerance",
                    flow_progression_solved,
                )
            ),
            "startup_warning_steps_allowed": int(
                flow_prog_acc.get("startup_warning_steps_allowed", 0)
            ),
            "startup_warning_steps_observed": list(
                flow_prog_acc.get("startup_warning_steps_observed", [])
            ),
            "nonstartup_failed_steps": list(
                flow_prog_acc.get("nonstartup_failed_steps", [])
            ),
            "flow_progression_acceptance_reason": str(flow_progression_reason),
            "flow_progression_final_metrics": flow_progression_final_metrics,
            "flow_progression_worst_step_metrics": flow_progression_worst_step_metrics,
            "viscous_progression_accepted": bool(viscous_progression_accepted),
            "viscous_progression_acceptance_reason": str(
                viscous_progression_acceptance_reason
            ),
            "viscous_progression_checks": viscous_progression_checks,
            "viscous_predictor_audit_summary": dict(
                viscous_predictor_audit.get("summary", {})
                if isinstance(viscous_predictor_audit, dict)
                else {}
            ),
            "navier_stokes_prototype_audit_summary": dict(
                navier_stokes_prototype_audit.get("summary", {})
                if isinstance(navier_stokes_prototype_audit, dict)
                else {}
            ),
            "convective_predictor_used": bool(
                any(
                    bool(r.get("convective_predictor_used", False))
                    for r in progression_history
                )
            ),
            "convective_prototype_accepted": bool(convective_prototype_accepted),
            "convective_prototype_acceptance_reason": str(
                convective_prototype_acceptance_reason
            ),
            "convective_prototype_checks": convective_prototype_checks,
            "convective_readiness_checks": convective_readiness_checks,
            "readiness_reason": str(convective_readiness_reason),
            "ready_for_long_ns_run_debug": bool(ready_for_long_ns_run_debug),
            "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
            "ready_for_long_ns_run": bool(ready_for_long_ns_run),
            "ns_auto_dt_accepted": bool(ns_auto_dt_accepted),
            "ns_auto_dt_warning": str(ns_auto_dt_warning),
            "auto_cfl_no_damping_readiness_reason": str(
                auto_cfl_no_damping_readiness_reason
            ),
            **warning_agg,
            **boundary_flux_policy_audit,
            **stabilization_audit,
            **ns_coupling_audit,
            "flow_coupling_metadata": dict(flow_coupling_export.get("metadata", {})),
            "flow_coupling_artifact_sha256": dict(
                flow_coupling_export.get("artifact_sha256", {})
            ),
            "viscous_predictor_damages_divergence": bool(
                viscous_predictor_damages_divergence
            ),
            "viscous_predictor_damage_ratio_l2": float(
                viscous_predictor_damage_ratio_l2
            ),
            "viscous_predictor_damage_ratio_linf": float(
                viscous_predictor_damage_ratio_linf
            ),
            "viscous_predictor_best_mode": str(viscous_predictor_best_mode),
            "stokes_ready_for_advection_term": bool(stokes_ready_for_advection_term),
            "stokes_baseline_accepted": bool(stokes_baseline_accepted),
            "stokes_baseline_acceptance_reason": str(stokes_baseline_acceptance_reason),
            "recommended_stokes_baseline_config": recommended_stokes_baseline_config,
            "ready_for_convective_prototype": bool(ready_for_convective_prototype),
            "stokes_baseline_sensitivity_sweep": stokes_baseline_sensitivity_sweep,
            "convective_sensitivity_sweep": convective_sensitivity_sweep,
            "convective_cfl_audit_peak_step": int(convective_cfl_audit_peak_step),
            "convective_cfl_audit_written": bool(convective_cfl_audit_written),
            "recommended_convective_debug_config": recommended_convective_debug_config,
            "pressure_solved": bool(pressure.get("pressure_solved", False)),
            "projection_limit_mode": projection_limit_mode,
            "projection_limit_experimental": bool(projection_limit_experimental),
            "projection_operator_consistent": bool(projection_operator_consistent),
            "projection_equation_consistent": bool(projection_equation_consistent),
            "projection_equation_residual_consistent": bool(
                projection_equation_residual_consistent
            ),
            "projection_equation_residual_reason": str(
                projection_equation_residual_reason
            ),
            "pressure_linear_solved_strict": bool(pressure_linear_solved_strict),
            "pressure_linear_accepted": bool(pressure_linear_accepted),
            "projection_accepted_physical": bool(projection_accepted_physical),
            "projection_accepted": bool(projection_accepted),
            "projection_solved_physical": bool(projection_solved_physical),
            "projection_solved": bool(projection_solved),
            "projection_acceptance_reason": str(projection_reason),
            "projection_acceptance_thresholds": projection_thresholds,
            "projection_acceptance_metrics": projection_acceptance_metrics,
            "projection_acceptance_checklist": projection_checklist,
            "artifacts": {
                "run_log": str(run_dir / "run.log"),
                "summary_json": str(run_dir / "summary.json"),
                "resume_manifest_json": str(run_dir / "resume_manifest.json"),
                "acceptance_report_json": str(run_dir / "acceptance_report.json"),
                "config_json": str(run_dir / "config.json"),
                "flow_diagnostics_json": str(run_dir / "flow_diagnostics.json"),
                "startup_bootstrap_history_json": str(
                    run_dir / "startup_bootstrap_history.json"
                ),
                "startup_root_cause_report_json": str(
                    run_dir / "startup_root_cause_report.json"
                ),
                "flow_progression_history_json": str(
                    run_dir / "flow_progression_history.json"
                ),
                "viscous_predictor_audit_json": str(
                    run_dir / "viscous_predictor_audit.json"
                ),
                "viscous_predictor_stage_hotspots_json": (
                    str(run_dir / "viscous_predictor_stage_hotspots.json")
                    if bool(viscous_predictor_stage_hotspots)
                    else None
                ),
                "navier_stokes_prototype_audit_json": str(
                    run_dir / "navier_stokes_prototype_audit.json"
                ),
                "projection_audit_json": str(run_dir / "projection_audit.json"),
                "pressure_solver_history_json": str(
                    run_dir / "pressure_solver_history.json"
                ),
                "boundary_flux_audit_json": str(run_dir / "boundary_flux_audit.json"),
                "region_divergence_audit_json": str(
                    run_dir / "region_divergence_audit.json"
                ),
                "operator_consistency_audit_json": str(
                    run_dir / "operator_consistency_audit.json"
                ),
                "operator_identity_audit_json": str(
                    run_dir / "operator_identity_audit.json"
                ),
                "projection_equation_residual_json": str(
                    run_dir / "projection_equation_residual.json"
                ),
                "projection_equation_residual_inconsistency_json": (
                    str(run_dir / "projection_equation_residual_inconsistency.json")
                    if bool(projection_equation_inconsistency_details)
                    else None
                ),
                "pressure_projection_scale_audit_json": str(
                    run_dir / "pressure_projection_scale_audit.json"
                ),
                "pressure_projection_hotspot_correlation_json": str(
                    run_dir / "pressure_projection_hotspot_correlation.json"
                ),
                "pressure_matrix_coefficient_audit_json": str(
                    run_dir / "pressure_matrix_coefficient_audit.json"
                ),
                "projection_volume_weighting_audit_json": str(
                    run_dir / "projection_volume_weighting_audit.json"
                ),
                "pressure_matrix_explicit_audit_json": (
                    str(run_dir / "pressure_matrix_explicit_audit.json")
                    if bool(args.audit_pressure_operator)
                    else None
                ),
                "pressure_operator_matrixfree_vs_explicit_json": (
                    str(run_dir / "pressure_operator_matrixfree_vs_explicit.json")
                    if bool(args.audit_pressure_operator)
                    else None
                ),
                "pressure_operator_spd_audit_json": (
                    str(run_dir / "pressure_operator_spd_audit.json")
                    if bool(args.audit_pressure_operator)
                    else None
                ),
                "pressure_reference_solver_comparison_json": (
                    str(run_dir / "pressure_reference_solver_comparison.json")
                    if bool(args.audit_pressure_operator)
                    else None
                ),
                "boundary_policy_comparison_json": str(
                    run_dir / "boundary_policy_comparison.json"
                ),
                "top_divergence_correction_breakdown_json": str(
                    run_dir / "top_divergence_correction_breakdown.json"
                ),
                "top_divergence_local_face_audit_json": str(
                    run_dir / "top_divergence_local_face_audit.json"
                ),
                "projection_sign_comparison_fixed_rhs_json": str(
                    run_dir / "projection_sign_comparison_fixed_rhs.json"
                ),
                "outlet_projection_audit_json": str(
                    run_dir / "outlet_projection_audit.json"
                ),
                "correction_limiter_audit_json": str(
                    run_dir / "correction_limiter_audit.json"
                ),
                "correction_limiter_conservation_audit_json": str(
                    run_dir / "correction_limiter_conservation_audit.json"
                ),
                "velocity_reconstruction_audit_json": str(
                    run_dir / "velocity_reconstruction_audit.json"
                ),
                "velocity_region_audit_json": str(
                    run_dir / "velocity_region_audit.json"
                ),
                "top_suspicious_velocity_cells_json": str(
                    run_dir / "top_suspicious_velocity_cells.json"
                ),
                "top_velocity_cells_json": str(run_dir / "top_velocity_cells.json"),
                "top_divergence_cells_json": str(run_dir / "top_divergence_cells.json"),
                "projection_hotspot_before_after_limiters_json": (
                    str(run_dir / "projection_hotspot_before_after_limiters.json")
                    if bool(args.compare_correction_limit_modes)
                    else None
                ),
                "pressure_solver_comparison_json": (
                    str(run_dir / "pressure_solver_comparison.json")
                    if bool(args.compare_pressure_solvers)
                    else None
                ),
                "projection_baseline_acceptance_comparison_json": (
                    str(run_dir / "projection_baseline_acceptance_comparison.json")
                    if bool(args.compare_pressure_solvers)
                    else None
                ),
                "projection_rhs_mode_comparison_json": (
                    str(run_dir / "projection_rhs_mode_comparison.json")
                    if bool(args.compare_pressure_solvers)
                    else None
                ),
                "projection_damping_comparison_json": (
                    str(run_dir / "projection_damping_comparison.json")
                    if bool(args.compare_pressure_solvers)
                    else None
                ),
                "projection_correction_limit_comparison_json": (
                    str(run_dir / "projection_correction_limit_comparison.json")
                    if bool(args.compare_correction_limit_modes)
                    else None
                ),
                "flow_mode_comparison_json": (
                    str(run_dir / "flow_mode_comparison.json")
                    if bool(flow_mode_comparison)
                    else None
                ),
                "viscous_predictor_mode_comparison_json": (
                    str(run_dir / "viscous_predictor_mode_comparison.json")
                    if bool(viscous_predictor_mode_comparison)
                    else None
                ),
                "stokes_baseline_sensitivity_sweep_json": (
                    str(run_dir / "stokes_baseline_sensitivity_sweep.json")
                    if bool(stokes_baseline_sensitivity_sweep)
                    else None
                ),
                "stokes_baseline_sensitivity_sweep_csv": (
                    str(run_dir / "stokes_baseline_sensitivity_sweep.csv")
                    if bool(stokes_baseline_sensitivity_sweep)
                    else None
                ),
                "convective_sensitivity_sweep_json": (
                    str(run_dir / "convective_sensitivity_sweep.json")
                    if bool(convective_sensitivity_sweep)
                    else None
                ),
                "convective_sensitivity_sweep_csv": (
                    str(run_dir / "convective_sensitivity_sweep.csv")
                    if bool(convective_sensitivity_sweep)
                    else None
                ),
                "top_convective_cfl_cells_json": (
                    str(run_dir / "top_convective_cfl_cells.json")
                    if bool(args.audit_convective_cfl)
                    else None
                ),
                "top_convective_cfl_faces_json": (
                    str(run_dir / "top_convective_cfl_faces.json")
                    if bool(args.audit_convective_cfl)
                    else None
                ),
                "convective_cfl_definition_report_json": (
                    str(run_dir / "convective_cfl_definition_report.json")
                    if bool(args.audit_convective_cfl)
                    else None
                ),
                "convective_stabilization_comparison_json": (
                    str(run_dir / "convective_stabilization_comparison.json")
                    if bool(convective_stabilization_comparison)
                    else None
                ),
                "ns_dt_mode_comparison_json": (
                    str(run_dir / "ns_dt_mode_comparison.json")
                    if bool(ns_dt_mode_comparison)
                    else None
                ),
                "outlet_mode_comparison_json": (
                    str(run_dir / "outlet_mode_comparison.json")
                    if bool(args.compare_outlet_projection_modes)
                    else None
                ),
                "pressure_xy_png": pressure_png,
                "velocity_magnitude_xy_png": vel_mag_png,
                "divergence_abs_log10_xy_png": div_png,
                "velocity_vectors_xy_downsampled_png": vec_ds_png,
                "velocity_vectors_xy_sparse_normalized_png": vec_sparse_png,
                "velocity_vectors_xy_sparse_normalized_seeded_png": vec_seeded_png,
                "velocity_vectors_xy_normalized_png": vec_norm_png,
                "velocity_vectors_xy_by_region_png": vec_region_png,
                "velocity_vectors_xy_region_panels_png": vec_region_panels_png,
                "velocity_vectors_xy_grid_binned_png": vec_grid_binned_png,
                "velocity_vectors_xy_region_panels_binned_png": vec_region_panels_binned_png,
                "velocity_vectors_xy_direction_colored_png": vec_direction_colored_png,
                "velocity_vectors_xy_raw_clipped_scale_png": vec_raw_png,
                "velocity_magnitude_p95_clipped_xy_png": vel_mag_p95_clip_png,
                "divergence_xy_step_0050_png": (
                    str(run_dir / "divergence_xy_step_0050.png")
                    if (run_dir / "divergence_xy_step_0050.png").exists()
                    else None
                ),
                "divergence_xy_step_0100_png": (
                    str(run_dir / "divergence_xy_step_0100.png")
                    if (run_dir / "divergence_xy_step_0100.png").exists()
                    else None
                ),
                "velocity_magnitude_p95_clipped_xy_step_0100_png": (
                    str(run_dir / "velocity_magnitude_p95_clipped_xy_step_0100.png")
                    if (
                        run_dir / "velocity_magnitude_p95_clipped_xy_step_0100.png"
                    ).exists()
                    else None
                ),
                "divergence_stage_comparison_step_0100_png": divergence_stage_comparison_step_0100_png,
                "velocity_magnitude_before_after_predictor_step_0100_png": (
                    velocity_magnitude_before_after_predictor_step_0100_png
                ),
                "face_flux_delta_predictor_xy_step_0100_png": face_flux_delta_predictor_xy_step_0100_png,
                "velocity_vectors_xy_grid_binned_step_0100_png": (
                    str(run_dir / "velocity_vectors_xy_grid_binned_step_0100.png")
                    if (
                        run_dir / "velocity_vectors_xy_grid_binned_step_0100.png"
                    ).exists()
                    else None
                ),
                "velocity_vectors_xy_region_panels_binned_step_0100_png": (
                    str(
                        run_dir
                        / "velocity_vectors_xy_region_panels_binned_step_0100.png"
                    )
                    if (
                        run_dir
                        / "velocity_vectors_xy_region_panels_binned_step_0100.png"
                    ).exists()
                    else None
                ),
                "outlet_flux_faces_xy_png": outlet_flux_png,
                "face_flux_stream_proxy_xy_png": face_flux_proxy_png,
                "pressure_outlet_zoom_xy_png": pressure_outlet_zoom_png,
                "divergence_hotspot_top_cells_xy_png": div_hotspot_png,
                "divergence_before_after_same_scale_png": div_before_after_png,
                "divergence_before_after_limiter_same_scale_png": div_before_after_limiter_png,
                "top_divergence_cells_before_after_limiter_xy_png": top_before_after_limiter_png,
                "correction_flux_limited_faces_xy_png": correction_flux_limited_faces_png,
                "limiter_effect_histogram_png": limiter_histogram_png,
                "pressure_operator_symmetry_hotspots_xy_png": operator_symmetry_hotspots_png,
                "pressure_operator_matrixfree_mismatch_xy_png": operator_matrixfree_mismatch_png,
                "explicit_vs_matrixfree_residual_histogram_png": explicit_vs_matrixfree_hist_png,
                "correction_flux_hotspots_xy_png": corr_hotspot_png,
                "pressure_correction_flux_magnitude_xy_png": pressure_corr_flux_png,
                "top_divergence_correction_breakdown_xy_png": top_breakdown_png,
                "boundary_policy_comparison_bar_png": boundary_policy_bar_png,
                "corrected_face_flux_npy": str(corrected_face_flux_path),
                "face_flux_star_npy": str(face_flux_star_path),
                "correction_flux_npy": str(correction_flux_path),
                "face_centers_npy": str(face_centers_path),
                "face_normals_npy": str(face_normals_path),
                "face_to_cells_npy": str(face_to_cells_path),
                "face_groups_npy": str(face_groups_path),
                "flow_coupling_metadata_json": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "flow_coupling_metadata_json", ""
                    )
                ),
                "final_corrected_face_flux_npy": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "final_corrected_face_flux_npy", ""
                    )
                ),
                "final_cell_velocity_npy": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "final_cell_velocity_npy", ""
                    )
                ),
                "final_pressure_npy": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "final_pressure_npy", ""
                    )
                ),
                "cell_centers_npy": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "cell_centers_npy", ""
                    )
                ),
                "cell_volumes_npy": str(
                    flow_coupling_export.get("artifacts", {}).get(
                        "cell_volumes_npy", ""
                    )
                ),
                "rebuilt_face_flux_from_reconstructed_velocity_npy": str(
                    rebuilt_face_flux_path
                ),
                "flux_reconstruction_mismatch_npy": str(flux_mismatch_path),
                "result_vtu": vtu_path if vtu_path else None,
            },
        }
        summary["expected_artifacts"] = _build_expected_artifacts_map(
            run_dir,
            flow_mode=str(flow_mode),
            flow_steps=int(flow_steps_requested),
            compare_viscous_predictor_modes=bool(compare_visc_modes),
            compare_flow_modes=bool(compare_modes),
            compare_pressure_solvers=bool(args.compare_pressure_solvers),
            run_stokes_sensitivity_sweep=bool(args.run_stokes_sensitivity_sweep),
            run_convective_sensitivity_sweep=bool(
                args.run_convective_sensitivity_sweep
            ),
            audit_convective_cfl=bool(args.audit_convective_cfl),
            compare_convective_stabilization_modes=bool(compare_conv_stab_modes),
            compare_ns_dt_modes=bool(args.compare_ns_dt_modes),
            postprocessing_mode=str(args.postprocessing_mode),
        )
        acceptance_report = {
            "run_dir": str(run_dir),
            "mesh_name": mesh_stem,
            "backend_execution": backend_exec,
            "pressure_solver": str(cfg.pressure_solver),
            "pressure_nonorthogonal_correction_mode_requested": str(
                cfg.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_mode": str(
                cfg_effective.pressure_nonorthogonal_correction_mode
            ),
            "pressure_nonorthogonal_correction_sweeps": int(
                cfg.pressure_nonorthogonal_correction_sweeps
            ),
            "pressure_nonorthogonal_correction_relaxation": float(
                cfg.pressure_nonorthogonal_correction_relaxation
            ),
            "pressure_nonorthogonal_outer_defect_relative_l2": float(
                dict(diag.get("pressure_nonorthogonal_correction", {})).get(
                    "outer_fixed_point_defect_relative_l2", 0.0
                )
            ),
            "pressure_stopping_reason": str(pressure.get("stopping_reason", "")),
            "pressure_linear_solved_strict": bool(pressure_linear_solved_strict),
            "pressure_linear_accepted": bool(pressure_linear_accepted),
            "projection_accepted_physical": bool(projection_accepted_physical),
            "projection_accepted": bool(projection_accepted),
            "projection_equation_residual_consistent": bool(
                projection_equation_residual_consistent
            ),
            "projection_equation_residual_reason": str(
                projection_equation_residual_reason
            ),
            "projection_equation_best_scaling": str(
                projection_equation_residual.get("projection_equation_best_scaling", "")
            ),
            "projection_equation_best_sign": str(
                projection_equation_residual.get("projection_equation_best_sign", "")
            ),
            "projection_equation_best_relative_l2": float(
                projection_equation_residual.get(
                    "projection_equation_best_relative_l2", 0.0
                )
            ),
            "projection_solved": bool(projection_solved),
            "projection_acceptance_reason": str(projection_reason),
            "projection_acceptance_thresholds": projection_thresholds,
            "projection_blocking_checks": dict(acc_main.get("blocking_checks", {})),
            "projection_failed_criteria": _projection_failed_criteria(acc_main),
            "strict_diagnostic_criteria_not_met": (
                _projection_strict_diagnostic_criteria_not_met(acc_main)
            ),
            "projection_acceptance_metrics": projection_acceptance_metrics,
            "checklist": projection_checklist,
            "flow_progression_enabled": bool(flow_progression_enabled),
            "startup_bootstrap": dict(startup_root_cause_report),
            "resume": dict(resume_metadata),
            "flow_steps_requested": int(flow_steps_requested),
            "flow_steps_completed": int(len(progression_history)),
            "flow_start_step": int(resume_start_step),
            "flow_step_end": int(resume_start_step + flow_steps_requested),
            "flow_steps_completed_total": int(
                resume_start_step + len(progression_history)
            ),
            "flow_dt": float(flow_dt),
            "flow_dt_mode": str(flow_dt_mode),
            "requested_flow_dt": float(requested_flow_dt),
            "flow_dt_min": float(flow_dt_min),
            "flow_dt_max": float(flow_dt_max),
            "convective_cfl_target": float(convective_cfl_target),
            "physical_time_initial": float(resume_start_time),
            "physical_time_advanced": float(physical_time - resume_start_time),
            "physical_time_final": float(physical_time),
            "used_dt_min": float(used_dt_min),
            "used_dt_mean": float(used_dt_mean),
            "used_dt_max": float(used_dt_max),
            "auto_dt_scale_min": float(auto_dt_scale_min),
            "auto_dt_scale_mean": float(auto_dt_scale_mean),
            "auto_dt_scale_max": float(auto_dt_scale_max),
            "auto_dt_min_hit_any": bool(auto_dt_min_hit_any),
            "auto_dt_max_hit_any": bool(auto_dt_max_hit_any),
            "raw_cfl_after_dt_selection_max": float(raw_cfl_after_dt_selection_max),
            "raw_cfl_after_dt_selection_p95": float(raw_cfl_after_dt_selection_p95),
            "flow_mode": str(flow_mode),
            "flow_progression_solved": bool(flow_progression_solved),
            "flow_progression_solved_with_startup_tolerance": bool(
                flow_prog_acc.get(
                    "flow_progression_solved_with_startup_tolerance",
                    flow_progression_solved,
                )
            ),
            "flow_progression_acceptance_reason": str(flow_progression_reason),
            "flow_progression_warning_step_count": int(
                flow_prog_acc.get("warning_step_count", 0)
            ),
            "flow_progression_warning_steps": list(
                flow_prog_acc.get("warning_steps", [])
            ),
            "startup_warning_steps_allowed": int(
                flow_prog_acc.get("startup_warning_steps_allowed", 0)
            ),
            "startup_warning_steps_observed": list(
                flow_prog_acc.get("startup_warning_steps_observed", [])
            ),
            "nonstartup_failed_steps": list(
                flow_prog_acc.get("nonstartup_failed_steps", [])
            ),
            "viscous_progression_accepted": bool(viscous_progression_accepted),
            "viscous_progression_acceptance_reason": str(
                viscous_progression_acceptance_reason
            ),
            "viscous_progression_checks": viscous_progression_checks,
            "convective_predictor_used": bool(
                any(
                    bool(r.get("convective_predictor_used", False))
                    for r in progression_history
                )
            ),
            "convective_prototype_accepted": bool(convective_prototype_accepted),
            "convective_prototype_acceptance_reason": str(
                convective_prototype_acceptance_reason
            ),
            "convective_prototype_checks": convective_prototype_checks,
            "convective_readiness_checks": convective_readiness_checks,
            "readiness_reason": str(convective_readiness_reason),
            "ready_for_long_ns_run_debug": bool(ready_for_long_ns_run_debug),
            "ready_for_long_ns_run_physical": bool(ready_for_long_ns_run_physical),
            "ready_for_long_ns_run": bool(ready_for_long_ns_run),
            "ns_auto_dt_accepted": bool(ns_auto_dt_accepted),
            "ns_auto_dt_warning": str(ns_auto_dt_warning),
            **warning_agg,
            **boundary_flux_policy_audit,
            **stabilization_audit,
            **ns_coupling_audit,
            "viscous_predictor_damages_divergence": bool(
                viscous_predictor_damages_divergence
            ),
            "viscous_predictor_damage_ratio_l2": float(
                viscous_predictor_damage_ratio_l2
            ),
            "viscous_predictor_damage_ratio_linf": float(
                viscous_predictor_damage_ratio_linf
            ),
            "viscous_predictor_best_mode": str(viscous_predictor_best_mode),
            "stokes_ready_for_advection_term": bool(stokes_ready_for_advection_term),
            "stokes_baseline_accepted": bool(stokes_baseline_accepted),
            "stokes_baseline_acceptance_reason": str(stokes_baseline_acceptance_reason),
            "recommended_stokes_baseline_config": recommended_stokes_baseline_config,
            "ready_for_convective_prototype": bool(ready_for_convective_prototype),
            "recommended_convective_debug_config": recommended_convective_debug_config,
        }
        flow_diag = _sync_flow_diagnostics_with_final_artifacts(
            flow_diag,
            acceptance_report=acceptance_report,
            summary=summary,
        )
        artifact_write_started_perf = perf_counter()
        _write_json(run_dir / "flow_diagnostics.json", flow_diag)
        _write_json(run_dir / "acceptance_report.json", acceptance_report)
        _write_json(run_dir / "resume_manifest.json", flow_resume_manifest)
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "config.json",
            {
                "cli_args": vars(args),
                "resolved_mesh_npz": str(mesh_npz),
                "mesh_sha256": mesh_sha256,
                "flow_backend_selected": exec_backend,
                "flow_device_selected": exec_device,
                "timing_mode": str(args.timing_mode),
                "postprocessing_mode": str(args.postprocessing_mode),
                "cuda_determinism": determinism_report,
                "fixed_work": fixed_work_manifest,
                "command_line": list(sys.argv),
                "resolved_startup_bootstrap_max_steps": int(
                    startup_bootstrap_max_steps
                ),
            },
        )
        final_metadata_write_seconds = float(
            perf_counter() - artifact_write_started_perf
        )
        manifest_recorder.record_completed(
            inputs=manifest_inputs,
            outputs={
                "mesh_npz": str(mesh_npz),
                "summary_json": str(run_dir / "summary.json"),
                "resume_manifest_json": str(run_dir / "resume_manifest.json"),
                "config_json": str(run_dir / "config.json"),
                "acceptance_report_json": str(run_dir / "acceptance_report.json"),
                "flow_diagnostics_json": str(run_dir / "flow_diagnostics.json"),
                "startup_bootstrap_history_json": str(
                    run_dir / "startup_bootstrap_history.json"
                ),
                "startup_root_cause_report_json": str(
                    run_dir / "startup_root_cause_report.json"
                ),
                "flow_coupling_metadata_json": str(
                    run_dir / "flow_coupling_metadata.json"
                ),
                "final_corrected_face_flux_npy": str(
                    run_dir / "final_corrected_face_flux.npy"
                ),
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
            },
        )
        finished_at = datetime.now(timezone.utc)
        wall_total_seconds = float(perf_counter() - process_started_perf)
        timing = summary["timing"]
        phases = timing["phases"]
        phases["final_metadata_write_seconds"] = final_metadata_write_seconds
        accounted_seconds = sum(
            float(phases[key])
            for key in (
                "setup_seconds",
                "startup_bootstrap_seconds",
                "flow_stepping_seconds",
                "postprocessing_seconds",
                "final_metadata_write_seconds",
            )
        )
        phases["unaccounted_seconds"] = max(wall_total_seconds - accounted_seconds, 0.0)
        completed_steps = int(timing["steps"]["completed"])
        flow_rate = _ratio(completed_steps, flow_stepping_seconds)
        timing["steps"]["flow_loop_steps_per_second"] = flow_rate
        timing["steps"]["flow_loop_cell_steps_per_second"] = float(
            flow_rate * int(mesh.tetrahedra.shape[0])
        )
        timing["components"]["unattributed_flow_loop_seconds"] = max(
            flow_stepping_seconds - sum(component_seconds.values()), 0.0
        )
        timing["finished_at_utc"] = finished_at.isoformat()
        timing["wall_total_seconds"] = wall_total_seconds
        timing["runtime_finalization_scope"] = (
            "ends immediately before the final summary.json write and final log output"
        )
        summary["runtime_seconds"] = wall_total_seconds
        summary["steps_per_second"] = flow_rate
        _write_json(run_dir / "summary.json", summary)
        print(
            "[gmsh-tetra-flow] runtime: "
            f"{_format_duration(wall_total_seconds)} "
            f"({wall_total_seconds:.3f}s, {flow_rate:.3f} steps/s)"
        )
        print(f"[gmsh-tetra-flow] summary written: {run_dir / 'summary.json'}")


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
