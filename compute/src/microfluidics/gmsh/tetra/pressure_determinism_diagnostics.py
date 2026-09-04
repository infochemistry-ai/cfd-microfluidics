"""Opt-in, fixed-input diagnostics for the tetra pressure projection.

The helpers in this module deliberately do not participate in the normal flow
path.  They make the inputs and results of a pressure solve inspectable without
changing its coefficients, PCG algorithm, or CUDA ``index_add_`` matvec.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import (
    TetraFlowConfig,
    TetraFlowState,
    _apply_face_flux_boundary_conditions_inplace,
    _assemble_poisson_rhs,
    _build_pressure_system_coefficients,
    _compute_cell_flux_sum,
    _inlet_face_sets,
    solve_tetra_pressure_projection,
)

DEFAULT_FIXED_WORK_DT = 1.0e-5
_STATE_ARRAYS = {
    "face_flux": "final_corrected_face_flux.npy",
    "cell_velocity": "final_cell_velocity.npy",
    "pressure": "final_pressure.npy",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_sha256(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(
            f"Fixed-work source summary lacks a valid recorded {key}. "
            "Re-run the source flow with mesh provenance enabled."
        )
    return value


def array_summary(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array))
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _sha256_bytes(arr.tobytes(order="C")),
        "finite_count": int(np.count_nonzero(finite)),
        "nan_count": int(np.count_nonzero(np.isnan(arr))),
        "inf_count": int(np.count_nonzero(np.isinf(arr))),
    }
    if finite_values.size:
        summary.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "l2": float(np.sqrt(np.mean(finite_values * finite_values))),
            }
        )
    else:
        summary.update({"min": None, "max": None, "mean": None, "l2": None})
    return summary


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    same_shape = ref.shape == got.shape
    if not same_shape:
        return {
            "shape_equal": False,
            "bitwise_equal": False,
            "max_abs": None,
            "mean_abs": None,
            "relative_l2": None,
        }
    ref_float = np.asarray(ref, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        diff = np.asarray(got, dtype=np.float64) - ref_float
    finite = np.isfinite(diff)
    finite_diff = diff[finite]
    comparison: dict[str, Any] = {
        "shape_equal": True,
        "bitwise_equal": bool(np.array_equal(ref, got, equal_nan=True)),
        "finite_count": int(np.count_nonzero(finite)),
        "nan_count": int(np.count_nonzero(np.isnan(diff))),
        "inf_count": int(np.count_nonzero(np.isinf(diff))),
    }
    if not finite_diff.size:
        comparison.update({"max_abs": None, "mean_abs": None, "relative_l2": None})
        return comparison

    ref_l2 = float(np.sqrt(np.sum(ref_float[finite] ** 2)))
    diff_l2 = float(np.sqrt(np.sum(finite_diff * finite_diff)))
    comparison.update(
        {
            "max_abs": float(np.max(np.abs(finite_diff))),
            "mean_abs": float(np.mean(np.abs(finite_diff))),
            "relative_l2": (
                float(diff_l2 / ref_l2) if ref_l2 > 0.0 else float(diff_l2)
            ),
        }
    )
    return comparison


def gauge_adjusted_comparison(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    if ref.shape == got.shape:
        paired_finite = np.isfinite(ref) & np.isfinite(got)
        offset = (
            float(np.mean(got[paired_finite] - ref[paired_finite]))
            if np.any(paired_finite)
            else 0.0
        )
    else:
        offset = 0.0
    result = compare_arrays(ref, got - offset)
    result["removed_constant_offset"] = offset
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def configure_cuda_determinism(mode: str) -> dict[str, Any]:
    """Configure the requested global Torch diagnostic mode before CUDA use."""
    requested = str(mode).strip().lower()
    if requested not in {"off", "warn", "error"}:
        raise ValueError("CUDA determinism mode must be 'off', 'warn', or 'error'.")
    report: dict[str, Any] = {
        "requested_mode": requested,
        "effective_mode": "off",
        "deterministic_algorithms_enabled": False,
        "warn_only": False,
        "cublas_workspace_config_requested": None,
        "cublas_workspace_config_effective": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_initialized_before_configuration": False,
        "diagnostic_success": True,
        "exception_type": None,
        "exception_message": None,
    }
    if requested == "off":
        return report
    try:
        import torch  # type: ignore

        cuda_initialized = bool(torch.cuda.is_initialized())
        report["cuda_initialized_before_configuration"] = cuda_initialized
        report["torch_version"] = str(torch.__version__)
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        requested_workspace = ":4096:8"
        report["cublas_workspace_config_requested"] = requested_workspace
        if not cuda_initialized:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = requested_workspace
            report["cublas_workspace_config_effective"] = requested_workspace
        else:
            report["cublas_workspace_config_effective"] = os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            )
        torch.use_deterministic_algorithms(True, warn_only=requested == "warn")
        report["effective_mode"] = requested
        report["deterministic_algorithms_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        report["warn_only"] = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
    except Exception as exc:  # report instead of silently degrading diagnostics
        report["diagnostic_success"] = False
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
    return report


def load_fixed_work_state(
    *,
    source_run_dir: Path,
    mesh: Any,
    mesh_npz: Path,
    legacy_source_mesh_sha256: str | None = None,
) -> tuple[TetraFlowState, dict[str, Any]]:
    source = source_run_dir.resolve()
    summary_path = source / "summary.json"
    required = [summary_path, *[source / name for name in _STATE_ARRAYS.values()]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Fixed-work source is missing: " + ", ".join(missing))
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_mesh_value = str(source_summary.get("resolved_mesh_npz", "")).strip()
    if not source_mesh_value:
        raise ValueError("Fixed-work source summary lacks resolved_mesh_npz.")
    source_mesh = Path(source_mesh_value)
    recorded_mesh_sha = str(source_summary.get("mesh_sha256", "")).strip().lower()
    mesh_sha_provenance = "source_summary"
    if re.fullmatch(r"[0-9a-f]{64}", recorded_mesh_sha):
        source_mesh_sha = recorded_mesh_sha
    elif legacy_source_mesh_sha256 is not None:
        source_mesh_sha = str(legacy_source_mesh_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", source_mesh_sha):
            raise ValueError("legacy_source_mesh_sha256 must be 64 hex chars.")
        mesh_sha_provenance = "explicit_legacy_override"
    else:
        source_mesh_sha = _recorded_sha256(source_summary, "mesh_sha256")
    requested_mesh_sha = sha256_file(mesh_npz)
    if source_mesh_sha != requested_mesh_sha:
        raise ValueError(
            "Fixed-work source mesh recorded at source-run time does not match "
            "--mesh-npz."
        )
    current_source_mesh_sha = (
        sha256_file(source_mesh) if source_mesh.is_file() else None
    )
    arrays = {
        key: np.load(source / filename, allow_pickle=False)
        for key, filename in _STATE_ARRAYS.items()
    }
    expected_shapes = {
        "face_flux": (int(mesh.face_vertices.shape[0]),),
        "cell_velocity": (int(mesh.tetrahedra.shape[0]), 3),
        "pressure": (int(mesh.tetrahedra.shape[0]),),
    }
    for key, array in arrays.items():
        if array.shape != expected_shapes[key]:
            raise ValueError(
                f"Fixed-work {key} shape {array.shape} does not match {expected_shapes[key]}."
            )
        if array.dtype != np.float64:
            raise ValueError(f"Fixed-work {key} must be float64, got {array.dtype}.")
    manifest = {
        "source_run_dir": str(source),
        "source_summary_json": str(summary_path),
        "source_mesh_npz": str(source_mesh),
        "source_mesh_sha256": source_mesh_sha,
        "source_mesh_sha256_provenance": mesh_sha_provenance,
        "source_mesh_current_sha256": current_source_mesh_sha,
        "source_mesh_path_matches_recorded": (
            current_source_mesh_sha == source_mesh_sha
            if current_source_mesh_sha is not None
            else None
        ),
        "requested_mesh_npz": str(mesh_npz.resolve()),
        "requested_mesh_sha256": requested_mesh_sha,
        "mesh_identity_equal": True,
        "input_arrays": {key: array_summary(array) for key, array in arrays.items()},
    }
    return TetraFlowState(**arrays, diagnostics={"fixed_work": manifest}), manifest


def _torch_matvec(
    coeff: dict[str, np.ndarray], probe: np.ndarray, device: str
) -> np.ndarray:
    import torch  # type: ignore

    dev = torch.device(device)
    pt = torch.as_tensor(probe, dtype=torch.float64, device=dev)
    diag = torch.as_tensor(coeff["diag"], dtype=torch.float64, device=dev)
    owner = torch.as_tensor(coeff["int_owner"], dtype=torch.long, device=dev)
    neigh = torch.as_tensor(coeff["int_neigh"], dtype=torch.long, device=dev)
    k = torch.as_tensor(coeff["int_k"], dtype=torch.float64, device=dev)
    torch.cuda.synchronize(dev)
    out = diag * pt
    if owner.numel() > 0:
        out.index_add_(0, owner, -k * pt[neigh])
        out.index_add_(0, neigh, -k * pt[owner])
    torch.cuda.synchronize(dev)
    return out.detach().cpu().numpy()


def run_pressure_determinism_diagnostic(
    *,
    mesh: Any,
    state: TetraFlowState,
    config: TetraFlowConfig,
    output_dir: Path,
    cuda_device: str = "cuda:0",
    repeats: int = 3,
    seed: int = 20260729,
) -> dict[str, Any]:
    """Compare CPU and CUDA PCG/projection outputs on one immutable state."""
    if str(config.pressure_solver) != "pcg_diag":
        raise ValueError(
            "Pressure determinism diagnostic requires pressure_solver='pcg_diag'."
        )
    if repeats < 3:
        raise ValueError(
            "Pressure determinism diagnostic requires at least three GPU repeats."
        )
    import torch  # type: ignore
    from microfluidics.gmsh.tetra.gmsh_tetra_flow_solver import _matvec_pressure_numpy

    if not torch.cuda.is_available():
        raise RuntimeError("Pressure determinism diagnostic requires CUDA.")
    output_dir.mkdir(parents=True, exist_ok=True)
    left, right = _inlet_face_sets(mesh)
    flux_star = np.asarray(state.face_flux, dtype=np.float64).copy()
    _apply_face_flux_boundary_conditions_inplace(
        mesh,
        flux_star,
        inlet_speed=float(config.inlet_speed),
        left_inlet_faces=left,
        right_inlet_faces=right,
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
        wall_faces=np.asarray(mesh.wall_faces, dtype=np.int64),
    )
    coeff = _build_pressure_system_coefficients(
        mesh,
        dt=float(config.projection_dt),
        density=float(config.density),
        outlet_faces=np.asarray(mesh.outlet_faces, dtype=np.int64),
    )
    rhs, outlet_rhs = _assemble_poisson_rhs(
        _compute_cell_flux_sum(mesh, flux_star),
        coeff,
        pressure_outlet_value=float(config.pressure_outlet_value),
        projection_sign=config.projection_sign,
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=np.float64),
        rhs_mode=config.projection_rhs_mode,
    )
    np.savez_compressed(
        output_dir / "pressure_system_inputs.npz",
        **{
            key: np.asarray(coeff[key])
            for key in (
                "diag",
                "base_diag",
                "int_owner",
                "int_neigh",
                "int_k",
                "base_int_k",
                "k_scale",
            )
        },
        rhs=rhs,
        p0=np.asarray(state.pressure, dtype=np.float64),
        face_flux_star=flux_star,
        outlet_rhs=outlet_rhs,
    )
    system_manifest = {
        key: array_summary(np.asarray(value))
        for key, value in {
            "diag": coeff["diag"],
            "base_diag": coeff["base_diag"],
            "int_owner": coeff["int_owner"],
            "int_neigh": coeff["int_neigh"],
            "int_k": coeff["int_k"],
            "base_int_k": coeff["base_int_k"],
            "k_scale": np.asarray([coeff["k_scale"]]),
            "rhs": rhs,
            "p0": state.pressure,
            "outlet_rhs": outlet_rhs,
        }.items()
    }
    rng = np.random.default_rng(seed)
    probes = {
        "p0": np.asarray(state.pressure, dtype=np.float64),
        "seeded_random": rng.standard_normal(state.pressure.shape).astype(np.float64),
        "deterministic_ramp": np.linspace(
            -1.0, 1.0, state.pressure.size, dtype=np.float64
        ),
    }
    matvec_report: dict[str, Any] = {}
    for name, probe in probes.items():
        cpu = _matvec_pressure_numpy(coeff, probe)
        gpu = [_torch_matvec(coeff, probe, cuda_device) for _ in range(repeats)]
        matvec_report[name] = {
            "input": array_summary(probe),
            "cpu": array_summary(cpu),
            "gpu": [array_summary(item) for item in gpu],
            "cpu_vs_gpu": [compare_arrays(cpu, item) for item in gpu],
            "gpu_repeat_comparisons": [
                compare_arrays(gpu[0], item) for item in gpu[1:]
            ],
            "uses_index_add": True,
        }
    cpu_cfg = replace(config, backend="numpy", device="cpu", debug_store_history=True)
    gpu_cfg = replace(
        config, backend="torch", device=cuda_device, debug_store_history=True
    )
    cpu_started = perf_counter()
    cpu_state = solve_tetra_pressure_projection(mesh, state, cpu_cfg)
    cpu_runtime = perf_counter() - cpu_started
    gpu_states: list[TetraFlowState] = []
    gpu_runtimes: list[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(cuda_device)
        started = perf_counter()
        gpu_states.append(solve_tetra_pressure_projection(mesh, state, gpu_cfg))
        torch.cuda.synchronize(cuda_device)
        gpu_runtimes.append(perf_counter() - started)
    cpu_pressure = np.asarray(cpu_state.pressure)
    gpu_pressure = [np.asarray(item.pressure) for item in gpu_states]
    cpu_history = cpu_state.diagnostics["pressure"]["pressure_history"]
    gpu_histories = [
        item.diagnostics["pressure"]["pressure_history"] for item in gpu_states
    ]
    write_json(output_dir / "cpu_residual_history.json", {"history": cpu_history})
    for index, history in enumerate(gpu_histories, start=1):
        write_json(
            output_dir / f"gpu_residual_history_{index}.json", {"history": history}
        )
    projection = {
        "corrected_face_flux": [
            compare_arrays(cpu_state.face_flux, item.face_flux) for item in gpu_states
        ],
        "cell_velocity": [
            compare_arrays(cpu_state.cell_velocity, item.cell_velocity)
            for item in gpu_states
        ],
        "pressure": [compare_arrays(cpu_pressure, item) for item in gpu_pressure],
        "pressure_gauge_adjusted": [
            gauge_adjusted_comparison(cpu_pressure, item) for item in gpu_pressure
        ],
        "gpu_repeat_pressure": [
            compare_arrays(gpu_pressure[0], item) for item in gpu_pressure[1:]
        ],
        "gpu_repeat_corrected_face_flux": [
            compare_arrays(gpu_states[0].face_flux, item.face_flux)
            for item in gpu_states[1:]
        ],
        "gpu_repeat_cell_velocity": [
            compare_arrays(gpu_states[0].cell_velocity, item.cell_velocity)
            for item in gpu_states[1:]
        ],
    }
    first_divergent_iteration = None
    for index, (cpu_item, gpu_item) in enumerate(
        zip(cpu_history, gpu_histories[0]), start=1
    ):
        if cpu_item != gpu_item:
            first_divergent_iteration = index
            break
    if first_divergent_iteration is None and len(cpu_history) != len(gpu_histories[0]):
        first_divergent_iteration = min(len(cpu_history), len(gpu_histories[0])) + 1
    matvec_identical = all(
        item["bitwise_equal"]
        for probe in matvec_report.values()
        for item in probe["gpu_repeat_comparisons"]
    )
    pcg_identical = all(
        comparison["bitwise_equal"] for comparison in projection["gpu_repeat_pressure"]
    ) and all(gpu_histories[0] == history for history in gpu_histories[1:])
    report = {
        "input_state_identical": True,
        "input_system_identical": True,
        "pressure_system": system_manifest,
        "pressure_boundary_data": {
            "left_inlet_faces": array_summary(left),
            "right_inlet_faces": array_summary(right),
            "outlet_faces": array_summary(np.asarray(mesh.outlet_faces)),
            "wall_faces": array_summary(np.asarray(mesh.wall_faces)),
        },
        "pcg_configuration": {
            "max_pressure_iterations": config.max_pressure_iterations,
            "pressure_tolerance": config.pressure_tolerance,
            "pressure_relative_tolerance": config.pressure_relative_tolerance,
            "cg_breakdown_eps": config.cg_breakdown_eps,
            "cg_stagnation_window": config.cg_stagnation_window,
            "cg_stagnation_ratio": config.cg_stagnation_ratio,
            "preconditioner": "diag",
        },
        "matvec": matvec_report,
        "cpu": {
            "runtime_seconds": cpu_runtime,
            "pressure": array_summary(cpu_pressure),
            "diagnostics": cpu_state.diagnostics["pressure"],
        },
        "gpu": [
            {
                "runtime_seconds": runtime,
                "pressure": array_summary(item.pressure),
                "diagnostics": item.diagnostics["pressure"],
            }
            for runtime, item in zip(gpu_runtimes, gpu_states)
        ],
        "projection_comparisons": projection,
        "repeated_gpu_matvec_identical": matvec_identical,
        "repeated_gpu_pcg_identical": pcg_identical,
        "first_divergent_stage": None
        if pcg_identical and matvec_identical
        else ("matvec" if not matvec_identical else "pcg"),
        "first_divergent_iteration": first_divergent_iteration,
        "suspected_operation": "index_add_" if not matvec_identical else None,
        "evidence": "CUDA matvec uses the same index_add_ accumulation path as Torch PCG; this is evidence only, not root-cause proof.",
        "diagnosis_limitations": "A divergence after equal matvec probes does not prove an alternative root cause; CUDA scheduling and library behavior remain environment-specific.",
    }
    write_json(output_dir / "pressure_determinism_report.json", report)
    return report
