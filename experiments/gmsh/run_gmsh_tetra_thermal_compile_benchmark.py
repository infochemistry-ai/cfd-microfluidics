"""Special-purpose torch.compile benchmark for tetra thermal runs.

This benchmark keeps pinned scenario defaults for repeatable measurements and is
not a general-purpose thermal launcher.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.gmsh._path_utils import _normalize_user_path  # noqa: E402
from experiments.gmsh.run_gmsh_tetra_thermal_debug import (  # noqa: E402
    _json_ready,
    _resolve_msh_input,
    _resolve_thermal_diffusivity,
    _write_json,
)
from microfluidics.gmsh.gmsh_mesh_import import import_gmsh_tetra_mesh  # noqa: E402
from microfluidics.gmsh.tetra.gmsh_tetra_operators import (  # noqa: E402
    build_face_normal_flux_from_velocity,
)
from microfluidics.gmsh.tetra.gmsh_tetra_thermal_solver import (  # noqa: E402
    GmshTetraThermalConfig,
    run_tetra_thermal_debug,
)
from microfluidics.gmsh.tetra.gmsh_tetra_velocity_fields import (  # noqa: E402
    build_prescribed_velocity_field,
)
from microfluidics.path_contract import (  # noqa: E402
    GMSH_TETRA_THERMAL_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)


def _synchronize_torch_if_needed(torch_device: str) -> None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return

    device_type = str(torch_device).split(":", 1)[0].lower()
    if device_type != "cuda" or not bool(torch.cuda.is_available()):
        return
    torch.cuda.synchronize(torch.device(torch_device))


def _bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


@lru_cache(maxsize=1)
def _get_windows_process_memory_reader() -> Callable[[], int | None] | None:
    """Lazily initialise the Windows RSS API once for the polling worker."""
    if platform.system().lower() != "windows":
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
    except Exception:
        return None

    def _read_rss() -> int | None:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)

    # The cached closure retains the psapi function and current-process handle.
    return _read_rss


def _current_process_rss_bytes_windows() -> int | None:
    reader = _get_windows_process_memory_reader()
    return reader() if reader is not None else None


def _start_process_peak_memory_sampler(
    *,
    sample_interval_seconds: float = 0.02,
) -> dict[str, object]:
    try:
        import psutil  # type: ignore
    except ModuleNotFoundError:
        start_rss = _current_process_rss_bytes_windows()
        if start_rss is None:
            return {
                "available": False,
                "method": "psutil unavailable",
            }
        peak_rss = {"value": start_rss}
        stop_event = threading.Event()

        def _windows_worker() -> None:
            while not stop_event.wait(sample_interval_seconds):
                current_rss = _current_process_rss_bytes_windows()
                if current_rss is None:
                    break
                if current_rss > peak_rss["value"]:
                    peak_rss["value"] = current_rss

        thread = threading.Thread(target=_windows_worker, daemon=True)
        thread.start()
        return {
            "available": True,
            "method": "windows GetProcessMemoryInfo working set polling",
            "start_rss_bytes": start_rss,
            "peak_rss_ref": peak_rss,
            "stop_event": stop_event,
            "thread": thread,
            "windows_fallback": True,
        }

    process = psutil.Process()
    start_rss = int(process.memory_info().rss)
    peak_rss = {"value": start_rss}
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(sample_interval_seconds):
            try:
                current_rss = int(process.memory_info().rss)
            except Exception:
                break
            if current_rss > peak_rss["value"]:
                peak_rss["value"] = current_rss

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return {
        "available": True,
        "method": "psutil rss polling",
        "process": process,
        "start_rss_bytes": start_rss,
        "peak_rss_ref": peak_rss,
        "stop_event": stop_event,
        "thread": thread,
    }


def _stop_process_peak_memory_sampler(handle: dict[str, object]) -> dict[str, object]:
    if not bool(handle.get("available", False)):
        return {
            "available": False,
            "method": str(handle.get("method", "unavailable")),
        }

    stop_event = handle["stop_event"]
    thread = handle["thread"]
    peak_rss_ref = handle["peak_rss_ref"]
    assert isinstance(stop_event, threading.Event)
    assert isinstance(thread, threading.Thread)
    stop_event.set()
    thread.join(timeout=0.2)
    if bool(handle.get("windows_fallback", False)):
        end_rss = _current_process_rss_bytes_windows()
        if end_rss is None:
            end_rss = int(handle["start_rss_bytes"])
    else:
        process = handle["process"]
        try:
            end_rss = int(process.memory_info().rss)
        except Exception:
            end_rss = int(handle["start_rss_bytes"])
    peak_rss = max(int(peak_rss_ref["value"]), end_rss)
    return {
        "available": True,
        "method": str(handle.get("method", "psutil rss polling")),
        "start_rss_bytes": int(handle["start_rss_bytes"]),
        "end_rss_bytes": end_rss,
        "peak_rss_bytes": peak_rss,
        "peak_rss_mib": _bytes_to_mib(peak_rss),
    }


def _reset_torch_peak_memory_if_needed(torch_device: str) -> dict[str, object]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return {"available": False, "method": "torch unavailable"}

    device_type = str(torch_device).split(":", 1)[0].lower()
    if device_type != "cuda" or not bool(torch.cuda.is_available()):
        return {"available": False, "method": "cuda peak memory not available"}

    device = torch.device(torch_device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "available": True,
        "method": "torch.cuda.max_memory_allocated/reserved",
        "device": str(device),
    }


def _read_torch_peak_memory_if_needed(torch_device: str) -> dict[str, object]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return {"available": False, "method": "torch unavailable"}

    device_type = str(torch_device).split(":", 1)[0].lower()
    if device_type != "cuda" or not bool(torch.cuda.is_available()):
        return {"available": False, "method": "cuda peak memory not available"}

    device = torch.device(torch_device)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "available": True,
        "method": "torch.cuda.max_memory_allocated/reserved",
        "device": str(device),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": _bytes_to_mib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": _bytes_to_mib(peak_reserved),
    }


def _collect_torch_environment(torch_device: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_device_requested": str(torch_device),
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_available": False,
        "torch_device_count": 0,
        "torch_gpu_name": None,
        "torch_compile_available": False,
        "triton_importable": False,
        "triton_version": None,
    }
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return payload

    payload["torch_available"] = True
    payload["torch_version"] = str(getattr(torch, "__version__", "unknown"))
    payload["torch_compile_available"] = bool(hasattr(torch, "compile"))
    cuda_available = bool(torch.cuda.is_available())
    payload["torch_cuda_available"] = cuda_available
    if cuda_available:
        try:
            payload["torch_device_count"] = int(torch.cuda.device_count())
        except Exception:
            payload["torch_device_count"] = 0
        requested = str(torch_device)
        device_index = 0
        if requested.startswith("cuda:"):
            try:
                device_index = int(requested.split(":", 1)[1])
            except ValueError:
                device_index = 0
        try:
            payload["torch_gpu_name"] = str(torch.cuda.get_device_name(device_index))
        except Exception:
            payload["torch_gpu_name"] = None

    try:
        import triton  # type: ignore
    except Exception:
        return payload
    payload["triton_importable"] = True
    payload["triton_version"] = str(getattr(triton, "__version__", "unknown"))
    return payload


def _collect_source_provenance() -> dict[str, object]:
    def _git_output(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    commit = _git_output("rev-parse", "HEAD")
    branch = _git_output("rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git_output("status", "--short")
    return {
        "source_commit": commit or None,
        "source_branch": branch or None,
        "source_tree_dirty": bool(dirty),
        "source_tree_status_short": dirty.splitlines() if dirty else [],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msh", type=str, default="t_junction.msh")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(
            resolve_repo_path(
                PROJECT_ROOT,
                GMSH_TETRA_THERMAL_RUNS_ROOT_REL / "compile_benchmarks",
            )
        ),
    )
    parser.add_argument("--backend", type=str, choices=("torch",), default="torch")
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument(
        "--compile-modes",
        nargs="+",
        choices=("off", "on", "auto"),
        default=("off", "on", "auto"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runtime-budget-seconds", type=float, default=0.0)
    parser.add_argument("--enforce-runtime-budget", action="store_true")
    parser.add_argument("--max-process-peak-memory-mib", type=float, default=0.0)
    parser.add_argument("--max-cuda-peak-memory-mib", type=float, default=0.0)
    parser.add_argument(
        "--max-temperature-diff-from-reference",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--require-compiled-step", action="store_true")
    parser.add_argument("--require-cuda-core-arrays", action="store_true")
    parser.add_argument("--target-mesh-label", type=str, default="")
    parser.add_argument("--target-min-tetra-cells", type=int, default=0)
    parser.add_argument("--target-max-tetra-cells", type=int, default=0)
    parser.add_argument("--target-min-faces", type=int, default=0)
    parser.add_argument("--target-max-faces", type=int, default=0)
    parser.add_argument("--enforce-target-mesh", action="store_true")
    parser.add_argument("--steps", type=int, default=400)
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
    parser.add_argument("--heat-source", type=float, default=2.5e5)
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
    parser.add_argument(
        "--execution-profile",
        type=str,
        choices=("production", "debug", "reference"),
        default="production",
    )
    parser.add_argument(
        "--diagnostics-mode",
        type=str,
        choices=("auto", "debug", "fast"),
        default="fast",
    )
    parser.add_argument(
        "--history-mode",
        type=str,
        choices=("auto", "dict", "compact", "off"),
        default="compact",
    )
    parser.add_argument("--history-stride", type=int, default=100)
    parser.add_argument(
        "--velocity-field",
        type=str,
        default="two_inlets_to_outlet_tj_axis_aligned_clean",
    )
    parser.add_argument("--inlet-speed", type=float, default=0.15)
    return parser


def _acceptance_check(
    *,
    name: str,
    passed: bool,
    observed: object,
    expected: object,
) -> dict[str, object]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }


def _run_one_benchmark_mode(
    *,
    mesh,
    face_normal_velocity: np.ndarray,
    flux_diag: dict[str, float],
    velocity_metadata: dict[str, object],
    common_config: dict[str, object],
    compile_mode: str,
    repeats: int,
    warmup: int,
    torch_device: str,
    runtime_budget_seconds: float,
) -> dict[str, object]:
    runtimes: list[float] = []
    warmup_runtimes: list[float] = []
    steps_per_second_values: list[float] = []
    process_peak_memory_runs: list[dict[str, object]] = []
    cuda_peak_memory_runs: list[dict[str, object]] = []
    last_result: dict[str, object] | None = None
    last_temperature: np.ndarray | None = None

    for iteration in range(warmup + repeats):
        cfg = GmshTetraThermalConfig(
            **common_config,
            torch_compile_step_body=compile_mode,  # type: ignore[arg-type]
        )
        process_memory_handle = _start_process_peak_memory_sampler()
        _reset_torch_peak_memory_if_needed(torch_device)
        _synchronize_torch_if_needed(torch_device)
        started = time.perf_counter()
        result = run_tetra_thermal_debug(
            mesh,
            cfg,
            face_normal_velocity=face_normal_velocity,
            flux_diagnostics=flux_diag,
            velocity_metadata=velocity_metadata,
        )
        _synchronize_torch_if_needed(torch_device)
        elapsed = float(time.perf_counter() - started)
        process_memory_result = _stop_process_peak_memory_sampler(process_memory_handle)
        torch_peak_result = _read_torch_peak_memory_if_needed(torch_device)
        if iteration < warmup:
            warmup_runtimes.append(elapsed)
            continue
        runtimes.append(elapsed)
        steps_per_second_values.append(
            float(cfg.steps) / elapsed if elapsed > 0.0 else 0.0
        )
        process_peak_memory_runs.append(process_memory_result)
        cuda_peak_memory_runs.append(torch_peak_result)
        last_result = result
        last_temperature = np.asarray(result["temperature"], dtype=np.float64).copy()

    if last_result is None or last_temperature is None:
        raise RuntimeError("Benchmark did not produce a measured run.")

    perf = dict(last_result.get("performance_diagnostics", {}))
    backend_execution = dict(last_result.get("backend_execution", {}))
    process_peak_available = any(
        bool(run.get("available", False)) for run in process_peak_memory_runs
    )
    process_peak_bytes = [
        int(run["peak_rss_bytes"])
        for run in process_peak_memory_runs
        if bool(run.get("available", False))
    ]
    cuda_peak_available = any(
        bool(run.get("available", False)) for run in cuda_peak_memory_runs
    )
    cuda_peak_allocated_bytes = [
        int(run["peak_allocated_bytes"])
        for run in cuda_peak_memory_runs
        if bool(run.get("available", False))
    ]
    cuda_peak_reserved_bytes = [
        int(run["peak_reserved_bytes"])
        for run in cuda_peak_memory_runs
        if bool(run.get("available", False))
    ]
    runtime_budget_enabled = bool(runtime_budget_seconds > 0.0)
    runtime_budget_median_pass = (
        bool(statistics.median(runtimes) <= runtime_budget_seconds)
        if runtime_budget_enabled
        else None
    )
    runtime_budget_all_runs_pass = (
        bool(all(runtime <= runtime_budget_seconds for runtime in runtimes))
        if runtime_budget_enabled
        else None
    )
    return {
        "compile_mode_requested": str(compile_mode),
        "warmup_runtimes_seconds": warmup_runtimes,
        "runtimes_seconds": runtimes,
        "wall_time_seconds": {
            "warmup_values": warmup_runtimes,
            "values": runtimes,
            "min": float(min(runtimes)),
            "median": float(statistics.median(runtimes)),
            "mean": float(statistics.mean(runtimes)),
            "max": float(max(runtimes)),
        },
        "runtime_min_seconds": float(min(runtimes)),
        "runtime_median_seconds": float(statistics.median(runtimes)),
        "runtime_mean_seconds": float(statistics.mean(runtimes)),
        "runtime_max_seconds": float(max(runtimes)),
        "steps_per_second": {
            "values": steps_per_second_values,
            "min": float(min(steps_per_second_values)),
            "median": float(statistics.median(steps_per_second_values)),
            "mean": float(statistics.mean(steps_per_second_values)),
            "max": float(max(steps_per_second_values)),
        },
        "peak_memory": {
            "process_rss": {
                "available": bool(process_peak_available),
                "method": (
                    str(process_peak_memory_runs[-1].get("method"))
                    if process_peak_memory_runs
                    else "unavailable"
                ),
                "runs": process_peak_memory_runs,
                "peak_bytes_values": process_peak_bytes,
                "peak_mib_values": [
                    _bytes_to_mib(value) for value in process_peak_bytes
                ],
                "peak_bytes_max": max(process_peak_bytes)
                if process_peak_bytes
                else None,
                "peak_mib_max": (
                    _bytes_to_mib(max(process_peak_bytes))
                    if process_peak_bytes
                    else None
                ),
            },
            "torch_cuda": {
                "available": bool(cuda_peak_available),
                "method": (
                    str(cuda_peak_memory_runs[-1].get("method"))
                    if cuda_peak_memory_runs
                    else "unavailable"
                ),
                "runs": cuda_peak_memory_runs,
                "peak_allocated_bytes_values": cuda_peak_allocated_bytes,
                "peak_allocated_mib_values": [
                    _bytes_to_mib(value) for value in cuda_peak_allocated_bytes
                ],
                "peak_allocated_bytes_max": (
                    max(cuda_peak_allocated_bytes)
                    if cuda_peak_allocated_bytes
                    else None
                ),
                "peak_allocated_mib_max": (
                    _bytes_to_mib(max(cuda_peak_allocated_bytes))
                    if cuda_peak_allocated_bytes
                    else None
                ),
                "peak_reserved_bytes_values": cuda_peak_reserved_bytes,
                "peak_reserved_mib_values": [
                    _bytes_to_mib(value) for value in cuda_peak_reserved_bytes
                ],
                "peak_reserved_bytes_max": (
                    max(cuda_peak_reserved_bytes) if cuda_peak_reserved_bytes else None
                ),
                "peak_reserved_mib_max": (
                    _bytes_to_mib(max(cuda_peak_reserved_bytes))
                    if cuda_peak_reserved_bytes
                    else None
                ),
            },
        },
        "backend_used": {
            "stepping_backend": str(
                backend_execution.get("stepping_backend", "unknown")
            ),
            "device": str(backend_execution.get("device", "unknown")),
            "torch_device": str(backend_execution.get("torch_device", "unknown")),
            "all_core_arrays_on_cuda": bool(
                backend_execution.get("all_core_arrays_on_cuda", False)
            ),
        },
        "backend_execution": backend_execution,
        "runtime_budget": {
            "enabled": runtime_budget_enabled,
            "seconds": float(runtime_budget_seconds),
            "median_pass": runtime_budget_median_pass,
            "all_runs_pass": runtime_budget_all_runs_pass,
        },
        "performance_diagnostics": perf,
        "history_settings": dict(last_result.get("history_settings", {})),
        "model_capabilities": dict(last_result.get("model_capabilities", {})),
        "thermal_observables": dict(last_result.get("thermal_observables", {})),
        "final_stats": dict(last_result.get("final_stats", {})),
        "cfl_warning": bool(last_result.get("cfl_warning", True)),
        "dt_control": dict(last_result.get("dt_control", {})),
        "temperature": last_temperature,
    }


def _evaluate_benchmark_acceptance(
    *,
    mode_summaries: dict[str, dict[str, object]],
    reference_mode: str | None,
    mesh_size: dict[str, int],
    require_compiled_step: bool,
    require_cuda_core_arrays: bool,
    runtime_budget_seconds: float,
    enforce_runtime_budget: bool,
    max_process_peak_memory_mib: float,
    max_cuda_peak_memory_mib: float,
    max_temperature_diff_from_reference: float,
    target_mesh_label: str,
    target_min_tetra_cells: int,
    target_max_tetra_cells: int,
    target_min_faces: int,
    target_max_faces: int,
    enforce_target_mesh: bool,
    min_temperature: float = 290.0,
    max_temperature: float = 400.0,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    for compile_mode, payload in mode_summaries.items():
        backend_used = dict(payload.get("backend_used", {}))
        backend_execution = dict(payload.get("backend_execution", {}))
        perf = dict(payload.get("performance_diagnostics", {}))
        final_stats = dict(payload.get("final_stats", {}))
        dt_control = dict(payload.get("dt_control", {}))

        stepping_backend = str(backend_used.get("stepping_backend", "unknown"))
        checks.append(
            _acceptance_check(
                name=f"production_backend_torch[{compile_mode}]",
                passed=stepping_backend == "torch",
                observed=stepping_backend,
                expected="torch",
            )
        )
        used_numpy_fallback = bool(backend_execution.get("used_numpy_fallback", True))
        checks.append(
            _acceptance_check(
                name=f"no_numpy_fallback[{compile_mode}]",
                passed=not used_numpy_fallback,
                observed=used_numpy_fallback,
                expected=False,
            )
        )
        degraded_execution_mode = bool(perf.get("degraded_execution_mode", True))
        checks.append(
            _acceptance_check(
                name=f"not_degraded_execution[{compile_mode}]",
                passed=not degraded_execution_mode,
                observed=degraded_execution_mode,
                expected=False,
            )
        )
        finite_stats = all(
            np.isfinite(float(final_stats.get(name, float("nan"))))
            for name in ("min", "max", "mean")
        )
        checks.append(
            _acceptance_check(
                name=f"finite_temperature_stats[{compile_mode}]",
                passed=finite_stats,
                observed={
                    "min": final_stats.get("min"),
                    "max": final_stats.get("max"),
                    "mean": final_stats.get("mean"),
                },
                expected="finite min/max/mean",
            )
        )
        if finite_stats:
            temperature_min = float(final_stats["min"])
            temperature_max = float(final_stats["max"])
            bounded_temperature = bool(
                temperature_min >= min_temperature - 1e-9
                and temperature_max <= max_temperature + 1e-9
            )
        else:
            bounded_temperature = False
        checks.append(
            _acceptance_check(
                name=f"temperature_bounds[{compile_mode}]",
                passed=bounded_temperature,
                observed={
                    "min": final_stats.get("min"),
                    "max": final_stats.get("max"),
                },
                expected={"min_gte": min_temperature, "max_lte": max_temperature},
            )
        )
        cfl_warning = bool(payload.get("cfl_warning", True))
        checks.append(
            _acceptance_check(
                name=f"no_cfl_warning[{compile_mode}]",
                passed=not cfl_warning,
                observed=cfl_warning,
                expected=False,
            )
        )
        diffusion_warning = bool(dt_control.get("diffusion_stability_warning", True))
        checks.append(
            _acceptance_check(
                name=f"no_diffusion_stability_warning[{compile_mode}]",
                passed=not diffusion_warning,
                observed=diffusion_warning,
                expected=False,
            )
        )

    if require_compiled_step:
        for compile_mode, payload in mode_summaries.items():
            if compile_mode == "off":
                continue
            perf = dict(payload.get("performance_diagnostics", {}))
            compiled_step_used = bool(perf.get("torch_compile_step_used", False))
            checks.append(
                _acceptance_check(
                    name=f"compiled_step_used[{compile_mode}]",
                    passed=compiled_step_used,
                    observed=compiled_step_used,
                    expected=True,
                )
            )

    if require_cuda_core_arrays:
        for compile_mode, payload in mode_summaries.items():
            backend_used = dict(payload.get("backend_used", {}))
            all_core_arrays_on_cuda = bool(
                backend_used.get("all_core_arrays_on_cuda", False)
            )
            checks.append(
                _acceptance_check(
                    name=f"cuda_core_arrays[{compile_mode}]",
                    passed=all_core_arrays_on_cuda,
                    observed=all_core_arrays_on_cuda,
                    expected=True,
                )
            )

    if enforce_runtime_budget:
        for compile_mode, payload in mode_summaries.items():
            runtime_budget = dict(payload.get("runtime_budget", {}))
            all_runs_pass = runtime_budget.get("all_runs_pass")
            checks.append(
                _acceptance_check(
                    name=f"runtime_budget_all_runs[{compile_mode}]",
                    passed=bool(all_runs_pass),
                    observed=all_runs_pass,
                    expected={
                        "all_runs_pass": True,
                        "runtime_budget_seconds": float(runtime_budget_seconds),
                    },
                )
            )

    if max_process_peak_memory_mib > 0.0:
        for compile_mode, payload in mode_summaries.items():
            process_rss = dict(payload.get("peak_memory", {}).get("process_rss", {}))
            observed_peak = process_rss.get("peak_mib_max")
            passed = observed_peak is not None and float(observed_peak) <= float(
                max_process_peak_memory_mib
            )
            checks.append(
                _acceptance_check(
                    name=f"process_peak_memory[{compile_mode}]",
                    passed=passed,
                    observed=observed_peak,
                    expected={"peak_mib_max_lte": float(max_process_peak_memory_mib)},
                )
            )

    if max_cuda_peak_memory_mib > 0.0:
        for compile_mode, payload in mode_summaries.items():
            torch_cuda = dict(payload.get("peak_memory", {}).get("torch_cuda", {}))
            observed_peak = torch_cuda.get("peak_reserved_mib_max")
            passed = observed_peak is not None and float(observed_peak) <= float(
                max_cuda_peak_memory_mib
            )
            checks.append(
                _acceptance_check(
                    name=f"cuda_peak_memory[{compile_mode}]",
                    passed=passed,
                    observed=observed_peak,
                    expected={
                        "peak_reserved_mib_max_lte": float(max_cuda_peak_memory_mib)
                    },
                )
            )

    if max_temperature_diff_from_reference >= 0.0:
        for compile_mode, payload in mode_summaries.items():
            if compile_mode == reference_mode:
                continue
            diff_payload = payload.get("temperature_vs_reference")
            observed_diff = None
            if isinstance(diff_payload, dict):
                observed_diff = diff_payload.get("max_abs_diff")
            passed = observed_diff is not None and float(observed_diff) <= float(
                max_temperature_diff_from_reference
            )
            checks.append(
                _acceptance_check(
                    name=f"temperature_vs_reference[{compile_mode}]",
                    passed=passed,
                    observed=observed_diff,
                    expected={
                        "reference_mode": reference_mode,
                        "max_abs_diff_lte": float(max_temperature_diff_from_reference),
                    },
                )
            )

    target_checks_requested = any(
        value > 0
        for value in (
            target_min_tetra_cells,
            target_max_tetra_cells,
            target_min_faces,
            target_max_faces,
        )
    )
    if enforce_target_mesh or target_checks_requested:
        tetra_cells = int(mesh_size["tetra_cells"])
        faces = int(mesh_size["faces"])
        if target_min_tetra_cells > 0:
            checks.append(
                _acceptance_check(
                    name="target_mesh_min_tetra_cells",
                    passed=tetra_cells >= int(target_min_tetra_cells),
                    observed=tetra_cells,
                    expected={"gte": int(target_min_tetra_cells)},
                )
            )
        if target_max_tetra_cells > 0:
            checks.append(
                _acceptance_check(
                    name="target_mesh_max_tetra_cells",
                    passed=tetra_cells <= int(target_max_tetra_cells),
                    observed=tetra_cells,
                    expected={"lte": int(target_max_tetra_cells)},
                )
            )
        if target_min_faces > 0:
            checks.append(
                _acceptance_check(
                    name="target_mesh_min_faces",
                    passed=faces >= int(target_min_faces),
                    observed=faces,
                    expected={"gte": int(target_min_faces)},
                )
            )
        if target_max_faces > 0:
            checks.append(
                _acceptance_check(
                    name="target_mesh_max_faces",
                    passed=faces <= int(target_max_faces),
                    observed=faces,
                    expected={"lte": int(target_max_faces)},
                )
            )

    return {
        "passed": bool(all(check["passed"] for check in checks)),
        "checks": checks,
        "requirements": {
            "require_compiled_step": bool(require_compiled_step),
            "require_cuda_core_arrays": bool(require_cuda_core_arrays),
            "runtime_budget_seconds": float(runtime_budget_seconds),
            "enforce_runtime_budget": bool(enforce_runtime_budget),
            "max_process_peak_memory_mib": float(max_process_peak_memory_mib),
            "max_cuda_peak_memory_mib": float(max_cuda_peak_memory_mib),
            "max_temperature_diff_from_reference": float(
                max_temperature_diff_from_reference
            ),
            "target_mesh": {
                "label": str(target_mesh_label),
                "min_tetra_cells": int(target_min_tetra_cells),
                "max_tetra_cells": int(target_max_tetra_cells),
                "min_faces": int(target_min_faces),
                "max_faces": int(target_max_faces),
                "enforce_target_mesh": bool(enforce_target_mesh),
            },
        },
    }


def _build_benchmark_stage_status(acceptance: dict[str, object]) -> dict[str, object]:
    """Separate numerical/physical benchmark health from full qualification."""
    checks = [
        dict(check) for check in acceptance.get("checks", []) if isinstance(check, dict)
    ]

    def _all_named(prefix: str) -> bool:
        selected = [
            bool(check.get("passed", False))
            for check in checks
            if str(check.get("name", "")).startswith(prefix)
        ]
        return bool(selected and all(selected))

    numerical_checks_passed = all(
        _all_named(prefix)
        for prefix in (
            "finite_temperature_stats[",
            "no_cfl_warning[",
            "no_diffusion_stability_warning[",
        )
    )
    temperature_bounds_passed = _all_named("temperature_bounds[")
    physically_ready = bool(numerical_checks_passed and temperature_bounds_passed)
    full_acceptance_passed = bool(acceptance.get("passed", False))
    if not numerical_checks_passed:
        reason = "thermal benchmark numerical stability failed"
    elif not temperature_bounds_passed:
        reason = "thermal benchmark temperature bounds failed"
    elif not full_acceptance_passed:
        reason = "thermal benchmark full acceptance failed"
    else:
        reason = "thermal benchmark acceptance satisfied"
    return {
        "run_completed": True,
        "numerically_stable": numerical_checks_passed,
        "physically_ready": physically_ready,
        "ready_for_next_stage": full_acceptance_passed,
        "ready_for_long_run": full_acceptance_passed,
        "stage_status_reason": reason,
        "stage_status_checks": {
            "numerical_checks_passed": numerical_checks_passed,
            "physical_checks_passed": physically_ready,
            "full_acceptance_passed": full_acceptance_passed,
        },
    }


def main() -> None:
    args = _build_parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.runtime_budget_seconds < 0.0:
        raise ValueError("--runtime-budget-seconds must be non-negative.")
    if args.enforce_runtime_budget and args.runtime_budget_seconds <= 0.0:
        raise ValueError(
            "--enforce-runtime-budget requires a positive --runtime-budget-seconds."
        )
    if args.max_process_peak_memory_mib < 0.0:
        raise ValueError("--max-process-peak-memory-mib must be non-negative.")
    if args.max_cuda_peak_memory_mib < 0.0:
        raise ValueError("--max-cuda-peak-memory-mib must be non-negative.")
    if args.target_min_tetra_cells < 0 or args.target_max_tetra_cells < 0:
        raise ValueError("--target tetra-cell bounds must be non-negative.")
    if args.target_min_faces < 0 or args.target_max_faces < 0:
        raise ValueError("--target face bounds must be non-negative.")
    if (
        args.target_min_tetra_cells > 0
        and args.target_max_tetra_cells > 0
        and args.target_min_tetra_cells > args.target_max_tetra_cells
    ):
        raise ValueError("target tetra-cell bounds are inconsistent.")
    if (
        args.target_min_faces > 0
        and args.target_max_faces > 0
        and args.target_min_faces > args.target_max_faces
    ):
        raise ValueError("target face bounds are inconsistent.")
    if args.enforce_target_mesh and not any(
        value > 0
        for value in (
            args.target_min_tetra_cells,
            args.target_max_tetra_cells,
            args.target_min_faces,
            args.target_max_faces,
        )
    ):
        raise ValueError(
            "--enforce-target-mesh requires at least one explicit target mesh bound."
        )
    try:
        import torch  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch is required for thermal compile benchmarking. "
            "A CPU-only torch install is enough; CUDA is not required."
        ) from exc

    msh_path, _ = _resolve_msh_input(_normalize_user_path(args.msh))
    output_root = _normalize_user_path(args.output_root).resolve()
    run_dir = create_timestamped_run_dir(
        output_root, f"{msh_path.stem}_thermal_compile_benchmark"
    )
    mesh = import_gmsh_tetra_mesh(msh_path)

    velocity_field = build_prescribed_velocity_field(
        mesh,
        field_name=str(args.velocity_field),
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
    velocity_metadata = {
        "velocity_source": "prescribed_debug_field",
        "flow_solved": False,
        "pressure_solved": False,
        "field_name": velocity_field.name,
        "inlet_speed": float(args.inlet_speed),
        **velocity_field.metadata,
    }
    alpha = _resolve_thermal_diffusivity(
        thermal_diffusivity=(
            None
            if float(args.thermal_diffusivity) <= 0.0
            else float(args.thermal_diffusivity)
        ),
        rho=float(args.rho),
        cp=float(args.cp),
        thermal_conductivity=float(args.thermal_conductivity),
    )

    common_config: dict[str, object] = {
        "steps": int(args.steps),
        "dt": float(args.dt),
        "dt_mode": str(args.dt_mode),
        "cfl_target": float(args.cfl_target),
        "cfl_limit": float(args.cfl_limit),
        "diffusion_stability_factor": float(args.diffusion_stability_factor),
        "thermal_diffusivity": float(alpha),
        "rho": float(args.rho),
        "cp": float(args.cp),
        "heat_source": float(args.heat_source),
        "initial_temperature": float(args.initial_temperature),
        "left_inlet_temperature": float(args.left_inlet_temperature),
        "right_inlet_temperature": float(args.right_inlet_temperature),
        "limiter_scheme": str(args.limiter_scheme),
        "clipping_enabled": bool(not args.no_clipping),
        "min_temperature": float(args.min_temperature),
        "max_temperature": float(args.max_temperature),
        "progress_every": 0,
        "gradient_method": str(args.gradient_method),
        "laplacian_method": str(args.laplacian_method),
        "backend": str(args.backend),
        "torch_device": str(args.torch_device),
        "execution_profile": str(args.execution_profile),
        "allow_numpy_production_fallback": False,
        "diagnostics_mode": str(args.diagnostics_mode),
        "collect_history": False,
        "history_mode": str(args.history_mode),
        "history_stride": int(args.history_stride),
        "diagnostics_stride": 0,
        "full_step_diagnostics": False,
        "debug_artifacts": False,
    }

    mode_summaries: dict[str, dict[str, object]] = {}
    temperatures_by_mode: dict[str, np.ndarray] = {}
    reference_mode: str | None = None
    compile_modes = [str(mode) for mode in args.compile_modes]
    if args.require_compiled_step and not any(mode != "off" for mode in compile_modes):
        raise ValueError(
            "--require-compiled-step needs at least one compile mode other than 'off'."
        )
    if args.max_temperature_diff_from_reference >= 0.0 and len(set(compile_modes)) < 2:
        raise ValueError(
            "--max-temperature-diff-from-reference needs at least two compile modes."
        )
    if "off" in compile_modes:
        reference_mode = "off"
    elif compile_modes:
        reference_mode = compile_modes[0]

    for compile_mode in compile_modes:
        result = _run_one_benchmark_mode(
            mesh=mesh,
            face_normal_velocity=np.asarray(face_normal_velocity, dtype=np.float64),
            flux_diag=flux_diag,
            velocity_metadata=velocity_metadata,
            common_config=common_config,
            compile_mode=compile_mode,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            torch_device=str(args.torch_device),
            runtime_budget_seconds=float(args.runtime_budget_seconds),
        )
        mode_temperature = np.asarray(result.pop("temperature"), dtype=np.float64)
        temperatures_by_mode[compile_mode] = mode_temperature.copy()
        mode_summaries[compile_mode] = result

    if reference_mode is not None and reference_mode in temperatures_by_mode:
        reference_temperature = temperatures_by_mode[reference_mode]
        for compile_mode, mode_temperature in temperatures_by_mode.items():
            if compile_mode == reference_mode:
                continue
            diff = mode_temperature - reference_temperature
            mode_summaries[compile_mode]["temperature_vs_reference"] = {
                "reference_mode": str(reference_mode),
                "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
                "l2_diff": float(np.linalg.norm(diff)),
                "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
            }

    baseline_median = None
    if reference_mode is not None and reference_mode in mode_summaries:
        baseline_median = float(
            mode_summaries[reference_mode]["runtime_median_seconds"]
        )
    if baseline_median is not None and baseline_median > 0.0:
        for compile_mode, payload in mode_summaries.items():
            payload["speedup_vs_reference_median"] = float(
                baseline_median / float(payload["runtime_median_seconds"])
            )

    mesh_size = {
        "nodes": int(mesh.points.shape[0]),
        "tetra_cells": int(mesh.tetrahedra.shape[0]),
        "boundary_triangles": int(mesh.boundary_triangles.shape[0]),
        "faces": int(mesh.face_vertices.shape[0]),
    }
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries=mode_summaries,
        reference_mode=reference_mode,
        mesh_size=mesh_size,
        require_compiled_step=bool(args.require_compiled_step),
        require_cuda_core_arrays=bool(args.require_cuda_core_arrays),
        runtime_budget_seconds=float(args.runtime_budget_seconds),
        enforce_runtime_budget=bool(args.enforce_runtime_budget),
        max_process_peak_memory_mib=float(args.max_process_peak_memory_mib),
        max_cuda_peak_memory_mib=float(args.max_cuda_peak_memory_mib),
        max_temperature_diff_from_reference=float(
            args.max_temperature_diff_from_reference
        ),
        min_temperature=float(args.min_temperature),
        max_temperature=float(args.max_temperature),
        target_mesh_label=str(args.target_mesh_label),
        target_min_tetra_cells=int(args.target_min_tetra_cells),
        target_max_tetra_cells=int(args.target_max_tetra_cells),
        target_min_faces=int(args.target_min_faces),
        target_max_faces=int(args.target_max_faces),
        enforce_target_mesh=bool(args.enforce_target_mesh),
    )

    stage_status = _build_benchmark_stage_status(acceptance)
    summary = {
        "benchmark_type": "gmsh_tetra_thermal_torch_compile_step_body",
        "mesh_path": str(msh_path),
        "run_dir": str(run_dir),
        "requested_compile_modes": compile_modes,
        "reference_mode": reference_mode,
        "benchmark_settings": {
            "repeats": int(args.repeats),
            "warmup": int(args.warmup),
            "runtime_budget_seconds": float(args.runtime_budget_seconds),
            "torch_device": str(args.torch_device),
            "backend": str(args.backend),
        },
        "source_provenance": _json_ready(_collect_source_provenance()),
        "torch_environment": _json_ready(
            _collect_torch_environment(str(args.torch_device))
        ),
        "mesh_size": mesh_size,
        "target_profile": {
            "label": str(args.target_mesh_label),
            "min_tetra_cells": int(args.target_min_tetra_cells),
            "max_tetra_cells": int(args.target_max_tetra_cells),
            "min_faces": int(args.target_min_faces),
            "max_faces": int(args.target_max_faces),
            "enforce_target_mesh": bool(args.enforce_target_mesh),
        },
        "thermal_config": _json_ready(common_config),
        "flux_diagnostics": _json_ready(flux_diag),
        "velocity_metadata": _json_ready(velocity_metadata),
        "acceptance": _json_ready(acceptance),
        "stage_status": stage_status,
        "results_by_mode": _json_ready(mode_summaries),
    }
    _write_json(run_dir / "summary.json", summary)
    (run_dir / "README.txt").write_text(
        "\n".join(
            [
                "Thermal torch.compile benchmark summary",
                f"mesh: {msh_path}",
                f"reference_mode: {reference_mode}",
                f"compile_modes: {', '.join(compile_modes)}",
                f"acceptance_passed: {acceptance['passed']}",
                "Open summary.json and compare runtime_median_seconds plus",
                "performance_diagnostics.torch_compile_step_used/reason.",
                "Also compare steps_per_second, peak_memory, backend_used,",
                "mesh_size, target_profile, thermal_observables, and",
                "acceptance checks.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[gmsh-tetra-thermal-benchmark] summary written: {run_dir / 'summary.json'}")
    for compile_mode, payload in mode_summaries.items():
        perf = payload.get("performance_diagnostics", {})
        print(
            "[gmsh-tetra-thermal-benchmark] "
            f"mode={compile_mode}, "
            f"median={float(payload['runtime_median_seconds']):.6f}s, "
            f"steps_per_sec={float(payload['steps_per_second']['median']):.3f}, "
            f"used={perf.get('torch_compile_step_used')}, "
            f"reason={perf.get('torch_compile_step_reason')}, "
            f"budget_pass={payload['runtime_budget']['median_pass']}"
        )
    print(
        "[gmsh-tetra-thermal-benchmark] "
        f"acceptance_passed={acceptance['passed']}, "
        f"checks={len(acceptance['checks'])}"
    )
    if not bool(acceptance["passed"]):
        failed_checks = [
            str(check["name"])
            for check in acceptance["checks"]
            if not bool(check["passed"])
        ]
        raise SystemExit(
            "Thermal benchmark acceptance failed. "
            f"Failed checks: {', '.join(failed_checks)}. "
            f"See {run_dir / 'summary.json'} for details."
        )


if __name__ == "__main__":
    main()
