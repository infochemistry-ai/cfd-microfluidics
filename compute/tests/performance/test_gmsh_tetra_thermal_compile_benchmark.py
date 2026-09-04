from __future__ import annotations

import subprocess

import experiments.gmsh.run_gmsh_tetra_thermal_compile_benchmark as benchmark_module
import pytest
from experiments.gmsh.run_gmsh_tetra_thermal_compile_benchmark import (
    _build_benchmark_stage_status,
    _collect_torch_environment,
    _evaluate_benchmark_acceptance,
    _start_process_peak_memory_sampler,
    _stop_process_peak_memory_sampler,
)


def _accepted_mode_payload(*, compiled: bool = False) -> dict[str, object]:
    return {
        "performance_diagnostics": {
            "torch_compile_step_used": bool(compiled),
            "degraded_execution_mode": False,
        },
        "backend_used": {
            "stepping_backend": "torch",
            "all_core_arrays_on_cuda": True,
        },
        "backend_execution": {"used_numpy_fallback": False},
        "runtime_budget": {"all_runs_pass": True},
        "peak_memory": {
            "process_rss": {"peak_mib_max": 128.0},
            "torch_cuda": {"peak_reserved_mib_max": 256.0},
        },
        "final_stats": {"min": 298.15, "max": 303.15, "mean": 300.0},
        "cfl_warning": False,
        "dt_control": {"diffusion_stability_warning": False},
    }


def test_compile_benchmark_torch_environment_payload_is_lightweight() -> None:
    payload = _collect_torch_environment("cuda:0")

    assert isinstance(payload, dict)
    assert payload["torch_device_requested"] == "cuda:0"
    assert "python_version" in payload
    assert "platform" in payload
    assert "torch_version" in payload
    assert "torch_cuda_available" in payload
    assert "torch_compile_available" in payload
    assert "triton_importable" in payload


def test_compile_benchmark_process_memory_sampler_reports_when_available() -> None:
    handle = _start_process_peak_memory_sampler()
    result = _stop_process_peak_memory_sampler(handle)

    assert "available" in result
    assert "method" in result
    if bool(result["available"]):
        assert int(result["peak_rss_bytes"]) > 0
        assert float(result["peak_rss_mib"]) > 0.0


def test_windows_memory_reader_initialization_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_module._get_windows_process_memory_reader.cache_clear()
    monkeypatch.setattr(benchmark_module.platform, "system", lambda: "Linux")

    assert benchmark_module._get_windows_process_memory_reader() is None
    assert benchmark_module._get_windows_process_memory_reader() is None
    cache_info = benchmark_module._get_windows_process_memory_reader.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 1
    benchmark_module._get_windows_process_memory_reader.cache_clear()


def test_source_provenance_git_timeout_is_bounded_and_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], float]] = []

    def _timed_out_run(command: list[str], **kwargs: object) -> object:
        timeout = float(kwargs["timeout"])
        calls.append((command, timeout))
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(benchmark_module.subprocess, "run", _timed_out_run)

    provenance = benchmark_module._collect_source_provenance()

    assert len(calls) == 3
    assert all(timeout == pytest.approx(10.0) for _, timeout in calls)
    assert provenance["source_commit"] is None
    assert provenance["source_branch"] is None
    assert provenance["source_tree_dirty"] is False
    assert provenance["source_tree_status_short"] == []


def test_compile_benchmark_acceptance_uses_custom_temperature_bounds() -> None:
    payload = _accepted_mode_payload()
    payload["final_stats"] = {"min": 449.0, "max": 501.0, "mean": 475.0}

    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": payload},
        reference_mode="off",
        mesh_size={"tetra_cells": 1, "faces": 1},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
        min_temperature=450.0,
        max_temperature=500.0,
    )

    temperature_check = next(
        check
        for check in acceptance["checks"]
        if check["name"] == "temperature_bounds[off]"
    )
    assert temperature_check["passed"] is False
    assert temperature_check["expected"] == {
        "min_gte": pytest.approx(450.0),
        "max_lte": pytest.approx(500.0),
    }


def test_compile_benchmark_acceptance_requires_compiled_step_when_requested() -> None:
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={
            "off": _accepted_mode_payload(compiled=False),
            "on": {
                **_accepted_mode_payload(compiled=False),
                "temperature_vs_reference": {"max_abs_diff": 0.0},
            },
        },
        reference_mode="off",
        mesh_size={"tetra_cells": 240000, "faces": 520000},
        require_compiled_step=True,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )

    assert acceptance["passed"] is False
    assert any(
        check["name"] == "compiled_step_used[on]" and check["passed"] is False
        for check in acceptance["checks"]
    )


def test_compile_benchmark_acceptance_fails_closed_on_nonfinite_temperature() -> None:
    payload = _accepted_mode_payload(compiled=False)
    payload["final_stats"] = {"min": 298.15, "max": float("nan"), "mean": 300.0}

    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": payload},
        reference_mode="off",
        mesh_size={"tetra_cells": 240000, "faces": 520000},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )

    assert acceptance["passed"] is False
    assert any(
        check["name"] == "finite_temperature_stats[off]" and check["passed"] is False
        for check in acceptance["checks"]
    )


def test_benchmark_stage_status_separates_runtime_failure() -> None:
    payload = _accepted_mode_payload()
    payload["runtime_budget"] = {"all_runs_pass": False}
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": payload},
        reference_mode="off",
        mesh_size={"tetra_cells": 1, "faces": 1},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=1.0,
        enforce_runtime_budget=True,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )
    status = _build_benchmark_stage_status(acceptance)
    assert status["numerically_stable"] is True
    assert status["physically_ready"] is True
    assert status["ready_for_next_stage"] is False
    assert status["ready_for_long_run"] is False


def test_benchmark_stage_status_handles_nonfinite_and_out_of_bounds_temperature() -> (
    None
):
    nonfinite = _accepted_mode_payload()
    nonfinite["final_stats"] = {"min": 298.15, "max": float("nan"), "mean": 300.0}
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": nonfinite},
        reference_mode="off",
        mesh_size={"tetra_cells": 1, "faces": 1},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )
    status = _build_benchmark_stage_status(acceptance)
    assert status["numerically_stable"] is False
    assert status["physically_ready"] is False

    out_of_bounds = _accepted_mode_payload()
    out_of_bounds["final_stats"] = {"min": 280.0, "max": 303.15, "mean": 300.0}
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": out_of_bounds},
        reference_mode="off",
        mesh_size={"tetra_cells": 1, "faces": 1},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )
    status = _build_benchmark_stage_status(acceptance)
    assert status["numerically_stable"] is True
    assert status["physically_ready"] is False
    assert status["ready_for_next_stage"] is False


def test_benchmark_stage_status_reports_full_success() -> None:
    acceptance = _evaluate_benchmark_acceptance(
        mode_summaries={"off": _accepted_mode_payload()},
        reference_mode="off",
        mesh_size={"tetra_cells": 1, "faces": 1},
        require_compiled_step=False,
        require_cuda_core_arrays=False,
        runtime_budget_seconds=0.0,
        enforce_runtime_budget=False,
        max_process_peak_memory_mib=0.0,
        max_cuda_peak_memory_mib=0.0,
        max_temperature_diff_from_reference=-1.0,
        target_mesh_label="",
        target_min_tetra_cells=0,
        target_max_tetra_cells=0,
        target_min_faces=0,
        target_max_faces=0,
        enforce_target_mesh=False,
    )
    status = _build_benchmark_stage_status(acceptance)
    assert all(
        status[key] is True
        for key in (
            "numerically_stable",
            "physically_ready",
            "ready_for_next_stage",
            "ready_for_long_run",
        )
    )
