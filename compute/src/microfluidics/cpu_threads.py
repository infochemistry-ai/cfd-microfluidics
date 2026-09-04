"""CPU thread configuration and diagnostics for numerical backends.

Native BLAS/OpenMP runtimes read their environment variables when the process
starts. This module additionally applies an explicit local limit to PyTorch
after it is imported and provides JSON-serializable diagnostics for run
artifacts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any


logger = logging.getLogger(__name__)

CPU_THREADS_ENV = "MICROFLUIDICS_CPU_THREADS"
TORCH_INTEROP_THREADS_ENV = "MICROFLUIDICS_TORCH_INTEROP_THREADS"
NATIVE_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def resolve_cpu_threads(
    environ: Mapping[str, str] | None = None,
) -> tuple[int | None, str]:
    """Resolve the explicit CPU limit without guessing from host CPU count."""

    env = os.environ if environ is None else environ
    raw = env.get(CPU_THREADS_ENV)
    if raw is not None and raw.strip():
        return _positive_int(raw.strip(), CPU_THREADS_ENV), CPU_THREADS_ENV
    return None, "runtime_default"


def _affinity_cpu_count() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is None:
        return None
    try:
        return len(get_affinity(0))
    except OSError:
        return None


def configure_torch_cpu_threads(
    torch_module: Any,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Apply an explicit CPU limit to PyTorch and return runtime diagnostics."""

    env = os.environ if environ is None else environ
    requested, source = resolve_cpu_threads(env)
    warnings: list[str] = []

    if requested is not None:
        torch_module.set_num_threads(requested)

        interop_raw = env.get(TORCH_INTEROP_THREADS_ENV, "1").strip() or "1"
        interop_requested = _positive_int(
            interop_raw,
            TORCH_INTEROP_THREADS_ENV,
        )
        current_interop = int(torch_module.get_num_interop_threads())
        if current_interop != interop_requested:
            try:
                torch_module.set_num_interop_threads(interop_requested)
            except RuntimeError as exc:
                warning = (
                    "PyTorch inter-op thread pool was already initialized; "
                    f"keeping {current_interop} threads ({exc})."
                )
                warnings.append(warning)
                logger.warning(warning)

    return cpu_thread_diagnostics(
        torch_module=torch_module,
        environ=env,
        warnings=warnings,
    )


def cpu_thread_diagnostics(
    *,
    torch_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    """Capture requested and effective numerical thread settings."""

    env = os.environ if environ is None else environ
    requested, source = resolve_cpu_threads(env)
    native_threads = {
        name: env.get(name, "").strip() for name in NATIVE_THREAD_ENV_NAMES
    }
    return {
        "requested_cpu_threads": requested,
        "request_source": source,
        "logical_cpu_count": os.cpu_count(),
        "affinity_cpu_count": _affinity_cpu_count(),
        "native_thread_env": native_threads,
        "torch_intraop_threads": (
            int(torch_module.get_num_threads()) if torch_module is not None else None
        ),
        "torch_interop_threads": (
            int(torch_module.get_num_interop_threads())
            if torch_module is not None
            else None
        ),
        "warnings": list(warnings or []),
    }


def format_cpu_thread_diagnostics(diagnostics: Mapping[str, object]) -> str:
    """Render a compact single-line diagnostic suitable for stage logs."""

    native = diagnostics.get("native_thread_env")
    native_text = ""
    if isinstance(native, Mapping):
        native_text = ",".join(
            f"{name}={value or 'unset'}" for name, value in native.items()
        )
    return (
        f"requested={diagnostics.get('requested_cpu_threads')} "
        f"source={diagnostics.get('request_source')} "
        f"affinity={diagnostics.get('affinity_cpu_count')} "
        f"torch_intraop={diagnostics.get('torch_intraop_threads')} "
        f"torch_interop={diagnostics.get('torch_interop_threads')} "
        f"native=[{native_text}]"
    )
