"""Backend selection helpers for Gmsh tetra debug operators and scalar solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from microfluidics.cpu_threads import (
    configure_torch_cpu_threads,
    cpu_thread_diagnostics,
    format_cpu_thread_diagnostics,
)

BackendName = Literal["numpy", "torch"]
BackendRequest = Literal["auto", "numpy", "torch"]


@dataclass(frozen=True)
class BackendSelection:
    requested_backend: BackendRequest
    selected_backend: BackendName
    device: str
    torch_available: bool
    torch_version: str | None
    torch_cuda_available: bool
    torch_device_count: int
    torch_gpu_name: str | None
    used_numpy_fallback: bool
    notes: tuple[str, ...]
    cpu_threads: dict[str, object]


def select_backend(
    requested: BackendRequest = "auto",
    *,
    torch_available_override: bool | None = None,
    torch_cuda_available_override: bool | None = None,
    torch_version_override: str | None = None,
    torch_device_count_override: int | None = None,
    torch_gpu_name_override: str | None = None,
) -> BackendSelection:
    """Resolve backend request into an executable backend/device with diagnostics."""

    notes: list[str] = []
    torch_available = False
    torch_version: str | None = None
    torch_cuda_available = False
    torch_device_count = 0
    torch_gpu_name: str | None = None
    torch_module = None

    if torch_available_override is not None:
        torch_available = bool(torch_available_override)
        torch_version = torch_version_override
        torch_cuda_available = bool(torch_cuda_available_override)
        torch_device_count = int(torch_device_count_override or 0)
        torch_gpu_name = torch_gpu_name_override
    else:
        try:
            import torch  # type: ignore
        except ModuleNotFoundError:
            torch_available = False
            notes.append("torch is not installed; falling back to numpy.")
        else:
            torch_module = torch
            torch_available = True
            torch_version = str(torch.__version__)
            torch_cuda_available = bool(torch.cuda.is_available())
            torch_device_count = (
                int(torch.cuda.device_count()) if torch_cuda_available else 0
            )
            if torch_cuda_available and torch_device_count > 0:
                torch_gpu_name = str(torch.cuda.get_device_name(0))

    if torch_module is not None:
        thread_diagnostics = configure_torch_cpu_threads(torch_module)
    else:
        thread_diagnostics = cpu_thread_diagnostics()
    notes.append(
        "CPU thread runtime: " + format_cpu_thread_diagnostics(thread_diagnostics)
    )
    notes.extend(str(item) for item in thread_diagnostics["warnings"])

    requested = str(requested).lower().strip()  # type: ignore[assignment]
    if requested not in {"auto", "numpy", "torch"}:
        raise ValueError(f"Unsupported backend request: {requested!r}")

    selected_backend: BackendName
    device: str
    used_numpy_fallback = False

    if requested == "numpy":
        selected_backend = "numpy"
        device = "cpu"
        if torch_available:
            notes.append("numpy backend requested explicitly.")
    elif requested == "torch":
        if torch_available:
            selected_backend = "torch"
            if torch_cuda_available and torch_device_count > 0:
                device = "cuda:0"
            else:
                device = "cpu"
                notes.append(
                    "torch backend selected, CUDA unavailable -> using torch CPU."
                )
        else:
            selected_backend = "numpy"
            device = "cpu"
            used_numpy_fallback = True
            notes.append(
                "torch backend requested but torch is unavailable; using numpy fallback."
            )
    else:
        if torch_available:
            selected_backend = "torch"
            if torch_cuda_available and torch_device_count > 0:
                device = "cuda:0"
            else:
                device = "cpu"
                notes.append("auto backend selected torch CPU (CUDA unavailable).")
        else:
            selected_backend = "numpy"
            device = "cpu"
            used_numpy_fallback = True
            notes.append("auto backend selected numpy because torch is unavailable.")

    return BackendSelection(
        requested_backend=requested,  # type: ignore[arg-type]
        selected_backend=selected_backend,
        device=device,
        torch_available=torch_available,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_device_count=torch_device_count,
        torch_gpu_name=torch_gpu_name,
        used_numpy_fallback=used_numpy_fallback,
        notes=tuple(notes),
        cpu_threads=thread_diagnostics,
    )
