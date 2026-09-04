from __future__ import annotations

import pytest

from microfluidics.cpu_threads import (
    CPU_THREADS_ENV,
    configure_torch_cpu_threads,
    format_cpu_thread_diagnostics,
    resolve_cpu_threads,
)


class _FakeTorch:
    def __init__(self, intraop: int = 12, interop: int = 6) -> None:
        self.intraop = intraop
        self.interop = interop

    def get_num_threads(self) -> int:
        return self.intraop

    def set_num_threads(self, value: int) -> None:
        self.intraop = value

    def get_num_interop_threads(self) -> int:
        return self.interop

    def set_num_interop_threads(self, value: int) -> None:
        self.interop = value


class _InitializedInteropTorch(_FakeTorch):
    def set_num_interop_threads(self, value: int) -> None:
        _ = value
        raise RuntimeError("cannot set number of interop threads")


def test_explicit_cpu_threads_are_used() -> None:
    assert resolve_cpu_threads({CPU_THREADS_ENV: "4"}) == (
        4,
        CPU_THREADS_ENV,
    )


@pytest.mark.parametrize("value", ["0", "-2", "nope"])
def test_invalid_explicit_cpu_threads_fail_fast(value: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_cpu_threads({CPU_THREADS_ENV: value})


def test_configure_torch_applies_intraop_and_safe_interop_limit() -> None:
    torch = _FakeTorch()
    diagnostics = configure_torch_cpu_threads(
        torch,
        {
            CPU_THREADS_ENV: "4",
            "MICROFLUIDICS_TORCH_INTEROP_THREADS": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        },
    )

    assert torch.intraop == 4
    assert torch.interop == 1
    assert diagnostics["requested_cpu_threads"] == 4
    assert diagnostics["torch_intraop_threads"] == 4
    assert diagnostics["torch_interop_threads"] == 1
    assert "requested=4" in format_cpu_thread_diagnostics(diagnostics)


def test_runtime_defaults_are_observed_but_not_overridden() -> None:
    torch = _FakeTorch(intraop=10, interop=5)
    diagnostics = configure_torch_cpu_threads(torch, {})

    assert torch.intraop == 10
    assert torch.interop == 5
    assert diagnostics["requested_cpu_threads"] is None
    assert diagnostics["request_source"] == "runtime_default"


def test_initialized_interop_pool_emits_operator_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    torch = _InitializedInteropTorch(interop=6)
    with caplog.at_level("WARNING", logger="microfluidics.cpu_threads"):
        diagnostics = configure_torch_cpu_threads(
            torch,
            {
                CPU_THREADS_ENV: "4",
                "MICROFLUIDICS_TORCH_INTEROP_THREADS": "1",
            },
        )

    assert diagnostics["torch_interop_threads"] == 6
    assert diagnostics["warnings"]
    assert "inter-op thread pool was already initialized" in caplog.text
