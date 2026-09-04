"""Pytest bootstrap for local-package imports in CI."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = REPO_ROOT / "compute" / "src"
SHARED_CONTRACTS_SRC = REPO_ROOT / "shared" / "contracts" / "src"
TESTS_ROOT = Path(__file__).resolve().parent
TEST_ZONES = {"unit", "integration", "regression", "system", "performance"}

for path in (REPO_ROOT, COMPUTE_SRC, SHARED_CONTRACTS_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests by responsibility zone and reject unclassified test modules."""

    for item in items:
        path = Path(item.path).resolve()
        try:
            relative = path.relative_to(TESTS_ROOT)
        except ValueError:
            continue
        if path.name.startswith("test_") and len(relative.parts) == 1:
            raise pytest.UsageError(
                f"Unclassified compute test at {relative.as_posix()}; "
                f"move it under one of {sorted(TEST_ZONES)}."
            )
        if not relative.parts or relative.parts[0] not in TEST_ZONES:
            continue
        zone = relative.parts[0]
        item.add_marker(getattr(pytest.mark, zone))
        if zone in {"system", "performance"}:
            item.add_marker(pytest.mark.slow)
        if path.name == "test_pipeline_mesh_regression.py":
            item.add_marker(pytest.mark.regression)
        if path.name == "test_pipeline_manifest_powershell_wrapper.py":
            item.add_marker(pytest.mark.windows)
