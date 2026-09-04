"""Pytest bootstrap for the MCP package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
COMPUTE_SRC = REPO_ROOT / "compute" / "src"
SHARED_CONTRACTS_SRC = REPO_ROOT / "shared" / "contracts" / "src"
MCP_SRC = REPO_ROOT / "mcp-server" / "src"

for path in (REPO_ROOT, BACKEND_ROOT, COMPUTE_SRC, SHARED_CONTRACTS_SRC, MCP_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture()
def symlinks(tmp_path: Path) -> None:
    """Skip when this account cannot create symlinks in `tmp_path`.

    Creating one on Windows needs SeCreateSymbolicLinkPrivilege - Developer
    Mode, or an elevated shell - and `Path.symlink_to` raises
    `OSError(WinError 1314)` without it. A test that asserts what the code
    does with a symlink has nothing left to assert when it cannot make one, so
    it skips with that reason instead of failing as if the code were wrong.

    The probe runs in the test's own `tmp_path`, so it answers for the
    directory the test will actually use. Its target deliberately does not
    exist: a dangling file symlink is the cheapest thing to create and the
    only thing `unlink` has to undo on either platform.

    Duplicated in `backend/tests/conftest.py`; the two suites install as
    separate packages and share no test-support module.
    """

    probe = tmp_path / "_symlink_probe"
    try:
        probe.symlink_to(tmp_path / "_symlink_probe_target")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable here: {error}")
    probe.unlink()
