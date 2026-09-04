"""Architecture locks for the deliberately uncoupled chemistry boundary."""

from __future__ import annotations

import ast
from pathlib import Path

CHEMISTRY_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "microfluidics" / "chemistry"
)
FORBIDDEN_IMPORT_PREFIXES = (
    "cantera",
    "celery",
    "compute",
    "scipy",
    "torch",
    "microfluidics.gmsh",
    "microfluidics.mesh",
)


def test_chemistry_has_no_solver_runtime_imports() -> None:
    violations: list[str] = []
    for path in sorted(CHEMISTRY_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{path.name}: {module}")

    assert violations == [], (
        "Standalone chemistry must not import mesh, transport, thermal, "
        "or heavyweight reactor runtimes: " + ", ".join(violations)
    )
