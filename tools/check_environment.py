"""Check the standalone MCP/CFD runtime before starting a calculation."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--mesh",
        default="data/meshes/gmsh/vertical_pipe_500.msh",
        help="Repository-relative .msh file checked for readability.",
    )
    parser.add_argument(
        "--require-gmsh",
        action="store_true",
        help="Also require the optional external Gmsh executable.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    checks: dict[str, object] = {}
    failures: list[str] = []

    checks["python"] = sys.version.split()[0]
    if sys.version_info < (3, 11):
        failures.append("Python 3.11 or newer is required.")

    for module_name in ("numpy", "scipy", "meshio", "yaml", "torch", "mcp"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            checks[module_name] = "missing"
            failures.append(f"Required Python module is missing: {module_name}.")
        else:
            checks[module_name] = str(getattr(module, "__version__", "installed"))

    torch = sys.modules.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        checks["cuda_available"] = cuda_available
        checks["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        if args.device == "cuda" and not cuda_available:
            failures.append(
                "CUDA requested but no usable CUDA device is visible. Install a "
                "CUDA-enabled PyTorch 2.5.1 build and verify the NVIDIA driver."
            )

    mesh = (PROJECT_ROOT / args.mesh).resolve()
    checks["mesh"] = args.mesh
    checks["mesh_readable"] = mesh.is_file() and mesh.is_relative_to(PROJECT_ROOT)
    if not checks["mesh_readable"]:
        failures.append("The selected mesh is missing or outside the project root.")
    elif "meshio" in sys.modules:
        try:
            mesh_data = sys.modules["meshio"].read(mesh)
        except Exception as exc:
            checks["mesh_parse"] = f"failed: {type(exc).__name__}"
            failures.append("The selected mesh could not be parsed by meshio.")
        else:
            tetra_count = sum(
                len(block.data)
                for block in mesh_data.cells
                if block.type.startswith("tetra")
            )
            checks["mesh_parse"] = "ok"
            checks["tetra_cells"] = tetra_count
            if tetra_count < 1:
                failures.append("The selected mesh contains no tetrahedral cells.")

    gmsh = shutil.which("gmsh") or shutil.which("gmsh.exe")
    checks["gmsh_executable"] = "available" if gmsh else "not found"
    if args.require_gmsh and not gmsh:
        failures.append("External Gmsh is required for mesh generation but was not found.")

    checks["status"] = "PASS" if not failures else "FAIL"
    checks["failures"] = failures
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
