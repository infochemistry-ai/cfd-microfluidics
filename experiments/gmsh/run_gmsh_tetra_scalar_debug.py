"""Special-purpose tetra-mesh operator diagnostics and diffusion-only scalar tool.

This script is a research/debug utility and is not a production orchestration path.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPUTE_SRC = PROJECT_ROOT / "compute" / "src"
for path in (PROJECT_ROOT, COMPUTE_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from experiments.gmsh._path_utils import _normalize_user_path  # noqa: E402
from microfluidics.path_contract import (  # noqa: E402
    GMSH_IMPORT_RUNS_ROOT_REL,
    GMSH_TETRA_SCALAR_RUNS_ROOT_REL,
    create_timestamped_run_dir,
    resolve_repo_path,
)


class _TeeWriter:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = _TeeWriter(sys.__stdout__, log_file)
        tee_err = _TeeWriter(sys.__stderr__, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            yield


def _json_ready(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _save_scalar_previews(
    *,
    centers: np.ndarray,
    scalar: np.ndarray,
    run_dir: Path,
    file_prefix: str,
) -> Dict[str, str]:
    previews: Dict[str, str] = {}

    fig_xy, ax_xy = plt.subplots(figsize=(7.4, 6.0))
    sc_xy = ax_xy.scatter(
        centers[:, 0],
        centers[:, 1],
        c=scalar,
        s=2.0,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax_xy.set_title("Scalar Concentration on Cell Centers (XY)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_aspect("equal", adjustable="box")
    fig_xy.colorbar(sc_xy, ax=ax_xy, label="C")
    fig_xy.tight_layout()
    xy_path = run_dir / f"{file_prefix}_scalar_xy.png"
    fig_xy.savefig(xy_path, dpi=180)
    plt.close(fig_xy)
    previews["scalar_xy"] = str(xy_path)

    z = centers[:, 2]
    z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
    z_span = float(np.max(z) - np.min(z))
    tol = max(z_span * 0.02, 1e-6)
    slice_mask = np.abs(z - z_mid) <= tol
    if not np.any(slice_mask):
        idx = np.argsort(np.abs(z - z_mid))[: max(100, centers.shape[0] // 50)]
        slice_mask = np.zeros(centers.shape[0], dtype=bool)
        slice_mask[idx] = True

    fig_mid, ax_mid = plt.subplots(figsize=(7.4, 6.0))
    sc_mid = ax_mid.scatter(
        centers[slice_mask, 0],
        centers[slice_mask, 1],
        c=scalar[slice_mask],
        s=5.0,
        cmap="viridis",
        alpha=0.9,
        linewidths=0,
        vmin=0.0,
        vmax=1.0,
    )
    ax_mid.set_title("Scalar Concentration near Mid-Z Slice (XY)")
    ax_mid.set_xlabel("x [m]")
    ax_mid.set_ylabel("y [m]")
    ax_mid.set_aspect("equal", adjustable="box")
    fig_mid.colorbar(sc_mid, ax=ax_mid, label="C")
    fig_mid.tight_layout()
    mid_path = run_dir / f"{file_prefix}_scalar_mid_z_xy.png"
    fig_mid.savefig(mid_path, dpi=180)
    plt.close(fig_mid)
    previews["scalar_mid_z_xy"] = str(mid_path)
    return previews


def _export_scalar_vtu(
    *,
    points: np.ndarray,
    tetrahedra: np.ndarray,
    scalar: np.ndarray,
    gradient_norm: np.ndarray,
    output_path: Path,
) -> Path:
    try:
        import meshio  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "meshio is required to export scalar VTU. Install with `pip install meshio`."
        ) from exc

    mesh = meshio.Mesh(
        points=points,
        cells=[("tetra", tetrahedra.astype(np.int64, copy=False))],
        cell_data={
            "concentration": [scalar.astype(np.float64, copy=False)],
            "gradient_magnitude": [gradient_norm.astype(np.float64, copy=False)],
        },
    )
    meshio.write(output_path, mesh)
    return output_path


def main() -> None:
    from microfluidics.gmsh.gmsh_mesh_validation import (
        format_validation_report,
        validate_imported_tetra_mesh,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_backend import select_backend
    from microfluidics.gmsh.tetra.gmsh_tetra_mesh_loader import (
        load_imported_tetra_mesh_npz,
    )
    from microfluidics.gmsh.tetra.gmsh_tetra_operators import run_operator_diagnostics
    from microfluidics.gmsh.tetra.gmsh_tetra_scalar_solver import (
        GmshTetraScalarConfig,
        run_scalar_diffusion_debug,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-npz",
        type=str,
        default="",
        help="Explicit path to *_imported_mesh.npz.",
    )
    parser.add_argument(
        "--import-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_IMPORT_RUNS_ROOT_REL)),
        help="Retained for compatibility and metadata. Mesh resolution requires --mesh-npz.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(resolve_repo_path(PROJECT_ROOT, GMSH_TETRA_SCALAR_RUNS_ROOT_REL)),
        help="Per-run output root for scalar debug artifacts.",
    )
    parser.add_argument(
        "--steps", type=int, default=120, help="Diffusion explicit step count."
    )
    parser.add_argument("--dt", type=float, default=5e-4, help="Diffusion time step.")
    parser.add_argument(
        "--diffusivity", type=float, default=3e-10, help="Diffusion coefficient."
    )
    parser.add_argument(
        "--gradient-method",
        type=str,
        choices=("face", "least_squares"),
        default="least_squares",
        help="Gradient reconstruction method for diagnostics/final output.",
    )
    parser.add_argument(
        "--laplacian-method",
        type=str,
        choices=("tpfa", "lsq_flux"),
        default="lsq_flux",
        help="Laplacian discretization for diffusion step.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "numpy", "torch"),
        default="auto",
        help="Compute backend for diffusion stepping.",
    )
    parser.add_argument(
        "--left-inlet-value",
        type=float,
        default=0.0,
        help="Dirichlet concentration for left inlet.",
    )
    parser.add_argument(
        "--right-inlet-value",
        type=float,
        default=1.0,
        help="Dirichlet concentration for right inlet.",
    )
    args = parser.parse_args()

    import_root = _normalize_user_path(args.import_root).resolve()
    output_root = _normalize_user_path(args.output_root).resolve()
    mesh_npz_input = (
        _normalize_user_path(args.mesh_npz).resolve()
        if str(args.mesh_npz).strip()
        else None
    )
    if mesh_npz_input is None:
        parser.error(
            "--mesh-npz is required. Automatic import-root resolution is no longer supported."
        )
    mesh_npz = mesh_npz_input
    if not mesh_npz.exists():
        raise FileNotFoundError(f"Mesh npz not found: {mesh_npz}")

    run_dir = create_timestamped_run_dir(output_root, mesh_npz.stem)
    with _tee_logging(run_dir / "run.log"):
        backend = select_backend(args.backend)
        print(f"[gmsh-tetra-scalar] run directory: {run_dir}")
        print(f"[gmsh-tetra-scalar] mesh npz: {mesh_npz}")
        print(
            "[gmsh-tetra-scalar] backend: "
            f"requested={backend.requested_backend}, selected={backend.selected_backend}, "
            f"device={backend.device}"
        )
        print(
            "[gmsh-tetra-scalar] torch: "
            f"available={backend.torch_available}, version={backend.torch_version}, "
            f"cuda_available={backend.torch_cuda_available}, "
            f"device_count={backend.torch_device_count}, gpu={backend.torch_gpu_name}"
        )
        for note in backend.notes:
            print(f"[gmsh-tetra-scalar] note: {note}")

        config_payload = {
            "mesh_npz": str(mesh_npz),
            "import_root": str(import_root),
            "output_root": str(output_root),
            "steps": int(args.steps),
            "dt": float(args.dt),
            "diffusivity": float(args.diffusivity),
            "left_inlet_value": float(args.left_inlet_value),
            "right_inlet_value": float(args.right_inlet_value),
            "gradient_method": str(args.gradient_method),
            "laplacian_method": str(args.laplacian_method),
            "backend": backend.__dict__,
        }
        _write_json(run_dir / "config.json", config_payload)

        mesh = load_imported_tetra_mesh_npz(mesh_npz)
        validation = validate_imported_tetra_mesh(mesh)
        (run_dir / "validation_report.txt").write_text(
            format_validation_report(validation),
            encoding="utf-8",
        )
        if not validation.is_valid:
            raise RuntimeError(
                "Loaded imported mesh failed validation. See validation_report.txt."
            )

        operator_diag = run_operator_diagnostics(
            mesh,
            gradient_method=args.gradient_method,  # type: ignore[arg-type]
        )
        _write_json(run_dir / "operator_diagnostics.json", operator_diag)
        _write_json(
            run_dir / "operator_diagnostics_legacy.json",
            {
                "gradient_face": operator_diag["gradient_face"],
                "laplacian_tpfa": operator_diag["laplacian_tpfa"],
            },
        )

        scalar_cfg = GmshTetraScalarConfig(
            dt=float(args.dt),
            steps=int(args.steps),
            diffusivity=float(args.diffusivity),
            left_inlet_value=float(args.left_inlet_value),
            right_inlet_value=float(args.right_inlet_value),
            gradient_method=args.gradient_method,  # type: ignore[arg-type]
            laplacian_method=args.laplacian_method,  # type: ignore[arg-type]
            backend=backend.selected_backend,
            torch_device=backend.device,
        )
        scalar_result = run_scalar_diffusion_debug(mesh, scalar_cfg)
        scalar = np.asarray(scalar_result["scalar"], dtype=np.float64)
        gradient_norm = np.asarray(scalar_result["gradient_norm"], dtype=np.float64)
        scalar_history = scalar_result["history"]
        _write_json(run_dir / "scalar_history.json", {"history": scalar_history})

        vtu_path = _export_scalar_vtu(
            points=mesh.points,
            tetrahedra=mesh.tetrahedra,
            scalar=scalar,
            gradient_norm=gradient_norm,
            output_path=run_dir / f"{mesh_npz.stem}_scalar_result.vtu",
        )
        previews = _save_scalar_previews(
            centers=mesh.cell_centers,
            scalar=scalar,
            run_dir=run_dir,
            file_prefix=mesh_npz.stem,
        )

        summary = {
            "mesh_npz": str(mesh_npz),
            "backend": backend.__dict__,
            "validation": validation.summary,
            "operator_diagnostics": operator_diag,
            "scalar_final_stats": scalar_result["final_stats"],
            "scalar_solver": {
                "stable_dt_estimate": scalar_result["stable_dt_estimate"],
                "stability_ratio_dt_to_limit": scalar_result[
                    "stability_ratio_dt_to_limit"
                ],
                "stability_warning": scalar_result["stability_warning"],
                "boundary_setup": scalar_result["boundary_setup"],
                "methods": scalar_result["methods"],
                "clipping": scalar_result["clipping"],
                "backend_execution": scalar_result["backend_execution"],
                "history_length": len(scalar_history),
            },
            "artifacts": {
                "run_log": str(run_dir / "run.log"),
                "config_json": str(run_dir / "config.json"),
                "summary_json": str(run_dir / "summary.json"),
                "operator_diagnostics_json": str(run_dir / "operator_diagnostics.json"),
                "operator_diagnostics_legacy_json": str(
                    run_dir / "operator_diagnostics_legacy.json"
                ),
                "scalar_history_json": str(run_dir / "scalar_history.json"),
                "validation_report_txt": str(run_dir / "validation_report.txt"),
                "scalar_vtu": str(vtu_path),
                "preview_png": previews,
            },
        }
        _write_json(run_dir / "summary.json", summary)

        selected = operator_diag["gradient_selected"]
        print("[gmsh-tetra-scalar] operator diagnostics (selected gradient):")
        print(
            f"  - linear_gradient_mean_abs_error: {selected['linear_gradient_mean_abs_error']}"
        )
        print(
            f"  - linear_gradient_max_abs_error: {selected['linear_gradient_max_abs_error']}"
        )
        print(
            "[gmsh-tetra-scalar] final scalar stats: "
            f"min={summary['scalar_final_stats']['min']:.6f}, "
            f"max={summary['scalar_final_stats']['max']:.6f}, "
            f"mean={summary['scalar_final_stats']['mean']:.6f}"
        )
        print(f"[gmsh-tetra-scalar] summary written: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
