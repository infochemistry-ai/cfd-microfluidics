"""Visualization utilities for mesh-aware flow+transport baseline runs."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _active_bbox(mask_2d: np.ndarray, pad_cells: int = 1) -> tuple[slice, slice]:
    idx0 = np.where(np.any(mask_2d, axis=1))[0]
    idx1 = np.where(np.any(mask_2d, axis=0))[0]
    n0, n1 = mask_2d.shape
    if idx0.size == 0 or idx1.size == 0:
        return slice(0, n0), slice(0, n1)
    i0 = max(0, int(idx0[0]) - pad_cells)
    i1 = min(n0, int(idx0[-1]) + pad_cells + 1)
    j0 = max(0, int(idx1[0]) - pad_cells)
    j1 = min(n1, int(idx1[-1]) + pad_cells + 1)
    return slice(i0, i1), slice(j0, j1)


def _axis_centers(lower: float, upper: float, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("Axis resolution must be positive.")
    if count == 1:
        return np.array([0.5 * (lower + upper)], dtype=float)
    step = (upper - lower) / float(count)
    return lower + (np.arange(count, dtype=float) + 0.5) * step


def _extent_mm(
    axis_centers: np.ndarray, idx_start: int, idx_stop: int
) -> tuple[float, float]:
    if idx_stop <= idx_start:
        idx_start = 0
        idx_stop = len(axis_centers)
    if len(axis_centers) > 1:
        spacing = float(axis_centers[1] - axis_centers[0])
    else:
        spacing = 1.0
    lower = float(axis_centers[idx_start] - 0.5 * spacing) * 1e3
    upper = float(axis_centers[idx_stop - 1] + 0.5 * spacing) * 1e3
    return lower, upper


def _grid_axes_mm(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = grid.bounds.lower.vector
    upper = grid.bounds.upper.vector
    nx = grid.shape.get_size("x")
    ny = grid.shape.get_size("y")
    nz = grid.shape.get_size("z")
    x = _axis_centers(float(lower[0].numpy()), float(upper[0].numpy()), nx)
    y = _axis_centers(float(lower[1].numpy()), float(upper[1].numpy()), ny)
    z = _axis_centers(float(lower[2].numpy()), float(upper[2].numpy()), nz)
    return x, y, z


def save_baseline_slice_pngs(
    output_dir: Path,
    case_tag: str,
    scalar,
    velocity,
    channel_mask,
    y_positions: Tuple[float, ...],
) -> Dict[str, str]:
    """Save concentration and velocity slices in y-sections and z-midplane."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_np = scalar.values.numpy("x,y,z")
    speed_np = np.linalg.norm(
        velocity.at_centers().values.numpy("x,y,z,vector"), axis=-1
    )
    mask_np = channel_mask.values.numpy("x,y,z") > 0.5
    x_axis, y_axis, z_axis = _grid_axes_mm(scalar)
    y_min = float(scalar.bounds.lower.vector[1].numpy())
    y_max = float(scalar.bounds.upper.vector[1].numpy())
    ny = scalar.shape.get_size("y")
    saved: Dict[str, str] = {}

    for y_pos in y_positions:
        if ny <= 1 or y_max <= y_min:
            y_idx = 0
        else:
            ratio = (y_pos - y_min) / (y_max - y_min)
            y_idx = int(np.clip(round(ratio * (ny - 1)), 0, ny - 1))
        section_mask = mask_np[:, y_idx, :]
        x_slice, z_slice = _active_bbox(section_mask, pad_cells=1)
        scalar_vis = np.ma.masked_where(
            ~section_mask[x_slice, z_slice].T,
            scalar_np[:, y_idx, :][x_slice, z_slice].T,
        )
        speed_vis = np.ma.masked_where(
            ~section_mask[x_slice, z_slice].T,
            speed_np[:, y_idx, :][x_slice, z_slice].T,
        )
        x_mm = _extent_mm(x_axis, x_slice.start, x_slice.stop)
        z_mm = _extent_mm(z_axis, z_slice.start, z_slice.stop)
        extent = (x_mm[0], x_mm[1], z_mm[0], z_mm[1])
        y_label = f"{y_pos * 1e3:.3f}".replace("-", "m").replace(".", "p")

        conc_path = output_dir / f"{case_tag}_concentration_y_{y_label}mm.png"
        vel_path = output_dir / f"{case_tag}_velocity_magnitude_y_{y_label}mm.png"

        fig_c, ax_c = plt.subplots(figsize=(7.2, 3.8))
        cmap_c = plt.get_cmap("viridis").copy()
        cmap_c.set_bad(color="black")
        im_c = ax_c.imshow(
            scalar_vis,
            origin="lower",
            cmap=cmap_c,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
        )
        ax_c.set_title(f"{case_tag} concentration y={y_pos * 1e3:.3f} mm")
        ax_c.set_xlabel("x [mm]")
        ax_c.set_ylabel("z [mm]")
        fig_c.colorbar(im_c, ax=ax_c, label="C")
        fig_c.tight_layout()
        fig_c.savefig(conc_path, dpi=180)
        plt.close(fig_c)

        fig_v, ax_v = plt.subplots(figsize=(7.2, 3.8))
        cmap_v = plt.get_cmap("magma").copy()
        cmap_v.set_bad(color="black")
        vmax_speed = (
            float(np.max(speed_vis.filled(0.0))) if speed_vis.count() > 0 else 1.0
        )
        im_v = ax_v.imshow(
            speed_vis,
            origin="lower",
            cmap=cmap_v,
            aspect="auto",
            vmin=0.0,
            vmax=max(vmax_speed, 1e-12),
            extent=extent,
        )
        ax_v.set_title(f"{case_tag} velocity magnitude y={y_pos * 1e3:.3f} mm")
        ax_v.set_xlabel("x [mm]")
        ax_v.set_ylabel("z [mm]")
        fig_v.colorbar(im_v, ax=ax_v, label="|u| [m/s]")
        fig_v.tight_layout()
        fig_v.savefig(vel_path, dpi=180)
        plt.close(fig_v)

        saved[f"concentration_y_{y_label}"] = str(conc_path)
        saved[f"velocity_y_{y_label}"] = str(vel_path)

    z_mid = scalar_np.shape[2] // 2
    xy_mask = mask_np[:, :, z_mid]
    x_slice, y_slice = _active_bbox(xy_mask, pad_cells=1)
    scalar_xy = np.ma.masked_where(
        ~xy_mask[x_slice, y_slice].T,
        scalar_np[:, :, z_mid][x_slice, y_slice].T,
    )
    speed_xy = np.ma.masked_where(
        ~xy_mask[x_slice, y_slice].T,
        speed_np[:, :, z_mid][x_slice, y_slice].T,
    )
    x_mm = _extent_mm(x_axis, x_slice.start, x_slice.stop)
    y_mm = _extent_mm(y_axis, y_slice.start, y_slice.stop)
    extent = (x_mm[0], x_mm[1], y_mm[0], y_mm[1])

    conc_mid = output_dir / f"{case_tag}_concentration_z_midplane.png"
    vel_mid = output_dir / f"{case_tag}_velocity_z_midplane.png"

    fig_mc, ax_mc = plt.subplots(figsize=(7.2, 4.2))
    cmap_mc = plt.get_cmap("viridis").copy()
    cmap_mc.set_bad(color="black")
    im_mc = ax_mc.imshow(
        scalar_xy,
        origin="lower",
        cmap=cmap_mc,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    ax_mc.set_title(f"{case_tag} concentration z=midplane")
    ax_mc.set_xlabel("x [mm]")
    ax_mc.set_ylabel("y [mm]")
    fig_mc.colorbar(im_mc, ax=ax_mc, label="C")
    fig_mc.tight_layout()
    fig_mc.savefig(conc_mid, dpi=180)
    plt.close(fig_mc)

    fig_mv, ax_mv = plt.subplots(figsize=(7.2, 4.2))
    cmap_mv = plt.get_cmap("magma").copy()
    cmap_mv.set_bad(color="black")
    vmax_mid = float(np.max(speed_xy.filled(0.0))) if speed_xy.count() > 0 else 1.0
    im_mv = ax_mv.imshow(
        speed_xy,
        origin="lower",
        cmap=cmap_mv,
        aspect="auto",
        vmin=0.0,
        vmax=max(vmax_mid, 1e-12),
        extent=extent,
    )
    ax_mv.set_title(f"{case_tag} velocity magnitude z=midplane")
    ax_mv.set_xlabel("x [mm]")
    ax_mv.set_ylabel("y [mm]")
    fig_mv.colorbar(im_mv, ax=ax_mv, label="|u| [m/s]")
    fig_mv.tight_layout()
    fig_mv.savefig(vel_mid, dpi=180)
    plt.close(fig_mv)

    saved["concentration_z_midplane"] = str(conc_mid)
    saved["velocity_z_midplane"] = str(vel_mid)
    return saved


def save_mesh_cell_scalar_preview(
    output_dir: Path,
    case_tag: str,
    mesh,
    scalar_values: np.ndarray,
    *,
    field_name: str = "concentration_native",
) -> str | None:
    """Save x-y preview for mesh cell-centered scalar values.

    Rendering mode:
    1) Try structured cell-shaded midplane image via `cell_indices`.
    2) Fallback to cell-center scatter if structured mapping is not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    centers = np.asarray(mesh.cell_centers, dtype=np.float64)
    values = np.asarray(scalar_values, dtype=np.float64).reshape(-1)
    if centers.shape[0] != values.shape[0]:
        raise ValueError("scalar_values size must match number of mesh cells.")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{case_tag}_{field_name}_mesh_xy_midplane.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.2))

    rendered = False
    if hasattr(mesh, "cell_indices") and "resolution" in mesh.metadata:
        ijk = np.asarray(mesh.cell_indices, dtype=np.int64)
        resolution = tuple(int(v) for v in mesh.metadata["resolution"])
        if ijk.ndim == 2 and ijk.shape[1] == 3 and len(resolution) == 3:
            nx, ny, nz = resolution
            if nx > 0 and ny > 0 and nz > 0:
                k_mid = nz // 2
                mid_mask = ijk[:, 2] == k_mid
                if np.any(mid_mask):
                    plane = np.full((nx, ny), np.nan, dtype=np.float64)
                    ii = ijk[mid_mask, 0]
                    jj = ijk[mid_mask, 1]
                    valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
                    plane[ii[valid], jj[valid]] = values[mid_mask][valid]

                    x = centers[:, 0]
                    y = centers[:, 1]
                    x_min, x_max = float(np.min(x)), float(np.max(x))
                    y_min, y_max = float(np.min(y)), float(np.max(y))
                    if nx > 1:
                        dx = (x_max - x_min) / float(nx - 1)
                    else:
                        dx = float(mesh.metadata.get("dx", 1.0))
                    if ny > 1:
                        dy = (y_max - y_min) / float(ny - 1)
                    else:
                        dy = float(mesh.metadata.get("dy", 1.0))
                    extent = (
                        (x_min - 0.5 * dx) * 1e3,
                        (x_max + 0.5 * dx) * 1e3,
                        (y_min - 0.5 * dy) * 1e3,
                        (y_max + 0.5 * dy) * 1e3,
                    )

                    vis = np.ma.masked_invalid(plane.T)
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad(color="black")
                    im = ax.imshow(
                        vis,
                        origin="lower",
                        cmap=cmap,
                        aspect="auto",
                        extent=extent,
                    )
                    ax.set_title(
                        f"{case_tag} {field_name} mesh preview "
                        "(cell-centered shaded slice, x-y @ z midplane)"
                    )
                    ax.set_xlabel("x [mm]")
                    ax.set_ylabel("y [mm]")
                    fig.colorbar(im, ax=ax, label=field_name)
                    rendered = True

    if not rendered:
        z = centers[:, 2]
        z_mid = 0.5 * (float(np.min(z)) + float(np.max(z)))
        dz = float(mesh.metadata.get("dz", 0.0))
        tol = max(0.5 * dz, 1e-12)
        mid_mask = np.abs(z - z_mid) <= tol
        if not np.any(mid_mask):
            mid_mask = np.ones(centers.shape[0], dtype=bool)
        sc = ax.scatter(
            centers[mid_mask, 0] * 1e3,
            centers[mid_mask, 1] * 1e3,
            c=values[mid_mask],
            s=4,
            cmap="viridis",
            alpha=0.9,
        )
        ax.set_title(
            f"{case_tag} {field_name} mesh preview (cell-center scatter, x-y @ z midplane)"
        )
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(sc, ax=ax, label=field_name)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)
